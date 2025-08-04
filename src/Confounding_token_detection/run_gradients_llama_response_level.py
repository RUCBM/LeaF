import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm

def load_model_and_tokenizer(model_name: str):
    """
    Load a causal language model and its tokenizer in training mode.

    Args:
        model_name (str): The path or identifier of the pretrained model.

    Returns:
        model (AutoModelForCausalLM): The loaded language model.
        tokenizer (AutoTokenizer): The corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.train()
    return model, tokenizer


def get_tokenized_data(tokenizer, data_point: dict):
    """
    Prepare tokenized input and label tensors for gradient calculation.

    Args:
        tokenizer (AutoTokenizer): Model tokenizer.
        data_point (dict): Contains 'instruction' and 'output' strings.

    Returns:
        tokenized_full (dict): Tokenized prompt with labels for loss.
        user_prompt_length (int): Number of tokens in the user instruction.
    """
    # Assemble conversation roles
    full_prompt_content = [
        {"role": "user", "content": data_point["instruction"]},
        {"role": "assistant", "content": data_point["output"]}
    ]
    user_prompt_content = [{"role": "user", "content": data_point["instruction"]}]

    # Render prompts without immediate tokenization
    full_prompt = tokenizer.apply_chat_template(
        full_prompt_content, tokenize=False, add_generation_prompt=True
    )
    user_prompt = tokenizer.apply_chat_template(
        user_prompt_content, tokenize=False, add_generation_prompt=True
    )

    # Tokenize both prompts
    tokenized_full = tokenizer(full_prompt, return_tensors="pt")
    tokenized_user = tokenizer(user_prompt, return_tensors="pt")
    user_prompt_length = tokenized_user["input_ids"].shape[1]

    # Initialize all labels to -100 (ignore index)
    tokenized_full["labels"] = torch.full_like(
        tokenized_full["input_ids"], -100
    )
    # Only compute loss on assistant output tokens
    tokenized_full["labels"][0, user_prompt_length:] = tokenized_full["input_ids"][0, user_prompt_length:]

    return tokenized_full, user_prompt_length


def compute_token_gradients(model, tokenized_point: dict, start_idx: int, end_idx: int):
    """
    Compute normalized gradient norms for instruction tokens.

    Args:
        model (AutoModelForCausalLM): The language model.
        tokenized_point (dict): Contains 'input_ids', 'attention_mask', and 'labels'.
        start_idx (int): Inclusive index of first instruction token.
        end_idx (int): Exclusive index of last instruction token.

    Returns:
        norms (np.ndarray): Normalized gradient norms per token.
        loss_value (float): Scalar loss value from the forward pass.
    """
    # Embed inputs and enable gradient tracking
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(tokenized_point["input_ids"])
    inputs_embeds.requires_grad_(True)

    # Forward and backward passes
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=tokenized_point["attention_mask"],
        labels=tokenized_point["labels"]
    )
    loss = outputs.loss
    loss.backward()

    # Extract gradients for the specified token range
    gradients = inputs_embeds.grad[0, start_idx:end_idx, :].detach().cpu().numpy()
    norms = np.linalg.norm(gradients, axis=1)
    min_norm, max_norm = norms.min(), norms.max()
    # Normalize to [0,1] if possible
    if max_norm > min_norm:
        norms = (norms - min_norm) / (max_norm - min_norm)

    return norms, loss.item()


def main(args):
    """
    Load data, compare token gradients between two models, and save results.
    """
    # Ensure output directory exists
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Load JSON dataset
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load both models and tokenizers
    model1, tokenizer1 = load_model_and_tokenizer(args.model_name_1)
    model2, tokenizer2 = load_model_and_tokenizer(args.model_name_2)

    mistakes = []

    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        sample_id = idx + 1
        try:
            # Tokenize prompts for each model
            tokenized1, user_len = get_tokenized_data(tokenizer1, sample)
            tokenized2, _ = get_tokenized_data(tokenizer2, sample)

            # Determine instruction span: after second header end or after '####'
            input_ids = tokenized1['input_ids'][0].cpu().tolist()
            start_idx = end_idx = None
            header_count = 0
            found_box = False
            for pos, tid in enumerate(input_ids):
                tok = tokenizer1.decode([tid]).strip()
                if tok == '<|end_header_id|>':
                    header_count += 1
                    if header_count == 2:
                        start_idx = pos + 1
                elif tok == '####' and not found_box:  # mark the start of response-level tokens to compute gradients
                    start_idx = pos + 1
                    found_box = True
                elif tok == '<|eot_id|>' and start_idx is not None:
                    end_idx = pos
                    break

            if start_idx is None or end_idx is None:
                raise ValueError(f"No valid token span found in sample {sample_id}")

            # Compute gradients and differences
            norms1, loss1 = compute_token_gradients(model1, tokenized1, start_idx, end_idx)
            norms2, loss2 = compute_token_gradients(model2, tokenized2, start_idx, end_idx)
            diff = norms2 - norms1
            dmin, dmax = diff.min(), diff.max()
            if dmax > dmin:
                diff = (diff - dmin) / (dmax - dmin)

            # Build and save sample result
            result = {
                "sample_id": sample_id,
                "loss_1": loss1,
                "loss_2": loss2,
                "grad_diff": []
            }
            for i, (d, n1, n2) in enumerate(zip(diff, norms1, norms2)):
                token_str = tokenizer1.decode([tokenized1['input_ids'][0, start_idx + i].item()]).strip()
                result["grad_diff"].append({
                    "token": token_str,
                    "grad_diff": float(d),
                    "grad_norm_1": float(n1),
                    "grad_norm_2": float(n2)
                })

            output_file = os.path.join(args.output_base_dir, f"sample_{sample_id}.json")
            with open(output_file, 'w', encoding='utf-8') as outf:
                json.dump(result, outf, ensure_ascii=False, indent=4)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Skipping sample {sample_id} due to OOM")
                mistakes.append({
                    "sample_id": sample_id,
                    "instruction": sample.get("instruction"),
                    "error": str(e)
                })
                torch.cuda.empty_cache()
                continue
            else:
                raise
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            mistakes.append({
                "sample_id": sample_id,
                "instruction": sample.get("instruction"),
                "error": str(e)
            })

    # Save mistakes log
    with open(args.mistake_file, 'w', encoding='utf-8') as mf:
        json.dump(mistakes, mf, ensure_ascii=False, indent=4)
    print(f"Mistakes saved to {args.mistake_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare token gradients between two causal LLMs and record differences."
    )
    parser.add_argument(
        '--file_path', type=str, required=True,
        help='Path to JSON file containing input samples.'
    )
    parser.add_argument(
        '--model_name_1', type=str, required=True,
        help='Identifier or path for the first model.'
    )
    parser.add_argument(
        '--model_name_2', type=str, required=True,
        help='Identifier or path for the second model.'
    )
    parser.add_argument(
        '--mistake_file', type=str, required=True,
        help='File path to record processing errors.'
    )
    parser.add_argument(
        '--output_base_dir', type=str, required=True,
        help='Directory to save per-sample gradient JSON files.'
    )
    args = parser.parse_args()
    main(args)
