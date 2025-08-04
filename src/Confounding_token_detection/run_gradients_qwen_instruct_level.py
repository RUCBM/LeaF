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
    Prepare tokenized input and label tensors for gradient computation.

    Args:
        tokenizer (AutoTokenizer): The model tokenizer.
        data_point (dict): Contains 'instruction' and 'output' keys.

    Returns:
        tokenized_full (dict): Tokenized prompt with labels for loss calculation.
        user_prompt_length (int): Number of tokens in the user instruction.
    """
    # Build the full conversation sequence
    full_prompt_content = [
        {"role": "user", "content": data_point["instruction"]},
        {"role": "assistant", "content": data_point["output"]}
    ]
    # Build only the user prompt
    user_prompt_content = [{"role": "user", "content": data_point["instruction"]}]

    # Render chat templates without tokenization
    full_prompt = tokenizer.apply_chat_template(
        full_prompt_content, tokenize=False, add_generation_prompt=True
    )
    user_prompt = tokenizer.apply_chat_template(
        user_prompt_content, tokenize=False, add_generation_prompt=True
    )

    # Tokenize prompts
    tokenized_full = tokenizer(full_prompt, return_tensors="pt")
    tokenized_user = tokenizer(user_prompt, return_tensors="pt")
    user_prompt_length = tokenized_user["input_ids"].shape[1]

    # Initialize labels to ignore index (-100)
    tokenized_full["labels"] = torch.full_like(
        tokenized_full["input_ids"], -100
    )
    # Compute loss only on assistant's output tokens
    tokenized_full["labels"][0, user_prompt_length:] = tokenized_full["input_ids"][0, user_prompt_length:]

    return tokenized_full, user_prompt_length


def compute_token_gradients(model, tokenized_point: dict, start_idx: int, end_idx: int):
    """
    Compute normalized gradient norms for tokens in the specified range.

    Args:
        model (AutoModelForCausalLM): The language model.
        tokenized_point (dict): Contains 'input_ids', 'attention_mask', and 'labels'.
        start_idx (int): Inclusive index where gradient computation starts.
        end_idx (int): Exclusive index where gradient computation ends.

    Returns:
        norms (np.ndarray): Normalized gradient norms for each token.
        loss_value (float): Loss from the model's forward pass.
    """
    # Get input embeddings and enable gradient tracking
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(tokenized_point["input_ids"])
    inputs_embeds.requires_grad_(True)

    # Forward and backward pass
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
    # Normalize gradient norms to [0,1]
    if max_norm > min_norm:
        norms = (norms - min_norm) / (max_norm - min_norm)

    return norms, loss.item()


def main(args):
    """
    Load dataset, compute gradient differences between two models, and save results per sample.
    """
    # Create output directory if missing
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Load JSON dataset
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize models and tokenizers
    model1, tokenizer1 = load_model_and_tokenizer(args.model_name_1)
    model2, tokenizer2 = load_model_and_tokenizer(args.model_name_2)

    mistakes = []

    # Iterate over each data sample
    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        sample_id = idx + 1
        try:
            tokenized1, user_len = get_tokenized_data(tokenizer1, sample)
            tokenized2, _ = get_tokenized_data(tokenizer2, sample)

            # Identify the range to compute gradients:
            # use '<|im_start|>' markers to mark start and '<|im_end|>' for end
            input_ids = tokenized1['input_ids'][0].cpu().tolist()
            start_idx = end_idx = None
            start_count = 0
            for pos, tid in enumerate(input_ids):
                tok = tokenizer1.decode([tid]).strip()
                if tok == '<|im_start|>':
                    start_count += 1
                    if start_count == 2:
                        # mark the start of response-level tokens to compute gradients
                        start_idx = pos + 1
                elif tok == '<|im_end|>' and start_idx is not None:
                    end_idx = pos
                    break

            if start_idx is None or end_idx is None:
                raise ValueError(f"No valid token span found in sample {sample_id}")

            # Compute gradients and calculate differences
            norms1, loss1 = compute_token_gradients(model1, tokenized1, start_idx, end_idx)
            norms2, loss2 = compute_token_gradients(model2, tokenized2, start_idx, end_idx)
            diff = norms2 - norms1
            dmin, dmax = diff.min(), diff.max()
            if dmax > dmin:
                diff = (diff - dmin) / (dmax - dmin)

            # Build result for this sample
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

            # Write sample result to JSON file
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

    # Save error log
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