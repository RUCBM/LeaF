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
    Tokenize the input and prepare labels for gradient computation.

    Args:
        tokenizer (AutoTokenizer): Model tokenizer.
        data_point (dict): A dict with 'instruction' and 'output' strings.

    Returns:
        tokenized_full (dict): Tensors including input_ids and labels for loss.
        user_prompt_length (int): Number of tokens in the user prompt.
    """
    # Construct user-assistant conversation turns
    full_conversation = [
        {"role": "user", "content": data_point["instruction"]},
        {"role": "assistant", "content": data_point["output"]}
    ]
    user_conversation = [
        {"role": "user", "content": data_point["instruction"]}
    ]

    # Render chat templates without tokenizing immediately
    full_prompt = tokenizer.apply_chat_template(
        full_conversation, tokenize=False, add_generation_prompt=True
    )
    user_prompt = tokenizer.apply_chat_template(
        user_conversation, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompts
    tokenized_full = tokenizer(full_prompt, return_tensors="pt")
    tokenized_user = tokenizer(user_prompt, return_tensors="pt")
    user_prompt_length = tokenized_user["input_ids"].shape[1]

    # Initialize all labels to ignore index (-100)
    tokenized_full["labels"] = torch.full_like(
        tokenized_full["input_ids"], -100
    )
    # Only compute loss on assistant response tokens
    tokenized_full["labels"][0, user_prompt_length:] = tokenized_full["input_ids"][0, user_prompt_length:]

    return tokenized_full, user_prompt_length


def compute_token_gradients(model, tokenized_data: dict, start_idx: int, end_idx: int):
    """
    Compute the gradient norms for tokens in a specified range.

    Args:
        model (AutoModelForCausalLM): The language model.
        tokenized_data (dict): Contains input_ids, attention_mask, and labels.
        start_idx (int): Index where gradient computation starts.
        end_idx (int): Index where gradient computation ends (exclusive).

    Returns:
        norms (np.ndarray): Normalized gradient norms for each token in the range.
        loss_value (float): The loss from the model's forward pass.
    """
    # Get embeddings and track gradients
    inputs_embeds = model.get_input_embeddings()(tokenized_data["input_ids"])
    inputs_embeds.requires_grad_(True)

    # Forward and backward passes
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=tokenized_data["attention_mask"],
        labels=tokenized_data["labels"]
    )
    loss = outputs.loss
    loss.backward()

    # Extract gradients for the token range
    grads = inputs_embeds.grad[0, start_idx:end_idx, :].detach().cpu().numpy()
    norms = np.linalg.norm(grads, axis=1)
    min_norm, max_norm = norms.min(), norms.max()
    # Normalize to [0,1]
    if max_norm > min_norm:
        norms = (norms - min_norm) / (max_norm - min_norm)

    return norms, loss.item()


def main(args):
    """
    Process each sample: tokenize, compute token gradients for two models, and save differences.
    """
    # Ensure output directory exists
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Load dataset from JSON file
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load two models and tokenizers
    model1, tokenizer1 = load_model_and_tokenizer(args.model_name_1)
    model2, tokenizer2 = load_model_and_tokenizer(args.model_name_2)

    mistakes = []

    # Iterate samples with progress bar
    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        sample_id = idx + 1
        try:
            # Tokenize data for both models
            tokenized1, user_len = get_tokenized_data(tokenizer1, sample)
            tokenized2, _ = get_tokenized_data(tokenizer2, sample)

            # Identify range using second '<|im_start|>' and '<|im_end|>' markers
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
                raise ValueError(f"No valid gradient token span in sample {sample_id}")

            # Compute and normalize gradient norms
            norms1, loss1 = compute_token_gradients(model1, tokenized1, start_idx, end_idx)
            norms2, loss2 = compute_token_gradients(model2, tokenized2, start_idx, end_idx)
            diff = norms2 - norms1
            dmin, dmax = diff.min(), diff.max()
            if dmax > dmin:
                diff = (diff - dmin) / (dmax - dmin)

            # Build result dictionary
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

            # Save per-sample output
            out_file = os.path.join(args.output_base_dir, f"sample_{sample_id}.json")
            with open(out_file, 'w', encoding='utf-8') as outf:
                json.dump(result, outf, ensure_ascii=False, indent=4)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Skipping sample {sample_id} due to OOM")
                mistakes.append({"sample_id": sample_id, "error": str(e)})
                torch.cuda.empty_cache()
                continue
            else:
                raise
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            mistakes.append({"sample_id": sample_id, "error": str(e)})

    # Save mistakes log
    with open(args.mistake_file, 'w', encoding='utf-8') as mf:
        json.dump(mistakes, mf, ensure_ascii=False, indent=4)
    print(f"Mistakes saved to {args.mistake_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute and compare token gradients for two causal LLMs."
    )
    parser.add_argument(
        '--file_path', type=str, required=True,
        help='JSON file path containing samples.'
    )
    parser.add_argument(
        '--model_name_1', type=str, required=True,
        help='First model identifier or path.'
    )
    parser.add_argument(
        '--model_name_2', type=str, required=True,
        help='Second model identifier or path.'
    )
    parser.add_argument(
        '--mistake_file', type=str, required=True,
        help='Path to save errors log.'
    )
    parser.add_argument(
        '--output_base_dir', type=str, required=True,
        help='Directory to save sample outputs.'
    )
    args = parser.parse_args()
    main(args)
