import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm

def load_model_and_tokenizer(model_name: str):
    """
    Load a causal language model and its tokenizer.

    Args:
        model_name (str): Path or identifier for the pretrained model.

    Returns:
        model (AutoModelForCausalLM): The loaded language model in training mode.
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
    Prepare tokenized inputs for gradient computation by splitting user and assistant roles.

    Args:
        tokenizer (AutoTokenizer): The model's tokenizer.
        data_point (dict): Contains 'instruction' and 'output' keys.

    Returns:
        tokenized_full (dict): Tokenized prompt with labels for loss computation.
        user_prompt_length (int): Number of tokens in the user prompt.
    """
    # Build conversation turns
    full_turns = [
        {"role": "user", "content": data_point["instruction"]},
        {"role": "assistant", "content": data_point["output"]}
    ]
    user_turns = [
        {"role": "user", "content": data_point["instruction"]}
    ]

    # Render chat templates without immediate tokenization
    full_prompt = tokenizer.apply_chat_template(full_turns, tokenize=False, add_generation_prompt=True)
    user_prompt = tokenizer.apply_chat_template(user_turns, tokenize=False, add_generation_prompt=True)

    # Tokenize and setup labels for loss
    tokenized_full = tokenizer(full_prompt, return_tensors="pt")
    tokenized_user = tokenizer(user_prompt, return_tensors="pt")
    user_prompt_length = tokenized_user["input_ids"].shape[1]

    tokenized_full["labels"] = torch.full_like(
        tokenized_full["input_ids"], -100
    )
    # Only compute loss on assistant response tokens
    tokenized_full["labels"][0, user_prompt_length:] = tokenized_full["input_ids"][0, user_prompt_length:]

    return tokenized_full, user_prompt_length


def compute_token_gradients(model, tokenized_point: dict, start_idx: int, end_idx: int):
    """
    Compute normalized gradient norms for each token in the instruction portion.

    Args:
        model (AutoModelForCausalLM): The language model.
        tokenized_point (dict): Contains input_ids, attention_mask, labels.
        start_idx (int): Start index of tokens to analyze.
        end_idx (int): End index (exclusive) of tokens to analyze.

    Returns:
        grad_norms (np.ndarray): Normalized gradient norms per token.
        loss_value (float): Scalar loss from the forward pass.
    """
    # Embed inputs and enable gradient tracking
    inputs_embeds = model.get_input_embeddings()(tokenized_point["input_ids"])
    inputs_embeds.requires_grad_(True)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=tokenized_point["attention_mask"],
        labels=tokenized_point["labels"]
    )
    loss = outputs.loss
    loss.backward()

    # Extract gradients for the instruction tokens and compute norms
    grads = inputs_embeds.grad[0, start_idx:end_idx, :].detach().cpu().numpy()
    norms = np.linalg.norm(grads, axis=1)
    min_norm, max_norm = norms.min(), norms.max()
    if max_norm > min_norm:
        norms = (norms - min_norm) / (max_norm - min_norm)

    return norms, loss.item()


def main(args):
    """
    Main entry point: load data, compute gradient differences between two models, save per-sample JSON.
    """
    # Ensure output directory exists
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Load JSON dataset
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize both models and tokenizers
    model1, tokenizer1 = load_model_and_tokenizer(args.model_name_1)
    model2, tokenizer2 = load_model_and_tokenizer(args.model_name_2)

    mistakes = []

    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        sample_id = idx + 1
        try:
            # Tokenize for both models
            tokenized1, user_len = get_tokenized_data(tokenizer1, sample)
            tokenized2, _ = get_tokenized_data(tokenizer2, sample)

            # Identify instruction token span via special IDs
            input_ids = tokenized1['input_ids'][0].cpu().tolist()
            start_idx = end_idx = None
            header_count = 0
            for pos, tid in enumerate(input_ids):
                token = tokenizer1.decode([tid]).strip()
                if token == '<|end_header_id|>':
                    header_count += 1
                    if header_count == 2:
                        start_idx = pos + 1
                elif token == '<|eot_id|>' and start_idx is not None:
                    end_idx = pos
                    break

            if start_idx is None or end_idx is None:
                raise ValueError(f"Valid token span not found in sample {sample_id}")
            # Adjust indices to skip templates
            start_idx += 1
            end_idx -= 15

            # Compute gradients and their differences
            norms1, loss1 = compute_token_gradients(model1, tokenized1, start_idx, end_idx)
            norms2, loss2 = compute_token_gradients(model2, tokenized2, start_idx, end_idx)
            diff = norms2 - norms1
            min_d, max_d = diff.min(), diff.max()
            if max_d > min_d:
                diff = (diff - min_d) / (max_d - min_d)

            # Build output record
            record = {
                "sample_id": sample_id,
                "loss_1": loss1,
                "loss_2": loss2,
                "grad_diff": []
            }
            for i, (d, n1, n2) in enumerate(zip(diff, norms1, norms2)):
                token = tokenizer1.decode([tokenized1['input_ids'][0, start_idx + i].item()]).strip()
                record["grad_diff"].append({
                    "token": token,
                    "grad_diff": float(d),
                    "grad_norm_1": float(n1),
                    "grad_norm_2": float(n2)
                })

            # Write per-sample JSON
            out_file = os.path.join(args.output_base_dir, f"sample_{sample_id}.json")
            with open(out_file, 'w', encoding='utf-8') as wf:
                json.dump(record, wf, ensure_ascii=False, indent=4)

        except RuntimeError as e:
            # Handle OOM by logging and clearing cache
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

    # Save mistake log
    with open(args.mistake_file, 'w', encoding='utf-8') as mf:
        json.dump(mistakes, mf, ensure_ascii=False, indent=4)
    print(f"Mistakes saved to {args.mistake_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare token gradients between two causal LLMs."
    )
    parser.add_argument(
        '--file_path', type=str, required=True,
        help='Path to the input JSON file containing data samples.'
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
        help='File path to record any processing errors.'
    )
    parser.add_argument(
        '--output_base_dir', type=str, required=True,
        help='Directory to save per-sample gradient JSON files.'
    )
    args = parser.parse_args()
    main(args)
