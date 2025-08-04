import json
import re
import argparse
import os

def get_tokens_info(grad_diff_list, instruction_text):
    """
    Analyze the list of tokens with their gradient differences and determine which tokens to mask.

    Args:
        grad_diff_list (list of dict): Each dict contains 'token' and 'grad_diff' values.
        instruction_text (str): The original instruction string.

    Returns:
        List[dict]: Each dict has 'token' (str) and 'masked' (bool) indicating if the token should be masked.
    """
    tokens_info = []
    grad_diff_copy = grad_diff_list.copy()

    # Extract all grad_diff values and sort them ascending
    grad_diff_values = [entry["grad_diff"] for entry in grad_diff_copy]
    sorted_values = sorted(grad_diff_values)

    # Compute threshold for the bottom 5% of gradients
    if sorted_values:
        threshold_idx = int(len(sorted_values) * 0.05)
        bottom_5_percent_threshold = sorted_values[threshold_idx]
    else:
        bottom_5_percent_threshold = 0.0
    # Cap maximum threshold at 0.5 to avoid extreme masking
    bottom_5_percent_threshold = min(bottom_5_percent_threshold, 0.5)

    # Iterate through each gradient entry and match tokens sequentially in the instruction
    while grad_diff_copy:
        diff = grad_diff_copy.pop(0)
        token = diff["token"]
        # Match the token at the start of the remaining instruction text (allowing leading whitespace)
        pattern = re.compile(r"^(\s*{})".format(re.escape(token)))
        match_obj = re.search(pattern, instruction_text)
        if not match_obj:
            raise Exception(f"Token '{token}' could not be matched in remaining instruction: [{instruction_text}]")

        matched_str = match_obj.group(1)
        # Remove the matched part from the instruction text for subsequent matches
        instruction_text = instruction_text[len(matched_str):]

        # Decide whether to mask: True if gradient difference is in the bottom 5%
        masked_flag = diff["grad_diff"] <= bottom_5_percent_threshold
        # Never mask pure whitespace or newline tokens
        if matched_str.strip() == "" or matched_str in ['\n', '\\']:
            masked_flag = False

        tokens_info.append({
            "token": matched_str,
            "masked": masked_flag
        })

    return tokens_info


def get_masked_instructions_by_runs(tokens_info):
    """
    Group consecutive masked tokens into runs and generate masked instruction variants by removing each run.

    Args:
        tokens_info (list of dict): Each dict has 'token' and 'masked'.

    Returns:
        Tuple[List[str], List[List[int]]]:
            - List of masked instruction strings (one variant per run).
            - List of runs (each run is a list of token indices that were masked).
    """
    runs = []
    current_run = []
    # Identify consecutive sequences of masked tokens
    for idx, info in enumerate(tokens_info):
        if info["masked"]:
            if not current_run or idx == current_run[-1] + 1:
                current_run.append(idx)
            else:
                runs.append(current_run)
                current_run = [idx]
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
    if current_run:
        runs.append(current_run)

    masked_texts = []
    # For each run, reconstruct the instruction without those tokens
    for run in runs:
        new_text = ""
        for i, info in enumerate(tokens_info):
            if i in run:
                continue  # skip masked tokens
            new_text += info["token"]
        masked_texts.append(new_text + "\n")

    return masked_texts, runs


def main(instruction_file, grads_file, output_file):
    """
    Load instruction and gradient data, apply token masking, and write results to a JSONL file.

    Args:
        instruction_file (str): Path to the JSON file containing instructions.
        grads_file (str): Path to the JSON file containing gradient differences.
        output_file (str): Path to the output JSONL file.
    """
    # Read the list of instructions
    with open(instruction_file, 'r', encoding='utf-8') as f:
        instructions = json.load(f)
    # Read the list of gradient entries
    with open(grads_file, 'r', encoding='utf-8') as f:
        grad_entries = json.load(f)

    results = []

    for entry in grad_entries:
        # Convert sample_id from 1-based to 0-based index
        sample_idx = entry["sample_id"] - 1
        if sample_idx < 0 or sample_idx >= len(instructions):
            raise Exception(f"sample_id {entry['sample_id']} is out of range.")

        inst_record = instructions[sample_idx]
        original_inst = inst_record.get("instruction")
        original_out = inst_record.get("output")
        prompt_answer = inst_record.get("prompt_answer")
        filepath = inst_record.get("filepath")
        record_type = inst_record.get("type")
        source = inst_record.get("source")

        try:
            tokens_info = get_tokens_info(entry.get("grad_diff", []), original_inst)
        except Exception:
            # Skip if token matching fails
            continue

        masked_variants, runs = get_masked_instructions_by_runs(tokens_info)
        if not masked_variants:
            masked_variants = [original_inst]

        # Build a result record for each masked variant
        for idx, variant in enumerate(masked_variants):
            masked_tokens = [info["token"] for i, info in enumerate(tokens_info)
                             if info["masked"] and i in runs[idx]]

            record = {
                "sample_id": entry["sample_id"],
                "instruction": variant,
                "masked_tokens": masked_tokens,
                "original_instruction": original_inst,
                "output": original_out,
                "filepath": filepath,
                "answer": prompt_answer,
                "type": record_type,
                "mask_run_index": idx,
                "source": source
            }
            results.append(record)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Write each result as a separate JSON line
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask instruction tokens based on gradient differences."
    )
    parser.add_argument(
        "--instruction_file", type=str, required=True,
        help="Path to the JSON file with input instructions."
    )
    parser.add_argument(
        "--grads_file", type=str, required=True,
        help="Path to the JSON file with gradient difference data."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Destination JSONL file for masked outputs."
    )
    args = parser.parse_args()
    main(args.instruction_file, args.grads_file, args.output_file)
