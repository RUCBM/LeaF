import json
import re
import argparse
import os

def get_tokens_info(grad_diff_list, instruction_text):
    """
    Analyze tokens and their gradient differences, decide which tokens to mask.

    Args:
        grad_diff_list (list of dict): Each dict has 'token' and 'grad_diff'.
        instruction_text (str): The original instruction string.

    Returns:
        list of dict: Each with 'token' and 'masked' boolean.
    """
    tokens_info = []
    grad_diff_copy = grad_diff_list.copy()

    # Extract all gradient differences and sort ascending
    grad_diff_values = [item["grad_diff"] for item in grad_diff_copy]
    sorted_values = sorted(grad_diff_values)

    # Compute threshold for bottom 10% of grad_diff values
    if sorted_values:
        threshold_index = int(len(sorted_values) * 0.10)
        bottom_10_threshold = sorted_values[threshold_index]
    else:
        bottom_10_threshold = 0
    # Cap the threshold at 0.5
    bottom_10_threshold = min(bottom_10_threshold, 0.5)

    # Iterate through tokens, match each at start of remaining instruction_text
    while grad_diff_copy:
        diff = grad_diff_copy.pop(0)
        token = diff["token"]
        pattern = re.compile(r"^(\s*{})".format(re.escape(token)))
        match_obj = re.search(pattern, instruction_text)
        if not match_obj:
            raise Exception(f"token '{token}' could not be matched in instruction: [{instruction_text}]")

        matched_str = match_obj.group(1)
        instruction_text = instruction_text[len(matched_str):]

        # Decide if this token should be masked (bottom 10% grad_diff)
        mask_flag = (diff["grad_diff"] <= bottom_10_threshold)

        # Never mask pure whitespace or newline characters
        if matched_str.strip() == "" or matched_str in ["\n", "\\"]:
            mask_flag = False

        tokens_info.append({"token": matched_str, "masked": mask_flag})

    return tokens_info


def get_masked_instructions_by_runs(tokens_info):
    """
    Group consecutive masked tokens into runs, then generate masked instruction variants by
    removing those runs.

    Args:
        tokens_info (list of dict): Each dict has 'token' and 'masked'.

    Returns:
        tuple: (list of str masked_instructions, list of list indices runs)
    """
    runs = []
    current_run = []
    # Identify consecutive masked token index runs
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

    # Build masked instruction variants by removing tokens in each run
    masked_texts = []
    for run in runs:
        new_inst = ""
        for i, info in enumerate(tokens_info):
            if i in run:
                continue
            new_inst += info["token"]
        masked_texts.append(new_inst + "\n")

    return masked_texts, runs


def main(instruction_file, grads_file, output_file):
    """
    Load instructions and gradient data, mask tokens, and save results.
    """
    # Load input JSON arrays
    with open(instruction_file, 'r', encoding='utf-8') as f:
        instructions = json.load(f)
    with open(grads_file, 'r', encoding='utf-8') as f:
        grad_entries = json.load(f)

    final_results = []

    for entry in grad_entries:
        # sample_id in grads is 1-based; convert to 0-based index
        sample_idx = entry["sample_id"] - 1
        if sample_idx < 0 or sample_idx >= len(instructions):
            raise Exception(f"sample_id {entry['sample_id']} out of range.")

        inst = instructions[sample_idx]
        original_inst = inst["instruction"]

        try:
            tokens_info = get_tokens_info(entry["grad_diff"], original_inst)
        except Exception:
            # Skip this sample if token matching fails
            continue

        masked_insts, runs = get_masked_instructions_by_runs(tokens_info)
        if not masked_insts:
            masked_insts = [original_inst]

        # For each variant, build a result record
        for idx, masked_inst in enumerate(masked_insts):
            # Collect masked tokens for this run
            masked_tokens = [info["token"] for j, info in enumerate(tokens_info)
                             if info["masked"] and j in runs[idx]]

            record = {
                "id": inst.get("id"),
                "sample_id": entry["sample_id"],
                "gpt_question": masked_inst,
                "masked_tokens": masked_tokens,
                "original_instruction": original_inst,
                "output": inst.get("output"),
                "mask_run_index": idx,
                "context_messages": inst.get("context_messages"),
                "tests": inst.get("tests"),
                "source": inst.get("source")
            }
            final_results.append(record)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Write results to JSON Lines file
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in final_results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask instruction tokens based on gradient differences."
    )
    parser.add_argument(
        "--instruction_file", type=str, required=True,
        help="Path to input JSON file containing instructions."
    )
    parser.add_argument(
        "--grads_file", type=str, required=True,
        help="Path to input JSON file containing gradient data."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to output JSONL file where results will be saved."
    )
    args = parser.parse_args()
    main(args.instruction_file, args.grads_file, args.output_file)
