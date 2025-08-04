INSTRUCTION_FILE="/acecode_qwen_base/Acecode_qwen_base_1.5_false_instruct_level.json"
GRADS_FILE="/acecode_qwen_base/gradient_qwen_base_1.5b_instruct_level.json"
OUTPUT_FILE="/Acecode_qwen_base_1.5b_misleading_0.10.jsonl"

# 运行 Python 脚本
python3 src/Confounding_token_detection/remove_confounding_tokens_code.py\
    --instruction_file "$INSTRUCTION_FILE" \
    --grads_file "$GRADS_FILE" \
    --output_file "$OUTPUT_FILE" \