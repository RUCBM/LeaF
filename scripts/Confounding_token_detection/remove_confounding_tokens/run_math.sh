INSTRUCTION_FILE="data/llama_1b_gradient_data/Numina_train_data_llama_1b_1.2w.json"
GRADS_FILE="gradient_data_Numina_train_data_llama_1b_1.2w.json"
OUTPUT_FILE="NuminaMATH_llama_1b_misleading_0.05.jsonl"

# 运行 Python 脚本
python3 src/Confounding_token_detection/remove_confounding_tokens_math.py\
    --instruction_file "$INSTRUCTION_FILE" \
    --grads_file "$GRADS_FILE" \
    --output_file "$OUTPUT_FILE" \