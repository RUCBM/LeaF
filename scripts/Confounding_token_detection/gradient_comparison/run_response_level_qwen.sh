export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FILE_PATH=""
MODEL_NAME_1="Qwen/Qwen2.5-Math-1.5B"
MODEL_NAME_2="Qwen/Qwen2.5-72B-Instruct"
MISTAKE_FILE="qwen2.5_math_1.5b_mistake.json"
OUTPUT_BASE_DIR="./qwen2.5_math_1.5b_gradient_data"

# 运行 Python 脚本，并传入参数
python src/Confounding_token_detection/run_gradients_qwen_response_level.py \
    --file_path "$FILE_PATH" \
    --model_name_1 "$MODEL_NAME_1" \
    --model_name_2 "$MODEL_NAME_2" \
    --mistake_file "$MISTAKE_FILE" \
    --output_base_dir "$OUTPUT_BASE_DIR"