export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FILE_PATH=""
MODEL_NAME_1="/meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME_2="/meta-llama/Llama-3.3-70B-Instruct"
MISTAKE_FILE="llama_1b_mistake.json"
OUTPUT_BASE_DIR="./llama_1b_gradient_data"

# 运行 Python 脚本，并传入参数
python src/Confounding_token_detection/run_gradients_llama_response_level.py\
    --file_path "$FILE_PATH" \
    --model_name_1 "$MODEL_NAME_1" \
    --model_name_2 "$MODEL_NAME_2" \
    --mistake_file "$MISTAKE_FILE" \
    --output_base_dir "$OUTPUT_BASE_DIR"