
SEED=1234
echo "seed=$SEED"
export WANDB_API_KEY="Please add your wandb api key."

export CUDA_VISIBLE_DEVICES=0,1,2,3
python src/Causal_attention_distillation/LeaF_response_level/distill_response_level.py \
    --base_model /meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model /meta-llama/Llama-3.3-70B-Instruct \
    --data_path data/llama_1b_response_level/Distill_NuminaMath_llama_1b_misleading_step_0.075.json \
    --valid_data_path data/llama_1b_response_level/Numina_eval_1035_with_step_response.json  \
    --output_dir /save_numina_llama/llama_1b_step_0.075 \
    --prompt_template_name llama3_step \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_steps 5 \
    --lr_scheduler "cosine" \
    --cutoff_len 4096 --padding 'max_length' \
    --val_set_size 1035 --seed ${SEED} \
    --group_by_length False  \
    --wandb_project 'numina_llama' \
    --wandb_run_name 'llama_1b_step_0.075' \
    --distill_loss_type KL  --hidden_beta 10000 