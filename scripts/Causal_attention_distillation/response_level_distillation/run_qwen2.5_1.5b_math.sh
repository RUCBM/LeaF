
SEED=1234
echo "seed=$SEED"
export WANDB_API_KEY="Please add your wandb api key."
export CUDA_VISIBLE_DEVICES=0,1,2,3
python src/Causal_attention_distillation/LeaF_response_level/distill_response_level.py \
    --base_model Qwen2.5-Math-1.5B \
    --teacher_model Qwen2.5-72B-Instruct \
    --data_path Distill_NuminaMath_qwen_1.5_base_misleading_instruct_0.05_step_0.05.json \
    --valid_data_path Numina_val_for_qwen_72b_1.2k_add_step.json \
    --output_dir /save_numina_qwen/qwen_base_instruct_0.05_step_0.05 \
    --prompt_template_name qwen_step \
    --batch_size 32 \
    --micro_batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_steps 5 \
    --lr_scheduler "cosine" \
    --cutoff_len 4096 --padding 'max_length' \
    --val_set_size 1200 --seed ${SEED} \
    --group_by_length False  \
    --wandb_project 'numina_qwen_base' \
    --wandb_run_name 'instruct_0.05_step_0.05' \
    --distill_loss_type KL  --hidden_beta 10000 