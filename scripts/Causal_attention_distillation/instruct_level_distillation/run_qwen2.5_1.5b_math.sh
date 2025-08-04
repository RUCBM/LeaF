SEED=1234
echo "seed=$SEED"
export WANDB_API_KEY="Please add your wandb api key."
export CUDA_VISIBLE_DEVICES=0,1,2,3
python src/Causal_attention_distillation/LeaF_instruct_level/distill_instruct_level.py \
    --base_model /home/test/testdata/models/Qwen2.5-Math-1.5B \
    --teacher_model /home/test/testdata/models/Qwen2.5-72B-Instruct \
    --data_path  \
    --valid_data_path  \
    --output_dir ./save_qwen_base/qwen_1.5_misleading_0.05 \
    --prompt_template_name qwen \
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
    --wandb_run_name 'qwen_1.5_misleading_0.05' \
    --distill_loss_type KL  --hidden_beta 10000 