SEED=1234
export WANDB_API_KEY="Please add your wandb api key."
echo "seed=$SEED"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python src/Causal_attention_distillation/LeaF_instruct_level/distill_instruct_level.py \
    --base_model meta-llama/Llama-3.2-3B-Instruct \
    --teacher_model /Llama-3.3-70B-Instruct \
    --data_path Distill_NuminaMATH_llama_3b_important_0.10.json \
    --valid_data_path Numina_val_1035_update.json \
    --output_dir ./save_numina_analysis/3b_important_0.10 \
    --prompt_template_name llama3 \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 5 \
    --learning_rate 1e-5 \
    --warmup_steps 5 \
    --lr_scheduler "cosine" \
    --cutoff_len 3000 --padding 'max_length' \
    --val_set_size 1035 --seed ${SEED} \
    --group_by_length False  \
    --wandb_project 'numina_llama' \
    --wandb_run_name '3b_important_0.10' \
    --distill_loss_type KL  --hidden_beta 10000 