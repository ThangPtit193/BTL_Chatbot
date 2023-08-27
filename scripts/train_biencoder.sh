timestamp=`date "+%Y%0m%0d_%T"`
model_dir="ckpts_negative/ckpt_$timestamp"
data_dir="/home/black/saturn/data"
wandb_run_name="namdp_negative"
s="123"
lr="5e-5"

CUDA_VISIBLE_DEVICES=1 python3 saturn/train_biencoder.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --token_level word-level \
        --model_type unsim-cse-vietnamese \
        --logging_steps 200 \
        --save_steps 200 \
        --wandb_run_name $wandb_run_name \
        --do_train \
        --gpu_id 1\
        --use_negative \
        --seed $s \
        --num_train_epochs 10 \
        --train_batch_size 1024 \
        --eval_batch_size 32 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr \
        --tuning_metric recall_bm_history_v400_5 \
        --early_stopping 10 \
        --pooler_type avg \
        --gradient_checkpointing \
        --sim_fn cosine \
        --pretrained \
        --pretrained_path /home/black/saturn/ckpts/ckpt_20230823_12:26:38