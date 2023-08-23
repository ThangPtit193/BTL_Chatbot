timestamp=`date "+%Y%0m%0d_%T"`
model_dir="ckpts/ckpt_$timestamp"
data_dir="/shared/vuth/semantic-similarity/collections/v1.0.0/"
wandb_run_name="vuth_avg_005_uniformity"
s="123"
lr="5e-5"

# export WORLD_SIZE=2
# export CUDA_VISIBLE_DEVICES=1,2

CUDA_VISIBLE_DEVICES=2 python3 saturn/train_biencoder.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --token_level word-level \
        --model_type sim-cse-vietnamese \
        --logging_steps 200 \
        --save_steps 200 \
        --wandb_run_name $wandb_run_name \
        --do_train \
        --do_eval \
        --seed $s \
        --num_train_epochs 10 \
        --train_batch_size 512 \
        --eval_batch_size 32 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr \
        --tuning_metric recall_bm-history-v400_5 \
        --early_stopping 25 \
        --resize_embedding_model \
        --pooler_type cls \
        --gradient_checkpointing \
        --optimizer 8bitAdam \
        --sim_fn dot