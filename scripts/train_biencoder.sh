timestamp=`date "+%Y%0m%0d_%T"`
model_dir="ckpts/ckpt_$timestamp"
data_dir="/home/black/saturn/data"
lr="5e-5"

CUDA_VISIBLE_DEVICES=1 python3 saturn/train_biencoder.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --token_level word-level \
        --model_type unsim-cse-vietnamese \
        --logging_steps 200 \
        --save_steps 200 \
        --do_train \
        --gpu_id 1\
        --num_train_epochs 10 \
        --train_batch_size 1024 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr \
