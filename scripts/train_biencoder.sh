model_dir="models"
data_dir="/saturn/data"
lr="5e-5"

CUDA_VISIBLE_DEVICES=1 python3 saturn/train_biencoder.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --model_type unsim-cse-vietnamese \
        --save_steps 50 \
        --do_train \
        --num_train_epochs 10 \
        --train_batch_size 1024 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr
