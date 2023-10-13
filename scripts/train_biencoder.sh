model_dir="models"
data_dir="/saturn/data"
lr="5e-5"

CUDA_VISIBLE_DEVICES=1 python3 saturn/train_biencoder.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --model_type GPT-HUST \
        --num_train_epochs 50 \
        --train_batch_size 277 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr
