export lr="5e-5"
export s="100"
echo "${lr}"
export MODEL_DIR=MODELBASE
echo "${MODEL_DIR}"
python3 saturn/train_biencoder.py \
        --model_dir $MODEL_DIR \
        --data_dir /shared/vuth/semantic-similarity/collections/v1.0.0/ \
        --token_level word-level \
        --model_type sim-cse-vietnamese \
        --logging_steps 50 \
        --save_steps 1000 \
        --wandb_run_name test_source \
        --do_train \
        --do_eval \
        --seed $s \
        --num_train_epochs 10 \
        --train_batch_size 256 \
        --eval_batch_size 256 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr \
        --tuning_metric total_loss \
        --early_stopping 25 \
        --resize_embedding_model \
        --pooler_type avg