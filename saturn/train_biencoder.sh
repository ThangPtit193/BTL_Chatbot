export lr="5e-5"
export s="100"
echo "${lr}"
export MODEL_DIR=MODELBASE
echo "${MODEL_DIR}"
python3 saturn/train_biencoder.py \
        --token_level word-level \
        --model_type phobert-base-v2 \
        --model_dir $MODEL_DIR \
        --data_dir /shared/vuth/semantic-similarity/collections/v1.0.0/word-level/ \
        --seed $s --do_train \
        --save_steps 1000 \
        --logging_steps 20 \
        --num_train_epochs 10 \
        --tuning_metric loss \
        --gpu_id 0 \
        --learning_rate $lr \
        --train_batch_size 128 \
        --eval_batch_size 256 \
        --early_stopping 25 \
        --wandb_run_name test_source \
        --benchmark /shared/vuth/semantic-similarity/collections/bm/bm_history_cttgt2.jsonl \
        --corpus /shared/vuth/semantic-similarity/collections/bm/corpus_history.json