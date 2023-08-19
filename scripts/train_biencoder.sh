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
        --logging_steps 1500 \
        --save_steps 1500 \
        --wandb_run_name vuth_avg_005_uniformity \
        --do_train \
        --do_eval \
        --seed $s \
        --num_train_epochs 10 \
        --train_batch_size 512 \
        --eval_batch_size 256 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr \
        --tuning_metric recall_bm-history-v400_5 \
        --early_stopping 25 \
        --resize_embedding_model \
        --use_uniformity_loss \
        --coff_uniformity 0.05 \
        --pooler_type cls \
        --pretrained \
        --pretrained_path /shared/lydhk/models/SIMCSE_UNI_LOSS