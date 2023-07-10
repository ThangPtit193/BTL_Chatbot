export lr="5e-5"
export s="100"
echo "${lr}"
export MODEL_DIR=MODELBASE
echo "${MODEL_DIR}"
CUDA_VISIBLE_DEVICES=2 python3 train_biencoder.py \
        --token_level syllable \
        --model_type phobert-base-v2 \
        --model_dir $MODEL_DIR \
        --data_dir /shared/vuth/semantic-similarity/collections/v1.0.0 \
        --seed $s --do_train \
        --save_steps 500 \
        --logging_steps 500 \
        --num_train_epochs 10 \
        --tuning_metric loss \
        --gpu_id 2 \
        --learning_rate $lr \
        --train_batch_size 128 \
        --eval_batch_size 256 \
        --early_stopping 25 \
        --wandb_run_name test_v1.0.0