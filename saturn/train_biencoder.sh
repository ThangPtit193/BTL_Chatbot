export lr="5e-5"
export s="100"
echo "${lr}"
export MODEL_DIR=MODELBASE
echo "${MODEL_DIR}"
python3 train_biencoder.py \
        --token_level syllable \
        --model_type phobert-base-v2 \
        --model_dir $MODEL_DIR \
        --data_dir /home/vth/semantic-search/src/saturn/data/dummy \
        --seed $s --do_train \
        --save_steps 20 \
        --logging_steps 5 \
        --num_train_epochs 100 \
        --tuning_metric loss \
        --gpu_id 0 \
        --learning_rate $lr \
        --train_batch_size 64 \
        --eval_batch_size 128 \
        --early_stopping 25 \