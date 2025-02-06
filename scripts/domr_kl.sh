#!/bin/bash

export WANDB_MODE=offline
export CUBLAS_WORKSPACE_CONFIG=:16:8
export MASTER_PORT=$((29000 + RANDOM % 1000))

cd /mnt/petrelfs/chenguanxu/SEER
mkdir -p log

model_name_or_path=/path/to/Llama-3.1-8B-Instruct
lorra_alpha=1
layers="25"
transform_layers="-1"
values=(740)
data_path="dataset/math/train"

for value in "${values[@]}"; do
    echo "data_path=$data_path"
    epoch=2
    echo "data: $value, Epoch:$epoch"

    output_dir="models/domr_kl/math_split/Llama-3.1-8B-Instruct-math-cat-cls-7-data-$value-epoch-$epoch-with_kl"
    echo $output_dir
    MASTER_PORT=$((29000 + RANDOM % 1000))
    now=$(date +"%Y%m%d-%H%M%S")
    srun --partition=AI4Good_L --gres=gpu:4 -n1 --ntasks-per-node=1 --job-name=train_emb --kill-on-bad-exit=1 deepspeed src/train.py \
        --deepspeed configs/ds_config.json \
        --model_name_or_path $model_name_or_path \
        --num_examples $value \
        --path $data_path\
        --target_layers $layers \
        --transform_layers $transform_layers \
        --lorra_alpha $lorra_alpha \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir  $output_dir \
        --overwrite_output_dir \
        --num_train_epochs $epoch \
        --bf16 True \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --use_refusal_retain \
        --save_total_limit 4 \
        --save_strategy "epoch" \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --lr_scheduler_type "constant" \
        --logging_strategy "steps" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 8192 \
        --q_lora False \
        --gradient_checkpointing True \
        --report_to none \
        --log_every 10 2>&1 | tee log/test_data-$now.log &
    
    sleep 20

done

lorra_alpha=1
layers="25"
transform_layers="-1"
values=(1600)
data_path="dataset/safety/multi_cls/train"

for value in "${values[@]}"; do
    epoch=2
    echo "data_path=$data_path"
    
    echo "data: $value, Epoch:$epoch"

    output_dir=models/domr_kl/safety/Llama-3.1-8B-Instruct-safety-cls-6-data-$value-epoch-$epoch-with_kl
    echo $output_dir
    MASTER_PORT=$((29000 + RANDOM % 1000))
    now=$(date +"%Y%m%d-%H%M%S")
    srun --partition=AI4Good_L --gres=gpu:4 -n1 --ntasks-per-node=1 --job-name=train_emb --kill-on-bad-exit=1 deepspeed src/train.py \
        --deepspeed configs/ds_config.json \
        --model_name_or_path $model_name_or_path \
        --num_examples $value \
        --path $data_path\
        --target_layers $layers \
        --transform_layers $transform_layers \
        --lorra_alpha $lorra_alpha \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir  $output_dir \
        --overwrite_output_dir \
        --num_train_epochs $epoch \
        --bf16 True \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --use_refusal_retain \
        --save_total_limit 4 \
        --save_strategy "epoch" \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --lr_scheduler_type "constant" \
        --logging_strategy "steps" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 8192 \
        --q_lora False \
        --gradient_checkpointing True \
        --report_to none \
        --log_every 10 2>&1 | tee log/test_data-$now.log &
    
    sleep 20

done


lorra_alpha=1
layers="25"
transform_layers="-1"
values=(200)
data_path="dataset/knowledge/train"

for value in "${values[@]}"; do
    # echo "output_dir=$output_dir"
    echo "data_path=$data_path"
    epoch=2
    echo "data: $value, Epoch:$epoch"

    output_dir="models/domr_kl/mmlu_split/Llama-3.1-8B-Instruct-mmlu-cat-cls-7-data-$value-epoch-$epoch-with_kl"
    echo $output_dir
    MASTER_PORT=$((29000 + RANDOM % 1000))
    now=$(date +"%Y%m%d-%H%M%S")
    srun --partition=AI4Good_L --gres=gpu:4 -n1 --ntasks-per-node=1 --job-name=train_emb --kill-on-bad-exit=1 deepspeed src/train.py \
        --deepspeed configs/ds_config.json \
        --model_name_or_path $model_name_or_path \
        --num_examples $value \
        --path $data_path\
        --target_layers $layers \
        --transform_layers $transform_layers \
        --lorra_alpha $lorra_alpha \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir  $output_dir \
        --overwrite_output_dir \
        --num_train_epochs $epoch \
        --bf16 True \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --use_refusal_retain \
        --save_total_limit 4 \
        --save_strategy "epoch" \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --lr_scheduler_type "constant" \
        --logging_strategy "steps" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 8192 \
        --q_lora False \
        --gradient_checkpointing True \
        --report_to none \
        --log_every 10 2>&1 | tee log/test_data-$now.log &
    
    sleep 20

done
