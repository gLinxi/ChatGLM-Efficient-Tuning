#!/bin/bash

# date
date=`date +%Y%m%d%H`

# hub default path
model_base_dir="/root/share/LocalModelHub"
# model_base_dir="/home/apps/gzx/LocalModelHub"
# model_base_dir="/data/jupyterlab/gzx/LocalModelHub/"

# models
baichuan_13b_chat=${model_base_dir}"/baichuan_13b_chat/hf"

# params
gpu_id=0
model="test_chatglm6b"
sft_data="tool_v2_train"
output_dir="${model_base_dir}/"${model}"/ckp"
log_dir="${model_base_dir}/"${model}"/log"

mkdir -p ${output_dir}
mkdir -p ${log_dir}

set -x
CUDA_VISIBLE_DEVICES=${gpu_id} python ./src/train_bash.py \
    --model_name_or_path ${baichuan_13b_chat} \
    --stage sft \
    --do_train \
    --dataset  ${sft_data} \
    --finetuning_type lora \
    --output_dir ${output_dir} \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 3000 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --ddp_find_unused_parameters False \
    --num_train_epochs 3.0 \
    --fp16 > ${log_dir}/${date} 2>&1
