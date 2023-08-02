#!/bin/bash

# date
date=`date +%Y%m%d%H`

# hub default path
# model_base_dir="/root/share/LocalModelHub"
# model_base_dir="/home/apps/gzx/LocalModelHub"
model_base_dir="/data/jupyterlab/gzx/LocalModelHub"

# models
chatglm2_6b=${model_base_dir}"/chatglm2_6b/hf"
chatglm_6b=${model_base_dir}"/chatglm_6b/hf"

# params
gpu_id=6
model="tool_v3_with_thought_chatglm2_6b_lora_nml"
sft_data="tool_v3_thought_train"
output_dir="${model_base_dir}/"${model}"/ckp"
log_dir="${model_base_dir}/"${model}"/log"

mkdir -p ${output_dir}
mkdir -p ${log_dir}

set -x
CUDA_VISIBLE_DEVICES=${gpu_id} python ./src/train_bash.py \
    --stage sft \
    --model_name_or_path ${chatglm2_6b} \
    --do_train \
    --dataset  ${sft_data}\
    --finetuning_type lora \
    --output_dir ${output_dir} \
    --overwrite_cache \
    --save_steps 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --dev_ratio 0.05 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --save_total_limit 5 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --ddp_find_unused_parameters False \
    --num_train_epochs 3.0 \
    --fp16 > ${log_dir}/${date} 2>&1
