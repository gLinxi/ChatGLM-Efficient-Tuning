#!/bin/bash

# params
output_model="rec_reason_v2_without_tag_chatglm6b"

# default
model_base_dir="/data/jupyterlab/gzx/LocalModelHub"
chatglm_6b=${model_base_dir}"/chatglm_6b/hf"
chatglm2_6b=${model_base_dir}"/chatglm2_6b/hf"
output_dir=${model_base_dir}"/${output_model}/hf"
checkpoint=${model_base_dir}"/${output_model}/ckp"

mkdir -p ${output_dir}

set -x
python src/export_model.py \
	--model_name_or_path ${chatglm_6b} \
        --checkpoint_dir ${checkpoint} \
	--output_dir ${output_dir}

cp ${chatglm_6b}/*.py ${output_dir}
