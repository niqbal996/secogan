#!/usr/bin/env bash
nvidia-smi
export PYTHONWARNINGS="ignore::DeprecationWarning"
python3 train.py \
    --name=syn2real_style_transfer_corn \
    --gpu_ids=0,1 \
    --data_source=/netscratch/naeem/syn_real_dataset_uda/syn_data/ \
    --data_target=/netscratch/naeem/syn_real_dataset_uda/real_data/ \
    --output_dir=/netscratch/naeem/secogan_output/ \
    --load_size=768 \
    --crop_size=512 \
    --num_workers=16 \
    --batch_size=12