#!/usr/bin/env bash
export PYTHONWARNINGS="ignore::DeprecationWarning"
python3 test.py \
    --name=syn2real_style_transfer_corn_2 \
    --gpu_ids=0 \
    --data_source=/netscratch/naeem/syn_real_dataset_uda/syn_data/ \
    --output_dir=/netscratch/naeem/secogan_output/ \
    --batch_size=2 \
    --num_workers=4 \
    --weights=/netscratch/naeem/secogan_output/syn2real_style_transfer_corn_2/checkpoints