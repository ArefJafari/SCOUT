#!/bin/bash

# Get the project root directory (assuming this script is in train/scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

#  elect which GPUs to use
# Example: export CUDA_VISIBLE_DEVICES="0,1,2,3" or just "0" for 1 GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Number of GPUs (based on how many you listed above)
NUM_GPUS=8

# Training config
NAME="SCOUT_SWA_1.59B"
MODEL="${PROJECT_ROOT}/configs/scout_swa_1.57b.json"
CONFIG="1024x2k_100B"
MICRO_BATCH_SIZE=8
EVAL_ITERS=15
LR=3e-4

# Paths
OUTPUT_ROOT="${PROJECT_ROOT}/train"
TRAIN_DATA="${PROJECT_ROOT}/datasets/fineweb-edu/100B/fla_tokenized"
VALIDATION_DATA=None
SAVE_DIR="${PROJECT_ROOT}/save/"

# Run training
torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 ${OUTPUT_ROOT}/pretrain.py --train_data_dir ${TRAIN_DATA} --val_data_dir ${VALIDATION_DATA} --output_root ${SAVE_DIR} --exp_name ${NAME} --model_name ${MODEL} --eval_iters ${EVAL_ITERS} --learning_rate ${LR} --micro_batch_size ${MICRO_BATCH_SIZE} --train_config ${CONFIG} 