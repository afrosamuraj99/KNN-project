#!/usr/bin/env bash

set -euo pipefail

MODEL_FLAGS="--image_size 128 --learn_sigma True --use_scale_shift_norm True --resblock_updown True --num_channels 128 --channel_mult 1,2,3,4 --num_res_blocks 2 --num_heads 4 --attention_resolutions 8,16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--batch_size 256 --microbatch 20 --save_interval 50 --log_interval 10 --lr 1e-4"

export OPENAI_LOGDIR="./logs-train"

python scripts/image_train.py --data_dir ../../datasets/coco/train_imgs/resized_128 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
