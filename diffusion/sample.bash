#!/usr/bin/env bash

set -euo pipefail

MODEL_FLAGS="--image_size 128 --learn_sigma True --use_scale_shift_norm True --resblock_updown True --num_channels 128 --channel_mult 1,2,3,4 --num_res_blocks 2 --num_heads 4 --attention_resolutions 8,16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--timestep_respacing ddim25 --use_ddim True --num_samples 16 --batch_size 16"

export OPENAI_LOGDIR="./logs-sample"

python scripts/image_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS --model_path "$1"
