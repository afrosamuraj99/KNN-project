#!/usr/bin/env bash

set -euo pipefail

MODEL_FLAGS="--image_size 32 --dropout 0.1 --learn_sigma True --use_scale_shift_norm True --resblock_updown True --num_channels 32 --channel_mult 1,2,4 --num_res_blocks 2 --num_heads 4 --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--timestep_respacing ddim25 --use_ddim True"

export OPENAI_LOGDIR="./logs-sample"

python scripts/image_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS $*
