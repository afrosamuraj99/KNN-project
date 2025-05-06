#!/usr/bin/env bash

python sample.py \
  --name cifar-grayscale-5 \
  --ddpm \
  --out samples_gray_cond.png \
  --num_samples 16 \
  --batch_size 16 \
  --grayscale ../diffusion/datasets/cifar_test