#!/bin/bash

# 第一阶段训练：单独训练第一个策略
echo "Starting Stage 1: Training first policy only"

python humanoidverse/train/train_three_stage.py \
    +config=stage1 \
    current_stage=1 \
    num_envs=1024 \
    seed=1029 \
    device=cuda:0 \
    use_wandb=false \
    headless=true 