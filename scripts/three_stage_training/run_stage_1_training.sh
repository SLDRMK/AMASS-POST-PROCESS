#!/bin/bash

# 第一阶段训练脚本
# 训练主策略

echo "Starting Stage 1 Training..."

python humanoidverse/train_three_stage.py \
    --stage 1 \
    --config humanoidverse/config/stage1_config.yaml \
    --log_dir logs/ThreeStage/Stage1_motion_tracking 