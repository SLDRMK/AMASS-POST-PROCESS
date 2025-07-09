#!/bin/bash

# 第二阶段训练脚本
# 冻结主策略，训练增量策略

echo "Starting Stage 2 Training..."

python humanoidverse/train_three_stage.py \
    --stage 2 \
    --config humanoidverse/config/stage2_config.yaml \
    --stage_1_checkpoint example/pretrained_horse_stance_pose/model_50000.pt \
    --log_dir logs/ThreeStage/Stage2_motion_tracking 