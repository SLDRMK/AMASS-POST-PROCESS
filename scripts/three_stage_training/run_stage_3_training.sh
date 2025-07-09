#!/bin/bash

# 第三阶段训练脚本
# 冻结增量策略，重新训练主策略

echo "Starting Stage 3 Training..."

python humanoidverse/train_three_stage.py \
    --stage 3 \
    --config humanoidverse/config/stage3_config.yaml \
    --stage_1_checkpoint example/pretrained_horse_stance_pose/model_50000.pt \
    --stage_2_checkpoint logs/ThreeStage/Stage2_motion_tracking/stage_2_checkpoint_00050100.pth \
    --log_dir logs/ThreeStage/Stage3_motion_tracking 