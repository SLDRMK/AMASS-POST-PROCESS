#!/bin/bash

# 第二阶段训练：冻结第一个策略，训练第二个策略，使用delta奖励
echo "Starting Stage 2: Frozen first policy, training second policy with delta rewards"

# 需要指定第一阶段的检查点路径
STAGE_1_CHECKPOINT=${1:-"example/pretrained_horse_stance_pose/model_50000.pt"}

if [ ! -f "$STAGE_1_CHECKPOINT" ]; then
    echo "Error: Stage 1 checkpoint not found at $STAGE_1_CHECKPOINT"
    echo "Usage: $0 [stage_1_checkpoint_path]"
    echo "Please run stage 1 training first or provide the correct checkpoint path"
    exit 1
fi

echo "Using Stage 1 checkpoint: $STAGE_1_CHECKPOINT"

python humanoidverse/train/train_three_stage.py \
    +stage_1_checkpoint="$STAGE_1_CHECKPOINT" 