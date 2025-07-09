#!/bin/bash

# 完整的三阶段训练脚本
# 自动执行所有三个阶段：主策略训练 -> 增量策略训练 -> 主策略重新训练

set -e  # 遇到错误时退出

echo "=========================================="
echo "Starting Full Three-Stage Training Pipeline"
echo "=========================================="

# 创建日志目录
mkdir -p logs/ThreeStage

# 第一阶段：训练主策略
echo ""
echo "=========================================="
echo "Stage 1: Training Main Policy"
echo "=========================================="
echo "Training main policy for motion tracking..."

python humanoidverse/train_three_stage.py \
    --stage 1 \
    --config humanoidverse/config/stage1_config.yaml \
    --log_dir logs/ThreeStage/Stage1_motion_tracking

echo "Stage 1 completed successfully!"

# 等待一下确保文件写入完成
sleep 2

# 检查第一阶段是否生成了检查点
STAGE1_CHECKPOINT="logs/ThreeStage/Stage1_motion_tracking/model_2000.pt"
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "Warning: Expected Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    echo "Looking for available checkpoints..."
    ls -la logs/ThreeStage/Stage1_motion_tracking/model_*.pt 2>/dev/null || echo "No checkpoints found"
    
    # 尝试找到最新的检查点
    LATEST_CHECKPOINT=$(ls -t logs/ThreeStage/Stage1_motion_tracking/model_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        STAGE1_CHECKPOINT="$LATEST_CHECKPOINT"
        echo "Using latest checkpoint: $STAGE1_CHECKPOINT"
    else
        echo "Error: No Stage 1 checkpoints found. Cannot proceed to Stage 2."
        exit 1
    fi
fi

echo "Using Stage 1 checkpoint: $STAGE1_CHECKPOINT"

# 第二阶段：冻结主策略，训练增量策略
echo ""
echo "=========================================="
echo "Stage 2: Training Delta Policy"
echo "=========================================="
echo "Freezing main policy and training delta policy..."

python humanoidverse/train_three_stage.py \
    --stage 2 \
    --config humanoidverse/config/stage2_config.yaml \
    --stage_1_checkpoint "$STAGE1_CHECKPOINT" \
    --log_dir logs/ThreeStage/Stage2_motion_tracking

echo "Stage 2 completed successfully!"

# 等待一下确保文件写入完成
sleep 2

# 检查第二阶段是否生成了检查点
STAGE2_CHECKPOINT="logs/ThreeStage/Stage2_motion_tracking/model_2000.pt"
if [ ! -f "$STAGE2_CHECKPOINT" ]; then
    echo "Warning: Expected Stage 2 checkpoint not found at $STAGE2_CHECKPOINT"
    echo "Looking for available checkpoints..."
    ls -la logs/ThreeStage/Stage2_motion_tracking/model_*.pt 2>/dev/null || echo "No checkpoints found"
    
    # 尝试找到最新的检查点
    LATEST_CHECKPOINT=$(ls -t logs/ThreeStage/Stage2_motion_tracking/model_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        STAGE2_CHECKPOINT="$LATEST_CHECKPOINT"
        echo "Using latest checkpoint: $STAGE2_CHECKPOINT"
    else
        echo "Error: No Stage 2 checkpoints found. Cannot proceed to Stage 3."
        exit 1
    fi
fi

echo "Using Stage 2 checkpoint: $STAGE2_CHECKPOINT"

# 第三阶段：冻结增量策略，重新训练主策略
echo ""
echo "=========================================="
echo "Stage 3: Re-training Main Policy"
echo "=========================================="
echo "Freezing delta policy and re-training main policy..."

python humanoidverse/train_three_stage.py \
    --stage 3 \
    --config humanoidverse/config/stage3_config.yaml \
    --stage_1_checkpoint "$STAGE1_CHECKPOINT" \
    --stage_2_checkpoint "$STAGE2_CHECKPOINT" \
    --log_dir logs/ThreeStage/Stage3_motion_tracking

echo "Stage 3 completed successfully!"

echo ""
echo "=========================================="
echo "Full Three-Stage Training Pipeline Completed!"
echo "=========================================="
echo ""
echo "Training Results:"
echo "- Stage 1: logs/ThreeStage/Stage1_motion_tracking/"
echo "- Stage 2: logs/ThreeStage/Stage2_motion_tracking/"
echo "- Stage 3: logs/ThreeStage/Stage3_motion_tracking/"
echo ""
echo "Final model: logs/ThreeStage/Stage3_motion_tracking/model_2000.pt"
echo ""
echo "You can now evaluate the trained model or use it for inference." 