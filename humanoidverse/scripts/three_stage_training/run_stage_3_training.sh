#!/bin/bash

# 第三阶段训练：冻结第二个策略，训练第一个策略，使用原始奖励
echo "Starting Stage 3: Frozen second policy, training first policy with original rewards"

# 需要指定第二阶段的检查点路径
STAGE_2_CHECKPOINT=${1:-"logs/ThreeStage/Stage2_motion_tracking/model_50000.pt"}

if [ ! -f "$STAGE_2_CHECKPOINT" ]; then
    echo "Error: Stage 2 checkpoint not found at $STAGE_2_CHECKPOINT"
    echo "Usage: $0 [stage_2_checkpoint_path]"
    echo "Please run stage 2 training first or provide the correct checkpoint path"
    echo "Available checkpoints in logs/ThreeStage/Stage2_motion_tracking/:"
    ls -la logs/ThreeStage/Stage2_motion_tracking/*.pt 2>/dev/null || echo "No .pt files found"
    exit 1
fi

echo "Using Stage 2 checkpoint: $STAGE_2_CHECKPOINT"

python humanoidverse/train/train_three_stage.py \
    +config=stage3 \
    current_stage=3 \
    stage_iterations=[5000,5000,5000] \
    algo.config.stage_2_checkpoint="$STAGE_2_CHECKPOINT" \
    num_envs=1024 \
    seed=1029 \
    device=cuda:0 \
    use_wandb=false \
    headless=true \
    robot.motion.motion_file=example/motion_data/Horse-stance_pose.pkl \
    obs.obs_dims.6.dif_local_rigid_body_pos=81 \
    obs.obs_dims.7.local_ref_rigid_body_pos=81 \
    robot.num_key_bodies=2 \
    algo.config.log_dir=logs/ThreeStage 