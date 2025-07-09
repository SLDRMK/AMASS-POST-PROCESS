#!/bin/bash

# 完整的三阶段双策略PPO训练脚本
echo "==================================================="
echo "三阶段双策略PPO训练 - 马步姿态跟踪"
echo "==================================================="
echo "第一阶段：单独训练第一个策略"
echo "第二阶段：冻结第一个策略，训练第二个策略（delta奖励）"
echo "第三阶段：冻结第二个策略，训练第一个策略（原始奖励）"
echo "==================================================="

# 检查GPU可用性
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Error: NVIDIA GPU not detected. Please ensure CUDA is properly installed."
    exit 1
fi

# 设置日志目录
BASE_LOG_DIR="logs/ThreeStageTraining"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${BASE_LOG_DIR}/${TIMESTAMP}_FullThreeStage"

echo "实验目录: $EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"

# 第一阶段训练
echo ""
echo "🚀 开始第一阶段训练..."
echo "训练目标：单独训练第一个策略"
STAGE_1_LOG="${EXPERIMENT_DIR}/stage_1.log"

python humanoidverse/train/train_three_stage.py \
    +simulator=isaacgym \
    +exp=motion_tracking \
    +terrain=terrain_locomotion_plane \
    +obs=motion_tracking/main \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    current_stage=1 \
    stage_iterations=[3000,3000,3000] \
    num_envs=1024 \
    seed=1029 \
    +device=cuda:0 \
    use_wandb=false \
    headless=true \
    robot.motion.motion_file=example/motion_data/Horse-stance_pose.pkl \
    obs.obs_dims.6.dif_local_rigid_body_pos=81 \
    obs.obs_dims.7.local_ref_rigid_body_pos=81 \
    robot.num_key_bodies=2 \
    log_dir="$EXPERIMENT_DIR" 2>&1 | tee "$STAGE_1_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 第一阶段训练失败！查看日志：$STAGE_1_LOG"
    exit 1
fi

echo "✅ 第一阶段训练完成！"

# 查找第一阶段检查点
STAGE_1_CHECKPOINT=$(find "$EXPERIMENT_DIR" -name "*stage_1_checkpoint.pth" | head -1)
if [ -z "$STAGE_1_CHECKPOINT" ]; then
    echo "❌ 找不到第一阶段检查点文件"
    exit 1
fi

echo "第一阶段检查点: $STAGE_1_CHECKPOINT"

# 第二阶段训练
echo ""
echo "🚀 开始第二阶段训练..."
echo "训练目标：冻结第一个策略，训练第二个策略（使用delta奖励）"
STAGE_2_LOG="${EXPERIMENT_DIR}/stage_2.log"

python humanoidverse/train/train_three_stage_horse_stance.py \
    +simulator=isaacgym \
    +exp=motion_tracking \
    +terrain=terrain_locomotion_plane \
    +obs=motion_tracking/main \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/delta \
    current_stage=2 \
    stage_iterations=[3000,3000,3000] \
    stage_1_checkpoint="$STAGE_1_CHECKPOINT" \
    num_envs=1024 \
    seed=1029 \
    +device=cuda:0 \
    use_wandb=false \
    headless=true \
    robot.motion.motion_file=example/motion_data/Horse-stance_pose.pkl \
    obs.obs_dims.6.dif_local_rigid_body_pos=81 \
    obs.obs_dims.7.local_ref_rigid_body_pos=81 \
    robot.num_key_bodies=2 \
    log_dir="$EXPERIMENT_DIR" 2>&1 | tee "$STAGE_2_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 第二阶段训练失败！查看日志：$STAGE_2_LOG"
    exit 1
fi

echo "✅ 第二阶段训练完成！"

# 查找第二阶段检查点
STAGE_2_CHECKPOINT=$(find "$EXPERIMENT_DIR" -name "*stage_2_checkpoint.pth" | head -1)
if [ -z "$STAGE_2_CHECKPOINT" ]; then
    echo "❌ 找不到第二阶段检查点文件"
    exit 1
fi

echo "第二阶段检查点: $STAGE_2_CHECKPOINT"

# 第三阶段训练
echo ""
echo "🚀 开始第三阶段训练..."
echo "训练目标：冻结第二个策略，训练第一个策略（使用原始奖励）"
STAGE_3_LOG="${EXPERIMENT_DIR}/stage_3.log"

python humanoidverse/train/train_three_stage_horse_stance.py \
    +simulator=isaacgym \
    +exp=motion_tracking \
    +terrain=terrain_locomotion_plane \
    +obs=motion_tracking/main \
    +robot=g1/g1_23dof_lock_wrist \
    +domain_rand=main \
    +rewards=motion_tracking/main \
    current_stage=3 \
    stage_iterations=[3000,3000,3000] \
    stage_2_checkpoint="$STAGE_2_CHECKPOINT" \
    num_envs=1024 \
    seed=1029 \
    +device=cuda:0 \
    use_wandb=false \
    headless=true \
    robot.motion.motion_file=example/motion_data/Horse-stance_pose.pkl \
    obs.obs_dims.6.dif_local_rigid_body_pos=81 \
    obs.obs_dims.7.local_ref_rigid_body_pos=81 \
    robot.num_key_bodies=2 \
    log_dir="$EXPERIMENT_DIR" 2>&1 | tee "$STAGE_3_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ 第三阶段训练失败！查看日志：$STAGE_3_LOG"
    exit 1
fi

echo "✅ 第三阶段训练完成！"

# 查找最终检查点
FINAL_CHECKPOINT=$(find "$EXPERIMENT_DIR" -name "*stage_3_final_checkpoint.pth" | head -1)

echo ""
echo "🎉 三阶段训练全部完成！"
echo "==================================================="
echo "实验目录: $EXPERIMENT_DIR"
echo "第一阶段检查点: $STAGE_1_CHECKPOINT"
echo "第二阶段检查点: $STAGE_2_CHECKPOINT"
echo "最终检查点: $FINAL_CHECKPOINT"
echo "==================================================="
echo "训练日志:"
echo "- 第一阶段: $STAGE_1_LOG"
echo "- 第二阶段: $STAGE_2_LOG"
echo "- 第三阶段: $STAGE_3_LOG"
echo "==================================================="

# 生成训练总结
SUMMARY_FILE="${EXPERIMENT_DIR}/training_summary.txt"
cat > "$SUMMARY_FILE" << EOF
三阶段双策略PPO训练总结
====================================

实验时间戳: $TIMESTAMP
实验目录: $EXPERIMENT_DIR

训练阶段:
1. 第一阶段：单独训练第一个策略
   - 检查点: $STAGE_1_CHECKPOINT
   - 日志: $STAGE_1_LOG

2. 第二阶段：冻结第一个策略，训练第二个策略（delta奖励）
   - 检查点: $STAGE_2_CHECKPOINT
   - 日志: $STAGE_2_LOG

3. 第三阶段：冻结第二个策略，训练第一个策略（原始奖励）
   - 检查点: $FINAL_CHECKPOINT
   - 日志: $STAGE_3_LOG

配置参数:
- 环境数量: 1024
- 每阶段迭代: 3000
- 设备: cuda:0
- 运动文件: example/motion_data/Horse-stance_pose.pkl

训练完成时间: $(date)
EOF

echo "训练总结已保存到: $SUMMARY_FILE" 