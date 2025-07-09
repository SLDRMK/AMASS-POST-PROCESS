# 三阶段双策略PPO训练

本项目实现了基于Isaac Gym的三阶段双策略PPO强化学习训练，用于人形机器人的运动跟踪任务。

## 文件结构

```
PBHC-main/
├── humanoidverse/
│   ├── train_three_stage.py                 # 主训练脚本
│   ├── config/
│   │   ├── stage1_config.yaml              # 第一阶段配置
│   │   ├── stage2_config.yaml              # 第二阶段配置
│   │   └── stage3_config.yaml              # 第三阶段配置
│   └── agents/dual_policy_ppo/
│       └── dual_policy_ppo.py              # 双策略PPO算法实现
└── scripts/three_stage_training/
    ├── run_stage_1_training.sh             # 第一阶段训练脚本
    ├── run_stage_2_training.sh             # 第二阶段训练脚本
    ├── run_stage_3_training.sh             # 第三阶段训练脚本
    ├── run_full_three_stage_training.sh    # 完整三阶段训练脚本
    └── README.md                           # 本文件
```

## 训练阶段

### 第一阶段：主策略训练
- **目标**：单独训练主策略网络
- **冻结策略**：无
- **奖励函数**：原始运动跟踪奖励
- **输出**：`logs/ThreeStage/Stage1_motion_tracking/model_*.pt`

### 第二阶段：增量策略训练
- **目标**：冻结主策略，训练增量策略网络
- **冻结策略**：主策略
- **奖励函数**：Delta奖励（增量学习）
- **输入**：第一阶段检查点
- **输出**：`logs/ThreeStage/Stage2_motion_tracking/model_*.pt`

### 第三阶段：主策略重新训练
- **目标**：冻结增量策略，重新训练主策略
- **冻结策略**：增量策略
- **奖励函数**：原始运动跟踪奖励
- **输入**：第一、二阶段检查点
- **输出**：`logs/ThreeStage/Stage3_motion_tracking/model_*.pt`

## 使用方法

### 1. 单独运行各阶段

```bash
# 第一阶段
./scripts/three_stage_training/run_stage_1_training.sh

# 第二阶段
./scripts/three_stage_training/run_stage_2_training.sh

# 第三阶段
./scripts/three_stage_training/run_stage_3_training.sh
```

### 2. 运行完整三阶段训练

```bash
./scripts/three_stage_training/run_full_three_stage_training.sh
```

### 3. 直接使用Python脚本

```bash
# 第一阶段
python humanoidverse/train_three_stage.py \
    --stage 1 \
    --config humanoidverse/config/stage1_config.yaml \
    --log_dir logs/ThreeStage/Stage1_motion_tracking

# 第二阶段
python humanoidverse/train_three_stage.py \
    --stage 2 \
    --config humanoidverse/config/stage2_config.yaml \
    --stage_1_checkpoint example/pretrained_horse_stance_pose/model_50000.pt \
    --log_dir logs/ThreeStage/Stage2_motion_tracking

# 第三阶段
python humanoidverse/train_three_stage.py \
    --stage 3 \
    --config humanoidverse/config/stage3_config.yaml \
    --stage_1_checkpoint example/pretrained_horse_stance_pose/model_50000.pt \
    --stage_2_checkpoint logs/ThreeStage/Stage2_motion_tracking/stage_2_checkpoint_00050100.pth \
    --log_dir logs/ThreeStage/Stage3_motion_tracking
```

## 配置说明

### 主要参数

- `stage`: 训练阶段 (1, 2, 3)
- `config`: 配置文件路径
- `stage_1_checkpoint`: 第一阶段检查点路径
- `stage_2_checkpoint`: 第二阶段检查点路径
- `log_dir`: 日志输出目录

### 训练参数

- `num_envs`: 并行环境数量 (默认: 1024)
- `stage_iterations`: 每阶段迭代次数 (默认: [2000, 2000, 2000])
- `save_interval`: 检查点保存间隔 (默认: 20)
- `device`: 训练设备 (默认: cuda:0)

## 检查点命名

修复后的检查点命名逻辑：
- **每个阶段从0开始计数**
- **格式**: `model_{stage_iteration}.pt`
- **示例**:
  - 阶段1: `model_0.pt`, `model_20.pt`, `model_40.pt`...
  - 阶段2: `model_0.pt`, `model_20.pt`, `model_40.pt`...
  - 阶段3: `model_0.pt`, `model_20.pt`, `model_40.pt`...

## 注意事项

1. **依赖检查**：确保Isaac Gym环境正确安装
2. **GPU要求**：需要NVIDIA GPU和CUDA支持
3. **内存要求**：建议至少16GB GPU内存
4. **检查点路径**：确保检查点文件路径正确
5. **日志目录**：脚本会自动创建日志目录

## 故障排除

### 常见问题

1. **检查点维度不匹配**：确保使用正确的检查点文件
2. **GPU内存不足**：减少`num_envs`参数
3. **配置文件错误**：检查YAML配置文件语法
4. **路径问题**：确保所有文件路径正确

### 调试建议

1. 先运行单个阶段测试
2. 检查日志输出
3. 验证检查点文件完整性
4. 确认环境配置正确

## 扩展

- 修改配置文件可以调整训练参数
- 添加新的奖励函数
- 支持不同的运动数据
- 集成其他强化学习算法 