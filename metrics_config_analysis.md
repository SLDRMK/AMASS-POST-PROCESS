# 测试脚本指标和配置分析

## 📊 测试脚本中的指标

### 1. 准确性指标 (Accuracy Metrics)

测试脚本 `sample_eps.py` 中使用的指标定义在 `humanoidverse/measure_traj.py` 的 `eval_accuracy()` 函数中：

#### 核心指标：
- **E_gmpjpe**: 全局平均关节位置误差 (Global Mean Per-Joint Position Error)
- **E_mpjpe**: 平均关节位置误差 (Mean Per-Joint Position Error) - 根节点相对
- **E_dof_mpjpe**: 自由度平均位置误差 (DOF Mean Position Error)
- **E_dof_vel**: 自由度速度误差 (DOF Velocity Error)
- **E_dof_acc**: 自由度加速度误差 (DOF Acceleration Error)
- **E_vel**: 根节点速度误差 (Root Velocity Error)
- **E_root_vel**: 根节点速度误差 (Root Velocity Error)
- **E_acc**: 根节点加速度误差 (Root Acceleration Error)
- **E_root_acc**: 根节点加速度误差 (Root Acceleration Error)
- **E_contact_acc**: 接触加速度误差 (Contact Acceleration Error) - 可选

#### 计算方式：
```python
# 全局MPJPE
gmpjpe = torch.norm(pol['global_translation'] - ref['global_translation'], dim=-1).mean(dim=-1).mean()

# 根节点相对MPJPE
root_relative_position = pol['global_translation'] - pol['global_translation'][..., 0:1, :]
root_relative_position_ref = ref['global_translation'] - ref['global_translation'][..., 0:1, :]
mpjpe = torch.norm(root_relative_position - root_relative_position_ref, dim=-1).mean(dim=-1).mean()

# 自由度MPJPE
dof_mpjpe = torch.norm(pol['dof_pos'] - ref['dof_pos'], dim=-1).mean(dim=-1).mean()
```

### 2. 平滑性指标 (Smoothness Metrics)

定义在 `eval_smoothness()` 函数中：

#### 核心指标：
- **L2_vel**: 速度L2范数
- **L2_acc**: 加速度L2范数
- **L2_jerk**: 加加速度L2范数
- **L2_dof_vel**: 自由度速度L2范数
- **L2_dof_acc**: 自由度加速度L2范数
- **L2_dof_jerk**: 自由度加加速度L2范数
- **L2_ref_vel**: 参考速度L2范数
- **L2_ref_acc**: 参考加速度L2范数
- **L2_ref_jerk**: 参考加加速度L2范数
- **L2_ref_dof_vel**: 参考自由度速度L2范数
- **L2_ref_dof_acc**: 参考自由度加速度L2范数
- **L2_ref_dof_jerk**: 参考自由度加加速度L2范数

#### 计算方式：
```python
# 差分函数
diff_fn = lambda x: (x[1:] - x[:-1]) * delta

# 计算平滑性
pol_vel = diff_fn(pol_pos)
pol_acc = diff_fn(pol_vel)
pol_jerk = diff_fn(pol_acc)

L2_vel = torch.norm(pol_vel, dim=-1).mean(dim=-1).mean()
L2_acc = torch.norm(pol_acc, dim=-1).mean(dim=-1).mean()
L2_jerk = torch.norm(pol_jerk, dim=-1).mean(dim=-1).mean()
```

## 📁 配置存储位置

### 1. 训练配置存储

#### 主要配置文件：
- **训练时**: `logs/{project_name}/{timestamp}-{experiment_name}-{log_task_name}-{robot_type}/config.yaml`
- **评估时**: `logs_eval/{eval_name}/{eval_timestamp}/config.yaml`

#### 配置加载逻辑：
```python
# 在 sample_eps.py 中的配置加载
if override_config.checkpoint is not None:
    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    
    with open(config_path) as file:
        train_config = OmegaConf.load(file)
    
    # 应用评估覆盖配置
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    
    config = OmegaConf.merge(train_config, override_config)
```

### 2. 评估覆盖配置 (eval_overrides)

#### 基础配置 (`humanoidverse/config/base.yaml`):
```yaml
eval_overrides:
  headless: False
  num_envs: 1
  auto_load_latest: False
  use_wandb: False
```

#### 运动跟踪配置 (`humanoidverse/config/env/motion_tracking.yaml`):
```yaml
eval_overrides:
  env:
    config: 
      max_episode_length_s: 100000
```

#### 记录配置 (`humanoidverse/config/opt/record.yaml`):
```yaml
env:
  config:
    save_motion: True
    save_total_steps: 10000
```

### 3. 指标结果存储

#### 存储路径：
```python
# 在 sample_eps.py 中
metric_path = (checkpoint.parent / "metrics" / f"ckpt_{ckpt_num}" / (EpsEvalName+".json"))
```

#### 实际路径示例：
```
logs/MotionTracking/20250624_165713-tennis-motion_tracking-g1_23dof_lock_wrist/
├── config.yaml                    # 训练配置
├── metrics/
│   └── ckpt_119200/
│       ├── sample_eps.json        # 指标结果
│       └── tmp.pkl               # 临时运动数据
└── renderings/
    └── ckpt_119200/              # 渲染结果
```

## 🔧 测试脚本配置参数

### 1. 命令行参数
```bash
python humanoidverse/sample_eps.py \
    checkpoint=/path/to/checkpoint.pt \
    num_episodes=1000 \
    eps_eval_name=sample_eps
```

### 2. 关键配置参数

#### 评估设置：
- **NoEarlyTermination**: `True` - 禁用早期终止
- **NoDR**: `True` - 禁用域随机化
- **NumTotalEps**: 总episode数量
- **EpsEvalName**: 评估名称

#### 环境设置：
```python
config.env.config.save_note = "SampleEps"
config.env.config.enforce_randomize_motion_start_eval = False
config.robot.motion.motion_lib_type = "WJX"
```

#### 保存设置：
```python
config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
config.env.config.ckpt_dir = str(checkpoint.parent)
```

## 📈 指标聚合方式

### 1. 批量处理
```python
# 对每个episode计算指标
for i in range(N):
    # 计算单个episode的指标
    metrics_accuracy = eval_accuracy(traj_data, True)
    metrics_smoothness = eval_smoothness(traj_data, True)
    
    # 转换为毫米单位
    metrics_accuracy = toolz.dicttoolz.valmap(lambda x: x.item() * 1e3, metrics_accuracy)
    metrics_smoothness = toolz.dicttoolz.valmap(lambda x: x.item() * 1e3, metrics_smoothness)
```

### 2. 统计聚合
```python
# 计算均值和标准差
for key in total_result['_raw'][0]['accuracy'].keys():
    key_arr = np.array([total_result['_raw'][i]['accuracy'][key] for i in range(N)])
    aggr_accuracy[key] = {
        'mean': np.mean(key_arr),
        'std': np.std(key_arr),
    }
```

## 🎯 指标评估标准

### 1. 位置精度评级
- 🟢 **优秀**: MPJPE < 50mm, GMPJPE < 100mm
- 🟡 **良好**: MPJPE < 100mm, GMPJPE < 200mm
- 🟠 **一般**: MPJPE < 150mm, GMPJPE < 300mm
- 🔴 **需要改进**: 超出上述范围

### 2. 平滑性评估
- 比较策略输出与参考运动的L2范数
- 策略输出值小于参考值表示更平滑
- 评估速度、加速度、加加速度三个层次

## 📋 总结

1. **指标定义**: 在 `humanoidverse/measure_traj.py` 中定义
2. **配置存储**: 训练配置存储在 `logs/` 目录，评估配置存储在 `logs_eval/` 目录
3. **结果存储**: 指标结果存储在 `metrics/ckpt_{num}/` 目录下的JSON文件
4. **配置覆盖**: 通过 `eval_overrides` 机制实现评估时的配置覆盖
5. **批量评估**: 支持多episode批量评估并计算统计指标 