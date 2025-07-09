"""
三阶段双策略PPO训练脚本 - 马步姿态跟踪

训练阶段：
1. 第一阶段：单独训练第一个策略，第一个策略直接输出动作
2. 第二阶段：冻结第一个策略，训练第二个策略，使用delta奖励配置，输出为两个策略的和
3. 第三阶段：冻结第二个策略，训练第一个策略，使用原始奖励配置，输出为两个策略的和
"""

# 首先导入isaacgym以避免导入顺序问题
import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
from loguru import logger

from hydra.utils import instantiate
from humanoidverse.agents.dual_policy_ppo.dual_policy_ppo import DualPolicyPPO
from humanoidverse.utils.helpers import pre_process_config
import torch


@hydra.main(version_base="1.1", config_path="humanoidverse/config", config_name="stage2_config")
def main(cfg: DictConfig) -> None:
    print('DEBUG: config keys:', list(cfg.keys()))
    logger.info("Starting Three-Stage Dual Policy PPO Training for Horse Stance")
    
    # 根据当前阶段选择配置文件
    current_stage = cfg.get('current_stage', 2)
    if current_stage == 1:
        # 第一阶段：使用stage1_config
        stage1_cfg = OmegaConf.load("humanoidverse/config/stage1_config.yaml")
        # 合并命令行覆盖
        cfg = OmegaConf.merge(stage1_cfg, cfg)
    
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 设备配置
    device = cfg.get('device', 'cuda:0')
    torch.cuda.set_device(device)
    
    # 预处理配置
    pre_process_config(cfg)
    
    # 创建环境
    logger.info("Creating environment...")
    env = instantiate(config=cfg.env, device=device)
    
    # 三阶段训练配置
    three_stage_config = {
        'three_stage_training': True,
        'current_stage': cfg.get('current_stage', 1),  # 当前训练阶段
        'stage_iterations': cfg.get('stage_iterations', [5000, 5000, 5000]),  # 每个阶段的迭代次数
        'stage_1_checkpoint': cfg.get('stage_1_checkpoint', None),
        'stage_2_checkpoint': cfg.algo.config.get('stage_2_checkpoint', None), 
        'delta_reward_config': 'humanoidverse/config/rewards/motion_tracking/delta.yaml',
        
        # 阶段特定配置
        'freeze_policy': cfg.get('freeze_policy', False),
        'freeze_delta': cfg.get('freeze_delta', True),
        'pretrained_policy_path': cfg.get('pretrained_policy_path', None),
    }
    
    # 合并三阶段配置到算法配置
    algo_cfg = cfg.algo.copy()
    algo_cfg.update(three_stage_config)
    
    # 创建日志目录
    log_dir = cfg.algo.config.get('log_dir', 'logs/ThreeStage')
    current_stage = three_stage_config['current_stage']
    exp_name = cfg.get('exp', {}).get('name', 'motion_tracking')
    log_dir = f"{log_dir}/Stage{current_stage}_{exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info(f"Log directory: {log_dir}")
    
    # 创建双策略PPO算法
    logger.info("Creating Dual Policy PPO algorithm...")
    algo = DualPolicyPPO(
        env=env,
        config=algo_cfg,
        log_dir=log_dir,
        device=device
    )
    
    # 设置算法
    logger.info("Setting up algorithm...")
    algo.setup()
    
    # 加载预训练策略（如果提供）
    if three_stage_config['pretrained_policy_path']:
        pretrained_path = three_stage_config['pretrained_policy_path']
        if Path(pretrained_path).exists():
            logger.info(f"Loading pretrained policy from {pretrained_path}")
            algo.load_pretrained_policy(pretrained_path)
        else:
            logger.warning(f"Pretrained policy path does not exist: {pretrained_path}")
    
    # 打印训练状态
    training_status = algo.get_training_status()
    logger.info(f"Training Status: {training_status}")
    
    # 开始训练
    logger.info(f"Starting Stage {current_stage} training...")
    try:
        algo.learn()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # 保存当前检查点
        checkpoint_path = Path(log_dir) / f"model_{algo.current_learning_iteration}.pt"
        algo.save(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # 保存错误检查点
        checkpoint_path = Path(log_dir) / f"model_{algo.current_learning_iteration}.pt"
        algo.save(checkpoint_path)
        logger.info(f"Error checkpoint saved to {checkpoint_path}")
        raise
    
    # 打印最终状态
    final_status = algo.get_training_status()
    logger.info(f"Final Training Status: {final_status}")


if __name__ == "__main__":
    main() 