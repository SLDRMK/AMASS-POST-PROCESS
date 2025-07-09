import torch
import torch.nn as nn
import torch.optim as optim
import time
import statistics
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

from loguru import logger
from hydra.utils import instantiate
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live

from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.modules.dual_policy_modules import DualPolicyActor, DualPolicyCritic
from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.data_utils import RolloutStorage
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback

console = Console()

class DualPolicyPPO(BaseAlgo):
    """
    双策略PPO算法实现
    
    架构：
    - Policy策略：主策略网络，负责基础动作生成
    - Delta策略：增量策略网络，负责动作精细调整
    
    训练模式：
    - freeze_policy=True: 冻结主策略，只训练增量策略
    - freeze_policy=False: 冻结增量策略，只训练主策略
    """
    
    def __init__(self,
                 env,
                 config,
                 log_dir=None,
                 device='cpu',
                 **kwargs):
        
        self.device = device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Book keeping
        self.ep_infos: List[Dict[str, Any]] = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()

        # 三阶段训练配置
        self.three_stage_training = self.config.get('three_stage_training', False)
        self.current_stage = self.config.get('current_stage', 1)  # 1: 单独训练Policy策略, 2: 冻结Policy训练Delta, 3: 冻结Delta训练Policy
        self.stage_iterations = self.config.get('stage_iterations', [10000, 10000, 10000])  # 每个阶段的迭代次数
        self.stage_1_checkpoint = self.config.get('stage_1_checkpoint', None)  # 第一阶段的检查点路径
        self.stage_2_checkpoint = self.config.get('stage_2_checkpoint', None)  # 第二阶段的检查点路径
        self.delta_reward_config = self.config.get('delta_reward_config', None)  # delta奖励配置路径
        
        # 阶段累计迭代计数
        self.stage_1_completed_iterations = 0
        self.stage_2_completed_iterations = 0
        self.stage_3_completed_iterations = 0

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim

        # 双策略特定配置
        self.freeze_policy = self.config.get('freeze_policy', True)
        self.freeze_delta = self.config.get('freeze_delta', False)
        self.pretrained_policy_path = self.config.get('pretrained_policy_path', None)
        
        # 确保不能同时冻结两个策略
        if self.freeze_policy and self.freeze_delta:
            raise ValueError("不能同时冻结两个策略！")
        
        # 如果都不冻结，默认冻结主策略
        if not self.freeze_policy and not self.freeze_delta:
            logger.warning("两个策略都未冻结，默认冻结主策略")
            self.freeze_policy = True

        # 获取算法配置（可能嵌套在config字典中）
        algo_config = self.config.get('config', self.config)

        # Logging related Config
        self.save_interval = algo_config.get('save_interval', 1000)
        self.logging_interval = algo_config.get('logging_interval', 10)
        
        # Training related Config
        self.num_steps_per_env = algo_config.get('num_steps_per_env', 24)
        self.load_optimizer = algo_config.get('load_optimizer', True)
        self.num_learning_iterations = algo_config.get('num_learning_iterations', 30000)
        self.init_at_random_ep_len = algo_config.get('init_at_random_ep_len', True)

        # Algorithm related Config
        self.desired_kl = algo_config.get('desired_kl', 0.01)
        self.schedule = algo_config.get('schedule', 'adaptive')
        self.actor_learning_rate = algo_config.get('actor_learning_rate', 3e-4)
        self.critic_learning_rate = algo_config.get('critic_learning_rate', 3e-4)
        self.clip_param = algo_config.get('clip_param', 0.2)
        self.num_learning_epochs = algo_config.get('num_learning_epochs', 5)
        self.num_mini_batches = algo_config.get('num_mini_batches', 4)
        self.gamma = algo_config.get('gamma', 0.99)
        self.lam = algo_config.get('lam', 0.95)
        self.value_loss_coef = algo_config.get('value_loss_coef', 1.0)
        self.entropy_coef = algo_config.get('entropy_coef', 0.01)
        self.max_grad_norm = algo_config.get('max_grad_norm', 1.0)
        self.use_clipped_value_loss = algo_config.get('use_clipped_value_loss', True)

    def _process_module_config(self, module_config_dict, output_size):
        """处理模块配置，替换特殊的输出维度标识符"""
        import copy
        processed_config = copy.deepcopy(module_config_dict)
        
        # 处理输出维度
        if 'output_dim' in processed_config:
            for idx, output_dim in enumerate(processed_config['output_dim']):
                if output_dim == 'robot_action_dim':
                    processed_config['output_dim'][idx] = self.num_act
                elif output_dim == 'num_rew_fn':
                    processed_config['output_dim'][idx] = output_size
                    
        return processed_config

    def setup(self):
        logger.info("Setting up Dual Policy PPO")
        self._setup_models_and_optimizer()
        logger.info("Setting up Storage")
        self._setup_storage()

    def _setup_models_and_optimizer(self):
        # 获取算法配置和模块配置
        algo_config = self.config.get('config', self.config)
        module_dict = algo_config.get('module_dict', self.config.get('module_dict', {}))
        
        # 创建双策略Actor
        self.dual_actor = DualPolicyActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self._process_module_config(module_dict.get('actor', {}), self.num_act),
            num_actions=self.num_act,
            init_noise_std=algo_config.get('init_noise_std', 1.0),
            freeze_policy=self.freeze_policy,
            freeze_delta=self.freeze_delta
        ).to(self.device)

        # 创建Critic（处理输出维度）
        critic_config = self._process_module_config(module_dict.get('critic', {}), 1)  # critic输出1个值
        self.critic = PPOCritic(
            self.algo_obs_dim_dict,
            critic_config
        ).to(self.device)

        # 设置优化器 - 只为未冻结的策略创建优化器
        if not self.freeze_policy:
            self.policy_optimizer = optim.Adam(
                self.dual_actor.policy.parameters(), 
                lr=self.actor_learning_rate
            )
        else:
            self.policy_optimizer = None
            
        if not self.freeze_delta:
            self.delta_optimizer = optim.Adam(
                self.dual_actor.delta.parameters(), 
                lr=self.actor_learning_rate
            )
        else:
            self.delta_optimizer = None
            
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env)
        
        # Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
        
        # 确定奖励形状
        if self.env.config.use_vec_reward:
            # 向量奖励：使用环境的num_rew_fn
            reward_shape = (self.env.num_rew_fn,)
        else:
            # 标量奖励
            reward_shape = (1,)
        
        # Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('rewards', shape=reward_shape, dtype=torch.float)
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool)
        self.storage.register_key('values', shape=(1,), dtype=torch.float)
        self.storage.register_key('returns', shape=(1,), dtype=torch.float)
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)
        
        # 存储主策略的输出（用于增量策略的输入）
        self.storage.register_key('policy_output', shape=(self.num_act,), dtype=torch.float)

    def _eval_mode(self):
        self.dual_actor.eval()
        self.critic.eval()

    def _train_mode(self):
        self.dual_actor.train()
        self.critic.train()

    def learn(self):
        if self.three_stage_training:
            logger.info(f"Starting three-stage dual policy training - Current Stage: {self.current_stage}")
            self._three_stage_learn()
        else:
            logger.info("Starting dual policy training")
            self._standard_learn()
            
    def _three_stage_learn(self):
        """三阶段训练主逻辑"""
        if self.current_stage == 1:
            self._stage_1_training()
        elif self.current_stage == 2:
            self._stage_2_training()
        elif self.current_stage == 3:
            self._stage_3_training()
        else:
            raise ValueError(f"Invalid training stage: {self.current_stage}")
    
    def _stage_1_training(self):
        """第一阶段：单独训练主策略"""
        logger.info("Stage 1: Training policy only")
        
        # 设置第一阶段的训练配置
        self.freeze_policy = False
        self.freeze_delta = True
        self.dual_actor.freeze_delta()
        self.dual_actor.current_stage = 1  # 同步阶段
        
        # 确保主策略网络处于训练模式且梯度启用
        self.dual_actor.policy.train()
        for param in self.dual_actor.policy.parameters():
            param.requires_grad = True
        
        # 确保增量策略网络处于评估模式且梯度禁用
        self.dual_actor.delta.eval()
        for param in self.dual_actor.delta.parameters():
            param.requires_grad = False
        
        # 重置迭代计数
        self.current_learning_iteration = 0
        
        # 重新配置优化器（只优化主策略）
        self._setup_stage_optimizers(stage=1)
        
        # 执行第一阶段训练
        self._execute_stage_training(
            stage=1,
            iterations=self.stage_iterations[0],
            save_path=self.stage_1_checkpoint or Path(self.log_dir) / "stage_1_checkpoint.pth"
        )
        
    def _stage_2_training(self):
        """第二阶段：冻结主策略，训练增量策略"""
        logger.info("Stage 2: Frozen policy, training delta")
        
        # 加载第一阶段的检查点
        if self.stage_1_checkpoint and Path(self.stage_1_checkpoint).exists():
            logger.info(f"Loading stage 1 checkpoint from {self.stage_1_checkpoint}")
            self.load(self.stage_1_checkpoint)
        
        # 设置第二阶段的训练配置
        self.freeze_policy = True
        self.freeze_delta = False
        self.dual_actor.freeze_policy()
        self.dual_actor.current_stage = 2  # 同步阶段
        
        # 确保主策略网络处于评估模式且梯度禁用
        self.dual_actor.policy.eval()
        for param in self.dual_actor.policy.parameters():
            param.requires_grad = False
        
        # 确保增量策略网络处于训练模式且梯度启用
        self.dual_actor.delta.train()
        for param in self.dual_actor.delta.parameters():
            param.requires_grad = True
        
        # 重置迭代计数
        self.current_learning_iteration = 0
        
        # 切换到delta奖励配置
        if self.delta_reward_config:
            self._switch_reward_config(self.delta_reward_config)
        
        # 重新配置优化器（只优化增量策略）
        self._setup_stage_optimizers(stage=2)
        
        # 执行第二阶段训练
        self._execute_stage_training(
            stage=2,
            iterations=self.stage_iterations[1],
            save_path=self.stage_2_checkpoint or Path(self.log_dir) / "stage_2_checkpoint.pth"
        )
        
    def _stage_3_training(self):
        """第三阶段：冻结增量策略，训练主策略"""
        logger.info("Stage 3: Frozen delta, training policy")
        
        # 加载第二阶段的检查点
        if self.stage_2_checkpoint and Path(self.stage_2_checkpoint).exists():
            logger.info(f"Loading stage 2 checkpoint from {self.stage_2_checkpoint}")
            self.load(self.stage_2_checkpoint)
        
        # 设置第三阶段的训练配置
        self.freeze_policy = False
        self.freeze_delta = True
        self.dual_actor.freeze_delta()
        self.dual_actor.current_stage = 3  # 同步阶段
        
        # 确保主策略网络处于训练模式且梯度启用
        self.dual_actor.policy.train()
        for param in self.dual_actor.policy.parameters():
            param.requires_grad = True
        
        # 确保增量策略网络处于评估模式且梯度禁用
        self.dual_actor.delta.eval()
        for param in self.dual_actor.delta.parameters():
            param.requires_grad = False
        
        # 重置迭代计数
        self.current_learning_iteration = 0
        
        # 切换回原始奖励配置
        self._switch_reward_config(None)  # 切换回原始配置
        
        # 重新配置优化器（只优化主策略）
        self._setup_stage_optimizers(stage=3)
        
        # 执行第三阶段训练
        self._execute_stage_training(
            stage=3,
            iterations=self.stage_iterations[2],
            save_path=Path(self.log_dir) / "stage_3_final_checkpoint.pth"
        )

    def _standard_learn(self):
        """标准双策略训练"""
        obs_dict = self.env.reset_all()
        
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
        
        self.start_time = time.time()
        
        for it in range(self.current_learning_iteration, self.num_learning_iterations):
            self.current_learning_iteration = it
            
            # 记录开始时间
            self.start_time = time.time()
            
            # 数据收集阶段
            obs_dict = self._rollout_step(obs_dict)
            
            # 训练阶段
            self._train_mode()
            train_start_time = time.time()
            loss_dict = self._training_step()
            
            # 计算时间统计
            self.stop_time = time.time()
            self.learn_time = self.stop_time - train_start_time
            # collection_time在_rollout_step中已经计算
            
            # 更新总时间和步数
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += self.collection_time + self.learn_time
            
            # 使用Rich可视化日志记录（和原始PPO相同）
            log_dict = {
                'it': it,
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'num_learning_iterations': self.num_learning_iterations
            }
            self._post_epoch_logging(log_dict)
            
            # 清空episode信息（避免重复记录）
            self.ep_infos.clear()
            
            # 保存检查点
            if it % self.save_interval == 0:
                self.save(Path(self.log_dir) / f"model_{it}.pt")
                
        # 保存最终检查点
        self.save(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt")
        logger.info("Training completed")

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                
                # 双策略动作生成
                policy_state_dict = self._dual_actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                # 存储状态
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                    
                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                
                # 环境步进
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                
                # 处理不同奖励格式
                if self.env.config.use_vec_reward:
                    # 向量奖励：rewards已经是[num_envs, num_reward_functions]格式，不需要unsqueeze
                    rewards_stored = rewards.clone()
                    if 'time_outs' in infos:
                        # values形状为[num_envs, 1]，需要广播到[num_envs, num_reward_functions]
                        time_out_values = self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                        # 将time_out_values广播到所有奖励函数
                        time_out_values = time_out_values.expand(-1, rewards_stored.shape[1])
                        rewards_stored += time_out_values
                else:
                    # 标量奖励：rewards是[num_envs]格式，需要unsqueeze为[num_envs, 1]
                    rewards_stored = rewards.clone().unsqueeze(1)
                    if 'time_outs' in infos:
                        rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                
                # 确保rewards_stored的维度正确（应该是2维）
                assert len(rewards_stored.shape) == 2, f"rewards_stored shape should be 2D, got {rewards_stored.shape}"
                    
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    
                    # 处理不同奖励格式的累积
                    if self.env.config.use_vec_reward:
                        # 向量奖励：对所有奖励函数求和后累积
                        reward_sum = rewards.sum(dim=-1)
                        self.cur_reward_sum += reward_sum
                    else:
                        # 标量奖励
                        self.cur_reward_sum += rewards
                        
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # 计算returns和advantages
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(
                    values=self.storage.query_key('values'), 
                    dones=self.storage.query_key('dones'), 
                    rewards=self.storage.query_key('rewards')
                )
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _dual_actor_rollout_step(self, obs_dict, policy_state_dict):
        """双策略动作生成步骤"""
        actions, policy_output = self._dual_actor_act_step(obs_dict)
        policy_state_dict["actions"] = actions
        policy_state_dict["policy_output"] = policy_output
        
        action_mean = self.dual_actor.action_mean.detach()
        action_sigma = self.dual_actor.action_std.detach()
        actions_log_prob = self.dual_actor.get_actions_log_prob(actions).detach().unsqueeze(1)
        
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        return policy_state_dict

    def _dual_actor_act_step(self, obs_dict):
        """双策略动作生成"""
        return self.dual_actor.act(obs_dict["actor_obs"])
    
    def _critic_eval_step(self, obs_dict):
        return self.critic.evaluate(obs_dict["critic_obs"])

    def _process_env_step(self, rewards, dones, infos):
        self.dual_actor.reset(dones)
        self.critic.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict):
        """计算returns和advantages"""
        last_values = self.critic.evaluate(last_obs_dict["critic_obs"]).detach()
        advantage = 0
        
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        # 处理向量奖励：求和为标量奖励
        if self.env.config.use_vec_reward:
            # rewards形状为[steps, envs, num_reward_functions]，需要对最后一维求和
            rewards = rewards.sum(dim=-1, keepdim=True)
        
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # 计算并标准化advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for policy_state_dict in generator:
            # 将数据移动到设备
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict['Value'] = 0
        loss_dict['Surrogate'] = 0
        loss_dict['Entropy'] = 0
        if not self.freeze_policy:
            loss_dict['Policy_Loss'] = 0
        if not self.freeze_delta:
            loss_dict['Delta_Loss'] = 0
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_dual_policy_ppo(policy_state_dict, loss_dict)
        return loss_dict

    def _update_dual_policy_ppo(self, policy_state_dict, loss_dict):
        """双策略PPO更新"""
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']
        policy_output_batch = policy_state_dict['policy_output']

        # 前向传播
        self._dual_actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.dual_actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.dual_actor.action_mean
        sigma_batch = self.dual_actor.action_std
        entropy_batch = self.dual_actor.entropy

        # KL散度自适应学习率
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                    (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                # 更新优化器学习率
                if self.policy_optimizer is not None:
                    for param_group in self.policy_optimizer.param_groups:
                        param_group['lr'] = self.actor_learning_rate
                if self.delta_optimizer is not None:
                    for param_group in self.delta_optimizer.param_groups:
                        param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        # 根据冻结状态分别计算损失
        if not self.freeze_policy:
            # 主策略未冻结，计算包含entropy的actor_loss
            # 确保从正确的网络获取熵损失
            if self.current_stage == 1 or self.freeze_delta:
                # 阶段1或delta冻结时，从主策略获取熵
                entropy_loss = self.dual_actor.policy.entropy.mean()
            else:
                # 阶段2时，从delta策略获取熵
                entropy_loss = self.dual_actor.delta.entropy.mean()
            actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        else:
            # 主策略冻结，只计算surrogate_loss
            actor_loss = surrogate_loss
            entropy_loss = torch.tensor(0.0, device=self.device)
        
        critic_loss = self.value_loss_coef * value_loss

        # 梯度更新
        if self.policy_optimizer is not None:
            self.policy_optimizer.zero_grad()
        if self.delta_optimizer is not None:
            self.delta_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # 根据冻结状态决定反向传播
        if not self.freeze_policy:
            # 主策略未冻结，可以反向传播actor_loss
            actor_loss.backward()
            if self.policy_optimizer is not None:
                nn.utils.clip_grad_norm_(self.dual_actor.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
        
        if not self.freeze_delta:
            # 增量策略未冻结，可以反向传播delta相关的损失
            if self.delta_optimizer is not None:
                nn.utils.clip_grad_norm_(self.dual_actor.delta.parameters(), self.max_grad_norm)
                self.delta_optimizer.step()
        
        # Critic总是可以训练的
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        
        return loss_dict

    def _post_epoch_logging(self, log_dict, width=80, pad=40):
        """记录训练信息，使用Rich可视化面板"""
        # 更新总时间步和总时间  
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']
        
        if log_dict['it'] % self.logging_interval != 0:  # 检查报告频率
            return

        # 生成计算性能日志
        def generate_computation_log():
            # 计算标准差均值和每秒帧数(FPS)
            mean_std = self.dual_actor.action_std.mean() if hasattr(self.dual_actor, 'action_std') else 0.0
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            
            # 添加阶段信息到标题
            if self.three_stage_training:
                stage_info = f"Stage {log_dict.get('stage', self.current_stage)}"
                stage_status = f"({self.freeze_policy=}, {self.freeze_delta=})"
                str_header = f" \033[1m {stage_info} Dual Policy Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "
                stage_line = f"""{'Training Stage:':>{pad}} {stage_info} {stage_status}\n"""
            else:
                str_header = f" \033[1m Dual Policy Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "
                stage_line = ""
            
            return (f"""{str_header.center(width, ' ')}\n\n"""
                    f"""{stage_line}"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std:>10.4f}\n""")

        # 生成奖励和回合长度日志
        def generate_reward_length_log():
            reward_length_string = ""
            
            if len(log_dict['rewbuffer']) > 0:
                reward_length_string += (f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):>10.4f}\n"""
                                         f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):>10.4f}\n""")
                
                self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
                self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            return reward_length_string

        # 生成环境日志
        def generate_env_log():
            env_log_string = ""
            env_log_dict = self.episode_env_tensors.mean_and_clear()
            env_log_dict = {f"{k}": v for k, v in env_log_dict.items()}
            
            for k, v in env_log_dict.items():
                entry = f"{f'{k}:':>{pad}} {v:>10.4f}"
                env_log_string += f"{entry}\n"
                self.writer.add_scalar('Env/'+k, v, log_dict['it'])
                
            # 记录损失
            for loss_key, loss_value in log_dict['loss_dict'].items():
                self.writer.add_scalar(f'Learn/{loss_key}', loss_value, log_dict['it'])
            
            # 记录学习率
            if self.policy_optimizer is not None:
                self.writer.add_scalar('Learn/policy_learning_rate', 
                                     self.policy_optimizer.param_groups[0]['lr'], log_dict['it'])
            if self.delta_optimizer is not None:
                self.writer.add_scalar('Learn/delta_learning_rate', 
                                     self.delta_optimizer.param_groups[0]['lr'], log_dict['it'])
            self.writer.add_scalar('Learn/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
            
            # 记录动作标准差
            if hasattr(self.dual_actor, 'action_std'):
                self.writer.add_scalar('Learn/mean_noise_std', self.dual_actor.action_std.mean().item(), log_dict['it'])
            
            return env_log_string

        # 生成回合信息日志
        def generate_episode_log():
            ep_string = f"{'-' * width}\n"  # 添加分隔线
            
            if log_dict['ep_infos']:
                # 初始化用于计算均值的字典
                mean_values = {key: 0.0 for key in log_dict['ep_infos'][0].keys()}
                total_episodes = 0

                for ep_info in log_dict['ep_infos']:
                    # 累加值用于计算均值
                    for key in mean_values.keys():
                        if key == 'end_epis_length':
                            # 累加回合长度
                            mean_values[key] += ep_info[key].sum().item()
                            total_episodes += ep_info[key].numel()
                        else:
                            mean_values[key] += (
                                        ep_info[key] / ep_info['end_epis_length'] * self.env.max_episode_length 
                                                ).sum().item()

                # 计算总奖励
                rew_total = 0
                for key, value in mean_values.items():
                    if key.startswith('rew_'):
                        rew_total += value
                        
                mean_values['rew_total'] = rew_total
                
                # 计算每个键的均值
                for key in mean_values.keys():
                    mean_values[key] /= total_episodes
                    self.writer.add_scalar('Env/' + key, mean_values[key], log_dict['it'])
                        
                # 准备日志字符串
                for key, value in mean_values.items():
                    if key == 'end_epis_length': 
                        continue
                    ep_string += f"""{f'{key}:':>{pad}} {value:>10.4f} \n"""
                    
            ep_string += f"Note: reward computed per step\n"
            return ep_string

        # 生成总时间日志
        def generate_total_time_log():
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            eta = self.tot_time / (log_dict['it'] + 1) * (log_dict['num_learning_iterations'] - log_dict['it'])
            
            self.writer.add_scalar('Perf/total_fps', fps, log_dict['it'])
            self.writer.add_scalar('Perf/collection_time', log_dict['collection_time'], log_dict['it'])
            self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
            self.writer.add_scalar('Perf/iter_time', iteration_time, log_dict['it'])
            self.writer.add_scalar('Perf/total_time', self.tot_time, log_dict['it'])
        
            return (f"""{'-' * width}\n"""
                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps:.0f}\n"""
                    f"""{'Collection time:':>{pad}} {log_dict['collection_time']:>10.4f}s\n"""
                    f"""{'Learning time:':>{pad}} {log_dict['learn_time']:>10.4f}s\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:>10.4f}s\n"""
                    f"""{'Total time:':>{pad}} {self.tot_time:>10.4f}s\n"""
                    f"""{'ETA:':>{pad}} {eta:>10.4f}s\n"""
                    f"""{'Time Now:':>{pad}} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n""")

        # 生成所有日志字符串
        log_string = (generate_computation_log() +
                      generate_reward_length_log() +
                      generate_env_log() +
                      generate_episode_log() +
                      generate_total_time_log() +
                      f"Logging Directory: {self.log_dir}")

        # 使用Rich Live更新控制台的特定部分
        with Live(Panel(log_string, title="Dual Policy Training Log"), refresh_per_second=4, console=console):
            pass

    def load_pretrained_policy(self, pretrained_path: str):
        """加载预训练的第一个策略"""
        logger.info(f"Loading pretrained policy from {pretrained_path}")
        loaded_dict = torch.load(pretrained_path, map_location=self.device)
        
        # 假设预训练模型是标准的PPO actor
        if "actor_model_state_dict" in loaded_dict:
            self.dual_actor.first_policy.load_state_dict(loaded_dict["actor_model_state_dict"])
        else:
            # 如果是直接的状态字典
            self.dual_actor.first_policy.load_state_dict(loaded_dict)
            
        logger.info("Pretrained policy loaded successfully")

    def load(self, ckpt_path):
        """加载检查点"""
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            
            # 检查是否是第一阶段检查点格式（标准PPO格式）
            if "actor_model_state_dict" in loaded_dict and "critic_model_state_dict" in loaded_dict:
                logger.info("Loading stage 1 checkpoint format (standard PPO)")
                # 第一阶段检查点：将actor加载到policy
                self.dual_actor.policy.load_state_dict(loaded_dict["actor_model_state_dict"])
                
                # 只在第一阶段训练时加载critic，其他阶段不加载critic以避免维度不匹配
                if self.current_stage == 1:
                    logger.info("Loading critic for stage 1 training")
                    self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
                else:
                    logger.info(f"Stage {self.current_stage}: Skipping critic loading to avoid dimension mismatch")
                
                if self.load_optimizer:
                    if "actor_optimizer_state_dict" in loaded_dict:
                        # 如果有policy_optimizer，加载到其中
                        if self.policy_optimizer is not None:
                            self.policy_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                    # 只在第一阶段加载critic优化器
                    if self.current_stage == 1 and "critic_optimizer_state_dict" in loaded_dict:
                        self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            else:
                # 双策略PPO格式
                logger.info("Loading dual policy PPO checkpoint format")
                self.dual_actor.load_state_dict(loaded_dict["dual_actor_model_state_dict"])
                
                # 只在第一阶段训练时加载critic
                if self.current_stage == 1:
                    logger.info("Loading critic for stage 1 training")
                    self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
                else:
                    logger.info(f"Stage {self.current_stage}: Skipping critic loading to avoid dimension mismatch")
                
                if self.load_optimizer:
                    if self.policy_optimizer is not None and "policy_optimizer_state_dict" in loaded_dict:
                        self.policy_optimizer.load_state_dict(loaded_dict["policy_optimizer_state_dict"])
                    if self.delta_optimizer is not None and "delta_optimizer_state_dict" in loaded_dict:
                        self.delta_optimizer.load_state_dict(loaded_dict["delta_optimizer_state_dict"])
                    # 只在第一阶段加载critic优化器
                    if self.current_stage == 1 and "critic_optimizer_state_dict" in loaded_dict:
                        self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                    
            self.current_learning_iteration = loaded_dict.get("iter", 0)
            return loaded_dict.get("infos", None)

    def save(self, path, infos=None):
        """保存检查点"""
        logger.info(f"Saving checkpoint to {path}")
        save_dict = {
            'dual_actor_model_state_dict': self.dual_actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        
        if self.policy_optimizer is not None:
            save_dict['policy_optimizer_state_dict'] = self.policy_optimizer.state_dict()
        if self.delta_optimizer is not None:
            save_dict['delta_optimizer_state_dict'] = self.delta_optimizer.state_dict()
        save_dict['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        
        torch.save(save_dict, path)

    @property
    def inference_model(self):
        return {
            "dual_actor": self.dual_actor,
            "critic": self.critic
        }

    @torch.no_grad()
    def evaluate_policy(self):
        """策略评估"""
        self._eval_mode()
        self.env.set_is_evaluating()
        obs_dict = self.env.reset_all()
        
        step = 0
        while True:
            if self.three_stage_training and self.current_stage == 1:
                # 第一阶段：只使用主策略
                actions = self.dual_actor.policy.act_inference(obs_dict['actor_obs'])
            else:
                # 其他阶段：使用双策略输出
                actions, _ = self.dual_actor.act_inference(obs_dict['actor_obs'])
            
            actor_state = {"actions": actions}
            obs_dict, rewards, dones, infos = self.env.step(actor_state)
            step += 1
            
            # 评估终止条件
            if step > 1000:
                break

    def _setup_stage_optimizers(self, stage):
        """为不同阶段设置优化器"""
        if stage == 1 or stage == 3:
            # 第一和第三阶段：只优化主策略
            self.policy_optimizer = optim.Adam(
                self.dual_actor.policy.parameters(), 
                lr=self.actor_learning_rate
            )
            self.delta_optimizer = None
        elif stage == 2:
            # 第二阶段：只优化增量策略
            self.policy_optimizer = None
            self.delta_optimizer = optim.Adam(
                self.dual_actor.delta.parameters(), 
                lr=self.actor_learning_rate
            )
        
        # 所有阶段都需要critic optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        
        logger.info(f"Stage {stage} optimizers configured")

    def _switch_reward_config(self, reward_config_path):
        """切换奖励配置"""
        if reward_config_path:
            logger.info(f"Switching to reward config: {reward_config_path}")
            # 这里需要根据具体的环境实现来重新配置奖励
            # 示例实现（需要根据实际环境API调整）
            try:
                if hasattr(self.env, 'update_reward_config'):
                    self.env.update_reward_config(reward_config_path)
                    logger.info("Reward config updated successfully")
                else:
                    logger.warning("Environment does not support dynamic reward config switching")
            except Exception as e:
                logger.error(f"Failed to switch reward config: {e}")
        else:
            logger.info("Switching back to original reward config")

    def _execute_stage_training(self, stage, iterations, save_path):
        """执行特定阶段的训练"""
        logger.info(f"Executing stage {stage} training for {iterations} iterations")
        
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
        
        self.start_time = time.time()
        start_iteration = self.current_learning_iteration
        end_iteration = start_iteration + iterations
        
        for it in range(start_iteration, end_iteration):
            self.current_learning_iteration = it
            
            # 记录开始时间
            self.start_time = time.time()
            
            # 数据收集阶段
            obs_dict = self._rollout_step(obs_dict)
            
            # 训练阶段
            self._train_mode()
            train_start_time = time.time()
            loss_dict = self._training_step()
            
            # 计算时间统计
            self.stop_time = time.time()
            self.learn_time = self.stop_time - train_start_time
            
            # 更新总时间和步数
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += self.collection_time + self.learn_time
            
            # 使用Rich可视化日志记录
            log_dict = {
                'it': it,
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'num_learning_iterations': iterations,
                'stage': stage  # 添加阶段信息
            }
            self._post_epoch_logging(log_dict)
            
            # 清空episode信息
            self.ep_infos.clear()
            
            # 保存检查点 - 使用阶段内迭代计数
            stage_iteration = it - start_iteration
            if stage_iteration % self.save_interval == 0:
                checkpoint_path = Path(self.log_dir) / f"model_{stage_iteration}.pt"
                self.save(checkpoint_path)
        
        # 保存阶段最终检查点
        final_stage_iteration = iterations
        final_checkpoint_path = Path(self.log_dir) / f"model_{final_stage_iteration}.pt"
        self.save(final_checkpoint_path)
        logger.info(f"Stage {stage} training completed. Checkpoint saved to {save_path}")
        
        # 更新阶段完成迭代计数
        if stage == 1:
            self.stage_1_completed_iterations = iterations
        elif stage == 2:
            self.stage_2_completed_iterations = iterations
        elif stage == 3:
            self.stage_3_completed_iterations = iterations

    def get_training_status(self):
        """获取三阶段训练状态"""
        if not self.three_stage_training:
            return {"message": "Standard dual policy training"}
        
        return {
            "three_stage_training": True,
            "current_stage": self.current_stage,
            "stage_iterations": self.stage_iterations,
            "completed_iterations": {
                "stage_1": self.stage_1_completed_iterations,
                "stage_2": self.stage_2_completed_iterations,
                "stage_3": self.stage_3_completed_iterations
            },
            "total_iterations": self.current_learning_iteration
        }

    def set_training_stage(self, stage):
        """设置训练阶段"""
        if stage not in [1, 2, 3]:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3")
        
        self.current_stage = stage
        logger.info(f"Training stage set to {stage}")

    def freeze_policy(self):
        """冻结主策略"""
        self.freeze_policy = True
        self.dual_actor._freeze_policy(self.dual_actor.policy)
        logger.info("Policy frozen")

    def freeze_delta(self):
        """冻结增量策略"""
        self.freeze_delta = True
        self.dual_actor._freeze_policy(self.dual_actor.delta)
        logger.info("Delta frozen")

    def unfreeze_policy(self):
        """解冻主策略"""
        self.freeze_policy = False
        self.dual_actor._unfreeze_policy(self.dual_actor.policy)
        logger.info("Policy unfrozen")

    def unfreeze_delta(self):
        """解冻增量策略"""
        self.freeze_delta = False
        self.dual_actor._unfreeze_policy(self.dual_actor.delta)
        logger.info("Delta unfrozen") 