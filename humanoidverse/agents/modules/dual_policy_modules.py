import torch
import torch.nn as nn
from torch.distributions import Normal
from copy import deepcopy
from typing import Tuple

from .modules import BaseModule
from .ppo_modules import PPOActor


class DualPolicyActor(nn.Module):
    """
    双策略Actor网络
    
    架构：
    - Policy策略：主策略网络，负责基础动作生成
    - Delta策略：增量策略网络，负责动作精细调整
    
    参数：
    - freeze_policy: 是否冻结主策略
    - freeze_delta: 是否冻结增量策略
    """
    
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 num_actions,
                 init_noise_std,
                 freeze_policy=True,
                 freeze_delta=False):
        super(DualPolicyActor, self).__init__()
        
        self.num_actions = num_actions
        self.freeze_policy_flag = freeze_policy
        self.freeze_delta_flag = freeze_delta
        print(f"[DEBUG] DualPolicyActor: policy obs_dim: {obs_dim_dict.get('actor_obs', 0)}")
        
        # 主策略：标准的PPO Actor
        self.policy = PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=deepcopy(module_config_dict),
            num_actions=num_actions,
            init_noise_std=init_noise_std
        )
        
        # 增量策略：接收观测 + 主策略输出
        delta_obs_dim_dict = deepcopy(obs_dim_dict)
        actor_obs_dim = delta_obs_dim_dict.get("actor_obs", 0)
        delta_obs_dim_dict["actor_obs"] = actor_obs_dim + num_actions
        print(f"[DEBUG] DualPolicyActor: delta obs_dim: {delta_obs_dim_dict['actor_obs']}")
        
        self.delta = DeltaPolicyActor(
            obs_dim_dict=delta_obs_dim_dict,
            module_config_dict=deepcopy(module_config_dict),
            num_actions=num_actions,
            init_noise_std=init_noise_std
        )
        
        # 冻结相应的策略
        if self.freeze_policy_flag:
            self._freeze_policy(self.policy)
        if self.freeze_delta_flag:
            self._freeze_policy(self.delta)
            
        self.distribution = None
        Normal.set_default_validate_args = False

    def _freeze_policy(self, policy):
        """冻结策略参数"""
        for param in policy.parameters():
            param.requires_grad = False
        policy.eval()

    def _unfreeze_policy(self, policy):
        """解冻策略参数"""
        for param in policy.parameters():
            param.requires_grad = True
        policy.train()

    def freeze_policy(self):
        """冻结主策略参数"""
        self.freeze_policy_flag = True
        self._freeze_policy(self.policy)
        self.policy.eval()

    def unfreeze_policy(self):
        """解冻主策略参数"""
        self.freeze_policy_flag = False
        self._unfreeze_policy(self.policy)
        self.policy.train()

    def freeze_delta(self):
        """冻结增量策略参数"""
        self.freeze_delta_flag = True
        self._freeze_policy(self.delta)
        self.delta.eval()

    def unfreeze_delta(self):
        """解冻增量策略参数"""
        self.freeze_delta_flag = False
        self._unfreeze_policy(self.delta)
        self.delta.train()

    def reset(self, dones=None):
        self.policy.reset(dones)
        self.delta.reset(dones)

    def act(self, actor_obs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三阶段动作生成：
        - 第一阶段：只用policy(actor_obs)作为action
        - 第二、三阶段：action = policy(actor_obs) + delta(concat(actor_obs, policy(actor_obs)))
        返回:
        - actions: 最终动作
        - policy_output: 主策略的输出
        """
        # 主策略生成初始动作
        policy_output = self.policy.act(actor_obs, **kwargs)
        if getattr(self, 'current_stage', 1) == 1 or getattr(self, 'freeze_delta_flag', True):
            # 第一阶段：只用policy输出
            self.distribution = self.policy.distribution
            return policy_output, policy_output
        else:
            # 第二、三阶段：policy + delta
            combined_obs = torch.cat([actor_obs, policy_output], dim=-1)
            delta_output = self.delta.act(combined_obs, **kwargs)
            final_actions = policy_output + delta_output
            self.distribution = self.delta.distribution
            return final_actions, policy_output

    def act_inference(self, actor_obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三阶段推理动作生成
        """
        policy_output = self.policy.act_inference(actor_obs)
        if getattr(self, 'current_stage', 1) == 1 or getattr(self, 'freeze_delta_flag', True):
            # 第一阶段：只用policy输出
            return policy_output, policy_output
        else:
            combined_obs = torch.cat([actor_obs, policy_output], dim=-1)
            delta_output = self.delta.act_inference(combined_obs)
            final_actions = policy_output + delta_output
            return final_actions, policy_output

    def get_actions_log_prob(self, actions):
        """获取动作的对数概率"""
        # 第一阶段log_prob来自policy，后两阶段来自delta
        if getattr(self, 'current_stage', 1) == 1 or getattr(self, 'freeze_delta_flag', True):
            return self.policy.get_actions_log_prob(actions)
        else:
            return self.delta.get_actions_log_prob(actions)

    @property
    def action_mean(self):
        if getattr(self, 'current_stage', 1) == 1 or getattr(self, 'freeze_delta_flag', True):
            return self.policy.action_mean
        else:
            return self.delta.action_mean

    @property
    def action_std(self):
        if getattr(self, 'current_stage', 1) == 1 or getattr(self, 'freeze_delta_flag', True):
            return self.policy.action_std
        else:
            return self.delta.action_std

    @property
    def entropy(self):
        if getattr(self, 'current_stage', 1) == 1 or getattr(self, 'freeze_delta_flag', True):
            return self.policy.entropy
        else:
            return self.delta.entropy

    def train(self, mode=True):
        """设置训练模式"""
        super().train(mode)
        
        # 根据冻结状态设置各策略的模式
        if not self.freeze_policy_flag:
            self.policy.train(mode)
        else:
            self.policy.eval()
            
        if not self.freeze_delta_flag:
            self.delta.train(mode)
        else:
            self.delta.eval()

    def eval(self):
        """设置评估模式"""
        super().eval()
        self.policy.eval()
        self.delta.eval()

    def switch_training_policy(self):
        """切换训练的策略"""
        self.freeze_policy_flag = not self.freeze_policy_flag
        self.freeze_delta_flag = not self.freeze_delta_flag
        
        if self.freeze_policy_flag:
            self._freeze_policy(self.policy)
            self._unfreeze_policy(self.delta)
        else:
            self._unfreeze_policy(self.policy)
            self._freeze_policy(self.delta)


class DeltaPolicyActor(nn.Module):
    """
    增量策略Actor，接收观测和主策略的输出
    """
    
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 num_actions,
                 init_noise_std):
        super(DeltaPolicyActor, self).__init__()
        
        module_config_dict = self._process_module_config(module_config_dict, num_actions)
        
        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, combined_obs):
        """更新动作分布"""
        mean = self.actor(combined_obs)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, combined_obs, **kwargs):
        """生成动作"""
        self.update_distribution(combined_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """获取动作的对数概率"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, combined_obs):
        """推理模式的动作生成"""
        actions_mean = self.actor(combined_obs)
        return actions_mean


class DualPolicyCritic(nn.Module):
    """
    双策略Critic网络（可选）
    可以为每个策略单独设计价值函数
    """
    
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 use_dual_critics=False):
        super(DualPolicyCritic, self).__init__()
        
        self.use_dual_critics = use_dual_critics
        
        if use_dual_critics:
            # 为每个策略创建单独的critic
            self.first_critic = BaseModule(obs_dim_dict, deepcopy(module_config_dict))
            self.second_critic = BaseModule(obs_dim_dict, deepcopy(module_config_dict))
        else:
            # 共享的critic
            self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    @property
    def critic(self):
        if self.use_dual_critics:
            return self.second_critic  # 默认使用第二个critic
        else:
            return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, critic_obs, use_first_critic=False, **kwargs):
        """评估价值函数"""
        if self.use_dual_critics:
            if use_first_critic:
                value = self.first_critic(critic_obs)
            else:
                value = self.second_critic(critic_obs)
        else:
            value = self.critic_module(critic_obs)
        return value

    def evaluate_both(self, critic_obs, **kwargs):
        """评估两个critic的价值（仅在use_dual_critics=True时有效）"""
        if not self.use_dual_critics:
            raise ValueError("This method is only available when use_dual_critics=True")
        
        first_value = self.first_critic(critic_obs)
        second_value = self.second_critic(critic_obs)
        return first_value, second_value


class PolicyTransitionModule(nn.Module):
    """
    策略转换模块
    用于在第一个策略和第二个策略之间进行信息传递和转换
    """
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=[64, 32],
                 activation='relu'):
        super(PolicyTransitionModule, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AdaptiveDualPolicyActor(DualPolicyActor):
    """
    自适应双策略Actor
    包含策略转换模块，可以更好地处理两个策略之间的信息传递
    """
    
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 num_actions,
                 init_noise_std,
                 freeze_policy=True,
                 freeze_delta=False,
                 use_transition_module=True):
        
        super(AdaptiveDualPolicyActor, self).__init__(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config_dict,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            freeze_policy=freeze_policy,
            freeze_delta=freeze_delta
        )
        
        self.use_transition_module = use_transition_module
        
        if use_transition_module:
            # 创建策略转换模块
            self.transition_module = PolicyTransitionModule(
                input_dim=num_actions,
                output_dim=num_actions,
                hidden_dims=[64, 32]
            )

    def act(self, actor_obs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自适应双策略动作生成
        """
        # 主策略生成初始动作
        policy_output = self.policy.act(actor_obs, **kwargs)
        
        # 如果使用转换模块，对主策略输出进行转换
        if self.use_transition_module:
            transformed_policy_output = self.transition_module(policy_output)
            combined_obs = torch.cat([actor_obs, transformed_policy_output], dim=-1)
        else:
            combined_obs = torch.cat([actor_obs, policy_output], dim=-1)
        
        # 增量策略生成最终动作
        final_actions = self.delta.act(combined_obs, **kwargs)
        
        # 更新分布
        self.distribution = self.delta.distribution
        
        return final_actions, policy_output

    def act_inference(self, actor_obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理模式的动作生成"""
        # 主策略推理
        policy_output = self.policy.act_inference(actor_obs)
        
        # 转换主策略输出
        if self.use_transition_module:
            transformed_policy_output = self.transition_module(policy_output)
            combined_obs = torch.cat([actor_obs, transformed_policy_output], dim=-1)
        else:
            combined_obs = torch.cat([actor_obs, policy_output], dim=-1)
        
        # 增量策略推理
        final_actions = self.delta.act_inference(combined_obs)
        
        return final_actions, policy_output 