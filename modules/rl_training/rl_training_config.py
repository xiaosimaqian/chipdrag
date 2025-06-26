#!/usr/bin/env python3
"""
ChipRAG强化学习训练配置
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import json

@dataclass
class RLTrainingConfig:
    """RL训练配置"""
    
    # 环境配置
    work_dir: str = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    max_iterations: int = 10
    target_hpwl: float = 1000000.0
    target_overflow: float = 0.1
    use_openroad: bool = True
    
    # 智能体配置
    state_size: int = 5
    action_size: int = 8  # 8种预定义策略
    learning_rate: float = 0.001
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.95
    target_update: int = 10
    
    # 训练配置
    episodes: int = 100
    max_steps: int = 10
    save_interval: int = 10
    
    # 奖励函数权重
    reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                "hpwl_improvement": 100.0,
                "overflow_penalty": -50.0,
                "density_reward": -10.0,
                "utilization_reward": 20.0,
                "convergence_reward": 200.0
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "work_dir": self.work_dir,
            "max_iterations": self.max_iterations,
            "target_hpwl": self.target_hpwl,
            "target_overflow": self.target_overflow,
            "use_openroad": self.use_openroad,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "target_update": self.target_update,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "save_interval": self.save_interval,
            "reward_weights": self.reward_weights
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLTrainingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'RLTrainingConfig':
        """从文件加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# 预定义配置
def get_default_config() -> RLTrainingConfig:
    """获取默认配置"""
    return RLTrainingConfig()

def get_fast_config() -> RLTrainingConfig:
    """获取快速训练配置（用于测试）"""
    config = RLTrainingConfig()
    config.episodes = 10
    config.max_iterations = 5
    config.use_openroad = False  # 使用模拟模式
    return config

def get_full_config() -> RLTrainingConfig:
    """获取完整训练配置"""
    config = RLTrainingConfig()
    config.episodes = 500
    config.max_iterations = 15
    config.memory_size = 50000
    config.batch_size = 64
    return config

def get_benchmark_configs() -> Dict[str, RLTrainingConfig]:
    """获取不同benchmark的配置"""
    configs = {}
    
    # mgc_des_perf_1
    configs["mgc_des_perf_1"] = RLTrainingConfig(
        work_dir="data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1",
        target_hpwl=800000.0,
        target_overflow=0.08
    )
    
    # mgc_fft_1
    configs["mgc_fft_1"] = RLTrainingConfig(
        work_dir="data/designs/ispd_2015_contest_benchmark/mgc_fft_1",
        target_hpwl=1200000.0,
        target_overflow=0.12
    )
    
    # mgc_pci_bridge32_a
    configs["mgc_pci_bridge32_a"] = RLTrainingConfig(
        work_dir="data/designs/ispd_2015_contest_benchmark/mgc_pci_bridge32_a",
        target_hpwl=1500000.0,
        target_overflow=0.15
    )
    
    return configs 