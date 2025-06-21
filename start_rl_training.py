#!/usr/bin/env python3
"""
强化学习训练启动脚本
使用现有训练数据开始Chip-D-RAG系统的强化学习训练
"""

import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# This is a new dependency, let's add it to the file
from modules.parsers.eda_parser import parse_def, parse_verilog

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 神经网络策略 ---

# 动作空间：k值从3到15
ACTION_SPACE = list(range(3, 16))
# 状态空间维度 (待定)
# 设计类型(5) + 技术节点(4) + 复杂度(1) + 约束数(1) = 11
STATE_DIM = 5  

# 定义超参数
DESIGN_TYPES = ['risc_v', 'dsp', 'memory', 'accelerator', 'controller']
TECH_NODES = ['14nm', '28nm', '40nm', '65nm']

# --- Real Design Environment ---

class RealDesignEnvironment:
    """
    管理与真实设计的交互，提取其状态并模拟EDA工具的运行。
    """
    def __init__(self, design_dir: str):
        self.design_dir = Path(design_dir)
        design_name = self.design_dir.name

        self.netlist_path = next(self.design_dir.glob('*.v'), None)
        self.def_path = next(self.design_dir.glob('*.def'), None)

        if not self.netlist_path:
            raise FileNotFoundError(f"在 {design_dir} 中未找到Verilog网表文件 (.v)")
        if not self.def_path:
            raise FileNotFoundError(f"在 {design_dir} 中未找到DEF文件 (.def)")
        
        logger.info(f"--- 正在为设计 '{design_name}' 初始化真实设计环境 ---")
        self.verilog_metrics = parse_verilog(str(self.netlist_path))
        self.def_metrics = parse_def(str(self.def_path))
        logger.info("成功解析Verilog和DEF文件。")

        self.state = self._build_state_vector()

    def _build_state_vector(self) -> torch.Tensor:
        """从解析的指标构建状态张量。"""
        global STATE_DIM
        die_area = self.def_metrics.get('die_area_microns', (0, 0))
        
        # 特征归一化
        features = [
            die_area[0] / 1000.0,
            die_area[1] / 1000.0,
            self.def_metrics.get('num_components', 0) / 1e6,
            self.verilog_metrics.get('num_instances', 0) / 1e6,
            len(self.def_metrics.get('components_summary', {})) / 1000.0,
        ]
        
        if STATE_DIM != len(features):
            logger.info(f"状态维度根据真实指标更新为 {len(features)}。")
            STATE_DIM = len(features)
        return torch.FloatTensor(features).unsqueeze(0)

    def get_state(self) -> torch.Tensor:
        return self.state

    def step(self, action_k: int) -> Tuple[torch.Tensor, float, bool]:
        """模拟EDA运行并计算奖励。"""
        logger.debug(f"环境步骤: k={action_k}。正在模拟EDA运行...")
        ppa_results = self._simulate_eda_run(action_k)
        reward = self._calculate_reward(ppa_results)
        logger.debug(f"模拟PPA得分: {ppa_results['overall_score']:.3f}, 奖励: {reward:.3f}")
        # 在这个模拟版本中，每个episode只有一步
        return self.state, reward, True

    def _simulate_eda_run(self, k_value: int) -> Dict[str, Any]:
        """
        基于k值和设计复杂度的PPA模拟。
        更真实的模拟：最优k值依赖于设计的复杂性。
        """
        num_components = self.def_metrics.get('num_components', 100000)
        complexity = np.clip(num_components / 200000.0, 0.5, 2.0) # 归一化并限制复杂度因子
        
        # 最优k值随复杂度增加而增加
        optimal_k = 4 + 6 * complexity 
        
        # 高斯惩罚项，惩罚偏离最优k值的选择
        k_penalty = np.exp(-((k_value - optimal_k)**2) / (2 * (4 / complexity)**2))
        
        base_quality = 0.85 - (complexity - 1) * 0.1
        
        # 性能（时序）: k值越大通常对时序越有利
        timing = min(0.98, base_quality * k_penalty + 0.1 * (k_value / 15.0))
        # 功耗: k值越大，检索的知识越多，可能导致更复杂的解决方案，功耗微增
        power = max(0.3, (base_quality * k_penalty) / (1 + 0.10 * (k_value / 15.0) * complexity))
        # 拥塞: k值过大可能引入不必要的复杂性，导致拥塞
        congestion = max(0.3, (base_quality * k_penalty) / (1 + 0.15 * (k_value / 15.0) * complexity))
        
        # WNS (Worst Negative Slack) in ns. Smaller is better.
        wns = (1 - timing) * -5.0 
        # Congestion score. Percentage. Smaller is better.
        congestion_score = (1 - congestion) * 15.0
        # Power in mW. Smaller is better.
        power_mw = 50 + (1 - power) * 50

        return {
            'overall_score': (timing*0.5 + power*0.25 + congestion*0.25), # 调整权重
            'wns': wns,
            'congestion': congestion_score,
            'power': power_mw,
            'k_value': k_value,
        }

    def _calculate_reward(self, ppa: Dict[str, Any]) -> float:
        """更精细的奖励计算函数。"""
        # 时序奖励: WNS为正（>0）时给予高奖励
        timing_r = np.clip(ppa['wns'] / 2.0, -1, 1) if ppa['wns'] > 0 else ppa['wns'] / 0.5
        # 拥塞奖励: 惩罚高拥塞
        cong_r = 1 - np.clip(ppa['congestion'] / 10.0, 0, 1)
        # 功耗奖励: 惩罚高功耗
        power_r = 1 - np.clip((ppa['power'] - 50) / 50, 0, 1)

        # 最终奖励是加权和
        reward = float(timing_r * 0.5 + cong_r * 0.25 + power_r * 0.25)
        
        # 对选择极值的k进行轻微惩罚，鼓励探索
        if ppa['k_value'] <= 3 or ppa['k_value'] >= 14:
            reward -= 0.05
            
        return reward

# --- RL Agent and Network ---

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络，包含共享层以及独立的actor和critic头。
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor头输出动作概率
        self.actor_head = nn.Linear(128, action_dim)
        # Critic头输出状态价值
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared_layer(state)
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        state_value = self.critic_head(x)
        return action_probs, state_value

def update_actor_critic(optimizer, log_prob, state_value, reward_tensor):
    """
    更新Actor-Critic网络。
    advantage = R - V(s)  (因为是单步episode, target_value = R)
    """
    advantage = reward_tensor - state_value
    
    # Actor loss (policy gradient)
    actor_loss = -(log_prob * advantage.detach())
    
    # Critic loss (mean squared error)
    critic_loss = F.mse_loss(state_value, reward_tensor)
    
    # 总损失
    loss = actor_loss + 0.5 * critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    optimizer.step()
    
    return loss.item(), actor_loss.item(), critic_loss.item()

# --- Main Training Logic ---

def start_real_design_training(design_dir: str, config: Dict[str, Any]) -> None:
    """针对单个真实设计的主训练循环。"""
    design_name = Path(design_dir).name
    logger.info(f"--- 开始对设计进行RL训练: {design_name} ---")
    
    # --- 定义相对于此脚本的输出目录 ---
    script_dir = Path(__file__).parent
    output_dir = script_dir / "models" / "rl_agent"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"模型和报告将保存到: {output_dir}")

    try:
        env = RealDesignEnvironment(design_dir)
    except FileNotFoundError as e:
        logger.error(f"初始化环境失败: {e}")
        return

    # 每次运行都重新创建网络和优化器
    ac_network = ActorCriticNetwork(STATE_DIM, len(ACTION_SPACE))
    optimizer = optim.Adam(ac_network.parameters(), lr=config.get('learning_rate', 0.001))

    num_episodes = config.get('num_episodes', 300)
    epsilon_start = config.get('epsilon_start', 0.9)
    epsilon_end = config.get('epsilon_end', 0.05)
    epsilon_decay = config.get('epsilon_decay', 0.99)
    epsilon = epsilon_start
    
    history = []
    for episode in range(num_episodes):
        state = env.get_state()
        
        is_exploratory = random.random() < epsilon
        
        if is_exploratory:
            action_idx = random.choice(range(len(ACTION_SPACE)))
            k_value = ACTION_SPACE[action_idx]
        else:
            with torch.no_grad():
                probs, _ = ac_network(state)
            m = Categorical(probs)
            action = m.sample()
            action_idx = action.item()
            k_value = ACTION_SPACE[action_idx]
        
        _, reward, _ = env.step(k_value)
        
        # 重新计算带梯度的输出以更新网络
        probs, state_value = ac_network(state)
        m = Categorical(probs)
        action_tensor = torch.tensor([action_idx])
        log_prob = m.log_prob(action_tensor)

        reward_tensor = torch.tensor([[reward]], dtype=torch.float)
        
        loss, actor_loss, critic_loss = update_actor_critic(optimizer, log_prob, state_value, reward_tensor)

        history.append({
            'episode': episode, 
            'reward': reward, 
            'k_value': k_value, 
            'epsilon': epsilon,
            'loss': loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
        })
        
        # Epsilon衰减
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode > 0 and episode % 50 == 0:
            avg_reward = np.mean([h['reward'] for h in history[-50:]])
            avg_loss = np.mean([h['loss'] for h in history[-50:]])
            logger.info(f"设计 {design_name} | Ep {episode}: 平均奖励: {avg_reward:.3f}, 平均损失: {avg_loss:.3f}, Epsilon: {epsilon:.3f}")

    logger.info(f"--- {design_name} 的真实设计训练完成 ---")
    
    # --- 保存模型和报告 ---
    # 查找最佳k值
    best_run = max(history, key=lambda x: x['reward'])
    logger.info(f"设计 {design_name} 的最佳运行: Episode {best_run['episode']} 使用 k={best_run['k_value']} 获得奖励 {best_run['reward']:.3f}")

    # 保存模型
    model_path = output_dir / f"{design_name}_ac_model.pt"
    torch.save(ac_network.state_dict(), model_path)
    logger.info(f"模型已保存到: {model_path}")

    # 保存训练报告
    report_path = output_dir / f"{design_name}_training_report.json"
    with open(report_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"训练报告已保存到: {report_path}")

# --- Main Execution ---

def main():
    """
    主函数：加载配置并为多个设计启动训练。
    """
    logger.info("=================================================")
    logger.info("=== Chip-RAG 强化学习训练脚本启动 ===")
    logger.info("=================================================")

    # 定义要训练的设计列表
    # 注意：请确保这些目录存在于 'chipdrag/data/designs/ispd_2015_contest_benchmark/' 下
    # 示例: chipdrag/data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/
    base_design_path = Path(__file__).parent / "data" / "designs" / "ispd_2015_contest_benchmark"
    
    designs_to_train = [
        "mgc_des_perf_1",
        "mgc_des_perf_a",
        "mgc_fft_1",
        "mgc_pci_bridge32_a"
    ]
    
    # 检查设计是否存在
    available_designs = []
    for design in designs_to_train:
        design_path = base_design_path / design
        if design_path.exists():
            available_designs.append(str(design_path))
        else:
            logger.warning(f"设计目录未找到，跳过: {design_path}")

    if not available_designs:
        logger.error("没有找到可用的设计目录进行训练。请检查 'chipdrag/data/designs/ispd_2015_contest_benchmark/' 目录结构。")
        return

    config = {
        "learning_rate": 0.0005,
        "num_episodes": 300,
        "epsilon_start": 0.95,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.985,
    }

    logger.info(f"将要训练 {len(available_designs)} 个设计: {', '.join([Path(d).name for d in available_designs])}")

    for design_dir in available_designs:
        start_real_design_training(design_dir, config)

if __name__ == '__main__':
    main() 