#!/usr/bin/env python3
"""
ChipRAG强化学习训练系统
用于训练布局优化智能体
"""

import numpy as np
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import random
import pickle
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入OpenROAD接口
try:
    from .real_openroad_interface_fixed import RealOpenROADInterface
    OPENROAD_AVAILABLE = True
    logger.info("OpenROAD接口导入成功")
except ImportError as e:
    logger.error(f"无法导入OpenROAD接口: {e}")
    logger.error("请确保OpenROAD环境已正确配置")
    OPENROAD_AVAILABLE = False

@dataclass
class LayoutState:
    """布局状态表示"""
    hpwl: float  # 半周长线长
    overflow: float  # 溢出率
    density: float  # 密度
    utilization: float  # 利用率
    iteration: int  # 当前迭代次数
    def_file: str  # 当前DEF文件路径
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组用于神经网络输入"""
        return np.array([
            self.hpwl,
            self.overflow,
            self.density,
            self.utilization,
            self.iteration
        ], dtype=np.float32)

@dataclass
class LayoutAction:
    """布局动作表示"""
    density_target: float  # 目标密度 (0.7-0.95)
    wirelength_weight: float  # 线长权重 (0.1-5.0)
    density_weight: float  # 密度权重 (0.1-5.0)
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([
            self.density_target,
            self.wirelength_weight,
            self.density_weight
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, action_array: np.ndarray) -> 'LayoutAction':
        """从numpy数组创建动作"""
        return cls(
            density_target=float(action_array[0]),
            wirelength_weight=float(action_array[1]),
            density_weight=float(action_array[2])
        )

class LayoutEnvironment:
    """布局优化环境"""
    
    def __init__(self, 
                 work_dir: str,
                 max_iterations: int = 10,
                 target_hpwl: float = 1000000.0,
                 target_overflow: float = 0.1,
                 use_openroad: bool = True):
        """
        Args:
            work_dir: 工作目录
            max_iterations: 最大迭代次数
            target_hpwl: 目标HPWL
            target_overflow: 目标溢出率
            use_openroad: 是否使用真实OpenROAD接口
        """
        self.work_dir = Path(work_dir)
        self.max_iterations = max_iterations
        self.target_hpwl = target_hpwl
        self.target_overflow = target_overflow
        self.current_iteration = 0
        self.current_state = None
        self.episode_data = []
        self.use_openroad = use_openroad
        
        # 确保输出目录存在
        self.output_dir = self.work_dir / "rl_training"
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化OpenROAD接口
        if self.use_openroad:
            if not OPENROAD_AVAILABLE:
                raise RuntimeError("OpenROAD接口不可用，无法进行真实训练")
            try:
                self.openroad_interface = RealOpenROADInterface(str(self.work_dir))
                logger.info("OpenROAD接口初始化成功")
            except Exception as e:
                logger.error(f"OpenROAD接口初始化失败: {e}")
                raise RuntimeError(f"OpenROAD接口初始化失败: {e}")
        
        # 布局策略配置
        self.placement_strategies = [
            {"name": "conservative", "density": 0.7, "wirelength_weight": 1.0, "density_weight": 1.0},
            {"name": "balanced", "density": 0.8, "wirelength_weight": 2.0, "density_weight": 1.5},
            {"name": "aggressive", "density": 0.9, "wirelength_weight": 3.0, "density_weight": 2.0},
            {"name": "wirelength_focused", "density": 0.75, "wirelength_weight": 4.0, "density_weight": 0.5},
            {"name": "density_focused", "density": 0.95, "wirelength_weight": 0.5, "density_weight": 4.0}
        ]
        
    def reset(self) -> LayoutState:
        """重置环境到初始状态"""
        self.current_iteration = 0
        self.episode_data = []
        
        # 运行初始布局（unplace_all后的状态）
        self._run_initial_placement()
        
        # 获取初始状态
        initial_state = self._get_current_state()
        self.current_state = initial_state
        
        logger.info(f"环境重置完成，初始状态: HPWL={initial_state.hpwl:.2f}, "
                   f"溢出率={initial_state.overflow:.4f}")
        
        return initial_state
    
    def step(self, action: LayoutAction) -> Tuple[LayoutState, float, bool, Dict]:
        """执行一步布局优化
        
        Args:
            action: 布局动作
            
        Returns:
            (next_state, reward, done, info)
        """
        self.current_iteration += 1
        
        # 执行布局动作
        success = self._execute_placement_action(action)
        
        if not success:
            # 如果执行失败，给予负奖励
            reward = -100.0
            done = True
            next_state = self.current_state
            info = {"error": "布局执行失败"}
        else:
            # 获取新状态
            next_state = self._get_current_state()
            
            # 计算奖励
            reward = self._calculate_reward(self.current_state, next_state, action)
            
            # 检查是否完成
            done = self.current_iteration >= self.max_iterations
            
            info = {
                "iteration": self.current_iteration,
                "hpwl": next_state.hpwl,
                "overflow": next_state.overflow,
                "density": next_state.density,
                "utilization": next_state.utilization
            }
        
        # 记录episode数据
        self.episode_data.append({
            "iteration": self.current_iteration,
            "action": action.to_array().tolist(),
            "state": next_state.to_array().tolist(),
            "reward": reward,
            "done": done
        })
        
        self.current_state = next_state
        
        return next_state, reward, done, info
    
    def _run_initial_placement(self):
        """运行初始布局（unplace_all后）"""
        if not self.use_openroad:
            raise RuntimeError("模拟模式已被禁用，只允许真实OpenROAD训练")
        
            try:
                # 使用OpenROAD接口进行初始布局
            result = self.openroad_interface.run_placement()
            if result['success']:
                logger.info("初始布局完成")
            else:
                raise RuntimeError(f"初始布局失败: {result.get('stderr', '未知错误')}")
            except Exception as e:
                logger.error(f"初始布局失败: {e}")
            raise RuntimeError(f"初始布局失败: {e}")
    
    def _execute_placement_action(self, action: LayoutAction) -> bool:
        """执行布局动作"""
        if not self.use_openroad:
            raise RuntimeError("模拟模式已被禁用，只允许真实OpenROAD训练")
        
            try:
            # 使用OpenROAD接口执行布局动作
            result = self.openroad_interface.run_placement(
                density_target=action.density_target,
                wirelength_weight=action.wirelength_weight,
                density_weight=action.density_weight
            )
                
            if result['success']:
                    logger.info(f"布局动作执行成功: 迭代{self.current_iteration}")
                    return True
                else:
                logger.error(f"布局动作执行失败: {result.get('stderr', '未知错误')}")
                    return False
                    
            except Exception as e:
                logger.error(f"执行布局动作时出错: {e}")
                return False
    
    def _generate_placement_tcl(self, action: LayoutAction) -> str:
        """生成布局TCL脚本"""
        # 自动检测设计文件
        lef_files = list(self.work_dir.glob("*.lef"))
        verilog_files = list(self.work_dir.glob("*.v"))
        def_files = list(self.work_dir.glob("*.def"))
        
        # 选择文件
        lef_file = lef_files[0] if lef_files else "design.lef"
        verilog_file = verilog_files[0] if verilog_files else "design.v"
        
        # 优先选择floorplan.def作为初始布局
        def_file = None
        for def_path in def_files:
            if "floorplan" in def_path.name.lower():
                def_file = def_path
                break
        if not def_file and def_files:
            def_file = def_files[0]
        
        tcl_script = f"""
# OpenROAD布局脚本 - 迭代{self.current_iteration}
# 动作参数: 密度目标={action.density_target}, 线长权重={action.wirelength_weight}, 密度权重={action.density_weight}

# 读取设计文件
read_lef {lef_file}
read_verilog {verilog_file}
read_def {def_file}

# 初始化设计
init_design

# 设置布局参数
set_placement_density {action.density_target}
set_wirelength_weight {action.wirelength_weight}
set_density_weight {action.density_weight}

# 执行全局布局
global_placement

# 执行详细布局
detailed_placement

# 获取布局指标
set hpwl [get_total_wirelength]
set overflow [get_placement_overflow]
set density [get_placement_density]
set utilization [get_design_utilization]

# 输出指标到日志文件
set log_file [open "placement_metrics.log" w]
puts $log_file "HPWL: $hpwl"
puts $log_file "Overflow: $overflow"
puts $log_file "Density: $density"
puts $log_file "Utilization: $utilization"
close $log_file

# 保存当前布局
write_def "iteration_{self.current_iteration}.def"

# 输出面积报告
report_design_area -outfile "area_report_{self.current_iteration}.rpt"

# 输出线长报告
report_wire_length -outfile "wirelength_report_{self.current_iteration}.rpt"

exit
"""
        return tcl_script
    
    def _get_current_state(self) -> LayoutState:
        """获取当前布局状态"""
        if not self.use_openroad:
            raise RuntimeError("模拟模式已被禁用，只允许真实OpenROAD训练")
        
            try:
            # 使用OpenROAD接口获取当前状态
            result = self.openroad_interface.get_placement_quality({})
                    
            # 从结果中提取指标
            hpwl = result.get('wirelength', 1000000.0)
            overflow = result.get('overflow', 0.2)
            density = result.get('density', 0.8)
            utilization = result.get('utilization', 0.7)
            
            # 如果某些指标缺失，使用默认值
            if hpwl <= 0:
                    hpwl = 1000000.0
            if overflow < 0:
                    overflow = 0.2
            if density <= 0:
                    density = 0.8
            if utilization <= 0:
                    utilization = 0.7
                    
            except Exception as e:
            logger.error(f"获取布局状态时出错: {e}")
                # 使用默认值
                hpwl = 1000000.0
                overflow = 0.2
                density = 0.8
                utilization = 0.7
        
        return LayoutState(
            hpwl=hpwl,
            overflow=overflow,
            density=density,
            utilization=utilization,
            iteration=self.current_iteration,
            def_file=f"placement_result.def"
        )
    
    def _calculate_reward(self, 
                         prev_state: LayoutState, 
                         curr_state: LayoutState, 
                         action: LayoutAction) -> float:
        """计算奖励函数"""
        # 基础奖励：HPWL改善
        hpwl_improvement = (prev_state.hpwl - curr_state.hpwl) / prev_state.hpwl
        hpwl_reward = hpwl_improvement * 100.0
        
        # 溢出率惩罚
        overflow_penalty = -50.0 * curr_state.overflow
        
        # 密度奖励（接近目标密度）
        density_reward = -10.0 * abs(curr_state.density - action.density_target)
        
        # 利用率奖励
        utilization_reward = 20.0 * curr_state.utilization
        
        # 收敛奖励（如果接近目标）
        convergence_reward = 0.0
        if curr_state.hpwl < self.target_hpwl and curr_state.overflow < self.target_overflow:
            convergence_reward = 200.0
        
        total_reward = hpwl_reward + overflow_penalty + density_reward + utilization_reward + convergence_reward
        
        return total_reward
    
    def save_episode_data(self, episode_id: int):
        """保存episode数据"""
        episode_file = self.output_dir / f"episode_{episode_id}.json"
        with open(episode_file, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
        logger.info(f"Episode数据已保存: {episode_file}")

class DQNNetwork(nn.Module):
    """DQN神经网络"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, 
                 state_size: int = 5,
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 gamma: float = 0.95,
                 target_update: int = 10):
        """
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            learning_rate: 学习率
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            memory_size: 经验回放缓冲区大小
            batch_size: 训练批次大小
            gamma: 折扣因子
            target_update: 目标网络更新频率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.update_count = 0
        
        # 检查是否有GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 创建主网络和目标网络
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
    def remember(self, state: np.ndarray, action: np.ndarray, 
                reward: float, next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        if np.random.random() <= self.epsilon:
            # 探索：随机动作
            return self._random_action()
        else:
            # 利用：选择最优动作
            return self._best_action(state)
    
    def _random_action(self) -> np.ndarray:
        """生成随机动作"""
        return np.array([
            random.uniform(0.7, 0.95),  # density_target
            random.uniform(0.1, 5.0),   # wirelength_weight
            random.uniform(0.1, 5.0)    # density_weight
        ])
    
    def _best_action(self, state: np.ndarray) -> np.ndarray:
        """选择最优动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        # 将离散动作索引转换为连续动作
        return self._action_index_to_continuous(action_idx)
    
    def _action_index_to_continuous(self, action_idx: int) -> np.ndarray:
        """将离散动作索引转换为连续动作"""
        # 预定义的连续动作空间
        action_space = [
            [0.7, 1.0, 1.0],   # 保守策略
            [0.8, 2.0, 1.5],   # 平衡策略
            [0.9, 3.0, 2.0],   # 激进策略
            [0.75, 4.0, 0.5],  # 线长优先
            [0.95, 0.5, 4.0],  # 密度优先
            [0.85, 1.5, 1.5],  # 中等策略
            [0.8, 2.5, 1.0],   # 线长中等
            [0.9, 1.0, 3.0],   # 密度中等
        ]
        
        if action_idx < len(action_space):
            return np.array(action_space[action_idx])
        else:
            return self._random_action()
    
    def replay(self, batch_size: int = None):
        """经验回放训练"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
        
        # 随机采样batch
        batch = random.sample(self.memory, batch_size)
        
        # 准备训练数据
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([self._continuous_to_action_index(exp[1]) for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _continuous_to_action_index(self, continuous_action: np.ndarray) -> int:
        """将连续动作转换为离散动作索引"""
        # 预定义的连续动作空间
        action_space = [
            [0.7, 1.0, 1.0],   # 保守策略
            [0.8, 2.0, 1.5],   # 平衡策略
            [0.9, 3.0, 2.0],   # 激进策略
            [0.75, 4.0, 0.5],  # 线长优先
            [0.95, 0.5, 4.0],  # 密度优先
            [0.85, 1.5, 1.5],  # 中等策略
            [0.8, 2.5, 1.0],   # 线长中等
            [0.9, 1.0, 3.0],   # 密度中等
        ]
        
        # 找到最接近的动作
        min_distance = float('inf')
        best_index = 0
        
        for i, action in enumerate(action_space):
            distance = np.linalg.norm(continuous_action - np.array(action))
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        return best_index
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }
        torch.save(model_data, filepath)
        logger.info(f"模型已保存: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        model_data = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(model_data['q_network_state_dict'])
        self.target_network.load_state_dict(model_data['target_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.epsilon = model_data['epsilon']
        self.memory = deque(model_data['memory'], maxlen=self.memory.maxlen)
        logger.info(f"模型已加载: {filepath}")

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, 
                 env: LayoutEnvironment,
                 agent: DQNAgent,
                 episodes: int = 100,
                 max_steps: int = 10):
        """
        Args:
            env: 布局环境
            agent: DQN智能体
            episodes: 训练episode数量
            max_steps: 每个episode最大步数
        """
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps = max_steps
        self.training_history = []
        
    def train(self):
        """开始训练"""
        logger.info(f"开始RL训练，总episodes: {self.episodes}")
        
        for episode in range(self.episodes):
            logger.info(f"开始Episode {episode + 1}/{self.episodes}")
            
            # 重置环境
            state = self.env.reset()
            total_reward = 0
            
            for step in range(self.max_steps):
                # 选择动作
                action_array = self.agent.act(state.to_array())
                action = LayoutAction.from_array(action_array)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.remember(
                    state.to_array(),
                    action_array,
                    reward,
                    next_state.to_array(),
                    done
                )
                
                total_reward += reward
                state = next_state
                
                logger.info(f"Step {step + 1}: 奖励={reward:.2f}, "
                           f"HPWL={info.get('hpwl', 0):.0f}, "
                           f"溢出率={info.get('overflow', 0):.4f}")
                
                if done:
                    break
            
            # 经验回放训练
            self.agent.replay()
            
            # 记录训练历史
            episode_data = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'steps': step + 1,
                'epsilon': self.agent.epsilon,
                'final_hpwl': info.get('hpwl', 0),
                'final_overflow': info.get('overflow', 0)
            }
            self.training_history.append(episode_data)
            
            # 保存episode数据
            self.env.save_episode_data(episode + 1)
            
            logger.info(f"Episode {episode + 1} 完成: "
                       f"总奖励={total_reward:.2f}, "
                       f"步数={step + 1}, "
                       f"探索率={self.agent.epsilon:.4f}")
            
            # 每10个episode保存一次模型
            if (episode + 1) % 10 == 0:
                model_path = self.env.output_dir / f"model_episode_{episode + 1}.pkl"
                self.agent.save(str(model_path))
        
        # 保存最终模型
        final_model_path = self.env.output_dir / "final_model.pkl"
        self.agent.save(str(final_model_path))
        
        # 生成训练报告
        self._generate_training_report()
        
        logger.info("RL训练完成！")
    
    def _generate_training_report(self):
        """生成训练报告"""
        # 创建训练历史DataFrame
        df = pd.DataFrame(self.training_history)
        
        # 保存训练历史
        history_file = self.env.output_dir / "training_history.csv"
        df.to_csv(history_file, index=False)
        
        # 绘制训练曲线
        self._plot_training_curves(df)
        
        logger.info(f"训练报告已生成: {self.env.output_dir}")
    
    def _plot_training_curves(self, df: pd.DataFrame):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总奖励曲线
        axes[0, 0].plot(df['episode'], df['total_reward'])
        axes[0, 0].set_title('Total Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # 探索率曲线
        axes[0, 1].plot(df['episode'], df['epsilon'])
        axes[0, 1].set_title('Epsilon Decay')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        
        # HPWL改善曲线
        axes[1, 0].plot(df['episode'], df['final_hpwl'])
        axes[1, 0].set_title('Final HPWL per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('HPWL')
        
        # 溢出率曲线
        axes[1, 1].plot(df['episode'], df['final_overflow'])
        axes[1, 1].set_title('Final Overflow per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Overflow')
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.env.output_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存: {plot_file}")

def main():
    """主函数"""
    # 配置参数
    work_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    episodes = 50
    max_iterations = 10
    
    # 创建环境
    env = LayoutEnvironment(
        work_dir=work_dir,
        max_iterations=max_iterations,
        target_hpwl=1000000.0,
        target_overflow=0.1
    )
    
    # 创建智能体
    agent = DQNAgent(
        state_size=5,
        action_size=3,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # 创建训练器
    trainer = RLTrainer(
        env=env,
        agent=agent,
        episodes=episodes,
        max_steps=max_iterations
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 