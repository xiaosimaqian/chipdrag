#!/usr/bin/env python3
"""
增强版强化学习训练系统
支持专家演示学习：使用floorplan.def作为初始布局，mgc_des_perf_1_place.def作为专家布局数据
"""

import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import subprocess
import time
import shutil

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.parsers.def_parser import parse_def
from modules.parsers.verilog_parser import VerilogParser
from modules.rl_training.real_openroad_interface import RealOpenROADInterface

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExpertDataManager:
    """专家数据管理器"""
    
    def __init__(self, design_dir: str):
        self.design_dir = Path(design_dir)
        self.design_name = self.design_dir.name
        
        # 文件路径
        self.floorplan_def = self.design_dir / "floorplan.def"
        self.expert_def = self.design_dir / "mgc_des_perf_1_place.def"
        self.verilog_file = self.design_dir / "design.v"
        self.lef_file = self.design_dir / "cells.lef"
        self.tech_lef = self.design_dir / "tech.lef"
        
        # 验证文件存在
        self._validate_files()
        
        # 解析专家数据
        self.expert_metrics = self._parse_expert_data()
        
    def _validate_files(self):
        """验证必要文件是否存在"""
        required_files = [
            self.floorplan_def, self.expert_def, 
            self.verilog_file, self.lef_file, self.tech_lef
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"缺少必要文件: {file_path}")
        
        logger.info(f"专家数据文件验证通过: {self.design_name}")
    
    def _parse_expert_data(self) -> Dict[str, Any]:
        """解析专家布局数据"""
        logger.info("正在解析专家布局数据...")
        
        # 解析专家DEF文件
        expert_def_metrics = parse_def(str(self.expert_def))
        
        # 解析Verilog文件
        verilog_metrics = VerilogParser(str(self.verilog_file)).parse()
        
        # 计算专家布局的关键指标
        expert_metrics = {
            'def_metrics': expert_def_metrics,
            'verilog_metrics': verilog_metrics,
            'component_positions': self._extract_component_positions(),
            'layout_quality': self._calculate_layout_quality(expert_def_metrics),
            'design_complexity': self._calculate_design_complexity(verilog_metrics)
        }
        
        logger.info(f"专家数据解析完成，包含 {len(expert_metrics['component_positions'])} 个组件位置")
        return expert_metrics
    
    def _extract_component_positions(self) -> Dict[str, Tuple[int, int]]:
        """提取专家布局中的组件位置"""
        positions = {}
        
        try:
            with open(self.expert_def, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip().startswith('- PLACED'):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        component_name = parts[1]
                        x = int(parts[3])
                        y = int(parts[4])
                        positions[component_name] = (x, y)
        
        except Exception as e:
            logger.warning(f"提取组件位置时出错: {e}")
        
        return positions
    
    def _calculate_layout_quality(self, def_metrics: Dict[str, Any]) -> Dict[str, float]:
        """计算布局质量指标"""
        die_area = def_metrics.get('die_area_microns', (0, 0))
        num_components = def_metrics.get('num_components', 0)
        
        # 计算密度
        total_area = die_area[0] * die_area[1]
        component_area = num_components * 100  # 假设每个组件平均100平方微米
        density = component_area / total_area if total_area > 0 else 0
        
        return {
            'density': density,
            'area_utilization': min(density, 0.8),  # 限制最大利用率
            'component_count': num_components,
            'die_area': total_area
        }
    
    def _calculate_design_complexity(self, verilog_metrics: Dict[str, Any]) -> Dict[str, float]:
        """计算设计复杂度"""
        num_instances = verilog_metrics.get('num_instances', 0)
        num_nets = verilog_metrics.get('num_nets', 0)
        
        return {
            'instance_count': num_instances,
            'net_count': num_nets,
            'complexity_score': np.log10(num_instances + 1) * np.log10(num_nets + 1)
        }

class EnhancedDesignEnvironment:
    """增强版设计环境，支持专家演示学习"""
    
    def __init__(self, design_dir: str, expert_data: ExpertDataManager):
        self.design_dir = Path(design_dir)
        self.expert_data = expert_data
        self.design_name = self.design_dir.name
        
        # OpenROAD接口
        self.openroad_interface = RealOpenROADInterface()
        
        # 当前布局状态
        self.current_def = self.design_dir / "floorplan.def"
        self.current_metrics = None
        
        # 专家基准
        self.expert_metrics = expert_data.expert_metrics
        
        # 状态维度
        self.state_dim = 8  # 增加状态维度以包含专家信息
        
        logger.info(f"增强版设计环境初始化完成: {self.design_name}")
    
    def get_state(self) -> torch.Tensor:
        """获取当前状态，包含专家信息"""
        if self.current_metrics is None:
            self.current_metrics = parse_def(str(self.current_def))
        
        # 构建状态向量
        state = self._build_state_vector()
        return state
    
    def _build_state_vector(self) -> torch.Tensor:
        """构建包含专家信息的状态向量"""
        current_def = self.current_metrics
        expert_def = self.expert_data.expert_metrics['def_metrics']
        
        # 当前布局特征
        current_die_area = current_def.get('die_area_microns', (0, 0))
        current_components = current_def.get('num_components', 0)
        
        # 专家布局特征
        expert_die_area = expert_def.get('die_area_microns', (0, 0))
        expert_components = expert_def.get('num_components', 0)
        
        # 布局质量对比
        current_density = current_components / (current_die_area[0] * current_die_area[1]) if current_die_area[0] * current_die_area[1] > 0 else 0
        expert_density = expert_components / (expert_die_area[0] * expert_die_area[1]) if expert_die_area[0] * expert_die_area[1] > 0 else 0
        
        # 状态特征
        features = [
            current_die_area[0] / 1000.0,  # 当前宽度
            current_die_area[1] / 1000.0,  # 当前高度
            current_components / 1e6,      # 当前组件数
            current_density,               # 当前密度
            expert_density,                # 专家密度
            (expert_components - current_components) / 1e6,  # 组件数差异
            (expert_density - current_density),              # 密度差异
            self.expert_data.expert_metrics['design_complexity']['complexity_score']  # 设计复杂度
        ]
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def step(self, action_k: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """执行动作并返回结果"""
        logger.debug(f"执行动作: k={action_k}")
        
        # 使用OpenROAD进行布局优化
        result = self._run_openroad_optimization(action_k)
        
        # 计算奖励
        reward = self._calculate_enhanced_reward(result, action_k)
        
        # 更新当前状态
        if result['success']:
            self.current_metrics = parse_def(str(self.current_def))
        
        # 额外信息
        info = {
            'ppa_results': result,
            'expert_similarity': self._calculate_expert_similarity(),
            'improvement': self._calculate_improvement(result)
        }
        
        return self.get_state(), reward, True, info
    
    def _run_openroad_optimization(self, k_value: int) -> Dict[str, Any]:
        """运行OpenROAD优化"""
        try:
            # 创建临时工作目录
            temp_dir = self.design_dir / f"rl_temp_{int(time.time())}"
            temp_dir.mkdir(exist_ok=True)
            
            # 复制必要文件
            shutil.copy(self.current_def, temp_dir / "current.def")
            shutil.copy(self.design_dir / "design.v", temp_dir / "design.v")
            shutil.copy(self.design_dir / "cells.lef", temp_dir / "cells.lef")
            shutil.copy(self.design_dir / "tech.lef", temp_dir / "tech.lef")
            
            # 运行OpenROAD
            result = self.openroad_interface.run_layout_optimization(
                design_dir=str(temp_dir),
                k_value=k_value,
                max_iterations=10  # 限制迭代次数以加快训练
            )
            
            # 如果成功，更新当前DEF
            if result['success']:
                output_def = temp_dir / "output" / "final.def"
                if output_def.exists():
                    shutil.copy(output_def, self.current_def)
            
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return result
            
        except Exception as e:
            logger.error(f"OpenROAD优化失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'ppa_metrics': {'area': 0, 'timing': 0, 'power': 0}
            }
    
    def _calculate_enhanced_reward(self, result: Dict[str, Any], k_value: int) -> float:
        """计算增强版奖励，包含专家信息"""
        if not result['success']:
            return -1.0  # 失败惩罚
        
        # 基础PPA奖励
        ppa_metrics = result.get('ppa_metrics', {})
        area_score = ppa_metrics.get('area', 0.5)
        timing_score = ppa_metrics.get('timing', 0.5)
        power_score = ppa_metrics.get('power', 0.5)
        
        base_reward = (area_score * 0.3 + timing_score * 0.4 + power_score * 0.3)
        
        # 专家相似度奖励
        expert_similarity = self._calculate_expert_similarity()
        expert_reward = expert_similarity * 0.3
        
        # 改进程度奖励
        improvement = self._calculate_improvement(result)
        improvement_reward = improvement * 0.2
        
        # k值选择奖励
        k_reward = self._calculate_k_reward(k_value)
        
        # 总奖励
        total_reward = base_reward + expert_reward + improvement_reward + k_reward
        
        # 归一化到[-1, 1]范围
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        return float(total_reward)
    
    def _calculate_expert_similarity(self) -> float:
        """计算与专家布局的相似度"""
        if self.current_metrics is None:
            return 0.0
        
        current_def = self.current_metrics
        expert_def = self.expert_data.expert_metrics['def_metrics']
        
        # 计算各种相似度指标
        area_similarity = self._calculate_area_similarity(current_def, expert_def)
        density_similarity = self._calculate_density_similarity(current_def, expert_def)
        component_similarity = self._calculate_component_similarity(current_def, expert_def)
        
        # 加权平均
        similarity = (area_similarity * 0.4 + density_similarity * 0.3 + component_similarity * 0.3)
        
        return similarity
    
    def _calculate_area_similarity(self, current: Dict, expert: Dict) -> float:
        """计算面积相似度"""
        current_area = current.get('die_area_microns', (0, 0))
        expert_area = expert.get('die_area_microns', (0, 0))
        
        if expert_area[0] * expert_area[1] == 0:
            return 0.0
        
        current_total = current_area[0] * current_area[1]
        expert_total = expert_area[0] * expert_area[1]
        
        similarity = 1.0 - abs(current_total - expert_total) / expert_total
        return max(0.0, similarity)
    
    def _calculate_density_similarity(self, current: Dict, expert: Dict) -> float:
        """计算密度相似度"""
        current_components = current.get('num_components', 0)
        expert_components = expert.get('num_components', 0)
        
        current_area = current.get('die_area_microns', (0, 0))
        expert_area = expert.get('die_area_microns', (0, 0))
        
        if expert_area[0] * expert_area[1] == 0:
            return 0.0
        
        current_density = current_components / (current_area[0] * current_area[1])
        expert_density = expert_components / (expert_area[0] * expert_area[1])
        
        if expert_density == 0:
            return 0.0
        
        similarity = 1.0 - abs(current_density - expert_density) / expert_density
        return max(0.0, similarity)
    
    def _calculate_component_similarity(self, current: Dict, expert: Dict) -> float:
        """计算组件相似度"""
        current_components = current.get('num_components', 0)
        expert_components = expert.get('num_components', 0)
        
        if expert_components == 0:
            return 0.0
        
        similarity = 1.0 - abs(current_components - expert_components) / expert_components
        return max(0.0, similarity)
    
    def _calculate_improvement(self, result: Dict[str, Any]) -> float:
        """计算改进程度"""
        # 这里可以根据历史记录计算改进程度
        # 简化版本：基于PPA指标
        ppa_metrics = result.get('ppa_metrics', {})
        overall_score = ppa_metrics.get('overall_score', 0.5)
        
        # 假设基准分数为0.5
        improvement = (overall_score - 0.5) * 2  # 归一化到[-1, 1]
        return max(-1.0, min(1.0, improvement))
    
    def _calculate_k_reward(self, k_value: int) -> float:
        """计算k值选择奖励"""
        # 基于设计复杂度的最优k值
        complexity = self.expert_data.expert_metrics['design_complexity']['complexity_score']
        optimal_k = 4 + int(complexity * 3)  # 复杂度越高，最优k值越大
        
        # 高斯奖励函数
        k_diff = abs(k_value - optimal_k)
        k_reward = np.exp(-(k_diff ** 2) / 8.0)  # 标准差为2的高斯函数
        
        return k_reward - 0.5  # 归一化到[-0.5, 0.5]

class ExpertGuidedActorCritic(nn.Module):
    """专家引导的Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(ExpertGuidedActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 专家知识网络（用于模仿学习）
        self.expert_knowledge = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        
        # Actor输出
        actor_logits = self.actor(features)
        action_probs = F.softmax(actor_logits, dim=-1)
        
        # Critic输出
        state_value = self.critic(features)
        
        # 专家知识输出
        expert_logits = self.expert_knowledge(state)
        expert_probs = F.softmax(expert_logits, dim=-1)
        
        return action_probs, state_value, expert_probs

def train_with_expert_guidance(design_dir: str, config: Dict[str, Any]) -> None:
    """使用专家指导进行训练"""
    design_name = Path(design_dir).name
    logger.info(f"=== 开始专家指导训练: {design_name} ===")
    
    # 初始化专家数据管理器
    expert_data = ExpertDataManager(design_dir)
    
    # 初始化增强版环境
    env = EnhancedDesignEnvironment(design_dir, expert_data)
    
    # 初始化网络
    action_space = list(range(3, 16))  # k值从3到15
    network = ExpertGuidedActorCritic(env.state_dim, len(action_space))
    optimizer = optim.Adam(network.parameters(), lr=config.get('learning_rate', 0.0003))
    
    # 训练参数
    num_episodes = config.get('num_episodes', 500)
    epsilon_start = config.get('epsilon_start', 0.9)
    epsilon_end = config.get('epsilon_end', 0.05)
    epsilon_decay = config.get('epsilon_decay', 0.995)
    expert_weight = config.get('expert_weight', 0.3)  # 专家知识权重
    
    epsilon = epsilon_start
    history = []
    
    for episode in range(num_episodes):
        state = env.get_state()
        
        # 选择动作
        if random.random() < epsilon:
            # 探索：随机选择
            action_idx = random.choice(range(len(action_space)))
        else:
            # 利用：网络选择
            with torch.no_grad():
                action_probs, _, expert_probs = network(state)
            
            # 混合策略：结合网络输出和专家知识
            mixed_probs = (1 - expert_weight) * action_probs + expert_weight * expert_probs
            action_dist = Categorical(mixed_probs)
            action_idx = action_dist.sample().item()
        
        k_value = action_space[action_idx]
        
        # 执行动作
        next_state, reward, done, info = env.step(k_value)
        
        # 计算损失
        action_probs, state_value, expert_probs = network(state)
        action_dist = Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor([action_idx]))
        
        # Actor-Critic损失
        advantage = reward - state_value.item()
        actor_loss = -(log_prob * advantage)
        critic_loss = F.mse_loss(state_value, torch.tensor([[reward]]))
        
        # 专家模仿损失
        expert_target = torch.zeros_like(action_probs)
        expert_target[0, action_idx] = 1.0  # 将选择的动作作为专家目标
        imitation_loss = F.kl_div(F.log_softmax(expert_probs, dim=-1), expert_target, reduction='batchmean')
        
        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss + expert_weight * imitation_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录历史
        history.append({
            'episode': episode,
            'reward': reward,
            'k_value': k_value,
            'epsilon': epsilon,
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'imitation_loss': imitation_loss.item(),
            'expert_similarity': info['expert_similarity'],
            'improvement': info['improvement']
        })
        
        # 更新探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 定期输出
        if episode > 0 and episode % 50 == 0:
            avg_reward = np.mean([h['reward'] for h in history[-50:]])
            avg_similarity = np.mean([h['expert_similarity'] for h in history[-50:]])
            logger.info(f"Episode {episode}: 平均奖励={avg_reward:.3f}, 专家相似度={avg_similarity:.3f}, ε={epsilon:.3f}")
    
    # 保存结果
    output_dir = Path(__file__).parent / "models" / "expert_guided_rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_dir / f"{design_name}_expert_guided_model.pt"
    torch.save(network.state_dict(), model_path)
    
    # 保存训练报告
    report_path = output_dir / f"{design_name}_expert_guided_report.json"
    with open(report_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"专家指导训练完成: {design_name}")
    logger.info(f"模型保存至: {model_path}")
    logger.info(f"报告保存至: {report_path}")

def main():
    """主函数"""
    logger.info("=================================================")
    logger.info("=== 专家指导强化学习训练系统启动 ===")
    logger.info("=================================================")
    
    # 设计目录
    design_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    design_path = Path(__file__).parent / design_dir
    
    if not design_path.exists():
        logger.error(f"设计目录不存在: {design_path}")
        return
    
    # 训练配置
    config = {
        "learning_rate": 0.0003,
        "num_episodes": 500,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
        "expert_weight": 0.3
    }
    
    # 开始训练
    train_with_expert_guidance(str(design_path), config)

if __name__ == '__main__':
    main() 