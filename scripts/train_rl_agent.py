#!/usr/bin/env python3
"""
强化学习智能体训练脚本
支持离线训练和在线学习
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import random

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from modules.core.rl_agent import QLearningAgent, State, Action, Experience
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.config_loader import ConfigLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLAgentTrainer:
    """RL智能体训练器"""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """初始化"""
        self.config = ConfigLoader.load_config(config_path)
        self.rl_agent = QLearningAgent(self.config.get('rl_agent', {}))
        self.knowledge_base = KnowledgeBase(self.config.get('knowledge_base', {}))
        
    def train_offline(self, training_data: List[Dict[str, Any]], epochs: int = 100):
        """离线训练RL智能体
        
        Args:
            training_data: 训练数据
            epochs: 训练轮数
        """
        logger.info(f"开始离线训练，数据量: {len(training_data)}, 轮数: {epochs}")
        
        for epoch in range(epochs):
            epoch_reward = 0.0
            epoch_experiences = 0
            
            # 随机打乱训练数据
            random.shuffle(training_data)
            
            for data in training_data:
                try:
                    # 构建状态
                    state = self._build_state_from_data(data)
                    
                    # 选择动作
                    action_dict = self.rl_agent.select_action(state)
                    action = Action(
                        k_value=action_dict['k_value'],
                        confidence=action_dict['confidence'],
                        exploration_type=action_dict['exploration_type']
                    )
                    
                    # 计算奖励
                    reward = self._calculate_reward_from_data(data)
                    
                    # 构建下一个状态
                    next_state = self._build_next_state(state, action, data)
                    
                    # 更新Q值
                    self.rl_agent.update_q_value(state, action.k_value, reward, next_state)
                    
                    # 添加到经验回放
                    experience = Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        timestamp='2025-07-11'
                    )
                    self.rl_agent.add_experience(experience)
                    
                    epoch_reward += reward
                    epoch_experiences += 1
                    
                except Exception as e:
                    logger.error(f"训练数据点处理失败: {str(e)}")
                    continue
            
            # 批量更新
            if epoch_experiences > 0:
                self.rl_agent.batch_update(batch_size=min(32, epoch_experiences))
            
            # 更新探索率
            self.rl_agent.epsilon = max(
                self.rl_agent.epsilon_min,
                self.rl_agent.epsilon * self.rl_agent.epsilon_decay
            )
            
            # 记录训练统计
            avg_reward = epoch_reward / epoch_experiences if epoch_experiences > 0 else 0.0
            self.rl_agent.training_stats['episodes'] += 1
            self.rl_agent.training_stats['total_reward'] += epoch_reward
            self.rl_agent.training_stats['average_reward'] = avg_reward
            self.rl_agent.training_stats['exploration_rate'] = self.rl_agent.epsilon
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: 平均奖励={avg_reward:.3f}, 探索率={self.rl_agent.epsilon:.3f}")
        
        # 保存训练后的模型
        self._save_trained_model()
        
        logger.info("离线训练完成！")
    
    def train_online(self, experiment_results: List[Dict[str, Any]]):
        """在线训练RL智能体
        
        Args:
            experiment_results: 实验结果数据
        """
        logger.info(f"开始在线训练，结果数量: {len(experiment_results)}")
        
        for result in experiment_results:
            try:
                # 从实验结果构建训练数据
                training_data = self._extract_training_data_from_result(result)
                
                for data in training_data:
                    # 构建状态
                    state = self._build_state_from_data(data)
                    
                    # 选择动作
                    action_dict = self.rl_agent.select_action(state)
                    action = Action(
                        k_value=action_dict['k_value'],
                        confidence=action_dict['confidence'],
                        exploration_type=action_dict['exploration_type']
                    )
                    
                    # 计算奖励
                    reward = self._calculate_reward_from_data(data)
                    
                    # 构建下一个状态
                    next_state = self._build_next_state(state, action, data)
                    
                    # 更新Q值
                    self.rl_agent.update_q_value(state, action.k_value, reward, next_state)
                    
                    # 添加到经验回放
                    experience = Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        timestamp='2025-07-11'
                    )
                    self.rl_agent.add_experience(experience)
                
            except Exception as e:
                logger.error(f"在线训练数据处理失败: {str(e)}")
                continue
        
        # 批量更新
        self.rl_agent.batch_update(batch_size=32)
        
        # 保存训练后的模型
        self._save_trained_model()
        
        logger.info("在线训练完成！")
    
    def _build_state_from_data(self, data: Dict[str, Any]) -> State:
        """从数据构建状态"""
        # 提取基本信息
        design_info = data.get('design_info', {})
        query_info = data.get('query_info', {})
        result_info = data.get('result_info', {})
        
        # 构建状态
        state = State(
            # 查询复杂度特征
            query_complexity=data.get('query_complexity', 0.5),
            query_length=len(str(query_info)),
            query_type=query_info.get('type', 'general'),
            
            # 设计特征
            design_type=design_info.get('design_type', 'unknown'),
            design_size=design_info.get('num_components', 0),
            design_area=design_info.get('area', 0.0),
            constraint_count=len(design_info.get('constraints', [])),
            constraint_types=list(set([c.get('type', 'unknown') for c in design_info.get('constraints', [])])),
            
            # 检索特征
            initial_relevance=result_info.get('initial_relevance', 0.5),
            result_diversity=result_info.get('diversity', 0.5),
            knowledge_coverage=result_info.get('coverage', 0.5),
            entity_count=result_info.get('entity_count', 0),
            
            # 性能特征
            historical_performance=data.get('historical_performance', 0.5),
            recent_success_rate=data.get('success_rate', 0.5),
            average_quality_score=data.get('quality_score', 0.5),
            
            # 上下文特征
            current_iteration=data.get('iteration', 1),
            optimization_stage=data.get('stage', 'initial'),
            
            # 时间戳
            timestamp='2025-07-11'
        )
        
        return state
    
    def _build_next_state(self, current_state: State, action: Action, data: Dict[str, Any]) -> State:
        """构建下一个状态"""
        # 基于当前状态和动作构建下一个状态
        next_state = State(
            # 查询复杂度特征（保持不变）
            query_complexity=current_state.query_complexity,
            query_length=current_state.query_length,
            query_type=current_state.query_type,
            
            # 设计特征（保持不变）
            design_type=current_state.design_type,
            design_size=current_state.design_size,
            design_area=current_state.design_area,
            constraint_count=current_state.constraint_count,
            constraint_types=current_state.constraint_types,
            
            # 检索特征（基于动作结果更新）
            initial_relevance=data.get('next_relevance', current_state.initial_relevance),
            result_diversity=data.get('next_diversity', current_state.result_diversity),
            knowledge_coverage=data.get('next_coverage', current_state.knowledge_coverage),
            entity_count=data.get('next_entity_count', current_state.entity_count),
            
            # 性能特征（基于奖励更新）
            historical_performance=data.get('next_performance', current_state.historical_performance),
            recent_success_rate=data.get('next_success_rate', current_state.recent_success_rate),
            average_quality_score=data.get('next_quality_score', current_state.average_quality_score),
            
            # 上下文特征（迭代更新）
            current_iteration=current_state.current_iteration + 1,
            optimization_stage=self._determine_next_stage(current_state.optimization_stage, action),
            
            # 时间戳
            timestamp='2025-07-11'
        )
        
        return next_state
    
    def _determine_next_stage(self, current_stage: str, action: Action) -> str:
        """确定下一个优化阶段"""
        if current_stage == 'initial':
            return 'refinement' if action.k_value > 3 else 'initial'
        elif current_stage == 'refinement':
            return 'final' if action.confidence > 0.8 else 'refinement'
        else:
            return 'final'
    
    def _calculate_reward_from_data(self, data: Dict[str, Any]) -> float:
        """从数据计算奖励"""
        # 基础奖励
        base_reward = data.get('quality_score', 0.5)
        
        # 性能奖励
        performance_bonus = data.get('performance_improvement', 0.0)
        
        # 效率奖励（k值越小，效率越高）
        k_value = data.get('k_value', 3)
        efficiency_bonus = max(0, (5 - k_value) / 5) * 0.2
        
        # 探索奖励（鼓励探索新策略）
        exploration_bonus = 0.1 if data.get('exploration_type') == 'explore' else 0.0
        
        total_reward = base_reward + performance_bonus + efficiency_bonus + exploration_bonus
        
        return min(1.0, max(0.0, total_reward))
    
    def _extract_training_data_from_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从实验结果提取训练数据"""
        training_data = []
        
        # 提取布局策略生成数据
        if 'layout_strategy' in result:
            strategy_data = {
                'design_info': result.get('design_info', {}),
                'query_info': result.get('query_info', {}),
                'result_info': result.get('retrieval_result', {}),
                'quality_score': result.get('quality_score', 0.5),
                'k_value': result.get('k_value', 3),
                'exploration_type': result.get('exploration_type', 'exploit'),
                'performance_improvement': result.get('performance_improvement', 0.0),
                'iteration': result.get('iteration', 1),
                'stage': result.get('stage', 'initial'),
                'historical_performance': result.get('historical_performance', 0.5),
                'success_rate': result.get('success_rate', 0.5)
            }
            training_data.append(strategy_data)
        
        # 提取HPWL对比数据
        if 'hpwl_comparison' in result:
            hpwl_data = result['hpwl_comparison']
            if 'improvement_rate' in hpwl_data:
                training_data.append({
                    'design_info': result.get('design_info', {}),
                    'query_info': {'type': 'hpwl_optimization'},
                    'result_info': {'initial_relevance': 0.5},
                    'quality_score': hpwl_data.get('improvement_rate', 0.0),
                    'k_value': result.get('k_value', 3),
                    'exploration_type': result.get('exploration_type', 'exploit'),
                    'performance_improvement': hpwl_data.get('improvement_rate', 0.0),
                    'iteration': result.get('iteration', 1),
                    'stage': 'final',
                    'historical_performance': result.get('historical_performance', 0.5),
                    'success_rate': 1.0 if hpwl_data.get('improvement_rate', 0) > 0 else 0.0
                })
        
        return training_data
    
    def _save_trained_model(self):
        """保存训练后的模型"""
        try:
            # 保存Q表
            self.rl_agent._save_q_table()
            
            # 保存训练统计
            stats_file = Path("data/rl_training/training_stats.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_file, 'w') as f:
                json.dump(self.rl_agent.training_stats, f, indent=2)
            
            logger.info(f"训练模型已保存，统计信息: {self.rl_agent.training_stats}")
            
        except Exception as e:
            logger.error(f"保存训练模型失败: {str(e)}")
    
    def generate_training_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """生成训练数据"""
        logger.info(f"生成 {num_samples} 个训练样本")
        
        training_data = []
        
        for i in range(num_samples):
            # 生成随机设计信息
            design_info = {
                'design_type': random.choice(['FFT', 'PCI', 'DES', 'Matrix', 'General']),
                'num_components': random.randint(1000, 50000),
                'area': random.uniform(100000, 10000000),
                'constraints': [{'type': random.choice(['timing', 'power', 'area'])} for _ in range(random.randint(1, 5))]
            }
            
            # 生成随机查询信息
            query_info = {
                'type': random.choice(['optimization', 'analysis', 'synthesis']),
                'complexity': random.uniform(0.1, 1.0)
            }
            
            # 生成随机结果信息
            result_info = {
                'initial_relevance': random.uniform(0.3, 0.9),
                'diversity': random.uniform(0.2, 0.8),
                'coverage': random.uniform(0.4, 0.9),
                'entity_count': random.randint(5, 50)
            }
            
            # 生成随机性能指标
            quality_score = random.uniform(0.3, 0.9)
            performance_improvement = random.uniform(-0.2, 0.3)
            
            training_data.append({
                'design_info': design_info,
                'query_info': query_info,
                'result_info': result_info,
                'quality_score': quality_score,
                'k_value': random.randint(1, 5),
                'exploration_type': random.choice(['explore', 'exploit']),
                'performance_improvement': performance_improvement,
                'iteration': random.randint(1, 10),
                'stage': random.choice(['initial', 'refinement', 'final']),
                'historical_performance': random.uniform(0.4, 0.8),
                'success_rate': random.uniform(0.5, 0.9),
                'query_complexity': query_info['complexity']
            })
        
        logger.info(f"生成了 {len(training_data)} 个训练样本")
        return training_data

def main():
    """主函数"""
    trainer = RLAgentTrainer()
    
    # 生成训练数据
    training_data = trainer.generate_training_data(num_samples=500)
    
    # 离线训练
    trainer.train_offline(training_data, epochs=50)
    
    logger.info("RL智能体训练完成！")

if __name__ == "__main__":
    main() 