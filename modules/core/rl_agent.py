"""
强化学习智能体模块
实现基于Q-Learning的动态检索策略优化
"""

import torch
import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random
from datetime import datetime
import hashlib
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class State:
    """状态表示"""
    query_complexity: float
    design_type: str
    constraint_count: int
    initial_relevance: float
    result_diversity: float
    historical_performance: float
    timestamp: str

@dataclass
class Action:
    """动作表示"""
    k_value: int
    confidence: float
    exploration_type: str  # 'explore' or 'exploit'

@dataclass
class Experience:
    """经验回放数据"""
    state: State
    action: Action
    reward: float
    next_state: State
    timestamp: str

class QLearningAgent:
    """Q-Learning智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Q-Learning智能体
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Q-Learning参数
        self.alpha = config.get('alpha', 0.01)  # 学习率
        self.gamma = config.get('gamma', 0.95)  # 折扣因子
        self.epsilon = config.get('epsilon', 0.9)  # 初始探索率
        self.epsilon_min = config.get('epsilon_min', 0.01)  # 最小探索率
        self.epsilon_decay = config.get('epsilon_decay', 0.995)  # 探索率衰减
        
        # 动作空间
        self.k_range = config.get('k_range', (3, 15))
        self.action_space = list(range(self.k_range[0], self.k_range[1] + 1))
        
        # Q表
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # 经验回放缓冲区
        self.experience_buffer = []
        self.buffer_size = config.get('buffer_size', 10000)
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'average_reward': 0.0,
            'exploration_rate': self.epsilon,
            'convergence_episodes': 0
        }
        
        # 加载预训练模型（如果存在）
        self._load_pretrained_model()
        
        self.logger.info(f"Q-Learning智能体初始化完成，动作空间: {self.action_space}")
    
    def choose_action(self, state: State) -> Action:
        """选择动作（k值）
        
        Args:
            state: 当前状态
            
        Returns:
            Action: 选择的动作
        """
        state_key = self._hash_state(state)
        
        # ε-greedy策略
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            k_value = random.choice(self.action_space)
            exploration_type = 'explore'
            confidence = 1.0 / len(self.action_space)  # 随机选择的置信度
        else:
            # 利用：选择Q值最大的动作
            q_values = self.q_table[state_key]
            if q_values:
                k_value = max(q_values.keys(), key=lambda x: q_values[x])
                confidence = q_values[k_value] / max(q_values.values()) if max(q_values.values()) > 0 else 0.0
            else:
                k_value = self.k_range[0]  # 默认最小k值
                confidence = 0.0
            exploration_type = 'exploit'
        
        action = Action(
            k_value=k_value,
            confidence=confidence,
            exploration_type=exploration_type
        )
        
        self.logger.debug(f"选择动作: k={k_value}, 类型={exploration_type}, 置信度={confidence:.3f}")
        return action
    
    def update(self, state: State, action: Action, reward: float, next_state: State):
        """更新Q值
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
        """
        state_key = self._hash_state(state)
        next_state_key = self._hash_state(next_state)
        
        # 获取当前Q值
        current_q = self.q_table[state_key].get(action.k_value, 0.0)
        
        # 获取下一状态的最大Q值
        next_q_values = self.q_table[next_state_key]
        next_max_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-Learning更新公式
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action.k_value] = new_q
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新训练统计
        self.training_stats['exploration_rate'] = self.epsilon
        
        self.logger.debug(f"Q值更新: 状态={state_key}, 动作={action.k_value}, "
                         f"旧Q值={current_q:.3f}, 新Q值={new_q:.3f}")
    
    def add_experience(self, experience: Experience):
        """添加经验到回放缓冲区
        
        Args:
            experience: 经验数据
        """
        self.experience_buffer.append(experience)
        
        # 保持缓冲区大小
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def batch_update(self, batch_size: int = 32):
        """批量更新Q值
        
        Args:
            batch_size: 批次大小
        """
        if len(self.experience_buffer) < batch_size:
            return
        
        # 随机采样经验
        batch = random.sample(self.experience_buffer, batch_size)
        
        for experience in batch:
            self.update(
                experience.state,
                Action(k_value=experience.action.k_value, confidence=0.0, exploration_type='batch'),
                experience.reward,
                experience.next_state
            )
    
    def _hash_state(self, state: State) -> str:
        """将状态哈希为字符串键
        
        Args:
            state: 状态对象
            
        Returns:
            str: 状态哈希值
        """
        # 将状态特征转换为可哈希的字符串
        state_features = [
            f"complexity_{state.query_complexity:.2f}",
            f"type_{state.design_type}",
            f"constraints_{state.constraint_count}",
            f"relevance_{state.initial_relevance:.2f}",
            f"diversity_{state.result_diversity:.2f}",
            f"performance_{state.historical_performance:.2f}"
        ]
        
        state_str = "|".join(state_features)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        model_path = self.config.get('pretrained_model_path')
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.q_table = saved_data.get('q_table', self.q_table)
                    self.epsilon = saved_data.get('epsilon', self.epsilon)
                    self.training_stats = saved_data.get('training_stats', self.training_stats)
                self.logger.info(f"成功加载预训练模型: {model_path}")
            except Exception as e:
                self.logger.warning(f"加载预训练模型失败: {str(e)}")
    
    def save_model(self, model_path: str):
        """保存模型
        
        Args:
            model_path: 模型保存路径
        """
        try:
            saved_data = {
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'training_stats': self.training_stats,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(saved_data, f)
            
            self.logger.info(f"模型保存成功: {model_path}")
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息
        
        Returns:
            Dict[str, Any]: 训练统计信息
        """
        return self.training_stats.copy()
    
    def get_q_table_stats(self) -> Dict[str, Any]:
        """获取Q表统计信息
        
        Returns:
            Dict[str, Any]: Q表统计信息
        """
        total_states = len(self.q_table)
        total_actions = sum(len(actions) for actions in self.q_table.values())
        
        # 计算Q值的统计信息
        all_q_values = []
        for actions in self.q_table.values():
            all_q_values.extend(actions.values())
        
        if all_q_values:
            q_stats = {
                'mean': np.mean(all_q_values),
                'std': np.std(all_q_values),
                'min': np.min(all_q_values),
                'max': np.max(all_q_values)
            }
        else:
            q_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'total_states': total_states,
            'total_actions': total_actions,
            'average_actions_per_state': total_actions / total_states if total_states > 0 else 0,
            'q_value_stats': q_stats
        }

class StateExtractor:
    """状态特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化状态提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 历史性能缓存
        self.performance_cache = defaultdict(list)
        self.cache_size = config.get('performance_cache_size', 1000)
    
    def extract_state_features(self, 
                             query: Dict[str, Any], 
                             design_info: Dict[str, Any],
                             initial_results: List[Any]) -> State:
        """提取状态特征
        
        Args:
            query: 查询信息
            design_info: 设计信息
            initial_results: 初始检索结果
            
        Returns:
            State: 状态对象
        """
        try:
            # 计算查询复杂度
            query_complexity = self._calculate_query_complexity(query)
            
            # 获取设计类型
            design_type = design_info.get('design_type', 'unknown')
            
            # 计算约束数量
            constraint_count = len(design_info.get('constraints', []))
            
            # 计算初始相关性
            initial_relevance = np.mean([getattr(r, 'relevance_score', 0.0) for r in initial_results]) if initial_results else 0.0
            
            # 计算结果多样性
            result_diversity = self._calculate_diversity(initial_results)
            
            # 获取历史性能
            historical_performance = self._get_historical_performance(query)
            
            state = State(
                query_complexity=query_complexity,
                design_type=design_type,
                constraint_count=constraint_count,
                initial_relevance=initial_relevance,
                result_diversity=result_diversity,
                historical_performance=historical_performance,
                timestamp=datetime.now().isoformat()
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"状态特征提取失败: {str(e)}")
            # 返回默认状态
            return State(
                query_complexity=0.5,
                design_type='unknown',
                constraint_count=0,
                initial_relevance=0.0,
                result_diversity=0.0,
                historical_performance=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def _calculate_query_complexity(self, query: Dict[str, Any]) -> float:
        """计算查询复杂度
        
        Args:
            query: 查询信息
            
        Returns:
            float: 复杂度分数 (0-1)
        """
        complexity_score = 0.0
        
        # 基于查询长度
        query_text = query.get('text', '')
        text_length = len(query_text.split())
        complexity_score += min(text_length / 50.0, 1.0) * 0.3
        
        # 基于约束数量
        constraints = query.get('constraints', {})
        constraint_count = len(constraints)
        complexity_score += min(constraint_count / 10.0, 1.0) * 0.4
        
        # 基于设计类型
        design_type = query.get('design_type', 'unknown')
        if design_type in ['complex', 'advanced', 'multi_module']:
            complexity_score += 0.3
        
        return min(complexity_score, 1.0)
    
    def _calculate_diversity(self, results: List[Any]) -> float:
        """计算结果多样性
        
        Args:
            results: 检索结果列表
            
        Returns:
            float: 多样性分数 (0-1)
        """
        if not results:
            return 0.0
        
        # 基于相关性分数的标准差
        relevance_scores = [getattr(r, 'relevance_score', 0.0) for r in results]
        if len(relevance_scores) > 1:
            diversity = np.std(relevance_scores) / np.mean(relevance_scores) if np.mean(relevance_scores) > 0 else 0.0
        else:
            diversity = 0.0
        
        return min(diversity, 1.0)
    
    def _get_historical_performance(self, query: Dict[str, Any]) -> float:
        """获取历史性能
        
        Args:
            query: 查询信息
            
        Returns:
            float: 历史性能分数 (0-1)
        """
        query_key = self._hash_query(query)
        
        if query_key in self.performance_cache:
            performances = self.performance_cache[query_key]
            return np.mean(performances[-10:]) if performances else 0.0
        
        return 0.0
    
    def update_performance_cache(self, query: Dict[str, Any], performance: float):
        """更新性能缓存
        
        Args:
            query: 查询信息
            performance: 性能分数
        """
        query_key = self._hash_query(query)
        
        self.performance_cache[query_key].append(performance)
        
        # 保持缓存大小
        if len(self.performance_cache[query_key]) > self.cache_size:
            self.performance_cache[query_key] = self.performance_cache[query_key][-self.cache_size:]
    
    def _hash_query(self, query: Dict[str, Any]) -> str:
        """哈希查询信息
        
        Args:
            query: 查询信息
            
        Returns:
            str: 查询哈希值
        """
        query_str = json.dumps(query, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()

class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化奖励计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 奖励权重
        self.weights = config.get('reward_weights', {
            'layout_quality': 0.4,
            'constraint_satisfaction': 0.3,
            'performance': 0.3
        })
        
        # 奖励阈值
        self.thresholds = config.get('reward_thresholds', {
            'layout_quality': 0.8,
            'constraint_satisfaction': 0.9,
            'performance': 0.85
        })
    
    def calculate_reward(self, quality_feedback: Dict[str, Any]) -> float:
        """计算奖励
        
        Args:
            quality_feedback: 质量反馈信息
            
        Returns:
            float: 奖励值
        """
        try:
            # 提取质量指标
            layout_quality = quality_feedback.get('layout_quality', {}).get('value', 0.0)
            constraint_satisfaction = quality_feedback.get('constraint_satisfaction', {}).get('value', 0.0)
            performance = quality_feedback.get('performance', {}).get('value', 0.0)
            
            # 计算加权奖励
            reward = (
                self.weights['layout_quality'] * self._normalize_score(layout_quality, self.thresholds['layout_quality']) +
                self.weights['constraint_satisfaction'] * self._normalize_score(constraint_satisfaction, self.thresholds['constraint_satisfaction']) +
                self.weights['performance'] * self._normalize_score(performance, self.thresholds['performance'])
            )
            
            # 添加奖励调整
            reward = self._apply_reward_adjustments(reward, quality_feedback)
            
            self.logger.debug(f"奖励计算: 布局质量={layout_quality:.3f}, "
                            f"约束满足={constraint_satisfaction:.3f}, "
                            f"性能={performance:.3f}, 总奖励={reward:.3f}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"奖励计算失败: {str(e)}")
            return 0.0
    
    def _normalize_score(self, score: float, threshold: float) -> float:
        """标准化分数
        
        Args:
            score: 原始分数
            threshold: 阈值
            
        Returns:
            float: 标准化后的分数
        """
        if score >= threshold:
            return 1.0
        else:
            return score / threshold
    
    def _apply_reward_adjustments(self, base_reward: float, quality_feedback: Dict[str, Any]) -> float:
        """应用奖励调整
        
        Args:
            base_reward: 基础奖励
            quality_feedback: 质量反馈
            
        Returns:
            float: 调整后的奖励
        """
        adjusted_reward = base_reward
        
        # 基于总体评分的调整
        overall_score = quality_feedback.get('overall', 0.0)
        if overall_score >= 0.9:
            adjusted_reward *= 1.2  # 优秀结果给予额外奖励
        elif overall_score < 0.5:
            adjusted_reward *= 0.8  # 较差结果给予惩罚
        
        return adjusted_reward 