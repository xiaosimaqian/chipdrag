"""
强化学习训练器模块
实现Chip-D-RAG系统的强化学习训练流程
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
import matplotlib.pyplot as plt
from pathlib import Path

from .rl_agent import QLearningAgent, StateExtractor, RewardCalculator, State, Action, Experience
from ..retrieval.dynamic_rag_retriever import DynamicRAGRetriever
from ..core.layout_generator import LayoutGenerator
from ..evaluation.multi_objective_evaluator import MultiObjectiveEvaluator

logger = logging.getLogger(__name__)

@dataclass
class TrainingEpisode:
    """训练episode数据"""
    episode_id: int
    query: Dict[str, Any]
    design_info: Dict[str, Any]
    state: State
    action: Action
    reward: float
    next_state: State
    quality_feedback: Dict[str, Any]
    timestamp: str

@dataclass
class TrainingMetrics:
    """训练指标"""
    episode: int
    total_reward: float
    average_reward: float
    exploration_rate: float
    convergence_score: float
    q_table_size: int
    timestamp: str

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self._init_components()
        
        # 训练参数
        self.max_episodes = config.get('max_episodes', 1000)
        self.batch_size = config.get('batch_size', 32)
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        self.convergence_window = config.get('convergence_window', 100)
        
        # 训练状态
        self.current_episode = 0
        self.training_history = []
        self.convergence_history = []
        
        # 性能监控
        self.performance_monitor = TrainingPerformanceMonitor()
        
        self.logger.info("强化学习训练器初始化完成")
    
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化强化学习智能体
            rl_config = self.config.get('reinforcement_learning', {})
            self.agent = QLearningAgent(rl_config)
            
            # 初始化状态提取器
            self.state_extractor = StateExtractor(rl_config)
            
            # 初始化奖励计算器
            self.reward_calculator = RewardCalculator(rl_config)
            
            # 初始化检索器
            retriever_config = self.config.get('retriever', {})
            self.retriever = DynamicRAGRetriever(retriever_config)
            
            # 初始化布局生成器
            generator_config = self.config.get('generator', {})
            self.generator = LayoutGenerator(generator_config)
            
            # 初始化评估器
            evaluator_config = self.config.get('evaluator', {})
            self.evaluator = MultiObjectiveEvaluator(evaluator_config)
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {str(e)}")
            raise
    
    def train(self, training_data: List[Dict[str, Any]], validation_data: Optional[List[Dict[str, Any]]] = None):
        """开始训练
        
        Args:
            training_data: 训练数据
            validation_data: 验证数据
        """
        self.logger.info(f"开始训练，总episodes: {self.max_episodes}")
        
        # 训练循环
        for episode in range(self.max_episodes):
            self.current_episode = episode
            
            # 随机选择训练样本
            episode_data = random.choice(training_data)
            
            # 训练一个episode
            episode_result = self._train_episode(episode_data)
            
            # 记录训练历史
            self.training_history.append(episode_result)
            
            # 批量更新
            if episode % self.batch_size == 0:
                self.agent.batch_update(self.batch_size)
            
            # 验证（如果提供验证数据）
            if validation_data and episode % 100 == 0:
                validation_score = self._validate(validation_data)
                self.logger.info(f"Episode {episode}: 验证分数 = {validation_score:.3f}")
            
            # 检查收敛性
            if self._check_convergence():
                self.logger.info(f"训练在第 {episode} 个episode收敛")
                break
            
            # 定期保存模型
            if episode % 100 == 0:
                self._save_checkpoint(episode)
            
            # 记录训练指标
            if episode % 10 == 0:
                metrics = self._calculate_training_metrics(episode)
                self.performance_monitor.record_metrics(metrics)
        
        # 训练完成
        self._finalize_training()
    
    def _train_episode(self, episode_data: Dict[str, Any]) -> TrainingEpisode:
        """训练一个episode
        
        Args:
            episode_data: episode数据
            
        Returns:
            TrainingEpisode: episode结果
        """
        try:
            query = episode_data['query']
            design_info = episode_data['design_info']
            
            # 1. 初始检索
            initial_results = self.retriever._initial_retrieval(query, design_info)
            
            # 2. 提取状态
            state = self.state_extractor.extract_state_features(
                query, design_info, initial_results
            )
            
            # 3. 选择动作
            action = self.agent.choose_action(state)
            
            # 4. 执行检索
            final_results = self.retriever.retrieve_with_dynamic_reranking(
                query, design_info
            )
            
            # 5. 生成布局
            layout_result = self.generator.generate_layout(query, final_results)
            
            # 6. 评估质量
            quality_feedback = self.evaluator.evaluate(layout_result)
            
            # 7. 计算奖励
            reward = self.reward_calculator.calculate_reward(quality_feedback)
            
            # 8. 提取下一状态
            next_state = self.state_extractor.extract_state_features(
                query, design_info, final_results
            )
            
            # 9. 更新智能体
            self.agent.update(state, action, reward, next_state)
            
            # 10. 添加经验
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                timestamp=datetime.now().isoformat()
            )
            self.agent.add_experience(experience)
            
            # 11. 更新性能缓存
            overall_score = quality_feedback.get('overall', 0.0)
            self.state_extractor.update_performance_cache(query, overall_score)
            
            # 创建episode结果
            episode_result = TrainingEpisode(
                episode_id=self.current_episode,
                query=query,
                design_info=design_info,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                quality_feedback=quality_feedback,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.debug(f"Episode {self.current_episode}: 奖励={reward:.3f}, "
                            f"动作={action.k_value}, 探索率={self.agent.epsilon:.3f}")
            
            return episode_result
            
        except Exception as e:
            self.logger.error(f"训练episode失败: {str(e)}")
            # 返回默认episode结果
            return TrainingEpisode(
                episode_id=self.current_episode,
                query={},
                design_info={},
                state=State(0.0, 'unknown', 0, 0.0, 0.0, 0.0, datetime.now().isoformat()),
                action=Action(3, 0.0, 'error'),
                reward=0.0,
                next_state=State(0.0, 'unknown', 0, 0.0, 0.0, 0.0, datetime.now().isoformat()),
                quality_feedback={},
                timestamp=datetime.now().isoformat()
            )
    
    def _validate(self, validation_data: List[Dict[str, Any]]) -> float:
        """验证模型性能
        
        Args:
            validation_data: 验证数据
            
        Returns:
            float: 验证分数
        """
        validation_scores = []
        
        for data in validation_data[:10]:  # 只验证前10个样本
            try:
                query = data['query']
                design_info = data['design_info']
                
                # 执行推理
                initial_results = self.retriever._initial_retrieval(query, design_info)
                state = self.state_extractor.extract_state_features(query, design_info, initial_results)
                action = self.agent.choose_action(state)
                
                # 生成布局
                final_results = self.retriever.retrieve_with_dynamic_reranking(query, design_info)
                layout_result = self.generator.generate_layout(query, final_results)
                
                # 评估质量
                quality_feedback = self.evaluator.evaluate(layout_result)
                overall_score = quality_feedback.get('overall', 0.0)
                
                validation_scores.append(overall_score)
                
            except Exception as e:
                self.logger.warning(f"验证样本失败: {str(e)}")
                validation_scores.append(0.0)
        
        return np.mean(validation_scores) if validation_scores else 0.0
    
    def _check_convergence(self) -> bool:
        """检查训练是否收敛
        
        Returns:
            bool: 是否收敛
        """
        if len(self.training_history) < self.convergence_window:
            return False
        
        # 计算最近窗口内的平均奖励
        recent_rewards = [ep.reward for ep in self.training_history[-self.convergence_window:]]
        current_avg = np.mean(recent_rewards)
        
        # 计算前一个窗口的平均奖励
        if len(self.training_history) >= 2 * self.convergence_window:
            previous_rewards = [ep.reward for ep in self.training_history[-2*self.convergence_window:-self.convergence_window]]
            previous_avg = np.mean(previous_rewards)
            
            # 检查奖励变化是否小于阈值
            reward_change = abs(current_avg - previous_avg)
            convergence_score = 1.0 - min(reward_change, 1.0)
            
            self.convergence_history.append(convergence_score)
            
            return reward_change < self.convergence_threshold
        
        return False
    
    def _calculate_training_metrics(self, episode: int) -> TrainingMetrics:
        """计算训练指标
        
        Args:
            episode: 当前episode
            
        Returns:
            TrainingMetrics: 训练指标
        """
        # 计算最近100个episode的统计信息
        recent_episodes = self.training_history[-100:] if len(self.training_history) >= 100 else self.training_history
        
        total_reward = sum(ep.reward for ep in recent_episodes)
        average_reward = total_reward / len(recent_episodes) if recent_episodes else 0.0
        exploration_rate = self.agent.epsilon
        
        # 计算收敛分数
        convergence_score = np.mean(self.convergence_history[-10:]) if self.convergence_history else 0.0
        
        # 获取Q表统计
        q_table_stats = self.agent.get_q_table_stats()
        q_table_size = q_table_stats['total_states']
        
        return TrainingMetrics(
            episode=episode,
            total_reward=total_reward,
            average_reward=average_reward,
            exploration_rate=exploration_rate,
            convergence_score=convergence_score,
            q_table_size=q_table_size,
            timestamp=datetime.now().isoformat()
        )
    
    def _save_checkpoint(self, episode: int):
        """保存检查点
        
        Args:
            episode: 当前episode
        """
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"rl_checkpoint_episode_{episode}.pkl"
        
        checkpoint_data = {
            'episode': episode,
            'agent_state': {
                'q_table': dict(self.agent.q_table),
                'epsilon': self.agent.epsilon,
                'training_stats': self.agent.training_stats
            },
            'training_history': self.training_history,
            'convergence_history': self.convergence_history,
            'performance_metrics': self.performance_monitor.get_metrics(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            self.logger.info(f"检查点保存成功: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"检查点保存失败: {str(e)}")
    
    def _finalize_training(self):
        """完成训练"""
        # 保存最终模型
        final_model_path = Path(self.config.get('final_model_path', 'models/rl_final_model.pkl'))
        final_model_path.parent.mkdir(exist_ok=True)
        self.agent.save_model(str(final_model_path))
        
        # 生成训练报告
        self._generate_training_report()
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        self.logger.info("训练完成")
    
    def _generate_training_report(self):
        """生成训练报告"""
        report = {
            'training_summary': {
                'total_episodes': len(self.training_history),
                'final_epsilon': self.agent.epsilon,
                'final_q_table_size': len(self.agent.q_table),
                'convergence_episode': self.current_episode if self._check_convergence() else None
            },
            'performance_metrics': self.performance_monitor.get_metrics(),
            'q_table_stats': self.agent.get_q_table_stats(),
            'training_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = Path(self.config.get('report_path', 'reports/training_report.json'))
        report_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"训练报告保存成功: {report_path}")
        except Exception as e:
            self.logger.error(f"训练报告保存失败: {str(e)}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        if not self.performance_monitor.metrics:
            return
        
        # 提取数据
        episodes = [m.episode for m in self.performance_monitor.metrics]
        rewards = [m.average_reward for m in self.performance_monitor.metrics]
        exploration_rates = [m.exploration_rate for m in self.performance_monitor.metrics]
        convergence_scores = [m.convergence_score for m in self.performance_monitor.metrics]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 平均奖励曲线
        axes[0, 0].plot(episodes, rewards)
        axes[0, 0].set_title('平均奖励')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('平均奖励')
        axes[0, 0].grid(True)
        
        # 探索率曲线
        axes[0, 1].plot(episodes, exploration_rates)
        axes[0, 1].set_title('探索率')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('探索率')
        axes[0, 1].grid(True)
        
        # 收敛分数曲线
        axes[1, 0].plot(episodes, convergence_scores)
        axes[1, 0].set_title('收敛分数')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('收敛分数')
        axes[1, 0].grid(True)
        
        # Q表大小曲线
        q_table_sizes = [m.q_table_size for m in self.performance_monitor.metrics]
        axes[1, 1].plot(episodes, q_table_sizes)
        axes[1, 1].set_title('Q表大小')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('状态数量')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = Path(self.config.get('plot_path', 'reports/training_curves.png'))
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线保存成功: {plot_path}")

class TrainingPerformanceMonitor:
    """训练性能监控器"""
    
    def __init__(self):
        """初始化监控器"""
        self.metrics = []
    
    def record_metrics(self, metrics: TrainingMetrics):
        """记录训练指标
        
        Args:
            metrics: 训练指标
        """
        self.metrics.append(metrics)
    
    def get_metrics(self) -> List[TrainingMetrics]:
        """获取所有指标
        
        Returns:
            List[TrainingMetrics]: 指标列表
        """
        return self.metrics.copy()
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """获取最新指标
        
        Returns:
            Optional[TrainingMetrics]: 最新指标
        """
        return self.metrics[-1] if self.metrics else None 