#!/usr/bin/env python3
"""
强化学习智能体训练脚本
使用生成的训练数据训练Chip-D-RAG系统的RL智能体
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import random
from datetime import datetime

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.rl_trainer import RLTrainer
from modules.core.rl_agent import QLearningAgent, StateExtractor, RewardCalculator
from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever
from modules.core.layout_generator import LayoutGenerator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLTrainerWithData:
    """带数据的强化学习训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.training_data_path = Path(config.get('training_data_path', 'data/training/training_data.json'))
        self.output_dir = Path(config.get('output_dir', 'models/rl_agent'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载训练数据
        self.training_data = self._load_training_data()
        
        # 初始化组件
        self._init_components()
        
        logger.info(f"训练器初始化完成，训练数据大小: {len(self.training_data)}")
    
    def _load_training_data(self) -> List[Dict[str, Any]]:
        """加载训练数据
        
        Returns:
            List[Dict[str, Any]]: 训练数据列表
        """
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {self.training_data_path}")
        
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        logger.info(f"成功加载 {len(training_data)} 个训练样本")
        return training_data
    
    def _init_components(self):
        """初始化训练组件"""
        # 强化学习配置
        rl_config = {
            'alpha': 0.01,
            'gamma': 0.95,
            'epsilon': 0.9,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'k_range': (3, 15),
            'buffer_size': 1000
        }
        
        # 检索器配置
        retriever_config = {
            'dynamic_k_range': [3, 15],
            'quality_threshold': 0.7,
            'learning_rate': 0.01,
            'compressed_entity_dim': 128,
            'entity_compression_ratio': 0.1,
            'entity_similarity_threshold': 0.8,
            'knowledge_base': {
                'path': 'data/knowledge_base/sample_kb.json',
                'format': 'json'
            }
        }
        
        # 初始化组件
        self.agent = QLearningAgent(rl_config)
        self.state_extractor = StateExtractor(rl_config)
        self.reward_calculator = RewardCalculator(rl_config)
        self.retriever = DynamicRAGRetriever(retriever_config)
        self.generator = LayoutGenerator(retriever_config)
        self.evaluator = MultiObjectiveEvaluator(retriever_config)
        
        logger.info("所有训练组件初始化完成")
    
    def train(self, num_episodes: int = 1000, batch_size: int = 32):
        """开始训练
        
        Args:
            num_episodes: 训练episodes数量
            batch_size: 批量大小
        """
        logger.info(f"开始训练，总episodes: {num_episodes}")
        
        # 训练历史
        training_history = []
        episode_rewards = []
        exploration_rates = []
        
        # 训练循环
        for episode in range(num_episodes):
            # 随机选择训练样本
            episode_data = random.choice(self.training_data)
            
            # 训练一个episode
            episode_result = self._train_episode(episode_data)
            
            # 记录训练历史
            training_history.append(episode_result)
            episode_rewards.append(episode_result['reward'])
            exploration_rates.append(self.agent.epsilon)
            
            # 批量更新
            if episode % batch_size == 0:
                self.agent.batch_update(batch_size)
            
            # 定期保存模型
            if episode % 100 == 0:
                self._save_checkpoint(episode)
                logger.info(f"Episode {episode}: 平均奖励 = {np.mean(episode_rewards[-100:]):.3f}, 探索率 = {self.agent.epsilon:.3f}")
            
            # 检查收敛性
            if self._check_convergence(episode_rewards):
                logger.info(f"训练在第 {episode} 个episode收敛")
                break
        
        # 训练完成
        self._finalize_training(training_history)
    
    def _train_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """训练一个episode
        
        Args:
            episode_data: episode数据
            
        Returns:
            Dict[str, Any]: episode结果
        """
        try:
            query = episode_data['query']
            design_info = episode_data['design_info']
            expected_quality = episode_data['expected_quality']
            
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
            
            # 7. 计算奖励（基于与期望质量的比较）
            reward = self._calculate_reward(quality_feedback, expected_quality)
            
            # 8. 提取下一状态
            next_state = self.state_extractor.extract_state_features(
                query, design_info, final_results
            )
            
            # 9. 更新智能体
            self.agent.update(state, action, reward, next_state)
            
            # 10. 记录episode结果
            episode_result = {
                'episode_id': len(training_history) if 'training_history' in locals() else 0,
                'query_id': query.get('id', 'unknown'),
                'design_type': design_info.get('design_type', 'unknown'),
                'action_k': action.k_value,
                'reward': reward,
                'quality_score': quality_feedback.get('overall_score', 0.0),
                'expected_quality': expected_quality.get('overall_score', 0.0),
                'exploration_rate': self.agent.epsilon,
                'timestamp': datetime.now().isoformat()
            }
            
            return episode_result
            
        except Exception as e:
            logger.error(f"训练episode失败: {str(e)}")
            return {
                'episode_id': len(training_history) if 'training_history' in locals() else 0,
                'error': str(e),
                'reward': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_reward(self, quality_feedback: Dict[str, Any], expected_quality: Dict[str, Any]) -> float:
        """计算奖励
        
        Args:
            quality_feedback: 实际质量反馈
            expected_quality: 期望质量
            
        Returns:
            float: 奖励值
        """
        try:
            actual_score = quality_feedback.get('overall_score', 0.0)
            expected_score = expected_quality.get('overall_score', 0.0)
            
            # 基于质量差异计算奖励
            quality_diff = actual_score - expected_score
            
            # 奖励函数：质量越高奖励越大，但超过期望质量有额外奖励
            if quality_diff >= 0:
                reward = 1.0 + quality_diff * 2.0  # 超过期望的额外奖励
            else:
                reward = max(0.0, 1.0 + quality_diff)  # 低于期望的惩罚
            
            return reward
            
        except Exception as e:
            logger.error(f"计算奖励失败: {str(e)}")
            return 0.0
    
    def _check_convergence(self, episode_rewards: List[float], window: int = 100) -> bool:
        """检查训练是否收敛
        
        Args:
            episode_rewards: episode奖励列表
            window: 检查窗口大小
            
        Returns:
            bool: 是否收敛
        """
        if len(episode_rewards) < window:
            return False
        
        recent_rewards = episode_rewards[-window:]
        reward_std = np.std(recent_rewards)
        
        # 如果最近奖励的标准差很小，认为收敛
        return reward_std < 0.01
    
    def _save_checkpoint(self, episode: int):
        """保存检查点
        
        Args:
            episode: episode编号
        """
        checkpoint = {
            'episode': episode,
            'agent_state': self.agent.get_state(),
            'training_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.output_dir / f'checkpoint_episode_{episode}.json'
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检查点已保存: {checkpoint_file}")
    
    def _finalize_training(self, training_history: List[Dict[str, Any]]):
        """完成训练
        
        Args:
            training_history: 训练历史
        """
        # 保存最终模型
        final_model = {
            'agent_state': self.agent.get_state(),
            'training_history': training_history,
            'final_config': self.config,
            'training_stats': self._calculate_training_stats(training_history),
            'timestamp': datetime.now().isoformat()
        }
        
        model_file = self.output_dir / 'final_model.json'
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(final_model, f, ensure_ascii=False, indent=2)
        
        # 生成训练报告
        self._generate_training_report(training_history)
        
        logger.info(f"训练完成，最终模型已保存: {model_file}")
    
    def _calculate_training_stats(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算训练统计
        
        Args:
            training_history: 训练历史
            
        Returns:
            Dict[str, Any]: 训练统计
        """
        if not training_history:
            return {}
        
        rewards = [ep['reward'] for ep in training_history if 'reward' in ep]
        quality_scores = [ep['quality_score'] for ep in training_history if 'quality_score' in ep]
        expected_qualities = [ep['expected_quality'] for ep in training_history if 'expected_quality' in ep]
        
        stats = {
            'total_episodes': len(training_history),
            'reward_statistics': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'quality_statistics': {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'quality_improvement': {
                'mean': np.mean([q - e for q, e in zip(quality_scores, expected_qualities)]),
                'std': np.std([q - e for q, e in zip(quality_scores, expected_qualities)])
            },
            'final_exploration_rate': self.agent.epsilon
        }
        
        return stats
    
    def _generate_training_report(self, training_history: List[Dict[str, Any]]):
        """生成训练报告
        
        Args:
            training_history: 训练历史
        """
        report = {
            'training_summary': {
                'total_episodes': len(training_history),
                'training_duration': 'N/A',  # 可以添加实际训练时间
                'final_exploration_rate': self.agent.epsilon,
                'convergence_achieved': self._check_convergence([ep['reward'] for ep in training_history if 'reward' in ep])
            },
            'performance_metrics': self._calculate_training_stats(training_history),
            'design_type_performance': self._analyze_design_type_performance(training_history),
            'recommendations': self._generate_recommendations(training_history)
        }
        
        report_file = self.output_dir / 'training_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练报告已生成: {report_file}")
    
    def _analyze_design_type_performance(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析不同设计类型的性能
        
        Args:
            training_history: 训练历史
            
        Returns:
            Dict[str, Any]: 设计类型性能分析
        """
        design_type_stats = {}
        
        for episode in training_history:
            design_type = episode.get('design_type', 'unknown')
            if design_type not in design_type_stats:
                design_type_stats[design_type] = {
                    'count': 0,
                    'rewards': [],
                    'quality_scores': []
                }
            
            design_type_stats[design_type]['count'] += 1
            if 'reward' in episode:
                design_type_stats[design_type]['rewards'].append(episode['reward'])
            if 'quality_score' in episode:
                design_type_stats[design_type]['quality_scores'].append(episode['quality_score'])
        
        # 计算统计信息
        for design_type, stats in design_type_stats.items():
            if stats['rewards']:
                stats['avg_reward'] = np.mean(stats['rewards'])
                stats['avg_quality'] = np.mean(stats['quality_scores'])
            else:
                stats['avg_reward'] = 0.0
                stats['avg_quality'] = 0.0
        
        return design_type_stats
    
    def _generate_recommendations(self, training_history: List[Dict[str, Any]]) -> List[str]:
        """生成训练建议
        
        Args:
            training_history: 训练历史
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 分析训练历史
        rewards = [ep['reward'] for ep in training_history if 'reward' in ep]
        quality_scores = [ep['quality_score'] for ep in training_history if 'quality_score' in ep]
        
        if len(rewards) > 0:
            avg_reward = np.mean(rewards)
            avg_quality = np.mean(quality_scores)
            
            if avg_reward < 0.5:
                recommendations.append("平均奖励较低，建议增加训练episodes或调整奖励函数")
            
            if avg_quality < 0.7:
                recommendations.append("平均质量分数较低，建议优化检索策略或增加知识库内容")
            
            if self.agent.epsilon > 0.1:
                recommendations.append("探索率仍然较高，建议继续训练以降低探索率")
        
        if len(recommendations) == 0:
            recommendations.append("训练表现良好，可以考虑在实际环境中部署")
        
        return recommendations

def main():
    """主函数"""
    # 配置
    config = {
        'training_data_path': 'data/training/training_data.json',
        'output_dir': 'models/rl_agent',
        'num_episodes': 500,
        'batch_size': 32,
        'random_seed': 42
    }
    
    # 设置随机种子
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # 初始化训练器
    trainer = RLTrainerWithData(config)
    
    # 开始训练
    logger.info("开始强化学习智能体训练...")
    trainer.train(config['num_episodes'], config['batch_size'])
    
    logger.info("训练完成！")

if __name__ == '__main__':
    main() 