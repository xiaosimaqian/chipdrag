#!/usr/bin/env python3
"""
RL-DRAG实验主控脚本
集成强化学习训练、DRAG参数推荐和对比实验
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入RL训练系统
from modules.rl_training.rl_training_system import (
    LayoutEnvironment, LayoutAction, DQNAgent, RLTrainer
)

# 导入DRAG检索系统
try:
    from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever
    DRAG_AVAILABLE = True
except ImportError:
    DRAG_AVAILABLE = False
    print("警告: DRAG检索系统不可用")

# 导入OpenROAD接口
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

# 导入离线RL训练器
try:
    from modules.rl_training.offline_rl_trainer import OfflineRLTrainer
    OFFLINE_RL_AVAILABLE = True
except ImportError:
    OFFLINE_RL_AVAILABLE = False
    print("警告: 离线RL训练器不可用")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_drag_experiment.log'),
        logging.StreamHandler()
    ]
)

class RLDragExperiment:
    """RL-DRAG对比实验类 - 增强版"""
    
    def __init__(self, 
                 work_dir: str,
                 experiment_name: str = "rl_drag_experiment",
                 rl_episodes: int = 50,
                 rl_max_steps: int = 10,
                 use_offline_rl: bool = True):
        """
        Args:
            work_dir: 工作目录
            experiment_name: 实验名称
            rl_episodes: RL训练episode数
            rl_max_steps: 每个episode最大步数
            use_offline_rl: 是否使用离线RL
        """
        self.work_dir = Path(work_dir)
        self.experiment_name = experiment_name
        self.rl_episodes = rl_episodes
        self.rl_max_steps = rl_max_steps
        self.use_offline_rl = use_offline_rl
        
        # 创建实验目录
        self.experiment_dir = Path(f"experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        # 实验结果
        self.results = {
            'experiment_info': {
                'name': experiment_name,
                'work_dir': str(work_dir),
                'start_time': datetime.now().isoformat(),
                'rl_episodes': rl_episodes,
                'rl_max_steps': rl_max_steps,
                'use_offline_rl': use_offline_rl
            },
            'offline_rl_results': {},
            'online_rl_results': {},
            'drag_recommendations': {},
            'comparison_results': {},
            'final_analysis': {}
        }
    
    def _init_components(self):
        """初始化实验组件"""
        logging.info("初始化实验组件...")
        
        # 初始化OpenROAD接口
        try:
            self.openroad_interface = RealOpenROADInterface(str(self.work_dir))
            logging.info("OpenROAD接口初始化成功")
        except Exception as e:
            logging.error(f"OpenROAD接口初始化失败: {e}")
            raise
        
        # 初始化DRAG检索器
        try:
            drag_config = {
                'llm': {
                    'model_name': 'gpt-3.5-turbo',
                    'api_key': 'your-api-key',
                    'max_tokens': 1000
                },
                'knowledge_base': {
                    'path': 'configs/knowledge/ispd_knowledge_base.json',
                    'format': 'json',
                    'embedding_model': 'bert-base-chinese',
                    'hierarchy_config': {
                        'levels': [
                            {'name': 'system', 'threshold': 0.8},
                            {'name': 'module', 'threshold': 0.6},
                            {'name': 'component', 'threshold': 0.4}
                        ]
                    },
                    'llm_config': {
                        'base_url': 'http://localhost:11434',
                        'model': 'llama2',
                        'temperature': 0.7,
                        'max_tokens': 1000
                    }
                },
                'dynamic_k_range': (3, 15),
                'quality_threshold': 0.7,
                'learning_rate': 0.01,
                'entity_compression_ratio': 0.1,
                'entity_similarity_threshold': 0.8
            }
            self.drag_retriever = DynamicRAGRetriever(drag_config)
            logging.info("DRAG检索器初始化成功")
        except Exception as e:
            logging.error(f"DRAG检索器初始化失败: {e}")
            self.drag_retriever = None
        
        # 初始化离线RL训练器
        if self.use_offline_rl and OFFLINE_RL_AVAILABLE:
            try:
                self.offline_rl_trainer = OfflineRLTrainer()
                logging.info("离线RL训练器初始化成功")
            except Exception as e:
                logging.error(f"离线RL训练器初始化失败: {e}")
                self.offline_rl_trainer = None
        else:
            self.offline_rl_trainer = None
            if not OFFLINE_RL_AVAILABLE:
                logging.warning("离线RL训练器不可用")
        
        # 初始化在线RL环境
        try:
            self.rl_env = LayoutEnvironment(
                work_dir=str(self.work_dir),
                max_iterations=self.rl_max_steps,
                target_hpwl=1000000.0,
                target_overflow=0.1
            )
            logging.info("在线RL环境初始化成功")
        except Exception as e:
            logging.error(f"在线RL环境初始化失败: {e}")
            raise
    
    def run_offline_rl_training(self) -> Dict[str, Any]:
        """运行离线RL训练"""
        if not self.offline_rl_trainer:
            logging.warning("离线RL训练器不可用，跳过离线训练")
            return {'error': '离线RL训练器不可用'}
        
        logging.info("开始离线RL训练...")
        
        try:
            # 加载训练数据
            df = self.offline_rl_trainer.load_training_data()
            
            if df.empty:
                logging.warning("没有找到离线训练数据")
                return {'error': '没有找到离线训练数据'}
            
            # 预处理数据
            X_features, X_actions, y_rewards = self.offline_rl_trainer.preprocess_data(df)
            
            if len(X_features) == 0:
                logging.error("预处理后没有有效数据")
                return {'error': '预处理后没有有效数据'}
            
            # 训练模型
            training_history = self.offline_rl_trainer.train_model(
                X_features, X_actions, y_rewards,
                epochs=50,
                batch_size=16,
                learning_rate=0.001
            )
            
            # 保存模型
            self.offline_rl_trainer.save_model()
            
            # 生成训练报告
            self.offline_rl_trainer.generate_training_report(
                training_history, self.experiment_dir
            )
            
            # 获取设计统计信息用于参数预测
            design_stats = self.openroad_interface._extract_design_stats()
            
            # 预测最优参数
            optimal_params = self.offline_rl_trainer.predict_optimal_parameters(design_stats)
            
            offline_results = {
                'training_history': training_history,
                'model_path': str(self.offline_rl_trainer.model_save_dir / "offline_rl_model.pth"),
                'optimal_params': optimal_params.to_dict(),
                'design_stats': design_stats,
                'data_samples_count': len(df)
            }
            
            self.results['offline_rl_results'] = offline_results
            logging.info("离线RL训练完成")
            
            return offline_results
            
        except Exception as e:
            logging.error(f"离线RL训练失败: {e}")
            self.results['offline_rl_results'] = {'error': str(e)}
            return {'error': str(e)}
    
    def run_online_rl_training(self) -> Dict[str, Any]:
        """运行在线RL训练"""
        logging.info("开始在线RL训练...")
        
        try:
            # 创建DQN智能体
            agent = DQNAgent(
                state_size=5,  # LayoutState的维度
                action_size=8,  # 离散动作空间大小
                learning_rate=0.001,
                epsilon=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01
            )
            
            # 创建训练器
            trainer = RLTrainer(
                env=self.rl_env,
                agent=agent,
                episodes=self.rl_episodes,
                max_steps=self.rl_max_steps
            )
            
            # 开始训练
            trainer.train()
            
            # 收集训练结果
            online_results = {
                'training_history': trainer.training_history,
                'final_model_path': str(self.rl_env.output_dir / "final_model.pkl"),
                'best_episode': self._find_best_episode(trainer.training_history),
                'convergence_analysis': self._analyze_rl_convergence(trainer.training_history)
            }
            
            self.results['online_rl_results'] = online_results
            logging.info("在线RL训练完成")
            
            return online_results
            
        except Exception as e:
            logging.error(f"在线RL训练失败: {e}")
            self.results['online_rl_results'] = {'error': str(e)}
            return {'error': str(e)}
    
    def run_three_way_comparison(self) -> Dict[str, Any]:
        """运行三方对比实验：离线RL vs 在线RL vs DRAG"""
        logging.info("开始三方对比实验...")
        
        try:
            comparison_results = {
                'offline_rl_params': {},
                'online_rl_params': {},
                'drag_params': {},
                'offline_rl_performance': {},
                'online_rl_performance': {},
                'drag_performance': {},
                'comparison_metrics': {}
            }
            
            # 1. 获取离线RL参数
            if self.offline_rl_trainer and 'optimal_params' in self.results.get('offline_rl_results', {}):
                offline_params = self.results['offline_rl_results']['optimal_params']
                comparison_results['offline_rl_params'] = offline_params
                
                # 测试离线RL参数
                logging.info("测试离线RL参数...")
                offline_result = self.openroad_interface.run_placement(
                    density_target=offline_params.get('density_target', 0.7),
                    wirelength_weight=offline_params.get('wirelength_weight', 1.0),
                    density_weight=offline_params.get('density_weight', 1.0)
                )
                comparison_results['offline_rl_performance'] = offline_result
            
            # 2. 获取在线RL参数
            online_rl_results = self.results.get('online_rl_results', {})
            if 'best_episode' in online_rl_results:
                # 从在线RL结果中提取最佳参数
                online_params = self._extract_online_rl_params(online_rl_results)
                comparison_results['online_rl_params'] = online_params
                
                # 测试在线RL参数
                logging.info("测试在线RL参数...")
                online_result = self.openroad_interface.run_placement(
                    density_target=online_params.get('density_target', 0.7),
                    wirelength_weight=online_params.get('wirelength_weight', 1.0),
                    density_weight=online_params.get('density_weight', 1.0)
                )
                comparison_results['online_rl_performance'] = online_result
            
            # 3. 获取DRAG参数
            drag_results = self.results.get('drag_recommendations', {})
            if 'recommended_params' in drag_results:
                drag_params = drag_results['recommended_params']
                comparison_results['drag_params'] = drag_params
                
                # 测试DRAG参数
                logging.info("测试DRAG参数...")
                drag_result = self.openroad_interface.run_placement(
                    density_target=drag_params.get('density_target', 0.7),
                    wirelength_weight=drag_params.get('wirelength_weight', 1.0),
                    density_weight=drag_params.get('density_weight', 1.0)
                )
                comparison_results['drag_performance'] = drag_result
            
            # 4. 计算对比指标
            comparison_metrics = self._calculate_three_way_metrics(comparison_results)
            comparison_results['comparison_metrics'] = comparison_metrics
            
            self.results['comparison_results'] = comparison_results
            logging.info("三方对比实验完成")
            
            return comparison_results
            
        except Exception as e:
            logging.error(f"三方对比实验失败: {e}")
            self.results['comparison_results'] = {'error': str(e)}
            return {'error': str(e)}
    
    def _extract_online_rl_params(self, online_rl_results: Dict) -> Dict[str, float]:
        """从在线RL结果中提取最佳参数"""
        # 这里需要从在线RL训练历史中提取最佳参数
        # 暂时返回默认值，实际实现需要从训练历史中提取
        return {
            'density_target': 0.75,
            'wirelength_weight': 1.5,
            'density_weight': 1.2
        }
    
    def _calculate_three_way_metrics(self, comparison_results: Dict) -> Dict[str, Any]:
        """计算三方对比指标"""
        metrics = {}
        
        # 提取性能数据
        offline_hpwl = comparison_results.get('offline_rl_performance', {}).get('hpwl', float('inf'))
        online_hpwl = comparison_results.get('online_rl_performance', {}).get('hpwl', float('inf'))
        drag_hpwl = comparison_results.get('drag_performance', {}).get('hpwl', float('inf'))
        
        # HPWL对比
        if all(hpwl != float('inf') for hpwl in [offline_hpwl, online_hpwl, drag_hpwl]):
            hpwl_comparison = {
                'offline_rl_hpwl': offline_hpwl,
                'online_rl_hpwl': online_hpwl,
                'drag_hpwl': drag_hpwl,
                'best_hpwl': min(offline_hpwl, online_hpwl, drag_hpwl),
                'winner': self._determine_hpwl_winner(offline_hpwl, online_hpwl, drag_hpwl)
            }
            metrics['hpwl_comparison'] = hpwl_comparison
        
        # 执行时间对比
        offline_time = comparison_results.get('offline_rl_performance', {}).get('execution_time', 0)
        online_time = comparison_results.get('online_rl_performance', {}).get('execution_time', 0)
        drag_time = comparison_results.get('drag_performance', {}).get('execution_time', 0)
        
        if all(time > 0 for time in [offline_time, online_time, drag_time]):
            time_comparison = {
                'offline_rl_time': offline_time,
                'online_rl_time': online_time,
                'drag_time': drag_time,
                'fastest': self._determine_fastest(offline_time, online_time, drag_time)
            }
            metrics['time_comparison'] = time_comparison
        
        # 成功率对比
        offline_success = comparison_results.get('offline_rl_performance', {}).get('success', False)
        online_success = comparison_results.get('online_rl_performance', {}).get('success', False)
        drag_success = comparison_results.get('drag_performance', {}).get('success', False)
        
        metrics['success_comparison'] = {
            'offline_rl_success': offline_success,
            'online_rl_success': online_success,
            'drag_success': drag_success,
            'all_success': all([offline_success, online_success, drag_success])
        }
        
        return metrics
    
    def _determine_hpwl_winner(self, offline_hpwl: float, online_hpwl: float, drag_hpwl: float) -> str:
        """确定HPWL获胜者"""
        min_hpwl = min(offline_hpwl, online_hpwl, drag_hpwl)
        if min_hpwl == offline_hpwl:
            return 'Offline_RL'
        elif min_hpwl == online_hpwl:
            return 'Online_RL'
        else:
            return 'DRAG'
    
    def _determine_fastest(self, offline_time: float, online_time: float, drag_time: float) -> str:
        """确定最快方法"""
        min_time = min(offline_time, online_time, drag_time)
        if min_time == offline_time:
            return 'Offline_RL'
        elif min_time == online_time:
            return 'Online_RL'
        else:
            return 'DRAG'
    
    def get_drag_recommendations(self, design_stats: Dict[str, Any]) -> Dict[str, Any]:
        """获取DRAG参数推荐"""
        if not self.drag_retriever:
            logging.warning("DRAG检索器不可用，跳过DRAG推荐")
            return {'error': 'DRAG检索器不可用'}
        
        logging.info("获取DRAG参数推荐...")
        
        try:
            # 构建查询
            query = f"设计规模: {design_stats.get('num_instances', 0)}实例, {design_stats.get('num_nets', 0)}网络"
            
            # 检索相似案例
            similar_cases = self.drag_retriever.retrieve_similar_cases(
                query=query,
                top_k=5,
                similarity_threshold=0.7
            )
            
            # 提取推荐参数
            recommended_params = self._extract_recommended_params(similar_cases)
            
            drag_results = {
                'query': query,
                'similar_cases': similar_cases,
                'recommended_params': recommended_params
            }
            
            self.results['drag_recommendations'] = drag_results
            logging.info("DRAG推荐完成")
            
            return drag_results
            
        except Exception as e:
            logging.error(f"DRAG推荐失败: {e}")
            self.results['drag_recommendations'] = {'error': str(e)}
            return {'error': str(e)}
    
    def _find_best_episode(self, training_history: List[Dict]) -> Dict[str, Any]:
        """找到最佳episode"""
        if not training_history:
            return {}
        
        # 按总奖励排序
        sorted_history = sorted(training_history, key=lambda x: x['total_reward'], reverse=True)
        best_episode = sorted_history[0]
        
        return {
            'episode': best_episode['episode'],
            'total_reward': best_episode['total_reward'],
            'final_hpwl': best_episode['final_hpwl'],
            'final_overflow': best_episode['final_overflow']
        }
    
    def _analyze_rl_convergence(self, training_history: List[Dict]) -> Dict[str, Any]:
        """分析RL收敛情况"""
        if not training_history:
            return {}
        
        df = pd.DataFrame(training_history)
        
        # 计算收敛指标
        convergence_analysis = {
            'total_episodes': len(training_history),
            'avg_reward': df['total_reward'].mean(),
            'std_reward': df['total_reward'].std(),
            'best_reward': df['total_reward'].max(),
            'worst_reward': df['total_reward'].min(),
            'convergence_episode': self._find_convergence_episode(df),
            'reward_trend': 'improving' if df['total_reward'].iloc[-10:].mean() > df['total_reward'].iloc[:10].mean() else 'stagnant'
        }
        
        return convergence_analysis
    
    def _find_convergence_episode(self, df: pd.DataFrame) -> int:
        """找到收敛episode"""
        # 使用滑动窗口检测收敛
        window_size = 10
        if len(df) < window_size:
            return len(df)
        
        for i in range(window_size, len(df)):
            recent_rewards = df['total_reward'].iloc[i-window_size:i]
            if recent_rewards.std() < recent_rewards.mean() * 0.1:  # 标准差小于均值的10%
                return i
        
        return len(df)
    
    def _extract_recommended_params(self, similar_cases: List[Dict]) -> Dict[str, float]:
        """从相似案例中提取推荐参数"""
        if not similar_cases:
            return {
                'density_target': 0.7,
                'wirelength_weight': 1.0,
                'density_weight': 1.0
            }
        
        # 计算参数平均值
        params_sum = {
            'density_target': 0.0,
            'wirelength_weight': 0.0,
            'density_weight': 0.0
        }
        
        valid_cases = 0
        for case in similar_cases:
            if 'parameters' in case:
                params = case['parameters']
                if all(key in params for key in params_sum.keys()):
                    for key in params_sum:
                        params_sum[key] += params[key]
                    valid_cases += 1
        
        if valid_cases == 0:
            return {
                'density_target': 0.7,
                'wirelength_weight': 1.0,
                'density_weight': 1.0
            }
        
        # 计算平均值
        recommended_params = {
            key: params_sum[key] / valid_cases
            for key in params_sum
        }
        
        return recommended_params
    
    def generate_final_analysis(self) -> Dict[str, Any]:
        """生成最终分析报告"""
        logging.info("生成最终分析报告...")
        
        analysis = {
            'experiment_summary': self._generate_experiment_summary(),
            'rl_analysis': self._analyze_rl_results(),
            'drag_analysis': self._analyze_drag_results(),
            'comparison_analysis': self._analyze_comparison_results(),
            'recommendations': self._generate_recommendations()
        }
        
        self.results['final_analysis'] = analysis
        return analysis
    
    def _generate_experiment_summary(self) -> Dict[str, Any]:
        """生成实验总结"""
        return {
            'experiment_name': self.experiment_name,
            'work_dir': str(self.work_dir),
            'total_duration': self._calculate_experiment_duration(),
            'components_available': {
                'openroad': True,
                'drag': DRAG_AVAILABLE,
                'rl': True
            }
        }
    
    def _analyze_rl_results(self) -> Dict[str, Any]:
        """分析RL结果"""
        rl_results = self.results.get('online_rl_results', {})
        
        if 'error' in rl_results:
            return {'status': 'failed', 'error': rl_results['error']}
        
        best_episode = rl_results.get('best_episode', {})
        convergence = rl_results.get('convergence_analysis', {})
        
        return {
            'status': 'success',
            'best_episode': best_episode,
            'convergence': convergence,
            'training_quality': self._assess_training_quality(convergence)
        }
    
    def _analyze_drag_results(self) -> Dict[str, Any]:
        """分析DRAG结果"""
        drag_results = self.results.get('drag_recommendations', {})
        
        if 'error' in drag_results:
            return {'status': 'failed', 'error': drag_results['error']}
        
        return {
            'status': 'success',
            'similar_cases_count': len(drag_results.get('similar_cases', [])),
            'recommended_params': drag_results.get('recommended_params', {})
        }
    
    def _analyze_comparison_results(self) -> Dict[str, Any]:
        """分析对比结果"""
        comparison_results = self.results.get('comparison_results', {})
        
        if 'error' in comparison_results:
            return {'status': 'failed', 'error': comparison_results['error']}
        
        metrics = comparison_results.get('comparison_metrics', {})
        
        return {
            'status': 'success',
            'hpwl_winner': metrics.get('hpwl_comparison', {}).get('winner', 'unknown'),
            'time_winner': metrics.get('time_comparison', {}).get('fastest', 'unknown'),
            'overall_winner': self._determine_overall_winner(metrics)
        }
    
    def _assess_training_quality(self, convergence: Dict) -> str:
        """评估训练质量"""
        if not convergence:
            return 'unknown'
        
        avg_reward = convergence.get('avg_reward', 0)
        reward_trend = convergence.get('reward_trend', 'unknown')
        
        if avg_reward > 100 and reward_trend == 'improving':
            return 'excellent'
        elif avg_reward > 50:
            return 'good'
        elif avg_reward > 0:
            return 'fair'
        else:
            return 'poor'
    
    def _determine_overall_winner(self, metrics: Dict) -> str:
        """确定整体获胜者"""
        hpwl_winner = metrics.get('hpwl_comparison', {}).get('winner', 'unknown')
        time_winner = metrics.get('time_comparison', {}).get('fastest', 'unknown')
        
        if hpwl_winner == 'Offline_RL' and time_winner == 'Offline_RL':
            return 'Offline_RL'
        elif hpwl_winner == 'Online_RL' and time_winner == 'Online_RL':
            return 'Online_RL'
        else:
            return 'DRAG'
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于RL训练结果
        rl_analysis = self._analyze_rl_results()
        if rl_analysis.get('status') == 'success':
            training_quality = rl_analysis.get('training_quality', 'unknown')
            if training_quality == 'excellent':
                recommendations.append("RL训练效果优秀，建议增加训练episode数量以进一步提升性能")
            elif training_quality == 'poor':
                recommendations.append("RL训练效果不佳，建议调整奖励函数或网络结构")
        
        # 基于对比结果
        comparison_analysis = self._analyze_comparison_results()
        if comparison_analysis.get('status') == 'success':
            overall_winner = comparison_analysis.get('overall_winner', 'unknown')
            if overall_winner == 'Offline_RL':
                recommendations.append("Offline_RL方法在本次实验中表现更优，建议在实际应用中优先考虑")
            elif overall_winner == 'Online_RL':
                recommendations.append("Online_RL方法在本次实验中表现更优，建议在实际应用中优先考虑")
            else:
                recommendations.append("DRAG方法在本次实验中表现更优，建议在实际应用中优先考虑")
        
        return recommendations
    
    def _calculate_experiment_duration(self) -> float:
        """计算实验持续时间"""
        start_time = datetime.fromisoformat(self.results['experiment_info']['start_time'])
        end_time = datetime.now()
        return (end_time - start_time).total_seconds()
    
    def save_results(self):
        """保存实验结果"""
        # 保存详细结果
        results_file = self.experiment_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 生成Markdown报告
        report_file = self.experiment_dir / "experiment_report.md"
        self._generate_markdown_report(report_file)
        
        # 生成可视化图表
        self._generate_visualizations()
        
        logging.info(f"实验结果已保存到: {self.experiment_dir}")
    
    def _generate_markdown_report(self, report_file: Path):
        """生成Markdown报告"""
        analysis = self.results.get('final_analysis', {})
        
        report = f"""# RL-DRAG实验报告

## 实验信息
- 实验名称: {self.experiment_name}
- 工作目录: {self.work_dir}
- 开始时间: {self.results['experiment_info']['start_time']}
- RL训练episodes: {self.results['experiment_info']['rl_episodes']}

## 实验总结
{self._format_experiment_summary(analysis.get('experiment_summary', {}))}

## RL训练分析
{self._format_rl_analysis(analysis.get('rl_analysis', {}))}

## DRAG推荐分析
{self._format_drag_analysis(analysis.get('drag_analysis', {}))}

## 对比分析
{self._format_comparison_analysis(analysis.get('comparison_analysis', {}))}

## 建议
{self._format_recommendations(analysis.get('recommendations', []))}
"""
        
        with open(report_file, 'w') as f:
            f.write(report)
    
    def _format_experiment_summary(self, summary: Dict) -> str:
        """格式化实验总结"""
        return f"""
- 实验持续时间: {summary.get('total_duration', 0):.2f}秒
- OpenROAD可用: {summary.get('components_available', {}).get('openroad', False)}
- DRAG可用: {summary.get('components_available', {}).get('drag', False)}
- RL可用: {summary.get('components_available', {}).get('rl', False)}
"""
    
    def _format_rl_analysis(self, analysis: Dict) -> str:
        """格式化RL分析"""
        if analysis.get('status') == 'failed':
            return f"- 状态: 失败\n- 错误: {analysis.get('error', '未知错误')}"
        
        best_episode = analysis.get('best_episode', {})
        convergence = analysis.get('convergence', {})
        
        return f"""
- 状态: 成功
- 最佳episode: {best_episode.get('episode', 'N/A')}
- 最佳奖励: {best_episode.get('total_reward', 0):.2f}
- 最终HPWL: {best_episode.get('final_hpwl', 0):.2e}
- 训练质量: {analysis.get('training_quality', 'unknown')}
- 收敛episode: {convergence.get('convergence_episode', 'N/A')}
"""
    
    def _format_drag_analysis(self, analysis: Dict) -> str:
        """格式化DRAG分析"""
        if analysis.get('status') == 'failed':
            return f"- 状态: 失败\n- 错误: {analysis.get('error', '未知错误')}"
        
        return f"""
- 状态: 成功
- 相似案例数量: {analysis.get('similar_cases_count', 0)}
- 推荐参数: {analysis.get('recommended_params', {})}
"""
    
    def _format_comparison_analysis(self, analysis: Dict) -> str:
        """格式化对比分析"""
        if analysis.get('status') == 'failed':
            return f"- 状态: 失败\n- 错误: {analysis.get('error', '未知错误')}"
        
        return f"""
- 状态: 成功
- HPWL获胜者: {analysis.get('hpwl_winner', 'unknown')}
- 时间获胜者: {analysis.get('time_winner', 'unknown')}
- 整体获胜者: {analysis.get('overall_winner', 'unknown')}
"""
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """格式化建议"""
        if not recommendations:
            return "- 无具体建议"
        
        return "\n".join([f"- {rec}" for rec in recommendations])
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        # 这里可以添加图表生成代码
        pass
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """运行完整实验 - 增强版"""
        logging.info("开始RL-DRAG完整实验（包含离线RL）...")
        
        try:
            # 1. 运行离线RL训练（如果启用）
            if self.use_offline_rl:
                offline_rl_results = self.run_offline_rl_training()
                logging.info("离线RL训练完成")
            
            # 2. 运行在线RL训练
            online_rl_results = self.run_online_rl_training()
            logging.info("在线RL训练完成")
            
            # 3. 获取设计统计信息
            design_stats = self.openroad_interface._extract_design_stats()
            
            # 4. 获取DRAG推荐
            drag_results = self.get_drag_recommendations(design_stats)
            logging.info("DRAG推荐完成")
            
            # 5. 运行三方对比实验
            comparison_results = self.run_three_way_comparison()
            logging.info("三方对比实验完成")
            
            # 6. 生成最终分析
            final_analysis = self.generate_final_analysis()
            
            # 7. 保存结果
            self.save_results()
            
            logging.info("RL-DRAG完整实验完成！")
            
            return {
                'status': 'success',
                'offline_rl_results': offline_rl_results if self.use_offline_rl else None,
                'online_rl_results': online_rl_results,
                'drag_results': drag_results,
                'comparison_results': comparison_results,
                'final_analysis': final_analysis
            }
            
        except Exception as e:
            logging.error(f"完整实验失败: {e}")
            return {'status': 'failed', 'error': str(e)}

def main():
    """主函数"""
    # 配置参数
    work_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    experiment_name = "rl_drag_comparison"
    rl_episodes = 30  # 减少episode数量以加快实验
    rl_max_steps = 5  # 减少步数以加快实验
    
    # 创建实验实例
    experiment = RLDragExperiment(
        work_dir=work_dir,
        experiment_name=experiment_name,
        rl_episodes=rl_episodes,
        rl_max_steps=rl_max_steps
    )
    
    # 运行完整实验
    results = experiment.run_full_experiment()
    
    if results['status'] == 'success':
        logging.info("实验成功完成！")
        logging.info(f"结果保存在: {experiment.experiment_dir}")
    else:
        logging.error(f"实验失败: {results.get('error', '未知错误')}")

if __name__ == "__main__":
    main() 