import os
import json
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 修改导入路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils.benchmark_loader import BenchmarkLoader
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.core.rag_system import RAGSystem
from modules.evaluation.layout_quality_evaluator import LayoutQualityEvaluator
from modules.evaluation.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from modules.core.hierarchy import HierarchicalDecompositionManager
from modules.knowledge.multi_modal_knowledge_graph import MultiModalKnowledgeGraph
from modules.core.knowledge_transfer import KnowledgeTransfer
from modules.utils.llm_manager import LLMManager
from modules.utils.embedding_manager import EmbeddingManager
from modules.core.layout_generator import LayoutGenerator
from modules.core.rag_model import RAGLayoutModel
from modules.evaluation.metrics import LayoutEvaluator
from modules.retrieval.hierarchical_retriever import HierarchicalRetriever
from modules.utils.data_loader import DataLoader
from modules.utils.expert_feedback import ExpertFeedbackSimulator
from modules.visualization.layout_visualizer import LayoutVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGLayoutExperiment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        self._init_components()
        self.visualizer = LayoutVisualizer()
        
    def _validate_config(self):
        """验证配置"""
        required_sections = ['name', 'version', 'data', 'retrieval', 'generation', 'evaluation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置缺少必要部分: {section}")
                
        # 验证环境配置
        if 'environment' not in self.config:
            self.config['environment'] = {
                'cache_dir': 'cache',
                'log_dir': 'logs',
                'output_dir': 'results',
                'temp_dir': 'temp'
            }
            
        # 验证模型配置
        if 'models' not in self.config:
            self.config['models'] = {
                'llm': {
                    'name': 'llama2',
                    'base_url': 'http://localhost:11434',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'layout': {
                    'name': 'layout_model',
                    'type': 'transformer',
                    'hidden_size': 512,
                    'num_layers': 6,
                    'num_heads': 8,
                    'dropout': 0.1
                }
            }
            
    def _init_components(self):
        """初始化实验组件"""
        try:
            # 1. 知识库配置
            kb_config = {
                'path': 'data/knowledge_base',
                'type': 'json',
                'format': 'json',
                'hierarchy_config': {
                    'levels': [
                        {
                            'name': 'system',
                            'threshold': 0.8,
                            'weight': 0.3
                        },
                        {
                            'name': 'module',
                            'threshold': 0.7,
                            'weight': 0.4
                        },
                        {
                            'name': 'component',
                            'threshold': 0.6,
                            'weight': 0.3
                        }
                    ]
                },
                'llm_config': {
                    'name': 'llama2',
                    'temperature': 0.7,
                    'max_tokens': 1000
                }
            }
            
            # 2. 布局模型配置
            layout_config = {
                # 基本信息
                'name': 'layout_model',
                'model_name': 'layout_transformer',
                'type': 'transformer',
                
                # 模型架构
                'hidden_size': 512,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': 0.1,
                
                # 生成器配置
                'generator': {
                    'type': 'transformer',
                    'hidden_size': 512,
                    'num_layers': 6,
                    'num_heads': 8,
                    'dropout': 0.1
                },
                
                # 优化器配置
                'optimizer': {
                    'type': 'adam',
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-8
                },
                
                # 训练配置
                'training': {
                    'batch_size': 32,
                    'max_epochs': 100,
                    'early_stopping': {
                        'patience': 10,
                        'min_delta': 1e-4
                    },
                    'gradient_clipping': 1.0
                },
                
                # 评估配置
                'evaluation': {
                    'metrics': ['accuracy', 'f1', 'precision', 'recall'],
                    'validation_split': 0.2,
                    'test_split': 0.1
                }
            }
            
            # 3. 初始化组件
            self.knowledge_base = KnowledgeBase(config=kb_config)
            self.retriever = HierarchicalRetriever(config=layout_config)
            
            # 4. 初始化生成器
            generator_config = {
                'name': 'layout_generator',
                'model_name': 'layout_transformer',
                'type': 'transformer',
                'generator': {
                    'type': 'transformer',
                    'hidden_size': 512,
                    'num_layers': 6,
                    'num_heads': 8,
                    'dropout': 0.1
                },
                'hidden_size': 512,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': 0.1,
                'optimizer': {
                    'type': 'adam',
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-8
                }
            }
            
            self.generator = RAGLayoutModel(config=generator_config)
            
            # 5. 初始化分解管理器
            decomposition_config = {
                'name': 'hierarchical_decomposition',
                'levels': [
                    {
                        'name': 'system',
                        'threshold': 0.8,
                        'weight': 0.3
                    },
                    {
                        'name': 'module',
                        'threshold': 0.7,
                        'weight': 0.4
                    },
                    {
                        'name': 'component',
                        'threshold': 0.6,
                        'weight': 0.3
                    }
                ],
                'llm_config': {
                    'name': 'llama2',
                    'temperature': 0.7,
                    'max_tokens': 1000
                }
            }
            
            self.decomposition_manager = HierarchicalDecompositionManager(config=decomposition_config)
            
            # 6. 初始化评估器
            evaluator_config = {
                'name': 'layout_evaluator',
                'metrics': {
                    'layout_quality': {
                        'threshold': 0.8,
                        'weight': 0.4
                    },
                    'constraint_satisfaction': {
                        'threshold': 0.9,
                        'weight': 0.3
                    },
                    'performance': {
                        'threshold': 0.85,
                        'weight': 0.3
                    }
                }
            }
            
            self.evaluator = LayoutEvaluator(config=evaluator_config)
            
        except Exception as e:
            logger.error(f"初始化组件失败: {str(e)}")
            raise
            
    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """运行实验
        
        Args:
            query: 查询信息，包含约束和上下文
            
        Returns:
            Dict[str, Any]: 实验结果
        """
        try:
            # 1. 分解查询
            decomposition = self.decomposition_manager.decompose(query)
            
            # 2. 检索相关组件
            retrieved_components = self.retriever.retrieve(query)
            
            # 3. 将检索到的组件添加到查询中
            query['retrieved_components'] = retrieved_components
            
            # 4. 生成布局
            layout_scheme = self.generator.generate(query=query)
            
            # 5. 可视化布局
            if 'layout' in layout_scheme:
                # 创建可视化目录
                vis_dir = os.path.join(self.config['environment']['output_dir'], 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                # 生成可视化文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                vis_path = os.path.join(vis_dir, f'layout_{timestamp}.png')
                
                # 可视化布局
                self.visualizer.visualize(layout_scheme['layout'], save_path=vis_path)
                logger.info(f"布局图已保存到: {vis_path}")
            
            # 6. 评估结果
            evaluation_results = self.evaluator.evaluate({
                'layout_quality': self._evaluate_layout_quality(layout_scheme),
                'constraint_satisfaction': self._evaluate_constraint_satisfaction(layout_scheme, query),
                'performance': self._evaluate_performance(layout_scheme)
            })
            
            # 7. 记录结果
            results = {
                'query': query,
                'decomposition': decomposition,
                'retrieved_components': retrieved_components,
                'layout_scheme': layout_scheme,
                'evaluation': evaluation_results
            }
            
            self._save_results(results)
            
            # 在检索过程中添加日志
            logger.info(f"检索到的组件：{retrieved_components}")
            logger.info(f"查询分解结果：{decomposition}")
            
            return results
            
        except Exception as e:
            logger.error(f"实验运行失败: {str(e)}")
            raise

    def _evaluate_layout_quality(self, layout_scheme: Dict[str, Any]) -> float:
        """评估布局质量"""
        try:
            # 评估布局密度
            density_score = self._evaluate_density(layout_scheme)
            
            # 评估拥塞情况
            congestion_score = self._evaluate_congestion(layout_scheme)
            
            # 评估时序裕量
            timing_score = self._evaluate_timing_margin(layout_scheme)
            
            # 计算综合得分
            total_score = (
                density_score * 0.3 +
                congestion_score * 0.3 +
                timing_score * 0.4
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"评估布局质量失败: {str(e)}")
            return 0.0

    def _evaluate_constraint_satisfaction(self, layout_scheme: Dict[str, Any], query: Dict[str, Any]) -> float:
        """评估约束满足度"""
        try:
            # 获取约束
            constraints = query.get('constraints', {})
            
            # 评估时序约束
            timing_score = self._evaluate_timing_constraints(layout_scheme, constraints)
            
            # 评估线长约束
            wirelength_score = self._evaluate_wirelength_constraints(layout_scheme, constraints)
            
            # 评估功耗约束
            power_score = self._evaluate_power_constraints(layout_scheme, constraints)
            
            # 计算综合得分
            total_score = (
                timing_score * 0.4 +
                wirelength_score * 0.3 +
                power_score * 0.3
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"评估约束满足度失败: {str(e)}")
            return 0.0

    def _evaluate_performance(self, layout_scheme: Dict[str, Any]) -> float:
        """评估性能指标"""
        try:
            # 评估面积
            area_score = self._evaluate_area(layout_scheme)
            
            # 评估功耗
            power_score = self._evaluate_power(layout_scheme)
            
            # 评估时序
            timing_score = self._evaluate_timing(layout_scheme)
            
            # 计算综合得分
            total_score = (
                area_score * 0.4 +
                power_score * 0.3 +
                timing_score * 0.3
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"评估性能指标失败: {str(e)}")
            return 0.0

    def _save_results(self, results: Dict[str, Any]):
        """保存实验结果"""
        try:
            # 创建输出目录
            output_dir = Path(self.config['environment']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"experiment_results_{timestamp}.json"
            
            # 保存结果
            with open(output_dir / filename, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"实验结果已保存到: {output_dir / filename}")
            
        except Exception as e:
            logger.error(f"保存实验结果失败: {str(e)}")
            raise

    def _evaluate_density(self, layout_scheme: Dict[str, Any]) -> float:
        """评估布局密度"""
        # TODO: 实现布局密度评估
        return 0.8

    def _evaluate_congestion(self, layout_scheme: Dict[str, Any]) -> float:
        """评估拥塞情况"""
        # TODO: 实现拥塞评估
        return 0.7

    def _evaluate_timing_margin(self, layout_scheme: Dict[str, Any]) -> float:
        """评估时序裕量"""
        # TODO: 实现时序裕量评估
        return 0.9

    def _evaluate_timing_constraints(self, layout_scheme: Dict[str, Any], constraints: Dict[str, Any]) -> float:
        """评估时序约束"""
        # TODO: 实现时序约束评估
        return 0.85

    def _evaluate_wirelength_constraints(self, layout_scheme: Dict[str, Any], constraints: Dict[str, Any]) -> float:
        """评估线长约束"""
        # TODO: 实现线长约束评估
        return 0.8

    def _evaluate_power_constraints(self, layout_scheme: Dict[str, Any], constraints: Dict[str, Any]) -> float:
        """评估功耗约束"""
        # TODO: 实现功耗约束评估
        return 0.75

    def _evaluate_area(self, layout_scheme: Dict[str, Any]) -> float:
        """评估面积"""
        # TODO: 实现面积评估
        return 0.9

    def _evaluate_power(self, layout_scheme: Dict[str, Any]) -> float:
        """评估功耗"""
        # TODO: 实现功耗评估
        return 0.85

    def _evaluate_timing(self, layout_scheme: Dict[str, Any]) -> float:
        """评估时序"""
        # TODO: 实现时序评估
        return 0.8

    def _check_knowledge_base(self):
        """检查知识库文件"""
        print("知识库文件：")
        print(os.listdir("data/knowledge_base"))

    def _check_retriever_config(self):
        """检查检索器配置"""
        print("检索器配置：")
        print(self.retriever.config)

if __name__ == "__main__":
    # 加载配置文件
    config_path = "configs/experiment_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    experiment = RAGLayoutExperiment(config)
    results = experiment.run_experiment() 