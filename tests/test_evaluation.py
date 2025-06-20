import unittest
import json
import os
import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.evaluation.layout_quality_evaluator import LayoutQualityEvaluator
from modules.evaluation.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator

logger = logging.getLogger(__name__)

class TestEvaluationSystem(unittest.TestCase):
    """测试评估系统功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        logger.info("开始步骤: 初始化测试环境")
        
        # 设置测试配置
        cls.config = {
            'knowledge_base': {
                'path': '/tmp/chiprag_test',
                'text_path': '/tmp/chiprag_test/text',
                'image_path': '/tmp/chiprag_test/images',
                'structured_data_path': '/tmp/chiprag_test/structured',
                'graph_path': '/tmp/chiprag_test/graph',
                'layout_experience_path': '/tmp/chiprag_test/layout',
                'cache_dir': '/tmp/chiprag_test/cache'
            },
            'evaluator': {
                'constraint_satisfaction': {
                    'constraints': {
                        'wirelength': {'max': 1000},
                        'congestion': {'max': 0.8},
                        'timing': {'max': 10}
                    },
                    'weights': {
                        'wirelength': 0.4,
                        'congestion': 0.3,
                        'timing': 0.3
                    }
                }
            },
            'evaluator_config': {
                'metrics': {
                    'wirelength': {
                        'type': 'minimize',
                        'weight': 0.4,
                        'threshold': 1000
                    },
                    'congestion': {
                        'type': 'minimize',
                        'weight': 0.3,
                        'threshold': 0.8
                    },
                    'timing': {
                        'type': 'minimize',
                        'weight': 0.3,
                        'threshold': 10
                    }
                },
                'weights': {
                    'wirelength': 0.4,
                    'congestion': 0.3,
                    'timing': 0.3
                },
                'thresholds': {
                    'wirelength': 1000,
                    'congestion': 0.8,
                    'timing': 10
                }
            }
        }
        
        # 初始化评估器
        cls.constraint_evaluator = ConstraintSatisfactionEvaluator(cls.config['evaluator']['constraint_satisfaction'])
        cls.evaluator = MultiObjectiveEvaluator(cls.config['evaluator_config'])
        
        # 创建测试数据
        cls.test_layout = {
            'area': 1000,
            'wirelength': 5000,
            'timing': 0.8
        }
        
        cls.test_constraints = {
            'area': {'max': 1200},
            'wirelength': {'max': 6000},
            'timing': {'min': 0.7}
        }
        
        cls.layout_evaluator = LayoutQualityEvaluator(cls.config)
        
    def test_layout_quality_evaluation(self):
        """测试布局质量评估"""
        test_layout = {
            "components": [
                {
                    "name": "memory_block",
                    "type": "component",
                    "position": {"x": 0, "y": 0},
                    "size": {"width": 100, "height": 100}
                },
                {
                    "name": "logic_block",
                    "type": "component",
                    "position": {"x": 150, "y": 0},
                    "size": {"width": 50, "height": 50}
                }
            ],
            "nets": [
                {
                    "name": "net1",
                    "source": "memory_block",
                    "target": "logic_block"
                }
            ]
        }
        quality_score = self.layout_evaluator.evaluate(test_layout)
        self.assertIsNotNone(quality_score)
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 1)
        
    def test_constraint_satisfaction_evaluation(self):
        """测试约束满足度评估"""
        test_layout = {
            "components": [
                {
                    "name": "memory_block",
                    "type": "component",
                    "power": 0.5
                }
            ],
            "constraints": [
                {
                    "type": "power",
                    "max_value": 1.0
                }
            ]
        }
        satisfaction_score = self.constraint_evaluator.evaluate(test_layout)
        self.assertIsNotNone(satisfaction_score)
        self.assertGreaterEqual(satisfaction_score, 0)
        self.assertLessEqual(satisfaction_score, 1)
        
    def test_multi_objective_evaluation(self):
        """测试多目标评估"""
        # 准备测试数据
        test_data = {
            'wirelength': 5000,
            'congestion': 0.8,
            'timing': 0.7
        }
        
        # 执行评估
        evaluation_result = self.evaluator.evaluate(test_data)
        
        # 验证结果
        self.assertIsInstance(evaluation_result, dict)
        self.assertIn('wirelength_score', evaluation_result)
        self.assertIn('congestion_score', evaluation_result)
        self.assertIn('timing_score', evaluation_result)
        self.assertIn('overall_score', evaluation_result)
        
        # 验证分数范围
        self.assertGreaterEqual(evaluation_result['wirelength_score'], 0)
        self.assertLessEqual(evaluation_result['wirelength_score'], 1)
        self.assertGreaterEqual(evaluation_result['congestion_score'], 0)
        self.assertLessEqual(evaluation_result['congestion_score'], 1)
        self.assertGreaterEqual(evaluation_result['timing_score'], 0)
        self.assertLessEqual(evaluation_result['timing_score'], 1)
        self.assertGreaterEqual(evaluation_result['overall_score'], 0)
        self.assertLessEqual(evaluation_result['overall_score'], 1)
        
    def test_evaluation_metrics(self):
        """测试评估指标计算"""
        test_layout = {
            "components": [
                {
                    "name": "memory_block",
                    "type": "component",
                    "position": {"x": 0, "y": 0},
                    "size": {"width": 100, "height": 100}
                }
            ],
            "nets": [
                {
                    "name": "net1",
                    "source": "memory_block",
                    "target": "memory_block"
                }
            ]
        }
        metrics = self.layout_evaluator.calculate_metrics(test_layout)
        self.assertIn("wirelength", metrics)
        self.assertIn("congestion", metrics)
        self.assertIn("timing", metrics)
        self.assertIn("power", metrics)

if __name__ == '__main__':
    unittest.main()
