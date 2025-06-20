import unittest
import json
import logging
from typing import Dict
import os
import sys
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils.llm_manager import LLMManager, NumpyEncoder
from modules.utils.embedding_manager import EmbeddingManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLLMIntegration(unittest.TestCase):
    """测试LLM集成"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
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
            'llm_config': {
                'base_url': 'http://localhost:8000',
                'model_name': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        }
        
        # 初始化 LLM 管理器
        cls.llm_manager = LLMManager(cls.config['llm_config'])
        
        # Mock LLM 分析结果
        cls.mock_analysis = {
            'needs_optimization': True,
            'optimization_suggestions': [
                'Reduce area by optimizing component placement',
                'Improve timing by adjusting wire routing',
                'Minimize wirelength by reorganizing layout'
            ],
            'score': 0.75,
            'vector': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        # Mock LLM 优化结果
        cls.mock_optimized_layout = {
            'area': 900,
            'wirelength': 4500,
            'timing': 0.85
        }
        
        # 加载测试数据
        cls.test_layout = cls._load_test_layout()
        
    @classmethod
    def _load_test_layout(cls) -> Dict:
        """加载测试布局数据"""
        # 创建一个简单的测试布局
        return {
            'name': 'test_layout',
            'components': [
                {
                    'name': 'comp1',
                    'type': 'memory',
                    'position': {'x': 0, 'y': 0},
                    'size': {'width': 100, 'height': 100}
                },
                {
                    'name': 'comp2',
                    'type': 'logic',
                    'position': {'x': 200, 'y': 0},
                    'size': {'width': 50, 'height': 50}
                },
                {
                    'name': 'comp3',
                    'type': 'io',
                    'position': {'x': 0, 'y': 200},
                    'size': {'width': 30, 'height': 30}
                }
            ],
            'nets': [
                {
                    'name': 'net1',
                    'pins': [
                        {'component': 'comp1', 'pin': 'out'},
                        {'component': 'comp2', 'pin': 'in'}
                    ]
                },
                {
                    'name': 'net2',
                    'pins': [
                        {'component': 'comp2', 'pin': 'out'},
                        {'component': 'comp3', 'pin': 'in'}
                    ]
                }
            ],
            'die_area': {
                'width': 500,
                'height': 500
            },
            'hierarchy': {
                'levels': ['top', 'module'],
                'modules': ['mem', 'logic', 'io'],
                'max_depth': 2
            }
        }
        
    def test_llm_analysis(self):
        """测试LLM分析功能"""
        logger.info("开始测试LLM分析功能")
        
        # 准备测试数据
        test_data = {
            'layout': {
                'area': 1000,
                'wirelength': 5000,
                'timing': 0.8
            },
            'constraints': {
                'area': {'max': 1200},
                'wirelength': {'max': 6000},
                'timing': {'min': 0.7}
            }
        }
        
        # Mock analyze_layout 方法
        self.llm_manager.analyze_layout = lambda x: self.mock_analysis
        
        # 执行分析
        analysis = self.llm_manager.analyze_layout(test_data)
        
        # 验证结果
        self.assertIsInstance(analysis, dict)
        self.assertIn('needs_optimization', analysis)
        self.assertIn('optimization_suggestions', analysis)
        self.assertIsInstance(analysis['needs_optimization'], bool)
        self.assertIsInstance(analysis['optimization_suggestions'], list)
        
    def test_llm_optimization(self):
        """测试LLM优化功能"""
        logger.info("开始测试LLM优化功能")
        
        # Mock analyze_layout 和 optimize_layout 方法
        self.llm_manager.analyze_layout = lambda x: self.mock_analysis
        self.llm_manager.optimize_layout = lambda x, y: self.mock_optimized_layout
        
        # 准备测试数据
        test_data = {
            'layout': {
                'area': 1000,
                'wirelength': 5000,
                'timing': 0.8
            },
            'constraints': {
                'area': {'max': 1200},
                'wirelength': {'max': 6000},
                'timing': {'min': 0.7}
            }
        }
        
        # 执行优化
        analysis = self.llm_manager.analyze_layout(test_data)
        optimized_layout = self.llm_manager.optimize_layout(test_data, analysis['optimization_suggestions'])
        
        # 验证结果
        self.assertIsInstance(optimized_layout, dict)
        self.assertIn('area', optimized_layout)
        self.assertIn('wirelength', optimized_layout)
        self.assertIn('timing', optimized_layout)
        self.assertLess(optimized_layout['area'], test_data['layout']['area'])
        self.assertLess(optimized_layout['wirelength'], test_data['layout']['wirelength'])
        self.assertGreater(optimized_layout['timing'], test_data['layout']['timing'])
        
    def test_embedding(self):
        """测试向量化功能"""
        logger.info("开始测试向量化功能")
        
        # 初始化向量化管理器
        embedding_manager = EmbeddingManager()
        
        # 生成向量
        vector = embedding_manager.embed_layout(self.test_layout)
        
        # 验证向量
        self.assertIsNotNone(vector)
        self.assertTrue(len(vector.shape) > 0)
        
        # 打印向量信息
        logger.info(f"向量形状: {vector.shape}")
        
    def test_similarity(self):
        """测试相似度计算"""
        logger.info("开始测试相似度计算")
        
        # 初始化向量化管理器
        embedding_manager = EmbeddingManager()
        
        # 生成两个布局的向量
        vector1 = embedding_manager.embed_layout(self.test_layout)
        
        # 创建稍微修改的布局
        modified_layout = self.test_layout.copy()
        modified_layout['components'][0]['position']['x'] += 10
        
        vector2 = embedding_manager.embed_layout(modified_layout)
        
        # 计算相似度
        similarity = embedding_manager.compute_similarity(vector1, vector2)
        
        # 验证相似度
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
        
        # 打印相似度
        logger.info(f"相似度: {similarity}")
        
if __name__ == '__main__':
    unittest.main() 