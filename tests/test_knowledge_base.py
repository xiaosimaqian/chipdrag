import unittest
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chiprag.modules.knowledge.knowledge_base import KnowledgeBase

class TestKnowledgeBase(unittest.TestCase):
    """测试知识库功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 设置测试配置
        cls.config = {
            'knowledge_base': {
                'path': '/tmp/chiprag_test',
                'format': 'json',
                'text_path': '/tmp/chiprag_test/text',
                'image_path': '/tmp/chiprag_test/images',
                'structured_data_path': '/tmp/chiprag_test/structured',
                'graph_path': '/tmp/chiprag_test/graph',
                'layout_experience_path': '/tmp/chiprag_test/layout',
                'cache_dir': '/tmp/chiprag_test/cache'
            },
            'hierarchy_config': {
                'max_depth': 3,
                'min_components': 2,
                'max_components': 10,
                'similarity_threshold': 0.8
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
        
        # 初始化知识库
        cls.knowledge_base = KnowledgeBase(cls.config['knowledge_base'])
        
        # 添加一条初始数据
        test_data = {
            'name': 'init_block',
            'type': 'block',
            'features': {
                'area': 100,
                'power': 0.5
            },
            'hierarchy': {
                'levels': ['top', 'block'],
                'modules': ['init_block'],
                'max_depth': 2
            },
            'components': [
                {
                    'name': 'comp1',
                    'type': 'module',
                    'x': 0,
                    'y': 0,
                    'width': 10,
                    'height': 10
                }
            ],
            'nets': [
                {
                    'name': 'net1',
                    'source': 'comp1',
                    'target': 'comp2',
                    'type': 'signal',
                    'connections': [
                        {'from': 'comp1', 'to': 'comp2'}
                    ]
                }
            ]
        }
        cls.knowledge_base.add_case(test_data, {})
        
    def test_knowledge_initialization(self):
        """测试知识库初始化"""
        self.assertIsNotNone(self.knowledge_base)
        self.assertTrue(hasattr(self.knowledge_base, 'load'))
        self.assertTrue(hasattr(self.knowledge_base, 'query'))
        self.assertTrue(hasattr(self.knowledge_base, 'update'))
        
    def test_knowledge_update(self):
        """测试知识更新"""
        # 添加测试数据
        test_data = {
            'name': 'test_block',
            'type': 'block',
            'features': {
                'area': 100,
                'power': 0.5
            },
            'hierarchy': {
                'levels': ['top', 'block'],
                'modules': ['test_block'],
                'max_depth': 2
            },
            'components': [
                {
                    'name': 'comp1',
                    'type': 'module',
                    'x': 0,
                    'y': 0,
                    'width': 10,
                    'height': 10
                }
            ],
            'nets': [
                {
                    'name': 'net1',
                    'source': 'comp1',
                    'target': 'comp2',
                    'type': 'signal',
                    'connections': [
                        {'from': 'comp1', 'to': 'comp2'}
                    ]
                }
            ]
        }
        self.knowledge_base.add_case(test_data, {})
        
        # 查询并验证
        results = self.knowledge_base.query({'type': 'block'})
        self.assertTrue(len(results) > 0)
        names = [r["name"] for r in results]
        self.assertIn("test_block", names)
        target = next((r for r in results if r["name"] == "test_block"), None)
        self.assertIsNotNone(target)
        self.assertTrue('hierarchy' in target)
        self.assertTrue('levels' in target['hierarchy'])
        self.assertIn('block', target['hierarchy']['levels'])
        
    def test_knowledge_loading(self):
        """测试知识加载"""
        self.knowledge_base.load()
        self.assertGreater(len(self.knowledge_base), 0)
        
    def test_knowledge_query(self):
        """测试知识查询"""
        query = {"type": "memory_block"}
        results = self.knowledge_base.query(query)
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main() 