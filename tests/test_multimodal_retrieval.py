# chiprag/scripts/test_multimodal_retrieval.py

import unittest
from typing import Dict, Any
import logging
from modules.core.modal_retriever import ModalRetriever
from modules.knowledge.knowledge_base import KnowledgeBase
from unittest.mock import patch
from PIL import Image
import os

class TestMultimodalRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
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
            'image_encoder': {
                'model_name': 'resnet50',
                'pretrained': True,
                'feature_dim': 2048,
                'device': 'cpu'
            },
            'text_encoder': {
                'model_name': 'bert-base-uncased',
                'max_length': 512,
                'device': 'cpu'
            },
            'graph_encoder': {
                'embedding_dim': 256,
                'num_layers': 2,
                'dropout': 0.1
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
        
        # 创建测试数据
        cls.test_data = {
            'circuit': {
                'features': [0.1, 0.2, 0.3],
                'metadata': {'name': 'Test Circuit'}
            },
            'module': {
                'features': [0.4, 0.5, 0.6],
                'metadata': {'name': 'Test Module'}
            }
        }
        
        # 初始化检索器
        cls.retriever = ModalRetriever(cls.config)
        cls.knowledge_base = KnowledgeBase(cls.config['knowledge_base'])

    def setUp(self):
        pass

    def test_modal_retrieval(self):
        """测试模态检索功能"""
        query = {
            'text': '测试查询',
            'image': 'data/test_images/test_layout.png',
            'graph': {'nodes': [], 'edges': []}
        }
        
        # 确保测试图像存在
        os.makedirs('data/test_images', exist_ok=True)
        test_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_images', 'test_layout.png')
        if not os.path.exists(test_image_path):
            # 创建一个测试图像
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='white')
            img.save(test_image_path)
        
        query['image'] = test_image_path
        results = self.retriever.retrieve(query)
        
        # 修复期望：retrieve方法返回的是列表，不是字典
        self.assertIsInstance(results, list)
        if results:
            self.assertIn('similarity', results[0])
            self.assertIn('modality_similarities', results[0])
            self.assertIn('metadata', results[0])

def main():
    unittest.main()

if __name__ == '__main__':
    main()