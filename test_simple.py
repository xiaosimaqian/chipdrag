#!/usr/bin/env python3
"""
简单的多模态融合检索测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from modules.utils.embedding_manager import EmbeddingManager
from modules.knowledge.knowledge_base import KnowledgeBase

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding_manager():
    """测试向量化管理器"""
    print("=== 测试向量化管理器 ===")
    
    # 初始化向量化管理器
    config = {
        'model_name': 'bge-m3',
        'api_base': 'http://localhost:11434',
        'use_local_model': True,
        'local_model_path': 'models/bert',
        'embedding_dim': 768
    }
    
    embedding_manager = EmbeddingManager(config)
    
    # 测试布局数据
    test_layout = {
        "name": "test_layout",
        "components": [
            {
                "name": "comp1",
                "type": "memory",
                "position": {"x": 0, "y": 0},
                "size": {"width": 100, "height": 100}
            },
            {
                "name": "comp2", 
                "type": "logic",
                "position": {"x": 200, "y": 0},
                "size": {"width": 150, "height": 80}
            }
        ],
        "nets": [
            {
                "name": "net1",
                "source": "comp1",
                "target": "comp2",
                "type": "signal"
            }
        ],
        "hierarchy": {
            "levels": ["top", "module"],
            "modules": ["mem", "logic"],
            "max_depth": 2
        }
    }
    
    try:
        # 测试向量化
        vector = embedding_manager.embed_layout(test_layout)
        print(f"✓ 向量化成功，向量维度: {len(vector)}")
        
        # 测试相似度计算
        vector2 = embedding_manager.embed_layout(test_layout)
        similarity = embedding_manager.compute_similarity(vector, vector2)
        print(f"✓ 相似度计算成功: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 向量化管理器测试失败: {str(e)}")
        return False

def test_knowledge_base():
    """测试知识库"""
    print("\n=== 测试知识库 ===")
    
    # 初始化知识库
    config = {
        'path': '/tmp/test_kb',
        'format': 'json',
        'layout_experience_path': '/tmp/test_kb/layout'
    }
    
    try:
        knowledge_base = KnowledgeBase(config)
        
        # 添加测试数据
        test_data = {
            'name': 'test_module',
            'type': 'module',
            'components': [
                {
                    'name': 'comp1',
                    'type': 'memory',
                    'x': 0,
                    'y': 0,
                    'width': 100,
                    'height': 100
                }
            ],
            'hierarchy': {
                'levels': ['top', 'module'],
                'modules': ['mem'],
                'max_depth': 2
            }
        }
        
        optimization_result = {
            'wirelength': 500,
            'congestion': 0.5,
            'timing': 5.0,
            'score': 0.8
        }
        
        knowledge_base.add_case(test_data, optimization_result)
        print(f"✓ 知识库初始化成功，包含 {len(knowledge_base.cases)} 个案例")
        
        # 测试检索
        query = {
            'hierarchy': {
                'levels': ['top', 'module'],
                'modules': ['mem']
            }
        }
        
        similar_cases = knowledge_base.get_similar_cases(query, top_k=3)
        print(f"✓ 检索成功，找到 {len(similar_cases)} 个相似案例")
        
        return True
        
    except Exception as e:
        print(f"✗ 知识库测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("开始多模态融合检索功能测试...\n")
    
    # 测试向量化管理器
    embedding_success = test_embedding_manager()
    
    # 测试知识库
    kb_success = test_knowledge_base()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"向量化管理器: {'✓ 通过' if embedding_success else '✗ 失败'}")
    print(f"知识库: {'✓ 通过' if kb_success else '✗ 失败'}")
    
    if embedding_success and kb_success:
        print("\n🎉 所有测试通过！多模态融合检索功能正常工作。")
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")

if __name__ == "__main__":
    main() 