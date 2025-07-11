#!/usr/bin/env python3
"""
知识库配置诊断脚本
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append('.')

def test_config_loading():
    """测试配置文件加载"""
    print("=== 测试配置文件加载 ===")
    
    # 检查配置文件是否存在
    config_paths = [
        'configs/knowledge_base.json',
        'configs/configs/knowledge_base.json',
        '/mnt/data/keqin/chipdrag/configs/knowledge_base.json'
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            print(f"✅ 找到配置文件: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"  配置内容: {list(config.keys())}")
                if 'path' in config:
                    print(f"  路径字段: {config['path']}")
                return config
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
        else:
            print(f"❌ 配置文件不存在: {path}")
    
    return None

def test_knowledge_base_init(config):
    """测试知识库初始化"""
    print("\n=== 测试知识库初始化 ===")
    
    try:
        from modules.knowledge.knowledge_base import KnowledgeBase
        
        print("正在初始化知识库...")
        kb = KnowledgeBase(config)
        
        print(f"✅ 知识库初始化成功")
        print(f"  案例数量: {len(kb.cases)}")
        
        if kb.cases:
            print("  第一个案例:")
            case = kb.cases[0]
            print(f"    ID: {case.get('id', 'unknown')}")
            print(f"    Metadata keys: {list(case.get('metadata', {}).keys())}")
            
            # 测试特征提取
            features = kb._extract_case_features_for_similarity(case)
            print(f"    提取的特征: {features}")
            
            # 测试相似度计算
            query_features = {
                'num_components': 30000,
                'area': 1000000000,
                'component_density': 0.03,
                'hierarchy': {'modules': ['FFT']},
                'constraints': {'timing': {'max_delay': 1000}}
            }
            
            similarity = kb._compute_similarity(query_features, features)
            print(f"    与查询的相似度: {similarity:.3f}")
            
            # 测试相似案例检索
            similar_cases = kb.get_similar_cases(query_features, top_k=3, similarity_threshold=0.3)
            print(f"    找到相似案例: {len(similar_cases)} 个")
            
        return kb
        
    except Exception as e:
        print(f"❌ 知识库初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("开始知识库配置诊断...")
    
    # 测试配置文件加载
    config = test_config_loading()
    if not config:
        print("❌ 无法加载配置文件")
        return
    
    # 测试知识库初始化
    kb = test_knowledge_base_init(config)
    if not kb:
        print("❌ 知识库初始化失败")
        return
    
    print("\n✅ 诊断完成")

if __name__ == "__main__":
    main() 