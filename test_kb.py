#!/usr/bin/env python3
"""
简单的知识库测试脚本
"""

import os
import pickle
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def check_kb():
    """检查知识库"""
    print("检查知识库数据...")
    
    # 检查数据目录
    data_dir = Path("data/knowledge_base")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    print(f"✅ 数据目录存在: {data_dir}")
    
    # 检查cases.pkl
    cases_file = data_dir / "cases.pkl"
    if cases_file.exists():
        try:
            with open(cases_file, 'rb') as f:
                cases = pickle.load(f)
            print(f"✅ cases.pkl 存在，包含 {len(cases)} 个案例")
            
            if len(cases) > 0:
                print(f"   第一个案例: {cases[0].get('id', 'unknown')}")
        except Exception as e:
            print(f"❌ 读取cases.pkl失败: {str(e)}")
    else:
        print("❌ cases.pkl 不存在")
    
    # 检查knowledge_graph.pkl
    graph_file = data_dir / "knowledge_graph.pkl"
    if graph_file.exists():
        size = os.path.getsize(graph_file)
        print(f"✅ knowledge_graph.pkl 存在，大小: {size} bytes")
    else:
        print("❌ knowledge_graph.pkl 不存在")

def test_kb_loading():
    """测试知识库加载"""
    print("\n测试知识库加载...")
    
    try:
        from modules.knowledge.knowledge_base import KnowledgeBase
        
        # 加载配置
        config = {
            "path": "data/knowledge_base",
            "format": "pkl",
            "layout_experience": "data/knowledge_base"
        }
        
        # 初始化知识库
        kb = KnowledgeBase(config)
        
        print(f"✅ 知识库初始化成功，包含 {len(kb.cases)} 个案例")
        
        # 测试相似案例检索
        test_query = {
            "name": "test_design",
            "components": [],
            "nets": []
        }
        
        similar_cases = kb.get_similar_cases(test_query, top_k=3)
        print(f"✅ 相似案例检索返回 {len(similar_cases)} 个结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 知识库加载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_kb()
    test_kb_loading()
    print("\n检查完成") 