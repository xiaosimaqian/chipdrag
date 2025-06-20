#!/usr/bin/env python3
"""
测试知识库修复效果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_knowledge_base():
    """测试知识库"""
    print("测试知识库修复效果...")
    
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
        
        if len(kb.cases) == 0:
            print("❌ 知识库中没有案例数据")
            return False
        
        # 测试相似案例检索
        test_query = {
            "name": "test_design",
            "components": [],
            "nets": []
        }
        
        print("开始检索相似案例...")
        similar_cases = kb.get_similar_cases(test_query, top_k=5, similarity_threshold=0.1)
        print(f"✅ 相似案例检索返回 {len(similar_cases)} 个结果")
        
        if len(similar_cases) > 0:
            print("✅ 相似案例检索成功！")
            # 显示第一个案例的信息
            first_case = similar_cases[0]
            print(f"   第一个案例ID: {first_case.get('id', 'unknown')}")
            print(f"   案例名称: {first_case.get('name', 'unnamed')}")
            return True
        else:
            print("❌ 相似案例检索失败")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_knowledge_base()
    if success:
        print("\n🎉 知识库修复成功！")
    else:
        print("\n❌ 知识库修复失败") 