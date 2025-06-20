#!/usr/bin/env python3
"""
简单的知识库测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test():
    try:
        from modules.knowledge.knowledge_base import KnowledgeBase
        
        # 使用正确的配置
        config = {
            "path": "data/knowledge_base",
            "format": "pkl",
            "layout_experience": "data/knowledge_base"  # 这个字段名很重要
        }
        
        print("初始化知识库...")
        kb = KnowledgeBase(config)
        
        print(f"知识库包含 {len(kb.cases)} 个案例")
        
        if len(kb.cases) > 0:
            print("✅ 知识库加载成功！")
            
            # 测试相似案例检索
            test_query = {"name": "test"}
            similar_cases = kb.get_similar_cases(test_query, top_k=3)
            print(f"找到 {len(similar_cases)} 个相似案例")
            
            return True
        else:
            print("❌ 知识库加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test() 