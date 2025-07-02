#!/usr/bin/env python3
"""
测试综合知识库加载情况
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.config_loader import ConfigLoader
import json

def test_comprehensive_knowledge_base():
    """测试综合知识库"""
    print("=== 测试综合知识库加载 ===")
    
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load_config('rag_config.json')
    
    # 初始化知识库
    kb = KnowledgeBase(config['knowledge_base'])
    
    # 加载知识库
    # print("正在加载知识库...")
    # kb.load_knowledge_base()
    
    # 获取案例数量
    case_count = len(kb.cases)
    print(f"知识库中的案例总数: {case_count}")
    
    # 按来源统计
    sources = {}
    designs = {}
    
    for case in kb.cases:
        source = case.get('source', 'unknown')
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
        
        design_name = case.get('design_name', 'unknown')
        if design_name not in designs:
            designs[design_name] = 0
        designs[design_name] += 1
    
    print("\n=== 数据来源统计 ===")
    for source, count in sources.items():
        print(f"{source}: {count} 个案例")
    
    print("\n=== 设计分布统计 ===")
    for design, count in sorted(designs.items(), key=lambda x: x[1], reverse=True):
        print(f"{design}: {count} 个案例")
    
    # 测试检索功能
    print("\n=== 测试检索功能 ===")
    # 构造简单特征字典
    test_query_features = {
        'design_name': 'mgc_des_perf_1',
        'num_components': 1000,
        'component_density': 0.5
    }
    results = kb.get_similar_cases(test_query_features, top_k=5)
    print(f"查询 'mgc_des_perf_1' 的检索结果数量: {len(results)}")
    
    if results:
        print("前3个检索结果:")
        for i, result in enumerate(results[:3]):
            case = result.get('case', {})
            print(f"  {i+1}. 设计: {case.get('design_name', 'unknown')}, "
                  f"来源: {case.get('source', 'unknown')}, "
                  f"相似度: {result.get('similarity', 0):.4f}")
    
    # 测试特定设计的检索
    print("\n=== 测试特定设计检索 ===")
    specific_design = "mgc_fft_1"
    design_cases = [case for case in kb.cases if case.get('design_name') == specific_design]
    print(f"设计 '{specific_design}' 的案例数量: {len(design_cases)}")
    
    if design_cases:
        print("该设计的案例来源:")
        design_sources = {}
        for case in design_cases:
            source = case.get('source', 'unknown')
            if source not in design_sources:
                design_sources[source] = 0
            design_sources[source] += 1
        
        for source, count in design_sources.items():
            print(f"  {source}: {count} 个案例")
    
    print("\n=== 知识库测试完成 ===")

if __name__ == "__main__":
    test_comprehensive_knowledge_base() 