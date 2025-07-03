 #!/usr/bin/env python3
"""
测试从自然语言中提取策略信息的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.utils.json_parser import RobustJSONParser

def test_natural_language_extraction():
    """测试自然语言提取功能"""
    
    # 模拟LLM的响应
    llm_response = """基于以下设计分析和知识，我们可以生成一个详细的芯片布局策略。

1. Placement Strategy:
采用greedy Randomized Adaptive Search (GRAS) 算法进行芯片布局。GRAS 算法可以快速地生成高质量的布局策略，同时也能够满足设计要求。
2. Routing Strategy:
采用weighted random routing (WRR) 算法进行芯片 Routing。WRR 算法可以尽可能地减少路径长度，提高芯片性能。
3. Optimization Priorities:
根据设计要求，我们将优先级设置为：
        * Reduce timing
        * Reduce power
        * Use special nets
4. Parameter Suggestions:
根据设计情况，我们可以建议以下参数：
        * Cell library: 使用 default cell library
        * Placement library: 使用default placement library
        * Routing library: 使用default routing library
        * Width and height of the chip: 尽可能地减少，以提高性能和降低功耗
5. Constraint Handling:
根据设计要求，我们将约束处理方式设置为：
        * Timing: 使用 GRAS 算法进行约束处理
        * Power: 使用Warns 算法进行约束处理
6. Quality Targets:
设置质量目标为：
        * Fan-out delay: ≤ 100 ps
        * Leakage power: ≤ 100 μW
7. Execution Plan:
根据设计情况，我们可以生成以下执行计划：
        * Run GRAS 算法 для芯片布局
        * Run WRR 算法 для芯片 Routing
8. Metadata:
根据设计情况，我们可以将metadata信息设置为：
        * Number of components: 108292
        * Number of nets: 110281
        * Number of pins: 374
        * Area: 866444488900 μm²
        * Width and height of the chip: 930830 x 930830 μm²
        * Module types: 16
        * Hierarchy: ['top']
        * Constraints: {'timing': {'max_delay': 1000}, 'power': {'max_power': 1000}, 'special_nets': 2}

总体来说，这个芯片布局策略能够准确地满足设计要求，同时也能够提高芯片性能和降低功耗。

Please note that this is just a sample layout strategy, and the actual implementation may vary depending on the specific design requirements and constraints."""

    print("=== 测试自然语言提取功能 ===")
    print(f"LLM响应长度: {len(llm_response)} 字符")
    print(f"LLM响应前200字符: {llm_response[:200]}...")
    print()
    
    # 测试提取功能
    extracted_info = RobustJSONParser._extract_from_natural_language(llm_response)
    
    if extracted_info:
        print("✅ 成功提取策略信息:")
        print(f"  布局策略: {extracted_info.get('placement_strategy', 'unknown')}")
        print(f"  布线策略: {extracted_info.get('routing_strategy', 'unknown')}")
        print(f"  优化优先级: {extracted_info.get('optimization_priorities', [])}")
        print(f"  元数据: {extracted_info.get('metadata', {})}")
    else:
        print("❌ 提取失败")
    
    print()
    
    # 测试完整的解析流程
    print("=== 测试完整解析流程 ===")
    parsed_result = RobustJSONParser.parse_llm_response(llm_response, "layout_strategy")
    
    print(f"解析结果: {parsed_result.get('placement_strategy', 'unknown')}")
    print(f"完整结果: {parsed_result}")

if __name__ == "__main__":
    test_natural_language_extraction()