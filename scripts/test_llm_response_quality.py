#!/usr/bin/env python3
"""
LLM回复质量测试脚本
验证优化后的LLM配置是否能生成更有效的回复
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.utils.llm_manager import LLMManager
from modules.utils.config_loader import ConfigLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llm_response_quality():
    """测试LLM回复质量"""
    
    try:
        # 加载LLM配置
        config_loader = ConfigLoader()
        llm_config = config_loader.load_config('configs/llm/ollama.json')
        
        # 如果加载失败，使用默认配置
        if not llm_config:
            logger.warning("使用默认LLM配置")
            llm_config = {
                'base_url': 'http://localhost:11434',
                'model': 'deepseek-coder:latest',
                'temperature': 0.8,
                'max_tokens': 1500,
                'timeout': 300,
                'retry_attempts': 2,
                'retry_delay': 5
            }
        
        # 初始化LLM管理器
        llm_manager = LLMManager(llm_config)
        
        logger.info("=== LLM回复质量测试开始 ===")
        
        # 测试用例1：布局策略生成
        logger.info("\n--- 测试1：布局策略生成 ---")
        design_analysis = {
            'design_type': 'risc_v',
            'complexity_level': 'high',
            'component_count': 5000,
            'net_count': 8000,
            'technology_node': '14nm',
            'area_constraint': 10000,
            'timing_constraint': 2.0,
            'power_budget': 5.0
        }
        
        knowledge = {
            'similar_cases': [
                {'design_type': 'risc_v', 'hpwl': 15000, 'success_rate': 0.9},
                {'design_type': 'dsp', 'hpwl': 12000, 'success_rate': 0.85}
            ],
            'optimization_tips': [
                '使用层次化布局算法',
                '优先考虑时序约束',
                '采用时序驱动的布线策略'
            ]
        }
        
        logger.info("生成布局策略...")
        strategy = llm_manager.generate_layout_strategy(design_analysis, knowledge)
        
        logger.info("布局策略结果:")
        logger.info(f"  布局算法: {strategy.get('placement_strategy', 'N/A')}")
        logger.info(f"  布线策略: {strategy.get('routing_strategy', 'N/A')}")
        logger.info(f"  优化优先级: {strategy.get('optimization_priorities', [])}")
        logger.info(f"  参数建议: {strategy.get('parameter_suggestions', {})}")
        logger.info(f"  质量目标: {strategy.get('quality_targets', {})}")
        
        # 测试用例2：设计分析
        logger.info("\n--- 测试2：设计分析 ---")
        design_info = {
            'design_type': 'matrix_multiplier',
            'component_count': 3000,
            'net_count': 5000,
            'technology_node': '28nm',
            'area': 5000,
            'power': 3.0,
            'timing': 1.5
        }
        
        logger.info("分析设计...")
        analysis = llm_manager.analyze_design(design_info)
        
        logger.info("设计分析结果:")
        logger.info(f"  复杂度评估: {analysis.get('complexity_level', 'N/A')}")
        logger.info(f"  关键特征: {analysis.get('key_features', [])}")
        logger.info(f"  约束分析: {analysis.get('constraint_analysis', {})}")
        logger.info(f"  优化建议: {analysis.get('optimization_suggestions', [])}")
        
        # 测试用例3：布局分析
        logger.info("\n--- 测试3：布局分析 ---")
        layout_info = {
            'hpwl': 18000,
            'area_utilization': 0.75,
            'timing_slack': 0.05,
            'power_consumption': 4.2,
            'congestion': 0.15,
            'routing_completion': 0.95
        }
        
        logger.info("分析布局...")
        layout_analysis = llm_manager.analyze_layout(layout_info)
        
        logger.info("布局分析结果:")
        logger.info(f"  质量评分: {layout_analysis.get('quality_score', 'N/A')}")
        logger.info(f"  问题列表: {layout_analysis.get('issues', [])}")
        logger.info(f"  优化建议: {layout_analysis.get('suggestions', [])}")
        logger.info(f"  优化优先级: {layout_analysis.get('optimization_priority', 'N/A')}")
        
        # 测试用例4：直接生成测试
        logger.info("\n--- 测试4：直接生成测试 ---")
        test_prompt = """你是一个专业的芯片布局设计专家。请为以下设计生成具体的布局策略：

设计类型: RISC-V处理器
技术节点: 14nm
组件数量: 5000
网络数量: 8000
面积约束: 10000
时序约束: 2.0ns
功耗预算: 5W

请提供：
1. 具体的布局算法选择
2. 详细的参数设置
3. 优化策略
4. 预期效果

请以JSON格式返回具体策略。"""
        
        logger.info("直接生成测试...")
        response = llm_manager.generate(test_prompt, model_type='layout')
        
        logger.info("直接生成结果:")
        logger.info(f"  回复长度: {len(response)} 字符")
        logger.info(f"  是否包含JSON: {'{' in response and '}' in response}")
        logger.info(f"  是否包含具体策略: {'placement' in response.lower() or 'routing' in response.lower()}")
        logger.info(f"  回复前100字符: {response[:100]}...")
        
        # 质量评估
        logger.info("\n=== 质量评估 ===")
        
        quality_indicators = {
            'has_concrete_strategy': any(keyword in str(strategy).lower() for keyword in ['placement', 'routing', 'optimization']),
            'has_parameters': 'parameter_suggestions' in strategy and strategy['parameter_suggestions'],
            'has_quality_targets': 'quality_targets' in strategy and strategy['quality_targets'],
            'has_execution_plan': 'execution_plan' in strategy and strategy['execution_plan'],
            'response_not_conservative': '没有能力' not in response and '无法生成' not in response and '无法回答' not in response,
            'response_has_json': '{' in response and '}' in response,
            'response_has_concrete_content': len(response) > 200
        }
        
        logger.info("质量指标:")
        for indicator, value in quality_indicators.items():
            status = "✅" if value else "❌"
            logger.info(f"  {indicator}: {status}")
        
        # 计算总体质量分数
        quality_score = sum(quality_indicators.values()) / len(quality_indicators) * 100
        logger.info(f"\n总体质量分数: {quality_score:.1f}%")
        
        if quality_score >= 80:
            logger.info("✅ LLM回复质量良好")
        elif quality_score >= 60:
            logger.info("⚠️ LLM回复质量一般，需要进一步优化")
        else:
            logger.info("❌ LLM回复质量较差，需要大幅优化")
        
        logger.info("\n=== LLM回复质量测试完成 ===")
        
        return quality_score
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return 0

if __name__ == "__main__":
    quality_score = test_llm_response_quality()
    print(f"\n最终质量分数: {quality_score:.1f}%") 