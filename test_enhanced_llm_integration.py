#!/usr/bin/env python3
"""
测试增强的LLM集成功能
验证四个关键问题的修复：
1. 数据格式修复
2. 策略多样性增强
3. 错误处理完善
4. 反馈机制建立
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from modules.utils.llm_manager import LLMManager
from modules.utils.config_loader import ConfigLoader
from paper_hpwl_comparison_experiment import EnhancedLLMIntegration

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_llm_integration():
    """测试增强的LLM集成功能"""
    logger.info("=== 开始测试增强的LLM集成功能 ===")
    
    # 1. 初始化LLM管理器和增强集成
    try:
        config = ConfigLoader().load_config('experiment_config.json')
        llm_manager = LLMManager(config.get('llm', {}))
        enhanced_llm = EnhancedLLMIntegration(llm_manager)
        logger.info("✅ LLM管理器和增强集成初始化成功")
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        return False
    
    # 2. 测试数据格式修复
    logger.info("\n--- 测试数据格式修复 ---")
    test_design_info = {
        'design_name': 'test_design',
        'num_components': 1000,
        'num_nets': 500,
        'design_type': 'des_perf',
        'complexity_level': 'medium',
        'area_utilization': 0.7,
        'component_density': 0.1,
        'design_size': 'medium',
        'area': 1000000
    }
    
    test_layout_strategy = {
        'parameter_suggestions': {
            'density_target': 0.8,
            'aspect_ratio': 1.2,
            'core_space': 1.5
        }
    }
    
    test_reward = 500000.0
    
    try:
        fixed_layout_result = enhanced_llm.fix_layout_data_format(
            test_design_info, test_layout_strategy, test_reward
        )
        
        # 验证修复结果
        assert 'components' in fixed_layout_result
        assert 'nets' in fixed_layout_result
        assert isinstance(fixed_layout_result['components'], list)
        assert isinstance(fixed_layout_result['nets'], list)
        assert len(fixed_layout_result['components']) > 0
        assert len(fixed_layout_result['nets']) > 0
        
        logger.info(f"✅ 数据格式修复成功:")
        logger.info(f"   - 组件数量: {len(fixed_layout_result['components'])}")
        logger.info(f"   - 网络数量: {len(fixed_layout_result['nets'])}")
        logger.info(f"   - 时序评分: {fixed_layout_result['timing']:.3f}")
        logger.info(f"   - 功耗评分: {fixed_layout_result['power']:.3f}")
        logger.info(f"   - 拥塞评分: {fixed_layout_result['congestion']:.3f}")
        
    except Exception as e:
        logger.error(f"❌ 数据格式修复失败: {e}")
        return False
    
    # 3. 测试策略多样性增强
    logger.info("\n--- 测试策略多样性增强 ---")
    
    # 模拟检索案例
    class MockRetrievalResult:
        def __init__(self, knowledge):
            self.knowledge = knowledge
    
    mock_cases = [
        MockRetrievalResult({
            'timing_constraints': True,
            'parameters': {'utilization': 0.75, 'aspect_ratio': 1.1, 'core_space': 1.8}
        }),
        MockRetrievalResult({
            'area_constraints': True,
            'parameters': {'utilization': 0.8, 'aspect_ratio': 1.0, 'core_space': 1.5}
        })
    ]
    
    # 模拟RL动作
    class MockAction:
        def __init__(self, k_value, confidence):
            self.k_value = k_value
            self.confidence = confidence
            self.exploration_type = 'epsilon_greedy'
    
    mock_action = MockAction(k_value=12, confidence=0.85)
    
    try:
        diverse_strategy = enhanced_llm.generate_diverse_layout_strategy(
            mock_cases, mock_action, test_design_info
        )
        
        # 验证策略生成
        assert isinstance(diverse_strategy, str)
        assert 'initialize_floorplan' in diverse_strategy
        assert 'global_placement' in diverse_strategy
        assert 'detailed_placement' in diverse_strategy
        
        logger.info(f"✅ 策略多样性增强成功:")
        logger.info(f"   - 策略类型: {enhanced_llm._select_strategy_type(mock_cases, test_design_info, mock_action)}")
        logger.info(f"   - 策略长度: {len(diverse_strategy)} 字符")
        logger.info(f"   - 包含时序优化: {'estimate_parasitics' in diverse_strategy}")
        
    except Exception as e:
        logger.error(f"❌ 策略多样性增强失败: {e}")
        return False
    
    # 4. 测试错误处理
    logger.info("\n--- 测试错误处理 ---")
    
    # 测试不同类型的错误
    error_types = [
        ('timeout', Exception("execution time exceeded")),
        ('memory', Exception("out of memory")),
        ('format_error', Exception("invalid format")),
        ('connection', Exception("connection refused")),
        ('unknown', Exception("unknown error"))
    ]
    
    for error_name, error in error_types:
        try:
            error_result = enhanced_llm.handle_llm_error(error, {
                'design_name': 'test_design',
                'stage': 'test'
            })
            
            assert 'quality_score' in error_result
            assert 'issues' in error_result
            assert 'suggestions' in error_result
            assert 'metadata' in error_result
            
            logger.info(f"✅ {error_name} 错误处理成功:")
            logger.info(f"   - 质量评分: {error_result['quality_score']:.3f}")
            logger.info(f"   - 问题数量: {len(error_result['issues'])}")
            logger.info(f"   - 建议数量: {len(error_result['suggestions'])}")
            
        except Exception as e:
            logger.error(f"❌ {error_name} 错误处理失败: {e}")
            return False
    
    # 5. 测试反馈机制
    logger.info("\n--- 测试反馈机制 ---")
    
    mock_llm_analysis = {
        'quality_score': 0.65,
        'area_utilization': 0.7,
        'routing_quality': 0.6,
        'timing_performance': 0.7,
        'power_distribution': 0.6,
        'issues': ['时序性能需要优化', '面积利用率偏低'],
        'suggestions': ['使用时序驱动布局', '提高面积利用率'],
        'needs_optimization': True,
        'optimization_priority': 'medium'
    }
    
    try:
        feedback_result = enhanced_llm.apply_feedback_mechanism(mock_llm_analysis, {
            'design_name': 'test_design',
            'stage': 'test',
            'reward': 600000.0
        })
        
        assert 'quality_score' in feedback_result
        assert 'needs_optimization' in feedback_result
        assert 'optimization_suggestions' in feedback_result
        assert 'system_updates' in feedback_result
        assert 'feedback_applied' in feedback_result
        
        logger.info(f"✅ 反馈机制测试成功:")
        logger.info(f"   - 质量评分: {feedback_result['quality_score']:.3f}")
        logger.info(f"   - 需要优化: {feedback_result['needs_optimization']}")
        logger.info(f"   - 优化优先级: {feedback_result['optimization_priority']}")
        logger.info(f"   - 优化建议数量: {len(feedback_result['optimization_suggestions'])}")
        logger.info(f"   - 系统更新数量: {len(feedback_result['system_updates'])}")
        logger.info(f"   - 反馈历史记录: {len(enhanced_llm.feedback_history)}")
        
    except Exception as e:
        logger.error(f"❌ 反馈机制测试失败: {e}")
        return False
    
    logger.info("\n=== 所有测试通过！增强的LLM集成功能正常 ===")
    return True

def main():
    """主函数"""
    logger.info("开始增强LLM集成功能测试")
    
    # 测试增强LLM集成功能
    test_result = test_enhanced_llm_integration()
    
    # 总结测试结果
    if test_result:
        logger.info("\n🎉 所有测试通过！增强的LLM集成已成功修复以下问题：")
        logger.info("   1. ✅ 数据格式修复：传递正确的组件列表而不是数字")
        logger.info("   2. ✅ 策略多样性增强：基于检索案例生成更丰富的布局策略")
        logger.info("   3. ✅ 错误处理完善：针对不同错误类型进行差异化处理")
        logger.info("   4. ✅ 反馈机制建立：将LLM分析结果用于系统优化")
        logger.info("\n📋 修复详情：")
        logger.info("   - 数据格式：从数字改为组件列表和网络列表")
        logger.info("   - 策略多样性：5种策略模板 + 动态参数调整")
        logger.info("   - 错误处理：5种错误类型 + 差异化处理策略")
        logger.info("   - 反馈机制：质量评估 + 优化建议 + 系统参数更新")
        return 0
    else:
        logger.error("\n❌ 测试失败！请检查增强LLM集成功能")
        return 1

if __name__ == "__main__":
    exit(main()) 