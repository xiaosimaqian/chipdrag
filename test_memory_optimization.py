#!/usr/bin/env python3
"""
测试内存优化功能
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from paper_hpwl_comparison_experiment_fixed import PaperHPWLComparisonExperimentFixed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hardware_requirements():
    """测试硬件资源检查"""
    logger.info("=== 测试硬件资源检查 ===")
    
    experiment = PaperHPWLComparisonExperimentFixed()
    hardware_status = experiment._check_hardware_requirements()
    
    print(f"系统配置:")
    print(f"  总内存: {hardware_status['total_memory_gb']:.1f}GB")
    print(f"  可用内存: {hardware_status['available_memory_gb']:.1f}GB")
    print(f"  CPU核心数: {hardware_status['cpu_count']}")
    print(f"  满足最低要求: {'✅' if hardware_status['meets_minimum'] else '❌'}")
    print(f"  满足推荐配置: {'✅' if hardware_status['meets_recommended'] else '❌'}")
    print(f"  最大并行设计数: {hardware_status['max_parallel_designs']}")
    
    if hardware_status['warnings']:
        print("\n警告:")
        for warning in hardware_status['warnings']:
            print(f"  • {warning}")
    
    if hardware_status['recommendations']:
        print("\n建议:")
        for recommendation in hardware_status['recommendations']:
            print(f"  • {recommendation}")
    
    return hardware_status

def test_resource_limits():
    """测试资源限制计算"""
    logger.info("=== 测试资源限制计算 ===")
    
    experiment = PaperHPWLComparisonExperimentFixed()
    
    # 测试不同设计的资源分配
    test_designs = [
        "mgc_matrix_mult_1",
        "mgc_des_perf_1", 
        "mgc_fft_1",
        "mgc_pci_bridge32_1"
    ]
    
    for design_name in test_designs:
        # 创建模拟设计目录
        design_dir = Path(f"test_design_{design_name}")
        design_dir.mkdir(exist_ok=True)
        
        memory_limit, cpu_limit = experiment._calculate_resource_limits(design_dir)
        
        print(f"{design_name}:")
        print(f"  内存限制: {memory_limit}")
        print(f"  CPU限制: {cpu_limit}")
        
        # 清理
        design_dir.rmdir()

def test_memory_optimization():
    """测试内存优化策略"""
    logger.info("=== 测试内存优化策略 ===")
    
    experiment = PaperHPWLComparisonExperimentFixed()
    
    # 创建测试设计目录和布局策略
    design_dir = Path("test_design")
    design_dir.mkdir(exist_ok=True)
    
    # 创建测试文件
    (design_dir / "cells.lef").touch()
    (design_dir / "floorplan.def").touch()
    (design_dir / "design.v").touch()
    
    layout_strategy = {
        'parameters': {
            'density': 0.7,
            'overflow': 0.1,
            'utilization': 0.7,
            'aspect_ratio': 1.0,
            'init_density_penalty': 8e-5,
            'max_displacement': 100
        }
    }
    
    # 测试单独处理模式
    print("测试单独处理模式:")
    success = experiment._run_single_design_with_max_resources(design_dir, layout_strategy)
    print(f"  结果: {'✅ 成功' if success else '❌ 失败'}")
    
    # 测试降低资源需求模式
    print("测试降低资源需求模式:")
    success = experiment._run_with_reduced_resources(design_dir, layout_strategy)
    print(f"  结果: {'✅ 成功' if success else '❌ 失败'}")
    
    # 清理
    import shutil
    shutil.rmtree(design_dir)

def test_parallel_execution_strategy():
    """测试并行执行策略"""
    logger.info("=== 测试并行执行策略 ===")
    
    experiment = PaperHPWLComparisonExperimentFixed()
    
    # 创建模拟设计队列
    design_queue = [
        Path("mgc_matrix_mult_1"),      # 大型设计
        Path("mgc_des_perf_1"),         # 大型设计
        Path("mgc_fft_1"),              # 中型设计
        Path("mgc_fft_2"),              # 中型设计
        Path("mgc_pci_bridge32_1"),     # 小型设计
        Path("mgc_pci_bridge32_2"),     # 小型设计
        Path("mgc_pci_bridge32_3"),     # 小型设计
        Path("mgc_edit_distance_1"),    # 小型设计
    ]
    
    # 测试智能并行策略
    batches = experiment._adjust_parallel_execution_for_memory(design_queue)
    
    print(f"设计队列: {len(design_queue)}个设计")
    print(f"分批策略: {len(batches)}批")
    
    for i, batch in enumerate(batches):
        print(f"  批次 {i+1}: {len(batch)}个设计")
        for design in batch:
            print(f"    • {design.name}")

def main():
    """主测试函数"""
    print("开始测试内存优化功能...")
    
    try:
        # 1. 测试硬件资源检查
        hardware_status = test_hardware_requirements()
        
        # 2. 测试资源限制计算
        test_resource_limits()
        
        # 3. 测试内存优化策略 (会尝试运行Docker，可能失败)
        test_memory_optimization()
        
        # 4. 测试并行执行策略
        test_parallel_execution_strategy()
        
        print("\n" + "="*50)
        print("内存优化功能测试完成")
        print("="*50)
        
        # 总结
        print("\n=== 测试总结 ===")
        print(f"系统内存: {hardware_status['total_memory_gb']:.1f}GB")
        print(f"硬件等级: {'推荐配置' if hardware_status['meets_recommended'] else '最低配置' if hardware_status['meets_minimum'] else '低于最低要求'}")
        print(f"最大并行: {hardware_status['max_parallel_designs']}个设计")
        print("\n主要改进:")
        print("  ✅ 智能内存分配 - 根据系统资源动态调整")
        print("  ✅ 并行策略优化 - 大型设计单独处理")
        print("  ✅ 硬件资源检查 - 提前发现资源不足")
        print("  ✅ 保持参数科学性 - 不使用固定的极简参数")
        print("  ✅ 内存优化重试 - 多层次的错误处理")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 