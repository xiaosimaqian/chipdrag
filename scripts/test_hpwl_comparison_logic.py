#!/usr/bin/env python3
"""
测试修正后的HPWL比较逻辑
验证：原值来自OpenROAD默认参数布局，新值来自ChipDRAG优化参数布局
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiment import UnifiedPaperExperiment

def test_hpwl_comparison_logic():
    """测试HPWL比较逻辑"""
    print("=== 测试修正后的HPWL比较逻辑 ===")
    
    # 初始化实验类
    experiment = UnifiedPaperExperiment()
    
    # 测试设计目录
    test_design_dir = Path("dataset/ispd_2015_contest_benchmark/mgc_des_perf_b")
    
    if not test_design_dir.exists():
        print(f"错误：测试设计目录不存在: {test_design_dir}")
        return
    
    print(f"测试设计目录: {test_design_dir}")
    
    # 测试OpenROAD默认布局生成
    print("\n=== 测试OpenROAD默认布局生成 ===")
    openroad_default_hpwl = experiment._generate_openroad_default_layout(test_design_dir)
    
    if openroad_default_hpwl is not None:
        print(f"✅ OpenROAD默认布局HPWL: {openroad_default_hpwl:.0f}")
        print(f"   实际微米值: {openroad_default_hpwl/1000.0:.2f}")
    else:
        print("❌ OpenROAD默认布局生成失败")
    
    # 测试ChipDRAG优化布局生成
    print("\n=== 测试ChipDRAG优化布局生成 ===")
    chipdrag_optimized_hpwl = experiment._generate_chipdrag_optimized_layout(test_design_dir)
    
    if chipdrag_optimized_hpwl is not None:
        print(f"✅ ChipDRAG优化布局HPWL: {chipdrag_optimized_hpwl:.0f}")
        print(f"   实际微米值: {chipdrag_optimized_hpwl/1000.0:.2f}")
    else:
        print("❌ ChipDRAG优化布局生成失败")
    
    # 比较结果
    if openroad_default_hpwl is not None and chipdrag_optimized_hpwl is not None:
        print(f"\n=== HPWL比较结果 ===")
        print(f"原值 (OpenROAD默认): {openroad_default_hpwl:.0f}")
        print(f"新值 (ChipDRAG优化): {chipdrag_optimized_hpwl:.0f}")
        
        improvement = ((openroad_default_hpwl - chipdrag_optimized_hpwl) / openroad_default_hpwl) * 100
        print(f"改进率: {improvement:.2f}%")
        
        if improvement > 0:
            print("✅ ChipDRAG优化效果良好")
        elif improvement == 0:
            print("⚠️ ChipDRAG优化效果持平")
        else:
            print("❌ ChipDRAG优化效果不佳")
    else:
        print("\n❌ 无法进行HPWL比较，布局生成失败")
    
    # 验证文件生成
    print(f"\n=== 验证生成的文件 ===")
    placed_def = test_design_dir / "placed.def"
    if placed_def.exists():
        print(f"✅ 找到生成的DEF文件: {placed_def}")
        print(f"   文件大小: {placed_def.stat().st_size} 字节")
    else:
        print("❌ 未找到生成的DEF文件")

def test_single_layout_generation():
    """测试单个布局生成过程"""
    print("\n=== 测试单个布局生成过程 ===")
    
    experiment = UnifiedPaperExperiment()
    test_design_dir = Path("dataset/ispd_2015_contest_benchmark/mgc_des_perf_b")
    
    # 测试默认策略
    print("测试OpenROAD默认策略...")
    default_strategy = {
        'parameters': {
            'utilization': 0.7,
            'aspect_ratio': 1.0,
            'placement_density': 0.7,
            'overflow_threshold': 0.15
        },
        'strategy_type': 'openroad_default'
    }
    
    success = experiment._execute_openroad_layout(test_design_dir, default_strategy)
    print(f"OpenROAD默认布局执行结果: {'✅ 成功' if success else '❌ 失败'}")
    
    # 检查生成的DEF文件
    placed_def = test_design_dir / "placed.def"
    if placed_def.exists():
        print(f"✅ 生成DEF文件: {placed_def}")
        # 提取HPWL
        hpwl = experiment._extract_hpwl_from_def_ispd2005_style(placed_def)
        if hpwl is not None:
            print(f"   提取HPWL: {hpwl:.0f}")
        else:
            print("   HPWL提取失败")
    else:
        print("❌ 未生成DEF文件")

if __name__ == "__main__":
    test_hpwl_comparison_logic()
    test_single_layout_generation() 