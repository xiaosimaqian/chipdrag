#!/usr/bin/env python3
"""
智能参数系统测试脚本
验证动态参数调整功能
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入OpenROAD接口
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

def test_smart_parameters():
    """测试智能参数系统"""
    print("=== 智能参数系统测试 ===")
    
    # 测试不同规模的设计
    test_designs = [
        "mgc_des_perf_b",      # 小型设计
        "mgc_fft_1",           # 中型设计
        "mgc_des_perf_a",      # 大型设计 (之前失败的)
    ]
    
    for design_name in test_designs:
        design_path = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
        
        if not os.path.exists(design_path):
            print(f"⚠️ 设计路径不存在: {design_path}")
            continue
            
        print(f"\n--- 测试设计: {design_name} ---")
        
        try:
            # 创建接口实例
            interface = RealOpenROADInterface(design_path)
            
            # 获取设计统计
            design_stats = interface._extract_design_stats()
            print(f"设计统计: {design_stats}")
            
            # 计算最优参数
            optimal_params = interface._calculate_optimal_parameters(design_stats)
            print(f"最优参数: {optimal_params}")
            
            # 生成TCL脚本
            tcl_script = interface._generate_tcl_script(
                density_target=optimal_params['density_target'],
                die_size=optimal_params['die_size'],
                core_size=optimal_params['core_size']
            )
            
            # 保存TCL脚本
            tcl_file = os.path.join(design_path, "smart_test_script.tcl")
            with open(tcl_file, 'w') as f:
                f.write(tcl_script)
            
            print(f"✅ TCL脚本已生成: {tcl_file}")
            
            # 运行布局优化
            print("运行布局优化...")
            result = interface.run_placement()
            
            print(f"结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
            print(f"执行时间: {result['execution_time']:.2f}秒")
            print(f"线长: {result['wirelength']}")
            print(f"面积: {result['area']}")
            
            if not result['success']:
                print(f"错误: {result['stderr']}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")

def test_parameter_calculation():
    """测试参数计算逻辑"""
    print("\n=== 参数计算测试 ===")
    
    # 模拟不同规模的设计统计
    test_cases = [
        {"num_instances": 5000, "num_nets": 6000, "core_area": 100000, "name": "小型设计"},
        {"num_instances": 25000, "num_nets": 30000, "core_area": 300000, "name": "中型设计"},
        {"num_instances": 60000, "num_nets": 70000, "core_area": 600000, "name": "大型设计"},
        {"num_instances": 120000, "num_nets": 150000, "core_area": 800000, "name": "超大型设计"},
    ]
    
    interface = RealOpenROADInterface("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_b")
    
    for case in test_cases:
        params = interface._calculate_optimal_parameters(case)
        print(f"\n{case['name']}:")
        print(f"  实例数: {case['num_instances']}, 网络数: {case['num_nets']}")
        print(f"  密度目标: {params['density_target']}")
        print(f"  芯片尺寸: {params['die_size']}x{params['die_size']}")
        print(f"  核心尺寸: {params['core_size']}x{params['core_size']}")

if __name__ == "__main__":
    # 测试参数计算
    test_parameter_calculation()
    
    # 测试实际设计
    test_smart_parameters() 