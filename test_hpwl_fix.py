#!/usr/bin/env python3
"""
测试HPWL提取修复
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

def test_hpwl_extraction():
    """测试HPWL提取功能"""
    print("=== 测试HPWL提取修复 ===")
    
    # 使用一个简单的设计进行测试
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_fft_1"
    
    if not os.path.exists(design_path):
        print(f"设计路径不存在: {design_path}")
        return
    
    try:
        # 创建接口实例
        interface = RealOpenROADInterface(work_dir=design_path)
        
        print(f"测试设计: {design_path}")
        print(f"设计文件: {interface.verilog_file.name}")
        print(f"DEF文件: {interface.def_file.name}")
        
        # 测试HPWL提取方法
        print("\n1. 测试从DEF文件提取HPWL...")
        hpwl_from_def = interface._extract_hpwl_from_def("placement_result.def")
        print(f"   从DEF提取的HPWL: {hpwl_from_def}")
        
        print("\n2. 测试从日志文件提取HPWL...")
        hpwl_from_log = interface._extract_hpwl_from_log("openroad_execution.log")
        print(f"   从日志提取的HPWL: {hpwl_from_log}")
        
        # 测试运行布局
        print("\n3. 测试运行布局...")
        result = interface.run_placement(
            density_target=0.75,
            wirelength_weight=1.0,
            density_weight=1.0
        )
        
        print(f"   布局结果: {result['success']}")
        print(f"   提取的HPWL: {result.get('hpwl', 'N/A')}")
        print(f"   执行时间: {result.get('execution_time', 0):.2f}秒")
        
        if not result['success']:
            print(f"   错误信息: {result.get('stderr', 'N/A')}")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hpwl_extraction() 