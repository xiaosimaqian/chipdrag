#!/usr/bin/env python3
"""
测试HPWL提取功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

def test_hpwl_extraction():
    """测试HPWL提取"""
    # 使用一个简单的设计进行测试
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_fft_1"
    
    print(f"测试设计: {design_path}")
    
    try:
        # 初始化接口
        interface = RealOpenROADInterface(design_path)
        
        # 运行布局
        print("运行OpenROAD布局...")
        result = interface.run_placement()
        
        print(f"成功: {result['success']}")
        print(f"执行时间: {result.get('execution_time', 0):.2f}秒")
        
        # 检查提取的指标
        metrics = result.get('metrics', {})
        print(f"提取的指标: {metrics}")
        
        # 检查HPWL
        wirelength = metrics.get('wirelength')
        area = metrics.get('area')
        density = metrics.get('density')
        overflow = metrics.get('overflow')
        
        print(f"HPWL: {wirelength}")
        print(f"面积: {area}")
        print(f"密度: {density}")
        print(f"溢出: {overflow}")
        
        if wirelength and wirelength != 31.0:
            print("✅ HPWL提取成功！")
        else:
            print("❌ HPWL提取失败或值异常")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_hpwl_extraction() 