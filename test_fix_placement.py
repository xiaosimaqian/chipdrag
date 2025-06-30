#!/usr/bin/env python3
"""
测试详细布局修复
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入OpenROAD接口
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

def test_fixed_placement():
    """测试修复后的布局"""
    print("=== 测试详细布局修复 ===")
    
    # 使用之前失败的设计
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_a"
    
    print(f"测试设计: {design_path}")
    
    try:
        # 创建接口实例
        interface = RealOpenROADInterface(design_path)
        
        # 运行布局优化
        print("运行布局优化...")
        start_time = time.time()
        result = interface.run_placement()
        execution_time = time.time() - start_time
        
        print(f"结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"线长: {result['wirelength']}")
        print(f"面积: {result['area']}")
        
        if result['success']:
            print("🎉 详细布局修复成功！")
        else:
            print(f"错误: {result['stderr']}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_fixed_placement() 