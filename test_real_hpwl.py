#!/usr/bin/env python3
"""测试从真正的布局结果中提取HPWL"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

def test_real_hpwl():
    """测试从真正的布局结果中提取HPWL"""
    # 使用一个有真正布局结果的设计
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    
    print(f"测试设计: {design_path}")
    
    try:
        # 初始化接口
        interface = RealOpenROADInterface(work_dir=design_path)
        
        # 测试从真正的布局结果提取HPWL
        print("从真正的布局结果提取HPWL...")
        hpwl = interface._extract_hpwl_from_def("placement_result.def")
        print(f"提取的HPWL: {hpwl}")
        
        if hpwl != float('inf') and hpwl > 10000:
            print("✅ 成功提取到合理的HPWL值！")
        else:
            print("❌ HPWL提取失败或值异常")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_real_hpwl() 