#!/usr/bin/env python3
"""
测试批量训练脚本的导入
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("开始测试导入...")

try:
    # 导入OpenROAD接口
    from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface
    print("✅ 成功导入 RealOpenROADInterface")
    
    # 测试创建实例
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_a"
    interface = RealOpenROADInterface(design_path)
    print("✅ 成功创建 OpenROAD 接口实例")
    
    # 测试生成TCL脚本
    tcl_script = interface._generate_tcl_script()
    print("✅ 成功生成 TCL 脚本")
    print(f"TCL脚本长度: {len(tcl_script)} 字符")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成") 