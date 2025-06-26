#!/usr/bin/env python3
"""
测试修正后的OpenROAD接口
根据OpenROAD官方手册验证接口的正确性
"""

import sys
import os
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent / "modules" / "rl_training"))

from real_openroad_interface_fixed import RealOpenROADInterface

def test_openroad_interface():
    """测试OpenROAD接口"""
    print("=== 测试修正后的OpenROAD接口 ===")
    
    try:
        # 创建接口实例
        interface = RealOpenROADInterface()
        
        print("✓ 接口初始化成功")
        print(f"  工作目录: {interface.work_dir}")
        print(f"  Verilog文件: {interface.verilog_file}")
        print(f"  DEF文件: {interface.def_file}")
        print(f"  LEF文件: {interface.tech_lef}, {interface.cells_lef}")
        print(f"  LIB文件数量: {len(interface.lib_files)}")
        
        # 生成TCL脚本
        tcl_script = interface._generate_tcl_script(
            density_target=0.7,
            wirelength_weight=1.0,
            density_weight=1.0
        )
        
        print("\n=== 生成的TCL脚本 ===")
        print(tcl_script)
        
        # 检查TCL脚本中的关键命令
        print("\n=== 验证TCL脚本 ===")
        
        # 检查LEF读取命令
        if "read_lef -tech" in tcl_script:
            print("✓ 技术LEF读取命令正确")
        else:
            print("✗ 技术LEF读取命令错误")
            
        if "read_lef -library" in tcl_script:
            print("✓ 单元LEF读取命令正确")
        else:
            print("✗ 单元LEF读取命令错误")
        
        # 检查Verilog读取和连接
        if "read_verilog" in tcl_script:
            print("✓ Verilog读取命令存在")
        else:
            print("✗ Verilog读取命令缺失")
            
        if "link_design top" in tcl_script:
            print("✓ 设计连接命令正确")
        else:
            print("✗ 设计连接命令错误")
        
        # 检查DEF读取
        if "read_def" in tcl_script:
            print("✓ DEF读取命令存在")
        else:
            print("✗ DEF读取命令缺失")
        
        # 检查布局命令
        if "global_placement" in tcl_script:
            print("✓ 全局布局命令存在")
        else:
            print("✗ 全局布局命令缺失")
            
        if "detailed_placement" in tcl_script:
            print("✓ 详细布局命令存在")
        else:
            print("✗ 详细布局命令缺失")
        
        print("\n=== 测试完成 ===")
        
        # 自动运行实际测试
        print("\n=== 运行实际OpenROAD测试 ===")
        result = interface.run_placement(
            density_target=0.7,
            wirelength_weight=1.0,
            density_weight=1.0
        )
        
        print(f"执行结果: {'成功' if result['success'] else '失败'}")
        print(f"返回码: {result['return_code']}")
        print(f"执行时间: {result['execution_time']:.2f}秒")
        
        if not result['success']:
            print("\n=== 错误信息 ===")
            if isinstance(result['stderr'], list):
                for error in result['stderr'][-10:]:  # 显示最后10行错误
                    print(error)
            else:
                print(result['stderr'])
        else:
            print("\n=== 成功信息 ===")
            if isinstance(result['stdout'], list) and result['stdout']:
                for line in result['stdout'][-10:]:  # 显示最后10行输出
                    print(line)
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openroad_interface() 