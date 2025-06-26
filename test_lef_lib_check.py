#!/usr/bin/env python3
"""
测试LEF/LIB文件检查功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chipdrag.enhanced_openroad_interface import EnhancedOpenROADInterface

def test_lef_lib_check():
    """测试LEF/LIB文件检查功能"""
    
    # 设计目录
    design_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    
    # 检查目录是否存在
    if not os.path.exists(design_dir):
        print(f"设计目录不存在: {design_dir}")
        return
    
    print(f"测试目录: {design_dir}")
    
    # 列出目录中的所有文件
    print("\n目录中的文件:")
    for file_path in Path(design_dir).iterdir():
        if file_path.is_file():
            print(f"  {file_path.name}")
    
    # 创建接口实例
    interface = EnhancedOpenROADInterface()
    
    # 运行文件检查（不执行OpenROAD，只检查文件）
    print("\n开始文件检查...")
    
    # 检查LEF文件
    lef_files = list(Path(design_dir).glob("*.lef"))
    lef_files = [str(f) for f in lef_files]
    if not lef_files:
        print("❌ 未找到LEF文件")
        return
    else:
        print(f"✅ 找到LEF文件: {lef_files}")
    
    # 检查LIB文件
    lib_files = list(Path(design_dir).glob("*.lib"))
    lib_files = [str(f) for f in lib_files]
    if not lib_files:
        print("⚠️  未找到LIB文件，部分功能如时序/功耗分析可能不可用")
    else:
        print(f"✅ 找到LIB文件: {lib_files}")
    
    # 检查Verilog文件
    v_files = list(Path(design_dir).glob("*.v"))
    v_files = [str(f) for f in v_files]
    if len(v_files) == 0:
        print("❌ 未找到Verilog文件")
        return
    elif len(v_files) > 1:
        print(f"⚠️  找到多个Verilog文件: {v_files}")
    else:
        print(f"✅ 找到Verilog文件: {v_files[0]}")
    
    # 检查DEF文件
    def_files = list(Path(design_dir).glob("*.def"))
    def_files = [str(f) for f in def_files]
    if len(def_files) == 0:
        print("❌ 未找到DEF文件")
        return
    else:
        print(f"✅ 找到DEF文件: {def_files}")
    
    print("\n文件检查完成！")
    
    # 测试TCL脚本生成（不执行）
    print("\n测试TCL脚本生成...")
    try:
        verilog_file = v_files[0]
        def_file = def_files[0]
        
        tcl_path = interface.create_iterative_placement_tcl(
            verilog_file=verilog_file,
            lef_files=lef_files,
            def_file=def_file,
            work_dir=design_dir,
            num_iterations=3
        )
        
        print(f"✅ TCL脚本生成成功: {tcl_path}")
        
        # 显示TCL脚本的前几行
        with open(tcl_path, 'r') as f:
            lines = f.readlines()
            print("\nTCL脚本前10行:")
            for i, line in enumerate(lines[:10]):
                print(f"  {i+1:2d}: {line.rstrip()}")
        
    except Exception as e:
        print(f"❌ TCL脚本生成失败: {e}")

if __name__ == "__main__":
    test_lef_lib_check() 