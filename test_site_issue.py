#!/usr/bin/env python3
"""
测试OpenROAD站点问题
"""

import os
import subprocess

def test_site_issue():
    """测试站点问题"""
    print("=== 测试站点问题 ===")
    
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_a"
    
    # 测试1: 先读取tech.lef，再读取cells.lef
    tcl_script1 = """
puts "=== 测试1: 先读取tech.lef，再读取cells.lef ==="
read_lef -tech tech.lef
puts "tech.lef读取完成"
read_lef -library cells.lef
puts "cells.lef读取完成"

# 列出所有站点
puts "\\n=== 列出所有站点 ==="
set sites [get_sites]
puts "站点数量: [llength \$sites]"
foreach site \$sites {
    puts "站点: \$site"
}

# 列出所有宏单元
puts "\\n=== 列出所有宏单元 ==="
set masters [get_lib_cells]
puts "宏单元数量: [llength \$masters]"
foreach master \$masters {
    puts "宏单元: \$master"
    set site [get_attribute \$master site_name]
    puts "  站点: \$site"
}

exit
"""
    
    # 写入TCL文件
    tcl_path1 = os.path.join(design_path, "test_site1.tcl")
    with open(tcl_path1, "w") as f:
        f.write(tcl_script1)
    
    cmd1 = [
        "docker", "run", "--rm",
        "-v", f"{os.path.abspath(design_path)}:/workspace",
        "-w", "/workspace",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c", "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -no_init -no_splash -exit test_site1.tcl"
    ]
    
    print("运行测试1...")
    try:
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=60)
        print(f"返回码: {result1.returncode}")
        print(f"输出: {result1.stdout}")
        print(f"错误: {result1.stderr}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_site_issue() 