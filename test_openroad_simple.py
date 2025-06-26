#!/usr/bin/env python3
"""
简化的OpenROAD测试脚本
逐步测试OpenROAD的各个组件
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_docker_basic():
    """测试Docker基本功能"""
    print("=== 测试Docker基本功能 ===")
    
    cmd = [
        "docker", "run", "--rm",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c", "echo 'Docker container works'"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"返回码: {result.returncode}")
        print(f"输出: {result.stdout}")
        print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_openroad_version():
    """测试OpenROAD版本"""
    print("\n=== 测试OpenROAD版本 ===")
    
    cmd = [
        "docker", "run", "--rm",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c", "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -version"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"返回码: {result.returncode}")
        print(f"输出: {result.stdout}")
        print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_simple_tcl():
    """测试简单的TCL脚本"""
    print("\n=== 测试简单TCL脚本 ===")
    
    simple_tcl = """
puts "Hello from OpenROAD"
exit
"""
    
    # 写入临时TCL文件
    with open("test_simple.tcl", "w") as f:
        f.write(simple_tcl)
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}:/workspace",
        "-w", "/workspace",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c", "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -no_init -no_splash -exit test_simple.tcl"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"返回码: {result.returncode}")
        print(f"输出: {result.stdout}")
        print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_lef_reading():
    """测试LEF文件读取"""
    print("\n=== 测试LEF文件读取 ===")
    
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_a"
    
    tcl_script = """
puts "开始读取LEF文件"
read_lef -tech tech.lef
puts "tech.lef读取完成"
read_lef -library cells.lef
puts "cells.lef读取完成"
puts "LEF文件读取测试完成"
exit
"""
    
    # 写入TCL文件
    tcl_path = os.path.join(design_path, "test_lef.tcl")
    with open(tcl_path, "w") as f:
        f.write(tcl_script)
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.abspath(design_path)}:/workspace",
        "-w", "/workspace",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c", "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -no_init -no_splash -exit test_lef.tcl"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(f"返回码: {result.returncode}")
        print(f"输出: {result.stdout}")
        print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_verilog_reading():
    """测试Verilog文件读取"""
    print("\n=== 测试Verilog文件读取 ===")
    
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_a"
    
    tcl_script = """
puts "开始读取LEF文件"
read_lef -tech tech.lef
read_lef -library cells.lef
puts "LEF文件读取完成"

puts "开始读取Verilog文件"
read_verilog design.v
puts "Verilog文件读取完成"

puts "开始连接设计"
link_design des_perf
puts "设计连接完成"

puts "Verilog读取测试完成"
exit
"""
    
    # 写入TCL文件
    tcl_path = os.path.join(design_path, "test_verilog.tcl")
    with open(tcl_path, "w") as f:
        f.write(tcl_script)
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.abspath(design_path)}:/workspace",
        "-w", "/workspace",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c", "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -no_init -no_splash -exit test_verilog.tcl"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(f"返回码: {result.returncode}")
        print(f"输出: {result.stdout}")
        print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    """主函数"""
    print("开始OpenROAD逐步测试...")
    
    tests = [
        ("Docker基本功能", test_docker_basic),
        ("OpenROAD版本", test_openroad_version),
        ("简单TCL脚本", test_simple_tcl),
        ("LEF文件读取", test_lef_reading),
        ("Verilog文件读取", test_verilog_reading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print('='*50)
        
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"❌ 测试失败: {test_name}")
            break
        else:
            print(f"✅ 测试通过: {test_name}")
    
    print(f"\n{'='*50}")
    print("测试总结")
    print('='*50)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    # 清理临时文件
    if os.path.exists("test_simple.tcl"):
        os.remove("test_simple.tcl")

if __name__ == "__main__":
    main() 