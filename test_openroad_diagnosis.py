#!/usr/bin/env python3
"""
OpenROAD 诊断脚本
用于测试OpenROAD是否能正常工作以及检查具体的错误原因
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openroad_basic():
    """测试OpenROAD基本功能"""
    print("🔍 测试OpenROAD基本功能...")
    
    try:
        # 测试OpenROAD版本
        result = subprocess.run(
            ["openroad", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"OpenROAD版本命令返回码: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ OpenROAD基本功能正常")
            return True
        else:
            print("❌ OpenROAD基本功能异常")
            return False
            
    except FileNotFoundError:
        print("❌ OpenROAD命令未找到，请检查OpenROAD是否正确安装")
        return False
    except Exception as e:
        print(f"❌ 测试OpenROAD基本功能时发生异常: {e}")
        return False

def test_openroad_simple_script():
    """测试OpenROAD简单脚本"""
    print("\n🔍 测试OpenROAD简单脚本...")
    
    # 创建一个简单的测试脚本
    test_script = """
puts "Hello from OpenROAD!"
puts "OpenROAD版本: [version]"
puts "当前目录: [pwd]"
exit 0
"""
    
    # 写入测试脚本
    script_file = Path("test_openroad_simple.tcl")
    try:
        with open(script_file, 'w') as f:
            f.write(test_script)
        
        # 执行测试脚本
        result = subprocess.run(
            ["openroad", "-no_init", "-no_splash", "-exit", str(script_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"简单脚本返回码: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # 清理
        if script_file.exists():
            script_file.unlink()
        
        if result.returncode == 0:
            print("✅ OpenROAD简单脚本执行成功")
            return True
        else:
            print("❌ OpenROAD简单脚本执行失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试OpenROAD简单脚本时发生异常: {e}")
        return False

def test_openroad_with_real_data():
    """测试OpenROAD处理真实数据"""
    print("\n🔍 测试OpenROAD处理真实数据...")
    
    # 使用mgc_fft_1数据进行测试
    test_design = Path("dataset/ispd_2015_contest_benchmark/mgc_fft_1")
    if not test_design.exists():
        print("❌ 测试数据目录不存在")
        return False
    
    # 创建测试脚本
    test_script = f"""
puts "=== OpenROAD真实数据测试 ==="
puts "当前目录: [pwd]"
puts "文件列表:"
foreach file [glob -nocomplain *] {{
    puts "  $file"
}}

# 测试读取LEF文件
puts "测试读取tech.lef..."
if {{[file exists tech.lef]}} {{
    puts "✅ tech.lef文件存在"
    if {{[catch {{read_lef tech.lef}} err]}} {{
        puts "❌ 读取tech.lef失败: $err"
        exit 1
    }} else {{
        puts "✅ tech.lef读取成功"
    }}
}} else {{
    puts "❌ tech.lef文件不存在"
    exit 1
}}

puts "测试读取cells.lef..."
if {{[file exists cells.lef]}} {{
    puts "✅ cells.lef文件存在"
    if {{[catch {{read_lef cells.lef}} err]}} {{
        puts "❌ 读取cells.lef失败: $err"
        exit 1
    }} else {{
        puts "✅ cells.lef读取成功"
    }}
}} else {{
    puts "❌ cells.lef文件不存在"
    exit 1
}}

puts "测试读取design.v..."
if {{[file exists design.v]}} {{
    puts "✅ design.v文件存在"
    if {{[catch {{read_verilog design.v}} err]}} {{
        puts "❌ 读取design.v失败: $err"
        exit 1
    }} else {{
        puts "✅ design.v读取成功"
    }}
}} else {{
    puts "❌ design.v文件不存在"
    exit 1
}}

puts "测试读取floorplan.def..."
if {{[file exists floorplan.def]}} {{
    puts "✅ floorplan.def文件存在"
    if {{[catch {{read_def floorplan.def}} err]}} {{
        puts "❌ 读取floorplan.def失败: $err"
        exit 1
    }} else {{
        puts "✅ floorplan.def读取成功"
    }}
}} else {{
    puts "❌ floorplan.def文件不存在"
    exit 1
}}

puts "=== 所有文件读取成功 ==="
exit 0
"""
    
    # 写入测试脚本
    script_file = test_design / "test_diagnosis.tcl"
    try:
        with open(script_file, 'w') as f:
            f.write(test_script)
        
        # 执行测试脚本
        result = subprocess.run(
            ["openroad", "-no_init", "-no_splash", "-exit", "test_diagnosis.tcl"],
            cwd=test_design,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"真实数据测试返回码: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # 清理
        if script_file.exists():
            script_file.unlink()
        
        if result.returncode == 0:
            print("✅ OpenROAD真实数据处理成功")
            return True
        else:
            print("❌ OpenROAD真实数据处理失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试OpenROAD真实数据处理时发生异常: {e}")
        return False

def test_openroad_placement():
    """测试OpenROAD布局功能"""
    print("\n🔍 测试OpenROAD布局功能...")
    
    # 使用mgc_fft_1数据进行测试
    test_design = Path("dataset/ispd_2015_contest_benchmark/mgc_fft_1")
    if not test_design.exists():
        print("❌ 测试数据目录不存在")
        return False
    
    # 创建布局测试脚本
    test_script = f"""
puts "=== OpenROAD布局功能测试 ==="

# 读取所有文件
read_lef tech.lef
read_lef cells.lef
read_verilog design.v
read_def floorplan.def

puts "✅ 所有文件读取完成"

# 测试初始化floorplan
puts "测试初始化floorplan..."
if {{[catch {{
    initialize_floorplan -utilization 0.7 -aspect_ratio 1.0 -core_space 20
    puts "✅ floorplan初始化成功"
}} err]}} {{
    puts "❌ floorplan初始化失败: $err"
    puts "尝试备用方法..."
    
    if {{[catch {{
        initialize_floorplan -die_area {{0 0 2000 2000}} -core_area {{100 100 1900 1900}}
        puts "✅ 备用floorplan初始化成功"
    }} err2]}} {{
        puts "❌ 备用floorplan初始化也失败: $err2"
        exit 1
    }}
}}

puts "=== 布局初始化测试完成 ==="
exit 0
"""
    
    # 写入测试脚本
    script_file = test_design / "test_placement.tcl"
    try:
        with open(script_file, 'w') as f:
            f.write(test_script)
        
        # 执行测试脚本
        result = subprocess.run(
            ["openroad", "-no_init", "-no_splash", "-exit", "test_placement.tcl"],
            cwd=test_design,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(f"布局功能测试返回码: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # 清理
        if script_file.exists():
            script_file.unlink()
        
        if result.returncode == 0:
            print("✅ OpenROAD布局功能测试成功")
            return True
        else:
            print("❌ OpenROAD布局功能测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试OpenROAD布局功能时发生异常: {e}")
        return False

def check_system_requirements():
    """检查系统要求"""
    print("\n🔍 检查系统要求...")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查操作系统
    print(f"操作系统: {os.name}")
    
    # 检查环境变量
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")
    
    # 检查OpenROAD路径
    try:
        which_result = subprocess.run(
            ["which", "openroad"],
            capture_output=True,
            text=True
        )
        print(f"OpenROAD路径: {which_result.stdout.strip()}")
    except:
        print("无法获取OpenROAD路径")

def main():
    """主函数"""
    print("🚀 OpenROAD 诊断开始...")
    
    # 检查系统要求
    check_system_requirements()
    
    # 测试OpenROAD基本功能
    basic_ok = test_openroad_basic()
    
    if basic_ok:
        # 测试简单脚本
        simple_ok = test_openroad_simple_script()
        
        if simple_ok:
            # 测试真实数据
            real_data_ok = test_openroad_with_real_data()
            
            if real_data_ok:
                # 测试布局功能
                placement_ok = test_openroad_placement()
                
                if placement_ok:
                    print("\n✅ 所有测试通过！OpenROAD功能正常")
                    return 0
                else:
                    print("\n❌ 布局功能测试失败")
                    return 1
            else:
                print("\n❌ 真实数据测试失败")
                return 1
        else:
            print("\n❌ 简单脚本测试失败")
            return 1
    else:
        print("\n❌ 基本功能测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 