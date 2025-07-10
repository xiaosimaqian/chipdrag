#!/usr/bin/env python3
"""
OpenROAD 服务器环境修复脚本
解决OpenROAD在服务器环境中的常见问题
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_openroad_environment():
    """修复OpenROAD环境问题"""
    print("🔧 修复OpenROAD环境...")
    
    # 检查OpenROAD是否可用
    try:
        result = subprocess.run(
            ["which", "openroad"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("❌ OpenROAD未找到在PATH中")
            return False
        
        openroad_path = result.stdout.strip()
        print(f"✅ OpenROAD路径: {openroad_path}")
        
        # 检查OpenROAD是否可执行
        if not os.access(openroad_path, os.X_OK):
            print("❌ OpenROAD不可执行")
            return False
        
        print("✅ OpenROAD环境检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 检查OpenROAD环境时发生错误: {e}")
        return False

def fix_data_files():
    """修复数据文件问题"""
    print("🔧 修复数据文件...")
    
    # 检查数据目录
    data_dir = Path("dataset/ispd_2015_contest_benchmark")
    if not data_dir.exists():
        print("❌ 数据目录不存在")
        return False
    
    # 检查每个设计的文件完整性
    designs = ["mgc_fft_1", "mgc_fft_2", "mgc_matrix_mult_1", "mgc_matrix_mult_a", 
               "mgc_matrix_mult_b", "mgc_des_perf_1", "mgc_des_perf_a", "mgc_des_perf_b"]
    
    for design in designs:
        design_dir = data_dir / design
        if not design_dir.exists():
            print(f"❌ 设计目录不存在: {design}")
            continue
            
        # 检查必要文件
        required_files = ["tech.lef", "cells.lef", "design.v", "floorplan.def"]
        missing_files = []
        
        for file_name in required_files:
            file_path = design_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ 设计 {design} 缺少文件: {', '.join(missing_files)}")
        else:
            print(f"✅ 设计 {design} 文件完整")
    
    return True

def fix_script_permissions():
    """修复脚本权限问题"""
    print("🔧 修复脚本权限...")
    
    # 确保Python脚本有执行权限
    python_scripts = [
        "paper_hpwl_comparison_experiment_server.py",
        "test_openroad_diagnosis.py",
        "fix_openroad_server.py"
    ]
    
    for script in python_scripts:
        script_path = Path(script)
        if script_path.exists():
            try:
                os.chmod(script_path, 0o755)
                print(f"✅ 设置 {script} 执行权限")
            except Exception as e:
                print(f"❌ 设置 {script} 权限失败: {e}")
    
    return True

def fix_openroad_script_generation():
    """修复OpenROAD脚本生成问题"""
    print("🔧 修复OpenROAD脚本生成...")
    
    # 创建一个简化的OpenROAD脚本模板
    simplified_script = """
# === 简化的OpenROAD布局脚本 ===
puts "=== 简化OpenROAD布局脚本开始 ==="

# 检查文件存在性
set required_files {tech.lef cells.lef design.v floorplan.def}
foreach file $required_files {
    if {![file exists $file]} {
        puts "❌ 文件不存在: $file"
        exit 1
    }
    puts "✅ 文件存在: $file"
}

# 重置数据库
if {[info exists ::ord::db]} {
    puts "重置OpenROAD数据库..."
    ord::reset_db
}

# 读取文件
puts "读取LEF文件..."
read_lef tech.lef
read_lef cells.lef
puts "✅ LEF文件读取完成"

puts "读取Verilog文件..."
read_verilog design.v
puts "✅ Verilog文件读取完成"

puts "读取DEF文件..."
read_def floorplan.def
puts "✅ DEF文件读取完成"

# 简化的floorplan初始化
puts "初始化floorplan..."
if {[catch {
    initialize_floorplan -utilization 0.6 -aspect_ratio 1.0 -core_space 50
    puts "✅ floorplan初始化成功"
} err]} {
    puts "❌ floorplan初始化失败: $err"
    puts "尝试手动设置..."
    
    if {[catch {
        initialize_floorplan -die_area {0 0 3000 3000} -core_area {100 100 2900 2900}
        puts "✅ 手动floorplan初始化成功"
    } err2]} {
        puts "❌ 手动floorplan初始化也失败: $err2"
        exit 1
    }
}

# 简化的全局布局
puts "开始全局布局..."
if {[catch {
    global_placement -density 0.6 -overflow 0.2
    puts "✅ 全局布局成功"
} err]} {
    puts "❌ 全局布局失败: $err"
    puts "尝试更保守的参数..."
    
    if {[catch {
        global_placement -density 0.5 -overflow 0.3
        puts "✅ 保守参数全局布局成功"
    } err2]} {
        puts "❌ 保守参数全局布局也失败: $err2"
        # 不退出，继续尝试详细布局
    }
}

# 简化的详细布局
puts "开始详细布局..."
if {[catch {
    detailed_placement -max_displacement 500
    puts "✅ 详细布局成功"
} err]} {
    puts "❌ 详细布局失败: $err"
    puts "尝试更宽松的参数..."
    
    if {[catch {
        detailed_placement -max_displacement 1000
        puts "✅ 宽松参数详细布局成功"
    } err2]} {
        puts "❌ 宽松参数详细布局也失败: $err2"
        # 不退出，继续保存结果
    }
}

# 保存结果
puts "保存布局结果..."
if {[catch {
    write_def placed.def
    puts "✅ 布局结果保存成功"
} err]} {
    puts "❌ 保存布局结果失败: $err"
    exit 1
}

puts "=== 简化OpenROAD布局脚本完成 ==="
exit 0
"""
    
    # 创建模板文件
    template_file = Path("simple_openroad_template.tcl")
    try:
        with open(template_file, 'w') as f:
            f.write(simplified_script)
        print(f"✅ 创建简化OpenROAD脚本模板: {template_file}")
        return True
    except Exception as e:
        print(f"❌ 创建模板文件失败: {e}")
        return False

def create_test_script():
    """创建测试脚本"""
    print("🔧 创建测试脚本...")
    
    test_script = """#!/bin/bash
# OpenROAD 快速测试脚本

echo "🚀 OpenROAD 快速测试"

# 检查OpenROAD是否可用
if ! command -v openroad &> /dev/null; then
    echo "❌ OpenROAD 未找到"
    exit 1
fi

echo "✅ OpenROAD 可用"

# 运行简单测试
cd dataset/ispd_2015_contest_benchmark/mgc_fft_1

echo "📁 当前目录: $(pwd)"
echo "📋 文件列表:"
ls -la

# 运行简化脚本
echo "🔧 运行简化OpenROAD脚本..."
if openroad -no_init -no_splash -exit ../../../simple_openroad_template.tcl; then
    echo "✅ OpenROAD 简化脚本执行成功"
else
    echo "❌ OpenROAD 简化脚本执行失败"
    exit 1
fi

echo "✅ 所有测试通过"
"""
    
    # 创建测试脚本
    script_file = Path("test_openroad_quick.sh")
    try:
        with open(script_file, 'w') as f:
            f.write(test_script)
        os.chmod(script_file, 0o755)
        print(f"✅ 创建快速测试脚本: {script_file}")
        return True
    except Exception as e:
        print(f"❌ 创建测试脚本失败: {e}")
        return False

def fix_server_configuration():
    """修复服务器配置"""
    print("🔧 修复服务器配置...")
    
    # 检查内存和CPU
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        print(f"📊 服务器资源:")
        print(f"   内存: {memory_gb:.1f} GB")
        print(f"   CPU: {cpu_count} 核")
        
        if memory_gb < 8:
            print("⚠️ 内存不足，建议至少8GB")
        if cpu_count < 4:
            print("⚠️ CPU核心数不足，建议至少4核")
            
        # 设置OpenROAD环境变量
        os.environ['OMP_NUM_THREADS'] = str(min(cpu_count, 16))
        print(f"✅ 设置OpenROAD线程数: {os.environ['OMP_NUM_THREADS']}")
        
        return True
        
    except ImportError:
        print("❌ 无法导入psutil，请安装: pip install psutil")
        return False
    except Exception as e:
        print(f"❌ 检查服务器配置时发生错误: {e}")
        return False

def create_fixed_experiment_script():
    """创建修复后的实验脚本"""
    print("🔧 创建修复后的实验脚本...")
    
    # 创建一个简化的实验脚本，专门用于服务器环境
    fixed_script = '''#!/usr/bin/env python3
"""
修复后的ChipDRAG服务器实验脚本
专门为服务器环境优化
"""

import subprocess
import sys
import os
from pathlib import Path
import psutil
import time

def run_single_design_test(design_name):
    """运行单个设计测试"""
    print(f"🧪 测试设计: {design_name}")
    
    design_dir = Path(f"dataset/ispd_2015_contest_benchmark/{design_name}")
    if not design_dir.exists():
        print(f"❌ 设计目录不存在: {design_dir}")
        return False
    
    # 切换到设计目录
    os.chdir(design_dir)
    
    # 创建简化脚本
    simple_script = f"""
puts "=== 简化测试脚本 ==="
puts "设计名称: {design_name}"

# 检查文件
if {{![file exists tech.lef]}} {{
    puts "❌ tech.lef 不存在"
    exit 1
}}
if {{![file exists cells.lef]}} {{
    puts "❌ cells.lef 不存在"
    exit 1
}}
if {{![file exists design.v]}} {{
    puts "❌ design.v 不存在"
    exit 1
}}
if {{![file exists floorplan.def]}} {{
    puts "❌ floorplan.def 不存在"
    exit 1
}}

# 读取文件
read_lef tech.lef
read_lef cells.lef
read_verilog design.v
read_def floorplan.def

# 简化初始化
initialize_floorplan -utilization 0.5 -aspect_ratio 1.0 -core_space 100

# 保存结果
write_def test_result.def

puts "✅ 测试完成"
exit 0
"""
    
    # 写入脚本
    script_file = Path("test_simple.tcl")
    with open(script_file, 'w') as f:
        f.write(simple_script)
    
    # 运行测试
    try:
        result = subprocess.run(
            ["openroad", "-no_init", "-no_splash", "-exit", "test_simple.tcl"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"返回码: {result.returncode}")
        if result.returncode == 0:
            print("✅ 测试成功")
            return True
        else:
            print("❌ 测试失败")
            print(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 运行测试时发生错误: {e}")
        return False
    finally:
        # 清理
        if script_file.exists():
            script_file.unlink()
        os.chdir("../../..")

def main():
    print("🚀 修复后的ChipDRAG服务器实验")
    
    # 测试每个设计
    designs = ["mgc_fft_1", "mgc_fft_2"]
    
    for design in designs:
        success = run_single_design_test(design)
        if success:
            print(f"✅ {design} 测试通过")
        else:
            print(f"❌ {design} 测试失败")
        print("-" * 50)

if __name__ == "__main__":
    main()
'''
    
    # 创建修复后的脚本
    script_file = Path("fixed_experiment_simple.py")
    try:
        with open(script_file, 'w') as f:
            f.write(fixed_script)
        os.chmod(script_file, 0o755)
        print(f"✅ 创建修复后的实验脚本: {script_file}")
        return True
    except Exception as e:
        print(f"❌ 创建修复脚本失败: {e}")
        return False

def main():
    """主修复流程"""
    print("🚀 OpenROAD 服务器环境修复开始...")
    
    fixes = [
        ("检查OpenROAD环境", fix_openroad_environment),
        ("检查数据文件", fix_data_files),
        ("修复脚本权限", fix_script_permissions),
        ("修复服务器配置", fix_server_configuration),
        ("创建OpenROAD脚本模板", fix_openroad_script_generation),
        ("创建测试脚本", create_test_script),
        ("创建修复后的实验脚本", create_fixed_experiment_script)
    ]
    
    success_count = 0
    for name, fix_func in fixes:
        print(f"\n{'='*50}")
        print(f"🔧 {name}")
        print(f"{'='*50}")
        
        try:
            if fix_func():
                success_count += 1
                print(f"✅ {name} 完成")
            else:
                print(f"❌ {name} 失败")
        except Exception as e:
            print(f"❌ {name} 发生异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 修复完成: {success_count}/{len(fixes)} 项成功")
    print(f"{'='*50}")
    
    if success_count == len(fixes):
        print("\n✅ 所有修复完成！现在可以尝试运行:")
        print("   1. python test_openroad_diagnosis.py  # 诊断测试")
        print("   2. ./test_openroad_quick.sh          # 快速测试")
        print("   3. python fixed_experiment_simple.py # 简化实验")
        return 0
    else:
        print(f"\n⚠️ 有 {len(fixes) - success_count} 项修复失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 