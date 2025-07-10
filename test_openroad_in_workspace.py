#!/usr/bin/env python3
"""
在工作目录中测试OpenROAD，保持dataset目录只读
"""

import subprocess
import shutil
from pathlib import Path

def test_openroad_in_workspace():
    """在工作目录中测试OpenROAD，不修改dataset目录"""
    
    # 创建工作目录
    work_dir = Path("temp_openroad_test")
    work_dir.mkdir(exist_ok=True)
    
    # 源数据目录
    source_dir = Path("dataset/ispd_2015_contest_benchmark/mgc_des_perf_b")
    
    if not source_dir.exists():
        print(f"❌ 源数据目录不存在: {source_dir}")
        return
    
    print(f"创建工作目录: {work_dir}")
    print(f"源数据目录: {source_dir}")
    
    try:
        # 复制必要的数据文件到工作目录（只读模式）
        required_files = ["tech.lef", "cells.lef", "design.v", "floorplan.def"]
        for file_name in required_files:
            source_file = source_dir / file_name
            if source_file.exists():
                dest_file = work_dir / file_name
                shutil.copy2(source_file, dest_file)
                print(f"✅ 复制文件: {file_name}")
            else:
                print(f"❌ 缺少文件: {file_name}")
                return
        
        # 创建修复后的OpenROAD脚本
        fixed_script = """
puts "=== 修复后的OpenROAD测试脚本 ==="
puts "工作目录: [pwd]"

# 第1步：加载tech.lef（技术层定义）
puts "\\n=== 第1步：加载tech.lef ==="
if {[catch {
    read_lef tech.lef
    puts "✅ tech.lef 加载成功"
} err]} {
    puts "❌ tech.lef 加载失败: $err"
    exit 1
}

# 第2步：加载cells.lef（单元库定义）
puts "\\n=== 第2步：加载cells.lef ==="
if {[catch {
    read_lef cells.lef
    puts "✅ cells.lef 加载成功"
} err]} {
    puts "❌ cells.lef 加载失败: $err"
    exit 1
}

# 第3步：加载Verilog
puts "\\n=== 第3步：加载Verilog ==="
if {[catch {
    read_verilog design.v
    puts "✅ design.v 加载成功"
} err]} {
    puts "❌ design.v 加载失败: $err"
    exit 1
}

# 第4步：智能设计名称检测和链接
puts "\\n=== 第4步：检测并链接设计 ==="
set design_name "unknown"
if {[catch {
    set def_content [read [open floorplan.def r]]
    regexp {DESIGN\\s+(\\w+)} $def_content match design_name
    puts "从DEF文件检测到设计名称: $design_name"
} err]} {
    puts "警告：无法从DEF文件检测设计名称，使用备选方案"
    set design_name "des_perf"
}

# 尝试链接设计
set link_success 0
foreach name [list $design_name "des_perf" "mgc_des_perf_b" "top" "design"] {
    if {![catch {link_design $name}]} {
        puts "✅ 设计链接成功: $name"
        set design_name $name
        set link_success 1
        break
    }
}

if {!$link_success} {
    puts "❌ 所有设计名称都无法链接"
    exit 1
}

# 第5步：初始化floorplan
puts "\\n=== 第5步：初始化floorplan ==="
if {[catch {
    initialize_floorplan -utilization 0.6 -aspect_ratio 1.0 -core_space 20 -site core
    puts "✅ floorplan 初始化成功"
} err]} {
    puts "❌ floorplan 初始化失败: $err"
    puts "尝试备选方案..."
    
    # 尝试其他site类型
    set fallback_sites {CoreSite unit CORE}
    set floorplan_success 0
    
    foreach site $fallback_sites {
        if {![catch {
            initialize_floorplan -utilization 0.6 -aspect_ratio 1.0 -core_space 20 -site $site
        }]} {
            puts "✅ 使用site $site 初始化成功"
            set floorplan_success 1
            break
        }
    }
    
    if {!$floorplan_success} {
        if {[catch {
            initialize_floorplan -die_area {0 0 1000 1000} -core_area {100 100 900 900} -site core
        } err2]} {
            puts "❌ 手动指定区域也失败: $err2"
            exit 1
        } else {
            puts "✅ 手动指定区域成功"
        }
    }
}

# 第6步：尝试简单的全局布局
puts "\\n=== 第6步：全局布局 ==="
if {[catch {
    global_placement -density 0.5 -overflow 0.2
    puts "✅ 全局布局成功"
} err]} {
    puts "❌ 全局布局失败: $err"
    exit 1
}

# 第7步：输出结果
puts "\\n=== 第7步：输出结果 ==="
write_def placed.def
puts "✅ 布局结果已保存到 placed.def"

puts "\\n🎉 所有步骤都成功完成！修复验证通过！"
"""
        
        # 在工作目录中写入脚本
        script_file = work_dir / "test_fixed_openroad.tcl"
        with open(script_file, 'w') as f:
            f.write(fixed_script)
        
        print(f"\n脚本已写入: {script_file}")
        
        # 执行Docker命令
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{work_dir.absolute()}:/work",
            "-w", "/work",
            "--memory", "3g",
            "--cpus", "2",
            "openroad/flow-ubuntu22.04-builder:21e414",
            "bash", "-c",
            f"export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -no_init -no_splash -exit test_fixed_openroad.tcl"
        ]
        
        print("\n执行修复后的OpenROAD测试...")
        
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\n=== 执行结果 ===")
        print(f"返回码: {result.returncode}")
        print(f"\n=== 标准输出 ===")
        print(result.stdout)
        
        if result.stderr:
            print(f"\n=== 标准错误 ===")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ 修复成功！OpenROAD可以正常运行")
            # 检查生成的文件
            placed_def = work_dir / "placed.def"
            if placed_def.exists():
                print(f"✅ 生成了布局文件: {placed_def} ({placed_def.stat().st_size} bytes)")
        else:
            print(f"\n❌ 仍有问题，返回码: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("❌ 执行超时")
    except Exception as e:
        print(f"❌ 执行异常: {e}")
    finally:
        # 清理工作目录
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print(f"\n已清理工作目录: {work_dir}")

if __name__ == "__main__":
    test_openroad_in_workspace() 