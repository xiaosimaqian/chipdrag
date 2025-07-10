#!/usr/bin/env python3
"""
详细的OpenROAD调试脚本
"""

import subprocess
from pathlib import Path

def debug_openroad_detailed():
    """详细调试OpenROAD执行错误"""
    design_dir = Path("dataset/ispd_2015_contest_benchmark/mgc_des_perf_b")
    
    if not design_dir.exists():
        print(f"❌ 测试设计不存在: {design_dir}")
        return
    
    # 创建逐步调试脚本
    debug_script = """
puts "=== 详细OpenROAD调试 ==="
puts "当前工作目录: [pwd]"
puts "文件列表: [glob *]"

# 检查文件存在性
foreach file {tech.lef cells.lef design.v floorplan.def} {
    if {[file exists $file]} {
        set size [file size $file]
        puts "✅ $file 存在 (${size} bytes)"
    } else {
        puts "❌ $file 不存在"
    }
}

# 第1步：加载tech.lef
puts "\\n=== 第1步：加载tech.lef ==="
if {[catch {
    read_lef tech.lef
    puts "✅ tech.lef 加载成功"
} err]} {
    puts "❌ tech.lef 加载失败: $err"
    exit 1
}

# 第2步：加载cells.lef
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

# 第4步：自动检测设计名称
puts "\\n=== 第4步：检测设计名称 ==="
set design_name "unknown"
if {[catch {
    set def_content [read [open floorplan.def r]]
    regexp {DESIGN\\s+(\\w+)} $def_content match design_name
    puts "从DEF文件检测到设计名称: $design_name"
} err]} {
    puts "警告：无法从DEF文件检测设计名称: $err"
    set design_name "des_perf"
}

# 第5步：尝试链接设计
puts "\\n=== 第5步：链接设计 ==="
set link_success 0
foreach name [list $design_name "des_perf" "mgc_des_perf_b" "top" "design"] {
    puts "尝试链接设计名称: $name"
    if {![catch {link_design $name}]} {
        puts "✅ 设计链接成功: $name"
        set design_name $name
        set link_success 1
        break
    } else {
        puts "❌ 链接失败: $name"
    }
}

if {!$link_success} {
    puts "❌ 所有设计名称都无法链接"
    exit 1
}

# 第6步：检查设计状态
puts "\\n=== 第6步：检查设计状态 ==="
puts "当前设计: [current_design]"
puts "单元数量: [llength [get_cells -quiet]]"
puts "网络数量: [llength [get_nets -quiet]]"

# 第7步：尝试初始化floorplan
puts "\\n=== 第7步：初始化floorplan ==="
if {[catch {
    initialize_floorplan -utilization 0.6 -aspect_ratio 1.0 -core_space 20 -site core
    puts "✅ floorplan 初始化成功"
} err]} {
    puts "❌ floorplan 初始化失败: $err"
    
    # 尝试其他site名称
    set fallback_sites {CoreSite unit CORE}
    set floorplan_success 0
    
    foreach site $fallback_sites {
        puts "尝试使用site: $site"
        if {![catch {
            initialize_floorplan -utilization 0.6 -aspect_ratio 1.0 -core_space 20 -site $site
        }]} {
            puts "✅ 使用site $site 初始化成功"
            set floorplan_success 1
            break
        }
    }
    
    if {!$floorplan_success} {
        puts "❌ 所有site都失败，尝试手动指定区域"
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

# 第8步：尝试全局布局
puts "\\n=== 第8步：全局布局 ==="
if {[catch {
    global_placement -density 0.5 -overflow 0.2
    puts "✅ 全局布局成功"
} err]} {
    puts "❌ 全局布局失败: $err"
    exit 1
}

puts "\\n🎉 所有步骤都成功完成！"
"""
    
    # 写入调试脚本
    debug_script_file = design_dir / "debug_detailed.tcl"
    with open(debug_script_file, 'w') as f:
        f.write(debug_script)
    
    print(f"调试脚本已写入: {debug_script_file}")
    
    # 执行Docker命令
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{design_dir.absolute()}:/work",
        "-w", "/work",
        "--memory", "3g",
        "--cpus", "2",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c",
        f"export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -no_init -no_splash -exit debug_detailed.tcl"
    ]
    
    print("执行详细OpenROAD调试...")
    print(f"设计目录: {design_dir}")
    
    try:
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
            print("\n✅ 调试完成，所有步骤成功!")
        else:
            print(f"\n❌ 在第{result.returncode}步失败")
            
    except subprocess.TimeoutExpired:
        print("❌ 执行超时")
    except Exception as e:
        print(f"❌ 执行异常: {e}")
    finally:
        # 清理调试脚本
        if debug_script_file.exists():
            debug_script_file.unlink()
            print(f"已清理调试脚本: {debug_script_file}")

if __name__ == "__main__":
    debug_openroad_detailed() 