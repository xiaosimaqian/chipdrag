#!/usr/bin/env python3
"""
测试只使用DEF文件进行布局，不读取Verilog网表
"""

import subprocess
from pathlib import Path

def test_def_only_layout(design_name="mgc_fft_2"):
    """测试只使用DEF文件的布局"""
    design_dir = Path(f"data/designs/ispd_2015_contest_benchmark/{design_name}")
    
    # 创建简化的TCL脚本
    tcl_content = f"""
# 简化的DEF布局测试
puts "=== 开始DEF布局测试 ==="

# 读取LEF文件
read_lef tech.lef
read_lef cells.lef
puts "LEF文件加载完成"

# 只读取DEF文件，不读取Verilog
read_def floorplan.def
puts "DEF文件加载完成"

# 获取设计信息
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]
set insts [$block getInsts]
set total_count [llength $insts]
puts "设计包含 $total_count 个实例"

# 简单的全局布局
puts "开始全局布局..."
if {{[catch {{global_placement -density 0.7}} result]}} {{
    puts "全局布局失败: $result"
    exit 1
}} else {{
    puts "全局布局成功"
}}

# 检查布局结果
set placed_count 0
foreach inst $insts {{
    if {{[$inst isPlaced]}} {{
        incr placed_count
    }}
}}
puts "布局完成: $placed_count/$total_count 实例已放置"

# 输出结果
write_def simple_layout_result.def
puts "=== 布局测试完成 ==="
exit
"""
    
    # 写入TCL文件
    tcl_file = design_dir / "test_def_only.tcl"
    with open(tcl_file, 'w') as f:
        f.write(tcl_content)
    
    print(f"测试设计: {design_name}")
    print(f"TCL脚本: {tcl_file}")
    
    # 运行Docker OpenROAD
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{design_dir.absolute()}:/workspace",
        "-w", "/workspace",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c",
        "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -no_init -exit test_def_only.tcl"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"返回码: {result.returncode}")
        print(f"标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print(f"标准错误:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ 执行超时")
        return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

if __name__ == "__main__":
    # 测试几个设计
    designs = ["mgc_fft_1", "mgc_fft_2", "mgc_fft_a"]
    
    for design in designs:
        print(f"\n{'='*50}")
        success = test_def_only_layout(design)
        if success:
            print(f"✅ {design} 测试成功")
        else:
            print(f"❌ {design} 测试失败") 