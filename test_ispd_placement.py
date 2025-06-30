#!/usr/bin/env python3
"""
测试ISPD设计的布局过程，使用统一的Docker接口
"""

import subprocess
import logging
import os
import re
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_openroad_with_docker(work_dir: Path, cmd: str, is_tcl: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    """
    统一通过Docker调用OpenROAD
    """
    if is_tcl:
        cmd_in_container = f"/workspace/{cmd}"
        openroad_cmd = f"openroad {cmd_in_container}"
    else:
        openroad_cmd = f"openroad {cmd}"
    
    docker_cmd_str = f'docker run --rm -v {work_dir.absolute()}:/workspace -w /workspace openroad/flow-ubuntu22.04-builder:21e414 bash -c "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\\$PATH && {openroad_cmd}"'
    
    logger.info(f"调用Docker OpenROAD: {openroad_cmd} @ {work_dir}")
    
    return subprocess.run(docker_cmd_str, shell=True, capture_output=True, text=True, timeout=timeout)

def check_def_placement(def_file: Path) -> dict:
    """检查DEF文件是否包含PLACEMENT段"""
    result = {
        "exists": False,
        "has_placement": False,
        "has_placement_content": False,
        "lines": 0,
        "size": 0
    }
    
    if not def_file.exists():
        return result
    
    result["exists"] = True
    result["size"] = def_file.stat().st_size
    
    try:
        with open(def_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        result["lines"] = len(lines)
        
        # 检查PLACEMENT段
        has_placement = any("PLACEMENT" in line for line in lines)
        result["has_placement"] = has_placement
        
        # 检查PLACEMENT段是否有实际内容
        if has_placement:
            placement_start = -1
            placement_end = -1
            for i, line in enumerate(lines):
                if "PLACEMENT" in line:
                    placement_start = i
                elif placement_start != -1 and ";" in line and "PLACEMENT" not in line:
                    placement_end = i
                    break
            
            if placement_start != -1 and placement_end != -1:
                placement_lines = lines[placement_start:placement_end+1]
                # 检查是否有实际的placement语句
                has_placement_content = any(
                    re.match(r'\s*\w+\s+\w+\s+\+?\s*PLACED\s+\(\s*\d+\s+\d+\s*\)', line)
                    for line in placement_lines
                )
                result["has_placement_content"] = has_placement_content
        
    except Exception as e:
        logger.error(f"读取DEF文件失败: {e}")
    
    return result

def test_ispd_placement(design_name: str = "mgc_fft_1"):
    """测试ISPD设计的布局过程"""
    logger.info(f"=== 测试ISPD设计布局: {design_name} ===")
    
    # 设计目录
    design_dir = Path(f"data/designs/ispd_2015_contest_benchmark/{design_name}")
    if not design_dir.exists():
        logger.error(f"设计目录不存在: {design_dir}")
        return False
    
    # 检查必要文件
    required_files = ["design.v", "floorplan.def", "tech.lef", "cells.lef"]
    missing_files = []
    for file_name in required_files:
        if not (design_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"缺少必要文件: {missing_files}")
        return False
    
    logger.info(f"设计目录: {design_dir}")
    logger.info("必要文件检查通过")
    
    # 生成简化的TCL脚本进行测试
    tcl_script = f"""
# 简化的OpenROAD布局测试脚本
puts "开始布局测试: {design_name}"

# 读取LEF文件
read_lef tech.lef
read_lef cells.lef
puts "LEF文件读取完成"

# 读取Verilog文件
read_verilog design.v
puts "Verilog文件读取完成"

# 连接设计
link_design fft
puts "设计连接完成"

# 读取DEF文件
read_def floorplan.def
puts "DEF文件读取完成"

# 设置布局参数
set_placement_padding -global -left 2 -right 2
puts "布局参数设置完成"

# 全局布局
puts "开始全局布局..."
global_placement -density 0.75
puts "全局布局完成"

# 详细布局
puts "开始详细布局..."
detailed_placement
puts "详细布局完成"

# 检查布局结果
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]
set insts [$block getInsts]
set placed_count 0
set total_count 0

foreach inst $insts {{
    if {{[$inst isPlaced]}} {{
        incr placed_count
    }}
    incr total_count
}}

puts "布局结果: $placed_count/$total_count 实例已放置"

# 输出结果
write_def placement_result.def
puts "布局DEF文件已输出: placement_result.def"

# 检查PLACEMENT段
set def_content [read [open placement_result.def r]]
if {{[string first "PLACEMENT" $def_content] != -1}} {{
    puts "✅ PLACEMENT段存在"
}} else {{
    puts "❌ PLACEMENT段不存在"
}}

puts "布局测试完成"
"""
    
    # 写入TCL脚本
    tcl_file = design_dir / "test_placement.tcl"
    with open(tcl_file, 'w') as f:
        f.write(tcl_script)
    
    logger.info(f"TCL脚本已写入: {tcl_file}")
    
    try:
        # 运行OpenROAD
        logger.info("开始执行OpenROAD布局...")
        result = run_openroad_with_docker(design_dir, "test_placement.tcl", timeout=1800)
        
        # 检查执行结果
        if result.returncode == 0:
            logger.info("✅ OpenROAD执行成功")
            logger.info(f"输出长度: {len(result.stdout)} 字符")
            
            # 显示关键输出信息
            logger.info("=== OpenROAD输出关键信息 ===")
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ["布局", "PLACEMENT", "实例已放置", "全局布局", "完成"]):
                    logger.info(f"  {line}")
            
            # 检查关键输出
            if "布局结果:" in result.stdout:
                logger.info("✅ 布局过程完成")
            if "PLACEMENT段存在" in result.stdout:
                logger.info("✅ PLACEMENT段已生成")
            elif "PLACEMENT段不存在" in result.stdout:
                logger.warning("⚠️ PLACEMENT段未生成")
            
            # 检查生成的DEF文件
            def_file = design_dir / "placement_result.def"
            def_check = check_def_placement(def_file)
            
            logger.info(f"DEF文件检查结果:")
            logger.info(f"  文件存在: {def_check['exists']}")
            logger.info(f"  文件大小: {def_check['size']:,} bytes")
            logger.info(f"  行数: {def_check['lines']:,}")
            logger.info(f"  有PLACEMENT段: {def_check['has_placement']}")
            logger.info(f"  有PLACEMENT内容: {def_check['has_placement_content']}")
            
            return def_check['has_placement_content']
        else:
            logger.error("❌ OpenROAD执行失败")
            logger.error(f"返回码: {result.returncode}")
            logger.error(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ OpenROAD执行超时")
        return False
    except Exception as e:
        logger.error(f"❌ OpenROAD执行异常: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始ISPD布局测试...")
    
    # 测试一个较小的设计
    success = test_ispd_placement("mgc_fft_1")
    
    if success:
        logger.info("🎉 ISPD布局测试成功！DEF文件包含PLACEMENT段")
    else:
        logger.warning("⚠️ ISPD布局测试失败或PLACEMENT段缺失")
    
    return success

if __name__ == "__main__":
    main() 