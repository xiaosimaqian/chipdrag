#!/usr/bin/env python3
"""
简单的OpenROAD布局布线测试
直接使用网表进行布局布线，跳过综合阶段
"""

import subprocess
import logging
import shutil
import os
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_openroad_with_docker(work_dir: Path, cmd: str, is_tcl: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    """
    统一通过Docker调用OpenROAD
    :param work_dir: 挂载和工作目录
    :param cmd: TCL脚本文件名（只需文件名，不带路径），或直接openroad命令
    :param is_tcl: 是否为TCL脚本（True则自动拼接/workspace/xxx.tcl）
    :param timeout: 超时时间（秒）
    :return: subprocess.CompletedProcess对象
    """
    if is_tcl:
        cmd_in_container = f"/workspace/{cmd}"
        openroad_cmd = f"openroad {cmd_in_container}"
    else:
        openroad_cmd = f"openroad {cmd}"
    docker_cmd = [
        'docker', 'run', '--rm',
        '-v', f'{work_dir}:/workspace',
        '-w', '/workspace',
        'openroad/flow-ubuntu22.04-builder:21e414',
        'bash', '-c',
        f'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && {openroad_cmd}'
    ]
    logger.info(f"调用Docker OpenROAD: {openroad_cmd} @ {work_dir}")
    return subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)

def write_tcl_script(script_file: Path, content: str):
    """写入TCL脚本并确保文件同步到磁盘"""
    with open(script_file, 'w') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    logger.info(f"✅ TCL脚本已写入并同步: {script_file}")

def test_openroad_version():
    """测试OpenROAD版本"""
    logger.info("=== 测试1: OpenROAD版本 ===")
    try:
        result = run_openroad_with_docker(Path.cwd(), "-version", is_tcl=False, timeout=30)
        if result.returncode == 0:
            logger.info("✅ OpenROAD版本检查成功")
            logger.info(f"版本信息: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ OpenROAD版本检查失败: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ OpenROAD版本检查异常: {e}")
        return False

def test_real_placement():
    """测试真实布局"""
    logger.info("=== 测试2: 真实布局测试 ===")
    
    # 使用mgc_des_perf_1基准测试
    benchmark_dir = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1")
    
    if not benchmark_dir.exists():
        logger.error(f"❌ 基准测试目录不存在: {benchmark_dir}")
        return False
    
    # 创建测试目录
    test_dir = Path("test_real_placement")
    test_dir.mkdir(exist_ok=True)
    
    # 复制必要文件
    files_to_copy = ['floorplan.def', 'design.v', 'tech.lef', 'cells.lef']
    for file in files_to_copy:
        src = benchmark_dir / file
        dst = test_dir / file
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"✅ 复制 {file}")
        else:
            logger.error(f"❌ 源文件不存在: {file}")
            return False
    
    # 创建布局TCL脚本
    placement_script = """
# 真实布局测试脚本
puts "开始真实布局测试..."

# 读取设计文件
read_lef tech.lef
read_lef cells.lef
read_def floorplan.def
read_verilog design.v

# 链接设计
link_design des_perf
puts "✅ 设计加载完成"

# 获取设计信息
set design_name [current_design]
puts "设计名称: $design_name"

set cell_count [llength [get_cells]]
puts "单元数量: $cell_count"

set net_count [llength [get_nets]]
puts "网络数量: $net_count"

# 执行布局
puts "开始执行布局..."
global_placement -density 0.91 -init_density_penalty 0.01 -skip_initial_place
puts "✅ 全局布局完成"

detailed_placement
puts "✅ 详细布局完成"

# 检查布局结果
check_placement -verbose
puts "✅ 布局检查完成"

# 获取布局指标
set final_hpwl [get_placement_wirelength]
set final_overflow [get_placement_overflow]
puts "最终HPWL: $final_hpwl"
puts "最终Overflow: $final_overflow"

# 保存布局结果
write_def final_placement.def
puts "✅ 布局结果已保存到 final_placement.def"

# 生成报告
report_placement_wirelength
report_placement_overflow

puts "真实布局测试完成"
"""
    
    script_file = test_dir / "real_placement.tcl"
    write_tcl_script(script_file, placement_script)
    
    try:
        logger.info("开始执行真实布局...")
        result = run_openroad_with_docker(test_dir, "real_placement.tcl", timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ 真实布局执行成功")
            logger.info(f"输出: {result.stdout}")
            
            # 检查是否生成了布局文件
            def_file = test_dir / "final_placement.def"
            if def_file.exists():
                logger.info(f"✅ 布局文件已生成: {def_file}")
                return True
            else:
                logger.warning("⚠️  布局文件未生成")
                return False
        else:
            logger.error("❌ 真实布局执行失败")
            logger.error(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 真实布局执行异常: {e}")
        return False

def test_routing():
    """测试布线"""
    logger.info("=== 测试3: 布线测试 ===")
    
    test_dir = Path("test_real_placement")
    if not test_dir.exists():
        logger.error("❌ 布局测试目录不存在，请先运行布局测试")
        return False
    
    # 检查是否有布局结果
    def_file = test_dir / "final_placement.def"
    if not def_file.exists():
        logger.error("❌ 布局文件不存在，请先运行布局测试")
        return False
    
    # 创建布线TCL脚本
    routing_script = """
# 布线测试脚本
puts "开始布线测试..."

# 读取布局结果
read_lef tech.lef
read_lef cells.lef
read_def final_placement.def
read_verilog design.v
link_design des_perf

puts "✅ 布局结果加载完成"

# 执行布线
puts "开始执行布线..."
global_route
puts "✅ 全局布线完成"

detailed_route
puts "✅ 详细布线完成"

# 检查布线结果
check_antennas
puts "✅ 天线检查完成"

# 获取布线指标
set final_hpwl [get_placement_wirelength]
set final_overflow [get_placement_overflow]
puts "最终HPWL: $final_hpwl"
puts "最终Overflow: $final_overflow"

# 保存布线结果
write_def final_routed.def
puts "✅ 布线结果已保存到 final_routed.def"

# 生成报告
report_route
report_timing

puts "布线测试完成"
"""
    
    script_file = test_dir / "routing.tcl"
    write_tcl_script(script_file, routing_script)
    
    try:
        logger.info("开始执行布线...")
        result = run_openroad_with_docker(test_dir, "routing.tcl", timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ 布线执行成功")
            logger.info(f"输出: {result.stdout}")
            
            # 检查是否生成了布线文件
            routed_file = test_dir / "final_routed.def"
            if routed_file.exists():
                logger.info(f"✅ 布线文件已生成: {routed_file}")
                return True
            else:
                logger.warning("⚠️  布线文件未生成")
                return False
        else:
            logger.error("❌ 布线执行失败")
            logger.error(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 布线执行异常: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始OpenROAD真实布局布线测试...")
    
    tests = [
        ("OpenROAD版本", test_openroad_version),
        ("真实布局", test_real_placement),
        ("布线", test_routing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    logger.info("\n" + "="*60)
    logger.info("OpenROAD真实布局布线测试结果")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！OpenROAD布局布线功能正常")
    else:
        logger.warning("⚠️  部分测试失败，需要进一步调试")
    
    return passed == total

if __name__ == "__main__":
    main() 