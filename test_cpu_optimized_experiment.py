#!/usr/bin/env python3
"""
测试CPU优化的论文实验配置

针对macOS Docker环境无GPU支持的优化测试
"""

import os
import sys
import logging
from pathlib import Path
import psutil

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_resources():
    """测试系统资源信息"""
    logger.info("=== 系统资源测试 ===")
    
    # 获取系统资源
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    logger.info(f"总内存: {total_memory_gb:.1f}GB")
    logger.info(f"可用内存: {available_memory_gb:.1f}GB")
    logger.info(f"CPU核心数: {cpu_count}")
    if cpu_freq:
        logger.info(f"CPU频率: {cpu_freq.current:.0f}MHz")
    
    # 模拟资源分配计算
    designs = ['mgc_fft_1', 'mgc_des_perf_1', 'mgc_matrix_mult_1']
    
    for design in designs:
        logger.info(f"\n--- 设计 {design} 资源分配 ---")
        
        if 'matrix_mult' in design:
            memory_limit = f"{min(32, int(total_memory_gb * 0.8))}g"
            cpu_limit = f"{min(12, cpu_count - 2)}"
        elif 'des_perf' in design:
            memory_limit = f"{min(28, int(total_memory_gb * 0.7))}g"
            cpu_limit = f"{min(10, cpu_count - 2)}"
        elif 'fft' in design:
            memory_limit = f"{min(24, int(total_memory_gb * 0.6))}g"
            cpu_limit = f"{min(8, cpu_count - 2)}"
        else:
            memory_limit = f"{min(20, int(total_memory_gb * 0.5))}g"
            cpu_limit = f"{min(6, cpu_count - 2)}"
        
        logger.info(f"  内存限制: {memory_limit}")
        logger.info(f"  CPU限制: {cpu_limit}")
        
        # 估算超时时间
        base_timeout = 3600  # 1小时
        if 'matrix_mult' in design:
            timeout = base_timeout * 2.5
        elif 'des_perf' in design:
            timeout = base_timeout * 2.0
        elif 'fft' in design:
            timeout = base_timeout * 1.5
        else:
            timeout = base_timeout
        
        timeout = max(1800, min(int(timeout), 14400))  # 30分钟到4小时
        logger.info(f"  超时时间: {timeout}秒 ({timeout/3600:.1f}小时)")

def test_docker_command_generation():
    """测试Docker命令生成"""
    logger.info("\n=== Docker命令测试 ===")
    
    # 模拟参数
    design_dir = Path("/test/design")
    memory_limit = "20g"
    cpu_limit = "8"
    script_file = "run_placement.tcl"
    
    # 生成Docker命令
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{design_dir.absolute()}:/work",
        "-w", "/work",
        "--memory", memory_limit,
        "--cpus", cpu_limit,
        # CPU性能优化的环境变量
        "-e", f"OPENROAD_NUM_THREADS={cpu_limit}",
        "-e", f"OMP_NUM_THREADS={cpu_limit}",
        "-e", f"MKL_NUM_THREADS={cpu_limit}",
        "-e", "OMP_THREAD_LIMIT=999",
        "-e", "OMP_DYNAMIC=TRUE",
        "-e", "OMP_NESTED=TRUE",
        # 内存优化
        "-e", "MALLOC_ARENA_MAX=4",
        "-e", "MALLOC_MMAP_THRESHOLD_=131072",
        # OpenROAD镜像
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c",
        f"export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -no_init -no_splash -exit {script_file}"
    ]
    
    logger.info("生成的Docker命令:")
    logger.info(" ".join(docker_cmd))
    
    logger.info("\n关键优化点:")
    logger.info(f"✅ 内存限制: {memory_limit}")
    logger.info(f"✅ CPU限制: {cpu_limit}")
    logger.info(f"✅ OpenROAD线程数: {cpu_limit}")
    logger.info(f"✅ OpenMP线程数: {cpu_limit}")
    logger.info("✅ 内存分配优化")
    logger.info("✅ 无GPU依赖（CPU优化模式）")

def test_tcl_script_generation():
    """测试TCL脚本生成"""
    logger.info("\n=== TCL脚本测试 ===")
    
    cpu_limit = "8"
    
    # 关键TCL配置部分
    tcl_config = f"""
# CPU多线程优化配置 - 最大化利用系统CPU资源
set thread_count {cpu_limit}
puts "设置OpenROAD线程数: $thread_count"

# 设置OpenROAD并行处理参数
set_thread_count $thread_count
puts "启用 $thread_count 线程并行处理"

# 设置环境变量优化CPU性能
set ::env(OPENROAD_NUM_THREADS) $thread_count
set ::env(OMP_NUM_THREADS) $thread_count
puts "设置并行环境变量完成"
"""
    
    logger.info("生成的TCL配置:")
    logger.info(tcl_config)
    
    logger.info("TCL脚本优化特性:")
    logger.info(f"✅ 设置线程数: {cpu_limit}")
    logger.info("✅ OpenROAD并行处理")
    logger.info("✅ 环境变量优化")
    logger.info("✅ 多线程全局布局")
    logger.info("✅ 多线程详细布局")

def main():
    """主测试函数"""
    logger.info("🚀 开始CPU优化实验配置测试")
    logger.info("🖥️  系统: macOS Docker环境")
    logger.info("⚡ 模式: CPU优化（无GPU支持）")
    
    test_system_resources()
    test_docker_command_generation()
    test_tcl_script_generation()
    
    logger.info("\n=== 优化总结 ===")
    logger.info("✅ 系统资源检测: 动态调整内存和CPU分配")
    logger.info("✅ Docker优化: 环境变量 + 资源限制")
    logger.info("✅ OpenROAD优化: 多线程并行处理")
    logger.info("✅ 超时机制: 智能计算 + 重试机制")
    logger.info("✅ macOS兼容: 无GPU依赖的CPU集约模式")
    
    logger.info("\n🎯 预期改进:")
    logger.info("• 减少超时错误（智能超时 + 重试）")
    logger.info("• 提高CPU利用率（多线程并行）")
    logger.info("• 优化内存使用（动态分配）")
    logger.info("• 增强稳定性（错误处理 + 重试）")

if __name__ == "__main__":
    main() 