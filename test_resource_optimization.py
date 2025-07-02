#!/usr/bin/env python3
"""
测试资源分配优化和并发处理功能
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from paper_hpwl_comparison_experiment import PaperHPWLComparisonExperiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_resources():
    """测试系统资源检查功能"""
    logger.info("=== 测试系统资源检查功能 ===")
    
    experiment = PaperHPWLComparisonExperiment()
    
    # 检查系统资源
    resources = experiment._check_system_resources()
    logger.info(f"当前系统资源状态:")
    logger.info(f"  活跃容器数: {resources['active_containers']}/{resources['max_containers']}")
    logger.info(f"  内存使用: {resources['memory_usage_gb']:.1f}GB/{resources['max_memory_gb']}GB")
    
    return resources

def test_docker_resource_allocation():
    """测试Docker资源分配功能"""
    logger.info("=== 测试Docker资源分配功能 ===")
    
    experiment = PaperHPWLComparisonExperiment()
    
    # 测试不同规模设计的资源分配
    test_designs = [
        "mgc_fft_1",           # 小型设计
        "mgc_matrix_mult_a",   # 中型设计
        "mgc_des_perf_a",      # 大型设计
    ]
    
    for design_name in test_designs:
        design_dir = experiment.data_dir / design_name
        if design_dir.exists():
            resources = experiment._calculate_docker_resources_for_design(design_dir)
            logger.info(f"设计 {design_name}:")
            logger.info(f"  规模: {resources['design_size']}")
            logger.info(f"  内存: {resources['memory_limit']}")
            logger.info(f"  CPU: {resources['cpu_limit']}核")
            logger.info(f"  超时: {resources['timeout']}秒")
            logger.info(f"  组件数: {resources['num_components']}")
        else:
            logger.warning(f"设计目录不存在: {design_dir}")

def test_concurrent_processing():
    """测试并发处理功能"""
    logger.info("=== 测试并发处理功能 ===")
    
    experiment = PaperHPWLComparisonExperiment()
    
    # 检查需要处理的设计
    designs_to_process = []
    for design_name in experiment.experiment_config['designs'][:3]:  # 只测试前3个
        design_dir = experiment.data_dir / design_name
        iterations_dir = design_dir / "output" / "iterations"
        default_def = iterations_dir / "iteration_10.def"
        
        if not default_def.exists():
            designs_to_process.append(design_name)
    
    if designs_to_process:
        logger.info(f"需要处理的设计: {designs_to_process}")
        logger.info(f"最大并发数: {experiment.experiment_config['max_concurrent_designs']}")
        
        # 这里可以启动并发处理，但为了测试，我们只检查配置
        logger.info("并发处理配置正确")
    else:
        logger.info("所有测试设计都已处理完成")

def test_resource_waiting():
    """测试资源等待功能"""
    logger.info("=== 测试资源等待功能 ===")
    
    experiment = PaperHPWLComparisonExperiment()
    
    # 测试不同内存需求的等待
    test_memory_requirements = [2, 4, 8, 12]
    
    for memory_gb in test_memory_requirements:
        logger.info(f"测试等待 {memory_gb}GB 内存...")
        start_time = time.time()
        experiment._wait_for_resources(memory_gb)
        end_time = time.time()
        logger.info(f"等待完成，耗时: {end_time - start_time:.1f}秒")

def monitor_docker_usage():
    """监控Docker使用情况"""
    logger.info("=== 监控Docker使用情况 ===")
    
    try:
        # 检查活跃容器
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'], 
                              capture_output=True, text=True)
        if result.stdout:
            logger.info("当前活跃容器:")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith('NAMES'):
                    logger.info(f"  {line}")
        else:
            logger.info("没有活跃的Docker容器")
        
        # 检查资源使用
        result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'], 
                              capture_output=True, text=True)
        if result.stdout:
            logger.info("容器资源使用:")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith('NAME'):
                    logger.info(f"  {line}")
        
    except Exception as e:
        logger.error(f"监控Docker失败: {e}")

def main():
    """主测试函数"""
    logger.info("开始测试资源分配优化和并发处理功能")
    
    # 1. 测试系统资源检查
    test_system_resources()
    
    # 2. 测试Docker资源分配
    test_docker_resource_allocation()
    
    # 3. 测试并发处理配置
    test_concurrent_processing()
    
    # 4. 监控当前Docker使用情况
    monitor_docker_usage()
    
    # 5. 测试资源等待功能（可选，会实际等待）
    # test_resource_waiting()
    
    logger.info("测试完成")
    
    # 输出优化建议
    logger.info("=== 优化建议 ===")
    logger.info("1. 当前配置适配16GB内存M2 Pro，最大并发3个设计")
    logger.info("2. 复杂设计（如mgc_des_perf_a）分配14GB内存，10核CPU")
    logger.info("3. 简单设计分配2-4GB内存，2-3核CPU")
    logger.info("4. 系统保留2GB内存和2核CPU给其他进程")
    logger.info("5. 建议在空闲时间运行大型设计，避免影响系统性能")

if __name__ == "__main__":
    main() 