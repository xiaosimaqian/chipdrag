#!/usr/bin/env python3
"""
批量验证ISPD 2015基准设计的统计信息
"""

import os
import sys
from pathlib import Path
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_all_designs():
    """验证所有ISPD设计的统计信息"""
    benchmark_dir = project_root / "data/designs/ispd_2015_contest_benchmark"
    
    if not benchmark_dir.exists():
        logger.error(f"基准测试目录不存在: {benchmark_dir}")
        return
    
    # 获取所有设计目录
    design_dirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
    design_dirs.sort()
    
    logger.info(f"找到 {len(design_dirs)} 个设计目录")
    
    # 验证每个设计
    results = []
    for design_dir in design_dirs:
        design_name = design_dir.name
        logger.info(f"验证设计: {design_name}")
        
        try:
            # 创建接口实例
            interface = RealOpenROADInterface(str(design_dir))
            
            # 提取统计信息
            stats = interface._extract_design_stats()
            
            # 验证数据合理性
            validation = validate_stats(stats, design_name)
            
            results.append({
                'design': design_name,
                'stats': stats,
                'valid': validation['valid'],
                'issues': validation['issues']
            })
            
            logger.info(f"  ✓ {design_name}: {stats['num_instances']}实例, {stats['num_nets']}网络, {stats['num_pins']}引脚")
            
        except Exception as e:
            logger.error(f"  ✗ {design_name}: 验证失败 - {e}")
            results.append({
                'design': design_name,
                'stats': {},
                'valid': False,
                'issues': [f"验证失败: {e}"]
            })
    
    # 生成报告
    generate_report(results)
    
    return results

def validate_stats(stats, design_name):
    """验证统计信息的合理性"""
    issues = []
    
    # 检查必要字段
    required_fields = ['num_instances', 'num_nets', 'num_pins', 'core_area']
    for field in required_fields:
        if field not in stats:
            issues.append(f"缺少字段: {field}")
            continue
        
        value = stats[field]
        if not isinstance(value, (int, float)) or value < 0:
            issues.append(f"字段 {field} 值无效: {value}")
    
    # 检查数值合理性
    if 'num_instances' in stats and 'num_nets' in stats:
        instances = stats['num_instances']
        nets = stats['num_nets']
        
        # 网络数应该与实例数相近（通常网络数略大于实例数）
        if nets < instances * 0.5:
            issues.append(f"网络数({nets})远小于实例数({instances})")
        elif nets > instances * 2:
            issues.append(f"网络数({nets})远大于实例数({instances})")
    
    if 'num_pins' in stats:
        pins = stats['num_pins']
        if pins < 10:
            issues.append(f"引脚数过少: {pins}")
        elif pins > 100000:
            issues.append(f"引脚数过多: {pins}")
    
    if 'core_area' in stats:
        area = stats['core_area']
        # ISPD基准测试面积单位是纳米，调整验证范围
        # 最小面积：1mm² = 1,000,000 um²
        # 最大面积：1000cm² = 1,000,000,000 um² (对于超大型设计)
        if area < 1000000:  # 小于1mm²
            issues.append(f"核心面积过小: {area}")
        elif area > 1000000000000:  # 大于1000cm² (1,000,000,000 um²)
            issues.append(f"核心面积过大: {area}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }

def generate_report(results):
    """生成验证报告"""
    logger.info("\n" + "="*80)
    logger.info("ISPD 2015基准设计统计信息验证报告")
    logger.info("="*80)
    
    # 统计总体情况
    total_designs = len(results)
    valid_designs = sum(1 for r in results if r['valid'])
    failed_designs = total_designs - valid_designs
    
    logger.info(f"总设计数: {total_designs}")
    logger.info(f"验证通过: {valid_designs}")
    logger.info(f"验证失败: {failed_designs}")
    logger.info(f"通过率: {valid_designs/total_designs*100:.1f}%")
    
    # 详细结果
    logger.info("\n详细结果:")
    logger.info("-" * 80)
    
    for result in results:
        design = result['design']
        stats = result['stats']
        valid = result['valid']
        issues = result['issues']
        
        if valid:
            logger.info(f"✓ {design:20s} | {stats.get('num_instances', 0):8d}实例 | {stats.get('num_nets', 0):8d}网络 | {stats.get('num_pins', 0):6d}引脚 | {stats.get('core_area', 0):12d}面积")
        else:
            logger.info(f"✗ {design:20s} | 验证失败: {', '.join(issues)}")
    
    # 失败案例详情
    failed_results = [r for r in results if not r['valid']]
    if failed_results:
        logger.info("\n失败案例详情:")
        logger.info("-" * 80)
        for result in failed_results:
            logger.info(f"设计: {result['design']}")
            for issue in result['issues']:
                logger.info(f"  问题: {issue}")
            logger.info("")
    
    logger.info("="*80)

if __name__ == "__main__":
    verify_all_designs() 