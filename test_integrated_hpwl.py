#!/usr/bin/env python3
"""
集成HPWL测试脚本

测试所有HPWL提取方法：
1. OpenROAD内置HPWL计算（最准确）
2. ISPD2005风格HPWL提取（基于官方脚本）
3. 原始HPWL提取（备选方案）
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiment import UnifiedPaperExperiment

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_sample_def_file():
    """创建一个示例DEF文件用于测试"""
    sample_def_content = """VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN sample_design ;
UNITS DISTANCE MICRONS 1000 ;

DIEAREA ( 0 0 ) ( 1000 1000 ) ;

COMPONENTS 3 ;
- comp1 comp1_cell + PLACED ( 100 100 ) N ;
- comp2 comp2_cell + PLACED ( 300 200 ) N ;
- comp3 comp3_cell + PLACED ( 500 300 ) N ;
END COMPONENTS

NETS 2 ;
- net1 ( comp1 ) ( comp2 ) ;
- net2 ( comp2 ) ( comp3 ) ;
END NETS

END DESIGN
"""
    
    sample_def_file = Path("sample_test.def")
    with open(sample_def_file, 'w') as f:
        f.write(sample_def_content)
    
    print(f"创建示例DEF文件: {sample_def_file}")
    return sample_def_file

def test_hpwl_methods_on_file(experiment, def_file: Path, design_dir: Path = None) -> Dict[str, Optional[float]]:
    """在单个DEF文件上测试所有HPWL提取方法"""
    logger = setup_logging()
    
    if design_dir is None:
        design_dir = def_file.parent
    
    logger.info(f"\n=== 测试DEF文件: {def_file} ===")
    
    results = {}
    
    # 1. 检查布局成功状态
    placement_success = experiment._check_placement_success(def_file)
    logger.info(f"布局成功状态: {placement_success}")
    
    # 2. 测试OpenROAD内置HPWL提取
    logger.info("--- OpenROAD内置HPWL提取方法 ---")
    try:
        hpwl_openroad = experiment._extract_hpwl_from_openroad_report(design_dir)
        results['openroad_builtin'] = hpwl_openroad
        if hpwl_openroad is not None:
            logger.info(f"✅ OpenROAD内置方法HPWL: {hpwl_openroad:.0f}")
        else:
            logger.warning("❌ OpenROAD内置方法HPWL提取失败")
    except Exception as e:
        logger.error(f"❌ OpenROAD内置方法异常: {e}")
        results['openroad_builtin'] = None
    
    # 3. 测试ISPD2005风格HPWL提取
    logger.info("--- ISPD2005风格HPWL提取方法 ---")
    try:
        hpwl_ispd2005 = experiment._extract_hpwl_from_def_ispd2005_style(def_file)
        results['ispd2005_style'] = hpwl_ispd2005
        if hpwl_ispd2005 is not None:
            logger.info(f"✅ ISPD2005方法HPWL: {hpwl_ispd2005:.0f}")
        else:
            logger.warning("❌ ISPD2005方法HPWL提取失败")
    except Exception as e:
        logger.error(f"❌ ISPD2005方法异常: {e}")
        results['ispd2005_style'] = None
    
    # 4. 测试原始HPWL提取
    logger.info("--- 原始HPWL提取方法 ---")
    try:
        hpwl_original = experiment._extract_hpwl_from_def(def_file)
        results['original'] = hpwl_original
        if hpwl_original is not None:
            logger.info(f"✅ 原始方法HPWL: {hpwl_original:.0f}")
        else:
            logger.warning("❌ 原始方法HPWL提取失败")
    except Exception as e:
        logger.error(f"❌ 原始方法异常: {e}")
        results['original'] = None
    
    # 5. 比较结果
    successful_results = {k: v for k, v in results.items() if v is not None}
    if len(successful_results) > 1:
        logger.info("📊 方法对比:")
        methods = list(successful_results.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                hpwl1 = successful_results[method1]
                hpwl2 = successful_results[method2]
                diff = abs(hpwl1 - hpwl2)
                diff_pct = (diff / max(hpwl1, hpwl2)) * 100
                logger.info(f"  {method1} vs {method2}: 差异 {diff:.0f} ({diff_pct:.1f}%)")
    
    # 6. 计算奖励
    try:
        reward = experiment._execute_layout_and_calculate_reward(design_dir, {})
        logger.info(f"💰 计算奖励: {reward:.3f}")
        results['reward'] = reward
    except Exception as e:
        logger.error(f"❌ 奖励计算异常: {e}")
        results['reward'] = None
    
    return results

def test_sample_def():
    """测试示例DEF文件"""
    logger = setup_logging()
    
    # 创建示例DEF文件
    sample_def = create_sample_def_file()
    
    # 创建实验实例 - 使用服务器模式避免Docker依赖
    experiment = UnifiedPaperExperiment(mode="server")
    
    logger.info(f"\n=== 测试示例DEF文件: {sample_def} ===")
    
    # 测试所有HPWL方法
    results = test_hpwl_methods_on_file(experiment, sample_def)
    
    # 清理
    sample_def.unlink()
    logger.info("示例DEF文件已删除")
    
    return results

def test_real_def_files():
    """测试真实DEF文件"""
    logger = setup_logging()
    
    # 创建实验实例 - 使用服务器模式
    experiment = UnifiedPaperExperiment(mode="server")
    
    # 查找测试用的DEF文件 - 检查多个可能的文件名和位置
    dataset_dir = Path("dataset/ispd_2015_contest_benchmark")
    data_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    def_files = []
    
    # 定义可能的DEF文件名（按优先级排序）
    possible_def_names = [
        "placed.def",           # experiment.py生成的
        "placement_result.def", # real_openroad_interface_fixed.py生成的
        "final_layout.def",     # 其他脚本生成的
        "floorplan.def"         # 初始未布局文件（用于对比）
    ]
    
    # 首先检查dataset目录
    if dataset_dir.exists():
        for design_dir in dataset_dir.iterdir():
            if design_dir.is_dir():
                for def_name in possible_def_names:
                    def_file = design_dir / def_name
                    if def_file.exists():
                        def_files.append((def_file, design_dir))
                        logger.info(f"找到DEF文件: {def_file}")
                        break  # 找到第一个就停止，避免重复
    
    # 如果dataset目录没有找到，则检查data目录
    if not def_files and data_dir.exists():
        for design_dir in data_dir.iterdir():
            if design_dir.is_dir():
                for def_name in possible_def_names:
                    def_file = design_dir / def_name
                    if def_file.exists():
                        def_files.append((def_file, design_dir))
                        logger.info(f"找到DEF文件: {def_file}")
                        break  # 找到第一个就停止，避免重复
    
    if not def_files:
        logger.error("未找到任何DEF文件")
        logger.info("请确保OpenROAD布局脚本已运行并生成了DEF文件")
        logger.info("可能的文件名: placed.def, placement_result.def, final_layout.def")
        return {}
    
    if not def_files:
        logger.error("未找到任何DEF文件")
        return {}
    
    logger.info(f"找到 {len(def_files)} 个DEF文件进行测试")
    
    all_results = {}
    
    # 测试每个DEF文件
    for def_file, design_dir in def_files:
        results = test_hpwl_methods_on_file(experiment, def_file, design_dir)
        all_results[def_file.name] = results
        
        logger.info("=" * 50)
    
    return all_results

def generate_summary_report(all_results: Dict[str, Dict[str, Optional[float]]]):
    """生成测试总结报告"""
    logger = setup_logging()
    
    logger.info("\n" + "="*60)
    logger.info("📋 HPWL提取方法测试总结报告")
    logger.info("="*60)
    
    # 统计各方法的成功情况
    method_stats = {
        'openroad_builtin': {'success': 0, 'total': 0},
        'ispd2005_style': {'success': 0, 'total': 0},
        'original': {'success': 0, 'total': 0}
    }
    
    # 收集所有成功的HPWL值
    hpwl_values = {
        'openroad_builtin': [],
        'ispd2005_style': [],
        'original': []
    }
    
    for file_name, results in all_results.items():
        logger.info(f"\n📁 文件: {file_name}")
        
        for method in method_stats.keys():
            method_stats[method]['total'] += 1
            if results.get(method) is not None:
                method_stats[method]['success'] += 1
                hpwl_values[method].append(results[method])
                logger.info(f"  ✅ {method}: {results[method]:.0f}")
            else:
                logger.info(f"  ❌ {method}: 失败")
        
        if results.get('reward') is not None:
            logger.info(f"  💰 奖励: {results['reward']:.3f}")
    
    # 输出统计信息
    logger.info("\n📊 方法成功率统计:")
    for method, stats in method_stats.items():
        success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        logger.info(f"  {method}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    # 输出HPWL值统计
    logger.info("\n📈 HPWL值统计:")
    for method, values in hpwl_values.items():
        if values:
            avg_hpwl = sum(values) / len(values)
            min_hpwl = min(values)
            max_hpwl = max(values)
            logger.info(f"  {method}: 平均={avg_hpwl:.0f}, 最小={min_hpwl:.0f}, 最大={max_hpwl:.0f}")
    
    # 推荐最佳方法
    best_method = max(method_stats.items(), key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0)
    logger.info(f"\n🏆 推荐方法: {best_method[0]} (成功率: {(best_method[1]['success'] / best_method[1]['total'] * 100):.1f}%)")

def main():
    """主函数"""
    logger = setup_logging()
    
    all_results = {}
    
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        # 测试示例DEF文件
        logger.info("🧪 运行示例DEF文件测试")
        sample_results = test_sample_def()
        all_results['sample_test.def'] = sample_results
    else:
        # 测试真实DEF文件
        logger.info("🔍 运行真实DEF文件测试")
        real_results = test_real_def_files()
        all_results.update(real_results)
    
    # 生成总结报告
    if all_results:
        generate_summary_report(all_results)
    else:
        logger.warning("没有获得任何测试结果")

if __name__ == "__main__":
    main() 