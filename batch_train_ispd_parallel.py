#!/usr/bin/env python3
"""
并行处理的ISPD 2015批量训练脚本
支持跳过大型设计，训练/测试集分割，并行处理
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_training.log'),
        logging.StreamHandler()
    ]
)

# 跳过大型设计（处理时间过长）
SKIP_LARGE_DESIGNS = {
    'mgc_superblue16_a', 'mgc_superblue11_a', 'mgc_des_perf_a'
}

def get_design_size(design_dir):
    """估算设计规模"""
    try:
        verilog_file = Path(design_dir) / 'design.v'
        with open(verilog_file, 'r') as f:
            content = f.read()
            # 统计实例数量
            instances = len([line for line in content.split('\n') if 'module' in line and 'endmodule' not in line])
            return instances
    except:
        return 0

def find_ispd_designs():
    """查找所有ISPD 2015设计，按规模分类"""
    designs_dir = Path('data/designs/ispd_2015_contest_benchmark')
    small_designs = []
    large_designs = []
    
    if not designs_dir.exists():
        logging.error(f"设计目录不存在: {designs_dir}")
        return small_designs, large_designs
    
    for design_dir in designs_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        design_name = design_dir.name
        if design_name in SKIP_LARGE_DESIGNS:
            logging.info(f"跳过大型设计: {design_name}")
            continue
            
        # 检查必要文件
        verilog_file = design_dir / 'design.v'
        floorplan_def = design_dir / 'floorplan.def'
        tech_lef = design_dir / 'tech.lef'
        cells_lef = design_dir / 'cells.lef'
        
        if all(f.exists() for f in [verilog_file, floorplan_def, tech_lef, cells_lef]):
            size = get_design_size(design_dir)
            if size > 100000:  # 大型设计
                large_designs.append(design_name)
                logging.info(f"找到大型设计: {design_name} ({size}实例)")
            else:
                small_designs.append(design_name)
                logging.info(f"找到中小型设计: {design_name} ({size}实例)")
    
    return small_designs, large_designs

def split_train_test(designs):
    """分割训练集和测试集"""
    if len(designs) <= 1:
        return designs, []
    
    # 保留1个设计作为测试集
    test_design = designs[-1]  # 选择最后一个作为测试集
    train_designs = designs[:-1]  # 其余作为训练集
    
    return train_designs, [test_design]

def train_single_design(design_name):
    """训练单个设计"""
    try:
        design_dir = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
        
        # 创建OpenROAD接口
        interface = RealOpenROADInterface(design_dir)
        
        # 根据设计规模设置超时
        size = get_design_size(design_dir)
        if size > 50000:
            timeout = 1200  # 20分钟
        else:
            timeout = 600   # 10分钟
        
        # 运行迭代布局训练
        logging.info(f"开始训练设计: {design_name} (超时: {timeout}秒)")
        result = interface.run_iterative_placement(num_iterations=10, timeout=timeout)
        
        if result['success']:
            logging.info(f"✅ 设计 {design_name} 训练成功")
            return {
                'design': design_name,
                'success': True,
                'execution_time': result['execution_time'],
                'iterations': result['iterations'],
                'final_hpwl': result.get('final_hpwl', None)
            }
        else:
            logging.error(f"❌ 设计 {design_name} 训练失败: {result.get('error', '未知错误')}")
            return {
                'design': design_name,
                'success': False,
                'error': result.get('error', '未知错误')
            }
            
    except Exception as e:
        logging.error(f"❌ 设计 {design_name} 训练异常: {str(e)}")
        return {
            'design': design_name,
            'success': False,
            'error': str(e)
        }

def main():
    """主函数"""
    logging.info("开始并行批量训练（训练/测试集分割）")
    
    # 查找所有设计
    small_designs, large_designs = find_ispd_designs()
    
    if not small_designs:
        logging.error("未找到任何有效的中小型设计")
        return
    
    # 分割训练集和测试集
    train_designs, test_designs = split_train_test(small_designs)
    
    logging.info(f"中小型设计总数: {len(small_designs)}")
    logging.info(f"训练集: {len(train_designs)} 个设计 - {train_designs}")
    logging.info(f"测试集: {len(test_designs)} 个设计 - {test_designs}")
    logging.info(f"跳过大型设计: {len(large_designs)} 个")
    
    # 创建结果目录
    results_dir = Path('results/parallel_training')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据集分割信息
    dataset_info = {
        'train_designs': train_designs,
        'test_designs': test_designs,
        'skipped_large_designs': list(large_designs),
        'total_small_designs': len(small_designs),
        'total_large_designs': len(large_designs)
    }
    
    with open(results_dir / 'dataset_split.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 处理训练集（并行）
    results = []
    start_time = time.time()
    
    if train_designs:
        logging.info(f"开始并行处理训练集: {len(train_designs)} 个设计")
        
        # 使用进程池并行处理
        max_workers = min(cpu_count(), 4)  # 最多4个进程
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_design = {executor.submit(train_single_design, design): design 
                              for design in train_designs}
            
            # 收集结果
            for future in as_completed(future_to_design):
                design = future_to_design[future]
                try:
                    result = future.result()
                    result['dataset'] = 'train'  # 标记为训练集
                    results.append(result)
                    logging.info(f"完成训练设计: {design}")
                except Exception as e:
                    logging.error(f"训练设计 {design} 处理异常: {e}")
                    results.append({
                        'design': design,
                        'dataset': 'train',
                        'success': False,
                        'error': str(e)
                    })
    
    total_time = time.time() - start_time
    
    # 生成训练报告
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    report = f"""
# 并行批量训练报告（训练/测试集分割）

## 训练概览
- 开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
- 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总耗时: {total_time:.2f} 秒

## 数据集分割
- 中小型设计总数: {len(small_designs)}
- 训练集: {len(train_designs)} 个设计
- 测试集: {len(test_designs)} 个设计
- 跳过大型设计: {len(large_designs)} 个

## 训练结果
- 训练设计总数: {len(train_designs)}
- 成功: {successful}
- 失败: {failed}
- 成功率: {successful/len(train_designs)*100:.1f}% (如果训练集不为空)

## 处理策略
- 跳过大型设计: {', '.join(SKIP_LARGE_DESIGNS)}
- 训练集: 并行处理
- 测试集: 保留用于后续实验
- 最大并行数: {min(cpu_count(), 4)}

## 训练集详细结果
"""
    
    for result in results:
        if result['success']:
            report += f"- ✅ {result['design']}: 耗时 {result['execution_time']:.1f}秒\n"
        else:
            report += f"- ❌ {result['design']}: {result.get('error', '未知错误')}\n"
    
    if test_designs:
        report += f"\n## 测试集（保留用于后续实验）\n"
        for design in test_designs:
            report += f"- 🔬 {design}\n"
    
    # 保存报告
    with open(results_dir / 'parallel_training_report.md', 'w') as f:
        f.write(report)
    
    # 保存结果
    with open(results_dir / 'parallel_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"并行批量训练完成！")
    logging.info(f"训练集成功: {successful}/{len(train_designs)}")
    logging.info(f"测试集保留: {len(test_designs)} 个设计")
    logging.info(f"结果已保存到: {results_dir}")

if __name__ == '__main__':
    main() 