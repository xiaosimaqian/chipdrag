#!/usr/bin/env python3
"""
优化密度参数的ISPD 2015批量训练脚本
降低布局密度，避免布线拥塞，提高HPWL提取成功率
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_training.log'),
        logging.StreamHandler()
    ]
)

def find_ispd_designs():
    """查找所有ISPD 2015设计"""
    designs_dir = Path('data/designs/ispd_2015_contest_benchmark')
    designs = []
    
    if not designs_dir.exists():
        logging.error(f"设计目录不存在: {designs_dir}")
        return designs
    
    for design_dir in designs_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        # 检查必要文件
        verilog_file = design_dir / 'design.v'
        floorplan_def = design_dir / 'floorplan.def'
        tech_lef = design_dir / 'tech.lef'
        cells_lef = design_dir / 'cells.lef'
        
        if all(f.exists() for f in [verilog_file, floorplan_def, tech_lef, cells_lef]):
            designs.append(design_dir.name)
            logging.info(f"找到设计: {design_dir.name}")
    
    return designs

def train_single_design(design_name):
    """训练单个设计"""
    try:
        design_dir = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
        
        # 创建OpenROAD接口
        interface = RealOpenROADInterface(design_dir)
        
        # 运行迭代布局训练
        logging.info(f"开始训练设计: {design_name}")
        result = interface.run_iterative_placement(num_iterations=10)
        
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
    logging.info("开始优化密度参数的批量训练")
    
    # 查找所有设计
    designs = find_ispd_designs()
    if not designs:
        logging.error("未找到任何有效的ISPD设计")
        return
    
    logging.info(f"总共找到 {len(designs)} 个有效设计")
    
    # 创建结果目录
    results_dir = Path('results/optimized_training')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练所有设计
    results = []
    start_time = time.time()
    
    for i, design_name in enumerate(designs, 1):
        logging.info(f"进度 {i}/{len(designs)}: 训练设计 {design_name}")
        
        result = train_single_design(design_name)
        results.append(result)
        
        # 保存中间结果
        with open(results_dir / 'optimized_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    
    # 生成训练报告
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    report = f"""
# 优化密度参数批量训练报告

## 训练概览
- 开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}
- 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总耗时: {total_time:.2f} 秒
- 设计总数: {len(designs)}
- 成功: {successful}
- 失败: {failed}
- 成功率: {successful/len(designs)*100:.1f}%

## 密度优化说明
- 超大型设计 (>500K实例): 密度 40%, 芯片面积 2500x2500
- 大型设计 (>200K实例): 密度 45%, 芯片面积 2000x2000  
- 中型设计 (>100K实例): 密度 50%, 芯片面积 1500x1500
- 小型设计: 密度 55%, 芯片面积 1200x1200

## 详细结果
"""
    
    for result in results:
        if result['success']:
            report += f"- ✅ {result['design']}: 耗时 {result['execution_time']:.1f}秒\n"
        else:
            report += f"- ❌ {result['design']}: {result.get('error', '未知错误')}\n"
    
    # 保存报告
    with open(results_dir / 'optimized_training_report.md', 'w') as f:
        f.write(report)
    
    logging.info(f"批量训练完成！成功: {successful}/{len(designs)}")
    logging.info(f"结果已保存到: {results_dir}")

if __name__ == '__main__':
    main() 