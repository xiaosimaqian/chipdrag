#!/usr/bin/env python3
"""
测试基线HPWL修复
"""

import os
import sys
import json
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_baseline_hpwl():
    """测试基线HPWL修复"""
    logger.info("开始测试基线HPWL修复...")
    
    # 测试设计
    test_designs = ["mgc_des_perf_1", "mgc_fft_1"]
    
    baseline_data = {}
    
    for design_name in test_designs:
        logger.info(f"测试设计: {design_name}")
        try:
            # 为每个设计运行基线OpenROAD布局
            design_path = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
            interface = RealOpenROADInterface(work_dir=design_path)
            
            # 使用默认参数运行基线布局
            logger.info(f"  运行基线布局...")
            start_time = datetime.now()
            baseline_result = interface.run_placement(
                density_target=0.7,  # 默认密度目标
                wirelength_weight=1.0,  # 默认线长权重
                density_weight=1.0  # 默认密度权重
            )
            baseline_time = (datetime.now() - start_time).total_seconds()
            
            baseline_hpwl = baseline_result.get('hpwl', float('inf'))
            
            baseline_data[design_name] = {
                'baseline_hpwl': baseline_hpwl,
                'baseline_success': baseline_result.get('success', False),
                'baseline_execution_time': baseline_time,
                'design_info': {
                    'design_name': design_name,
                    'design_path': design_path
                }
            }
            
            logger.info(f"  基线HPWL: {baseline_hpwl:.2e}, 成功: {baseline_result.get('success', False)}, 耗时: {baseline_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"基线布局 {design_name} 失败: {str(e)}")
            baseline_data[design_name] = {
                'baseline_hpwl': float('inf'),
                'baseline_success': False,
                'baseline_execution_time': 0,
                'design_info': {
                    'design_name': design_name,
                    'design_path': design_path
                }
            }
    
    logger.info(f"完成基线布局测试，成功运行 {sum(1 for d in baseline_data.values() if d['baseline_success'])}/{len(baseline_data)} 个设计")
    
    # 保存结果
    os.makedirs("results/test_baseline", exist_ok=True)
    with open("results/test_baseline/baseline_test_results.json", "w") as f:
        json.dump(baseline_data, f, indent=2, default=str)
    
    logger.info("基线测试结果已保存到: results/test_baseline/baseline_test_results.json")
    
    return baseline_data

if __name__ == "__main__":
    test_baseline_hpwl() 