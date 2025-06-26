#!/usr/bin/env python3
"""
测试迭代数据收集功能
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_openroad_interface import EnhancedOpenROADInterface

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_collection():
    """测试数据收集功能"""
    try:
        # 初始化接口
        interface = EnhancedOpenROADInterface()
        
        # 设置测试目录
        work_dir = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1")
        iterations_dir = work_dir / "output" / "iterations"
        
        if not iterations_dir.exists():
            logger.error(f"迭代目录不存在: {iterations_dir}")
            return False
        
        logger.info(f"开始收集迭代数据从: {iterations_dir}")
        
        # 收集迭代数据
        iteration_data = interface.collect_iteration_data(str(iterations_dir))
        
        logger.info(f"成功收集到 {len(iteration_data)} 个迭代数据")
        
        # 显示收集到的数据
        for data in iteration_data:
            logger.info(f"迭代 {data['iteration']}: "
                       f"HPWL={data.get('hpwl', 'None')}, "
                       f"溢出率={data.get('overflow', 'None')}, "
                       f"文件={Path(data['def_file']).name}")
        
        # 检查是否有HPWL和溢出率数据
        hpwl_count = sum(1 for data in iteration_data if data.get('hpwl') is not None)
        overflow_count = sum(1 for data in iteration_data if data.get('overflow') is not None)
        
        logger.info(f"包含HPWL数据的迭代: {hpwl_count}/{len(iteration_data)}")
        logger.info(f"包含溢出率数据的迭代: {overflow_count}/{len(iteration_data)}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_data_collection()
    if success:
        print("✅ 数据收集测试成功")
    else:
        print("❌ 数据收集测试失败")
        sys.exit(1) 