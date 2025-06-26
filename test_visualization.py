#!/usr/bin/env python3
"""
测试可视化功能
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

def test_visualization():
    """测试可视化功能"""
    try:
        # 初始化接口
        interface = EnhancedOpenROADInterface()
        
        # 设置测试目录
        work_dir = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1")
        iterations_dir = work_dir / "output" / "iterations"
        
        if not iterations_dir.exists():
            logger.error(f"迭代目录不存在: {iterations_dir}")
            return False
        
        logger.info(f"开始测试可视化功能")
        
        # 收集迭代数据
        iteration_data = interface.collect_iteration_data(str(iterations_dir))
        
        if not iteration_data:
            logger.error("没有收集到迭代数据")
            return False
        
        logger.info(f"收集到 {len(iteration_data)} 个迭代数据")
        
        # 生成可视化
        output_dir = str(work_dir / "output")
        viz_path = interface.generate_iteration_visualization(iteration_data, output_dir)
        
        logger.info(f"可视化文件保存在: {viz_path}")
        
        # 检查生成的文件
        viz_dir = Path(viz_path)
        if viz_dir.exists():
            files = list(viz_dir.glob("*"))
            logger.info(f"生成的可视化文件: {[f.name for f in files]}")
            
            # 检查文件大小
            for file in files:
                size = file.stat().st_size
                logger.info(f"{file.name}: {size} bytes")
                if size < 1000:
                    logger.warning(f"{file.name} 文件太小，可能没有内容")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization()
    if success:
        print("✅ 可视化测试成功")
    else:
        print("❌ 可视化测试失败")
        sys.exit(1) 