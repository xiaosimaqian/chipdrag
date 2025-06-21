#!/usr/bin/env python3
"""
运行布局可视化脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_layout_visualization import OpenROADLayoutVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 设计目录
    design_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    output_dir = f"{design_dir}/output"
    
    # 检查DEF文件是否存在
    initial_def = f"{output_dir}/initial_layout.def"
    final_def = f"{output_dir}/after_global_placement.def"
    
    if not os.path.exists(initial_def):
        logger.error(f"初始布局DEF文件不存在: {initial_def}")
        logger.info("请先运行OpenROAD流程生成DEF文件")
        return
    
    if not os.path.exists(final_def):
        logger.error(f"最终布局DEF文件不存在: {final_def}")
        logger.info("请先运行OpenROAD流程生成DEF文件")
        return
    
    # 创建可视化器
    visualizer = OpenROADLayoutVisualizer()
    
    try:
        logger.info("开始生成布局可视化图")
        
        # 生成初始布局图
        logger.info("生成初始布局图")
        initial_image = visualizer.generate_layout_visualization(
            initial_def, 
            f"{output_dir}/visualization",
            {'iteration': 'initial'}
        )
        
        # 生成最终布局图
        logger.info("生成最终布局图")
        final_image = visualizer.generate_layout_visualization(
            final_def, 
            f"{output_dir}/visualization",
            {'iteration': 'final'}
        )
        
        # 生成对比图
        logger.info("生成布局对比图")
        comparison_image = visualizer.create_comparison_plot(
            [initial_image, final_image],
            f"{output_dir}/visualization",
            "OpenROAD布局迭代对比"
        )
        
        logger.info("布局可视化生成完成")
        logger.info(f"初始布局图: {initial_image}")
        logger.info(f"最终布局图: {final_image}")
        logger.info(f"对比图: {comparison_image}")
        
    except Exception as e:
        logger.error(f"生成布局可视化失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 