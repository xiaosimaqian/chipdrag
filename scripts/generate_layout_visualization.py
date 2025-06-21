#!/usr/bin/env python3
"""
OpenROAD布局可视化生成器
从DEF文件生成迭代后的布局图
"""

import os
import sys
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.parsers.def_parser import DEFParser
from modules.visualization.layout_visualizer import LayoutVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenROADLayoutVisualizer:
    """OpenROAD布局可视化器"""
    
    def __init__(self):
        self.def_parser = DEFParser()
        self.visualizer = LayoutVisualizer()
        
    def parse_def_file(self, def_file_path: str) -> Dict[str, Any]:
        """解析DEF文件
        
        Args:
            def_file_path: DEF文件路径
            
        Returns:
            Dict[str, Any]: 解析后的布局信息
        """
        try:
            logger.info(f"解析DEF文件: {def_file_path}")
            
            # 解析DEF文件
            layout_info = self.def_parser.parse(def_file_path)
            
            # 提取关键信息
            die_area = layout_info.get('die_area', {})
            components = layout_info.get('components', [])
            nets = layout_info.get('nets', [])
            
            logger.info(f"Die区域: {die_area}")
            logger.info(f"组件数量: {len(components)}")
            logger.info(f"网络数量: {len(nets)}")
            
            return layout_info
            
        except Exception as e:
            logger.error(f"解析DEF文件失败: {str(e)}")
            raise
    
    def generate_layout_visualization(self, 
                                    def_file_path: str, 
                                    output_dir: str,
                                    iteration_info: Dict[str, Any] = None) -> str:
        """生成布局可视化图
        
        Args:
            def_file_path: DEF文件路径
            output_dir: 输出目录
            iteration_info: 迭代信息（可选）
            
        Returns:
            str: 生成的图片路径
        """
        try:
            # 解析DEF文件
            layout_info = self.parse_def_file(def_file_path)
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成图片文件名
            def_name = Path(def_file_path).stem
            if iteration_info:
                iteration_num = iteration_info.get('iteration', 'final')
                image_name = f"{def_name}_iteration_{iteration_num}_layout.png"
            else:
                image_name = f"{def_name}_final_layout.png"
            
            image_path = os.path.join(output_dir, image_name)
            
            # 添加迭代信息到布局数据
            if iteration_info:
                layout_info['iteration_info'] = iteration_info
            
            # 生成可视化
            self.visualizer.visualize(layout_info, image_path)
            
            logger.info(f"布局图已生成: {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"生成布局可视化失败: {str(e)}")
            raise
    
    def generate_iteration_sequence(self, 
                                  def_files: List[str], 
                                  output_dir: str,
                                  iteration_data: List[Dict[str, Any]] = None) -> List[str]:
        """生成迭代序列的布局图
        
        Args:
            def_files: DEF文件列表（按迭代顺序）
            output_dir: 输出目录
            iteration_data: 迭代数据（可选）
            
        Returns:
            List[str]: 生成的图片路径列表
        """
        try:
            image_paths = []
            
            for i, def_file in enumerate(def_files):
                iteration_info = None
                if iteration_data and i < len(iteration_data):
                    iteration_info = iteration_data[i]
                else:
                    iteration_info = {'iteration': i}
                
                image_path = self.generate_layout_visualization(
                    def_file, output_dir, iteration_info
                )
                image_paths.append(image_path)
            
            logger.info(f"生成了 {len(image_paths)} 张布局图")
            return image_paths
            
        except Exception as e:
            logger.error(f"生成迭代序列失败: {str(e)}")
            raise
    
    def create_comparison_plot(self, 
                             image_paths: List[str], 
                             output_dir: str,
                             title: str = "布局迭代对比") -> str:
        """创建对比图
        
        Args:
            image_paths: 图片路径列表
            output_dir: 输出目录
            title: 标题
            
        Returns:
            str: 对比图路径
        """
        try:
            n_images = len(image_paths)
            if n_images == 0:
                raise ValueError("没有图片路径")
            
            # 计算子图布局
            cols = min(3, n_images)
            rows = (n_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_images == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            # 加载并显示每张图片
            for i, image_path in enumerate(image_paths):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                # 加载图片
                img = plt.imread(image_path)
                ax.imshow(img)
                ax.set_title(f"迭代 {i}")
                ax.axis('off')
            
            # 隐藏多余的子图
            for i in range(n_images, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            # 设置总标题
            fig.suptitle(title, fontsize=16, y=0.95)
            
            # 保存对比图
            comparison_path = os.path.join(output_dir, "layout_iteration_comparison.png")
            plt.tight_layout()
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"对比图已生成: {comparison_path}")
            return comparison_path
            
        except Exception as e:
            logger.error(f"创建对比图失败: {str(e)}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OpenROAD布局可视化生成器")
    parser.add_argument("--def_file", type=str, required=True, help="DEF文件路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--iteration", type=int, help="迭代编号")
    parser.add_argument("--comparison", action="store_true", help="生成对比图")
    parser.add_argument("--def_files", nargs="+", help="多个DEF文件路径（用于对比）")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = OpenROADLayoutVisualizer()
    
    try:
        if args.def_files and len(args.def_files) > 1:
            # 生成迭代序列对比
            iteration_data = []
            for i in range(len(args.def_files)):
                iteration_data.append({
                    'iteration': i,
                    'hpwl': None,  # 可以从日志中提取
                    'overflow': None
                })
            
            image_paths = visualizer.generate_iteration_sequence(
                args.def_files, args.output_dir, iteration_data
            )
            
            if args.comparison:
                visualizer.create_comparison_plot(image_paths, args.output_dir)
                
        else:
            # 生成单个布局图
            iteration_info = None
            if args.iteration is not None:
                iteration_info = {'iteration': args.iteration}
            
            visualizer.generate_layout_visualization(
                args.def_file, args.output_dir, iteration_info
            )
        
        logger.info("布局可视化生成完成")
        
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 