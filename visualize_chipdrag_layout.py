#!/usr/bin/env python3
"""
ChipDRAG布局可视化工具

从DEF文件中解析布局信息并生成可视化图片
支持对比显示：原始布局 vs ChipDRAG优化布局
"""

import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DEFLayoutParser:
    """DEF文件布局解析器"""
    
    def __init__(self):
        self.components = []
        self.die_area = None
        self.design_name = ""
        
    def parse_def_file(self, def_file: Path) -> Dict:
        """解析DEF文件获取布局信息"""
        logger.info(f"解析DEF文件: {def_file}")
        
        layout_data = {
            'design_name': '',
            'die_area': None,
            'components': [],
            'component_count': 0,
            'placed_count': 0
        }
        
        try:
            with open(def_file, 'r') as f:
                content = f.read()
            
            # 1. 提取设计名称
            design_match = re.search(r'DESIGN\s+(\w+)', content)
            if design_match:
                layout_data['design_name'] = design_match.group(1)
            
            # 2. 提取芯片区域
            die_area_match = re.search(r'DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
            if die_area_match:
                x1, y1, x2, y2 = map(int, die_area_match.groups())
                layout_data['die_area'] = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1, 'height': y2 - y1
                }
                logger.info(f"芯片区域: ({x1}, {y1}) -> ({x2}, {y2}), 尺寸: {x2-x1} x {y2-y1}")
            
            # 3. 提取组件信息
            components_section = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END COMPONENTS', content, re.DOTALL)
            if components_section:
                component_count = int(components_section.group(1))
                components_text = components_section.group(2)
                layout_data['component_count'] = component_count
                
                logger.info(f"总组件数: {component_count}")
                
                # 解析每个组件 - 修正正则表达式
                # 格式: - FE_OFC0_n_17395 in01f01 + PLACED ( 132600 10000 ) N ;
                component_pattern = r'-\s+(\w+)\s+(\w+)\s*\+\s*PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\w+)\s*;'
                components = re.findall(component_pattern, components_text)
                
                logger.info(f"成功解析的组件数: {len(components)}")
                
                for comp_name, comp_type, x, y, orientation in components:
                    component = {
                        'name': comp_name,
                        'type': comp_type,
                        'placed': True,  # 有PLACED关键字就是已放置
                        'x': int(x),
                        'y': int(y),
                        'orientation': orientation,
                        'width': 200,    # 从LEF文件中应该能获取真实尺寸，这里用默认值
                        'height': 2000   # 默认高度
                    }
                    
                    layout_data['placed_count'] += 1
                    layout_data['components'].append(component)
                
                # 如果没有解析到PLACED组件，尝试解析未放置的组件
                if layout_data['placed_count'] == 0:
                    logger.info("未找到PLACED组件，尝试解析未放置组件...")
                    # 格式: - component_name cell_type ;
                    unplaced_pattern = r'-\s+(\w+)\s+(\w+)\s*;'
                    unplaced_components = re.findall(unplaced_pattern, components_text)
                    
                    logger.info(f"找到未放置组件数: {len(unplaced_components)}")
                    
                    for comp_name, comp_type in unplaced_components:
                        component = {
                            'name': comp_name,
                            'type': comp_type,
                            'placed': False,
                            'x': 0,
                            'y': 0,
                            'orientation': 'N',
                            'width': 200,
                            'height': 2000
                        }
                        layout_data['components'].append(component)
                
                logger.info(f"已放置组件数: {layout_data['placed_count']}/{component_count}")
            else:
                logger.warning("未找到COMPONENTS段")
            
            return layout_data
            
        except Exception as e:
            logger.error(f"解析DEF文件失败: {e}")
            return layout_data

class ChipDRAGLayoutVisualizer:
    """ChipDRAG布局可视化器"""
    
    def __init__(self):
        self.colors = plt.cm.Set3.colors
        
    def visualize_layout(self, layout_data: Dict, title: str = "布局可视化", save_path: Optional[Path] = None):
        """可视化单个布局"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取芯片区域
        die_area = layout_data.get('die_area')
        if die_area:
            # 绘制芯片边界
            boundary = patches.Rectangle(
                (die_area['x1'], die_area['y1']),
                die_area['width'],
                die_area['height'],
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                label='芯片边界'
            )
            ax.add_patch(boundary)
            
            # 设置坐标轴范围
            margin = max(die_area['width'], die_area['height']) * 0.05
            ax.set_xlim(die_area['x1'] - margin, die_area['x2'] + margin)
            ax.set_ylim(die_area['y1'] - margin, die_area['y2'] + margin)
        
        # 绘制组件
        placed_components = [comp for comp in layout_data['components'] if comp['placed']]
        logger.info(f"绘制 {len(placed_components)} 个已放置组件")
        
        # 如果组件太多，只显示部分代表性组件
        if len(placed_components) > 1000:
            # 采样显示，保持空间分布
            step = len(placed_components) // 1000
            placed_components = placed_components[::step]
            logger.info(f"组件过多，采样显示 {len(placed_components)} 个组件")
        
        for i, comp in enumerate(placed_components):
            color = self.colors[i % len(self.colors)]
            
            # 绘制组件矩形
            rect = patches.Rectangle(
                (comp['x'], comp['y']),
                comp['width'],
                comp['height'],
                linewidth=0.5,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title(f"{title}\n设计: {layout_data['design_name']}, 组件: {layout_data['placed_count']}/{layout_data['component_count']}")
        
        # 添加图例
        if die_area:
            ax.legend()
        
        # 保存或显示
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"布局图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_layouts(self, original_layout: Dict, optimized_layout: Dict, save_path: Optional[Path] = None):
        """对比显示原始布局和优化布局"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 绘制原始布局
        self._draw_single_layout(ax1, original_layout, "原始布局 (floorplan.def)")
        
        # 绘制优化布局
        self._draw_single_layout(ax2, optimized_layout, "ChipDRAG优化布局 (placed.def)")
        
        # 添加总标题
        fig.suptitle(f"ChipDRAG布局对比 - {optimized_layout['design_name']}", fontsize=16)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"对比图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _draw_single_layout(self, ax, layout_data: Dict, title: str):
        """在指定轴上绘制单个布局"""
        # 获取芯片区域
        die_area = layout_data.get('die_area')
        if die_area:
            # 绘制芯片边界
            boundary = patches.Rectangle(
                (die_area['x1'], die_area['y1']),
                die_area['width'],
                die_area['height'],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(boundary)
            
            # 设置坐标轴范围
            margin = max(die_area['width'], die_area['height']) * 0.05
            ax.set_xlim(die_area['x1'] - margin, die_area['x2'] + margin)
            ax.set_ylim(die_area['y1'] - margin, die_area['y2'] + margin)
        
        # 绘制组件
        placed_components = [comp for comp in layout_data['components'] if comp['placed']]
        
        # 采样显示大量组件
        if len(placed_components) > 500:
            step = len(placed_components) // 500
            placed_components = placed_components[::step]
        
        for i, comp in enumerate(placed_components):
            color = self.colors[i % len(self.colors)]
            
            rect = patches.Rectangle(
                (comp['x'], comp['y']),
                comp['width'],
                comp['height'],
                linewidth=0.3,
                edgecolor='black',
                facecolor=color,
                alpha=0.6
            )
            ax.add_patch(rect)
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title(f"{title}\n组件: {layout_data['placed_count']}/{layout_data['component_count']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ChipDRAG布局可视化工具')
    parser.add_argument('design', help='设计名称 (如: mgc_fft_1)')
    parser.add_argument('--compare', action='store_true', help='对比显示原始布局和优化布局')
    parser.add_argument('--output', '-o', help='输出图片路径')
    
    args = parser.parse_args()
    
    # 设计目录
    design_dir = Path(f"dataset/ispd_2015_contest_benchmark/{args.design}")
    
    if not design_dir.exists():
        logger.error(f"设计目录不存在: {design_dir}")
        return
    
    # 初始化解析器和可视化器
    parser = DEFLayoutParser()
    visualizer = ChipDRAGLayoutVisualizer()
    
    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"layout_visualization_{args.design}.png")
    
    if args.compare:
        # 对比模式
        logger.info("=== 对比模式：原始布局 vs ChipDRAG优化布局 ===")
        
        # 解析原始布局
        floorplan_def = design_dir / "floorplan.def"
        if not floorplan_def.exists():
            logger.error(f"原始布局文件不存在: {floorplan_def}")
            return
        
        original_layout = parser.parse_def_file(floorplan_def)
        
        # 解析优化布局
        placed_def = design_dir / "placed.def"
        if not placed_def.exists():
            logger.error(f"优化布局文件不存在: {placed_def}")
            logger.info("请先运行ChipDRAG实验生成placed.def文件")
            return
        
        optimized_layout = parser.parse_def_file(placed_def)
        
        # 生成对比图
        visualizer.compare_layouts(original_layout, optimized_layout, output_path)
        
    else:
        # 单一布局模式
        logger.info("=== 单一布局模式 ===")
        
        # 优先显示ChipDRAG优化布局
        placed_def = design_dir / "placed.def"
        if placed_def.exists():
            logger.info("显示ChipDRAG优化布局")
            layout_data = parser.parse_def_file(placed_def)
            title = f"ChipDRAG优化布局 - {args.design}"
        else:
            logger.info("显示原始布局")
            floorplan_def = design_dir / "floorplan.def"
            if not floorplan_def.exists():
                logger.error(f"布局文件不存在: {floorplan_def}")
                return
            layout_data = parser.parse_def_file(floorplan_def)
            title = f"原始布局 - {args.design}"
        
        # 生成可视化
        visualizer.visualize_layout(layout_data, title, output_path)

if __name__ == "__main__":
    main() 