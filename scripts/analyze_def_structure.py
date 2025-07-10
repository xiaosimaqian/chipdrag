#!/usr/bin/env python3
"""
DEF文件结构分析工具

专门用于分析DEF文件的结构，帮助调试HPWL提取问题
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def analyze_def_file_structure(def_file: Path):
    """分析DEF文件结构"""
    logger = setup_logging()
    
    logger.info(f"\n=== 分析DEF文件结构: {def_file} ===")
    
    try:
        with open(def_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # 统计各部分
        components_section = False
        nets_section = False
        component_lines = []
        net_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('COMPONENTS'):
                components_section = True
                logger.info(f"COMPONENTS声明: {line}")
                continue
            elif line.startswith('END COMPONENTS'):
                components_section = False
                logger.info(f"END COMPONENTS在第{i+1}行")
                continue
            elif line.startswith('NETS'):
                nets_section = True
                logger.info(f"NETS声明: {line}")
                continue
            elif line.startswith('END NETS'):
                nets_section = False
                logger.info(f"END NETS在第{i+1}行")
                continue
            
            if components_section and line.startswith('-'):
                component_lines.append((i+1, line))
            elif nets_section and line.startswith('-'):
                net_lines.append((i+1, line))
        
        logger.info(f"组件行数: {len(component_lines)}")
        logger.info(f"网络行数: {len(net_lines)}")
        
        # 显示前几个组件
        if component_lines:
            logger.info("前5个组件:")
            for i, (line_num, line) in enumerate(component_lines[:5]):
                logger.info(f"  第{line_num}行: {line}")
        
        # 显示前几个网络
        if net_lines:
            logger.info("前5个网络:")
            for i, (line_num, line) in enumerate(net_lines[:5]):
                logger.info(f"  第{line_num}行: {line}")
        
        # 查找DIEAREA
        for line in lines:
            if 'DIEAREA' in line:
                logger.info(f"DIEAREA: {line}")
                break
        
        # 分析PLACED组件
        placed_components = []
        for line_num, line in component_lines:
            if 'PLACED' in line:
                placed_components.append((line_num, line))
        
        logger.info(f"已放置组件数: {len(placed_components)}")
        if placed_components:
            logger.info("前3个已放置组件:")
            for i, (line_num, line) in enumerate(placed_components[:3]):
                logger.info(f"  第{line_num}行: {line}")
        
    except Exception as e:
        logger.error(f"分析DEF文件失败: {e}")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 如果提供了DEF文件路径，分析该文件
        def_file = Path(sys.argv[1])
        if def_file.exists():
            analyze_def_file_structure(def_file)
        else:
            print(f"DEF文件不存在: {def_file}")
    else:
        print("用法: python scripts/analyze_def_structure.py <def_file_path>")
        print("示例: python scripts/analyze_def_structure.py data/design1/placed.def")

if __name__ == "__main__":
    main() 