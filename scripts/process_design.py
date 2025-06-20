import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.parsers.design_parser import DesignParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(directory: str) -> None:
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def process_design(design_dir: str, output_dir: str) -> None:
    """处理设计文件并生成JSON输出
    
    Args:
        design_dir: 设计文件目录
        output_dir: 输出目录
    """
    try:
        # 确保输出目录存在
        ensure_dir(output_dir)
        
        # 创建设计解析器
        parser = DesignParser()
        
        # 解析设计文件
        logger.info(f"开始解析设计目录: {design_dir}")
        design_data = parser.parse_design(design_dir)
        
        # 保存解析结果
        output_file = os.path.join(output_dir, "design.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(design_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"设计解析完成，结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"处理设计文件时出错: {str(e)}")
        raise

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python process_design.py <设计目录> <输出目录>")
        sys.exit(1)
        
    design_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_design(design_dir, output_dir)

if __name__ == "__main__":
    main() 