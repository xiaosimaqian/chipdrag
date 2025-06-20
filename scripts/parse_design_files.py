import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.parsers.verilog_parser import parse_verilog_file
from modules.parsers.def_parser import parse_def_file
from modules.parsers.lef_parser import parse_lef_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(directory: str) -> None:
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_parsed_result(output_dir: str, design_name: str, file_type: str, data: Dict[str, Any]) -> None:
    """保存解析结果到JSON文件"""
    output_file = os.path.join(output_dir, f"{design_name}_{file_type}.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"已保存{file_type}解析结果到: {output_file}")
    except Exception as e:
        logger.error(f"保存{file_type}解析结果时出错: {str(e)}")

def parse_design_files(design_dir: str, output_dir: str) -> None:
    """解析设计文件"""
    try:
        # 确保输出目录存在
        ensure_dir(output_dir)
        
        # 遍历设计目录
        for design_name in os.listdir(design_dir):
            design_path = os.path.join(design_dir, design_name)
            if not os.path.isdir(design_path):
                continue
                
            logger.info(f"开始解析设计: {design_name}")
        
            # 创建设计特定的输出目录
            design_output_dir = os.path.join(output_dir, design_name)
            ensure_dir(design_output_dir)
        
            # 解析Verilog文件
            v_files = list(Path(design_path).rglob("*.v"))
            if v_files:
                logger.info(f"找到{len(v_files)}个Verilog文件")
                for v_file in v_files:
                    try:
                        logger.info(f"正在解析Verilog文件: {v_file}")
                        result = parse_verilog_file(str(v_file))
                        save_parsed_result(design_output_dir, v_file.stem, "verilog", result)
                    except Exception as e:
                        logger.error(f"解析Verilog文件{v_file}时出错: {str(e)}")
                
            # 解析DEF文件
            def_files = list(Path(design_path).rglob("*.def"))
            if def_files:
                logger.info(f"找到{len(def_files)}个DEF文件")
                for def_file in def_files:
                    try:
                        logger.info(f"正在解析DEF文件: {def_file}")
                        result = parse_def_file(str(def_file))
                        save_parsed_result(design_output_dir, def_file.stem, "def", result)
                    except Exception as e:
                        logger.error(f"解析DEF文件{def_file}时出错: {str(e)}")
            
            # 解析LEF文件
            lef_files = list(Path(design_path).rglob("*.lef"))
            if lef_files:
                logger.info(f"找到{len(lef_files)}个LEF文件")
                for lef_file in lef_files:
                    try:
                        logger.info(f"正在解析LEF文件: {lef_file}")
                        result = parse_lef_file(str(lef_file))
                        save_parsed_result(design_output_dir, lef_file.stem, "lef", result)
                    except Exception as e:
                        logger.error(f"解析LEF文件{lef_file}时出错: {str(e)}")
            
            logger.info(f"完成设计 {design_name} 的解析")
            
    except Exception as e:
        logger.error(f"解析设计文件时出错: {str(e)}")
        raise

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='解析设计文件')
    parser.add_argument('--design-dir', required=True, help='设计文件目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    try:
        parse_design_files(args.design_dir, args.output_dir)
    except KeyboardInterrupt:
        logger.info("用户中断解析过程")
    except Exception as e:
        logger.error(f"解析过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 