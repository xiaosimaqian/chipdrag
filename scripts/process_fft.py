import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.parsers.verilog_parser import parse_verilog_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(directory: str) -> None:
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def process_fft_design(design_file: str, output_dir: str) -> None:
    """处理FFT设计文件并生成JSON输出
    
    Args:
        design_file: FFT设计文件路径
        output_dir: 输出目录
    """
    try:
        # 确保输出目录存在
        ensure_dir(output_dir)
        
        # 解析Verilog文件
        logger.info(f"开始解析FFT设计文件: {design_file}")
        verilog_data = parse_verilog_file(design_file)
        
        # 保存解析结果
        output_file = os.path.join(output_dir, "fft_design.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(verilog_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"FFT设计解析完成，结果已保存到: {output_file}")
        
        # 打印一些统计信息
        if 'modules' in verilog_data:
            logger.info(f"共解析到 {len(verilog_data['modules'])} 个模块")
            for module in verilog_data['modules']:
                logger.info(f"模块 {module['name']}:")
                logger.info(f"  - 端口数量: {len(module.get('ports', []))}")
                logger.info(f"  - 实例数量: {len(module.get('instances', []))}")
                logger.info(f"  - 网络数量: {len(module.get('nets', []))}")
        
    except Exception as e:
        logger.error(f"处理FFT设计文件时出错: {str(e)}")
        raise

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python process_fft.py <设计文件> <输出目录>")
        sys.exit(1)
        
    design_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_fft_design(design_file, output_dir)

if __name__ == "__main__":
    main() 