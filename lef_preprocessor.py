#!/usr/bin/env python3
"""
LEF文件预处理脚本
自动将SITE core定义插入tech.lef最前面，解决OpenROAD兼容性问题
"""

import os
import sys
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LEFPreprocessor:
    """LEF文件预处理器"""
    
    def __init__(self):
        # 标准的SITE core定义
        self.site_core_definition = """SITE core
  SIZE 0.20 BY 2.00 ;
  CLASS CORE ;
  SYMMETRY Y  ;
END core

"""
        
        # 需要插入SITE定义的位置标识
        self.insertion_markers = [
            "VERSION",
            "NAMESCASESENSITIVE", 
            "BUSBITCHARS",
            "DIVIDERCHAR",
            "UNITS"
        ]
    
    def find_insertion_position(self, lines: List[str]) -> int:
        """
        找到插入SITE定义的最佳位置
        
        Args:
            lines: LEF文件的行列表
            
        Returns:
            插入位置（行号）
        """
        # 查找第一个关键标记的位置
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for marker in self.insertion_markers:
                if line_stripped.startswith(marker):
                    logger.info(f"找到插入位置: 第{i+1}行 ({marker})")
                    return i
        
        # 如果没找到标记，插入到文件开头
        logger.warning("未找到标准标记，插入到文件开头")
        return 0
    
    def extract_existing_site_definition(self, lines: List[str]) -> Optional[str]:
        """
        提取已存在的SITE core定义
        
        Args:
            lines: LEF文件的行列表
            
        Returns:
            SITE core定义字符串，如果不存在则返回None
        """
        site_start = None
        site_end = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith("SITE core"):
                site_start = i
            elif site_start is not None and line.strip().startswith("END core"):
                site_end = i + 1
                break
        
        if site_start is not None and site_end is not None:
            site_definition = "".join(lines[site_start:site_end])
            logger.info(f"找到已存在的SITE core定义 (第{site_start+1}-{site_end}行)")
            return site_definition
        
        return None
    
    def remove_existing_site_definition(self, lines: List[str]) -> List[str]:
        """
        移除已存在的SITE core定义
        
        Args:
            lines: LEF文件的行列表
            
        Returns:
            移除SITE定义后的行列表
        """
        site_start = None
        site_end = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith("SITE core"):
                site_start = i
            elif site_start is not None and line.strip().startswith("END core"):
                site_end = i + 1
                break
        
        if site_start is not None and site_end is not None:
            logger.info(f"移除已存在的SITE core定义 (第{site_start+1}-{site_end}行)")
            return lines[:site_start] + lines[site_end:]
        
        return lines
    
    def preprocess_tech_lef(self, tech_lef_path: str, backup: bool = True) -> bool:
        """
        预处理tech.lef文件
        
        Args:
            tech_lef_path: tech.lef文件路径
            backup: 是否创建备份文件
            
        Returns:
            是否成功处理
        """
        tech_lef_path = Path(tech_lef_path)
        
        if not tech_lef_path.exists():
            logger.error(f"文件不存在: {tech_lef_path}")
            return False
        
        logger.info(f"开始预处理: {tech_lef_path}")
        
        try:
            # 读取原文件
            with open(tech_lef_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 检查是否已经有SITE core定义
            existing_site = self.extract_existing_site_definition(lines)
            
            if existing_site:
                logger.info("检测到已存在的SITE core定义")
                # 检查是否在文件开头
                if existing_site in "".join(lines[:10]):
                    logger.info("SITE core定义已在文件开头，无需处理")
                    return True
                else:
                    logger.info("SITE core定义不在文件开头，需要移动")
            
            # 创建备份
            if backup:
                backup_path = tech_lef_path.with_suffix('.lef.backup')
                shutil.copy2(tech_lef_path, backup_path)
                logger.info(f"创建备份文件: {backup_path}")
            
            # 移除已存在的SITE定义
            lines = self.remove_existing_site_definition(lines)
            
            # 找到插入位置
            insert_pos = self.find_insertion_position(lines)
            
            # 插入SITE定义
            lines.insert(insert_pos, self.site_core_definition)
            
            # 写入处理后的文件
            with open(tech_lef_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            logger.info(f"成功处理: {tech_lef_path}")
            logger.info(f"SITE core定义已插入到第{insert_pos+1}行")
            
            return True
            
        except Exception as e:
            logger.error(f"处理文件时发生错误: {e}")
            return False
    
    def preprocess_design_directory(self, design_dir: str, backup: bool = True) -> bool:
        """
        预处理设计目录中的tech.lef文件
        
        Args:
            design_dir: 设计目录路径
            backup: 是否创建备份文件
            
        Returns:
            是否成功处理
        """
        design_dir = Path(design_dir)
        tech_lef_path = design_dir / "tech.lef"
        
        if not tech_lef_path.exists():
            logger.error(f"tech.lef文件不存在: {tech_lef_path}")
            return False
        
        return self.preprocess_tech_lef(str(tech_lef_path), backup)
    
    def preprocess_ispd_benchmarks(self, benchmark_root: str, backup: bool = True) -> Tuple[int, int]:
        """
        预处理所有ISPD基准测试
        
        Args:
            benchmark_root: ISPD基准测试根目录
            backup: 是否创建备份文件
            
        Returns:
            (成功数量, 总数量)的元组
        """
        benchmark_root = Path(benchmark_root)
        
        if not benchmark_root.exists():
            logger.error(f"基准测试目录不存在: {benchmark_root}")
            return 0, 0
        
        success_count = 0
        total_count = 0
        
        # 遍历所有设计目录
        for design_dir in benchmark_root.iterdir():
            if design_dir.is_dir() and (design_dir / "tech.lef").exists():
                total_count += 1
                logger.info(f"处理设计: {design_dir.name}")
                
                if self.preprocess_design_directory(str(design_dir), backup):
                    success_count += 1
                    logger.info(f"✅ {design_dir.name}: 成功")
                else:
                    logger.error(f"❌ {design_dir.name}: 失败")
        
        logger.info(f"预处理完成: {success_count}/{total_count} 个设计成功")
        return success_count, total_count
    
    def validate_tech_lef(self, tech_lef_path: str) -> bool:
        """
        验证tech.lef文件是否正确处理
        
        Args:
            tech_lef_path: tech.lef文件路径
            
        Returns:
            是否验证通过
        """
        tech_lef_path = Path(tech_lef_path)
        
        if not tech_lef_path.exists():
            logger.error(f"文件不存在: {tech_lef_path}")
            return False
        
        try:
            with open(tech_lef_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查SITE core定义是否在文件开头
            lines = content.split('\n')
            site_found = False
            
            for i, line in enumerate(lines[:20]):  # 只检查前20行
                if line.strip().startswith("SITE core"):
                    site_found = True
                    logger.info(f"✅ 验证通过: SITE core定义在第{i+1}行")
                    break
            
            if not site_found:
                logger.error("❌ 验证失败: 未找到SITE core定义")
                return False
            
            # 检查SITE定义是否完整
            if "SIZE 0.20 BY 2.00" in content and "CLASS CORE" in content:
                logger.info("✅ 验证通过: SITE core定义完整")
                return True
            else:
                logger.error("❌ 验证失败: SITE core定义不完整")
                return False
                
        except Exception as e:
            logger.error(f"验证时发生错误: {e}")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LEF文件预处理器")
    parser.add_argument("--design-dir", help="单个设计目录路径")
    parser.add_argument("--benchmark-root", default="data/designs/ispd_2015_contest_benchmark", 
                       help="ISPD基准测试根目录")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份文件")
    parser.add_argument("--validate", action="store_true", help="验证处理结果")
    
    args = parser.parse_args()
    
    preprocessor = LEFPreprocessor()
    
    if args.design_dir:
        # 处理单个设计
        logger.info(f"处理单个设计: {args.design_dir}")
        success = preprocessor.preprocess_design_directory(args.design_dir, not args.no_backup)
        
        if success and args.validate:
            preprocessor.validate_tech_lef(os.path.join(args.design_dir, "tech.lef"))
        
        if success:
            logger.info("✅ 单个设计处理成功")
        else:
            logger.error("❌ 单个设计处理失败")
            sys.exit(1)
    
    else:
        # 处理所有ISPD基准测试
        logger.info(f"处理ISPD基准测试: {args.benchmark_root}")
        success_count, total_count = preprocessor.preprocess_ispd_benchmarks(
            args.benchmark_root, not args.no_backup
        )
        
        if args.validate and success_count > 0:
            logger.info("开始验证处理结果...")
            validation_success = 0
            
            benchmark_root = Path(args.benchmark_root)
            for design_dir in benchmark_root.iterdir():
                if design_dir.is_dir() and (design_dir / "tech.lef").exists():
                    if preprocessor.validate_tech_lef(str(design_dir / "tech.lef")):
                        validation_success += 1
            
            logger.info(f"验证结果: {validation_success}/{success_count} 个文件验证通过")
        
        if success_count == total_count:
            logger.info("✅ 所有设计处理成功")
        else:
            logger.error(f"❌ 部分设计处理失败: {success_count}/{total_count}")
            sys.exit(1)

if __name__ == "__main__":
    main() 