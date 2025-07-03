#!/usr/bin/env python3
"""
修复ISPD 2015设计文件
从原始官方数据集复制正确的文件到工作目录
"""

import os
import shutil
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_ispd_files():
    """修复ISPD文件"""
    
    # 路径配置
    original_dataset_dir = Path("dataset/ispd_2015_contest_benchmark")
    working_data_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    if not original_dataset_dir.exists():
        logger.error(f"原始数据集目录不存在: {original_dataset_dir}")
        return False
    
    if not working_data_dir.exists():
        logger.error(f"工作数据目录不存在: {working_data_dir}")
        return False
    
    # 需要修复的设计列表 - 完整的ISPD 2015数据集
    designs = [
        'mgc_des_perf_1', 'mgc_des_perf_a', 'mgc_des_perf_b',
        'mgc_edit_dist_a',
        'mgc_fft_1', 'mgc_fft_2', 'mgc_fft_a', 'mgc_fft_b',
        'mgc_matrix_mult_1', 'mgc_matrix_mult_a', 'mgc_matrix_mult_b',
        'mgc_pci_bridge32_a', 'mgc_pci_bridge32_b',
        'mgc_superblue11_a', 'mgc_superblue12', 'mgc_superblue16_a'
    ]
    
    # 需要复制的官方文件
    official_files = ['floorplan.def', 'cells.lef', 'tech.lef', 'design.v', 'placement.constraints']
    
    success_count = 0
    
    for design in designs:
        logger.info(f"处理设计: {design}")
        
        original_dir = original_dataset_dir / design
        working_dir = working_data_dir / design
        
        if not original_dir.exists():
            logger.warning(f"原始设计目录不存在: {original_dir}")
            continue
        
        if not working_dir.exists():
            logger.warning(f"工作设计目录不存在: {working_dir}")
            continue
        
        # 复制官方文件
        for file_name in official_files:
            original_file = original_dir / file_name
            working_file = working_dir / file_name
            
            if original_file.exists():
                # 备份现有文件（如果存在）
                if working_file.exists():
                    backup_file = working_dir / f"{file_name}.backup"
                    shutil.copy2(working_file, backup_file)
                    logger.info(f"  备份文件: {file_name} -> {file_name}.backup")
                
                # 复制官方文件
                shutil.copy2(original_file, working_file)
                logger.info(f"  ✅ 复制文件: {file_name}")
            else:
                logger.warning(f"  ❌ 原始文件不存在: {file_name}")
        
        success_count += 1
        logger.info(f"  设计 {design} 处理完成")
    
    logger.info(f"修复完成: {success_count}/{len(designs)} 个设计处理成功")
    return success_count == len(designs)

def verify_files():
    """验证文件是否正确"""
    working_data_dir = Path("data/designs/ispd_2015_contest_benchmark")
    # 完整的ISPD 2015数据集设计列表
    designs = [
        'mgc_des_perf_1', 'mgc_des_perf_a', 'mgc_des_perf_b',
        'mgc_edit_dist_a',
        'mgc_fft_1', 'mgc_fft_2', 'mgc_fft_a', 'mgc_fft_b',
        'mgc_matrix_mult_1', 'mgc_matrix_mult_a', 'mgc_matrix_mult_b',
        'mgc_pci_bridge32_a', 'mgc_pci_bridge32_b',
        'mgc_superblue11_a', 'mgc_superblue12', 'mgc_superblue16_a'
    ]
    
    logger.info("验证文件正确性...")
    
    for design in designs:
        working_dir = working_data_dir / design
        floorplan_def = working_dir / "floorplan.def"
        
        if floorplan_def.exists():
            # 读取DEF文件的设计名称
            with open(floorplan_def, 'r') as f:
                content = f.read(1000)  # 只读前1000字符
                if "DESIGN " in content:
                    design_line = [line for line in content.split('\n') if 'DESIGN ' in line][0]
                    design_name = design_line.split()[1].rstrip(' ;')
                    
                    # 验证设计名称是否匹配 - 扩展的映射表
                    expected_names = {
                        'mgc_des_perf_1': 'des_perf',
                        'mgc_des_perf_a': 'des_perf',
                        'mgc_des_perf_b': 'des_perf',
                        'mgc_edit_dist_a': 'edit_dist',
                        'mgc_fft_1': 'fft',
                        'mgc_fft_2': 'fft',
                        'mgc_fft_a': 'fft',
                        'mgc_fft_b': 'fft',
                        'mgc_matrix_mult_1': 'matrix_mult',
                        'mgc_matrix_mult_a': 'matrix_mult',
                        'mgc_matrix_mult_b': 'matrix_mult',
                        'mgc_pci_bridge32_a': 'pci_bridge32',
                        'mgc_pci_bridge32_b': 'pci_bridge32',
                        'mgc_superblue11_a': 'superblue11',
                        'mgc_superblue12': 'superblue12',
                        'mgc_superblue16_a': 'superblue16'
                    }
                    
                    expected = expected_names.get(design, design)
                    if design_name == expected:
                        logger.info(f"  ✅ {design}: 设计名称正确 ({design_name})")
                    else:
                        logger.error(f"  ❌ {design}: 设计名称错误 (期望: {expected}, 实际: {design_name})")
                else:
                    logger.warning(f"  ⚠️ {design}: 无法找到设计名称")
        else:
            logger.error(f"  ❌ {design}: floorplan.def 文件不存在")

if __name__ == "__main__":
    logger.info("开始修复ISPD文件...")
    
    if fix_ispd_files():
        logger.info("✅ 所有文件修复完成")
        verify_files()
    else:
        logger.error("❌ 文件修复失败") 