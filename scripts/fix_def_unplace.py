#!/usr/bin/env python3
"""
批量修改ISPD 2015基准电路的floorplan.def文件
将所有单元状态从PLACED改为UNPLACED，确保OpenROAD能重新布局
"""

import os
import re
import argparse
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_def_file(def_file_path: str, backup: bool = True) -> bool:
    """修改DEF文件，将单元状态从PLACED改为UNPLACED
    
    Args:
        def_file_path: DEF文件路径
        backup: 是否备份原文件
    
    Returns:
        bool: 是否成功修改
    """
    def_file = Path(def_file_path)
    if not def_file.exists():
        logger.error(f"DEF文件不存在: {def_file_path}")
        return False
    
    # 备份原文件
    if backup:
        backup_file = def_file.with_suffix('.def.bak')
        try:
            import shutil
            shutil.copy2(def_file, backup_file)
            logger.info(f"已备份原文件: {backup_file}")
        except Exception as e:
            logger.warning(f"备份失败: {e}")
    
    # 读取文件内容
    try:
        with open(def_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return False
    
    # 统计修改前的PLACED单元数量
    placed_count_before = len(re.findall(r'\s+PLACED\s+', content))
    logger.info(f"修改前PLACED单元数量: {placed_count_before}")
    
    # 替换PLACED为UNPLACED
    # 注意：只替换COMPONENTS段中的PLACED，避免影响其他部分
    lines = content.split('\n')
    in_components = False
    modified_lines = []
    placed_count_after = 0
    
    for line in lines:
        if line.strip().startswith('COMPONENTS'):
            in_components = True
            modified_lines.append(line)
        elif line.strip().startswith('END COMPONENTS'):
            in_components = False
            modified_lines.append(line)
        elif in_components and 'PLACED' in line:
            # 在COMPONENTS段中替换PLACED为UNPLACED
            new_line = line.replace(' PLACED ', ' UNPLACED ')
            modified_lines.append(new_line)
            placed_count_after += 1
        else:
            modified_lines.append(line)
    
    # 写回文件
    try:
        with open(def_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines))
        logger.info(f"已修改: {def_file_path}")
        logger.info(f"修改后UNPLACED单元数量: {placed_count_after}")
        return True
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        return False

def batch_fix_def_files(benchmark_dir: str, backup: bool = True) -> dict:
    """批量修改所有ISPD设计的DEF文件
    
    Args:
        benchmark_dir: 基准测试目录
        backup: 是否备份原文件
    
    Returns:
        dict: 修改结果统计
    """
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        logger.error(f"基准测试目录不存在: {benchmark_dir}")
        return {"success": False, "error": "目录不存在"}
    
    # 查找所有设计目录
    design_dirs = [d for d in benchmark_path.iterdir() if d.is_dir()]
    logger.info(f"发现 {len(design_dirs)} 个设计目录")
    
    results = {
        "total": len(design_dirs),
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    for design_dir in design_dirs:
        design_name = design_dir.name
        def_file = design_dir / "floorplan.def"
        
        if not def_file.exists():
            logger.warning(f"设计 {design_name} 缺少floorplan.def文件")
            results["failed"] += 1
            results["details"].append({
                "design": design_name,
                "status": "failed",
                "reason": "缺少floorplan.def文件"
            })
            continue
        
        logger.info(f"处理设计: {design_name}")
        if fix_def_file(str(def_file), backup):
            results["success"] += 1
            results["details"].append({
                "design": design_name,
                "status": "success",
                "def_file": str(def_file)
            })
        else:
            results["failed"] += 1
            results["details"].append({
                "design": design_name,
                "status": "failed",
                "reason": "修改失败"
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="批量修改ISPD DEF文件，将单元状态改为UNPLACED")
    parser.add_argument("--benchmark-dir", 
                       default="data/designs/ispd_2015_contest_benchmark",
                       help="ISPD基准测试目录")
    parser.add_argument("--no-backup", action="store_true",
                       help="不备份原文件")
    parser.add_argument("--design", type=str,
                       help="只处理指定设计（可选）")
    
    args = parser.parse_args()
    
    if args.design:
        # 只处理指定设计
        design_dir = Path(args.benchmark_dir) / args.design
        if not design_dir.exists():
            logger.error(f"设计目录不存在: {design_dir}")
            return
        
        def_file = design_dir / "floorplan.def"
        if not def_file.exists():
            logger.error(f"DEF文件不存在: {def_file}")
            return
        
        logger.info(f"处理单个设计: {args.design}")
        if fix_def_file(str(def_file), not args.no_backup):
            logger.info(f"设计 {args.design} 修改成功")
        else:
            logger.error(f"设计 {args.design} 修改失败")
    else:
        # 批量处理所有设计
        logger.info("开始批量修改DEF文件...")
        results = batch_fix_def_files(args.benchmark_dir, not args.no_backup)
        
        # 输出结果
        print("\n" + "="*60)
        print("批量修改DEF文件结果")
        print("="*60)
        print(f"总设计数: {results['total']}")
        print(f"成功: {results['success']}")
        print(f"失败: {results['failed']}")
        print(f"成功率: {results['success']/results['total']*100:.1f}%")
        
        if results['failed'] > 0:
            print("\n失败的设计:")
            for detail in results['details']:
                if detail['status'] == 'failed':
                    print(f"  - {detail['design']}: {detail.get('reason', '未知错误')}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main() 