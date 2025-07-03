#!/usr/bin/env python3
"""
修复DEF文件选择问题
确保实验使用floorplan.def而不是placement_result.def
"""

import re
from pathlib import Path

def fix_def_file_selection():
    """修复paper_hpwl_comparison_experiment_fixed.py中的DEF文件选择逻辑"""
    
    file_path = Path("paper_hpwl_comparison_experiment_fixed.py")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换DEF文件选择逻辑
    old_pattern = r'# 查找设计文件\s*\n\s*def_file = next\(design_dir\.glob\("\*\.def"\), None\)\s*\n\s*lef_file = next\(design_dir\.glob\("\*\.lef"\), None\)\s*\n\s*\n\s*if not def_file or not lef_file:\s*\n\s*logger\.error\("缺少必要的设计文件"\)\s*\n\s*logger\.error\(f"设计目录: \{design_dir\}"\)\s*\n\s*logger\.error\(f"DEF文件: \{def_file\}"\)\s*\n\s*logger\.error\(f"LEF文件: \{lef_file\}"\)\s*\n\s*return False'
    
    new_code = '''# 查找设计文件 - 严格要求使用floorplan.def作为初始布局
            def_file = None
            for df in design_dir.glob("*.def"):
                if 'floorplan' in df.name.lower():
                    def_file = df
                    break
            
            if def_file is None:
                logger.error("未找到floorplan.def文件，论文实验要求使用原始布局文件")
                logger.error(f"设计目录: {design_dir}")
                available_def_files = list(design_dir.glob("*.def"))
                logger.error(f"可用DEF文件: {[f.name for f in available_def_files]}")
                return False
            
            lef_file = next(design_dir.glob("*.lef"), None)
            
            if not lef_file:
                logger.error("缺少必要的LEF文件")
                logger.error(f"设计目录: {design_dir}")
                return False'''
    
    # 尝试简单替换
    if 'def_file = next(design_dir.glob("*.def"), None)' in content:
        print("找到目标代码，开始替换...")
        
        # 替换单行
        content = content.replace(
            'def_file = next(design_dir.glob("*.def"), None)',
            '''def_file = None
            for df in design_dir.glob("*.def"):
                if 'floorplan' in df.name.lower():
                    def_file = df
                    break
            
            if def_file is None:
                logger.error("未找到floorplan.def文件，论文实验要求使用原始布局文件")
                logger.error(f"设计目录: {design_dir}")
                available_def_files = list(design_dir.glob("*.def"))
                logger.error(f"可用DEF文件: {[f.name for f in available_def_files]}")
                return False
                
            # 重新设置def_file为找到的floorplan.def
            def_file'''
        )
        
        # 备份原文件
        backup_path = file_path.with_suffix('.py.backup_def_fix')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"原文件已备份到: {backup_path}")
        
        # 写入修复后的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ DEF文件选择逻辑已修复")
        print("现在实验将使用floorplan.def而不是placement_result.def")
        return True
    else:
        print("❌ 未找到目标代码")
        return False

if __name__ == "__main__":
    fix_def_file_selection() 