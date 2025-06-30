#!/usr/bin/env python3
"""
修复所有设计的PINS段，恢复几何信息以便HPWL提取
"""

import os
import re
import glob
from pathlib import Path

def extract_pins_section(def_file):
    """从DEF文件中提取PINS段"""
    with open(def_file, 'r') as f:
        content = f.read()
    
    # 查找PINS段
    pins_match = re.search(r'(PINS \d+ ;.*?END PINS)', content, re.DOTALL)
    if pins_match:
        return pins_match.group(1)
    return None

def replace_pins_section(def_file, new_pins_section):
    """替换DEF文件中的PINS段"""
    with open(def_file, 'r') as f:
        content = f.read()
    
    # 替换PINS段
    new_content = re.sub(r'PINS \d+ ;.*?END PINS', new_pins_section, content, flags=re.DOTALL)
    
    with open(def_file, 'w') as f:
        f.write(new_content)

def fix_design_pins(design_dir):
    """修复单个设计的PINS段"""
    design_path = Path(design_dir)
    floorplan_def = design_path / "floorplan.def"
    
    if not floorplan_def.exists():
        print(f"❌ {design_dir}: floorplan.def不存在")
        return False
    
    # 提取原始PINS段
    original_pins = extract_pins_section(floorplan_def)
    if not original_pins:
        print(f"❌ {design_dir}: 无法从floorplan.def提取PINS段")
        return False
    
    # 修复所有迭代DEF文件
    fixed_count = 0
    iteration_defs = list(design_path.glob("output/iterations/iteration_*_rl_training.def"))
    iteration_defs.append(design_path / "output/final_layout.def")
    
    for def_file in iteration_defs:
        if def_file.exists():
            try:
                replace_pins_section(def_file, original_pins)
                fixed_count += 1
            except Exception as e:
                print(f"❌ {design_dir}: 修复{def_file.name}失败: {e}")
    
    if fixed_count > 0:
        print(f"✅ {design_dir}: 修复了{fixed_count}个DEF文件")
        return True
    else:
        print(f"❌ {design_dir}: 没有找到需要修复的DEF文件")
        return False

def main():
    """主函数"""
    # 查找所有ISPD设计目录
    ispd_dir = Path("data/designs/ispd_2015_contest_benchmark")
    if not ispd_dir.exists():
        print("❌ ISPD基准测试目录不存在")
        return
    
    design_dirs = [d for d in ispd_dir.iterdir() if d.is_dir() and d.name.startswith("mgc_")]
    
    print(f"🔧 开始修复{len(design_dirs)}个设计的PINS段...")
    
    success_count = 0
    for design_dir in design_dirs:
        if fix_design_pins(design_dir):
            success_count += 1
    
    print(f"\n📊 修复完成: {success_count}/{len(design_dirs)}个设计成功")

if __name__ == "__main__":
    main() 