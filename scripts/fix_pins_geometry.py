import os
import re
from pathlib import Path
import shutil

def extract_pins_from_def(def_path):
    """从DEF文件中提取完整的PINS段"""
    with open(def_path, 'r') as f:
        content = f.read()
    
    # 查找PINS段
    pins_pattern = r'(PINS \d+ ;\n(?:.*?\n)*?END PINS)'
    match = re.search(pins_pattern, content, re.DOTALL)
    
    if match:
        return match.group(1)
    else:
        print(f"警告: 在 {def_path} 中未找到PINS段")
        return None

def replace_pins_in_def(target_def_path, pins_content):
    """替换DEF文件中的PINS段"""
    # 备份原文件
    backup_path = target_def_path + '.pins_backup'
    shutil.copy2(target_def_path, backup_path)
    print(f"已备份: {backup_path}")
    
    with open(target_def_path, 'r') as f:
        content = f.read()
    
    # 替换PINS段
    pins_pattern = r'PINS \d+ ;\n(?:.*?\n)*?END PINS'
    new_content = re.sub(pins_pattern, pins_content, content, flags=re.DOTALL)
    
    with open(target_def_path, 'w') as f:
        f.write(new_content)
    
    print(f"已替换PINS段: {target_def_path}")

def fix_pins_geometry():
    """批量修复所有设计的PINS几何信息"""
    base_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        return
    
    print("开始修复PINS几何信息...")
    
    for design_dir in base_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        design_name = design_dir.name
        floorplan_def = design_dir / "floorplan.def"
        placement_def = design_dir / "placement_result.def"
        
        if not floorplan_def.exists():
            print(f"跳过 {design_name}: floorplan.def不存在")
            continue
            
        if not placement_def.exists():
            print(f"跳过 {design_name}: placement_result.def不存在")
            continue
        
        print(f"\n处理设计: {design_name}")
        
        # 提取floorplan.def的PINS段
        pins_content = extract_pins_from_def(floorplan_def)
        if pins_content is None:
            print(f"  跳过: 无法提取PINS段")
            continue
        
        # 替换placement_result.def的PINS段
        replace_pins_in_def(placement_def, pins_content)
        print(f"  ✅ 已修复PINS几何信息")

if __name__ == "__main__":
    fix_pins_geometry()
    print("\n修复完成！现在可以重新运行HPWL提取脚本。") 