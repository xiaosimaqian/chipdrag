import os
import shutil
from pathlib import Path

def restore_original_tech_lef():
    # 原始数据目录
    original_dir = Path("dataset/ispd_2015_contest_benchmark")
    # 目标数据目录
    target_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    if not original_dir.exists():
        print(f"原始数据目录不存在: {original_dir}")
        return
    
    if not target_dir.exists():
        print(f"目标数据目录不存在: {target_dir}")
        return
    
    print("开始恢复原始tech.lef文件...")
    
    # 遍历所有设计目录
    for design_dir in original_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        design_name = design_dir.name
        original_tech_lef = design_dir / "tech.lef"
        target_tech_lef = target_dir / design_name / "tech.lef"
        
        if not original_tech_lef.exists():
            print(f"跳过 {design_name}: 原始tech.lef不存在")
            continue
            
        if not target_tech_lef.parent.exists():
            print(f"跳过 {design_name}: 目标目录不存在")
            continue
            
        try:
            # 备份当前文件（如果存在）
            if target_tech_lef.exists():
                backup_path = target_tech_lef.with_suffix('.lef.backup')
                shutil.copy2(target_tech_lef, backup_path)
                print(f"  {design_name}: 已备份当前tech.lef")
            
            # 复制原始文件
            shutil.copy2(original_tech_lef, target_tech_lef)
            print(f"  ✓ {design_name}: 已恢复原始tech.lef")
            
        except Exception as e:
            print(f"  ✗ {design_name}: 恢复失败 - {e}")
    
    print("原始tech.lef恢复完成！")

if __name__ == "__main__":
    restore_original_tech_lef() 