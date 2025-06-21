#!/usr/bin/env python3
"""
修复 LEF/DEF 文件兼容性问题
1. 在 LEF 文件头部添加 SITE 定义
2. 降级 DEF 文件版本
3. 添加 ROW 定义
"""

import re
from pathlib import Path

def fix_lef_file(lef_file_path):
    """修复 LEF 文件，在头部添加 SITE 定义"""
    
    print(f"修复 LEF 文件: {lef_file_path}")
    
    # 读取 LEF 文件
    with open(lef_file_path, 'r') as f:
        content = f.read()
    
    # 检查是否已有 SITE 定义
    if re.search(r'SITE\s+core\s*\n\s+CLASS\s+CORE', content):
        print("✅ LEF 文件已有 SITE 定义")
        return False
    
    # 在 VERSION 后添加 SITE 定义
    site_definition = '''SITE core
  CLASS CORE ;
  SIZE 0.200 BY 2.000 ;
  SYMMETRY Y ;
END core

'''
    
    # 在 VERSION 行后插入 SITE 定义
    pattern = r'(VERSION\s+\d+\.\d+\s*;\s*\n)'
    replacement = r'\1' + site_definition
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        # 备份原文件
        backup_path = lef_file_path.with_suffix('.lef.backup')
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"✅ 原文件已备份到: {backup_path}")
        
        # 写入修复后的文件
        with open(lef_file_path, 'w') as f:
            f.write(new_content)
        print("✅ LEF 文件已修复，添加了 SITE 定义")
        return True
    else:
        print("❌ 未找到 VERSION 行，无法插入 SITE 定义")
        return False

def fix_def_file(def_file_path):
    """修复 DEF 文件，降级版本并添加 ROW 定义"""
    
    print(f"修复 DEF 文件: {def_file_path}")
    
    # 读取 DEF 文件
    with open(def_file_path, 'r') as f:
        content = f.read()
    
    # 降级版本到 5.6
    content = re.sub(r'VERSION\s+5\.8\s*;', 'VERSION 5.6 ;', content)
    
    # 查找 COMPONENTS 区块
    components_match = re.search(r'(COMPONENTS\s+\d+\s*;.*?END COMPONENTS)', content, re.DOTALL)
    if not components_match:
        print("❌ 未找到 COMPONENTS 区块")
        return False
    
    # 在 COMPONENTS 前添加 ROW 定义
    row_definition = '''
ROWS 1 N ;
  - core_row core 0.000 0.000 N DO 2225 BY 1 STEP 0.200 2.000 ;
END ROWS

'''
    
    # 在 COMPONENTS 前插入 ROW 定义
    pattern = r'(COMPONENTS\s+\d+\s*;.*?END COMPONENTS)'
    replacement = row_definition + r'\1'
    
    new_content = re.sub(pattern, replacement, content)
    
    # 备份原文件
    backup_path = def_file_path.with_suffix('.def.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ 原文件已备份到: {backup_path}")
    
    # 写入修复后的文件
    with open(def_file_path, 'w') as f:
        f.write(new_content)
    print("✅ DEF 文件已修复，降级版本并添加了 ROW 定义")
    return True

def main():
    """主函数"""
    
    # 文件路径
    lef_file = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/cells.lef")
    def_file = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/mgc_des_perf_1_place.def")
    
    print("=== 修复 LEF/DEF 文件兼容性问题 ===")
    
    # 修复 LEF 文件
    if lef_file.exists():
        lef_fixed = fix_lef_file(lef_file)
    else:
        print(f"❌ LEF 文件不存在: {lef_file}")
        lef_fixed = False
    
    # 修复 DEF 文件
    if def_file.exists():
        def_fixed = fix_def_file(def_file)
    else:
        print(f"❌ DEF 文件不存在: {def_file}")
        def_fixed = False
    
    print("\n=== 修复结果 ===")
    if lef_fixed or def_fixed:
        print("✅ 文件已修复，请重新运行 OpenROAD 测试")
        print("建议运行: python real_openroad_interface.py")
    else:
        print("❌ 文件修复失败或无修改")

if __name__ == "__main__":
    main() 