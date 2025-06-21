#!/usr/bin/env python3
"""
手动修复 DEF 文件，在 COMPONENTS 前插入 ROW 定义
"""

def fix_def_file_manual():
    """手动修复 DEF 文件"""
    
    def_file = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/mgc_des_perf_1_place.def"
    
    print(f"修复 DEF 文件: {def_file}")
    
    # 读取文件
    with open(def_file, 'r') as f:
        lines = f.readlines()
    
    # 查找 COMPONENTS 行
    components_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith('COMPONENTS'):
            components_line = i
            break
    
    if components_line is None:
        print("❌ 未找到 COMPONENTS 行")
        return False
    
    # 在 COMPONENTS 前插入 ROW 定义
    row_definition = [
        '\n',
        'ROWS 1 N ;\n',
        '  - core_row core 0.000 0.000 N DO 2225 BY 1 STEP 0.200 2.000 ;\n',
        'END ROWS\n',
        '\n'
    ]
    
    # 插入 ROW 定义
    lines[components_line:components_line] = row_definition
    
    # 备份原文件
    backup_file = def_file + '.backup2'
    with open(backup_file, 'w') as f:
        f.writelines(lines[:components_line])
        f.writelines(lines[components_line + len(row_definition):])
    print(f"✅ 原文件已备份到: {backup_file}")
    
    # 写入修复后的文件
    with open(def_file, 'w') as f:
        f.writelines(lines)
    print("✅ DEF 文件已修复，添加了 ROW 定义")
    
    return True

if __name__ == "__main__":
    fix_def_file_manual() 