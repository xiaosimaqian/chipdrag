#!/usr/bin/env python3
"""
自动修正DEF文件中所有PLACED坐标到site grid（200 DBU）
"""
import re
from pathlib import Path

def align_to_grid(val, grid=200):
    return int(round(val / grid) * grid)

def fix_def_coordinates(def_path, grid=200):
    with open(def_path, 'r') as f:
        content = f.read()
    
    # 匹配所有PLACED ( x y )
    def repl(match):
        x = int(match.group(1))
        y = int(match.group(2))
        x_new = align_to_grid(x, grid)
        y_new = align_to_grid(y, grid)
        if (x, y) != (x_new, y_new):
            print(f"修正: ({x}, {y}) -> ({x_new}, {y_new})")
        return f"PLACED ( {x_new} {y_new} )"
    
    new_content, n = re.subn(r'PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)', repl, content)
    
    if n > 0:
        backup = str(def_path) + '.bak_align'
        with open(backup, 'w') as f:
            f.write(content)
        with open(def_path, 'w') as f:
            f.write(new_content)
        print(f"✅ 已修正所有PLACED坐标，原文件备份为: {backup}")
    else:
        print("未找到PLACED坐标，无需修改。")

def main():
    def_file = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/mgc_des_perf_1_place.def"
    fix_def_coordinates(def_file)

if __name__ == "__main__":
    main() 