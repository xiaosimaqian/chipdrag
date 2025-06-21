#!/usr/bin/env python3
"""
检查DEF文件中单元坐标是否对齐site grid
"""
import re

def check_coordinates(def_file):
    """检查DEF文件中的坐标对齐情况"""
    
    with open(def_file, 'r') as f:
        content = f.read()
    
    # 查找所有PLACED坐标
    pattern = r'PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)'
    matches = re.findall(pattern, content)
    
    print(f"找到 {len(matches)} 个PLACED坐标")
    
    # 检查前20个坐标
    print("\n前20个坐标对齐检查（site grid = 200 DBU）:")
    print("坐标\t\tX/200\tY/200\tX对齐\tY对齐")
    print("-" * 50)
    
    misaligned_count = 0
    
    for i, (x, y) in enumerate(matches[:20]):
        x_val = int(x)
        y_val = int(y)
        x_align = x_val % 200 == 0
        y_align = y_val % 200 == 0
        
        if not x_align or not y_align:
            misaligned_count += 1
            status = "❌"
        else:
            status = "✅"
            
        print(f"({x_val}, {y_val})\t{x_val//200}\t{y_val//200}\t{x_align}\t{y_align}\t{status}")
    
    # 检查所有坐标
    total_misaligned = 0
    for x, y in matches:
        x_val = int(x)
        y_val = int(y)
        if x_val % 200 != 0 or y_val % 200 != 0:
            total_misaligned += 1
    
    print(f"\n总结:")
    print(f"总坐标数: {len(matches)}")
    print(f"不对齐坐标数: {total_misaligned}")
    print(f"对齐率: {(len(matches) - total_misaligned) / len(matches) * 100:.2f}%")
    
    if total_misaligned > 0:
        print(f"\n❌ 发现 {total_misaligned} 个坐标未对齐site grid!")
        return False
    else:
        print(f"\n✅ 所有坐标都对齐site grid!")
        return True

if __name__ == "__main__":
    def_file = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/mgc_des_perf_1_place.def"
    check_coordinates(def_file) 