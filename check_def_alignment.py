#!/usr/bin/env python3
"""
检查 DEF 文件中单元坐标是否对齐到 site grid (DBU单位)
Site size: 0.20 × 2.00 um, DBU=1000, site grid: 200 × 2000
"""

import re
from pathlib import Path

def check_def_alignment(def_file_path, site_x=200, site_y=2000):
    """检查 DEF 文件中单元坐标是否对齐到 site grid (DBU单位)"""
    
    print(f"检查 DEF 文件: {def_file_path}")
    print(f"Site grid (DBU): X={site_x}, Y={site_y}")
    print("-" * 60)
    
    # 读取 DEF 文件
    with open(def_file_path, 'r') as f:
        content = f.read()
    
    # 查找 COMPONENTS 区块
    components_match = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END COMPONENTS', content, re.DOTALL)
    if not components_match:
        print("❌ 未找到 COMPONENTS 区块")
        return
    
    components_count = int(components_match.group(1))
    components_content = components_match.group(2)
    
    print(f"找到 {components_count} 个单元")
    
    # 解析单元坐标
    # 匹配格式: - instance_name cell_type + PLACED (x y) N ;
    component_pattern = r'-\s+(\w+)\s+(\w+)\s+\+\s+PLACED\s+\(([^)]+)\)\s+(\w+)\s*;'
    components = re.findall(component_pattern, components_content)
    
    print(f"解析到 {len(components)} 个单元实例")
    
    # 检查坐标对齐
    misaligned_count = 0
    misaligned_components = []
    
    for i, (instance, cell_type, coords, orientation) in enumerate(components[:100]):  # 只检查前100个
        try:
            # 解析坐标
            coord_match = re.search(r'(\d+)(?:\.\d+)?\s+(\d+)(?:\.\d+)?', coords)
            if coord_match:
                x = int(float(coord_match.group(1)))
                y = int(float(coord_match.group(2)))
                
                # 检查是否对齐到 site grid
                x_aligned = (x % site_x) == 0
                y_aligned = (y % site_y) == 0
                
                if not (x_aligned and y_aligned):
                    misaligned_count += 1
                    misaligned_components.append({
                        'instance': instance,
                        'cell_type': cell_type,
                        'x': x,
                        'y': y,
                        'x_remainder': x % site_x,
                        'y_remainder': y % site_y
                    })
                    
                    if misaligned_count <= 10:  # 只显示前10个不对齐的
                        print(f"❌ {instance} ({cell_type}): ({x}, {y}) - X余数:{x%site_x}, Y余数:{y%site_y}")
        
        except Exception as e:
            print(f"解析坐标失败: {coords} - {e}")
    
    # 统计结果
    aligned_count = len(components[:100]) - misaligned_count
    print("-" * 60)
    print(f"检查结果 (前100个单元):")
    print(f"✅ 对齐到 site grid: {aligned_count}")
    print(f"❌ 未对齐到 site grid: {misaligned_count}")
    print(f"对齐率: {aligned_count/len(components[:100])*100:.1f}%")
    
    if misaligned_count > 0:
        print(f"\n前10个未对齐的单元:")
        for comp in misaligned_components[:10]:
            print(f"  {comp['instance']}: ({comp['x']}, {comp['y']})")
    
    return misaligned_count > 0

if __name__ == "__main__":
    def_file = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/mgc_des_perf_1_place.def")
    
    if def_file.exists():
        has_misalignment = check_def_alignment(def_file)
        if has_misalignment:
            print("\n🔧 建议:")
            print("1. 检查 DEF/LEF 单位定义和 site size 是否一致")
            print("2. 或者联系布局工具导出对齐的 DEF 文件")
        else:
            print("\n✅ 所有检查的单元都对齐到 site grid (DBU)")
    else:
        print(f"❌ DEF 文件不存在: {def_file}") 