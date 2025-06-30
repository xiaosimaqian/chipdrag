#!/usr/bin/env python3
"""
从DEF文件中计算HPWL (Half-Perimeter Wire Length)
"""

import re
import sys
from pathlib import Path

def parse_def_file(def_file):
    """解析DEF文件，提取组件位置和网络连接"""
    components = {}  # {component_name: (x, y)}
    nets = {}        # {net_name: [component_pins]}
    
    with open(def_file, 'r') as f:
        content = f.read()
    
    # 解析COMPONENTS段
    components_match = re.search(r'COMPONENTS (\d+) ;(.*?)END COMPONENTS', content, re.DOTALL)
    if components_match:
        components_section = components_match.group(2)
        for line in components_section.strip().split('\n'):
            line = line.strip()
            if line and line.startswith('-'):
                # 格式: - component_name cell_name + PLACED ( x y ) N ;
                match = re.search(r'- (\S+) \S+ \+ PLACED \( (-?\d+) (-?\d+) \)', line)
                if match:
                    comp_name, x, y = match.groups()
                    components[comp_name] = (int(x), int(y))
    
    # 解析NETS段 - 修复版本
    nets_match = re.search(r'NETS (\d+) ;(.*?)END NETS', content, re.DOTALL)
    if nets_match:
        nets_section = nets_match.group(2)
        
        # 使用正则表达式直接匹配网络定义
        # 格式: - net_name ( comp1 pin1 ) ( comp2 pin2 ) ... + USE SIGNAL ;
        net_pattern = r'- (\S+)\s+((?:\([^)]+\)\s*)+)\s*\+ USE SIGNAL\s*;'
        
        for match in re.finditer(net_pattern, nets_section, re.DOTALL):
            net_name = match.group(1)
            connections = match.group(2)
            
            # 解析连接
            comp_names = []
            for conn_match in re.finditer(r'\( (\S+) \S+ \)', connections):
                comp_name = conn_match.group(1)
                if comp_name in components:
                    comp_names.append(comp_name)
            
            if len(comp_names) >= 2:
                nets[net_name] = comp_names
    
    return components, nets

def calculate_hpwl(components, nets):
    """计算总HPWL"""
    total_hpwl = 0
    valid_nets = 0
    total_nets = len(nets)
    
    print(f"Processing {total_nets} nets...")
    
    for net_name, net_components in nets.items():
        if len(net_components) < 2:
            continue
            
        # 获取网络中所有组件的坐标
        coords = [components[comp] for comp in net_components if comp in components]
        if len(coords) < 2:
            continue
            
        # 计算边界框
        x_coords = [x for x, y in coords]
        y_coords = [y for x, y in coords]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # HPWL = (max_x - min_x) + (max_y - min_y)
        hpwl = (max_x - min_x) + (max_y - min_y)
        total_hpwl += hpwl
        valid_nets += 1
        
        # 打印前几个网络的详细信息用于调试
        if valid_nets <= 5:
            print(f"Net {net_name}: {len(coords)} components, HPWL = {hpwl}")
    
    return total_hpwl, valid_nets, total_nets

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_hpwl.py <def_file>")
        sys.exit(1)
    
    def_file = sys.argv[1]
    if not Path(def_file).exists():
        print(f"Error: File {def_file} not found")
        sys.exit(1)
    
    try:
        components, nets = parse_def_file(def_file)
        hpwl, valid_nets, total_nets = calculate_hpwl(components, nets)
        
        print(f"\nResults:")
        print(f"Total HPWL: {hpwl}")
        print(f"Number of components: {len(components)}")
        print(f"Total nets: {total_nets}")
        print(f"Valid nets (>=2 components): {valid_nets}")
        print(f"Average HPWL per valid net: {hpwl/valid_nets if valid_nets > 0 else 0}")
        
        # 转换为实际微米值
        actual_hpwl = hpwl / 1000.0
        print(f"Actual HPWL (microns): {actual_hpwl}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 