import os
import re
from pathlib import Path

def rewrite_tech_lef(lef_path):
    with open(lef_path, 'r') as f:
        content = f.read()
    
    # 移除所有TRACKS行
    content = re.sub(r'TRACKS.*?\n', '', content)
    
    # 移除LAYER段内部的TRACKS段
    # 匹配LAYER metalX到END metalX之间的内容，移除其中的TRACKS
    lines = content.split('\n')
    new_lines = []
    in_metal_layer = False
    
    for line in lines:
        if line.strip().startswith('LAYER metal'):
            in_metal_layer = True
            new_lines.append(line)
        elif line.strip().startswith('END metal'):
            in_metal_layer = False
            new_lines.append(line)
        elif in_metal_layer and line.strip().startswith('TRACKS'):
            # 跳过LAYER段内的TRACKS行
            continue
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # 在文件末尾添加正确的TRACKS段
    tracks_content = "\n"
    for i in range(1, 6):  # metal1到metal5
        tracks_content += f"TRACKS Y 0.000 0.200 1000 LAYER metal{i} ;\n"
        tracks_content += f"TRACKS X 0.000 0.200 1000 LAYER metal{i} ;\n"
    
    content += tracks_content
    
    # 写回文件
    with open(lef_path, 'w') as f:
        f.write(content)
    
    print(f"  重写了tech.lef，添加了5个routing层的TRACKS段")
    return True

def main():
    base_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        return
    
    print("开始重写tech.lef文件...")
    
    for design_dir in base_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        lef_path = design_dir / "tech.lef"
        if not lef_path.exists():
            print(f"跳过 {design_dir.name}: 未找到tech.lef")
            continue
            
        print(f"处理 {design_dir.name}...")
        try:
            if rewrite_tech_lef(lef_path):
                print(f"  ✓ {design_dir.name} 处理完成")
            else:
                print(f"  ⚠ {design_dir.name} 处理失败")
        except Exception as e:
            print(f"  ✗ {design_dir.name} 处理失败: {e}")
    
    print("重写tech.lef完成！")

if __name__ == "__main__":
    main() 