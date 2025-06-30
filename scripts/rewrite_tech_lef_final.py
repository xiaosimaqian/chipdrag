import os
import re
from pathlib import Path

def rewrite_tech_lef_final(lef_path):
    with open(lef_path, 'r') as f:
        content = f.read()
    
    # 移除所有TRACKS行
    content = re.sub(r'TRACKS.*?\n', '', content)
    
    # 移除LAYER段内部的TRACKS段
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
    
    # 在VIA定义之前插入TRACKS段
    via_pattern = r'(VIA.*?DEFAULT.*?\n.*?LAYER.*?;.*?\n)'
    
    tracks_content = "\nTRACKS Y 0.000 0.200 1000 LAYER metal1 ;\n"
    tracks_content += "TRACKS X 0.000 0.200 1000 LAYER metal1 ;\n"
    tracks_content += "TRACKS Y 0.000 0.200 1000 LAYER metal2 ;\n"
    tracks_content += "TRACKS X 0.000 0.200 1000 LAYER metal2 ;\n"
    tracks_content += "TRACKS Y 0.000 0.200 1000 LAYER metal3 ;\n"
    tracks_content += "TRACKS X 0.000 0.200 1000 LAYER metal3 ;\n"
    tracks_content += "TRACKS Y 0.000 0.200 1000 LAYER metal4 ;\n"
    tracks_content += "TRACKS X 0.000 0.200 1000 LAYER metal4 ;\n"
    tracks_content += "TRACKS Y 0.000 0.200 1000 LAYER metal5 ;\n"
    tracks_content += "TRACKS X 0.000 0.200 1000 LAYER metal5 ;\n\n"
    
    # 替换VIA段，在VIA前插入TRACKS
    content = re.sub(via_pattern, tracks_content + r'\1', content, flags=re.DOTALL)
    
    # 写回文件
    with open(lef_path, 'w') as f:
        f.write(content)
    
    print(f"  重写了tech.lef，在VIA前正确插入了TRACKS段")
    return True

def main():
    base_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        return
    
    print("开始最终修正tech.lef文件...")
    
    for design_dir in base_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        lef_path = design_dir / "tech.lef"
        if not lef_path.exists():
            print(f"跳过 {design_dir.name}: 未找到tech.lef")
            continue
            
        print(f"处理 {design_dir.name}...")
        try:
            if rewrite_tech_lef_final(lef_path):
                print(f"  ✓ {design_dir.name} 处理完成")
            else:
                print(f"  ⚠ {design_dir.name} 处理失败")
        except Exception as e:
            print(f"  ✗ {design_dir.name} 处理失败: {e}")
    
    print("最终修正tech.lef完成！")

if __name__ == "__main__":
    main() 