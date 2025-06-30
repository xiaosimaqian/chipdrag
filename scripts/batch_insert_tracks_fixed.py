import os
import re
from pathlib import Path

def insert_tracks_to_lef(lef_path):
    with open(lef_path, 'r') as f:
        content = f.read()
    
    # 查找所有routing层
    routing_layers = []
    layer_pattern = r'LAYER (metal\d+)\s*\n(?:.*?\n)*?END \1'
    
    for match in re.finditer(layer_pattern, content, re.DOTALL):
        layer_name = match.group(1)
        layer_content = match.group(0)
        
        # 提取PITCH
        pitch_match = re.search(r'PITCH\s+([0-9.]+)', layer_content)
        if pitch_match:
            pitch = pitch_match.group(1)
            routing_layers.append((layer_name, pitch))
    
    if not routing_layers:
        return False
    
    # 生成TRACKS段
    tracks_content = "\n"
    for layer_name, pitch in routing_layers:
        tracks_content += f"TRACKS Y 0.000 {pitch} 1000 LAYER {layer_name} ;\n"
        tracks_content += f"TRACKS X 0.000 {pitch} 1000 LAYER {layer_name} ;\n"
    
    # 在文件末尾插入TRACKS
    content += tracks_content
    
    # 写回文件
    with open(lef_path, 'w') as f:
        f.write(content)
    
    print(f"  为 {len(routing_layers)} 个routing层插入了TRACKS段")
    return True

def main():
    base_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        return
    
    print("开始批量插入TRACKS段...")
    
    for design_dir in base_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        lef_path = design_dir / "tech.lef"
        if not lef_path.exists():
            print(f"跳过 {design_dir.name}: 未找到tech.lef")
            continue
            
        print(f"处理 {design_dir.name}...")
        try:
            if insert_tracks_to_lef(lef_path):
                print(f"  ✓ {design_dir.name} 处理完成")
            else:
                print(f"  ⚠ {design_dir.name} 未找到routing层")
        except Exception as e:
            print(f"  ✗ {design_dir.name} 处理失败: {e}")
    
    print("批量插入TRACKS完成！")

if __name__ == "__main__":
    main() 