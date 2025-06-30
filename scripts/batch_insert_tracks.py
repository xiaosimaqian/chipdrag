import os
import re
from pathlib import Path

def insert_tracks_to_lef(lef_path):
    with open(lef_path, 'r') as f:
        lines = f.readlines()
    
    output = []
    routing_layers = []
    i = 0
    
    # 第一遍：收集所有routing层信息
    while i < len(lines):
        line = lines[i]
        m = re.match(r'LAYER (metal\d+)', line.strip())
        if m:
            layer_name = m.group(1)
            pitch = None
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('END'):
                if 'PITCH' in lines[j]:
                    pitch_match = re.search(r'PITCH\s+([0-9.]+)', lines[j])
                    if pitch_match:
                        pitch = pitch_match.group(1)
                        break
                j += 1
            if pitch:
                routing_layers.append((layer_name, pitch))
        i += 1
    
    # 第二遍：重新处理文件，在最后插入TRACKS
    i = 0
    while i < len(lines):
        line = lines[i]
        output.append(line)
        
        # 检查是否到达文件末尾或遇到其他全局段
        if (i == len(lines) - 1 or 
            (line.strip().startswith('SITE') and not any('LAYER' in l for l in lines[i-5:i])) or
            (line.strip().startswith('VIA') and not any('LAYER' in l for l in lines[i-5:i]))):
            
            # 插入TRACKS段
            for layer_name, pitch in routing_layers:
                tracks_content = f"TRACKS Y 0.000 {pitch} 1000 LAYER {layer_name} ;\n"
                tracks_content += f"TRACKS X 0.000 {pitch} 1000 LAYER {layer_name} ;\n"
                output.append(tracks_content)
            
            print(f"  为 {len(routing_layers)} 个routing层插入了TRACKS段")
            break
        
        i += 1
    
    # 写回文件
    with open(lef_path, 'w') as f:
        f.writelines(output)
    
    return len(routing_layers) > 0

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