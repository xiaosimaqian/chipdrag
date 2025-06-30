import os
import re
from pathlib import Path

def clean_and_fix_tracks(lef_path):
    with open(lef_path, 'r') as f:
        content = f.read()
    
    # 移除LAYER段内部的TRACKS段
    # 匹配模式：LAYER metalX 后面跟着的TRACKS行，直到END metalX
    pattern = r'(LAYER metal\d+\n)((?:TRACKS.*?\n)*?)(END metal\d+\n)'
    
    def replace_layer(match):
        layer_start = match.group(1)
        tracks_lines = match.group(2)
        layer_end = match.group(3)
        
        # 移除TRACKS行
        tracks_lines = re.sub(r'TRACKS.*?\n', '', tracks_lines)
        
        return layer_start + tracks_lines + layer_end
    
    content = re.sub(pattern, replace_layer, content, flags=re.MULTILINE)
    
    # 查找所有routing层
    routing_layers = []
    layer_pattern = r'LAYER (metal\d+)\n(?:.*?\n)*?END \1'
    
    for match in re.finditer(layer_pattern, content, re.DOTALL):
        layer_name = match.group(1)
        layer_content = match.group(0)
        
        # 提取PITCH（从TRACKS行中提取）
        pitch_match = re.search(r'TRACKS.*?(\d+\.\d+)', layer_content)
        if pitch_match:
            pitch = pitch_match.group(1)
            routing_layers.append((layer_name, pitch))
    
    if not routing_layers:
        # 如果没有找到routing层，尝试从LAYER定义中提取PITCH
        for match in re.finditer(r'LAYER (metal\d+)\n(?:.*?\n)*?END \1', content, re.DOTALL):
            layer_name = match.group(1)
            layer_content = match.group(0)
            
            # 查找PITCH定义
            pitch_match = re.search(r'PITCH\s+([0-9.]+)', layer_content)
            if pitch_match:
                pitch = pitch_match.group(1)
                routing_layers.append((layer_name, pitch))
    
    if not routing_layers:
        return False
    
    # 在文件末尾插入正确的TRACKS段
    tracks_content = "\n"
    for layer_name, pitch in routing_layers:
        tracks_content += f"TRACKS Y 0.000 {pitch} 1000 LAYER {layer_name} ;\n"
        tracks_content += f"TRACKS X 0.000 {pitch} 1000 LAYER {layer_name} ;\n"
    
    content += tracks_content
    
    # 写回文件
    with open(lef_path, 'w') as f:
        f.write(content)
    
    print(f"  清理并修复了 {len(routing_layers)} 个routing层的TRACKS段")
    return True

def main():
    base_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        return
    
    print("开始清理并修复TRACKS段...")
    
    for design_dir in base_dir.iterdir():
        if not design_dir.is_dir():
            continue
            
        lef_path = design_dir / "tech.lef"
        if not lef_path.exists():
            print(f"跳过 {design_dir.name}: 未找到tech.lef")
            continue
            
        print(f"处理 {design_dir.name}...")
        try:
            if clean_and_fix_tracks(lef_path):
                print(f"  ✓ {design_dir.name} 处理完成")
            else:
                print(f"  ⚠ {design_dir.name} 未找到routing层")
        except Exception as e:
            print(f"  ✗ {design_dir.name} 处理失败: {e}")
    
    print("清理并修复TRACKS完成！")

if __name__ == "__main__":
    main() 