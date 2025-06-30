import os
import re
from pathlib import Path
import shutil

def extract_tracks(def_path):
    """从DEF文件中提取TRACKS段（支持多段）"""
    tracks = []
    with open(def_path, 'r') as f:
        lines = f.readlines()
    in_tracks = False
    for line in lines:
        if line.strip().startswith('TRACKS'):
            in_tracks = True
            tracks.append(line)
        elif in_tracks:
            tracks.append(line)
            if line.strip().endswith(';'):
                in_tracks = False
    return tracks

def insert_tracks_to_def(def_path, tracks):
    """将TRACKS段插入到所有ROW段之后（如已存在则替换）"""
    with open(def_path, 'r') as f:
        lines = f.readlines()
    
    # 先移除已有TRACKS段
    new_lines = []
    in_tracks = False
    for line in lines:
        if line.strip().startswith('TRACKS'):
            in_tracks = True
            continue
        if in_tracks:
            if line.strip().endswith(';'):
                in_tracks = False
            continue
        new_lines.append(line)
    
    # 找到最后一个ROW段插入点
    last_row_idx = 0
    for i, line in enumerate(new_lines):
        if line.strip().startswith('ROW'):
            last_row_idx = i
    insert_idx = last_row_idx + 1
    # 插入TRACKS段
    fixed_lines = new_lines[:insert_idx] + tracks + new_lines[insert_idx:]
    with open(def_path, 'w') as f:
        f.writelines(fixed_lines)

def main():
    base_dir = Path("data/designs/ispd_2015_contest_benchmark")
    for design_dir in base_dir.iterdir():
        if not design_dir.is_dir():
            continue
        floorplan_def = design_dir / "floorplan.def"
        placement_def = design_dir / "placement_result.def"
        if not floorplan_def.exists() or not placement_def.exists():
            print(f"跳过 {design_dir.name}: 缺少floorplan.def或placement_result.def")
            continue
        # 备份
        backup_path = placement_def.with_suffix('.def.backup')
        shutil.copy2(placement_def, backup_path)
        print(f"{design_dir.name}: 已备份placement_result.def -> {backup_path.name}")
        # 提取并插入TRACKS
        tracks = extract_tracks(floorplan_def)
        if not tracks:
            print(f"{design_dir.name}: floorplan.def未找到TRACKS段，跳过")
            continue
        insert_tracks_to_def(placement_def, tracks)
        print(f"{design_dir.name}: 已插入/修复TRACKS段")
    print("全部处理完成！")

if __name__ == "__main__":
    main() 