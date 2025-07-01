import os
from pathlib import Path

def clean_tracks_from_def(def_path: Path):
    with open(def_path, 'r') as f:
        lines = f.readlines()
    
    # 删除所有手动添加的密集TRACKS段（STEP 1的）
    cleaned_lines = []
    for line in lines:
        if 'TRACKS' in line and 'STEP 1' in line:
            # 跳过手动添加的密集TRACKS
            continue
        cleaned_lines.append(line)
    
    # 强制覆盖文件，恢复官方TRACKS
    with open(def_path, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"[OK] 已清理手动TRACKS，恢复官方TRACKS到 {def_path}")

def main():
    base_dir = Path('data/designs/ispd_2015_contest_benchmark')
    for design_dir in base_dir.iterdir():
        if not design_dir.is_dir():
            continue
        def_file = design_dir / 'floorplan.def'
        if def_file.exists():
            clean_tracks_from_def(def_file)
        else:
            print(f"[WARN] {def_file} 不存在")

if __name__ == '__main__':
    main() 