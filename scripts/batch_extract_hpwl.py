import os
import subprocess
import csv
from pathlib import Path

# ISPD设计根目录
ROOT = Path('data/designs/ispd_2015_contest_benchmark')
# 结果输出文件
RESULT_CSV = 'hpwl_results.csv'
# OpenROAD docker镜像
OPENROAD_IMAGE = 'openroad/flow-ubuntu22.04-builder:21e414'


def extract_hpwl_with_openroad(work_dir, def_file):
    """
    用OpenROAD提取单个DEF文件的HPWL
    """
    tech_lef = os.path.join(work_dir, 'tech.lef')
    cells_lef = os.path.join(work_dir, 'cells.lef')
    tcl_path = os.path.join(work_dir, 'extract_hpwl.tcl')
    # 生成TCL脚本
    with open(tcl_path, 'w') as f:
        f.write(f"""
read_lef tech.lef
read_lef cells.lef
read_def {def_file}
report_wire_length
exit
""")
    # 调用OpenROAD
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{os.path.abspath(work_dir)}:/workspace',
        '-w', '/workspace',
        OPENROAD_IMAGE,
        'openroad', 'extract_hpwl.tcl'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stdout = result.stdout
        # 解析HPWL
        for line in stdout.split('\n'):
            if 'Total wire length:' in line:
                # 例: Total wire length: 123456789 um
                try:
                    hpwl = float(line.split(':')[1].split()[0])
                    return hpwl
                except Exception:
                    continue
        return None
    except Exception as e:
        print(f"[ERROR] {def_file} 提取HPWL失败: {e}")
        return None

def main():
    results = []
    for design_dir in ROOT.iterdir():
        if not design_dir.is_dir():
            continue
        design_name = design_dir.name
        iter_dir = design_dir / 'output' / 'iterations'
        if not iter_dir.exists():
            continue
        for def_file in iter_dir.glob('*.def'):
            rel_def = def_file.relative_to(design_dir)
            print(f"提取: {design_name} - {rel_def}")
            hpwl = extract_hpwl_with_openroad(design_dir, rel_def)
            results.append({
                'design': design_name,
                'def_file': str(rel_def),
                'hpwl': hpwl
            })
    # 写入CSV
    with open(RESULT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['design', 'def_file', 'hpwl'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"批量HPWL提取完成，结果已保存到 {RESULT_CSV}")

if __name__ == '__main__':
    main() 