import os
import re
import subprocess
import csv
from pathlib import Path

ROOT = Path('data/designs/ispd_2015_contest_benchmark')
RESULT_CSV = 'hpwl_results_fixed.csv'
OPENROAD_IMAGE = 'openroad/flow-ubuntu22.04-builder:21e414'

def get_design_size(def_path):
    """估算设计规模"""
    try:
        with open(def_path, 'r') as f:
            content = f.read()
            # 统计组件数量
            components = len(re.findall(r'^\s*-\s+\w+', content, re.MULTILINE))
            # 统计网络数量
            nets = len(re.findall(r'^\s*-\s+\w+.*\+ USE SIGNAL', content, re.MULTILINE))
            return components, nets
    except:
        return 0, 0

def extract_pins_section(def_path):
    """提取DEF文件中的PINS段"""
    pins_lines = []
    in_pins = False
    with open(def_path, 'r') as f:
        for line in f:
            if line.strip().startswith('PINS'):
                in_pins = True
            if in_pins:
                pins_lines.append(line)
                if line.strip().startswith('END PINS'):
                    break
    return pins_lines if pins_lines else None

def replace_pins_section(target_def, pins_lines):
    """用pins_lines替换target_def中的PINS段"""
    with open(target_def, 'r') as f:
        lines = f.readlines()
    new_lines = []
    in_pins = False
    for line in lines:
        if line.strip().startswith('PINS'):
            in_pins = True
            new_lines.extend(pins_lines)
        if not in_pins:
            new_lines.append(line)
        if in_pins and line.strip().startswith('END PINS'):
            in_pins = False
    # 若未找到PINS段，直接返回原内容
    if not any(l.strip().startswith('PINS') for l in lines):
        return False
    with open(target_def, 'w') as f:
        f.writelines(new_lines)
    return True

def extract_hpwl_with_openroad(work_dir, def_file):
    tech_lef = os.path.join(work_dir, 'tech.lef')
    cells_lef = os.path.join(work_dir, 'cells.lef')
    tcl_path = os.path.join(work_dir, 'extract_hpwl.tcl')
    
    # 估算设计规模，决定超时时间
    components, nets = get_design_size(os.path.join(work_dir, def_file))
    if components > 100000:  # 大型设计
        timeout = 600  # 10分钟
    elif components > 50000:  # 中型设计
        timeout = 300  # 5分钟
    else:  # 小型设计
        timeout = 120  # 2分钟
    
    # 生成TCL脚本
    with open(tcl_path, 'w') as f:
        f.write(f"""
read_lef tech.lef
read_lef cells.lef
read_def {def_file}
global_route
report_wire_length
exit
""")
    # 调用OpenROAD
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{os.path.abspath(work_dir)}:/workspace',
        '-w', '/workspace',
        OPENROAD_IMAGE,
        'bash', '-c',
        'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad extract_hpwl.tcl'
    ]
    try:
        print(f"  执行OpenROAD (超时: {timeout}秒)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # 解析HPWL
        for line in result.stdout.split('\n'):
            if 'Total wire length:' in line:
                hpwl = re.findall(r'Total wire length:\s*([\d\.eE\+\-]+)', line)
                if hpwl:
                    return float(hpwl[0])
            if 'HPWL:' in line:
                hpwl = re.findall(r'HPWL:\s*([\d\.eE\+\-]+)', line)
                if hpwl:
                    return float(hpwl[0])
        # 如果没有找到HPWL，检查是否有错误
        if result.stderr and 'Error:' in result.stderr:
            error_msg = result.stderr.split('Error:')[-1].split('\n')[0]
            print(f"  OpenROAD错误: {error_msg}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  OpenROAD超时 ({timeout}秒)")
        return None
    except Exception as e:
        print(f"  OpenROAD执行失败: {e}")
        return None

def main():
    results = []
    total_designs = len([d for d in ROOT.iterdir() if d.is_dir()])
    processed = 0
    
    for design_dir in ROOT.iterdir():
        if not design_dir.is_dir():
            continue
        processed += 1
        print(f"处理设计 {processed}/{total_designs}: {design_dir.name}")
        
        floorplan_def = design_dir / 'floorplan.def'
        if not floorplan_def.exists():
            print(f"  跳过: 未找到floorplan.def")
            continue
            
        pins_lines = extract_pins_section(floorplan_def)
        if not pins_lines:
            print(f"  跳过: 未能提取PINS段")
            continue
            
        iter_dir = design_dir / 'output' / 'iterations'
        if not iter_dir.exists():
            print(f"  跳过: 未找到iterations目录")
            continue
            
        def_files = list(iter_dir.glob('*.def'))
        print(f"  找到 {len(def_files)} 个DEF文件")
        
        for i, def_file in enumerate(def_files):
            print(f"    处理 {i+1}/{len(def_files)}: {def_file.name}")
            
            # 替换PINS段
            replaced = replace_pins_section(def_file, pins_lines)
            if not replaced:
                print(f"      跳过: 未能替换PINS段")
                continue
                
            # 提取HPWL
            hpwl = extract_hpwl_with_openroad(design_dir, os.path.relpath(def_file, design_dir))
            print(f"      HPWL: {hpwl}")
            results.append([design_dir.name, def_file.name, hpwl])
    
    # 保存结果
    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['design', 'def_file', 'hpwl'])
        writer.writerows(results)
    
    # 统计结果
    successful = sum(1 for r in results if r[2] is not None)
    total = len(results)
    print(f"\n批量修复并提取HPWL完成!")
    print(f"总计处理: {total} 个DEF文件")
    print(f"成功提取HPWL: {successful} 个")
    print(f"结果已保存到: {RESULT_CSV}")

if __name__ == '__main__':
    main() 