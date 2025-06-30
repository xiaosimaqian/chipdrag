import os
import re
import subprocess
import csv
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path('data/designs/ispd_2015_contest_benchmark')
RESULT_CSV = 'hpwl_results_fast.csv'
PROGRESS_FILE = 'hpwl_extraction_progress.json'
OPENROAD_IMAGE = 'openroad/flow-ubuntu22.04-builder:21e414'

# 跳过的大型设计（处理时间过长）
SKIP_LARGE_DESIGNS = {
    'mgc_superblue16_a', 'mgc_superblue11_a', 'mgc_des_perf_a'
}

def get_design_size(def_path):
    """快速估算设计规模"""
    try:
        with open(def_path, 'r') as f:
            content = f.read(10000)  # 只读取前10KB
            components = len(re.findall(r'^\s*-\s+\w+', content, re.MULTILINE))
            return components
    except:
        return 0

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
    if not any(l.strip().startswith('PINS') for l in lines):
        return False
    with open(target_def, 'w') as f:
        f.writelines(new_lines)
    return True

def process_single_def(args):
    """处理单个DEF文件（用于多进程）"""
    design_dir, def_file, pins_lines = args
    
    # 替换PINS段
    replaced = replace_pins_section(def_file, pins_lines)
    if not replaced:
        return design_dir.name, def_file.name, None
    
    # 提取HPWL
    hpwl = extract_hpwl_with_openroad(design_dir, os.path.relpath(def_file, design_dir))
    return design_dir.name, def_file.name, hpwl

def extract_hpwl_with_openroad(work_dir, def_file):
    """使用OpenROAD提取HPWL"""
    tcl_path = os.path.join(work_dir, 'extract_hpwl.tcl')
    
    # 根据设计规模设置超时
    components = get_design_size(os.path.join(work_dir, def_file))
    if components > 50000:
        timeout = 300  # 5分钟
    else:
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
        return None
    except:
        return None

def load_progress():
    """加载进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'processed': set(), 'results': []}

def save_progress(processed, results):
    """保存进度"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            'processed': list(processed),
            'results': results
        }, f)

def main():
    # 加载进度
    progress = load_progress()
    processed_designs = set(progress['processed'])
    results = progress['results']
    
    # 收集所有任务
    tasks = []
    for design_dir in ROOT.iterdir():
        if not design_dir.is_dir() or design_dir.name in SKIP_LARGE_DESIGNS:
            continue
            
        if design_dir.name in processed_designs:
            continue
            
        floorplan_def = design_dir / 'floorplan.def'
        if not floorplan_def.exists():
            continue
            
        pins_lines = extract_pins_section(floorplan_def)
        if not pins_lines:
            continue
            
        iter_dir = design_dir / 'output' / 'iterations'
        if not iter_dir.exists():
            continue
            
        for def_file in iter_dir.glob('*.def'):
            tasks.append((design_dir, def_file, pins_lines))
    
    if not tasks:
        print("没有新的任务需要处理")
        return
    
    print(f"找到 {len(tasks)} 个DEF文件需要处理")
    
    # 使用多进程处理
    with Pool(processes=min(cpu_count(), 4)) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_def, tasks)):
            design_name, def_name, hpwl = result
            results.append([design_name, def_name, hpwl])
            
            # 更新进度
            processed_designs.add(design_name)
            save_progress(processed_designs, results)
            
            print(f"进度 {i+1}/{len(tasks)}: {design_name} - {def_name} - HPWL: {hpwl}")
    
    # 保存最终结果
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