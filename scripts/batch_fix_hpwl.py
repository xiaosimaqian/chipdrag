#!/usr/bin/env python3
"""
批量补全训练数据中HPWL为null的情况，自动调用calculate_hpwl.py计算DEF文件HPWL
"""
import json
import subprocess
from pathlib import Path

TRAINING_FILE = 'results/iterative_training/batch_training_results_with_hpwl.json'
OUTPUT_FILE = 'results/iterative_training/batch_training_results_with_hpwl_filled.json'
CALC_HPWL_SCRIPT = 'calculate_hpwl.py'

with open(TRAINING_FILE, 'r') as f:
    data = json.load(f)

fixed_count = 0
for result in data['results']:
    if result.get('success'):
        for iteration in result.get('iteration_data', []):
            if iteration.get('hpwl') is None:
                def_file = iteration.get('def_file')
                if def_file and Path(def_file).exists():
                    try:
                        hpwl = None
                        # 调用calculate_hpwl.py
                        proc = subprocess.run(['python', CALC_HPWL_SCRIPT, def_file], capture_output=True, text=True, timeout=30)
                        for line in proc.stdout.splitlines():
                            if 'Total HPWL:' in line:
                                hpwl = float(line.split(':')[-1].strip())
                                break
                        if hpwl is not None:
                            iteration['hpwl'] = hpwl
                            fixed_count += 1
                            print(f"补全: {def_file} -> {hpwl}")
                        else:
                            print(f"未能提取HPWL: {def_file}")
                    except Exception as e:
                        print(f"计算HPWL失败: {def_file}, {e}")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=2)

print(f"已补全HPWL: {fixed_count} 处。新文件: {OUTPUT_FILE}") 