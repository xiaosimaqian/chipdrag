#!/usr/bin/env python3
"""
统计每个设计iteration 0、iteration 10、iteration 10_rl_training的HPWL，并输出提升率表格。
"""
import json
import os
from tabulate import tabulate

def extract_hpwl_from_iterations(iteration_data):
    hpwl_0 = None
    hpwl_10 = None
    hpwl_10_rl = None
    for item in iteration_data:
        if item['iteration'] == 0:
            hpwl_0 = item['hpwl']
        elif item['iteration'] == 10 and not item['def_file'].endswith('rl_training.def'):
            hpwl_10 = item['hpwl']
        elif item['iteration'] == 10 and item['def_file'].endswith('rl_training.def'):
            hpwl_10_rl = item['hpwl']
    return hpwl_0, hpwl_10, hpwl_10_rl

def main():
    data_file = "results/iterative_training/batch_training_results_with_hpwl_filled.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    table = []
    header = ["设计名", "极差HPWL(iter0)", "OpenROAD HPWL(iter10)", "ChipDRAG HPWL(iter10_rl)", "传统提升率", "智能提升率", "总提升率"]
    for result in data['results']:
        design = result['design']
        if 'iteration_data' not in result:
            continue
        hpwl_0, hpwl_10, hpwl_10_rl = extract_hpwl_from_iterations(result['iteration_data'])
        if not (hpwl_0 and hpwl_10 and hpwl_10_rl):
            continue
        # 计算提升率
        try:
            base2openroad = (hpwl_0 - hpwl_10) / hpwl_0 * 100 if hpwl_0 > 0 else 0
            openroad2drag = (hpwl_10 - hpwl_10_rl) / hpwl_10 * 100 if hpwl_10 > 0 else 0
            base2drag = (hpwl_0 - hpwl_10_rl) / hpwl_0 * 100 if hpwl_0 > 0 else 0
        except Exception as e:
            base2openroad = openroad2drag = base2drag = 0
        table.append([
            design,
            f"{hpwl_0:,.0f}",
            f"{hpwl_10:,.0f}",
            f"{hpwl_10_rl:,.0f}",
            f"{base2openroad:.2f}%",
            f"{openroad2drag:.2f}%",
            f"{base2drag:.2f}%"
        ])
    print(tabulate(table, headers=header, tablefmt="github", numalign="right", stralign="center"))

if __name__ == "__main__":
    main() 