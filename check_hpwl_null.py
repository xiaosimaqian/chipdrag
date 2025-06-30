#!/usr/bin/env python3
"""检查训练数据中HPWL为null的情况"""

import json

# 加载训练数据
with open('results/iterative_training/batch_training_results_with_hpwl.json', 'r') as f:
    data = json.load(f)

null_count = 0
total_count = 0
designs_with_null = []

for result in data['results']:
    if result.get('success'):
        design_name = result['design']
        has_null = False
        
        for iteration in result.get('iteration_data', []):
            total_count += 1
            if iteration.get('hpwl') is None:
                null_count += 1
                has_null = True
        
        if has_null:
            designs_with_null.append(design_name)

print(f'HPWL为null的迭代数: {null_count}/{total_count}')
print(f'包含null HPWL的设计: {designs_with_null}')
print(f'设计数量: {len(designs_with_null)}') 