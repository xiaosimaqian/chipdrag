#!/usr/bin/env python3
"""检查训练数据中有效HPWL的设计数量"""

import json

def check_training_data():
    """检查训练数据"""
    with open('results/iterative_training/batch_training_results_with_hpwl.json', 'r') as f:
        data = json.load(f)
    
    print(f"总设计数量: {len(data['results'])}")
    
    valid_count = 0
    valid_designs = []
    
    for design in data['results']:
        design_name = design['design']
        hpwl = None
        
        # 查找迭代0的HPWL
        for iter_data in design.get('iteration_data', []):
            if iter_data.get('iteration') == 0:
                hpwl = iter_data.get('hpwl')
                break
        
        if hpwl and hpwl > 1000:
            valid_count += 1
            valid_designs.append((design_name, hpwl))
        else:
            print(f"无效HPWL: {design_name} = {hpwl}")
    
    print(f"有效HPWL设计数量: {valid_count}")
    print("有效设计列表:")
    for name, hpwl in valid_designs:
        print(f"  {name}: {hpwl:.2e}")

if __name__ == "__main__":
    check_training_data() 