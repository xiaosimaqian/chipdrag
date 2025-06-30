#!/usr/bin/env python3
import json
import os
import statistics

def analyze_real_data():
    results_dir = 'results/ispd_training_fixed_v10'
    summary_file = f'{results_dir}/training_summary.json'
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    successful_designs = data['successful_designs']
    failed_designs = data['failed_designs']
    
    total_designs = len(successful_designs) + len(failed_designs)
    success_rate = len(successful_designs) / total_designs * 100
    
    print("=== ISPD 2015 真实实验结果分析 ===")
    print(f"总设计数: {total_designs}")
    print(f"成功设计数: {len(successful_designs)}")
    print(f"失败设计数: {len(failed_designs)}")
    print(f"成功率: {success_rate:.2f}%")
    print()
    
    print("成功设计列表:")
    for design in successful_designs:
        print(f"  - {design}")
    print()
    
    print("失败设计列表:")
    for design in failed_designs:
        print(f"  - {design}")
    print()
    
    # 分析执行时间
    execution_times = []
    for design in successful_designs:
        result_file = f'{results_dir}/{design}_result.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                execution_time = result_data.get('execution_time', 0)
                execution_times.append(execution_time)
    
    if execution_times:
        print("执行时间统计:")
        print(f"  平均执行时间: {statistics.mean(execution_times):.2f} 秒")
        print(f"  最短执行时间: {min(execution_times):.2f} 秒")
        print(f"  最长执行时间: {max(execution_times):.2f} 秒")
        print(f"  中位数执行时间: {statistics.median(execution_times):.2f} 秒")
        print()
    
    # 分析设计规模
    design_categories = {
        'DES': [d for d in successful_designs if 'des' in d.lower()],
        'FFT': [d for d in successful_designs if 'fft' in d.lower()],
        'Matrix': [d for d in successful_designs if 'matrix' in d.lower()],
        'PCI': [d for d in successful_designs if 'pci' in d.lower()],
        'Superblue': [d for d in successful_designs if 'superblue' in d.lower()]
    }
    
    print("设计类别分布:")
    for category, designs in design_categories.items():
        if designs:
            print(f"  {category}: {len(designs)} 个设计")
            for design in designs:
                print(f"    - {design}")
    print()
    
    return {
        'total_designs': total_designs,
        'successful_designs': len(successful_designs),
        'failed_designs': len(failed_designs),
        'success_rate': success_rate,
        'execution_times': execution_times,
        'design_categories': design_categories
    }

if __name__ == "__main__":
    analyze_real_data() 