#!/usr/bin/env python3
"""
从真实训练数据中提取HPWL数据并更新optimization_results.json
"""

import json
import os
from pathlib import Path

def extract_real_hpwl_data():
    """从真实训练数据中提取HPWL数据"""
    real_data_file = "results/iterative_training/batch_training_results_with_hpwl_filled.json"
    
    with open(real_data_file, 'r') as f:
        real_data = json.load(f)
    
    # 提取真实HPWL数据
    real_hpwl_data = {}
    
    for result in real_data['results']:
        design_name = result['design']
        
        # 检查是否有iteration_data字段
        if 'iteration_data' not in result:
            print(f"警告: {design_name} 没有iteration_data字段")
            continue
            
        iteration_data = result['iteration_data']
        
        # 找到初始HPWL（iteration 0）和最终HPWL（iteration 10）
        initial_hpwl = None
        final_hpwl = None
        
        for iteration in iteration_data:
            if iteration['iteration'] == 0:
                initial_hpwl = iteration['hpwl']
            elif iteration['iteration'] == 10:
                final_hpwl = iteration['hpwl']
        
        if initial_hpwl is not None and final_hpwl is not None:
            real_hpwl_data[design_name] = {
                'baseline_hpwl': initial_hpwl,
                'optimized_hpwl': final_hpwl,
                'improvement': ((initial_hpwl - final_hpwl) / initial_hpwl) * 100 if initial_hpwl > 0 else 0,
                'success': result['success'],
                'duration': result['duration']
            }
        else:
            print(f"警告: {design_name} 缺少初始或最终HPWL数据")
    
    return real_hpwl_data

def update_optimization_results(real_hpwl_data):
    """更新optimization_results.json文件"""
    optimization_file = "results/rl_optimization_experiment/optimization_results.json"
    
    # 备份原文件
    backup_file = optimization_file + ".backup"
    if not os.path.exists(backup_file):
        os.system(f"cp {optimization_file} {backup_file}")
        print(f"已备份原文件到: {backup_file}")
    
    with open(optimization_file, 'r') as f:
        optimization_data = json.load(f)
    
    # 更新数据
    updated_count = 0
    
    for exp_name, exp_results in optimization_data.items():
        for result in exp_results:
            design_name = result['design_name']
            
            if design_name in real_hpwl_data:
                real_data = real_hpwl_data[design_name]
                
                # 更新HPWL数据
                result['baseline_hpwl'] = real_data['baseline_hpwl']
                result['optimized_hpwl'] = real_data['optimized_hpwl']
                result['improvement'] = real_data['improvement']
                result['success'] = real_data['success']
                result['execution_time'] = real_data['duration']
                
                # 如果失败，设置error信息
                if not real_data['success']:
                    result['error'] = "Real experiment failed"
                else:
                    result['error'] = None
                
                updated_count += 1
                print(f"已更新 {design_name}: 基线HPWL={real_data['baseline_hpwl']:.0f}, "
                      f"优化HPWL={real_data['optimized_hpwl']:.0f}, "
                      f"改进率={real_data['improvement']:.2f}%")
    
    # 保存更新后的文件
    with open(optimization_file, 'w') as f:
        json.dump(optimization_data, f, indent=2)
    
    print(f"\n总共更新了 {updated_count} 个实验结果")
    print(f"文件已保存到: {optimization_file}")

def main():
    """主函数"""
    print("开始从真实训练数据中提取HPWL数据...")
    
    # 提取真实数据
    real_hpwl_data = extract_real_hpwl_data()
    
    print(f"提取到 {len(real_hpwl_data)} 个设计的真实HPWL数据:")
    for design, data in real_hpwl_data.items():
        print(f"  {design}: 基线={data['baseline_hpwl']:.0f}, "
              f"优化={data['optimized_hpwl']:.0f}, "
              f"改进={data['improvement']:.2f}%")
    
    # 更新optimization_results.json
    print("\n开始更新optimization_results.json...")
    update_optimization_results(real_hpwl_data)
    
    print("\n更新完成！")

if __name__ == "__main__":
    main() 