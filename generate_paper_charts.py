#!/usr/bin/env python3
"""
生成论文图表
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import json
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
rcParams['figure.figsize'] = (12, 8)

def load_validation_data():
    """加载验证数据"""
    with open('results/paper_method_validation/paper_method_validation_results.json', 'r') as f:
        return json.load(f)

def create_hpwl_improvement_chart():
    """创建HPWL改进对比图"""
    data = load_validation_data()
    
    designs = list(data['baseline_data'].keys())
    baseline_hpwls = []
    optimized_hpwls = []
    
    for design in designs:
        baseline_hpwl = data['baseline_data'][design]['baseline_hpwl']
        if baseline_hpwl == float('inf'):
            baseline_hpwl = 0  # 失败案例设为0
        baseline_hpwls.append(baseline_hpwl / 1e6)  # 转换为百万单位
        
        optimized_hpwl = data['dynamic_reranking_results'][design]['drag_hpwl']
        optimized_hpwls.append(optimized_hpwl / 1e6)
    
    x = np.arange(len(designs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, baseline_hpwls, width, label='基线HPWL', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized_hpwls, width, label='优化HPWL', color='lightblue', alpha=0.8)
    
    ax.set_xlabel('设计实例', fontsize=14)
    ax.set_ylabel('HPWL (百万单位)', fontsize=14)
    ax.set_title('HPWL改进效果对比', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('mgc_', '').replace('_', '\n') for d in designs], fontsize=10)
    ax.legend(fontsize=12)
    
    # 添加数值标签
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        if baseline_hpwls[i] > 0:
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 5, 
                   f'{baseline_hpwls[i]:.1f}', ha='center', va='bottom', fontsize=9)
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 5, 
               f'{optimized_hpwls[i]:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/paper_method_validation/hpwl_improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_improvement_rate_chart():
    """创建改进率对比图"""
    data = load_validation_data()
    
    designs = list(data['dynamic_reranking_results'].keys())
    improvement_rates = []
    
    for design in designs:
        result = data['dynamic_reranking_results'][design]
        if result.get('improvement_type') == 'normal':
            improvement_rates.append(result['improvement_percent'])
        else:
            improvement_rates.append(0)  # 特殊情况设为0
    
    # 只显示正常改进的设计
    valid_designs = [d for d, rate in zip(designs, improvement_rates) if rate > 0]
    valid_rates = [rate for rate in improvement_rates if rate > 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(valid_designs)), valid_rates, color='skyblue', alpha=0.8)
    
    ax.set_xlabel('设计实例', fontsize=14)
    ax.set_ylabel('HPWL改进率 (%)', fontsize=14)
    ax.set_title('各设计HPWL改进率', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(valid_designs)))
    ax.set_xticklabels([d.replace('mgc_', '').replace('_', '\n') for d in valid_designs], fontsize=10)
    
    # 添加数值标签
    for bar, rate in zip(bars, valid_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加平均线
    avg_rate = np.mean(valid_rates)
    ax.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2, label=f'平均改进率: {avg_rate:.2f}%')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/paper_method_validation/improvement_rate_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_execution_time_chart():
    """创建执行时间分析图"""
    data = load_validation_data()
    
    designs = list(data['dynamic_reranking_results'].keys())
    execution_times = []
    
    for design in designs:
        result = data['dynamic_reranking_results'][design]
        execution_times.append(result['execution_time'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 柱状图
    bars = ax1.bar(range(len(designs)), execution_times, color='lightgreen', alpha=0.8)
    ax1.set_xlabel('设计实例', fontsize=12)
    ax1.set_ylabel('执行时间 (秒)', fontsize=12)
    ax1.set_title('各设计执行时间', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(designs)))
    ax1.set_xticklabels([d.replace('mgc_', '').replace('_', '\n') for d in designs], fontsize=9)
    
    # 添加数值标签
    for bar, time in zip(bars, execution_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{time:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 饼图
    total_time = sum(execution_times)
    percentages = [t/total_time*100 for t in execution_times]
    labels = [d.replace('mgc_', '').replace('_', '\n') for d in designs]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(designs)))
    wedges, texts, autotexts = ax2.pie(execution_times, labels=labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('执行时间分布', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/paper_method_validation/execution_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_method_comparison_chart():
    """创建方法对比图"""
    data = load_validation_data()
    comparison = data['comparison_analysis']
    
    methods = ['动态重排序', '实体增强', '质量反馈']
    improvement_rates = [
        comparison['improvement_rates']['dynamic_reranking'],
        comparison['improvement_rates']['entity_enhancement'],
        comparison['improvement_rates']['quality_feedback']
    ]
    success_rates = [
        comparison['success_rates']['dynamic_reranking'] * 100,
        comparison['success_rates']['entity_enhancement'] * 100,
        comparison['success_rates']['quality_feedback'] * 100
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 改进率对比
    bars1 = ax1.bar(x, improvement_rates, width, color='lightblue', alpha=0.8)
    ax1.set_xlabel('方法', fontsize=12)
    ax1.set_ylabel('HPWL改进率 (%)', fontsize=12)
    ax1.set_title('各方法HPWL改进率对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    
    for bar, rate in zip(bars1, improvement_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 成功率对比
    bars2 = ax2.bar(x, success_rates, width, color='lightgreen', alpha=0.8)
    ax2.set_xlabel('方法', fontsize=12)
    ax2.set_ylabel('成功率 (%)', fontsize=12)
    ax2.set_title('各方法成功率对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.set_ylim(0, 110)
    
    for bar, rate in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/paper_method_validation/method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_iteration_analysis_chart():
    """创建迭代优化分析图"""
    data = load_validation_data()
    
    designs = list(data['quality_feedback_results'].keys())
    iterations_data = {}
    
    for design in designs:
        iterations = data['quality_feedback_results'][design]['iterations']
        hpwls = [iter['hpwl'] / 1e6 for iter in iterations]  # 转换为百万单位
        iterations_data[design] = hpwls
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, (design, hpwls) in enumerate(iterations_data.items()):
        design_name = design.replace('mgc_', '').replace('_', ' ')
        ax.plot(range(1, len(hpwls) + 1), hpwls, 
               marker=markers[i], linewidth=2, markersize=8, 
               label=design_name, color=colors[i])
    
    ax.set_xlabel('迭代轮次', fontsize=14)
    ax.set_ylabel('HPWL (百万单位)', fontsize=14)
    ax.set_title('质量反馈驱动的迭代优化过程', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/paper_method_validation/iteration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("开始生成论文图表...")
    
    # 确保输出目录存在
    os.makedirs('results/paper_method_validation', exist_ok=True)
    
    # 生成各种图表
    create_hpwl_improvement_chart()
    print("✓ HPWL改进对比图已生成")
    
    create_improvement_rate_chart()
    print("✓ 改进率对比图已生成")
    
    create_execution_time_chart()
    print("✓ 执行时间分析图已生成")
    
    create_method_comparison_chart()
    print("✓ 方法对比图已生成")
    
    create_iteration_analysis_chart()
    print("✓ 迭代优化分析图已生成")
    
    print("\n所有图表已生成完成！")
    print("图表保存在: results/paper_method_validation/")

if __name__ == "__main__":
    main() 