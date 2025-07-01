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
import matplotlib.font_manager as fm
from graphviz import Digraph

# 自动检测并设置可用的中文字体，彻底解决中文乱码
zh_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STHeiti', 'Heiti TC', 'PingFang SC', 'WenQuanYi Zen Hei']
avail_fonts = set(f.name for f in fm.fontManager.ttflist)
for font in zh_fonts:
    if font in avail_fonts:
        plt.rcParams['font.sans-serif'] = [font]
        break
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
    
    bars1 = ax.bar(x - width/2, baseline_hpwls, width, label='Baseline HPWL', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized_hpwls, width, label='Optimized HPWL', color='lightblue', alpha=0.8)
    
    ax.set_xlabel('Design', fontsize=14)
    ax.set_ylabel('HPWL (Millions)', fontsize=14)
    ax.set_title('HPWL Improvement Comparison', fontsize=16, fontweight='bold')
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
    
    ax.set_xlabel('Design', fontsize=14)
    ax.set_ylabel('HPWL Improvement Rate (%)', fontsize=14)
    ax.set_title('HPWL Improvement Rate by Design', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(valid_designs)))
    ax.set_xticklabels([d.replace('mgc_', '').replace('_', '\n') for d in valid_designs], fontsize=10)
    
    # 添加数值标签
    for bar, rate in zip(bars, valid_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加平均线
    avg_rate = np.mean(valid_rates)
    ax.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2, label=f'Average Improvement Rate: {avg_rate:.2f}%')
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
    ax1.set_xlabel('Design', fontsize=12)
    ax1.set_ylabel('Execution Time (s)', fontsize=12)
    ax1.set_title('Execution Time by Design', fontsize=14, fontweight='bold')
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
    ax2.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/paper_method_validation/execution_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_method_comparison_chart():
    """创建方法对比图"""
    data = load_validation_data()
    comparison = data['comparison_analysis']
    
    methods = ['Dynamic Reranking', 'Entity Enhancement', 'Quality Feedback']
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
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('HPWL Improvement Rate (%)', fontsize=12)
    ax1.set_title('HPWL Improvement by Method', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    
    for bar, rate in zip(bars1, improvement_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 成功率对比
    bars2 = ax2.bar(x, success_rates, width, color='lightgreen', alpha=0.8)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate by Method', fontsize=14, fontweight='bold')
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
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('HPWL (Millions)', fontsize=14)
    ax.set_title('Iterative Optimization Driven by Quality Feedback', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/paper_method_validation/iteration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_chip_d_rag_framework_diagram():
    dot = Digraph('Chip-D-RAG-Framework', format='png')
    dot.attr(rankdir='LR', size='10,6')
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='12')

    # 输入
    dot.node('Input', 'Design Query / Input', fillcolor='#f9f')

    # 知识库分组
    with dot.subgraph(name='cluster_kb') as kb:
        kb.attr(label='Knowledge Base', style='filled', color='#bbf', fillcolor='#eef')
        kb.node('Case', 'Case Retrieval', fillcolor='#bbf')
        kb.node('Constraint', 'Constraint DB', fillcolor='#bbf')

    # 动态检索分组
    with dot.subgraph(name='cluster_retriever') as retr:
        retr.attr(label='Dynamic Retriever', style='filled', color='#ffd700', fillcolor='#fff8dc')
        retr.node('RL', 'RL Agent', fillcolor='#ffd700')
        retr.node('Sim', 'Similarity Calculation', fillcolor='#ffd700')

    # 实体增强分组
    with dot.subgraph(name='cluster_entity') as ent:
        ent.attr(label='Entity Enhancement', style='filled', color='#90ee90', fillcolor='#eaffea')
        ent.node('Extract', 'Entity Extraction', fillcolor='#90ee90')
        ent.node('Compress', 'Compression', fillcolor='#90ee90')
        ent.node('Inject', 'Injection', fillcolor='#90ee90')

    # 多模态融合
    with dot.subgraph(name='cluster_multi') as multi:
        multi.attr(label='Multimodal Fusion', style='filled', color='#87ceeb', fillcolor='#e6f7ff')
        multi.node('Text', 'Text', fillcolor='#87ceeb')
        multi.node('Image', 'Image', fillcolor='#87ceeb')
        multi.node('Struct', 'Structured Data', fillcolor='#87ceeb')

    # 布局生成
    dot.node('Gen', 'Layout Generator', fillcolor='#ffa07a')

    # 质量评估分组
    with dot.subgraph(name='cluster_eval') as eval:
        eval.attr(label='Quality Evaluator', style='filled', color='#ffb347', fillcolor='#fff5e6')
        eval.node('HPWL', 'HPWL', fillcolor='#ffb347')
        eval.node('Cong', 'Congestion', fillcolor='#ffb347')
        eval.node('Timing', 'Timing', fillcolor='#ffb347')
        eval.node('Power', 'Power', fillcolor='#ffb347')

    # 反馈闭环
    with dot.subgraph(name='cluster_fb') as fb:
        fb.attr(label='Feedback Loop', style='filled', color='#b0e0e6', fillcolor='#e0f7fa')
        fb.node('Policy', 'Policy Update', fillcolor='#b0e0e6')
        fb.node('Signal', 'Quality Signal', fillcolor='#b0e0e6')

    # 箭头连接
    dot.edge('Input', 'Case')
    dot.edge('Input', 'Constraint')
    dot.edge('Case', 'RL')
    dot.edge('Constraint', 'RL')
    dot.edge('RL', 'Sim')
    dot.edge('Sim', 'Extract')
    dot.edge('Extract', 'Compress')
    dot.edge('Compress', 'Inject')
    dot.edge('Inject', 'Text')
    dot.edge('Inject', 'Image')
    dot.edge('Inject', 'Struct')
    dot.edge('Text', 'Gen')
    dot.edge('Image', 'Gen')
    dot.edge('Struct', 'Gen')
    dot.edge('Gen', 'HPWL')
    dot.edge('Gen', 'Cong')
    dot.edge('Gen', 'Timing')
    dot.edge('Gen', 'Power')
    dot.edge('HPWL', 'Policy', label='Quality')
    dot.edge('Cong', 'Policy')
    dot.edge('Timing', 'Policy')
    dot.edge('Power', 'Policy')
    dot.edge('Policy', 'RL', color='red', label='Feedback')
    dot.edge('Policy', 'Signal')

    dot.render('results/paper_method_validation/chip_d_rag_framework', view=False)

def main():
    """主函数"""
    print("开始生成论文图表...")
    
    # 确保输出目录存在
    os.makedirs('results/paper_method_validation', exist_ok=True)
    
    # 生成各种图表
    create_hpwl_improvement_chart()
    print("✓ HPWL Improvement Comparison已生成")
    
    create_improvement_rate_chart()
    print("✓ HPWL Improvement Rate Comparison已生成")
    
    create_execution_time_chart()
    print("✓ Execution Time Analysis已生成")
    
    create_method_comparison_chart()
    print("✓ Method Comparison已生成")
    
    create_iteration_analysis_chart()
    print("✓ Iteration Analysis已生成")
    
    create_chip_d_rag_framework_diagram()
    print('✓ Chip-D-RAG Framework Diagram generated!')
    
    print("\n所有图表已生成完成！")
    print("图表保存在: results/paper_method_validation/")

if __name__ == "__main__":
    main() 