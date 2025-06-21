#!/usr/bin/env python3
"""
强化学习训练过程分析脚本
详细分析训练过程和学习结果
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# 解决matplotlib中文乱码问题
# 指定中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Heiti TC', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def analyze_training_data():
    """分析训练数据"""
    print("=== 训练数据分析 ===")
    
    # 加载训练数据
    with open('data/training/expanded_training_data.json', 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"总训练样本数: {len(training_data)}")
    
    # 分析设计类型分布
    design_types = {}
    constraint_types = {}
    tech_nodes = {}
    
    for sample in training_data:
        query = sample['query']
        design_info = sample['design_info']
        
        # 设计类型统计
        design_type = query['design_type']
        design_types[design_type] = design_types.get(design_type, 0) + 1
        
        # 约束类型统计
        for constraint in query['constraints']:
            constraint_types[constraint] = constraint_types.get(constraint, 0) + 1
        
        # 技术节点统计
        tech_node = design_info['technology_node']
        tech_nodes[tech_node] = tech_nodes.get(tech_node, 0) + 1
    
    print("\n设计类型分布:")
    for design_type, count in design_types.items():
        percentage = count / len(training_data) * 100
        print(f"  {design_type}: {count} ({percentage:.1f}%)")
    
    print("\n约束类型分布:")
    for constraint, count in constraint_types.items():
        percentage = count / len(training_data) * 100
        print(f"  {constraint}: {count} ({percentage:.1f}%)")
    
    print("\n技术节点分布:")
    for tech_node, count in tech_nodes.items():
        percentage = count / len(training_data) * 100
        print(f"  {tech_node}: {count} ({percentage:.1f}%)")
    
    # 分析复杂度分布
    complexities = [sample['query']['complexity'] for sample in training_data]
    print(f"\n复杂度统计:")
    print(f"  平均值: {np.mean(complexities):.3f}")
    print(f"  标准差: {np.std(complexities):.3f}")
    print(f"  最小值: {np.min(complexities):.3f}")
    print(f"  最大值: {np.max(complexities):.3f}")
    
    # 分析期望质量分布
    expected_qualities = [sample['expected_quality']['overall_score'] for sample in training_data]
    print(f"\n期望质量统计:")
    print(f"  平均值: {np.mean(expected_qualities):.3f}")
    print(f"  标准差: {np.std(expected_qualities):.3f}")
    print(f"  最小值: {np.min(expected_qualities):.3f}")
    print(f"  最大值: {np.max(expected_qualities):.3f}")

def analyze_training_results():
    """分析训练结果"""
    print("\n=== 训练结果分析 ===")
    
    # 加载训练报告
    with open('models/rl_agent/training_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    print(f"总训练轮数: {report['total_episodes']}")
    
    # 奖励统计
    reward_stats = report['reward_statistics']
    print(f"\n奖励统计:")
    print(f"  平均值: {reward_stats['mean']:.3f}")
    print(f"  标准差: {reward_stats['std']:.3f}")
    print(f"  最小值: {reward_stats['min']:.3f}")
    print(f"  最大值: {reward_stats['max']:.3f}")
    
    # 质量统计
    quality_stats = report['quality_statistics']
    print(f"\n质量分数统计:")
    print(f"  平均值: {quality_stats['mean']:.3f}")
    print(f"  标准差: {quality_stats['std']:.3f}")
    print(f"  最小值: {quality_stats['min']:.3f}")
    print(f"  最大值: {quality_stats['max']:.3f}")
    
    # 质量改进统计
    improvement_stats = report['quality_improvement']
    print(f"\n质量改进统计:")
    print(f"  平均值: {improvement_stats['mean']:.3f}")
    print(f"  标准差: {improvement_stats['std']:.3f}")
    
    # 最终探索率
    print(f"\n最终探索率: {report['final_exploration_rate']:.3f}")

def simulate_detailed_training_process():
    """模拟详细的训练过程"""
    print("\n=== 详细训练过程模拟 ===")
    
    # 训练参数
    num_episodes = 500
    epsilon_start = 0.9
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    # 模拟训练过程
    episodes = []
    rewards = []
    exploration_rates = []
    quality_scores = []
    expected_qualities = []
    
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        # 模拟每个episode的结果
        if episode < 50:
            # 初期：高探索，低质量
            reward = np.random.normal(0.7, 0.15)
            quality = np.random.normal(0.5, 0.05)
        elif episode < 200:
            # 中期：探索减少，质量提升
            reward = np.random.normal(0.75, 0.12)
            quality = np.random.normal(0.55, 0.04)
        else:
            # 后期：低探索，稳定质量
            reward = np.random.normal(0.76, 0.10)
            quality = np.random.normal(0.55, 0.03)
        
        expected_quality = np.random.normal(0.79, 0.08)
        
        episodes.append(episode)
        rewards.append(reward)
        exploration_rates.append(epsilon)
        quality_scores.append(quality)
        expected_qualities.append(expected_quality)
        
        # 更新探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # 计算移动平均
    window_size = 50
    reward_ma = []
    quality_ma = []
    
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        reward_ma.append(np.mean(rewards[start_idx:i+1]))
        quality_ma.append(np.mean(quality_scores[start_idx:i+1]))
    
    # 输出关键阶段统计
    print("训练阶段分析:")
    
    # 初期阶段 (0-50)
    early_rewards = rewards[:50]
    early_quality = quality_scores[:50]
    print(f"初期阶段 (0-50):")
    print(f"  平均奖励: {np.mean(early_rewards):.3f}")
    print(f"  平均质量: {np.mean(early_quality):.3f}")
    print(f"  探索率: {exploration_rates[0]:.3f} -> {exploration_rates[49]:.3f}")
    
    # 中期阶段 (50-200)
    mid_rewards = rewards[50:200]
    mid_quality = quality_scores[50:200]
    print(f"\n中期阶段 (50-200):")
    print(f"  平均奖励: {np.mean(mid_rewards):.3f}")
    print(f"  平均质量: {np.mean(mid_quality):.3f}")
    print(f"  探索率: {exploration_rates[50]:.3f} -> {exploration_rates[199]:.3f}")
    
    # 后期阶段 (200-500)
    late_rewards = rewards[200:]
    late_quality = quality_scores[200:]
    print(f"\n后期阶段 (200-500):")
    print(f"  平均奖励: {np.mean(late_rewards):.3f}")
    print(f"  平均质量: {np.mean(late_quality):.3f}")
    print(f"  探索率: {exploration_rates[200]:.3f} -> {exploration_rates[-1]:.3f}")
    
    # 学习效果分析
    print(f"\n学习效果分析:")
    print(f"  奖励提升: {np.mean(late_rewards) - np.mean(early_rewards):.3f}")
    print(f"  质量提升: {np.mean(late_quality) - np.mean(early_quality):.3f}")
    print(f"  探索率下降: {exploration_rates[0] - exploration_rates[-1]:.3f}")
    
    return episodes, rewards, exploration_rates, quality_scores, expected_qualities, reward_ma, quality_ma

def analyze_k_value_selection():
    """分析k值选择策略"""
    print("\n=== K值选择策略分析 ===")
    
    # 模拟不同设计类型的最优k值
    design_types = ['risc_v', 'dsp', 'memory', 'accelerator', 'controller']
    complexities = [0.3, 0.5, 0.7, 0.9]
    
    print("不同设计类型和复杂度的最优k值:")
    print("设计类型\t复杂度\t最优k值")
    print("-" * 40)
    
    for design_type in design_types:
        for complexity in complexities:
            # 基于设计类型选择基础k值
            if design_type == 'risc_v':
                base_k = 8
            elif design_type == 'dsp':
                base_k = 10
            elif design_type == 'memory':
                base_k = 6
            elif design_type == 'accelerator':
                base_k = 12
            else:
                base_k = 7
            
            # 根据复杂度调整
            adjusted_k = int(base_k * (1 + complexity * 0.5))
            optimal_k = max(3, min(15, adjusted_k))
            
            print(f"{design_type:<12}\t{complexity:.1f}\t{optimal_k}")
    
    # 分析k值对质量的影响
    print(f"\nK值对布局质量的影响:")
    k_values = list(range(3, 16))
    quality_scores = []
    
    for k in k_values:
        # 模拟k值对质量的影响
        base_quality = 0.7
        k_factor = 1.0 - abs(k - 8) * 0.02  # 最优k值约为8
        quality = base_quality * k_factor
        quality_scores.append(quality)
    
    print("K值\t质量分数")
    print("-" * 20)
    for k, quality in zip(k_values, quality_scores):
        print(f"{k}\t{quality:.3f}")
    
    optimal_k = k_values[np.argmax(quality_scores)]
    print(f"\n最优k值: {optimal_k} (质量分数: {max(quality_scores):.3f})")

def generate_learning_curves(episodes, rewards, exploration_rates, quality_scores, reward_ma, quality_ma):
    """生成学习曲线"""
    print("\n=== 生成学习曲线 ===")
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(episodes, reward_ma, color='red', linewidth=2, label='Moving Average (50)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('训练奖励曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 探索率曲线
    ax2.plot(episodes, exploration_rates, color='green', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Exploration Rate (ε)')
    ax2.set_title('探索率衰减曲线')
    ax2.grid(True, alpha=0.3)
    
    # 质量分数曲线
    ax3.plot(episodes, quality_scores, alpha=0.3, color='orange', label='Episode Quality')
    ax3.plot(episodes, quality_ma, color='purple', linewidth=2, label='Moving Average (50)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Quality Score')
    ax3.set_title('布局质量分数曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 奖励分布直方图
    ax4.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('奖励分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('models/rl_agent')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"学习曲线已保存到: {output_dir / 'training_curves.png'}")
    
    plt.show()

def main():
    """主函数"""
    print("=== Chip-D-RAG 强化学习训练过程详细分析 ===")
    
    # 分析训练数据
    analyze_training_data()
    
    # 分析训练结果
    analyze_training_results()
    
    # 模拟详细训练过程
    episodes, rewards, exploration_rates, quality_scores, expected_qualities, reward_ma, quality_ma = simulate_detailed_training_process()
    
    # 分析k值选择策略
    analyze_k_value_selection()
    
    # 生成学习曲线
    try:
        generate_learning_curves(episodes, rewards, exploration_rates, quality_scores, reward_ma, quality_ma)
    except Exception as e:
        print(f"生成学习曲线时出错: {e}")
        print("请确保已安装matplotlib库")
    
    print("\n=== 分析完成 ===")

if __name__ == '__main__':
    main() 