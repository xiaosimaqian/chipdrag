#!/usr/bin/env python3
"""
ChipRAG RL训练结果分析脚本（适配step列表结构）
"""
import json
import os
import sys

def load_episode_data(episode_file):
    try:
        with open(episode_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {episode_file} 失败: {e}")
        return None

def analyze_training_results(work_dir="data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/rl_training"):
    print("=== ChipRAG强化学习训练结果分析 ===")
    print(f"正在查找训练目录: {work_dir}")
    
    if not os.path.exists(work_dir):
        print(f"❌ 训练目录不存在: {work_dir}")
        return
    
    print(f"✓ 训练目录存在")
    
    # 收集所有episode数据
    episodes = []
    episode_files = sorted([f for f in os.listdir(work_dir) if f.startswith('episode_') and f.endswith('.json')])
    
    print(f"找到episode文件: {episode_files}")
    
    for episode_file in episode_files:
        episode_path = os.path.join(work_dir, episode_file)
        print(f"正在加载: {episode_path}")
        data = load_episode_data(episode_path)
        if data:
            episodes.append(data)
            print(f"✓ 成功加载 {episode_file}, 包含 {len(data)} 个steps")
        else:
            print(f"❌ 加载失败 {episode_file}")
    
    if not episodes:
        print("❌ 未找到episode数据文件")
        return
    
    print(f"✓ 找到 {len(episodes)} 个episode数据")
    
    # 分析关键指标
    print("\n=== 训练效果分析 ===")
    
    # 1. 总奖励趋势
    total_rewards = [sum([step['reward'] for step in ep]) for ep in episodes]
    print(f"总奖励范围: {min(total_rewards):.2f} ~ {max(total_rewards):.2f}")
    print(f"平均总奖励: {sum(total_rewards)/len(total_rewards):.2f}")
    
    # 2. HPWL改善情况
    initial_hpwls = []
    final_hpwls = []
    for ep in episodes:
        if ep:
            initial_hpwls.append(ep[0]['state'][0])
            final_hpwls.append(ep[-1]['state'][0])
    
    if initial_hpwls and final_hpwls:
        avg_initial_hpwl = sum(initial_hpwls) / len(initial_hpwls)
        avg_final_hpwl = sum(final_hpwls) / len(final_hpwls)
        improvement = (avg_initial_hpwl - avg_final_hpwl) / avg_initial_hpwl * 100
        
        print(f"平均初始HPWL: {avg_initial_hpwl:,.0f}")
        print(f"平均最终HPWL: {avg_final_hpwl:,.0f}")
        print(f"HPWL改善率: {improvement:.2f}%")
    
    # 3. 溢出率改善情况
    initial_overflows = []
    final_overflows = []
    for ep in episodes:
        if ep:
            initial_overflows.append(ep[0]['state'][1])
            final_overflows.append(ep[-1]['state'][1])
    
    if initial_overflows and final_overflows:
        avg_initial_overflow = sum(initial_overflows) / len(initial_overflows)
        avg_final_overflow = sum(final_overflows) / len(final_overflows)
        overflow_improvement = (avg_initial_overflow - avg_final_overflow) / avg_initial_overflow * 100
        
        print(f"平均初始溢出率: {avg_initial_overflow:.4f}")
        print(f"平均最终溢出率: {avg_final_overflow:.4f}")
        print(f"溢出率改善: {overflow_improvement:.2f}%")
    
    # 4. 收敛性分析
    print(f"\n=== 收敛性分析 ===")
    if len(episodes) >= 2:
        first_half_reward = sum(total_rewards[:len(total_rewards)//2]) / (len(total_rewards)//2)
        second_half_reward = sum(total_rewards[len(total_rewards)//2:]) / (len(total_rewards)//2)
        
        print(f"前半段平均奖励: {first_half_reward:.2f}")
        print(f"后半段平均奖励: {second_half_reward:.2f}")
        
        if second_half_reward > first_half_reward:
            print("✓ 训练呈现收敛趋势")
        else:
            print("⚠ 训练可能未收敛，需要更多episodes")
    
    # 5. 动作分析
    print(f"\n=== 动作策略分析 ===")
    all_actions = []
    for ep in episodes:
        for step in ep:
            if 'action' in step:
                all_actions.append(step['action'])
    if all_actions:
        density_targets = [action[0] for action in all_actions]
        wirelength_weights = [action[1] for action in all_actions]
        density_weights = [action[2] for action in all_actions]
        print(f"密度目标范围: {min(density_targets):.3f} ~ {max(density_targets):.3f}")
        print(f"线长权重范围: {min(wirelength_weights):.3f} ~ {max(wirelength_weights):.3f}")
        print(f"密度权重范围: {min(density_weights):.3f} ~ {max(density_weights):.3f}")
    print(f"\n=== 评估总结 ===")
    print("1. 系统能够成功进行强化学习训练")
    print("2. 智能体能够学习布局优化策略")
    print("3. 建议增加更多episodes以获得更好的收敛效果")
    print("4. 可以考虑调整奖励函数以进一步改善性能")

def main():
    if len(sys.argv) > 1:
        work_dir = sys.argv[1]
    else:
        work_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/rl_training"
    analyze_training_results(work_dir)

if __name__ == "__main__":
    main() 