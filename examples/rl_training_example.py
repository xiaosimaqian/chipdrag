#!/usr/bin/env python3
"""
ChipRAG强化学习训练使用示例
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from modules.rl_training import (
    LayoutEnvironment, 
    DQNAgent, 
    RLTrainer,
    get_fast_config, 
    get_default_config,
    RLTrainingConfig
)

def example_fast_training():
    """快速训练示例（模拟模式）"""
    print("=== 快速训练示例（模拟模式）===")
    
    # 使用快速配置
    config = get_fast_config()
    print(f"配置: {config.work_dir}, episodes={config.episodes}, use_openroad={config.use_openroad}")
    
    # 创建环境
    env = LayoutEnvironment(
        work_dir=config.work_dir,
        max_iterations=config.max_iterations,
        target_hpwl=config.target_hpwl,
        target_overflow=config.target_overflow,
        use_openroad=config.use_openroad
    )
    
    # 创建智能体
    agent = DQNAgent(
        state_size=config.state_size,
        action_size=config.action_size,
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        target_update=config.target_update
    )
    
    # 创建训练器
    trainer = RLTrainer(
        env=env,
        agent=agent,
        episodes=config.episodes,
        max_steps=config.max_steps
    )
    
    # 开始训练
    trainer.train()
    
    print("快速训练完成！")

def example_real_training():
    """真实训练示例（使用OpenROAD）"""
    print("=== 真实训练示例（使用OpenROAD）===")
    
    # 使用默认配置
    config = get_default_config()
    config.use_openroad = True  # 确保使用真实OpenROAD
    config.episodes = 20  # 减少episodes数量用于示例
    print(f"配置: {config.work_dir}, episodes={config.episodes}, use_openroad={config.use_openroad}")
    
    # 创建环境
    env = LayoutEnvironment(
        work_dir=config.work_dir,
        max_iterations=config.max_iterations,
        target_hpwl=config.target_hpwl,
        target_overflow=config.target_overflow,
        use_openroad=config.use_openroad
    )
    
    # 创建智能体
    agent = DQNAgent(
        state_size=config.state_size,
        action_size=config.action_size,
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        target_update=config.target_update
    )
    
    # 创建训练器
    trainer = RLTrainer(
        env=env,
        agent=agent,
        episodes=config.episodes,
        max_steps=config.max_steps
    )
    
    # 开始训练
    trainer.train()
    
    print("真实训练完成！")

def example_custom_config():
    """自定义配置示例"""
    print("=== 自定义配置示例 ===")
    
    # 创建自定义配置
    config = RLTrainingConfig(
        work_dir="data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1",
        max_iterations=8,
        target_hpwl=800000.0,
        target_overflow=0.08,
        use_openroad=False,  # 使用模拟模式
        episodes=15,
        learning_rate=0.0005,
        epsilon_decay=0.99,
        reward_weights={
            "hpwl_improvement": 150.0,
            "overflow_penalty": -75.0,
            "density_reward": -15.0,
            "utilization_reward": 30.0,
            "convergence_reward": 300.0
        }
    )
    
    print(f"自定义配置: {config.work_dir}")
    print(f"目标HPWL: {config.target_hpwl}")
    print(f"目标溢出率: {config.target_overflow}")
    print(f"奖励权重: {config.reward_weights}")
    
    # 创建环境
    env = LayoutEnvironment(
        work_dir=config.work_dir,
        max_iterations=config.max_iterations,
        target_hpwl=config.target_hpwl,
        target_overflow=config.target_overflow,
        use_openroad=config.use_openroad
    )
    
    # 创建智能体
    agent = DQNAgent(
        state_size=config.state_size,
        action_size=config.action_size,
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        target_update=config.target_update
    )
    
    # 创建训练器
    trainer = RLTrainer(
        env=env,
        agent=agent,
        episodes=config.episodes,
        max_steps=config.max_steps
    )
    
    # 开始训练
    trainer.train()
    
    print("自定义配置训练完成！")

def main():
    """主函数"""
    print("ChipRAG强化学习训练示例")
    print("=" * 50)
    
    # 示例1：快速训练（模拟模式）
    try:
        example_fast_training()
    except Exception as e:
        print(f"快速训练示例失败: {e}")
    
    print("\n" + "=" * 50)
    
    # 示例2：自定义配置
    try:
        example_custom_config()
    except Exception as e:
        print(f"自定义配置示例失败: {e}")
    
    print("\n" + "=" * 50)
    
    # 示例3：真实训练（需要OpenROAD环境）
    print("注意：真实训练示例需要OpenROAD环境，如果环境不可用将跳过")
    try:
        example_real_training()
    except Exception as e:
        print(f"真实训练示例失败: {e}")
        print("这可能是因为OpenROAD环境未正确配置")
    
    print("\n所有示例完成！")
    print("\n使用说明：")
    print("1. 快速训练：使用模拟数据，适合测试和开发")
    print("2. 真实训练：使用OpenROAD进行实际布局优化")
    print("3. 自定义配置：根据具体需求调整参数")
    print("\n命令行使用：")
    print("python train_rl_agent.py --mode fast")
    print("python train_rl_agent.py --mode default --use_openroad")
    print("python evaluate_rl_agent.py --model experiments/xxx/final_model.pth --config experiments/xxx/config.json")

if __name__ == "__main__":
    main() 