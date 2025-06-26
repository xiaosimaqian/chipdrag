#!/usr/bin/env python3
"""
ChipRAG强化学习系统测试脚本
"""

import sys
from pathlib import Path
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from modules.rl_training import LayoutEnvironment, DQNAgent, RLTrainer, LayoutState, LayoutAction
        print("✓ modules.rl_training 导入成功")
    except ImportError as e:
        print(f"✗ modules.rl_training 导入失败: {e}")
        return False
    
    try:
        from modules.rl_training import RLTrainingConfig, get_fast_config, get_default_config
        print("✓ RL训练配置导入成功")
    except ImportError as e:
        print(f"✗ RL训练配置导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch 导入成功 (版本: {torch.__version__})")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False
    
    return True

def test_config():
    """测试配置系统"""
    print("\n=== 测试配置系统 ===")
    
    try:
        from modules.rl_training import RLTrainingConfig, get_fast_config
        
        # 测试快速配置
        config = get_fast_config()
        print(f"✓ 快速配置创建成功: episodes={config.episodes}, use_openroad={config.use_openroad}")
        
        # 测试自定义配置
        custom_config = RLTrainingConfig(
            work_dir="test_dir",
            episodes=5,
            use_openroad=False
        )
        print(f"✓ 自定义配置创建成功: work_dir={custom_config.work_dir}")
        
        # 测试配置保存和加载
        config.save("test_config.json")
        loaded_config = RLTrainingConfig.load("test_config.json")
        print("✓ 配置保存和加载成功")
        
        # 清理测试文件
        Path("test_config.json").unlink(missing_ok=True)
        
        return True
    except Exception as e:
        print(f"✗ 配置系统测试失败: {e}")
        return False

def test_environment():
    """测试环境创建"""
    print("\n=== 测试环境创建 ===")
    
    try:
        from modules.rl_training import LayoutEnvironment, get_fast_config
        
        config = get_fast_config()
        
        # 创建环境
        env = LayoutEnvironment(
            work_dir=config.work_dir,
            max_iterations=config.max_iterations,
            target_hpwl=config.target_hpwl,
            target_overflow=config.target_overflow,
            use_openroad=config.use_openroad
        )
        print("✓ 环境创建成功")
        
        # 测试环境重置
        state = env.reset()
        print(f"✓ 环境重置成功: HPWL={state.hpwl:.2f}, 溢出率={state.overflow:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        return False

def test_agent():
    """测试智能体创建"""
    print("\n=== 测试智能体创建 ===")
    
    try:
        from modules.rl_training import DQNAgent, LayoutAction, get_fast_config
        import torch
        
        config = get_fast_config()
        
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
        print("✓ 智能体创建成功")
        
        # 测试动作选择
        import numpy as np
        test_state = np.array([1000000.0, 0.2, 0.8, 0.7, 1], dtype=np.float32)
        action = agent.act(test_state)
        print(f"✓ 动作选择成功: {action}")
        
        # 测试经验存储
        next_state = np.array([950000.0, 0.15, 0.85, 0.75, 2], dtype=np.float32)
        agent.remember(test_state, action, 50.0, next_state, False)
        print("✓ 经验存储成功")
        
        return True
    except Exception as e:
        print(f"✗ 智能体测试失败: {e}")
        return False

def test_training_loop():
    """测试训练循环"""
    print("\n=== 测试训练循环 ===")
    
    try:
        from modules.rl_training import LayoutEnvironment, DQNAgent, RLTrainer, LayoutAction, get_fast_config
        
        config = get_fast_config()
        config.episodes = 2  # 只测试2个episodes
        
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
        print("✓ 训练循环测试成功")
        
        return True
    except Exception as e:
        print(f"✗ 训练循环测试失败: {e}")
        return False

def test_openroad_integration():
    """测试OpenROAD集成"""
    print("\n=== 测试OpenROAD集成 ===")
    
    try:
        from modules.rl_training import LayoutEnvironment, get_default_config
        
        config = get_default_config()
        config.use_openroad = True
        config.episodes = 1
        config.max_iterations = 2
        
        # 尝试创建使用OpenROAD的环境
        env = LayoutEnvironment(
            work_dir=config.work_dir,
            max_iterations=config.max_iterations,
            target_hpwl=config.target_hpwl,
            target_overflow=config.target_overflow,
            use_openroad=config.use_openroad
        )
        
        if env.use_openroad:
            print("✓ OpenROAD集成测试成功（环境可用）")
        else:
            print("⚠ OpenROAD集成测试：环境不可用，使用模拟模式")
        
        return True
    except Exception as e:
        print(f"✗ OpenROAD集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("ChipRAG强化学习系统测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("配置系统", test_config),
        ("环境创建", test_environment),
        ("智能体创建", test_agent),
        ("训练循环", test_training_loop),
        ("OpenROAD集成", test_openroad_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关配置和依赖。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 