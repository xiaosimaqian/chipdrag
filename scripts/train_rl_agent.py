#!/usr/bin/env python3
"""
ChipRAG强化学习训练主脚本
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from modules.rl_training import (
    LayoutEnvironment, 
    DQNAgent, 
    RLTrainer,
    RLTrainingConfig, 
    get_default_config, 
    get_fast_config, 
    get_full_config,
    get_benchmark_configs
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(config: RLTrainingConfig, experiment_name: str = None) -> Path:
    """设置实验目录"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"rl_experiment_{timestamp}"
    
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config.save(experiment_dir / "config.json")
    
    logger.info(f"实验目录: {experiment_dir}")
    return experiment_dir

def train_single_benchmark(config: RLTrainingConfig, experiment_dir: Path):
    """训练单个benchmark"""
    logger.info(f"开始训练benchmark: {config.work_dir}")
    
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
    
    # 保存最终模型
    final_model_path = experiment_dir / "final_model.pth"
    agent.save(str(final_model_path))
    
    logger.info(f"训练完成，模型已保存: {final_model_path}")

def train_multiple_benchmarks(configs: dict, experiment_dir: Path):
    """训练多个benchmark"""
    logger.info(f"开始训练多个benchmark: {list(configs.keys())}")
    
    results = {}
    
    for benchmark_name, config in configs.items():
        logger.info(f"训练benchmark: {benchmark_name}")
        
        # 为每个benchmark创建子目录
        benchmark_dir = experiment_dir / benchmark_name
        benchmark_dir.mkdir(exist_ok=True)
        
        try:
            train_single_benchmark(config, benchmark_dir)
            results[benchmark_name] = "success"
        except Exception as e:
            logger.error(f"训练benchmark {benchmark_name} 失败: {e}")
            results[benchmark_name] = f"failed: {e}"
    
    # 保存训练结果摘要
    summary_file = experiment_dir / "training_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"多benchmark训练完成，结果: {results}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ChipRAG强化学习训练")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--benchmark", type=str, help="指定benchmark名称")
    parser.add_argument("--mode", type=str, default="default", 
                       choices=["default", "fast", "full", "all"],
                       help="训练模式")
    parser.add_argument("--experiment", type=str, help="实验名称")
    parser.add_argument("--work_dir", type=str, help="工作目录")
    parser.add_argument("--episodes", type=int, help="训练episodes数量")
    parser.add_argument("--use_openroad", action="store_true", help="使用真实OpenROAD")
    parser.add_argument("--simulate", action="store_true", help="使用模拟模式")
    
    args = parser.parse_args()
    
    # 根据模式选择配置
    if args.mode == "fast":
        config = get_fast_config()
    elif args.mode == "full":
        config = get_full_config()
    elif args.mode == "all":
        configs = get_benchmark_configs()
    else:
        config = get_default_config()
    
    # 应用命令行参数
    if args.config:
        config = RLTrainingConfig.load(args.config)
    
    if args.work_dir:
        if args.mode == "all":
            for cfg in configs.values():
                cfg.work_dir = args.work_dir
        else:
            config.work_dir = args.work_dir
    
    if args.episodes:
        if args.mode == "all":
            for cfg in configs.values():
                cfg.episodes = args.episodes
        else:
            config.episodes = args.episodes
    
    if args.use_openroad:
        if args.mode == "all":
            for cfg in configs.values():
                cfg.use_openroad = True
        else:
            config.use_openroad = True
    
    if args.simulate:
        if args.mode == "all":
            for cfg in configs.values():
                cfg.use_openroad = False
        else:
            config.use_openroad = False
    
    # 设置实验目录
    experiment_dir = setup_experiment(
        config if args.mode != "all" else list(configs.values())[0],
        args.experiment
    )
    
    # 开始训练
    if args.mode == "all":
        # 训练所有benchmark
        if args.benchmark:
            # 只训练指定的benchmark
            if args.benchmark in configs:
                train_single_benchmark(configs[args.benchmark], experiment_dir)
            else:
                logger.error(f"未找到benchmark: {args.benchmark}")
                return
        else:
            # 训练所有benchmark
            train_multiple_benchmarks(configs, experiment_dir)
    else:
        # 训练单个benchmark
        train_single_benchmark(config, experiment_dir)
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main() 