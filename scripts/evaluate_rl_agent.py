#!/usr/bin/env python3
"""
ChipRAG强化学习智能体评估脚本
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from modules.rl_training import LayoutEnvironment, DQNAgent, LayoutAction, RLTrainingConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLAgentEvaluator:
    """RL智能体评估器"""
    
    def __init__(self, model_path: str, config: RLTrainingConfig):
        """
        Args:
            model_path: 模型文件路径
            config: 训练配置
        """
        self.model_path = Path(model_path)
        self.config = config
        
        # 创建环境
        self.env = LayoutEnvironment(
            work_dir=config.work_dir,
            max_iterations=config.max_iterations,
            target_hpwl=config.target_hpwl,
            target_overflow=config.target_overflow,
            use_openroad=config.use_openroad
        )
        
        # 创建智能体
        self.agent = DQNAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            learning_rate=config.learning_rate,
            epsilon=0.0,  # 评估时不探索
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            memory_size=config.memory_size,
            batch_size=config.batch_size,
            gamma=config.gamma,
            target_update=config.target_update
        )
        
        # 加载模型
        self.agent.load(str(self.model_path))
        logger.info(f"模型已加载: {self.model_path}")
    
    def evaluate_single_episode(self, episode_id: int = 0) -> dict:
        """评估单个episode"""
        logger.info(f"开始评估episode {episode_id}")
        
        # 重置环境
        state = self.env.reset()
        total_reward = 0
        episode_data = []
        
        for step in range(self.config.max_steps):
            # 选择动作（不探索）
            action_array = self.agent.act(state.to_array())
            action = LayoutAction.from_array(action_array)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录数据
            step_data = {
                "step": step + 1,
                "state": state.to_array().tolist(),
                "action": action_array.tolist(),
                "reward": reward,
                "hpwl": info.get("hpwl", 0),
                "overflow": info.get("overflow", 0),
                "density": info.get("density", 0),
                "utilization": info.get("utilization", 0)
            }
            episode_data.append(step_data)
            
            total_reward += reward
            state = next_state
            
            logger.info(f"Step {step + 1}: 奖励={reward:.2f}, "
                       f"HPWL={info.get('hpwl', 0):.0f}, "
                       f"溢出率={info.get('overflow', 0):.4f}")
            
            if done:
                break
        
        # 计算评估指标
        final_hpwl = episode_data[-1]["hpwl"] if episode_data else 0
        final_overflow = episode_data[-1]["overflow"] if episode_data else 0
        final_density = episode_data[-1]["density"] if episode_data else 0
        final_utilization = episode_data[-1]["utilization"] if episode_data else 0
        
        evaluation_result = {
            "episode_id": episode_id,
            "total_reward": total_reward,
            "steps": len(episode_data),
            "final_hpwl": final_hpwl,
            "final_overflow": final_overflow,
            "final_density": final_density,
            "final_utilization": final_utilization,
            "hpwl_improvement": (self.config.target_hpwl - final_hpwl) / self.config.target_hpwl,
            "overflow_improvement": (self.config.target_overflow - final_overflow) / self.config.target_overflow,
            "episode_data": episode_data
        }
        
        logger.info(f"Episode {episode_id} 评估完成: "
                   f"总奖励={total_reward:.2f}, "
                   f"最终HPWL={final_hpwl:.0f}, "
                   f"最终溢出率={final_overflow:.4f}")
        
        return evaluation_result
    
    def evaluate_multiple_episodes(self, num_episodes: int = 10) -> list:
        """评估多个episodes"""
        logger.info(f"开始评估 {num_episodes} 个episodes")
        
        results = []
        for episode_id in range(num_episodes):
            try:
                result = self.evaluate_single_episode(episode_id)
                results.append(result)
            except Exception as e:
                logger.error(f"评估episode {episode_id} 失败: {e}")
                continue
        
        return results
    
    def generate_evaluation_report(self, results: list, output_dir: Path):
        """生成评估报告"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 计算统计指标
        total_rewards = [r["total_reward"] for r in results]
        final_hpwls = [r["final_hpwl"] for r in results]
        final_overflows = [r["final_overflow"] for r in results]
        final_densities = [r["final_density"] for r in results]
        final_utilizations = [r["final_utilization"] for r in results]
        
        stats = {
            "num_episodes": len(results),
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "avg_final_hpwl": np.mean(final_hpwls),
            "std_final_hpwl": np.std(final_hpwls),
            "avg_final_overflow": np.mean(final_overflows),
            "std_final_overflow": np.std(final_overflows),
            "avg_final_density": np.mean(final_densities),
            "std_final_density": np.std(final_densities),
            "avg_final_utilization": np.mean(final_utilizations),
            "std_final_utilization": np.std(final_utilizations),
            "best_hpwl": min(final_hpwls),
            "worst_hpwl": max(final_hpwls),
            "best_overflow": min(final_overflows),
            "worst_overflow": max(final_overflows)
        }
        
        # 保存统计结果
        stats_file = output_dir / "evaluation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 生成可视化图表
        self._plot_evaluation_results(results, stats, output_dir)
        
        logger.info(f"评估报告已生成: {output_dir}")
        return stats
    
    def _plot_evaluation_results(self, results: list, stats: dict, output_dir: Path):
        """绘制评估结果图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 提取数据
        episode_ids = [r["episode_id"] for r in results]
        total_rewards = [r["total_reward"] for r in results]
        final_hpwls = [r["final_hpwl"] for r in results]
        final_overflows = [r["final_overflow"] for r in results]
        final_densities = [r["final_density"] for r in results]
        final_utilizations = [r["final_utilization"] for r in results]
        
        # 总奖励分布
        axes[0, 0].bar(episode_ids, total_rewards)
        axes[0, 0].set_title('Total Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].axhline(y=stats["avg_total_reward"], color='r', linestyle='--', 
                          label=f'Avg: {stats["avg_total_reward"]:.2f}')
        axes[0, 0].legend()
        
        # HPWL分布
        axes[0, 1].bar(episode_ids, final_hpwls)
        axes[0, 1].set_title('Final HPWL per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('HPWL')
        axes[0, 1].axhline(y=self.config.target_hpwl, color='g', linestyle='--', 
                          label=f'Target: {self.config.target_hpwl}')
        axes[0, 1].legend()
        
        # 溢出率分布
        axes[0, 2].bar(episode_ids, final_overflows)
        axes[0, 2].set_title('Final Overflow per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Overflow')
        axes[0, 2].axhline(y=self.config.target_overflow, color='g', linestyle='--', 
                          label=f'Target: {self.config.target_overflow}')
        axes[0, 2].legend()
        
        # 密度分布
        axes[1, 0].bar(episode_ids, final_densities)
        axes[1, 0].set_title('Final Density per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Density')
        
        # 利用率分布
        axes[1, 1].bar(episode_ids, final_utilizations)
        axes[1, 1].set_title('Final Utilization per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Utilization')
        
        # 奖励vs HPWL散点图
        axes[1, 2].scatter(final_hpwls, total_rewards)
        axes[1, 2].set_title('Reward vs HPWL')
        axes[1, 2].set_xlabel('Final HPWL')
        axes[1, 2].set_ylabel('Total Reward')
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = output_dir / "evaluation_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"评估图表已保存: {plot_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ChipRAG强化学习智能体评估")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--episodes", type=int, default=10, help="评估episodes数量")
    parser.add_argument("--output", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    config = RLTrainingConfig.load(args.config)
    
    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("evaluation_results") / f"eval_{Path(args.model).stem}"
    
    # 创建评估器
    evaluator = RLAgentEvaluator(args.model, config)
    
    # 执行评估
    results = evaluator.evaluate_multiple_episodes(args.episodes)
    
    # 生成报告
    stats = evaluator.generate_evaluation_report(results, output_dir)
    
    # 打印摘要
    print("\n=== 评估摘要 ===")
    print(f"评估episodes数量: {stats['num_episodes']}")
    print(f"平均总奖励: {stats['avg_total_reward']:.2f} ± {stats['std_total_reward']:.2f}")
    print(f"平均最终HPWL: {stats['avg_final_hpwl']:.0f} ± {stats['std_final_hpwl']:.0f}")
    print(f"平均最终溢出率: {stats['avg_final_overflow']:.4f} ± {stats['std_final_overflow']:.4f}")
    print(f"最佳HPWL: {stats['best_hpwl']:.0f}")
    print(f"最佳溢出率: {stats['best_overflow']:.4f}")
    
    logger.info("评估完成！")

if __name__ == "__main__":
    main() 