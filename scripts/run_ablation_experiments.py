#!/usr/bin/env python3
"""
消融实验主控脚本
- 自动遍历所有消融配置
- 每个配置调用RL实验系统，切换功能开关
- 自动保存和汇总所有实验结果
"""
import os
import json
from pathlib import Path
import logging
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# RL实验系统主类
from rl_experiment_system import LayoutRLExperiment

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 消融配置定义 - 只运行完整系统
ABLATION_CONFIGS = [
    {
        "name": "full_system",
        "desc": "完整Chip-D-RAG系统",
        "dynamic_reranking": True,
        "entity_enhancement": True,
        "multimodal_fusion": True,
        "quality_feedback": True,
        "hierarchical_retrieval": True
    }
]

# 获取ISPD基准设计
BENCHMARK_DIR = Path("../data/designs/ispd_2015_contest_benchmark")
BENCHMARK_DESIGNS = [d for d in BENCHMARK_DIR.iterdir() if d.is_dir()]

RESULTS_ROOT = Path("results/ablation_experiments")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# 扩展RL实验系统，支持功能开关
class AblationRLExperiment(LayoutRLExperiment):
    def __init__(self, output_dir, config):
        super().__init__(output_dir)
        self.config = config
        # 这里可根据config切换各功能开关
        # 例如: self.enable_dynamic_reranking = config["dynamic_reranking"]
        # 实际功能需在RL实验系统内部实现对应开关

    # 可在此重载train_episode等方法，实现不同消融逻辑
    # 这里只做占位，具体功能需在LayoutRLExperiment内部实现


def run_full_system_experiment():
    """运行完整Chip-D-RAG系统实验"""
    summary = []
    for config in ABLATION_CONFIGS:
        logger.info(f"==== 运行完整系统实验: {config['name']} - {config['desc']} ====")
        output_dir = RESULTS_ROOT / config['name']
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行实验
        experiment = AblationRLExperiment(output_dir, config)
        results = experiment.run_experiment(BENCHMARK_DESIGNS, num_episodes=20)
        
        # 汇总主要指标
        df = pd.read_csv(output_dir / "episode_results.csv")
        avg_reward = df["episode_reward"].mean()
        avg_wirelength = df["final_wirelength"].mean()
        avg_congestion = df["final_congestion"].mean()
        avg_timing = df["final_timing"].mean()
        avg_power = df["final_power"].mean()
        summary.append({
            "config": config['name'],
            "desc": config['desc'],
            "avg_reward": avg_reward,
            "avg_wirelength": avg_wirelength,
            "avg_congestion": avg_congestion,
            "avg_timing": avg_timing,
            "avg_power": avg_power
        })
    # 保存汇总表
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(RESULTS_ROOT / "full_system_summary.csv", index=False)
    logger.info("完整系统实验完成，结果已汇总！")
    # 可视化
    plot_full_system_results(df_summary)

def plot_full_system_results(df_summary):
    plt.figure(figsize=(10,6))
    plt.bar(df_summary['config'], df_summary['avg_reward'], color='skyblue')
    plt.ylabel('平均奖励')
    plt.title('完整Chip-D-RAG系统实验结果')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "full_system_results.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_full_system_experiment() 