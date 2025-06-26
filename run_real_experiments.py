#!/usr/bin/env python3
"""
基于真实数据的完整实验系统
整合RL训练结果、实验评估数据和基准测试数据
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

class RealDataExperimentSystem:
    """基于真实数据的实验系统"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.rl_training_dir = self.data_dir / "designs/ispd_2015_contest_benchmark/mgc_des_perf_1/rl_training"
        
        # 创建输出目录
        self.output_dir = self.base_dir / "real_experiment_output"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"实验系统初始化完成，输出目录: {self.output_dir}")
    
    def load_real_rl_data(self) -> Dict[str, Any]:
        """加载真实的RL训练数据"""
        logger.info("加载真实RL训练数据...")
        
        # 加载训练历史
        training_history = pd.read_csv(self.rl_training_dir / "training_history.csv")
        
        # 加载episode数据
        episodes = {}
        for i in range(1, 6):  # 5个episode
            episode_file = self.rl_training_dir / f"episode_{i}.json"
            if episode_file.exists():
                with open(episode_file, 'r') as f:
                    episodes[f"episode_{i}"] = json.load(f)
        
        # 分析训练曲线
        training_analysis = {
            'total_episodes': len(training_history),
            'avg_reward': float(training_history['total_reward'].mean()),
            'reward_std': float(training_history['total_reward'].std()),
            'best_episode': training_history.loc[training_history['total_reward'].idxmax()].to_dict(),
            'worst_episode': training_history.loc[training_history['total_reward'].idxmin()].to_dict(),
            'convergence_analysis': self._analyze_convergence(training_history),
            'hpwl_improvement': self._analyze_hpwl_improvement(training_history),
            'overflow_analysis': self._analyze_overflow(training_history)
        }
        
        return {
            'training_history': training_history.to_dict('records'),
            'episodes': episodes,
            'analysis': training_analysis
        }
    
    def load_real_experiment_results(self) -> Dict[str, Any]:
        """加载真实的实验评估结果"""
        logger.info("加载真实实验评估结果...")
        
        # 找到最新的实验结果
        result_files = list(self.results_dir.glob("experiment_results_*.json"))
        if not result_files:
            logger.warning("未找到实验结果文件")
            return {}
        
        # 按时间排序，取最新的
        latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"使用最新实验结果: {latest_result.name}")
        
        with open(latest_result, 'r') as f:
            experiment_data = json.load(f)
        
        # 加载实验摘要
        summary_file = self.results_dir / "experiment_summary.csv"
        if summary_file.exists():
            summary_data = pd.read_csv(summary_file)
        else:
            summary_data = pd.DataFrame()
        
        return {
            'experiment_data': experiment_data,
            'summary_data': summary_data.to_dict('records') if not summary_data.empty else [],
            'file_info': {
                'latest_result': latest_result.name,
                'total_results': len(result_files)
            }
        }
    
    def load_benchmark_data(self) -> Dict[str, Any]:
        """加载基准测试数据"""
        logger.info("加载基准测试数据...")
        
        benchmark_dir = self.data_dir / "designs/ispd_2015_contest_benchmark"
        benchmarks = {}
        
        for benchmark in ['mgc_des_perf_1', 'mgc_fft_1', 'mgc_pci_bridge32_a']:
            benchmark_path = benchmark_dir / benchmark
            if benchmark_path.exists():
                # 检查是否有DEF文件
                def_files = list(benchmark_path.glob("*.def"))
                lib_files = list(benchmark_path.glob("*.lib"))
                lef_files = list(benchmark_path.glob("*.lef"))
                
                benchmarks[benchmark] = {
                    'def_files': [f.name for f in def_files],
                    'lib_files': [f.name for f in lib_files],
                    'lef_files': [f.name for f in lef_files],
                    'has_rl_training': (benchmark_path / "rl_training").exists(),
                    'total_files': len(def_files) + len(lib_files) + len(lef_files)
                }
        
        return benchmarks
    
    def _analyze_convergence(self, training_history: pd.DataFrame) -> Dict[str, Any]:
        """分析训练收敛性"""
        rewards = training_history['total_reward'].values
        
        # 计算收敛指标
        convergence = {
            'reward_trend': 'increasing' if rewards[-1] > rewards[0] else 'decreasing',
            'reward_variance': np.var(rewards),
            'final_reward': rewards[-1],
            'initial_reward': rewards[0],
            'improvement_ratio': (rewards[-1] - rewards[0]) / abs(rewards[0]) if rewards[0] != 0 else 0
        }
        
        return convergence
    
    def _analyze_hpwl_improvement(self, training_history: pd.DataFrame) -> Dict[str, Any]:
        """分析HPWL改进情况"""
        hpwl_values = training_history['final_hpwl'].values
        
        improvement = {
            'initial_hpwl': hpwl_values[0],
            'final_hpwl': hpwl_values[-1],
            'best_hpwl': hpwl_values.min(),
            'hpwl_reduction': (hpwl_values[0] - hpwl_values[-1]) / hpwl_values[0] * 100,
            'hpwl_std': np.std(hpwl_values)
        }
        
        return improvement
    
    def _analyze_overflow(self, training_history: pd.DataFrame) -> Dict[str, Any]:
        """分析overflow情况"""
        overflow_values = training_history['final_overflow'].values
        
        overflow_analysis = {
            'initial_overflow': overflow_values[0],
            'final_overflow': overflow_values[-1],
            'best_overflow': overflow_values.min(),
            'overflow_improvement': (overflow_values[0] - overflow_values[-1]) / overflow_values[0] * 100 if overflow_values[0] != 0 else 0,
            'overflow_std': np.std(overflow_values)
        }
        
        return overflow_analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合实验报告"""
        logger.info("生成综合实验报告...")
        
        # 加载所有真实数据
        rl_data = self.load_real_rl_data()
        experiment_data = self.load_real_experiment_results()
        benchmark_data = self.load_benchmark_data()
        
        # 生成综合报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'experiment_summary': {
                'rl_training_episodes': len(rl_data.get('training_history', [])),
                'experiment_results_count': experiment_data.get('file_info', {}).get('total_results', 0),
                'available_benchmarks': len(benchmark_data),
                'total_benchmark_files': sum(b['total_files'] for b in benchmark_data.values())
            },
            'rl_training_analysis': rl_data.get('analysis', {}),
            'experiment_performance': self._analyze_experiment_performance(experiment_data),
            'benchmark_coverage': benchmark_data,
            'system_capabilities': self._assess_system_capabilities(rl_data, experiment_data, benchmark_data)
        }
        
        # 保存报告（修复序列化问题）
        report_serializable = to_serializable(report)
        report_file = self.output_dir / "comprehensive_experiment_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"综合报告已保存: {report_file}")
        return report
    
    def _analyze_experiment_performance(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验性能"""
        summary_data = experiment_data.get('summary_data', [])
        
        if not summary_data:
            return {'status': 'no_data_available'}
        
        # 转换为DataFrame进行分析
        df = pd.DataFrame(summary_data)
        
        performance = {
            'total_cases': len(df),
            'feasible_rate': (df['feasible'] == True).mean() * 100 if 'feasible' in df.columns else 0,
            'avg_timing_margin': df['timing_margin'].mean() if 'timing_margin' in df.columns else 0,
            'avg_area_utilization': df['area'].mean() if 'area' in df.columns else 0,
            'avg_power_consumption': df['power'].mean() if 'power' in df.columns else 0,
            'avg_overall_score': df['score'].mean() if 'score' in df.columns else 0,
            'avg_knowledge_reuse': df['reuse_rate'].mean() if 'reuse_rate' in df.columns else 0
        }
        
        return performance
    
    def _assess_system_capabilities(self, rl_data: Dict, experiment_data: Dict, benchmark_data: Dict) -> Dict[str, Any]:
        """评估系统能力"""
        capabilities = {
            'rl_training': {
                'implemented': bool(rl_data.get('training_history')),
                'episodes_completed': len(rl_data.get('training_history', [])),
                'convergence_achieved': rl_data.get('analysis', {}).get('convergence_analysis', {}).get('improvement_ratio', 0) > 0
            },
            'experiment_evaluation': {
                'implemented': bool(experiment_data.get('experiment_data')),
                'results_available': experiment_data.get('file_info', {}).get('total_results', 0) > 0,
                'performance_metrics': bool(experiment_data.get('summary_data'))
            },
            'benchmark_support': {
                'benchmarks_available': len(benchmark_data),
                'design_files_present': sum(b['total_files'] for b in benchmark_data.values()) > 0,
                'rl_training_data': any(b.get('has_rl_training') for b in benchmark_data.values())
            },
            'overall_readiness': {
                'ready_for_paper': self._assess_paper_readiness(rl_data, experiment_data, benchmark_data),
                'missing_components': self._identify_missing_components(rl_data, experiment_data, benchmark_data)
            }
        }
        
        return capabilities
    
    def _assess_paper_readiness(self, rl_data: Dict, experiment_data: Dict, benchmark_data: Dict) -> Dict[str, Any]:
        """评估论文准备就绪程度"""
        readiness = {
            'rl_training_results': bool(rl_data.get('training_history')),
            'experiment_evaluation': bool(experiment_data.get('experiment_data')),
            'benchmark_coverage': len(benchmark_data) >= 3,  # 至少3个基准测试
            'performance_metrics': bool(experiment_data.get('summary_data')),
            'convergence_analysis': bool(rl_data.get('analysis', {}).get('convergence_analysis'))
        }
        
        # 计算总体就绪度
        readiness['overall_score'] = sum(readiness.values()) / len(readiness) * 100
        
        return readiness
    
    def _identify_missing_components(self, rl_data: Dict, experiment_data: Dict, benchmark_data: Dict) -> List[str]:
        """识别缺失的组件"""
        missing = []
        
        if not rl_data.get('training_history'):
            missing.append("RL训练历史数据")
        
        if not experiment_data.get('experiment_data'):
            missing.append("实验评估结果")
        
        if len(benchmark_data) < 3:
            missing.append("足够的基准测试数据")
        
        if not experiment_data.get('summary_data'):
            missing.append("实验性能摘要")
        
        return missing
    
    def generate_visualizations(self):
        """生成可视化图表"""
        logger.info("生成可视化图表...")
        
        # 加载数据
        rl_data = self.load_real_rl_data()
        experiment_data = self.load_real_experiment_results()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chip-D-RAG系统真实实验数据可视化', fontsize=16, fontweight='bold')
        
        # 1. RL训练奖励曲线
        if rl_data.get('training_history'):
            training_df = pd.DataFrame(rl_data['training_history'])
            axes[0, 0].plot(training_df['episode'], training_df['total_reward'], 'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_title('RL训练奖励曲线')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('总奖励')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. HPWL改进情况
        if rl_data.get('training_history'):
            training_df = pd.DataFrame(rl_data['training_history'])
            axes[0, 1].plot(training_df['episode'], training_df['final_hpwl'], 'r-s', linewidth=2, markersize=6)
            axes[0, 1].set_title('HPWL优化情况')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('最终HPWL')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 实验性能指标
        if experiment_data.get('summary_data'):
            summary_df = pd.DataFrame(experiment_data['summary_data'])
            if 'score' in summary_df.columns:
                axes[1, 0].bar(summary_df['case'], summary_df['score'], color='green', alpha=0.7)
                axes[1, 0].set_title('各案例总体评分')
                axes[1, 0].set_xlabel('测试案例')
                axes[1, 0].set_ylabel('评分')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 约束满足情况
        if experiment_data.get('summary_data'):
            summary_df = pd.DataFrame(experiment_data['summary_data'])
            if 'timing_margin' in summary_df.columns and 'area' in summary_df.columns:
                axes[1, 1].scatter(summary_df['area'], summary_df['timing_margin'], 
                                 s=100, c='purple', alpha=0.7)
                axes[1, 1].set_title('面积利用率 vs 时序裕量')
                axes[1, 1].set_xlabel('面积利用率')
                axes[1, 1].set_ylabel('时序裕量')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / "experiment_visualizations.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logger.info(f"可视化图表已保存: {chart_file}")
        
        plt.show()
    
    def run_complete_analysis(self):
        """运行完整分析"""
        logger.info("开始运行完整实验分析...")
        
        # 生成综合报告
        report = self.generate_comprehensive_report()
        
        # 生成可视化
        self.generate_visualizations()
        
        # 打印关键结果
        self._print_key_results(report)
        
        logger.info("完整实验分析完成！")
        return report
    
    def _print_key_results(self, report: Dict[str, Any]):
        """打印关键结果"""
        print("\n" + "="*60)
        print("Chip-D-RAG系统真实实验数据关键结果")
        print("="*60)
        
        # RL训练结果
        rl_analysis = report.get('rl_training_analysis', {})
        if rl_analysis:
            print(f"\n🔧 RL训练结果:")
            print(f"   - 训练轮数: {report['experiment_summary']['rl_training_episodes']}")
            print(f"   - 平均奖励: {rl_analysis.get('avg_reward', 0):.2f}")
            print(f"   - HPWL改进: {rl_analysis.get('hpwl_improvement', {}).get('hpwl_reduction', 0):.1f}%")
            print(f"   - 收敛状态: {rl_analysis.get('convergence_analysis', {}).get('reward_trend', 'unknown')}")
        
        # 实验性能
        exp_performance = report.get('experiment_performance', {})
        if exp_performance:
            print(f"\n📊 实验性能:")
            print(f"   - 测试案例数: {exp_performance.get('total_cases', 0)}")
            print(f"   - 可行率: {exp_performance.get('feasible_rate', 0):.1f}%")
            print(f"   - 平均时序裕量: {exp_performance.get('avg_timing_margin', 0):.3f}")
            print(f"   - 平均总体评分: {exp_performance.get('avg_overall_score', 0):.3f}")
        
        # 系统能力
        capabilities = report.get('system_capabilities', {})
        if capabilities:
            readiness = capabilities.get('overall_readiness', {})
            print(f"\n🎯 系统就绪度:")
            print(f"   - 总体就绪度: {readiness.get('overall_score', 0):.1f}%")
            print(f"   - 基准测试覆盖: {len(report.get('benchmark_coverage', {}))}")
            print(f"   - 实验结果可用: {capabilities.get('experiment_evaluation', {}).get('results_available', 0)}")
        
        # 缺失组件
        missing = capabilities.get('overall_readiness', {}).get('missing_components', [])
        if missing:
            print(f"\n⚠️  缺失组件:")
            for component in missing:
                print(f"   - {component}")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    # 创建实验系统
    experiment_system = RealDataExperimentSystem()
    
    # 运行完整分析
    report = experiment_system.run_complete_analysis()
    
    print(f"\n📁 所有结果已保存到: {experiment_system.output_dir}")
    print("📄 详细报告: comprehensive_experiment_report.json")
    print("📊 可视化图表: experiment_visualizations.png")

if __name__ == "__main__":
    main() 