import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self, results_dir="output"):
        self.results_dir = Path(results_dir)
        
    def analyze_experiment(self, experiment_dir):
        """分析单个实验结果"""
        experiment_dir = Path(experiment_dir)
        
        # 加载结果
        with open(experiment_dir / "results.json", "r") as f:
            results = json.load(f)
        
        # 分析布局质量
        quality_metrics = self._analyze_quality(results["quality_results"])
        
        # 分析约束满足度
        constraint_metrics = self._analyze_constraints(results["constraint_results"])
        
        # 分析多目标评分
        objective_metrics = self._analyze_objectives(results["objective_results"])
        
        # 生成分析报告
        report = {
            "benchmark": results["benchmark"],
            "timestamp": results["timestamp"],
            "quality_metrics": quality_metrics,
            "constraint_metrics": constraint_metrics,
            "objective_metrics": objective_metrics
        }
        
        # 保存分析报告
        report_file = experiment_dir / "analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # 生成可视化图表
        self._generate_visualizations(experiment_dir, report)
        
        return report
    
    def _analyze_quality(self, quality_results):
        """分析布局质量指标"""
        return {
            "overall_score": quality_results["score"],
            "timing_score": quality_results.get("timing_score", 0),
            "power_score": quality_results.get("power_score", 0),
            "area_score": quality_results.get("area_score", 0),
            "congestion_score": quality_results.get("congestion_score", 0)
        }
    
    def _analyze_constraints(self, constraint_results):
        """分析约束满足度指标"""
        return {
            "satisfaction_rate": constraint_results["satisfaction_rate"],
            "timing_constraints": constraint_results.get("timing_constraints", {}),
            "power_constraints": constraint_results.get("power_constraints", {}),
            "area_constraints": constraint_results.get("area_constraints", {})
        }
    
    def _analyze_objectives(self, objective_results):
        """分析多目标评分指标"""
        return {
            "total_score": objective_results["total_score"],
            "timing_objective": objective_results.get("timing_objective", 0),
            "power_objective": objective_results.get("power_objective", 0),
            "area_objective": objective_results.get("area_objective", 0)
        }
    
    def _generate_visualizations(self, experiment_dir, report):
        """生成可视化图表"""
        # 创建图表目录
        plots_dir = experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 布局质量雷达图
        self._plot_quality_radar(plots_dir, report["quality_metrics"])
        
        # 2. 约束满足度柱状图
        self._plot_constraint_bars(plots_dir, report["constraint_metrics"])
        
        # 3. 多目标评分折线图
        self._plot_objective_lines(plots_dir, report["objective_metrics"])
    
    def _plot_quality_radar(self, plots_dir, quality_metrics):
        """绘制布局质量雷达图"""
        metrics = ["timing_score", "power_score", "area_score", "congestion_score"]
        values = [quality_metrics[m] for m in metrics]
        
        # 设置雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title("布局质量评估")
        
        plt.savefig(plots_dir / "quality_radar.png")
        plt.close()
    
    def _plot_constraint_bars(self, plots_dir, constraint_metrics):
        """绘制约束满足度柱状图"""
        constraints = list(constraint_metrics.keys())
        values = [constraint_metrics[c] for c in constraints]
        
        plt.figure(figsize=(10, 6))
        plt.bar(constraints, values)
        plt.title("约束满足度分析")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(plots_dir / "constraint_bars.png")
        plt.close()
    
    def _plot_objective_lines(self, plots_dir, objective_metrics):
        """绘制多目标评分折线图"""
        objectives = ["timing_objective", "power_objective", "area_objective"]
        values = [objective_metrics[o] for o in objectives]
        
        plt.figure(figsize=(10, 6))
        plt.plot(objectives, values, marker='o')
        plt.title("多目标评分分析")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(plots_dir / "objective_lines.png")
        plt.close()

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行分析
    analyzer = ResultsAnalyzer()
    experiment_dir = "output/experiment_latest"  # 替换为实际的实验目录
    report = analyzer.analyze_experiment(experiment_dir)
    
    # 打印分析结果摘要
    print("\n分析结果摘要:")
    print(f"基准测试: {report['benchmark']}")
    print(f"总体质量评分: {report['quality_metrics']['overall_score']}")
    print(f"约束满足度: {report['constraint_metrics']['satisfaction_rate']}")
    print(f"多目标总评分: {report['objective_metrics']['total_score']}") 