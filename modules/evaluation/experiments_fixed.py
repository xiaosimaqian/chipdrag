import os
import time
import json
import numpy as np
import logging
from typing import Dict, List, Any
from datetime import datetime
from modules.utils.benchmark_loader import BenchmarkLoader
from modules.core.rag_system import RAGSystem
from chiprag.config.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

class ExperimentEvaluator:
    def __init__(self):
        """初始化实验评估器"""
        try:
            # 加载实验配置
            self.config = ExperimentConfig()
            
            # 验证实验参数
            self._validate_experiment_params()
            
            # 初始化基准测试加载器
            self.benchmark_loader = BenchmarkLoader(self.config.benchmark_path)
            
            # 初始化RAG系统
            self.rag_system = None  # 暂时不初始化，等待需要时再初始化
            
            # 设置随机种子
            np.random.seed(self.config.experiment_params['seed'])
            
        except Exception as e:
            logger.error(f"初始化实验评估器失败: {e}")
            raise
            
    def _validate_experiment_params(self):
        """验证实验参数"""
        required_params = ['num_runs', 'seed']
        for param in required_params:
            if param not in self.config.experiment_params:
                raise ValueError(f"缺少必需的实验参数: {param}")
            if not isinstance(self.config.experiment_params[param], (int, float)):
                raise ValueError(f"实验参数 {param} 必须是数值类型")
                
    def run_all_experiments(self) -> Dict:
        """运行所有实验
        
        Returns:
            Dict: 实验结果
        """
        try:
            results = {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'config': self.config.to_dict(),
                'benchmarks': {}
            }
            
            # 遍历所有基准测试
            for benchmark_name in self.config.benchmarks:
                try:
                    logger.info(f"开始处理基准测试: {benchmark_name}")
                    
                    # 加载设计信息
                    design_info = self.benchmark_loader.load_design(benchmark_name)
                    
                    # 初始化RAG系统
                    if self.rag_system is None:
                        self.rag_system = RAGSystem(self.config.rag_config)
                    
                    # 运行实验
                    benchmark_results = self._run_benchmark_experiments(
                        benchmark_name,
                        design_info
                    )
                    
                    results['benchmarks'][benchmark_name] = benchmark_results
                    
                except Exception as e:
                    logger.error(f"处理基准测试 {benchmark_name} 失败: {e}")
                    results['benchmarks'][benchmark_name] = {
                        'error': str(e)
                    }
            
            # 保存结果
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"运行实验失败: {e}")
            raise
            
    def _run_benchmark_experiments(
        self,
        benchmark_name: str,
        design_info: Dict
    ) -> Dict:
        """运行单个基准测试的实验
        
        Args:
            benchmark_name: 基准测试名称
            design_info: 设计信息
            
        Returns:
            Dict: 实验结果
        """
        try:
            results = {
                'design_info': design_info,
                'experiments': {}
            }
            
            # 运行多次实验
            for run_id in range(self.config.experiment_params['num_runs']):
                try:
                    logger.info(f"运行实验 {run_id + 1}/{self.config.experiment_params['num_runs']}")
                    
                    # 使用RAG系统生成布局
                    layout = self.rag_system.generate_layout(design_info)
                    
                    # 评估布局质量
                    quality_metrics = self._evaluate_layout_quality(layout, design_info)
                    
                    # 评估约束满足度
                    constraint_metrics = self._evaluate_constraint_satisfaction(
                        layout,
                        design_info
                    )
                    
                    # 评估优化效率
                    efficiency_metrics = self._evaluate_optimization_efficiency(
                        layout,
                        design_info
                    )
                    
                    # 记录结果
                    results['experiments'][f'run_{run_id}'] = {
                        'quality': quality_metrics,
                        'constraints': constraint_metrics,
                        'efficiency': efficiency_metrics
                    }
                    
                except Exception as e:
                    logger.error(f"运行实验 {run_id} 失败: {e}")
                    results['experiments'][f'run_{run_id}'] = {
                        'error': str(e)
                    }
            
            # 计算平均值
            results['averages'] = self._calculate_averages(results['experiments'])
            
            return results
            
        except Exception as e:
            logger.error(f"运行基准测试实验失败: {e}")
            raise
            
    def _evaluate_layout_quality(
        self,
        layout: Dict,
        design_info: Dict
    ) -> Dict:
        """评估布局质量
        
        Args:
            layout: 布局信息
            design_info: 设计信息
            
        Returns:
            Dict: 质量评估指标
        """
        try:
            # 计算密度
            density = self._calculate_density(layout, design_info)
            
            # 计算拥塞度
            congestion = self._calculate_congestion(layout, design_info)
            
            # 计算时序裕量
            timing_margin = self._calculate_timing_margin(layout, design_info)
            
            return {
                'density': density,
                'congestion': congestion,
                'timing_margin': timing_margin
            }
            
        except Exception as e:
            logger.error(f"评估布局质量失败: {e}")
            raise
            
    def _evaluate_constraint_satisfaction(
        self,
        layout: Dict,
        design_info: Dict
    ) -> Dict:
        """评估约束满足度
        
        Args:
            layout: 布局信息
            design_info: 设计信息
            
        Returns:
            Dict: 约束满足度指标
        """
        try:
            # 评估面积约束
            area_satisfaction = self._evaluate_area_constraints(layout, design_info)
            
            # 评估时序约束
            timing_satisfaction = self._evaluate_timing_constraints(layout, design_info)
            
            # 评估功耗约束
            power_satisfaction = self._evaluate_power_constraints(layout, design_info)
            
            return {
                'area': area_satisfaction,
                'timing': timing_satisfaction,
                'power': power_satisfaction
            }
            
        except Exception as e:
            logger.error(f"评估约束满足度失败: {e}")
            raise
            
    def _evaluate_optimization_efficiency(
        self,
        layout: Dict,
        design_info: Dict
    ) -> Dict:
        """评估优化效率
        
        Args:
            layout: 布局信息
            design_info: 设计信息
            
        Returns:
            Dict: 优化效率指标
        """
        try:
            # 计算优化时间
            optimization_time = layout.get('optimization_time', 0)
            
            # 计算内存使用
            memory_usage = layout.get('memory_usage', 0)
            
            # 计算迭代次数
            iteration_count = layout.get('iteration_count', 0)
            
            return {
                'optimization_time': optimization_time,
                'memory_usage': memory_usage,
                'iteration_count': iteration_count
            }
            
        except Exception as e:
            logger.error(f"评估优化效率失败: {e}")
            raise
            
    def _calculate_averages(self, experiments: Dict) -> Dict:
        """计算实验结果的平均值
        
        Args:
            experiments: 实验结果
            
        Returns:
            Dict: 平均值
        """
        try:
            averages = {
                'quality': {},
                'constraints': {},
                'efficiency': {}
            }
            
            # 收集所有有效结果
            quality_metrics = []
            constraint_metrics = []
            efficiency_metrics = []
            
            for run_result in experiments.values():
                if 'error' not in run_result:
                    quality_metrics.append(run_result['quality'])
                    constraint_metrics.append(run_result['constraints'])
                    efficiency_metrics.append(run_result['efficiency'])
            
            # 计算平均值
            if quality_metrics:
                averages['quality'] = {
                    'density': np.mean([m['density'] for m in quality_metrics]),
                    'congestion': np.mean([m['congestion'] for m in quality_metrics]),
                    'timing_margin': np.mean([m['timing_margin'] for m in quality_metrics])
                }
            
            if constraint_metrics:
                averages['constraints'] = {
                    'area': np.mean([m['area'] for m in constraint_metrics]),
                    'timing': np.mean([m['timing'] for m in constraint_metrics]),
                    'power': np.mean([m['power'] for m in constraint_metrics])
                }
            
            if efficiency_metrics:
                averages['efficiency'] = {
                    'optimization_time': np.mean([m['optimization_time'] for m in efficiency_metrics]),
                    'memory_usage': np.mean([m['memory_usage'] for m in efficiency_metrics]),
                    'iteration_count': np.mean([m['iteration_count'] for m in efficiency_metrics])
                }
            
            return averages
            
        except Exception as e:
            logger.error(f"计算平均值失败: {e}")
            raise
            
    def _save_results(self, results: Dict):
        """保存实验结果
        
        Args:
            results: 实验结果
        """
        try:
            # 创建结果目录
            os.makedirs(self.config.results_dir, exist_ok=True)
            
            # 生成结果文件名
            timestamp = results['timestamp']
            filename = f"experiment_results_{timestamp}.json"
            filepath = os.path.join(self.config.results_dir, filename)
            
            # 保存结果
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"实验结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存实验结果失败: {e}")
            raise 