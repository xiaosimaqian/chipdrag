import os
import time
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from modules.utils.benchmark_loader import BenchmarkLoader
from .rag_system import RAGSystem
from .layout_generators import ORAssistant, GLayout, ORAssistantSim, GLayoutSim
from chiprag.config.experiment_config import ExperimentConfig
import logging

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
            
            # 初始化布局生成器
            self.layout_generators = {}
            for name, config in self.config.layout_generators.items():
                try:
                    if name == 'ORAssistant':
                        self.layout_generators[name] = ORAssistant(config['config'])
                    elif name == 'GLayout':
                        self.layout_generators[name] = GLayout(config['config'])
                    elif name == 'ORAssistant-Sim':
                        self.layout_generators[name] = ORAssistantSim(config['config'])
                    elif name == 'GLayout-Sim':
                        self.layout_generators[name] = GLayoutSim(config['config'])
                    else:
                        logger.warning(f"未知的布局生成器类型: {name}")
                except Exception as e:
                    logger.error(f"初始化布局生成器 {name} 失败: {e}")
                    raise
            
            # 初始化RAG系统
            self.rag_system = RAGSystem()
            
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
                
        if not isinstance(self.config.experiment_params['num_runs'], int) or \
           self.config.experiment_params['num_runs'] <= 0:
            raise ValueError("实验运行次数必须是正整数")
            
        if not isinstance(self.config.experiment_params['seed'], int):
            raise ValueError("随机种子必须是整数")
            
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有实验
        
        Returns:
            实验结果字典
            
        Raises:
            ValueError: 当实验参数无效时
            RuntimeError: 当实验运行失败时
        """
        try:
            results = {
                'layout_quality': {},
                'constraint_satisfaction': {},
                'optimization_efficiency': {}
            }
            
            # 对每个布局生成器运行实验
            for method_name, generator in self.layout_generators.items():
                logger.info(f"\n运行 {method_name} 实验...")
                
                # 初始化结果
                results['layout_quality'][method_name] = {
                    'density': [],
                    'congestion': [],
                    'timing_margin': []
                }
                results['constraint_satisfaction'][method_name] = {
                    'mgc_fft_1': [],
                    'mgc_des_perf_1': [],
                    'mgc_pci_bridge32_a': []
                }
                results['optimization_efficiency'][method_name] = {
                    'generation_time': [],
                    'iterations': [],
                    'total_time': []
                }
                
                # 对每个基准测试运行多次实验
                for benchmark in self.config.benchmarks:
                    logger.info(f"\n处理基准测试: {benchmark}")
                    
                    try:
                        # 加载基准测试数据
                        design_info = self.benchmark_loader.load_benchmark(benchmark)
                        
                        for run in range(self.config.experiment_params['num_runs']):
                            logger.info(f"运行 {run + 1}/{self.config.experiment_params['num_runs']}")
                            
                            try:
                                # 生成布局
                                start_time = time.time()
                                layout = generator.generate_layout(design_info)
                                generation_time = time.time() - start_time
                                
                                # 评估布局质量
                                quality = self._evaluate_layout_quality(layout, design_info)
                                results['layout_quality'][method_name]['density'].append(quality['density'])
                                results['layout_quality'][method_name]['congestion'].append(quality['congestion'])
                                results['layout_quality'][method_name]['timing_margin'].append(quality['timing_margin'])
                                
                                # 评估约束满足率
                                satisfaction = self._evaluate_constraint_satisfaction(layout, design_info)
                                results['constraint_satisfaction'][method_name][benchmark].append(satisfaction)
                                
                                # 评估优化效率
                                results['optimization_efficiency'][method_name]['generation_time'].append(generation_time)
                                results['optimization_efficiency'][method_name]['iterations'].append(generator.iterations)
                                results['optimization_efficiency'][method_name]['total_time'].append(time.time() - start_time)
                                
                            except Exception as e:
                                logger.error(f"运行 {run + 1} 失败: {e}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"处理基准测试 {benchmark} 失败: {e}")
                        continue
            
            # 计算平均值
            self._calculate_averages(results)
            
            # 保存结果
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"运行实验失败: {e}")
            raise RuntimeError(f"运行实验失败: {e}")
    
    def _evaluate_layout_quality(self, layout: Dict, design_info: Dict) -> Dict[str, float]:
        """评估布局质量"""
        # 计算单元密度
        total_area = design_info['die_area'][2] * design_info['die_area'][3]
        placed_area = sum(comp['area'] for comp in layout['components'].values())
        density = placed_area / total_area
        
        # 计算布线拥塞度
        congestion = self._calculate_congestion(layout, design_info)
        
        # 计算时序裕量
        timing_margin = self._calculate_timing_margin(layout, design_info)
        
        return {
            'density': density,
            'congestion': congestion,
            'timing_margin': timing_margin
        }
    
    def _evaluate_constraint_satisfaction(self, layout: Dict, design_info: Dict) -> float:
        """评估约束满足率"""
        # 计算面积约束满足率
        area_satisfaction = self._calculate_area_satisfaction(layout, design_info)
        
        # 计算时序约束满足率
        timing_satisfaction = self._calculate_timing_satisfaction(layout, design_info)
        
        # 计算功耗约束满足率
        power_satisfaction = self._calculate_power_satisfaction(layout, design_info)
        
        # 计算总满足率
        weights = self.config.metrics['constraint_satisfaction']
        total_satisfaction = (
            weights['area_weight'] * area_satisfaction +
            weights['timing_weight'] * timing_satisfaction +
            weights['power_weight'] * power_satisfaction
        )
        
        return total_satisfaction
    
    def _calculate_congestion(self, layout: Dict, design_info: Dict) -> float:
        """计算布线拥塞度"""
        # 将布局区域划分为网格
        grid_size = 10  # 网格大小
        x_cells = int(design_info['die_area'][2] / grid_size)
        y_cells = int(design_info['die_area'][3] / grid_size)
        
        # 初始化网格
        grid = np.zeros((x_cells, y_cells))
        
        # 计算每个网格中的单元数量
        for comp in layout['components'].values():
            x = int(comp['x'] / grid_size)
            y = int(comp['y'] / grid_size)
            if 0 <= x < x_cells and 0 <= y < y_cells:
                grid[x, y] += 1
        
        # 计算拥塞度（使用标准差作为拥塞度的度量）
        congestion = np.std(grid)
        
        return float(congestion)
    
    def _calculate_timing_margin(self, layout: Dict, design_info: Dict) -> float:
        """计算时序裕量"""
        # 简化的时序分析
        # 实际应用中应该使用更复杂的时序分析工具
        timing_margin = 0.0
        
        # 计算每个路径的延迟
        for net in design_info['nets']:
            # 获取源和汇点
            source = net.get('source')
            sinks = net.get('sinks', [])
            
            if not source or not sinks:
                continue
                
            # 计算源到每个汇的延迟
            for sink in sinks:
                if source not in layout['components'] or sink not in layout['components']:
                    continue
                    
                # 简化的曼哈顿距离作为延迟估计
                delay = abs(layout['components'][source]['x'] - layout['components'][sink]['x']) + \
                       abs(layout['components'][source]['y'] - layout['components'][sink]['y'])
                
                # 更新时序裕量
                timing_margin = max(timing_margin, delay)
        
        return timing_margin
    
    def _calculate_area_satisfaction(self, layout: Dict, design_info: Dict) -> float:
        """计算面积约束满足率"""
        # 计算实际使用的面积
        used_area = sum(comp['area'] for comp in layout['components'].values())
        
        # 计算可用面积
        available_area = design_info['die_area'][2] * design_info['die_area'][3]
        
        # 计算满足率（使用面积不超过可用面积的90%）
        satisfaction = 1.0 - (used_area / available_area)
        
        return max(0.0, min(1.0, satisfaction))
    
    def _calculate_timing_satisfaction(self, layout: Dict, design_info: Dict) -> float:
        """计算时序约束满足率"""
        # 获取时序约束
        timing_constraint = design_info.get('timing_constraint', 1000.0)  # 默认1000ps
        
        # 计算最大路径延迟
        max_delay = self._calculate_timing_margin(layout, design_info)
        
        # 计算满足率
        satisfaction = 1.0 - (max_delay / timing_constraint)
        
        return max(0.0, min(1.0, satisfaction))
    
    def _calculate_power_satisfaction(self, layout: Dict, design_info: Dict) -> float:
        """计算功耗约束满足率"""
        # 获取功耗约束
        power_constraint = design_info.get('power_constraint', 1000.0)  # 默认1000mW
        
        # 计算总功耗（简化模型）
        total_power = 0.0
        for comp in layout['components'].values():
            # 简化的功耗模型：与面积成正比
            total_power += comp['area'] * 0.1  # 假设每单位面积消耗0.1mW
        
        # 计算满足率
        satisfaction = 1.0 - (total_power / power_constraint)
        
        return max(0.0, min(1.0, satisfaction))
    
    def _calculate_averages(self, results: Dict[str, Any]):
        """计算所有指标的平均值"""
        for category in results:
            for method in results[category]:
                if isinstance(results[category][method], dict):
                    for metric in results[category][method]:
                        if isinstance(results[category][method][metric], list):
                            results[category][method][metric] = np.mean(results[category][method][metric])
    
    def _save_results(self, results: Dict[str, Any]):
        """保存实验结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(os.path.dirname(__file__), '../../results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细结果
        detailed_file = os.path.join(results_dir, f'experiment_results_{timestamp}.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存摘要结果
        summary = {
            'timestamp': timestamp,
            'benchmarks': self.config.benchmarks,
            'methods': list(self.layout_generators.keys()),
            'summary': {
                'layout_quality': {},
                'constraint_satisfaction': {},
                'optimization_efficiency': {}
            }
        }
        
        # 计算每个方法的平均指标
        for category in results:
            for method in results[category]:
                if isinstance(results[category][method], dict):
                    summary['summary'][category][method] = {
                        metric: np.mean(values) if isinstance(values, list) else values
                        for metric, values in results[category][method].items()
                    }
        
        summary_file = os.path.join(results_dir, f'experiment_summary_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4) 