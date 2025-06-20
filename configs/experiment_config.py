from typing import Dict, List
import os

class ExperimentConfig:
    def __init__(self):
        # 基准测试集配置
        self.benchmarks = [
            'mgc_fft_1',
            'mgc_des_perf_1',
            'mgc_pci_bridge32_a'
        ]
        
        # 基准测试文件路径
        self.benchmark_path = '/Users/keqin/Documents/workspace/chip-rag/chip_design/ispd_2015_contest_benchmark'
        
        # 布局生成器配置
        self.layout_generators = {
            # 官方实现
            'ORAssistant': {
                'type': 'official',
                'config': {
                    'api_endpoint': 'http://localhost:8000',  # ORAssistant API端点
                    'timeout': 300,  # 超时时间（秒）
                    'max_retries': 3
                }
            },
            'GLayout': {
                'type': 'simulation',
                'config': {
                    'max_iterations': 1000,
                    'learning_rate': 0.01,
                    'force_strength': 0.5
                }
            },
            # 简化实现
            'ORAssistant-Sim': {
                'type': 'simulation',
                'config': {
                    'max_iterations': 1000,
                    'learning_rate': 0.01,
                    'force_strength': 0.5
                }
            },
            'GLayout-Sim': {
                'type': 'simulation',
                'config': {
                    'max_iterations': 1000,
                    'learning_rate': 0.01,
                    'force_strength': 0.5
                }
            }
        }
        
        # 评估指标配置
        self.metrics = {
            'layout_quality': {
                'density_weight': 0.4,
                'congestion_weight': 0.3,
                'timing_weight': 0.3
            },
            'constraint_satisfaction': {
                'area_weight': 0.4,
                'timing_weight': 0.3,
                'power_weight': 0.3
            },
            'optimization_efficiency': {
                'time_weight': 0.5,
                'iterations_weight': 0.5
            }
        }
        
        # 实验参数
        self.experiment_params = {
            'num_runs': 5,  # 每个配置运行次数
            'timeout': 3600,  # 总超时时间（秒）
            'seed': 42  # 随机种子
        } 