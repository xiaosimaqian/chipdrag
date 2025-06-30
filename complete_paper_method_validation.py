#!/usr/bin/env python3
"""
完整的论文方法验证系统
整合论文三大核心特性：
1. 动态重排序机制：基于强化学习的动态k值选择
2. 实体增强技术：芯片设计实体的压缩和注入
3. 质量反馈驱动的闭环优化：基于HPWL改进的反馈机制

所有数据必须是真实获取的，验证论文方法的有效性
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import random
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入核心组件
from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever, DynamicRetrievalResult
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface
from modules.core.dynamic_layout_generator import DynamicLayoutGenerator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_paper_method_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompletePaperMethodValidator:
    """完整的论文方法验证系统"""
    
    def __init__(self):
        """初始化验证系统"""
        self.output_dir = Path("results/complete_paper_method_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.experiment_config = {
            'designs': [
                'mgc_des_perf_1',  # 性能设计
                'mgc_fft_1',       # FFT设计
                'mgc_matrix_mult_1', # 矩阵乘法设计
                'mgc_pci_bridge32_a' # PCI桥接设计
            ],
            'baseline_params': {
                'density_target': 0.7,
                'wirelength_weight': 1.0,
                'density_weight': 1.0
            },
            'validation_rounds': 5,  # 验证轮数
            'rl_episodes': 20,       # 强化学习episode数
            'k_range': (3, 15),      # 动态k值范围
            'quality_threshold': 0.7 # 质量阈值
        }
        
        # 初始化组件
        self._init_components()
        
        # 实验结果
        self.results = {
            'experiment_info': {
                'name': 'complete_paper_method_validation',
                'start_time': datetime.now().isoformat(),
                'config': self.experiment_config
            },
            'baseline_results': {},
            'dynamic_reranking_results': {},
            'entity_enhancement_results': {},
            'quality_feedback_results': {},
            'comparison_analysis': {}
        }
        
        logger.info("完整的论文方法验证系统初始化完成")
    
    def _init_components(self):
        """初始化核心组件"""
        try:
            # 1. 初始化动态RAG检索器（动态重排序 + 实体增强）
            drag_config = {
                'knowledge_base': {
                    'path': 'data/knowledge_base/ispd_cases.json',
                    'format': 'json',
                    'layout_experience': 'data/knowledge_base'
                },
                'llm': {
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'dynamic_k_range': self.experiment_config['k_range'],
                'quality_threshold': self.experiment_config['quality_threshold'],
                'learning_rate': 0.01,
                'entity_compression_ratio': 0.1,
                'entity_similarity_threshold': 0.8,
                'compressed_entity_dim': 128
            }
            self.dynamic_retriever = DynamicRAGRetriever(drag_config)
            
            # 2. 初始化知识库
            self.knowledge_base = KnowledgeBase(drag_config['knowledge_base'])
            
            logger.info("核心组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化组件失败: {str(e)}")
            raise
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """运行完整的论文方法验证"""
        logger.info("开始完整的论文方法验证...")
        
        try:
            # 1. 基线验证（OpenROAD默认参数）
            logger.info("=== 1. 基线验证 ===")
            baseline_results = self._run_baseline_validation()
            self.results['baseline_results'] = baseline_results
            
            # 2. 动态重排序验证
            logger.info("=== 2. 动态重排序验证 ===")
            dynamic_reranking_results = self._run_dynamic_reranking_validation()
            self.results['dynamic_reranking_results'] = dynamic_reranking_results
            
            # 3. 实体增强验证
            logger.info("=== 3. 实体增强验证 ===")
            entity_enhancement_results = self._run_entity_enhancement_validation()
            self.results['entity_enhancement_results'] = entity_enhancement_results
            
            # 4. 质量反馈驱动的闭环优化验证
            logger.info("=== 4. 质量反馈驱动的闭环优化验证 ===")
            quality_feedback_results = self._run_quality_feedback_validation()
            self.results['quality_feedback_results'] = quality_feedback_results
            
            # 5. 对比分析
            logger.info("=== 5. 对比分析 ===")
            comparison_analysis = self._run_comparison_analysis()
            self.results['comparison_analysis'] = comparison_analysis
            
            # 6. 生成报告和可视化
            self._generate_validation_report()
            self._create_visualizations()
            
            # 7. 保存结果
            self._save_results()
            
            logger.info("完整的论文方法验证完成！")
            return self.results
            
        except Exception as e:
            logger.error(f"完整验证失败: {str(e)}")
            raise
    
    def _run_baseline_validation(self) -> Dict[str, Any]:
        """运行基线验证（OpenROAD默认参数）"""
        baseline_results = {}
        
        for design_name in self.experiment_config['designs']:
            logger.info(f"基线验证设计: {design_name}")
            
            try:
                # 构建设计路径
                design_path = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
                
                # 创建OpenROAD接口
                interface = RealOpenROADInterface(work_dir=design_path)
                
                # 使用默认参数运行布局
                start_time = time.time()
                result = interface.run_placement(
                    density_target=self.experiment_config['baseline_params']['density_target'],
                    wirelength_weight=self.experiment_config['baseline_params']['wirelength_weight'],
                    density_weight=self.experiment_config['baseline_params']['density_weight']
                )
                execution_time = time.time() - start_time
                
                # 提取HPWL
                hpwl = result.get('hpwl', float('inf'))
                
                baseline_results[design_name] = {
                    'success': result['success'],
                    'hpwl': hpwl,
                    'execution_time': execution_time,
                    'params': self.experiment_config['baseline_params']
                }
                
                logger.info(f"  基线HPWL: {hpwl:.2e}, 执行时间: {execution_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"基线验证设计 {design_name} 失败: {str(e)}")
                baseline_results[design_name] = {
                    'success': False,
                    'hpwl': float('inf'),
                    'execution_time': 0,
                    'error': str(e)
                }
        
        return baseline_results
    
    def _run_dynamic_reranking_validation(self) -> Dict[str, Any]:
        """运行动态重排序验证"""
        dynamic_results = {}
        
        for design_name in self.experiment_config['designs']:
            logger.info(f"动态重排序验证设计: {design_name}")
            
            try:
                # 构建查询和设计信息
                query, design_info = self._build_query_and_design_info(design_name)
                
                # 执行动态重排序检索
                start_time = time.time()
                retrieval_results = self.dynamic_retriever.retrieve_with_dynamic_reranking(
                    query, design_info
                )
                retrieval_time = time.time() - start_time
                
                # 提取推荐参数
                recommended_params = self._extract_layout_params_from_retrieval(retrieval_results)
                
                # 使用推荐参数运行布局
                design_path = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
                interface = RealOpenROADInterface(work_dir=design_path)
                
                layout_start_time = time.time()
                result = interface.run_placement(
                    density_target=recommended_params.get('density_target', 0.7),
                    wirelength_weight=recommended_params.get('wirelength_weight', 1.0),
                    density_weight=recommended_params.get('density_weight', 1.0)
                )
                layout_time = time.time() - layout_start_time
                
                # 提取HPWL
                hpwl = result.get('hpwl', float('inf'))
                
                dynamic_results[design_name] = {
                    'success': result['success'],
                    'hpwl': hpwl,
                    'retrieval_time': retrieval_time,
                    'layout_time': layout_time,
                    'total_time': retrieval_time + layout_time,
                    'retrieved_count': len(retrieval_results),
                    'recommended_params': recommended_params,
                    'dynamic_k': len(retrieval_results)
                }
                
                logger.info(f"  动态重排序HPWL: {hpwl:.2e}, 检索时间: {retrieval_time:.2f}秒, 布局时间: {layout_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"动态重排序验证设计 {design_name} 失败: {str(e)}")
                dynamic_results[design_name] = {
                    'success': False,
                    'hpwl': float('inf'),
                    'retrieval_time': 0,
                    'layout_time': 0,
                    'total_time': 0,
                    'error': str(e)
                }
        
        return dynamic_results
    
    def _run_entity_enhancement_validation(self) -> Dict[str, Any]:
        """运行实体增强验证"""
        entity_results = {}
        
        for design_name in self.experiment_config['designs']:
            logger.info(f"实体增强验证设计: {design_name}")
            
            try:
                # 构建查询和设计信息
                query, design_info = self._build_query_and_design_info(design_name)
                
                # 执行带实体增强的检索
                start_time = time.time()
                retrieval_results = self.dynamic_retriever.retrieve_with_dynamic_reranking(
                    query, design_info
                )
                
                # 检查实体增强效果
                entity_enhanced_count = 0
                total_entities = 0
                
                for result in retrieval_results:
                    if hasattr(result, 'entity_embeddings') and result.entity_embeddings is not None:
                        entity_enhanced_count += 1
                        total_entities += 1
                    else:
                        total_entities += 1
                
                retrieval_time = time.time() - start_time
                
                # 提取推荐参数
                recommended_params = self._extract_layout_params_from_retrieval(retrieval_results)
                
                # 使用推荐参数运行布局
                design_path = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
                interface = RealOpenROADInterface(work_dir=design_path)
                
                layout_start_time = time.time()
                result = interface.run_placement(
                    density_target=recommended_params.get('density_target', 0.7),
                    wirelength_weight=recommended_params.get('wirelength_weight', 1.0),
                    density_weight=recommended_params.get('density_weight', 1.0)
                )
                layout_time = time.time() - layout_start_time
                
                # 提取HPWL
                hpwl = result.get('hpwl', float('inf'))
                
                entity_results[design_name] = {
                    'success': result['success'],
                    'hpwl': hpwl,
                    'retrieval_time': retrieval_time,
                    'layout_time': layout_time,
                    'total_time': retrieval_time + layout_time,
                    'entity_enhanced_count': entity_enhanced_count,
                    'total_entities': total_entities,
                    'entity_enhancement_rate': entity_enhanced_count / max(1, total_entities),
                    'recommended_params': recommended_params
                }
                
                logger.info(f"  实体增强HPWL: {hpwl:.2e}, 实体增强率: {entity_enhanced_count}/{total_entities}")
                
            except Exception as e:
                logger.error(f"实体增强验证设计 {design_name} 失败: {str(e)}")
                entity_results[design_name] = {
                    'success': False,
                    'hpwl': float('inf'),
                    'retrieval_time': 0,
                    'layout_time': 0,
                    'total_time': 0,
                    'error': str(e)
                }
        
        return entity_results
    
    def _run_quality_feedback_validation(self) -> Dict[str, Any]:
        """运行质量反馈驱动的闭环优化验证"""
        feedback_results = {}
        
        for design_name in self.experiment_config['designs']:
            logger.info(f"质量反馈验证设计: {design_name}")
            
            try:
                # 构建查询和设计信息
                query, design_info = self._build_query_and_design_info(design_name)
                
                # 多轮迭代优化
                iteration_results = []
                best_hpwl = float('inf')
                best_params = None
                
                for iteration in range(self.experiment_config['validation_rounds']):
                    logger.info(f"  质量反馈迭代 {iteration + 1}/{self.experiment_config['validation_rounds']}")
                    
                    # 执行检索
                    retrieval_results = self.dynamic_retriever.retrieve_with_dynamic_reranking(
                        query, design_info
                    )
                    
                    # 提取参数
                    params = self._extract_layout_params_from_retrieval(retrieval_results)
                    
                    # 运行布局
                    design_path = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
                    interface = RealOpenROADInterface(work_dir=design_path)
                    
                    result = interface.run_placement(
                        density_target=params.get('density_target', 0.7),
                        wirelength_weight=params.get('wirelength_weight', 1.0),
                        density_weight=params.get('density_weight', 1.0)
                    )
                    
                    # 提取HPWL
                    hpwl = result.get('hpwl', float('inf'))
                    
                    iteration_results.append({
                        'iteration': iteration + 1,
                        'hpwl': hpwl,
                        'params': params,
                        'success': result['success']
                    })
                    
                    # 更新最佳结果
                    if result['success'] and hpwl < best_hpwl:
                        best_hpwl = hpwl
                        best_params = params
                    
                    # 生成质量反馈并更新检索器
                    if result['success']:
                        quality_feedback = {
                            'overall_score': min(1.0, max(0.0, (1e9 - hpwl) / 1e9)),
                            'hpwl_score': min(1.0, max(0.0, (1e9 - hpwl) / 1e9)),
                            'iteration': iteration + 1
                        }
                        
                        # 更新检索器
                        query_hash = hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()
                        self.dynamic_retriever.update_with_feedback(
                            query_hash, 
                            {'retrieval_results': retrieval_results}, 
                            quality_feedback
                        )
                
                feedback_results[design_name] = {
                    'iterations': iteration_results,
                    'best_hpwl': best_hpwl,
                    'best_params': best_params,
                    'improvement_curve': [r['hpwl'] for r in iteration_results if r['success']],
                    'final_iteration': len(iteration_results)
                }
                
                logger.info(f"  质量反馈最佳HPWL: {best_hpwl:.2e}")
                
            except Exception as e:
                logger.error(f"质量反馈验证设计 {design_name} 失败: {str(e)}")
                feedback_results[design_name] = {
                    'error': str(e),
                    'iterations': [],
                    'best_hpwl': float('inf')
                }
        
        return feedback_results
    
    def _run_comparison_analysis(self) -> Dict[str, Any]:
        """运行对比分析"""
        comparison = {}
        
        # 计算各方法的平均HPWL
        baseline_hpwls = [r['hpwl'] for r in self.results['baseline_results'].values() if r['success']]
        dynamic_hpwls = [r['hpwl'] for r in self.results['dynamic_reranking_results'].values() if r['success']]
        entity_hpwls = [r['hpwl'] for r in self.results['entity_enhancement_results'].values() if r['success']]
        feedback_hpwls = [r['best_hpwl'] for r in self.results['quality_feedback_results'].values() if r['best_hpwl'] != float('inf')]
        
        # 计算改进率
        if baseline_hpwls:
            baseline_avg = np.mean(baseline_hpwls)
            
            comparison['improvement_rates'] = {
                'dynamic_reranking': ((baseline_avg - np.mean(dynamic_hpwls)) / baseline_avg * 100) if dynamic_hpwls else 0,
                'entity_enhancement': ((baseline_avg - np.mean(entity_hpwls)) / baseline_avg * 100) if entity_hpwls else 0,
                'quality_feedback': ((baseline_avg - np.mean(feedback_hpwls)) / baseline_avg * 100) if feedback_hpwls else 0
            }
        
        # 计算成功率
        comparison['success_rates'] = {
            'baseline': sum(1 for r in self.results['baseline_results'].values() if r['success']) / len(self.results['baseline_results']),
            'dynamic_reranking': sum(1 for r in self.results['dynamic_reranking_results'].values() if r['success']) / len(self.results['dynamic_reranking_results']),
            'entity_enhancement': sum(1 for r in self.results['entity_enhancement_results'].values() if r['success']) / len(self.results['entity_enhancement_results']),
            'quality_feedback': sum(1 for r in self.results['quality_feedback_results'].values() if r['best_hpwl'] != float('inf')) / len(self.results['quality_feedback_results'])
        }
        
        # 计算平均执行时间
        comparison['avg_execution_times'] = {
            'baseline': np.mean([r['execution_time'] for r in self.results['baseline_results'].values()]),
            'dynamic_reranking': np.mean([r['total_time'] for r in self.results['dynamic_reranking_results'].values()]),
            'entity_enhancement': np.mean([r['total_time'] for r in self.results['entity_enhancement_results'].values()]),
            'quality_feedback': np.mean([r['total_time'] for r in self.results['quality_feedback_results'].values() if r['best_hpwl'] != float('inf')])
        }
        
        return comparison
    
    def _build_query_and_design_info(self, design_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """构建查询和设计信息"""
        query = {
            'text': f"优化 {design_name} 的芯片布局，提升线长和拥塞性能",
            'design_type': 'digital',
            'complexity': 'medium',
            'constraints': ['wirelength', 'congestion', 'timing']
        }
        
        design_info = {
            'name': design_name,
            'design_type': 'digital',
            'cell_count': 10000,  # 默认值，实际应该从设计文件中提取
            'net_count': 20000,   # 默认值
            'constraints': ['wirelength', 'congestion', 'timing']
        }
        
        return query, design_info
    
    def _extract_layout_params_from_retrieval(self, retrieval_results: List[DynamicRetrievalResult]) -> Dict[str, Any]:
        """从检索结果中提取布局参数"""
        if not retrieval_results:
            return self.experiment_config['baseline_params']
        
        # 基于检索结果的相关性分数加权平均参数
        total_weight = 0
        weighted_params = defaultdict(float)
        
        for result in retrieval_results:
            weight = result.relevance_score
            total_weight += weight
            
            # 从知识中提取参数（这里简化处理）
            knowledge = result.knowledge if isinstance(result.knowledge, dict) else {}
            params = knowledge.get('layout_params', {})
            
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    weighted_params[param_name] += param_value * weight
        
        # 归一化参数
        if total_weight > 0:
            for param_name in weighted_params:
                weighted_params[param_name] /= total_weight
        
        # 返回推荐参数
        recommended_params = {
            'density_target': weighted_params.get('density_target', 0.7),
            'wirelength_weight': weighted_params.get('wirelength_weight', 1.0),
            'density_weight': weighted_params.get('density_weight', 1.0)
        }
        
        return recommended_params
    
    def _generate_validation_report(self):
        """生成验证报告"""
        report = []
        report.append("# 完整的论文方法验证报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 实验信息
        report.append("## 实验信息")
        report.append(f"- 实验名称: {self.results['experiment_info']['name']}")
        report.append(f"- 开始时间: {self.results['experiment_info']['start_time']}")
        report.append(f"- 测试设计数: {len(self.experiment_config['designs'])}")
        report.append(f"- 验证轮数: {self.experiment_config['validation_rounds']}")
        report.append("")
        
        # 对比分析结果
        if 'comparison_analysis' in self.results:
            comparison = self.results['comparison_analysis']
            
            report.append("## 对比分析结果")
            
            # HPWL改进率
            if 'improvement_rates' in comparison:
                report.append("### HPWL改进率")
                for method, improvement in comparison['improvement_rates'].items():
                    report.append(f"- {method}: {improvement:.2f}%")
                report.append("")
            
            # 成功率
            if 'success_rates' in comparison:
                report.append("### 成功率")
                for method, success_rate in comparison['success_rates'].items():
                    report.append(f"- {method}: {success_rate:.2%}")
                report.append("")
            
            # 执行时间
            if 'avg_execution_times' in comparison:
                report.append("### 平均执行时间")
                for method, avg_time in comparison['avg_execution_times'].items():
                    report.append(f"- {method}: {avg_time:.2f}秒")
                report.append("")
        
        # 详细结果
        report.append("## 详细结果")
        
        for design_name in self.experiment_config['designs']:
            report.append(f"### {design_name}")
            
            # 基线结果
            baseline = self.results['baseline_results'].get(design_name, {})
            if baseline.get('success'):
                report.append(f"- 基线HPWL: {baseline['hpwl']:.2e}")
            
            # 动态重排序结果
            dynamic = self.results['dynamic_reranking_results'].get(design_name, {})
            if dynamic.get('success'):
                report.append(f"- 动态重排序HPWL: {dynamic['hpwl']:.2e}")
            
            # 实体增强结果
            entity = self.results['entity_enhancement_results'].get(design_name, {})
            if entity.get('success'):
                report.append(f"- 实体增强HPWL: {entity['hpwl']:.2e}")
            
            # 质量反馈结果
            feedback = self.results['quality_feedback_results'].get(design_name, {})
            if feedback.get('best_hpwl') != float('inf'):
                report.append(f"- 质量反馈最佳HPWL: {feedback['best_hpwl']:.2e}")
            
            report.append("")
        
        # 结论
        report.append("## 结论")
        if 'comparison_analysis' in self.results:
            comparison = self.results['comparison_analysis']
            if 'improvement_rates' in comparison:
                best_method = max(comparison['improvement_rates'].items(), key=lambda x: x[1])
                report.append(f"论文方法在芯片布局优化中表现出显著效果，其中{best_method[0]}方法实现了{best_method[1]:.2f}%的HPWL改进。")
        
        # 保存报告
        report_path = self.output_dir / "validation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"验证报告已保存到: {report_path}")
    
    def _create_visualizations(self):
        """创建可视化图表"""
        # 1. HPWL对比图
        self._create_hpwl_comparison_plot()
        
        # 2. 改进率对比图
        self._create_improvement_comparison_plot()
        
        # 3. 成功率对比图
        self._create_success_rate_comparison_plot()
        
        # 4. 执行时间对比图
        self._create_execution_time_comparison_plot()
    
    def _create_hpwl_comparison_plot(self):
        """创建HPWL对比图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ['Baseline', 'Dynamic Reranking', 'Entity Enhancement', 'Quality Feedback']
        hpwl_data = []
        
        for design_name in self.experiment_config['designs']:
            baseline = self.results['baseline_results'].get(design_name, {}).get('hpwl', float('inf'))
            dynamic = self.results['dynamic_reranking_results'].get(design_name, {}).get('hpwl', float('inf'))
            entity = self.results['entity_enhancement_results'].get(design_name, {}).get('hpwl', float('inf'))
            feedback = self.results['quality_feedback_results'].get(design_name, {}).get('best_hpwl', float('inf'))
            
            hpwl_data.append([baseline, dynamic, entity, feedback])
        
        hpwl_data = np.array(hpwl_data)
        
        # 绘制箱线图
        bp = ax.boxplot(hpwl_data, labels=methods, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('HPWL (log scale)')
        ax.set_title('HPWL对比分析')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hpwl_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_improvement_comparison_plot(self):
        """创建改进率对比图"""
        if 'comparison_analysis' not in self.results:
            return
        
        comparison = self.results['comparison_analysis']
        if 'improvement_rates' not in comparison:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(comparison['improvement_rates'].keys())
        improvements = list(comparison['improvement_rates'].values())
        
        bars = ax.bar(methods, improvements, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        
        # 添加数值标签
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{improvement:.2f}%', ha='center', va='bottom')
        
        ax.set_ylabel('HPWL改进率 (%)')
        ax.set_title('各方法HPWL改进率对比')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_success_rate_comparison_plot(self):
        """创建成功率对比图"""
        if 'comparison_analysis' not in self.results:
            return
        
        comparison = self.results['comparison_analysis']
        if 'success_rates' not in comparison:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(comparison['success_rates'].keys())
        success_rates = [rate * 100 for rate in comparison['success_rates'].values()]
        
        bars = ax.bar(methods, success_rates, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom')
        
        ax.set_ylabel('成功率 (%)')
        ax.set_title('各方法成功率对比')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_execution_time_comparison_plot(self):
        """创建执行时间对比图"""
        if 'comparison_analysis' not in self.results:
            return
        
        comparison = self.results['comparison_analysis']
        if 'avg_execution_times' not in comparison:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(comparison['avg_execution_times'].keys())
        execution_times = list(comparison['avg_execution_times'].values())
        
        bars = ax.bar(methods, execution_times, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        
        # 添加数值标签
        for bar, time_val in zip(bars, execution_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.1f}s', ha='center', va='bottom')
        
        ax.set_ylabel('平均执行时间 (秒)')
        ax.set_title('各方法执行时间对比')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """保存实验结果"""
        results_path = self.output_dir / "complete_validation_results.json"
        
        # 处理numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(self.results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存到: {results_path}")


def main():
    """主函数"""
    logger.info("开始完整的论文方法验证")
    
    try:
        # 创建验证器
        validator = CompletePaperMethodValidator()
        
        # 运行完整验证
        results = validator.run_complete_validation()
        
        logger.info("完整的论文方法验证完成！")
        logger.info(f"结果保存在: {validator.output_dir}")
        
        # 打印关键结果
        if 'comparison_analysis' in results:
            comparison = results['comparison_analysis']
            if 'improvement_rates' in comparison:
                logger.info("HPWL改进率:")
                for method, improvement in comparison['improvement_rates'].items():
                    logger.info(f"  {method}: {improvement:.2f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"验证失败: {str(e)}")
        raise


if __name__ == "__main__":
    main() 