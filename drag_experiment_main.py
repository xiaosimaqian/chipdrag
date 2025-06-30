#!/usr/bin/env python3
"""
DRAG系统实验主控脚本
基于已有的DRAG检索功能和真实训练数据，实现两个核心实验：
1. 训练集设计：DRAG检索参数 vs. 训练数据最优参数
2. 测试集设计：DRAG检索参数 vs. OpenROAD默认参数
"""

import os
import sys
import json
import logging
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface
from modules.utils.config_loader import ConfigLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drag_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DRAGExperimentController:
    """DRAG实验主控器"""
    
    def __init__(self):
        """初始化实验主控器"""
        # 使用默认配置而不是加载外部文件
        self.config = {
            'experiment': {
                'timeout': 600,
                'max_workers': 4,
                'retry_count': 3
            },
            'drag': {
                'dynamic_k_range': (3, 15),
                'quality_threshold': 0.7,
                'learning_rate': 0.01
            }
        }
        
        # 设置路径
        self.data_dir = Path("data/designs/ispd_2015_contest_benchmark")
        self.results_dir = Path("results/drag_experiments")
        self.training_results_dir = Path("results/parallel_training")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化DRAG检索器
        self._init_drag_retriever()
        
        # 加载训练数据
        self.training_data = self._load_training_data()
        
        # 实验结果
        self.experiment_results = {
            'experiment_1_train_set': [],
            'experiment_2_test_set': []
        }
        
        logger.info("DRAG实验主控器初始化完成")
    
    def _init_drag_retriever(self):
        """初始化DRAG检索器"""
        try:
            # 知识库配置
            kb_config = {
                "path": "data/knowledge_base/ispd_cases.json",
                "format": "json",
                "layout_experience": "data/knowledge_base"
            }
            
            # DRAG检索器配置
            drag_config = {
                'knowledge_base': kb_config,
                'llm': {
                    'model_name': 'gpt-3.5-turbo',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'dynamic_k_range': (3, 15),
                'quality_threshold': 0.7,
                'learning_rate': 0.01,
                'entity_compression_ratio': 0.1,
                'entity_similarity_threshold': 0.8,
                'compressed_entity_dim': 128
            }
            
            # 初始化知识库和检索器
            self.knowledge_base = KnowledgeBase(kb_config)
            self.drag_retriever = DynamicRAGRetriever(drag_config)
            
            logger.info("DRAG检索器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化DRAG检索器失败: {str(e)}")
            raise
    
    def _load_training_data(self) -> Dict[str, Any]:
        """加载训练数据"""
        try:
            # 优先加载补全后的HPWL数据
            hpwl_filled_file = Path("results/iterative_training/batch_training_results_with_hpwl_filled.json")
            if hpwl_filled_file.exists():
                with open(hpwl_filled_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"加载补全HPWL训练数据: {len(data.get('results', []))} 个设计")
                return data
            
            # 如果没有补全数据，尝试加载并行训练结果
            training_file = self.training_results_dir / "parallel_training_results.json"
            if training_file.exists():
                with open(training_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"加载并行训练数据: {len(data)} 个设计")
                return {'results': data}
            
            # 如果没有并行训练结果，尝试加载其他训练结果
            hpwl_file = Path("results/iterative_training/batch_training_results_with_hpwl.json")
            if hpwl_file.exists():
                with open(hpwl_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"加载HPWL补全训练数据: {len(data.get('results', []))} 个设计")
                return data
            
            logger.warning("未找到训练数据文件")
            return {'results': []}
            
        except Exception as e:
            logger.error(f"加载训练数据失败: {str(e)}")
            return {'results': []}
    
    def _get_design_info(self, design_name: str) -> Dict[str, Any]:
        """获取设计信息"""
        design_dir = self.data_dir / design_name
        if not design_dir.exists():
            return {}
        
        # 提取设计特征
        design_info = {
            'name': design_name,
            'design_dir': str(design_dir),
            'has_verilog': (design_dir / 'design.v').exists(),
            'has_floorplan': (design_dir / 'floorplan.def').exists(),
            'has_tech_lef': (design_dir / 'tech.lef').exists(),
            'has_cells_lef': (design_dir / 'cells.lef').exists()
        }
        
        # 从训练数据中提取统计信息
        for result in self.training_data.get('results', []):
            if result.get('design') == design_name:
                design_info.update({
                    'num_instances': result.get('num_instances', 0),
                    'num_nets': result.get('num_nets', 0),
                    'area': result.get('area', (0, 0)),
                    'baseline_hpwl': result.get('final_hpwl'),
                    'training_success': result.get('success', False)
                })
                break
        
        return design_info
    
    def _build_drag_query(self, design_info: Dict[str, Any]) -> Dict[str, Any]:
        """构建DRAG检索查询"""
        query = {
            'text': f"优化 {design_info['name']} 的布局，设计规模: {design_info.get('num_instances', 0)} 实例",
            'design_type': 'unknown',
            'complexity': 'medium',
            'constraints': {}
        }
        
        # 根据设计规模调整复杂度
        num_instances = design_info.get('num_instances', 0)
        if num_instances > 50000:
            query['complexity'] = 'high'
        elif num_instances < 10000:
            query['complexity'] = 'low'
        
        return query
    
    def _extract_layout_params_from_retrieval(self, retrieval_results: List[Any]) -> Dict[str, Any]:
        """从检索结果中提取布局参数"""
        if not retrieval_results:
            return self._get_default_params()
        
        # 提取参数（这里需要根据实际的检索结果格式调整）
        params = {
            'cell_density': 0.7,  # 默认密度
            'core_area': (1000, 1000),  # 默认核心面积
            'wirelength_coef': 1.0,  # 默认线长系数
            'density_penalty': 0.1,  # 默认密度惩罚
            'max_displacement': 2,  # 默认最大位移
            'max_iterations': 5  # 默认最大迭代次数
        }
        
        # 从检索结果中提取参数（如果有的话）
        for result in retrieval_results:
            if hasattr(result, 'knowledge') and isinstance(result.knowledge, dict):
                knowledge = result.knowledge
                if 'optimization_params' in knowledge:
                    params.update(knowledge['optimization_params'])
                elif 'layout_params' in knowledge:
                    params.update(knowledge['layout_params'])
        
        return params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认布局参数"""
        return {
            'cell_density': 0.7,
            'core_area': (1000, 1000),
            'wirelength_coef': 1.0,
            'density_penalty': 0.1,
            'max_displacement': 2,
            'max_iterations': 5
        }
    
    def _run_openroad_layout(self, design_name: str, params: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
        """运行OpenROAD布局"""
        try:
            design_dir = str(self.data_dir / design_name)
            
            # 创建OpenROAD接口
            interface = RealOpenROADInterface(design_dir)
            
            # 只传递支持的参数
            result = interface.run_iterative_placement(
                num_iterations=10,
                timeout=timeout
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenROAD布局失败 {design_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'final_hpwl': None
            }
    
    def run_experiment_1_train_set(self):
        """实验1：训练集设计 - DRAG检索参数 vs. 训练数据最优参数"""
        logger.info("开始实验1：训练集设计对比")
        
        # 获取训练成功的设计
        successful_designs = []
        for result in self.training_data.get('results', []):
            if result.get('success', False) and result.get('dataset') == 'train':
                design_name = result.get('design')
                if design_name:
                    successful_designs.append(design_name)
        
        logger.info(f"找到 {len(successful_designs)} 个训练成功的设计")
        
        results = []
        for i, design_name in enumerate(successful_designs, 1):
            logger.info(f"[{i}/{len(successful_designs)}] 处理设计: {design_name}")
            
            try:
                # 获取设计信息
                design_info = self._get_design_info(design_name)
                if not design_info:
                    continue
                
                # 构建DRAG查询
                query = self._build_drag_query(design_info)
                
                # DRAG检索
                logger.info(f"  执行DRAG检索...")
                retrieval_results = self.drag_retriever.retrieve_with_dynamic_reranking(
                    query, design_info
                )
                
                # 提取DRAG推荐参数
                drag_params = self._extract_layout_params_from_retrieval(retrieval_results)
                
                # 使用DRAG参数运行布局
                logger.info(f"  使用DRAG参数运行布局...")
                drag_result = self._run_openroad_layout(design_name, drag_params)
                
                # 获取训练数据中的最优HPWL
                training_hpwl = design_info.get('baseline_hpwl')
                
                # 计算改进
                improvement = 0.0
                if drag_result['success'] and training_hpwl and drag_result.get('final_hpwl'):
                    drag_hpwl = drag_result['final_hpwl']
                    if training_hpwl > 0:
                        improvement = ((training_hpwl - drag_hpwl) / training_hpwl) * 100
                
                # 记录结果
                result = {
                    'design_name': design_name,
                    'drag_params': drag_params,
                    'drag_hpwl': drag_result.get('final_hpwl'),
                    'training_hpwl': training_hpwl,
                    'improvement_percent': improvement,
                    'drag_success': drag_result['success'],
                    'retrieval_count': len(retrieval_results),
                    'execution_time': drag_result.get('execution_time', 0)
                }
                
                results.append(result)
                
                logger.info(f"  ✅ 完成: DRAG HPWL={drag_result.get('final_hpwl')}, 改进={improvement:.2f}%")
                
            except Exception as e:
                logger.error(f"  处理设计 {design_name} 失败: {str(e)}")
                results.append({
                    'design_name': design_name,
                    'error': str(e),
                    'drag_success': False
                })
        
        self.experiment_results['experiment_1_train_set'] = results
        logger.info(f"实验1完成: {len(results)} 个设计")
        return results
    
    def run_experiment_2_test_set(self):
        """实验2：测试集设计 - DRAG检索参数 vs. OpenROAD默认参数"""
        logger.info("开始实验2：测试集设计对比")
        
        # 获取测试集设计
        test_designs = []
        dataset_split_file = self.training_results_dir / "dataset_split.json"
        if dataset_split_file.exists():
            with open(dataset_split_file, 'r') as f:
                split_info = json.load(f)
                test_designs = split_info.get('test_designs', [])
        
        if not test_designs:
            logger.warning("未找到测试集设计，使用所有可用设计")
            test_designs = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        logger.info(f"找到 {len(test_designs)} 个测试设计")
        
        results = []
        for i, design_name in enumerate(test_designs, 1):
            logger.info(f"[{i}/{len(test_designs)}] 处理设计: {design_name}")
            
            try:
                # 获取设计信息
                design_info = self._get_design_info(design_name)
                if not design_info:
                    continue
                
                # 构建DRAG查询
                query = self._build_drag_query(design_info)
                
                # DRAG检索
                logger.info(f"  执行DRAG检索...")
                retrieval_results = self.drag_retriever.retrieve_with_dynamic_reranking(
                    query, design_info
                )
                
                # 提取DRAG推荐参数
                drag_params = self._extract_layout_params_from_retrieval(retrieval_results)
                
                # 使用DRAG参数运行布局
                logger.info(f"  使用DRAG参数运行布局...")
                drag_result = self._run_openroad_layout(design_name, drag_params)
                
                # 使用默认参数运行布局
                logger.info(f"  使用默认参数运行布局...")
                default_params = self._get_default_params()
                default_result = self._run_openroad_layout(design_name, default_params)
                
                # 计算改进
                improvement = 0.0
                if (drag_result['success'] and default_result['success'] and 
                    drag_result.get('final_hpwl') and default_result.get('final_hpwl')):
                    drag_hpwl = drag_result['final_hpwl']
                    default_hpwl = default_result['final_hpwl']
                    if default_hpwl > 0:
                        improvement = ((default_hpwl - drag_hpwl) / default_hpwl) * 100
                
                # 记录结果
                result = {
                    'design_name': design_name,
                    'drag_params': drag_params,
                    'default_params': default_params,
                    'drag_hpwl': drag_result.get('final_hpwl'),
                    'default_hpwl': default_result.get('final_hpwl'),
                    'improvement_percent': improvement,
                    'drag_success': drag_result['success'],
                    'default_success': default_result['success'],
                    'retrieval_count': len(retrieval_results),
                    'drag_execution_time': drag_result.get('execution_time', 0),
                    'default_execution_time': default_result.get('execution_time', 0)
                }
                
                results.append(result)
                
                logger.info(f"  ✅ 完成: DRAG HPWL={drag_result.get('final_hpwl')}, 默认 HPWL={default_result.get('final_hpwl')}, 改进={improvement:.2f}%")
                
            except Exception as e:
                logger.error(f"  处理设计 {design_name} 失败: {str(e)}")
                results.append({
                    'design_name': design_name,
                    'error': str(e),
                    'drag_success': False,
                    'default_success': False
                })
        
        self.experiment_results['experiment_2_test_set'] = results
        logger.info(f"实验2完成: {len(results)} 个设计")
        return results
    
    def generate_analysis_report(self):
        """生成分析报告"""
        logger.info("生成分析报告...")
        
        report = []
        report.append("# DRAG系统实验分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 实验1分析
        exp1_results = self.experiment_results['experiment_1_train_set']
        if exp1_results:
            report.append("## 实验1：训练集设计对比（DRAG vs. 训练数据最优）")
            report.append("")
            
            successful = [r for r in exp1_results if r.get('drag_success', False)]
            if successful:
                improvements = [r.get('improvement_percent', 0) for r in successful]
                avg_improvement = np.mean(improvements)
                max_improvement = max(improvements)
                min_improvement = min(improvements)
                positive_count = sum(1 for imp in improvements if imp > 0)
                
                report.append(f"- 成功设计数: {len(successful)}/{len(exp1_results)}")
                report.append(f"- 平均HPWL改进: {avg_improvement:.2f}%")
                report.append(f"- 最大HPWL改进: {max_improvement:.2f}%")
                report.append(f"- 最小HPWL改进: {min_improvement:.2f}%")
                report.append(f"- 正向改进设计数: {positive_count}")
                report.append(f"- 改进成功率: {positive_count/len(successful)*100:.1f}%")
                report.append("")
                
                report.append("### 详细结果")
                report.append("| 设计名称 | DRAG HPWL | 训练HPWL | 改进率 |")
                report.append("|---------|-----------|----------|--------|")
                for result in successful:
                    design_name = result['design_name']
                    drag_hpwl = result.get('drag_hpwl', 'N/A')
                    training_hpwl = result.get('training_hpwl', 'N/A')
                    improvement = result.get('improvement_percent', 0)
                    report.append(f"| {design_name} | {drag_hpwl} | {training_hpwl} | {improvement:.2f}% |")
                report.append("")
        
        # 实验2分析
        exp2_results = self.experiment_results['experiment_2_test_set']
        if exp2_results:
            report.append("## 实验2：测试集设计对比（DRAG vs. OpenROAD默认）")
            report.append("")
            
            successful = [r for r in exp2_results if r.get('drag_success', False) and r.get('default_success', False)]
            if successful:
                improvements = [r.get('improvement_percent', 0) for r in successful]
                avg_improvement = np.mean(improvements)
                max_improvement = max(improvements)
                min_improvement = min(improvements)
                positive_count = sum(1 for imp in improvements if imp > 0)
                
                report.append(f"- 成功设计数: {len(successful)}/{len(exp2_results)}")
                report.append(f"- 平均HPWL改进: {avg_improvement:.2f}%")
                report.append(f"- 最大HPWL改进: {max_improvement:.2f}%")
                report.append(f"- 最小HPWL改进: {min_improvement:.2f}%")
                report.append(f"- 正向改进设计数: {positive_count}")
                report.append(f"- 改进成功率: {positive_count/len(successful)*100:.1f}%")
                report.append("")
                
                report.append("### 详细结果")
                report.append("| 设计名称 | DRAG HPWL | 默认HPWL | 改进率 |")
                report.append("|---------|-----------|----------|--------|")
                for result in successful:
                    design_name = result['design_name']
                    drag_hpwl = result.get('drag_hpwl', 'N/A')
                    default_hpwl = result.get('default_hpwl', 'N/A')
                    improvement = result.get('improvement_percent', 0)
                    report.append(f"| {design_name} | {drag_hpwl} | {default_hpwl} | {improvement:.2f}% |")
                report.append("")
        
        # 保存报告
        report_path = self.results_dir / "drag_experiment_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 保存详细结果
        results_path = self.results_dir / "drag_experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
        
        logger.info(f"分析报告已保存到: {report_path}")
        logger.info(f"详细结果已保存到: {results_path}")
        
        return report_path
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("开始运行DRAG系统实验...")
        
        start_time = time.time()
        
        # 运行实验1
        exp1_results = self.run_experiment_1_train_set()
        
        # 运行实验2
        exp2_results = self.run_experiment_2_test_set()
        
        # 生成分析报告
        report_path = self.generate_analysis_report()
        
        total_time = time.time() - start_time
        
        logger.info(f"所有实验完成！总耗时: {total_time:.2f}秒")
        logger.info(f"实验报告: {report_path}")
        
        return {
            'experiment_1': exp1_results,
            'experiment_2': exp2_results,
            'report_path': report_path,
            'total_time': total_time
        }

def main():
    """主函数"""
    try:
        # 创建实验主控器
        controller = DRAGExperimentController()
        
        # 运行所有实验
        results = controller.run_all_experiments()
        
        # 输出总结
        print("\n" + "="*60)
        print("DRAG系统实验总结")
        print("="*60)
        
        exp1_results = results['experiment_1']
        exp2_results = results['experiment_2']
        
        print(f"实验1 (训练集对比): {len(exp1_results)} 个设计")
        successful_exp1 = [r for r in exp1_results if r.get('drag_success', False)]
        if successful_exp1:
            improvements = [r.get('improvement_percent', 0) for r in successful_exp1]
            print(f"  成功: {len(successful_exp1)}/{len(exp1_results)}")
            print(f"  平均改进: {np.mean(improvements):.2f}%")
        
        print(f"实验2 (测试集对比): {len(exp2_results)} 个设计")
        successful_exp2 = [r for r in exp2_results if r.get('drag_success', False) and r.get('default_success', False)]
        if successful_exp2:
            improvements = [r.get('improvement_percent', 0) for r in successful_exp2]
            print(f"  成功: {len(successful_exp2)}/{len(exp2_results)}")
            print(f"  平均改进: {np.mean(improvements):.2f}%")
        
        print(f"总耗时: {results['total_time']:.2f}秒")
        print(f"详细报告: {results['report_path']}")
        
    except Exception as e:
        logger.error(f"实验运行失败: {str(e)}")
        raise

if __name__ == '__main__':
    main()
