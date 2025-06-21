#!/usr/bin/env python3
"""
训练数据准备脚本
为Chip-D-RAG系统准备训练数据
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataPreparer:
    """训练数据准备器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设计类型和约束类型
        self.design_types = ['risc_v', 'dsp', 'memory', 'accelerator', 'controller']
        self.constraint_types = ['timing', 'power', 'area', 'reliability', 'yield']
        self.technology_nodes = ['14nm', '28nm', '40nm', '65nm']
        
    def generate_training_queries(self, num_queries: int = 1000) -> List[Dict[str, Any]]:
        """生成训练查询数据
        
        Args:
            num_queries: 查询数量
            
        Returns:
            List[Dict[str, Any]]: 查询列表
        """
        queries = []
        
        # 查询模板
        query_templates = [
            "Generate layout for {design_type} with {constraints} constraints",
            "Design {design_type} layout optimized for {constraints}",
            "Create {design_type} layout with focus on {constraints}",
            "Optimize {design_type} layout for {constraints} requirements",
            "Generate {design_type} layout considering {constraints} constraints"
        ]
        
        for i in range(num_queries):
            # 随机选择设计类型和约束
            design_type = random.choice(self.design_types)
            num_constraints = random.randint(1, 3)
            constraints = random.sample(self.constraint_types, num_constraints)
            
            # 生成查询文本
            template = random.choice(query_templates)
            query_text = template.format(
                design_type=design_type,
                constraints=', '.join(constraints)
            )
            
            # 添加复杂度
            complexity = random.uniform(0.3, 0.9)
            
            query = {
                'id': f'query_{i:04d}',
                'text': query_text,
                'design_type': design_type,
                'constraints': constraints,
                'complexity': complexity,
                'priority': random.choice(['low', 'medium', 'high']),
                'timestamp': datetime.now().isoformat()
            }
            
            queries.append(query)
            
        return queries
    
    def generate_design_info(self, num_designs: int = 1000) -> List[Dict[str, Any]]:
        """生成设计信息数据
        
        Args:
            num_designs: 设计数量
            
        Returns:
            List[Dict[str, Any]]: 设计信息列表
        """
        designs = []
        
        for i in range(num_designs):
            design_type = random.choice(self.design_types)
            technology_node = random.choice(self.technology_nodes)
            
            # 根据设计类型生成合理的约束范围
            if design_type == 'risc_v':
                area_range = (3000, 8000)
                power_range = (2.0, 8.0)
                timing_range = (1.5, 3.0)
            elif design_type == 'dsp':
                area_range = (2000, 6000)
                power_range = (1.5, 6.0)
                timing_range = (1.0, 2.5)
            elif design_type == 'memory':
                area_range = (1000, 4000)
                power_range = (0.5, 3.0)
                timing_range = (2.0, 4.0)
            else:
                area_range = (1500, 5000)
                power_range = (1.0, 5.0)
                timing_range = (1.5, 3.0)
            
            design = {
                'id': f'design_{i:04d}',
                'design_type': design_type,
                'technology_node': technology_node,
                'area_constraint': random.uniform(*area_range),
                'power_budget': random.uniform(*power_range),
                'timing_constraint': random.uniform(*timing_range),
                'constraints': self._generate_constraints(design_type),
                'metadata': {
                    'generation_method': 'synthetic',
                    'quality_level': random.choice(['low', 'medium', 'high']),
                    'optimization_target': random.choice(['area', 'power', 'timing', 'balanced'])
                }
            }
            
            designs.append(design)
            
        return designs
    
    def _generate_constraints(self, design_type: str) -> List[Dict[str, Any]]:
        """生成约束列表
        
        Args:
            design_type: 设计类型
            
        Returns:
            List[Dict[str, Any]]: 约束列表
        """
        constraints = []
        
        # 基础约束
        base_constraints = ['timing', 'power', 'area']
        
        for constraint_type in base_constraints:
            if constraint_type == 'timing':
                value = random.uniform(0.7, 1.0)
                weight = random.uniform(0.3, 0.5)
            elif constraint_type == 'power':
                value = random.uniform(0.7, 1.0)
                weight = random.uniform(0.2, 0.4)
            else:  # area
                value = random.uniform(0.7, 1.0)
                weight = random.uniform(0.1, 0.3)
            
            constraints.append({
                'type': constraint_type,
                'value': value,
                'weight': weight
            })
        
        # 根据设计类型添加特定约束
        if design_type == 'memory':
            constraints.append({
                'type': 'reliability',
                'value': random.uniform(0.8, 1.0),
                'weight': random.uniform(0.2, 0.3)
            })
        elif design_type == 'accelerator':
            constraints.append({
                'type': 'yield',
                'value': random.uniform(0.7, 1.0),
                'weight': random.uniform(0.1, 0.2)
            })
        
        return constraints
    
    def generate_quality_feedback(self, num_feedbacks: int = 1000) -> List[Dict[str, Any]]:
        """生成质量反馈数据
        
        Args:
            num_feedbacks: 反馈数量
            
        Returns:
            List[Dict[str, Any]]: 反馈列表
        """
        feedbacks = []
        
        for i in range(num_feedbacks):
            # 生成各项质量指标
            wirelength_score = random.uniform(0.5, 0.95)
            congestion_score = random.uniform(0.5, 0.95)
            timing_score = random.uniform(0.5, 0.95)
            power_score = random.uniform(0.5, 0.95)
            
            # 计算整体分数
            overall_score = np.mean([wirelength_score, congestion_score, timing_score, power_score])
            
            feedback = {
                'id': f'feedback_{i:04d}',
                'wirelength_score': wirelength_score,
                'congestion_score': congestion_score,
                'timing_score': timing_score,
                'power_score': power_score,
                'overall_score': overall_score,
                'feedback_timestamp': datetime.now().isoformat(),
                'layout_metadata': {
                    'generation_method': random.choice(['traditional_rag', 'chiprag', 'chip_d_rag']),
                    'k_value': random.randint(3, 15),
                    'reranking_applied': random.choice([True, False]),
                    'entity_enhancement': random.choice([True, False])
                }
            }
            
            feedbacks.append(feedback)
            
        return feedbacks
    
    def generate_training_samples(self, num_samples: int = 2000) -> List[Dict[str, Any]]:
        """生成训练样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            List[Dict[str, Any]]: 训练样本列表
        """
        # 生成基础数据
        queries = self.generate_training_queries(num_samples)
        designs = self.generate_design_info(num_samples)
        feedbacks = self.generate_quality_feedback(num_samples)
        
        # 组合训练样本
        training_samples = []
        
        for i in range(num_samples):
            sample = {
                'id': f'sample_{i:04d}',
                'query': queries[i],
                'design_info': designs[i],
                'expected_quality': feedbacks[i],
                'historical_interactions': self._generate_historical_interactions(),
                'metadata': {
                    'generation_timestamp': datetime.now().isoformat(),
                    'data_source': 'synthetic',
                    'difficulty_level': random.choice(['easy', 'medium', 'hard'])
                }
            }
            
            training_samples.append(sample)
            
        return training_samples
    
    def _generate_historical_interactions(self) -> List[Dict[str, Any]]:
        """生成历史交互数据
        
        Returns:
            List[Dict[str, Any]]: 历史交互列表
        """
        interactions = []
        num_interactions = random.randint(0, 5)
        
        for i in range(num_interactions):
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'k_value': random.randint(3, 15),
                'retrieval_method': random.choice(['traditional', 'chiprag', 'dynamic']),
                'quality_score': random.uniform(0.5, 0.9),
                'user_feedback': random.choice(['positive', 'negative', 'neutral'])
            }
            interactions.append(interaction)
            
        return interactions
    
    def save_training_data(self, training_samples: List[Dict[str, Any]]):
        """保存训练数据
        
        Args:
            training_samples: 训练样本列表
        """
        # 保存完整训练数据
        training_file = self.output_dir / 'training_data.json'
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
        
        # 分别保存各部分数据
        queries = [sample['query'] for sample in training_samples]
        designs = [sample['design_info'] for sample in training_samples]
        feedbacks = [sample['expected_quality'] for sample in training_samples]
        
        # 保存查询数据
        queries_file = self.output_dir / 'queries.json'
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        
        # 保存设计信息
        designs_file = self.output_dir / 'designs.json'
        with open(designs_file, 'w', encoding='utf-8') as f:
            json.dump(designs, f, ensure_ascii=False, indent=2)
        
        # 保存质量反馈
        feedbacks_file = self.output_dir / 'quality_feedbacks.json'
        with open(feedbacks_file, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)
        
        # 生成数据统计报告
        self._generate_data_report(training_samples)
        
        logger.info(f"训练数据已保存到 {self.output_dir}")
        logger.info(f"总样本数: {len(training_samples)}")
    
    def _generate_data_report(self, training_samples: List[Dict[str, Any]]):
        """生成数据统计报告
        
        Args:
            training_samples: 训练样本列表
        """
        # 统计设计类型分布
        design_type_counts = {}
        constraint_type_counts = {}
        quality_scores = []
        
        for sample in training_samples:
            design_type = sample['design_info']['design_type']
            design_type_counts[design_type] = design_type_counts.get(design_type, 0) + 1
            
            for constraint in sample['design_info']['constraints']:
                constraint_type = constraint['type']
                constraint_type_counts[constraint_type] = constraint_type_counts.get(constraint_type, 0) + 1
            
            quality_scores.append(sample['expected_quality']['overall_score'])
        
        # 生成报告
        report = {
            'data_statistics': {
                'total_samples': len(training_samples),
                'design_type_distribution': design_type_counts,
                'constraint_type_distribution': constraint_type_counts,
                'quality_score_statistics': {
                    'mean': np.mean(quality_scores),
                    'std': np.std(quality_scores),
                    'min': np.min(quality_scores),
                    'max': np.max(quality_scores)
                }
            },
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }
        }
        
        # 保存报告
        report_file = self.output_dir / 'data_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("数据统计报告已生成")

def main():
    """主函数"""
    # 配置
    config = {
        'output_dir': 'data/training',
        'num_samples': 2000,
        'random_seed': 42
    }
    
    # 设置随机种子
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # 初始化数据准备器
    preparer = TrainingDataPreparer(config)
    
    # 生成训练数据
    logger.info("开始生成训练数据...")
    training_samples = preparer.generate_training_samples(config['num_samples'])
    
    # 保存数据
    preparer.save_training_data(training_samples)
    
    logger.info("训练数据准备完成！")

if __name__ == '__main__':
    main() 