#!/usr/bin/env python3
"""
训练数据扩充脚本
快速扩充强化学习训练数据
"""

import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import copy

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataExpander:
    """训练数据扩充器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 扩充参数
        self.expansion_factor = config.get('expansion_factor', 5)  # 扩充5倍
        self.noise_level = config.get('noise_level', 0.1)  # 添加10%噪声
        self.constraint_variations = config.get('constraint_variations', True)
        
        logger.info(f"训练数据扩充器初始化完成，目标扩充倍数: {self.expansion_factor}")
    
    def load_existing_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载现有数据
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: 现有数据
        """
        data = {}
        
        # 加载真实数据
        real_data_dir = Path('data/real')
        if real_data_dir.exists():
            for file_path in real_data_dir.glob('*.json'):
                if 'statistics' not in file_path.name and 'summary' not in file_path.name:
                    data_type = file_path.stem
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data[data_type] = json.load(f)
        
        # 加载训练数据
        training_data_path = Path('data/training/training_data.json')
        if training_data_path.exists():
            with open(training_data_path, 'r', encoding='utf-8') as f:
                data['training'] = json.load(f)
        
        logger.info(f"加载了 {len(data)} 种类型的数据")
        return data
    
    def expand_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扩充查询数据
        
        Args:
            queries: 原始查询数据
            
        Returns:
            List[Dict[str, Any]]: 扩充后的查询数据
        """
        expanded_queries = []
        
        for query in queries:
            # 复制原始查询
            expanded_queries.append(query)
            
            # 生成变体
            for i in range(self.expansion_factor - 1):
                variant = self._generate_query_variant(query)
                expanded_queries.append(variant)
        
        logger.info(f"查询数据从 {len(queries)} 扩充到 {len(expanded_queries)}")
        return expanded_queries
    
    def _generate_query_variant(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """生成查询变体
        
        Args:
            query: 原始查询
            
        Returns:
            Dict[str, Any]: 查询变体
        """
        variant = copy.deepcopy(query)
        
        # 修改查询ID
        variant['query_id'] = f"{query['query_id']}_variant_{random.randint(1, 1000)}"
        
        # 添加噪声到复杂度
        if 'complexity' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['complexity'] = max(0.1, min(1.0, variant['complexity'] + noise))
        
        # 变化约束组合
        if self.constraint_variations and 'constraints' in variant:
            constraint_types = ['timing', 'power', 'area', 'reliability', 'yield']
            num_constraints = random.randint(1, 3)
            variant['constraints'] = random.sample(constraint_types, num_constraints)
        
        # 修改时间戳
        variant['timestamp'] = datetime.now().isoformat()
        variant['source'] = 'expanded'
        
        return variant
    
    def expand_designs(self, designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扩充设计数据
        
        Args:
            designs: 原始设计数据
            
        Returns:
            List[Dict[str, Any]]: 扩充后的设计数据
        """
        expanded_designs = []
        
        for design in designs:
            # 复制原始设计
            expanded_designs.append(design)
            
            # 生成变体
            for i in range(self.expansion_factor - 1):
                variant = self._generate_design_variant(design)
                expanded_designs.append(variant)
        
        logger.info(f"设计数据从 {len(designs)} 扩充到 {len(expanded_designs)}")
        return expanded_designs
    
    def _generate_design_variant(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """生成设计变体
        
        Args:
            design: 原始设计
            
        Returns:
            Dict[str, Any]: 设计变体
        """
        variant = copy.deepcopy(design)
        
        # 修改设计ID
        variant['design_id'] = f"{design['design_id']}_variant_{random.randint(1, 1000)}"
        
        # 添加噪声到约束参数
        if 'area_constraint' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['area_constraint'] = max(100, variant['area_constraint'] * (1 + noise))
        
        if 'power_budget' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['power_budget'] = max(0.1, variant['power_budget'] * (1 + noise))
        
        if 'timing_constraint' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['timing_constraint'] = max(0.1, variant['timing_constraint'] * (1 + noise))
        
        # 变化技术节点
        tech_nodes = ['14nm', '28nm', '40nm', '65nm']
        if random.random() < 0.3:  # 30%概率改变技术节点
            variant['technology_node'] = random.choice(tech_nodes)
        
        # 修改时间戳
        variant['timestamp'] = datetime.now().isoformat()
        variant['source'] = 'expanded'
        
        return variant
    
    def expand_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扩充结果数据
        
        Args:
            results: 原始结果数据
            
        Returns:
            List[Dict[str, Any]]: 扩充后的结果数据
        """
        expanded_results = []
        
        for result in results:
            # 复制原始结果
            expanded_results.append(result)
            
            # 生成变体
            for i in range(self.expansion_factor - 1):
                variant = self._generate_result_variant(result)
                expanded_results.append(variant)
        
        logger.info(f"结果数据从 {len(results)} 扩充到 {len(expanded_results)}")
        return expanded_results
    
    def _generate_result_variant(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成结果变体
        
        Args:
            result: 原始结果
            
        Returns:
            Dict[str, Any]: 结果变体
        """
        variant = copy.deepcopy(result)
        
        # 修改结果ID
        variant['result_id'] = f"{result['result_id']}_variant_{random.randint(1, 1000)}"
        
        # 添加噪声到质量指标
        if 'wirelength' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['wirelength'] = max(100, variant['wirelength'] * (1 + noise))
        
        if 'congestion' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['congestion'] = max(0.01, min(1.0, variant['congestion'] + noise))
        
        if 'timing_score' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['timing_score'] = max(0.1, min(1.0, variant['timing_score'] + noise))
        
        if 'power_score' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['power_score'] = max(0.1, min(1.0, variant['power_score'] + noise))
        
        if 'area_utilization' in variant:
            noise = random.uniform(-self.noise_level, self.noise_level)
            variant['area_utilization'] = max(0.1, min(1.0, variant['area_utilization'] + noise))
        
        # 变化生成方法
        methods = ['chip_d_rag', 'traditional', 'baseline']
        if random.random() < 0.2:  # 20%概率改变方法
            variant['method'] = random.choice(methods)
        
        # 修改时间戳
        variant['timestamp'] = datetime.now().isoformat()
        
        return variant
    
    def expand_feedback(self, feedbacks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扩充反馈数据
        
        Args:
            feedbacks: 原始反馈数据
            
        Returns:
            List[Dict[str, Any]]: 扩充后的反馈数据
        """
        expanded_feedbacks = []
        
        for feedback in feedbacks:
            # 复制原始反馈
            expanded_feedbacks.append(feedback)
            
            # 生成变体
            for i in range(self.expansion_factor - 1):
                variant = self._generate_feedback_variant(feedback)
                expanded_feedbacks.append(variant)
        
        logger.info(f"反馈数据从 {len(feedbacks)} 扩充到 {len(expanded_feedbacks)}")
        return expanded_feedbacks
    
    def _generate_feedback_variant(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """生成反馈变体
        
        Args:
            feedback: 原始反馈
            
        Returns:
            Dict[str, Any]: 反馈变体
        """
        variant = copy.deepcopy(feedback)
        
        # 修改反馈ID
        variant['feedback_id'] = f"{feedback['feedback_id']}_variant_{random.randint(1, 1000)}"
        
        # 添加噪声到评分
        score_fields = ['wirelength_score', 'congestion_score', 'timing_score', 'power_score', 'overall_score']
        for field in score_fields:
            if field in variant:
                noise = random.uniform(-self.noise_level, self.noise_level)
                variant[field] = max(0.1, min(1.0, variant[field] + noise))
        
        # 变化反馈文本
        feedback_texts = [
            "Good layout quality, meets timing requirements",
            "Layout is well optimized for power",
            "Area utilization is efficient",
            "Some congestion issues in certain areas",
            "Timing closure achieved successfully",
            "Power optimization could be improved",
            "Overall layout quality is satisfactory",
            "Minor timing violations detected",
            "Excellent area utilization",
            "Good balance between timing and power"
        ]
        if random.random() < 0.3:  # 30%概率改变反馈文本
            variant['feedback_text'] = random.choice(feedback_texts)
        
        # 修改时间戳
        variant['timestamp'] = datetime.now().isoformat()
        
        return variant
    
    def generate_training_pairs(self, expanded_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """生成训练数据对
        
        Args:
            expanded_data: 扩充后的数据
            
        Returns:
            List[Dict[str, Any]]: 训练数据对
        """
        training_pairs = []
        
        queries = expanded_data.get('sample_queries', [])
        designs = expanded_data.get('sample_designs', [])
        results = expanded_data.get('sample_results', [])
        feedbacks = expanded_data.get('sample_feedback', [])
        
        # 创建查询-设计-结果-反馈的四元组
        for i in range(min(len(queries), len(designs), len(results), len(feedbacks))):
            # 确保ID匹配
            query = queries[i % len(queries)]
            design = designs[i % len(designs)]
            result = results[i % len(results)]
            feedback = feedbacks[i % len(feedbacks)]
            
            # 创建训练对
            training_pair = {
                'query': query,
                'design_info': design,
                'layout_result': result,
                'quality_feedback': feedback,
                'expected_quality': {
                    'overall_score': feedback.get('overall_score', 0.7),
                    'wirelength_score': feedback.get('wirelength_score', 0.7),
                    'congestion_score': feedback.get('congestion_score', 0.7),
                    'timing_score': feedback.get('timing_score', 0.7),
                    'power_score': feedback.get('power_score', 0.7)
                },
                'training_id': f"training_pair_{i:04d}",
                'timestamp': datetime.now().isoformat()
            }
            
            training_pairs.append(training_pair)
        
        logger.info(f"生成了 {len(training_pairs)} 个训练数据对")
        return training_pairs
    
    def save_expanded_data(self, expanded_data: Dict[str, List[Dict[str, Any]]], training_pairs: List[Dict[str, Any]]):
        """保存扩充后的数据
        
        Args:
            expanded_data: 扩充后的数据
            training_pairs: 训练数据对
        """
        # 保存扩充后的各类数据
        for data_type, data_list in expanded_data.items():
            output_file = self.output_dir / f'expanded_{data_type}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            logger.info(f"扩充后的 {data_type} 数据已保存到 {output_file}")
        
        # 保存训练数据对
        training_file = self.output_dir / 'expanded_training_data.json'
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, ensure_ascii=False, indent=2)
        logger.info(f"训练数据对已保存到 {training_file}")
        
        # 生成数据报告
        self._generate_expansion_report(expanded_data, training_pairs)
    
    def _generate_expansion_report(self, expanded_data: Dict[str, List[Dict[str, Any]]], training_pairs: List[Dict[str, Any]]):
        """生成扩充报告
        
        Args:
            expanded_data: 扩充后的数据
            training_pairs: 训练数据对
        """
        report = {
            'expansion_summary': {
                'expansion_factor': self.expansion_factor,
                'noise_level': self.noise_level,
                'constraint_variations': self.constraint_variations
            },
            'data_statistics': {
                'total_training_pairs': len(training_pairs),
                'data_types': {}
            },
            'expansion_timestamp': datetime.now().isoformat()
        }
        
        # 统计各类数据
        for data_type, data_list in expanded_data.items():
            report['data_statistics']['data_types'][data_type] = len(data_list)
        
        # 保存报告
        report_file = self.output_dir / 'expansion_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"扩充报告已生成: {report_file}")
    
    def expand_all_data(self):
        """扩充所有数据"""
        logger.info("开始扩充训练数据...")
        
        # 加载现有数据
        existing_data = self.load_existing_data()
        
        # 扩充各类数据
        expanded_data = {}
        
        if 'sample_queries' in existing_data:
            expanded_data['sample_queries'] = self.expand_queries(existing_data['sample_queries'])
        
        if 'sample_designs' in existing_data:
            expanded_data['sample_designs'] = self.expand_designs(existing_data['sample_designs'])
        
        if 'sample_results' in existing_data:
            expanded_data['sample_results'] = self.expand_results(existing_data['sample_results'])
        
        if 'sample_feedback' in existing_data:
            expanded_data['sample_feedback'] = self.expand_feedback(existing_data['sample_feedback'])
        
        # 生成训练数据对
        training_pairs = self.generate_training_pairs(expanded_data)
        
        # 保存扩充后的数据
        self.save_expanded_data(expanded_data, training_pairs)
        
        logger.info("训练数据扩充完成！")

def main():
    """主函数"""
    # 配置
    config = {
        'output_dir': 'data/training',
        'expansion_factor': 5,  # 扩充5倍
        'noise_level': 0.1,     # 10%噪声
        'constraint_variations': True
    }
    
    # 初始化扩充器
    expander = TrainingDataExpander(config)
    
    # 扩充数据
    expander.expand_all_data()
    
    logger.info("数据扩充脚本执行完成！")

if __name__ == '__main__':
    main() 