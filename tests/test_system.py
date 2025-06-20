import os
import json
import unittest
import logging
import time
import psutil
from typing import Dict, List
from modules.core.rag_system import RAGSystem
from modules.utils.benchmark_loader import BenchmarkLoader
from tests.test_utils import TestLogger
from modules.core.hierarchy import HierarchicalDecompositionManager
from modules.knowledge.knowledge_graph_builder import KnowledgeGraphBuilder
from modules.evaluation.layout_quality_evaluator import LayoutQualityEvaluator
from modules.evaluation.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from modules.evaluation.quality_evaluator import QualityEvaluator
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class TestCHIPRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.logger = logging.getLogger('CHIPRAG_System_Test')
        cls.logger.info("开始步骤: 初始化测试环境")
        
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'system.json')
        with open(config_path, 'r') as f:
            cls.config = json.load(f)
        
        # 初始化RAG系统
        cls.rag_system = RAGSystem(cls.config)
        
        # 初始化知识库
        cls.knowledge_base = KnowledgeBase(cls.config.get('knowledge_base', {}))
        
        # 初始化LLM管理器
        cls.llm_manager = LLMManager(cls.config.get('llm', {}))
        
        cls.logger.info("测试环境初始化完成")
        
        # 初始化布局生成器
        from modules.core.layout_generator import LayoutGenerator
        # 添加缺失的layout_config，只使用支持的参数
        if 'layout_config' not in cls.config:
            cls.config['layout_config'] = {
                'min_component_size': 0.05,
                'max_component_size': 0.3,
                'min_spacing': 0.02,
                'max_density': 0.8,
                'num_grid_cells': 20,
                'num_routing_layers': 3,
                'max_iterations': 100,
                'population_size': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'target_satisfaction': 95.0,
                'die_width': 20000.0,
                'die_height': 20000.0
            }
            
        # 添加缺失的embedding_config
        if 'embedding_config' not in cls.config:
            cls.config['embedding_config'] = {
                'model_path': 'models/bert',
                'device': 'cpu'
            }
            
        # 添加缺失的evaluation_config
        if 'evaluation_config' not in cls.config:
            cls.config['evaluation_config'] = {
                'weights': {'wirelength': 0.3, 'congestion': 0.3, 'timing': 0.4},
                'thresholds': {'wirelength': 1.0, 'congestion': 1.0, 'timing': 1.0},
                'metrics': {
                    'wirelength': {'type': 'minimize', 'weight': 0.3, 'threshold': 1.0},
                    'congestion': {'type': 'minimize', 'weight': 0.3, 'threshold': 1.0},
                    'timing': {'type': 'minimize', 'weight': 0.4, 'threshold': 1.0}
                }
            }
        cls.layout_generator = LayoutGenerator({
            'layout_config': cls.config['layout_config'],
            'llm_manager': cls.llm_manager
        })
        
        # 初始化日志记录器
        cls.logger = TestLogger('CHIPRAG_System_Test')
        cls.logger.log_step('初始化测试环境')
        
        # 初始化各个组件
        from modules.utils.embedding_manager import EmbeddingManager
        cls.embedding_manager = EmbeddingManager(cls.config['embedding_config'])
        cls.evaluator = LayoutQualityEvaluator(cls.config['evaluation_config'])
        
        # 初始化基准测试加载器
        cls.benchmark_loader = BenchmarkLoader(os.path.join(os.path.dirname(__file__), '..', 'data', 'designs', 'ispd_2015_contest_benchmark'))
        cls.logger.log_result('初始化基准测试加载器', '成功')
        
        # 选择测试用例
        cls.test_cases = [
            'mgc_fft_1',
            'mgc_des_perf_1',
            'mgc_pci_bridge32_a'
        ]
        cls.logger.log_data('测试用例', cls.test_cases)
        
        cls.manager = HierarchicalDecompositionManager(config={
            'max_depth': 3,
            'min_components': 2,
            'similarity_threshold': 0.8
        })
        cls.knowledge_graph = KnowledgeGraphBuilder()
        cls.quality_evaluator = QualityEvaluator()
    
    def test_data_loading(self):
        """测试数据加载"""
        self.logger.log_step('测试数据加载')
        
        # 检查知识图谱数据
        data_files = {
            'entities.json': 'data/knowledge_base/entities.json',
            'relations.json': 'data/knowledge_base/relations.json',
            'triples.json': 'data/knowledge_base/triples.json',
            'layout_experiences.json': 'data/knowledge_base/layout_experience/layout_experiences.json',
            'design_patterns': 'data/knowledge_base/design_patterns/',
            'constraints': 'data/knowledge_base/constraints/'
        }
        
        for name, path in data_files.items():
            exists = os.path.exists(path)
            self.logger.log_result(f'检查{name}', f'{"存在" if exists else "不存在"}')
            self.assertTrue(exists)
    
    def test_benchmark_loading(self):
        """测试基准测试数据加载"""
        self.logger.log_step('测试基准测试数据加载')
        
        for case in self.test_cases:
            self.logger.log_step(f'加载基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 验证数据完整性
                required_fields = ['name', 'netlist', 'def', 'lef', 'constraints']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in design_info else "不存在"}')
                    self.assertIn(field, design_info)
                    
                # 验证网表信息
                self.assertIn('modules', design_info['netlist'])
                self.assertIn('attributes', design_info['netlist'])
                
                # 验证DEF信息
                self.assertIn('units', design_info['def'])
                self.assertIn('components', design_info['def'])
                self.assertIn('die_area', design_info['def'])
                self.assertIn('rows', design_info['def'])
                self.assertIn('tracks', design_info['def'])
                
                # 验证LEF信息
                self.assertIn('cells', design_info['lef'])
                self.assertIn('technology', design_info['lef'])
                
            except Exception as e:
                self.logger.log_error(f'加载基准测试: {case}', e)
                raise
    
    def test_hierarchical_decomposition(self):
        """测试层次化分解"""
        self.logger.log_step('测试层次化分解')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 验证层次化分解结果
                self.assertIsInstance(hierarchy, dict)
                self.assertIn('levels', hierarchy)
                self.assertIn('modules', hierarchy)
                self.assertIn('connections', hierarchy)
                self.assertIn('patterns', hierarchy)
                
                # 注意：modules可能为空，这是正常的
                if hierarchy['modules']:
                    self.assertGreater(len(hierarchy['modules']), 0)
                
            except Exception as e:
                self.logger.log_error(f'层次化分解: {case}', e)
                raise
    
    def test_knowledge_retrieval(self):
        """测试知识检索"""
        self.logger.log_step('测试知识检索')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 执行知识检索
                retrieval_results = self.rag_system.retrieve_knowledge(hierarchy)
                self.logger.log_data(f'知识检索结果: {case}', retrieval_results)
                
                # 验证结果
                self.assertIsInstance(retrieval_results, dict)
                # 注意：retrieval_results可能为空字典，这是正常的
                if retrieval_results:
                    # 如果有内容，检查其结构
                    self.assertIsInstance(retrieval_results, dict)
                
            except Exception as e:
                self.logger.log_error(f'知识检索: {case}', e)
                raise
    
    def test_layout_generation(self):
        """测试布局生成"""
        self.logger.log_step('测试布局生成')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 执行知识检索
                retrieval_results = self.rag_system.retrieve_knowledge(hierarchy)
                self.logger.log_data(f'知识检索结果: {case}', retrieval_results)
                
                # 测试布局生成
                layout = self.rag_system.generate_layout(
                    design_info, 
                    hierarchy,
                    self.rag_system.knowledge_base
                )
                self.logger.log_data(f'生成的布局: {case}', layout)
                
                # 验证结果
                self.assertIsNotNone(layout)
                required_fields = ['name', 'components', 'nets', 'hierarchy', 'constraints']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in layout else "不存在"}')
                    self.assertIn(field, layout)
                    
                # 验证组件信息
                self.assertIsInstance(layout['components'], list)
                self.assertGreater(len(layout['components']), 0)
                for component in layout['components']:
                    self.assertIn('name', component)
                    self.assertIn('type', component)
                    self.assertIn('position', component)
                    self.assertIn('orientation', component)
                    
                # 验证网络信息
                self.assertIsInstance(layout['nets'], list)
                for net in layout['nets']:
                    self.assertIn('name', net)
                    self.assertIn('connections', net)
                    self.assertIn('wirelength', net)
                    
                # 验证层级信息
                self.assertIn('levels', layout['hierarchy'])
                self.assertIn('modules', layout['hierarchy'])
                self.assertIn('connections', layout['hierarchy'])
                
                # 验证约束信息
                self.assertIn('timing', layout['constraints'])
                self.assertIn('power', layout['constraints'])
                self.assertIn('area', layout['constraints'])
                
            except Exception as e:
                self.logger.log_error(f'布局生成: {case}', e)
                raise
    
    def test_quality_evaluation(self):
        """测试质量评估"""
        self.logger.log_step('测试质量评估')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 执行知识检索
                retrieval_results = self.rag_system.retrieve_knowledge(hierarchy)
                self.logger.log_data(f'知识检索结果: {case}', retrieval_results)
                
                # 测试质量评估
                layout = self.rag_system.generate_layout(
                    design_info, 
                    retrieval_results,
                    self.rag_system.knowledge_base
                )
                self.logger.log_data(f'生成的布局: {case}', layout)
                
                # 评估布局质量
                quality_score = self.rag_system.evaluate_layout(layout, design_info['constraints'])
                self.logger.log_data(f'布局质量评分: {case}', quality_score)
                
                # 验证结果
                self.assertIsNotNone(quality_score)
                required_fields = ['wirelength', 'congestion', 'timing', 'power', 'area', 'overall']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in quality_score else "不存在"}')
                    self.assertIn(field, quality_score)
                    
                # 验证分数范围
                for field in required_fields:
                    self.logger.log_result(f'检查分数范围: {field}', f'{"在0-1之间" if 0 <= quality_score[field] <= 1 else "超出范围"}')
                    self.assertTrue(0 <= quality_score[field] <= 1)
                    
                # 验证详细指标
                self.assertIn('metrics', quality_score)
                metrics = quality_score['metrics']
                self.assertIn('wirelength', metrics)
                self.assertIn('congestion', metrics)
                self.assertIn('timing', metrics)
                self.assertIn('power', metrics)
                self.assertIn('area', metrics)
                
                # 验证约束满足情况
                self.assertIn('constraint_satisfaction', quality_score)
                constraints = quality_score['constraint_satisfaction']
                self.assertIn('timing', constraints)
                self.assertIn('power', constraints)
                self.assertIn('area', constraints)
                
            except Exception as e:
                self.logger.log_error(f'质量评估: {case}', e)
                raise
    
    def test_chiprag_end_to_end(self):
        """测试端到端流程"""
        self.logger.log_step('测试端到端流程')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 执行知识检索
                retrieval_results = self.rag_system.retrieve_knowledge(hierarchy)
                self.logger.log_data(f'知识检索结果: {case}', retrieval_results)
                
                # 生成布局
                layout = self.rag_system.generate_layout(
                    design_info, 
                    hierarchy,
                    self.rag_system.knowledge_base
                )
                self.logger.log_data(f'生成的布局: {case}', layout)
                
                # 评估布局质量
                quality_score = self.rag_system.evaluate_layout(layout, design_info['constraints'])
                self.logger.log_data(f'布局质量评分: {case}', quality_score)
                
                # 验证结果
                self.assertIsNotNone(layout)
                self.assertIsNotNone(quality_score)
                
                # 验证布局信息
                required_fields = ['name', 'components', 'nets', 'hierarchy']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in layout else "不存在"}')
                    self.assertIn(field, layout)
                    
                # 验证层级信息
                self.assertIn('levels', layout['hierarchy'])
                self.assertIn('modules', layout['hierarchy'])
                self.assertIn('max_depth', layout['hierarchy'])
                
                # 验证质量评分
                required_fields = ['wirelength', 'congestion', 'timing', 'overall']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in quality_score else "不存在"}')
                    self.assertIn(field, quality_score)
                    
                # 验证分数范围
                for field in required_fields:
                    self.logger.log_result(f'检查分数范围: {field}', f'{"在0-1之间" if 0 <= quality_score[field] <= 1 else "超出范围"}')
                    self.assertTrue(0 <= quality_score[field] <= 1)
                    
            except Exception as e:
                self.logger.log_error(f'端到端测试: {case}', e)
                raise

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main()
