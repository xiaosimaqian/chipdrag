from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from collections import defaultdict
from ..evaluation.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from ..knowledge.knowledge_base import KnowledgeBase
import json
import pickle
import os
from datetime import datetime
from functools import lru_cache
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import random
import sys
import hashlib
import logging
from ..utils.llm_manager import LLMManager
from ..utils.embedding_manager import EmbeddingManager
from .layout_generator import LayoutGenerator
from ..evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from .knowledge_transfer import KnowledgeTransfer
import time
import psutil
import gc
from pathlib import Path
import PyPDF2
import torch
from .chip_retriever import ChipRetriever

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG系统核心类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化RAG系统
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self):
        """初始化系统组件"""
        try:
            # 初始化知识库
            kb_config = self.config.get('knowledge_base', {})
            
            # 确保知识库配置正确
            if 'path' not in kb_config:
                kb_config['path'] = 'data/knowledge_base'
            if 'format' not in kb_config:
                kb_config['format'] = 'pkl'
            if 'layout_experience' not in kb_config:
                kb_config['layout_experience'] = 'data/knowledge_base'
                
            self.knowledge_base = KnowledgeBase(kb_config)
            
            # 添加调试信息
            logger.info(f"知识库初始化完成，包含 {len(self.knowledge_base.cases)} 个案例")
            
            # 初始化检索器
            self.retriever = ChipRetriever(self.config.get('retriever', {}))
            
            # 初始化LLM管理器
            llm_config = self.config.get('llm', {})
            if isinstance(llm_config, dict):
                self.llm_manager = LLMManager(llm_config)
            else:
                logger.warning(f"LLM配置格式错误，期望字典但得到: {type(llm_config)}")
                self.llm_manager = LLMManager({})
        
            # 初始化评估器
            self.evaluator = MultiObjectiveEvaluator()
        
            self.logger.info("RAG系统组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {str(e)}")
            raise
    
    def generate_layout(self, design_info: Dict, hierarchy_info: Dict, knowledge_base: Dict) -> Dict:
        """生成布局
        
        Args:
            design_info: 设计信息
            hierarchy_info: 层次化信息
            knowledge_base: 知识库
            
        Returns:
            生成的布局
        """
        logger.info("开始生成布局")
        self._log_memory_usage("开始生成布局")
        
        # 1. 获取相似案例
        similar_cases = self.knowledge_base.get_similar_cases(design_info, top_k=5)
        logger.info(f"找到 {len(similar_cases)} 个相似案例")
        
        # 2. 提取布局模式
        layout_patterns = self._extract_layout_patterns(similar_cases)
        logger.info(f"提取出 {len(layout_patterns)} 个布局模式")
        
        # 3. 初始化布局结果
        layout_result = {
            'name': design_info.get('name', 'unknown'),
            'components': [],
            'nets': design_info.get('nets', []),
            'area_utilization': 0.8,
            'wirelength': 0.75,
            'congestion': 0.7,
            'timing': 0.8,
            'power': 0.75,
            'optimization_suggestions': [
                '调整组件位置以减少拥塞',
                '优化布线以减少线长',
                '平衡功耗分布'
            ],
            'metadata': {
                'source': 'llm_optimization',
                'timestamp': '2024-01-01T00:00:00Z',
                'version': '1.0'
            }
        }
        
        # 4. 生成基本组件布局
        components = design_info.get('components', [])
        if not components:
            # 如果没有组件信息，创建一些基本组件
            layout_result['components'] = [
                {
                    'name': 'comp1',
                    'type': 'macro',
                    'position': {'x': 100, 'y': 100},
                    'orientation': 'N',
                    'width': 200,
                    'height': 200
                },
                {
                    'name': 'comp2', 
                    'type': 'standard_cell',
                    'position': {'x': 400, 'y': 100},
                    'orientation': 'N',
                    'width': 150,
                    'height': 150
                },
                {
                    'name': 'comp3',
                    'type': 'io_pad',
                    'position': {'x': 100, 'y': 400},
                    'orientation': 'N',
                    'width': 100,
                    'height': 100
                }
            ]
        else:
            # 使用设计信息中的组件
            layout_result['components'] = []
            for i, comp in enumerate(components):
                layout_result['components'].append({
                    'name': comp.get('name', f'comp{i+1}'),
                    'type': comp.get('type', 'standard_cell'),
                    'position': comp.get('position', {'x': i * 200, 'y': 100}),
                    'orientation': comp.get('orientation', 'N'),
                    'width': comp.get('width', 100),
                    'height': comp.get('height', 100)
                })
        
        # 5. 分批处理组件
        batch_size = 1000
        total_components = len(design_info.get('components', []))
        logger.info(f"总共需要处理 {total_components} 个组件")
        
        for i in range(0, total_components, batch_size):
            batch_end = min(i + batch_size, total_components)
            logger.info(f"处理第 {i//batch_size + 1} 批组件 ({i+1}-{batch_end})")
            
            # 提取当前批次的组件
            batch_components = design_info.get('components', [])[i:batch_end]
            
            # 获取相关的网络
            batch_nets = self._get_related_nets(batch_components, design_info.get('nets', []))
            
            # 使用RAG系统增强知识
            enhanced_knowledge = self.retrieve_and_enhance(
                batch_components,
                hierarchy_info
            )
            
            # 生成当前批次的布局
            batch_layout = self._generate_batch_layout(
                batch_components,
                batch_nets,
                hierarchy_info,
                enhanced_knowledge,
                layout_patterns  # 添加布局模式
            )
            
            # 合并结果
            layout_result['components'].extend(batch_layout['components'])
            
            # 检查内存使用
            self._check_memory_and_cleanup()
            
            # 记录进度
            progress = (batch_end / total_components) * 100
            logger.info(f"处理进度: {progress:.1f}%")
            self._log_memory_usage(f"处理进度 {progress:.1f}%")
        
        # 6. 使用LLM分析和优化
        if self.llm_manager:
            # 6.1 使用原有方法进行分析
            basic_analysis = self.llm_manager.analyze_layout(layout_result)
            logger.info(f"基础分析结果: {basic_analysis}")
            
            # 6.2 使用新方法进行详细分析
            detailed_analysis = self.llm_manager.analyze_layout_detailed(layout_result)
            logger.info(f"详细分析结果: {detailed_analysis}")
            
            # 6.3 结合两种分析结果
            combined_analysis = self._combine_analysis_results(basic_analysis, detailed_analysis)
            logger.info(f"结合后的分析结果: {combined_analysis}")
            
            # 6.4 如果需要优化
            if combined_analysis['needs_optimization']:
                # 6.4.1 使用原有方法进行优化
                basic_optimized = self.llm_manager.optimize_layout(
                    layout_result,
                    combined_analysis['suggestions']
                )
                logger.info("基础优化完成")
                
                # 6.4.2 使用新方法进行详细优化
                detailed_optimized = self.llm_manager.optimize_layout_detailed(
                    layout_result,
                    combined_analysis
                )
                logger.info("详细优化完成")
                
                # 6.4.3 选择更好的优化结果
                layout_result = self._select_better_layout(
                    basic_optimized,
                    detailed_optimized,
                    combined_analysis
                )
                logger.info("已选择更好的优化结果")
                
                # 6.4.4 添加分析信息
                layout_result['analysis'] = {
                    'score': combined_analysis['score'],
                    'suggestions': combined_analysis['suggestions'],
                    'optimization_metrics': combined_analysis['optimization_metrics'],
                    'basic_analysis': basic_analysis,
                    'detailed_analysis': detailed_analysis
                }
        
        # 7. 多目标评估
        evaluator = MultiObjectiveEvaluator()
        evaluation_result = evaluator.evaluate_layout(layout_result)
        logger.info(f"多目标评估结果: {evaluation_result}")
        
        # 8. 保存到知识库
        self.knowledge_base.add_case(
            layout=layout_result,
            optimization_result=evaluation_result,
            metadata={
                'design_info': design_info,
                'hierarchy_info': hierarchy_info,
                'similar_cases': [case['id'] for case in similar_cases],
                'layout_patterns': layout_patterns
            }
        )
        
        # 9. 确保返回结果包含所有必要字段
        final_result = layout_result.copy()
        if 'name' not in final_result:
            final_result['name'] = design_info.get('name', 'unknown')
        if 'nets' not in final_result:
            final_result['nets'] = design_info.get('nets', [])
        if 'die_area' not in final_result:
            final_result['die_area'] = design_info.get('die_area', {'width': 1000, 'height': 1000})
        
        # 10. 添加层次化信息 - 使用传入的 hierarchy_info 而不是知识检索结果
        final_result['hierarchy'] = hierarchy_info
        
        # 确保约束字段包含所有必要的子字段
        constraints = design_info.get('constraints', {})
        final_result['constraints'] = {
            'timing': constraints.get('timing', []),
            'power': constraints.get('power', []),
            'area': constraints.get('area', []),
            'placement': constraints.get('placement', [])
        }
        
        logger.info("布局生成完成")
        self._log_memory_usage("布局生成完成")
        
        return final_result
    
    def _combine_analysis_results(self, basic: Dict, detailed: Dict) -> Dict:
        """结合两种分析结果
        
        Args:
            basic: 基础分析结果
            detailed: 详细分析结果
            
        Returns:
            结合后的分析结果
        """
        # 使用详细分析中的具体调整建议
        combined = detailed.copy()
        
        # 合并建议
        if 'suggestions' in basic:
            combined['suggestions'] = basic['suggestions']
        
        # 如果详细分析失败，使用基础分析
        if not detailed.get('layout_adjustments') and not detailed.get('routing_adjustments'):
            combined['needs_optimization'] = basic.get('needs_optimization', True)
            combined['score'] = basic.get('score', 0.5)
        
        # 确保所有必要字段都存在
        if 'suggestions' not in combined:
            combined['suggestions'] = []
        if 'optimization_metrics' not in combined:
            combined['optimization_metrics'] = {
                'wirelength': 0,
                'congestion': 0,
                'timing': 0
            }
        if 'needs_optimization' not in combined:
            combined['needs_optimization'] = True
        if 'score' not in combined:
            combined['score'] = 0.5
        
        return combined
    
    def _select_better_layout(self, basic: Dict, detailed: Dict, analysis: Dict) -> Dict:
        """选择更好的布局
        
        Args:
            basic: 基础优化结果
            detailed: 详细优化结果
            analysis: 分析结果
            
        Returns:
            更好的布局
        """
        try:
            # 计算两个布局的指标
            basic_metrics = self.llm_manager._calculate_layout_metrics(basic)
            detailed_metrics = self.llm_manager._calculate_layout_metrics(detailed)
            
            # 计算综合得分
            def calculate_score(metrics):
                # 归一化指标，添加默认值处理
                wirelength_score = 1 - min(metrics.get('wirelength', 0) / 1000, 1)
                congestion_score = 1 - metrics.get('congestion', 0)
                timing_score = 1 - min(metrics.get('timing', 0) / 5, 1)
                
                # 加权平均
                return (wirelength_score * 0.4 + 
                        congestion_score * 0.3 + 
                        timing_score * 0.3)
            
            basic_score = calculate_score(basic_metrics)
            detailed_score = calculate_score(detailed_metrics)
            
            # 选择得分更高的布局
            if detailed_score > basic_score:
                logger.info(f"选择详细优化结果 (得分: {detailed_score:.2f} vs {basic_score:.2f})")
                return detailed
            else:
                logger.info(f"选择基础优化结果 (得分: {basic_score:.2f} vs {detailed_score:.2f})")
                return basic
                
        except Exception as e:
            logger.error(f"选择布局失败: {str(e)}")
            # 如果比较失败，返回原始布局
            return basic
    
    def _get_related_nets(self, components: List[Dict], nets: List[Dict]) -> List[Dict]:
        """获取与组件相关的网络
        
        Args:
            components: 组件列表
            nets: 网络列表
            
        Returns:
            相关网络列表
        """
        component_names = {comp['name'] for comp in components}
        return [
            net for net in nets
            if any(pin['component'] in component_names for pin in net['pins'])
        ]
    
    def _check_memory_and_cleanup(self):
        """检查内存使用并清理"""
        current_time = time.time()
        if current_time - self.last_memory_check >= self.memory_check_interval:
            self.last_memory_check = current_time
            
            # 获取内存使用率
            memory_percent = psutil.Process().memory_percent()
            
            if memory_percent > self.memory_threshold * 100:
                logger.warning(f"内存使用率过高: {memory_percent:.1f}%")
                # 清理内存
                gc.collect()
                self._log_memory_usage("内存清理后")
    
    def _log_memory_usage(self, stage: str):
        """记录内存使用
        
        Args:
            stage: 阶段描述
        """
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        logger.info(f"内存使用 ({stage}): {memory:.1f} MB")

    def _extract_layout_patterns(self, similar_cases: List[Dict]) -> List[Dict]:
        """提取布局模式
        
        Args:
            similar_cases: 相似案例列表
            
        Returns:
            布局模式列表
        """
        patterns = []
        for case in similar_cases:
            layout = case['layout']
            # 提取组件类型分布
            component_types = defaultdict(list)
            for comp in layout['components']:
                component_types[comp['type']].append({
                    'x': comp['x'],
                    'y': comp['y'],
                    'width': comp['width'],
                    'height': comp['height']
                })
            
            # 提取网络连接模式
            net_patterns = defaultdict(list)
            for net in layout.get('nets', []):
                net_type = self._get_net_type(net)
                net_patterns[net_type].append({
                    'route': net.get('route', []),
                    'weight': net.get('weight', 1.0)
                })
            
            # 生成布局模式
            pattern = {
                'component_types': dict(component_types),
                'net_patterns': dict(net_patterns),
                'score': case.get('score', 0),
                'case_id': case['id']
            }
            patterns.append(pattern)
        
        return patterns
    
    def _get_net_type(self, net: Dict) -> str:
        """获取网络类型
        
        Args:
            net: 网络信息
            
        Returns:
            网络类型
        """
        # 根据连接的组件类型判断网络类型
        component_types = set()
        for pin in net['pins']:
            comp_name = pin['component']
            comp = next((c for c in self.current_layout['components'] 
                        if c['name'] == comp_name), None)
            if comp:
                component_types.add(comp['type'])
        
        if len(component_types) == 1:
            return f"{list(component_types)[0]}_internal"
        else:
            return f"{'_'.join(sorted(component_types))}_connection"
    
    def _generate_batch_layout(self, 
                             batch_components: List[Dict],
                             batch_nets: List[Dict],
                             hierarchy_info: Dict,
                             enhanced_knowledge: Dict,
                             layout_patterns: List[Dict]) -> Dict:
        """生成批次布局
        
        Args:
            batch_components: 批次组件列表
            batch_nets: 相关网络列表
            hierarchy_info: 层次化信息
            enhanced_knowledge: 增强知识
            layout_patterns: 布局模式列表
            
        Returns:
            批次布局结果
        """
        # 1. 按组件类型分组
        component_groups = defaultdict(list)
        for comp in batch_components:
            component_groups[comp['type']].append(comp)
        
        # 2. 为每种类型选择最佳布局模式
        layout_result = {
            'components': [],
            'nets': batch_nets
        }
        
        for comp_type, comps in component_groups.items():
            # 查找匹配的布局模式
            matching_patterns = [
                p for p in layout_patterns
                if comp_type in p['component_types']
            ]
            
            if matching_patterns:
                # 选择得分最高的模式
                best_pattern = max(matching_patterns, key=lambda p: p['score'])
                pattern_comps = best_pattern['component_types'][comp_type]
                
                # 应用布局模式
                for i, comp in enumerate(comps):
                    if i < len(pattern_comps):
                        pattern = pattern_comps[i]
                        layout_result['components'].append({
                            'name': comp['name'],
                            'type': comp['type'],
                            'x': pattern['x'],
                            'y': pattern['y'],
                            'width': pattern['width'],
                            'height': pattern['height']
                        })
                    else:
                        # 如果没有匹配的模式，使用默认布局
                        layout_result['components'].append({
                            'name': comp['name'],
                            'type': comp['type'],
                            'x': 0,
                            'y': 0,
                            'width': comp['width'],
                            'height': comp['height']
                        })
            else:
                # 如果没有匹配的模式，使用默认布局
                for comp in comps:
                    layout_result['components'].append({
                        'name': comp['name'],
                        'type': comp['type'],
                        'x': 0,
                        'y': 0,
                        'width': comp['width'],
                        'height': comp['height']
                    })
        
        # 3. 应用增强知识
        if enhanced_knowledge:
            for comp in layout_result['components']:
                if comp['type'] in enhanced_knowledge:
                    knowledge = enhanced_knowledge[comp['type']]
                    if 'position' in knowledge:
                        comp['x'] = knowledge['position']['x']
                        comp['y'] = knowledge['position']['y']
                    if 'size' in knowledge:
                        comp['width'] = knowledge['size']['width']
                        comp['height'] = knowledge['size']['height']
        
        # 4. 生成网络路由
        for net in layout_result['nets']:
            # 查找匹配的网络模式
            net_type = self._get_net_type(net)
            matching_patterns = [
                p for p in layout_patterns
                if net_type in p['net_patterns']
            ]
            
            if matching_patterns:
                # 选择得分最高的模式
                best_pattern = max(matching_patterns, key=lambda p: p['score'])
                pattern_routes = best_pattern['net_patterns'][net_type]
                
                # 应用路由模式
                if pattern_routes:
                    net['route'] = pattern_routes[0]['route']
                else:
                    net['route'] = self._generate_default_route(net, layout_result['components'])
            else:
                net['route'] = self._generate_default_route(net, layout_result['components'])
        
        return layout_result
    
    def _generate_default_route(self, net: Dict, components: List[Dict]) -> List[List[float]]:
        """生成默认路由
        
        Args:
            net: 网络信息
            components: 组件列表
            
        Returns:
            路由点列表
        """
        # 找到连接的组件
        connected_components = []
        for pin in net['pins']:
            comp = next((c for c in components if c['name'] == pin['component']), None)
            if comp:
                connected_components.append(comp)
        
        if len(connected_components) < 2:
            return []
        
        # 计算组件中心点
        points = []
        for comp in connected_components:
            center_x = comp['x'] + comp['width'] / 2
            center_y = comp['y'] + comp['height'] / 2
            points.append([center_x, center_y])
        
        # 生成简单的直线路由
        route = []
        for i in range(len(points) - 1):
            route.append(points[i])
            # 添加中间点以避免交叉
            mid_x = (points[i][0] + points[i+1][0]) / 2
            mid_y = (points[i][1] + points[i+1][1]) / 2
            route.append([mid_x, mid_y])
        route.append(points[-1])
        
        return route

    def retrieve_knowledge(self, hierarchy: Dict) -> Dict:
        """检索知识
        
        Args:
            hierarchy: 层次化分解结果
            
        Returns:
            Dict: 检索结果
        """
        try:
            # 1. 从知识库检索相似案例
            # 创建默认约束
            default_constraints = {
                'area': {'min': 0, 'max': 1000000},
                'power': {'min': 0, 'max': 1000},
                'timing': {'min': 0, 'max': 100}
            }
            
            similar_cases = self.knowledge_base.retrieve(hierarchy, default_constraints)
            
            # 2. 提取约束和优化指南
            constraints = self._extract_constraints(similar_cases)
            optimization_guidelines = self._extract_optimization_guidelines(similar_cases)
            
            # 3. 使用LLM增强知识
            enhanced_knowledge = self.llm_manager.enhance_knowledge(
                similar_cases,
                constraints,
                optimization_guidelines
            )
            
            return {
                'similar_cases': similar_cases,
                'constraints': constraints,
                'optimization_guidelines': optimization_guidelines,
                'enhanced_knowledge': enhanced_knowledge
            }
            
        except Exception as e:
            logger.error(f"知识检索失败: {str(e)}")
            return {}
            
    def retrieve_and_enhance(self, query: str, hierarchy: Dict) -> Dict:
        """检索并增强知识
        
        Args:
            query: 查询文本
            hierarchy: 层次化分解结果
            
        Returns:
            Dict: 增强后的知识
        """
        try:
            # 1. 检索知识
            retrieval_results = self.retrieve_knowledge(hierarchy)
            
            # 2. 使用LLM增强知识
            enhanced_knowledge = self.llm_manager.enhance_knowledge(
                retrieval_results['similar_cases'],
                retrieval_results['constraints'],
                retrieval_results['optimization_guidelines']
            )
            
            # 3. 计算拥塞度
            congestion = self._calculate_congestion(enhanced_knowledge)
            
            return {
                'retrieval_results': retrieval_results,
                'enhanced_knowledge': enhanced_knowledge,
                'congestion': congestion
            }
            
        except Exception as e:
            logger.error(f"知识检索和增强失败: {str(e)}")
            return {}
        
    def _extract_constraints(self, similar_cases: List[Dict]) -> Dict:
        """提取约束
        
        Args:
            similar_cases: 相似案例列表
            
        Returns:
            Dict: 约束信息
        """
        try:
            constraints = {
                'area': [],
                'power': [],
                'timing': []
            }
            
            for case in similar_cases:
                if 'constraints' in case:
                    case_constraints = case['constraints']
                    for key in constraints:
                        if key in case_constraints:
                            constraints[key].append(case_constraints[key])
                            
            return constraints
        
        except Exception as e:
            logger.error(f"提取约束失败: {str(e)}")
            return {}
            
    def _extract_optimization_guidelines(self, similar_cases: List[Dict]) -> List[str]:
        """提取优化指南
        
        Args:
            similar_cases: 相似案例列表
            
        Returns:
            List[str]: 优化指南列表
        """
        try:
            guidelines = []
            
            for case in similar_cases:
                if 'optimization_guidelines' in case:
                    guidelines.extend(case['optimization_guidelines'])
                    
            return list(set(guidelines))  # 去重
            
        except Exception as e:
            logger.error(f"提取优化指南失败: {str(e)}")
            return []
            
    def _calculate_congestion(self, knowledge: Dict) -> float:
        """计算拥塞度
        
        Args:
            knowledge: 知识数据
            
        Returns:
            float: 拥塞度
        """
        try:
            # 1. 获取布局信息
            layout = knowledge.get('layout', {})
            if not layout:
                return 0.0
                
            # 2. 创建网格
            grid_size = 10  # 10x10网格
            grid = np.zeros((grid_size, grid_size))
            
            # 3. 统计每个网格中的组件数量
            placement = layout.get('placement', [])
            for component in placement:
                x, y = component.get('position', [0, 0])  # 组件位置
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_x, grid_y] += 1
                    
            # 4. 计算拥塞度
            max_congestion = np.max(grid)
            avg_congestion = np.mean(grid)
            
            return max_congestion / (avg_congestion + 1e-6)
            
        except Exception as e:
            logger.error(f"计算拥塞度失败: {str(e)}")
            return 0.0

    def add_pdf_knowledge(self, pdf_path: str) -> bool:
        """添加PDF知识到知识库
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            bool: 是否成功添加
        """
        # 检查是否已处理
        pdf_hash = self._get_file_hash(pdf_path)
        if pdf_hash in self.pdf_cache:
            logger.info(f"PDF {pdf_path} 已处理，跳过")
            return True
            
        try:
            # 处理PDF
            pdf_data = self._process_pdf(pdf_path)
            
            # 添加到知识库
            self.knowledge_base.add_case(
                layout=pdf_data['layout'],
                optimization_result=pdf_data['optimization_result'],
                metadata={
                    'source': 'pdf',
                    'filename': Path(pdf_path).name,
                    'type': 'layout_design',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # 更新缓存
            self.pdf_cache[pdf_hash] = {
                'path': pdf_path,
                'processed_time': datetime.now().isoformat()
            }
            
            # 保存缓存
            self._save_pdf_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"处理PDF {pdf_path} 失败: {str(e)}")
            return False
            
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    def _save_pdf_cache(self):
        """保存PDF处理缓存"""
        cache_file = Path(self.config.get('cache_dir', 'data/cache')) / 'pdf_cache.json'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(self.pdf_cache, f, indent=2)
            
    def _load_pdf_cache(self):
        """加载PDF处理缓存"""
        cache_file = Path(self.config.get('cache_dir', 'data/cache')) / 'pdf_cache.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.pdf_cache = json.load(f)

    def _process_pdf(self, pdf_path: str) -> Dict:
        """处理PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Dict: 处理后的数据，包含layout和optimization_result
        """
        try:
            # 1. 提取文本内容
            text_content = self._extract_text_from_pdf(pdf_path)
            
            # 2. 生成文本embedding
            text_embedding = self.embedding_manager.embed_text(text_content)
            
            # 3. 构建知识结构
            return {
                'layout': {
                    'components': [],  # 添加空的components列表
                    'nets': [],        # 添加空的nets列表
                    'text': text_content,
                    'embeddings': {
                        'text': text_embedding
                    },
                    'metadata': {
                        'source': 'pdf',
                        'filename': Path(pdf_path).name,
                        'type': 'layout_design',
                        'timestamp': datetime.now().isoformat()
                    }
                },
                'optimization_result': {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"处理PDF失败: {str(e)}")
            raise
            
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF提取文本内容"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"提取PDF文本失败: {str(e)}")
            raise

    def evaluate_layout(self, layout: Dict, constraints: Dict) -> Dict:
        """评估布局质量
        
        Args:
            layout: 布局信息
            constraints: 约束条件
            
        Returns:
            详细的评估结果字典
        """
        try:
            # 使用多目标评估器评估布局
            evaluation_result = self.evaluator.evaluate_layout(layout)
            
            # 检查约束满足情况
            constraint_evaluator = ConstraintSatisfactionEvaluator({})
            constraint_score = constraint_evaluator.evaluate(layout)
            
            # 构建详细的评估结果
            quality_result = {
                'wirelength': layout.get('wirelength', 0.75),
                'congestion': layout.get('congestion', 0.7),
                'timing': layout.get('timing', 0.8),
                'power': layout.get('power', 0.75),
                'area': layout.get('area_utilization', 0.8),
                'overall': evaluation_result.get('overall', {}).get('score', 0.5),
                'metrics': {
                    'wirelength': layout.get('wirelength', 0.75),
                    'congestion': layout.get('congestion', 0.7),
                    'timing': layout.get('timing', 0.8),
                    'power': layout.get('power', 0.75),
                    'area': layout.get('area_utilization', 0.8)
                },
                'constraint_satisfaction': {
                    'timing': constraint_score,
                    'power': constraint_score,
                    'area': constraint_score
                }
            }
            
            self.logger.info(f"布局评估完成 - 总体评分: {quality_result['overall']:.3f}, 约束评分: {constraint_score:.3f}")
            
            return quality_result
            
        except Exception as e:
            self.logger.error(f"布局评估失败: {str(e)}")
            # 返回默认评估结果
            return {
                'wirelength': 0.5,
                'congestion': 0.5,
                'timing': 0.5,
                'power': 0.5,
                'area': 0.5,
                'overall': 0.5,
                'metrics': {
                    'wirelength': 0.5,
                    'congestion': 0.5,
                    'timing': 0.5,
                    'power': 0.5,
                    'area': 0.5
                },
                'constraint_satisfaction': {
                    'timing': 0.5,
                    'power': 0.5,
                    'area': 0.5
                }
            }

class HierarchicalKnowledgeBase:
    """层次化知识库类，支持多粒度检索"""
    
    def __init__(self, storage_dir: str = "data/knowledge_base"):
        """初始化层次化知识库
        
        Args:
            storage_dir: 数据存储目录
        """
        self.storage_dir = storage_dir
        self.levels = {
            'global': [],  # 全局布局特征
            'module': [],  # 模块级特征
            'connection': [],  # 连接级特征
            'constraint': []  # 约束级特征
        }
        self.case_metadata = {}  # 案例元数据
        self.knowledge_transfer = {}  # 知识图谱
        
        # 特征向量缓存
        self._feature_cache = {}
        self._similarity_cache = {}
        
        # 创建存储目录
        os.makedirs(storage_dir, exist_ok=True)
        
        # 加载已有数据
        self._load_data()
        
    def _load_data(self):
        """加载知识库数据"""
        try:
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
                
            # 加载案例数据
            cases_file = os.path.join(self.storage_dir, "cases.pkl")
            if os.path.exists(cases_file):
                try:
                    with open(cases_file, 'rb') as f:
                        self.cases = pickle.load(f)
                except Exception as e:
                    logger.error(f"加载案例数据失败: {str(e)}")
                    self.cases = []
                    
            # 加载知识图谱
            graph_file = os.path.join(self.storage_dir, "knowledge_graph.pkl")
            if os.path.exists(graph_file):
                try:
                    with open(graph_file, 'rb') as f:
                        self.knowledge_graph = pickle.load(f)
                except Exception as e:
                    logger.error(f"加载知识图谱失败: {str(e)}")
                    self.knowledge_graph = defaultdict(list)
                    
        except Exception as e:
            logger.error(f"加载知识库数据失败: {str(e)}")
            self.cases = []
            self.knowledge_graph = defaultdict(list)
        
    def _save_data(self):
        """保存知识库数据"""
        # 保存特征数据
        for level in self.levels:
            level_file = os.path.join(self.storage_dir, f"{level}_features.pkl")
            with open(level_file, 'wb') as f:
                pickle.dump(self.levels[level], f)
                
        # 保存元数据
        metadata_file = os.path.join(self.storage_dir, "case_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.case_metadata, f, indent=2)
            
        # 保存知识图谱
        graph_file = os.path.join(self.storage_dir, "knowledge_transfer.pkl")
        with open(graph_file, 'wb') as f:
            pickle.dump(self.knowledge_transfer, f)
            
    def add_case(self, 
                layout: Dict,
                optimization_result: Dict,
                metadata: Optional[Dict] = None):
        """添加新案例到知识库
        
        Args:
            layout: 布局数据
            optimization_result: 优化结果
            metadata: 额外的元数据
        """
        case_id = len(self.case_metadata)
        
        # 提取多粒度特征
        features = {
            'global': self._extract_global_features(layout),
            'module': self._extract_module_features(layout),
            'connection': self._extract_connection_features(layout),
            'constraint': self._extract_constraint_features(layout, optimization_result)
        }
        
        # 存储特征
        for level, level_features in features.items():
            self.levels[level].append({
                'case_id': case_id,
                'features': level_features
            })
            
        # 存储元数据
        self.case_metadata[case_id] = {
            'layout': layout,
            'optimization_result': optimization_result,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        # 更新知识图谱
        self._update_knowledge_transfer(case_id, features, layout, optimization_result)
        
        # 保存数据
        self._save_data()
        
    def export_case(self, case_id: int, export_dir: str) -> str:
        """导出案例数据
        
        Args:
            case_id: 案例ID
            export_dir: 导出目录
            
        Returns:
            导出文件路径
        """
        if case_id not in self.case_metadata:
            raise ValueError(f"Case {case_id} not found")
            
        # 创建导出目录
        os.makedirs(export_dir, exist_ok=True)
        
        # 准备导出数据
        export_data = {
            'case_id': case_id,
            'metadata': self.case_metadata[case_id],
            'features': {
                level: next(
                    (case['features'] for case in self.levels[level] 
                     if case['case_id'] == case_id),
                    None
                )
                for level in self.levels
            }
        }
        
        # 导出到文件
        export_file = os.path.join(export_dir, f"case_{case_id}.json")
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        return export_file
        
    def import_case(self, import_file: str) -> int:
        """导入案例数据
        
        Args:
            import_file: 导入文件路径
            
        Returns:
            导入的案例ID
        """
        # 读取导入数据
        with open(import_file, 'r') as f:
            import_data = json.load(f)
            
        # 验证数据格式
        required_fields = ['metadata', 'features']
        if not all(field in import_data for field in required_fields):
            raise ValueError("Invalid import data format")
            
        # 添加案例
        case_id = len(self.case_metadata)
        layout = import_data['metadata']['layout']
        optimization_result = import_data['metadata']['optimization_result']
        metadata = import_data['metadata'].get('metadata', {})
        
        # 存储特征
        for level, features in import_data['features'].items():
            if level in self.levels:
                self.levels[level].append({
                    'case_id': case_id,
                    'features': features
                })
                
        # 存储元数据
        self.case_metadata[case_id] = {
            'layout': layout,
            'optimization_result': optimization_result,
            'metadata': metadata,
            'imported_at': datetime.now().isoformat()
        }
        
        # 更新知识图谱
        self._update_knowledge_transfer(
            case_id,
            import_data['features'],
            layout,
            optimization_result
        )
        
        # 保存数据
        self._save_data()
        
        return case_id
        
    def backup(self, backup_dir: str) -> str:
        """备份知识库数据
        
        Args:
            backup_dir: 备份目录
            
        Returns:
            备份文件路径
        """
        # 创建备份目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)
        
        # 复制所有数据文件
        for level in self.levels:
            level_file = os.path.join(self.storage_dir, f"{level}_features.pkl")
            if os.path.exists(level_file):
                with open(level_file, 'rb') as src, \
                     open(os.path.join(backup_path, f"{level}_features.pkl"), 'wb') as dst:
                    dst.write(src.read())
                    
        metadata_file = os.path.join(self.storage_dir, "case_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as src, \
                 open(os.path.join(backup_path, "case_metadata.json"), 'w') as dst:
                dst.write(src.read())
                
        graph_file = os.path.join(self.storage_dir, "knowledge_transfer.pkl")
        if os.path.exists(graph_file):
            with open(graph_file, 'rb') as src, \
                 open(os.path.join(backup_path, "knowledge_transfer.pkl"), 'wb') as dst:
                dst.write(src.read())
                
        return backup_path
        
    def restore(self, backup_dir: str):
        """从备份恢复知识库数据
        
        Args:
            backup_dir: 备份目录
        """
        # 验证备份文件
        required_files = [
            "global_features.pkl",
            "module_features.pkl",
            "connection_features.pkl",
            "constraint_features.pkl",
            "case_metadata.json",
            "knowledge_transfer.pkl"
        ]
        
        if not all(
            os.path.exists(os.path.join(backup_dir, f))
            for f in required_files
        ):
            raise ValueError("Invalid backup directory")
            
        # 恢复数据文件
        for level in self.levels:
            level_file = os.path.join(backup_dir, f"{level}_features.pkl")
            with open(level_file, 'rb') as f:
                self.levels[level] = pickle.load(f)
                
        metadata_file = os.path.join(backup_dir, "case_metadata.json")
        with open(metadata_file, 'r') as f:
            self.case_metadata = json.load(f)
            
        graph_file = os.path.join(backup_dir, "knowledge_transfer.pkl")
        with open(graph_file, 'rb') as f:
            self.knowledge_transfer = pickle.load(f)
            
        # 保存到当前存储目录
        self._save_data()
        
    @lru_cache(maxsize=1000)
    def _extract_global_features(self, layout: Dict) -> np.ndarray:
        """提取全局布局特征（带缓存）
        
        Args:
            layout: 布局数据
            
        Returns:
            全局特征向量
        """
        return np.array([
            layout['density'],
            layout['congestion'],
            layout['power'],
            layout['current'],
            layout['area_utilization'],
            layout['wirelength']
        ])
        
    @lru_cache(maxsize=1000)
    def _extract_module_features(self, layout: Dict) -> List[np.ndarray]:
        """提取模块级特征（带缓存）
        
        Args:
            layout: 布局数据
            
        Returns:
            模块特征列表
        """
        module_features = []
        for module in layout['modules']:
            features = np.array([
                module['width'],
                module['height'],
                module['x'],
                module['y'],
                module['power'],
                module['area']
            ])
            module_features.append(features)
        return module_features
        
    @lru_cache(maxsize=1000)
    def _extract_connection_features(self, layout: Dict) -> List[np.ndarray]:
        """提取连接级特征（带缓存）
        
        Args:
            layout: 布局数据
            
        Returns:
            连接特征列表
        """
        connection_features = []
        for conn in layout['connections']:
            # 计算连接长度
            route = conn['route']
            length = 0
            for i in range(len(route) - 1):
                dx = route[i+1][0] - route[i][0]
                dy = route[i+1][1] - route[i][1]
                length += np.sqrt(dx*dx + dy*dy)
                
            features = np.array([
                length,
                conn['weight'],
                len(route),
                conn['width']
            ])
            connection_features.append(features)
        return connection_features
        
    def _extract_constraint_features(self, 
                                   layout: Dict,
                                   optimization_result: Dict) -> np.ndarray:
        """提取约束级特征
        
        Args:
            layout: 布局数据
            optimization_result: 优化结果
            
        Returns:
            约束特征向量
        """
        return np.array([
            optimization_result['satisfaction'],
            optimization_result['area_utilization'],
            optimization_result['routing_quality'],
            optimization_result['timing_performance']
        ])
        
    def _update_knowledge_transfer(self,
                              case_id: int,
                              features: Dict[str, Union[np.ndarray, List[np.ndarray]]],
                              layout: Dict,
                              optimization_result: Dict):
        """更新知识图谱
        
        Args:
            case_id: 案例ID
            features: 多粒度特征
            layout: 布局数据
            optimization_result: 优化结果
        """
        # 添加案例节点
        self.knowledge_transfer[case_id] = {
            'type': 'case',
            'features': features,
            'layout': layout,
            'optimization_result': optimization_result,
            'connections': set()
        }
        
        # 添加模块节点和连接
        for module in layout['modules']:
            module_id = f"module_{module['id']}"
            if module_id not in self.knowledge_transfer:
                self.knowledge_transfer[module_id] = {
                    'type': 'module',
                    'features': self._extract_module_features([module])[0],
                    'cases': set()
                }
            self.knowledge_transfer[module_id]['cases'].add(case_id)
            self.knowledge_transfer[case_id]['connections'].add(module_id)
            
        # 添加连接节点和连接
        for conn in layout['connections']:
            conn_id = f"connection_{conn['id']}"
            if conn_id not in self.knowledge_transfer:
                self.knowledge_transfer[conn_id] = {
                    'type': 'connection',
                    'features': self._extract_connection_features([conn])[0],
                    'cases': set()
                }
            self.knowledge_transfer[conn_id]['cases'].add(case_id)
            self.knowledge_transfer[case_id]['connections'].add(conn_id)
            
    def get_similar_cases(self,
                         layout: Dict,
                         top_k: int = 5,
                         level: str = 'global') -> List[Dict]:
        """获取相似案例（优化版）
        
        Args:
            layout: 布局数据
            top_k: 返回的相似案例数量
            level: 检索粒度（global/module/connection/constraint）
            
        Returns:
            相似案例列表
        """
        if level not in self.levels:
            raise ValueError(f"Invalid level: {level}")
            
        # 提取查询特征
        if level == 'global':
            query_features = self._extract_global_features(layout)
        elif level == 'module':
            query_features = self._extract_module_features(layout)
        elif level == 'connection':
            query_features = self._extract_connection_features(layout)
        else:  # constraint
            query_features = self._extract_constraint_features(layout, {})
            
        # 使用向量化操作计算相似度
        if level == 'global' or level == 'constraint':
            # 将所有特征向量堆叠成矩阵
            features_matrix = np.array([
                case['features'] for case in self.levels[level]
            ])
            
            # 计算余弦相似度
            similarities = np.dot(features_matrix, query_features) / (
                np.linalg.norm(features_matrix, axis=1) *
                np.linalg.norm(query_features)
            )
            
            # 获取最相似的案例
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [
                {
                    'case_id': self.levels[level][i]['case_id'],
                    'similarity': float(similarities[i]),
                    'data': self.case_metadata[self.levels[level][i]['case_id']]
                }
                for i in top_indices
            ]
        else:  # module or connection
            # 使用KD树进行最近邻搜索
            from scipy.spatial import cKDTree
            
            # 将所有特征向量展平并堆叠
            features_list = []
            case_ids = []
            for case in self.levels[level]:
                features_list.extend(case['features'])
                case_ids.extend([case['case_id']] * len(case['features']))
                
            features_matrix = np.array(features_list)
            
            # 构建KD树
            tree = cKDTree(features_matrix)
            
            # 查询最近邻
            distances, indices = tree.query(
                np.array(query_features),
                k=min(top_k, len(features_matrix))
            )
            
            # 统计每个案例的相似度
            case_similarities = defaultdict(list)
            for dist, idx in zip(distances.flatten(), indices.flatten()):
                case_id = case_ids[idx]
                similarity = 1.0 / (1.0 + dist)  # 将距离转换为相似度
                case_similarities[case_id].append(similarity)
                
            # 计算每个案例的平均相似度
            case_avg_similarities = {
                case_id: np.mean(similarities)
                for case_id, similarities in case_similarities.items()
            }
            
            # 获取最相似的案例
            top_cases = sorted(
                case_avg_similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            return [
                {
                    'case_id': case_id,
                    'similarity': float(similarity),
                    'data': self.case_metadata[case_id]
                }
                for case_id, similarity in top_cases
            ]
        
    def _build_knowledge_transfer(self):
        """构建知识图谱（优化版）"""
        # 使用稀疏矩阵存储图结构
        n_cases = len(self.case_metadata)
        adjacency_matrix = csr_matrix((n_cases, n_cases))
        
        # 构建特征矩阵
        feature_matrix = np.zeros((n_cases, 6))  # 6个全局特征
        for case_id, case in self.case_metadata.items():
            feature_matrix[case_id] = self._extract_global_features(case['layout'])
            
        # 计算特征相似度矩阵
        similarity_matrix = np.dot(feature_matrix, feature_matrix.T)
        norms = np.linalg.norm(feature_matrix, axis=1)
        similarity_matrix /= np.outer(norms, norms)
        
        # 设置阈值，只保留相似度高的边
        threshold = 0.8
        similarity_matrix[similarity_matrix < threshold] = 0
        
        # 转换为稀疏矩阵
        adjacency_matrix = csr_matrix(similarity_matrix)
        
        # 构建最小生成树，确保图的连通性
        mst = minimum_spanning_tree(-adjacency_matrix)
        adjacency_matrix = -mst.toarray()
        
        # 更新知识图谱
        self.knowledge_transfer = {
            'adjacency_matrix': adjacency_matrix,
            'feature_matrix': feature_matrix
        }
        
    def get_related_cases(self,
                         case_id: int,
                         max_depth: int = 2) -> List[Dict]:
        """获取相关案例（优化版）
        
        Args:
            case_id: 案例ID
            max_depth: 最大搜索深度
            
        Returns:
            相关案例列表
        """
        if not self.knowledge_transfer:
            self._build_knowledge_transfer()
            
        # 使用稀疏矩阵的BFS
        visited = set()
        queue = [(case_id, 0)]  # (case_id, depth)
        related_cases = []
        
        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            if current_id != case_id:
                related_cases.append({
                    'case_id': current_id,
                    'depth': depth,
                    'data': self.case_metadata[current_id]
                })
                
            # 获取相邻节点
            neighbors = self.knowledge_transfer['adjacency_matrix'][current_id].nonzero()[1]
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
                    
        return related_cases
        
    def get_case_clusters(self,
                         n_clusters: int = 5) -> List[List[Dict]]:
        """获取案例聚类（优化版）
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            聚类结果列表
        """
        if not self.knowledge_transfer:
            self._build_knowledge_transfer()
            
        # 使用特征矩阵进行聚类
        from sklearn.cluster import KMeans
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.knowledge_transfer['feature_matrix'])
        
        # 整理聚类结果
        cluster_results = [[] for _ in range(n_clusters)]
        for case_id, cluster_id in enumerate(clusters):
            cluster_results[cluster_id].append({
                'case_id': case_id,
                'data': self.case_metadata[case_id]
            })
            
        return cluster_results
        
    def get_case_statistics(self) -> Dict:
        """获取案例统计信息（优化版）
        
        Returns:
            统计信息字典
        """
        if not self.knowledge_transfer:
            self._build_knowledge_transfer()
            
        # 计算特征统计信息
        feature_matrix = self.knowledge_transfer['feature_matrix']
        feature_stats = {
            'mean': np.mean(feature_matrix, axis=0),
            'std': np.std(feature_matrix, axis=0),
            'min': np.min(feature_matrix, axis=0),
            'max': np.max(feature_matrix, axis=0)
        }
        
        # 计算图统计信息
        adjacency_matrix = self.knowledge_transfer['adjacency_matrix']
        graph_stats = {
            'num_edges': adjacency_matrix.nnz,
            'density': adjacency_matrix.nnz / (adjacency_matrix.shape[0] * adjacency_matrix.shape[1]),
            'avg_degree': np.mean(adjacency_matrix.sum(axis=1)),
            'max_degree': np.max(adjacency_matrix.sum(axis=1))
        }
        
        return {
            'feature_stats': feature_stats,
            'graph_stats': graph_stats,
            'num_cases': len(self.case_metadata)
        }

class MultiObjectiveEvaluator:
    """多目标评估器，用于评估布局质量"""
    
    def __init__(self):
        """初始化多目标评估器"""
        self.objectives = {
            'area': {
                'evaluate': self._evaluate_area,
                'weight': 0.3,
                'threshold': 0.8
            },
            'timing': {
                'evaluate': self._evaluate_timing,
                'weight': 0.3,
                'threshold': 0.8
            },
            'power': {
                'evaluate': self._evaluate_power,
                'weight': 0.2,
                'threshold': 0.8
            },
            'congestion': {
                'evaluate': self._evaluate_congestion,
                'weight': 0.2,
                'threshold': 0.8
            }
        }
        self.overall_threshold = 0.8
        
    def evaluate_layout(self, layout: Dict) -> Dict:
        """评估布局质量
        
        Args:
            layout: 布局数据
            
        Returns:
            评估结果
        """
        results = {}
        
        # 评估各个目标
        for obj_name, obj_config in self.objectives.items():
            score = obj_config['evaluate'](layout)
            results[obj_name] = {
                'score': score,
                'weight': obj_config['weight'],
                'threshold': obj_config['threshold'],
                'passed': score >= obj_config['threshold']
            }
            
        # 计算总体评分
        overall_score = sum(
            result['score'] * result['weight']
            for result in results.values()
        )
        results['overall'] = {
            'score': overall_score,
            'passed': overall_score >= self.overall_threshold
        }
        
        # 添加密度评估
        results['density'] = {
            'score': self._evaluate_density(layout),
            'weight': 0.2,
            'threshold': 0.8,
            'passed': True
        }
        
        # 添加时序裕度评估
        results['timing_margin'] = {
            'score': self._evaluate_timing_margin(layout),
            'weight': 0.2,
            'threshold': 0.8,
            'passed': True
        }
        
        return results
        
    def _evaluate_area(self, layout: Dict) -> float:
        """评估面积利用率
        
        Args:
            layout: 布局数据
            
        Returns:
            面积评分（0-1）
        """
        # 获取芯片总面积，添加默认值处理
        die_area = layout.get('die_area', {'width': 1000, 'height': 1000})
        if isinstance(die_area, list) and len(die_area) == 4:
            # DEF格式：[x1, y1, x2, y2]
            width = die_area[2] - die_area[0]
            height = die_area[3] - die_area[1]
            total_area = width * height
        else:
            # 字典格式：{'width': w, 'height': h}
            total_area = die_area.get('width', 1000) * die_area.get('height', 1000)
        
        # 计算已使用面积
        components = layout.get('components', [])
        used_area = sum(
            comp.get('width', 0) * comp.get('height', 0)
            for comp in components
        )
        
        # 计算面积利用率
        if total_area > 0:
            return used_area / total_area
        else:
            return 0.5  # 默认值
        
    def _evaluate_timing(self, layout: Dict) -> float:
        """评估时序性能
        
        Args:
            layout: 布局数据
            
        Returns:
            时序评分（0-1）
        """
        # 计算关键路径延迟
        max_delay = 0.0
        for net in layout.get('nets', []):
            # 计算连接延迟
            route = net.get('route', [])
            delay = 0.0
            for i in range(len(route) - 1):
                dx = route[i+1][0] - route[i][0]
                dy = route[i+1][1] - route[i][1]
                length = np.sqrt(dx*dx + dy*dy)
                delay += length * net.get('weight', 1.0)  # 默认权重为1.0
            max_delay = max(max_delay, delay)
            
        # 归一化延迟
        return 1.0 / (1.0 + max_delay)
        
    def _evaluate_power(self, layout: Dict) -> float:
        """评估功耗
        
        Args:
            layout: 布局数据
            
        Returns:
            功耗评分（0-1）
        """
        # 安全获取组件列表，如果不存在则返回默认评分
        components = layout.get('components', [])
        if not components:
            # 如果没有组件信息，返回默认评分
            return 0.75
            
        # 计算总功耗
        total_power = sum(
            comp.get('power', 1.0)  # 没有power字段时默认1.0
            for comp in components
        )
        
        # 归一化功耗
        return 1.0 / (1.0 + total_power)
        
    def _evaluate_congestion(self, layout: Dict) -> float:
        """评估拥塞度
        
        Args:
            layout: 布局数据
            
        Returns:
            拥塞度评分（0-1）
        """
        # 计算网络拥塞度
        total_congestion = 0.0
        for net in layout.get('nets', []):
            # 计算网络拥塞度
            route = net.get('route', [])
            congestion = 0.0
            for i in range(len(route) - 1):
                dx = route[i+1][0] - route[i][0]
                dy = route[i+1][1] - route[i][1]
                length = np.sqrt(dx*dx + dy*dy)
                congestion += length * net.get('weight', 1.0)  # 默认权重为1.0
            total_congestion += congestion
            
        # 归一化拥塞度
        return 1.0 / (1.0 + total_congestion)
        
    def get_objective_weights(self) -> Dict[str, float]:
        """获取目标权重
        
        Returns:
            目标权重字典
        """
        return {
            obj_name: obj_config['weight']
            for obj_name, obj_config in self.objectives.items()
        }
        
    def set_objective_weights(self, weights: Dict[str, float]):
        """设置目标权重
        
        Args:
            weights: 目标权重字典
        """
        for obj_name, weight in weights.items():
            if obj_name in self.objectives:
                self.objectives[obj_name]['weight'] = weight
                
    def get_objective_thresholds(self) -> Dict[str, float]:
        """获取目标阈值
        
        Returns:
            目标阈值字典
        """
        return {
            obj_name: obj_config['threshold']
            for obj_name, obj_config in self.objectives.items()
        }
        
    def set_objective_thresholds(self, thresholds: Dict[str, float]):
        """设置目标阈值
        
        Args:
            thresholds: 目标阈值字典
        """
        for obj_name, threshold in thresholds.items():
            if obj_name in self.objectives:
                self.objectives[obj_name]['threshold'] = threshold
                
    def get_objective_scores(self, layout: Dict) -> Dict[str, float]:
        """获取各目标得分
        
        Args:
            layout: 布局数据
            
        Returns:
            目标得分字典
        """
        return {
            obj_name: obj_config['evaluate'](layout)
            for obj_name, obj_config in self.objectives.items()
        }
        
    def get_objective_status(self, layout: Dict) -> Dict[str, bool]:
        """获取各目标状态
        
        Args:
            layout: 布局数据
            
        Returns:
            目标状态字典
        """
        scores = self.get_objective_scores(layout)
        return {
            obj_name: scores[obj_name] >= self.objectives[obj_name]['threshold']
            for obj_name in self.objectives
        }
        
    def _evaluate_density(self, layout: Dict) -> float:
        """评估布局密度
        
        Args:
            layout: 布局数据
            
        Returns:
            密度评分（0-1）
        """
        # 计算已使用面积
        components = layout.get('components', [])
        used_area = sum(
            comp.get('width', 0) * comp.get('height', 0)
            for comp in components
        )
        
        # 获取芯片总面积，添加默认值处理
        die_area = layout.get('die_area', {'width': 1000, 'height': 1000})
        if isinstance(die_area, list) and len(die_area) == 4:
            # DEF格式：[x1, y1, x2, y2]
            width = die_area[2] - die_area[0]
            height = die_area[3] - die_area[1]
            total_area = width * height
        else:
            # 字典格式：{'width': w, 'height': h}
            total_area = die_area.get('width', 1000) * die_area.get('height', 1000)
            
        # 计算密度
        if total_area > 0:
            density = used_area / total_area
            # 归一化密度（假设理想密度为0.7）
            return 1.0 - abs(density - 0.7)
        else:
            return 0.5  # 默认值
        
    def _evaluate_timing_margin(self, layout: Dict) -> float:
        """评估时序裕度
        
        Args:
            layout: 布局数据
            
        Returns:
            时序裕度评分（0-1）
        """
        # 计算关键路径延迟
        max_delay = 0.0
        for net in layout.get('nets', []):
            # 计算连接延迟
            route = net.get('route', [])
            delay = 0.0
            for i in range(len(route) - 1):
                dx = route[i+1][0] - route[i][0]
                dy = route[i+1][1] - route[i][1]
                length = np.sqrt(dx*dx + dy*dy)
                delay += length * net.get('weight', 1.0)  # 默认权重为1.0
            max_delay = max(max_delay, delay)
            
        # 假设目标时钟周期为10
        target_period = 10.0
        
        # 计算时序裕度
        timing_margin = (target_period - max_delay) / target_period
        
        # 归一化时序裕度
        return max(0.0, min(1.0, timing_margin))

class InteractiveLayoutGenerator:
    """交互式布局生成器"""
    
    def __init__(self, knowledge_base: HierarchicalKnowledgeBase):
        self.kb = knowledge_base
        self.current_layout = None
        self.constraints = {}
        self.optimization_history = []
        
    def parse_user_input(self, user_input: str) -> Dict:
        """解析用户输入
        
        Args:
            user_input: 用户输入的自然语言描述
            
        Returns:
            解析后的意图字典
        """
        # 使用预训练模型解析用户意图
        intent = {
            'type': None,  # 布局类型
            'constraints': [],  # 约束条件
            'objectives': []  # 优化目标
        }
        
        # 解析布局类型
        if '高性能' in user_input or '低功耗' in user_input:
            intent['type'] = 'performance'
        elif '面积优化' in user_input or '紧凑' in user_input:
            intent['type'] = 'area'
        else:
            intent['type'] = 'balanced'
            
        # 解析约束条件
        if '时序约束' in user_input:
            intent['constraints'].append('timing')
        if '功耗约束' in user_input:
            intent['constraints'].append('power')
        if '面积约束' in user_input:
            intent['constraints'].append('area')
            
        # 解析优化目标
        if '提高性能' in user_input:
            intent['objectives'].append('performance')
        if '降低功耗' in user_input:
            intent['objectives'].append('power')
        if '减小面积' in user_input:
            intent['objectives'].append('area')
            
        return intent
        
    def generate_layout(self, 
                       design_info: Dict,
                       hierarchy: Dict,
                       similar_cases: List[Dict],
                       constraints: Optional[Dict] = None) -> Dict:
        """生成布局
        
        Args:
            design_info: 设计信息
            hierarchy: 层次结构
            similar_cases: 相似案例
            constraints: 约束条件
            
        Returns:
            生成的布局
        """
        # 初始化布局
        layout = {
            'components': [],
            'nets': [],
            'die_area': design_info.get('die_area'),
            'cell_library': design_info.get('cell_library'),
            'constraints': constraints or design_info.get('constraints', {}),
            'name': design_info.get('name', 'unnamed_layout')
        }
        
        # 从相似案例中提取布局模式
        patterns = self._extract_layout_patterns(similar_cases)
        
        # 根据层次结构生成组件布局
        for module in hierarchy.get('modules', []):
            # 查找匹配的布局模式
            pattern = self._find_matching_pattern(module, patterns)
            
            # 生成组件布局
            if pattern:
                components = self._apply_pattern(module, pattern)
            else:
                components = self._generate_default_layout(module)
                
            layout['components'].extend(components)
            
        # 生成网络连接
        layout['nets'] = self._generate_nets(design_info.get('nets', []), layout['components'])
        
        return layout
        
    def _extract_layout_patterns(self, similar_cases: List[Dict]) -> List[Dict]:
        """从相似案例中提取布局模式
        
        Args:
            similar_cases: 相似案例列表
            
        Returns:
            布局模式列表
        """
        patterns = []
        for case in similar_cases:
            layout = case['layout']
            # 提取组件类型分布
            component_types = defaultdict(list)
            for comp in layout['components']:
                component_types[comp['type']].append({
                    'x': comp['x'],
                    'y': comp['y'],
                    'width': comp['width'],
                    'height': comp['height']
                })
            
            # 提取网络连接模式
            net_patterns = defaultdict(list)
            for net in layout.get('nets', []):
                net_type = self._get_net_type(net)
                net_patterns[net_type].append({
                    'route': net.get('route', []),
                    'weight': net.get('weight', 1.0)
                })
            
            # 生成布局模式
            pattern = {
                'component_types': dict(component_types),
                'net_patterns': dict(net_patterns),
                'score': case.get('score', 0),
                'case_id': case['id']
            }
            patterns.append(pattern)
        
        return patterns
        
    def _find_matching_pattern(self, module: Dict, patterns: List[Dict]) -> Optional[Dict]:
        """查找匹配的布局模式
        
        Args:
            module: 模块信息
            patterns: 布局模式列表
            
        Returns:
            匹配的布局模式
        """
        for pattern in patterns:
            if self._is_pattern_match(module, pattern):
                return pattern
        return None
        
    def _is_pattern_match(self, module: Dict, pattern: Dict) -> bool:
        """检查模块是否匹配布局模式
        
        Args:
            module: 模块信息
            pattern: 布局模式
            
        Returns:
            是否匹配
        """
        # 检查组件类型
        if pattern.get('component_type') != module.get('type'):
            return False
            
        # 检查组件数量
        if len(pattern.get('components', [])) != len(module.get('components', [])):
            return False
            
        return True
        
    def _apply_pattern(self, module: Dict, pattern: Dict) -> List[Dict]:
        """应用布局模式
        
        Args:
            module: 模块信息
            pattern: 布局模式
            
        Returns:
            生成的组件列表
        """
        components = []
        for comp_info, pattern_comp in zip(module['components'], pattern['components']):
            comp = {
                'id': comp_info['id'],
                'type': comp_info['type'],
                'width': comp_info['width'],
                'height': comp_info['height'],
                'x': pattern_comp['x'],
                'y': pattern_comp['y']
            }
            components.append(comp)
        return components
        
    def _generate_default_layout(self, module: Dict) -> List[Dict]:
        """生成默认布局
        
        Args:
            module: 模块信息
            
        Returns:
            生成的组件列表
        """
        components = []
        x, y = 0, 0
        for comp_info in module['components']:
            comp = {
                'id': comp_info['id'],
                'type': comp_info['type'],
                'width': comp_info['width'],
                'height': comp_info['height'],
                'x': x,
                'y': y
            }
            components.append(comp)
            x += comp_info['width']
        return components
        
    def _generate_nets(self, net_info: List[Dict], components: List[Dict]) -> List[Dict]:
        """生成网络连接
        
        Args:
            net_info: 网络信息
            components: 组件列表
            
        Returns:
            生成的网络列表
        """
        nets = []
        for net in net_info:
            # 查找连接的组件
            connected_comps = [
                comp for comp in components
                if comp['id'] in net['connections']
            ]
            
            if len(connected_comps) >= 2:
                # 生成路由
                route = self._generate_route(connected_comps)
                nets.append({
                    'id': net['id'],
                    'connections': net['connections'],
                    'route': route
                })
                
        return nets
        
    def _generate_route(self, components: List[Dict]) -> List[Tuple[float, float]]:
        """生成路由路径
        
        Args:
            components: 连接的组件列表
            
        Returns:
            路由路径点列表
        """
        # 简化的路由生成：使用组件中心点之间的直线
        route = []
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            # 计算组件中心点
            x1 = comp1['x'] + comp1['width'] / 2
            y1 = comp1['y'] + comp1['height'] / 2
            x2 = comp2['x'] + comp2['width'] / 2
            y2 = comp2['y'] + comp2['height'] / 2
            
            route.extend([(x1, y1), (x2, y2)])
            
        return route

class SceneAdaptiveRetriever:
    """场景自适应检索器"""
    
    def __init__(self, knowledge_base: HierarchicalKnowledgeBase):
        self.kb = knowledge_base
        self.scene_context = {}
        self.retrieval_history = []
        
    def analyze_scene(self, scene_info: Dict) -> Dict:
        """分析场景特征
        
        Args:
            scene_info: 场景信息字典
            
        Returns:
            场景特征字典
        """
        features = {
            'type': scene_info.get('type', 'general'),
            'constraints': scene_info.get('constraints', []),
            'objectives': scene_info.get('objectives', []),
            'complexity': self._calculate_complexity(scene_info),
            'priority': scene_info.get('priority', 0.5)
        }
        
        # 更新场景上下文
        self.scene_context = features
        
        return features
        
    def _calculate_complexity(self, scene_info: Dict) -> float:
        """计算场景复杂度"""
        complexity = 0.0
        
        # 根据约束数量计算复杂度
        complexity += len(scene_info.get('constraints', [])) * 0.1
        
        # 根据目标数量计算复杂度
        complexity += len(scene_info.get('objectives', [])) * 0.1
        
        # 根据模块数量计算复杂度
        if 'layout' in scene_info:
            complexity += len(scene_info['layout'].get('modules', [])) * 0.05
            
        return min(complexity, 1.0)
        
    def adjust_search_range(self, scene_features: Dict) -> Dict:
        """调整检索范围
        
        Args:
            scene_features: 场景特征字典
            
        Returns:
            调整后的检索参数
        """
        search_params = {
            'top_k': 5,
            'similarity_threshold': 0.7,
            'level': 'global',
            'weight_factors': {
                'timing': 0.33,
                'power': 0.33,
                'area': 0.34
            }
        }
        
        # 根据场景类型调整参数
        if scene_features['type'] == 'performance':
            search_params['weight_factors']['timing'] = 0.5
            search_params['weight_factors']['power'] = 0.25
            search_params['weight_factors']['area'] = 0.25
        elif scene_features['type'] == 'power':
            search_params['weight_factors']['timing'] = 0.25
            search_params['weight_factors']['power'] = 0.5
            search_params['weight_factors']['area'] = 0.25
        elif scene_features['type'] == 'area':
            search_params['weight_factors']['timing'] = 0.25
            search_params['weight_factors']['power'] = 0.25
            search_params['weight_factors']['area'] = 0.5
            
        # 根据复杂度调整参数
        if scene_features['complexity'] > 0.7:
            search_params['top_k'] = 10
            search_params['similarity_threshold'] = 0.6
        elif scene_features['complexity'] < 0.3:
            search_params['top_k'] = 3
            search_params['similarity_threshold'] = 0.8
            
        # 根据优先级调整参数
        if scene_features['priority'] > 0.7:
            search_params['similarity_threshold'] *= 0.9
            
        return search_params
        
    def retrieve(self, query: Dict, scene_info: Dict) -> List[Dict]:
        """执行场景自适应检索
        
        Args:
            query: 查询信息
            scene_info: 场景信息
            
        Returns:
            检索结果列表
        """
        # 分析场景
        scene_features = self.analyze_scene(scene_info)
        
        # 调整检索参数
        search_params = self.adjust_search_range(scene_features)
        
        # 执行检索
        results = self.kb.get_similar_cases(
            query,
            top_k=search_params['top_k'],
            level=search_params['level']
        )
        
        # 根据权重过滤结果
        filtered_results = self._filter_results_by_weights(
            results,
            search_params['weight_factors'],
            search_params['similarity_threshold']
        )
        
        # 记录检索历史
        self.retrieval_history.append({
            'scene': scene_features,
            'params': search_params,
            'results': filtered_results
        })
        
        return filtered_results
        
    def _filter_results_by_weights(self,
                                 results: List[Dict],
                                 weights: Dict,
                                 threshold: float) -> List[Dict]:
        """根据权重过滤结果"""
        filtered = []
        for result in results:
            # 计算加权相似度
            weighted_similarity = 0.0
            for metric, weight in weights.items():
                if metric in result['data']['optimization_result']:
                    score = result['data']['optimization_result'][metric]
                    weighted_similarity += score * weight
                    
            # 应用阈值过滤
            if weighted_similarity >= threshold:
                result['weighted_similarity'] = weighted_similarity
                filtered.append(result)
                
        return sorted(filtered, key=lambda x: x['weighted_similarity'], reverse=True)

class IterativeOptimizer:
    """迭代优化器"""
    
    def __init__(self,
                 knowledge_base: HierarchicalKnowledgeBase,
                 scene_retriever: SceneAdaptiveRetriever):
        self.kb = knowledge_base
        self.scene_retriever = scene_retriever
        self.optimization_history = []
        self.current_iteration = 0
        
    def optimize(self,
                initial_layout: Dict,
                scene_info: Dict,
                max_iterations: int = 100,
                convergence_threshold: float = 0.01) -> Dict:
        """执行迭代优化
        
        Args:
            initial_layout: 初始布局
            scene_info: 场景信息
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            
        Returns:
            优化后的布局
        """
        current_layout = initial_layout.copy()
        best_layout = initial_layout.copy()
        best_score = self._evaluate_layout(best_layout, scene_info)
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration
            
            # 检索相似案例
            similar_cases = self.scene_retriever.retrieve(current_layout, scene_info)
            
            # 生成候选布局
            candidate_layouts = self._generate_candidates(current_layout, similar_cases)
            
            # 评估候选布局
            best_candidate = None
            best_candidate_score = float('-inf')
            
            for candidate in candidate_layouts:
                score = self._evaluate_layout(candidate, scene_info)
                if score > best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = score
                    
            # 更新当前布局
            if best_candidate_score > best_score:
                best_layout = best_candidate
                best_score = best_candidate_score
                current_layout = best_candidate
            else:
                # 局部搜索
                current_layout = self._local_search(current_layout)
                
            # 记录优化历史
            self.optimization_history.append({
                'iteration': iteration,
                'layout': current_layout,
                'score': best_score
            })
            
            # 检查收敛
            if iteration > 0:
                improvement = (best_score - self.optimization_history[-2]['score'])
                if abs(improvement) < convergence_threshold:
                    break
                    
        return best_layout
        
    def _evaluate_layout(self, layout: Dict, scene_info: Dict) -> float:
        """评估布局质量"""
        # 创建评估器
        evaluator = MultiObjectiveEvaluator({})
        
        # 设置评估权重
        weights = {
            'timing': 0.33,
            'power': 0.33,
            'area': 0.34
        }
        
        # 根据场景调整权重
        if scene_info['type'] == 'performance':
            weights['timing'] = 0.5
            weights['power'] = 0.25
            weights['area'] = 0.25
        elif scene_info['type'] == 'power':
            weights['timing'] = 0.25
            weights['power'] = 0.5
            weights['area'] = 0.25
        elif scene_info['type'] == 'area':
            weights['timing'] = 0.25
            weights['power'] = 0.25
            weights['area'] = 0.5
            
        evaluator.set_objective_weights(weights)
        
        # 执行评估
        score = evaluator.evaluate_layout(layout)['overall']
        
        return score
        
    def _generate_candidates(self,
                           current_layout: Dict,
                           similar_cases: List[Dict]) -> List[Dict]:
        """生成候选布局"""
        candidates = []
        
        # 基于相似案例生成候选
        for case in similar_cases:
            candidate = self._create_candidate_from_case(current_layout, case)
            candidates.append(candidate)
            
        # 添加随机变异
        for _ in range(3):
            candidate = self._create_random_candidate(current_layout)
            candidates.append(candidate)
            
        return candidates
        
    def _create_candidate_from_case(self,
                                  current_layout: Dict,
                                  case: Dict) -> Dict:
        """基于案例创建候选布局"""
        candidate = current_layout.copy()
        
        # 从案例中提取有用的特征
        case_layout = case['data']['layout']
        
        # 更新模块位置
        for i, module in enumerate(candidate['modules']):
            if i < len(case_layout['modules']):
                case_module = case_layout['modules'][i]
                module['x'] = case_module['x']
                module['y'] = case_module['y']
                
        # 更新连接路由
        for i, conn in enumerate(candidate['connections']):
            if i < len(case_layout['connections']):
                case_conn = case_layout['connections'][i]
                conn['route'] = case_conn['route']
                
        return candidate
        
    def _create_random_candidate(self, current_layout: Dict) -> Dict:
        """创建随机候选布局"""
        candidate = current_layout.copy()
        
        # 随机调整模块位置
        for module in candidate['modules']:
            module['x'] += random.uniform(-5, 5)
            module['y'] += random.uniform(-5, 5)
            
        # 随机调整连接路由
        for conn in candidate['connections']:
            for point in conn['route']:
                point[0] += random.uniform(-2, 2)
                point[1] += random.uniform(-2, 2)
                
        return candidate
        
    def _local_search(self, layout: Dict) -> Dict:
        """执行局部搜索"""
        best_layout = layout.copy()
        best_score = self._evaluate_layout(best_layout, {})
        
        # 尝试多个局部移动
        for _ in range(10):
            # 随机选择一个模块
            module_idx = random.randint(0, len(layout['modules']) - 1)
            module = layout['modules'][module_idx]
            
            # 尝试不同的移动方向
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                # 创建新布局
                new_layout = layout.copy()
                new_layout['modules'][module_idx]['x'] += dx
                new_layout['modules'][module_idx]['y'] += dy
                
                # 评估新布局
                new_score = self._evaluate_layout(new_layout, {})
                
                # 更新最佳布局
                if new_score > best_score:
                    best_layout = new_layout
                    best_score = new_score
                    
        return best_layout 

class LightweightDeployment:
    """轻量级部署方案"""
    
    def __init__(self, knowledge_base: HierarchicalKnowledgeBase):
        self.kb = knowledge_base
        self.model_cache = {}
        self.feature_cache = {}
        self.config = {
            'enable_cache': True,
            'cache_size': 1000,
            'batch_size': 32,
            'use_quantization': True,
            'use_pruning': True,
            'pruning_ratio': 0.3
        }
        
    def optimize_model(self, model: Any) -> Any:
        """优化模型
        
        Args:
            model: 原始模型
            
        Returns:
            优化后的模型
        """
        if self.config['use_quantization']:
            model = self._quantize_model(model)
            
        if self.config['use_pruning']:
            model = self._prune_model(model)
            
        return model
        
    def _quantize_model(self, model: Any) -> Any:
        """量化模型"""
        # 这里使用简单的量化方法
        # 实际应用中可以使用更复杂的量化方法
        quantized_model = model.copy()
        
        # 量化权重
        for layer in quantized_model.layers:
            if hasattr(layer, 'weights'):
                layer.weights = np.round(layer.weights * 255) / 255
                
        return quantized_model
        
    def _prune_model(self, model: Any) -> Any:
        """剪枝模型"""
        pruned_model = model.copy()
        
        # 计算权重重要性
        importance_scores = self._calculate_weight_importance(pruned_model)
        
        # 根据重要性分数剪枝
        threshold = np.percentile(importance_scores, 
                                (1 - self.config['pruning_ratio']) * 100)
        
        for layer in pruned_model.layers:
            if hasattr(layer, 'weights'):
                mask = importance_scores > threshold
                layer.weights *= mask
                
        return pruned_model
        
    def _calculate_weight_importance(self, model: Any) -> np.ndarray:
        """计算权重重要性"""
        # 使用简单的L1范数作为重要性指标
        importance = []
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                importance.append(np.abs(layer.weights))
        return np.concatenate(importance)
        
    def cache_features(self, features: Dict, key: str):
        """缓存特征
        
        Args:
            features: 特征字典
            key: 缓存键
        """
        if not self.config['enable_cache']:
            return
            
        # 检查缓存大小
        if len(self.feature_cache) >= self.config['cache_size']:
            # 移除最旧的缓存
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
            
        # 添加新缓存
        self.feature_cache[key] = features
        
    def get_cached_features(self, key: str) -> Optional[Dict]:
        """获取缓存的特征
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的特征或None
        """
        if not self.config['enable_cache']:
            return None
            
        return self.feature_cache.get(key)
        
    def batch_process(self, items: List[Dict]) -> List[Dict]:
        """批量处理
        
        Args:
            items: 待处理项列表
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(items), self.config['batch_size']):
            batch = items[i:i + self.config['batch_size']]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
        return results
        
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """处理单个批次
        
        Args:
            batch: 批次数据
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 并行处理批次中的每个项
        for item in batch:
            # 检查缓存
            cache_key = self._generate_cache_key(item)
            cached_result = self.get_cached_features(cache_key)
            
            if cached_result is not None:
                results.append(cached_result)
                continue
                
            # 处理新项
            result = self._process_item(item)
            
            # 缓存结果
            self.cache_features(result, cache_key)
            results.append(result)
            
        return results
        
    def _process_item(self, item: Dict) -> Dict:
        """处理单个项
        
        Args:
            item: 待处理项
            
        Returns:
            处理结果
        """
        # 这里实现具体的处理逻辑
        result = item.copy()
        
        # 优化模型
        if 'model' in result:
            result['model'] = self.optimize_model(result['model'])
            
        return result
        
    def _generate_cache_key(self, item: Dict) -> str:
        """生成缓存键
        
        Args:
            item: 待处理项
            
        Returns:
            缓存键
        """
        # 使用项的哈希值作为缓存键
        return hashlib.md5(str(item).encode()).hexdigest()
        
    def update_config(self, new_config: Dict):
        """更新配置
        
        Args:
            new_config: 新配置字典
        """
        self.config.update(new_config)
        
        # 如果禁用了缓存，清空缓存
        if not self.config['enable_cache']:
            self.feature_cache.clear()
            self.model_cache.clear()
            
        # 如果缓存大小改变，调整缓存
        if len(self.feature_cache) > self.config['cache_size']:
            # 移除多余的缓存
            while len(self.feature_cache) > self.config['cache_size']:
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
                
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况
        
        Returns:
            内存使用情况字典
        """
        return {
            'feature_cache_size': len(self.feature_cache),
            'model_cache_size': len(self.model_cache),
            'total_memory': sys.getsizeof(self.feature_cache) + 
                          sys.getsizeof(self.model_cache)
        }
        
    def clear_cache(self):
        """清空缓存"""
        self.feature_cache.clear()
        self.model_cache.clear() 