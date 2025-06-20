from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from ..evaluation.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from .hierarchy import HierarchicalDecompositionManager
import psutil
import os
import gc
import logging
import time
from ..knowledge.knowledge_base import KnowledgeBase
from ..utils.llm_manager import LLMManager
import torch
import torch.nn as nn
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class LayoutConfig:
    min_component_size: float = 0.05
    max_component_size: float = 0.3
    min_spacing: float = 0.02
    max_density: float = 0.8
    num_grid_cells: int = 20
    num_routing_layers: int = 3
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    target_satisfaction: float = 95.0
    die_width: float = 20000.0  # 扩大10倍
    die_height: float = 20000.0  # 扩大10倍
    die_area: Dict[str, float] = field(default_factory=lambda: {
        'width': 20000.0,  # 扩大10倍
        'height': 20000.0  # 扩大10倍
    })

class LayoutGenerator(nn.Module):
    """布局生成器类，负责生成满足约束的布局"""
    
    def __init__(self, config: Dict):
        """初始化布局生成器
        
        Args:
            config: 配置信息
        """
        super().__init__()
        self.config = config
        self.layout_config = LayoutConfig(**config.get('layout_config', {}))
        
        # 初始化LLM管理器
        self.llm_manager = config.get('llm_manager')
        
        # 初始化网络层
        self._init_network()
        
    def _init_network(self):
        """初始化网络层"""
        # 1. 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 2. 布局生成层
        self.placement_generator = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # x, y, width, height
        )
        
        # 3. 布线生成层
        self.routing_generator = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # start_x, start_y, end_x, end_y
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            features: 输入特征
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 布局和布线结果
        """
        # 1. 特征提取
        x = self.feature_extractor(features)
        
        # 2. 生成布局
        placement = self.placement_generator(x)
        
        # 3. 生成布线
        routing = self.routing_generator(x)
        
        return placement, routing
        
    def generate(self, features: torch.Tensor) -> Dict[str, Any]:
        """生成布局
        
        Args:
            features: 输入特征
            
        Returns:
            Dict[str, Any]: 布局结果
        """
        try:
            # 1. 生成初始布局
            placement, routing = self(features)
            
            # 2. 应用约束
            placement = self._apply_constraints(placement)
            
            # 3. 优化布局
            placement = self._optimize_placement(placement)
            
            # 4. 生成布线
            routing = self._generate_routing(placement, routing)
            
            # 5. 验证结果
            self._verify_layout(placement, routing)
            
            return {
                'placement': placement.tolist(),
                'routing': routing.tolist(),
                'metrics': self._calculate_metrics(placement, routing)
            }
            
        except Exception as e:
            logger.error(f"布局生成失败: {str(e)}")
            raise
            
    def _apply_constraints(self, placement: torch.Tensor) -> torch.Tensor:
        """应用约束
        
        Args:
            placement: 布局结果
            
        Returns:
            torch.Tensor: 调整后的布局
        """
        # 1. 限制组件尺寸
        placement[:, 2:] = torch.clamp(
            placement[:, 2:],
            self.layout_config.min_component_size,
            self.layout_config.max_component_size
        )
        
        # 2. 限制组件位置
        placement[:, :2] = torch.clamp(placement[:, :2], -1.0, 1.0)
        
        return placement
        
    def _optimize_placement(self, placement: torch.Tensor) -> torch.Tensor:
        """优化布局
        
        Args:
            placement: 布局结果
            
        Returns:
            torch.Tensor: 优化后的布局
        """
        # 1. 检测重叠
        overlaps = self._detect_overlaps(placement)
        
        # 2. 调整重叠组件
        if overlaps.any():
            placement = self._resolve_overlaps(placement, overlaps)
            
        # 3. 控制密度
        density = self._calculate_density(placement)
        if density > self.layout_config.max_density:
            placement = self._adjust_density(placement)
            
        return placement
        
    def _detect_overlaps(self, placement: torch.Tensor) -> torch.Tensor:
        """检测组件重叠
        
        Args:
            placement: 布局结果
            
        Returns:
            torch.Tensor: 重叠矩阵
        """
        n = len(placement)
        overlaps = torch.zeros((n, n), dtype=torch.bool)
        
        for i in range(n):
            for j in range(i+1, n):
                if self._check_overlap(placement[i], placement[j]):
                    overlaps[i, j] = overlaps[j, i] = True
                    
        return overlaps
        
    def _check_overlap(self, comp1: torch.Tensor, comp2: torch.Tensor) -> bool:
        """检查两个组件是否重叠
        
        Args:
            comp1: 组件1
            comp2: 组件2
            
        Returns:
            bool: 是否重叠
        """
        x1 = max(comp1[0], comp2[0])
        y1 = max(comp1[1], comp2[1])
        x2 = min(comp1[0] + comp1[2], comp2[0] + comp2[2])
        y2 = min(comp1[1] + comp1[3], comp2[1] + comp2[3])
        
        return x2 > x1 and y2 > y1
        
    def _resolve_overlaps(self, placement: torch.Tensor,
                        overlaps: torch.Tensor) -> torch.Tensor:
        """解决组件重叠
        
        Args:
            placement: 布局结果
            overlaps: 重叠矩阵
            
        Returns:
            torch.Tensor: 调整后的布局
        """
        n = len(placement)
        for i in range(n):
            for j in range(i+1, n):
                if overlaps[i, j]:
                    # 计算移动方向
                    dx = placement[j, 0] - placement[i, 0]
                    dy = placement[j, 1] - placement[i, 1]
                    
                    # 移动组件
                    if abs(dx) > abs(dy):
                        placement[j, 0] += self.layout_config.min_spacing * (1 if dx > 0 else -1)
                    else:
                        placement[j, 1] += self.layout_config.min_spacing * (1 if dy > 0 else -1)
                        
        return placement
        
    def _calculate_density(self, placement: torch.Tensor) -> float:
        """计算布局密度
        
        Args:
            placement: 布局结果
            
        Returns:
            float: 布局密度
        """
        total_area = torch.sum(placement[:, 2] * placement[:, 3])
        layout_area = self.layout_config.die_width * self.layout_config.die_height
        return total_area.item() / layout_area
        
    def _adjust_density(self, placement: torch.Tensor) -> torch.Tensor:
        """调整布局密度
        
        Args:
            placement: 布局结果
            
        Returns:
            torch.Tensor: 调整后的布局
        """
        # 计算需要移动的距离
        move_distance = (self._calculate_density(placement) - self.layout_config.max_density) * 0.1
        
        # 移动组件
        for i in range(len(placement)):
            # 随机选择移动方向
            angle = np.random.random() * 2 * np.pi
            placement[i, 0] += move_distance * np.cos(angle)
            placement[i, 1] += move_distance * np.sin(angle)
            
        return placement
        
    def _generate_routing(self, placement: torch.Tensor,
                        initial_routing: torch.Tensor) -> torch.Tensor:
        """生成布线
        
        Args:
            placement: 布局结果
            initial_routing: 初始布线结果
            
        Returns:
            torch.Tensor: 布线结果
        """
        # 1. 获取连接关系
        connections = self._get_connections(placement)
        
        # 2. 生成布线
        routing = []
        for start, end in connections:
            # 获取起点和终点
            start_pos = placement[start, :2]
            end_pos = placement[end, :2]
            
            # 生成布线路径
            path = self._generate_path(start_pos, end_pos)
            routing.append(path)
            
        return torch.tensor(routing)
        
    def _get_connections(self, placement: torch.Tensor) -> List[Tuple[int, int]]:
        """获取组件连接关系
        
        Args:
            placement: 布局结果
            
        Returns:
            List[Tuple[int, int]]: 连接关系列表
        """
        # 这里需要实现获取连接关系的逻辑
        # 暂时返回空列表
        return []
        
    def _generate_path(self, start: torch.Tensor,
                     end: torch.Tensor) -> torch.Tensor:
        """生成布线路径
        
        Args:
            start: 起点
            end: 终点
            
        Returns:
            torch.Tensor: 路径
        """
        # 这里需要实现路径生成逻辑
        # 暂时返回直线路径
        return torch.tensor([start[0], start[1], end[0], end[1]])
        
    def _verify_layout(self, placement: torch.Tensor, routing: torch.Tensor):
        """验证布局
        
        Args:
            placement: 布局结果
            routing: 布线结果
        """
        # 1. 验证组件尺寸
        if torch.any(placement[:, 2:] < self.layout_config.min_component_size):
            raise ValueError("存在组件尺寸过小")
        if torch.any(placement[:, 2:] > self.layout_config.max_component_size):
            raise ValueError("存在组件尺寸过大")
            
        # 2. 验证组件位置
        if torch.any(placement[:, :2] < -1.0) or torch.any(placement[:, :2] > 1.0):
            raise ValueError("存在组件位置超出范围")
            
        # 3. 验证组件重叠
        overlaps = self._detect_overlaps(placement)
        if overlaps.any():
            raise ValueError("存在组件重叠")
            
        # 4. 验证布局密度
        density = self._calculate_density(placement)
        if density > self.layout_config.max_density:
            raise ValueError("布局密度过高")
            
    def _calculate_metrics(self, placement: torch.Tensor,
                         routing: torch.Tensor) -> Dict[str, float]:
        """计算布局指标
        
        Args:
            placement: 布局结果
            routing: 布线结果
            
        Returns:
            Dict[str, float]: 布局指标
        """
        return {
            'density': self._calculate_density(placement),
            'wirelength': self._calculate_wirelength(routing),
            'component_count': len(placement),
            'routing_count': len(routing)
        }
        
    def _calculate_wirelength(self, routing: torch.Tensor) -> float:
        """计算总布线长度
        
        Args:
            routing: 布线结果
            
        Returns:
            float: 总布线长度
        """
        if len(routing) == 0:
            return 0.0
            
        dx = routing[:, 2] - routing[:, 0]
        dy = routing[:, 3] - routing[:, 1]
        return torch.sum(torch.sqrt(dx * dx + dy * dy)).item()
        
    def optimize(self,
                layout: Dict[str, Any],
                suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """优化布局
        
        Args:
            layout: 当前布局
            suggestions: 优化建议
            
        Returns:
            优化后的布局
        """
        logger.info("开始优化布局...")
        
        # 1. 分析当前布局
        layout_analysis = self._analyze_layout(layout)
        
        # 2. 生成优化策略
        optimization_strategy = self._generate_optimization_strategy(
            layout_analysis=layout_analysis,
            suggestions=suggestions
        )
        
        # 3. 应用优化策略
        optimized_layout = self._apply_optimization(
            layout=layout,
            strategy=optimization_strategy
        )
        
        return optimized_layout
        
    def _analyze_design(self, design_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析设计信息
        
        Args:
            design_info: 设计信息
            
        Returns:
            设计分析结果
        """
        # 使用LLM分析设计信息
        analysis = self.llm_manager.analyze_design(design_info)
        return analysis
        
    def _generate_strategy(self,
                          design_analysis: Dict[str, Any],
                          knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """生成布局策略
        
        Args:
            design_analysis: 设计分析结果
            knowledge: 相关知识
            
        Returns:
            布局策略
        """
        # 使用LLM生成布局策略
        strategy = self.llm_manager.generate_layout_strategy(
            design_analysis=design_analysis,
            knowledge=knowledge
        )
        return strategy
        
    def _apply_strategy(self,
                       design_info: Dict[str, Any],
                       strategy: Dict[str, Any]) -> Dict[str, Any]:
        """应用布局策略
        
        Args:
            design_info: 设计信息
            strategy: 布局策略
            
        Returns:
            生成的布局
        """
        # 使用LLM应用布局策略
        layout = self.llm_manager.apply_layout_strategy(
            design_info=design_info,
            strategy=strategy
        )
        return layout
        
    def _analyze_layout(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """分析布局
        
        Args:
            layout: 当前布局
            
        Returns:
            布局分析结果
        """
        # 使用LLM分析布局
        analysis = self.llm_manager.analyze_layout(layout)
        return analysis
        
    def _generate_optimization_strategy(self,
                                      layout_analysis: Dict[str, Any],
                                      suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成优化策略
        
        Args:
            layout_analysis: 布局分析结果
            suggestions: 优化建议
            
        Returns:
            优化策略
        """
        # 使用LLM生成优化策略
        strategy = self.llm_manager.generate_optimization_strategy(
            layout_analysis=layout_analysis,
            suggestions=suggestions
        )
        return strategy
        
    def _apply_optimization(self,
                          layout: Dict[str, Any],
                          strategy: Dict[str, Any]) -> Dict[str, Any]:
        """应用优化策略
        
        Args:
            layout: 当前布局
            strategy: 优化策略
            
        Returns:
            优化后的布局
        """
        # 使用LLM应用优化策略
        optimized_layout = self.llm_manager.apply_optimization_strategy(
            layout=layout,
            strategy=strategy
        )
        return optimized_layout
        
    def _calculate_module_positions(self, modules: List[Dict], area: float) -> List[Dict]:
        """计算模块的初始位置
        
        Args:
            modules: 模块列表
            area: 可用面积
            
        Returns:
            带有位置的模块列表
        """
        # 计算总面积
        total_area = sum(m['width'] * m['height'] for m in modules)
        
        # 计算缩放因子
        scale = np.sqrt(area / total_area)
        
        # 计算网格大小
        grid_size = int(np.sqrt(len(modules)))
        
        # 计算网格间距
        spacing = np.sqrt(area) / (grid_size + 1)
        
        # 放置模块
        positioned_modules = []
        for i, module in enumerate(modules):
            row = i // grid_size
            col = i % grid_size
            
            x = (col + 1) * spacing
            y = (row + 1) * spacing
            
            positioned_modules.append({
                'id': module['id'],
                'x': x,
                'y': y,
                'width': module['width'] * scale,
                'height': module['height'] * scale
            })
            
        return positioned_modules
        
    def _generate_initial_routes(self, 
                               modules: List[Dict], 
                               connections: List[Dict]) -> List[Dict]:
        """生成初始布线
        
        Args:
            modules: 模块列表
            connections: 连接列表
            
        Returns:
            带有路由的连接列表
        """
        routed_connections = []
        
        for conn in connections:
            # 找到源模块和目标模块
            from_module = next(m for m in modules if m['id'] == conn['from'])
            to_module = next(m for m in modules if m['id'] == conn['to'])
            
            # 计算模块中心点
            from_center = (
                from_module['x'] + from_module['width'] / 2,
                from_module['y'] + from_module['height'] / 2
            )
            to_center = (
                to_module['x'] + to_module['width'] / 2,
                to_module['y'] + to_module['height'] / 2
            )
            
            # 生成简单的直线路由
            route = [from_center, to_center]
            
            routed_connections.append({
                'id': conn.get('id', f'c{len(routed_connections)}'),
                'from': conn['from'],
                'to': conn['to'],
                'weight': conn.get('weight', 1.0),
                'route': route
            })
            
        return routed_connections
        
    def _calculate_layout_metrics(self, 
                                modules: List[Dict],
                                connections: List[Dict]) -> Dict:
        """计算布局指标
        
        Args:
            modules: 模块列表
            connections: 连接列表
            
        Returns:
            布局指标字典
        """
        # 计算间距
        spacing = []
        for i, m1 in enumerate(modules):
            for m2 in modules[i+1:]:
                dx = abs(m1['x'] - m2['x'])
                dy = abs(m1['y'] - m2['y'])
                spacing.append(min(dx, dy))
                
        # 计算线宽（基于连接权重）
        width = [conn['weight'] * 0.1 for conn in connections]
        
        # 计算面积
        area = [m['width'] * m['height'] for m in modules]
        
        return {
            'spacing': spacing,
            'width': width,
            'area': area
        }
        
    def generate_layout(self, design_info: Dict,
                       hierarchy_info: Dict,
                       knowledge_base: Union[Dict, 'KnowledgeBase']) -> Dict:
        """生成布局
        
        Args:
            design_info: 设计信息
            hierarchy_info: 层次化信息
            knowledge_base: 知识库或知识字典
            
        Returns:
            布局结果
        """
        logger.info("开始生成布局")
        self._log_memory_usage("开始生成布局")
        
        # 初始化布局结果
        layout_result = {
            'name': design_info['name'],
            'components': [],
            'nets': design_info['nets'],
            'die_area': design_info['die_area'],
            'hierarchy': hierarchy_info
        }
        
        # 分批处理组件
        batch_size = 1000
        total_components = len(design_info['components'])
        logger.info(f"总共需要处理 {total_components} 个组件")
        
        for i in range(0, total_components, batch_size):
            batch_end = min(i + batch_size, total_components)
            logger.info(f"处理第 {i//batch_size + 1} 批组件 ({i+1}-{batch_end})")
            
            # 提取当前批次的组件
            batch_components = design_info['components'][i:batch_end]
            
            # 获取相关的网络
            batch_nets = self._get_related_nets(batch_components, design_info['nets'])
            
            # 使用RAG系统增强知识（如果有）
            enhanced_knowledge = None
            if hasattr(self, 'rag_system') and self.rag_system:
                enhanced_knowledge = self.rag_system.retrieve_and_enhance(
                    batch_components,
                    hierarchy_info
                )
            
            # 生成当前批次的布局
            batch_layout = self._generate_batch_layout(
                batch_components,
                batch_nets,
                hierarchy_info,
                enhanced_knowledge or knowledge_base
            )
            
            # 合并结果
            layout_result['components'].extend(batch_layout['components'])
            
            # 检查内存使用
            self._check_memory_and_cleanup()
            
            # 记录进度
            progress = (batch_end / total_components) * 100
            logger.info(f"处理进度: {progress:.1f}%")
            self._log_memory_usage(f"处理进度 {progress:.1f}%")
        
        # 使用LLM分析和优化布局
        if self.llm_manager:
            # 分析布局
            analysis = self.llm_manager.analyze_layout(layout_result)
            
            # 如果需要优化
            if analysis.get('needs_optimization', False):
                # 优化布局
                optimized_layout = self.llm_manager.optimize_layout(
                    layout_result,
                    analysis.get('suggestions', [])
                )
                layout_result = optimized_layout
        
        return layout_result
        
    def _generate_layout(self, enhanced_knowledge):
        """生成布局
        
        Args:
            enhanced_knowledge: 增强后的知识
            
        Returns:
            布局结果
        """
        # 使用增强后的知识生成布局
        layout = self._apply_layout_patterns(enhanced_knowledge)
        return layout
        
    def _optimize_layout(self, layout_result, optimization_suggestions):
        """优化布局
        
        Args:
            layout_result: 布局结果
            optimization_suggestions: 优化建议
            
        Returns:
            优化后的布局
        """
        # 根据优化建议调整布局
        for suggestion in optimization_suggestions:
            if suggestion['confidence'] > 0.7:  # 只应用高置信度的建议
                layout_result = self._apply_suggestion(
                    layout_result, 
                    suggestion
                )
        return layout_result
        
    def _apply_suggestion(self, layout, suggestion):
        """应用优化建议
        
        Args:
            layout: 当前布局
            suggestion: 优化建议
            
        Returns:
            调整后的布局
        """
        # 根据建议类型应用不同的优化策略
        if suggestion['metric'] == 'timing':
            layout = self._optimize_timing(layout, suggestion)
        elif suggestion['metric'] == 'area':
            layout = self._optimize_area(layout, suggestion)
        elif suggestion['metric'] == 'power':
            layout = self._optimize_power(layout, suggestion)
            
        return layout
        
    def generate_initial_layout(self, design_spec: Dict) -> Dict:
        """生成初始布局
        
        Args:
            design_spec: 设计规格，包含模块信息、连接关系等
            
        Returns:
            初始布局数据
        """
        # 1. 组件数量减为50
        components = design_spec.get('components', [])[:50]
        logger.info(f"输入组件数量: {len(components)}")
        nets = []  # 3. 取消布线
        
        # 组件尺寸缩小为原来的1/10
        SCALE = 0.1
        SPACING = 10  # 4. 组件间距设置为10
        
        # 计算所有组件面积总和
        total_area = 0.0
        comp_sizes = []
        for comp in components:
            comp_type = comp.get('type', 'L')
            if comp_type in design_spec.get('cell_library', {}):
                size = design_spec['cell_library'][comp_type]['SIZE']
                comp_width = size['width'] * SCALE
                comp_height = size['height'] * SCALE
            else:
                comp_width = 1.0 * SCALE
                comp_height = 1.0 * SCALE
            comp_sizes.append((comp_width, comp_height))
            total_area += comp_width * comp_height
        
        # 1. die面积为组件总面积的100倍，取正方形
        die_area_val = total_area * 100
        die_width = die_height = np.sqrt(die_area_val)
        die_area = {'width': die_width, 'height': die_height}
        width = die_width
        height = die_height
        area = die_area_val
        logger.info(f"die_width={die_width}, die_height={die_height}, total_area={total_area}")
        
        # 行优先紧凑排布
        comp_with_size = list(zip(components, comp_sizes))
        comp_with_size.sort(key=lambda x: x[1][0]*x[1][1], reverse=True)
        
        positioned_components = []
        current_x, current_y, row_max_height = 0, 0, 0
        for idx, (comp, (comp_width, comp_height)) in enumerate(comp_with_size):
            logger.info(f"排布组件{idx}: x={current_x}, y={current_y}, w={comp_width}, h={comp_height}, die_w={width}, die_h={height}")
            if current_x + comp_width > width:
                # 换行
                current_x = 0
                current_y += row_max_height + SPACING
                row_max_height = 0
            if current_y + comp_height > height:
                logger.warning(f"组件{idx}放不下，跳出排布循环")
                break
                positioned_components.append({
                'id': comp.get('id', f'comp_{idx}'),
                'type': comp.get('type', 'L'),
                'x': current_x,
                'y': current_y,
                    'width': comp_width,
                    'height': comp_height,
                'orientation': 'N'
            })
            current_x += comp_width + SPACING
            row_max_height = max(row_max_height, comp_height)
        
        # 7. 计算布局指标
        total_component_area = sum(
            comp['width'] * comp['height']
            for comp in positioned_components
        )
        
        density = total_component_area / area if area > 0 else 0
        
        # 8. 构建布局数据
        layout = {
            'components': positioned_components,
            'nets': [],  # 3. 取消布线
            'die_area': die_area,
            'cell_library': design_spec.get('cell_library', {}),
            'density': density,
            'congestion': 0.0,
            'power': 0.8,  # 初始估计值
            'current': 0.08  # 初始估计值
        }
        
        # 更新当前布局
        self.current_layout = layout
        
        return layout
        
    def _evaluate_layout(self, layout: Dict) -> float:
        """评估布局的约束满足率
        
        Args:
            layout: 布局数据
            
        Returns:
            约束满足率
        """
        results = self.constraint_evaluator.evaluate_constraint_satisfaction(layout)
        return results['overall_satisfaction']
        
    def _identify_violations(self, layout: Dict) -> List[Dict]:
        """识别布局中的约束违反
        
        Args:
            layout: 布局数据
            
        Returns:
            违反列表
        """
        violations = []
        
        # 检查间距约束
        for i, spacing in enumerate(layout['spacing']):
            if spacing < 0.1:  # 最小间距约束
                violations.append({
                    'type': 'spacing',
                    'index': i,
                    'value': spacing,
                    'threshold': 0.1
                })
                
        # 检查线宽约束
        for i, width in enumerate(layout['width']):
            if width < 0.05:  # 最小线宽约束
                violations.append({
                    'type': 'width',
                    'index': i,
                    'value': width,
                    'threshold': 0.05
                })
                
        # 检查面积约束
        for i, area in enumerate(layout['area']):
            if area < 0.1:  # 最小面积约束
                violations.append({
                    'type': 'area',
                    'index': i,
                    'value': area,
                    'threshold': 0.1
                })
                
        # 检查密度约束
        if layout['density'] > 0.8:  # 最大密度约束
            violations.append({
                'type': 'density',
                'value': layout['density'],
                'threshold': 0.8
            })
            
        # 检查拥塞约束
        if layout['congestion'] > 0.8:  # 最大拥塞约束
            violations.append({
                'type': 'congestion',
                'value': layout['congestion'],
                'threshold': 0.8
            })
            
        return violations
        
    def _apply_optimization_step(self, layout: Dict, violations: List[Dict]) -> Dict:
        """应用单步优化
        
        Args:
            layout: 当前布局
            violations: 违反列表
            
        Returns:
            优化后的布局
        """
        optimized_layout = layout.copy()
        
        for violation in violations:
            if violation['type'] == 'spacing':
                # 调整模块位置以增加间距
                idx = violation['index']
                module1 = optimized_layout['components'][idx]
                module2 = optimized_layout['components'][idx + 1]
                
                # 计算移动方向
                dx = module2['x'] - module1['x']
                dy = module2['y'] - module1['y']
                
                # 移动模块
                module1['x'] -= dx * 0.1
                module1['y'] -= dy * 0.1
                module2['x'] += dx * 0.1
                module2['y'] += dy * 0.1
                
            elif violation['type'] == 'width':
                # 增加线宽
                idx = violation['index']
                optimized_layout['width'][idx] *= 1.2
                
            elif violation['type'] == 'area':
                # 增加模块面积
                idx = violation['index']
                module = optimized_layout['components'][idx]
                scale = 1.2
                module['width'] *= scale
                module['height'] *= scale
                
            elif violation['type'] == 'density':
                # 减少模块面积
                scale = 0.9
                for module in optimized_layout['components']:
                    module['width'] *= scale
                    module['height'] *= scale
                    
            elif violation['type'] == 'congestion':
                # 优化布线
                for conn in optimized_layout['nets']:
                    if conn['weight'] > 1.0:
                        # 增加路由点以分散拥塞
                        route = conn['route']
                        if len(route) == 2:
                            mid_x = (route[0][0] + route[1][0]) / 2
                            mid_y = (route[0][1] + route[1][1]) / 2
                            conn['route'] = [route[0], (mid_x, mid_y), route[1]]
                            
        # 更新布局指标
        metrics = self._calculate_layout_metrics(
            optimized_layout['components'],
            optimized_layout['nets']
        )
        optimized_layout.update(metrics)
        
        return optimized_layout
        
    def optimize_layout(self, 
                       layout: Dict,
                       max_iterations: int = 100,
                       target_satisfaction: float = 95.0) -> Dict:
        """优化布局以满足约束
        
        Args:
            layout: 当前布局
            max_iterations: 最大迭代次数
            target_satisfaction: 目标满足率
            
        Returns:
            优化后的布局
        """
        current_layout = layout.copy()
        best_layout = layout.copy()
        best_satisfaction = self._evaluate_layout(layout)
        
        for iteration in range(max_iterations):
            # 评估当前布局
            current_satisfaction = self._evaluate_layout(current_layout)
            
            # 更新最佳布局
            if current_satisfaction > best_satisfaction:
                best_layout = current_layout.copy()
                best_satisfaction = current_satisfaction
                
            # 检查是否达到目标
            if current_satisfaction >= target_satisfaction:
                break
                
            # 识别约束违反
            violations = self._identify_violations(current_layout)
            
            # 如果没有违反，尝试进一步优化
            if not violations:
                # 尝试微调布局
                current_layout = self._apply_optimization_step(
                    current_layout,
                    [{'type': 'fine_tune'}]
                )
                continue
                
            # 应用优化步骤
            current_layout = self._apply_optimization_step(current_layout, violations)
            
        # 更新最佳布局
        self.best_layout = best_layout
        self.best_satisfaction = best_satisfaction
        
        return best_layout
        
    def _analyze_suggestion(self, suggestion: Dict, layout: Dict) -> bool:
        """分析建议的可行性
        
        Args:
            suggestion: 优化建议
            layout: 当前布局
            
        Returns:
            建议是否可行
        """
        if suggestion['type'] == 'module_position':
            # 检查模块是否存在
            module_id = suggestion['module_id']
            if not any(m['id'] == module_id for m in layout['components']):
                return False
                
            # 检查新位置是否在合理范围内
            x, y = suggestion['suggested_x'], suggestion['suggested_y']
            if x < 0 or y < 0:
                return False
                
        elif suggestion['type'] == 'connection_route':
            # 检查连接是否存在
            conn_id = suggestion['connection_id']
            if not any(c['id'] == conn_id for c in layout['nets']):
                return False
                
            # 检查路由点是否在合理范围内
            for point in suggestion['suggested_route']:
                if point[0] < 0 or point[1] < 0:
                    return False
                    
        return True
        
    def _apply_suggestion(self, suggestion: Dict, layout: Dict) -> Dict:
        """应用单个建议
        
        Args:
            suggestion: 优化建议
            layout: 当前布局
            
        Returns:
            更新后的布局
        """
        updated_layout = layout.copy()
        
        if suggestion['type'] == 'module_position':
            # 更新模块位置
            module_id = suggestion['module_id']
            for module in updated_layout['components']:
                if module['id'] == module_id:
                    module['x'] = suggestion['suggested_x']
                    module['y'] = suggestion['suggested_y']
                    break
                    
            # 更新相关连接的路由
            for conn in updated_layout['nets']:
                if conn['from'] == module_id or conn['to'] == module_id:
                    # 重新计算路由
                    from_module = next(m for m in updated_layout['components'] 
                                    if m['id'] == conn['from'])
                    to_module = next(m for m in updated_layout['components'] 
                                   if m['id'] == conn['to'])
                    
                    from_center = (
                        from_module['x'] + from_module['width'] / 2,
                        from_module['y'] + from_module['height'] / 2
                    )
                    to_center = (
                        to_module['x'] + to_module['width'] / 2,
                        to_module['y'] + to_module['height'] / 2
                    )
                    
                    conn['route'] = [from_center, to_center]
                    
        elif suggestion['type'] == 'connection_route':
            # 更新连接路由
            conn_id = suggestion['connection_id']
            for conn in updated_layout['nets']:
                if conn['id'] == conn_id:
                    conn['route'] = suggestion['suggested_route']
                    break
                    
        # 更新布局指标
        metrics = self._calculate_layout_metrics(
            updated_layout['components'],
            updated_layout['nets']
        )
        updated_layout.update(metrics)
        
        return updated_layout
        
    def _verify_improvement(self, 
                          original_layout: Dict,
                          updated_layout: Dict) -> bool:
        """验证布局改进
        
        Args:
            original_layout: 原始布局
            updated_layout: 更新后的布局
            
        Returns:
            是否有改进
        """
        original_satisfaction = self._evaluate_layout(original_layout)
        updated_satisfaction = self._evaluate_layout(updated_layout)
        
        return updated_satisfaction > original_satisfaction
        
    def apply_rag_guidance(self, 
                          layout: Dict,
                          rag_suggestions: List[Dict]) -> Dict:
        """应用RAG系统的建议来改进布局
        
        Args:
            layout: 当前布局
            rag_suggestions: RAG系统提供的建议列表
            
        Returns:
            改进后的布局
        """
        current_layout = layout.copy()
        
        for suggestion in rag_suggestions:
            # 分析建议可行性
            if not self._analyze_suggestion(suggestion, current_layout):
                continue
                
            # 应用建议
            updated_layout = self._apply_suggestion(suggestion, current_layout)
            
            # 验证改进
            if self._verify_improvement(current_layout, updated_layout):
                current_layout = updated_layout
                
        return current_layout
        
    def _calculate_area_utilization(self, layout: Dict) -> float:
        """计算面积利用率
        
        Args:
            layout: 布局数据
            
        Returns:
            面积利用率（0-100）
        """
        # 计算模块总面积
        module_area = sum(m['width'] * m['height'] for m in layout['components'])
        
        # 计算布局总面积（使用边界框）
        x_coords = [m['x'] + m['width'] for m in layout['components']]
        y_coords = [m['y'] + m['height'] for m in layout['components']]
        total_area = max(x_coords) * max(y_coords)
        
        # 计算利用率
        utilization = (module_area / total_area) * 100
        return min(utilization, 100.0)
        
    def _evaluate_routing_quality(self, layout: Dict) -> float:
        """评估布线质量
        
        Args:
            layout: 布局数据
            
        Returns:
            布线质量分数（0-100）
        """
        # 计算平均线长
        total_length = 0
        for conn in layout['nets']:
            route = conn['route']
            for i in range(len(route) - 1):
                dx = route[i+1][0] - route[i][0]
                dy = route[i+1][1] - route[i][1]
                total_length += np.sqrt(dx*dx + dy*dy)
                
        avg_length = total_length / len(layout['nets'])
        
        # 计算拥塞度
        congestion = layout['congestion']
        
        # 综合评分
        length_score = max(0, 100 - avg_length * 10)  # 线长越短越好
        congestion_score = max(0, 100 - congestion * 100)  # 拥塞度越低越好
        
        return (length_score + congestion_score) / 2
        
    def _analyze_timing_performance(self, layout: Dict) -> float:
        """分析时序性能
        
        Args:
            layout: 布局数据
            
        Returns:
            时序性能分数（0-100）
        """
        # 计算平均线长
        total_length = 0
        for conn in layout['nets']:
            route = conn['route']
            for i in range(len(route) - 1):
                dx = route[i+1][0] - route[i][0]
                dy = route[i+1][1] - route[i][1]
                total_length += np.sqrt(dx*dx + dy*dy)
                
        avg_length = total_length / len(layout['nets'])
        
        # 计算平均线宽
        avg_width = sum(layout['width']) / len(layout['width'])
        
        # 估算延迟
        estimated_delay = avg_length * (1 / avg_width)
        
        # 转换为分数
        delay_score = max(0, 100 - estimated_delay * 10)
        
        return delay_score
        
    def _evaluate_power_distribution(self, layout: Dict) -> float:
        """评估功耗分布
        
        Args:
            layout: 布局数据
            
        Returns:
            功耗分布分数（0-100）
        """
        # 计算功耗密度
        power_density = layout['power'] / layout['density']
        
        # 计算电流密度
        current_density = layout['current'] / layout['density']
        
        # 评估分布均匀性
        power_score = max(0, 100 - power_density * 100)
        current_score = max(0, 100 - current_density * 1000)
        
        return (power_score + current_score) / 2
        
    def evaluate_layout_quality(self, layout: Dict) -> Dict:
        """评估布局质量
        
        Args:
            layout: 布局数据
            
        Returns:
            包含各项质量指标的字典
        """
        # 计算各项指标
        area_utilization = self._calculate_area_utilization(layout)
        routing_quality = self._evaluate_routing_quality(layout)
        timing_performance = self._analyze_timing_performance(layout)
        power_distribution = self._evaluate_power_distribution(layout)
        
        # 计算总体质量分数
        overall_quality = (
            area_utilization * 0.3 +
            routing_quality * 0.3 +
            timing_performance * 0.2 +
            power_distribution * 0.2
        )
        
        return {
            'area_utilization': area_utilization,
            'routing_quality': routing_quality,
            'timing_performance': timing_performance,
            'power_distribution': power_distribution,
            'overall_quality': overall_quality
        }
        
    def _log_memory_usage(self, stage: str):
        """记录内存使用情况
        
        Args:
            stage: 当前阶段
        """
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info(f"内存使用 ({stage}): {memory_usage:.1f} MB")
        
    def _check_memory_and_cleanup(self):
        """检查内存使用并清理"""
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        if memory_usage > 2000:  # 2GB
            logger.info(f"内存使用超过阈值 ({memory_usage:.1f} MB > 2000 MB)，执行垃圾回收")
            gc.collect()
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            logger.info(f"垃圾回收后内存使用: {memory_usage:.1f} MB")
        
    def _get_related_nets(self, batch_components, all_nets):
        """获取与指定组件批次相关的网络"""
        batch_comp_names = {comp['name'] for comp in batch_components}
        return [net for net in all_nets if any(pin.get('component') in batch_comp_names for pin in net.get('pins', []))]
        
    def _generate_batch_layout(self, batch_components, batch_nets, hierarchy_info, enhanced_knowledge):
        """生成批次组件的布局
        
        Args:
            batch_components: 批次组件列表
            batch_nets: 相关网络列表
            hierarchy_info: 层次化信息
            enhanced_knowledge: 增强后的知识或知识库对象
            
        Returns:
            批次布局结果
        """
        # 1. 初始化布局结果
        batch_layout = {
            'components': [],
            'nets': batch_nets
        }
        
        # 2. 按组件类型分组
        component_groups = {}
        for comp in batch_components:
            comp_type = comp.get('type', 'L')
            if comp_type not in component_groups:
                component_groups[comp_type] = []
            component_groups[comp_type].append(comp)
        
        # 3. 计算布局网格
        total_components = len(batch_components)
        grid_size = int(np.sqrt(total_components * 1.5))  # 增加1.5倍空间以优化布局
        
        # 4. 计算每个网格单元的大小
        die_area = hierarchy_info.get('die_area', {'width': 1000, 'height': 1000})
        cell_width = die_area['width'] / grid_size
        cell_height = die_area['height'] / grid_size
        
        # 5. 为每种类型的组件分配区域
        current_x = 0
        current_y = 0
        max_height = 0
        
        for comp_type, comps in component_groups.items():
            # 计算这种类型组件需要的行数
            comps_per_row = int(cell_width / comps[0]['width'])
            if comps_per_row < 1:
                comps_per_row = 1
            
            rows_needed = int(np.ceil(len(comps) / comps_per_row))
            
            # 为每个组件分配位置
            for i, comp in enumerate(comps):
                row = i // comps_per_row
                col = i % comps_per_row
                
                x = current_x + col * (comp['width'] + cell_width * 0.1)  # 添加10%间距
                y = current_y + row * (comp['height'] + cell_height * 0.1)
                
                # 使用增强知识调整位置（如果有）
                if enhanced_knowledge:
                    # 如果是知识库对象，获取相关知识
                    if hasattr(enhanced_knowledge, 'retrieve'):
                        knowledge = enhanced_knowledge.retrieve(
                            query=comp,
                            constraints=hierarchy_info,
                            top_k=1
                        )
                        if knowledge:
                            # 使用知识调整位置
                            if 'position' in knowledge:
                                x = knowledge['position'].get('x', x)
                                y = knowledge['position'].get('y', y)
                
                # 添加组件到布局
                batch_layout['components'].append({
                    'name': comp['name'],
                    'type': comp['type'],
                    'x': x,
                    'y': y,
                    'width': comp['width'],
                    'height': comp['height']
                })
                
                # 更新最大高度
                max_height = max(max_height, y + comp['height'])
            
            # 更新当前Y坐标
            current_y = max_height + cell_height * 0.2  # 添加20%间距
        
        return batch_layout 