# modules/hierarchy.py

import json
import logging
import psutil
import os
from collections import defaultdict
from typing import Dict, List, Union, Any, Optional
import gc
from dataclasses import dataclass, field
from ..utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

@dataclass
class Node:
    """层次结构节点"""
    name: str
    type: str
    level: int = 0
    children: List['Node'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    parent: Optional['Node'] = None
    knowledge: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}
        if self.knowledge is None:
            self.knowledge = {}
    
    def add_child(self, child: 'Node'):
        """添加子节点
        
        Args:
            child: 子节点
        """
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)
        
    def remove_child(self, child: 'Node'):
        """移除子节点
        
        Args:
            child: 子节点
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            
    def get_path(self) -> str:
        """获取节点路径
        
        Returns:
            节点路径
        """
        if self.parent is None:
            return self.name
        return f"{self.parent.get_path()}/{self.name}"
        
    def find_child(self, name: str) -> Optional['Node']:
        """查找子节点
        
        Args:
            name: 节点名称
            
        Returns:
            找到的节点，如果不存在则返回None
        """
        for child in self.children:
            if child.name == name:
                return child
            result = child.find_child(name)
            if result is not None:
                return result
        return None
        
    def get_knowledge(self) -> Dict[str, Any]:
        """获取节点知识
        
        Returns:
            节点知识
        """
        return self.knowledge
        
    def add_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """添加节点知识
        
        Args:
            knowledge: 要添加的知识
        """
        self.knowledge.update(knowledge)

class Hierarchy:
    """层次结构管理类"""
    
    def __init__(self, llm_manager: LLMManager):
        """初始化层次结构管理器
        
        Args:
            llm_manager: LLM管理器实例
        """
        self.llm_manager = llm_manager
        self.root = None
        logger.info("层次结构管理器初始化完成")
        
    def build(self, design_info: Dict[str, Any]) -> None:
        """构建层次结构
        
        Args:
            design_info: 设计信息
        """
        logger.info("开始构建层次结构...")
        
        # 1. 分析设计信息
        hierarchy_info = self.llm_manager.analyze_hierarchy(design_info)
        
        # 2. 构建层次结构
        self.root = self._build_tree(hierarchy_info)
        
        logger.info("层次结构构建完成")
        
    def get_node(self, path: str) -> Optional[Node]:
        """获取指定路径的节点
        
        Args:
            path: 节点路径
            
        Returns:
            找到的节点，如果不存在则返回None
        """
        if self.root is None:
            return None
            
        if path == self.root.name:
            return self.root
            
        parts = path.split('/')
        current = self.root
        
        for part in parts[1:]:
            current = current.find_child(part)
            if current is None:
                return None
                
        return current
        
    def add_node(self, node):
        """添加节点到层次结构
        
        Args:
            node: 要添加的节点
        """
        if not hasattr(self, 'nodes'):
            self.nodes = {}
        
        if node.name in self.nodes:
            logger.warning(f"节点 {node.name} 已存在")
            return
        
        self.nodes[node.name] = node
        logger.info(f"添加节点: {node.name}")
        
    def remove_node(self, path: str) -> bool:
        """移除节点
        
        Args:
            path: 节点路径
            
        Returns:
            是否移除成功
        """
        node = self.get_node(path)
        if node is None or node.parent is None:
            return False
            
        node.parent.remove_child(node)
        return True
        
    def update_node(self, path: str, attributes: Dict[str, Any]) -> bool:
        """更新节点属性
        
        Args:
            path: 节点路径
            attributes: 要更新的属性
            
        Returns:
            是否更新成功
        """
        node = self.get_node(path)
        if node is None:
            return False
            
        node.attributes.update(attributes)
        return True
        
    def _build_tree(self, hierarchy_info: Dict[str, Any]) -> Node:
        """构建树结构
        
        Args:
            hierarchy_info: 层次结构信息
            
        Returns:
            根节点
        """
        def build_node(info: Dict[str, Any]) -> Node:
            node = Node(
                name=info['name'],
                type=info['type'],
                children=[],
                attributes=info.get('attributes', {})
            )
            
            for child_info in info.get('children', []):
                child = build_node(child_info)
                node.add_child(child)
                
            return node
            
        return build_node(hierarchy_info)

class HierarchicalDecomposer:
    def __init__(self):
        # 定义实例类型到功能模块的映射
        self.type_to_function = {
            # 逻辑门
            'in01f01': 'logic_gate',
            'in01f02': 'logic_gate',
            'in01f03': 'logic_gate',
            'in01f04': 'logic_gate',
            # 寄存器
            'sdfrtnq': 'register',
            'sdfrtpq': 'register',
            # 存储器
            'ram': 'memory',
            'rom': 'memory',
            # 运算单元
            'adder': 'arithmetic',
            'multiplier': 'arithmetic',
            'comparator': 'arithmetic',
            # 控制单元
            'mux': 'control',
            'demux': 'control',
            'decoder': 'control',
            'encoder': 'control'
        }
        
        # 定义功能模块到子模块的映射
        self.function_to_submodule = {
            'logic_gate': ['input_buffer', 'logic_core', 'output_buffer'],
            'register': ['clock_control', 'data_latch', 'output_buffer'],
            'memory': ['address_decoder', 'memory_cell', 'sense_amplifier'],
            'arithmetic': ['input_reg', 'compute_core', 'output_reg'],
            'control': ['input_buffer', 'control_logic', 'output_buffer']
        }

    def _log_memory_usage(self, stage: str):
        """记录内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"内存使用 ({stage}): {memory_info.rss / 1024 / 1024:.1f} MB")

    def decompose(self, netlist: Dict) -> Hierarchy:
        """
        基于实例类型和连接关系进行层次分解
        """
        self._log_memory_usage("开始层次分解")
        
        hierarchy = Hierarchy()
        
        # 1. 创建顶层模块节点
        top_module = netlist['modules'][0]
        top_node = Node(top_module['name'], 'system', [], top_module['attributes'])
        hierarchy.add_node(top_node)
        
        # 2. 按实例类型分组
        logger.info("开始按类型分组实例")
        type_groups = self._group_by_type(top_module['instances'])
        self._log_memory_usage("类型分组完成")
        
        # 3. 为每个类型组创建功能模块
        logger.info("开始创建功能模块")
        for type_name, instances in type_groups.items():
            # 获取功能类型
            function_type = self.type_to_function.get(type_name, 'unknown')
            
            # 创建功能模块节点
            func_node = Node(f"{function_type}_module", 'function', [], {'type': function_type})
            func_node.add_child(top_node)
            top_node.add_child(func_node)
            hierarchy.add_node(func_node)
            
            # 4. 创建子模块
            if function_type in self.function_to_submodule:
                for submodule_type in self.function_to_submodule[function_type]:
                    sub_node = Node(f"{function_type}_{submodule_type}", 'submodule', [], {'type': submodule_type})
                    func_node.add_child(sub_node)
                    hierarchy.add_node(sub_node)
                    
                    # 5. 将实例分配到子模块
                    self._assign_instances_to_submodule(sub_node, instances, submodule_type)
        
            # 定期记录内存使用
            self._log_memory_usage(f"处理类型 {type_name}")
        
        self._log_memory_usage("层次分解完成")
        return hierarchy
    
    def _group_by_type(self, instances: List[Dict]) -> Dict[str, List[Dict]]:
        """按实例类型分组"""
        groups = defaultdict(list)
        for instance in instances:
            type_name = instance['type']
            groups[type_name].append(instance)
        return dict(groups)
    
    def _assign_instances_to_submodule(self, submodule_node: Node, instances: List[Dict], submodule_type: str):
        """将实例分配到子模块"""
        # 根据子模块类型和实例的连接关系进行分配
        for instance in instances:
            # 创建单元节点
            cell_node = Node(instance['name'], 'cell', [], {
                'type': instance['type'],
                'connections': instance.get('connections', {})
            })
            cell_node.add_child(submodule_node)
            # 将节点添加到层次结构中
            hierarchy = submodule_node.parent
            if hasattr(hierarchy, 'nodes'):
                hierarchy.nodes[cell_node.name] = cell_node

class HierarchicalDecompositionManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}  # 确保config不为None
        self._validate_config()
        self._init_components()
        
    def _validate_config(self):
        """验证配置"""
        # 验证层次配置
        if 'levels' not in self.config:
            self.config['levels'] = [
                {'name': 'system', 'threshold': 0.8},
                {'name': 'module', 'threshold': 0.6},
                {'name': 'component', 'threshold': 0.4}
            ]
            
        # 验证LLM配置
        if 'llm_config' not in self.config:
            self.config['llm_config'] = {
                'base_url': 'http://localhost:11434',
                'model': 'llama2',
                'temperature': 0.7,
                'max_tokens': 1000
            }
                
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化层次结构
            self.levels = self.config['levels']
            
            # 初始化LLM管理器
            from modules.utils.llm_manager import LLMManager
            self.llm_manager = LLMManager(
                self.config['llm_config']
            )
            
            # 初始化其他组件
            self._init_hierarchy()
            self._init_metrics()
            
        except Exception as e:
            logging.error(f"初始化组件失败: {str(e)}")
            raise
            
    def _init_hierarchy(self):
        """初始化层次结构"""
        self.hierarchy = {}
        for level in self.levels:
            self.hierarchy[level['name']] = {
                'threshold': level['threshold'],
                'components': []
            }
            
    def _init_metrics(self):
        """初始化评估指标"""
        self.metrics = {
            'similarity': {},
            'coherence': {},
            'completeness': {}
        }
        
    def hierarchical_decomposition(self, design_info: Dict[str, Any]) -> Dict[str, Any]:
        """执行层次化分解
        
        Args:
            design_info: 设计信息
            
        Returns:
            层次化分解结果
        """
        try:
            # 1. 分析设计信息
            analysis = self.llm_manager.analyze_hierarchy(design_info)
            
            # 2. 构建层次结构
            hierarchy = {
                'levels': [],
                'modules': {},
                'connections': [],
                'patterns': []
            }
            
            # 3. 处理每个层次
            for level in self.levels:
                level_name = level['name']
                threshold = level['threshold']
                
                # 获取当前层次的组件
                components = self._get_components(level_name)
                
                # 计算相似度
                similarities = self._calculate_similarities(analysis, components)
                
                # 过滤结果
                filtered = self._filter_results(similarities, threshold)
                
                # 添加到层次结构
                hierarchy['levels'].append({
                    'name': level_name,
                    'components': filtered
                })
                
                # 更新模块信息
                for item in filtered:
                    module_name = item['component']['name']
                    hierarchy['modules'][module_name] = item['component']
                    
                    # 添加连接信息
                    if 'connections' in item['component']:
                        hierarchy['connections'].extend(item['component']['connections'])
                        
                    # 添加模式信息
                    if 'patterns' in item['component']:
                        hierarchy['patterns'].extend(item['component']['patterns'])
            
            # 4. 计算最大深度
            hierarchy['max_depth'] = len(hierarchy['levels'])
            
            return hierarchy
            
        except Exception as e:
            logging.error(f"层次化分解失败: {str(e)}")
            raise
        
    def decompose(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分解查询"""
        try:
            results = []
            for level in self.levels:
                # 获取当前层次的组件
                components = self._get_components(level['name'])
                
                # 计算相似度
                similarities = self._calculate_similarities(query, components)
                
                # 过滤结果
                filtered = self._filter_results(similarities, level['threshold'])
                
                # 添加到结果
                results.extend(filtered)
                
            return results
            
        except Exception as e:
            logging.error(f"分解失败: {str(e)}")
            raise
            
    def _get_components(self, level: str) -> List[Dict[str, Any]]:
        """获取指定层次的组件"""
        return self.hierarchy[level]['components']
        
    def _calculate_similarities(self, query: Dict[str, Any], 
                              components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算相似度"""
        similarities = []
        for component in components:
            # 计算相似度
            similarity = self._compute_similarity(query, component)
            similarities.append({
                'component': component,
                'similarity': similarity
            })
        return similarities
        
    def _filter_results(self, similarities: List[Dict[str, Any]], 
                       threshold: float) -> List[Dict[str, Any]]:
        """过滤结果"""
        return [item for item in similarities if item['similarity'] >= threshold]
        
    def _compute_similarity(self, query: Dict[str, Any], 
                          component: Dict[str, Any]) -> float:
        """计算相似度"""
        # 实现相似度计算逻辑
        return 0.0  # 临时返回值

# 测试代码
def example_netlist():
    return {
        'modules': [
            {
                'name': 'top',
                'functions': [
                    {
                        'name': 'ALU',
                        'submodules': [
                            {
                                'name': 'adder',
                                'cells': [{'name': 'adder_cell1'}, {'name': 'adder_cell2'}]
                            }
                        ]
                    }
                ]
            }
        ]
    }

def test_instance_grouping():
    """测试实例分组功能"""
    logger.info("开始测试实例分组")
    
    # 创建测试数据（字典格式）
    test_modules_dict = {
        'top': {
            'instances': {
                'inst1': {'type': 'in01f01', 'connections': {'in': 'a', 'out': 'b'}},
                'inst2': {'type': 'in01f01', 'connections': {'in': 'c', 'out': 'd'}},
                'inst3': {'type': 'sdfrtnq', 'connections': {'clk': 'clk', 'd': 'e', 'q': 'f'}},
                'inst4': {'type': 'sdfrtnq', 'connections': {'clk': 'clk', 'd': 'g', 'q': 'h'}}
            }
        },
        'sub': {
            'instances': {
                'inst5': {'type': 'in01f01', 'connections': {'in': 'i', 'out': 'j'}},
                'inst6': {'type': 'sdfrtnq', 'connections': {'clk': 'clk', 'd': 'k', 'q': 'l'}}
            }
        }
    }
    
    # 创建测试数据（列表格式）
    test_modules_list = [
        {
            'instances': {
                'inst1': {'type': 'in01f01', 'connections': {'in': 'a', 'out': 'b'}},
                'inst2': {'type': 'in01f01', 'connections': {'in': 'c', 'out': 'd'}}
            }
        },
        {
            'instances': {
                'inst3': {'type': 'sdfrtnq', 'connections': {'clk': 'clk', 'd': 'e', 'q': 'f'}},
                'inst4': {'type': 'sdfrtnq', 'connections': {'clk': 'clk', 'd': 'g', 'q': 'h'}}
            }
        }
    ]
    
    # 创建管理器实例
    manager = HierarchicalDecompositionManager()
    
    # 测试字典格式输入
    logger.info("测试字典格式输入")
    type_groups_dict = manager._group_instances_by_type(test_modules_dict)
    
    # 验证字典格式结果
    assert 'in01f01' in type_groups_dict, "字典格式：应该包含in01f01类型"
    assert 'sdfrtnq' in type_groups_dict, "字典格式：应该包含sdfrtnq类型"
    assert len(type_groups_dict['in01f01']) == 3, "字典格式：应该有3个in01f01实例"
    assert len(type_groups_dict['sdfrtnq']) == 3, "字典格式：应该有3个sdfrtnq实例"
    
    # 测试列表格式输入
    logger.info("测试列表格式输入")
    type_groups_list = manager._group_instances_by_type(test_modules_list)
    
    # 验证列表格式结果
    assert 'in01f01' in type_groups_list, "列表格式：应该包含in01f01类型"
    assert 'sdfrtnq' in type_groups_list, "列表格式：应该包含sdfrtnq类型"
    assert len(type_groups_list['in01f01']) == 2, "列表格式：应该有2个in01f01实例"
    assert len(type_groups_list['sdfrtnq']) == 2, "列表格式：应该有2个sdfrtnq实例"
    
    # 验证实例信息
    for inst in type_groups_dict['in01f01']:
        assert 'module' in inst, "实例应该包含module信息"
        assert 'name' in inst, "实例应该包含name信息"
        assert 'type' in inst, "实例应该包含type信息"
        assert 'connections' in inst, "实例应该包含connections信息"
    
    logger.info("实例分组测试通过")

def test_functional_modules():
    """测试功能模块创建"""
    logger.info("开始测试功能模块创建")
    
    # 创建测试数据
    test_type_groups = {
        'in01f01': [
            {
                'module': 'top',
                'name': 'inst1',
                'type': 'in01f01',
                'connections': {'in': 'a', 'out': 'b'}
            },
            {
                'module': 'top',
                'name': 'inst2',
                'type': 'in01f01',
                'connections': {'in': 'c', 'out': 'd'}
            }
        ],
        'sdfrtnq': [
            {
                'module': 'top',
                'name': 'inst3',
                'type': 'sdfrtnq',
                'connections': {'clk': 'clk', 'd': 'e', 'q': 'f'}
            },
            {
                'module': 'top',
                'name': 'inst4',
                'type': 'sdfrtnq',
                'connections': {'clk': 'clk', 'd': 'g', 'q': 'h'}
            }
        ]
    }
    
    # 创建管理器实例
    manager = HierarchicalDecompositionManager()
    
    # 执行功能模块创建
    functional_modules = manager._create_functional_modules(test_type_groups)
    
    # 验证结果
    assert 'func_in01f01' in functional_modules, "应该包含func_in01f01模块"
    assert 'func_sdfrtnq' in functional_modules, "应该包含func_sdfrtnq模块"
    
    # 验证in01f01模块
    in01f01_module = functional_modules['func_in01f01']
    assert len(in01f01_module['instances']) == 2, "应该有2个in01f01实例"
    assert 'inst1' in in01f01_module['instances'], "应该包含inst1"
    assert 'inst2' in in01f01_module['instances'], "应该包含inst2"
    assert set(in01f01_module['connections']) == {'a', 'b', 'c', 'd'}, "连接信息不正确"
    
    # 验证sdfrtnq模块
    sdfrtnq_module = functional_modules['func_sdfrtnq']
    assert len(sdfrtnq_module['instances']) == 2, "应该有2个sdfrtnq实例"
    assert 'inst3' in sdfrtnq_module['instances'], "应该包含inst3"
    assert 'inst4' in sdfrtnq_module['instances'], "应该包含inst4"
    assert set(sdfrtnq_module['connections']) == {'clk', 'e', 'f', 'g', 'h'}, "连接信息不正确"
    
    logger.info("功能模块创建测试通过")

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_instance_grouping()
    test_functional_modules()