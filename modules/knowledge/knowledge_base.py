"""
知识库模块
"""

import json
import os
import logging
import pickle
from typing import Dict, List, Optional, Union, Any
import numpy as np
from collections import defaultdict
from functools import lru_cache
from datetime import datetime
import hashlib
from ..core.hierarchy import HierarchicalDecompositionManager
from ..utils.llm_manager import LLMManager
from pathlib import Path
from .pdf_processing_status import PDFProcessingStatus

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """知识库管理器，整合层次化分解和知识检索功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化知识库管理器
        
        Args:
            config: 知识库配置
        """
        self.config = config
        self._validate_config()
        self.data = {}
        self.cases = []
        self.knowledge_graph = {
            'global': [],
            'module': [],
            'connection': [],
            'constraint': []
        }
        self.layout_experience_path = self.config.get('layout_experience', './layout_experience')
        self._init_components()
        self._init_knowledge_base()
        self.load()  # 在初始化时就加载数据
        
    def _validate_config(self):
        """验证配置"""
        required_fields = ['path', 'format']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in knowledge base config")
                
        # 验证层次配置
        if 'hierarchy_config' not in self.config:
            self.config['hierarchy_config'] = {
                'levels': [
                    {'name': 'system', 'threshold': 0.8},
                    {'name': 'module', 'threshold': 0.6},
                    {'name': 'component', 'threshold': 0.4}
                ]
            }
            
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
            # 初始化路径
            self.path = Path(self.config['path'])
            self.format = self.config['format']
            
            # 初始化层次管理器
            from modules.core.hierarchy import HierarchicalDecompositionManager
            hierarchy_config = {
                'levels': self.config.get('hierarchy_config', {}).get('levels', [
                    {'name': 'system', 'threshold': 0.8},
                    {'name': 'module', 'threshold': 0.6},
                    {'name': 'component', 'threshold': 0.4}
                ]),
                'llm_config': self.config.get('llm_config', {
                    'base_url': 'http://localhost:11434',
                    'model': 'llama2',
                    'temperature': 0.7,
                    'max_tokens': 1000
                })
            }
            self.hierarchy_manager = HierarchicalDecompositionManager(hierarchy_config)
            
            # 初始化LLM管理器
            from modules.utils.llm_manager import LLMManager
            self.llm_manager = LLMManager(
                self.config.get('llm_config', {
                    'base_url': 'http://localhost:11434',
                    'model': 'llama2',
                    'temperature': 0.7,
                    'max_tokens': 1000
                })
            )
            
        except Exception as e:
            logging.error(f"初始化组件失败: {str(e)}")
            raise
        
    def _init_knowledge_base(self):
        """初始化知识库"""
        try:
            # 确保目录存在
            os.makedirs(self.layout_experience_path, exist_ok=True)
            
            # 初始化数据文件路径
            self.data_file = os.path.join(self.layout_experience_path, "data.pkl")
            
            # 如果数据文件不存在，创建空的知识库
            if not os.path.exists(self.data_file):
                with open(self.data_file, 'wb') as f:
                    pickle.dump([], f)
                logger.info(f"创建新的知识库文件: {self.data_file}")
                
        except Exception as e:
            logger.error(f"初始化知识库失败: {str(e)}")
            raise
            
    def _load_data(self) -> List[Dict]:
        """加载知识库数据
        
        Returns:
            List[Dict]: 知识列表
        """
        try:
            # 优先尝试加载layout_experience/cases.pkl（扩展的151个案例）
            layout_cases_file = os.path.join(self.layout_experience_path, "cases.pkl")
            if os.path.exists(layout_cases_file) and os.path.getsize(layout_cases_file) > 1000:
                logger.info(f"加载扩展案例数据: {layout_cases_file}")
                with open(layout_cases_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"成功加载 {len(data)} 个扩展案例")
                return data
            
            # 其次尝试加载cases.pkl（真实ISPD数据）
            cases_file = os.path.join(self.layout_experience_path, "cases.pkl")
            if os.path.exists(cases_file) and os.path.getsize(cases_file) > 1000:  # 确保文件有实际内容
                logger.info(f"加载真实ISPD案例数据: {cases_file}")
                with open(cases_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"成功加载 {len(data)} 个真实ISPD案例")
                return data
            
            # 如果cases.pkl不存在或为空，尝试加载JSON格式的案例数据
            if self.format == 'json' and os.path.exists(self.path):
                logger.info(f"加载JSON格式案例数据: {self.path}")
                try:
                    with open(self.path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"成功加载 {len(data)} 个JSON案例")
                    return data
                except Exception as e:
                    logger.error(f"加载JSON案例数据失败: {str(e)}")
            
              # 如果JSON加载失败，尝试加载data.pkl
            if os.path.exists(self.data_file):
                logger.info(f"加载知识库文件: {self.data_file}")
                # 检查文件大小
                if os.path.getsize(self.data_file) == 0:
                    logger.info("知识库文件为空")
                    return []
                    
                # 加载数据
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # 确保返回的是列表
                if data is None:
                    return []
                
                logger.info(f"从data.pkl加载了 {len(data)} 个案例")
                return data
            else:
                logger.info(f"知识库文件不存在: {self.data_file}")
                return []
        except Exception as e:
            logger.error(f"加载知识库数据失败: {str(e)}")
            return []
            
    def _save_data(self):
        """保存知识库数据"""
        try:
            if not os.path.exists(self.layout_experience_path):
                os.makedirs(self.layout_experience_path)
            
            # 保存案例数据
            cases_file = os.path.join(self.layout_experience_path, "cases.pkl")
            try:
                with open(cases_file, 'wb') as f:
                    pickle.dump(self.cases, f)
            except Exception as e:
                logger.error(f"保存案例数据失败: {str(e)}")
            
            # 保存知识图谱
            graph_file = os.path.join(self.layout_experience_path, "knowledge_graph.pkl")
            try:
                with open(graph_file, 'wb') as f:
                    pickle.dump(self.knowledge_graph, f)
            except Exception as e:
                logger.error(f"保存知识图谱失败: {str(e)}")
            
        except Exception as e:
            logger.error(f"保存知识库数据失败: {str(e)}")
            
    def add_case(self, 
                layout: Dict,
                optimization_result: Dict,
                metadata: Optional[Dict] = None):
        """添加布局经验到知识库
        
        Args:
            layout: 布局数据
            optimization_result: 优化结果
            metadata: 额外的元数据
        """
        case = {
            'id': len(self.cases),
            'layout': layout,
            'optimization_result': optimization_result,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 提取特征
        features = {
            'global': self._extract_global_features(layout),
            'module': self._extract_module_features(layout),
            'connection': self._extract_connection_features(layout),
            'constraint': self._extract_constraint_features(layout, optimization_result)
        }
        
        # 更新知识图谱
        self._update_knowledge_graph(case['id'], features, layout, optimization_result)
        
        # 添加案例
        self.cases.append(case)
        
        # 保存数据
        self._save_data()
        
    def _extract_global_features(self, layout: Dict) -> Dict:
        """提取全局特征
        
        Args:
            layout: 布局数据
            
        Returns:
            Dict: 全局特征
        """
        try:
            if not layout:
                logger.warning("布局数据为空")
                return {}
                
            features = {}
            
            # 提取组件数量
            if 'components' in layout and layout['components']:
                features['num_components'] = len(layout['components'])
                
                # 提取组件尺寸信息
                widths = []
                heights = []
                for comp in layout['components']:
                    if comp and 'width' in comp:
                        widths.append(comp['width'])
                    if comp and 'height' in comp:
                        heights.append(comp['height'])
                        
                if widths:
                    features['avg_width'] = sum(widths) / len(widths)
                    features['max_width'] = max(widths)
                if heights:
                    features['avg_height'] = sum(heights) / len(heights)
                    features['max_height'] = max(heights)
            
            # 提取网络信息
            if 'nets' in layout and layout['nets']:
                features['num_nets'] = len(layout['nets'])
                
                # 提取连接信息
                connections = []
                for net in layout['nets']:
                    if net and 'connections' in net and net['connections']:
                        connections.extend(net['connections'])
                features['num_connections'] = len(connections)
            
            # 提取元数据信息
            if 'metadata' in layout and layout['metadata']:
                features['has_metadata'] = True
                if 'dexceptription' in layout['metadata']:
                    features['has_dexceptription'] = True
                if 'author' in layout['metadata']:
                    features['has_author'] = True
                if 'date' in layout['metadata']:
                    features['has_date'] = True
            
            return features
            
        except Exception as e:
            logger.error(f"提取全局特征失败: {str(e)}")
            return {}
        
    def _extract_module_features(self, layout: Dict) -> List[np.ndarray]:
        """提取模块特征
        
        Args:
            layout: 布局数据
            
        Returns:
            模块特征列表
        """
        try:
            if not layout or 'components' not in layout or not layout['components']:
                return []
                
            features = []
            for comp in layout['components']:
                if not comp:
                    continue
                    
                # 计算面积
                if 'width' not in comp or 'height' not in comp:
                    continue
                    
                area = comp['width'] * comp['height']
                
                comp_features = [
                    area,
                    comp['width'],
                    comp['height'],
                    len(comp.get('connections', {}))
                ]
                features.append(np.array(comp_features))
            return features
            
        except Exception as e:
            logger.error(f"提取模块特征失败: {str(e)}")
            return []
        
    def _extract_connection_features(self, layout: Dict) -> List[np.ndarray]:
        """提取连接特征
        
        Args:
            layout: 布局数据
            
        Returns:
            连接特征列表
        """
        try:
            if not layout or 'nets' not in layout or not layout['nets']:
                return []
                
            features = []
            for net in layout['nets']:
                if not net or 'connections' not in net or not net['connections']:
                    continue
                    
                net_features = [
                    len(net['connections']),
                    net.get('length', 0),
                    net.get('width', 0)
                ]
                features.append(np.array(net_features))
            return features
            
        except Exception as e:
            logger.error(f"提取连接特征失败: {str(e)}")
            return []
        
    def _extract_constraint_features(self, 
                                   layout: Dict,
                                   optimization_result: Dict) -> np.ndarray:
        """提取约束特征
        
        Args:
            layout: 布局数据
            optimization_result: 优化结果
            
        Returns:
            约束特征向量
        """
        features = []
        
        # 时序约束
        features.append(optimization_result.get('timing_slack', 0))
        
        # 功耗约束
        features.append(optimization_result.get('power', 0))
        
        # 面积约束
        features.append(optimization_result.get('area', 0))
        
        # 拥塞约束
        features.append(optimization_result.get('congestion', 0))
        
        return np.array(features)
        
    def _update_knowledge_graph(self,
                              case_id: int,
                              features: Dict[str, Union[np.ndarray, List[np.ndarray]]],
                              layout: Dict,
                              optimization_result: Dict):
        """更新知识图谱
        
        Args:
            case_id: 案例ID
            features: 特征字典
            layout: 布局数据
            optimization_result: 优化结果
        """
        # 添加全局特征节点
        self.knowledge_graph['global'].append({
            'case_id': case_id,
            'features': features['global']
        })
        
        # 添加模块特征节点
        for i, module_features in enumerate(features['module']):
            self.knowledge_graph['module'].append({
                'case_id': case_id,
                'module_id': i,
                'features': module_features
            })
            
        # 添加连接特征节点
        for i, conn_features in enumerate(features['connection']):
            self.knowledge_graph['connection'].append({
                'case_id': case_id,
                'connection_id': i,
                'features': conn_features
            })
            
        # 添加约束特征节点
        self.knowledge_graph['constraint'].append({
            'case_id': case_id,
            'features': features['constraint']
        })
        
    def get_similar_cases(self, query_features: Dict, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Dict]:
        """获取相似案例
        
        Args:
            query_features: 查询特征
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            List[Dict]: 相似案例列表
        """
        try:
            if not self.cases:
                logger.warning("知识库为空，返回默认案例")
                return [self._get_default_case()]
            
            # 计算所有案例的相似度
            similarities = []
            for case in self.cases:
                try:
                    # 从案例中提取特征用于相似度计算
                    case_features = self._extract_case_features_for_similarity(case)
                    
                    # 使用改进的相似度计算
                    similarity = self._compute_similarity(query_features, case_features)
                    
                    similarities.append({
                        'case': case,
                        'similarity': similarity
                    })
                        
                except Exception as e:
                    logger.error(f"计算案例相似度失败: {str(e)}")
                    continue
                    
            # 按相似度排序
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 过滤低于阈值的结果
            filtered_results = [
                item for item in similarities 
                if item['similarity'] >= similarity_threshold
            ]
            
            # 返回top_k结果
            result = filtered_results[:top_k]
            
            # 记录日志
            if result:
                logger.info(f"找到 {len(result)} 个相似案例，最高相似度: {result[0]['similarity']:.3f}")
            else:
                logger.warning(f"未找到相似度 >= {similarity_threshold} 的案例")
                # 返回相似度最高的案例作为备选
                if similarities:
                    result = [similarities[0]]
                    logger.info(f"返回相似度最高的案例: {result[0]['similarity']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"获取相似案例失败: {str(e)}")
            return [self._get_default_case()]
    
    def _get_default_case(self) -> Dict:
        """获取默认案例"""
        return {
            'case': {
                'id': 'default',
                'features': {
                    'num_components': 1000,
                    'area': 100000000,
                    'component_density': 0.1,
                    'hierarchy': {'modules': ['default']},
                    'constraints': {'timing': {'max_delay': 1000}, 'power': {'max_power': 1000}, 'special_nets': 2}
                },
                'layout_strategy': 'default',
                'parameters': {'k': 3, 'iterations': 10}
            },
            'similarity': 0.1
        }
    
    def _compute_similarity(self, features1: Dict, features2: Dict) -> float:
        """计算两个特征字典之间的相似度
        
        Args:
            features1: 第一个特征字典
            features2: 第二个特征字典
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            if not features1 or not features2:
                return 0.0
                
            total_score = 0.0
            total_weight = 0.0
            
            # 1. 组件数量相似度 (权重: 0.20)
            if 'num_components' in features1 and 'num_components' in features2:
                comp1 = features1['num_components']
                comp2 = features2['num_components']
                if comp1 > 0 and comp2 > 0:
                    comp_sim = 1.0 - abs(comp1 - comp2) / max(comp1, comp2)
                    total_score += comp_sim * 0.20
                    total_weight += 0.20
            
            # 2. 面积相似度 (权重: 0.15)
            if 'area' in features1 and 'area' in features2:
                area1 = features1['area']
                area2 = features2['area']
                if area1 > 0 and area2 > 0:
                    area_sim = 1.0 - abs(area1 - area2) / max(area1, area2)
                    total_score += area_sim * 0.15
                    total_weight += 0.15
            
            # 3. 组件密度相似度 (权重: 0.15)
            if 'component_density' in features1 and 'component_density' in features2:
                density1 = features1['component_density']
                density2 = features2['component_density']
                density_sim = 1.0 - abs(density1 - density2)
                total_score += density_sim * 0.15
                total_weight += 0.15
            
            # 4. 复杂度相似度 (权重: 0.15)
            if 'complexity' in features1 and 'complexity' in features2:
                complexity1 = features1['complexity']
                complexity2 = features2['complexity']
                complexity_sim = 1.0 - abs(complexity1 - complexity2)
                total_score += complexity_sim * 0.15
                total_weight += 0.15
            
            # 5. 层次结构相似度 (权重: 0.15)
            if 'hierarchy' in features1 and 'hierarchy' in features2:
                hierarchy1 = features1['hierarchy']
                hierarchy2 = features2['hierarchy']
                hierarchy_sim = self._compute_hierarchy_similarity(hierarchy1, hierarchy2)
                total_score += hierarchy_sim * 0.15
                total_weight += 0.15
            
            # 6. 约束条件相似度 (权重: 0.20)
            if 'constraints' in features1 and 'constraints' in features2:
                constraints1 = features1['constraints']
                constraints2 = features2['constraints']
                constraint_sim = self._compute_constraint_similarity(constraints1, constraints2)
                total_score += constraint_sim * 0.20
                total_weight += 0.20
            
            # 计算加权平均相似度
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            return 0.0
            
    def _compute_hierarchy_similarity(self, hierarchy1: Dict, hierarchy2: Dict) -> float:
        """计算层次结构相似度"""
        try:
            if not hierarchy1 or not hierarchy2:
                return 0.0
                
            # 提取模块列表
            modules1 = set(hierarchy1.get('modules', []))
            modules2 = set(hierarchy2.get('modules', []))
            
            if not modules1 or not modules2:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = len(modules1.intersection(modules2))
            union = len(modules1.union(modules2))
            
            if union > 0:
                return intersection / union
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"层次结构相似度计算失败: {str(e)}")
            return 0.0
            
    def _extract_case_features_for_similarity(self, case: Dict) -> Dict:
        """从案例中提取特征用于相似度计算
        
        Args:
            case: 案例数据
            
        Returns:
            Dict: 提取的特征
        """
        try:
            features = {}
            
            # 从metadata中提取基本信息
            metadata = case.get('metadata', {})
            optimization_result = case.get('optimization_result', {})
            
            # 1. 组件数量 - 从port_count估算，使用更合理的估算方法
            port_count = metadata.get('port_count', 0)
            if port_count > 0:
                # 根据设计类型和复杂度调整估算系数
                design_type = metadata.get('design_type', '')
                complexity = metadata.get('complexity', 0.5)
                
                # 不同设计类型的组件密度不同
                if '超大规模' in design_type:
                    multiplier = 50  # 超大规模设计组件密度高
                elif 'FFT' in design_type or '变换' in design_type:
                    multiplier = 30  # FFT设计组件密度中等
                elif 'PCI' in design_type or '桥接' in design_type:
                    multiplier = 20  # PCI桥接组件密度较低
                else:
                    multiplier = 25  # 默认值
                
                # 根据复杂度调整
                complexity_factor = 0.5 + complexity  # 复杂度越高，组件数越多
                features['num_components'] = int(port_count * multiplier * complexity_factor)
            else:
                # 如果没有port_count，尝试从其他信息估算
                design_type = metadata.get('design_type', '')
                if '超大规模' in design_type:
                    features['num_components'] = 50000
                elif 'FFT' in design_type:
                    features['num_components'] = 30000
                elif 'PCI' in design_type:
                    features['num_components'] = 15000
                else:
                    features['num_components'] = 20000
            
            # 2. 面积 - 从optimization_result中获取
            area = optimization_result.get('area', 0)
            if area <= 0:
                # 如果area为0，尝试从core_bbox估算
                core_bbox = metadata.get('core_bbox', '')
                if core_bbox and 'um' in core_bbox:
                    try:
                        # 解析bbox格式: "( x1 y1 ) ( x2 y2 ) um"
                        bbox_parts = core_bbox.replace('(', '').replace(')', '').replace('um', '').split()
                        if len(bbox_parts) >= 4:
                            x1, y1, x2, y2 = map(float, bbox_parts[:4])
                            area = (x2 - x1) * (y2 - y1)
                    except:
                        area = 1000000  # 默认面积
                else:
                    area = 1000000  # 默认面积
            
            features['area'] = area
            
            # 3. 组件密度 - 基于面积和组件数量计算
            if area > 0 and features['num_components'] > 0:
                features['component_density'] = features['num_components'] / area
            else:
                features['component_density'] = 0.1  # 默认值
            
            # 4. 层次结构 - 从metadata中提取
            design_type = metadata.get('design_type', '')
            features['hierarchy'] = {
                'levels': ['top'],
                'modules': [design_type] if design_type else ['default']
            }
            
            # 5. 约束条件 - 从metadata中提取
            constraints_list = metadata.get('constraints', [])
            features['constraints'] = {
                'timing': {'max_delay': 1000},  # 默认值
                'power': {'max_power': 1000},   # 默认值
                'special_nets': len(constraints_list)
            }
            
            # 如果有具体的约束信息，更新约束条件
            if 'area' in constraints_list:
                features['constraints']['area'] = {'max_area': area}
            
            # 6. 添加复杂度信息
            features['complexity'] = metadata.get('complexity', 0.5)
            
            return features
            
        except Exception as e:
            logger.error(f"提取案例特征失败: {str(e)}")
            return {
                'num_components': 1000,
                'area': 100000000,
                'component_density': 0.1,
                'hierarchy': {'levels': ['top'], 'modules': ['default']},
                'constraints': {'timing': {'max_delay': 1000}, 'power': {'max_power': 1000}, 'special_nets': 2},
                'complexity': 0.5
            }
        
    def _compute_constraint_similarity(self, constraints1: Dict, constraints2: Dict) -> float:
        """计算约束条件相似度"""
        try:
            if not constraints1 or not constraints2:
                return 0.0
                
            total_score = 0.0
            total_weight = 0.0
            
            # 1. 时序约束相似度
            if 'timing' in constraints1 and 'timing' in constraints2:
                timing1 = constraints1['timing']
                timing2 = constraints2['timing']
                if 'max_delay' in timing1 and 'max_delay' in timing2:
                    delay1 = timing1['max_delay']
                    delay2 = timing2['max_delay']
                    if delay1 > 0 and delay2 > 0:
                        delay_sim = 1.0 - abs(delay1 - delay2) / max(delay1, delay2)
                        total_score += delay_sim * 0.5
                        total_weight += 0.5
            
            # 2. 功耗约束相似度
            if 'power' in constraints1 and 'power' in constraints2:
                power1 = constraints1['power']
                power2 = constraints2['power']
                if 'max_power' in power1 and 'max_power' in power2:
                    power_val1 = power1['max_power']
                    power_val2 = power2['max_power']
                    if power_val1 > 0 and power_val2 > 0:
                        power_sim = 1.0 - abs(power_val1 - power_val2) / max(power_val1, power_val2)
                        total_score += power_sim * 0.3
                        total_weight += 0.3
            
            # 3. 特殊网络数量相似度
            if 'special_nets' in constraints1 and 'special_nets' in constraints2:
                nets1 = constraints1['special_nets']
                nets2 = constraints2['special_nets']
                if isinstance(nets1, (int, float)) and isinstance(nets2, (int, float)):
                    nets_sim = 1.0 - abs(nets1 - nets2) / max(nets1, nets2, 1)
                    total_score += nets_sim * 0.2
                    total_weight += 0.2
            
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"约束条件相似度计算失败: {str(e)}")
            return 0.0
        
    def hierarchical_decomposition(self, design_info: Dict) -> Dict:
        """对设计进行层次化分解
        
        Args:
            design_info: 设计信息
            
        Returns:
            层次化分解结果
        """
        return self.decomposition_manager.hierarchical_decomposition(design_info)

    def retrieve(self, query: Dict, constraints: Dict, top_k: int = 5) -> Dict:
        """检索相关知识
        
        Args:
            query: 查询信息
            constraints: 约束条件
            top_k: 返回的相似案例数量
            
        Returns:
            检索到的知识，包含：
            - area_utilization: 面积利用率
            - routing_quality: 布线质量
            - timing_performance: 时序性能
            - power_distribution: 功耗分布
        """
        # 获取相似案例
        similar_cases = self.get_similar_cases(
            query_features=query,
            top_k=top_k,
            similarity_threshold=0.5
        )
        
        # 如果没有相似案例，返回默认知识
        if not similar_cases:
            return {
                'area_utilization': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                },
                'routing_quality': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                },
                'timing_performance': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                },
                'power_distribution': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                }
            }
        
        # 从相似案例中提取知识
        area_scores = []
        routing_scores = []
        timing_scores = []
        power_scores = []
        
        for case in similar_cases:
            result = case['optimization_result']
            if 'area_utilization' in result:
                area_scores.append(result['area_utilization'].get('score', 0.5))
            if 'routing_quality' in result:
                routing_scores.append(result['routing_quality'].get('score', 0.5))
            if 'timing_performance' in result:
                timing_scores.append(result['timing_performance'].get('score', 0.5))
            if 'power_distribution' in result:
                power_scores.append(result['power_distribution'].get('score', 0.5))
        
        # 计算平均分数
        def avg_score(scores):
            return sum(scores) / len(scores) if scores else 0.5
        
        return {
            'area_utilization': {
                'score': avg_score(area_scores),
                'issues': ['基于历史案例的平均分数']
            },
            'routing_quality': {
                'score': avg_score(routing_scores),
                'issues': ['基于历史案例的平均分数']
            },
            'timing_performance': {
                'score': avg_score(timing_scores),
                'issues': ['基于历史案例的平均分数']
            },
            'power_distribution': {
                'score': avg_score(power_scores),
                'issues': ['基于历史案例的平均分数']
            }
        } 

    def build(self, case: Dict):
        """构建知识库
        
        Args:
            case: 测试用例信息
        """
        logger.info(f"开始构建知识库: {case['name']}")
        
        # 构建文本知识库
        self._build_text_kb(case)
        
        # 构建图像知识库
        self._build_image_kb(case)
        
        # 构建结构化知识库
        self._build_structured_kb(case)
        
        # 构建图知识库
        self._build_graph_kb(case)
        
        logger.info("知识库构建完成")
        
    def _build_text_kb(self, case: Dict):
        """构建文本知识库
        
        Args:
            case: 测试用例信息
        """
        text_kb = {
            'case_name': case['name'],
            'netlist_info': self._extract_netlist_info(case['netlist']),
            'constraints': case['constraints'],
            'design_rules': self._extract_design_rules(case),
            'optimization_guidelines': self._extract_optimization_guidelines(case)
        }
        
        kb_path = self.text_path / f"{case['name']}_text.json"
        with open(kb_path, 'w') as f:
            json.dump(text_kb, f, indent=2)
            
    def _build_image_kb(self, case: Dict):
        """构建图像知识库
        
        Args:
            case: 测试用例信息
        """
        # 提取布局图像
        layout_images = self._extract_layout_images(case['layout'])
        
        # 保存图像
        for name, image in layout_images.items():
            image_path = self.image_path / f"{case['name']}_{name}.png"
            image.save(image_path)
            
    def _build_structured_kb(self, case: Dict):
        """构建结构化知识库
        
        Args:
            case: 测试用例信息
        """
        structured_kb = {
            'case_name': case['name'],
            'hierarchy': self._extract_hierarchy(case['netlist']),
            'timing_info': self._extract_timing_info(case),
            'power_info': self._extract_power_info(case),
            'area_info': self._extract_area_info(case)
        }
        
        kb_path = self.structured_path / f"{case['name']}_structured.json"
        with open(kb_path, 'w') as f:
            json.dump(structured_kb, f, indent=2)
            
    def _build_graph_kb(self, case: Dict):
        """构建图知识库
        
        Args:
            case: 测试用例信息
        """
        graph_kb = {
            'case_name': case['name'],
            'connectivity': self._extract_connectivity(case['netlist']),
            'dependencies': self._extract_dependencies(case),
            'constraint_graph': self._extract_constraint_graph(case)
        }
        
        kb_path = self.graph_path / f"{case['name']}_graph.json"
        with open(kb_path, 'w') as f:
            json.dump(graph_kb, f, indent=2)
            
    def _extract_netlist_info(self, netlist: Dict) -> Dict:
        """提取网表信息
        
        Args:
            netlist: 网表数据
            
        Returns:
            Dict: 网表信息
        """
        return {
            'module_count': len(netlist.get('modules', [])),
            'instance_count': len(netlist.get('instances', [])),
            'net_count': len(netlist.get('nets', [])),
            'pin_count': len(netlist.get('pins', [])),
            'hierarchy_levels': self._calculate_hierarchy_levels(netlist)
        }
        
    def _extract_design_rules(self, case: Dict) -> Dict:
        """提取设计规则
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 设计规则
        """
        return {
            'min_spacing': case['constraints'].get('min_spacing', 0.1),
            'min_width': case['constraints'].get('min_width', 0.1),
            'max_density': case['constraints'].get('max_density', 0.8),
            'power_grid': case['constraints'].get('power_grid', {}),
            'clock_tree': case['constraints'].get('clock_tree', {})
        }
        
    def _extract_optimization_guidelines(self, case: Dict) -> Dict:
        """提取优化指南
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 优化指南
        """
        return {
            'timing_optimization': {
                'critical_paths': case['constraints'].get('critical_paths', []),
                'clock_constraints': case['constraints'].get('clock_constraints', {}),
                'setup_hold_margins': case['constraints'].get('setup_hold_margins', {})
            },
            'power_optimization': {
                'power_budget': case['constraints'].get('power_budget', {}),
                'leakage_targets': case['constraints'].get('leakage_targets', {}),
                'dynamic_power_limits': case['constraints'].get('dynamic_power_limits', {})
            },
            'area_optimization': {
                'area_budget': case['constraints'].get('area_budget', {}),
                'utilization_targets': case['constraints'].get('utilization_targets', {})
            }
        }
        
    def _extract_layout_images(self, layout: Dict) -> Dict[str, Any]:
        """提取布局图像
        
        Args:
            layout: 布局数据
            
        Returns:
            Dict[str, Any]: 布局图像
        """
        # TODO: 实现布局图像提取
        return {}
        
    def _extract_hierarchy(self, netlist: Dict) -> Dict:
        """提取层次结构
        
        Args:
            netlist: 网表数据
            
        Returns:
            Dict: 层次结构
        """
        hierarchy = {}
        for module in netlist.get('modules', []):
            hierarchy[module['name']] = {
                'instances': [inst['name'] for inst in module.get('instances', [])],
                'nets': [net['name'] for net in module.get('nets', [])],
                'pins': [pin['name'] for pin in module.get('pins', [])]
            }
        return hierarchy
        
    def _extract_timing_info(self, case: Dict) -> Dict:
        """提取时序信息
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 时序信息
        """
        return {
            'clock_period': case['constraints'].get('clock_period', 0),
            'setup_time': case['constraints'].get('setup_time', 0),
            'hold_time': case['constraints'].get('hold_time', 0),
            'max_delay': case['constraints'].get('max_delay', 0),
            'min_delay': case['constraints'].get('min_delay', 0)
        }
        
    def _extract_power_info(self, case: Dict) -> Dict:
        """提取功耗信息
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 功耗信息
        """
        return {
            'total_power': case['constraints'].get('total_power', 0),
            'dynamic_power': case['constraints'].get('dynamic_power', 0),
            'leakage_power': case['constraints'].get('leakage_power', 0),
            'power_density': case['constraints'].get('power_density', 0)
        }
        
    def _extract_area_info(self, case: Dict) -> Dict:
        """提取面积信息
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 面积信息
        """
        return {
            'total_area': case['constraints'].get('total_area', 0),
            'cell_area': case['constraints'].get('cell_area', 0),
            'routing_area': case['constraints'].get('routing_area', 0),
            'utilization': case['constraints'].get('utilization', 0)
        }
        
    def _extract_connectivity(self, netlist: Dict) -> Dict:
        """提取连接关系
        
        Args:
            netlist: 网表数据
            
        Returns:
            Dict: 连接关系
        """
        connectivity = {}
        for net in netlist.get('nets', []):
            connectivity[net['name']] = {
                'source': net.get('source', ''),
                'sinks': net.get('sinks', []),
                'fanout': len(net.get('sinks', [])),
                'net_type': net.get('type', 'signal')
            }
        return connectivity
        
    def _extract_dependencies(self, case: Dict) -> Dict:
        """提取依赖关系
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 依赖关系
        """
        return {
            'timing_dependencies': case['constraints'].get('timing_dependencies', {}),
            'power_dependencies': case['constraints'].get('power_dependencies', {}),
            'area_dependencies': case['constraints'].get('area_dependencies', {})
        }
        
    def _extract_constraint_graph(self, case: Dict) -> Dict:
        """提取约束图
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 约束图
        """
        return {
            'timing_constraints': case['constraints'].get('timing_constraints', {}),
            'power_constraints': case['constraints'].get('power_constraints', {}),
            'area_constraints': case['constraints'].get('area_constraints', {}),
            'physical_constraints': case['constraints'].get('physical_constraints', {})
        }
        
    def _calculate_hierarchy_levels(self, netlist: Dict) -> int:
        """计算层次级别
        
        Args:
            netlist: 网表数据
            
        Returns:
            int: 层次级别
        """
        # TODO: 实现层次级别计算
        return 1

    def query(self, query_params: Dict) -> List[Dict]:
        """查询知识库
        
        Args:
            query_params: 查询参数
            
        Returns:
            List[Dict]: 查询结果列表
        """
        try:
            results = []
            
            # 遍历所有案例
            for case in self.cases:
                layout = case.get('layout', {})
                match = True
                
                # 检查每个查询条件
                for key, value in query_params.items():
                    if key not in layout:
                        match = False
                        break
                    if layout[key] != value:
                        match = False
                        break
                        
                if match:
                    results.append(layout)
                    
            return results
            
        except Exception as e:
            logger.error(f"查询知识库失败: {str(e)}")
            return []

    def save(self, path: str):
        """保存知识库
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.knowledge, f, indent=2)
            
    def load(self, path=None):
        """加载知识库数据
        
        Args:
            path: 知识库路径，如果为None则使用layout_experience_path
            
        Returns:
            self: 知识库实例
        """
        try:
            if path is None:
                path = self.layout_experience_path
                
            # 确保目录存在
            os.makedirs(path, exist_ok=True)
            
            # 优先尝试加载JSON格式的案例数据
            if self.format == 'json' and os.path.exists(self.path):
                try:
                    with open(self.path, 'r', encoding='utf-8') as f:
                        self.cases = json.load(f)
                    logger.info(f"成功加载JSON案例数据，包含 {len(self.cases)} 个案例")
                except Exception as e:
                    logger.error(f"加载JSON案例数据失败: {str(e)}")
                    self.cases = []
            
            # 如果JSON加载失败，尝试加载cases.pkl
            elif os.path.exists(os.path.join(path, "cases.pkl")):
                cases_file = os.path.join(path, "cases.pkl")
                try:
                    with open(cases_file, 'rb') as f:
                        self.cases = pickle.load(f)
                    logger.info(f"成功加载案例数据，包含 {len(self.cases)} 个案例")
                except Exception as e:
                    logger.error(f"加载案例数据失败: {str(e)}")
                    self.cases = []
            else:
                logger.warning(f"案例文件不存在")
                self.cases = []
                
            # 加载知识图谱
            graph_file = os.path.join(path, "knowledge_graph.pkl")
            if os.path.exists(graph_file):
                try:
                    with open(graph_file, 'rb') as f:
                        self.knowledge_graph = pickle.load(f)
                    logger.info("成功加载知识图谱")
                except Exception as e:
                    logger.error(f"加载知识图谱失败: {str(e)}")
                    self.knowledge_graph = {
                        'global': [],
                        'module': [],
                        'connection': [],
                        'constraint': []
                    }
            else:
                logger.warning(f"知识图谱文件不存在: {graph_file}")
                self.knowledge_graph = {
                    'global': [],
                    'module': [],
                    'connection': [],
                    'constraint': []
                }
                
            logger.info(f"知识库加载完成，包含 {len(self.cases)} 个案例")
            return self
            
        except Exception as e:
            logger.error(f"加载知识库数据失败: {str(e)}")
            self.cases = []
            self.knowledge_graph = {
                'global': [],
                'module': [],
                'connection': [],
                'constraint': []
            }
            return self

    def update(self, new_knowledge):
        """更新知识库数据"""
        # 确保每个知识条目都含有'name'字段
        for item in new_knowledge:
            if 'name' not in item:
                item['name'] = item.get('id', 'unknown')
        self.data.update(new_knowledge)
        self.cases.extend(new_knowledge.get('cases', []))
        return self

    def get_circuit_knowledge(self) -> Dict:
        """获取电路级知识
        
        Returns:
            Dict: 电路级知识
        """
        return self.knowledge['circuit']
        
    def get_module_knowledge(self) -> Dict:
        """获取模块级知识
        
        Returns:
            Dict: 模块级知识
        """
        return self.knowledge['module']
        
    def get_cell_knowledge(self) -> Dict:
        """获取单元级知识
        
        Returns:
            Dict: 单元级知识
        """
        return self.knowledge['cell']

    def add_pdf_knowledge(self, pdf_path: str) -> bool:
        """添加PDF知识到知识库
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            bool: 是否成功添加
        """
        # 检查是否已处理
        if self.pdf_status.is_processed(pdf_path):
            logger.info(f"PDF {pdf_path} 已处理，跳过")
            return True
            
        try:
            # 处理PDF
            pdf_data = self._process_pdf(pdf_path)
            
            # 添加到知识库
            self.add_case(
                layout=pdf_data['layout'],
                optimization_result=pdf_data['optimization_result'],
                metadata={
                    'source': 'pdf',
                    'filename': Path(pdf_path).name,
                    'type': 'layout_design',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # 标记为已处理
            self.pdf_status.mark_as_processed(
                pdf_path,
                metadata={'type': 'layout_design'}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"处理PDF {pdf_path} 失败: {str(e)}")
            return False
            
    def _process_pdf(self, pdf_path: str) -> Dict:
        """处理PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Dict: 处理后的数据
        """
        # PDF处理逻辑
        # ... 

    def add_knowledge(self, knowledge: Dict) -> bool:
        """添加知识到知识库
        
        Args:
            knowledge: 知识字典
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 获取现有知识
            all_knowledge = self._load_data()
            
            # 添加新知识
            all_knowledge.append(knowledge)
            
            # 保存数据
            with open(self.data_file, 'wb') as f:
                pickle.dump(all_knowledge, f)
                
            return True
            
        except Exception as e:
            logger.error(f"添加知识失败: {str(e)}")
            return False 

    def __len__(self):
        return len(self.cases) 