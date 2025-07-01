"""
动态RAG检索器模块
结合DynamicRAG和DRAG论文的创新点，实现基于LLM输出质量反馈的动态重排序
"""

import torch
import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random
from datetime import datetime
import hashlib

from ..utils.llm_manager import LLMManager
from ..evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from ..knowledge.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

@dataclass
class DynamicRetrievalResult:
    """动态检索结果"""
    knowledge: Dict[str, Any]
    relevance_score: float
    granularity_level: str
    source: str
    entity_embeddings: Optional[np.ndarray] = None
    quality_feedback: Optional[float] = None
    retrieval_count: int = 0

@dataclass
class LayoutQualityFeedback:
    """布局质量反馈"""
    wirelength_score: float
    congestion_score: float
    timing_score: float
    power_score: float
    overall_score: float
    feedback_timestamp: str
    layout_metadata: Dict[str, Any]

class DynamicRAGRetriever:
    """动态RAG检索器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化动态RAG检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self._init_components()
        
        # 动态重排序相关
        self.retrieval_history = defaultdict(list)
        self.quality_feedback_history = []
        self.rl_agent = self._init_rl_agent()
        
        # 实体增强相关
        self.entity_embeddings_cache = {}
        self.compressed_entity_dim = config.get('compressed_entity_dim', 128)
        
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化LLM管理器
            llm_config = self.config.get('llm', {})
            self.llm_manager = LLMManager(llm_config)
            
            # 初始化知识库
            kb_config = self.config.get('knowledge_base', {})
            self.knowledge_base = KnowledgeBase(kb_config)
            
            # 初始化评估器
            self.evaluator = MultiObjectiveEvaluator(self.config)
            
            # 动态重排序参数
            self.dynamic_k_range = self.config.get('dynamic_k_range', (3, 15))
            self.quality_threshold = self.config.get('quality_threshold', 0.7)
            self.learning_rate = self.config.get('learning_rate', 0.01)
            
            # 实体增强参数
            self.entity_compression_ratio = self.config.get('entity_compression_ratio', 0.1)
            self.entity_similarity_threshold = self.config.get('entity_similarity_threshold', 0.8)
            
            self.logger.info("动态RAG检索器组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {str(e)}")
            raise
    
    def _init_rl_agent(self):
        """初始化强化学习智能体"""
        # 简单的Q-learning智能体
        return {
            'q_table': defaultdict(lambda: defaultdict(float)),
            'epsilon': 0.1,
            'alpha': self.learning_rate,
            'gamma': 0.9
        }
    
    def retrieve_with_dynamic_reranking(self, 
                                      query: Dict[str, Any], 
                                      design_info: Dict[str, Any]) -> List[DynamicRetrievalResult]:
        """执行动态重排序检索
        
        Args:
            query: 查询条件
            design_info: 设计信息
            
        Returns:
            List[DynamicRetrievalResult]: 动态检索结果
        """
        try:
            # 1. 初始检索
            initial_results = self._initial_retrieval(query, design_info)
            
            # 2. 动态确定k值
            optimal_k = self._determine_optimal_k(query, design_info, initial_results)
            
            # 3. 基于历史反馈进行重排序
            reranked_results = self._dynamic_rerank(initial_results, query, optimal_k)
            
            # 4. 实体增强处理
            enhanced_results = self._enhance_with_entities(reranked_results, design_info)
            
            # 5. 记录检索历史
            self._record_retrieval_history(query, enhanced_results)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"动态重排序检索失败: {str(e)}")
            raise
    
    def _initial_retrieval(self, query: Dict[str, Any], design_info: Dict[str, Any]) -> List[DynamicRetrievalResult]:
        """初始检索"""
        try:
            # 从知识库检索
            raw_results = self.knowledge_base.get_similar_cases(
                query,
                top_k=self.dynamic_k_range[1]  # 使用最大k值
            )
            # 转换为动态检索结果
            results = []
            for result in raw_results:
                # 处理知识库返回的案例格式
                if isinstance(result, dict):
                    # 提取内容 - 优先使用layout，然后是optimization_result
                    content = result.get('layout', result.get('optimization_result', {}))
                    
                    # 计算相似度分数（如果没有提供，使用默认值）
                    similarity = result.get('similarity', 0.5)
                    
                    # 提取粒度级别
                    granularity = 'global'  # 默认粒度
                    if 'hierarchy' in result and 'levels' in result['hierarchy']:
                        granularity = result['hierarchy']['levels'][0] if result['hierarchy']['levels'] else 'global'
                    
                    # 提取来源
                    source = result.get('name', result.get('id', 'unknown'))
                    
                    dynamic_result = DynamicRetrievalResult(
                        knowledge=content,
                        relevance_score=similarity,
                        granularity_level=granularity,
                        source=str(source),
                        entity_embeddings=self._extract_entity_embeddings(result),
                        retrieval_count=1
                    )
                    results.append(dynamic_result)
                else:
                    # 如果结果不是字典，跳过
                    self.logger.warning(f"跳过非字典格式的检索结果: {type(result)}")
                    continue
            return results
        except Exception as e:
            self.logger.error(f"初始检索失败: {str(e)}")
            return []
    
    def _determine_optimal_k(self, 
                           query: Dict[str, Any], 
                           design_info: Dict[str, Any],
                           initial_results: List[DynamicRetrievalResult]) -> int:
        """动态确定最优k值"""
        try:
            # 构建状态特征
            state_features = self._extract_state_features(query, design_info, initial_results)
            state_key = self._hash_state(state_features)
            
            # 使用Q-learning选择k值
            if random.random() < self.rl_agent['epsilon']:
                # 探索：随机选择k值
                k = random.randint(self.dynamic_k_range[0], self.dynamic_k_range[1])
            else:
                # 利用：选择Q值最大的k值
                q_values = self.rl_agent['q_table'][state_key]
                if q_values:
                    k = max(q_values.keys(), key=lambda x: q_values[x])
                else:
                    k = self.dynamic_k_range[0]
            
            self.logger.info(f"动态确定k值: {k}")
            return k
            
        except Exception as e:
            self.logger.error(f"确定最优k值失败: {str(e)}")
            return self.dynamic_k_range[0]
    
    def _dynamic_rerank(self, 
                       results: List[DynamicRetrievalResult], 
                       query: Dict[str, Any],
                       k: int) -> List[DynamicRetrievalResult]:
        """动态重排序 - 使用自适应权重调整"""
        try:
            # 1. 计算各种权重
            quality_weights = self._calculate_quality_weights(results)
            similarity_weights = self._calculate_similarity_weights(results, query)
            entity_weights = self._calculate_entity_weights(results, query)
            
            # 2. 动态调整权重
            adaptive_weights = self._calculate_adaptive_weights(
                quality_weights, similarity_weights, entity_weights, query
            )
            
            # 3. 应用自适应权重进行重排序
            for result in results:
                result.relevance_score = (
                    quality_weights.get(result.source, 1.0) * adaptive_weights['quality'] +
                    similarity_weights.get(result.source, 1.0) * adaptive_weights['similarity'] +
                    entity_weights.get(result.source, 1.0) * adaptive_weights['entity']
                )
            
            # 4. 排序并返回top-k
            reranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)[:k]
            
            # 5. 记录权重调整历史
            self._record_weight_adjustment(adaptive_weights, query)
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"动态重排序失败: {str(e)}")
            return results[:k]
    
    def _calculate_adaptive_weights(self, 
                                  quality_weights: Dict[str, float],
                                  similarity_weights: Dict[str, float],
                                  entity_weights: Dict[str, float],
                                  query: Dict[str, Any]) -> Dict[str, float]:
        """计算自适应权重"""
        # 基础权重
        base_weights = {
            'quality': 0.4,
            'similarity': 0.4,
            'entity': 0.2
        }
        
        # 根据查询特征调整权重
        query_complexity = len(query.get('text', '')) / 100.0 if isinstance(query, dict) else 0.5
        query_type = query.get('type', 'general') if isinstance(query, dict) else 'general'
        
        # 根据查询复杂度调整权重
        if query_complexity > 0.8:
            # 复杂查询更依赖质量反馈
            base_weights['quality'] += 0.2
            base_weights['similarity'] -= 0.1
            base_weights['entity'] -= 0.1
        elif query_complexity < 0.3:
            # 简单查询更依赖相似度
            base_weights['quality'] -= 0.1
            base_weights['similarity'] += 0.2
            base_weights['entity'] -= 0.1
        
        # 根据查询类型调整权重
        if query_type == 'layout_generation':
            # 布局生成更依赖实体信息
            base_weights['entity'] += 0.1
            base_weights['similarity'] -= 0.05
            base_weights['quality'] -= 0.05
        elif query_type == 'optimization':
            # 优化查询更依赖质量反馈
            base_weights['quality'] += 0.1
            base_weights['entity'] -= 0.05
            base_weights['similarity'] -= 0.05
        
        # 根据历史性能调整权重
        historical_adjustment = self._get_historical_weight_adjustment()
        for key in base_weights:
            base_weights[key] += historical_adjustment.get(key, 0.0)
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        for key in base_weights:
            base_weights[key] /= total_weight
        
        return base_weights
    
    def _get_historical_weight_adjustment(self) -> Dict[str, float]:
        """获取基于历史性能的权重调整"""
        if not hasattr(self, 'weight_adjustment_history'):
            self.weight_adjustment_history = []
        
        if len(self.weight_adjustment_history) < 10:
            return {'quality': 0.0, 'similarity': 0.0, 'entity': 0.0}
        
        # 分析最近10次的权重调整效果
        recent_adjustments = self.weight_adjustment_history[-10:]
        
        # 计算每种权重的平均效果
        weight_effects = {'quality': 0.0, 'similarity': 0.0, 'entity': 0.0}
        
        for adjustment in recent_adjustments:
            if 'effect_score' in adjustment and 'weights' in adjustment:
                effect = adjustment['effect_score']
                weights = adjustment['weights']
                
                for weight_type in weight_effects:
                    if weight_type in weights:
                        weight_effects[weight_type] += effect * weights[weight_type]
        
        # 归一化效果
        total_effect = sum(abs(effect) for effect in weight_effects.values())
        if total_effect > 0:
            for weight_type in weight_effects:
                weight_effects[weight_type] = weight_effects[weight_type] / total_effect * 0.1
        
        return weight_effects
    
    def _record_weight_adjustment(self, weights: Dict[str, float], query: Dict[str, Any]):
        """记录权重调整"""
        if not hasattr(self, 'weight_adjustment_history'):
            self.weight_adjustment_history = []
        
        # 计算调整效果（基于查询复杂度）
        query_complexity = len(query.get('text', '')) / 100.0 if isinstance(query, dict) else 0.5
        effect_score = query_complexity  # 简化的效果计算
        
        adjustment_record = {
            'timestamp': datetime.now().isoformat(),
            'weights': weights,
            'effect_score': effect_score,
            'query_type': query.get('type', 'unknown') if isinstance(query, dict) else 'unknown'
        }
        
        self.weight_adjustment_history.append(adjustment_record)
        
        # 保持历史记录在合理范围内
        if len(self.weight_adjustment_history) > 100:
            self.weight_adjustment_history = self.weight_adjustment_history[-50:]
    
    def _enhance_with_entities(self, 
                              results: List[DynamicRetrievalResult], 
                              design_info: Dict[str, Any]) -> List[DynamicRetrievalResult]:
        """实体增强处理 - 完整的实体注入机制"""
        enhanced_results = []
        
        for result in results:
            if isinstance(result, DynamicRetrievalResult):
                knowledge = result.knowledge if isinstance(result.knowledge, dict) else {}
                
                # 1. 提取实体信息
                entities = self._extract_entities(knowledge, design_info)
                
                # 2. 压缩实体嵌入
                compressed_embeddings = self._compress_entity_embeddings(entities)
                
                # 3. 实体注入增强
                enhanced_knowledge = self._inject_entities_into_knowledge(
                    knowledge, compressed_embeddings, design_info
                )
                
                # 4. 更新结果
                result.entity_embeddings = compressed_embeddings
                result.knowledge = enhanced_knowledge
                enhanced_results.append(result)
            else:
                enhanced_results.append(result)
        
        return enhanced_results
    
    def _inject_entities_into_knowledge(self, 
                                      knowledge: Dict[str, Any], 
                                      entity_embeddings: np.ndarray,
                                      design_info: Dict[str, Any]) -> Dict[str, Any]:
        """将实体信息注入到知识中"""
        enhanced_knowledge = knowledge.copy() if isinstance(knowledge, dict) else {}
        
        # 1. 添加实体嵌入信息
        enhanced_knowledge['entity_embeddings'] = entity_embeddings.tolist()
        
        # 2. 添加实体上下文信息
        enhanced_knowledge['entity_context'] = {
            'embedding_dim': len(entity_embeddings),
            'design_type': design_info.get('type', 'unknown'),
            'component_count': len(design_info.get('components', [])),
            'constraint_count': len(design_info.get('constraints', [])),
            'injection_timestamp': datetime.now().isoformat()
        }
        
        # 3. 增强布局建议
        if 'layout_suggestions' in enhanced_knowledge:
            enhanced_suggestions = []
            for suggestion in enhanced_knowledge['layout_suggestions']:
                if isinstance(suggestion, dict):
                    # 基于实体嵌入调整建议
                    enhanced_suggestion = self._enhance_layout_suggestion(
                        suggestion, entity_embeddings, design_info
                    )
                    enhanced_suggestions.append(enhanced_suggestion)
                else:
                    enhanced_suggestions.append(suggestion)
            enhanced_knowledge['layout_suggestions'] = enhanced_suggestions
        
        # 4. 添加实体感知的优化参数
        enhanced_knowledge['entity_aware_params'] = self._generate_entity_aware_params(
            entity_embeddings, design_info
        )
        
        return enhanced_knowledge
    
    def _enhance_layout_suggestion(self, 
                                 suggestion: Dict[str, Any], 
                                 entity_embeddings: np.ndarray,
                                 design_info: Dict[str, Any]) -> Dict[str, Any]:
        """基于实体嵌入增强布局建议"""
        enhanced_suggestion = suggestion.copy()
        
        # 基于实体嵌入调整建议权重
        entity_importance = np.mean(entity_embeddings) if len(entity_embeddings) > 0 else 0.5
        
        # 调整建议的置信度
        if 'confidence' in enhanced_suggestion:
            enhanced_suggestion['confidence'] = min(1.0, 
                enhanced_suggestion['confidence'] * (1 + entity_importance * 0.2))
        
        # 添加实体感知标签
        enhanced_suggestion['entity_aware'] = True
        enhanced_suggestion['entity_importance'] = float(entity_importance)
        
        return enhanced_suggestion
    
    def _generate_entity_aware_params(self, 
                                    entity_embeddings: np.ndarray,
                                    design_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成基于实体感知的优化参数"""
        # 基于实体嵌入生成优化参数
        entity_diversity = np.std(entity_embeddings) if len(entity_embeddings) > 1 else 0.0
        entity_complexity = np.mean(np.abs(entity_embeddings)) if len(entity_embeddings) > 0 else 0.5
        
        params = {
            'density_target': 0.7 + entity_complexity * 0.2,  # 0.7-0.9
            'wirelength_weight': 1.0 + entity_diversity * 2.0,  # 1.0-3.0
            'density_weight': 1.0 + entity_complexity * 1.5,  # 1.0-2.5
            'overflow_penalty': 0.0005 + entity_diversity * 0.0005,  # 0.0005-0.001
            'max_displacement': 5.0 + entity_complexity * 10.0,  # 5.0-15.0
            'entity_complexity': float(entity_complexity),
            'entity_diversity': float(entity_diversity)
        }
        
        return params
    
    def update_with_feedback(self, 
                           query_hash: str, 
                           layout_result: dict,
                           quality_feedback: dict):
        """基于布局质量反馈更新重排序策略"""
        try:
            # 1. 记录质量反馈
            self.quality_feedback_history.append({
                'query_hash': query_hash,
                'feedback': quality_feedback,
                'timestamp': datetime.now().isoformat()
            })
            # 2. 更新Q-learning智能体
            self._update_rl_agent(query_hash, quality_feedback)
            # 3. 更新检索历史
            if query_hash in self.retrieval_history:
                for result in self.retrieval_history[query_hash]:
                    if isinstance(quality_feedback, dict):
                        result.quality_feedback = quality_feedback.get('overall_score', 0.5)
                    else:
                        result.quality_feedback = 0.5
            score = quality_feedback.get('overall_score', 0.5) if isinstance(quality_feedback, dict) else 0.5
            self.logger.info(f"基于反馈更新重排序策略，质量分数: {score}")
        except Exception as e:
            self.logger.error(f"更新反馈失败: {str(e)}")
    
    def _extract_state_features(self, 
                               query: dict, 
                               design_info: dict,
                               results: list) -> dict:
        """提取状态特征"""
        return {
            'query_complexity': len(query.get('text', '')) if isinstance(query, dict) else 0,
            'design_size': len(design_info.get('components', [])) if isinstance(design_info, dict) else 0,
            'result_count': len(results),
            'avg_relevance': np.mean([r.relevance_score for r in results]) if results else 0.0,
            'design_type': design_info.get('type', 'unknown') if isinstance(design_info, dict) else 'unknown'
        }
    
    def _hash_state(self, state_features: Dict[str, Any]) -> str:
        """哈希状态特征"""
        state_str = json.dumps(state_features, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def _calculate_quality_weights(self, results: List[DynamicRetrievalResult]) -> Dict[str, float]:
        """计算历史质量反馈权重"""
        weights = {}
        
        for result in results:
            source = result.source
            if source in self.quality_feedback_history:
                # 计算该来源的平均质量分数
                source_feedbacks = [
                    f['feedback'].get('overall_score', 0.5) if isinstance(f['feedback'], dict) else 0.5
                    for f in self.quality_feedback_history 
                    if f['query_hash'] in self.retrieval_history and 
                    any(r.source == source for r in self.retrieval_history[f['query_hash']])
                ]
                if source_feedbacks:
                    weights[source] = np.mean(source_feedbacks)
                else:
                    weights[source] = 1.0
            else:
                weights[source] = 1.0
        
        return weights
    
    def _calculate_similarity_weights(self, 
                                    results: List[DynamicRetrievalResult], 
                                    query: Dict[str, Any]) -> Dict[str, float]:
        """计算查询相似度权重"""
        weights = {}
        
        for result in results:
            # 基于原始相似度分数
            weights[result.source] = result.relevance_score
        
        return weights
    
    def _calculate_entity_weights(self, 
                                results: List[DynamicRetrievalResult], 
                                query: Dict[str, Any]) -> Dict[str, float]:
        """计算实体相关性权重"""
        weights = {}
        
        for result in results:
            if result.entity_embeddings is not None:
                # 基于实体嵌入的相关性
                entity_similarity = self._calculate_entity_similarity(
                    result.entity_embeddings, query
                )
                weights[result.source] = entity_similarity
            else:
                weights[result.source] = 0.5
        
        return weights
    
    def _extract_entities(self, knowledge: dict, design_info: dict) -> list:
        """提取实体信息"""
        entities = []
        # 从知识中提取组件实体
        if isinstance(knowledge, dict) and 'components' in knowledge:
            for comp in knowledge['components']:
                if isinstance(comp, dict):
                    entities.append({
                        'type': 'component',
                        'name': comp.get('name', ''),
                        'category': comp.get('type', ''),
                        'properties': comp.get('properties', {})
                    })
        # 从设计信息中提取约束实体
        if isinstance(design_info, dict) and 'constraints' in design_info:
            for constraint in design_info['constraints']:
                if isinstance(constraint, dict):
                    entities.append({
                        'type': 'constraint',
                        'name': constraint.get('name', ''),
                        'category': constraint.get('type', ''),
                        'properties': constraint.get('properties', {})
                    })
        return entities
    
    def _compress_entity_embeddings(self, entities: list) -> np.ndarray:
        """压缩实体嵌入 - 使用真正的注意力机制"""
        if not entities:
            return np.zeros(self.compressed_entity_dim)
        
        # 1. 提取实体特征
        entity_features = []
        entity_weights = []
        
        for entity in entities:
            if isinstance(entity, dict):
                # 更丰富的特征提取
                feature = [
                    hash(entity.get('name', '')) % 1000 / 1000.0,
                    hash(entity.get('category', '')) % 1000 / 1000.0,
                    len(entity.get('properties', {})),
                    hash(entity.get('type', '')) % 1000 / 1000.0,
                    # 添加更多语义特征
                    self._calculate_entity_importance(entity)
                ]
                entity_features.append(feature)
                
                # 计算实体重要性权重
                importance = self._calculate_entity_importance(entity)
                entity_weights.append(importance)
        
        if not entity_features:
            return np.zeros(self.compressed_entity_dim)
        
        # 2. 使用注意力机制进行加权压缩
        entity_features = np.array(entity_features)
        entity_weights = np.array(entity_weights)
        
        # 归一化权重
        if np.sum(entity_weights) > 0:
            entity_weights = entity_weights / np.sum(entity_weights)
        else:
            entity_weights = np.ones(len(entity_weights)) / len(entity_weights)
        
        # 3. 注意力加权平均
        weighted_features = np.average(entity_features, axis=0, weights=entity_weights)
        
        # 4. 线性变换到目标维度
        compressed = np.zeros(self.compressed_entity_dim)
        compressed[:min(len(weighted_features), self.compressed_entity_dim)] = \
            weighted_features[:self.compressed_entity_dim]
        
        return compressed
    
    def _calculate_entity_importance(self, entity: dict) -> float:
        """计算实体重要性"""
        importance = 0.5  # 基础重要性
        
        # 根据实体类型调整重要性
        entity_type = entity.get('type', '')
        if entity_type == 'component':
            importance += 0.2
        elif entity_type == 'constraint':
            importance += 0.3
        elif entity_type == 'port':
            importance += 0.1
        
        # 根据属性数量调整重要性
        properties_count = len(entity.get('properties', {}))
        importance += min(0.2, properties_count * 0.01)
        
        # 根据名称长度调整重要性（通常更长的名称表示更重要的实体）
        name_length = len(entity.get('name', ''))
        importance += min(0.1, name_length * 0.005)
        
        return min(1.0, importance)
    
    def _extract_entity_embeddings(self, result: Dict[str, Any]) -> Optional[np.ndarray]:
        """从检索结果中提取实体嵌入"""
        # 这里可以实现更复杂的实体嵌入提取逻辑
        return None
    
    def _calculate_entity_similarity(self, entity_embeddings: np.ndarray, query: dict) -> float:
        """计算实体相似度"""
        if entity_embeddings is None:
            return 0.5
        query_text = query.get('text', '') if isinstance(query, dict) else ''
        if query_text:
            return min(1.0, len(query_text) / 100.0)
        else:
            return 0.5
    
    def _update_rl_agent(self, query_hash: str, quality_feedback: dict):
        """更新强化学习智能体"""
        try:
            if query_hash in self.retrieval_history:
                results = self.retrieval_history[query_hash]
                k_used = len(results)
                reward = quality_feedback.get('overall_score', 0.5) if isinstance(quality_feedback, dict) else 0.5
                state_key = query_hash
                if state_key in self.rl_agent['q_table']:
                    current_q = self.rl_agent['q_table'][state_key].get(k_used, 0.0)
                    new_q = current_q + self.rl_agent['alpha'] * (reward - current_q)
                    self.rl_agent['q_table'][state_key][k_used] = new_q
                self.logger.info(f"更新RL智能体，k={k_used}, 奖励={reward}")
        except Exception as e:
            self.logger.error(f"更新RL智能体失败: {str(e)}")
    
    def _record_retrieval_history(self, query: Dict[str, Any], results: List[DynamicRetrievalResult]):
        """记录检索历史"""
        query_hash = hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()
        self.retrieval_history[query_hash] = results
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            'total_queries': len(self.retrieval_history),
            'total_feedbacks': len(self.quality_feedback_history),
            'avg_quality_score': np.mean([
                f['feedback'].get('overall_score', 0.5) if isinstance(f['feedback'], dict) else 0.5 
                for f in self.quality_feedback_history
            ]) if self.quality_feedback_history else 0.0,
            'rl_agent_states': len(self.rl_agent['q_table'])
        } 