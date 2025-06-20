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
            self.evaluator = MultiObjectiveEvaluator()
            
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
            raw_results = self.knowledge_base.query(
                text=query.get('text', ''),
                top_k=self.dynamic_k_range[1]  # 使用最大k值
            )
            
            # 转换为动态检索结果
            results = []
            for result in raw_results:
                dynamic_result = DynamicRetrievalResult(
                    knowledge=result.get('content', {}),
                    relevance_score=result.get('similarity', 0.0),
                    granularity_level=result.get('granularity', 'global'),
                    source=result.get('source', 'unknown'),
                    entity_embeddings=self._extract_entity_embeddings(result),
                    retrieval_count=1
                )
                results.append(dynamic_result)
            
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
        """动态重排序"""
        try:
            # 1. 计算历史质量反馈权重
            quality_weights = self._calculate_quality_weights(results)
            
            # 2. 计算查询相似度权重
            similarity_weights = self._calculate_similarity_weights(results, query)
            
            # 3. 计算实体相关性权重
            entity_weights = self._calculate_entity_weights(results, query)
            
            # 4. 综合重排序
            for result in results:
                result.relevance_score = (
                    quality_weights.get(result.source, 1.0) * 0.4 +
                    similarity_weights.get(result.source, 1.0) * 0.4 +
                    entity_weights.get(result.source, 1.0) * 0.2
                )
            
            # 5. 排序并返回top-k
            reranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)[:k]
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"动态重排序失败: {str(e)}")
            return results[:k]
    
    def _enhance_with_entities(self, 
                              results: List[DynamicRetrievalResult], 
                              design_info: Dict[str, Any]) -> List[DynamicRetrievalResult]:
        """实体增强处理"""
        try:
            enhanced_results = []
            
            for result in results:
                # 1. 提取实体信息
                entities = self._extract_entities(result.knowledge, design_info)
                
                # 2. 压缩实体嵌入
                compressed_embeddings = self._compress_entity_embeddings(entities)
                
                # 3. 更新结果
                result.entity_embeddings = compressed_embeddings
                enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"实体增强处理失败: {str(e)}")
            return results
    
    def update_with_feedback(self, 
                           query_hash: str, 
                           layout_result: Dict[str, Any],
                           quality_feedback: LayoutQualityFeedback):
        """基于布局质量反馈更新重排序策略
        
        Args:
            query_hash: 查询哈希
            layout_result: 布局结果
            quality_feedback: 质量反馈
        """
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
                    result.quality_feedback = quality_feedback.overall_score
            
            self.logger.info(f"基于反馈更新重排序策略，质量分数: {quality_feedback.overall_score}")
            
        except Exception as e:
            self.logger.error(f"更新反馈失败: {str(e)}")
    
    def _extract_state_features(self, 
                               query: Dict[str, Any], 
                               design_info: Dict[str, Any],
                               results: List[DynamicRetrievalResult]) -> Dict[str, Any]:
        """提取状态特征"""
        return {
            'query_complexity': len(query.get('text', '')),
            'design_size': len(design_info.get('components', [])),
            'result_count': len(results),
            'avg_relevance': np.mean([r.relevance_score for r in results]) if results else 0.0,
            'design_type': design_info.get('type', 'unknown')
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
                    f['feedback'].overall_score 
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
    
    def _extract_entities(self, knowledge: Dict[str, Any], design_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取实体信息"""
        entities = []
        
        # 从知识中提取组件实体
        if 'components' in knowledge:
            for comp in knowledge['components']:
                entities.append({
                    'type': 'component',
                    'name': comp.get('name', ''),
                    'category': comp.get('type', ''),
                    'properties': comp.get('properties', {})
                })
        
        # 从设计信息中提取约束实体
        if 'constraints' in design_info:
            for constraint in design_info['constraints']:
                entities.append({
                    'type': 'constraint',
                    'name': constraint.get('name', ''),
                    'category': constraint.get('type', ''),
                    'properties': constraint.get('properties', {})
                })
        
        return entities
    
    def _compress_entity_embeddings(self, entities: List[Dict[str, Any]]) -> np.ndarray:
        """压缩实体嵌入"""
        if not entities:
            return np.zeros(self.compressed_entity_dim)
        
        # 简单的实体嵌入压缩（实际应用中可以使用更复杂的压缩方法）
        entity_features = []
        for entity in entities:
            # 将实体信息转换为特征向量
            feature = [
                hash(entity.get('name', '')) % 1000 / 1000.0,
                hash(entity.get('category', '')) % 1000 / 1000.0,
                len(entity.get('properties', {}))
            ]
            entity_features.append(feature)
        
        # 平均池化
        if entity_features:
            avg_features = np.mean(entity_features, axis=0)
            # 压缩到指定维度
            compressed = np.zeros(self.compressed_entity_dim)
            compressed[:min(len(avg_features), self.compressed_entity_dim)] = \
                avg_features[:self.compressed_entity_dim]
            return compressed
        else:
            return np.zeros(self.compressed_entity_dim)
    
    def _extract_entity_embeddings(self, result: Dict[str, Any]) -> Optional[np.ndarray]:
        """从检索结果中提取实体嵌入"""
        # 这里可以实现更复杂的实体嵌入提取逻辑
        return None
    
    def _calculate_entity_similarity(self, 
                                   entity_embeddings: np.ndarray, 
                                   query: Dict[str, Any]) -> float:
        """计算实体相似度"""
        # 简单的相似度计算（实际应用中可以使用更复杂的相似度度量）
        if entity_embeddings is None:
            return 0.5
        
        # 基于查询文本的简单相似度
        query_text = query.get('text', '')
        if query_text:
            # 简单的文本相似度计算
            return min(1.0, len(query_text) / 100.0)
        else:
            return 0.5
    
    def _update_rl_agent(self, query_hash: str, quality_feedback: LayoutQualityFeedback):
        """更新强化学习智能体"""
        try:
            if query_hash in self.retrieval_history:
                # 获取当前状态和动作
                results = self.retrieval_history[query_hash]
                k_used = len(results)
                
                # 计算奖励（基于质量反馈）
                reward = quality_feedback.overall_score
                
                # 更新Q值（这里简化处理，实际应用中需要更复杂的Q-learning更新）
                state_key = query_hash  # 简化状态表示
                if state_key in self.rl_agent['q_table']:
                    current_q = self.rl_agent['q_table'][state_key].get(k_used, 0.0)
                    # 简单的Q值更新
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
                f['feedback'].overall_score for f in self.quality_feedback_history
            ]) if self.quality_feedback_history else 0.0,
            'rl_agent_states': len(self.rl_agent['q_table'])
        } 