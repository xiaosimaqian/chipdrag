# chiprag/modules/core/basic_retriever.py

from typing import Dict, Any, List, Optional
from .base_retriever import BaseRetriever
from ..utils.llm_manager import LLMManager
import torch
import logging

logger = logging.getLogger(__name__)

class BasicRetriever(BaseRetriever):
    """基础检索器，使用 Ollama 进行特征提取和解释生成"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        super().__init__(config)
        
        # 修复LLM配置类型检查
        llm_config = config.get('llm', {})
        if isinstance(llm_config, dict):
            self.llm_manager = LLMManager(llm_config)
        else:
            logger.warning(f"LLM配置格式错误，期望字典但得到: {type(llm_config)}")
            self.llm_manager = LLMManager({})
        
    def retrieve(self, query: str, context: Optional[Dict] = None, knowledge_base: Optional[Any] = None) -> List[Dict]:
        """执行检索
        
        Args:
            query: 查询文本
            context: 上下文信息
            knowledge_base: 知识库实例
            
        Returns:
            List[Dict]: 检索结果列表
        """
        # 类型保护
        if isinstance(query, str):
            query = {'text': query, 'type': 'text'}
        elif not isinstance(query, dict):
            logger.warning(f'查询格式错误，期望字典但得到: {type(query)}')
            query = {'text': str(query), 'type': 'text'}
        # 1. 特征提取
        features = self._extract_features(query, context)
        
        # 2. 相似度计算
        if knowledge_base is not None:
            results = self._compute_similarities_with_kb(features, knowledge_base)
        else:
            results = self._compute_similarities(features)
        
        # 3. 生成解释
        results = self._generate_explanations(query, results)
        
        return results
        
    def compute_similarity(self, query: str, item: str) -> float:
        """计算文本相似度
        
        Args:
            query: 查询文本
            item: 待比较文本
            
        Returns:
            float: 相似度分数
        """
        # 使用 LLM 计算相似度
        return self.llm_manager.compute_similarity(query, item)
        
    def _extract_features(self, query: str, context: Optional[Dict] = None) -> Dict:
        """提取特征
        
        Args:
            query: 查询文本
            context: 上下文信息
            
        Returns:
            Dict: 特征字典
        """
        # 从LLMManager提取特征
        features = self.llm_manager.extract_features(query, context)
        
        # 确保返回正确的格式
        if isinstance(features, dict):
            features['query'] = query
            if 'candidates' not in features:
                # 如果没有候选，创建一些默认候选
                features['candidates'] = [
                    '芯片设计基础知识',
                    '布局优化技术',
                    '时序分析',
                    '功耗优化',
                    '面积优化'
                ]
        else:
            # 如果LLMManager返回的不是字典，创建默认格式
            features = {
                'query': query,
                'candidates': [
                    '芯片设计基础知识',
                    '布局优化技术',
                    '时序分析',
                    '功耗优化',
                    '面积优化'
                ]
            }
        
        return features
        
    def _compute_similarities(self, features: Dict) -> List[Dict]:
        """计算相似度
        
        Args:
            features: 特征字典
            
        Returns:
            List[Dict]: 相似度结果
        """
        # 从特征中提取查询和候选
        query_text = features.get('query', '')
        candidates = features.get('candidates', [])
        
        if not candidates:
            # 如果没有候选，返回默认结果
            return [{
                'id': 'default',
                'content': '默认结果',
                'similarity': 0.0,
                'metadata': {'type': 'default'}
            }]
        
        # 调用LLMManager的compute_similarities方法
        similarities = self.llm_manager.compute_similarities([query_text], candidates)
        
        # 转换为标准格式
        results = []
        for i, similarity in enumerate(similarities):
            if i < len(candidates):
                results.append({
                    'id': f'candidate_{i}',
                    'content': candidates[i],
                    'similarity': similarity,
                    'metadata': {'type': 'candidate'}
                })
        
        return results
        
    def _compute_similarities_with_kb(self, features: Dict, knowledge_base: Any) -> List[Dict]:
        """使用知识库计算相似度
        
        Args:
            features: 特征字典
            knowledge_base: 知识库实例
            
        Returns:
            List[Dict]: 相似度结果
        """
        # 从知识库中检索相似案例
        similar_cases = knowledge_base.get_similar_cases(features)
        
        # 转换为标准格式
        results = []
        for case in similar_cases:
            results.append({
                'id': case.get('id', ''),
                'content': case.get('content', ''),
                'similarity': case.get('similarity', 0.0),
                'metadata': case.get('metadata', {})
            })
            
        return results
        
    def _generate_explanations(self, query: str, results: List[Dict]) -> List[Dict]:
        """生成解释
        
        Args:
            query: 查询文本
            results: 检索结果
            
        Returns:
            List[Dict]: 带解释的结果
        """
        return self.llm_manager.generate_explanations(query, results)