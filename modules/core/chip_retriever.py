# chiprag/modules/core/chip_retriever.py

from .base_retriever import BaseRetriever
from .granularity_retriever import GranularityRetriever
from .modal_retriever import ModalRetriever
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Any
import os
from pathlib import Path
import json
from ..utils.config_loader import ConfigLoader
import torch
import numpy as np
from modules.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# 更新配置路径
CONFIG_DIR = Path(__file__).parent.parent.parent / 'configs'

def load_config(config_name: str) -> Dict:
    """加载配置文件
    
    Args:
        config_name: 配置文件名
        
    Returns:
        Dict: 配置字典
    """
    config_path = CONFIG_DIR / config_name
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class ChipRetriever(BaseRetriever):
    """芯片设计检索器，整合多粒度和多模态检索"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # 加载配置
        config_loader = ConfigLoader()
        
        # 加载各组件配置
        self.config = config or {}
        self.config.update({
            'granularity': config_loader.load_config('retriever/granularity.json'),
            'modal': config_loader.load_config('retriever/modal.json'),
            'llm': config_loader.load_config('llm/ollama.json'),
            'knowledge': config_loader.load_config('knowledge/transfer.json')
        })
        
        super().__init__(self.config)
        
        # 1. 缓存管理
        self.cache = {
            'query': {},  # 查询缓存
            'results': {},  # 结果缓存
            'embeddings': {}  # 嵌入缓存
        }
        
        # 2. 批处理配置
        self.batch_size = self.config.get('batch_size', 32)
        
        # 3. 并行处理配置
        self.num_workers = self.config.get('num_workers', 4)
        
        # 初始化多粒度检索器
        self.granularity_retriever = GranularityRetriever(
            self.config.get('granularity', {})
        )
        
        # 初始化多模态检索器
        self.modal_retriever = ModalRetriever(
            self.config.get('modal', {})
        )
        
        # 获取LLM配置并添加调试日志
        llm_config = {}
        if isinstance(self.config.get('llm'), dict):
            llm_config = self.config['llm'].get('models', {}).get('default', {})
        else:
            logger.warning(f"LLM配置格式错误，期望字典但得到: {type(self.config.get('llm'))}")
            # 使用默认配置
            llm_config = {
                "name": "llama2:latest",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        
        logger.info(f"LLM配置: {llm_config}")
        self.llm_manager = LLMManager(llm_config)
        
    def retrieve(self, query: Dict[str, Any], context: Optional[Dict] = None, knowledge_base: Optional[Dict] = None) -> List[Dict]:
        """优化的检索实现
        
        Args:
            query: 查询信息
            context: 上下文信息
            knowledge_base: 知识库信息
            
        Returns:
            检索结果列表
        """
        try:
            # 确保查询是字典格式
            if isinstance(query, str):
                query = {'text': query, 'type': 'text'}
            elif not isinstance(query, dict):
                logger.warning(f"查询格式错误，期望字典但得到: {type(query)}")
                query = {'text': str(query), 'type': 'text'}
            
            # 1. 检查缓存
            cache_key = self._generate_cache_key(query, context)
            if cache_key in self.cache['results']:
                return self.cache['results'][cache_key]
            
            # 2. 准备知识库数据
            if knowledge_base is None:
                # 如果没有传入知识库，尝试从配置中加载
                try:
                    from ..knowledge.knowledge_base import KnowledgeBase
                    kb = KnowledgeBase(self.config.get('knowledge_base', {}))
                    knowledge_base = kb
                except Exception as e:
                    logger.warning(f"无法加载知识库: {str(e)}")
                    knowledge_base = None
            
            # 3. 并行处理
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 并行执行多粒度检索
                granularity_future = executor.submit(
                    self.granularity_retriever.retrieve,
                    query,
                    context,
                    knowledge_base
                )
                
                # 并行执行多模态检索
                modal_future = executor.submit(
                    self.modal_retriever.retrieve,
                    query,
                    context,
                    knowledge_base
                )
                
                # 获取结果
                granularity_results = granularity_future.result()
                modal_results = modal_future.result()
            
            # 4. 批处理融合
            results = self._batch_fuse_results(
                granularity_results,
                modal_results
            )
            
            # 5. 结果增强
            enhanced_results = self._enhance_results(results, query)
            
            # 6. 更新缓存
            self.cache['results'][cache_key] = enhanced_results
            
            # 7. 确保返回非空结果
            if not enhanced_results:
                # 生成默认结果
                default_result = {
                    'id': 'default',
                    'content': '默认结果：未找到相关匹配',
                    'score': 0.0,
                    'features': [],
                    'metadata': {
                        'type': 'default',
                        'description': '默认结果'
                    }
                }
                enhanced_results = [default_result]
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            # 返回默认结果
            return [{
                'id': 'default',
                'content': '检索失败：系统错误',
                'score': 0.0,
                'features': [],
                'metadata': {
                    'type': 'error',
                    'description': '检索失败'
                }
            }]
        
    def _batch_fuse_results(self, granularity_results: List[Dict],
                           modal_results: List[Dict]) -> List[Dict]:
        """批处理结果融合"""
        # 1. 准备批处理数据
        batches = self._prepare_batches(
            granularity_results,
            modal_results
        )
        
        # 2. 并行处理批次
        fused_results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._process_batch, batch)
                for batch in batches
            ]
            
            # 收集结果
            for future in as_completed(futures):
                fused_results.extend(future.result())
                
        return fused_results
        
    def _prepare_batches(self, granularity_results: List[Dict],
                        modal_results: List[Dict]) -> List[List[Dict]]:
        """准备批处理数据"""
        # 合并结果
        all_results = granularity_results + modal_results
        
        # 分批
        return [
            all_results[i:i + self.batch_size]
            for i in range(0, len(all_results), self.batch_size)
        ]
        
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """处理单个批次"""
        # 1. 标准化 - 修复参数传递
        normalized = self._normalize_results(batch, [])
        
        # 2. 去重
        deduplicated = self._deduplicate_results(normalized)
        
        # 3. 排序
        sorted_results = self._sort_results(deduplicated)
        
        # 4. 增强 - 传递空查询作为默认值
        return self._enhance_results(sorted_results, {})
        
    def _generate_cache_key(self, query: Dict, context: Optional[Dict]) -> str:
        """生成缓存键"""
        import hashlib
        import json
        
        # 序列化查询和上下文
        query_str = json.dumps(query, sort_keys=True, default=str)
        context_str = json.dumps(context or {}, sort_keys=True, default=str)
        
        # 生成哈希
        content = f"{query_str}:{context_str}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def compute_similarity(self, query: Dict[str, Any], item: Dict[str, Any]) -> float:
        """计算综合相似度
        
        Args:
            query: 查询字典
            item: 待比较项
            
        Returns:
            float: 相似度分数
        """
        # 计算多粒度相似度
        granularity_sim = self.granularity_retriever.compute_similarity(
            query=query,
            item=item
        )
        
        # 计算多模态相似度
        modal_sim = self.modal_retriever.compute_similarity(
            query=query,
            item=item
        )
        
        # 加权融合
        weights = self.config.get('fusion_weights', {
            'granularity': 0.5,
            'modal': 0.5
        })
        
        return (
            weights['granularity'] * granularity_sim +
            weights['modal'] * modal_sim
        )
        
    def _fuse_results(self, granularity_results: List[Dict],
                     modal_results: List[Dict]) -> List[Dict]:
        """融合检索结果
        
        Args:
            granularity_results: 多粒度检索结果
            modal_results: 多模态检索结果
            
        Returns:
            List[Dict]: 融合后的结果
        """
        # 1. 结果标准化
        normalized_results = self._normalize_results(
            granularity_results, modal_results
        )
        
        # 2. 结果去重
        deduplicated_results = self._deduplicate_results(
            normalized_results
        )
        
        # 3. 结果排序
        sorted_results = self._sort_results(
            deduplicated_results
        )
        
        # 4. 结果增强
        enhanced_results = self._enhance_results(
            sorted_results
        )
        
        return enhanced_results
        
    def _normalize_results(self, granularity_results: List[Dict],
                          modal_results: List[Dict]) -> List[Dict]:
        """标准化结果格式"""
        normalized = []
        
        # 处理多粒度结果
        for result in granularity_results:
            normalized.append({
                'id': result.get('id', ''),
                'content': result.get('content', ''),
                'metadata': {
                    'source': 'granularity',
                    'level': result.get('level', 'unknown'),
                    'similarity': result.get('similarity', 0.0),
                    'timestamp': result.get('timestamp', ''),
                    'type': result.get('type', 'unknown')
                },
                'features': {
                    'text': result.get('text', ''),
                    'layout': result.get('layout', {}),
                    'constraints': result.get('constraints', [])
                }
            })
            
        # 处理多模态结果
        for result in modal_results:
            normalized.append({
                'id': result.get('id', ''),
                'content': result.get('content', ''),
                'metadata': {
                    'source': 'modal',
                    'modalities': result.get('modalities', []),
                    'similarity': result.get('similarity', 0.0),
                    'timestamp': result.get('timestamp', ''),
                    'type': result.get('type', 'unknown')
                },
                'features': {
                    'text': result.get('text', ''),
                    'image': result.get('image', {}),
                    'graph': result.get('graph', {})
                }
            })
            
        return normalized
        
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重结果"""
        seen = set()
        unique_results = []
        
        for result in results:
            # 使用内容和特征组合作为唯一标识
            key = (
                result['content'],
                str(result['features'])
            )
            
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
                
        return unique_results
        
    def _sort_results(self, results: List[Dict]) -> List[Dict]:
        """排序结果"""
        # 计算综合得分
        for result in results:
            # 基础相似度
            base_score = result['metadata']['similarity']
            
            # 时间衰减因子
            time_factor = self._calculate_time_factor(
                result['metadata']['timestamp']
            )
            
            # 来源权重
            source_weight = self.config.get('source_weights', {
                'granularity': 0.6,
                'modal': 0.4
            })[result['metadata']['source']]
            
            # 计算综合得分
            result['score'] = (
                base_score * time_factor * source_weight
            )
            
        # 按综合得分排序
        return sorted(results, key=lambda x: x['score'], reverse=True)
        
    def _enhance_results(self, results: List[Dict], query: Dict[str, Any]) -> List[Dict]:
        """增强检索结果
        
        Args:
            results: 检索结果列表
            query: 查询信息
            
        Returns:
            增强后的结果列表
        """
        try:
            # 1. 确保结果非空
            if not results:
                return [{
                    'id': 'default',
                    'score': 0.0,
                    'features': [],
                    'metadata': {
                        'type': 'default',
                        'description': '默认结果'
                    }
                }]
            
            # 2. 计算时间因子
            for result in results:
                if 'timestamp' in result:
                    result['time_factor'] = self._calculate_time_factor(result['timestamp'])
            
            # 3. 计算置信度
            for result in results:
                result['confidence'] = self._calculate_confidence(result)
            
            # 4. 计算特征匹配度
            for result in results:
                if 'features' in result:
                    result['feature_match'] = self._calculate_feature_match(result['features'])
            
            # 5. 计算元数据可靠性
            for result in results:
                if 'metadata' in result:
                    result['metadata_reliability'] = self._calculate_metadata_reliability(result['metadata'])
            
            # 6. 使用LLM增强结果
            enhanced_results = []
            for result in results:
                try:
                    enhanced = self.llm_manager.enhance_result(result, query)
                    enhanced_results.append(enhanced)
                except Exception as e:
                    logger.warning(f"LLM增强失败: {str(e)}")
                    enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"结果增强失败: {str(e)}")
            return [{
                'id': 'default',
                'score': 0.0,
                'features': [],
                'metadata': {
                    'type': 'default',
                    'description': '默认结果'
                }
            }]
        
    def _calculate_time_factor(self, timestamp: str) -> float:
        """计算时间衰减因子"""
        # 实现时间衰减逻辑
        pass
        
    def _calculate_confidence(self, result: Dict) -> float:
        """计算结果置信度"""
        confidence = 0.0
        
        # 1. 相似度分数
        if 'score' in result:
            confidence += result['score'] * 0.4
            
        # 2. 特征匹配度
        if 'features' in result:
            feature_match = self._calculate_feature_match(result['features'])
            confidence += feature_match * 0.3
            
        # 3. 元数据可靠性
        if 'metadata' in result:
            metadata_reliability = self._calculate_metadata_reliability(result['metadata'])
            confidence += metadata_reliability * 0.3
            
        return min(confidence, 1.0)
        
    def _calculate_feature_match(self, features: Dict) -> float:
        """计算特征匹配度"""
        # 简单实现，可以根据需要调整
        return 0.8
        
    def _calculate_metadata_reliability(self, metadata: Dict) -> float:
        """计算元数据可靠性"""
        # 简单实现，可以根据需要调整
        return 0.9