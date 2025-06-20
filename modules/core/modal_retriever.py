# chiprag/modules/core/modal_retriever.py

from .base_retriever import BaseRetriever
from ..encoders import ResNetImageEncoder, BertTextEncoder, KGEncoder
import torch
import logging
from typing import Dict, Any, List, Optional
from modules.encoders.text.bert_encoder import BertTextEncoder as TextEncoder
from modules.encoders.image.resnet_encoder import ResNetImageEncoder as ImageEncoder
from modules.encoders.graph.kg_encoder import KGEncoder
import numpy as np

logger = logging.getLogger(__name__)

class ModalRetriever(BaseRetriever):
    """多模态检索器，支持文本、图像、知识图谱等模态"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化多模态检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.encoders = {
            'text': BertTextEncoder(config.get('text_encoder', {})),
            'image': ResNetImageEncoder(config.get('image_encoder', {})),
            'graph': KGEncoder({
                'num_entities': 2,
                'num_relations': 10,
                **config.get('graph_encoder', {})
            })
        }
        self._init_components()
        
        # 设置权重
        self.weights = config.get('modal_weights', {
            'text': 0.4,
            'image': 0.3,
            'graph': 0.3
        })
        
    def retrieve(self, query: Dict[str, Any], context: Optional[Dict] = None, knowledge_base: Optional[Any] = None) -> List[Dict]:
        """执行多模态检索
        
        Args:
            query: 查询字典
            context: 上下文信息
            knowledge_base: 知识库实例
            
        Returns:
            List[Dict]: 检索结果
        """
        # 类型保护
        if isinstance(query, str):
            query = {'text': query, 'type': 'text'}
        elif not isinstance(query, dict):
            logger.warning(f'查询格式错误，期望字典但得到: {type(query)}')
            query = {'text': str(query), 'type': 'text'}
        # 1. 编码查询
        query_encodings = self._encode_query(query)
        
        # 2. 计算相似度
        results = self._compute_similarities(query_encodings, context, knowledge_base)
        
        # 3. 结果融合
        return self._fuse_results(results)
        
    def compute_similarity(self, query: Dict[str, Any], item: Dict[str, Any]) -> float:
        """计算多模态相似度
        
        Args:
            query: 查询字典
            item: 待比较项
            
        Returns:
            float: 相似度分数
        """
        # 计算各模态的相似度
        similarities = {}
        for modality, encoder in self.encoders.items():
            if modality in query:
                try:
                    # 处理不同模态的数据格式
                    if modality == 'text':
                        query_data = query[modality]
                        # 安全获取item中的文本内容
                        if isinstance(item, dict):
                            item_data = item.get('content', '')
                        else:
                            item_data = str(item)
                    elif modality == 'image':
                        query_data = query[modality]
                        # 从item中提取图像数据
                        if isinstance(item, dict) and 'features' in item:
                            item_data = item['features'].get('image', [])
                            if isinstance(item_data, list) and len(item_data) > 0:
                                item_data = np.array(item_data)
                            else:
                                item_data = query_data  # 使用查询数据作为默认值
                        else:
                            item_data = query_data  # 使用查询数据作为默认值
                    elif modality == 'graph':
                        query_data = query[modality]
                        if isinstance(item, dict) and 'features' in item:
                            item_data = item['features'].get('graph', [])
                        else:
                            item_data = query_data  # 使用查询数据作为默认值
                    else:
                        query_data = query[modality]
                        if isinstance(item, dict) and modality in item:
                            item_data = item[modality]
                        else:
                            item_data = query_data  # 使用查询数据作为默认值
                    
                    similarities[modality] = encoder.compute_similarity(query_data, item_data)
                except Exception as e:
                    logger.error(f"计算{modality}模态相似度失败: {str(e)}")
                    similarities[modality] = 0.0
                
        # 加权融合
        return self._weighted_fusion(similarities)
        
    def _encode_query(self, query: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """编码查询
        
        Args:
            query: 查询字典
            
        Returns:
            Dict[str, torch.Tensor]: 编码结果
        """
        encodings = {}
        for modality, encoder in self.encoders.items():
            if modality in query:
                encodings[modality] = encoder.encode(query[modality])
        return encodings
        
    def _compute_similarities(self, query_encodings: Dict[str, torch.Tensor],
                            context: Optional[Dict] = None,
                            knowledge_base: Optional[Any] = None) -> List[Dict]:
        """计算相似度
        
        Args:
            query_encodings: 查询编码
            context: 上下文信息
            knowledge_base: 知识库实例
            
        Returns:
            List[Dict]: 相似度结果
        """
        results = []
        
        # 获取知识库数据
        kb_items = []
        if knowledge_base is not None:
            if hasattr(knowledge_base, 'get_similar_cases'):
                # 如果是KnowledgeBase对象
                kb_items = knowledge_base.get_similar_cases({}, top_k=10)
            elif isinstance(knowledge_base, list):
                # 如果是列表
                kb_items = knowledge_base
            elif isinstance(knowledge_base, dict) and 'knowledge_base' in knowledge_base:
                # 如果是包含knowledge_base的字典
                kb_items = knowledge_base['knowledge_base']
        elif context and 'knowledge_base' in context:
            kb_items = context['knowledge_base']
        
        # 如果没有知识库数据，使用默认数据
        if not kb_items:
            kb_items = [
                {
                    'id': 'default_item',
                    'content': '默认知识库项目',
                    'features': {
                        'text': np.random.rand(768).tolist(),
                        'image': np.random.rand(2048).tolist(),
                        'graph': np.random.rand(512).tolist()
                    }
                }
            ]
        
        for item in kb_items:
            try:
                # 编码知识项
                item_encodings = {}
                for modality, encoder in self.encoders.items():
                    if modality in item and modality in query_encodings:
                        if modality == 'text':
                            item_encodings[modality] = encoder.encode(item.get('content', ''))
                        elif modality == 'image':
                            # 处理图像数据，支持numpy数组
                            image_data = item.get('features', {}).get('image', [])
                            if isinstance(image_data, list) and len(image_data) > 0:
                                # 如果是特征列表，转换为numpy数组
                                image_data = np.array(image_data)
                            item_encodings[modality] = encoder.encode(image_data)
                        elif modality == 'graph':
                            item_encodings[modality] = encoder.encode(item.get('features', {}).get('graph', []))
                
                # 计算各模态相似度
                similarities = {}
                for modality in query_encodings:
                    if modality in item_encodings:
                        similarities[modality] = torch.nn.functional.cosine_similarity(
                            query_encodings[modality],
                            item_encodings[modality]
                        ).item()
                
                # 计算加权平均相似度
                weighted_sim = sum(
                    similarities.get(modality, 0) * self.weights[modality]
                    for modality in self.weights
                ) / sum(self.weights.values())
                
                results.append({
                    'id': item.get('id', 'unknown'),
                    'content': item.get('content', ''),
                    'similarity': weighted_sim,
                    'modality_similarities': similarities,
                    'metadata': item.get('metadata', {})
                })
                
            except Exception as e:
                logger.error(f"计算相似度失败: {str(e)}")
                continue
                
        return results
        
    def _fuse_results(self, results: List[Dict]) -> List[Dict]:
        """融合多模态结果
        
        Args:
            results: 检索结果
            
        Returns:
            List[Dict]: 融合后的结果
        """
        if not results:
            return []
            
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 去重
        seen = set()
        unique_results = []
        for result in results:
            item_id = result['id']
            if item_id not in seen:
                seen.add(item_id)
                unique_results.append(result)
                
        # 获取top-k结果
        top_k = self.config.get('top_k', 5)
        return unique_results[:top_k]
        
    def _weighted_fusion(self, similarities: Dict[str, float]) -> float:
        """加权融合相似度
        
        Args:
            similarities: 各模态相似度
            
        Returns:
            float: 融合后的相似度
        """
        return sum(
            similarities[modality] * self.weights[modality]
            for modality in similarities
        ) / sum(self.weights.values())

    def _init_components(self):
        """初始化组件"""
        self.encoders = {
            'text': TextEncoder(config=self.config.get('text_encoder', {})),
            'image': ImageEncoder(config=self.config.get('image_encoder', {})),
            'graph': KGEncoder(config={
                'num_entities': 2,
                'num_relations': 10,
                **self.config.get('graph_encoder', {})
            })
        }