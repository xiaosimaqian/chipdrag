# chiprag/modules/core/multimodal_retriever.py

import torch
from typing import Dict, List, Any, Optional
import logging
import numpy as np
from ..encoders import ResNetImageEncoder, BertTextEncoder, KGEncoder
from .multimodal_fusion import MultimodalFusion

logger = logging.getLogger(__name__)

class MultimodalRetriever:
    """多模态检索器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化多模态检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化编码器
        self.image_encoder = ResNetImageEncoder(config.get('image_encoder', {}))
        self.text_encoder = BertTextEncoder(config.get('text_encoder', {}))
        self.graph_encoder = KGEncoder(config.get('graph_encoder', {}))
        
        # 设置权重
        self.weights = {
            'image': config.get('image_weight', 0.3),
            'text': config.get('text_weight', 0.4),
            'graph': config.get('graph_weight', 0.3)
        }
        
        # 初始化融合器
        self.fusion = MultimodalFusion(self.weights)
        
    def _validate_encoding(self, encoding: torch.Tensor, modality: str) -> None:
        """验证编码结果
        
        Args:
            encoding: 编码结果
            modality: 模态名称
            
        Raises:
            ValueError: 编码结果无效
        """
        if not isinstance(encoding, torch.Tensor):
            raise ValueError(f"{modality}编码结果必须是torch.Tensor")
        if encoding.numel() == 0:
            raise ValueError(f"{modality}编码结果不能为空")
        if torch.isnan(encoding).any():
            raise ValueError(f"{modality}编码结果包含NaN值")
            
    def _validate_knowledge_base(self, knowledge_base: List[Dict[str, Any]]) -> None:
        """验证知识库
        
        Args:
            knowledge_base: 知识库
            
        Raises:
            ValueError: 知识库无效
        """
        if not isinstance(knowledge_base, list):
            raise ValueError("知识库必须是列表")
        if not knowledge_base:
            raise ValueError("知识库不能为空")
        for item in knowledge_base:
            if not isinstance(item, dict):
                raise ValueError("知识库中的项目必须是字典")
                
    def _compute_similarity(self, query_encoding: torch.Tensor, item_encoding: torch.Tensor, modality: str) -> float:
        """计算相似度
        
        Args:
            query_encoding: 查询编码
            item_encoding: 项目编码
            modality: 模态名称
            
        Returns:
            float: 相似度分数
        """
        # 选择对应的编码器计算相似度
        if modality == 'image':
            return self.image_encoder.compute_similarity(query_encoding, item_encoding)
        elif modality == 'text':
            return self.text_encoder.compute_similarity(query_encoding, item_encoding)
        elif modality == 'graph':
            return self.graph_encoder.compute_similarity(query_encoding, item_encoding)
        else:
            raise ValueError(f"未知的模态: {modality}")
            
    def encode_query(self, query: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """编码查询
        
        Args:
            query: 查询字典，包含文本、图像和知识图谱
            
        Returns:
            Dict[str, torch.Tensor]: 各模态的编码向量
            
        Raises:
            ValueError: 查询格式错误或编码失败
        """
        try:
            encodings = {}
            
            # 编码文本
            if 'text' in query and query['text']:
                text_encoding = self.text_encoder.encode(query['text'])
                self._validate_encoding(text_encoding, 'text')
                encodings['text'] = text_encoding
                
            # 编码图像
            if 'image' in query and query['image']:
                image_encoding = self.image_encoder.encode(query['image'])
                self._validate_encoding(image_encoding, 'image')
                encodings['image'] = image_encoding
                
            # 编码知识图谱
            if 'graph' in query and query['graph']:
                graph_encoding = self.graph_encoder.encode(query['graph'])
                self._validate_encoding(graph_encoding, 'graph')
                encodings['graph'] = graph_encoding
                
            if not encodings:
                raise ValueError("查询中没有任何有效的模态")
                
            return encodings
            
        except Exception as e:
            logger.error(f"查询编码失败: {str(e)}")
            raise
            
    def retrieve(self, query: Dict[str, Any], knowledge_base: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关知识
        
        Args:
            query: 查询字典
            knowledge_base: 知识库
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 检索结果
            
        Raises:
            ValueError: 参数错误或检索失败
        """
        # 类型保护
        if isinstance(query, str):
            query = {'text': query, 'type': 'text'}
        elif not isinstance(query, dict):
            logger.warning(f'查询格式错误，期望字典但得到: {type(query)}')
            query = {'text': str(query), 'type': 'text'}
        
        try:
            # 验证知识库
            self._validate_knowledge_base(knowledge_base)
            
            # 编码查询
            query_encodings = self.encode_query(query)
            
            # 计算相似度
            results = []
            for item in knowledge_base:
                try:
                    item_encodings = self.encode_query(item)
                    
                    # 计算各模态的相似度
                    similarities = {}
                    for modality in query_encodings:
                        if modality in item_encodings:
                            sim = self._compute_similarity(
                                query_encodings[modality],
                                item_encodings[modality],
                                modality
                            )
                            similarities[modality] = sim
                            
                    # 计算加权平均相似度
                    weighted_sim = sum(
                        similarities.get(modality, 0) * self.weights[modality]
                        for modality in self.weights
                    )
                    
                    results.append({
                        'item': item,
                        'similarity': weighted_sim,
                        'modality_similarities': similarities
                    })
                    
                except Exception as e:
                    logger.warning(f"处理知识库项目时出错: {str(e)}")
                    continue
                    
            if not results:
                raise ValueError("没有找到任何匹配的结果")
                
            # 按相似度排序
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []