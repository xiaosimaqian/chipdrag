# chiprag/modules/core/modal_retriever.py

from .base_retriever import BaseRetriever
from ..encoders.multimodal_encoder_manager import MultiModalEncoderManager
import torch
import logging
from typing import Dict, Any, List, Optional
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
        
        # 使用新的多模态编码器管理器
        self.encoder_manager = MultiModalEncoderManager()
        
        # 设置权重
        self.weights = config.get('modal_weights', {
            'text': 0.4,
            'image': 0.3,
            'graph': 0.3
        })
        
        logger.info("多模态检索器初始化完成")
        
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
    
    def _encode_query(self, query: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """编码查询
        
        Args:
            query: 查询字典
            
        Returns:
            Dict[str, torch.Tensor]: 各模态的编码向量
        """
        encodings = {}
        
        # 编码文本
        if 'text' in query and query['text']:
            try:
                text_encoding = self.encoder_manager.encode_text(query['text'])
                encodings['text'] = text_encoding
                logger.debug(f"文本编码完成，维度: {text_encoding.shape}")
            except Exception as e:
                logger.warning(f"文本编码失败: {e}")
        
        # 编码图像
        if 'image' in query and query['image']:
            try:
                image_encoding = self.encoder_manager.encode_image(query['image'])
                encodings['image'] = image_encoding
                logger.debug(f"图像编码完成，维度: {image_encoding.shape}")
            except Exception as e:
                logger.warning(f"图像编码失败: {e}")
        
        # 编码图数据
        if 'graph' in query and query['graph']:
            try:
                graph_encoding = self.encoder_manager.encode_graph(query['graph'])
                encodings['graph'] = graph_encoding
                logger.debug(f"图编码完成，维度: {graph_encoding.shape}")
            except Exception as e:
                logger.warning(f"图编码失败: {e}")
        
        if not encodings:
            logger.warning("没有成功编码任何模态")
            # 返回默认编码
            encodings['text'] = torch.zeros(1, 768)
        
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
            List[Dict]: 相似度结果列表
        """
        results = []
        
        # 获取知识库项目
        kb_items = []
        if knowledge_base is not None:
            if hasattr(knowledge_base, 'get_all_cases'):
                kb_items = knowledge_base.get_all_cases()
            elif isinstance(knowledge_base, list):
                kb_items = knowledge_base
            else:
                logger.warning("知识库格式不支持")
        
        if not kb_items:
            logger.warning("知识库为空，返回空结果")
            return []
        
        logger.info(f"计算 {len(kb_items)} 个知识库项目的相似度")
        
        for item in kb_items:
            try:
                # 编码知识项
                item_encodings = {}
                
                # 编码文本内容
                if 'text' in query_encodings and 'content' in item:
                    try:
                        item_encodings['text'] = self.encoder_manager.encode_text(item['content'])
                    except Exception as e:
                        logger.debug(f"知识项文本编码失败: {e}")
                
                # 编码图像特征
                if 'image' in query_encodings and 'features' in item:
                    image_data = item['features'].get('image', [])
                    if image_data:
                        try:
                            if isinstance(image_data, list):
                                image_data = np.array(image_data)
                            item_encodings['image'] = self.encoder_manager.encode_image(image_data)
                        except Exception as e:
                            logger.debug(f"知识项图像编码失败: {e}")
                
                # 编码图特征
                if 'graph' in query_encodings and 'features' in item:
                    graph_data = item['features'].get('graph', {})
                    if graph_data:
                        try:
                            item_encodings['graph'] = self.encoder_manager.encode_graph(graph_data)
                        except Exception as e:
                            logger.debug(f"知识项图编码失败: {e}")
                
                # 计算各模态相似度
                similarities = {}
                for modality in query_encodings:
                    if modality in item_encodings:
                        try:
                            similarities[modality] = torch.nn.functional.cosine_similarity(
                                query_encodings[modality],
                                item_encodings[modality]
                            ).item()
                        except Exception as e:
                            logger.debug(f"计算{modality}相似度失败: {e}")
                
                # 计算加权平均相似度
                if similarities:
                    weighted_sim = sum(
                        similarities.get(modality, 0) * self.weights[modality]
                        for modality in self.weights
                    ) / sum(self.weights.values())
                    
                    results.append({
                        'item': item,
                        'similarity': weighted_sim,
                        'modality_similarities': similarities
                    })
                
            except Exception as e:
                logger.warning(f"处理知识库项目时出错: {e}")
                continue
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"相似度计算完成，返回 {len(results)} 个结果")
        return results
    
    def _fuse_results(self, results: List[Dict]) -> List[Dict]:
        """融合多模态结果
        
        Args:
            results: 原始结果列表
            
        Returns:
            List[Dict]: 融合后的结果列表
        """
        if not results:
            return []
        
        # 简单的结果融合（可以扩展为更复杂的融合策略）
        fused_results = []
        
        for result in results:
            fused_result = {
                'content': result['item'],
                'score': result['similarity'],
                'modalities': result.get('modality_similarities', {}),
                'metadata': {
                    'fusion_method': 'weighted_average',
                    'weights': self.weights
                }
            }
            fused_results.append(fused_result)
        
        return fused_results