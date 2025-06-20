import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import gc
import os
import json

logger = logging.getLogger(__name__)

def get_system_config_path():
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../configs/system.json'))
    if os.path.exists(abs_path):
        return abs_path
    alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/system.json'))
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(f"未找到系统配置文件，建议放在: {abs_path}")

class KGEncoder(nn.Module):
    """知识图谱编码器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化知识图谱编码器
        
        Args:
            config: 配置字典，包含以下字段：
                - num_entities: 实体数量
                - num_relations: 关系数量
                - embedding_dim: 嵌入维度
                - hidden_dim: 隐藏层维度
                - num_layers: 层数
                - dropout: Dropout比率
                - batch_size: 批处理大小
        """
        super().__init__()
        self.num_entities = config['num_entities']
        self.num_relations = config['num_relations']
        self.embedding_dim = config.get('embedding_dim', 100)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.batch_size = config.get('batch_size', 32)
        
        # 读取系统配置
        system_config_path = get_system_config_path()
        with open(system_config_path, 'r') as f:
            system_config = json.load(f)
            
        device_config = system_config.get('device', {})
        device_type = device_config.get('type', 'cuda')
        device_index = device_config.get('index', 0)
        fallback_to_cpu = device_config.get('fallback_to_cpu', True)
        
        if device_type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_index}')
            logger.info(f"使用GPU设备: {self.device}")
        else:
            if fallback_to_cpu:
                self.device = torch.device('cpu')
                logger.info(f"GPU不可用，使用CPU设备: {self.device}")
            else:
                raise RuntimeError("GPU不可用且不允许回退到CPU")
        
        # 实体嵌入
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        # 关系嵌入
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        
        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # 添加多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )
        
        # 将模型移到指定设备
        self.to(self.device)
        
    def _validate_triples(self, triples: torch.Tensor) -> None:
        """验证三元组
        
        Args:
            triples: 三元组张量
            
        Raises:
            ValueError: 三元组无效
        """
        if not isinstance(triples, torch.Tensor):
            raise ValueError("三元组必须是torch.Tensor")
        if triples.dim() != 2 or triples.size(1) != 3:
            raise ValueError("三元组必须是[batch_size, 3]形状的张量")
        if triples.numel() == 0:
            raise ValueError("三元组不能为空")
        if torch.isnan(triples).any():
            raise ValueError("三元组包含NaN值")
            
    def _validate_entity_id(self, entity_id: int) -> None:
        """验证实体ID
        
        Args:
            entity_id: 实体ID
            
        Raises:
            ValueError: 实体ID无效
        """
        if not isinstance(entity_id, int):
            raise ValueError("实体ID必须是整数")
        if entity_id < 0 or entity_id >= self.num_entities:
            raise ValueError(f"实体ID必须在[0, {self.num_entities-1}]范围内")
            
    def _validate_relation_id(self, relation_id: int) -> None:
        """验证关系ID
        
        Args:
            relation_id: 关系ID
            
        Raises:
            ValueError: 关系ID无效
        """
        if not isinstance(relation_id, int):
            raise ValueError("关系ID必须是整数")
        if relation_id < 0 or relation_id >= self.num_relations:
            raise ValueError(f"关系ID必须在[0, {self.num_relations-1}]范围内")
            
    def _process_batch(self, triples: torch.Tensor, start_idx: int) -> torch.Tensor:
        """处理一批三元组
        
        Args:
            triples: 三元组张量
            start_idx: 起始索引
            
        Returns:
            三元组得分
        """
        end_idx = min(start_idx + self.batch_size, len(triples))
        batch_triples = triples[start_idx:end_idx].to(self.device)
        
        # 获取实体和关系的嵌入
        head_emb = self.entity_embeddings(batch_triples[:, 0])
        relation_emb = self.relation_embeddings(batch_triples[:, 1])
        tail_emb = self.entity_embeddings(batch_triples[:, 2])
        
        # 计算三元组得分
        score = self._compute_score(head_emb, relation_emb, tail_emb)
        
        # 清理内存
        del head_emb, relation_emb, tail_emb, batch_triples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return score
        
    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            triples: 三元组张量 [batch_size, 3]
            
        Returns:
            三元组得分
        """
        self._validate_triples(triples)
        
        # 分批处理
        all_scores = []
        for i in range(0, len(triples), self.batch_size):
            batch_scores = self._process_batch(triples, i)
            all_scores.append(batch_scores)
            
        # 合并所有批次的得分
        return torch.cat(all_scores, dim=0)
        
    def _compute_score(self, head_emb: torch.Tensor, relation_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        """
        计算三元组得分
        
        Args:
            head_emb: 头实体嵌入
            relation_emb: 关系嵌入
            tail_emb: 尾实体嵌入
            
        Returns:
            三元组得分
        """
        # 使用TransE评分函数
        score = head_emb + relation_emb - tail_emb
        return torch.norm(score, p=2, dim=1)
        
    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """
        获取实体嵌入
        
        Args:
            entity_id: 实体ID
            
        Returns:
            实体嵌入向量
        """
        self._validate_entity_id(entity_id)
        return self.entity_embeddings(torch.tensor(entity_id, device=self.device))
        
    def get_relation_embedding(self, relation_id: int) -> torch.Tensor:
        """
        获取关系嵌入
        
        Args:
            relation_id: 关系ID
            
        Returns:
            关系嵌入向量
        """
        self._validate_relation_id(relation_id)
        return self.relation_embeddings(torch.tensor(relation_id, device=self.device))
        
    def compute_entity_similarity(self, entity1_id: int, entity2_id: int) -> float:
        """
        计算两个实体的相似度
        
        Args:
            entity1_id: 第一个实体ID
            entity2_id: 第二个实体ID
            
        Returns:
            相似度分数
        """
        emb1 = self.get_entity_embedding(entity1_id)
        emb2 = self.get_entity_embedding(entity2_id)
        
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        
        # 清理内存
        del emb1, emb2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return similarity.item()
        
    def compute_relation_similarity(self, relation1_id: int, relation2_id: int) -> float:
        """
        计算两个关系的相似度
        
        Args:
            relation1_id: 第一个关系ID
            relation2_id: 第二个关系ID
            
        Returns:
            相似度分数
        """
        emb1 = self.get_relation_embedding(relation1_id)
        emb2 = self.get_relation_embedding(relation2_id)
        
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        
        # 清理内存
        del emb1, emb2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return similarity.item()
        
    def predict_tail(self, head_id: int, relation_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        预测尾实体
        
        Args:
            head_id: 头实体ID
            relation_id: 关系ID
            k: 返回的候选数量
            
        Returns:
            候选尾实体列表及其得分
        """
        self._validate_entity_id(head_id)
        self._validate_relation_id(relation_id)
        
        head_emb = self.get_entity_embedding(head_id)
        relation_emb = self.get_relation_embedding(relation_id)
        
        # 分批计算所有可能的尾实体得分
        all_scores = []
        for i in range(0, self.num_entities, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_entities)
            batch_tails = torch.arange(i, end_idx, device=self.device)
            tail_embs = self.entity_embeddings(batch_tails)
            
            batch_scores = torch.norm(head_emb + relation_emb - tail_embs, p=2, dim=1)
            all_scores.append(batch_scores)
            
            # 清理内存
            del tail_embs, batch_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # 合并所有批次的得分
        scores = torch.cat(all_scores, dim=0)
        
        # 获取得分最高的k个尾实体
        topk_scores, topk_indices = torch.topk(scores, k)
        
        # 清理内存
        del head_emb, relation_emb, scores, all_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return [(idx.item(), score.item()) for idx, score in zip(topk_indices, topk_scores)]
        
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_entities': self.num_entities,
                'num_relations': self.num_relations,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'batch_size': self.batch_size
            }
        }, path)
        logger.info(f"模型已保存到: {path}")
        
    @classmethod
    def load_model(cls, path: str) -> 'KGEncoder':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 {path} 加载")
        return model 

    def encode(self, graph: Dict[str, Any]) -> np.ndarray:
        """
        编码知识图谱
        
        Args:
            graph: 知识图谱字典，包含nodes和edges
            
        Returns:
            np.ndarray: 图编码向量
        """
        try:
            # 提取节点和边
            nodes = graph.get('nodes', [])
            edges = graph.get('edges', [])
            
            # 如果没有节点，返回零向量
            if not nodes:
                return np.zeros(self.embedding_dim)
            
            # 计算节点嵌入的平均值作为图表示
            node_embeddings = []
            for node in nodes:
                if isinstance(node, dict) and 'id' in node:
                    node_id = node['id']
                    if isinstance(node_id, int) and 0 <= node_id < self.num_entities:
                        emb = self.get_entity_embedding(node_id).detach().cpu().numpy()
                        node_embeddings.append(emb)
            
            if node_embeddings:
                return np.mean(node_embeddings, axis=0)
            else:
                return np.zeros(self.embedding_dim)
                
        except Exception as e:
            logger.error(f"图编码失败: {str(e)}")
            return np.zeros(self.embedding_dim)
    
    def compute_similarity(self, graph1: Dict[str, Any], graph2: Dict[str, Any]) -> float:
        """
        计算两个图的相似度
        
        Args:
            graph1: 第一个图
            graph2: 第二个图
            
        Returns:
            float: 相似度分数
        """
        try:
            # 编码两个图
            emb1 = self.encode(graph1)
            emb2 = self.encode(graph2)
            
            # 计算余弦相似度
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算图相似度失败: {str(e)}")
            return 0.5  # 返回默认相似度