from typing import Dict, List, Any, Tuple
import numpy as np
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

class MultiModalKnowledgeGraph:
    def __init__(self):
        """初始化多模态知识图谱"""
        self.graph = nx.DiGraph()
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.image_encoder = torch.nn.Sequential(*list(self.image_encoder.children())[:-1])
        self.image_encoder.eval()
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def add_entity(self, entity_id: str, entity_type: str, 
                  text_features: str = None, 
                  image_features: str = None,
                  structured_features: Dict = None) -> None:
        """添加实体到知识图谱
        
        Args:
            entity_id: 实体ID
            entity_type: 实体类型
            text_features: 文本特征
            image_features: 图像特征路径
            structured_features: 结构化特征
        """
        features = {}
        
        # 处理文本特征
        if text_features:
            with torch.no_grad():
                text_embedding = self.text_encoder.encode(text_features)
                features['text_embedding'] = text_embedding
        
        # 处理图像特征
        if image_features and Path(image_features).exists():
            image = Image.open(image_features).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
            with torch.no_grad():
                image_embedding = self.image_encoder(image_tensor).squeeze()
                features['image_embedding'] = image_embedding.numpy()
        
        # 处理结构化特征
        if structured_features:
            features['structured'] = structured_features
            
        self.graph.add_node(entity_id, 
                           type=entity_type,
                           features=features)
    
    def add_relation(self, source_id: str, target_id: str, 
                    relation_type: str, 
                    relation_features: Dict = None) -> None:
        """添加关系到知识图谱
        
        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            relation_type: 关系类型
            relation_features: 关系特征
        """
        self.graph.add_edge(source_id, target_id,
                           type=relation_type,
                           features=relation_features or {})
    
    def retrieve_knowledge(self, query: str, 
                          query_type: str = 'text',
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """检索相关知识
        
        Args:
            query: 查询内容
            query_type: 查询类型 ('text'/'image'/'structured')
            top_k: 返回结果数量
            
        Returns:
            相关实体列表及其相似度分数
        """
        if query_type == 'text':
            with torch.no_grad():
                query_embedding = self.text_encoder.encode(query)
        elif query_type == 'image':
            image = Image.open(query).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
            with torch.no_grad():
                query_embedding = self.image_encoder(image_tensor).squeeze().numpy()
        else:
            query_embedding = query
            
        results = []
        for node_id, node_data in self.graph.nodes(data=True):
            if 'features' not in node_data:
                continue
                
            features = node_data['features']
            if query_type in features:
                similarity = self._calculate_similarity(
                    query_embedding, 
                    features[f'{query_type}_embedding']
                )
                results.append((node_id, similarity))
                
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _calculate_similarity(self, vec1: np.ndarray, 
                            vec2: np.ndarray) -> float:
        """计算两个向量的相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def analyze_feedback(self, feedback: str, 
                        layout_data: Dict) -> Dict[str, Any]:
        """分析用户反馈
        
        Args:
            feedback: 用户反馈文本
            layout_data: 当前布局数据
            
        Returns:
            反馈分析结果
        """
        # 提取反馈类型和严重程度
        feedback_type = self._classify_feedback(feedback)
        severity = self._assess_severity(feedback)
        
        # 检索相关历史经验
        relevant_experiences = self.retrieve_knowledge(feedback)
        
        # 生成优化建议
        suggestions = self._generate_suggestions(
            feedback_type, 
            severity,
            relevant_experiences,
            layout_data
        )
        
        return {
            'type': feedback_type,
            'severity': severity,
            'suggestions': suggestions
        }
    
    def _classify_feedback(self, feedback: str) -> str:
        """对反馈进行分类
        
        Args:
            feedback: 反馈文本
            
        Returns:
            反馈类型
        """
        # 使用文本编码器对反馈进行分类
        with torch.no_grad():
            feedback_embedding = self.text_encoder.encode(feedback)
            
        # 预定义的反馈类型
        feedback_types = {
            'timing': '时序问题',
            'power': '功耗问题',
            'area': '面积问题',
            'congestion': '拥塞问题',
            'other': '其他问题'
        }
        
        # 计算与各类别的相似度
        similarities = {}
        for type_name, type_desc in feedback_types.items():
            type_embedding = self.text_encoder.encode(type_desc)
            similarities[type_name] = self._calculate_similarity(
                feedback_embedding,
                type_embedding
            )
            
        return max(similarities.items(), key=lambda x: x[1])[0]
    
    def _assess_severity(self, feedback: str) -> float:
        """评估反馈的严重程度
        
        Args:
            feedback: 反馈文本
            
        Returns:
            严重程度分数 (0-1)
        """
        # 使用文本编码器提取情感特征
        with torch.no_grad():
            feedback_embedding = self.text_encoder.encode(feedback)
            
        # 预定义的严重程度描述
        severity_levels = {
            'critical': '严重问题',
            'major': '主要问题',
            'minor': '次要问题'
        }
        
        # 计算与各严重程度的相似度
        similarities = {}
        for level, desc in severity_levels.items():
            level_embedding = self.text_encoder.encode(desc)
            similarities[level] = self._calculate_similarity(
                feedback_embedding,
                level_embedding
            )
            
        # 将相似度映射到0-1范围
        max_sim = max(similarities.values())
        min_sim = min(similarities.values())
        if max_sim == min_sim:
            return 0.5
            
        severity = (max_sim - min_sim) / (max_sim - min_sim)
        return severity
    
    def _generate_suggestions(self, 
                            feedback_type: str,
                            severity: float,
                            relevant_experiences: List[Tuple[str, float]],
                            layout_data: Dict) -> List[Dict[str, Any]]:
        """生成优化建议
        
        Args:
            feedback_type: 反馈类型
            severity: 严重程度
            relevant_experiences: 相关历史经验
            layout_data: 当前布局数据
            
        Returns:
            优化建议列表
        """
        suggestions = []
        
        # 根据反馈类型和严重程度生成建议
        if feedback_type == 'timing':
            suggestions.append({
                'type': 'timing_optimization',
                'priority': severity,
                'action': '调整关键路径布局',
                'parameters': {
                    'timing_weight': severity,
                    'congestion_weight': 1 - severity
                }
            })
        elif feedback_type == 'power':
            suggestions.append({
                'type': 'power_optimization',
                'priority': severity,
                'action': '优化电源网络',
                'parameters': {
                    'power_weight': severity,
                    'area_weight': 1 - severity
                }
            })
        elif feedback_type == 'area':
            suggestions.append({
                'type': 'area_optimization',
                'priority': severity,
                'action': '优化单元布局密度',
                'parameters': {
                    'area_weight': severity,
                    'congestion_weight': 1 - severity
                }
            })
        elif feedback_type == 'congestion':
            suggestions.append({
                'type': 'congestion_optimization',
                'priority': severity,
                'action': '调整布线拥塞区域',
                'parameters': {
                    'congestion_weight': severity,
                    'timing_weight': 1 - severity
                }
            })
            
        # 根据历史经验补充建议
        for exp_id, similarity in relevant_experiences:
            if similarity > 0.7:  # 相似度阈值
                exp_data = self.graph.nodes[exp_id]['features']
                if 'structured' in exp_data:
                    suggestions.append({
                        'type': 'experience_based',
                        'priority': similarity,
                        'action': '应用历史优化经验',
                        'parameters': exp_data['structured']
                    })
                    
        return suggestions 