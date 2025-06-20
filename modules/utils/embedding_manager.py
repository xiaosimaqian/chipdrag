import numpy as np
import torch
from typing import Dict, List
import logging
import json
import requests
import datetime
import os

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """布局数据向量化管理器"""
    
    def __init__(self, config: Dict = None):
        """初始化向量化管理器
        
        Args:
            config: 配置字典，包含model_name和api_base
        """
        if config is None:
            config = {}
        # 使用 Ollama 的配置
        self.model_name = config.get('model_name', 'bge-m3')  # Ollama 模型名称
        self.api_base = config.get('api_base', 'http://localhost:11434')  # Ollama API 地址
        
        # 添加本地模型支持
        self.use_local_model = config.get('use_local_model', True)
        self.local_model_path = config.get('local_model_path', 'models/bert')
        self.embedding_dim = config.get('embedding_dim', 768)
        
        # 初始化本地模型（如果需要）
        self.local_model = None
        if self.use_local_model:
            self._init_local_model()
        
    def _init_local_model(self):
        """初始化本地模型"""
        try:
            from transformers import AutoTokenizer, AutoModel
            if os.path.exists(self.local_model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
                self.model = AutoModel.from_pretrained(self.local_model_path)
                self.model.eval()
                logger.info(f"成功加载本地模型: {self.local_model_path}")
            else:
                logger.warning(f"本地模型路径不存在: {self.local_model_path}")
                self.local_model = None
        except Exception as e:
            logger.warning(f"本地模型初始化失败: {str(e)}")
            self.local_model = None
        
    def embed_layout(self, layout: Dict) -> np.ndarray:
        """将布局数据转换为向量
        
        Args:
            layout: 布局数据
            
        Returns:
            布局向量
        """
        try:
            # 提取布局特征
            features = self._extract_layout_features(layout)
            
            # 将特征转换为文本
            text = self._features_to_text(features)
            
            # 尝试使用Ollama
            try:
                response = requests.post(
                    f"{self.api_base}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=5  # 添加超时
                )
                response.raise_for_status()
                vector = response.json()["embedding"]
                return np.array(vector)
            except Exception as e:
                logger.warning(f"Ollama服务不可用: {str(e)}")
                
                # 尝试使用本地模型
                if hasattr(self, 'model') and self.model is not None:
                    return self._embed_with_local_model(text)
                else:
                    # 返回模拟向量
                    return self._generate_mock_embedding(text)
                    
        except Exception as e:
            logger.error(f"布局向量化失败: {str(e)}")
            # 返回模拟向量作为后备
            return self._generate_mock_embedding("fallback")
            
    def _embed_with_local_model(self, text: str) -> np.ndarray:
        """使用本地模型进行向量化"""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                outputs = self.model(**inputs)
                # 使用[CLS]标记的输出作为句子表示
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                return embedding.flatten()
        except Exception as e:
            logger.error(f"本地模型向量化失败: {str(e)}")
            return self._generate_mock_embedding(text)
            
    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """生成模拟向量"""
        # 基于文本内容生成一致的模拟向量
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # 将哈希值转换为数值
        seed = int(hash_hex[:8], 16)
        np.random.seed(seed)
        
        # 生成模拟向量
        mock_vector = np.random.normal(0, 1, self.embedding_dim)
        # 归一化
        mock_vector = mock_vector / np.linalg.norm(mock_vector)
        
        return mock_vector
        
    def _extract_layout_features(self, layout: Dict) -> Dict:
        """提取布局特征
        
        Args:
            layout: 布局数据
            
        Returns:
            布局特征
        """
        features = {
            'components': [],
            'nets': [],
            'hierarchy': layout.get('hierarchy', {}),
            'die_area': layout.get('die_area', {})
        }
        
        # 提取组件特征
        for comp in layout.get('components', []):
            comp_feature = {
                'name': comp.get('name', ''),
                'type': comp.get('type', ''),
                'position': comp.get('position', {}),
                'size': comp.get('size', {})
            }
            features['components'].append(comp_feature)
            
        # 提取网络特征
        for net in layout.get('nets', []):
            net_feature = {
                'name': net.get('name', ''),
                'pins': net.get('pins', [])
            }
            features['nets'].append(net_feature)
            
        return features
        
    def _features_to_text(self, features: Dict) -> str:
        """将特征转换为文本
        
        Args:
            features: 布局特征
            
        Returns:
            文本描述
        """
        text_parts = []
        
        # 添加层次结构信息
        if features['hierarchy']:
            text_parts.append(f"Hierarchy: {json.dumps(features['hierarchy'])}")
            
        # 添加芯片面积信息
        if features['die_area']:
            text_parts.append(f"Die Area: {json.dumps(features['die_area'])}")
            
        # 添加组件信息
        if features['components']:
            text_parts.append("Components:")
            for comp in features['components']:
                text_parts.append(f"- {json.dumps(comp)}")
                
        # 添加网络信息
        if features['nets']:
            text_parts.append("Nets:")
            for net in features['nets']:
                text_parts.append(f"- {json.dumps(net)}")
                
        return "\n".join(text_parts)
        
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        try:
            # 计算余弦相似度
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            raise

    def optimize_representation(self, knowledge: Dict, context: Dict) -> Dict:
        """优化知识表示
        
        Args:
            knowledge: 知识数据
            context: 上下文数据
            
        Returns:
            优化后的知识
        """
        try:
            # 1. 提取知识特征
            knowledge_features = self._extract_knowledge_features(knowledge)
            
            # 2. 提取上下文特征
            context_features = self._extract_context_features(context)
            
            # 3. 合并特征
            merged_features = self._merge_features(knowledge_features, context_features)
            
            # 4. 生成优化后的知识
            optimized_knowledge = self._generate_optimized_knowledge(merged_features)
            
            # 5. 添加增强建议
            optimized_knowledge['enhanced_suggestions'] = []
            for section in ['area_utilization', 'routing_quality', 'timing_performance', 'power_distribution']:
                if section in optimized_knowledge and 'enhanced_suggestions' in optimized_knowledge[section]:
                    optimized_knowledge['enhanced_suggestions'].extend(optimized_knowledge[section]['enhanced_suggestions'])
            
            # 6. 添加元数据
            optimized_knowledge['metadata'] = {
                'source': 'embedding_optimization',
                'timestamp': str(datetime.datetime.now()),
                'version': '1.0.0'
            }
            
            return optimized_knowledge
            
        except Exception as e:
            logger.error(f"知识表示优化失败: {str(e)}")
            return knowledge
            
    def _extract_knowledge_features(self, knowledge: Dict) -> Dict:
        """提取知识特征
        
        Args:
            knowledge: 知识数据
            
        Returns:
            知识特征
        """
        features = {
            'area_utilization': knowledge.get('area_utilization', {}),
            'routing_quality': knowledge.get('routing_quality', {}),
            'timing_performance': knowledge.get('timing_performance', {}),
            'power_distribution': knowledge.get('power_distribution', {}),
            'optimization_metrics': knowledge.get('optimization_metrics', {})
        }
        return features
        
    def _extract_context_features(self, context: Dict) -> Dict:
        """提取上下文特征
        
        Args:
            context: 上下文数据
            
        Returns:
            上下文特征
        """
        features = {
            'area': context.get('area', {}),
            'components': context.get('components', []),
            'nets': context.get('nets', []),
            'constraints': context.get('constraints', {})
        }
        return features
        
    def _merge_features(self, knowledge_features: Dict, context_features: Dict) -> Dict:
        """合并特征
        
        Args:
            knowledge_features: 知识特征
            context_features: 上下文特征
            
        Returns:
            合并后的特征
        """
        merged = {
            'knowledge': knowledge_features,
            'context': context_features
        }
        return merged
        
    def _generate_optimized_knowledge(self, features: Dict) -> Dict:
        """生成优化后的知识
        
        Args:
            features: 合并后的特征
            
        Returns:
            优化后的知识
        """
        knowledge = features['knowledge']
        
        # 添加增强建议
        knowledge['enhanced_suggestions'] = []
        for section in ['area_utilization', 'routing_quality', 'timing_performance', 'power_distribution']:
            if section in knowledge and 'enhanced_suggestions' in knowledge[section]:
                knowledge['enhanced_suggestions'].extend(knowledge[section]['enhanced_suggestions'])
        
        # 添加元数据
        knowledge['metadata'] = {
            'source': 'embedding_optimization',
            'timestamp': str(datetime.datetime.now()),
            'version': '1.0.0'
        }
        
        return knowledge

    def embed_text(self, text: str) -> np.ndarray:
        """将文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            文本向量
        """
        try:
            # 调用 Ollama API
            response = requests.post(
                f"{self.api_base}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            vector = response.json()["embedding"]
            
            return np.array(vector)
        except Exception as e:
            logger.error(f"文本向量化失败: {str(e)}")
            raise 