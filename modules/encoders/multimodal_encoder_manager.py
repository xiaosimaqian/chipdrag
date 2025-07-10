"""
多模态编码器管理器
统一管理文本、图像和图编码器
"""
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)

class MultiModalEncoderManager:
    """多模态编码器管理器"""
    
    def __init__(self, config_path: str = "configs/multimodal_config.json"):
        """初始化多模态编码器管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化编码器
        self._initialize_encoders()
        
        logger.info(f"多模态编码器管理器已初始化，设备: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "multimodal_encoders": {
                "text_encoder": {
                    "name": "bert-base-uncased",
                    "type": "text",
                    "embedding_dim": 768,
                    "max_length": 512,
                    "batch_size": 32
                },
                "image_encoder": {
                    "name": "resnet50",
                    "type": "image",
                    "embedding_dim": 2048,
                    "input_size": [224, 224],
                    "batch_size": 16
                },
                "graph_encoder": {
                    "name": "graph_sage",
                    "type": "graph",
                    "embedding_dim": 256,
                    "num_layers": 3,
                    "batch_size": 8
                }
            }
        }
    
    def _initialize_encoders(self):
        """初始化所有编码器"""
        encoder_configs = self.config.get("multimodal_encoders", {})
        
        for encoder_name, config in encoder_configs.items():
            try:
                if config["type"] == "text":
                    self.encoders[encoder_name] = TextEncoder(config, self.device)
                elif config["type"] == "image":
                    self.encoders[encoder_name] = ImageEncoder(config, self.device)
                elif config["type"] == "graph":
                    self.encoders[encoder_name] = GraphEncoder(config, self.device)
                else:
                    logger.warning(f"未知编码器类型: {config['type']}")
                    
                logger.info(f"成功初始化编码器: {encoder_name}")
            except Exception as e:
                logger.error(f"初始化编码器 {encoder_name} 失败: {e}")
    
    def encode_text(self, text: Union[str, List[str]], encoder_name: str = "text_encoder") -> torch.Tensor:
        """编码文本
        
        Args:
            text: 文本或文本列表
            encoder_name: 编码器名称
            
        Returns:
            torch.Tensor: 编码结果
        """
        if encoder_name not in self.encoders:
            raise ValueError(f"编码器 {encoder_name} 未找到")
        
        return self.encoders[encoder_name].encode(text)
    
    def encode_image(self, image: Union[Image.Image, str, List], encoder_name: str = "image_encoder") -> torch.Tensor:
        """编码图像
        
        Args:
            image: 图像、图像路径或图像列表
            encoder_name: 编码器名称
            
        Returns:
            torch.Tensor: 编码结果
        """
        if encoder_name not in self.encoders:
            raise ValueError(f"编码器 {encoder_name} 未找到")
        
        return self.encoders[encoder_name].encode(image)
    
    def encode_graph(self, graph_data: Dict, encoder_name: str = "graph_encoder") -> torch.Tensor:
        """编码图数据
        
        Args:
            graph_data: 图数据
            encoder_name: 编码器名称
            
        Returns:
            torch.Tensor: 编码结果
        """
        if encoder_name not in self.encoders:
            raise ValueError(f"编码器 {encoder_name} 未找到")
        
        return self.encoders[encoder_name].encode(graph_data)
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """获取编码器信息"""
        info = {}
        for name, encoder in self.encoders.items():
            info[name] = {
                "type": encoder.encoder_type,
                "embedding_dim": encoder.embedding_dim,
                "device": str(encoder.device),
                "initialized": encoder.is_initialized()
            }
        return info

class TextEncoder(BaseEncoder):
    """文本编码器 - 基于BERT"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """初始化文本编码器
        
        Args:
            config: 编码器配置
            device: 计算设备
        """
        super().__init__(config, device)
        self.encoder_type = "text"
        self.model_name = config.get("name", "bert-base-uncased")
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 32)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化BERT模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 获取embedding维度
            self.embedding_dim = self.model.config.hidden_size
            self._initialized = True
            
            logger.info(f"文本编码器已初始化: {self.model_name}, 维度: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"初始化文本编码器失败: {e}")
            self._initialized = False
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """编码文本
        
        Args:
            text: 文本或文本列表
            
        Returns:
            torch.Tensor: 编码结果
        """
        if not self._initialized:
            raise RuntimeError("文本编码器未初始化")
        
        if isinstance(text, str):
            text = [text]
        
        # 批量处理
        embeddings = []
        for i in range(0, len(text), self.batch_size):
            batch_text = text[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_text)
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
    
    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """批量编码文本"""
        with torch.no_grad():
            # 分词
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取embeddings
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # 池化
            
            return embeddings.cpu()

class ImageEncoder(BaseEncoder):
    """图像编码器 - 基于ResNet"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """初始化图像编码器
        
        Args:
            config: 编码器配置
            device: 计算设备
        """
        super().__init__(config, device)
        self.encoder_type = "image"
        self.model_name = config.get("name", "resnet50")
        self.input_size = config.get("input_size", [224, 224])
        self.batch_size = config.get("batch_size", 16)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化ResNet模型"""
        try:
            # 加载预训练ResNet50
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # 移除最后的分类层
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            
            # 设置embedding维度
            self.embedding_dim = 2048
            
            # 图像预处理
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self._initialized = True
            logger.info(f"图像编码器已初始化: {self.model_name}, 维度: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"初始化图像编码器失败: {e}")
            self._initialized = False
    
    def encode(self, image: Union[Image.Image, str, List]) -> torch.Tensor:
        """编码图像
        
        Args:
            image: 图像、图像路径或图像列表
            
        Returns:
            torch.Tensor: 编码结果
        """
        if not self._initialized:
            raise RuntimeError("图像编码器未初始化")
        
        if isinstance(image, (str, Image.Image)):
            image = [image]
        
        # 批量处理
        embeddings = []
        for i in range(0, len(image), self.batch_size):
            batch_image = image[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_image)
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
    
    def _encode_batch(self, images: List[Union[Image.Image, str]]) -> torch.Tensor:
        """批量编码图像"""
        with torch.no_grad():
            # 预处理图像
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                processed_images.append(self.transform(img))
            
            # 转为batch tensor
            batch_tensor = torch.stack(processed_images).to(self.device)
            
            # 获取embeddings
            embeddings = self.model(batch_tensor)
            embeddings = embeddings.flatten(start_dim=1)  # 展平
            
            return embeddings.cpu()

class GraphEncoder(BaseEncoder):
    """图编码器 - 基于GraphSAGE"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """初始化图编码器
        
        Args:
            config: 编码器配置
            device: 计算设备
        """
        super().__init__(config, device)
        self.encoder_type = "graph"
        self.model_name = config.get("name", "graph_sage")
        self.embedding_dim = config.get("embedding_dim", 256)
        self.num_layers = config.get("num_layers", 3)
        self.batch_size = config.get("batch_size", 8)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化图模型"""
        try:
            # 简单的图编码器实现
            self.node_embedding = nn.Embedding(10000, self.embedding_dim)
            self.layers = nn.ModuleList([
                nn.Linear(self.embedding_dim, self.embedding_dim)
                for _ in range(self.num_layers)
            ])
            self.activation = nn.ReLU()
            
            self.node_embedding.to(self.device)
            self.layers.to(self.device)
            
            self._initialized = True
            logger.info(f"图编码器已初始化: {self.model_name}, 维度: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"初始化图编码器失败: {e}")
            self._initialized = False
    
    def encode(self, graph_data: Dict) -> torch.Tensor:
        """编码图数据
        
        Args:
            graph_data: 图数据字典
            
        Returns:
            torch.Tensor: 编码结果
        """
        if not self._initialized:
            raise RuntimeError("图编码器未初始化")
        
        # 简单的图编码实现
        node_ids = torch.tensor(graph_data.get("node_ids", [0]), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # 节点嵌入
            x = self.node_embedding(node_ids)
            
            # 多层传播
            for layer in self.layers:
                x = self.activation(layer(x))
            
            # 图级别聚合（简单平均）
            graph_embedding = x.mean(dim=0, keepdim=True)
            
            return graph_embedding.cpu() 