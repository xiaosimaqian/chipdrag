import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging
from typing import Dict, Any, List, Union
import gc
import os
import json
import numpy as np

logger = logging.getLogger(__name__)

def get_system_config_path():
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../configs/system.json'))
    if os.path.exists(abs_path):
        return abs_path
    alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/system.json'))
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(f"未找到系统配置文件，建议放在: {abs_path}")

class ResNetImageEncoder(nn.Module):
    """基于ResNet的图像编码器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化ResNet编码器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config or {}
        self.model_name = self.config.get('model_name', 'resnet50')
        self.pretrained = self.config.get('pretrained', True)
        self.feature_dim = self.config.get('feature_dim', 2048)
        self.batch_size = self.config.get('batch_size', 32)
        
        # 从系统配置中读取设备设置
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
        
        # 加载模型
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _load_model(self):
        """加载ResNet模型"""
        if self.model_name == 'resnet50':
            return models.resnet50(pretrained=self.pretrained)
        elif self.model_name == 'resnet34':
            return models.resnet34(pretrained=self.pretrained)
        elif self.model_name == 'resnet18':
            return models.resnet18(pretrained=self.pretrained)
        else:
            raise ValueError(f"不支持的ResNet模型: {self.model_name}")
            
    def _validate_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """验证输入图像
        
        Args:
            image: 输入图像、图像路径或numpy数组
            
        Returns:
            验证后的PIL图像
            
        Raises:
            ValueError: 图像无效
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"图像文件不存在: {image}")
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"无法打开图像文件: {str(e)}")
        elif isinstance(image, np.ndarray):
            # 处理numpy数组
            if image.ndim == 3:
                # 确保是RGB格式
                if image.shape[2] == 3:
                    image = Image.fromarray(image.astype(np.uint8))
                elif image.shape[2] == 1:
                    # 灰度图像转换为RGB
                    image = Image.fromarray(image.squeeze().astype(np.uint8)).convert('RGB')
                else:
                    raise ValueError(f"不支持的图像通道数: {image.shape[2]}")
            elif image.ndim == 2:
                # 灰度图像
                image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
            else:
                raise ValueError(f"不支持的图像维度: {image.ndim}")
                
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL图像、图像文件路径或numpy数组")
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    def _validate_images(self, images: List[Union[str, Image.Image]]) -> List[Image.Image]:
        """验证输入图像列表
        
        Args:
            images: 输入图像列表
            
        Returns:
            验证后的PIL图像列表
            
        Raises:
            ValueError: 图像列表无效
        """
        if not isinstance(images, list):
            raise ValueError("输入必须是列表")
        if not images:
            raise ValueError("输入列表不能为空")
            
        return [self._validate_image(image) for image in images]
        
    def _process_batch(self, images: List[Image.Image], start_idx: int) -> torch.Tensor:
        """处理一批图像
        
        Args:
            images: 图像列表
            start_idx: 起始索引
            
        Returns:
            图像编码向量
        """
        end_idx = min(start_idx + self.batch_size, len(images))
        batch_images = images[start_idx:end_idx]
        
        # 预处理图像
        image_tensors = torch.stack([
            self.transform(image) for image in batch_images
        ])
        
        # 将输入移到模型所在设备
        image_tensors = image_tensors.to(self.device)
        
        # 获取模型输出
        with torch.no_grad():
            features = self.model(image_tensors)
            # 移除batch维度
            features = features.squeeze()
            
        # 清理内存
        del image_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return features
        
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像列表
            
        Returns:
            图像编码向量
        """
        images = self._validate_images(images)
        
        # 分批处理
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch_features = self._process_batch(images, i)
            all_features.append(batch_features)
            
        # 合并所有批次的编码
        return torch.cat(all_features, dim=0)
        
    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        编码单个图像
        
        Args:
            image: PIL图像或文件路径
            
        Returns:
            图像编码向量
        """
        image = self._validate_image(image)
        return self.forward([image])[0]
        
    def encode(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        encode_image的别名方法
        
        Args:
            image: PIL图像或文件路径
            
        Returns:
            图像编码向量
        """
        return self.encode_image(image)
        
    def encode_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        批量编码图像
        
        Args:
            images: PIL图像列表
            
        Returns:
            图像编码向量矩阵
        """
        return self.forward(images)
        
    def compute_similarity(self, image1: Union[str, Image.Image, np.ndarray], image2: Union[str, Image.Image, np.ndarray]) -> float:
        """
        计算两个图像的相似度
        
        Args:
            image1: 第一个图像
            image2: 第二个图像
            
        Returns:
            相似度分数
        """
        try:
            # 编码图像
            emb1 = self.encode_image(image1)
            emb2 = self.encode_image(image2)
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
            
            # 清理内存
            del emb1, emb2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return similarity.item()
            
        except Exception as e:
            logger.error(f"计算图像相似度失败: {str(e)}")
            # 返回默认相似度
            return 0.5
        
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'transform': self.transform,
            'config': self.config
        }, path)
        logger.info(f"模型已保存到: {path}")
        
    @classmethod
    def load_model(cls, path: str) -> 'ResNetImageEncoder':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.transform = checkpoint['transform']
        logger.info(f"模型已从 {path} 加载")
        return model 