# chiprag/modules/encoders/base_encoder.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import torch
import logging
import json
import os

logger = logging.getLogger(__name__)

def get_system_config_path():
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../configs/system.json'))
    if os.path.exists(abs_path):
        return abs_path
    alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/system.json'))
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(f"未找到系统配置文件，建议放在: {abs_path}")

class BaseEncoder(ABC):
    """编码器基类"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        """初始化编码器
        
        Args:
            config: 配置字典
            device: 计算设备（可选）
        """
        self.config = config
        
        # 如果提供了设备，使用提供的设备；否则使用系统配置
        if device is not None:
            self.device = device
        else:
            # 读取系统配置
            try:
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
            except Exception as e:
                logger.warning(f"读取系统配置失败: {e}，使用默认设备")
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        # 基本属性
        self.model = None
        self.embedding_dim = config.get('embedding_dim', 768)
        self.batch_size = config.get('batch_size', 32)
        self.encoder_type = config.get('type', 'base')
        self._initialized = False
        
        self._init_model()
        
    @abstractmethod
    def _init_model(self):
        """初始化模型"""
        pass
        
    @abstractmethod
    def encode(self, data: Any) -> torch.Tensor:
        """编码数据
        
        Args:
            data: 输入数据
            
        Returns:
            torch.Tensor: 编码后的向量
        """
        pass
        
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """预处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            Any: 预处理后的数据
        """
        pass
        
    def compute_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """计算两个向量的相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            float: 相似度分数
        """
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
    
    def is_initialized(self) -> bool:
        """检查编码器是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        return self._initialized
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度
        
        Returns:
            int: 嵌入维度
        """
        return self.embedding_dim