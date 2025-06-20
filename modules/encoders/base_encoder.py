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
    
    def __init__(self, config: Dict[str, Any]):
        """初始化编码器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
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
                
        self.model = None
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