# chiprag/modules/core/base_retriever.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import torch
import logging

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def retrieve(self, query: Any, context: Optional[Dict] = None) -> List[Dict]:
        """执行检索
        
        Args:
            query: 查询内容
            context: 上下文信息
            
        Returns:
            List[Dict]: 检索结果
        """
        pass
        
    @abstractmethod
    def compute_similarity(self, query: Any, item: Any) -> float:
        """计算相似度
        
        Args:
            query: 查询内容
            item: 待比较项
            
        Returns:
            float: 相似度分数
        """
        pass
        
    def preprocess(self, data: Any) -> Any:
        """数据预处理
        
        Args:
            data: 输入数据
            
        Returns:
            Any: 预处理后的数据
        """
        return data
        
    def postprocess(self, results: List[Dict]) -> List[Dict]:
        """结果后处理
        
        Args:
            results: 检索结果
            
        Returns:
            List[Dict]: 处理后的结果
        """
        return results