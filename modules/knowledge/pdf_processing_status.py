# chiprag/modules/knowledge/pdf_processing_status.py

"""
PDF处理状态管理模块
"""

import json
import os
from pathlib import Path
import hashlib
from datetime import datetime
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PDFProcessingStatus:
    """PDF处理状态管理类"""
    
    def __init__(self, cache_dir: str):
        """初始化PDF处理状态管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.status_file = self.cache_dir / 'pdf_processing_status.json'
        self.status = self._load_status()
        
    def _load_status(self) -> Dict:
        """加载处理状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载PDF处理状态失败: {str(e)}")
                return {}
        return {}
        
    def _save_status(self):
        """保存处理状态"""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"保存PDF处理状态失败: {str(e)}")
            
    def is_processed(self, pdf_path: str) -> bool:
        """检查PDF是否已处理
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            bool: 是否已处理
        """
        pdf_hash = self._get_file_hash(pdf_path)
        return pdf_hash in self.status
        
    def mark_as_processed(self, pdf_path: str, metadata: Optional[Dict] = None):
        """标记PDF为已处理
        
        Args:
            pdf_path: PDF文件路径
            metadata: 额外的元数据
        """
        pdf_hash = self._get_file_hash(pdf_path)
        self.status[pdf_hash] = {
            'path': pdf_path,
            'processed_time': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_status()
        
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件哈希值
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希值失败: {str(e)}")
            return ""
            
    def get_processing_info(self, pdf_path: str) -> Optional[Dict]:
        """获取PDF处理信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Optional[Dict]: 处理信息
        """
        pdf_hash = self._get_file_hash(pdf_path)
        return self.status.get(pdf_hash)
        
    def clear_status(self):
        """清除所有处理状态"""
        self.status = {}
        self._save_status()