# chiprag/modules/utils/config_loader.py

import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_dir: str = None):
        """初始化配置加载器
        
        Args:
            config_dir: 配置目录路径
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent.parent / 'configs'
        else:
            self.config_dir = Path(config_dir)
            
        # 缓存配置
        self._config_cache = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_name: 配置文件名
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        # 检查缓存
        if config_name in self._config_cache:
            return self._config_cache[config_name]
            
        # 加载配置
        config_path = self.config_dir / config_name
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 更新缓存
            self._config_cache[config_name] = config
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {config_name}, 错误: {str(e)}")
            return {}
            
    def get_config(self, config_name: str, key: str, default: Any = None) -> Any:
        """获取配置项
        
        Args:
            config_name: 配置文件名
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        config = self.load_config(config_name)
        return config.get(key, default)
        
    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """重新加载配置
        
        Args:
            config_name: 配置文件名
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        # 清除缓存
        if config_name in self._config_cache:
            del self._config_cache[config_name]
            
        # 重新加载
        return self.load_config(config_name)