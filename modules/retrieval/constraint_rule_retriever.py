import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class ConstraintRuleRetriever:
    """约束规则检索器，用于从知识库中检索和匹配约束规则"""
    
    def __init__(self, knowledge_base_path: Union[str, Path]):
        """
        初始化约束规则检索器
        
        Args:
            knowledge_base_path: 知识库路径
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.rules_path = self.knowledge_base_path / "constraints" / "constraint_rules.json"
        self.index_path = self.knowledge_base_path / "constraints" / "constraint_index.json"
        
        # 加载约束规则和索引
        self.rules = self._load_rules()
        self.index = self._load_index()
        
    def _load_rules(self) -> Dict:
        """加载约束规则数据"""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"约束规则文件不存在: {self.rules_path}")
            return {"rules": []}
        except json.JSONDecodeError:
            logger.error(f"约束规则文件格式错误: {self.rules_path}")
            return {"rules": []}
            
    def _load_index(self) -> Dict:
        """加载约束规则索引"""
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"约束规则索引文件不存在: {self.index_path}")
            return {"index": {}}
        except json.JSONDecodeError:
            logger.error(f"约束规则索引文件格式错误: {self.index_path}")
            return {"index": {}}
            
    def retrieve_by_category(self, category: str) -> List[Dict]:
        """
        按类别检索约束规则
        
        Args:
            category: 规则类别（timing/power/area/congestion）
            
        Returns:
            约束规则列表
        """
        if category not in self.index["index"]["by_category"]:
            logger.warning(f"未找到类别 {category} 的约束规则")
            return []
            
        rule_ids = self.index["index"]["by_category"][category]
        return [rule for rule in self.rules["rules"] if rule["id"] in rule_ids]
        
    def retrieve_by_parameter(self, parameter: str) -> List[Dict]:
        """
        按参数名称检索约束规则
        
        Args:
            parameter: 参数名称
            
        Returns:
            约束规则列表
        """
        if parameter not in self.index["index"]["by_parameter"]:
            logger.warning(f"未找到参数 {parameter} 的约束规则")
            return []
            
        rule_ids = self.index["index"]["by_parameter"][parameter]
        return [rule for rule in self.rules["rules"] if rule["id"] in rule_ids]
        
    def retrieve_by_metric(self, metric: str) -> List[Dict]:
        """
        按评估指标检索约束规则
        
        Args:
            metric: 评估指标名称
            
        Returns:
            约束规则列表
        """
        if metric not in self.index["index"]["by_metric"]:
            logger.warning(f"未找到指标 {metric} 的约束规则")
            return []
            
        rule_ids = self.index["index"]["by_metric"][metric]
        return [rule for rule in self.rules["rules"] if rule["id"] in rule_ids]
        
    def get_check_method(self, rule_id: str) -> Optional[Dict]:
        """
        获取指定约束规则的检查方法
        
        Args:
            rule_id: 约束规则ID
            
        Returns:
            检查方法字典
        """
        for rule in self.rules["rules"]:
            if rule["id"] == rule_id:
                return rule["check_method"]
        return None
        
    def get_optimization_guidelines(self, rule_id: str) -> List[Dict]:
        """
        获取指定约束规则的优化指南
        
        Args:
            rule_id: 约束规则ID
            
        Returns:
            优化指南列表
        """
        for rule in self.rules["rules"]:
            if rule["id"] == rule_id:
                return rule["optimization_guidelines"]
        return []
        
    def get_parameters(self, rule_id: str) -> Optional[Dict]:
        """
        获取指定约束规则的参数定义
        
        Args:
            rule_id: 约束规则ID
            
        Returns:
            参数定义字典
        """
        for rule in self.rules["rules"]:
            if rule["id"] == rule_id:
                return rule["parameters"]
        return None 