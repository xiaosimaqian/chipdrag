import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class LayoutExperienceRetriever:
    """布局经验检索器，用于从知识库中检索和匹配布局经验"""
    
    def __init__(self, knowledge_base_path: Union[str, Path]):
        """
        初始化布局经验检索器
        
        Args:
            knowledge_base_path: 知识库路径
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.experiences_path = self.knowledge_base_path / "layout_experience" / "layout_experiences.json"
        self.index_path = self.knowledge_base_path / "layout_experience" / "experience_index.json"
        
        # 加载布局经验和索引
        self.experiences = self._load_experiences()
        self.index = self._load_index()
        
    def _load_experiences(self) -> Dict:
        """加载布局经验数据"""
        try:
            with open(self.experiences_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"布局经验文件不存在: {self.experiences_path}")
            return {"experiences": []}
        except json.JSONDecodeError:
            logger.error(f"布局经验文件格式错误: {self.experiences_path}")
            return {"experiences": []}
            
    def _load_index(self) -> Dict:
        """加载布局经验索引"""
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"布局经验索引文件不存在: {self.index_path}")
            return {"index": {}}
        except json.JSONDecodeError:
            logger.error(f"布局经验索引文件格式错误: {self.index_path}")
            return {"index": {}}
            
    def retrieve_by_category(self, category: str) -> List[Dict]:
        """
        按类别检索布局经验
        
        Args:
            category: 经验类别（timing/power/area/congestion）
            
        Returns:
            布局经验列表
        """
        if category not in self.index["index"]["by_category"]:
            logger.warning(f"未找到类别 {category} 的布局经验")
            return []
            
        experience_ids = self.index["index"]["by_category"][category]
        return [exp for exp in self.experiences["experiences"] if exp["id"] in experience_ids]
        
    def retrieve_by_design(self, design: str) -> List[Dict]:
        """
        按设计名称检索布局经验
        
        Args:
            design: 设计名称
            
        Returns:
            布局经验列表
        """
        if design not in self.index["index"]["by_design"]:
            logger.warning(f"未找到设计 {design} 的布局经验")
            return []
            
        experience_ids = self.index["index"]["by_design"][design]
        return [exp for exp in self.experiences["experiences"] if exp["id"] in experience_ids]
        
    def retrieve_by_constraints(self, constraints: Dict) -> List[Dict]:
        """
        按约束条件检索布局经验
        
        Args:
            constraints: 约束条件字典，包含timing/power/area/congestion的目标值
            
        Returns:
            布局经验列表
        """
        matching_experiences = []
        
        for exp in self.experiences["experiences"]:
            exp_constraints = exp["constraints"]
            is_match = True
            
            for key, value in constraints.items():
                if key in exp_constraints:
                    target = exp_constraints[key]["target"]
                    # 简单的数值比较，实际应用中可能需要更复杂的匹配逻辑
                    if float(value) < float(target.replace("ns", "").replace("mW", "").replace("%", "")):
                        is_match = False
                        break
                        
            if is_match:
                matching_experiences.append(exp)
                
        return matching_experiences
        
    def get_optimization_strategies(self, experience_id: str) -> List[Dict]:
        """
        获取指定布局经验的优化策略
        
        Args:
            experience_id: 布局经验ID
            
        Returns:
            优化策略列表
        """
        for exp in self.experiences["experiences"]:
            if exp["id"] == experience_id:
                return exp["optimization_strategies"]
        return []
        
    def get_metrics(self, experience_id: str) -> Optional[Dict]:
        """
        获取指定布局经验的评估指标
        
        Args:
            experience_id: 布局经验ID
            
        Returns:
            评估指标字典
        """
        for exp in self.experiences["experiences"]:
            if exp["id"] == experience_id:
                return exp["metrics"]
        return None 