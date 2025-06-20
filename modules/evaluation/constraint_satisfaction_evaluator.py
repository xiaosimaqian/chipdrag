import logging
from typing import Dict, List, Any, Optional
import json
import os

class ConstraintSatisfactionEvaluator:
    """约束满足率评估器
    
    用于评估布局是否满足各种约束条件，包括：
    1. 设计规则约束（DRC）
    2. 时序约束
    3. 物理约束
    4. 功耗约束
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.constraints = self._load_constraints(config.get('constraint_file'))
        
    def _load_constraints(self, constraint_file: Optional[str]) -> Dict[str, Any]:
        """加载约束规则
        
        Args:
            constraint_file: 约束规则文件路径
            
        Returns:
            约束规则字典
        """
        if constraint_file and os.path.exists(constraint_file):
            try:
                with open(constraint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"加载约束文件失败: {e}")
                return self._get_default_constraints()
        return self._get_default_constraints()
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """获取默认约束规则
        
        Returns:
            默认约束规则字典
        """
        return {
            "drc": {
                "min_spacing": 0.1,  # 最小间距（微米）
                "min_width": 0.1,    # 最小线宽（微米）
                "min_area": 0.1      # 最小面积（平方微米）
            },
            "timing": {
                "max_delay": 10.0,   # 最大延迟（纳秒）
                "min_slack": 0.1     # 最小裕量（纳秒）
            },
            "physical": {
                "max_density": 0.8,  # 最大密度
                "max_congestion": 0.8  # 最大拥塞度
            },
            "power": {
                "max_power": 1.0,    # 最大功耗（瓦特）
                "max_current": 0.1   # 最大电流（安培）
            }
        }
    
    def evaluate_drc_satisfaction(self, layout: Dict[str, Any]) -> float:
        """评估设计规则约束满足率
        
        Args:
            layout: 布局信息字典
            
        Returns:
            设计规则约束满足率（0-100%）
        """
        try:
            violations = 0
            total_rules = 0
            
            # 检查最小间距
            if 'spacing' in layout:
                for spacing in layout['spacing']:
                    total_rules += 1
                    if spacing < self.constraints['drc']['min_spacing']:
                        violations += 1
                        
            # 检查最小线宽
            if 'width' in layout:
                for width in layout['width']:
                    total_rules += 1
                    if width < self.constraints['drc']['min_width']:
                        violations += 1
                        
            # 检查最小面积
            if 'area' in layout:
                for area in layout['area']:
                    total_rules += 1
                    if area < self.constraints['drc']['min_area']:
                        violations += 1
            
            if total_rules == 0:
                return 100.0
                
            satisfaction_rate = (total_rules - violations) / total_rules * 100
            self.logger.info(f"DRC满足率: {satisfaction_rate:.2f}%")
            return satisfaction_rate
            
        except Exception as e:
            self.logger.error(f"DRC评估失败: {e}")
            return 0.0
    
    def evaluate_timing_satisfaction(self, layout: Dict[str, Any]) -> float:
        """评估时序约束满足率
        
        Args:
            layout: 布局信息字典
            
        Returns:
            时序约束满足率（0-100%）
        """
        try:
            violations = 0
            total_paths = 0
            
            if 'timing_paths' in layout:
                for path in layout['timing_paths']:
                    total_paths += 1
                    if path['delay'] > self.constraints['timing']['max_delay']:
                        violations += 1
                    if path['slack'] < self.constraints['timing']['min_slack']:
                        violations += 1
            
            if total_paths == 0:
                return 100.0
                
            satisfaction_rate = (total_paths - violations) / total_paths * 100
            self.logger.info(f"时序约束满足率: {satisfaction_rate:.2f}%")
            return satisfaction_rate
            
        except Exception as e:
            self.logger.error(f"时序约束评估失败: {e}")
            return 0.0
    
    def evaluate_physical_satisfaction(self, layout: Dict[str, Any]) -> float:
        """评估物理约束满足率
        
        Args:
            layout: 布局信息字典
            
        Returns:
            物理约束满足率（0-100%）
        """
        try:
            violations = 0
            total_constraints = 0
            
            # 检查密度约束
            if 'density' in layout:
                total_constraints += 1
                if layout['density'] > self.constraints['physical']['max_density']:
                    violations += 1
                    
            # 检查拥塞度约束
            if 'congestion' in layout:
                total_constraints += 1
                if layout['congestion'] > self.constraints['physical']['max_congestion']:
                    violations += 1
            
            if total_constraints == 0:
                return 100.0
                
            satisfaction_rate = (total_constraints - violations) / total_constraints * 100
            self.logger.info(f"物理约束满足率: {satisfaction_rate:.2f}%")
            return satisfaction_rate
            
        except Exception as e:
            self.logger.error(f"物理约束评估失败: {e}")
            return 0.0
    
    def evaluate_power_satisfaction(self, layout: Dict[str, Any]) -> float:
        """评估功耗约束满足率
        
        Args:
            layout: 布局信息字典
            
        Returns:
            功耗约束满足率（0-100%）
        """
        try:
            violations = 0
            total_constraints = 0
            
            # 检查功耗约束
            if 'power' in layout:
                total_constraints += 1
                if layout['power'] > self.constraints['power']['max_power']:
                    violations += 1
                    
            # 检查电流约束
            if 'current' in layout:
                total_constraints += 1
                if layout['current'] > self.constraints['power']['max_current']:
                    violations += 1
            
            if total_constraints == 0:
                return 100.0
                
            satisfaction_rate = (total_constraints - violations) / total_constraints * 100
            self.logger.info(f"功耗约束满足率: {satisfaction_rate:.2f}%")
            return satisfaction_rate
            
        except Exception as e:
            self.logger.error(f"功耗约束评估失败: {e}")
            return 0.0
    
    def evaluate_constraint_satisfaction(self, layout: Dict[str, Any]) -> Dict[str, float]:
        """评估所有约束的满足率
        
        Args:
            layout: 布局信息字典
            
        Returns:
            包含各类约束满足率的字典
        """
        results = {
            'drc_satisfaction': self.evaluate_drc_satisfaction(layout),
            'timing_satisfaction': self.evaluate_timing_satisfaction(layout),
            'physical_satisfaction': self.evaluate_physical_satisfaction(layout),
            'power_satisfaction': self.evaluate_power_satisfaction(layout)
        }
        
        # 计算总体满足率
        results['overall_satisfaction'] = sum(results.values()) / len(results)
        
        self.logger.info(f"约束满足率评估完成，总体满足率: {results['overall_satisfaction']:.2f}%")
        return results

    def evaluate(self, layout: Dict[str, Any]) -> float:
        """评估布局的约束满足情况
        
        Args:
            layout: 布局信息字典
            
        Returns:
            约束满足率 (0-1)
        """
        try:
            # 获取所有约束的满足率
            satisfaction_results = self.evaluate_constraint_satisfaction(layout)
            
            # 返回总体满足率（转换为0-1范围）
            overall_satisfaction = satisfaction_results.get('overall_satisfaction', 0.0) / 100.0
            
            return overall_satisfaction
            
        except Exception as e:
            self.logger.error(f"约束评估失败: {e}")
            return 0.5  # 返回默认评分 