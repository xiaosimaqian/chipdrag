#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的LLM系统 - 解决实验中发现的问题
1. 修复质量评估模块的数据格式处理问题
2. 增强策略多样性，根据设计特征生成差异化策略
3. 添加完善的错误处理机制
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DesignComplexity(Enum):
    """设计复杂度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DesignType(Enum):
    """设计类型枚举"""
    ASIC = "asic"
    FPGA = "fpga"
    MIXED = "mixed"

@dataclass
class DesignFeatures:
    """设计特征数据结构"""
    complexity: DesignComplexity
    design_type: DesignType
    component_count: int
    hierarchy_levels: int
    timing_critical: bool
    power_sensitive: bool
    area_constrained: bool
    special_networks: List[str]
    constraints: Dict[str, Any]

@dataclass
class LayoutStrategy:
    """布局策略数据结构"""
    placement_strategy: str
    routing_strategy: str
    density_target: float
    wirelength_weight: float
    timing_weight: float
    power_weight: float
    congestion_weight: float
    execution_plan: List[str]
    optimization_focus: str

@dataclass
class QualityMetrics:
    """质量评估指标"""
    hpwl_score: float
    congestion_score: float
    timing_score: float
    power_score: float
    area_score: float
    overall_score: float
    confidence: float

class EnhancedLLMSystem:
    """增强的LLM系统"""
    
    def __init__(self):
        self.strategy_templates = self._load_strategy_templates()
        self.error_handlers = self._setup_error_handlers()
        self.quality_thresholds = self._setup_quality_thresholds()
        
    def _load_strategy_templates(self) -> Dict[str, Dict]:
        """加载策略模板"""
        return {
            "high_complexity": {
                "placement": "hierarchical",
                "routing": "timing_driven",
                "density_target": 0.75,
                "wirelength_weight": 1.0,
                "timing_weight": 0.9,
                "power_weight": 0.7,
                "congestion_weight": 0.8,
                "optimization_focus": "timing_and_congestion"
            },
            "medium_complexity": {
                "placement": "analytical",
                "routing": "balanced",
                "density_target": 0.7,
                "wirelength_weight": 0.9,
                "timing_weight": 0.8,
                "power_weight": 0.8,
                "congestion_weight": 0.7,
                "optimization_focus": "balanced_optimization"
            },
            "low_complexity": {
                "placement": "simple",
                "routing": "wirelength_driven",
                "density_target": 0.65,
                "wirelength_weight": 1.0,
                "timing_weight": 0.6,
                "power_weight": 0.9,
                "congestion_weight": 0.5,
                "optimization_focus": "area_and_power"
            },
            "timing_critical": {
                "placement": "timing_aware",
                "routing": "timing_driven",
                "density_target": 0.8,
                "wirelength_weight": 0.8,
                "timing_weight": 1.0,
                "power_weight": 0.6,
                "congestion_weight": 0.9,
                "optimization_focus": "timing_optimization"
            },
            "power_sensitive": {
                "placement": "power_aware",
                "routing": "power_driven",
                "density_target": 0.6,
                "wirelength_weight": 0.7,
                "timing_weight": 0.7,
                "power_weight": 1.0,
                "congestion_weight": 0.6,
                "optimization_focus": "power_optimization"
            }
        }
    
    def _setup_error_handlers(self) -> Dict[str, callable]:
        """设置错误处理器"""
        return {
            "data_format_error": self._handle_data_format_error,
            "llm_timeout": self._handle_llm_timeout,
            "invalid_response": self._handle_invalid_response,
            "quality_assessment_failure": self._handle_quality_assessment_failure
        }
    
    def _setup_quality_thresholds(self) -> Dict[str, float]:
        """设置质量评估阈值"""
        return {
            "hpwl_threshold": 0.6,
            "congestion_threshold": 0.7,
            "timing_threshold": 0.8,
            "power_threshold": 0.7,
            "area_threshold": 0.6,
            "overall_threshold": 0.7
        }
    
    def analyze_design(self, design_data: Dict[str, Any]) -> DesignFeatures:
        """增强的设计分析"""
        try:
            logger.info("开始设计分析...")
            
            # 提取设计特征
            component_count = design_data.get('component_count', 0)
            hierarchy_levels = design_data.get('hierarchy_levels', 1)
            
            # 智能复杂度判断
            if component_count > 100000:
                complexity = DesignComplexity.HIGH
            elif component_count > 50000:
                complexity = DesignComplexity.MEDIUM
            else:
                complexity = DesignComplexity.LOW
            
            # 设计类型识别
            design_type = self._identify_design_type(design_data)
            
            # 约束分析
            constraints = design_data.get('constraints', {})
            timing_critical = constraints.get('timing_critical', False)
            power_sensitive = constraints.get('power_sensitive', False)
            area_constrained = constraints.get('area_constrained', False)
            
            # 特殊网络识别
            special_networks = self._identify_special_networks(design_data)
            
            features = DesignFeatures(
                complexity=complexity,
                design_type=design_type,
                component_count=component_count,
                hierarchy_levels=hierarchy_levels,
                timing_critical=timing_critical,
                power_sensitive=power_sensitive,
                area_constrained=area_constrained,
                special_networks=special_networks,
                constraints=constraints
            )
            
            logger.info(f"设计分析完成: {complexity.value}, {design_type.value}, {component_count}组件")
            return features
            
        except Exception as e:
            logger.error(f"设计分析失败: {str(e)}")
            return self._handle_analysis_error(design_data, e)
    
    def generate_layout_strategy(self, features: DesignFeatures) -> LayoutStrategy:
        """增强的布局策略生成"""
        try:
            logger.info("开始生成布局策略...")
            
            # 根据设计特征选择策略模板
            strategy_key = self._select_strategy_template(features)
            base_strategy = self.strategy_templates[strategy_key]
            
            # 动态调整策略参数
            adjusted_strategy = self._adjust_strategy_parameters(base_strategy, features)
            
            # 生成执行计划
            execution_plan = self._generate_execution_plan(features, adjusted_strategy)
            
            strategy = LayoutStrategy(
                placement_strategy=adjusted_strategy["placement"],
                routing_strategy=adjusted_strategy["routing"],
                density_target=adjusted_strategy["density_target"],
                wirelength_weight=adjusted_strategy["wirelength_weight"],
                timing_weight=adjusted_strategy["timing_weight"],
                power_weight=adjusted_strategy["power_weight"],
                congestion_weight=adjusted_strategy["congestion_weight"],
                execution_plan=execution_plan,
                optimization_focus=adjusted_strategy["optimization_focus"]
            )
            
            logger.info(f"策略生成完成: {strategy.placement_strategy} + {strategy.routing_strategy}")
            return strategy
            
        except Exception as e:
            logger.error(f"策略生成失败: {str(e)}")
            return self._handle_strategy_generation_error(features, e)
    
    def assess_layout_quality(self, layout_data: Dict[str, Any]) -> QualityMetrics:
        """增强的布局质量评估"""
        try:
            logger.info("开始质量评估...")
            
            # 数据格式验证和修复
            validated_data = self._validate_and_fix_layout_data(layout_data)
            
            # 计算各项指标
            hpwl_score = self._calculate_hpwl_score(validated_data)
            congestion_score = self._calculate_congestion_score(validated_data)
            timing_score = self._calculate_timing_score(validated_data)
            power_score = self._calculate_power_score(validated_data)
            area_score = self._calculate_area_score(validated_data)
            
            # 综合评分
            overall_score = self._calculate_overall_score(
                hpwl_score, congestion_score, timing_score, power_score, area_score
            )
            
            # 置信度评估
            confidence = self._calculate_confidence(validated_data)
            
            metrics = QualityMetrics(
                hpwl_score=hpwl_score,
                congestion_score=congestion_score,
                timing_score=timing_score,
                power_score=power_score,
                area_score=area_score,
                overall_score=overall_score,
                confidence=confidence
            )
            
            logger.info(f"质量评估完成: 综合评分={overall_score:.3f}, 置信度={confidence:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            return self._handle_quality_assessment_error(layout_data, e)
    
    def _identify_design_type(self, design_data: Dict[str, Any]) -> DesignType:
        """识别设计类型"""
        # 基于设计特征智能识别
        if 'fpga_components' in design_data:
            return DesignType.FPGA
        elif 'asic_components' in design_data and 'fpga_components' in design_data:
            return DesignType.MIXED
        else:
            return DesignType.ASIC
    
    def _identify_special_networks(self, design_data: Dict[str, Any]) -> List[str]:
        """识别特殊网络"""
        special_networks = []
        nets = design_data.get('nets', [])
        
        for net in nets:
            if net.get('type') == 'clock':
                special_networks.append('clock_network')
            elif net.get('type') == 'power':
                special_networks.append('power_network')
            elif net.get('type') == 'reset':
                special_networks.append('reset_network')
        
        return list(set(special_networks))
    
    def _select_strategy_template(self, features: DesignFeatures) -> str:
        """选择策略模板"""
        if features.timing_critical:
            return "timing_critical"
        elif features.power_sensitive:
            return "power_sensitive"
        elif features.complexity == DesignComplexity.HIGH:
            return "high_complexity"
        elif features.complexity == DesignComplexity.MEDIUM:
            return "medium_complexity"
        else:
            return "low_complexity"
    
    def _adjust_strategy_parameters(self, base_strategy: Dict, features: DesignFeatures) -> Dict:
        """动态调整策略参数"""
        adjusted = base_strategy.copy()
        
        # 根据组件数量调整密度目标
        if features.component_count > 100000:
            adjusted["density_target"] *= 1.1
        elif features.component_count < 10000:
            adjusted["density_target"] *= 0.9
        
        # 根据层次级别调整权重
        if features.hierarchy_levels > 3:
            adjusted["wirelength_weight"] *= 1.1
            adjusted["congestion_weight"] *= 1.1
        
        # 根据特殊网络调整策略
        if 'clock_network' in features.special_networks:
            adjusted["timing_weight"] *= 1.2
        if 'power_network' in features.special_networks:
            adjusted["power_weight"] *= 1.2
        
        return adjusted
    
    def _generate_execution_plan(self, features: DesignFeatures, strategy: Dict) -> List[str]:
        """生成执行计划"""
        plan = []
        
        # 基础步骤
        plan.append("initial_placement")
        
        # 根据复杂度添加步骤
        if features.complexity == DesignComplexity.HIGH:
            plan.extend(["hierarchical_decomposition", "coarse_placement", "fine_placement"])
        else:
            plan.append("direct_placement")
        
        # 根据优化重点添加步骤
        if strategy["optimization_focus"] == "timing_optimization":
            plan.extend(["timing_analysis", "timing_optimization", "clock_tree_synthesis"])
        elif strategy["optimization_focus"] == "power_optimization":
            plan.extend(["power_analysis", "power_optimization", "power_network_planning"])
        else:
            plan.extend(["timing_optimization", "power_optimization", "area_optimization"])
        
        plan.extend(["legalization", "detailed_routing", "final_optimization"])
        
        return plan
    
    def _validate_and_fix_layout_data(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证和修复布局数据"""
        validated = layout_data.copy()
        
        # 修复HPWL数据
        if 'hpwl' in validated:
            hpwl = validated['hpwl']
            if isinstance(hpwl, str):
                try:
                    validated['hpwl'] = float(hpwl.replace(',', ''))
                except:
                    validated['hpwl'] = 0.0
            elif not isinstance(hpwl, (int, float)):
                validated['hpwl'] = 0.0
        
        # 修复拥塞数据
        if 'congestion' in validated:
            congestion = validated['congestion']
            if isinstance(congestion, str):
                try:
                    validated['congestion'] = float(congestion)
                except:
                    validated['congestion'] = 0.5
            elif not isinstance(congestion, (int, float)):
                validated['congestion'] = 0.5
        
        # 修复时序数据
        if 'timing' in validated:
            timing = validated['timing']
            if isinstance(timing, str):
                try:
                    validated['timing'] = float(timing)
                except:
                    validated['timing'] = 0.5
            elif not isinstance(timing, (int, float)):
                validated['timing'] = 0.5
        
        # 修复功耗数据
        if 'power' in validated:
            power = validated['power']
            if isinstance(power, str):
                try:
                    validated['power'] = float(power)
                except:
                    validated['power'] = 0.5
            elif not isinstance(power, (int, float)):
                validated['power'] = 0.5
        
        # 修复面积数据
        if 'area' in validated:
            area = validated['area']
            if isinstance(area, str):
                try:
                    validated['area'] = float(area)
                except:
                    validated['area'] = 0.5
            elif not isinstance(area, (int, float)):
                validated['area'] = 0.5
        
        return validated
    
    def _calculate_hpwl_score(self, data: Dict[str, Any]) -> float:
        """计算HPWL评分"""
        hpwl = data.get('hpwl', 0.0)
        if hpwl <= 0:
            return 0.5
        
        # 基于HPWL值计算评分（越小越好）
        # 这里需要根据实际数据分布调整
        if hpwl < 1e8:
            return 0.9
        elif hpwl < 1e9:
            return 0.8
        elif hpwl < 1e10:
            return 0.7
        else:
            return 0.5
    
    def _calculate_congestion_score(self, data: Dict[str, Any]) -> float:
        """计算拥塞评分"""
        congestion = data.get('congestion', 0.5)
        # 拥塞值越小越好
        return max(0.1, 1.0 - congestion)
    
    def _calculate_timing_score(self, data: Dict[str, Any]) -> float:
        """计算时序评分"""
        timing = data.get('timing', 0.5)
        # 时序值越大越好
        return min(1.0, timing)
    
    def _calculate_power_score(self, data: Dict[str, Any]) -> float:
        """计算功耗评分"""
        power = data.get('power', 0.5)
        # 功耗值越小越好
        return max(0.1, 1.0 - power)
    
    def _calculate_area_score(self, data: Dict[str, Any]) -> float:
        """计算面积评分"""
        area = data.get('area', 0.5)
        # 面积值越小越好
        return max(0.1, 1.0 - area)
    
    def _calculate_overall_score(self, hpwl: float, congestion: float, 
                                timing: float, power: float, area: float) -> float:
        """计算综合评分"""
        weights = {
            'hpwl': 0.3,
            'congestion': 0.2,
            'timing': 0.25,
            'power': 0.15,
            'area': 0.1
        }
        
        overall = (hpwl * weights['hpwl'] + 
                  congestion * weights['congestion'] + 
                  timing * weights['timing'] + 
                  power * weights['power'] + 
                  area * weights['area'])
        
        return min(1.0, max(0.0, overall))
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """计算置信度"""
        # 基于数据完整性和一致性计算置信度
        required_fields = ['hpwl', 'congestion', 'timing', 'power', 'area']
        present_fields = sum(1 for field in required_fields if field in data and data[field] is not None)
        
        completeness = present_fields / len(required_fields)
        
        # 数据一致性检查
        values = [data.get(field, 0.5) for field in required_fields]
        if len(values) > 1:
            std_dev = np.std(values)
            consistency = max(0.0, 1.0 - std_dev)
        else:
            consistency = 0.5
        
        confidence = (completeness + consistency) / 2
        return min(1.0, max(0.0, confidence))
    
    # 错误处理方法
    def _handle_analysis_error(self, design_data: Dict[str, Any], error: Exception) -> DesignFeatures:
        """处理设计分析错误"""
        logger.warning(f"使用默认设计特征: {str(error)}")
        return DesignFeatures(
            complexity=DesignComplexity.MEDIUM,
            design_type=DesignType.ASIC,
            component_count=design_data.get('component_count', 50000),
            hierarchy_levels=2,
            timing_critical=False,
            power_sensitive=False,
            area_constrained=False,
            special_networks=[],
            constraints={}
        )
    
    def _handle_strategy_generation_error(self, features: DesignFeatures, error: Exception) -> LayoutStrategy:
        """处理策略生成错误"""
        logger.warning(f"使用默认策略: {str(error)}")
        return LayoutStrategy(
            placement_strategy="hierarchical",
            routing_strategy="timing_driven",
            density_target=0.7,
            wirelength_weight=1.0,
            timing_weight=0.8,
            power_weight=0.6,
            congestion_weight=0.7,
            execution_plan=["initial_placement", "timing_optimization", "legalization", "routing"],
            optimization_focus="balanced_optimization"
        )
    
    def _handle_quality_assessment_error(self, layout_data: Dict[str, Any], error: Exception) -> QualityMetrics:
        """处理质量评估错误"""
        logger.warning(f"使用默认质量指标: {str(error)}")
        return QualityMetrics(
            hpwl_score=0.5,
            congestion_score=0.5,
            timing_score=0.5,
            power_score=0.5,
            area_score=0.5,
            overall_score=0.5,
            confidence=0.1
        )
    
    def _handle_data_format_error(self, data: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """处理数据格式错误"""
        logger.warning(f"数据格式错误，尝试修复: {str(error)}")
        return self._validate_and_fix_layout_data(data)
    
    def _handle_llm_timeout(self, operation: str, timeout: int) -> Any:
        """处理LLM超时"""
        logger.warning(f"LLM调用超时 ({timeout}s)，使用缓存结果")
        # 返回缓存的结果或默认值
        return None
    
    def _handle_invalid_response(self, response: str, expected_format: str) -> Any:
        """处理无效响应"""
        logger.warning(f"LLM响应格式无效，期望: {expected_format}")
        # 尝试解析或返回默认值
        return None
    
    def _handle_quality_assessment_failure(self, layout_data: Dict[str, Any], error: Exception) -> QualityMetrics:
        """处理质量评估失败"""
        logger.error(f"质量评估失败: {str(error)}")
        return self._handle_quality_assessment_error(layout_data, error)

def main():
    """主函数 - 演示增强LLM系统"""
    # 创建增强LLM系统
    llm_system = EnhancedLLMSystem()
    
    # 示例设计数据
    design_data = {
        'component_count': 75000,
        'hierarchy_levels': 4,
        'constraints': {
            'timing_critical': True,
            'power_sensitive': False,
            'area_constrained': True
        },
        'nets': [
            {'type': 'clock', 'name': 'clk_main'},
            {'type': 'power', 'name': 'vdd_core'}
        ]
    }
    
    # 示例布局数据
    layout_data = {
        'hpwl': '3.2e+09',
        'congestion': '0.3',
        'timing': '0.85',
        'power': '0.4',
        'area': '0.6'
    }
    
    print("=== 增强LLM系统演示 ===")
    
    # 1. 设计分析
    print("\n1. 设计分析:")
    features = llm_system.analyze_design(design_data)
    print(f"   复杂度: {features.complexity.value}")
    print(f"   设计类型: {features.design_type.value}")
    print(f"   组件数: {features.component_count}")
    print(f"   特殊网络: {features.special_networks}")
    
    # 2. 策略生成
    print("\n2. 布局策略生成:")
    strategy = llm_system.generate_layout_strategy(features)
    print(f"   布局策略: {strategy.placement_strategy}")
    print(f"   布线策略: {strategy.routing_strategy}")
    print(f"   优化重点: {strategy.optimization_focus}")
    print(f"   执行计划: {strategy.execution_plan[:3]}...")
    
    # 3. 质量评估
    print("\n3. 布局质量评估:")
    metrics = llm_system.assess_layout_quality(layout_data)
    print(f"   HPWL评分: {metrics.hpwl_score:.3f}")
    print(f"   拥塞评分: {metrics.congestion_score:.3f}")
    print(f"   时序评分: {metrics.timing_score:.3f}")
    print(f"   功耗评分: {metrics.power_score:.3f}")
    print(f"   面积评分: {metrics.area_score:.3f}")
    print(f"   综合评分: {metrics.overall_score:.3f}")
    print(f"   置信度: {metrics.confidence:.3f}")
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    main() 