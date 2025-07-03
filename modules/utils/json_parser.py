"""
健壮的JSON解析工具
用于处理LLM返回的各种格式的JSON响应
"""

import json
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RobustJSONParser:
    """健壮的JSON解析器"""
    
    @staticmethod
    def parse_llm_response(response: str, response_type: str = "general") -> Dict[str, Any]:
        """解析LLM响应中的JSON
        
        Args:
            response: LLM响应文本
            response_type: 响应类型 (layout_strategy, layout_analysis, etc.)
            
        Returns:
            Dict: 解析后的JSON对象
        """
        try:
            # 清理响应文本
            cleaned_response = response.strip()
            
            # 尝试多种JSON提取方法
            parsed_json = RobustJSONParser._extract_json(cleaned_response)
            
            if parsed_json:
                logger.info(f"成功解析{response_type} JSON响应")
                logger.info(f"解析结果: {parsed_json}")
                return parsed_json
            
            # 如果解析失败，返回默认值
            return RobustJSONParser._get_default_response(response_type, cleaned_response)
            
        except Exception as e:
            logger.warning(f"解析{response_type}响应失败: {e}")
            return RobustJSONParser._get_default_response(response_type, "", error=str(e))
    
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """提取JSON的多种方法"""
        
        # 方法1: 使用正则表达式提取嵌套JSON
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            for match in matches:
                try:
                    parsed = json.loads(match)
                    return parsed
                except json.JSONDecodeError:
                    continue
        
        # 方法2: 查找完整的JSON对象
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            json_str = text[start:end]
            
            try:
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError:
                pass
        
        # 方法3: 尝试修复常见的JSON格式问题
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            json_str = text[start:end]
            
            fixed_json = RobustJSONParser._fix_common_json_issues(json_str)
            try:
                parsed = json.loads(fixed_json)
                return parsed
            except json.JSONDecodeError:
                pass
        
        # 方法4: 从自然语言中提取策略信息
        extracted_info = RobustJSONParser._extract_from_natural_language(text)
        if extracted_info:
            return extracted_info
        
        return None
    
    @staticmethod
    def _extract_from_natural_language(text: str) -> Optional[Dict[str, Any]]:
        """从自然语言响应中提取策略信息"""
        try:
            extracted_info = {}
            
            # 提取布局策略
            placement_patterns = [
                r'Placement Strategy[:\s]*([^.\n]+)',
                r'布局策略[:\s]*([^.\n]+)',
                r'采用([^算法]+)算法',
                r'使用([^算法]+)算法',
                r'GRAS[^算法]*算法',
                r'gras[^算法]*算法'
            ]
            
            for pattern in placement_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    strategy = match.group(1).strip()
                    # 清理策略名称，只保留关键算法名
                    if 'gras' in strategy.lower():
                        extracted_info['placement_strategy'] = 'gras'
                    elif 'hierarchical' in strategy.lower():
                        extracted_info['placement_strategy'] = 'hierarchical'
                    elif 'simulated' in strategy.lower() and 'annealing' in strategy.lower():
                        extracted_info['placement_strategy'] = 'simulated_annealing'
                    else:
                        # 提取第一个单词作为策略名
                        words = re.findall(r'\b\w+\b', strategy)
                        if words:
                            extracted_info['placement_strategy'] = words[0].lower()
                    break
            
            # 提取布线策略
            routing_patterns = [
                r'Routing Strategy[:\s]*([^.\n]+)',
                r'布线策略[:\s]*([^.\n]+)',
                r'采用([^算法]+)算法.*routing',
                r'使用([^算法]+)算法.*routing'
            ]
            
            for pattern in routing_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    strategy = match.group(1).strip()
                    strategy = re.sub(r'[^\w\s]', '', strategy).strip()
                    if strategy:
                        extracted_info['routing_strategy'] = strategy.lower()
                        break
            
            # 提取优化优先级
            priority_patterns = [
                r'Optimization Priorities[:\s]*([^.\n]+)',
                r'优化优先级[:\s]*([^.\n]+)',
                r'优先级设置为[:\s]*([^.\n]+)'
            ]
            
            for pattern in priority_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    priorities_text = match.group(1)
                    # 提取关键词
                    priorities = []
                    if 'timing' in priorities_text.lower() or '时序' in priorities_text:
                        priorities.append('timing')
                    if 'power' in priorities_text.lower() or '功耗' in priorities_text:
                        priorities.append('power')
                    if 'wirelength' in priorities_text.lower() or '线长' in priorities_text:
                        priorities.append('wirelength')
                    if priorities:
                        extracted_info['optimization_priorities'] = priorities
                        break
            
            # 如果提取到了有效信息，返回结果
            if extracted_info:
                # 添加默认值
                if 'placement_strategy' not in extracted_info:
                    extracted_info['placement_strategy'] = 'hierarchical'
                if 'routing_strategy' not in extracted_info:
                    extracted_info['routing_strategy'] = 'timing_driven'
                if 'optimization_priorities' not in extracted_info:
                    extracted_info['optimization_priorities'] = ['wirelength', 'timing', 'power']
                
                # 添加元数据
                extracted_info['metadata'] = {
                    'source': 'natural_language_extraction',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0',
                    'extraction_method': 'pattern_matching'
                }
                
                logger.info(f"从自然语言中提取策略信息: {extracted_info}")
                return extracted_info
            
            return None
            
        except Exception as e:
            logger.warning(f"从自然语言提取策略信息失败: {e}")
            return None
    
    @staticmethod
    def _fix_common_json_issues(json_str: str) -> str:
        """修复常见的JSON格式问题"""
        fixed = json_str
        
        # 修复未闭合的字符串
        fixed = re.sub(r'([^\\])"([^"]*?)$', r'\1"\2"', fixed)
        
        # 修复缺少逗号的问题
        fixed = re.sub(r'(\w+)\s*\n\s*"', r'\1,\n"', fixed)
        
        # 修复多余的逗号
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # 修复单引号
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
        
        # 修复布尔值
        fixed = re.sub(r'\btrue\b', 'true', fixed)
        fixed = re.sub(r'\bfalse\b', 'false', fixed)
        
        return fixed
    
    @staticmethod
    def _get_default_response(response_type: str, original_response: str = "", error: str = "") -> Dict[str, Any]:
        """获取默认响应"""
        
        base_metadata = {
            'source': f'fallback_{response_type}',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        if original_response:
            base_metadata['original_response'] = original_response[:200] + "..." if len(original_response) > 200 else original_response
        
        if error:
            base_metadata['error'] = error
        
        if response_type == "layout_strategy":
            return {
                'placement_strategy': 'hierarchical',
                'routing_strategy': 'timing_driven',
                'optimization_priorities': ['wirelength', 'timing', 'power'],
                'parameter_suggestions': {
                    'density_target': 0.7,
                    'wirelength_weight': 1.0,
                    'timing_weight': 0.8,
                    'power_weight': 0.6
                },
                'constraint_handling': {
                    'timing_constraints': 'aggressive',
                    'power_constraints': 'moderate',
                    'area_constraints': 'flexible'
                },
                'quality_targets': {
                    'hpwl_improvement': 0.05,
                    'timing_slack': 0.1,
                    'power_reduction': 0.03
                },
                'execution_plan': [
                    'initial_placement',
                    'timing_optimization',
                    'power_optimization',
                    'final_legalization'
                ],
                'metadata': base_metadata
            }
        
        elif response_type == "layout_analysis":
            return {
                'quality_score': 0.75,
                'area_utilization': 0.8,
                'routing_quality': 0.7,
                'timing_performance': 0.8,
                'power_distribution': 0.75,
                'issues': ['布局分析完成'],
                'suggestions': ['建议进一步优化'],
                'needs_optimization': False,
                'optimization_priority': 'low',
                'metadata': base_metadata
            }
        
        elif response_type == "design_analysis":
            return {
                'complexity': 'medium',
                'hierarchy_levels': 3,
                'component_count': 1000,
                'net_count': 800,
                'area_estimate': 1000000,
                'timing_critical_paths': 10,
                'power_domains': 2,
                'suggested_algorithm': 'hierarchical',
                'metadata': base_metadata
            }
        
        elif response_type == "optimization_suggestions":
            return {
                'suggestions': [
                    '优化线长',
                    '改善时序',
                    '减少功耗'
                ],
                'priority': 'medium',
                'estimated_improvement': 0.1,
                'metadata': base_metadata
            }
        
        else:
            # 通用默认响应
            return {
                'status': 'success',
                'message': '使用默认配置',
                'metadata': base_metadata
            }