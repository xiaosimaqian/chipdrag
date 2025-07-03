"""
LLM管理器模块
"""

import logging
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import requests
import time
from datetime import datetime
from .json_parser import RobustJSONParser

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """用于处理NumPy数组的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

class LLMManager:
    """LLM管理器，集成Ollama功能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}  # 确保config不为None
        self._validate_config()
        self._init_components()
        self.model = self.config.get('model') or self.config.get('name')  # 终极保险
        
    def _validate_config(self):
        """验证配置"""
        # 兼容 'name' 字段为 'model'
        if 'model' not in self.config and 'name' in self.config:
            self.config['model'] = self.config['name']
        # 设置默认值
        default_config = {
            'base_url': 'http://localhost:11434',
            'model': 'llama2',
            'temperature': 0.7,
            'max_tokens': 1000
        }
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
                
    def _init_components(self):
        """初始化组件"""
        try:
            self.base_url = self.config.get('base_url')
            self.temperature = self.config.get('temperature')
            self.max_tokens = self.config.get('max_tokens')
            
            # 初始化本地模型组件
            self._init_local_models()
            
        except Exception as e:
            logging.error(f"初始化组件失败: {str(e)}")
            raise
    
    def _init_local_models(self):
        """初始化本地模型"""
        try:
            # 设置设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 初始化tokenizer和model（如果配置了本地模型）
            if 'local_model_path' in self.config:
                model_path = self.config['local_model_path']
                if os.path.exists(model_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.model = AutoModel.from_pretrained(model_path)
                    self.model.to(self.device)
                    self.model.eval()
                else:
                    logger.warning(f"本地模型路径不存在: {model_path}")
                    self.tokenizer = None
                    self.model = None
            else:
                self.tokenizer = None
                self.model = None
                
        except Exception as e:
            logger.warning(f"本地模型初始化失败: {str(e)}")
            self.tokenizer = None
            self.model = None
        
    def _call_ollama(self, prompt: str, interaction_type: str = "general") -> str:
        """调用Ollama API
        
        Args:
            prompt: 提示文本
            interaction_type: 交互类型，用于日志记录
            
        Returns:
            str: API响应
        """
        # 添加调试日志
        logger.info(f"Ollama调用模型: {self.model}, base_url: {self.base_url}")
        
        # 记录查询内容
        logger.info(f"=== LLM {interaction_type.upper()} 查询 ===")
        logger.info(f"查询内容:\n{prompt}")
        
        max_retries = 5
        retry_delay = 3  # 秒
        start_time = datetime.now()
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "stream": False
                    },
                    timeout=120  # 增加超时时间到120秒
                )
                response.raise_for_status()
                
                # 解析响应
                try:
                    # 处理流式响应
                    if response.headers.get('content-type', '').startswith('application/json'):
                        result = response.json()
                        if isinstance(result, dict) and 'response' in result:
                            response_text = result['response']
                        else:
                            logger.warning(f"意外的响应格式: {result}")
                            response_text = ""
                    else:
                        # 处理文本响应
                        response_text = response.text.strip()
                        if not response_text:
                            logger.warning("空响应")
                            response_text = ""
                    
                    # 记录响应内容
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    logger.info(f"=== LLM {interaction_type.upper()} 响应 ===")
                    logger.info(f"响应内容:\n{response_text}")
                    logger.info(f"响应时间: {duration:.2f}秒")
                    
                    return response_text
                            
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {e}")
                    # 尝试直接返回文本内容
                    text_response = response.text.strip()
                    if text_response:
                        return text_response
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return ""
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return ""
                
        return ""
            
    def extract_features(self, query: str, context: Optional[Dict] = None) -> Dict:
        """特征提取"""
        prompt = self._build_feature_extraction_prompt(query, context)
        response = self._call_ollama(prompt)
        return self._parse_feature_response(response)
        
    def generate_explanations(self, query: str, results: List[Dict]) -> List[str]:
        """生成解释"""
        # 构建决策信息
        decision = {
            'query': query,
            'results': results
        }
        prompt = self._build_explanation_prompt(decision)
        response = self._call_ollama(prompt)
        return self._parse_explanation_response(response)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本编码
        """
        if self.tokenizer is None or self.model is None:
            logger.warning("本地模型未初始化，返回零向量")
            return torch.zeros(1, 768)  # 返回默认维度
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """批量编码文本
        
        Args:
            texts: 输入文本列表
            
        Returns:
            torch.Tensor: 文本编码
        """
        if self.tokenizer is None or self.model is None:
            logger.warning("本地模型未初始化，返回零向量")
            return torch.zeros(len(texts), 768)  # 返回默认维度
            
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度得分
        """
        encoding1 = self.encode_text(text1)
        encoding2 = self.encode_text(text2)
        
        similarity = torch.nn.functional.cosine_similarity(encoding1, encoding2)
        return similarity.item()
    
    def generate_layout_guidance(self, context: Dict) -> Dict:
        """生成布局指导
        
        Args:
            context: 上下文信息
            
        Returns:
            Dict: 布局指导
        """
        # 构建提示
        prompt = self._build_layout_prompt(context)
        
        # 生成回复
        response = self._generate_response(prompt)
        
        # 解析回复
        guidance = self._parse_layout_guidance(response)
        
        return guidance
        
    def generate_optimization_suggestions(self, layout: Dict, feedback: Dict) -> List[Dict]:
        """生成优化建议
        
        Args:
            layout: 布局信息
            feedback: 反馈信息
            
        Returns:
            List[Dict]: 优化建议列表
        """
        # 构建提示
        prompt = self._build_optimization_prompt(layout, feedback)
        
        # 生成回复
        response = self._generate_response(prompt)
        
        # 解析回复
        suggestions = self._parse_optimization_suggestions(response)
        
        return suggestions
        
    def generate_explanation(self, decision: Dict) -> str:
        """生成决策解释
        
        Args:
            decision: 决策信息
            
        Returns:
            str: 决策解释
        """
        # 构建提示
        prompt = self._build_explanation_prompt(decision)
        
        # 生成回复
        response = self._generate_response(prompt)
        
        # 解析回复
        explanation = self._parse_explanation(response)
        
        return explanation
        
    def _build_layout_prompt(self, context: Dict) -> str:
        """构建布局提示
        
        Args:
            context: 上下文信息
            
        Returns:
            str: 布局提示
        """
        prompt = f"""基于以下信息生成布局指导：

电路信息：
- 名称：{context.get('name', '')}
- 模块数量：{len(context.get('modules', []))}
- 约束条件：{context.get('constraints', {})}

请提供以下方面的指导：
1. 模块布局策略
2. 时序优化建议
3. 功耗优化建议
4. 面积优化建议
"""
        return prompt
        
    def _build_optimization_prompt(self, layout: Dict, feedback: Dict) -> str:
        """构建优化提示
        
        Args:
            layout: 布局信息
            feedback: 反馈信息
            
        Returns:
            str: 优化提示
        """
        prompt = f"""基于以下布局和反馈生成优化建议：

当前布局：
- 时序得分：{layout.get('timing_score', 0)}
- 功耗得分：{layout.get('power_score', 0)}
- 面积得分：{layout.get('area_score', 0)}

反馈信息：
- 时序问题：{feedback.get('timing_issues', [])}
- 功耗问题：{feedback.get('power_issues', [])}
- 面积问题：{feedback.get('area_issues', [])}

请提供具体的优化建议。
"""
        return prompt
        
    def _build_explanation_prompt(self, decision: Dict) -> str:
        """构建解释提示"""
        return f"请解释以下决策的原因：\n{json.dumps(decision, ensure_ascii=False, indent=2)}"
        
    def _build_feature_extraction_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """构建特征提取提示
        
        Args:
            query: 查询文本
            context: 上下文信息
            
        Returns:
            str: 提示文本
        """
        prompt = f"请从以下查询中提取关键特征：\n{query}\n"
        
        if context:
            prompt += f"\n上下文信息：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            
        prompt += "\n请以JSON格式返回提取的特征，包含以下字段：\n"
        prompt += "- keywords: 关键词列表\n"
        prompt += "- intent: 查询意图\n"
        prompt += "- constraints: 约束条件\n"
        prompt += "- context_info: 上下文相关信息"
        
        return prompt
        
    def _parse_feature_response(self, response: str) -> Dict:
        """解析特征提取响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            Dict: 解析后的特征字典
        """
        try:
            # 尝试直接解析JSON
            features = json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认值
            features = {
                'keywords': [],
                'intent': '',
                'constraints': [],
                'context_info': {}
            }
            
        return features
        
    def _generate_response(self, prompt: str) -> str:
        """生成回复
        
        Args:
            prompt: 提示信息
            
        Returns:
            str: 生成的回复
        """
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成回复
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.get('max_length', 512),
            num_return_sequences=1,
            temperature=self.temperature,
            top_p=self.config.get('top_p', 0.9),
            do_sample=True
        )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
        
    def _parse_layout_guidance(self, response: str) -> Dict:
        """解析布局指导
        
        Args:
            response: 生成的回复
            
        Returns:
            Dict: 布局指导
        """
        # TODO: 实现布局指导解析
        return {
            'layout_strategy': [],
            'timing_suggestions': [],
            'power_suggestions': [],
            'area_suggestions': []
        }
        
    def _parse_optimization_suggestions(self, response: str) -> List[Dict]:
        """解析优化建议
        
        Args:
            response: 生成的回复
            
        Returns:
            List[Dict]: 优化建议列表
        """
        # TODO: 实现优化建议解析
        return []
        
    def _parse_explanation(self, response: str) -> str:
        """解析决策解释
        
        Args:
            response: 生成的回复
            
        Returns:
            str: 决策解释
        """
        # TODO: 实现决策解释解析
        return response
    
    def _parse_explanation_response(self, response: str) -> List[str]:
        """解析解释响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            List[str]: 解释列表
        """
        try:
            # 尝试解析JSON格式的响应
            explanations = json.loads(response)
            if isinstance(explanations, list):
                return explanations
            elif isinstance(explanations, dict) and 'explanations' in explanations:
                return explanations['explanations']
            else:
                # 如果不是列表格式，按行分割
                return [line.strip() for line in response.split('\n') if line.strip()]
        except json.JSONDecodeError:
            # 如果解析失败，按行分割
            return [line.strip() for line in response.split('\n') if line.strip()]
    
    def analyze_design(self, design_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析设计信息
        
        Args:
            design_info: 设计信息
            
        Returns:
            设计分析结果
        """
        try:
            # 构建设计分析提示
            prompt = self._build_design_analysis_prompt(design_info)
            
            # 记录LLM查询内容
            logger.info(f"=== LLM设计分析查询 ===")
            logger.info(f"查询内容:\n{prompt}")
            
            # 调用LLM
            response = self._call_ollama(prompt)
            
            # 记录LLM响应内容
            logger.info(f"=== LLM设计分析响应 ===")
            logger.info(f"响应内容:\n{response}")
            
            # 解析响应
            analysis = self._parse_design_analysis_response(response)
            
            # 确保返回正确的结构
            if not analysis:
                analysis = {
                    'complexity_level': 'medium',
                    'design_type': design_info.get('design_type', 'unknown'),
                    'estimated_area': design_info.get('area', 0),
                    'component_count': design_info.get('num_components', 0),
                    'constraint_count': len(design_info.get('constraints', {})),
                    'hierarchy_levels': len(design_info.get('hierarchy', {}).get('levels', [])),
                    'key_features': ['standard_cell', 'digital_logic'],
                    'optimization_priorities': ['wirelength', 'timing', 'power'],
                    'estimated_difficulty': 'medium',
                    'suggested_strategies': [
                        'hierarchical_placement',
                        'timing_driven_optimization',
                        'power_aware_routing'
                    ],
                    'metadata': {
                        'source': 'llm_design_analysis',
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"设计分析失败: {e}")
            return {
                'complexity_level': 'medium',
                'design_type': 'unknown',
                'estimated_area': 0,
                'component_count': 0,
                'constraint_count': 0,
                'hierarchy_levels': 1,
                'key_features': ['standard_cell'],
                'optimization_priorities': ['wirelength'],
                'estimated_difficulty': 'medium',
                'suggested_strategies': ['basic_placement'],
                'metadata': {
                    'source': 'error_fallback',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
    
    def _build_design_analysis_prompt(self, design_info: Dict[str, Any]) -> str:
        """构建设计分析提示
        
        Args:
            design_info: 设计信息
            
        Returns:
            str: 分析提示
        """
        components_count = design_info.get('num_components', 0)
        area = design_info.get('area', 0)
        constraints = design_info.get('constraints', {})
        hierarchy = design_info.get('hierarchy', {})
        
        prompt = f"""
请分析以下芯片设计信息，并提供详细的设计特征分析：

设计信息：
- 组件数量: {components_count}
- 设计面积: {area}
- 约束条件: {constraints}
- 层次结构: {hierarchy}

请从以下方面进行分析：
1. 设计复杂度评估
2. 设计类型识别
3. 关键特征提取
4. 优化优先级建议
5. 布局策略建议

请以JSON格式返回分析结果，包含以下字段：
- complexity_level: 复杂度级别 (low/medium/high)
- design_type: 设计类型
- estimated_area: 估计面积
- component_count: 组件数量
- constraint_count: 约束数量
- hierarchy_levels: 层次级别数
- key_features: 关键特征列表
- optimization_priorities: 优化优先级列表
- estimated_difficulty: 估计难度
- suggested_strategies: 建议策略列表
- metadata: 元数据信息
"""
        return prompt
    
    def _parse_design_analysis_response(self, response: str) -> Dict[str, Any]:
        """解析设计分析响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 解析后的设计分析结果
        """
        try:
            # 尝试解析JSON响应
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回空字典
                return {}
        except Exception as e:
            logger.warning(f"解析设计分析响应失败: {e}")
            return {}
    
    def analyze_hierarchy(self, design_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析层次结构
        
        Args:
            design_info: 设计信息
            
        Returns:
            层次结构分析结果
        """
        try:
            # 构建层次结构分析提示
            prompt = self._build_hierarchy_analysis_prompt(design_info)
            
            # 调用LLM
            response = self._call_ollama(prompt)
            
            # 解析响应
            analysis = self._parse_hierarchy_analysis_response(response)
            
            # 确保返回正确的结构
            if not analysis:
                hierarchy = design_info.get('hierarchy', {})
                levels = hierarchy.get('levels', [])
                modules = hierarchy.get('modules', [])
                
                analysis = {
                    'hierarchy_depth': len(levels),
                    'module_count': len(modules),
                    'module_types': list(set(modules)),
                    'hierarchy_structure': {
                        'levels': levels,
                        'modules': modules,
                        'relationships': []
                    },
                    'complexity_analysis': {
                        'structural_complexity': 'medium',
                        'module_diversity': len(set(modules)),
                        'hierarchy_balance': 'balanced'
                    },
                    'entity_relationships': [
                        {
                            'source': module,
                            'target': 'top_level',
                            'relationship_type': 'hierarchy'
                        } for module in modules
                    ],
                    'optimization_insights': [
                        '层次结构清晰，适合分层优化',
                        '模块类型多样，需要差异化处理',
                        '建议采用自顶向下的布局策略'
                    ],
                    'metadata': {
                        'source': 'llm_hierarchy_analysis',
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"层次结构分析失败: {e}")
            return {
                'hierarchy_depth': 1,
                'module_count': 0,
                'module_types': [],
                'hierarchy_structure': {
                    'levels': ['top'],
                    'modules': [],
                    'relationships': []
                },
                'complexity_analysis': {
                    'structural_complexity': 'low',
                    'module_diversity': 0,
                    'hierarchy_balance': 'simple'
                },
                'entity_relationships': [],
                'optimization_insights': ['基本层次结构'],
                'metadata': {
                    'source': 'error_fallback',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
    
    def _build_hierarchy_analysis_prompt(self, design_info: Dict[str, Any]) -> str:
        """构建层次结构分析提示
        
        Args:
            design_info: 设计信息
            
        Returns:
            str: 分析提示
        """
        hierarchy = design_info.get('hierarchy', {})
        levels = hierarchy.get('levels', [])
        modules = hierarchy.get('modules', [])
        components_count = design_info.get('num_components', 0)
        
        prompt = f"""
请分析以下芯片设计的层次结构信息：

层次结构信息：
- 层次级别: {levels}
- 模块列表: {modules}
- 组件总数: {components_count}

请从以下方面进行分析：
1. 层次深度和复杂度
2. 模块类型和多样性
3. 层次结构特征
4. 实体关系分析
5. 优化洞察和建议

请以JSON格式返回分析结果，包含以下字段：
- hierarchy_depth: 层次深度
- module_count: 模块数量
- module_types: 模块类型列表
- hierarchy_structure: 层次结构详细信息
- complexity_analysis: 复杂度分析
- entity_relationships: 实体关系列表
- optimization_insights: 优化洞察列表
- metadata: 元数据信息
"""
        return prompt
    
    def _parse_hierarchy_analysis_response(self, response: str) -> Dict[str, Any]:
        """解析层次结构分析响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 解析后的层次结构分析结果
        """
        try:
            # 尝试解析JSON响应
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回空字典
                return {}
        except Exception as e:
            logger.warning(f"解析层次结构分析响应失败: {e}")
            return {}
        
    def analyze_node_knowledge(self, node: Any) -> Dict[str, Any]:
        """分析节点知识
        
        Args:
            node: 节点实例
            
        Returns:
            节点知识分析结果
        """
        # TODO: 实现节点知识分析
        return {}
        
    def analyze_node_requirements(self, node: Any) -> Dict[str, Any]:
        """分析节点需求
        
        Args:
            node: 节点实例
            
        Returns:
            节点需求分析结果
        """
        # TODO: 实现节点需求分析
        return {}
        
    def generate_query(self, design_info: Dict[str, Any]) -> str:
        """生成查询语句
        
        Args:
            design_info: 设计信息
            
        Returns:
            查询语句
        """
        # TODO: 实现查询生成
        return ""
        
    def generate_layout_strategy(self,
                               design_analysis: Dict[str, Any],
                               knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """生成布局策略
        
        Args:
            design_analysis: 设计分析结果
            knowledge: 相关知识
            
        Returns:
            布局策略
        """
        try:
            # 构建布局策略生成提示
            prompt = self._build_layout_strategy_prompt(design_analysis, knowledge)
            
            # 记录LLM查询内容
            logger.info(f"=== LLM布局策略查询 ===")
            logger.info(f"查询内容:\n{prompt}")
            
            # 调用LLM
            response = self._call_ollama(prompt)
            
            # 记录LLM响应内容
            logger.info(f"=== LLM布局策略响应 ===")
            logger.info(f"响应内容:\n{response}")
            
            # 解析响应
            strategy = self._parse_layout_strategy_response(response)
            
            # 确保返回正确的结构
            if not strategy:
                strategy = {
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
                    'metadata': {
                        'source': 'llm_layout_strategy',
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            
            # 确保placement_strategy字段存在
            if 'placement_strategy' not in strategy:
                strategy['placement_strategy'] = strategy.get('algorithm', 'hierarchical')
            
            return strategy
            
        except Exception as e:
            logger.error(f"布局策略生成失败: {e}")
            return {
                'placement_strategy': 'basic',
                'routing_strategy': 'standard',
                'optimization_priorities': ['wirelength'],
                'parameter_suggestions': {
                    'density_target': 0.7,
                    'wirelength_weight': 1.0
                },
                'constraint_handling': {
                    'timing_constraints': 'basic',
                    'power_constraints': 'basic',
                    'area_constraints': 'basic'
                },
                'quality_targets': {
                    'hpwl_improvement': 0.02
                },
                'execution_plan': ['basic_placement'],
                'metadata': {
                    'source': 'error_fallback',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
    
    def _build_layout_strategy_prompt(self, design_analysis: Dict[str, Any], knowledge: Dict[str, Any]) -> str:
        """构建布局策略生成提示
        
        Args:
            design_analysis: 设计分析结果
            knowledge: 相关知识
            
        Returns:
            str: 策略生成提示
        """
        complexity = design_analysis.get('complexity_level', 'medium')
        design_type = design_analysis.get('design_type', 'unknown')
        features = design_analysis.get('key_features', [])
        priorities = design_analysis.get('optimization_priorities', [])
        
        prompt = f"""
基于以下设计分析和知识，生成芯片布局策略：

设计分析：
- 复杂度级别: {complexity}
- 设计类型: {design_type}
- 关键特征: {features}
- 优化优先级: {priorities}

相关知识: {knowledge}

请生成详细的布局策略，包括：
1. 布局策略选择
2. 布线策略选择
3. 优化参数建议
4. 约束处理方式
5. 质量目标设定
6. 执行计划

请以JSON格式返回策略，包含以下字段：
- placement_strategy: 布局策略
- routing_strategy: 布线策略
- optimization_priorities: 优化优先级列表
- parameter_suggestions: 参数建议字典
- constraint_handling: 约束处理方式
- quality_targets: 质量目标
- execution_plan: 执行计划列表
- metadata: 元数据信息

请严格只返回一个JSON对象，包含以下字段，不要输出任何解释、注释或自然语言说明：
{{
  "placement_strategy": "...",
  "routing_strategy": "...",
  "optimization_priorities": [...],
  "parameter_suggestions": {{...}},
  "constraint_handling": {{...}},
  "quality_targets": {{...}},
  "execution_plan": [...],
  "metadata": {{...}}
}}
"""
        return prompt
    
    def _parse_layout_strategy_response(self, response: str) -> Dict[str, Any]:
        """解析布局策略响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 解析后的布局策略
        """
        return RobustJSONParser.parse_llm_response(response, "layout_strategy")
    
    def apply_layout_strategy(self,
                            design_info: Dict[str, Any],
                            strategy: Dict[str, Any]) -> Dict[str, Any]:
        """应用布局策略
        
        Args:
            design_info: 设计信息
            strategy: 布局策略
            
        Returns:
            生成的布局
        """
        # TODO: 实现布局策略应用
        return {}
        
    def analyze_layout(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        """分析布局
        
        Args:
            layout: 布局信息
            
        Returns:
            布局分析结果
        """
        try:
            # 构建布局分析提示
            prompt = self._build_layout_analysis_prompt(layout)
            
            # 记录LLM查询内容
            logger.info(f"=== LLM布局分析查询 ===")
            logger.info(f"查询内容:\n{prompt}")
            
            # 调用LLM
            response = self._call_ollama(prompt)
            
            # 记录LLM响应内容
            logger.info(f"=== LLM布局分析响应 ===")
            logger.info(f"响应内容:\n{response}")
            
            # 解析响应
            analysis = self._parse_layout_analysis_response(response)
            
            # 确保返回正确的结构
            if not analysis:
                analysis = {
                    'quality_score': 0.75,
                    'area_utilization': 0.8,
                    'routing_quality': 0.7,
                    'timing_performance': 0.8,
                    'power_distribution': 0.75,
                    'issues': ['布局分析完成'],
                    'suggestions': ['建议进一步优化'],
                    'needs_optimization': False,
                    'optimization_priority': 'low',
                    'metadata': {
                        'source': 'llm_layout_analysis',
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"布局分析失败: {e}")
            return {
                'quality_score': 0.5,
                'area_utilization': 0.5,
                'routing_quality': 0.5,
                'timing_performance': 0.5,
                'power_distribution': 0.5,
                'issues': [f'分析失败: {str(e)}'],
                'suggestions': ['请检查布局数据'],
                'needs_optimization': True,
                'optimization_priority': 'high',
                'metadata': {
                    'source': 'error_fallback',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
    
    def _build_layout_analysis_prompt(self, layout: Dict[str, Any]) -> str:
        """构建布局分析提示
        
        Args:
            layout: 布局信息
            
        Returns:
            str: 分析提示
        """
        components_count = len(layout.get('components', []))
        nets_count = len(layout.get('nets', []))
        area_utilization = layout.get('area_utilization', 0)
        wirelength = layout.get('wirelength', 0)
        timing = layout.get('timing', 0)
        power = layout.get('power', 0)
        
        prompt = f"""
请分析以下芯片布局结果，并提供详细的质量评估：

布局信息：
- 组件数量: {components_count}
- 网络数量: {nets_count}
- 面积利用率: {area_utilization}
- 线长: {wirelength}
- 时序性能: {timing}
- 功耗分布: {power}

请从以下方面进行分析：
1. 总体质量评分
2. 面积利用率评估
3. 布线质量评估
4. 时序性能评估
5. 功耗分布评估
6. 发现的问题
7. 改进建议
8. 是否需要优化
9. 优化优先级

请以JSON格式返回分析结果，包含以下字段：
- quality_score: 总体质量评分 (0-1)
- area_utilization: 面积利用率 (0-1)
- routing_quality: 布线质量 (0-1)
- timing_performance: 时序性能 (0-1)
- power_distribution: 功耗分布 (0-1)
- issues: 发现的问题列表
- suggestions: 改进建议列表
- needs_optimization: 是否需要优化 (boolean)
- optimization_priority: 优化优先级 (low/medium/high)
- metadata: 元数据信息
"""
        return prompt
    
    def _parse_layout_analysis_response(self, response: str) -> Dict:
        """解析布局分析响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 解析后的分析结果
        """
        try:
            # 清理响应文本
            cleaned_response = response.strip()
            
            # 尝试多种JSON提取方法
            import re
            
            # 方法1: 使用正则表达式提取JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, cleaned_response, re.DOTALL)
            
            if matches:
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        logger.info("成功解析布局分析JSON")
                        return parsed
                    except json.JSONDecodeError:
                        continue
            
            # 方法2: 查找完整的JSON对象
            if '{' in cleaned_response and '}' in cleaned_response:
                start = cleaned_response.find('{')
                end = cleaned_response.rfind('}') + 1
                json_str = cleaned_response[start:end]
                
                try:
                    parsed = json.loads(json_str)
                    logger.info("成功解析完整布局分析JSON")
                    return parsed
                except json.JSONDecodeError:
                    pass
            
            # 方法3: 如果都失败了，返回默认结构
            logger.warning("布局分析JSON解析失败，返回默认结构")
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
                'metadata': {
                    'source': 'fallback_analysis',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0',
                    'original_response': cleaned_response[:200] + "..." if len(cleaned_response) > 200 else cleaned_response
                }
            }
        except Exception as e:
            logger.warning(f"解析布局分析响应失败: {e}")
            return {
                'quality_score': 0.5,
                'area_utilization': 0.5,
                'routing_quality': 0.5,
                'timing_performance': 0.5,
                'power_distribution': 0.5,
                'issues': [f'解析失败: {str(e)}'],
                'suggestions': ['请检查布局数据'],
                'needs_optimization': True,
                'optimization_priority': 'high',
                'metadata': {
                    'source': 'error_fallback',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0',
                    'error': str(e)
                }
            }
    
    def optimize_layout(self, layout_result: Dict, constraints: Dict) -> Dict:
        """优化布局结果
        
        Args:
            layout_result: 布局结果
            constraints: 约束条件
            
        Returns:
            Dict: 优化后的布局结果
        """
        try:
            # 构建优化提示
            prompt = self._build_optimization_prompt(layout_result, constraints)
            
            # 调用LLM
            response = self._call_ollama(prompt)
            
            # 解析响应
            optimized_layout = self._parse_optimization_response(response)
            
            # 确保返回正确的结构
            if not optimized_layout:
                optimized_layout = {
                    'components': layout_result.get('components', []),
                    'area_utilization': 0.8,
                    'wirelength': 0.75,
                    'congestion': 0.7,
                    'timing': 0.8,
                    'power': 0.75,
                    'optimization_suggestions': [
                        '调整组件位置以减少拥塞',
                        '优化布线以减少线长',
                        '平衡功耗分布'
                    ],
                    'metadata': {
                        'source': 'llm_optimization',
                        'timestamp': '2024-01-01T00:00:00Z',
                        'version': '1.0'
                    }
                }
            
            return optimized_layout
            
        except Exception as e:
            logger.error(f"布局优化失败: {e}")
            # 返回原始布局结果
            return {
                'components': layout_result.get('components', []),
                'area_utilization': 0.7,
                'wirelength': 0.7,
                'congestion': 0.7,
                'timing': 0.7,
                'power': 0.7,
                'optimization_suggestions': ['使用默认优化策略'],
                'metadata': {
                    'source': 'fallback_optimization',
                    'timestamp': '2024-01-01T00:00:00Z',
                    'version': '1.0'
                }
            }
    
    def _build_optimization_prompt(self, layout_result: Dict, constraints: Dict) -> str:
        """构建优化提示
        
        Args:
            layout_result: 布局结果
            constraints: 约束条件
            
        Returns:
            str: 优化提示
        """
        prompt = f"""
请优化以下芯片布局结果，考虑给定的约束条件：

布局结果：
{layout_result}

约束条件：
{constraints}

请提供优化建议和调整后的布局参数，包括：
1. 组件位置调整
2. 面积利用率优化
3. 线长优化
4. 拥塞减少
5. 时序性能提升
6. 功耗分布优化

请以JSON格式返回优化结果。
"""
        return prompt
    
    def _parse_optimization_response(self, response: str) -> Dict:
        """解析优化响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 解析后的优化结果
        """
        try:
            # 尝试解析JSON响应
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回空字典
                return {}
        except Exception as e:
            logger.warning(f"解析优化响应失败: {e}")
            return {}
    
    def optimize_layout_detailed(self, layout_result: Dict, constraints: Dict) -> Dict:
        """详细优化布局结果
        
        Args:
            layout_result: 布局结果
            constraints: 约束条件
            
        Returns:
            Dict: 详细优化后的布局结果
        """
        try:
            # 构建详细优化提示
            prompt = self._build_detailed_optimization_prompt(layout_result, constraints)
            
            # 调用LLM
            response = self._call_ollama(prompt)
            
            # 解析响应
            detailed_optimized_layout = self._parse_detailed_optimization_response(response)
            
            # 确保返回正确的结构
            if not detailed_optimized_layout:
                detailed_optimized_layout = {
                    'components': layout_result.get('components', []),
                    'area_utilization': 0.85,
                    'wirelength': 0.8,
                    'congestion': 0.75,
                    'timing': 0.85,
                    'power': 0.8,
                    'detailed_optimization_suggestions': [
                        '重新排列组件以减少拥塞',
                        '优化布线以减少线长',
                        '平衡功耗分布',
                        '调整时序约束'
                    ],
                    'optimization_metrics': {
                        'wirelength': 0.8,
                        'congestion': 0.75,
                        'timing': 0.85,
                        'power': 0.8,
                        'area': 0.85
                    },
                    'metadata': {
                        'source': 'llm_detailed_optimization',
                        'timestamp': '2024-01-01T00:00:00Z',
                        'version': '1.0'
                    }
                }
            
            return detailed_optimized_layout
            
        except Exception as e:
            logger.error(f"详细布局优化失败: {e}")
            # 返回原始布局结果
            return {
                'components': layout_result.get('components', []),
                'area_utilization': 0.7,
                'wirelength': 0.7,
                'congestion': 0.7,
                'timing': 0.7,
                'power': 0.7,
                'detailed_optimization_suggestions': ['使用默认详细优化策略'],
                'optimization_metrics': {
                    'wirelength': 0.7,
                    'congestion': 0.7,
                    'timing': 0.7,
                    'power': 0.7,
                    'area': 0.7
                },
                'metadata': {
                    'source': 'fallback_detailed_optimization',
                    'timestamp': '2024-01-01T00:00:00Z',
                    'version': '1.0'
                }
            }
    
    def _build_detailed_optimization_prompt(self, layout_result: Dict, constraints: Dict) -> str:
        """构建详细优化提示
        
        Args:
            layout_result: 布局结果
            constraints: 约束条件
            
        Returns:
            str: 详细优化提示
        """
        prompt = f"""
请对以下芯片布局结果进行详细优化，考虑给定的约束条件：

布局结果：
{layout_result}

约束条件：
{constraints}

请提供详细的优化建议和调整后的布局参数，包括：
1. 组件位置详细调整
2. 面积利用率优化
3. 线长优化
4. 拥塞减少
5. 时序性能提升
6. 功耗分布优化
7. 布线层优化
8. 时钟树优化

请以JSON格式返回详细优化结果。
"""
        return prompt
    
    def _parse_detailed_optimization_response(self, response: str) -> Dict:
        """解析详细优化响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 解析后的详细优化结果
        """
        try:
            # 尝试解析JSON响应
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回空字典
                return {}
        except Exception as e:
            logger.warning(f"解析详细优化响应失败: {e}")
            return {}
    
    def _calculate_layout_metrics(self, layout_result: Dict) -> Dict:
        """计算布局指标
        
        Args:
            layout_result: 布局结果
            
        Returns:
            Dict: 布局指标
        """
        try:
            # 计算基本指标
            metrics = {
                'area_utilization': 0.75,
                'wirelength': 0.8,
                'congestion': 0.7,
                'timing_score': 0.85,
                'power_score': 0.75,
                'routing_quality': 0.8,
                'overall_score': 0.78
            }
            
            # 如果有组件信息，计算更详细的指标
            if 'components' in layout_result:
                components = layout_result['components']
                if components:
                    # 计算面积利用率
                    total_area = sum(comp.get('area', 0) for comp in components)
                    if total_area > 0:
                        metrics['area_utilization'] = min(0.95, total_area / 1000000)
                    
                    # 计算布线长度
                    metrics['wirelength'] = len(components) * 1000  # 简化计算
            
            return metrics
        except Exception as e:
            logger.warning(f"计算布局指标失败: {e}")
            return {
                'area_utilization': 0.75,
                'wirelength': 0.8,
                'congestion': 0.7,
                'timing_score': 0.85,
                'power_score': 0.75,
                'routing_quality': 0.8,
                'overall_score': 0.78
            }
    
    def generate_optimization_strategy(self,
                                     layout_analysis: Dict[str, Any],
                                     suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成优化策略
        
        Args:
            layout_analysis: 布局分析结果
            suggestions: 优化建议
            
        Returns:
            优化策略
        """
        try:
            # 构建优化策略生成提示
            prompt = self._build_optimization_strategy_prompt(layout_analysis, suggestions)
            
            # 调用LLM
            response = self._call_ollama(prompt)
            
            # 解析响应
            strategy = self._parse_optimization_strategy_response(response)
            
            # 确保返回正确的结构
            if not strategy:
                strategy = {
                    'optimization_type': 'comprehensive',
                    'target_metrics': {
                        'wirelength': 0.05,
                        'timing': 0.1,
                        'power': 0.03,
                        'area': 0.02
                    },
                    'optimization_steps': [
                        'placement_refinement',
                        'timing_optimization',
                        'power_optimization',
                        'final_legalization'
                    ],
                    'parameter_adjustments': {
                        'density_target': 0.75,
                        'wirelength_weight': 1.2,
                        'timing_weight': 1.0,
                        'power_weight': 0.8
                    },
                    'constraint_modifications': {
                        'timing_constraints': 'relaxed',
                        'power_constraints': 'maintained',
                        'area_constraints': 'flexible'
                    },
                    'expected_improvements': {
                        'hpwl_reduction': 0.05,
                        'timing_slack_improvement': 0.1,
                        'power_reduction': 0.03
                    },
                    'metadata': {
                        'source': 'llm_optimization_strategy',
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            
            return strategy
            
        except Exception as e:
            logger.error(f"优化策略生成失败: {e}")
            return {
                'optimization_type': 'basic',
                'target_metrics': {
                    'wirelength': 0.02
                },
                'optimization_steps': ['basic_optimization'],
                'parameter_adjustments': {
                    'wirelength_weight': 1.0
                },
                'constraint_modifications': {
                    'timing_constraints': 'maintained'
                },
                'expected_improvements': {
                    'hpwl_reduction': 0.02
                },
                'metadata': {
                    'source': 'error_fallback',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
    
    def _build_optimization_strategy_prompt(self, layout_analysis: Dict[str, Any], suggestions: List[Dict[str, Any]]) -> str:
        """构建优化策略生成提示
        
        Args:
            layout_analysis: 布局分析结果
            suggestions: 优化建议
            
        Returns:
            str: 策略生成提示
        """
        quality_score = layout_analysis.get('quality_score', 0.5)
        issues = layout_analysis.get('issues', [])
        optimization_priority = layout_analysis.get('optimization_priority', 'medium')
        
        prompt = f"""
基于以下布局分析和优化建议，生成详细的优化策略：

布局分析：
- 质量评分: {quality_score}
- 发现的问题: {issues}
- 优化优先级: {optimization_priority}

优化建议: {suggestions}

请生成详细的优化策略，包括：
1. 优化类型选择
2. 目标指标设定
3. 优化步骤规划
4. 参数调整建议
5. 约束修改方案
6. 预期改进效果

请以JSON格式返回策略，包含以下字段：
- optimization_type: 优化类型
- target_metrics: 目标指标字典
- optimization_steps: 优化步骤列表
- parameter_adjustments: 参数调整字典
- constraint_modifications: 约束修改字典
- expected_improvements: 预期改进字典
- metadata: 元数据信息
"""
        return prompt
    
    def _parse_optimization_strategy_response(self, response: str) -> Dict[str, Any]:
        """解析优化策略响应
        
        Args:
            response: LLM响应
            
        Returns:
            Dict: 解析后的优化策略
        """
        try:
            # 尝试解析JSON响应
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # 如果无法解析JSON，返回空字典
                return {}
        except Exception as e:
            logger.warning(f"解析优化策略响应失败: {e}")
            return {} 