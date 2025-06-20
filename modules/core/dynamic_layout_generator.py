"""
动态布局生成器模块
集成动态RAG检索器，实现基于质量反馈的布局生成优化
"""

import logging
import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..retrieval.dynamic_rag_retriever import DynamicRAGRetriever, LayoutQualityFeedback
from ..utils.llm_manager import LLMManager
from ..evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from ..knowledge.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

@dataclass
class LayoutGenerationContext:
    """布局生成上下文"""
    design_info: Dict[str, Any]
    retrieved_knowledge: List[Any]
    constraints: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    quality_feedback: Optional[LayoutQualityFeedback] = None

@dataclass
class LayoutOptimizationStep:
    """布局优化步骤"""
    step_id: int
    layout_state: Dict[str, Any]
    knowledge_used: List[str]
    quality_metrics: Dict[str, float]
    optimization_action: str
    timestamp: str

class DynamicLayoutGenerator:
    """动态布局生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化动态布局生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self._init_components()
        
        # 优化历史
        self.optimization_history = []
        self.quality_feedback_history = []
        
        # 动态调整参数
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.quality_threshold = config.get('quality_threshold', 0.8)
        self.max_iterations = config.get('max_iterations', 10)
        
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化动态RAG检索器
            retriever_config = self.config.get('dynamic_rag', {})
            self.dynamic_retriever = DynamicRAGRetriever(retriever_config)
            
            # 初始化LLM管理器
            llm_config = self.config.get('llm', {})
            self.llm_manager = LLMManager(llm_config)
            
            # 初始化评估器
            self.evaluator = MultiObjectiveEvaluator()
            
            # 初始化知识库
            kb_config = self.config.get('knowledge_base', {})
            self.knowledge_base = KnowledgeBase(kb_config)
            
            self.logger.info("动态布局生成器组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {str(e)}")
            raise
    
    def generate_layout_with_dynamic_rag(self, 
                                       design_info: Dict[str, Any],
                                       constraints: Dict[str, Any]) -> Dict[str, Any]:
        """使用动态RAG生成布局
        
        Args:
            design_info: 设计信息
            constraints: 约束条件
            
        Returns:
            Dict[str, Any]: 生成的布局结果
        """
        try:
            self.logger.info("开始动态RAG布局生成")
            
            # 1. 构建查询
            query = self._build_layout_query(design_info, constraints)
            
            # 2. 动态检索知识
            retrieved_knowledge = self.dynamic_retriever.retrieve_with_dynamic_reranking(
                query, design_info
            )
            
            # 3. 创建生成上下文
            context = LayoutGenerationContext(
                design_info=design_info,
                retrieved_knowledge=retrieved_knowledge,
                constraints=constraints,
                optimization_history=[]
            )
            
            # 4. 迭代优化布局
            final_layout = self._iterative_layout_optimization(context)
            
            # 5. 评估最终布局
            final_evaluation = self.evaluator.evaluate(final_layout)
            
            # 6. 生成质量反馈
            quality_feedback = self._generate_quality_feedback(final_layout, final_evaluation)
            
            # 7. 更新动态RAG检索器
            query_hash = self._hash_query(query)
            self.dynamic_retriever.update_with_feedback(
                query_hash, final_layout, quality_feedback
            )
            
            # 8. 构建结果
            result = {
                'layout': final_layout,
                'evaluation': final_evaluation,
                'quality_feedback': quality_feedback,
                'optimization_history': self.optimization_history,
                'retrieved_knowledge': [
                    {
                        'source': r.source,
                        'relevance_score': r.relevance_score,
                        'granularity_level': r.granularity_level,
                        'entity_embeddings_shape': r.entity_embeddings.shape if r.entity_embeddings is not None else None
                    }
                    for r in retrieved_knowledge
                ],
                'metadata': {
                    'generation_timestamp': datetime.now().isoformat(),
                    'dynamic_rag_version': '1.0',
                    'total_iterations': len(self.optimization_history)
                }
            }
            
            self.logger.info(f"动态RAG布局生成完成，质量分数: {quality_feedback.overall_score}")
            return result
            
        except Exception as e:
            self.logger.error(f"动态RAG布局生成失败: {str(e)}")
            raise
    
    def _iterative_layout_optimization(self, context: LayoutGenerationContext) -> Dict[str, Any]:
        """迭代布局优化"""
        try:
            # 1. 生成初始布局
            current_layout = self._generate_initial_layout(context)
            
            # 2. 迭代优化
            for iteration in range(self.max_iterations):
                self.logger.info(f"开始第 {iteration + 1} 轮优化")
                
                # 评估当前布局
                current_evaluation = self.evaluator.evaluate(current_layout)
                
                # 记录优化步骤
                optimization_step = LayoutOptimizationStep(
                    step_id=iteration,
                    layout_state=current_layout.copy(),
                    knowledge_used=[r.source for r in context.retrieved_knowledge],
                    quality_metrics=current_evaluation,
                    optimization_action='layout_optimization',
                    timestamp=datetime.now().isoformat()
                )
                self.optimization_history.append(optimization_step)
                
                # 检查是否达到质量阈值
                overall_score = self._calculate_overall_score(current_evaluation)
                if overall_score >= self.quality_threshold:
                    self.logger.info(f"达到质量阈值 {self.quality_threshold}，停止优化")
                    break
                
                # 基于反馈调整检索策略
                self._adapt_retrieval_strategy(context, current_evaluation)
                
                # 重新检索知识
                updated_query = self._build_adaptive_query(context, current_evaluation)
                context.retrieved_knowledge = self.dynamic_retriever.retrieve_with_dynamic_reranking(
                    updated_query, context.design_info
                )
                
                # 优化布局
                current_layout = self._optimize_layout_with_knowledge(
                    current_layout, context.retrieved_knowledge, current_evaluation
                )
            
            return current_layout
            
        except Exception as e:
            self.logger.error(f"迭代布局优化失败: {str(e)}")
            raise
    
    def _generate_initial_layout(self, context: LayoutGenerationContext) -> Dict[str, Any]:
        """生成初始布局"""
        try:
            # 使用LLM生成初始布局
            prompt = self._build_layout_generation_prompt(context)
            
            # 调用LLM
            llm_response = self.llm_manager.generate(prompt)
            
            # 解析布局
            initial_layout = self._parse_layout_response(llm_response)
            
            # 应用约束
            initial_layout = self._apply_constraints(initial_layout, context.constraints)
            
            return initial_layout
            
        except Exception as e:
            self.logger.error(f"生成初始布局失败: {str(e)}")
            # 返回默认布局
            return self._generate_default_layout(context.design_info)
    
    def _optimize_layout_with_knowledge(self, 
                                      current_layout: Dict[str, Any],
                                      knowledge: List[Any],
                                      evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """基于知识优化布局"""
        try:
            # 1. 分析当前布局的问题
            problems = self._analyze_layout_problems(current_layout, evaluation)
            
            # 2. 从知识中提取优化建议
            optimization_suggestions = self._extract_optimization_suggestions(knowledge, problems)
            
            # 3. 构建优化提示词
            optimization_prompt = self._build_optimization_prompt(
                current_layout, problems, optimization_suggestions
            )
            
            # 4. 调用LLM进行优化
            optimized_response = self.llm_manager.generate(optimization_prompt)
            
            # 5. 解析优化后的布局
            optimized_layout = self._parse_layout_response(optimized_response)
            
            # 6. 验证优化效果
            if self._is_optimization_effective(current_layout, optimized_layout, evaluation):
                return optimized_layout
            else:
                self.logger.warning("优化效果不明显，保持当前布局")
                return current_layout
                
        except Exception as e:
            self.logger.error(f"基于知识优化布局失败: {str(e)}")
            return current_layout
    
    def _adapt_retrieval_strategy(self, context: LayoutGenerationContext, evaluation: Dict[str, Any]):
        """适应检索策略"""
        try:
            # 分析评估结果，调整检索策略
            overall_score = self._calculate_overall_score(evaluation)
            
            if overall_score < 0.5:
                # 质量较差，增加检索范围
                self.dynamic_retriever.dynamic_k_range = (
                    self.dynamic_retriever.dynamic_k_range[0],
                    min(self.dynamic_retriever.dynamic_k_range[1] + 2, 20)
                )
                self.logger.info("增加检索范围")
            elif overall_score > 0.8:
                # 质量较好，减少检索范围
                self.dynamic_retriever.dynamic_k_range = (
                    max(self.dynamic_retriever.dynamic_k_range[0] - 1, 2),
                    self.dynamic_retriever.dynamic_k_range[1]
                )
                self.logger.info("减少检索范围")
                
        except Exception as e:
            self.logger.error(f"适应检索策略失败: {str(e)}")
    
    def _build_layout_query(self, design_info: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """构建布局查询"""
        return {
            'text': f"生成芯片布局，设计类型: {design_info.get('type', 'unknown')}, "
                   f"组件数量: {len(design_info.get('components', []))}, "
                   f"约束条件: {json.dumps(constraints, ensure_ascii=False)}",
            'type': 'layout_generation',
            'design_info': design_info,
            'constraints': constraints
        }
    
    def _build_adaptive_query(self, context: LayoutGenerationContext, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """构建自适应查询"""
        problems = self._analyze_layout_problems(context.layout_state, evaluation)
        
        return {
            'text': f"优化芯片布局，当前问题: {json.dumps(problems, ensure_ascii=False)}, "
                   f"设计类型: {context.design_info.get('type', 'unknown')}",
            'type': 'layout_optimization',
            'design_info': context.design_info,
            'problems': problems,
            'current_evaluation': evaluation
        }
    
    def _build_layout_generation_prompt(self, context: LayoutGenerationContext) -> str:
        """构建布局生成提示词"""
        knowledge_summary = self._summarize_knowledge(context.retrieved_knowledge)
        
        prompt = f"""请基于以下信息生成芯片布局：

设计信息：
{json.dumps(context.design_info, indent=2, ensure_ascii=False)}

约束条件：
{json.dumps(context.constraints, indent=2, ensure_ascii=False)}

相关知识：
{knowledge_summary}

请生成一个优化的芯片布局，考虑以下方面：
1. 组件位置优化
2. 布线长度最小化
3. 拥塞控制
4. 时序满足
5. 功耗优化

请以JSON格式返回布局结果。
"""
        return prompt
    
    def _build_optimization_prompt(self, 
                                 current_layout: Dict[str, Any],
                                 problems: Dict[str, Any],
                                 suggestions: List[str]) -> str:
        """构建优化提示词"""
        prompt = f"""请优化以下芯片布局：

当前布局：
{json.dumps(current_layout, indent=2, ensure_ascii=False)}

发现的问题：
{json.dumps(problems, indent=2, ensure_ascii=False)}

优化建议：
{chr(10).join(suggestions)}

请基于以上信息优化布局，重点关注问题解决。
请以JSON格式返回优化后的布局。
"""
        return prompt
    
    def _parse_layout_response(self, response: str) -> Dict[str, Any]:
        """解析布局响应"""
        try:
            # 尝试解析JSON
            layout = json.loads(response)
            return layout
        except json.JSONDecodeError:
            self.logger.warning("无法解析JSON响应，使用默认布局")
            return self._generate_default_layout({})
    
    def _apply_constraints(self, layout: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """应用约束条件"""
        try:
            # 这里实现约束应用逻辑
            # 例如：边界约束、间距约束等
            return layout
        except Exception as e:
            self.logger.error(f"应用约束失败: {str(e)}")
            return layout
    
    def _generate_default_layout(self, design_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成默认布局"""
        return {
            'components': [
                {
                    'name': 'default_comp',
                    'position': {'x': 100, 'y': 100},
                    'size': {'width': 100, 'height': 100}
                }
            ],
            'nets': [],
            'metadata': {
                'type': 'default_layout',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _analyze_layout_problems(self, layout: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """分析布局问题"""
        problems = {}
        
        # 分析各项指标
        for metric, score in evaluation.items():
            if score < 0.7:  # 质量阈值
                problems[metric] = {
                    'score': score,
                    'status': 'poor',
                    'description': f"{metric}指标不达标"
                }
        
        return problems
    
    def _extract_optimization_suggestions(self, knowledge: List[Any], problems: Dict[str, Any]) -> List[str]:
        """提取优化建议"""
        suggestions = []
        
        for result in knowledge:
            if hasattr(result, 'knowledge') and result.knowledge:
                # 从知识中提取相关建议
                if 'optimization_suggestions' in result.knowledge:
                    suggestions.extend(result.knowledge['optimization_suggestions'])
        
        return suggestions[:5]  # 限制建议数量
    
    def _is_optimization_effective(self, 
                                 original_layout: Dict[str, Any],
                                 optimized_layout: Dict[str, Any],
                                 evaluation: Dict[str, Any]) -> bool:
        """检查优化是否有效"""
        # 简单的有效性检查
        # 实际应用中可以使用更复杂的评估逻辑
        return True
    
    def _summarize_knowledge(self, knowledge: List[Any]) -> str:
        """总结知识"""
        summary = []
        
        for result in knowledge:
            if hasattr(result, 'knowledge') and result.knowledge:
                summary.append(f"来源: {result.source}, 相关性: {result.relevance_score:.3f}")
                if 'description' in result.knowledge:
                    summary.append(f"描述: {result.knowledge['description']}")
        
        return chr(10).join(summary) if summary else "无相关知识"
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """计算总体质量分数"""
        weights = {
            'wirelength': 0.25,
            'congestion': 0.25,
            'timing': 0.3,
            'power': 0.2
        }
        
        overall_score = 0.0
        for metric, weight in weights.items():
            if metric in evaluation:
                overall_score += evaluation[metric] * weight
        
        return overall_score
    
    def _generate_quality_feedback(self, layout: Dict[str, Any], evaluation: Dict[str, Any]) -> LayoutQualityFeedback:
        """生成质量反馈"""
        return LayoutQualityFeedback(
            wirelength_score=evaluation.get('wirelength', 0.0),
            congestion_score=evaluation.get('congestion', 0.0),
            timing_score=evaluation.get('timing', 0.0),
            power_score=evaluation.get('power', 0.0),
            overall_score=self._calculate_overall_score(evaluation),
            feedback_timestamp=datetime.now().isoformat(),
            layout_metadata=layout.get('metadata', {})
        )
    
    def _hash_query(self, query: Dict[str, Any]) -> str:
        """哈希查询"""
        query_str = json.dumps(query, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'total_optimizations': len(self.optimization_history),
            'avg_iterations_per_optimization': len(self.optimization_history) / max(1, len(self.quality_feedback_history)),
            'quality_feedback_count': len(self.quality_feedback_history),
            'avg_quality_score': np.mean([
                f.overall_score for f in self.quality_feedback_history
            ]) if self.quality_feedback_history else 0.0,
            'dynamic_rag_stats': self.dynamic_retriever.get_retrieval_statistics()
        } 