"""
RAG控制器核心模块
"""

import logging
from typing import Dict, Optional, List
from ..knowledge.knowledge_base import KnowledgeBase
from ..utils.llm_manager import LLMManager
from ..utils.embedding_manager import EmbeddingManager
from .layout_generator import LayoutGenerator
from ..evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from .rag_system import RAGSystem
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGController:
    """RAG控制器核心类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化RAG控制器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._init_components()
        logger.info("RAG控制器核心初始化完成")
        
    def _init_components(self):
        """初始化系统组件"""
        # 初始化知识库
        self.knowledge_base = KnowledgeBase(
            config=self.config.get('knowledge_base', {})
        )
        
        # 初始化LLM管理器
        self.llm_manager = LLMManager(
            config=self.config.get('llm', {})
        )
        
        # 初始化嵌入管理器
        self.embedding_manager = EmbeddingManager(
            config=self.config.get('embedding', {})
        )
        
        # 初始化布局生成器
        self.layout_generator = LayoutGenerator(
            knowledge_base=self.knowledge_base,
            llm_manager=self.llm_manager
        )
        
        # 初始化评估器
        self.evaluator = MultiObjectiveEvaluator(
            config=self.config.get('evaluation', {})
        )
        
        # 初始化RAG系统
        self.rag_system = RAGSystem(
            knowledge_base=self.knowledge_base,
            llm_manager=self.llm_manager,
            embedding_manager=self.embedding_manager,
            layout_generator=self.layout_generator,
            evaluator=self.evaluator,
            config=self.config
        )
        
    def run(self, design_info: Dict) -> Dict:
        """运行RAG系统
        
        Args:
            design_info: 设计信息
            
        Returns:
            布局结果
        """
        logger.info("启动RAG系统...")
        
        try:
            # 1. 知识检索
            logger.info("开始知识检索...")
            relevant_knowledge = self.rag_system.retrieve_knowledge(design_info)
            
            # 2. 布局生成
            logger.info("开始布局生成...")
            initial_layout = self.layout_generator.generate(
                design_info=design_info,
                knowledge=relevant_knowledge
            )
            
            # 3. 布局评估
            logger.info("开始布局评估...")
            evaluation_results = self.evaluator.evaluate(initial_layout)
            
            # 4. 布局优化
            logger.info("开始布局优化...")
            optimized_layout = self.rag_system.optimize_layout(
                layout=initial_layout,
                evaluation=evaluation_results
            )
            
            # 5. 结果验证
            logger.info("开始结果验证...")
            final_evaluation = self.evaluator.evaluate(optimized_layout)
            
            return {
                "layout": optimized_layout,
                "evaluation": final_evaluation,
                "knowledge_used": relevant_knowledge
            }
            
        except Exception as e:
            logger.error(f"RAG系统运行失败: {str(e)}")
            raise 

    def add_pdf_knowledge(self, pdf_path: str) -> bool:
        """添加PDF知识到知识库
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            bool: 是否成功添加
        """
        return self.rag_system.add_pdf_knowledge(pdf_path)
        
    def verify_knowledge(self, query: Dict, top_k: int = 5) -> List[Dict]:
        """验证知识库中的知识
        
        Args:
            query: 查询条件，包含设计需求
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 知识列表
        """
        try:
            # 从查询条件中提取设计需求
            design_requirement = query.get('text', '')
            if not design_requirement:
                logger.warning("查询条件中缺少设计需求文本")
                return []
            
            # 从知识库中检索知识
            results = self.knowledge_base.query(
                text=design_requirement,
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"验证知识失败: {str(e)}")
            return [] 

    def query_knowledge(self, design_requirement_path: str, top_k: int = 5) -> List[Dict]:
        """基于设计需求进行多模态知识检索
        
        Args:
            design_requirement_path: 设计需求文本文件路径
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 多模态知识列表
        """
        try:
            # 读取设计需求
            with open(design_requirement_path, 'r', encoding='utf-8') as f:
                design_requirement = f.read().strip()
            logger.info(f"设计需求: {design_requirement}")
            
            # 构建查询条件
            query = {
                'text': design_requirement,
                'type': 'layout_design'  # 不限制来源，支持多模态
            }
            
            # 从知识库中检索知识
            results = self.knowledge_base.query(
                text=design_requirement,
                top_k=top_k
            )
            
            # 处理多模态结果
            processed_results = []
            for result in results:
                processed_result = {
                    'text': result.get('text', ''),
                    'metadata': {
                        'type': result.get('type', 'unknown'),
                        'source': result.get('source', 'unknown'),
                        'filename': result.get('filename', 'unknown'),
                        'timestamp': result.get('timestamp', 'unknown')
                    },
                    'similarity': result.get('similarity', 0.0)
                }
                
                # 添加其他模态的信息
                if result.get('source') == 'pdf':
                    processed_result['layout'] = result.get('layout', {})
                elif result.get('source') == 'image':
                    processed_result['image'] = result.get('image', {})
                elif result.get('source') == 'structured':
                    processed_result['structured_data'] = result.get('structured_data', {})
                    
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"查询知识失败: {str(e)}")
            return [] 