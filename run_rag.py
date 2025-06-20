"""
RAG控制器模块
"""

import logging
from typing import Dict, Optional, Any
from modules.core.rag_controller import RAGController
from modules.core.rag_system import RAGSystem as BaseRAGSystem
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.llm_manager import LLMManager
from modules.utils.embedding_manager import EmbeddingManager
from modules.core.layout_generator import LayoutGenerator
from modules.core.knowledge_transfer import KnowledgeTransfer
from modules.core.multimodal_fusion import MultimodalFusion

logger = logging.getLogger(__name__)

class RAGSystem(BaseRAGSystem):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._controller = None
        
    @property
    def controller(self):
        if self._controller is None:
            from chiprag.modules.core.rag_controller import RAGController
            self._controller = RAGController(self.config)
        return self._controller

class RAGController:
    """RAG控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化RAG控制器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化RAG控制器")
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化RAG系统
            self.rag_system = RAGSystem(self.config)
            
            # 初始化知识库 - 传递knowledge_base子配置
            knowledge_config = self.config.get('knowledge_base', {})
            self.knowledge_base = KnowledgeBase(knowledge_config)
            
            # 初始化知识迁移器
            self.knowledge_transfer = KnowledgeTransfer(self.config)
            
            # 初始化多模态融合器
            self.multimodal_fusion = MultimodalFusion(self.config)
            
            # 初始化评估器
            from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
            evaluator_config = self.config.get('evaluation_config', {
                'weights': {'wirelength': 0.3, 'congestion': 0.3, 'timing': 0.4},
                'thresholds': {'wirelength': 1.0, 'congestion': 1.0, 'timing': 1.0},
                'metrics': {
                    'wirelength': {'type': 'minimize', 'weight': 0.3, 'threshold': 1.0},
                    'congestion': {'type': 'minimize', 'weight': 0.3, 'threshold': 1.0},
                    'timing': {'type': 'minimize', 'weight': 0.4, 'threshold': 1.0}
                }
            })
            self.evaluator = MultiObjectiveEvaluator(evaluator_config)
            
            # 初始化LLM管理器
            llm_config = self.config.get('llm', {})
            if isinstance(llm_config, dict):
                self.llm_manager = LLMManager(llm_config)
            else:
                self.llm_manager = LLMManager({})
            
            # 初始化布局生成器
            from modules.core.layout_generator import LayoutGenerator
            self.layout_generator = LayoutGenerator({
                'layout_config': self.config.get('layout_config', {}),
                'llm_manager': self.llm_manager
            })
            
            # 初始化嵌入管理器
            from modules.utils.embedding_manager import EmbeddingManager
            embedding_config = self.config.get('embedding_config', {})
            self.embedding_manager = EmbeddingManager(embedding_config)
            
            self.logger.info("组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {str(e)}")
            raise
        
    def run(self, design_info: Dict) -> Dict:
        """运行RAG系统
        
        Args:
            design_info: 设计信息
            
        Returns:
            布局结果
        """
        return self.controller.run(design_info) 