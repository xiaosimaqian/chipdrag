"""
知识迁移模块
"""

import logging
from typing import Dict, List, Any, Optional
from ..utils.llm_manager import LLMManager
from .hierarchy import Node

logger = logging.getLogger(__name__)

class KnowledgeTransfer:
    """知识迁移类"""
    
    def __init__(self, llm_manager: LLMManager):
        """初始化知识迁移器
        
        Args:
            llm_manager: LLM管理器实例
        """
        self.llm_manager = llm_manager
        logger.info("知识迁移器初始化完成")
        
    def transfer(self,
                source_node: Node,
                target_node: Node,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """迁移知识
        
        Args:
            source_node: 源节点
            target_node: 目标节点
            context: 上下文信息
            
        Returns:
            迁移后的知识
        """
        logger.info(f"开始知识迁移: {source_node.name} -> {target_node.name}")
        
        try:
            # 1. 分析源节点知识
            source_knowledge = self._analyze_source_knowledge(source_node)
            
            # 2. 分析目标节点需求
            target_requirements = self._analyze_target_requirements(target_node)
            
            # 3. 生成迁移策略
            transfer_strategy = self._generate_transfer_strategy(
                source_knowledge=source_knowledge,
                target_requirements=target_requirements,
                context=context
            )
            
            # 4. 执行知识迁移
            transferred_knowledge = self._execute_transfer(
                source_knowledge=source_knowledge,
                strategy=transfer_strategy
            )
            
            # 确保返回结果包含 features 字段
            if not isinstance(transferred_knowledge, dict):
                transferred_knowledge = {}
            
            if 'features' not in transferred_knowledge:
                transferred_knowledge['features'] = []
                
            return transferred_knowledge
            
        except Exception as e:
            logger.error(f"知识迁移失败: {str(e)}")
            return {'features': []}
        
    def _analyze_source_knowledge(self, source_node: Node) -> Dict[str, Any]:
        """分析源节点知识
        
        Args:
            source_node: 源节点
            
        Returns:
            源节点知识分析结果
        """
        # 使用LLM分析源节点知识
        knowledge = self.llm_manager.analyze_node_knowledge(source_node)
        return knowledge
        
    def _analyze_target_requirements(self, target_node: Node) -> Dict[str, Any]:
        """分析目标节点需求
        
        Args:
            target_node: 目标节点
            
        Returns:
            目标节点需求分析结果
        """
        # 使用LLM分析目标节点需求
        requirements = self.llm_manager.analyze_node_requirements(target_node)
        return requirements
        
    def _generate_transfer_strategy(self,
                                  source_knowledge: Dict[str, Any],
                                  target_requirements: Dict[str, Any],
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成迁移策略
        
        Args:
            source_knowledge: 源节点知识
            target_requirements: 目标节点需求
            context: 上下文信息
            
        Returns:
            迁移策略
        """
        # 使用LLM生成迁移策略
        strategy = self.llm_manager.generate_transfer_strategy(
            source_knowledge=source_knowledge,
            target_requirements=target_requirements,
            context=context
        )
        return strategy
        
    def _execute_transfer(self,
                         source_knowledge: Dict[str, Any],
                         strategy: Dict[str, Any]) -> Dict[str, Any]:
        """执行知识迁移
        
        Args:
            source_knowledge: 源节点知识
            strategy: 迁移策略
            
        Returns:
            迁移后的知识
        """
        try:
            # 使用LLM执行知识迁移
            transferred_knowledge = self.llm_manager.execute_knowledge_transfer(
                source_knowledge=source_knowledge,
                strategy=strategy
            )
            
            # 确保返回结果包含 features 字段
            if not isinstance(transferred_knowledge, dict):
                transferred_knowledge = {}
            
            if 'features' not in transferred_knowledge:
                transferred_knowledge['features'] = []
                
            return transferred_knowledge
            
        except Exception as e:
            logger.error(f"知识迁移执行失败: {str(e)}")
            return {'features': []} 