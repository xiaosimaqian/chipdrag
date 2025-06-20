import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from modules.core.rag_system import RAGSystem
from modules.core.layout_generator import LayoutGenerator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.llm_manager import LLMManager
from modules.utils.embedding_manager import EmbeddingManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RAGSystem')

class RAGController:
    def __init__(self):
        self.config = self._load_config()
        self._check_requirements()
        self._init_components()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_path = Path(__file__).parent / 'config' / 'rag_config.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"配置文件格式错误: {config_path}")
            raise
            
    def _check_requirements(self):
        """检查系统要求"""
        missing = []
        
        # 检查设计数据
        design_path = Path(self.config['paths']['design_data'])
        if not design_path.exists():
            missing.append(f"设计数据: {design_path}")
            
        # 检查知识库
        kb_path = Path(self.config['paths']['knowledge_base'])
        if not kb_path.exists():
            missing.append(f"知识库: {kb_path}")
            
        # 检查模型文件
        models_path = Path(self.config['paths']['models'])
        if not models_path.exists():
            missing.append(f"模型文件: {models_path}")
            
        if missing:
            logger.warning("缺少以下必要数据：")
            for item in missing:
                logger.warning(f"- {item}")
            raise FileNotFoundError("缺少必要的数据文件")
            
    def _init_components(self):
        """初始化系统组件"""
        # 初始化知识库
        self.knowledge_base = KnowledgeBase(
            config=self.config['knowledge_base']
        )
        
        # 初始化LLM管理器
        self.llm_manager = LLMManager(
            config=self.config['llm']
        )
        
        # 初始化嵌入管理器
        self.embedding_manager = EmbeddingManager(
            config=self.config['embedding']
        )
        
        # 初始化布局生成器
        self.layout_generator = LayoutGenerator(
            knowledge_base=self.knowledge_base,
            llm_manager=self.llm_manager
        )
        
        # 初始化评估器
        self.evaluator = MultiObjectiveEvaluator(
            config=self.config['evaluation']
        )
        
        # 初始化RAG系统
        self.rag_system = RAGSystem(
            knowledge_base=self.knowledge_base,
            llm_manager=self.llm_manager,
            embedding_manager=self.embedding_manager,
            layout_generator=self.layout_generator,
            evaluator=self.evaluator
        )
        
    def run(self, design_spec: Dict) -> Dict:
        """运行RAG系统
        
        Args:
            design_spec: 设计规范
            
        Returns:
            生成的布局结果
        """
        logger.info("启动RAG系统...")
        
        try:
            # 1. 知识检索
            logger.info("开始知识检索...")
            relevant_knowledge = self.rag_system.retrieve_knowledge(design_spec)
            
            # 2. 布局生成
            logger.info("开始布局生成...")
            initial_layout = self.layout_generator.generate(
                design_spec=design_spec,
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
            
if __name__ == "__main__":
    # 示例设计规范
    design_spec = {
        "name": "test_design",
        "area": {
            "width": 1000,
            "height": 1000
        },
        "components": [
            {
                "name": "comp1",
                "type": "macro",
                "width": 100,
                "height": 100
            }
        ],
        "constraints": {
            "max_wirelength": 1000,
            "max_congestion": 0.8
        }
    }
    
    # 运行系统
    controller = RAGController()
    result = controller.run(design_spec)
    
    # 输出结果
    print(json.dumps(result, indent=2)) 

class RAGSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._controller = None
        
    @property
    def controller(self):
        if self._controller is None:
            from modules.core.rag_controller import RAGController
            self._controller = RAGController(self.config)
        return self._controller

    def initialize(self):
        if self._controller is None:
            self._controller = RAGController(self.config) 