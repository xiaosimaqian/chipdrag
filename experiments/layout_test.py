import os
import json
import logging
from datetime import datetime
from pathlib import Path

from modules.benchmark_loader import BenchmarkLoader
from modules.knowledge_base import KnowledgeBase
from modules.rag_system import RAGSystem
from modules.layout_quality_evaluator import LayoutQualityEvaluator
from modules.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from modules.multi_objective_evaluator import MultiObjectiveEvaluator
from modules.hierarchy import HierarchicalDecompositionManager
from modules.multi_modal_knowledge_graph import MultiModalKnowledgeGraph
from modules.knowledge_transfer import KnowledgeTransfer
from modules.llm_manager import LLMManager
from modules.embedding_manager import EmbeddingManager
from modules.layout_generator import LayoutGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LayoutTest:
    def __init__(self, output_dir="output", benchmark_dir="data/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 知识库配置
        kb_config = {
            'layout_experience': os.path.join(self.output_dir, 'knowledge_base', 'layout_experience'),
            'similarity_threshold': 0.5,
            'top_k': 5,
            'feature_weights': {
                'global': 0.4,
                'module': 0.3,
                'connection': 0.2,
                'constraint': 0.1
            }
        }
        
        # LLM配置
        llm_config = {
            'model_name': 'deepseek-coder',
            'api_base': 'http://localhost:11434',
            'temperature': 0.7,
            'max_tokens': 2000,
            'top_p': 0.9
        }
        
        # 嵌入配置
        embedding_config = {
            'model_name': 'bge-m3:latest',
            'api_base': 'http://localhost:11434',
            'dimension': 1024,
            'batch_size': 32
        }
        
        # 布局生成器配置
        layout_config = {
            'grid_size': 100,
            'min_spacing': 2,
            'max_iterations': 1000,
            'temperature': 0.8
        }
        
        # 多目标评估器配置
        objective_config = {
            'layout_quality': {
                'wirelength_weight': 0.4,
                'congestion_weight': 0.3,
                'timing_weight': 0.3
            }
        }
        
        # 初始化组件
        self.benchmark_loader = BenchmarkLoader(benchmark_dir=benchmark_dir)
        self.knowledge_base = KnowledgeBase(config=kb_config)
        self.llm_manager = LLMManager(config=llm_config)
        self.embedding_manager = EmbeddingManager(config=embedding_config)
        self.quality_evaluator = LayoutQualityEvaluator()
        self.constraint_evaluator = ConstraintSatisfactionEvaluator()
        self.objective_evaluator = MultiObjectiveEvaluator(config=objective_config)
        self.hierarchy_manager = HierarchicalDecompositionManager()
        self.knowledge_graph = MultiModalKnowledgeGraph()
        self.knowledge_transfer = KnowledgeTransfer()
        
        # 初始化RAG系统
        self.rag_system = RAGSystem(
            knowledge_base=self.knowledge_base,
            llm_manager=self.llm_manager,
            embedding_manager=self.embedding_manager,
            layout_generator=None,  # 先设为None，因为需要RAG系统来初始化布局生成器
            evaluator=self.objective_evaluator
        )
        
        # 初始化布局生成器
        self.layout_generator = LayoutGenerator(
            knowledge_base=self.knowledge_base,
            llm_manager=self.llm_manager,
            rag_system=self.rag_system
        )
        
        # 更新RAG系统的布局生成器
        self.rag_system.layout_generator = self.layout_generator
        
        # 创建实验目录
        self.experiment_dir = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def run_test(self, benchmark_name="mgc_fft_2"):
        """运行完整的测试流程"""
        logger.info(f"开始测试 {benchmark_name}")
        
        try:
            # 1. 加载基准测试数据
            logger.info("加载基准测试数据...")
            design_data = self.benchmark_loader.load_design(benchmark_name)
            
            # 2. 层次化分析
            logger.info("进行层次化分析...")
            hierarchy_info = self.hierarchy_manager.hierarchical_decomposition(design_data)
            
            # 3. 构建知识图谱
            logger.info("构建知识图谱...")
            # self.knowledge_graph.build_graph(design_data, hierarchy_info)
            
            # 4. 生成布局
            logger.info("生成布局...")
            layout = self.rag_system.generate_layout(
                design_info=design_data,
                hierarchy_info=hierarchy_info,
                knowledge_base=self.knowledge_base
            )
            
            # 5. 将布局结果加入知识库
            self.knowledge_base.add_case(
                layout=layout,
                optimization_result={},  # 可根据实际情况填写
                metadata={
                    'name': benchmark_name,
                    'type': 'generated',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # 6. 评估布局质量
            logger.info("评估布局质量...")
            quality_results = self.quality_evaluator.evaluate(layout)
            constraint_results = self.constraint_evaluator.evaluate(layout)
            objective_results = self.objective_evaluator.evaluate(layout)
            
            # 7. 保存结果
            self._save_results(
                benchmark_name=benchmark_name,
                layout=layout,
                quality_results=quality_results,
                constraint_results=constraint_results,
                objective_results=objective_results
            )
            
            logger.info("测试完成")
            return {
                "layout": layout,
                "quality_results": quality_results,
                "constraint_results": constraint_results,
                "objective_results": objective_results
            }
            
        except Exception as e:
            logger.error(f"测试过程中出现错误: {str(e)}")
            raise
    
    def _save_results(self, benchmark_name, layout, quality_results, constraint_results, objective_results):
        """保存测试结果"""
        results = {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "layout": layout,
            "quality_results": quality_results,
            "constraint_results": constraint_results,
            "objective_results": objective_results
        }
        
        # 保存布局结果
        layout_file = self.experiment_dir / "layout.json"
        with open(layout_file, "w") as f:
            json.dump(layout, f, indent=2)
        
        # 保存评估结果
        results_file = self.experiment_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # 保存详细日志
        log_file = self.experiment_dir / "test.log"
        with open(log_file, "w") as f:
            f.write(f"Benchmark: {benchmark_name}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write("\nQuality Results:\n")
            f.write(json.dumps(quality_results, indent=2))
            f.write("\n\nConstraint Results:\n")
            f.write(json.dumps(constraint_results, indent=2))
            f.write("\n\nObjective Results:\n")
            f.write(json.dumps(objective_results, indent=2))

if __name__ == "__main__":
    # 运行测试
    test = LayoutTest()
    results = test.run_test("mgc_fft_2")
    
    # 打印结果摘要
    print("\n测试结果摘要:")
    print(f"布局质量评分: {results['quality_results']['score']}")
    print(f"约束满足度: {results['constraint_results']['satisfaction_rate']}")
    print(f"多目标评分: {results['objective_results']['total_score']}") 