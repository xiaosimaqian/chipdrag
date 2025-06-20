import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# 修改导入路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.rag_layout_experiment import RAGLayoutExperiment
from modules.core.rag_system import RAGSystem
from modules.utils.benchmark_loader import BenchmarkLoader
from modules.core.hierarchy import HierarchicalDecompositionManager
from modules.knowledge.knowledge_graph_builder import KnowledgeGraphBuilder
from modules.evaluation.layout_quality_evaluator import LayoutQualityEvaluator
from modules.evaluation.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from modules.evaluation.quality_evaluator import QualityEvaluator

def load_config(config_path: str) -> Dict[str, Any]:
    """加载实验配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 确保配置包含必要的部分
        required_sections = ['name', 'version', 'data', 'retrieval', 'generation', 'evaluation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置缺少必要部分: {section}")
                
        # 添加默认的知识库配置
        if 'knowledge_base' not in config:
            config['knowledge_base'] = {
                'path': 'data/knowledge_base',
                'type': 'json',
                'format': 'json',
                'hierarchy_config': {
                    'levels': ['system', 'module', 'component'],
                    'weights': [0.3, 0.4, 0.3]
                },
                'llm_config': {
                    'name': 'llama2',
                    'temperature': 0.7,
                    'max_tokens': 1000
                }
            }
            
        return config
        
    except Exception as e:
        logging.error(f"加载配置失败: {str(e)}")
        raise
        
def setup_logging(config: Dict[str, Any]):
    """设置日志
    
    Args:
        config: 配置信息
    """
    log_dir = Path(config['environment']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    
def run_experiments(config: Dict[str, Any]):
    """运行实验
    
    Args:
        config: 配置信息
    """
    try:
        # 设置日志
        setup_logging(config)
        
        # 初始化实验
        experiment = RAGLayoutExperiment(config=config)  # 只传入config参数
        
        # 运行实验
        results = experiment.run({
            'query': 'Generate layout for ALU module',
            'constraints': {
                'area': '100x100',
                'power': '1W',
                'timing': '1ns'
            }
        })
        
        # 保存结果
        output_dir = Path(config['environment']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
            yaml.dump(results, f, allow_unicode=True)
            
        logging.info("实验完成")
        
    except Exception as e:
        logging.error(f"实验运行失败: {str(e)}")
        raise
        
def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config('configs/experiment_config.yaml')
        
        # 运行实验
        run_experiments(config)
        
    except Exception as e:
        logging.error(f"实验运行失败: {str(e)}")
        raise
        
if __name__ == '__main__':
    main() 