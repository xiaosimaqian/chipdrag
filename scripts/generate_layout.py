import os
import json
import logging
from chiprag.modules.rag_system import RAGSystem
from chiprag.modules.benchmark_loader import BenchmarkLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Layout_Generation')

def main():
    """主函数"""
    logger.info("开始布局生成")
    
    # 初始化RAG系统
    rag_system = RAGSystem()
    logger.info("初始化RAG系统成功")
    
    # 初始化基准测试加载器
    benchmark_dir = os.path.join(os.path.dirname(__file__), 'data', 'designs', 'ispd_2015_contest_benchmark')
    benchmark_loader = BenchmarkLoader(benchmark_dir)
    logger.info("初始化基准测试加载器成功")
    
    # 加载设计
    design_name = 'mgc_fft_1'
    logger.info(f"开始加载设计: {design_name}")
    design_info = benchmark_loader.load_design(design_name)
    logger.info("设计信息加载完成")
    
    # 执行层次化分解
    logger.info("开始执行层次化分解")
    hierarchy = rag_system.knowledge_base.hierarchical_decomposition(design_info)
    logger.info("层次化分解完成")
    
    # 生成布局
    logger.info("开始生成布局")
    layout = rag_system.layout_generator.generate_layout(hierarchy)
    logger.info("布局生成完成")
    
    # 评估布局质量
    logger.info("开始评估布局质量")
    quality = rag_system.quality_evaluator.evaluate_layout(layout)
    logger.info("布局质量评估完成")
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存布局结果
    layout_file = os.path.join(output_dir, f'{design_name}_layout.json')
    with open(layout_file, 'w') as f:
        json.dump(layout, f, indent=2)
    logger.info(f"布局结果已保存到: {layout_file}")
    
    # 保存质量评估结果
    quality_file = os.path.join(output_dir, f'{design_name}_quality.json')
    with open(quality_file, 'w') as f:
        json.dump(quality, f, indent=2)
    logger.info(f"质量评估结果已保存到: {quality_file}")
    
    logger.info("布局生成完成")

if __name__ == '__main__':
    main() 