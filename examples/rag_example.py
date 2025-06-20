from pathlib import Path
import json
import logging

from ..modules.rag_controller import RAGController

def main():
    # 设置路径
    design_path = Path('chip_design/ispd_2015_contest_benchmark/mgc_fft_1')
    knowledge_base_path = Path('knowledge_base')
    config_path = Path('config/rag_config.json')
    
    # 初始化RAG控制器
    controller = RAGController(
        design_path=str(design_path),
        knowledge_base_path=str(knowledge_base_path),
        config_path=str(config_path)
    )
    
    # 示例查询
    query = """
    优化FFT模块的布局，重点关注：
    1. 时序性能：关键路径延迟需要小于1ns
    2. 功耗：总功耗需要控制在100mW以内
    3. 面积：单元密度不超过85%
    4. 拥塞：平均拥塞度不超过0.8
    """
    
    # 运行RAG流程
    result = controller.run(query)
    
    # 输出结果
    print("\n=== RAG Optimization Results ===")
    
    # 布局结果
    print("\nLayout Results:")
    layout = result['layout']
    print(f"Timing:")
    print(f"  Critical Path Delay: {layout['timing']['critical_path_delay']:.3f}ns")
    print(f"  Timing Slack: {layout['timing']['timing_slack']:.3f}ns")
    print(f"Power:")
    print(f"  Total Power: {layout['power']['total_power']:.3f}mW")
    print(f"  Dynamic Power: {layout['power']['dynamic_power']:.3f}mW")
    print(f"Area:")
    print(f"  Total Area: {layout['area']['total_area']:.3f}um²")
    print(f"  Cell Density: {layout['area']['cell_density']:.2%}")
    print(f"Congestion:")
    print(f"  Average Congestion: {layout['congestion']['average_congestion']:.3f}")
    print(f"  Max Congestion: {layout['congestion']['max_congestion']:.3f}")
    
    # 评估结果
    print("\nEvaluation Results:")
    evaluation = result['evaluation']
    print(f"Total Score: {evaluation['total_score']:.3f}")
    print("\nObjective Scores:")
    for obj, score in evaluation['scores'].items():
        print(f"  {obj.capitalize()}: {score:.3f}")
    print("\nConstraint Satisfaction:")
    for constr, satisfied in evaluation['constraints'].items():
        print(f"  {constr.capitalize()}: {'✓' if satisfied else '✗'}")
        
    # 反馈信息
    print("\nFeedback:")
    feedback = result['feedback']
    if 'suggestions' in feedback:
        print("\nOptimization Suggestions:")
        for i, suggestion in enumerate(feedback['suggestions'], 1):
            print(f"{i}. {suggestion}")
            
if __name__ == '__main__':
    main() 