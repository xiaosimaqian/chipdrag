import os
import sys
import time
from datetime import datetime
from chiprag.modules.experiments import ExperimentEvaluator

def main():
    print("开始运行实验...")
    
    # 创建实验评估器
    evaluator = ExperimentEvaluator()
    
    # 运行所有实验
    results = evaluator.run_all_experiments()
    
    # 打印结果摘要
    print("\n实验结果摘要:")
    print("=" * 50)
    
    # 布局质量
    print("\n布局质量:")
    print("-" * 30)
    for method, metrics in results['layout_quality'].items():
        print(f"\n{method}:")
        print(f"  密度: {metrics['density']:.4f}")
        print(f"  拥塞度: {metrics['congestion']:.4f}")
        print(f"  时序裕量: {metrics['timing_margin']:.4f}")
    
    # 约束满足率
    print("\n约束满足率:")
    print("-" * 30)
    for method, benchmarks in results['constraint_satisfaction'].items():
        print(f"\n{method}:")
        for benchmark, satisfaction in benchmarks.items():
            print(f"  {benchmark}: {satisfaction:.4f}")
    
    # 优化效率
    print("\n优化效率:")
    print("-" * 30)
    for method, metrics in results['optimization_efficiency'].items():
        print(f"\n{method}:")
        print(f"  生成时间: {metrics['generation_time']:.2f}秒")
        print(f"  迭代次数: {metrics['iterations']}")
        print(f"  总时间: {metrics['total_time']:.2f}秒")

if __name__ == "__main__":
    main() 