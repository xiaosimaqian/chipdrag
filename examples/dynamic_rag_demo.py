#!/usr/bin/env python3
"""
Chip-D-RAG 系统演示脚本
展示动态RAG系统的核心功能
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import random
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 使用绝对导入
from modules.core.rl_agent import QLearningAgent, StateExtractor, RewardCalculator, State, Action
from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever
from experiments.dynamic_rag_experiment import DynamicRAGExperiment

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def demo_rl_agent():
    """演示强化学习智能体"""
    print("\n" + "="*50)
    print("强化学习智能体演示")
    print("="*50)
    
    # 配置
    config = {
        'alpha': 0.01,
        'gamma': 0.95,
        'epsilon': 0.9,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'k_range': (3, 15),
        'buffer_size': 1000
    }
    
    # 初始化智能体
    agent = QLearningAgent(config)
    state_extractor = StateExtractor(config)
    reward_calculator = RewardCalculator(config)
    
    print(f"智能体初始化完成，动作空间: {agent.action_space}")
    print(f"初始探索率: {agent.epsilon}")
    
    # 模拟一些训练episodes
    print("\n开始训练演示...")
    
    for episode in range(10):
        # 创建模拟状态
        state = State(
            query_complexity=random.uniform(0.3, 0.9),
            design_type=random.choice(['risc_v', 'dsp', 'memory']),
            constraint_count=random.randint(2, 8),
            initial_relevance=random.uniform(0.5, 0.9),
            result_diversity=random.uniform(0.2, 0.8),
            historical_performance=random.uniform(0.4, 0.8),
            timestamp=datetime.now().isoformat()
        )
        
        # 选择动作
        action = agent.choose_action(state)
        
        # 模拟奖励
        reward = random.uniform(0.6, 0.9)
        
        # 创建下一状态
        next_state = State(
            query_complexity=random.uniform(0.3, 0.9),
            design_type=state.design_type,
            constraint_count=state.constraint_count,
            initial_relevance=random.uniform(0.5, 0.9),
            result_diversity=random.uniform(0.2, 0.8),
            historical_performance=random.uniform(0.4, 0.8),
            timestamp=datetime.now().isoformat()
        )
        
        # 更新智能体
        agent.update(state, action, reward, next_state)
        
        print(f"Episode {episode+1}: 动作={action.k_value}, 奖励={reward:.3f}, 探索率={agent.epsilon:.3f}")
    
    # 显示训练统计
    stats = agent.get_training_stats()
    q_stats = agent.get_q_table_stats()
    
    print(f"\n训练统计:")
    print(f"  Q表大小: {q_stats['total_states']} 个状态")
    print(f"  总动作数: {q_stats['total_actions']}")
    print(f"  最终探索率: {stats['exploration_rate']:.3f}")

def demo_dynamic_retrieval():
    """演示动态检索"""
    print("\n" + "="*50)
    print("动态检索演示")
    print("="*50)
    
    # 配置
    config = {
        'dynamic_k_range': [3, 15],
        'quality_threshold': 0.7,
        'learning_rate': 0.01,
        'compressed_entity_dim': 128,
        'entity_compression_ratio': 0.1,
        'entity_similarity_threshold': 0.8,
        'knowledge_base': {
            'path': 'data/knowledge_base/sample_kb.json',
            'format': 'json'
        }
    }
    
    # 初始化检索器
    retriever = DynamicRAGRetriever(config)
    
    # 模拟查询
    query = {
        'text': 'Generate layout for RISC-V processor with timing and power constraints',
        'design_type': 'risc_v',
        'constraints': ['timing', 'power'],
        'complexity': 0.8
    }
    
    design_info = {
        'design_type': 'risc_v',
        'technology_node': '14nm',
        'area_constraint': 5000,
        'power_budget': 5.0,
        'timing_constraint': 2.0,
        'constraints': [
            {'type': 'timing', 'value': 0.9, 'weight': 0.4},
            {'type': 'power', 'value': 0.8, 'weight': 0.3}
        ]
    }
    
    print(f"查询: {query['text']}")
    print(f"设计类型: {design_info['design_type']}")
    print(f"技术节点: {design_info['technology_node']}")
    
    # 执行检索
    try:
        results = retriever.retrieve_with_dynamic_reranking(query, design_info)
        print(f"\n检索完成，返回 {len(results)} 个结果")
        
        # 显示检索统计
        stats = retriever.get_retrieval_statistics()
        print(f"检索统计: {stats}")
        
    except Exception as e:
        print(f"检索失败: {str(e)}")

def demo_experiment():
    """演示实验功能"""
    print("\n" + "="*50)
    print("实验功能演示")
    print("="*50)
    
    # 配置
    config = {
        'experiment_name': 'demo_experiment',
        'num_runs': 3,
        'test_data_size': 20
    }
    
    # 初始化实验设计器
    experiment = DynamicRAGExperiment(config)
    
    # 生成测试数据
    test_data = []
    for i in range(10):
        test_data.append({
            'query': {
                'text': f'Generate layout for design {i}',
                'design_type': random.choice(['risc_v', 'dsp', 'memory']),
                'constraints': ['timing', 'power'],
                'complexity': random.uniform(0.5, 0.9)
            },
            'design_info': {
                'design_type': 'risc_v',
                'technology_node': '14nm',
                'area_constraint': random.uniform(1000, 10000),
                'power_budget': random.uniform(1.0, 10.0),
                'timing_constraint': random.uniform(1.0, 5.0),
                'constraints': [
                    {'type': 'timing', 'value': random.uniform(0.7, 1.0), 'weight': 0.4},
                    {'type': 'power', 'value': random.uniform(0.7, 1.0), 'weight': 0.3}
                ]
            }
        })
    
    print(f"生成 {len(test_data)} 个测试样本")
    
    # 运行对比实验
    print("\n运行对比实验...")
    results = experiment.run_comparison_experiment(test_data)
    
    # 显示结果
    for method, method_results in results.items():
        scores = [r.overall_score for r in method_results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {method}: {mean_score:.3f} ± {std_score:.3f}")
    
    # 运行消融实验
    print("\n运行消融实验...")
    ablation_results = experiment.run_ablation_study(test_data[:5])
    
    for config_name, result in ablation_results.items():
        print(f"  {config_name}: {result.results.overall_score:.3f}")

def demo_case_study():
    """演示案例分析"""
    print("\n" + "="*50)
    print("案例分析演示")
    print("="*50)
    
    # 创建具体案例
    case = {
        'name': 'RISC-V处理器布局优化',
        'query': {
            'text': 'Generate optimal layout for RISC-V processor with 5-stage pipeline and low power consumption',
            'design_type': 'risc_v',
            'constraints': ['timing', 'power', 'area'],
            'complexity': 0.85,
            'priority': 'high'
        },
        'design_info': {
            'design_type': 'risc_v',
            'technology_node': '14nm',
            'area_constraint': 5000,
            'power_budget': 3.0,
            'timing_constraint': 2.5,
            'constraints': [
                {'type': 'timing', 'value': 0.95, 'weight': 0.4},
                {'type': 'power', 'value': 0.9, 'weight': 0.4},
                {'type': 'area', 'value': 0.8, 'weight': 0.2}
            ]
        }
    }
    
    print(f"案例: {case['name']}")
    print(f"查询: {case['query']['text']}")
    print(f"设计类型: {case['design_info']['design_type']}")
    print(f"技术节点: {case['design_info']['technology_node']}")
    print(f"约束数量: {len(case['design_info']['constraints'])}")
    
    # 运行不同方法
    config = {'experiment_name': 'case_study_demo'}
    experiment = DynamicRAGExperiment(config)
    
    methods = ['TraditionalRAG', 'ChipRAG', 'Chip-D-RAG']
    
    print("\n方法对比结果:")
    print("| 方法 | 整体分数 | 布局质量 | 约束满足度 | 执行时间 |")
    print("|------|----------|----------|------------|----------|")
    
    for method in methods:
        try:
            if method == 'TraditionalRAG':
                result = experiment._run_traditional_rag(case['query'], case['design_info'])
            elif method == 'ChipRAG':
                result = experiment._run_chiprag(case['query'], case['design_info'])
            elif method == 'Chip-D-RAG':
                result = experiment._run_chip_d_rag(case['query'], case['design_info'])
            
            print(f"| {method} | {result['overall_score']:.3f} | {result['layout_quality']:.3f} | {result['constraint_satisfaction']:.3f} | {result.get('execution_time', 0.0):.3f}s |")
            
        except Exception as e:
            print(f"| {method} | 0.000 | 0.000 | 0.000 | 0.000s | (错误: {str(e)})")

def main():
    """主函数"""
    print("Chip-D-RAG 系统演示")
    print("="*60)
    
    # 设置日志
    setup_logging()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    try:
        # 1. 强化学习智能体演示
        demo_rl_agent()
        
        # 2. 动态检索演示
        demo_dynamic_retrieval()
        
        # 3. 实验功能演示
        demo_experiment()
        
        # 4. 案例分析演示
        demo_case_study()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        
        print("\n系统特点总结:")
        print("1. 强化学习智能体能够动态调整检索策略")
        print("2. 动态重排序机制提升检索质量")
        print("3. 实体增强技术改善布局生成")
        print("4. 质量反馈机制确保持续改进")
        print("5. 完整的实验框架支持性能评估")
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 