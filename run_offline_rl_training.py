#!/usr/bin/env python3
"""
离线强化学习训练启动脚本
从已有数据中学习布局参数优化策略
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入离线RL训练器
try:
    from modules.rl_training.offline_rl_trainer import OfflineRLTrainer
except ImportError as e:
    print(f"无法导入离线RL训练器: {e}")
    print("请确保相关依赖已安装")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('offline_rl_training.log'),
        logging.StreamHandler()
    ]
)

def main():
    """主函数"""
    print("=== 离线强化学习训练 ===")
    print("从已有数据中学习布局参数优化策略")
    
    # 配置参数
    data_dir = "results/parallel_training"
    model_save_dir = "models/offline_rl"
    
    print(f"数据目录: {data_dir}")
    print(f"模型保存目录: {model_save_dir}")
    
    # 检查数据目录
    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请先运行批量训练脚本生成训练数据")
        return
    
    # 创建离线RL训练器
    print("初始化离线RL训练器...")
    trainer = OfflineRLTrainer(data_dir, model_save_dir)
    
    # 加载训练数据
    print("加载训练数据...")
    df = trainer.load_training_data()
    
    if df.empty:
        print("❌ 没有找到有效的训练数据")
        print("请确保数据文件包含以下字段:")
        print("- parameters: 包含density_target, wirelength_weight, density_weight")
        print("- performance: 包含hpwl, overflow")
        print("- design_features: 包含num_instances, num_nets, num_pins")
        return
    
    print(f"✅ 成功加载 {len(df)} 条训练数据")
    
    # 显示数据样本
    print("\n数据样本预览:")
    print(df.head())
    
    # 预处理数据
    print("\n预处理训练数据...")
    X_features, X_actions, y_rewards = trainer.preprocess_data(df)
    
    if len(X_features) == 0:
        print("❌ 预处理后没有有效数据")
        return
    
    print(f"✅ 预处理完成: {len(X_features)} 个有效样本")
    print(f"特征维度: {X_features.shape}")
    print(f"动作维度: {X_actions.shape}")
    print(f"奖励范围: [{y_rewards.min():.2f}, {y_rewards.max():.2f}]")
    
    # 训练模型
    print("\n开始训练离线RL模型...")
    training_history = trainer.train_model(
        X_features, X_actions, y_rewards,
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    print("✅ 模型训练完成")
    
    # 保存模型
    print("保存训练好的模型...")
    trainer.save_model()
    
    # 生成训练报告
    print("生成训练报告...")
    trainer.generate_training_report(training_history, Path(model_save_dir))
    
    # 测试参数预测
    print("\n测试参数预测功能...")
    
    # 测试不同规模的设计
    test_designs = [
        {
            'name': '小型设计',
            'features': {'num_instances': 10000, 'num_nets': 12000, 'num_pins': 500}
        },
        {
            'name': '中型设计', 
            'features': {'num_instances': 50000, 'num_nets': 60000, 'num_pins': 1000}
        },
        {
            'name': '大型设计',
            'features': {'num_instances': 100000, 'num_nets': 120000, 'num_pins': 2000}
        }
    ]
    
    print("\n参数预测结果:")
    for test_design in test_designs:
        optimal_params = trainer.predict_optimal_parameters(test_design['features'])
        print(f"\n{test_design['name']}:")
        print(f"  设计特征: {test_design['features']}")
        print(f"  推荐参数: {optimal_params.to_dict()}")
    
    # 生成使用指南
    print("\n=== 使用指南 ===")
    print("1. 训练好的模型已保存到:", model_save_dir)
    print("2. 可以在其他脚本中加载模型进行参数预测:")
    print("   from modules.rl_training.offline_rl_trainer import OfflineRLTrainer")
    print("   trainer = OfflineRLTrainer()")
    print("   trainer.load_model()")
    print("   params = trainer.predict_optimal_parameters(design_features)")
    
    print("\n✅ 离线RL训练完成！")

if __name__ == "__main__":
    main() 