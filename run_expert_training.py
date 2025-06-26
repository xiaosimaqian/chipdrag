#!/usr/bin/env python3
"""
专家指导强化学习训练启动脚本
使用floorplan.def作为初始布局，mgc_des_perf_1_place.def作为专家布局数据
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_simple_demo(design_dir: str, config: Dict[str, Any]):
    """运行简化演示"""
    logger.info("运行简化专家训练演示...")
    
    # 导入简化训练模块
    from simple_expert_training_demo import run_simple_expert_training
    
    # 获取简化演示配置
    demo_config = config.get('simple_demo', {})
    
    # 运行训练
    run_simple_expert_training(design_dir, demo_config)

def run_full_training(design_dir: str, config: Dict[str, Any]):
    """运行完整训练"""
    logger.info("运行完整专家指导训练...")
    
    # 导入完整训练模块
    from enhanced_rl_training_with_expert import train_with_expert_guidance
    
    # 获取完整训练配置
    training_config = config.get('expert_training', {})
    
    # 运行训练
    train_with_expert_guidance(design_dir, training_config)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='专家指导强化学习训练')
    parser.add_argument('--design-dir', type=str, 
                       default='data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1',
                       help='设计目录路径')
    parser.add_argument('--config', type=str,
                       default='configs/expert_training_config.json',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['demo', 'full'],
                       default='demo',
                       help='训练模式: demo(简化演示) 或 full(完整训练)')
    
    args = parser.parse_args()
    
    # 构建完整路径
    project_root = Path(__file__).parent
    design_path = project_root / args.design_dir
    config_path = project_root / args.config
    
    # 检查路径
    if not design_path.exists():
        logger.error(f"设计目录不存在: {design_path}")
        return
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    # 加载配置
    config = load_config(str(config_path))
    
    logger.info("=================================================")
    logger.info("=== 专家指导强化学习训练启动 ===")
    logger.info("=================================================")
    logger.info(f"设计目录: {design_path}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"训练模式: {args.mode}")
    
    # 根据模式运行训练
    if args.mode == 'demo':
        run_simple_demo(str(design_path), config)
    else:
        run_full_training(str(design_path), config)
    
    logger.info("训练完成!")

if __name__ == '__main__':
    main() 