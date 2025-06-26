#!/usr/bin/env python3
"""
专家训练系统测试脚本
验证系统各个组件是否正常工作
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_file_structure():
    """测试文件结构"""
    logger.info("=== 测试文件结构 ===")
    
    required_files = [
        "enhanced_rl_training_with_expert.py",
        "simple_expert_training_demo.py", 
        "run_expert_training.py",
        "configs/expert_training_config.json",
        "docs/EXPERT_TRAINING_GUIDE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            logger.info(f"✓ {file_path}")
    
    if missing_files:
        logger.error(f"缺少文件: {missing_files}")
        return False
    
    logger.info("文件结构测试通过")
    return True

def test_design_data():
    """测试设计数据"""
    logger.info("=== 测试设计数据 ===")
    
    design_dir = project_root / "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    
    if not design_dir.exists():
        logger.error(f"设计目录不存在: {design_dir}")
        return False
    
    required_design_files = [
        "floorplan.def",
        "mgc_des_perf_1_place.def", 
        "design.v",
        "cells.lef",
        "tech.lef"
    ]
    
    missing_files = []
    for file_name in required_design_files:
        file_path = design_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            file_size = file_path.stat().st_size
            logger.info(f"✓ {file_name} ({file_size / 1024 / 1024:.1f} MB)")
    
    if missing_files:
        logger.error(f"缺少设计文件: {missing_files}")
        return False
    
    logger.info("设计数据测试通过")
    return True

def test_config_loading():
    """测试配置文件加载"""
    logger.info("=== 测试配置文件加载 ===")
    
    config_path = project_root / "configs/expert_training_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 检查配置结构
        required_sections = ['expert_training', 'simple_demo']
        for section in required_sections:
            if section not in config:
                logger.error(f"配置缺少部分: {section}")
                return False
        
        logger.info("✓ 配置文件加载成功")
        logger.info(f"  - 专家训练配置: {len(config['expert_training'])} 个部分")
        logger.info(f"  - 简化演示配置: {len(config['simple_demo'])} 个部分")
        
        return True
        
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return False

def test_imports():
    """测试模块导入"""
    logger.info("=== 测试模块导入 ===")
    
    try:
        # 测试基础模块导入
        import torch
        import numpy as np
        logger.info("✓ 基础模块导入成功")
        
        # 测试项目模块导入
        from modules.parsers.def_parser import parse_def
        from modules.parsers.design_parser import parse_verilog
        logger.info("✓ 项目模块导入成功")
        
        # 测试专家训练模块导入
        from enhanced_rl_training_with_expert import ExpertDataManager, EnhancedDesignEnvironment
        logger.info("✓ 专家训练模块导入成功")
        
        return True
        
    except ImportError as e:
        logger.error(f"模块导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"导入测试出错: {e}")
        return False

def test_expert_data_parsing():
    """测试专家数据解析"""
    logger.info("=== 测试专家数据解析 ===")
    
    try:
        design_dir = project_root / "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
        
        # 测试DEF解析
        floorplan_def = design_dir / "floorplan.def"
        expert_def = design_dir / "mgc_des_perf_1_place.def"
        
        from modules.parsers.def_parser import parse_def
        from modules.parsers.design_parser import parse_verilog
        
        floorplan_metrics = parse_def(str(floorplan_def))
        expert_metrics = parse_def(str(expert_def))
        
        logger.info(f"✓ Floorplan解析成功: {floorplan_metrics.get('num_components', 0)} 个组件")
        logger.info(f"✓ Expert解析成功: {expert_metrics.get('num_components', 0)} 个组件")
        
        # 比较两个DEF文件
        floorplan_area = floorplan_metrics.get('die_area_microns', (0, 0))
        expert_area = expert_metrics.get('die_area_microns', (0, 0))
        
        logger.info(f"  - Floorplan面积: {floorplan_area[0]} x {floorplan_area[1]}")
        logger.info(f"  - Expert面积: {expert_area[0]} x {expert_area[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"专家数据解析失败: {e}")
        return False

def test_network_creation():
    """测试网络创建"""
    logger.info("=== 测试网络创建 ===")
    
    try:
        import torch
        from enhanced_rl_training_with_expert import ExpertGuidedActorCritic
        
        # 创建网络
        state_dim = 8
        action_dim = 13
        network = ExpertGuidedActorCritic(state_dim, action_dim)
        
        # 测试前向传播
        dummy_state = torch.randn(1, state_dim)
        action_probs, state_value, expert_probs = network(dummy_state)
        
        logger.info(f"✓ 网络创建成功")
        logger.info(f"  - 动作概率形状: {action_probs.shape}")
        logger.info(f"  - 状态价值形状: {state_value.shape}")
        logger.info(f"  - 专家概率形状: {expert_probs.shape}")
        
        # 检查输出合理性
        assert action_probs.sum().item() > 0.99, "动作概率和不为1"
        assert expert_probs.sum().item() > 0.99, "专家概率和不为1"
        
        return True
        
    except Exception as e:
        logger.error(f"网络创建失败: {e}")
        return False

def test_simple_demo():
    """测试简化演示"""
    logger.info("=== 测试简化演示 ===")
    
    try:
        from simple_expert_training_demo import SimpleExpertDataManager, SimpleExpertEnvironment
        
        design_dir = project_root / "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
        
        # 创建专家数据管理器
        expert_data = SimpleExpertDataManager(str(design_dir))
        logger.info("✓ 专家数据管理器创建成功")
        
        # 创建环境
        env = SimpleExpertEnvironment(str(design_dir), expert_data)
        logger.info("✓ 简化环境创建成功")
        
        # 测试状态获取
        state = env.get_state()
        logger.info(f"✓ 状态获取成功，维度: {state.shape}")
        
        # 测试动作执行
        action_k = 6
        next_state, reward, done, info = env.step(action_k)
        logger.info(f"✓ 动作执行成功，奖励: {reward:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"简化演示测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("=================================================")
    logger.info("=== 专家训练系统测试开始 ===")
    logger.info("=================================================")
    
    tests = [
        ("文件结构", test_file_structure),
        ("设计数据", test_design_data),
        ("配置加载", test_config_loading),
        ("模块导入", test_imports),
        ("专家数据解析", test_expert_data_parsing),
        ("网络创建", test_network_creation),
        ("简化演示", test_simple_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} 测试通过")
            else:
                logger.error(f"✗ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {e}")
    
    logger.info("=================================================")
    logger.info(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！系统可以正常使用。")
        logger.info("\n下一步:")
        logger.info("1. 运行简化演示: python run_expert_training.py --mode demo")
        logger.info("2. 运行完整训练: python run_expert_training.py --mode full")
        logger.info("3. 查看使用指南: docs/EXPERT_TRAINING_GUIDE.md")
    else:
        logger.error("❌ 部分测试失败，请检查系统配置。")
    
    return passed == total

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 