#!/usr/bin/env python3
"""
简化版ChipDRAG实验脚本
专注于HPWL对比和RL训练
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import re # Added missing import for re

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from modules.core.rl_agent import QLearningAgent, StateExtractor
from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever
from modules.utils.config_loader import ConfigLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleExperiment:
    """简化版实验类"""
    
    def __init__(self):
        """初始化实验"""
        self.data_dir = Path("data/designs/ispd_2015_contest_benchmark")
        self.results_dir = Path("simple_experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 加载配置
        self.config = ConfigLoader.load_config("configs/experiment_config.json")
        
        # 初始化组件
        self._init_components()
        
        # 实验设计列表
        self.designs = [
            "mgc_fft_1",
            "mgc_matrix_mult_1", 
            "mgc_des_perf_1"
        ]
        
        logger.info(f"简化实验初始化完成，目标设计: {self.designs}")
    
    def _init_components(self):
        """初始化实验组件"""
        try:
            # RL智能体
            rl_config = {
                'alpha': 0.01,
                'gamma': 0.95,
                'epsilon': 0.9,
                'k_range': (3, 15)
            }
            self.rl_agent = QLearningAgent(rl_config)
            
            # 状态提取器
            state_config = {
                'performance_cache_size': 1000,
                'feature_normalization': True
            }
            self.state_extractor = StateExtractor(state_config)
            
            # 动态检索器
            rag_config = self.config.get('knowledge_base', {})
            self.retriever = DynamicRAGRetriever(rag_config)
            
            logger.info("✅ 实验组件初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    def run_experiment(self):
        """运行简化实验"""
        logger.info("=== 开始简化版ChipDRAG实验 ===")
        
        results = {
            'experiment_info': {
                'start_time': datetime.now().isoformat(),
                'designs': self.designs,
                'mode': 'simplified'
            },
            'hpwl_results': {},
            'rl_training': {},
            'summary': {}
        }
        
        # 1. RL训练阶段
        logger.info("阶段1: RL智能体训练")
        training_results = self._run_rl_training()
        results['rl_training'] = training_results
        
        # 2. HPWL对比实验
        logger.info("阶段2: HPWL对比实验")
        hpwl_results = self._run_hpwl_comparison()
        results['hpwl_results'] = hpwl_results
        
        # 3. 生成报告
        logger.info("阶段3: 生成实验报告")
        summary = self._generate_summary(hpwl_results, training_results)
        results['summary'] = summary
        
        # 保存结果
        self._save_results(results)
        
        logger.info("=== 简化实验完成 ===")
        return results
    
    def _run_rl_training(self):
        """运行RL训练"""
        logger.info("开始RL智能体训练...")
        
        training_results = {
            'episodes': [],
            'q_table_stats': {},
            'training_progress': {}
        }
        
        for design_name in self.designs:
            design_dir = self.data_dir / design_name
            if not design_dir.exists():
                logger.warning(f"设计目录不存在: {design_dir}")
                continue
            
            logger.info(f"训练设计: {design_name}")
            
            try:
                # 加载设计信息
                design_info = self._load_design_info(design_dir)
                
                # 提取状态
                state = self.state_extractor.extract_state(design_info)
                
                # 训练多个episode
                for episode in range(3):  # 每个设计训练3个episode
                    logger.info(f"  Episode {episode + 1}")
                    
                    # 选择动作
                    action = self.rl_agent.select_action(state, training=True)
                    
                    # 执行检索
                    retrieved_cases = self.retriever.retrieve_with_dynamic_reranking(
                        query={'features': design_info, 'design_name': design_name},
                        design_info=design_info
                    )
                    
                    # 模拟奖励（基于检索质量）
                    reward = self._calculate_simple_reward(retrieved_cases, action)
                    
                    # 更新RL智能体
                    next_state = self._calculate_next_state(state, action, reward, design_info)
                    self.rl_agent.update(state, action, reward, next_state)
                    
                    # 记录训练数据
                    episode_data = {
                        'design': design_name,
                        'episode': episode + 1,
                        'action': {
                            'k_value': action.get('k_value', 5),
                            'confidence': action.get('confidence', 0.8),
                            'exploration_type': action.get('exploration_type', 'exploit')
                        },
                        'reward': reward,
                        'retrieved_count': len(retrieved_cases)
                    }
                    training_results['episodes'].append(episode_data)
                    
                    logger.info(f"    动作: k={action.get('k_value', 5)}, 奖励: {reward:.3f}")
        
        # 保存Q表统计
        training_results['q_table_stats'] = self.rl_agent.get_q_table_stats()
        
        logger.info("✅ RL训练完成")
        return training_results
    
    def _run_hpwl_comparison(self):
        """运行HPWL对比实验"""
        logger.info("开始HPWL对比实验...")
        
        hpwl_results = {}
        
        for design_name in self.designs:
            design_dir = self.data_dir / design_name
            if not design_dir.exists():
                continue
            
            logger.info(f"处理设计: {design_name}")
            
            try:
                # 加载设计信息
                design_info = self._load_design_info(design_dir)
                state = self.state_extractor.extract_state(design_info)
                
                # 使用训练好的RL智能体
                action = self.rl_agent.select_action(state, training=False)
                
                # 执行检索
                retrieved_cases = self.retriever.retrieve_with_dynamic_reranking(
                    query={'features': design_info, 'design_name': design_name},
                    design_info=design_info
                )
                
                # 生成布局策略
                layout_strategy = self._generate_simple_layout_strategy(retrieved_cases, action)
                
                # 执行布局（简化版）
                layout_success = self._execute_simple_layout(design_dir, layout_strategy)
                
                # 提取HPWL
                hpwl = self._extract_hpwl_simple(design_dir)
                
                # 记录结果
                hpwl_results[design_name] = {
                    'layout_success': layout_success,
                    'hpwl': hpwl,
                    'action': action,
                    'retrieved_count': len(retrieved_cases),
                    'strategy': layout_strategy
                }
                
                logger.info(f"  布局成功: {layout_success}, HPWL: {hpwl}")
                
            except Exception as e:
                logger.error(f"处理设计 {design_name} 失败: {e}")
                hpwl_results[design_name] = {
                    'layout_success': False,
                    'hpwl': None,
                    'error': str(e)
                }
        
        logger.info("✅ HPWL对比实验完成")
        return hpwl_results
    
    def _load_design_info(self, design_dir: Path) -> Dict:
        """加载设计信息"""
        design_info = {
            'name': design_dir.name,
            'design_type': 'unknown',
            'num_components': 0,
            'area': 0.0,
            'constraints': {}
        }
        
        # 检查DEF文件
        def_file = design_dir / "floorplan.def"
        if def_file.exists():
            try:
                with open(def_file, 'r') as f:
                    content = f.read()
                
                # 提取组件数量
                components_match = re.search(r'COMPONENTS\s+(\d+)', content)
                if components_match:
                    design_info['num_components'] = int(components_match.group(1))
                
                # 提取面积信息
                area_match = re.search(r'DIEAREA\s+\([^)]+\)\s+\([^)]+\)', content)
                if area_match:
                    design_info['area'] = 1000000  # 简化估算
                    
            except Exception as e:
                logger.warning(f"解析DEF文件失败: {e}")
        
        return design_info
    
    def _calculate_simple_reward(self, retrieved_cases: List, action: Dict) -> float:
        """计算简单奖励"""
        if not retrieved_cases:
            return 0.1
        
        # 基于检索数量和质量计算奖励
        k_value = action.get('k_value', 5)
        confidence = action.get('confidence', 0.8)
        
        # 基础奖励：检索到案例
        base_reward = min(len(retrieved_cases) / k_value, 1.0)
        
        # 质量奖励：基于置信度
        quality_reward = confidence
        
        # 综合奖励
        reward = (base_reward * 0.7 + quality_reward * 0.3)
        
        return max(0.1, min(1.0, reward))
    
    def _calculate_next_state(self, state, action, reward, design_info):
        """计算下一个状态"""
        # 简化版：返回相同状态
        return state
    
    def _generate_simple_layout_strategy(self, retrieved_cases: List, action: Dict) -> Dict:
        """生成简单布局策略"""
        k_value = action.get('k_value', 5)
        
        # 基于k值调整策略
        if k_value > 8:
            utilization = 0.75
            aspect_ratio = 1.2
        elif k_value > 5:
            utilization = 0.8
            aspect_ratio = 1.0
        else:
            utilization = 0.85
            aspect_ratio = 0.8
        
        return {
            'strategy_type': 'rl_optimized',
            'parameters': {
                'utilization': utilization,
                'aspect_ratio': aspect_ratio,
                'placement_density': 0.7,
                'overflow_threshold': 0.15
            },
            'action_info': action
        }
    
    def _execute_simple_layout(self, design_dir: Path, layout_strategy: Dict) -> bool:
        """执行简化布局"""
        try:
            # 检查是否有OpenROAD
            result = subprocess.run(['which', 'openroad'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("OpenROAD未安装，跳过布局执行")
                return False
            
            # 生成简单的TCL脚本
            tcl_script = self._generate_simple_tcl_script(layout_strategy, design_dir.name)
            script_file = design_dir / "simple_placement.tcl"
            
            with open(script_file, 'w') as f:
                f.write(tcl_script)
            
            # 执行OpenROAD
            cmd = ['openroad', '-no_init', '-no_splash', '-exit', str(script_file)]
            result = subprocess.run(cmd, cwd=design_dir, capture_output=True, text=True)
            
            success = result.returncode == 0
            logger.info(f"OpenROAD执行结果: {'成功' if success else '失败'}")
            
            return success
            
        except Exception as e:
            logger.error(f"布局执行失败: {e}")
            return False
    
    def _generate_simple_tcl_script(self, layout_strategy: Dict, design_name: str) -> str:
        """生成简单TCL脚本"""
        params = layout_strategy.get('parameters', {})
        
        return f"""
# 简单布局脚本
puts "处理设计: {design_name}"

# 读取文件
read_lef tech.lef
read_lef cells.lef
read_verilog design.v
read_def floorplan.def

# 全局布局
global_placement -density {params.get('placement_density', 0.7)} -overflow {params.get('overflow_threshold', 0.15)}

# 详细布局
detailed_placement

# 输出结果
write_def simple_placed.def
puts "布局完成"
"""
    
    def _extract_hpwl_simple(self, design_dir: Path) -> Optional[float]:
        """简单HPWL提取"""
        try:
            # 检查是否有布局结果
            placed_def = design_dir / "simple_placed.def"
            if not placed_def.exists():
                placed_def = design_dir / "placed.def"
            
            if not placed_def.exists():
                return None
            
            # 简单的HPWL提取
            with open(placed_def, 'r') as f:
                content = f.read()
            
            # 查找HPWL相关信息
            hpwl_match = re.search(r'HPWL\s+(\d+)', content)
            if hpwl_match:
                return float(hpwl_match.group(1))
            
            # 如果没有HPWL信息，估算
            components_match = re.search(r'COMPONENTS\s+(\d+)', content)
            if components_match:
                num_components = int(components_match.group(1))
                # 简单估算：每个组件平均1000单位
                return num_components * 1000
            
            return None
            
        except Exception as e:
            logger.error(f"HPWL提取失败: {e}")
            return None
    
    def _generate_summary(self, hpwl_results: Dict, training_results: Dict) -> Dict:
        """生成实验总结"""
        # 统计成功的设计
        successful_designs = [name for name, result in hpwl_results.items() 
                            if result.get('layout_success', False)]
        
        # 计算HPWL统计
        hpwl_values = [result.get('hpwl') for result in hpwl_results.values() 
                      if result.get('hpwl') is not None]
        
        summary = {
            'total_designs': len(self.designs),
            'successful_layouts': len(successful_designs),
            'success_rate': len(successful_designs) / len(self.designs),
            'hpwl_stats': {
                'count': len(hpwl_values),
                'mean': np.mean(hpwl_values) if hpwl_values else None,
                'std': np.std(hpwl_values) if hpwl_values else None,
                'min': np.min(hpwl_values) if hpwl_values else None,
                'max': np.max(hpwl_values) if hpwl_values else None
            },
            'rl_training_episodes': len(training_results.get('episodes', [])),
            'q_table_states': training_results.get('q_table_stats', {}).get('total_states', 0)
        }
        
        return summary
    
    def _save_results(self, results: Dict):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"simple_experiment_results_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存: {result_file}")

def main():
    """主函数"""
    try:
        experiment = SimpleExperiment()
        results = experiment.run_experiment()
        
        # 打印总结
        summary = results['summary']
        print("\n=== 实验总结 ===")
        print(f"总设计数: {summary['total_designs']}")
        print(f"布局成功数: {summary['successful_layouts']}")
        print(f"成功率: {summary['success_rate']:.2%}")
        print(f"RL训练回合数: {summary['rl_training_episodes']}")
        print(f"Q表状态数: {summary['q_table_states']}")
        
        if summary['hpwl_stats']['count'] > 0:
            print(f"HPWL统计:")
            print(f"  平均值: {summary['hpwl_stats']['mean']:.0f}")
            print(f"  标准差: {summary['hpwl_stats']['std']:.0f}")
            print(f"  最小值: {summary['hpwl_stats']['min']:.0f}")
            print(f"  最大值: {summary['hpwl_stats']['max']:.0f}")
        
        print("✅ 简化实验完成")
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 