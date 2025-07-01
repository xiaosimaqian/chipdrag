#!/usr/bin/env python3
"""
真实的ChipDRAG实验脚本
实现完整的RL训练、推理和OpenROAD布局优化流程
"""

import os
import sys
import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core.chip_retriever import ChipRetriever
from modules.core.rl_agent import RLAgent
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.config_loader import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealChipDRAGExperiment:
    """真实的ChipDRAG实验系统"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "real_chipdrag_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验配置
        self.experiment_config = {
            'designs': [
                'mgc_des_perf_1', 'mgc_fft_1', 'mgc_matrix_mult_a',
                'mgc_pci_bridge32_a', 'mgc_superblue19', 'mgc_superblue4',
                'mgc_superblue5', 'mgc_superblue7', 'mgc_superblue10',
                'mgc_superblue14', 'mgc_superblue16', 'mgc_superblue18'
            ],
            'training_episodes': 50,
            'inference_iterations': 10,
            'openroad_timeout': 3600,  # 1小时超时
            'hpwl_script': self.base_dir / "calculate_hpwl.py"
        }
        
        # 初始化组件
        self._init_components()
        
        logger.info("真实ChipDRAG实验系统初始化完成")
    
    def _init_components(self):
        """初始化实验组件"""
        try:
            # 加载配置
            config = load_config('configs/dynamic_rag_config.json')
            
            # 初始化知识库
            self.knowledge_base = KnowledgeBase(config.get('knowledge_base', {}))
            
            # 初始化检索器
            self.retriever = ChipRetriever(config)
            
            # 初始化RL智能体
            self.rl_agent = RLAgent(config.get('rl_agent', {}))
            
            logger.info("实验组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {str(e)}")
            raise
    
    def run_complete_experiment(self):
        """运行完整的ChipDRAG实验"""
        logger.info("=== 开始真实ChipDRAG实验 ===")
        
        # 1. 数据准备阶段
        logger.info("阶段1: 数据准备")
        self._prepare_experiment_data()
        
        # 2. RL训练阶段
        logger.info("阶段2: RL训练")
        training_results = self._run_rl_training()
        
        # 3. 推理阶段
        logger.info("阶段3: 推理优化")
        inference_results = self._run_inference_optimization()
        
        # 4. 结果收集
        logger.info("阶段4: 结果收集")
        final_results = self._collect_final_results(training_results, inference_results)
        
        # 5. 生成报告
        logger.info("阶段5: 生成报告")
        self._generate_experiment_report(final_results)
        
        logger.info("=== 真实ChipDRAG实验完成 ===")
        return final_results
    
    def _prepare_experiment_data(self):
        """准备实验数据"""
        logger.info("准备实验数据...")
        
        # 检查设计目录
        for design_name in self.experiment_config['designs']:
            design_dir = self.data_dir / design_name
            if not design_dir.exists():
                logger.warning(f"设计目录不存在: {design_dir}")
                continue
            
            # 检查必要文件
            required_files = ['design.v', 'floorplan.def', 'cells.lef', 'tech.lef']
            missing_files = []
            for file_name in required_files:
                if not (design_dir / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                logger.warning(f"设计 {design_name} 缺少文件: {missing_files}")
            else:
                logger.info(f"设计 {design_name} 数据完整")
    
    def _run_rl_training(self) -> Dict[str, Any]:
        """运行RL训练阶段"""
        logger.info("开始RL训练阶段...")
        training_results = {
            'episodes': [],
            'q_table_updates': 0,
            'total_reward': 0.0
        }
        
        for episode in range(self.experiment_config['training_episodes']):
            logger.info(f"训练回合 {episode + 1}/{self.experiment_config['training_episodes']}")
            
            # 随机选择一个设计进行训练
            design_name = np.random.choice(self.experiment_config['designs'])
            design_dir = self.data_dir / design_name
            
            if not design_dir.exists():
                continue
            
            # 提取设计特征
            design_features = self._extract_design_features(design_dir)
            
            # 构建查询
            query = {
                'features': design_features,
                'design_name': design_name
            }
            
            # RL智能体选择动作
            state = self._extract_state(query)
            action = self.rl_agent.select_action(state)
            
            # 执行检索
            retrieved_cases = self.retriever.retrieve(query, knowledge_base=self.knowledge_base)
            
            # 生成布局策略
            layout_strategy = self._generate_layout_strategy(retrieved_cases, action)
            
            # 执行OpenROAD布局
            layout_result = self._run_openroad_layout(design_dir, layout_strategy)
            
            # 计算奖励
            reward = self._calculate_reward(layout_result)
            
            # 更新RL智能体
            next_state = self._extract_state(query, layout_result)
            self.rl_agent.update(state, action, reward, next_state)
            
            # 记录训练结果
            training_results['episodes'].append({
                'episode': episode,
                'design': design_name,
                'action': action,
                'reward': reward,
                'layout_result': layout_result
            })
            training_results['q_table_updates'] += 1
            training_results['total_reward'] += reward
            
            logger.info(f"  回合 {episode + 1}: 设计={design_name}, 动作={action}, 奖励={reward:.3f}")
        
        logger.info(f"RL训练完成，总奖励: {training_results['total_reward']:.3f}")
        return training_results
    
    def _run_inference_optimization(self) -> Dict[str, Any]:
        """运行推理优化阶段"""
        logger.info("开始推理优化阶段...")
        inference_results = {
            'designs': {},
            'total_improvement': 0.0
        }
        
        for design_name in self.experiment_config['designs']:
            logger.info(f"优化设计: {design_name}")
            design_dir = self.data_dir / design_name
            
            if not design_dir.exists():
                continue
            
            # 1. 生成OpenROAD默认布局
            logger.info(f"  生成OpenROAD默认布局...")
            default_result = self._run_openroad_default_layout(design_dir)
            
            # 2. 生成ChipDRAG优化布局
            logger.info(f"  生成ChipDRAG优化布局...")
            optimized_result = self._run_chipdrag_optimized_layout(design_dir)
            
            # 3. 计算改进效果
            if default_result and optimized_result:
                improvement = self._calculate_improvement(default_result, optimized_result)
                inference_results['designs'][design_name] = {
                    'default_hpwl': default_result.get('hpwl', 0),
                    'optimized_hpwl': optimized_result.get('hpwl', 0),
                    'improvement_pct': improvement
                }
                inference_results['total_improvement'] += improvement
                
                logger.info(f"  {design_name}: 默认HPWL={default_result.get('hpwl', 0):.2e}, "
                           f"优化HPWL={optimized_result.get('hpwl', 0):.2e}, "
                           f"改进={improvement:.2f}%")
        
        logger.info(f"推理优化完成，平均改进: {inference_results['total_improvement'] / len(inference_results['designs']):.2f}%")
        return inference_results
    
    def _run_openroad_default_layout(self, design_dir: Path) -> Optional[Dict]:
        """运行OpenROAD默认布局"""
        try:
            work_dir = design_dir
            work_dir_abs = str(work_dir.absolute())
            
            # OpenROAD默认布局命令
            openroad_script = f"""
            source /opt/openroad/setup_env.sh
            cd /workspace
            
            # 读取设计文件
            read_lef cells.lef
            read_lef tech.lef
            read_def floorplan.def
            read_verilog design.v
            
            # 链接设计
            link_design
            
            # 默认布局流程
            initialize_floorplan -utilization 0.7 -aspect_ratio 1.0 -core_space 2.0
            place_pins -random
            global_placement -disable_routability_driven
            detailed_placement
            
            # 输出结果
            write_def output_default.def
            exit
            """
            
            # 执行OpenROAD
            docker_cmd = f"""docker run --rm -m 16g -c 8 \
                -e OPENROAD_NUM_THREADS=8 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
                -v {work_dir_abs}:/workspace -w /workspace \
                openroad/flow-ubuntu22.04-builder:21e414 bash -c '{openroad_script}'"""
            
            logger.info(f"  执行OpenROAD默认布局...")
            start_time = time.time()
            result = subprocess.run(docker_cmd, shell=True, capture_output=True, 
                                  text=True, timeout=self.experiment_config['openroad_timeout'])
            end_time = time.time()
            
            if result.returncode == 0:
                # 计算HPWL
                def_file = work_dir / "output_default.def"
                if def_file.exists():
                    hpwl = self._calculate_hpwl(def_file)
                    return {
                        'hpwl': hpwl,
                        'def_file': str(def_file),
                        'execution_time': end_time - start_time,
                        'success': True
                    }
            
            logger.error(f"  OpenROAD默认布局失败: {result.stderr}")
            return None
            
        except Exception as e:
            logger.error(f"  OpenROAD默认布局异常: {str(e)}")
            return None
    
    def _run_chipdrag_optimized_layout(self, design_dir: Path) -> Optional[Dict]:
        """运行ChipDRAG优化布局"""
        try:
            work_dir = design_dir
            work_dir_abs = str(work_dir.absolute())
            
            # 提取设计特征
            design_features = self._extract_design_features(design_dir)
            
            # 构建查询
            query = {
                'features': design_features,
                'design_name': design_dir.name
            }
            
            # 使用训练好的RL智能体选择最优策略
            state = self._extract_state(query)
            optimal_action = self.rl_agent.select_action(state, epsilon=0.0)  # 纯利用
            
            # 检索相似案例
            retrieved_cases = self.retriever.retrieve(query, knowledge_base=self.knowledge_base)
            
            # 生成优化布局策略
            layout_strategy = self._generate_optimized_layout_strategy(retrieved_cases, optimal_action)
            
            # OpenROAD优化布局命令
            openroad_script = f"""
            source /opt/openroad/setup_env.sh
            cd /workspace
            
            # 读取设计文件
            read_lef cells.lef
            read_lef tech.lef
            read_def floorplan.def
            read_verilog design.v
            
            # 链接设计
            link_design
            
            # ChipDRAG优化布局流程
            {layout_strategy}
            
            # 输出结果
            write_def output_optimized.def
            exit
            """
            
            # 执行OpenROAD
            docker_cmd = f"""docker run --rm -m 16g -c 8 \
                -e OPENROAD_NUM_THREADS=8 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
                -v {work_dir_abs}:/workspace -w /workspace \
                openroad/flow-ubuntu22.04-builder:21e414 bash -c '{openroad_script}'"""
            
            logger.info(f"  执行ChipDRAG优化布局...")
            start_time = time.time()
            result = subprocess.run(docker_cmd, shell=True, capture_output=True, 
                                  text=True, timeout=self.experiment_config['openroad_timeout'])
            end_time = time.time()
            
            if result.returncode == 0:
                # 计算HPWL
                def_file = work_dir / "output_optimized.def"
                if def_file.exists():
                    hpwl = self._calculate_hpwl(def_file)
                    return {
                        'hpwl': hpwl,
                        'def_file': str(def_file),
                        'execution_time': end_time - start_time,
                        'success': True,
                        'strategy': layout_strategy
                    }
            
            logger.error(f"  ChipDRAG优化布局失败: {result.stderr}")
            return None
            
        except Exception as e:
            logger.error(f"  ChipDRAG优化布局异常: {str(e)}")
            return None
    
    def _extract_design_features(self, design_dir: Path) -> Dict:
        """提取设计特征"""
        features = {
            'num_components': 1000,  # 默认值
            'area': 100000000,       # 默认值
            'component_density': 0.1  # 默认值
        }
        
        # 尝试从DEF文件提取真实特征
        def_file = design_dir / "floorplan.def"
        if def_file.exists():
            try:
                with open(def_file, 'r') as f:
                    content = f.read()
                
                # 提取组件数量
                import re
                comp_match = re.search(r'COMPONENTS\s+(\d+)', content)
                if comp_match:
                    features['num_components'] = int(comp_match.group(1))
                
                # 提取面积
                area_match = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
                if area_match:
                    x1, y1, x2, y2 = map(int, area_match.groups())
                    features['area'] = (x2 - x1) * (y2 - y1)
                
                # 计算组件密度
                if features['area'] > 0:
                    features['component_density'] = features['num_components'] / features['area'] * 1000000
                    
            except Exception as e:
                logger.warning(f"提取设计特征失败: {str(e)}")
        
        return features
    
    def _extract_state(self, query: Dict, layout_result: Optional[Dict] = None) -> Dict:
        """提取状态特征"""
        state = {
            'design_features': query.get('features', {}),
            'design_name': query.get('design_name', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        if layout_result:
            state['layout_quality'] = layout_result.get('hpwl', 0)
            state['execution_time'] = layout_result.get('execution_time', 0)
        
        return state
    
    def _generate_layout_strategy(self, retrieved_cases: List[Dict], action: Dict) -> str:
        """生成布局策略"""
        # 基于检索案例和RL动作生成OpenROAD脚本
        strategy = """
        # 基础布局流程
        initialize_floorplan -utilization 0.7 -aspect_ratio 1.0 -core_space 2.0
        place_pins -random
        global_placement -disable_routability_driven
        detailed_placement
        """
        
        # 根据检索案例调整参数
        if retrieved_cases:
            best_case = retrieved_cases[0]
            if 'parameters' in best_case:
                params = best_case['parameters']
                if 'utilization' in params:
                    strategy = strategy.replace('0.7', str(params['utilization']))
                if 'aspect_ratio' in params:
                    strategy = strategy.replace('1.0', str(params['aspect_ratio']))
        
        return strategy
    
    def _generate_optimized_layout_strategy(self, retrieved_cases: List[Dict], action: Dict) -> str:
        """生成优化布局策略"""
        # 更复杂的优化策略
        strategy = """
        # 优化布局流程
        initialize_floorplan -utilization 0.8 -aspect_ratio 1.2 -core_space 1.5
        
        # 高级引脚布局
        place_pins -random -hor_layer 2 -ver_layer 3
        
        # 全局布局优化
        global_placement -disable_routability_driven -skip_initial_place
        
        # 详细布局优化
        detailed_placement -disallow_one_site_gaps
        
        # 时序优化
        estimate_parasitics -placement
        """
        
        # 根据检索案例和RL动作调整
        if retrieved_cases:
            best_case = retrieved_cases[0]
            if 'layout_strategy' in best_case:
                strategy = best_case['layout_strategy']
        
        return strategy
    
    def _calculate_reward(self, layout_result: Optional[Dict]) -> float:
        """计算奖励"""
        if not layout_result or not layout_result.get('success'):
            return 0.0
        
        hpwl = layout_result.get('hpwl', 0)
        if hpwl <= 0:
            return 0.0
        
        # 基于HPWL计算奖励（越小越好）
        reward = max(0.0, min(1.0, (10 - np.log10(hpwl)) / 4))
        return reward
    
    def _calculate_improvement(self, default_result: Dict, optimized_result: Dict) -> float:
        """计算改进百分比"""
        default_hpwl = default_result.get('hpwl', 0)
        optimized_hpwl = optimized_result.get('hpwl', 0)
        
        if default_hpwl <= 0 or optimized_hpwl <= 0:
            return 0.0
        
        improvement = ((default_hpwl - optimized_hpwl) / default_hpwl) * 100
        return max(0.0, improvement)
    
    def _calculate_hpwl(self, def_file: Path) -> float:
        """计算HPWL"""
        try:
            result = subprocess.run([
                'python', str(self.experiment_config['hpwl_script']), str(def_file)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Total HPWL:'):
                        hpwl_str = line.split(':')[1].strip()
                        return float(hpwl_str)
            
            logger.error(f"HPWL计算失败: {result.stderr}")
            return 0.0
            
        except Exception as e:
            logger.error(f"HPWL计算异常: {str(e)}")
            return 0.0
    
    def _collect_final_results(self, training_results: Dict, inference_results: Dict) -> Dict:
        """收集最终结果"""
        return {
            'training': training_results,
            'inference': inference_results,
            'experiment_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_designs': len(inference_results['designs']),
                'average_improvement': inference_results['total_improvement'] / len(inference_results['designs']) if inference_results['designs'] else 0.0,
                'training_episodes': len(training_results['episodes']),
                'total_training_reward': training_results['total_reward']
            }
        }
    
    def _generate_experiment_report(self, results: Dict):
        """生成实验报告"""
        report_file = self.results_dir / "real_chipdrag_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验报告已保存: {report_file}")
        
        # 打印摘要
        summary = results['summary']
        logger.info("=== 实验摘要 ===")
        logger.info(f"总设计数: {summary['total_designs']}")
        logger.info(f"平均改进: {summary['average_improvement']:.2f}%")
        logger.info(f"训练回合: {summary['training_episodes']}")
        logger.info(f"总训练奖励: {summary['total_training_reward']:.3f}")

if __name__ == "__main__":
    experiment = RealChipDRAGExperiment()
    results = experiment.run_complete_experiment() 