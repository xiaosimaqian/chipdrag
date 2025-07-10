#!/usr/bin/env python3
"""
扩充知识库脚本
收集更多高质量的布局案例，提高RAG系统的检索质量
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.config_loader import ConfigLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeBaseExpander:
    """知识库扩充器"""
    
    def __init__(self):
        """初始化"""
        self.config_loader = ConfigLoader()
        
        # 尝试加载配置文件，如果失败则使用默认配置
        try:
            self.config = self.config_loader.load_config('knowledge_base.json')
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {str(e)}")
            self.config = {
                "path": "data/knowledge_base",
                "format": "json",
                "layout_experience": "data/knowledge_base",
                "similarity": {
                    "threshold": 0.5,
                    "top_k": 5
                }
            }
        
        self.knowledge_base = KnowledgeBase(self.config)
        
    def collect_real_cases(self) -> List[Dict]:
        """收集真实案例
        
        Returns:
            List[Dict]: 真实案例列表
        """
        cases = []
        
        # 从ISPD基准测试收集案例
        ispd_cases = self._collect_ispd_cases()
        cases.extend(ispd_cases)
        
        # 从OpenROAD结果收集案例
        openroad_cases = self._collect_openroad_cases()
        cases.extend(openroad_cases)
        
        # 从实验数据收集案例
        experiment_cases = self._collect_experiment_cases()
        cases.extend(experiment_cases)
        
        return cases
    
    def _collect_ispd_cases(self) -> List[Dict]:
        """从ISPD基准测试收集案例"""
        cases = []
        
        try:
            # 检查ISPD数据目录
            ispd_dir = "data/designs/ispd_2015_contest_benchmark"
            if os.path.exists(ispd_dir):
                logger.info(f"从 {ispd_dir} 收集ISPD案例")
                
                for design_dir in os.listdir(ispd_dir):
                    design_path = os.path.join(ispd_dir, design_dir)
                    if os.path.isdir(design_path):
                        case = self._extract_ispd_case(design_path, design_dir)
                        if case:
                            cases.append(case)
            
            logger.info(f"收集到 {len(cases)} 个ISPD案例")
            
        except Exception as e:
            logger.error(f"收集ISPD案例失败: {str(e)}")
        
        return cases
    
    def _extract_ispd_case(self, design_path: str, design_name: str) -> Dict:
        """提取ISPD案例
        
        Args:
            design_path: 设计路径
            design_name: 设计名称
            
        Returns:
            Dict: 案例数据
        """
        try:
            # 查找DEF文件
            def_files = []
            for file in os.listdir(design_path):
                if file.endswith('.def'):
                    def_files.append(os.path.join(design_path, file))
            
            if not def_files:
                return None
            
            # 分析DEF文件
            def_file = def_files[0]  # 使用第一个DEF文件
            case_data = self._analyze_def_file(def_file)
            
            if not case_data:
                return None
            
            # 构建案例
            case = {
                'id': f'ispd_{design_name}',
                'name': design_name,
                'source': 'ispd_2015_contest',
                'features': {
                    'num_components': case_data.get('num_components', 0),
                    'area': case_data.get('area', 0),
                    'component_density': case_data.get('density', 0),
                    'hierarchy': {'modules': [design_name]},
                    'constraints': {
                        'timing': {'max_delay': 1000},
                        'power': {'max_power': 1000},
                        'special_nets': case_data.get('num_nets', 0)
                    }
                },
                'layout_strategy': {
                    'placement_strategy': 'analytical',
                    'routing_strategy': 'timing_driven',
                    'optimization_priorities': ['timing', 'wirelength', 'power'],
                    'parameter_suggestions': {
                        'density_target': 0.7,
                        'wirelength_weight': 1.0,
                        'timing_weight': 1.0,
                        'power_weight': 0.6,
                        'area_weight': 0.5
                    },
                    'constraint_handling': {
                        'timing_constraints': 'aggressive',
                        'power_constraints': 'moderate',
                        'area_constraints': 'flexible'
                    },
                    'quality_targets': {
                        'hpwl_improvement': 0.03,
                        'timing_slack': 0.15,
                        'power_reduction': 0.02,
                        'area_utilization': 0.75
                    },
                    'execution_plan': [
                        'initial_placement',
                        'timing_optimization',
                        'wirelength_optimization',
                        'final_legalization'
                    ]
                },
                'optimization_guidelines': {
                    'placement_guidelines': [
                        '基于ISPD基准测试的布局策略',
                        '优先考虑时序约束',
                        '平衡面积和性能需求'
                    ],
                    'routing_guidelines': [
                        '时序驱动的布线策略',
                        '考虑拥塞避免',
                        '优化关键路径'
                    ],
                    'optimization_guidelines': [
                        '多目标优化平衡',
                        '迭代改进策略',
                        '约束满足检查'
                    ]
                },
                'metadata': {
                    'source': 'ispd_2015_contest',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            return case
            
        except Exception as e:
            logger.error(f"提取ISPD案例失败 {design_name}: {str(e)}")
            return None
    
    def _analyze_def_file(self, def_file: str) -> Dict:
        """分析DEF文件
        
        Args:
            def_file: DEF文件路径
            
        Returns:
            Dict: 分析结果
        """
        try:
            with open(def_file, 'r') as f:
                content = f.read()
            
            # 提取基本信息
            num_components = 0
            area = 0
            num_nets = 0
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('COMPONENTS'):
                    # 解析组件数量
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # 处理可能的括号格式
                            comp_str = parts[1].strip('()')
                            num_components = int(comp_str)
                        except ValueError:
                            logger.warning(f"无法解析组件数量: {parts[1]}")
                            continue
                elif line.startswith('DIEAREA'):
                    # 解析面积 - 处理格式: DIEAREA ( x1 y1 ) ( x2 y2 ) ;
                    try:
                        # 使用正则表达式提取坐标
                        import re
                        pattern = r'DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)'
                        match = re.search(pattern, line)
                        if match:
                            x1, y1, x2, y2 = map(int, match.groups())
                            area = (x2 - x1) * (y2 - y1)
                        else:
                            # 备用解析方法
                            parts = line.split()
                            if len(parts) >= 7:  # DIEAREA ( x1 y1 ) ( x2 y2 ) ;
                                x1 = int(parts[1].strip('()'))
                                y1 = int(parts[2].strip('()'))
                                x2 = int(parts[4].strip('()'))
                                y2 = int(parts[5].strip('()'))
                                area = (x2 - x1) * (y2 - y1)
                            else:
                                raise ValueError(f"DIEAREA格式不正确: {line}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"无法解析DIEAREA坐标: {line.strip()}, 错误: {str(e)}")
                        continue
                elif line.startswith('NETS'):
                    # 解析网络数量
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # 处理可能的括号格式
                            nets_str = parts[1].strip('()')
                            num_nets = int(nets_str)
                        except ValueError:
                            logger.warning(f"无法解析网络数量: {parts[1]}")
                            continue
            
            # 计算密度
            density = 0
            if area > 0:
                density = num_components / area
            
            # 验证数据有效性
            if num_components == 0:
                logger.warning(f"DEF文件中未找到有效组件数量: {def_file}")
                return {}
            
            logger.info(f"解析DEF文件成功: 组件={num_components}, 面积={area}, 网络={num_nets}")
            
            return {
                'num_components': num_components,
                'area': area,
                'num_nets': num_nets,
                'density': density
            }
            
        except Exception as e:
            logger.error(f"分析DEF文件失败 {def_file}: {str(e)}")
            return {}
    
    def _collect_openroad_cases(self) -> List[Dict]:
        """从OpenROAD结果收集案例"""
        cases = []
        
        try:
            # 检查OpenROAD结果目录
            openroad_dir = "data/processed"
            if os.path.exists(openroad_dir):
                logger.info(f"从 {openroad_dir} 收集OpenROAD案例")
                
                for result_dir in os.listdir(openroad_dir):
                    result_path = os.path.join(openroad_dir, result_dir)
                    if os.path.isdir(result_path):
                        case = self._extract_openroad_case(result_path, result_dir)
                        if case:
                            cases.append(case)
            
            logger.info(f"收集到 {len(cases)} 个OpenROAD案例")
            
        except Exception as e:
            logger.error(f"收集OpenROAD案例失败: {str(e)}")
        
        return cases
    
    def _extract_openroad_case(self, result_path: str, result_name: str) -> Dict:
        """提取OpenROAD案例"""
        try:
            # 查找结果文件
            result_file = os.path.join(result_path, 'result.json')
            if not os.path.exists(result_file):
                return None
            
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # 构建案例
            case = {
                'id': f'openroad_{result_name}',
                'name': result_name,
                'source': 'openroad_optimization',
                'features': {
                    'num_components': result_data.get('num_components', 0),
                    'area': result_data.get('area', 0),
                    'component_density': result_data.get('density', 0),
                    'hierarchy': {'modules': [result_name]},
                    'constraints': result_data.get('constraints', {})
                },
                'layout_strategy': result_data.get('layout_strategy', {}),
                'optimization_guidelines': result_data.get('optimization_guidelines', {}),
                'metadata': {
                    'source': 'openroad_optimization',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            return case
            
        except Exception as e:
            logger.error(f"提取OpenROAD案例失败 {result_name}: {str(e)}")
            return None
    
    def _collect_experiment_cases(self) -> List[Dict]:
        """从实验数据收集案例"""
        cases = []
        
        try:
            # 检查实验结果目录
            experiment_dir = "paper_hpwl_results"
            if os.path.exists(experiment_dir):
                logger.info(f"从 {experiment_dir} 收集实验案例")
                
                for exp_dir in os.listdir(experiment_dir):
                    exp_path = os.path.join(experiment_dir, exp_dir)
                    if os.path.isdir(exp_path):
                        case = self._extract_experiment_case(exp_path, exp_dir)
                        if case:
                            cases.append(case)
            
            logger.info(f"收集到 {len(cases)} 个实验案例")
            
        except Exception as e:
            logger.error(f"收集实验案例失败: {str(e)}")
        
        return cases
    
    def _extract_experiment_case(self, exp_path: str, exp_name: str) -> Dict:
        """提取实验案例"""
        try:
            # 查找实验结果文件
            result_file = os.path.join(exp_path, 'experiment_result.json')
            if not os.path.exists(result_file):
                return None
            
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # 构建案例
            case = {
                'id': f'experiment_{exp_name}',
                'name': exp_name,
                'source': 'chipdrag_experiment',
                'features': result_data.get('features', {}),
                'layout_strategy': result_data.get('layout_strategy', {}),
                'optimization_guidelines': result_data.get('optimization_guidelines', {}),
                'metadata': {
                    'source': 'chipdrag_experiment',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            return case
            
        except Exception as e:
            logger.error(f"提取实验案例失败 {exp_name}: {str(e)}")
            return None
    
    def add_cases_to_knowledge_base(self, cases: List[Dict]) -> int:
        """将案例添加到知识库
        
        Args:
            cases: 案例列表
            
        Returns:
            int: 成功添加的案例数量
        """
        added_count = 0
        
        for case in cases:
            try:
                # 添加到知识库
                self.knowledge_base.add_case(
                    layout=case.get('features', {}),
                    optimization_result=case.get('layout_strategy', {}),
                    metadata=case.get('metadata', {})
                )
                added_count += 1
                logger.info(f"成功添加案例: {case.get('name', 'unknown')}")
                
            except Exception as e:
                logger.error(f"添加案例失败: {str(e)}")
        
        return added_count
    
    def run(self):
        """运行知识库扩充"""
        logger.info("开始扩充知识库...")
        
        # 收集真实案例
        cases = self.collect_real_cases()
        logger.info(f"收集到 {len(cases)} 个案例")
        
        if not cases:
            logger.warning("未收集到任何案例，请检查数据源")
            return
        
        # 添加到知识库
        added_count = self.add_cases_to_knowledge_base(cases)
        logger.info(f"成功添加 {added_count} 个案例到知识库")
        
        # 保存知识库
        self.knowledge_base._save_data()
        logger.info("知识库扩充完成")

def main():
    """主函数"""
    expander = KnowledgeBaseExpander()
    expander.run()

if __name__ == "__main__":
    main() 