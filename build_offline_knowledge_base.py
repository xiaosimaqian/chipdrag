#!/usr/bin/env python3
"""
离线知识库构建脚本
从已有的批量训练结果中提取真实案例数据，构建包含真实HPWL、特征、层次结构的知识库
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('offline_knowledge_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OfflineKnowledgeBaseBuilder:
    """离线知识库构建器"""
    
    def __init__(self, 
                 data_dir: str = "data/designs/ispd_2015_contest_benchmark",
                 results_dir: str = "results/iterative_training",
                 output_dir: str = "layout_experience"):
        """初始化离线知识库构建器
        
        Args:
            data_dir: ISPD设计数据目录
            results_dir: 批量训练结果目录
            output_dir: 知识库输出目录
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 知识库文件路径
        self.cases_file = self.output_dir / "cases.pkl"
        self.data_file = self.output_dir / "data.pkl"
        self.knowledge_graph_file = self.output_dir / "knowledge_graph.pkl"
        
        logger.info(f"离线知识库构建器初始化完成")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"结果目录: {self.results_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def extract_design_features(self, design_dir: Path) -> Dict[str, Any]:
        """从设计目录提取特征
        
        Args:
            design_dir: 设计目录路径
            
        Returns:
            Dict[str, Any]: 设计特征
        """
        features = {}
        
        try:
            # 查找DEF文件
            def_files = list(design_dir.glob("*.def"))
            if def_files:
                def_file = def_files[0]  # 使用第一个DEF文件
                features.update(self._extract_def_features(def_file))
            
            # 查找LEF文件
            lef_files = list(design_dir.glob("*.lef"))
            if lef_files:
                lef_file = lef_files[0]
                features.update(self._extract_lef_features(lef_file))
            
            # 查找Verilog文件
            v_files = list(design_dir.glob("*.v"))
            if v_files:
                v_file = v_files[0]
                features.update(self._extract_verilog_features(v_file))
            
            return features
            
        except Exception as e:
            logger.error(f"提取设计特征失败 {design_dir.name}: {e}")
            return {}
    
    def _extract_def_features(self, def_file: Path) -> Dict[str, Any]:
        """从DEF文件提取特征"""
        features = {}
        
        try:
            with open(def_file, 'r') as f:
                content = f.read()
            
            # 提取组件数量
            components_match = re.search(r'COMPONENTS\s+(\d+)', content)
            if components_match:
                features['num_components'] = int(components_match.group(1))
            
            # 提取网络数量
            nets_match = re.search(r'NETS\s+(\d+)', content)
            if nets_match:
                features['num_nets'] = int(nets_match.group(1))
            
            # 提取引脚数量
            pins_match = re.search(r'PINS\s+(\d+)', content)
            if pins_match:
                features['num_pins'] = int(pins_match.group(1))
            
            # 提取设计面积
            diearea_match = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
            if diearea_match:
                x1, y1, x2, y2 = map(int, diearea_match.groups())
                features['area'] = (x2 - x1) * (y2 - y1)
                features['width'] = x2 - x1
                features['height'] = y2 - y1
            
            # 提取特殊网络数量
            special_nets_match = re.search(r'SPECIALNETS\s+(\d+)', content)
            if special_nets_match:
                features['num_special_nets'] = int(special_nets_match.group(1))
            
            # 提取模块信息
            module_matches = re.findall(r'-\s+(\w+)\s+(\w+)', content)
            if module_matches:
                modules = list(set([match[1] for match in module_matches]))
                features['modules'] = modules[:20]  # 限制数量
                features['num_module_types'] = len(modules)
            
            return features
            
        except Exception as e:
            logger.error(f"提取DEF特征失败: {e}")
            return {}
    
    def _extract_lef_features(self, lef_file: Path) -> Dict[str, Any]:
        """从LEF文件提取特征"""
        features = {}
        
        try:
            with open(lef_file, 'r') as f:
                content = f.read()
            
            # 提取制造网格
            grid_match = re.search(r'MANUFACTURINGGRID\s+(\d+\.?\d*)', content)
            if grid_match:
                features['manufacturing_grid'] = float(grid_match.group(1))
            
            # 提取单元库数量
            cell_count = len(re.findall(r'MACRO\s+(\w+)', content))
            features['cell_types'] = cell_count
            
            # 提取SITE信息
            site_matches = re.findall(r'SITE\s+(\w+)', content)
            if site_matches:
                features['sites'] = list(set(site_matches))
            
            return features
            
        except Exception as e:
            logger.error(f"提取LEF特征失败: {e}")
            return {}
    
    def _extract_verilog_features(self, v_file: Path) -> Dict[str, Any]:
        """从Verilog文件提取特征"""
        features = {}
        
        try:
            with open(v_file, 'r') as f:
                content = f.read()
            
            # 提取模块数量
            module_count = len(re.findall(r'module\s+(\w+)', content))
            features['num_modules'] = module_count
            
            # 提取端口数量
            port_count = len(re.findall(r'(input|output|inout)', content))
            features['num_ports'] = port_count
            
            return features
            
        except Exception as e:
            logger.error(f"提取Verilog特征失败: {e}")
            return {}
    
    def extract_hpwl_from_training_results(self, design_dir: Path) -> Optional[float]:
        """从训练结果中提取HPWL
        
        Args:
            design_dir: 设计目录路径
            
        Returns:
            Optional[float]: HPWL值
        """
        try:
            # 查找训练结果中的HPWL
            iterations_dir = design_dir / "output" / "iterations"
            if not iterations_dir.exists():
                return None
            
            # 查找最新的DEF文件
            def_files = list(iterations_dir.glob("iteration_10*.def"))
            if not def_files:
                return None
            
            # 优先使用RL训练结果
            def_file = None
            for f in def_files:
                if "rl_training" in f.name:
                    def_file = f
                    break
            if def_file is None:
                def_file = def_files[0]
            
            # 调用HPWL计算脚本
            from calculate_hpwl import parse_def_file, calculate_hpwl
            
            components, nets = parse_def_file(str(def_file))
            hpwl, valid_nets, total_nets = calculate_hpwl(components, nets)
            
            return hpwl
            
        except Exception as e:
            logger.error(f"提取HPWL失败 {design_dir.name}: {e}")
            return None
    
    def build_knowledge_base(self) -> Dict[str, Any]:
        """构建知识库
        
        Returns:
            Dict[str, Any]: 构建结果统计
        """
        logger.info("开始构建离线知识库...")
        
        # 获取所有设计
        designs = []
        for design_dir in self.data_dir.iterdir():
            if design_dir.is_dir():
                designs.append(design_dir)
        
        logger.info(f"找到 {len(designs)} 个设计")
        
        # 构建案例数据
        cases = []
        knowledge_graph = {
            'global': [],
            'module': [],
            'connection': [],
            'constraint': []
        }
        
        for i, design_dir in enumerate(designs):
            design_name = design_dir.name
            logger.info(f"处理设计 {i+1}/{len(designs)}: {design_name}")
            
            try:
                # 提取设计特征
                features = self.extract_design_features(design_dir)
                
                # 提取HPWL
                hpwl = self.extract_hpwl_from_training_results(design_dir)
                
                # 构建案例
                case = {
                    'id': i,
                    'design_name': design_name,
                    'features': features,
                    'hpwl': hpwl,
                    'layout': {
                        'name': design_name,
                        'type': 'ispd_design',
                        'features': features,
                        'hierarchy': {
                            'levels': ['top', 'module'],
                            'modules': features.get('modules', []),
                            'max_depth': 2
                        },
                        'components': [],  # 简化，实际可从DEF解析
                        'nets': []  # 简化，实际可从DEF解析
                    },
                    'optimization_result': {
                        'hpwl': hpwl,
                        'area': features.get('area', 0),
                        'num_components': features.get('num_components', 0),
                        'num_nets': features.get('num_nets', 0)
                    },
                    'metadata': {
                        'source': 'ispd_2015_contest',
                        'extraction_time': datetime.now().isoformat(),
                        'design_path': str(design_dir)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                cases.append(case)
                
                # 更新知识图谱
                knowledge_graph['global'].append({
                    'case_id': i,
                    'features': features
                })
                
                logger.info(f"  特征: {features}")
                logger.info(f"  HPWL: {hpwl}")
                
            except Exception as e:
                logger.error(f"处理设计 {design_name} 失败: {e}")
                continue
        
        # 保存知识库
        logger.info(f"保存知识库，包含 {len(cases)} 个案例")
        
        # 保存案例数据
        with open(self.cases_file, 'wb') as f:
            pickle.dump(cases, f)
        
        # 保存数据文件（兼容性）
        with open(self.data_file, 'wb') as f:
            pickle.dump(cases, f)
        
        # 保存知识图谱
        with open(self.knowledge_graph_file, 'wb') as f:
            pickle.dump(knowledge_graph, f)
        
        # 生成统计报告
        stats = {
            'total_designs': len(designs),
            'successful_cases': len(cases),
            'failed_cases': len(designs) - len(cases),
            'success_rate': len(cases) / len(designs) if designs else 0,
            'features_summary': self._generate_features_summary(cases),
            'hpwl_summary': self._generate_hpwl_summary(cases)
        }
        
        # 保存统计报告
        stats_file = self.output_dir / "knowledge_base_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info("离线知识库构建完成")
        logger.info(f"成功案例: {stats['successful_cases']}")
        logger.info(f"成功率: {stats['success_rate']:.2%}")
        
        return stats
    
    def _generate_features_summary(self, cases: List[Dict]) -> Dict[str, Any]:
        """生成特征统计摘要"""
        if not cases:
            return {}
        
        # 收集所有数值特征
        numeric_features = {}
        for case in cases:
            features = case['features']
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_features:
                        numeric_features[key] = []
                    numeric_features[key].append(value)
        
        # 计算统计信息
        summary = {}
        for key, values in numeric_features.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return summary
    
    def _generate_hpwl_summary(self, cases: List[Dict]) -> Dict[str, Any]:
        """生成HPWL统计摘要"""
        hpwl_values = [case['hpwl'] for case in cases if case['hpwl'] is not None]
        
        if not hpwl_values:
            return {}
        
        return {
            'mean': np.mean(hpwl_values),
            'std': np.std(hpwl_values),
            'min': np.min(hpwl_values),
            'max': np.max(hpwl_values),
            'count': len(hpwl_values)
        }

def main():
    """主函数"""
    builder = OfflineKnowledgeBaseBuilder()
    stats = builder.build_knowledge_base()
    
    print("\n=== 离线知识库构建完成 ===")
    print(f"总设计数: {stats['total_designs']}")
    print(f"成功案例: {stats['successful_cases']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    
    if stats['hpwl_summary']:
        print(f"HPWL统计:")
        print(f"  平均值: {stats['hpwl_summary']['mean']:.2e}")
        print(f"  标准差: {stats['hpwl_summary']['std']:.2e}")
        print(f"  最小值: {stats['hpwl_summary']['min']:.2e}")
        print(f"  最大值: {stats['hpwl_summary']['max']:.2e}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 