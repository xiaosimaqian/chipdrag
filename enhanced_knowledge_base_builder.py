#!/usr/bin/env python3
"""
增强知识库构建器
自动积累布局经验数据，包括不同参数配置的HPWL结果
"""

import pickle
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedKnowledgeBaseBuilder:
    def __init__(self, data_dir: str = "data/designs/ispd_2015_contest_benchmark"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("layout_experience")
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_hpwl_from_def(self, def_file: Path) -> Optional[float]:
        """从DEF文件提取HPWL"""
        try:
            result = subprocess.run(
                ["python", "calculate_hpwl.py", str(def_file)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                # 解析输出
                for line in result.stdout.split('\n'):
                    if 'Actual HPWL (microns):' in line:
                        hpwl_str = line.split(':')[-1].strip()
                        return float(hpwl_str)
            return None
        except Exception as e:
            logger.error(f"提取HPWL失败 {def_file}: {e}")
            return None
    
    def extract_design_features(self, design_dir: Path) -> Dict[str, Any]:
        """提取设计特征"""
        features = {}
        
        # 从DEF文件提取特征
        def_files = list(design_dir.glob('*.def'))
        if def_files:
            def_file = def_files[0]
            try:
                with open(def_file, 'r') as f:
                    content = f.read()
                
                # 基本特征
                comp_match = re.search(r'COMPONENTS\s+(\d+)', content)
                if comp_match:
                    features['num_components'] = int(comp_match.group(1))
                
                net_match = re.search(r'NETS\s+(\d+)', content)
                if net_match:
                    features['num_nets'] = int(net_match.group(1))
                
                pin_match = re.search(r'PINS\s+(\d+)', content)
                if pin_match:
                    features['num_pins'] = int(pin_match.group(1))
                
                # 面积特征
                die_match = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
                if die_match:
                    x1, y1, x2, y2 = map(int, die_match.groups())
                    features['die_area'] = (x2 - x1) * (y2 - y1)
                    features['die_width'] = x2 - x1
                    features['die_height'] = y2 - y1
                    features['aspect_ratio'] = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
                
                # 组件密度
                if features.get('num_components') and features.get('die_area'):
                    features['component_density'] = features['num_components'] / features['die_area']
                
                # 网络密度
                if features.get('num_nets') and features.get('die_area'):
                    features['net_density'] = features['num_nets'] / features['die_area']
                
            except Exception as e:
                logger.error(f"解析DEF文件失败: {e}")
        
        # 从Verilog文件提取特征
        v_files = list(design_dir.glob('*.v'))
        if v_files:
            v_file = v_files[0]
            try:
                with open(v_file, 'r') as f:
                    content = f.read()
                
                features['num_modules'] = len(re.findall(r'module\s+\w+', content))
                features['num_ports'] = len(re.findall(r'input\s+|output\s+|inout\s+', content))
                
                # 时序相关特征
                features['has_clock'] = 'clock' in content.lower() or 'clk' in content.lower()
                features['has_reset'] = 'reset' in content.lower() or 'rst' in content.lower()
                
            except Exception as e:
                logger.error(f"解析Verilog文件失败: {e}")
        
        return features
    
    def collect_layout_experience(self, design_dir: Path, design_name: str) -> List[Dict[str, Any]]:
        """收集布局经验数据"""
        experiences = []
        iterations_dir = design_dir / "output" / "iterations"
        
        if not iterations_dir.exists():
            return experiences
        
        # 收集不同迭代的HPWL数据
        for def_file in iterations_dir.glob("*.def"):
            if "iteration" in def_file.name:
                hpwl = self.extract_hpwl_from_def(def_file)
                if hpwl:
                    # 解析迭代信息
                    iteration_info = self._parse_iteration_info(def_file.name)
                    
                    experience = {
                        'design_name': design_name,
                        'iteration': iteration_info.get('iteration', 0),
                        'layout_type': iteration_info.get('type', 'unknown'),
                        'hpwl': hpwl,
                        'timestamp': datetime.now().isoformat(),
                        'def_file': str(def_file)
                    }
                    experiences.append(experience)
        
        return experiences
    
    def _parse_iteration_info(self, filename: str) -> Dict[str, Any]:
        """解析迭代文件名信息"""
        info = {'iteration': 0, 'type': 'unknown'}
        
        # 匹配 iteration_X.def 或 iteration_X_type.def
        match = re.search(r'iteration_(\d+)(?:_(\w+))?\.def', filename)
        if match:
            info['iteration'] = int(match.group(1))
            if match.group(2):
                info['type'] = match.group(2)
        
        return info
    
    def generate_optimization_strategies(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于设计特征生成优化策略建议"""
        strategies = []
        
        # 基于组件数量的策略
        num_components = features.get('num_components', 0)
        if num_components > 100000:
            strategies.append({
                'type': 'large_design',
                'k_value_range': [5, 15],
                'weight_quality': 0.5,
                'weight_similarity': 0.3,
                'weight_entity': 0.2,
                'description': '大设计，需要更多相似案例'
            })
        elif num_components > 50000:
            strategies.append({
                'type': 'medium_design',
                'k_value_range': [3, 10],
                'weight_quality': 0.4,
                'weight_similarity': 0.4,
                'weight_entity': 0.2,
                'description': '中等设计，平衡质量和相似性'
            })
        else:
            strategies.append({
                'type': 'small_design',
                'k_value_range': [2, 6],
                'weight_quality': 0.3,
                'weight_similarity': 0.5,
                'weight_entity': 0.2,
                'description': '小设计，优先相似性'
            })
        
        # 基于密度的策略
        component_density = features.get('component_density', 0)
        if component_density > 1e-6:
            strategies.append({
                'type': 'high_density',
                'layout_tips': ['使用更严格的布局约束', '考虑时序优化'],
                'description': '高密度设计，需要更精细的布局'
            })
        
        return strategies
    
    def build_enhanced_knowledge_base(self) -> Dict[str, Any]:
        """构建增强知识库"""
        logger.info("开始构建增强知识库...")
        
        enhanced_cases = []
        layout_experiences = []
        
        for design_dir in self.data_dir.iterdir():
            if not design_dir.is_dir():
                continue
            
            design_name = design_dir.name
            logger.info(f"处理设计: {design_name}")
            
            # 提取设计特征
            features = self.extract_design_features(design_dir)
            
            # 收集布局经验
            experiences = self.collect_layout_experience(design_dir, design_name)
            layout_experiences.extend(experiences)
            
            # 生成优化策略
            strategies = self.generate_optimization_strategies(features)
            
            # 构建增强案例
            enhanced_case = {
                'id': len(enhanced_cases) + 1,
                'design_name': design_name,
                'features': features,
                'strategies': strategies,
                'experiences': experiences,
                'timestamp': datetime.now().isoformat()
            }
            
            enhanced_cases.append(enhanced_case)
            
            logger.info(f"  特征: {len(features)} 项")
            logger.info(f"  经验: {len(experiences)} 条")
            logger.info(f"  策略: {len(strategies)} 种")
        
        # 构建知识库
        knowledge_base = {
            'cases': enhanced_cases,
            'layout_experiences': layout_experiences,
            'statistics': {
                'total_cases': len(enhanced_cases),
                'total_experiences': len(layout_experiences),
                'avg_experiences_per_case': len(layout_experiences) / len(enhanced_cases) if enhanced_cases else 0,
                'build_timestamp': datetime.now().isoformat()
            }
        }
        
        # 保存知识库
        with open(self.output_dir / 'enhanced_cases.pkl', 'wb') as f:
            pickle.dump(knowledge_base, f)
        
        # 保存为JSON格式便于查看
        with open(self.output_dir / 'enhanced_cases.json', 'w') as f:
            json.dump(knowledge_base, f, indent=2, default=str)
        
        logger.info(f"✅ 增强知识库构建完成！")
        logger.info(f"   案例数量: {len(enhanced_cases)}")
        logger.info(f"   经验数据: {len(layout_experiences)} 条")
        logger.info(f"   保存路径: {self.output_dir}")
        
        return knowledge_base

def main():
    builder = EnhancedKnowledgeBaseBuilder()
    knowledge_base = builder.build_enhanced_knowledge_base()
    
    # 打印统计信息
    stats = knowledge_base['statistics']
    print(f"\n=== 知识库统计信息 ===")
    print(f"总案例数: {stats['total_cases']}")
    print(f"总经验数: {stats['total_experiences']}")
    print(f"平均经验/案例: {stats['avg_experiences_per_case']:.1f}")

if __name__ == "__main__":
    main() 