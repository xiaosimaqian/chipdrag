#!/usr/bin/env python3
"""
案例库扩充脚本
从真实DEF/LEF文件中提取高质量案例，改进相似度计算
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from modules.knowledge.knowledge_base import KnowledgeBase
from modules.parsers.def_parser import DEFParser
from modules.parsers.lef_parser import LEFParser
from modules.utils.config_loader import ConfigLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CaseLibraryEnhancer:
    """案例库增强器"""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """初始化"""
        self.config = ConfigLoader.load_config(config_path)
        self.knowledge_base = KnowledgeBase(self.config.get('knowledge_base', {}))
        self.def_parser = DEFParser()
        self.lef_parser = LEFParser()
        
    def enhance_case_library(self, data_dir: str = "dataset/ispd_2015_contest_benchmark"):
        """增强案例库"""
        logger.info("开始增强案例库...")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"数据目录不存在: {data_path}")
            return False
        
        enhanced_cases = []
        
        # 遍历所有设计目录
        for design_dir in data_path.iterdir():
            if not design_dir.is_dir():
                continue
                
            design_name = design_dir.name
            logger.info(f"处理设计: {design_name}")
            
            # 提取真实特征
            case = self._extract_real_case(design_dir, design_name)
            if case:
                enhanced_cases.append(case)
                logger.info(f"成功提取案例: {design_name}")
        
        # 保存增强的案例
        self._save_enhanced_cases(enhanced_cases)
        
        # 更新知识库
        self._update_knowledge_base(enhanced_cases)
        
        logger.info(f"案例库增强完成，共处理 {len(enhanced_cases)} 个案例")
        return True
    
    def _extract_real_case(self, design_dir: Path, design_name: str) -> Dict[str, Any]:
        """从真实设计文件中提取案例"""
        try:
            # 查找DEF和LEF文件
            def_files = list(design_dir.glob("*.def"))
            lef_files = list(design_dir.glob("*.lef"))
            
            if not def_files:
                logger.warning(f"未找到DEF文件: {design_name}")
                return None
            
            def_file = def_files[0]
            lef_file = lef_files[0] if lef_files else None
            
            # 解析DEF文件
            def_info = self._parse_def_file(def_file)
            if not def_info:
                return None
            
            # 解析LEF文件（如果存在）
            lef_info = self._parse_lef_file(lef_file) if lef_file else {}
            
            # 提取真实特征
            features = self._extract_real_features(def_info, lef_info, design_name)
            
            # 创建案例
            case = {
                'id': design_name,
                'name': design_name,
                'metadata': {
                    'design_type': self._infer_design_type(design_name),
                    'source': 'ispd_2015',
                    'def_file': str(def_file),
                    'lef_file': str(lef_file) if lef_file else None,
                    'timestamp': '2025-07-11'
                },
                'def_info': def_info,
                'lef_info': lef_info,
                'features': features,
                'optimization_result': {
                    'area': features.get('area', 0),
                    'components': features.get('num_components', 0),
                    'nets': features.get('num_nets', 0)
                }
            }
            
            return case
            
        except Exception as e:
            logger.error(f"提取案例失败 {design_name}: {str(e)}")
            return None
    
    def _parse_def_file(self, def_file: Path) -> Dict[str, Any]:
        """解析DEF文件"""
        try:
            with open(def_file, 'r') as f:
                content = f.read()
            
            # 提取关键信息
            def_info = {}
            
            # 提取组件数量
            import re
            components_match = re.search(r'COMPONENTS\s+(\d+)', content)
            if components_match:
                def_info['COMPONENTS'] = int(components_match.group(1))
            
            # 提取网络数量
            nets_match = re.search(r'NETS\s+(\d+)', content)
            if nets_match:
                def_info['NETS'] = int(nets_match.group(1))
            
            # 提取引脚数量
            pins_match = re.search(r'PINS\s+(\d+)', content)
            if pins_match:
                def_info['PINS'] = int(pins_match.group(1))
            
            # 提取芯片面积
            diearea_match = re.search(r'DIEAREA\s+\([^)]+\)\s+\([^)]+\)', content)
            if diearea_match:
                def_info['DIEAREA'] = diearea_match.group(0)
            
            # 提取特殊网络数量
            specialnets_match = re.search(r'SPECIALNETS\s+(\d+)', content)
            if specialnets_match:
                def_info['SPECIALNETS'] = int(specialnets_match.group(1))
            
            # 提取组件详细信息
            component_details = []
            component_section = re.search(r'COMPONENTS\s+\d+\s*;(.*?)END\s+COMPONENTS', content, re.DOTALL)
            if component_section:
                component_lines = component_section.group(1).strip().split('\n')
                for line in component_lines:
                    if line.strip() and 'PLACED' in line:
                        # 提取组件类型
                        type_match = re.search(r'(\w+)\s+PLACED', line)
                        if type_match:
                            component_details.append({
                                'type': type_match.group(1)
                            })
            
            def_info['component_details'] = component_details
            
            return def_info
            
        except Exception as e:
            logger.error(f"解析DEF文件失败: {str(e)}")
            return {}
    
    def _parse_lef_file(self, lef_file: Path) -> Dict[str, Any]:
        """解析LEF文件"""
        try:
            with open(lef_file, 'r') as f:
                content = f.read()
            
            # 提取关键信息
            lef_info = {}
            
            # 提取技术信息
            import re
            
            # 提取单位信息
            units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
            if units_match:
                lef_info['units'] = int(units_match.group(1))
            
            # 提取层信息
            layers = re.findall(r'LAYER\s+(\w+)', content)
            lef_info['layers'] = layers
            
            return lef_info
            
        except Exception as e:
            logger.error(f"解析LEF文件失败: {str(e)}")
            return {}
    
    def _extract_real_features(self, def_info: Dict, lef_info: Dict, design_name: str) -> Dict[str, Any]:
        """提取真实特征"""
        features = {}
        
        # 1. 组件数量
        features['num_components'] = def_info.get('COMPONENTS', 0)
        
        # 2. 网络数量
        features['num_nets'] = def_info.get('NETS', 0)
        
        # 3. 引脚数量
        features['num_pins'] = def_info.get('PINS', 0)
        
        # 4. 面积计算
        area = self._calculate_area(def_info)
        features['area'] = area
        
        # 5. 组件密度
        if area > 0 and features['num_components'] > 0:
            features['component_density'] = features['num_components'] / area
        else:
            features['component_density'] = 0.1
        
        # 6. 层次结构
        features['hierarchy'] = self._extract_hierarchy(def_info, design_name)
        
        # 7. 约束条件
        features['constraints'] = self._extract_constraints(def_info)
        
        # 8. 复杂度
        features['complexity'] = self._calculate_complexity(features)
        
        return features
    
    def _calculate_area(self, def_info: Dict) -> float:
        """计算芯片面积"""
        diearea = def_info.get('DIEAREA', '')
        if diearea:
            try:
                import re
                coords = re.findall(r'\(\s*(\d+)\s+(\d+)\s*\)', diearea)
                if len(coords) >= 2:
                    x1, y1 = map(int, coords[0])
                    x2, y2 = map(int, coords[1])
                    return (x2 - x1) * (y2 - y1)
            except:
                pass
        
        return 1000000  # 默认面积
    
    def _extract_hierarchy(self, def_info: Dict, design_name: str) -> Dict:
        """提取层次结构"""
        # 从组件类型推断层次结构
        component_types = set()
        for comp in def_info.get('component_details', []):
            if 'type' in comp:
                component_types.add(comp['type'])
        
        modules = list(component_types)[:10]  # 限制模块数量
        if not modules:
            modules = [self._infer_design_type(design_name)]
        
        return {
            'levels': ['top'],
            'modules': modules
        }
    
    def _extract_constraints(self, def_info: Dict) -> Dict:
        """提取约束条件"""
        return {
            'timing': {'max_delay': 1000},
            'power': {'max_power': 1000},
            'special_nets': def_info.get('SPECIALNETS', 0)
        }
    
    def _calculate_complexity(self, features: Dict) -> float:
        """计算设计复杂度"""
        complexity = 0.5  # 基础复杂度
        
        # 基于组件数量调整
        num_components = features.get('num_components', 0)
        if num_components > 50000:
            complexity += 0.3
        elif num_components > 30000:
            complexity += 0.2
        elif num_components > 15000:
            complexity += 0.1
        
        # 基于组件密度调整
        density = features.get('component_density', 0)
        if density > 0.1:
            complexity += 0.2
        elif density > 0.05:
            complexity += 0.1
        
        # 基于网络数量调整
        num_nets = features.get('num_nets', 0)
        if num_nets > 100000:
            complexity += 0.2
        elif num_nets > 50000:
            complexity += 0.1
        
        return min(1.0, complexity)
    
    def _infer_design_type(self, design_name: str) -> str:
        """推断设计类型"""
        design_name_lower = design_name.lower()
        
        if 'fft' in design_name_lower:
            return 'FFT'
        elif 'pci' in design_name_lower:
            return 'PCI'
        elif 'des' in design_name_lower:
            return 'DES'
        elif 'matrix' in design_name_lower:
            return 'Matrix'
        else:
            return 'General'
    
    def _save_enhanced_cases(self, cases: List[Dict[str, Any]]):
        """保存增强的案例"""
        output_dir = Path("data/enhanced_cases")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON格式
        json_file = output_dir / "enhanced_cases.json"
        with open(json_file, 'w') as f:
            json.dump(cases, f, indent=2, default=str)
        
        # 生成统计报告
        self._generate_statistics_report(cases, output_dir)
        
        logger.info(f"增强案例已保存到: {json_file}")
    
    def _generate_statistics_report(self, cases: List[Dict[str, Any]], output_dir: Path):
        """生成统计报告"""
        if not cases:
            return
        
        # 计算统计信息
        components = [case['features']['num_components'] for case in cases]
        areas = [case['features']['area'] for case in cases]
        complexities = [case['features']['complexity'] for case in cases]
        
        stats = {
            'total_cases': len(cases),
            'components': {
                'min': min(components),
                'max': max(components),
                'mean': np.mean(components),
                'std': np.std(components)
            },
            'areas': {
                'min': min(areas),
                'max': max(areas),
                'mean': np.mean(areas),
                'std': np.std(areas)
            },
            'complexities': {
                'min': min(complexities),
                'max': max(complexities),
                'mean': np.mean(complexities),
                'std': np.std(complexities)
            }
        }
        
        # 保存统计报告
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"统计报告已保存到: {stats_file}")
    
    def _update_knowledge_base(self, cases: List[Dict[str, Any]]):
        """更新知识库"""
        try:
            # 将增强的案例添加到知识库
            for case in cases:
                self.knowledge_base.add_case(
                    layout=case.get('def_info', {}),
                    optimization_result=case.get('optimization_result', {}),
                    metadata=case.get('metadata', {})
                )
            
            # 保存知识库
            self.knowledge_base.save_data()
            
            logger.info(f"知识库已更新，添加了 {len(cases)} 个案例")
            
        except Exception as e:
            logger.error(f"更新知识库失败: {str(e)}")

def main():
    """主函数"""
    enhancer = CaseLibraryEnhancer()
    
    # 增强案例库
    success = enhancer.enhance_case_library()
    
    if success:
        logger.info("案例库增强完成！")
    else:
        logger.error("案例库增强失败！")

if __name__ == "__main__":
    main() 