#!/usr/bin/env python3
"""
批量导入ISPD 2015实验结果到知识库
将 results/ispd_training_fixed_v10/ 下的所有 *_result.json 及其对应设计的结构化文件导入知识库
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ISPDKnowledgeBaseImporter:
    """ISPD实验结果知识库导入器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results/ispd_training_fixed_v11"
        self.benchmark_dir = self.base_dir / "data/designs/ispd_2015_contest_benchmark"
        self.knowledge_base_dir = self.base_dir / "data/knowledge_base"
        
        # 创建知识库目录
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 知识库文件路径
        self.cases_file = self.knowledge_base_dir / "ispd_cases.json"
        self.metadata_file = self.knowledge_base_dir / "ispd_metadata.json"
        
        # 统计信息
        self.imported_cases = 0
        self.failed_cases = 0
        self.cases_data = []
        self.metadata = {
            "total_cases": 0,
            "successful_cases": 0,
            "failed_cases": 0,
            "import_date": datetime.now().isoformat(),
            "source": "ISPD 2015 Contest Benchmark",
            "description": "真实ISPD 2015基准测试的布局优化结果"
        }
        
        logger.info(f"ISPD知识库导入器初始化完成")
        logger.info(f"结果目录: {self.results_dir}")
        logger.info(f"基准目录: {self.benchmark_dir}")
        logger.info(f"知识库目录: {self.knowledge_base_dir}")
    
    def load_existing_data(self):
        """加载现有的知识库数据"""
        if self.cases_file.exists():
            with open(self.cases_file, 'r', encoding='utf-8') as f:
                self.cases_data = json.load(f)
            logger.info(f"加载现有案例数据: {len(self.cases_data)} 个案例")
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info("加载现有元数据")
    
    def get_design_files(self, design_name: str) -> Dict[str, Optional[str]]:
        """获取设计相关的文件路径"""
        design_dir = self.benchmark_dir / design_name
        if not design_dir.exists():
            logger.warning(f"设计目录不存在: {design_dir}")
            return {}
        
        files = {}
        file_types = {
            'verilog': 'design.v',
            'def': 'floorplan.def',
            'tech_lef': 'tech.lef',
            'cells_lef': 'cells.lef',
            'placement_result': 'placement_result.def',
            'placement_verilog': 'placement_result.v'
        }
        
        for file_type, filename in file_types.items():
            file_path = design_dir / filename
            if file_path.exists():
                # 读取文件内容（对于大文件只读取前1000行）
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if len(lines) > 1000:
                            content = ''.join(lines[:1000]) + f"\n... (文件过大，只显示前1000行，共{len(lines)}行)"
                        else:
                            content = ''.join(lines)
                        files[file_type] = content
                except Exception as e:
                    logger.warning(f"读取文件失败 {file_path}: {e}")
                    files[file_type] = None
            else:
                files[file_type] = None
        
        return files
    
    def extract_design_features(self, design_name: str, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取设计特征"""
        features = {
            'design_name': design_name,
            'success': result_data.get('success', False),
            'execution_time': result_data.get('execution_time', 0),
            'wirelength': result_data.get('wirelength', 0),
            'area': result_data.get('area', 0),
            'design_type': self._classify_design_type(design_name),
            'complexity': self._estimate_complexity(design_name),
            'constraints': self._extract_constraints(result_data),
            'timestamp': datetime.now().isoformat()
        }
        
        # 从stdout中提取更多信息
        stdout = result_data.get('stdout', '')
        features.update(self._parse_stdout_info(stdout))
        
        return features
    
    def _classify_design_type(self, design_name: str) -> str:
        """分类设计类型"""
        design_name_lower = design_name.lower()
        if 'des' in design_name_lower:
            return 'DES加密'
        elif 'fft' in design_name_lower:
            return 'FFT变换'
        elif 'matrix' in design_name_lower:
            return '矩阵乘法'
        elif 'pci' in design_name_lower:
            return 'PCI桥接'
        elif 'superblue' in design_name_lower:
            return '超大规模设计'
        elif 'edit_dist' in design_name_lower:
            return '编辑距离'
        else:
            return '未知类型'
    
    def _estimate_complexity(self, design_name: str) -> float:
        """估算设计复杂度"""
        # 基于设计名称的简单复杂度估算
        complexity_map = {
            'mgc_des_perf': 0.8,
            'mgc_fft': 0.7,
            'mgc_matrix_mult': 0.6,
            'mgc_pci_bridge32': 0.9,
            'mgc_superblue': 0.85,
            'mgc_edit_dist': 0.5
        }
        
        for key, value in complexity_map.items():
            if key in design_name.lower():
                return value
        
        return 0.5  # 默认复杂度
    
    def _extract_constraints(self, result_data: Dict[str, Any]) -> List[str]:
        """提取约束信息"""
        constraints = []
        stdout = result_data.get('stdout', '')
        
        # 基于stdout内容推断约束
        if 'timing' in stdout.lower():
            constraints.append('timing')
        if 'power' in stdout.lower():
            constraints.append('power')
        if 'area' in stdout.lower():
            constraints.append('area')
        if 'congestion' in stdout.lower():
            constraints.append('congestion')
        
        return constraints if constraints else ['timing']  # 默认约束
    
    def _parse_stdout_info(self, stdout: str) -> Dict[str, Any]:
        """解析stdout中的信息"""
        info = {}
        
        # 提取核心区域信息
        if 'CoreBBox:' in stdout:
            try:
                bbox_line = [line for line in stdout.split('\n') if 'CoreBBox:' in line][0]
                bbox_parts = bbox_line.split('CoreBBox:')[1].strip()
                info['core_bbox'] = bbox_parts
            except:
                pass
        
        # 提取行信息
        if 'Added' in stdout and 'rows' in stdout:
            try:
                rows_line = [line for line in stdout.split('\n') if 'Added' in line and 'rows' in line][0]
                info['rows_info'] = rows_line.strip()
            except:
                pass
        
        # 提取端口数量
        port_warnings = [line for line in stdout.split('\n') if 'toplevel port is not placed' in line]
        info['port_count'] = len(port_warnings)
        
        return info
    
    def create_case_entry(self, design_name: str, result_data: Dict[str, Any], design_files: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """创建案例条目"""
        # 生成唯一ID
        case_id = hashlib.md5(f"{design_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # 提取特征
        features = self.extract_design_features(design_name, result_data)
        
        # 创建案例条目
        case_entry = {
            'case_id': case_id,
            'design_name': design_name,
            'layout': {
                'verilog': design_files.get('verilog'),
                'def': design_files.get('def'),
                'tech_lef': design_files.get('tech_lef'),
                'cells_lef': design_files.get('cells_lef'),
                'placement_result': design_files.get('placement_result'),
                'placement_verilog': design_files.get('placement_verilog')
            },
            'optimization_result': {
                'success': features['success'],
                'execution_time': features['execution_time'],
                'wirelength': features['wirelength'],
                'area': features['area'],
                'stdout_summary': result_data.get('stdout', '')[:1000] + '...' if len(result_data.get('stdout', '')) > 1000 else result_data.get('stdout', '')
            },
            'metadata': {
                'design_type': features['design_type'],
                'complexity': features['complexity'],
                'constraints': features['constraints'],
                'port_count': features.get('port_count', 0),
                'core_bbox': features.get('core_bbox', ''),
                'rows_info': features.get('rows_info', ''),
                'import_date': features['timestamp'],
                'source': 'ISPD 2015 Contest Benchmark'
            },
            'features': {
                'text': f"{features['design_type']}设计，复杂度{features['complexity']}，约束{features['constraints']}",
                'structured': features
            }
        }
        
        return case_entry
    
    def import_all_cases(self):
        """导入所有案例"""
        logger.info("开始批量导入ISPD案例...")
        
        # 加载现有数据
        self.load_existing_data()
        
        # 获取所有结果文件
        result_files = list(self.results_dir.glob("*_result.json"))
        logger.info(f"找到 {len(result_files)} 个结果文件")
        
        # 处理每个结果文件
        for result_file in result_files:
            try:
                design_name = result_file.stem.replace('_result', '')
                logger.info(f"处理设计: {design_name}")
                
                # 读取结果数据
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 获取设计文件
                design_files = self.get_design_files(design_name)
                
                # 创建案例条目
                case_entry = self.create_case_entry(design_name, result_data, design_files)
                
                # 检查是否已存在
                existing_case = next((case for case in self.cases_data if case['design_name'] == design_name), None)
                if existing_case:
                    logger.info(f"更新现有案例: {design_name}")
                    # 更新现有案例
                    case_index = next(i for i, case in enumerate(self.cases_data) if case['design_name'] == design_name)
                    self.cases_data[case_index] = case_entry
                else:
                    logger.info(f"添加新案例: {design_name}")
                    # 添加新案例
                    self.cases_data.append(case_entry)
                    self.imported_cases += 1
                
            except Exception as e:
                logger.error(f"处理设计 {design_name} 失败: {e}")
                self.failed_cases += 1
        
        # 更新元数据
        self.metadata['total_cases'] = len(self.cases_data)
        self.metadata['successful_cases'] = self.imported_cases
        self.metadata['failed_cases'] = self.failed_cases
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        # 保存数据
        self.save_data()
        
        logger.info(f"导入完成！")
        logger.info(f"总案例数: {self.metadata['total_cases']}")
        logger.info(f"新增案例: {self.imported_cases}")
        logger.info(f"失败案例: {self.failed_cases}")
    
    def save_data(self):
        """保存数据到文件"""
        # 保存案例数据
        with open(self.cases_file, 'w', encoding='utf-8') as f:
            json.dump(self.cases_data, f, ensure_ascii=False, indent=2)
        logger.info(f"案例数据已保存到: {self.cases_file}")
        
        # 保存元数据
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"元数据已保存到: {self.metadata_file}")
    
    def generate_summary_report(self):
        """生成导入总结报告"""
        report = f"""
# ISPD 2015 知识库导入总结报告

## 导入统计
- 总案例数: {self.metadata['total_cases']}
- 新增案例: {self.imported_cases}
- 失败案例: {self.failed_cases}
- 导入时间: {self.metadata['import_date']}

## 设计类型分布
"""
        
        # 统计设计类型
        design_types = {}
        for case in self.cases_data:
            design_type = case['metadata']['design_type']
            design_types[design_type] = design_types.get(design_type, 0) + 1
        
        for design_type, count in design_types.items():
            report += f"- {design_type}: {count} 个\n"
        
        report += f"""
## 成功率统计
- 成功案例: {sum(1 for case in self.cases_data if case['optimization_result']['success'])} 个
- 失败案例: {sum(1 for case in self.cases_data if not case['optimization_result']['success'])} 个
- 成功率: {sum(1 for case in self.cases_data if case['optimization_result']['success']) / len(self.cases_data) * 100:.1f}%

## 文件结构
- 案例数据: {self.cases_file}
- 元数据: {self.metadata_file}
- 知识库目录: {self.knowledge_base_dir}
"""
        
        # 保存报告
        report_file = self.knowledge_base_dir / "import_summary.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"总结报告已保存到: {report_file}")
        return report

def main():
    """主函数"""
    importer = ISPDKnowledgeBaseImporter()
    importer.import_all_cases()
    importer.generate_summary_report()
    
    print("\n" + "="*50)
    print("ISPD知识库导入完成！")
    print("="*50)
    print(f"知识库位置: {importer.knowledge_base_dir}")
    print(f"案例数据: {importer.cases_file}")
    print(f"元数据: {importer.metadata_file}")
    print(f"总结报告: {importer.knowledge_base_dir}/import_summary.md")

if __name__ == "__main__":
    main() 