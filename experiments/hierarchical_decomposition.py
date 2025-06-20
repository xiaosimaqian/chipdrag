import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from chiprag.modules.hierarchy import HierarchicalDecomposer, Hierarchy, Node
from chiprag.modules.verilog_parser import parse_verilog

class HierarchicalDecompositionExperiment:
    def __init__(self, benchmark_dir: str):
        self.benchmark_dir = Path(benchmark_dir)
        self.decomposer = HierarchicalDecomposer()
        self.datasets = [
            "mgc_des_perf_1",
            "mgc_fft_1",
            "mgc_matrix_mult_1"
        ]
        
    def load_netlist(self, dataset_name: str) -> Dict:
        """加载netlist数据"""
        design_file = self.benchmark_dir / dataset_name / "design.v"
        if not design_file.exists():
            raise FileNotFoundError(f"找不到设计文件: {design_file}")
            
        # 使用Verilog解析器解析设计文件
        netlist = parse_verilog(str(design_file))
        
        # 将解析结果转换为层次化分解器需要的格式
        return self._convert_to_hierarchy_format(netlist)
    
    def _convert_to_hierarchy_format(self, netlist: Dict) -> Dict:
        """将Verilog解析结果转换为层次化分解器需要的格式"""
        hierarchy_format = {'modules': []}
        
        for module in netlist['modules']:
            # 创建功能模块
            functions = []
            for instance in module['instances']:
                function = {
                    'name': instance['name'],
                    'submodules': []
                }
                
                # 创建子模块
                for port in module['ports']:
                    submodule = {
                        'name': f"{instance['name']}_{port['name']}",
                        'cells': []
                    }
                    
                    # 创建单元
                    if port['width'] != '1':
                        for i in range(int(port['width'])):
                            cell = {
                                'name': f"{submodule['name']}_bit{i}"
                            }
                            submodule['cells'].append(cell)
                    else:
                        cell = {
                            'name': f"{submodule['name']}_bit"
                        }
                        submodule['cells'].append(cell)
                    
                    function['submodules'].append(submodule)
                
                functions.append(function)
            
            hierarchy_format['modules'].append({
                'name': module['name'],
                'functions': functions
            })
        
        return hierarchy_format
    
    def calculate_accuracy(self, predicted_hierarchy: Hierarchy, ground_truth: Dict) -> float:
        """计算分解准确率"""
        # 获取预测的模块列表
        predicted_modules = set()
        def collect_modules(node):
            predicted_modules.add(node.name)
            for child in node.children:
                collect_modules(child)
        
        if predicted_hierarchy.root:
            collect_modules(predicted_hierarchy.root)
        
        # 获取ground truth的模块列表
        ground_truth_modules = set()
        def collect_ground_truth(module):
            ground_truth_modules.add(module['name'])
            for func in module.get('functions', []):
                ground_truth_modules.add(func['name'])
                for sub in func.get('submodules', []):
                    ground_truth_modules.add(sub['name'])
                    for cell in sub.get('cells', []):
                        ground_truth_modules.add(cell['name'])
        
        for module in ground_truth.get('modules', []):
            collect_ground_truth(module)
        
        # 计算准确率
        if not ground_truth_modules:
            return 0.0
        
        correct = len(predicted_modules & ground_truth_modules)
        total = len(ground_truth_modules)
        
        return correct / total if total > 0 else 0.0
    
    def run_experiment(self) -> Dict[str, float]:
        """运行实验并返回结果"""
        results = {}
        
        for dataset in self.datasets:
            print(f"处理数据集: {dataset}")
            try:
                # 1. 加载netlist
                netlist = self.load_netlist(dataset)
                
                # 2. 执行层次化分解
                hierarchy = self.decomposer.decompose(netlist)
                
                # 3. 计算准确率
                accuracy = self.calculate_accuracy(hierarchy, netlist)
                
                results[dataset] = accuracy
                print(f"数据集 {dataset} 的分解准确率: {accuracy:.2%}")
                
            except Exception as e:
                print(f"处理数据集 {dataset} 时出错: {str(e)}")
                results[dataset] = 0.0
        
        return results
    
    def print_results(self, results: Dict[str, float]):
        """打印实验结果"""
        print("\n层次化分解准确率结果:")
        print("-" * 50)
        print(f"{'数据集':<20} {'准确率':<10}")
        print("-" * 50)
        for dataset, accuracy in results.items():
            print(f"{dataset:<20} {accuracy:.2%}")
        print("-" * 50)

def main():
    # 设置基准测试目录
    benchmark_dir = "/Users/keqin/Documents/workspace/chip-rag/chip_design/ispd_2015_contest_benchmark"
    
    # 创建实验实例
    experiment = HierarchicalDecompositionExperiment(benchmark_dir)
    
    # 运行实验
    results = experiment.run_experiment()
    
    # 打印结果
    experiment.print_results(results)

if __name__ == "__main__":
    main() 