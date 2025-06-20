import os
import sys
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Generator
import numpy as np
from chiprag.modules.hierarchy import HierarchicalDecomposer, Hierarchy, Node
from chiprag.modules.verilog_parser import parse_verilog
from chiprag.modules.glayout import GLayout
from chiprag.modules.orassistant import ORAssistant
from .hierarchical_decomposition import HierarchicalDecompositionExperiment

class ComparisonExperiment:
    def __init__(self):
        self.benchmark_dir = Path('chip_design/source/documents/riscv0裁剪版')
        self.design_file = 'riscv0_cl028_synth.v'
        self.methods = {
            'GLayout': GLayout(),
            'ORAssistant': ORAssistant(),
            'HierarchicalDecomposer': HierarchicalDecomposer()
        }
        self.results = {method: 0.0 for method in self.methods.keys()}
        self.batch_size = 100  # 每批处理的模块数
        
    def _process_modules_in_batches(self, netlist: Dict) -> Generator[Dict, None, None]:
        """分批处理模块"""
        if 'modules' not in netlist:
            return
            
        total_modules = len(netlist['modules'])
        for i in range(0, total_modules, self.batch_size):
            batch = {
                'modules': netlist['modules'][i:i + self.batch_size]
            }
            yield batch
            
    def run_comparison(self):
        """运行比较实验"""
        print(f"\n开始处理设计文件: {self.design_file}")
        start_time = time.time()
        
        # 加载设计文件
        design_path = self.benchmark_dir / self.design_file
        print(f"正在解析Verilog文件...")
        netlist = parse_verilog(str(design_path))
        
        # 清理内存
        gc.collect()
        
        # 运行每种方法
        for method_name, method in self.methods.items():
            print(f"\n运行 {method_name}...")
            method_start = time.time()
            
            try:
                total_accuracy = 0.0
                batch_count = 0
                
                # 分批处理
                for batch in self._process_modules_in_batches(netlist):
                    # 运行方法
                    result = method.decompose(batch)
                    
                    # 计算准确率
                    accuracy = self._calculate_accuracy(result, batch)
                    total_accuracy += accuracy
                    batch_count += 1
                    
                    # 清理内存
                    gc.collect()
                
                # 计算平均准确率
                if batch_count > 0:
                    self.results[method_name] = total_accuracy / batch_count
                
                method_time = time.time() - method_start
                print(f"{method_name} 完成，耗时: {method_time:.2f}秒")
                print(f"准确率: {self.results[method_name]:.2f}%")
                
            except Exception as e:
                print(f"{method_name} 运行出错: {str(e)}")
                self.results[method_name] = 0.0
            
            # 清理内存
            gc.collect()
        
        total_time = time.time() - start_time
        print(f"\n实验完成，总耗时: {total_time:.2f}秒")
        
    def _calculate_accuracy(self, result: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """计算准确率"""
        if not result or not ground_truth:
            return 0.0
            
        # 获取所有模块信息
        result_modules = self._get_all_modules(result)
        truth_modules = self._get_all_modules(ground_truth)
        
        # 计算匹配的模块数
        matched = 0
        for r_module in result_modules:
            for t_module in truth_modules:
                if (r_module['name'] == t_module['name'] and 
                    r_module['type'] == t_module['type']):
                    matched += 1
                    break
        
        # 计算准确率
        total = len(truth_modules)
        if total == 0:
            return 0.0
            
        return (matched / total) * 100
        
    def _get_all_modules(self, netlist: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取所有模块信息"""
        modules = []
        if 'modules' in netlist:
            for module in netlist['modules']:
                modules.append({
                    'name': module['name'],
                    'type': module['type']
                })
                if 'submodules' in module:
                    for submodule in module['submodules']:
                        modules.append({
                            'name': submodule['name'],
                            'type': submodule['type']
                        })
        return modules
        
    def print_results(self):
        """打印实验结果"""
        print("\n实验结果:")
        print("-" * 50)
        for method, accuracy in self.results.items():
            print(f"{method}: {accuracy:.2f}%")
        print("-" * 50)

def main():
    experiment = ComparisonExperiment()
    experiment.run_comparison()
    experiment.print_results()

if __name__ == "__main__":
    main() 