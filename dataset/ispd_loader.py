# chiprag/dataset/ispd_loader.py

import os
from pathlib import Path
import re

class ISPDLoader:
    def __init__(self, benchmark_dir):
        self.benchmark_dir = Path(benchmark_dir)
        self.designs = self._scan_designs()

    def _scan_designs(self):
        """扫描所有可用的设计"""
        designs = []
        for design_dir in self.benchmark_dir.iterdir():
            if design_dir.is_dir() and not design_dir.name.startswith('.'):
                verilog_file = design_dir / 'design.v'
                def_file = design_dir / 'floorplan.def'
                if verilog_file.exists() and def_file.exists():
                    designs.append({
                        'name': design_dir.name,
                        'verilog_path': str(verilog_file),
                        'def_path': str(def_file)
                    })
        return designs

    def load_design(self, design_name):
        """加载指定设计的网表和DEF文件"""
        design = next((d for d in self.designs if d['name'] == design_name), None)
        if not design:
            raise ValueError(f"Design {design_name} not found")

        # 加载Verilog网表
        with open(design['verilog_path'], 'r') as f:
            verilog_content = f.read()

        # 加载DEF文件
        with open(design['def_path'], 'r') as f:
            def_content = f.read()

        return {
            'name': design['name'],
            'verilog': verilog_content,
            'def': def_content
        }

    def parse_verilog(self, verilog_content):
        """解析Verilog网表，提取模块层次结构"""
        modules = []
        current_module = None
        
        # 简单的Verilog解析器
        for line in verilog_content.split('\n'):
            line = line.strip()
            
            # 匹配模块定义
            module_match = re.match(r'module\s+(\w+)', line)
            if module_match:
                current_module = {
                    'name': module_match.group(1),
                    'ports': [],
                    'instances': []
                }
                modules.append(current_module)
                continue

            # 匹配实例化
            inst_match = re.match(r'\s*(\w+)\s+(\w+)\s*\(', line)
            if inst_match and current_module:
                current_module['instances'].append({
                    'type': inst_match.group(1),
                    'name': inst_match.group(2)
                })

        return modules

    def parse_def(self, def_content):
        """解析DEF文件，提取布局信息"""
        # 提取组件位置信息
        components = []
        for line in def_content.split('\n'):
            if line.startswith('- '):  # 组件定义行
                parts = line.split()
                if len(parts) >= 4:
                    components.append({
                        'name': parts[1],
                        'x': int(parts[2]),
                        'y': int(parts[3])
                    })
        return components