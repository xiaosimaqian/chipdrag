import os
import logging
from typing import Dict, Any, List, Tuple
from .verilog_parser import VerilogParser
from .def_parser import DEFParser
from .lef_parser import LEFParser

logger = logging.getLogger(__name__)

class DesignParser:
    def __init__(self):
        self.verilog_parser = VerilogParser()
        self.def_parser = DEFParser()
        self.lef_parser = LEFParser()
        self.data = {}
        
    def parse_design(self, design_dir: str) -> Dict[str, Any]:
        """解析设计文件
        
        Args:
            design_dir: 设计目录路径
            
        Returns:
            解析后的数据字典
        """
        design_dir = os.path.abspath(design_dir)
        logger.info(f"开始解析设计目录: {design_dir}")
        
        try:
            # 解析 Verilog 文件
            verilog_file = os.path.join(design_dir, "design.v")
            if os.path.exists(verilog_file):
                logger.info(f"解析 Verilog 文件: {verilog_file}")
                verilog_data = self.verilog_parser.parse_verilog(verilog_file)
                self.data.update(verilog_data)
            
            # 解析 DEF 文件
            def_file = os.path.join(design_dir, "floorplan.def")
            if os.path.exists(def_file):
                logger.info(f"解析 DEF 文件: {def_file}")
                def_data = self.def_parser.parse_def(def_file)
                self.data['floorplan'] = def_data
            
            # 解析 LEF 文件
            tech_lef = os.path.join(design_dir, "tech.lef")
            cells_lef = os.path.join(design_dir, "cells.lef")
            if os.path.exists(tech_lef):
                logger.info(f"解析 LEF 文件: {tech_lef}, {cells_lef}")
                lef_data = self.lef_parser.parse_lef(tech_lef, cells_lef)
                self.data['technology'] = lef_data
            
            # 解析约束文件
            constraints_file = os.path.join(design_dir, "placement.constraints")
            if os.path.exists(constraints_file):
                logger.info(f"解析约束文件: {constraints_file}")
                self._parse_constraints(constraints_file)
            
            # 生成层次结构
            self._generate_hierarchy()
            
            logger.info("设计解析完成")
            return self.data
            
        except Exception as e:
            logger.error(f"解析设计时出错: {str(e)}")
            raise
            
    def _parse_constraints(self, constraints_file: str):
        """解析约束文件
        
        Args:
            constraints_file: 约束文件路径
        """
        try:
            with open(constraints_file, 'r') as f:
                content = f.read()
                
            # 解析约束
            constraints = {}
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) >= 2:
                    constraint_type = parts[0]
                    constraint_value = parts[1]
                    constraints[constraint_type] = constraint_value
                    
            self.data['constraints'] = constraints
            
        except Exception as e:
            logger.error(f"解析约束文件时出错: {str(e)}")
            raise
            
    def _generate_hierarchy(self):
        """生成设计层次结构"""
        try:
            hierarchy = {
                'name': self.data.get('DESIGN', 'unknown'),
                'type': 'design',
                'children': []
            }
            
            # 从 Verilog 数据中提取模块层次
            for module in self.data.get('modules', []):
                module_node = {
                    'name': module['name'],
                    'type': 'module',
                    'children': []
                }
                
                # 添加实例
                for instance in module.get('instances', []):
                    instance_node = {
                        'name': instance['name'],
                        'type': 'instance',
                        'module': instance['type'],
                        'parameters': instance.get('parameters', {}),
                        'connections': instance.get('connections', {})
                    }
                    module_node['children'].append(instance_node)
                
                hierarchy['children'].append(module_node)
            
            self.data['hierarchy'] = hierarchy
            
        except Exception as e:
            logger.error(f"生成层次结构时出错: {str(e)}")
            raise