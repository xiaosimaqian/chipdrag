import os
import json
import logging
from typing import Dict, List, Set, Tuple
from ..parsers.verilog_parser import VerilogParser

class KnowledgeGraphBuilder:
    def __init__(self, output_dir: str = "chiprag/data/knowledge_base"):
        self.output_dir = output_dir
        self.entities: Dict[str, Dict] = {}
        self.relations: Dict[str, Dict] = {}
        self.triples: List[Dict] = []
        
    def build_from_verilog(self, verilog_file: str) -> None:
        """从Verilog文件构建知识图谱数据"""
        # 解析Verilog文件
        module_info_list = VerilogParser.parse_verilog(verilog_file)
        
        # 处理每个模块
        for module_info in module_info_list:
            module_name = module_info['name']
            module_data = module_info
            
            # 添加模块实体
            self._add_module_entity(module_name, module_data)
            
            # 添加端口实体和关系
            self._add_port_entities_and_relations(module_name, module_data.get('ports', []))
            
            # 添加实例实体和关系
            self._add_instance_entities_and_relations(module_name, module_data.get('instances', []))
        
        # 保存知识图谱数据
        self._save_knowledge_graph()
    
    def _add_module_entity(self, module_name: str, module_data: Dict) -> None:
        """添加模块实体"""
        entity = {
            "id": f"module_{module_name}",
            "type": "module",
            "name": module_name,
            "properties": {
                "parameter_count": module_data.get('parameter_count', 0),
                "port_count": module_data.get('port_count', 0),
                "instance_count": module_data.get('instance_count', 0),
                "submodule_count": module_data.get('submodule_count', 0)
            }
        }
        self.entities[entity["id"]] = entity
    
    def _add_port_entities_and_relations(self, module_name: str, ports: List[Dict]) -> None:
        """添加端口实体和关系"""
        for port in ports:
            # 添加端口实体
            port_id = f"port_{module_name}_{port['name']}"
            entity = {
                "id": port_id,
                "type": "port",
                "name": port['name'],
                "properties": {
                    "direction": port['type'],
                    "width": port.get('width', 1)
                }
            }
            self.entities[port_id] = entity
            
            # 添加端口-模块关系
            self.triples.append({
                "head": port_id,
                "relation": "belongs_to",
                "tail": f"module_{module_name}"
            })
    
    def _add_instance_entities_and_relations(self, module_name: str, instances: List[Dict]) -> None:
        """添加实例实体和关系"""
        for instance in instances:
            # 添加实例实体
            instance_id = f"instance_{module_name}_{instance['name']}"
            entity = {
                "id": instance_id,
                "type": "instance",
                "name": instance['name'],
                "properties": {
                    "module_type": instance['type']
                }
            }
            self.entities[instance_id] = entity
            
            # 添加实例-模块关系
            self.triples.append({
                "head": instance_id,
                "relation": "instance_of",
                "tail": f"module_{module_name}"
            })
            
            # 添加实例-子模块关系
            if instance['type'] != 'ms00f80':  # 如果不是基本单元
                self.triples.append({
                    "head": instance_id,
                    "relation": "is_submodule",
                    "tail": f"module_{instance['type']}"
                })
    
    def _save_knowledge_graph(self) -> None:
        """保存知识图谱数据到文件"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存实体数据
        with open(os.path.join(self.output_dir, "entities.json"), "w") as f:
            json.dump(list(self.entities.values()), f, indent=2)
        
        # 保存关系数据
        relations = {
            "belongs_to": {"type": "port_to_module", "description": "端口属于模块"},
            "instance_of": {"type": "instance_to_module", "description": "实例属于模块"},
            "is_submodule": {"type": "instance_to_submodule", "description": "实例是子模块"}
        }
        with open(os.path.join(self.output_dir, "relations.json"), "w") as f:
            json.dump(relations, f, indent=2)
        
        # 保存三元组数据
        with open(os.path.join(self.output_dir, "triples.json"), "w") as f:
            json.dump(self.triples, f, indent=2) 