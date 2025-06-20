import os
import json
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: str
    properties: tuple
    
    @staticmethod
    def from_dict(id: str, name: str, type_: str, properties: dict):
        # dict转为tuple，便于哈希
        return Entity(id=id, name=name, type=type_, properties=tuple(sorted(properties.items())))

@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    type: str
    properties: tuple
    
    @staticmethod
    def from_dict(source: str, target: str, type_: str, properties: dict):
        return Relation(source=source, target=target, type=type_, properties=tuple(sorted(properties.items())))

class KnowledgeGraphCollector:
    def __init__(self):
        self.output_dir = Path("chiprag/data/knowledge_base")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实体和关系集合
        self.entities: Set[Entity] = set()
        self.relations: Set[Relation] = set()
        
        # 实体类型映射
        self.entity_types = {
            "module": "模块",
            "cell": "单元",
            "net": "网络",
            "pin": "引脚",
            "port": "端口",
            "block": "功能块",
            "constraint": "约束",
            "technology": "工艺"
        }
        
        # 关系类型映射
        self.relation_types = {
            "connects": "连接",
            "contains": "包含",
            "depends_on": "依赖",
            "implements": "实现",
            "optimizes": "优化",
            "constrains": "约束"
        }
        
        # 添加基础实体
        self._add_base_entities()
    
    def _add_base_entities(self):
        """添加基础实体"""
        self.entities.add(Entity.from_dict(
            id="tech_45nm",
            name="45nm工艺",
            type_="technology",
            properties={
                "node": "45nm",
                "category": "technology_node"
            }
        ))
        
        self.entities.add(Entity.from_dict(
            id="isa_riscv",
            name="RISC-V ISA",
            type_="module",
            properties={
                "category": "instruction_set",
                "open_source": True
            }
        ))
        
        self.entities.add(Entity.from_dict(
            id="optimization_placement",
            name="布局优化",
            type_="optimization",
            properties={
                "category": "physical_design",
                "description": "VLSI布局优化"
            }
        ))
    
    def collect_all(self):
        """收集所有知识图谱数据"""
        logger.info("开始收集知识图谱数据...")
        
        # 从ISPD基准测试收集
        self._collect_from_ispd()
        
        # 从OpenCores收集
        self._collect_from_opencores()
        
        # 从DREAMPlace收集
        self._collect_from_dreamplace()
        
        # 保存数据
        self._save_data()
        
        logger.info("知识图谱数据收集完成！")
    
    def _collect_from_ispd(self):
        """从ISPD基准测试收集数据"""
        logger.info("从ISPD基准测试收集数据...")
        
        # 基准测试列表
        benchmarks = [
            "mgc_des_perf_a", "mgc_des_perf_b",
            "mgc_edit_dist_a", "mgc_fft_a", "mgc_fft_b",
            "mgc_matrix_mult_a", "mgc_matrix_mult_b",
            "mgc_pci_bridge32_a", "mgc_pci_bridge32_b",
            "mgc_superblue11_a", "superblue12_es_gr",
            "mgc_superblue16_a"
        ]
        
        for bench in benchmarks:
            self.entities.add(Entity.from_dict(
                id=f"ispd_{bench}",
                name=bench,
                type_="module",
                properties={
                    "source": "ISPD",
                    "year": 2015,
                    "category": "benchmark"
                }
            ))
            
            self.relations.add(Relation.from_dict(
                source=f"ispd_{bench}",
                target="tech_45nm",
                type_="implements",
                properties={"year": 2015}
            ))
    
    def _collect_from_opencores(self):
        """从OpenCores收集数据"""
        logger.info("从OpenCores收集数据...")
        
        # OpenCores项目列表
        projects = [
            "riscv", "openrisc", "mor1kx",
            "picoRV32", "picorv32", "minerva"
        ]
        
        for project in projects:
            self.entities.add(Entity.from_dict(
                id=f"opencores_{project}",
                name=project,
                type_="module",
                properties={
                    "source": "OpenCores",
                    "category": "open_source"
                }
            ))
            
            if project in ["riscv", "picoRV32", "picorv32"]:
                self.relations.add(Relation.from_dict(
                    source=f"opencores_{project}",
                    target="isa_riscv",
                    type_="implements",
                    properties={"category": "processor"}
                ))
    
    def _collect_from_dreamplace(self):
        """从DREAMPlace收集数据"""
        logger.info("从DREAMPlace收集数据...")
        
        self.entities.add(Entity.from_dict(
            id="tool_dreamplace",
            name="DREAMPlace",
            type_="module",
            properties={
                "source": "GitHub",
                "category": "placement_tool",
                "description": "Deep learning toolkit for VLSI placement"
            }
        ))
        
        self.relations.add(Relation.from_dict(
            source="tool_dreamplace",
            target="optimization_placement",
            type_="optimizes",
            properties={"category": "deep_learning"}
        ))
    
    def _save_data(self):
        """保存收集的数据"""
        # 保存实体数据
        entities_data = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "properties": dict(e.properties)
            }
            for e in self.entities
        ]
        
        with open(self.output_dir / "entities.json", "w") as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        
        # 保存关系数据
        relations_data = [
            {
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "properties": dict(r.properties)
            }
            for r in self.relations
        ]
        
        with open(self.output_dir / "relations.json", "w") as f:
            json.dump(relations_data, f, indent=2, ensure_ascii=False)
        
        # 生成三元组数据
        triples = []
        for relation in self.relations:
            # 获取源实体和目标实体
            source_entity = next(e for e in self.entities if e.id == relation.source)
            target_entity = next(e for e in self.entities if e.id == relation.target)
            
            # 添加三元组
            triples.append({
                "head": {
                    "id": source_entity.id,
                    "name": source_entity.name,
                    "type": source_entity.type
                },
                "relation": {
                    "type": relation.type,
                    "properties": dict(relation.properties)
                },
                "tail": {
                    "id": target_entity.id,
                    "name": target_entity.name,
                    "type": target_entity.type
                }
            })
        
        with open(self.output_dir / "triples.json", "w") as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已保存 {len(entities_data)} 个实体")
        logger.info(f"已保存 {len(relations_data)} 个关系")
        logger.info(f"已保存 {len(triples)} 个三元组")

def main():
    collector = KnowledgeGraphCollector()
    collector.collect_all()

if __name__ == "__main__":
    main() 