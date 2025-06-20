# 修改 modules/knowledge/graph.py
from typing import Dict, Any, List, Optional

class MultiModalKnowledgeGraph:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes = {}
        self.edges = {}
        
    def add_node(self, node_data: Dict[str, Any]) -> None:
        node_id = node_data.get('id')
        if node_id:
            self.nodes[node_id] = node_data
            
    def add_edge(self, source: str, target: str, edge_data: Dict[str, Any]) -> None:
        if source in self.nodes and target in self.nodes:
            edge_id = f"{source}_{target}"
            self.edges[edge_id] = {
                "source": source,
                "target": target,
                **edge_data
            }
            
    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        # 实现查询逻辑
        return results
        
    def visualize(self) -> str:
        # 实现可视化逻辑
        return "graph visualization"