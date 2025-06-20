import unittest
from typing import Dict, Any
from modules.knowledge.graph import MultiModalKnowledgeGraph

class TestKnowledgeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            'graph': {
                'node_types': ['component', 'module', 'system'],
                'edge_types': ['contains', 'connects', 'depends']
            }
        }
        cls.knowledge_graph = MultiModalKnowledgeGraph(cls.config)
        
    def test_graph_initialization(self):
        """测试图谱初始化"""
        self.assertTrue(hasattr(self.knowledge_graph, 'add_node'))
        self.assertTrue(hasattr(self.knowledge_graph, 'add_edge'))
        self.assertTrue(hasattr(self.knowledge_graph, 'query'))
        self.assertTrue(hasattr(self.knowledge_graph, 'visualize'))
        
    def test_node_addition(self):
        """测试节点添加"""
        node_data = {
            'id': 'test_node',
            'type': 'component',
            'properties': {
                'name': 'Test Component',
                'area': 100
            }
        }
        self.knowledge_graph.add_node(node_data)
        self.assertIn('test_node', self.knowledge_graph.nodes)
        
    def test_edge_addition(self):
        """测试边添加"""
        source_node = {
            'id': 'source',
            'type': 'component'
        }
        target_node = {
            'id': 'target',
            'type': 'component'
        }
        self.knowledge_graph.add_node(source_node)
        self.knowledge_graph.add_node(target_node)
        
        edge_data = {
            'type': 'connects',
            'properties': {
                'weight': 1.0
            }
        }
        self.knowledge_graph.add_edge('source', 'target', edge_data)
        self.assertIn('source_target', self.knowledge_graph.edges)
        
    def test_graph_query(self):
        """测试图谱查询"""
        node = {
            'id': 'query_node',
            'type': 'component'
        }
        self.knowledge_graph.add_node(node)
        
        query = {
            'type': 'component',
            'properties': {
                'name': 'Query Component'
            }
        }
        results = self.knowledge_graph.query(query)
        self.assertIsInstance(results, list)
        
    def test_graph_visualization(self):
        """测试图谱可视化"""
        node = {
            'id': 'viz_node',
            'type': 'component'
        }
        self.knowledge_graph.add_node(node)
        
        viz = self.knowledge_graph.visualize()
        self.assertIsInstance(viz, str)
        self.assertGreater(len(viz), 0)

def main():
    unittest.main()

if __name__ == '__main__':
    main() 