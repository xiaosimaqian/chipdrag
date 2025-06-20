import unittest
from modules.core.hierarchy import Node, Hierarchy, HierarchicalDecomposer
from modules.knowledge.transfer import KnowledgeTransfer
from modules.core.granularity_retriever import MultiGranularityRetrieval
from modules.utils.llm_manager import LLMManager

class TestMultiGranularityRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            'llm_config': {
                'base_url': 'http://localhost:11434',
                'model_name': 'gpt2',
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
        cls.llm_manager = LLMManager(cls.config['llm_config'])
        
    def setUp(self):
        """设置测试环境"""
        self.config = {
            'knowledge_base': {
                'path': '/tmp/chiprag_test',
                'format': 'json',
                'text_path': '/tmp/chiprag_test/text',
                'image_path': '/tmp/chiprag_test/images',
                'structured_data_path': '/tmp/chiprag_test/structured',
                'graph_path': '/tmp/chiprag_test/graph',
                'layout_experience_path': '/tmp/chiprag_test/layout',
                'cache_dir': '/tmp/chiprag_test/cache'
            },
            'hierarchy_config': {
                'max_depth': 3,
                'min_components': 2,
                'max_components': 10,
                'similarity_threshold': 0.8
            },
            'llm_config': {
                'base_url': 'http://localhost:8000',
                'model_name': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        }
        
        # 初始化 LLM 管理器
        self.llm_manager = LLMManager(self.config['llm_config'])
        
        # 构造 mock 层次结构
        self.hierarchy = Hierarchy(llm_manager=self.llm_manager)
        system_node = Node(name="system1", type="system", features={})
        module_node = Node(name="module1", type="module", features={})
        component_node = Node(name="component1", type="component", features={})
        self.hierarchy.add_node(system_node)
        self.hierarchy.add_node(module_node)
        self.hierarchy.add_node(component_node)
        system_node.children = [module_node]
        module_node.children = [component_node]
        
        # 初始化检索器
        self.retriever = MultiGranularityRetrieval(self.config)
        self.knowledge_transfer = KnowledgeTransfer(self.config)

    def test_knowledge_transfer(self):
        """测试知识迁移功能"""
        # 获取源节点和目标节点
        source_node = self.hierarchy.nodes["module1"]
        target_node = self.hierarchy.nodes["component1"]
        
        # 执行知识迁移
        transferred_knowledge = self.knowledge_transfer.transfer(source_node, target_node)
        
        # 验证迁移结果
        self.assertIsNotNone(transferred_knowledge)
        self.assertIn('features', transferred_knowledge)

    def test_multi_granularity_retrieval(self):
        """测试多粒度检索功能"""
        # 创建测试查询
        query = {
            'name': 'test_query',
            'features': {
                'structural': {'num_children': 2},
                'functional': {},
                'performance': {}
            }
        }
        
        # 执行检索
        results = self.retriever.retrieve(query, top_k=3)
        
        # 验证检索结果
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        # 验证结果格式
        for node, score in results:
            self.assertIsInstance(node, Node)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_retrieval_evaluation(self):
        """测试检索评估功能"""
        # 创建测试查询
        query = {
            'name': 'test_query',
            'features': {
                'structural': {'num_children': 2},
                'functional': {},
                'performance': {}
            }
        }
        
        # 创建ground truth
        ground_truth = [self.hierarchy.nodes["module1"]]
        
        # 执行评估
        metrics = self.retriever.evaluate_retrieval(query, ground_truth)
        
        # 验证评估指标
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # 验证指标值范围
        for metric in metrics.values():
            self.assertGreaterEqual(metric, 0)
            self.assertLessEqual(metric, 1)

if __name__ == '__main__':
    unittest.main() 