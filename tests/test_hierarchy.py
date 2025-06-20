# chiprag/tests/test_hierarchy.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset.ispd_loader import ISPDLoader
from chiprag.modules.core.hierarchy import HierarchicalDecomposer

def test_hierarchical_decomposition():
    # 初始化数据加载器
    loader = ISPDLoader('chip_design/ispd_2015_contest_benchmark')
    
    # 选择测试用例
    test_designs = ['mgc_fft_1', 'mgc_matrix_mult_1']  # 从小规模设计开始测试
    
    for design_name in test_designs:
        print(f"\n测试设计: {design_name}")
        print("=" * 50)
        
        # 加载设计数据
        design_data = loader.load_design(design_name)
        
        # 解析网表
        modules = loader.parse_verilog(design_data['verilog'])
        print(f"发现 {len(modules)} 个模块")
        
        # 层次化分解
        decomposer = HierarchicalDecomposer()
        hierarchy = decomposer.decompose({
            'modules': modules,
            'name': design_name
        })
        
        # 打印层次结构
        def print_hierarchy(node, level=0):
            indent = "  " * level
            print(f"{indent}{node.name} ({node.type})")
            for child in node.children:
                print_hierarchy(child, level + 1)
        
        print("\n层次化分解结果:")
        print_hierarchy(hierarchy.root)
        
        # 验证布局信息
        components = loader.parse_def(design_data['def'])
        print(f"\n布局组件数量: {len(components)}")
        
        # 输出一些统计信息
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count
        
        total_nodes = count_nodes(hierarchy.root)
        print(f"层次树节点总数: {total_nodes}")

if __name__ == '__main__':
    test_hierarchical_decomposition()