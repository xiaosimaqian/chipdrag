# chiprag/tests/validate_results.py

import json
from pathlib import Path

class ResultValidator:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def save_results(self, design_name, hierarchy, components):
        """保存分解结果"""
        result = {
            'design_name': design_name,
            'hierarchy': self._serialize_hierarchy(hierarchy.root),
            'component_count': len(components),
            'timestamp': str(datetime.now())
        }
        
        output_file = self.results_dir / f"{design_name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

    def _serialize_hierarchy(self, node):
        """将层次结构序列化为字典"""
        return {
            'name': node.name,
            'type': node.type,
            'children': [self._serialize_hierarchy(child) for child in node.children]
        }

    def validate_results(self, design_name):
        """验证分解结果"""
        result_file = self.results_dir / f"{design_name}_results.json"
        if not result_file.exists():
            return False, "结果文件不存在"

        with open(result_file, 'r') as f:
            result = json.load(f)

        # 验证层次结构完整性
        if not self._validate_hierarchy(result['hierarchy']):
            return False, "层次结构不完整"

        return True, "验证通过"

    def _validate_hierarchy(self, node):
        """验证单个节点的层次结构"""
        if not all(k in node for k in ['name', 'type', 'children']):
            return False
        
        for child in node['children']:
            if not self._validate_hierarchy(child):
                return False
        
        return True