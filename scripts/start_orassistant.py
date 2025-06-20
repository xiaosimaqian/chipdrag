from flask import Flask, request, jsonify
import numpy as np
from typing import Dict, Any, List

app = Flask(__name__)

class ORAssistantService:
    def __init__(self):
        self.iterations = 0
    
    def generate_layout(self, design_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """生成布局"""
        # 初始化布局
        layout = self._initialize_layout(design_info)
        
        # 迭代优化
        for i in range(config.get('max_iterations', 1000)):
            self.iterations = i + 1
            
            # 计算力
            forces = self._calculate_forces(layout, design_info, config)
            
            # 更新位置
            self._update_positions(layout, forces, config)
            
            # 检查收敛
            if self._check_convergence(forces):
                break
        
        return layout
    
    def _initialize_layout(self, design_info: Dict[str, Any]) -> Dict[str, Any]:
        """初始化布局"""
        layout = {
            'components': {},
            'nets': [],
            'pins': []
        }
        
        # 为每个组件生成初始位置
        for comp_id, comp_info in design_info['components'].items():
            layout['components'][comp_id] = {
                'x': np.random.uniform(0, design_info['die_area'][2]),
                'y': np.random.uniform(0, design_info['die_area'][3]),
                'orientation': np.random.choice(['N', 'S', 'E', 'W'])
            }
        
        return layout
    
    def _calculate_forces(self, layout: Dict[str, Any], design_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, List[float]]:
        """计算力"""
        forces = {}
        force_strength = config.get('force_strength', 0.5)
        
        # 计算斥力（组件间）
        for comp_id, comp_info in layout['components'].items():
            forces[comp_id] = [0.0, 0.0]
            
            for other_id, other_info in layout['components'].items():
                if comp_id != other_id:
                    dx = comp_info['x'] - other_info['x']
                    dy = comp_info['y'] - other_info['y']
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0:
                        force = force_strength / (distance * distance)
                        forces[comp_id][0] += force * dx / distance
                        forces[comp_id][1] += force * dy / distance
        
        # 计算引力（连接间）
        for net in design_info.get('nets', []):
            source = net.get('source')
            sinks = net.get('sinks', [])
            
            if not source or not sinks:
                continue
                
            if source in layout['components']:
                for sink in sinks:
                    if sink in layout['components']:
                        source_pos = layout['components'][source]
                        target_pos = layout['components'][sink]
                        
                        dx = target_pos['x'] - source_pos['x']
                        dy = target_pos['y'] - source_pos['y']
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0:
                            force = -force_strength * distance
                            forces[source][0] += force * dx / distance
                            forces[source][1] += force * dy / distance
                            forces[sink][0] -= force * dx / distance
                            forces[sink][1] -= force * dy / distance
        
        return forces
    
    def _update_positions(self, layout: Dict[str, Any], forces: Dict[str, List[float]], config: Dict[str, Any]):
        """更新位置"""
        learning_rate = config.get('learning_rate', 0.01)
        
        for comp_id, force in forces.items():
            if comp_id in layout['components']:
                layout['components'][comp_id]['x'] += learning_rate * force[0]
                layout['components'][comp_id]['y'] += learning_rate * force[1]
    
    def _check_convergence(self, forces: Dict[str, List[float]]) -> bool:
        """检查收敛"""
        max_force = 0.0
        for force in forces.values():
            max_force = max(max_force, np.sqrt(force[0]*force[0] + force[1]*force[1]))
        return max_force < 0.1

# 创建ORAssistant服务实例
orassistant = ORAssistantService()

@app.route('/generate_layout', methods=['POST'])
def generate_layout():
    """生成布局API端点"""
    data = request.get_json()
    design_info = data.get('design_info', {})
    config = data.get('config', {})
    
    try:
        layout = orassistant.generate_layout(design_info, config)
        return jsonify(layout)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 