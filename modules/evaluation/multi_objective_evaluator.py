from typing import Dict, List, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MultiObjectiveEvaluator:
    def __init__(self, config: Dict):
        """初始化多目标评估器
        
        Args:
            config: 评估器配置
        """
        self.config = config
        self.weights = config.get('weights', {
            'wirelength': 0.3,
            'congestion': 0.3,
            'timing': 0.4
        })
        self.thresholds = config.get('thresholds', {
            'wirelength': 1.0,
            'congestion': 1.0,
            'timing': 1.0
        })
        self.metrics = config.get('metrics', {
            'wirelength': {
                'type': 'minimize',
                'weight': 0.3,
                'threshold': 1.0
            },
            'congestion': {
                'type': 'minimize',
                'weight': 0.3,
                'threshold': 1.0
            },
            'timing': {
                'type': 'minimize',
                'weight': 0.4,
                'threshold': 1.0
            }
        })
        
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估布局质量
        
        Args:
            data: 包含布局和指标的字典
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 获取布局和指标数据
            layout = data.get('layout', {})
            metrics = data.get('metrics', {})
            
            # 1. 计算各项指标的得分
            wirelength = metrics.get('wirelength', self._calculate_wirelength(layout))
            congestion = metrics.get('congestion', self._calculate_congestion(layout))
            timing = metrics.get('timing', self._calculate_timing(layout))
            
            wirelength_score = self._normalize_score(wirelength, 'wirelength')
            congestion_score = self._normalize_score(congestion, 'congestion')
            timing_score = self._normalize_score(timing, 'timing')
            
            # 2. 计算加权总分
            overall_score = (
                self.weights['wirelength'] * wirelength_score +
                self.weights['congestion'] * congestion_score +
                self.weights['timing'] * timing_score
            )
            
            # 3. 计算约束满足度
            constraint_satisfaction = self._evaluate_constraints(data)
            
            # 4. 生成评估报告
            evaluation_report = self._generate_report(
                wirelength_score,
                congestion_score,
                timing_score,
                overall_score,
                constraint_satisfaction
            )
            
            # 5. 返回评估结果
            return {
                'wirelength': wirelength,
                'wirelength_score': wirelength_score,
                'congestion': congestion,
                'congestion_score': congestion_score,
                'timing': timing,
                'timing_score': timing_score,
                'overall_score': overall_score,
                'constraint_satisfaction': constraint_satisfaction,
                'evaluation_report': evaluation_report,
                'details': {
                    'wirelength': {
                        'value': wirelength,
                        'threshold': self.thresholds['wirelength'],
                        'score': wirelength_score,
                        'weight': self.weights['wirelength']
                    },
                    'congestion': {
                        'value': congestion,
                        'threshold': self.thresholds['congestion'],
                        'score': congestion_score,
                        'weight': self.weights['congestion']
                    },
                    'timing': {
                        'value': timing,
                        'threshold': self.thresholds['timing'],
                        'score': timing_score,
                        'weight': self.weights['timing']
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            return {
                'wirelength': 0.0,
                'wirelength_score': 0.0,
                'congestion': 0.0,
                'congestion_score': 0.0,
                'timing': 0.0,
                'timing_score': 0.0,
                'overall_score': 0.0,
                'constraint_satisfaction': 0.0,
                'evaluation_report': f"评估失败: {str(e)}",
                'error': str(e)
            }

    def _normalize_score(self, value: float, metric: str) -> float:
        """归一化分数
        
        Args:
            value: 原始值
            metric: 指标名称
            
        Returns:
            归一化后的分数
        """
        logger.info(f"_normalize_score: value={value}, metric={metric}")
        logger.info(f"_normalize_score: self.metrics={self.metrics}")
        logger.info(f"_normalize_score: self.metrics type={type(self.metrics)}")
        if metric in self.metrics:
            metric_config = self.metrics[metric]
            logger.info(f"_normalize_score: metric_config={metric_config}")
            logger.info(f"_normalize_score: metric_config type={type(metric_config)}")
            threshold = metric_config['threshold']
            logger.info(f"_normalize_score: threshold={threshold}")
            
            if metric_config['type'] == 'minimize':
                return min(1.0, threshold / (value + 1e-6))
            else:  # maximize
                return min(1.0, value / (threshold + 1e-6))
        return value
        
    def _calculate_wirelength(self, layout: Dict) -> float:
        """计算线长
        
        Args:
            layout: 布局数据
            
        Returns:
            线长得分
        """
        logger.info(f"_calculate_wirelength: layout={layout}")
        try:
            # 1. 获取布线信息
            nets = layout.get('nets', [])
            logger.info(f"_calculate_wirelength: nets={nets}")
            if not nets:
                return 0.0
                
            # 2. 计算每条线的长度
            total_length = 0.0
            for net in nets:
                logger.info(f"_calculate_wirelength: processing net={net}")
                source = net.get('source')
                target = net.get('target')
                logger.info(f"_calculate_wirelength: source={source}, target={target}")
                
                # 获取源组件和目标组件的位置
                components = layout.get('components', [])
                logger.info(f"_calculate_wirelength: components={components}")
                source_comp = next((c for c in components if c.get('name') == source), None)
                target_comp = next((c for c in components if c.get('name') == target), None)
                logger.info(f"_calculate_wirelength: source_comp={source_comp}, target_comp={target_comp}")
                
                if source_comp and target_comp:
                    source_pos = source_comp.get('position', {})
                    target_pos = target_comp.get('position', {})
                    logger.info(f"_calculate_wirelength: source_pos={source_pos}, target_pos={target_pos}")
                    
                    # 获取坐标
                    if isinstance(source_pos, dict):
                        x1 = float(source_pos.get('x', 0))
                        y1 = float(source_pos.get('y', 0))
                    else:
                        x1, y1 = map(float, source_pos if isinstance(source_pos, (list, tuple)) else (0, 0))
                        
                    if isinstance(target_pos, dict):
                        x2 = float(target_pos.get('x', 0))
                        y2 = float(target_pos.get('y', 0))
                    else:
                        x2, y2 = map(float, target_pos if isinstance(target_pos, (list, tuple)) else (0, 0))
                    
                    logger.info(f"_calculate_wirelength: coordinates: ({x1}, {y1}) -> ({x2}, {y2})")
                    
                    # 计算曼哈顿距离
                    length = abs(x2 - x1) + abs(y2 - y1)
                    total_length += length
                    
            logger.info(f"_calculate_wirelength: total_length={total_length}")
            return total_length
            
        except Exception as e:
            logger.error(f"计算线长失败: {str(e)}")
            return 0.0
        
    def _calculate_congestion(self, layout: Dict) -> float:
        """计算拥塞度
        
        Args:
            layout: 布局数据
            
        Returns:
            拥塞度得分
        """
        print(f"_calculate_congestion: layout={layout}")
        try:
            # 1. 获取布局信息
            components = layout.get('components', [])
            if not components:
                return 0.0
                
            # 2. 创建网格
            grid_size = 10  # 10x10网格
            grid = np.zeros((grid_size, grid_size))
            
            # 3. 统计每个网格中的组件数量
            for component in components:
                position = component.get('position', {})
                if isinstance(position, dict):
                    x = float(position.get('x', 0))
                    y = float(position.get('y', 0))
                else:
                    x, y = map(float, position if isinstance(position, (list, tuple)) else (0, 0))
                    
                # 将坐标映射到网格
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_x, grid_y] += 1
                    
            # 4. 计算拥塞度
            max_congestion = np.max(grid)
            avg_congestion = np.mean(grid)
            
            return max_congestion / (avg_congestion + 1e-6)
            
        except Exception as e:
            logger.error(f"计算拥塞度失败: {str(e)}")
            return 0.0
        
    def _calculate_timing(self, layout: Dict) -> float:
        """计算时序得分
        
        Args:
            layout: 布局数据
            
        Returns:
            时序得分
        """
        print(f"_calculate_timing: layout={layout}")
        try:
            # 1. 获取时序信息
            timing_info = layout.get('timing', {})
            if not timing_info:
                return 0.0
                
            # 2. 计算关键路径延迟
            critical_paths = timing_info.get('critical_paths', [])
            if not critical_paths:
                return 0.0
                
            # 3. 计算平均延迟
            total_delay = 0.0
            for path in critical_paths:
                if isinstance(path, dict):
                    delay = path.get('delay', 0.0)
                else:
                    delay = float(path) if isinstance(path, (int, float)) else 0.0
                total_delay += delay
                
            avg_delay = total_delay / len(critical_paths) if critical_paths else 0.0
            
            # 4. 计算时序得分
            timing_score = 1.0 / (1.0 + avg_delay)
            
            return timing_score
            
        except Exception as e:
            logger.error(f"计算时序得分失败: {str(e)}")
            return 0.0
            
    def _evaluate_constraints(self, data: Dict[str, Any]) -> float:
        """评估约束满足度
        
        Args:
            data: 布局数据
            
        Returns:
            float: 约束满足度得分
        """
        try:
            # 1. 获取约束信息
            constraints = data.get('constraints', [])  # 确保是列表类型
            if not constraints:
                return 1.0  # 如果没有约束，则认为完全满足
                
            # 2. 获取布局信息
            layout = data.get('layout', {})
            components = layout.get('components', [])
            
            # 3. 检查每个约束
            satisfied_constraints = 0
            total_constraints = len(constraints)
            
            for constraint in constraints:
                if isinstance(constraint, dict):
                    # 检查组件位置约束
                    if 'component' in constraint and 'position' in constraint:
                        comp_name = constraint['component']
                        required_pos = constraint['position']
                        
                        # 查找组件
                        comp = next((c for c in components if c.get('name') == comp_name), None)
                        if comp:
                            actual_pos = comp.get('position', {})
                            if isinstance(actual_pos, dict):
                                x = actual_pos.get('x', 0)
                                y = actual_pos.get('y', 0)
                            else:
                                x, y = actual_pos if isinstance(actual_pos, (list, tuple)) else (0, 0)
                                
                            if x == required_pos.get('x', 0) and y == required_pos.get('y', 0):
                                satisfied_constraints += 1
                                
                    # 检查组件大小约束
                    elif 'component' in constraint and 'size' in constraint:
                        comp_name = constraint['component']
                        required_size = constraint['size']
                        
                        # 查找组件
                        comp = next((c for c in components if c.get('name') == comp_name), None)
                        if comp:
                            actual_size = comp.get('size', {})
                            if isinstance(actual_size, dict):
                                width = actual_size.get('width', 0)
                                height = actual_size.get('height', 0)
                            else:
                                width, height = actual_size if isinstance(actual_size, (list, tuple)) else (0, 0)
                                
                            if width == required_size.get('width', 0) and height == required_size.get('height', 0):
                                satisfied_constraints += 1
                                
            # 4. 计算约束满足度
            satisfaction_score = satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
            
            return satisfaction_score
            
        except Exception as e:
            logger.error(f"评估约束满足度失败: {str(e)}")
            return 0.0
            
    def _generate_report(self, wirelength_score: float,
                        congestion_score: float,
                        timing_score: float,
                        overall_score: float,
                        constraint_satisfaction: float) -> str:
        """生成评估报告
        
        Args:
            wirelength_score: 线长得分
            congestion_score: 拥塞度得分
            timing_score: 时序得分
            overall_score: 总体得分
            constraint_satisfaction: 约束满足度
            
        Returns:
            str: 评估报告
        """
        report = []
        report.append("布局评估报告")
        report.append("=" * 50)
        report.append(f"1. 总体得分: {overall_score:.2f}")
        report.append(f"2. 约束满足度: {constraint_satisfaction:.2f}")
        report.append("\n详细指标:")
        report.append(f"- 线长得分: {wirelength_score:.2f}")
        report.append(f"- 拥塞度得分: {congestion_score:.2f}")
        report.append(f"- 时序得分: {timing_score:.2f}")
        report.append("\n建议:")
        
        if overall_score < 0.6:
            report.append("- 布局质量较差，建议重新生成")
        elif overall_score < 0.8:
            report.append("- 布局质量一般，可以考虑优化")
        else:
            report.append("- 布局质量良好")
            
        if constraint_satisfaction < 0.8:
            report.append("- 约束满足度不足，需要调整布局")
            
        return "\n".join(report)

    def __str__(self):
        logger.info(f"self.thresholds: {self.thresholds} ({type(self.thresholds)})")
        logger.info(f"self.weights: {self.weights} ({type(self.weights)})")
        return f"MultiObjectiveEvaluator(config={self.config})" 