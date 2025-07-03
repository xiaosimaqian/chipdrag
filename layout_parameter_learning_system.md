# ChipDRAG布局参数学习和检索系统设计

## 1. 系统概述

布局参数学习和检索系统是ChipDRAG的核心组成部分，负责根据设计特征动态调整OpenROAD布局参数，以实现最优的HPWL和布局质量。

## 2. 关键参数分类

### 2.1 Floorplan初始化参数 (`initialize_floorplan`)

| 参数名 | 类型 | 范围 | 影响 | 学习重要性 |
|--------|------|------|------|------------|
| `utilization` | float | [0.3, 0.9] | 芯片面积利用率，影响拥塞度 | ⭐⭐⭐⭐⭐ |
| `aspect_ratio` | float | [0.5, 2.0] | 芯片长宽比，影响HPWL | ⭐⭐⭐⭐ |
| `core_space` | int | [5, 50] | 核心边界空间，影响I/O布局 | ⭐⭐⭐ |
| `site` | string | LEF定义 | 标准单元站点，影响行创建 | ⭐⭐⭐⭐⭐ |

### 2.2 全局布局参数 (`global_placement`)

| 参数名 | 类型 | 范围 | 影响 | 学习重要性 |
|--------|------|------|------|------------|
| `density` | float | [0.4, 0.9] | 目标布局密度，影响可路由性 | ⭐⭐⭐⭐⭐ |
| `overflow` | float | [0.05, 0.3] | 溢出容忍度，影响收敛速度 | ⭐⭐⭐⭐ |
| `bin_grid_count` | int | [64, 512] | 布局网格数，影响精度和速度 | ⭐⭐⭐ |
| `init_density_penalty` | float | [1e-6, 1e-3] | 初始密度惩罚，影响收敛 | ⭐⭐⭐ |
| `wirelength_coef` | float | [0.1, 1.0] | 线长权重，影响HPWL优化 | ⭐⭐⭐⭐ |

### 2.3 详细布局参数 (`detailed_placement`)

| 参数名 | 类型 | 范围 | 影响 | 学习重要性 |
|--------|------|------|------|------------|
| `max_displacement` | int | [50, 500] | 最大移动距离，影响优化幅度 | ⭐⭐⭐ |
| `disallow_one_site_gaps` | bool | [true, false] | 禁止单站点间隙 | ⭐⭐ |

## 3. 参数学习策略

### 3.1 设计特征提取

```python
class DesignFeatureExtractor:
    def extract_features(self, def_file, lef_file):
        return {
            'component_count': int,      # 组件数量
            'net_count': int,           # 网络数量
            'pin_count': int,           # 引脚数量
            'design_area': float,       # 设计面积
            'aspect_ratio': float,      # 当前长宽比
            'io_pin_count': int,        # I/O引脚数量
            'critical_nets': int,       # 关键网络数量
            'macro_count': int,         # 宏单元数量
            'memory_blocks': int,       # 存储块数量
            'clock_domains': int        # 时钟域数量
        }
```

### 3.2 参数映射学习

```python
class ParameterLearner:
    def __init__(self):
        self.parameter_history = []
        self.hpwl_results = []
        
    def learn_parameters(self, design_features, hpwl_result):
        """基于设计特征和HPWL结果学习最优参数"""
        
        # 1. 利用率学习
        if design_features['component_count'] > 10000:
            utilization = 0.6  # 大设计使用较低利用率
        elif design_features['component_count'] < 1000:
            utilization = 0.8  # 小设计可以使用较高利用率
        else:
            utilization = 0.7  # 中等设计使用中等利用率
            
        # 2. 长宽比学习
        if design_features['io_pin_count'] > 500:
            aspect_ratio = 1.2  # 多I/O设计倾向于矩形
        else:
            aspect_ratio = 1.0  # 少I/O设计倾向于正方形
            
        # 3. 密度学习
        density = min(utilization, 0.85)  # 密度不超过利用率
        
        # 4. 溢出阈值学习
        if design_features['net_count'] > 5000:
            overflow = 0.15  # 复杂网络使用较高溢出阈值
        else:
            overflow = 0.1   # 简单网络使用较低溢出阈值
            
        return {
            'utilization': utilization,
            'aspect_ratio': aspect_ratio,
            'density': density,
            'overflow': overflow,
            'core_space': 10 + design_features['io_pin_count'] // 50
        }
```

### 3.3 动态检索策略

```python
class ParameterRetriever:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        
    def retrieve_similar_cases(self, design_features, k=5):
        """检索相似设计案例的参数配置"""
        
        similar_cases = []
        for case in self.kb:
            similarity = self.calculate_similarity(design_features, case['features'])
            if similarity > 0.7:  # 相似度阈值
                similar_cases.append((case, similarity))
                
        # 按相似度排序，返回top-k
        similar_cases.sort(key=lambda x: x[1], reverse=True)
        return similar_cases[:k]
        
    def calculate_similarity(self, features1, features2):
        """计算设计特征相似度"""
        
        # 归一化特征
        norm_features1 = self.normalize_features(features1)
        norm_features2 = self.normalize_features(features2)
        
        # 计算加权欧氏距离
        weights = {
            'component_count': 0.3,
            'net_count': 0.25,
            'design_area': 0.2,
            'io_pin_count': 0.15,
            'aspect_ratio': 0.1
        }
        
        distance = 0
        for key, weight in weights.items():
            if key in norm_features1 and key in norm_features2:
                distance += weight * (norm_features1[key] - norm_features2[key])**2
                
        return 1.0 / (1.0 + distance)  # 转换为相似度
```

## 4. RL代理参数优化

### 4.1 状态空间设计

```python
class ParameterState:
    def __init__(self, design_features, current_params, hpwl_history):
        self.design_features = design_features
        self.current_params = current_params
        self.hpwl_history = hpwl_history
        self.exploration_count = 0
```

### 4.2 动作空间设计

```python
class ParameterAction:
    def __init__(self):
        self.param_adjustments = {
            'utilization': [-0.1, -0.05, 0, 0.05, 0.1],
            'aspect_ratio': [-0.2, -0.1, 0, 0.1, 0.2],
            'density': [-0.1, -0.05, 0, 0.05, 0.1],
            'overflow': [-0.05, -0.02, 0, 0.02, 0.05]
        }
```

### 4.3 奖励函数设计

```python
def calculate_parameter_reward(old_hpwl, new_hpwl, layout_success):
    """计算参数调整的奖励"""
    
    if not layout_success:
        return -1.0  # 布局失败严重惩罚
        
    hpwl_improvement = (old_hpwl - new_hpwl) / old_hpwl
    
    if hpwl_improvement > 0.1:
        return 1.0  # 显著改善
    elif hpwl_improvement > 0.05:
        return 0.5  # 中等改善
    elif hpwl_improvement > 0:
        return 0.2  # 轻微改善
    elif hpwl_improvement > -0.05:
        return -0.1  # 轻微恶化
    else:
        return -0.5  # 显著恶化
```

## 5. 实现集成

### 5.1 在现有系统中集成

```python
class EnhancedLayoutStrategy:
    def __init__(self):
        self.feature_extractor = DesignFeatureExtractor()
        self.parameter_learner = ParameterLearner()
        self.parameter_retriever = ParameterRetriever(knowledge_base)
        
    def generate_layout_strategy(self, design_dir):
        # 1. 提取设计特征
        features = self.feature_extractor.extract_features(
            f"{design_dir}/floorplan.def",
            f"{design_dir}/cells.lef"
        )
        
        # 2. 检索相似案例
        similar_cases = self.parameter_retriever.retrieve_similar_cases(features)
        
        # 3. 学习最优参数
        learned_params = self.parameter_learner.learn_parameters(features, None)
        
        # 4. 融合检索和学习结果
        final_params = self.fuse_parameters(learned_params, similar_cases)
        
        return {
            'parameters': final_params,
            'features': features,
            'similar_cases': similar_cases
        }
```

## 6. 实验验证

### 6.1 参数敏感性分析

- 测试每个参数对HPWL的影响程度
- 识别关键参数和参数组合
- 建立参数-性能映射模型

### 6.2 学习效果评估

- 比较随机参数 vs 学习参数的HPWL结果
- 评估检索系统的准确性
- 测试RL代理的参数优化能力

## 7. 预期效果

1. **HPWL改善**: 相比默认参数，预期改善5-15%
2. **布局成功率**: 提高布局收敛成功率到95%以上
3. **参数自适应**: 根据设计特征自动调整参数
4. **知识积累**: 建立参数-性能知识库，持续改进

这个系统将使ChipDRAG能够智能地调整布局参数，实现真正的自适应布局优化。 