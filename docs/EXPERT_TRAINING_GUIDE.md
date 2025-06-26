# 专家指导强化学习训练系统使用指南

## 概述

本系统实现了基于专家演示的强化学习训练，使用 `floorplan.def` 作为初始布局，`mgc_des_perf_1_place.def` 作为专家布局数据进行训练。

## 系统架构

### 核心组件

1. **ExpertDataManager**: 专家数据管理器
   - 解析 `floorplan.def` 和 `mgc_des_perf_1_place.def`
   - 提取专家布局的关键指标和组件位置
   - 计算布局质量和设计复杂度

2. **EnhancedDesignEnvironment**: 增强版设计环境
   - 支持专家相似度计算
   - 集成OpenROAD布局优化
   - 提供增强版奖励函数

3. **ExpertGuidedActorCritic**: 专家引导的神经网络
   - Actor-Critic架构
   - 专家知识网络用于模仿学习
   - 混合策略选择

### 文件结构

```
chipdrag/
├── enhanced_rl_training_with_expert.py    # 完整专家训练系统
├── simple_expert_training_demo.py         # 简化演示版本
├── run_expert_training.py                 # 训练启动脚本
├── configs/
│   └── expert_training_config.json        # 训练配置文件
└── docs/
    └── EXPERT_TRAINING_GUIDE.md           # 本指南
```

## 使用方法

### 1. 快速开始（简化演示）

运行简化演示版本，快速体验专家训练功能：

```bash
cd chipdrag
python run_expert_training.py --mode demo
```

### 2. 完整训练

运行完整的专家指导训练：

```bash
cd chipdrag
python run_expert_training.py --mode full
```

### 3. 自定义配置

使用自定义配置文件：

```bash
python run_expert_training.py --mode demo --config path/to/config.json
```

### 4. 指定设计目录

使用不同的设计目录：

```bash
python run_expert_training.py --mode demo --design-dir path/to/design
```

## 配置说明

### 训练配置 (expert_training_config.json)

```json
{
    "expert_training": {
        "training_config": {
            "learning_rate": 0.0003,      # 学习率
            "num_episodes": 500,          # 训练轮数
            "epsilon_start": 0.9,         # 初始探索率
            "epsilon_end": 0.05,          # 最终探索率
            "epsilon_decay": 0.995,       # 探索率衰减
            "expert_weight": 0.3          # 专家知识权重
        },
        "reward_config": {
            "ppa_weights": {
                "timing": 0.4,            # 时序权重
                "area": 0.3,              # 面积权重
                "power": 0.3              # 功耗权重
            },
            "expert_similarity_weight": 0.3,  # 专家相似度权重
            "improvement_weight": 0.2,        # 改进程度权重
            "k_selection_weight": 0.1         # k值选择权重
        }
    }
}
```

## 核心特性

### 1. 专家相似度计算

系统计算当前布局与专家布局的相似度：

- **面积相似度**: 比较芯片面积
- **密度相似度**: 比较组件密度
- **组件相似度**: 比较组件数量

### 2. 增强版奖励函数

奖励函数包含多个组成部分：

```python
total_reward = (
    base_ppa_reward * 0.4 +           # 基础PPA奖励
    expert_similarity * 0.3 +         # 专家相似度奖励
    improvement * 0.2 +               # 改进程度奖励
    k_selection_reward * 0.1          # k值选择奖励
)
```

### 3. 混合策略选择

动作选择结合网络输出和专家知识：

```python
mixed_probs = (1 - expert_weight) * action_probs + expert_weight * expert_probs
```

### 4. 专家模仿学习

使用KL散度损失进行专家模仿：

```python
imitation_loss = F.kl_div(F.log_softmax(expert_probs, dim=-1), expert_target)
```

## 训练流程

### 1. 数据准备

确保设计目录包含必要文件：
- `floorplan.def`: 初始布局文件
- `mgc_des_perf_1_place.def`: 专家布局文件
- `design.v`: Verilog网表文件
- `cells.lef`, `tech.lef`: LEF文件

### 2. 环境初始化

```python
# 初始化专家数据管理器
expert_data = ExpertDataManager(design_dir)

# 初始化增强版环境
env = EnhancedDesignEnvironment(design_dir, expert_data)
```

### 3. 网络训练

```python
# 初始化网络
network = ExpertGuidedActorCritic(state_dim, action_dim)

# 训练循环
for episode in range(num_episodes):
    state = env.get_state()
    
    # 选择动作（混合策略）
    action_probs, _, expert_probs = network(state)
    mixed_probs = (1 - expert_weight) * action_probs + expert_weight * expert_probs
    
    # 执行动作
    next_state, reward, done, info = env.step(k_value)
    
    # 计算损失并更新
    # ...
```

### 4. 结果保存

训练完成后，系统会保存：
- 训练好的模型 (`*_expert_guided_model.pt`)
- 训练报告 (`*_expert_guided_report.json`)
- 训练日志

## 输出分析

### 训练报告内容

```json
{
    "episode": 0,
    "reward": 0.75,
    "k_value": 6,
    "epsilon": 0.9,
    "actor_loss": 0.12,
    "critic_loss": 0.08,
    "imitation_loss": 0.05,
    "expert_similarity": 0.85,
    "improvement": 0.2
}
```

### 关键指标

- **平均奖励**: 训练效果的主要指标
- **专家相似度**: 与专家布局的接近程度
- **改进程度**: 相对于基准的改进
- **探索率**: 当前探索与利用的平衡

## 高级用法

### 1. 自定义奖励函数

修改 `_calculate_enhanced_reward` 方法：

```python
def _calculate_enhanced_reward(self, result, k_value):
    # 自定义奖励计算逻辑
    base_reward = self._calculate_ppa_reward(result)
    expert_reward = self._calculate_expert_reward()
    custom_reward = self._calculate_custom_reward()
    
    return base_reward + expert_reward + custom_reward
```

### 2. 添加新的相似度指标

在 `ExpertDataManager` 中添加新的相似度计算：

```python
def _calculate_wirelength_similarity(self, current, expert):
    # 计算线长相似度
    pass
```

### 3. 扩展状态空间

修改 `_build_state_vector` 方法添加新的状态特征：

```python
features = [
    # 现有特征
    current_die_area[0] / 1000.0,
    current_die_area[1] / 1000.0,
    # 新增特征
    wirelength_ratio,
    congestion_score,
    # ...
]
```

## 故障排除

### 常见问题

1. **文件不存在错误**
   - 检查设计目录路径是否正确
   - 确保所有必要文件都存在

2. **OpenROAD接口错误**
   - 检查Docker环境是否正确配置
   - 验证OpenROAD工具是否可用

3. **训练不收敛**
   - 调整学习率和探索率
   - 检查奖励函数设计
   - 增加训练轮数

4. **内存不足**
   - 减少批次大小
   - 降低网络复杂度
   - 使用梯度累积

## 性能优化

### 1. 训练加速

- 使用GPU训练（如果可用）
- 减少OpenROAD调用频率
- 使用缓存机制

### 2. 内存优化

- 使用梯度检查点
- 减少状态维度
- 优化数据加载

### 3. 收敛优化

- 调整奖励权重
- 使用学习率调度
- 增加正则化

## 扩展开发

### 1. 添加新的专家数据源

```python
class CustomExpertDataManager(ExpertDataManager):
    def _parse_expert_data(self):
        # 自定义专家数据解析逻辑
        pass
```

### 2. 实现新的网络架构

```python
class CustomExpertNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        # 自定义网络架构
        pass
```

### 3. 添加新的评估指标

```python
def evaluate_expert_training(model, env, expert_data):
    # 自定义评估逻辑
    pass
```

## 总结

专家指导强化学习训练系统提供了一个完整的框架，用于学习工业级EDA工具的布局策略。通过结合专家演示和强化学习，系统能够：

1. 学习高质量的布局策略
2. 提高与专家布局的相似度
3. 在PPA指标上取得改进
4. 适应不同的设计复杂度

该系统为芯片布局优化提供了一个新的研究方向，结合了传统EDA工具的专业知识和现代机器学习的学习能力。 