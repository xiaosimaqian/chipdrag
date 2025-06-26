# ChipRAG强化学习训练系统

## 概述

ChipRAG强化学习训练系统是一个用于训练布局优化智能体的完整框架。该系统通过强化学习算法学习最优的布局策略，能够自动优化芯片布局的HPWL（半周长线长）、溢出率、密度和利用率等关键指标。

## 系统架构

### 核心组件

1. **LayoutEnvironment（布局环境）**
   - 封装OpenROAD布局工具
   - 提供状态观察和动作执行接口
   - 计算奖励函数
   - 支持真实OpenROAD和模拟模式

2. **DQNAgent（DQN智能体）**
   - 基于PyTorch的深度Q网络
   - 支持经验回放和目标网络
   - 8种预定义布局策略
   - 可配置的探索策略

3. **RLTrainer（训练器）**
   - 管理训练流程
   - 记录训练历史
   - 生成训练报告和可视化

4. **RLAgentEvaluator（评估器）**
   - 评估训练好的智能体
   - 生成详细的性能报告
   - 可视化评估结果

## 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install pathlib

# 可选：GPU支持
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 1. 快速训练（模拟模式）

```bash
# 使用快速配置进行训练
python train_rl_agent.py --mode fast

# 或者使用示例脚本
python examples/rl_training_example.py
```

### 2. 真实训练（使用OpenROAD）

```bash
# 使用真实OpenROAD进行训练
python train_rl_agent.py --mode default --use_openroad

# 指定工作目录
python train_rl_agent.py --work_dir data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1 --use_openroad
```

### 3. 评估训练好的智能体

```bash
# 评估智能体性能
python evaluate_rl_agent.py --model experiments/xxx/final_model.pth --config experiments/xxx/config.json --episodes 10
```

## 配置说明

### 环境配置

```python
from rl_training_config import RLTrainingConfig

config = RLTrainingConfig(
    work_dir="data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1",
    max_iterations=10,           # 每次episode的最大迭代次数
    target_hpwl=1000000.0,       # 目标HPWL
    target_overflow=0.1,         # 目标溢出率
    use_openroad=True            # 是否使用真实OpenROAD
)
```

### 智能体配置

```python
config = RLTrainingConfig(
    state_size=5,                # 状态空间大小
    action_size=8,               # 动作空间大小（8种策略）
    learning_rate=0.001,         # 学习率
    epsilon=1.0,                 # 初始探索率
    epsilon_decay=0.995,         # 探索率衰减
    epsilon_min=0.01,            # 最小探索率
    memory_size=10000,           # 经验回放缓冲区大小
    batch_size=32,               # 训练批次大小
    gamma=0.95,                  # 折扣因子
    target_update=10             # 目标网络更新频率
)
```

### 训练配置

```python
config = RLTrainingConfig(
    episodes=100,                # 训练episodes数量
    max_steps=10,                # 每个episode最大步数
    save_interval=10,            # 模型保存间隔
    reward_weights={             # 奖励函数权重
        "hpwl_improvement": 100.0,
        "overflow_penalty": -50.0,
        "density_reward": -10.0,
        "utilization_reward": 20.0,
        "convergence_reward": 200.0
    }
)
```

## 布局策略

系统预定义了8种布局策略：

1. **保守策略** (0.7, 1.0, 1.0) - 低密度，平衡权重
2. **平衡策略** (0.8, 2.0, 1.5) - 中等密度，平衡权重
3. **激进策略** (0.9, 3.0, 2.0) - 高密度，高权重
4. **线长优先** (0.75, 4.0, 0.5) - 优先优化线长
5. **密度优先** (0.95, 0.5, 4.0) - 优先优化密度
6. **中等策略** (0.85, 1.5, 1.5) - 中等参数
7. **线长中等** (0.8, 2.5, 1.0) - 中等线长权重
8. **密度中等** (0.9, 1.0, 3.0) - 中等密度权重

## 奖励函数

奖励函数综合考虑多个指标：

```python
reward = (hpwl_improvement * hpwl_weight + 
          overflow_penalty * overflow_weight + 
          density_reward * density_weight + 
          utilization_reward * utilization_weight + 
          convergence_reward * convergence_weight)
```

- **HPWL改善奖励**：鼓励线长优化
- **溢出率惩罚**：避免布局溢出
- **密度奖励**：鼓励接近目标密度
- **利用率奖励**：鼓励高利用率
- **收敛奖励**：达到目标时的额外奖励

## 训练模式

### 1. 快速模式（fast）
- 使用模拟数据
- 少量episodes（10个）
- 适合开发和测试

### 2. 默认模式（default）
- 使用真实OpenROAD
- 中等episodes数量（100个）
- 平衡训练时间和效果

### 3. 完整模式（full）
- 使用真实OpenROAD
- 大量episodes（500个）
- 最佳训练效果

### 4. 多benchmark模式（all）
- 同时训练多个benchmark
- 生成对比结果

## 输出文件

### 训练输出
- `experiments/xxx/config.json` - 训练配置
- `experiments/xxx/training_history.csv` - 训练历史
- `experiments/xxx/training_curves.png` - 训练曲线
- `experiments/xxx/final_model.pth` - 最终模型
- `experiments/xxx/episode_*.json` - 每个episode的详细数据

### 评估输出
- `evaluation_results/xxx/evaluation_results.json` - 详细评估结果
- `evaluation_results/xxx/evaluation_stats.json` - 统计指标
- `evaluation_results/xxx/evaluation_results.png` - 评估图表

## 使用示例

### 基本训练

```python
from rl_training_system import LayoutEnvironment, DQNAgent, RLTrainer
from rl_training_config import get_default_config

# 获取配置
config = get_default_config()

# 创建环境
env = LayoutEnvironment(
    work_dir=config.work_dir,
    max_iterations=config.max_iterations,
    target_hpwl=config.target_hpwl,
    target_overflow=config.target_overflow,
    use_openroad=config.use_openroad
)

# 创建智能体
agent = DQNAgent(
    state_size=config.state_size,
    action_size=config.action_size,
    learning_rate=config.learning_rate,
    epsilon=config.epsilon,
    epsilon_decay=config.epsilon_decay,
    epsilon_min=config.epsilon_min,
    memory_size=config.memory_size,
    batch_size=config.batch_size,
    gamma=config.gamma,
    target_update=config.target_update
)

# 创建训练器
trainer = RLTrainer(
    env=env,
    agent=agent,
    episodes=config.episodes,
    max_steps=config.max_steps
)

# 开始训练
trainer.train()
```

### 自定义配置

```python
from rl_training_config import RLTrainingConfig

# 创建自定义配置
config = RLTrainingConfig(
    work_dir="your_design_directory",
    max_iterations=15,
    target_hpwl=800000.0,
    target_overflow=0.08,
    use_openroad=True,
    episodes=200,
    learning_rate=0.0005,
    reward_weights={
        "hpwl_improvement": 150.0,
        "overflow_penalty": -75.0,
        "density_reward": -15.0,
        "utilization_reward": 30.0,
        "convergence_reward": 300.0
    }
)

# 保存配置
config.save("my_config.json")
```

## 故障排除

### 1. OpenROAD环境问题
```bash
# 检查OpenROAD是否可用
which openroad

# 如果使用Docker
docker run -it --rm -v $(pwd):/workspace openroad/flow:latest
```

### 2. 内存不足
- 减少`memory_size`和`batch_size`
- 使用模拟模式进行测试

### 3. 训练不收敛
- 调整学习率和探索率
- 修改奖励函数权重
- 增加训练episodes数量

### 4. GPU内存不足
- 减少批次大小
- 使用CPU训练：`torch.device("cpu")`

## 性能优化

### 1. 使用GPU训练
```python
# 自动检测GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. 并行训练
- 使用多进程进行数据收集
- 并行评估多个episodes

### 3. 模型保存和加载
```python
# 保存模型
agent.save("model.pth")

# 加载模型
agent.load("model.pth")
```

## 扩展功能

### 1. 自定义奖励函数
```python
def custom_reward_function(prev_state, curr_state, action):
    # 实现自定义奖励逻辑
    return reward
```

### 2. 新的布局策略
```python
# 在DQNAgent中添加新的动作
action_space = [
    # 现有策略...
    [0.88, 1.8, 1.8],  # 新策略
]
```

### 3. 多目标优化
```python
# 支持多个优化目标
objectives = ["hpwl", "overflow", "density", "utilization"]
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。 