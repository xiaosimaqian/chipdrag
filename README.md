# Chip-D-RAG: Dynamic Retrieval-Augmented Generation for Chip Layout Design

## 项目概述

Chip-D-RAG是一个基于动态检索增强生成（Dynamic RAG）技术的芯片布局设计系统。该系统结合了强化学习、实体增强、多模态融合和质量反馈等先进技术，旨在提升芯片布局生成的效率和质量。

## 核心特性

### 🚀 动态检索策略
- **强化学习智能体**: 基于Q-Learning的动态k值选择
- **质量反馈驱动**: 根据布局质量动态调整检索策略
- **历史经验学习**: 利用历史交互记录优化决策

### 🔧 实体增强技术
- **实体压缩**: 高效压缩实体嵌入信息
- **注意力注入**: 通过注意力机制注入实体信息
- **相似性计算**: 基于实体相似性的检索优化

### 🎯 多模态知识融合
- **跨模态检索**: 支持文本、图像、结构化数据
- **融合层设计**: 多模态信息的有效融合
- **知识图谱**: 结构化知识的表示和利用

### 📊 质量反馈机制
- **多目标评估**: 布局质量、约束满足度、性能指标
- **反馈循环**: 持续的质量改进机制
- **奖励设计**: 基于质量反馈的奖励计算

## 系统架构

```
Chip-D-RAG System
├── 强化学习智能体 (RL Agent)
│   ├── Q-Learning算法
│   ├── 状态提取器
│   └── 奖励计算器
├── 动态检索器 (Dynamic Retriever)
│   ├── 动态重排序
│   ├── 实体增强
│   └── 质量反馈
├── 布局生成器 (Layout Generator)
│   ├── 多模态融合
│   └── 约束满足
├── 评估器 (Evaluator)
│   ├── 多目标评估
│   └── 质量反馈
└── 实验框架 (Experiment Framework)
    ├── 对比实验
    ├── 消融实验
    └── 案例分析
```

## 安装指南

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/xiaosimaqian/chipdrag.git
```

2. **创建虚拟环境**
```bash
python -m venv chiprag_env
source chiprag_env/bin/activate  # Linux/Mac
# 或
chiprag_env\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **安装项目**
```bash
pip install -e .
```

## 快速开始

### 1. 运行演示

```bash
python examples/dynamic_rag_demo.py
```

这将运行一个完整的系统演示，包括：
- 强化学习智能体训练
- 动态检索演示
- 实验功能展示
- 案例分析

### 2. 运行完整实验

```bash
# 运行所有实验
python run_experiments.py --experiment all

# 运行特定实验
python run_experiments.py --experiment comparison
python run_experiments.py --experiment ablation
python run_experiments.py --experiment case_study
```

### 3. 自定义配置

编辑配置文件 `configs/dynamic_rag_config.json`：

```json
{
  "dynamic_rag": {
    "enabled": true,
    "retriever": {
      "dynamic_k_range": [3, 15],
      "quality_threshold": 0.7,
      "learning_rate": 0.01
    },
    "reinforcement_learning": {
      "epsilon": 0.1,
      "alpha": 0.01,
      "gamma": 0.9
    }
  }
}
```

## 使用示例

### 基本使用

```python
from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever
from modules.core.rl_agent import QLearningAgent

# 初始化系统
config = {
    'dynamic_k_range': [3, 15],
    'quality_threshold': 0.7
}

retriever = DynamicRAGRetriever(config)
agent = QLearningAgent(config)

# 处理查询
query = {
    'text': 'Generate layout for RISC-V processor',
    'design_type': 'risc_v',
    'constraints': ['timing', 'power']
}

design_info = {
    'design_type': 'risc_v',
    'technology_node': '14nm',
    'constraints': [...]
}

# 执行检索
results = retriever.retrieve_with_dynamic_reranking(query, design_info)
```

### 强化学习训练

```python
from modules.core.rl_trainer import RLTrainer

# 准备训练数据
training_data = [
    {
        'query': {...},
        'design_info': {...},
        'expected_quality': 0.8
    }
]

# 初始化训练器
trainer = RLTrainer(config)

# 开始训练
trainer.train(training_data)
```

### 实验评估

```python
from experiments.dynamic_rag_experiment import DynamicRAGExperiment

# 初始化实验设计器
experiment = DynamicRAGExperiment(config)

# 运行对比实验
results = experiment.run_comparison_experiment(test_data)

# 运行消融实验
ablation_results = experiment.run_ablation_study(test_data)

# 生成报告
report_path = experiment.generate_experiment_report()
```

## 实验设计

### 对比实验
- **基线方法**: TraditionalRAG, ChipRAG
- **评估指标**: 布局质量、约束满足度、性能指标
- **统计检验**: t检验、显著性分析

### 消融实验
- **完整系统**: 包含所有组件
- **移除组件**: 动态重排序、实体增强、多模态融合、质量反馈
- **贡献分析**: 各组件对系统性能的贡献

### 案例分析
- **RISC-V处理器**: 复杂处理器布局
- **DSP加速器**: 高性能计算单元
- **内存控制器**: 存储接口设计

## 性能指标

### 布局质量
- **布线长度**: 总布线长度优化
- **拥塞度**: 布线拥塞情况
- **时序性能**: 关键路径延迟
- **功耗效率**: 动态功耗优化

### 约束满足度
- **时序约束**: 时钟频率要求
- **功耗约束**: 功耗预算限制
- **面积约束**: 芯片面积限制
- **布线约束**: 布线密度要求

### 系统效率
- **响应时间**: 查询处理时间
- **收敛速度**: 训练收敛时间
- **资源使用**: 内存和计算资源

## 配置说明

### 强化学习配置
```json
{
  "reinforcement_learning": {
    "agent_type": "q_learning",
    "epsilon": 0.1,
    "alpha": 0.01,
    "gamma": 0.9,
    "max_states": 10000,
    "update_frequency": 10
  }
}
```

### 检索器配置
```json
{
  "retriever": {
    "dynamic_k_range": [3, 15],
    "quality_threshold": 0.7,
    "learning_rate": 0.01,
    "compressed_entity_dim": 128
  }
}
```

### 评估器配置
```json
{
  "evaluation": {
    "weights": {
      "wirelength": 0.25,
      "congestion": 0.25,
      "timing": 0.3,
      "power": 0.2
    }
  }
}
```

## 文件结构

```
chipdrag/
├── configs/                 # 配置文件
│   ├── dynamic_rag_config.json
│   └── experiment_config.json
├── modules/                 # 核心模块
│   ├── core/               # 核心组件
│   │   ├── rl_agent.py     # 强化学习智能体
│   │   ├── rl_trainer.py   # 训练器
│   │   └── layout_generator.py
│   ├── retrieval/          # 检索模块
│   │   └── dynamic_rag_retriever.py
│   ├── evaluation/         # 评估模块
│   └── utils/              # 工具模块
├── experiments/            # 实验模块
│   └── dynamic_rag_experiment.py
├── examples/               # 示例代码
│   └── dynamic_rag_demo.py
├── logs/                   # 日志文件
├── reports/                # 实验报告
├── checkpoints/            # 检查点文件
├── models/                 # 模型文件
├── run_experiments.py      # 主实验脚本
└── README.md              # 项目说明
```

## 贡献指南

### 开发环境设置
1. Fork项目仓库
2. 创建功能分支
3. 提交代码更改
4. 创建Pull Request

### 代码规范
- 遵循PEP 8代码风格
- 添加适当的文档字符串
- 编写单元测试
- 更新相关文档

### 测试
```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
python -m pytest tests/integration/

# 运行性能测试
python -m pytest tests/performance/
```

## 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 引用

如果您在研究中使用了Chip-D-RAG，请引用我们的论文：

```bibtex
@article{chipdrag2024,
  title={Chip-D-RAG: Dynamic Retrieval-Augmented Generation for Chip Layout Design},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2024}
}
```

## 联系方式

- 项目主页: https://github.com/xiaosimaqian/chipdrag
- 问题反馈: https://github.com/xiaosimaqian/chipdrag/issues
- 邮箱: sunkeqin11@mails.ucas.edu.cn

## 更新日志

### v0.0.1 (2025-06-20)
- 初始版本发布
- 实现核心动态RAG功能
- 完成强化学习智能体
- 建立实验框架


---

**注意**: 这是一个研究项目，主要用于学术研究目的。在生产环境中使用前，请进行充分的测试和验证。 