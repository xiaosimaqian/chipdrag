# ChipDRAG统一实验系统使用指南

## 概述

原本分散的三个工具已完全整合到 `experiment.py` 中，提供统一的实验管理界面：

- ✅ **训练案例提取** (`extract_training_cases.py` → `experiment.py`)
- ✅ **案例相似度改进** (`improve_case_similarity.py` → `experiment.py`)  
- ✅ **性能监控** (`performance_monitor.py` → `experiment.py`)

## 使用方式

### 1. 完整HPWL对比实验（推荐）

```bash
# 本地模式（使用Docker）
python experiment.py --mode local --experiment-type hpwl

# 服务器模式（直接使用OpenROAD） 
python experiment.py --mode server --experiment-type hpwl

# 启用性能监控
python experiment.py --mode server --experiment-type hpwl --enable-monitoring --monitor-interval 3
```

**说明**：完整实验会自动执行以下步骤：
1. 自动提取训练案例
2. 自动改进案例相似度
3. 执行RL训练
4. 进行ChipDRAG优化
5. 收集HPWL对比数据
6. 生成完整报告

### 2. 消融实验

```bash
# 验证三大核心技术贡献
python experiment.py --mode server --experiment-type ablation

# 启用性能监控
python experiment.py --mode server --experiment-type ablation --enable-monitoring
```

### 3. 单独提取训练案例

```bash
# 仅提取训练案例
python experiment.py --experiment-type extract-cases

# 启用性能监控
python experiment.py --experiment-type extract-cases --enable-monitoring
```

### 4. 单独改进案例相似度

```bash
# 仅改进案例相似度
python experiment.py --experiment-type improve-similarity

# 启用性能监控
python experiment.py --experiment-type improve-similarity --enable-monitoring
```

## 核心功能说明

### 🔧 训练案例提取功能

**原功能**：`extract_training_cases.py`
**新位置**：`experiment.extract_training_cases()`

**特点**：
- ✅ 完全基于真实DEF/LEF文件数据
- ✅ 智能文件匹配算法
- ✅ 预加载真实设计特征缓存
- ✅ 找不到真实数据时跳过案例（不生成虚假数据）

**输出**：
- `data/knowledge_base/cases.pkl` - 案例数据（pickle格式）
- `data/knowledge_base/cases.json` - 案例数据（JSON格式）

### 🎯 案例相似度改进功能

**原功能**：`improve_case_similarity.py`
**新位置**：`experiment.improve_case_similarity()`

**特点**：
- ✅ 基于真实DEF文件特征提取
- ✅ 改进的相似度计算算法
- ✅ 标准化特征向量生成
- ✅ 相似度矩阵计算

**输出**：
- `data/knowledge_base/improved_cases.pkl` - 改进的案例数据
- `data/knowledge_base/improved_cases.json` - 改进的案例数据
- `data/knowledge_base/similarity_report.json` - 相似度分析报告

### 📊 性能监控功能

**原功能**：`performance_monitor.py`
**新位置**：`experiment.start_performance_monitoring()`

**特点**：
- ✅ 实时监控CPU、内存使用率
- ✅ 统计OpenROAD进程数量
- ✅ 后台线程监控，不影响实验执行
- ✅ 生成详细性能报告

**使用方法**：
```bash
# 启用监控（5秒间隔）
python experiment.py --enable-monitoring

# 自定义监控间隔（3秒）
python experiment.py --enable-monitoring --monitor-interval 3
```

## 实验结果文件

### 主要输出目录

```
paper_hpwl_results/
├── unified_experiment_YYYYMMDD_HHMMSS/
│   ├── complete_results.json           # 完整实验结果
│   ├── hpwl_comparison_results.json    # HPWL对比结果
│   └── performance_report.json         # 性能监控报告
├── ablation_experiment_YYYYMMDD_HHMMSS/
│   ├── ablation_analysis.json          # 消融实验分析
│   └── ablation_report.md              # 消融实验报告
└── experiment_log_YYYYMMDD_HHMMSS.log  # 实验日志

data/knowledge_base/
├── cases.pkl                           # 训练案例数据
├── cases.json                          # 训练案例数据（JSON格式）
├── improved_cases.pkl                  # 改进的案例数据
├── improved_cases.json                 # 改进的案例数据（JSON格式）
└── similarity_report.json              # 相似度分析报告
```

## 性能优化配置

### 本地模式性能配置

```python
# Docker资源限制
max_parallel_designs = 1        # 单任务模式确保内存充足
max_parallel_containers = 1     # 单容器模式
memory_limit_gb = 8             # 最大8GB内存
cpu_limit = 8                   # 最大8核CPU
```

### 服务器模式性能配置

服务器模式会自动检测硬件配置并优化：

```python
# 超级服务器配置（160+核，900+GB内存）
max_parallel_designs = 16
max_parallel_containers = 32
openroad_threads = 16
rl_training_threads = 20

# 高性能服务器配置（32+核，100+GB内存）
max_parallel_designs = 8
max_parallel_containers = 16
openroad_threads = 8
rl_training_threads = 12

# 标准服务器配置
max_parallel_designs = 4
max_parallel_containers = 8
openroad_threads = 4
rl_training_threads = 8
```

## 常见问题解答

### Q1: 整合后如何单独运行某个功能？

A1: 使用 `--experiment-type` 参数：
```bash
python experiment.py --experiment-type extract-cases      # 仅提取案例
python experiment.py --experiment-type improve-similarity # 仅改进相似度
python experiment.py --experiment-type hpwl              # 完整HPWL实验
python experiment.py --experiment-type ablation          # 消融实验
```

### Q2: 性能监控数据在哪里？

A2: 性能监控数据保存在：
- 实时输出到控制台
- 详细数据保存到 `performance_report.json`
- 实验结束时显示性能摘要

### Q3: 如何查看案例提取结果？

A3: 案例提取结果保存在 `data/knowledge_base/` 目录：
- `cases.json` - 人类可读的JSON格式
- `cases.pkl` - 程序使用的pickle格式

### Q4: 相似度改进效果如何验证？

A4: 查看 `similarity_report.json` 文件：
```json
{
  "high_similarity_pairs": [...],    // 高相似度案例对 (>0.7)
  "low_similarity_cases": [...]      // 低相似度案例 (<0.3)
}
```

### Q5: 如何在实验中途停止？

A5: 使用 `Ctrl+C` 安全停止：
- 性能监控会自动停止并生成报告
- 已完成的实验结果会保存
- 正在执行的OpenROAD任务会等待完成

## 优势总结

### 🎯 统一管理
- 所有功能集中在一个脚本中
- 统一的命令行接口
- 统一的日志和错误处理

### 🚀 自动化程度高
- 完整实验自动执行所有步骤
- 自动提取案例和改进相似度
- 自动生成完整报告

### 📊 全面监控
- 实时性能监控
- 详细的实验日志
- 完整的结果追踪

### 🔧 高度可配置
- 支持本地和服务器两种模式
- 灵活的并行度配置
- 可选的性能监控

### 💾 结果完整保存
- 所有实验数据自动保存
- 人类可读的报告格式
- 便于后续分析和复现

## 使用建议

1. **首次使用**：建议先运行 `extract-cases` 和 `improve-similarity` 确保数据质量
2. **正式实验**：使用 `hpwl` 类型进行完整实验
3. **验证研究**：使用 `ablation` 类型验证技术贡献
4. **性能分析**：启用 `--enable-monitoring` 监控系统资源使用
5. **服务器环境**：优先使用 `server` 模式以获得更好的性能

通过这种整合，您现在可以通过一个统一的接口管理所有实验功能，大大简化了实验流程和结果管理。 