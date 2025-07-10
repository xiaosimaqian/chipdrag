# 统一版论文实验使用说明

## 简介

`experiment.py` 是统一版论文实验脚本，支持两种实验类型和两种执行模式：

### 实验类型
- **HPWL对比实验（hpwl）**：ChipDRAG vs OpenROAD默认布局的HPWL对比
- **消融实验（ablation）**：验证三大核心技术贡献的有效性

### 执行模式
- **本地模式（local）**：使用Docker容器执行OpenROAD
- **服务器模式（server）**：直接使用系统安装的OpenROAD

## 使用方法

### 基本命令

```bash
# HPWL对比实验（默认）
python experiment.py

# 显式指定HPWL对比实验，本地模式
python experiment.py --mode local --experiment-type hpwl

# HPWL对比实验，服务器模式
python experiment.py --mode server --experiment-type hpwl

# 消融实验，本地模式
python experiment.py --mode local --experiment-type ablation

# 消融实验，服务器模式
python experiment.py --mode server --experiment-type ablation
```

### 参数说明

- `--mode {local,server}`: 执行模式
  - `local`: 本地Docker模式（默认）
  - `server`: 服务器直接执行模式
- `--experiment-type {hpwl,ablation}`: 实验类型
  - `hpwl`: HPWL对比实验（默认）
  - `ablation`: 消融实验

## 实验内容

### HPWL对比实验（--experiment-type hpwl）

包含以下步骤：

1. **数据准备阶段**：加载设计数据和配置
2. **RL训练阶段**：训练强化学习代理
3. **ChipDRAG优化阶段**：使用训练好的模型进行布局优化
4. **HPWL对比分析**：对比OpenROAD默认布局和ChipDRAG优化布局
5. **推理验证**：验证训练好的模型性能
6. **消融实验**：验证三大创新点的贡献
7. **结果生成**：生成完整的实验报告

### 消融实验（--experiment-type ablation）

验证ChipDRAG的三个核心技术贡献：

1. **基线实验**：完整ChipDRAG系统性能
2. **消融1**：移除强化学习驱动的动态重排序机制
3. **消融2**：移除实体压缩和注入技术
4. **消融3**：移除质量反馈驱动的闭环优化框架
5. **分析比较**：量化各技术贡献的重要性

## 输出结果

### HPWL对比实验结果

实验完成后会在 `paper_hpwl_results/` 目录下生成：
- 带时间戳的实验结果目录
- `complete_results.json` 完整实验结果
- 详细的日志文件

### 消融实验结果

实验完成后会在 `paper_hpwl_results/` 目录下生成：
- 带时间戳的消融实验结果目录
- `ablation_analysis.json` 消融实验分析
- `ablation_report.md` 消融实验报告
- 详细的日志文件

## 环境要求

### 本地模式
- Docker 已安装并运行
- 至少3GB可用内存
- 足够的CPU资源

### 服务器模式
- 系统已安装OpenROAD
- 必要的设计文件存在于 `dataset/` 或 `data/` 目录

## 注意事项

1. 实验需要真实的设计文件，不使用模拟数据
2. 确保数据目录 `dataset/ispd_2015_contest_benchmark` 或 `data/designs/ispd_2015_contest_benchmark` 存在
3. 本地模式需要Docker镜像 `openroad/flow-ubuntu22.04-builder:21e414`
4. 服务器模式需要系统PATH中包含OpenROAD二进制文件

## 技术特点

- **统一接口**：一个脚本支持两种执行环境
- **智能适配**：根据模式自动调整OpenROAD执行方式
- **完整流程**：从数据准备到结果生成的完整实验流程
- **错误处理**：完善的异常处理和日志记录

## 故障排除

1. **导入错误**：确保在项目根目录下运行
2. **Docker错误**：确保Docker服务正在运行
3. **OpenROAD错误**：检查OpenROAD安装和PATH配置
4. **文件缺失**：确保设计文件目录结构正确

## 版本历史

- v2.0: 完全统一的论文实验脚本
  - 集成了原有的三个实验文件：`paper_hpwl_comparison_experiment_fixed.py`、`paper_hpwl_comparison_experiment_server.py`、`paper_ablation_experiment.py`
  - 支持两种实验类型：HPWL对比实验和消融实验
  - 支持两种执行模式：本地Docker和服务器直接执行
  - 统一的命令行接口和结果格式
- v1.0: 基础版本
  - 统一了HPWL对比实验的本地和服务器模式
  - 支持通过命令行参数区分执行模式
  - 改进了错误处理和日志记录 