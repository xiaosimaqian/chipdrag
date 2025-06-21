# 真实数据收集指南

## 概述

本文档介绍如何收集芯片设计领域的真实数据，用于Chip-D-RAG系统的训练和优化。

## 数据收集策略

### 1. EDA工具日志数据

#### 1.1 Synopsys工具
- **Design Compiler (DC)**: 综合日志、约束文件、网表
- **IC Compiler**: 布局布线日志、时序报告、功耗报告
- **PrimeTime**: 时序分析报告、约束违反报告

#### 1.2 Cadence工具
- **Innovus**: 布局布线日志、质量报告
- **Genus**: 综合日志、网表文件
- **Tempus**: 时序分析报告

#### 1.3 Mentor工具
- **Calibre**: DRC/LVS报告、验证结果
- **Olympus**: 布局布线日志

### 2. 设计仓库数据

#### 2.1 开源设计项目
- **GitHub**: 搜索关键词如 "RISC-V", "DSP", "ASIC", "FPGA"
- **GitLab**: 企业级设计项目
- **OpenCores**: 开源IP核

#### 2.2 设计文件类型
- **RTL**: `.v`, `.vhd` (Verilog/VHDL)
- **网表**: `.v`, `.sp` (SPICE)
- **布局**: `.def`, `.lef`, `.gds`
- **约束**: `.sdc`, `.tcl`
- **库文件**: `.lib`, `.db`

### 3. 用户交互数据

#### 3.1 查询日志
- 用户输入的布局生成查询
- 查询类型和复杂度
- 用户反馈和评分

#### 3.2 质量反馈
- 布局质量评估
- 约束满足度
- 性能指标反馈

## 数据收集工具

### 1. 自动化收集脚本

```python
# 示例：从EDA工具收集数据
def collect_from_eda_tools():
    # 解析日志文件
    # 提取设计信息
    # 提取质量指标
    pass

# 示例：从GitHub收集数据
def collect_from_github():
    # 使用GitHub API
    # 下载设计文件
    # 解析文件内容
    pass
```

### 2. 数据解析器

```python
# Verilog文件解析器
def parse_verilog_file(file_path):
    # 提取模块信息
    # 提取端口信息
    # 提取实例信息
    pass

# DEF文件解析器
def parse_def_file(file_path):
    # 提取布局信息
    # 提取组件信息
    # 提取网络信息
    pass
```

## 数据质量要求

### 1. 数据完整性
- 查询-设计-结果三元组完整
- 质量评估指标齐全
- 元数据信息完整

### 2. 数据多样性
- 不同设计类型 (RISC-V, DSP, Memory等)
- 不同技术节点 (14nm, 28nm, 40nm等)
- 不同约束组合

### 3. 数据真实性
- 来自真实设计项目
- 经过实际验证
- 有质量保证

## 数据收集流程

### 阶段1: 数据源识别
1. 确定可用的EDA工具环境
2. 识别开源设计项目
3. 建立用户交互渠道

### 阶段2: 数据收集
1. 配置收集工具
2. 执行自动化收集
3. 手动收集补充数据

### 阶段3: 数据预处理
1. 数据清洗和去重
2. 格式标准化
3. 质量检查

### 阶段4: 数据标注
1. 设计类型标注
2. 约束类型标注
3. 质量分数标注

## 数据存储格式

### 1. 结构化数据 (JSON)
```json
{
  "query": {
    "text": "Generate layout for RISC-V processor",
    "design_type": "risc_v",
    "constraints": ["timing", "power"],
    "complexity": 0.8
  },
  "design_info": {
    "technology_node": "14nm",
    "area_constraint": 5000,
    "power_budget": 3.0,
    "timing_constraint": 2.0
  },
  "layout_result": {
    "wirelength": 15000,
    "congestion": 0.15,
    "timing_score": 0.85,
    "power_score": 0.78
  },
  "quality_feedback": {
    "overall_score": 0.82,
    "user_rating": 4.5,
    "feedback_text": "Good layout quality"
  }
}
```

### 2. 数据库存储
- SQLite数据库
- 分表存储不同类型数据
- 建立关联关系

## 数据收集工具配置

### 1. EDA工具配置
```yaml
eda_tools:
  synopsys:
    dc_log_dir: "/path/to/dc/logs"
    ic_log_dir: "/path/to/ic/logs"
    pt_log_dir: "/path/to/pt/logs"
  cadence:
    innovus_log_dir: "/path/to/innovus/logs"
    genus_log_dir: "/path/to/genus/logs"
```

### 2. 仓库配置
```yaml
repositories:
  github:
    token: "your_github_token"
    repos:
      - "riscv/riscv-cores"
      - "opencores/openrisc"
  gitlab:
    token: "your_gitlab_token"
    base_url: "https://gitlab.com"
    projects:
      - "group/project1"
```

### 3. 用户交互配置
```yaml
user_interactions:
  log_dir: "/path/to/query/logs"
  feedback_dir: "/path/to/feedback"
  api_endpoint: "http://localhost:8000/api"
```

## 数据验证和质量控制

### 1. 数据验证规则
- 文件格式检查
- 数据完整性检查
- 数值范围检查

### 2. 质量控制指标
- 数据覆盖率
- 数据准确性
- 数据一致性

### 3. 数据清洗流程
- 去除重复数据
- 修复格式错误
- 补充缺失信息

## 隐私和安全考虑

### 1. 数据脱敏
- 移除敏感信息
- 匿名化处理
- 加密存储

### 2. 访问控制
- 权限管理
- 审计日志
- 数据备份

### 3. 合规要求
- 遵守数据保护法规
- 获得必要授权
- 建立数据使用协议

## 持续数据收集

### 1. 自动化收集
- 定时任务
- 事件触发
- 增量更新

### 2. 数据监控
- 收集进度监控
- 数据质量监控
- 系统性能监控

### 3. 反馈循环
- 用户反馈收集
- 系统性能评估
- 数据质量改进

## 最佳实践

### 1. 数据收集
- 建立标准化的收集流程
- 使用版本控制管理配置
- 定期备份收集的数据

### 2. 数据处理
- 建立数据质量检查机制
- 使用自动化工具处理
- 保持数据格式一致性

### 3. 数据管理
- 建立数据目录
- 实施数据生命周期管理
- 定期清理过期数据

## 常见问题和解决方案

### 1. 数据格式不统一
- 建立数据格式标准
- 开发格式转换工具
- 实施数据验证

### 2. 数据质量参差不齐
- 建立质量评估标准
- 实施数据清洗流程
- 建立质量反馈机制

### 3. 数据量不足
- 扩大数据源范围
- 实施数据增强技术
- 建立数据共享机制

## 总结

真实数据收集是Chip-D-RAG系统成功的关键。通过系统性的数据收集策略、完善的工具支持、严格的质量控制，可以建立高质量的训练数据集，为系统性能提升提供有力支撑。 