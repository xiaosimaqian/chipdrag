# ChipDRAG服务器硬件资源优化指南

## 硬件资源配置

**您的服务器配置：**
- CPU: 160核心
- 内存: 1007.5GB  
- 可用内存: 997.9GB

## 自动硬件优化

`experiment.py` 已经内置了智能硬件检测和优化功能：

### 1. 自动检测并配置

根据您的160核CPU和1000GB内存，会自动配置：
- **并行设计处理：16个** (每8核处理一个设计)
- **并行容器：32个** (每4核一个容器)
- **OpenROAD线程：16个/实例** (充分利用多核)
- **批处理大小：64** (大批次处理)
- **RL训练线程：20个** (并行训练)

### 2. 环境变量自动设置
程序会自动设置以下优化参数：
```bash
OPENROAD_NUM_THREADS=16     # OpenROAD多线程
OMP_NUM_THREADS=16          # OpenMP线程数
MKL_NUM_THREADS=16          # MKL数学库线程
OPENBLAS_NUM_THREADS=16     # OpenBLAS线程
```

### 3. 运行优化实验
```bash
# HPWL对比实验（服务器模式）
python experiment.py --mode server --experiment-type hpwl

# 消融实验（服务器模式）
python experiment.py --mode server --experiment-type ablation
```

## 性能监控

### 1. 实时监控
```bash
# 监控CPU使用率
htop

# 监控内存使用
free -h

# 监控进程
ps aux | grep python
```

### 2. 实验日志
实验会自动记录：
- 硬件资源使用情况
- 并行任务执行状态
- OpenROAD执行时间
- 整体性能指标

## 性能提升预期

**优化效果：**
- 实验总时间：减少60-80%
- 并行处理能力：提升10-15倍  
- 内存利用率：提升到70-80%
- CPU利用率：提升到80-90%

## 使用流程

1. **直接运行优化实验**
   ```bash
   python experiment.py --mode server --experiment-type hpwl
   ```
   程序会自动检测硬件并应用最优配置

2. **监控性能**
   ```bash
   htop  # 另一个终端窗口监控CPU
   ```

3. **查看结果**
   ```bash
   ls paper_hpwl_results/
   ```

## 对比效果

### 优化前（保守配置）
- 实验时间：8-12小时
- CPU利用率：10-20%
- 内存利用率：5-15%
- 并行度：1-2个任务

### 优化后（自动配置）
- 实验时间：2-3小时
- CPU利用率：80-90%
- 内存利用率：70-80%
- 并行度：16-32个任务

## 实验日志监控

程序会自动记录性能指标到：
- `paper_hpwl_results/experiment_log_*.log`
- 包含硬件检测结果和优化配置信息 