# ChipDRAG服务器部署指南

## 📋 概述

本指南详细介绍如何在服务器上部署和运行ChipDRAG芯片布局优化系统。

## 🔧 系统要求

### 硬件配置
- **CPU**: 16核以上 (推荐Intel Xeon或AMD EPYC)
- **内存**: 64GB以上 (推荐128GB)
- **存储**: 500GB以上 SSD
- **网络**: 千兆网络

### 软件要求
- **操作系统**: Ubuntu 20.04/22.04 LTS 或 CentOS 8/9
- **Python**: 3.8以上
- **Docker**: 20.10以上
- **Git**: 2.25以上

## 🚀 部署步骤

### 步骤1: 准备服务器环境

```bash
# 1. 下载并执行环境安装脚本
curl -O https://raw.githubusercontent.com/your-repo/chipdrag/main/server_setup.sh
chmod +x server_setup.sh
sudo ./server_setup.sh

# 2. 重新登录以应用docker组权限
exit
# 重新SSH登录
```

### 步骤2: 项目部署

```bash
# 1. 创建项目目录
sudo mkdir -p /opt/chipdrag
sudo chown $USER:$USER /opt/chipdrag

# 2. 上传项目文件
# 方式1: 使用scp上传
scp -r /path/to/chipdrag/* user@server:/opt/chipdrag/

# 方式2: 使用rsync同步
rsync -av --exclude='.*' /path/to/chipdrag/ user@server:/opt/chipdrag/

# 3. 执行部署脚本
cd /opt/chipdrag
chmod +x deploy_server.sh
sudo ./deploy_server.sh
```

### 步骤3: 配置系统服务

```bash
# 1. 检查服务状态
sudo systemctl status chipdrag

# 2. 启动服务
sudo systemctl start chipdrag

# 3. 设置开机自启
sudo systemctl enable chipdrag

# 4. 查看服务日志
sudo journalctl -u chipdrag -f
```

### 步骤4: 验证部署

```bash
# 1. 检查Python环境
/opt/chipdrag/venv/bin/python --version

# 2. 检查依赖包
/opt/chipdrag/venv/bin/pip list

# 3. 检查Docker
docker ps
docker images | grep openroad

# 4. 检查Ollama
ollama list

# 5. 测试基本功能
cd /opt/chipdrag
/opt/chipdrag/venv/bin/python -c "from modules.core.rl_agent import QLearningAgent; print('✓ 模块导入成功')"
```

## 🔧 配置说明

### 1. 修改服务器配置

编辑 `/opt/chipdrag/server_config.json`:

```json
{
  "server_config": {
    "system": {
      "project_dir": "/opt/chipdrag",
      "log_dir": "/opt/chipdrag/logs",
      "data_dir": "/opt/chipdrag/data"
    },
    "resources": {
      "cpu": {
        "cores": 16,
        "max_usage": "80%"
      },
      "memory": {
        "total": "64GB",
        "max_usage": "75%"
      }
    }
  }
}
```

### 2. 修改实验配置

编辑 `/opt/chipdrag/configs/experiment_config.json`:

```json
{
  "experiment": {
    "designs": ["mgc_fft_1", "mgc_des_perf_1", "mgc_matrix_mult_1"],
    "max_concurrent_designs": 2,
    "max_concurrent_containers": 1
  }
}
```

### 3. 配置LLM服务

编辑 `/opt/chipdrag/configs/llm/ollama.json`:

```json
{
  "base_url": "http://localhost:11434",
  "model": "deepseek-coder",
  "temperature": 0.7,
  "timeout": 30
}
```

## 🎯 运行实验

### 1. 手动运行实验

```bash
# 激活虚拟环境
source /opt/chipdrag/venv/bin/activate

# 运行HPWL对比实验
cd /opt/chipdrag
python paper_hpwl_comparison_experiment_fixed.py

# 运行消融实验
python paper_ablation_experiment.py
```

### 2. 使用系统服务

```bash
# 启动实验服务
sudo systemctl start chipdrag

# 查看实验进度
sudo journalctl -u chipdrag -f

# 停止实验
sudo systemctl stop chipdrag
```

### 3. 批量运行实验

```bash
# 批量运行多个设计
python batch_train_ispd_optimized.py

# 并行运行实验
python batch_train_ispd_parallel.py
```

## 📊 监控和管理

### 1. 启动监控系统

```bash
# 启动监控
cd /opt/chipdrag
python monitor_server.py --interval 30

# 后台运行监控
nohup python monitor_server.py > monitor.log 2>&1 &

# 查看监控状态
python monitor_server.py --once
```

### 2. 查看实验结果

```bash
# 查看结果目录
ls -la /opt/chipdrag/paper_hpwl_results*/

# 查看最新实验结果
ls -la /opt/chipdrag/paper_hpwl_results_*/experiment_results.json

# 查看实验报告
ls -la /opt/chipdrag/paper_hpwl_results_*/experiment_report.md
```

### 3. 日志管理

```bash
# 查看系统日志
sudo journalctl -u chipdrag -f

# 查看应用日志
tail -f /opt/chipdrag/logs/experiment_log_*.log

# 查看监控日志
tail -f /opt/chipdrag/logs/monitor.log
```

## 🔧 维护和故障排除

### 1. 常见问题

#### 内存不足
```bash
# 检查内存使用
free -h
top -o %MEM

# 调整Docker内存限制
# 编辑 /opt/chipdrag/server_config.json
"docker": {
  "memory_limit": "8GB"
}
```

#### Docker容器失败
```bash
# 检查Docker状态
docker ps -a

# 查看容器日志
docker logs <container_id>

# 清理失败的容器
docker system prune -f
```

#### OpenROAD执行失败
```bash
# 检查OpenROAD镜像
docker images | grep openroad

# 重新拉取镜像
docker pull openroad/openroad:latest

# 测试OpenROAD
docker run --rm openroad/openroad:latest openroad -version
```

### 2. 性能优化

#### CPU优化
```bash
# 调整CPU核心数
# 编辑 /opt/chipdrag/server_config.json
"resources": {
  "cpu": {
    "cores": 32,
    "max_usage": "90%"
  }
}
```

#### 内存优化
```bash
# 增加交换空间
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
```

#### 磁盘优化
```bash
# 清理临时文件
sudo rm -rf /tmp/chipdrag/*

# 清理旧的实验结果
find /opt/chipdrag/paper_hpwl_results_* -mtime +30 -type d -exec rm -rf {} \;
```

### 3. 备份和恢复

#### 备份配置
```bash
# 创建备份目录
sudo mkdir -p /backup/chipdrag

# 备份配置文件
sudo cp -r /opt/chipdrag/configs /backup/chipdrag/

# 备份实验结果
sudo cp -r /opt/chipdrag/paper_hpwl_results_* /backup/chipdrag/
```

#### 恢复配置
```bash
# 恢复配置文件
sudo cp -r /backup/chipdrag/configs /opt/chipdrag/

# 重启服务
sudo systemctl restart chipdrag
```

## 📈 性能调优建议

### 1. 硬件优化
- 使用SSD存储以提高I/O性能
- 增加内存容量以减少swap使用
- 使用多核CPU以提高并行处理能力

### 2. 软件优化
- 调整Docker资源限制
- 优化Python虚拟环境
- 使用本地缓存加速模型加载

### 3. 实验优化
- 合理设置并发数量
- 使用增量实验减少重复计算
- 启用结果缓存机制

## 🚨 安全注意事项

1. **防火墙配置**: 只开放必要端口
2. **访问控制**: 使用SSH密钥认证
3. **数据备份**: 定期备份重要数据
4. **日志监控**: 监控异常访问记录
5. **系统更新**: 及时更新系统和软件

## 📞 技术支持

如有问题，请联系：
- 邮箱: support@chipdrag.com
- 文档: https://chipdrag.readthedocs.io
- GitHub: https://github.com/your-repo/chipdrag

---

*本文档最后更新时间: 2024年7月6日* 