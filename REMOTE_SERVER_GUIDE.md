# 远程服务器执行指南

## 🚀 远程服务器部署和运行ChipDRAG实验

### 1. 文件传输到服务器

#### 方法1：使用scp传输文件
```bash
# 传输整个项目目录到服务器
scp -r /path/to/chipdrag username@server_ip:/path/to/destination/

# 或者只传输必要文件
scp paper_hpwl_comparison_experiment_server.py username@server_ip:~/
scp test_openroad_diagnosis.py username@server_ip:~/
scp fix_openroad_server.py username@server_ip:~/
scp -r dataset username@server_ip:~/
```

#### 方法2：使用rsync同步
```bash
# 同步整个项目（推荐）
rsync -avz --progress /path/to/chipdrag/ username@server_ip:/path/to/destination/

# 排除不必要的文件
rsync -avz --progress --exclude='*.pyc' --exclude='__pycache__' /path/to/chipdrag/ username@server_ip:~/chipdrag/
```

### 2. 远程服务器登录

```bash
# SSH登录到服务器
ssh username@server_ip

# 或者使用密钥登录
ssh -i /path/to/private_key username@server_ip
```

### 3. 服务器环境检查和修复

#### 步骤1：检查Python环境
```bash
# 检查Python版本
python3 --version

# 检查pip
pip3 --version

# 安装必要的Python包
pip3 install psutil numpy torch transformers
```

#### 步骤2：检查OpenROAD安装
```bash
# 检查OpenROAD是否安装
which openroad

# 如果未安装，可能需要安装OpenROAD
# 具体安装方法取决于服务器操作系统
```

#### 步骤3：运行诊断脚本
```bash
# 进入项目目录
cd ~/chipdrag

# 运行OpenROAD诊断
python3 test_openroad_diagnosis.py

# 运行修复脚本
python3 fix_openroad_server.py
```

### 4. 远程运行实验

#### 方法1：直接在SSH会话中运行
```bash
# 运行完整实验
python3 paper_hpwl_comparison_experiment_server.py

# 运行简化实验
python3 fixed_experiment_simple.py
```

#### 方法2：使用screen或tmux（推荐）
```bash
# 安装screen（如果没有）
sudo apt-get install screen  # Ubuntu/Debian
# 或
sudo yum install screen      # CentOS/RHEL

# 创建新的screen会话
screen -S chipdrag_experiment

# 在screen会话中运行实验
python3 paper_hpwl_comparison_experiment_server.py

# 离开screen会话（实验继续运行）
# 按 Ctrl+A, 然后按 D

# 重新连接到screen会话
screen -r chipdrag_experiment
```

#### 方法3：使用nohup后台运行
```bash
# 后台运行实验
nohup python3 paper_hpwl_comparison_experiment_server.py > experiment_output.log 2>&1 &

# 查看进程
ps aux | grep python

# 查看实时输出
tail -f experiment_output.log
```

### 5. 监控和管理

#### 查看系统资源
```bash
# 查看CPU和内存使用
top
htop  # 如果安装了htop

# 查看磁盘使用
df -h

# 查看特定进程
ps aux | grep openroad
ps aux | grep python
```

#### 查看实验进度
```bash
# 查看日志文件
tail -f logs/experiment_*.log

# 查看结果目录
ls -la paper_hpwl_results_*
```

### 6. 结果获取

#### 下载结果到本地
```bash
# 下载结果目录
scp -r username@server_ip:~/chipdrag/paper_hpwl_results_* ./

# 下载日志文件
scp -r username@server_ip:~/chipdrag/logs ./

# 下载报告文件
scp username@server_ip:~/chipdrag/complete_experiment_report.md ./
```

### 7. 故障排除

#### 常见问题和解决方案

1. **OpenROAD命令未找到**
   ```bash
   # 检查PATH环境变量
   echo $PATH
   
   # 查找OpenROAD安装位置
   find /usr -name "openroad" 2>/dev/null
   find /opt -name "openroad" 2>/dev/null
   ```

2. **内存不足**
   ```bash
   # 检查内存使用
   free -h
   
   # 检查交换空间
   swapon -s
   
   # 减少并行任务数量
   # 编辑实验脚本，降低max_parallel_designs参数
   ```

3. **磁盘空间不足**
   ```bash
   # 检查磁盘使用
   df -h
   
   # 清理临时文件
   rm -rf /tmp/openroad_*
   rm -rf paper_hpwl_results_*/work_*
   ```

### 8. 远程执行脚本模板

创建一个本地脚本来自动化远程执行：

```bash
#!/bin/bash
# remote_execute.sh

SERVER_IP="your_server_ip"
USERNAME="your_username"
REMOTE_DIR="~/chipdrag"

echo "🚀 远程执行ChipDRAG实验"

# 1. 传输文件
echo "📤 传输文件到服务器..."
rsync -avz --progress --exclude='*.pyc' --exclude='__pycache__' ./ $USERNAME@$SERVER_IP:$REMOTE_DIR/

# 2. 远程执行
echo "🔧 远程执行实验..."
ssh $USERNAME@$SERVER_IP << 'EOF'
    cd ~/chipdrag
    python3 test_openroad_diagnosis.py
    if [ $? -eq 0 ]; then
        echo "✅ 诊断通过，开始实验"
        nohup python3 paper_hpwl_comparison_experiment_server.py > experiment_output.log 2>&1 &
        echo "🏃 实验已在后台启动"
    else
        echo "❌ 诊断失败，请检查环境"
    fi
EOF

# 3. 监控进度
echo "📊 监控实验进度..."
while true; do
    ssh $USERNAME@$SERVER_IP "cd $REMOTE_DIR && tail -10 experiment_output.log"
    sleep 30
done
```

### 9. 安全建议

1. **使用SSH密钥认证**
2. **定期备份实验数据**
3. **监控服务器资源使用**
4. **设置适当的文件权限**

### 10. 性能优化

1. **调整OpenROAD线程数**
2. **优化内存使用**
3. **使用SSD存储**
4. **配置合适的并行度**

---

## 📞 技术支持

如果在远程服务器执行过程中遇到问题，请提供：
1. 服务器操作系统版本
2. OpenROAD版本信息
3. 错误日志文件
4. 系统资源状态

这样可以获得更精确的技术支持。 