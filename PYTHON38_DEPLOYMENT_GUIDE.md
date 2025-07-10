# ChipDRAG Python 3.8.5 部署指南

## 📋 概述

本指南专门针对Python 3.8.5环境部署ChipDRAG系统，确保所有依赖的版本兼容性。

## ✅ Python 3.8.5 兼容性确认

### 支持的组件
- ✅ **核心框架**: 项目原生支持Python 3.8+
- ✅ **PyTorch**: 1.12.0-2.0.1 完全兼容
- ✅ **Transformers**: 4.20.0-4.33.0 兼容
- ✅ **科学计算**: NumPy, Pandas, Matplotlib 等全部兼容
- ✅ **系统工具**: 所有系统工具库兼容

### 已测试的版本组合
```
Python: 3.8.5
PyTorch: 2.0.1
Transformers: 4.33.0
NumPy: 1.23.5
Pandas: 1.4.4
```

## 🚀 快速部署

### 方式1: 使用专用部署脚本

```bash
# 1. 下载Python 3.8.5专用部署脚本
wget https://raw.githubusercontent.com/your-repo/chipdrag/main/deploy_python38.sh

# 2. 设置执行权限
chmod +x deploy_python38.sh

# 3. 以管理员权限运行
sudo ./deploy_python38.sh

# 4. 验证安装
/opt/chipdrag/start_python38.sh test
```

### 方式2: 手动部署

```bash
# 1. 创建项目目录
sudo mkdir -p /opt/chipdrag
sudo chown $USER:$USER /opt/chipdrag
cd /opt/chipdrag

# 2. 创建Python 3.8虚拟环境
python3.8 -m venv venv38
source venv38/bin/activate

# 3. 升级pip到兼容版本
pip install --upgrade pip==23.1.2

# 4. 安装兼容依赖
pip install -r requirements_python38.txt

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 📊 使用方法

### 基本运行

```bash
# 1. 激活环境并运行默认实验
/opt/chipdrag/start_python38.sh

# 2. 运行特定实验
/opt/chipdrag/start_python38.sh hpwl    # HPWL对比实验
/opt/chipdrag/start_python38.sh ablation  # 消融实验
/opt/chipdrag/start_python38.sh monitor   # 监控系统
```

### 系统服务方式

```bash
# 启动服务
sudo systemctl start chipdrag-python38

# 查看状态
sudo systemctl status chipdrag-python38

# 查看日志
sudo journalctl -u chipdrag-python38 -f

# 停止服务
sudo systemctl stop chipdrag-python38
```

### 手动运行

```bash
# 1. 激活虚拟环境
source /opt/chipdrag/venv38/bin/activate

# 2. 设置环境变量
export PYTHONPATH=/opt/chipdrag:$PYTHONPATH

# 3. 运行实验
cd /opt/chipdrag
python paper_hpwl_comparison_experiment_fixed.py
```

## 🔧 兼容性配置

### 1. 依赖版本锁定

如果遇到版本冲突，可以使用更严格的版本锁定：

```bash
# 创建严格版本锁定文件
cat > requirements_python38_strict.txt << EOF
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
transformers==4.33.0
numpy==1.23.5
pandas==1.4.4
matplotlib==3.6.3
scikit-learn==1.1.3
scipy==1.9.3
requests==2.28.2
psutil==5.9.4
EOF

# 安装严格版本
pip install -r requirements_python38_strict.txt
```

### 2. 环境变量配置

```bash
# 添加到 ~/.bashrc
export PYTHONPATH="/opt/chipdrag:$PYTHONPATH"
export CHIPDRAG_HOME="/opt/chipdrag"

# 针对Python 3.8的优化
export PYTHONPATH="/opt/chipdrag/venv38/lib/python3.8/site-packages:$PYTHONPATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 3. 内存优化配置

```bash
# 针对Python 3.8的内存优化
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=131072
```

## 🐛 常见问题解决

### 1. 依赖版本冲突

**问题**: 某些依赖无法安装或版本冲突

**解决方案**:
```bash
# 清理环境重新安装
rm -rf /opt/chipdrag/venv38
python3.8 -m venv /opt/chipdrag/venv38
source /opt/chipdrag/venv38/bin/activate

# 按顺序安装依赖
pip install wheel==0.40.0
pip install setuptools==67.8.0
pip install -r requirements_python38.txt
```

### 2. 模块导入错误

**问题**: 找不到模块或导入失败

**解决方案**:
```bash
# 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"

# 确保项目目录在Python路径中
export PYTHONPATH="/opt/chipdrag:$PYTHONPATH"

# 检查模块是否存在
python -c "from modules.core.rl_agent import QLearningAgent; print('OK')"
```

### 3. PyTorch版本问题

**问题**: PyTorch版本与Python 3.8不兼容

**解决方案**:
```bash
# 卸载当前PyTorch
pip uninstall torch torchvision torchaudio -y

# 安装Python 3.8兼容版本
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### 4. 性能优化

**问题**: 在Python 3.8环境下性能不佳

**解决方案**:
```bash
# 编译优化
export CFLAGS="-O2 -pipe"
export CXXFLAGS="-O2 -pipe"

# 使用优化版本安装
pip install --upgrade --force-reinstall numpy scipy scikit-learn
```

## 📋 验证清单

部署完成后，请检查以下项目：

```bash
# 1. Python版本
python --version  # 应该显示 3.8.x

# 2. 核心依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 3. 系统模块
python -c "from modules.core.rl_agent import QLearningAgent; print('RL Agent: OK')"
python -c "from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever; print('RAG Retriever: OK')"

# 4. 运行基础测试
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import platform; print(f'Platform: {platform.platform()}')"
```

## 🔄 维护和升级

### 定期维护

```bash
# 1. 更新兼容的依赖版本
pip list --outdated | grep -E "(numpy|pandas|matplotlib|scikit-learn)"

# 2. 清理缓存
pip cache purge
python -m pip cache purge

# 3. 检查环境健康
python -m pip check
```

### 升级策略

```bash
# 1. 备份当前环境
cp -r /opt/chipdrag/venv38 /opt/chipdrag/venv38_backup

# 2. 测试升级
pip install --upgrade transformers==4.35.0  # 测试新版本

# 3. 验证功能
python -c "from modules.core.rl_agent import QLearningAgent; print('OK')"

# 4. 回滚（如果需要）
rm -rf /opt/chipdrag/venv38
mv /opt/chipdrag/venv38_backup /opt/chipdrag/venv38
```

## 🚨 注意事项

1. **不要升级Python版本**: 保持使用Python 3.8.5以确保兼容性
2. **依赖版本锁定**: 不要随意升级核心依赖版本
3. **环境隔离**: 使用独立的虚拟环境避免冲突
4. **定期备份**: 在重要实验前备份环境
5. **资源监控**: 密切监控内存和CPU使用情况

## 📞 技术支持

如果在Python 3.8.5环境中遇到问题，请提供以下信息：
- Python版本: `python --version`
- 系统信息: `uname -a`
- 依赖版本: `pip list`
- 错误日志: 完整的错误信息

---

*本指南针对Python 3.8.5环境优化，确保ChipDRAG系统在该版本下稳定运行。* 