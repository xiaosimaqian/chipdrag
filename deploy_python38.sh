#!/bin/bash

# ChipDRAG Python 3.8.5 专用部署脚本

echo "=== ChipDRAG Python 3.8.5 环境部署 ==="

# 检查Python版本
echo "1. 检查Python版本..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "检测到Python版本: ${python_version}"

if [[ "${python_version}" == "3.8" ]]; then
    echo "✅ Python 3.8 版本兼容"
else
    echo "⚠️ 警告: 当前Python版本为 ${python_version}，推荐使用Python 3.8.5"
fi

# 设置项目目录
PROJECT_DIR="/opt/chipdrag"
VENV_DIR="$PROJECT_DIR/venv38"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data"

# 2. 创建项目目录
echo "2. 创建项目目录..."
mkdir -p $PROJECT_DIR
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
cd $PROJECT_DIR

# 3. 创建Python 3.8专用虚拟环境
echo "3. 创建Python 3.8专用虚拟环境..."
python3.8 -m venv $VENV_DIR || python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 4. 升级pip到兼容版本
echo "4. 升级pip到兼容版本..."
pip install --upgrade pip==23.1.2

# 5. 安装Python 3.8.5兼容依赖
echo "5. 安装Python 3.8.5兼容依赖..."

# 先安装基础依赖
pip install wheel==0.40.0
pip install setuptools==67.8.0

# 安装核心依赖 (确保版本兼容)
echo "安装核心科学计算库..."
pip install "numpy>=1.21.0,<1.24.0"
pip install "pandas>=1.3.0,<1.5.0"
pip install "matplotlib>=3.5.0,<3.7.0"
pip install "seaborn>=0.11.0,<0.12.0"

# 安装PyTorch (Python 3.8兼容版本)
echo "安装PyTorch (Python 3.8兼容版本)..."
pip install "torch>=1.12.0,<=2.0.1"
pip install "torchvision>=0.13.0,<=0.15.2"
pip install "torchaudio>=0.12.0,<=2.0.2"

# 安装Transformers (兼容版本)
echo "安装Transformers和相关库..."
pip install "transformers>=4.20.0,<=4.33.0"
pip install "sentence-transformers>=2.2.0,<2.3.0"
pip install "huggingface-hub>=0.10.0,<0.16.0"

# 安装系统工具
echo "安装系统工具..."
pip install "psutil>=5.8.0"
pip install "requests>=2.25.0,<2.32.0"
pip install "aiohttp>=3.8.0,<3.9.0"

# 安装数据处理库
echo "安装数据处理库..."
pip install "scikit-learn>=1.0.0,<1.2.0"
pip install "scipy>=1.7.0,<1.10.0"

# 如果有Python 3.8兼容的requirements文件，使用它
if [ -f requirements_python38.txt ]; then
    echo "使用Python 3.8兼容版本依赖..."
    pip install -r requirements_python38.txt
fi

# 6. 验证安装
echo "6. 验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas版本: {pandas.__version__}')"

# 7. 测试模块导入
echo "7. 测试关键模块导入..."
python -c "from modules.core.rl_agent import QLearningAgent; print('✅ RL Agent模块导入成功')" || echo "❌ RL Agent模块导入失败"
python -c "from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever; print('✅ RAG检索器模块导入成功')" || echo "❌ RAG检索器模块导入失败"

# 8. 配置环境变量
echo "8. 配置环境变量..."
echo "export PYTHONPATH=\$PYTHONPATH:$PROJECT_DIR" >> $HOME/.bashrc
echo "export CHIPDRAG_HOME=$PROJECT_DIR" >> $HOME/.bashrc

# 9. 创建Python 3.8专用启动脚本
echo "9. 创建Python 3.8专用启动脚本..."
cat > $PROJECT_DIR/start_python38.sh << 'EOF'
#!/bin/bash
# ChipDRAG Python 3.8.5 专用启动脚本

# 激活Python 3.8虚拟环境
source /opt/chipdrag/venv38/bin/activate

# 设置环境变量
export PYTHONPATH=${PYTHONPATH}:/opt/chipdrag
export CHIPDRAG_HOME=/opt/chipdrag

# 运行实验
cd /opt/chipdrag

echo "=== 启动ChipDRAG (Python 3.8.5环境) ==="
echo "Python版本: $(python --version)"
echo "工作目录: $(pwd)"

# 根据参数运行不同实验
case "${1:-default}" in
    "hpwl")
        echo "运行HPWL对比实验..."
        python paper_hpwl_comparison_experiment_fixed.py
        ;;
    "ablation")
        echo "运行消融实验..."
        python paper_ablation_experiment.py
        ;;
    "monitor")
        echo "启动监控系统..."
        python monitor_server.py --interval 30
        ;;
    "test")
        echo "运行系统测试..."
        python -m pytest tests/ -v
        ;;
    *)
        echo "运行默认实验..."
        python paper_hpwl_comparison_experiment_fixed.py
        ;;
esac
EOF

chmod +x $PROJECT_DIR/start_python38.sh

# 10. 创建系统服务配置
echo "10. 创建系统服务配置..."
cat > /etc/systemd/system/chipdrag-python38.service << EOF
[Unit]
Description=ChipDRAG Python 3.8.5 Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/start_python38.sh
Restart=on-failure
RestartSec=5
Environment=PYTHONPATH=$PROJECT_DIR

[Install]
WantedBy=multi-user.target
EOF

# 11. 启用服务
systemctl daemon-reload
systemctl enable chipdrag-python38

echo "=== Python 3.8.5 部署完成 ==="
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: $VENV_DIR"
echo "启动脚本: $PROJECT_DIR/start_python38.sh"
echo ""
echo "使用方法:"
echo "  基本启动: $PROJECT_DIR/start_python38.sh"
echo "  HPWL实验: $PROJECT_DIR/start_python38.sh hpwl"
echo "  消融实验: $PROJECT_DIR/start_python38.sh ablation"
echo "  监控系统: $PROJECT_DIR/start_python38.sh monitor"
echo "  系统测试: $PROJECT_DIR/start_python38.sh test"
echo ""
echo "系统服务:"
echo "  启动服务: systemctl start chipdrag-python38"
echo "  查看状态: systemctl status chipdrag-python38"
echo "  查看日志: journalctl -u chipdrag-python38 -f" 