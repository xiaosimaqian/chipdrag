#!/bin/bash

# ChipDRAG项目服务器部署脚本

echo "=== ChipDRAG项目部署脚本 ==="

# 设置项目目录
PROJECT_DIR="/opt/chipdrag"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data"

# 1. 创建项目目录
echo "1. 创建项目目录..."
mkdir -p $PROJECT_DIR
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
cd $PROJECT_DIR

# 2. 克隆项目（如果从Git仓库）
echo "2. 部署项目代码..."
# 方式1: 从Git仓库克隆
# git clone https://github.com/your-repo/chipdrag.git .

# 方式2: 从本地上传（推荐）
echo "请将项目文件上传到: $PROJECT_DIR"
echo "使用命令: scp -r /path/to/chipdrag/* user@server:$PROJECT_DIR/"

# 3. 创建Python虚拟环境
echo "3. 创建Python虚拟环境..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 4. 安装Python依赖
echo "4. 安装Python依赖..."
pip install --upgrade pip

# 基础依赖
pip install numpy pandas matplotlib seaborn
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install requests aiohttp
pip install pathlib configparser
pip install psutil

# 如果有requirements.txt文件
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# 5. 配置Ollama模型
echo "5. 配置Ollama模型..."
ollama pull deepseek-coder
ollama pull llama2

# 6. 拉取OpenROAD Docker镜像
echo "6. 拉取OpenROAD Docker镜像..."
docker pull openroad/openroad:latest

# 7. 设置权限
echo "7. 设置权限..."
chown -R $USER:$USER $PROJECT_DIR
chmod +x $PROJECT_DIR/*.py
chmod +x $PROJECT_DIR/scripts/*.py

# 8. 创建服务配置
echo "8. 创建系统服务配置..."
cat > /etc/systemd/system/chipdrag.service << EOF
[Unit]
Description=ChipDRAG Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
ExecStart=$VENV_DIR/bin/python paper_hpwl_comparison_experiment_fixed.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# 9. 启用服务
systemctl daemon-reload
systemctl enable chipdrag

echo "=== 部署完成 ==="
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: $VENV_DIR"
echo "日志目录: $LOG_DIR"
echo ""
echo "启动服务: systemctl start chipdrag"
echo "查看状态: systemctl status chipdrag"
echo "查看日志: journalctl -u chipdrag -f" 