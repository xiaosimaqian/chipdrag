#!/bin/bash

# ChipDRAG服务器环境安装脚本
# 适用于Ubuntu 20.04/22.04和CentOS 8/9

echo "=== ChipDRAG 服务器环境安装 ==="

# 1. 更新系统
echo "1. 更新系统..."
if [ -f /etc/debian_version ]; then
    apt update && apt upgrade -y
    apt install -y curl wget git vim build-essential
elif [ -f /etc/redhat-release ]; then
    yum update -y
    yum install -y curl wget git vim gcc gcc-c++ make
fi

# 2. 安装Python 3.8+
echo "2. 安装Python 3.8+..."
if [ -f /etc/debian_version ]; then
    apt install -y python3.8 python3.8-venv python3.8-dev python3-pip
elif [ -f /etc/redhat-release ]; then
    yum install -y python3 python3-venv python3-devel python3-pip
fi

# 3. 安装Docker
echo "3. 安装Docker..."
if [ -f /etc/debian_version ]; then
    apt install -y docker.io docker-compose
elif [ -f /etc/redhat-release ]; then
    yum install -y docker docker-compose
fi

# 启动Docker服务
systemctl enable docker
systemctl start docker

# 添加用户到docker组
usermod -aG docker $USER

# 4. 安装Ollama (用于本地LLM)
echo "4. 安装Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# 5. 验证安装
echo "5. 验证安装..."
python3 --version
pip3 --version
docker --version
ollama --version

echo "=== 环境安装完成 ==="
echo "请重新登录以应用docker组权限" 