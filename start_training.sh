#!/bin/bash
# ColaMem 简化启动脚本 - 适配 Gemini 平台
# 使用 Python 模块方式启动 DeepSpeed，避免依赖 deepspeed 命令

set -e

echo "🚀 ColaMem 训练启动 (Gemini 平台)"
echo ""

# 初始化 conda（如果存在）
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda (miniconda3)..."
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda (/opt/conda)..."
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda ($HOME/miniconda3)..."
    source $HOME/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "🔧 初始化 conda (anaconda3)..."
    source $HOME/anaconda3/etc/profile.d/conda.sh
elif [ -n "$CONDA_EXE" ]; then
    echo "🔧 使用环境变量中的 conda..."
    eval "$($CONDA_EXE shell.bash hook)"
else
    echo "⚠️  未找到 conda，尝试直接使用 conda 命令..."
fi

# 激活 conda 环境
if command -v conda &> /dev/null; then
    echo "✅ conda 已就绪"
    
    # 检查 colamem 环境是否存在
    if conda env list | grep -q "^colamem "; then
        echo "🚀 激活 conda 环境: colamem"
        conda activate colamem
        echo "✅ conda 环境已激活: $(conda info --envs | grep '*' | awk '{print $1}')"
    else
        echo "⚠️  未找到 conda 环境 'colamem'，使用当前环境"
    fi
else
    echo "⚠️  conda 命令不可用，尝试激活 Python 虚拟环境..."
    
    # 回退到 Python 虚拟环境
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -d "/root/colamem" ]; then
            echo "检测到虚拟环境 /root/colamem，正在激活..."
            source /root/colamem/bin/activate
        elif [ -d "$HOME/colamem" ]; then
            echo "检测到虚拟环境 $HOME/colamem，正在激活..."
            source $HOME/colamem/bin/activate
        elif [ -d "colamem" ]; then
            echo "检测到虚拟环境 ./colamem，正在激活..."
            source colamem/bin/activate
        elif [ -d "venv" ]; then
            echo "检测到虚拟环境 ./venv，正在激活..."
            source venv/bin/activate
        else
            echo "⚠️  未检测到任何环境，使用系统 Python"
        fi
    else
        echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
    fi
fi
echo ""

# 检测 Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "Python: $($PYTHON --version)"
echo ""

# 检测 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "8")
else
    GPU_COUNT=8  # Gemini 平台默认 8 卡
fi

echo "GPU 数量: $GPU_COUNT"
echo ""

# 创建输出目录
mkdir -p outputs/colamem_phase1

# 使用 DeepSpeed 启动训练
echo "开始训练..."
echo ""

# 使用 deepspeed 命令（推荐方式）
deepspeed --num_gpus=$GPU_COUNT \
    scripts/train.py \
    --config configs/train_phase1_lora.yaml

echo ""
echo "✅ 训练完成！"
