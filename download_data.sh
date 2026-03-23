
set -e

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
    echo "⚠️  conda 命令不可用"
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
echo "开始下载数据..."
echo ""

$PYTHON scripts/download_data.py \

echo ""
echo "✅ 下载完成！"
