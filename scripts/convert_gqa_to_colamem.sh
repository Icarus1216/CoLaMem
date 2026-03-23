#!/bin/bash
# GQA 数据格式转换为 ColaMem 训练格式脚本
# 用于将处理后的 GQA 数据转换为 ColaMem 训练所需的格式

set -e

echo "🔄 GQA -> ColaMem 格式转换脚本"
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

# ============================================================
# 配置参数
# ============================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# 输入输出路径
INPUT_JSON="/mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/processed/gqa_train_balanced.json"
OUTPUT_JSON="/mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/processed/colamem_gqa_train.json"
IMAGE_BASE_DIR="/mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/processed"

echo "============================================================"
echo "📁 输入文件: ${INPUT_JSON}"
echo "📁 输出文件: ${OUTPUT_JSON}"
echo "🖼️ 图像基础目录: ${IMAGE_BASE_DIR}"
echo "============================================================"
echo ""

# 检查输入文件是否存在
if [ ! -f "${INPUT_JSON}" ]; then
    echo "❌ 错误: 输入文件不存在: ${INPUT_JSON}"
    exit 1
fi

# 运行转换脚本
echo "开始转换..."
echo ""

$PYTHON ${PROJECT_DIR}/convert_gqa_to_colamem.py \
    --input "${INPUT_JSON}" \
    --output "${OUTPUT_JSON}" \
    --image_base_dir "${IMAGE_BASE_DIR}"

echo ""
echo "============================================================"
echo "✅ 格式转换完成！"
echo "============================================================"
