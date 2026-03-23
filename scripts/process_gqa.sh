#!/bin/bash
# GQA 数据集 Parquet 处理脚本 - 适配 Gemini 平台
# 用于将 GQA 数据集的 parquet 文件转换为 JSON 和图像文件

set -e

echo "🚀 GQA 数据集处理启动 (Gemini 平台)"
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

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# GQA 数据集路径（分离存储格式）
# QA 数据目录（包含 id, imageId, question, answer 等列）
INSTRUCTIONS_DIR="/mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/train_balanced_instructions"

# 图像数据目录（包含 id, image 列）
IMAGES_DIR="/mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/train_balanced_images"

# 输出路径：处理后的数据输出目录
OUTPUT_DIR="/mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/processed"

# 输出 JSON 文件名
OUTPUT_JSON="gqa_train_balanced.json"

# 图像格式
IMAGE_FORMAT="jpg"

# 是否使用完整答案（fullAnswer）而不是简短答案（answer）
USE_FULL_ANSWER=false

# 最大处理样本数
MAX_SAMPLES="200000"

# ============================================================
# 显示配置信息
# ============================================================

echo "============================================================"
echo "📦 GQA 数据集处理配置（分离存储格式）"
echo "============================================================"
echo "📁 项目目录: ${PROJECT_DIR}"
echo "📥 QA 数据目录: ${INSTRUCTIONS_DIR}"
echo "📥 图像数据目录: ${IMAGES_DIR}"
echo "📤 输出目录: ${OUTPUT_DIR}"
echo "📄 输出 JSON: ${OUTPUT_JSON}"
echo "🖼️ 图像格式: ${IMAGE_FORMAT}"
echo "📝 使用完整答案: ${USE_FULL_ANSWER}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "🔢 最大样本数: ${MAX_SAMPLES}"
else
    echo "🔢 最大样本数: 全部处理"
fi
echo "============================================================"
echo ""

# 检查输入目录是否存在
if [ ! -d "$INSTRUCTIONS_DIR" ]; then
    echo "❌ 错误：QA 数据目录不存在: ${INSTRUCTIONS_DIR}"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ 错误：图像数据目录不存在: ${IMAGES_DIR}"
    exit 1
fi

# 显示输入目录中的 parquet 文件
echo "📋 QA 数据目录中的 Parquet 文件："
ls -lh "${INSTRUCTIONS_DIR}"/*.parquet 2>/dev/null || echo "  (未找到 .parquet 文件，将搜索子目录)"
echo ""
echo "📋 图像数据目录中的 Parquet 文件："
ls -lh "${IMAGES_DIR}"/*.parquet 2>/dev/null | head -5 || echo "  (未找到 .parquet 文件，将搜索子目录)"
echo "  ...(更多文件省略)"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
echo "✅ 输出目录已创建: ${OUTPUT_DIR}"
echo ""

# ============================================================
# 运行处理脚本
# ============================================================

echo "🔄 开始处理 GQA 数据集..."
echo ""

# 构建命令（使用分离存储格式）
CMD="$PYTHON ${SCRIPT_DIR}/process_gqa_parquet.py \
    --instructions_dir \"${INSTRUCTIONS_DIR}\" \
    --images_dir \"${IMAGES_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --output_json \"${OUTPUT_JSON}\" \
    --image_format \"${IMAGE_FORMAT}\""

# 如果使用完整答案，添加参数
if [ "$USE_FULL_ANSWER" = true ]; then
    CMD="$CMD --use_full_answer"
fi

# 如果指定了最大样本数，添加参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples ${MAX_SAMPLES}"
fi

# 执行命令
echo "执行命令："
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "✅ GQA 数据集处理完成！"
echo "============================================================"
echo ""
echo "📁 输出文件："
echo "  - JSON: ${OUTPUT_DIR}/${OUTPUT_JSON}"
echo "  - 图像: ${OUTPUT_DIR}/images/"
echo ""

# 显示输出统计
if [ -f "${OUTPUT_DIR}/${OUTPUT_JSON}" ]; then
    echo "📊 输出统计："
    echo "  - JSON 文件大小: $(ls -lh "${OUTPUT_DIR}/${OUTPUT_JSON}" | awk '{print $5}')"
    echo "  - 图像数量: $(ls "${OUTPUT_DIR}/images/" 2>/dev/null | wc -l)"
fi
