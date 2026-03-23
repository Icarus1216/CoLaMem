#!/bin/bash
# TextVQA 数据集 Parquet 处理脚本 - 适配 Gemini 平台
# 用于将 TextVQA 数据集的 parquet 文件转换为 JSON 和图像文件

set -e

echo "🚀 TextVQA 数据集处理启动 (Gemini 平台)"
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

# TextVQA 数据集路径
TEXTVQA_ROOT="/mnt/cephszjt/user_juntianzhang/ColaMem/data/textvqa"

# Parquet 数据目录
PARQUET_DIR="${TEXTVQA_ROOT}/data"

# 图像输出目录
IMAGES_DIR="${TEXTVQA_ROOT}/images"

# JSON 输出目录
OUTPUT_DIR="${TEXTVQA_ROOT}"

# 处理脚本路径
PROCESS_SCRIPT="${SCRIPT_DIR}/process_textvqa.py"

# ============================================================
# 显示配置信息
# ============================================================

echo "============================================================"
echo "📦 TextVQA 数据集处理配置"
echo "============================================================"
echo "📁 项目目录: ${PROJECT_DIR}"
echo "📁 TextVQA 根目录: ${TEXTVQA_ROOT}"
echo "📥 Parquet 数据目录: ${PARQUET_DIR}"
echo "📤 图像输出目录: ${IMAGES_DIR}"
echo "📤 JSON 输出目录: ${OUTPUT_DIR}"
echo "📜 处理脚本: ${PROCESS_SCRIPT}"
echo "============================================================"
echo ""

# ============================================================
# 检查输入目录和文件是否存在
# ============================================================

if [ ! -d "$PARQUET_DIR" ]; then
    echo "❌ 错误：Parquet 数据目录不存在: ${PARQUET_DIR}"
    exit 1
fi

if [ ! -f "$PROCESS_SCRIPT" ]; then
    echo "❌ 错误：处理脚本不存在: ${PROCESS_SCRIPT}"
    exit 1
fi

# 显示输入目录中的 parquet 文件
echo "📋 Parquet 数据目录中的文件："
echo ""
echo "  训练集文件："
ls -lh "${PARQUET_DIR}"/train-*.parquet 2>/dev/null | awk '{print "    " $NF " (" $5 ")"}' || echo "    (未找到训练集 parquet 文件)"
echo ""
echo "  验证集文件："
ls -lh "${PARQUET_DIR}"/validation-*.parquet 2>/dev/null | awk '{print "    " $NF " (" $5 ")"}' || echo "    (未找到验证集 parquet 文件)"
echo ""
echo "  测试集文件："
ls -lh "${PARQUET_DIR}"/test-*.parquet 2>/dev/null | awk '{print "    " $NF " (" $5 ")"}' || echo "    (未找到测试集 parquet 文件)"
echo ""

# 统计 parquet 文件数量
TOTAL_PARQUET=$(ls "${PARQUET_DIR}"/*.parquet 2>/dev/null | wc -l)
echo "📊 共计 ${TOTAL_PARQUET} 个 Parquet 文件"
echo ""

# 创建输出目录
mkdir -p "${IMAGES_DIR}"
echo "✅ 输出目录已创建/确认: ${IMAGES_DIR}"
echo ""

# ============================================================
# 运行处理脚本
# ============================================================

echo "🔄 开始处理 TextVQA 数据集..."
echo ""

# 构建命令
CMD="$PYTHON ${PROCESS_SCRIPT}"

# 执行命令
echo "执行命令："
echo "  $CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "✅ TextVQA 数据集处理完成！"
echo "============================================================"
echo ""

# ============================================================
# 输出统计
# ============================================================

echo "📁 输出文件："
for SPLIT in train validation test; do
    JSON_FILE="${OUTPUT_DIR}/textvqa_${SPLIT}.json"
    IMG_DIR="${IMAGES_DIR}/${SPLIT}"

    if [ -f "$JSON_FILE" ]; then
        JSON_SIZE=$(ls -lh "$JSON_FILE" | awk '{print $5}')
        QA_COUNT=$($PYTHON -c "import json; print(len(json.load(open('${JSON_FILE}'))))" 2>/dev/null || echo "?")
        IMG_COUNT=$(ls "$IMG_DIR" 2>/dev/null | wc -l)
        echo "  📄 ${SPLIT}:"
        echo "     - JSON: ${JSON_FILE} (${JSON_SIZE}, ${QA_COUNT} 条 QA)"
        echo "     - 图像: ${IMG_DIR}/ (${IMG_COUNT} 张)"
    else
        echo "  ⚠️  ${SPLIT}: JSON 文件未生成"
    fi
done
echo ""
echo "🎉 全部完成！"
