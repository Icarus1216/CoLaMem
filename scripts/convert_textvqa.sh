#!/bin/bash
# TextVQA 数据集格式转换脚本 - 转换为 ColaMem 训练格式
# 调用 convert_textvqa_to_colamem.py 进行格式转换

set -e

echo "🚀 TextVQA -> ColaMem 格式转换启动"
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
TEXTVQA_DATA_DIR="/mnt/cephszjt/user_juntianzhang/ColaMem/data/textvqa"

# 输出目录（转换后的 ColaMem 格式文件存放位置）
OUTPUT_DIR="${TEXTVQA_DATA_DIR}"

# 要转换的 split 列表（train / validation / test）
# 注意：test 通常没有 answers，可能需要跳过
SPLITS="train validation"

# 是否将 OCR tokens 注入到 compress.text 中
USE_OCR=false

# 是否检查图像文件是否存在
CHECK_IMAGE=false

# compress.text 的默认内容（为空则不填写）
COMPRESS_TEXT=""

# ============================================================
# 显示配置信息
# ============================================================

echo "============================================================"
echo "📦 TextVQA -> ColaMem 格式转换配置"
echo "============================================================"
echo "📁 项目目录: ${PROJECT_DIR}"
echo "📥 TextVQA 数据目录: ${TEXTVQA_DATA_DIR}"
echo "📤 输出目录: ${OUTPUT_DIR}"
echo "📋 转换 splits: ${SPLITS}"
echo "📝 使用 OCR: ${USE_OCR}"
echo "🔍 检查图像: ${CHECK_IMAGE}"
if [ -n "$COMPRESS_TEXT" ]; then
    echo "📄 压缩文本: '${COMPRESS_TEXT}'"
else
    echo "📄 压缩文本: (空)"
fi
echo "============================================================"
echo ""

# 检查数据目录是否存在
if [ ! -d "$TEXTVQA_DATA_DIR" ]; then
    echo "❌ 错误：TextVQA 数据目录不存在: ${TEXTVQA_DATA_DIR}"
    exit 1
fi

# 检查转换脚本是否存在
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_textvqa_to_colamem.py"
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "❌ 错误：转换脚本不存在: ${CONVERT_SCRIPT}"
    exit 1
fi

echo "✅ 转换脚本: ${CONVERT_SCRIPT}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# ============================================================
# 逐个 split 进行转换
# ============================================================

TOTAL_SUCCESS=0
TOTAL_FAIL=0

for SPLIT in $SPLITS; do
    echo ""
    echo "============================================================"
    echo "🔄 开始转换 split: ${SPLIT}"
    echo "============================================================"
    
    # 输入文件
    INPUT_FILE="${TEXTVQA_DATA_DIR}/textvqa_${SPLIT}.json"
    
    # 输出文件
    OUTPUT_FILE="${OUTPUT_DIR}/colamem_textvqa_${SPLIT}.json"
    
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_FILE" ]; then
        echo "⚠️  跳过 ${SPLIT}: 输入文件不存在 ${INPUT_FILE}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        continue
    fi
    
    echo "📥 输入: ${INPUT_FILE}"
    echo "📤 输出: ${OUTPUT_FILE}"
    echo ""
    
    # 构建命令
    CMD="$PYTHON ${CONVERT_SCRIPT} \
        --input \"${INPUT_FILE}\" \
        --output \"${OUTPUT_FILE}\""
    
    # 如果使用 OCR
    if [ "$USE_OCR" = true ]; then
        CMD="$CMD --use_ocr"
    fi
    
    # 如果检查图像
    if [ "$CHECK_IMAGE" = true ]; then
        CMD="$CMD --check_image"
    fi
    
    # 如果指定了压缩文本
    if [ -n "$COMPRESS_TEXT" ]; then
        CMD="$CMD --compress_text \"${COMPRESS_TEXT}\""
    fi
    
    # 执行命令
    echo "执行命令："
    echo "$CMD"
    echo ""
    
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ ${SPLIT} 转换成功！"
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        echo ""
        echo "❌ ${SPLIT} 转换失败！"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
done

# ============================================================
# 最终统计
# ============================================================

echo ""
echo "============================================================"
echo "📊 TextVQA -> ColaMem 格式转换完成！"
echo "============================================================"
echo ""
echo "  ✅ 成功: ${TOTAL_SUCCESS} 个 split"
if [ $TOTAL_FAIL -gt 0 ]; then
    echo "  ❌ 失败: ${TOTAL_FAIL} 个 split"
fi
echo ""

# 显示输出文件信息
echo "📁 输出文件列表："
for SPLIT in $SPLITS; do
    OUTPUT_FILE="${OUTPUT_DIR}/colamem_textvqa_${SPLIT}.json"
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        RECORD_COUNT=$($PYTHON -c "import json; data=json.load(open('${OUTPUT_FILE}')); print(len(data))" 2>/dev/null || echo "?")
        echo "  📄 ${OUTPUT_FILE}"
        echo "     大小: ${FILE_SIZE}, 记录数: ${RECORD_COUNT}"
    fi
done
echo ""
echo "============================================================"
