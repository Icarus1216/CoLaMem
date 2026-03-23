#!/bin/bash
# ColaMem 推理脚本 - 适配 Gemini 平台
# 用于测试训练好的模型效果

set -e

echo "🚀 ColaMem 推理启动 (Gemini 平台)"
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
MODEL_PATH="/mnt/cephszjt/user_juntianzhang/ColaMem_lora/outputs/colamem_phase0.1_lora/model"
BASE_MODEL="/mnt/wfs/mmchongqingssdwfssz/project_luban_infra/luban_infra/model_factory/for_lucas_Qwen3-VL-8B-Instruct_export"
VAL_JSON="/mnt/cephszjt/user_juntianzhang/ColaMem/data/ShareGPT4V/colamem_val.json"
# 注意：验证集中图片路径已是绝对路径，IMAGE_DIR 仅在图片为相对路径时使用
IMAGE_DIR=""
NUM_SAMPLES=20
OUTPUT_FILE="/mnt/cephszjt/user_juntianzhang/ColaMem_lora/outputs/colamem_phase0.1_lora/inference_results.json"

echo "============================================================"
echo "📦 模型路径: ${MODEL_PATH}"
echo "🔄 基础模型: ${BASE_MODEL}"
echo "📄 测试数据: ${VAL_JSON}"
echo "🖼️ 图像目录: ${IMAGE_DIR}"
echo "🔢 测试样本数: ${NUM_SAMPLES}"
echo "📝 结果输出: ${OUTPUT_FILE}"
echo "============================================================"
echo ""

# 运行推理
echo "开始推理..."
echo ""

# 构建推理命令
INFER_CMD="$PYTHON scripts/inference_debug.py \
    --checkpoint \"${MODEL_PATH}\" \
    --base_model \"${BASE_MODEL}\" \
    --val_json \"${VAL_JSON}\" \
    --num_samples ${NUM_SAMPLES} \
    --max_new_tokens 256 \
    --temperature 0 \
    --output \"${OUTPUT_FILE}\" \
    --device cuda"

# 如果 IMAGE_DIR 非空，则传递 --image_dir 参数
if [ -n "${IMAGE_DIR}" ]; then
    INFER_CMD="${INFER_CMD} --image_dir \"${IMAGE_DIR}\""
fi

eval ${INFER_CMD}

echo ""
echo "✅ 推理完成！"
