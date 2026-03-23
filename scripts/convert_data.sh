#!/bin/bash
# 一键转换 ShareGPT4V 数据到 ColaMem 训练格式

set -e

echo "========================================"
echo "ShareGPT4V → ColaMem 训练格式转换"
echo "========================================"

# 配置
INPUT_JSON="data/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json"
OUTPUT_DIR="data/ShareGPT4V"
IMAGES_BASE_DIR="data/ShareGPT4V/images"

# 检查输入文件
if [ ! -f "$INPUT_JSON" ]; then
    echo "❌ 错误: 找不到输入文件 $INPUT_JSON"
    exit 1
fi

echo ""
echo "配置:"
echo "  输入文件: $INPUT_JSON"
echo "  输出目录: $OUTPUT_DIR"
echo "  图像目录: $IMAGES_BASE_DIR"
echo ""

# 运行转换
python3 scripts/convert_sharegpt4v_to_colamem.py \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --images_base_dir "$IMAGES_BASE_DIR" \
    --filter_sources coco

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 转换完成！"
    echo "========================================"
    echo ""
    echo "生成的文件:"
    echo "  训练集: $OUTPUT_DIR/colamem_train.json"
    echo "  验证集: $OUTPUT_DIR/colamem_val.json"
    echo "  完整数据: $OUTPUT_DIR/colamem_full.json"
    echo ""
    
    # 显示第一个样本
    echo "数据格式示例:"
    python3 -c "import json; data=json.load(open('$OUTPUT_DIR/colamem_train.json')); print(json.dumps(data[0], indent=2, ensure_ascii=False))" 2>/dev/null || echo "无法显示示例"
    
    echo ""
    echo "下一步:"
    echo "1. 如果图像缺失，下载 COCO 2017 训练集:"
    echo "   bash scripts/download_coco.sh"
    echo ""
    echo "2. 开始训练:"
    echo "   修改 configs/train_phase1.yaml 中的 data_path:"
    echo "   data_path: \"$OUTPUT_DIR/colamem_train.json\""
    echo "   然后运行: bash scripts/train.sh"
else
    echo ""
    echo "❌ 转换失败"
    exit 1
fi
