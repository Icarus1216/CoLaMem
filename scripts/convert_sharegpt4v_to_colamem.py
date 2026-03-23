#!/usr/bin/env python3
"""
将 ShareGPT4V 数据转换为 ColaMem 训练格式

ShareGPT4V 格式:
{
    "id": "xxx",
    "image": "coco/train2017/000000000009.jpg",
    "conversations": [
        {"from": "human", "value": "What do you see?<image>"},
        {"from": "gpt", "value": "详细描述..."}
    ]
}

ColaMem 训练格式:
{
    "compress": {
        "text": "",
        "images": ["/absolute/path/to/image.jpg"]
    },
    "solve": {
        "question": "What do you see?",
        "answer": "详细描述..."
    }
}
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

def clean_text(text):
    """清理文本，移除 <image> 标签"""
    return text.replace('<image>', '').strip()

def convert_sharegpt4v_to_colamem(
    input_json,
    output_dir,
    images_base_dir,
    filter_sources=None,
    max_samples=None,
    train_ratio=0.9
):
    """
    转换 ShareGPT4V 格式到 ColaMem 训练格式
    """
    
    print(f"\n{'='*60}")
    print(f"ShareGPT4V → ColaMem 训练格式转换")
    print(f"{'='*60}\n")
    
    # 读取数据
    print(f"读取数据: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"原始样本数: {len(data)}")
    
    # 转换数据
    colamem_data = []
    skipped = 0
    missing_images = 0
    
    images_base_path = Path(images_base_dir)
    
    print(f"\n转换数据...")
    for item in tqdm(data):
        # 提取图像路径
        image_rel_path = item.get('image', '')
        
        # 判断是绝对路径还是相对路径
        if image_rel_path.startswith('/'):
            # 已经是绝对路径，直接使用
            image_path = Path(image_rel_path)
            # 从绝对路径中提取来源（如 coco）
            if filter_sources:
                # 尝试从路径中提取来源，如 /data/.../coco/train2017/xxx.jpg -> coco
                path_parts = image_rel_path.split('/')
                source = None
                for part in path_parts:
                    if part in filter_sources:
                        source = part
                        break
                if not source:
                    skipped += 1
                    continue
        else:
            # 相对路径，需要拼接
            # 过滤图像来源
            if filter_sources:
                source = image_rel_path.split('/')[0] if '/' in image_rel_path else ''
                if source not in filter_sources:
                    skipped += 1
                    continue
            
            # 构建完整图像路径
            image_path = images_base_path / image_rel_path
        
        # 检查图像是否存在
        if not image_path.exists():
            missing_images += 1
        
        # 提取对话内容
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            skipped += 1
            continue
        
        # 提取 human 的问题和 gpt 的回答
        question = ""
        answer = ""
        
        for conv in conversations:
            if conv.get('from') == 'human':
                question = clean_text(conv.get('value', ''))
            elif conv.get('from') == 'gpt':
                answer = clean_text(conv.get('value', ''))
        
        if not question or not answer:
            skipped += 1
            continue
        
        # 构建 ColaMem 格式
        colamem_item = {
            "compress": {
                "text": "",  # compress 部分的 text 为空
                "images": [str(image_path.absolute())]
            },
            "solve": {
                "question": question,
                "answer": answer
            }
        }
        
        colamem_data.append(colamem_item)
        
        # 限制样本数
        if max_samples and len(colamem_data) >= max_samples:
            break
    
    print(f"\n转换完成:")
    print(f"  有效样本: {len(colamem_data)}")
    print(f"  跳过样本: {skipped}")
    print(f"  缺失图像: {missing_images}")
    
    if missing_images > 0:
        print(f"\n⚠️  警告: {missing_images} 个图像文件不存在")
        print(f"  请下载 COCO 2017 训练集到: {images_base_path}")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(colamem_data)
    
    # 划分训练集和验证集
    split_idx = int(len(colamem_data) * train_ratio)
    train_data = colamem_data[:split_idx]
    val_data = colamem_data[split_idx:]
    
    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / "colamem_train.json"
    val_path = output_path / "colamem_val.json"
    full_path = output_path / "colamem_full.json"
    
    # 保存完整数据
    print(f"\n保存数据...")
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(colamem_data, f, indent=2, ensure_ascii=False)
    
    # 保存训练集
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # 保存验证集
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ 数据保存完成!")
    print(f"{'='*60}")
    print(f"\n输出文件:")
    print(f"  完整数据: {full_path} ({len(colamem_data)} 样本)")
    print(f"  训练集: {train_path} ({len(train_data)} 样本)")
    print(f"  验证集: {val_path} ({len(val_data)} 样本)")
    
    # 显示示例
    if colamem_data:
        print(f"\n数据格式示例:")
        print(json.dumps(colamem_data[0], indent=2, ensure_ascii=False))
    
    return train_path, val_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="转换 ShareGPT4V 数据到 ColaMem 训练格式")
    parser.add_argument("--input_json", type=str, required=True,
                        help="ShareGPT4V 标注文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--images_base_dir", type=str, required=True,
                        help="图像基础目录（包含 coco/train2017 等）")
    parser.add_argument("--filter_sources", type=str, nargs="+",
                        help="只使用指定来源的图像，如: coco sam")
    parser.add_argument("--max_samples", type=int,
                        help="最大样本数（用于测试）")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="训练集比例（默认 0.9）")
    
    args = parser.parse_args()
    
    train_path, val_path = convert_sharegpt4v_to_colamem(
        input_json=args.input_json,
        output_dir=args.output_dir,
        images_base_dir=args.images_base_dir,
        filter_sources=args.filter_sources,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio
    )
    
    print(f"\n{'='*60}")
    print(f"下一步:")
    print(f"{'='*60}")
    print(f"\n1. 如果图像缺失，下载 COCO 2017 训练集:")
    print(f"   mkdir -p {args.images_base_dir}/coco")
    print(f"   cd {args.images_base_dir}/coco")
    print(f"   wget http://images.cocodataset.org/zips/train2017.zip")
    print(f"   unzip train2017.zip")
    
    print(f"\n2. 验证数据格式:")
    print(f"   python3 -c \"import json; print(json.dumps(json.load(open('{train_path}'))[0], indent=2, ensure_ascii=False))\"")
    
    print(f"\n3. 开始训练:")
    print(f"   修改 scripts/train.sh 或 configs/train_phase1.yaml 中的配置:")
    print(f"   DATA_PATH=\"{train_path}\"")
    print(f"   然后运行: bash scripts/train.sh")

if __name__ == "__main__":
    main()
