#!/usr/bin/env python3
"""
GQA 数据格式转换为 ColaMem 训练格式脚本

输入格式 (GQA):
[
  {
    "id": "02930152",
    "image": "images/2354786.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nIs the sky dark?"},
      {"from": "gpt", "value": "yes"}
    ]
  },
  ...
]

输出格式 (ColaMem):
[
  {
    "compress": {
      "text": "",
      "images": ["/absolute/path/to/image.jpg"]
    },
    "solve": {
      "question": "Is the sky dark?",
      "answer": "yes"
    }
  },
  ...
]

使用方法：
    python convert_gqa_to_colamem.py \
        --input /path/to/gqa_data.json \
        --output /path/to/colamem_data.json \
        --image_base_dir /path/to/gqa/processed
"""

import os
import json
import argparse
from typing import Dict, List, Any
from tqdm import tqdm


def convert_gqa_to_colamem(
    input_path: str,
    output_path: str,
    image_base_dir: str,
    compress_text: str = ""
) -> List[Dict[str, Any]]:
    """
    将 GQA 格式数据转换为 ColaMem 训练格式
    
    Args:
        input_path: GQA 格式的输入 JSON 文件路径
        output_path: ColaMem 格式的输出 JSON 文件路径
        image_base_dir: 图像文件的基础目录（用于构建绝对路径）
        compress_text: compress 部分的 text 字段内容（默认为空）
    
    Returns:
        转换后的数据列表
    """
    print(f"[读取] 加载输入文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        gqa_data = json.load(f)
    
    print(f"[信息] 输入数据量: {len(gqa_data)} 条")
    
    colamem_data = []
    skipped = 0
    
    for item in tqdm(gqa_data, desc="转换数据"):
        try:
            # 获取图像路径
            image_rel_path = item.get('image', '')
            if not image_rel_path:
                skipped += 1
                continue
            
            # 构建绝对路径
            image_abs_path = os.path.join(image_base_dir, image_rel_path)
            
            # 从 conversations 中提取问题和答案
            conversations = item.get('conversations', [])
            if len(conversations) < 2:
                skipped += 1
                continue
            
            # 提取问题（移除 <image>\n 前缀）
            human_msg = conversations[0].get('value', '')
            question = human_msg.replace('<image>\n', '').replace('<image>', '').strip()
            
            # 提取答案
            answer = conversations[1].get('value', '').strip()
            
            if not question or not answer:
                skipped += 1
                continue
            
            # 构建 ColaMem 格式
            colamem_item = {
                "compress": {
                    "text": compress_text,
                    "images": [image_abs_path]
                },
                "solve": {
                    "question": question,
                    "answer": answer
                }
            }
            
            colamem_data.append(colamem_item)
            
        except Exception as e:
            print(f"\n[警告] 处理条目时出错: {e}")
            skipped += 1
            continue
    
    print(f"\n[统计]")
    print(f"  - 成功转换: {len(colamem_data)} 条")
    print(f"  - 跳过: {skipped} 条")
    
    # 保存输出文件
    print(f"\n[保存] 写入输出文件: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(colamem_data, f, ensure_ascii=False, indent=2)
    
    return colamem_data


def main():
    parser = argparse.ArgumentParser(
        description='将 GQA 数据格式转换为 ColaMem 训练格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
    # 基本用法
    python convert_gqa_to_colamem.py \\
        --input /path/to/gqa_data.json \\
        --output /path/to/colamem_data.json \\
        --image_base_dir /path/to/gqa/processed

    # 使用当前 GQA 处理后的数据
    python convert_gqa_to_colamem.py \\
        --input /mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/processed/gqa_train_balanced.json \\
        --output /mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/processed/colamem_gqa_train.json \\
        --image_base_dir /mnt/cephszjt/user_juntianzhang/ColaMem/data/GQA/processed
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入的 GQA 格式 JSON 文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出的 ColaMem 格式 JSON 文件路径'
    )
    parser.add_argument(
        '--image_base_dir',
        type=str,
        required=True,
        help='图像文件的基础目录（用于构建绝对路径）'
    )
    parser.add_argument(
        '--compress_text',
        type=str,
        default='',
        help='compress 部分的 text 字段内容（默认为空）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GQA -> ColaMem 格式转换脚本")
    print("=" * 60)
    print(f"\n[配置]")
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  图像基础目录: {args.image_base_dir}")
    print(f"  压缩文本: '{args.compress_text}'")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"\n[错误] 输入文件不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行转换
    result = convert_gqa_to_colamem(
        input_path=args.input,
        output_path=args.output,
        image_base_dir=args.image_base_dir,
        compress_text=args.compress_text
    )
    
    if result:
        print(f"\n" + "=" * 60)
        print("转换完成！")
        print("=" * 60)
        
        # 显示示例数据
        print(f"\n[示例数据] (前 2 条)")
        for i, item in enumerate(result[:2]):
            print(f"\n--- 样本 {i+1} ---")
            print(f"图像: {item['compress']['images'][0]}")
            print(f"问题: {item['solve']['question']}")
            ans = item['solve']['answer']
            print(f"答案: {ans[:100]}..." if len(ans) > 100 else f"答案: {ans}")
    else:
        print("\n[错误] 转换失败，没有有效数据")


if __name__ == '__main__':
    main()
