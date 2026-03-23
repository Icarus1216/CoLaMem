#!/usr/bin/env python3
"""
TextVQA 数据格式转换为 ColaMem 训练格式脚本

输入格式 (TextVQA - 由 process_textvqa.py 生成的 JSON):
[
  {
    "image_id": "0054c91397f2fe05",
    "question_id": 0,
    "question": "what is the brand of phone?",
    "image_path": "/absolute/path/to/image.jpg",
    "image_width": 1024,
    "image_height": 730,
    "answers": ["nokia", "nokia", "nokia", ...],
    "ocr_tokens": ["MIA", "NOKIA"],
    "image_classes": ["Mobile phone", "Camera", ...],
    ...
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
      "question": "what is the brand of phone?",
      "answer": "nokia"
    }
  },
  ...
]

使用方法：
    python convert_textvqa_to_colamem.py \
        --input /path/to/textvqa_train.json \
        --output /path/to/colamem_textvqa_train.json

    # 将 OCR tokens 注入到 compress.text 中
    python convert_textvqa_to_colamem.py \
        --input /path/to/textvqa_train.json \
        --output /path/to/colamem_textvqa_train.json \
        --use_ocr
"""

import os
import json
import argparse
from collections import Counter
from typing import Dict, List, Any, Optional
from tqdm import tqdm


def majority_vote(answers: List[str]) -> str:
    """
    从多个标注者的答案中选取出现次数最多的答案（多数投票）
    
    Args:
        answers: 标注者答案列表
    
    Returns:
        出现次数最多的答案
    """
    if not answers:
        return ""
    counter = Counter(answers)
    # most_common(1) 返回 [(answer, count)]
    return counter.most_common(1)[0][0]


def convert_textvqa_to_colamem(
    input_path: str,
    output_path: str,
    compress_text: str = "",
    use_ocr: bool = False,
    check_image: bool = False,
) -> List[Dict[str, Any]]:
    """
    将 TextVQA 格式数据转换为 ColaMem 训练格式
    
    Args:
        input_path: TextVQA 格式的输入 JSON 文件路径
        output_path: ColaMem 格式的输出 JSON 文件路径
        compress_text: compress 部分的 text 字段内容（默认为空）
        use_ocr: 是否将 OCR tokens 注入到 compress.text 中
        check_image: 是否检查图像文件是否存在
    
    Returns:
        转换后的数据列表
    """
    print(f"[读取] 加载输入文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        textvqa_data = json.load(f)
    
    print(f"[信息] 输入数据量: {len(textvqa_data)} 条")
    
    colamem_data = []
    skipped = 0
    skip_reasons = Counter()
    
    for item in tqdm(textvqa_data, desc="转换数据"):
        try:
            # 获取图像路径（已是绝对路径）
            image_path = item.get('image_path', '')
            if not image_path:
                skipped += 1
                skip_reasons['缺少图像路径'] += 1
                continue
            
            # 可选：检查图像文件是否存在
            if check_image and not os.path.exists(image_path):
                skipped += 1
                skip_reasons['图像文件不存在'] += 1
                continue
            
            # 获取问题
            question = item.get('question', '').strip()
            if not question:
                skipped += 1
                skip_reasons['缺少问题'] += 1
                continue
            
            # 从多个标注者答案中选取最佳答案（多数投票）
            answers = item.get('answers', [])
            if not answers:
                skipped += 1
                skip_reasons['缺少答案'] += 1
                continue
            
            answer = majority_vote(answers)
            if not answer:
                skipped += 1
                skip_reasons['答案为空'] += 1
                continue
            
            # 构建 compress.text
            text = compress_text
            if use_ocr:
                ocr_tokens = item.get('ocr_tokens', [])
                if ocr_tokens:
                    ocr_str = ", ".join(ocr_tokens)
                    text = f"OCR: {ocr_str}" if not text else f"{text}\nOCR: {ocr_str}"
            
            # 构建 ColaMem 格式
            colamem_item = {
                "compress": {
                    "text": text,
                    "images": [image_path]
                },
                "solve": {
                    "question": question,
                    "answer": answer
                }
            }
            
            colamem_data.append(colamem_item)
            
        except Exception as e:
            print(f"\n[警告] 处理条目时出错 (question_id={item.get('question_id', '?')}): {e}")
            skipped += 1
            skip_reasons['处理异常'] += 1
            continue
    
    print(f"\n[统计]")
    print(f"  - 成功转换: {len(colamem_data)} 条")
    print(f"  - 跳过: {skipped} 条")
    if skip_reasons:
        print(f"  - 跳过原因:")
        for reason, count in skip_reasons.most_common():
            print(f"      {reason}: {count} 条")
    
    # 答案分布统计
    if colamem_data:
        all_answers = [d['solve']['answer'] for d in colamem_data]
        unique_answers = len(set(all_answers))
        print(f"\n[答案统计]")
        print(f"  - 唯一答案数: {unique_answers}")
        top_answers = Counter(all_answers).most_common(10)
        print(f"  - Top 10 答案:")
        for ans, cnt in top_answers:
            print(f"      '{ans}': {cnt} 次 ({cnt/len(colamem_data)*100:.1f}%)")
    
    # 保存输出文件
    print(f"\n[保存] 写入输出文件: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(colamem_data, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_path)
    print(f"[保存] 文件大小: {file_size / 1024 / 1024:.1f} MB")
    
    return colamem_data


def main():
    parser = argparse.ArgumentParser(
        description='将 TextVQA 数据格式转换为 ColaMem 训练格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
    # 基本用法
    python convert_textvqa_to_colamem.py \\
        --input /path/to/textvqa_train.json \\
        --output /path/to/colamem_textvqa_train.json

    # 将 OCR tokens 注入到 compress.text 中
    python convert_textvqa_to_colamem.py \\
        --input /path/to/textvqa_train.json \\
        --output /path/to/colamem_textvqa_train.json \\
        --use_ocr

    # 使用当前 TextVQA 处理后的数据
    python convert_textvqa_to_colamem.py \\
        --input /mnt/cephszjt/user_juntianzhang/ColaMem/data/textvqa/textvqa_train.json \\
        --output /mnt/cephszjt/user_juntianzhang/ColaMem/data/textvqa/colamem_textvqa_train.json

    # 同时检查图像是否存在
    python convert_textvqa_to_colamem.py \\
        --input /mnt/cephszjt/user_juntianzhang/ColaMem/data/textvqa/textvqa_train.json \\
        --output /mnt/cephszjt/user_juntianzhang/ColaMem/data/textvqa/colamem_textvqa_train.json \\
        --check_image
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入的 TextVQA 格式 JSON 文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出的 ColaMem 格式 JSON 文件路径'
    )
    parser.add_argument(
        '--compress_text',
        type=str,
        default='',
        help='compress 部分的 text 字段内容（默认为空）'
    )
    parser.add_argument(
        '--use_ocr',
        action='store_true',
        help='将 OCR tokens 注入到 compress.text 中（格式：OCR: token1, token2, ...）'
    )
    parser.add_argument(
        '--check_image',
        action='store_true',
        help='检查图像文件是否存在，不存在则跳过'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TextVQA -> ColaMem 格式转换脚本")
    print("=" * 60)
    print(f"\n[配置]")
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  压缩文本: '{args.compress_text}'")
    print(f"  使用 OCR: {args.use_ocr}")
    print(f"  检查图像: {args.check_image}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"\n[错误] 输入文件不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行转换
    result = convert_textvqa_to_colamem(
        input_path=args.input,
        output_path=args.output,
        compress_text=args.compress_text,
        use_ocr=args.use_ocr,
        check_image=args.check_image,
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
            if item['compress']['text']:
                text = item['compress']['text']
                print(f"文本: {text[:100]}..." if len(text) > 100 else f"文本: {text}")
            print(f"问题: {item['solve']['question']}")
            ans = item['solve']['answer']
            print(f"答案: {ans[:100]}..." if len(ans) > 100 else f"答案: {ans}")
    else:
        print("\n[错误] 转换失败，没有有效数据")


if __name__ == '__main__':
    main()
