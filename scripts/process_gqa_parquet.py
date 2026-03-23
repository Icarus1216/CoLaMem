#!/usr/bin/env python3
"""
GQA 数据集 Parquet 文件处理脚本 (分离存储格式)

功能：
1. 从 instructions parquet 文件中读取 QA 数据（包含 id, imageId, question, answer）
2. 从 images parquet 文件中读取图像数据（包含 id, image）
3. 通过 imageId 关联两者，生成完整的训练数据

使用方法：
    python process_gqa_parquet.py \
        --instructions_dir /path/to/train_balanced_instructions \
        --images_dir /path/to/train_balanced_images \
        --output_dir /path/to/output

输出结构：
    output_dir/
    ├── images/              # 保存的图像文件
    │   ├── 2354786.jpg
    │   ├── 2354787.jpg
    │   └── ...
    └── gqa_data.json        # 包含图像路径和QA的JSON文件

JSON 格式示例：
    [
        {
            "id": "unique_id",
            "image": "images/2354786.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\n问题内容"},
                {"from": "gpt", "value": "答案内容"}
            ]
        },
        ...
    ]
"""

import os
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import BytesIO
from tqdm import tqdm

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def to_python_native(obj):
    """递归将 numpy 类型转为 Python 原生类型，确保 JSON 可序列化"""
    if HAS_NUMPY:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [to_python_native(item) for item in obj.tolist()]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
    if isinstance(obj, (list, tuple)):
        return [to_python_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    else:
        return obj


def _save_json(data, json_path):
    """安全保存 JSON 文件（先写临时文件，再原子替换，防止写入中途崩溃导致文件损坏）"""
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)  # 原子替换

# 尝试导入依赖库
try:
    import pandas as pd
except ImportError:
    print("错误：需要安装 pandas 库")
    print("请运行：pip install pandas")
    exit(1)

try:
    import pyarrow.parquet as pq
except ImportError:
    print("错误：需要安装 pyarrow 库")
    print("请运行：pip install pyarrow")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("错误：需要安装 Pillow 库")
    print("请运行：pip install Pillow")
    exit(1)


def extract_image_from_row(row: Dict[str, Any], image_column: str) -> Optional[bytes]:
    """
    从行数据中提取图像字节
    
    支持多种格式：
    1. 直接的 bytes 数据
    2. dict 格式 {"bytes": ..., "path": ...}
    3. PIL Image 对象
    """
    image_data = row.get(image_column)
    
    if image_data is None:
        return None
    
    # 情况 1：直接是 bytes
    if isinstance(image_data, bytes):
        return image_data
    
    # 情况 2：dict 格式 (Hugging Face 常用格式)
    if isinstance(image_data, dict):
        if 'bytes' in image_data:
            return image_data['bytes']
        elif 'path' in image_data:
            try:
                with open(image_data['path'], 'rb') as f:
                    return f.read()
            except Exception:
                return None
    
    # 情况 3：PIL Image 对象
    if hasattr(image_data, 'tobytes') or str(type(image_data)).find('PIL') != -1:
        try:
            buffer = BytesIO()
            image_data.save(buffer, format='JPEG')
            return buffer.getvalue()
        except Exception:
            return None
    
    return None


def load_all_parquet_files(directory: str) -> pd.DataFrame:
    """
    加载目录中的所有 Parquet 文件并合并
    """
    parquet_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        if os.path.isfile(directory) and directory.endswith('.parquet'):
            parquet_files = [directory]
        else:
            return pd.DataFrame()
    
    print(f"[信息] 找到 {len(parquet_files)} 个 Parquet 文件")
    
    # 读取并合并所有文件
    dfs = []
    for f in tqdm(parquet_files, desc="加载 Parquet 文件"):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"[警告] 无法读取 {f}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def process_gqa(
    instructions_dir: str,
    images_dir: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    image_format: str = 'jpg',
    use_full_answer: bool = False
) -> List[Dict[str, Any]]:
    """
    处理 GQA 数据集（分离存储格式）
    
    Args:
        instructions_dir: QA 数据目录（包含 question, answer, imageId 等）
        images_dir: 图像数据目录（包含 id, image）
        output_dir: 输出目录
        max_samples: 最大处理样本数
        image_format: 输出图像格式
        use_full_answer: 是否使用 fullAnswer 而不是 answer
    
    Returns:
        处理后的数据列表
    """
    print("\n" + "=" * 60)
    print("加载 QA 数据 (Instructions)")
    print("=" * 60)
    
    # 加载 QA 数据
    qa_df = load_all_parquet_files(instructions_dir)
    if qa_df.empty:
        print(f"[错误] 在 {instructions_dir} 中未找到有效数据")
        return []
    
    print(f"[信息] QA 数据总量: {len(qa_df)} 条")
    print(f"[信息] QA 数据列: {qa_df.columns.tolist()}")
    
    # 限制样本数
    if max_samples and len(qa_df) > max_samples:
        qa_df = qa_df.head(max_samples)
        print(f"[信息] 限制处理数量为 {max_samples} 条")
    
    print("\n" + "=" * 60)
    print("加载图像数据 (Images)")
    print("=" * 60)
    
    # 加载图像数据
    images_df = load_all_parquet_files(images_dir)
    if images_df.empty:
        print(f"[错误] 在 {images_dir} 中未找到有效数据")
        return []
    
    print(f"[信息] 图像数据总量: {len(images_df)} 条")
    print(f"[信息] 图像数据列: {images_df.columns.tolist()}")
    
    # 检测列名
    qa_id_col = 'id' if 'id' in qa_df.columns else None
    image_id_col = 'imageId' if 'imageId' in qa_df.columns else None
    question_col = 'question' if 'question' in qa_df.columns else None
    answer_col = 'fullAnswer' if (use_full_answer and 'fullAnswer' in qa_df.columns) else 'answer' if 'answer' in qa_df.columns else None
    
    img_id_col = 'id' if 'id' in images_df.columns else None
    img_data_col = 'image' if 'image' in images_df.columns else None
    
    print(f"\n[配置] 列映射:")
    print(f"  QA 数据:")
    print(f"    - ID 列: {qa_id_col}")
    print(f"    - 图像ID 列: {image_id_col}")
    print(f"    - 问题列: {question_col}")
    print(f"    - 答案列: {answer_col}")
    print(f"  图像数据:")
    print(f"    - ID 列: {img_id_col}")
    print(f"    - 图像列: {img_data_col}")
    
    if not all([image_id_col, question_col, answer_col, img_id_col, img_data_col]):
        print("[错误] 缺少必要的列，请检查数据格式")
        return []
    
    # 构建图像 ID 到图像数据的映射
    print("\n[处理] 构建图像索引...")
    image_map = {}
    for _, row in tqdm(images_df.iterrows(), total=len(images_df), desc="索引图像"):
        img_id = str(row[img_id_col])
        image_map[img_id] = row[img_data_col]
    
    print(f"[信息] 图像索引大小: {len(image_map)}")
    
    # 创建输出目录
    images_out_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_out_dir, exist_ok=True)
    
    # 处理数据
    print("\n[处理] 生成训练数据...")
    results = []
    saved_images = set()  # 记录已保存的图像，避免重复保存
    skipped = 0
    no_image = 0
    
    for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="处理 QA 数据"):
        qa_id = str(row[qa_id_col]) if qa_id_col else str(idx)
        image_id = str(row[image_id_col])
        question = str(row[question_col]) if row[question_col] else ''
        answer = str(row[answer_col]) if row[answer_col] else ''
        
        # 跳过空问题或答案
        if not question.strip() or not answer.strip():
            skipped += 1
            continue
        
        # 查找对应图像
        if image_id not in image_map:
            no_image += 1
            continue
        
        # 图像文件名使用 imageId，避免重复保存
        image_filename = f"{image_id}.{image_format}"
        image_path = f"images/{image_filename}"
        image_full_path = os.path.join(images_out_dir, image_filename)
        
        # 如果图像尚未保存，则保存（支持磁盘级去重，跳过已存在的文件）
        if image_id not in saved_images:
            if os.path.exists(image_full_path):
                # 磁盘上已存在，直接跳过保存，标记为已处理
                saved_images.add(image_id)
            else:
                image_data = image_map[image_id]
                image_bytes = extract_image_from_row({'image': image_data}, 'image')
                
                if image_bytes:
                    try:
                        img = Image.open(BytesIO(image_bytes))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(image_full_path, quality=95)
                        saved_images.add(image_id)
                    except Exception as e:
                        print(f"\n[警告] 保存图像失败 (imageId={image_id}): {e}")
                        no_image += 1
                        continue
                else:
                    no_image += 1
                    continue
        
        # 构建输出数据（使用 to_python_native 确保 JSON 可序列化）
        output_item = to_python_native({
            "id": qa_id,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        })
        
        results.append(output_item)
    
    print(f"\n[统计]")
    print(f"  - 成功处理: {len(results)} 条")
    print(f"  - 保存图像: {len(saved_images)} 张（去重后）")
    print(f"  - 跳过（空内容）: {skipped} 条")
    print(f"  - 跳过（无图像）: {no_image} 条")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='GQA 数据集 Parquet 文件处理脚本（分离存储格式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
    python process_gqa_parquet.py \\
        --instructions_dir /path/to/train_balanced_instructions \\
        --images_dir /path/to/train_balanced_images \\
        --output_dir /path/to/output
    
    # 限制处理数量
    python process_gqa_parquet.py \\
        --instructions_dir /path/to/train_balanced_instructions \\
        --images_dir /path/to/train_balanced_images \\
        --output_dir /path/to/output \\
        --max_samples 100000
    
    # 使用完整答案（fullAnswer）
    python process_gqa_parquet.py \\
        --instructions_dir /path/to/train_balanced_instructions \\
        --images_dir /path/to/train_balanced_images \\
        --output_dir /path/to/output \\
        --use_full_answer
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--instructions_dir',
        type=str,
        required=True,
        help='QA 数据目录（包含 question, answer, imageId 等列的 parquet 文件）'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='图像数据目录（包含 id, image 列的 parquet 文件）'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='输出目录（将创建 images/ 子目录和 JSON 文件）'
    )
    
    # 可选参数
    parser.add_argument(
        '--max_samples', '-n',
        type=int,
        default=None,
        help='最大处理样本数（默认：全部处理）'
    )
    parser.add_argument(
        '--image_format',
        type=str,
        default='jpg',
        choices=['jpg', 'png', 'webp'],
        help='输出图像格式（默认：jpg）'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='gqa_data.json',
        help='输出 JSON 文件名（默认：gqa_data.json）'
    )
    parser.add_argument(
        '--use_full_answer',
        action='store_true',
        help='使用 fullAnswer 列而不是 answer 列'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GQA 数据集 Parquet 文件处理脚本")
    print("=" * 60)
    print(f"\n[配置]")
    print(f"  QA 数据目录: {args.instructions_dir}")
    print(f"  图像数据目录: {args.images_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  最大样本数: {args.max_samples or '全部'}")
    print(f"  图像格式: {args.image_format}")
    print(f"  使用完整答案: {args.use_full_answer}")
    print(f"  输出 JSON: {args.output_json}")
    
    json_path = os.path.join(args.output_dir, args.output_json)
    try:
        results = process_gqa(
            instructions_dir=args.instructions_dir,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            image_format=args.image_format,
            use_full_answer=args.use_full_answer
        )
    except Exception as e:
        print(f"\n❌ 处理过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        # 尝试保存已有的中间结果（如果 results 变量已存在于调用栈外则无法捕获，此处仅做日志提示）
        print(f"   请检查输出目录 {args.output_dir} 中是否有已保存的图像文件。")
        return
    
    if not results:
        print("\n[错误] 没有处理到有效数据！")
        return
    
    # 保存 JSON 文件（使用原子写入）
    print(f"\n[保存] 写入 JSON 文件: {json_path}")
    _save_json(results, json_path)
    
    print(f"\n" + "=" * 60)
    print(f"处理完成！")
    print("=" * 60)
    print(f"\n[输出统计]")
    print(f"  - 总样本数: {len(results)}")
    print(f"  - 图像目录: {os.path.join(args.output_dir, 'images')}")
    print(f"  - JSON 文件: {json_path}")
    
    # 显示示例数据
    print(f"\n[示例数据] (前 2 条)")
    for i, item in enumerate(results[:2]):
        print(f"\n--- 样本 {i+1} ---")
        print(f"ID: {item['id']}")
        print(f"图像: {item['image']}")
        q_text = item['conversations'][0]['value']
        a_text = item['conversations'][1]['value']
        print(f"问题: {q_text[:100]}..." if len(q_text) > 100 else f"问题: {q_text}")
        print(f"答案: {a_text[:100]}..." if len(a_text) > 100 else f"答案: {a_text}")


if __name__ == '__main__':
    main()
