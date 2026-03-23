"""
处理 TextVQA 数据集的 parquet 文件：
1. 将图像提取并保存到 images/ 目录
2. 将 QA 元数据（含图像路径）保存到 JSON 文件
"""

import os
import json
import glob
import io
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def to_python_native(obj):
    """递归将 numpy 类型转为 Python 原生类型，确保 JSON 可序列化"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [to_python_native(item) for item in obj.tolist()]
    elif isinstance(obj, (list, tuple)):
        return [to_python_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    else:
        return obj

# ============ 配置（绝对路径，对脚本位置不敏感） ============
TEXTVQA_ROOT = "/mnt/cephszjt/user_juntianzhang/ColaMem/data/textvqa"
PARQUET_DIR = os.path.join(TEXTVQA_ROOT, "data")
IMAGES_DIR = os.path.join(TEXTVQA_ROOT, "images")
OUTPUT_DIR = TEXTVQA_ROOT  # JSON 输出目录

# 每个 split 对应的 parquet 文件模式
SPLITS = ["train", "validation", "test"]


def save_image(image_data, image_path):
    """将图像数据保存到指定路径"""
    if os.path.exists(image_path):
        return  # 同一张图可能在不同 QA 中重复出现，跳过已保存的

    if isinstance(image_data, dict):
        # HuggingFace datasets 格式：{'bytes': b'...', 'path': ...}
        img_bytes = image_data.get("bytes", None)
        if img_bytes is not None:
            img = Image.open(io.BytesIO(img_bytes))
        else:
            # 可能直接是 path
            img_path = image_data.get("path", None)
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
            else:
                print(f"  [警告] 无法解析图像数据: {list(image_data.keys())}")
                return
    elif isinstance(image_data, Image.Image):
        img = image_data
    elif isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    else:
        print(f"  [警告] 未知图像类型: {type(image_data)}")
        return

    # 统一转为 RGB 后保存为 JPEG
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(image_path, "JPEG", quality=95)


def _save_json(data, json_path):
    """安全保存 JSON 文件（先写临时文件，再原子替换，防止写入中途崩溃导致文件损坏）"""
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)  # 原子替换


def process_split(split_name):
    """处理单个 split (train/validation/test)"""
    pattern = os.path.join(PARQUET_DIR, f"{split_name}-*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"[跳过] 未找到 {split_name} 的 parquet 文件")
        return

    print(f"\n{'='*60}")
    print(f"处理 {split_name} split ({len(parquet_files)} 个文件)")
    print(f"{'='*60}")

    # 为每个 split 创建图像子目录
    split_images_dir = os.path.join(IMAGES_DIR, split_name)
    os.makedirs(split_images_dir, exist_ok=True)

    metadata_list = []
    saved_images = set()  # 记录已保存的 image_id，避免重复保存

    for pf in parquet_files:
        print(f"\n读取文件: {os.path.basename(pf)}")
        df = pd.read_parquet(pf)
        print(f"  行数: {len(df)}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理中"):
            image_id = str(row["image_id"])
            image_filename = f"{image_id}.jpg"
            image_abs_path = os.path.join(IMAGES_DIR, split_name, image_filename)

            # 保存图像（同一 image_id 只保存一次）
            if image_id not in saved_images:
                if "image" in row and row["image"] is not None:
                    save_image(row["image"], image_abs_path)
                    saved_images.add(image_id)

            # 构建元数据条目
            meta_entry = {
                "image_id": image_id,
                "question_id": int(row["question_id"]) if pd.notna(row.get("question_id")) else None,
                "question": str(row["question"]) if pd.notna(row.get("question")) else "",
                "image_path": image_abs_path,
                "image_width": int(row["image_width"]) if pd.notna(row.get("image_width")) else None,
                "image_height": int(row["image_height"]) if pd.notna(row.get("image_height")) else None,
            }

            # 可选字段（使用 to_python_native 确保 JSON 可序列化）
            if "answers" in row and row["answers"] is not None:
                meta_entry["answers"] = to_python_native(row["answers"])

            if "question_tokens" in row and row["question_tokens"] is not None:
                meta_entry["question_tokens"] = to_python_native(row["question_tokens"])

            if "ocr_tokens" in row and row["ocr_tokens"] is not None:
                meta_entry["ocr_tokens"] = to_python_native(row["ocr_tokens"])

            if "image_classes" in row and row["image_classes"] is not None:
                meta_entry["image_classes"] = to_python_native(row["image_classes"])

            if "flickr_original_url" in row and pd.notna(row.get("flickr_original_url")):
                meta_entry["flickr_original_url"] = str(row["flickr_original_url"])

            if "flickr_300k_url" in row and pd.notna(row.get("flickr_300k_url")):
                meta_entry["flickr_300k_url"] = str(row["flickr_300k_url"])

            metadata_list.append(meta_entry)

        # 每处理完一个 parquet 文件就增量保存一次 JSON（防止中断丢失数据）
        json_path = os.path.join(OUTPUT_DIR, f"textvqa_{split_name}.json")
        _save_json(metadata_list, json_path)
        print(f"  💾 已保存进度: {len(metadata_list)} 条 QA -> {json_path}")

    print(f"\n✅ {split_name} 处理完成:")
    print(f"   图像数量: {len(saved_images)}")
    print(f"   QA 条目数: {len(metadata_list)}")
    print(f"   JSON 文件: {json_path}")
    print(f"   图像目录: {split_images_dir}")


def main():
    print("TextVQA 数据集处理脚本")
    print(f"Parquet 目录: {PARQUET_DIR}")
    print(f"图像输出目录: {IMAGES_DIR}")
    print(f"JSON 输出目录: {OUTPUT_DIR}")

    os.makedirs(IMAGES_DIR, exist_ok=True)

    for split in SPLITS:
        try:
            process_split(split)
        except Exception as e:
            print(f"\n❌ 处理 {split} 时发生异常: {e}")
            print(f"   已处理的数据已保存到对应的 JSON 文件中。")
            import traceback
            traceback.print_exc()
            continue  # 继续处理下一个 split

    print(f"\n{'='*60}")
    print("全部处理完成！")
    print(f"{'='*60}")

    # 统计汇总
    for split in SPLITS:
        json_path = os.path.join(OUTPUT_DIR, f"textvqa_{split}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            split_img_dir = os.path.join(IMAGES_DIR, split)
            n_images = len(os.listdir(split_img_dir)) if os.path.exists(split_img_dir) else 0
            print(f"  {split}: {len(data)} 条 QA, {n_images} 张图像")


if __name__ == "__main__":
    main()
