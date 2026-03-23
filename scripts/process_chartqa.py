"""
处理 ChartQA 数据集的 parquet 文件：
1. 将图像提取并保存到 images/ 目录
2. 将 QA 元数据（含图像路径）保存到 JSON 文件

ChartQA parquet 列：
  - image: 图像数据（HuggingFace datasets 格式，dict {'bytes': ..., 'path': ...}）
  - query: 问题文本（字符串）
  - label: 答案列表（list of str）
  - human_or_machine: 类别标签（'human' 或 'machine'）
"""

import os
import json
import glob
import io
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd


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
CHARTQA_ROOT = "/mnt/cephszjt/user_juntianzhang/ColaMem/data/ChartQA"
PARQUET_DIR = os.path.join(CHARTQA_ROOT, "data")
IMAGES_DIR = os.path.join(CHARTQA_ROOT, "images")
OUTPUT_DIR = CHARTQA_ROOT  # JSON 输出目录

# 每个 split 对应的 parquet 文件前缀
SPLITS = ["train", "val", "test"]


def save_image(image_data, image_path):
    """将图像数据保存到指定路径"""
    if os.path.exists(image_path):
        return  # 跳过已保存的

    if isinstance(image_data, dict):
        # HuggingFace datasets 格式：{'bytes': b'...', 'path': ...}
        img_bytes = image_data.get("bytes", None)
        if img_bytes is not None:
            img = Image.open(io.BytesIO(img_bytes))
        else:
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

    # 统一转为 RGB 后保存为 PNG（ChartQA 图表图像用 PNG 保留清晰度更好）
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(image_path, "PNG")


def get_image_id(image_data, idx):
    """
    为每张图像生成唯一 ID。
    ChartQA 数据集没有 image_id 字段，所以：
    1. 如果 image_data 是 dict 且有 path 字段，从 path 提取文件名作为 ID
    2. 否则使用图像字节数据的 md5 哈希前16位
    3. 最后兜底使用索引
    """
    if isinstance(image_data, dict):
        path = image_data.get("path", None)
        if path and isinstance(path, str):
            # 去掉扩展名作为 ID
            return os.path.splitext(os.path.basename(path))[0]
        img_bytes = image_data.get("bytes", None)
        if img_bytes is not None:
            return hashlib.md5(img_bytes).hexdigest()[:16]
    elif isinstance(image_data, bytes):
        return hashlib.md5(image_data).hexdigest()[:16]
    
    return f"chartqa_{idx:06d}"


def _save_json(data, json_path):
    """安全保存 JSON 文件（先写临时文件，再原子替换，防止写入中途崩溃导致文件损坏）"""
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)  # 原子替换


def process_split(split_name):
    """处理单个 split (train/val/test)"""
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
    global_idx = 0  # 全局索引，用于兜底生成 image_id

    for pf in parquet_files:
        print(f"\n读取文件: {os.path.basename(pf)}")
        df = pd.read_parquet(pf)
        print(f"  行数: {len(df)}")
        print(f"  列名: {df.columns.tolist()}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理中"):
            # 获取图像数据和 ID
            image_data = row.get("image", None)
            image_id = get_image_id(image_data, global_idx)
            image_filename = f"{image_id}.png"
            image_abs_path = os.path.join(IMAGES_DIR, split_name, image_filename)

            # 保存图像（同一 image_id 只保存一次）
            if image_id not in saved_images:
                if image_data is not None:
                    try:
                        save_image(image_data, image_abs_path)
                        saved_images.add(image_id)
                    except Exception as e:
                        print(f"  [警告] 保存图像失败 (idx={global_idx}): {e}")

            # 获取图像尺寸（如果图像已保存）
            image_width, image_height = None, None
            if os.path.exists(image_abs_path):
                try:
                    with Image.open(image_abs_path) as img:
                        image_width, image_height = img.size
                except Exception:
                    pass

            # 获取问题和答案
            query = str(row["query"]) if pd.notna(row.get("query")) else ""
            label = row.get("label", None)
            if label is not None:
                label = to_python_native(label)
            human_or_machine = str(row["human_or_machine"]) if pd.notna(row.get("human_or_machine")) else ""

            # 构建元数据条目
            meta_entry = {
                "image_id": image_id,
                "question_id": global_idx,
                "question": query,
                "image_path": image_abs_path,
                "image_width": image_width,
                "image_height": image_height,
                "answers": label if isinstance(label, list) else ([label] if label is not None else []),
                "human_or_machine": human_or_machine,
            }

            metadata_list.append(meta_entry)
            global_idx += 1

        # 每处理完一个 parquet 文件就增量保存一次 JSON（防止中断丢失数据）
        json_path = os.path.join(OUTPUT_DIR, f"chartqa_{split_name}.json")
        _save_json(metadata_list, json_path)
        print(f"  💾 已保存进度: {len(metadata_list)} 条 QA -> {json_path}")

    json_path = os.path.join(OUTPUT_DIR, f"chartqa_{split_name}.json")
    print(f"\n✅ {split_name} 处理完成:")
    print(f"   图像数量: {len(saved_images)}")
    print(f"   QA 条目数: {len(metadata_list)}")
    print(f"   JSON 文件: {json_path}")
    print(f"   图像目录: {split_images_dir}")

    # 统计 human / machine 分布
    human_count = sum(1 for m in metadata_list if m["human_or_machine"] == "human")
    machine_count = sum(1 for m in metadata_list if m["human_or_machine"] == "machine")
    print(f"   human 问题: {human_count}")
    print(f"   machine 问题: {machine_count}")


def main():
    print("ChartQA 数据集处理脚本")
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
        json_path = os.path.join(OUTPUT_DIR, f"chartqa_{split}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            split_img_dir = os.path.join(IMAGES_DIR, split)
            n_images = len(os.listdir(split_img_dir)) if os.path.exists(split_img_dir) else 0
            print(f"  {split}: {len(data)} 条 QA, {n_images} 张图像")


if __name__ == "__main__":
    main()
