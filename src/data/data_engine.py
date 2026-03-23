import json
import os
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

class ColaMemDataset(Dataset):
    def __init__(self, json_path, processor):
        """
        Args:
            json_path: 数据索引文件路径
            processor: Qwen3VLProcessor
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        
        # 确保 tokenizer 使用 left padding（推理时的标准配置）
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        
        # [诊断] 预检查前 N 条数据的图像路径
        self._preflight_image_check(max_check=min(10, len(self.data)))

    def _preflight_image_check(self, max_check=10):
        """训练前预检查图像路径是否有效"""
        missing_count = 0
        checked_count = 0
        missing_examples = []
        
        for idx in range(max_check):
            item = self.data[idx]
            compress_data = item.get('compress', item)
            
            image_paths = []
            if 'content' in compress_data:
                for element in compress_data['content']:
                    if element['type'] == 'image':
                        path = element.get('path', element.get('image', ''))
                        if path:
                            image_paths.append(path)
            else:
                image_list = compress_data.get('images', [])
                if isinstance(image_list, str):
                    image_list = [image_list]
                image_paths = image_list
            
            for path in image_paths:
                checked_count += 1
                # 跳过 URL（http/https 开头）
                if path.startswith('http://') or path.startswith('https://'):
                    continue
                if not os.path.exists(path):
                    missing_count += 1
                    if len(missing_examples) < 3:
                        missing_examples.append(f"  idx={idx}: {path}")
        
        # 输出诊断结果
        if missing_count > 0:
            msg = (f"[ColaMemDataset] ⚠️ 图像路径预检查: {missing_count}/{checked_count} 个图像文件不存在！\n"
                   f"  示例:\n" + "\n".join(missing_examples))
            warnings.warn(msg)
            print(msg)
        elif checked_count > 0:
            print(f"[ColaMemDataset] ✅ 图像路径预检查: {checked_count} 个图像文件均存在 (抽检前 {max_check} 条)")
        else:
            print(f"[ColaMemDataset] ℹ️ 图像路径预检查: 前 {max_check} 条数据中无图像")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- 1. 准备 Compressor 输入 (Context) ---
        # 支持原生的图文交错序列格式
        compress_messages = [
            {"role": "user", "content": []}
        ]
        
        compress_data = item.get('compress', item)
        
        # 直接使用 content 列表（保持图像在文本间的原始位置）
        if 'content' in compress_data:
            for element in compress_data['content']:
                if element['type'] == 'image':
                    compress_messages[0]["content"].append({
                        "type": "image", 
                        "image": element.get('path', element.get('image', ''))
                    })
                elif element['type'] == 'text':
                    compress_messages[0]["content"].append({
                        "type": "text", 
                        "text": element['text']
                    })
        else:
            # 兼容旧格式：先图像后文本
            image_list = compress_data.get('images', [])
            if isinstance(image_list, str): 
                image_list = [image_list]
            
            for img_path in image_list:
                compress_messages[0]["content"].append({"type": "image", "image": img_path})
            
            text_content = compress_data.get('text', '').strip()
            if text_content:
                compress_messages[0]["content"].append({"type": "text", "text": text_content})
        
        # 确保有触发指令
        if not compress_messages[0]["content"] or \
           compress_messages[0]["content"][-1].get("type") != "text":
            compress_messages[0]["content"].append({"type": "text", "text": "Observe and memorize."})

        # 预处理视觉信息
        compress_text = self.processor.apply_chat_template(compress_messages, tokenize=False, add_generation_prompt=False)
        compress_image_inputs, compress_video_inputs = process_vision_info(compress_messages)
        
        # [诊断] 检查图像是否成功加载
        has_image_in_content = any(
            el.get('type') == 'image' for el in compress_messages[0]['content']
        )
        if has_image_in_content and (compress_image_inputs is None or len(compress_image_inputs) == 0):
            warnings.warn(
                f"[ColaMemDataset] ⚠️ 样本 {idx}: content 中包含图像但 process_vision_info 返回空！"
                f" 图像可能加载失败。"
            )
        
        # Tokenize Compressor Input
        compress_inputs = self.processor(
            text=[compress_text],
            images=compress_image_inputs,
            videos=compress_video_inputs,
            padding=False, # Collator 会做 padding
            return_tensors="pt"
        )

        # --- 2. 准备 Solver 输入 (Q & A) ---
        # 目标: [Memory, Query, Workspace, Answer]
        # 注意: Solver 不需要图片输入，只看 Text，图片信息来自 Latent Memory
        # 
        # 关键设计：分开返回 Query 和 Answer 部分
        # 这样 forward_solve 可以将 Workspace 插入到中间
        # 确保 Answer 能通过 Causal Attention "看到" Workspace，梯度能正确回传
        
        solve_data = item.get('solve', item)
        question = solve_data.get('question', '')
        answer = solve_data.get('answer', '')

        # --- 2.1 处理 Query 部分 ---
        # Query = User message + Assistant header (带 generation_prompt)
        query_messages = [
            {"role": "user", "content": question}
        ]
        query_text = self.processor.apply_chat_template(
            query_messages, 
            tokenize=False, 
            add_generation_prompt=True  # 包含 "<|im_start|>assistant\n"
        )
        query_inputs = self.processor(
            text=[query_text],
            images=None,
            padding=False,
            return_tensors="pt"
        )
        query_input_ids = query_inputs['input_ids'][0]
        query_attention_mask = query_inputs['attention_mask'][0]
        
        # Query 部分不计算 loss
        query_labels = torch.full_like(query_input_ids, -100)
        
        # --- 2.2 处理 Answer 部分 ---
        # Answer = 纯回答内容 + EOS token
        # 注意：不包含 "\<|im_start|\>assistant\n"，因为它已经在 Query 里了
        answer_tokens = self.processor.tokenizer(
            answer, 
            add_special_tokens=False,
            return_tensors="pt"
        )
        answer_input_ids = answer_tokens['input_ids'][0]
        answer_attention_mask = answer_tokens['attention_mask'][0]
        
        # 添加结尾 token: <|im_end|>
        im_end_token = self.processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        im_end_tensor = torch.tensor(im_end_token, dtype=answer_input_ids.dtype)
        answer_input_ids = torch.cat([answer_input_ids, im_end_tensor], dim=0)
        answer_attention_mask = torch.cat([
            answer_attention_mask, 
            torch.ones(len(im_end_token), dtype=answer_attention_mask.dtype)
        ], dim=0)
        
        # Answer 部分计算 loss（labels = input_ids）
        answer_labels = answer_input_ids.clone()
        
        return {
            "compress": {
                "input_ids": compress_inputs['input_ids'][0],
                "attention_mask": compress_inputs['attention_mask'][0],
                # 修复：pixel_values 和 image_grid_thw 不需要 [0] 索引
                # pixel_values: [num_patches, patch_dim]
                # image_grid_thw: [num_images, 3]
                "pixel_values": compress_inputs.get('pixel_values'),
                "image_grid_thw": compress_inputs.get('image_grid_thw'),
                "has_image": compress_inputs.get('pixel_values') is not None
            },
            "solve": {
                # Query 部分：包含 User message + Assistant header
                "query_input_ids": query_input_ids,
                "query_attention_mask": query_attention_mask,
                "query_labels": query_labels,
                # Answer 部分：纯回答内容 + <|im_end|>
                "answer_input_ids": answer_input_ids,
                "answer_attention_mask": answer_attention_mask,
                "answer_labels": answer_labels,
            }
        }