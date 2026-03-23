"""
ColaMem Data Collator

为双模型架构准备数据批次，完全兼容 Qwen3-VL 的批处理规范

关键特性：
1. Left Padding: Qwen3-VL 推理时使用 left padding（右对齐序列）
2. 图像批处理: 正确处理 pixel_values 和 image_grid_thw 的拼接
3. 数据键名与模型的 forward 参数完全匹配
4. 支持 compress 和 solve 两个阶段的输入
"""

import torch


class ColaMemCollator:
    """
    ColaMem Data Collator - 完全兼容 Qwen3-VL
    
    将数据集的样本整理成批次，格式与 ColaMemModel.forward 兼容
    
    输入格式（来自 Dataset）：
    ```python
    {
        "compress": {
            "input_ids": torch.LongTensor,           # [seq_len]
            "attention_mask": torch.LongTensor,      # [seq_len]
            "pixel_values": torch.FloatTensor or None,  # [num_patches, patch_dim]
            "image_grid_thw": torch.LongTensor or None, # [num_images, 3]
            "has_image": bool
        },
        "solve": {
            "query_input_ids": torch.LongTensor,        # [seq_len] Query 部分
            "query_attention_mask": torch.LongTensor,   # [seq_len]
            "query_labels": torch.LongTensor,           # [seq_len] 全为 -100
            "answer_input_ids": torch.LongTensor,       # [seq_len] Answer 部分
            "answer_attention_mask": torch.LongTensor,  # [seq_len]
            "answer_labels": torch.LongTensor           # [seq_len] 用于计算 loss
        }
    }
    ```
    
    输出格式（传给 Model）：
    ```python
    {
        "compress_input_ids": torch.LongTensor,      # [batch, max_seq_len] - left padded
        "compress_attention_mask": torch.LongTensor, # [batch, max_seq_len] - left padded
        "pixel_values": torch.FloatTensor or None,   # [total_patches, patch_dim]
        "image_grid_thw": torch.LongTensor or None,  # [total_images, 3]
        "query_input_ids": torch.LongTensor,         # [batch, max_seq_len] - left padded
        "query_attention_mask": torch.LongTensor,    # [batch, max_seq_len] - left padded
        "query_labels": torch.LongTensor,            # [batch, max_seq_len] - left padded (全 -100)
        "answer_input_ids": torch.LongTensor,        # [batch, max_seq_len] - right padded
        "answer_attention_mask": torch.LongTensor,   # [batch, max_seq_len] - right padded
        "answer_labels": torch.LongTensor,           # [batch, max_seq_len] - right padded
    }
    ```
    
    拼接顺序: [Latent Memory, Query, Workspace, Answer]
    """
    
    def __init__(self, processor, num_mem_tokens=64, mem_token_id=None):
        """
        Args:
            processor: Qwen3VLProcessor 实例
            num_mem_tokens: Memory Anchor 数量（默认 64）
            mem_token_id: <mem> 特殊 token 的 id（由 ColaMemModel 注册后传入）
        """
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.num_mem_tokens = num_mem_tokens
        
        # <mem> token id：用于在 compress input_ids 末尾拼接占位符
        if mem_token_id is not None:
            self.mem_token_id = mem_token_id
        else:
            # 尝试从 tokenizer 获取
            vocab = processor.tokenizer.get_vocab()
            if '<mem>' in vocab:
                self.mem_token_id = vocab['<mem>']
            else:
                raise ValueError(
                    "[ColaMemCollator] <mem> token 未在 tokenizer 中注册！"
                    "请确保先调用 ColaMemModel.__init__（会自动注册 <mem> token），"
                    "然后再创建 Collator。"
                )
        
        print(f"[ColaMemCollator] ✅ 使用 <mem> token (id={self.mem_token_id}), "
              f"每个样本拼接 {num_mem_tokens} 个")
        
        # 确保有 pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id
            print(f"[ColaMemCollator] Warning: pad_token_id is None, using eos_token_id: {self.pad_token_id}")
        
        # 确保使用 left padding（Qwen3-VL 推理标准）
        if processor.tokenizer.padding_side != "left":
            print(f"[ColaMemCollator] Warning: tokenizer padding_side is '{processor.tokenizer.padding_side}', "
                  f"but Qwen3-VL requires 'left' padding for inference. Setting to 'left'.")
            processor.tokenizer.padding_side = "left"
    
    def _left_pad_sequence(self, sequences, padding_value=0):
        """
        Left padding (右对齐序列)
        
        Args:
            sequences: List[torch.Tensor]，每个 tensor 的 shape 是 [seq_len]
            padding_value: padding 值
        
        Returns:
            torch.Tensor: [batch, max_seq_len]，左侧填充
        """
        batch_size = len(sequences)
        max_len = max(seq.size(0) for seq in sequences)
        
        # 创建全 padding 的 tensor
        padded = torch.full((batch_size, max_len), padding_value, dtype=sequences[0].dtype)
        
        # 右对齐填充（左侧 padding）
        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)
            padded[i, max_len - seq_len:] = seq
        
        return padded
    
    def _right_pad_sequence(self, sequences, padding_value=0):
        """
        Right padding (左对齐序列)
        
        Args:
            sequences: List[torch.Tensor]，每个 tensor 的 shape 是 [seq_len]
            padding_value: padding 值
        
        Returns:
            torch.Tensor: [batch, max_seq_len]，右侧填充
        """
        batch_size = len(sequences)
        max_len = max(seq.size(0) for seq in sequences)
        
        # 创建全 padding 的 tensor
        padded = torch.full((batch_size, max_len), padding_value, dtype=sequences[0].dtype)
        
        # 左对齐填充（右侧 padding）
        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)
            padded[i, :seq_len] = seq
        
        return padded
    
    def __call__(self, batch):
        """
        整理批次数据
        
        Args:
            batch: List[Dict]，来自 Dataset.__getitem__
        
        Returns:
            Dict，格式与 ColaMemModel.forward 兼容
        """
        # 1. 拆包 compress 和 solve 数据
        compress_batch = [item['compress'] for item in batch]
        solve_batch = [item['solve'] for item in batch]
        
        # 2. 为每个样本的 compress input_ids 末尾拼接 num_mem 个 <mem> token
        # 这些 <mem> token 在 forward_compress 中会被替换为 mem_anchors embedding
        # 拼接在 padding 之前（即原始序列末尾），这样 left padding 后 <mem> 仍在序列末尾
        mem_suffix = torch.full((self.num_mem_tokens,), self.mem_token_id, dtype=torch.long)
        mem_mask_suffix = torch.ones(self.num_mem_tokens, dtype=torch.long)
        
        compress_input_ids_with_mem = []
        compress_attention_mask_with_mem = []
        for x in compress_batch:
            # 原始 input_ids + <mem> token
            ids_with_mem = torch.cat([x['input_ids'], mem_suffix], dim=0)
            mask_with_mem = torch.cat([x['attention_mask'], mem_mask_suffix], dim=0)
            compress_input_ids_with_mem.append(ids_with_mem)
            compress_attention_mask_with_mem.append(mask_with_mem)
        
        # Left Pad Compress 输入（Qwen3-VL 标准）
        compress_input_ids = self._left_pad_sequence(
            compress_input_ids_with_mem,
            padding_value=self.pad_token_id
        )
        compress_attention_mask = self._left_pad_sequence(
            compress_attention_mask_with_mem,
            padding_value=0
        )
        
        # 3. 处理图像数据（Qwen3-VL 批处理规范）
        # pixel_values: 每个样本是 [num_patches, patch_dim]，在 dim=0 拼接
        # image_grid_thw: 每个样本是 [num_images, 3]，在 dim=0 拼接
        has_images = [x['has_image'] for x in compress_batch]
        
        if any(has_images):
            # 只拼接有图像的样本
            pixel_values_list = [x['pixel_values'] for x in compress_batch if x['has_image']]
            image_grid_thw_list = [x['image_grid_thw'] for x in compress_batch if x['has_image']]
            
            # 检查是否有样本缺失图像
            num_with_image = sum(has_images)
            num_total = len(has_images)
            if num_with_image != num_total:
                print(f"[Collator] ⚠️ Batch 中 {num_total - num_with_image}/{num_total} "
                      f"个样本无图像！缺图样本的 Encoder 只能看到纯文本。")
            
            # 在 dim=0 拼接（不是 stack）
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        else:
            pixel_values = None
            image_grid_thw = None
        
        # 4. Left Pad Query 输入 (Query 在左侧，需要 left padding 以保持右对齐)
        query_input_ids = self._left_pad_sequence(
            [x['query_input_ids'] for x in solve_batch],
            padding_value=self.pad_token_id
        )
        query_attention_mask = self._left_pad_sequence(
            [x['query_attention_mask'] for x in solve_batch],
            padding_value=0
        )
        query_labels = self._left_pad_sequence(
            [x['query_labels'] for x in solve_batch],
            padding_value=-100
        )
        
        # 5. Right Pad Answer 输入 (Answer 在右侧，需要 right padding 以保持左对齐)
        answer_input_ids = self._right_pad_sequence(
            [x['answer_input_ids'] for x in solve_batch],
            padding_value=self.pad_token_id
        )
        answer_attention_mask = self._right_pad_sequence(
            [x['answer_attention_mask'] for x in solve_batch],
            padding_value=0
        )
        answer_labels = self._right_pad_sequence(
            [x['answer_labels'] for x in solve_batch],
            padding_value=-100
        )
        
        # 6. 返回批次数据（键名与 ColaMemModel.forward 参数匹配）
        return {
            # Compress 阶段的输入
            "compress_input_ids": compress_input_ids,
            "compress_attention_mask": compress_attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            
            # Solve 阶段的输入 - Query 部分 (left padded)
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "query_labels": query_labels,
            
            # Solve 阶段的输入 - Answer 部分 (right padded)
            "answer_input_ids": answer_input_ids,
            "answer_attention_mask": answer_attention_mask,
            "answer_labels": answer_labels,
        }
