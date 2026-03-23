"""
ColaMem Inference Script (Dual Model Architecture) - Debug Version
包含了 Latent Memory 统计、Attention 权重检查 以及 暴力视觉塔移植修复功能

使用示例：
    python scripts/inference_with_frozen_visual.py \
        --checkpoint ./outputs/colamem_phase1/model \
        --base_model /path/to/qwen3vl \
        --val_json ./data/ShareGPT4V/colamem_val.json \
        --image_dir ./data/coco/train2017 \
        --num_samples 5 \
        --output ./outputs/debug_results.json
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Any
import gc

# 添加 src 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 使用 Qwen3VL
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration

from src.models.modeling import ColaMemModel


class ColaMemInferenceDebug:
    """
    ColaMem 推理引擎 (带诊断与强制修复功能)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        base_model_path: str = None
    ):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.base_model_path = base_model_path
        
        print(f"\n{'='*70}")
        print(f"ColaMem Inference Debug Mode")
        print(f"{'='*70}")
        print(f"[ColaMem Inference Debug] 初始化推理引擎...")
        print(f"  - Checkpoint: {checkpoint_dir}")
        print(f"  - Device: {device}")
        print(f"  - Dtype: {torch_dtype}")
        print(f"  - Model Type: Qwen3VL")
        print(f"  - Attention: {attn_implementation}")
        print(f"{'='*70}\n")
        
        # 1. 加载 Processor
        encoder_dir = os.path.join(checkpoint_dir, "encoder")
        if os.path.exists(os.path.join(checkpoint_dir, "preprocessor_config.json")):
            processor_dir = checkpoint_dir
        elif os.path.exists(os.path.join(encoder_dir, "preprocessor_config.json")):
            processor_dir = encoder_dir
        elif base_model_path and os.path.exists(os.path.join(base_model_path, "preprocessor_config.json")):
            processor_dir = base_model_path
            print(f"  - 从基础模型加载 Processor: {base_model_path}")
        else:
            processor_dir = base_model_path if base_model_path else encoder_dir
            print(f"  - 尝试从 {processor_dir} 加载 Processor")
            
        print(f"[ColaMem Inference Debug] 加载 Processor: {processor_dir}")
        
        self.processor = Qwen3VLProcessor.from_pretrained(
            processor_dir,
            min_pixels=256*28*28,
            max_pixels=1280*28*28
        )
        
        # 2. 加载 ColaMem 模型 (不使用 trust_remote_code)
        print(f"[ColaMem Inference Debug] 加载模型...")
        self.model = ColaMemModel.from_pretrained(
            checkpoint_dir,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            gradient_checkpointing=False,
            tokenizer=self.processor.tokenizer,
            base_model_path=base_model_path
        )
        self.model.eval()
        
        # 获取模型参数用于诊断
        self.num_mem = getattr(self.model, "num_mem", 64)
        self.num_work = getattr(self.model, "num_work", 32)
        
        print(f"[ColaMem Inference Debug] 模型参数:")
        print(f"  - num_mem (Memory slots): {self.num_mem}")
        print(f"  - num_work (Workspace slots): {self.num_work}")
        
        # =============================================
        # [权重验尸] 检查 Vision Encoder 是否为随机初始化
        # =============================================
        self._verify_vision_encoder_weights()
        
        # =========================================================
        # [关键修复] 执行暴力视觉塔移植 (Organ Transplant)
        # =========================================================
        if self.base_model_path:
            self._force_vision_transplant()
        else:
            print("\033[93m[Warning] 未提供 --base_model，跳过视觉塔修复。如果模型输出乱码，请务必提供 base_model。\033[0m")

        print(f"[ColaMem Inference Debug] ✅ 初始化完成!")
    
    def _verify_vision_encoder_weights(self):
        """
        验证 Vision Encoder 权重是否为随机初始化
        """
        print(f"\n{'='*60}")
        print(f"[权重验尸] 检查 Vision Encoder 权重是否为随机初始化...")
        print(f"{'='*60}")
        
        try:
            # 获取 Vision Encoder 模块
            visual_module = self.model._get_encoder_inner_model().visual
            
            # 1. 检查 patch_embed (第一层卷积)
            if hasattr(visual_module, 'patch_embed'):
                patch_embed = visual_module.patch_embed
                if hasattr(patch_embed, 'proj'):
                    weight = patch_embed.proj.weight
                    print(f"\n[1] Vision Patch Embed 权重统计:")
                    print(f"    Shape: {weight.shape}")
                    print(f"    Mean:  {weight.mean().item():.6f}")
                    print(f"    Std:   {weight.std().item():.6f}")
                    print(f"    Max:   {weight.max().item():.6f}")
                    print(f"    Min:   {weight.min().item():.6f}")
                    
                    # 判断是否为随机初始化
                    is_random = (abs(weight.mean().item()) < 1e-3 and 0.005 < weight.std().item() < 0.15)
                    if is_random:
                        print(f"\033[91m    [CRITICAL] 权重符合随机初始化特征！预训练权重未加载！\033[0m")
                    else:
                        print(f"\033[92m    [OK] 权重看起来是训练过的。\033[0m")
            
            # 2. 检查第一个 Transformer Block
            if hasattr(visual_module, 'blocks') and len(visual_module.blocks) > 0:
                first_block = visual_module.blocks[0]
                
                # QKV 权重
                if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'qkv'):
                    qkv_weight = first_block.attn.qkv.weight
                    print(f"\n[2] Vision Block[0] QKV 权重统计:")
                    print(f"    Shape: {qkv_weight.shape}")
                    print(f"    Mean:  {qkv_weight.mean().item():.6f}")
                    print(f"    Std:   {qkv_weight.std().item():.6f}")
                    
                    is_random = (abs(qkv_weight.mean().item()) < 1e-3 and 0.005 < qkv_weight.std().item() < 0.15)
                    if is_random:
                        print(f"\033[91m    [CRITICAL] 权重符合随机初始化特征！\033[0m")
                    else:
                        print(f"\033[92m    [OK] 权重看起来是训练过的。\033[0m")
            
            # 3. 检查 merger (投影层)
            if hasattr(visual_module, 'merger'):
                merger = visual_module.merger
                for name, module in merger.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        print(f"\n[3] Vision Merger ({name}) 权重统计:")
                        print(f"    Shape: {module.weight.shape}")
                        print(f"    Mean:  {module.weight.mean().item():.6f}")
                        print(f"    Std:   {module.weight.std().item():.6f}")
                        
                        is_random = (abs(module.weight.mean().item()) < 1e-3 and 0.005 < module.weight.std().item() < 0.15)
                        if is_random:
                            print(f"\033[91m    [CRITICAL] 权重符合随机初始化特征！\033[0m")
                        else:
                            print(f"\033[92m    [OK] 权重看起来是训练过的。\033[0m")
                        break
            
            print(f"\n{'='*60}")
            print(f"[权重验尸] 检查完成")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\033[93m[权重验尸] 检查失败: {e}\033[0m")
            import traceback
            traceback.print_exc()
    
    def _force_vision_transplant(self):
        """
        [暴力修复] 直接替换 Vision Tower 对象
        解决 load_state_dict 因前缀不匹配导致加载失败的问题
        """
        print(f"\n{'='*60}")
        print(f"🚑 启动【暴力器官移植】修复程序")
        print(f"{'='*60}")
        
        try:
            # 1. 定位目标模型的 Vision 部分
            recipient_parent = None
            if hasattr(self.model, 'encoder'):
                recipient_parent = self.model._get_encoder_inner_model()
                print("[Target] 锁定目标: encoder inner model (Qwen3VLModel).visual")
            elif hasattr(self.model, 'compressor'):
                recipient_parent = self.model.compressor.model
                print("[Target] 锁定目标: self.model.compressor.model.visual")
            else:
                print("\033[91m[Error] 无法在 ColaMem 中找到 encoder 或 compressor 模块！\033[0m")
                return

            # 打印修复前的状态
            old_weight = recipient_parent.visual.patch_embed.proj.weight
            old_std = old_weight.std().item()
            print(f"[修复前] Patch Embed Std: {old_std:.6f}")
            
            # 检查是否需要修复
            is_random = (abs(old_weight.mean().item()) < 1e-3 and 0.005 < old_std < 0.15)
            if not is_random:
                print("\033[92m✅ [SKIP] 权重看起来已经是预训练的，跳过移植。\033[0m")
                print(f"{'='*60}\n")
                return

            # 2. 加载供体 (Base Model) - Qwen3VL
            print(f"[Donor] 正在加载基础模型 (CPU): {self.base_model_path}")
            
            # 获取模型的 dtype (从参数中推断)
            model_dtype = next(self.model.parameters()).dtype
            
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                device_map="cpu", 
                torch_dtype=model_dtype
            )
            donor_visual = base_model.model.visual
            
            donor_std = donor_visual.patch_embed.proj.weight.std().item()
            print(f"[Donor] 供体 Vision Std: {donor_std:.6f}")

            # 3. 执行移植 (直接对象替换)
            print("🔄 正在执行视觉塔替换 (Pointer Swapping)...")
            
            # 关键一步：直接把 base_model 的 visual 对象赋给 colamem
            recipient_parent.visual = donor_visual
            
            # 将替换后的模块移动到 GPU
            recipient_parent.visual.to(self.device)
            
            # 4. 验证
            new_std = recipient_parent.visual.patch_embed.proj.weight.std().item()
            print(f"[修复后] Patch Embed Std: {new_std:.6f}")
            
            if abs(new_std - donor_std) < 1e-9:
                print("\033[92m✅ [SUCCESS] 视觉塔移植成功！权重已完全同步。\033[0m")
            else:
                print("\033[91m❌ [FAIL] 移植看起来失败了，Std 不匹配！\033[0m")
            
            # 清理
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\033[91m[CRITICAL ERROR] 修复过程中崩溃: {e}\033[0m")
            import traceback
            traceback.print_exc()
            
        print(f"{'='*60}\n")

    def _prepare_images(self, image_paths: List[str]) -> List[Image.Image]:
        """加载图像"""
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"图像不存在: {path}")
            img = Image.open(path).convert("RGB")
            images.append(img)
        return images
    
    def _build_compress_prompt(self, images: List[Image.Image], context: Optional[str] = None) -> str:
        image_tokens = ""
        for _ in images:
            image_tokens += "<|vision_start|><|image_pad|><|vision_end|>"
        
        if context:
            user_content = f"{image_tokens}\n{context}"
        else:
            user_content = image_tokens
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ]
        
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return prompt
    
    def _build_solve_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    def _diagnose_latent_memory(self, latent_memory: torch.Tensor) -> Dict[str, Any]:
        """诊断 Latent Memory 健康度"""
        diag = {
            "shape": list(latent_memory.shape),
            "has_nan": False,
            "has_inf": False,
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "min": 0.0,
            "status": "OK",
            "warnings": []
        }
        
        print(f"\n{'='*60}")
        print(f"[诊断探针 1] Latent Memory 健康度检查")
        print(f"{'='*60}")
        print(f"  Shape: {latent_memory.shape}")
        
        # 检查 NaN/Inf
        if torch.isnan(latent_memory).any():
            diag["has_nan"] = True
            diag["status"] = "CRITICAL"
            diag["warnings"].append("检测到 NaN！")
            print("\033[91m  [CRITICAL] 检测到 NaN！\033[0m")
            return diag
        
        if torch.isinf(latent_memory).any():
            diag["has_inf"] = True
            diag["status"] = "CRITICAL"
            diag["warnings"].append("检测到 Inf！")
            print("\033[91m  [CRITICAL] 检测到 Inf！\033[0m")
            return diag

        mean_val = latent_memory.mean().item()
        std_val = latent_memory.std().item()
        max_val = latent_memory.max().item()
        min_val = latent_memory.min().item()
        
        diag["mean"] = mean_val
        diag["std"] = std_val
        diag["max"] = max_val
        diag["min"] = min_val
        
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Max:  {max_val:.4f}")
        print(f"  Min:  {min_val:.4f}")
        
        # 检查是否被 Projector 放大
        if std_val > 5.0:
            print("\033[93m  [WARNING] Memory 数值过大 (Std > 5)，可能导致 Attention 饱和。\033[0m")
            diag["warnings"].append("Memory 数值过大")
        
        if std_val < 1e-4:
            diag["status"] = "CRITICAL"
            diag["warnings"].append("Posterior Collapse！")
            print("\033[91m  [CRITICAL] Posterior Collapse！所有 Memory 向量趋于相同。\033[0m")
        
        if diag["status"] == "OK":
            print("  \033[92m[OK] Memory 数值分布正常。\033[0m")
            
        return diag
    
    def _diagnose_attention(self, latent_memory: torch.Tensor, text_embeds: torch.Tensor) -> Dict[str, Any]:
        """诊断 Solver 对 Memory 的关注度"""
        diag = {
            "attn_implementation": "unknown",
            "similarity": 0.0,
            "status": "OK",
            "warnings": []
        }
        
        print(f"\n{'='*60}")
        print(f"[诊断探针 2] Solver 注意力检查")
        print(f"{'='*60}")
        
        attn_impl = getattr(self.model.decoder.config, '_attn_implementation', 'unknown')
        diag["attn_implementation"] = attn_impl
        print(f"  Attention Implementation: {attn_impl}")
        
        # 使用余弦相似度作为替代诊断
        print("  [替代诊断] 检查 Memory Embedding 与 Text Embedding 的相似度...")
        try:
            mem_flat = latent_memory.reshape(-1, latent_memory.shape[-1])
            text_flat = text_embeds.reshape(-1, text_embeds.shape[-1])
            
            mem_norm = F.normalize(mem_flat, dim=-1)
            text_norm = F.normalize(text_flat, dim=-1)
            
            similarity = (mem_norm @ text_norm.T).abs().mean().item()
            diag["similarity"] = similarity
            print(f"  Memory-Text Cosine Similarity: {similarity:.4f}")
            
            if similarity < 0.01:
                diag["status"] = "WARNING"
                diag["warnings"].append("Memory 与 Text 几乎正交")
                print("\033[93m  [WARNING] Memory 与 Text 几乎正交，可能存在表示空间不对齐。\033[0m")
            else:
                print("  \033[92m[OK] Memory 与 Text 有一定关联性。\033[0m")
        except Exception as e:
            diag["status"] = "ERROR"
            diag["warnings"].append(f"诊断失败: {e}")
            print(f"  [Info] 诊断失败: {e}")
        
        return diag
    
    @torch.no_grad()
    def compress(self, image_paths: List[str], context: Optional[str] = None) -> tuple:
        print(f"\n[Compress] 压缩 {len(image_paths)} 张图像...")
        
        images = self._prepare_images(image_paths)
        prompt = self._build_compress_prompt(images, context)
        
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # [诊断 0] Pixel Values
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            print(f"\n{'='*60}")
            print(f"[诊断探针 0] Pixel Values 检查")
            print(f"{'='*60}")
            print(f"  Shape: {pv.shape}")
            print(f"  Dtype: {pv.dtype}")
            print(f"  Max:  {pv.max().item():.4f}")
            print(f"  Min:  {pv.min().item():.4f}")
            print(f"  Mean: {pv.mean().item():.4f}")
            print(f"  Std:  {pv.std().item():.4f}")
            if pv.abs().sum() < 1e-6:
                 print("\033[91m  [CRITICAL] 图片输入全为 0！\033[0m")
        else:
            print("\033[91m  [CRITICAL] inputs 中没有 pixel_values！\033[0m")
        
        # 检查 image_grid_thw
        if "image_grid_thw" in inputs:
            grid = inputs["image_grid_thw"]
            print(f"  image_grid_thw: {grid}")
        
        # 在 input_ids 末尾拼接 <mem> token（占位符方案）
        if hasattr(self.model, 'mem_token_id') and self.model.mem_token_id is not None:
            num_mem = self.model.num_mem
            mem_ids = torch.full(
                (inputs["input_ids"].shape[0], num_mem),
                self.model.mem_token_id,
                dtype=inputs["input_ids"].dtype,
                device=inputs["input_ids"].device
            )
            mem_mask = torch.ones_like(mem_ids)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], mem_ids], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], mem_mask], dim=1)
        
        latent_memory = self.model.forward(
            compress_input_ids=inputs["input_ids"],
            compress_attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mode='compress'
        )
        
        # [Hotfix] 强制归一化 (如果数值太大)
        if latent_memory.std() > 2.0:
            print("\033[93m  [Hotfix] 检测到 Memory 数值过大，执行临时缩放 (/2.3)...\033[0m")
            latent_memory = latent_memory / 2.3
        
        memory_diag = self._diagnose_latent_memory(latent_memory)
        
        # [诊断 3] 差异检查
        if not hasattr(self, '_last_memory'):
            self._last_memory = latent_memory.clone()
            print(f"\n[诊断探针 3] Latent Memory 差异检查: 首次运行，已缓存 Memory")
        else:
            diff = (latent_memory - self._last_memory).abs().sum().item()
            max_diff = (latent_memory - self._last_memory).abs().max().item()
            print(f"\n{'='*60}")
            print(f"[诊断探针 3] Latent Memory 差异检查")
            print(f"{'='*60}")
            print(f"  与上一次 Memory 的差异:")
            print(f"    - L1 总差异: {diff:.6f}")
            print(f"    - 最大差异:  {max_diff:.6f}")
            if diff < 1e-6:
                print("\033[91m  [CRITICAL] Memory 无变化！Vision Encoder 仍然没有工作。\033[0m")
                memory_diag["status"] = "CRITICAL"
                memory_diag["warnings"].append("Memory 无变化")
            elif diff < 0.1:
                print("\033[93m  [WARNING] Memory 变化极小，可能信息压缩不足。\033[0m")
            else:
                print(f"  \033[92m[OK] Memory 随图片变化。\033[0m")
            self._last_memory = latent_memory.clone()
        
        print(f"\n[Compress] ✅ 压缩完成！Memory shape: {latent_memory.shape}")
        return latent_memory, memory_diag
    
    @torch.no_grad()
    def solve(self, latent_memory, question, max_new_tokens=512, **kwargs):
        print(f"\n[Solve] 回答问题: {question}")
        prompt = self._build_solve_prompt(question)
        inputs = self.processor.tokenizer(prompt, return_tensors="pt", padding=True)
        
        solve_input_ids = inputs["input_ids"].to(self.device)
        solve_attention_mask = inputs["attention_mask"].to(self.device)
        batch_size = latent_memory.shape[0]
        
        # 获取 Embedding
        try:
            embed_layer = self.model.decoder.get_input_embeddings()
        except:
            embed_layer = self.model.decoder.get_input_embeddings()
        text_embeds = embed_layer(solve_input_ids)
        
        # 诊断 Attention
        attn_diag = self._diagnose_attention(latent_memory, text_embeds)
        
        # 拼接 prefix: [Memory, Query, Workspace]
        workspace = self.model.work_tokens.expand(batch_size, -1, -1)
        prefix_embeds = torch.cat([latent_memory, text_embeds, workspace], dim=1)
        
        # 构造 prefix 的 attention mask
        memory_mask = torch.ones((batch_size, self.num_mem), device=self.device, dtype=solve_attention_mask.dtype)
        workspace_mask = torch.ones((batch_size, self.num_work), device=self.device, dtype=solve_attention_mask.dtype)
        prefix_attention_mask = torch.cat([memory_mask, solve_attention_mask, workspace_mask], dim=1)
        
        prefix_len = prefix_embeds.shape[1]
        print(f"[Solve] Prefix 长度: {prefix_len} (Memory: {self.num_mem}, Query: {text_embeds.shape[1]}, Workspace: {self.num_work})")
        
        # =====================================================================
        # 手动 Autoregressive 生成（避免 generate() 的 cache_position 问题）
        # =====================================================================
        print("[Solve] 计算 prefix 的 KV cache...")
        
        # 步骤 1: 先 forward prefix，获取 KV cache
        prefix_outputs = self.model.decoder(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_attention_mask,
            use_cache=True,
            return_dict=True
        )
        past_key_values = prefix_outputs.past_key_values
        
        # 从 prefix 最后一个 logit 贪婪采样第一个 token
        logits = prefix_outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
        
        generated_tokens = [next_token]
        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        print("[Solve] 开始手动 autoregressive 生成...")
        
        # =====================================================================
        # [关键修复] 累积的 attention mask
        # 在使用 KV cache 时，attention mask 必须覆盖 [所有已处理的位置]
        # 否则新 token 无法 attend 到 cache 中的 Memory！
        # =====================================================================
        current_attn_mask = prefix_attention_mask.clone()  # 初始化为 prefix 的 mask
        
        # 步骤 2: 手动循环生成
        for step in range(max_new_tokens - 1):
            # 获取当前 token 的 embedding
            cur_embeds = embed_layer(next_token)  # [batch, 1, dim]
            
            # [关键] 累积 attention mask：为新 token 添加一个 1
            new_mask = torch.ones((batch_size, 1), device=self.device, dtype=current_attn_mask.dtype)
            current_attn_mask = torch.cat([current_attn_mask, new_mask], dim=1)
            
            # Forward 一步
            # 注意：传递完整的累积 attention mask，确保能 attend 到 Memory
            outputs = self.model.decoder(
                inputs_embeds=cur_embeds,
                attention_mask=current_attn_mask,  # [batch, prefix_len + step + 1]
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            # 更新 KV cache
            past_key_values = outputs.past_key_values
            
            # 贪婪解码下一个 token
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            
            # 检查是否生成了 EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        # 拼接所有生成的 tokens
        generated_ids = torch.cat(generated_tokens, dim=1)  # [batch, num_generated]
        
        # 解码输出
        answer = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"[Solve] ✅ 回答完成！生成了 {len(generated_tokens)} 个 tokens")
        print(f"\n模型回答: {answer}")
        return answer, attn_diag

    def predict(
        self,
        image_paths: List[str],
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """一次性推理：Compress + Solve"""
        latent_memory, memory_diag = self.compress(image_paths, context)
        answer, attn_diag = self.solve(latent_memory, question, max_new_tokens=max_new_tokens, **kwargs)
        
        return {
            "answer": answer,
            "diagnostics": {
                "memory": memory_diag,
                "attention": attn_diag
            }
        }

    def batch_predict(self, samples, image_dir=None, **kwargs):
        results = []
        for i, sample in enumerate(samples, 1):
            print(f"\n{'='*70}")
            print(f"[Batch] 处理样本 {i}/{len(samples)}")
            print(f"{'='*70}")
            
            try:
                img_files = sample.get("compress", {}).get("images", [])
                img_paths = [os.path.join(image_dir, f) for f in img_files] if image_dir else img_files
                q = sample.get("solve", {}).get("question")
                gt_answer = sample.get("solve", {}).get("answer", "N/A")
                
                print(f"图像: {img_paths}")
                print(f"问题: {q}")
                print(f"GT答案: {gt_answer[:100]}..." if len(gt_answer) > 100 else f"GT答案: {gt_answer}")
                
                result = self.predict(img_paths, q, **kwargs)
                
                results.append({
                    "id": sample.get("id", i),
                    "images": img_files,
                    "question": q,
                    "gt_answer": gt_answer,
                    "model_answer": result["answer"],
                    "diagnostics": result["diagnostics"],
                    "status": "success"
                })
            except Exception as e:
                print(f"\033[91m[Error] 处理失败: {e}\033[0m")
                import traceback
                traceback.print_exc()
                results.append({
                    "id": sample.get("id", i),
                    "status": "failed",
                    "error": str(e)
                })
        return results


def main():
    parser = argparse.ArgumentParser(description="ColaMem Inference Debug (带诊断功能)")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点目录")
    parser.add_argument("--base_model", type=str, required=True, help="必需！用于修复 Vision 权重")
    parser.add_argument("--val_json", type=str, default=None, help="验证集 JSON 文件路径")
    parser.add_argument("--image_dir", type=str, default=None, help="图像目录路径")
    parser.add_argument("--num_samples", type=int, default=5, help="采样数量")
    parser.add_argument("--images", nargs='+', help="图像路径列表（手动指定模式）")
    parser.add_argument("--question", type=str, help="问题（手动指定模式必需）")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--output", default="debug_res.json", help="结果输出文件")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度，0 表示贪婪解码")
    
    args = parser.parse_args()
    
    engine = ColaMemInferenceDebug(
        args.checkpoint, 
        device=args.device, 
        base_model_path=args.base_model
    )
    
    # 构建生成参数
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    
    if args.val_json:
        with open(args.val_json) as f:
            data = json.load(f)
        import random
        samples = random.sample(data, min(args.num_samples, len(data)))
        print(f"\n从验证集采样 {len(samples)} 个样本进行测试...")
        
        results = engine.batch_predict(samples, args.image_dir, **gen_kwargs)
        
        # 统计
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\n{'='*70}")
        print(f"[统计] 完成！成功: {success_count}/{len(results)}")
        print(f"{'='*70}")
        
        # 保存结果
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    "checkpoint": args.checkpoint,
                    "val_json": args.val_json,
                    "num_samples": len(results),
                    "success_count": success_count,
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 结果已保存到: {args.output}")
            
    elif args.images and args.question:
        result = engine.predict(args.images, args.question, **gen_kwargs)
        
        print(f"\n{'='*70}")
        print(f"回答: {result['answer']}")
        print(f"{'='*70}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 结果已保存到: {args.output}")
    else:
        print("[Error] 请指定输入参数：")
        print("  1. 使用 --val_json 和 --image_dir 从验证集读取数据")
        print("  2. 使用 --images 和 --question 手动指定")


if __name__ == "__main__":
    main()