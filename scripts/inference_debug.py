"""
ColaMem Inference Script (Dual Model Architecture) - Debug Version
包含了 Latent Memory 统计和 Attention 权重检查功能

使用示例：
    # 从验证集读取数据进行测试
    python scripts/inference_debug.py \
        --checkpoint ./outputs/colamem_phase1/model \
        --base_model /path/to/qwen3vl \
        --val_json ./data/ShareGPT4V/colamem_val.json \
        --image_dir ./data/coco/train2017 \
        --num_samples 5 \
        --output ./outputs/debug_results.json
    
    # 手动指定模式
    python scripts/inference_debug.py \
        --checkpoint ./outputs/colamem_phase1/model \
        --images ./data/test/image1.jpg \
        --question "Describe this image."
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

# 添加 src 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import Qwen3VLProcessor
from src.models.modeling import ColaMemModel


class ColaMemInferenceDebug:
    """
    ColaMem 推理引擎 (带诊断功能)
    
    在推理过程中输出 Latent Memory 统计和 Attention 诊断信息
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        base_model_path: str = None
    ):
        """
        初始化推理引擎
        
        Args:
            checkpoint_dir: 模型检查点目录
            device: 推理设备
            torch_dtype: 数据类型
            attn_implementation: 注意力实现方式
            base_model_path: 原始 Qwen3VL 模型路径
        """
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        print(f"[ColaMem Inference Debug] 初始化推理引擎...")
        print(f"  - Checkpoint: {checkpoint_dir}")
        print(f"  - Device: {device}")
        print(f"  - Dtype: {torch_dtype}")
        
        # 保存 base_model_path 用于后续自动修复
        self.base_model_path = base_model_path
        
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
        
        print(f"[ColaMem Inference Debug] 加载 Processor...")
        self.processor = Qwen3VLProcessor.from_pretrained(
            processor_dir,
            min_pixels=256*28*28,
            max_pixels=1280*28*28
        )
        
        # 2. 加载模型
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
        
        # [重要] 检测是否启用了 workspace
        # Phase 1 训练时 use_workspace=false，模型中没有 work_tokens 或 num_work=0
        # 需要同时检查:
        #   1. work_tokens 参数存在
        #   2. work_tokens 不为 None
        #   3. num_work > 0 (避免空张量 expand 失败)
        self.use_workspace = (
            hasattr(self.model, 'work_tokens') and 
            self.model.work_tokens is not None and
            self.num_work > 0
        )
        
        print(f"[ColaMem Inference Debug] 模型参数:")
        print(f"  - num_mem (Memory slots): {self.num_mem}")
        print(f"  - num_work (Workspace slots): {self.num_work}")
        print(f"  - use_workspace: {self.use_workspace}")
        
        # =============================================
        # [权重验尸] 检查 Vision Encoder 是否为随机初始化
        # =============================================
        self._verify_vision_encoder_weights()
        
        print(f"[ColaMem Inference Debug] ✅ 初始化完成!")
    
    def _verify_vision_encoder_weights(self):
        """
        验证 Vision Encoder 权重是否为随机初始化
        
        随机初始化的特征：
        - Mean ≈ 0
        - Std ≈ 0.01 ~ 0.1 (取决于初始化方法)
        - 所有层的统计特性高度相似
        
        训练后的特征：
        - Mean 可能偏离 0
        - Std 变化较大
        - 不同层统计特性差异明显
        """
        print(f"\n{'='*60}")
        print(f"[权重验尸] 检查 Vision Encoder 权重是否为随机初始化...")
        print(f"{'='*60}")
        
        try:
            # 获取 Vision Encoder 模块
            # Qwen3VL 结构: encoder.model.visual
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
                
                # MLP 权重
                if hasattr(first_block, 'mlp'):
                    mlp = first_block.mlp
                    if hasattr(mlp, 'fc1'):
                        fc1_weight = mlp.fc1.weight
                        print(f"\n[3] Vision Block[0] MLP.fc1 权重统计:")
                        print(f"    Shape: {fc1_weight.shape}")
                        print(f"    Mean:  {fc1_weight.mean().item():.6f}")
                        print(f"    Std:   {fc1_weight.std().item():.6f}")
                        
                        is_random = (abs(fc1_weight.mean().item()) < 1e-3 and 0.005 < fc1_weight.std().item() < 0.15)
                        if is_random:
                            print(f"\033[91m    [CRITICAL] 权重符合随机初始化特征！\033[0m")
                        else:
                            print(f"\033[92m    [OK] 权重看起来是训练过的。\033[0m")
            
            # 3. 检查 merger (投影层)
            if hasattr(visual_module, 'merger'):
                merger = visual_module.merger
                # 尝试获取 merger 的第一个线性层
                for name, module in merger.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        print(f"\n[4] Vision Merger ({name}) 权重统计:")
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
            
            # =============================================
            # 自动修复：如果 Vision Encoder 是随机初始化的，从 base_model 重载
            # =============================================
            if hasattr(self, 'base_model_path') and self.base_model_path:
                # 检查 patch_embed 权重是否符合随机初始化特征
                if hasattr(visual_module, 'patch_embed') and hasattr(visual_module.patch_embed, 'proj'):
                    weight = visual_module.patch_embed.proj.weight
                    is_random = (abs(weight.mean().item()) < 1e-3 and 0.005 < weight.std().item() < 0.15)
                    
                    if is_random:
                        print(f"\n\033[93m[自动修复] 检测到 Vision Encoder 未加载预训练权重！\033[0m")
                        print(f"[自动修复] 正在从 base_model 重载 Vision Tower 权重...")
                        
                        try:
                            from transformers import Qwen3VLForConditionalGeneration
                            
                            # 加载 base_model 的 Vision 部分
                            print(f"[自动修复] 加载基础模型: {self.base_model_path}")
                            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                                self.base_model_path,
                                device_map="cpu",
                                torch_dtype=torch.bfloat16
                            )
                            
                            # 获取 base_model 的 visual 模块
                            base_visual = base_model.model.visual
                            
                            # 复制 Vision 权重到 Encoder
                            print(f"[自动修复] 复制 Vision 权重到 Encoder...")
                            self.model._get_encoder_inner_model().visual.load_state_dict(
                                base_visual.state_dict()
                            )
                            
                            # 验证修复结果
                            weight_after = visual_module.patch_embed.proj.weight
                            print(f"[自动修复] 修复后 patch_embed 权重统计:")
                            print(f"    Mean:  {weight_after.mean().item():.6f}")
                            print(f"    Std:   {weight_after.std().item():.6f}")
                            
                            is_still_random = (abs(weight_after.mean().item()) < 1e-3 and 0.005 < weight_after.std().item() < 0.15)
                            if is_still_random:
                                print(f"\033[91m[自动修复] ❌ 修复失败，权重仍为随机初始化！\033[0m")
                            else:
                                print(f"\033[92m[自动修复] ✅ Vision Encoder 权重已成功修复！\033[0m")
                            
                            # 释放 base_model
                            del base_model
                            torch.cuda.empty_cache()
                            
                        except Exception as fix_error:
                            print(f"\033[91m[自动修复] 修复失败: {fix_error}\033[0m")
                            import traceback
                            traceback.print_exc()
            
        except Exception as e:
            print(f"\033[93m[权重验尸] 检查失败: {e}\033[0m")
            import traceback
            traceback.print_exc()
    
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
        """构建 Compress 阶段的 Prompt"""
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
        """构建 Solve 阶段的 Prompt"""
        messages = [
            {"role": "user", "content": question}
        ]
        
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    def _diagnose_latent_memory(self, latent_memory: torch.Tensor) -> Dict[str, Any]:
        """
        诊断 Latent Memory 健康度
        
        Returns:
            诊断结果字典
        """
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
            diag["warnings"].append("检测到 NaN！梯度爆炸或数值溢出。")
            print("\033[91m  [CRITICAL] 检测到 NaN！梯度爆炸或数值溢出。\033[0m")
        
        if torch.isinf(latent_memory).any():
            diag["has_inf"] = True
            diag["status"] = "CRITICAL"
            diag["warnings"].append("检测到 Inf！数值溢出。")
            print("\033[91m  [CRITICAL] 检测到 Inf！数值溢出。\033[0m")
        
        if not diag["has_nan"] and not diag["has_inf"]:
            # 统计数值分布
            mean_val = latent_memory.mean().item()
            std_val = latent_memory.std().item()
            max_val = latent_memory.max().item()
            min_val = latent_memory.min().item()
            
            diag["mean"] = mean_val
            diag["std"] = std_val
            diag["max"] = max_val
            diag["min"] = min_val
            
            print(f"  Mean: {mean_val:.4f} (预期: 变化)")
            print(f"  Std:  {std_val:.4f}  (预期: 变化)")
            print(f"  Max:  {max_val:.4f}")
            print(f"  Min:  {min_val:.4f}")
            
            # 新增：检测 LayerNorm 过度归一化问题
            # 如果 mean ≈ 0 且 std ≈ 1，说明可能被 LayerNorm 归一化了
            is_layernorm_normalized = (abs(mean_val) < 0.01 and abs(std_val - 1.0) < 0.01)
            if is_layernorm_normalized:
                diag["status"] = "WARNING"
                diag["warnings"].append("Memory 显示 LayerNorm 归一化特征 (mean≈0, std≈1)！不同图像可能无法区分。")
                print("\033[93m  [WARNING] Memory 显示 LayerNorm 归一化特征 (mean≈0, std≈1)！\033[0m")
                print("  -> 问题: 所有图像的 Memory 可能具有相同的统计特性，导致无法区分。")
                print("  -> 建议: 移除 Projector 输出层的 LayerNorm，或使用 RMSNorm。")
            
            # 诊断规则
            if std_val < 1e-4:
                diag["status"] = "CRITICAL"
                diag["warnings"].append("Posterior Collapse (坍塌)！所有 Memory 向量趋于相同。")
                print("\033[91m  [CRITICAL] Posterior Collapse (坍塌)！所有 Memory 向量趋于相同。\033[0m")
                print("  -> 原因可能：学习率过大、Loss权重不对、或Projector断连。")
            elif std_val < 1e-2:
                if diag["status"] == "OK":
                    diag["status"] = "WARNING"
                diag["warnings"].append("Memory 信号极弱 (Std太小)，Solver 可能无法区分。")
                print("\033[93m  [WARNING] Memory 信号极弱 (Std太小)，Solver 可能无法区分。\033[0m")
            
            if abs(mean_val) > 5.0:
                if diag["status"] == "OK":
                    diag["status"] = "WARNING"
                diag["warnings"].append("Memory 均值偏移过大，可能缺乏 LayerNorm 约束。")
                print("\033[93m  [WARNING] Memory 均值偏移过大，可能缺乏 LayerNorm 约束。\033[0m")
        
        if diag["status"] == "OK":
            print("  \033[92m[OK] Memory 数值分布正常。\033[0m")
        
        return diag
    
    def _diagnose_attention(self, latent_memory: torch.Tensor, text_embeds: torch.Tensor) -> Dict[str, Any]:
        """
        诊断 Solver 对 Memory 的关注度
        
        由于 Flash Attention 2 不支持返回 attention weights，
        使用 Memory-Text 余弦相似度作为替代诊断
        """
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
                diag["warnings"].append("Memory 与 Text 几乎正交，可能存在表示空间不对齐。")
                print("\033[93m  [WARNING] Memory 与 Text 几乎正交，可能存在表示空间不对齐。\033[0m")
            else:
                print("  \033[92m[OK] Memory 与 Text 有一定关联性。\033[0m")
        except Exception as e:
            diag["status"] = "ERROR"
            diag["warnings"].append(f"替代诊断失败: {e}")
            print(f"  [Info] 替代诊断失败: {e}")
        
        return diag
    
    @torch.no_grad()
    def compress(
        self,
        image_paths: List[str],
        context: Optional[str] = None
    ) -> tuple:
        """
        Phase 1: Compress (带健康度诊断)
        
        Returns:
            (latent_memory, memory_diag): Memory 张量和诊断结果
        """
        print(f"\n[Compress] 压缩 {len(image_paths)} 张图像...")
        
        images = self._prepare_images(image_paths)
        prompt = self._build_compress_prompt(images, context)
        
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # =============================================
        # [诊断 0] 检查 Pixel Values 输入是否有效
        # =============================================
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
            
            # 检查是否全 0
            if pv.abs().sum() < 1e-6:
                print("\033[91m  [CRITICAL] 图片输入全为 0！预处理（Processor）失效！\033[0m")
            
            # 检查数值范围是否异常
            if pv.max().item() > 100 or pv.min().item() < -100:
                print("\033[93m  [WARNING] Pixel Values 数值范围异常，可能未归一化。\033[0m")
        else:
            print("\033[91m  [CRITICAL] inputs 中没有 pixel_values！\033[0m")
        
        # 检查 image_grid_thw (Qwen3VL 特有)
        if "image_grid_thw" in inputs:
            grid = inputs["image_grid_thw"]
            print(f"  image_grid_thw: {grid}")
        else:
            print("\033[93m  [WARNING] inputs 中没有 image_grid_thw！\033[0m")
        
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
        
        # 诊断 Latent Memory
        memory_diag = self._diagnose_latent_memory(latent_memory)
        
        # =============================================
        # [诊断 3] Latent Memory 差异检查
        # =============================================
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
                print("\033[91m  [CRITICAL] 警告：本次 Memory 与上次完全相同！Compressor 未对新图片做出反应。\033[0m")
                print("  -> 可能原因：")
                print("     1. Vision Encoder 输入无效（pixel_values 全 0 或恒定）")
                print("     2. Vision Encoder 输出未传入 Projector")
                print("     3. Cross-Attention 的 mask 屏蔽了图像特征")
                print("     4. Projector 输出被 LayerNorm 过度归一化")
                memory_diag["status"] = "CRITICAL"
                memory_diag["warnings"].append("Memory 与上次完全相同！Compressor 未对新图片做出反应。")
            elif diff < 0.1:
                print("\033[93m  [WARNING] Memory 变化极小，可能信息压缩不足。\033[0m")
            else:
                print(f"  \033[92m[OK] Memory 有明显变化。\033[0m")
            self._last_memory = latent_memory.clone()
        
        print(f"\n[Compress] ✅ 压缩完成！Memory shape: {latent_memory.shape}")
        return latent_memory, memory_diag
    
    @torch.no_grad()
    def solve(
        self,
        latent_memory: torch.Tensor,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> tuple:
        """
        Phase 2: Solve (带 Attention 诊断)
        
        Returns:
            (answer, attn_diag): 回答和诊断结果
        """
        print(f"\n[Solve] 回答问题: {question}")
        
        prompt = self._build_solve_prompt(question)
        inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        )
        
        solve_input_ids = inputs["input_ids"].to(self.device)
        solve_attention_mask = inputs["attention_mask"].to(self.device)
        
        batch_size = latent_memory.shape[0]
        embed_layer = self.model.decoder.get_input_embeddings()
        text_embeds = embed_layer(solve_input_ids)
        
        # 诊断 Attention
        attn_diag = self._diagnose_attention(latent_memory, text_embeds)
        
        # 根据 use_workspace 设置拼接序列
        if self.use_workspace:
            # 拼接 [Latent Memory, Question, Workspace]
            workspace = self.model.work_tokens.expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([latent_memory, text_embeds, workspace], dim=1)
            
            # 构造 Attention Mask
            memory_mask = torch.ones((batch_size, self.num_mem), device=self.device, dtype=solve_attention_mask.dtype)
            workspace_mask = torch.ones((batch_size, self.num_work), device=self.device, dtype=solve_attention_mask.dtype)
            attention_mask = torch.cat([memory_mask, solve_attention_mask, workspace_mask], dim=1)
            print(f"[Solve] 序列构成: [Memory({self.num_mem}) + Text({text_embeds.shape[1]}) + Workspace({self.num_work})]")
        else:
            # [Phase 1 模式] 拼接 [Latent Memory, Question] 不使用 Workspace
            inputs_embeds = torch.cat([latent_memory, text_embeds], dim=1)
            
            # 构造 Attention Mask
            memory_mask = torch.ones((batch_size, self.num_mem), device=self.device, dtype=solve_attention_mask.dtype)
            attention_mask = torch.cat([memory_mask, solve_attention_mask], dim=1)
            print(f"[Solve] 序列构成: [Memory({self.num_mem}) + Text({text_embeds.shape[1]})] (无 Workspace)")
        
        # 生成
        print("[Solve] 开始生成...")
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        if temperature <= 0 or not do_sample:
            # 贪婪解码：不采样，不传 temperature（避免 HF generate 报错）
            gen_kwargs["do_sample"] = False
        else:
            # 采样模式
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        
        outputs = self.model.decoder.generate(**gen_kwargs)
        
        answer = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[Solve] ✅ 回答完成！")
        return answer, attn_diag
    
    def predict(
        self,
        image_paths: List[str],
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        一次性推理：Compress + Solve (带完整诊断)
        
        Returns:
            {
                "answer": str,
                "diagnostics": {
                    "memory": {...},
                    "attention": {...}
                }
            }
        """
        # Phase 1: Compress
        latent_memory, memory_diag = self.compress(image_paths, context)
        
        # Phase 2: Solve
        answer, attn_diag = self.solve(
            latent_memory,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        
        return {
            "answer": answer,
            "diagnostics": {
                "memory": memory_diag,
                "attention": attn_diag
            }
        }
    
    def batch_predict(
        self,
        samples: List[Dict],
        image_dir: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> List[Dict]:
        """
        批量推理 (带诊断)
        """
        results = []
        
        for i, sample in enumerate(samples, 1):
            print(f"\n{'='*70}")
            print(f"[Batch] 处理样本 {i}/{len(samples)}")
            print(f"{'='*70}")
            
            try:
                # 获取图像路径
                image_files = sample.get("compress", {}).get("images", [])
                # 智能处理图片路径：如果已经是绝对路径则直接使用，否则拼接 image_dir
                image_paths = []
                for img in image_files:
                    if os.path.isabs(img):
                        image_paths.append(img)
                    elif image_dir:
                        image_paths.append(os.path.join(image_dir, img))
                    else:
                        image_paths.append(img)
                
                # 获取问题和答案
                question = sample.get("solve", {}).get("question", "Describe this image.")
                gt_answer = sample.get("solve", {}).get("answer", "N/A")
                
                print(f"图像: {image_paths}")
                print(f"问题: {question}")
                print(f"GT答案: {gt_answer[:100]}..." if len(gt_answer) > 100 else f"GT答案: {gt_answer}")
                
                # 推理
                result = self.predict(
                    image_paths=image_paths,
                    question=question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                
                print(f"\n模型回答: {result['answer']}")
                
                results.append({
                    "id": i,
                    "images": image_files,
                    "question": question,
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
                    "id": i,
                    "images": sample.get("compress", {}).get("images", []),
                    "question": sample.get("solve", {}).get("question", ""),
                    "gt_answer": sample.get("solve", {}).get("answer", ""),
                    "model_answer": None,
                    "diagnostics": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="ColaMem Inference Debug (带诊断功能)")
    
    # 模型参数
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点目录")
    parser.add_argument("--base_model", type=str, default=None,
                        help="原始 Qwen3VL 模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="数据类型 (default: bfloat16)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="注意力实现方式 (default: flash_attention_2)")
    
    # 输入参数
    parser.add_argument("--val_json", type=str, default=None,
                        help="验证集 JSON 文件路径")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="图像目录路径")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="从 val_json 中采样的数量 (default: 5)")
    parser.add_argument("--images", nargs='+', default=None,
                        help="图像路径列表（手动指定模式）")
    parser.add_argument("--context", type=str, default=None,
                        help="可选的文本上下文")
    parser.add_argument("--question", type=str, default=None,
                        help="问题（手动指定模式必需）")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数 (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度 (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p 采样 (default: 0.9)")
    parser.add_argument("--do_sample", action="store_true",
                        help="是否采样 (default: False, 使用贪婪搜索以便复现)")
    
    # 输出参数
    parser.add_argument("--output", type=str, default=None,
                        help="推理结果输出文件路径（JSON 格式）")
    
    args = parser.parse_args()
    
    # 数据类型映射
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    
    print("\n" + "="*70)
    print("ColaMem Inference Debug Mode")
    print("="*70)
    print(f"📦 模型路径: {args.checkpoint}")
    print(f"🔄 基础模型: {args.base_model}")
    print(f"💻 设备: {args.device}")
    print(f"📊 数据类型: {args.dtype}")
    print(f"🔧 Attention: {args.attn_implementation}")
    print(f"🎲 采样模式: {'是' if args.do_sample else '否 (贪婪搜索)'}")
    print("="*70 + "\n")
    
    # 初始化推理引擎
    try:
        engine = ColaMemInferenceDebug(
            checkpoint_dir=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
            base_model_path=args.base_model
        )
    except Exception as e:
        print(f"\n\033[91m[Error] 初始化失败: {e}\033[0m")
        import traceback
        traceback.print_exc()
        return
    
    # 执行推理
    try:
        if args.val_json:
            # 从验证集 JSON 读取数据进行测试
            print(f"\n从验证集加载数据: {args.val_json}")
            with open(args.val_json, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            
            # 采样
            import random
            samples = random.sample(val_data, min(args.num_samples, len(val_data)))
            print(f"采样 {len(samples)} 个样本进行测试...")
            
            # 批量推理
            results = engine.batch_predict(
                samples=samples,
                image_dir=args.image_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
            )
            
            # 统计
            success_count = sum(1 for r in results if r["status"] == "success")
            print(f"\n{'='*70}")
            print(f"[统计] 完成！成功: {success_count}/{len(results)}")
            print(f"{'='*70}")
            
            # 汇总诊断结果
            print("\n[诊断汇总]")
            memory_issues = []
            attn_issues = []
            for r in results:
                if r["diagnostics"]:
                    if r["diagnostics"]["memory"]["status"] != "OK":
                        memory_issues.extend(r["diagnostics"]["memory"]["warnings"])
                    if r["diagnostics"]["attention"]["status"] != "OK":
                        attn_issues.extend(r["diagnostics"]["attention"]["warnings"])
            
            if memory_issues:
                print(f"  Memory 问题: {len(memory_issues)} 个警告")
                for w in set(memory_issues):
                    print(f"    - {w}")
            else:
                print("  Memory: ✅ 全部正常")
            
            if attn_issues:
                print(f"  Attention 问题: {len(attn_issues)} 个警告")
                for w in set(attn_issues):
                    print(f"    - {w}")
            else:
                print("  Attention: ✅ 全部正常")
            
            # 保存结果
            if args.output:
                print(f"\n保存结果到: {args.output}")
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
                
                print(f"✅ 结果已保存！")
        
        elif args.images and args.question:
            # 手动指定模式：一次性推理
            print("\n" + "="*70)
            print("手动指定模式")
            print("="*70)
            print(f"图像: {args.images}")
            if args.context:
                print(f"上下文: {args.context}")
            print(f"问题: {args.question}")
            print("-"*70)
            
            result = engine.predict(
                image_paths=args.images,
                question=args.question,
                context=args.context,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
            )
            
            print("\n" + "="*70)
            print("回答:")
            print("="*70)
            print(result["answer"])
            print("="*70 + "\n")
            
            # 保存结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({
                        "images": args.images,
                        "question": args.question,
                        "context": args.context,
                        "answer": result["answer"],
                        "diagnostics": result["diagnostics"]
                    }, f, ensure_ascii=False, indent=2)
                print(f"结果已保存到: {args.output}")
        
        else:
            print("[Error] 请指定输入参数：")
            print("  1. 使用 --val_json 和 --image_dir 从验证集读取数据")
            print("  2. 使用 --images 和 --question 手动指定")
            return
    
    except Exception as e:
        print(f"\n\033[91m[Error] 推理失败: {e}\033[0m")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()