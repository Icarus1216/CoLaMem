"""
ColaMem Inference Script (Dual Model Architecture)

支持的推理模式：
1. 一次性推理：压缩 + 回答（适合单次问答）
2. 交互式推理：压缩一次，多次提问（适合多轮对话）
3. 批量推理：批量处理多个样本

使用示例：
    # 一次性推理
    python scripts/inference.py \
        --checkpoint ./output/checkpoints/final \
        --images ./data/test/image1.jpg ./data/test/image2.jpg \
        --question "Describe these images in detail."
    
    # 交互式推理
    python scripts/inference.py \
        --checkpoint ./output/checkpoints/final \
        --images ./data/test/image1.jpg \
        --interactive
    
    # 批量推理
    python scripts/inference.py \
        --checkpoint ./output/checkpoints/final \
        --batch_file ./data/test/batch_queries.json
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Any

# 添加 src 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import Qwen3VLProcessor
from src.models.modeling import ColaMemModel


class ColaMemInference:
    """
    ColaMem 推理引擎
    
    支持：
    - 一次性推理（compress + solve）
    - 交互式推理（compress once, solve multiple times）
    - 批量推理
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
            checkpoint_dir: 模型检查点目录（包含 encoder/, decoder/, custom_params.pt）
            device: 推理设备
            torch_dtype: 数据类型
            attn_implementation: 注意力实现方式
            base_model_path: 原始 Qwen3VL 模型路径，当保存的权重损坏时使用
        """
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        print(f"[ColaMem Inference] 初始化推理引擎...")
        print(f"  - Checkpoint: {checkpoint_dir}")
        print(f"  - Device: {device}")
        print(f"  - Dtype: {torch_dtype}")
        
        # 1. 检查 checkpoint 结构（兼容全参数和 LoRA 两种保存格式）
        encoder_dir = os.path.join(checkpoint_dir, "encoder")
        decoder_dir = os.path.join(checkpoint_dir, "decoder")
        encoder_lora_dir = os.path.join(checkpoint_dir, "encoder_lora")
        decoder_lora_dir = os.path.join(checkpoint_dir, "decoder_lora")

        has_full_checkpoint = os.path.exists(encoder_dir) or os.path.exists(decoder_dir)
        has_lora_checkpoint = os.path.exists(encoder_lora_dir) or os.path.exists(decoder_lora_dir)

        if not has_full_checkpoint and not has_lora_checkpoint:
            raise ValueError(
                f"无效 checkpoint 目录: {checkpoint_dir}\n"
                f"未找到全参数目录(encoder/decoder)或 LoRA 目录(encoder_lora/decoder_lora)"
            )

        print(f"[ColaMem Inference] 加载 Processor...")
        # 优先从主目录加载 Processor（包含 preprocessor_config.json、tokenizer 等）
        # 如果主目录没有，则回退到 encoder 目录，最后回退到 base_model_path
        if os.path.exists(os.path.join(checkpoint_dir, "preprocessor_config.json")):
            processor_dir = checkpoint_dir
        elif os.path.exists(encoder_dir) and os.path.exists(os.path.join(encoder_dir, "preprocessor_config.json")):
            processor_dir = encoder_dir
        elif base_model_path and os.path.exists(os.path.join(base_model_path, "preprocessor_config.json")):
            processor_dir = base_model_path
            print(f"  - 从基础模型加载 Processor: {base_model_path}")
        else:
            if os.path.exists(encoder_dir):
                processor_dir = encoder_dir
            elif base_model_path:
                processor_dir = base_model_path
                print(f"  - 从基础模型回退加载 Processor: {base_model_path}")
            else:
                raise ValueError(
                    "未找到可用的 Processor 配置文件。"
                    "LoRA checkpoint 请提供 --base_model，或确保 checkpoint 根目录包含 processor 文件。"
                )
            print(f"  - 尝试从 {processor_dir} 加载 Processor")
        self.processor = Qwen3VLProcessor.from_pretrained(
            processor_dir,
            min_pixels=256*28*28,
            max_pixels=1280*28*28
        )
        
        # 2. 加载模型
        print(f"[ColaMem Inference] 加载模型...")
        self.model = ColaMemModel.from_pretrained(
            checkpoint_dir,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            gradient_checkpointing=False,  # 推理时不需要
            tokenizer=self.processor.tokenizer,
            base_model_path=base_model_path  # 当保存的权重损坏时回退使用
        )
        self.model.eval()
        
        # [重要] 检测是否启用了 workspace
        # Phase 1 训练时 use_workspace=false，模型中没有 work_tokens 或 num_work=0
        # 需要同时检查:
        #   1. work_tokens 参数存在
        #   2. work_tokens 不为 None
        #   3. num_work > 0 (避免空张量 expand 失败)
        self.use_workspace = (
            hasattr(self.model, 'work_tokens') and 
            self.model.work_tokens is not None and
            self.model.num_work > 0
        )
        
        print(f"[ColaMem Inference] 模型参数:")
        print(f"  - num_mem: {self.model.num_mem}")
        print(f"  - num_work: {self.model.num_work}")
        print(f"  - use_workspace: {self.use_workspace}")
        print(f"[ColaMem Inference] ✅ 初始化完成！")
    
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
        """
        构建 Compress 阶段的 Prompt
        
        格式：
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        <|vision_start|><|image_pad|><|vision_end|>
        [Context if provided]<|im_end|>
        """
        # 构建图像占位符
        image_tokens = ""
        for _ in images:
            image_tokens += "<|vision_start|><|image_pad|><|vision_end|>"
        
        # 构建完整 prompt
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
        """
        构建 Solve 阶段的 Prompt
        
        格式：
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        """
        messages = [
            {"role": "user", "content": question}
        ]
        
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    @torch.no_grad()
    def compress(
        self,
        image_paths: List[str],
        context: Optional[str] = None
    ) -> torch.Tensor:
        """
        Phase 1: Compress
        
        将图像和上下文压缩为 Latent Memory
        
        Args:
            image_paths: 图像路径列表
            context: 可选的文本上下文
        
        Returns:
            latent_memory: [1, num_mem, hidden_size]
        """
        print(f"\n[Compress] 压缩 {len(image_paths)} 张图像...")
        
        # 1. 加载图像
        images = self._prepare_images(image_paths)
        
        # 2. 构建 prompt
        prompt = self._build_compress_prompt(images, context)
        
        # 3. 处理输入
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # 4. 移动到设备
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # 4.5 在 input_ids 末尾拼接 <mem> token（占位符方案）
        if hasattr(self.model, 'mem_token_id') and self.model.mem_token_id is not None:
            num_mem = self.model.num_mem
            mem_ids = torch.full(
                (inputs["input_ids"].shape[0], num_mem),
                self.model.mem_token_id,
                dtype=inputs["input_ids"].dtype,
                device=self.device
            )
            mem_mask = torch.ones_like(mem_ids)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], mem_ids], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], mem_mask], dim=1)
        
        # 5. 执行压缩
        latent_memory = self.model.forward(
            compress_input_ids=inputs["input_ids"],
            compress_attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mode='compress'
        )
        
        print(f"[Compress] ✅ 压缩完成！Memory shape: {latent_memory.shape}")
        return latent_memory
    
    @torch.no_grad()
    def solve(
        self,
        latent_memory: torch.Tensor,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Phase 2: Solve
        
        基于 Latent Memory 回答问题
        
        Args:
            latent_memory: 压缩后的记忆 [1, num_mem, hidden_size]
            question: 问题
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Top-p 采样
            do_sample: 是否采样
        
        Returns:
            answer: 生成的回答
        """
        print(f"\n[Solve] 回答问题: {question}")
        
        # 1. 构建 prompt
        prompt = self._build_solve_prompt(question)
        
        # 2. Tokenize
        inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        )
        
        # 3. 移动到设备
        solve_input_ids = inputs["input_ids"].to(self.device)
        solve_attention_mask = inputs["attention_mask"].to(self.device)
        
        # 4. 准备生成参数
        batch_size = latent_memory.shape[0]
        embed_layer = self.model.decoder.get_input_embeddings()
        text_embeds = embed_layer(solve_input_ids)
        
        # 5. 根据 use_workspace 设置拼接序列
        if self.use_workspace:
            # 拼接 [Latent Memory, Question, Workspace]
            workspace = self.model.work_tokens.expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([latent_memory, text_embeds, workspace], dim=1)
            
            # 构造 Attention Mask: Memory + Question + Workspace
            memory_mask = torch.ones((batch_size, self.model.num_mem), device=self.device, dtype=solve_attention_mask.dtype)
            workspace_mask = torch.ones((batch_size, self.model.num_work), device=self.device, dtype=solve_attention_mask.dtype)
            attention_mask = torch.cat([memory_mask, solve_attention_mask, workspace_mask], dim=1)
            print(f"[Solve] 序列构成: [Memory({self.model.num_mem}) + Text({text_embeds.shape[1]}) + Workspace({self.model.num_work})]")
        else:
            # [Phase 1 模式] 拼接 [Latent Memory, Question] 不使用 Workspace
            inputs_embeds = torch.cat([latent_memory, text_embeds], dim=1)
            
            # 构造 Attention Mask: Memory + Question
            memory_mask = torch.ones((batch_size, self.model.num_mem), device=self.device, dtype=solve_attention_mask.dtype)
            attention_mask = torch.cat([memory_mask, solve_attention_mask], dim=1)
            print(f"[Solve] 序列构成: [Memory({self.model.num_mem}) + Text({text_embeds.shape[1]})] (无 Workspace)")
        
        # 7. 生成
        # 处理 temperature=0 的情况：使用贪婪解码
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        if temperature <= 0 or not do_sample:
            # 贪婪解码：不采样，不需要 temperature
            gen_kwargs["do_sample"] = False
        else:
            # 采样模式：需要 temperature 和 top_p
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        
        outputs = self.model.decoder.generate(**gen_kwargs)
        
        # 8. 解码（只取新生成的部分）
        # outputs 包含了 inputs_embeds 对应的 token，需要跳过
        # 但由于我们用的是 inputs_embeds，outputs 只包含生成的 token
        answer = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[Solve] ✅ 回答完成！")
        return answer
    
    def predict(
        self,
        image_paths: List[str],
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        一次性推理：Compress + Solve
        
        Args:
            image_paths: 图像路径列表
            question: 问题
            context: 可选的上下文
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Top-p 采样
            do_sample: 是否采样
        
        Returns:
            answer: 生成的回答
        """
        # Phase 1: Compress
        latent_memory = self.compress(image_paths, context)
        
        # Phase 2: Solve
        answer = self.solve(
            latent_memory,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        
        return answer
    
    def interactive(
        self,
        image_paths: List[str],
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ):
        """
        交互式推理：压缩一次，多次提问
        
        Args:
            image_paths: 图像路径列表
            context: 可选的上下文
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Top-p 采样
            do_sample: 是否采样
        """
        print("\n" + "="*60)
        print("ColaMem 交互式推理模式")
        print("="*60)
        print(f"图像: {image_paths}")
        if context:
            print(f"上下文: {context}")
        print("\n提示：输入 'quit' 或 'exit' 退出")
        print("-"*60)
        
        # Phase 1: Compress（只执行一次）
        latent_memory = self.compress(image_paths, context)
        
        # Phase 2: 多次 Solve
        while True:
            try:
                question = input("\n[You] ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n再见！")
                    break
                
                if not question:
                    continue
                
                answer = self.solve(
                    latent_memory,
                    question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                
                print(f"\n[ColaMem] {answer}")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n[Error] {e}")
                import traceback
                traceback.print_exc()
    
    def batch_predict(
        self,
        batch_file: str,
        output_file: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ):
        """
        批量推理
        
        Args:
            batch_file: 批量查询文件（JSON 格式）
            output_file: 输出文件路径
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Top-p 采样
            do_sample: 是否采样
        
        批量文件格式：
        [
            {
                "id": "sample_1",
                "images": ["path/to/image1.jpg", "path/to/image2.jpg"],
                "context": "optional context",
                "question": "What do you see?"
            },
            ...
        ]
        """
        print(f"\n[Batch] 加载批量查询文件: {batch_file}")
        
        with open(batch_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        print(f"[Batch] 共 {len(samples)} 个样本")
        
        results = []
        for i, sample in enumerate(samples, 1):
            print(f"\n{'='*60}")
            print(f"[Batch] 处理样本 {i}/{len(samples)}: {sample.get('id', f'sample_{i}')}")
            print(f"{'='*60}")
            
            try:
                answer = self.predict(
                    image_paths=sample["images"],
                    question=sample["question"],
                    context=sample.get("context"),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                
                result = {
                    "id": sample.get("id", f"sample_{i}"),
                    "images": sample["images"],
                    "question": sample["question"],
                    "context": sample.get("context"),
                    "answer": answer,
                    "status": "success"
                }
                
            except Exception as e:
                print(f"[Error] 处理失败: {e}")
                result = {
                    "id": sample.get("id", f"sample_{i}"),
                    "images": sample["images"],
                    "question": sample["question"],
                    "context": sample.get("context"),
                    "answer": None,
                    "status": "failed",
                    "error": str(e)
                }
            
            results.append(result)
        
        # 保存结果
        if output_file:
            print(f"\n[Batch] 保存结果到: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\n[Batch] 完成！成功: {success_count}/{len(samples)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="ColaMem Inference (Dual Model Architecture)")
    
    # 模型参数
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点目录（包含 encoder/, decoder/, custom_params.pt）")
    parser.add_argument("--base_model", type=str, default=None,
                        help="原始 Qwen3VL 模型路径（当保存的权重损坏时回退使用，例如 DeepSpeed ZeRO-3 保存问题）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="数据类型 (default: bfloat16)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="注意力实现方式 (default: flash_attention_2)")
    
    # 推理模式
    parser.add_argument("--interactive", action="store_true",
                        help="交互式推理模式（压缩一次，多次提问）")
    parser.add_argument("--batch_file", type=str, default=None,
                        help="批量推理文件（JSON 格式）")
    
    # 输入参数（默认从 val_json 读取）
    parser.add_argument("--val_json", type=str, default=None,
                        help="验证集 JSON 文件路径（默认模式：自动读取数据进行测试）")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="图像目录路径（与 val_json 配合使用）")
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
                        help="采样温度，设为 0 表示贪婪解码 (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p 采样 (default: 0.9)")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="是否采样 (default: True)")
    
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
    
    # 初始化推理引擎
    try:
        inference = ColaMemInference(
            checkpoint_dir=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
            base_model_path=args.base_model  # 当保存的权重损坏时回退使用
        )
    except Exception as e:
        print(f"\n[Error] 初始化失败: {e}")
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
            
            # 收集结果
            results = []
            
            for i, sample in enumerate(samples, 1):
                print(f"\n{'='*60}")
                print(f"样本 {i}/{len(samples)}")
                print(f"{'='*60}")
                
                # 获取图像路径（智能处理：绝对路径直接使用，相对路径拼接 image_dir）
                image_files = sample.get("compress", {}).get("images", [])
                image_paths = []
                for img in image_files:
                    if os.path.isabs(img):
                        image_paths.append(img)
                    elif args.image_dir:
                        image_paths.append(os.path.join(args.image_dir, img))
                    else:
                        image_paths.append(img)
                
                # 获取问题和答案
                question = sample.get("solve", {}).get("question", "Describe this image.")
                gt_answer = sample.get("solve", {}).get("answer", "N/A")
                
                print(f"图像: {image_paths}")
                print(f"问题: {question}")
                print(f"GT答案: {gt_answer[:200]}..." if len(gt_answer) > 200 else f"GT答案: {gt_answer}")
                
                # 推理
                try:
                    answer = inference.predict(
                        image_paths=image_paths,
                        question=question,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample
                    )
                    status = "success"
                except Exception as e:
                    answer = f"Error: {e}"
                    status = "failed"
                
                print(f"\n模型回答: {answer}")
                
                # 收集结果
                results.append({
                    "id": i,
                    "images": image_files,
                    "question": question,
                    "gt_answer": gt_answer,
                    "model_answer": answer,
                    "status": status
                })
            
            # 保存结果到文件
            if args.output:
                print(f"\n{'='*60}")
                print(f"保存推理结果到: {args.output}")
                print(f"{'='*60}")
                
                # 确保输出目录存在
                output_dir = os.path.dirname(args.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({
                        "checkpoint": args.checkpoint,
                        "val_json": args.val_json,
                        "num_samples": len(results),
                        "success_count": sum(1 for r in results if r["status"] == "success"),
                        "results": results
                    }, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 结果已保存！共 {len(results)} 条，成功 {sum(1 for r in results if r['status'] == 'success')} 条")
        
        elif args.batch_file:
            # 批量推理模式
            inference.batch_predict(
                batch_file=args.batch_file,
                output_file=args.output,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
            )
        
        elif args.interactive:
            # 交互式推理模式
            if not args.images:
                print("[Error] 交互式模式需要提供 --images 参数")
                return
            
            inference.interactive(
                image_paths=args.images,
                context=args.context,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
            )
        
        elif args.images and args.question:
            # 手动指定模式：一次性推理
            print("\n" + "="*60)
            print("ColaMem 一次性推理")
            print("="*60)
            print(f"图像: {args.images}")
            if args.context:
                print(f"上下文: {args.context}")
            print(f"问题: {args.question}")
            print("-"*60)
            
            answer = inference.predict(
                image_paths=args.images,
                question=args.question,
                context=args.context,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
            )
            
            print("\n" + "="*60)
            print("回答:")
            print("="*60)
            print(answer)
            print("="*60 + "\n")
        
        else:
            # 默认行为：提示用户
            print("[Error] 请指定输入参数：")
            print("  1. 使用 --val_json 和 --image_dir 从验证集读取数据")
            print("  2. 使用 --images 和 --question 手动指定")
            print("  3. 使用 --batch_file 进行批量推理")
            print("  4. 使用 --interactive 进入交互模式")
            return
    
    except Exception as e:
        print(f"\n[Error] 推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
