#!/usr/bin/env python3
"""
训练脚本
使用方法：
```bash
# 基础训练
python scripts/train.py --config configs/train_phase1.yaml

# 覆盖关键参数
python scripts/train.py --config configs/train_phase1.yaml \
    --output_dir ./my_output \
    --learning_rate 1e-5 \
    --num_train_epochs 5

# 多卡训练
deepspeed --num_gpus=4 scripts/train.py --config configs/train_phase1.yaml

# 从检查点恢复
python scripts/train.py --config configs/train_phase1.yaml \
    --resume_from_checkpoint ./outputs/checkpoint-1000
```
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    TrainingArguments,
    Qwen3VLProcessor,
    set_seed
)

from src.models.modeling import ColaMemModel
from src.data.data_engine import ColaMemDataset
from src.data.data_collator import ColaMemCollator
from src.engine.trainer import ColaMemTrainerWithMonitoring


def parse_args():
    """解析命令行参数（仅用于覆盖 YAML 配置中的关键参数）"""
    parser = argparse.ArgumentParser(
        description="ColaMem Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 使用 YAML 配置文件
  python scripts/train.py --config configs/train_phase1.yaml
  
  # 覆盖输出目录
  python scripts/train.py --config configs/train_phase1.yaml --output_dir ./my_output
  
  # 覆盖学习率和训练轮数
  python scripts/train.py --config configs/train_phase1.yaml --learning_rate 1e-5 --num_train_epochs 5
  
  # 从检查点恢复训练
  python scripts/train.py --config configs/train_phase1.yaml --resume_from_checkpoint ./outputs/checkpoint-1000
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="训练配置文件路径（YAML 格式）"
    )
    
    # 常用覆盖参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（覆盖 YAML 配置）"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率（覆盖 YAML 配置）"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="训练轮数（覆盖 YAML 配置）"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    
    # DeepSpeed 参数（由启动器自动设置）
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="DeepSpeed 使用的 local rank")
    
    return parser.parse_args()


def load_config_from_yaml(config_path):
    """从 YAML 文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 验证必需的配置项
    required_sections = ['model', 'data', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的 '{section}' 部分")
    
    return config


def apply_cli_overrides(config, args):
    """应用命令行参数覆盖 YAML 配置（仅覆盖非 None 的参数）"""
    # 覆盖输出目录
    if args.output_dir is not None:
        config['training']['output_dir'] = args.output_dir
    
    # 覆盖学习率
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    # 覆盖训练轮数
    if args.num_train_epochs is not None:
        config['training']['num_epochs'] = args.num_train_epochs
    
    # 覆盖恢复检查点
    if args.resume_from_checkpoint is not None:
        config['training']['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    return config


def _is_main_process():
    """判断当前进程是否为主进程（rank 0），避免多卡重复打印"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    return local_rank in (-1, 0)


def main():
    """主训练函数"""
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 加载 YAML 配置
    config = load_config_from_yaml(args.config)
    
    # 3. 应用命令行覆盖
    config = apply_cli_overrides(config, args)
    
    # 4. 提取配置
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    seed = config.get('seed', 42)
    
    # 5. 设置随机种子
    set_seed(seed)
    
    is_main = _is_main_process()
    
    # 6. 加载 Processor
    if is_main:
        print("[1/6] 加载 Processor...")
    processor = Qwen3VLProcessor.from_pretrained(model_config['model_path'])
    
    # 7. 创建数据集
    if is_main:
        print("\n[2/6] 创建数据集...")
    train_dataset = ColaMemDataset(
        json_path=data_config['train_json'],
        processor=processor
    )
    if is_main:
        print(f"训练样本数: {len(train_dataset)}")
    
    eval_dataset = None
    if data_config.get('eval_json'):
        eval_dataset = ColaMemDataset(
            json_path=data_config['eval_json'],
            processor=processor
        )
        if is_main:
            print(f"评估样本数: {len(eval_dataset)}")
    
    # 8. 创建模型（必须在 Collator 之前，因为 ColaMemModel.__init__ 会注册 <mem> token）
    if is_main:
        print("\n[4/6] 创建模型...")
    freeze_vision = model_config.get('freeze_vision_encoder', True)  # 默认冻结 Vision
    freeze_decoder = model_config.get('freeze_decoder', False)  # Phase 1 推荐冻结 Decoder
    use_workspace = model_config.get('use_workspace', True)  # Phase 1 建议禁用 workspace
    
    # Memory Mismatch Ranking Loss (Anti-Bypass) 参数
    use_mmr_loss = model_config.get('use_mmr_loss', False)
    mmr_gamma = model_config.get('mmr_gamma', 0.5)
    mmr_beta = model_config.get('mmr_beta', 0.1)
    mmr_beta_start = model_config.get('mmr_beta_start', None)  # warmup 起始值
    mmr_beta_warmup_steps = model_config.get('mmr_beta_warmup_steps', 0)  # warmup 步数
    
    # Answer-Prefix Dropout (Anti-Bypass 增强) 参数
    use_answer_prefix_dropout = model_config.get('use_answer_prefix_dropout', False)
    answer_prefix_dropout_ratio = model_config.get('answer_prefix_dropout_ratio', 0.3)
    answer_prefix_dropout_prob = model_config.get('answer_prefix_dropout_prob', 0.5)
    
    # LoRA 参数（支持 encoder 和 decoder 独立配置）
    use_lora = model_config.get('use_lora', False)
    # Encoder LoRA 配置
    encoder_lora = model_config.get('encoder_lora', True)
    encoder_lora_r = model_config.get('encoder_lora_r', model_config.get('lora_r', 16))  # 兼容旧配置
    encoder_lora_alpha = model_config.get('encoder_lora_alpha', model_config.get('lora_alpha', 32))
    encoder_lora_dropout = model_config.get('encoder_lora_dropout', model_config.get('lora_dropout', 0.05))
    encoder_lora_target_modules = model_config.get('encoder_lora_target_modules', model_config.get('lora_target_modules', None))
    # Decoder LoRA 配置
    decoder_lora = model_config.get('decoder_lora', True)
    decoder_lora_r = model_config.get('decoder_lora_r', model_config.get('lora_r', 16))  # 兼容旧配置
    decoder_lora_alpha = model_config.get('decoder_lora_alpha', model_config.get('lora_alpha', 32))
    decoder_lora_dropout = model_config.get('decoder_lora_dropout', model_config.get('lora_dropout', 0.05))
    decoder_lora_target_modules = model_config.get('decoder_lora_target_modules', model_config.get('lora_target_modules', None))
    
    model = ColaMemModel(
        model_path=model_config['model_path'],
        num_mem_tokens=model_config['num_mem_tokens'],
        num_work_tokens=model_config['num_work_tokens'],
        use_workspace=use_workspace,
        gradient_checkpointing=model_config.get('gradient_checkpointing', True),
        torch_dtype=getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
        attn_implementation=model_config.get('attn_implementation', 'flash_attention_2'),
        use_semantic_init=model_config.get('use_semantic_init', False),
        lambda_orth=model_config.get('lambda_orth', 0.0),
        orth_warmup_steps=model_config.get('orth_warmup_steps', 0),
        freeze_vision_encoder=freeze_vision,
        freeze_decoder=freeze_decoder,
        use_mmr_loss=use_mmr_loss,
        mmr_gamma=mmr_gamma,
        mmr_beta=mmr_beta,
        mmr_beta_start=mmr_beta_start,
        mmr_beta_warmup_steps=mmr_beta_warmup_steps,
        # Answer-Prefix Dropout
        use_answer_prefix_dropout=use_answer_prefix_dropout,
        answer_prefix_dropout_ratio=answer_prefix_dropout_ratio,
        answer_prefix_dropout_prob=answer_prefix_dropout_prob,
        tokenizer=processor.tokenizer,
        # LoRA 参数（支持 encoder 和 decoder 独立配置）
        use_lora=use_lora,
        # Encoder LoRA 配置
        encoder_lora=encoder_lora,
        encoder_lora_r=encoder_lora_r,
        encoder_lora_alpha=encoder_lora_alpha,
        encoder_lora_dropout=encoder_lora_dropout,
        encoder_lora_target_modules=encoder_lora_target_modules,
        # Decoder LoRA 配置
        decoder_lora=decoder_lora,
        decoder_lora_r=decoder_lora_r,
        decoder_lora_alpha=decoder_lora_alpha,
        decoder_lora_dropout=decoder_lora_dropout,
        decoder_lora_target_modules=decoder_lora_target_modules
    )
    
    # =========================================================================
    # Vision Encoder 处理逻辑
    # =========================================================================
    # 如果冻结 Vision Encoder（推荐）：
    #   - Vision 权重直接使用 from_pretrained 加载的预训练权重
    #   - 不需要额外的器官移植或权重检查
    #   - 节省显存，加快训练
    #
    # 如果不冻结 Vision Encoder：
    #   - 执行器官移植确保权重正确加载
    #   - 执行 Pre-flight Check 验证权重
    # =========================================================================
    if not freeze_vision:
        # 非冻结模式：执行器官移植和权重检查
        base_model_path = model_config.get('base_model_path') or model_config.get('model_path')
        if is_main:
            print(f"\n[Vision] 非冻结模式，执行 Vision Encoder 权重加载...")
        model.load_base_vision_weights(base_model_path)
        
        # Pre-flight Check
        def preflight_vision_check(model):
            """训练前的 Vision 权重检查"""
            if not is_main:
                return
            print("\n" + "=" * 60)
            print("✈️  [Pre-flight Check] Vision Encoder 权重检查")
            print("=" * 60)
            
            w = model._get_encoder_inner_model().visual.patch_embed.proj.weight
            std = w.std().item()
            mean = w.mean().item()
            
            print(f"  Vision Patch Embed:")
            print(f"    - Shape: {tuple(w.shape)}")
            print(f"    - Mean:  {mean:.6f}")
            print(f"    - Std:   {std:.6f}")
            
            EXPECTED_STD = 0.007202
            tolerance = 0.001
            
            if abs(std - EXPECTED_STD) < tolerance:
                print(f"\n✅ [PASSED] Vision Encoder 权重正常 (预训练)")
                print(f"   Std = {std:.6f} ≈ {EXPECTED_STD} (预期值)")
                print("=" * 60 + "\n")
            else:
                print(f"\n⚠️  [WARNING] Vision Encoder 权重 Std 与预期不符")
                print(f"   当前 Std = {std:.6f}")
                print(f"   预期 Std ≈ {EXPECTED_STD}")
                print(f"   差异可能来自模型版本不同，继续训练...")
                print("=" * 60 + "\n")
        
        preflight_vision_check(model)
    else:
        if is_main:
            print(f"\n[Vision] ✅ Vision Encoder 已冻结，使用预训练权重")
    
    # 9. 创建数据整理器（必须在 Model 之后，因为需要 model.mem_token_id）
    if is_main:
        print("\n[5/7] 创建数据整理器...")
    data_collator = ColaMemCollator(
        processor,
        num_mem_tokens=model_config['num_mem_tokens'],
        mem_token_id=model.mem_token_id
    )
    
    # 10. 配置训练参数
    if is_main:
        print("\n[5/6] 配置训练参数...")
    torch_dtype = model_config.get('torch_dtype', 'bfloat16')
    training_args = TrainingArguments(
        # 基本参数
        output_dir=training_config['output_dir'],
        overwrite_output_dir=True,
        
        # 训练参数
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        
        # 优化器参数
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 100),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        
        # 精度参数
        bf16=(torch_dtype == "bfloat16"),
        fp16=(torch_dtype == "float16"),
        
        # 日志参数
        logging_dir=os.path.join(training_config['output_dir'], 'logs'),
        logging_steps=training_config.get('logging_steps', 10),
        logging_first_step=True,
        
        # 保存参数
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 2),
        
        # 评估参数
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=training_config.get('eval_steps', 500) if eval_dataset else None,
        
        # 数据加载参数
        dataloader_num_workers=training_config.get('num_workers', 4),
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # 重要：保留所有列
        
        # DeepSpeed 配置
        deepspeed=training_config.get('deepspeed_config'),
        
        # 其他参数
        report_to=["tensorboard"],
        seed=seed,
    )
    
    # Encoder / Decoder 独立学习率
    encoder_lr = training_config.get('encoder_lr', None)
    decoder_lr = training_config.get('decoder_lr', None)
    
    # 自定义组件独立学习率（可选）
    anchors_lr = training_config.get('anchors_lr', None)
    projector_lr = training_config.get('projector_lr', None)
    gate_lr = training_config.get('gate_lr', None)
    scale_lr = training_config.get('scale_lr', None)
    
    if is_main and (encoder_lr is not None or decoder_lr is not None):
        print(f"[Config] 🎯 启用独立学习率:")
        print(f"  - Encoder (Compressor) LR: {encoder_lr if encoder_lr else '默认 ' + str(training_config['learning_rate'])}")
        print(f"  - Decoder (Solver) LR: {decoder_lr if decoder_lr else '默认 ' + str(training_config['learning_rate'])}")
        if anchors_lr is not None:
            print(f"  - mem_anchors LR: {anchors_lr}")
        if projector_lr is not None:
            print(f"  - Projector LR: {projector_lr}")
        if gate_lr is not None:
            print(f"  - Gate LR: {gate_lr}")
        if scale_lr is not None:
            print(f"  - Scale LR: {scale_lr}")
    
    # 11. 创建 Trainer（使用高级 Trainer）
    if is_main:
        print("\n[6/6] 创建 Trainer...")
    trainer = ColaMemTrainerWithMonitoring(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        monitor_drift_every_n_steps=training_config.get('monitor_drift_every_n_steps', 100),
        save_drift_report=training_config.get('save_drift_report', True),
        debug_inputs=training_config.get('debug_inputs', True),  # 默认开启输入诊断
        debug_inputs_every_n_steps=training_config.get('debug_inputs_every_n_steps', 50),  # 每 50 步检查一次
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        anchors_lr=anchors_lr,
        projector_lr=projector_lr,
        gate_lr=gate_lr,
        scale_lr=scale_lr,
    )
    
    # 12. 开始训练（只在主进程打印，避免多卡重复输出）
    if training_args.local_rank in (-1, 0):
        print("\n" + "=" * 80)
        print("🎯 开始训练...")
        print("=" * 80 + "\n")
    
    # 检查是否从 checkpoint 恢复
    resume_from_checkpoint = training_config.get('resume_from_checkpoint')
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        if training_args.local_rank in (-1, 0):
            print(f"从 checkpoint 恢复训练: {resume_from_checkpoint}\n")
    else:
        resume_from_checkpoint = None
    
    # 训练
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 13. 保存模型
    if training_args.local_rank in (-1, 0):
        print("\n" + "=" * 80)
        print("💾 保存最终模型...")
        print("=" * 80)
    
    final_model_dir = os.path.join(training_config['output_dir'], 'model')
    model.save_pretrained(final_model_dir)
    
    # 获取当前 rank（仅在 rank 0 上保存 processor）
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    
    if rank == 0:
        # 保存 processor（包含 preprocessor_config.json, tokenizer 等）
        print(f"[ColaMem] 保存 Processor 到 {final_model_dir} ...")
        try:
            processor.save_pretrained(final_model_dir)
            print(f"[ColaMem] ✅ Processor 保存成功")
            
            # 验证 processor 文件是否存在
            expected_files = ['preprocessor_config.json', 'tokenizer_config.json']
            missing_files = [f for f in expected_files if not os.path.exists(os.path.join(final_model_dir, f))]
            if missing_files:
                print(f"[ColaMem] ⚠️ Processor 保存后缺少文件: {missing_files}")
        except Exception as e:
            print(f"[ColaMem] ❌ Processor 保存失败: {e}")
            print(f"[ColaMem] 尝试手动复制 processor 文件...")
            # 如果保存失败，尝试从基础模型复制
            import shutil
            base_model_path = model_config.get('pretrained_path') or model_config.get('base_model_path')
            if base_model_path:
                processor_files = ['preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json', 
                                   'special_tokens_map.json', 'vocab.json', 'merges.txt']
                for pf in processor_files:
                    src = os.path.join(base_model_path, pf)
                    dst = os.path.join(final_model_dir, pf)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        print(f"[ColaMem] 复制 {pf} 成功")
    
    # 同步所有进程
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    if is_main:
        print(f"✅ 模型已保存到 {final_model_dir}")
    
    # 14. 生成最终漂移报告
    if model_config.get('use_semantic_init') and hasattr(model, 'explainability_probe') and model.explainability_probe is not None:
        if is_main:
            print("\n" + "=" * 80)
            print("📊 生成最终 Memory Anchor 漂移报告...")
            print("=" * 80)
        
        final_metrics = model.explainability_probe.detect_drift(
            current_anchors=model.mem_anchors,
            initial_anchors=model.initial_mem_anchors
        )
        
        model.explainability_probe.print_drift_report(final_metrics)
        
        # 保存最终报告
        final_report_path = os.path.join(training_config['output_dir'], "final_drift_report.json")
        with open(final_report_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        if is_main:
            print(f"✅ 最终报告已保存到 {final_report_path}")
    
    if is_main:
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)


if __name__ == "__main__":
    main()
