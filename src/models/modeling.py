"""
ColaMem Dual Model Implementation

基于双模型架构的设计：
- 使用两个独立的 Qwen3VL 模型（encoder 和 decoder）
- 全参数微调（而非 LoRA）
- 兼容标准 Hugging Face Trainer
- 支持 DDP 和 DeepSpeed

架构对比：
┌─────────────────────────────────────────────────────────────┐
│ ColaMem Original (LoRA)    │  ColaMem C3 Style (Full FT)   │
├────────────────────────────┼───────────────────────────────┤
│ Single Qwen3VL + Dual LoRA │  Dual Qwen3VL (独立模型)       │
│ 25M 可训练参数              │  8B 可训练参数                │
│ ~30GB 训练显存              │  ~170GB 训练显存              │
│ 需要适配器切换              │  无需适配器切换               │
│ 自定义 Trainer              │  兼容标准 Trainer             │
└────────────────────────────┴───────────────────────────────┘

使用场景：
1. ✅ 有充足的计算资源（4× H20 80GB 或 2× A100 80GB）
2. ✅ 追求最大性能上限
3. ✅ 希望实现简单、调试方便
4. ✅ 可以接受较长的训练时间

资源需求：
- 训练：~170GB 显存（需要 DeepSpeed ZeRO-3 + CPU Offload）
- 推理：~32GB 显存（两个 4B 模型）
- 训练速度：比 LoRA 方案慢 5-6x
"""

import os
import torch
import torch.nn as nn
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union

# 导入 PEFT (Parameter-Efficient Fine-Tuning) 库
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[ColaMem] 警告: peft 库未安装，LoRA 微调功能将不可用。请运行: pip install peft")


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    
    相比 LayerNorm 的优势：
    1. 计算效率更高 (~19.5% 训练加速)
    2. 显存占用更低 (~7.1% 节省)
    3. 参数量更少 (无 bias)
    4. 数值稳定性更好 (无减法操作)
    
    公式: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    
    与 Qwen3-VL 原生使用的归一化方式保持一致
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 RMS
        # 使用 float32 计算以保证数值稳定性，然后转回原始 dtype
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

# 导入软约束与初始化策略模块
try:
    from .memory_initialization import (
        MemoryAnchorInitializer,
        OrthogonalityRegularizer,
        MemoryExplainabilityProbe,
        initialize_memory_anchors,
        create_orthogonality_regularizer,
        create_explainability_probe
    )
    MEMORY_INIT_AVAILABLE = True
except ImportError:
    MEMORY_INIT_AVAILABLE = False
    print("[ColaMem] 警告: memory_initialization 模块未找到，软约束功能将不可用")


def _is_main_process():
    """判断当前进程是否为主进程（rank 0），避免多卡重复打印"""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank() == 0
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', -1)))
    return local_rank in (-1, 0)


class ColaMemConfig(Qwen3VLConfig):
    """
    ColaMem 自定义配置类
    
    继承自 Qwen3VLConfig，添加 ColaMem C3 特有的配置参数
    
    Args:
        num_mem_tokens: Memory Anchors 的数量（默认 64）
        num_work_tokens: Workspace Tokens 的数量（默认 32）
        use_workspace: 是否使用 Workspace（默认 True，Phase 1 建议关闭）
        use_semantic_init: 是否使用多样化语义初始化（默认 False）
        lambda_orth: 正交正则化权重（默认 0.0，不启用）
        orth_warmup_steps: 正交正则化 warmup 步数（默认 0，不使用 warmup）
        freeze_vision_encoder: 是否冻结 Vision Encoder（默认 True，推荐）
        use_mmr_loss: 是否启用 Memory Mismatch Ranking Loss (Anti-Bypass)（默认 False）
        mmr_gamma: Ranking Loss 的 margin 参数（默认 0.5）
        mmr_beta: Ranking Loss 的权重系数（默认 0.1）
        mmr_beta_start: Beta warmup 起始值（默认与 mmr_beta 相同，即不使用 warmup）
        mmr_beta_warmup_steps: Beta 从 start 线性增长到 mmr_beta 所需的步数（默认 0，即不使用 warmup）
        
        # LoRA 相关参数（支持 encoder 和 decoder 独立配置）
        use_lora: 是否使用 LoRA 微调（默认 False，即全参数微调）
        
        # Encoder LoRA 配置
        encoder_lora: 是否对 Encoder 应用 LoRA（默认 True）
        encoder_lora_r: Encoder LoRA rank（默认 16）
        encoder_lora_alpha: Encoder LoRA alpha 缩放因子（默认 32）
        encoder_lora_dropout: Encoder LoRA dropout 概率（默认 0.05）
        encoder_lora_target_modules: Encoder LoRA 目标模块（默认 None，自动选择）
        
        # Decoder LoRA 配置
        decoder_lora: 是否对 Decoder 应用 LoRA（默认 True）
        decoder_lora_r: Decoder LoRA rank（默认 16）
        decoder_lora_alpha: Decoder LoRA alpha 缩放因子（默认 32）
        decoder_lora_dropout: Decoder LoRA dropout 概率（默认 0.05）
        decoder_lora_target_modules: Decoder LoRA 目标模块（默认 None，自动选择）
        
        **kwargs: 其他 Qwen3VLConfig 参数
    
    Example:
        ```python
        from transformers import AutoConfig
        
        # 创建配置（全参数微调）
        config = ColaMemConfig(
            num_mem_tokens=64,
            num_work_tokens=32,
            use_semantic_init=True,
            lambda_orth=0.05,
            freeze_vision_encoder=True
        )
        
        # 创建配置（LoRA 微调 - 统一配置）
        config = ColaMemConfig(
            use_lora=True,
            encoder_lora_r=16,
            encoder_lora_alpha=32,
            decoder_lora_r=8,  # Decoder 使用较小的 rank
            decoder_lora_alpha=16
        )
        
        # 创建配置（LoRA 微调 - 独立配置）
        config = ColaMemConfig(
            use_lora=True,
            encoder_lora=True,
            encoder_lora_r=32,           # Encoder 使用较大 rank
            encoder_lora_alpha=64,
            encoder_lora_dropout=0.1,
            decoder_lora=True,
            decoder_lora_r=8,             # Decoder 使用较小 rank
            decoder_lora_alpha=16,
            decoder_lora_dropout=0.05
        )
        
        # 保存配置
        config.save_pretrained("./my_colamem_c3")
        
        # 加载配置
        config = AutoConfig.from_pretrained("./my_colamem_c3")
        ```
    """
    model_type = "colamem"  # 模型类型标识符
    
    def __init__(
        self,
        num_mem_tokens: int = 64,
        num_work_tokens: int = 32,
        use_workspace: bool = True,
        use_semantic_init: bool = False,
        lambda_orth: float = 0.0,
        orth_warmup_steps: int = 0,
        freeze_vision_encoder: bool = True,
        freeze_decoder: bool = False,
        use_mmr_loss: bool = False,
        mmr_gamma: float = 0.5,
        mmr_beta: float = 0.1,
        mmr_beta_start: float = None,  # warmup 起始值，None 表示使用 mmr_beta（不 warmup）
        mmr_beta_warmup_steps: int = 0,  # warmup 步数，0 表示不使用 warmup
        # Answer-Prefix Dropout (Anti-Bypass 增强)
        use_answer_prefix_dropout: bool = False,
        answer_prefix_dropout_ratio: float = 0.3,
        answer_prefix_dropout_prob: float = 0.5,
        # LoRA 相关参数（支持 encoder 和 decoder 独立配置）
        use_lora: bool = False,
        # Encoder LoRA 配置
        encoder_lora: bool = True,
        encoder_lora_r: int = 16,
        encoder_lora_alpha: int = 32,
        encoder_lora_dropout: float = 0.05,
        encoder_lora_target_modules: List[str] = None,
        # Decoder LoRA 配置
        decoder_lora: bool = True,
        decoder_lora_r: int = 16,
        decoder_lora_alpha: int = 32,
        decoder_lora_dropout: float = 0.05,
        decoder_lora_target_modules: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_mem_tokens = num_mem_tokens
        self.num_work_tokens = num_work_tokens
        self.use_workspace = use_workspace
        self.use_semantic_init = use_semantic_init
        self.lambda_orth = lambda_orth
        self.orth_warmup_steps = orth_warmup_steps
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_decoder = freeze_decoder
        self.use_mmr_loss = use_mmr_loss
        self.mmr_gamma = mmr_gamma
        self.mmr_beta = mmr_beta
        # Answer-Prefix Dropout
        self.use_answer_prefix_dropout = use_answer_prefix_dropout
        self.answer_prefix_dropout_ratio = answer_prefix_dropout_ratio
        self.answer_prefix_dropout_prob = answer_prefix_dropout_prob
        # LoRA 相关参数（支持独立配置）
        self.use_lora = use_lora
        # Encoder LoRA
        self.encoder_lora = encoder_lora
        self.encoder_lora_r = encoder_lora_r
        self.encoder_lora_alpha = encoder_lora_alpha
        self.encoder_lora_dropout = encoder_lora_dropout
        self.encoder_lora_target_modules = encoder_lora_target_modules
        # Decoder LoRA
        self.decoder_lora = decoder_lora
        self.decoder_lora_r = decoder_lora_r
        self.decoder_lora_alpha = decoder_lora_alpha
        self.decoder_lora_dropout = decoder_lora_dropout
        self.decoder_lora_target_modules = decoder_lora_target_modules


class ColaMemModel(nn.Module):
    """
    ColaMem: Dual Model Architecture
    
    核心设计：
    1. Encoder (Qwen3VL): 压缩上下文 + 图像 → Latent Memory
    2. Decoder (Qwen3VL): Latent Memory + Question → Answer
    3. 支持两种微调模式：
       - 全参数微调（默认）：两个模型全部参数参与训练
       - LoRA 微调：只训练 LoRA 适配器，大幅减少显存和参数量
    
    LoRA 微调架构（use_lora=True）：
    ┌─────────────────────────────────────────────────────────────┐
    │ 全参数微调 (Full FT)       │  LoRA 微调 (PEFT)              │
    ├────────────────────────────┼───────────────────────────────┤
    │ Dual Qwen3VL (独立模型)     │  Dual Qwen3VL + LoRA Adapters │
    │ ~8B 可训练参数              │  ~25M 可训练参数               │
    │ ~170GB 训练显存             │  ~40GB 训练显存                │
    │ 需要 DeepSpeed ZeRO-3       │  可用 DDP / ZeRO-2            │
    │ 最大性能上限                │  高效 + 接近全参性能           │
    └────────────────────────────┴───────────────────────────────┘
    
    Forward 流程：
    ```
    # Phase 1: Compress (使用 encoder)
    Context + Images → [Encoder] → Hidden States → [Projector] → Latent Memory
    
    # Phase 2: Solve (使用 decoder)
    Latent Memory + Question + Workspace → [Decoder] → Answer
    ```
    """
    
    # Qwen3-VL 默认的 LoRA 目标模块
    # 包含 attention 的 q/k/v/o 投影和 MLP 层
    DEFAULT_LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 层
        "gate_proj", "up_proj", "down_proj"       # MLP 层
    ]
    
    def __init__(
        self, 
        model_path: str, 
        num_mem_tokens: int = 64, 
        num_work_tokens: int = 32,
        use_workspace: bool = True,
        gradient_checkpointing: bool = True,
        device_map=None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        use_semantic_init: bool = False,
        lambda_orth: float = 0.0,
        orth_warmup_steps: int = 0,
        freeze_vision_encoder: bool = True,
        freeze_decoder: bool = False,
        use_mmr_loss: bool = False,
        mmr_gamma: float = 0.5,
        mmr_beta: float = 0.1,
        mmr_beta_start: float = None,  # warmup 起始值，None 表示使用 mmr_beta（不 warmup）
        mmr_beta_warmup_steps: int = 0,  # warmup 步数，0 表示不使用 warmup
        # Answer-Prefix Dropout (Anti-Bypass 增强)
        use_answer_prefix_dropout: bool = False,  # 是否启用
        answer_prefix_dropout_ratio: float = 0.3,  # 被 mask 的 answer 前缀比例
        answer_prefix_dropout_prob: float = 0.5,   # 每个 batch 触发 dropout 的概率
        tokenizer=None,
        # LoRA 相关参数（支持 encoder 和 decoder 独立配置）
        use_lora: bool = False,
        # Encoder LoRA 配置
        encoder_lora: bool = True,
        encoder_lora_r: int = 16,
        encoder_lora_alpha: int = 32,
        encoder_lora_dropout: float = 0.05,
        encoder_lora_target_modules: List[str] = None,
        # Decoder LoRA 配置
        decoder_lora: bool = True,
        decoder_lora_r: int = 16,
        decoder_lora_alpha: int = 32,
        decoder_lora_dropout: float = 0.05,
        decoder_lora_target_modules: List[str] = None
    ):
        super().__init__()
        self.num_mem = num_mem_tokens
        self.num_work = num_work_tokens if use_workspace else 0  # Phase 1 禁用 workspace 时设为 0
        self.use_workspace = use_workspace
        self.model_path = model_path
        self.use_semantic_init = use_semantic_init
        self.lambda_orth = lambda_orth
        self.orth_warmup_steps = orth_warmup_steps
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_decoder = freeze_decoder
        self.tokenizer = tokenizer
        
        # LoRA 相关参数（支持 encoder 和 decoder 独立配置）
        self.use_lora = use_lora
        # Encoder LoRA 配置
        self.encoder_lora = encoder_lora
        self.encoder_lora_r = encoder_lora_r
        self.encoder_lora_alpha = encoder_lora_alpha
        self.encoder_lora_dropout = encoder_lora_dropout
        self.encoder_lora_target_modules = encoder_lora_target_modules or self.DEFAULT_LORA_TARGET_MODULES
        # Decoder LoRA 配置
        self.decoder_lora = decoder_lora
        self.decoder_lora_r = decoder_lora_r
        self.decoder_lora_alpha = decoder_lora_alpha
        self.decoder_lora_dropout = decoder_lora_dropout
        self.decoder_lora_target_modules = decoder_lora_target_modules or self.DEFAULT_LORA_TARGET_MODULES
        
        # Memory Mismatch Ranking Loss (Anti-Bypass) 参数
        self.use_mmr_loss = use_mmr_loss
        self.mmr_gamma = mmr_gamma
        self.mmr_beta = mmr_beta  # 最终目标 beta
        self.mmr_beta_start = mmr_beta_start if mmr_beta_start is not None else mmr_beta  # warmup 起始 beta
        self.mmr_beta_warmup_steps = mmr_beta_warmup_steps  # warmup 总步数
        self._mmr_step = 0  # 当前训练步数（用于 beta warmup）
        
        # Answer-Prefix Dropout (Anti-Bypass 增强)
        # 训练时随机 mask 掉 answer 前 K 个 token 的 embedding，
        # 迫使 decoder 不能靠「看到答案开头就猜出后续」来绕过 memory
        self.use_answer_prefix_dropout = use_answer_prefix_dropout
        self.answer_prefix_dropout_ratio = answer_prefix_dropout_ratio
        self.answer_prefix_dropout_prob = answer_prefix_dropout_prob
        
        # 调试标志：控制是否打印详细调试信息
        self.debug_forward = False
        self._debug_step = 0
        
        # 多进程打印控制：只在 rank 0 打印初始化日志
        self._verbose = _is_main_process()
        
        if self._verbose:
            print(f"[ColaMem] 正在加载双模型架构...")
        
        # 加载配置
        init_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation
        }
        if device_map is not None:
            init_kwargs["device_map"] = device_map
        
        # 1. 加载 Encoder (用于 Compress 阶段)
        if self._verbose:
            print(f"[ColaMem] 加载 Encoder: {model_path} ...")
        self.encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **init_kwargs
        )
        
        # 2. 加载 Decoder (用于 Solve 阶段)
        if self._verbose:
            print(f"[ColaMem] 加载 Decoder: {model_path} ...")
        self.decoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **init_kwargs
        )
        
        # 2.5 LoRA 微调：包装 encoder 和 decoder
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "LoRA 微调需要安装 peft 库！请运行: pip install peft"
                )
            
            if self._verbose:
                print(f"\n[ColaMem] 🎯 启用 LoRA 微调模式")
            
            # 包装 Encoder
            if encoder_lora:
                if self._verbose:
                    print(f"[ColaMem] Encoder LoRA 配置:")
                    print(f"  - r (rank): {encoder_lora_r}")
                    print(f"  - alpha: {encoder_lora_alpha}")
                    print(f"  - dropout: {encoder_lora_dropout}")
                    print(f"  - target_modules: {self.encoder_lora_target_modules}")
                
                # 创建 Encoder LoRA 配置
                encoder_lora_config = LoraConfig(
                    r=encoder_lora_r,
                    lora_alpha=encoder_lora_alpha,
                    target_modules=self.encoder_lora_target_modules,
                    lora_dropout=encoder_lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                
                if self._verbose:
                    print(f"[ColaMem] 为 Encoder 添加 LoRA 适配器...")
                self.encoder = get_peft_model(self.encoder, encoder_lora_config)
                if self._verbose:
                    self.encoder.print_trainable_parameters()
            
            # 包装 Decoder (使用独立配置)
            if decoder_lora:
                if self._verbose:
                    print(f"[ColaMem] Decoder LoRA 配置:")
                    print(f"  - r (rank): {decoder_lora_r}")
                    print(f"  - alpha: {decoder_lora_alpha}")
                    print(f"  - dropout: {decoder_lora_dropout}")
                    print(f"  - target_modules: {self.decoder_lora_target_modules}")
                
                # 创建 Decoder LoRA 配置
                decoder_lora_config = LoraConfig(
                    r=decoder_lora_r,
                    lora_alpha=decoder_lora_alpha,
                    target_modules=self.decoder_lora_target_modules,
                    lora_dropout=decoder_lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                
                if self._verbose:
                    print(f"[ColaMem] 为 Decoder 添加 LoRA 适配器...")
                self.decoder = get_peft_model(self.decoder, decoder_lora_config)
                if self._verbose:
                    self.decoder.print_trainable_parameters()
            
            if self._verbose:
                print(f"[ColaMem] ✅ LoRA 适配器添加完成\n")
        
        # 3. 启用梯度检查点（节省显存）
        # 重要：必须使用 use_reentrant=False，否则 mem_anchors 等非叶子参数无法收到梯度！
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # 获取配置
        # Qwen3VL 是多模态模型，hidden_size 在 text_config 中
        encoder_config = self.encoder.config
        if hasattr(encoder_config, 'text_config'):
            hidden_size = encoder_config.text_config.hidden_size
        elif hasattr(encoder_config, 'hidden_size'):
            hidden_size = encoder_config.hidden_size
        else:
            raise AttributeError(f"无法从 {type(encoder_config).__name__} 获取 hidden_size")
        
        # 3.5 注册 <mem> 特殊 token（占位符方案）
        # 核心思路：在 tokenizer 中注册 <mem> token，compress 阶段在 input_ids 末尾
        # 拼接 num_mem 个 <mem> token id，然后在 embedding 后将其替换为 mem_anchors。
        # 这样 get_rope_index / attention_mask / visual_pos_masks / deepstack 全部自动正确处理，
        # 可以直接调用 Qwen3VLModel.forward() 的完整路径，无需手动拼接。
        if tokenizer is not None:
            # 检查是否已注册
            if '<mem>' not in tokenizer.get_vocab():
                tokenizer.add_tokens(['<mem>'], special_tokens=True)
                if self._verbose:
                    print(f"[ColaMem] ✅ 注册 <mem> 特殊 token (id={tokenizer.convert_tokens_to_ids('<mem>')})")
            self.mem_token_id = tokenizer.convert_tokens_to_ids('<mem>')
            
            # 扩展 encoder 和 decoder 的 embedding 层以适配新 token
            # 注意：LoRA 模式下需要获取 base model 来 resize
            encoder_base = self._get_encoder_base_model()
            encoder_base.resize_token_embeddings(len(tokenizer))
            decoder_base = self.decoder.base_model.model if (PEFT_AVAILABLE and isinstance(self.decoder, PeftModel)) else self.decoder
            decoder_base.resize_token_embeddings(len(tokenizer))
            
            if self._verbose:
                print(f"[ColaMem] ✅ 已扩展 embedding 层至 {len(tokenizer)} tokens")
        else:
            self.mem_token_id = None
            if self._verbose:
                print(f"[ColaMem] ⚠️ 未提供 tokenizer，<mem> token 未注册")
        
        embed_layer = self.encoder.get_input_embeddings()
        device = embed_layer.weight.device
        
        # 4. 定义 Memory Anchors (可学习的查询向量)
        # 支持非对称语义初始化
        if use_semantic_init and MEMORY_INIT_AVAILABLE and tokenizer is not None:
            if self._verbose:
                print(f"[ColaMem] 使用多样化语义初始化（统一概念词池，无分组）")
            mem_anchors_init = initialize_memory_anchors(
                model=self.encoder,
                tokenizer=tokenizer,
                num_mem_tokens=num_mem_tokens,
                std_scale=1.0
            )
            # 先创建 mem_anchors 参数
            self.mem_anchors = nn.Parameter(mem_anchors_init.to(device=device, dtype=torch_dtype))
            # 保存初始值（用于漂移检测）
            # 重要：从 mem_anchors.data 创建，而非 mem_anchors_init
            # 使用 .detach().clone() 确保完全独立，避免共享内存
            self.register_buffer(
                'initial_mem_anchors', 
                self.mem_anchors.data.detach().clone()  # 从最终参数创建副本
            )
        else:
            if use_semantic_init:
                if self._verbose:
                    print(f"[ColaMem] 警告: 无法使用语义初始化（缺少 tokenizer 或 memory_initialization 模块），使用随机初始化")
            self.mem_anchors = nn.Parameter(
                torch.randn(1, num_mem_tokens, hidden_size, device=device, dtype=torch_dtype) * 0.02
            )
            # 同样从 mem_anchors.data 创建
            self.register_buffer(
                'initial_mem_anchors',
                self.mem_anchors.data.detach().clone()
            )
        
        # 5. 定义 Workspace Tokens（Phase 1 可禁用）
        if use_workspace:
            self.work_tokens = nn.Parameter(
                torch.randn(1, num_work_tokens, hidden_size, device=device, dtype=torch_dtype) * 0.02
            )
            if self._verbose:
                print(f"[ColaMem] ✅ Workspace 已启用 (num_work_tokens={num_work_tokens})")
        else:
            # Phase 1 禁用 workspace：注册空参数以保持接口一致
            self.register_buffer('work_tokens', torch.empty(0, device=device, dtype=torch_dtype))
            if self._verbose:
                print(f"[ColaMem] ⏭️  Workspace 已禁用 (Phase 1 推荐)")
        
        # 6. 定义 Projector (投影 Encoder 输出到 Decoder 输入空间)
        # 注意：不在输出层使用归一化！
        # 原因：归一化会把所有 Memory 向量归一化到 mean≈0, std≈1，
        #       导致不同图像的 Memory 无法区分（Posterior Collapse）
        # 修复：使用中间层 RMSNorm 来稳定训练，但保留输出层的信息多样性
        # 使用 RMSNorm 替代 LayerNorm：训练快 19.5%，显存省 7.1%
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            RMSNorm(hidden_size),  # 中间层 RMSNorm - 稳定训练
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            # 不加输出归一化 - 保留信息多样性！
        ).to(device=device, dtype=torch_dtype)
        
        # 7. [方案A] 移除多余的 memory_norm（RMSNorm）
        # 原因：projector 内部已有中间层 RMSNorm，额外再做一次会压缩方差，
        # 导致不同样本的 memory 更难区分，加剧 slot collapse / memory 趋同
        # 现在只保留可学习的 memory_scale 来匹配 decoder embedding 尺度
        
        # 7.5 可学习的残差门控 (Residual Gate)
        # 让每个 slot 自主决定「保留多少 anchor 身份 vs 多少 encoder 输出」
        # 初始化为 logit=-2（sigmoid≈0.12），训练初期以 projected_memory 为主
        self.residual_gate_logit = nn.Parameter(
            torch.full((1, num_mem_tokens, 1), -2.0, device=device, dtype=torch_dtype)
        )
        
        # 7.6 输出尺度匹配（可学习的缩放因子）
        # 通过可学习 scale 匹配 decoder embedding 尺度（~0.35），避免 memory 在残差支路中过强
        self.memory_scale = nn.Parameter(
            torch.tensor(0.35, device=device, dtype=torch_dtype)
        )
        
        # 8. 创建正交正则化器（如果启用）
        # [改进] orth 约束作用在 anchor 参数本身（而非 latent_memory 输出）
        # 原因：latent_memory 是图片相关的，强制正交会破坏必要的相关性
        if lambda_orth > 0 and MEMORY_INIT_AVAILABLE:
            if self._verbose:
                print(f"[ColaMem] 启用正交正则化 (λ_orth={lambda_orth}, warmup={orth_warmup_steps} steps)")
                print(f"[ColaMem]   约束对象: anchor 参数 (mem_anchors)，非 latent_memory 输出")
                print(f"[ColaMem]   Loss 类型: off-diagonal only (只惩罚互相似度)")
            self.orth_regularizer = create_orthogonality_regularizer(
                lambda_orth=lambda_orth,
                normalize=True,
                warmup_steps=orth_warmup_steps
            )
        else:
            self.orth_regularizer = None
        
        # 9. 创建可解释性探针（如果使用语义初始化）
        if use_semantic_init and MEMORY_INIT_AVAILABLE and tokenizer is not None:
            self.explainability_probe = create_explainability_probe(
                model=self.encoder,
                tokenizer=tokenizer
            )
        else:
            self.explainability_probe = None
        
        # 10. 创建 Memory 输出可分性探针
        # 用于监控 batch 内不同样本的 latent_memory 是否可区分
        # （anchor 多样 ≠ 输出多样，需要独立监控）
        if MEMORY_INIT_AVAILABLE:
            from src.models.memory_initialization import MemorySeparabilityProbe
            self.separability_probe = MemorySeparabilityProbe()
        else:
            self.separability_probe = None
        
        # 缓存最近一次 latent_memory，供外部 probe 使用（不参与计算图）
        self._last_latent_memory = None
        
        # 10. 设置可训练参数
        if freeze_vision_encoder:
            self._freeze_vision_encoder()
        else:
            self._enable_all_gradients()
        
        # 11. Phase 1 专用：冻结 Decoder (强烈推荐)
        # 强迫 Projector 去适应 Solver，而不是让 Solver 变傻来适应 Projector
        if freeze_decoder:
            self._freeze_decoder()
        
        # 12. Memory Mismatch Ranking Loss (Anti-Bypass) 信息
        if use_mmr_loss:
            if self._verbose:
                print(f"[ColaMem] ✅ Memory Mismatch Ranking Loss (Anti-Bypass) 已启用")
                print(f"[ColaMem]    - gamma (margin): {mmr_gamma}")
                print(f"[ColaMem]    - beta (最终目标): {mmr_beta}")
                if self.mmr_beta_warmup_steps > 0:
                    print(f"[ColaMem]    - beta_start (起始): {self.mmr_beta_start}")
                    print(f"[ColaMem]    - warmup_steps: {self.mmr_beta_warmup_steps}")
                    print(f"[ColaMem]    - Beta Warmup: {self.mmr_beta_start} → {mmr_beta} (线性增长)")
                print(f"[ColaMem]    - 功能: 防止 Solver 绕过 Memory 依靠语言先验生成")
        else:
            if self._verbose:
                print(f"[ColaMem] ⏭️  Memory Mismatch Ranking Loss 已禁用")
        
        self.print_stats()

    def _get_encoder_inner_model(self):
        """
        获取 encoder 内部的 Qwen3VLModel 实例。
        
        当使用 LoRA 时，self.encoder 被 PeftModel 包装：
          PeftModel.model -> Qwen3VLForConditionalGeneration.model -> Qwen3VLModel
        不使用 LoRA 时：
          Qwen3VLForConditionalGeneration.model -> Qwen3VLModel
        """
        if PEFT_AVAILABLE and isinstance(self.encoder, PeftModel):
            # LoRA 模式：PeftModel -> base_model -> model -> Qwen3VLForConditionalGeneration
            # 我们需要 Qwen3VLForConditionalGeneration.model (即 Qwen3VLModel)
            return self.encoder.model.model
        else:
            # 非 LoRA 模式：直接取 .model
            return self.encoder.model

    def _get_final_norm(self, encoder_inner):
        """
        获取 encoder 内部模型的 final RMSNorm 层。
        
        不同版本的 Qwen 模型结构不同：
        - Qwen2VL: Qwen2VLModel 直接有 .norm 属性
        - Qwen3VL: Qwen3VLModel 没有 .norm，final norm 在 .text_model.norm 中
          结构：Qwen3VLModel -> text_model (Qwen3VLTextModel) -> norm
        
        本方法按优先级依次尝试查找，确保兼容不同版本。
        """
        # 优先级 1：直接有 .norm（Qwen2VL 等旧版结构）
        if hasattr(encoder_inner, 'norm'):
            return encoder_inner.norm
        
        # 优先级 2：Qwen3VL 结构，norm 在 text_model 里
        if hasattr(encoder_inner, 'text_model') and hasattr(encoder_inner.text_model, 'norm'):
            return encoder_inner.text_model.norm
        
        # 优先级 3：尝试 language_model.norm（某些其他 VLM 的结构）
        if hasattr(encoder_inner, 'language_model') and hasattr(encoder_inner.language_model, 'norm'):
            return encoder_inner.language_model.norm
        
        # 优先级 4：遍历子模块，找最后一个名为 'norm' 的 RMSNorm/LayerNorm
        final_norm = None
        for name, module in encoder_inner.named_modules():
            # 只匹配顶层或一级子模块中名为 'norm' 的模块（排除 layer 内部的 norm）
            parts = name.split('.')
            if parts[-1] == 'norm' and len(parts) <= 2:
                final_norm = module
        
        if final_norm is not None:
            return final_norm
        
        raise AttributeError(
            f"无法在 {type(encoder_inner).__name__} 中找到 final norm 层。"
            f" 已检查路径: .norm, .text_model.norm, .language_model.norm"
            f" 子模块列表: {[n for n, _ in encoder_inner.named_children()]}"
        )

    def _enable_all_gradients(self):
        """启用所有参数的梯度（全参数微调）"""
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        self.mem_anchors.requires_grad = True
        self.work_tokens.requires_grad = True
        for p in self.projector.parameters():
            p.requires_grad = True
        self.residual_gate_logit.requires_grad = True
        self.memory_scale.requires_grad = True
    
    def _freeze_vision_encoder(self):
        """
        冻结 Vision Encoder，只训练其他部分
        
        优势：
        1. 避免 Vision 权重加载/保存问题
        2. 节省显存（冻结的参数不需要梯度）
        3. 加快训练速度
        4. 保留预训练的视觉能力
        
        冻结内容：
        - Encoder 的 Vision 模块 (encoder.model.visual)
        
        可训练内容：
        - Encoder 的 LLM 部分
        - Decoder 全部
        - Memory Anchors, Workspace, Projector, Memory Norm
        """
        if self._verbose:
            print(f"[ColaMem] 🔒 冻结 Vision Encoder...")
        
        # 1. 冻结 Encoder 的 Vision 模块
        # 注意：在 LoRA 模式下，get_peft_model() 已经冻结了基础模型参数，
        #       只有 LoRA 适配器参数是可训练的。这里不能把所有非-visual参数
        #       设为 requires_grad=True，否则会覆盖 LoRA 的冻结设置！
        vision_params = 0
        if self.use_lora:
            # LoRA 模式：只冻结 visual 参数，保留 LoRA 已设置的 requires_grad 状态
            for name, p in self.encoder.named_parameters():
                if 'visual' in name:
                    p.requires_grad = False
                    vision_params += p.numel()
                # 非 visual 参数保持 LoRA 设置的状态（LoRA 适配器可训练，基础权重冻结）
        else:
            # 非 LoRA 模式：冻结 visual，其余全部可训练
            for name, p in self.encoder.named_parameters():
                if 'visual' in name:
                    p.requires_grad = False
                    vision_params += p.numel()
                else:
                    p.requires_grad = True
        
        # 2. Decoder 可训练
        # LoRA 模式下保持 LoRA 已设置的 requires_grad 状态
        if not self.use_lora:
            for p in self.decoder.parameters():
                p.requires_grad = True
        
        # 3. 自定义参数可训练
        self.mem_anchors.requires_grad = True
        if self.use_workspace:
            self.work_tokens.requires_grad = True
        for p in self.projector.parameters():
            p.requires_grad = True
        self.residual_gate_logit.requires_grad = True
        self.memory_scale.requires_grad = True
        
        if self._verbose:
            print(f"[ColaMem] ✅ 已冻结 {vision_params/1e6:.2f}M Vision 参数")
    
    def _freeze_decoder(self):
        """
        [Phase 1 专用] 冻结 Decoder (Solver)，只训练 Compressor 和 Projector
        
        核心原理：
        - Phase 1 的目标是让 Projector 学会将 Latent Memory 映射到 Decoder 可理解的空间
        - 如果 Decoder 也参与训练，它可能会"变傻"来适应 Projector 的随机输出
        - 冻结 Decoder 强迫 Projector 去适应 Decoder，而不是反过来
        
        优势：
        1. 防止 Decoder 过拟合（变成复读机）
        2. 保留 Decoder 的预训练生成能力
        3. 大幅减少可训练参数 (~8.77B -> 0)
        4. 加快训练速度，节省显存
        
        冻结内容：
        - Decoder 全部参数
        
        可训练内容：
        - Encoder 的 LLM 部分（如果 freeze_vision_encoder=True，则不含 Vision）
        - Memory Anchors, Workspace, Projector, Memory Norm
        """
        if self._verbose:
            print(f"[ColaMem] 🔒 冻结 Decoder (Phase 1 模式)...")
        
        # 冻结 Decoder 参数
        # 注意：在 LoRA 模式下，如果仍然希望冻结 Decoder，需要冻结所有参数
        #       包括 LoRA 适配器参数（因为 Phase 1 冻结 Decoder 的目的是
        #       强迫 Projector 适应 Decoder，而不是让 Decoder 变化）
        decoder_params = 0
        for name, p in self.decoder.named_parameters():
            p.requires_grad = False
            decoder_params += p.numel()
        
        if self._verbose:
            print(f"[ColaMem] ✅ 已冻结 {decoder_params/1e9:.2f}B Decoder 参数")
            print(f"[ColaMem] 📋 Phase 1 可训练内容:")
            print(f"  - Encoder LLM 部分 (Compressor)")
            print(f"  - Projector (Latent Memory 投影)")
            print(f"  - Memory Anchors + Workspace")

    def load_base_vision_weights(self, base_model_path: str):
        """
        [Phase 1 关键修复] 从基础模型加载 Vision Encoder 权重
        
        问题背景：
        - from_pretrained 可能无法正确加载 Vision Encoder 权重
        - 导致 Vision Encoder 使用随机初始化的权重
        - Solver 在训练时学会忽略噪声输入，变成"复读机"
        
        修复方案：
        - 在训练开始前，手动从基础模型加载 Vision 权重
        - 使用"器官移植"策略直接替换整个 visual 模块
        
        Args:
            base_model_path: 原始 Qwen3VL 模型路径
        """
        if self._verbose:
            print(f"\n{'='*60}")
            print(f"🏥 [Vision 器官移植] 从基础模型加载 Vision 权重")
            print(f"{'='*60}")
            print(f"[Donor] 供体模型: {base_model_path}")
        
        # 1. 检查修复前的 Vision 权重状态
        encoder_visual = self._get_encoder_inner_model().visual
        before_std = encoder_visual.patch_embed.proj.weight.std().item()
        if self._verbose:
            print(f"[修复前] Encoder Vision Patch Embed Std: {before_std:.6f}")
        
        # 判断是否需要修复（随机初始化通常 Std < 0.03）
        if self._verbose:
            if before_std < 0.03:
                print(f"[诊断] Vision 权重疑似随机初始化！开始器官移植...")
            else:
                print(f"[诊断] Vision 权重看起来正常，但仍执行移植以确保安全...")
        
        # 2. 加载供体模型 (在 CPU 上，节省显存)
        if self._verbose:
            print(f"[Donor] 正在加载供体模型到 CPU...")
        from transformers import Qwen3VLForConditionalGeneration
        
        donor_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=self.encoder.dtype,
            device_map="cpu"
        )
        
        donor_std = donor_model.model.visual.patch_embed.proj.weight.std().item()
        if self._verbose:
            print(f"[Donor] 供体 Vision Patch Embed Std: {donor_std:.6f}")
        
        # 3. 执行器官移植：直接替换整个 visual 模块
        if self._verbose:
            print(f"🔄 正在执行 Vision 模块替换 (Encoder)...")
        
        # 获取目标设备
        target_device = self._get_encoder_inner_model().visual.patch_embed.proj.weight.device
        target_dtype = self._get_encoder_inner_model().visual.patch_embed.proj.weight.dtype
        
        # 替换 Encoder 的 Vision
        self._get_encoder_inner_model().visual = donor_model.model.visual.to(device=target_device, dtype=target_dtype)
        
        # 4. 验证修复结果
        after_std = self._get_encoder_inner_model().visual.patch_embed.proj.weight.std().item()
        if self._verbose:
            print(f"[修复后] Encoder Vision Patch Embed Std: {after_std:.6f}")
        
        if abs(after_std - donor_std) < 1e-6:
            if self._verbose:
                print(f"✅ [SUCCESS] Vision 器官移植成功！")
        else:
            print(f"❌ [FAILED] Vision 器官移植失败！权重不匹配！")
            raise RuntimeError("Vision 权重移植失败，请检查代码")
        
        # 5. 清理供体模型
        del donor_model
        torch.cuda.empty_cache()
        
        if self._verbose:
            print(f"{'='*60}\n")
    
    def get_custom_parameters(self):
        """
        获取自定义参数（mem_anchors, work_tokens, projector）
        
        这些参数需要单独处理以确保在 DeepSpeed ZeRO-3 下正确优化
        
        Returns:
            list: 包含所有自定义参数的列表
        """
        params = [self.mem_anchors]
        if self.use_workspace:
            params.append(self.work_tokens)
        params.extend(self.projector.parameters())
        return params
    
    def get_custom_named_parameters(self):
        """
        获取带名称的自定义参数
        
        Returns:
            Generator: (name, parameter) 元组的生成器
        """
        yield ('mem_anchors', self.mem_anchors)
        if self.use_workspace:
            yield ('work_tokens', self.work_tokens)
        for name, param in self.projector.named_parameters():
            yield (f'projector.{name}', param)
        yield ('residual_gate_logit', self.residual_gate_logit)
        yield ('memory_scale', self.memory_scale)
    
    def print_stats(self):
        """打印参数统计（仅在主进程 rank 0 打印）"""
        if not self._verbose:
            return
        
        # 获取基础模型（如果使用 LoRA，需要获取内部模型）
        encoder_base = self.encoder.base_model if (PEFT_AVAILABLE and isinstance(self.encoder, PeftModel)) else self.encoder
        decoder_base = self.decoder.base_model if (PEFT_AVAILABLE and isinstance(self.decoder, PeftModel)) else self.decoder
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        work_tokens_numel = self.work_tokens.numel() if self.use_workspace else 0
        custom_params = (
            self.mem_anchors.numel() + 
            work_tokens_numel + 
            sum(p.numel() for p in self.projector.parameters()) +
            self.residual_gate_logit.numel() +
            self.memory_scale.numel()
        )
        total_params = encoder_params + decoder_params + custom_params
        
        # 统计可训练参数
        trainable_encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        trainable_decoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        trainable_custom = self.mem_anchors.numel() if self.mem_anchors.requires_grad else 0
        if self.use_workspace and self.work_tokens.requires_grad:
            trainable_custom += self.work_tokens.numel()
        trainable_custom += sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        if self.residual_gate_logit.requires_grad:
            trainable_custom += self.residual_gate_logit.numel()
        if self.memory_scale.requires_grad:
            trainable_custom += self.memory_scale.numel()
        trainable_total = trainable_encoder + trainable_decoder + trainable_custom
        
        # Vision 参数统计（使用基础模型来查找 visual 参数）
        vision_params = sum(p.numel() for name, p in encoder_base.named_parameters() if 'visual' in name)
        frozen_vision = sum(p.numel() for name, p in encoder_base.named_parameters() if 'visual' in name and not p.requires_grad)
        
        print(f"[ColaMem] 参数统计:")
        print(f"  - Encoder: {encoder_params/1e9:.2f}B (可训练: {trainable_encoder/1e9:.2f}B)")
        print(f"  - Decoder: {decoder_params/1e9:.2f}B (可训练: {trainable_decoder/1e9:.2f}B)")
        print(f"  - Custom: {custom_params/1e6:.2f}M (可训练: {trainable_custom/1e6:.2f}M)")
        print(f"  - Total: {total_params/1e9:.2f}B (可训练: {trainable_total/1e9:.2f}B)")
        
        # LoRA 模式信息
        if self.use_lora:
            print(f"[ColaMem] 🎯 LoRA 微调模式:")
            
            # Encoder LoRA 配置
            if self.encoder_lora:
                print(f"  [Encoder LoRA]")
                print(f"    - r (rank): {self.encoder_lora_r}")
                print(f"    - alpha: {self.encoder_lora_alpha}")
                print(f"    - dropout: {self.encoder_lora_dropout}")
                print(f"    - target_modules: {self.encoder_lora_target_modules}")
            else:
                print(f"  [Encoder LoRA] 已禁用")
            
            # Decoder LoRA 配置
            if self.decoder_lora:
                print(f"  [Decoder LoRA]")
                print(f"    - r (rank): {self.decoder_lora_r}")
                print(f"    - alpha: {self.decoder_lora_alpha}")
                print(f"    - dropout: {self.decoder_lora_dropout}")
                print(f"    - target_modules: {self.decoder_lora_target_modules}")
            else:
                print(f"  [Decoder LoRA] 已禁用")
            
            # 统计 LoRA 参数
            lora_encoder_params = sum(p.numel() for name, p in self.encoder.named_parameters() if 'lora_' in name.lower())
            lora_decoder_params = sum(p.numel() for name, p in self.decoder.named_parameters() if 'lora_' in name.lower())
            print(f"  [参数统计]")
            print(f"    - Encoder LoRA 参数: {lora_encoder_params/1e6:.2f}M")
            print(f"    - Decoder LoRA 参数: {lora_decoder_params/1e6:.2f}M")
            print(f"    - 总 LoRA 参数: {(lora_encoder_params + lora_decoder_params)/1e6:.2f}M")
        
        # Vision 冻结状态
        print(f"[ColaMem] Vision Encoder 状态:")
        if self.freeze_vision_encoder:
            print(f"  - 🔒 已冻结: {frozen_vision/1e6:.2f}M 参数")
        else:
            print(f"  - 🔓 可训练: {vision_params/1e6:.2f}M 参数")
        
        # Decoder 冻结状态
        print(f"[ColaMem] Decoder (Solver) 状态:")
        if self.freeze_decoder:
            print(f"  - 🔒 已冻结: {decoder_params/1e9:.2f}B 参数 (Phase 1 模式)")
        elif self.use_lora and self.decoder_lora:
            print(f"  - 🎯 LoRA 微调: 仅训练 LoRA 适配器")
        else:
            print(f"  - 🔓 可训练: {trainable_decoder/1e9:.2f}B 参数")
        
        print(f"[ColaMem] 软约束配置:")
        print(f"  - 语义初始化: {'✅ 已启用（统一概念词池，无分组）' if self.use_semantic_init else '❌ 未启用'}")
        print(f"  - 正交正则化: {'✅ 已启用' if self.lambda_orth > 0 else '❌ 未启用'}")
        if self.lambda_orth > 0:
            print(f"    * λ_orth: {self.lambda_orth}")
    
    @staticmethod
    def _maybe_to_device(tensor, device):
        """安全地将 tensor 移动到指定设备"""
        if tensor is None:
            return None
        if tensor.device == device:
            return tensor
        return tensor.to(device)
    
    def _get_encoder_base_model(self):
        """
        获取 encoder 的基础模型（Qwen3VLForConditionalGeneration 实例）。
        
        当使用 LoRA 时，self.encoder 是 PeftModel：
          PeftModel -> base_model -> model (Qwen3VLForConditionalGeneration)
        不使用 LoRA 时：
          直接就是 Qwen3VLForConditionalGeneration
        
        注意：不能用 hasattr(self.encoder, 'base_model') 判断，
        因为 PreTrainedModel 本身也有 base_model property。
        """
        if PEFT_AVAILABLE and isinstance(self.encoder, PeftModel):
            return self.encoder.base_model.model
        else:
            return self.encoder
    
    def _get_encoder_text_model(self):
        """
        获取 encoder 内部的 Qwen3VLTextModel 实例。
        
        层级关系：
        - Qwen3VLForConditionalGeneration.model (Qwen3VLModel)
        - Qwen3VLModel.language_model (Qwen3VLTextModel)
        
        Qwen3VLTextModel 是真正的 text decoder，包含：
        - embed_tokens, layers, norm, rotary_emb
        - forward() 接受 visual_pos_masks + deepstack_visual_embeds
        """
        inner = self._get_encoder_inner_model()  # Qwen3VLModel
        return inner.language_model  # Qwen3VLTextModel
    
    def forward_compress(
        self, 
        input_ids, 
        attention_mask, 
        pixel_values=None, 
        image_grid_thw=None, 
        **kwargs
    ):
        """
        Phase 1: Compress (使用 Encoder)
        
        输入：Context + Images（input_ids 末尾已拼接 num_mem 个 <mem> token）
        输出：Latent Memory [batch, num_mem, hidden_size]
        
        核心设计（v5 - 手动复制 Qwen3VLModel.forward 内部逻辑，不修改源码）：
        =====================================================================
        由于 Qwen3VLModel.forward() 有 XOR 检查，不允许同时传 input_ids 和 
        inputs_embeds，但我们需要：
          - input_ids：用于 get_placeholder_mask、get_rope_index
          - inputs_embeds：已替换 <mem> 位置的 embedding
        
        方案：手动在此方法中执行 Qwen3VLModel.forward() 的内部逻辑：
        1. embed_tokens(input_ids) → inputs_embeds
        2. 替换 <mem> 位置为 mem_anchors
        3. get_image_features → masked_scatter（image fusion）
        4. get_placeholder_mask → visual_pos_masks + deepstack_visual_embeds
        5. get_rope_index（用 input_ids 计算 mRoPE）
        6. 调用 language_model.forward()（Qwen3VLTextModel）
        
        这样完全绕开 XOR 检查，同时保持完整的 mRoPE + image fusion + deepstack。
        =====================================================================
        """
        batch_size = input_ids.shape[0]
        
        # 1. 确保数据在正确设备
        embed_layer = self.encoder.get_input_embeddings()
        device = embed_layer.weight.device
        dtype = embed_layer.weight.dtype
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = self._maybe_to_device(pixel_values, device)
        image_grid_thw = self._maybe_to_device(image_grid_thw, device)
        
        # 2. 获取内部模型引用
        encoder_inner = self._get_encoder_inner_model()   # Qwen3VLModel
        text_model = self._get_encoder_text_model()        # Qwen3VLTextModel (= encoder_inner.language_model)
        
        # 3. 获取文本 Embedding
        inputs_embeds = encoder_inner.get_input_embeddings()(input_ids)
        
        # 4. 将 <mem> 占位符位置的 embedding 替换为 mem_anchors
        assert self.mem_token_id is not None, \
            "[ColaMem] mem_token_id 未设置！请确保 tokenizer 已注册 <mem> token"
        
        mem_mask = (input_ids == self.mem_token_id)  # [B, seq_len]
        mem_counts = mem_mask.sum(dim=1)
        assert (mem_counts == self.num_mem).all(), \
            f"[ColaMem] <mem> token 数量不一致！期望每个样本 {self.num_mem} 个，" \
            f"实际: {mem_counts.tolist()}"
        
        anchors = self.mem_anchors.to(dtype=dtype).expand(batch_size, -1, -1)  # [B, num_mem, H]
        mem_mask_3d = mem_mask.unsqueeze(-1).expand_as(inputs_embeds)  # [B, seq_len, H]
        inputs_embeds = inputs_embeds.masked_scatter(mem_mask_3d, anchors)
        
        # [调试] <mem> 位置验证（前3步 + 异常时打印）
        _step = kwargs.get('global_step', 0)
        _do_init_debug = self._verbose and (not self.training or _step < 3)
        if _do_init_debug:
            first_mem_pos = mem_mask[0].nonzero(as_tuple=True)[0][0].item()
            seq_len = input_ids.shape[1]
            at_end = (first_mem_pos == seq_len - self.num_mem)
            print(f"[Compress Step {_step}] <mem>: seq={seq_len}, "
                  f"起始pos={first_mem_pos}, 在末尾={'✅' if at_end else '❌'}")
        
        # =====================================================================
        # 5. 手动执行 Qwen3VLModel.forward() 的内部逻辑
        #    （对齐 modeling_qwen3_vl.py 中 Qwen3VLModel.forward 的完整流程）
        # =====================================================================
        
        # 5a. Image Fusion: get_image_features → masked_scatter
        image_mask = None
        visual_pos_masks = None
        deepstack_visual_embeds = None
        
        if pixel_values is not None:
            # 直接内联 get_image_features 逻辑
            # 旧版 Qwen3VLVisionModel.forward() 返回: (merged_hidden_states, deepstack_feature_lists)
            _pixel_values = pixel_values.type(encoder_inner.visual.dtype)
            _merged_hidden_states, _deepstack_feature_lists = encoder_inner.visual(
                _pixel_values, grid_thw=image_grid_thw
            )
            split_sizes = (image_grid_thw.prod(-1) // encoder_inner.visual.spatial_merge_size**2).tolist()
            image_embeds = list(torch.split(_merged_hidden_states, split_sizes))
            deepstack_image_embeds = _deepstack_feature_lists
            
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            
            # [调试] Image Fusion 数量匹配（前3步打印）
            if _do_init_debug:
                n_img_tokens = (input_ids == encoder_inner.config.image_token_id).sum().item()
                n_img_embeds = image_embeds.shape[0]
                match = '✅' if n_img_tokens == n_img_embeds else '❌'
                print(f"[Compress Step {_step}] Image: tokens={n_img_tokens}, "
                      f"embeds={n_img_embeds} {match}")
            
            # get_placeholder_mask 返回 (image_mask, video_mask)
            image_mask, _ = encoder_inner.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            # image fusion: 将 image_embeds 注入到 image_token 位置
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            # 构造 deepstack 参数
            image_mask_2d = image_mask[..., 0]  # [B, seq_len]
            visual_pos_masks = image_mask_2d
            deepstack_visual_embeds = deepstack_image_embeds
        
        # 5b. 计算 mRoPE 3D position_ids（需要 input_ids）
        # 重置 rope_deltas 缓存，确保 compress 和 solve 的 position_ids 独立
        encoder_inner.rope_deltas = None
        position_ids, rope_deltas = encoder_inner.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        # 再次重置 rope_deltas，避免影响后续的 forward_solve
        encoder_inner.rope_deltas = None
        
        # [调试] position_ids 验证
        if position_ids is None and self._verbose:
            print(f"[Compress Step {_step}] ⚠️ position_ids=None! mRoPE 将由 TextModel fallback")
        elif _do_init_debug:
            print(f"[Compress Step {_step}] mRoPE: shape={position_ids.shape}")
        
        # 5c. 调用 Qwen3VLTextModel.forward()
        #     只传 inputs_embeds，不传 input_ids → 不触发任何 XOR 检查
        #     position_ids、visual_pos_masks、deepstack_visual_embeds 由我们手动构造
        # Qwen3VLTextModel.forward 返回 BaseModelOutputWithPast
        encoder_outputs = text_model(
            input_ids=None,                  # 不传 input_ids，避免重复 embed_tokens
            inputs_embeds=inputs_embeds,     # 已替换 <mem> + 已融合 image
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=False,
        )
        
        # 6. 提取最后 num_mem 个 token 的隐藏状态
        # Qwen3VLTextModel.forward 返回 BaseModelOutputWithPast
        #   - last_hidden_state: 经过 TextModel.norm() 的最终隐藏状态
        # 兼容 tuple 返回: tuple[0] = last_hidden_state
        if isinstance(encoder_outputs, tuple):
            _last_hidden = encoder_outputs[0]
        else:
            _last_hidden = encoder_outputs.last_hidden_state
        raw_memory = _last_hidden[:, -self.num_mem:, :]
        
        # [调试] raw_memory 健康度（前3步 + 异常时打印）
        if _do_init_debug or (self._verbose and self.training and _step % 200 == 0):
            with torch.no_grad():
                rm = raw_memory.float()
                rm_nan = torch.isnan(rm).any().item()
                rm_inf = torch.isinf(rm).any().item()
                if rm_nan or rm_inf:
                    print(f"[Compress Step {_step}] 🚨 raw_memory 含 {'NaN' if rm_nan else ''}{'Inf' if rm_inf else ''}！")
                elif _do_init_debug:
                    print(f"[Compress Step {_step}] raw_memory: "
                          f"mean={rm.mean().item():.4f}, std={rm.std().item():.4f}")
        
        # 8. 通过 Projector 投影（无残差连接，与 Phase 0.1 训练一致）
        # 注意：Phase 0.1 训练时未使用残差连接，后续训练如需启用残差请改回：
        #   projected_memory = self.projector(raw_memory) + raw_memory
        projected_memory = self.projector(raw_memory)
        
        # 9. 残差连接（fp32 + 可学习门控）
        # 核心改进：让每个 slot 通过可学习的 gate 自主决定
        # 「保留多少 anchor 身份 vs 多少 encoder 输出」
        # 这样 64 个 slot 即使 projected_memory 趋同，
        # 也能通过不同的 anchor 身份保持区分度
        anchors_for_residual = self.mem_anchors.to(dtype=dtype).expand(batch_size, -1, -1)
        
        # 在 fp32 下做残差，避免 bf16 下大小值相加精度丢失
        proj_f32 = projected_memory.float()
        anc_f32 = anchors_for_residual.float()
        gate = torch.sigmoid(self.residual_gate_logit.float())  # [1, num_mem, 1]
        
        # gate≈0 时以 projected_memory 为主，gate≈1 时 anchor 身份注入更强
        latent_memory = (1.0 - gate) * proj_f32 + gate * anc_f32
        latent_memory = latent_memory.to(dtype)  # 转回原始精度
        
        # 10. [方案A] 只用可学习 memory_scale 匹配 decoder 尺度，不再做 RMSNorm
        # 移除 memory_norm 避免多余的方差压缩导致 slot collapse
        latent_memory = latent_memory * self.memory_scale
        
        # [调试] Slot 多样性 + gate/scale 周期打印（每50步）
        if self._verbose and self.training and (_step < 3 or _step % 50 == 0):
            with torch.no_grad():
                gate_val = torch.sigmoid(self.residual_gate_logit.float()).mean().item()
                scale_val = self.memory_scale.item()
                mem_f = latent_memory.float()
                mem_n = torch.nn.functional.normalize(mem_f, p=2, dim=-1)
                cos_sim = torch.bmm(mem_n, mem_n.transpose(1, 2))
                N = self.num_mem
                offdiag_mask = ~torch.eye(N, dtype=torch.bool, device=cos_sim.device).unsqueeze(0)
                offdiag = cos_sim[offdiag_mask.expand_as(cos_sim)]
                print(f"[Compress Step {_step}] gate={gate_val:.4f} scale={scale_val:.4f} | "
                      f"slot_cos: mean={offdiag.mean().item():.4f}, max={offdiag.max().item():.4f}")
                # [P0-B] 跨样本 Memory 相似度诊断
                # 检测不同图片的 memory 是否正在坍塌为同一个常量向量
                if batch_size > 1:
                    mem_pooled = torch.nn.functional.normalize(
                        latent_memory.float().mean(dim=1), dim=-1)  # [B, H]
                    batch_sim = mem_pooled @ mem_pooled.T  # [B, B]
                    batch_offdiag = batch_sim[~torch.eye(
                        batch_size, dtype=torch.bool, device=batch_sim.device)]
                    print(f"[BatchMemCos] mean={batch_offdiag.mean().item():.4f}, "
                          f"max={batch_offdiag.max().item():.4f}")
        
        # 11. 计算正交正则化 Loss
        # orth 约束作用在 anchor 参数本身（self.mem_anchors），
        # 而非 raw_memory（Encoder 输出，形状 [B, num_mem, hidden]）。
        # 原因：
        #   - anchor 参数是全局共享的 [1, num_mem, hidden]，orth 约束有明确语义
        #   - raw_memory 是图片相关的 batch 输出，对其做 mean(dim=0) 再 orth
        #     会把不同样本的 memory 混在一起，梯度互相干扰
        #   - 对 latent_memory 强制正交会破坏视觉细节间必要的相关性
        orth_loss = None
        if self.orth_regularizer is not None and self.training:
            # 传入 global_step 避免 gradient accumulation 下 warmup 比预期快结束
            orth_loss = self.orth_regularizer.compute_loss(self.mem_anchors, step=kwargs.get('global_step', None))
        
        # 12. 输出 Slot 去相关（Soft Repulse 正则）
        # 轻量正则：惩罚 batch 内同一样本的 64 个 slot 输出之间过高的相似度
        # 与 orth_regularizer（作用在 anchor 参数）不同，这里作用在 latent_memory 输出
        # 权重从 1e-4 起步，非常轻，不会干扰正常训练
        slot_repulse_loss = None
        if self.training:
            # 在 fp32 下计算，避免 bf16 数值不稳
            # 仅用于 loss 的版本需要保留梯度
            mem_grad = latent_memory.float()  # [B, num_mem, H]
            mem_norm = torch.nn.functional.normalize(mem_grad, p=2, dim=-1)
            # 每个样本内部的 slot 间余弦相似度: [B, num_mem, num_mem]
            slot_sim = torch.bmm(mem_norm, mem_norm.transpose(1, 2))
            # 只取 off-diagonal
            B_r, N_r = slot_sim.shape[0], slot_sim.shape[1]
            offdiag_mask = ~torch.eye(N_r, dtype=torch.bool, device=slot_sim.device).unsqueeze(0)
            offdiag = slot_sim[offdiag_mask.expand(B_r, -1, -1)]
            # [P1-A] 增强 slot repulse：阈值 0.5→0.7，权重 1e-4→5e-3，加 warmup
            # Hinge: 只惩罚 |sim| > 0.7 的对（比 0.5 更严格，推动 slot 更好分化）
            slot_repulse_threshold = 0.7
            slot_repulse_target_weight = 5e-3  # 目标权重（比 1e-4 提升 50 倍）
            slot_repulse_warmup_steps = 200     # warmup 步数
            current_step = kwargs.get('global_step', 0)
            # warmup: 前 200 步线性从 0 增长到目标权重
            if current_step < slot_repulse_warmup_steps:
                slot_repulse_weight = slot_repulse_target_weight * (current_step / slot_repulse_warmup_steps)
            else:
                slot_repulse_weight = slot_repulse_target_weight
            penalty = torch.relu(offdiag.abs() - slot_repulse_threshold)
            slot_repulse_loss = slot_repulse_weight * (penalty ** 2).mean()
        
        # 缓存中间量，供 separability probe 定位坍塌层
        # 只在 training 模式下缓存，避免推理时占用显存
        if self.training:
            self._last_raw_memory = raw_memory.detach()
            self._last_projected_memory = projected_memory.detach()
            # _last_latent_memory 在 two_phase 的 forward 中缓存（memory_scale 之后的最终值）
        
        # 合并 orth_loss 和 slot_repulse_loss
        total_aux_loss = None
        if orth_loss is not None:
            total_aux_loss = orth_loss
        if slot_repulse_loss is not None:
            total_aux_loss = (total_aux_loss + slot_repulse_loss) if total_aux_loss is not None else slot_repulse_loss
        
        return latent_memory, total_aux_loss
    

    
    def enable_debug(self, enable=True):
        """启用/禁用调试输出"""
        self.debug_forward = enable
        self._debug_step = 0
        print(f"[ColaMem] 调试模式: {'启用' if enable else '禁用'}")
    
    def _debug_print_solve_inputs(self, latent_memory, query_input_ids, query_attention_mask,
                                    query_labels, answer_input_ids, answer_attention_mask, 
                                    answer_labels, workspace, inputs_embeds, total_mask, final_labels):
        """
        打印 forward_solve 的关键监控信息（简化版）
        
        只输出关键验证信息，不输出具体数据内容
        """
        self._debug_step += 1
        
        batch_size = latent_memory.shape[0]
        query_len = query_input_ids.shape[1]
        answer_len = answer_input_ids.shape[1]
        total_len = self.num_mem + query_len + self.num_work + answer_len
        
        # 位置边界
        mem_end = self.num_mem
        query_end = mem_end + query_len
        work_end = query_end + self.num_work
        answer_end = work_end + answer_len
        
        print(f"\n{'='*70}")
        print(f"🔍 Step {self._debug_step} - forward_solve 监控")
        print(f"{'='*70}")
        
        # 1. 形状摘要
        print(f"📊 形状: batch={batch_size}, total_seq={inputs_embeds.shape[1]}")
        print(f"   Memory={self.num_mem}, Query={query_len}, Work={self.num_work}, Answer={answer_len}")
        
        # 2. 拼接顺序验证
        shape_ok = (inputs_embeds.shape[1] == total_len)
        print(f"🔗 拼接: [0:{mem_end}|{mem_end}:{query_end}|{query_end}:{work_end}|{work_end}:{answer_end}] {'✅' if shape_ok else '❌'}")
        
        # 3. Labels 验证（关键）
        labels_sample = final_labels[0]
        mem_ignore = (labels_sample[:mem_end] == -100).sum().item()
        query_ignore = (labels_sample[mem_end:query_end] == -100).sum().item()
        work_ignore = (labels_sample[query_end:work_end] == -100).sum().item()
        answer_valid = (labels_sample[work_end:answer_end] != -100).sum().item()
        
        mem_ok = (mem_ignore == self.num_mem)
        query_ok = (query_ignore == query_len)
        work_ok = (work_ignore == self.num_work)
        answer_ok = (answer_valid > 0)
        
        print(f"🏷️ Labels: Mem={mem_ignore}/{self.num_mem}{'✅' if mem_ok else '❌'} | "
              f"Query={query_ignore}/{query_len}{'✅' if query_ok else '❌'} | "
              f"Work={work_ignore}/{self.num_work}{'✅' if work_ok else '❌'} | "
              f"Answer={answer_valid}/{answer_len}{'✅' if answer_ok else '❌'}")
        
        # 4. Attention Mask 验证
        mask_sample = total_mask[0]
        total_visible = mask_sample.sum().item()
        query_visible = mask_sample[mem_end:query_end].sum().item()
        answer_visible = mask_sample[work_end:answer_end].sum().item()
        
        print(f"👁️ Mask: total_visible={total_visible}, query_visible={query_visible}, answer_visible={answer_visible}")
        
        # 5. 总体状态
        all_ok = mem_ok and query_ok and work_ok and answer_ok and shape_ok
        if all_ok:
            print(f"✅ 数据流验证通过")
        else:
            print(f"❌ 数据流存在问题，请检查!")
            if not mem_ok:
                print(f"   - Memory 部分有 {self.num_mem - mem_ignore} 个 token 会计算 loss")
            if not query_ok:
                print(f"   - Query 部分有 {query_len - query_ignore} 个 token 会计算 loss")
            if not work_ok:
                print(f"   - Workspace 部分有 {self.num_work - work_ignore} 个 token 会计算 loss")
            if not answer_ok:
                print(f"   - Answer 部分没有任何 token 计算 loss")
        
        print(f"{'='*70}\n")
    
    def forward_solve(
        self, 
        latent_memory, 
        query_input_ids,
        query_attention_mask,
        query_labels,
        answer_input_ids,
        answer_attention_mask,
        answer_labels,
        **kwargs
    ):
        """
        Phase 2: Solve (使用 Decoder)
        
        输入：Latent Memory + Query + Answer（分开传入）
        输出：CausalLMOutputWithPast (包含 loss)
        
        拼接顺序: [Latent Memory, Query, Workspace, Answer]
        
        关键设计：
        - Query 和 Answer 在 Data Engine 层分开
        - Workspace 插入到 Query 和 Answer 之间
        - Answer 可以通过 Causal Attention "看到" Workspace
        - 梯度能正确从 Answer 回传到 Workspace 和 Memory
        """
        batch_size = latent_memory.shape[0]
        
        # 1. 确保数据在正确设备
        embed_layer = self.decoder.get_input_embeddings()
        device = embed_layer.weight.device
        dtype = embed_layer.weight.dtype
        
        latent_memory = latent_memory.to(device=device, dtype=dtype)
        query_input_ids = query_input_ids.to(device)
        query_attention_mask = query_attention_mask.to(device)
        query_labels = query_labels.to(device)
        answer_input_ids = answer_input_ids.to(device)
        answer_attention_mask = answer_attention_mask.to(device)
        answer_labels = answer_labels.to(device)
        
        # 2. 获取文本 Embedding
        query_embeds = embed_layer(query_input_ids)
        answer_embeds = embed_layer(answer_input_ids)
        
        # 2.5 Answer-Prefix Dropout (Anti-Bypass 增强)
        # 训练时随机将 answer 前 K 个 token 的 embedding 置零，
        # 迫使 decoder 不能靠「看到答案开头就猜出后续」来绕过 memory。
        # 这是防止 bypass 最优先级最高的手段。
        # 注意：MMR 的 neg forward 必须禁用 prefix dropout，否则 pos/neg 的 dropout 决策不一致
        # 会导致 ~25% 概率产生反向 gap（L_pos > L_neg 或反之），污染 MMR 信号。
        _mmr_neg_forward = kwargs.get('_mmr_neg_forward', False)
        if self.training and self.use_answer_prefix_dropout and not _mmr_neg_forward:
            if torch.rand(1).item() < self.answer_prefix_dropout_prob:
                ans_len = answer_embeds.shape[1]
                prefix_k = max(1, int(ans_len * self.answer_prefix_dropout_ratio))
                # 将前 prefix_k 个 token 的 embedding 置零
                answer_embeds = answer_embeds.clone()
                answer_embeds[:, :prefix_k, :] = 0.0
        
        # 3. 准备 Workspace Tokens（Phase 1 可禁用）
        if self.use_workspace:
            workspace = self.work_tokens.to(dtype=dtype).expand(batch_size, -1, -1)
            # 4. 拼接: [Latent Memory, Query, Workspace, Answer]
            inputs_embeds = torch.cat([latent_memory, query_embeds, workspace, answer_embeds], dim=1)
            # 5. 构造 Attention Mask
            memory_mask = torch.ones((batch_size, self.num_mem), device=device, dtype=query_attention_mask.dtype)
            workspace_mask = torch.ones((batch_size, self.num_work), device=device, dtype=query_attention_mask.dtype)
            total_mask = torch.cat([memory_mask, query_attention_mask, workspace_mask, answer_attention_mask], dim=1)
            # 6. 构造 Labels
            memory_labels = torch.full((batch_size, self.num_mem), -100, device=device, dtype=query_labels.dtype)
            workspace_labels = torch.full((batch_size, self.num_work), -100, device=device, dtype=query_labels.dtype)
            final_labels = torch.cat([memory_labels, query_labels, workspace_labels, answer_labels], dim=1)
        else:
            # Phase 1 禁用 workspace：简化拼接
            # 4. 拼接: [Latent Memory, Query, Answer]（无 Workspace）
            inputs_embeds = torch.cat([latent_memory, query_embeds, answer_embeds], dim=1)
            workspace = None  # 用于调试输出
            # 5. 构造 Attention Mask
            memory_mask = torch.ones((batch_size, self.num_mem), device=device, dtype=query_attention_mask.dtype)
            total_mask = torch.cat([memory_mask, query_attention_mask, answer_attention_mask], dim=1)
            # 6. 构造 Labels
            memory_labels = torch.full((batch_size, self.num_mem), -100, device=device, dtype=query_labels.dtype)
            final_labels = torch.cat([memory_labels, query_labels, answer_labels], dim=1)
        
        # 调试输出（如果启用）
        if self.debug_forward:
            self._debug_print_solve_inputs(
                latent_memory=latent_memory,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                query_labels=query_labels,
                answer_input_ids=answer_input_ids,
                answer_attention_mask=answer_attention_mask,
                answer_labels=answer_labels,
                workspace=workspace,
                inputs_embeds=inputs_embeds,
                total_mask=total_mask,
                final_labels=final_labels
            )
        
        # 7. 通过 Decoder 推理
        # 过滤掉自定义参数（如 global_step），避免传入 Qwen3VL forward 报错
        decoder_kwargs = {k: v for k, v in kwargs.items() if k not in ('global_step', '_mmr_neg_forward')}
        # Qwen3VLForConditionalGeneration.forward 返回 Qwen3VLCausalLMOutputWithPast
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=total_mask,
            labels=final_labels,
            **decoder_kwargs
        )
        # 兼容 tuple 返回: 如果是 tuple，转成 dict 以便后续访问 .loss 等属性
        if isinstance(outputs, tuple):
            from transformers.modeling_outputs import CausalLMOutputWithPast
            outputs = CausalLMOutputWithPast(
                loss=outputs[0] if len(outputs) > 0 else None,
                logits=outputs[1] if len(outputs) > 1 else None,
            )
        
        return outputs
    
    def forward(
        self,
        compress_input_ids=None,
        compress_attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        query_input_ids=None,
        query_attention_mask=None,
        query_labels=None,
        answer_input_ids=None,
        answer_attention_mask=None,
        answer_labels=None,
        mode='two_phase',
        **kwargs
    ):
        """
        统一 forward 入口，兼容标准 Trainer
        
        支持两种模式：
        1. mode='two_phase' (默认): 完整的两阶段训练
           - 自动执行 compress -> solve
           - 输出: CausalLMOutputWithPast (包含 loss)
           - 用于: 标准 Trainer（推荐）
           
        2. mode='compress' 或 'solve': 单阶段模式
           - 用于: 自定义训练循环或推理
        
        Args:
            compress_input_ids: Compress 阶段的 input_ids
            compress_attention_mask: Compress 阶段的 attention_mask
            pixel_values: 图像像素值
            image_grid_thw: 图像网格信息
            query_input_ids: Query 部分的 input_ids
            query_attention_mask: Query 部分的 attention_mask
            query_labels: Query 部分的 labels（通常为 -100）
            answer_input_ids: Answer 部分的 input_ids
            answer_attention_mask: Answer 部分的 attention_mask
            answer_labels: Answer 部分的 labels（用于计算 loss）
            mode: 'two_phase' | 'compress' | 'solve'
        
        Returns:
            - mode='two_phase': CausalLMOutputWithPast (包含 loss)
            - mode='compress': latent_memory tensor
            - mode='solve': CausalLMOutputWithPast (包含 loss)
        
        拼接顺序: [Latent Memory, Query, Workspace, Answer]
        
        关键设计：
        - Workspace 插入到 Query 和 Answer 之间
        - Answer 可以通过 Causal Attention "看到" Workspace
        - 梯度能正确从 Answer 回传到 Workspace 和 Memory
        """
        if mode == 'two_phase':
            # 完整的两阶段训练（推荐用于标准 Trainer）
            
            # Phase 1: Compress
            latent_memory, aux_loss = self.forward_compress(
                input_ids=compress_input_ids,
                attention_mask=compress_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs
            )
            
            # Phase 2: Solve (正例)
            outputs = self.forward_solve(
                latent_memory=latent_memory,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                query_labels=query_labels,
                answer_input_ids=answer_input_ids,
                answer_attention_mask=answer_attention_mask,
                answer_labels=answer_labels,
                **kwargs
            )
            
            # 缓存 latent_memory 供外部 probe 使用（detach，不影响计算图）
            self._last_latent_memory = latent_memory.detach()
            
            # ===============================================================
            # [重要修复] aux_loss 不再提前加入 outputs.loss
            # 之前的 bug：aux_loss 只加在了 loss_pos 中却没加在 loss_neg 中，
            # 导致 MMR 比较不公平（L_pos 被系统性抬高 → gap 偏小 → 更易误判 BYPASS）
            # 修复：MMR 的 pos/neg 只比较纯 CE loss，aux_loss 在所有路径最后统一加回
            # ===============================================================
            # 保存纯 CE loss 用于 MMR 比较（不包含 aux_loss）
            loss_ce_pos = outputs.loss  # 纯 CE loss
            
            # ===============================================================
            # Memory Mismatch Ranking Loss (Anti-Bypass)
            # ===============================================================
            # 核心思想：如果在"错误 Memory"下也能把 Caption 写得很顺，
            # 说明 Solver 在靠语言先验"瞎编"；Ranking Loss 会直接惩罚这种行为
            # 
            # L_rank = max(0, gamma + L_pos - L_neg)
            # 目标：让 L_neg 越大越好 (Confusion)，让 L_pos 越小越好
            # ===============================================================
            batch_size = latent_memory.shape[0]
            if self.use_mmr_loss and self.training and batch_size > 1 and outputs.loss is not None:
                # 构建负例 Memory：在 Batch 维度上滚动位移
                # e.g., Sample 0 -> Memory 1, Sample 1 -> Memory 2 ...
                # 不 detach：让 Encoder/Projector 也参与对比学习，
                # 学着把不同图片的 Memory 拉得更远
                latent_memory_neg = torch.roll(latent_memory, shifts=1, dims=0)
                
                # 负例 forward（使用错误的 Memory）
                # [修复] 传入 _mmr_neg_forward=True，禁用 answer prefix dropout
                # 确保 pos/neg 的比较条件一致，避免 ~25% 概率产生虚假 gap
                neg_kwargs = dict(kwargs)
                neg_kwargs['_mmr_neg_forward'] = True
                outputs_neg = self.forward_solve(
                    latent_memory=latent_memory_neg,
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    query_labels=query_labels,
                    answer_input_ids=answer_input_ids,
                    answer_attention_mask=answer_attention_mask,
                    answer_labels=answer_labels,
                    **neg_kwargs
                )
                
                # [修复] MMR 只比较纯 CE loss，不包含 aux_loss
                loss_pos = loss_ce_pos  # 纯 CE loss（不含 aux_loss）
                loss_neg = outputs_neg.loss  # 纯 CE loss
                
                # ====== 救火检查 4：L_pos / L_neg / L_zero 诊断 bypass 程度 ======
                # L_pos: 正确 memory 下的 loss（越低越好）
                # L_neg: 错误 memory 下的 loss（应该远大于 L_pos）
                # gap = L_neg - L_pos：gap 越大说明 decoder 越依赖 memory
                # 如果 gap ≈ 0，说明 decoder 在 bypass memory！
                with torch.no_grad():
                    gap = (loss_neg - loss_pos).item()
                    ratio = loss_neg.item() / (loss_pos.item() + 1e-8)
                    # 每 20 步打印一次诊断
                    mmr_diag_step = kwargs.get('global_step', self._mmr_step)
                    if mmr_diag_step % 20 == 0 and self._verbose:
                        bypass_flag = '🚨 BYPASS' if abs(gap) < 0.05 else ('⚠️ 弱依赖' if gap < 0.2 else '✅ 依赖memory')
                        print(f"[MMR诊断] Step {mmr_diag_step}: L_pos={loss_pos.item():.4f}, L_neg={loss_neg.item():.4f}, "
                              f"gap={gap:.4f}, ratio={ratio:.3f} {bypass_flag}")
                        
                        # [P0-A] L_zero / L_anchor 诊断：区分"memory坍塌"还是"decoder忽略"
                        # L_zero: 把 memory 全置零，测试 decoder 是否完全不依赖 memory
                        # L_anchor: 把 memory 换成纯 anchors（不含图像信息），测试是否退化为常量提示
                        # 解读：
                        #   L_pos ≈ L_neg ≈ L_zero   → decoder 彻底无视 memory
                        #   L_pos ≈ L_neg << L_zero, L_anchor ≈ L_pos → memory 退化为常量提示
                        #   L_pos << L_neg, L_zero 很高 → memory 正常携带区分信息（理想状态）
                        try:
                            latent_zero = torch.zeros_like(latent_memory)
                            out_zero = self.forward_solve(
                                latent_memory=latent_zero,
                                query_input_ids=query_input_ids,
                                query_attention_mask=query_attention_mask,
                                query_labels=query_labels,
                                answer_input_ids=answer_input_ids,
                                answer_attention_mask=answer_attention_mask,
                                answer_labels=answer_labels,
                                **kwargs
                            )
                            L_zero = out_zero.loss.item()
                            
                            latent_anchor = (
                                self.mem_anchors.to(device=latent_memory.device, dtype=latent_memory.dtype)
                                .expand(batch_size, -1, -1) * self.memory_scale
                            )
                            out_anchor = self.forward_solve(
                                latent_memory=latent_anchor,
                                query_input_ids=query_input_ids,
                                query_attention_mask=query_attention_mask,
                                query_labels=query_labels,
                                answer_input_ids=answer_input_ids,
                                answer_attention_mask=answer_attention_mask,
                                answer_labels=answer_labels,
                                **kwargs
                            )
                            L_anchor = out_anchor.loss.item()
                            
                            print(f"[Bypass诊断] L_zero={L_zero:.4f}, L_anchor={L_anchor:.4f} | "
                                  f"L_pos={loss_pos.item():.4f}, L_neg={loss_neg.item():.4f}")
                            # 自动判断 bypass 类型
                            if abs(loss_pos.item() - L_zero) < 0.1:
                                print(f"  → 🚨 decoder 彻底忽略 memory (L_pos≈L_zero)")
                            elif abs(loss_pos.item() - L_anchor) < 0.1 and L_zero - loss_pos.item() > 0.3:
                                print(f"  → ⚠️ memory 退化为常量提示 (L_pos≈L_anchor, L_zero明显更高)")
                            elif loss_neg.item() - loss_pos.item() > 0.3:
                                print(f"  → ✅ memory 正常携带区分信息")
                            else:
                                print(f"  → ❓ 中间状态，需继续观察")
                        except Exception as e:
                            print(f"[Bypass诊断] 诊断失败: {e}")
                
                # Ranking Loss: max(0, gamma + L_pos - L_neg)
                # 如果 L_pos 和 L_neg 接近（说明 Solver 不看 Memory），Loss 会很大
                # 如果 L_neg >> L_pos（说明 Solver 依赖 Memory），Loss 会很小
                loss_rank = torch.relu(self.mmr_gamma + loss_pos - loss_neg)
                
                # Beta Warmup: 从 mmr_beta_start 线性增长到 mmr_beta
                # 让模型在训练初期先稳定学习基本对齐，后期再加大惩罚力度
                # 优先使用外部传入的 global_step，避免 gradient accumulation 下 warmup 比预期快结束
                mmr_current_step = kwargs.get('global_step', None)
                if mmr_current_step is None:
                    mmr_current_step = self._mmr_step
                    self._mmr_step += 1  # 仅在未传入外部 step 时使用内部计数器
                
                if self.mmr_beta_warmup_steps > 0 and mmr_current_step < self.mmr_beta_warmup_steps:
                    progress = mmr_current_step / self.mmr_beta_warmup_steps
                    current_beta = self.mmr_beta_start + (self.mmr_beta - self.mmr_beta_start) * progress
                else:
                    current_beta = self.mmr_beta
                
                # 更新总 Loss
                # [修复] loss_pos 是纯 CE loss，aux_loss 最后统一添加
                total_loss = loss_pos + current_beta * loss_rank
                
                # 最后统一添加 aux_loss（对 pos/neg 是同一个常数项，不应参与 MMR 比较）
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss
                
                outputs.loss = total_loss
                
                # 调试打印（可选）
                if self.debug_forward:
                    warmup_info = f"beta: {current_beta:.3f} (step {mmr_current_step}/{self.mmr_beta_warmup_steps})" if self.mmr_beta_warmup_steps > 0 else f"beta: {current_beta:.3f}"
                    aux_info = f", aux={aux_loss.item():.4f}" if aux_loss is not None else ""
                    print(f"[MMR-Loss] CE_pos: {loss_pos.item():.4f}, CE_neg: {loss_neg.item():.4f}, "
                          f"L_rank: {loss_rank.item():.4f}, {warmup_info}{aux_info}, Total: {total_loss.item():.4f}")
            else:
                # 非 MMR 路径：也要将 aux_loss 加回
                if aux_loss is not None and outputs.loss is not None:
                    outputs.loss = outputs.loss + aux_loss
            
            return outputs
        
        elif mode == 'compress':
            # 单独的 Compress 阶段
            latent_memory, _aux_loss = self.forward_compress(
                input_ids=compress_input_ids,
                attention_mask=compress_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs
            )
            # 推理模式下不返回 aux_loss
            return latent_memory
        
        elif mode == 'solve':
            # 单独的 Solve 阶段（需要提供 latent_memory）
            latent_memory = kwargs.get('latent_memory')
            if latent_memory is None:
                raise ValueError("mode='solve' requires 'latent_memory' argument")
            return self.forward_solve(
                latent_memory=latent_memory,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                query_labels=query_labels,
                answer_input_ids=answer_input_ids,
                answer_attention_mask=answer_attention_mask,
                answer_labels=answer_labels,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'two_phase', 'compress', or 'solve'")
    
    def _is_deepspeed_zero3_enabled(self):
        """检查是否启用了 DeepSpeed ZeRO-3"""
        try:
            import deepspeed
            if hasattr(self, 'encoder') and hasattr(self.encoder, 'model'):
                # 检查参数是否被 ZeRO-3 分区
                for param in self.encoder.parameters():
                    if hasattr(param, 'ds_id'):
                        return True
            return False
        except ImportError:
            return False
    
    def _save_model_zero3(self, model, save_dir, model_name="model"):
        """在 DeepSpeed ZeRO-3 下保存模型"""
        import deepspeed
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取当前进程的 rank
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        
        # 收集所有参数并保存 state_dict
        state_dict = {}
        for name, param in model.named_parameters():
            # 使用 GatheredParameters 收集分布在各 GPU 上的参数
            with GatheredParameters([param], modifier_rank=0):
                if rank == 0:
                    state_dict[name] = param.detach().cpu().clone()
        
        # 只在 rank 0 上保存
        if rank == 0:
            # 保存 config
            if hasattr(model, 'config'):
                model.config.save_pretrained(save_dir)
            
            # 保存 state_dict（使用 safetensors 格式）
            try:
                from safetensors.torch import save_file
                # 分片保存大模型
                total_size = sum(p.numel() * p.element_size() for p in state_dict.values())
                max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB per shard
                
                if total_size > max_shard_size:
                    # 分片保存
                    shard_idx = 0
                    current_shard = {}
                    current_size = 0
                    index = {"metadata": {"total_size": total_size}, "weight_map": {}}
                    
                    for name, tensor in state_dict.items():
                        tensor_size = tensor.numel() * tensor.element_size()
                        if current_size + tensor_size > max_shard_size and current_shard:
                            shard_file = f"model-{shard_idx+1:05d}-of-XXXXX.safetensors"
                            save_file(current_shard, os.path.join(save_dir, shard_file))
                            shard_idx += 1
                            current_shard = {}
                            current_size = 0
                        
                        current_shard[name] = tensor
                        current_size += tensor_size
                        index["weight_map"][name] = f"model-{shard_idx+1:05d}-of-{shard_idx+2:05d}.safetensors"
                    
                    # 保存最后一个 shard
                    if current_shard:
                        num_shards = shard_idx + 1
                        for i in range(num_shards):
                            old_name = f"model-{i+1:05d}-of-XXXXX.safetensors"
                            new_name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
                            old_path = os.path.join(save_dir, old_name)
                            if os.path.exists(old_path):
                                os.rename(old_path, os.path.join(save_dir, new_name))
                        
                        shard_file = f"model-{num_shards:05d}-of-{num_shards:05d}.safetensors"
                        save_file(current_shard, os.path.join(save_dir, shard_file))
                        
                        # 更新 weight_map
                        for name in index["weight_map"]:
                            old_shard = index["weight_map"][name]
                            shard_num = int(old_shard.split("-")[1])
                            index["weight_map"][name] = f"model-{shard_num:05d}-of-{num_shards:05d}.safetensors"
                        
                        # 保存 index
                        import json
                        with open(os.path.join(save_dir, "model.safetensors.index.json"), "w") as f:
                            json.dump(index, f, indent=2)
                else:
                    save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
            except ImportError:
                # 回退到 PyTorch 格式
                torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
            
            print(f"[ColaMem] {model_name} 已保存到 {save_dir}")
        
        # 同步所有进程
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    def save_pretrained(self, save_dir):
        """
        保存模型
        
        目录结构：
        # 全参数微调模式
        save_dir/
        ├── encoder/          # Encoder 模型（完整权重）
        ├── decoder/          # Decoder 模型（完整权重）
        └── custom_params.pt  # 自定义参数
        
        # LoRA 微调模式
        save_dir/
        ├── encoder_lora/     # Encoder LoRA 适配器
        ├── decoder_lora/     # Decoder LoRA 适配器
        └── custom_params.pt  # 自定义参数
        
        注意：在 DeepSpeed ZeRO-3 下会自动收集分片的权重
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 检查是否使用 DeepSpeed ZeRO-3
        use_zero3 = self._is_deepspeed_zero3_enabled()
        
        # 获取当前 rank
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        
        # LoRA 模式：只保存适配器权重
        if self.use_lora:
            print(f"[ColaMem] 🎯 LoRA 模式：保存适配器权重...")
            
            if self.encoder_lora and hasattr(self.encoder, 'save_pretrained'):
                encoder_lora_dir = os.path.join(save_dir, "encoder_lora")
                print(f"[ColaMem] 保存 Encoder LoRA 到 {encoder_lora_dir} ...")
                if rank == 0:
                    self.encoder.save_pretrained(encoder_lora_dir)
            
            if self.decoder_lora and hasattr(self.decoder, 'save_pretrained'):
                decoder_lora_dir = os.path.join(save_dir, "decoder_lora")
                print(f"[ColaMem] 保存 Decoder LoRA 到 {decoder_lora_dir} ...")
                if rank == 0:
                    self.decoder.save_pretrained(decoder_lora_dir)
        else:
            # 全参数微调模式：保存完整模型
            if use_zero3:
                print("[ColaMem] 检测到 DeepSpeed ZeRO-3，使用特殊保存逻辑...")
                
                # 使用 ZeRO-3 保存方法
                encoder_dir = os.path.join(save_dir, "encoder")
                print(f"[ColaMem] 保存 Encoder 到 {encoder_dir} ...")
                self._save_model_zero3(self.encoder, encoder_dir, "Encoder")
                
                decoder_dir = os.path.join(save_dir, "decoder")
                print(f"[ColaMem] 保存 Decoder 到 {decoder_dir} ...")
                self._save_model_zero3(self.decoder, decoder_dir, "Decoder")
            else:
                # 标准保存方法
                encoder_dir = os.path.join(save_dir, "encoder")
                print(f"[ColaMem] 保存 Encoder 到 {encoder_dir} ...")
                self.encoder.save_pretrained(encoder_dir)
                
                decoder_dir = os.path.join(save_dir, "decoder")
                print(f"[ColaMem] 保存 Decoder 到 {decoder_dir} ...")
                self.decoder.save_pretrained(decoder_dir)
        
        # 获取当前 rank（仅在 rank 0 上保存自定义参数）
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        
        if rank == 0:
            # 保存自定义参数
            custom_params_path = os.path.join(save_dir, "custom_params.pt")
            print(f"[ColaMem] 保存自定义参数到 {custom_params_path} ...")
            
            # 收集自定义参数（处理 ZeRO-3 分区的情况）
            if use_zero3:
                import deepspeed
                from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                
                # 收集 mem_anchors
                with GatheredParameters([self.mem_anchors], modifier_rank=0):
                    mem_anchors_cpu = self.mem_anchors.detach().cpu().clone()
                
                # 收集 work_tokens（如果启用）
                if self.use_workspace:
                    with GatheredParameters([self.work_tokens], modifier_rank=0):
                        work_tokens_cpu = self.work_tokens.detach().cpu().clone()
                else:
                    work_tokens_cpu = torch.empty(0)
                
                # Projector 参数
                projector_state = {}
                for name, param in self.projector.named_parameters():
                    with GatheredParameters([param], modifier_rank=0):
                        projector_state[name] = param.detach().cpu().clone()
                
            else:
                mem_anchors_cpu = self.mem_anchors.detach().cpu()
                work_tokens_cpu = self.work_tokens.detach().cpu() if self.use_workspace else torch.empty(0)
                projector_state = self.projector.state_dict()
            
            save_dict = {
                "mem_anchors": mem_anchors_cpu,
                "work_tokens": work_tokens_cpu,
                "projector": projector_state,
                "num_mem": self.num_mem,
                "num_work": self.num_work,
                "use_workspace": self.use_workspace,  # 保存 workspace 状态
                "use_semantic_init": self.use_semantic_init,
                "lambda_orth": self.lambda_orth,
                "freeze_vision_encoder": self.freeze_vision_encoder,
                "freeze_decoder": self.freeze_decoder,
                "base_model_path": self.model_path,  # 记录原始模型路径，加载时用于恢复 Vision Encoder
                # LoRA 配置（支持 encoder 和 decoder 独立配置）
                "use_lora": self.use_lora,
                # Encoder LoRA 配置
                "encoder_lora": self.encoder_lora,
                "encoder_lora_r": self.encoder_lora_r,
                "encoder_lora_alpha": self.encoder_lora_alpha,
                "encoder_lora_dropout": self.encoder_lora_dropout,
                "encoder_lora_target_modules": self.encoder_lora_target_modules,
                # Decoder LoRA 配置
                "decoder_lora": self.decoder_lora,
                "decoder_lora_r": self.decoder_lora_r,
                "decoder_lora_alpha": self.decoder_lora_alpha,
                "decoder_lora_dropout": self.decoder_lora_dropout,
                "decoder_lora_target_modules": self.decoder_lora_target_modules,
                # [修复] 保存 residual_gate_logit 和 memory_scale（此前遗漏）
                "residual_gate_logit": self.residual_gate_logit.detach().cpu().clone(),
                "memory_scale": self.memory_scale.detach().cpu().clone(),
            }
            
            # 保存初始锚点（如果有）
            if self.initial_mem_anchors is not None:
                if use_zero3:
                    with GatheredParameters([self.initial_mem_anchors], modifier_rank=0):
                        save_dict["initial_mem_anchors"] = self.initial_mem_anchors.detach().cpu().clone()
                else:
                    save_dict["initial_mem_anchors"] = self.initial_mem_anchors.detach().cpu()
            
            torch.save(save_dict, custom_params_path)
            print(f"[ColaMem] 模型已保存到 {save_dir}")
            
            # 验证保存的 checkpoint
            print(f"[ColaMem] 验证保存的 checkpoint...")
            is_valid, errors = ColaMemModel.verify_checkpoint(save_dir)
            if is_valid:
                print(f"[ColaMem] ✅ Checkpoint 验证通过！")
            else:
                print(f"[ColaMem] ⚠️ Checkpoint 验证失败:")
                for err in errors:
                    print(f"[ColaMem]    - {err}")
                print(f"[ColaMem] ⚠️ 建议: 如果使用 DeepSpeed ZeRO-3，请检查模型保存逻辑")
        
        # 同步所有进程
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    @classmethod
    def verify_checkpoint(cls, save_dir, expected_vocab_size=151936, expected_hidden_size=4096):
        """
        验证保存的 checkpoint 是否正确
        
        Args:
            save_dir: checkpoint 目录
            expected_vocab_size: 期望的词表大小（Qwen3VL-4B 默认 151936）
            expected_hidden_size: 期望的隐藏层大小（Qwen3VL-4B 默认 4096）
            
        Returns:
            (is_valid, errors): 是否有效, 错误信息列表
        """
        errors = []
        
        # 1. 检查目录结构
        encoder_dir = os.path.join(save_dir, "encoder")
        decoder_dir = os.path.join(save_dir, "decoder")
        custom_params_path = os.path.join(save_dir, "custom_params.pt")
        
        if not os.path.exists(encoder_dir):
            errors.append(f"Encoder 目录不存在: {encoder_dir}")
        if not os.path.exists(decoder_dir):
            errors.append(f"Decoder 目录不存在: {decoder_dir}")
        if not os.path.exists(custom_params_path):
            errors.append(f"自定义参数文件不存在: {custom_params_path}")
        
        if errors:
            return False, errors
        
        # 2. 验证 custom_params.pt
        try:
            custom_params = torch.load(custom_params_path, map_location="cpu")
            
            # 检查必要的键
            required_keys = ["mem_anchors", "work_tokens", "projector", "num_mem", "num_work"]
            for key in required_keys:
                if key not in custom_params:
                    errors.append(f"custom_params.pt 缺少键: {key}")
            
            # 检查张量形状
            if "mem_anchors" in custom_params:
                shape = custom_params["mem_anchors"].shape
                # mem_anchors 形状应该是 (1, num_mem, hidden_size) 或 (num_mem, hidden_size)
                if len(shape) not in [2, 3] or shape[-1] == 0:
                    errors.append(f"mem_anchors 形状异常: {shape}（期望 2D 或 3D 张量）")
            
            if "work_tokens" in custom_params:
                shape = custom_params["work_tokens"].shape
                # work_tokens 形状应该是 (1, num_work, hidden_size) 或 (num_work, hidden_size)
                if len(shape) not in [2, 3] or shape[-1] == 0:
                    errors.append(f"work_tokens 形状异常: {shape}（期望 2D 或 3D 张量）")
                    
        except Exception as e:
            errors.append(f"加载 custom_params.pt 失败: {e}")
        
        # 3. 验证 encoder 权重
        encoder_errors = cls._verify_model_weights(encoder_dir, "encoder", expected_vocab_size, expected_hidden_size)
        errors.extend(encoder_errors)
        
        # 4. 验证 decoder 权重
        decoder_errors = cls._verify_model_weights(decoder_dir, "decoder", expected_vocab_size, expected_hidden_size)
        errors.extend(decoder_errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _verify_model_weights(cls, model_dir, model_name, expected_vocab_size, expected_hidden_size):
        """验证单个模型的权重"""
        errors = []
        
        # 检查 config.json
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            errors.append(f"{model_name}: config.json 不存在")
            return errors
        
        # 检查权重文件
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        safetensors_index_path = os.path.join(model_dir, "model.safetensors.index.json")
        pytorch_bin_path = os.path.join(model_dir, "pytorch_model.bin")
        
        has_weights = (
            os.path.exists(safetensors_path) or 
            os.path.exists(safetensors_index_path) or 
            os.path.exists(pytorch_bin_path)
        )
        
        if not has_weights:
            errors.append(f"{model_name}: 未找到权重文件")
            return errors
        
        # 如果使用 safetensors 格式，检查权重形状
        try:
            if os.path.exists(safetensors_index_path):
                # 分片模式
                import json
                with open(safetensors_index_path, 'r') as f:
                    index = json.load(f)
                
                weight_map = index.get("weight_map", {})
                if not weight_map:
                    errors.append(f"{model_name}: weight_map 为空")
                else:
                    # 检查第一个分片文件
                    first_shard = list(set(weight_map.values()))[0] if weight_map else None
                    if first_shard:
                        shard_path = os.path.join(model_dir, first_shard)
                        if os.path.exists(shard_path):
                            shard_errors = cls._check_safetensors_shapes(
                                shard_path, model_name, expected_vocab_size, expected_hidden_size
                            )
                            errors.extend(shard_errors)
                            
            elif os.path.exists(safetensors_path):
                shard_errors = cls._check_safetensors_shapes(
                    safetensors_path, model_name, expected_vocab_size, expected_hidden_size
                )
                errors.extend(shard_errors)
                
        except Exception as e:
            errors.append(f"{model_name}: 检查权重时出错: {e}")
        
        return errors
    
    @classmethod
    def _check_safetensors_shapes(cls, safetensors_path, model_name, expected_vocab_size, expected_hidden_size):
        """检查 safetensors 文件中关键权重的形状"""
        errors = []
        
        try:
            import struct
            import json
            
            # 读取 safetensors header
            with open(safetensors_path, 'rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                header = f.read(header_size).decode('utf-8')
                metadata = json.loads(header)
            
            # 检查关键权重
            critical_weights = {
                "lm_head.weight": (expected_vocab_size, expected_hidden_size),
                "model.embed_tokens.weight": (expected_vocab_size, expected_hidden_size),
            }
            
            for key, info in metadata.items():
                if key == "__metadata__":
                    continue
                    
                # 检查形状是否为空
                shape = info.get("shape", [])
                if not shape or 0 in shape:
                    errors.append(f"{model_name}: 权重 '{key}' 形状异常: {shape}（包含零维度）")
                    continue
                
                # 检查关键权重的期望形状
                for critical_key, expected_shape in critical_weights.items():
                    if critical_key in key:
                        if tuple(shape) != expected_shape:
                            # 只在形状明显异常时报错（允许不同模型配置）
                            if shape[0] == 0 or (len(shape) > 1 and shape[1] == 0):
                                errors.append(
                                    f"{model_name}: 关键权重 '{key}' 形状异常: {shape}，"
                                    f"期望类似 {expected_shape}"
                                )
                        break
                        
        except ImportError:
            # 如果没有 safetensors 库，跳过详细检查
            pass
        except Exception as e:
            errors.append(f"{model_name}: 解析 safetensors header 失败: {e}")
        
        return errors
    
    @classmethod
    def from_pretrained(
        cls,
        save_dir,
        num_mem_tokens=64,
        num_work_tokens=32,
        gradient_checkpointing=True,
        device_map=None,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        tokenizer=None,
        base_model_path=None  # 原始模型路径，当保存的权重损坏时回退使用
    ):
        """
        加载预训练模型
        
        支持两种模式：
        1. 全参数微调模式：从 encoder/ 和 decoder/ 加载完整权重
        2. LoRA 微调模式：从 base_model_path 加载基础模型 + encoder_lora/ 和 decoder_lora/ 加载适配器
        
        Args:
            save_dir: 模型保存目录
            base_model_path: 原始 Qwen3VL 模型路径，LoRA 模式下必须提供
            其他参数: 与 __init__ 相同
        
        Returns:
            ColaMemModel 实例
        """
        if _is_main_process():
            print(f"[ColaMem] 从 {save_dir} 加载模型...")
        
        # 检查目录结构
        encoder_dir = os.path.join(save_dir, "encoder")
        decoder_dir = os.path.join(save_dir, "decoder")
        encoder_lora_dir = os.path.join(save_dir, "encoder_lora")
        decoder_lora_dir = os.path.join(save_dir, "decoder_lora")
        custom_params_path = os.path.join(save_dir, "custom_params.pt")
        
        # 加载自定义参数（如果存在）
        use_semantic_init = False
        lambda_orth = 0.0
        freeze_decoder = False
        use_workspace = True  # 默认启用 workspace
        # LoRA 配置（从保存的参数中读取，支持 encoder 和 decoder 独立配置）
        use_lora = False
        # Encoder LoRA 默认值
        encoder_lora = True
        encoder_lora_r = 16
        encoder_lora_alpha = 32
        encoder_lora_dropout = 0.05
        encoder_lora_target_modules = None
        # Decoder LoRA 默认值
        decoder_lora = True
        decoder_lora_r = 16
        decoder_lora_alpha = 32
        decoder_lora_dropout = 0.05
        decoder_lora_target_modules = None
        
        if os.path.exists(custom_params_path):
            print(f"[ColaMem] 加载自定义参数...")
            custom_params = torch.load(custom_params_path, map_location="cpu")
            num_mem_tokens = custom_params.get("num_mem", num_mem_tokens)
            num_work_tokens = custom_params.get("num_work", num_work_tokens)
            use_workspace = custom_params.get("use_workspace", True)
            use_semantic_init = custom_params.get("use_semantic_init", False)
            lambda_orth = custom_params.get("lambda_orth", 0.0)
            freeze_decoder = custom_params.get("freeze_decoder", False)
            # LoRA 配置（支持 encoder 和 decoder 独立配置）
            use_lora = custom_params.get("use_lora", False)
            # Encoder LoRA 配置
            encoder_lora = custom_params.get("encoder_lora", True)
            encoder_lora_r = custom_params.get("encoder_lora_r", custom_params.get("lora_r", 16))  # 兼容旧版本
            encoder_lora_alpha = custom_params.get("encoder_lora_alpha", custom_params.get("lora_alpha", 32))
            encoder_lora_dropout = custom_params.get("encoder_lora_dropout", custom_params.get("lora_dropout", 0.05))
            encoder_lora_target_modules = custom_params.get("encoder_lora_target_modules", custom_params.get("lora_target_modules", None))
            # Decoder LoRA 配置
            decoder_lora = custom_params.get("decoder_lora", True)
            decoder_lora_r = custom_params.get("decoder_lora_r", custom_params.get("lora_r", 16))  # 兼容旧版本
            decoder_lora_alpha = custom_params.get("decoder_lora_alpha", custom_params.get("lora_alpha", 32))
            decoder_lora_dropout = custom_params.get("decoder_lora_dropout", custom_params.get("lora_dropout", 0.05))
            decoder_lora_target_modules = custom_params.get("decoder_lora_target_modules", custom_params.get("lora_target_modules", None))
            # 如果没有提供 base_model_path，尝试从保存的参数中获取
            if base_model_path is None:
                base_model_path = custom_params.get("base_model_path")
        
        # 根据模式检查目录结构
        if use_lora:
            # LoRA 模式：需要 base_model_path 和 LoRA 适配器目录
            if base_model_path is None:
                raise ValueError(
                    "LoRA 模式需要提供 base_model_path！"
                    "请在调用时指定 base_model_path 参数。"
                )
            if encoder_lora and not os.path.exists(encoder_lora_dir):
                raise ValueError(f"Encoder LoRA 目录不存在: {encoder_lora_dir}")
            if decoder_lora and not os.path.exists(decoder_lora_dir):
                raise ValueError(f"Decoder LoRA 目录不存在: {decoder_lora_dir}")
        else:
            # 全参数微调模式：需要 encoder/ 和 decoder/ 目录
            if not os.path.exists(encoder_dir) and not os.path.exists(decoder_dir):
                # 如果两个目录都不存在，检查是否是 LoRA 模式
                if os.path.exists(encoder_lora_dir) or os.path.exists(decoder_lora_dir):
                    print(f"[ColaMem] 检测到 LoRA 适配器目录，自动切换到 LoRA 模式")
                    use_lora = True
                    if base_model_path is None:
                        raise ValueError(
                            "检测到 LoRA 适配器但未提供 base_model_path！"
                            "请在调用时指定 base_model_path 参数。"
                        )
                else:
                    raise ValueError(f"Encoder 目录不存在: {encoder_dir}")
        
        # ========== 加载模型 ==========
        if use_lora:
            # LoRA 模式：从 base_model_path 加载基础模型，然后加载适配器
            print(f"[ColaMem] 🎯 LoRA 模式：从 {base_model_path} 加载基础模型...")
            
            # 加载基础 Encoder
            print(f"[ColaMem] 加载 Encoder 基础模型...")
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                device_map=device_map
            )
            
            # 加载基础 Decoder
            print(f"[ColaMem] 加载 Decoder 基础模型...")
            decoder = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                device_map=device_map
            )
            
            # 加载 LoRA 适配器
            if encoder_lora and os.path.exists(encoder_lora_dir):
                print(f"[ColaMem] 加载 Encoder LoRA 适配器: {encoder_lora_dir}")
                encoder = PeftModel.from_pretrained(encoder, encoder_lora_dir)
                print(f"[ColaMem] ✅ Encoder LoRA 适配器加载成功")
            
            if decoder_lora and os.path.exists(decoder_lora_dir):
                print(f"[ColaMem] 加载 Decoder LoRA 适配器: {decoder_lora_dir}")
                decoder = PeftModel.from_pretrained(decoder, decoder_lora_dir)
                print(f"[ColaMem] ✅ Decoder LoRA 适配器加载成功")
            
            init_path = base_model_path
        else:
            # 全参数微调模式
            encoder_loaded_from_checkpoint = False
            decoder_loaded_from_checkpoint = False
            
            # 尝试从 checkpoint 加载 Encoder
            try:
                print(f"[ColaMem] 尝试从 checkpoint 加载 Encoder: {encoder_dir}")
                encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    encoder_dir,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    device_map=device_map
                )
                encoder_loaded_from_checkpoint = True
                print(f"[ColaMem] ✅ Encoder 已从 checkpoint 加载")
            except Exception as e:
                print(f"[ColaMem] ⚠️ 从 checkpoint 加载 Encoder 失败: {e}")
                encoder = None
            
            # 尝试从 checkpoint 加载 Decoder
            try:
                print(f"[ColaMem] 尝试从 checkpoint 加载 Decoder: {decoder_dir}")
                decoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    decoder_dir,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    device_map=device_map
                )
                decoder_loaded_from_checkpoint = True
                print(f"[ColaMem] ✅ Decoder 已从 checkpoint 加载")
            except Exception as e:
                print(f"[ColaMem] ⚠️ 从 checkpoint 加载 Decoder 失败: {e}")
                decoder = None
            
            # 如果 checkpoint 加载失败，回退到 base_model_path
            if (not encoder_loaded_from_checkpoint or not decoder_loaded_from_checkpoint) and base_model_path:
                print(f"[ColaMem] 回退到基础模型: {base_model_path}")
                if not encoder_loaded_from_checkpoint:
                    print(f"[ColaMem] 从基础模型加载 Encoder...")
                    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                        base_model_path,
                        torch_dtype=torch_dtype,
                        attn_implementation=attn_implementation,
                        device_map=device_map
                    )
                if not decoder_loaded_from_checkpoint:
                    print(f"[ColaMem] 从基础模型加载 Decoder...")
                    decoder = Qwen3VLForConditionalGeneration.from_pretrained(
                        base_model_path,
                        torch_dtype=torch_dtype,
                        attn_implementation=attn_implementation,
                        device_map=device_map
                    )
            elif not encoder_loaded_from_checkpoint or not decoder_loaded_from_checkpoint:
                raise ValueError(
                    f"无法加载模型！checkpoint 加载失败且未提供 base_model_path。\n"
                    f"  Encoder 加载成功: {encoder_loaded_from_checkpoint}\n"
                    f"  Decoder 加载成功: {decoder_loaded_from_checkpoint}"
                )
            
            # 打印加载来源信息
            print(f"[ColaMem] 模型加载来源:")
            print(f"  - Encoder: {'checkpoint ✅' if encoder_loaded_from_checkpoint else 'base_model ⚠️'}")
            print(f"  - Decoder: {'checkpoint ✅' if decoder_loaded_from_checkpoint else 'base_model ⚠️'}")
            
            init_path = encoder_dir if encoder_loaded_from_checkpoint else base_model_path
        
        # 创建模型实例
        # 注意：from_pretrained 场景下，如果是 LoRA 模式，外部已经加载好了 PeftModel，
        # 所以这里创建实例时 use_lora=False，避免 __init__ 中重复包装 LoRA。
        # 后续会将外部加载好的 encoder/decoder 替换进去。
        model = cls(
            model_path=init_path,
            num_mem_tokens=num_mem_tokens,
            num_work_tokens=num_work_tokens,
            use_workspace=use_workspace,
            gradient_checkpointing=gradient_checkpointing,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            use_semantic_init=False,  # 加载时不重新初始化
            lambda_orth=lambda_orth,
            freeze_decoder=freeze_decoder,
            tokenizer=tokenizer,
            # from_pretrained 不启用 LoRA（外部已加载好适配器，直接替换）
            use_lora=False,
        )
        
        # 替换为已加载的 encoder/decoder（可能是 PeftModel 或原始模型）
        model.encoder = encoder
        model.decoder = decoder
        
        # 恢复 LoRA 相关的配置字段（用于后续逻辑判断，如 _get_encoder_inner_model）
        model.use_lora = use_lora
        model.encoder_lora = encoder_lora
        model.encoder_lora_r = encoder_lora_r
        model.encoder_lora_alpha = encoder_lora_alpha
        model.encoder_lora_dropout = encoder_lora_dropout
        model.encoder_lora_target_modules = encoder_lora_target_modules or cls.DEFAULT_LORA_TARGET_MODULES
        model.decoder_lora = decoder_lora
        model.decoder_lora_r = decoder_lora_r
        model.decoder_lora_alpha = decoder_lora_alpha
        model.decoder_lora_dropout = decoder_lora_dropout
        model.decoder_lora_target_modules = decoder_lora_target_modules or cls.DEFAULT_LORA_TARGET_MODULES
        
        # 加载自定义参数
        if os.path.exists(custom_params_path):
            print(f"[ColaMem] 加载自定义参数...")
            custom_params = torch.load(custom_params_path, map_location="cpu")
            with torch.no_grad():
                # 加载 mem_anchors，处理形状问题
                saved_mem_anchors = custom_params["mem_anchors"]
                if saved_mem_anchors.shape != model.mem_anchors.shape:
                    # 尝试 squeeze/unsqueeze 来匹配形状
                    if saved_mem_anchors.dim() == 2 and model.mem_anchors.dim() == 3:
                        saved_mem_anchors = saved_mem_anchors.unsqueeze(0)
                    elif saved_mem_anchors.dim() == 3 and model.mem_anchors.dim() == 2:
                        saved_mem_anchors = saved_mem_anchors.squeeze(0)
                    print(f"[ColaMem] 调整 mem_anchors 形状: {custom_params['mem_anchors'].shape} -> {saved_mem_anchors.shape}")
                model.mem_anchors.copy_(saved_mem_anchors.to(model.mem_anchors.device))
                
                # 加载 work_tokens，处理形状问题（仅在启用 workspace 时）
                if model.use_workspace and "work_tokens" in custom_params:
                    saved_work_tokens = custom_params["work_tokens"]
                    if saved_work_tokens.numel() > 0:  # 非空才加载
                        if saved_work_tokens.shape != model.work_tokens.shape:
                            if saved_work_tokens.dim() == 2 and model.work_tokens.dim() == 3:
                                saved_work_tokens = saved_work_tokens.unsqueeze(0)
                            elif saved_work_tokens.dim() == 3 and model.work_tokens.dim() == 2:
                                saved_work_tokens = saved_work_tokens.squeeze(0)
                            print(f"[ColaMem] 调整 work_tokens 形状: {custom_params['work_tokens'].shape} -> {saved_work_tokens.shape}")
                        model.work_tokens.copy_(saved_work_tokens.to(model.work_tokens.device))
                
                # Projector: 4层结构（无输出 LayerNorm）
                # 如果 checkpoint 是旧版本（5层，有输出 LayerNorm），只加载前4层
                saved_projector_state = custom_params["projector"]
                saved_has_layer4 = "4.weight" in saved_projector_state or "4.bias" in saved_projector_state
                
                if saved_has_layer4:
                    # 旧版本有输出 LayerNorm -> 只加载前4层
                    print(f"[ColaMem] ⚠️ 检测到旧版本 Projector（有输出 LayerNorm），跳过第5层...")
                    current_projector_state = model.projector.state_dict()
                    for key in current_projector_state:
                        if key in saved_projector_state:
                            current_projector_state[key] = saved_projector_state[key]
                    model.projector.load_state_dict(current_projector_state)
                    print(f"[ColaMem] ✅ Projector 加载成功（跳过旧的输出 LayerNorm）")
                else:
                    # 新版本，直接加载
                    model.projector.load_state_dict(saved_projector_state)
                
                # memory_norm 已移除（方案A），跳过加载
                if "memory_norm" in custom_params:
                    print(f"[ColaMem] ⚠️ checkpoint 中有 memory_norm 但已移除，跳过加载")
                
                # [修复] 加载 residual_gate_logit 和 memory_scale（此前遗漏）
                if "residual_gate_logit" in custom_params:
                    model.residual_gate_logit.copy_(custom_params["residual_gate_logit"].to(model.residual_gate_logit.device))
                    gate_val = torch.sigmoid(model.residual_gate_logit.float()).mean().item()
                    print(f"[ColaMem] ✅ residual_gate_logit 加载成功 (gate={gate_val:.4f})")
                else:
                    print(f"[ColaMem] ⚠️ checkpoint 中无 residual_gate_logit，使用默认值")
                
                if "memory_scale" in custom_params:
                    model.memory_scale.copy_(custom_params["memory_scale"].to(model.memory_scale.device))
                    print(f"[ColaMem] ✅ memory_scale 加载成功 (scale={model.memory_scale.item():.4f})")
                else:
                    print(f"[ColaMem] ⚠️ checkpoint 中无 memory_scale，使用默认值")
                
                # 加载初始锚点（如果有）
                if "initial_mem_anchors" in custom_params and custom_params["initial_mem_anchors"] is not None:
                    saved_initial = custom_params["initial_mem_anchors"]
                    # 处理形状问题
                    if model.initial_mem_anchors is not None:
                        if saved_initial.shape != model.initial_mem_anchors.shape:
                            if saved_initial.dim() == 2 and model.initial_mem_anchors.dim() == 3:
                                saved_initial = saved_initial.unsqueeze(0)
                            elif saved_initial.dim() == 3 and model.initial_mem_anchors.dim() == 2:
                                saved_initial = saved_initial.squeeze(0)
                        model.initial_mem_anchors.copy_(saved_initial.to(model.mem_anchors.device))
                    else:
                        # 如果模型没有 initial_mem_anchors，则注册它
                        model.register_buffer('initial_mem_anchors', saved_initial.to(model.mem_anchors.device))
            # 恢复软约束配置
            model.use_semantic_init = custom_params.get("use_semantic_init", False)
            model.lambda_orth = custom_params.get("lambda_orth", 0.0)
        
        if _is_main_process():
            print(f"[ColaMem] 模型加载完成！")
        return model


# ============================================================================
# 注册 ColaMem 到 Hugging Face Auto Classes
# ============================================================================

# 注册配置类
AutoConfig.register("colamem", ColaMemConfig)
print("[ColaMem] ✅ ColaMemConfig 已注册到 AutoConfig")
