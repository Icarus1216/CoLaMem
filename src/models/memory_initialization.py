"""
ColaMem Memory Initialization & Soft Constraints

本模块实现了针对 Memory Anchors 的软约束与初始化策略，旨在解决端到端训练中的
"模式坍塌 (Mode Collapse)" 问题。

核心设计：
1. 统一多样化初始化 (Unified Diverse Initialization)
   - 64 个 Memory Anchors 不人工分组
   - 使用统一的大概念词池（覆盖语义/情景/文本/视觉）提供初始多样性
   - 每个 anchor 可以自由学习编码任意模态的信息
   - 对多 sub-token 概念词使用 mean pooling 提取更稳定的语义表示

2. 正交正则化场 (Orthogonality Regularization Field)
   - 在 Loss 函数中增加"斥力项"，强迫锚点在向量空间中尽可能张开
   - 只惩罚 off-diagonal 元素（互相似度），不干扰对角线
   - 支持 warmup：训练初期不施加约束，后期逐步增强
   - 最大化记忆容量（熵），防止锚点学到重复特征

设计哲学：
- 拒绝硬编码：不强制规定某些向量只能存某类信息
- 不引入人工先验分组：让模型自己决定哪些 anchor 存什么
- 引入多样化归纳偏置：通过统一概念词池和正交约束引导模型自发分化
- 零推理成本：所有操作都在训练阶段完成

参考文献：
- Perceiver IO: A General Architecture for Structured Inputs & Outputs (2022)
- Preventing Dimensional Collapse via Orthogonality Regularization (2024)
- Soft Orthogonal Proxies for Deep Metric Learning (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np
import os


def _is_main_process():
    """判断当前进程是否为主进程（rank 0），避免多卡重复打印"""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank() == 0
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', -1)))
    return local_rank in (-1, 0)


# ============================================================================
# 模块一：统一多样化初始化 (Unified Diverse Initialization)
# ============================================================================

class MemoryAnchorInitializer:
    """
    Memory Anchors 初始化器（无分组版本）
    
    功能：
    1. 使用统一的大概念词池提供初始多样性
    2. 使用基座模型的 Embedding 提取概念向量（多 token mean pooling）
    3. 从概念词池的高斯分布中采样初始化所有 anchor
    4. 每个 anchor 加独立随机扰动，确保起始点多样
    
    设计理由：
    - 不人工区分"语义 anchor"和"情景 anchor"
    - Encoder 是 Full Attention，每个 anchor 天然可以同时编码文本和视觉信息
    - 人工分组会限制模型自由度，模型能自己发现最优的信息编码方式
    - 真正需要的是"多样性"而非"分组"
    
    Example:
        ```python
        from transformers import Qwen3VLForConditionalGeneration
        
        model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        
        initializer = MemoryAnchorInitializer(
            embedding_layer=model.get_input_embeddings(),
            tokenizer=processor.tokenizer,
            num_mem_tokens=64
        )
        
        mem_anchors = initializer.initialize()
        ```
    """
    
    # 统一概念词池（~100 个，混合语义+情景+文本+视觉，覆盖 VQA/Caption 全场景）
    # 不分组：让模型自己决定每个 anchor 的"职责"
    UNIFIED_CONCEPTS = [
        # === 语义/抽象/概括 ===
        "summary", "abstract", "concept", "outline", "gist",
        "conclusion", "theme", "principle", "overview", "meaning",
        
        # === 逻辑/推理/因果 ===
        "logic", "reasoning", "inference", "deduction", "cause",
        "effect", "comparison", "contrast", "analogy", "hypothesis",
        
        # === 关系/结构/层次 ===
        "relationship", "connection", "category", "type", "group",
        "hierarchy", "sequence", "order", "structure", "context",
        
        # === 数量/计数 ===
        "number", "count", "quantity", "amount", "total",
        "few", "many", "several", "pair", "single",
        
        # === 空间/方位/距离 ===
        "direction", "distance", "above", "below", "between",
        "left", "right", "center", "top", "bottom",
        "front", "back", "inside", "outside", "nearby",
        
        # === 视觉场景/构图 ===
        "object", "visual", "image", "scene", "view",
        "photo", "picture", "background", "foreground", "frame",
        
        # === 颜色/材质/纹理 ===
        "color", "red", "blue", "green", "white",
        "texture", "material", "surface", "pattern", "fabric",
        
        # === 形状/尺寸/外观 ===
        "shape", "round", "square", "size", "large",
        "small", "tall", "wide", "edge", "corner",
        
        # === 物体/实体 ===
        "person", "animal", "food", "vehicle", "building",
        "furniture", "clothing", "plant", "sign", "device",
        
        # === 动作/状态/事件 ===
        "action", "motion", "standing", "sitting", "walking",
        "holding", "eating", "playing", "working", "looking",
        
        # === 属性/判断/答案 ===
        "purpose", "function", "reason", "difference", "similarity",
        "yes", "no", "true", "false", "answer",
    ]
    
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        tokenizer,
        num_mem_tokens: int = 64,
        std_scale: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            embedding_layer: 基座模型的 Embedding Layer
            tokenizer: 对应的 Tokenizer
            num_mem_tokens: Memory Anchors 总数（默认 64）
            std_scale: 标准差缩放因子（默认 1.0）
            device: 目标设备
        """
        self.embedding_layer = embedding_layer
        self.tokenizer = tokenizer
        self.num_mem_tokens = num_mem_tokens
        self.std_scale = std_scale
        self.device = device or embedding_layer.weight.device
        self.dtype = embedding_layer.weight.dtype
        
        if _is_main_process():
            print(f"[MemoryAnchorInitializer] 配置:")
            print(f"  - 总锚点数: {num_mem_tokens}")
            print(f"  - 统一概念词池: {len(self.UNIFIED_CONCEPTS)} 个（无分组）")
            print(f"  - 设计理念: 不引入人工先验，让模型自由学习")
    
    def _extract_concept_embeddings(self, concepts: List[str]) -> torch.Tensor:
        """
        提取概念词的 Embedding（多 token mean pooling）
        
        对 BPE/SentencePiece 拆分成多个 sub-token 的概念词，
        将所有 sub-token 的 embedding 取均值，获取更稳定的语义表示。
        
        Args:
            concepts: 概念词列表
        
        Returns:
            embeddings: [num_concepts, hidden_size]
        """
        all_embeddings = []
        
        for concept in concepts:
            # Tokenize 概念词（可能产生多个 sub-token）
            ids = self.tokenizer.encode(concept, add_special_tokens=False)
            if len(ids) == 0:
                continue
            
            # 获取所有 sub-token 的 embedding 并取平均
            token_ids = torch.tensor(ids, device=self.device)
            with torch.no_grad():
                token_embeds = self.embedding_layer(token_ids)  # [num_subtokens, hidden_size]
                # Mean pooling: 将多个 sub-token 的语义合并为一个稳定向量
                concept_embed = token_embeds.mean(dim=0)  # [hidden_size]
            
            all_embeddings.append(concept_embed)
        
        if len(all_embeddings) == 0:
            raise ValueError("没有有效的概念词可提取 Embedding！请检查 tokenizer。")
        
        # Stack 成 [num_concepts, hidden_size]
        embeddings = torch.stack(all_embeddings, dim=0)
        
        return embeddings
    
    def _compute_distribution_params(
        self, 
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 Embedding 的均值和全局标准差
        
        使用全局标准差（而非逐维 std）更稳定，避免小概念池导致的"脆弱分布"。
        
        Args:
            embeddings: [num_concepts, hidden_size]
        
        Returns:
            mean: [hidden_size]
            std: scalar（全局标准差，broadcast 到所有维度）
        """
        mean = embeddings.mean(dim=0)
        # 使用全局标准差：更稳定，不会因为某些维度方差极小而出问题
        global_std = embeddings.std() * self.std_scale
        
        # 防止标准差过小
        global_std = torch.clamp(global_std, min=0.01)
        
        return mean, global_std
    
    def initialize(self) -> torch.Tensor:
        """
        初始化 Memory Anchors（Prototype-Jitter 策略，无分组）
        
        策略：每个 anchor 从概念池中随机选一个概念 embedding 作为 prototype，
        再加小幅 jitter 打破对称性。
        
        为什么不用 concept_mean + jitter：
        - concept_mean 是所有概念的"灰色中心"，模长大但语义模糊
        - 如果 ||concept_mean|| >> ||noise||，64 个 anchor 方向高度一致
        - 概念池的"多峰结构"（颜色/空间/逻辑/对象...）被平均掉了
        
        Prototype-Jitter 的优势：
        - 每个 anchor 从不同语义方向起步（不依赖 concept_mean 的大小）
        - 初始 Gram 矩阵 off-diagonal 天然较低（不同概念本身就不太相似）
        - 正交正则化的起点更好，收敛更快
        
        Returns:
            mem_anchors: [1, num_mem_tokens, hidden_size]
        """
        _verbose = _is_main_process()
        if _verbose:
            print(f"[MemoryAnchorInitializer] 开始初始化（Prototype-Jitter 策略，无分组）...")
        
        # 1. 提取所有概念词的 Embedding
        if _verbose:
            print(f"[MemoryAnchorInitializer] 提取概念 ({len(self.UNIFIED_CONCEPTS)} 个, 多 token pooling)...")
        concept_embeds = self._extract_concept_embeddings(self.UNIFIED_CONCEPTS)
        concept_mean, global_std = self._compute_distribution_params(concept_embeds)
        
        # 2. Prototype-Jitter：每个 anchor 选一个不同的概念原型 + 小幅噪声
        num_concepts = concept_embeds.shape[0]
        if _verbose:
            print(f"[MemoryAnchorInitializer] Prototype-Jitter: 从 {num_concepts} 个概念中为 {self.num_mem_tokens} 个 anchor 选取原型...")
        
        if self.num_mem_tokens <= num_concepts:
            # anchor 数 <= 概念数：不放回采样，确保尽可能覆盖不同概念
            indices = torch.randperm(num_concepts)[:self.num_mem_tokens]
        else:
            # anchor 数 > 概念数：先不放回采完一轮，再随机补充
            full_round = torch.randperm(num_concepts)
            extra = torch.randint(0, num_concepts, (self.num_mem_tokens - num_concepts,))
            indices = torch.cat([full_round, extra])
        
        # 选取基准向量（每个 anchor 的起点是不同的概念 embedding）
        base_anchors = concept_embeds[indices]  # [num_mem, hidden_size]
        
        # 添加小幅扰动（主要方差由 prototype 多样性提供，jitter 只打破完全对称）
        # jitter 尺度 = 全局 std × 0.1（比 prototype 间距小得多）
        jitter = torch.randn_like(base_anchors) * (global_std * 0.1)
        mem_anchors = (base_anchors + jitter).to(dtype=self.dtype)
        
        mem_anchors = mem_anchors.unsqueeze(0)  # [1, num_mem_tokens, hidden_size]
        
        # 3. 验证
        if _verbose:
            print(f"[MemoryAnchorInitializer] 初始化完成！")
            print(f"  - 形状: {mem_anchors.shape}")
            print(f"  - 数据类型: {mem_anchors.dtype}")
            print(f"  - 设备: {mem_anchors.device}")
        
        # 4. 计算初始多样性指标
        with torch.no_grad():
            anchors_2d = mem_anchors.squeeze(0)  # [num_mem, hidden]
            anchors_norm = F.normalize(anchors_2d, p=2, dim=1)
            gram = torch.matmul(anchors_norm, anchors_norm.t())
            num_mem = gram.shape[0]
            offdiag_mask = ~torch.eye(num_mem, dtype=torch.bool, device=gram.device)
            offdiag = gram[offdiag_mask]
            
            avg_sim_to_concept = F.cosine_similarity(
                anchors_2d.mean(dim=0, keepdim=True),
                concept_mean.unsqueeze(0),
                dim=1
            ).item()
        
        if _verbose:
            print(f"[MemoryAnchorInitializer] 初始多样性验证:")
            print(f"  - 锚点 ↔ 概念池均值 余弦相似度: {avg_sim_to_concept:.4f}")
            print(f"  - 锚点间 off-diagonal 绝对值均值: {offdiag.abs().mean().item():.4f}")
            print(f"  - 锚点间 off-diagonal 最大绝对值: {offdiag.abs().max().item():.4f}")
            print(f"  ✅ 预期: 锚点间相似度低 (<0.3)，说明初始多样性良好")
        
        return mem_anchors


# ============================================================================
# 模块二：正交正则化场 (Orthogonality Regularization Field)
# ============================================================================

class OrthogonalityRegularizer:
    """
    正交正则化器
    
    功能：
    1. 计算 Memory Anchors 参数的 Gram 矩阵
    2. 只惩罚 off-diagonal 元素（互相似度），不干扰对角线
    3. 支持 warmup：训练初期不施加约束，后期逐步增强
    
    数学原理：
    - Gram 矩阵 G = norm(A) @ norm(A)^T（L2 归一化后的点积）
    - Loss = mean(G[off-diagonal]^2)（只惩罚非对角线元素）
    - 目标：让所有锚点互不相似（off-diagonal → 0）
    - 不强制对角线为 1（normalize 后自动为 1，无需额外约束）
    
    关键设计决策：
    - 只约束 **anchor 参数本身**，而非 latent_memory 输出
      （后者是图片相关的，强制正交会破坏必要的相关性）
    - 使用 warmup 防止初始化的多样性被冲掉
    
    Example:
        ```python
        regularizer = OrthogonalityRegularizer(
            lambda_orth=0.05,
            warmup_steps=500
        )
        
        orth_loss = regularizer.compute_loss(model.mem_anchors)
        total_loss = lm_loss + orth_loss
        ```
    """
    
    def __init__(
        self,
        lambda_orth: float = 0.05,
        normalize: bool = True,
        warmup_steps: int = 0,
        eps: float = 1e-8
    ):
        """
        Args:
            lambda_orth: 正交正则化权重（默认 0.05，建议 0.01~0.1）
            normalize: 是否对锚点进行 L2 归一化（默认 True）
            warmup_steps: warmup 步数（默认 0，不使用 warmup）
                          前 warmup_steps 步内 lambda 从 0 线性增长到 lambda_orth
            eps: 数值稳定性常数
        """
        self.lambda_orth = lambda_orth
        self.normalize = normalize
        self.warmup_steps = warmup_steps
        self.eps = eps
        self._step = 0
        
        if _is_main_process():
            print(f"[OrthogonalityRegularizer] 配置:")
            print(f"  - λ_orth: {lambda_orth}")
            print(f"  - 归一化: {normalize}")
            print(f"  - Warmup 步数: {warmup_steps}")
            print(f"  - Loss 类型: off-diagonal only (不惩罚对角线)")
    
    def _get_current_lambda(self) -> float:
        """获取当前步的 lambda 值（考虑 warmup）"""
        if self.warmup_steps <= 0 or self._step >= self.warmup_steps:
            return self.lambda_orth
        # 线性 warmup: 0 → lambda_orth
        progress = self._step / self.warmup_steps
        return self.lambda_orth * progress
    
    def compute_gram_matrix(self, anchors: torch.Tensor) -> torch.Tensor:
        """
        计算 Gram 矩阵
        
        Args:
            anchors: [1, num_mem, hidden_size] 或 [num_mem, hidden_size]
                     应传入 anchor 参数本身，而非 batch 输出
        
        Returns:
            gram: [num_mem, num_mem]（FP32 精度）
        """
        # 处理维度：anchor 参数通常是 [1, num_mem, hidden] 或 [num_mem, hidden]
        if anchors.dim() == 3:
            anchors = anchors.squeeze(0)  # [num_mem, hidden_size]
        
        # 强制转为 FP32 计算，防止 BF16/FP16 训练时数值不稳定
        # （64×64 Gram 矩阵计算开销极小，但低精度下 matmul 可能产生伪影）
        anchors_f32 = anchors.to(torch.float32)
        
        # L2 归一化（只优化方向，不影响模长）
        if self.normalize:
            anchors_f32 = F.normalize(anchors_f32, p=2, dim=1, eps=self.eps)
        
        # 计算 Gram 矩阵: G = A A^T（在 FP32 下计算，梯度回传时 PyTorch Autograd 会自动处理类型转换）
        gram = torch.matmul(anchors_f32, anchors_f32.transpose(-2, -1))
        
        return gram
    
    def compute_loss(self, anchors: torch.Tensor, step: int = None) -> torch.Tensor:
        """
        计算正交正则化 Loss（Hinge 式，只在互相似度超过阈值时惩罚）
        
        设计理由：
        - 在 4096 维空间中，随机向量天生接近正交（offdiag ~ 1/sqrt(H)）
        - 纯 MSE 式 loss 在 anchors 已经分散时几乎为 0，浪费梯度预算
        - Hinge 式：只在 |sim| > threshold 时惩罚，"足够分散"时不浪费梯度，
          "开始聚团"时强力拉开
        
        公式：Loss = mean(relu(|offdiag| - τ)^2)
        - τ = 0.1：允许微弱的相关性存在（某些 anchor 间的相关性可能是合理的）
        - 当所有 offdiag < τ 时，loss = 0（完全不干扰训练）
        - 当某些 offdiag > τ 时，只惩罚超出部分（精准打击）
        
        Args:
            anchors: [1, num_mem, hidden_size] 或 [num_mem, hidden_size]
                     应传入 model.mem_anchors（anchor 参数本身）
            step: 外部传入的 optimizer step 数（推荐）。如果不传，则使用内部计数器。
                  在 gradient_accumulation_steps > 1 时，不传 step 会导致 warmup 比预期快结束。
        
        Returns:
            loss: 标量 tensor
        """
        # 计算 Gram 矩阵
        gram = self.compute_gram_matrix(anchors)
        
        num_mem = gram.shape[0]
        
        # 只惩罚 off-diagonal 元素
        offdiag_mask = ~torch.eye(num_mem, dtype=torch.bool, device=gram.device)
        offdiag_elements = gram[offdiag_mask]
        
        # Hinge 式 Loss：只惩罚 |sim| > threshold 的对
        # 允许微弱的相关性（τ=0.1），只在"真的开始聚团"时才施加力
        threshold = 0.1
        penalty = torch.relu(offdiag_elements.abs() - threshold)
        loss = (penalty ** 2).mean()
        
        # 获取当前 lambda（考虑 warmup）
        # 优先使用外部传入的 step（对应 optimizer step），避免 gradient accumulation 导致过快递增
        if step is not None:
            self._step = step
        current_lambda = self._get_current_lambda()
        
        # 加权
        loss = current_lambda * loss
        
        # 仅在未传入外部 step 时使用内部计数器（向后兼容）
        if step is None:
            self._step += 1
        
        return loss
    
    def compute_diversity_metrics(self, anchors: torch.Tensor) -> dict:
        """
        计算多样性指标（用于监控和可解释性）
        
        Args:
            anchors: [1, num_mem, hidden_size] 或 [num_mem, hidden_size]
        
        Returns:
            metrics: {
                'gram_diag_mean': 对角线均值（归一化后应接近 1）,
                'gram_offdiag_mean': 非对角线均值（应接近 0）,
                'gram_offdiag_std': 非对角线标准差,
                'gram_offdiag_abs_mean': 非对角线绝对值均值（关键指标）,
                'gram_offdiag_max': 非对角线最大绝对值（最相似的一对锚点）,
                'condition_number': 条件数（越小越好）,
                'current_lambda': 当前 lambda 值（考虑 warmup）,
                'warmup_progress': warmup 进度
            }
        """
        with torch.no_grad():
            gram = self.compute_gram_matrix(anchors)
            num_mem = gram.shape[0]
            
            # 对角线统计
            diag = torch.diag(gram)
            diag_mean = diag.mean().item()
            
            # 非对角线统计
            mask = ~torch.eye(num_mem, dtype=torch.bool, device=gram.device)
            offdiag = gram[mask]
            offdiag_mean = offdiag.mean().item()
            offdiag_std = offdiag.std().item()
            offdiag_abs_mean = offdiag.abs().mean().item()
            offdiag_max = offdiag.abs().max().item()
            
            # 条件数（衡量矩阵的"病态"程度）
            try:
                singular_values = torch.linalg.svdvals(gram)
                condition_number = (singular_values.max() / (singular_values.min() + self.eps)).item()
            except:
                condition_number = float('inf')
            
            # Warmup 进度
            warmup_progress = min(1.0, self._step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
            
            metrics = {
                'gram_diag_mean': diag_mean,
                'gram_offdiag_mean': offdiag_mean,
                'gram_offdiag_std': offdiag_std,
                'gram_offdiag_abs_mean': offdiag_abs_mean,
                'gram_offdiag_max': offdiag_max,
                'condition_number': condition_number,
                'current_lambda': self._get_current_lambda(),
                'warmup_progress': warmup_progress,
            }
        
        return metrics


# ============================================================================
# 模块三：可解释性验证探针 (Explainability Probe) — 无分组版本
# ============================================================================

class MemoryExplainabilityProbe:
    """
    可解释性验证探针（无分组版本）
    
    功能：
    1. 整体漂移检测（当前 anchors vs 初始 anchors）
    2. 多样性监控（anchor 间的相互相似度分布）
    3. 与概念池的语义对齐度监控
    
    不再区分"语义组"和"情景组"，而是监控所有 anchors 的整体行为。
    
    Example:
        ```python
        probe = MemoryExplainabilityProbe(
            embedding_layer=model.encoder.get_input_embeddings(),
            tokenizer=processor.tokenizer
        )
        
        drift_metrics = probe.detect_drift(
            current_anchors=model.mem_anchors,
            initial_anchors=initial_mem_anchors
        )
        
        print(f"整体漂移: {drift_metrics['overall_drift']:.4f}")
        print(f"多样性 (offdiag_abs_mean): {drift_metrics['diversity_offdiag_abs_mean']:.4f}")
        ```
    """
    
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        tokenizer,
        concepts: Optional[List[str]] = None
    ):
        """
        Args:
            embedding_layer: 基座模型的 Embedding Layer
            tokenizer: 对应的 Tokenizer
            concepts: 概念词列表（默认使用 MemoryAnchorInitializer.UNIFIED_CONCEPTS）
        """
        self.embedding_layer = embedding_layer
        self.tokenizer = tokenizer
        
        # 使用统一概念词池
        self.concepts = concepts or MemoryAnchorInitializer.UNIFIED_CONCEPTS
        
        # 预计算概念向量
        self._precompute_concept_vectors()
    
    def _precompute_concept_vectors(self):
        """预计算概念向量（多 token mean pooling，用于对齐度检测）"""
        print(f"[MemoryExplainabilityProbe] 预计算概念向量 (多 token pooling, {len(self.concepts)} 个)...") if _is_main_process() else None
        
        device = self.embedding_layer.weight.device
        
        concept_embeds = []
        for concept in self.concepts:
            ids = self.tokenizer.encode(concept, add_special_tokens=False)
            if len(ids) > 0:
                token_ids = torch.tensor(ids, device=device)
                with torch.no_grad():
                    # 统一保存为 FP32，避免后续与外部监控张量出现 dtype 冲突
                    embed = self.embedding_layer(token_ids).float().mean(dim=0)  # mean pooling
                concept_embeds.append(embed)
        
        with torch.no_grad():
            self.concept_vectors = torch.stack(concept_embeds, dim=0)
        
        if _is_main_process():
            print(f"  - 概念向量: {self.concept_vectors.shape}")
    
    def detect_drift(
        self,
        current_anchors: torch.Tensor,
        initial_anchors: Optional[torch.Tensor] = None
    ) -> dict:
        """
        检测 anchor 漂移和多样性（无分组版本）
        
        Args:
            current_anchors: 当前的 Memory Anchors [1, num_mem, hidden_size]
            initial_anchors: 初始的 Memory Anchors（可选）
        
        Returns:
            metrics: {
                'anchor_to_concepts_mean': 所有 anchor 与概念池的平均相似度,
                'anchor_to_concepts_max_mean': 每个 anchor 与最相似概念的平均值,
                'diversity_offdiag_abs_mean': anchor 间互相似度绝对值均值（越低越多样）,
                'diversity_offdiag_max': anchor 间最大互相似度（最相似的一对）,
                'overall_drift': 整体漂移程度（如果提供 initial_anchors）,
                'per_anchor_drift_mean': 每个 anchor 的平均漂移,
                'per_anchor_drift_max': 漂移最大的 anchor,
                'per_anchor_drift_min': 漂移最小的 anchor,
            }
        """
        with torch.no_grad():
            # 处理维度
            if current_anchors.dim() == 3:
                current_anchors = current_anchors.squeeze(0)  # [num_mem, hidden_size]

            # 监控计算统一使用 FP32，避免 bf16/fp32 混合导致 matmul 失败
            current_anchors = current_anchors.float()
            
            num_mem = current_anchors.shape[0]
            device = current_anchors.device
            
            # 确保概念向量在正确设备且与 anchors 使用相同 dtype
            concept_vectors = self.concept_vectors.to(device=device, dtype=current_anchors.dtype)
            
            # 归一化
            anchors_norm = F.normalize(current_anchors, p=2, dim=1)
            concepts_norm = F.normalize(concept_vectors, p=2, dim=1)
            
            # 1. Anchor ↔ 概念池对齐度
            # [num_mem, num_concepts]
            sim_matrix = torch.matmul(anchors_norm, concepts_norm.t())
            anchor_to_concepts_mean = sim_matrix.mean().item()
            # 每个 anchor 对概念池的 top-1 相似度
            anchor_to_concepts_max = sim_matrix.max(dim=1).values  # [num_mem]
            anchor_to_concepts_max_mean = anchor_to_concepts_max.mean().item()
            
            # 2. Anchor 间多样性（互相似度）
            gram = torch.matmul(anchors_norm, anchors_norm.t())
            offdiag_mask = ~torch.eye(num_mem, dtype=torch.bool, device=device)
            offdiag = gram[offdiag_mask]
            diversity_offdiag_abs_mean = offdiag.abs().mean().item()
            diversity_offdiag_max = offdiag.abs().max().item()
            
            metrics = {
                'anchor_to_concepts_mean': anchor_to_concepts_mean,
                'anchor_to_concepts_max_mean': anchor_to_concepts_max_mean,
                'diversity_offdiag_abs_mean': diversity_offdiag_abs_mean,
                'diversity_offdiag_max': diversity_offdiag_max,
            }
            
            # 3. 如果提供了初始锚点，计算漂移
            if initial_anchors is not None:
                if initial_anchors.dim() == 3:
                    initial_anchors = initial_anchors.squeeze(0)
                
                initial_norm = F.normalize(
                    initial_anchors.to(device=device, dtype=current_anchors.dtype),
                    p=2,
                    dim=1
                )
                
                # 每个 anchor 的余弦漂移（1 - cosine_sim）
                per_anchor_sim = F.cosine_similarity(anchors_norm, initial_norm, dim=1)  # [num_mem]
                per_anchor_drift = 1.0 - per_anchor_sim
                
                metrics['overall_drift'] = per_anchor_drift.mean().item()
                metrics['per_anchor_drift_mean'] = per_anchor_drift.mean().item()
                metrics['per_anchor_drift_max'] = per_anchor_drift.max().item()
                metrics['per_anchor_drift_min'] = per_anchor_drift.min().item()
        
        return metrics
    
    # 保持向后兼容：detect_semantic_drift 重定向到 detect_drift
    def detect_semantic_drift(
        self,
        current_anchors: torch.Tensor,
        initial_anchors: Optional[torch.Tensor] = None
    ) -> dict:
        """向后兼容接口：重定向到 detect_drift"""
        return self.detect_drift(current_anchors, initial_anchors)
    
    def print_drift_report(self, metrics: dict):
        """打印漂移报告（无分组版本，仅在主进程打印）"""
        if not _is_main_process():
            return
        print("\n" + "=" * 60)
        print("📊 Memory Anchor 漂移检测报告")
        print("=" * 60)
        
        print("\n【概念对齐度】")
        print(f"  Anchor ↔ 概念池 平均相似度: {metrics.get('anchor_to_concepts_mean', 0):.4f}")
        print(f"  Anchor ↔ 最相似概念 平均值: {metrics.get('anchor_to_concepts_max_mean', 0):.4f}")
        
        print("\n【多样性】")
        abs_mean = metrics.get('diversity_offdiag_abs_mean', 0)
        offdiag_max = metrics.get('diversity_offdiag_max', 0)
        print(f"  Anchor 间互相似度 |mean|: {abs_mean:.4f} {'✅' if abs_mean < 0.3 else '⚠️ 偏高，可能坍塌'}")
        print(f"  Anchor 间互相似度 |max|:  {offdiag_max:.4f} {'✅' if offdiag_max < 0.5 else '⚠️ 有高度相似的锚点对'}")
        
        if 'overall_drift' in metrics:
            print("\n【漂移分析】")
            drift = metrics['overall_drift']
            print(f"  整体漂移: {drift:.4f} {'✅' if drift < 0.5 else '⚠️ 漂移较大'}")
            print(f"  逐 anchor 漂移 mean: {metrics.get('per_anchor_drift_mean', 0):.4f}")
            print(f"  逐 anchor 漂移 max:  {metrics.get('per_anchor_drift_max', 0):.4f}")
            print(f"  逐 anchor 漂移 min:  {metrics.get('per_anchor_drift_min', 0):.4f}")
        
        print("\n【诊断建议】")
        if abs_mean > 0.3:
            print("  ⚠️  锚点多样性不足，建议：")
            print("     1. 增大 λ_orth（正交正则化权重）")
            print("     2. 检查学习率是否过大导致坍塌")
            print("     3. 增加训练数据的多样性")
        elif metrics.get('overall_drift', 0) > 0.7:
            print("  ⚠️  漂移过大，锚点偏离初始位置较远，建议：")
            print("     1. 增大 λ_orth")
            print("     2. 降低学习率")
            print("     3. 检查训练数据质量")
        else:
            print("  ✅ 锚点多样性良好，训练状态正常！")
        
        print("=" * 60 + "\n")


# ============================================================================
# 便捷函数
# ============================================================================

def initialize_memory_anchors(
    model,
    tokenizer,
    num_mem_tokens: int = 64,
    std_scale: float = 1.0,
    **kwargs  # 向后兼容：忽略旧版 semantic_ratio 参数
) -> torch.Tensor:
    """
    便捷函数：初始化 Memory Anchors（统一概念词池，无分组）
    
    Args:
        model: ColaMem 模型或基座模型
        tokenizer: Tokenizer
        num_mem_tokens: Memory Anchors 总数
        std_scale: 标准差缩放因子
        **kwargs: 向后兼容（忽略旧版 semantic_ratio 等参数）
    
    Returns:
        mem_anchors: [1, num_mem_tokens, hidden_size]
    """
    # 向后兼容：提醒用户旧参数已废弃
    if 'semantic_ratio' in kwargs:
        print(f"[MemoryAnchorInitializer] ⚠️ semantic_ratio 参数已废弃（不再使用分组），将被忽略")
    
    # 获取 Embedding Layer
    if hasattr(model, 'encoder'):
        embedding_layer = model.encoder.get_input_embeddings()
    else:
        embedding_layer = model.get_input_embeddings()
    
    # 创建初始化器
    initializer = MemoryAnchorInitializer(
        embedding_layer=embedding_layer,
        tokenizer=tokenizer,
        num_mem_tokens=num_mem_tokens,
        std_scale=std_scale
    )
    
    # 初始化
    mem_anchors = initializer.initialize()
    
    return mem_anchors


def create_orthogonality_regularizer(
    lambda_orth: float = 0.05,
    normalize: bool = True,
    warmup_steps: int = 0
) -> OrthogonalityRegularizer:
    """
    便捷函数：创建正交正则化器
    
    Args:
        lambda_orth: 正交正则化权重（建议 0.01~0.1）
        normalize: 是否归一化
        warmup_steps: warmup 步数（建议设为总步数的 5%~10%）
    
    Returns:
        regularizer: OrthogonalityRegularizer 实例
    """
    return OrthogonalityRegularizer(
        lambda_orth=lambda_orth,
        normalize=normalize,
        warmup_steps=warmup_steps
    )


def create_explainability_probe(
    model,
    tokenizer,
    **kwargs  # 向后兼容：忽略旧版 num_semantic 参数
) -> MemoryExplainabilityProbe:
    """
    便捷函数：创建可解释性探针（无分组版本）
    
    Args:
        model: ColaMem 模型或基座模型
        tokenizer: Tokenizer
        **kwargs: 向后兼容（忽略旧版 num_semantic 等参数）
    
    Returns:
        probe: MemoryExplainabilityProbe 实例
    """
    # 向后兼容：提醒用户旧参数已废弃
    if 'num_semantic' in kwargs:
        print(f"[MemoryExplainabilityProbe] ⚠️ num_semantic 参数已废弃（不再使用分组），将被忽略")
    
    # 获取 Embedding Layer
    if hasattr(model, 'encoder'):
        embedding_layer = model.encoder.get_input_embeddings()
    else:
        embedding_layer = model.get_input_embeddings()
    
    return MemoryExplainabilityProbe(
        embedding_layer=embedding_layer,
        tokenizer=tokenizer
    )


# ============================================================================
# 模块四：Memory 输出可分性探针 (Memory Separability Probe)
# ============================================================================

class MemorySeparabilityProbe:
    """
    Memory 输出可分性探针
    
    核心目的：监控 batch 内不同样本的 latent_memory 输出是否可区分。
    
    为什么需要这个 probe：
    - Anchor 参数保持多样 ≠ latent_memory 输出多样
    - Writer Encoder 可能学到退化映射："不管输入是什么，输出都很像"
    - mean≈0/std≈1 可能只是 RMSNorm 的正常行为，不代表坍塌
    - 真正需要监控的是："同一 batch 内，不同图片的 memory 是否可区分"
    
    监控指标：
    - inter_sample_cosine_mean: batch 内不同样本 memory 的平均余弦相似度（越低越好）
    - inter_sample_cosine_max: 最相似的一对样本（应 < 0.9）
    - retrieval_at_1: 自检索准确率（每个样本与自身应该最相似，理想值=1.0）
    - slot_usage_entropy: slot 使用率的熵（越高越说明 64 个 slot 都被利用了）
    
    Example:
        ```python
        sep_probe = MemorySeparabilityProbe()
        
        # 在 training_step 中（每 N 步）
        with torch.no_grad():
            metrics = sep_probe.compute_separability(latent_memory)  # [B, 64, H]
            print(f"Inter-sample sim: {metrics['inter_sample_cosine_mean']:.4f}")
            print(f"Retrieval@1: {metrics['retrieval_at_1']:.4f}")
        ```
    """
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
    
    def compute_separability(self, latent_memory: torch.Tensor) -> dict:
        """
        计算 batch 内 memory 输出的可分性指标
        
        Args:
            latent_memory: [B, num_mem, hidden_size] — Encoder 输出的 memory tokens
        
        Returns:
            metrics: {
                'inter_sample_cosine_mean': 样本间平均余弦相似度（越低越多样）,
                'inter_sample_cosine_max': 样本间最大余弦相似度,
                'inter_sample_cosine_min': 样本间最小余弦相似度,
                'retrieval_at_1': 自检索准确率（理想值 1.0）,
                'slot_variance_mean': 每个 slot 在 batch 内的方差均值（越高越好）,
                'effective_rank': 有效秩（越高说明信息利用越充分）,
            }
        """
        with torch.no_grad():
            B, num_mem, H = latent_memory.shape
            
            if B < 2:
                # batch size < 2 无法计算样本间指标
                return {
                    'inter_sample_cosine_mean': 0.0,
                    'inter_sample_cosine_max': 0.0,
                    'inter_sample_cosine_min': 0.0,
                    'retrieval_at_1': 1.0,
                    'slot_variance_mean': 0.0,
                    'effective_rank': 0.0,
                }
            
            # 强制 FP32 计算
            mem_f32 = latent_memory.to(torch.float32)
            
            # === 1. 样本间余弦相似度 ===
            # 将每个样本的 memory flatten 成一个向量：[B, num_mem*H]
            mem_flat = mem_f32.reshape(B, -1)  # [B, num_mem*H]
            mem_flat_norm = F.normalize(mem_flat, p=2, dim=1, eps=self.eps)
            
            # 样本间余弦相似度矩阵：[B, B]
            sim_matrix = torch.matmul(mem_flat_norm, mem_flat_norm.t())
            
            # 提取 off-diagonal（不同样本间的相似度）
            offdiag_mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
            offdiag = sim_matrix[offdiag_mask]
            
            inter_sample_cosine_mean = offdiag.mean().item()
            inter_sample_cosine_max = offdiag.max().item()
            inter_sample_cosine_min = offdiag.min().item()
            
            # === 2. 自检索准确率 (Retrieval@1) ===
            # 对角线应该是每行最大值（每个样本与自身最相似）
            # 如果不是，说明某些样本的 memory 被其他样本"覆盖"了
            max_indices = sim_matrix.argmax(dim=1)  # [B]
            correct = (max_indices == torch.arange(B, device=max_indices.device)).float()
            retrieval_at_1 = correct.mean().item()
            
            # === 3. 每个 slot 在 batch 内的方差 ===
            # 如果某个 slot 对所有样本输出相同的值，说明它没有编码样本特异信息
            # [B, num_mem, H] -> 对 B 维取方差 -> [num_mem, H] -> 对 H 取均值 -> [num_mem]
            per_slot_var = mem_f32.var(dim=0).mean(dim=1)  # [num_mem]
            slot_variance_mean = per_slot_var.mean().item()
            
            # === 4. 有效秩 (Effective Rank) ===
            # 用 SVD 衡量 memory 表征的信息利用程度
            # 有效秩越高 = 信息分布越均匀 = 所有 slot 都在干活
            try:
                # 对 batch 内所有 memory flatten 后做 SVD
                # [B, num_mem*H] -> SVD
                singular_values = torch.linalg.svdvals(mem_flat)
                # 归一化为概率分布
                sv_norm = singular_values / (singular_values.sum() + self.eps)
                # 计算熵
                entropy = -(sv_norm * torch.log(sv_norm + self.eps)).sum()
                # 有效秩 = exp(entropy)
                effective_rank = torch.exp(entropy).item()
            except:
                effective_rank = 0.0
            
            metrics = {
                'inter_sample_cosine_mean': inter_sample_cosine_mean,
                'inter_sample_cosine_max': inter_sample_cosine_max,
                'inter_sample_cosine_min': inter_sample_cosine_min,
                'retrieval_at_1': retrieval_at_1,
                'slot_variance_mean': slot_variance_mean,
                'effective_rank': effective_rank,
            }
        
        return metrics
    
    def print_separability_report(self, metrics: dict, step: int = 0):
        """打印可分性报告"""
        print(f"\n{'='*60}")
        print(f"🔬 Step {step} - Memory 输出可分性监控")
        print(f"{'='*60}")
        
        cosine_mean = metrics.get('inter_sample_cosine_mean', 0)
        cosine_max = metrics.get('inter_sample_cosine_max', 0)
        retrieval = metrics.get('retrieval_at_1', 0)
        
        print(f"【样本间区分度】")
        print(f"  余弦相似度 mean: {cosine_mean:.4f} {'✅' if cosine_mean < 0.7 else '⚠️ 偏高，memory 区分度不足'}")
        print(f"  余弦相似度 max:  {cosine_max:.4f} {'✅' if cosine_max < 0.9 else '⚠️ 有几乎相同的 memory 对'}")
        print(f"  余弦相似度 min:  {metrics.get('inter_sample_cosine_min', 0):.4f}")
        
        print(f"\n【自检索准确率】")
        print(f"  Retrieval@1: {retrieval:.4f} {'✅' if retrieval >= 0.9 else '⚠️ memory 无法区分不同输入'}")
        
        print(f"\n【信息利用度】")
        print(f"  Slot 方差均值:  {metrics.get('slot_variance_mean', 0):.6f}")
        print(f"  有效秩:         {metrics.get('effective_rank', 0):.2f}")
        
        print(f"\n【诊断建议】")
        if cosine_mean > 0.8:
            print(f"  🚨 严重：不同图片的 memory 几乎相同！")
            print(f"     → Writer Encoder 可能学到了退化映射")
            print(f"     → 建议：增大 MMR Loss 权重 / 检查数据多样性")
        elif retrieval < 0.5:
            print(f"  ⚠️  自检索准确率低，memory 区分度不足")
            print(f"     → 建议：增大 MMR Loss 的 gamma / 检查 Projector 容量")
        else:
            print(f"  ✅ Memory 输出区分度良好！")
        
        print(f"{'='*60}\n")
