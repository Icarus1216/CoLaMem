
# CoLaMem: Compressed Latent Memory for Vision-Language Models

<p align="center">
  <b>将视觉上下文压缩为潜在记忆，实现高效的多模态问答</b>
</p>

<p align="center">
  <a href="#架构设计">架构设计</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#训练">训练</a> •
  <a href="#推理">推理</a> •
  <a href="#配置说明">配置说明</a>
</p>

---

## 简介

CoLaMem (Compressed Latent Memory) 是一种基于双模型架构的视觉语言模型训练框架。其核心思想是将视觉上下文信息（图像 + 文本）压缩为一组**潜在记忆向量 (Latent Memory)**，然后 Decoder 仅凭这些记忆向量和用户问题生成回答，无需再次访问原始图像。

### 核心特性

- 🧠 **Latent Memory 机制**：将图文上下文压缩为 64 个记忆向量，实现信息的高效编码
- 🔀 **双模型架构**：Encoder（压缩器）和 Decoder（求解器）各自独立，支持全参数微调和 LoRA 微调
- 🛡️ **Anti-Bypass 策略**：Memory Mismatch Ranking Loss + Answer-Prefix Dropout，防止 Decoder 绕过 Memory
- 🔬 **可解释性监控**：Memory Anchor 漂移检测、正交正则化、可分性探针
- ⚡ **高效训练**：LoRA 模式下可训练参数减少 99.7%（~8B → ~25M），显存从 ~170GB 降至 ~40GB

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     CoLaMem 训练流程                              │
│                                                                 │
│  Phase 1: Compress (Encoder)                                    │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐  │
│  │ Image + │───▶│ Qwen3-VL │───▶│ Projector │───▶│  Latent  │  │
│  │  Text   │    │ Encoder  │    │  (MLP)    │    │  Memory  │  │
│  └─────────┘    └──────────┘    └───────────┘    │ [64, H]  │  │
│                                                   └────┬─────┘  │
│                                                        │        │
│  Phase 2: Solve (Decoder)                              ▼        │
│  ┌─────────┐    ┌──────────────────────────────────────────┐    │
│  │Question │───▶│ [Latent Memory] + [Query] + [Workspace] │    │
│  └─────────┘    │          ▼                               │    │
│                 │    Qwen3-VL Decoder ──▶ Answer           │    │
│                 └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 关键组件

| 组件 | 说明 |
|------|------|
| **Encoder (Compressor)** | 基于 Qwen3-VL-8B，接收图像+文本，输出 Hidden States |
| **Memory Anchors** | 64 个可学习的潜在向量，通过语义概念词池初始化 |
| **Projector** | MLP 投影层，将 Encoder 隐藏状态映射到 Memory 空间 |
| **Decoder (Solver)** | 基于 Qwen3-VL-8B，从 Latent Memory + Query 生成回答 |
| **Workspace** | 可选的可学习"草稿纸"向量（Phase 2 启用，辅助复杂推理） |

### 训练模式对比

| | 全参数微调 (Full FT) | LoRA 微调 (PEFT) |
|---|---|---|
| **可训练参数** | ~8B | ~25M |
| **训练显存** | ~170GB (ZeRO-3) | ~40GB (ZeRO-2) |
| **DeepSpeed** | 需要 ZeRO-3 + CPU Offload | 可用 DDP / ZeRO-2 |
| **训练速度** | 基准 | ~5-6x 更快 |
| **性能** | 最大上限 | 接近全参 |

## 目录结构

```
CoLaMem/
├── configs/                          # 配置文件
│   ├── train_phase1.yaml             # Phase 1 全参数微调配置
│   ├── train_phase1_lora.yaml        # Phase 1 LoRA 微调配置（推荐）
│   ├── train_phase2.yaml             # Phase 2 VQA 任务配置
│   ├── ds_config_zero2.json          # DeepSpeed ZeRO-2 配置（LoRA 模式）
│   └── ds_zero3_config.json          # DeepSpeed ZeRO-3 配置（全参模式）
├── src/                              # 核心源代码
│   ├── models/
│   │   ├── modeling.py               # ColaMemModel 双模型架构
│   │   ├── modeling_qwen3_vl.py      # 自定义 Qwen3-VL 模型修改
│   │   ├── processing_qwen3_vl.py    # 自定义 Qwen3-VL 处理器
│   │   └── memory_initialization.py  # Memory Anchor 初始化与软约束
│   ├── data/
│   │   ├── data_engine.py            # 数据集加载与预处理
│   │   └── data_collator.py          # 批次数据组装（Left/Right Padding）
│   ├── engine/
│   │   └── trainer.py                # 自定义 Trainer（漂移监控 + 多学习率）
│   └── train/
│       └── LlamaFactory/             # LlamaFactory 子模块（Git Submodule）
├── scripts/                          # 脚本工具
│   ├── train.py                      # 训练入口脚本
│   ├── inference.py                  # 推理脚本
│   ├── inference_debug.py            # 推理调试脚本（详细输出）
│   ├── convert_*_to_colamem.py       # 数据格式转换工具
│   └── process_*.py                  # 数据预处理工具
├── start_training.sh                 # 一键训练启动脚本
├── infer.sh                          # 一键推理脚本
├── download_data.sh                  # 数据下载脚本
├── requirements.txt                  # Python 依赖
└── LICENSE                           # Apache 2.0 许可证
```

## 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.9
- CUDA >= 12.x
- GPU 显存：LoRA 模式 ≥ 40GB（推荐 8×H20/A100）；全参模式 ≥ 170GB

### 安装

```bash
# 克隆仓库（含子模块）
git clone --recursive https://github.com/Icarus1216/CoLaMem.git
cd CoLaMem

# 创建 conda 环境
conda create -n colamem python=3.10 -y
conda activate colamem

# 安装依赖
pip install -r requirements.txt
```

### 核心依赖

| 包 | 版本 | 说明 |
|---|---|---|
| `transformers` | 4.57.6 | Qwen3-VL 模型支持 |
| `peft` | 0.18.1 | LoRA 微调 |
| `deepspeed` | 0.18.4 | 分布式训练 |
| `flash_attn` | 2.8.3 | Flash Attention 加速 |
| `accelerate` | 1.12.0 | 训练加速 |
| `qwen-vl-utils` | 0.0.14 | Qwen-VL 视觉处理工具 |

## 训练

### 数据格式

CoLaMem 使用 JSON 格式的训练数据，每条样本包含 `compress`（上下文）和 `solve`（问答）两部分：

```json
{
  "compress": {
    "content": [
      {"type": "image", "path": "/path/to/image.jpg"},
      {"type": "text", "text": "Observe and memorize."}
    ]
  },
  "solve": {
    "question": "Describe the image in detail.",
    "answer": "The image shows a ..."
  }
}
```

支持的数据集：ShareGPT4V、GQA、TextVQA、ChartQA 等。使用 `scripts/convert_*_to_colamem.py` 进行格式转换。

### 多阶段训练策略

| 阶段 | 任务 | 目标 | 关键设置 |
|------|------|------|---------|
| **Phase 1** | 图像描述重建 | Projector 对齐，Loss ~2.0 → ~0.8 | 冻结 Vision + Decoder，禁用 Workspace |
| **Phase 2** | 视觉问答 (VQA) | 复杂推理能力 | 继承 Phase 1，启用 Workspace |

### Phase 1 训练（LoRA 模式，推荐）

```bash
# 方式 1：一键启动
bash start_training.sh

# 方式 2：手动启动（支持参数覆盖）
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/train_phase1_lora.yaml

# 方式 3：覆盖关键参数
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/train_phase1_lora.yaml \
    --output_dir ./my_output \
    --learning_rate 1e-4 \
    --num_train_epochs 3
```

### Phase 1 训练（全参数微调）

```bash
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/train_phase1.yaml
```

### 从 Checkpoint 恢复训练

```bash
deepspeed --num_gpus=8 scripts/train.py \
    --config configs/train_phase1_lora.yaml \
    --resume_from_checkpoint ./outputs/colamem_phase1_lora/checkpoint-200
```

## 推理

```bash
# 一键推理
bash infer.sh

# 手动推理
python scripts/inference_debug.py \
    --checkpoint ./outputs/colamem_phase0.1_lora/model \
    --base_model /path/to/Qwen3-VL-8B-Instruct \
    --val_json /path/to/colamem_val.json \
    --num_samples 20 \
    --max_new_tokens 256 \
    --temperature 0 \
    --output ./results.json \
    --device cuda
```

## 配置说明

### LoRA 配置（`configs/train_phase1_lora.yaml`）

CoLaMem 支持 Encoder 和 Decoder **独立配置** LoRA 参数：

```yaml
model:
  use_lora: true
  
  # Encoder LoRA（信息压缩，推荐较大 rank）
  encoder_lora: true
  encoder_lora_r: 64           # rank
  encoder_lora_alpha: 128      # alpha = 2 * r
  encoder_lora_dropout: 0.05
  
  # Decoder LoRA（回答生成，可用较小 rank）
  decoder_lora: true
  decoder_lora_r: 16
  decoder_lora_alpha: 32
  decoder_lora_dropout: 0.05
```

### 独立学习率

支持为不同组件设置独立学习率：

```yaml
training:
  learning_rate: 2.0e-4      # 默认学习率
  encoder_lr: 2.0e-4         # Encoder (Compressor)
  decoder_lr: 2.0e-5         # Decoder (Solver)
  anchors_lr: 6.0e-4         # Memory Anchors (3x encoder)
  projector_lr: 4.0e-4       # Projector (2x encoder)
  gate_lr: 1.0e-4            # Residual Gate
  scale_lr: 5.0e-5           # Memory Scale
```

### Anti-Bypass 策略

防止 Decoder 绕过 Memory 依靠语言先验生成回答：

```yaml
model:
  # Memory Mismatch Ranking Loss
  use_mmr_loss: true
  mmr_gamma: 0.5              # Ranking margin
  mmr_beta: 0.5               # Loss 权重（最终值）
  mmr_beta_start: 0.1         # Warmup 起始值
  mmr_beta_warmup_steps: 150
  
  # Answer-Prefix Dropout
  use_answer_prefix_dropout: true
  answer_prefix_dropout_ratio: 0.3
  answer_prefix_dropout_prob: 0.5
```

### Memory 初始化

使用语义概念词池进行多样化初始化，配合正交正则化防止模式坍塌：

```yaml
model:
  use_semantic_init: true     # 统一概念词池初始化（~110 个概念词）
  lambda_orth: 0.08           # 正交正则化权重
  orth_warmup_steps: 100      # 正交正则化 Warmup
```

## 训练监控

训练过程中自动进行以下监控：

- 📊 **Memory Anchor 漂移检测**：监控 anchor 与初始位置的余弦距离变化
- 🔬 **多样性监控**：anchor 间 off-diagonal 余弦相似度（目标 < 0.3）
- 📈 **TensorBoard 日志**：Loss 曲线、学习率、梯度范数等

```bash
# 查看训练日志
tensorboard --logdir ./outputs/colamem_phase1_lora/logs
```

## 技术细节

### Memory Anchor 初始化策略

采用 **Prototype-Jitter** 策略：从 ~110 个统一概念词池（涵盖语义、空间、视觉、逻辑等）中为每个 anchor 选取一个概念 embedding 作为原型，再加小幅随机扰动打破对称性。不采用人工分组，让模型自由学习最优的信息编码方式。

### 正交正则化

使用 Hinge 式正交正则化，只在 anchor 间余弦相似度超过阈值（τ=0.1）时惩罚，避免在 anchor 已充分分散时浪费梯度。支持 Warmup，训练初期不施加约束。

### Memory Mismatch Ranking Loss

通过打乱 batch 内样本的 Memory 分配，构造"错误 Memory"的负样本。如果 Decoder 在错误 Memory 下也能流畅生成，说明它在绕过 Memory 走捷径。Ranking Loss 确保正确 Memory 下的 Loss 显著低于错误 Memory。

## License

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — 基座视觉语言模型
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory) — LLM 微调工具
- [PEFT](https://github.com/huggingface/peft) — 参数高效微调
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — 分布式训练加速
