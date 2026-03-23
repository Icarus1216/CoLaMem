"""
ColaMem Advanced Trainer

扩展的 Trainer，支持软约束训练和语义漂移监控

核心特性：
1. ✅ Memory Anchor 漂移监控：定期检测 Memory Anchors 的漂移和多样性
2. ✅ 可解释性验证：监控 anchor 间多样性和概念对齐度
3. ✅ 自动报告生成：保存漂移历史和最终报告
4. ✅ 兼容 DDP 和 DeepSpeed（包括 ZeRO-3）
5. ✅ 梯度监控：检测关键组件的梯度状态
6. ✅ 自定义优化器：确保 mem_anchors 等参数被正确优化

使用方法：
```python
from src.engine.trainer import ColaMemTrainerWithMonitoring
from transformers import TrainingArguments

trainer = ColaMemTrainerWithMonitoring(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    monitor_drift_every_n_steps=100,  # 每 100 步监控一次
    save_drift_report=True            # 保存漂移报告
)

trainer.train()
```

数据格式要求：
```python
batch = {
    "compress_input_ids": torch.LongTensor,      # [batch, seq_len]
    "compress_attention_mask": torch.LongTensor, # [batch, seq_len]
    "pixel_values": torch.FloatTensor,           # [num_patches, patch_dim]
    "image_grid_thw": torch.LongTensor,          # [num_images, 3]
    "query_input_ids": torch.LongTensor,         # [batch, seq_len] - Query 部分
    "query_attention_mask": torch.LongTensor,    # [batch, seq_len]
    "query_labels": torch.LongTensor,            # [batch, seq_len] - 全 -100
    "answer_input_ids": torch.LongTensor,        # [batch, seq_len] - Answer 部分
    "answer_attention_mask": torch.LongTensor,   # [batch, seq_len]
    "answer_labels": torch.LongTensor,           # [batch, seq_len] - 计算 loss
}

拼接顺序: [Latent Memory, Query, Workspace, Answer]
```

注意：
- 对于基础训练（不需要监控），可以直接使用标准的 transformers.Trainer
- 本 Trainer 主要用于研究和实验，需要启用语义初始化才能进行漂移监控
"""

import os
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from contextlib import nullcontext


class ColaMemTrainerWithMonitoring(Trainer):
    """
    扩展的 Trainer，添加语义漂移监控和自定义参数优化
    
    适用场景：
    - 使用非对称语义初始化训练
    - 需要监控训练过程中的语义漂移
    - 研究和实验阶段
    
    参数：
        monitor_drift_every_n_steps: 每 N 步监控一次语义漂移（默认 100）
        save_drift_report: 是否保存漂移报告（默认 True）
        debug_inputs: 是否打印输入调试信息（默认 False）
        debug_inputs_every_n_steps: 每 N 步打印一次输入调试信息（默认 10）
    """
    
    def __init__(self, *args, monitor_drift_every_n_steps=100, save_drift_report=True, 
                 debug_inputs=False, debug_inputs_every_n_steps=10,
                 encoder_lr=None, decoder_lr=None,
                 anchors_lr=None, projector_lr=None, gate_lr=None, scale_lr=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor_drift_every_n_steps = monitor_drift_every_n_steps
        self.save_drift_report = save_drift_report
        self.debug_inputs = debug_inputs
        self.debug_inputs_every_n_steps = debug_inputs_every_n_steps
        self.drift_history = []
        self._gradient_hooks = []
        self._last_grad_norms = {}  # 存储最后记录的梯度范数
        
        # 防止 gradient accumulation 下同一 global_step 重复触发监控
        self._last_drift_monitored_step = -1
        self._last_input_debug_step = -1
        
        # Encoder / Decoder 独立学习率
        # 如果未指定，默认使用 training_args.learning_rate
        self.encoder_lr = encoder_lr if encoder_lr is not None else self.args.learning_rate
        self.decoder_lr = decoder_lr if decoder_lr is not None else self.args.learning_rate
        
        # 自定义组件独立学习率（如果未指定，按 encoder_lr 倍率自动计算）
        self.anchors_lr = anchors_lr if anchors_lr is not None else self.encoder_lr * 3
        self.projector_lr = projector_lr if projector_lr is not None else self.encoder_lr * 2
        self.gate_lr = gate_lr if gate_lr is not None else self.encoder_lr * 0.5
        self.scale_lr = scale_lr if scale_lr is not None else self.encoder_lr * 0.25
        
        # 注册梯度钩子，用于调试
        self._register_gradient_hooks()
    
    def _register_gradient_hooks(self):
        """注册梯度钩子，用于追踪自定义参数的梯度"""
        unwrapped_model = self.accelerator.unwrap_model(self.model) if hasattr(self, 'accelerator') else self.model
        
        def make_hook(name):
            def hook(grad):
                if grad is not None:
                    self._last_grad_norms[name] = grad.norm().item()
                return grad
            return hook
        
        # 为自定义参数注册钩子
        if hasattr(unwrapped_model, 'mem_anchors'):
            handle = unwrapped_model.mem_anchors.register_hook(make_hook('mem_anchors'))
            self._gradient_hooks.append(handle)
        
        # work_tokens 只在启用 workspace 时注册钩子
        if hasattr(unwrapped_model, 'work_tokens') and getattr(unwrapped_model, 'use_workspace', True):
            handle = unwrapped_model.work_tokens.register_hook(make_hook('work_tokens'))
            self._gradient_hooks.append(handle)
        
        # 为 Projector 参数注册钩子
        if hasattr(unwrapped_model, 'projector'):
            for name, param in unwrapped_model.projector.named_parameters():
                if param.requires_grad:
                    hook_name = f'projector.{name}'
                    handle = param.register_hook(make_hook(hook_name))
                    self._gradient_hooks.append(handle)
        
        # 为 residual_gate_logit 和 memory_scale 注册钩子
        if hasattr(unwrapped_model, 'residual_gate_logit'):
            handle = unwrapped_model.residual_gate_logit.register_hook(make_hook('residual_gate_logit'))
            self._gradient_hooks.append(handle)
        if hasattr(unwrapped_model, 'memory_scale'):
            handle = unwrapped_model.memory_scale.register_hook(make_hook('memory_scale'))
            self._gradient_hooks.append(handle)
    
    def create_optimizer(self):
        """
        创建优化器，确保 mem_anchors、work_tokens 和 projector 参数被正确添加
        
        这是解决 DeepSpeed ZeRO-3 下自定义参数梯度问题的关键！
        """
        # 获取模型（处理 DeepSpeed 包装）
        model = self.model
        
        # 如果优化器已创建，直接返回
        if self.optimizer is not None:
            return self.optimizer
        
        # 获取需要 decay 的参数名（排除 bias 和所有 norm 层）
        # 重要修复：Qwen 系列和 ColaMem 自定义模块使用 RMSNorm 而非 LayerNorm
        # 必须同时排除两者，否则 RMSNorm 的 weight 会被错误地施加 weight decay
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters 
                           if "bias" not in name and "norm" not in name.lower()]
        
        # 获取 unwrapped model 以访问自定义参数
        unwrapped_model = self.accelerator.unwrap_model(model) if hasattr(self, 'accelerator') else model
        
        # 收集自定义参数名称（用于排除重复）
        custom_param_names = set()
        if hasattr(unwrapped_model, 'get_custom_named_parameters'):
            for name, _ in unwrapped_model.get_custom_named_parameters():
                custom_param_names.add(name)
        
        # 构建参数组
        optimizer_grouped_parameters = []
        
        # 1. Encoder 和 Decoder 的参数（支持独立学习率）
        encoder_decay = []
        encoder_no_decay = []
        decoder_decay = []
        decoder_no_decay = []
        
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 跳过自定义参数（它们会单独处理）
            # 注意：work_tokens 只在启用 workspace 时才是自定义参数
            if 'mem_anchors' in n or 'projector' in n:
                continue
            if 'residual_gate_logit' in n or 'memory_scale' in n:
                continue
            if 'work_tokens' in n and getattr(unwrapped_model, 'use_workspace', True):
                continue
            
            # 区分 encoder 和 decoder 参数
            is_encoder = 'encoder' in n
            is_decoder = 'decoder' in n
            
            if is_encoder:
                if n in decay_parameters:
                    encoder_decay.append(p)
                else:
                    encoder_no_decay.append(p)
            elif is_decoder:
                if n in decay_parameters:
                    decoder_decay.append(p)
                else:
                    decoder_no_decay.append(p)
            else:
                # 其他参数（如果有）归入 encoder 组
                if n in decay_parameters:
                    encoder_decay.append(p)
                else:
                    encoder_no_decay.append(p)
        
        # Encoder 参数组
        if encoder_decay:
            optimizer_grouped_parameters.append({
                "params": encoder_decay,
                "weight_decay": self.args.weight_decay,
                "lr": self.encoder_lr,
                "name": "encoder_decay",
            })
        
        if encoder_no_decay:
            optimizer_grouped_parameters.append({
                "params": encoder_no_decay,
                "weight_decay": 0.0,
                "lr": self.encoder_lr,
                "name": "encoder_no_decay",
            })
        
        # Decoder 参数组
        if decoder_decay:
            optimizer_grouped_parameters.append({
                "params": decoder_decay,
                "weight_decay": self.args.weight_decay,
                "lr": self.decoder_lr,
                "name": "decoder_decay",
            })
        
        if decoder_no_decay:
            optimizer_grouped_parameters.append({
                "params": decoder_no_decay,
                "weight_decay": 0.0,
                "lr": self.decoder_lr,
                "name": "decoder_no_decay",
            })
        
        # 打印 encoder/decoder 学习率信息（只在主进程打印）
        enc_params = sum(p.numel() for p in encoder_decay + encoder_no_decay)
        dec_params = sum(p.numel() for p in decoder_decay + decoder_no_decay)
        is_main = self.accelerator.is_main_process if hasattr(self, 'accelerator') else True
        if is_main:
            print(f"[Trainer] Encoder 学习率: {self.encoder_lr:.2e} (参数量: {enc_params/1e6:.2f}M)")
            print(f"[Trainer] Decoder 学习率: {self.decoder_lr:.2e} (参数量: {dec_params/1e6:.2f}M)")
        
        # 2. 自定义参数（mem_anchors, work_tokens, projector）
        # 各组件使用独立配置的学习率（在 __init__ 中已计算好）
        anchors_lr = self.anchors_lr
        proj_lr = self.projector_lr
        gate_lr = self.gate_lr
        scale_lr = self.scale_lr
        
        if hasattr(unwrapped_model, 'mem_anchors'):
            optimizer_grouped_parameters.append({
                "params": [unwrapped_model.mem_anchors],
                "weight_decay": 0.0,  # mem_anchors 不使用 weight decay
                "lr": anchors_lr,
                "name": "mem_anchors",  # 用于调试
            })
            if is_main:
                print(f"[Trainer] ✅ mem_anchors 已添加到优化器 (lr={anchors_lr:.2e})")
        
        # work_tokens 只在启用 workspace 时添加到优化器
        if hasattr(unwrapped_model, 'work_tokens') and getattr(unwrapped_model, 'use_workspace', True):
            optimizer_grouped_parameters.append({
                "params": [unwrapped_model.work_tokens],
                "weight_decay": 0.0,
                "lr": anchors_lr,
                "name": "work_tokens",
            })
            if is_main:
                print(f"[Trainer] ✅ work_tokens 已添加到优化器 (lr={anchors_lr:.2e})")
        
        if hasattr(unwrapped_model, 'projector'):
            projector_params_decay = []
            projector_params_no_decay = []
            proj_decay_names = []
            proj_no_decay_names = []
            
            # 收集 nn.Sequential 中所有 norm 类模块的参数 id
            # 重要修复：nn.Sequential 中 RMSNorm 的参数名是 '1.weight'（数字索引），
            # 不含 'norm' 字符串，所以必须用模块类型检查而非名字匹配
            norm_param_ids = set()
            for module in unwrapped_model.projector.modules():
                if isinstance(module, (torch.nn.LayerNorm,)) or type(module).__name__ in ('RMSNorm', 'Qwen3RMSNorm'):
                    for p in module.parameters():
                        norm_param_ids.add(id(p))
            
            for n, p in unwrapped_model.projector.named_parameters():
                if p.requires_grad:
                    if 'bias' in n or id(p) in norm_param_ids:
                        projector_params_no_decay.append(p)
                        proj_no_decay_names.append(n)
                    else:
                        projector_params_decay.append(p)
                        proj_decay_names.append(n)
            
            if projector_params_decay:
                optimizer_grouped_parameters.append({
                    "params": projector_params_decay,
                    "weight_decay": self.args.weight_decay,
                    "lr": proj_lr,
                    "name": "projector_decay",
                })
            if projector_params_no_decay:
                optimizer_grouped_parameters.append({
                    "params": projector_params_no_decay,
                    "weight_decay": 0.0,
                    "lr": proj_lr,
                    "name": "projector_no_decay",
                })
            
            # 打印 projector 参数分组详情（帮助确认 RMSNorm weight 未被误放入 decay）
            if is_main:
                print(f"[Trainer]   projector_decay 参数: {proj_decay_names}")
                print(f"[Trainer]   projector_no_decay 参数: {proj_no_decay_names}")
                print(f"[Trainer] ✅ projector 已添加到优化器 (lr={proj_lr:.2e})")
        
        # memory_norm 已移除（方案A），不再添加到优化器
        
        # residual_gate_logit 参数（可学习的残差门控）
        if hasattr(unwrapped_model, 'residual_gate_logit'):
            optimizer_grouped_parameters.append({
                "params": [unwrapped_model.residual_gate_logit],
                "weight_decay": 0.0,
                "lr": gate_lr,
                "name": "residual_gate",
            })
            if is_main:
                print(f"[Trainer] ✅ residual_gate_logit 已添加到优化器 (lr={gate_lr:.2e})")
        
        # memory_scale 参数（输出尺度匹配因子）
        if hasattr(unwrapped_model, 'memory_scale'):
            optimizer_grouped_parameters.append({
                "params": [unwrapped_model.memory_scale],
                "weight_decay": 0.0,
                "lr": scale_lr,
                "name": "memory_scale",
            })
            if is_main:
                print(f"[Trainer] ✅ memory_scale 已添加到优化器 (lr={scale_lr:.2e})")
        
        # 创建优化器
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)
        
        # 移除 lr 参数，因为我们在参数组中已经指定了
        if 'lr' in optimizer_kwargs:
            del optimizer_kwargs['lr']
        
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        # 打印优化器统计信息（只在主进程）
        total_params = sum(
            sum(p.numel() for p in group['params']) 
            for group in optimizer_grouped_parameters
        )
        if is_main:
            print(f"[Trainer] 优化器参数组数量: {len(optimizer_grouped_parameters)}")
            print(f"[Trainer] 优化器总参数量: {total_params/1e9:.2f}B")
        
        return self.optimizer
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """重写训练步骤，添加监控
        
        注意：gradient_accumulation_steps > 1 时，training_step() 每个 global_step
        会被调用多次（每个 micro-batch 一次），但 global_step 只在 optimizer.step()
        之后才 +1。因此需要 guard 防止同一 global_step 重复触发监控。
        """
        # 定期打印输入调试信息（在训练之前，每个 global_step 只触发一次）
        if self.debug_inputs and self.state.global_step % self.debug_inputs_every_n_steps == 0:
            if self.state.global_step != self._last_input_debug_step:
                self._last_input_debug_step = self.state.global_step
                self._debug_print_inputs(inputs)
        
        # 将 global_step 注入 inputs，让 model.forward 通过 **kwargs 接收
        # 用于 orth_regularizer 和 MMR loss 的 warmup，避免 gradient accumulation 下 step 过快递增
        inputs['global_step'] = self.state.global_step
        
        # 执行标准训练步骤
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # 定期监控 anchor 漂移和多样性（每个 global_step 只触发一次）
        if self.state.global_step % self.monitor_drift_every_n_steps == 0:
            if self.state.global_step != self._last_drift_monitored_step:
                self._last_drift_monitored_step = self.state.global_step
                # 使用 torch.no_grad() 确保监控不影响计算图
                with torch.no_grad():
                    self._monitor_anchor_drift(model)
                    self._monitor_gradients(model)
                    self._monitor_memory_separability(model)
        
        return loss
    
    def _debug_print_inputs(self, inputs):
        """打印输入数据的关键监控信息（精简版，只报告异常）"""
        if not self.accelerator.is_main_process:
            return
        
        step = self.state.global_step
        problems = []
        
        # 1. 检查图像数据
        if inputs.get('pixel_values') is not None:
            pv = inputs['pixel_values']
            if torch.isnan(pv).any() or torch.isinf(pv).any():
                problems.append(f"pixel_values 含 NaN/Inf!")
            if inputs.get('image_grid_thw') is not None:
                grid = inputs['image_grid_thw']
                total_patches = grid.prod(dim=1).sum().item()
                if int(total_patches) != pv.shape[0]:
                    problems.append(f"grid patches({int(total_patches)}) != pixel_values({pv.shape[0]})")
        elif 'compress_input_ids' in inputs:
            problems.append("无图像输入")
        
        # 2. 检查 answer labels
        if 'answer_labels' in inputs:
            loss_tokens = (inputs['answer_labels'] != -100).sum(dim=1)
            if (loss_tokens == 0).any():
                problems.append(f"有样本 answer loss_tokens=0")
        
        # 只在有异常或前3步时打印
        if problems:
            print(f"[Step {step}] ⚠️ 输入异常: {'; '.join(problems)}")
        elif step < 3:
            bs = inputs.get('compress_input_ids', inputs.get('input_ids', torch.empty(0))).shape[0]
            has_img = '有图' if inputs.get('pixel_values') is not None else '无图'
            print(f"[Step {step}] ✅ 输入正常 (batch={bs}, {has_img})")
    
    def _get_model_num_mem(self):
        """获取模型的 num_mem 参数"""
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            return getattr(unwrapped_model, 'num_mem', '?')
        except:
            return '?'
    
    def _get_model_num_work(self):
        """获取模型的 num_work 参数"""
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            return getattr(unwrapped_model, 'num_work', '?')
        except:
            return '?'
    
    def _monitor_gradients(self, model):
        """监控关键组件的梯度状态（DeepSpeed 安全版本）"""
        # 只在 rank 0 打印
        if not self.accelerator.is_main_process:
            return
        
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        print(f"\n{'='*60}")
        print(f"📈 Step {self.state.global_step} - 梯度监控")
        print(f"{'='*60}")
        
        # 使用钩子记录的梯度范数（更可靠）
        if 'mem_anchors' in self._last_grad_norms:
            grad_norm = self._last_grad_norms['mem_anchors']
            print(f"mem_anchors 梯度范数 (hook): {grad_norm:.6f}")
            if grad_norm == 0:
                print("  ⚠️ 警告: mem_anchors 梯度为 0!")
            elif grad_norm < 1e-8:
                print("  ⚠️ 警告: mem_anchors 梯度非常小!")
            else:
                print("  ✅ mem_anchors 梯度正常!")
        else:
            # 回退到直接检查
            if hasattr(unwrapped_model, 'mem_anchors'):
                param = unwrapped_model.mem_anchors
                if hasattr(param, 'grad') and param.grad is not None:
                    try:
                        grad_norm = param.grad.norm().item()
                        print(f"mem_anchors 梯度范数 (本地): {grad_norm:.6f}")
                        if grad_norm == 0:
                            print("  ⚠️ 警告: mem_anchors 本地梯度为 0!")
                        elif grad_norm < 1e-8:
                            print("  ⚠️ 警告: mem_anchors 梯度非常小!")
                    except Exception as e:
                        print(f"mem_anchors 梯度范数: 无法获取 ({e})")
                else:
                    print(f"mem_anchors 梯度: 不可用 (可能已被 DeepSpeed 处理)")
        
        # 检查 work_tokens 的梯度（仅在启用 workspace 时）
        if getattr(unwrapped_model, 'use_workspace', True):
            if 'work_tokens' in self._last_grad_norms:
                grad_norm = self._last_grad_norms['work_tokens']
                print(f"work_tokens 梯度范数 (hook): {grad_norm:.6f}")
            else:
                if hasattr(unwrapped_model, 'work_tokens'):
                    param = unwrapped_model.work_tokens
                    try:
                        if hasattr(param, 'grad') and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            print(f"work_tokens 梯度范数 (本地): {grad_norm:.6f}")
                        else:
                            print(f"work_tokens 梯度: 不可用")
                    except Exception as e:
                        print(f"work_tokens 梯度范数: 无法获取 ({e})")
        
        # 检查 projector 的梯度（优先使用钩子记录）
        if hasattr(unwrapped_model, 'projector'):
            # 收集所有 projector 相关的梯度
            projector_grads = {k: v for k, v in self._last_grad_norms.items() if k.startswith('projector.')}
            
            if projector_grads:
                # 计算总梯度范数
                total_grad_norm = sum(v ** 2 for v in projector_grads.values()) ** 0.5
                print(f"Projector 梯度范数 (hook): {total_grad_norm:.6f}")
                if total_grad_norm == 0:
                    print("  ⚠️ 警告: Projector 梯度为 0!")
                elif total_grad_norm < 1e-8:
                    print("  ⚠️ 警告: Projector 梯度非常小!")
                else:
                    print("  ✅ Projector 梯度正常!")
            else:
                # 回退到直接检查
                total_grad_norm = 0
                has_grad = False
                try:
                    for name, param in unwrapped_model.projector.named_parameters():
                        if hasattr(param, 'grad') and param.grad is not None:
                            has_grad = True
                            total_grad_norm += param.grad.norm().item() ** 2
                    if has_grad:
                        total_grad_norm = total_grad_norm ** 0.5
                        print(f"Projector 梯度范数 (本地): {total_grad_norm:.6f}")
                    else:
                        print(f"Projector 梯度: 不可用 (钩子未触发，可能是 DeepSpeed ZeRO-3 处理)")
                except Exception as e:
                    print(f"Projector 梯度范数: 无法获取 ({e})")
        
        # [P1-B] 补上 residual_gate_logit 和 memory_scale 的梯度打印
        if 'residual_gate_logit' in self._last_grad_norms:
            grad_norm = self._last_grad_norms['residual_gate_logit']
            gate_val = torch.sigmoid(unwrapped_model.residual_gate_logit.float()).mean().item() if hasattr(unwrapped_model, 'residual_gate_logit') else '?'
            print(f"residual_gate_logit 梯度范数: {grad_norm:.8f} (当前 gate={gate_val:.6f})")
            if grad_norm == 0:
                print("  ⚠️ 警告: gate 梯度为 0!")
        
        if 'memory_scale' in self._last_grad_norms:
            grad_norm = self._last_grad_norms['memory_scale']
            scale_val = unwrapped_model.memory_scale.item() if hasattr(unwrapped_model, 'memory_scale') else '?'
            print(f"memory_scale 梯度范数: {grad_norm:.8f} (当前 scale={scale_val:.6f})")
            if grad_norm == 0:
                print("  ⚠️ 警告: scale 梯度为 0!")
        
        print(f"{'='*60}\n")
    
    def _monitor_anchor_drift(self, model):
        """
        监控 Memory Anchor 漂移和多样性（DeepSpeed ZeRO-3 安全版本）
        
        注意：在 ZeRO-3 下，参数是分片的，我们只能监控本地分片的变化。
        这仍然可以反映参数是否在更新，虽然数值可能与完整参数不同。
        """
        # 获取实际模型（处理 DDP/DeepSpeed 包装）
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # 检查是否有可解释性探针
        if not hasattr(unwrapped_model, 'explainability_probe') or unwrapped_model.explainability_probe is None:
            if self.accelerator.is_main_process:
                print(f"[Drift Monitor] 跳过：没有 explainability_probe")
            return
        
        # 检查是否有 initial_mem_anchors
        if not hasattr(unwrapped_model, 'initial_mem_anchors') or unwrapped_model.initial_mem_anchors is None:
            if self.accelerator.is_main_process:
                print(f"[Drift Monitor] 跳过：没有 initial_mem_anchors")
            return
        
        try:
            # 获取完整参数数据（兼容 ZeRO-3 分片）
            # mem_anchors 只有 64×4096（~0.5MB），gather 成本可忽略
            anchors_param = unwrapped_model.mem_anchors
            if hasattr(anchors_param, 'ds_id'):  # ZeRO-3 下参数被分片
                from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                with GatheredParameters([anchors_param], modifier_rank=0):
                    current_anchors = anchors_param.data.detach().clone().float()
            else:
                current_anchors = anchors_param.data.detach().clone().float()
            # 与 current_anchors 对齐 dtype/device，避免 probe 内 matmul dtype 冲突
            initial_anchors = unwrapped_model.initial_mem_anchors.detach().clone().to(
                device=current_anchors.device,
                dtype=current_anchors.dtype
            )
            
            # 计算范数变化（即使是分片数据也能反映变化趋势）
            current_norm = current_anchors.norm().item()
            initial_norm = initial_anchors.norm().item()
            diff_norm = (current_anchors - initial_anchors).norm().item()
            
            # 尝试使用 explainability_probe 检测漂移
            probe = unwrapped_model.explainability_probe
            try:
                metrics = probe.detect_drift(
                    current_anchors=current_anchors,
                    initial_anchors=initial_anchors
                )
            except Exception as e:
                # 如果 probe 失败，使用简化的指标
                metrics = {
                    'anchor_to_concepts_mean': 0,
                    'anchor_to_concepts_max_mean': 0,
                    'diversity_offdiag_abs_mean': 0,
                    'diversity_offdiag_max': 0,
                    'overall_drift': diff_norm / (initial_norm + 1e-8),
                }
                if self.accelerator.is_main_process:
                    print(f"[Drift Monitor] probe.detect_drift 失败: {e}")
            
            # 记录到历史
            self.drift_history.append({
                'step': self.state.global_step,
                **metrics
            })
            
            # 只在主进程打印报告
            if self.accelerator.is_main_process:
                print(f"\n{'='*60}")
                print(f"📊 Step {self.state.global_step} - Memory Anchor 漂移监控")
                print(f"{'='*60}")
                print(f"Anchor ↔ 概念池 平均相似度: {metrics.get('anchor_to_concepts_mean', 0):.4f}")
                print(f"Anchor ↔ 最相似概念 平均值: {metrics.get('anchor_to_concepts_max_mean', 0):.4f}")
                abs_mean = metrics.get('diversity_offdiag_abs_mean', 0)
                print(f"Anchor 间互相似度 |mean|: {abs_mean:.4f} {'✅' if abs_mean < 0.3 else '⚠️ 偏高'}")
                print(f"Anchor 间互相似度 |max|:  {metrics.get('diversity_offdiag_max', 0):.4f}")
                if 'overall_drift' in metrics:
                    print(f"整体漂移: {metrics['overall_drift']:.4f}")
                print(f"--- 调试信息 ---")
                print(f"当前 mem_anchors 范数: {current_norm:.4f}")
                print(f"初始 mem_anchors 范数: {initial_norm:.4f}")
                print(f"参数变化范数: {diff_norm:.4f}")
                if diff_norm > 0:
                    print(f"✅ 参数已更新！")
                elif self.state.global_step == 0:
                    print(f"ℹ️ Step 0 参数尚未更新（正常，梯度将在下一步应用）")
                else:
                    print(f"⚠️ 参数未变化，请检查训练配置")
                print(f"{'='*60}\n")
            
            # 记录到 TensorBoard/WandB（仅主进程）
            if self.state.global_step > 0:
                self.log({
                    'drift/anchor_to_concepts_mean': metrics.get('anchor_to_concepts_mean', 0),
                    'drift/anchor_to_concepts_max_mean': metrics.get('anchor_to_concepts_max_mean', 0),
                    'drift/diversity_offdiag_abs_mean': metrics.get('diversity_offdiag_abs_mean', 0),
                    'drift/diversity_offdiag_max': metrics.get('diversity_offdiag_max', 0),
                    'drift/overall_drift': metrics.get('overall_drift', 0),
                    'drift/param_diff_norm': diff_norm,
                    'drift/current_norm': current_norm,
                    'drift/initial_norm': initial_norm,
                })
                
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"[Drift Monitor] 监控失败: {e}")
                import traceback
                traceback.print_exc()
    
    def on_save(self, args, state, control, **kwargs):
        """保存时额外保存漂移历史（只在主进程执行，避免多进程写文件冲突）"""
        super().on_save(args, state, control, **kwargs)
        
        # 只在主进程保存，避免 DDP/DeepSpeed 下多进程竞争写同一文件
        if not self.is_world_process_zero():
            return
        
        if self.save_drift_report and len(self.drift_history) > 0:
            import json
            drift_path = os.path.join(args.output_dir, f"drift_history_step_{state.global_step}.json")
            with open(drift_path, 'w') as f:
                json.dump(self.drift_history, f, indent=2)
            print(f"[Trainer] 漂移历史已保存到 {drift_path}")
    
    def _monitor_memory_separability(self, model):
        """
        监控 Memory 输出的可分性（batch 内不同样本的 memory 是否可区分）
        
        核心指标：
        - inter_sample_cosine_mean: 不同图片的 memory 之间的平均余弦相似度（越低越好）
        - retrieval_at_1: 自检索准确率（每个样本与自身最相似，理想值=1.0）
        - effective_rank: 有效秩（信息利用程度）
        
        为什么需要这个监控：
        - Anchor 参数多样 ≠ memory 输出多样
        - mean≈0/std≈1 可能只是 RMSNorm 的正常行为，不代表坍塌
        - 真正的坍塌指标是"不同输入的 memory 是否可区分"
        """
        if not self.accelerator.is_main_process:
            return
        
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # 检查是否有 separability probe 和缓存的 latent_memory
        if not hasattr(unwrapped_model, 'separability_probe') or unwrapped_model.separability_probe is None:
            return
        if not hasattr(unwrapped_model, '_last_latent_memory') or unwrapped_model._last_latent_memory is None:
            return
        
        try:
            probe = unwrapped_model.separability_probe
            stage_tensors = {
                "latent": unwrapped_model._last_latent_memory.detach()
            }
            if hasattr(unwrapped_model, '_last_raw_memory') and unwrapped_model._last_raw_memory is not None:
                stage_tensors["raw"] = unwrapped_model._last_raw_memory.detach()
            if hasattr(unwrapped_model, '_last_projected_memory') and unwrapped_model._last_projected_memory is not None:
                stage_tensors["projected"] = unwrapped_model._last_projected_memory.detach()

            stage_metrics = {}
            for stage_name, stage_tensor in stage_tensors.items():
                # 重要：将 bf16 转为 float32 再计算 separability 指标
                # bf16 下 SVD/cosine 数值不稳定，会让相似度更贴近 1、有效秩偏低
                stage_tensor = stage_tensor.float()
                if stage_tensor.shape[0] < 2:
                    continue
                stage_metrics[stage_name] = probe.compute_separability(stage_tensor)

            # 如果没有可用指标，直接跳过
            if "latent" not in stage_metrics:
                return

            sep_metrics = stage_metrics["latent"]
            
            # 打印核心报告（与历史输出保持一致）
            probe.print_separability_report(sep_metrics, step=self.state.global_step)

            # 分层诊断：定位坍塌发生在 raw / projected / latent 的哪一层
            if self.accelerator.is_main_process and len(stage_metrics) > 1:
                print(f"[MemorySep-Layerwise] raw/projected/latent 可分性对比:")
                for name in ("raw", "projected", "latent"):
                    if name not in stage_metrics:
                        continue
                    m = stage_metrics[name]
                    print(
                        f"  - {name:<9} cos_mean={m.get('inter_sample_cosine_mean', 0):.4f}, "
                        f"slot_var={m.get('slot_variance_mean', 0):.6f}, "
                        f"rank={m.get('effective_rank', 0):.2f}"
                    )
            
            # 记录到 TensorBoard/WandB
            if self.state.global_step > 0:
                log_metrics = {
                    'memory_sep/inter_sample_cosine_mean': sep_metrics.get('inter_sample_cosine_mean', 0),
                    'memory_sep/inter_sample_cosine_max': sep_metrics.get('inter_sample_cosine_max', 0),
                    'memory_sep/retrieval_at_1': sep_metrics.get('retrieval_at_1', 0),
                    'memory_sep/slot_variance_mean': sep_metrics.get('slot_variance_mean', 0),
                    'memory_sep/effective_rank': sep_metrics.get('effective_rank', 0),
                }
                for name in ("raw", "projected"):
                    if name in stage_metrics:
                        m = stage_metrics[name]
                        log_metrics[f'memory_sep/{name}_inter_sample_cosine_mean'] = m.get('inter_sample_cosine_mean', 0)
                        log_metrics[f'memory_sep/{name}_slot_variance_mean'] = m.get('slot_variance_mean', 0)
                        log_metrics[f'memory_sep/{name}_effective_rank'] = m.get('effective_rank', 0)
                self.log(log_metrics)
        
        except Exception as e:
            print(f"[Separability Monitor] 监控失败: {e}")
    
    
    def __del__(self):
        """清理梯度钩子"""
        for handle in self._gradient_hooks:
            handle.remove()
