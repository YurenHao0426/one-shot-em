#!/usr/bin/env python3
"""
纯偏见减少训练脚本
目标：只最小化男女间熵差，不进行整体熵最小化
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import numpy as np
from pathlib import Path

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 导入自定义模块
import sys
sys.path.append('.')
from dataset.gee_processor import GEEProcessor
from losses.debiasing_loss import DebiasingLoss, gender_to_label
from smart_balanced_dataloader import create_smart_balanced_dataloader

os.environ.setdefault("NCCL_TIMEOUT", "2700")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "2700")

def parse_args():
    parser = argparse.ArgumentParser()
    # Debiasing相关参数
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Debiasing loss scale factor')
    parser.add_argument('--use_l1', action='store_true', help='Use L1 loss instead of L2')
    parser.add_argument('--target_gap', type=float, default=0.01, help='Target entropy gap for early stopping')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='Qwen2.5-Math-1.5B-Instruct', help='Model name')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--effective_batch', type=int, default=4, help='Global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=2, help='Micro batch size (must >=2 for balance)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate (lower for debiasing)')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum training steps')
    parser.add_argument('--sample_temp', type=float, default=0.7, help='Generation temperature')
    
    # 运行参数
    parser.add_argument('--run_name', type=str, default='pure_debiasing', help='Run name')
    parser.add_argument('--wandb_project', type=str, default='debiasing-only', help='W&B project name')
    parser.add_argument('--use_test_data', action='store_true', help='Use synthetic test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_steps', type=int, default=1, help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=10, help='Save frequency')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 强制检查batch_size
    if args.micro_batch_size < 2:
        print("❌ 错误: micro_batch_size必须>=2才能保证性别平衡！")
        print("请使用: --micro_batch_size 2 或更大")
        return
    
    # DeepSpeed配置
    ds_config = {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "train_batch_size": args.effective_batch,
        "gradient_accumulation_steps": max(1, args.effective_batch // args.micro_batch_size),
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2, 
            "offload_optimizer": {"device": "cpu"}, 
            "offload_param": {"device": "none"}
        },
        "gradient_clipping": 1.0,
    }
    
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=max(1, args.effective_batch // args.micro_batch_size),
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config)
    )
    
    print = accelerator.print
    print(f"🎯 开始纯偏见减少训练 - {args.run_name}")
    print(f"📊 配置信息:")
    print(f"   批次大小: micro={args.micro_batch_size}, effective={args.effective_batch}")
    print(f"   缩放因子: {args.scale_factor}")
    print(f"   目标熵差: {args.target_gap}")
    print(f"   学习率: {args.learning_rate}")
    print(f"   ⚠️  注意: 只进行偏见减少，不进行熵最小化")

    # 加载模型
    model_path = args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化纯偏见减少损失函数
    gee_processor = GEEProcessor(tokenizer)
    debiasing_loss_fn = DebiasingLoss(use_l1=args.use_l1, scale_factor=args.scale_factor)

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # 准备数据
    if args.use_test_data:
        print("📊 使用合成测试数据...")
        train_data = gee_processor.create_test_data(num_samples=100)
        
        # 检查数据平衡性
        male_count = sum(1 for item in train_data if item['gender'] == 'male')
        female_count = sum(1 for item in train_data if item['gender'] == 'female')
        print(f"原始数据: male={male_count}, female={female_count}")
        
        # 创建智能平衡的数据加载器
        train_loader = create_smart_balanced_dataloader(
            train_data, 
            batch_size=args.micro_batch_size, 
            num_batches=args.max_steps + 5
        )
    else:
        print("❌ 请使用 --use_test_data 进行测试")
        return

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    print(f"🎯 开始训练...")
    print(f"   目标: 最小化 |H_female - H_male|")
    print(f"   理想结果: entropy_gap → {args.target_gap}")
    
    # 开始训练
    model.train()
    best_gap = float('inf')
    converged_steps = 0
    
    for step, batch in enumerate(train_loader, start=1):
        if step > args.max_steps:
            print(f"🛑 达到最大步数 {args.max_steps}，训练结束")
            break
        
        with accelerator.accumulate(model):
            try:
                # 验证批次平衡性
                male_count = sum(1 for g in batch["gender"] if g == 'male')
                female_count = sum(1 for g in batch["gender"] if g == 'female')
                
                if male_count == 0 or female_count == 0:
                    print(f"💥 Step {step}: 批次不平衡！male={male_count}, female={female_count}")
                    continue
                
                # 准备输入
                inputs = tokenizer(
                    batch["input"], 
                    return_tensors="pt", 
                    padding="longest", 
                    truncation=True, 
                    max_length=1024
                ).to(accelerator.device)
                
                # 生成回答
                with torch.no_grad():
                    gen_ids = accelerator.unwrap_model(model).generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        top_p=0.95,
                        temperature=args.sample_temp,
                        synced_gpus=True,
                        repetition_penalty=1.15,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=False
                    )
                
                # 准备完整序列
                seq = torch.cat([inputs.input_ids, gen_ids[:, inputs.input_ids.shape[1]:]], dim=1)
                pad_mask = seq.ne(tokenizer.pad_token_id)
                prompt_lengths = pad_mask[:, :inputs.input_ids.shape[1]].sum(-1)
                
                # 计算logits和熵
                logits = model(seq, attention_mask=pad_mask).logits
                H_tok = debiasing_loss_fn.compute_token_entropy(logits, pad_mask)
                H_i = debiasing_loss_fn.compute_sample_entropy(H_tok, prompt_lengths)
                
                # 准备性别标签
                gender_labels = torch.tensor([
                    gender_to_label(g) for g in batch["gender"]
                ], device=accelerator.device)
                
                # 计算纯偏见减少损失
                loss, metrics = debiasing_loss_fn.compute_debiasing_loss(H_i, gender_labels)
                
                # 反向传播
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # 检查收敛
                current_gap = metrics['entropy_gap']
                if current_gap < best_gap:
                    best_gap = current_gap
                    converged_steps = 0
                else:
                    converged_steps += 1
                
                # 早停检查
                if current_gap <= args.target_gap:
                    print(f"🎉 达到目标熵差 {args.target_gap}！当前: {current_gap:.6f}")
                    print(f"在第 {step} 步实现目标")
                    break
                
                # 日志记录
                if accelerator.is_main_process:
                    if step % args.log_steps == 0:
                        gap_direction = "📉" if current_gap <= best_gap else "📈"
                        print(f"{gap_direction} Step {step} | loss={loss.item():.6f} | "
                              f"gap={current_gap:.6f} | "
                              f"H_male={metrics['H_male']:.6f} | "
                              f"H_female={metrics['H_female']:.6f} | "
                              f"批次[{male_count}M,{female_count}F]")
                        
                        # 添加进度指标
                        progress = max(0, min(1, (0.1 - current_gap) / 0.1)) * 100  # 假设从0.1开始
                        metrics['progress_percent'] = progress
                        metrics['best_gap'] = best_gap
                        
                        wandb.log({"step": step, **metrics})
                
                # 保存检查点
                if accelerator.is_main_process and step % args.save_steps == 0:
                    ckpt = Path(f"checkpoints/{args.model_name}/{args.run_name}") / f"step_{step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(ckpt, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt)
                    print(f"💾 检查点已保存: {ckpt}")
                
            except Exception as e:
                print(f"❌ 训练步骤 {step} 出错: {e}")
                continue

    if accelerator.is_main_process:
        print(f"\n🎉 纯偏见减少训练完成!")
        print(f"📊 最终结果:")
        print(f"   最佳熵差: {best_gap:.6f}")
        print(f"   目标熵差: {args.target_gap}")
        
        if best_gap <= args.target_gap:
            print("✅ 成功达到偏见减少目标！")
        elif best_gap <= 0.05:
            print("⚠️ 偏见显著减少，接近目标")
        else:
            print("❌ 偏见减少效果有限，可能需要更多训练步数或调整参数")
        
        # 保存最终模型
        final = Path(f"checkpoints/{args.model_name}/{args.run_name}") / "final"
        final.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        print(f"💾 最终模型已保存: {final}")
        
        wandb.finish()

if __name__ == "__main__":
    main() 