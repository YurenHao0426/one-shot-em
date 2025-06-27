#!/usr/bin/env python3
"""
GEE训练脚本 - 使用平衡数据加载器
确保每个批次都包含男女样本
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import numpy as np

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 导入自定义模块
import sys
sys.path.append('.')
from dataset.gee_processor import GEEProcessor
from losses.gee_loss import GEELoss, gender_to_label
from balanced_dataloader import create_balanced_dataloader, balanced_collate

os.environ.setdefault("NCCL_TIMEOUT", "2700")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "2700")

def parse_args():
    parser = argparse.ArgumentParser()
    # GEE相关参数
    parser.add_argument('--lambda_weight', type=float, default=1.0, help='GEE lambda weight')
    parser.add_argument('--use_l1', action='store_true', help='Use L1 loss instead of L2')
    parser.add_argument('--auto_anneal', action='store_true', help='Use automatic annealing')
    
    # 简化参数
    parser.add_argument('--model_name', type=str, default='Qwen2.5-Math-1.5B-Instruct', help='Model name')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--effective_batch', type=int, default=4, help='Global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=2, help='Micro batch size (must >=2 for balance)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=5, help='Maximum training steps')
    parser.add_argument('--sample_temp', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--run_name', type=str, default='gee_balanced', help='Run name')
    parser.add_argument('--wandb_project', type=str, default='one-shot-gee', help='W&B project name')
    parser.add_argument('--use_test_data', action='store_true', help='Use synthetic test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_steps', type=int, default=1, help='Logging frequency')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 检查batch_size
    if args.micro_batch_size < 2:
        print("⚠️ 警告: micro_batch_size < 2，无法保证性别平衡！")
        print("建议设置 --micro_batch_size 至少为 2")
    
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
    print(f"🚀 开始平衡版GEE训练 - {args.run_name}")
    print(f"📊 批次配置: micro_batch={args.micro_batch_size}, effective_batch={args.effective_batch}")

    # 加载模型
    model_path = args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化GEE处理器和损失函数
    gee_processor = GEEProcessor(tokenizer)
    gee_loss_fn = GEELoss(lambda_weight=args.lambda_weight, use_l1=args.use_l1)

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # 准备数据 - 使用平衡数据加载器
    if args.use_test_data:
        print("📊 使用合成测试数据...")
        train_data = gee_processor.create_test_data(num_samples=50)
        
        # 检查数据平衡性
        male_count = sum(1 for item in train_data if item['gender'] == 'male')
        female_count = sum(1 for item in train_data if item['gender'] == 'female')
        print(f"原始数据: male={male_count}, female={female_count}")
        
        # 创建平衡的数据加载器
        train_loader = create_balanced_dataloader(
            train_data, 
            batch_size=args.micro_batch_size, 
            num_batches=args.max_steps * 2  # 确保有足够的批次
        )
    else:
        print("❌ 请使用 --use_test_data 进行测试")
        return

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    print(f"🎯 训练配置:")
    print(f"   Lambda权重: {args.lambda_weight}")
    print(f"   最大步数: {args.max_steps}")
    print(f"   学习率: {args.learning_rate}")
    
    # 开始训练
    model.train()
    initial_entropy_gap = None
    
    for step, batch in enumerate(train_loader, start=1):
        if step > args.max_steps:
            print(f"🛑 达到最大步数 {args.max_steps}，训练结束")
            break
        
        with accelerator.accumulate(model):
            try:
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
                H_tok = gee_loss_fn.compute_token_entropy(logits, pad_mask)
                H_i = gee_loss_fn.compute_sample_entropy(H_tok, prompt_lengths)
                
                # 准备性别标签
                gender_labels = torch.tensor([
                    gender_to_label(g) for g in batch["gender"]
                ], device=accelerator.device)
                
                # 计算GEE损失
                loss, metrics = gee_loss_fn.compute_gee_loss(H_i, gender_labels)
                
                # 自动退火（可选）
                if args.auto_anneal and initial_entropy_gap is None:
                    initial_entropy_gap = metrics['entropy_gap']
                
                if args.auto_anneal and initial_entropy_gap > 0:
                    current_gap = metrics['entropy_gap']
                    anneal_factor = current_gap / initial_entropy_gap
                    new_lambda = args.lambda_weight * anneal_factor
                    gee_loss_fn.update_lambda(new_lambda)
                    metrics['lambda_weight'] = new_lambda
                
                # 反向传播
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # 日志记录
                if accelerator.is_main_process:
                    if step % args.log_steps == 0:
                        print(f"📈 Step {step} | loss={loss.item():.6f} | "
                              f"entropy_gap={metrics['entropy_gap']:.6f} | "
                              f"H_male={metrics['H_male']:.6f} | "
                              f"H_female={metrics['H_female']:.6f} | "
                              f"批次性别={batch['gender']}")
                        wandb.log({"step": step, **metrics})
                
            except Exception as e:
                print(f"❌ 训练步骤 {step} 出错: {e}")
                continue

    if accelerator.is_main_process:
        print("🎉 平衡版GEE训练完成!")
        wandb.finish()

if __name__ == "__main__":
    main() 