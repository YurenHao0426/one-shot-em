import argparse
import os
import random
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import wandb

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 导入自定义模块
import sys
sys.path.append('.')
from dataset.gee_processor import GEEProcessor
from losses.gee_loss import GEELoss, gender_to_label

os.environ.setdefault("NCCL_TIMEOUT", "2700")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "2700")

class GEEDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch):
    return {
        "input": [item["input"] for item in batch],
        "gender": [item["gender"] for item in batch]
    }

def parse_args():
    parser = argparse.ArgumentParser()
    # GEE相关参数
    parser.add_argument('--lambda_weight', type=float, default=3.0, help='GEE lambda weight')
    parser.add_argument('--use_l1', action='store_true', help='Use L1 loss instead of L2')
    parser.add_argument('--auto_anneal', action='store_true', help='Use automatic annealing')
    parser.add_argument('--bias_eval_steps', type=int, default=10, help='Bias evaluation frequency')
    parser.add_argument('--balance_dataset', action='store_true', default=True, help='Balance dataset by gender')
    parser.add_argument('--target_size', type=int, default=None, help='Target dataset size for balancing')
    
    # 保留原有参数
    parser.add_argument('--model_name', type=str, default='Qwen2.5-Math-7B', help='Model name')
    parser.add_argument('--model_path', type=str, default=None, help='Local model path')
    parser.add_argument('--train_data', type=str, default='dataset/1shot_rlvr/pi1_r1280.parquet', help='Training data file path')
    parser.add_argument('--save_root', type=str, default=None, help='Checkpoint save root directory')
    parser.add_argument('--effective_batch', type=int, default=64, help='Global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=2, help='Micro batch size')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature coefficient')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--log_steps', type=int, default=1, help='Logging step interval')
    parser.add_argument('--save_steps', type=int, default=1, help='Checkpoint saving step interval')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum training steps')
    parser.add_argument('--sample_temp', type=float, default=0.5, help='Generation temperature parameter')
    parser.add_argument('--run_name', type=str, default='one_shot_gee', help='Experiment run name')
    parser.add_argument('--wandb_project', type=str, default='one-shot-gee', help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    parser.add_argument('--use_test_data', action='store_true', help='Use synthetic test data instead of real data')
    return parser.parse_args()

def apply_chat_template(tokenizer, problem: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False, add_generation_prompt=True
    )

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    micro_bs = args.micro_batch_size
    eff_bs = args.effective_batch
    accum_steps = max(1, eff_bs // (micro_bs * world_size))
    temp = args.temperature
    lr = args.learning_rate

    save_root = args.save_root or (f"checkpoints/{args.model_name}/{args.run_name}" if args.run_name else f"checkpoints/{args.model_name}")
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_bs,
        "train_batch_size": eff_bs,
        "gradient_accumulation_steps": accum_steps,
        "bf16": {"enabled": True},
        "zero_optimization": {
                              "stage": 2, 
                              "offload_optimizer": {"device": "cpu"}, 
                              "offload_param": {"device": "none"}
                             },
        "gradient_clipping": 1.0,
    }
    accelerator = Accelerator(mixed_precision="bf16", 
                              gradient_accumulation_steps=accum_steps, 
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config))
    print = accelerator.print

    model_path = args.model_path or f"/volume/pt-train/models/{args.model_name}"
    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    # 初始化GEE处理器和损失函数
    gee_processor = GEEProcessor(tokenizer)
    gee_loss_fn = GEELoss(lambda_weight=args.lambda_weight, use_l1=args.use_l1)

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name or args.wandb_name or args.model_name, config=vars(args))

    # 准备数据
    if args.use_test_data:
        print("使用合成测试数据...")
        train_data = gee_processor.create_test_data(num_samples=200)
    else:
        print("使用真实数据...")
        train_data = gee_processor.prepare_gee_data(
            args.train_data, 
            balance=args.balance_dataset, 
            target_size=args.target_size
        )
    
    train_loader = DataLoader(
        GEEDataset(train_data), 
        batch_size=micro_bs, 
        shuffle=True, 
        collate_fn=custom_collate
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    initial_entropy_gap = None
    model.train()
    
    for step, batch in enumerate(train_loader, start=1):
        if step > args.max_steps:
            print(f"Exceed max step {args.max_steps}, training stopped.")
            break
        
        with accelerator.accumulate(model):
            # 准备输入
            inputs = tokenizer(
                batch["input"], 
                return_tensors="pt", 
                padding="longest", 
                truncation=True, 
                max_length=2048
            ).to(accelerator.device)
            
            # 生成回答
            with torch.no_grad():
                gen_ids = accelerator.unwrap_model(model).generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=0.95,
                    temperature=args.sample_temp,
                    synced_gpus=True,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False
                )
            
            # 准备完整序列
            seq = torch.cat([inputs.input_ids, gen_ids[:, inputs.input_ids.shape[1]:]], dim=1)[:, :4096]
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
                print(f"Step {step} | loss={loss.item():.6f} | "
                      f"entropy_gap={metrics['entropy_gap']:.6f} | "
                      f"H_male={metrics['H_male']:.6f} | "
                      f"H_female={metrics['H_female']:.6f}")
                wandb.log({"step": step, **metrics})
                
            if step % args.save_steps == 0:
                ckpt = Path(save_root) / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt, safe_serialization=True)
                tokenizer.save_pretrained(ckpt)
                print(f"Checkpoint saved to {ckpt}")

    if accelerator.is_main_process:
        final = Path(save_root) / "final"
        final.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        print(f"Final checkpoint saved to {final}")
        wandb.finish()

if __name__ == "__main__":
    main() 