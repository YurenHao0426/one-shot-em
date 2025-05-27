import argparse
import random
import string
from pathlib import Path
import torch.nn.functional as F
import os, math, torch, pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AdamW


parser = argparse.ArgumentParser(description="Train model with configurable hyperparameters")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--effective_batch", type=int, default=64, help="Effective batch size")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

temp_str = str(args.temperature).replace(".", "")
lr_str = f"{args.lr:.0e}"
bsz_str = str(args.effective_batch)
save_root = f"/volume/ailab4sci/ztgao/em/checkpoints_32b/t{temp_str}_lr{lr_str}_bsz{bsz_str}_seed{args.seed}"

temperature = args.temperature
learning_rate = args.lr
effective_batch = args.effective_batch
micro_batch_size = 2
world_size = int(os.environ.get("WORLD_SIZE", 1))
accum_steps = max(1, effective_batch // (micro_batch_size * world_size))

DEEPSPEED_CONFIG = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "train_batch_size": effective_batch,
    "gradient_accumulation_steps": accum_steps,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "offload_param":     {"device": "none"}
    },
    "gradient_clipping": 1.0,
}

ds_plugin = DeepSpeedPlugin(hf_ds_config=DEEPSPEED_CONFIG)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=accum_steps,
    deepspeed_plugin=ds_plugin,
)
print = accelerator.print

model_name = "Qwen2.5-32B-Instruct"
model_path = f"/volume/ailab4sci/models/{model_name}"
config = AutoConfig.from_pretrained(model_path)
config.use_cache = False
model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class FTDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def custom_collate(batch):
    return {"input": [item["input"] for item in batch]}

def apply_chat_template(problem: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False, add_generation_prompt=True
    )

df = pd.read_parquet("/volume/ailab4sci/ztgao/em/dataset/1shot_rlvr/pi1_r1280.parquet")
data = [{"input": apply_chat_template(p)} for p in df["problem"].dropna().tolist()]
dataset = FTDataset(data)
data_loader = DataLoader(
    dataset,
    batch_size=micro_batch_size,
    shuffle=True,
    collate_fn=custom_collate,
)

optimizer = AdamW(model.parameters(), lr=learning_rate)
model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

model.train()
for step, batch in enumerate(data_loader, start=1):
    if step > 30:
        break
    
    with accelerator.accumulate(model):
        enc = tokenizer(
            batch["input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2048,
        ).to(accelerator.device)

        with torch.no_grad():
            gen_ids = accelerator.unwrap_model(model).generate(
                **enc,
                max_new_tokens=2048,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                synced_gpus=True,
                repetition_penalty=1.15,
                pad_token_id=151643
            )

        seq = torch.cat(
            [enc.input_ids, gen_ids[:, enc.input_ids.shape[1]:]],
            dim=1
        )[:, :4096]

        pad_mask = seq.ne(tokenizer.pad_token_id)
        prompt_len = pad_mask[:, :enc.input_ids.shape[1]].sum(-1)
        token_idx = torch.arange(seq.size(1), device=seq.device)
        gen_mask = (token_idx.unsqueeze(0) >= prompt_len.unsqueeze(1)) & pad_mask

        logits = model(seq, attention_mask=pad_mask).logits
        probs = F.softmax(logits / temperature, dim=-1)
        H_tok = -(probs * torch.log(probs + 1e-12)).sum(-1)
        loss = (H_tok * gen_mask).sum() / gen_mask.sum().clamp_min(1)

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        print(f"Step {step} | loss={loss.item():.8f}")

    if accelerator.is_main_process:
        ckpt_dir = Path(save_root) / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(ckpt_dir, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_dir)
        accelerator.wait_for_everyone()
        print(f"Checkpoint saved to {ckpt_dir}")

if accelerator.is_main_process:
    final_dir = Path(save_root) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"Final checkpoint saved to {final_dir}")