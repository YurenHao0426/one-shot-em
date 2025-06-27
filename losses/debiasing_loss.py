import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class DebiasingLoss:
    """
    纯偏见减少损失函数
    目标：最小化男女间的熵差，不包含整体熵最小化
    """
    def __init__(self, use_l1: bool = False, scale_factor: float = 1.0):
        self.use_l1 = use_l1
        self.scale_factor = scale_factor  # 可选的缩放因子
    
    def compute_token_entropy(self, logits: torch.Tensor, 
                            attention_mask: torch.Tensor = None) -> torch.Tensor:
        """计算token级别的条件熵"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        H_tok = -(probs * log_probs).sum(-1)  # (B, T)
        
        if attention_mask is not None:
            H_tok = H_tok * attention_mask
        
        return H_tok
    
    def compute_sample_entropy(self, H_tok: torch.Tensor, 
                             prompt_lengths: torch.Tensor) -> torch.Tensor:
        """计算样本平均熵"""
        batch_size = H_tok.size(0)
        H_i = torch.zeros(batch_size, device=H_tok.device)
        
        for i in range(batch_size):
            # 只计算生成部分的熵（排除prompt部分）
            gen_start = prompt_lengths[i]
            if gen_start < H_tok.size(1):
                gen_entropy = H_tok[i, gen_start:]
                
                if gen_entropy.numel() > 0:
                    H_i[i] = gen_entropy.mean()
                else:
                    H_i[i] = 0.0
        
        return H_i
    
    def compute_group_entropy(self, H_i: torch.Tensor, 
                            gender_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算各组平均熵"""
        male_mask = (gender_labels == 0)  # 假设0=male, 1=female
        female_mask = (gender_labels == 1)
        
        male_count = male_mask.sum().item()
        female_count = female_mask.sum().item()
        
        if male_count == 0:
            print(f"⚠️ 警告: 批次中没有男性样本")
            H_male = torch.tensor(0.0, device=H_i.device)
        else:
            H_male = H_i[male_mask].mean()
            
        if female_count == 0:
            print(f"⚠️ 警告: 批次中没有女性样本")
            H_female = torch.tensor(0.0, device=H_i.device)
        else:
            H_female = H_i[female_mask].mean()
        
        return H_male, H_female
    
    def compute_debiasing_loss(self, H_i: torch.Tensor, 
                             gender_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算纯偏见减少损失
        目标：最小化 |H_female - H_male|
        """
        # 计算各组平均熵
        H_male, H_female = self.compute_group_entropy(H_i, gender_labels)
        
        # 计算熵差距（这是我们要最小化的目标）
        entropy_gap = H_female - H_male
        
        if self.use_l1:
            # L1损失：|H_female - H_male|
            debiasing_loss = torch.abs(entropy_gap) * self.scale_factor
        else:
            # L2损失：(H_female - H_male)²
            debiasing_loss = (entropy_gap ** 2) * self.scale_factor
        
        # 计算监控指标
        H_bar = H_i.mean()  # 仅用于监控，不参与损失计算
        
        metrics = {
            'loss_debiasing': debiasing_loss.item(),
            'entropy_gap': abs(entropy_gap.item()),
            'entropy_gap_signed': entropy_gap.item(),  # 带符号的差距
            'H_bar': H_bar.item(),  # 整体平均熵（仅监控）
            'H_male': H_male.item(),
            'H_female': H_female.item(),
            'scale_factor': self.scale_factor
        }
        
        return debiasing_loss, metrics
    
    def update_scale_factor(self, new_scale: float):
        """更新缩放因子（用于调整损失大小）"""
        self.scale_factor = new_scale

def gender_to_label(gender_str: str) -> int:
    """将性别字符串转换为标签"""
    return 0 if gender_str == 'male' else 1

def label_to_gender(label: int) -> str:
    """将标签转换为性别字符串"""
    return 'male' if label == 0 else 'female' 