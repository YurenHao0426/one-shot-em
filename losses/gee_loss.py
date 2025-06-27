import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class GEELoss:
    def __init__(self, lambda_weight: float = 3.0, use_l1: bool = False):
        self.lambda_weight = lambda_weight
        self.use_l1 = use_l1
    
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
        """计算样本平均熵 - 修复版本"""
        batch_size = H_tok.size(0)
        H_i = torch.zeros(batch_size, device=H_tok.device)
        
        for i in range(batch_size):
            # 只计算生成部分的熵（排除prompt部分）
            gen_start = prompt_lengths[i]
            if gen_start < H_tok.size(1):
                gen_entropy = H_tok[i, gen_start:]
                
                # 🔧 修复: 不要过滤熵值为0的token！
                # 熵值为0是合理的（模型确定性高时）
                # 只过滤掉真正的padding token（用attention_mask标记）
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
        
        # 🔧 修复: 添加调试信息
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
    
    def compute_gee_loss(self, H_i: torch.Tensor, 
                        gender_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """计算GEE损失"""
        H_bar = H_i.mean()  # 全批平均熵
        
        # 计算各组平均熵
        H_male, H_female = self.compute_group_entropy(H_i, gender_labels)
        
        # 🔧 修复: 改进组间差异计算
        if self.use_l1:
            # L1版本
            loss_bias = torch.abs(H_female - H_male)
        else:
            # L2版本 - 简化计算
            loss_bias = (H_female - H_male) ** 2
        
        # 总损失
        loss_em = H_bar
        loss_total = loss_em + self.lambda_weight * loss_bias
        
        # 返回损失和监控指标
        metrics = {
            'loss_em': loss_em.item(),
            'loss_bias': loss_bias.item(),
            'loss_total': loss_total.item(),
            'H_bar': H_bar.item(),
            'H_male': H_male.item(),
            'H_female': H_female.item(),
            'entropy_gap': abs(H_female - H_male).item(),
            'lambda_weight': self.lambda_weight
        }
        
        return loss_total, metrics
    
    def update_lambda(self, new_lambda: float):
        """更新lambda权重（用于自动退火）"""
        self.lambda_weight = new_lambda

def gender_to_label(gender_str: str) -> int:
    """将性别字符串转换为标签"""
    return 0 if gender_str == 'male' else 1

def label_to_gender(label: int) -> str:
    """将标签转换为性别字符串"""
    return 'male' if label == 0 else 'female' 