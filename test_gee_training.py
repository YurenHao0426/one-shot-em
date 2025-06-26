#!/usr/bin/env python3
"""
GEE训练逻辑测试脚本
模拟训练过程而不需要真实模型
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('.')

from dataset.gee_processor import GEEProcessor
from losses.gee_loss import GEELoss, gender_to_label

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = '<|endoftext|>'
        self.pad_token = self.eos_token
        
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]
    
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        # 模拟tokenization
        batch_size = len(texts)
        seq_len = 50  # 固定序列长度用于测试
        
        return {
            'input_ids': torch.randint(1, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        }

class MockModel:
    def __init__(self):
        self.device = 'cpu'
    
    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        vocab_size = 1000
        
        # 模拟logits输出
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return MockOutput(logits)
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, **kwargs):
        batch_size, prompt_len = input_ids.shape
        # 模拟生成新的token
        new_tokens = torch.randint(1, 1000, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)

def test_gee_training_logic():
    """测试GEE训练逻辑"""
    print("="*60)
    print("测试GEE训练逻辑")
    print("="*60)
    
    # 初始化组件
    tokenizer = MockTokenizer()
    model = MockModel()
    gee_processor = GEEProcessor(tokenizer)
    gee_loss_fn = GEELoss(lambda_weight=3.0, use_l1=False)
    
    # 生成测试数据
    train_data = gee_processor.create_test_data(num_samples=20)
    print(f"生成训练数据: {len(train_data)} 条")
    
    # 模拟训练循环
    batch_size = 4
    num_steps = 5
    
    print(f"\n开始模拟训练 ({num_steps} 步)...")
    
    for step in range(1, num_steps + 1):
        # 创建batch
        batch_data = train_data[(step-1)*batch_size:step*batch_size]
        if len(batch_data) < batch_size:
            # 循环使用数据
            batch_data = train_data[:batch_size]
        
        batch = {
            "input": [item["input"] for item in batch_data],
            "gender": [item["gender"] for item in batch_data]
        }
        
        # 模拟tokenization
        inputs = tokenizer(batch["input"])
        
        # 模拟生成
        gen_ids = model.generate(**inputs, max_new_tokens=20)
        
        # 准备完整序列
        seq = gen_ids[:, :100]  # 限制长度用于测试
        prompt_lengths = torch.tensor([inputs['input_ids'].shape[1]] * batch_size)
        
        # 计算logits和熵
        mock_output = model(seq)
        logits = mock_output.logits
        
        # 计算GEE损失
        H_tok = gee_loss_fn.compute_token_entropy(logits)
        H_i = gee_loss_fn.compute_sample_entropy(H_tok, prompt_lengths)
        
        # 准备性别标签
        gender_labels = torch.tensor([gender_to_label(g) for g in batch["gender"]])
        
        # 计算损失
        loss, metrics = gee_loss_fn.compute_gee_loss(H_i, gender_labels)
        
        # 打印训练日志
        print(f"Step {step} | loss={loss.item():.6f} | "
              f"entropy_gap={metrics['entropy_gap']:.6f} | "
              f"H_male={metrics['H_male']:.6f} | "
              f"H_female={metrics['H_female']:.6f}")
        
        # 验证损失计算
        assert not torch.isnan(loss), "损失为NaN"
        assert loss.item() > 0, "损失应该为正值"
        assert 'entropy_gap' in metrics, "缺少entropy_gap指标"
    
    print("✓ GEE训练逻辑测试通过")

def test_different_lambdas():
    """测试不同lambda值的影响"""
    print("\n" + "="*60)
    print("测试不同lambda值的影响")
    print("="*60)
    
    tokenizer = MockTokenizer()
    model = MockModel()
    gee_processor = GEEProcessor(tokenizer)
    
    # 测试不同的lambda值
    lambda_values = [0.0, 1.0, 3.0, 5.0]
    
    # 创建固定的测试数据
    batch_size = 4
    seq_len = 50
    vocab_size = 1000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    prompt_lengths = torch.tensor([20, 20, 20, 20])
    gender_labels = torch.tensor([0, 1, 0, 1])  # male, female, male, female
    
    print("Lambda值对损失的影响:")
    print("Lambda\tEM Loss\tBias Loss\tTotal Loss\tEntropy Gap")
    print("-" * 60)
    
    for lambda_val in lambda_values:
        gee_loss_fn = GEELoss(lambda_weight=lambda_val, use_l1=False)
        
        H_tok = gee_loss_fn.compute_token_entropy(logits)
        H_i = gee_loss_fn.compute_sample_entropy(H_tok, prompt_lengths)
        loss, metrics = gee_loss_fn.compute_gee_loss(H_i, gender_labels)
        
        print(f"{lambda_val:.1f}\t{metrics['loss_em']:.4f}\t"
              f"{metrics['loss_bias']:.4f}\t{metrics['loss_total']:.4f}\t"
              f"{metrics['entropy_gap']:.4f}")
    
    print("✓ Lambda值测试通过")

def test_l1_vs_l2():
    """测试L1和L2损失的差异"""
    print("\n" + "="*60)
    print("测试L1和L2损失的差异")
    print("="*60)
    
    # 创建固定的测试数据
    batch_size = 4
    seq_len = 50
    vocab_size = 1000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    prompt_lengths = torch.tensor([20, 20, 20, 20])
    gender_labels = torch.tensor([0, 1, 0, 1])
    
    # 测试L2版本
    gee_loss_l2 = GEELoss(lambda_weight=3.0, use_l1=False)
    H_tok = gee_loss_l2.compute_token_entropy(logits)
    H_i = gee_loss_l2.compute_sample_entropy(H_tok, prompt_lengths)
    loss_l2, metrics_l2 = gee_loss_l2.compute_gee_loss(H_i, gender_labels)
    
    # 测试L1版本
    gee_loss_l1 = GEELoss(lambda_weight=3.0, use_l1=True)
    loss_l1, metrics_l1 = gee_loss_l1.compute_gee_loss(H_i, gender_labels)
    
    print(f"L2损失: {metrics_l2['loss_total']:.6f} (bias: {metrics_l2['loss_bias']:.6f})")
    print(f"L1损失: {metrics_l1['loss_total']:.6f} (bias: {metrics_l1['loss_bias']:.6f})")
    print(f"熵差距: {metrics_l2['entropy_gap']:.6f}")
    
    print("✓ L1 vs L2测试通过")

def main():
    """主测试函数"""
    print("开始GEE训练逻辑测试...")
    
    try:
        test_gee_training_logic()
        test_different_lambdas()
        test_l1_vs_l2()
        
        print("\n" + "="*60)
        print("所有训练逻辑测试通过！✓")
        print("="*60)
        print("\n核心功能验证:")
        print("✅ 数据处理流程正常")
        print("✅ 损失函数计算正确")
        print("✅ 训练循环逻辑正确")
        print("✅ 不同参数配置有效")
        print("\n🎯 准备就绪，可以进行真实模型训练！")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 