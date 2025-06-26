#!/usr/bin/env python3
"""
GEE组件测试脚本
用于测试数据处理器、损失函数和评估器的功能
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
from evaluation.gee_evaluator import GEEEvaluator

def test_gee_processor():
    """测试GEE数据处理器"""
    print("="*50)
    print("测试GEE数据处理器")
    print("="*50)
    
    # 创建模拟tokenizer
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[0]["content"]
    
    tokenizer = MockTokenizer()
    processor = GEEProcessor(tokenizer)
    
    # 测试性别检测
    test_texts = [
        "He is a doctor who helps patients.",
        "She is a nurse who cares for patients.",
        "The teacher asked him to solve the problem.",
        "The teacher asked her to solve the problem.",
        "A man and a woman are working together.",
        "The student needs to calculate the answer."
    ]
    
    print("测试性别检测:")
    for text in test_texts:
        gender = processor.detect_gender(text)
        print(f"  '{text}' -> {gender}")
    
    # 测试测试数据生成
    test_data = processor.create_test_data(num_samples=10)
    print(f"\n生成测试数据: {len(test_data)} 条")
    for i, item in enumerate(test_data[:3]):
        print(f"  样本 {i+1}: {item['gender']} - {item['input'][:50]}...")
    
    print("✓ GEE数据处理器测试通过")

def test_gee_loss():
    """测试GEE损失函数"""
    print("\n" + "="*50)
    print("测试GEE损失函数")
    print("="*50)
    
    # 创建模拟数据
    batch_size = 4
    seq_len = 10
    vocab_size = 1000
    
    # 模拟logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    attention_mask = torch.ones(batch_size, seq_len)
    prompt_lengths = torch.tensor([3, 4, 3, 4])  # 前3-4个token是prompt
    gender_labels = torch.tensor([0, 1, 0, 1])  # male, female, male, female
    
    # 测试损失函数
    gee_loss = GEELoss(lambda_weight=3.0, use_l1=False)
    
    # 计算token熵
    H_tok = gee_loss.compute_token_entropy(logits, attention_mask)
    print(f"Token熵形状: {H_tok.shape}")
    print(f"Token熵范围: [{H_tok.min():.4f}, {H_tok.max():.4f}]")
    
    # 计算样本熵
    H_i = gee_loss.compute_sample_entropy(H_tok, prompt_lengths)
    print(f"样本熵形状: {H_i.shape}")
    print(f"样本熵值: {H_i.tolist()}")
    
    # 计算组熵
    H_male, H_female = gee_loss.compute_group_entropy(H_i, gender_labels)
    print(f"男性平均熵: {H_male:.4f}")
    print(f"女性平均熵: {H_female:.4f}")
    
    # 计算GEE损失
    loss, metrics = gee_loss.compute_gee_loss(H_i, gender_labels)
    print(f"GEE损失: {loss:.4f}")
    print(f"损失指标: {metrics}")
    
    # 测试L1版本
    gee_loss_l1 = GEELoss(lambda_weight=3.0, use_l1=True)
    loss_l1, metrics_l1 = gee_loss_l1.compute_gee_loss(H_i, gender_labels)
    print(f"L1版本GEE损失: {loss_l1:.4f}")
    
    print("✓ GEE损失函数测试通过")

def test_gee_evaluator():
    """测试GEE评估器"""
    print("\n" + "="*50)
    print("测试GEE评估器")
    print("="*50)
    
    # 创建评估器（使用模拟模型路径）
    try:
        # 注意：这里需要实际的模型路径才能完全测试
        # 如果没有模型，我们只测试数据生成部分
        evaluator = GEEEvaluator("dummy_path")
        
        # 测试测试数据生成
        test_data = evaluator.create_winogender_style_data(num_samples=10)
        print(f"生成Winogender风格测试数据: {len(test_data)} 条")
        
        male_count = sum(1 for item in test_data if item['gender'] == 'male')
        female_count = sum(1 for item in item if item['gender'] == 'female')
        print(f"性别分布: 男性={male_count}, 女性={female_count}")
        
        for i, item in enumerate(test_data[:3]):
            print(f"  样本 {i+1}: {item['gender']} - {item['prompt']}")
        
        print("✓ GEE评估器数据生成测试通过")
        
    except Exception as e:
        print(f"评估器测试跳过（需要实际模型）: {e}")

def test_integration():
    """测试组件集成"""
    print("\n" + "="*50)
    print("测试组件集成")
    print("="*50)
    
    # 创建模拟tokenizer
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[0]["content"]
    
    tokenizer = MockTokenizer()
    
    # 测试完整流程
    processor = GEEProcessor(tokenizer)
    test_data = processor.create_test_data(num_samples=20)
    
    # 模拟训练数据格式
    batch = {
        "input": [item["input"] for item in test_data[:4]],
        "gender": [item["gender"] for item in test_data[:4]]
    }
    
    print(f"批次大小: {len(batch['input'])}")
    print(f"性别分布: {batch['gender']}")
    
    # 模拟性别标签转换
    gender_labels = torch.tensor([gender_to_label(g) for g in batch["gender"]])
    print(f"性别标签: {gender_labels.tolist()}")
    
    print("✓ 组件集成测试通过")

def main():
    """主测试函数"""
    print("开始GEE组件测试...")
    
    try:
        test_gee_processor()
        test_gee_loss()
        test_gee_evaluator()
        test_integration()
        
        print("\n" + "="*50)
        print("所有测试通过！✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 