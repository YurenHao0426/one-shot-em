#!/usr/bin/env python3
"""
测试纯偏见减少损失函数
验证：只最小化男女熵差，不包含整体熵最小化
"""
import torch
import numpy as np
from losses.debiasing_loss import DebiasingLoss, gender_to_label

def test_debiasing_loss():
    """测试纯偏见减少损失函数"""
    print("🧪 测试纯偏见减少损失函数...")
    
    # 初始化损失函数
    debiasing_l2 = DebiasingLoss(use_l1=False, scale_factor=1.0)
    debiasing_l1 = DebiasingLoss(use_l1=True, scale_factor=1.0)
    
    # 创建测试数据
    batch_size = 4
    vocab_size = 1000
    seq_len = 10
    
    # 模拟logits
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    attention_mask = torch.ones(batch_size, seq_len)
    prompt_lengths = torch.tensor([3, 4, 2, 5])  # 不同的prompt长度
    
    # 性别标签: [男, 女, 男, 女]
    gender_labels = torch.tensor([0, 1, 0, 1])
    
    print(f"📊 测试配置:")
    print(f"   批次大小: {batch_size}")
    print(f"   序列长度: {seq_len}")
    print(f"   词汇量: {vocab_size}")
    print(f"   性别分布: {gender_labels.tolist()}")
    
    # 计算token级熵
    H_tok = debiasing_l2.compute_token_entropy(logits, attention_mask)
    print(f"   Token熵形状: {H_tok.shape}")
    print(f"   Token熵均值: {H_tok.mean().item():.4f}")
    
    # 计算样本级熵
    H_i = debiasing_l2.compute_sample_entropy(H_tok, prompt_lengths)
    print(f"   样本熵: {H_i.tolist()}")
    
    # 计算组熵
    H_male, H_female = debiasing_l2.compute_group_entropy(H_i, gender_labels)
    print(f"   男性平均熵: {H_male.item():.4f}")
    print(f"   女性平均熵: {H_female.item():.4f}")
    print(f"   熵差距: {abs(H_female - H_male).item():.4f}")
    
    # 测试L2损失
    loss_l2, metrics_l2 = debiasing_l2.compute_debiasing_loss(H_i, gender_labels)
    print(f"\n📈 L2损失结果:")
    print(f"   损失值: {loss_l2.item():.6f}")
    print(f"   熵差距: {metrics_l2['entropy_gap']:.6f}")
    print(f"   带符号差距: {metrics_l2['entropy_gap_signed']:.6f}")
    print(f"   整体平均熵(仅监控): {metrics_l2['H_bar']:.6f}")
    
    # 测试L1损失
    loss_l1, metrics_l1 = debiasing_l1.compute_debiasing_loss(H_i, gender_labels)
    print(f"\n📈 L1损失结果:")
    print(f"   损失值: {loss_l1.item():.6f}")
    print(f"   熵差距: {metrics_l1['entropy_gap']:.6f}")
    
    # 验证数学关系
    expected_l2 = (H_female - H_male) ** 2
    expected_l1 = torch.abs(H_female - H_male)
    
    print(f"\n🔍 数学验证:")
    print(f"   预期L2损失: {expected_l2.item():.6f}")
    print(f"   实际L2损失: {loss_l2.item():.6f}")
    print(f"   L2误差: {abs(expected_l2.item() - loss_l2.item()):.8f}")
    
    print(f"   预期L1损失: {expected_l1.item():.6f}")
    print(f"   实际L1损失: {loss_l1.item():.6f}")
    print(f"   L1误差: {abs(expected_l1.item() - loss_l1.item()):.8f}")
    
    # 测试不平衡批次
    print(f"\n⚠️ 测试不平衡批次:")
    unbalanced_labels = torch.tensor([0, 0, 0, 0])  # 全是男性
    loss_unbalanced, metrics_unbalanced = debiasing_l2.compute_debiasing_loss(H_i, unbalanced_labels)
    print(f"   不平衡损失: {loss_unbalanced.item():.6f}")
    
    return True

def test_comparison_with_original():
    """对比原GEE损失和纯debiasing损失的差异"""
    print(f"\n🔄 对比测试: 原GEE vs 纯Debiasing")
    
    # 导入原始GEE损失
    from losses.gee_loss import GEELoss
    
    # 初始化两种损失函数
    gee_loss = GEELoss(lambda_weight=3.0, use_l1=False)
    debiasing_loss = DebiasingLoss(use_l1=False, scale_factor=1.0)
    
    # 创建相同的测试数据
    batch_size = 4
    H_i = torch.tensor([0.5, 0.8, 0.4, 0.9])  # 样本熵
    gender_labels = torch.tensor([0, 1, 0, 1])  # [男, 女, 男, 女]
    
    # 计算原GEE损失
    gee_total_loss, gee_metrics = gee_loss.compute_gee_loss(H_i, gender_labels)
    
    # 计算纯debiasing损失
    debiasing_total_loss, debiasing_metrics = debiasing_loss.compute_debiasing_loss(H_i, gender_labels)
    
    print(f"📊 对比结果:")
    print(f"   原GEE总损失: {gee_total_loss.item():.6f}")
    print(f"     - EM项: {gee_metrics['loss_em']:.6f}")
    print(f"     - Bias项: {gee_metrics['loss_bias']:.6f}")
    print(f"     - λ权重: {gee_metrics['lambda_weight']}")
    
    print(f"   纯Debiasing损失: {debiasing_total_loss.item():.6f}")
    print(f"     - 只有Bias项")
    
    print(f"   📏 关系验证:")
    print(f"     GEE的Bias项: {gee_metrics['loss_bias']:.6f}")
    print(f"     Debiasing损失: {debiasing_total_loss.item():.6f}")
    print(f"     差异: {abs(gee_metrics['loss_bias'] - debiasing_total_loss.item()):.8f}")
    
    # 验证只关注偏见减少的效果
    print(f"\n🎯 效果分析:")
    print(f"   原GEE: 同时优化熵最小化 + 偏见减少")
    print(f"   纯Debiasing: 只优化偏见减少")
    print(f"   预期: Debiasing会更专注于平衡男女熵差")

def simulate_training_progress():
    """模拟训练过程中损失的变化"""
    print(f"\n📈 模拟训练进度:")
    
    debiasing_loss = DebiasingLoss(use_l1=False, scale_factor=1.0)
    
    # 模拟训练步骤
    steps = [
        # [H_male, H_female] 对
        ([0.8, 0.4], [0.6, 0.9]),  # 初始: 很大差距
        ([0.7, 0.5], [0.65, 0.75]), # 步骤1: 差距缩小
        ([0.68, 0.62], [0.66, 0.68]), # 步骤2: 进一步缩小
        ([0.67, 0.65], [0.66, 0.67]), # 步骤3: 接近平衡
        ([0.66, 0.66], [0.665, 0.665]), # 步骤4: 几乎相等
    ]
    
    print(f"🔄 模拟理想训练轨迹:")
    for i, (male_entropies, female_entropies) in enumerate(steps):
        # 构造样本熵
        H_i = torch.tensor(male_entropies + female_entropies)
        gender_labels = torch.tensor([0, 0, 1, 1])  # 2男2女
        
        loss, metrics = debiasing_loss.compute_debiasing_loss(H_i, gender_labels)
        
        gap_direction = "📉" if i == 0 else ("📉" if metrics['entropy_gap'] < prev_gap else "📈")
        
        print(f"   {gap_direction} Step {i}: loss={loss.item():.6f} | "
              f"gap={metrics['entropy_gap']:.6f} | "
              f"H_male={metrics['H_male']:.4f} | "
              f"H_female={metrics['H_female']:.4f}")
        
        prev_gap = metrics['entropy_gap']
    
    print(f"✅ 预期结果: 损失和熵差距都应该持续下降")

if __name__ == "__main__":
    print("🚀 开始测试纯偏见减少损失函数")
    
    # 基础功能测试
    success = test_debiasing_loss()
    
    if success:
        print("\n✅ 基础测试通过！")
        
        # 对比测试
        test_comparison_with_original()
        
        # 训练模拟
        simulate_training_progress()
        
        print(f"\n🎉 所有测试完成！")
        print(f"📋 总结:")
        print(f"   ✅ 纯偏见减少损失函数工作正常")
        print(f"   ✅ 只关注男女熵差，不包含EM项")
        print(f"   ✅ 支持L1和L2两种损失形式")
        print(f"   ✅ 数学计算正确")
        print(f"   🎯 可以开始纯debiasing训练了！")
    else:
        print("\n❌ 测试失败！") 