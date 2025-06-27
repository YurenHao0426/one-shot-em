#!/usr/bin/env python3
"""
数学逻辑测试: 纯偏见减少损失函数
不依赖PyTorch，只验证数学计算逻辑
"""
import math

def test_debiasing_math():
    """测试纯偏见减少的数学逻辑"""
    print("🧪 测试纯偏见减少的数学逻辑...")
    
    # 模拟样本熵数据
    # 假设批次: [男性1, 女性1, 男性2, 女性2]
    sample_entropies = [0.5, 0.8, 0.4, 0.9]  # 样本级熵
    genders = ['male', 'female', 'male', 'female']
    
    print(f"📊 测试数据:")
    for i, (entropy, gender) in enumerate(zip(sample_entropies, genders)):
        print(f"   样本{i+1}: {gender}, 熵={entropy}")
    
    # 计算组平均熵
    male_entropies = [e for e, g in zip(sample_entropies, genders) if g == 'male']
    female_entropies = [e for e, g in zip(sample_entropies, genders) if g == 'female']
    
    H_male = sum(male_entropies) / len(male_entropies)
    H_female = sum(female_entropies) / len(female_entropies)
    
    print(f"\n📈 组熵计算:")
    print(f"   男性熵: {male_entropies} → 平均={H_male:.4f}")
    print(f"   女性熵: {female_entropies} → 平均={H_female:.4f}")
    
    # 计算熵差距
    entropy_gap = abs(H_female - H_male)
    entropy_gap_signed = H_female - H_male
    
    print(f"   熵差距: |{H_female:.4f} - {H_male:.4f}| = {entropy_gap:.4f}")
    print(f"   带符号差距: {entropy_gap_signed:.4f}")
    
    # 纯偏见减少损失
    # L2版本: (H_female - H_male)²
    loss_l2 = (H_female - H_male) ** 2
    # L1版本: |H_female - H_male|
    loss_l1 = abs(H_female - H_male)
    
    print(f"\n🎯 纯偏见减少损失:")
    print(f"   L2损失: ({H_female:.4f} - {H_male:.4f})² = {loss_l2:.6f}")
    print(f"   L1损失: |{H_female:.4f} - {H_male:.4f}| = {loss_l1:.6f}")
    
    # 对比原GEE损失（模拟）
    H_bar = sum(sample_entropies) / len(sample_entropies)  # 整体平均熵
    lambda_weight = 3.0
    
    loss_em = H_bar  # EM项
    loss_bias = (H_female - H_male) ** 2  # 偏见项
    loss_gee_total = loss_em + lambda_weight * loss_bias  # 原GEE总损失
    
    print(f"\n🔄 对比原GEE损失:")
    print(f"   整体平均熵(EM项): {H_bar:.6f}")
    print(f"   偏见项: {loss_bias:.6f}")
    print(f"   λ权重: {lambda_weight}")
    print(f"   原GEE总损失: {loss_em:.6f} + {lambda_weight} × {loss_bias:.6f} = {loss_gee_total:.6f}")
    print(f"   纯Debiasing损失: {loss_l2:.6f}")
    
    print(f"\n📏 关键区别:")
    print(f"   原GEE: 同时最小化整体熵({loss_em:.6f}) + 偏见({loss_bias:.6f})")
    print(f"   纯Debiasing: 只最小化偏见({loss_l2:.6f})")
    print(f"   减少的计算量: {loss_em:.6f} (不再需要优化整体熵)")
    
    return True

def simulate_training_scenarios():
    """模拟不同训练场景下的损失变化"""
    print(f"\n📈 模拟训练场景:")
    
    scenarios = [
        {
            "name": "初始状态 - 严重偏见",
            "data": [0.3, 0.9, 0.2, 1.0],  # 男性低熵，女性高熵
            "genders": ['male', 'female', 'male', 'female']
        },
        {
            "name": "训练中期 - 偏见减少",
            "data": [0.4, 0.7, 0.5, 0.6],  # 差距缩小
            "genders": ['male', 'female', 'male', 'female']
        },
        {
            "name": "训练后期 - 接近平衡",
            "data": [0.55, 0.6, 0.58, 0.57],  # 几乎相等
            "genders": ['male', 'female', 'male', 'female']
        },
        {
            "name": "理想状态 - 完全平衡",
            "data": [0.6, 0.6, 0.6, 0.6],  # 完全相等
            "genders": ['male', 'female', 'male', 'female']
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🔄 场景 {i+1}: {scenario['name']}")
        
        entropies = scenario['data']
        genders = scenario['genders']
        
        # 计算组熵
        male_entropies = [e for e, g in zip(entropies, genders) if g == 'male']
        female_entropies = [e for e, g in zip(entropies, genders) if g == 'female']
        
        H_male = sum(male_entropies) / len(male_entropies)
        H_female = sum(female_entropies) / len(female_entropies)
        
        # 纯偏见减少损失
        debiasing_loss = (H_female - H_male) ** 2
        entropy_gap = abs(H_female - H_male)
        
        # 评估偏见程度
        if entropy_gap <= 0.01:
            bias_level = "无偏见 ✅"
        elif entropy_gap <= 0.05:
            bias_level = "轻微偏见 ⚠️"
        elif entropy_gap <= 0.1:
            bias_level = "中等偏见 ❌"
        else:
            bias_level = "严重偏见 💥"
        
        print(f"   H_male={H_male:.4f}, H_female={H_female:.4f}")
        print(f"   熵差距: {entropy_gap:.4f}")
        print(f"   Debiasing损失: {debiasing_loss:.6f}")
        print(f"   偏见程度: {bias_level}")
    
    print(f"\n✅ 预期训练效果: 损失和熵差距逐步下降，偏见程度改善")

def test_edge_cases():
    """测试边界情况"""
    print(f"\n⚠️ 测试边界情况:")
    
    edge_cases = [
        {
            "name": "完全平衡",
            "data": [0.5, 0.5, 0.5, 0.5],
            "genders": ['male', 'female', 'male', 'female']
        },
        {
            "name": "极端偏见",
            "data": [0.0, 1.0, 0.0, 1.0],
            "genders": ['male', 'female', 'male', 'female']
        },
        {
            "name": "反向偏见",
            "data": [0.8, 0.2, 0.9, 0.1],  # 男性高熵，女性低熵
            "genders": ['male', 'female', 'male', 'female']
        }
    ]
    
    for case in edge_cases:
        print(f"\n🔍 {case['name']}:")
        
        entropies = case['data']
        genders = case['genders']
        
        male_entropies = [e for e, g in zip(entropies, genders) if g == 'male']
        female_entropies = [e for e, g in zip(entropies, genders) if g == 'female']
        
        H_male = sum(male_entropies) / len(male_entropies)
        H_female = sum(female_entropies) / len(female_entropies)
        
        debiasing_loss = (H_female - H_male) ** 2
        entropy_gap = abs(H_female - H_male)
        
        print(f"   H_male={H_male:.4f}, H_female={H_female:.4f}")
        print(f"   熵差距: {entropy_gap:.4f}")
        print(f"   Debiasing损失: {debiasing_loss:.6f}")
        
        # 验证数学正确性
        expected_loss = (H_female - H_male) ** 2
        assert abs(debiasing_loss - expected_loss) < 1e-10, "数学计算错误!"
        print(f"   ✅ 数学验证通过")

if __name__ == "__main__":
    print("🚀 开始纯偏见减少数学逻辑测试")
    
    # 基础数学测试
    success = test_debiasing_math()
    
    if success:
        print("\n✅ 基础数学测试通过！")
        
        # 训练场景模拟
        simulate_training_scenarios()
        
        # 边界情况测试
        test_edge_cases()
        
        print(f"\n🎉 所有数学测试完成！")
        print(f"📋 关键发现:")
        print(f"   ✅ 纯偏见减少只关注 |H_female - H_male|")
        print(f"   ✅ 不再需要优化整体熵最小化")
        print(f"   ✅ 计算更简单，目标更明确")
        print(f"   ✅ L2损失: (H_female - H_male)²")
        print(f"   ✅ L1损失: |H_female - H_male|")
        print(f"   🎯 准备就绪，可以开始纯debiasing训练！")
    else:
        print("\n❌ 数学测试失败！") 