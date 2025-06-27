#!/usr/bin/env python3
"""
测试修复后的GEE损失函数
"""
import torch
import sys
sys.path.append('.')

from losses.gee_loss import GEELoss, gender_to_label
from dataset.gee_processor import GEEProcessor

print("🧪 测试修复后的GEE损失函数")
print("="*50)

# 创建模拟tokenizer
class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

# 1. 测试数据生成
processor = GEEProcessor(MockTokenizer())
test_data = processor.create_test_data(num_samples=6)

print(f"📊 生成 {len(test_data)} 条测试数据")
for i, item in enumerate(test_data):
    print(f"  {i+1}. {item['gender']}: {item['input'][:50]}...")

# 2. 创建批次
batch = {
    "input": [item["input"] for item in test_data[:4]],
    "gender": [item["gender"] for item in test_data[:4]]
}

print(f"\n📦 批次信息:")
print(f"性别: {batch['gender']}")

gender_labels = torch.tensor([gender_to_label(g) for g in batch["gender"]])
print(f"标签: {gender_labels.tolist()}")

# 3. 测试修复后的损失函数
gee_loss = GEELoss(lambda_weight=1.0)  # 降低lambda权重

# 模拟合理的熵值（包含一些接近0的值）
H_i_test = torch.tensor([0.8, 0.1, 0.6, 0.2])  # male, female, male, female

print(f"\n🧮 测试修复后的GEE损失:")
print(f"输入熵值: {H_i_test.tolist()}")
print(f"性别标签: {batch['gender']}")

loss, metrics = gee_loss.compute_gee_loss(H_i_test, gender_labels)

print(f"\n📈 结果:")
print(f"总损失: {loss:.6f}")
print(f"熵最小化损失: {metrics['loss_em']:.6f}")
print(f"偏见损失: {metrics['loss_bias']:.6f}")
print(f"男性平均熵: {metrics['H_male']:.6f}")
print(f"女性平均熵: {metrics['H_female']:.6f}")
print(f"熵差距: {metrics['entropy_gap']:.6f}")
print(f"Lambda权重: {metrics['lambda_weight']}")

# 4. 验证修复效果
print(f"\n✅ 修复验证:")
if metrics['H_female'] > 0:
    print("✅ H_female不再为0")
else:
    print("❌ H_female仍为0，可能还有问题")

if metrics['entropy_gap'] < 1.0:
    print("✅ 熵差距在合理范围内")
else:
    print("⚠️ 熵差距较大")

if loss < 10.0:
    print("✅ 总损失在合理范围内")
else:
    print("⚠️ 总损失可能过大")

print(f"\n💡 修复要点:")
print("1. 移除了错误的零熵值过滤")
print("2. 简化了GEE损失计算")
print("3. 添加了调试信息")
print("4. 建议降低lambda权重到0.5-1.0") 