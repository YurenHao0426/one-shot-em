#!/usr/bin/env python3
"""
测试第一个批次修复是否有效
"""
import sys
sys.path.append('.')

from dataset.gee_processor import GEEProcessor
from smart_balanced_dataloader import create_smart_balanced_dataloader

class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

print("🧪 测试第一个批次修复")
print("="*50)

# 创建测试数据
processor = GEEProcessor(MockTokenizer())
test_data = processor.create_test_data(num_samples=20)

# 测试多次运行，确保第一个批次总是平衡
print("🔄 测试5次运行，确保第一个批次总是平衡:")

for test_run in range(5):
    print(f"\n--- 测试运行 {test_run + 1} ---")
    
    # 创建数据加载器
    dataloader = create_smart_balanced_dataloader(test_data, batch_size=2, num_batches=3)
    
    # 只关注第一个批次
    first_batch = next(iter(dataloader))
    
    male_count = sum(1 for g in first_batch['gender'] if g == 'male')
    female_count = sum(1 for g in first_batch['gender'] if g == 'female')
    
    print(f"第一批次: {first_batch['gender']}")
    print(f"统计: male={male_count}, female={female_count}")
    
    if male_count > 0 and female_count > 0:
        print("✅ 第一批次平衡")
    else:
        print("❌ 第一批次仍然不平衡！")

print(f"\n🎯 如果以上所有测试都显示'✅ 第一批次平衡'，则修复成功！") 