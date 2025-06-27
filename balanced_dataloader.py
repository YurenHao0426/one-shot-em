#!/usr/bin/env python3
"""
平衡的数据加载器 - 确保每个批次包含男女样本
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict

class BalancedGEEDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
        # 按性别分组
        self.male_data = [item for item in data if item['gender'] == 'male']
        self.female_data = [item for item in data if item['gender'] == 'female']
        
        print(f"📊 数据分布: male={len(self.male_data)}, female={len(self.female_data)}")
        
        # 确保有足够的数据
        if len(self.male_data) == 0 or len(self.female_data) == 0:
            raise ValueError("数据中必须包含男性和女性样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def create_balanced_batches(self, batch_size: int, num_batches: int = None):
        """创建平衡的批次"""
        if batch_size < 2:
            raise ValueError("batch_size必须>=2才能保证性别平衡")
        
        # 每个批次中男女样本的数量
        male_per_batch = batch_size // 2
        female_per_batch = batch_size - male_per_batch
        
        batches = []
        max_batches = num_batches or (len(self.data) // batch_size)
        
        for i in range(max_batches):
            batch = []
            
            # 随机选择男性样本
            male_samples = random.sample(self.male_data, 
                                       min(male_per_batch, len(self.male_data)))
            batch.extend(male_samples)
            
            # 随机选择女性样本
            female_samples = random.sample(self.female_data, 
                                         min(female_per_batch, len(self.female_data)))
            batch.extend(female_samples)
            
            # 打乱批次内的顺序
            random.shuffle(batch)
            batches.append(batch)
            
            print(f"批次 {i+1}: male={len(male_samples)}, female={len(female_samples)}")
        
        return batches

def balanced_collate(batch, verbose=False):
    """平衡的collate函数"""
    inputs = [item["input"] for item in batch]
    genders = [item["gender"] for item in batch]
    
    # 检查批次平衡性
    male_count = sum(1 for g in genders if g == 'male')
    female_count = sum(1 for g in genders if g == 'female')
    
    if verbose:
        print(f"🔍 批次检查: male={male_count}, female={female_count}")
    
    # 只在不平衡时打印警告
    if male_count == 0:
        print("⚠️ 警告: 批次中没有男性样本！")
    if female_count == 0:
        print("⚠️ 警告: 批次中没有女性样本！")
    
    return {
        "input": inputs,
        "gender": genders
    }

class BalancedDataLoader:
    """自定义平衡数据加载器"""
    def __init__(self, balanced_batches):
        self.batches = balanced_batches
        self.current_idx = 0
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.batches):
            raise StopIteration
        
        batch = self.batches[self.current_idx]
        self.current_idx += 1
        
        # 应用collate函数 (不显示详细信息，只显示警告)
        return balanced_collate(batch, verbose=False)

def create_balanced_dataloader(data: List[Dict], batch_size: int, num_batches: int = 10):
    """创建平衡的数据加载器 - 修复版本"""
    dataset = BalancedGEEDataset(data)
    
    if batch_size < 2:
        print("⚠️ 警告: batch_size < 2，无法保证性别平衡")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: balanced_collate(batch, verbose=True))
    
    # 🔧 修复: 直接返回预构造的平衡批次
    balanced_batches = dataset.create_balanced_batches(batch_size, num_batches)
    
    print(f"✅ 创建了 {len(balanced_batches)} 个平衡批次")
    
    return BalancedDataLoader(balanced_batches)

# 测试函数
if __name__ == "__main__":
    # 测试平衡数据加载器
    import sys
    sys.path.append('.')
    from dataset.gee_processor import GEEProcessor
    
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[0]["content"]
    
    processor = GEEProcessor(MockTokenizer())
    test_data = processor.create_test_data(num_samples=20)
    
    print("🧪 测试修复后的平衡数据加载器")
    dataloader = create_balanced_dataloader(test_data, batch_size=4, num_batches=3)
    
    for i, batch in enumerate(dataloader):
        print(f"\n批次 {i+1}:")
        print(f"  输入数量: {len(batch['input'])}")
        print(f"  性别: {batch['gender']}")
        
        # 验证平衡性
        male_count = sum(1 for g in batch['gender'] if g == 'male')
        female_count = sum(1 for g in batch['gender'] if g == 'female')
        
        if male_count > 0 and female_count > 0:
            print(f"  ✅ 批次平衡: male={male_count}, female={female_count}")
        else:
            print(f"  ❌ 批次不平衡: male={male_count}, female={female_count}")
        
        if i >= 2:  # 只测试前3个批次
            break
    
    print("\n✅ 平衡数据加载器测试完成!") 