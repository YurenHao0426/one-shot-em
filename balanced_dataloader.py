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

def balanced_collate(batch):
    """平衡的collate函数"""
    inputs = [item["input"] for item in batch]
    genders = [item["gender"] for item in batch]
    
    # 检查批次平衡性
    male_count = sum(1 for g in genders if g == 'male')
    female_count = sum(1 for g in genders if g == 'female')
    
    print(f"🔍 批次检查: male={male_count}, female={female_count}")
    
    if male_count == 0:
        print("⚠️ 警告: 批次中没有男性样本！")
    if female_count == 0:
        print("⚠️ 警告: 批次中没有女性样本！")
    
    return {
        "input": inputs,
        "gender": genders
    }

def create_balanced_dataloader(data: List[Dict], batch_size: int, num_batches: int = 10):
    """创建平衡的数据加载器"""
    dataset = BalancedGEEDataset(data)
    
    if batch_size < 2:
        print("⚠️ 警告: batch_size < 2，无法保证性别平衡")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=balanced_collate)
    
    # 创建平衡批次
    balanced_batches = dataset.create_balanced_batches(batch_size, num_batches)
    
    # 展平批次为单个数据点
    flat_data = []
    for batch in balanced_batches:
        flat_data.extend(batch)
    
    # 创建新的数据集
    balanced_dataset = BalancedGEEDataset(flat_data)
    
    return DataLoader(balanced_dataset, batch_size=batch_size, shuffle=False, collate_fn=balanced_collate)

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
    
    print("🧪 测试平衡数据加载器")
    dataloader = create_balanced_dataloader(test_data, batch_size=4, num_batches=3)
    
    for i, batch in enumerate(dataloader):
        print(f"\n批次 {i+1}:")
        print(f"  输入数量: {len(batch['input'])}")
        print(f"  性别: {batch['gender']}")
        if i >= 2:  # 只测试前3个批次
            break 