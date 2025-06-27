#!/usr/bin/env python3
"""
智能平衡数据加载器 - 自动检测并重新生成不平衡批次
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict
import warnings

class SmartBalancedGEEDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
        # 按性别分组
        self.male_data = [item for item in data if item['gender'] == 'male']
        self.female_data = [item for item in data if item['gender'] == 'female']
        
        print(f"📊 数据分布: male={len(self.male_data)}, female={len(self.female_data)}")
        
        # 确保有足够的数据
        if len(self.male_data) == 0 or len(self.female_data) == 0:
            raise ValueError("数据中必须包含男性和女性样本")
        
        # 确保有足够的样本进行重新生成
        min_samples = min(len(self.male_data), len(self.female_data))
        if min_samples < 10:
            warnings.warn(f"样本数量较少 (min={min_samples})，可能影响批次生成质量")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def generate_balanced_batch(self, batch_size: int, max_retries: int = 10):
        """生成一个平衡的批次，如果失败会重试"""
        if batch_size < 2:
            raise ValueError("batch_size必须>=2才能保证性别平衡")
        
        # 计算每个性别需要的样本数
        male_per_batch = batch_size // 2
        female_per_batch = batch_size - male_per_batch
        
        for attempt in range(max_retries):
            try:
                batch = []
                
                # 🔧 修复: 使用更好的随机策略确保平衡
                # 强制选择所需数量的男性样本
                if len(self.male_data) >= male_per_batch:
                    male_samples = random.sample(self.male_data, male_per_batch)
                else:
                    # 如果男性样本不够，用替换的方式采样
                    male_samples = random.choices(self.male_data, k=male_per_batch)
                
                batch.extend(male_samples)
                
                # 强制选择所需数量的女性样本
                if len(self.female_data) >= female_per_batch:
                    female_samples = random.sample(self.female_data, female_per_batch)
                else:
                    # 如果女性样本不够，用替换的方式采样
                    female_samples = random.choices(self.female_data, k=female_per_batch)
                
                batch.extend(female_samples)
                
                # 打乱批次内的顺序
                random.shuffle(batch)
                
                # 最终验证批次平衡性
                male_count = sum(1 for item in batch if item['gender'] == 'male')
                female_count = sum(1 for item in batch if item['gender'] == 'female')
                
                if male_count > 0 and female_count > 0:
                    return batch
                else:
                    print(f"❌ 尝试 {attempt+1}: 批次不平衡 (male={male_count}, female={female_count})，重新生成...")
                    
            except Exception as e:
                print(f"⚠️ 尝试 {attempt+1} 失败: {e}")
                continue
        
        # 如果所有尝试都失败，返回一个强制平衡的批次
        print(f"❌ {max_retries} 次尝试后仍然失败，强制生成平衡批次")
        return self._force_balanced_batch(batch_size)
    
    def _force_balanced_batch(self, batch_size: int):
        """强制生成一个平衡批次"""
        batch = []
        male_per_batch = batch_size // 2
        female_per_batch = batch_size - male_per_batch
        
        # 强制添加男性样本（允许重复）
        for _ in range(male_per_batch):
            batch.append(random.choice(self.male_data))
        
        # 强制添加女性样本（允许重复）
        for _ in range(female_per_batch):
            batch.append(random.choice(self.female_data))
        
        random.shuffle(batch)
        return batch

class SmartBalancedDataLoader:
    """智能平衡数据加载器"""
    def __init__(self, dataset: SmartBalancedGEEDataset, batch_size: int, num_batches: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_idx = 0
        
        print(f"🧠 智能数据加载器初始化: batch_size={batch_size}, num_batches={num_batches}")
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.num_batches:
            raise StopIteration
        
        # 🔧 特殊处理第一个批次，确保绝对平衡
        if self.current_idx == 0:
            print(f"🎯 特殊处理第一个批次")
            batch = self._generate_guaranteed_balanced_batch()
        else:
            # 其他批次使用标准方法
            batch = self.dataset.generate_balanced_batch(self.batch_size)
        
        self.current_idx += 1
        
        # 应用collate函数并验证
        return self._smart_collate(batch)
    
    def _generate_guaranteed_balanced_batch(self):
        """保证生成平衡的第一个批次"""
        batch_size = self.batch_size
        male_per_batch = batch_size // 2
        female_per_batch = batch_size - male_per_batch
        
        batch = []
        
        # 强制选择男性样本（取前N个或轮换选择）
        if len(self.dataset.male_data) >= male_per_batch:
            # 不使用随机，而是轮换选择
            male_samples = self.dataset.male_data[:male_per_batch]
        else:
            # 重复选择
            male_samples = (self.dataset.male_data * ((male_per_batch // len(self.dataset.male_data)) + 1))[:male_per_batch]
        
        batch.extend(male_samples)
        
        # 强制选择女性样本（取前N个或轮换选择）
        if len(self.dataset.female_data) >= female_per_batch:
            # 不使用随机，而是轮换选择
            female_samples = self.dataset.female_data[:female_per_batch]
        else:
            # 重复选择
            female_samples = (self.dataset.female_data * ((female_per_batch // len(self.dataset.female_data)) + 1))[:female_per_batch]
        
        batch.extend(female_samples)
        
        # 验证
        male_count = sum(1 for item in batch if item['gender'] == 'male')
        female_count = sum(1 for item in batch if item['gender'] == 'female')
        print(f"🎯 第一批次: male={male_count}, female={female_count} (强制平衡)")
        
        # 打乱顺序
        random.shuffle(batch)
        
        return batch
    
    def _smart_collate(self, batch, max_regenerate: int = 3):
        """智能collate函数，如果检测到不平衡会重新生成"""
        inputs = [item["input"] for item in batch]
        genders = [item["gender"] for item in batch]
        
        # 检查批次平衡性
        male_count = sum(1 for g in genders if g == 'male')
        female_count = sum(1 for g in genders if g == 'female')
        
        # 如果不平衡，尝试重新生成
        regenerate_count = 0
        while (male_count == 0 or female_count == 0) and regenerate_count < max_regenerate:
            print(f"🔄 检测到不平衡批次 (male={male_count}, female={female_count})，重新生成...")
            
            # 重新生成批次
            batch = self.dataset.generate_balanced_batch(self.batch_size)
            inputs = [item["input"] for item in batch]
            genders = [item["gender"] for item in batch]
            
            # 重新检查
            male_count = sum(1 for g in genders if g == 'male')
            female_count = sum(1 for g in genders if g == 'female')
            regenerate_count += 1
        
        # 最终检查
        if male_count == 0:
            print("❌ 最终警告: 批次中没有男性样本！")
        if female_count == 0:
            print("❌ 最终警告: 批次中没有女性样本！")
        
        if male_count > 0 and female_count > 0:
            print(f"✅ 平衡批次: male={male_count}, female={female_count}")
        
        return {
            "input": inputs,
            "gender": genders
        }

def create_smart_balanced_dataloader(data: List[Dict], batch_size: int, num_batches: int = 10):
    """创建智能平衡数据加载器"""
    
    if batch_size < 2:
        print("⚠️ 警告: batch_size < 2，无法保证性别平衡")
        # 回退到普通DataLoader
        dataset = SmartBalancedGEEDataset(data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dataset = SmartBalancedGEEDataset(data)
    
    print(f"🧠 创建智能平衡数据加载器")
    print(f"   批次大小: {batch_size}")
    print(f"   批次数量: {num_batches}")
    print(f"   每批次配置: male={batch_size//2}, female={batch_size-batch_size//2}")
    
    return SmartBalancedDataLoader(dataset, batch_size, num_batches)

# 测试函数
if __name__ == "__main__":
    # 测试智能平衡数据加载器
    import sys
    sys.path.append('.')
    from dataset.gee_processor import GEEProcessor
    
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[0]["content"]
    
    processor = GEEProcessor(MockTokenizer())
    test_data = processor.create_test_data(num_samples=20)
    
    print("🧪 测试智能平衡数据加载器")
    dataloader = create_smart_balanced_dataloader(test_data, batch_size=4, num_batches=5)
    
    for i, batch in enumerate(dataloader):
        print(f"\n=== 批次 {i+1} ===")
        print(f"输入数量: {len(batch['input'])}")
        print(f"性别分布: {batch['gender']}")
        
        # 验证平衡性
        male_count = sum(1 for g in batch['gender'] if g == 'male')
        female_count = sum(1 for g in batch['gender'] if g == 'female')
        
        if male_count > 0 and female_count > 0:
            print(f"✅ 批次完美平衡: male={male_count}, female={female_count}")
        else:
            print(f"❌ 批次仍然不平衡: male={male_count}, female={female_count}")
        
        if i >= 4:  # 测试5个批次
            break
    
    print("\n�� 智能平衡数据加载器测试完成!") 