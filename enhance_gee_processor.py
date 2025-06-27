#!/usr/bin/env python3
"""
增强GEE处理器以支持真实数据集
支持Numina数学推理数据和其他真实数据源
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import sys
sys.path.append('.')

from dataset.gee_processor import GEEProcessor

class EnhancedGEEProcessor(GEEProcessor):
    """增强版GEE处理器，支持多种真实数据源"""
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.name_patterns = {
            'male': ['Tom', 'John', 'Mike', 'Bob', 'David', 'James', 'Robert', 'Michael', 'William', 'Richard'],
            'female': ['Sarah', 'Lisa', 'Emma', 'Alice', 'Mary', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan']
        }
    
    def process_numina_data(self, file_path: str, target_size: int = 1000) -> list:
        """处理Numina数学推理数据"""
        print(f"📊 处理Numina数据: {file_path}")
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        print(f"原始数据量: {len(df)}")
        
        # 随机采样
        if len(df) > target_size:
            df = df.sample(n=target_size, random_state=42)
            print(f"采样后数据量: {len(df)}")
        
        processed_data = []
        for idx, row in df.iterrows():
            # 提取问题和答案
            problem = row.get('problem', row.get('question', ''))
            solution = row.get('solution', row.get('answer', ''))
            
            if problem and solution:
                # 生成性别平衡的变体
                male_version = self._genderize_text(problem, 'male')
                female_version = self._genderize_text(problem, 'female')
                
                processed_data.extend([
                    {
                        'input': self.apply_chat_template(male_version),
                        'output': solution,
                        'gender': 'male',
                        'original_id': idx,
                        'source': 'numina'
                    },
                    {
                        'input': self.apply_chat_template(female_version), 
                        'output': solution,
                        'gender': 'female',
                        'original_id': idx,
                        'source': 'numina'
                    }
                ])
        
        print(f"✅ 处理完成，生成 {len(processed_data)} 个样本")
        return processed_data
    
    def process_1shot_rlvr_data(self, file_path: str) -> list:
        """处理1shot RLVR数据"""
        print(f"�� 处理1shot RLVR数据: {file_path}")
        
        df = pd.read_parquet(file_path)
        print(f"原始数据量: {len(df)}")
        
        processed_data = []
        for idx, row in df.iterrows():
            # 根据实际数据结构调整
            prompt = row.get('prompt', row.get('input', ''))
            
            if prompt:
                # 生成性别变体
                for gender in ['male', 'female']:
                    genderized_prompt = self._genderize_text(prompt, gender)
                    
                    processed_data.append({
                        'input': self.apply_chat_template(genderized_prompt),
                        'gender': gender,
                        'original_id': idx,
                        'source': '1shot_rlvr'
                    })
        
        print(f"✅ 处理完成，生成 {len(processed_data)} 个样本")
        return processed_data
    
    def _genderize_text(self, text: str, target_gender: str) -> str:
        """将文本中的性别引用转换为指定性别"""
        
        # 选择名字
        names = self.name_patterns[target_gender]
        
        # 替换通用占位符
        if '[NAME]' in text or '{name}' in text:
            name = np.random.choice(names)
            text = text.replace('[NAME]', name).replace('{name}', name)
            return text
        
        # 检测现有性别名字并替换
        all_male_names = self.name_patterns['male']
        all_female_names = self.name_patterns['female'] 
        
        for male_name in all_male_names:
            if male_name in text:
                replacement = np.random.choice(names)
                text = text.replace(male_name, replacement)
                break
                
        for female_name in all_female_names:
            if female_name in text:
                replacement = np.random.choice(names)
                text = text.replace(female_name, replacement)
                break
        
        # 如果没有找到名字，随机添加一个
        if not any(name in text for name in all_male_names + all_female_names):
            name = np.random.choice(names)
            # 在合适的地方插入名字
            if "person" in text.lower():
                text = text.replace("person", name)
            elif "student" in text.lower():
                text = text.replace("student", f"student named {name}")
            elif "someone" in text.lower():
                text = text.replace("someone", name)
            else:
                # 在句子开头添加
                text = f"{name} is working on this problem: {text}"
        
        return text
    
    def create_balanced_dataset(self, data_sources: list, balance_method: str = 'oversample') -> list:
        """创建性别平衡的数据集"""
        
        all_data = []
        for source_config in data_sources:
            source_type = source_config['type']
            file_path = source_config['path']
            
            if source_type == 'numina':
                data = self.process_numina_data(file_path, source_config.get('target_size', 1000))
            elif source_type == '1shot_rlvr':
                data = self.process_1shot_rlvr_data(file_path)
            else:
                print(f"⚠️ 未知数据源类型: {source_type}")
                continue
                
            all_data.extend(data)
        
        # 统计性别分布
        male_data = [item for item in all_data if item['gender'] == 'male']
        female_data = [item for item in all_data if item['gender'] == 'female']
        
        print(f"\n📊 原始数据分布:")
        print(f"   男性样本: {len(male_data)}")
        print(f"   女性样本: {len(female_data)}")
        
        # 平衡处理
        if balance_method == 'oversample':
            target_size = max(len(male_data), len(female_data))
            
            if len(male_data) < target_size:
                male_data = male_data * (target_size // len(male_data)) + male_data[:target_size % len(male_data)]
            if len(female_data) < target_size:
                female_data = female_data * (target_size // len(female_data)) + female_data[:target_size % len(female_data)]
                
        elif balance_method == 'undersample':
            target_size = min(len(male_data), len(female_data))
            male_data = male_data[:target_size]
            female_data = female_data[:target_size]
        
        balanced_data = male_data + female_data
        np.random.shuffle(balanced_data)
        
        print(f"📊 平衡后数据分布:")
        male_count = sum(1 for item in balanced_data if item['gender'] == 'male')
        female_count = sum(1 for item in balanced_data if item['gender'] == 'female')
        print(f"   男性样本: {male_count}")
        print(f"   女性样本: {female_count}")
        print(f"   总样本数: {len(balanced_data)}")
        
        return balanced_data

def main():
    """示例用法"""
    from transformers import AutoTokenizer
    
    print("🔧 测试增强版GEE处理器...")
    
    # 初始化
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", trust_remote_code=True)
    processor = EnhancedGEEProcessor(tokenizer)
    
    # 配置数据源
    data_sources = [
        {
            'type': 'numina',
            'path': 'dataset/numina/numina_00.parquet',
            'target_size': 100  # 测试用小样本
        }
        # 可以添加更多数据源
    ]
    
    # 处理数据
    try:
        balanced_data = processor.create_balanced_dataset(data_sources, balance_method='oversample')
        
        # 保存结果
        output_file = 'enhanced_training_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(balanced_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 增强数据已保存: {output_file}")
        
        # 显示示例
        print(f"\n📝 数据示例:")
        for i, sample in enumerate(balanced_data[:4]):
            print(f"  示例 {i+1} ({sample['gender']}):")
            print(f"    输入: {sample['input'][:100]}...")
            print()
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    main()
