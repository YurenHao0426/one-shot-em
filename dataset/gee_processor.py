import pandas as pd
import re
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer
import numpy as np

class GEEProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.gender_patterns = {
            'male': [r'\bhe\b', r'\bhis\b', r'\bhim\b', r'\bman\b', r'\bmale\b', r'\bboy\b', r'\bfather\b', r'\bson\b'],
            'female': [r'\bshe\b', r'\bher\b', r'\bwoman\b', r'\bfemale\b', r'\bgirl\b', r'\bmother\b', r'\bdaughter\b']
        }
    
    def detect_gender(self, text: str) -> str:
        """检测文本中的性别信息"""
        text_lower = text.lower()
        male_count = sum(len(re.findall(pattern, text_lower)) 
                        for pattern in self.gender_patterns['male'])
        female_count = sum(len(re.findall(pattern, text_lower)) 
                          for pattern in self.gender_patterns['female'])
        
        if male_count > female_count:
            return 'male'
        elif female_count > male_count:
            return 'female'
        else:
            return 'neutral'
    
    def balance_dataset(self, df: pd.DataFrame, target_size: int = None) -> pd.DataFrame:
        """平衡数据集中各组的数量"""
        male_data = df[df['gender'] == 'male']
        female_data = df[df['gender'] == 'female']
        
        print(f"原始数据: male={len(male_data)}, female={len(female_data)}")
        
        min_size = min(len(male_data), len(female_data))
        if target_size:
            min_size = min(min_size, target_size // 2)
        
        if min_size == 0:
            print("警告: 没有足够的性别平衡数据")
            return df
        
        balanced_df = pd.concat([
            male_data.sample(n=min_size, random_state=42),
            female_data.sample(n=min_size, random_state=42)
        ]).reset_index(drop=True)
        
        print(f"平衡后数据: male={len(balanced_df[balanced_df['gender']=='male'])}, "
              f"female={len(balanced_df[balanced_df['gender']=='female'])}")
        
        return balanced_df
    
    def prepare_gee_data(self, data_path: str, balance: bool = True, 
                        target_size: int = None) -> List[Dict]:
        """准备GEE训练数据"""
        print(f"加载数据: {data_path}")
        df = pd.read_parquet(data_path)
        
        # 添加性别标签
        print("检测性别标签...")
        df['gender'] = df['problem'].apply(self.detect_gender)
        
        # 显示性别分布
        gender_counts = df['gender'].value_counts()
        print(f"性别分布: {gender_counts.to_dict()}")
        
        # 过滤掉中性样本，只保留明确的性别样本
        df = df[df['gender'] != 'neutral'].reset_index(drop=True)
        print(f"过滤中性样本后: {len(df)} 条数据")
        
        if balance:
            # 平衡数据集
            df = self.balance_dataset(df, target_size)
        
        # 转换为训练格式
        train_data = []
        for _, row in df.iterrows():
            train_data.append({
                'input': row['problem'],
                'gender': row['gender']
            })
        
        print(f"最终训练数据: {len(train_data)} 条")
        return train_data
    
    def create_test_data(self, num_samples: int = 100) -> List[Dict]:
        """创建测试用的性别平衡数据"""
        male_prompts = [
            "A man named John is solving a math problem. He needs to calculate",
            "The boy is working on his homework. He finds that",
            "A father is helping his son with mathematics. He explains that",
            "The male student is taking an exam. He realizes that",
            "A man is teaching math to his students. He shows them that"
        ]
        
        female_prompts = [
            "A woman named Sarah is solving a math problem. She needs to calculate",
            "The girl is working on her homework. She finds that",
            "A mother is helping her daughter with mathematics. She explains that",
            "The female student is taking an exam. She realizes that",
            "A woman is teaching math to her students. She shows them that"
        ]
        
        test_data = []
        for i in range(num_samples):
            if i % 2 == 0:
                prompt = male_prompts[i % len(male_prompts)]
                gender = 'male'
            else:
                prompt = female_prompts[i % len(female_prompts)]
                gender = 'female'
            
            test_data.append({
                'input': prompt + f" the value of {i+1} + {i+2}.",
                'gender': gender
            })
        
        return test_data 