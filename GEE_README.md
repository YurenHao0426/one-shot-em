# One-shot Group-Entropy Equalization (GEE)

基于One-shot Entropy Minimization框架实现的Group-Entropy Equalization (GEE)方法，用于减少大语言模型在性别、种族等敏感属性上的偏见。

## 项目概述

GEE通过在熵最小化训练中强制各敏感组的平均条件熵保持相等，让"自信度提升"在不同组之间均衡分配，从而避免模型放大刻板印象。

## 核心组件

### 1. 数据处理器 (`dataset/gee_processor.py`)
- **性别检测**: 自动检测文本中的性别信息
- **数据平衡**: 确保训练数据中各组数量平衡
- **测试数据生成**: 创建合成测试数据

### 2. 损失函数 (`losses/gee_loss.py`)
- **Token级熵计算**: 计算每个token的条件熵
- **组熵计算**: 计算各组的平均熵
- **GEE损失**: 实现L2和L1版本的GEE损失函数

### 3. 训练脚本 (`train_gee.py`)
- **GEE训练**: 支持GEE损失函数的训练流程
- **自动退火**: 可选的lambda权重自动调整
- **WandB集成**: 实验跟踪和可视化

### 4. 评估器 (`evaluation/gee_evaluator.py`)
- **偏见评估**: 评估模型在性别偏见上的表现
- **模型比较**: 比较不同模型的偏见减少效果
- **结果可视化**: 生成评估结果图表

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保有足够的GPU内存（建议16GB+）
```

### 2. 快速测试

```bash
# 运行组件测试
python test_gee_components.py

# 运行快速训练测试（使用合成数据）
bash scripts/quick_test_gee.sh
```

### 3. 完整训练

```bash
# 修改脚本中的模型路径
vim scripts/train_one_shot_gee.sh

# 运行训练
bash scripts/train_one_shot_gee.sh
```

### 4. 评估结果

```bash
# 运行评估
bash scripts/evaluate_gee.sh
```

## 使用方法

### 基本训练命令

```bash
accelerate launch train_gee.py \
  --model_name Qwen2.5-Math-7B \
  --model_path /path/to/Qwen2.5-Math-7B \
  --train_data dataset/1shot_rlvr/pi1_r1280.parquet \
  --effective_batch 64 \
  --micro_batch_size 2 \
  --lambda_weight 3.0 \
  --max_steps 50 \
  --run_name one_shot_gee
```

### 主要参数说明

- `--lambda_weight`: GEE损失权重（默认3.0）
- `--use_l1`: 使用L1损失而不是L2损失
- `--auto_anneal`: 启用自动退火
- `--balance_dataset`: 平衡数据集中的性别分布
- `--use_test_data`: 使用合成测试数据

### 评估命令

```python
from evaluation.gee_evaluator import GEEEvaluator

# 创建评估器
evaluator = GEEEvaluator("path/to/model")

# 生成测试数据
test_data = evaluator.create_winogender_style_data(num_samples=100)

# 评估偏见
results = evaluator.evaluate_bias(test_data)

# 比较多个模型
model_paths = {
    'Base': 'path/to/base/model',
    'GEE': 'path/to/gee/model'
}
comparison_results = evaluator.compare_models(model_paths, test_data)
```

## 实验结果

### 预期效果

- **熵差距减少**: 70-80%的性别间熵差距减少
- **性能保持**: MMLU/GSM-8K等基准测试性能退化<1%
- **训练效率**: 10步LoRA训练，A100-80G < 3分钟

### 监控指标

- `entropy_gap`: 男女组间熵差距
- `H_male/H_female`: 各组平均熵
- `loss_em`: 熵最小化损失
- `loss_bias`: 偏见惩罚损失

## 文件结构

```
one-shot-em/
├── dataset/
│   └── gee_processor.py          # 数据处理器
├── losses/
│   └── gee_loss.py              # GEE损失函数
├── evaluation/
│   └── gee_evaluator.py         # 评估器
├── scripts/
│   ├── train_one_shot_gee.sh    # 训练脚本
│   ├── evaluate_gee.sh          # 评估脚本
│   └── quick_test_gee.sh        # 快速测试脚本
├── train_gee.py                 # 主训练脚本
├── test_gee_components.py       # 组件测试
└── GEE_README.md               # 本文档
```

## 扩展开发

### 1. 多组扩展
支持更多敏感属性（种族、年龄等）：
```python
# 修改gender_to_label函数
def attribute_to_label(attribute_str: str, attribute_type: str) -> int:
    if attribute_type == 'gender':
        return 0 if attribute_str == 'male' else 1
    elif attribute_type == 'race':
        # 添加种族标签逻辑
        pass
```

### 2. 混合任务
为不同类型的prompt设置不同的权重：
```python
def compute_weighted_gee_loss(H_i, labels, prompt_types):
    # 根据prompt类型调整权重
    weights = torch.where(prompt_types == 'factual', 0.0, 1.0)
    # 应用权重到GEE损失
```

### 3. 高级评估
集成更多偏见评估基准：
- Winogender
- WinoBias
- StereoSet
- CrowS-Pairs

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用gradient_checkpointing
   - 启用CPU offload

2. **数据不平衡**
   - 检查性别检测逻辑
   - 调整balance_dataset参数
   - 手动平衡数据

3. **训练不收敛**
   - 调整lambda_weight
   - 检查学习率
   - 启用自动退火

### 调试技巧

```bash
# 启用详细日志
export TORCH_LOGS=+dynamo

# 检查GPU使用情况
nvidia-smi

# 监控训练过程
tail -f wandb/latest-run/logs/debug.log
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 运行测试
5. 提交Pull Request

## 许可证

本项目基于MIT许可证开源。

## 引用

如果您使用了本项目，请引用：

```bibtex
@misc{gao2025oneshotentropyminimization,
      title={One-shot Entropy Minimization}, 
      author={Zitian Gao and Lynx Chen and Haoming Luo and Joey Zhou and Bryan Dai},
      year={2025},
      eprint={2505.20282},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20282}, 
}
``` 