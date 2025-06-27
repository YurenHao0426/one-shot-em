# 纯偏见减少(Pure Debiasing)使用指南

## 🎯 概述

纯偏见减少训练专注于**最小化男女间的熵差**，不包含整体熵最小化(EM)。这使得训练目标更加明确，计算更加高效。

### 核心目标
```
原GEE损失: L = H̄ + λ * (H_female - H_male)²
          ↓
纯Debiasing: L = (H_female - H_male)²
```

**关键优势:**
- ✅ 目标更明确：只关注性别偏见
- ✅ 计算更简单：去除熵最小化项
- ✅ 训练更稳定：单一优化目标
- ✅ 效果更直接：熵差距直接下降

## 🚀 快速开始

### 1. 基础运行
```bash
# 使用默认参数
./scripts/run_pure_debiasing.sh /path/to/your/model

# 自定义参数
./scripts/run_pure_debiasing.sh /path/to/model my_run_name 0.005 30
```

### 2. 手动运行
```bash
python train_debiasing.py \
    --model_path /path/to/model \
    --run_name pure_debiasing_test \
    --target_gap 0.01 \
    --max_steps 20 \
    --micro_batch_size 2 \
    --effective_batch 4 \
    --learning_rate 1e-5 \
    --use_test_data
```

## 📊 核心组件

### 1. 损失函数 (`losses/debiasing_loss.py`)
```python
class DebiasingLoss:
    def __init__(self, use_l1=False, scale_factor=1.0):
        """
        use_l1: False=L2损失, True=L1损失
        scale_factor: 损失缩放因子
        """
```

**损失计算:**
- L2版本: `(H_female - H_male)²`
- L1版本: `|H_female - H_male|`

### 2. 训练脚本 (`train_debiasing.py`)
专门的纯debiasing训练，包含：
- 智能批次平衡
- 早停机制(达到目标熵差)
- 实时监控和可视化

### 3. 测试验证
```bash
# 数学逻辑测试
python test_debiasing_math.py

# 完整功能测试(需要PyTorch)
python test_debiasing_loss.py
```

## 🔧 参数配置

### 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target_gap` | 0.01 | 目标熵差距，达到后早停 |
| `--scale_factor` | 1.0 | 损失缩放因子 |
| `--use_l1` | False | 使用L1损失替代L2 |
| `--learning_rate` | 1e-5 | 学习率(建议较低) |
| `--micro_batch_size` | 2 | 必须≥2保证性别平衡 |

### 训练建议
- **学习率**: 1e-5 到 5e-5 (比普通训练更低)
- **批次大小**: 确保每批至少1男1女
- **目标熵差**: 0.005-0.02 (根据应用需求调整)
- **训练步数**: 通常10-50步即可看到效果

## 📈 监控指标

训练过程中关键指标：
```
📉 Step 1 | loss=0.160000 | gap=0.400000 | H_male=0.4500 | H_female=0.8500
📉 Step 2 | loss=0.040000 | gap=0.200000 | H_male=0.5000 | H_female=0.7000
📉 Step 3 | loss=0.010000 | gap=0.100000 | H_male=0.5500 | H_female=0.6500
```

**理想训练轨迹:**
- 损失持续下降
- 熵差距(`gap`)持续缩小
- `H_male`和`H_female`趋于相等

## 🎯 预期效果

### 训练前
```
H_male=0.25, H_female=0.95, gap=0.70 (严重偏见 💥)
```

### 训练后
```
H_male=0.58, H_female=0.60, gap=0.02 (轻微偏见 ⚠️)
```

### 理想状态
```
H_male=0.60, H_female=0.60, gap=0.00 (无偏见 ✅)
```

## 🔄 与原GEE的对比

| 方面 | 原GEE | 纯Debiasing |
|------|-------|-------------|
| 损失函数 | `H̄ + λ*(H_f-H_m)²` | `(H_f-H_m)²` |
| 优化目标 | 熵最小化+偏见减少 | 仅偏见减少 |
| 参数数量 | 需要调节λ权重 | 无需权重调节 |
| 训练复杂度 | 高(双目标平衡) | 低(单目标) |
| 收敛速度 | 较慢 | 较快 |
| 偏见减少效果 | 可能被EM目标稀释 | 直接且强烈 |

## 💡 最佳实践

### 1. 数据准备
```python
# 确保数据性别平衡
male_samples = [s for s in data if s['gender'] == 'male']
female_samples = [s for s in data if s['gender'] == 'female']
print(f"男女比例: {len(male_samples)}:{len(female_samples)}")
```

### 2. 超参调优
```bash
# 保守设置(稳定但慢)
--learning_rate 5e-6 --target_gap 0.005

# 激进设置(快速但可能不稳定)
--learning_rate 2e-5 --target_gap 0.02
```

### 3. 监控要点
- 关注`entropy_gap`是否持续下降
- 检查批次平衡性(无警告信息)
- 观察损失收敛曲线

### 4. 故障排除
```bash
# 如果批次不平衡
--micro_batch_size 4  # 增加批次大小

# 如果训练不稳定
--learning_rate 1e-6  # 降低学习率

# 如果收敛太慢
--scale_factor 2.0    # 增加损失权重
```

## 📁 文件结构

```
losses/
├── debiasing_loss.py        # 纯debiasing损失函数
└── gee_loss.py             # 原GEE损失(对比用)

train_debiasing.py          # 纯debiasing训练脚本
test_debiasing_math.py      # 数学逻辑测试
scripts/
└── run_pure_debiasing.sh   # 便捷运行脚本
```

## 🎉 总结

纯偏见减少方法提供了一个**更专注、更高效**的debiasing解决方案。通过去除熵最小化的干扰，训练过程更加直接，效果更加明显。

**适用场景:**
- 只关心减少性别偏见，不需要整体性能优化
- 需要快速原型验证debiasing效果
- 资源有限的环境下进行偏见减少

**下一步:** 根据你的具体需求调整参数，开始纯debiasing训练！ 