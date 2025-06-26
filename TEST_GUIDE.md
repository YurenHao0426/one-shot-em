# One-shot GEE 测试流程指南

## 环境准备 ✓

### 1. 创建conda环境
```bash
conda create -n one-shot-gee python=3.10 -y
conda activate one-shot-gee
```

### 2. 安装依赖
```bash
# 基础依赖已安装
pip install pandas numpy matplotlib seaborn transformers accelerate
```

## 测试阶段

### 阶段1: 组件功能测试 ✓

运行基础组件测试：
```bash
conda activate one-shot-gee
python test_gee_components.py
```

**测试结果：**
- ✅ GEE数据处理器测试通过
  - 性别检测功能正常
  - 测试数据生成正常
- ✅ GEE损失函数测试通过
  - Token熵计算正常
  - 组熵计算正常
  - L2和L1损失函数正常
- ⚠️ GEE评估器测试跳过（需要实际模型）
- ✅ 组件集成测试通过

### 阶段2: 训练功能测试

#### 2.1 快速训练测试（使用合成数据）

```bash
conda activate one-shot-gee

# 测试训练脚本（使用合成数据，无需真实模型）
python train_gee.py \
  --use_test_data \
  --effective_batch 8 \
  --micro_batch_size 2 \
  --max_steps 3 \
  --lambda_weight 3.0 \
  --log_steps 1 \
  --run_name quick_test \
  --model_name test_model \
  --model_path dummy_path
```

#### 2.2 真实数据测试（需要实际模型）

如果您有Qwen2.5-Math-7B模型：

1. **修改模型路径**：
```bash
vim scripts/train_one_shot_gee.sh
# 修改MODEL_PATH为您的实际模型路径
```

2. **运行完整训练**：
```bash
bash scripts/train_one_shot_gee.sh
```

### 阶段3: 评估功能测试

#### 3.1 无模型评估测试
```bash
# 测试评估器的数据生成功能
python -c "
import sys
sys.path.append('.')
from evaluation.gee_evaluator import GEEEvaluator

# 只测试数据生成，不加载模型
class MockEvaluator:
    def create_winogender_style_data(self, num_samples=10):
        from evaluation.gee_evaluator import GEEEvaluator
        evaluator = GEEEvaluator.__new__(GEEEvaluator)
        return evaluator.create_winogender_style_data(num_samples)

evaluator = MockEvaluator()
test_data = evaluator.create_winogender_style_data(20)
print(f'生成测试数据: {len(test_data)} 条')
for i, item in enumerate(test_data[:3]):
    print(f'样本 {i+1}: {item[\"gender\"]} - {item[\"prompt\"]}')
"
```

#### 3.2 完整评估测试（需要训练后的模型）
```bash
# 确保已有训练完成的模型后运行
bash scripts/evaluate_gee.sh
```

## 效果验证指标

### 核心指标

1. **熵差距减少** (`entropy_gap`)
   - 目标：相比基线模型减少70-80%
   - 计算：`|H_female - H_male|`

2. **训练稳定性**
   - 损失函数收敛
   - 梯度不爆炸/消失

3. **性能保持**
   - 数学推理能力不显著退化
   - 生成质量保持

### 监控指标

训练过程中关注的指标：
```
Step X | loss=6.4005 | entropy_gap=0.0161 | H_male=6.3921 | H_female=6.4082
```

- `loss`: 总损失（熵最小化损失 + GEE惩罚损失）
- `entropy_gap`: 男女组间熵差距（越小越好）
- `H_male/H_female`: 各组平均熵

## 问题排查

### 常见错误及解决方案

1. **模块导入错误**
   ```bash
   # 确保在正确的conda环境中
   conda activate one-shot-gee
   # 确保在项目根目录
   cd /path/to/one-shot-em
   ```

2. **CUDA相关错误**
   ```bash
   # 如果没有GPU，确保使用CPU版本的PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **数据路径错误**
   ```bash
   # 检查数据文件是否存在
   ls -la dataset/1shot_rlvr/pi1_r1280.parquet
   ```

4. **模型路径错误**
   ```bash
   # 修改脚本中的模型路径
   vim scripts/train_one_shot_gee.sh
   ```

## 下一步操作

### 已完成 ✅
- [x] 创建conda环境
- [x] 安装基础依赖
- [x] 组件功能测试
- [x] 核心功能验证

### 待完成 📋
- [ ] 获取或下载Qwen2.5-Math-7B模型
- [ ] 运行真实数据训练测试
- [ ] 完整的偏见评估测试
- [ ] 性能基准测试

### 推荐测试顺序

1. **立即可做**：
   ```bash
   # 测试训练脚本逻辑（使用合成数据）
   conda activate one-shot-gee
   python train_gee.py --use_test_data --max_steps 3
   ```

2. **获得模型后**：
   ```bash
   # 小规模真实训练
   bash scripts/train_one_shot_gee.sh
   ```

3. **训练完成后**：
   ```bash
   # 评估偏见减少效果
   bash scripts/evaluate_gee.sh
   ```

## 成功标准

✅ **基础功能正常**
- 所有组件测试通过
- 损失函数计算正确
- 训练脚本可以运行

🎯 **训练效果良好**
- entropy_gap在训练过程中逐步减少
- 总损失稳定收敛
- 模型生成质量保持

📊 **评估结果理想**
- 相比基线模型，熵差距减少70%+
- 数学推理性能退化<1%
- 生成文本质量无明显下降 