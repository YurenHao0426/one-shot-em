# 🎯 纯偏见减少训练：Benchmark测试与数据完善计划

## 📊 Phase 1: 核心偏见评估

### 1.1 对比评估
```bash
# 训练前后偏见对比
python create_bias_benchmark.py  # 我们需要创建
python run_bias_evaluation.py \
    --original_model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --debiased_model "checkpoints/Qwen2.5-Math-1.5B-Instruct/colab_pure_debiasing/final" \
    --output_dir "results/bias_comparison"
```

**目标指标:**
- 熵差距减少: ✅ 已实现95.3%改善
- 生成质量保持: 待验证
- 推理能力保持: 待验证

## 📚 Phase 2: 真实数据集训练

### 2.1 Numina数学数据集 (460MB+)
```bash
# 使用真实数学推理数据重新训练
accelerate launch train_debiasing.py \
    --model_path "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --train_data "dataset/numina/numina_00.parquet" \
    --run_name "pure_debiasing_numina" \
    --target_gap 0.01 \
    --max_steps 50 \
    --micro_batch_size 2 \
    --effective_batch 8
```

### 2.2 数据预处理改进
```bash
# 增强GEE处理器支持真实数据
python enhance_gee_processor.py  # 需要创建
```

## 🧪 Phase 3: 多维Benchmark测试

### 3.1 代码生成能力
```bash
# HumanEval测试
python code_eval/OpenCodeEval/main.py \
    --model_path checkpoints/.../final \
    --benchmark HumanEval \
    --output_dir results/humaneval

# MBPP测试
python code_eval/OpenCodeEval/main.py \
    --model_path checkpoints/.../final \
    --benchmark mbpp \
    --output_dir results/mbpp
```

### 3.2 数学推理能力
```bash
# GSM8K测试
python Qwen2.5-Eval/evaluation/math_eval.py \
    --model_path checkpoints/.../final \
    --data_path Qwen2.5-Eval/evaluation/data/gsm8k/test.jsonl

# MATH测试  
python Qwen2.5-Eval/evaluation/math_eval.py \
    --model_path checkpoints/.../final \
    --data_path Qwen2.5-Eval/evaluation/data/math/test.jsonl
```

### 3.3 综合能力测试
```bash
# BigCodeBench
python code_eval/OpenCodeEval/main.py \
    --model_path checkpoints/.../final \
    --benchmark BigCodeBench

# LiveCodeBench (最新)
python code_eval/OpenCodeEval/main.py \
    --model_path checkpoints/.../final \
    --benchmark LiveCodeBench
```

## 📈 Phase 4: 评估分析框架

### 4.1 性能保持度分析
- **代码生成**: pass@1, pass@10
- **数学推理**: 准确率, 推理步骤质量  
- **偏见减少**: 熵差距, 响应多样性

### 4.2 详细对比报告
```
原始模型 vs 纯Debiasing模型:
┌─────────────────┬──────────┬──────────┬────────────┐
│     指标        │  原始    │ Debiasing│   变化     │
├─────────────────┼──────────┼──────────┼────────────┤
│ 熵差距          │  33.2%   │   1.6%   │ -95.3% ✅  │
│ HumanEval pass@1│    ?     │    ?     │     ?      │
│ GSM8K 准确率    │    ?     │    ?     │     ?      │
│ MATH 准确率     │    ?     │    ?     │     ?      │
│ 生成流畅度      │    ?     │    ?     │     ?      │
└─────────────────┴──────────┴──────────┴────────────┘
```

## 🔄 Phase 5: 数据来源扩展

### 5.1 现有数据资产
- ✅ **Numina**: 460MB+ 数学推理数据  
- ✅ **1shot_rlvr**: 强化学习训练数据
- ✅ **合成数据**: 已验证的测试数据

### 5.2 新增数据源建议
```bash
# WinoGender风格偏见测试集
wget https://github.com/rudinger/winogender-schemas/raw/master/data/...

# CodeBLEU性别平衡代码数据
# Math Word Problems性别平衡数学问题
```

### 5.3 数据质量保证
- 性别标注准确性验证
- 数据平衡性检查  
- 领域覆盖度分析

## 🚀 实施时间线

### Week 1: 基础评估
- [ ] 创建偏见评估脚本
- [ ] 在现有模型上运行完整benchmark  
- [ ] 建立评估基线

### Week 2: 真实数据训练
- [ ] 增强数据处理器支持Numina
- [ ] 在真实数据上训练纯debiasing模型
- [ ] 初步效果验证

### Week 3: 全面评估
- [ ] 所有benchmark测试
- [ ] 性能对比分析
- [ ] 结果可视化

### Week 4: 优化与扩展  
- [ ] 根据结果调优超参数
- [ ] 扩展到更大模型
- [ ] 撰写技术报告

## 🎯 成功标准

### 核心目标
- ✅ **偏见减少**: 熵差距 < 2%
- 🎯 **性能保持**: 主要benchmark性能下降 < 5%
- 🎯 **训练效率**: 训练时间 < 原GEE方法50%

### 评估指标权重
- 偏见减少效果: 40%
- 代码生成能力: 25%  
- 数学推理能力: 25%
- 训练效率: 10%

## 💡 下一步行动

1. **立即可做**: 创建偏见评估脚本
2. **本周内**: 在真实数据上训练  
3. **本月内**: 完成全面benchmark评估
4. **长期目标**: 建立标准化debiasing评估流程
