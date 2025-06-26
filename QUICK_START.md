# 🚀 One-shot GEE 快速启动指南

## 📋 准备工作

### 1. 激活环境
```bash
conda activate one-shot-gee
```

### 2. 验证环境
```bash
# 检查Python版本
python --version  # 应该显示 Python 3.10.x

# 检查关键包
python -c "import torch, pandas, transformers; print('环境正常')"
```

## 🧪 测试流程

### 步骤1: 基础组件测试
```bash
# 运行组件功能测试
python test_gee_components.py
```

**期望输出：**
```
==================================================
测试GEE数据处理器
==================================================
测试性别检测:
  'He is a doctor...' -> male
  'She is a nurse...' -> female
...
✓ GEE数据处理器测试通过
✓ GEE损失函数测试通过
✓ 组件集成测试通过
所有测试通过！✓
```

### 步骤2: 训练逻辑测试
```bash
# 运行训练逻辑测试
python test_gee_training.py
```

**期望输出：**
```
============================================================
测试GEE训练逻辑
============================================================
Step 1 | loss=6.411685 | entropy_gap=0.004735 | H_male=6.409283 | H_female=6.414019
...
✓ GEE训练逻辑测试通过
🎯 准备就绪，可以进行真实模型训练！
```

## 🎯 成功标准

### ✅ 通过标准
- 所有测试显示 "✓ 通过"
- 没有错误或异常
- 熵值在合理范围内 (6.0-7.0)
- 性别标签转换正确 (male=0, female=1)

### ❌ 失败情况
如果遇到以下问题：

**1. 模块导入错误**
```bash
ModuleNotFoundError: No module named 'xxx'
```
解决方案：
```bash
conda activate one-shot-gee
pip install 缺失的包名
```

**2. 路径错误**
```bash
FileNotFoundError: [Errno 2] No such file or directory
```
解决方案：
```bash
# 确保在项目根目录
cd /path/to/one-shot-em
```

**3. CUDA错误**
```bash
CUDA out of memory
```
解决方案：使用CPU版本测试（当前配置已经是CPU版本）

## 🔄 完整测试命令

```bash
# 一键运行所有测试
conda activate one-shot-gee && \
python test_gee_components.py && \
echo "组件测试完成 ✅" && \
python test_gee_training.py && \
echo "训练逻辑测试完成 ✅" && \
echo "所有测试通过！准备就绪 🎉"
```

## 📊 结果解读

### 组件测试结果
- **性别检测**: 应该正确识别male/female/neutral
- **熵计算**: Token熵应该在6-7范围内
- **损失函数**: L2和L1版本应该有明显差异

### 训练测试结果
- **损失收敛**: 损失值应该稳定在6.4左右
- **熵差距**: 应该在0.001-0.1范围内
- **参数影响**: 不同lambda值应该影响偏见损失

## 🎯 下一步

### 如果测试通过 ✅
您可以：
1. 获取Qwen2.5-Math-7B模型
2. 修改 `scripts/train_one_shot_gee.sh` 中的模型路径
3. 运行真实训练：`bash scripts/train_one_shot_gee.sh`

### 如果测试失败 ❌
请：
1. 检查错误信息
2. 参考 `TEST_GUIDE.md` 的故障排除部分
3. 确保环境配置正确

## 📞 需要帮助？

查看详细文档：
- `GEE_README.md` - 完整项目文档
- `TEST_GUIDE.md` - 详细测试指南
- `IMPLEMENTATION_SUMMARY.md` - 实现总结

---

**记住**: 当前测试使用模拟数据和模型，不需要真实的Qwen2.5-Math-7B模型。这些测试验证的是代码逻辑的正确性！ 