#!/bin/bash

# GEE快速测试脚本
# 使用合成数据进行快速验证，无需真实模型

echo "开始GEE快速测试..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

echo "运行组件测试..."
python test_gee_components.py

if [ $? -eq 0 ]; then
    echo "✓ 组件测试通过"
else
    echo "✗ 组件测试失败"
    exit 1
fi

echo ""
echo "运行快速训练测试（使用合成数据）..."
accelerate launch train_gee.py \
  --model_name Qwen2.5-Math-7B \
  --model_path /volume/pt-train/models/Qwen2.5-Math-7B \
  --use_test_data \
  --effective_batch 8 \
  --micro_batch_size 2 \
  --max_steps 5 \
  --lambda_weight 3.0 \
  --log_steps 1 \
  --save_steps 5 \
  --run_name quick_test_gee \
  --wandb_project one-shot-gee

if [ $? -eq 0 ]; then
    echo "✓ 快速训练测试通过"
else
    echo "✗ 快速训练测试失败"
    exit 1
fi

echo ""
echo "所有快速测试通过！✓"
echo "现在可以运行完整的训练和评估脚本" 