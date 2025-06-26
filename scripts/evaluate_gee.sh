#!/bin/bash

# GEE评估脚本
# 使用方法: bash scripts/evaluate_gee.sh

echo "开始GEE评估..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径（请根据实际情况修改）
BASE_MODEL_PATH="/volume/pt-train/models/Qwen2.5-Math-7B"
GEE_MODEL_PATH="checkpoints/Qwen2.5-Math-7B/one_shot_gee/final"

# 检查模型路径
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "错误: 基础模型路径不存在: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -d "$GEE_MODEL_PATH" ]; then
    echo "错误: GEE模型路径不存在: $GEE_MODEL_PATH"
    echo "请先运行训练脚本"
    exit 1
fi

echo "基础模型: $BASE_MODEL_PATH"
echo "GEE模型: $GEE_MODEL_PATH"

# 运行评估
python -c "
import sys
sys.path.append('.')
from evaluation.gee_evaluator import GEEEvaluator

# 创建评估器
evaluator = GEEEvaluator('$BASE_MODEL_PATH')

# 生成测试数据
test_data = evaluator.create_winogender_style_data(num_samples=100)

# 定义要比较的模型
model_paths = {
    'Base': '$BASE_MODEL_PATH',
    'GEE': '$GEE_MODEL_PATH'
}

# 比较模型
results = evaluator.compare_models(model_paths, test_data)

# 打印摘要
evaluator.print_summary(results)

# 绘制结果
evaluator.plot_results(results, 'gee_evaluation_results.png')
"

echo "评估完成！结果已保存到 gee_evaluation_results.png" 