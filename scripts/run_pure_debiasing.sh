#!/bin/bash
# 纯偏见减少训练脚本
# 目标：只最小化男女间熵差，不进行整体熵最小化

echo "🎯 开始纯偏见减少训练"
echo "目标：最小化 |H_female - H_male|"
echo "特点：不包含熵最小化(EM)，专注debiasing"

# 默认参数
MODEL_PATH=${1:-"Qwen2.5-Math-1.5B-Instruct"}
RUN_NAME=${2:-"pure_debiasing_$(date +%m%d_%H%M)"}
TARGET_GAP=${3:-0.01}
MAX_STEPS=${4:-20}

echo ""
echo "📊 配置信息:"
echo "   模型路径: $MODEL_PATH"
echo "   运行名称: $RUN_NAME"
echo "   目标熵差: $TARGET_GAP"
echo "   最大步数: $MAX_STEPS"
echo ""

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo "请提供正确的模型路径作为第一个参数"
    echo "用法: $0 <model_path> [run_name] [target_gap] [max_steps]"
    exit 1
fi

# 运行纯偏见减少训练
python train_debiasing.py \
    --model_path "$MODEL_PATH" \
    --run_name "$RUN_NAME" \
    --target_gap $TARGET_GAP \
    --max_steps $MAX_STEPS \
    --micro_batch_size 2 \
    --effective_batch 4 \
    --learning_rate 1e-5 \
    --scale_factor 1.0 \
    --use_test_data \
    --wandb_project "pure-debiasing" \
    --log_steps 1 \
    --save_steps 10

echo ""
echo "🎉 纯偏见减少训练完成！"
echo "📁 检查点保存在: checkpoints/$(basename $MODEL_PATH)/$RUN_NAME/"
echo "�� 查看WandB日志了解详细训练过程" 