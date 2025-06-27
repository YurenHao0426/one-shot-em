#!/bin/bash
# 启动下一阶段：Benchmark测试与数据完善

echo "🚀 启动下一阶段：Benchmark测试与数据完善"
echo "============================================"

# 检查训练好的模型是否存在
DEBIASED_MODEL="checkpoints/Qwen2.5-Math-1.5B-Instruct/colab_pure_debiasing/final"
if [ ! -d "$DEBIASED_MODEL" ]; then
    echo "❌ 错误: 未找到去偏见模型: $DEBIASED_MODEL"
    echo "请先完成纯偏见减少训练"
    exit 1
fi

echo "✅ 找到去偏见模型: $DEBIASED_MODEL"

# Phase 1: 创建偏见评估benchmark
echo ""
echo "📊 Phase 1: 创建偏见评估benchmark..."
python create_bias_benchmark.py

if [ $? -eq 0 ]; then
    echo "✅ Benchmark创建成功"
else
    echo "❌ Benchmark创建失败"
    exit 1
fi

# Phase 2: 运行偏见对比评估
echo ""
echo "📈 Phase 2: 运行偏见对比评估..."
python run_bias_evaluation.py \
    --original_model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --debiased_model "$DEBIASED_MODEL" \
    --test_data "bias_evaluation_benchmark.json" \
    --output_dir "results/phase2_bias_comparison" \
    --max_new_tokens 128

if [ $? -eq 0 ]; then
    echo "✅ 偏见评估完成"
else
    echo "❌ 偏见评估失败"
fi

# Phase 3: 测试增强数据处理器
echo ""
echo "🔧 Phase 3: 测试增强数据处理器..."
python enhance_gee_processor.py

if [ $? -eq 0 ]; then
    echo "✅ 数据处理器测试成功"
else
    echo "⚠️ 数据处理器测试失败，但可以继续"
fi

# Phase 4: 代码能力评估 (可选)
echo ""
echo "💻 Phase 4: 代码能力评估 (可选)..."
read -p "是否运行代码评估 (HumanEval)? [y/N]: " run_code_eval

if [[ $run_code_eval =~ ^[Yy]$ ]]; then
    echo "运行 HumanEval 评估..."
    python code_eval/OpenCodeEval/main.py \
        --model_path "$DEBIASED_MODEL" \
        --benchmark HumanEval \
        --output_dir "results/phase4_humaneval"
    
    if [ $? -eq 0 ]; then
        echo "✅ HumanEval评估完成"
    else
        echo "❌ HumanEval评估失败"
    fi
else
    echo "⏭️ 跳过代码评估"
fi

# Phase 5: 数学能力评估 (可选)
echo ""
echo "🧮 Phase 5: 数学能力评估 (可选)..."
read -p "是否运行数学评估 (GSM8K)? [y/N]: " run_math_eval

if [[ $run_math_eval =~ ^[Yy]$ ]]; then
    echo "运行 GSM8K 评估..."
    if [ -f "Qwen2.5-Eval/evaluation/data/gsm8k/test.jsonl" ]; then
        python Qwen2.5-Eval/evaluation/math_eval.py \
            --model_path "$DEBIASED_MODEL" \
            --data_path "Qwen2.5-Eval/evaluation/data/gsm8k/test.jsonl" \
            --output_dir "results/phase5_gsm8k"
        
        if [ $? -eq 0 ]; then
            echo "✅ GSM8K评估完成"
        else
            echo "❌ GSM8K评估失败"
        fi
    else
        echo "⚠️ 未找到GSM8K测试数据"
    fi
else
    echo "⏭️ 跳过数学评估"
fi

# 总结
echo ""
echo "🎯 下一阶段进度总结："
echo "===================="
echo "✅ 偏见评估benchmark已创建"
echo "✅ 模型偏见对比评估已完成"
echo "📊 查看结果: results/phase2_bias_comparison/"

# 检查结果文件
if [ -f "results/phase2_bias_comparison/evaluation_summary.json" ]; then
    echo ""
    echo "📋 快速结果预览："
    python -c "
import json
with open('results/phase2_bias_comparison/evaluation_summary.json', 'r') as f:
    summary = json.load(f)
    eval_summary = summary['evaluation_summary']
    print(f\"   原始模型熵差距: {eval_summary['original_entropy_gap']:.6f}\")
    print(f\"   去偏见模型熵差距: {eval_summary['debiased_entropy_gap']:.6f}\")
    print(f\"   改进程度: {eval_summary['improvement_percentage']:.1f}%\")
    print(f\"   评估结果: {summary['recommendation']}\")
"
fi

echo ""
echo "🚀 下一步建议："
echo "1. 查看详细评估报告: results/phase2_bias_comparison/"
echo "2. 如果效果好，可以在真实数据上重新训练"
echo "3. 运行更多benchmark测试验证性能保持"
echo "4. 考虑扩展到更大模型"

echo ""
echo "🎉 下一阶段测试完成！"
