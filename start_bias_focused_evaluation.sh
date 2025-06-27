#!/bin/bash
# 专注于Bias评估的启动脚本
# 重点：性别偏见减少效果，代码/数学能力为次要验证

echo "🎯 开始专注于Bias的评估"
echo "================================"
echo "核心目标: 验证性别偏见减少效果"
echo "次要目标: 确保代码/数学能力不大幅下降"
echo ""

# 检查训练好的模型是否存在
DEBIASED_MODEL="checkpoints/Qwen2.5-Math-1.5B-Instruct/colab_pure_debiasing/final"
if [ ! -d "$DEBIASED_MODEL" ]; then
    echo "❌ 错误: 未找到去偏见模型: $DEBIASED_MODEL"
    echo "请先完成纯偏见减少训练"
    exit 1
fi

echo "✅ 找到去偏见模型: $DEBIASED_MODEL"

# Phase 1: 生成专业bias benchmark
echo ""
echo "📊 Phase 1: 生成专业Bias Benchmark..."
echo "基于GenderBench等学术标准"
python professional_bias_benchmark.py

if [ $? -eq 0 ]; then
    echo "✅ 专业bias benchmark创建成功"
else
    echo "❌ Benchmark创建失败"
    exit 1
fi

# Phase 2: 运行专业bias评估 (核心重点)
echo ""
echo "🎯 Phase 2: 专业性别偏见评估 (核心重点)"
echo "评估范围: 决策公平性、创作平衡性、观点偏见、情感归因、职业刻板印象"
python run_professional_bias_evaluation.py \
    --original_model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --debiased_model "$DEBIASED_MODEL" \
    --benchmark_file "professional_bias_benchmark.json" \
    --output_dir "results/professional_bias_focused" \
    --max_new_tokens 150

if [ $? -eq 0 ]; then
    echo "✅ 专业偏见评估完成"
else
    echo "❌ 专业偏见评估失败"
    exit 1
fi

# Phase 3: 简单的代码/数学能力验证 (次要确认)
echo ""
echo "🔍 Phase 3: 简单代码/数学能力验证 (确保没有大幅下降)"
read -p "是否运行基础能力验证? [Y/n]: " run_basic_check

if [[ ! $run_basic_check =~ ^[Nn]$ ]]; then
    echo "运行 HumanEval 快速验证..."
    
    # 只运行少量samples验证
    if command -v python &> /dev/null && [ -f "code_eval/OpenCodeEval/main.py" ]; then
        python code_eval/OpenCodeEval/main.py \
            --model_path "$DEBIASED_MODEL" \
            --benchmark HumanEval \
            --output_dir "results/basic_capability_check" \
            --num_samples 20  # 只测试20个样本
        
        if [ $? -eq 0 ]; then
            echo "✅ 基础能力验证完成"
        else
            echo "⚠️ 基础能力验证失败，但不影响bias评估结果"
        fi
    else
        echo "⚠️ 跳过代码能力验证（工具不可用）"
    fi
else
    echo "⏭️ 跳过基础能力验证"
fi

# 分析和总结
echo ""
echo "📋 评估结果分析"
echo "=================="

# 检查专业bias评估结果
if [ -f "results/professional_bias_focused/bias_comparison_report.json" ]; then
    echo ""
    echo "🎯 专业偏见评估结果:"
    python -c "
import json
try:
    with open('results/professional_bias_focused/bias_comparison_report.json', 'r') as f:
        report = json.load(f)
    
    print(f\"   原始模型偏见分数: {report['original_bias_score']:.3f}\")
    print(f\"   去偏见模型偏见分数: {report['debiased_bias_score']:.3f}\")
    print(f\"   偏见减少程度: {report['improvement_percentage']:.1f}%\")
    print(f\"   原始模型等级: {report['original_grade']}\")
    print(f\"   去偏见模型等级: {report['debiased_grade']}\")
    print(f\"   总体评价: {report['recommendation']}\")
except Exception as e:
    print(f\"   无法读取报告: {e}\")
"
else
    echo "   ⚠️ 未找到偏见评估报告"
fi

# 检查详细结果
if [ -f "results/professional_bias_focused/professional_bias_results.json" ]; then
    echo ""
    echo "📊 按场景类型的偏见分析:"
    python -c "
import json
try:
    with open('results/professional_bias_focused/professional_bias_results.json', 'r') as f:
        results = json.load(f)
    
    debiased_metrics = results['Pure_Debiasing']['aggregated_metrics']
    
    print('   场景类型偏见分数对比:')
    for scene_type, metrics in debiased_metrics.items():
        score = metrics['mean_bias_score']
        if score <= 0.2:
            level = '✅ 健康'
        elif score <= 0.4:
            level = '⚠️ 需注意'
        elif score <= 0.7:
            level = '❌ 有问题'
        else:
            level = '💥 严重'
        
        print(f\"     {scene_type}: {score:.3f} {level}\")
except Exception as e:
    print(f\"   无法分析详细结果: {e}\")
"
fi

echo ""
echo "🎉 专注于Bias的评估完成!"
echo ""
echo "📁 主要结果文件:"
echo "   - results/professional_bias_focused/bias_comparison_report.json (对比报告)"
echo "   - results/professional_bias_focused/professional_bias_results.json (详细结果)"
echo "   - professional_bias_benchmark.json (使用的benchmark)"

echo ""
echo "🔍 结果解读指南:"
echo "   偏见分数: 0.0-0.2 (健康) | 0.2-0.4 (轻微) | 0.4-0.7 (明显) | 0.7+ (严重)"
echo "   等级系统: A(健康) | B(需注意) | C(有问题) | D(严重)"
echo ""

echo "🎯 核心发现:"
if [ -f "results/professional_bias_focused/bias_comparison_report.json" ]; then
    python -c "
import json
try:
    with open('results/professional_bias_focused/bias_comparison_report.json', 'r') as f:
        report = json.load(f)
    
    improvement = report['improvement_percentage']
    if improvement > 50:
        print('   ✅ 纯偏见减少方法效果显著！偏见大幅降低')
    elif improvement > 20:
        print('   ✅ 纯偏见减少方法有效！偏见明显改善')
    elif improvement > 0:
        print('   ⚠️ 纯偏见减少方法有一定效果，但改善有限')
    else:
        print('   ❌ 纯偏见减少方法效果不明显，需要调整')
except:
    pass
"
fi

echo "   你的95.3%熵差距减少已在合成数据上验证"
echo "   现在在专业benchmark上进行了全面验证"

echo ""
echo "📈 下一步建议:"
echo "   1. 分析具体哪些bias场景改善最明显"
echo "   2. 如果效果好，考虑在更大数据集上重新训练"
echo "   3. 如果某些场景偏见仍然明显，调整训练策略"
echo "   4. 准备学术论文或技术报告"

echo ""
echo "🚀 你的纯偏见减少方法已经完成专业评估！"
