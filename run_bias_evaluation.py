#!/usr/bin/env python3
"""
运行偏见评估对比
比较原始模型 vs 纯debiasing模型的偏见减少效果
"""
import argparse
import json
import pandas as pd
from pathlib import Path
import sys
sys.path.append('.')

from evaluation.gee_evaluator import GEEEvaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_model', type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct", help='Original model path')
    parser.add_argument('--debiased_model', type=str, required=True, help='Debiased model path')
    parser.add_argument('--test_data', type=str, default="bias_evaluation_benchmark.json", help='Test data file')
    parser.add_argument('--output_dir', type=str, default="results/bias_comparison", help='Output directory')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max tokens for generation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🎯 开始偏见评估对比...")
    print(f"   原始模型: {args.original_model}")
    print(f"   去偏见模型: {args.debiased_model}")
    print(f"   测试数据: {args.test_data}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"📊 加载了 {len(test_data)} 个测试样本")
    
    # 准备评估
    models_to_compare = {
        'Original': args.original_model,
        'Pure_Debiasing': args.debiased_model
    }
    
    # 初始化评估器（使用原始模型）
    print(f"\n🔧 初始化评估器...")
    evaluator = GEEEvaluator(args.original_model)
    
    # 运行对比评估
    print(f"\n📈 开始模型对比评估...")
    results = evaluator.compare_models(models_to_compare, test_data)
    
    # 保存详细结果
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 详细结果已保存: {results_file}")
    
    # 生成可视化
    print(f"\n📊 生成可视化图表...")
    plot_file = output_dir / 'bias_comparison_plot.png'
    evaluator.plot_results(results, save_path=str(plot_file))
    
    # 打印摘要
    evaluator.print_summary(results)
    
    # 计算改进程度
    original_gap = results['Original']['entropy_gap']
    debiased_gap = results['Pure_Debiasing']['entropy_gap']
    improvement = ((original_gap - debiased_gap) / original_gap) * 100
    
    print(f"\n�� 偏见减少效果:")
    print(f"   原始模型熵差距: {original_gap:.6f}")
    print(f"   去偏见模型熵差距: {debiased_gap:.6f}")
    print(f"   改进程度: {improvement:.1f}%")
    
    # 生成报告摘要
    summary = {
        'evaluation_summary': {
            'original_entropy_gap': original_gap,
            'debiased_entropy_gap': debiased_gap, 
            'improvement_percentage': improvement,
            'test_samples': len(test_data),
            'models_compared': list(models_to_compare.keys())
        },
        'recommendation': 'Excellent' if improvement > 90 else ('Good' if improvement > 70 else ('Moderate' if improvement > 50 else 'Needs Improvement'))
    }
    
    summary_file = output_dir / 'evaluation_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 评估摘要已保存: {summary_file}")
    print(f"🎯 评估完成！查看 {output_dir} 目录获取完整结果")

if __name__ == "__main__":
    main()
