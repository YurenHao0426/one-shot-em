#!/usr/bin/env python3
"""
快速启动GenderBench评估
测试训练结果的性别偏见
"""
import os
import sys
from pathlib import Path

def main():
    """主函数"""
    print("🎯 GenderBench评估工具")
    print("=" * 50)
    
    # 检查可用的模型
    potential_models = []
    
    # 检查常见的模型保存路径
    model_dirs = [
        "checkpoints",
        "models", 
        "output",
        "saved_models",
        "."
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    # 检查是否包含模型文件
                    model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
                    if any(os.path.exists(os.path.join(item_path, mf)) for mf in model_files):
                        potential_models.append(item_path)
    
    if potential_models:
        print(f"🔍 发现可能的模型路径:")
        for i, model_path in enumerate(potential_models):
            print(f"   {i+1}. {model_path}")
        
        print(f"\n📝 使用示例:")
        print(f"python genderbench_integration.py \\")
        print(f"  --models {' '.join(potential_models[:2])} \\")
        print(f"  --names baseline_model trained_model \\")
        print(f"  --output genderbench_results")
        
    else:
        print("❌ 未发现模型文件")
        print("📝 请手动指定模型路径:")
        print("python genderbench_integration.py \\")
        print("  --models /path/to/model1 /path/to/model2 \\")
        print("  --names model1_name model2_name \\")
        print("  --output genderbench_results")
    
    print(f"\n🔧 可用选项:")
    print(f"   --models: 模型路径列表（必需）")
    print(f"   --names: 模型名称列表（可选）")
    print(f"   --output: 输出目录（默认: genderbench_results）")
    
    print(f"\n📊 评估内容:")
    print(f"   1. 决策公平性 - 招聘和晋升决策中的性别偏见")
    print(f"   2. 创作代表性 - 创作内容中的性别平衡")
    print(f"   3. 刻板印象推理 - 对性别刻板印象的认同程度")
    
    print(f"\n📈 输出结果:")
    print(f"   - 详细JSON报告")
    print(f"   - CSV对比表格")
    print(f"   - HTML可视化报告")
    
    # 如果有参数，直接运行
    if len(sys.argv) > 1:
        print(f"\n🚀 开始运行评估...")
        from genderbench_integration import main as run_evaluation
        run_evaluation()

if __name__ == "__main__":
    main() 