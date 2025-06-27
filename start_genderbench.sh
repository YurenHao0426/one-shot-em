#!/bin/bash
# GenderBench评估启动脚本

echo "🎯 GenderBench性别偏见评估工具"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装，请先安装Python"
    exit 1
fi

# 检查必要的包
echo "📦 检查依赖包..."
python -c "import pandas, numpy" 2>/dev/null || {
    echo "⚠️  缺少依赖包，正在安装..."
    pip install pandas numpy
}

echo "✅ 环境检查完成"

# 显示使用选项
echo ""
echo "🔧 使用选项:"
echo "1. 运行演示 (使用模拟模型)"
echo "2. 检查可用模型"
echo "3. 运行实际评估"
echo "4. 查看帮助文档"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "🚀 运行GenderBench演示..."
        python demo_genderbench.py
        ;;
    2)
        echo "🔍 检查可用模型..."
        python run_genderbench_evaluation.py
        ;;
    3)
        echo "📝 运行实际评估需要指定模型路径"
        echo "示例:"
        echo "python genderbench_integration.py \\"
        echo "  --models /path/to/model1 /path/to/model2 \\"
        echo "  --names baseline trained \\"
        echo "  --output results"
        echo ""
        read -p "是否继续查看详细帮助? (y/n): " continue_help
        if [[ $continue_help == "y" || $continue_help == "Y" ]]; then
            python genderbench_integration.py --help
        fi
        ;;
    4)
        echo "📖 查看帮助文档..."
        if [[ -f "GENDERBENCH_GUIDE.md" ]]; then
            echo "详细文档: GENDERBENCH_GUIDE.md"
            echo "主要功能:"
            echo "• 决策公平性评估"
            echo "• 创作代表性分析"  
            echo "• 刻板印象推理测试"
            echo ""
            echo "快速开始:"
            echo "python demo_genderbench.py  # 运行演示"
        else
            echo "❌ 帮助文档未找到"
        fi
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 感谢使用GenderBench评估工具!"
echo "📧 如有问题，请查看项目文档或联系开发者" 