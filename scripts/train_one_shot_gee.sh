#!/bin/bash

# One-shot GEE训练脚本
# 使用方法: bash scripts/train_one_shot_gee.sh

echo "开始One-shot GEE训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export NCCL_TIMEOUT=2700
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2700

# 训练参数
MODEL_NAME="Qwen2.5-Math-7B"
MODEL_PATH="/volume/pt-train/models/Qwen2.5-Math-7B"  # 请根据实际路径修改
TRAIN_DATA="dataset/1shot_rlvr/pi1_r1280.parquet"
RUN_NAME="one_shot_gee_lambda3"
WANDB_PROJECT="one-shot-gee"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请修改脚本中的MODEL_PATH变量"
    exit 1
fi

# 检查训练数据
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据文件不存在: $TRAIN_DATA"
    echo "请检查数据文件路径"
    exit 1
fi

echo "模型路径: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA"
echo "运行名称: $RUN_NAME"

# 开始训练
accelerate launch train_gee.py \
  --model_name $MODEL_NAME \
  --model_path $MODEL_PATH \
  --train_data $TRAIN_DATA \
  --effective_batch 64 \
  --micro_batch_size 2 \
  --temperature 0.5 \
  --learning_rate 2e-5 \
  --max_steps 50 \
  --lambda_weight 3.0 \
  --auto_anneal \
  --balance_dataset \
  --log_steps 1 \
  --save_steps 1 \
  --run_name $RUN_NAME \
  --wandb_project $WANDB_PROJECT

echo "训练完成！" 