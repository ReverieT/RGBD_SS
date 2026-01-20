#!/bin/bash

# =========================================================
# 使用方法: 
# bash scripts/train_dist.sh <GPU_IDS> <CONFIG_PATH>
# 示例: 
# bash scripts/train_dist.sh 0,1 configs/nyu_v2/resnet50_baseline.yaml
# =========================================================

# 1. 获取参数
GPUS=$1
CONFIG=$2

# 检查参数是否为空
if [ -z "$GPUS" ] || [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/train_dist.sh <GPU_IDS> <CONFIG_PATH>"
    echo "Example: bash scripts/train_dist.sh 0,1 configs/nyu_v2/resnet50_baseline.yaml"
    exit 1
fi

# 2. 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPUS
# 设置主端口 (随机生成一个端口防止冲突，或者固定)
export MASTER_PORT=${MASTER_PORT:-29500}
# 将当前目录添加到 PYTHONPATH，确保能 import seg_core
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 计算 GPU 数量 (根据逗号分隔符计算)
GPU_COUNT=$(echo $GPUS | tr ',' '\n' | wc -l)

echo "🚀 Launching training on GPUs: $GPUS (Count: $GPU_COUNT)"
echo "📄 Using Config: $CONFIG"

# 3. 启动 torchrun
# --nproc_per_node: 启动的进程数，通常等于显卡数
torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=$MASTER_PORT \
    tools/train.py \
    --config $CONFIG
