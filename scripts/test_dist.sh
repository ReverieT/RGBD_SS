#!/bin/bash

# =========================================================
# 功能: 启动分布式测试 (Evaluation)
# 使用方法: 
# bash scripts/test_dist.sh <GPU_IDS> <CONFIG_PATH> <CHECKPOINT_PATH>
# 示例: 
# bash scripts/test_dist.sh 0,1 configs/nyu_v2/resnet50_baseline.yaml outputs/nyu_exp/checkpoint_best.pth
# =========================================================

GPUS=$1
CONFIG=$2
CHECKPOINT=$3

# 检查参数
if [ -z "$GPUS" ] || [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: bash scripts/test_dist.sh <GPU_IDS> <CONFIG_PATH> <CHECKPOINT_PATH>"
    exit 1
fi

# 设置环境
export CUDA_VISIBLE_DEVICES=$GPUS
export MASTER_PORT=${MASTER_PORT:-29501} # 默认测试端口用 29501，防止和训练冲突
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 计算 GPU 数量
GPU_COUNT=$(echo $GPUS | tr ',' '\n' | wc -l)

echo "🚀 Starting Distributed Testing..."
echo "   GPUs: $GPUS (Count: $GPU_COUNT)"
echo "   Config: $CONFIG"
echo "   Checkpoint: $CHECKPOINT"

# 启动
torchrun \
    --nproc_per_node=$GPU_COUNT \
    --master_port=$MASTER_PORT \
    tools/test.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT
