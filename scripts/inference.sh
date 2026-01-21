#!/bin/bash

# =========================================================
# 功能: 启动推理可视化 (Inference & Visualization)
# 使用方法: 
# bash scripts/inference.sh <GPU_ID> <CONFIG_PATH> <CHECKPOINT_PATH> [OUTPUT_DIR]
# 示例: 
# bash scripts/inference.sh 0 configs/nyu_v2/resnet50_baseline.yaml outputs/nyu_exp/checkpoint_best.pth
# =========================================================

GPU=$1
CONFIG=$2
CHECKPOINT=$3
OUTPUT_DIR=${4:-"outputs/visual_results"} # 默认输出目录

# 检查参数
if [ -z "$GPU" ] || [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: bash scripts/inference.sh <GPU_ID> <CONFIG_PATH> <CHECKPOINT_PATH> [OUTPUT_DIR]"
    exit 1
fi

# 设置环境
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

echo "🎨 Starting Inference Visualization..."
echo "   GPU: $GPU"
echo "   Config: $CONFIG"
echo "   Checkpoint: $CHECKPOINT"
echo "   Output Dir: $OUTPUT_DIR"

# 启动 (单卡直接用 python)
python tools/inference.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR