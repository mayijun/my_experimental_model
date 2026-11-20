#!/usr/bin/env bash
# 在 A100 (40GB) 上运行 0.3B 规模的 Flow Matching + JEPA 预训练。
# 基于 HuggingFace Accelerate + datasets（见 pretrain.py）。
#
# 用法：
#   bash run_pretrain_a100.sh            # 前台运行，日志同时写入 logs/
#   nohup bash run_pretrain_a100.sh &    # 后台运行，适合多天训练
#   多卡：把最后一行改成 `accelerate launch pretrain.py ...`（先 accelerate config）
#
# 依赖：项目根目录已有 .venv（含 PyTorch + CUDA + accelerate + datasets）。
set -euo pipefail

# ---------- 数据 ----------
DATASET_NAME=wikitext
DATASET_CONFIG=wikitext-103-raw-v1
SEQ_LEN=256
# ---------- 模型 ----------
DIM=1024
N_HEADS=16
ENCODER_LAYERS=4      # JEPA 编码器层数（online + target 各一份）。
FLOW_LAYERS=12        # Flow 主干层数。
QUANTIZER=fsq         # none | fsq | vq | rvq
# ---------- 训练 ----------
BATCH_SIZE=16
GRAD_ACCUM=1
LR=2e-3
STAGE1_STEPS=2000     # 表征 + 量化/解码器预训练步数。
STAGE2_STEPS=5000     # Flow Matching 训练步数。
# ---------- checkpoint / 日志 ----------
CHECKPOINT_DIR=checkpoints
SAVE_EVERY=500
LOG_EVERY=10
RESUME=0              # 置 1 则从 CHECKPOINT_DIR 续训。
# ------------------------------

cd "$(dirname "$0")"

# 单卡选择（多卡机器上按需取消注释并填 GPU 编号）。
# export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 减少 PyTorch 显存碎片，降低大模型 OOM 概率。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 解释器：优先使用项目自带的 venv。
PYTHON="${PYTHON:-.venv/bin/python}"

LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pretrain_$(date +%Y%m%d_%H%M%S).log"

echo "启动预训练：dim=$DIM heads=$N_HEADS enc=$ENCODER_LAYERS flow=$FLOW_LAYERS seq=$SEQ_LEN batch=$BATCH_SIZE"
echo "数据集：$DATASET_NAME/$DATASET_CONFIG"
echo "日志文件：$LOG_FILE"

RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume"
fi

"$PYTHON" pretrain.py \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --seq-len "$SEQ_LEN" \
    --dim "$DIM" \
    --n-heads "$N_HEADS" \
    --encoder-layers "$ENCODER_LAYERS" \
    --flow-layers "$FLOW_LAYERS" \
    --quantizer "$QUANTIZER" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --stage1-steps "$STAGE1_STEPS" \
    --stage2-steps "$STAGE2_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-every "$SAVE_EVERY" \
    --log-every "$LOG_EVERY" \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"
