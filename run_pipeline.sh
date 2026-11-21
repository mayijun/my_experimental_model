#!/usr/bin/env bash
# 中英混合 Flow Matching + JEPA：数据准备 -> 预训练 -> 生成评测，一条龙。
# 在 AutoDL / 远程 GPU 机器上运行（需已安装 torch + accelerate + datasets）。
#
# 用法：
#   MODE=smoke bash run_pipeline.sh      # 小规模冒烟：几分钟跑通端到端
#   MODE=full  bash run_pipeline.sh      # 完整 0.3B 预训练 + 评测
#   nohup MODE=full bash run_pipeline.sh > logs/pipeline.log 2>&1 &   # 后台跑
set -euo pipefail

# ---------- 运行环境 ----------
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com      # 关键：HuggingFace 国内镜像
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- 目录 ----------
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints}"

# ---------- 数据规模（条数，中英各一半） ----------
EN_CONFIG="${EN_CONFIG:-wikitext-103-raw-v1}"
EN_TRAIN="${EN_TRAIN:-400000}"
ZH_TRAIN="${ZH_TRAIN:-400000}"
EN_EVAL="${EN_EVAL:-60000}"
ZH_EVAL="${ZH_EVAL:-20000}"

# ---------- 模型 / 训练 ----------
MODE="${MODE:-smoke}"
EPOCHS="${EPOCHS:-0}"   # >0 时按 epoch 训练（忽略 STAGE1/STAGE2 步数）
if [ "$MODE" = "full" ]; then
    DIM=1024; N_HEADS=16; ENC=4; FLOW=12; SEQ=256; BATCH=16
    STAGE1=2000; STAGE2=5000; QUANT=fsq
else
    DIM=64; N_HEADS=8; ENC=1; FLOW=2; SEQ=64; BATCH=8
    STAGE1=30; STAGE2=60; QUANT=fsq
    EN_CONFIG=wikitext-2-raw-v1   # 冒烟用秒下的 6MB 小集
    EN_TRAIN=2000; ZH_TRAIN=800; EN_EVAL=200; ZH_EVAL=100
fi

mkdir -p "$ROOT/logs"

echo "======================================================"
echo "MODE=$MODE | dim=$DIM heads=$N_HEADS enc=$ENC flow=$FLOW seq=$SEQ"
echo "数据: en_train=$EN_TRAIN zh_train=$ZH_TRAIN en_eval=$EN_EVAL zh_eval=$ZH_EVAL"
echo "数据目录: $DATA_DIR | checkpoint: $CKPT_DIR"
echo "======================================================"

# ---------- 1) 数据准备 ----------
echo "[1/3] 准备中英混合语料 ..."
python "$ROOT/data_prepare.py" \
    --out-dir "$DATA_DIR" \
    --en-config "$EN_CONFIG" \
    --en-train "$EN_TRAIN" --zh-train "$ZH_TRAIN" \
    --en-eval "$EN_EVAL" --zh-eval "$ZH_EVAL"

# ---------- 2) 预训练 ----------
echo "[2/3] 预训练 ..."
TRAIN_ARGS=(
    --dataset-files "$DATA_DIR/train.jsonl"
    --seq-len "$SEQ" --dim "$DIM" --n-heads "$N_HEADS"
    --encoder-layers "$ENC" --flow-layers "$FLOW" --quantizer "$QUANT"
    --batch-size "$BATCH"
    --checkpoint-dir "$CKPT_DIR" --save-every 1000000 --log-every 5
)
if [ "$(echo "$EPOCHS > 0" | bc 2>/dev/null || echo 0)" = "1" ]; then
    TRAIN_ARGS+=( --epochs "$EPOCHS" )
else
    TRAIN_ARGS+=( --stage1-steps "$STAGE1" --stage2-steps "$STAGE2" )
fi
python "$ROOT/pretrain.py" "${TRAIN_ARGS[@]}"

# ---------- 3) 生成评测 ----------
echo "[3/3] 生成评测 ..."
python "$ROOT/eval_generate.py" \
    --checkpoint-dir "$CKPT_DIR" \
    --eval-file "$DATA_DIR/eval.jsonl" \
    --inference-steps 20 --solver heun

echo "全部完成。日志见 $ROOT/logs/"
