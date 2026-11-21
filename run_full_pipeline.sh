#!/usr/bin/env bash
# 全量数据预训练：等待下载完成 -> 准备全量数据 -> 启动 1-epoch 预训练。
# 下载（download_full.py）与评测（eval_generate.py）结束后，本脚本自动接力。
set -u
cd /root/autodl-tmp/jepa
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com

# 1) 等待全量下载完成
echo "[1/3] 等待全量下载完成 ..."
while pgrep -f "download_full[.]py" >/dev/null; do sleep 30; done
if ! grep -q "全部完成" /root/autodl-tmp/download_full.log 2>/dev/null; then
    echo "下载未完成（进程退出但无完成标记），中止"; exit 1
fi
echo "[1/3] 下载完成"

# 2) 准备全量数据（命中缓存，预算设无穷大以读全部分片）
echo "[2/3] 准备全量数据 ..."
python data_prepare.py \
    --out-dir /root/autodl-tmp/data_full_all \
    --en-train-chars 10000000000 \
    --zh-train-chars 10000000000 \
    --en-eval-chars 2000000 \
    --zh-eval-chars 2000000 \
    || { echo "数据准备失败"; exit 1; }
echo "[2/3] 数据准备完成"

# 3) 启动全量预训练（1 epoch，全量 0.3B 配置）
echo "[3/3] 启动全量预训练（1 epoch）..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python pretrain.py \
    --dataset-files /root/autodl-tmp/data_full_all/train.jsonl \
    --seq-len 256 --dim 1024 --n-heads 16 \
    --encoder-layers 4 --flow-layers 12 --quantizer fsq \
    --batch-size 16 \
    --checkpoint-dir /root/autodl-tmp/jepa/checkpoints_full \
    --epochs 1 --save-every 1000000 --log-every 100
echo "[3/3] 预训练结束"
