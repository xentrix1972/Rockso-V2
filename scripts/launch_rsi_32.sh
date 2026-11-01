#!/usr/bin/env bash
# Portable recipe with Nick's lower fresh-run base rates; other settings retained.
# Use a fresh output directory for a new run. Review inputs/environment first.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ROCKSO_DATA:?Set ROCKSO_DATA to your packed dataset directory}"
: "${ROCKSO_TOKENIZER:?Set ROCKSO_TOKENIZER to the exact tokenizer directory}"
: "${ROCKSO_OUT:?Set ROCKSO_OUT to a separate run output directory}"
PYTHON="${ROCKSO_PYTHON:-python3}"
LEDGER="${ROCKSO_LEDGER:-$ROCKSO_DATA/ledger/order.npy}"
LEDGER_META="${ROCKSO_LEDGER_META:-$ROCKSO_DATA/ledger/ledger.json}"
ulimit -n 65535
export PYTHONWARNINGS=ignore
export CUDA_DEVICE_MAX_CONNECTIONS=2
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/inductor}"

exec "$PYTHON" -u "$ROOT/train.py" \
  --config "$ROOT/configs/rockso_v2_32.json" \
  --out "$ROCKSO_OUT" --resume auto \
  --data "$ROCKSO_DATA" --split train --ledger "$LEDGER" --ledger-meta "$LEDGER_META" \
  --start-row 0 --tokenizer "$ROCKSO_TOKENIZER" \
  --residual-lr-mult 1.0 --branch-lr-mult 4.0 \
  --device cuda --dtype bfloat16 --fp8 --compile --checkpointing --keep-last 2 \
  --fp8-mlp-pad-to 6912 --backend flex --attention-mode prefixlm --loss liger \
  --optimizer normuon --fp32-master-weights --normuon-stochastic-rounding \
  --batch-size 20 --grad-accum 8 --num-workers 2 --prefetch-factor 2 \
  --persistent-workers --pin-memory --non-blocking \
  --steps 176491 --lr-schedule wsd --wsd-decay-style linear \
  --stable-until-step 141193 --schedule-steps 176491 --warmup-steps 1000 --min-lr-ratio 0.0 \
  --muon-lr 0.007 --muon-momentum-start 0.85 --muon-momentum 0.95 \
  --muon-momentum-warmup-steps 1000 --adam-lr 0.0005 --weight-decay 0.025 \
  --ve-lr-mult 64 --ve-weight-decay 0.000390625 --clip-grad 1.0 \
  --save-every 1000 --log-every 10 --wandb-mode disabled \
  --wandb-every 10 --monitor-every 50 --monitor-layer-every 250 --monitor-sample-size 2048 \
  --seed 42 --threads 4 "$@"
