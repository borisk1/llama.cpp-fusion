#!/bin/bash
# ============================================================
# DSV4 Flash Production Server — v2-gpu-layers
# Usage: ./scripts/prod-server.sh [profile]
#   profile: "fast" (1 slot, ub=4096) or "prod" (4 slots, ub=512)
# ============================================================
set -euo pipefail

REPO_DIR="/mnt/raid0/mtp-prefetch/llama.cpp"
MODEL="/mnt/raid0/Paket/DSV4-Flash-IQ2_XXS-MTP-merged.gguf"
PORT=8011
HOST="0.0.0.0"
CTX=200000
THREADS=12

PROFILE="${1:-prod}"

case "$PROFILE" in
  fast)
    SLOTS=1
    BATCH=4096
    UBATCH=4096
    MOE_CACHE=11000
    DESC="1 slot, max speed (599 PP / 20.76 TG)"
    ;;
  prod)
    SLOTS=4
    BATCH=4096
    UBATCH=512
    MOE_CACHE=11000
    DESC="4 slots, balanced (multi-user)"
    ;;
  *)
    echo "Usage: $0 [fast|prod]"
    exit 1
    ;;
esac

cd "$REPO_DIR"
pkill -f "llama-server.*port $PORT" 2>/dev/null || true
sleep 2

export GGML_CUDA_MOE_CACHE=1
export GGML_CUDA_MOE_CACHE_BUDGET_MB=$MOE_CACHE
export GGML_CUDA_MOE_CACHE_STATS=200

echo "=== DSV4 Flash Production Server ==="
echo "  Profile: $PROFILE — $DESC"
echo "  Model:   $MODEL"
echo "  Port:    $HOST:$PORT"
echo "  Context: $CTX"
echo "  Slots:   $SLOTS"
echo "  ubatch:  $UBATCH"
echo "  moe-cache budget: ${MOE_CACHE}MB"

exec numactl --cpunodebind=0 --membind=0 \
  ./build/bin/llama-server \
    -m "$MODEL" \
    --no-mmap \
    --flash-attn on \
    --port "$PORT" \
    --host "$HOST" \
    -c "$CTX" \
    -t "$THREADS" \
    --parallel "$SLOTS" \
    -np "$SLOTS" \
    -b "$BATCH" \
    -ub "$UBATCH" \
    --no-jinja \
    --cache-type-k q8_0 \
    --cache-type-v q8_0
