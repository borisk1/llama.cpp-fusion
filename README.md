# llama.cpp-fusion

**CPU + GPU + Multi-GPU + Thread Copy fusion** — jedno mjesto gdje se spajaju svi resursi.

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with advanced hybrid inference for MoE models (DeepSeek V4 Flash, Qwen3 MoE, DeepSeek, etc.).

## Key Features

| # | Feature | What it does |
|---|---------|--------------|
| 🧵 | **Thread Copy (TC)** | NUMA-aware multi-group CPU execution for dual-Xeon/single-GPU setups. `--thread-copy 0-5,6-11` splits CPU threads into groups — each group drives a separate pipeline stage. Up to +30% on multi-socket. |
| 🎯 | **MoE Cache** | GPU-based expert weight cache (leloch). Dramatically reduces CPU→GPU transfers for MoE models. Up to 70% hit rate, ~11 GB budget. |
| 🔥 | **Prefill-driven Hot Cache** | Predicts and preloads hot experts during prefill. Ideal for long-context sessions. `LLAMA_HOT_EXPERTS=64`. |
| 🔄 | **Multi-GPU Offloading** | Distribute 79 GB models across 4× RTX 3090. Layer-split with auto-balance. Achieves **599 PP / 37.78 TG** on DSV4 Flash IQ2_XXS. |
| ⚡ | **GPU + CPU Hybrid** | `--cpu-moe` keeps experts on CPU while attention runs on GPU. Perfect when model doesn't fit in VRAM — **11.78 TG** on single 3090 with 79 GB model. |
| 🔗 | **Pipeline Parallel** | Multi-GPU pipeline parallel for full-model offload (requires all layers on GPU). |

## Performance

| Setup | Model | PP (t/s) | TG (t/s) |
|-------|-------|----------|----------|
| 1× RTX 3090 + 128 GB DDR4 | DSV4 Flash IQ2_XXS (79 GB) | 186 | 11.78 |
| 4× RTX 3090 (full VRAM) | DSV4 Flash IQ2_XXS (79 GB) | **599** | **37.78** |
| 4× RTX 3090 (200K ctx, 4 slots) | DSV4 Flash IQ2_XXS (79 GB) | 571 | 37.78 |

## Quick Start

```bash
# Production server (4 slots, balanced)
bash scripts/prod-server.sh prod

# Fast single-slot (max speed)
bash scripts/prod-server.sh fast

# Single GPU + CPU hybrid
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server -m model.gguf \
  --no-mmap --flash-attn on -c 200000 -t 12 \
  -b 4096 -ub 512 --cpu-moe --thread-copy 0-5,6-11
```

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) with contributions from the community. Thread Copy ported from DeepSeek-thread fork. MoE cache inspired by leloch's work.
