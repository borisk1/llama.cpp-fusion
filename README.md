# llama.cpp-fusion

**CPU + GPU + Multi-GPU + Thread Copy fusion** — one place where all resources come together.

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) focused on advanced hybrid inference for large MoE models (DeepSeek V4 Flash, Qwen3 MoE, DeepSeek, and more). Built for systems with limited VRAM, multi-socket CPUs, and multi-GPU setups — making the most of whatever hardware you have.

---

## Why llama.cpp-fusion?

| Problem | Solution |
|---------|----------|
| MoE model too large for VRAM | **CPU-MoE hybrid** — attention on GPU, experts on CPU + MoE Cache |
| Dual Xeon underutilized | **Thread Copy (TC)** — multi-group NUMA-aware execution |
| Multiple GPUs stuck at PCIe bottleneck | **Multi-GPU offloading** — distribute 79 GB models across 4+ GPUs |
| Expert weights thrashing PCIe | **MoE Cache** (leloch) — GPU-side expert cache with 70%+ hit rate |
| Slow generation on long contexts | **Prefill-driven Hot Cache** — predict and preload hot experts |

---

## Features

### 🧵 Thread Copy (TC)
NUMA-aware multi-group CPU execution for dual-Xeon / multi-socket systems.

```
--thread-copy 0-5,6-11
```

Splits CPU threads into groups — each group drives a separate pipeline stage. Up to +30% on multi-socket vs. vanilla threadpool. Ported from the DeepSeek-thread fork.

### 🎯 MoE Cache (leloch)
GPU-based expert weight cache that dramatically reduces CPU→GPU transfers for MoE models.

```
GGML_CUDA_MOE_CACHE=1 GGML_CUDA_MOE_CACHE_BUDGET_MB=11000
```

- Up to **70% hit rate** (measured: 566K hits / 1.2M lookups)
- ~11 GB budget, ~2 MB per expert slot
- Covers all MoE models: DeepSeek V4, Qwen3 MoE, DeepSeek, DBRX, Grok

### 🔥 Prefill-driven Hot Cache
Predicts which experts will be needed during generation based on prefill routing patterns.

```
LLAMA_PREFILL_DRIVEN=1 LLAMA_HOT_EXPERTS=64
```

Implements Insight 1 from [arxiv 2510.05497](https://arxiv.org/abs/2510.05497) — activates hot expert cache after the first decode token. Ideal for long-conference sessions and chatbots.

### 🔄 Multi-GPU Offloading
Distribute models across all available GPUs with automatic layer-split by free memory.

```
# 4 GPUs, full model offload (79 GB IQ2_XXS → 96 GB across 4×RTX 3090)
# Results: 599 PP / 37.78 TG on DeepSeek V4 Flash
```

Layer-split with auto-balance. Pipeline parallelism available when all layers fit on GPU.

### ⚡ GPU + CPU Hybrid
Keep MoE experts on CPU while attention runs on GPU — perfect when the model doesn't fit entirely in VRAM.

```
--cpu-moe
```

Achieves **11.78 TG** on a single RTX 3090 with a 79 GB model. Most of the model stays in system RAM; only attention weights occupy VRAM.

---

## Performance

| Setup | Model | PP (t/s) | TG (t/s) |
|-------|-------|----------|----------|
| 1× RTX 3090 + 128 GB DDR4 | DeepSeek V4 Flash IQ2_XXS (79 GB) | 186 | **11.78** |
| 4× RTX 3090 (full offload) | DeepSeek V4 Flash IQ2_XXS (79 GB) | **599** | **37.78** |
| 4× RTX 3090 (200K ctx, 4 slots) | DeepSeek V4 Flash IQ2_XXS (79 GB) | 571 | 37.78 |

All measurements with `--flash-attn on -b 4096 -ub 512 --cache-type-k/q q8_0`.

---

## Quick Start

### Production server (4 slots, balanced)

```bash
bash scripts/prod-server.sh prod
```

### Fast single-slot (max throughput)

```bash
bash scripts/prod-server.sh fast
```

### Single GPU + CPU hybrid

```bash
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server -m model.gguf \
  --no-mmap --flash-attn on -c 200000 -t 12 \
  -b 4096 -ub 512 --cpu-moe --thread-copy 0-5,6-11
```

### Multi-GPU with MoE cache

```bash
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server -m model.gguf \
  --no-mmap --flash-attn on -c 200000 -t 12 \
  -b 4096 -ub 512 --cache-type-k q8_0 --cache-type-v q8_0
```

### Systemd service

```bash
cp scripts/llama-dsv4.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now llama-dsv4
```

---

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```

For CPU-only or hybrid builds, see the upstream [build docs](docs/build.md).

---

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), the best open-source inference engine.

- **Thread Copy** — ported from the DeepSeek-thread fork
- **MoE Cache (leloch)** — inspired by community work on expert caching
- **Hot Cache** — based on arxiv 2510.05497
- **DSV4 support** — thanks to the fairydreaming, bullerwins, and Unsloth communities

---

## License

Same as upstream — MIT. See [LICENSE](LICENSE).
