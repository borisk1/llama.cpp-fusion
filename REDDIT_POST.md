# Introducing llama.cpp-fusion — CPU + GPU + Multi-GPU + Thread Copy, one place where all resources come together

**GitHub:** https://github.com/borisk1/llama.cpp-fusion

I've been working on a fork of llama.cpp that focuses on making the most of whatever hardware you have — whether that's a single RTX 3090 with 128 GB DDR4, a dual Xeon server with 4 GPUs, or anything in between.

The main goal was squeezing decent performance out of DeepSeek V4 Flash (79 GB model) on consumer and workstation hardware without buying a 5090.

## What's different

**🧵 Thread Copy** — NUMA-aware multi-group CPU execution for dual Xeon / multi-socket systems. Instead of one thread pool, you get independent groups pinned to specific NUMA nodes. Each group processes a pipeline stage with local memory access, avoiding UPI/QPI cross-socket penalties.

```
--thread-copy 0-5,6-11    # 2 groups on same socket
--thread-copy 0-5,24-29   # cross-socket
```

**🎯 MoE Cache (leloch)** — GPU-side expert weight cache. Intercepts expert lookups, maintains an LRU pool in VRAM, and async-prefetches misses. 70%+ hit rate, ~28 us overhead per lookup.

```
GGML_CUDA_MOE_CACHE=1 GGML_CUDA_MOE_CACHE_BUDGET_MB=11000
```

**🔥 Prefill-driven Hot Cache** — predicts which experts will be needed during generation based on prefill routing patterns. Implements Insight 1 from arxiv 2510.05497.

**🔄 Multi-GPU offloading** — distributes model layers across all available GPUs by free memory. No manual tensor_split needed.

**⚡ CPU-MoE hybrid** — attention on GPU, experts on CPU. When your model doesn't fit in VRAM, this keeps generation fast without requiring more GPUs.

## Benchmarks (DeepSeek V4 Flash IQ2_XXS, 79 GB)

All tests on **dual Xeon Platinum 8160 (Skylake)** — 2x24 cores, 768 GB DDR4, **4x RTX 3090 via OCuLink**.

| Config | PP (t/s) | TG (t/s) |
|--------|----------|----------|
| 1x 3090, CPU-MoE hybrid, MoE cache | 186 | **11.78** |
| 4x 3090, full offload, MoE cache | **571** | **37.78** |

The 4-GPU numbers are on OCuLink (PCIe 3.0 x4 per link), so direct PCIe 4.0 slots would be even faster.

## Why another fork?

I started with the fairydreaming/llama.cpp dsv4 branch and mainline b10064, and ran into several issues that needed fixing:

1. **Flash Attention tensors missing required names** — FA auto-detection crashed on DSV4 graphs
2. **Missing op names** — LIGHTNING_INDEXER and DSV4 HC ops weren't in the name/symbol tables
3. **ACCEL buffer type in CPU buft list** — caused CUDA device allocation for CPU-bound tensors, leading to OOM on 24 GB cards

These are all fixed in the fork. The mainline builds b10064+ work fine for basic inference, but the hybrid features (Thread Copy, MoE cache, hot cache) are only in this fork.

## Who is this for?

- **Single GPU users** with large MoE models that don't fit in VRAM — CPU-MoE hybrid + MoE cache gives ~12 t/s on a 3090 with a 79 GB model
- **Dual Xeon / multi-socket users** — Thread Copy makes use of both sockets with NUMA-aware thread groups
- **Multi-GPU setups** — automatic layer distribution across all GPUs
- **Anyone with a lot of RAM and at least one GPU** — the hybrid mode was designed exactly for this

## Quick start

```bash
# Single GPU hybrid (most common scenario)
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server -m model.gguf \
  --no-mmap --flash-attn on -c 200000 -t 12 \
  -b 4096 -ub 512 --cpu-moe

# Multi-GPU (4x 3090)
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server -m model.gguf \
  --no-mmap --flash-attn on -c 200000 -t 12 \
  -b 4096 -ub 512 --cache-type-k q8_0 --cache-type-v q8_0
```

More examples and full docs at https://github.com/borisk1/llama.cpp-fusion

Would love to hear if anyone tests this on different hardware — Ryzen + single GPU, Threadripper, dual Epyc, etc.
