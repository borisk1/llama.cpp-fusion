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
NUMA-aware multi-group CPU execution for dual-Xeon / multi-socket systems. The core idea: split CPU threads into independent groups, each processing a separate pipeline stage with NUMA-local memory access.

```
--thread-copy <group1>[,<group2>,<group3>,...]
```

**How the numbers work:**

Each group is a comma-separated list of CPU core ranges. The groups process different pipeline stages in parallel. Memory allocated by a group stays local to its NUMA node.

```
# 2 groups, both on NUMA node 0 (cores 0-11)
--thread-copy 0-5,6-11
#   ^--- group 0: cores 0-5
#             ^--- group 1: cores 6-11

# 2 groups, cross-socket (NUMA node 0 + NUMA node 1)
--thread-copy 0-5,24-29
#   ^--- group 0 on socket 0 (cores 0-5)
#             ^--- group 1 on socket 1 (cores 24-29)

# 4 groups, 2 per socket
--thread-copy 0-2,3-5,24-26,27-29
```

**How to pick the right numbers:**

1. Check your NUMA topology:
```bash
numactl --hardware
# Example output on dual Xeon:
# node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# node 1 cpus: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
```

2. Pick cores from the **same NUMA node** for each group, and pin with `numactl`:

```bash
# 2 groups, same socket (node 0)
numactl --cpunodebind=0 --membind=0 \
  ./build/bin/llama-server ... --thread-copy 0-5,6-11

# 2 groups, cross-socket (node 0 + node 1)
numactl --cpunodebind=0 --membind=0 \
  ./build/bin/llama-server ... --thread-copy 0-5,24-29

# 4 groups, spread across both sockets
numactl --cpunodebind=0 --membind=0 \
  ./build/bin/llama-server ... --thread-copy 0-2,3-5,24-26,27-29
```

**Why it matters on dual Xeon:**

On a dual-socket system, memory access patterns dominate decode performance. When a thread on socket 1 reads model weights allocated on socket 0's DRAM, it crosses the UPI/QPI interconnect — adding 100-200 ns latency per access. For a 79 GB model streamed token-by-token, this overhead adds up.

Thread Copy ensures:
- Each group processes a contiguous pipeline stage with NUMA-local memory
- Cross-socket transfers only happen between pipeline stages, not within them
- Book-keeping overhead (plan/dispatch/collision) is isolated per-group

**Results (DSV4 Flash, 1× RTX 3090):**
- Baseline (single thread pool, 24 threads): 6.37 t/s
- Same-socket TC (12 threads, 2 groups): 6.89 t/s (+8%)
- Cross-socket TC (24 threads, 2 groups): 6.6-6.8 t/s (+4-7%)

**When NOT to use:**
- Single-socket systems (no NUMA benefit)
- Systems with few cores (dividing into groups reduces per-group parallelism)
- GPU-only inference (no CPU compute to accelerate)

### 🎯 MoE Cache (leloch)
GPU-based expert weight cache that dramatically reduces CPU→GPU transfers for MoE models. Works with all MoE architectures.

```
GGML_CUDA_MOE_CACHE=1 GGML_CUDA_MOE_CACHE_BUDGET_MB=11000
```

**How it works:**
- Intercepts expert weight lookups during inference (both prefill and decode)
- Maintains a GPU-side LRU pool of recently used expert weights
- Pool: up to 2,666 slots × 2112 KB each ≈ 11 GB total budget
- Async prefetch: while GPU computes with cached experts, missed experts are prefetched from CPU RAM in the background

**Measured perf (DSV4 Flash, 200K ctx, 1 GPU):**
- 11.78 TG t/s with cache enabled vs 8.0 TG without — **+47%**
- 70.3% hit rate over 1.2 million lookups
- Average plan/dispatch/collision overhead: ~28 µs per lookup
- Redirect mechanism handles repeated miss patterns (up to 93K redirects)

**Key env vars:**
| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_CUDA_MOE_CACHE` | 0 | Enable (1) or disable |
| `GGML_CUDA_MOE_CACHE_BUDGET_MB` | auto | Max VRAM for cache pool |
| `GGML_CUDA_MOE_CACHE_STATS` | 0 | Print stats every N tokens |

### 🤖 MTP Stack (Multi-Token Prediction / Speculative Decoding)
Native support for DeepSeek V4 Flash's built-in MTP (Multi-Token Prediction) draft model. The merged GGUF contains both the trunk model (43 layers) and the MTP head (1 nextn block) — no separate draft model file needed.

```
# Enable speculative decoding with built-in draft model
--speculative-model model.gguf --speculative-n-draft 3
```

**How it works (DSV4 Flash architecture):**
- The model includes one MTP block (`blk.43.nextn.*`) that acts as a lightweight draft model
- During generation, the draft model proposes up to N tokens in a single forward pass
- The trunk model verifies all proposals in parallel (tree attention)
- Accepted tokens are emitted; rejected tokens trigger a draft re-roll

**Architecture details:**
- Single MTP block (same architecture as a plain trunk block — no compression, no hash routing)
- Input: `e_proj * enorm(token)` + `h_proj * hnorm(hidden)` — hybrid embedding + hidden state
- Tied input/output: shares `tok_embd` and `output` (lm_head) with the trunk
- Shares the same HC (Heavy Compensation) head functions

**Note:** MTP is optional. The trunk model runs fine without it. Enable when you need maximum single-request throughput and have VRAM to spare (the draft model adds ~2-3 GB VRAM overhead).

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

## Run Examples

All examples assume a DeepSeek V4 Flash model (`DSV4-Flash-IQ2_XXS-MTP-merged.gguf`, ~79 GB). Adjust the model path for your setup.

> **Note on hardware:** All benchmarks were performed on a system with 4× RTX 3090 connected via **OCuLink** (PCIe 3.0 x4 per link, aggregated). OCuLink bandwidth is lower than direct PCIe 3.0 x16 slots — expect higher throughput on systems with native PCIe 4.0. The MoE Cache and CPU-MoE hybrid modes are especially beneficial on OCuLink setups, as they minimize PCIe transfers.

### Single GPU + CPU hybrid (1× RTX 3090, model doesn't fit in VRAM)

Most common scenario — the 79 GB model far exceeds 24 GB VRAM. Attention runs on GPU, MoE experts stay in CPU RAM via CUDA_Host pinned memory.

```bash
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server \
  -m /path/to/DSV4-Flash-IQ2_XXS-MTP-merged.gguf \
  --no-mmap --flash-attn on \
  -c 200000 -t 12 \
  -b 4096 -ub 512 \
  --cpu-moe \
  --no-jinja
```

**Expected:** ~11-12 TG t/s, ~200 PP t/s (5K+ prompt)

### Single GPU + MoE cache (no cpu-moe)

Let the GPU handle everything it can. MoE cache reduces expert weight transfers by 70%.

```bash
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server \
  -m /path/to/DSV4-Flash-IQ2_XXS-MTP-merged.gguf \
  --no-mmap --flash-attn on \
  -c 200000 -t 12 \
  -b 4096 -ub 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --no-jinja
```

### Multi-GPU (4× RTX 3090, full VRAM offload)

When your total VRAM exceeds model size (96 GB > 79 GB), the entire model fits on GPUs. No PCIe weight streaming during inference.

```bash
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server \
  -m /path/to/DSV4-Flash-IQ2_XXS-MTP-merged.gguf \
  --no-mmap --flash-attn on \
  -c 200000 -t 12 \
  -b 4096 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --no-jinja
```

**Expected:** ~35-38 TG t/s, ~500+ PP t/s (5K+ prompt)

### Multi-GPU + Thread Copy (dual Xeon + 4 GPU)

Maximum hardware utilization — NUMA-aware CPU execution across two sockets + all GPUs.

```bash
numactl --cpunodebind=0 --membind=0 \
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-server \
  -m /path/to/DSV4-Flash-IQ2_XXS-MTP-merged.gguf \
  --no-mmap --flash-attn on \
  -c 200000 -t 12 \
  -b 4096 -ub 512 \
  --cpu-moe \
  --thread-copy 0-5,6-11 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --no-jinja
```

`--thread-copy 0-5,6-11` splits 12 threads into 2 groups of 6. Group 0 runs on NUMA node 0 cores 0-5, group 1 on cores 6-11. Adjust for your CPU topology.

### Thread Copy for dual Xeon (CPU-only decode)

When you have massive RAM but zero or limited GPU. Thread Copy splits threads across sockets for NUMA-local memory access.

```bash
numactl --cpunodebind=0 --membind=0 \
  ./build/bin/llama-server \
  -m /path/to/model.gguf \
  --no-mmap -c 100000 -t 24 \
  --thread-copy 0-5,6-11,12-17,18-23
```

4 groups of 6 threads each — one group per L3 cache domain. Improves memory-bound decode on multi-socket systems by keeping memory access local.

### Production server (systemd)

```bash
# Install service
cp scripts/llama-dsv4.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now llama-dsv4

# Or run directly
bash scripts/prod-server.sh prod   # 4 slots, balanced
bash scripts/prod-server.sh fast   # 1 slot, max speed
```

### Benchmark quick test

```bash
GGML_CUDA_MOE_CACHE=1 \
GGML_CUDA_MOE_CACHE_BUDGET_MB=11000 \
  ./build/bin/llama-bench \
  -m /path/to/DSV4-Flash-IQ2_XXS-MTP-merged.gguf \
  -p 512 -n 128 -ngl 999 -t 12 -r 1
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
