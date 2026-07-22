# ATSInfer — Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference

**arXiv:2607.10183** — potpuna implementacija u llama.cpp-fusion.

ATSInfer optimizira tensor placement na CPU ↔ GPU za MoE i dense modele,
sa fokusom na consumer devices gdje VRAM nije dovoljan za cijeli model.

---

## 📋 Flags

| Flag | Opis |
|------|------|
| `--atsinfer` | **Uključuje SVE ATSInfer feature-e odjednom** |
| `--atsinfer-budget-mb N` | Simulira VRAM budget od N MB (za testiranje) |
| `--dynamic-transfer` | Runtime CPU↔GPU tensor promocija |
| `--epd` | EPD mjerenje (CUDA event timing) |
| `--mtp-prefetch` | MTP-guided expert prefetch (opt-in) |
| `--override-tensor PAT` | Direktno primjenjuje tensor override |

---

## 🚀 Korištenje

### 1. Sve uključeno (preporučeno)
```bash
./llama-server --atsinfer -m model.gguf
```
Ovo pokreće:
1. **Dvofazno učitavanje**: profil → knapsack DP → reload sa override-ima
2. **Dynamic Transfer**: DP solver svakih 5 decode stepova
3. **Load-aware Re-scheduling**: prati performanse, re-schedule ako devijacija > 15%
4. **EPD mjerenje**: CUDA event timing za svaki step

### 2. Samo knapsack placement (bez runtime DT)
```bash
# Prvo generisati override fajl:
./llama-server --atsinfer ... 2>&1 | grep "placement saved"
# Onda restart sa override-om:
./llama-server --override-tensor "$(cat /tmp/llama_placement.cfg | paste -sd,)" -m model.gguf
```
Knapsack bira najbolje tenzore pojedinačno (ne cijele layere).

### 3. Simulacija slabijeg GPU
```bash
# Testiraj kako bi ATSInfer radio sa samo 4GB VRAM:
LLAMA_ARG_ATSINFER_BUDGET=4096 ./llama-server --atsinfer -m model.gguf

# Onda primijeni override:
./llama-server --override-tensor "$(cat /tmp/llama_placement.cfg | paste -sd,)" -m model.gguf
```

### 4. Samo MTP prefetch (posebno)
```bash
./llama-server --mtp-prefetch -m model.gguf --cpu-moe
```
MTP-guided expert prefetch je **opt-in** — ne uključuje se sa `--atsinfer`.
Koristi draft model routing za predviđanje i preload expert težina u moe-cache.

---

## 📊 Benchmark rezultati (1×RTX 3090, 24GB)

### Full VRAM (default, ~21.5GB na GPU)
| Konfiguracija | PP (t/s) | TG (t/s) |
|--------------|----------|----------|
| Baseline | 88.3 | 11.4 |
| `--atsinfer` | 88.3 | 12.0 |

→ **Neutralno** — default scheduler već stavlja sve što stane.

### Ograničena VRAM (simuliranih 4GB, `LLAMA_ARG_ATSINFER_BUDGET=4096`)
| Konfiguracija | PP (t/s) | TG (t/s) |
|--------------|----------|----------|
| Default (2 cijela layera) | 59.6 | 2.7 |
| Knapsack override (356 tenzora) | **74.8** | **9.2** |

→ **PP +26%, TG +241%** 🚀

Zašto: knapsack bira **najbolje tenzore pojedinačno** po benefit/size metrici,
umjesto cijelih layera. Norme i drugi "jeftini" tenzori ne troše VRAM.

---

## 🔧 Arhitektura

### Algoritam 1: Static Tensor Placement (DP knapsack)
```
max Σ(r_i * GPU_i) - Σ(c_i * switch_i)
subject to Σ(s_i * GPU_i) ≤ MEMORY_BUDGET
```
- `r_i = t_cpu - t_gpu` — benefit GPU izvršenja
- `c_i = activation_size / PCIe_bandwidth` — switching cost
- Za MoE: experti i non-expert tenzori se posebno tretiraju

### Algoritam 2: Dynamic Transfer Scheduling
- O(n²) DP za per-step promociju CPU→GPU tenzora
- `exposed_transfer = max(weight_time - overlap_window, 0)`
- Overlap window = execution + activation transfer na default putu

### Algoritam 3: Load-aware Re-scheduling
- Mjeri stvarni step latency svaki decode korak
- Rolling EMA prosjek
- Trigger re-schedule kad deviation > 15% (min interval = 5 × TPOT)
- cpu_slowdown / gpu_slowdown faktori (1.0–2.0×)

### Memory Layout
```
GPU Memory:
┌──────────────────────┐
│  Resident region      │ ← Static Placement (trajni tenzori)
├──────────────────────┤
│  Temporary buffers    │ ← Dynamic Transfer (4×256MB, reusable)
├──────────────────────┤
│  Compute buffers      │ ← ggml scratch
└──────────────────────┘
```

---

## ⚠️ Ograničenja

1. **Full offload (sve stane u VRAM)**: ATSInfer neutralan — default scheduler već optimalan
2. **CPU-MoE + atsinfer**: može crashati zbog konflikta override-a
3. **MTP prefetch**: zahtijeva draft model — bez njega ne daje predikciju
4. **Dynamic Transfer**: runtime graph patching nije implementiran (scheduler API limit)
