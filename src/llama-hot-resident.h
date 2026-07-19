#pragma once

// Hot-expert VRAM residency: keeps the top-N most frequently used experts of
// each MoE layer resident on the GPU while the full (fused) expert tensors
// stay in host memory (--cpu-moe). build_moe_ffn splits each expert matmul
// into a GPU hot path and a CPU cold path.
//
// Enable via env:
//   LLAMA_HOT_RESIDENT=/path/to/hot_cache.json   (dump from LLAMA_HOT_EXPERTS_DUMP)
//   LLAMA_HOT_N=64                               (optional, default: n_hot from JSON)

#include <cstdint>

struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// 1 when LLAMA_HOT_RESIDENT is set
int llama_hot_resident_enabled(void);

// register a layer's fused expert tensors (called during model load, before finalize)
void llama_hot_resident_register_layer(int il, struct ggml_tensor * gate_exps,
                                       struct ggml_tensor * up_exps,
                                       struct ggml_tensor * down_exps,
                                       int n_expert, int n_expert_used);

// allocate GPU buffers, copy hot expert slices, build id-remap tables.
// call once after tensor data is fully loaded. returns 0 on failure/disabled.
int llama_hot_resident_finalize(void);

// graph-build lookup: fused expert tensor -> hot twin + remap tables.
// returns 1 and fills outputs when the tensor participates in a hot split.
int llama_hot_resident_get(const struct ggml_tensor * fused,
                           struct ggml_tensor ** hot,
                           struct ggml_tensor ** map_hot,   // F32[n_expert]: hot slot or 0 (dummy)
                           struct ggml_tensor ** map_cold,  // F32[n_expert]: global id or OOR
                           struct ggml_tensor ** mask_hot); // F32[n_expert]: 1.0 hot / 0.0 cold

#ifdef __cplusplus
}
#endif
