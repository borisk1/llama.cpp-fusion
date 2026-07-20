#pragma once

#include "ggml.h"
#include <cstdint>

// Initialize MTP router trace (call once during model load)
void llama_router_trace_init(int n_layers, int n_experts, int k_active);

// Set current step and draft/main mode
void llama_router_trace_set_step(int32_t step);
void llama_router_trace_set_draft(int32_t is_draft);

// Callback after each tensor eval — captures expert routing decisions
void llama_router_trace_callback(struct ggml_tensor * t, bool ask, void * user_data);

// Close trace file (flush + close)
void llama_router_trace_close();

// --- MTP-guided prefetching ---

// Set a callback that gets called when expert routing is known for a layer.
// The callback receives (layer_id, expert_indices[], n_experts, is_draft).
// This is used to prefetch experts into the hot-cache.
typedef void (*mtp_router_callback_t)(int layer, const int32_t * experts, int n_experts, int is_draft);

// Register the prefetch callback (call once during init)
void llama_router_trace_set_callback(mtp_router_callback_t cb, void * user_data);
