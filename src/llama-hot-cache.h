#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the hot expert cache
void hot_cache_init(int n_layers, int n_experts);

// Record an expert activation during decode
void hot_cache_record(int layer, const int * expert_indices, int n_active);

// Check if expert is marked as hot (GPU-resident)
bool hot_cache_is_hot(int layer, int expert);

// Get number of hot experts
int hot_cache_get_n_hot(void);

// Check if cache has been computed
bool hot_cache_is_active(void);

// Get number of experts
int hot_cache_get_n_experts(void);

// 1 when any hot-cache/hit-rate env knob is set — graph build then marks
// routing tensors as outputs so post-compute readback sees live data
int hot_cache_capture_enabled(void);

// Read back expert routing from graph result after compute
// is_mtp_graph: 1 when the result comes from an MTP draft context (LLM_GRAPH_TYPE_DECODER_MTP)
// n_tokens/pos/token: ubatch token metadata (positions + token ids), needed by the
// C1 acceptance-conditioned routing test; pass 0/NULL when unavailable
struct llm_graph_result;
void hot_cache_readback_from_result(struct llm_graph_result * res, int is_mtp_graph,
                                    int n_tokens, const int32_t * pos, const int32_t * token);

#ifdef __cplusplus
}
#endif
