#pragma once

#include <stdint.h>

struct llm_graph_result;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// Store expert indices tensor during graph build (called from build_moe_ffn)
void hot_cache_store_tensor(int layer, struct ggml_tensor * expert_indices_tensor);

// Read back expert routing after graph compute (called from process_ubatch)
// is_mtp_graph: 1 when the result comes from an MTP draft context
// n_tokens/pos/token: ubatch metadata for the C1 routing test (0/NULL when unavailable)
void hot_cache_readback_from_result(struct llm_graph_result * res, int is_mtp_graph,
                                    int n_tokens, const int32_t * pos, const int32_t * token);

#ifdef __cplusplus
}
#endif
