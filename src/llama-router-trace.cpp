// MTP router trace + prefetch — captures expert routing from MTP draft
// and feeds predictions into the hot-cache for better expert prefetching.
#include "llama-router-trace.h"
#include "llama-hot-cache.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>

// Trace record format
static const uint32_t TRACE_MAGIC = 0x4D545052;
static const int      TRACE_TOP_K = 16;

struct trace_record {
    int32_t step;
    int32_t layer;
    int32_t is_draft;
    int32_t token_id;
    int32_t topk[TRACE_TOP_K];
    float   weights[TRACE_TOP_K];
};

// Global state
static FILE *              g_trace_file   = nullptr;
static int32_t             g_trace_step   = 0;
static int32_t             g_trace_draft  = 0;
static mtp_router_callback_t g_callback  = nullptr;
static void *              g_cb_user_data = nullptr;
static int32_t             g_n_layers     = 0;
static int32_t             g_n_experts    = 0;
static std::mutex          g_trace_mutex;
static int                 g_initialized  = 0;

void llama_router_trace_init(int n_layers, int n_experts, int k_active) {
    if (g_initialized) return;
    g_n_layers  = n_layers;
    g_n_experts = n_experts;
    g_initialized = 1;

    const char * path = std::getenv("LLAMA_TRACE_ROUTER");
    if (path && path[0]) {
        g_trace_file = std::fopen(path, "wb");
        if (!g_trace_file) {
            fprintf(stderr, "[router-trace] cannot open %s\n", path);
        } else {
            int32_t hdr[8] = {(int32_t)TRACE_MAGIC, 1, n_layers, n_experts, k_active, 0, 0, 0};
            fwrite(hdr, sizeof(int32_t), 8, g_trace_file);
            fflush(g_trace_file);
            fprintf(stderr, "[router-trace] trace file %s (%d layers, %d experts)\n", path, n_layers, n_experts);
        }
    }

    fprintf(stderr, "[router-trace] initialized: %d layers, %d experts\n", n_layers, n_experts);
}

void llama_router_trace_set_step(int32_t step) { g_trace_step = step; }
void llama_router_trace_set_draft(int32_t is_draft) { g_trace_draft = is_draft; }
void llama_router_trace_set_callback(mtp_router_callback_t cb, void * user_data) {
    g_callback = cb;
    g_cb_user_data = user_data;
}

void llama_router_trace_callback(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!g_initialized) return;
    if (!t || ask) return;

    const char * name = ggml_get_name(t);
    if (!name) return;

    // Only interested in expert routing tensors
    bool is_indices = false;
    if (std::strstr(name, "ffn_moe_argsort")) {
        is_indices = true;
    } else if (!std::strstr(name, "ffn_moe_topk")) {
        return;
    }

    // Extract layer index from tensor name (e.g. "ffn_moe_topk-34" or "blk.34.ffn_moe_topk")
    int layer = -1;
    // Try "blk.N." pattern first
    const char * blk = std::strstr(name, "blk.");
    if (blk) {
        char * end;
        long val = std::strtol(blk + 4, &end, 10);
        if (end != blk + 4 && val >= 0 && val < 256) layer = (int)val;
    }
    if (layer < 0) {
        // Try "-N" or "_N" suffix
        for (const char * p = name; *p; p++) {
            if (*p == '-' || *p == '_') {
                char * end;
                long val = std::strtol(p + 1, &end, 10);
                if (end != p + 1 && val >= 0 && val < 256) { layer = (int)val; break; }
            }
        }
    }
    if (layer < 0) return;

    // Skip batched (only single-token decode matters for prefetching)
    if (t->ne[2] > 1) return;

    // Read tensor data (CPU or GPU)
    void * data = t->data;
    std::vector<uint8_t> local_buf;
    if (!data && t->buffer) {
        size_t sz = ggml_nbytes(t);
        local_buf.resize(sz);
        ggml_backend_tensor_get(t, local_buf.data(), 0, sz);
        data = local_buf.data();
    }
    if (!data) return;

    int n_elements = (int)(t->ne[0] * t->ne[1]);
    if (n_elements <= 0 || n_elements > 256) return;

    // --- MTP-guided prefetching: feed draft predictions into hot-cache ---
    // During warmup, this helps the hot-cache learn which experts the MTP predicts.
    // After activation, this is a no-op (hot_cache_record silently returns).
    if (g_trace_draft && !is_indices) {
        float * probs = (float *)data;
        std::vector<int> top_experts;
        for (int i = 0; i < n_elements && i < 64; i++) {
            if (probs[i] > 0.01f) top_experts.push_back(i);
        }
        if (!top_experts.empty()) {
            hot_cache_record(layer, top_experts.data(), (int)top_experts.size());
        }
    }

    // --- Callback for external prefetching (e.g., hot-cache re-activation) ---
    if (g_callback && g_trace_draft && !is_indices) {
        // ffn_moe_topk contains the top-k probabilities
        // The corresponding indices are in ffn_moe_argsort (already processed)
        // For now, we capture all top-k as potential prefetch candidates
        float * probs = (float *)data;
        std::vector<int32_t> top_experts;
        std::vector<std::pair<float, int>> sorted;
        for (int i = 0; i < n_elements; i++) {
            sorted.push_back({probs[i], i});
        }
        std::sort(sorted.begin(), sorted.end(), [](auto & a, auto & b) { return a.first > b.first; });
        // Take top-k (up to k_active)
        for (int i = 0; i < n_elements && i < (int)sorted.size(); i++) {
            if (sorted[i].first > 0.01f) { // threshold
                top_experts.push_back(sorted[i].second);
            }
        }
        if (!top_experts.empty()) {
            g_callback(layer, top_experts.data(), (int)top_experts.size(), 1);
        }
    }

    // --- Tracing ---
    if (!g_trace_file) return;

    std::lock_guard<std::mutex> lock(g_trace_mutex);

    trace_record rec = {};
    rec.step     = g_trace_step;
    rec.layer    = (int32_t)layer;
    rec.is_draft = g_trace_draft;

    if (is_indices) {
        int32_t * indices = (int32_t *)data;
        for (int i = 0; i < TRACE_TOP_K && i < n_elements; i++) rec.topk[i] = indices[i];
        for (int i = n_elements; i < TRACE_TOP_K; i++) rec.topk[i] = -1;
    } else {
        float * probs = (float *)data;
        for (int i = 0; i < TRACE_TOP_K && i < n_elements; i++) rec.weights[i] = probs[i];
        for (int i = 0; i < TRACE_TOP_K; i++) rec.topk[i] = -1;
    }

    fwrite(&rec, sizeof(trace_record), 1, g_trace_file);
}

void llama_router_trace_close() {
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    g_initialized = 0;
    if (g_trace_file) {
        fflush(g_trace_file);
        fclose(g_trace_file);
        g_trace_file = nullptr;
    }
    fprintf(stderr, "[router-trace] closed\n");
}
