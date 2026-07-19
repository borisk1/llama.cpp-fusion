// Hot Expert Cache — complete implementation
// Single compilation unit ensures shared global state

#include "llama-hot-cache.h"
#include "llama-graph.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <mutex>

// ===================================================================
// Global state (shared across all translation units via this .cpp)
// ===================================================================
static const int MAX_LAYERS = 512;
static const int MAX_EXPERTS = 512;

static struct {
    int n_layers;
    int n_experts;
    int n_hot;
    int warmup_tokens;
    int tokens_seen;
    bool cache_active;
    bool prefill_driven;   // LLAMA_PREFILL_DRIVEN: use prefill traces instead of warmup
    bool had_prefill;      // detected multi-token batch (prefill)
    std::vector<std::vector<int32_t>> usage;
    std::vector<std::vector<bool>> hot_mask;
    std::mutex mtx;
} g;

// ===================================================================
// Configuration from environment
// ===================================================================
static void hot_cache_load_config() {
    const char * s;
    if ((s = std::getenv("LLAMA_HOT_EXPERTS"))) g.n_hot = std::atoi(s);
    if (g.n_hot < 1) g.n_hot = 64;
    if (g.n_hot > MAX_EXPERTS) g.n_hot = MAX_EXPERTS;
    
    if ((s = std::getenv("LLAMA_WARMUP_TOKENS"))) g.warmup_tokens = std::atoi(s);
    if (g.warmup_tokens < 1) g.warmup_tokens = 1;
    
    if ((s = std::getenv("LLAMA_PREFILL_DRIVEN"))) g.prefill_driven = std::atoi(s) != 0;
}

// ===================================================================
// Cache activation (shared between warmup and prefill-driven modes)
// ===================================================================
static void hot_cache_activate_now(void) {
    if (g.cache_active) return;
    
    std::lock_guard<std::mutex> lock(g.mtx);
    if (g.cache_active) return;  // double-check
    g.cache_active = true;
    
    // Compute hot mask from accumulated usage
    for (int l = 0; l < g.n_layers; l++) {
        std::vector<std::pair<int, int>> ranked;
        for (int e = 0; e < g.n_experts; e++) {
            ranked.push_back({g.usage[l][e], e});
        }
        std::sort(ranked.begin(), ranked.end(),
            std::greater<std::pair<int, int>>());
        
        for (int i = 0; i < g.n_hot && i < (int)ranked.size(); i++) {
            g.hot_mask[l][ranked[i].second] = true;
        }
        
        if (l == 0 || l == g.n_layers - 1) {
            int cov = 0, total = 0;
            for (int i = 0; i < g.n_hot && i < (int)ranked.size(); i++) cov += ranked[i].first;
            for (auto & r : ranked) total += r.first;
            if (total > 0) std::fprintf(stderr, "[hot-cache] layer %d: top-%d cover %d/%d (%.1f%%)\n",
                l, g.n_hot, cov, total, 100.0*cov/total);
        }
    }
    
    std::fprintf(stderr, "[hot-cache] activated with %d hot experts per layer\n", g.n_hot);
}

// ===================================================================
// Public API implementation
// ===================================================================

extern "C" {

void hot_cache_init(int n_layers, int n_experts) {
    if (g.n_layers > 0) return; // already initialized
    
    g.n_layers = n_layers;
    g.n_experts = n_experts;
    g.cache_active = false;
    g.tokens_seen = 0;
    
    hot_cache_load_config();
    
    g.usage.resize(n_layers);
    g.hot_mask.resize(n_layers);
    for (int l = 0; l < n_layers; l++) {
        g.usage[l].resize(n_experts, 0);
        g.hot_mask[l].resize(n_experts, false);
    }
    
    std::fprintf(stderr, "[hot-cache] init: %d layers, %d experts, %d hot, %d warmup tokens\n",
        n_layers, n_experts, g.n_hot, g.warmup_tokens);
}

void hot_cache_record(int layer, const int * expert_indices, int n_active) {
    if (g.cache_active) return;
    if (layer < 0 || layer >= g.n_layers) return;
    
    bool need_activate = false;

    {
        std::lock_guard<std::mutex> lock(g.mtx);
        
        for (int i = 0; i < n_active; i++) {
            int e = expert_indices[i];
            if (e >= 0 && e < g.n_experts) {
                g.usage[layer][e]++;
            }
        }
        
        g.tokens_seen++;
        
        // Check warmup: each call = one decode step (called once per step per layer)
        // We need warmup_tokens total steps across all layers
        int steps_approx = g.tokens_seen / g.n_layers;
        if (steps_approx >= g.warmup_tokens && !g.cache_active) {
            need_activate = true;
        }
    }

    // Activate outside the lock to avoid deadlock with hot_cache_activate_now()
    if (need_activate) {
        std::fprintf(stderr, "[hot-cache] warmup complete (%d tokens across all layers)\n", g.tokens_seen);
        hot_cache_activate_now();
        
        // Dump JSON
        const char * dump = std::getenv("LLAMA_HOT_EXPERTS_DUMP");
        if (dump) {
            FILE * f = std::fopen(dump, "w");
            if (f) {
                std::fprintf(f, "{\"n_layers\":%d,\"n_experts\":%d,\"n_hot\":%d,\"warmup_tokens\":%d,\"tokens_seen\":%d,\"layers\":{",
                    g.n_layers, g.n_experts, g.n_hot, g.warmup_tokens, g.tokens_seen);
                for (int l = 0; l < g.n_layers; l++) {
                    std::fprintf(f, "\"%d\":[", l);
                    int cnt = 0;
                    for (int e = 0; e < g.n_experts; e++) {
                        if (g.hot_mask[l][e]) {
                            if (cnt++) std::fprintf(f, ",");
                            std::fprintf(f, "%d", e);
                        }
                    }
                    std::fprintf(f, "]%s", l < g.n_layers - 1 ? "," : "");
                }
                std::fprintf(f, "},\"counts\":{");
                for (int l = 0; l < g.n_layers; l++) {
                    std::fprintf(f, "\"%d\":[", l);
                    for (int e = 0; e < g.n_experts; e++) {
                        std::fprintf(f, "%d%s", g.usage[l][e], e < g.n_experts - 1 ? "," : "");
                    }
                    std::fprintf(f, "]%s", l < g.n_layers - 1 ? "," : "");
                }
                std::fprintf(f, "}}\n");
                std::fclose(f);
                std::fprintf(stderr, "[hot-cache] dumped to %s\n", dump);
            }
        }
    }
}

bool hot_cache_is_hot(int layer, int expert) {
    if (!g.cache_active) return false;
    if (layer < 0 || layer >= g.n_layers) return false;
    if (expert < 0 || expert >= g.n_experts) return false;
    return g.hot_mask[layer][expert];
}

int hot_cache_get_n_hot() { return g.n_hot; }
bool hot_cache_is_active() { return g.cache_active; }
int hot_cache_get_n_experts() { return g.n_experts; }

int hot_cache_capture_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = (std::getenv("LLAMA_HOT_EXPERTS") ||
                  std::getenv("LLAMA_HOT_EXPERTS_DUMP") ||
                  std::getenv("LLAMA_MTP_HITRATE")) ? 1 : 0;
        if (cached) std::fprintf(stderr, "[hot-cache] capture enabled (routing tensors marked as graph outputs)\n");
    }
    return cached;
}

// ===================================================================
// MTP hit-rate measurement (paper §5.1)
// Predicted set = top-K experts from the MTP draft layer's router.
// Scored vs real top-8 of every main-model layer on the next decode step.
// Naive baseline = previous token's top-8 per layer.
// Enable: LLAMA_MTP_HITRATE=1, K via LLAMA_MTP_TOPK (default 16),
// dump via LLAMA_MTP_HITRATE_DUMP=/path.json (rewritten periodically).
// ===================================================================
static struct {
    int  enabled = -1;      // lazy init: -1 unknown, 0 off, 1 on
    int  K = 16;
    std::vector<int> pred;             // last MTP-layer top-K prediction (control signal)
    bool have_pred = false;
    // S2: per-layer top-K routing of the DRAFT token (tok1) from the previous verify batch
    std::vector<std::vector<int>> predL;
    long pred_serial = -10;            // main-batch serial when predL was written
    long cur_serial  = 0;              // main-batch serial counter
    std::vector<long> hits8, hits12, hits16, slots;   // per-layer, draft-routing predictor
    std::vector<long> hits_mtpl, slots_mtpl;          // MTP-layer cross-layer control
    std::vector<std::vector<int>> prev_top;           // per-layer previous committed top-8
    std::vector<long> hits_naive, slots_naive;        // per-layer naive baseline (committed lag)
    long steps_scored = 0;
    // pairwise slot-correlation diagnostic: prev-batch token i -> cur-batch token j
    std::vector<std::vector<std::vector<int>>> prev_batch; // [layer][tok][8]
    long pair_hits[4][4] = {{0}};
    long pair_slots[4][4] = {{0}};
    // ---- C1: acceptance-conditioned routing match (RETEST_MTP_ROUTING.md) ----
    // Store pos/token/top-8 of ALL slots of the previous decode/verify batch.
    // When the next batch's committed token (tok=0) lands on a position that was
    // a draft slot in the previous batch, compare its fresh routing vs the routing
    // that position had with the draft token:
    //   - token identical  -> "same" bucket (re-eval determinism control, expect ~100%)
    //   - token differs    -> "diff" bucket (rejected-draft substitution: context-vs-token signal)
    // Reddit "78%" reconstruction: acc*1.0 + (1-acc)*diff_hit8 ~= reported number.
    std::vector<int32_t> c1_prev_pos, c1_prev_tok;            // [tok]
    std::vector<std::vector<std::array<int,8>>> c1_prev_top;  // [layer][tok]
    std::vector<int32_t> c1_cur_pos, c1_cur_tok;
    std::vector<std::vector<std::array<int,8>>> c1_cur_top;
    long c1_prev_serial = -10;
    bool c1_active = false;      // current batch eligible for C1 capture
    int  c1_j = -1;              // matched prev-batch slot for tok=0 of current batch
    bool c1_same = false;        // token id identical on the matched slot
    std::vector<long> c1_hits_diff, c1_slots_diff;  // per-layer substitution overlap
    std::vector<long> c1_hits_same, c1_slots_same;  // per-layer same-token re-eval overlap
    long c1_events_diff = 0, c1_events_same = 0, c1_events_nomatch = 0;
    // recycled-buffer guard counters (topk reads certified vs rejected)
    long reads_valid = 0, reads_reject = 0;
} mtpstat;

static void mtp_hitrate_lazy_init() {
    if (mtpstat.enabled != -1) return;
    const char * s = std::getenv("LLAMA_MTP_HITRATE");
    mtpstat.enabled = (s && atoi(s) != 0) ? 1 : 0;
    if (!mtpstat.enabled) return;
    if ((s = std::getenv("LLAMA_MTP_TOPK"))) mtpstat.K = atoi(s);
    if (mtpstat.K < 8) mtpstat.K = 8;
    if (mtpstat.K > 64) mtpstat.K = 64;
    mtpstat.hits8.assign(g.n_layers, 0);
    mtpstat.hits12.assign(g.n_layers, 0);
    mtpstat.hits16.assign(g.n_layers, 0);
    mtpstat.slots.assign(g.n_layers, 0);
    mtpstat.hits_mtpl.assign(g.n_layers, 0);
    mtpstat.slots_mtpl.assign(g.n_layers, 0);
    mtpstat.predL.assign(g.n_layers, {});
    mtpstat.prev_top.assign(g.n_layers, {});
    mtpstat.hits_naive.assign(g.n_layers, 0);
    mtpstat.slots_naive.assign(g.n_layers, 0);
    mtpstat.prev_batch.assign(g.n_layers, {});
    mtpstat.c1_hits_diff.assign(g.n_layers, 0);
    mtpstat.c1_slots_diff.assign(g.n_layers, 0);
    mtpstat.c1_hits_same.assign(g.n_layers, 0);
    mtpstat.c1_slots_same.assign(g.n_layers, 0);
    std::fprintf(stderr, "[mtp-hitrate] enabled, K=%d\n", mtpstat.K);
}

static void mtp_hitrate_dump() {
    const char * path = std::getenv("LLAMA_MTP_HITRATE_DUMP");
    if (!path) return;
    FILE * f = std::fopen(path, "w");
    if (!f) return;
    long h8=0,h12=0,h16=0,sl=0,hn=0,sn=0,hm=0,sm=0;
    for (int l = 0; l < g.n_layers; l++) {
        h8+=mtpstat.hits8[l]; h12+=mtpstat.hits12[l]; h16+=mtpstat.hits16[l]; sl+=mtpstat.slots[l];
        hn+=mtpstat.hits_naive[l]; sn+=mtpstat.slots_naive[l];
        hm+=mtpstat.hits_mtpl[l]; sm+=mtpstat.slots_mtpl[l];
    }
    std::fprintf(f, "{\"steps_scored\":%ld,\"K\":%d,", mtpstat.steps_scored, mtpstat.K);
    std::fprintf(f, "\"global\":{\"draft_hit8\":%.4f,\"draft_hit12\":%.4f,\"draft_hit16\":%.4f,\"naive_hit8\":%.4f,\"mtplayer_hit8\":%.4f,\"slots\":%ld},",
        sl?(double)h8/sl:0, sl?(double)h12/sl:0, sl?(double)h16/sl:0, sn?(double)hn/sn:0, sm?(double)hm/sm:0, sl);
    std::fprintf(f, "\"per_layer_hit8\":[");
    for (int l = 0; l < g.n_layers; l++)
        std::fprintf(f, "%.4f%s", mtpstat.slots[l]?(double)mtpstat.hits8[l]/mtpstat.slots[l]:0, l<g.n_layers-1?",":"");
    std::fprintf(f, "],\"per_layer_hit16\":[");
    for (int l = 0; l < g.n_layers; l++)
        std::fprintf(f, "%.4f%s", mtpstat.slots[l]?(double)mtpstat.hits16[l]/mtpstat.slots[l]:0, l<g.n_layers-1?",":"");
    std::fprintf(f, "],\"per_layer_naive8\":[");
    for (int l = 0; l < g.n_layers; l++)
        std::fprintf(f, "%.4f%s", mtpstat.slots_naive[l]?(double)mtpstat.hits_naive[l]/mtpstat.slots_naive[l]:0, l<g.n_layers-1?",":"");
    std::fprintf(f, "],\"pair_matrix\":[");
    for (int i = 0; i < 4; i++) {
        std::fprintf(f, "[");
        for (int j = 0; j < 4; j++)
            std::fprintf(f, "%.4f%s", mtpstat.pair_slots[i][j]?(double)mtpstat.pair_hits[i][j]/mtpstat.pair_slots[i][j]:0, j<3?",":"");
        std::fprintf(f, "]%s", i<3?",":"");
    }
    std::fprintf(f, "],");
    // C1: acceptance-conditioned routing match
    long cd=0,csd=0,cs=0,css=0;
    for (int l = 0; l < g.n_layers; l++) {
        cd+=mtpstat.c1_hits_diff[l]; csd+=mtpstat.c1_slots_diff[l];
        cs+=mtpstat.c1_hits_same[l]; css+=mtpstat.c1_slots_same[l];
    }
    std::fprintf(f, "\"c1\":{\"events_diff\":%ld,\"events_same\":%ld,\"events_nomatch\":%ld,",
        mtpstat.c1_events_diff, mtpstat.c1_events_same, mtpstat.c1_events_nomatch);
    std::fprintf(f, "\"reads_valid\":%ld,\"reads_reject\":%ld,",
        mtpstat.reads_valid, mtpstat.reads_reject);
    std::fprintf(f, "\"diff_hit8\":%.4f,\"same_hit8\":%.4f,\"diff_slots\":%ld,\"same_slots\":%ld,",
        csd?(double)cd/csd:0, css?(double)cs/css:0, csd, css);
    std::fprintf(f, "\"per_layer_diff8\":[");
    for (int l = 0; l < g.n_layers; l++)
        std::fprintf(f, "%.4f%s", mtpstat.c1_slots_diff[l]?(double)mtpstat.c1_hits_diff[l]/mtpstat.c1_slots_diff[l]:0, l<g.n_layers-1?",":"");
    std::fprintf(f, "],\"per_layer_same8\":[");
    for (int l = 0; l < g.n_layers; l++)
        std::fprintf(f, "%.4f%s", mtpstat.c1_slots_same[l]?(double)mtpstat.c1_hits_same[l]/mtpstat.c1_slots_same[l]:0, l<g.n_layers-1?",":"");
    std::fprintf(f, "]}}\n");
    std::fclose(f);
}

// top-N expert indices from a probs row (host-side partial sort)
static void topn_from_probs(const float * probs, int n_experts, int N, int * out) {
    std::vector<std::pair<float,int>> ranked(n_experts);
    for (int e = 0; e < n_experts; e++) ranked[e] = {probs[e], e};
    if (N > n_experts) N = n_experts;
    std::partial_sort(ranked.begin(), ranked.begin()+N, ranked.end(),
        [](const std::pair<float,int> & a, const std::pair<float,int> & b){ return a.first > b.first; });
    for (int i = 0; i < N; i++) out[i] = ranked[i].second;
}

// Cross-validation of a topk read against an independently-read selprobs row.
// ggml-alloc buffer recycling can silently corrupt EITHER tensor after full-graph
// compute (set_output protection proved insufficient on the hot-resident branch).
// The probability that two independent recycled reads AGREE on the same top-8 set
// is negligible, so agreement certifies both. Requires: ids unique, in-range, and
// >= nsel-1 of them present in the host-side top-nsel of probs (1 slack for FP ties).
static bool topk_agrees_selprobs(const int32_t * ids, int nsel, const float * probs, int n_experts) {
    for (int a = 0; a < nsel; a++) {
        if (ids[a] < 0 || ids[a] >= n_experts) return false;
        for (int b = a + 1; b < nsel; b++) if (ids[a] == ids[b]) return false;
    }
    int host_top[16];
    int N = nsel < 16 ? nsel : 16;
    topn_from_probs(probs, n_experts, N, host_top);
    int agree = 0;
    for (int a = 0; a < nsel; a++)
        for (int b = 0; b < N; b++)
            if (ids[a] == host_top[b]) { agree++; break; }
    return agree >= nsel - 1;
}

// ===================================================================
// Graph readback integration
// ===================================================================

void hot_cache_readback_from_result(llm_graph_result * res, int is_mtp_graph,
                                    int ub_n_tokens, const int32_t * ub_pos, const int32_t * ub_token) {
    if (!res) return;
    if (g.n_layers <= 0) return;

    // ---- prefill-driven: activate cache after prefill completes (Insight 1) ----
    if (g.prefill_driven && !g.cache_active) {
        if (ub_n_tokens > 1) {
            g.had_prefill = true;  // multi-token batch = prefill
        } else if (g.had_prefill) {
            // First decode step after prefill → activate cache using prefill traces
            std::fprintf(stderr, "[hot-cache] prefill-driven activation (%d tokens across all layers)\n",
                g.tokens_seen);
            hot_cache_activate_now();
        }
    }

    // ---- hot-resident id-chain debug (LLAMA_HR_DEBUG=1) ----
    static int hr_dbg = -1;
    if (hr_dbg == -1) { const char * s = std::getenv("LLAMA_HR_DEBUG"); hr_dbg = (s && atoi(s)) ? 1 : 0; }
    if (hr_dbg == 1 && !is_mtp_graph) {
        auto * gfd = res->get_gf();
        if (gfd) {
            static int hr_dbg_count = 0;
            if (hr_dbg_count < 20) {
                int nn = ggml_graph_n_nodes(gfd);
                int32_t topk[8] = {0}, hids[8] = {0}, cids[8] = {0};
                bool have_t = false, have_h = false, have_c = false;
                for (int i = 0; i < nn; i++) {
                    ggml_tensor * t = ggml_graph_node(gfd, i);
                    const char * nm = ggml_get_name(t);
                    if (!nm) continue;
                    if (!have_t && strcmp(nm, "ffn_moe_topk-0") == 0 && t->ne[1] <= 4) {
                        for (int s2 = 0; s2 < 8 && s2 < (int)t->ne[0]; s2++)
                            ggml_backend_tensor_get(t, &topk[s2], (size_t)s2*t->nb[0], 4);
                        have_t = true;
                    }
                    if (!have_h && strcmp(nm, "hr_hot_ids-0") == 0 && t->ne[1] <= 4) {
                        ggml_backend_tensor_get(t, hids, 0, 8*4); have_h = true;
                    }
                    if (!have_c && strcmp(nm, "hr_cold_ids-0") == 0 && t->ne[1] <= 4) {
                        ggml_backend_tensor_get(t, cids, 0, 8*4); have_c = true;
                    }
                }
                if (have_t || have_h || have_c) {
                    hr_dbg_count++;
                    std::fprintf(stderr, "[hr-dbg] topk0=[%d,%d,%d,%d,%d,%d,%d,%d] hot=[%d,%d,%d,%d,%d,%d,%d,%d] cold=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                        topk[0],topk[1],topk[2],topk[3],topk[4],topk[5],topk[6],topk[7],
                        hids[0],hids[1],hids[2],hids[3],hids[4],hids[5],hids[6],hids[7],
                        cids[0],cids[1],cids[2],cids[3],cids[4],cids[5],cids[6],cids[7]);
                }
                // also dump hot/cold down outputs for layer 0
                float d_hot[8]={0}, d_cold[8]={0}, d_ref[8]={0};
                bool have_dh=false, have_dc=false, have_dr=false;
                for (int i = 0; i < nn; i++) {
                    ggml_tensor * t = ggml_graph_node(gfd, i);
                    const char * nm = ggml_get_name(t);
                    if (!nm) continue;
                    if (!have_dh && strcmp(nm, "ffn_moe_down_hot-0") == 0) {
                        ggml_backend_tensor_get(t, d_hot, 0, 8*4); have_dh = true;
                    }
                    if (!have_dc && strcmp(nm, "ffn_moe_down_cold-0") == 0) {
                        ggml_backend_tensor_get(t, d_cold, 0, 8*4); have_dc = true;
                    }
                    if (!have_dr && strcmp(nm, "ffn_moe_down-0") == 0) {
                        ggml_backend_tensor_get(t, d_ref, 0, 8*4); have_dr = true;
                    }
                }
                // also dump gate_hot to see where zeros originate
                float g_hot[8]={0}, u_hot[8]={0};
                bool have_gh=false, have_uh=false;
                for (int i = 0; i < nn; i++) {
                    ggml_tensor * t = ggml_graph_node(gfd, i);
                    const char * nm = ggml_get_name(t);
                    if (!nm) continue;
                    if (!have_gh && strcmp(nm, "ffn_moe_gate_hot-0") == 0) {
                        ggml_backend_tensor_get(t, g_hot, 0, 8*4); have_gh = true;
                    }
                    if (!have_uh && strcmp(nm, "ffn_moe_up_hot-0") == 0) {
                        ggml_backend_tensor_get(t, u_hot, 0, 8*4); have_uh = true;
                    }
                }
                if (have_gh || have_uh) {
                    std::fprintf(stderr, "[hr-dbg] L0 gate_hot[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        g_hot[0],g_hot[1],g_hot[2],g_hot[3],g_hot[4],g_hot[5],g_hot[6],g_hot[7]);
                    std::fprintf(stderr, "[hr-dbg] L0 up_hot[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        u_hot[0],u_hot[1],u_hot[2],u_hot[3],u_hot[4],u_hot[5],u_hot[6],u_hot[7]);
                }
                if (have_dh || have_dc || have_dr) {
                    std::fprintf(stderr, "[hr-dbg] L0 down_hot[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        d_hot[0],d_hot[1],d_hot[2],d_hot[3],d_hot[4],d_hot[5],d_hot[6],d_hot[7]);
                    std::fprintf(stderr, "[hr-dbg] L0 down_cold[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        d_cold[0],d_cold[1],d_cold[2],d_cold[3],d_cold[4],d_cold[5],d_cold[6],d_cold[7]);
                    std::fprintf(stderr, "[hr-dbg] L0 down_ref[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        d_ref[0],d_ref[1],d_ref[2],d_ref[3],d_ref[4],d_ref[5],d_ref[6],d_ref[7]);
                    float sum[8], max_e=0;
                    for (int j=0;j<8;j++) { sum[j]=d_hot[j]+d_cold[j]; float e=sum[j]-d_ref[j]; if(e<0)e=-e; if(e>max_e)max_e=e; }
                    std::fprintf(stderr, "[hr-dbg] L0 sum[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f max_err=%.2e\n",
                        sum[0],sum[1],sum[2],sum[3],sum[4],sum[5],sum[6],sum[7], max_e);
                }
                // per-layer ffn_moe_down[0] fingerprint: locate the first
                // diverging layer between dual-chain and reference runs
                {
                    char pbuf[4096]; int off = 0;
                    off += snprintf(pbuf+off, sizeof(pbuf)-off, "[hr-dbg] down0:");
                    for (int i = 0; i < nn && off < (int)sizeof(pbuf)-24; i++) {
                        ggml_tensor * t = ggml_graph_node(gfd, i);
                        const char * nm = ggml_get_name(t);
                        if (!nm || strncmp(nm, "ffn_moe_down-", 13) != 0) continue;
                        if (strchr(nm + 13, ' ') || strchr(nm + 13, '(')) continue; // skip derived names
                        float v = 0; ggml_backend_tensor_get(t, &v, 0, 4);
                        off += snprintf(pbuf+off, sizeof(pbuf)-off, " %s:%.4f", nm + 13, v);
                    }
                    std::fprintf(stderr, "%s\n", pbuf);
                }
            }
        }
    }

    if (!hot_cache_capture_enabled()) return;   // avoid GPU readback syncs when profiling is off

    auto * gf = res->get_gf();
    if (!gf) return;

    mtp_hitrate_lazy_init();

    if (!is_mtp_graph && mtpstat.enabled == 1) {
        mtpstat.cur_serial++;   // one serial per main-graph batch; gaps invalidate stale predictions
    }

    // ---- C1 prep: decide eligibility + match tok=0 against previous batch slots ----
    // Verify batch layout (server spec path): [sampled@P, draft0@P+1, ..., draftk@P+k].
    // If some draft was rejected, the next batch's committed tok=0 lands on a position
    // that was a draft slot in this batch (same pos, different token). If all accepted,
    // tok=0 moves to a brand-new position (nomatch).
    mtpstat.c1_active = false;
    mtpstat.c1_j = -1;
    mtpstat.c1_same = false;
    if (mtpstat.enabled == 1 && !is_mtp_graph &&
        ub_n_tokens >= 1 && ub_n_tokens <= 4 && ub_pos && ub_token) {
        mtpstat.c1_active = true;
        // score target: committed tok=0 vs previous-batch slot at the same position
        if (mtpstat.c1_prev_serial == mtpstat.cur_serial - 1 && !mtpstat.c1_prev_pos.empty()) {
            for (size_t j = 0; j < mtpstat.c1_prev_pos.size(); j++) {
                if (mtpstat.c1_prev_pos[j] == ub_pos[0]) {
                    mtpstat.c1_j = (int)j;
                    mtpstat.c1_same = (mtpstat.c1_prev_tok[j] == ub_token[0]);
                    break;
                }
            }
            if (mtpstat.c1_j > 0) {  // j==0 (same committed pos) shouldn't happen; count draft slots only
                if (mtpstat.c1_same) mtpstat.c1_events_same++; else mtpstat.c1_events_diff++;
                if ((mtpstat.c1_events_diff + mtpstat.c1_events_same) % 25 == 0) mtp_hitrate_dump();
            } else if (mtpstat.c1_j == 0) {
                mtpstat.c1_j = -1;   // anomaly: do not score against committed slot
            } else if (mtpstat.c1_prev_pos.size() > 1) {
                mtpstat.c1_events_nomatch++;   // spec batch preceded, but all drafts accepted
            }
        }
        // prepare capture scratch for this batch
        mtpstat.c1_cur_pos.assign(ub_pos, ub_pos + ub_n_tokens);
        mtpstat.c1_cur_tok.assign(ub_token, ub_token + ub_n_tokens);
        std::array<int,8> sentinel; sentinel.fill(-1);
        mtpstat.c1_cur_top.assign(g.n_layers, std::vector<std::array<int,8>>(ub_n_tokens, sentinel));
    }

    static int scan_count = 0;
    scan_count++;

    const char * dbg_env = std::getenv("LLAMA_MTP_DEBUG");
    const int dbg = (dbg_env && atoi(dbg_env) != 0) ? 1 : 0;

    int n_nodes = ggml_graph_n_nodes(gf);

    int dbg_selprobs = 0, dbg_topk = 0;

    // Two passes: FIRST score all exact-selection (topk) nodes against the
    // predictions captured from the PREVIOUS batch, THEN update predictions
    // from this batch's selprobs nodes. Within the graph, selprobs-N precedes
    // topk-N, so a single pass would overwrite predictions before scoring.
    std::vector<ggml_tensor *> topk_nodes, selprobs_nodes;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        const char * name = ggml_get_name(t);
        if (!name) continue;
        if (strstr(name, "ffn_moe_selprobs")) { selprobs_nodes.push_back(t); dbg_selprobs++; }
        else if (strstr(name, "ffn_moe_topk")) { topk_nodes.push_back(t); dbg_topk++; }
    }

    // ---------- PASS 0: read selprobs candidates up-front (decode batches) ----------
    // Used to cross-validate every topk read (see topk_agrees_selprobs). There can
    // be >1 node whose name contains "ffn_moe_selprobs-N" per layer — keep all
    // candidates; agreement with any of them certifies the topk read.
    std::vector<std::vector<std::vector<float>>> sp_cand;   // [layer][cand][n_experts*n_tokens]
    std::vector<int> sp_nexp;
    if (!is_mtp_graph) {
        sp_cand.assign(g.n_layers, {});
        sp_nexp.assign(g.n_layers, 0);
        for (ggml_tensor * t : selprobs_nodes) {
            const char * name = ggml_get_name(t);
            const char * dash = strrchr(name, '-');
            if (!dash) continue;
            int layer = atoi(dash + 1);
            if (layer < 0 || layer >= g.n_layers) continue;
            if (t->type != GGML_TYPE_F32) continue;
            const int n_experts = (int)t->ne[0];
            const int n_tokens  = (int)t->ne[1];
            const int n_elements = n_experts * n_tokens;
            if (n_tokens > 4 || n_elements <= 0 || n_elements > 65536) continue;
            std::vector<float> fbuf(n_elements);
            bool sp_ok = true;
            for (int tok = 0; tok < n_tokens && sp_ok; tok++) {   // stride-safe (view-proof) read
                const size_t row_off = (size_t)tok * t->nb[1];
                if (t->buffer) {
                    ggml_backend_tensor_get(t, fbuf.data() + (size_t)tok * n_experts,
                                            row_off, (size_t)n_experts * sizeof(float));
                } else if (t->data) {
                    std::memcpy(fbuf.data() + (size_t)tok * n_experts,
                                (const char *)t->data + row_off, (size_t)n_experts * sizeof(float));
                } else {
                    sp_ok = false;
                }
            }
            if (!sp_ok) continue;
            sp_nexp[layer] = n_experts;
            sp_cand[layer].push_back(std::move(fbuf));
        }
    }

    // ---------- PASS 1: exact selected experts (I32 [n_expert_used, n_tokens]) ----------
    static int tkdbg = -1;
    if (tkdbg == -1) { const char * s = std::getenv("LLAMA_TOPK_DEBUG"); tkdbg = (s && atoi(s)) ? 1 : 0; }
    static int tkdbg_scans = 0;
    if (tkdbg) tkdbg_scans++;
    for (ggml_tensor * t : topk_nodes) {
        const char * name = ggml_get_name(t);

        const char * dash = strrchr(name, '-');
        if (!dash) continue;
        int layer = atoi(dash + 1);

        const int n_tokens = (int)t->ne[1];

        if (tkdbg && tkdbg_scans <= 4 && !is_mtp_graph) {
            std::fprintf(stderr, "[topk-dbg] scan%d name='%s' L%d type=%d ne=[%d,%d] nb1=%zu buf=%p(%s) data=%p src0=%s\n",
                tkdbg_scans, name, layer, (int)t->type, (int)t->ne[0], (int)t->ne[1], t->nb[1],
                (void*)t->buffer, t->buffer ? ggml_backend_buffer_name(t->buffer) : "null",
                t->data, t->src[0] ? ggml_get_name(t->src[0]) : "null");
        }

        if (is_mtp_graph) continue;               // MTP layer never feeds usage stats
        if (layer < 0 || layer >= g.n_layers) continue;
        if (t->type != GGML_TYPE_I32) continue;

        const int n_used = (int)t->ne[0];
        const int n_elements = n_used * n_tokens;
        if (n_used <= 0 || n_used > 16 || n_elements <= 0 || n_elements > 65536) continue;

        // ffn_moe_topk is a strided view ([k, n_tokens] over [n_expert, n_tokens]
        // argsort output) — read row by row honoring nb[1]
        std::vector<int32_t> ids(n_elements);
        bool read_ok = true;
        for (int tok = 0; tok < n_tokens && read_ok; tok++) {
            const size_t row_off = (size_t)tok * t->nb[1];
            if (t->buffer) {
                ggml_backend_tensor_get(t, ids.data() + (size_t)tok * n_used,
                                        row_off, (size_t)n_used * sizeof(int32_t));
            } else if (t->data) {
                std::memcpy(ids.data() + (size_t)tok * n_used,
                            (const char *)t->data + row_off, (size_t)n_used * sizeof(int32_t));
            } else {
                read_ok = false;
            }
        }
        if (!read_ok) continue;

        std::vector<std::vector<int>> cur_tops;   // per-token exact selection, for pairwise diagnostic
        for (int tok = 0; tok < n_tokens; tok++) {
            const int32_t * sel = ids.data() + (size_t)tok * n_used;
            int top[16];
            int nsel = n_used < 16 ? n_used : 16;
            for (int a = 0; a < nsel; a++) top[a] = (int)sel[a];

            // ---- recycled-buffer guard: certify this (layer, tok) read against
            // an independent selprobs read; reject silently-corrupted rows ----
            bool tok_valid = false;
            if (n_tokens <= 4 && layer < (int)sp_cand.size()) {
                for (const auto & cand : sp_cand[layer]) {
                    if ((int)cand.size() != sp_nexp[layer] * n_tokens) continue;
                    if (topk_agrees_selprobs(sel, nsel, cand.data() + (size_t)tok * sp_nexp[layer], sp_nexp[layer])) {
                        tok_valid = true;
                        break;
                    }
                }
            }
            if (mtpstat.enabled == 1 && n_tokens <= 4) {
                if (tok_valid) mtpstat.reads_valid++; else mtpstat.reads_reject++;
                cur_tops.emplace_back();                       // slot for EVERY tok (keeps indexing)
                if (tok_valid) cur_tops.back().assign(top, top + nsel);
            }
            if (!tok_valid) continue;

            if (dbg && layer == 0 && n_tokens <= 4) {
                std::fprintf(stderr, "[dbg] serial=%ld n_tok=%d tok=%d L0 sel=[%d,%d,%d,%d,...] pred_serial=%ld predL0=%zu en=%d\n",
                    mtpstat.cur_serial, n_tokens, tok, top[0], top[1], top[2], top[3],
                    mtpstat.pred_serial, mtpstat.predL.empty() ? (size_t)0 : mtpstat.predL[0].size(), mtpstat.enabled);
            }

            hot_cache_record(layer, top, nsel);

            // ---- C1: capture routing of ALL batch slots; score tok=0 vs matched prev draft slot ----
            if (mtpstat.c1_active && nsel >= 8 && tok < (int)mtpstat.c1_cur_pos.size()) {
                auto & dst = mtpstat.c1_cur_top[layer][tok];
                for (int a = 0; a < 8; a++) dst[a] = top[a];
                if (tok == 0 && mtpstat.c1_j >= 0 &&
                    layer < (int)mtpstat.c1_prev_top.size() &&
                    mtpstat.c1_j < (int)mtpstat.c1_prev_top[layer].size()) {
                    const auto & prev8 = mtpstat.c1_prev_top[layer][mtpstat.c1_j];
                    if (prev8[0] >= 0) {
                        int h = 0;
                        for (int a = 0; a < 8; a++)
                            for (int b = 0; b < 8; b++)
                                if (top[a] == prev8[b]) { h++; break; }
                        if (mtpstat.c1_same) { mtpstat.c1_hits_same[layer] += h; mtpstat.c1_slots_same[layer] += 8; }
                        else                 { mtpstat.c1_hits_diff[layer] += h; mtpstat.c1_slots_diff[layer] += 8; }
                    }
                }
            }

            // ---- hit-rate scoring: decode/verify batches only, first token = committed ----
            if (mtpstat.enabled == 1 && n_tokens <= 4 && tok == 0 && nsel >= 8) {
                // naive committed-lag baseline
                if (!mtpstat.prev_top[layer].empty()) {
                    int hn = 0;
                    for (int a = 0; a < 8; a++)
                        for (int b = 0; b < 8 && b < (int)mtpstat.prev_top[layer].size(); b++)
                            if (top[a] == mtpstat.prev_top[layer][b]) { hn++; break; }
                    mtpstat.hits_naive[layer] += hn;
                    mtpstat.slots_naive[layer] += 8;
                }
                mtpstat.prev_top[layer].assign(top, top + 8);

                // S2: last-token selection-prob predictor from the immediately preceding batch
                if (mtpstat.pred_serial == mtpstat.cur_serial - 1 &&
                    (int)mtpstat.predL[layer].size() >= 8) {
                    const auto & P = mtpstat.predL[layer];
                    int h8 = 0, h12 = 0, h16 = 0;
                    for (int a = 0; a < 8; a++) {
                        for (int b = 0; b < (int)P.size(); b++) {
                            if (top[a] == P[b]) {
                                if (b < 8)  h8++;
                                if (b < 12) h12++;
                                if (b < 16) h16++;
                                break;
                            }
                        }
                    }
                    mtpstat.hits8[layer]  += h8;
                    mtpstat.hits12[layer] += h12;
                    mtpstat.hits16[layer] += h16;
                    mtpstat.slots[layer]  += 8;
                    if (layer == g.n_layers - 1) {
                        mtpstat.steps_scored++;
                        if (mtpstat.steps_scored % 25 == 0) mtp_hitrate_dump();
                    }
                }

                // S3 control: MTP-layer router prediction (cross-layer)
                if (mtpstat.have_pred) {
                    int hm = 0;
                    for (int a = 0; a < 8; a++)
                        for (int b = 0; b < (int)mtpstat.pred.size() && b < 8; b++)
                            if (top[a] == mtpstat.pred[b]) { hm++; break; }
                    mtpstat.hits_mtpl[layer] += hm;
                    mtpstat.slots_mtpl[layer] += 8;
                }
            }
        }

        // pairwise slot-correlation diagnostic: prev-batch token i vs cur-batch token j
        if (mtpstat.enabled == 1 && !cur_tops.empty()) {
            auto & prev = mtpstat.prev_batch[layer];
            if (!prev.empty()) {
                int ni = (int)prev.size() < 4 ? (int)prev.size() : 4;
                int nj = (int)cur_tops.size() < 4 ? (int)cur_tops.size() : 4;
                for (int i2 = 0; i2 < ni; i2++) {
                    for (int j = 0; j < nj; j++) {
                        if (prev[i2].empty() || cur_tops[j].empty()) continue;  // rejected reads
                        int h = 0;
                        for (size_t a = 0; a < cur_tops[j].size() && a < 8; a++)
                            for (size_t b = 0; b < prev[i2].size() && b < 8; b++)
                                if (cur_tops[j][a] == prev[i2][b]) { h++; break; }
                        mtpstat.pair_hits[i2][j] += h;
                        mtpstat.pair_slots[i2][j] += 8;
                    }
                }
            }
            prev = cur_tops;
        }
    }

    // ---------- PASS 2: selection probs — capture predictions for the NEXT batch ----------
    if (mtpstat.enabled == 1) {
        for (ggml_tensor * t : selprobs_nodes) {
            const char * name = ggml_get_name(t);
            const char * dash = strrchr(name, '-');
            if (!dash) continue;
            int layer = atoi(dash + 1);

            const int n_experts = (int)t->ne[0];
            const int n_tokens  = (int)t->ne[1];
            const int n_elements = n_experts * n_tokens;
            if (n_elements <= 0 || n_elements > 65536) continue;

            std::vector<float> fbuf(n_elements);
            if (t->buffer) {
                ggml_backend_tensor_get(t, fbuf.data(), 0, (size_t)n_elements * sizeof(float));
            } else if (t->data) {
                std::memcpy(fbuf.data(), t->data, (size_t)n_elements * sizeof(float));
            } else {
                continue;
            }

            if (is_mtp_graph) {
                // MTP draft layer router: cross-layer control signal
                mtpstat.pred.resize(mtpstat.K);
                topn_from_probs(fbuf.data() + (size_t)(n_tokens - 1) * n_experts,
                                n_experts, mtpstat.K, mtpstat.pred.data());
                mtpstat.have_pred = true;
            } else if (layer >= 0 && layer < g.n_layers && n_tokens <= 4) {
                // S2 predictor: top-K selection candidates of the LAST token in this batch.
                // Recycled-buffer guard: only use a selprobs read that was certified in
                // PASS 1 (its host top-8 agreed with the independently-read topk of the
                // same layer — i.e. the layer produced at least one valid tok this batch).
                bool certified = mtpstat.c1_active &&
                    layer < (int)mtpstat.c1_cur_top.size() &&
                    !mtpstat.c1_cur_top[layer].empty() &&
                    mtpstat.c1_cur_top[layer].back()[0] >= 0;
                if (certified) {
                    mtpstat.predL[layer].resize(mtpstat.K);
                    topn_from_probs(fbuf.data() + (size_t)(n_tokens - 1) * n_experts,
                                    n_experts, mtpstat.K, mtpstat.predL[layer].data());
                } else {
                    mtpstat.predL[layer].clear();   // stale prediction must not survive
                }
                if (layer == g.n_layers - 1) {
                    mtpstat.pred_serial = mtpstat.cur_serial;
                }
            }
        }
    }

    // ---- C1 commit: current batch becomes "previous" for the next one ----
    if (mtpstat.c1_active) {
        // C4 trace: single-token decode batches only (no-spec run) — dump
        // committed token id + per-layer top-8 for offline token-identity analysis
        static FILE * c4f = (FILE *) (intptr_t) -1;
        if (c4f == (FILE *) (intptr_t) -1) {
            const char * p = std::getenv("LLAMA_C4_TRACE");
            c4f = p ? std::fopen(p, "a") : nullptr;
        }
        if (c4f && mtpstat.c1_cur_pos.size() == 1) {
            if (tkdbg) {
                static int miss_prints = 0;
                for (int l = 0; l < g.n_layers && miss_prints < 60; l++) {
                    if (mtpstat.c1_cur_top[l][0][0] < 0) {
                        bool seen = false; int seen_ne1 = -1;
                        for (ggml_tensor * tt : topk_nodes) {
                            const char * nm2 = ggml_get_name(tt);
                            const char * d2 = strrchr(nm2, '-');
                            if (d2 && atoi(d2+1) == l) { seen = true; seen_ne1 = (int)tt->ne[1]; break; }
                        }
                        std::fprintf(stderr, "[c4-miss] serial=%ld L%d node_seen=%d ne1=%d ub_n=%zu\n",
                            mtpstat.cur_serial, l, seen ? 1 : 0, seen_ne1, mtpstat.c1_cur_pos.size());
                        miss_prints++;
                    }
                }
            }
            std::fprintf(c4f, "%d %d", mtpstat.c1_cur_pos[0], mtpstat.c1_cur_tok[0]);
            for (int l = 0; l < g.n_layers; l++) {
                const auto & t8 = mtpstat.c1_cur_top[l][0];
                std::fprintf(c4f, " %d,%d,%d,%d,%d,%d,%d,%d",
                    t8[0],t8[1],t8[2],t8[3],t8[4],t8[5],t8[6],t8[7]);
            }
            std::fprintf(c4f, "\n");
        }
        mtpstat.c1_prev_pos = std::move(mtpstat.c1_cur_pos);
        mtpstat.c1_prev_tok = std::move(mtpstat.c1_cur_tok);
        mtpstat.c1_prev_top = std::move(mtpstat.c1_cur_top);
        mtpstat.c1_prev_serial = mtpstat.cur_serial;
        mtpstat.c1_active = false;
    }

    if (scan_count <= 6) {
        std::fprintf(stderr, "[hot-cache] readback scan %d (mtp=%d): %d selprobs, %d topk nodes, n_nodes=%d\n",
            scan_count, is_mtp_graph, dbg_selprobs, dbg_topk, n_nodes);
    }
}

} // extern "C"
