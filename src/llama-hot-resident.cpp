// Hot-expert VRAM residency — implementation.
// See llama-hot-resident.h for the concept.

#include "llama-hot-resident.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

// out-of-range sentinel for the cold path (skipped by patched CPU mul_mat_id)
static const float HOT_OOR_ID = 100000.0f;

struct hot_layer_reg {
    int il = -1;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * down = nullptr;
    int n_expert = 0;
    int n_expert_used = 0;
};

struct hot_layer_entry {
    ggml_tensor * gate_hot = nullptr;
    ggml_tensor * up_hot   = nullptr;
    ggml_tensor * down_hot = nullptr;
    ggml_tensor * map_hot  = nullptr;
    ggml_tensor * map_cold = nullptr;
    ggml_tensor * mask_hot = nullptr;
};

static struct {
    std::vector<hot_layer_reg> regs;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> lookup; // fused -> (entry idx, which: 0=gate 1=up 2=down)
    std::vector<hot_layer_entry> entries;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    bool finalized = false;
    int n_hot = 0;
} S;

int llama_hot_resident_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = std::getenv("LLAMA_HOT_RESIDENT") ? 1 : 0;
    }
    return cached;
}

void llama_hot_resident_register_layer(int il, ggml_tensor * gate_exps, ggml_tensor * up_exps,
                                       ggml_tensor * down_exps, int n_expert, int n_expert_used) {
    if (!llama_hot_resident_enabled()) return;
    if (!gate_exps || !up_exps || !down_exps) return;
    hot_layer_reg r;
    r.il = il; r.gate = gate_exps; r.up = up_exps; r.down = down_exps;
    r.n_expert = n_expert; r.n_expert_used = n_expert_used;
    S.regs.push_back(r);
}

// --- minimal parser for our own hot_cache.json dump ---
// extracts counts["<layer>"] = [c0, c1, ...] (preferred) or layers["<layer>"] = [ids]
static bool parse_layer_array(const std::string & text, const char * section, int layer,
                              std::vector<long> & out) {
    out.clear();
    size_t sec = text.find(std::string("\"") + section + "\"");
    if (sec == std::string::npos) return false;
    char key[32];
    snprintf(key, sizeof(key), "\"%d\":[", layer);
    size_t k = text.find(key, sec);
    if (k == std::string::npos) return false;
    size_t p = k + strlen(key);
    size_t end = text.find(']', p);
    if (end == std::string::npos) return false;
    const char * s = text.c_str() + p;
    const char * e = text.c_str() + end;
    while (s < e) {
        char * next = nullptr;
        long v = strtol(s, &next, 10);
        if (next == s) break;
        out.push_back(v);
        s = next;
        while (s < e && (*s == ',' || *s == ' ')) s++;
    }
    return !out.empty();
}

int llama_hot_resident_finalize(void) {
    if (!llama_hot_resident_enabled() || S.regs.empty() || S.finalized) {
        return S.finalized ? 1 : 0;
    }

    const char * path = std::getenv("LLAMA_HOT_RESIDENT");
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[hot-resident] cannot open %s\n", path);
        return 0;
    }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string text(sz, '\0');
    size_t rd = std::fread(text.data(), 1, sz, f);
    std::fclose(f);
    if ((long) rd != sz) {
        fprintf(stderr, "[hot-resident] short read on %s\n", path);
        return 0;
    }

    int n_hot = 64;
    if (const char * s = std::getenv("LLAMA_HOT_N")) n_hot = atoi(s);

    // resolve hot sets first; drop layers without profile data (e.g. MTP layer)
    std::vector<std::vector<int>> all_hot_ids(S.regs.size());
    {
        std::vector<hot_layer_reg> kept;
        std::vector<std::vector<int>> kept_ids;
        for (auto & r : S.regs) {
            const int NE = r.n_expert;
            const int NH = std::min(n_hot, NE);
            std::vector<long> counts;
            std::vector<int> hot_ids;
            if (parse_layer_array(text, "counts", r.il, counts) && (int) counts.size() == NE) {
                std::vector<std::pair<long,int>> ranked(NE);
                for (int x = 0; x < NE; x++) ranked[x] = {counts[x], x};
                std::partial_sort(ranked.begin(), ranked.begin() + NH, ranked.end(),
                    [](const std::pair<long,int> & a, const std::pair<long,int> & b){ return a.first > b.first; });
                for (int x = 0; x < NH; x++) hot_ids.push_back(ranked[x].second);
            } else {
                std::vector<long> lst;
                if (parse_layer_array(text, "layers", r.il, lst)) {
                    for (long v : lst) if ((int) hot_ids.size() < NH) hot_ids.push_back((int) v);
                }
            }
            if (hot_ids.empty()) {
                fprintf(stderr, "[hot-resident] no profile data for layer %d — skipping\n", r.il);
                continue;
            }
            // debug: identity mapping (slot == global id) to isolate remap bugs
            if (std::getenv("LLAMA_HOT_IDENTITY")) {
                std::sort(hot_ids.begin(), hot_ids.end());
            }
            kept.push_back(r);
            kept_ids.push_back(std::move(hot_ids));
        }
        S.regs = std::move(kept);
        all_hot_ids = std::move(kept_ids);
    }
    if (S.regs.empty()) {
        fprintf(stderr, "[hot-resident] no usable layers\n");
        return 0;
    }

    // GPU device buffer type (backend-agnostic)
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "[hot-resident] no GPU device found\n");
        return 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    const size_t n_tensors = S.regs.size() * 6;
    ggml_init_params ip = {
        /*.mem_size  =*/ ggml_tensor_overhead() * n_tensors,
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    S.ctx = ggml_init(ip);
    if (!S.ctx) return 0;

    // create tensors (metadata only)
    S.entries.resize(S.regs.size());
    for (size_t i = 0; i < S.regs.size(); i++) {
        const auto & r = S.regs[i];
        auto & e = S.entries[i];
        const int NH = (int) all_hot_ids[i].size();

        auto mk_hot = [&](ggml_tensor * fused, const char * sfx) {
            // NH real hot experts + n_expert_used zero "dummy" slots.
            // Cold slots route to dummy NH+s (s = slot index): ids stay UNIQUE
            // per token (required by CUDA mmid grouping paths) and dummy experts
            // are all-zero so no masking is needed on the hot chain.
            ggml_tensor * t = ggml_new_tensor_3d(S.ctx, fused->type, fused->ne[0], fused->ne[1],
                                                 NH + r.n_expert_used);
            ggml_format_name(t, "hot_%s-%d", sfx, r.il);
            return t;
        };
        e.gate_hot = mk_hot(r.gate, "gate");
        e.up_hot   = mk_hot(r.up,   "up");
        e.down_hot = mk_hot(r.down, "down");
        e.map_hot  = ggml_new_tensor_1d(S.ctx, GGML_TYPE_F32, r.n_expert);
        e.map_cold = ggml_new_tensor_1d(S.ctx, GGML_TYPE_F32, r.n_expert);
        e.mask_hot = ggml_new_tensor_1d(S.ctx, GGML_TYPE_F32, r.n_expert);
        ggml_format_name(e.map_hot,  "hot_map-%d",  r.il);
        ggml_format_name(e.map_cold, "cold_map-%d", r.il);
        ggml_format_name(e.mask_hot, "hot_mask-%d", r.il);
    }

    S.buf = ggml_backend_alloc_ctx_tensors_from_buft(S.ctx, buft);
    if (!S.buf) {
        fprintf(stderr, "[hot-resident] GPU buffer allocation failed\n");
        return 0;
    }
    // Mark as WEIGHTS: the scheduler must treat hot slices as immovable model
    // weights (ops using them run on THIS backend). Without this, with -ngl 0
    // (CPU-resident attention) the scheduler assigns the hot mmid to the CPU
    // backend and streams the multi-GB hot tensors GPU->CPU on every graph
    // (measured: 0.45 t/s). With the flag the mmid is pinned to the GPU and
    // only small activations cross PCIe.
    ggml_backend_buffer_set_usage(S.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // zero the whole buffer first: quant kernels may read padded regions
    ggml_backend_buffer_clear(S.buf, 0);

    // fill: copy hot expert slices, upload maps
    std::vector<char> staging;
    size_t vram_used = 0;
    for (size_t i = 0; i < S.regs.size(); i++) {
        const auto & r = S.regs[i];
        auto & e = S.entries[i];
        const int NE = r.n_expert;
        const std::vector<int> & hot_ids = all_hot_ids[i];

        // maps
        std::vector<float> map_hot(NE, 0.0f), map_cold(NE), mask(NE, 1.0f); // mask: 1=cold (default), 0=hot
        for (int g = 0; g < NE; g++) map_cold[g] = (float) g;
        for (int slot = 0; slot < (int) hot_ids.size(); slot++) {
            const int g = hot_ids[slot];
            map_hot[g]  = (float) slot;
            map_cold[g] = HOT_OOR_ID;   // hot slots → OOR: skipped+zeroed by the
                                        // patched CPU mul_mat_id (saves ~78% of
                                        // CPU expert work). Requires --no-op-offload
                                        // (sched expert-copy asserts on OOR ids).
            mask[g]     = 0.0f;   // COLD mask: 1=cold, 0=hot
        }
        ggml_backend_tensor_set(e.map_hot,  map_hot.data(),  0, NE * sizeof(float));
        ggml_backend_tensor_set(e.map_cold, map_cold.data(), 0, NE * sizeof(float));
        ggml_backend_tensor_set(e.mask_hot, mask.data(),     0, NE * sizeof(float));

        // copy expert slices (expert dim is outermost => each slice is contiguous)
        ggml_tensor * fused[3] = { r.gate, r.up, r.down };
        ggml_tensor * hot[3]   = { e.gate_hot, e.up_hot, e.down_hot };
        const bool alias_mode = std::getenv("LLAMA_HOT_ALIAS") != nullptr;
        if (alias_mode) {
            // debug: use the original fused tensors directly (no copy, no VRAM)
            e.gate_hot = r.gate; e.up_hot = r.up; e.down_hot = r.down;
        } else {
        std::vector<char> verify;
        for (int k = 0; k < 3; k++) {
            const size_t slice = fused[k]->nb[2];
            if (staging.size() < slice) staging.resize(slice);
            for (int slot = 0; slot < (int) hot_ids.size(); slot++) {
                const size_t src_off = (size_t) hot_ids[slot] * slice;
                ggml_backend_tensor_get(fused[k], staging.data(), src_off, slice);
                ggml_backend_tensor_set(hot[k], staging.data(), (size_t) slot * slice, slice);
            }
            vram_used += slice * hot_ids.size();
            // verify round-trip for slot 0
            if (i == 0 && k == 0) {
                verify.resize(slice);
                ggml_backend_tensor_get(hot[k], verify.data(), 0, slice);
                ggml_backend_tensor_get(fused[k], staging.data(), (size_t) hot_ids[0] * slice, slice);
                fprintf(stderr, "[hot-resident] slice verify layer %d: %s (slice=%zu B, expert %d -> slot 0)\n",
                    r.il, memcmp(verify.data(), staging.data(), slice) == 0 ? "OK" : "MISMATCH", slice, hot_ids[0]);
            }
        }
        }

        S.lookup[r.gate] = { (int) i, 0 };
        S.lookup[r.up]   = { (int) i, 1 };
        S.lookup[r.down] = { (int) i, 2 };
    }

    S.n_hot = n_hot;
    S.finalized = true;
    fprintf(stderr, "[hot-resident] %zu layers, %d hot experts/layer, %.1f MiB VRAM\n",
        S.regs.size(), n_hot, vram_used / 1024.0 / 1024.0);
    return 1;
}

int llama_hot_resident_get(const ggml_tensor * fused, ggml_tensor ** hot,
                           ggml_tensor ** map_hot, ggml_tensor ** map_cold, ggml_tensor ** mask_hot) {
    if (!S.finalized) return 0;
    auto it = S.lookup.find(fused);
    if (it == S.lookup.end()) return 0;
    const auto & e = S.entries[it->second.first];
    switch (it->second.second) {
        case 0: *hot = e.gate_hot; break;
        case 1: *hot = e.up_hot;   break;
        default:*hot = e.down_hot; break;
    }
    *map_hot  = e.map_hot;
    *map_cold = e.map_cold;
    *mask_hot = e.mask_hot;
    return 1;
}
