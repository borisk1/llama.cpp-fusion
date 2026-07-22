// llama-tensor-profiler.cpp — Tensor profiling implementation
#include "llama-tensor-profiler.h"
#include "llama-impl.h"

// CUDA event timing functions (from llama-cuda-stream.cu)
extern "C" {
    void * llama_cuda_event_create();
    void   llama_cuda_event_destroy(void *);
    int    llama_cuda_event_record(void *, void *);
    int    llama_cuda_event_sync(void *);
    float  llama_cuda_event_elapsed_ms(void *, void *);
}
#include <algorithm>
#include <numeric>
#include <cstring>

void llama_tensor_profiler::begin_op(const std::string & tensor_name, const std::string & op_type, 
                                      ggml_backend_dev_t backend, size_t tensor_size) {
    // Create entries with a pending measurement
    auto & entry = entries[tensor_name];
    entry.name    = tensor_name;
    entry.size    = tensor_size;
    entry.op_type = op_type;
    entry.count   = 1;

    // Mark as pending (will be filled by measure_tensor_on_backend)
    active_timing timing;
    timing.start   = std::chrono::high_resolution_clock::now();
    timing.is_cpu  = (backend && ggml_backend_dev_type(backend) == GGML_BACKEND_DEVICE_TYPE_CPU);
    timing.size    = tensor_size;
    timing.op_type = op_type;
    active[tensor_name] = timing;
}

void llama_tensor_profiler::end_op(const std::string & tensor_name) {
    auto it = active.find(tensor_name);
    if (it == active.end()) return;

    auto end = std::chrono::high_resolution_clock::now();
    auto & timing = it->second;
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - timing.start).count();

    auto & entry = entries[tensor_name];
    entry.name    = tensor_name;
    entry.size    = timing.size;
    entry.op_type = timing.op_type;
    entry.count++;

    if (timing.is_cpu) {
        entry.t_cpu += elapsed_ms;
    } else {
        entry.t_gpu += elapsed_ms;
    }

    active.erase(it);
}

void llama_tensor_profiler::calculate_epd() {
    for (auto & [name, entry] : entries) {
        // Average timing over count
        double avg_cpu = entry.t_cpu / std::max(entry.count, 1);
        double avg_gpu = entry.t_gpu / std::max(entry.count, 1);

        entry.benefit = avg_cpu - avg_gpu;

        // Quant-type aware benefit adjustment
        // Higher-bit quant types have more compute per byte,
        // so GPU offload delivers larger absolute benefit.
        // Infer quant density from tensor name patterns.
        // Ex: ffn_gate_exps = MoE experts (IQ2_XXS or Q4_K)
        //     attn weights = Q8_0 or F16 in typical DSV4 configs
        if (entry.size > 0) {
            double bpw = 0; // bits per weight (0 = unknown)
            // Detect from known DSV4 tensor quant patterns
            if (name.find("ffn_gate_exps") != std::string::npos ||
                name.find("ffn_up_exps")   != std::string::npos ||
                name.find("ffn_down_exps") != std::string::npos) {
                bpw = 4.5; // Q4_K expert layers 37-42, IQ2_XXS rest
            } else if (name.find("attn_q_a") != std::string::npos ||
                       name.find("attn_q_b") != std::string::npos ||
                       name.find("attn_kv_a") != std::string::npos ||
                       name.find("attn_kv_b") != std::string::npos ||
                       name.find("attn_output") != std::string::npos ||
                       name.find("wo_a") != std::string::npos ||
                       name.find("wo_b") != std::string::npos) {
                bpw = 8.0; // Q8_0 attention weights
            } else if (name.find("norm") != std::string::npos) {
                bpw = 16.0; // F16 norms
            }
            if (bpw > 0) {
                // Normalize to IQ2_XXS (2.5 bpw) = 1.0x factor
                double quant_factor = bpw / 2.5;
                entry.benefit *= quant_factor;
            }
        }

        // EPD: time per byte (lower is better for CPU, higher benefit for GPU)
        if (entry.size > 0) {
            entry.epd_cpu = avg_cpu / entry.size;
            entry.epd_gpu = avg_gpu / entry.size;
        }

        // Extract layer index from tensor name if present
        // Format: "blk.N.name" or just "name"
        if (name.substr(0, 4) == "blk.") {
            auto dot2 = name.find('.', 4);
            if (dot2 != std::string::npos) {
                try {
                    entry.layer = std::stoi(name.substr(4, dot2 - 4));
                } catch (...) {
                    entry.layer = -1;
                }
            }
        } else {
            entry.layer = -1;
        }
    }
}

void llama_tensor_profiler::print_report() const {
    fprintf(stderr, "=== Tensor Profiler Report ===\n");
    fprintf(stderr, "%-50s %10s %10s %10s %12s %12s %12s\n",
                   "Tensor", "Size(MB)", "t_cpu(ms)", "t_gpu(ms)", "Benefit", "EPD_CPU", "EPD_GPU");
    fprintf(stderr, "%s\n", std::string(120, '-').c_str());

    // Sort by benefit descending
    std::vector<const tensor_profile_entry*> sorted;
    for (const auto & [name, entry] : entries) {
        sorted.push_back(&entry);
    }
    std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) {
        return a->benefit > b->benefit;
    });

    for (const auto * e : sorted) {
        if (e->count == 0) continue;
        double avg_cpu = e->t_cpu / e->count;
        double avg_gpu = e->t_gpu / e->count;
        fprintf(stderr, "%-50s %10.2f %10.4f %10.4f %12.4f %12.6f %12.6f\n",
                       e->name.c_str(),
                       e->size / (1024.0 * 1024.0),
                       avg_cpu, avg_gpu, e->benefit,
                       e->epd_cpu, e->epd_gpu);
    }
    fprintf(stderr, "=== End Report ===\n");
    fflush(stderr);
}

std::vector<std::string> llama_tensor_profiler::get_top_tensors(int n, double vram_budget_bytes) const {
    // Return top N tensors by benefit that fit within VRAM budget
    std::vector<const tensor_profile_entry*> sorted;
    for (const auto & [name, entry] : entries) {
        sorted.push_back(&entry);
    }
    std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) {
        return (a->benefit / a->size) > (b->benefit / b->size);
    });

    std::vector<std::string> result;
    double used = 0;
    for (const auto * e : sorted) {
        if (used + e->size > vram_budget_bytes) continue;
        result.push_back(e->name);
        used += e->size;
    }
    return result;
}

std::vector<std::string> llama_tensor_profiler::generate_overrides(const placement_solution & solution) const {
    // Generate --override-tensor commands from knapsack solution
    std::vector<std::string> overrides;

    for (const auto & name : solution.gpu_tensors) {
        overrides.push_back(name + "=CUDA0");
    }
    for (const auto & name : solution.cpu_tensors) {
        overrides.push_back(name + "=CPU");
    }

    fprintf(stderr, "tensor-profiler: generated %zu tensor overrides (%zu GPU, %zu CPU)\n",
                   overrides.size(), solution.gpu_tensors.size(),
                   solution.cpu_tensors.size());
    return overrides;
}

double llama_tensor_profiler::measure_tensor_on_backend(
        const std::string & tensor_name,
        ggml_tensor * tensor,
        ggml_backend_t backend,
        const std::string & op_type) {
    if (!tensor || !backend) return 0.0;

    size_t nbytes = ggml_nbytes(tensor);
    if (nbytes == 0) return 0.0;

    auto * dev = ggml_backend_get_device(backend);

    // --- Real GPU measurement via CUDA events ---
    // Only for small tensors with sufficient free VRAM.
    // Skip if dev is null (backend without device).

    size_t needed = nbytes * 3 + 128ULL * 1024 * 1024;
    size_t free_mem = 0, total_mem = 0;
    bool can_measure = false;

    if (dev && nbytes <= 4ULL * 1024ULL * 1024ULL) {
        // Only attempt real GPU measurement on ACCEL devices
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);
            can_measure = (free_mem > needed + 256ULL * 1024 * 1024);
        }
    }

    if (!can_measure) {
        // Estimation fallback with quant-type awareness
        // Different quant types have different compute density:
        // Higher bpw = more compute per byte = more GPU benefit
        double compute_factor = 1.0;
        if (tensor) {
            auto tt = tensor->type;
            // Normalize to IQ2_XXS (2.5 bpw) = 1.0
            if (tt == GGML_TYPE_F32)       compute_factor = 12.8;
            else if (tt == GGML_TYPE_F16)   compute_factor = 6.4;
            else if (tt == GGML_TYPE_Q8_0)  compute_factor = 3.2;
            else if (tt == GGML_TYPE_Q4_K)  compute_factor = 1.8;
            else if (tt == GGML_TYPE_Q2_K)  compute_factor = 1.0;
            else if (tt == GGML_TYPE_IQ4_NL)compute_factor = 1.5;
            else if (tt == GGML_TYPE_IQ3_S) compute_factor = 1.2;
            else if (tt == GGML_TYPE_IQ2_S) compute_factor = 1.0;
            else if (tt == GGML_TYPE_IQ2_XS) compute_factor = 1.0;
            else if (tt == GGML_TYPE_IQ2_XXS) compute_factor = 1.0;
            // IQ1_S is experimental, use 0.5
            else if (tt == GGML_TYPE_IQ1_S) compute_factor = 0.5;
        }
        if (dev) {
            if (nbytes <= 4ULL * 1024ULL * 1024ULL) {
                fprintf(stderr, "measure: %s %s = skipped real (VRAM: %zu MB free, need %zu MB)\n",
                        op_type.c_str(), tensor_name.c_str(),
                        free_mem >> 20, needed >> 20);
            }
            double bw = std::min((total_mem / 1e9) * 15.0, 2000.0);
            if (bw < 1) bw = 40.0; // fallback DDR4
            double t = (2.0 * nbytes) / (bw * 1e9 / 1000.0);
            fprintf(stderr, "measure: %s %s = %.4f ms (est, %.0f MB, %.0f GB/s)\n",
                    op_type.c_str(), tensor_name.c_str(), t,
                    nbytes / (1024.0 * 1024.0), bw);
            return t;
        }
        return 0.0;
    }

    // Compute ggml context size proportional to tensor size
    size_t ctx_size = std::max((size_t)1024 * 1024, nbytes * 4);
    ctx_size = std::min(ctx_size, (size_t)512 * 1024 * 1024); // cap at 512 MB

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return 0.0;

    // Create a COPY of the weight in our context
    struct ggml_tensor * wc = NULL;
    int n3 = tensor->ne[3], n2 = tensor->ne[2], n1 = tensor->ne[1], n0 = tensor->ne[0];
    int ndims = (n3>1?4:n2>1?3:n1>1?2:1);
    switch (ndims) {
        case 4: wc = ggml_new_tensor_4d(ctx, tensor->type, n0, n1, n2, n3); break;
        case 3: wc = ggml_new_tensor_3d(ctx, tensor->type, n0, n1, n2); break;
        case 2: wc = ggml_new_tensor_2d(ctx, tensor->type, n0, n1); break;
        default: wc = ggml_new_tensor_1d(ctx, tensor->type, n0); break;
    }
    if (!wc) { ggml_free(ctx); return 0.0; }

    // Build graph with local weight copy
    struct ggml_tensor * inp = NULL, * res = NULL;
    if (op_type == "MUL_MAT" || op_type == "MUL_MAT_ID") {
        inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0);
        if (!inp) { ggml_free(ctx); return 0.0; }
        res = ggml_mul_mat(ctx, wc, inp);
    } else if (op_type == "RMS_NORM") {
        inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0);
        if (!inp) { ggml_free(ctx); return 0.0; }
        res = ggml_rms_norm(ctx, inp, 1e-6f);
    } else { ggml_free(ctx); return 0.0; }
    if (!res) { ggml_free(ctx); return 0.0; }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    if (!gf) { ggml_free(ctx); return 0.0; }
    ggml_build_forward_expand(gf, res);

    // Allocate ALL (weight copy + input + result) on the backend
    // All tensors are local - no cross-context issues
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        double bw = std::min(((double)(dev ? 24 : 0) * 13.0), 2000.0);
        return (2.0 * nbytes) / (bw * 1e9 / 1000.0);
    }

    // Copy weight data to our local copy (CPU/GPU via backend)
    if (tensor->data && wc->data) {
        auto dt = ggml_backend_dev_type(dev);
        if (dt == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ggml_backend_tensor_copy(tensor, wc);
            // Sync via graph compute with a trivial graph
            struct ggml_cgraph * sync_gf = ggml_new_graph(ctx);
            if (sync_gf) ggml_backend_graph_compute(backend, sync_gf);
        } else {
            std::memcpy(wc->data, tensor->data, nbytes);
        }
    }

    // Fill input
    if (inp && inp->data) {
        float * d = (float *)inp->data;
        for (int64_t i = 0; i < ggml_nelements(inp) && i < 256; i++) d[i] = 1.0f;
    }

    // Warmup + timed runs with CUDA events for precise GPU timing
    ggml_backend_graph_compute(backend, gf);

    // Try CUDA event timing
    void * ev_start = llama_cuda_event_create();
    void * ev_stop  = llama_cuda_event_create();

    if (ev_start && ev_stop) {
        llama_cuda_event_record(ev_start, nullptr);
        ggml_backend_graph_compute(backend, gf);
        llama_cuda_event_record(ev_stop, nullptr);
        llama_cuda_event_sync(ev_stop);

        float gpu_ms = llama_cuda_event_elapsed_ms(ev_start, ev_stop);
        llama_cuda_event_destroy(ev_start);
        llama_cuda_event_destroy(ev_stop);

        if (gpu_ms > 0) {
            double avg_ms = gpu_ms;
            fprintf(stderr, "measure: %s %s = %.4f ms (CUDA event, %.0f MB)\n",
                    op_type.c_str(), tensor_name.c_str(), avg_ms,
                    nbytes / (1024.0 * 1024.0));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return avg_ms;
        }
    } else {
        if (ev_start) llama_cuda_event_destroy(ev_start);
        if (ev_stop)  llama_cuda_event_destroy(ev_stop);
    }

    // Fallback: multi-iteration chrono timing
    int n_iter = std::max(1, std::min(30, (int)(10000 / std::max(nbytes, (size_t)1))));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_iter;
    fprintf(stderr, "measure: %s %s = %.4f ms (%d runs, %.0f MB)\n",
            op_type.c_str(), tensor_name.c_str(), avg_ms, n_iter,
            nbytes / (1024.0 * 1024.0));
    return avg_ms;
}

void llama_tensor_profiler::record_measurement(
        const std::string & name, const std::string & op_type,
        size_t size, double t_cpu_ms, double t_gpu_ms) {
    auto & entry = entries[name];
    entry.name    = name;
    entry.size    = size;
    entry.op_type = op_type;
    entry.count   = 1;
    entry.t_cpu   = t_cpu_ms;
    entry.t_gpu   = t_gpu_ms;
    entry.benefit = t_cpu_ms - t_gpu_ms;
    if (size > 0) {
        entry.epd_cpu = t_cpu_ms / size;
        entry.epd_gpu = t_gpu_ms / size;
    }
    // Extract layer index
    if (name.substr(0, 4) == "blk.") {
        auto dot2 = name.find('.', 4);
        if (dot2 != std::string::npos) {
            try { entry.layer = std::stoi(name.substr(4, dot2 - 4)); }
            catch (...) { entry.layer = -1; }
        }
    }
}

void llama_tensor_profiler::clear() {
    entries.clear();
    active.clear();
}

llama_tensor_profiler::placement_solution llama_tensor_profiler::solve_knapsack(double vram_budget_bytes, double pcie_bw_gbs) const {
    struct item {
        std::string name;
        double      benefit;    // r_i = t_cpu - t_gpu
        size_t      size;       // s_i
        bool        is_expert;  // expert tensor?
        int         layer;
        std::string op_type;
    };

    if (entries.empty()) {
        fprintf(stderr, "knapsack WARNING: no profiling data available\n");
        return {};
    }

    // Build ordered list of tensors (graph order: layer → operation)
    std::vector<item> items;
    for (const auto & [name, entry] : entries) {
        if (entry.count == 0 || entry.size == 0) continue;
        if (entry.layer < 0) continue;
        items.push_back({
            name, entry.benefit, entry.size,
            (name.find("exps") != std::string::npos),
            entry.layer, entry.op_type
        });
    }
    if (items.empty()) { return {}; }

    double mb = 1024.0 * 1024.0;

    // Sort by (layer, operation_order) — must match graph topology
    auto op_order = [](const std::string & n) {
        if (n.find("ffn_gate_exps") != std::string::npos) return 10;
        if (n.find("ffn_gate_inp")  != std::string::npos) return 11;
        if (n.find("ffn_up_exps")   != std::string::npos) return 12;
        if (n.find("ffn_down_exps") != std::string::npos) return 13;
        if (n.find("ffn_gate_shexp")!= std::string::npos) return 20;
        if (n.find("ffn_up_shexp")  != std::string::npos) return 21;
        if (n.find("ffn_down_shexp")!= std::string::npos) return 22;
        if (n.find("attn_q")        != std::string::npos) return 30;
        if (n.find("attn_kv")       != std::string::npos) return 31;
        if (n.find("attn_output")   != std::string::npos) return 32;
        if (n.find("wo_a")          != std::string::npos) return 33;
        if (n.find("wo_b")          != std::string::npos) return 34;
        if (n.find("_norm")         != std::string::npos) return 40;
        if (n.find("hc_")           != std::string::npos) return 45;
        return 99;
    };
    std::sort(items.begin(), items.end(),
        [&](const item & a, const item & b) {
            if (a.layer != b.layer) return a.layer < b.layer;
            int oa = op_order(a.name), ob = op_order(b.name);
            return oa != ob ? oa < ob : a.name < b.name;
        });

    // MoE partitioning (Algorithm 1 lines 10-21)
    std::vector<item> exp_items, non_exp_items;
    double exp_sz = 0, non_exp_sz = 0;
    for (auto & it : items) {
        if (it.is_expert) { exp_items.push_back(it); exp_sz += it.size; }
        else { non_exp_items.push_back(it); non_exp_sz += it.size; }
    }

    // DP with switching cost: max Σ(r_i*x_i) - Σ(c_i*switch_i) s.t. Σ(s_i*x_i) ≤ M
    auto solve_dp = [&](const std::vector<item> & tensor_list, double budget) -> std::vector<bool> {
        int m = (int)tensor_list.size();
        if (m == 0 || budget <= 0) return std::vector<bool>(m, false);
        double bw = (pcie_bw_gbs > 0) ? pcie_bw_gbs * 1e9 / 1e3 : 30.0; // bytes/ms
        int M = std::max(1, (int)(budget / mb));
        std::vector<double> dp(M + 1, -1e18);
        std::vector<std::vector<char>> choice(m, std::vector<char>(M + 1, 0));
        dp[0] = 0;

        for (int i = 0; i < m; i++) {
            int s = std::max(1, (int)(tensor_list[i].size / mb));
            double r = tensor_list[i].benefit;
            double c = 0.005; // default switching cost ~0.005ms (small activation)
            if (tensor_list[i].op_type == "MUL_MAT_ID")
                c = (128.0 * 4096.0 * 4.0) / bw; // expert output activation
            else if (tensor_list[i].op_type == "MUL_MAT")
                c = (4096.0 * 4.0) / bw; // attention activation

            std::vector<double> ndp(M + 1, -1e18);
            for (int w = 0; w <= M; w++) {
                if (dp[w] < -1e17) continue;
                // CPU placement (no VRAM cost)
                double cpu_v = dp[w] - (i > 0 && choice[i-1][w] == 1 ? c : 0);
                if (cpu_v > ndp[w]) { ndp[w] = cpu_v; choice[i][w] = 0; }
                // GPU placement (costs s MB, gains r benefit)
                if (w + s <= M) {
                    double gpu_v = dp[w] + r - (i > 0 && choice[i-1][w] == 0 ? c : 0);
                    if (gpu_v > ndp[w + s]) { ndp[w + s] = gpu_v; choice[i][w + s] = 1; }
                }
            }
            dp = std::move(ndp);
        }

        int best_w = (int)(std::max_element(dp.begin(), dp.end()) - dp.begin());
        std::vector<bool> placement(m, false);
        for (int i = m - 1, w = best_w; i >= 0; i--) {
            placement[i] = (choice[i][w] == 1);
            if (placement[i]) w -= std::max(1, (int)(tensor_list[i].size / mb));
        }
        return placement;
    };

    placement_solution result;
    auto add_gpu = [&](const item & it) {
        result.gpu_tensors.push_back(it.name);
        result.total_benefit += it.benefit;
        result.total_size += it.size;
    };
    auto add_cpu = [&](const item & it) {
        result.cpu_tensors.push_back(it.name);
    };

    if (!exp_items.empty() && !non_exp_items.empty()) {
        // MoE: Algorithm 1 partitioning
        if (non_exp_sz <= vram_budget_bytes) {
            double exp_budget = vram_budget_bytes - non_exp_sz;
            double total_exp = 0;
            for (auto & e : exp_items) total_exp += e.size;
            if (total_exp <= exp_budget) {
                for (auto & it : items) add_gpu(it);
                fprintf(stderr, "knapsack DP: all fit (%.0f MB)\n", result.total_size/mb);
                return result;
            }
            for (auto & it : non_exp_items) add_gpu(it);
            auto p = solve_dp(exp_items, exp_budget);
            for (int i = 0; i < (int)exp_items.size(); i++)
                p[i] ? add_gpu(exp_items[i]) : add_cpu(exp_items[i]);
        } else {
            auto p = solve_dp(non_exp_items, vram_budget_bytes);
            for (int i = 0; i < (int)non_exp_items.size(); i++)
                p[i] ? add_gpu(non_exp_items[i]) : add_cpu(non_exp_items[i]);
            for (auto & it : exp_items) add_cpu(it);
        }
    } else {
        auto p = solve_dp(items, vram_budget_bytes);
        for (int i = 0; i < (int)items.size(); i++)
            p[i] ? add_gpu(items[i]) : add_cpu(items[i]);
    }

    int switches = 0;
    for (size_t i = 1; i < result.gpu_tensors.size(); i++) {
        auto it1 = entries.find(result.gpu_tensors[i-1]);
        auto it2 = entries.find(result.gpu_tensors[i]);
        if (it1 != entries.end() && it2 != entries.end() &&
            it1->second.layer != it2->second.layer) switches++;
    }
    fprintf(stderr, "knapsack DP: selected %zu GPU (%.0f MB, benefit %.2f ms, %d switches)\n",
            result.gpu_tensors.size(), result.total_size / mb,
            result.total_benefit, switches);
    return result;
}
