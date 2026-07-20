// llama-tensor-profiler.cpp — Tensor profiling implementation
#include "llama-tensor-profiler.h"
#include "llama-impl.h"
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

    // Check if real GPU measurement is feasible.
    // Real measurement creates a weight copy on the GPU, which requires
    // free VRAM. On multi-GPU setups with models loaded, VRAM is tight.
    auto * dev = ggml_backend_get_device(backend);
    bool use_real = false;
    if (dev && nbytes <= 4ULL * 1024ULL * 1024ULL) {
        size_t total = 0, free = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        use_real = (free > 512ULL * 1024ULL * 1024ULL);
    }

    if (!use_real) {
        // Estimation fallback
        if (dev) {
            size_t total = 0, free = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            double bw = std::min((total / 1e9) * 15.0, 2000.0);
            double t = (2.0 * nbytes) / (bw * 1e9 / 1000.0);
            fprintf(stderr, "measure: %s %s = %.4f ms (est, %.0f MB, %.0f GB/s)\n",
                    op_type.c_str(), tensor_name.c_str(), t,
                    nbytes / (1024.0 * 1024.0), bw);
            return t;
        }
        return 0.0;
    }

    // Real GPU measurement via weight copy
    // Creates a local copy of the weight tensor in our ggml context,
    // avoiding cross-context issues with ggml_backend_alloc_ctx_tensors
    if (!dev) return 0.0;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024 * 128, // 128 MB
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

    // Warmup + timed runs
    ggml_backend_graph_compute(backend, gf);

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
    // Build list of items for knapsack
    struct item {
        std::string name;
        double      benefit;
        size_t      size;
        double      benefit_per_byte;
        int         layer;
        std::string op_type;
    };

    if (entries.empty()) {
        fprintf(stderr, "knapsack WARNING: no profiling data available\n");
        return {};
    }

    // Group entries by layer
    std::unordered_map<int, std::vector<const tensor_profile_entry*>> by_layer;
    int max_layer = 0;
    for (const auto & [name, entry] : entries) {
        if (entry.count == 0 || entry.size == 0) continue;
        int l = entry.layer;
        if (l < 0) continue; // skip unlayered tensors
        by_layer[l].push_back(&entry);
        max_layer = std::max(max_layer, l);
    }

    // For each layer, compute total benefit, size, and priority
    // Priority = benefit / size * layer_depth_factor
    // where layer_depth_factor = 1 + (max_layer - l) / max_layer  (0..2× boost)
    // This gives earlier layers up to 2× priority vs last layer
    struct layer_item {
        int    layer_id;
        double total_benefit;
        size_t total_size;
        double priority;
        std::vector<const tensor_profile_entry*> tensors;
    };

    std::vector<layer_item> layers;
    for (auto & [l, tensors] : by_layer) {
        double benefit = 0;
        size_t size = 0;
        for (auto * t : tensors) {
            benefit += t->benefit;
            size    += t->size;
        }
        // Layer depth factor: earlier layers get higher priority
        double depth_factor = 1.0;
        int layer_count = max_layer + 1;
        if (layer_count > 0) {
            // Linear boost: layer 0 gets ~2×, last layer gets 1×
            depth_factor = 1.0 + (double)(layer_count - 1 - l) / layer_count;
        }
        double priority = (size > 0) ? (benefit / size) * depth_factor : 0;
        layers.push_back({l, benefit, size, priority, tensors});
    }

    // Sort layers by priority descending
    std::sort(layers.begin(), layers.end(), [](const layer_item & a, const layer_item & b) {
        return a.priority > b.priority;
    });

    // Re-order: create a per-tensor list sorted by layer priority first,
    // then within each layer by individual benefit_per_byte.
    // This way, all tensors from high-priority layers come first.
    std::vector<const tensor_profile_entry*> sorted_all;
    for (const auto & layer : layers) {
        std::vector<const tensor_profile_entry*> layer_tensors = layer.tensors;
        std::sort(layer_tensors.begin(), layer_tensors.end(),
            [](const tensor_profile_entry * a, const tensor_profile_entry * b) {
                return (a->benefit / a->size) > (b->benefit / b->size);
            });
        for (auto * t : layer_tensors) {
            sorted_all.push_back(t);
        }
    }

    // Greedy selection within budget (from sorted_all)
    placement_solution result;
    result.total_benefit = 0;
    result.total_size = 0;
    size_t used = 0;

    // First pass: try to place COMPLETE layers (for minimum switching)
    for (const auto & layer : layers) {
        if (used + layer.total_size > vram_budget_bytes) continue;
        for (auto * t : layer.tensors) {
            result.gpu_tensors.push_back(t->name);
        }
        result.total_benefit += layer.total_benefit;
        result.total_size += layer.total_size;
        used += layer.total_size;
    }

    // Second pass: fill remaining VRAM with individual high-benefit tensors
    // from layers that didn't fit completely
    for (const auto * t : sorted_all) {
        // Skip if already placed
        if (std::find(result.gpu_tensors.begin(), result.gpu_tensors.end(), t->name) != result.gpu_tensors.end()) {
            continue;
        }
        if (used + t->size > vram_budget_bytes) continue;
        result.gpu_tensors.push_back(t->name);
        result.total_benefit += t->benefit;
        result.total_size += t->size;
        used += t->size;
    }

    // Build CPU list (everything not in GPU list)
    for (const auto & [name, entry] : entries) {
        if (entry.count == 0 || entry.size == 0) continue;
        if (std::find(result.gpu_tensors.begin(), result.gpu_tensors.end(), name) == result.gpu_tensors.end()) {
            result.cpu_tensors.push_back(name);
        }
    }

    // Apply switching cost penalty
    int switches = 0;
    for (size_t i = 1; i < result.gpu_tensors.size(); i++) {
        int l1 = -1, l2 = -1;
        auto it1 = entries.find(result.gpu_tensors[i-1]);
        auto it2 = entries.find(result.gpu_tensors[i]);
        if (it1 != entries.end() && it2 != entries.end()) {
            l1 = it1->second.layer;
            l2 = it2->second.layer;
        }
        if (l1 != l2) switches++;
    }

    double switching_penalty = switches * 0.001; // ~1ms per switch
    result.total_benefit -= switching_penalty;

    fprintf(stderr, "knapsack: selected %zu GPU tensors (%.0f MB, benefit %.2f ms, %d switches)\n",
                   result.gpu_tensors.size(), result.total_size / (1024.0*1024.0),
                   result.total_benefit, switches);

    return result;
}
