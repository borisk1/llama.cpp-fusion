// llama-tensor-profiler.cpp — Tensor profiling implementation
#include "llama-tensor-profiler.h"
#include "llama-impl.h"
#include <algorithm>
#include <numeric>

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

    // For tensors > 64 MB, use estimation to avoid GPU memory pressure
    if (nbytes > 64ULL * 1024ULL * 1024ULL) {
        auto * dev = ggml_backend_get_device(backend);
        size_t total = 0, free = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        double bw = std::min((total / 1e9) * 15.0, 2000.0);
        double t = (2.0 * nbytes) / (bw * 1e9 / 1000.0);
        fprintf(stderr, "measure: %s %s = %.4f ms (est, >64MB)\n",
                op_type.c_str(), tensor_name.c_str(), t);
        return t;
    }

    // NOTE: Real GPU measurement via ggml_backend_sched is currently blocked by
    // internal assertions in ggml_gallocr_reserve_n when handling cross-context
    // tensor references (view tensors from model context). This requires deeper
    // ggml backend changes.
    //
    // For now, use bandwidth-based estimation which is stable and gives meaningful
    // relative timings for the knapsack solver.
    auto * dev = ggml_backend_get_device(backend);
    if (!dev) return 0.0;
    size_t total_ram = 0, free_ram = 0;
    ggml_backend_dev_memory(dev, &free_ram, &total_ram);
    double bw = 100.0;
    double cr = 1.0;
    auto dt = ggml_backend_dev_type(dev);
    if (dt == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        double v = total_ram / 1e9;
        bw = std::min(v * 15.0, 2000.0); cr = 20.0;
    } else { bw = 40.0; cr = 1.0; }
    double t = 0;
    if (op_type == "MUL_MAT" || op_type == "MUL_MAT_ID") {
        double b = 2.0 * nbytes, m = b / (bw * 1e9 / 1000.0);
        t = m + m/cr + (dt == GGML_BACKEND_DEVICE_TYPE_ACCEL ? 0.02 : 0.2);
    } else if (op_type == "RMS_NORM") {
        double e = (double)ggml_nelements(tensor) * sizeof(float);
        t = e / (bw * 1e9 / 1000.0) / (1.0 + 0.5*(cr-1.0));
    }
    t = std::max(t, 0.001);
    fprintf(stderr, "measure: %s %s = %.4f ms (est, %.0f MB, %.0f GB/s)\n",
            op_type.c_str(), tensor_name.c_str(), t,
            nbytes/(1024.0*1024.0), bw);
    return t;
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
        double      benefit;    // r_i = t_cpu - t_gpu (time saved)
        size_t      size;       // s_i (memory footprint)
        double      benefit_per_byte;
        int         group;      // layer index for switching cost
    };

    if (entries.empty()) {
        fprintf(stderr, "knapsack WARNING: no profiling data available\n");
        return {};
    }

    std::vector<item> items;
    for (const auto & [name, entry] : entries) {
        if (entry.count == 0 || entry.size == 0) continue;
        items.push_back({
            entry.name,
            entry.benefit,
            entry.size,
            entry.benefit / entry.size,
            entry.layer
        });
    }

    // Sort by benefit-per-byte descending (greedy knapsack for large N)
    std::sort(items.begin(), items.end(), [](const item & a, const item & b) {
        return a.benefit_per_byte > b.benefit_per_byte;
    });

    // Greedy selection within budget
    placement_solution result;
    result.total_benefit = 0;
    result.total_size = 0;

    for (const auto & it : items) {
        if (result.total_size + it.size > vram_budget_bytes) continue;
        result.gpu_tensors.push_back(it.name);
        result.total_benefit += it.benefit;
        result.total_size += it.size;
    }

    // Build CPU list (everything not in GPU list)
    for (const auto & it : items) {
        if (std::find(result.gpu_tensors.begin(), result.gpu_tensors.end(), it.name) == result.gpu_tensors.end()) {
            result.cpu_tensors.push_back(it.name);
        }
    }

    // Apply switching cost penalty
    // When adjacent tensors in the computation graph are on different backends,
    // we incur a switching cost = input_size / PCIe_BW
    // For simplicity, estimate switching cost based on layer transitions
    int switches = 0;
    for (size_t i = 1; i < items.size(); i++) {
        bool prev_gpu = std::find(result.gpu_tensors.begin(), result.gpu_tensors.end(), items[i-1].name) != result.gpu_tensors.end();
        bool curr_gpu = std::find(result.gpu_tensors.begin(), result.gpu_tensors.end(), items[i].name) != result.gpu_tensors.end();
        if (prev_gpu != curr_gpu && items[i-1].group != items[i].group) {
            switches++;
        }
    }

    double switching_penalty = switches * 0.001; // ~1ms per switch
    result.total_benefit -= switching_penalty;

    fprintf(stderr, "knapsack: selected %zu GPU tensors (%.0f MB, benefit %.2f ms, %d switches)\n",
                   result.gpu_tensors.size(), result.total_size / (1024*1024),
                   result.total_benefit, switches);

    return result;
}
