// llama-dynamic-transfer.cpp — Load-aware dynamic tensor transfer
#include "llama-dynamic-transfer.h"
#include "llama-impl.h"
#include <algorithm>
#include <numeric>

llama_dynamic_transfer::llama_dynamic_transfer() {
    _last_monitor = std::chrono::high_resolution_clock::now();
}

llama_dynamic_transfer::~llama_dynamic_transfer() = default;

void llama_dynamic_transfer::init(const dynamic_transfer_config & cfg,
                                   ggml_backend_t gpu_backend,
                                   ggml_backend_t cpu_backend,
                                   size_t total_vram_bytes) {
    _cfg = cfg;
    _gpu_backend = gpu_backend;
    _cpu_backend = cpu_backend;
    _total_vram = total_vram_bytes;
    _initialized = true;
    LLAMA_LOG_INFO("%s: dynamic transfer initialized (interval=%.1fs, threshold=%.2f, temp_buf=%zu MB)\n",
                   __func__, cfg.monitor_interval, cfg.promotion_threshold,
                   cfg.temp_buffer_size / (1024*1024));
}

void llama_dynamic_transfer::record_sample(const perf_sample & sample) {
    if (!_initialized) return;

    _history.push_back(sample);
    if ((int)_history.size() > _cfg.history_window) {
        _history.pop_front();
    }

    // Periodically update load estimates
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - _last_monitor).count();
    if (elapsed >= _cfg.monitor_interval) {
        update_load_estimates();
        _last_monitor = now;
    }
}

void llama_dynamic_transfer::update_load_estimates() {
    if (_history.empty()) return;

    // Calculate moving averages
    double total_cpu_lat = 0, total_gpu_lat = 0, total_pcie_lat = 0;
    size_t total_bytes = 0;
    int n = 0;

    for (const auto & s : _history) {
        total_cpu_lat  += s.cpu_latency_ms;
        total_gpu_lat  += s.gpu_latency_ms;
        total_pcie_lat += s.pcie_transfer_ms;
        total_bytes    += s.bytes_transferred;
        n++;
    }

    double avg_cpu  = total_cpu_lat / n;
    double avg_gpu  = total_gpu_lat / n;
    double avg_pcie = total_pcie_lat / n;

    // Normalize loads relative to max observed
    // Higher latency = higher load
    static double max_cpu_seen = 1.0, max_gpu_seen = 1.0, max_pcie_seen = 1.0;
    max_cpu_seen  = std::max(max_cpu_seen, avg_cpu);
    max_gpu_seen  = std::max(max_gpu_seen, avg_gpu);
    max_pcie_seen = std::max(max_pcie_seen, avg_pcie);

    _cpu_load  = avg_cpu / max_cpu_seen;
    _gpu_load  = avg_gpu / max_gpu_seen;
    _pcie_load = avg_pcie / max_pcie_seen;

    LLAMA_LOG_DEBUG("%s: loads CPU=%.2f GPU=%.2f PCIe=%.2f (samples=%zu)\n",
                    __func__, _cpu_load, _gpu_load, _pcie_load, _history.size());
}

double llama_dynamic_transfer::effective_benefit(double base_benefit, size_t size) const {
    // Adjust benefit based on current load conditions
    // If GPU is heavily loaded, GPU placement is less beneficial
    // If CPU is heavily loaded, GPU placement is more beneficial
    // If PCIe is loaded, transfers are more expensive
    
    double cpu_factor  = 1.0 + _cpu_load;   // CPU load multiplies benefit
    double gpu_factor  = 2.0 - _gpu_load;   // GPU load reduces benefit
    double pcie_factor = 2.0 - _pcie_load;  // PCIe load reduces transfer advantage
    
    return base_benefit * cpu_factor * gpu_factor * pcie_factor;
}

llama_tensor_profiler::placement_solution
llama_dynamic_transfer::decide_transfers(const llama_tensor_profiler & profiler,
                                          size_t available_vram_bytes) {
    if (!_initialized || _history.empty()) {
        return {};
    }

    // Get base knapsack solution with adjusted budget
    double effective_budget = available_vram_bytes * (1.0 - _cpu_load * 0.3);
    auto solution = profiler.solve_knapsack(effective_budget, 12.0);

    LLAMA_LOG_INFO("%s: promoting %zu tensors (%.0f MB) under load CPU=%.2f GPU=%.2f\n",
                   __func__, solution.gpu_tensors.size(),
                   solution.total_size / (1024*1024),
                   _cpu_load, _gpu_load);

    return solution;
}

bool llama_dynamic_transfer::promote_tensor(const std::string & tensor_name,
                                             ggml_tensor * tensor,
                                             ggml_backend_buffer_t temp_buf) {
    if (!tensor || !temp_buf) return false;

    // Calculate size needed
    size_t nbytes = ggml_nbytes(tensor);
    if (nbytes == 0) return false;

    // Check if already promoted
    for (const auto & p : _promoted) {
        if (p.name == tensor_name) return true;
    }

    // Record promotion
    promoted_tensor pt;
    pt.name     = tensor_name;
    pt.size     = nbytes;
    pt.benefit  = 0; // will be updated
    pt.last_used = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    _promoted.push_back(pt);

    LLAMA_LOG_DEBUG("%s: promoted '%s' (%zu bytes)\n", __func__, tensor_name.c_str(), nbytes);
    return true;
}

void llama_dynamic_transfer::demote_tensor(const std::string & tensor_name) {
    auto it = std::find_if(_promoted.begin(), _promoted.end(),
        [&](const promoted_tensor & p) { return p.name == tensor_name; });
    if (it != _promoted.end()) {
        LLAMA_LOG_DEBUG("%s: demoted '%s' (%zu bytes)\n", __func__, it->name.c_str(), it->size);
        _promoted.erase(it);
    }
}

void llama_dynamic_transfer::reset() {
    _history.clear();
    _promoted.clear();
    _cpu_load = _gpu_load = _pcie_load = 0.0;
    _initialized = false;
}
