// llama-dynamic-transfer.h — Load-aware dynamic tensor transfer (ATSInfer §4.4)
#ifndef LLAMA_DYNAMIC_TRANSFER_H
#define LLAMA_DYNAMIC_TRANSFER_H

#include "llama-tensor-profiler.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <deque>
#include <chrono>
#include <atomic>

struct perf_sample {
    double cpu_latency_ms;
    double gpu_latency_ms;
    double pcie_transfer_ms;
    size_t bytes_transferred;
    double wall_time;
};

struct dynamic_transfer_config {
    bool   enabled              = false;
    double monitor_interval     = 1.0;
    double promotion_threshold  = 0.15;
    double demotion_threshold   = 0.05;
    size_t temp_buffer_size     = 512 * 1024 * 1024; // 512 MB
    int    history_window       = 10;
};

class llama_dynamic_transfer {
public:
    llama_dynamic_transfer();
    ~llama_dynamic_transfer();

    void init(const dynamic_transfer_config & cfg,
              ggml_backend_t gpu_backend, ggml_backend_t cpu_backend,
              size_t total_vram_bytes);
    void record_sample(const perf_sample & sample);

    double get_cpu_load() const { return _cpu_load; }
    double get_gpu_load() const { return _gpu_load; }
    double get_pcie_load() const { return _pcie_load; }

    llama_tensor_profiler::placement_solution
    decide_transfers(const llama_tensor_profiler & profiler, size_t available_vram_bytes);

    bool promote_tensor(const std::string & tensor_name, ggml_tensor * tensor, ggml_backend_buffer_t temp_buf);
    void demote_tensor(const std::string & tensor_name);
    void reset();

private:
    dynamic_transfer_config _cfg;
    ggml_backend_t _gpu_backend = nullptr;
    ggml_backend_t _cpu_backend = nullptr;
    size_t _total_vram = 0;
    std::deque<perf_sample> _history;
    double _cpu_load = 0.0, _gpu_load = 0.0, _pcie_load = 0.0;

    struct promoted_tensor {
        std::string name; size_t size; double benefit; double last_used;
    };
    std::vector<promoted_tensor> _promoted;
    std::chrono::high_resolution_clock::time_point _last_monitor;
    std::atomic<bool> _initialized{false};

    void update_load_estimates();
    double effective_benefit(double base_benefit, size_t size) const;
};

#endif
