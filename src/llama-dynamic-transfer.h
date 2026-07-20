// llama-dynamic-transfer.h — Load-aware dynamic tensor transfer (ATSInfer §4.4)
// Monitors runtime hardware load and dynamically promotes CPU-resident tensors to GPU.
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

// Performance sample for load monitoring
struct perf_sample {
    double cpu_latency_ms;     // measured CPU op latency
    double gpu_latency_ms;     // measured GPU op latency  
    double pcie_transfer_ms;   // measured PCIe transfer time
    size_t bytes_transferred;  // bytes transferred in this sample
    double wall_time;          // wall clock timestamp
};

// Dynamic transfer config
struct dynamic_transfer_config {
    bool   enabled           = false;
    double monitor_interval  = 1.0;    // seconds between load checks
    double promotion_threshold = 0.15; // min benefit/byte ratio for promotion
    double demotion_threshold  = 0.05; // below this, demote from GPU
    size_t temp_buffer_size   = 512 * 1024 * 1024; // 512 MB temp GPU buffer for promotions
    int    history_window     = 10;    // number of samples for moving average
};

class llama_dynamic_transfer {
public:
    llama_dynamic_transfer();
    ~llama_dynamic_transfer();

    // Initialize with config and device info
    void init(const dynamic_transfer_config & cfg, 
              ggml_backend_t gpu_backend,
              ggml_backend_t cpu_backend,
              size_t total_vram_bytes);

    // Record a performance sample (called after each step)
    void record_sample(const perf_sample & sample);

    // Get current load estimate (normalized 0.0-1.0)
    double get_cpu_load() const { return _cpu_load; }
    double get_gpu_load() const { return _gpu_load; }
    double get_pcie_load() const { return _pcie_load; }

    // Decide which tensors to promote/demote for the next step
    // Uses the knapsack solution updated with current load conditions
    llama_tensor_profiler::placement_solution 
    decide_transfers(const llama_tensor_profiler & profiler,
                     size_t available_vram_bytes);

    // Execute a promotion (copy tensor from CPU to GPU temp buffer)
    // Returns true if successful
    bool promote_tensor(const std::string & tensor_name, 
                        ggml_tensor * tensor,
                        ggml_backend_buffer_t temp_buf);

    // Execute a demotion (release GPU temp buffer)
    void demote_tensor(const std::string & tensor_name);

    // Reset all state
    void reset();

private:
    dynamic_transfer_config _cfg;
    ggml_backend_t _gpu_backend = nullptr;
    ggml_backend_t _cpu_backend = nullptr;
    size_t _total_vram = 0;

    // Load history for moving average
    std::deque<perf_sample> _history;
    
    // Current load estimates
    double _cpu_load  = 0.0;
    double _gpu_load  = 0.0;
    double _pcie_load = 0.0;

    // Track promoted tensors
    struct promoted_tensor {
        std::string name;
        size_t      size;
        double      benefit;
        double      last_used; // timestamp
    };
    std::vector<promoted_tensor> _promoted;

    // Timing helpers
    std::chrono::high_resolution_clock::time_point _last_monitor;
    std::atomic<bool> _initialized{false};

    // Update load estimates from history
    void update_load_estimates();

    // Calculate effective benefit under current load
    double effective_benefit(double base_benefit, size_t size) const;
};

#endif // LLAMA_DYNAMIC_TRANSFER_H
