// llama-tensor-profiler.h — Tensor profiling for ATSInfer-style scheduling
// Measures per-tensor execution time on CPU and GPU to calculate
// Empirical Performance Density (EPD) for optimal tensor placement.
#ifndef LLAMA_TENSOR_PROFILER_H
#define LLAMA_TENSOR_PROFILER_H

#include "llama-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

struct tensor_profile_entry {
    std::string name;       // tensor name (e.g. "blk.0.attn_norm.weight")
    size_t      size;       // memory footprint in bytes
    double      t_cpu;      // execution time on CPU (ms)
    double      t_gpu;      // execution time on GPU (ms)
    double      benefit;    // t_cpu - t_gpu (time saved on GPU)
    double      epd_cpu;    // empirical performance density CPU (t_cpu / size)
    double      epd_gpu;    // empirical performance density GPU (t_gpu / size)
    int         layer;      // layer index (-1 for global)
    std::string op_type;    // GGML op type name
    int         count;      // how many times measured
};

class llama_tensor_profiler {
public:
    llama_tensor_profiler() = default;
    ~llama_tensor_profiler() = default;

    // Begin timing an op on a specific backend
    void begin_op(const std::string & tensor_name, const std::string & op_type, 
                  ggml_backend_dev_t backend, size_t tensor_size);

    // End timing and record
    void end_op(const std::string & tensor_name);

    // Calculate EPD and benefit for all recorded tensors
    void calculate_epd();

    // Knapsack solver: select optimal GPU placement given VRAM budget
    // Using DP formulation from ATSInfer paper (arXiv:2607.10183)
    // max Σ(r_i * x_i) - Σ(c_i * switch_i) subject to Σ(s_i * x_i) ≤ M
    struct placement_solution {
        std::vector<std::string> gpu_tensors;  // tensors placed on GPU
        std::vector<std::string> cpu_tensors;  // tensors placed on CPU
        double total_benefit;                   // total time saved
        double total_size;                      // total VRAM used
    };

    placement_solution solve_knapsack(double vram_budget_bytes, double pcie_bw_gbs) const;

    // Print profiling report
    void print_report() const;

    // Record a measured tensor execution time
    void record_measurement(const std::string & name, const std::string & op_type,
                            size_t size, double t_cpu_ms, double t_gpu_ms);

    // Generate tensor buffer type overrides from knapsack solution
    // Returns override commands that can be passed to --override-tensor
    std::vector<std::string> generate_overrides(const placement_solution & solution) const;

    // Measure tensor execution time on a specific backend
    // Creates a minimal computation graph and runs it multiple times
    double measure_tensor_on_backend(const std::string & tensor_name,
                                     ggml_tensor * tensor,
                                     ggml_backend_t backend,
                                     const std::string & op_type);

    // Get top N tensors by benefit/size ratio for GPU placement
    std::vector<std::string> get_top_tensors(int n, double vram_budget_bytes) const;

    // Clear all data
    void clear();

private:
    struct active_timing {
        std::chrono::high_resolution_clock::time_point start;
        bool     is_cpu;
        size_t   size;
        std::string op_type;
    };

    std::unordered_map<std::string, tensor_profile_entry> entries;
    std::unordered_map<std::string, active_timing> active;
};

#endif // LLAMA_TENSOR_PROFILER_H
