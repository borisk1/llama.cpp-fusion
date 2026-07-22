// llama-dynamic-transfer.h — Dynamic Transfer Scheduling (ATSInfer §4.4, Algorithm 2)
// Decides which CPU-resident tensors to temporarily promote to GPU each step,
// overlapping weight transfers with computation.
#ifndef LLAMA_DYNAMIC_TRANSFER_H
#define LLAMA_DYNAMIC_TRANSFER_H

#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

// Info about one tensor for the dynamic transfer solver
struct dt_tensor_info {
    std::string name;
    size_t      size;           // tensor size in bytes
    bool        is_gpu_default; // true if static placement put it on GPU
    double      t_cpu;          // measured CPU execution time (ms)
    double      t_gpu;          // measured GPU execution time (ms)
    size_t      activation_size; // activation size at backend boundary (bytes)
    int         layer;
    std::string op_type;
};

// Result: which tensors to promote and what to transfer
struct dt_promotion_plan {
    // Tensors to promote: CPU → GPU for this step
    std::vector<std::string> promote_tensor_names;
    size_t total_transfer_bytes = 0;
    double estimated_benefit_ms = 0;  // time saved vs no promotion
    double exposed_transfer_ms  = 0;  // non-overlapped transfer time
    int    n_promoted           = 0;
    bool   valid                = false;

    // For each promoted tensor, the estimated overlap window
    std::vector<double> overlap_windows_ms;
};

// Dynamic Transfer Scheduler (Algorithm 2 from ATSInfer paper)
// O(n²) time, O(n) auxiliary space
// n = number of CPU-resident tensors (after static placement)
class llama_dynamic_transfer {
public:
    llama_dynamic_transfer();
    ~llama_dynamic_transfer();

    // Initialize with PCIe bandwidth estimate
    void init(double pcie_bw_gbs);

    // Reset internal state
    void reset();

    // Solve dynamic transfer DP for current step
    // Input: list of all tensors in execution order, current CPU/GPU/PCIe measurements
    // Returns: which tensors to promote
    dt_promotion_plan solve(
        const std::vector<dt_tensor_info> & tensors,
        double current_pcie_bw_gbs,
        double current_cpu_slowdown,  // 1.0 = normal, >1.0 = CPU slower
        double current_gpu_slowdown   // 1.0 = normal, >1.0 = GPU slower
    ) const;

    // Print stats
    void print_stats() const;

private:
    double _pcie_bw_gbs = 30.0; // default PCIe 3.0 x16

    // Stats
    mutable size_t _solve_count = 0;
    mutable double _total_solve_us = 0;
};

#endif // LLAMA_DYNAMIC_TRANSFER_H
