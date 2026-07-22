// llama-dynamic-transfer.cpp — Dynamic Transfer Scheduling (ATSInfer §4.4)
// Implements Algorithm 2 from the paper.
#include "llama-dynamic-transfer.h"
#include "llama-impl.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>

llama_dynamic_transfer::llama_dynamic_transfer() = default;
llama_dynamic_transfer::~llama_dynamic_transfer() = default;

void llama_dynamic_transfer::init(double pcie_bw_gbs) {
    _pcie_bw_gbs = (pcie_bw_gbs > 0) ? pcie_bw_gbs : 30.0;
    fprintf(stderr, "DT_SCHED: init with PCIe %.1f GB/s\n", _pcie_bw_gbs);
}

void llama_dynamic_transfer::reset() {
    _solve_count = 0;
    _total_solve_us = 0;
}

dt_promotion_plan llama_dynamic_transfer::solve(
    const std::vector<dt_tensor_info> & tensors,
    double current_pcie_bw_gbs,
    double current_cpu_slowdown,
    double current_gpu_slowdown
) const {
    auto t0 = std::chrono::high_resolution_clock::now();
    _solve_count++;

    dt_promotion_plan plan;
    int n = (int)tensors.size();
    if (n == 0) return plan;

    double bw = (current_pcie_bw_gbs > 0) ? current_pcie_bw_gbs : _pcie_bw_gbs;
    double bw_bytes_per_ms = bw * 1e9 / 1e3; // bytes/ms

    // Algorithm 2 from the paper
    // DP[i][GPU] = min latency ending with tensor i on GPU
    // DP[i][CPU] = min latency ending with tensor i on CPU
    // For CPU-resident tensors, we can either:
    //   a) Keep on CPU: extend previous CPU state
    //   b) Promote to GPU: find best start point j, pay exposed transfer
    
    const double INF = 1e18;
    std::vector<double> dp_gpu(n, INF), dp_cpu(n, INF);
    std::vector<char> choice_gpu(n, -1), choice_cpu(n, -1);
    // -1 = infeasible, 0 = from CPU, 1 = from GPU

    // Precompute activation switching costs c_i
    // c_i = activation_size / bandwidth (time to transfer activations across backends)
    std::vector<double> c(n, 0);
    for (int i = 0; i < n; i++) {
        if (tensors[i].activation_size > 0) {
            c[i] = (double)tensors[i].activation_size / bw_bytes_per_ms;
        } else {
            // Default: ~16 KB activation for RMS norm, ~2 MB for attention
            c[i] = (tensors[i].op_type == "MUL_MAT" || tensors[i].op_type == "MUL_MAT_ID")
                   ? (2.0 * 1024.0 * 1024.0) / bw_bytes_per_ms   // ~0.05ms @ 30GB/s
                   : (16.0 * 1024.0) / bw_bytes_per_ms;          // ~0.0005ms
        }
    }

    // Precompute weight transfer times w_i (time to move weights from CPU to GPU)
    std::vector<double> w(n, 0);
    for (int i = 0; i < n; i++) {
        if (!tensors[i].is_gpu_default) {
            // This tensor is CPU-resident → if promoted, must transfer weights
            w[i] = (double)tensors[i].size / bw_bytes_per_ms;
        }
    }

    // Compute adjusted execution times with slowdown factors
    std::vector<double> t_cpu_adj(n), t_gpu_adj(n);
    for (int i = 0; i < n; i++) {
        t_cpu_adj[i] = tensors[i].t_cpu * current_cpu_slowdown;
        t_gpu_adj[i] = tensors[i].t_gpu * current_gpu_slowdown;
    }

    // Precompute seg(j,i) = total execution + activation transfer along default
    // path between CPU-side endpoint j and tensor i.
    // O(n²) precompute, stored as 2D style but using flat vector for O(n) memory
    // seg_start[i] = seg values for all j < i for tensor i
    // We compute seg on the fly in the inner loop (O(n²) time, O(1) space per pair)

    for (int i = 0; i < n; i++) {
        // --- States for tensor i on GPU ---
        double best_gpu = INF;
        int best_gpu_from = -1;

        // If default is GPU, just extend previous state
        if (tensors[i].is_gpu_default) {
            // Previous on GPU: just add t_gpu
            if (i == 0) {
                dp_gpu[i] = t_gpu_adj[i];
            } else {
                // Extend from GPU state: no switching cost
                double from_gpu = dp_gpu[i-1] + t_gpu_adj[i];
                if (from_gpu < dp_gpu[i]) {
                    dp_gpu[i] = from_gpu;
                    choice_gpu[i] = 1;
                }
                // Extend from CPU state: pay activation transfer CPU→GPU
                double from_cpu = dp_cpu[i-1] + c[i] + t_gpu_adj[i];
                if (from_cpu < dp_gpu[i]) {
                    dp_gpu[i] = from_cpu;
                    choice_gpu[i] = 0;
                }
            }
        } else {
            // CPU-resident tensor — consider promoting to GPU
            // Option: extend from previous GPU state (pay activation)
            if (i > 0 && dp_gpu[i-1] < INF) {
                double from_gpu = dp_gpu[i-1] + c[i] + t_gpu_adj[i];
                if (from_gpu < best_gpu) {
                    best_gpu = from_gpu;
                    best_gpu_from = 1; // from GPU
                }
            }

            // Option: promote from a CPU segment endpoint j
            // Try all j from 0 to i-1 where tensor j ends on CPU
            // seg(j,i) = computation + activation on default path from j to i
            if (i > 0) {
                for (int j = 0; j < i; j++) {
                    // Can only promote from a CPU endpoint
                    if (dp_cpu[j] >= INF) continue;
                    
                    // Compute seg(j,i): execution + activation on default path
                    double seg = 0;
                    for (int k = j; k < i; k++) {
                        if (tensors[k].is_gpu_default) {
                            seg += t_gpu_adj[k];
                        } else {
                            seg += t_cpu_adj[k];
                        }
                        // Add activation transfer cost at boundaries
                        if (k > j && tensors[k-1].is_gpu_default != tensors[k].is_gpu_default) {
                            seg += c[k];
                        }
                    }
                    // Also count the activation at the start of segment
                    if (j > 0 && tensors[j-1].is_gpu_default != tensors[j].is_gpu_default) {
                        seg += c[j];
                    }

                    // Exposed transfer time = max(weight_transfer - overlap_window, 0)
                    double overlap = seg; // seg is the overlap window
                    double exposed = std::max(w[i] - overlap, 0.0);
                    double total = dp_cpu[j] + exposed + t_gpu_adj[i];

                    if (total < best_gpu) {
                        best_gpu = total;
                        best_gpu_from = 0; // from CPU
                    }
                }
            } else {
                // i=0: first tensor, no previous
                double exposed = std::max(w[0] - 0.0, 0.0);
                best_gpu = exposed + t_gpu_adj[0];
                best_gpu_from = -1;
            }

            dp_gpu[i] = best_gpu;
            choice_gpu[i] = (char)best_gpu_from;
        }

        // --- States for tensor i on CPU ---
        double best_cpu = INF;
        int best_cpu_from = -1;

        // Option: previous on CPU (no switch, no promotion)
        if (i == 0) {
            dp_cpu[i] = t_cpu_adj[0];
            choice_cpu[i] = 0;
        } else {
            // From CPU
            double from_cpu = dp_cpu[i-1] + t_cpu_adj[i];
            if (from_cpu < best_cpu) {
                best_cpu = from_cpu;
                best_cpu_from = 0;
            }
            // From GPU (pay activation transfer GPU→CPU)
            double from_gpu = dp_gpu[i-1] + c[i] + t_cpu_adj[i];
            if (from_gpu < best_cpu) {
                best_cpu = from_cpu;
                best_cpu_from = 1;
            }
            dp_cpu[i] = best_cpu;
            choice_cpu[i] = (char)best_cpu_from;
        }
    }

    // Find best final state
    int best_end_gpu = (dp_gpu[n-1] <= dp_cpu[n-1]) ? 1 : 0;
    double best_latency = std::min(dp_gpu[n-1], dp_cpu[n-1]);

    // Backtrack to get promotion plan
    // We only care about which tensors are promoted (CPU→GPU for non-default-GPU tensors)
    // Backward pass: follow the choices
    int cur = best_end_gpu; // 0 = CPU, 1 = GPU
    for (int i = n - 1; i >= 0; i--) {
        bool on_gpu = (cur == 1);

        if (on_gpu && !tensors[i].is_gpu_default) {
            // This CPU-resident tensor was promoted to GPU
            plan.promote_tensor_names.push_back(tensors[i].name);
            plan.total_transfer_bytes += tensors[i].size;
            plan.n_promoted++;
        }

        // Follow choice to previous state
        if (on_gpu) {
            cur = (choice_gpu[i] == 0) ? 0 : 1;
        } else {
            cur = (choice_cpu[i] == 0) ? 0 : 1;
        }
    }

    // Reverse the list (we built it backwards)
    std::reverse(plan.promote_tensor_names.begin(), plan.promote_tensor_names.end());

    // Estimate benefit vs no-promotion baseline
    double no_promo_latency = 0;
    for (int i = 0; i < n; i++) {
        no_promo_latency += tensors[i].is_gpu_default ? t_gpu_adj[i] : t_cpu_adj[i];
    }
    plan.estimated_benefit_ms = no_promo_latency - best_latency;
    plan.exposed_transfer_ms = best_latency - no_promo_latency + plan.estimated_benefit_ms;
    plan.valid = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    _total_solve_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

    return plan;
}

void llama_dynamic_transfer::print_stats() const {
    fprintf(stderr, "DT_SCHED: solves=%zu, avg_solve=%.1f us\n",
            _solve_count,
            _solve_count > 0 ? _total_solve_us / _solve_count : 0.0);
}
