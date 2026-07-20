// llama-async-coord.h — Asynchronous CPU-GPU coordination (ATSInfer §4.2)
// Overlaps PCIe transfers with computation using CUDA streams.
#ifndef LLAMA_ASYNC_COORD_H
#define LLAMA_ASYNC_COORD_H

#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

// Async pipeline stage
enum class async_stage {
    idle,
    transfer_in,    // CPU → GPU
    compute_gpu,    // GPU kernel
    transfer_out,   // GPU → CPU (or next device)
    compute_cpu     // CPU kernel
};

// Async transfer request
struct async_transfer_request {
    void * dst          = nullptr;
    const void * src    = nullptr;
    size_t    size      = 0;
    bool      dst_is_gpu = false;  // true: CPU→GPU, false: GPU→CPU
    std::string tensor_name;
};

class llama_async_coord {
public:
    llama_async_coord();
    ~llama_async_coord();

    // Initialize with backends
    void init(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend);

    // Start an async transfer (CPU↔GPU)
    // Returns immediately; use sync() to wait for completion
    bool async_transfer(const async_transfer_request & req);

    // Execute GPU computation while transfers are in flight
    // The callback runs on the GPU stream
    bool compute_async(std::function<bool()> gpu_compute_fn);

    // Synchronize all pending operations
    bool sync();

    // Print transfer/compute stats
    void print_stats() const;

    // Check if the backend supports async operations
    bool supports_async() const { return _gpu_backend != nullptr && _cuda_stream_avail; }

    // Get the CUDA stream for manual operations (if available)
    void * get_cuda_stream() const { return _cuda_stream; }

    // Reset
    void reset();

private:
    ggml_backend_t _gpu_backend = nullptr;
    ggml_backend_t _cpu_backend = nullptr;
    
    // CUDA stream for async operations
    void * _cuda_stream = nullptr;
    bool   _cuda_stream_avail = false;

    // Track pending operations
    std::vector<async_transfer_request> _pending_transfers;
    bool _has_pending_compute = false;

    // Stats
    size_t _total_transfer_bytes = 0;
    double _total_transfer_us = 0;
    size_t _transfer_count = 0;
    double _total_compute_us = 0;
    size_t _compute_count = 0;

    // Helper to get CUDA stream from backend
    bool init_cuda_stream();
};

#endif // LLAMA_ASYNC_COORD_H
