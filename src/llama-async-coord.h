// llama-async-coord.h — Asynchronous CPU-GPU coordination (ATSInfer §4.2)
// Extended with temporary GPU buffer pool for Dynamic Transfer (§4.4)
#ifndef LLAMA_ASYNC_COORD_H
#define LLAMA_ASYNC_COORD_H

#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>

// Max temporary GPU buffer slots for dynamic weight promotion
#define ATSINFER_MAX_TEMP_SLOTS 8

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

// Temporary GPU buffer slot for Dynamic Transfer
// Each slot holds a pre-allocated GPU buffer that can store
// one promoted weight at a time. Buffers are reused across tensors
// whose live ranges don't overlap.
struct gpu_temp_slot {
    ggml_backend_buffer_t buffer = nullptr;
    void *                data   = nullptr;
    size_t                capacity = 0;  // allocated size
    bool                  in_use  = false;
    std::string           current_tensor; // tensor currently occupying this slot
};

class llama_async_coord {
public:
    llama_async_coord();
    ~llama_async_coord();

    // Initialize with backends
    void init(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend);

    // Start an async transfer (CPU↔GPU)
    bool async_transfer(const async_transfer_request & req);

    // Execute GPU computation while transfers are in flight
    bool compute_async(std::function<bool()> gpu_compute_fn);

    // Synchronize all pending operations
    bool sync();

    // === Temporary GPU Buffer Pool (§4.2.1) ===

    // Allocate temporary GPU buffer pool (call after model is loaded,
    // with max_tensor_size = largest weight tensor in the model)
    bool init_temp_pool(size_t max_tensor_size_bytes, int n_slots = ATSINFER_MAX_TEMP_SLOTS);

    // Acquire a temporary GPU buffer slot for a tensor.
    // Returns pointer to GPU memory, or nullptr if all slots busy.
    // The slot is marked in_use until release_temp_buffer().
    void * acquire_temp_buffer(const std::string & tensor_name, size_t size_bytes);

    // Release a temporary GPU buffer slot
    void release_temp_buffer(const std::string & tensor_name);

    // Get total capacity of temp pool
    size_t temp_pool_capacity() const;

    // Print stats
    void print_stats() const;

    // Check if backend supports async
    bool supports_async() const { return _gpu_backend != nullptr && _cuda_stream_avail; }

    // Get the CUDA stream for manual ops
    void * get_cuda_stream() const { return _cuda_stream; }

    // Reset (frees all buffers)
    void reset();

private:
    ggml_backend_t _gpu_backend = nullptr;
    ggml_backend_t _cpu_backend = nullptr;

    // CUDA stream for async operations
    void * _cuda_stream = nullptr;
    bool   _cuda_stream_avail = false;

    // Pending transfer tracking
    std::vector<async_transfer_request> _pending_transfers;
    bool _has_pending_compute = false;

    // Temporary GPU buffer pool
    gpu_temp_slot _temp_slots[ATSINFER_MAX_TEMP_SLOTS];
    int           _n_temp_slots = 0;
    mutable std::mutex _pool_mutex;

    // Stats
    size_t _total_transfer_bytes = 0;
    double _total_transfer_us = 0;
    size_t _transfer_count = 0;
    double _total_compute_us = 0;
    size_t _compute_count = 0;
    size_t _temp_hits = 0;
    size_t _temp_misses = 0;

    // Helper
    bool init_cuda_stream();
};

#endif // LLAMA_ASYNC_COORD_H
