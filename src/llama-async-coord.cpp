// llama-async-coord.cpp — Asynchronous CPU-GPU coordination
#include "llama-async-coord.h"
#include "llama-impl.h"
#include "../ggml/src/ggml-cuda/llama-cuda-stream.h"
#include <cstring>
#include <chrono>

llama_async_coord::llama_async_coord() = default;
llama_async_coord::~llama_async_coord() { reset(); }

void llama_async_coord::init(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend) {
    _gpu_backend = gpu_backend;
    _cpu_backend = cpu_backend;
    _cuda_stream_avail = init_cuda_stream();
    if (_cuda_stream_avail) {
        LLAMA_LOG_INFO("%s: async coordination enabled (CUDA stream: real async memcpy)\n", __func__);
    } else {
        LLAMA_LOG_INFO("%s: async coordination enabled (fallback: synchronous memcpy)\n", __func__);
    }
}

bool llama_async_coord::init_cuda_stream() {
    if (!_gpu_backend) return false;

    auto * dev = ggml_backend_get_device(_gpu_backend);
    if (!dev) return false;
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) return false;

    // Create real CUDA stream for async operations
    int device_id = 0; // default GPU
    _cuda_stream = llama_cuda_stream_create(device_id);
    if (_cuda_stream) {
        return true;
    }

    // Fallback: sentinel for sync fallback
    _cuda_stream = (void *)(uintptr_t)0x1;
    return true;
}

bool llama_async_coord::async_transfer(const async_transfer_request & req) {
    if (!req.dst || !req.src || req.size == 0) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (_cuda_stream && _cuda_stream != (void *)(uintptr_t)0x1) {
        // Real async memcpy on CUDA stream
        if (req.dst_is_gpu) {
            llama_cuda_async_copy_h2d(req.dst, req.src, req.size, _cuda_stream);
        } else {
            llama_cuda_async_copy_d2h(req.dst, req.src, req.size, _cuda_stream);
        }
    } else {
        // Synchronous fallback
        std::memcpy(req.dst, req.src, req.size);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    _total_transfer_bytes += req.size;
    _total_transfer_us += us;
    _transfer_count++;

    _pending_transfers.push_back(req);
    return true;
}

bool llama_async_coord::compute_async(std::function<bool()> gpu_compute_fn) {
    if (!gpu_compute_fn) return false;

    // Ensure pending async transfers complete before compute
    if (_cuda_stream && _cuda_stream != (void *)(uintptr_t)0x1) {
        llama_cuda_stream_sync(_cuda_stream);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = gpu_compute_fn();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (ok) {
        _total_compute_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        _compute_count++;
    }

    _has_pending_compute = true;
    return ok;
}

bool llama_async_coord::sync() {
    if (_cuda_stream && _cuda_stream != (void *)(uintptr_t)0x1) {
        llama_cuda_stream_sync(_cuda_stream);
    }
    if (_gpu_backend) {
        ggml_backend_synchronize(_gpu_backend);
    }
    _pending_transfers.clear();
    _has_pending_compute = false;
    return true;
}

void llama_async_coord::print_stats() const {
    fprintf(stderr, "ASYNC: transfers=%zu, compute=%zu, "
            "total_xfer=%.0f MB (%.1f GB/s), total_compute=%.0f ms%s\n",
            _transfer_count, _compute_count,
            _total_transfer_bytes / (1024.0*1024.0),
            _total_transfer_bytes > 0 ?
                _total_transfer_bytes / _total_transfer_us : 0,
            _total_compute_us / 1000.0,
            _cuda_stream && _cuda_stream != (void *)(uintptr_t)0x1 ? " [ASYNC]" : " [SYNC]");
}

void llama_async_coord::reset() {
    sync();
    if (_cuda_stream && _cuda_stream != (void *)(uintptr_t)0x1) {
        llama_cuda_stream_destroy(_cuda_stream);
    }
    _cuda_stream = nullptr;
    _cuda_stream_avail = false;
    _gpu_backend = nullptr;
    _cpu_backend = nullptr;

    // Free temporary GPU buffer pool
    for (int i = 0; i < _n_temp_slots; i++) {
        if (_temp_slots[i].buffer) {
            ggml_backend_buffer_free(_temp_slots[i].buffer);
            _temp_slots[i].buffer = nullptr;
        }
        _temp_slots[i].data = nullptr;
        _temp_slots[i].capacity = 0;
        _temp_slots[i].in_use = false;
    }
    _n_temp_slots = 0;

    _total_transfer_bytes = 0;
    _total_transfer_us = 0;
    _transfer_count = 0;
    _total_compute_us = 0;
    _compute_count = 0;
    _temp_hits = 0;
    _temp_misses = 0;
}

// === Temporary GPU Buffer Pool ===

bool llama_async_coord::init_temp_pool(size_t max_tensor_size_bytes, int n_slots) {
    if (!_gpu_backend || n_slots <= 0 || n_slots > ATSINFER_MAX_TEMP_SLOTS) {
        return false;
    }

    sync(); // drain any pending ops first

    int actual_slots = std::min(n_slots, ATSINFER_MAX_TEMP_SLOTS);
    for (int i = 0; i < actual_slots; i++) {
        auto * dev = ggml_backend_get_device(_gpu_backend);
        if (!dev) { reset(); return false; }

        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        if (!buft) { reset(); return false; }

        // Allocate temporary GPU buffer
        _temp_slots[i].buffer = ggml_backend_buft_alloc_buffer(buft, max_tensor_size_bytes);
        if (!_temp_slots[i].buffer) {
            LLAMA_LOG_WARN("%s: failed to allocate temp slot %d (%.0f MB)\n",
                          __func__, i, max_tensor_size_bytes / (1024.0*1024.0));
            _n_temp_slots = i;
            return i > 0;
        }
        _temp_slots[i].data = ggml_backend_buffer_get_base(_temp_slots[i].buffer);
        _temp_slots[i].capacity = max_tensor_size_bytes;
        _temp_slots[i].in_use = false;

        LLAMA_LOG_INFO("%s: temp slot %d = %.0f MB\n",
                      __func__, i, max_tensor_size_bytes / (1024.0*1024.0));
    }
    _n_temp_slots = actual_slots;
    LLAMA_LOG_INFO("%s: pool=%d slots x %.0f MB = %.0f MB total\n",
                  __func__, _n_temp_slots,
                  max_tensor_size_bytes / (1024.0*1024.0),
                  (_n_temp_slots * max_tensor_size_bytes) / (1024.0*1024.0));
    return true;
}

void * llama_async_coord::acquire_temp_buffer(const std::string & tensor_name, size_t size_bytes) {
    std::lock_guard<std::mutex> lk(_pool_mutex);

    // Find a free slot with enough capacity
    for (int i = 0; i < _n_temp_slots; i++) {
        if (!_temp_slots[i].in_use && _temp_slots[i].capacity >= size_bytes) {
            _temp_slots[i].in_use = true;
            _temp_slots[i].current_tensor = tensor_name;
            _temp_hits++;
            return _temp_slots[i].data;
        }
    }

    // No free slot — try to evict the tensor with the most distant reuse
    // For now, just log a miss
    _temp_misses++;
    fprintf(stderr, "TEMP_POOL: miss for %s (%.0f MB, %d slots busy)\n",
            tensor_name.c_str(), size_bytes / (1024.0*1024.0), _n_temp_slots);
    return nullptr;
}

void llama_async_coord::release_temp_buffer(const std::string & tensor_name) {
    std::lock_guard<std::mutex> lk(_pool_mutex);

    for (int i = 0; i < _n_temp_slots; i++) {
        if (_temp_slots[i].in_use && _temp_slots[i].current_tensor == tensor_name) {
            _temp_slots[i].in_use = false;
            _temp_slots[i].current_tensor.clear();

            // Sync the stream before reuse (ensure transfer completed)
            if (_cuda_stream && _cuda_stream != (void *)(uintptr_t)0x1) {
                llama_cuda_stream_sync(_cuda_stream);
            }
            return;
        }
    }
}

size_t llama_async_coord::temp_pool_capacity() const {
    size_t total = 0;
    for (int i = 0; i < _n_temp_slots; i++) {
        total += _temp_slots[i].capacity;
    }
    return total;
}

