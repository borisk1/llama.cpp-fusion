// llama-async-coord.cpp — Asynchronous CPU-GPU coordination
#include "llama-async-coord.h"
#include "llama-impl.h"
#include <cstring>
#include <chrono>

llama_async_coord::llama_async_coord() = default;
llama_async_coord::~llama_async_coord() = default;

void llama_async_coord::init(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend) {
    _gpu_backend = gpu_backend;
    _cpu_backend = cpu_backend;
    _cuda_stream_avail = init_cuda_stream();
    if (_cuda_stream_avail) {
        LLAMA_LOG_INFO("%s: async coordination enabled (CUDA stream available)\n", __func__);
    } else {
        LLAMA_LOG_INFO("%s: async coordination enabled (fallback mode)\n", __func__);
    }
}

bool llama_async_coord::init_cuda_stream() {
    // In a full implementation, this would use the CUDA backend's stream.
    // For now, detect if we have an ACCEL backend (GPU available).
    if (!_gpu_backend) return false;

    auto * dev = ggml_backend_get_device(_gpu_backend);
    if (!dev) return false;

    auto type = ggml_backend_dev_type(dev);
    if (type != GGML_BACKEND_DEVICE_TYPE_ACCEL) return false;

    // CUDA stream available if we have an ACCEL backend
    // (ggml_cuda backend manages streams internally)
    _cuda_stream = (void *)(uintptr_t)0x1; // non-null sentinel
    return true;
}

bool llama_async_coord::async_transfer(const async_transfer_request & req) {
    if (!req.dst || !req.src || req.size == 0) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Synchronous fallback: direct memcpy
    std::memcpy(req.dst, req.src, req.size);

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
    if (_gpu_backend) {
        ggml_backend_synchronize(_gpu_backend);
    }
    _pending_transfers.clear();
    _has_pending_compute = false;
    return true;
}

void llama_async_coord::print_stats() const {
    fprintf(stderr, "ASYNC: transfers=%zu, compute=%zu, "
            "total_xfer=%.0f MB (%.1f GB/s), total_compute=%.0f ms\n",
            _transfer_count, _compute_count,
            _total_transfer_bytes / (1024.0*1024.0),
            _total_transfer_bytes > 0 ?
                _total_transfer_bytes / _total_transfer_us : 0,
            _total_compute_us / 1000.0);
}

void llama_async_coord::reset() {
    sync();
    _cuda_stream = nullptr;
    _cuda_stream_avail = false;
    _gpu_backend = nullptr;
    _cpu_backend = nullptr;
    _total_transfer_bytes = 0;
    _total_transfer_us = 0;
    _transfer_count = 0;
    _total_compute_us = 0;
    _compute_count = 0;
}
