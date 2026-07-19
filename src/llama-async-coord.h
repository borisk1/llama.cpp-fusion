// llama-async-coord.h — Async CPU-GPU coordination (ATSInfer §4.2)
#ifndef LLAMA_ASYNC_COORD_H
#define LLAMA_ASYNC_COORD_H

#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <functional>

enum class async_stage { idle, transfer_in, compute_gpu, transfer_out, compute_cpu };

struct async_transfer_request {
    void * dst = nullptr;
    const void * src = nullptr;
    size_t size = 0;
    bool dst_is_gpu = false;
    std::string tensor_name;
};

class llama_async_coord {
public:
    llama_async_coord();
    ~llama_async_coord();

    void init(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend);
    bool async_transfer(const async_transfer_request & req);
    bool compute_async(std::function<bool()> gpu_compute_fn);
    bool sync();
    bool supports_async() const { return _gpu_backend != nullptr && _cuda_stream_avail; }
    void * get_cuda_stream() const { return _cuda_stream; }
    void reset();

private:
    ggml_backend_t _gpu_backend = nullptr;
    ggml_backend_t _cpu_backend = nullptr;
    void * _cuda_stream = nullptr;
    bool _cuda_stream_avail = false;
    std::vector<async_transfer_request> _pending_transfers;
    bool _has_pending_compute = false;
    bool init_cuda_stream();
};

#endif
