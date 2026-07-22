#pragma once
// llama-cuda-stream.h — C-callable CUDA stream wrappers

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Create/destroy a non-blocking CUDA stream
void * llama_cuda_stream_create(int device_id);
void   llama_cuda_stream_destroy(void * stream);

// Async memcpy on the given stream (returns immediately)
int  llama_cuda_async_copy_h2d(void * dst, const void * src, size_t bytes, void * stream);
int  llama_cuda_async_copy_d2h(void * dst, const void * src, size_t bytes, void * stream);

// Sync/query stream
void llama_cuda_stream_sync(void * stream);
int  llama_cuda_stream_query(void * stream);

// CUDA event timing
void * llama_cuda_event_create();
void   llama_cuda_event_destroy(void * evt);
int    llama_cuda_event_record(void * evt, void * stream);
int    llama_cuda_event_sync(void * evt);
float  llama_cuda_event_elapsed_ms(void * start, void * stop);

// Simple GPU compute throughput measurement (proxy)
float llama_cuda_measure_gpu_compute(void * stream);
// Measure GPU execution time of a function using CUDA events.
// 'fn' is called once (after warmup), and the GPU time is measured via
// cudaEventRecord before and after the call, synced on the default stream.
// Returns elapsed GPU time in milliseconds, or 0 on error.
float llama_cuda_time_gpu(void (*fn)(void*), void * arg);

// Async H2D copy on a specific stream.
// Copies 'size' bytes from 'src' (host) to 'dst' (device) on the given stream.
// The stream must be created with llama_cuda_stream_create() or be NULL.
// Returns 0 on success, non-zero on error.
int llama_cuda_async_h2d(void * dst, const void * src, size_t size, void * stream);

#ifdef __cplusplus
}
#endif
