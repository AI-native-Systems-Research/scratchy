// Stub for c10/cuda/CUDAException.h — replaces PyTorch dependency.
// Provides C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK macros.
#pragma once
#include <cstdio>
#include <cstdlib>

#define C10_CUDA_CHECK(expr)                                                   \
  do {                                                                         \
    cudaError_t __err = (expr);                                                \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(__err));                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                         \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                     \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",               \
              __FILE__, __LINE__, cudaGetErrorString(__err));                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)
