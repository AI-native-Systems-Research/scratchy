// SPDX-License-Identifier: Apache-2.0
// Common utilities for CUTLASS scaled_mm kernels.
// Stripped of PyTorch dependencies — uses raw CUDA/CUTLASS APIs only.
#pragma once

#include "cutlass/cutlass.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"

#define CUTLASS_CHECK(status)                                        \
  {                                                                  \
    cutlass::Status error = status;                                  \
    if (error != cutlass::Status::kSuccess) {                        \
      fprintf(stderr, "CUTLASS error: %s at %s:%d\n",               \
              cutlassGetStatusString(error), __FILE__, __LINE__);    \
      abort();                                                       \
    }                                                                \
  }

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

// Architecture guards — restrict kernel execution to specific SM ranges.
template <typename Kernel>
struct enable_sm75_to_sm80 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[75, 80).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 890
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[80, 89).\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm89_to_sm90 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[89, 90).\n");
    asm("trap;");
  #endif
#endif
  }
};

// CUTLASS 3.x (Hopper) kernels invoke via operator(), not the static invoke()
// used by the 2.x guards above. This guard restricts the C3X SM90 FP8 GEMM to
// sm_90+ devices and traps otherwise (matches Python vLLM's enable_sm90_or_later).
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm >= 90.\n");
    asm("trap;");
  #endif
#endif
  }
};

template <typename Kernel>
struct enable_sm89_to_sm100 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__
  #if __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 1000
    Kernel::invoke(std::forward<Args>(args)...);
  #else
    printf("This kernel only supports sm[89, 100).\n");
    asm("trap;");
  #endif
#endif
  }
};
