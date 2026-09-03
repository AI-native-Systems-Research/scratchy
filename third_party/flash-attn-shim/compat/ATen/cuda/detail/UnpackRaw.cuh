// Stub for ATen/cuda/detail/UnpackRaw.cuh — replaces PyTorch dependency.
// Provides at::cuda::philox::unpack used by flash_fwd_kernel.h for dropout RNG.
// We never use dropout, so this always returns (0, 0).
#pragma once
#include <tuple>
#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace at { namespace cuda { namespace philox {

__device__ __forceinline__ std::tuple<uint64_t, uint64_t>
unpack(at::PhiloxCudaState philox_args) {
    return {philox_args.seed_, philox_args.offset_};
}

}}} // namespace at::cuda::philox
