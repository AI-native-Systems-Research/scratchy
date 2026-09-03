// Standalone stub: we don't use dropout so PhiloxCudaState unpacking is unused.
// The kernel code references at::cuda::philox::unpack but we stub it out.
#pragma once

namespace at { namespace cuda { namespace philox {
__forceinline__ __device__ auto unpack(at::PhiloxCudaState state) {
    return std::make_tuple(state.seed_, state.offset_);
}
}}}  // namespace at::cuda::philox
