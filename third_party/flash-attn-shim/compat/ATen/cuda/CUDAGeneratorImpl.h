// Stub for ATen/cuda/CUDAGeneratorImpl.h — replaces PyTorch dependency.
// Provides at::PhiloxCudaState used by Flash_fwd_params::philox_args.
// We never use dropout, so this struct is zero-initialized and never read.
#pragma once
#include <cstdint>
#include <optional>  // upstream gets this transitively via PyTorch headers

namespace at {
struct PhiloxCudaState {
    uint64_t seed_;
    uint64_t offset_;
    PhiloxCudaState() : seed_(0), offset_(0) {}
    PhiloxCudaState(uint64_t seed, uint64_t offset) : seed_(seed), offset_(offset) {}
};
} // namespace at
