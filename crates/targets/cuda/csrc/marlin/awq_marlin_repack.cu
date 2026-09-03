/*
 * AWQ weight repacking for Marlin tile layout — C entry point for vllm-rs FFI.
 *
 * Repacks AWQ INT4-in-INT32 packed weights from the standard HuggingFace
 * layout [K, N/pack_factor] (with AWQ interleave) into Marlin's tiled format
 * required by the fused GEMM kernel.
 *
 * Adapted from Python vLLM's awq_marlin_repack.cu with torch deps removed.
 */

#include "marlin.cuh"
#include <cassert>
#include <cstdio>

#define MARLIN_CHECK(cond, ...)                              \
  do {                                                       \
    if (!(cond)) {                                           \
      fprintf(stderr, "AWQ repack check failed: %s\n", #cond); \
      abort();                                               \
    }                                                        \
  } while (0)

namespace marlin {

template <int const num_threads, int const num_bits, bool is_a_8bit>
__global__ void awq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  constexpr int target_tile_n_size = tile_n_size / (is_a_8bit ? 2 : 1);
  constexpr int target_tile_k_size = tile_k_size * (is_a_8bit ? 2 : 1);
  int k_tiles = size_k / target_tile_k_size;
  int n_tiles = size_n / target_tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  auto wait_for_stage = [&]() {
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int tile_n_ints = target_tile_n_size / pack_factor;

  constexpr int stage_n_threads = tile_n_ints / 4;
  constexpr int stage_k_threads = target_tile_k_size;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * target_tile_n_size;
    int first_n_packed = first_n / pack_factor;

    int4* sh_ptr = sh + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * target_tile_k_size;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k + k_id) * (size_n / pack_factor) +
                                     first_n_packed + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * (is_a_8bit ? 4 : 2);

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = (warp_id / (is_a_8bit ? 2 : 1)) * 16 + tc_col;
    int cur_n_packed = cur_n / pack_factor;
    int cur_n_pos = cur_n % pack_factor;

    constexpr int sh_stride = tile_n_ints;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    // Undo interleaving
    int cur_n_pos_unpacked;
    if constexpr (num_bits == 4) {
      constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    } else {
      constexpr int undo_pack[4] = {0, 2, 1, 3};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    }

    uint32_t vals[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if constexpr (is_a_8bit) {
        int cur_elem = tc_row + i;

        int packed_src_0 =
            sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) * (warp_id % 2) +
                             sh_stride * cur_elem];
        int packed_src_1 =
            sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) * (warp_id % 2) +
                             sh_stride * (cur_elem + 16)];

        vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
        vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
      } else {
        int cur_elem = tc_row + tc_offsets[i];

        int packed_src_0 =
            sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
        int packed_src_1 = sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) +
                                            sh_stride * cur_elem];

        vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
        vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
      }
    }

    constexpr int tile_size =
        target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    if constexpr (!is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else if constexpr (is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int ii = is_a_8bit ? i : pack_idx[i];
        res1 |= vals[ii] << (i * 8);
        res2 |= vals[4 + ii] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

}  // namespace marlin

#define CALL_IF(NUM_BITS)                                                  \
  else if (num_bits == NUM_BITS) {                                         \
    cudaFuncSetAttribute(                                                  \
        marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS, \
                                         false>,                            \
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);      \
    marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS,     \
                                     false>                                 \
        <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(      \
            b_q_weight_ptr, out_ptr, size_k, size_n);                      \
  }

extern "C" {

/// Repack AWQ INT4 weights from HF layout to Marlin tiled layout.
///
/// b_q_weight: [size_k, size_n/pack_factor] packed INT4 weights (device ptr, u32)
/// out:        output buffer, size [size_k/tile_size, size_n*tile_size/pack_factor] (device ptr, u32)
/// size_k:     number of input features (unquantized)
/// size_n:     number of output features
void awq_marlin_repack_4bit(
    const uint32_t* b_q_weight, uint32_t* out,
    int size_k, int size_n,
    cudaStream_t stream, int device_id) {

  int num_bits = 4;

  MARLIN_CHECK(size_k % marlin::tile_k_size == 0);
  MARLIN_CHECK(size_n % marlin::tile_n_size == 0);

  const uint32_t* b_q_weight_ptr = b_q_weight;
  uint32_t* out_ptr = out;

  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, device_id);

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
  MARLIN_CHECK(max_shared_mem > 0);

  if (false) {
  }
  CALL_IF(4)
  else {
    MARLIN_CHECK(false);
  }
}

}  // extern "C"
