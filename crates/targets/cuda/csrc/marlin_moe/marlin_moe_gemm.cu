/*
 * Marlin MoE W4A16 fused GEMM — C entry point for vllm-rs FFI.
 *
 * Wraps marlin_moe_wna16::marlin_mm() (the dispatch function from Python vLLM's
 * moe/marlin_moe_wna16/ops.cu) with extern "C" functions callable from Rust.
 *
 * This is the MoE variant: routes tokens to experts via sorted_token_ids,
 * expert_ids, and applies topk weights as part of the GEMM reduction.
 */

#include "kernel.h"
#include "core/scalar_type.hpp"
#include <cassert>
#include <cstdio>
#include <algorithm>

using std::min;
using std::max;

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

#define MARLIN_CHECK(cond, ...)                                      \
  do {                                                               \
    if (!(cond)) {                                                   \
      fprintf(stderr, "Marlin MoE GEMM check failed: %s\n", #cond); \
      abort();                                                       \
    }                                                                \
  } while (0)

namespace marlin_moe_wna16 {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr, int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr, int size_m,
    int size_k, int top_k) {
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
  int32_t block_sorted_ids[moe_block_size];
  int block_num_valid_tokens = 0;
  int64_t old_expert_id = 0;
  int64_t expert_id = 0;
  int row_stride = size_k * sizeof(half) / 16;

  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] =
          ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= size_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }
  };

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int in_offset = (row / top_k) * row_stride;
    int out_offset = row * row_stride;

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + in_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];
      out_half[cur_k] = a_row_half[src_pos];
      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];
        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
    old_expert_id = expert_id;
    int tmp_expert_id = expert_ids_ptr[index];
    if (tmp_expert_id == -1) continue;
    expert_id = tmp_expert_id;
    perm_int_ptr += (expert_id - old_expert_id) * size_k;
    read_moe_block_data(index);

    for (int i = 0; i < block_num_valid_tokens; i++)
      permute_row(block_sorted_ids[i]);
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * stages * 2;
    load_groups = max(load_groups, 32);
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;
    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, bool m_block_size_8,
                          int thread_m_blocks, int prob_m, int prob_n,
                          int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int has_zp,
                          int is_zp_float, bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  int sh_block_meta_size = tb_m * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size +
                   sh_g_idx_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, bool m_block_size_8,
                     int thread_m_blocks, int prob_m, int prob_n, int prob_k,
                     int num_bits, int group_size, bool has_act_order,
                     bool is_k_full, int has_zp, int is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  if (th_config.num_threads < 128) {
    return false;
  }

  int cache_size =
      get_kernel_cache_size(th_config, m_block_size_8, thread_m_blocks, prob_m,
                            prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

#include "kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int num_experts, int top_k, int thread_m_blocks,
    bool m_block_size_8, int num_bits, int group_size, bool has_act_order,
    bool is_k_full, bool has_zp, bool is_zp_float, bool is_a_8bit, int stages,
    int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, m_block_size_8, thread_m_blocks, prob_m,
                         prob_n, prob_k, num_bits, group_size, has_act_order,
                         is_k_full, has_zp, is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
        num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float,
        is_a_8bit, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : (group_size / 16);
    }

    auto kernel =
        get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                          th_config.thread_n / 16, th_config.thread_k / 16,
                          m_block_size_8, has_act_order, has_zp, group_blocks,
                          th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
    int allow_count = min(device_max_reg_size / reg_size,
                          max_shared_mem / (cache_size + 1536));
    if (thread_m_blocks == 1)
      allow_count = max(min(allow_count, 4), 1);
    else
      allow_count = max(min(allow_count, 2), 1);

    if (prob_n / th_config.thread_n * prob_m * top_k * 4 < sms * allow_count) {
      allow_count =
          max(prob_n / th_config.thread_n * prob_m * top_k * 4 / sms, 1);
    }

    if (allow_count > count) {
      count = allow_count;
      exec_cfg = {count, th_config};
    };
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, void* sorted_token_ids,
               void* expert_ids, void* num_tokens_past_padded,
               void* topk_weights, int moe_block_size, int num_experts,
               int top_k, bool mul_topk_weights, int prob_m, int prob_n,
               int prob_k, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;
  bool is_a_8bit = a_type.size_bits() == 8;

  MARLIN_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0);

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      MARLIN_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      MARLIN_CHECK(prob_k % group_blocks == 0);
    } else {
      MARLIN_CHECK(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      MARLIN_CHECK(prob_k % group_blocks == 0);
    }
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const uint16_t* g_s_ptr = (const uint16_t*)g_s;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr =
      (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    auto kernel = permute_cols_kernel<8>;
    if (moe_block_size == 8) {
    } else if (moe_block_size == 16)
      kernel = permute_cols_kernel<16>;
    else if (moe_block_size == 32)
      kernel = permute_cols_kernel<32>;
    else if (moe_block_size == 48)
      kernel = permute_cols_kernel<48>;
    else if (moe_block_size == 64)
      kernel = permute_cols_kernel<64>;
    else
      MARLIN_CHECK(false);

    // clang-format off
    kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
    // clang-format on
    A_ptr = a_tmp_ptr;
    prob_m = prob_m * top_k;
    top_k = 1;

    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  MARLIN_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  MARLIN_CHECK(major_capability * 10 + minor_capability >= 75);
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
  }

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, thread_k * thread_n / 64};
    if (blocks_per_sm == -1) blocks_per_sm = 1;
    exec_cfg = exec_config_t{blocks_per_sm, thread_tfg};
    MARLIN_CHECK(prob_n % thread_n == 0);
    MARLIN_CHECK(prob_k % thread_k == 0);
  } else {
    // Auto config
    exec_cfg = determine_exec_config(
        a_type, b_type, c_type, s_type, prob_m, prob_n, prob_k, num_experts,
        top_k, thread_m_blocks, m_block_size_8, num_bits, group_size,
        has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages,
        max_shared_mem, sms);
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1)
    max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  MARLIN_CHECK(is_valid_config(thread_tfg, m_block_size_8, thread_m_blocks,
                              prob_m, prob_n, prob_k, num_bits, group_size,
                              has_act_order, is_k_full, has_zp, is_zp_float,
                              is_a_8bit, stages, max_shared_mem));

  int sh_cache_size =
      get_kernel_cache_size(thread_tfg, m_block_size_8, thread_m_blocks, prob_m,
                            prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, has_zp, is_zp_float, is_a_8bit, stages);

  auto kernel = get_marlin_kernel(
      a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
      thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
      num_threads, is_zp_float, stages);

  MARLIN_CHECK(kernel != MarlinDefault);

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
      prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce);
  // clang-format on
}

}  // namespace marlin_moe_wna16

// ---------------------------------------------------------------------------
// extern "C" entry points for Rust FFI
// ---------------------------------------------------------------------------

extern "C" {

void marlin_moe_gemm_bf16(
    const void* a,              // [size_m, size_k] BF16 activations
    void* c,                    // [size_m * top_k, size_n] BF16 output
    void* c_tmp,                // FP32 reduction temp buffer
    const void* b_q_weight,     // [E, K/tile, N*tile] Marlin-packed u32
    const void* b_scales,       // [E, num_groups, N] BF16
    const void* b_zeros,        // [E, num_groups, N/8] u32 (AWQ, or nullptr)
    const void* g_idx,          // [E, K] or nullptr
    const void* perm,           // [E, K] or nullptr
    void* a_tmp,                // [size_m * top_k, size_k] temp for act_order
    void* workspace,            // [sms * 4] i32 locks
    const void* sorted_token_ids,     // from moe_align_block_size
    const void* expert_ids,           // from moe_align_block_size
    const void* num_tokens_past_padded,
    const void* topk_weights,         // [size_m, top_k] f32
    int moe_block_size,
    int num_experts,
    int top_k,
    bool mul_topk_weights,
    int size_m, int size_n, int size_k,
    int num_groups, int group_size,
    bool has_act_order, bool is_k_full,
    bool has_zp, bool is_zp_float,
    bool use_fp32_reduce,
    int b_type_id,              // 1 = kU4 (AWQ), 0 = kU4B8 (GPTQ)
    cudaStream_t stream, int device_id) {

  vllm::ScalarType a_type = vllm::kBFloat16;
  vllm::ScalarType c_type = vllm::kBFloat16;
  vllm::ScalarType s_type = vllm::kBFloat16;
  vllm::ScalarType b_type = (b_type_id == 1) ? vllm::kU4 : vllm::kU4B8;

  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);

  // Zero workspace locks — always zero sms*4 entries (the max the kernel uses).
  // Python sizes this as min(max_n_tiles * sorted_token_ids.size(0)/block_size, sms*4)
  // but sms*4 is the upper bound and zeroing ~2KB is negligible.
  int min_workspace_size = sms * 4;
  cudaMemsetAsync(workspace, 0, min_workspace_size * sizeof(int), stream);

  marlin_moe_wna16::marlin_mm(
      a, b_q_weight, c, c_tmp,
      /*b_bias=*/nullptr, /*a_s=*/nullptr,
      const_cast<void*>(b_scales), /*g_s=*/nullptr,
      const_cast<void*>(b_zeros), const_cast<void*>(g_idx),
      const_cast<void*>(perm), a_tmp,
      const_cast<void*>(sorted_token_ids),
      const_cast<void*>(expert_ids),
      const_cast<void*>(num_tokens_past_padded),
      const_cast<void*>(topk_weights),
      moe_block_size, num_experts, top_k, mul_topk_weights,
      size_m, size_n, size_k, workspace,
      a_type, b_type, c_type, s_type,
      /*has_bias=*/false, has_act_order, is_k_full, has_zp,
      num_groups, group_size, device_id, stream,
      /*thread_k=*/-1, /*thread_n=*/-1, sms, /*blocks_per_sm=*/-1,
      /*use_atomic_add=*/false, use_fp32_reduce, is_zp_float);
}

void marlin_moe_gemm_f16(
    const void* a,
    void* c,
    void* c_tmp,
    const void* b_q_weight,
    const void* b_scales,
    const void* b_zeros,
    const void* g_idx,
    const void* perm,
    void* a_tmp,
    void* workspace,
    const void* sorted_token_ids,
    const void* expert_ids,
    const void* num_tokens_past_padded,
    const void* topk_weights,
    int moe_block_size,
    int num_experts,
    int top_k,
    bool mul_topk_weights,
    int size_m, int size_n, int size_k,
    int num_groups, int group_size,
    bool has_act_order, bool is_k_full,
    bool has_zp, bool is_zp_float,
    bool use_fp32_reduce,
    int b_type_id,
    cudaStream_t stream, int device_id) {

  vllm::ScalarType a_type = vllm::kFloat16;
  vllm::ScalarType c_type = vllm::kFloat16;
  vllm::ScalarType s_type = vllm::kFloat16;
  vllm::ScalarType b_type = (b_type_id == 1) ? vllm::kU4 : vllm::kU4B8;

  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);

  int min_workspace_size = sms * 4;
  cudaMemsetAsync(workspace, 0, min_workspace_size * sizeof(int), stream);

  marlin_moe_wna16::marlin_mm(
      a, b_q_weight, c, c_tmp,
      /*b_bias=*/nullptr, /*a_s=*/nullptr,
      const_cast<void*>(b_scales), /*g_s=*/nullptr,
      const_cast<void*>(b_zeros), const_cast<void*>(g_idx),
      const_cast<void*>(perm), a_tmp,
      const_cast<void*>(sorted_token_ids),
      const_cast<void*>(expert_ids),
      const_cast<void*>(num_tokens_past_padded),
      const_cast<void*>(topk_weights),
      moe_block_size, num_experts, top_k, mul_topk_weights,
      size_m, size_n, size_k, workspace,
      a_type, b_type, c_type, s_type,
      /*has_bias=*/false, has_act_order, is_k_full, has_zp,
      num_groups, group_size, device_id, stream,
      /*thread_k=*/-1, /*thread_n=*/-1, sms, /*blocks_per_sm=*/-1,
      /*use_atomic_add=*/false, use_fp32_reduce, is_zp_float);
}

}  // extern "C"
