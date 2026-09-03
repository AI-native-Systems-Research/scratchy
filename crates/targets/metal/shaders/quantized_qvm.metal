// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX `affine_qvm` (`mlx/backend/metal/kernels/
// quantized.h:1600`) + `affine_qvm_split_k` (`:1652`) vector ×
// matrix kernels for transpose=false matvec (`x` is `[M, K]`,
// `W` is `[K, N]` int4-packed, output is `[M, N]`). The body of
// `qvm_impl` (`:978`) is reproduced verbatim; the splitk wrapper
// computes per-partition pointer offsets inline from function
// constants instead of going through `adjust_matrix_offsets`
// (`:1351`), because scratchy specializes pipelines per-shape and
// every stride is a compile-time constant for B=1.
//
// Per `INT4_PARITY_PLAN.md` §P5, instantiations cover:
//   bits = 4
//   group_size in {32, 64, 128}
//   dtype in {f16, bf16}
//   batched = 0 only (batched=1 needs adjust_matrix_offsets,
//     deferred to P13 with the MoE gather variants)
//
// Symbol naming follows MLX's `concatenate` pattern
// (`quantized.cpp:446-454` for qvm, `:366-379` for qvm_split_k):
//   affine_qvm_<dtype>_gs_<gs>_b_<bits>_batch_<batched>
//   affine_qvm_split_k_<dtype>_gs_<gs>_b_<bits>_spk_<split_k>
//
// W layout note (different from qmm_t):
//   qmm_t storage: [N, K/pack_factor] u32, scales/biases [N, K/gs]
//   qvm/qmm_n storage: [K, N/pack_factor] u32, scales/biases [K, N/gs]
// (Last-axis quantize follows W's last axis; transpose=true has
//  W shaped [N, K] so last=K; transpose=false has W shaped [K, N]
//  so last=N. Both layouts are pre-determined by the checkpoint;
//  scratchy respects whichever the macro emits — for qmm_n / qvm
//  paths we read [K, N] layout.)

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

MLX_MTL_CONST int SIMD_SIZE = 32;

// ─────────────────────────────────────────────────────────────────
// Function constants — baked at pipeline build time by
// `MetalAffineQvm::execute`. K and N ride as constants so the
// kernel is ICB-recordable (no setBytes for shape metadata).
// For the splitk variant, QVM_K_PARTITION_SIZE and
// QVM_FINAL_BLOCK_SIZE hold the per-partition K-extent (= split_D
// from `quantized.cpp:315`) and the last partition's smaller
// extent (= final_block_size from `:352`). M rides as a constant
// for the splitk output-offset arithmetic (always 1 for decode
// today, but qvm_split_k can fire for M ∈ {1,2,3} per the
// vector_limit=4 rule at `quantized.cpp:1409`).
//
// Indices match `ConstantValue::int(N, value)` in the dispatcher;
// keep them stable, ICB-recorded commands key on the constant bag.
// ─────────────────────────────────────────────────────────────────

constant int QVM_K                  [[function_constant(0)]];
constant int QVM_N                  [[function_constant(1)]];
constant int QVM_M                  [[function_constant(2)]];
constant int QVM_K_PARTITION_SIZE   [[function_constant(3)]];
constant int QVM_FINAL_BLOCK_SIZE   [[function_constant(4)]];
constant int QVM_SPLIT_K            [[function_constant(5)]];

// ─────────────────────────────────────────────────────────────────
// Pack helpers — quantized.h:17-26 (same constants as
// quantized_qmv.metal; duplicated here so each .metal compiles
// stand-alone — `build.rs` produces one .metallib per file).
// ─────────────────────────────────────────────────────────────────

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// ─────────────────────────────────────────────────────────────────
// ConditionalType — quantized.h:475-481 (used by qvm_impl below to
// switch between u32 and u8 W pointer types depending on whether
// `bits` is a power of two).
// ─────────────────────────────────────────────────────────────────

template <bool C, typename A, typename B>
struct ConditionalType {
  using type = A;
};

template <typename A, typename B>
struct ConditionalType<false, A, B> {
  using type = B;
};

// ─────────────────────────────────────────────────────────────────
// qouter — quantized.h:394-481. Vector-outer kernel of the K-block
// loop: accumulates `x * dequant(w) + bias` into the per-N-col
// result array. We keep the full bits switch (2/3/4/5/6/8) so the
// body is byte-identical to MLX; only bits=4 is instantiated for
// P5 and the other branches dead-code under the constexpr.
// ─────────────────────────────────────────────────────────────────

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// qvm_impl_inline — quantized.h:978-1084. Reproduces MLX's
// qvm_impl byte-for-byte. The dispatcher gives each threadgroup a
// 64-N-col × 1-M-row slice (bn = min(group_size, 32) * 2 = 64 for
// gs ∈ {32, 64, 128}), with 2 simdgroups of 32 lanes each. Each
// simdgroup handles tn*pack_factor = 32 N-cols; the 32 lanes
// distribute the K reduction in stride of SIMD_SIZE.
//
// Output: each thread accumulates a `result[tn * pack_factor]`
// of partial sums; `simd_sum` reduces across the K-lane direction;
// simd_lid==0 writes the final value to y.
//
// in_vec_size_arg is the K-extent the kernel should reduce over.
// For the standard qvm kernel this equals QVM_K (full K); for the
// splitk kernel the wrapper picks `QVM_K_PARTITION_SIZE` or
// `QVM_FINAL_BLOCK_SIZE` based on the partition index.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void qvm_impl_inline(
    const device uint32_t*  w,
    const device T_scale*   scales,
    const device T_scale*   biases,
    const device T_act*     x,
    device T_act*           y,
    const int               in_vec_size_arg,
    const int               out_vec_size,
    uint3 tid,
    uint  simd_gid,
    uint  simd_lid)
{
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = SIMD_SIZE;

  using W_T =
      typename ConditionalType<power_of_2_bits, uint32_t, uint8_t>::type;
  const device W_T* ws = (const device W_T*)w;

  typedef float U;
  typedef struct {
    W_T wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};
  thread U scale = 1;
  thread U bias = 0;
  thread U x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);
  ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;
  scales += out_col / group_size + simd_lid * out_vec_size_g;
  biases += out_col / group_size + simd_lid * out_vec_size_g;
  x += tid.x * in_vec_size_arg + simd_lid;
  y += tid.x * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of block_size
  int remaining = in_vec_size_arg % block_size;
  if (remaining == 0) {
    for (int i = 0; i < in_vec_size_arg; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
  } else {
    for (int i = block_size; i < in_vec_size_arg; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);

      qouter<U, tn * pack_factor, bits>(
          (thread uint8_t*)&w_local, x_local, scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
    if (static_cast<int>(simd_lid) < remaining) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w*)ws);
    } else {
      x_local = 0;
      scale = 0;
      bias = 0;
    }
    qouter<U, tn * pack_factor, bits>(
        (thread uint8_t*)&w_local, x_local, scale, bias, result);
  }

  // Accumulate in the simdgroup
  MLX_MTL_PRAGMA_UNROLL
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T_act>(result[k]);
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// affine_qvm kernel wrapper — quantized.h:1600-1649
//
// batched=0 only; batched=1 needs `adjust_matrix_offsets`
// (`quantized.h:1351`), deferred to P13 alongside MoE gather.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void affine_qvm_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // K (in_vec_size) and N (out_vec_size) ride as file-scope
    // function constants QVM_K / QVM_N. MLX passes them as
    // buffer(5)/buffer(6) setBytes; scratchy specializes per-shape.
    uint3 tid           [[threadgroup_position_in_grid]],
    uint  simd_gid      [[simdgroup_index_in_threadgroup]],
    uint  simd_lid      [[thread_index_in_simdgroup]])
{
  qvm_impl_inline<T_act, T_scale, group_size, bits>(
      w, scales, biases, x, y,
      /*in_vec_size_arg=*/QVM_K,
      /*out_vec_size=*/QVM_N,
      tid, simd_gid, simd_lid);
}

// ─────────────────────────────────────────────────────────────────
// affine_qvm_split_k kernel wrapper — quantized.h:1652-1705
//
// MLX uses `adjust_matrix_offsets` to walk the reshape
// `[B, split_k, M, split_D]` stride bag (quantized.cpp:340-358).
// scratchy is B=1 only (batched=0), so the per-partition pointer
// offsets are deterministic from (QVM_K_PARTITION_SIZE = split_D,
// QVM_N, group_size, QVM_M):
//
//   tid.z indexes partition in [0, QVM_SPLIT_K) (B=1).
//   x:        + tid.z * QVM_K_PARTITION_SIZE                halves
//   w (u32):  + tid.z * QVM_K_PARTITION_SIZE * QVM_N/pack_factor
//                                                           u32s
//   scales:   + tid.z * QVM_K_PARTITION_SIZE * QVM_N/group_size
//                                                           halves
//   biases:   + same
//   y:        + tid.z * QVM_M * QVM_N halves (output [split_k, M, N]
//             flat in dtype, sum-reduced by downstream
//             splitk_reduce_sum)
//
// in_vec_size_adj = (tid.z == QVM_SPLIT_K - 1)
//                     ? QVM_FINAL_BLOCK_SIZE : QVM_K_PARTITION_SIZE
// — matches `quantized.h:1691-1692`.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void affine_qvm_split_k_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    uint3 tid           [[threadgroup_position_in_grid]],
    uint  simd_gid      [[simdgroup_index_in_threadgroup]],
    uint  simd_lid      [[thread_index_in_simdgroup]])
{
  // W is stored as `[K, N/pack_factor_u32]` u32 for transpose=false
  // (last axis quantized). pack_factor_u32 = 32 / bits = 8 for
  // bits=4, so each u32 holds 8 dequant'd halves. The number of u32s
  // per K-row is `N / pack_factor_u32 = N / 8` for bits=4.
  //
  // Note the dual pack_factor convention in MLX: `get_pack_factor<
  // bits, 32>()` returns the per-u32 pack factor (used for u32
  // pointer arithmetic — that's `out_vec_size_w` inside qvm_impl),
  // while `get_pack_factor<bits, 8>()` returns the per-byte pack
  // factor (used when reading raw bytes). For the split_k
  // wrapper's u32* pointer-shift, we want the per-u32 form.
  constexpr int pack_factor_u32 = get_pack_factor<bits, 32>();
  const int n_packed = QVM_N / pack_factor_u32;  // = N/8 for bits=4

  const int part_idx     = int(tid.z);
  const int k_part_off   = part_idx * QVM_K_PARTITION_SIZE;
  const device uint32_t* w_part      = w      + int64_t(k_part_off) * n_packed;
  const device T_scale*  scales_part = scales + int64_t(k_part_off) * (QVM_N / group_size);
  const device T_scale*  biases_part = biases + int64_t(k_part_off) * (QVM_N / group_size);
  const device T_act*    x_part      = x      + int64_t(k_part_off);
  device T_act*          y_part      = y      + int64_t(part_idx) * QVM_M * QVM_N;

  const int in_vec_size_adj = (part_idx == QVM_SPLIT_K - 1)
      ? QVM_FINAL_BLOCK_SIZE
      : QVM_K_PARTITION_SIZE;

  // Force tid.z = 0 inside qvm_impl so its x/y pointer math doesn't
  // re-apply the partition offset (we already shifted the base
  // pointers above; adjust_matrix_offsets+qvm_impl in MLX work
  // together via the reshape so a single advance covers both, but
  // here we collapse to a single pre-shift since strides are known).
  uint3 inner_tid = uint3(tid.x, tid.y, 0u);
  qvm_impl_inline<T_act, T_scale, group_size, bits>(
      w_part, scales_part, biases_part, x_part, y_part,
      in_vec_size_adj, QVM_N,
      inner_tid, simd_gid, simd_lid);
}

// ─────────────────────────────────────────────────────────────────
// Instantiations — one symbol per (dtype, group_size) for qvm
// (batched=0) and qvm_split_k. split_k value is a function
// constant (QVM_SPLIT_K), so the same symbol serves split_k ∈
// {8, 32} via specialized pipelines — different from MLX's
// template-arg approach but functionally identical.
// ─────────────────────────────────────────────────────────────────

#define INST_QVM(act_tag, act_type, scale_tag, scale_type, gs)                 \
  template [[host_name(                                                        \
      "affine_qvm_" #act_tag "_s_" #scale_tag "_gs_" #gs                       \
      "_b_4_batch_0")]] [[kernel]] void                                        \
  affine_qvm_kernel<act_type, scale_type, gs, 4>(                              \
      const device uint32_t*   w        [[buffer(0)]],                         \
      const device scale_type* scales   [[buffer(1)]],                         \
      const device scale_type* biases   [[buffer(2)]],                         \
      const device act_type*   x        [[buffer(3)]],                         \
      device act_type*         y        [[buffer(4)]],                         \
      uint3 tid           [[threadgroup_position_in_grid]],                    \
      uint  simd_gid      [[simdgroup_index_in_threadgroup]],                  \
      uint  simd_lid      [[thread_index_in_simdgroup]]);

#define INST_QVM_SPLIT_K(act_tag, act_type, scale_tag, scale_type, gs)         \
  template [[host_name(                                                        \
      "affine_qvm_split_k_" #act_tag "_s_" #scale_tag "_gs_" #gs               \
      "_b_4")]] [[kernel]] void                                                \
  affine_qvm_split_k_kernel<act_type, scale_type, gs, 4>(                      \
      const device uint32_t*   w        [[buffer(0)]],                         \
      const device scale_type* scales   [[buffer(1)]],                         \
      const device scale_type* biases   [[buffer(2)]],                         \
      const device act_type*   x        [[buffer(3)]],                         \
      device act_type*         y        [[buffer(4)]],                         \
      uint3 tid           [[threadgroup_position_in_grid]],                    \
      uint  simd_gid      [[simdgroup_index_in_threadgroup]],                  \
      uint  simd_lid      [[thread_index_in_simdgroup]]);

#define INST_QVM_ALL(act_tag, act_type, scale_tag, scale_type, gs)  \
  INST_QVM(act_tag, act_type, scale_tag, scale_type, gs)            \
  INST_QVM_SPLIT_K(act_tag, act_type, scale_tag, scale_type, gs)

// Coverage: see header note in `quantized_qmv.metal`.
INST_QVM_ALL(f16,  half,   f16, half,    32)
INST_QVM_ALL(f16,  half,   f16, half,    64)
INST_QVM_ALL(f16,  half,   f16, half,   128)
INST_QVM_ALL(bf16, bfloat, f16, half,    32)
INST_QVM_ALL(bf16, bfloat, f16, half,    64)
INST_QVM_ALL(bf16, bfloat, f16, half,   128)
// bf16-scale variants — Qwen3-MoE / `torch_dtype: bfloat16` ships BF16
// scales/biases. Mirrors MLX's `T_scale = bfloat16_t` instantiations.
INST_QVM_ALL(bf16, bfloat, bf16, bfloat, 32)
INST_QVM_ALL(bf16, bfloat, bf16, bfloat, 64)
INST_QVM_ALL(bf16, bfloat, bf16, bfloat, 128)
INST_QVM_ALL(f16,  half,   bf16, bfloat, 32)
INST_QVM_ALL(f16,  half,   bf16, bfloat, 64)
INST_QVM_ALL(f16,  half,   bf16, bfloat, 128)
