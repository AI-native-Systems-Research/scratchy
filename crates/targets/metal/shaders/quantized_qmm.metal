// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX `affine_qmm_t` (`mlx/backend/metal/kernels/
// quantized.h:1707`) + `affine_qmm_t_splitk` (`:1780`) + `affine_qmm_n`
// (`:1847`) prefill-matmul kernels. The bodies of `qmm_t_impl`
// (`:1094`) and `qmm_n_impl` (`:1221`) are reproduced via an inlined
// steel-style BlockMMA / BlockLoader / QuantizedBlockLoader trio.
// We keep the kernel surface identical to MLX (same tile sizes
// BM=BN=BK=32, same warp shape WM=WN=2, same per-thread load layout,
// same `aligned_N` template flag for qmm_t, same K loop) so that the
// four `if (m_full)` × `if (aligned_N && n_full)` branches at
// `quantized.h:1158-1202` map line-for-line into the four load paths
// below.
//
// qmm_n (transpose=false) differs from qmm_t in three places:
//   1. W storage is `[K, N]` u32-packed (last axis = N), so the K loop
//      reads BK rows × BCOLS_PACKED packed bytes per BK iter.
//   2. Scales / biases layout is `[K, N/group_size]` — group runs
//      along N, not K — so the per-BK step advances them by
//      `BK * N / group_size` (`group_stride` in MLX's
//      `QuantizedBlockLoader::next()` reduction_dim=0 branch).
//   3. BlockMMA's B-frag is loaded without transpose because Ws is
//      already laid out as `[BK, BN_padded]` (the natural shape MMA
//      expects when `transpose_b == false` per
//      `quantized.h:1124`-`:1252`).
// MLX qmm_n has no `aligned_N` template flag — the dispatcher assumes
// `N % 32 == 0` (the matmul-branch grid at `quantized.cpp:721` ceils
// N, but the store path is unconditional `store_result` for
// `num_els == BM` and silently writes past N when N is unaligned).
// All linear shapes in our model coverage matrix have N % 32 == 0
// (typically a multiple of 64), so we match MLX 1:1 and don't expose
// an aligned_N flag for qmm_n.
//
// Why we inline rather than vendor `mlx/backend/metal/kernels/steel/
// gemm/{mma,loader}.h`: the prior 2026-05-09 session
// (`project_metal_gemm_port_dead_end`) vendored the full steel tree
// and hit a deterministic zero-output wall on scattered threadgroups
// at grid X >= 1024 (N >= ~32K). Llama-1B/3B q4 prefill shapes have
// N ≤ 8192 → grid X ≤ 256, well inside the regime where the existing
// `fused_gate_up_silu_mul_gemm_steel_*_specialized` kernels (same
// BM=BN=32, WM=WN=2 pattern) work; the lm_head N=128256 shape only
// fires at M=1 and routes through `qmv*` (P3), never through `qmm_t`.
// The dispatcher in `MetalAffineQmmT` enforces this routing per the
// `quantized.cpp:1411` `M >= vector_limit` rule.
//
// Per `INT4_PARITY_PLAN.md` §P4, instantiations cover:
//   bits = 4
//   group_size in {32, 64, 128}
//   dtype in {f16, bf16}
//   aligned_N in {true, false}
//   batched = 0 only (qmm_t batched=1 needs `adjust_matrix_offsets`
//     from `quantized.h:1351`, deferred to P13 with the MoE gather
//     variants; dispatcher rejects B>1 here.)
//
// Symbol naming follows the MLX dispatcher's `concatenate` at
// `quantized.cpp:728`:
//   affine_qmm_t_<dtype>_gs_<gs>_b_<bits>_alN_<true|false>_batch_0
//   affine_qmm_t_splitk_<dtype>_gs_<gs>_b_<bits>_alN_<true|false>
//
// QuantizedBlockLoader port:
//   `quantized.h:572-689`. We use the same per-thread layout
//   (n_reads=4 packed bytes per K-iter for bits=4, gs=32+), inline
//   the `dequantize<T, pack_factor, bits>` body (`quantized.h:483`),
//   and step scale/bias every `group_size / BK` BK iterations to
//   match `QuantizedBlockLoader::next()` (`:671`).

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

// MLX steel/gemm vendor — pulls in `BlockMMA<T, U, BM, BN, BK, WM, WN, ...>`.
// Used by qmm_t_impl_inline below to drive the inner MMA loop with the
// SAME structure as MLX's prebuilt `affine_qmm_t_*` binary. The
// per-fragment dtype + cast pattern this gives us is what compiles to
// `multiply_accumulate.v64f32.v64f32.v64f32.v64f32` on Apple7 — the
// hardware-fast MMA op that MLX's binary uses.
#include "mlx_steel_gemm/mma.h"
#include "mlx_steel_gemm/loader.h"

// MLX quantized vendor — `QuantizedBlockLoader<T, BROWS, BCOLS, dst_ld,
// reduction_dim, tgp_size, group_size, bits>` plus its `dequantize<U,N,
// bits>` helper from `quantized.h:483-689`. Drives the per-K-iter
// weight tile load + dequant with the SAME unrolled structure as
// MLX's prebuilt qmm_t binary.
#include "mlx_quantized/quantized_loader.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

MLX_MTL_CONST int SIMD_SIZE = 32;

// ─────────────────────────────────────────────────────────────────
// Function constants — baked at pipeline build time by
// `MetalAffineQmmT::execute` (and the lower_one path once
// `Instruction::AffineQmm` lands). MLX passes K / N / M (and the
// splitk wrapper passes k_partition_size + split_k_partition_stride)
// as setBytes runtime args; scratchy specializes per-shape so each
// (K, N, M[, k_partition_size]) tuple gets its own pipeline. This is
// the same trade `fused_gate_up_silu_mul` already takes — and is
// required for ICB recording, which exposes setKernelBuffer but not
// setKernelBytes.
//
// `split_k_partition_stride` (M * N) is computed inline from QMM_M /
// QMM_N rather than baked separately, so the constant slots stay
// shared across the standard and splitk variants.
//
// Indices match `ConstantValue::uint(N, value)` in the dispatcher;
// keep them stable.
// ─────────────────────────────────────────────────────────────────

constant int QMM_K                [[function_constant(0)]];
constant int QMM_N                [[function_constant(1)]];
constant int QMM_M                [[function_constant(2)]];
constant int QMM_K_PARTITION_SIZE [[function_constant(3)]];
// MoE grouped GEMM only: the number of valid experts. Padded output
// tiles whose expert index == QMM_NUM_EXPERTS are sentinels (trailing
// unused rows) and skip — lets the host pad to a static row bound.
constant int QMM_NUM_EXPERTS      [[function_constant(4)]];

// ─────────────────────────────────────────────────────────────────
// Pack helpers — quantized.h:17-26 (same constants as
// quantized_qmv.metal; duplicated here so each .metal compiles
// stand-alone — `build.rs` produces one .metallib per file).
// ─────────────────────────────────────────────────────────────────

// `get_pack_factor` / `get_bytes_per_pack` come from
// `mlx_quantized/quantized_loader.h` (vendored from
// `quantized.h:18-26`); duplicating them here would conflict with the
// vendored definitions on the same translation unit.

// ─────────────────────────────────────────────────────────────────
// NVFP4 E2M1 decode (sign-magnitude 4-bit; bit 3 = sign, bits 0-2 =
// magnitude index). Same byte layout as affine int4, so the prefill
// tile machinery is shared; only the per-code dequant differs (LUT,
// no per-group bias). Duplicated from quantized_qmv.metal — each
// .metal compiles to its own metallib. Matches Python vLLM's
// `kE2M1ToFloat` (`nvfp4_emulation_utils.py`).
// ─────────────────────────────────────────────────────────────────
// E2M1 decode computed arithmetically — NO lookup table. A file-scope
// `constant float[]` LUT compiles to a static initializer / global
// constructor (`air.static_init`), which the MTL4 pipeline-compiler
// path does NOT run → uninitialized table → garbage under MTL4 (MTL3 /
// offline runs the ctor, so it only failed in production). E2M1 =
// 1 sign · 2 exp · 1 mantissa: mag = (e==0) ? m·0.5 : (1+m·0.5)·2^(e-1)
// → {0,.5,1,1.5,2,3,4,6} for codes 0..7, bit-identical to the old LUT.
inline float nvfp4_decode(uint code) {
  uint e = (code >> 1u) & 0x3u;
  uint m = code & 0x1u;
  float mant = (e == 0u) ? (float(m) * 0.5f) : (1.0f + float(m) * 0.5f);
  float scale = (e == 0u) ? 1.0f : float(1u << (e - 1u));
  float mag = mant * scale;
  return (code & 0x8u) ? -mag : mag;
}

// ─────────────────────────────────────────────────────────────────
// qmm_t_impl — quantized.h:1094-1212. Implemented for bits=4 by
// inlining the 4-bit branch of `dequantize` (`:521-527`) directly
// into the W-loader. Other bits land alongside their first model.
//
// Template params match MLX 1:1 (sans the BlockMMA/BlockLoader hooks
// which are spelled out inline):
//   T          activation/output element type (half or bfloat)
//   group_size affine-quant group size (32, 64, or 128 here)
//   bits       = 4 (P4)
//   aligned_N  N % BN == 0 → skip N-tail handling
//
// Hard-coded tile shape (faithful to MLX `qmm_t_impl` defaults at
// `quantized.h:1091-1093`):
//   BM = BN = BK = 32, WM = WN = 2, TM = TN = 2 (4 8×8 frags/SG)
//   BK_padded = BK + 16 / sizeof(T)   (= 40 for half, 40 for bfloat)
//   TGP = WM * WN * SIMD_SIZE = 128 threads / threadgroup
// ─────────────────────────────────────────────────────────────────

// T_compute is the dtype the threadgroup tiles + simdgroup MMA run on.
// Defaults to T_act (current behavior). On Apple7 (M1 family) bf16
// simdgroup_multiply_accumulate is a slow-path emulation (~1.7× slower
// than the f16 path); a `T_act = bfloat, T_compute = half` instantiation
// reads bf16 from device memory, casts to half when populating Xs/Ws,
// runs MMA in half, casts back to bf16 on output store. Output dtype on
// disk stays T_act so the residual stream's bf16 dynamic range is
// preserved (full-f16 streams break Llama-3.x exponent range — see
// `instr.rs:199-202`).
template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N, bool nvfp4 = false>
METAL_FUNC void qmm_t_impl_inline(
    const device uint32_t*  w,
    const device T_scale*   scales,
    const device T_scale*   biases,
    const device T_act*     x,
    device T_act*           y,
    threadgroup T_compute*  Xs,
    threadgroup T_compute*  Ws,
    threadgroup float*      out_scratch,
    const int              K,
    const int              N,
    const int              M,
    const int              K_eff,
    uint  simd_group_id,
    uint  simd_lane_id,
    uint3 tgid)
{
  // bits=8 rides the same MLX QuantizedBlockLoader path (taken when
  // T_compute == T_scale); the hand-inlined mixed-compute W loader
  // below is nibble(4-bit)-specialized, so b8 is only instantiated
  // for the same-compute symbol family (Gemma4 8-bit MLP, bf16/bf16).
  static_assert(bits == 4 || bits == 8,
                "qmm_t_impl_inline only instantiated for bits in {4, 8}");
  static_assert(group_size == 16 || group_size == 32 || group_size == 64 ||
                    group_size == 128,
                "qmm_t_impl_inline expects group_size in {16, 32, 64, 128}");
  static_assert(!nvfp4 || group_size == 16, "NVFP4 group_size is 16");

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int TM = BM / (8 * WM);            // 2
  constexpr int TN = BN / (8 * WN);            // 2
  constexpr int KFR = BK / 8;                  // 4 K-frags per BK iter
  // Threadgroup tiles live in T_compute, so BK_padded keys off
  // sizeof(T_compute) (= 40 for both fp16 and bf16, since both are 2 B).
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  constexpr int TGP = WM * WN * SIMD_SIZE;     // 128

  constexpr int pack_factor    = get_pack_factor<bits, 8>();    // 2
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();    // 1
  constexpr int BCOLS_PACKED   = BK / pack_factor;              // 16
  // Per-thread W reads: n_reads packed bytes → n_reads*pack_factor halves.
  constexpr int N_READS = (BCOLS_PACKED * BN) / TGP;            // 4
  constexpr int group_steps = group_size / BK;                  // 1, 2, or 4

  // Per-thread X read: 8 halves per thread (covers BM*BK = 1024 / 128 = 8
  // halves/thread). Each thread loads 2 vec4's of T (16 bytes per vec
  // = 8 halves) — the same vec4 pattern as the steel BlockLoader (one
  // unrolled iter per thread for BROWS==TROWS==32).
  constexpr int X_N_READS = (BM * BK) / TGP;                    // 8
  constexpr int X_TCOLS   = BK / X_N_READS;                     // 4
  static_assert(X_N_READS == 8, "expected 8 X halves per thread");

  const uint thread_idx = simd_group_id * SIMD_SIZE + simd_lane_id;

  // ── Output-tile origin and per-tile valid extents ──────────────
  const uint c_row = tgid.y * BM;  // along M
  const uint c_col = tgid.x * BN;  // along N
  if (c_row >= uint(M) || c_col >= uint(N)) return;
  const uint m_tile = (c_row + BM <= uint(M)) ? uint(BM) : uint(M) - c_row;
  const uint n_tile = aligned_N
      ? uint(BN)
      : ((c_col + BN <= uint(N)) ? uint(BN) : uint(N) - c_col);
  const bool m_full = (m_tile == BM);
  const bool n_full = aligned_N ? true : (n_tile == BN);

  // ── X loader thread coords (per BlockLoader; transpose dim=1) ──
  const uint bi_x = thread_idx / uint(X_TCOLS);          // [0, BM)
  const uint bj_x = uint(X_N_READS) * (thread_idx % uint(X_TCOLS));  // 0,8,16,24

  // ── W loader thread coords (per QuantizedBlockLoader) ──────────
  const uint bi_w = (uint(N_READS) * thread_idx) / uint(BCOLS_PACKED);     // [0, BN)
  const uint bj_w = (uint(N_READS) * thread_idx) % uint(BCOLS_PACKED);     // 0,4,8,12

  // ── Block base pointers (after y_row / y_col shift per MLX :1145) ──
  const int K_w = K * bytes_per_pack / pack_factor;       // packed bytes per row of W
  const int K_g = K / group_size;                         // groups per row of W
  const device uint8_t* wl_base = (const device uint8_t*)w;
  const device T_act* x_block    = x + int64_t(c_row) * K;
  const device uint8_t* w_block  = wl_base + c_col * K_w;
  const device T_scale* s_block  = scales + c_col * K_g;
  const device T_scale* b_block  = biases + c_col * K_g;
  device T_act* y_block          = y + int64_t(c_row) * N + c_col;

  // ── X loader: per-thread source/dest pointers. The X path stays
  // inline because `BlockLoader<T, ...>` requires `T_act == T_compute`
  // (single-T template); we need a bf16→half cast on the way into Xs.
  threadgroup T_compute* Xs_dst = Xs + bi_x * BK_padded + bj_x;
  const device T_act* X_src = x_block + bi_x * K + bj_x;

  // ── W loader. When `T_compute == T_scale` (M1 fast-path with
  //    T_compute = half = T_scale), delegate to MLX's
  //    `QuantizedBlockLoader<T, BN, BK, BK_padded, reduction_dim=1,
  //    TGP, group_size, 4>` (vendored verbatim from
  //    `quantized.h:572-689`). The loader reads packed-int4 weights,
  //    dequantises to T = T_compute inside `load_unsafe`/`load_safe`,
  //    advances scales/biases per `next()`. Same shape as MLX's
  //    prebuilt qmm_t binary.
  //
  //    On the legacy path (T_compute = bfloat ≠ T_scale = half — only
  //    instantiated when the M1 f16-compute fast-path is disabled),
  //    the inline dequant block inside the K-loop runs instead;
  //    `loader_w` is unused there. `if constexpr` (C++17, supported
  //    by metal-stdlib) keeps the unused branch from being
  //    instantiated.
  // NVFP4 forces the inline W-dequant path (its group_size=16 < BK=32
  // violates `QuantizedBlockLoader`'s `BCOLS <= group_size` assert), so
  // the loader is constructed but never used. Instantiate its type with
  // a valid placeholder group_size (BK=32) to keep the static_assert
  // happy; the construction stores pointers only and is harmless.
  constexpr int loader_gs = nvfp4 ? BK : group_size;
  using loader_w_t = QuantizedBlockLoader<
      /* T = */ T_scale,
      /* BROWS = */ BN,
      /* BCOLS = */ BK,
      /* dst_ld = */ BK_padded,
      /* reduction_dim = */ 1,
      /* tgp_size = */ TGP,
      /* group_size = */ loader_gs,
      /* bits = */ bits>;
  loader_w_t loader_w(
      (const device uint8_t*)w_block,
      s_block,
      b_block,
      /*src_ld=*/K,
      reinterpret_cast<threadgroup T_scale*>(Ws),
      simd_group_id,
      simd_lane_id);

  // ── BlockMMA — VERBATIM MLX `BlockMMA<T_compute, T_act, BM, BN, BK,
  //    WM, WN, transpose_a=false, transpose_b=true, lda_tgp, ldb_tgp,
  //    AccumType=float>`. Owns the simdgroup-matrix accumulators
  //    (Ctile, MMATile<float, TM, TN>), the As/Bs offsets per lane,
  //    and the K-loop MMA kernel (`mma_t::mma(Xs, Ws)`).
  //
  //    Template params:
  //      T (compute dtype, MMA fragments)        : T_compute
  //      U (output / store dtype)                : T_act
  //      BM, BN, BK                              : 32, 32, 32
  //      WM, WN                                  : 2, 2
  //      transpose_a / transpose_b               : false / true
  //      lda_tgp / ldb_tgp                       : BK_padded / BK_padded
  //      AccumType                               : float (default)
  //      Epilogue                                : TransformNone (default)
  using mma_t = mlx::steel::BlockMMA<
      /* T = */ T_compute,
      /* U = */ T_act,
      /* BM = */ BM,
      /* BN = */ BN,
      /* BK = */ BK,
      /* WM = */ WM,
      /* WN = */ WN,
      /* transpose_a = */ false,
      /* transpose_b = */ true,
      /* lda_tgp = */ BK_padded,
      /* ldb_tgp = */ BK_padded>;
  mma_t mma_op(simd_group_id, simd_lane_id);
  // Accumulator is BlockMMA's `Ctile` (zero-initialised by MMATile's
  // default constructor — `mma.h:222-225`); it lives across the K loop
  // in registers.

  // ── K loop: VERBATIM MLX `qmm_t_impl` (`quantized.h:1158-1202`).
  //   Four specialized loops based on the (m_full, n_full) cross
  //   product so each loop body runs UNCONDITIONAL load_safe /
  //   load_unsafe — no per-iter branches, no register-pressure cost
  //   for unused branches. Each call to `load_X` lambdas is the one
  //   inline cast we can't replace with `BlockLoader<T,...>` (need
  //   T_act → T_compute conversion); load_W goes through the
  //   vendored `QuantizedBlockLoader::load_unsafe` /
  //   `load_safe(short2(BK, n_tile))` when `T_compute == T_scale`
  //   (M1 fast-path), else falls back to the inline dequant. The
  //   `mma_op.mma(Xs, Ws)` and `loader_x.next()` / `loader_w.next()`
  //   calls match MLX line-for-line.
  //
  //   Lambdas keep the X-cast and the W-load body declared once and
  //   reused across the four loops. Apple's metal compiler inlines
  //   them at the call site so the unrolling structure is the same
  //   as if they were inlined manually.
  auto load_x_unsafe = [&]() {
    const device vec<T_act, 4>* X_src_v4 =
        (const device vec<T_act, 4>*)X_src;
    threadgroup vec<T_compute, 4>* Xs_dst_v4 =
        (threadgroup vec<T_compute, 4>*)Xs_dst;
    vec<T_act, 4> a0 = X_src_v4[0];
    vec<T_act, 4> a1 = X_src_v4[1];
    Xs_dst_v4[0] = vec<T_compute, 4>(
        T_compute(a0[0]), T_compute(a0[1]), T_compute(a0[2]), T_compute(a0[3]));
    Xs_dst_v4[1] = vec<T_compute, 4>(
        T_compute(a1[0]), T_compute(a1[1]), T_compute(a1[2]), T_compute(a1[3]));
  };
  auto load_x_safe_m = [&](short bm_lim) {
    if (bi_x < uint(bm_lim)) {
      load_x_unsafe();
    } else {
      ((threadgroup vec<T_compute, 4>*)Xs_dst)[0] = vec<T_compute, 4>(0);
      ((threadgroup vec<T_compute, 4>*)Xs_dst)[1] = vec<T_compute, 4>(0);
    }
  };
  // W: legacy inline dequant for the (T_compute != T_scale) case
  // (M1 fast-path uses QuantizedBlockLoader; this branch is
  // dead-code-eliminated when the constexpr check is true).
  auto load_w_inline = [&](int kk, bool n_unsafe) {
    threadgroup T_compute* Ws_dst_inline =
        Ws + bi_w * BK_padded + bj_w * pack_factor;
    const device uint8_t* W_src_inline =
        w_block + bi_w * K_w + bj_w * bytes_per_pack;
    const device T_scale* Sc_row_inline = s_block + bi_w * K_g;
    const device T_scale* Bs_row_inline = b_block + bi_w * K_g;
    W_src_inline += kk * BCOLS_PACKED * bytes_per_pack;
    // NVFP4 group_size (16) is smaller than BK (32), so a BK column-
    // block spans two scale groups; index the scale generally by the
    // thread's absolute K offset. The general formula reduces to the
    // affine `sb_step` when group_size >= BK (the `bj_w*pack_factor`
    // term is then < group_size, contributing 0 within a block).
    const int sb_step = nvfp4
        ? ((kk * BK + int(bj_w) * pack_factor) / group_size)
        : ((group_steps > 1) ? (kk / group_steps) : kk);
    Sc_row_inline += sb_step;
    Bs_row_inline += sb_step;
    bool in_bounds = n_unsafe ? true : (bi_w < n_tile);
    if (in_bounds) {
      T_compute scale = static_cast<T_compute>(*Sc_row_inline);
      T_compute bias  = static_cast<T_compute>(*Bs_row_inline);
      T_compute s0 = scale;
      T_compute s1 = scale / static_cast<T_compute>(16.0f);
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS; ++i) {
        uint8_t b = W_src_inline[i * bytes_per_pack];
        if (nvfp4) {
          // E2M1 LUT decode, no per-group bias. Low nibble = even K
          // element, high nibble = odd (same byte layout as affine).
          Ws_dst_inline[i * pack_factor + 0] =
              scale * static_cast<T_compute>(nvfp4_decode(b & 0x0fu));
          Ws_dst_inline[i * pack_factor + 1] =
              scale * static_cast<T_compute>(nvfp4_decode((uint(b) >> 4) & 0x0fu));
        } else {
          Ws_dst_inline[i * pack_factor + 0] =
              s0 * static_cast<T_compute>(b & 0x0f) + bias;
          Ws_dst_inline[i * pack_factor + 1] =
              s1 * static_cast<T_compute>(b & 0xf0) + bias;
        }
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS * pack_factor; ++i) {
        Ws_dst_inline[i] = T_compute(0);
      }
    }
  };
  auto load_w_unsafe = [&](int kk) {
    if (metal::is_same_v<T_compute, T_scale> && !nvfp4) {
      loader_w.load_unsafe();
    } else {
      load_w_inline(kk, /*n_unsafe=*/true);
    }
  };
  auto load_w_safe = [&](int kk) {
    if (metal::is_same_v<T_compute, T_scale> && !nvfp4) {
      loader_w.load_safe(short2(BK, n_tile));
    } else {
      load_w_inline(kk, /*n_unsafe=*/false);
    }
  };
  auto next_loaders = [&]() {
    X_src += BK;
    if (metal::is_same_v<T_compute, T_scale> && !nvfp4) {
      loader_w.next();
    }
  };

  if (!m_full) {
    if (!aligned_N && !n_full) {
      // m_full=false, n_full=false: load_x_safe + load_w_safe.
      int kk = 0;
      for (int k = 0; k < K_eff; k += BK, ++kk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_x_safe_m(short(m_tile));
        load_w_safe(kk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        next_loaders();
      }
    } else {
      // m_full=false, n_full=true: load_x_safe + load_w_unsafe.
      int kk = 0;
      for (int k = 0; k < K_eff; k += BK, ++kk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_x_safe_m(short(m_tile));
        load_w_unsafe(kk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        next_loaders();
      }
    }
  } else {
    if (!aligned_N && !n_full) {
      // m_full=true, n_full=false: load_x_unsafe + load_w_safe.
      int kk = 0;
      for (int k = 0; k < K_eff; k += BK, ++kk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_x_unsafe();
        load_w_safe(kk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        next_loaders();
      }
    } else {
      // m_full=true, n_full=true: HOT PATH — both unsafe loads.
      int kk = 0;
      for (int k = 0; k < K_eff; k += BK, ++kk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_x_unsafe();
        load_w_unsafe(kk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        next_loaders();
      }
    }
  }

  // ─── Epilogue: direct per-lane register→device write with inline
  //   float→T_act cast. Matches MLX's `BlockMMA::store_result` +
  //   `MMAFrag::store` (steel/gemm/mma.h:534-546, :100-112).
  //
  //   Previous version went through a `threadgroup float` scratch
  //   buffer with a `simdgroup_store` + threadgroup_barrier + per-
  //   thread cast loop. That required an extra 4 KB of threadgroup
  //   memory, an extra barrier per output tile, and 2× the
  //   threadgroup-memory traffic (write scratch + read scratch).
  //   The direct path below uses Apple's `thread_elements()` MSL
  //   extension to access the 2 per-lane floats of each
  //   `simdgroup_float8x8` accumulator fragment in registers, casts
  //   to T_act, and writes straight to device memory.
  //
  //   Per-lane element layout in an 8×8 simdgroup_matrix on Apple7+
  //   (BaseMMAFrag<T, 8, 8>::get_coord):
  //     qid = lane / 4
  //     fm  = (qid & 4) | ((lane / 2) % 4)
  //     fn  = (qid & 2) * 2 + (lane % 2) * 2
  //   Each lane holds two consecutive elements at (fm, fn) and
  //   (fm, fn+1) within the 8×8 frag.
  // ── Epilogue (MLX BlockMMA::store_result / store_result_safe) ────
  // VERBATIM MLX. `store_result(D, ldd)` writes the float
  // accumulator to device memory cast to U=T_act. The safe variant
  // bounds-checks against `(num_outs, num_els)` for the M/N tail.
  if (m_full && n_full) {
    mma_op.store_result(y_block, N);
  } else {
    mma_op.store_result_safe(y_block, N, short2(int(n_tile), int(m_tile)));
  }
  (void)out_scratch;
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_t kernel wrapper — quantized.h:1707-1778
//
// batched=0 only; batched=1 needs `adjust_matrix_offsets`
// (`quantized.h:1351`), deferred to P13 alongside MoE gather.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N>
[[kernel]] void affine_qmm_t_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // buffer(5) / buffer(6) / buffer(7) (K / N / M) replaced by file-
    // scope function constants QMM_K / QMM_N / QMM_M so this kernel is
    // recordable into an MTLIndirectComputeCommand (no setKernelBytes).
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  threadgroup T_compute Xs[BM * BK_padded];
  threadgroup T_compute Ws[BN * BK_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_t_impl_inline<T_act, T_compute, T_scale, group_size, bits, aligned_N>(
      w, scales, biases, x, y,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M, /*K_eff=*/QMM_K,
      simd_group_id, simd_lane_id, tgid);
}

// ─────────────────────────────────────────────────────────────────
// affine_gather_qmm_t kernel wrapper — the MoE grouped expert GEMM
// (mlx `affine_gather_qmm_rhs`, quantized.h:2252). Identical to
// `affine_qmm_t_kernel` except the per-expert weight slab is selected
// by `indices[c_row]`: the host sorts the (token,expert) rows by expert
// and PADS each expert's run to a multiple of BM=32, so every BM-row
// output tile maps to exactly ONE expert — the boundary case mlx's
// kernel handles with an in-tile loop is removed, so the tested
// `qmm_t_impl_inline` is reused verbatim with only the weight base
// shifted by `expert * per-expert-stride`. Weights are the gemma4
// SwitchGLU expert stack `[num_experts, N(out), K(in)]` (same int4
// affine format as the qmv path). Decode (m=1) keeps `affine_gather_qmv`.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N>
[[kernel]] void affine_gather_qmm_t_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // [M] u32 — the (padded, expert-sorted) per-row expert id. All BM
    // rows of a tile share one expert (host padding guarantees it).
    const device uint32_t*  indices  [[buffer(5)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  constexpr int pack_factor    = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  // Per-expert weight slab strides (mlx quantized.h:2298-2304, transpose):
  //   w:      N rows × (K * bytes_per_pack / pack_factor) packed bytes
  //   scales/biases: N rows × (K / group_size) groups
  const uint expert = indices[tgid.y * uint(BM)];
  // Sentinel tile (trailing padded rows beyond the real Mpad) → skip.
  if (expert >= uint(QMM_NUM_EXPERTS)) return;
  const int K_w = QMM_K * bytes_per_pack / pack_factor;
  const int K_g = QMM_K / group_size;
  const size_t w_stride = size_t(QMM_N) * size_t(K_w);   // bytes
  const size_t s_stride = size_t(QMM_N) * size_t(K_g);   // elements
  const device uint32_t* w_e = (const device uint32_t*)(
      (const device uint8_t*)w + size_t(expert) * w_stride);
  const device T_scale* s_e = scales + size_t(expert) * s_stride;
  const device T_scale* b_e = biases + size_t(expert) * s_stride;

  threadgroup T_compute Xs[BM * BK_padded];
  threadgroup T_compute Ws[BN * BK_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_t_impl_inline<T_act, T_compute, T_scale, group_size, bits, aligned_N>(
      w_e, s_e, b_e, x, y,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M, /*K_eff=*/QMM_K,
      simd_group_id, simd_lane_id, tgid);
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_t_splitk kernel wrapper — quantized.h:1780-1837
//
// Shifts the W / scales / biases / x / y pointers by `tid.z *
// k_partition_size` along K and `tid.z * split_k_partition_stride`
// along the M*N output dim, then calls qmm_t_impl with K_eff =
// k_partition_size. Caller (qmm_splitk in quantized.cpp:774) sums
// the resulting [split_k, M, N] intermediate along axis 0 into the
// final out.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N>
[[kernel]] void affine_qmm_t_splitk_kernel(
    const device uint32_t*  w                        [[buffer(0)]],
    const device T_scale*   scales                   [[buffer(1)]],
    const device T_scale*   biases                   [[buffer(2)]],
    const device T_act*     x                        [[buffer(3)]],
    device T_act*           y                        [[buffer(4)]],
    // buffer(5)..buffer(9) replaced by file-scope function constants
    // QMM_K / QMM_N / QMM_M / QMM_K_PARTITION_SIZE; the
    // split_k_partition_stride that MLX passes at buffer(9) is
    // computed inline as QMM_M * QMM_N (see `quantized.cpp:808`).
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int pack_factor    = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  const int k_start = int(tgid.z) * QMM_K_PARTITION_SIZE;
  const int split_k_partition_stride = QMM_M * QMM_N;

  const device T_act*   x_shift = x + k_start;
  const device uint8_t* wl      = (const device uint8_t*)w;
  wl += int64_t(k_start) * bytes_per_pack / pack_factor;
  const device T_scale* scales_shift = scales + k_start / group_size;
  const device T_scale* biases_shift = biases + k_start / group_size;
  device T_act*         y_shift      = y + int64_t(tgid.z) * split_k_partition_stride;

  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  threadgroup T_compute Xs[BM * BK_padded];
  threadgroup T_compute Ws[BN * BK_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_t_impl_inline<T_act, T_compute, T_scale, group_size, bits, aligned_N>(
      (const device uint32_t*)wl,
      scales_shift,
      biases_shift,
      x_shift,
      y_shift,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M,
      /*K_eff=*/QMM_K_PARTITION_SIZE,
      simd_group_id, simd_lane_id, tgid);
}

// ─────────────────────────────────────────────────────────────────
// qmm_n_impl_inline — quantized.h:1221-1348. Same BlockMMA tile
// (BM=BN=BK=32, WM=WN=2, TM=TN=2, 4 8×8 frags/SG) as qmm_t, but with
// transpose_b=false. The differences vs qmm_t_impl_inline:
//
// 1. Ws layout is `[BK, BN_padded]` (K rows × N cols + pad) instead
//    of `[BN, BK_padded]`. B-frag is loaded without transpose, stride
//    `BN_padded`.
// 2. W src layout is `[K, N/pack_factor]` u32-packed (each K-row has
//    N packed values). Per-thread load: bi indexes K-row in [0, BK),
//    bj indexes N-col-packed in [0, BCOLS_PACKED). Per BK iter,
//    advance W_src by `BK * N * bytes_per_pack / pack_factor` bytes
//    (BlockLoader<reduction_dim=0>::tile_stride at
//    `quantized.h:614-616`).
// 3. Scales / biases layout is `[K, N/group_size]` (group along N).
//    Per-thread scale offset is `bi * (N / group_size)` (each K-row
//    has its own scales row). Per BK iter, advance by
//    `group_stride = BK * N / group_size`
//    (QuantizedBlockLoader<reduction_dim=0>::next() at
//    `quantized.h:686-687`).
// 4. No aligned_N flag — MLX qmm_n asserts caller guarantees N % BN
//    == 0 (see comment above; `quantized.cpp:1847` template signature).
//
// bits=4 only. group_size ∈ {32, 64, 128} (assert at
// `quantized.h:573-578`: BCOLS=BN=32 ≤ group_size). For gs=32, BN=gs
// → exactly one (scale, bias) per K-row per N-tile.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
METAL_FUNC void qmm_n_impl_inline(
    const device uint32_t*  w,
    const device T_scale*   scales,
    const device T_scale*   biases,
    const device T_act*     x,
    device T_act*           y,
    threadgroup T_act*      Xs,
    threadgroup T_act*      Ws,
    threadgroup float*      out_scratch,
    const int              K,
    const int              N,
    const int              M,
    uint  simd_group_id,
    uint  simd_lane_id,
    uint3 tgid)
{
  static_assert(bits == 4, "qmm_n_impl_inline only instantiated for bits=4");
  static_assert(group_size == 32 || group_size == 64 || group_size == 128,
                "qmm_n_impl_inline expects group_size in {32, 64, 128}");

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int TM = BM / (8 * WM);            // 2
  constexpr int TN = BN / (8 * WN);            // 2
  constexpr int KFR = BK / 8;                  // 4 K-frags per BK iter
  constexpr int BK_padded = BK + 16 / int(sizeof(T_act));
  constexpr int BN_padded = BN + 16 / int(sizeof(T_act));
  constexpr int TGP = WM * WN * SIMD_SIZE;     // 128

  constexpr int pack_factor    = get_pack_factor<bits, 8>();    // 2
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();    // 1
  constexpr int BCOLS_PACKED   = BN / pack_factor;              // 16
  // Per-thread W reads: BROWS=BK, BCOLS=BN → BCOLS_PACKED=16, n_reads
  // = (BCOLS_PACKED * BROWS) / tgp_size = (16 * 32) / 128 = 4
  // (`quantized.h:587-588`).
  constexpr int N_READS = (BCOLS_PACKED * BK) / TGP;            // 4

  // Per-thread X read: 8 halves per thread (BM*BK / TGP = 1024/128).
  constexpr int X_N_READS = (BM * BK) / TGP;                    // 8
  constexpr int X_TCOLS   = BK / X_N_READS;                     // 4
  static_assert(X_N_READS == 8, "expected 8 X halves per thread");

  const uint thread_idx = simd_group_id * SIMD_SIZE + simd_lane_id;

  // ── Output-tile origin and per-tile valid extents ──────────────
  const uint c_row = tgid.y * BM;  // along M
  const uint c_col = tgid.x * BN;  // along N
  if (c_row >= uint(M) || c_col >= uint(N)) return;
  const uint m_tile = (c_row + BM <= uint(M)) ? uint(BM) : uint(M) - c_row;
  const bool m_full = (m_tile == BM);

  // ── X loader thread coords (per BlockLoader; reduction_dim=1) ──
  const uint bi_x = thread_idx / uint(X_TCOLS);          // [0, BM)
  const uint bj_x = uint(X_N_READS) * (thread_idx % uint(X_TCOLS));  // 0,8,16,24

  // ── W loader thread coords (per QuantizedBlockLoader,
  //    reduction_dim=0 — BROWS=BK, BCOLS=BN) ────────────────────
  const uint bi_w = (uint(N_READS) * thread_idx) / uint(BCOLS_PACKED);     // [0, BK)
  const uint bj_w = (uint(N_READS) * thread_idx) % uint(BCOLS_PACKED);     // 0,4,8,12

  // ── Block base pointers (after y_row / y_col shift per
  //    `quantized.h:1267-1273`) ──────────────────────────────────
  const int N_w = N * bytes_per_pack / pack_factor;       // packed bytes per row of W
  const int N_g = N / group_size;                         // groups per row of W
  const device uint8_t* wl_base = (const device uint8_t*)w;
  const device T_act* x_block    = x + int64_t(c_row) * K;
  const device uint8_t* w_block  = wl_base + c_col * bytes_per_pack / pack_factor;
  const device T_scale* s_block  = scales + c_col / group_size;
  const device T_scale* b_block  = biases + c_col / group_size;
  device T_act* y_block          = y + int64_t(c_row) * N + c_col;

  // ── Per-thread source/dest pointers (BlockLoader constructor :47-58
  //    and QuantizedBlockLoader<reduction_dim=0> constructor
  //    :605-626) ─────────────────────────────────────────────────
  threadgroup T_act* Xs_dst = Xs + bi_x * BK_padded + bj_x;
  const device T_act* X_src = x_block + bi_x * K + bj_x;

  threadgroup T_act* Ws_dst = Ws + bi_w * BN_padded + bj_w * pack_factor;
  const device uint8_t* W_src = w_block + bi_w * N_w + bj_w * bytes_per_pack;
  const device T_scale* Sc_row = s_block + bi_w * N_g;
  const device T_scale* Bs_row = b_block + bi_w * N_g;
  // Group stride per BK iter for reduction_dim=0
  // (`quantized.h:618`): BROWS * src_ld / group_size = BK * N / gs.
  const int group_stride = BK * N / group_size;

  // ── Accumulators (BlockMMA Ctile = TM×TN of 8×8 float frags) ───
  simdgroup_float8x8 acc[TM][TN];
  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = simdgroup_float8x8(0.0f);
    }
  }

  const int sgM = int(simd_group_id) / WN;
  const int sgN = int(simd_group_id) % WN;

  // K loop — same MLX `(K % BK) != 0` tail handling pattern as
  // qmm_t. K_eff allows shortening (no qmm_n splitk in MLX so we
  // always use K_eff = K here, but keep the parameter for symmetry).
  const int k_blocks    = K / BK;
  const int k_tail      = K - k_blocks * BK;

  for (int kb = 0; kb < k_blocks; ++kb) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ─── X loader (load_unsafe for full M, load_safe for M-tail) ──
    if (m_full || bi_x < m_tile) {
      ((threadgroup vec<T_act, 4>*)Xs_dst)[0] =
          ((const device vec<T_act, 4>*)X_src)[0];
      ((threadgroup vec<T_act, 4>*)Xs_dst)[1] =
          ((const device vec<T_act, 4>*)X_src)[1];
    } else {
      ((threadgroup vec<T_act, 4>*)Xs_dst)[0] = vec<T_act, 4>(0);
      ((threadgroup vec<T_act, 4>*)Xs_dst)[1] = vec<T_act, 4>(0);
    }

    // ─── W loader (QuantizedBlockLoader<T, BK, BN, BN_padded, 0,
    //   TGP, gs, 4>::load_unsafe — no N-tail handling per MLX
    //   qmm_n's no-aligned_N assumption). Inline 4-bit dequant. ────
    {
      // In-register T_scale → T_act cast (`INT4_PARITY_PROBES.md` §7):
      // scales/biases ship F16 on disk; dequant math stays in T_act
      // (MLX quantized.h:521-527 keeps it in the kernel's scalar type).
      T_act scale = static_cast<T_act>(*Sc_row);
      T_act bias  = static_cast<T_act>(*Bs_row);
      T_act s0 = scale;
      T_act s1 = scale / static_cast<T_act>(16.0f);
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS; ++i) {
        uint8_t b = W_src[i * bytes_per_pack];
        Ws_dst[i * pack_factor + 0] =
            s0 * static_cast<T_act>(b & 0x0f) + bias;
        Ws_dst[i * pack_factor + 1] =
            s1 * static_cast<T_act>(b & 0xf0) + bias;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ─── BlockMMA::mma — KFR K-frag MMAs per BK iter; B-frag is
    //   loaded WITHOUT transpose because Ws is already [BK, BN]. ─
    MLX_MTL_PRAGMA_UNROLL
    for (int kf = 0; kf < KFR; ++kf) {
      simdgroup_matrix<T_act, 8, 8> A_frag[TM];
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        threadgroup const T_act* a_ptr =
            Xs + (sgM * 16 + i * 8) * BK_padded + kf * 8;
        simdgroup_load(A_frag[i], a_ptr, BK_padded);
      }

      simdgroup_matrix<T_act, 8, 8> B_frag[TN];
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; ++j) {
        int n_off = sgN * 16 + j * 8;
        threadgroup const T_act* b_ptr =
            Ws + (kf * 8) * BN_padded + n_off;
        simdgroup_load(B_frag[j], b_ptr, BN_padded);
      }

      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              acc[i][j], A_frag[i], B_frag[j], acc[i][j]);
        }
      }
    }

    // ─── Advance pointers per BlockLoader / QuantizedBlockLoader.next() ──
    //   X: tile_stride = BCOLS = BK halves (reduction_dim=1).
    //   W: tile_stride = BROWS * src_ld * bytes_per_pack / pack_factor
    //      = BK * N * bytes_per_pack / pack_factor (reduction_dim=0).
    //   Scales/biases: + group_stride = BK * N / group_size per BK iter
    //      (reduction_dim=0 branch at `quantized.h:686-687`).
    X_src  += BK;
    W_src  += BK * N_w;
    Sc_row += group_stride;
    Bs_row += group_stride;
  }

  // ─── K-tail: load_safe with short2(BN, num_k) for both X and W.
  //   X: zero past `num_k` cols; W: zero past `num_k` rows. ─────
  if (k_tail > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // X tail — for each thread, zero positions ≥ num_k in the row.
    {
      vec<T_act, 4> v0 = vec<T_act, 4>(0);
      vec<T_act, 4> v1 = vec<T_act, 4>(0);
      if (m_full || bi_x < m_tile) {
        MLX_MTL_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          int col = int(bj_x) + i;
          if (col < k_tail) {
            ((thread T_act*)&v0)[i] = X_src[i];
          }
        }
        MLX_MTL_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          int col = int(bj_x) + 4 + i;
          if (col < k_tail) {
            ((thread T_act*)&v1)[i] = X_src[4 + i];
          }
        }
      }
      ((threadgroup vec<T_act, 4>*)Xs_dst)[0] = v0;
      ((threadgroup vec<T_act, 4>*)Xs_dst)[1] = v1;
    }

    // W tail — zero rows past num_k in the BK direction
    // (load_safe with reduction_dim=0 short2(BCOLS=BN, num_k)
    // at `quantized.h:653-657`).
    if (int(bi_w) < k_tail) {
      // In-register T_scale → T_act cast (`INT4_PARITY_PROBES.md` §7):
      // scales/biases ship F16 on disk; dequant math stays in T_act
      // (MLX quantized.h:521-527 keeps it in the kernel's scalar type).
      T_act scale = static_cast<T_act>(*Sc_row);
      T_act bias  = static_cast<T_act>(*Bs_row);
      T_act s0 = scale;
      T_act s1 = scale / static_cast<T_act>(16.0f);
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS; ++i) {
        uint8_t b = W_src[i * bytes_per_pack];
        Ws_dst[i * pack_factor + 0] =
            s0 * static_cast<T_act>(b & 0x0f) + bias;
        Ws_dst[i * pack_factor + 1] =
            s1 * static_cast<T_act>(b & 0xf0) + bias;
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < N_READS * pack_factor; ++i) {
        Ws_dst[i] = T_act(0);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    MLX_MTL_PRAGMA_UNROLL
    for (int kf = 0; kf < KFR; ++kf) {
      simdgroup_matrix<T_act, 8, 8> A_frag[TM];
      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        threadgroup const T_act* a_ptr =
            Xs + (sgM * 16 + i * 8) * BK_padded + kf * 8;
        simdgroup_load(A_frag[i], a_ptr, BK_padded);
      }

      simdgroup_matrix<T_act, 8, 8> B_frag[TN];
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; ++j) {
        int n_off = sgN * 16 + j * 8;
        threadgroup const T_act* b_ptr =
            Ws + (kf * 8) * BN_padded + n_off;
        simdgroup_load(B_frag[j], b_ptr, BN_padded);
      }

      MLX_MTL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        MLX_MTL_PRAGMA_UNROLL
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              acc[i][j], A_frag[i], B_frag[j], acc[i][j]);
        }
      }
    }
  }

  // ─── Epilogue (same as qmm_t: simdgroup_store → out_scratch →
  //   per-thread cast to T). MLX qmm_n uses store_result vs
  //   store_result_safe based only on M-tail; we keep the same
  //   short-circuit (m_full ⇒ unconditional N-cols write since N
  //   is assumed aligned to BN by the caller). ──────────────────
  threadgroup_barrier(mem_flags::mem_threadgroup);

  MLX_MTL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < TN; ++j) {
      int row_base = sgM * 16 + i * 8;
      int col_base = sgN * 16 + j * 8;
      simdgroup_store(acc[i][j],
                      out_scratch + row_base * BN + col_base,
                      BN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (m_full) {
    for (uint t = thread_idx; t < uint(BM * BN); t += uint(TGP)) {
      uint r = t / uint(BN);
      uint c = t % uint(BN);
      y_block[r * N + c] = T_act(out_scratch[t]);
    }
  } else {
    for (uint t = thread_idx; t < uint(BM * BN); t += uint(TGP)) {
      uint r = t / uint(BN);
      uint c = t % uint(BN);
      if (r < m_tile) {
        y_block[r * N + c] = T_act(out_scratch[t]);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_n kernel wrapper — quantized.h:1847-1897
//
// batched=0 only; batched=1 needs `adjust_matrix_offsets`
// (`quantized.h:1351`), deferred to P13 alongside MoE gather.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits>
[[kernel]] void affine_qmm_n_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // K / N / M ride as file-scope function constants
    // QMM_K / QMM_N / QMM_M (same as qmm_t for ICB recording).
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_act));
  constexpr int BN_padded = BN + 16 / int(sizeof(T_act));
  threadgroup T_act Xs[BM * BK_padded];
  threadgroup T_act Ws[BK * BN_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_n_impl_inline<T_act, T_scale, group_size, bits>(
      w, scales, biases, x, y,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M,
      simd_group_id, simd_lane_id, tgid);
}

// ─────────────────────────────────────────────────────────────────
// Instantiations — one symbol per (dtype, group_size, aligned_N)
// for affine_qmm_t (batched=0 only) and affine_qmm_t_splitk, plus
// one symbol per (dtype, group_size) for affine_qmm_n (batched=0).
// Matches the MLX dispatcher's `concatenate(...)` kernel-name pattern
// at `quantized.cpp:728-737` (qmm_t / qmm_n) and `:830-838` (splitk).
// ─────────────────────────────────────────────────────────────────

// INST_QMM_T_C: explicit compute-dtype instantiation. Symbol carries
// `_c_<compute_tag>` after `_<act_tag>` so existing `_s_<scale_tag>_gs_..`
// suffix parsing keeps working. Existing INST_QMM_T (below) is a thin
// wrapper that fixes T_compute = T_act, preserving every legacy symbol.
#define INST_QMM_T_C(act_tag, act_type, ctag, ctype, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                            \
      "affine_qmm_t_" #act_tag "_c_" #ctag "_s_" #scale_tag "_gs_" #gs                             \
      "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                                           \
  affine_qmm_t_kernel<act_type, ctype, scale_type, gs, 4, aln_val>(                                \
      const device uint32_t*   w        [[buffer(0)]],                                             \
      const device scale_type* scales   [[buffer(1)]],                                             \
      const device scale_type* biases   [[buffer(2)]],                                             \
      const device act_type*   x        [[buffer(3)]],                                             \
      device act_type*         y        [[buffer(4)]],                                             \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                                      \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                           \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_T_SPLITK_C(act_tag, act_type, ctag, ctype, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                                   \
      "affine_qmm_t_splitk_" #act_tag "_c_" #ctag "_s_" #scale_tag "_gs_" #gs                            \
      "_b_4_alN_" #aln_tag)]] [[kernel]] void                                                            \
  affine_qmm_t_splitk_kernel<act_type, ctype, scale_type, gs, 4, aln_val>(                               \
      const device uint32_t*   w                        [[buffer(0)]],                                   \
      const device scale_type* scales                   [[buffer(1)]],                                   \
      const device scale_type* biases                   [[buffer(2)]],                                   \
      const device act_type*   x                        [[buffer(3)]],                                   \
      device act_type*         y                        [[buffer(4)]],                                   \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                                            \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                                 \
      uint3 tgid          [[threadgroup_position_in_grid]]);

// Legacy INST_QMM_T: T_compute = T_act. Emits the old symbol name
// (no `_c_` segment) so existing pipeline-cache lookups keep hitting.
#define INST_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val)   \
  template [[host_name(                                                               \
      "affine_qmm_t_" #act_tag "_s_" #scale_tag "_gs_" #gs                            \
      "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                              \
  affine_qmm_t_kernel<act_type, act_type, scale_type, gs, 4, aln_val>(                \
      const device uint32_t*   w        [[buffer(0)]],                                \
      const device scale_type* scales   [[buffer(1)]],                                \
      const device scale_type* biases   [[buffer(2)]],                                \
      const device act_type*   x        [[buffer(3)]],                                \
      device act_type*         y        [[buffer(4)]],                                \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                         \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                              \
      uint3 tgid          [[threadgroup_position_in_grid]]);

// MoE grouped expert GEMM (T_compute = T_act). Extra buffer(5) = the
// padded, expert-sorted per-row expert index.
#define INST_GATHER_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                    \
      "affine_gather_qmm_t_" #act_tag "_s_" #scale_tag "_gs_" #gs                          \
      "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                                   \
  affine_gather_qmm_t_kernel<act_type, act_type, scale_type, gs, 4, aln_val>(              \
      const device uint32_t*   w        [[buffer(0)]],                                     \
      const device scale_type* scales   [[buffer(1)]],                                     \
      const device scale_type* biases   [[buffer(2)]],                                     \
      const device act_type*   x        [[buffer(3)]],                                     \
      device act_type*         y        [[buffer(4)]],                                     \
      const device uint32_t*   indices  [[buffer(5)]],                                     \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                              \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                   \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_T_SPLITK(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                   \
      "affine_qmm_t_splitk_" #act_tag "_s_" #scale_tag "_gs_" #gs                         \
      "_b_4_alN_" #aln_tag)]] [[kernel]] void                                             \
  affine_qmm_t_splitk_kernel<act_type, act_type, scale_type, gs, 4, aln_val>(             \
      const device uint32_t*   w                        [[buffer(0)]],                    \
      const device scale_type* scales                   [[buffer(1)]],                    \
      const device scale_type* biases                   [[buffer(2)]],                    \
      const device act_type*   x                        [[buffer(3)]],                    \
      device act_type*         y                        [[buffer(4)]],                    \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                             \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                  \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_N(act_tag, act_type, scale_tag, scale_type, gs)               \
  template [[host_name(                                                        \
      "affine_qmm_n_" #act_tag "_s_" #scale_tag "_gs_" #gs                     \
      "_b_4_batch_0")]] [[kernel]] void                                        \
  affine_qmm_n_kernel<act_type, scale_type, gs, 4>(                            \
      const device uint32_t*   w        [[buffer(0)]],                         \
      const device scale_type* scales   [[buffer(1)]],                         \
      const device scale_type* biases   [[buffer(2)]],                         \
      const device act_type*   x        [[buffer(3)]],                         \
      device act_type*         y        [[buffer(4)]],                         \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                       \
      uint3 tgid          [[threadgroup_position_in_grid]]);

#define INST_QMM_ALL(act_tag, act_type, scale_tag, scale_type, gs)             \
  INST_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, true,  true)        \
  INST_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, false, false)       \
  INST_GATHER_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, true,  true) \
  INST_GATHER_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, false, false)\
  INST_QMM_T_SPLITK(act_tag, act_type, scale_tag, scale_type, gs, true,  true) \
  INST_QMM_T_SPLITK(act_tag, act_type, scale_tag, scale_type, gs, false, false)\
  INST_QMM_N(act_tag, act_type, scale_tag, scale_type, gs)

// Coverage: see header note in `quantized_qmv.metal` — T_scale=half
// (every mlx-community 4bit ships F16 scales), T_act per `torch_dtype`.
// `bfloat × bfloat` is removed — that was the P1-P6 regression path.
INST_QMM_ALL(f16,  half,   f16, half,  32)
INST_QMM_ALL(f16,  half,   f16, half,  64)
INST_QMM_ALL(f16,  half,   f16, half, 128)
INST_QMM_ALL(bf16, bfloat, f16, half,  32)
INST_QMM_ALL(bf16, bfloat, f16, half,  64)
INST_QMM_ALL(bf16, bfloat, f16, half, 128)

// bf16-scale variants — Qwen3-MoE / `torch_dtype: bfloat16` ships BF16
// scales/biases. Mirrors MLX's `T_scale = bfloat16_t` instantiations.
INST_QMM_ALL(bf16, bfloat, bf16, bfloat,  32)
INST_QMM_ALL(bf16, bfloat, bf16, bfloat,  64)
INST_QMM_ALL(bf16, bfloat, bf16, bfloat, 128)
INST_QMM_ALL(f16,  half,   bf16, bfloat,  32)
INST_QMM_ALL(f16,  half,   bf16, bfloat,  64)
INST_QMM_ALL(f16,  half,   bf16, bfloat, 128)

// Apple7 (M1) fast-path: bf16 device dtype, f16 compute. Skips the
// slow bf16 simdgroup_multiply_accumulate emulation and runs the MMA
// in half — ~1.7× faster on M1 Max for prefill matmuls. Output stays
// bf16 so the residual stream's bf16 dynamic range is preserved.
// Only qmm_t (not qmm_n; qmm_n isn't on the prefill hot path).
// 8-bit Standard qmm_t (Gemma4 MLP projections: 8-bit g64 with bf16
// activations + bf16 scales; f16/f16 combo kept for unit tests). Only
// the Standard kernel family — NAX static_asserts bits==4 and the
// 8-bit path never routes to SplitK (`pick_qmm_t_kernel` gate in the
// lowering). Same-compute only: the mixed-compute inline W loader is
// nibble-specialized.
#define INST_QMM_T_B8(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                               \
      "affine_qmm_t_" #act_tag "_s_" #scale_tag "_gs_" #gs                            \
      "_b_8_alN_" #aln_tag "_batch_0")]] [[kernel]] void                              \
  affine_qmm_t_kernel<act_type, act_type, scale_type, gs, 8, aln_val>(                \
      const device uint32_t*   w        [[buffer(0)]],                                \
      const device scale_type* scales   [[buffer(1)]],                                \
      const device scale_type* biases   [[buffer(2)]],                                \
      const device act_type*   x        [[buffer(3)]],                                \
      device act_type*         y        [[buffer(4)]],                                \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                         \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                              \
      uint3 tgid          [[threadgroup_position_in_grid]]);

INST_QMM_T_B8(bf16, bfloat, bf16, bfloat, 64, true,  true)
INST_QMM_T_B8(bf16, bfloat, bf16, bfloat, 64, false, false)
INST_QMM_T_B8(f16,  half,   f16,  half,   64, true,  true)
INST_QMM_T_B8(f16,  half,   f16,  half,   64, false, false)

// 8-bit MoE grouped expert GEMM (non-NAX). OptiQ / mixed-bit Gemma4 ships
// 8-bit SwitchGLU experts on edge layers; M5 runs them through the NAX gather
// (which has b8), but the non-NAX path here was b4-only, so the 8-bit expert
// PREFILL requested a nil function on M1 → `computeFunction must not be nil`.
// The kernel body is already bits-generic (get_pack_factor<bits,8> +
// qmm_t_impl_inline<...,bits,...>, same as the dense b8 above); only the
// instantiation was missing. gs=64 (OptiQ experts are group_size 64).
#define INST_GATHER_QMM_T_B8(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                       \
      "affine_gather_qmm_t_" #act_tag "_s_" #scale_tag "_gs_" #gs                            \
      "_b_8_alN_" #aln_tag "_batch_0")]] [[kernel]] void                                     \
  affine_gather_qmm_t_kernel<act_type, act_type, scale_type, gs, 8, aln_val>(                \
      const device uint32_t*   w        [[buffer(0)]],                                       \
      const device scale_type* scales   [[buffer(1)]],                                       \
      const device scale_type* biases   [[buffer(2)]],                                       \
      const device act_type*   x        [[buffer(3)]],                                       \
      device act_type*         y        [[buffer(4)]],                                       \
      const device uint32_t*   indices  [[buffer(5)]],                                       \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                                \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                     \
      uint3 tgid          [[threadgroup_position_in_grid]]);

INST_GATHER_QMM_T_B8(bf16, bfloat, bf16, bfloat, 64, true,  true)
INST_GATHER_QMM_T_B8(bf16, bfloat, bf16, bfloat, 64, false, false)
INST_GATHER_QMM_T_B8(f16,  half,   f16,  half,   64, true,  true)
INST_GATHER_QMM_T_B8(f16,  half,   f16,  half,   64, false, false)

INST_QMM_T_C(bf16, bfloat, f16, half, f16, half,  64, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half,  64, false, false)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half,  32, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half,  32, false, false)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 128, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, f16, half, 128, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half,  64, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half,  64, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half,  32, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half,  32, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 128, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, f16, half, 128, false, false)

INST_QMM_T_C(bf16, bfloat, f16, half, bf16, bfloat,  64, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, bf16, bfloat,  64, false, false)
INST_QMM_T_C(bf16, bfloat, f16, half, bf16, bfloat,  32, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, bf16, bfloat,  32, false, false)
INST_QMM_T_C(bf16, bfloat, f16, half, bf16, bfloat, 128, true,  true)
INST_QMM_T_C(bf16, bfloat, f16, half, bf16, bfloat, 128, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, bf16, bfloat,  64, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, bf16, bfloat,  64, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, bf16, bfloat,  32, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, bf16, bfloat,  32, false, false)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, bf16, bfloat, 128, true,  true)
INST_QMM_T_SPLITK_C(bf16, bfloat, f16, half, bf16, bfloat, 128, false, false)

// ─────────────────────────────────────────────────────────────────
// nvfp4_qmm_t — NVFP4 prefill matmul (standard 32×32 tile)
//
// Reuses `qmm_t_impl_inline` with the `nvfp4` template flag: E2M1 LUT
// decode, no per-group bias, and the inline W-dequant path forced (its
// group_size=16 < BK=32 can't go through `QuantizedBlockLoader`).
// 4-buffer layout (no biases buffer) — `scales` is passed as the
// ignored bias pointer so the shared pointer arithmetic stays valid.
// T_compute == T_act (no separate f16-compute fast-path wired for NVFP4
// yet). Standard tile only; NAX prefill is the planned perf path.
// ─────────────────────────────────────────────────────────────────
template <typename T_act, typename T_compute, typename T_scale,
          int group_size, int bits, bool aligned_N>
[[kernel]] void nvfp4_qmm_t_kernel(
    const device uint32_t*  w       [[buffer(0)]],
    const device T_scale*   scales  [[buffer(1)]],
    // biases [[buffer(2)]] = scales buffer (dummy). 5-buffer layout MUST
    // match affine's (w,scales,biases,x,y); MTL4 mis-dispatches the
    // 4-buffer (x@2,y@3) variant — see note in quantized_qmv.metal.
    const device T_scale*   biases  [[buffer(2)]],
    const device T_act*     x       [[buffer(3)]],
    device T_act*           y       [[buffer(4)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint3 tgid          [[threadgroup_position_in_grid]])
{
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BK_padded = BK + 16 / int(sizeof(T_compute));
  threadgroup T_compute Xs[BM * BK_padded];
  threadgroup T_compute Ws[BN * BK_padded];
  threadgroup float out_scratch[BM * BN];
  qmm_t_impl_inline<T_act, T_compute, T_scale, group_size, bits, aligned_N,
                    /*nvfp4=*/true>(
      w, scales, biases, x, y,
      Xs, Ws, out_scratch,
      QMM_K, QMM_N, QMM_M, /*K_eff=*/QMM_K,
      simd_group_id, simd_lane_id, tgid);
}

#define INST_NVFP4_QMM_T(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
  template [[host_name(                                                                  \
      "nvfp4_qmm_t_" #act_tag "_s_" #scale_tag "_gs_" #gs                                 \
      "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                                  \
  nvfp4_qmm_t_kernel<act_type, act_type, scale_type, gs, 4, aln_val>(                      \
      const device uint32_t*   w       [[buffer(0)]],                                     \
      const device scale_type* scales  [[buffer(1)]],                                     \
      const device scale_type* biases  [[buffer(2)]],                                     \
      const device act_type*   x       [[buffer(3)]],                                     \
      device act_type*         y       [[buffer(4)]],                                     \
      uint  simd_group_id [[simdgroup_index_in_threadgroup]],                             \
      uint  simd_lane_id  [[thread_index_in_simdgroup]],                                  \
      uint3 tgid          [[threadgroup_position_in_grid]]);

// group_size always 16 for NVFP4; T_scale always half (folded F16).
INST_NVFP4_QMM_T(f16, half, f16, half, 16, true, true)
INST_NVFP4_QMM_T(f16, half, f16, half, 16, false, false)
INST_NVFP4_QMM_T(bf16, bfloat, f16, half, 16, true, true)
INST_NVFP4_QMM_T(bf16, bfloat, f16, half, 16, false, false)
