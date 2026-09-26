// SPDX-License-Identifier: Apache-2.0
//
// NAX (Apple9 / M4+) port of MLX's `qmm_t_nax_tgp_impl` /
// `affine_qmm_t_nax` from `mlx/backend/metal/kernels/quantized_nax.h`.
//
// Tile shape: BM=BN=BK=64, WM=WN=2, SK=32, TGP=128 (4 simdgroups).
//   BK=64 matches MLX's production instantiation in
//   `mlx/backend/metal/kernels/quantized_nax.metal`
//   (`instantiate_quantized_aligned_batched(affine_qmm_t_nax, ..., 64, 64, 64, 2, 2, ...)`).
//
// Why BK=64 (not 32):
//   - QuantizedBlockLoader requires `BCOLS_PACKED/n_reads == n_groups`
//     in its gs=32 specialization. With BK=32, gs=32 → BCOLS_PACKED=16,
//     n_reads=8, n_groups=1 → 16/8 != 1; MLX never instantiates this
//     combination. BK=64 with gs=32 → BCOLS_PACKED=32, n_reads=16,
//     n_groups=2 → 32/16 == 2 ✓ (matches MLX's `quantized_nax.metal`).
//   - For NAX we currently skip gs=32 entirely (the dispatcher gates
//     `group_size != 32` for the Nax pick); the general QuantizedBlockLoader
//     path is inlined here for gs=64 and gs=128. gs=32 would need the
//     specialized loader with per-thread `group_id` — straightforward
//     follow-up if production models start shipping gs=32 quants.
//
// MLX reference: `mlx/backend/metal/quantized.cpp:695` dispatches NAX
// when `K % 64 == 0`. Same gate is used Rust-side in
// `pick_qmm_t_kernel`.
//
// Scratchy adaptations relative to MLX's kernel:
//   - K, N, M come from file-scope function constants (indices 0/1/2)
//     rather than `const constant int& [[buffer(5/6/7)]]` — required
//     so the pipeline is recordable into an MTLIndirectComputeCommand.
//   - W-loader is inlined (QuantizedBlockLoader struct not imported);
//     general-loader path only (gs ≥ 64).
//   - batched=0 only; no `adjust_matrix_offsets` (B=1).
//
// Buffer layout matches the existing quantized_qmm.metal:
//   w[0], scales[1], biases[2], x[3], y[4]
//
// Dispatch: grid = (ceil(N/64), ceil(M/64), 1), tpg = (128, 1, 1).
// Kernel selected only on M4+ (is_nax_capable check in pick_qmm_t_kernel).

#include "metal_nax.h"   // vendors NAXTile + tile_matmad_nax + MPP internals
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

#define MLX_MTL_CONST static constant constexpr const

#ifndef MLX_MTL_PRAGMA_NO_UNROLL
#define MLX_MTL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
#endif

MLX_MTL_CONST int NAX_SIMD_SIZE = 32;

// ─────────────────────────────────────────────────────────────────
// Function constants — same slot indices as quantized_qmm.metal so
// the lowering pass shares the same ConstantValue::int slot map.
// ─────────────────────────────────────────────────────────────────

constant int QMM_K [[function_constant(0)]];
constant int QMM_N [[function_constant(1)]];
constant int QMM_M [[function_constant(2)]];
// Block-diagonal span attention (hd512 unfused QKᵀ): per-position span-label
// granularity (== metal KV block size). 0 / undefined → no bound (dense path
// byte-identical).
constant uint QK_SPAN_BLOCK_NAX_RAW [[function_constant(3)]];
constant uint QK_SPAN_BLOCK_NAX =
    is_function_constant_defined(QK_SPAN_BLOCK_NAX_RAW) ? QK_SPAN_BLOCK_NAX_RAW : 0u;
// MoE grouped GEMM only: number of valid experts. Padded tiles whose
// expert index == QMM_NUM_EXPERTS are trailing sentinels and skip.
constant int QMM_NUM_EXPERTS [[function_constant(4)]];

// ─────────────────────────────────────────────────────────────────
// NVFP4 E2M1 decode (sign-magnitude 4-bit). Same byte layout as affine
// int4, so the NAX tile machinery is shared via a `bool nvfp4` flag;
// the decode differs (LUT, no per-group bias) and — because NVFP4's
// group_size (16) is smaller than BK (64) — the scale is indexed per
// byte by absolute K position rather than once per thread. Duplicated
// per-file (each .metal → its own metallib). Matches Python vLLM's
// `kE2M1ToFloat`.
// ─────────────────────────────────────────────────────────────────
// E2M1 decode computed arithmetically — NO lookup table. A file-scope
// `constant float[]` LUT compiles to a static initializer / global
// constructor (`air.static_init`), which the MTL4 pipeline-compiler
// path does NOT run → uninitialized table → garbage under MTL4. E2M1 =
// 1 sign · 2 exp · 1 mantissa: mag = (e==0) ? m·0.5 : (1+m·0.5)·2^(e-1)
// → {0,.5,1,1.5,2,3,4,6} for codes 0..7, bit-identical to the old LUT.
// (NAX is runtime-source-compiled, but keep this consistent so all
// three nvfp4 decoders share one ctor-free definition.)
inline float nvfp4_decode(uint code) {
  uint e = (code >> 1u) & 0x3u;
  uint m = code & 0x1u;
  float mant = (e == 0u) ? (float(m) * 0.5f) : (1.0f + float(m) * 0.5f);
  float scale = (e == 0u) ? 1.0f : float(1u << (e - 1u));
  float mag = mant * scale;
  return (code & 0x8u) ? -mag : mag;
}

// ─────────────────────────────────────────────────────────────────
// qmm_t_nax_impl — port of qmm_t_nax_tgp_impl (quantized_nax.h)
//
// Template params:
//   T_act      activation/output type (half or bfloat)
//   T_scale    scale/bias type (half)
//   group_size affine-quant group size (64 or 128 — gs=32 not
//              supported here; the dispatcher falls back to Standard
//              qmm_t for gs=32)
//   bits       = 4 or 8
//   aligned_N  N % 64 == 0 → skip N-tail handling
//
// Tile shape (matches MLX BM=BN=BK=64, WM=WN=2):
//   SM = SN = BM/WM = BN/WN = 32   (per-simdgroup M/N extent)
//   SK = 32                          (inner K step; BK/SK = 2 iters)
//   TM = TN = TK = 2                 (NAXTile row/col counts)
//   TGP = 128 threads, 4 simdgroups
//   BK_padded = BK + 16/sizeof(T)  = 72 for half/bfloat (bank-conflict pad)
//
// W-loader (QuantizedBlockLoader equivalent, reduction_dim=1, general):
//   BROWS = BN = 64, BCOLS = BK = 64
//   BCOLS_PACKED = BK/pack_factor = 32
//   n_reads = (BCOLS_PACKED * BN) / TGP = (32*64)/128 = 16
//   bi_w = (n_reads * thread_idx) / BCOLS_PACKED = thread_idx/2 ∈ [0,63]
//   bj_w = (n_reads * thread_idx) % BCOLS_PACKED ∈ {0, 16}
//   scales start at s_block + bi_w * K_g (no bj offset — gs ≥ BK)
//   scale advance: group_steps = group_size/BK = group_size/64
//     gs=64 → advance every BK step (group_steps == 1)
//     gs=128 → advance every 2 BK steps (group_steps == 2)
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, bool aligned_N,
          bool nvfp4 = false>
METAL_FUNC void qmm_t_nax_impl(
    const device uint32_t*  w,
    const device T_scale*   scales,
    const device T_scale*   biases,
    const device T_act*     x,
    device T_act*           y,
    threadgroup T_act*      Ws,
    const int K,
    const int N,
    const int M,
    uint  simd_gid,
    uint  simd_lid,
    uint3 tgid)
{
    static_assert(bits == 4 || bits == 8,
                  "qmm_t_nax_impl: bits in {4, 8} (8-bit = byte-per-element "
                  "dequant in the W-loader; the NAX MMA runs on dequantized "
                  "T_act values either way)");
    static_assert(!nvfp4 || bits == 4, "NVFP4 is 4-bit by definition");
    // Affine NAX requires gs ∈ {64,128} (gs ≥ BK=64, one scale per
    // thread). NVFP4 is always gs=16 (< BK), handled by the per-byte
    // general scale index in the W-loader below.
    static_assert((!nvfp4 && (group_size == 64 || group_size == 128)) ||
                      (nvfp4 && group_size == 16),
                  "qmm_t_nax_impl: affine gs in {64,128}, NVFP4 gs=16");

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;   // K-reduction tile; matches MLX production
    constexpr int WM = 2;
    constexpr int WN = 2;
    constexpr int TGP = WM * WN * NAX_SIMD_SIZE;  // 128

    // Per-simdgroup tile extents
    constexpr int SM = BM / WM;   // 32
    constexpr int SN = BN / WN;   // 32
    constexpr int SK = 32;
    constexpr int TM = SM / 16;   // 2
    constexpr int TN = SN / 16;   // 2
    constexpr int TK = SK / 16;   // 2

    // BK_padded: pad for bank-conflict avoidance (BK + 16/sizeof(T) = 64+8=72)
    constexpr int BK_padded = BK + 16 / int(sizeof(T_act));

    // bits=4: pack_factor=2 (2 int4 per byte); bits=8: pack_factor=1
    // (1 int8 per byte). bytes_per_pack=1 either way.
    constexpr int pack_factor    = (bits == 4) ? 2 : 1;
    constexpr int bytes_per_pack = 1;

    // W-loader tile layout
    //   bits=4: BCOLS_PACKED = 64/2 = 32, N_READS = (32×64)/128 = 16
    //   bits=8: BCOLS_PACKED = 64/1 = 64, N_READS = (64×64)/128 = 32
    // Per-thread dequant element count is N_READS × pack_factor = 32
    // in both cases; only the byte count read per thread doubles for
    // bits=8 (8-bit weights are inherently 2× the bytes).
    constexpr int BCOLS_PACKED = BK / pack_factor;
    constexpr int N_READS      = (BCOLS_PACKED * BN) / TGP;

    // ── Per-simdgroup M/N offsets within the 64×64 tile ────────────
    //   simd_gid 0 → tm=0,  tn=0
    //   simd_gid 1 → tm=0,  tn=32
    //   simd_gid 2 → tm=32, tn=0
    //   simd_gid 3 → tm=32, tn=32
    const int tm = SM * int(simd_gid / WN);
    const int tn = SN * int(simd_gid % WN);

    // ── Tile origin (grid level) ────────────────────────────────────
    const int y_row = int(tgid.y) * BM;
    const int y_col = int(tgid.x) * BN;
    if (y_row >= M || y_col >= N) return;

    // ── Valid sub-tile extents for M/N tail handling ────────────────
    const short sgp_sm = min(short(SM), short(M - (y_row + tm)));
    const bool is_unaligned_sm = (sgp_sm != short(SM));

    const short tgp_bn = aligned_N ? short(BN) : short(min(BN, N - y_col));
    const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != short(BN));
    const short sgp_sn = aligned_N ? short(SN)
                                   : short(min(int(SN), N - (y_col + tn)));

    // ── Flat thread index for the W-loader layout ───────────────────
    const uint thread_idx = simd_gid * NAX_SIMD_SIZE + simd_lid;

    // W-loader per-thread coords:
    //   bi_w: row in [0, BN) of the BN×BK tile
    //   bj_w: packed-column in [0, BCOLS_PACKED=32) → values {0, 16}
    const uint bi_w = (uint(N_READS) * thread_idx) / uint(BCOLS_PACKED);
    const uint bj_w = (uint(N_READS) * thread_idx) % uint(BCOLS_PACKED);

    // ── Block-level base pointers ───────────────────────────────────
    const int K_w = K * bytes_per_pack / pack_factor;  // K/2 packed bytes per W row
    const int K_g = K / group_size;                    // scale groups per W row

    const device uint8_t* wl_base = (const device uint8_t*)w;
    const device T_act* x_block   = x + int64_t(y_row) * K;
    const device uint8_t* w_block = wl_base + y_col * K_w;
    const device T_scale* s_block = scales  + y_col * K_g;
    const device T_scale* b_block = biases  + y_col * K_g;
    device T_act* y_block         = y + int64_t(y_row) * N + y_col;

    // ── Per-thread W destination in Ws, source in device W ─────────
    threadgroup T_act* Ws_dst = Ws + bi_w * BK_padded + bj_w * pack_factor;
    const device uint8_t* W_src =
        w_block + bi_w * K_w + bj_w * bytes_per_pack;

    // Per-simdgroup X pointer: x_block points to top of the M-tile;
    // shift by tm rows for this simdgroup.
    const device T_act* x_ptr = x_block + int64_t(tm) * K;

    // ── Float accumulator (NAX register tiles) ──────────────────────
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();

    // For the affine path we run MLX's exact `QuantizedBlockLoader` over
    // the K-loop (the hand-inlined W-loader below was numerically
    // identical but ~1.33x slower at every M=4096 prefill shape — the
    // struct loader gives the MPP `matmul2d` scheduler the load pattern
    // it wants). NVFP4 (gs=16 < BK=64) can't use the loader (its
    // `BCOLS<=group_size` invariant) so it keeps the inlined path.
    using loader_w_t = mlx::steel::QuantizedBlockLoader<
        T_act, T_scale, BN, BK, BK_padded, /*reduction_dim=*/1,
        TGP, group_size, bits>;

    // ── Outer K loop + M/N alignment dispatch ───────────────────────
    dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {

          if constexpr (!nvfp4) {
            // ── Affine: MLX QuantizedBlockLoader-driven K loop ───────
            loader_w_t loader_w(
                w_block, s_block, b_block, K, Ws, simd_gid, simd_lid);

            for (int k = 0; k < K; k += BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if constexpr (kAlignedN.value) {
                    loader_w.load_unsafe();
                } else {
                    loader_w.load_safe(short2(BK, tgp_bn));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                MLX_MTL_PRAGMA_NO_UNROLL
                for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                    NAXTile<T_act, TM, TK> Atile;
                    NAXTile<T_act, TN, TK> Btile;
                    volatile int compiler_barrier;

                    if constexpr (kAlignedM.value) {
                        Atile.load(x_ptr + kk1, K);
                    } else {
                        Atile.load_safe(x_ptr + kk1, K, short2(SK, sgp_sm));
                    }
                    Btile.template load<T_act, BK_padded, 1>(
                        Ws + tn * BK_padded + kk1);
                    tile_matmad_nax(
                        Dtile,
                        Atile, metal::bool_constant<false>{},
                        Btile, metal::bool_constant<true>{});
                    (void)compiler_barrier;
                }

                x_ptr += BK;
                loader_w.next();
            }
          } else {
            // ── NVFP4: gs=16 < BK, inlined per-byte scale index ──────
            for (int k = 0; k < K; k += BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if constexpr (kAlignedN.value) {
                    for (int i = 0; i < N_READS; ++i) {
                        uint8_t b = W_src[i * bytes_per_pack];
                        int g = (k + (int(bj_w) + i) * pack_factor) / group_size;
                        T_act scale = T_act(s_block[int(bi_w) * K_g + g]);
                        Ws_dst[i * pack_factor + 0] =
                            scale * T_act(nvfp4_decode(b & 0x0fu));
                        Ws_dst[i * pack_factor + 1] =
                            scale * T_act(nvfp4_decode((uint(b) >> 4) & 0x0fu));
                    }
                } else {
                    if (bi_w < uint(tgp_bn)) {
                        for (int i = 0; i < N_READS; ++i) {
                            uint8_t b = W_src[i * bytes_per_pack];
                            int g = (k + (int(bj_w) + i) * pack_factor) / group_size;
                            T_act scale = T_act(s_block[int(bi_w) * K_g + g]);
                            Ws_dst[i * pack_factor + 0] =
                                scale * T_act(nvfp4_decode(b & 0x0fu));
                            Ws_dst[i * pack_factor + 1] =
                                scale * T_act(nvfp4_decode((uint(b) >> 4) & 0x0fu));
                        }
                    } else {
                        for (int i = 0; i < N_READS * pack_factor; ++i) {
                            Ws_dst[i] = T_act(0);
                        }
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                MLX_MTL_PRAGMA_NO_UNROLL
                for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                    NAXTile<T_act, TM, TK> Atile;
                    NAXTile<T_act, TN, TK> Btile;
                    volatile int compiler_barrier;

                    if constexpr (kAlignedM.value) {
                        Atile.load(x_ptr + kk1, K);
                    } else {
                        Atile.load_safe(x_ptr + kk1, K, short2(SK, sgp_sm));
                    }
                    Btile.template load<T_act, BK_padded, 1>(
                        Ws + tn * BK_padded + kk1);
                    tile_matmad_nax(
                        Dtile,
                        Atile, metal::bool_constant<false>{},
                        Btile, metal::bool_constant<true>{});
                    (void)compiler_barrier;
                }

                x_ptr += BK;
                W_src += BCOLS_PACKED * bytes_per_pack;  // 32 bytes
            }
          }

            // ── Epilogue: store float accumulator to device ──────────
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if constexpr (kAlignedM.value && kAlignedN.value) {
                Dtile.store(y_block + tm * N + tn, N);
            } else if (kAlignedM.value && sgp_sn == short(SN)) {
                Dtile.store(y_block + tm * N + tn, N);
            } else {
                Dtile.store_safe(y_block + tm * N + tn, N, short2(sgp_sn, sgp_sm));
            }

        });
    });
}

// ─────────────────────────────────────────────────────────────────
// affine_qmm_t_nax kernel wrapper — batched=0 only
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, bool aligned_N>
[[kernel]] void affine_qmm_t_nax_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    // K, N, M ride as file-scope function constants QMM_K/N/M so this
    // kernel is recordable into an MTLIndirectComputeCommand.
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BK_padded = BK + 16 / int(sizeof(T_act));  // 72
    threadgroup T_act Ws[BN * BK_padded];  // 64 × 72 = 4608 elements
    qmm_t_nax_impl<T_act, T_scale, group_size, bits, aligned_N>(
        w, scales, biases, x, y, Ws,
        QMM_K, QMM_N, QMM_M,
        simd_gid, simd_lid, tgid);
}

// ─────────────────────────────────────────────────────────────────
// affine_gather_qmm_t_nax kernel — the MoE grouped expert GEMM on the
// M5 matrix accelerator. Identical to affine_qmm_t_nax_kernel except
// the per-expert weight slab is selected by `indices[tile_row]` (the
// host pads each expert's run to a multiple of BM=64 so every 64-row
// output tile maps to one expert; trailing sentinel tiles skip). This
// replaces the ~5-10x slower simdgroup steel grouped GEMM
// (affine_gather_qmm_t) for the MoE prefill on M5+/A19+.
// ─────────────────────────────────────────────────────────────────

template <typename T_act, typename T_scale, int group_size, int bits, bool aligned_N>
[[kernel]] void affine_gather_qmm_t_nax_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    const device uint32_t*  indices  [[buffer(5)]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BM = 64;  // NAX m-tile (qmm_t_nax grid = ceil(M/64))
    constexpr int BK_padded = BK + 16 / int(sizeof(T_act));

    // bits=4: 8 nibbles per uint32 → K_w (packed bytes/row) = K/2,
    // K_g (groups/row) = K/group_size. Per-expert weight slab =
    // [N rows × K_w bytes]; scales/biases = [N rows × K_g].
    const uint expert = indices[tgid.y * uint(BM)];
    if (expert >= uint(QMM_NUM_EXPERTS)) return;  // sentinel tile
    const int K_w = (QMM_K * bits) / 8;       // bits=4 → K/2 bytes, bits=8 → K bytes
    const int K_g = QMM_K / group_size;
    const size_t w_stride = size_t(QMM_N) * size_t(K_w);
    const size_t s_stride = size_t(QMM_N) * size_t(K_g);
    const device uint32_t* w_e = (const device uint32_t*)(
        (const device uint8_t*)w + size_t(expert) * w_stride);
    const device T_scale* s_e = scales + size_t(expert) * s_stride;
    const device T_scale* b_e = biases + size_t(expert) * s_stride;

    threadgroup T_act Ws[BN * BK_padded];
    qmm_t_nax_impl<T_act, T_scale, group_size, bits, aligned_N>(
        w_e, s_e, b_e, x, y, Ws,
        QMM_K, QMM_N, QMM_M,
        simd_gid, simd_lid, tgid);
}

#define INST_GATHER_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, bits, aln_tag, aln_val) \
    template [[host_name(                                                                \
        "affine_gather_qmm_t_nax_" #act_tag "_s_" #scale_tag "_gs_" #gs                \
        "_b_" #bits "_alN_" #aln_tag "_batch_0")]] [[kernel]] void                      \
    affine_gather_qmm_t_nax_kernel<act_type, scale_type, gs, bits, aln_val>(           \
        const device uint32_t*   w        [[buffer(0)]],                                \
        const device scale_type* scales   [[buffer(1)]],                                \
        const device scale_type* biases   [[buffer(2)]],                                \
        const device act_type*   x        [[buffer(3)]],                                \
        device act_type*         y        [[buffer(4)]],                                \
        const device uint32_t*   indices  [[buffer(5)]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                              \
        uint  simd_lid [[thread_index_in_simdgroup]],                                   \
        uint3 tgid     [[threadgroup_position_in_grid]]);

// ─────────────────────────────────────────────────────────────────
// Instantiations — one symbol per (dtype, group_size, aligned_N).
// Naming mirrors the non-NAX pattern with "nax" inserted after
// "qmm_t_":
//   affine_qmm_t_nax_<dtype>_s_<scale_dtype>_gs_<gs>_b_<bits>_alN_<bool>_batch_0
//
// gs=32 deliberately not instantiated — the dispatcher routes gs=32
// to Standard qmm_t (BK=64 violates QuantizedBlockLoader's
// `BCOLS<=group_size` for gs=32; the gs=32 specialized loader has
// different scale-indexing semantics and would need a separate path).
// ─────────────────────────────────────────────────────────────────

#define INST_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, bits, aln_tag, aln_val) \
    template [[host_name(                                                                \
        "affine_qmm_t_nax_" #act_tag "_s_" #scale_tag "_gs_" #gs                       \
        "_b_" #bits "_alN_" #aln_tag "_batch_0")]] [[kernel]] void                      \
    affine_qmm_t_nax_kernel<act_type, scale_type, gs, bits, aln_val>(                  \
        const device uint32_t*   w        [[buffer(0)]],                                \
        const device scale_type* scales   [[buffer(1)]],                                \
        const device scale_type* biases   [[buffer(2)]],                                \
        const device act_type*   x        [[buffer(3)]],                                \
        device act_type*         y        [[buffer(4)]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                              \
        uint  simd_lid [[thread_index_in_simdgroup]],                                   \
        uint3 tgid     [[threadgroup_position_in_grid]]);

#define INST_QMM_T_NAX_ALL(act_tag, act_type, scale_tag, scale_type, gs, bits) \
    INST_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, bits, true,  true)  \
    INST_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, bits, false, false)

// T_scale is purely the dequant-read type for the per-group scales /
// biases (`w = q*scale + bias`); it does not touch the NAX MMA, which
// runs on the dequantized T_act values. So every (T_act, T_scale, gs)
// combo is valid — instantiate both f16- and bf16-scale variants so
// bf16-scale models (e.g. Qwen3) get NAX too, not just f16-scale ones
// (e.g. Llama). The dispatcher picks the symbol by the model's scale
// dtype; a missing instantiation would abort with a nil computeFunction.
INST_QMM_T_NAX_ALL(f16,  half,   f16,  half,    64, 4)
INST_QMM_T_NAX_ALL(f16,  half,   f16,  half,   128, 4)
INST_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,    64, 4)
INST_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,   128, 4)
INST_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat,  64, 4)
INST_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat, 128, 4)
INST_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat,  64, 4)
INST_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat, 128, 4)

// MoE grouped expert GEMM (NAX) — 4-bit only (gemma4 experts). Same
// dtype/gs matrix as the b4 dense set above.
#define INST_GATHER_QMM_T_NAX_ALL(act_tag, act_type, scale_tag, scale_type, gs, bits) \
    INST_GATHER_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, bits, true,  true)  \
    INST_GATHER_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, bits, false, false)
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   f16,  half,    64, 4)
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   f16,  half,   128, 4)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,    64, 4)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,   128, 4)
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat,  64, 4)
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat, 128, 4)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat,  64, 4)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat, 128, 4)

// MoE grouped expert GEMM (NAX) — 8-bit rows for MLX-native mixed/dynamic
// quant (OptiQ) whose sensitive edge layers ship 8-bit switch_glu experts.
// Same dtype/gs matrix as the b4 gather set above so no (model-dtype, gs)
// combo hits a nil computeFunction.
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   f16,  half,    64, 8)
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   f16,  half,   128, 8)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,    64, 8)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,   128, 8)
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat,  64, 8)
INST_GATHER_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat, 128, 8)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat,  64, 8)
INST_GATHER_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat, 128, 8)

// 8-bit rows (Gemma4 mlp.{gate,up,down} at gs=64 bf16×bf16-scale in
// production; f16×f16 for the parity tests; gs=128 + cross-dtype rows
// for symmetry with the b4 set so no (model-dtype, gs) combo can hit
// a nil computeFunction).
INST_QMM_T_NAX_ALL(f16,  half,   f16,  half,    64, 8)
INST_QMM_T_NAX_ALL(f16,  half,   f16,  half,   128, 8)
INST_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,    64, 8)
INST_QMM_T_NAX_ALL(bf16, bfloat, f16,  half,   128, 8)
INST_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat,  64, 8)
INST_QMM_T_NAX_ALL(f16,  half,   bf16, bfloat, 128, 8)
INST_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat,  64, 8)
INST_QMM_T_NAX_ALL(bf16, bfloat, bf16, bfloat, 128, 8)

// ─────────────────────────────────────────────────────────────────
// Buffer-dims variant: K/N/M as `const constant int&` buffer args
// (5/6/7) exactly like MLX — NO function constants, so the pipeline
// is built from the unspecialized function. The fn-const variants
// above force a specialization recompile at pipeline creation; this
// entry exists to A/B whether that specialization pass de-optimizes
// MPP cooperative-tensor code (the fn-consts were for ICB
// recordability, which was removed).
// ─────────────────────────────────────────────────────────────────
template <typename T_act, typename T_scale, int group_size, int bits, bool aligned_N>
[[kernel]] void affine_qmm_t_nax_dims_kernel(
    const device uint32_t*  w        [[buffer(0)]],
    const device T_scale*   scales   [[buffer(1)]],
    const device T_scale*   biases   [[buffer(2)]],
    const device T_act*     x        [[buffer(3)]],
    device T_act*           y        [[buffer(4)]],
    const constant int&     K        [[buffer(5)]],
    const constant int&     N        [[buffer(6)]],
    const constant int&     M        [[buffer(7)]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BK_padded = BK + 16 / int(sizeof(T_act));
    threadgroup T_act Ws[BN * BK_padded];
    qmm_t_nax_impl<T_act, T_scale, group_size, bits, aligned_N>(
        w, scales, biases, x, y, Ws,
        K, N, M,
        simd_gid, simd_lid, tgid);
}

#define INST_QMM_T_NAX_DIMS(act_tag, act_type, scale_tag, scale_type, gs, bits, aln_tag, aln_val) \
    template [[host_name(                                                                \
        "affine_qmm_t_nax_dims_" #act_tag "_s_" #scale_tag "_gs_" #gs                  \
        "_b_" #bits "_alN_" #aln_tag "_batch_0")]] [[kernel]] void                      \
    affine_qmm_t_nax_dims_kernel<act_type, scale_type, gs, bits, aln_val>(             \
        const device uint32_t*   w        [[buffer(0)]],                                \
        const device scale_type* scales   [[buffer(1)]],                                \
        const device scale_type* biases   [[buffer(2)]],                                \
        const device act_type*   x        [[buffer(3)]],                                \
        device act_type*         y        [[buffer(4)]],                                \
        const constant int&      K        [[buffer(5)]],                                \
        const constant int&      N        [[buffer(6)]],                                \
        const constant int&      M        [[buffer(7)]],                                \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                              \
        uint  simd_lid [[thread_index_in_simdgroup]],                                   \
        uint3 tgid     [[threadgroup_position_in_grid]]);

INST_QMM_T_NAX_DIMS(bf16, bfloat, bf16, bfloat, 64, 8, true, true)
INST_QMM_T_NAX_DIMS(bf16, bfloat, bf16, bfloat, 64, 4, true, true)

// ─────────────────────────────────────────────────────────────────
// nvfp4_qmm_t_nax — NVFP4 NAX prefill matmul
//
// Reuses qmm_t_nax_impl with the `nvfp4` flag: E2M1 LUT decode, no
// per-group bias, and a per-byte scale index (gs=16 < BK=64, so a
// thread's W read spans multiple scale groups). 4-buffer layout (no
// biases) — `scales` passed as the ignored bias pointer. T_scale is
// always half (folded F16). gs=16 only.
// ─────────────────────────────────────────────────────────────────
template <typename T_act, typename T_scale, int group_size, int bits, bool aligned_N>
[[kernel]] void nvfp4_qmm_t_nax_kernel(
    const device uint32_t*  w       [[buffer(0)]],
    const device T_scale*   scales  [[buffer(1)]],
    // biases [[buffer(2)]] = scales (dummy). 5-buffer layout MUST match
    // affine's; MTL4 mis-dispatches the 4-buffer variant — see note in
    // quantized_qmv.metal.
    const device T_scale*   biases  [[buffer(2)]],
    const device T_act*     x       [[buffer(3)]],
    device T_act*           y       [[buffer(4)]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BK_padded = BK + 16 / int(sizeof(T_act));
    threadgroup T_act Ws[BN * BK_padded];
    qmm_t_nax_impl<T_act, T_scale, group_size, bits, aligned_N, /*nvfp4=*/true>(
        w, scales, biases, x, y, Ws,
        QMM_K, QMM_N, QMM_M,
        simd_gid, simd_lid, tgid);
}

#define INST_NVFP4_QMM_T_NAX(act_tag, act_type, scale_tag, scale_type, gs, aln_tag, aln_val) \
    template [[host_name(                                                                     \
        "nvfp4_qmm_t_nax_" #act_tag "_s_" #scale_tag "_gs_" #gs                                \
        "_b_4_alN_" #aln_tag "_batch_0")]] [[kernel]] void                                     \
    nvfp4_qmm_t_nax_kernel<act_type, scale_type, gs, 4, aln_val>(                              \
        const device uint32_t*   w       [[buffer(0)]],                                       \
        const device scale_type* scales  [[buffer(1)]],                                       \
        const device scale_type* biases  [[buffer(2)]],                                       \
        const device act_type*   x       [[buffer(3)]],                                       \
        device act_type*         y       [[buffer(4)]],                                        \
        uint  simd_gid [[simdgroup_index_in_threadgroup]],                                    \
        uint  simd_lid [[thread_index_in_simdgroup]],                                         \
        uint3 tgid     [[threadgroup_position_in_grid]]);

// gs=16 only for NVFP4; T_scale always half (folded F16).
INST_NVFP4_QMM_T_NAX(f16, half, f16, half, 16, true, true)
INST_NVFP4_QMM_T_NAX(f16, half, f16, half, 16, false, false)
INST_NVFP4_QMM_T_NAX(bf16, bfloat, f16, half, 16, true, true)
INST_NVFP4_QMM_T_NAX(bf16, bfloat, f16, half, 16, false, false)

// ─────────────────────────────────────────────────────────────────
// PLAIN bf16 NAX GEMM: y[M,N] = x[M,K] @ w[N,K]^T  (no quantization).
// For the gemma4 hd512 unfused-attention QK^T / PV matmuls. Identical
// to qmm_t_nax_impl except the int4 QuantizedBlockLoader (dequant W ->
// Ws) is replaced by a plain mlx::steel::BlockLoader (copy bf16 B ->
// Ws). Same NAXTile MXU matmul + store. K/N/M ride QMM_K/N/M consts so
// it records into an ICB exactly like affine_qmm_t_nax.
// ─────────────────────────────────────────────────────────────────
template <typename T>
METAL_FUNC void gemm_t_nax_impl(
    const device T* x,    // [M, K]
    const device T* w,    // [N, K]
    device T*       y,    // [M, N]
    threadgroup T*  Ws,   // [BN, BK_padded]
    const int K, const int N, const int M,
    const int w_ld,       // w (B) ROW stride; == K for dense/QKᵀ, but for PV the
                          // V^T dense buffer has static stride max_kv while the
                          // contraction K = kv_len — they MUST be decoupled.
    uint simd_gid, uint simd_lid, uint3 tgid,
    // Contraction loop bounds [k_lo, k_hi). For dense/QKᵀ this is [0, K] (full);
    // for the block-diagonal PV it is [span_lo, causal] so the dropped band is
    // never contracted (real FLOP cut). k_lo is BK-aligned (== span block size).
    int k_lo, int k_hi)
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WM = 2;
    constexpr int WN = 2;
    constexpr int SM = BM / WM;   // 32
    constexpr int SN = BN / WN;   // 32
    constexpr int SK = 32;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int TK = 2;
    constexpr int TGP_NAX_GEMM = WM * WN * 32;            // 128 (4 simdgroups)
    constexpr int BK_padded = BK + 16 / int(sizeof(T));  // 72 (bf16)

    const int tm = SM * int(simd_gid / WN);
    const int tn = SN * int(simd_gid % WN);

    const int y_row = int(tgid.y) * BM;
    const int y_col = int(tgid.x) * BN;
    if (y_row >= M || y_col >= N) return;

    const short sgp_sm = min(short(SM), short(M - (y_row + tm)));
    const bool  is_unaligned_sm = (sgp_sm != short(SM));
    const short tgp_bn = short(min(BN, N - y_col));
    const bool  is_unaligned_bn = (tgp_bn != short(BN));
    const short sgp_sn = short(min(int(SN), N - (y_col + tn)));

    const device T* x_block = x + int64_t(y_row) * K;
    const device T* w_block = w + int64_t(y_col) * w_ld;
    device T*       y_block = y + int64_t(y_row) * N + y_col;
    // Start the contraction at k_lo (BK-aligned): skip x rows + V rows before it.
    const device T* x_ptr   = x_block + int64_t(tm) * K + k_lo;

    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();

    using loader_b_t = mlx::steel::BlockLoader<T, BN, BK, BK_padded, 1, TGP_NAX_GEMM>;
    loader_b_t loader_b(w_block + k_lo, w_ld, Ws, simd_gid, simd_lid);

    dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(!is_unaligned_bn, [&](auto kAlignedN) {
            for (int k = k_lo; k < k_hi; k += BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if constexpr (kAlignedN.value) {
                    loader_b.load_unsafe();
                } else {
                    loader_b.load_safe(short2(BK, tgp_bn));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                MLX_MTL_PRAGMA_NO_UNROLL
                for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                    NAXTile<T, TM, TK> Atile;
                    NAXTile<T, TN, TK> Btile;
                    if constexpr (kAlignedM.value) {
                        Atile.load(x_ptr + kk1, K);
                    } else {
                        Atile.load_safe(x_ptr + kk1, K, short2(SK, sgp_sm));
                    }
                    Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
                    tile_matmad_nax(
                        Dtile,
                        Atile, metal::bool_constant<false>{},
                        Btile, metal::bool_constant<true>{});
                }
                x_ptr += BK;
                loader_b.next();
            }

            if (kAlignedM.value && sgp_sn == short(SN)) {
                Dtile.store(y_block + tm * N + tn, N);
            } else {
                Dtile.store_safe(y_block + tm * N + tn, N, short2(sgp_sn, sgp_sm));
            }
        });
    });
}

[[kernel, max_total_threads_per_threadgroup(128)]] void gemm_nax_bf16(
    const device bfloat* x   [[buffer(0)]],   // A [M, K]
    const device bfloat* w   [[buffer(1)]],   // B [N, K]
    device bfloat*       y   [[buffer(2)]],   // C [M, N] = A @ B^T
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BK_padded = BK + 16 / int(sizeof(bfloat));
    threadgroup bfloat Ws[BN * BK_padded];
    // dense: w row stride == contraction == QMM_K. Full contraction [0, QMM_K].
    gemm_t_nax_impl<bfloat>(x, w, y, Ws, QMM_K, QMM_N, QMM_M, QMM_K, simd_gid, simd_lid, tgid,
                            0, QMM_K);
}

// Dynamic-kv_len GEMM wrappers for hd512 unfused attention. kv_len grows per
// forward but the tape bakes dims, so the changing dim is read from
// seq_used[0] at runtime (grid baked at the max; tiles with y_col>=N or the
// K-loop past K early-return / stop). QK^T: N=kv_len. PV: K=kv_len.
[[kernel, max_total_threads_per_threadgroup(128)]] void gemm_nax_bf16_qk(
    const device bfloat* x        [[buffer(0)]],   // Q head [M=Lq, K=hd]
    const device bfloat* w        [[buffer(1)]],   // K dense [N=kv_len, K=hd]
    device bfloat*       y        [[buffer(2)]],   // scores [M=Lq, N=kv_len]
    const device uint*   seq_used     [[buffer(3)]],   // [1] kv_len
    const device uint*   span_ids     [[buffer(4)]],   // per-block span label; 0 = no span
    const device uint*   cu_seqlens_q [[buffer(5)]],   // new-query span (ACTUAL, not bucket)
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BK_padded = BK + 16 / int(sizeof(bfloat));
    // Block-diagonal span bound: a span-uniform query-tile attends only its own
    // span's keys [span_lo, causal]. Skip both bands before the contraction
    // (real FLOP cut). span_ids 0 / boundary-straddling tile → full causal range.
    if (QK_SPAN_BLOCK_NAX != 0u) {
        const uint N = uint(seq_used[0]);
        const uint c_row = tgid.y * uint(BM);
        const uint c_col = tgid.x * uint(BN);
        // Causal offset from the ACTUAL new-query count (cu_seqlens), NOT QMM_M
        // (the over-sized prefill bucket) — N - QMM_M underflows when bucket>kv_len.
        const uint lq_actual = cu_seqlens_q[1] - cu_seqlens_q[0];
        const uint causal_off = N - lq_actual;
        const uint sf = span_ids[causal_off + c_row]; // per-TOKEN index
        const uint sl = span_ids[causal_off + c_row + uint(BM) - 1u];
        if (sf != 0u && sf == sl) {
            if (c_col >= causal_off + c_row + uint(BM)) {
                return;  // causal upper
            }
            const uint span_lo = sf - 1u; // label-1 IS the span's first TOKEN (exact)
            if (c_col + uint(BN) <= span_lo) {
                return;  // span lower
            }
        }
    }
    threadgroup bfloat Ws[BN * BK_padded];
    // QK^T: N=kv_len dynamic; K dense (kdense) row stride == hd == QMM_K. The
    // span bound is on the OUTPUT key-tile (above), not the hd contraction → full
    // contraction [0, QMM_K].
    gemm_t_nax_impl<bfloat>(x, w, y, Ws, QMM_K, int(seq_used[0]), QMM_M, QMM_K,
                            simd_gid, simd_lid, tgid, 0, QMM_K);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void gemm_nax_bf16_pv(
    const device bfloat* x        [[buffer(0)]],   // probs [M=Lq, K=kv_len]
    const device bfloat* w        [[buffer(1)]],   // V^T dense [N=hd, rows strided by QMM_K=max_kv]
    device bfloat*       y        [[buffer(2)]],   // out head [M=Lq, N=hd]
    const device uint*   seq_used     [[buffer(3)]],   // [1] kv_len
    const device uint*   span_ids     [[buffer(4)]],   // per-block span label; 0 = no span
    const device uint*   cu_seqlens_q [[buffer(5)]],   // new-query span (ACTUAL, not bucket)
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]])
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int BK_padded = BK + 16 / int(sizeof(bfloat));
    const int K = int(seq_used[0]);  // contraction = kv_len
    // Block-diagonal PV: bound the contraction to the query-tile's span
    // [span_lo, causal] — the probs are zero outside it (the softmax span-lower
    // zeroed them), so the dropped band is pure FLOP waste. Mirrors the QKᵀ bound.
    int k_lo = 0, k_hi = K;
    if (QK_SPAN_BLOCK_NAX != 0u) {
        const uint c_row = tgid.y * uint(BM);  // query-tile (M dim)
        const uint lq_actual = cu_seqlens_q[1] - cu_seqlens_q[0];
        const uint causal_off = uint(K) - lq_actual;
        const uint sf = span_ids[causal_off + c_row]; // per-TOKEN index
        const uint sl = span_ids[causal_off + c_row + uint(BM) - 1u];
        if (sf != 0u && sf == sl) {
            k_lo = int(sf - 1u); // label-1 IS the span's first TOKEN (exact)
            k_hi = int(min(uint(K), causal_off + c_row + uint(BM)));
        }
    }
    threadgroup bfloat Ws[BN * BK_padded];
    // PV: contraction K=kv_len (seq_used), but V^T dense rows are strided by the
    // STATIC max_kv (= QMM_K, the gather's GDK_MAX_KV) — decoupled via w_ld.
    gemm_t_nax_impl<bfloat>(x, w, y, Ws, K, QMM_N, QMM_M, QMM_K,
                            simd_gid, simd_lid, tgid, k_lo, k_hi);
}

// ─────────────────────────────────────────────────────────────────
// Small-M (decode-batch) MLX-affine 4-bit GEMM on the matrix unit:
// `y = x · Wᵀ`, W = s·q + b per group of G along K. The 4-bit codes are
// the `uint4b_format` matmul2d operand as stored — no dequant stage — and
// one threadgroup's TM rows cover the whole batch (up to TM), so each
// weight is read once per TM rows instead of once per row (qmv) or as a
// mostly-padding 64-row tile (qmm_t NAX).
//
//   y[m,n] = Σ_g s[n,g]·(x_g·q_g[n]) + b[n,g]·xs[m,g],  xs[m,g] = Σ_{k∈g} x[m,k]
//
// The group sums `xs` of the tile's rows are staged in threadgroup memory
// SMALL_M_XS_GROUPS groups at a time — a small footprint, so it doesn't cap
// how many threadgroups a core holds. Buffers as
// `affine_qmm_t_nax`: w[0], scales[1], biases[2], x[3], y[4]; K/N/M from
// function constants 0/1/2. N % TN == 0 (static N slices skip bounds
// checks — the lowering checks it); M rows are bounds-checked
// (`dynamic_extent`). Grid: (N/TN, ceil(M/TM)), 32·NSG threads.
// ─────────────────────────────────────────────────────────────────

MLX_MTL_CONST int SMALL_M_XS_GROUPS = 32;

template <typename T, typename S, int G, int TM, int TN, int NSG>
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void affine_qmm_small_m_kernel(
    const device uint32_t* w      [[buffer(0)]],
    const device S*        scales [[buffer(1)]],
    const device S*        biases [[buffer(2)]],
    const device T*        x      [[buffer(3)]],
    device T*              y      [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    using namespace mpp::tensor_ops;
    using Ext = dextents<int32_t, 2>;
    const int K = QMM_K, N = QMM_N, M = QMM_M, KG = K / G;
    const int col0 = int(tgid.x) * TN;
    const int row0 = int(tgid.y) * TM;
    const int rows = min(TM, M - row0);

    using XA = tensor<device T, Ext, tensor_inline>;
    using XB = tensor<device uint4b_format, Ext, tensor_inline>;
    XA A((device T*)x, Ext(K, M));
    XB B((typename XB::data_handle_type)w, Ext(K, N));
    tensor<device T, Ext, tensor_inline> C(y, Ext(N, M));
    constexpr auto desc =
        matmul2d_descriptor(TM, TN, G, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto sA0 = A.template slice<G, dynamic_extent>(0, row0);
    auto sB0 = B.template slice<G, TN>(0, col0);
    auto part = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    auto acc = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    auto out = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), T>();
    _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    }
    threadgroup float xs[TM * SMALL_M_XS_GROUPS];
    for (int g0 = 0; g0 < KG; g0 += SMALL_M_XS_GROUPS) {
        const int chunk = min(SMALL_M_XS_GROUPS, KG - g0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < rows * chunk; i += NSG * 32) {
            const int m = i / chunk, gg = i % chunk;
            const device T* xg = x + (row0 + m) * K + (g0 + gg) * G;
            float s = 0.0f;
            for (int k = 0; k < G; ++k) {
                s += float(xg[k]);
            }
            xs[m * SMALL_M_XS_GROUPS + gg] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int gg = 0; gg < chunk; ++gg) {
            const int g = g0 + gg;
            auto sA = A.template slice<G, dynamic_extent>(g * G, row0);
            auto sB = B.template slice<G, TN>(g * G, col0);
            op.run(sA, sB, part);
            _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < part.get_capacity(); ++i) {
                if (part.is_valid_element(i)) {
                    const auto idx = part.get_multidimensional_index(i);
                    const int n = col0 + idx[0], m = idx[1];
                    if (m < rows) {
                        acc[i] = fma(part[i], float(scales[n * KG + g]),
                                     fma(float(biases[n * KG + g]),
                                         xs[m * SMALL_M_XS_GROUPS + gg], acc[i]));
                    }
                }
            }
        }
    }
    _Pragma("clang loop unroll(full)") for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) out[i] = T(acc[i]);
    }
    auto mC = C.template slice<TN, dynamic_extent>(col0, row0);
    out.store(mC);
}

#define INST_QMM_SMALL_M(act_tag, act_type, scale_tag, scale_type, gs, tm, tn, nsg)          \
    template [[host_name(                                                                 \
        "affine_qmm_small_m_" #act_tag "_s_" #scale_tag "_gs_" #gs "_b_4_tm_" #tm "_tn_" #tn \
        "_nsg_" #nsg)]] [[kernel]] void                                                    \
    affine_qmm_small_m_kernel<act_type, scale_type, gs, tm, tn, nsg>(                      \
        const device uint32_t*   w      [[buffer(0)]],                                    \
        const device scale_type* scales [[buffer(1)]],                                    \
        const device scale_type* biases [[buffer(2)]],                                    \
        const device act_type*   x      [[buffer(3)]],                                    \
        device act_type*         y      [[buffer(4)]],                                    \
        uint2 tgid [[threadgroup_position_in_grid]],                                      \
        uint  tid  [[thread_index_in_threadgroup]]);

// 16 columns, one simdgroup: the best (or tied) tile at 5–16 rows on the
// Llama-3.2-3B gate/up and down shapes, over 32/1 and 64/2 — the narrow
// tile keeps enough threadgroups in flight on narrow layers.
#define INST_QMM_SMALL_M_TILES(act_tag, act_type, scale_tag, scale_type, gs)             \
    INST_QMM_SMALL_M(act_tag, act_type, scale_tag, scale_type, gs, 8, 16, 1)             \
    INST_QMM_SMALL_M(act_tag, act_type, scale_tag, scale_type, gs, 16, 16, 1)

#define INST_QMM_SMALL_M_GS(act_tag, act_type, scale_tag, scale_type)                    \
    INST_QMM_SMALL_M_TILES(act_tag, act_type, scale_tag, scale_type, 32)                 \
    INST_QMM_SMALL_M_TILES(act_tag, act_type, scale_tag, scale_type, 64)                 \
    INST_QMM_SMALL_M_TILES(act_tag, act_type, scale_tag, scale_type, 128)

INST_QMM_SMALL_M_GS(f16,  half,   f16,  half)
INST_QMM_SMALL_M_GS(f16,  half,   bf16, bfloat)
INST_QMM_SMALL_M_GS(bf16, bfloat, f16,  half)
INST_QMM_SMALL_M_GS(bf16, bfloat, bf16, bfloat)
