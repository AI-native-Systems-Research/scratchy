// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! MLX-affine int4 dequantization dispatcher.
//!
//! Wraps the `affine_dequantize_*_gs_*_b_4` kernels in
//! `shaders/quantized_dequantize.metal` (faithful port of
//! `mlx/backend/metal/kernels/quantized.h:2536`).
//!
//! Mirrors `mlx/backend/metal/quantized.cpp:1657
//! fast::Quantize::eval_gpu`'s dequantize path:
//!
//! ```text
//!   constexpr int simd_size = 32;                       // unused for dequant
//!   int packs_per_int = 8 / bits;                       // 2 for bits=4
//!   size_t nthreads = out.size() / packs_per_int;       // = n_bytes
//!   auto grid_shape = w.shape();
//!   grid_shape.back() *= uint8_per_uint32;              // u32 → bytes
//!   compute_encoder.dispatch_threads(grid_dims, group_dims);
//! ```
//!
//! Bits = 4 only; other bits land alongside the qmv/qmm kernels.

/// Activation / output dtype the affine quant kernels read and write —
/// the kernel template parameter `T_act`.
/// Picks between the `affine_*_f16_s_*_*` and `affine_*_bf16_s_*_*`
/// symbol families. Historical name retained: this was `DequantDtype`
/// pre-P10b, when the `<T>` template covered both activation and scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DequantDtype {
    F16,
    Bf16,
}

impl DequantDtype {
    pub fn symbol_infix(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    pub fn elem_size(self) -> usize {
        2
    }
}

/// `ScaleDtype` relocated to the cfg-free `scratchy-tensors` core so
/// the `scratchy-ir` `CanonicalParams::SCALE_DTYPE` const can name
/// it. Re-exported here so the metal quant kernels (+ the
/// `interpreter::metal::mod` re-export) keep resolving via this path.
pub use scratchy_tensors::ScaleDtype;

// ─────────────────────────────────────────────────────────────────
// Decode-bucket qmv dispatcher — port of `quantized.cpp:1365
// dispatch_qmv` + `:177 qmv_quad` + `:235 qmv` (which itself picks
// qmv_fast vs qmv).
// ─────────────────────────────────────────────────────────────────

/// Picked qmv variant for a given `(M, N, K, bits)` shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QmvKernel {
    /// `affine_qmv_quad_*_d_<d>_*` — K must equal 64 or 128, bits must
    /// be a power of two. Most efficient on tiny K (e.g. head_dim
    /// projections). MLX `qmv_quad`.
    Quad { d: u32 },
    /// `affine_qmv_fast_*` — `N % 8 == 0 && K % 512 == 0`. The decode
    /// hot path for Llama / Qwen / Gemma. MLX `qmv_fast`.
    Fast,
    /// `affine_qmv_*` — generic fallback with bounds-checked tail.
    Generic,
}

/// Pick the right qmv variant per MLX `dispatch_qmv` (`quantized.cpp:1365`)
/// followed by the inner `qmv_fast` vs `qmv` choice (`:259`).
///
/// Mirrors the C++ exactly:
///
/// ```text
/// // dispatch_qmv:
/// if ((K == 128 || K == 64) && is_power_of_2(bits)) → qmv_quad(d=K)
/// else                                              → qmv(...)
/// // qmv():
/// bool fast = N % bn == 0 && K % 512 == 0;  // bn = 8
/// kernel = fast ? "qmv_fast" : "qmv";
/// ```
pub fn pick_qmv_kernel(n: u32, k: u32, bits: u32) -> QmvKernel {
    let pow2_bits = bits != 0 && (bits & (bits - 1)) == 0;
    if (k == 64 || k == 128) && pow2_bits {
        QmvKernel::Quad { d: k }
    } else if n.is_multiple_of(8) && k.is_multiple_of(512) {
        QmvKernel::Fast
    } else {
        QmvKernel::Generic
    }
}

/// Threadgroup grid + threads-per-group for a picked qmv variant.
///
/// `qmv_quad`: `bn = quads_per_simd * results_per_quadgroup = 8 * 8 = 64`
/// (`quantized.cpp:193-198`); group is one simdgroup
/// (`(simdgroup_size=32, 1, 1)`).
///
/// `qmv` / `qmv_fast`: `bn = 8`, `bk = 32`, group `(bk=32, 2, 1)` —
/// 2 simdgroups (`quantized.cpp:251-254`).
pub fn qmv_dispatch_shape(
    kernel: QmvKernel,
    m: u32,
    n: u32,
    b: u32,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    match kernel {
        QmvKernel::Quad { .. } => {
            let bn: u32 = 64;
            ((m, n.div_ceil(bn), b), (32, 1, 1))
        }
        QmvKernel::Fast | QmvKernel::Generic => {
            let bn: u32 = 8;
            ((m, n.div_ceil(bn), b), (32, 2, 1))
        }
    }
}

/// Format the kernel symbol name for a picked qmv variant. Matches the
/// `INST_QMV_*` macros in `shaders/quantized_qmv.metal`.
pub fn qmv_kernel_name(
    kernel: QmvKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
    batched: bool,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    let batch = if batched { 1 } else { 0 };
    match kernel {
        QmvKernel::Quad { d } => {
            format!("affine_qmv_quad_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_d_{d}_batch_{batch}",)
        }
        QmvKernel::Fast => {
            format!("affine_qmv_fast_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_{batch}",)
        }
        QmvKernel::Generic => {
            format!("affine_qmv_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_{batch}",)
        }
    }
}

/// `&'static str` view of [`qmv_kernel_name`] for the lowering pass.
/// `LoweredCommand::function` is `&'static str`, so the macro-emit
/// path can't allocate a `String` here. Covers the non-batched (B=1)
/// instantiations only (`batch_0`); MoE batched=1 lands with P13.
///
/// Composes the `affine_qmv_{quad,fast,generic}_<dtype>_s_<scale>_gs_<gs>_b_4[_d_<D>]_batch_0`
/// symbol from the kernel-variant axes. Returns `&'static str` via a
/// process-lifetime `LazyLock` cache so the same `(kernel, dtype,
/// scale, gs)` key always returns the same pointer (the worker's
/// `SpecializedPipelineCache` uses it as a HashMap key). The cache is
/// the seam where `(F16 × BF16 scale-dtype × {32,64,128} group-size ×
/// 3-kernel-variant)` instantiations get materialized as leaked
/// `&'static str` — no per-call format! and no 144-arm match table.
pub fn qmv_kernel_static_name(
    kernel: QmvKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
) -> &'static str {
    debug_assert!(
        matches!(bits, 4 | 8),
        "qmv_kernel_static_name: only bits 4 and 8 are wired (got {bits})"
    );
    let key = (kernel, dtype, scale_dtype, group_size, bits);
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type NameKey = (QmvKernel, DequantDtype, ScaleDtype, u32, u32);
    type NameCache = std::sync::Mutex<HashMap<NameKey, &'static str>>;
    static CACHE: OnceLock<NameCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("qmv_kernel_static_name cache poisoned");
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    let supported_gs = matches!(group_size, 32 | 64 | 128);
    if !supported_gs {
        panic!(
            "qmv_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    let dtype_s = dtype.symbol_infix();
    let scale_s = scale_dtype.symbol_infix();
    let owned = match kernel {
        QmvKernel::Quad { d } => {
            if !matches!(d, 64 | 128) {
                panic!(
                    "qmv_kernel_static_name: QmvKernel::Quad with unsupported D={d} \
                     — only 64 and 128 instantiated"
                );
            }
            format!("affine_qmv_quad_{dtype_s}_s_{scale_s}_gs_{group_size}_b_{bits}_d_{d}_batch_0")
        }
        QmvKernel::Fast => {
            format!("affine_qmv_fast_{dtype_s}_s_{scale_s}_gs_{group_size}_b_{bits}_batch_0")
        }
        QmvKernel::Generic => {
            format!("affine_qmv_{dtype_s}_s_{scale_s}_gs_{group_size}_b_{bits}_batch_0")
        }
    };
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

// ─────────────────────────────────────────────────────────────────
// `get_qmv_batch_limit` — port of `quantized.cpp:84`. Decides where
// the matvec / matmul boundary sits per arch generation. Used by
// `lower_one` to choose between `Instruction::AffineQmm` (matvec
// branch) and `Instruction::Gemm`-equivalent (matmul branch).
// ─────────────────────────────────────────────────────────────────

/// Vector-vs-matrix limit for a given `(K, N, arch_gen)`. M < limit
/// routes to `qmv*`; M >= limit routes to `qmm*` (P4).
///
/// MLX models the `arch_size` ('d' for desktop variants like M3 Ultra
/// vs. anything else) — scratchy's `MetalTargetProfile` doesn't track
/// that today, so we conservatively use the non-'d' branch (smaller
/// limits, more aggressive matvec routing). This matches MacBook Pro
/// M3/M4 Pro/Max behavior; M3/M4 Ultra would over-route to matvec
/// versus MLX, which is correct (qmv kernels handle small M fine —
/// just slightly less efficient than qmm at the high-M boundary).
pub fn get_qmv_batch_limit(k: u32, n: u32, arch_gen: crate::tape::targets::AppleSiliconGen) -> u32 {
    use crate::tape::targets::AppleSiliconGen as G;
    match arch_gen {
        G::M1 | G::M2 => {
            if k <= 2048 && n <= 2048 {
                14
            } else if k <= 4096 && n <= 4096 {
                10
            } else {
                6
            }
        }
        G::M3 | G::M4 | G::M5 => {
            if k <= 2048 && n <= 2048 {
                18
            } else if k <= 4096 && n <= 4096 {
                12
            } else {
                10
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Prefill-bucket qmm_t dispatcher — port of `quantized.cpp:680 qmm()`
// (transpose=true branch) + `:774 qmm_splitk()`. Used when M >=
// vector_limit (matmul branch in `dispatch_qmv`'s outer
// `QuantizedMatmul::eval_gpu` rule).
// ─────────────────────────────────────────────────────────────────

/// Picked qmm_t variant for a given `(M, N, K, B)` shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QmmTKernel {
    /// `affine_qmm_t_*_alN_<bool>_batch_0` — standard prefill matmul,
    /// transpose=true. Always fires for `B > 1` or when split_k
    /// reduces to 1. MLX `qmm` at `quantized.cpp:680`.
    Standard,
    /// `affine_qmm_t_splitk_*_alN_<bool>` — split-K variant. Fires only
    /// when `B == 1` and a non-trivial split_k is feasible (the
    /// `qmm_splitk` heuristic at `quantized.cpp:788-805` targets
    /// ~512 threadgroups; falls back to `Standard` if split_k ≤ 1).
    SplitK { split_k: u32, k_partition_size: u32 },
    /// `affine_qmm_t_nax_*_alN_<bool>_batch_0` — NAX (Apple9 / M4+)
    /// MMA path using `MetalPerformancePrimitives matmul2d`. Only
    /// selected when `is_nax == true` AND `K % 64 == 0`.
    /// 64×64×64 tile, no split-K (MLX NAX path at `quantized.cpp:695`).
    Nax,
}

/// Compute the splitk plan per MLX `qmm_splitk` (`quantized.cpp:788-805`):
///   bm=bn=32, target ~512 active threadgroups → split_k = max(1, 512 / (n_tiles*m_tiles))
///   cap by K/group_size, ensure K % (split_k * group_size) == 0
///   if split_k <= 1: fall back to standard qmm_t
pub fn pick_qmm_t_split_k(m: u32, n: u32, k: u32, group_size: u32) -> u32 {
    const BM: u32 = 32;
    const BN: u32 = 32;
    let n_tiles = n.div_ceil(BN);
    let m_tiles = m.div_ceil(BM);
    let current_tgs = n_tiles * m_tiles;
    if current_tgs == 0 {
        return 1;
    }
    let mut split_k = (512u32 / current_tgs).max(1);
    let group_cap = k / group_size;
    if group_cap == 0 {
        return 1;
    }
    split_k = split_k.min(group_cap);
    while split_k > 1 && !k.is_multiple_of(split_k * group_size) {
        split_k -= 1;
    }
    split_k
}

/// Pick the right qmm_t variant per MLX's matmul-branch routing.
/// Mirrors `quantized.cpp:1411-1424` plus the NAX gate at `:695`:
///
/// ```text
/// if is_nax && M >= 32 && K % 64 == 0 && group_size != 32: qmm_t_nax (M5+)
/// else if transpose && B == 1:                      qmm_splitk
/// else if transpose:                                qmm (transpose=true)
/// ```
///
/// `is_nax` should be `crate::tape::targets::is_nax_capable(profile.generation)`.
///
/// gs=32 is excluded from NAX dispatch because the BK=64 NAX shader
/// violates `BCOLS <= group_size` for gs=32; MLX handles that with a
/// specialized QuantizedBlockLoader path (different scale-indexing
/// semantics) which we haven't ported. gs=32 quants fall through to
/// the Standard qmm_t kernel instead.
pub fn pick_qmm_t_kernel(
    m: u32,
    n: u32,
    k: u32,
    b: u32,
    group_size: u32,
    is_nax: bool,
) -> QmmTKernel {
    // NAX's 64×64 tiling wastes work and yields too few threadgroups at
    // tiny M: it regresses to ~0.8× vs SplitK at M=16 but wins (≥1.7×)
    // from M=32 up (see `nax_vs_standard_qmm_t_bench`). Gate it to
    // M ≥ 32 so very short prefills keep the SplitK/Standard path.
    // (Belt-and-suspenders: the production qmm_t buckets start at 64,
    // so this threshold isn't normally reached.)
    const NAX_MIN_M: u32 = 32;
    if is_nax && m >= NAX_MIN_M && k.is_multiple_of(64) && group_size != 32 {
        return QmmTKernel::Nax;
    }
    if b == 1 {
        let split_k = pick_qmm_t_split_k(m, n, k, group_size);
        if split_k > 1 {
            return QmmTKernel::SplitK {
                split_k,
                k_partition_size: k / split_k,
            };
        }
    }
    QmmTKernel::Standard
}

/// Threadgroup grid + threads-per-group for a picked qmm_t variant.
///
/// `qmm_t`: bm=bn=32, wm=wn=2 → group (32, 2, 2); grid
/// `(ceil(N/32), ceil(M/32), B)` (`quantized.cpp:720-721`).
///
/// `qmm_t_splitk`: same group; grid `(n_tiles, m_tiles, split_k)`
/// (`quantized.cpp:822-824`).
pub fn qmm_t_dispatch_shape(
    kernel: QmmTKernel,
    m: u32,
    n: u32,
    b: u32,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    match kernel {
        QmmTKernel::Nax => {
            // BM=BN=64 tile, TGP=128 = 4 simdgroups × 32 threads.
            // Threadgroup geometry is MLX's (32, wn=2, wm=2): the kernel
            // maps its 2×2 output sub-tile grid purely from
            // `simdgroup_index_in_threadgroup` (numbered identically in a
            // flat (128,1,1) and a 3D (32,2,2) group), so the 3D shape is
            // bit-identical but ~7% faster — the MPP `matmul2d` scheduler
            // prefers it. Total threads = 32·2·2 = 128 either way, so the
            // `(n_tiles, m_tiles, b)` threadgroup grid + Y-axis m_scaling
            // are unchanged.
            let n_tiles = n.div_ceil(64);
            let m_tiles = m.div_ceil(64);
            ((n_tiles, m_tiles, b), (32, 2, 2))
        }
        _ => {
            let n_tiles = n.div_ceil(32);
            let m_tiles = m.div_ceil(32);
            match kernel {
                QmmTKernel::Standard => ((n_tiles, m_tiles, b), (32, 2, 2)),
                QmmTKernel::SplitK { split_k, .. } => ((n_tiles, m_tiles, split_k), (32, 2, 2)),
                QmmTKernel::Nax => unreachable!(),
            }
        }
    }
}

/// Format the kernel symbol name. Matches the `INST_QMM_*` macros in
/// `shaders/quantized_qmm.metal`.
pub fn qmm_t_kernel_name(
    kernel: QmmTKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
    aligned_n: bool,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    let aln = if aligned_n { "true" } else { "false" };
    match kernel {
        QmmTKernel::Standard => {
            format!("affine_qmm_t_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_alN_{aln}_batch_0",)
        }
        QmmTKernel::SplitK { .. } => {
            format!("affine_qmm_t_splitk_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_alN_{aln}",)
        }
        QmmTKernel::Nax => {
            format!("affine_qmm_t_nax_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_alN_{aln}_batch_0",)
        }
    }
}

/// `&'static str` view of [`qmm_t_kernel_name`] for the lowering pass
/// — see [`qmv_kernel_static_name`] for the analogous discussion.
/// `LoweredCommand::function` is `&'static str`; the table below
/// enumerates every entry the `INST_QMM_ALL` macro produces in
/// `shaders/quantized_qmm.metal`. `scale_dtype` is plumbed for
/// forward-compat with P11 — only `ScaleDtype::F16` is instantiated.
pub fn qmm_t_kernel_static_name(
    kernel: QmmTKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
    aligned_n: bool,
) -> &'static str {
    debug_assert!(
        matches!(bits, 4 | 8),
        "qmm_t_kernel_static_name: only bits 4 and 8 are wired (got {bits})"
    );
    debug_assert!(
        !(bits == 8 && matches!(kernel, QmmTKernel::SplitK { .. })),
        "qmm_t_kernel_static_name: SplitK only instantiates bits=4 — \
         the lowering arm must route 8-bit weights to Standard or Nax"
    );
    let key = (
        std::mem::discriminant(&kernel),
        dtype,
        scale_dtype,
        group_size,
        aligned_n,
        bits,
    );
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Key = (
        std::mem::Discriminant<QmmTKernel>,
        DequantDtype,
        ScaleDtype,
        u32,
        bool,
        u32,
    );
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .expect("qmm_t_kernel_static_name cache poisoned");
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    if !matches!(group_size, 32 | 64 | 128) {
        panic!(
            "qmm_t_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    if matches!(kernel, QmmTKernel::Nax) && group_size == 32 {
        panic!(
            "qmm_t_kernel_static_name: NAX dispatched with gs=32 — \
             `pick_qmm_t_kernel` should have routed to Standard"
        );
    }
    let owned = qmm_t_kernel_name(kernel, dtype, scale_dtype, group_size, bits, aligned_n);
    // SplitK kernel names lack the trailing `_batch_0` suffix —
    // `qmm_t_kernel_name` already handles that distinction.
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

// ─────────────────────────────────────────────────────────────────
// Small-M matrix-unit GEMM (NAX) — decode batches.
// ─────────────────────────────────────────────────────────────────

/// Step token counts the small-M matrix-unit GEMM (`affine_qmm_small_m_*`)
/// serves on NAX devices. Below, `qmv_fast`'s per-row weight re-reads cost
/// no more than the matrix unit's padding; above, a second M tile re-reads
/// the weights and NAX `qmm_t`'s 64-row tile wins (a 32-row tile loses
/// too). Base M5, Llama-3.2-3B, weights streamed from memory: ~1.35×
/// `qmv_fast` at 6 tokens and ~1.9× at 8; at 9–16 level with NAX `qmm_t` on
/// the wide gate/up/down shapes, and ~15–20% lower decode TPOT at 16
/// concurrent sequences end to end than routing them to `qmm_t`.
pub const SMALL_M_TOKENS: std::ops::RangeInclusive<u32> = 5..=16;

/// Output columns per small-M threadgroup (one simdgroup).
pub const SMALL_M_TILE_COLS: u32 = 16;

/// Rows per small-M threadgroup: the whole batch in the buckets up to 8
/// tokens, 16 in the larger ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SmallMTile {
    Rows8,
    Rows16,
}

impl SmallMTile {
    pub fn rows(self) -> u32 {
        match self {
            Self::Rows8 => 8,
            Self::Rows16 => 16,
        }
    }

    /// The tile of a bucket that can see a [`SMALL_M_TOKENS`] step: from the
    /// range's first count up to the 64-token bucket, the largest that serves
    /// the range under the default ladder.
    pub fn for_bucket(bucket_m: u32) -> Option<Self> {
        match bucket_m {
            m if m < *SMALL_M_TOKENS.start() || m > 64 => None,
            ..=8 => Some(Self::Rows8),
            _ => Some(Self::Rows16),
        }
    }
}

/// `affine_qmm_small_m_<act>_s_<scale>_gs_<gs>_b_4_tm_<rows>_tn_16_nsg_1`
/// (`quantized_qmm_nax.metal`).
pub fn small_m_kernel_static_name(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    tile: SmallMTile,
) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (DequantDtype, ScaleDtype, u32, SmallMTile);
    static CACHE: OnceLock<Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
    let mut guard = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("small_m_kernel_static_name cache poisoned");
    guard
        .entry((dtype, scale_dtype, group_size, tile))
        .or_insert_with(|| {
            Box::leak(
                format!(
                    "affine_qmm_small_m_{}_s_{}_gs_{group_size}_b_4_tm_{}_tn_{SMALL_M_TILE_COLS}_nsg_1",
                    dtype.symbol_infix(),
                    scale_dtype.symbol_infix(),
                    tile.rows(),
                )
                .into_boxed_str(),
            )
        })
}

/// Compute-aware variant. When `compute_dtype != dtype`, picks the
/// extended `affine_qmm_t_<act>_c_<compute>_s_<scale>_*` symbol. Only
/// the (Bf16-act, F16-compute) combo is currently instantiated — used
/// on Apple7 (M1) where bf16 simdgroup MMAs are slow-path emulation.
/// Falls back to [`qmm_t_kernel_static_name`] when compute == act.
pub fn qmm_t_kernel_static_name_with_compute(
    kernel: QmmTKernel,
    act_dtype: DequantDtype,
    compute_dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
    aligned_n: bool,
) -> &'static str {
    if compute_dtype == act_dtype {
        return qmm_t_kernel_static_name(
            kernel,
            act_dtype,
            scale_dtype,
            bits,
            group_size,
            aligned_n,
        );
    }
    // The mixed-compute (`_c_f16_`) match below is instantiated for bits=4
    // ONLY and its arms omit `bits` from the key, so a non-4-bit weight would
    // silently resolve to a wrong `_b_4_` symbol (8-bit decoded as 4-bit →
    // garbage). There is no mixed-compute kernel for other widths, so fall
    // back to the same-compute name (which threads `bits`); callers must not
    // request a compute-flip for bits!=4 (see the `bits_v == 4` gate on the
    // M1 f16 flip in lowering). This makes the mis-dispatch structurally
    // impossible rather than a release-elided debug_assert.
    if bits != 4 {
        return qmm_t_kernel_static_name(
            kernel,
            act_dtype,
            scale_dtype,
            bits,
            group_size,
            aligned_n,
        );
    }
    use DequantDtype::*;
    use ScaleDtype as S;
    match (
        kernel,
        act_dtype,
        compute_dtype,
        scale_dtype,
        group_size,
        aligned_n,
    ) {
        // ── qmm_t Standard, bf16-act + f16-compute ───────────────
        (QmmTKernel::Standard, Bf16, F16, S::F16, 32, true) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_32_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 32, false) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_32_b_4_alN_false_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 64, true) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_64_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 64, false) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_64_b_4_alN_false_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 128, true) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_128_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 128, false) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_128_b_4_alN_false_batch_0"
        }
        // ── qmm_t SplitK, bf16-act + f16-compute ─────────────────
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 32, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_32_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 32, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_32_b_4_alN_false"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 64, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_64_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 64, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_64_b_4_alN_false"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 128, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_128_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 128, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_128_b_4_alN_false"
        }
        // ── qmm_t Standard, bf16-act + f16-compute + bf16-scale ──
        (QmmTKernel::Standard, Bf16, F16, S::Bf16, 32, true) => {
            "affine_qmm_t_bf16_c_f16_s_bf16_gs_32_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::Bf16, 32, false) => {
            "affine_qmm_t_bf16_c_f16_s_bf16_gs_32_b_4_alN_false_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::Bf16, 64, true) => {
            "affine_qmm_t_bf16_c_f16_s_bf16_gs_64_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::Bf16, 64, false) => {
            "affine_qmm_t_bf16_c_f16_s_bf16_gs_64_b_4_alN_false_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::Bf16, 128, true) => {
            "affine_qmm_t_bf16_c_f16_s_bf16_gs_128_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::Bf16, 128, false) => {
            "affine_qmm_t_bf16_c_f16_s_bf16_gs_128_b_4_alN_false_batch_0"
        }
        // ── qmm_t SplitK, bf16-act + f16-compute + bf16-scale ────
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::Bf16, 32, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_bf16_gs_32_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::Bf16, 32, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_bf16_gs_32_b_4_alN_false"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::Bf16, 64, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_bf16_gs_64_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::Bf16, 64, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_bf16_gs_64_b_4_alN_false"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::Bf16, 128, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_bf16_gs_128_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::Bf16, 128, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_bf16_gs_128_b_4_alN_false"
        }
        // NAX path: don't override compute dtype — Apple9 has hardware
        // bf16 acceleration via the matrix unit, no fast-path needed.
        (QmmTKernel::Nax, _, _, _, _, _) => panic!(
            "qmm_t_kernel_static_name_with_compute: NAX path doesn't need a compute override"
        ),
        _ => panic!(
            "qmm_t_kernel_static_name_with_compute: unsupported (act={act_dtype:?}, \
             compute={compute_dtype:?}, scale={scale_dtype:?}, gs={group_size}) — only \
             (Bf16, F16, F16) is instantiated for the M1 fast-path"
        ),
    }
}

// ─────────────────────────────────────────────────────────────────
// SplitK reduce dispatcher — port of MLX's
// `strided_reduce_general_dispatch` invocation at
// `quantized.cpp:861`. Reduces the [split_k, M, N] intermediate
// `affine_qmm_t_splitk` produces down to the final [M, N] output by
// summing along axis 0.
//
// Backed by `splitk_reduce_sum_<dtype>` in
// `shaders/quantized_splitk_reduce.metal`. Function constants 0/1/2
// hold (M, N, split_k) — same `[[function_constant(N)]]` pattern
// used by qmv/qmm_t for dispatch readiness.
// ─────────────────────────────────────────────────────────────────

/// Format the splitk-reduce kernel symbol. Matches the
/// `INST_REDUCE` macros in `shaders/quantized_splitk_reduce.metal`.
pub fn splitk_reduce_kernel_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "splitk_reduce_sum_f16",
        DequantDtype::Bf16 => "splitk_reduce_sum_bf16",
    }
}

// ─────────────────────────────────────────────────────────────────
// Prefill-bucket qmm_n dispatcher — port of `quantized.cpp:680
// qmm()` (transpose=false branch). Used when `M >= vector_limit`
// AND transpose=false (which for MLX means `vector_limit = 4`
// flat per `quantized.cpp:1409`; scratchy mirrors that for the
// transpose=false matmul branch). No SplitK variant for qmm_n in
// MLX (only qmm_t has splitk per `quantized.cpp:1413`).
// ─────────────────────────────────────────────────────────────────

/// Format the qmm_n kernel symbol. Matches the `INST_QMM_N` macro in
/// `shaders/quantized_qmm.metal`.
pub fn qmm_n_kernel_name(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    format!("affine_qmm_n_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_0")
}

/// `&'static str` view of [`qmm_n_kernel_name`] for the lowering pass.
pub fn qmm_n_kernel_static_name(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
) -> &'static str {
    debug_assert_eq!(bits, 4, "qmm_n_kernel_static_name: only bits=4 is wired");
    if !matches!(group_size, 32 | 64 | 128) {
        panic!(
            "qmm_n_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Key = (DequantDtype, ScaleDtype, u32);
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .expect("qmm_n_kernel_static_name cache poisoned");
    let key = (dtype, scale_dtype, group_size);
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    let owned = qmm_n_kernel_name(dtype, scale_dtype, group_size, 4);
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

/// Threadgroup grid + threads-per-group for qmm_n. Matches MLX
/// `quantized.cpp:720-721`: bm=bn=32, wm=wn=2 → group (32, 2, 2);
/// grid `(ceil(N/32), ceil(M/32), B)`.
pub fn qmm_n_dispatch_shape(m: u32, n: u32, b: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    let n_tiles = n.div_ceil(32);
    let m_tiles = m.div_ceil(32);
    ((n_tiles, m_tiles, b), (32, 2, 2))
}

// ─────────────────────────────────────────────────────────────────
// Decode-bucket qvm / qvm_split_k dispatcher — port of
// `quantized.cpp:419 qvm()` + `:298 qvm_split_k()`. Fires when
// M < vector_limit AND transpose=false (the matvec-transpose=false
// branch at `:1444-1453`):
//   K <  1024  →  qvm
//   K >= 1024  →  qvm_split_k (split_k = K > 8192 ? 32 : 8)
// ─────────────────────────────────────────────────────────────────

/// Picked qvm variant for a given `(M, N, K)` shape (matvec
/// transpose=false branch). Mirrors `QuantizedMatmul::eval_gpu`
/// at `quantized.cpp:1445-1452`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QvmKernel {
    /// `affine_qvm_*_batch_0` — K < 1024. MLX `qvm` at
    /// `quantized.cpp:419`.
    Standard,
    /// `affine_qvm_split_k_*` — K >= 1024. MLX `qvm_split_k` at
    /// `quantized.cpp:298`; `split_k = K > 8192 ? 32 : 8` and
    /// `split_D = ceil(K / split_k)`.
    SplitK {
        split_k: u32,
        k_partition_size: u32,
        final_block_size: u32,
    },
}

/// Pick the right qvm variant for `(M, N, K)`. Mirrors MLX's
/// transpose=false matvec routing at `quantized.cpp:1444-1453`:
///
/// ```text
/// if (K < 1024)  qvm(...)
/// else           qvm_split_k(...)
///   split_k = K > 8192 ? 32 : 8
///   split_D = ceil(K / split_k)
///   final_block_size = K - (split_k - 1) * split_D
/// ```
pub fn pick_qvm_kernel(k: u32) -> QvmKernel {
    if k < 1024 {
        QvmKernel::Standard
    } else {
        let split_k: u32 = if k > 8192 { 32 } else { 8 };
        let split_d = k.div_ceil(split_k);
        let final_block_size = k - (split_k - 1) * split_d;
        QvmKernel::SplitK {
            split_k,
            k_partition_size: split_d,
            final_block_size,
        }
    }
}

/// Threadgroup grid + threads-per-group for a picked qvm variant.
/// Matches MLX's dispatch:
///
/// qvm (`quantized.cpp:438-439`):
///   group (bk=32, num_simdgroups=2, 1); grid (M, (N+bn-1)/bn, B)
///   where `bn = min(group_size, 32) * 2`.
///
/// qvm_split_k (`quantized.cpp:322-323`):
///   group (bk=32, num_simdgroups=2, 1); grid (M, N/bn, B*split_k).
pub fn qvm_dispatch_shape(
    kernel: QvmKernel,
    m: u32,
    n: u32,
    b: u32,
    group_size: u32,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    let bn: u32 = group_size.min(32) * 2; // 64 for gs ∈ {32, 64, 128}
    match kernel {
        QvmKernel::Standard => ((m, n.div_ceil(bn), b), (32, 2, 1)),
        QvmKernel::SplitK { split_k, .. } => ((m, n / bn, b * split_k), (32, 2, 1)),
    }
}

/// Format the kernel symbol name. Matches the `INST_QVM_*` macros
/// in `shaders/quantized_qvm.metal`.
pub fn qvm_kernel_name(
    kernel: QvmKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    match kernel {
        QvmKernel::Standard => {
            format!("affine_qvm_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_0")
        }
        QvmKernel::SplitK { .. } => {
            format!("affine_qvm_split_k_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}")
        }
    }
}

/// `&'static str` view of [`qvm_kernel_name`] for the lowering pass.
pub fn qvm_kernel_static_name(
    kernel: QvmKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
) -> &'static str {
    debug_assert_eq!(bits, 4, "qvm_kernel_static_name: only bits=4 is wired");
    if !matches!(group_size, 32 | 64 | 128) {
        panic!(
            "qvm_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Key = (
        std::mem::Discriminant<QvmKernel>,
        DequantDtype,
        ScaleDtype,
        u32,
    );
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("qvm_kernel_static_name cache poisoned");
    let key = (
        std::mem::discriminant(&kernel),
        dtype,
        scale_dtype,
        group_size,
    );
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    let owned = qvm_kernel_name(kernel, dtype, scale_dtype, group_size, 4);
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::targets::AppleSiliconGen;

    #[test]
    fn qmv_kernel_pick_matches_mlx_dispatch_qmv() {
        // K==64 + pow2 bits → quad
        assert_eq!(pick_qmv_kernel(2048, 64, 4), QmvKernel::Quad { d: 64 });
        assert_eq!(pick_qmv_kernel(2048, 128, 4), QmvKernel::Quad { d: 128 });
        // K==96 → fast/generic, not quad
        assert_eq!(pick_qmv_kernel(2048, 96, 4), QmvKernel::Generic);
        // bits=3 (not power of 2) at K=64 → not quad
        assert_eq!(pick_qmv_kernel(2048, 64, 3), QmvKernel::Generic);

        // N%8==0 && K%512==0 → fast (Llama-1B q_proj: K=2048, N=2048)
        assert_eq!(pick_qmv_kernel(2048, 2048, 4), QmvKernel::Fast);
        // Llama-1B kv_proj: K=2048, N=512 → fast
        assert_eq!(pick_qmv_kernel(512, 2048, 4), QmvKernel::Fast);
        // Llama-1B gate/up_proj: K=2048, N=8192 → fast
        assert_eq!(pick_qmv_kernel(8192, 2048, 4), QmvKernel::Fast);
        // Llama-1B down_proj: K=8192, N=2048 → fast
        assert_eq!(pick_qmv_kernel(2048, 8192, 4), QmvKernel::Fast);

        // K%512!=0 → generic
        assert_eq!(pick_qmv_kernel(2048, 1024, 4), QmvKernel::Fast);
        assert_eq!(pick_qmv_kernel(2048, 1023, 4), QmvKernel::Generic);
        // N%8!=0 → generic
        assert_eq!(pick_qmv_kernel(2049, 2048, 4), QmvKernel::Generic);
    }

    #[test]
    fn qmv_dispatch_shape_matches_mlx_grid_dims() {
        // qmv_quad: bn = 64
        let ((tx, ty, tz), (gx, gy, gz)) =
            qmv_dispatch_shape(QmvKernel::Quad { d: 64 }, 1, 2048, 1);
        assert_eq!((tx, ty, tz), (1, 2048u32.div_ceil(64), 1));
        assert_eq!((gx, gy, gz), (32, 1, 1));

        // qmv_fast: bn = 8
        let ((tx, ty, tz), (gx, gy, gz)) = qmv_dispatch_shape(QmvKernel::Fast, 1, 2048, 1);
        assert_eq!((tx, ty, tz), (1, 2048u32.div_ceil(8), 1));
        assert_eq!((gx, gy, gz), (32, 2, 1));

        // qmv_generic shares qmv_fast's grid
        let ((tx, ty, tz), _) = qmv_dispatch_shape(QmvKernel::Generic, 1, 2049, 1);
        assert_eq!((tx, ty, tz), (1, 2049u32.div_ceil(8), 1));
    }

    #[test]
    fn qmv_kernel_name_matches_metallib_symbols() {
        // Decoded against the actual exported symbols in
        // shaders/quantized_qmv.metal's INST_QMV_* macros.
        assert_eq!(
            qmv_kernel_name(
                QmvKernel::Fast,
                DequantDtype::Bf16,
                ScaleDtype::F16,
                64,
                4,
                false
            ),
            "affine_qmv_fast_bf16_s_f16_gs_64_b_4_batch_0"
        );
        assert_eq!(
            qmv_kernel_name(
                QmvKernel::Generic,
                DequantDtype::F16,
                ScaleDtype::F16,
                32,
                4,
                true
            ),
            "affine_qmv_f16_s_f16_gs_32_b_4_batch_1"
        );
        assert_eq!(
            qmv_kernel_name(
                QmvKernel::Quad { d: 128 },
                DequantDtype::Bf16,
                ScaleDtype::F16,
                128,
                4,
                false
            ),
            "affine_qmv_quad_bf16_s_f16_gs_128_b_4_d_128_batch_0"
        );
    }

    #[test]
    fn qmm_t_split_k_matches_mlx_heuristic() {
        // Llama-1B prefill q_proj: M=64, N=2048, K=2048, gs=64, B=1.
        // n_tiles = 64, m_tiles = 2 → current_tgs = 128 → target_split_k =
        // 512/128 = 4. K/group_size = 32 → cap 32. K % (4*64) = 0 → 4 stands.
        assert_eq!(pick_qmm_t_split_k(64, 2048, 2048, 64), 4);

        // Llama-1B prefill o_proj: same shape, B=1.
        // Same as above.
        assert_eq!(pick_qmm_t_split_k(64, 2048, 2048, 64), 4);

        // Long-prompt prefill: M=512, N=2048, K=2048, gs=64.
        // n_tiles = 64, m_tiles = 16 → current_tgs = 1024 → target_split_k
        // = 512/1024 = 0, clamped to 1.
        assert_eq!(pick_qmm_t_split_k(512, 2048, 2048, 64), 1);

        // Decode-shape qmm border: M=18, N=8192, K=2048, gs=64.
        // n_tiles = 256, m_tiles = 1 → tgs = 256 → 512/256 = 2. cap = 32.
        // K % (2*64) = 0 → 2 stands.
        assert_eq!(pick_qmm_t_split_k(18, 8192, 2048, 64), 2);
    }

    #[test]
    fn qmm_t_kernel_pick_routes_splitk_only_for_b_eq_1() {
        // B == 1, decent splitk → SplitK. is_nax=false: this test pins the
        // non-NAX SplitK/Standard routing.
        let k = pick_qmm_t_kernel(64, 2048, 2048, 1, 64, false);
        assert!(matches!(k, QmmTKernel::SplitK { split_k: 4, .. }));

        // B > 1 → always Standard
        let k = pick_qmm_t_kernel(64, 2048, 2048, 2, 64, false);
        assert_eq!(k, QmmTKernel::Standard);

        // B == 1, splitk collapses to 1 → Standard
        let k = pick_qmm_t_kernel(512, 2048, 2048, 1, 64, false);
        assert_eq!(k, QmmTKernel::Standard);
    }

    #[test]
    fn qmm_t_kernel_name_matches_metallib_symbols() {
        assert_eq!(
            qmm_t_kernel_name(
                QmmTKernel::Standard,
                DequantDtype::Bf16,
                ScaleDtype::F16,
                64,
                4,
                true
            ),
            "affine_qmm_t_bf16_s_f16_gs_64_b_4_alN_true_batch_0"
        );
        assert_eq!(
            qmm_t_kernel_name(
                QmmTKernel::Standard,
                DequantDtype::F16,
                ScaleDtype::F16,
                32,
                4,
                false
            ),
            "affine_qmm_t_f16_s_f16_gs_32_b_4_alN_false_batch_0"
        );
        assert_eq!(
            qmm_t_kernel_name(
                QmmTKernel::SplitK {
                    split_k: 4,
                    k_partition_size: 512
                },
                DequantDtype::Bf16,
                ScaleDtype::F16,
                128,
                4,
                true
            ),
            "affine_qmm_t_splitk_bf16_s_f16_gs_128_b_4_alN_true"
        );
    }

    #[test]
    fn qmm_t_dispatch_shape_matches_mlx_grid_dims() {
        // Standard: grid (ceil(N/32), ceil(M/32), B); group (32, 2, 2).
        let ((tx, ty, tz), (gx, gy, gz)) = qmm_t_dispatch_shape(QmmTKernel::Standard, 64, 2048, 1);
        assert_eq!((tx, ty, tz), (64, 2, 1));
        assert_eq!((gx, gy, gz), (32, 2, 2));

        // SplitK: grid (n_tiles, m_tiles, split_k); same group.
        let ((tx, ty, tz), (gx, gy, gz)) = qmm_t_dispatch_shape(
            QmmTKernel::SplitK {
                split_k: 4,
                k_partition_size: 512,
            },
            64,
            2048,
            1,
        );
        assert_eq!((tx, ty, tz), (64, 2, 4));
        assert_eq!((gx, gy, gz), (32, 2, 2));

        // Nax: grid (ceil(N/64), ceil(M/64), B); group (32, wn=2, wm=2) —
        // MLX's 3D threadgroup (see `qmm_nax` in mlx quantized.cpp). Tile
        // mapping is from simdgroup_index, so (32,2,2) is bit-identical to
        // a flat (128,1,1) but ~7% faster on the MPP matmul2d scheduler.
        let ((tx, ty, tz), (gx, gy, gz)) = qmm_t_dispatch_shape(QmmTKernel::Nax, 128, 2048, 1);
        assert_eq!((tx, ty, tz), (32, 2, 1));
        assert_eq!((gx, gy, gz), (32, 2, 2));
    }

    #[test]
    fn qvm_kernel_pick_matches_mlx_rule() {
        // K < 1024 → Standard
        assert_eq!(pick_qvm_kernel(64), QvmKernel::Standard);
        assert_eq!(pick_qvm_kernel(1023), QvmKernel::Standard);
        // K == 1024 → SplitK with split_k=8
        let k = pick_qvm_kernel(1024);
        assert!(matches!(
            k,
            QvmKernel::SplitK {
                split_k: 8,
                k_partition_size: 128,
                final_block_size: 128,
            }
        ));
        // K = 4096 → SplitK with split_k=8, split_D=512
        let k = pick_qvm_kernel(4096);
        assert!(matches!(
            k,
            QvmKernel::SplitK {
                split_k: 8,
                k_partition_size: 512,
                final_block_size: 512,
            }
        ));
        // K = 8193 → split_k=32 (K > 8192)
        let k = pick_qvm_kernel(8193);
        assert!(matches!(k, QvmKernel::SplitK { split_k: 32, .. }));
        // K = 8200 → split_k=32, split_D=257, final_block_size=200
        // (K - 31*257 = 8200 - 7967 = 233, but 8200/32 ceil = 257)
        let k = pick_qvm_kernel(8200);
        if let QvmKernel::SplitK {
            split_k,
            k_partition_size,
            final_block_size,
        } = k
        {
            assert_eq!(split_k, 32);
            assert_eq!(k_partition_size, 257); // ceil(8200/32)
            assert_eq!(final_block_size, 8200 - 31 * 257);
        } else {
            panic!("expected SplitK, got {k:?}");
        }
    }

    #[test]
    fn qvm_dispatch_shape_matches_mlx() {
        // qvm: grid (M, ceil(N / bn), B); bn = min(gs, 32)*2 = 64
        let ((tx, ty, tz), (gx, gy, gz)) = qvm_dispatch_shape(QvmKernel::Standard, 1, 4096, 1, 64);
        assert_eq!((tx, ty, tz), (1, 4096 / 64, 1));
        assert_eq!((gx, gy, gz), (32, 2, 1));

        // qvm_split_k: grid (M, N/bn, B * split_k)
        let kernel = QvmKernel::SplitK {
            split_k: 8,
            k_partition_size: 512,
            final_block_size: 512,
        };
        let ((tx, ty, tz), _) = qvm_dispatch_shape(kernel, 1, 4096, 1, 64);
        assert_eq!((tx, ty, tz), (1, 4096 / 64, 8));
    }

    #[test]
    fn qvm_kernel_name_matches_metallib_symbols() {
        assert_eq!(
            qvm_kernel_name(
                QvmKernel::Standard,
                DequantDtype::Bf16,
                ScaleDtype::F16,
                64,
                4
            ),
            "affine_qvm_bf16_s_f16_gs_64_b_4_batch_0"
        );
        assert_eq!(
            qvm_kernel_name(
                QvmKernel::SplitK {
                    split_k: 8,
                    k_partition_size: 512,
                    final_block_size: 512
                },
                DequantDtype::F16,
                ScaleDtype::F16,
                128,
                4
            ),
            "affine_qvm_split_k_f16_s_f16_gs_128_b_4"
        );
    }

    #[test]
    fn qmm_n_kernel_name_matches_metallib_symbols() {
        assert_eq!(
            qmm_n_kernel_name(DequantDtype::Bf16, ScaleDtype::F16, 64, 4),
            "affine_qmm_n_bf16_s_f16_gs_64_b_4_batch_0"
        );
        assert_eq!(
            qmm_n_kernel_name(DequantDtype::F16, ScaleDtype::F16, 32, 4),
            "affine_qmm_n_f16_s_f16_gs_32_b_4_batch_0"
        );
    }

    #[test]
    fn qmm_n_dispatch_shape_matches_mlx() {
        // qmm_n: grid (ceil(N/32), ceil(M/32), B); group (32, 2, 2)
        let ((tx, ty, tz), (gx, gy, gz)) = qmm_n_dispatch_shape(64, 2048, 1);
        assert_eq!((tx, ty, tz), (64, 2, 1));
        assert_eq!((gx, gy, gz), (32, 2, 2));
    }

    #[test]
    fn qmv_batch_limit_matches_mlx_table() {
        // M3+ default branch (non-'d'): 18, 12, 10 per (K, N) buckets
        assert_eq!(get_qmv_batch_limit(2048, 2048, AppleSiliconGen::M3), 18);
        assert_eq!(get_qmv_batch_limit(4096, 4096, AppleSiliconGen::M3), 12);
        assert_eq!(get_qmv_batch_limit(8192, 8192, AppleSiliconGen::M3), 10);
        assert_eq!(get_qmv_batch_limit(2048, 2048, AppleSiliconGen::M4), 18);

        // M1/M2 branch: 14, 10, 6
        assert_eq!(get_qmv_batch_limit(2048, 2048, AppleSiliconGen::M1), 14);
        assert_eq!(get_qmv_batch_limit(4096, 4096, AppleSiliconGen::M2), 10);
        assert_eq!(get_qmv_batch_limit(8192, 8192, AppleSiliconGen::M1), 6);
    }
}

/// outlier-heavy KV (e.g. Qwen's massive activations) where 3-bit degrades. Read
/// once at worker init from `SCRATCHY_TQ_BITS`; the lowering bakes the SAME value
/// into the kernel constants, so they always agree within a run.
pub fn tq_bits(arch_default: u32) -> u32 {
    std::env::var("SCRATCHY_TQ_BITS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&b| (2..=8).contains(&b))
        .unwrap_or(arch_default)
}
