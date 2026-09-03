// SPDX-License-Identifier: Apache-2.0
//! Metal-only quantized linear / embedding LOADERS (MLX-affine INT4 + NVFP4).
//!
//! The payload TYPES (`AffineQuantLinear` / `Nvfp4Linear` /
//! `AffineQuantEmbedding`) + their pure accessors live in the
//! `scratchy-quantizations` crate (so `scratchy_target_cuda::layers::LinearLayer`
//! can hold them behind `AffineQuant` / `Nvfp4` enum variants). Their `load()`
//! constructors stay here as metal-local extension traits — they `load()`
//! against the neutral [`WeightSource`] seam so they need not name
//! `GpuWeights` (which lives in targets/cuda), letting them sit below it. The
//! `e4m3_to_f32` decode (used by the NVFP4 fold) is a free fn below.

use anyhow::Result;
use scratchy_layers::{Embedding, Linear, LinearLayer};
use scratchy_quantizations::{
    AffineQuantEmbedding, AffineQuantLinear, Nvfp4Linear, affine_dequant_b4_to_dtype,
};
use scratchy_tensors::{DType, GpuTensor, WeightSource};

// ---------------------------------------------------------------------------
// MLX-affine int4 dequantize-at-load helper (WeightSource-generic, Metal-only)
// ---------------------------------------------------------------------------

/// CPU-dequantize one MLX-affine int4 weight prefix to a dense `[N, K]`
/// `dtype_out` device tensor, through the neutral [`WeightSource`] seam.
///
/// Reads `<prefix>.{weight,scales,biases}` to host (`take_cpu`), validates the
/// layout (`U32` packed weight `[N, K/8]`; `F16`/`BF16` scales/biases
/// `[N, K/group_size]`), runs the pure [`affine_dequant_b4_to_dtype`] math, then
/// `alloc_packed_from_host`s the dense `[N, K]` result. Returns
/// `(weight, n, k)`.
///
/// Uses ONLY the frozen `WeightSource` vocabulary (`take_cpu` +
/// `alloc_packed_from_host`); the dequant arithmetic is byte-for-byte the math
/// that used to live in `GpuWeights::take_affine_dequant_b4_bytes`. NOTE: the
/// `take_cpu` read copies the small packed 4-bit weight (vs the old zero-copy
/// mmap read) — one extra small copy at load, correctness-preserving.
fn affine_dequant_b4_bytes<W: WeightSource + ?Sized>(
    weights: &mut W,
    prefix: &str,
    group_size: u32,
    bits: u32,
    dtype_out: DType,
) -> Result<(Vec<u8>, usize, usize)> {
    anyhow::ensure!(
        bits == 4 || bits == 8,
        "affine_dequant_b4: only bits=4 or bits=8 supported, got bits={bits}"
    );
    anyhow::ensure!(
        matches!(dtype_out, DType::F16 | DType::BF16),
        "affine_dequant_b4: dtype_out must be F16 or BF16, got {dtype_out}"
    );

    let (w_bytes, w_shape, w_dtype) = weights.take_cpu(&format!("{prefix}.weight"))?;
    let (s_bytes, s_shape, s_dtype) = weights.take_cpu(&format!("{prefix}.scales"))?;
    let (b_bytes, b_shape, b_dtype) = weights.take_cpu(&format!("{prefix}.biases"))?;

    anyhow::ensure!(
        w_dtype == DType::U32,
        "affine_dequant_b4: `{prefix}.weight` dtype is {w_dtype} (expected U32)",
    );
    // mlx-community 4bit repos ship scales/biases as either F16
    // (older Llama / Mixtral) or BF16 (newer Qwen3-MoE). Both are
    // 2 bytes per element; the dequant loop decodes the right f16/bf16
    // bit pattern based on the actual dtype.
    anyhow::ensure!(
        matches!(s_dtype, DType::F16 | DType::BF16),
        "affine_dequant_b4: `{prefix}.scales` dtype is {s_dtype} (expected F16 or BF16)",
    );
    anyhow::ensure!(
        b_dtype == s_dtype,
        "affine_dequant_b4: `{prefix}.biases` dtype is {b_dtype} (expected same as scales {s_dtype})",
    );
    anyhow::ensure!(
        w_shape.len() == 2,
        "affine_dequant_b4: packed weight shape rank {} (expected 2)",
        w_shape.len(),
    );

    // pack_factor = 32 / bits (8 for bits=4 U32-packed nibbles; 4 for bits=8
    // U32-packed bytes). The packed weight is `[N, K / pack_factor]`.
    let pack_factor = (32 / bits) as usize;
    let n = w_shape[0];
    let k = w_shape[1] * pack_factor;
    anyhow::ensure!(
        k.is_multiple_of(group_size as usize),
        "affine_dequant_b4: K={k} not divisible by group_size={group_size}"
    );
    anyhow::ensure!(
        s_shape == [n, k / group_size as usize],
        "affine_dequant_b4: scales shape {:?} != [{n}, {}]",
        s_shape,
        k / group_size as usize,
    );
    anyhow::ensure!(
        b_shape == s_shape,
        "affine_dequant_b4: biases shape {:?} != scales shape {:?}",
        b_shape,
        s_shape,
    );

    let out_bytes = affine_dequant_b4_to_dtype(
        &w_bytes, &s_bytes, &b_bytes, n, k, group_size, bits, s_dtype, dtype_out,
    )?;
    Ok((out_bytes, n, k))
}

/// Dequantize one MLX-affine int4 prefix into a dense `[N, K]` device tensor.
/// Wraps [`affine_dequant_b4_bytes`] + `alloc_packed_from_host`.
///
/// `pub` so the cuda-crate `GpuWeights::take_affine_dequant_b4` thin wrapper
/// (used by the metal-cfg `AffineFusedMoELayer` router-gate load, which still
/// names `GpuWeights`) can delegate here through the `WeightSource` seam.
pub fn affine_dequant_b4<W: WeightSource + ?Sized>(
    weights: &mut W,
    prefix: &str,
    group_size: u32,
    bits: u32,
    dtype_out: DType,
) -> Result<GpuTensor> {
    let (out_bytes, n, k) = affine_dequant_b4_bytes(weights, prefix, group_size, bits, dtype_out)?;
    weights.alloc_packed_from_host(&out_bytes, &[n, k], dtype_out)
}

/// Concat sibling of [`affine_dequant_b4`]: CPU-dequantizes each prefix's
/// affine triple, byte-concats the `[N, K]` results along dim 0 into one
/// packed buffer, then `alloc_packed_from_host`s the fused `[sum(N), K]`
/// dense weight. All sources must share `K`, `group_size`, `bits`, `dtype_out`.
fn affine_dequant_b4_concat<W: WeightSource + ?Sized>(
    weights: &mut W,
    prefixes: &[&str],
    group_size: u32,
    bits: u32,
    dtype_out: DType,
    expected_in_features: u32,
) -> Result<GpuTensor> {
    anyhow::ensure!(
        !prefixes.is_empty(),
        "affine_dequant_b4_concat: empty prefix list"
    );
    let elem_size = match dtype_out {
        DType::F16 | DType::BF16 => 2,
        _ => anyhow::bail!(
            "affine_dequant_b4_concat: dtype_out must be F16 or BF16, got {dtype_out}"
        ),
    };

    let mut packed: Vec<u8> = Vec::new();
    let mut total_n: usize = 0;
    let mut k_shared: Option<usize> = None;
    for prefix in prefixes {
        let (bytes, n, k) = affine_dequant_b4_bytes(weights, prefix, group_size, bits, dtype_out)?;
        // Alignment check: reject a checkpoint whose packed `.weight` implies
        // a K that disagrees with the compiled preset's declared in_features
        // (i.e. it was quantized at different bits than this build expects).
        anyhow::ensure!(
            k == expected_in_features as usize,
            "affine dequant checkpoint/preset mismatch at `{prefix}`: packed `.weight` implies \
             in_features={k} at bits={bits} (pack_factor={}); this build expects \
             in_features={expected_in_features}. The checkpoint is quantized at a different \
             bit-width than this build's preset.",
            32 / bits,
        );
        if let Some(prev_k) = k_shared {
            anyhow::ensure!(
                prev_k == k,
                "affine_dequant_b4_concat: in_features mismatch across prefixes \
                 ({prev_k} vs {k} at `{prefix}`)"
            );
        } else {
            k_shared = Some(k);
        }
        total_n += n;
        // Sanity: byte length matches [N, K] dtype_out layout.
        anyhow::ensure!(
            bytes.len() == n * k * elem_size,
            "affine_dequant_b4_concat: `{prefix}` produced {} bytes; expected {}",
            bytes.len(),
            n * k * elem_size,
        );
        packed.extend_from_slice(&bytes);
    }
    let k = k_shared.expect("affine_dequant_b4_concat: prefixes non-empty above");
    weights.alloc_packed_from_host(&packed, &[total_n, k], dtype_out)
}

// ---------------------------------------------------------------------------
// AffineQuantLinear loader (MLX-native int4 affine — Metal-only)
// ---------------------------------------------------------------------------

/// Load extension trait on the neutral [`AffineQuantLinear`]. Defined +
/// impl'd here (orphan rule) since the type lives in `scratchy-layers`.
pub trait MetalAffineQuantOps {
    fn load<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl MetalAffineQuantOps for AffineQuantLinear {
    /// Load from `GpuWeights` by prefix. Reads `<prefix>.weight`
    /// (`U32`, `[N, K/pack_factor]`), `<prefix>.scales` (`F16`,
    /// `[N, K/group_size]`), `<prefix>.biases` (`F16`, same shape),
    /// and optional `<prefix>.bias` (model dtype, `[N]`).
    ///
    /// `group_size` and `bits` come from the model's
    /// `quantization_config` (parsed at compile time by the macro);
    /// `in_features` / `out_features` are derived from the weight
    /// tensor's shape, then aligned against `expected_in_features` (K
    /// from the arch manifest) so a checkpoint quantized at different
    /// bits/group_size than this build's preset is rejected rather than
    /// silently mis-shaped.
    fn load<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self> {
        let weight = weights.take(&format!("{prefix}.weight"))?;
        // Scales / biases ship F16 on every mlx-community 4bit repo
        // sampled in P0; scratchy-target-metal's qmv / qmm_t / qvm / qmm_n
        // kernels now read them as `T_scale = half` regardless of the
        // activation dtype and cast to `T_act` in-register.
        // Use `take_keep_dtype` to skip the loader-side F16→BF16 cast
        // that P1-P6 silently inherited from `set_target_dtype(BF16)`
        // — that path truncated 3 mantissa bits per scale (10→7) and
        // was the P10 late-token drift contributor this repair fixes.
        let scales = weights.take_keep_dtype(&format!("{prefix}.scales"))?;
        let affine_biases = weights.take_keep_dtype(&format!("{prefix}.biases"))?;
        let bias_name = format!("{prefix}.bias");
        let linear_bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };
        let pack_factor = (32 / bits) as usize;
        let out_features = weight.dim(0);
        let in_features = weight.dim(1) * pack_factor;
        // Alignment check: the on-disk packed `.weight` must have exactly
        // `K / pack_factor` columns for the compiled `bits`. If it doesn't,
        // the checkpoint is quantized at a different bit-width than this
        // build's quantization preset — reject it instead of running the
        // qmv kernel against a mis-derived in_features (silent garbage).
        anyhow::ensure!(
            in_features == expected_in_features as usize,
            "affine quant checkpoint/preset mismatch at `{prefix}`: packed `.weight` is \
             [{out_features}, {}], implying in_features={in_features} at bits={bits} \
             (pack_factor={pack_factor}); this build expects in_features={expected_in_features}. \
             The checkpoint is quantized at a different bit-width than this build's preset.",
            weight.dim(1),
        );
        Ok(Self {
            weight,
            scales,
            affine_biases,
            linear_bias,
            in_features,
            out_features,
            group_size,
            bits,
        })
    }
}

// ---------------------------------------------------------------------------
// Nvfp4Linear loader — NVIDIA ModelOpt NVFP4 4-bit linear (Metal-only)
// ---------------------------------------------------------------------------

/// Decode one `float8_e4m3fn` byte to `f32`. Format: 1 sign / 4 exponent
/// (bias 7) / 3 mantissa. e4m3fn has no infinities; the only NaN is the
/// `S.1111.111` pattern (others in the `1111` exponent are normal numbers,
/// max magnitude 448). Used to fold the per-block NVFP4 weight scales into
/// `f32` at load time. Mirrors `torch.float8_e4m3fn → float32`.
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    let mag = if exp == 0 {
        // Subnormal: mant * 2^(1-7) / 8 = mant * 2^(-9).
        (mant as f32) * 2.0f32.powi(-9)
    } else {
        // Normal: (1 + mant/8) * 2^(exp-7).
        (1.0 + (mant as f32) / 8.0) * 2.0f32.powi(exp - 7)
    };
    sign * mag
}

/// Load extension trait on the neutral [`Nvfp4Linear`].
pub trait MetalNvfp4Ops {
    fn load<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
    ) -> Result<Self>
    where
        Self: Sized;

    fn load_concat<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefixes: &[&str],
        group_size: u32,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl MetalNvfp4Ops for Nvfp4Linear {
    /// Load from `GpuWeights` by prefix. Reads `<prefix>.weight` (`U8`,
    /// `[N, K/2]`), `<prefix>.weight_scale` (`float8_e4m3`, `[N, K/gs]`),
    /// `<prefix>.weight_scale_2` (`f32` scalar / per-partition → `max`),
    /// and optional `<prefix>.bias`. Folds the two-level scale into a
    /// single `F16` per-group `scales` tensor on the CPU and uploads it.
    fn load<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
    ) -> Result<Self> {
        // Packed E2M1 codes — keep `U8`, no float cast.
        let weight = weights.take(&format!("{prefix}.weight"))?;

        // Per-block scale: float8_e4m3, [N, K/group_size]. Read raw bytes.
        let (sf_bytes, sf_shape, sf_dtype) = weights.take_cpu(&format!("{prefix}.weight_scale"))?;
        anyhow::ensure!(
            sf_dtype == DType::Fp8E4m3,
            "NVFP4 {prefix}.weight_scale must be float8_e4m3, got {sf_dtype}"
        );

        // Per-tensor global scale: f32 scalar, or per-partition vector
        // collapsed via `max` (matches Python `weight_scale_2.max()`).
        // ModelOpt names it `weight_scale_2`; compressed-tensors uses
        // `weight_global_scale` — accept either.
        let gscale_name = if weights.contains(&format!("{prefix}.weight_scale_2")) {
            format!("{prefix}.weight_scale_2")
        } else {
            format!("{prefix}.weight_global_scale")
        };
        let gscale = weights
            .take_to_cpu_f32(&gscale_name)?
            .into_iter()
            .fold(f32::NEG_INFINITY, f32::max);
        anyhow::ensure!(
            gscale.is_finite() && gscale != 0.0,
            "NVFP4 {gscale_name} global scale is {gscale}, expected finite non-zero"
        );

        // Fold: scales[n,g] = e4m3_to_f32(sf[n,g]) * gscale, stored F16.
        // (`weight_scale_2` is the per-tensor multiplier — the kernel
        // reconstructs `w = E2M1[code] * e4m3(weight_scale) * weight_scale_2`.)
        let folded: Vec<u8> = sf_bytes
            .iter()
            .flat_map(|&b| half::f16::from_f32(e4m3_to_f32(b) * gscale).to_le_bytes())
            .collect();
        let scales = weights.alloc_packed_from_host(&folded, &sf_shape, DType::F16)?;

        let bias_name = format!("{prefix}.bias");
        let linear_bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };

        let out_features = weight.dim(0);
        // Two E2M1 codes per packed byte.
        let in_features = weight.dim(1) * 2;
        Ok(Self {
            weight,
            scales,
            linear_bias,
            in_features,
            out_features,
            group_size,
            bits: 4,
        })
    }

    /// Load a fused (qkv_proj / gate_up_proj) NVFP4 linear from multiple
    /// on-disk prefixes. Each prefix is folded independently (its own
    /// global scale baked into the per-group F16 scales), then the
    /// packed weights and folded scales are byte-concatenated along
    /// dim 0 (out_features) — valid because dim 0 is the outermost axis,
    /// so rows are contiguous and per-row dequant is unaffected by the
    /// concat. No fused linear bias (ModelOpt fused projections ship
    /// none); panics if a prefix carries a `.bias` so it isn't silently
    /// dropped.
    fn load_concat<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefixes: &[&str],
        group_size: u32,
    ) -> Result<Self> {
        anyhow::ensure!(
            !prefixes.is_empty(),
            "Nvfp4Linear::load_concat: empty prefix list"
        );
        let mut weight_bytes: Vec<u8> = Vec::new();
        let mut scale_bytes: Vec<u8> = Vec::new();
        let mut total_n = 0usize;
        let mut packed_k: Option<usize> = None; // K/2
        let mut scale_cols: Option<usize> = None; // K/group_size
        for prefix in prefixes {
            anyhow::ensure!(
                !weights.contains(&format!("{prefix}.bias")),
                "NVFP4 fused concat {prefix}.bias present — fused-projection bias not supported"
            );
            let (w_bytes, w_shape, w_dtype) = weights.take_cpu(&format!("{prefix}.weight"))?;
            anyhow::ensure!(
                w_dtype == DType::U8 && w_shape.len() == 2,
                "NVFP4 {prefix}.weight must be U8 [N, K/2], got {w_dtype} {w_shape:?}"
            );
            let (ni, pk) = (w_shape[0], w_shape[1]);
            match packed_k {
                Some(pk0) => anyhow::ensure!(
                    pk == pk0,
                    "NVFP4 concat K mismatch: {prefix} packed_k {pk} != {pk0}"
                ),
                None => packed_k = Some(pk),
            }

            let (sf_bytes, sf_shape, sf_dtype) =
                weights.take_cpu(&format!("{prefix}.weight_scale"))?;
            anyhow::ensure!(
                sf_dtype == DType::Fp8E4m3 && sf_shape.len() == 2 && sf_shape[0] == ni,
                "NVFP4 {prefix}.weight_scale must be float8_e4m3 [{ni}, K/gs], got {sf_dtype} {sf_shape:?}"
            );
            match scale_cols {
                Some(sc0) => anyhow::ensure!(
                    sf_shape[1] == sc0,
                    "NVFP4 concat scale-cols mismatch: {prefix} {} != {sc0}",
                    sf_shape[1]
                ),
                None => scale_cols = Some(sf_shape[1]),
            }

            let gscale_name = if weights.contains(&format!("{prefix}.weight_scale_2")) {
                format!("{prefix}.weight_scale_2")
            } else {
                format!("{prefix}.weight_global_scale")
            };
            let gscale = weights
                .take_to_cpu_f32(&gscale_name)?
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max);
            anyhow::ensure!(
                gscale.is_finite() && gscale != 0.0,
                "NVFP4 {gscale_name} global scale is {gscale}, expected finite non-zero"
            );

            // dim-0 byte concat: append this prefix's rows.
            weight_bytes.extend_from_slice(&w_bytes);
            scale_bytes.extend(
                sf_bytes
                    .iter()
                    .flat_map(|&b| half::f16::from_f32(e4m3_to_f32(b) * gscale).to_le_bytes()),
            );
            total_n += ni;
        }
        let packed_k = packed_k.expect("at least one prefix");
        let scale_cols = scale_cols.expect("at least one prefix");
        let weight =
            weights.alloc_packed_from_host(&weight_bytes, &[total_n, packed_k], DType::U8)?;
        let scales =
            weights.alloc_packed_from_host(&scale_bytes, &[total_n, scale_cols], DType::F16)?;
        Ok(Self {
            weight,
            scales,
            linear_bias: None,
            in_features: packed_k * 2,
            out_features: total_n,
            group_size,
            bits: 4,
        })
    }
}

// ---------------------------------------------------------------------------
// AffineQuantEmbedding loader — MLX-affine int4 quantized token embedding
// ---------------------------------------------------------------------------

/// Load extension trait on the neutral [`AffineQuantEmbedding`].
pub trait MetalAffineEmbedOps {
    fn load<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl MetalAffineEmbedOps for AffineQuantEmbedding {
    /// Load from `GpuWeights` by prefix. Reads `<prefix>.weight`
    /// (`U32`, `[vocab, hidden / pack_factor]`), `<prefix>.scales`
    /// (`F16`, `[vocab, hidden / group_size]`), `<prefix>.biases`
    /// (`F16`, same shape).
    ///
    /// `group_size` and `bits` come from the model's
    /// `quantization_config` (parsed at compile time by the macro);
    /// `vocab_size` / `hidden_size` are derived from the weight
    /// tensor's shape.
    fn load<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self> {
        let weight = weights.take(&format!("{prefix}.weight"))?;
        // Scales / biases ship F16 on every mlx-community 4bit repo
        // sampled in P0; scratchy-target-metal's affine_embed kernel reads
        // them as `T_scale = half` and casts to T_act in-register.
        // `take_keep_dtype` skips the
        // loader-side F16→BF16 cast — see `AffineQuantLinear::load`
        // for the full reasoning.
        let scales = weights.take_keep_dtype(&format!("{prefix}.scales"))?;
        let affine_biases = weights.take_keep_dtype(&format!("{prefix}.biases"))?;
        let pack_factor = (32 / bits) as usize;
        let vocab_size = weight.dim(0);
        let hidden_size = weight.dim(1) * pack_factor;
        Ok(Self {
            weight,
            scales,
            affine_biases,
            vocab_size,
            hidden_size,
            group_size,
            bits,
        })
    }
}

// ---------------------------------------------------------------------------
// MetalLinearLayerOps — metal-only `LinearLayer` quant constructors
// ---------------------------------------------------------------------------

/// Metal-only forward-time quant constructors on the neutral [`LinearLayer`]
/// enum. Defined + impl'd here (orphan rule) since `LinearLayer` lives in
/// `scratchy-layers`; each wraps the corresponding metal payload loader above
/// (`AffineQuantLinear::load` / `Nvfp4Linear::load{,_concat}`), which already
/// `load()` against the neutral [`WeightSource`] seam — so these need not name
/// `GpuWeights` (which lives in targets/cuda, above this crate).
///
/// Re-exported from `scratchy_target_cuda::layers` so the macro-emitted
/// `LinearLayer::load_affine_quant(...)` / `load_nvfp4_quant(...)` path-syntax
/// calls (in model crates, which depend on targets/cuda but not on
/// targets/metal directly) resolve.
pub trait MetalLinearLayerOps {
    fn load_affine_quant<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self>
    where
        Self: Sized;

    fn load_nvfp4_quant<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
    ) -> Result<Self>
    where
        Self: Sized;

    fn load_nvfp4_quant_concat<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefixes: &[&str],
        group_size: u32,
    ) -> Result<Self>
    where
        Self: Sized;

    fn load_affine_dequant_as_dense<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self>
    where
        Self: Sized;

    fn load_affine_dequant_concat_as_dense<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefixes: &[&str],
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl MetalLinearLayerOps for LinearLayer {
    /// Load an MLX-affine int4 quantized linear layer (Metal-only).
    /// Reads `<prefix>.weight` + `<prefix>.scales` + `<prefix>.biases`
    /// (and optional `<prefix>.bias`) from the safetensors. Wraps
    /// `AffineQuantLinear::load`. Forward-time qmv dispatch consumer
    /// (P3+) — the macro currently emits `load_affine_dequant_as_dense`
    /// instead, which dequantizes at load time and returns `Dense`.
    fn load_affine_quant<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self> {
        Ok(Self::AffineQuant(Box::new(AffineQuantLinear::load(
            weights,
            prefix,
            group_size,
            bits,
            expected_in_features,
        )?)))
    }

    /// Load an NVFP4 4-bit quantized linear by safetensors prefix
    /// (Metal-only). Reads the `.weight` / `.weight_scale` /
    /// `.weight_scale_2` triple and optional `.bias` via
    /// [`Nvfp4Linear::load`].
    fn load_nvfp4_quant<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
    ) -> Result<Self> {
        Ok(Self::Nvfp4(Box::new(Nvfp4Linear::load(
            weights, prefix, group_size,
        )?)))
    }

    /// Load a fused (qkv_proj / gate_up_proj) NVFP4 linear from multiple
    /// safetensors prefixes (Metal-only). See [`Nvfp4Linear::load_concat`].
    fn load_nvfp4_quant_concat<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefixes: &[&str],
        group_size: u32,
    ) -> Result<Self> {
        Ok(Self::Nvfp4(Box::new(Nvfp4Linear::load_concat(
            weights, prefixes, group_size,
        )?)))
    }

    /// MLX-affine int4 → BF16 dequantize-at-load (Metal-only).
    ///
    /// CPU-dequantizes `<prefix>.{weight,scales,biases}` via the neutral
    /// [`affine_dequant_b4`] helper, producing a single `[N, K]` BF16 tensor
    /// the rest of the forward path consumes as a normal `Dense` linear. This
    /// is INT4 P2's slow-reference path — kernel-validated against the Metal
    /// `affine_dequantize` shader in
    /// `scratchy-target-metal/tests/quantized_dequantize_test.rs`. P3 will
    /// introduce forward-time qmv dispatch and the macro will switch back to
    /// [`Self::load_affine_quant`].
    ///
    /// The optional fp linear-layer bias (`<prefix>.bias`, absent on Llama-3.2
    /// / Qwen3-bf16) is loaded as-is into the Dense layer's bias slot.
    fn load_affine_dequant_as_dense<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self> {
        let weight = affine_dequant_b4(weights, prefix, group_size, bits, DType::BF16)?;
        let bias_name = format!("{prefix}.bias");
        let bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };
        Ok(Self::Dense(Linear::new(weight, bias)))
    }

    /// Fused-concat sibling of [`Self::load_affine_dequant_as_dense`] for
    /// `gate_up_proj` / `qkv_proj` accessors whose source weights are
    /// MLX-affine on disk (Metal-only).
    ///
    /// mlx-community 4bit repos ship gate_proj / up_proj (and q/k/v_proj) as
    /// separate affine triples; the forward DSL fuses them into a single
    /// linear so the metal `FusedGateUpSiluMul` impl can match. We CPU-dequant
    /// each prefix, byte-concat the resulting `[N, K]` BF16 chunks along dim 0,
    /// and wrap the fused `[sum(N), K]` weight as Dense. No per-prefix bias is
    /// supported here — every affine accessor in the int4 P2 coverage matrix
    /// has bias=false at the fused level (matches the dense
    /// `load_dense_concat_packed` no-bias case for these same fuses).
    fn load_affine_dequant_concat_as_dense<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefixes: &[&str],
        group_size: u32,
        bits: u32,
        expected_in_features: u32,
    ) -> Result<Self> {
        let weight = affine_dequant_b4_concat(
            weights,
            prefixes,
            group_size,
            bits,
            DType::BF16,
            expected_in_features,
        )?;
        Ok(Self::Dense(Linear::new(weight, None)))
    }
}

// ---------------------------------------------------------------------------
// MetalEmbeddingOps — metal-only `Embedding` dequant-at-load constructor
// ---------------------------------------------------------------------------

/// Metal-only dequant-at-load constructor on the neutral [`Embedding`] type.
/// Defined + impl'd here (orphan rule) since `Embedding` lives in
/// `scratchy-layers`. Re-exported from `scratchy_target_cuda::layers` so any
/// macro-emitted / compiler `Embedding::load_affine_dequant(...)` path-syntax
/// call resolves (model crates depend on targets/cuda, not targets/metal).
pub trait MetalEmbeddingOps {
    fn load_affine_dequant<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl MetalEmbeddingOps for Embedding {
    /// MLX-affine int4 → BF16 dequantize-at-load (Metal-only).
    ///
    /// Sibling of [`MetalLinearLayerOps::load_affine_dequant_as_dense`] for
    /// quantized token embeddings. CPU-dequantizes `<prefix>.{weight,scales,
    /// biases}` via the neutral [`affine_dequant_b4`] helper and wraps the
    /// resulting `[vocab_size, hidden_size]` BF16 tensor in an `Embedding`.
    fn load_affine_dequant<W: WeightSource + ?Sized>(
        weights: &mut W,
        prefix: &str,
        group_size: u32,
        bits: u32,
    ) -> Result<Self> {
        let weight = affine_dequant_b4(weights, prefix, group_size, bits, DType::BF16)?;
        Ok(Self::new(weight))
    }
}

#[cfg(test)]
mod alignment_tests {
    use super::*;
    use std::collections::HashMap;

    /// CPU-only [`WeightSource`]: `take_cpu` returns pre-staged bytes; the
    /// device methods are unreachable before the concat alignment check
    /// fires. `alloc_packed_from_host` returns a sentinel error so the
    /// matched-width case can prove it got past the check.
    struct CpuMock {
        tensors: HashMap<String, (Vec<u8>, Vec<usize>, DType)>,
    }

    impl WeightSource for CpuMock {
        fn take(&mut self, name: &str) -> Result<GpuTensor> {
            anyhow::bail!("mock: take({name}) unreachable in this test")
        }
        fn take_keep_dtype(&mut self, name: &str) -> Result<GpuTensor> {
            anyhow::bail!("mock: take_keep_dtype({name}) unreachable in this test")
        }
        fn take_cpu(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, DType)> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("mock: missing tensor `{name}`"))
        }
        fn take_to_cpu_f32(&mut self, name: &str) -> Result<Vec<f32>> {
            anyhow::bail!("mock: take_to_cpu_f32({name}) unreachable in this test")
        }
        fn contains(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }
        fn alloc_packed_from_host(
            &mut self,
            _data: &[u8],
            _shape: &[usize],
            _dtype: DType,
        ) -> Result<GpuTensor> {
            anyhow::bail!("mock alloc reached")
        }
    }

    /// A self-consistent bits=4 / group_size=32 affine triple with K=32
    /// (n=2). Passes `affine_dequant_b4_bytes`' internal shape checks, so
    /// only the concat alignment check can reject it.
    fn staged_k32() -> CpuMock {
        // bits=4 → pack_factor 8; w_shape=[n=2, k/8=4]; scales/biases [n, k/32=1].
        let n = 2usize;
        let w_cols = 4usize; // 4 u32 columns → k = 4 * 8 = 32
        let g_cols = 1usize; // k / group_size = 32 / 32
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.weight".to_string(),
            (vec![0u8; n * w_cols * 4], vec![n, w_cols], DType::U32),
        );
        tensors.insert(
            "blk.scales".to_string(),
            (vec![0u8; n * g_cols * 2], vec![n, g_cols], DType::BF16),
        );
        tensors.insert(
            "blk.biases".to_string(),
            (vec![0u8; n * g_cols * 2], vec![n, g_cols], DType::BF16),
        );
        CpuMock { tensors }
    }

    #[test]
    fn concat_rejects_in_features_mismatch() {
        let mut w = staged_k32();
        // On-disk K is 32, but the compiled preset expects 64 (e.g. a
        // checkpoint re-quantized at a different bit-width) → reject.
        let err = affine_dequant_b4_concat(&mut w, &["blk"], 32, 4, DType::BF16, 64)
            .expect_err("expected a checkpoint/preset mismatch rejection");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("checkpoint/preset mismatch") && msg.contains("in_features=32"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn concat_accepts_matching_in_features() {
        let mut w = staged_k32();
        // On-disk K is 32 and the preset expects 32 → passes the alignment
        // check and proceeds to the device alloc (the mock's sentinel),
        // proving a well-matched checkpoint is not falsely rejected.
        let err = affine_dequant_b4_concat(&mut w, &["blk"], 32, 4, DType::BF16, 32)
            .expect_err("mock has no device; the alloc sentinel should surface");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("mock alloc reached"),
            "matching in_features should pass the alignment check; got: {msg}"
        );
    }
}
