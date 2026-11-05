// SPDX-License-Identifier: Apache-2.0
//! Rotary positional embedding cache and rope-scaling config.
//!
//! Backend-neutral half of the RoPE cache: the cos/sin trig math is
//! identical across backends, and the stream-free `*_from_gpuweights`
//! constructors upload through the active [`scratchy_tensors::WeightSource`]
//! (the `GpuWeights` allocator), naming no backend type. The YaRN host
//! math lives in [`yarn_cos_sin_table`].

use crate::weights::GpuWeights;
use anyhow::Result;
use rayon::prelude::*;
use scratchy_tensors::{DType, DeviceAllocator, GpuTensor, WeightSource};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parsed LLaMA config (mirrors the HF config).
/// Llama 3.x rope_scaling parameters.
#[derive(Debug, Clone)]
pub struct Llama3RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

/// DeepSeek-V2 YaRN NTK-by-parts scaling parameters.
/// Mirrors the `rope_scaling` subobject in the HF config JSON.
#[derive(Debug, Clone)]
pub struct YarnRopeScaling {
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    pub original_max_position_embeddings: usize,
}

/// Phi-3 / Phi-3.5 LongRoPE (su-scaling) parameters. Mirrors the
/// fields Python vLLM's `Phi3LongRoPEScaledRotaryEmbedding`
/// consumes 1:1 (see
/// `vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.py`).
///
/// Selection between `short_factor` and `long_factor` is a GLOBAL
/// flag keyed on `max_model_len > original_max_position_embeddings`
/// (not a per-position switchover). The caller supplies
/// `max_model_len` — typically the HF config's
/// `max_position_embeddings` unless the user passes a smaller
/// `--max-model-len` — and the constructor bakes exactly one of
/// (short_factor, short_mscale) or (long_factor, long_mscale) into
/// the cache for every position.
///
/// `short_mscale` / `long_mscale` scale the cos/sin values
/// (equivalent to scaling attention logits). When the HF config
/// omits them, Python falls back to the Phi-3 paper formula
/// `sqrt(1 + ln(max_pos/original_max_pos) / ln(original_max_pos))`
/// for both; the caller must materialize that default here.
#[derive(Debug, Clone)]
pub struct LongRopeScaling {
    pub short_factor: Vec<f64>,
    pub long_factor: Vec<f64>,
    pub original_max_position_embeddings: usize,
    pub short_mscale: f64,
    pub long_mscale: f64,
}

/// The two position lengths a LongRoPE cache needs, which are NOT the
/// same number: `max_pos` is how many rows to build (the clamped
/// context), `max_model_len` is what selects the short-vs-long
/// factor/mscale pair. Passing them as one value keeps a caller from
/// swapping them — both are `usize` and only one is the context the
/// model was configured with.
#[derive(Debug, Clone, Copy)]
pub struct LongRopeExtent {
    pub max_pos: usize,
    pub max_model_len: usize,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    pub llama3_rope_scaling: Option<Llama3RopeScaling>,
}

// ---------------------------------------------------------------------------
// RoPE cache
// ---------------------------------------------------------------------------

/// Pre-computed rotary embedding cos/sin cache on GPU.
pub struct RotaryCache {
    /// `[max_pos, rotary_dim]` combined cos|sin cache (used by RoPE kernels).
    pub cos_sin_cache: GpuTensor,
    /// `[max_pos, rotary_dim/2]` separate cos cache (used by FA2 fused RoPE).
    pub cos_cache: GpuTensor,
    /// `[max_pos, rotary_dim/2]` separate sin cache (used by FA2 fused RoPE).
    pub sin_cache: GpuTensor,
    pub head_dim: usize,
    /// MRoPE section lengths (T/H/W) for arches with multimodal RoPE
    /// (Qwen2-VL, Qwen2.5-VL). `None` for every text-only arch — the
    /// kernel takes the existing 1D-positions fast path. `Some([a,b,c])`
    /// with `a+b+c == rotary_dim/2` selects the MRoPE path: kernel reads
    /// three position values per token and dispatches each rotary pair
    /// through the section that owns it. Plumbed through Phase A1; the
    /// MRoPE math path itself lands in Phase A2. See
    /// `~/.claude/plans/distributed-mapping-map.md`.
    pub mrope_section: Option<[u32; 3]>,
}

/// Cast an f32 cos/sin buffer to `dtype` (F32 / F16 / BF16) and
/// upload via the active [`WeightSource`] allocator. Backend-neutral
/// — used by the `*_from_gpuweights` constructors.
fn upload_via_gpuweights<W: WeightSource + ?Sized>(
    weights: &mut W,
    cache: &[f32],
    shape: &[usize],
    dtype: DType,
) -> Result<GpuTensor> {
    // Converted in parallel into one typed buffer, then viewed as bytes. Pushing each element's bytes
    // onto a growing `Vec<u8>` was ~130 ms of gemma-4-26b's load: its tables run to 262144 positions.
    let bits: Vec<u16> = match dtype {
        DType::F32 => {
            // SAFETY: any initialized `f32` is four initialized bytes, and `u8` has alignment 1.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    cache.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(cache),
                )
            };
            return weights.alloc_packed_from_host(bytes, shape, dtype);
        }
        DType::F16 => cache
            .par_iter()
            .map(|&f| half::f16::from_f32(f).to_bits())
            .collect(),
        DType::BF16 => cache
            .par_iter()
            .map(|&f| half::bf16::from_f32(f).to_bits())
            .collect(),
        _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
    };
    // SAFETY: as above, for `u16`.
    let bytes = unsafe {
        std::slice::from_raw_parts(bits.as_ptr().cast::<u8>(), std::mem::size_of_val(&bits[..]))
    };
    weights.alloc_packed_from_host(bytes, shape, dtype)
}

/// Rows per angle-addition block in [`write_cos_sin`].
const ROPE_BLOCK_ROWS: usize = 64;

/// The `[max_pos, rotary_dim]` cos|sin cache of `pos · inv_freqs[i]` in `dtype`, what every rope
/// kernel reads: cos in `[0, half)` of each row, sin in `[half, 2·half)`. Built straight in
/// `dtype`, with no f32 table in between, and in place where the device memory is host-visible
/// ([`GpuWeights::alloc_packed_with`]): a model's table covers every position it serves (gemma-4:
/// 262144 rows, 192 MiB of bf16) and is on every launch's critical path, where copying it up
/// costs as much as building it.
fn upload_cos_sin<A: DeviceAllocator>(
    weights: &mut GpuWeights<A>,
    inv_freqs: &[f64],
    max_pos: usize,
    rotary_dim: usize,
    dtype: DType,
) -> Result<GpuTensor> {
    weights.alloc_packed_with(&[max_pos, rotary_dim], dtype, |table| {
        write_cos_sin(table, inv_freqs, rotary_dim, dtype)
    })
}

/// Fill `table`, whole `rotary_dim`-wide rows of `dtype`, with [`upload_cos_sin`]'s cache.
///
/// By angle addition rather than a cos and a sin per entry. Row `p0 + k` of a
/// [`ROPE_BLOCK_ROWS`]-row block composes the block's first angle `a = p0·θ` with the offset
/// `b = k·θ`: `cos(a + b) = cos a·cos b − sin a·sin b`, `sin(a + b) = sin a·cos b + cos a·sin b`.
/// The trig is one pair per block and frequency, and an entry is four products. In f64 the composed
/// value is within ~3e-11 of the per-entry `cos(pos·θ)`, the order of that product's own rounding:
/// about one f32 in 10⁴ lands an ulp away, and rounding to bf16 removes even that — the bf16 tables
/// of gemma-4, Llama-3.2 and Qwen3.6 match the per-entry build bit for bit, every row.
fn write_cos_sin(
    table: &mut [u8],
    inv_freqs: &[f64],
    rotary_dim: usize,
    dtype: DType,
) -> Result<()> {
    match dtype {
        DType::F32 => cos_sin_rows(table, inv_freqs, rotary_dim, f32::to_ne_bytes),
        DType::F16 => cos_sin_rows(table, inv_freqs, rotary_dim, |x| {
            half::f16::from_f32(x).to_bits().to_ne_bytes()
        }),
        DType::BF16 => cos_sin_rows(table, inv_freqs, rotary_dim, |x| bf16_bits(x).to_ne_bytes()),
        _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
    }
    Ok(())
}

/// `x`'s bf16 bits, rounded to nearest, ties to even: `half::bf16::from_f32` for every finite `x`
/// (a cos or sin always is), without its NaN branch, so the fill loop vectorizes.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// [`write_cos_sin`] for an `N`-byte element `cast` from f32. The angles are kept as planes, sin
/// apart from cos, and each row's cos half is written before its sin half, so the loops vectorize.
fn cos_sin_rows<const N: usize>(
    table: &mut [u8],
    inv_freqs: &[f64],
    rotary_dim: usize,
    cast: impl Fn(f32) -> [u8; N] + Sync,
) {
    let half = inv_freqs.len();
    let row_bytes = rotary_dim * N;
    // sin and cos of each in-block offset's angle: `ROPE_BLOCK_ROWS` rows of `half`.
    let (offset_sin, offset_cos): (Vec<f64>, Vec<f64>) = (0..ROPE_BLOCK_ROWS)
        .flat_map(|k| inv_freqs.iter().map(move |&t| (k as f64 * t).sin_cos()))
        .unzip();
    table
        .par_chunks_mut(ROPE_BLOCK_ROWS * row_bytes)
        .enumerate()
        .for_each(|(block, rows)| {
            let p0 = (block * ROPE_BLOCK_ROWS) as f64;
            let (sin_a, cos_a): (Vec<f64>, Vec<f64>) =
                inv_freqs.iter().map(|&t| (p0 * t).sin_cos()).unzip();
            let rows = rows.chunks_exact_mut(row_bytes);
            let offsets = offset_sin
                .chunks_exact(half)
                .zip(offset_cos.chunks_exact(half));
            for (row, (sin_b, cos_b)) in rows.zip(offsets) {
                let (cos, sin) = row.split_at_mut(half * N);
                let a = sin_a.iter().zip(&cos_a);
                let b = sin_b.iter().zip(cos_b);
                for ((out, (&sa, &ca)), (&sb, &cb)) in cos
                    .as_chunks_mut::<N>()
                    .0
                    .iter_mut()
                    .zip(a.clone())
                    .zip(b.clone())
                {
                    *out = cast((ca * cb - sa * sb) as f32);
                }
                for ((out, (&sa, &ca)), (&sb, &cb)) in
                    sin.as_chunks_mut::<N>().0.iter_mut().zip(a).zip(b)
                {
                    *out = cast((sa * cb + ca * sb) as f32);
                }
            }
        });
}

// ---------------------------------------------------------------------------
// YaRN host math
// ---------------------------------------------------------------------------

/// YaRN magnitude-scaling factor — the attention-temperature term
/// DeepSeek's `DeepseekScalingRotaryEmbedding` bakes into cos/sin.
/// Python (`yarn_get_mscale`): `0.1 * mscale * ln(scale) + 1` for
/// `scale > 1`, else `1.0`.
pub fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// YaRN NTK-by-parts correction range: the frequency-index band over
/// which inv-freqs ramp between the original (extrapolating) frequency
/// and the fully-interpolated (÷`factor`) one. Returns
/// `(floor(low), ceil(high))` clamped to `[0, dim/2 - 1]` — Python's
/// `yarn_find_correction_range`.
pub fn yarn_find_correction_range(
    beta_fast: f64,
    beta_slow: f64,
    dim: usize,
    base: f64,
    original_max_pos: usize,
) -> (f64, f64) {
    let n_orig = original_max_pos as f64;
    let low =
        (n_orig / (beta_fast * 2.0 * std::f64::consts::PI)).ln() / (2.0 / dim as f64 * base.ln());
    let high =
        (n_orig / (beta_slow * 2.0 * std::f64::consts::PI)).ln() / (2.0 / dim as f64 * base.ln());
    (
        low.floor().max(0.0),
        high.ceil().min(dim as f64 / 2.0 - 1.0),
    )
}

/// Linear ramp over the first `dim/2` frequency indices: 0 below `low`
/// (keep the original high frequency), 1 above `high` (fully
/// interpolated), linear in between — Python's `yarn_linear_ramp_mask`.
pub fn yarn_linear_ramp_mask(low: f64, high: f64, dim: usize) -> Vec<f64> {
    let len = dim / 2;
    (0..len)
        .map(|i| {
            let t = i as f64;
            if low >= high {
                if t < low { 0.0 } else { 1.0 }
            } else if t < low {
                0.0
            } else if t > high {
                1.0
            } else {
                (t - low) / (high - low)
            }
        })
        .collect()
}

/// The `[max_pos, rope_head_dim]` f32 combined cos|sin table for YaRN
/// rope, behind [`RotaryCache::new_yarn_from_gpuweights`]. cos occupies `[0, half)`, sin `[half, rope_head_dim)` per
/// row, matching every other builder the rope kernels read.
pub fn yarn_cos_sin_table(
    rope_head_dim: usize,
    max_pos: usize,
    rope_theta: f64,
    yarn: &YarnRopeScaling,
) -> Vec<f32> {
    let half = rope_head_dim / 2;
    let factor = yarn.factor;
    let (low, high) = yarn_find_correction_range(
        yarn.beta_fast,
        yarn.beta_slow,
        rope_head_dim,
        rope_theta,
        yarn.original_max_position_embeddings,
    );
    let ramp = yarn_linear_ramp_mask(low, high, rope_head_dim);
    // Python's DeepseekScalingRotaryEmbedding bakes
    //   mscale = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
    // into the cos/sin cache (mscale_all_dim^2 is handled separately in the attention
    // softmax scale). For DeepSeek V2-Lite mscale==mscale_all_dim so this is 1.0;
    // for any config omitting the pair (gpt-oss) the parse defaults are HF's
    // 1.0/1.0, so the ratio is likewise 1.0 — plain interpolation.
    let mscale_num = yarn_get_mscale(factor, yarn.mscale);
    let mscale_den = if yarn.mscale_all_dim != 0.0 {
        yarn_get_mscale(factor, yarn.mscale_all_dim)
    } else {
        1.0
    };
    let mscale = (mscale_num / mscale_den) as f32;

    let inv_freqs: Vec<f64> = (0..half)
        .map(|i| {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rope_head_dim as f64);
            let freq_inter = freq / factor;
            // ramp[i]=0 → original (high-freq, small i), ramp[i]=1 → interpolated (low-freq, large i).
            // Matches Python: inv_freq = interp*ramp + extrap*(1-ramp)
            freq * (1.0 - ramp[i]) + freq_inter * ramp[i]
        })
        .collect();

    let mut cache = vec![0f32; max_pos * rope_head_dim];
    cache
        .par_chunks_mut(rope_head_dim)
        .enumerate()
        .for_each(|(pos, row)| {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                row[i] = angle.cos() as f32 * mscale;
                row[half + i] = angle.sin() as f32 * mscale;
            }
        });
    cache
}

impl RotaryCache {
    /// Backend-neutral, stream-free constructor. Computes the cos/sin
    /// tables on CPU and uploads them through `weights`' allocator
    /// ([`GpuWeights::alloc_packed_with`]), one table per cache.
    ///
    /// Currently covers the basic case: full rotary
    /// (`rotary_dim == head_dim`), no scaling or Llama3 scaling. The
    /// other rope families have their own constructors:
    /// [`Self::new_partial_from_gpuweights`] (partial rotary),
    /// [`Self::new_longrope_from_gpuweights`], and
    /// [`Self::new_yarn_from_gpuweights`].
    pub fn new_from_gpuweights<A: DeviceAllocator>(
        weights: &mut GpuWeights<A>,
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
    ) -> Result<Self> {
        // Full rotary is the `rotary_dim == head_dim` special case.
        Self::new_partial_from_gpuweights(
            weights,
            head_dim,
            head_dim,
            max_pos,
            rope_theta,
            llama3_scaling,
            dtype,
        )
    }

    /// Partial-rotary (and full, when `rotary_dim == head_dim`) cos/sin
    /// cache, built on CPU and uploaded through `weights`' allocator. Builds a `[max_pos, rotary_dim]`
    /// cache where `rotary_dim = partial_rotary_factor * head_dim` (e.g.
    /// Qwen3.5: `0.25 * 256 = 64`). The metal rope kernels read it
    /// with `ROT_DIM = rotary_dim` and pass the trailing
    /// `head_dim - rotary_dim` channels through unrotated. Covers
    /// no-scaling and Llama3 scaling; LongRoPE and YaRN have their own
    /// `*_from_gpuweights` constructors.
    pub fn new_partial_from_gpuweights<A: DeviceAllocator>(
        weights: &mut GpuWeights<A>,
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
    ) -> Result<Self> {
        let half = rotary_dim / 2;

        let inv_freqs: Vec<f64> = (0..half)
            .map(|i| {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64);
                if let Some(scaling) = llama3_scaling {
                    let old_context_len = scaling.original_max_position_embeddings as f64;
                    let low_freq_wavelen = old_context_len / scaling.low_freq_factor;
                    let high_freq_wavelen = old_context_len / scaling.high_freq_factor;
                    let wavelen = 2.0 * std::f64::consts::PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / scaling.factor
                    } else {
                        let smooth = (old_context_len / wavelen - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1.0 - smooth) * freq / scaling.factor + smooth * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        let cos_sin_cache = upload_cos_sin(weights, &inv_freqs, max_pos, rotary_dim, dtype)?;

        // Split `cos_cache` / `sin_cache` are wgpu-only — every metal
        // kernel reads the unified `cos_sin_cache` above. Building +
        // uploading two ~32 MB caches that nothing reads costs ~30-50ms
        // on Llama-3.2-3B at max_pos = 131072, so skip them.
        let (cos_cache, sin_cache) = (
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
        );

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }

    /// LongRoPE (Phi-3 `rope_scaling.type == "longrope"` / `"su"`)
    /// cos/sin cache, built on CPU and uploaded via the
    /// [`WeightSource`] (f32 throughout, matching Python's float32
    /// `inv_freq` path).
    ///
    /// LongRoPE differs from the plain/Llama3 builder in exactly two
    /// places: a per-channel `factor[k]` divides into the frequency
    /// base, and the cos/sin are scaled by `mscale`. Which
    /// factor/mscale PAIR applies is chosen once, here, by Python's
    /// `use_long_rope = max_model_len > original_max_position_embeddings`
    /// — vLLM builds both caches and indexes whichever; baking the
    /// selected one keeps a single cache and one kernel path.
    pub fn new_longrope_from_gpuweights<W: WeightSource + ?Sized>(
        weights: &mut W,
        head_dim: usize,
        rotary_dim: usize,
        extent: LongRopeExtent,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
    ) -> Result<Self> {
        let LongRopeExtent {
            max_pos,
            max_model_len,
        } = extent;
        let half = rotary_dim / 2;
        if longrope.short_factor.len() != half || longrope.long_factor.len() != half {
            anyhow::bail!(
                "LongRoPE factor length mismatch: short={}, long={}, expected {} (= rotary_dim/2)",
                longrope.short_factor.len(),
                longrope.long_factor.len(),
                half,
            );
        }
        let use_long_rope = max_model_len > longrope.original_max_position_embeddings;
        let (factor, mscale): (&[f64], f64) = if use_long_rope {
            (&longrope.long_factor, longrope.long_mscale)
        } else {
            (&longrope.short_factor, longrope.short_mscale)
        };

        let base_f32 = rope_theta as f32;
        let rotary_dim_f32 = rotary_dim as f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| {
                let exp = (2 * k) as f32 / rotary_dim_f32;
                1.0f32 / ((factor[k] as f32) * base_f32.powf(exp))
            })
            .collect();
        let mscale_f32 = mscale as f32;

        let mut cache = vec![0f32; max_pos * rotary_dim];
        cache
            .par_chunks_mut(rotary_dim)
            .enumerate()
            .for_each(|(pos, row)| {
                for i in 0..half {
                    let angle = pos as f32 * inv_freq[i];
                    row[i] = angle.cos() * mscale_f32;
                    row[half + i] = angle.sin() * mscale_f32;
                }
            });

        let cos_sin_cache = upload_via_gpuweights(weights, &cache, &[max_pos, rotary_dim], dtype)?;
        let (cos_cache, sin_cache) = (
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
        );
        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }

    /// YaRN (`rope_scaling.type == "yarn"` — DeepSeek-V2 MLA, gpt-oss)
    /// cos/sin cache, built on CPU and uploaded via the
    /// [`WeightSource`], through the shared [`yarn_cos_sin_table`].
    /// NTK-by-parts interpolation between the original frequencies and
    /// the ÷`factor` interpolated ones, ramped over the correction
    /// range; the cos/sin carry the mscale ratio, which is 1.0 for
    /// every config whose `mscale`/`mscale_all_dim` pair is equal —
    /// including all configs that omit the pair (the parse defaults
    /// are HF's 1.0/1.0).
    ///
    /// `rope_head_dim` is the rope-portion head dim: the full
    /// `head_dim` for standard attention, `qk_rope_head_dim` for MLA
    /// models.
    pub fn new_yarn_from_gpuweights<W: WeightSource + ?Sized>(
        weights: &mut W,
        rope_head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        yarn: &YarnRopeScaling,
        dtype: DType,
    ) -> Result<Self> {
        let cache = yarn_cos_sin_table(rope_head_dim, max_pos, rope_theta, yarn);
        let cos_sin_cache =
            upload_via_gpuweights(weights, &cache, &[max_pos, rope_head_dim], dtype)?;
        let (cos_cache, sin_cache) = (
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
        );
        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim: rope_head_dim,
            mrope_section: None,
        })
    }

    /// "Proportional" partial rope (Gemma4 global-attention layers).
    ///
    /// Like [`Self::new_partial_from_gpuweights`] the cache is
    /// `[max_pos, rotary_dim]` and only the first `rotary_dim` of
    /// `head_dim` rotate — but the frequency exponent denominator is
    /// the FULL `head_dim`, not `rotary_dim`:
    ///
    ///   `freq_i = 1 / theta^(2i / head_dim)`  for 2i < rotary_dim
    ///
    /// Faithful to mlx `ProportionalRoPE` (`rope_utils.py`):
    /// `exponents = arange(0, rotated_dims, 2) / dims` with the
    /// trailing freqs infinite (pass-through, which the ROT_DIM
    /// kernel tail already implements). Gemma4 global layers:
    /// head_dim 512, rotary_dim 128, theta 1e6.
    ///
    /// # Safety
    /// Same as `new_partial_from_gpuweights`.
    pub fn new_proportional_from_gpuweights<A: DeviceAllocator>(
        weights: &mut GpuWeights<A>,
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        dtype: DType,
    ) -> Result<Self> {
        assert!(
            rotary_dim <= head_dim,
            "proportional rope: rotary_dim {rotary_dim} > head_dim {head_dim}"
        );
        let half = rotary_dim / 2;
        let inv_freqs: Vec<f64> = (0..half)
            // Denominator = head_dim (NOT rotary_dim) — the defining
            // difference vs standard partial rope.
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();

        let cos_sin_cache = upload_cos_sin(weights, &inv_freqs, max_pos, rotary_dim, dtype)?;
        let (cos_cache, sin_cache) = (
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
            unsafe { GpuTensor::new(std::ptr::null_mut(), &[0usize], dtype) },
        );

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }
}

#[cfg(test)]
mod yarn_tests {
    use super::*;

    /// gpt-oss rope parameters: theta 150000, YaRN factor 32,
    /// original context 4096, beta 32/1, and NO mscale pair — the
    /// parse defaults (HF signature 1.0/1.0) make the mscale ratio
    /// 1.0, i.e. plain interpolation with no attention temperature.
    /// This is exactly what mlx-lm/vLLM compute for gpt-oss.
    fn gpt_oss_yarn() -> YarnRopeScaling {
        YarnRopeScaling {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 1.0,
            original_max_position_embeddings: 4096,
        }
    }

    const GPT_OSS_HEAD_DIM: usize = 64;
    const GPT_OSS_MAX_POS: usize = 131_072;
    const GPT_OSS_THETA: f64 = 150_000.0;

    /// The correction range for gpt-oss's parameters. freq indices
    /// 0..=8 keep the original frequency, 9..=17 ramp, 18..=31 are
    /// fully interpolated (÷32) — the floor/ceil law of
    /// `yarn_find_correction_range` decides the band edges, and an
    /// off-by-one here is a silent numerics bug in every position.
    #[test]
    fn yarn_correction_range_boundaries() {
        let (low, high) = yarn_find_correction_range(32.0, 1.0, 64, 150_000.0, 4096);
        assert_eq!(low, 8.0);
        assert_eq!(high, 18.0);
    }

    /// Independent reference for the YaRN angle at one (position,
    /// freq-index) pair, transcribed from the HF/vLLM
    /// `YarnRotaryEmbedding` formulas in a deliberately different
    /// spelling from the implementation (vLLM's
    /// `dim * ln(orig/(beta*2π)) / (2*ln(base))` correction form, the
    /// clamp-form ramp) — agreement is evidence about the math, not
    /// about shared code.
    fn reference_yarn_angle(pos: usize, i: usize) -> f64 {
        const DIM: f64 = 64.0;
        const BASE: f64 = 150_000.0;
        const FACTOR: f64 = 32.0;
        const ORIG: f64 = 4096.0;
        let correction =
            |beta: f64| DIM * (ORIG / (beta * 2.0 * std::f64::consts::PI)).ln() / (2.0 * BASE.ln());
        let low = correction(32.0).floor().max(0.0);
        let high = correction(1.0).ceil().min(DIM / 2.0 - 1.0);
        let t = i as f64;
        let ramp = ((t - low) / (high - low)).clamp(0.0, 1.0);
        let freq = BASE.powf(-2.0 * t / DIM);
        pos as f64 * (freq * (1.0 - ramp) + freq / FACTOR * ramp)
    }

    /// The built table against the reference at the boundary
    /// positions: 0/1 (identity), 4095/4096 (the original-context
    /// edge YaRN corrects around), 8192, and the last row of the
    /// gpt-oss context — every freq index, both halves.
    #[test]
    fn yarn_table_matches_reference_at_boundary_positions() {
        let table = yarn_cos_sin_table(
            GPT_OSS_HEAD_DIM,
            GPT_OSS_MAX_POS,
            GPT_OSS_THETA,
            &gpt_oss_yarn(),
        );
        let half = GPT_OSS_HEAD_DIM / 2;
        for pos in [0usize, 1, 4095, 4096, 8192, 131_071] {
            for i in 0..half {
                let angle = reference_yarn_angle(pos, i);
                let (cos_ref, sin_ref) = (angle.cos() as f32, angle.sin() as f32);
                let row = pos * GPT_OSS_HEAD_DIM;
                assert!(
                    (table[row + i] - cos_ref).abs() <= 1e-6,
                    "cos mismatch at pos {pos} freq {i}: {} vs {cos_ref}",
                    table[row + i]
                );
                assert!(
                    (table[row + half + i] - sin_ref).abs() <= 1e-6,
                    "sin mismatch at pos {pos} freq {i}: {} vs {sin_ref}",
                    table[row + half + i]
                );
            }
        }
    }

    /// With the HF-default mscale pair (any equal pair), the cos/sin
    /// are NOT scaled: position 0 is exactly (1.0, 0.0) and every
    /// entry stays within [-1, 1]. This is the regression pin for the
    /// mscale parse defaults — the pre-fix default (mscale_all_dim
    /// 0.0 → numerator-only) folded `0.1·ln(32)+1 ≈ 1.3466` into
    /// every value.
    #[test]
    fn yarn_hf_default_mscale_pair_is_plain_interpolation() {
        let table = yarn_cos_sin_table(
            GPT_OSS_HEAD_DIM,
            GPT_OSS_MAX_POS,
            GPT_OSS_THETA,
            &gpt_oss_yarn(),
        );
        let half = GPT_OSS_HEAD_DIM / 2;
        for i in 0..GPT_OSS_HEAD_DIM {
            assert_eq!(table[i], if i < half { 1.0 } else { 0.0 });
        }
        // No temperature anywhere: sample every 97th row (prime
        // stride) rather than all 131072 — the value law is per-entry.
        for pos in (0..GPT_OSS_MAX_POS).step_by(97) {
            for &v in &table[pos * GPT_OSS_HEAD_DIM..(pos + 1) * GPT_OSS_HEAD_DIM] {
                assert!(v.abs() <= 1.0, "cos/sin exceeds 1.0 at pos {pos}: {v}");
            }
        }
    }

    /// An UNEQUAL pair scales cos/sin by the mscale ratio, and
    /// `mscale_all_dim == 0.0` falls back to a denominator of 1.0
    /// (the `!= 0.0` guard). Position 0 isolates the ratio exactly:
    /// cos(0)·mscale, sin(0)·mscale.
    #[test]
    fn yarn_unequal_mscale_pair_scales_cos_sin() {
        let mut yarn = gpt_oss_yarn();
        yarn.mscale_all_dim = 0.707;
        let ratio = (yarn_get_mscale(32.0, 1.0) / yarn_get_mscale(32.0, 0.707)) as f32;
        let table = yarn_cos_sin_table(GPT_OSS_HEAD_DIM, 64, GPT_OSS_THETA, &yarn);
        let half = GPT_OSS_HEAD_DIM / 2;
        for i in 0..half {
            assert_eq!(table[i], ratio);
            assert_eq!(table[half + i], 0.0);
        }

        // mscale_all_dim == 0.0 → denominator 1.0.
        yarn.mscale_all_dim = 0.0;
        let table = yarn_cos_sin_table(GPT_OSS_HEAD_DIM, 64, GPT_OSS_THETA, &yarn);
        assert_eq!(table[0], yarn_get_mscale(32.0, 1.0) as f32);
    }
}

#[cfg(test)]
mod table_tests {
    use super::*;

    /// `1 / θ^(2i / denom)` for `half` frequencies.
    fn inv_freqs(theta: f64, half: usize, denom: usize) -> Vec<f64> {
        (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / denom as f64))
            .collect()
    }

    /// [`write_cos_sin`] into a host buffer: `max_pos` full-rotary rows of `dtype`.
    fn table(inv: &[f64], max_pos: usize, dtype: DType) -> Result<Vec<u8>> {
        let rotary_dim = 2 * inv.len();
        let mut table = vec![0u8; max_pos * rotary_dim * dtype.size_bytes()];
        write_cos_sin(&mut table, inv, rotary_dim, dtype)?;
        Ok(table)
    }

    /// The angle-addition table is the per-entry `cos/sin(pos·θ)` table in bf16, bit for bit, at
    /// the frequencies of gemma-4's sliding (θ 1e4, 256 dims) and global (proportional: 128 of 512
    /// dims, θ 1e6) tables, Llama-3.2's (θ 5e5, 64) and Qwen3.6's (θ 1e7, 64 of 256), across their
    /// longest context: every row of the first two blocks and of the last, and every 97th row
    /// (prime stride, so the sampled offsets cover each in-block position). The full tables were
    /// also compared, every row: no bf16 entry differs.
    #[test]
    fn angle_addition_matches_per_entry_trig_in_bf16() {
        let bf16 = |x: f64| half::bf16::from_f32(x as f32).to_bits().to_ne_bytes();
        for (inv, max_pos) in [
            (inv_freqs(10_000.0, 128, 256), 262_144),
            (inv_freqs(1_000_000.0, 64, 512), 262_144),
            (inv_freqs(500_000.0, 32, 64), 131_072),
            (inv_freqs(10_000_000.0, 32, 64), 262_144),
        ] {
            let table = table(&inv, max_pos, DType::BF16).expect("bf16 table");
            let half = inv.len();
            let row_bytes = 2 * half * 2;
            assert_eq!(table.len(), max_pos * row_bytes);
            let edges = (0..2 * ROPE_BLOCK_ROWS).chain(max_pos - ROPE_BLOCK_ROWS..max_pos);
            for pos in edges.chain((0..max_pos).step_by(97)) {
                let row = &table[pos * row_bytes..(pos + 1) * row_bytes];
                for (i, &t) in inv.iter().enumerate() {
                    let angle = pos as f64 * t;
                    assert_eq!(
                        row[2 * i..2 * i + 2],
                        bf16(angle.cos()),
                        "cos pos {pos} freq {i}"
                    );
                    let sin = 2 * (half + i);
                    assert_eq!(
                        row[sin..sin + 2],
                        bf16(angle.sin()),
                        "sin pos {pos} freq {i}"
                    );
                }
            }
        }
    }

    /// `bf16_bits` is `half::bf16::from_f32` over every magnitude a cos or sin takes: a strided
    /// sweep of [-1, 1]'s bit patterns, and every pattern whose low half is a tie, with the kept
    /// bit odd and even.
    #[test]
    fn bf16_bits_rounds_as_half_does() {
        let one = 1.0f32.to_bits();
        let ties = (0..one >> 16).map(|hi| (hi << 16) | 0x8000);
        for bits in (0..=one).step_by(1009).chain(ties) {
            for x in [f32::from_bits(bits), -f32::from_bits(bits)] {
                assert_eq!(bf16_bits(x), half::bf16::from_f32(x).to_bits(), "{x:e}");
            }
        }
    }

    /// Each dtype's bytes are the f32 value's cast: position 0 is cos 1 / sin 0 in every dtype, and
    /// an unsupported dtype is an error, not a table.
    #[test]
    fn write_cos_sin_casts_to_each_dtype() {
        let inv = inv_freqs(10_000.0, 4, 8);
        let f32_table = table(&inv, 3, DType::F32).expect("f32");
        assert_eq!(f32_table.len(), 3 * 8 * 4);
        assert_eq!(f32_table[..4], 1.0f32.to_ne_bytes());
        assert_eq!(f32_table[16..20], 0.0f32.to_ne_bytes());
        let f16_table = table(&inv, 3, DType::F16).expect("f16");
        assert_eq!(f16_table[..2], half::f16::ONE.to_bits().to_ne_bytes());
        let row1 = 8 * 4;
        let cos1 = f32::from_ne_bytes(f32_table[row1..row1 + 4].try_into().expect("4 bytes"));
        assert_eq!(
            f16_table[8 * 2..8 * 2 + 2],
            half::f16::from_f32(cos1).to_bits().to_ne_bytes()
        );
        let mut i32_table = vec![0u8; 3 * 8 * 4];
        assert!(write_cos_sin(&mut i32_table, &inv, 8, DType::I32).is_err());
    }
}
