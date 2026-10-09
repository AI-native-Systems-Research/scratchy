// SPDX-License-Identifier: Apache-2.0
//! Rotary positional embedding cache and rope-scaling config.
//!
//! Backend-neutral half of the RoPE cache: the cos/sin trig math is
//! identical across backends, and the stream-free `*_from_gpuweights`
//! constructors upload through the active [`scratchy_tensors::WeightSource`]
//! (the `GpuWeights` allocator), naming no backend type. The YaRN host
//! math ([`yarn_cos_sin_table`]) is shared: cuda's stream-based
//! `new_yarn_from_stream` and this crate's `new_yarn_from_gpuweights`
//! both build the table through it, so the two backends cannot drift.
//! The cuda-only stream-based upload constructors (`*_from_stream`)
//! live in `scratchy-target-cuda`'s `CudaRotaryExt` extension trait.

use anyhow::Result;
use rayon::prelude::*;
use scratchy_tensors::{DType, GpuTensor, WeightSource};

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
    let bytes: Vec<u8> = match dtype {
        DType::F32 => {
            let mut v = Vec::with_capacity(cache.len() * 4);
            for &f in cache {
                v.extend_from_slice(&f.to_ne_bytes());
            }
            v
        }
        DType::F16 => {
            let mut v = Vec::with_capacity(cache.len() * 2);
            for &f in cache {
                v.extend_from_slice(&half::f16::from_f32(f).to_bits().to_ne_bytes());
            }
            v
        }
        DType::BF16 => {
            let mut v = Vec::with_capacity(cache.len() * 2);
            for &f in cache {
                v.extend_from_slice(&half::bf16::from_f32(f).to_bits().to_ne_bytes());
            }
            v
        }
        _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
    };
    weights.alloc_packed_from_host(&bytes, shape, dtype)
}

// ---------------------------------------------------------------------------
// YaRN host math (shared with the cuda `new_yarn_from_stream`)
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
/// rope — the one shared host-math implementation behind BOTH the cuda
/// `new_yarn_from_stream` and the neutral
/// [`RotaryCache::new_yarn_from_gpuweights`], so the backends cannot
/// drift. cos occupies `[0, half)`, sin `[half, rope_head_dim)` per
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
    /// Backend-neutral, stream-free counterpart to the cuda
    /// `new_from_stream`. Computes the cos/sin tables on CPU (identical
    /// math) and uploads them via the [`WeightSource`] active allocator.
    /// Single `alloc_packed_from_host` call per cache (combined / cos / sin).
    ///
    /// Currently covers the basic case: full rotary
    /// (`rotary_dim == head_dim`), no scaling or Llama3 scaling. The
    /// other rope families have their own constructors:
    /// [`Self::new_partial_from_gpuweights`] (partial rotary),
    /// [`Self::new_longrope_from_gpuweights`], and
    /// [`Self::new_yarn_from_gpuweights`].
    pub fn new_from_gpuweights<W: WeightSource + ?Sized>(
        weights: &mut W,
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
    /// cache, built on CPU and uploaded via the [`WeightSource`] — the
    /// backend-neutral, stream-free counterpart to the cuda
    /// `new_partial_from_stream`. Builds a `[max_pos, rotary_dim]`
    /// cache where `rotary_dim = partial_rotary_factor * head_dim` (e.g.
    /// Qwen3.5: `0.25 * 256 = 64`). The cuda + metal rope kernels read it
    /// with `ROT_DIM = rotary_dim` and pass the trailing
    /// `head_dim - rotary_dim` channels through unrotated. Covers
    /// no-scaling and Llama3 scaling; LongRoPE and YaRN have their own
    /// `*_from_gpuweights` constructors.
    pub fn new_partial_from_gpuweights<W: WeightSource + ?Sized>(
        weights: &mut W,
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

        // Parallel cos/sin fill — `max_pos × rotary_dim` is millions of
        // trig calls for long-context models (Llama-3.2 ships
        // max_position_embeddings = 131072). Rayon par_chunks_mut over
        // rows is safe (each pos writes its own row) and turns ~80ms of
        // single-threaded math into ~10ms on a typical M-series core
        // count.
        let mut cache = vec![0f32; max_pos * rotary_dim];
        cache
            .par_chunks_mut(rotary_dim)
            .enumerate()
            .for_each(|(pos, row)| {
                for i in 0..half {
                    let angle = pos as f64 * inv_freqs[i];
                    row[i] = angle.cos() as f32;
                    row[half + i] = angle.sin() as f32;
                }
            });

        let cos_sin_cache = upload_via_gpuweights(weights, &cache, &[max_pos, rotary_dim], dtype)?;

        // Split `cos_cache` / `sin_cache` are wgpu-only — every cuda
        // and metal kernel reads the unified `cos_sin_cache` above.
        // Building + uploading two ~32 MB caches that nothing reads
        // costs ~30-50ms on Llama-3.2-3B at max_pos = 131072. Skip
        // them under the cuda + metal feature combos.
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
    /// [`WeightSource`] — the backend-neutral counterpart to cuda's
    /// `new_longrope_from_stream`, and arithmetically identical to it
    /// (f32 throughout, matching Python's float32 `inv_freq` path).
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
    /// [`WeightSource`] — the backend-neutral counterpart to cuda's
    /// `new_yarn_from_stream`, and arithmetically identical to it by
    /// construction: both call the shared [`yarn_cos_sin_table`].
    /// NTK-by-parts interpolation between the original frequencies and
    /// the ÷`factor` interpolated ones, ramped over the correction
    /// range; the cos/sin carry the mscale ratio, which is 1.0 for
    /// every config whose `mscale`/`mscale_all_dim` pair is equal —
    /// including all configs that omit the pair (the parse defaults
    /// are HF's 1.0/1.0).
    ///
    /// `rope_head_dim` is the rope-portion head dim: the full
    /// `head_dim` for standard attention, `qk_rope_head_dim` for MLA
    /// models (the codegen arm mirrors cuda's override).
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
    pub fn new_proportional_from_gpuweights<W: WeightSource + ?Sized>(
        weights: &mut W,
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

        let mut cache = vec![0f32; max_pos * rotary_dim];
        cache
            .par_chunks_mut(rotary_dim)
            .enumerate()
            .for_each(|(pos, row)| {
                for i in 0..half {
                    let angle = pos as f64 * inv_freqs[i];
                    row[i] = angle.cos() as f32;
                    row[half + i] = angle.sin() as f32;
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
