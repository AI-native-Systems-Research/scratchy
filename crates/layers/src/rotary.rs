// SPDX-License-Identifier: Apache-2.0
//! Rotary positional embedding cache and rope-scaling config.
//!
//! Backend-neutral half of the RoPE cache: the cos/sin trig math is
//! identical across backends, and the stream-free `*_from_gpuweights`
//! constructors upload through the active [`scratchy_tensors::WeightSource`]
//! (the `GpuWeights` allocator), naming no backend type. The cuda-only
//! stream-based constructors (`*_from_stream`, LongRoPE/YaRN host-math)
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

impl RotaryCache {
    /// Backend-neutral, stream-free counterpart to the cuda
    /// `new_from_stream`. Computes the cos/sin tables on CPU (identical
    /// math) and uploads them via the [`WeightSource`] active allocator.
    /// Single `alloc_packed_from_host` call per cache (combined / cos / sin).
    ///
    /// Currently covers the basic case: full rotary
    /// (`rotary_dim == head_dim`), no scaling or Llama3 scaling. Other
    /// variants (LongRoPE, partial-rotary, YaRN) are still cuda-only —
    /// porting them follows the same pattern but lifts more host-math
    /// helpers; deferred until a metal model needs them.
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
    /// no-scaling and Llama3 scaling; LongRoPE / YaRN stay cuda-only
    /// (their `*_from_stream` host-math helpers aren't lifted yet).
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
