// SPDX-License-Identifier: Apache-2.0
//! Rotary positional embedding cache — cuda runtime half.
//!
//! The neutral [`RotaryCache`] type, the rope-scaling config structs, and the
//! stream-free `*_from_gpuweights` constructors live in `scratchy-layers`
//! (re-exported below). This module adds the cuda-only stream-based
//! constructors (`*_from_stream`, LongRoPE / YaRN host-math) as a
//! [`CudaRotaryExt`] extension trait so the macro-emitted cuda `load_with`
//! body can keep calling `RotaryCache::new_from_stream(...)` path-syntax.

pub use scratchy_layers::rotary::*;

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::dtype::DType;
#[cfg(feature = "cuda")]
use crate::tensor::GpuTensor;
#[cfg(feature = "cuda")]
use anyhow::Result;

#[cfg(feature = "cuda")]
fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

#[cfg(feature = "cuda")]
fn yarn_find_correction_range(
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

#[cfg(feature = "cuda")]
fn yarn_linear_ramp_mask(low: f64, high: f64, dim: usize) -> Vec<f64> {
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

/// Allocate + upload a `[max_pos, rotary_dim]` f32 cos|sin cache to
/// GPU memory in the target dtype. Extracted so variants (standard,
/// LongRoPE, partial) share one upload path.
///
/// # Safety
/// Requires valid CUDA context and stream.
#[cfg(feature = "cuda")]
unsafe fn upload_combined_cos_sin(
    cache: &[f32],
    max_pos: usize,
    rotary_dim: usize,
    dtype: DType,
    stream: cudarc::driver::sys::CUstream,
) -> Result<GpuTensor> {
    let nbytes = max_pos * rotary_dim * dtype.size_bytes();
    let gpu_ptr = crate::driver::mem_alloc(nbytes)?;
    let host = crate::driver::mem_alloc_host(nbytes)?;
    match dtype {
        DType::F32 => {
            std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
        }
        DType::F16 => {
            let f16_data: Vec<half::f16> = cache.iter().map(|&v| half::f16::from_f32(v)).collect();
            std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
        }
        DType::BF16 => {
            let bf16_data: Vec<half::bf16> =
                cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
            std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
        }
        _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
    }
    crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
    crate::driver::stream_synchronize(stream)?;
    crate::driver::mem_free_host(host)?;
    Ok(GpuTensor::new(gpu_ptr, &[max_pos, rotary_dim], dtype))
}

#[cfg(feature = "cuda")]
unsafe fn build_separate_cos_sin(
    cache: &[f32],
    max_pos: usize,
    rotary_dim: usize,
    dtype: DType,
    stream: cudarc::driver::sys::CUstream,
) -> Result<(GpuTensor, GpuTensor)> {
    let half = rotary_dim / 2;
    let half_nbytes = max_pos * half * dtype.size_bytes();
    let cos_data: Vec<f32> = (0..max_pos)
        .flat_map(|p| (0..half).map(move |i| cache[p * rotary_dim + i]))
        .collect();
    let sin_data: Vec<f32> = (0..max_pos)
        .flat_map(|p| (0..half).map(move |i| cache[p * rotary_dim + half + i]))
        .collect();

    let cos_gpu = crate::driver::mem_alloc(half_nbytes)?;
    let sin_gpu = crate::driver::mem_alloc(half_nbytes)?;
    let host = crate::driver::mem_alloc_host(half_nbytes)?;

    macro_rules! upload {
        ($data:expr, $gpu:expr, $T:ty) => {{
            let converted: Vec<$T> = $data.iter().map(|&v| <$T>::from_f32(v)).collect();
            std::ptr::copy_nonoverlapping(converted.as_ptr() as *const u8, host, half_nbytes);
            crate::driver::memcpy_htod_async($gpu, host, half_nbytes, stream)?;
        }};
    }

    match dtype {
        DType::F16 => {
            upload!(cos_data, cos_gpu, half::f16);
            upload!(sin_data, sin_gpu, half::f16);
        }
        DType::BF16 => {
            upload!(cos_data, cos_gpu, half::bf16);
            upload!(sin_data, sin_gpu, half::bf16);
        }
        DType::F32 => {
            std::ptr::copy_nonoverlapping(cos_data.as_ptr() as *const u8, host, half_nbytes);
            crate::driver::memcpy_htod_async(cos_gpu, host, half_nbytes, stream)?;
            std::ptr::copy_nonoverlapping(sin_data.as_ptr() as *const u8, host, half_nbytes);
            crate::driver::memcpy_htod_async(sin_gpu, host, half_nbytes, stream)?;
        }
        _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
    }
    crate::driver::stream_synchronize(stream)?;
    crate::driver::mem_free_host(host)?;

    Ok((
        GpuTensor::new(cos_gpu, &[max_pos, half], dtype),
        GpuTensor::new(sin_gpu, &[max_pos, half], dtype),
    ))
}

/// Shared backing for `new_longrope_from_stream` (full rotary,
/// `rotary_dim == head_dim`) and `new_partial_longrope_from_stream`
/// (Phi-4-mini, `rotary_dim < head_dim`). Values are computed in
/// `f32` end-to-end to mirror Python vLLM's
/// `Phi3LongRoPEScaledRotaryEmbedding` (`torch.float` = float32):
/// `inv_freq`, `t`, `freqs = t ⊗ inv_freq`, `cos/sin`, and the
/// `* mscale` multiply all happen at f32 precision before the
/// final cast to the target dtype.
///
/// The caller supplies `max_model_len`. The Python side sets
/// `use_long_rope = max_model_len > original_max_position_embeddings`
/// as a global flag at init time and threads it into `forward` by
/// offsetting positions into a concatenated `[short ; long]`
/// cache. We collapse that to one cache of shape
/// `[max_pos, rotary_dim]` whose values at each position match
/// what Python's `index_select(long_short_cache, positions + off)`
/// returns:
/// - `use_long_rope = true`  ⇒ cache[pos] uses `long_factor` + `long_mscale` for pos in 0..max_pos
/// - `use_long_rope = false` ⇒ cache[pos] uses `short_factor` + `short_mscale` for pos in 0..orig_max
///   (positions ≥ orig_max are unreachable when max_model_len ≤ orig_max; filled with the
///   short-factor extrapolation for safety.)
///
/// # Safety
/// Requires valid CUDA context and stream.
#[cfg(feature = "cuda")]
unsafe fn build_longrope(
    head_dim: usize,
    rotary_dim: usize,
    max_pos: usize,
    max_model_len: usize,
    rope_theta: f64,
    longrope: &LongRopeScaling,
    dtype: DType,
    stream: cudarc::driver::sys::CUstream,
) -> Result<RotaryCache> {
    let half = rotary_dim / 2;
    if longrope.short_factor.len() != half || longrope.long_factor.len() != half {
        anyhow::bail!(
            "LongRoPE factor length mismatch: short={}, long={}, expected {} (= rotary_dim/2)",
            longrope.short_factor.len(),
            longrope.long_factor.len(),
            half,
        );
    }
    // Python's `use_long_rope = max_model_len > original_max_position_embeddings`.
    let use_long_rope = max_model_len > longrope.original_max_position_embeddings;
    // Factor + mscale are a PAIR chosen by the flag — Python
    // builds two caches and indexes whichever; we bake the
    // selected one into a single cache.
    let (factor, mscale): (&[f64], f64) = if use_long_rope {
        (&longrope.long_factor, longrope.long_mscale)
    } else {
        (&longrope.short_factor, longrope.short_mscale)
    };

    // Python: `torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim` (float32).
    // Then `base ** exp_arr` (float32). Then `rescale * (...)` (float32).
    // Then `inv_freq = 1.0 / (...)` (float32).
    let base_f32 = rope_theta as f32;
    let rotary_dim_f32 = rotary_dim as f32;
    let inv_freq: Vec<f32> = (0..half)
        .map(|k| {
            let exp = (2 * k) as f32 / rotary_dim_f32;
            let scaled = (factor[k] as f32) * base_f32.powf(exp);
            1.0f32 / scaled
        })
        .collect();

    let mscale_f32 = mscale as f32;

    // Python: `t = torch.arange(max_pos, dtype=torch.float)` (float32).
    // `freqs = t outer inv_freq` (float32).
    // `cos/sin = freqs.cos()/sin() * mscale` (float32).
    // `cache = cat([cos, sin], dim=-1)` — layout matches my
    // `cache[pos, 0..half] = cos`, `cache[pos, half..] = sin`.
    let mut cache = vec![0f32; max_pos * rotary_dim];
    for pos in 0..max_pos {
        let pos_f32 = pos as f32;
        for i in 0..half {
            let angle = pos_f32 * inv_freq[i];
            cache[pos * rotary_dim + i] = angle.cos() * mscale_f32;
            cache[pos * rotary_dim + half + i] = angle.sin() * mscale_f32;
        }
    }

    let cos_sin_cache = upload_combined_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;
    let (cos_cache, sin_cache) =
        build_separate_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;

    Ok(RotaryCache {
        cos_sin_cache,
        cos_cache,
        sin_cache,
        head_dim,
        mrope_section: None,
    })
}

/// Cuda stream-based [`RotaryCache`] constructors. Extension trait over the
/// neutral type so the macro-emitted cuda `load_with` keeps the
/// `RotaryCache::new_from_stream(...)` path-call (bring this trait into scope
/// with `use crate::rotary::CudaRotaryExt as _;`).
#[cfg(feature = "cuda")]
pub trait CudaRotaryExt: Sized {
    /// Build the cos/sin cache on GPU.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    unsafe fn new(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self>;

    /// Build the cos/sin cache on GPU using an explicit CUDA stream.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    unsafe fn new_from_stream(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self>;

    /// Phi-3 / Phi-3.5 LongRoPE (su-scaling) variant of `new_from_stream`.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    unsafe fn new_longrope_from_stream(
        head_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self>;

    /// Phi-4-mini variant: partial rotary (`rotary_dim < head_dim`) + LongRoPE.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    unsafe fn new_partial_longrope_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self>;

    /// Build a YaRN NTK-by-parts RoPE cos/sin cache for DeepSeek-V2 MLA.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    unsafe fn new_yarn_from_stream(
        rope_head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        yarn: &YarnRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self>;

    /// Build cos/sin cache with partial rotary dimension (rotary_dim < head_dim).
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    unsafe fn new_partial(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self>;

    /// Partial-rotary variant of `new_from_stream` (takes an explicit CUDA stream).
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    unsafe fn new_partial_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self>;
}

#[cfg(feature = "cuda")]
impl CudaRotaryExt for RotaryCache {
    unsafe fn new(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        Self::new_from_stream(
            head_dim,
            max_pos,
            rope_theta,
            llama3_scaling,
            dtype,
            device.compute_stream,
        )
    }

    unsafe fn new_from_stream(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let rotary_dim = head_dim; // full rotary for LLaMA
        let half = rotary_dim / 2;

        // Compute inverse frequencies, optionally with llama3 scaling.
        let inv_freqs: Vec<f64> = (0..half)
            .map(|i| {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64);
                if let Some(scaling) = llama3_scaling {
                    let old_context_len = scaling.original_max_position_embeddings as f64;
                    let low_freq_wavelen = old_context_len / scaling.low_freq_factor;
                    let high_freq_wavelen = old_context_len / scaling.high_freq_factor;
                    let wavelen = 2.0 * std::f64::consts::PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq // high frequency: keep as-is
                    } else if wavelen > low_freq_wavelen {
                        freq / scaling.factor // low frequency: scale down
                    } else {
                        // smooth interpolation
                        let smooth = (old_context_len / wavelen - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1.0 - smooth) * freq / scaling.factor + smooth * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        // Build on CPU, then copy to GPU.
        let mut cache = vec![0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rotary_dim + i] = angle.cos() as f32;
                cache[pos * rotary_dim + half + i] = angle.sin() as f32;
            }
        }

        let nbytes = max_pos * rotary_dim * dtype.size_bytes();
        let gpu_ptr = crate::driver::mem_alloc(nbytes)?;

        // Convert to target dtype and upload.
        match dtype {
            DType::F32 => {
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
                crate::driver::stream_synchronize(stream)?;
                crate::driver::mem_free_host(host)?;
            }
            DType::F16 => {
                let f16_data: Vec<half::f16> =
                    cache.iter().map(|&v| half::f16::from_f32(v)).collect();
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
                crate::driver::stream_synchronize(stream)?;
                crate::driver::mem_free_host(host)?;
            }
            DType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, stream)?;
                crate::driver::stream_synchronize(stream)?;
                crate::driver::mem_free_host(host)?;
            }
            _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
        }

        let cos_sin_cache = GpuTensor::new(gpu_ptr, &[max_pos, rotary_dim], dtype);

        // Build separate cos/sin caches for FA2 fused RoPE (needs contiguous buffers).
        let (cos_cache, sin_cache) =
            build_separate_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }

    unsafe fn new_longrope_from_stream(
        head_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        build_longrope(
            head_dim,
            head_dim,
            max_pos,
            max_model_len,
            rope_theta,
            longrope,
            dtype,
            stream,
        )
    }

    unsafe fn new_partial_longrope_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        build_longrope(
            head_dim,
            rotary_dim,
            max_pos,
            max_model_len,
            rope_theta,
            longrope,
            dtype,
            stream,
        )
    }

    unsafe fn new_yarn_from_stream(
        rope_head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        yarn: &YarnRopeScaling,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
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
        // softmax scale). For DeepSeek V2-Lite mscale==mscale_all_dim so this is 1.0.
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
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rope_head_dim + i] = angle.cos() as f32 * mscale;
                cache[pos * rope_head_dim + half + i] = angle.sin() as f32 * mscale;
            }
        }

        let cos_sin_cache = upload_combined_cos_sin(&cache, max_pos, rope_head_dim, dtype, stream)?;
        let (cos_cache, sin_cache) =
            build_separate_cos_sin(&cache, max_pos, rope_head_dim, dtype, stream)?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim: rope_head_dim,
            mrope_section: None,
        })
    }

    unsafe fn new_partial(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        Self::new_partial_from_stream(
            head_dim,
            rotary_dim,
            max_pos,
            rope_theta,
            llama3_scaling,
            dtype,
            device.compute_stream,
        )
    }

    unsafe fn new_partial_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
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

        let mut cache = vec![0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rotary_dim + i] = angle.cos() as f32;
                cache[pos * rotary_dim + half + i] = angle.sin() as f32;
            }
        }

        let cos_sin_cache = upload_combined_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;
        let (cos_cache, sin_cache) =
            build_separate_cos_sin(&cache, max_pos, rotary_dim, dtype, stream)?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
            mrope_section: None,
        })
    }
}
