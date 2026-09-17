// SPDX-License-Identifier: Apache-2.0
//! Quantized weight loading — Marlin, BNB, FP8, GGML repacking.
//!
//! Split from weights.rs: core GpuWeights lives in scratchy-target-cuda,
//! quantized loaders stay here (they depend on layers + kernels).

use anyhow::{Result, bail};
use cudarc::driver::sys::CUstream;

use crate::alloc::CachingAllocator;
use crate::driver;
use crate::dtype::DType;
use crate::layers::MarlinLinear;
use crate::quant::{self, QuantConfig};
use crate::tensor::GpuTensor;
#[cfg(feature = "cuda")]
use crate::weights::CudaWeightsExt;
// Quant/dense load extension traits — the `MarlinLinear::load_awq*` /
// `Fp8Linear::load*` / `Fp8BlockLinear::load*` / `Linear::load` calls below
// resolve through these now that the loaders moved off the neutral type defs
// into `*Ops` traits in targets/cuda.
use crate::layers::LinearOps;
use crate::layers_quant::{Fp8BlockLoadOps, Fp8LoadOps, MarlinLoadOps};
use crate::layers_quant::{concat_cpu_dim1, fuse_bias_parts};
use crate::weights::GpuWeights;

// Re-export so callers that reach the Marlin workspace via
// `gpu_weights::alloc_marlin_workspace` keep working. One source of
// truth now lives in scratchy-target-cuda.
pub use crate::layers_quant::alloc_marlin_workspace;

/// Load a single quantized linear layer (AWQ or GPTQ) and repack to Marlin format.
///
/// Loads qweight, scales, qzeros from safetensors, uploads to GPU,
/// runs repack kernels, and applies scale/zero-point permutations.
///
/// # Arguments
/// * `weights` — mmap'd safetensors
/// * `prefix` — weight name prefix (e.g. "model.layers.0.self_attn.q_proj")
/// * `qconfig` — AWQ or GPTQ config
/// * `workspace` — shared Marlin workspace tensor `[num_sms]` i32
/// * `device_id` — CUDA device ordinal
/// * `alloc` — caching allocator for repack output
#[allow(clippy::too_many_arguments)]
pub fn load_marlin_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    qconfig: &QuantConfig,
    workspace: GpuTensor,
    device_id: i32,
    alloc: &mut CachingAllocator,
) -> Result<MarlinLinear> {
    let stream = weights.stream();

    match qconfig {
        QuantConfig::Awq(cfg) => {
            load_awq_marlin_linear(weights, prefix, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::Gptq(cfg) => {
            load_gptq_marlin_linear(weights, prefix, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::None => bail!("load_marlin_linear called with QuantConfig::None"),
        QuantConfig::Bnb4bit(_) => bail!("load_marlin_linear called with Bnb4bit config"),
        QuantConfig::Fp8(_) => bail!("load_marlin_linear called with Fp8 config"),
    }
}

/// Adapter: unwrap the runtime `AwqConfig` into explicit args and call
/// the single-source-of-truth loader in `scratchy-target-cuda`.
#[allow(clippy::too_many_arguments)]
fn load_awq_marlin_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    cfg: &quant::AwqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    _stream: CUstream,
) -> Result<MarlinLinear> {
    MarlinLinear::load_awq(weights, prefix, cfg.group_size, workspace, device_id)
}

/// Load GPTQ quantized linear layer and repack to Marlin format.
#[allow(clippy::too_many_arguments)]
fn load_gptq_marlin_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    cfg: &quant::GptqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Result<MarlinLinear> {
    // GPTQ qweight: [K/8, N] i32 — packed along input dim
    // compressed-tensors: weight_packed [N, K/8] — needs transpose
    let qw_name = format!("{prefix}.qweight");
    let ct_qw_name = format!("{prefix}.weight_packed");
    let is_compressed_tensors = !weights.contains(&qw_name) && weights.contains(&ct_qw_name);

    let (size_k, size_n) = if is_compressed_tensors {
        let (shape, _) = weights
            .tensor_info(&ct_qw_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {ct_qw_name}"))?;
        // compressed-tensors: [N, K/8]
        (shape[1] * 8, shape[0])
    } else {
        let (shape, _) = weights
            .tensor_info(&qw_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {qw_name}"))?;
        // GPTQ: [K/8, N]
        (shape[0] * 8, shape[1])
    };

    let group_size = cfg.group_size;
    let num_groups = size_k.checked_div(group_size).unwrap_or(1);

    // GPTQ uses uint4b8 scalar type which bakes in the zero-point (bias=8).
    // Python vLLM never passes zero-points for GPTQ — just consume and discard.
    let qzeros_name = format!("{prefix}.qzeros");
    if weights.contains(&qzeros_name) {
        let _ = weights.take(&qzeros_name);
    }

    // Handle g_idx for desc_act (activation ordering).
    // Must be done BEFORE repack because repack needs `perm` (sort_indices) on GPU.
    // compressed-tensors doesn't use desc_act.
    let g_idx_name = format!("{prefix}.g_idx");
    let (g_idx_gpu, sort_indices_gpu, has_act_order) =
        if cfg.desc_act && weights.contains(&g_idx_name) {
            // Load g_idx on CPU: Vec<i32> of shape [K]
            let (g_idx_bytes, _g_idx_shape, _g_idx_dtype) = weights.take_cpu(&g_idx_name)?;
            let g_idx_i32: Vec<i32> = g_idx_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Argsort: stable ascending sort by group ID
            let mut sort_indices: Vec<i32> = (0..g_idx_i32.len() as i32).collect();
            sort_indices.sort_by_key(|&i| g_idx_i32[i as usize]);

            // Compute sorted_g_idx
            let sorted_g_idx: Vec<i32> = sort_indices
                .iter()
                .map(|&i| g_idx_i32[i as usize])
                .collect();

            // Upload sorted_g_idx to GPU
            let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let g_idx_nbytes = g_idx_bytes.len();
            let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
            weights.record_alloc(g_idx_ptr, g_idx_nbytes);
            unsafe {
                driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
            }
            let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

            // Upload sort_indices to GPU (used as `perm` for repack and GEMM)
            let si_bytes: Vec<u8> = sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let si_nbytes = si_bytes.len();
            let si_ptr = unsafe { driver::mem_alloc(si_nbytes)? };
            weights.record_alloc(si_ptr, si_nbytes);
            unsafe { driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_nbytes, stream)? };
            let sort_indices_gpu = unsafe { GpuTensor::new(si_ptr, &[size_k], DType::I32) };

            (Some(g_idx_gpu), Some(sort_indices_gpu), true)
        } else {
            // Consume g_idx if present (not needed without desc_act)
            if weights.contains(&g_idx_name) {
                let _ = weights.take(&g_idx_name);
            }
            (None, None, false)
        };

    // Upload qweight to GPU (raw alloc — will be freed after repack).
    // For compressed-tensors, transpose from [N, K/8] to [K/8, N] on CPU first.
    let qweight_gpu = if is_compressed_tensors {
        let (qw_bytes, qw_shape, qw_dtype) = weights.take_cpu(&ct_qw_name)?;
        let n = qw_shape[0];
        let k_packed = qw_shape[1];
        let transposed = transpose_2d_cpu(&qw_bytes, n, k_packed, qw_dtype.size_bytes());
        // Consume weight_shape if present
        let shape_name = format!("{prefix}.weight_shape");
        if weights.contains(&shape_name) {
            let _ = weights.take_cpu(&shape_name);
        }
        let nbytes = transposed.len();
        let ptr = unsafe { driver::mem_alloc(nbytes)? };
        unsafe { driver::memcpy_htod_async(ptr, transposed.as_ptr(), nbytes, stream)? };
        unsafe { GpuTensor::new(ptr, &[k_packed, n], qw_dtype) }
    } else {
        weights.take(&qw_name)?
    };

    // Repack GPTQ → Marlin tiled layout on GPU.
    // When has_act_order, pass sort_indices as perm so the repack kernel
    // physically reorders weights so same-group channels are contiguous.
    // Use driver::mem_alloc for the output (NOT caching allocator) because model
    // weights must survive free_leaked_blocks() during profiling.
    let num_u32 = size_k * size_n / 8;
    let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
    let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
    weights.record_alloc(repack_ptr, repack_nbytes);
    unsafe {
        crate::kernels::gptq_repack_into(
            qweight_gpu,
            sort_indices_gpu,
            repack_ptr,
            size_k,
            size_n,
            device_id,
            stream,
        );
        // Sync so the repack kernel finishes before we free the source qweight
        driver::stream_synchronize(stream)?;
        // Free original qweight (it was raw-allocated by weights.take())
        weights.unrecord_alloc(qweight_gpu.raw_ptr());
        driver::mem_free(qweight_gpu.raw_ptr())?;
    }
    let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

    // Load and permute scales (CPU).
    // For compressed-tensors, transpose from [N, num_groups] to [num_groups, N] first.
    let scales_bytes = if is_compressed_tensors {
        let ct_scales_name = format!("{prefix}.weight_scale");
        let (sc_bytes, sc_shape, sc_dtype) = weights.take_cpu(&ct_scales_name)?;
        let sc_n = sc_shape[0];
        let sc_groups = sc_shape[1];
        let transposed = transpose_2d_cpu(&sc_bytes, sc_n, sc_groups, sc_dtype.size_bytes());
        (transposed, vec![sc_groups, sc_n], sc_dtype)
    } else {
        let scales_name = format!("{prefix}.scales");
        weights.take_cpu(&scales_name)?
    };
    let scales_dtype = scales_bytes.2;
    let mut scales_u16: Vec<u16> = scales_bytes
        .0
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    quant::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

    let scales_bytes_permuted: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let scales_nbytes = scales_bytes_permuted.len();
    let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
    weights.record_alloc(scales_ptr, scales_nbytes);
    unsafe {
        driver::memcpy_htod_async(
            scales_ptr,
            scales_bytes_permuted.as_ptr(),
            scales_nbytes,
            stream,
        )?;
    }
    let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

    // Load bias if present
    let bias_name = format!("{prefix}.bias");
    let bias_gpu = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };

    unsafe { driver::stream_synchronize(stream)? };

    Ok(MarlinLinear {
        qweight: qweight_marlin,
        scales: scales_gpu,
        zeros: None,
        g_idx: g_idx_gpu,
        g_idx_sort_indices: sort_indices_gpu,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: false,
        has_act_order,
        b_type_id: 0, // GPTQ = uint4b8
        device_id,
        bias: bias_gpu,
    })
}

// `concat_cpu_dim1` + `fuse_bias_parts` now live in
// `crate::layers_quant` so scratchy-forward-compiler-emitted loader
// code can reach them; this module's GPTQ path still calls them via
// the `use` at the top.

/// Load multiple quantized linear layers and fuse into a single Marlin GEMM.
///
/// This is the key optimization: instead of 3 separate Marlin GEMMs for QKV
/// (or 2 for gate_up), we concatenate the raw qweights/scales/qzeros along the
/// N dimension on CPU, repack once, and get a single wider Marlin GEMM.
/// This matches Python vLLM's `MergedColumnParallelLinear`.
///
/// 5→2 GEMMs per layer (QKV fused, gate_up fused).
#[allow(clippy::too_many_arguments)]
pub fn load_fused_marlin_linear(
    weights: &mut GpuWeights,
    prefixes: &[String],
    qconfig: &QuantConfig,
    workspace: GpuTensor,
    device_id: i32,
    alloc: &mut CachingAllocator,
) -> Result<MarlinLinear> {
    let stream = weights.stream();
    match qconfig {
        QuantConfig::Awq(cfg) => {
            load_fused_awq_marlin(weights, prefixes, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::Gptq(cfg) => {
            load_fused_gptq_marlin(weights, prefixes, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::None => bail!("load_fused_marlin_linear called with QuantConfig::None"),
        QuantConfig::Bnb4bit(_) => bail!("load_fused_marlin_linear called with Bnb4bit config"),
        QuantConfig::Fp8(_) => bail!("load_fused_marlin_linear called with Fp8 config"),
    }
}

/// Load and fuse multiple GPTQ layers into a single Marlin layer.
///
/// GPTQ qweight: `[K/8, N]` i32 — concat along dim1 → `[K/8, N_total]`.
#[allow(clippy::too_many_arguments)]
fn load_fused_gptq_marlin(
    weights: &mut GpuWeights,
    prefixes: &[String],
    cfg: &quant::GptqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Result<MarlinLinear> {
    // Gather raw CPU tensors for concat.
    let mut qw_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut sc_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut g_idx_i32: Option<Vec<i32>> = None;
    let mut bias_parts: Vec<Vec<u8>> = Vec::new();
    let mut bias_dtype: Option<DType> = None;

    for (i, prefix) in prefixes.iter().enumerate() {
        let qw_name = format!("{prefix}.qweight");
        let ct_qw_name = format!("{prefix}.weight_packed");
        let is_ct = !weights.contains(&qw_name) && weights.contains(&ct_qw_name);

        if is_ct {
            // compressed-tensors: weight_packed [N, K/8] → transpose to [K/8, N]
            let (qw_bytes, qw_shape, qw_dtype) = weights.take_cpu(&ct_qw_name)?;
            let n = qw_shape[0];
            let k_packed = qw_shape[1];
            let transposed = transpose_2d_cpu(&qw_bytes, n, k_packed, qw_dtype.size_bytes());
            qw_parts.push((transposed, vec![k_packed, n], qw_dtype));

            // weight_scale [N, num_groups] → transpose to [num_groups, N]
            let ct_sc_name = format!("{prefix}.weight_scale");
            let (sc_bytes, sc_shape, sc_dtype) = weights.take_cpu(&ct_sc_name)?;
            let sc_n = sc_shape[0];
            let sc_groups = sc_shape[1];
            let sc_transposed = transpose_2d_cpu(&sc_bytes, sc_n, sc_groups, sc_dtype.size_bytes());
            sc_parts.push((sc_transposed, vec![sc_groups, sc_n], sc_dtype));

            // Consume weight_shape if present
            let shape_name = format!("{prefix}.weight_shape");
            if weights.contains(&shape_name) {
                let _ = weights.take_cpu(&shape_name);
            }
        } else {
            let scales_name = format!("{prefix}.scales");
            let qzeros_name = format!("{prefix}.qzeros");

            qw_parts.push(weights.take_cpu(&qw_name)?);
            sc_parts.push(weights.take_cpu(&scales_name)?);

            // GPTQ uses uint4b8 scalar type — never pass zero-points. Consume and discard.
            if weights.contains(&qzeros_name) {
                let _ = weights.take_cpu(&qzeros_name);
            }
        }

        // For desc_act: all sub-layers share the same K dimension → same g_idx.
        // Take from the first prefix, consume and discard from the rest.
        let g_idx_name = format!("{prefix}.g_idx");
        if weights.contains(&g_idx_name) {
            if i == 0 && cfg.desc_act {
                let (g_bytes, _g_shape, _g_dtype) = weights.take_cpu(&g_idx_name)?;
                g_idx_i32 = Some(
                    g_bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            } else {
                let _ = weights.take_cpu(&g_idx_name);
            }
        }

        // Collect bias if present (e.g. Qwen3 MoE attention has QKV bias).
        let bias_name = format!("{prefix}.bias");
        if weights.contains(&bias_name) {
            let (b_data, _b_shape, b_dtype) = weights.take_cpu(&bias_name)?;
            bias_parts.push(b_data);
            bias_dtype = Some(b_dtype);
        }
    }

    // Concat qweights along dim1: [K/8, N1] + [K/8, N2] + ... → [K/8, N_total]
    let qw_refs: Vec<_> = qw_parts
        .iter()
        .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
        .collect();
    let (qw_fused, qw_shape, _) = concat_cpu_dim1(&qw_refs);
    let size_k = qw_shape[0] * 8;
    let size_n = qw_shape[1];
    let group_size = cfg.group_size;
    let num_groups = size_k.checked_div(group_size).unwrap_or(1);

    // Handle g_idx for desc_act — must be done BEFORE repack.
    let (g_idx_gpu, sort_indices_gpu, has_act_order) = if let Some(g_idx) = g_idx_i32 {
        // Argsort: stable ascending sort by group ID
        let mut sort_indices: Vec<i32> = (0..g_idx.len() as i32).collect();
        sort_indices.sort_by_key(|&i| g_idx[i as usize]);

        // Compute sorted_g_idx
        let sorted_g_idx: Vec<i32> = sort_indices.iter().map(|&i| g_idx[i as usize]).collect();

        // Upload sorted_g_idx to GPU
        let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let g_idx_nbytes = g_idx_bytes.len();
        let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
        weights.record_alloc(g_idx_ptr, g_idx_nbytes);
        unsafe {
            driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
        }
        let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

        // Upload sort_indices to GPU (used as `perm` for repack and GEMM)
        let si_bytes: Vec<u8> = sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let si_nbytes = si_bytes.len();
        let si_ptr = unsafe { driver::mem_alloc(si_nbytes)? };
        weights.record_alloc(si_ptr, si_nbytes);
        unsafe { driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_nbytes, stream)? };
        let sort_indices_gpu = unsafe { GpuTensor::new(si_ptr, &[size_k], DType::I32) };

        (Some(g_idx_gpu), Some(sort_indices_gpu), true)
    } else {
        (None, None, false)
    };

    // Upload fused qweight to GPU and repack.
    let qw_nbytes = qw_fused.len();
    let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
    unsafe { driver::memcpy_htod_async(qw_gpu_ptr, qw_fused.as_ptr(), qw_nbytes, stream)? };
    let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &qw_shape, DType::I32) };

    let num_u32 = size_k * size_n / 8;
    let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
    let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
    weights.record_alloc(repack_ptr, repack_nbytes);
    unsafe {
        crate::kernels::gptq_repack_into(
            qw_gpu,
            sort_indices_gpu,
            repack_ptr,
            size_k,
            size_n,
            device_id,
            stream,
        );
        driver::stream_synchronize(stream)?;
        driver::mem_free(qw_gpu_ptr)?;
    }
    let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

    // Concat and permute scales: [num_groups, N1] + ... → [num_groups, N_total]
    let sc_refs: Vec<_> = sc_parts
        .iter()
        .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
        .collect();
    let (sc_fused, _sc_shape, scales_dtype) = concat_cpu_dim1(&sc_refs);
    let mut scales_u16: Vec<u16> = sc_fused
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    quant::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

    let scales_bytes_permuted: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let scales_nbytes = scales_bytes_permuted.len();
    let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
    weights.record_alloc(scales_ptr, scales_nbytes);
    unsafe {
        driver::memcpy_htod_async(
            scales_ptr,
            scales_bytes_permuted.as_ptr(),
            scales_nbytes,
            stream,
        )?;
    }
    let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

    unsafe { driver::stream_synchronize(stream)? };

    Ok(MarlinLinear {
        qweight: qweight_marlin,
        scales: scales_gpu,
        zeros: None,
        g_idx: g_idx_gpu,
        g_idx_sort_indices: sort_indices_gpu,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: false,
        has_act_order,
        b_type_id: 0, // GPTQ = uint4b8
        device_id,
        bias: fuse_bias_parts(&bias_parts, bias_dtype, weights, stream)?,
    })
}

/// Adapter: unwrap the runtime `AwqConfig` into explicit args and call
/// the single-source-of-truth fused loader in `scratchy-target-cuda`.
/// AWQ qweight is `[K, N/8]` i32 — scratchy-target-cuda concatenates along
/// dim 1 and does a single repack.
#[allow(clippy::too_many_arguments)]
fn load_fused_awq_marlin(
    weights: &mut GpuWeights,
    prefixes: &[String],
    cfg: &quant::AwqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    _stream: CUstream,
) -> Result<MarlinLinear> {
    let prefix_refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
    MarlinLinear::load_awq_concat(weights, &prefix_refs, cfg.group_size, workspace, device_id)
}

// `alloc_marlin_workspace` and `fuse_bias_parts` live in
// `crate::layers_quant`; see the `pub use` / `use` at the
// top of this module.

// ---------------------------------------------------------------------------
// BitsAndBytes 4-bit weight loading
// ---------------------------------------------------------------------------

use crate::layers::Bnb4bitLinear;
use crate::quant::Bnb4bitConfig;

/// Parse the BNB quant_state JSON blob.
///
/// Returns `(nested_offset, blocksize, nested_blocksize)`.
fn parse_bnb_quant_state_json(data: &[u8]) -> Result<(f32, usize, usize)> {
    // Quant state blobs may have trailing null bytes after the JSON.
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    let s = std::str::from_utf8(&data[..end])?;
    let v: serde_json::Value = serde_json::from_str(s)?;
    let nested_offset = v
        .get("nested_offset")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let blocksize = v.get("blocksize").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let nested_blocksize = v
        .get("nested_blocksize")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    Ok((nested_offset, blocksize, nested_blocksize))
}

/// Upload the NF4/FP4 lookup table to GPU (shared across all layers).
pub fn upload_bnb_code(code: &[f32; 16], stream: CUstream) -> Result<GpuTensor> {
    let nbytes = 16 * std::mem::size_of::<f32>();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    unsafe {
        driver::memcpy_htod_async(ptr, code.as_ptr() as *const u8, nbytes, stream)?;
    }
    Ok(unsafe { GpuTensor::new(ptr, &[16], DType::F32) })
}

/// Allocate a shared dequantization scratch buffer for BNB 4-bit.
///
/// Size: max(out_features * in_features) across all linear layers × elem_size.
/// The caller should track the max dimensions during model loading.
pub fn alloc_bnb_dequant_scratch(
    max_elements: usize,
    dtype: DType,
    stream: CUstream,
) -> Result<GpuTensor> {
    let nbytes = max_elements * dtype.size_bytes();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    // Zero-initialize (not strictly needed, but helps debugging).
    unsafe { driver::memset_d8(ptr, 0, nbytes, stream)? };
    Ok(unsafe { GpuTensor::new(ptr, &[max_elements], dtype) })
}

/// Dequantize double-quantized absmax on CPU.
///
/// BNB pre-quantized models store absmax as U8 (double-quantized).
/// This matches Python `_dequantize_dq`:
/// ```text
/// absmax_f32[i] = nested_quant_map[absmax_u8[i]] * nested_absmax[i / 256]
/// ```
pub fn dequantize_double_quant_absmax(
    absmax_u8: &[u8],
    nested_quant_map: &[f32], // [256]
    nested_absmax: &[f32],    // [num_blocks / nested_blocksize]
    nested_blocksize: usize,  // typically 256
    nested_offset: f32,       // from quant_state JSON
) -> Vec<f32> {
    absmax_u8
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let scale = nested_absmax[i / nested_blocksize];
            nested_quant_map[val as usize] * scale + nested_offset
        })
        .collect()
}

/// Load a single BNB 4-bit linear layer from safetensors.
///
/// Expects:
/// - `{prefix}.weight` — U8 packed nibbles
/// - `{prefix}.weight.absmax` — U8 double-quantized absmax
/// - `{prefix}.weight.nested_absmax` — F32 absmax of absmax
/// - `{prefix}.weight.nested_quant_map` — F32 [256] nested dequant table
/// - `{prefix}.weight.quant_map` — F32 [16] NF4/FP4 code (used to detect quant type)
/// - `{prefix}.weight.quant_state.bitsandbytes__nf4` — U8 metadata blob
///
/// `code_gpu` is the shared NF4/FP4 lookup table already on GPU.
/// `dequant_scratch` is the shared dequant scratch buffer.
/// `out_features`/`in_features` are the original weight dimensions.
#[allow(clippy::too_many_arguments)]
pub fn load_bnb4bit_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    _qconfig: &Bnb4bitConfig,
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features: usize,
    in_features: usize,
    blocksize: usize,
    stream: CUstream,
) -> Result<Bnb4bitLinear> {
    let weight_name = format!("{prefix}.weight");
    let absmax_name = format!("{prefix}.weight.absmax");
    let nested_absmax_name = format!("{prefix}.weight.nested_absmax");
    let nested_quant_map_name = format!("{prefix}.weight.nested_quant_map");

    // Parse quant_state JSON to get nested_offset and actual blocksize.
    let quant_state_name = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");
    let (nested_offset, actual_blocksize, nested_blocksize) = if weights.contains(&quant_state_name)
    {
        let (qs_bytes, _, _) = weights.take_cpu(&quant_state_name)?;
        parse_bnb_quant_state_json(&qs_bytes)?
    } else {
        (0.0, blocksize, 256)
    };
    let blocksize = actual_blocksize;

    // Load packed weight to GPU (U8).
    let packed_weight = weights.take(&weight_name)?;

    // Load and dequantize double-quantized absmax on CPU.
    let (absmax_bytes, _absmax_shape, absmax_dtype) = weights.take_cpu(&absmax_name)?;

    let absmax_f32 = if absmax_dtype == DType::U8 {
        // Double quantized — need nested_absmax and nested_quant_map.
        let (nqm_bytes, _nqm_shape, _) = weights.take_cpu(&nested_quant_map_name)?;
        let nested_quant_map: Vec<f32> = nqm_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let (na_bytes, _na_shape, _) = weights.take_cpu(&nested_absmax_name)?;
        let nested_absmax: Vec<f32> = na_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        dequantize_double_quant_absmax(
            &absmax_bytes,
            &nested_quant_map,
            &nested_absmax,
            nested_blocksize,
            nested_offset,
        )
    } else {
        // Already F32 (non-double-quantized).
        absmax_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // Upload absmax F32 to GPU.
    let absmax_nbytes = absmax_f32.len() * 4;
    let absmax_ptr = unsafe { driver::mem_alloc(absmax_nbytes)? };
    weights.record_alloc(absmax_ptr, absmax_nbytes);
    unsafe {
        driver::memcpy_htod_async(
            absmax_ptr,
            absmax_f32.as_ptr() as *const u8,
            absmax_nbytes,
            stream,
        )?;
    }
    let absmax_gpu = unsafe { GpuTensor::new(absmax_ptr, &[absmax_f32.len()], DType::F32) };

    // Load bias if present.
    let bias_name = format!("{prefix}.bias");
    let bias = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };

    // Consume remaining BNB metadata tensors so they don't cause "unused weight" warnings.
    let quant_map_name = format!("{prefix}.weight.quant_map");
    for name in &[
        &quant_map_name,
        &quant_state_name,
        &nested_absmax_name,
        &nested_quant_map_name,
    ] {
        if weights.contains(name) {
            let _ = weights.take_cpu(name);
        }
    }

    Ok(Bnb4bitLinear {
        packed_weight,
        absmax: absmax_gpu,
        code: code_gpu,
        dequant_scratch,
        out_features,
        in_features,
        blocksize,
        bias,
    })
}

/// Load fused BNB 4-bit linear (e.g., QKV or gate_up) from multiple prefixes.
///
/// Concatenates packed bytes + absmax from multiple shards on CPU, then uploads once.
/// Absmax blocks are independent per shard, so concat is straightforward.
#[allow(clippy::too_many_arguments)]
pub fn load_fused_bnb4bit_linear(
    weights: &mut GpuWeights,
    prefixes: &[String],
    _qconfig: &Bnb4bitConfig,
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features_per_shard: &[usize],
    in_features: usize,
    blocksize: usize,
    stream: CUstream,
) -> Result<Bnb4bitLinear> {
    let total_out_features: usize = out_features_per_shard.iter().sum();

    // Concatenate packed bytes from all shards on CPU.
    let mut all_packed: Vec<u8> = Vec::new();
    let mut all_absmax_f32: Vec<f32> = Vec::new();

    for (prefix, &out_feat) in prefixes.iter().zip(out_features_per_shard.iter()) {
        let weight_name = format!("{prefix}.weight");
        let absmax_name = format!("{prefix}.weight.absmax");
        let nested_absmax_name = format!("{prefix}.weight.nested_absmax");
        let nested_quant_map_name = format!("{prefix}.weight.nested_quant_map");
        let quant_state_name = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");

        // Parse quant_state JSON for nested_offset.
        let (nested_offset, _actual_blocksize, nested_blocksize) =
            if weights.contains(&quant_state_name) {
                let (qs_bytes, _, _) = weights.take_cpu(&quant_state_name)?;
                parse_bnb_quant_state_json(&qs_bytes)?
            } else {
                (0.0, blocksize, 256)
            };

        // Load packed weight to CPU.
        let (packed_bytes, _packed_shape, _) = weights.take_cpu(&weight_name)?;
        all_packed.extend_from_slice(&packed_bytes);

        // Load and dequantize absmax.
        let (absmax_bytes, _absmax_shape, absmax_dtype) = weights.take_cpu(&absmax_name)?;
        let shard_absmax = if absmax_dtype == DType::U8 {
            let (nqm_bytes, _, _) = weights.take_cpu(&nested_quant_map_name)?;
            let nested_quant_map: Vec<f32> = nqm_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let (na_bytes, _, _) = weights.take_cpu(&nested_absmax_name)?;
            let nested_absmax: Vec<f32> = na_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            dequantize_double_quant_absmax(
                &absmax_bytes,
                &nested_quant_map,
                &nested_absmax,
                nested_blocksize,
                nested_offset,
            )
        } else {
            absmax_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        all_absmax_f32.extend_from_slice(&shard_absmax);

        // Consume remaining BNB metadata.
        let quant_map_name = format!("{prefix}.weight.quant_map");
        for name in &[
            &quant_map_name,
            &quant_state_name,
            &nested_absmax_name,
            &nested_quant_map_name,
        ] {
            if weights.contains(name) {
                let _ = weights.take_cpu(name);
            }
        }

        let _ = out_feat; // used for shape validation in debug
    }

    // Upload packed weight to GPU.
    let packed_nbytes = all_packed.len();
    let packed_ptr = unsafe { driver::mem_alloc(packed_nbytes)? };
    weights.record_alloc(packed_ptr, packed_nbytes);
    unsafe {
        driver::memcpy_htod_async(packed_ptr, all_packed.as_ptr(), packed_nbytes, stream)?;
    }
    let packed_gpu = unsafe { GpuTensor::new(packed_ptr, &[packed_nbytes], DType::U8) };

    // Upload absmax F32 to GPU.
    let absmax_nbytes = all_absmax_f32.len() * 4;
    let absmax_ptr = unsafe { driver::mem_alloc(absmax_nbytes)? };
    weights.record_alloc(absmax_ptr, absmax_nbytes);
    unsafe {
        driver::memcpy_htod_async(
            absmax_ptr,
            all_absmax_f32.as_ptr() as *const u8,
            absmax_nbytes,
            stream,
        )?;
    }
    let absmax_gpu = unsafe { GpuTensor::new(absmax_ptr, &[all_absmax_f32.len()], DType::F32) };

    Ok(Bnb4bitLinear {
        packed_weight: packed_gpu,
        absmax: absmax_gpu,
        code: code_gpu,
        dequant_scratch,
        out_features: total_out_features,
        in_features,
        blocksize,
        bias: None,
    })
}

// ---------------------------------------------------------------------------
// FP8 Weight Loading
// ---------------------------------------------------------------------------

// `ensure_f32_scale` and `block_scale_name` live in
// `crate::layers_quant`; see the `use` at the top of this
// module. `_tp` variants below reach them via the aliased names.
use crate::layers_quant::{block_scale_name, ensure_f32_scale};

/// Adapter: delegate to the single-source-of-truth loader in `scratchy-target-cuda`.
pub fn load_fp8_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    output_dtype: DType,
) -> Result<crate::layers::Fp8Linear> {
    crate::layers::Fp8Linear::load(weights, prefix, output_dtype)
}

/// Load a fused FP8 linear layer by concatenating multiple FP8 projections.
///
/// For fused QKV (3 projections) or gate_up (2 projections), concatenates
/// FP8 weights along dim=0 and merges per-shard weight scales by taking
/// the max, then re-quantizes shards with smaller scales to use the unified
/// max scale (matching Python's `requantize_with_max_scale`).
///
/// Also supports online quantization: if weights are BF16, quantizes each
/// shard to FP8 on the fly (matching Python's `Fp8OnlineLinearMethod`).
/// Adapter: delegate to the single-source-of-truth fused loader in `scratchy-target-cuda`.
/// `stream` arg retained for back-compat; `Fp8Linear::load_concat` reads it from
/// `weights` internally.
pub fn load_fused_fp8_linear(
    weights: &mut GpuWeights,
    prefixes: &[String],
    output_dtype: DType,
    _stream: CUstream,
) -> Result<crate::layers::Fp8Linear> {
    let prefix_refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
    crate::layers::Fp8Linear::load_concat(weights, &prefix_refs, output_dtype)
}

// ---------------------------------------------------------------------------
// FP8 Block-quantized Weight Loading
// ---------------------------------------------------------------------------

/// Adapter: delegate to the single-source-of-truth block loader in `scratchy-target-cuda`.
pub fn load_fp8_block_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    output_dtype: DType,
) -> Result<crate::layers::Fp8BlockLinear> {
    crate::layers::Fp8BlockLinear::load(weights, prefix, output_dtype)
}

/// Adapter: delegate to the single-source-of-truth fused block loader in `scratchy-target-cuda`.
/// `stream` arg retained for back-compat; `Fp8BlockLinear::load_concat` reads it from
/// `weights` internally.
pub fn load_fused_fp8_block_linear(
    weights: &mut GpuWeights,
    prefixes: &[String],
    output_dtype: DType,
    _stream: CUstream,
) -> Result<crate::layers::Fp8BlockLinear> {
    let prefix_refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
    crate::layers::Fp8BlockLinear::load_concat(weights, &prefix_refs, output_dtype)
}

/// TP variant: load fused FP8 block-quantized linear with dim=0 sharding.
/// Each prefix's weight and scale are sharded along dim=0 (output dimension)
/// before fusing.
pub fn load_fused_fp8_block_linear_tp(
    weights: &mut GpuWeights,
    prefixes: &[String],
    output_dtype: DType,
    rank: usize,
    world_size: usize,
    stream: CUstream,
) -> Result<crate::layers::Fp8BlockLinear> {
    anyhow::ensure!(
        !prefixes.is_empty(),
        "load_fused_fp8_block_linear_tp: no prefixes"
    );

    // Get shapes and derive block size from first prefix.
    let first_weight_name = format!("{}.weight", prefixes[0]);
    let (first_shape, first_dtype) = weights
        .tensor_info(&first_weight_name)
        .ok_or_else(|| anyhow::anyhow!("FP8 block: weight not found: {first_weight_name}"))?;
    anyhow::ensure!(first_dtype == DType::Fp8E4m3);
    anyhow::ensure!(first_shape.len() == 2);
    let in_features = first_shape[1];

    let first_scale_name = block_scale_name(weights, &prefixes[0]);
    let (first_scale_shape, _) = weights
        .tensor_info(&first_scale_name)
        .ok_or_else(|| anyhow::anyhow!("FP8 block: scale not found: {first_scale_name}"))?;
    let block_n = first_shape[0] / first_scale_shape[0];
    let block_k = first_shape[1] / first_scale_shape[1];
    let scale_cols = first_scale_shape[1];

    // Compute sharded output dimensions.
    let mut total_out = 0usize;
    let mut total_scale_rows = 0usize;
    let mut shard_sizes = Vec::with_capacity(prefixes.len());
    let mut shard_scale_rows = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let wname = format!("{prefix}.weight");
        let (shape, _) = weights
            .tensor_info(&wname)
            .ok_or_else(|| anyhow::anyhow!("FP8 block: weight not found: {wname}"))?;
        let sharded_out = shape[0] / world_size;
        shard_sizes.push(sharded_out);
        total_out += sharded_out;

        let sname = block_scale_name(weights, prefix);
        let (sshape, _) = weights
            .tensor_info(&sname)
            .ok_or_else(|| anyhow::anyhow!("FP8 block: scale not found: {sname}"))?;
        let sharded_sr = sshape[0] / world_size;
        shard_scale_rows.push(sharded_sr);
        total_scale_rows += sharded_sr;
    }

    // Allocate fused FP8 weight buffer (sharded).
    let elem_size = DType::Fp8E4m3.size_bytes();
    let total_bytes = total_out * in_features * elem_size;
    let fused_ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };

    let mut offset = 0usize;
    for (i, prefix) in prefixes.iter().enumerate() {
        let wname = format!("{prefix}.weight");
        let shard_bytes = shard_sizes[i] * in_features * elem_size;
        unsafe {
            weights.take_shard_into(&wname, 0, rank, world_size, fused_ptr.add(offset), stream)?;
        }
        offset += shard_bytes;
    }

    let fused_weight =
        unsafe { GpuTensor::new(fused_ptr, &[total_out, in_features], DType::Fp8E4m3) };

    // Allocate fused scale buffer (sharded along dim=0).
    let total_scale_bytes = total_scale_rows * scale_cols * 4;
    let scale_ptr = unsafe { crate::driver::mem_alloc(total_scale_bytes)? };
    let mut scale_offset = 0usize;
    for (i, prefix) in prefixes.iter().enumerate() {
        let sname = block_scale_name(weights, prefix);
        let raw_scale = weights.take_shard(&sname, 0, rank, world_size)?;
        let shard_scale = ensure_f32_scale(raw_scale, stream)?;
        let shard_bytes = shard_scale_rows[i] * scale_cols * 4;
        unsafe {
            crate::driver::memcpy_dtod_async(
                scale_ptr.add(scale_offset),
                shard_scale.raw_ptr(),
                shard_bytes,
                stream,
            )?;
        }
        scale_offset += shard_bytes;
    }
    let fused_scale =
        unsafe { GpuTensor::new(scale_ptr, &[total_scale_rows, scale_cols], DType::F32) };

    // Consume input_scale if present.
    for prefix in prefixes {
        let input_scale_name = format!("{prefix}.input_scale");
        if weights.contains(&input_scale_name) {
            let _ = weights.take(&input_scale_name);
        }
    }

    Ok(crate::layers::Fp8BlockLinear {
        weight: fused_weight,
        weight_scale_inv: fused_scale,
        block_size: [block_n, block_k],
        bias: None,
        output_dtype,
    })
}

/// TP variant: load single FP8 block-quantized linear with dim=1 sharding
/// (row parallel, e.g. o_proj, down_proj).
pub fn load_fp8_block_linear_tp(
    weights: &mut GpuWeights,
    prefix: &str,
    output_dtype: DType,
    rank: usize,
    world_size: usize,
) -> Result<crate::layers::Fp8BlockLinear> {
    let weight_name = format!("{prefix}.weight");
    let scale_name = block_scale_name(weights, prefix);

    let weight = weights.take_shard(&weight_name, 1, rank, world_size)?;
    anyhow::ensure!(weight.ndim() == 2);
    let n = weight.dim(0);
    let k = weight.dim(1);

    let stream = weights.stream();
    let raw_scale = weights.take_shard(&scale_name, 1, rank, world_size)?;
    let scale = ensure_f32_scale(raw_scale, stream)?;
    let scale_rows = scale.dim(0);
    let scale_cols = scale.dim(1);
    let block_n = n / scale_rows;
    let block_k = k / scale_cols;

    // Consume input_scale if present.
    let input_scale_name = format!("{prefix}.input_scale");
    if weights.contains(&input_scale_name) {
        let _ = weights.take(&input_scale_name);
    }

    Ok(crate::layers::Fp8BlockLinear {
        weight,
        weight_scale_inv: scale,
        block_size: [block_n, block_k],
        bias: None,
        output_dtype,
    })
}

// ---------------------------------------------------------------------------
// Per-tensor FP8 + Tensor Parallel
// ---------------------------------------------------------------------------

/// Load a **fused** per-tensor FP8 linear with column-parallel TP (QKV, gate+up).
///
/// Each prefix's weight is sharded along dim=0 (output dimension). Per-tensor
/// scales are merged by taking the max and re-quantizing shards with smaller
/// scales, matching Python's `requantize_with_max_scale()`.
pub fn load_fused_fp8_linear_tp(
    weights: &mut GpuWeights,
    prefixes: &[String],
    output_dtype: DType,
    rank: usize,
    world_size: usize,
    stream: CUstream,
) -> Result<crate::layers::Fp8Linear> {
    anyhow::ensure!(
        !prefixes.is_empty(),
        "load_fused_fp8_linear_tp: no prefixes"
    );

    // Get shapes from first prefix.
    let first_weight_name = format!("{}.weight", prefixes[0]);
    let (first_shape, first_dtype) = weights
        .tensor_info(&first_weight_name)
        .ok_or_else(|| anyhow::anyhow!("FP8 TP: weight not found: {first_weight_name}"))?;
    let is_online_quant = first_dtype == DType::BF16 || first_dtype == DType::F16;
    anyhow::ensure!(
        first_dtype == DType::Fp8E4m3 || is_online_quant,
        "FP8 TP fused weight expected Fp8E4m3 or BF16, got {first_dtype}"
    );
    anyhow::ensure!(first_shape.len() == 2, "FP8 TP fused weight must be 2D");
    let in_features = first_shape[1];

    // Compute sharded output dimensions (each prefix sharded along dim=0).
    let mut total_out = 0usize;
    let mut shard_sizes = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let wname = format!("{prefix}.weight");
        let (shape, _) = weights
            .tensor_info(&wname)
            .ok_or_else(|| anyhow::anyhow!("FP8 TP: weight not found: {wname}"))?;
        let sharded_out = shape[0] / world_size;
        shard_sizes.push(sharded_out);
        total_out += sharded_out;
    }

    let (fused_weight, merged_scale) = if is_online_quant {
        // Online quantization: load BF16 shards → fuse → quantize entire fused weight.
        let bf16_elem_size = first_dtype.size_bytes();
        let bf16_total_bytes = total_out * in_features * bf16_elem_size;
        let bf16_ptr = unsafe { crate::driver::mem_alloc(bf16_total_bytes)? };

        let mut offset = 0usize;
        for (i, prefix) in prefixes.iter().enumerate() {
            let wname = format!("{prefix}.weight");
            let shard_bytes = shard_sizes[i] * in_features * bf16_elem_size;
            unsafe {
                weights.take_shard_into(
                    &wname,
                    0,
                    rank,
                    world_size,
                    bf16_ptr.add(offset),
                    stream,
                )?;
            }
            offset += shard_bytes;
        }

        let num_elements = total_out * in_features;
        let fp8_ptr = unsafe { crate::driver::mem_alloc(num_elements)? };
        let scale_ptr = unsafe { crate::driver::mem_alloc(4)? };

        unsafe {
            crate::kernels::fp8_quantize_weight_bf16_raw(
                bf16_ptr as *const u16,
                fp8_ptr as *mut u8,
                scale_ptr as *mut f32,
                num_elements as i32,
                stream,
            );
            crate::driver::mem_free(bf16_ptr)?;
        }

        let fused_weight =
            unsafe { GpuTensor::new(fp8_ptr, &[total_out, in_features], DType::Fp8E4m3) };
        let scale = unsafe { GpuTensor::new(scale_ptr, &[1], DType::F32) };

        // Consume any checkpoint weight_scale tensors.
        for prefix in prefixes {
            let scale_name = format!("{prefix}.weight_scale");
            if weights.contains(&scale_name) {
                let _ = weights.take(&scale_name);
            }
        }

        (fused_weight, scale)
    } else {
        // Serialized FP8 checkpoint: shard each prefix's weight along dim=0.
        let elem_size = DType::Fp8E4m3.size_bytes();
        let total_bytes = total_out * in_features * elem_size;
        let fused_ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };

        let mut offset = 0usize;
        for (i, prefix) in prefixes.iter().enumerate() {
            let wname = format!("{prefix}.weight");
            let shard_bytes = shard_sizes[i] * in_features * elem_size;
            unsafe {
                weights.take_shard_into(
                    &wname,
                    0,
                    rank,
                    world_size,
                    fused_ptr.add(offset),
                    stream,
                )?;
            }
            offset += shard_bytes;
        }

        let fused_weight =
            unsafe { GpuTensor::new(fused_ptr, &[total_out, in_features], DType::Fp8E4m3) };

        // Check if scales are per-channel or per-tensor.
        let first_scale_name = format!("{}.weight_scale", prefixes[0]);
        let (first_scale_shape, _) = weights
            .tensor_info(&first_scale_name)
            .ok_or_else(|| anyhow::anyhow!("FP8 TP: weight_scale not found: {first_scale_name}"))?;
        let is_per_channel = first_scale_shape.iter().product::<usize>() > 1;

        let merged_scale = if is_per_channel {
            // Per-channel scales: shard along dim=0, concatenate.
            let total_scale_f32_bytes = total_out * 4;
            let scale_ptr = unsafe { crate::driver::mem_alloc(total_scale_f32_bytes)? };
            let mut f32_offset = 0usize;

            for (i, prefix) in prefixes.iter().enumerate() {
                let scale_name = format!("{prefix}.weight_scale");
                let raw_scale = weights.take_shard(&scale_name, 0, rank, world_size)?;
                let shard_scale = ensure_f32_scale(raw_scale, stream)?;
                let shard_bytes = shard_sizes[i] * 4;
                unsafe {
                    crate::driver::memcpy_dtod_async(
                        scale_ptr.add(f32_offset),
                        shard_scale.raw_ptr(),
                        shard_bytes,
                        stream,
                    )?;
                }
                f32_offset += shard_bytes;
            }

            unsafe { GpuTensor::new(scale_ptr, &[total_out], DType::F32) }
        } else {
            // Per-tensor scales: take max across shards, re-quantize.
            let mut shard_scales = Vec::with_capacity(prefixes.len());
            let mut max_scale = 0.0f32;
            for prefix in prefixes {
                let scale_name = format!("{prefix}.weight_scale");
                let scale_cpu = weights.take_to_cpu_f32(&scale_name)?;
                let s = scale_cpu.first().copied().unwrap_or(1.0);
                if s > max_scale {
                    max_scale = s;
                }
                shard_scales.push(s);
            }

            let mut row_offset = 0usize;
            for (i, &shard_scale) in shard_scales.iter().enumerate() {
                if (shard_scale - max_scale).abs() > 1e-12 {
                    unsafe {
                        crate::kernels::fp8_requantize_weight_rows(
                            fused_weight,
                            in_features,
                            row_offset,
                            shard_sizes[i],
                            shard_scale,
                            max_scale,
                            stream,
                        );
                    }
                }
                row_offset += shard_sizes[i];
            }

            let scale_ptr = unsafe { crate::driver::mem_alloc(4)? };
            unsafe {
                crate::driver::memcpy_htod_async(
                    scale_ptr,
                    &max_scale as *const f32 as *const u8,
                    4,
                    stream,
                )?;
            }
            unsafe { GpuTensor::new(scale_ptr, &[1], DType::F32) }
        };

        (fused_weight, merged_scale)
    };

    // Input scale: use first prefix's if available.
    let input_scale_name = format!("{}.input_scale", prefixes[0]);
    let input_scale = if weights.contains(&input_scale_name) {
        let raw = weights.take(&input_scale_name)?;
        Some(ensure_f32_scale(raw, stream)?)
    } else {
        None
    };

    // Consume remaining input_scale tensors.
    for prefix in prefixes.iter().skip(1) {
        let isname = format!("{prefix}.input_scale");
        if weights.contains(&isname) {
            let _ = weights.take(&isname);
        }
    }

    // Fuse bias if present (also sharded along dim=0).
    let bias_name = format!("{}.bias", prefixes[0]);
    let bias = if weights.contains(&bias_name) {
        let (_bias_shape, bias_dtype) = weights
            .tensor_info(&bias_name)
            .ok_or_else(|| anyhow::anyhow!("FP8 TP: bias not found"))?;
        let bias_elem_size = bias_dtype.size_bytes();
        let mut total_bias_bytes = 0;
        for &sz in &shard_sizes {
            total_bias_bytes += sz * bias_elem_size;
        }
        let bias_ptr = unsafe { crate::driver::mem_alloc(total_bias_bytes)? };
        let mut boff = 0;
        for (i, prefix) in prefixes.iter().enumerate() {
            let bname = format!("{prefix}.bias");
            let bbytes = shard_sizes[i] * bias_elem_size;
            unsafe {
                weights.take_shard_into(&bname, 0, rank, world_size, bias_ptr.add(boff), stream)?;
            }
            boff += bbytes;
        }
        let total_bias_elems = total_bias_bytes / bias_elem_size;
        Some(unsafe { GpuTensor::new(bias_ptr, &[total_bias_elems], bias_dtype) })
    } else {
        None
    };

    Ok(crate::layers::Fp8Linear {
        weight: fused_weight,
        weight_scale: merged_scale,
        input_scale,
        bias,
        output_dtype,
    })
}

/// Load a single per-tensor FP8 linear with row-parallel TP (O proj, down proj).
///
/// The weight is sharded along dim=1 (input/reduction dimension). The per-tensor
/// scale is kept as-is since each rank's shard was quantized under the same
/// global scale (a valid upper bound for any shard).
pub fn load_fp8_linear_tp(
    weights: &mut GpuWeights,
    prefix: &str,
    output_dtype: DType,
    rank: usize,
    world_size: usize,
) -> Result<crate::layers::Fp8Linear> {
    let weight_name = format!("{prefix}.weight");
    let scale_name = format!("{prefix}.weight_scale");
    let input_scale_name = format!("{prefix}.input_scale");
    let bias_name = format!("{prefix}.bias");

    let (shape, dtype) = weights
        .tensor_info(&weight_name)
        .ok_or_else(|| anyhow::anyhow!("FP8 TP: weight not found: {weight_name}"))?;
    anyhow::ensure!(shape.len() == 2, "FP8 weight must be 2D");

    let stream = weights.stream();
    let is_online_quant = dtype == DType::BF16 || dtype == DType::F16;

    let (fp8_weight, weight_scale) = if is_online_quant {
        // Online quantization: shard BF16 along dim=1, then quantize.
        anyhow::ensure!(
            dtype == DType::BF16,
            "Online FP8 quant currently supports BF16 only, got {}",
            dtype
        );
        let shard = weights.take_shard(&weight_name, 1, rank, world_size)?;
        let n = shard.dim(0);
        let k = shard.dim(1);
        let num_elements = n * k;

        let fp8_ptr = unsafe { crate::driver::mem_alloc(num_elements)? };
        let scale_ptr = unsafe { crate::driver::mem_alloc(4)? };

        unsafe {
            crate::kernels::fp8_quantize_weight_bf16_raw(
                shard.as_ptr() as *const u16,
                fp8_ptr as *mut u8,
                scale_ptr as *mut f32,
                num_elements as i32,
                std::ptr::null_mut(),
            );
        }

        let fp8_weight = unsafe { GpuTensor::new(fp8_ptr, &[n, k], DType::Fp8E4m3) };
        let weight_scale = unsafe { GpuTensor::new(scale_ptr, &[1], DType::F32) };

        // Consume checkpoint scale if present.
        if weights.contains(&scale_name) {
            let _ = weights.take(&scale_name);
        }

        (fp8_weight, weight_scale)
    } else {
        // Serialized FP8: shard weight along dim=1, keep scale.
        anyhow::ensure!(
            dtype == DType::Fp8E4m3,
            "FP8 TP: expected Fp8E4m3 or BF16, got {dtype}"
        );
        let weight = weights.take_shard(&weight_name, 1, rank, world_size)?;
        let raw_scale = weights.take(&scale_name)?;
        let weight_scale = ensure_f32_scale(raw_scale, stream)?;
        (weight, weight_scale)
    };

    let input_scale = if weights.contains(&input_scale_name) {
        let raw = weights.take(&input_scale_name)?;
        Some(ensure_f32_scale(raw, stream)?)
    } else {
        None
    };

    let bias = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };

    Ok(crate::layers::Fp8Linear {
        weight: fp8_weight,
        weight_scale,
        input_scale,
        bias,
        output_dtype,
    })
}

// ---------------------------------------------------------------------------
// FP8 MoE Expert Weight Loading
// ---------------------------------------------------------------------------

use crate::layers::Linear;
use crate::layers_moe::Fp8FusedMoELayer;

/// Load FP8 MoE expert weights and per-expert scales.
///
/// Stacks per-expert FP8 E4M3 weights into `[E, N, K]` tensors and merges
/// gate/up scales with max() (matching Python's `process_fp8_weight_tensor_strategy_moe`).
///
/// Returns `Fp8FusedMoELayer` with FP8 weights + f32 per-expert scales.
///
/// Weight naming conventions (per-expert):
/// - `gate_name`: "w1" (Mixtral) or "gate_proj" (Qwen/DeepSeek)
/// - `up_name`: "w3" (Mixtral) or "up_proj" (Qwen/DeepSeek)
/// - `down_name`: "w2" (Mixtral) or "down_proj" (Qwen/DeepSeek)
#[allow(clippy::too_many_arguments)]
pub fn load_fp8_moe_experts(
    weights: &mut GpuWeights,
    prefix: &str,
    num_experts: usize,
    intermediate_size: usize,
    hidden_size: usize,
    top_k: usize,
    renormalize: bool,
    gate_name: &str,
    up_name: &str,
    down_name: &str,
) -> Result<Fp8FusedMoELayer> {
    let stream = weights.stream();

    // Gate weight — always dense BF16.
    let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

    // Allocate stacked FP8 expert weight buffers.
    // w1: [E, 2*inter, hidden] FP8 (1 byte/elem)
    // w2: [E, hidden, inter] FP8
    let w1_bytes = num_experts * 2 * intermediate_size * hidden_size;
    let w2_bytes = num_experts * hidden_size * intermediate_size;
    let w1_ptr = unsafe { crate::driver::mem_alloc(w1_bytes)? };
    let w2_ptr = unsafe { crate::driver::mem_alloc(w2_bytes)? };
    weights.record_alloc(w1_ptr, w1_bytes);
    weights.record_alloc(w2_ptr, w2_bytes);

    // Per-expert scales (f32).
    let mut w1_scales_host = vec![0.0f32; num_experts];
    let mut w2_scales_host = vec![0.0f32; num_experts];

    for e in 0..num_experts {
        let gate_prefix = format!("{prefix}.experts.{e}.{gate_name}");
        let up_prefix = format!("{prefix}.experts.{e}.{up_name}");
        let down_prefix = format!("{prefix}.experts.{e}.{down_name}");

        // Copy gate_proj FP8 weight into w1[e, 0..inter, :]
        let expert_w1_offset = e * 2 * intermediate_size * hidden_size;
        let gate_proj_bytes = intermediate_size * hidden_size;
        unsafe {
            weights.take_into(
                &format!("{gate_prefix}.weight"),
                w1_ptr.add(expert_w1_offset),
                stream,
            )?;
            // Copy up_proj FP8 weight into w1[e, inter..2*inter, :]
            weights.take_into(
                &format!("{up_prefix}.weight"),
                w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                stream,
            )?;
            // Copy down_proj FP8 weight into w2[e, :, :]
            let expert_w2_offset = e * hidden_size * intermediate_size;
            weights.take_into(
                &format!("{down_prefix}.weight"),
                w2_ptr.add(expert_w2_offset),
                stream,
            )?;
        }

        // Load per-expert scales.
        // gate_proj.weight_scale and up_proj.weight_scale → w1_scale = max(gate, up)
        // down_proj.weight_scale → w2_scale
        let gate_scale = read_f32_scale(weights, &format!("{gate_prefix}.weight_scale"), stream)?;
        let up_scale = read_f32_scale(weights, &format!("{up_prefix}.weight_scale"), stream)?;
        let down_scale = read_f32_scale(weights, &format!("{down_prefix}.weight_scale"), stream)?;

        w1_scales_host[e] = gate_scale.max(up_scale);
        w2_scales_host[e] = down_scale;
    }

    // Upload per-expert scale vectors to GPU.
    let w1_scale_bytes = num_experts * 4;
    let w1_scale_ptr = unsafe { crate::driver::mem_alloc(w1_scale_bytes)? };
    let w2_scale_ptr = unsafe { crate::driver::mem_alloc(w1_scale_bytes)? };
    weights.record_alloc(w1_scale_ptr, w1_scale_bytes);
    weights.record_alloc(w2_scale_ptr, w1_scale_bytes);
    unsafe {
        crate::driver::memcpy_htod_async(
            w1_scale_ptr,
            w1_scales_host.as_ptr() as *const u8,
            w1_scale_bytes,
            stream,
        )?;
        crate::driver::memcpy_htod_async(
            w2_scale_ptr,
            w2_scales_host.as_ptr() as *const u8,
            w1_scale_bytes,
            stream,
        )?;
    }

    let w1 = unsafe {
        GpuTensor::new(
            w1_ptr,
            &[num_experts, 2 * intermediate_size, hidden_size],
            DType::Fp8E4m3,
        )
    };
    let w2 = unsafe {
        GpuTensor::new(
            w2_ptr,
            &[num_experts, hidden_size, intermediate_size],
            DType::Fp8E4m3,
        )
    };
    let w1_scale = unsafe { GpuTensor::new(w1_scale_ptr, &[num_experts], DType::F32) };
    let w2_scale = unsafe { GpuTensor::new(w2_scale_ptr, &[num_experts], DType::F32) };

    Ok(Fp8FusedMoELayer {
        gate,
        w1,
        w2,
        w1_scale,
        w2_scale,
        num_experts,
        top_k,
        intermediate_size,
        hidden_size,
        renormalize,
        e_score_correction_bias: None,
        n_expert_group: 0,
        topk_group: 0,
        routed_scaling_factor: 1.0,
    })
}

/// Load FP8 block-quantized MoE expert weights, keeping FP8 storage with
/// 3D block scale tensors. Returns `Fp8BlockFusedMoELayer`.
///
/// FP8 weights are stacked into `[E, N, K]` buffers (same as per-tensor path).
/// Block scales are stacked into `[E, ceil(N/bn), ceil(K/bk)]` f32 tensors.
/// For w1 (gate+up), gate and up scale rows are concatenated along dim=1.
///
/// With TP (`tp != None` as `Some((rank, world_size))`), each expert's
/// intermediate dimension is sharded: gate/up weights sharded along dim=0,
/// down along dim=1, scales correspondingly.
#[allow(clippy::too_many_arguments)]
pub fn load_fp8_block_moe_experts(
    weights: &mut GpuWeights,
    prefix: &str,
    num_experts: usize,
    intermediate_size: usize,
    hidden_size: usize,
    top_k: usize,
    renormalize: bool,
    gate_name: &str,
    up_name: &str,
    down_name: &str,
    tp: Option<(usize, usize)>,
) -> Result<crate::layers_moe::Fp8BlockFusedMoELayer> {
    let stream = weights.stream();
    let (rank, world_size) = tp.unwrap_or((0, 1));
    let ipp = intermediate_size / world_size; // per-partition intermediate size

    // Gate weight — always dense BF16 (not sharded, same on all ranks).
    let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

    // Derive block size from first expert's scale shape.
    let first_gate_weight = format!("{prefix}.experts.0.{gate_name}.weight");
    let first_gate_scale = format!("{prefix}.experts.0.{gate_name}.weight_scale_inv");
    let (w_shape, _) = weights
        .tensor_info(&first_gate_weight)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate_weight}"))?;
    let (s_shape, _) = weights
        .tensor_info(&first_gate_scale)
        .ok_or_else(|| anyhow::anyhow!("scale not found: {first_gate_scale}"))?;
    let block_n = w_shape[0] / s_shape[0];
    let block_k = w_shape[1] / s_shape[1];

    // Scale dimensions for stacked tensors (using per-partition sizes).
    let w1_n = 2 * ipp;
    let w1_k = hidden_size;
    let w2_n = hidden_size;
    let w2_k = ipp;
    let w1_scale_rows = w1_n.div_ceil(block_n);
    let w1_scale_cols = w1_k.div_ceil(block_k);
    let w2_scale_rows = w2_n.div_ceil(block_n);
    let w2_scale_cols = w2_k.div_ceil(block_k);
    let gate_scale_rows = ipp.div_ceil(block_n);

    // Allocate stacked FP8 weight buffers (per-partition sizes).
    let w1_bytes = num_experts * w1_n * w1_k;
    let w2_bytes = num_experts * w2_n * w2_k;
    let w1_ptr = unsafe { crate::driver::mem_alloc(w1_bytes)? };
    let w2_ptr = unsafe { crate::driver::mem_alloc(w2_bytes)? };
    weights.record_alloc(w1_ptr, w1_bytes);
    weights.record_alloc(w2_ptr, w2_bytes);

    // Allocate stacked scale buffers: [E, scale_rows, scale_cols] f32.
    let w1_scale_bytes = num_experts * w1_scale_rows * w1_scale_cols * 4;
    let w2_scale_bytes = num_experts * w2_scale_rows * w2_scale_cols * 4;
    let w1_scale_ptr = unsafe { crate::driver::mem_alloc(w1_scale_bytes)? };
    let w2_scale_ptr = unsafe { crate::driver::mem_alloc(w2_scale_bytes)? };
    weights.record_alloc(w1_scale_ptr, w1_scale_bytes);
    weights.record_alloc(w2_scale_ptr, w2_scale_bytes);

    for e in 0..num_experts {
        let gate_pfx = format!("{prefix}.experts.{e}.{gate_name}");
        let up_pfx = format!("{prefix}.experts.{e}.{up_name}");
        let down_pfx = format!("{prefix}.experts.{e}.{down_name}");

        // --- FP8 weights: shard and copy into stacked buffers ---
        let expert_w1_offset = e * w1_n * w1_k;
        let gate_proj_bytes = ipp * hidden_size;
        unsafe {
            if world_size > 1 {
                // gate_proj [inter, hidden] → shard dim=0 → [ipp, hidden]
                weights.take_shard_into(
                    &format!("{gate_pfx}.weight"),
                    0,
                    rank,
                    world_size,
                    w1_ptr.add(expert_w1_offset),
                    stream,
                )?;
                // up_proj [inter, hidden] → shard dim=0 → [ipp, hidden]
                weights.take_shard_into(
                    &format!("{up_pfx}.weight"),
                    0,
                    rank,
                    world_size,
                    w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                    stream,
                )?;
                // down_proj [hidden, inter] → shard dim=1 → [hidden, ipp]
                let expert_w2_offset = e * w2_n * w2_k;
                weights.take_shard_into(
                    &format!("{down_pfx}.weight"),
                    1,
                    rank,
                    world_size,
                    w2_ptr.add(expert_w2_offset),
                    stream,
                )?;
            } else {
                weights.take_into(
                    &format!("{gate_pfx}.weight"),
                    w1_ptr.add(expert_w1_offset),
                    stream,
                )?;
                weights.take_into(
                    &format!("{up_pfx}.weight"),
                    w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                    stream,
                )?;
                let expert_w2_offset = e * w2_n * w2_k;
                weights.take_into(
                    &format!("{down_pfx}.weight"),
                    w2_ptr.add(expert_w2_offset),
                    stream,
                )?;
            }
        }

        // --- Block scales: shard and stack into 3D tensors ---
        // gate_proj scale: [ceil(inter/bn), ceil(hidden/bk)] → shard dim=0 → [ceil(ipp/bn), ...]
        let expert_w1_scale_offset = e * w1_scale_rows * w1_scale_cols * 4;
        let gate_scale = if world_size > 1 {
            let raw =
                weights.take_shard(&format!("{gate_pfx}.weight_scale_inv"), 0, rank, world_size)?;
            ensure_f32_scale(raw, stream)?
        } else {
            let raw = weights.take(&format!("{gate_pfx}.weight_scale_inv"))?;
            ensure_f32_scale(raw, stream)?
        };
        let gate_scale_bytes = gate_scale_rows * w1_scale_cols * 4;
        unsafe {
            crate::driver::memcpy_dtod_async(
                w1_scale_ptr.add(expert_w1_scale_offset),
                gate_scale.raw_ptr(),
                gate_scale_bytes,
                stream,
            )?;
        }

        // up_proj scale: same sharding as gate
        let up_scale = if world_size > 1 {
            let raw =
                weights.take_shard(&format!("{up_pfx}.weight_scale_inv"), 0, rank, world_size)?;
            ensure_f32_scale(raw, stream)?
        } else {
            let raw = weights.take(&format!("{up_pfx}.weight_scale_inv"))?;
            ensure_f32_scale(raw, stream)?
        };
        let up_scale_rows = w1_scale_rows - gate_scale_rows;
        let up_scale_bytes = up_scale_rows * w1_scale_cols * 4;
        unsafe {
            crate::driver::memcpy_dtod_async(
                w1_scale_ptr.add(expert_w1_scale_offset + gate_scale_bytes),
                up_scale.raw_ptr(),
                up_scale_bytes,
                stream,
            )?;
        }

        // down_proj scale: [ceil(hidden/bn), ceil(inter/bk)] → shard dim=1 → [..., ceil(ipp/bk)]
        let expert_w2_scale_offset = e * w2_scale_rows * w2_scale_cols * 4;
        let down_scale = if world_size > 1 {
            let raw =
                weights.take_shard(&format!("{down_pfx}.weight_scale_inv"), 1, rank, world_size)?;
            ensure_f32_scale(raw, stream)?
        } else {
            let raw = weights.take(&format!("{down_pfx}.weight_scale_inv"))?;
            ensure_f32_scale(raw, stream)?
        };
        let down_scale_bytes = w2_scale_rows * w2_scale_cols * 4;
        unsafe {
            crate::driver::memcpy_dtod_async(
                w2_scale_ptr.add(expert_w2_scale_offset),
                down_scale.raw_ptr(),
                down_scale_bytes,
                stream,
            )?;
        }

        // Consume input_scale if present.
        for pfx in &[&gate_pfx, &up_pfx, &down_pfx] {
            let is_name = format!("{pfx}.input_scale");
            if weights.contains(&is_name) {
                let _ = weights.take(&is_name);
            }
        }
    }

    let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, w1_n, w1_k], DType::Fp8E4m3) };
    let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, w2_n, w2_k], DType::Fp8E4m3) };
    let w1_scale_inv = unsafe {
        GpuTensor::new(
            w1_scale_ptr,
            &[num_experts, w1_scale_rows, w1_scale_cols],
            DType::F32,
        )
    };
    let w2_scale_inv = unsafe {
        GpuTensor::new(
            w2_scale_ptr,
            &[num_experts, w2_scale_rows, w2_scale_cols],
            DType::F32,
        )
    };

    Ok(crate::layers_moe::Fp8BlockFusedMoELayer {
        gate,
        w1,
        w2,
        w1_scale_inv,
        w2_scale_inv,
        block_size: [block_n, block_k],
        num_experts,
        top_k,
        intermediate_size: ipp,
        hidden_size,
        renormalize,
        e_score_correction_bias: None,
        n_expert_group: 0,
        topk_group: 0,
        routed_scaling_factor: 1.0,
    })
}

/// Read a single f32 scalar scale from a weight tensor.
/// Handles the case where the scale is stored as f32, BF16, or F16.
fn read_f32_scale(weights: &mut GpuWeights, name: &str, stream: CUstream) -> Result<f32> {
    let scale_tensor = weights.take(name)?;
    let num_bytes = scale_tensor.numel() * scale_tensor.dtype().size_bytes();
    let mut host_buf = vec![0u8; num_bytes];
    unsafe {
        crate::driver::memcpy_dtoh_async(
            host_buf.as_mut_ptr(),
            scale_tensor.raw_ptr(),
            num_bytes,
            stream,
        )?;
        crate::driver::stream_synchronize(stream)?;
    }

    let val = match scale_tensor.dtype() {
        DType::F32 => {
            let p = host_buf.as_ptr() as *const f32;
            unsafe { *p }
        }
        DType::BF16 => {
            let bits = u16::from_le_bytes([host_buf[0], host_buf[1]]);
            half::bf16::from_bits(bits).to_f32()
        }
        DType::F16 => {
            let bits = u16::from_le_bytes([host_buf[0], host_buf[1]]);
            half::f16::from_bits(bits).to_f32()
        }
        dt => anyhow::bail!("read_f32_scale: unsupported dtype {dt}"),
    };
    Ok(val)
}

// ---------------------------------------------------------------------------
// Marlin MoE Expert Weight Loading (AWQ/GPTQ INT4)
// ---------------------------------------------------------------------------

use crate::layers_moe::MarlinFusedMoELayer;

/// Load MoE expert weights in AWQ/GPTQ INT4 format and repack to Marlin.
///
/// For each expert, loads gate+up (concat along N) → repack → w1[e],
/// and down → repack → w2[e]. Scales and zero points are similarly
/// processed per-expert and stacked.
///
/// Returns a `MarlinFusedMoELayer` with all expert weights in
/// `[num_experts, ...]` stacked Marlin-packed format.
#[allow(clippy::too_many_arguments)]
pub fn load_marlin_moe_layer(
    weights: &mut GpuWeights,
    prefix: &str,
    num_experts: usize,
    intermediate_size: usize,
    hidden_size: usize,
    top_k: usize,
    renormalize: bool,
    qconfig: &crate::quant::QuantConfig,
    device_id: i32,
) -> Result<MarlinFusedMoELayer> {
    let stream = weights.stream();

    let (group_size, has_zp, b_type_id) = match qconfig {
        crate::quant::QuantConfig::Awq(cfg) => (cfg.group_size, true, 1i32),
        crate::quant::QuantConfig::Gptq(cfg) => (cfg.group_size, false, 0i32),
        _ => bail!(
            "load_marlin_moe_layer: unsupported quant config {:?}",
            qconfig
        ),
    };

    let num_groups_w1 = hidden_size.checked_div(group_size).unwrap_or(1);
    let num_groups_w2 = intermediate_size.checked_div(group_size).unwrap_or(1);

    // Marlin tile size: each u32 contains 8 INT4 values (4 bits each).
    // Marlin-packed shape for [K, N]: [K*N/8] u32.
    // We store [E, K*N/8] for the stacked experts.
    let w1_n = 2 * intermediate_size; // gate+up fused
    let w1_packed_per_expert = hidden_size * w1_n / 8; // u32 count
    let w2_packed_per_expert = intermediate_size * hidden_size / 8;

    // Allocate stacked expert weight buffers (persistent alloc).
    let w1_total_bytes = num_experts * w1_packed_per_expert * 4;
    let w2_total_bytes = num_experts * w2_packed_per_expert * 4;
    let w1_ptr = unsafe { driver::mem_alloc(w1_total_bytes)? };
    let w2_ptr = unsafe { driver::mem_alloc(w2_total_bytes)? };
    weights.record_alloc(w1_ptr, w1_total_bytes);
    weights.record_alloc(w2_ptr, w2_total_bytes);

    // Determine scale dtype from first expert's scales.
    // Try standard GPTQ/AWQ naming first, then compressed-tensors naming.
    let first_scales_name = format!("{prefix}.experts.0.gate_proj.scales");
    let first_ct_scales_name = format!("{prefix}.experts.0.gate_proj.weight_scale");
    let scales_dtype = weights
        .tensor_info(&first_scales_name)
        .or_else(|| weights.tensor_info(&first_ct_scales_name))
        .map(|(_, dt)| dt)
        .unwrap_or(DType::BF16);
    let scale_elem = scales_dtype.size_bytes();

    // Allocate stacked scales.
    let w1_scales_bytes = num_experts * num_groups_w1 * w1_n * scale_elem;
    let w2_scales_bytes = num_experts * num_groups_w2 * hidden_size * scale_elem;
    let w1_scales_ptr = unsafe { driver::mem_alloc(w1_scales_bytes)? };
    let w2_scales_ptr = unsafe { driver::mem_alloc(w2_scales_bytes)? };
    weights.record_alloc(w1_scales_ptr, w1_scales_bytes);
    weights.record_alloc(w2_scales_ptr, w2_scales_bytes);

    // Allocate stacked zero points (AWQ only).
    let (w1_zeros_ptr, w2_zeros_ptr) = if has_zp {
        let w1_zp_bytes = num_experts * num_groups_w1 * (w1_n / 8) * 4; // u32
        let w2_zp_bytes = num_experts * num_groups_w2 * (hidden_size / 8) * 4;
        let p1 = unsafe { driver::mem_alloc(w1_zp_bytes)? };
        let p2 = unsafe { driver::mem_alloc(w2_zp_bytes)? };
        weights.record_alloc(p1, w1_zp_bytes);
        weights.record_alloc(p2, w2_zp_bytes);
        (Some(p1), Some(p2))
    } else {
        (None, None)
    };

    let is_gptq = b_type_id == 0;

    for e in 0..num_experts {
        let gate_prefix = format!("{prefix}.experts.{e}.gate_proj");
        let up_prefix = format!("{prefix}.experts.{e}.up_proj");
        let down_prefix = format!("{prefix}.experts.{e}.down_proj");

        if is_gptq {
            // GPTQ / compressed-tensors path: qweight [K/8, N], gptq_repack_into
            let (gate_qw, gate_sc) = load_expert_gptq_cpu(weights, &gate_prefix)?;
            let (up_qw, up_sc) = load_expert_gptq_cpu(weights, &up_prefix)?;

            // GPTQ qweight is [K/8, N] — concat along dim1 gives [K/8, N1+N2]
            let k_packed = gate_qw.1[0]; // K/8
            let fused_k = k_packed * 8;
            let fused_n = intermediate_size * 2;

            let fused_qw = {
                let gate_n = gate_qw.1[1];
                let up_n = up_qw.1[1];
                let elem = 4usize; // i32
                let row_gate = gate_n * elem;
                let row_up = up_n * elem;
                let row_out = (gate_n + up_n) * elem;
                let mut out = vec![0u8; k_packed * row_out];
                for r in 0..k_packed {
                    out[r * row_out..r * row_out + row_gate]
                        .copy_from_slice(&gate_qw.0[r * row_gate..(r + 1) * row_gate]);
                    out[r * row_out + row_gate..r * row_out + row_out]
                        .copy_from_slice(&up_qw.0[r * row_up..(r + 1) * row_up]);
                }
                out
            };

            // Upload fused qweight to GPU for repack.
            let qw_nbytes = fused_qw.len();
            let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(qw_gpu_ptr, fused_qw.as_ptr(), qw_nbytes, stream)?;
            }
            let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &[k_packed, fused_n], DType::I32) };

            // Repack GPTQ → Marlin into the stacked buffer at expert offset.
            let expert_w1_offset = e * w1_packed_per_expert * 4;
            unsafe {
                crate::kernels::gptq_repack_into(
                    qw_gpu,
                    None, // no sort_indices for MoE experts
                    w1_ptr.add(expert_w1_offset),
                    fused_k,
                    fused_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(qw_gpu_ptr)?;
            }

            // Fuse and permute scales: [num_groups, N1] + [num_groups, N2] → [num_groups, N_total]
            let fused_sc =
                concat_u16_dim1(&gate_sc.0, &up_sc.0, gate_sc.1[0], gate_sc.1[1], up_sc.1[1]);
            let mut scales_u16 = fused_sc;
            crate::quant::marlin_permute_scales(&mut scales_u16, fused_k, fused_n, group_size);

            let expert_scales_offset = e * num_groups_w1 * fused_n * scale_elem;
            let scales_bytes: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
            unsafe {
                driver::memcpy_htod_async(
                    w1_scales_ptr.add(expert_scales_offset),
                    scales_bytes.as_ptr(),
                    scales_bytes.len(),
                    stream,
                )?;
            }

            // --- w2: down proj ---
            let (down_qw, down_sc) = load_expert_gptq_cpu(weights, &down_prefix)?;
            let down_k_packed = down_qw.1[0]; // K/8
            let down_k = down_k_packed * 8;
            let down_n = down_qw.1[1];

            let down_qw_nbytes = down_qw.0.len();
            let down_qw_gpu_ptr = unsafe { driver::mem_alloc(down_qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(
                    down_qw_gpu_ptr,
                    down_qw.0.as_ptr(),
                    down_qw_nbytes,
                    stream,
                )?;
            }
            let down_qw_gpu =
                unsafe { GpuTensor::new(down_qw_gpu_ptr, &[down_k_packed, down_n], DType::I32) };

            let expert_w2_offset = e * w2_packed_per_expert * 4;
            unsafe {
                crate::kernels::gptq_repack_into(
                    down_qw_gpu,
                    None,
                    w2_ptr.add(expert_w2_offset),
                    down_k,
                    down_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(down_qw_gpu_ptr)?;
            }

            let mut down_scales_u16 = bytes_to_u16(&down_sc.0);
            crate::quant::marlin_permute_scales(&mut down_scales_u16, down_k, down_n, group_size);
            let expert_w2_scales_offset = e * num_groups_w2 * hidden_size * scale_elem;
            let down_scales_bytes: Vec<u8> = down_scales_u16
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            unsafe {
                driver::memcpy_htod_async(
                    w2_scales_ptr.add(expert_w2_scales_offset),
                    down_scales_bytes.as_ptr(),
                    down_scales_bytes.len(),
                    stream,
                )?;
            }
        } else {
            // AWQ path: qweight [K, N/8], awq_repack_into
            // --- w1: gate + up fused ---
            let (gate_qw, gate_sc, gate_zp) = load_expert_awq_cpu(weights, &gate_prefix)?;
            let (up_qw, up_sc, up_zp) = load_expert_awq_cpu(weights, &up_prefix)?;

            // AWQ qweight is [K, N/8] — concat along dim1 gives [K, (N1+N2)/8]
            let fused_qw =
                concat_bytes_dim1(&gate_qw.0, &up_qw.0, gate_qw.1[0], gate_qw.1[1], up_qw.1[1]);

            // Upload fused qweight to GPU for repack.
            let fused_k = gate_qw.1[0]; // K
            let fused_n = intermediate_size * 2; // gate_N + up_N
            let qw_nbytes = fused_qw.len();
            let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(qw_gpu_ptr, fused_qw.as_ptr(), qw_nbytes, stream)?;
            }
            let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &[fused_k, fused_n / 8], DType::U32) };

            // Repack AWQ → Marlin into the stacked buffer at expert offset.
            let expert_w1_offset = e * w1_packed_per_expert * 4;
            unsafe {
                crate::kernels::awq_repack_into(
                    qw_gpu,
                    w1_ptr.add(expert_w1_offset),
                    fused_k,
                    fused_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(qw_gpu_ptr)?;
            }

            // Fuse and permute scales.
            let fused_sc =
                concat_u16_dim1(&gate_sc.0, &up_sc.0, gate_sc.1[0], gate_sc.1[1], up_sc.1[1]);
            let mut scales_u16 = fused_sc;
            crate::quant::marlin_permute_scales(&mut scales_u16, fused_k, fused_n, group_size);

            let expert_scales_offset = e * num_groups_w1 * fused_n * scale_elem;
            let scales_bytes: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
            unsafe {
                driver::memcpy_htod_async(
                    w1_scales_ptr.add(expert_scales_offset),
                    scales_bytes.as_ptr(),
                    scales_bytes.len(),
                    stream,
                )?;
            }

            // Zero points (AWQ only).
            if has_zp && let (Some(gate_zp), Some(up_zp)) = (gate_zp, up_zp) {
                let fused_zp_u32 =
                    concat_u32_dim1(&gate_zp.0, &up_zp.0, gate_zp.1[0], gate_zp.1[1], up_zp.1[1]);
                let marlin_zp =
                    crate::quant::awq_to_marlin_zero_points(&fused_zp_u32, num_groups_w1, fused_n);
                let zp_bytes: Vec<u8> = marlin_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let expert_zp_offset = e * num_groups_w1 * (fused_n / 8) * 4;
                unsafe {
                    driver::memcpy_htod_async(
                        w1_zeros_ptr.unwrap().add(expert_zp_offset),
                        zp_bytes.as_ptr(),
                        zp_bytes.len(),
                        stream,
                    )?;
                }
            }

            // --- w2: down proj ---
            let (down_qw, down_sc, down_zp) = load_expert_awq_cpu(weights, &down_prefix)?;
            let down_k = down_qw.1[0];
            let down_n = down_qw.1[1] * 8;

            // Upload and repack.
            let down_qw_nbytes = down_qw.0.len();
            let down_qw_gpu_ptr = unsafe { driver::mem_alloc(down_qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(
                    down_qw_gpu_ptr,
                    down_qw.0.as_ptr(),
                    down_qw_nbytes,
                    stream,
                )?;
            }
            let down_qw_gpu =
                unsafe { GpuTensor::new(down_qw_gpu_ptr, &[down_k, down_n / 8], DType::U32) };

            let expert_w2_offset = e * w2_packed_per_expert * 4;
            unsafe {
                crate::kernels::awq_repack_into(
                    down_qw_gpu,
                    w2_ptr.add(expert_w2_offset),
                    down_k,
                    down_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(down_qw_gpu_ptr)?;
            }

            // Down scales.
            let mut down_scales_u16 = bytes_to_u16(&down_sc.0);
            crate::quant::marlin_permute_scales(&mut down_scales_u16, down_k, down_n, group_size);
            let expert_w2_scales_offset = e * num_groups_w2 * hidden_size * scale_elem;
            let down_scales_bytes: Vec<u8> = down_scales_u16
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            unsafe {
                driver::memcpy_htod_async(
                    w2_scales_ptr.add(expert_w2_scales_offset),
                    down_scales_bytes.as_ptr(),
                    down_scales_bytes.len(),
                    stream,
                )?;
            }

            // Down zero points.
            if has_zp && let Some(down_zp) = down_zp {
                let down_zp_u32 = bytes_to_u32(&down_zp.0);
                let marlin_zp =
                    crate::quant::awq_to_marlin_zero_points(&down_zp_u32, num_groups_w2, down_n);
                let zp_bytes: Vec<u8> = marlin_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let expert_w2_zp_offset = e * num_groups_w2 * (hidden_size / 8) * 4;
                unsafe {
                    driver::memcpy_htod_async(
                        w2_zeros_ptr.unwrap().add(expert_w2_zp_offset),
                        zp_bytes.as_ptr(),
                        zp_bytes.len(),
                        stream,
                    )?;
                }
            }
        }
    }

    unsafe { driver::stream_synchronize(stream)? };

    // Build stacked tensors.
    let w1_gpu =
        unsafe { GpuTensor::new(w1_ptr, &[num_experts, w1_packed_per_expert], DType::U32) };
    let w2_gpu =
        unsafe { GpuTensor::new(w2_ptr, &[num_experts, w2_packed_per_expert], DType::U32) };
    let w1_scales_gpu = unsafe {
        GpuTensor::new(
            w1_scales_ptr,
            &[num_experts, num_groups_w1, w1_n],
            scales_dtype,
        )
    };
    let w2_scales_gpu = unsafe {
        GpuTensor::new(
            w2_scales_ptr,
            &[num_experts, num_groups_w2, hidden_size],
            scales_dtype,
        )
    };

    let w1_zeros_gpu = w1_zeros_ptr
        .map(|p| unsafe { GpuTensor::new(p, &[num_experts, num_groups_w1, w1_n / 8], DType::U32) });
    let w2_zeros_gpu = w2_zeros_ptr.map(|p| unsafe {
        GpuTensor::new(
            p,
            &[num_experts, num_groups_w2, hidden_size / 8],
            DType::U32,
        )
    });

    // Allocate workspace (barrier locks for Marlin kernel).
    // Python uses sms * 4. We use a fixed upper bound; the kernel clamps internally.
    let workspace_bytes = 256 * 4 * std::mem::size_of::<i32>(); // 256 SMs * 4
    let workspace_ptr = unsafe { driver::mem_alloc(workspace_bytes)? };
    weights.record_alloc(workspace_ptr, workspace_bytes);
    let workspace_gpu =
        unsafe { GpuTensor::new(workspace_ptr, &[workspace_bytes / 4], DType::I32) };

    // Load router gate (always dense).
    let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

    Ok(MarlinFusedMoELayer {
        gate,
        w1: w1_gpu,
        w2: w2_gpu,
        w1_scales: w1_scales_gpu,
        w2_scales: w2_scales_gpu,
        w1_zeros: w1_zeros_gpu,
        w2_zeros: w2_zeros_gpu,
        workspace: workspace_gpu,
        num_experts,
        top_k,
        intermediate_size,
        hidden_size,
        group_size,
        has_zp,
        b_type_id,
        renormalize,
        e_score_correction_bias: None,
        n_expert_group: 0,
        topk_group: 0,
        routed_scaling_factor: 1.0,
    })
}

/// Build the stacked Marlin INT4 expert weights for the Gemma-4 SwitchGLU MoE
/// from a per-expert compressed-tensors / AutoGPTQ checkpoint, WITHOUT a router
/// gate (Gemma supplies its own folded-RMSNorm + per_expert_scale routing).
///
/// Mirrors the GPTQ branch of [`load_marlin_moe_layer`] but for the
/// `experts.{e}.{gate,up,down}_proj` naming Gemma uses, and keeps experts INT4
/// in VRAM (~0.4 GiB) — the dequant-to-bf16 path would need ~43 GiB and not fit
/// one 46 GB card. compressed-tensors symmetric INT4 packs as uint4b8 (the
/// `load_expert_gptq_cpu` helper transposes `weight_packed [N,K/8]` → GPTQ-native
/// `[K/8,N]`; `b_type_id = 0` = kU4B8 in `marlin_moe_gemm`).
///
/// Returns `(w1, w2, w1_scales, w2_scales, workspace)`:
/// * `w1`: fused gate+up `[E, hidden*2*moe_inter/8]` u32 Marlin
/// * `w2`: down `[E, moe_inter*hidden/8]` u32 Marlin
/// * `w1_scales`: `[E, hidden/group_size, 2*moe_inter]`
/// * `w2_scales`: `[E, moe_inter/group_size, hidden]`
///
#[allow(clippy::too_many_arguments)]
/// Re-encode a slice of IEEE-754 half (F16) bit patterns in place as bf16 bit
/// patterns. The Marlin MoE GEMM reads `b_scales` as the ACTIVATION scalar type
/// (`marlin_moe_gemm` dispatches on `a.dtype()`, which is bf16 for gemma-4), so
/// F16 on-disk scales MUST be converted to bf16 — otherwise the kernel
/// reinterprets the F16 bits as bf16 (a real scale of ~0.15 becomes ~1e-9) and
/// every expert weight is annihilated to ~zero, silently killing the entire
/// sparse-MoE contribution while the dense MLP keeps output fluent.
fn f16_bits_to_bf16_bits(scales: &mut [u16]) {
    for s in scales.iter_mut() {
        *s = half::bf16::from_f32(half::f16::from_bits(*s).to_f32()).to_bits();
    }
}

pub fn load_gemma_marlin_experts(
    weights: &mut GpuWeights,
    experts_prefix: &str,
    num_experts: usize,
    intermediate_size: usize,
    hidden_size: usize,
    group_size: usize,
    device_id: i32,
) -> Result<(GpuTensor, GpuTensor, GpuTensor, GpuTensor, GpuTensor)> {
    let stream = weights.stream();

    let num_groups_w1 = hidden_size.checked_div(group_size).unwrap_or(1);
    let num_groups_w2 = intermediate_size.checked_div(group_size).unwrap_or(1);
    let w1_n = 2 * intermediate_size;
    let w1_packed_per_expert = hidden_size * w1_n / 8;
    let w2_packed_per_expert = intermediate_size * hidden_size / 8;

    let w1_total_bytes = num_experts * w1_packed_per_expert * 4;
    let w2_total_bytes = num_experts * w2_packed_per_expert * 4;
    let w1_ptr = unsafe { driver::mem_alloc(w1_total_bytes)? };
    let w2_ptr = unsafe { driver::mem_alloc(w2_total_bytes)? };
    weights.record_alloc(w1_ptr, w1_total_bytes);
    weights.record_alloc(w2_ptr, w2_total_bytes);

    let first_ct = format!("{experts_prefix}.0.gate_proj.weight_scale");
    let first_gptq = format!("{experts_prefix}.0.gate_proj.scales");
    let scales_dtype = weights
        .tensor_info(&first_ct)
        .or_else(|| weights.tensor_info(&first_gptq))
        .map(|(_, dt)| dt)
        .unwrap_or(DType::BF16);
    // The Marlin MoE GEMM reads scales as the bf16 activation type; F16 on-disk
    // scales are re-encoded to bf16 below (else the experts go dead — see
    // `f16_bits_to_bf16_bits`). bf16/f16 are both 2 bytes so buffer sizing,
    // permute, and offsets are identical; only the 16-bit encoding changes.
    anyhow::ensure!(
        matches!(scales_dtype, DType::F16 | DType::BF16),
        "load_gemma_marlin_experts: expert weight_scale dtype must be F16 or BF16 \
         (got {scales_dtype:?}); the Marlin GEMM reads scales as bf16."
    );
    let out_scales_dtype = DType::BF16;
    let scale_elem = scales_dtype.size_bytes();

    let w1_scales_bytes = num_experts * num_groups_w1 * w1_n * scale_elem;
    let w2_scales_bytes = num_experts * num_groups_w2 * hidden_size * scale_elem;
    let w1_scales_ptr = unsafe { driver::mem_alloc(w1_scales_bytes)? };
    let w2_scales_ptr = unsafe { driver::mem_alloc(w2_scales_bytes)? };
    weights.record_alloc(w1_scales_ptr, w1_scales_bytes);
    weights.record_alloc(w2_scales_ptr, w2_scales_bytes);

    for e in 0..num_experts {
        let gate_prefix = format!("{experts_prefix}.{e}.gate_proj");
        let up_prefix = format!("{experts_prefix}.{e}.up_proj");
        let down_prefix = format!("{experts_prefix}.{e}.down_proj");

        // --- w1: gate + up fused (concat along N) ---
        let (gate_qw, gate_sc) = load_expert_gptq_cpu(weights, &gate_prefix)?;
        let (up_qw, up_sc) = load_expert_gptq_cpu(weights, &up_prefix)?;

        let k_packed = gate_qw.1[0]; // K/8
        let fused_k = k_packed * 8;
        let fused_n = intermediate_size * 2;

        let fused_qw = {
            let gate_n = gate_qw.1[1];
            let up_n = up_qw.1[1];
            let elem = 4usize; // i32
            let row_gate = gate_n * elem;
            let row_up = up_n * elem;
            let row_out = (gate_n + up_n) * elem;
            let mut out = vec![0u8; k_packed * row_out];
            for r in 0..k_packed {
                out[r * row_out..r * row_out + row_gate]
                    .copy_from_slice(&gate_qw.0[r * row_gate..(r + 1) * row_gate]);
                out[r * row_out + row_gate..r * row_out + row_out]
                    .copy_from_slice(&up_qw.0[r * row_up..(r + 1) * row_up]);
            }
            out
        };

        let qw_nbytes = fused_qw.len();
        let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
        unsafe { driver::memcpy_htod_async(qw_gpu_ptr, fused_qw.as_ptr(), qw_nbytes, stream)? };
        let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &[k_packed, fused_n], DType::I32) };

        let expert_w1_offset = e * w1_packed_per_expert * 4;
        unsafe {
            crate::kernels::gptq_repack_into(
                qw_gpu,
                None,
                w1_ptr.add(expert_w1_offset),
                fused_k,
                fused_n,
                device_id,
                stream,
            );
            driver::stream_synchronize(stream)?;
            driver::mem_free(qw_gpu_ptr)?;
        }

        let fused_sc =
            concat_u16_dim1(&gate_sc.0, &up_sc.0, gate_sc.1[0], gate_sc.1[1], up_sc.1[1]);
        let mut scales_u16 = fused_sc;
        if scales_dtype == DType::F16 {
            f16_bits_to_bf16_bits(&mut scales_u16);
        }
        crate::quant::marlin_permute_scales(&mut scales_u16, fused_k, fused_n, group_size);
        let expert_scales_offset = e * num_groups_w1 * fused_n * scale_elem;
        let scales_bytes: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        unsafe {
            driver::memcpy_htod_async(
                w1_scales_ptr.add(expert_scales_offset),
                scales_bytes.as_ptr(),
                scales_bytes.len(),
                stream,
            )?;
        }

        // --- w2: down proj ---
        let (down_qw, down_sc) = load_expert_gptq_cpu(weights, &down_prefix)?;
        let down_k_packed = down_qw.1[0];
        let down_k = down_k_packed * 8;
        let down_n = down_qw.1[1];

        let down_qw_nbytes = down_qw.0.len();
        let down_qw_gpu_ptr = unsafe { driver::mem_alloc(down_qw_nbytes)? };
        unsafe {
            driver::memcpy_htod_async(down_qw_gpu_ptr, down_qw.0.as_ptr(), down_qw_nbytes, stream)?;
        }
        let down_qw_gpu =
            unsafe { GpuTensor::new(down_qw_gpu_ptr, &[down_k_packed, down_n], DType::I32) };

        let expert_w2_offset = e * w2_packed_per_expert * 4;
        unsafe {
            crate::kernels::gptq_repack_into(
                down_qw_gpu,
                None,
                w2_ptr.add(expert_w2_offset),
                down_k,
                down_n,
                device_id,
                stream,
            );
            driver::stream_synchronize(stream)?;
            driver::mem_free(down_qw_gpu_ptr)?;
        }

        let mut down_scales_u16 = bytes_to_u16(&down_sc.0);
        if scales_dtype == DType::F16 {
            f16_bits_to_bf16_bits(&mut down_scales_u16);
        }
        crate::quant::marlin_permute_scales(&mut down_scales_u16, down_k, down_n, group_size);
        let expert_w2_scales_offset = e * num_groups_w2 * hidden_size * scale_elem;
        let down_scales_bytes: Vec<u8> = down_scales_u16
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        unsafe {
            driver::memcpy_htod_async(
                w2_scales_ptr.add(expert_w2_scales_offset),
                down_scales_bytes.as_ptr(),
                down_scales_bytes.len(),
                stream,
            )?;
        }
    }

    unsafe { driver::stream_synchronize(stream)? };

    let w1_gpu =
        unsafe { GpuTensor::new(w1_ptr, &[num_experts, w1_packed_per_expert], DType::U32) };
    let w2_gpu =
        unsafe { GpuTensor::new(w2_ptr, &[num_experts, w2_packed_per_expert], DType::U32) };
    let w1_scales_gpu = unsafe {
        GpuTensor::new(
            w1_scales_ptr,
            &[num_experts, num_groups_w1, w1_n],
            out_scales_dtype,
        )
    };
    let w2_scales_gpu = unsafe {
        GpuTensor::new(
            w2_scales_ptr,
            &[num_experts, num_groups_w2, hidden_size],
            out_scales_dtype,
        )
    };

    let workspace_bytes = 256 * 4 * std::mem::size_of::<i32>();
    let workspace_ptr = unsafe { driver::mem_alloc(workspace_bytes)? };
    weights.record_alloc(workspace_ptr, workspace_bytes);
    let workspace_gpu =
        unsafe { GpuTensor::new(workspace_ptr, &[workspace_bytes / 4], DType::I32) };

    Ok((w1_gpu, w2_gpu, w1_scales_gpu, w2_scales_gpu, workspace_gpu))
}

/// Load AWQ expert tensors to CPU: (qweight, scales, qzeros).
/// Each returns (bytes, shape, dtype).
#[allow(clippy::type_complexity)]
pub fn load_expert_awq_cpu(
    weights: &mut GpuWeights,
    prefix: &str,
) -> Result<(
    (Vec<u8>, Vec<usize>, DType),
    (Vec<u8>, Vec<usize>, DType),
    Option<(Vec<u8>, Vec<usize>, DType)>,
)> {
    let qw_name = format!("{prefix}.qweight");
    let scales_name = format!("{prefix}.scales");
    let qzeros_name = format!("{prefix}.qzeros");

    let qw = weights.take_cpu(&qw_name)?;
    let sc = weights.take_cpu(&scales_name)?;
    let zp = if weights.contains(&qzeros_name) {
        Some(weights.take_cpu(&qzeros_name)?)
    } else {
        None
    };

    Ok((qw, sc, zp))
}

/// Concat two row-major 2D byte arrays along dim1.
/// a: [rows, cols_a * elem_size], b: [rows, cols_b * elem_size]
/// → [rows, (cols_a + cols_b) * elem_size]
pub fn concat_bytes_dim1(a: &[u8], b: &[u8], rows: usize, cols_a: usize, cols_b: usize) -> Vec<u8> {
    let elem = 4usize; // u32 for qweight
    let row_a = cols_a * elem;
    let row_b = cols_b * elem;
    let row_out = (cols_a + cols_b) * elem;
    let mut out = vec![0u8; rows * row_out];
    for r in 0..rows {
        out[r * row_out..r * row_out + row_a].copy_from_slice(&a[r * row_a..(r + 1) * row_a]);
        out[r * row_out + row_a..r * row_out + row_out]
            .copy_from_slice(&b[r * row_b..(r + 1) * row_b]);
    }
    out
}

/// Concat two 2D u16 arrays (stored as bytes) along dim1.
pub fn concat_u16_dim1(
    a_bytes: &[u8],
    b_bytes: &[u8],
    rows: usize,
    cols_a: usize,
    cols_b: usize,
) -> Vec<u16> {
    let a: Vec<u16> = a_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let b: Vec<u16> = b_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut out = vec![0u16; rows * (cols_a + cols_b)];
    for r in 0..rows {
        out[r * (cols_a + cols_b)..r * (cols_a + cols_b) + cols_a]
            .copy_from_slice(&a[r * cols_a..(r + 1) * cols_a]);
        out[r * (cols_a + cols_b) + cols_a..(r + 1) * (cols_a + cols_b)]
            .copy_from_slice(&b[r * cols_b..(r + 1) * cols_b]);
    }
    out
}

/// Concat two 2D u32 arrays (stored as bytes) along dim1.
pub fn concat_u32_dim1(
    a_bytes: &[u8],
    b_bytes: &[u8],
    rows: usize,
    cols_a: usize,
    cols_b: usize,
) -> Vec<u32> {
    let a: Vec<u32> = a_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let b: Vec<u32> = b_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = vec![0u32; rows * (cols_a + cols_b)];
    for r in 0..rows {
        out[r * (cols_a + cols_b)..r * (cols_a + cols_b) + cols_a]
            .copy_from_slice(&a[r * cols_a..(r + 1) * cols_a]);
        out[r * (cols_a + cols_b) + cols_a..(r + 1) * (cols_a + cols_b)]
            .copy_from_slice(&b[r * cols_b..(r + 1) * cols_b]);
    }
    out
}

pub fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

pub fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Transpose a 2D byte array from `[rows, cols]` to `[cols, rows]`.
/// `elem_size` is the size of each element in bytes (e.g. 4 for i32/u32, 2 for bf16/f16).
fn transpose_2d_cpu(data: &[u8], rows: usize, cols: usize, elem_size: usize) -> Vec<u8> {
    let row_bytes = cols * elem_size;
    assert_eq!(
        data.len(),
        rows * row_bytes,
        "transpose: data size mismatch"
    );
    let mut out = vec![0u8; cols * rows * elem_size];
    for r in 0..rows {
        for c in 0..cols {
            let src_off = r * row_bytes + c * elem_size;
            let dst_off = c * rows * elem_size + r * elem_size;
            out[dst_off..dst_off + elem_size].copy_from_slice(&data[src_off..src_off + elem_size]);
        }
    }
    out
}

/// Load quantized expert tensors to CPU with compressed-tensors name fallback.
///
/// Tries standard GPTQ names (`{prefix}.qweight`, `{prefix}.scales`) first,
/// then falls back to compressed-tensors names (`{prefix}.weight_packed`,
/// `{prefix}.weight_scale`). For compressed-tensors, transposes from
/// `[N, K/8]` to `[K/8, N]` for qweight and `[N, num_groups]` to
/// `[num_groups, N]` for scales (matching GPTQ layout).
///
/// Returns (qweight, scales) — no zero points for GPTQ/compressed-tensors (symmetric).
#[allow(clippy::type_complexity)]
pub fn load_expert_gptq_cpu(
    weights: &mut GpuWeights,
    prefix: &str,
) -> Result<((Vec<u8>, Vec<usize>, DType), (Vec<u8>, Vec<usize>, DType))> {
    let qw_name = format!("{prefix}.qweight");
    let ct_qw_name = format!("{prefix}.weight_packed");

    if weights.contains(&qw_name) {
        // Standard GPTQ naming: qweight [K/8, N], scales [num_groups, N]
        let qw = weights.take_cpu(&qw_name)?;
        let scales_name = format!("{prefix}.scales");
        let sc = weights.take_cpu(&scales_name)?;
        // Consume qzeros if present (GPTQ symmetric discards them)
        let qzeros_name = format!("{prefix}.qzeros");
        if weights.contains(&qzeros_name) {
            let _ = weights.take_cpu(&qzeros_name);
        }
        // Consume g_idx if present
        let g_idx_name = format!("{prefix}.g_idx");
        if weights.contains(&g_idx_name) {
            let _ = weights.take_cpu(&g_idx_name);
        }
        Ok((qw, sc))
    } else if weights.contains(&ct_qw_name) {
        // compressed-tensors naming: weight_packed [N, K/8], weight_scale [N, num_groups]
        let (qw_bytes, qw_shape, qw_dtype) = weights.take_cpu(&ct_qw_name)?;
        let scales_name = format!("{prefix}.weight_scale");
        let (sc_bytes, sc_shape, sc_dtype) = weights.take_cpu(&scales_name)?;
        // Consume weight_shape if present
        let shape_name = format!("{prefix}.weight_shape");
        if weights.contains(&shape_name) {
            let _ = weights.take_cpu(&shape_name);
        }

        // Transpose qweight from [N, K/8] to [K/8, N]
        let n = qw_shape[0];
        let k_packed = qw_shape[1]; // K/8
        let qw_transposed = transpose_2d_cpu(&qw_bytes, n, k_packed, qw_dtype.size_bytes());

        // Transpose scales from [N, num_groups] to [num_groups, N]
        let sc_n = sc_shape[0];
        let num_groups = sc_shape[1];
        let sc_transposed = transpose_2d_cpu(&sc_bytes, sc_n, num_groups, sc_dtype.size_bytes());

        Ok((
            (qw_transposed, vec![k_packed, n], qw_dtype),
            (sc_transposed, vec![num_groups, sc_n], sc_dtype),
        ))
    } else {
        bail!("weight not found: neither {qw_name} nor {ct_qw_name} exists");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype_mapping() {
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::F16).unwrap(),
            DType::F16
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::BF16).unwrap(),
            DType::BF16
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::F32).unwrap(),
            DType::F32
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::I64).unwrap(),
            DType::I64
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::U32).unwrap(),
            DType::U32
        );
    }

    #[test]
    fn test_unsupported_dtype() {
        assert!(safetensors_dtype(safetensors::Dtype::BOOL).is_err());
    }

    #[test]
    fn test_u8_dtype() {
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::U8).unwrap(),
            DType::U8
        );
    }

    #[test]
    fn test_concat_cpu_dim1() {
        // Two [2, 3] i32 tensors → [2, 6]
        let a: Vec<u8> = [1i32, 2, 3, 4, 5, 6]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b: Vec<u8> = [7i32, 8, 9, 10, 11, 12]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let (out, shape, _) =
            concat_cpu_dim1(&[(&a, &[2, 3], DType::I32), (&b, &[2, 3], DType::I32)]);

        assert_eq!(shape, vec![2, 6]);
        let vals: Vec<i32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Row 0: [1,2,3] ++ [7,8,9] = [1,2,3,7,8,9]
        // Row 1: [4,5,6] ++ [10,11,12] = [4,5,6,10,11,12]
        assert_eq!(vals, vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]);
    }

    #[test]
    fn test_concat_cpu_dim1_three_tensors() {
        // Three [1, 2] tensors → [1, 6]
        let a: Vec<u8> = [1i32, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b: Vec<u8> = [3i32, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        let c: Vec<u8> = [5i32, 6].iter().flat_map(|v| v.to_le_bytes()).collect();

        let (out, shape, _) = concat_cpu_dim1(&[
            (&a, &[1, 2], DType::I32),
            (&b, &[1, 2], DType::I32),
            (&c, &[1, 2], DType::I32),
        ]);

        assert_eq!(shape, vec![1, 6]);
        let vals: Vec<i32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_parse_bnb_quant_state_json() {
        let json = br#"{"quant_type":"nf4","blocksize":64,"dtype":"bfloat16","shape":[2048,1024],"nested_blocksize":256,"nested_dtype":"float32","nested_offset":0.07990148663520813}"#;
        let (offset, blocksize, nested_blocksize) = parse_bnb_quant_state_json(json).unwrap();
        assert!((offset - 0.0799).abs() < 1e-4);
        assert_eq!(blocksize, 64);
        assert_eq!(nested_blocksize, 256);
    }

    #[test]
    fn test_parse_bnb_quant_state_json_with_null_padding() {
        // Quant state blobs often have null bytes after the JSON.
        let mut json = br#"{"quant_type":"nf4","blocksize":64}"#.to_vec();
        json.extend_from_slice(&[0, 0, 0]);
        let (offset, blocksize, _) = parse_bnb_quant_state_json(&json).unwrap();
        assert_eq!(offset, 0.0); // no nested_offset key → default 0.0
        assert_eq!(blocksize, 64);
    }

    #[test]
    fn test_dequantize_double_quant_absmax_with_offset() {
        // Matches Python: nested_quant_map[val] * nested_absmax[i/bs] + offset
        let absmax_u8 = [3u8, 7, 1, 5];
        let nqm: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let na = [2.0f32, 3.0];
        let result = dequantize_double_quant_absmax(&absmax_u8, &nqm, &na, 2, 0.5);
        // [0]: 0.03*2.0+0.5=0.56, [1]: 0.07*2.0+0.5=0.64
        // [2]: 0.01*3.0+0.5=0.53, [3]: 0.05*3.0+0.5=0.65
        assert!((result[0] - 0.56).abs() < 1e-5);
        assert!((result[1] - 0.64).abs() < 1e-5);
        assert!((result[2] - 0.53).abs() < 1e-5);
        assert!((result[3] - 0.65).abs() < 1e-5);
    }

    #[test]
    fn test_dequantize_double_quant_absmax_zero_offset() {
        let absmax_u8 = [100u8, 200];
        let nqm: Vec<f32> = (0..256).map(|i| i as f32 * 0.001).collect();
        let na = [1.0f32];
        let result = dequantize_double_quant_absmax(&absmax_u8, &nqm, &na, 256, 0.0);
        // [0]: 0.100*1.0+0.0=0.1, [1]: 0.200*1.0+0.0=0.2
        assert!((result[0] - 0.1).abs() < 1e-5);
        assert!((result[1] - 0.2).abs() < 1e-5);
    }

    // GPU tests for actual weight loading.
    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;

        fn init_cuda() -> CUstream {
            unsafe {
                driver::init().expect("CUDA init");
                let dev = driver::device_get(0).expect("device");
                let _ctx = driver::ctx_create(dev).expect("context");
                driver::stream_create().expect("stream")
            }
        }

        #[test]
        fn test_load_safetensors_file() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
            let data_b: Vec<f32> = vec![0.1, 0.2, 0.3]; // [3]

            let a_bytes: Vec<u8> = data_a.iter().flat_map(|f| f.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = data_b.iter().flat_map(|f| f.to_le_bytes()).collect();

            let tensors = vec![
                (
                    "weight_a",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![2, 3],
                        &a_bytes,
                    )
                    .unwrap(),
                ),
                (
                    "weight_b",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![3],
                        &b_bytes,
                    )
                    .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            assert_eq!(gw.len(), 2);
            assert!(gw.contains("weight_a"));
            assert!(gw.contains("weight_b"));

            let a = gw.take("weight_a").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(a.ndim(), 2);
            assert_eq!(a.dim(0), 2);
            assert_eq!(a.dim(1), 3);
            assert_eq!(a.dtype(), DType::F32);

            let b = gw.take("weight_b").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(b.ndim(), 1);
            assert_eq!(b.dim(0), 3);

            // Verify data roundtrip: read back from GPU.
            let host = unsafe { driver::mem_alloc_host(a.size_bytes()).unwrap() };
            unsafe {
                driver::memcpy_dtoh_async(host.as_ptr(), a.raw_ptr(), a.size_bytes(), stream)
                    .unwrap();
                driver::stream_synchronize(stream).unwrap();
            }
            let gpu_data = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const f32, 6) };
            for (i, (got, exp)) in gpu_data.iter().zip(data_a.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "weight_a mismatch at {i}: got {got}, expected {exp}"
                );
            }
            unsafe { driver::mem_free_host(host).unwrap() };
            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_load_from_dir() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![1.0; 16];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "test.weight",
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4, 4], &bytes)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_dir(dir.path(), stream).unwrap();

            assert_eq!(gw.len(), 1);
            let t = gw.take("test.weight").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(t.dim(0), 4);
            assert_eq!(t.dim(1), 4);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_take_and_strip_prefix() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![0.0; 8];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "model.layer.weight",
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 4], &bytes)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            // Strip prefix.
            gw.strip_prefix("model.");
            assert!(gw.contains("layer.weight"));
            assert!(!gw.contains("model.layer.weight"));

            // Take.
            let t = gw.take("layer.weight").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(t.dim(0), 2);
            assert_eq!(gw.len(), 0);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_nonexistent_dir() {
            let stream = init_cuda();
            let result = GpuWeights::from_dir("/nonexistent/path", stream);
            assert!(result.is_err());
            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_bf16_weights() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let bf16_data: Vec<u8> = vec![0x00, 0x3F, 0x00, 0x40]; // 0.5 and 2.0 in BF16
            let tensors = vec![(
                "w",
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &bf16_data)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            let w = gw.take("w").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(w.dtype(), DType::BF16);
            assert_eq!(w.numel(), 2);
            assert_eq!(w.size_bytes(), 4);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_names_iterator() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<u8> = vec![0; 16];
            let tensors = vec![
                (
                    "a",
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &data)
                        .unwrap(),
                ),
                (
                    "b",
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &data)
                        .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let gw = GpuWeights::from_single_file(&path, stream).unwrap();

            let mut names: Vec<&str> = gw.names().collect();
            names.sort();
            assert_eq!(names, vec!["a", "b"]);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_take_into() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let data_b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
            let a_bytes: Vec<u8> = data_a.iter().flat_map(|f| f.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = data_b.iter().flat_map(|f| f.to_le_bytes()).collect();

            let tensors = vec![
                (
                    "a",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![4],
                        &a_bytes,
                    )
                    .unwrap(),
                ),
                (
                    "b",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![4],
                        &b_bytes,
                    )
                    .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            // Pre-allocate a fused GPU buffer for both tensors.
            let total_bytes = 32; // 8 floats * 4 bytes
            let gpu_buf = unsafe { driver::mem_alloc(total_bytes).unwrap() };

            // Copy both tensors into the fused buffer.
            let size_a = unsafe { gw.take_into("a", gpu_buf, stream).unwrap() };
            assert_eq!(size_a, 16);
            let size_b = unsafe {
                gw.take_into("b", gpu_buf.wrapping_add(size_a), stream)
                    .unwrap()
            };
            assert_eq!(size_b, 16);

            // Verify roundtrip.
            let host = unsafe { driver::mem_alloc_host(total_bytes).unwrap() };
            unsafe {
                driver::memcpy_dtoh_async(host.as_ptr(), gpu_buf as *mut u8, total_bytes, stream)
                    .unwrap();
                driver::stream_synchronize(stream).unwrap();
            }
            let gpu_data = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const f32, 8) };
            let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            for (i, (got, exp)) in gpu_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "mismatch at {i}: got {got}, expected {exp}"
                );
            }

            unsafe {
                driver::mem_free_host(host).unwrap();
                driver::mem_free(gpu_buf).unwrap();
                driver::stream_destroy(stream).unwrap();
            };
        }
    }

    #[test]
    fn test_merge_lora_f32() {
        use safetensors::tensor::TensorView;

        let dir = tempfile::tempdir().unwrap();

        // Create base model: single weight "model.layers.0.self_attn.q_proj.weight" [8, 4].
        let base_w: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let base_bytes: Vec<u8> = base_w.iter().flat_map(|f| f.to_le_bytes()).collect();

        let base_views = vec![(
            "model.layers.0.self_attn.q_proj.weight",
            TensorView::new(safetensors::Dtype::F32, vec![8, 4], &base_bytes).unwrap(),
        )];
        let base_st = safetensors::tensor::serialize(base_views, None).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), base_st).unwrap();

        // Create LoRA adapter: rank=2, alpha=4 → scaling=2.0.
        let adapter_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            adapter_dir.path().join("adapter_config.json"),
            r#"{"r": 2, "lora_alpha": 4.0, "target_modules": ["q_proj"]}"#,
        )
        .unwrap();

        // A: [2, 4], B: [8, 2]
        let a_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // identity-ish
        let b_data: Vec<f32> = vec![
            0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5,
        ];
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let adapter_views = vec![
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                TensorView::new(safetensors::Dtype::F32, vec![2, 4], &a_bytes).unwrap(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                TensorView::new(safetensors::Dtype::F32, vec![8, 2], &b_bytes).unwrap(),
            ),
        ];
        let adapter_st = safetensors::tensor::serialize(adapter_views, None).unwrap();
        std::fs::write(
            adapter_dir.path().join("adapter_model.safetensors"),
            adapter_st,
        )
        .unwrap();

        // Load weights (no GPU needed — merge is CPU-only).
        // LoRA merge is CPU-only; null stream is fine.
        let mut gw = GpuWeights::from_single_file(
            dir.path().join("model.safetensors"),
            std::ptr::null_mut(),
        )
        .unwrap();

        // Strip "model." prefix to match what CudaWorker does.
        // Actually, merge_lora looks for "{prefix}.weight" keys, so let's check
        // what keys we have.
        let keys: Vec<String> = gw.tensors.keys().cloned().collect();
        assert!(keys.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));

        let merged = gw.merge_lora(adapter_dir.path()).unwrap();
        assert_eq!(merged, 1);

        // Verify merged values: W_merged = W + 2.0 * B @ A
        // B @ A: [8, 2] @ [2, 4] → [8, 4]
        // B has pattern: row i = [0.5, 0.0] or [0.0, 0.5] alternating
        // A = [[1,0,0,0],[0,1,0,0]]
        // B @ A row 0: 0.5*[1,0,0,0] + 0.0*[0,1,0,0] = [0.5,0,0,0]
        // B @ A row 1: 0.0*[1,0,0,0] + 0.5*[0,1,0,0] = [0,0.5,0,0]
        // etc.
        // delta = 2.0 * B@A
        let (data, shape, dtype) = gw
            .take_cpu("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(shape, vec![8, 4]);
        assert_eq!(dtype, DType::F32);
        let merged_w: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Row 0 of delta: [0.5, 0, 0, 0] * 2 = [1, 0, 0, 0]
        // Row 0 of W: [0, 0.1, 0.2, 0.3]
        // Merged: [1.0, 0.1, 0.2, 0.3]
        assert!((merged_w[0] - 1.0).abs() < 1e-5, "got {}", merged_w[0]);
        assert!((merged_w[1] - 0.1).abs() < 1e-5);
        assert!((merged_w[2] - 0.2).abs() < 1e-5);
        assert!((merged_w[3] - 0.3).abs() < 1e-5);

        // Row 1 of delta: [0, 0.5, 0, 0] * 2 = [0, 1, 0, 0]
        // Row 1 of W: [0.4, 0.5, 0.6, 0.7]
        // Merged: [0.4, 1.5, 0.6, 0.7]
        assert!((merged_w[4] - 0.4).abs() < 1e-5);
        assert!((merged_w[5] - 1.5).abs() < 1e-5, "got {}", merged_w[5]);
        assert!((merged_w[6] - 0.6).abs() < 1e-5);
        assert!((merged_w[7] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_merge_lora_rslora_scaling() {
        use safetensors::tensor::TensorView;

        let dir = tempfile::tempdir().unwrap();

        // Base weight: identity-like [2, 2].
        let base_w: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let base_bytes: Vec<u8> = base_w.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base_views = vec![(
            "model.layers.0.self_attn.q_proj.weight",
            TensorView::new(safetensors::Dtype::F32, vec![2, 2], &base_bytes).unwrap(),
        )];
        std::fs::write(
            dir.path().join("model.safetensors"),
            safetensors::tensor::serialize(base_views, None).unwrap(),
        )
        .unwrap();

        // LoRA with rsLoRA: rank=4, alpha=8 → scaling = 8/sqrt(4) = 4.0
        // (normal would be 8/4 = 2.0)
        let adapter_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            adapter_dir.path().join("adapter_config.json"),
            r#"{"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj"], "use_rslora": true}"#,
        )
        .unwrap();

        // A: [4, 2], B: [2, 4] — simple so B@A = [[1,0],[0,1]] (identity)
        // A = [[1,0],[0,1],[0,0],[0,0]], B = [[1,0,0,0],[0,1,0,0]]
        let a_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let adapter_views = vec![
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                TensorView::new(safetensors::Dtype::F32, vec![4, 2], &a_bytes).unwrap(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                TensorView::new(safetensors::Dtype::F32, vec![2, 4], &b_bytes).unwrap(),
            ),
        ];
        std::fs::write(
            adapter_dir.path().join("adapter_model.safetensors"),
            safetensors::tensor::serialize(adapter_views, None).unwrap(),
        )
        .unwrap();

        // LoRA merge is CPU-only; null stream is fine.
        let mut gw = GpuWeights::from_single_file(
            dir.path().join("model.safetensors"),
            std::ptr::null_mut(),
        )
        .unwrap();

        let merged = gw.merge_lora(adapter_dir.path()).unwrap();
        assert_eq!(merged, 1);

        let (data, shape, _) = gw
            .take_cpu("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(shape, vec![2, 2]);
        let w: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // B@A = identity, scaling = 4.0 (rsLoRA), base = identity
        // merged = [[1,0],[0,1]] + 4.0 * [[1,0],[0,1]] = [[5,0],[0,5]]
        assert!((w[0] - 5.0).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 0.0).abs() < 1e-5, "got {}", w[1]);
        assert!((w[2] - 0.0).abs() < 1e-5, "got {}", w[2]);
        assert!((w[3] - 5.0).abs() < 1e-5, "got {}", w[3]);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_read_write_f32_roundtrip() {
        let original = vec![1.0f32, -2.5, 3.14, 0.0];

        // F32 roundtrip.
        let bytes = write_from_f32(&original, DType::F32);
        let mut out = vec![0.0f32; 4];
        read_to_f32(&bytes, DType::F32, &mut out);
        assert_eq!(original, out);

        // BF16 roundtrip (lossy).
        let bytes = write_from_f32(&original, DType::BF16);
        let mut out = vec![0.0f32; 4];
        read_to_f32(&bytes, DType::BF16, &mut out);
        for (a, b) in original.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.1, "BF16 roundtrip: {a} vs {b}");
        }

        // F16 roundtrip (lossy).
        let bytes = write_from_f32(&original, DType::F16);
        let mut out = vec![0.0f32; 4];
        read_to_f32(&bytes, DType::F16, &mut out);
        for (a, b) in original.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.1, "F16 roundtrip: {a} vs {b}");
        }
    }
}
