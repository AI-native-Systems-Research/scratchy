// SPDX-License-Identifier: Apache-2.0
//! Quantized weight loading — AWQ/GPTQ → Marlin repack.
//!
//! Moved here from `scratchy-serving-cuda::weights_quant` + `scratchy-serving-cuda::quant` so
//! that `scratchy-forward-compiler`-emitted `Weights::load` bodies can call
//! these directly. The hand-written path in `scratchy-serving-cuda` still reaches
//! them via the re-exports it always had (`crate::layers`,
//! `crate::layers_quant`) — behavior-preserving move.
//!
//! AWQ and GPTQ live side-by-side because both feed the same Marlin
//! kernel after repack. They stay on distinct loader fn names
//! (`load_awq[_concat]` vs `load_gptq[_concat]`) because the on-disk
//! shape / metadata differ — AWQ qweight is `[K, N/8]` with zero
//! points, GPTQ qweight is `[K/8, N]` with an optional `.g_idx`
//! activation-order tensor and symmetric-mode zero-point-free
//! scales. Codegen picks the right entry point at macro-expansion
//! time from `quantization_config.quant_method`; there's no runtime
//! AWQ-vs-GPTQ branch in the generated loader.
//!
//! Compressed-tensors INT4 (Neural Magic / RedHatAI's INT4 pack
//! format that mirrors GPTQ's uint4b8 packing on a transposed
//! layout) is consumed through the same [`MarlinLinear::load_gptq`]
//! / [`MarlinLinear::load_gptq_concat`] entry points — the caller
//! passes [`GptqLayout::WeightPacked`] to select the `.weight_packed`
//! / `.weight_scale` on-disk tensor names, and the loader
//! transposes both back to GPTQ-native `[K/8, N]` / `[num_groups,
//! N]` before `gptq_repack_into`. Everything past the transpose is
//! bit-identical to AutoGPTQ.

use anyhow::Result;
use cudarc::driver::sys::CUstream;

use crate::alloc::CachingAllocator;
use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
#[cfg(feature = "cuda")]
use crate::weights::CudaWeightsExt;
use crate::weights::GpuWeights;

use crate::layers::{Bnb4bitLinear, Fp8BlockLinear, Fp8Linear, MarlinLinear};

// Quant-load extension traits. The payload TYPES (MarlinLinear / Bnb4bitLinear /
// Fp8Linear / Fp8BlockLinear) are defined in `scratchy-layers`; an inherent
// `impl <Type>` here would be an orphan (E0116), so the `GpuWeights`-bound
// loaders become trait methods. The whole module is `#[cfg(cuda)]`, so these
// (and their impls) are cuda-only.

/// AWQ/GPTQ/CT → Marlin loaders on [`MarlinLinear`].
pub trait MarlinLoadOps {
    fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        storage: MarlinFormat,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self>
    where
        Self: Sized;
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        storage: MarlinFormat,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self>
    where
        Self: Sized;
    fn load_awq(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: usize,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self>
    where
        Self: Sized;
    fn load_awq_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: usize,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self>
    where
        Self: Sized;
    #[allow(clippy::too_many_arguments)]
    fn load_gptq(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: usize,
        desc_act: bool,
        layout: GptqLayout,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self>
    where
        Self: Sized;
    #[allow(clippy::too_many_arguments)]
    fn load_gptq_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: usize,
        desc_act: bool,
        layout: GptqLayout,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self>
    where
        Self: Sized;
}

/// BitsAndBytes 4-bit loaders on [`Bnb4bitLinear`].
pub trait Bnb4bitLoadOps {
    #[allow(clippy::too_many_arguments)]
    fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        out_features: usize,
        in_features: usize,
        blocksize: usize,
    ) -> Result<Self>
    where
        Self: Sized;
    #[allow(clippy::too_many_arguments)]
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        out_features_per_shard: &[usize],
        in_features: usize,
        blocksize: usize,
    ) -> Result<Self>
    where
        Self: Sized;
}

/// FP8 (per-tensor / per-channel) loaders on [`Fp8Linear`].
pub trait Fp8LoadOps {
    fn load(weights: &mut GpuWeights, prefix: &str, output_dtype: DType) -> Result<Self>
    where
        Self: Sized;
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        output_dtype: DType,
    ) -> Result<Self>
    where
        Self: Sized;
}

/// FP8 block-scaled loaders on [`Fp8BlockLinear`].
pub trait Fp8BlockLoadOps {
    fn load(weights: &mut GpuWeights, prefix: &str, output_dtype: DType) -> Result<Self>
    where
        Self: Sized;
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        output_dtype: DType,
    ) -> Result<Self>
    where
        Self: Sized;
}

/// GPTQ on-disk layout passed by scratchy-forward-compiler-emitted
/// `Weights::load` bodies to [`MarlinLinear::load_gptq`] /
/// [`MarlinLinear::load_gptq_concat`].
///
/// Both variants end up at the same `gptq_repack_into` kernel with
/// the same `[K/8, N]` uint4b8 packed weight on GPU; the only thing
/// the layout selects is which tensor names the loader reads from
/// `GpuWeights` (and whether it transposes the bytes on the CPU
/// before upload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptqLayout {
    /// AutoGPTQ native: `.qweight [K/8, N]`, `.scales [num_groups,
    /// N]`, optional `.qzeros` + `.g_idx`.
    Qweight,
    /// compressed-tensors INT4: `.weight_packed [N, K/8]`,
    /// `.weight_scale [N, num_groups]`. No `.qzeros`, no `.g_idx`
    /// (CT INT4 is always symmetric + never uses activation
    /// ordering). The optional `.weight_shape` tensor is consumed
    /// and discarded.
    WeightPacked,
}

/// Unified runtime quant-format spec. Scratchy-forward-emitted
/// `Weights::load_with(gw, stream, storage: MarlinFormat)` threads
/// the variant's knobs through a single [`MarlinLinear::load`] /
/// [`MarlinLinear::load_concat`] call per accessor. Dispatch on
/// storage happens inside the kernel crate — every AWQ/GPTQ/CT
/// variant in the same equivalence class shares one emitted load
/// body, with `MarlinFormat` as the only thing that differs per-
/// variant (a const the variant's one-line `load()` wrapper passes
/// to the canonical `load_with`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarlinFormat {
    /// AutoAWQ 4-bit — `.qweight [K, N/8]` + `.qzeros` + `.scales`.
    Awq { group_size: u32 },
    /// AutoGPTQ / compressed-tensors 4-bit uint4b8. `layout` selects
    /// the on-disk tensor names + transpose.
    Gptq {
        group_size: u32,
        desc_act: bool,
        layout: GptqLayout,
    },
}

// ---------------------------------------------------------------------------
// Marlin workspace
// ---------------------------------------------------------------------------

/// Allocate the shared Marlin workspace buffer `[max(2*num_sm, 1M)]` i32.
///
/// One allocation per device serves every MarlinLinear on that device —
/// Marlin's kernel uses it as a barrier-lock region. Must be zero'd
/// on creation; subsequent forwards re-zero it implicitly on use.
pub fn alloc_marlin_workspace(num_sm: i32, stream: CUstream) -> Result<GpuTensor> {
    // Match Python: max(2 * num_sm, 1024 * 1024) elements.
    let num_elements = std::cmp::max(2 * num_sm as usize, 1024 * 1024);
    let nbytes = num_elements * std::mem::size_of::<i32>();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    unsafe { driver::memset_d8(ptr, 0, nbytes, stream)? };
    Ok(unsafe { GpuTensor::new(ptr, &[num_elements], DType::I32) })
}

// ---------------------------------------------------------------------------
// Marlin scale / zero-point permutations (CPU, at load time)
// ---------------------------------------------------------------------------

/// Marlin scale permutation indices for group quantization.
/// Python: `[i + 8*j for i in range(8) for j in range(8)]`.
pub fn scale_perm() -> [usize; 64] {
    let mut p = [0usize; 64];
    let mut idx = 0;
    for i in 0..8 {
        for j in 0..8 {
            p[idx] = i + 8 * j;
            idx += 1;
        }
    }
    p
}

/// Marlin scale permutation indices for per-channel / act_order.
/// Python: `[2*i + j for i in range(4) for j in [0,1,8,9,16,17,24,25]]`.
pub fn scale_perm_single() -> [usize; 32] {
    let mut p = [0usize; 32];
    let offsets = [0, 1, 8, 9, 16, 17, 24, 25];
    let mut idx = 0;
    for i in 0..4 {
        for &j in &offsets {
            p[idx] = 2 * i + j;
            idx += 1;
        }
    }
    p
}

/// Apply the Marlin scale permutation to a flat scale buffer in place.
///
/// `scales`: logical `[num_groups, size_n]` of f16/bf16 bits reinterpreted as u16.
pub fn marlin_permute_scales(scales: &mut [u16], size_k: usize, size_n: usize, group_size: usize) {
    let num_groups = if group_size > 0 && group_size < size_k {
        size_k / group_size
    } else {
        1
    };
    let use_single = group_size >= size_k || group_size == 0;

    if use_single {
        let perm = scale_perm_single();
        let chunk = perm.len(); // 32
        let num_chunks = (num_groups * size_n) / chunk;
        let mut tmp = vec![0u16; chunk];
        for c in 0..num_chunks {
            let base = c * chunk;
            for (i, &p) in perm.iter().enumerate() {
                tmp[i] = scales[base + p];
            }
            scales[base..base + chunk].copy_from_slice(&tmp);
        }
    } else {
        let perm = scale_perm();
        let chunk = perm.len(); // 64
        let num_chunks = (num_groups * size_n) / chunk;
        let mut tmp = vec![0u16; chunk];
        for c in 0..num_chunks {
            let base = c * chunk;
            for (i, &p) in perm.iter().enumerate() {
                tmp[i] = scales[base + p];
            }
            scales[base..base + chunk].copy_from_slice(&tmp);
        }
    }
}

/// Unpack packed INT4 columns: `[rows, cols/pack_factor]` u32 → `[rows, cols]` u8.
///
/// Matches Python vLLM's `quant_utils.unpack_cols` convention: consecutive
/// output columns `[pack_factor*p .. pack_factor*p + pack_factor]` are
/// packed into input column `p`'s low-to-high nibbles. (Our earlier
/// implementation used a strided layout — `b * (cols/pack_factor) + p` —
/// which silently corrupted the AWQ → Marlin zero-point pipeline since
/// AutoAWQ stores qzeros in Python's convention.)
pub fn unpack_cols_4bit(packed: &[u32], rows: usize, cols: usize) -> Vec<u8> {
    let pack_factor = 8; // 32 / 4
    assert_eq!(packed.len(), rows * (cols / pack_factor));
    assert_eq!(cols % pack_factor, 0);
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        for p in 0..cols / pack_factor {
            let val = packed[r * (cols / pack_factor) + p];
            for i in 0..pack_factor {
                out[r * cols + pack_factor * p + i] = ((val >> (4 * i)) & 0xF) as u8;
            }
        }
    }
    out
}

/// Pack INT4 columns: `[rows, cols]` u8 → `[rows, cols/pack_factor]` u32.
///
/// Inverse of [`unpack_cols_4bit`]; matches Python vLLM's `pack_cols`.
pub fn pack_cols_4bit(unpacked: &[u8], rows: usize, cols: usize) -> Vec<u32> {
    let pack_factor = 8;
    assert_eq!(unpacked.len(), rows * cols);
    assert_eq!(cols % pack_factor, 0);
    let mut out = vec![0u32; rows * (cols / pack_factor)];
    for r in 0..rows {
        for p in 0..cols / pack_factor {
            let mut val = 0u32;
            for i in 0..pack_factor {
                val |= (unpacked[r * cols + pack_factor * p + i] as u32) << (4 * i);
            }
            out[r * (cols / pack_factor) + p] = val;
        }
    }
    out
}

/// Convert AWQ zero points to Marlin format (CPU).
///
/// Unpacks AWQ qzeros, undoes the AWQ interleave, applies Marlin's
/// scale permutation, re-interleaves for Marlin's dequantizer, and
/// repacks to u32. Matches Python vLLM's `awq_to_marlin_zero_points`.
pub fn awq_to_marlin_zero_points(
    packed_qzeros: &[u32],
    num_groups: usize,
    size_n: usize,
) -> Vec<u32> {
    let mut zp = unpack_cols_4bit(packed_qzeros, num_groups, size_n);

    // Undo AWQ interleaving [0,2,4,6,1,3,5,7].
    let undo_interleave: [usize; 8] = {
        let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
        let mut inv = [0usize; 8];
        for (i, &v) in interleave.iter().enumerate() {
            inv[v] = i;
        }
        inv
    };

    let total = num_groups * size_n;
    let mut tmp = [0u8; 8];
    for chunk_start in (0..total).step_by(8) {
        for (i, &p) in undo_interleave.iter().enumerate() {
            tmp[i] = zp[chunk_start + p];
        }
        zp[chunk_start..chunk_start + 8].copy_from_slice(&tmp);
    }

    // Apply Marlin scale_perm to the zero points.
    let perm = scale_perm();
    let chunk_size = 64;
    let num_chunks = total / chunk_size;
    let mut tmp64 = vec![0u8; chunk_size];
    for c in 0..num_chunks {
        let base = c * chunk_size;
        for (i, &p) in perm.iter().enumerate() {
            tmp64[i] = zp[base + p];
        }
        zp[base..base + chunk_size].copy_from_slice(&tmp64);
    }

    // Re-interleave for Marlin dequantizer [0,2,4,6,1,3,5,7].
    let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let mut tmp8 = [0u8; 8];
    for chunk_start in (0..total).step_by(8) {
        for (i, &p) in interleave.iter().enumerate() {
            tmp8[i] = zp[chunk_start + p];
        }
        zp[chunk_start..chunk_start + 8].copy_from_slice(&tmp8);
    }

    pack_cols_4bit(&zp, num_groups, size_n)
}

// ---------------------------------------------------------------------------
// CPU concat / transpose helpers used by fused loaders
// ---------------------------------------------------------------------------

/// Transpose a 2D CPU tensor `[rows, cols]` → `[cols, rows]` with
/// element size `elem_size` bytes. Used by compressed-tensors INT4
/// to flip `.weight_packed [N, K/8]` → `[K/8, N]` (GPTQ-native) and
/// `.weight_scale [N, num_groups]` → `[num_groups, N]` before the
/// shared repack / permute pipeline takes over.
pub fn transpose_2d_cpu(data: &[u8], rows: usize, cols: usize, elem_size: usize) -> Vec<u8> {
    assert_eq!(
        data.len(),
        rows * cols * elem_size,
        "transpose_2d_cpu: data size mismatch (rows={rows} cols={cols} elem={elem_size})",
    );
    let mut out = vec![0u8; rows * cols * elem_size];
    let row_bytes = cols * elem_size;
    for r in 0..rows {
        for c in 0..cols {
            let src = r * row_bytes + c * elem_size;
            let dst = c * rows * elem_size + r * elem_size;
            out[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
        }
    }
    out
}

/// Concatenate multiple 2D CPU tensors along dim 1 (output/N).
/// All tensors must share dim 0 and dtype; row-interleaves the bytes.
pub fn concat_cpu_dim1(tensors: &[(&[u8], &[usize], DType)]) -> (Vec<u8>, Vec<usize>, DType) {
    assert!(!tensors.is_empty());
    let dtype = tensors[0].2;
    let dim0 = tensors[0].1[0];
    let elem_size = dtype.size_bytes();

    let total_dim1: usize = tensors.iter().map(|(_, shape, _)| shape[1]).sum();

    let total_bytes = dim0 * total_dim1 * elem_size;
    let mut out = vec![0u8; total_bytes];

    for row in 0..dim0 {
        let mut col_offset = 0usize;
        for (data, shape, _) in tensors {
            let n = shape[1];
            let src_row_bytes = n * elem_size;
            let src_start = row * src_row_bytes;
            let dst_start = (row * total_dim1 + col_offset) * elem_size;
            out[dst_start..dst_start + src_row_bytes]
                .copy_from_slice(&data[src_start..src_start + src_row_bytes]);
            col_offset += n;
        }
    }

    (out, vec![dim0, total_dim1], dtype)
}

/// Concat per-source bias CPU buffers and upload to GPU.
/// Returns `None` when no sources have a bias tensor.
pub fn fuse_bias_parts(
    bias_parts: &[Vec<u8>],
    bias_dtype: Option<DType>,
    weights: &mut GpuWeights,
    stream: CUstream,
) -> Result<Option<GpuTensor>> {
    if bias_parts.is_empty() {
        return Ok(None);
    }
    let dtype = bias_dtype.unwrap();
    let total_bytes: usize = bias_parts.iter().map(|b| b.len()).sum();
    let mut fused = Vec::with_capacity(total_bytes);
    for part in bias_parts {
        fused.extend_from_slice(part);
    }
    let num_elements = total_bytes / dtype.size_bytes();
    let ptr = unsafe { driver::mem_alloc(total_bytes)? };
    weights.record_alloc(ptr, total_bytes);
    unsafe { driver::memcpy_htod_async(ptr, fused.as_ptr(), total_bytes, stream)? };
    Ok(Some(unsafe { GpuTensor::new(ptr, &[num_elements], dtype) }))
}

// ---------------------------------------------------------------------------
// AWQ → Marlin loaders (constructors on MarlinLinear)
// ---------------------------------------------------------------------------

impl MarlinLoadOps for MarlinLinear {
    /// Unified runtime-dispatched loader. Dispatches to
    /// [`Self::load_awq`] / [`Self::load_gptq`] based on the
    /// variant's `MarlinFormat`. Scratchy-forward-emitted
    /// `Weights::load_with(gw, stream, storage)` calls this with a
    /// single static shape per accessor regardless of quant
    /// method, so the emitted load-body compiles to a single
    /// canonical function per (arch, size, Impl-family)
    /// equivalence class — one rustc-optimized copy shared by
    /// every AWQ/GPTQ/CT variant in that class.
    fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        storage: MarlinFormat,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        match storage {
            MarlinFormat::Awq { group_size } => {
                Self::load_awq(weights, prefix, group_size as usize, workspace, device_id)
            }
            MarlinFormat::Gptq {
                group_size,
                desc_act,
                layout,
            } => Self::load_gptq(
                weights,
                prefix,
                group_size as usize,
                desc_act,
                layout,
                workspace,
                device_id,
            ),
        }
    }

    /// Unified fused-accessor loader — dispatches to
    /// `load_awq_concat` / `load_gptq_concat` on `MarlinFormat`.
    /// Same shared-body rationale as [`Self::load`].
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        storage: MarlinFormat,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        match storage {
            MarlinFormat::Awq { group_size } => {
                Self::load_awq_concat(weights, prefixes, group_size as usize, workspace, device_id)
            }
            MarlinFormat::Gptq {
                group_size,
                desc_act,
                layout,
            } => Self::load_gptq_concat(
                weights,
                prefixes,
                group_size as usize,
                desc_act,
                layout,
                workspace,
                device_id,
            ),
        }
    }

    /// Load one AWQ-packed INT4 weight and repack to Marlin's tiled layout.
    ///
    /// Reads `{prefix}.qweight` / `.scales` / `.qzeros` (optional `.bias`)
    /// from `weights`. AWQ qweight shape is `[K, N/8]` i32. Caller
    /// supplies the shared marlin workspace and the CUDA device id.
    ///
    /// # AWQ assumptions
    /// - `bits == 4` (AWQ scratchy path is 4-bit only).
    /// - `zero_point == true` (AWQ ships zero points; `has_zp` is
    ///   hard-set on the returned `MarlinLinear`).
    /// - On-disk packing is AutoAWQ's `gemm` (equivalently `gemv`).
    ///   Pre-packed `version = "marlin"` isn't handled here yet; the
    ///   caller is responsible for gating that.
    fn load_awq(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: usize,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        let stream = weights.stream();

        let qw_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        let (qw_shape, _qw_dtype) = weights
            .tensor_info(&qw_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {qw_name}"))?;
        let size_k = qw_shape[0];
        let size_n = qw_shape[1] * 8;
        let num_groups = size_k.checked_div(group_size).unwrap_or(1);

        // Upload qweight → GPU, repack AWQ → Marlin, free original.
        let qweight_gpu = weights.take(&qw_name)?;
        let num_u32 = size_k * size_n / 8;
        let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
        let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
        weights.record_alloc(repack_ptr, repack_nbytes);
        unsafe {
            crate::kernels::awq_repack_into(
                qweight_gpu,
                repack_ptr,
                size_k,
                size_n,
                device_id,
                stream,
            );
            driver::stream_synchronize(stream)?;
            weights.unrecord_alloc(qweight_gpu.raw_ptr());
            driver::mem_free(qweight_gpu.raw_ptr())?;
        }
        let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

        // Scales: CPU-load → permute → upload.
        let (scales_bytes, _scales_shape, scales_dtype) = weights.take_cpu(&scales_name)?;
        let mut scales_u16: Vec<u16> = scales_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
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

        // Zero points: CPU-load → AWQ-to-Marlin convert → upload.
        let (qzeros_bytes, _qzeros_shape, _qzeros_dtype) = weights.take_cpu(&qzeros_name)?;
        let qzeros_u32: Vec<u32> = qzeros_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let marlin_zp = awq_to_marlin_zero_points(&qzeros_u32, num_groups, size_n);

        let zp_bytes: Vec<u8> = marlin_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let zp_nbytes = zp_bytes.len();
        let zp_ptr = unsafe { driver::mem_alloc(zp_nbytes)? };
        weights.record_alloc(zp_ptr, zp_nbytes);
        unsafe { driver::memcpy_htod_async(zp_ptr, zp_bytes.as_ptr(), zp_nbytes, stream)? };
        let zeros_gpu = unsafe { GpuTensor::new(zp_ptr, &[num_groups, size_n / 8], DType::U32) };

        let bias_name = format!("{prefix}.bias");
        let bias_gpu = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };

        unsafe { driver::stream_synchronize(stream)? };

        Ok(Self {
            qweight: qweight_marlin,
            scales: scales_gpu,
            zeros: Some(zeros_gpu),
            g_idx: None,
            g_idx_sort_indices: None,
            workspace,
            size_k,
            size_n,
            group_size,
            num_groups,
            has_zp: true,
            has_act_order: false,
            b_type_id: 1, // AWQ = uint4
            device_id,
            bias: bias_gpu,
        })
    }

    /// Load several AWQ weights and fuse into one Marlin GEMM by
    /// concatenating qweight / scales / qzeros along dim N before
    /// the single repack. All source weights must share `K` and
    /// `group_size`; each contributes its own `N_i`, summing into
    /// `N_total = sum(N_i)`.
    ///
    /// This matches Python vLLM's `MergedColumnParallelLinear` — e.g.
    /// `[q, k, v]` collapse to one wider Marlin layer.
    ///
    /// # Bias handling
    /// If any source weight has a bias, the fused layer carries a
    /// concatenated `[N_total]` bias. Mixing bias/no-bias across
    /// sources is allowed: the caller's upstream HF repos should be
    /// consistent, but partial biases fall through (HF never does
    /// this for QKV / gate-up, but the upstream AWQ loader tolerated
    /// it, so keep parity).
    fn load_awq_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: usize,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        assert!(
            !prefixes.is_empty(),
            "load_awq_concat called with empty prefix list",
        );
        let stream = weights.stream();

        let mut qw_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut sc_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut qz_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut bias_parts: Vec<Vec<u8>> = Vec::new();
        let mut bias_dtype: Option<DType> = None;
        let mut part_n_sizes: Vec<usize> = Vec::new();

        for prefix in prefixes {
            let qw_name = format!("{prefix}.qweight");
            let scales_name = format!("{prefix}.scales");
            let qzeros_name = format!("{prefix}.qzeros");

            let (qw_data, qw_shape, qw_dt) = weights.take_cpu(&qw_name)?;
            let part_n = qw_shape[1] * 8;
            part_n_sizes.push(part_n);
            qw_parts.push((qw_data, qw_shape, qw_dt));
            sc_parts.push(weights.take_cpu(&scales_name)?);
            qz_parts.push(weights.take_cpu(&qzeros_name)?);

            let bias_name = format!("{prefix}.bias");
            if weights.contains(&bias_name) {
                let (b_data, _b_shape, b_dtype) = weights.take_cpu(&bias_name)?;
                bias_parts.push(b_data);
                bias_dtype = Some(b_dtype);
            }
        }

        // Concat qweights along dim 1.
        let qw_refs: Vec<_> = qw_parts
            .iter()
            .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
            .collect();
        let (qw_fused, qw_shape, _) = concat_cpu_dim1(&qw_refs);
        let size_k = qw_shape[0];
        let size_n = qw_shape[1] * 8;
        let num_groups = size_k.checked_div(group_size).unwrap_or(1);

        // Upload fused qweight and repack once.
        let qw_nbytes = qw_fused.len();
        let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
        unsafe { driver::memcpy_htod_async(qw_gpu_ptr, qw_fused.as_ptr(), qw_nbytes, stream)? };
        let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &qw_shape, DType::I32) };

        let num_u32 = size_k * size_n / 8;
        let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
        let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
        weights.record_alloc(repack_ptr, repack_nbytes);
        unsafe {
            crate::kernels::awq_repack_into(qw_gpu, repack_ptr, size_k, size_n, device_id, stream);
            driver::stream_synchronize(stream)?;
            driver::mem_free(qw_gpu_ptr)?;
        }
        let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

        // Concat + permute scales.
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
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
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

        // Concat zero points: AWQ qzeros convert per-part, then row-interleave.
        let mut all_marlin_zp: Vec<Vec<u32>> = Vec::new();
        for (i, (qz_data, _qz_shape, _qz_dt)) in qz_parts.iter().enumerate() {
            let qzeros_u32: Vec<u32> = qz_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let part_zp = awq_to_marlin_zero_points(&qzeros_u32, num_groups, part_n_sizes[i]);
            all_marlin_zp.push(part_zp);
        }
        let total_n_div8 = size_n / 8;
        let mut fused_zp = vec![0u32; num_groups * total_n_div8];
        let mut col_offsets: Vec<usize> = Vec::new();
        let mut cumulative = 0usize;
        for &pn in &part_n_sizes {
            col_offsets.push(cumulative);
            cumulative += pn / 8;
        }
        for (i, &pn) in part_n_sizes.iter().enumerate() {
            let part_cols = pn / 8;
            for g in 0..num_groups {
                let dst_start = g * total_n_div8 + col_offsets[i];
                let src_start = g * part_cols;
                fused_zp[dst_start..dst_start + part_cols]
                    .copy_from_slice(&all_marlin_zp[i][src_start..src_start + part_cols]);
            }
        }

        let zp_bytes: Vec<u8> = fused_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let zp_nbytes = zp_bytes.len();
        let zp_ptr = unsafe { driver::mem_alloc(zp_nbytes)? };
        weights.record_alloc(zp_ptr, zp_nbytes);
        unsafe { driver::memcpy_htod_async(zp_ptr, zp_bytes.as_ptr(), zp_nbytes, stream)? };
        let zeros_gpu = unsafe { GpuTensor::new(zp_ptr, &[num_groups, total_n_div8], DType::U32) };

        unsafe { driver::stream_synchronize(stream)? };

        Ok(Self {
            qweight: qweight_marlin,
            scales: scales_gpu,
            zeros: Some(zeros_gpu),
            g_idx: None,
            g_idx_sort_indices: None,
            workspace,
            size_k,
            size_n,
            group_size,
            num_groups,
            has_zp: true,
            has_act_order: false,
            b_type_id: 1, // AWQ = uint4
            device_id,
            bias: fuse_bias_parts(&bias_parts, bias_dtype, weights, stream)?,
        })
    }

    /// Load one GPTQ-packed INT4 weight and repack to Marlin's tiled
    /// layout.
    ///
    /// Reads `{prefix}.qweight` / `.scales` / (optional `.qzeros`) /
    /// (optional `.g_idx`) / (optional `.bias`) from `weights`. GPTQ
    /// qweight shape is `[K/8, N]` i32 (note: input-dim packed,
    /// opposite of AWQ's `[K, N/8]`). Caller supplies the shared
    /// marlin workspace and the CUDA device id.
    ///
    /// # GPTQ assumptions
    /// - `bits == 4`.
    /// - Symmetric quantization: `.qzeros` is consumed and discarded
    ///   because Marlin's `uint4b8` scalar type bakes in the
    ///   zero-point (bias=8). Asymmetric GPTQ isn't handled here
    ///   yet — scratchy's parser rejects `sym == false` downstream,
    ///   but we don't re-check that here.
    /// - `desc_act`: when `true` and the model ships `.g_idx`, the
    ///   loader argsort-permutes the group-id vector and hands the
    ///   resulting sort_indices to `gptq_repack_into` so same-group
    ///   columns are contiguous in the repacked weight.
    #[allow(clippy::too_many_arguments)]
    fn load_gptq(
        weights: &mut GpuWeights,
        prefix: &str,
        group_size: usize,
        desc_act: bool,
        layout: GptqLayout,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        let stream = weights.stream();

        // Pick the on-disk tensor names per layout. The GPTQ path
        // reads `.qweight` directly; compressed-tensors repacks into
        // `.weight_packed` with transposed axes.
        let qw_name = match layout {
            GptqLayout::Qweight => format!("{prefix}.qweight"),
            GptqLayout::WeightPacked => format!("{prefix}.weight_packed"),
        };
        let (qw_shape, _qw_dtype) = weights
            .tensor_info(&qw_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {qw_name}"))?;
        let (size_k, size_n) = match layout {
            // AutoGPTQ: [K/8, N]
            GptqLayout::Qweight => (qw_shape[0] * 8, qw_shape[1]),
            // compressed-tensors: [N, K/8]
            GptqLayout::WeightPacked => (qw_shape[1] * 8, qw_shape[0]),
        };
        let num_groups = size_k.checked_div(group_size).unwrap_or(1);

        // GPTQ symmetric stores zero points on disk as a formality.
        // Python vLLM never passes them to Marlin — consume and drop.
        // compressed-tensors INT4 symmetric skips `.qzeros` entirely.
        let qzeros_name = format!("{prefix}.qzeros");
        if weights.contains(&qzeros_name) {
            let _ = weights.take(&qzeros_name);
        }

        // Handle g_idx for desc_act (activation ordering) BEFORE
        // repack — repack needs `perm` (sort_indices) on GPU.
        // AutoGPTQ ships `.g_idx`; compressed-tensors ships
        // `.weight_g_idx` when the quantizer used `actorder: "group"`.
        // Tensor presence drives the code path — the per-preset
        // compile-time `desc_act` flag only governs whether we emit
        // the act-order branch, not whether the actual checkpoint
        // carries a permutation. Runtime tensor presence is the
        // source of truth (matches Python vLLM's CT path, which
        // builds `has_g_idx` from the live weights dict).
        let g_idx_name = match layout {
            GptqLayout::Qweight => format!("{prefix}.g_idx"),
            GptqLayout::WeightPacked => format!("{prefix}.weight_g_idx"),
        };
        // For AutoGPTQ (`Qweight`), the compile-time `desc_act` flag
        // is authoritative — distinct `gptq-sym` and `gptq-sym-desc_act`
        // variants are synthesized, and the fingerprint's g_idx
        // disambiguation picks exactly the variant whose desc_act
        // matches the checkpoint. Runtime-detecting here would flip
        // on `has_act_order` for desc_act=false variants that happen
        // to ship an inert `.g_idx` tensor, breaking them.
        //
        // For compressed-tensors (`WeightPacked`), there's no per-
        // preset `desc_act` knob (the preset file doesn't carry an
        // `actorder` field — it varies per checkpoint), so tensor
        // presence IS the source of truth.
        let take_g_idx = match layout {
            GptqLayout::Qweight => desc_act && weights.contains(&g_idx_name),
            GptqLayout::WeightPacked => weights.contains(&g_idx_name),
        };
        let (g_idx_gpu, sort_indices_gpu, has_act_order) = if take_g_idx {
            let (g_idx_bytes, _g_idx_shape, _g_idx_dtype) = weights.take_cpu(&g_idx_name)?;
            let g_idx_i32: Vec<i32> = g_idx_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Stable ascending argsort by group id.
            let mut sort_indices: Vec<i32> = (0..g_idx_i32.len() as i32).collect();
            sort_indices.sort_by_key(|&i| g_idx_i32[i as usize]);

            let sorted_g_idx: Vec<i32> = sort_indices
                .iter()
                .map(|&i| g_idx_i32[i as usize])
                .collect();

            let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let g_idx_nbytes = g_idx_bytes.len();
            let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
            weights.record_alloc(g_idx_ptr, g_idx_nbytes);
            unsafe {
                driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
            }
            let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

            let si_bytes: Vec<u8> = sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let si_nbytes = si_bytes.len();
            let si_ptr = unsafe { driver::mem_alloc(si_nbytes)? };
            weights.record_alloc(si_ptr, si_nbytes);
            unsafe { driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_nbytes, stream)? };
            let sort_indices_gpu = unsafe { GpuTensor::new(si_ptr, &[size_k], DType::I32) };

            (Some(g_idx_gpu), Some(sort_indices_gpu), true)
        } else {
            // Consume g_idx if present but unused — upstream may ship
            // it even with desc_act=false.
            if weights.contains(&g_idx_name) {
                let _ = weights.take(&g_idx_name);
            }
            (None, None, false)
        };

        // Upload qweight → GPU, repack GPTQ → Marlin, free original.
        // For compressed-tensors we CPU-transpose from [N, K/8] to
        // [K/8, N] on the way up so the repack kernel sees GPTQ-
        // native axes. `.weight_shape` (if present) is consumed and
        // discarded.
        let qweight_gpu = match layout {
            GptqLayout::Qweight => weights.take(&qw_name)?,
            GptqLayout::WeightPacked => {
                let (qw_bytes, shape_ct, qw_dtype) = weights.take_cpu(&qw_name)?;
                let n = shape_ct[0];
                let k_packed = shape_ct[1];
                let transposed = transpose_2d_cpu(&qw_bytes, n, k_packed, qw_dtype.size_bytes());
                let shape_name = format!("{prefix}.weight_shape");
                if weights.contains(&shape_name) {
                    let _ = weights.take_cpu(&shape_name);
                }
                let nbytes = transposed.len();
                let ptr = unsafe { driver::mem_alloc(nbytes)? };
                unsafe {
                    driver::memcpy_htod_async(ptr, transposed.as_ptr(), nbytes, stream)?;
                }
                unsafe { GpuTensor::new(ptr, &[k_packed, n], qw_dtype) }
            }
        };
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
            driver::stream_synchronize(stream)?;
            // `weights.take` records the alloc; CT path's raw
            // `mem_alloc` above does not. Only unrecord when the
            // loader took ownership through `GpuWeights`.
            if matches!(layout, GptqLayout::Qweight) {
                weights.unrecord_alloc(qweight_gpu.raw_ptr());
            }
            driver::mem_free(qweight_gpu.raw_ptr())?;
        }
        let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

        // Scales: CPU-load → Marlin permute → upload. CT stores
        // `.weight_scale` at `[N, num_groups]` — transpose to match
        // AutoGPTQ's `[num_groups, N]` before the permute.
        let (scales_bytes_raw, scales_dtype) = match layout {
            GptqLayout::Qweight => {
                let scales_name = format!("{prefix}.scales");
                let (bytes, _shape, dtype) = weights.take_cpu(&scales_name)?;
                (bytes, dtype)
            }
            GptqLayout::WeightPacked => {
                let ct_scales = format!("{prefix}.weight_scale");
                let (bytes, shape, dtype) = weights.take_cpu(&ct_scales)?;
                let sc_n = shape[0];
                let sc_groups = shape[1];
                let transposed = transpose_2d_cpu(&bytes, sc_n, sc_groups, dtype.size_bytes());
                (transposed, dtype)
            }
        };
        let mut scales_u16: Vec<u16> = scales_bytes_raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
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

        let bias_name = format!("{prefix}.bias");
        let bias_gpu = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };

        unsafe { driver::stream_synchronize(stream)? };

        Ok(Self {
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

    /// Load several GPTQ weights and fuse into one Marlin GEMM by
    /// concatenating qweight / scales along dim 1 (output/N) before
    /// the single repack. All source weights must share `K` and
    /// `group_size`; each contributes its own `N_i`.
    ///
    /// Matches Python vLLM's `MergedColumnParallelLinear` for GPTQ.
    ///
    /// # g_idx handling (desc_act)
    /// `.g_idx` is indexed by K (input dim), and all fused sub-layers
    /// share K, so a single `.g_idx` covers the fused layer. Take it
    /// from the first prefix; consume and discard from the rest.
    #[allow(clippy::too_many_arguments)]
    fn load_gptq_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        group_size: usize,
        desc_act: bool,
        layout: GptqLayout,
        workspace: GpuTensor,
        device_id: i32,
    ) -> Result<Self> {
        assert!(
            !prefixes.is_empty(),
            "load_gptq_concat called with empty prefix list",
        );
        let stream = weights.stream();

        let mut qw_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut sc_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
        let mut g_idx_i32: Option<Vec<i32>> = None;
        let mut bias_parts: Vec<Vec<u8>> = Vec::new();
        let mut bias_dtype: Option<DType> = None;

        for (i, prefix) in prefixes.iter().enumerate() {
            match layout {
                GptqLayout::Qweight => {
                    let qw_name = format!("{prefix}.qweight");
                    qw_parts.push(weights.take_cpu(&qw_name)?);

                    let scales_name = format!("{prefix}.scales");
                    sc_parts.push(weights.take_cpu(&scales_name)?);
                }
                GptqLayout::WeightPacked => {
                    // compressed-tensors: `.weight_packed [N, K/8]` →
                    // transpose to `[K/8, N]` to match GPTQ-native
                    // before concat-along-dim-1. Same transpose on
                    // `.weight_scale [N, num_groups]` → `[num_groups,
                    // N]`.
                    let qw_name = format!("{prefix}.weight_packed");
                    let (qw_bytes, qw_shape, qw_dtype) = weights.take_cpu(&qw_name)?;
                    let n = qw_shape[0];
                    let k_packed = qw_shape[1];
                    let transposed =
                        transpose_2d_cpu(&qw_bytes, n, k_packed, qw_dtype.size_bytes());
                    qw_parts.push((transposed, vec![k_packed, n], qw_dtype));

                    let sc_name = format!("{prefix}.weight_scale");
                    let (sc_bytes, sc_shape, sc_dtype) = weights.take_cpu(&sc_name)?;
                    let sc_n = sc_shape[0];
                    let sc_groups = sc_shape[1];
                    let sc_transposed =
                        transpose_2d_cpu(&sc_bytes, sc_n, sc_groups, sc_dtype.size_bytes());
                    sc_parts.push((sc_transposed, vec![sc_groups, sc_n], sc_dtype));

                    let shape_name = format!("{prefix}.weight_shape");
                    if weights.contains(&shape_name) {
                        let _ = weights.take_cpu(&shape_name);
                    }
                }
            }

            // Symmetric GPTQ: consume-and-discard `.qzeros` if
            // present. Marlin's `uint4b8` bakes in the bias=8 zp.
            // compressed-tensors INT4 symmetric doesn't ship qzeros.
            let qzeros_name = format!("{prefix}.qzeros");
            if weights.contains(&qzeros_name) {
                let _ = weights.take_cpu(&qzeros_name);
            }

            // `.g_idx` shared across fused sub-weights — read from
            // first prefix, consume from the rest. compressed-tensors
            // ships the same info as `.weight_g_idx` when the config
            // carries `actorder: "group"`; pick the right name per
            // layout so the permutation lands in the repack.
            let g_idx_name = match layout {
                GptqLayout::Qweight => format!("{prefix}.g_idx"),
                GptqLayout::WeightPacked => format!("{prefix}.weight_g_idx"),
            };
            // Same Qweight-vs-WeightPacked split as single-shot
            // `load_gptq`: compile-time `desc_act` is authoritative
            // for AutoGPTQ; runtime tensor presence for CT.
            let take_g_idx = match layout {
                GptqLayout::Qweight => desc_act && weights.contains(&g_idx_name),
                GptqLayout::WeightPacked => weights.contains(&g_idx_name),
            };
            if take_g_idx {
                if i == 0 {
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

            let bias_name = format!("{prefix}.bias");
            if weights.contains(&bias_name) {
                let (b_data, _b_shape, b_dtype) = weights.take_cpu(&bias_name)?;
                bias_parts.push(b_data);
                bias_dtype = Some(b_dtype);
            }
        }

        // Concat qweights along dim 1: [K/8, N_i] → [K/8, N_total].
        let qw_refs: Vec<_> = qw_parts
            .iter()
            .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
            .collect();
        let (qw_fused, qw_shape, _) = concat_cpu_dim1(&qw_refs);
        let size_k = qw_shape[0] * 8;
        let size_n = qw_shape[1];
        let num_groups = size_k.checked_div(group_size).unwrap_or(1);

        // g_idx BEFORE repack — repack consumes sort_indices.
        let (g_idx_gpu, sort_indices_gpu, has_act_order) = if let Some(g_idx) = g_idx_i32 {
            let mut sort_indices: Vec<i32> = (0..g_idx.len() as i32).collect();
            sort_indices.sort_by_key(|&i| g_idx[i as usize]);

            let sorted_g_idx: Vec<i32> = sort_indices.iter().map(|&i| g_idx[i as usize]).collect();

            let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let g_idx_nbytes = g_idx_bytes.len();
            let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
            weights.record_alloc(g_idx_ptr, g_idx_nbytes);
            unsafe {
                driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
            }
            let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

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

        // Upload fused qweight, repack once.
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

        // Concat + permute scales: [num_groups, N_i] → [num_groups, N_total].
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
        marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

        let scales_bytes_permuted: Vec<u8> =
            scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
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

        Ok(Self {
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
}

// `CachingAllocator` isn't used inside these loaders (the AWQ ones in
// scratchy-serving-cuda took `_alloc` for symmetry with the GPTQ path). Imported
// here so the `use` stays minimal should a future loader want it.
#[allow(dead_code)]
fn _touch(alloc: &mut CachingAllocator) -> &mut CachingAllocator {
    alloc
}

// ---------------------------------------------------------------------------
// BitsAndBytes 4-bit (NF4 / FP4) loader support
// ---------------------------------------------------------------------------

/// BNB 4-bit packing type. Selects the 16-entry lookup table the
/// dequant kernel consults — `NF4` (quantiles of N(0,1)) or `FP4`
/// (E2M1 float values). Picked at macro-expansion time from the HF
/// `quantization_config.bnb_4bit_quant_type` field and folded into
/// the emitted `Weights::load` prelude's `upload_bnb_code` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BnbQuantType {
    NF4,
    FP4,
}

/// NF4 code table — 16 quantiles of the standard normal distribution,
/// rescaled to `[-1, 1]`. Straight from bitsandbytes; fixed forever.
#[allow(clippy::excessive_precision)]
pub const NF4_CODE: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// FP4 code table — E2M1 floats used by bitsandbytes FP4 quant.
pub const FP4_CODE: [f32; 16] = [
    0.0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0,
];

/// Upload the 16-entry NF4/FP4 lookup table to GPU. One per-model
/// allocation — every `Bnb4bitLinear` on that device captures the
/// same pointer by value (`GpuTensor` is `Copy`).
pub fn upload_bnb_code(code: &[f32; 16], stream: CUstream) -> Result<GpuTensor> {
    let nbytes = 16 * std::mem::size_of::<f32>();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    unsafe {
        driver::memcpy_htod_async(ptr, code.as_ptr() as *const u8, nbytes, stream)?;
    }
    Ok(unsafe { GpuTensor::new(ptr, &[16], DType::F32) })
}

/// Allocate the per-model shared dequantization scratch buffer used
/// by every `Bnb4bitLinear::forward` on this device. Sized to
/// `max(out_features × in_features)` across every linear layer in
/// the model — the caller computes the max at macro-expansion time
/// from the arch config and hands it in. `dtype` is the compute
/// dtype (bf16/fp16); the dequant kernel writes into this buffer
/// before cuBLAS reads it.
pub fn alloc_bnb_dequant_scratch(
    max_elements: usize,
    dtype: DType,
    stream: CUstream,
) -> Result<GpuTensor> {
    let nbytes = max_elements * dtype.size_bytes();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    unsafe { driver::memset_d8(ptr, 0, nbytes, stream)? };
    Ok(unsafe { GpuTensor::new(ptr, &[max_elements], dtype) })
}

/// Parse the BNB `quant_state.bitsandbytes__nf4` JSON blob that
/// bitsandbytes embeds in the safetensors. Returns
/// `(nested_offset, blocksize, nested_blocksize)`. Unknown keys are
/// ignored; trailing NUL bytes are tolerated.
fn parse_bnb_quant_state_json(data: &[u8]) -> Result<(f32, usize, usize)> {
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

/// CPU-side dequantization of BNB's double-quantized absmax.
///
/// Mirrors Python bitsandbytes' `_dequantize_dq`: each u8 index
/// picks a value from the nested dequant table, scales it by the
/// appropriate per-`nested_blocksize`-block absmax, and adds back
/// the `nested_offset`. Produces the f32 absmax vector the GPU
/// dequant kernel reads at runtime.
fn dequantize_double_quant_absmax(
    absmax_u8: &[u8],
    nested_quant_map: &[f32], // 256 entries
    nested_absmax: &[f32],    // len = num_blocks / nested_blocksize
    nested_blocksize: usize,
    nested_offset: f32,
) -> Vec<f32> {
    absmax_u8
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scale = nested_absmax[i / nested_blocksize];
            nested_quant_map[v as usize] * scale + nested_offset
        })
        .collect()
}

impl Bnb4bitLoadOps for Bnb4bitLinear {
    /// Load a single BNB 4-bit linear from safetensors.
    ///
    /// Expects: `{prefix}.weight` (U8 packed nibbles),
    /// `{prefix}.weight.absmax` (U8 double-quantized OR F32),
    /// `{prefix}.weight.nested_absmax` (F32, when double-quantized),
    /// `{prefix}.weight.nested_quant_map` (F32[256], when double-
    /// quantized), `{prefix}.weight.quant_map` (F32[16], consumed),
    /// `{prefix}.weight.quant_state.bitsandbytes__nf4` (U8 JSON
    /// blob carrying `nested_offset` / `blocksize` /
    /// `nested_blocksize`; consumed). Optional `{prefix}.bias`.
    ///
    /// `code_gpu` is the shared NF4/FP4 LUT on GPU (one per model).
    /// `dequant_scratch` is the shared dequant scratch buffer.
    #[allow(clippy::too_many_arguments)]
    fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        out_features: usize,
        in_features: usize,
        blocksize: usize,
    ) -> Result<Self> {
        let stream = weights.stream();

        let weight_name = format!("{prefix}.weight");
        let absmax_name = format!("{prefix}.weight.absmax");
        let nested_absmax_name = format!("{prefix}.weight.nested_absmax");
        let nested_quant_map_name = format!("{prefix}.weight.nested_quant_map");
        let quant_state_name = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");

        // Parse quant_state JSON for `nested_offset` + actual
        // `blocksize` (the arg is only the fallback when the tensor
        // is absent — some ancient checkpoints).
        let (nested_offset, actual_blocksize, nested_blocksize) =
            if weights.contains(&quant_state_name) {
                let (qs_bytes, _, _) = weights.take_cpu(&quant_state_name)?;
                parse_bnb_quant_state_json(&qs_bytes)?
            } else {
                (0.0, blocksize, 256)
            };
        let blocksize = actual_blocksize;

        // Packed weight goes straight to GPU (U8).
        let packed_weight = weights.take(&weight_name)?;

        // Absmax: either U8 (double-quantized — dequant on CPU) or
        // F32 (direct).
        let (absmax_bytes, _absmax_shape, absmax_dtype) = weights.take_cpu(&absmax_name)?;
        let absmax_f32: Vec<f32> = if absmax_dtype == DType::U8 {
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

        // Upload absmax to GPU.
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

        // Optional bias.
        let bias_name = format!("{prefix}.bias");
        let bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };

        // Consume remaining BNB metadata tensors so GpuWeights
        // doesn't warn about unused entries at the end of load.
        let quant_map_name = format!("{prefix}.weight.quant_map");
        for name in &[&quant_map_name, &nested_absmax_name, &nested_quant_map_name] {
            if weights.contains(name) {
                let _ = weights.take_cpu(name);
            }
        }

        Ok(Self {
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

    /// Load several BNB 4-bit linears and fuse into one wider linear
    /// by byte-concatenating the packed nibbles + absmax along the N
    /// axis. Each shard's absmax blocks are independent (blocksize
    /// divides in_features, which is shared across shards), so the
    /// concat is a straight byte append — no cross-shard rescaling
    /// needed. The fused result has `out_features =
    /// sum(out_features_per_shard)` and a single shared dequant
    /// scratch / code LUT.
    ///
    /// Bias tensors (when present on any shard) are concatenated along
    /// the N axis into a single `[sum(out_features_per_shard)]` bias on
    /// the fused layer — mirrors `MarlinLinear::load_awq_concat`. Mixing
    /// bias/no-bias across shards falls through with the biased shards
    /// only (HF checkpoints never mix, so this is just parity with the
    /// Marlin loader's tolerance).
    #[allow(clippy::too_many_arguments)]
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        out_features_per_shard: &[usize],
        in_features: usize,
        blocksize: usize,
    ) -> Result<Self> {
        assert_eq!(
            prefixes.len(),
            out_features_per_shard.len(),
            "load_concat: prefixes and out_features_per_shard length mismatch",
        );
        let stream = weights.stream();
        let total_out_features: usize = out_features_per_shard.iter().sum();

        let mut all_packed: Vec<u8> = Vec::new();
        let mut all_absmax_f32: Vec<f32> = Vec::new();
        let mut bias_parts: Vec<Vec<u8>> = Vec::new();
        let mut bias_dtype: Option<DType> = None;

        for prefix in prefixes {
            let weight_name = format!("{prefix}.weight");
            let absmax_name = format!("{prefix}.weight.absmax");
            let nested_absmax_name = format!("{prefix}.weight.nested_absmax");
            let nested_quant_map_name = format!("{prefix}.weight.nested_quant_map");
            let quant_state_name = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");

            let (nested_offset, _actual_blocksize, nested_blocksize) =
                if weights.contains(&quant_state_name) {
                    let (qs_bytes, _, _) = weights.take_cpu(&quant_state_name)?;
                    parse_bnb_quant_state_json(&qs_bytes)?
                } else {
                    (0.0, blocksize, 256)
                };

            let (packed_bytes, _, _) = weights.take_cpu(&weight_name)?;
            all_packed.extend_from_slice(&packed_bytes);

            let (absmax_bytes, _, absmax_dtype) = weights.take_cpu(&absmax_name)?;
            let shard_absmax: Vec<f32> = if absmax_dtype == DType::U8 {
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

            let quant_map_name = format!("{prefix}.weight.quant_map");
            for name in &[&quant_map_name, &nested_absmax_name, &nested_quant_map_name] {
                if weights.contains(name) {
                    let _ = weights.take_cpu(name);
                }
            }

            // Bias is NOT part of BNB's 4-bit quantization; shipped as
            // a regular `[N_shard]` tensor per-prefix alongside `.weight`.
            // Concat along N so the fused layer carries one
            // `[sum_N_shard]` bias — matches Python vLLM's per-shard
            // `matmul_4bit` + single post-GEMM bias add.
            let bias_name = format!("{prefix}.bias");
            if weights.contains(&bias_name) {
                let (b_data, _b_shape, b_dtype) = weights.take_cpu(&bias_name)?;
                bias_parts.push(b_data);
                bias_dtype = Some(b_dtype);
            }
        }

        // Upload fused packed bytes.
        let packed_nbytes = all_packed.len();
        let packed_ptr = unsafe { driver::mem_alloc(packed_nbytes)? };
        weights.record_alloc(packed_ptr, packed_nbytes);
        unsafe {
            driver::memcpy_htod_async(packed_ptr, all_packed.as_ptr(), packed_nbytes, stream)?;
        }
        let packed_gpu = unsafe { GpuTensor::new(packed_ptr, &[packed_nbytes], DType::U8) };

        // Upload fused absmax.
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

        let bias = fuse_bias_parts(&bias_parts, bias_dtype, weights, stream)?;

        Ok(Self {
            packed_weight: packed_gpu,
            absmax: absmax_gpu,
            code: code_gpu,
            dequant_scratch,
            out_features: total_out_features,
            in_features,
            blocksize,
            bias,
        })
    }
}

// ---------------------------------------------------------------------------
// FP8 Weight Loading
// ---------------------------------------------------------------------------

/// Ensure a scale tensor is f32. If it's BF16/F16, download → convert → re-upload.
/// Also flatten `[N, 1]` → `[N]`.
///
/// CUTLASS epilogue templates require `float*` scale pointers. Compressed-tensors
/// models (e.g., RedHatAI, neuralmagic) store `weight_scale` as BF16 `[N, 1]`.
pub fn ensure_f32_scale(scale: GpuTensor, stream: CUstream) -> Result<GpuTensor> {
    // Flatten [N, 1] → [N]
    let flat = if scale.ndim() == 2 && scale.dim(1) == 1 {
        scale.reshape(&[scale.dim(0)])
    } else {
        scale
    };

    if flat.dtype() == DType::F32 {
        return Ok(flat);
    }

    // Scale tensors are small (at most N elements, e.g., 4096).
    // Download to CPU, convert BF16/F16 → f32, re-upload.
    let numel = flat.numel();
    let src_bytes = numel * flat.dtype().size_bytes();
    let mut host_src = vec![0u8; src_bytes];
    unsafe {
        driver::memcpy_dtoh_async(host_src.as_mut_ptr(), flat.raw_ptr(), src_bytes, stream)?;
        driver::stream_synchronize(stream)?;
    }

    let f32_data: Vec<f32> = match flat.dtype() {
        DType::BF16 => {
            let u16s =
                unsafe { std::slice::from_raw_parts(host_src.as_ptr() as *const u16, numel) };
            u16s.iter()
                .map(|&bits| half::bf16::from_bits(bits).to_f32())
                .collect()
        }
        DType::F16 => {
            let u16s =
                unsafe { std::slice::from_raw_parts(host_src.as_ptr() as *const u16, numel) };
            u16s.iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect()
        }
        dt => anyhow::bail!("ensure_f32_scale: unsupported dtype {dt}"),
    };

    let f32_bytes = numel * 4;
    let f32_ptr = unsafe { driver::mem_alloc(f32_bytes)? };
    unsafe {
        driver::memcpy_htod_async(f32_ptr, f32_data.as_ptr() as *const u8, f32_bytes, stream)?;
    }

    let shape_usize: Vec<usize> = flat.shape().iter().map(|&d| d as usize).collect();
    Ok(unsafe { GpuTensor::new(f32_ptr, &shape_usize, DType::F32) })
}

/// Resolve the block scale tensor name for a prefix.
///
/// Block-quantized FP8 checkpoints use either `weight_scale_inv` (DeepSeek-V3,
/// Qwen3-MoE) or `weight_scale` (unsloth, some compressed-tensors models).
/// Try `weight_scale_inv` first, fall back to `weight_scale`.
pub fn block_scale_name(weights: &GpuWeights, prefix: &str) -> String {
    let inv = format!("{prefix}.weight_scale_inv");
    if weights.contains(&inv) {
        inv
    } else {
        format!("{prefix}.weight_scale")
    }
}

impl Fp8LoadOps for Fp8Linear {
    /// Load an FP8 linear layer from a serialized FP8 checkpoint.
    ///
    /// Expects:
    /// - `{prefix}.weight`: FP8 E4M3 `[out_features, in_features]`
    /// - `{prefix}.weight_scale`: f32 or BF16 (per-tensor `[1]` or per-channel `[N, 1]`)
    /// - `{prefix}.input_scale` (optional): f32 scalar (static activation scale)
    /// - `{prefix}.bias` (optional): BF16/F16 `[out_features]`
    ///
    /// Matches Python vLLM's `Fp8LinearMethod.create_weights()` +
    /// `process_weights_after_loading()` for serialized FP8 checkpoints.
    /// Also supports online BF16→FP8 quantization when the checkpoint ships BF16.
    fn load(weights: &mut GpuWeights, prefix: &str, output_dtype: DType) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let input_scale_name = format!("{prefix}.input_scale");
        let bias_name = format!("{prefix}.bias");

        let weight = weights.take(&weight_name)?;
        anyhow::ensure!(weight.ndim() == 2, "FP8 weight must be 2D");

        let stream = weights.stream();
        let (fp8_weight, weight_scale) = if weight.dtype() == DType::Fp8E4m3 {
            // Serialized FP8 checkpoint: weight is already FP8, scale is pre-computed.
            let raw_scale = weights.take(&scale_name)?;
            // Ensure scale is f32 (CUTLASS epilogue requires float* scales).
            // compressed-tensors models store weight_scale as BF16 [N, 1].
            let weight_scale = ensure_f32_scale(raw_scale, stream)?;
            (weight, weight_scale)
        } else if weight.dtype() == DType::BF16 || weight.dtype() == DType::F16 {
            // Online FP8 quantization: BF16/F16 checkpoint → quantize to FP8 at load time.
            // Matches Python's `Fp8OnlineLinearMethod`.
            anyhow::ensure!(
                weight.dtype() == DType::BF16,
                "Online FP8 quant currently supports BF16 only, got {}",
                weight.dtype()
            );
            let n = weight.dim(0);
            let k = weight.dim(1);
            let num_elements = n * k;

            // Allocate FP8 output weight + scale on GPU.
            let fp8_ptr = unsafe { driver::mem_alloc(num_elements)? };
            let scale_ptr = unsafe { driver::mem_alloc(4)? };

            // Run online weight quantization kernel (absmax → scale → quantize).
            unsafe {
                crate::kernels::fp8_quantize_weight_bf16_raw(
                    weight.as_ptr() as *const u16,
                    fp8_ptr as *mut u8,
                    scale_ptr as *mut f32,
                    num_elements as i32,
                    std::ptr::null_mut(), // null stream = synchronous
                );
            }

            let fp8_weight = unsafe { GpuTensor::new(fp8_ptr, &[n, k], DType::Fp8E4m3) };
            let weight_scale = unsafe { GpuTensor::new(scale_ptr, &[1], DType::F32) };

            // Free the original BF16 weight — we no longer need it.
            let _ = weight;

            (fp8_weight, weight_scale)
        } else {
            anyhow::bail!(
                "FP8 linear: expected Fp8E4m3 or BF16 weight, got {}",
                weight.dtype()
            );
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

        Ok(Fp8Linear {
            weight: fp8_weight,
            weight_scale,
            input_scale,
            bias,
            output_dtype,
        })
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
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        output_dtype: DType,
    ) -> Result<Self> {
        anyhow::ensure!(!prefixes.is_empty(), "Fp8Linear::load_concat: no prefixes");
        let stream = weights.stream();

        // Get shapes from first prefix.
        let first_weight_name = format!("{}.weight", prefixes[0]);
        let (first_shape, first_dtype) = weights
            .tensor_info(&first_weight_name)
            .ok_or_else(|| anyhow::anyhow!("FP8: weight not found: {first_weight_name}"))?;
        let is_online_quant = first_dtype == DType::BF16 || first_dtype == DType::F16;
        anyhow::ensure!(
            first_dtype == DType::Fp8E4m3 || is_online_quant,
            "FP8 fused weight expected Fp8E4m3 or BF16, got {first_dtype}"
        );
        anyhow::ensure!(first_shape.len() == 2, "FP8 fused weight must be 2D");
        let in_features = first_shape[1];

        // Sum up output dimensions.
        let mut total_out = 0usize;
        let mut shard_sizes = Vec::with_capacity(prefixes.len());
        for prefix in prefixes {
            let wname = format!("{prefix}.weight");
            let (shape, _) = weights
                .tensor_info(&wname)
                .ok_or_else(|| anyhow::anyhow!("FP8: weight not found: {wname}"))?;
            shard_sizes.push(shape[0]);
            total_out += shape[0];
        }

        let (fused_weight, merged_scale) = if is_online_quant {
            // Online quantization: load BF16 shards → fuse → quantize entire fused weight to FP8.
            // This produces a single per-tensor FP8 weight + scale (no re-quantization needed
            // since we quantize the fused weight as a whole).
            let bf16_elem_size = first_dtype.size_bytes();
            let bf16_total_bytes = total_out * in_features * bf16_elem_size;
            let bf16_ptr = unsafe { driver::mem_alloc(bf16_total_bytes)? };

            // Copy each BF16 shard into the fused buffer.
            let mut offset = 0usize;
            for (i, prefix) in prefixes.iter().enumerate() {
                let wname = format!("{prefix}.weight");
                let shard_bytes = shard_sizes[i] * in_features * bf16_elem_size;
                unsafe {
                    weights.take_into(&wname, bf16_ptr.add(offset), stream)?;
                }
                offset += shard_bytes;
            }

            // Allocate FP8 output + scale.
            let num_elements = total_out * in_features;
            let fp8_ptr = unsafe { driver::mem_alloc(num_elements)? };
            let scale_ptr = unsafe { driver::mem_alloc(4)? };

            // Quantize the entire fused BF16 weight to FP8.
            unsafe {
                crate::kernels::fp8_quantize_weight_bf16_raw(
                    bf16_ptr as *const u16,
                    fp8_ptr as *mut u8,
                    scale_ptr as *mut f32,
                    num_elements as i32,
                    stream,
                );
                // Free the BF16 buffer.
                driver::mem_free(bf16_ptr)?;
            }

            let fused_weight =
                unsafe { GpuTensor::new(fp8_ptr, &[total_out, in_features], DType::Fp8E4m3) };
            let scale = unsafe { GpuTensor::new(scale_ptr, &[1], DType::F32) };

            // Consume any weight_scale tensors that exist in the checkpoint
            // (online quant models may or may not have them).
            for prefix in prefixes {
                let scale_name = format!("{prefix}.weight_scale");
                if weights.contains(&scale_name) {
                    let _ = weights.take(&scale_name);
                }
            }

            (fused_weight, scale)
        } else {
            // Serialized FP8 checkpoint: weights already FP8, merge per-shard scales.
            let elem_size = DType::Fp8E4m3.size_bytes();
            let total_bytes = total_out * in_features * elem_size;
            let fused_ptr = unsafe { driver::mem_alloc(total_bytes)? };

            // Copy each FP8 shard directly.
            let mut offset = 0usize;
            for (i, prefix) in prefixes.iter().enumerate() {
                let wname = format!("{prefix}.weight");
                let shard_bytes = shard_sizes[i] * in_features * elem_size;
                unsafe {
                    weights.take_into(&wname, fused_ptr.add(offset), stream)?;
                }
                offset += shard_bytes;
            }

            let fused_weight =
                unsafe { GpuTensor::new(fused_ptr, &[total_out, in_features], DType::Fp8E4m3) };

            // Load per-shard scales and determine if they're per-tensor [1] or per-channel [N, 1].
            let first_scale_name = format!("{}.weight_scale", prefixes[0]);
            let (first_scale_shape, _first_scale_dtype) =
                weights.tensor_info(&first_scale_name).ok_or_else(|| {
                    anyhow::anyhow!("FP8: weight_scale not found: {first_scale_name}")
                })?;
            let is_per_channel = first_scale_shape.iter().product::<usize>() > 1;

            let merged_scale = if is_per_channel {
                // Per-channel scales: concatenate along dim=0 and convert to f32.
                // Each shard has [N_shard, 1] scale → fused is [N_total] f32.
                let total_scale_f32_bytes = total_out * 4;
                let scale_ptr = unsafe { driver::mem_alloc(total_scale_f32_bytes)? };
                let mut f32_offset = 0usize;

                for (i, prefix) in prefixes.iter().enumerate() {
                    let scale_name = format!("{prefix}.weight_scale");
                    let raw_scale = weights.take(&scale_name)?;
                    let shard_scale = ensure_f32_scale(raw_scale, stream)?;
                    let shard_bytes = shard_sizes[i] * 4;
                    unsafe {
                        driver::memcpy_dtod_async(
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
                // Per-tensor scales: take max of all per-shard scales, then re-quantize
                // shards with smaller scales so all rows use the unified max scale.
                // This matches Python's `requantize_with_max_scale()`.
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

                // Re-quantize shards whose scale differs from max_scale.
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

                // Upload merged scale to GPU.
                let scale_ptr = unsafe { driver::mem_alloc(4)? };
                unsafe {
                    driver::memcpy_htod_async(
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

        // Input scale: use first prefix's if available (they should all be the same).
        let input_scale_name = format!("{}.input_scale", prefixes[0]);
        let input_scale = if weights.contains(&input_scale_name) {
            let raw = weights.take(&input_scale_name)?;
            Some(ensure_f32_scale(raw, stream)?)
        } else {
            None
        };

        // Fuse bias if present.
        let bias_name = format!("{}.bias", prefixes[0]);
        let bias = if weights.contains(&bias_name) {
            let (_bias_shape, bias_dtype) = weights
                .tensor_info(&bias_name)
                .ok_or_else(|| anyhow::anyhow!("FP8: bias not found"))?;
            let bias_elem_size = bias_dtype.size_bytes();
            let mut total_bias_bytes = 0;
            for &sz in &shard_sizes[..prefixes.len()] {
                total_bias_bytes += sz * bias_elem_size;
            }
            let bias_ptr = unsafe { driver::mem_alloc(total_bias_bytes)? };
            let mut boff = 0;
            for (i, prefix) in prefixes.iter().enumerate() {
                let bname = format!("{prefix}.bias");
                let bbytes = shard_sizes[i] * bias_elem_size;
                unsafe {
                    weights.take_into(&bname, bias_ptr.add(boff), stream)?;
                }
                boff += bbytes;
            }
            let total_bias_elems = total_bias_bytes / bias_elem_size;
            Some(unsafe { GpuTensor::new(bias_ptr, &[total_bias_elems], bias_dtype) })
        } else {
            None
        };

        Ok(Fp8Linear {
            weight: fused_weight,
            weight_scale: merged_scale,
            input_scale,
            bias,
            output_dtype,
        })
    }
}

impl Fp8BlockLoadOps for Fp8BlockLinear {
    /// Load a single FP8 block-quantized linear layer.
    ///
    /// Expects:
    /// - `{prefix}.weight`: FP8 E4M3 `[out_features, in_features]`
    /// - `{prefix}.weight_scale_inv` or `{prefix}.weight_scale`: f32 2D block scale
    /// - `{prefix}.input_scale` (optional): not used for block quant but consumed if present
    /// - `{prefix}.bias` (optional)
    ///
    /// Derives `block_size` from the ratio of weight shape to scale shape.
    fn load(weights: &mut GpuWeights, prefix: &str, output_dtype: DType) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let scale_name = block_scale_name(weights, prefix);
        let input_scale_name = format!("{prefix}.input_scale");
        let bias_name = format!("{prefix}.bias");

        let weight = weights.take(&weight_name)?;
        anyhow::ensure!(weight.ndim() == 2, "FP8 block weight must be 2D");
        anyhow::ensure!(
            weight.dtype() == DType::Fp8E4m3,
            "FP8 block linear: expected Fp8E4m3 weight, got {}. \
             Online block quantization is not yet supported in the Rust backend.",
            weight.dtype()
        );

        let n = weight.dim(0);
        let k = weight.dim(1);

        let stream = weights.stream();
        let raw_scale = weights.take(&scale_name)?;
        let scale = ensure_f32_scale(raw_scale, stream)?;
        anyhow::ensure!(
            scale.ndim() == 2,
            "FP8 block scale must be 2D, got {}D",
            scale.ndim()
        );

        let scale_rows = scale.dim(0);
        let scale_cols = scale.dim(1);
        let block_n = n / scale_rows;
        let block_k = k / scale_cols;

        // Consume input_scale if present (block quant uses dynamic activation).
        if weights.contains(&input_scale_name) {
            let _ = weights.take(&input_scale_name);
        }

        let bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };

        Ok(Fp8BlockLinear {
            weight,
            weight_scale_inv: scale,
            block_size: [block_n, block_k],
            bias,
            output_dtype,
        })
    }

    /// Load a fused FP8 block-quantized linear layer (QKV or gate_up).
    ///
    /// Concatenates multiple FP8 weight shards along dim=0 and their 2D block
    /// scales along dim=0. All shards share the same in_features, so scale dim=1
    /// (input blocks) is identical across shards.
    fn load_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        output_dtype: DType,
    ) -> Result<Self> {
        anyhow::ensure!(
            !prefixes.is_empty(),
            "Fp8BlockLinear::load_concat: no prefixes"
        );
        let stream = weights.stream();

        // Get shapes from first prefix.
        let first_weight_name = format!("{}.weight", prefixes[0]);
        let (first_shape, first_dtype) = weights
            .tensor_info(&first_weight_name)
            .ok_or_else(|| anyhow::anyhow!("FP8 block: weight not found: {first_weight_name}"))?;
        anyhow::ensure!(
            first_dtype == DType::Fp8E4m3,
            "FP8 block fused: expected Fp8E4m3 weight, got {first_dtype}. \
             Online block quantization is not yet supported in the Rust backend."
        );
        anyhow::ensure!(first_shape.len() == 2, "FP8 block fused weight must be 2D");
        let in_features = first_shape[1];

        // Derive block_size from first shard's weight and scale shapes.
        let first_scale_name = block_scale_name(weights, prefixes[0]);
        let (first_scale_shape, _) = weights
            .tensor_info(&first_scale_name)
            .ok_or_else(|| anyhow::anyhow!("FP8 block: scale not found: {first_scale_name}"))?;
        anyhow::ensure!(
            first_scale_shape.len() == 2,
            "FP8 block scale must be 2D, got {}D",
            first_scale_shape.len()
        );
        let block_n = first_shape[0] / first_scale_shape[0];
        let block_k = first_shape[1] / first_scale_shape[1];
        let scale_cols = first_scale_shape[1]; // same for all shards

        // Sum up output dimensions and scale rows.
        let mut total_out = 0usize;
        let mut total_scale_rows = 0usize;
        let mut shard_sizes = Vec::with_capacity(prefixes.len());
        let mut shard_scale_rows = Vec::with_capacity(prefixes.len());
        for prefix in prefixes {
            let wname = format!("{prefix}.weight");
            let (shape, _) = weights
                .tensor_info(&wname)
                .ok_or_else(|| anyhow::anyhow!("FP8 block: weight not found: {wname}"))?;
            shard_sizes.push(shape[0]);
            total_out += shape[0];

            let sname = block_scale_name(weights, prefix);
            let (sshape, _) = weights
                .tensor_info(&sname)
                .ok_or_else(|| anyhow::anyhow!("FP8 block: scale not found: {sname}"))?;
            shard_scale_rows.push(sshape[0]);
            total_scale_rows += sshape[0];
        }

        // Allocate fused FP8 weight buffer.
        let elem_size = DType::Fp8E4m3.size_bytes();
        let total_bytes = total_out * in_features * elem_size;
        let fused_ptr = unsafe { driver::mem_alloc(total_bytes)? };

        // Copy each FP8 shard into the fused buffer.
        let mut offset = 0usize;
        for (i, prefix) in prefixes.iter().enumerate() {
            let wname = format!("{prefix}.weight");
            let shard_bytes = shard_sizes[i] * in_features * elem_size;
            unsafe {
                weights.take_into(&wname, fused_ptr.add(offset), stream)?;
            }
            offset += shard_bytes;
        }

        let fused_weight =
            unsafe { GpuTensor::new(fused_ptr, &[total_out, in_features], DType::Fp8E4m3) };

        // Allocate fused scale buffer and concat scale shards along dim=0.
        let total_scale_bytes = total_scale_rows * scale_cols * 4; // f32
        let scale_ptr = unsafe { driver::mem_alloc(total_scale_bytes)? };
        let mut scale_offset = 0usize;
        for (i, prefix) in prefixes.iter().enumerate() {
            let sname = block_scale_name(weights, prefix);
            let raw_scale = weights.take(&sname)?;
            let shard_scale = ensure_f32_scale(raw_scale, stream)?;
            let shard_bytes = shard_scale_rows[i] * scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
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

        // Consume input_scale if present (block quant uses dynamic activation).
        let input_scale_name = format!("{}.input_scale", prefixes[0]);
        if weights.contains(&input_scale_name) {
            let _ = weights.take(&input_scale_name);
        }

        // Fuse bias if present.
        let bias_name = format!("{}.bias", prefixes[0]);
        let bias = if weights.contains(&bias_name) {
            let (_bias_shape, bias_dtype) = weights
                .tensor_info(&bias_name)
                .ok_or_else(|| anyhow::anyhow!("FP8 block: bias not found"))?;
            let bias_elem_size = bias_dtype.size_bytes();
            let mut total_bias_bytes = 0;
            for &sz in &shard_sizes {
                total_bias_bytes += sz * bias_elem_size;
            }
            let bias_ptr = unsafe { driver::mem_alloc(total_bias_bytes)? };
            let mut boff = 0;
            for (i, prefix) in prefixes.iter().enumerate() {
                let bname = format!("{prefix}.bias");
                let bbytes = shard_sizes[i] * bias_elem_size;
                unsafe {
                    weights.take_into(&bname, bias_ptr.add(boff), stream)?;
                }
                boff += bbytes;
            }
            let total_bias_elems = total_bias_bytes / bias_elem_size;
            Some(unsafe { GpuTensor::new(bias_ptr, &[total_bias_elems], bias_dtype) })
        } else {
            None
        };

        Ok(Fp8BlockLinear {
            weight: fused_weight,
            weight_scale_inv: fused_scale,
            block_size: [block_n, block_k],
            bias,
            output_dtype,
        })
    }
}
