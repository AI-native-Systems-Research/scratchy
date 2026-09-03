// SPDX-License-Identifier: Apache-2.0
//! Spyre (host) quantized-linear LOADER surface — the `-Fspyre` peer of
//! `crates/targets/cuda/src/layers_quant.rs`.
//!
//! The payload TYPES (`MarlinLinear` / `Bnb4bitLinear` / `Fp8Linear` /
//! `Fp8BlockLinear`) are NEUTRAL — they live in `scratchy-layers` and hold
//! `GpuTensor` handles that, under `-Fspyre`, wrap host pointers behind
//! [`crate::SpyreAllocator`]. An inherent `impl <Type>` here would be an
//! orphan (E0116), so the `GpuWeights`-bound loaders are extension trait
//! methods on the neutral types — exactly the shape cuda uses (`MarlinLoadOps`
//! / `Fp8LoadOps` / `Fp8BlockLoadOps` / `Bnb4bitLoadOps`), with byte-identical
//! method names and signatures so the `#[forward]` macro's shared
//! `Weights::load` body type-checks under spyre with no special-casing.
//!
//! Spyre is a host fp16 backend that loads weights allocator-generically
//! (read bytes from `GpuWeights<SpyreAllocator>` + construct the neutral
//! struct — no GPU kernels). The dense / RmsNorm / Embedding loads are real;
//! the QUANT packers below (AWQ/GPTQ→Marlin, FP8, FP8-block, BNB4) are
//! kernel-bound repack pipelines that aren't wired for the Spyre AIU yet, so
//! every method here is a clean recoverable `anyhow::bail!` — the SIGNATURES
//! exist so a bf16/fp16 checkpoint's load body compiles (such a checkpoint
//! never reaches these arms), and a quantized checkpoint fails at runtime with
//! a precise "not yet implemented" message rather than a compile error.
//!
//! The on-disk quant-format SPEC types (`MarlinFormat` / `GptqLayout`) and the
//! BNB code tables (`NF4_CODE` / `FP4_CODE`) are NOT in `scratchy-layers` (they
//! were cuda-local), and spyre may not depend on `targets/cuda`, so they are
//! re-defined here verbatim — the macro names them by path
//! (`crate::__gpu::layers_quant::MarlinFormat`, `::GptqLayout`, `::NF4_CODE`,
//! `::upload_bnb_code`, …), so they must resolve in THIS module.

use anyhow::Result;

use crate::weights::GpuWeights;
use scratchy_layers::layers::{Bnb4bitLinear, Fp8BlockLinear, Fp8Linear, MarlinLinear};
use scratchy_quantizations::AffineQuantEmbedding;
use scratchy_tensors::{DType, GpuTensor, LoadStream};

/// Load extension trait on the neutral [`AffineQuantEmbedding`] — the MLX-affine
/// int4 token embedding the `#[forward]` macro's `qembed` variant constructs via
/// `AffineQuantEmbedding::load(gw, prefix, group_size, bits)`. The host int4
/// embedding path is not yet wired, so it `bail!`s — the signature exists so the
/// qembed variant's load body type-checks (a bf16 checkpoint never calls it).
pub trait AffineQuantEmbeddingOps {
    fn load(gw: &mut GpuWeights, prefix: &str, group_size: u32, bits: u32) -> Result<Self>
    where
        Self: Sized;
}

impl AffineQuantEmbeddingOps for AffineQuantEmbedding {
    fn load(_gw: &mut GpuWeights, _prefix: &str, _group_size: u32, _bits: u32) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: AffineQuantEmbedding load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// Quant-format spec types (mirrored from targets/cuda — spyre can't depend on
// targets/cuda, and these aren't in the neutral scratchy-layers leaf).
// ---------------------------------------------------------------------------

/// GPTQ on-disk layout passed by `#[forward]`-emitted `Weights::load` bodies
/// to [`MarlinLinear::load_gptq`] / [`MarlinLinear::load_gptq_concat`].
///
/// Both variants would end up at the same repack with the same `[K/8, N]`
/// uint4b8 packed weight; the layout only selects which tensor names the
/// loader reads (and whether it transposes the bytes on the CPU first).
/// Mirror of `scratchy_target_cuda::layers_quant::GptqLayout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptqLayout {
    /// AutoGPTQ native: `.qweight [K/8, N]`, `.scales [num_groups, N]`,
    /// optional `.qzeros` + `.g_idx`.
    Qweight,
    /// compressed-tensors INT4: `.weight_packed [N, K/8]`,
    /// `.weight_scale [N, num_groups]`. No `.qzeros`, no `.g_idx`.
    WeightPacked,
}

/// Unified runtime quant-format spec. The macro threads the variant's knobs
/// through a single [`MarlinLinear::load`] / [`MarlinLinear::load_concat`] call
/// per accessor, with `MarlinFormat` as the only per-variant difference.
/// Mirror of `scratchy_target_cuda::layers_quant::MarlinFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarlinFormat {
    /// AutoAWQ 4-bit — `.qweight [K, N/8]` + `.qzeros` + `.scales`.
    Awq { group_size: u32 },
    /// AutoGPTQ / compressed-tensors 4-bit uint4b8. `layout` selects the
    /// on-disk tensor names + transpose.
    Gptq {
        group_size: u32,
        desc_act: bool,
        layout: GptqLayout,
    },
}

// ---------------------------------------------------------------------------
// Marlin workspace (host stub — Marlin's barrier-lock buffer is a CUDA notion;
// Spyre has no such kernel yet, so allocating one is a clean runtime error).
// ---------------------------------------------------------------------------

/// Allocate the shared Marlin workspace buffer. On the Spyre host backend
/// there is no Marlin GEMM kernel yet, so this is a recoverable error — the
/// `marlin_prelude` that calls it is only ever emitted for a quantized model,
/// which Spyre doesn't load. Signature mirrors
/// `scratchy_target_cuda::layers_quant::alloc_marlin_workspace`
/// (`stream` is `LoadStream` = `()` host under `-Fspyre`).
pub fn alloc_marlin_workspace(_num_sm: i32, _stream: LoadStream) -> Result<GpuTensor> {
    anyhow::bail!("scratchy-target-spyre: alloc_marlin_workspace not yet implemented")
}

// ---------------------------------------------------------------------------
// AWQ/GPTQ/CT → Marlin loaders on MarlinLinear.
// ---------------------------------------------------------------------------

/// AWQ/GPTQ/CT → Marlin loaders on [`MarlinLinear`]. Method names + signatures
/// match `scratchy_target_cuda::layers_quant::MarlinLoadOps` exactly so the
/// macro's emitted `MarlinLinear::load(...)` / `load_concat(...)` calls resolve.
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

impl MarlinLoadOps for MarlinLinear {
    fn load(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _storage: MarlinFormat,
        _workspace: GpuTensor,
        _device_id: i32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: MarlinLinear::load load not yet implemented")
    }

    fn load_concat(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _storage: MarlinFormat,
        _workspace: GpuTensor,
        _device_id: i32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: MarlinLinear::load_concat load not yet implemented")
    }

    fn load_awq(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _group_size: usize,
        _workspace: GpuTensor,
        _device_id: i32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: MarlinLinear::load_awq load not yet implemented")
    }

    fn load_awq_concat(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _group_size: usize,
        _workspace: GpuTensor,
        _device_id: i32,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: MarlinLinear::load_awq_concat load not yet implemented"
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_gptq(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _group_size: usize,
        _desc_act: bool,
        _layout: GptqLayout,
        _workspace: GpuTensor,
        _device_id: i32,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: MarlinLinear::load_gptq load not yet implemented")
    }

    #[allow(clippy::too_many_arguments)]
    fn load_gptq_concat(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _group_size: usize,
        _desc_act: bool,
        _layout: GptqLayout,
        _workspace: GpuTensor,
        _device_id: i32,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: MarlinLinear::load_gptq_concat load not yet implemented"
        )
    }
}

// ---------------------------------------------------------------------------
// FP8 (per-tensor / per-channel) loaders on Fp8Linear.
// ---------------------------------------------------------------------------

/// FP8 (per-tensor / per-channel) loaders on [`Fp8Linear`]. Method names +
/// signatures match `scratchy_target_cuda::layers_quant::Fp8LoadOps`.
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

impl Fp8LoadOps for Fp8Linear {
    /// Load a compressed-tensors fp8-DYNAMIC linear (e.g. RedHatAI granite-3.1-2b-FP8-dynamic):
    /// `{prefix}.weight` = F8_E4M3 `[out,in]` kept PACKED 1-byte (SEN143_FP8 resident = ½ f16 = the ÷2
    /// decode-bandwidth win; NO host dequant — that would be the rejected dequant-on-load), and
    /// `{prefix}.weight_scale` = BF16 `[out,1]` PER-CHANNEL (consumed on-card as `Fp8W8A8Dequant`'s
    /// `w_scale[n]`). Activations are per-token DYNAMIC ⇒ no `input_scale` in the checkpoint. The
    /// `ensure!`s validate the external checkpoint (recoverable `Err`, like the dense loader) — a bf16
    /// checkpoint never reaches this arm.
    fn load(weights: &mut GpuWeights, prefix: &str, output_dtype: DType) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take(&weight_name)?;
        anyhow::ensure!(
            weight.ndim() == 2,
            "spyre Fp8Linear: {weight_name} must be 2-D [out,in], got {}-D",
            weight.ndim()
        );
        anyhow::ensure!(
            weight.dtype() == DType::Fp8E4m3,
            "spyre Fp8Linear expects a serialized F8_E4M3 checkpoint for {weight_name} (got {}); \
             online BF16→fp8 weight quant is not wired on the host SDSC path",
            weight.dtype()
        );
        let weight_scale = weights.take(&format!("{prefix}.weight_scale"))?;
        let input_scale = {
            let n = format!("{prefix}.input_scale");
            if weights.contains(&n) {
                Some(weights.take(&n)?) // static-activation ckpts; absent for fp8-DYNAMIC (per-token)
            } else {
                None
            }
        };
        let bias = {
            let n = format!("{prefix}.bias");
            if weights.contains(&n) {
                Some(weights.take(&n)?)
            } else {
                None
            }
        };
        Ok(Fp8Linear {
            weight,
            weight_scale,
            input_scale,
            bias,
            output_dtype,
        })
    }

    fn load_concat(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _output_dtype: DType,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: Fp8Linear::load_concat load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// FP8 block-scaled loaders on Fp8BlockLinear.
// ---------------------------------------------------------------------------

/// FP8 block-scaled loaders on [`Fp8BlockLinear`]. Method names + signatures
/// match `scratchy_target_cuda::layers_quant::Fp8BlockLoadOps`.
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

impl Fp8BlockLoadOps for Fp8BlockLinear {
    fn load(_weights: &mut GpuWeights, _prefix: &str, _output_dtype: DType) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: Fp8BlockLinear::load load not yet implemented")
    }

    fn load_concat(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _output_dtype: DType,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: Fp8BlockLinear::load_concat load not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// BitsAndBytes 4-bit (NF4 / FP4) loader support.
// ---------------------------------------------------------------------------

/// BNB 4-bit packing type. Selects the 16-entry lookup table the dequant
/// kernel consults — `NF4` (quantiles of N(0,1)) or `FP4` (E2M1 float values).
/// Mirror of `scratchy_target_cuda::layers_quant::BnbQuantType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BnbQuantType {
    NF4,
    FP4,
}

/// NF4 code table — 16 quantiles of the standard normal distribution, rescaled
/// to `[-1, 1]`. Straight from bitsandbytes; fixed forever. Mirror of cuda's.
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

/// FP4 code table — E2M1 floats used by bitsandbytes FP4 quant. Mirror of cuda's.
pub const FP4_CODE: [f32; 16] = [
    0.0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0,
];

/// Upload the 16-entry NF4/FP4 lookup table. On the Spyre host backend there
/// is no BNB dequant kernel yet, so this is a recoverable error — the
/// `bnb4_prelude` that calls it is only emitted for a BNB-quantized model,
/// which Spyre doesn't load. Signature mirrors
/// `scratchy_target_cuda::layers_quant::upload_bnb_code` (`stream` is
/// `LoadStream` = `()` host under `-Fspyre`).
pub fn upload_bnb_code(_code: &[f32; 16], _stream: LoadStream) -> Result<GpuTensor> {
    anyhow::bail!("scratchy-target-spyre: upload_bnb_code not yet implemented")
}

/// Allocate the per-model shared BNB dequant scratch buffer. Recoverable error
/// on Spyre (no BNB dequant kernel yet). Signature mirrors
/// `scratchy_target_cuda::layers_quant::alloc_bnb_dequant_scratch`.
pub fn alloc_bnb_dequant_scratch(
    _max_elements: usize,
    _dtype: DType,
    _stream: LoadStream,
) -> Result<GpuTensor> {
    anyhow::bail!("scratchy-target-spyre: alloc_bnb_dequant_scratch not yet implemented")
}

/// BitsAndBytes 4-bit loaders on [`Bnb4bitLinear`]. Method names + signatures
/// match `scratchy_target_cuda::layers_quant::Bnb4bitLoadOps`.
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

impl Bnb4bitLoadOps for Bnb4bitLinear {
    #[allow(clippy::too_many_arguments)]
    fn load(
        _weights: &mut GpuWeights,
        _prefix: &str,
        _code_gpu: GpuTensor,
        _dequant_scratch: GpuTensor,
        _out_features: usize,
        _in_features: usize,
        _blocksize: usize,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: Bnb4bitLinear::load load not yet implemented")
    }

    #[allow(clippy::too_many_arguments)]
    fn load_concat(
        _weights: &mut GpuWeights,
        _prefixes: &[&str],
        _code_gpu: GpuTensor,
        _dequant_scratch: GpuTensor,
        _out_features_per_shard: &[usize],
        _in_features: usize,
        _blocksize: usize,
    ) -> Result<Self> {
        anyhow::bail!("scratchy-target-spyre: Bnb4bitLinear::load_concat load not yet implemented")
    }
}
