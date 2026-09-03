// SPDX-License-Identifier: Apache-2.0
//! Rust dispatcher for the ascending row-wise argsort kernel
//! (port of MLX `block_sort`). MoE router uses this to express
//! `mx.argpartition(±gates, kth=±k, axis=-1)` — a full sort
//! plus trailing-k slice is correct for the small router widths
//! (E ≤ 256) we target.
//!
//! The output is `[rows, axis_size]` u32 with each row holding
//! the ascending-sorted indices of the input row. NaN entries
//! (padding when N_PER_BLOCK > axis_size) land at the right
//! tail per MLX `LessThan`'s NaN-as-greater convention; the
//! `slice_trailing_cols_u32` kernel pulls the contiguous
//! `[axis_size - top_k, axis_size)` window which is exactly the
//! top-k indices by value.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgsortDType {
    F32,
    F16,
    Bf16,
    U32,
}

/// (BLOCK_THREADS, N_PER_THREAD) pairs we instantiate the kernel at.
/// `N_PER_BLOCK = bn * tn`. Pick the smallest that covers
/// `axis_size`.
const PIPELINES: &[(usize, usize)] = &[(32, 4), (64, 4)];

pub struct ArgsortKernels {
    pub f32_bn32_tn4: ComputePipelineState,
    pub f16_bn32_tn4: ComputePipelineState,
    pub bf16_bn32_tn4: ComputePipelineState,
    pub u32_bn32_tn4: ComputePipelineState,
    pub f32_bn64_tn4: ComputePipelineState,
    pub u32_bn64_tn4: ComputePipelineState,
    pub f16_bn64_tn4: ComputePipelineState,
    pub bf16_bn64_tn4: ComputePipelineState,
    _library: Library,
}

impl ArgsortKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("argpartition"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `argpartition.metallib`: {e}"
                ))
            })?;
        let f32_bn32_tn4 =
            build_pipeline(device, &library, "c_arg_block_sort_float32_uint32_bn32_tn4")?;
        let f16_bn32_tn4 =
            build_pipeline(device, &library, "c_arg_block_sort_float16_uint32_bn32_tn4")?;
        let bf16_bn32_tn4 = build_pipeline(
            device,
            &library,
            "c_arg_block_sort_bfloat16_uint32_bn32_tn4",
        )?;
        let u32_bn32_tn4 =
            build_pipeline(device, &library, "c_arg_block_sort_uint32_uint32_bn32_tn4")?;
        let f32_bn64_tn4 =
            build_pipeline(device, &library, "c_arg_block_sort_float32_uint32_bn64_tn4")?;
        let u32_bn64_tn4 =
            build_pipeline(device, &library, "c_arg_block_sort_uint32_uint32_bn64_tn4")?;
        let f16_bn64_tn4 =
            build_pipeline(device, &library, "c_arg_block_sort_float16_uint32_bn64_tn4")?;
        let bf16_bn64_tn4 = build_pipeline(
            device,
            &library,
            "c_arg_block_sort_bfloat16_uint32_bn64_tn4",
        )?;
        Ok(Self {
            f32_bn32_tn4,
            f16_bn32_tn4,
            bf16_bn32_tn4,
            u32_bn32_tn4,
            f32_bn64_tn4,
            u32_bn64_tn4,
            f16_bn64_tn4,
            bf16_bn64_tn4,
            _library: library,
        })
    }

    pub fn pipeline_for(
        &self,
        dtype: ArgsortDType,
        bn: usize,
        tn: usize,
    ) -> Result<&ComputePipelineState, MetalStreamError> {
        match (dtype, bn, tn) {
            (ArgsortDType::F32, 32, 4) => Ok(&self.f32_bn32_tn4),
            (ArgsortDType::F16, 32, 4) => Ok(&self.f16_bn32_tn4),
            (ArgsortDType::Bf16, 32, 4) => Ok(&self.bf16_bn32_tn4),
            (ArgsortDType::U32, 32, 4) => Ok(&self.u32_bn32_tn4),
            (ArgsortDType::F32, 64, 4) => Ok(&self.f32_bn64_tn4),
            (ArgsortDType::U32, 64, 4) => Ok(&self.u32_bn64_tn4),
            (ArgsortDType::F16, 64, 4) => Ok(&self.f16_bn64_tn4),
            (ArgsortDType::Bf16, 64, 4) => Ok(&self.bf16_bn64_tn4),
            _ => Err(MetalStreamError::ShaderCompilationFailed(format!(
                "no pipeline for ({:?}, bn={bn}, tn={tn})",
                dtype
            ))),
        }
    }
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
}

/// Smallest (bn, tn) pair such that `bn * tn >= axis_size`. Used
/// by callers that don't know the size of the input ahead of time.
pub fn pick_pipeline_shape(axis_size: usize) -> Result<(usize, usize), MetalStreamError> {
    for &(bn, tn) in PIPELINES {
        if bn * tn >= axis_size {
            return Ok((bn, tn));
        }
    }
    Err(MetalStreamError::ShaderCompilationFailed(format!(
        "axis_size={axis_size} exceeds max argpartition N_PER_BLOCK ({})",
        PIPELINES.iter().map(|&(b, t)| b * t).max().unwrap_or(0)
    )))
}
