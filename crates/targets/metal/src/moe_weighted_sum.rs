// SPDX-License-Identifier: Apache-2.0
//! MoE weighted-sum reduction dispatcher. Mirrors the trailing
//! Python expression `y = (y * scores[..., None]).sum(axis=-2)`
//! from qwen3_moe.py:137 / qwen2_moe.py:138 / mixtral.py:119.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Threadgroup width cap for the `(hidden, N)` grid: one thread per
/// `(n, d)`, threadgroups span `hidden` in chunks of this many lanes.
pub const MOE_WEIGHTED_SUM_TG_WIDTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeSumDType {
    F32,
    F16,
    BF16,
}

impl MoeSumDType {
    fn symbol(self) -> &'static str {
        match self {
            Self::F32 => "moe_weighted_sum_float32",
            Self::F16 => "moe_weighted_sum_float16",
            Self::BF16 => "moe_weighted_sum_bfloat16",
        }
    }
}

pub struct MoeWeightedSumKernels {
    library: Library,
    device: Device,
}

impl MoeWeightedSumKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library =
            load_library_from_bytes(device, crate::embedded_metallib!("moe_weighted_sum"))
                .map_err(|e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "load `moe_weighted_sum.metallib`: {e}"
                    ))
                })?;
        Ok(Self {
            library,
            device: device.clone(),
        })
    }

    pub fn build_pipeline(
        &self,
        dtype: MoeSumDType,
        top_k: u32,
        hidden: u32,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        let constants = MTLFunctionConstantValues::new();
        let top_k_i = top_k as i32;
        let hidden_i = hidden as i32;
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new(&top_k_i as *const i32 as *mut c_void).unwrap(),
                MTLDataType::Int,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new(&hidden_i as *const i32 as *mut c_void).unwrap(),
                MTLDataType::Int,
                1,
            );
        }
        let name = NSString::from_str(dtype.symbol());
        let func = self
            .library
            .newFunctionWithName_constantValues_error(&name, &constants)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("{}: {e:?}", dtype.symbol()))
            })?;
        self.device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "{} pipeline: {e:?}",
                    dtype.symbol()
                ))
            })
    }
}

/// CPU reference: `out[n, d] = Σ_k expert[n, k, d] * scores[n, k]`.
pub fn moe_weighted_sum_cpu_f32(
    expert: &[f32],
    scores: &[f32],
    out: &mut [f32],
    rows: usize,
    top_k: usize,
    hidden: usize,
) {
    assert_eq!(expert.len(), rows * top_k * hidden);
    assert_eq!(scores.len(), rows * top_k);
    assert_eq!(out.len(), rows * hidden);
    for n in 0..rows {
        for d in 0..hidden {
            let mut acc = 0.0_f32;
            for k in 0..top_k {
                acc += expert[n * top_k * hidden + k * hidden + d] * scores[n * top_k + k];
            }
            out[n * hidden + d] = acc;
        }
    }
}
