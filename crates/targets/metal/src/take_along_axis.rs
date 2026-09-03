// SPDX-License-Identifier: Apache-2.0
//! Rust dispatcher for the 2D-contiguous take_along_axis kernel.
//! Used in the MoE router to pull per-token scores from the
//! softmax'd router probabilities via the top-k indices.

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
pub enum TakeAlongDType {
    F32,
    F16,
    BF16,
}

impl TakeAlongDType {
    fn symbol(self) -> &'static str {
        match self {
            Self::F32 => "take_along_axis_2d_contig_float32",
            Self::F16 => "take_along_axis_2d_contig_float16",
            Self::BF16 => "take_along_axis_2d_contig_bfloat16",
        }
    }
}

pub struct TakeAlongAxisKernels {
    pub f32: ComputePipelineState,
    pub f16: ComputePipelineState,
    pub bf16: ComputePipelineState,
    _library: Library,
}

impl TakeAlongAxisKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("take_along_axis"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `take_along_axis.metallib`: {e}"
                ))
            })?;
        let f32 = build(device, &library, TakeAlongDType::F32.symbol())?;
        let f16 = build(device, &library, TakeAlongDType::F16.symbol())?;
        let bf16 = build(device, &library, TakeAlongDType::BF16.symbol())?;
        Ok(Self {
            f32,
            f16,
            bf16,
            _library: library,
        })
    }

    pub fn pipeline_for(&self, dtype: TakeAlongDType) -> &ComputePipelineState {
        match dtype {
            TakeAlongDType::F32 => &self.f32,
            TakeAlongDType::F16 => &self.f16,
            TakeAlongDType::BF16 => &self.bf16,
        }
    }
}

fn build(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns = NSString::from_str(name);
    let func = library
        .newFunctionWithName(&ns)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
}
