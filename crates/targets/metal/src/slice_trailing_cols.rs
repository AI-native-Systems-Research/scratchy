// SPDX-License-Identifier: Apache-2.0
//! `out[n, k] = in[n, axis_size - top_k + k]` for u32 buffers.
//! Used in the MoE router lowering: argpartition produces the full
//! sorted-ascending [N, num_experts] index tensor, and the top-k
//! indices live in the trailing `top_k` columns. MTLBuffer offsets
//! are per-binding (not per-row) so a row-wise trailing slice
//! needs an explicit kernel.

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

pub struct SliceTrailingColsKernels {
    pub u32_pipeline: ComputePipelineState,
    _library: Library,
}

impl SliceTrailingColsKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library =
            load_library_from_bytes(device, crate::embedded_metallib!("slice_trailing_cols"))
                .map_err(|e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "load `slice_trailing_cols.metallib`: {e}"
                    ))
                })?;
        let ns = NSString::from_str("slice_trailing_cols_u32");
        let func = library.newFunctionWithName(&ns).ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed("slice_trailing_cols_u32 fn missing".into())
        })?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "slice_trailing_cols_u32 pipeline: {e:?}"
                ))
            })?;
        Ok(Self {
            u32_pipeline: pipeline,
            _library: library,
        })
    }
}
