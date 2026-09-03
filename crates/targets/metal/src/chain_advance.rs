// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `chain_advance` — Phase 6 K-step chain primitive. Between iterations
//! of the draft K-step chain, advances per-req `positions`,
//! `slot_mapping`, and `seqused_k` in-place on the GPU so the next
//! iter's forward dispatches read the right values without a host
//! roundtrip. Encoded onto the SAME MTL4 compute encoder as the
//! adjacent forward dispatches; Metal's intra-encoder hazard tracking
//! serializes the write→read on the shared buffers.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

pub struct ChainAdvanceKernel {
    pub pipeline: ComputePipelineState,
    _library: Library,
}

impl ChainAdvanceKernel {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("chain_advance"))
            .map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!("load `chain_advance.metallib`: {e}"))
        })?;
        let ns_name = NSString::from_str("chain_advance");
        let function = library.newFunctionWithName(&ns_name).ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed("chain_advance fn missing".into())
        })?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "newComputePipelineStateWithFunction(chain_advance): {e:?}"
                ))
            })?;
        Ok(Self {
            pipeline,
            _library: library,
        })
    }
}

/// Encode the chain-advance kernel onto an MTL4 compute encoder.
/// The argument table must be pre-built with addresses for:
///   index 0: positions     [num_reqs]  read+write u32
///   index 1: slot_mapping  [num_reqs]  write u32
///   index 2: seqused_k     [num_reqs]  write u32
///   index 3: block_table   [num_reqs * block_table_stride]  read u32
///   index 4: block_size           constant uint (8-byte buffer with u32 at offset 0)
///   index 5: block_table_stride   constant uint (8-byte buffer with u32 at offset 0)
///   index 6: num_reqs             constant uint (8-byte buffer with u32 at offset 0)
///
/// A barrier is inserted before the dispatch so the previous iter's
/// argmax_dual_write (which wrote `runtime.input_ids`) is visible to
/// any reader, and so this dispatch's writes to `positions`/
/// `slot_mapping`/`seqused_k` happen-after any in-flight forward
/// dispatches that read them.
pub fn encode_chain_advance_into_mtl4(
    kernel: &ChainAdvanceKernel,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    num_reqs: u32,
) -> Result<(), MetalStreamError> {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    if num_reqs == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(
            "chain_advance encode: num_reqs=0".into(),
        ));
    }
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(&kernel.pipeline);
    encoder.setArgumentTable(Some(arg_table));
    // One threadgroup, num_reqs threads. K-step batches have
    // num_reqs <= max_num_seqs which is typically <= 256. Single
    // threadgroup is fine.
    let threadgroups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: num_reqs as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    Ok(())
}
