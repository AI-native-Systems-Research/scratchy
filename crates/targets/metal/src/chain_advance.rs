// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `chain_advance` — Phase 6 K-step chain primitive. Between iterations
//! of the draft K-step chain, advances per-req `positions`,
//! `slot_mapping`, and `seqused_k` in-place on the GPU so the next
//! iter's forward dispatches read the right values without a host
//! roundtrip. Encoded onto the SAME MTL4 compute encoder as the
//! adjacent forward dispatches; Metal's intra-encoder hazard tracking
//! serializes the write→read on the shared buffers. Baked per model with
//! its KV block size, once per rung of its KV cap ladder ([`crate::off_tape`]).

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSize;

use crate::off_tape::{OffTapeKernels, OffTapePipeline};
use crate::shader_cache::Device;
use crate::stream::MetalStreamError;
use crate::tape::ids::MaxBlocksPerSeq;
use crate::tape::kernel_constants::ChainAdvanceConstants;

pub struct ChainAdvanceKernel {
    pub pipeline: OffTapePipeline,
}

impl ChainAdvanceKernel {
    /// The chain advance over block tables `cap` wide — the KV cap rung of the pool it advances,
    /// which its kernel is baked for.
    pub fn new(
        device: &Device,
        kernels: &OffTapeKernels,
        cap: MaxBlocksPerSeq,
    ) -> Result<Self, MetalStreamError> {
        let stride = |k: &&crate::tape::lowered::BakedKernel| {
            (k.constants.iter())
                .find(|c| c.index == ChainAdvanceConstants::STRIDE.get())
                .map(|c| c.bits)
        };
        let kernel = (kernels.chain_advance.iter())
            .find(|k| stride(k) == Some(cap.get()))
            .ok_or_else(|| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "chain_advance: no kernel baked for KV cap rung {}",
                    cap.get()
                ))
            })?;
        Ok(Self {
            pipeline: OffTapePipeline::new(device, kernel)?,
        })
    }
}

/// Encode the chain-advance kernel onto an MTL4 compute encoder.
/// The argument table must be pre-built with addresses for:
///   index 0: positions     [num_reqs]  read+write u32
///   index 1: slot_mapping  [num_reqs]  write u32
///   index 2: seqused_k     [num_reqs]  write u32
///   index 3: block_table   [num_reqs * the baked stride]  read u32
///   index 5: num_reqs             constant uint
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
