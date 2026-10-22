// SPDX-License-Identifier: Apache-2.0

//! `mtp_chain` — between a multi-token-prediction head's passes over one speculative step, the
//! next pass's inputs, picked on the device (`shaders/mtp_chain.metal`): the passes run in the
//! step's command buffer with no host round trip between them. Baked per head with its hidden
//! width and drafts a step, once after each pass ([`crate::off_tape`]).

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSize;

use crate::off_tape::{OffTapeKernels, OffTapePipeline};
use crate::shader_cache::Device;
use crate::stream::MetalStreamError;
use crate::tape::ids::HeadPass;
use crate::tape::kernel_constants::MTP_CHAIN_THREADS;

pub struct MtpChainKernel {
    /// The chain after each pass, in order.
    pub after: Vec<OffTapePipeline>,
}

impl MtpChainKernel {
    /// `None` for a model that is no multi-token-prediction head.
    pub fn new(
        device: &Device,
        kernels: &OffTapeKernels,
    ) -> Result<Option<Self>, MetalStreamError> {
        (kernels.mtp_chain)
            .map(|after| {
                let after = after.iter().map(|k| OffTapePipeline::new(device, k));
                Ok(Self {
                    after: after.collect::<Result<_, _>>()?,
                })
            })
            .transpose()
    }
}

/// Encode the chain after head pass `depth` onto `encoder`, one threadgroup for each of
/// `num_seqs` drafting sequences, behind a barrier on everything before it (the pass, its argmax).
pub fn encode_mtp_chain_into_mtl4(
    kernel: &MtpChainKernel,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    num_seqs: u32,
    depth: HeadPass,
) -> Result<(), MetalStreamError> {
    let pipeline = (depth.get().checked_sub(1))
        .and_then(|i| kernel.after.get(i as usize))
        .ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "mtp_chain: no kernel baked after pass {} of {}",
                depth.get(),
                kernel.after.len()
            ))
        })?;
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(pipeline);
    encoder.setArgumentTable(Some(arg_table));
    let threadgroups = MTLSize {
        width: num_seqs as usize,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: MTP_CHAIN_THREADS as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
    Ok(())
}
