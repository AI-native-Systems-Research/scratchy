// SPDX-License-Identifier: Apache-2.0

//! `mtp_chain` — between a multi-token-prediction head's passes over one speculative step, the
//! next pass's inputs, picked on the device (`shaders/mtp_chain.metal`): the passes run in the
//! step's command buffer with no host round trip between them. Baked per head with its hidden
//! width and drafts a step ([`crate::off_tape`]).

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSize;

use crate::off_tape::{OffTapeKernels, OffTapePipeline};
use crate::shader_cache::Device;
use crate::stream::MetalStreamError;

pub struct MtpChainKernel {
    pub pipeline: OffTapePipeline,
}

impl MtpChainKernel {
    /// `None` for a model that is no multi-token-prediction head.
    pub fn new(
        device: &Device,
        kernels: &OffTapeKernels,
    ) -> Result<Option<Self>, MetalStreamError> {
        (kernels.mtp_chain.as_ref())
            .map(|k| {
                Ok(Self {
                    pipeline: OffTapePipeline::new(device, k)?,
                })
            })
            .transpose()
    }
}

/// Encode the chain after a head pass onto `encoder`, one threadgroup for each of `num_seqs`
/// drafting sequences, behind a barrier on everything before it (the pass, its argmax).
pub fn encode_mtp_chain_into_mtl4(
    kernel: &MtpChainKernel,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    num_seqs: u32,
) {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(&kernel.pipeline);
    encoder.setArgumentTable(Some(arg_table));
    let threadgroups = MTLSize {
        width: num_seqs as usize,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
}
