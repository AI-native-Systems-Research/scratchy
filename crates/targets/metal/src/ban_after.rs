// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `ban_after` — a two-token bad word's ban on the device + Rust dispatcher, baked per model
//! ([`crate::off_tape`]).
//!
//! A bad word `[after, banned]` bans `banned` at a sampled row only when the token before that row
//! is `after`. That token is the row's own input token — a decode row's last token, a verify row's
//! draft before it — so the kernel reads it from the forward's `input_ids` rather than the host
//! reading it back: a step whose input tokens are still on the device (a deferred one, or one
//! behind a pipelined speculative step) bans as a synchronous one does. One job a sampled row and
//! bad word, `(logits row, input index, after, banned)`; dispatched onto the forward's encoder
//! before argmax and the sampler, as the grammar mask is.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSize;

use crate::off_tape::{OffTapeKernels, OffTapePipeline};
use crate::stream::MetalStreamError;

/// Threads per threadgroup; one thread a job.
pub const BAN_AFTER_TG_SIZE: usize = 64;

/// `u32` words a job takes: its logits row, its input index, the token before, the banned token.
pub const JOB_WORDS: usize = 4;

pub struct BanAfterKernel {
    pub ban: OffTapePipeline,
}

impl BanAfterKernel {
    pub fn new(
        device: &crate::shader_cache::Device,
        kernels: &OffTapeKernels,
    ) -> Result<Self, MetalStreamError> {
        Ok(Self {
            ban: OffTapePipeline::new(device, &kernels.ban_after)?,
        })
    }
}

/// `after` of a job that bans whatever the token before it (`BAN_ALWAYS` in the shader).
pub const ALWAYS: u32 = u32::MAX;

/// One job: ban `banned` at logits row `row` when `input_ids[input]` is `after`, or always when
/// `after` is [`ALWAYS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BanJob {
    pub row: u32,
    pub input: u32,
    pub after: u32,
    pub banned: u32,
}

impl BanJob {
    /// Its words, in the kernel's order.
    pub fn words(self) -> [u32; JOB_WORDS] {
        [self.row, self.input, self.after, self.banned]
    }
}

/// MTL4 encoder-tail ban dispatcher.
///
/// The argument table must be pre-built with:
/// - index 0: logits GPU address     (`[total_n, vocab]`, in/out)
/// - index 1: input_ids GPU address  (the forward's runtime `input_ids`)
/// - index 2: jobs GPU address       (`[1 + count * JOB_WORDS]` u32: `count`, then the jobs)
///
/// `count` sizes the grid; the kernel reads it from the jobs buffer.
pub fn encode_ban_after_into_mtl4(
    kernel: &BanAfterKernel,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    count: u32,
) -> Result<(), MetalStreamError> {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    if count == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(
            "ban_after encode: count=0".into(),
        ));
    }
    // Barrier before: the kernel reads and writes the lm_head output (the forward's last write)
    // and reads `input_ids`, which the step's head wrote from an earlier step's tokens. Argmax's
    // own pre-barrier then orders this write before its read. See the grammar mask's.
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(&kernel.ban);
    encoder.setArgumentTable(Some(arg_table));
    let threadgroups = MTLSize {
        width: (count as usize).div_ceil(BAN_AFTER_TG_SIZE),
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: BAN_AFTER_TG_SIZE,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    Ok(())
}
