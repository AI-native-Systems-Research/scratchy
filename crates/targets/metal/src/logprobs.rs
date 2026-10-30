// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The OpenAI `logprobs` / `top_logprobs` of emitted tokens (`shaders/logprobs.metal`), baked per
//! model ([`crate::off_tape`]): for each scored logits row, the log-softmax at the token emitted
//! from it, that token's rank, and the row's [`TOP`] most likely tokens — what cuda's
//! `log_softmax_topk_gather` gives its worker. Encoded after the step's tokens are final, onto the
//! forward's own encoder.

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CommandEncoder as _, MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLBuffer as _,
    MTLSize, MTLStages,
};

use crate::mtl4_dispatch::{Buffer, Device};
use crate::off_tape::{OffTapeKernels, OffTapePipeline};
use crate::stream::MetalStreamError;

/// The top tokens scored for every entry: OpenAI's `top_logprobs` maximum. A request keeps the
/// first as many as it asked for.
pub const TOP: usize = 20;
/// An entry's slots: its emitted token, then the top tokens.
pub const SLOTS: usize = TOP + 1;
/// Threads a threadgroup: the kernel's `LOGPROBS_THREADS`.
pub const THREADS: usize = 1024;

pub struct LogprobsKernels {
    pipeline: OffTapePipeline,
}

impl LogprobsKernels {
    pub fn new(device: &Device, kernels: &OffTapeKernels) -> Result<Self, MetalStreamError> {
        Ok(Self {
            pipeline: OffTapePipeline::new(device, &kernels.logprobs)?,
        })
    }
}

/// A step's scored entries in one buffer, as byte offsets: the count, each entry's logits row,
/// then each entry's [`SLOTS`] log-probabilities, token ids and ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogprobsLayout {
    pub entries: usize,
    pub count: u64,
    pub rows: u64,
    pub logprob: u64,
    pub token: u64,
    pub rank: u64,
    pub total: u64,
}

impl LogprobsLayout {
    pub fn of(entries: usize) -> Self {
        let section = |n: usize| ((n * 4) as u64).next_multiple_of(16);
        let count = 0;
        let rows = count + section(1);
        let logprob = rows + section(entries);
        let token = logprob + section(entries * SLOTS);
        let rank = token + section(entries * SLOTS);
        Self {
            entries,
            count,
            rows,
            logprob,
            token,
            rank,
            total: rank + section(entries * SLOTS),
        }
    }

    /// The argument table's addresses in `buffer` at `base`, with the logits rows and the tokens
    /// emitted from them (one a row): the kernel's bindings 0-6.
    pub fn addresses(&self, base: u64, logits: u64, tokens: u64) -> [u64; 7] {
        [
            logits,
            base + self.rows,
            tokens,
            base + self.count,
            base + self.logprob,
            base + self.token,
            base + self.rank,
        ]
    }

    /// Write the count and the logits row of each entry into `buffer`.
    pub fn write_rows(&self, buffer: &Buffer, rows: &[u32]) {
        assert_eq!(rows.len(), self.entries, "an entry a row");
        assert!(
            buffer.length() as u64 >= self.total,
            "the buffer holds the layout"
        );
        // SAFETY: `buffer` is host-visible and holds `total` bytes (checked above), and no command
        // buffer reads it while the host writes (the step's command buffer is encoded after).
        unsafe {
            let at = |offset: u64| buffer.contents().as_ptr().cast::<u8>().add(offset as usize);
            at(self.count).cast::<u32>().write(rows.len() as u32);
            std::ptr::copy_nonoverlapping(rows.as_ptr(), at(self.rows).cast::<u32>(), rows.len());
        }
    }

    /// Each entry's scores, once the command buffer that wrote them is done.
    pub fn read(&self, buffer: &Buffer) -> Vec<Scored> {
        let n = self.entries * SLOTS;
        // SAFETY: the buffer holds the layout, written by a command buffer that is done.
        let (logprob, token, rank) = unsafe {
            let at = |offset: u64| buffer.contents().as_ptr().cast::<u8>().add(offset as usize);
            (
                std::slice::from_raw_parts(at(self.logprob).cast::<f32>(), n),
                std::slice::from_raw_parts(at(self.token).cast::<u32>(), n),
                std::slice::from_raw_parts(at(self.rank).cast::<u32>(), n),
            )
        };
        (0..self.entries)
            .map(|e| {
                let slot = |s: usize| TokenScore {
                    token: token[e * SLOTS + s],
                    logprob: logprob[e * SLOTS + s],
                    rank: rank[e * SLOTS + s],
                };
                Scored {
                    emitted: slot(0),
                    top: (1..SLOTS).map(slot).collect(),
                }
            })
            .collect()
    }
}

/// A token's score in its row: its log-probability and 1-based rank.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenScore {
    pub token: u32,
    pub logprob: f32,
    pub rank: u32,
}

/// One entry's scores: the token emitted from its row, and the row's [`TOP`] tokens, most likely
/// first.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub emitted: TokenScore,
    pub top: Vec<TokenScore>,
}

/// Encode the scoring of `entries` entries onto `encoder`, whose argument table holds
/// [`LogprobsLayout::addresses`]. A Device-visible barrier first: the logits and the tokens are
/// the step's last writes.
pub fn encode(
    kernels: &LogprobsKernels,
    encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    entries: usize,
) {
    if entries == 0 {
        return;
    }
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(&kernels.pipeline);
    encoder.setArgumentTable(Some(table));
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: entries,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
}
