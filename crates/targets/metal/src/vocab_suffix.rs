// SPDX-License-Identifier: Apache-2.0
//! A multi-token-prediction head's reach past the vocabulary prefix it drafts over
//! ([`VocabSuffix`]): after a pass whose lm_head wrote its prefix's logits, a matvec of the pass's
//! rows over its target's lm_head rows past the prefix, and the argmax across both parts
//! (`argmax_split`), so the step drafts over the whole vocabulary.

use std::ops::Deref;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CommandEncoder as _, MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLBuffer as _,
    MTLSize, MTLStages,
};

use crate::argmax::ARGMAX_DEFAULT_TG_SIZE;
use crate::mtl4_dispatch::{Buffer, Device, build_arg_table};
use crate::off_tape::VocabSuffix;
use crate::residency::Pinned;
use crate::shader_cache::ComputePipelineState;
use crate::tape::quantized::qmv_dispatch_shape;

/// A part of a row-major MLX-affine lm_head: its buffer, and the byte offset of its row 0 in it.
pub struct LmHeadPart {
    pub buffer: Buffer,
    pub offset: u64,
}

impl LmHeadPart {
    /// The address of its row `row`, `row_bytes` a row.
    fn row(&self, row: u32, row_bytes: usize) -> u64 {
        self.buffer.gpuAddress() + self.offset + (row as usize * row_bytes) as u64
    }
}

/// A target's whole MLX-affine lm_head: its codes, scales and biases.
pub struct LmHead {
    pub codes: LmHeadPart,
    pub scales: LmHeadPart,
    pub biases: LmHeadPart,
}

/// What a head pass's argmax reads and writes: its `batch` rows of hidden state at `hidden`, the
/// logits its lm_head wrote from them at `prefix_logits`, and its picks at `picked`, its batch
/// read at `batch_const`.
#[derive(Clone, Copy, Debug)]
pub struct PassPick {
    pub hidden: u64,
    pub prefix_logits: u64,
    pub picked: u64,
    pub batch_const: u64,
    pub batch: usize,
}

/// A [`VocabSuffix`] head's matvec over its target's lm_head rows past the prefix and the argmax
/// across both parts, ready to encode after a pass. `P`: a pipeline (a baked kernel's).
pub struct SuffixDraft<P> {
    suffix: VocabSuffix,
    /// Held for the addresses below.
    _lm_head: LmHead,
    /// The address of row `prefix` of the lm_head's codes, scales and biases.
    codes: u64,
    scales: u64,
    biases: u64,
    /// The suffix's logits, `verify_rows` rows of `rows` each.
    logits: Pinned,
    /// The matvec of `m` rows at `m - 1`, with its grid and threadgroup.
    qmv: Vec<(P, MTLSize, MTLSize)>,
    argmax: P,
}

impl<P: Deref<Target = ComputePipelineState>> SuffixDraft<P> {
    /// `qmv`: the matvec of `m` rows at `m - 1`, `suffix.verify_rows` of them, reading `lm_head`'s
    /// codes the way its binding stores them; `argmax`: the argmax across both parts; `logits`:
    /// room, resident, for `verify_rows` rows of the suffix's logits.
    pub fn new(
        suffix: VocabSuffix,
        lm_head: LmHead,
        qmv: Vec<P>,
        argmax: P,
        logits: Pinned,
    ) -> Result<Self, String> {
        if qmv.len() != suffix.verify_rows as usize {
            return Err(format!(
                "{} suffix matvecs for a head verifying {} rows",
                qmv.len(),
                suffix.verify_rows
            ));
        }
        let logits_bytes = (suffix.verify_rows * suffix.rows) as usize * size_of::<u16>();
        if logits.length() < logits_bytes {
            return Err(format!(
                "{} bytes of suffix logits for {logits_bytes}",
                logits.length()
            ));
        }
        let size = |(width, height, depth): (u32, u32, u32)| MTLSize {
            width: width as usize,
            height: height as usize,
            depth: depth as usize,
        };
        let qmv = (qmv.into_iter().zip(1u32..))
            .map(|(pso, m)| {
                let (grid, threads) = qmv_dispatch_shape(suffix.qmv_kernel(m), m, suffix.rows, 1);
                (pso, size(grid), size(threads))
            })
            .collect();
        let (code_bytes, scale_bytes) = suffix.row_bytes();
        Ok(Self {
            codes: lm_head.codes.row(suffix.prefix, code_bytes),
            scales: lm_head.scales.row(suffix.prefix, scale_bytes),
            biases: lm_head.biases.row(suffix.prefix, scale_bytes),
            _lm_head: lm_head,
            suffix,
            logits,
            qmv,
            argmax,
        })
    }

    pub fn suffix(&self) -> VocabSuffix {
        self.suffix
    }

    /// After a pass ([`PassPick`]): the matvec of its rows over the suffix, then the argmax across
    /// the prefix's logits and the suffix's into its picks, each behind a fence on what it reads.
    pub fn encode(
        &self,
        enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        device: &Device,
        pick: PassPick,
    ) -> Result<(), String> {
        let (pso, grid, threads) = (pick.batch.checked_sub(1))
            .and_then(|i| self.qmv.get(i))
            .ok_or_else(|| {
                format!(
                    "{} rows, past the head's {} verify rows",
                    pick.batch, self.suffix.verify_rows
                )
            })?;
        let fence = || {
            enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            )
        };
        let table = |addresses: &[u64]| {
            let bindings: Vec<(u64, usize)> = addresses.iter().copied().zip(0..).collect();
            build_arg_table(device, &bindings, addresses[0])
        };
        let logits = self.logits.gpuAddress();
        let qmv_table = table(&[self.codes, self.scales, self.biases, pick.hidden, logits]);
        fence();
        enc.setComputePipelineState(pso);
        enc.setArgumentTable(Some(&qmv_table));
        enc.dispatchThreadgroups_threadsPerThreadgroup(*grid, *threads);
        let argmax_table = table(&[pick.prefix_logits, pick.picked, pick.batch_const, logits]);
        fence();
        enc.setComputePipelineState(&self.argmax);
        enc.setArgumentTable(Some(&argmax_table));
        let one = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            one(pick.batch),
            one(ARGMAX_DEFAULT_TG_SIZE),
        );
        Ok(())
    }
}
