// SPDX-License-Identifier: Apache-2.0
//! The kernels a model runs outside its tape — greedy argmax, the grammar mask and the sampler on
//! the logits its forward leaves, the spec-decode chain advance, and a multi-token-prediction
//! head's chain between its passes — baked like the tape's kernels ([`crate::aot::bake`]) with the
//! model's logits width, dtype, KV block size and hidden width compiled in. The `#[forward]` macro emits one [`OffTapeKernels`] per model
//! (`ScratchyWeights::metal_off_tape`); the worker builds
//! [`ArgmaxKernels`](crate::argmax::ArgmaxKernels),
//! [`GrammarMaskKernels`](crate::grammar_mask::GrammarMaskKernels),
//! [`SamplerKernels`](crate::sampling::SamplerKernels),
//! [`ChainAdvanceKernel`](crate::chain_advance::ChainAdvanceKernel) and
//! [`MtpChainKernel`](crate::mtp_chain::MtpChainKernel) from it.

use objc2_foundation::NSString;
use objc2_metal::{MTLDevice as _, MTLLibrary as _};

use crate::sampling::{CastDtype, SamplerStage};
use crate::shader_cache::{ComputePipelineState, Device, Library, load_library_from_bytes};
use crate::specialized_pipeline_cache::PipelineKey;
use crate::stream::MetalStreamError;
use crate::tape::ids::{BitsetWords, BlockSize, LogitsWidth};
use crate::tape::kernel_constants::{
    ArgmaxConstants, ChainAdvanceConstants, GrammarMaskConstants, MtpChainConstants,
};
use crate::tape::lowered::{BakedKernel, MetalDtype};

/// One model's off-tape kernels: each a [`PipelineKey`] at expansion, a [`BakedKernel`] in the
/// binary.
#[derive(Clone, Copy)]
pub struct OffTape<K> {
    /// Logits per row, compiled into every kernel below.
    pub vocab: LogitsWidth,
    pub argmax: K,
    pub argmax_dual_write: K,
    pub grammar_mask: K,
    /// Every sampler stage, `[telemetry spill compiled in][stage]`; the `sampler-telemetry`
    /// feature picks the row ([`OffTapeKernels::sampler`]).
    pub sampler: [[K; SamplerStage::COUNT]; 2],
    /// The spec-decode chain advance, the model's KV block size compiled in.
    pub chain_advance: K,
    /// A multi-token-prediction head's chain between its passes; `None` for any other model.
    pub mtp_chain: Option<K>,
}

pub type OffTapeKernels = OffTape<BakedKernel>;

impl OffTape<PipelineKey> {
    /// The keys of the off-tape kernels reading logits `vocab` wide, of activation `dtype`, over a
    /// KV cache paged in blocks of `block_size`; `head`: a multi-token-prediction head's chain.
    pub fn keys(
        vocab: LogitsWidth,
        dtype: MetalDtype,
        block_size: BlockSize,
        head: Option<MtpChainConstants>,
    ) -> Self {
        let (argmax, dual_write, grammar_mask, cast) = match dtype {
            MetalDtype::F16 => (
                "argmax_f16",
                "argmax_f16_dual_write",
                "grammar_mask_f16",
                CastDtype::F16,
            ),
            MetalDtype::Bf16 => (
                "argmax_bf16",
                "argmax_bf16_dual_write",
                "grammar_mask_bf16",
                CastDtype::Bf16,
            ),
            MetalDtype::Int4 => panic!("off-tape kernels: int4 logits have no kernel"),
        };
        let argmax_key = |f| PipelineKey::new("argmax", f, ArgmaxConstants { vocab }.into());
        let words_per_row = BitsetWords(crate::grammar_mask::words_per_row(vocab.get()));
        let grammar = GrammarMaskConstants {
            vocab,
            words_per_row,
        };
        Self {
            vocab,
            argmax: argmax_key(argmax),
            argmax_dual_write: argmax_key(dual_write),
            grammar_mask: PipelineKey::new("grammar_mask", grammar_mask, grammar.into()),
            sampler: [false, true].map(|t| SamplerStage::ALL.map(|s| s.key(vocab, cast, t))),
            chain_advance: PipelineKey::new(
                "chain_advance",
                "chain_advance",
                ChainAdvanceConstants { block_size }.into(),
            ),
            mtp_chain: head.map(|c| PipelineKey::new("mtp_chain", "mtp_chain", c.into())),
        }
    }
}

impl<K> OffTape<K> {
    pub fn map<U>(self, mut f: impl FnMut(K) -> U) -> OffTape<U> {
        OffTape {
            vocab: self.vocab,
            argmax: f(self.argmax),
            argmax_dual_write: f(self.argmax_dual_write),
            grammar_mask: f(self.grammar_mask),
            sampler: self.sampler.map(|row| row.map(&mut f)),
            chain_advance: f(self.chain_advance),
            mtp_chain: self.mtp_chain.map(f),
        }
    }
}

impl OffTapeKernels {
    /// The off-tape kernels `ScratchyWeights::metal_off_tape` hands out (typed there as `Any`:
    /// the trait cannot name this crate).
    pub fn of(
        baked: &'static (dyn std::any::Any + Send + Sync),
    ) -> Result<&'static Self, MetalStreamError> {
        baked
            .downcast_ref()
            .ok_or(MetalStreamError::NotOffTapeKernels)
    }

    /// The sampler stages this build dispatches: with the telemetry spill under
    /// `sampler-telemetry`.
    pub fn sampler(&self) -> &[BakedKernel; SamplerStage::COUNT] {
        &self.sampler[usize::from(cfg!(feature = "sampler-telemetry"))]
    }
}

/// A baked kernel's pipeline, held with the library it was built from.
pub struct OffTapePipeline {
    pipeline: ComputePipelineState,
    _library: Library,
}

impl OffTapePipeline {
    pub fn new(device: &Device, kernel: &BakedKernel) -> Result<Self, MetalStreamError> {
        let name = format!("{}.{}", kernel.library, kernel.function);
        let fail =
            |what: &str| MetalStreamError::ShaderCompilationFailed(format!("{name}: {what}"));
        let library = load_library_from_bytes(device, kernel.metallib).map_err(|e| fail(&e))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(kernel.entry))
            .ok_or_else(|| fail("function missing"))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| fail(&format!("pipeline: {e:?}")))?;
        Ok(Self {
            pipeline,
            _library: library,
        })
    }
}

impl std::ops::Deref for OffTapePipeline {
    type Target = ComputePipelineState;
    fn deref(&self) -> &ComputePipelineState {
        &self.pipeline
    }
}
