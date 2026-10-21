// SPDX-License-Identifier: Apache-2.0
//! The kernels a model runs outside its tape — greedy argmax, the grammar mask and the sampler on
//! the logits its forward leaves, the spec-decode chain advance, and a multi-token-prediction
//! head's chain between its passes — baked like the tape's kernels ([`crate::aot::bake`]) with the
//! model's logits width, dtype, KV block size and hidden width compiled in. The `#[forward]` macro emits one [`OffTapeKernels`] per model
//! (`ScratchyWeights::metal_off_tape`); the worker builds
//! [`ArgmaxKernels`](crate::argmax::ArgmaxKernels),
//! [`GrammarMaskKernels`](crate::grammar_mask::GrammarMaskKernels),
//! [`SamplerKernels`](crate::sampling::SamplerKernels),
//! [`ChainAdvanceKernel`](crate::chain_advance::ChainAdvanceKernel),
//! [`MtpChainKernel`](crate::mtp_chain::MtpChainKernel) and
//! [`SelectRowsKernel`](crate::select_rows::SelectRowsKernel) from it.

use objc2_foundation::NSString;
use objc2_metal::{MTLDevice as _, MTLLibrary as _};

use crate::sampling::{CastDtype, SamplerStage};
use crate::shader_cache::{ComputePipelineState, Device, Library, load_library_from_bytes};
use crate::specialized_pipeline_cache::{ConstantValue, PipelineKey};
use crate::stream::MetalStreamError;
use crate::tape::ids::{BitsetWords, BlockSize, KDimI32, LogitsWidth, MDimI32, NDimI32};
use crate::tape::kernel_constants::{
    AffineCodes, AffineQmvConstants, AffineQmvWideConstants, ArgmaxConstants,
    ChainAdvanceConstants, GrammarMaskConstants, MtpChainConstants,
};
use crate::tape::lowered::{BakedKernel, MetalDtype};
use crate::tape::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, pick_qmv_kernel, pick_qmv_kernel_wide,
    qmv_kernel_static_name,
};

/// A multi-token-prediction head drafting over the first `prefix` token ids of its target's
/// vocabulary, whose lm_head's `rows` rows past them are `hidden` wide: `bits`-bit MLX-affine
/// codes, a `scale`-typed scale and bias a `group` columns. A pass's lm_head reads at most
/// `verify_rows` rows: the head's `spec_verify_rows`, the rows a step's verify holds, which bound
/// its drafting rows (`DraftBudget`). A step drafts over the whole vocabulary while one of its
/// drafting replies has a token past the prefix among its last `window` (`spec_wide_window`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VocabSuffix {
    pub prefix: u32,
    pub rows: u32,
    pub hidden: u32,
    pub bits: u32,
    pub group: u32,
    pub scale: ScaleDtype,
    pub verify_rows: u32,
    pub window: u32,
}

impl VocabSuffix {
    /// The matvec of `m` rows (`1..=verify_rows`) over the suffix.
    pub fn qmv_kernel(self, m: u32) -> QmvKernel {
        match m {
            1 => pick_qmv_kernel(self.rows, self.hidden, self.bits),
            _ => pick_qmv_kernel_wide(self.rows, self.hidden, self.bits, m, true),
        }
    }

    /// The bytes of one lm_head row of its codes, and of its scales (and biases).
    pub fn row_bytes(self) -> (usize, usize) {
        let codes = self.hidden as usize * self.bits as usize / 8;
        let scales = (self.hidden / self.group) as usize * self.scale.as_dtype().size_bytes();
        (codes, scales)
    }

    /// The keys of its kernels for `dtype` activations: a matvec for each row count and way of
    /// reading codes, and the argmax across the prefix's logits and the suffix's.
    pub fn kernel_keys(self, dtype: MetalDtype) -> SuffixKernels<PipelineKey, Vec<PipelineKey>> {
        let (dequant, argmax_split) = match dtype {
            MetalDtype::F16 => (DequantDtype::F16, "argmax_split_f16"),
            MetalDtype::Bf16 => (DequantDtype::Bf16, "argmax_split_bf16"),
            MetalDtype::Int4 => panic!("a vocabulary suffix: int4 logits have no kernel"),
        };
        let rows = |codes| {
            (1..=self.verify_rows)
                .map(|m| self.qmv_key(dequant, codes, m))
                .collect()
        };
        SuffixKernels {
            suffix: self,
            qmv: ByCodes {
                as_written: rows(AffineCodes::AsWritten),
                offset8: rows(AffineCodes::Offset8),
            },
            argmax: PipelineKey::new(
                "argmax",
                argmax_split,
                vec![
                    ConstantValue::uint(0, self.prefix),
                    ConstantValue::uint(1, self.rows),
                ],
            ),
        }
    }

    /// Its key, `dtype` activations, its codes read as `codes`.
    fn qmv_key(self, dtype: DequantDtype, codes: AffineCodes, m: u32) -> PipelineKey {
        let kernel = self.qmv_kernel(m);
        let name = qmv_kernel_static_name(kernel, dtype, self.scale, self.bits, self.group);
        let (k, n) = (KDimI32(self.hidden as i32), NDimI32(self.rows as i32));
        let constants: Vec<ConstantValue> = match kernel {
            QmvKernel::Wide { .. } => AffineQmvWideConstants {
                k,
                n,
                m: MDimI32(m as i32),
                codes,
            }
            .into(),
            _ => AffineQmvConstants { k, n, codes }.into(),
        };
        PipelineKey::new("quantized_qmv", name, constants)
    }
}

/// One of each way a device's kernels read MLX-affine codes ([`AffineCodes`]).
#[derive(Clone, Copy)]
pub struct ByCodes<T> {
    pub as_written: T,
    pub offset8: T,
}

impl<T> ByCodes<T> {
    pub fn get(&self, codes: AffineCodes) -> &T {
        match codes {
            AffineCodes::AsWritten => &self.as_written,
            AffineCodes::Offset8 => &self.offset8,
        }
    }

    fn map<U>(self, mut f: impl FnMut(T) -> U) -> ByCodes<U> {
        ByCodes {
            as_written: f(self.as_written),
            offset8: f(self.offset8),
        }
    }
}

/// A [`VocabSuffix`] head's kernels past its prefix: the matvec of a pass's `m` rows over the
/// suffix at `qmv.get(codes)[m - 1]` — its codes read as the device reads them, `verify_rows`
/// of them (`L`: a `Vec` at expansion, a slice in the binary) — and the argmax across the
/// prefix's logits and the suffix's.
#[derive(Clone, Copy)]
pub struct SuffixKernels<K, L> {
    pub suffix: VocabSuffix,
    pub qmv: ByCodes<L>,
    pub argmax: K,
}

/// One model's off-tape kernels: each a [`PipelineKey`] at expansion, a [`BakedKernel`] in the
/// binary (`L`: a list of them, a `Vec` at expansion, a slice in the binary).
#[derive(Clone, Copy)]
pub struct OffTape<K, L> {
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
    /// A drafting model's pick of a step's inputs by the step before it; `None` for any other.
    pub select_rows: Option<K>,
    /// A multi-token-prediction head drafting over a prefix of its target's vocabulary: its
    /// kernels past the prefix; `None` for any other model.
    pub vocab_suffix: Option<SuffixKernels<K, L>>,
}

pub type OffTapeKernels = OffTape<BakedKernel, &'static [BakedKernel]>;

impl OffTape<PipelineKey, Vec<PipelineKey>> {
    /// The keys of the off-tape kernels reading logits `vocab` wide, of activation `dtype`, over a
    /// KV cache paged in blocks of `block_size`; `head`: a multi-token-prediction head's chain;
    /// `drafts`: the model drafts; `vocab_suffix`: a head drafting over a prefix of its target's
    /// vocabulary, the rest of it.
    pub fn keys(
        vocab: LogitsWidth,
        dtype: MetalDtype,
        block_size: BlockSize,
        head: Option<MtpChainConstants>,
        drafts: bool,
        vocab_suffix: Option<VocabSuffix>,
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
        let vocab_suffix = vocab_suffix.map(|suffix| {
            assert_eq!(
                suffix.prefix,
                vocab.get(),
                "a head drafting over a prefix of its target's vocabulary reads logits that wide"
            );
            suffix.kernel_keys(dtype)
        });
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
            select_rows: drafts.then(|| PipelineKey::new("select_rows", "select_rows", Vec::new())),
            vocab_suffix,
        }
    }
}

impl<K> OffTape<K, Vec<K>> {
    pub fn map<U>(self, mut f: impl FnMut(K) -> U) -> OffTape<U, Vec<U>> {
        OffTape {
            vocab: self.vocab,
            argmax: f(self.argmax),
            argmax_dual_write: f(self.argmax_dual_write),
            grammar_mask: f(self.grammar_mask),
            sampler: self.sampler.map(|row| row.map(&mut f)),
            chain_advance: f(self.chain_advance),
            mtp_chain: self.mtp_chain.map(&mut f),
            select_rows: self.select_rows.map(&mut f),
            vocab_suffix: self.vocab_suffix.map(|s| SuffixKernels {
                suffix: s.suffix,
                qmv: s.qmv.map(|rows| rows.into_iter().map(&mut f).collect()),
                argmax: f(s.argmax),
            }),
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
