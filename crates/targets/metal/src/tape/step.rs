// SPDX-License-Identifier: Apache-2.0
//! Metal's step records — the rows [`lower_subtile_tape_to_metal`] lowers.
//!
//! One [`MetalStep`] per command-emitting step, one variant per kind metal lowers, every field
//! typed: an arena slot cannot stand where a layer goes, nor a head count where a width goes.
//! [`StepRow::Loop`] rolls the rows after it into a layer loop.
//!
//! [`lower_subtile_tape_to_metal`]: super::lowering::lower_subtile_tape_to_metal

pub use super::ids::{
    AffineBits, AffineGroupSize, ArenaSlotIdx, BodyLen, HiddenSize, IntermediateSize, KDim,
    LayerId, LayerStride, LoopIters, NDim, NumExperts, QmvBatchLimit, RowsDivisor, RowsPerToken,
    SourceIx, TopK,
};
pub use scratchy_ir::{BiasStorage, KvOffset, KvOffsets};
use scratchy_subtile::handoff::WeightKind;
/// A KV writer's geometry class on hybrid sliding/global arches is its layer's [`AttnMask`]
/// (`Causal` takes the `GLOBAL_*` consts, `SlidingWindow` the base ones; on uniform arches the two
/// are equal), and a rope pairs a head's lanes as its [`RopeFormTag`] says — the tape's own types.
pub use scratchy_subtile::subtile_ir::{
    AttnMask, ExpertBundle, ExpertProj, GatedAct, KvOperand, RopeFormTag, RotatedRows, RouterBundle,
};

use super::ids::ArenaSlotIdx as Slot;
use super::lowered::RuntimeGate;

/// An `f32` field compared by bit pattern: two rows are equal iff they bake the same constant.
macro_rules! f32_bits {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name(pub f32);
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0.to_bits() == other.0.to_bits()
            }
        }
        impl Eq for $name {}
    )*};
}

f32_bits!(
    /// A norm's gain offset: `rmsnorm(x, w + offset)` (Gemma's zero-centred `1.0`).
    GainOffset,
    /// A scalar multiplier: `x * scale` (granite's embedding and logit scaling).
    Scale,
    /// A norm's epsilon.
    Eps,
);

/// A region of the MoE op scratch — where a value the tape keeps off the arena lives
/// (`op_abi::moe_write`). Where each region sits is the bake's: it depends on the bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRegion {
    RouterNormed,
    RouterLogits,
    SortedExperts,
    TopKIndices,
    TopKScores,
    /// The token rows sorted by expert (grouped bakes only).
    SortedRows,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    /// The projected pair rows in token order.
    TokenRows,
}

/// The scores a MoE softmax normalizes: every expert's, or the chosen top-k's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeScores {
    Router,
    TopK,
}

/// Where a MoE step's operand rows are: the arena's token rows (read through the expert sort
/// when a bake groups), or a scratch region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRows {
    Tokens(Slot),
    Scratch(MoeRegion),
}

/// Whether a router normalizes its input into the scratch first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterInput {
    PreNormed,
    Raw,
}

/// A MoE block's geometry, carried by each of its steps: the scratch layout every step binds
/// is a function of it and the bake's bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeBlock {
    pub experts: NumExperts,
    pub top_k: TopK,
    pub inter: IntermediateSize,
    pub hidden: HiddenSize,
    pub router: RouterBundle,
    pub bundle: ExpertBundle,
    pub input: RouterInput,
    /// The experts' packed storage, as the block's gate projection op declares it: what decides
    /// whether a grouped bake's expert GEMMs run W4A8 (every step of the block must agree).
    pub quant: ExpertQuant,
}

/// An expert bank's MLX-affine group size and code width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertQuant {
    pub group_size: AffineGroupSize,
    pub bits: AffineBits,
}

/// An expert projection's packed width: the MoE op's own (one width for all three
/// projections), or the width the model's quantization declares for this projection.
///
/// ⚠️ THE TWO NEVER COMPARE EQUAL, and that is the baseline's roll behaviour: the rolled and
/// unrolled layouts carry declared widths, a peeled candidate the op's (`MoeBits`), so a peeled
/// Qwen-MoE roll never proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertWidth {
    OpUniform(AffineBits),
    Declared(AffineBits),
}

impl ExpertWidth {
    pub fn bits(self) -> AffineBits {
        match self {
            Self::OpUniform(b) | Self::Declared(b) => b,
        }
    }
}

/// The runtime index buffer an `EmbeddingGather` permutes rows by (Qwen2.5-VL windowing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherIndices {
    /// `vision_window_index`: natural order into window order.
    WindowIndex,
    /// `vision_reverse_indices`: back to natural order.
    ReverseIndices,
}

/// The segment boundaries a `VarlenAttention` attends within.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuSeqlens {
    /// The step's per-sequence (per-image) `cu_seqlens_q`.
    Batch,
    /// Qwen2.5-VL full-attention layers: per-image boundaries.
    VisionFull,
    /// Qwen2.5-VL windowed layers: per-window boundaries.
    VisionWindow,
}

/// One command-emitting step. Slot operands are arena colours; `LayerId` is the step's own layer
/// (a rolled body's rows name iteration 0's). Field order per kind: inputs, outputs, layer, shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalStep {
    /// `(out)`: token embedding into the hidden.
    Embed(Slot),
    /// `(out, group_size, bits)`: MLX-affine quantized token embedding.
    AffineEmbed(Slot, AffineGroupSize, AffineBits),
    /// `(slot)`: projected vision rows spliced into the embedded hidden, in place.
    SpliceMmEmbeds(Slot),
    /// `(in, out, rows_mult, rows_div, cols)`: a view — no kernel runs; later rows see
    /// `cols` as the width and `num_tokens / rows_div` rows.
    Reshape(Slot, Slot, RowsPerToken, RowsDivisor, HiddenSize),
    /// `(in, out, layer, width, rows_per_token)`.
    RmsNorm(Slot, Slot, LayerId, HiddenSize, RowsPerToken),
    /// `(in, out, layer, offset, width, rows_per_token)`: `rmsnorm(x, w + offset)`.
    ScalarOffsetRmsNorm(Slot, Slot, LayerId, GainOffset, HiddenSize, RowsPerToken),
    /// `(in, out, width, rows_per_token)`: unit-gain RMSNorm.
    RmsNormUnit(Slot, Slot, HiddenSize, RowsPerToken),
    /// `(in, out, layer)`: centred (mean-subtracted) RMSNorm.
    MeanSubRmsNorm(Slot, Slot, LayerId),
    /// `(in, out, layer)`: LayerNorm with bias.
    MeanSubRmsNormBiasAdd(Slot, Slot, LayerId),
    /// `(delta, residual, layer, width, rows_per_token)`: `residual += delta`, then
    /// `delta = rmsnorm(residual)`.
    FusedAddRmsNorm(Slot, Slot, LayerId, HiddenSize, RowsPerToken),
    /// `(delta, residual, layer, offset)`: [`MetalStep::FusedAddRmsNorm`] with a gain offset.
    FusedAddRmsNormWithOffset(Slot, Slot, LayerId, GainOffset),
    /// `(delta, residual, out, layer, width)`: `out = (rmsnorm(delta) + residual) * layer_scalar`.
    NormAddScalarMul(Slot, Slot, Slot, LayerId, HiddenSize),
    /// `(in, out, layer)`: multiply by the layer's loaded scalar.
    ScalarWeightMul(Slot, Slot, LayerId),
    /// `(in, out, scale)`.
    ScalarMul(Slot, Slot, Scale),
    /// `(in, out, layer, n, k)`: dense GEMM.
    Gemm(Slot, Slot, LayerId, NDim, KDim),
    /// MLX-affine GEMM.
    AffineQmm(AffineMatmul),
    /// `(in, out, layer, n, storage)`: row-broadcast projection bias.
    MetalBiasAdd(Slot, Slot, LayerId, NDim, BiasStorage),
    /// `(in, out, layer)`: dense fused gate/up GEMM + `silu(gate) * up`.
    FusedGateUpSiluMul(Slot, Slot, LayerId),
    /// `(in, out, layer)`: dense fused gate/up GEMM + `gelu(gate) * up`.
    FusedGateUpGeluMul(Slot, Slot, LayerId),
    /// `(gate, up, out, width)`: `silu(gate) * up`.
    SiluMul(Slot, Slot, Slot, IntermediateSize),
    /// `(gate, up, out)`: `gelu(gate) * up`.
    GeluMul(Slot, Slot, Slot),
    /// `(in, out)`: GELU (tanh approximation).
    Gelu(Slot, Slot),
    /// `(in, out)`: GELU (erf form).
    GeluErf(Slot, Slot),
    /// `(in, out)`: quick GELU.
    QuickGelu(Slot, Slot),
    /// `(in, out)`: final logit softcap.
    TanhSoftCap(Slot, Slot),
    /// `(delta, residual)`: `residual += delta`.
    Add(Slot, Slot),
    /// `(q, k, v, q_out, k_out, v_out, layer, pairing, class, kv_offsets)`: rope + paged KV write.
    RopeAppend(
        Slot,
        Slot,
        Slot,
        Slot,
        Slot,
        Slot,
        LayerId,
        RopeFormTag,
        AttnMask,
        KvOffsets,
    ),
    /// `(q, k, v, q_out, k_out, v_out, layer, pairing, class)`: [`MetalStep::RopeAppend`] with the
    /// per-head q/k norms and the v unit norm folded in; in-slots are the raw projections.
    RopeAppendNormed(
        Slot,
        Slot,
        Slot,
        Slot,
        Slot,
        Slot,
        LayerId,
        RopeFormTag,
        AttnMask,
    ),
    /// `(q, out, layer, pairing)`: decode attention over the paged cache.
    AttentionViaCache(Slot, Slot, LayerId, RopeFormTag),
    /// `(q, out, layer, pairing)`: sliding-window decode attention.
    SlidingAttentionViaCache(Slot, Slot, LayerId, RopeFormTag),
    /// `(q, out, layer, pairing)`: prefill attention over the paged cache.
    AttentionPrefillPaged(Slot, Slot, LayerId, RopeFormTag),
    /// `(q, out, layer, pairing)`: sliding-window prefill attention.
    SlidingAttentionPrefillPaged(Slot, Slot, LayerId, RopeFormTag),
    /// `(q, k, v, out)`: bidirectional encoder attention.
    EncoderAttention(Slot, Slot, Slot, Slot),
    /// `(q, k, v, out, segments)`: vision variable-length attention.
    VarlenAttention(Slot, Slot, Slot, Slot, CuSeqlens),
    /// `(q, k, q_out, k_out)`: vision 2-D rope.
    VisionRope(Slot, Slot, Slot, Slot),
    /// `(out)`: the staged vision pixels.
    LoadPixels(Slot),
    /// `(out)`: the staged vision position embeddings.
    LoadPosEmbeds(Slot),
    /// `(in, out, indices)`: row permutation by a runtime index buffer.
    EmbeddingGather(Slot, Slot, GatherIndices),
    /// `(qg, q, gate)`: split the doubled q projection into query and gate.
    GateSplit(Slot, Slot, Slot),
    /// `(attn, gate, out)`: `attn * sigmoid(gate)`.
    GateApply(Slot, Slot, Slot),
    /// `(routed, shared, gate, out)`: `routed + shared * sigmoid(gate)`.
    GateScale(Slot, Slot, Slot, Slot),
    /// `(qkv, z, a, b, out, layer)`: Gated-DeltaNet linear attention.
    GatedDeltaNet(Slot, Slot, Slot, Slot, Slot, LayerId),
    /// One step of a MoE block.
    Moe(MoeBlock, MoeStep),
    /// `(operand, layer, offsets)`: a KV writer's new `operand` rows encoded into its layer's
    /// packed store, the writer's `offsets` removed first. Its site is the writer's.
    KvEncode(KvOperand, LayerId, KvOffsets),
    /// `(operand, layer, class, offsets)`: the layer's `operand` staged out of its packed store for
    /// an attention of `class`, the writer's `offsets` restored. Its site is the writer's.
    KvStage(KvOperand, LayerId, AttnMask, KvOffsets),
    /// `(rows, which)`: an attention's query or output rows turned into or out of the codebook's
    /// domain, in place.
    RotateRows(Slot, RotatedRows),
    /// `(q, out, layer, pairing, class, offsets)`: the decode attention of `class`, read straight
    /// off the packed store into `out`, the writer's `offsets` restored. Its site is the writer's.
    AttnPackedKv(Slot, Slot, LayerId, RopeFormTag, AttnMask, KvOffsets),
    /// One step of the result matmul's sampled rows: each lowers from how its matmul does (its
    /// site is the matmul's).
    SampleRows(AffineMatmul, SampleRowsStep),
}

/// An MLX-affine matmul: `input · W` into `output`, `W` the layer's `n × k` weight packed `bits`
/// wide in groups of `group_size`; `vector_limit` rows or more take the matrix kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineMatmul {
    pub input: Slot,
    pub output: Slot,
    pub layer: LayerId,
    pub n: NDim,
    pub k: KDim,
    pub group_size: AffineGroupSize,
    pub bits: AffineBits,
    pub vector_limit: QmvBatchLimit,
}

/// The steps of a result matmul's sampled rows, in tape order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleRowsStep {
    /// Each sequence's sampled row to the front of the input, in place.
    Gather,
    /// The matmul, of the gathered rows.
    Matmul,
    /// The output's front rows back to the sampled rows, in place.
    Scatter,
    /// The matmul of every row, over the output — on the steps the other three do not run on.
    AllRows,
}

/// A MoE block's steps. Off-arena values live in [`MoeRegion`]s of the op scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeStep {
    /// `(x, layer, eps)`: the router's pre-norm of `x`.
    RouterNorm(Slot, LayerId, Eps),
    /// `(rows, layer)`: router logits.
    RouterLogits(MoeRows, LayerId),
    /// Row softmax, in place.
    Softmax(MoeScores),
    /// The logits' argsort.
    Argsort,
    /// Each row's top-k expert indices.
    TopK,
    /// The logits at the top-k indices.
    GatherScores,
    /// The scores times a constant, in place.
    Scale(Scale),
    /// The scores renormalized to sum to one, in place.
    Renorm,
    /// `(layer)`: each score times its expert's learned scale, in place.
    ExpertScale(LayerId),
    /// `(x)`: the pair rows sorted by expert — no command when the bake gathers.
    Sort(Slot),
    /// `(rows, layer, projection, group_size, width)`: one expert projection.
    ExpertMatmul(MoeRows, LayerId, ExpertProj, AffineGroupSize, ExpertWidth),
    /// `act(gate) * up`, in place on the gate rows.
    GatedAct(GatedAct),
    /// The pair rows back in token order — no command when the bake gathers.
    Unsort,
    /// `(out)`: each token's pair rows summed by its scores.
    Combine(Slot),
}

impl MetalStep {
    /// The step's layer, for the kinds that carry one.
    pub fn layer(&self) -> Option<LayerId> {
        let mut step = *self;
        step.layer_mut().copied()
    }

    /// This step `by` layers on — its row in a later iteration of the loop it is rolled into.
    pub fn advanced(mut self, by: u32) -> Self {
        if let Some(layer) = self.layer_mut() {
            layer.0 += by;
        }
        self
    }

    fn layer_mut(&mut self) -> Option<&mut LayerId> {
        use MetalStep as S;
        match self {
            S::RmsNorm(_, _, l, ..)
            | S::ScalarOffsetRmsNorm(_, _, l, ..)
            | S::MeanSubRmsNorm(_, _, l)
            | S::MeanSubRmsNormBiasAdd(_, _, l)
            | S::FusedAddRmsNorm(_, _, l, ..)
            | S::FusedAddRmsNormWithOffset(_, _, l, _)
            | S::ScalarWeightMul(_, _, l)
            | S::Gemm(_, _, l, ..)
            | S::AffineQmm(AffineMatmul { layer: l, .. })
            | S::MetalBiasAdd(_, _, l, ..)
            | S::FusedGateUpSiluMul(_, _, l)
            | S::FusedGateUpGeluMul(_, _, l)
            | S::AttentionViaCache(_, _, l, _)
            | S::SlidingAttentionViaCache(_, _, l, _)
            | S::AttentionPrefillPaged(_, _, l, _)
            | S::SlidingAttentionPrefillPaged(_, _, l, _)
            | S::NormAddScalarMul(_, _, _, l, _)
            | S::RopeAppend(_, _, _, _, _, _, l, ..)
            | S::RopeAppendNormed(_, _, _, _, _, _, l, ..)
            | S::GatedDeltaNet(_, _, _, _, _, l)
            | S::KvEncode(_, l, _)
            | S::KvStage(_, l, ..)
            | S::AttnPackedKv(_, _, l, ..)
            | S::SampleRows(
                AffineMatmul { layer: l, .. },
                SampleRowsStep::Matmul | SampleRowsStep::AllRows,
            )
            | S::Moe(
                _,
                MoeStep::RouterNorm(_, l, _)
                | MoeStep::RouterLogits(_, l)
                | MoeStep::ExpertScale(l)
                | MoeStep::ExpertMatmul(_, l, ..),
            ) => Some(l),
            S::Embed(..)
            | S::AffineEmbed(..)
            | S::SpliceMmEmbeds(..)
            | S::Reshape(..)
            | S::RmsNormUnit(..)
            | S::ScalarMul(..)
            | S::SiluMul(..)
            | S::GeluMul(..)
            | S::Gelu(..)
            | S::GeluErf(..)
            | S::QuickGelu(..)
            | S::TanhSoftCap(..)
            | S::Add(..)
            | S::EncoderAttention(..)
            | S::VarlenAttention(..)
            | S::VisionRope(..)
            | S::LoadPixels(..)
            | S::LoadPosEmbeds(..)
            | S::EmbeddingGather(..)
            | S::GateSplit(..)
            | S::GateApply(..)
            | S::GateScale(..)
            | S::RotateRows(..)
            | S::SampleRows(..)
            | S::Moe(..) => None,
        }
    }
}

/// One row of a metal step tape: a step and the runtime gate every command of it runs under
/// (`None`: always), or a rolled loop over the `body` rows after it (iteration `i` is the body
/// with every layer `i * stride` on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepRow {
    Step(MetalStep, Option<RuntimeGate>),
    Loop {
        iters: LoopIters,
        body: BodyLen,
        stride: LayerStride,
    },
}

/// One weight of a row's site: its kind (the lowering asks for "the `n`th `kind` weight",
/// [`accessor_slot`]'s rule) and the model source family it binds.
///
/// [`accessor_slot`]: scratchy_subtile::handoff::accessor_slot
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSource {
    pub kind: WeightKind,
    pub ix: SourceIx,
}

/// The rotary table each attention layer CLASS re-ropes cached K with (rope-on-read) — not an
/// operand of any step, so the model hands it to the bake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotaryTables {
    pub global: SourceIx,
    pub sliding: SourceIx,
}

impl RotaryTables {
    pub fn of(self, is_global: bool) -> SourceIx {
        if is_global { self.global } else { self.sliding }
    }
}

/// Memory a row can touch that another row of the same forward also touches — the barrier walk's
/// hazard locations, named. Weights, runtime inputs and inline scalars are absent: no row writes
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetalLoc {
    Arena(Slot),
    /// A layer's paged KV cache.
    Kv(LayerId),
    /// The MoE / GDN op scratch, ONE location.
    OpScratch,
    /// The KV codec's staging buffer, ONE location.
    CodecStaging,
    /// With the KV codec on: the ONE fp16 scratch every coded layer's cache half resolves to.
    TqScratch,
    /// With the KV codec on: a coded layer's own packed codes and norms.
    TqStore(LayerId),
}

impl MetalLoc {
    /// This location as it is in loop iteration `by` of a rolled body: a KV layer advances.
    pub fn advanced(self, by: u32) -> Self {
        match self {
            Self::Kv(l) => Self::Kv(l.advance(by)),
            Self::TqStore(l) => Self::TqStore(l.advance(by)),
            Self::Arena(_) | Self::OpScratch | Self::CodecStaging | Self::TqScratch => self,
        }
    }
}

/// A row's dataflow: what its commands read and write. An in-place operand is in both lists; the
/// rows one construct expanded to carry the construct's union (they fence as one).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowAccess {
    pub reads: Vec<MetalLoc>,
    pub writes: Vec<MetalLoc>,
}

/// One bucket's step tape: the backbone and the lm_head halves, each with one barrier flag, one
/// weight site and one dataflow per row. The default is the empty tape a refused canonical bakes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetalStepTape {
    pub backbone: Vec<StepRow>,
    pub backbone_barriers: Vec<bool>,
    pub backbone_sources: Vec<Vec<RowSource>>,
    pub backbone_access: Vec<RowAccess>,
    pub lm_head: Vec<StepRow>,
    pub lm_head_barriers: Vec<bool>,
    pub lm_head_sources: Vec<Vec<RowSource>>,
    pub lm_head_access: Vec<RowAccess>,
}
