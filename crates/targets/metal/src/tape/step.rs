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

pub use super::lowered::ActivationWidth;

use super::ids::ArenaSlotIdx as Slot;
use super::lowered::RuntimeGate;

/// Where a KV writer puts the step's new K and V rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvWrite {
    /// The paged pool.
    Pool,
    /// The pool, and the KV codec's packed store: the codec's encodes of the rows, folded into the
    /// writer (`MetalFusion::KvEncoded`).
    PoolAndPacked,
}

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
    /// The token rows a norm the step absorbed reads (`MetalFusion::NormedQmv`): the step
    /// normalizes each as it loads it. One row, gathered.
    Normed(Slot, RowNorm),
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
    /// The experts' packed storage, as the block's projection steps lower it: what decides
    /// whether a grouped bake's expert GEMMs run W4A8 (every step of the block must agree).
    pub quant: ExpertQuant,
}

/// An expert bank's MLX-affine group size and each projection's code width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertQuant {
    pub group_size: AffineGroupSize,
    pub widths: ExpertWidths,
}

/// The code width each expert projection lowers at — apart when the quantization declares the
/// roles apart (a mixed-width bank).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertWidths {
    pub gate: AffineBits,
    pub up: AffineBits,
    pub down: AffineBits,
}

impl ExpertWidths {
    /// The one width all three projections share, if they do.
    pub fn uniform(self) -> Option<AffineBits> {
        (self.gate == self.up && self.up == self.down).then_some(self.gate)
    }
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
/// (a rolled body's rows name iteration 0's); an [`ActivationWidth`] is the width of the rows the
/// step writes. Field order per kind: inputs, outputs, layer, shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalStep {
    /// `(out)`: token embedding into the hidden.
    Embed(Slot),
    /// `(out, group_size, bits)`: MLX-affine quantized token embedding.
    AffineEmbed(Slot, AffineGroupSize, AffineBits),
    /// `(slot)`: projected vision rows spliced into the embedded hidden, in place.
    SpliceMmEmbeds(Slot),
    /// `(in, out, rows_mult, rows_div)`: a view — no kernel runs; later rows see
    /// `num_tokens / rows_div` rows.
    Reshape(Slot, Slot, RowsPerToken, RowsDivisor),
    /// `(in, out, layer, width, rows_per_token)`.
    RmsNorm(Slot, Slot, LayerId, HiddenSize, RowsPerToken),
    /// `(in, out, layer, offset, width, rows_per_token)`: `rmsnorm(x, w + offset)`.
    ScalarOffsetRmsNorm(Slot, Slot, LayerId, GainOffset, HiddenSize, RowsPerToken),
    /// `(in, out, width, rows_per_token)`: unit-gain RMSNorm.
    RmsNormUnit(Slot, Slot, HiddenSize, RowsPerToken),
    /// `(in, out, layer, width)`: centred (mean-subtracted) RMSNorm.
    MeanSubRmsNorm(Slot, Slot, LayerId, ActivationWidth),
    /// `(in, out, layer, width)`: LayerNorm with bias.
    MeanSubRmsNormBiasAdd(Slot, Slot, LayerId, ActivationWidth),
    /// `(delta, residual, layer, width, rows_per_token)`: `residual += delta`, then
    /// `delta = rmsnorm(residual)`.
    FusedAddRmsNorm(Slot, Slot, LayerId, HiddenSize, RowsPerToken),
    /// `(delta, residual, layer, offset)`: [`MetalStep::FusedAddRmsNorm`] with a gain offset.
    FusedAddRmsNormWithOffset(Slot, Slot, LayerId, GainOffset),
    /// `(delta, residual, out, layer, width)`: `out = (rmsnorm(delta) + residual) * layer_scalar`.
    NormAddScalarMul(Slot, Slot, Slot, LayerId, HiddenSize),
    /// `(delta, residual, out, norm, width)`: `out = rmsnorm(delta) + residual`.
    NormAdd(Slot, Slot, Slot, RowNorm, HiddenSize),
    /// `(in, out, layer)`: multiply by the layer's loaded scalar.
    ScalarWeightMul(Slot, Slot, LayerId),
    /// `(in, out, scale, width)`.
    ScalarMul(Slot, Slot, Scale, ActivationWidth),
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
    /// `(gate, act)`: one row's MLX-affine gate and up matvecs and `act(gate) * up`, written to
    /// the gate matmul's output. The up projection has the gate's shape; its weight is the step's
    /// second linear site.
    AffineGatedQmv(AffineMatmul, GatedAct),
    /// `(gate, up, out, width)`: `silu(gate) * up`.
    SiluMul(Slot, Slot, Slot, IntermediateSize),
    /// `(gate, up, out)`: `gelu(gate) * up`.
    GeluMul(Slot, Slot, Slot),
    /// `(in, out, width)`: GELU (tanh approximation).
    Gelu(Slot, Slot, ActivationWidth),
    /// `(in, out, width)`: GELU (erf form).
    GeluErf(Slot, Slot, ActivationWidth),
    /// `(in, out, width)`: quick GELU.
    QuickGelu(Slot, Slot, ActivationWidth),
    /// `(in, out, width)`: final logit softcap.
    TanhSoftCap(Slot, Slot, ActivationWidth),
    /// `(delta, residual, width)`: `residual += delta`.
    Add(Slot, Slot, ActivationWidth),
    /// `(q, k, v, q_out, k_out, v_out, layer, pairing, class, kv_offsets, write)`: rope + paged KV
    /// write.
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
        KvWrite,
    ),
    /// `(q, k, v, q_out, k_out, v_out, layer, pairing, class, write)`: [`MetalStep::RopeAppend`]
    /// with the per-head q/k norms and the v unit norm folded in; in-slots are the raw projections.
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
        KvWrite,
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
    /// `(in, out, indices, width)`: row permutation by a runtime index buffer.
    EmbeddingGather(Slot, Slot, GatherIndices, ActivationWidth),
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
    /// A group of row-wise steps over one width as one command (`MetalFusion::RowProgram`).
    RowProgram(Box<RowProgram>),
    /// A one-row step's decode attention that runs its KV writer (`MetalFusion::RopedAttention`).
    RopedAttention(Box<RopedAttention>),
}

/// A decode attention and the KV writer it runs: each as it lowers on its own. The row's first
/// `writer_sources` weight sources are the writer's, the rest the attention's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RopedAttention {
    pub writer: MetalStep,
    pub attention: MetalStep,
    pub writer_sources: usize,
}

/// The instructions a row program holds, at most.
pub const ROW_PROGRAM_INSTRS: usize = 16;

/// The registers a row program holds (`row_program.metal`'s `RP_REGS`): one per row it loads and
/// per step.
pub const ROW_PROGRAM_REGISTERS: usize = 12;

/// A row program (`row_program.metal`): its instructions over registers, the activation rows it
/// loads, the buffers it stores, and the layer of each weight it reads — `gains` its norms', in
/// order, then `scalars` its scalar weights'; its site lists the weights in instruction order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowProgram {
    pub instrs: [Option<RowInstr>; ROW_PROGRAM_INSTRS],
    pub inputs: [Option<Slot>; 4],
    pub outputs: [Option<Slot>; 3],
    pub gains: [Option<LayerId>; 8],
    pub scalars: [Option<LayerId>; 2],
    pub width: HiddenSize,
}

/// A row program's instruction; registers, inputs, outputs, gains and scalars by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowInstr {
    Load {
        dst: u8,
        input: u8,
    },
    Add {
        dst: u8,
        a: u8,
        b: u8,
    },
    Norm {
        dst: u8,
        a: u8,
        gain: u8,
        eps: Eps,
        offset: GainOffset,
    },
    ScaleWeight {
        dst: u8,
        a: u8,
        scalar: u8,
    },
    Scale {
        dst: u8,
        a: u8,
        scale: Scale,
    },
    Store {
        output: u8,
        a: u8,
    },
}

/// An MLX-affine matmul: `input · W` into `output`, `W` the layer's `n × k` weight packed `bits`
/// wide in groups of `group_size`; `vector_limit` rows or more take the matrix kernel. `ends`:
/// what its one-row matvec does around the dot.
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
    pub ends: QmvEnds,
}

/// What a one-row MLX-affine matvec does around its dot: normalize its input as it loads it
/// (`MetalFusion::NormedQmv` — the input is the norm's, the gain its site's `RmsNorm`); then
/// (`MetalFusion::QmvEpilogue`) add its projection's bias, scale the row, and add it into the
/// residual stream its output holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QmvEnds {
    pub norm: Option<RowNorm>,
    pub bias: Option<BiasStorage>,
    pub scale: Option<Scale>,
    pub residual: bool,
}

/// What a gathered expert combine computes as it stores each row (`MetalFusion::CombineEpilogue`):
/// a shared expert's `(rows, gate)` added scaled by σ of the token's gate (`GateScale`), then the
/// add into the residual stream its output buffer holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CombineEnds {
    pub gate_scale: Option<(Slot, Slot)>,
    pub residual: bool,
}

/// The RMSNorm a matvec applies to its input: the norm's layer (its gain's), epsilon and gain
/// offset (`rmsnorm(x, w + offset)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowNorm {
    pub layer: LayerId,
    pub eps: Eps,
    pub offset: GainOffset,
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
    /// `(rows, layer, norm)`: router logits — of the rows RMS-normalized by the epsilon and the
    /// router's scale as they load, with `norm` (`MetalFusion::NormedRouter`, one row only).
    RouterLogits(MoeRows, LayerId, Option<Eps>),
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
    /// One expert projection.
    ExpertMatmul(ExpertMatmul),
    /// `act(gate) * up`, in place on the gate rows.
    GatedAct(GatedAct),
    /// The pair rows back in token order — no command when the bake gathers.
    Unsort,
    /// `(out)`: each token's pair rows summed by its scores.
    Combine(Slot),
    /// `(gate, up width, act, routing)`: the gate and up projections and `act(gate) * up` — the
    /// up reads the gate's rows, layer and group size — routing the block's token first with
    /// `routing` and storing its picks and scores (`MetalFusion::MoeRouted`).
    GateUpAct(ExpertMatmul, ExpertWidth, GatedAct, Option<RouteProgram>),
    /// `(down, out, ends)`: the down projection, the unsort and the combine into `out`, and what
    /// it computes as it stores each row.
    DownCombine(ExpertMatmul, Slot, CombineEnds),
    /// The routing from the router logits to the top-k indices and scores.
    Route(RouteProgram),
}

/// A MoE block's routing from its router logits to each token's top-k experts and their scores,
/// as one command (`moe_route.metal`): the sort, the top-k and their scores always, the rest as
/// the router says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteProgram {
    /// A softmax over every expert before the sort (Qwen's shared-expert router).
    pub pre_softmax: bool,
    /// The top-k scores times a constant (Gemma's `hidden^-0.5`).
    pub scale: Option<Scale>,
    /// What the top-k scores go through next.
    pub post: RoutePost,
    /// Each score times its expert's learned scale, from layer `l`'s router (Gemma).
    pub expert_scale: Option<LayerId>,
}

/// The top-k scores' last row-wide step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutePost {
    None,
    Softmax,
    Renorm,
}

/// One expert projection: its `rows` times the layer's `proj` experts, packed `width` wide in
/// groups of `group_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertMatmul {
    pub rows: MoeRows,
    pub layer: LayerId,
    pub proj: ExpertProj,
    pub group_size: AffineGroupSize,
    pub width: ExpertWidth,
}

impl MetalStep {
    /// The step's layer, for the kinds that carry one.
    pub fn layer(&self) -> Option<LayerId> {
        let mut step = self.clone();
        step.layer_mut().copied()
    }

    /// This step `by` layers on — its row in a later iteration of the loop it is rolled into.
    pub fn advanced(mut self, by: u32) -> Self {
        if let Some(layer) = self.layer_mut() {
            layer.0 += by;
        }
        // A matvec's folded norm reads its own layer's gain, a row program each weight its own.
        if let MetalStep::AffineQmm(g) | MetalStep::AffineGatedQmv(g, _) = &mut self
            && let Some(norm) = &mut g.ends.norm
        {
            norm.layer.0 += by;
        }
        if let MetalStep::RowProgram(r) = &mut self {
            for l in r.gains.iter_mut().chain(r.scalars.iter_mut()).flatten() {
                l.0 += by;
            }
        }
        // The attention's layer is the step's; its writer's moves with it.
        if let MetalStep::RopedAttention(f) = &mut self {
            f.writer = f.writer.clone().advanced(by);
        }
        // Rows a MoE step normalizes read its norm's own layer's gain.
        if let MetalStep::Moe(
            _,
            MoeStep::RouterLogits(MoeRows::Normed(_, norm), ..)
            | MoeStep::ExpertMatmul(ExpertMatmul {
                rows: MoeRows::Normed(_, norm),
                ..
            })
            | MoeStep::GateUpAct(
                ExpertMatmul {
                    rows: MoeRows::Normed(_, norm),
                    ..
                },
                ..
            ),
        ) = &mut self
        {
            norm.layer.0 += by;
        }
        self
    }

    fn layer_mut(&mut self) -> Option<&mut LayerId> {
        use MetalStep as S;
        match self {
            S::RopedAttention(f) => f.attention.layer_mut(),
            S::RmsNorm(_, _, l, ..)
            | S::ScalarOffsetRmsNorm(_, _, l, ..)
            | S::MeanSubRmsNorm(_, _, l, _)
            | S::MeanSubRmsNormBiasAdd(_, _, l, _)
            | S::FusedAddRmsNorm(_, _, l, ..)
            | S::FusedAddRmsNormWithOffset(_, _, l, _)
            | S::ScalarWeightMul(_, _, l)
            | S::Gemm(_, _, l, ..)
            | S::AffineQmm(AffineMatmul { layer: l, .. })
            | S::MetalBiasAdd(_, _, l, ..)
            | S::FusedGateUpSiluMul(_, _, l)
            | S::FusedGateUpGeluMul(_, _, l)
            | S::AffineGatedQmv(AffineMatmul { layer: l, .. }, _)
            | S::AttentionViaCache(_, _, l, _)
            | S::SlidingAttentionViaCache(_, _, l, _)
            | S::AttentionPrefillPaged(_, _, l, _)
            | S::SlidingAttentionPrefillPaged(_, _, l, _)
            | S::NormAddScalarMul(_, _, _, l, _)
            | S::NormAdd(_, _, _, RowNorm { layer: l, .. }, _)
            | S::RopeAppend(_, _, _, _, _, _, l, ..)
            | S::RopeAppendNormed(_, _, _, _, _, _, l, ..)
            | S::GatedDeltaNet(_, _, _, _, _, l)
            | S::KvStage(_, l, ..)
            | S::AttnPackedKv(_, _, l, ..)
            | S::SampleRows(
                AffineMatmul { layer: l, .. },
                SampleRowsStep::Matmul | SampleRowsStep::AllRows,
            )
            | S::Moe(
                _,
                MoeStep::RouterNorm(_, l, _)
                | MoeStep::RouterLogits(_, l, _)
                | MoeStep::ExpertScale(l)
                | MoeStep::ExpertMatmul(ExpertMatmul { layer: l, .. })
                | MoeStep::GateUpAct(ExpertMatmul { layer: l, .. }, ..)
                | MoeStep::DownCombine(ExpertMatmul { layer: l, .. }, ..)
                | MoeStep::Route(RouteProgram {
                    expert_scale: Some(l),
                    ..
                }),
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
            | S::Moe(..)
            | S::RowProgram(..) => None,
        }
    }
}

/// One row of a metal step tape: a step and the runtime gate every command of it runs under
/// (`None`: always), or a rolled loop over the `body` rows after it (iteration `i` is the body
/// with every layer `i * stride` on).
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// One bucket's step tape: the backbone and the lm_head halves, each with one barrier flag and
/// one weight site per row. The default is the empty tape a refused canonical bakes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetalStepTape {
    pub backbone: Vec<StepRow>,
    pub backbone_barriers: Vec<bool>,
    pub backbone_sources: Vec<Vec<RowSource>>,
    pub lm_head: Vec<StepRow>,
    pub lm_head_barriers: Vec<bool>,
    pub lm_head_sources: Vec<Vec<RowSource>>,
}
