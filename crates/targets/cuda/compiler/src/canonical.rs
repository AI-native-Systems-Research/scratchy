// SPDX-License-Identifier: Apache-2.0
//! ONE CANONICAL, LOWERED — CUDA's orchestration of the shared passes.
//!
//! In metal's order (`scratchy-target-metal-compiler`'s `canonical::lower_canonical`), under CUDA's
//! declared tables (`scratchy_target_cuda::op_abi`): the sampled rows, the graph and tape built
//! from whole ops, the fold pass, the colourer. Then CUDA's opcode lowering ([`steps_from_tape`]),
//! by each step's fold role: a kept step lowers on its own, a step driving a fold lowers to the
//! fused kernel, and a step a fold absorbed emits no launch.
//!
//! No KV codec (CUDA is bf16 only) and no wave order (that pass exists because metal's barriers
//! drain everything in flight; CUDA runs one stream in order, so its tape has no barrier at all).
//! No re-roll either: the unrolled tape is what the colourer and the fold pass read, and nothing
//! here lays steps out along a loop.
//!
//! The result is one bucket's tape in parts ([`CudaCanonical`]): the arena's layout, sized here
//! from the colourer's per-colour shapes, the input and result spans, each weight's bytes, and the
//! steps. They are the fields of a `static` `scratchy_target_cuda::tape::CudaTape`.
//!
//! [`steps_from_tape`]: crate::steps_from_tape

use std::collections::BTreeMap;

use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::GemmWeightKind;
use scratchy_subtile::sample_rows::{SampleRowsError, expand_sample_rows};
use scratchy_subtile::subtile_ir::{SubtileIR, TensorShape, ValidatedGraph, lower_region};
use scratchy_subtile::subtile_tape::{Instr, SubtileTape, lower_dag_to_tape};
use scratchy_subtile::tape_colouring::{Colour, ColourError, TapeColouring, colour_tape};
use scratchy_subtile::tape_folding::{FoldError, ModelFoldFacts, StepRole, TapeFolds, fold_tape};
use scratchy_target_cuda::op_abi::{
    CUDA_COLOUR_FACTS, CUDA_COLUMN_BLOCK, CUDA_FUSIONS, CUDA_SAMPLE_ROWS, CudaFusion,
};
use scratchy_target_cuda::step::CudaStep;
use scratchy_target_cuda::tape::{
    ArenaOffset, ArenaSpan, CUDA_ACT_BYTES, CUDA_ARENA_ALIGN, RotaryTable,
};

use crate::steps_from_tape::{StepCx, TapeStep, lower_fused, lower_step};

/// The model facts CUDA's opcode lowering reads, beside the canonical: expansion-time constants,
/// the counterpart of the `MetalModelConsts` fields metal's rope and gate lowerings read.
///
/// `GateSplit` carries only `half_cols`, so its `head_dim` is this one; `RopeRotate` carries its
/// `head_dim` but not its rotated width, so `rot_dim` (`partial_rotary_factor · head_dim`) is this
/// one, and `rope_theta` builds the rotary table's rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaModelFacts {
    pub head_dim: u32,
    pub rot_dim: u32,
    pub rope_theta: f64,
}

/// Which canonical this is, for the refusals.
#[derive(Clone, Copy, Debug)]
pub struct CanonicalAt<'a> {
    pub stem: &'a str,
    /// The canonical's rows.
    pub m: u64,
}

/// The shared passes' result for one canonical, before opcode lowering.
pub struct CudaPlan {
    /// The canonical after CUDA's sampled rows were inserted: what the graph was built from.
    pub lowered: LoweredDecode,
    pub graph: SubtileIR,
    /// The unrolled tape of [`Self::graph`].
    pub tape: SubtileTape,
    pub folds: TapeFolds<CudaFusion>,
    pub colouring: TapeColouring,
    /// The model facts the plan was made with, which opcode lowering reads.
    pub facts: CudaModelFacts,
}

/// One decode canonical, lowered: the parts of one bucket's `CudaTape`, every one a constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaCanonical {
    /// The bucket's token count: the rows of the embedded hidden, colour 0.
    pub bucket_m: u32,
    /// Per colour: its value's `rows × cols × CUDA_ACT_BYTES`.
    pub colour_bytes: Vec<u64>,
    /// Per colour: the sum of the `CUDA_ARENA_ALIGN`-aligned sizes of the colours before it.
    pub colour_offsets: Vec<u64>,
    /// Every colour's aligned size, summed, and the rotary table's after them.
    pub total_bytes: u64,
    /// Colour 0, the embedded hidden the forward uploads.
    pub input: ArenaSpan,
    /// The colourer's result colour, which the forward downloads.
    pub result: ArenaSpan,
    /// Per source index: its weight buffer's bytes; 0 for a source that is not a weight.
    pub weight_bytes: Vec<u64>,
    /// One step per launch, in tape order: each kept `Compute` but a view, and each fold's
    /// driver; a view or a `Compute` a fold absorbed has none.
    pub steps: Vec<CudaStep>,
    /// The rotary table, after every colour: `Some` exactly when the canonical binds a
    /// `Cos { local: false }` source.
    pub rotary: Option<RotaryTable>,
}

/// Why a canonical could not be lowered.
#[derive(Debug)]
pub enum CanonicalRefusal {
    SampleRows(SampleRowsError),
    /// The graph built from the canonical is not a valid DAG.
    Graph(String),
    Fold(FoldError),
    Colour(ColourError),
    /// The `step`-th `Compute` on the tape is an op CUDA has no lowering for yet.
    NotLowered {
        step: usize,
        op: &'static str,
    },
    /// The `step`-th `Compute` is a GEMM of a shape the bf16 kernel does not take.
    GemmShape {
        step: usize,
        m: u32,
        n: u32,
        k: u32,
        rule: GemmShapeRule,
    },
    /// The `step`-th `Compute` is a GEMM whose weight is not dense bf16.
    GemmWeight {
        step: usize,
        weight: GemmWeightKind,
    },
    /// An operand of the `step`-th `Compute` that its kernel cannot bind.
    Operand {
        step: usize,
        op: &'static str,
        operand: u8,
        why: OperandRefusal,
    },
    /// The `step`-th `Compute` writes part of its output tensor.
    PartialWrite {
        step: usize,
        op: &'static str,
    },
    /// The `step`-th `Compute` is a row kernel's op (RmsNorm, Add, SiluMul) of a shape its kernel
    /// does not take: the launcher's rule, stated at expansion time.
    RowShape {
        step: usize,
        op: &'static str,
        rows: u32,
        cols: u32,
        rule: RowShapeRule,
    },
    /// The `step`-th `Compute` is an RmsNorm whose `eps` is not finite, or not above 0.
    NormEps {
        step: usize,
        eps: f32,
    },
    /// The `step`-th `Compute` is an RmsNorm whose gain is `[rows, cols]`, not `[1, want]`.
    GainShape {
        step: usize,
        rows: u32,
        cols: u32,
        want: u32,
    },
    /// Operand `operand` of the `step`-th `Compute` is `[rows, cols]`, not the output's
    /// `[want_rows, want_cols]`: the row kernels broadcast nothing.
    OperandShape {
        step: usize,
        op: &'static str,
        operand: u8,
        rows: u32,
        cols: u32,
        want_rows: u32,
        want_cols: u32,
    },
    /// The `step`-th `Compute` is half of SwiGLU outside the `SiluMul` fold (a `Silu` the fold did
    /// not absorb, a `Mul` driving no fold) or the pre-fused `SiluMul`: CUDA has no standalone
    /// kernel for it.
    Unfused {
        step: usize,
        op: &'static str,
    },
    /// The `step`-th `Compute` drives a `SiluMul` fold that took the gate or up projection too;
    /// CUDA's `SiluMul` kernel reads both as computed buffers.
    FusionShape {
        step: usize,
    },
    /// The `step`-th `Compute` names a producer slot (a fold's activation, a gate's split) that no
    /// `Compute` of the tape writes.
    ProducerMissing {
        step: usize,
        op: &'static str,
    },
    /// The `step`-th `Compute` is a `Reshape` CUDA cannot run as a view (no launch, its input's
    /// buffer).
    View {
        step: usize,
        why: ViewRefusal,
    },
    /// Operand `operand` of the `step`-th `Compute`, a `RopeRotate`, is not the rotary `cos`
    /// (operand 1) or `sin` (operand 2) source of the global table: it is this binding, or a
    /// computed value (`None`).
    RotaryOperand {
        step: usize,
        operand: u8,
        binding: Option<SourceBinding>,
    },
    /// The `step`-th `Compute` is a `RopeRotate` of a shape the bf16 rope does not take.
    RopeShape {
        step: usize,
        rows: u32,
        cols: u32,
        head_dim: u32,
        rot_dim: u32,
        rule: RopeShapeRule,
    },
    /// The `step`-th `Compute` is a `GateSplit` of a shape the bf16 split does not take.
    GateShape {
        step: usize,
        rows: u32,
        half_cols: u32,
        head_dim: u32,
        rule: GateShapeRule,
    },
    /// The `step`-th `Compute` is a `GateApply` whose gate (operand 1) is not a `GateSplit`'s
    /// output.
    GateOperand {
        step: usize,
    },
    /// The `step`-th `Compute` is a `GateSplit` the colourer gave no second colour, so its gate
    /// has no buffer.
    SecondOutput {
        step: usize,
    },
    /// The arena cannot be laid out.
    Arena(ArenaRefusal),
}

/// Which rule a row kernel's shape breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowShapeRule {
    /// `dim` is zero.
    Empty { dim: &'static str },
    /// `cols` is not a multiple of the kernel's vector width, `multiple` bf16 per 16-byte load.
    OffVector { cols: u32, multiple: u32 },
    /// `rows · cols` is above `i32::MAX`, the largest element count the launcher indexes.
    TooManyElements { elements: u64 },
}

/// Why a `Reshape` cannot be a view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewRefusal {
    /// Its operand is a source, not a computed slot: the colourer aliases a view only to a step,
    /// so a view of a source would get a fresh colour nothing writes.
    OfASource,
    /// Its element count is not its input's.
    ElementCount { input: u64, output: u64 },
    /// Its colour is not its input's. CUDA's colour rule aliases every `Reshape` to operand 0,
    /// so this cannot happen under it; the arm checks rather than trusts it.
    Colour { input: u32, output: u32 },
}

/// Which rule a rope's shape breaks: the launcher's, stated at expansion time, and the table's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeShapeRule {
    /// `dim` is zero.
    Empty { dim: &'static str },
    /// `rot_dim` is not a multiple of `multiple`: each half of the rotated lanes is whole 16-byte
    /// vectors.
    RotOffVector { multiple: u32 },
    /// `rot_dim` is above `head_dim`.
    RotAboveHead,
    /// `head_dim` is not a multiple of `multiple`, the kernel's 16-byte bf16 vector.
    HeadOffVector { multiple: u32 },
    /// `cols` is not a whole number of heads.
    ColsOffHead,
    /// `rows · cols` is above `i32::MAX`, the largest element count the launcher indexes.
    TooManyElements { elements: u64 },
    /// `rows` is not the bucket's `tokens`: the rotary table has one row per token.
    RowsNotTokens { tokens: u32 },
}

/// Which rule a gate split's shape breaks: the launcher's, stated at expansion time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateShapeRule {
    /// `dim` is zero.
    Empty { dim: &'static str },
    /// `head_dim` is not a multiple of `multiple`, the kernel's 16-byte bf16 vector.
    HeadOffVector { multiple: u32 },
    /// `half_cols` is not a whole number of heads.
    HalfOffHead,
    /// `rows · 2·half_cols` is above `i32::MAX`, the largest element count the launcher indexes.
    TooManyElements { elements: u64 },
}

/// Which rule a GEMM's shape breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmShapeRule {
    /// `m = 1`: a decode GEMV, which has no kernel yet.
    Decode,
    /// `dim` is zero or not a multiple of the kernel's `tile`.
    OffTile {
        dim: &'static str,
        value: u32,
        tile: u32,
    },
    /// `dim` is above `i32::MAX`, the largest dimension ThunderKittens' `gl` stores.
    AboveIntMax { dim: &'static str, value: u32 },
}

/// Why an operand cannot be bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperandRefusal {
    /// The step has no such operand.
    Missing,
    /// The operand reads part of its tensor; a kernel binds whole buffers.
    PartialRead,
    /// The operand is a re-rolled per-layer source; CUDA does not re-roll.
    PerLayer,
    /// The operand has this many writers; one buffer cannot hold it.
    SplitWriters(usize),
    /// An activation operand is a source other than the embedded hidden.
    NotHiddenOrSlot {
        source: usize,
        binding: SourceBinding,
    },
    /// A weight operand is not a model weight: this source, or a computed value (`None`).
    NotAWeight {
        source: Option<(usize, SourceBinding)>,
    },
}

/// Why the arena cannot be laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaRefusal {
    /// No source is bound to the embedded hidden, so colour 0 has no size and the forward has no
    /// input. Until the embedding opcode lowers, colour 0's contents are the forward's input.
    NoEmbeddedHidden,
    /// Several sources are bound to the embedded hidden; colour 0 is one buffer.
    SeveralEmbeddedHidden,
    /// The colourer reports no shape for this colour (a two-output step's second buffer), so it
    /// cannot be sized.
    UnsizedColour(u32),
    /// The arena's bytes overflow `u64`.
    TooLarge,
}

impl std::fmt::Display for CanonicalRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SampleRows(e) => write!(f, "the shared sample-rows pass refused: {e}"),
            Self::Graph(e) => write!(f, "the graph is not a valid DAG: {e}"),
            Self::Fold(e) => write!(f, "the shared fold pass refused: {e}"),
            Self::Colour(e) => write!(f, "tape colorer refused: {e}"),
            Self::NotLowered { step, op } => write!(
                f,
                "step {step} computes {op}, which the cuda backend does not lower yet"
            ),
            Self::GemmShape {
                step,
                m,
                n,
                k,
                rule,
            } => write!(
                f,
                "step {step} is a GEMM of m={m} n={n} k={k}, which the bf16 kernel refuses: {rule}"
            ),
            Self::GemmWeight { step, weight } => write!(
                f,
                "step {step} is a GEMM over a {weight:?} weight; the bf16 kernel takes Dense only"
            ),
            Self::Operand {
                step,
                op,
                operand,
                why,
            } => write!(
                f,
                "step {step} ({op}) operand {operand} cannot be bound: {why}"
            ),
            Self::PartialWrite { step, op } => write!(
                f,
                "step {step} ({op}) writes part of its output tensor; a kernel writes a whole one"
            ),
            Self::RowShape {
                step,
                op,
                rows,
                cols,
                rule,
            } => write!(
                f,
                "step {step} ({op}) is rows={rows} cols={cols}, which its bf16 kernel refuses: \
                 {rule}"
            ),
            Self::NormEps { step, eps } => write!(
                f,
                "step {step} (RmsNorm) has eps = {eps}; it must be finite and above 0"
            ),
            Self::GainShape {
                step,
                rows,
                cols,
                want,
            } => write!(
                f,
                "step {step} (RmsNorm) has a gain of [{rows}, {cols}]; the kernel takes [1, {want}]"
            ),
            Self::OperandShape {
                step,
                op,
                operand,
                rows,
                cols,
                want_rows,
                want_cols,
            } => write!(
                f,
                "step {step} ({op}) operand {operand} is [{rows}, {cols}], not the output's \
                 [{want_rows}, {want_cols}]; the bf16 kernel broadcasts nothing"
            ),
            Self::Unfused { step, op } => write!(
                f,
                "step {step} computes {op} on its own; the cuda backend has no standalone kernel \
                 for it and runs SwiGLU only as the SiluMul fold of a Silu into the Mul it gates"
            ),
            Self::FusionShape { step } => write!(
                f,
                "step {step} drives a SiluMul fold that folded the gate or up projection in; the \
                 cuda SiluMul kernel reads both as computed buffers"
            ),
            Self::ProducerMissing { step, op } => write!(
                f,
                "step {step} ({op}) names a producer slot that no step of the tape writes"
            ),
            Self::View { step, why } => {
                write!(f, "step {step} (Reshape) cannot be a view: {why}")
            }
            Self::RotaryOperand {
                step,
                operand,
                binding: Some(binding),
            } => write!(
                f,
                "step {step} (RopeRotate) operand {operand} reads a {binding:?} source; the cuda \
                 rope reads the global rotary table, operand 1 Cos {{ local: false }} and operand 2 \
                 Sin {{ local: false }}"
            ),
            Self::RotaryOperand {
                step,
                operand,
                binding: None,
            } => write!(
                f,
                "step {step} (RopeRotate) operand {operand} is a computed value; the cuda rope \
                 reads the global rotary table, operand 1 Cos {{ local: false }} and operand 2 \
                 Sin {{ local: false }}"
            ),
            Self::RopeShape {
                step,
                rows,
                cols,
                head_dim,
                rot_dim,
                rule,
            } => write!(
                f,
                "step {step} (RopeRotate) is rows={rows} cols={cols} head_dim={head_dim} \
                 rot_dim={rot_dim}, which the bf16 rope refuses: {rule}"
            ),
            Self::GateShape {
                step,
                rows,
                half_cols,
                head_dim,
                rule,
            } => write!(
                f,
                "step {step} (GateSplit) is rows={rows} half_cols={half_cols} head_dim={head_dim}, \
                 which the bf16 gate split refuses: {rule}"
            ),
            Self::GateOperand { step } => write!(
                f,
                "step {step} (GateApply) operand 1 is not a GateSplit's output; the gate is a \
                 GateSplit's second buffer"
            ),
            Self::SecondOutput { step } => write!(
                f,
                "step {step} (GateSplit) has no second colour, so its gate has no buffer"
            ),
            Self::Arena(e) => write!(f, "the arena cannot be laid out: {e}"),
        }
    }
}

impl std::fmt::Display for RowShapeRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty { dim } => write!(f, "{dim} = 0"),
            Self::OffVector { cols, multiple } => write!(
                f,
                "cols = {cols} must be a multiple of {multiple}, the kernel's 16-byte bf16 vector"
            ),
            Self::TooManyElements { elements } => write!(
                f,
                "rows·cols = {elements} is above {}, the largest element count the launcher \
                 indexes",
                i32::MAX
            ),
        }
    }
}

impl std::fmt::Display for ViewRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OfASource => write!(
                f,
                "its operand is a source, and a view aliases only a computed slot"
            ),
            Self::ElementCount { input, output } => write!(
                f,
                "it has {output} elements and its input {input}; a view keeps the count"
            ),
            Self::Colour { input, output } => write!(
                f,
                "its colour is {output} and its input's {input}; a view is its input's buffer"
            ),
        }
    }
}

impl std::fmt::Display for RopeShapeRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty { dim } => write!(f, "{dim} = 0"),
            Self::RotOffVector { multiple } => write!(
                f,
                "rot_dim must be a multiple of {multiple}, two 16-byte bf16 vectors"
            ),
            Self::RotAboveHead => write!(f, "rot_dim is above head_dim"),
            Self::HeadOffVector { multiple } => write!(
                f,
                "head_dim must be a multiple of {multiple}, the kernel's 16-byte bf16 vector"
            ),
            Self::ColsOffHead => write!(f, "cols must be a whole number of heads"),
            Self::TooManyElements { elements } => write!(
                f,
                "rows·cols = {elements} is above {}, the largest element count the launcher \
                 indexes",
                i32::MAX
            ),
            Self::RowsNotTokens { tokens } => write!(
                f,
                "rows must be the bucket's {tokens} tokens, one rotary table row each"
            ),
        }
    }
}

impl std::fmt::Display for GateShapeRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty { dim } => write!(f, "{dim} = 0"),
            Self::HeadOffVector { multiple } => write!(
                f,
                "head_dim must be a multiple of {multiple}, the kernel's 16-byte bf16 vector"
            ),
            Self::HalfOffHead => write!(f, "half_cols must be a whole number of heads"),
            Self::TooManyElements { elements } => write!(
                f,
                "rows·2·half_cols = {elements} is above {}, the largest element count the \
                 launcher indexes",
                i32::MAX
            ),
        }
    }
}

impl std::fmt::Display for GemmShapeRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode => write!(
                f,
                "m = 1 is a decode GEMV, and CUDA has no decode GEMV kernel yet"
            ),
            Self::OffTile { dim, value, tile } => write!(
                f,
                "{dim} = {value} must be a non-zero multiple of {tile}, the bf16 GEMM's tile"
            ),
            Self::AboveIntMax { dim, value } => write!(
                f,
                "{dim} = {value} is above {}, the largest dimension ThunderKittens' `gl` stores",
                i32::MAX
            ),
        }
    }
}

impl std::fmt::Display for OperandRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(f, "the step has no such operand"),
            Self::PartialRead => {
                write!(f, "it reads part of its tensor; a kernel binds a whole one")
            }
            Self::PerLayer => write!(f, "it is a per-layer source, and CUDA does not re-roll"),
            Self::SplitWriters(n) => write!(f, "it has {n} writers; one buffer cannot hold it"),
            Self::NotHiddenOrSlot { source, binding } => write!(
                f,
                "it reads source {source} ({binding:?}); an activation must be a computed slot \
                 or the embedded hidden"
            ),
            Self::NotAWeight {
                source: Some((source, binding)),
            } => write!(
                f,
                "it reads source {source} ({binding:?}); a weight operand must be a model weight"
            ),
            Self::NotAWeight { source: None } => write!(
                f,
                "it reads a computed value; a weight operand must be a model weight"
            ),
        }
    }
}

impl std::fmt::Display for ArenaRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoEmbeddedHidden => write!(
                f,
                "no source is bound to the embedded hidden: colour 0 has no size and the forward \
                 has no input"
            ),
            Self::SeveralEmbeddedHidden => write!(
                f,
                "several sources are bound to the embedded hidden, and colour 0 is one buffer"
            ),
            Self::UnsizedColour(c) => write!(
                f,
                "colour {c} has no shape (a two-output step's second buffer), so it cannot be sized"
            ),
            Self::TooLarge => write!(f, "the arena's bytes overflow u64"),
        }
    }
}

/// Run the shared passes on one decode canonical under CUDA's tables: sampled rows, graph and
/// tape, folds, colours. `facts` ride along to opcode lowering; no shared pass reads them.
pub fn plan_canonical(
    l: &LoweredDecode,
    facts: CudaModelFacts,
    at: CanonicalAt<'_>,
) -> Result<CudaPlan, CanonicalRefusal> {
    let CanonicalAt { stem, m } = at;
    // CUDA's lm_head slice: none is declared, so this inserts nothing — but it runs, so a row
    // added to the table takes effect without touching this function.
    let lowered = expand_sample_rows(l, &CUDA_SAMPLE_ROWS).map_err(CanonicalRefusal::SampleRows)?;
    let graph = lower_region(&lowered.input, CUDA_COLUMN_BLOCK);
    let tape = {
        let valid = ValidatedGraph::new(&graph)
            .map_err(|e| CanonicalRefusal::Graph(format!("{stem} m={m}: {e}")))?;
        lower_dag_to_tape(&valid)
    };
    // The facts only gate patterns. CUDA's one pattern, `Gated`, reads `fold_projections` alone,
    // and CUDA has no kernel that folds the gate and up projections in, so every fact is off.
    let model = ModelFoldFacts {
        fold_projections: false,
        matvec_ends: false,
        matvec_norms: false,
        row_programs: false,
        hidden_out: false,
    };
    let folds =
        fold_tape(&graph, &tape, &lowered, &CUDA_FUSIONS, model).map_err(CanonicalRefusal::Fold)?;
    let colouring = colour_tape(
        &graph,
        &tape,
        &lowered.bindings,
        &folds.fold_facts(),
        &CUDA_COLOUR_FACTS,
    )
    .map_err(CanonicalRefusal::Colour)?;
    Ok(CudaPlan {
        lowered,
        graph,
        tape,
        folds,
        colouring,
        facts,
    })
}

/// Lower one decode canonical: [`plan_canonical`], the arena laid out from the colouring, then
/// every `Compute` on the tape, in tape order, by its fold role: a kept step through
/// [`lower_step`] (a view to no launch), a fold's driver through [`lower_fused`], and an absorbed
/// step (or an epilogue, which no CUDA fold makes) to no launch.
///
/// A step's refusal comes before the arena's: a canonical whose first op has no lowering is
/// refused for that op, whatever its bindings. A refusal's `step` is the step's position among
/// the tape's `Compute`s, absorbed ones included.
pub fn lower_canonical(
    l: &LoweredDecode,
    facts: CudaModelFacts,
    at: CanonicalAt<'_>,
) -> Result<CudaCanonical, CanonicalRefusal> {
    let plan = plan_canonical(l, facts, at)?;
    let arena = arena_layout(&plan);
    let cx = StepCx {
        graph: &plan.graph,
        bindings: &plan.lowered.bindings,
        colouring: &plan.colouring,
        facts: plan.facts,
        tokens: embedded_hidden(&plan).ok().map(|h| h.rows),
        offsets: match &arena {
            Ok(a) => Ok(a.offsets.as_slice()),
            Err(e) => Err(*e),
        },
        rotary: arena.as_ref().ok().and_then(|a| a.rotary).map(|t| t.offset),
    };
    let computes: Vec<TapeStep<'_>> = (plan.tape.instrs().iter())
        .filter_map(|i| match i {
            Instr::Compute {
                node,
                writes,
                inputs,
                ..
            } => Some((&plan.graph.nodes[node.index()], *writes, inputs)),
            _ => None,
        })
        .enumerate()
        .map(|(step, (node, writes, inputs))| TapeStep {
            step,
            node,
            writes,
            inputs,
        })
        .collect();
    let steps = (computes.iter())
        .filter_map(|s| match plan.folds.role(s.writes) {
            StepRole::Kept => lower_step(&cx, s, &computes).transpose(),
            StepRole::Drives(f) => Some(lower_fused(&cx, s, f, &computes)),
            StepRole::Absorbed { .. } | StepRole::Epilogue { .. } => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    let arena = arena.map_err(CanonicalRefusal::Arena)?;
    let span = |c: Colour| ArenaSpan {
        offset: ArenaOffset(arena.offsets[c.index() as usize]),
        bytes: arena.bytes[c.index() as usize],
    };
    let (input, result) = (span(Colour::ZERO), span(plan.colouring.result()));
    let weight_bytes = (plan.lowered.bindings.iter().enumerate())
        .map(|(t, binding)| match binding {
            SourceBinding::Weight { .. } => {
                let shape = plan.graph.tensors[t];
                elements_bytes(shape.rows, shape.cols).ok_or(ArenaRefusal::TooLarge)
            }
            _ => Ok(0),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(CanonicalRefusal::Arena)?;
    Ok(CudaCanonical {
        bucket_m: arena.bucket_m,
        colour_bytes: arena.bytes,
        colour_offsets: arena.offsets,
        total_bytes: arena.total,
        input,
        result,
        weight_bytes,
        steps,
        rotary: arena.rotary.map(|span| RotaryTable {
            span,
            rot_dim: plan.facts.rot_dim,
            theta_bits: plan.facts.rope_theta.to_bits(),
        }),
    })
}

/// The arena of one canonical: each colour's bytes and offset, the rotary table's span, and
/// their total.
struct Arena {
    bucket_m: u32,
    bytes: Vec<u64>,
    offsets: Vec<u64>,
    rotary: Option<ArenaSpan>,
    total: u64,
}

/// The shape of the one source bound to the embedded hidden: colour 0, the bucket's tokens.
fn embedded_hidden(plan: &CudaPlan) -> Result<TensorShape, ArenaRefusal> {
    let mut hidden = (plan.lowered.bindings.iter().enumerate())
        .filter(|(_, b)| **b == SourceBinding::EmbeddedHidden)
        .map(|(t, _)| plan.graph.tensors[t]);
    match (hidden.next(), hidden.next()) {
        (Some(h), None) => Ok(h),
        (None, _) => Err(ArenaRefusal::NoEmbeddedHidden),
        (Some(_), Some(_)) => Err(ArenaRefusal::SeveralEmbeddedHidden),
    }
}

/// Size every colour from the shape its values share, and place the colours one after another in
/// colour order, each at a multiple of [`CUDA_ARENA_ALIGN`]. Colour 0 is the embedded hidden's
/// buffer, so it is sized from the source bound to it. When the canonical binds the global rotary
/// `cos`, the rotary table (`[bucket_m, rot_dim]` bf16) follows the colours, aligned the same way.
fn arena_layout(plan: &CudaPlan) -> Result<Arena, ArenaRefusal> {
    let hidden = embedded_hidden(plan)?;
    // Every colour the tape's steps were given, by index: a slot's colour, and a two-output
    // step's second one. The colourer mints colours `0..count`, each for some step.
    let mut colours: BTreeMap<u32, Colour> = BTreeMap::from([(0, Colour::ZERO)]);
    for instr in plan.tape.instrs() {
        if let Instr::Compute { writes, .. } = instr {
            let c = plan.colouring.colour_of(*writes);
            let second = plan.colouring.second_colour_of(*writes);
            colours.extend(c.into_iter().chain(second).map(|c| (c.index(), c)));
        }
    }
    let bytes = (0..plan.colouring.count().get())
        .map(|i| {
            let (rows, cols) = match colours.get(&i) {
                Some(&Colour::ZERO) => (hidden.rows, hidden.cols),
                Some(&c) => plan
                    .colouring
                    .shape_of(c)
                    .ok_or(ArenaRefusal::UnsizedColour(i))?,
                None => return Err(ArenaRefusal::UnsizedColour(i)),
            };
            elements_bytes(rows, cols).ok_or(ArenaRefusal::TooLarge)
        })
        .collect::<Result<Vec<u64>, _>>()?;
    let mut offsets = Vec::with_capacity(bytes.len());
    let mut total = 0u64;
    let mut place = |b: u64| -> Result<u64, ArenaRefusal> {
        let at = total;
        total = (b.checked_next_multiple_of(CUDA_ARENA_ALIGN))
            .and_then(|aligned| total.checked_add(aligned))
            .ok_or(ArenaRefusal::TooLarge)?;
        Ok(at)
    };
    for &b in &bytes {
        offsets.push(place(b)?);
    }
    let rotary = (plan.lowered.bindings)
        .contains(&SourceBinding::Cos { local: false })
        .then(|| {
            let bytes =
                elements_bytes(hidden.rows, plan.facts.rot_dim).ok_or(ArenaRefusal::TooLarge)?;
            Ok(ArenaSpan {
                offset: ArenaOffset(place(bytes)?),
                bytes,
            })
        })
        .transpose()?;
    Ok(Arena {
        bucket_m: hidden.rows,
        bytes,
        offsets,
        rotary,
        total,
    })
}

/// `rows × cols` activations' bytes, or `None` on overflow.
fn elements_bytes(rows: u32, cols: u32) -> Option<u64> {
    u64::from(rows)
        .checked_mul(u64::from(cols))?
        .checked_mul(CUDA_ACT_BYTES)
}
