// SPDX-License-Identifier: Apache-2.0
//! THE TAPE FOLDER — which tape steps a target computes inside another step's fused command.
//!
//! A target's kernels fuse: a residual add into the norm that reads it, an activation into the
//! multiply it gates, a query rotation into the rope that appends K/V. WHICH fusions a target has
//! is a target fact, declared as a [`FusionTable`]: the patterns that apply, in which order, over
//! which op kinds, and the fused command each fold becomes. Facts about one model (whether its
//! projections fold) arrive as [`ModelFoldFacts`]. The matching that honours them is the same for
//! every target and lives here.
//!
//! The pass reads the UNROLLED tape — a step's operands are the tape's dataflow, its op is its
//! source provenance — and decides for every step: kept, absorbed into another step's fused
//! command, an epilogue another step's command writes, or the driver of fusions. The decisions are
//! an overlay keyed on the [`SlotId`] each step writes. The tape itself is untouched, so the
//! re-roll and the layer-loop search see exactly the tape they saw before.
//!
//! The rules:
//! - fusion legality counts CONSUMERS: every step operand, plus one for the result — except the
//!   operands a table declares are not dataflow ([`CountedOperands`]);
//! - patterns apply in SWEEPS: a sweep walks the steps in source-op order and, at each step,
//!   applies every pattern of the sweep driven by that step's kind, in table order; a later sweep
//!   sees what an earlier one absorbed;
//! - a step already absorbed is never absorbed again, except by a gated activation's
//!   projections, which fold unconditionally once the activation has.

use std::collections::BTreeMap;

use crate::handoff::{LoweredDecode, SourceBinding};
use crate::lower::GemmWeightKind;
use crate::ops::SubOpKind;
use crate::subtile_ir::{KvOperand, SubOp, SubtileIR, TensorId};
use crate::subtile_tape::{SlotId, SubtileTape};
use crate::tape_colouring::FoldFacts;
use crate::tape_steps::{Operand, OperandIx, StepPos, TapeReadError, read_steps};

/// Operand `operand` of the step writing `step`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepOperand {
    pub step: SlotId,
    pub operand: OperandIx,
}

/// For `kind`, only the operands before `before` are dataflow a fold counts; the rest name a buffer
/// another step wrote (an attention's new K/V are the cache its rope appended to).
#[derive(Clone, Copy, Debug)]
pub struct CountedOperands {
    pub kind: SubOpKind,
    pub before: OperandIx,
}

/// An activation a gated fold absorbs, and the fused command each form of the fold becomes.
#[derive(Clone, Copy, Debug)]
pub struct GatedKernel<K> {
    pub activation: SubOpKind,
    /// The gate and up projections stay separate commands.
    pub split: K,
    /// The gate and up projections fold in too.
    pub fused: K,
}

/// One structural fusion pattern: op-kind roles a target's table fills, and the fused command it
/// becomes. Operand positions are the op vocabulary's own (see each [`SubOp`]'s docs).
#[derive(Clone, Copy, Debug)]
pub enum FoldPattern<K: 'static> {
    /// A `driver` absorbs the closest preceding unabsorbed `sibling` (a rope append and the query
    /// rotation of the same attention).
    Sibling {
        driver: SubOpKind,
        sibling: SubOpKind,
        kernel: K,
    },
    /// `norm(sub(x, mean(x)))`: the sub (the norm's operand 0) and the mean (the sub's operand 1)
    /// fold into the norm. A single-consumer norm that a `bias` step reads as its operand 0 becomes
    /// `biased`, whose command writes that step's buffer: the step is its epilogue.
    CentredNorm {
        norm: SubOpKind,
        sub: SubOpKind,
        mean: SubOpKind,
        bias: SubOpKind,
        kernel: K,
        biased: K,
    },
    /// `norm(add(..))`: the add folds into ONE norm reading it — of the unabsorbed norms reading
    /// it as operand 0, the first in the canonical graph's node order. The add may have other
    /// readers.
    ResidualNorm {
        norm: SubOpKind,
        add: SubOpKind,
        kernel: K,
    },
    /// `mul(act(gate), up)`: the activation (the mul's operand 0) folds into the mul; when the
    /// model folds projections, so do the activation's operand-0 and the mul's operand-1
    /// producers.
    Gated {
        mul: SubOpKind,
        activations: &'static [GatedKernel<K>],
    },
    /// A `rope` that absorbed its query rotation, whose Q (the rotation's operand 0) and K (its
    /// operand 0) chains peel back through single-consumer row-preserving views, `gain_norm`s and
    /// `unit_norm`s to the raw projections — firing only when its V (operand 3) IS a
    /// single-consumer `unit_norm`.
    NormedRope {
        rope: SubOpKind,
        gain_norm: SubOpKind,
        unit_norm: SubOpKind,
        kernel: K,
    },
    /// `scale(add(norm(delta, gain), residual), w)`: the add and the norm fold into the scale.
    NormAddScale {
        scale: SubOpKind,
        add: SubOpKind,
        norm: SubOpKind,
        kernel: K,
    },
    /// `add(norm(delta, gain), residual)`, the norm either operand and read by the add alone: the
    /// norm folds into the add. Apply it after the folds that take the add or the norm with more.
    NormAdd {
        add: SubOpKind,
        norm: SubOpKind,
        kernel: K,
    },
    /// `act(gate, up)` over expert rows: the gate and up `matmul`s (the activation's operands 0
    /// and 1), each read by the activation alone, fold into it.
    ExpertGated {
        act: SubOpKind,
        matmul: SubOpKind,
        kernel: K,
    },
    /// `combine(unsort(matmul(..)), scores)`: the unsort (the combine's operand 0) and the
    /// `matmul` whose rows it unsorts (its operand 0), each read by the next alone, fold into the
    /// combine.
    ExpertCombined {
        combine: SubOpKind,
        unsort: SubOpKind,
        matmul: SubOpKind,
        kernel: K,
    },
    /// A router's routing, driven at its `top_k` step. The `sort` it reads folds into it, and so
    /// does the `pre` step the sort reads when the sort and the `gather` are its only readers.
    /// The `gather` of the top-k scores (operand 0 the sort's logits, operand 1 the top-k) and
    /// the steps the scores then pass through, each the sole reader of the last, are its
    /// epilogues: at most one per `tail` stage, the stages in order.
    Route {
        top_k: SubOpKind,
        sort: SubOpKind,
        pre: SubOpKind,
        gather: SubOpKind,
        tail: &'static [&'static [SubOpKind]; ROUTE_TAIL_STAGES],
        kernel: K,
    },
    /// A KV `writer`'s codec encodes — the K and V `encode` steps the codec expanded it into, each
    /// reading back rows the writer just wrote — fold into the writer, which encodes the rows it
    /// holds. Apply it after the writer's own folds: it extends whatever command they made.
    Encoded {
        writer: SubOpKind,
        encode: SubOpKind,
        kernel: K,
    },
    /// A `norm` that only `matmul`s read, each as its operand 0 and of `weights`, folds into every
    /// one of them: each normalizes the norm's input as it loads it (`normed`). A residual `add`
    /// the norm reads, whose delta (operand 0) only a `matmul` of `weights` reads, becomes that
    /// matmul's epilogue: its rows add into the residual (`residual`). Only on a model whose
    /// matvecs normalize ([`ModelFoldFacts::normed_matvecs`]); apply it before the folds that take
    /// a norm whole.
    NormedMatvecs {
        norm: SubOpKind,
        add: SubOpKind,
        matmul: SubOpKind,
        weights: GemmWeightKind,
        normed: K,
        residual: K,
    },
}

/// The stages a routing fold's scores may pass through after the gather.
pub const ROUTE_TAIL_STAGES: usize = 3;

impl<K> FoldPattern<K> {
    /// The kind of step the pattern is matched at.
    const fn driver(&self) -> SubOpKind {
        match self {
            Self::Sibling { driver, .. } => *driver,
            Self::CentredNorm { norm, .. } | Self::ResidualNorm { norm, .. } => *norm,
            Self::Gated { mul, .. } => *mul,
            Self::NormedRope { rope, .. } => *rope,
            Self::NormAddScale { scale, .. } => *scale,
            Self::NormAdd { add, .. } => *add,
            Self::ExpertGated { act, .. } => *act,
            Self::ExpertCombined { combine, .. } => *combine,
            Self::Route { top_k, .. } => *top_k,
            Self::Encoded { writer, .. } => *writer,
            Self::NormedMatvecs { norm, .. } => *norm,
        }
    }
}

/// A target's fusions, as declared data.
pub struct FusionTable<K: 'static> {
    /// Operands that are not consumer edges.
    pub counted: &'static [CountedOperands],
    /// The patterns, sweep by sweep; within a sweep, in the order they are tried at a step.
    pub sweeps: &'static [&'static [FoldPattern<K>]],
}

/// Facts about one model the patterns read — data the caller derives, never code in the pass.
#[derive(Clone, Copy, Debug)]
pub struct ModelFoldFacts {
    /// A gated activation's gate and up projections fold into its command.
    pub fold_projections: bool,
    /// The matvecs normalize their input and add into a residual ([`FoldPattern::NormedMatvecs`]).
    pub normed_matvecs: bool,
}

/// What a fused command computes beyond its driver step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedShape {
    Sibling {
        sibling: SlotId,
    },
    CentredNorm {
        sub: SlotId,
        mean: SlotId,
        bias: Option<SlotId>,
    },
    ResidualNorm {
        add: SlotId,
    },
    Gated {
        activation: SlotId,
        gate: Option<SlotId>,
        up: Option<SlotId>,
    },
    /// `q`/`k`/`v` are the raw operands the rope works on in place; the gains are the peeled gain
    /// norms' weight operands.
    NormedRope {
        q: SlotId,
        k: SlotId,
        v: SlotId,
        q_gain: Option<StepOperand>,
        k_gain: Option<StepOperand>,
    },
    /// The norm's input and gain, and the add's other operand.
    NormAdd {
        norm: SlotId,
        delta: StepOperand,
        residual: StepOperand,
        gain: StepOperand,
    },
    /// `residual` is the add's other operand.
    NormAddScale {
        add: SlotId,
        norm: SlotId,
        delta: SlotId,
        residual: StepOperand,
        gain: StepOperand,
        scale: StepOperand,
    },
    ExpertGated {
        gate: SlotId,
        up: SlotId,
    },
    ExpertCombined {
        unsort: SlotId,
        down: SlotId,
    },
    /// `tail[s]` is the step of tail stage `s`, if the scores pass through one.
    Route {
        sort: SlotId,
        pre: Option<SlotId>,
        gather: SlotId,
        tail: [Option<SlotId>; ROUTE_TAIL_STAGES],
    },
    /// The writer's K and V encodes; the command is the writer's earlier fold's, extended.
    Encoded {
        k: SlotId,
        v: SlotId,
    },
    /// The norm this matmul's input passes through: the command reads the norm's input and gain.
    NormedMatvec {
        norm: SlotId,
    },
    /// The residual add this matmul's rows feed: the command writes the add's buffer.
    ResidualMatvec {
        add: SlotId,
    },
}

/// A fold a step drives: the fused command it becomes, and what that command computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fusion<K> {
    pub kernel: K,
    pub shape: FusedShape,
}

/// How a step lowers under the folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepRole<'a, K> {
    /// On its own.
    Kept,
    /// Inside `into`'s fused command; its output never materialises.
    Absorbed { into: SlotId },
    /// By `of`'s fused command, which writes this step's buffer.
    Epilogue { of: SlotId },
    /// As this fused command.
    Drives(&'a Fusion<K>),
}

/// The fold decisions for every step of an unrolled tape, keyed on the slot each step writes.
#[derive(Clone, Debug)]
pub struct TapeFolds<K> {
    absorbed: BTreeMap<SlotId, SlotId>,
    epilogue: BTreeMap<SlotId, SlotId>,
    /// Per driver, the folds it drives in the order they were applied; the last is its command.
    fusions: BTreeMap<SlotId, Vec<Fusion<K>>>,
}

impl<K> TapeFolds<K> {
    /// How the step writing `step` lowers.
    pub fn role(&self, step: SlotId) -> StepRole<'_, K> {
        if let Some(into) = self.absorbed.get(&step) {
            return StepRole::Absorbed { into: *into };
        }
        if let Some(of) = self.epilogue.get(&step)
            && !self.absorbed.contains_key(of)
        {
            return StepRole::Epilogue { of: *of };
        }
        match self.fusions.get(&step).and_then(|f| f.last()) {
            Some(f) => StepRole::Drives(f),
            None => StepRole::Kept,
        }
    }

    /// Every fold `driver` drives, in the order they were applied.
    pub fn driven(&self, driver: SlotId) -> &[Fusion<K>] {
        self.fusions.get(&driver).map_or(&[], Vec::as_slice)
    }

    /// Every absorbed step, with the step whose fused command computes it.
    pub fn absorbed(&self) -> impl Iterator<Item = (SlotId, SlotId)> + '_ {
        self.absorbed.iter().map(|(a, w)| (*a, *w))
    }

    /// Every fold, with the step driving it: by driver, then in the order they were applied.
    pub fn fusions(&self) -> impl Iterator<Item = (SlotId, &Fusion<K>)> {
        self.fusions
            .iter()
            .flat_map(|(d, fs)| fs.iter().map(move |f| (*d, f)))
    }

    /// The folds as the colourer reads them.
    pub fn fold_facts(&self) -> FoldFacts {
        FoldFacts {
            absorbed: self.absorbed.clone(),
            normed_rope: self
                .fusions()
                .filter(|(_, f)| matches!(f.shape, FusedShape::NormedRope { .. }))
                .map(|(d, _)| d)
                .collect(),
        }
    }
}

/// A source op's index in the `LoweringInput` the graph was lowered from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceOpIx(u32);

impl std::fmt::Display for SourceOpIx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "source op {}", self.0)
    }
}

/// Why a tape could not be folded.
#[derive(Clone, Debug)]
pub enum FoldError {
    /// The tape could not be read as steps.
    Tape(TapeReadError),
    /// No step computes this source op, so no decision can be keyed on it.
    Unstepped { at: SourceOpIx, op: &'static str },
    /// Two steps compute one source op; one decision cannot key on both.
    SeveralSteps {
        op: &'static str,
        first: StepPos,
        second: StepPos,
    },
    /// A step lacks the operand a pattern reads there.
    MissingOperand {
        step: StepPos,
        op: &'static str,
        operand: OperandIx,
    },
}

impl std::fmt::Display for FoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tape(e) => write!(f, "{e}"),
            Self::Unstepped { at, op } => write!(f, "{at} ({op}) is computed by no step"),
            Self::SeveralSteps { op, first, second } => {
                write!(f, "{first} and {second} both compute one {op}")
            }
            Self::MissingOperand { step, op, operand } => {
                write!(f, "{step} ({op}) has no operand {operand}")
            }
        }
    }
}

/// Fold the unrolled `tape` of `graph` under a target's `table` and one model's `model` facts.
///
/// `lowered` carries the op list `graph` was lowered from (the table is keyed on its ops), the
/// source bindings (a weight operand is one bound to a weight), and each op's canonical tile — the
/// node order a matcher over the canonical graph meets candidates in.
pub fn fold_tape<K: Copy>(
    graph: &SubtileIR,
    tape: &SubtileTape,
    lowered: &LoweredDecode,
    table: &FusionTable<K>,
    model: ModelFoldFacts,
) -> Result<TapeFolds<K>, FoldError> {
    let ops = Ops::read(graph, tape, lowered)?;
    let n = ops.slot.len();
    let mut consumers = vec![0usize; n];
    for (j, args) in ops.args.iter().enumerate() {
        let kind = ops.kind(j);
        let counted = table
            .counted
            .iter()
            .find(|c| c.kind == kind)
            .map_or(args.len(), |c| c.before.index().min(args.len()));
        for a in &args[..counted] {
            if let Arg::Op(q) = a {
                consumers[*q] += 1;
            }
        }
    }
    if lowered.input.result < n {
        consumers[lowered.input.result] += 1;
    }
    let mut folder = Folder {
        ops,
        consumers,
        model,
        absorbed: vec![None; n],
        epilogue: vec![None; n],
        fusions: (0..n).map(|_| Vec::new()).collect(),
    };
    for sweep in table.sweeps {
        for i in 0..n {
            let kind = folder.ops.kind(i);
            for pattern in sweep.iter().filter(|p| p.driver() == kind) {
                folder.apply(pattern, i)?;
            }
        }
    }
    Ok(folder.finish())
}

/// A source op's operand, as the tape reads it.
#[derive(Clone, Copy)]
enum Arg {
    Op(usize),
    Source(TensorId),
}

/// The unrolled tape indexed by source op: each source op is computed by exactly one step.
struct Ops<'a> {
    lowered: &'a LoweredDecode,
    /// Each source op's step's node op.
    ops: Vec<&'a SubOp>,
    pos: Vec<StepPos>,
    slot: Vec<SlotId>,
    args: Vec<Vec<Arg>>,
}

impl<'a> Ops<'a> {
    fn read(
        graph: &'a SubtileIR,
        tape: &SubtileTape,
        lowered: &'a LoweredDecode,
    ) -> Result<Self, FoldError> {
        let steps = read_steps(graph, tape).map_err(FoldError::Tape)?;
        let mut at: Vec<Option<usize>> = vec![None; lowered.input.ops.len()];
        for (p, s) in steps.iter().enumerate() {
            if let Some(q) = at[s.source_op].replace(p) {
                return Err(FoldError::SeveralSteps {
                    op: s.op.name(),
                    first: StepPos(q as u32),
                    second: StepPos(p as u32),
                });
            }
        }
        let at = at
            .into_iter()
            .enumerate()
            .map(|(j, p)| {
                p.ok_or(FoldError::Unstepped {
                    at: SourceOpIx(j as u32),
                    op: lowered.input.ops[j].op.name(),
                })
            })
            .collect::<Result<Vec<usize>, _>>()?;
        let args = at
            .iter()
            .map(|&p| {
                steps[p]
                    .operands
                    .iter()
                    .map(|o| match o {
                        Operand::Step(q) => Arg::Op(steps[*q].source_op),
                        Operand::Source(t) => Arg::Source(*t),
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            lowered,
            ops: at.iter().map(|&p| steps[p].op).collect(),
            pos: at.iter().map(|&p| StepPos(p as u32)).collect(),
            slot: at.iter().map(|&p| steps[p].writes).collect(),
            args,
        })
    }

    fn op(&self, j: usize) -> &'a SubOp {
        self.ops[j]
    }

    fn kind(&self, j: usize) -> SubOpKind {
        self.op(j).kind()
    }

    /// Operand `k` of `j`, which the pattern reading it requires.
    fn arg(&self, j: usize, k: u8) -> Result<Arg, FoldError> {
        self.args[j]
            .get(k as usize)
            .copied()
            .ok_or(FoldError::MissingOperand {
                step: self.pos[j],
                op: self.op(j).name(),
                operand: OperandIx(k),
            })
    }

    /// The op producing operand `k` of `j`, when a step produces it.
    fn in_op(&self, j: usize, k: u8) -> Result<Option<usize>, FoldError> {
        Ok(match self.arg(j, k)? {
            Arg::Op(q) => Some(q),
            Arg::Source(_) => None,
        })
    }

    /// The op producing `j`'s operand 0, if `j` has one and a step produces it.
    fn first_op(&self, j: usize) -> Option<usize> {
        match self.args[j].first() {
            Some(Arg::Op(q)) => Some(*q),
            _ => None,
        }
    }

    /// `j`'s first operand bound to a weight.
    fn weight_operand(&self, j: usize) -> Option<StepOperand> {
        self.args[j]
            .iter()
            .position(|a| {
                matches!(a, Arg::Source(t) if matches!(
                    self.lowered.bindings.get(t.index()),
                    Some(SourceBinding::Weight { .. } | SourceBinding::WeightScale { .. })
                ))
            })
            .map(|k| self.operand(j, k as u8))
    }

    fn operand(&self, j: usize, k: u8) -> StepOperand {
        StepOperand {
            step: self.slot[j],
            operand: OperandIx(k),
        }
    }

    /// Where `j` sits in the canonical graph's node order; an op with no canonical tile sorts last.
    fn canonical_rank(&self, j: usize) -> u32 {
        self.lowered.op_tiles[j].map_or(u32::MAX, |(tile, _)| tile)
    }
}

/// A view that keeps the row count — the reshape back from per-head rows.
fn row_preserving_view(op: &SubOp) -> bool {
    matches!(op, SubOp::Reshape { rows, .. } if rows.preserves_rows())
}

/// A dry-run peel of a rope operand chain: where it ends, the gain it passed, what it would fold.
struct Peeled {
    raw: usize,
    gain: Option<StepOperand>,
    folded: Vec<usize>,
}

/// The decisions so far, by source op.
struct Folder<'a, K> {
    ops: Ops<'a>,
    consumers: Vec<usize>,
    model: ModelFoldFacts,
    absorbed: Vec<Option<usize>>,
    epilogue: Vec<Option<usize>>,
    fusions: Vec<Vec<Fusion<K>>>,
}

impl<K: Copy> Folder<'_, K> {
    fn apply(&mut self, pattern: &FoldPattern<K>, i: usize) -> Result<(), FoldError> {
        match *pattern {
            FoldPattern::Sibling {
                sibling, kernel, ..
            } => {
                self.sibling(i, sibling, kernel);
                Ok(())
            }
            FoldPattern::CentredNorm {
                sub,
                mean,
                bias,
                kernel,
                biased,
                ..
            } => self.centred_norm(i, [sub, mean, bias], kernel, biased),
            FoldPattern::ResidualNorm {
                norm, add, kernel, ..
            } => {
                self.residual_norm(i, norm, add, kernel);
                Ok(())
            }
            FoldPattern::Gated { activations, .. } => self.gated(i, activations),
            FoldPattern::NormedRope {
                gain_norm,
                unit_norm,
                kernel,
                ..
            } => self.normed_rope(i, gain_norm, unit_norm, kernel),
            FoldPattern::NormAddScale {
                add, norm, kernel, ..
            } => self.norm_add_scale(i, add, norm, kernel),
            FoldPattern::NormAdd { norm, kernel, .. } => self.norm_add(i, norm, kernel),
            FoldPattern::ExpertGated { matmul, kernel, .. } => {
                let gate = self.sole_producer(i, 0, matmul)?;
                let up = self.sole_producer(i, 1, matmul)?;
                if let (Some(g), Some(u)) = (gate, up) {
                    let (gate, up) = (self.ops.slot[g], self.ops.slot[u]);
                    self.absorb(i, &[g, u], kernel, FusedShape::ExpertGated { gate, up });
                }
                Ok(())
            }
            FoldPattern::ExpertCombined {
                unsort,
                matmul,
                kernel,
                ..
            } => {
                let Some(un) = self.sole_producer(i, 0, unsort)? else {
                    return Ok(());
                };
                if let Some(d) = self.sole_producer(un, 0, matmul)? {
                    let (unsort, down) = (self.ops.slot[un], self.ops.slot[d]);
                    self.absorb(
                        i,
                        &[un, d],
                        kernel,
                        FusedShape::ExpertCombined { unsort, down },
                    );
                }
                Ok(())
            }
            FoldPattern::Route {
                sort,
                pre,
                gather,
                tail,
                kernel,
                ..
            } => self.route(i, [sort, pre, gather], tail, kernel),
            FoldPattern::Encoded { encode, kernel, .. } => {
                self.encoded(i, encode, kernel);
                Ok(())
            }
            FoldPattern::NormedMatvecs {
                add,
                matmul,
                weights,
                normed,
                residual,
                ..
            } => self.normed_matvecs(i, add, (matmul, weights), normed, residual),
        }
    }

    /// Norm `i` into every matmul reading it, and the residual add it reads into the add's delta
    /// matmul. The norm is absorbed into its last reader in tape order, where its input is last read.
    fn normed_matvecs(
        &mut self,
        i: usize,
        add: SubOpKind,
        (matmul, weights): (SubOpKind, GemmWeightKind),
        normed: K,
        residual: K,
    ) -> Result<(), FoldError> {
        if !self.model.normed_matvecs || self.absorbed[i].is_some() || !self.fusions[i].is_empty() {
            return Ok(());
        }
        let ops = &self.ops;
        let is_matvec = |j: usize| {
            ops.kind(j) == matmul
                && matches!(ops.op(j), SubOp::MatmulTile { weight, .. } if weight.kind() == weights)
        };
        let reads = |j: usize| {
            ops.args[j]
                .iter()
                .filter(|a| matches!(a, Arg::Op(q) if *q == i))
        };
        let readers: Vec<usize> = (0..ops.slot.len())
            .filter(|&j| reads(j).count() > 0)
            .collect();
        let normalizes = |j: usize| {
            is_matvec(j)
                && self.absorbed[j].is_none()
                && ops.first_op(j) == Some(i)
                && reads(j).count() == 1
        };
        if readers.len() != self.consumers[i] || !readers.iter().all(|&j| normalizes(j)) {
            return Ok(());
        }
        let Some(&last) = readers.iter().max_by_key(|&&j| ops.pos[j]) else {
            return Ok(());
        };
        let delta = match ops.first_op(i) {
            Some(a) if ops.kind(a) == add && self.absorbed[a].is_none() => self
                .sole_producer(a, 0, matmul)?
                .filter(|&d| is_matvec(d))
                .map(|d| (a, d)),
            _ => None,
        };
        if let Some((a, d)) = delta {
            self.epilogue[a] = Some(d);
            let add = self.ops.slot[a];
            self.record(d, residual, FusedShape::ResidualMatvec { add });
        }
        self.absorbed[i] = Some(last);
        let norm = self.ops.slot[i];
        for j in readers {
            self.record(j, normed, FusedShape::NormedMatvec { norm });
        }
        Ok(())
    }

    /// `i`'s K and V `encode` steps — in the construct it was expanded into, not yet taken.
    fn encoded(&mut self, i: usize, encode: SubOpKind, kernel: K) {
        let expansion = &self.ops.lowered.op_expansion;
        let Some(id) = expansion[i].map(|x| x.id) else {
            return;
        };
        let of = |operand: KvOperand| {
            (0..self.ops.slot.len()).find(|&j| {
                self.ops.kind(j) == encode
                    && expansion[j].map(|x| x.id) == Some(id)
                    && matches!(self.ops.op(j), SubOp::KvEncode { operand: o } if *o == operand)
                    && self.absorbed[j].is_none()
            })
        };
        if let (Some(k), Some(v)) = (of(KvOperand::K), of(KvOperand::V)) {
            let (ks, vs) = (self.ops.slot[k], self.ops.slot[v]);
            self.absorb(i, &[k, v], kernel, FusedShape::Encoded { k: ks, v: vs });
        }
    }

    fn route(
        &mut self,
        i: usize,
        [sort, pre, gather]: [SubOpKind; 3],
        tail: &[&[SubOpKind]; ROUTE_TAIL_STAGES],
        kernel: K,
    ) -> Result<(), FoldError> {
        let Some(s) = self.sole_producer(i, 0, sort)? else {
            return Ok(());
        };
        let ops = &self.ops;
        let logits = ops.in_op(s, 0)?;
        let free = |j: usize| self.absorbed[j].is_none() && self.epilogue[j].is_none();
        let reads_route = |g: usize| -> Result<bool, FoldError> {
            Ok(ops.kind(g) == gather && free(g) && ops.in_op(g, 1)? == Some(i))
        };
        let mut g = None;
        for j in 0..ops.slot.len() {
            if reads_route(j)? && ops.in_op(j, 0)? == logits && logits.is_some() {
                g = Some(j);
                break;
            }
        }
        let Some(g) = g else {
            return Ok(());
        };
        let pre = logits.filter(|&l| ops.kind(l) == pre && self.consumers[l] == 2 && free(l));
        let mut stages = [None; ROUTE_TAIL_STAGES];
        let (mut cur, mut next) = (g, 0);
        while self.consumers[cur] == 1 {
            let Some(c) = (0..ops.slot.len()).find(|&c| ops.first_op(c) == Some(cur)) else {
                break;
            };
            let Some(st) = (next..ROUTE_TAIL_STAGES).find(|&st| tail[st].contains(&ops.kind(c)))
            else {
                break;
            };
            if !free(c) {
                break;
            }
            (stages[st], next, cur) = (Some(c), st + 1, c);
        }
        let slot = |j: usize| self.ops.slot[j];
        let shape = FusedShape::Route {
            sort: slot(s),
            pre: pre.map(slot),
            gather: slot(g),
            tail: stages.map(|t| t.map(slot)),
        };
        for e in std::iter::once(g).chain(stages.into_iter().flatten()) {
            self.epilogue[e] = Some(i);
        }
        self.absorb(
            i,
            &[s].into_iter().chain(pre).collect::<Vec<_>>(),
            kernel,
            shape,
        );
        Ok(())
    }

    /// The `kind` step producing `j`'s operand `k`, when nothing else reads it and no fold took it.
    fn sole_producer(&self, j: usize, k: u8, kind: SubOpKind) -> Result<Option<usize>, FoldError> {
        let p = self.ops.in_op(j, k)?;
        Ok(p.filter(|&p| {
            self.ops.kind(p) == kind && self.consumers[p] == 1 && self.absorbed[p].is_none()
        }))
    }

    /// `i` drives `kernel`, computing `ops` inside it.
    fn absorb(&mut self, i: usize, ops: &[usize], kernel: K, shape: FusedShape) {
        for &j in ops {
            self.absorbed[j] = Some(i);
        }
        self.record(i, kernel, shape);
    }

    fn record(&mut self, i: usize, kernel: K, shape: FusedShape) {
        self.fusions[i].push(Fusion { kernel, shape });
    }

    fn sibling(&mut self, i: usize, sibling: SubOpKind, kernel: K) {
        let Some(j) = (0..i)
            .rev()
            .find(|&j| self.ops.kind(j) == sibling && self.absorbed[j].is_none())
        else {
            return;
        };
        self.absorbed[j] = Some(i);
        let sibling = self.ops.slot[j];
        self.record(i, kernel, FusedShape::Sibling { sibling });
    }

    fn centred_norm(
        &mut self,
        i: usize,
        [sub, mean, bias]: [SubOpKind; 3],
        kernel: K,
        biased: K,
    ) -> Result<(), FoldError> {
        let ops = &self.ops;
        let Some(sb) = ops.first_op(i) else {
            return Ok(());
        };
        if ops.kind(sb) != sub || self.absorbed[sb].is_some() {
            return Ok(());
        }
        let Some(mu) = ops.in_op(sb, 1)? else {
            return Ok(());
        };
        if ops.kind(mu) != mean || self.absorbed[mu].is_some() {
            return Ok(());
        }
        self.absorbed[sb] = Some(i);
        self.absorbed[mu] = Some(i);
        let b = if self.consumers[i] == 1 {
            (0..ops.slot.len()).find(|&b| ops.kind(b) == bias && ops.first_op(b) == Some(i))
        } else {
            None
        };
        let shape = FusedShape::CentredNorm {
            sub: ops.slot[sb],
            mean: ops.slot[mu],
            bias: b.map(|b| ops.slot[b]),
        };
        if let Some(b) = b {
            self.epilogue[b] = Some(i);
        }
        self.record(i, if b.is_some() { biased } else { kernel }, shape);
        Ok(())
    }

    fn residual_norm(&mut self, i: usize, norm: SubOpKind, add: SubOpKind, kernel: K) {
        let ops = &self.ops;
        let Some(a) = ops.first_op(i) else { return };
        if ops.kind(a) != add {
            return;
        }
        let winner = (0..ops.slot.len())
            .filter(|&j| {
                ops.kind(j) == norm && ops.first_op(j) == Some(a) && self.absorbed[j].is_none()
            })
            .min_by_key(|&j| ops.canonical_rank(j));
        if winner == Some(i) && self.absorbed[a].is_none() {
            self.absorbed[a] = Some(i);
            let add = ops.slot[a];
            self.record(i, kernel, FusedShape::ResidualNorm { add });
        }
    }

    fn gated(&mut self, i: usize, activations: &[GatedKernel<K>]) -> Result<(), FoldError> {
        let ops = &self.ops;
        let Some(si) = ops.in_op(i, 0)? else {
            return Ok(());
        };
        let Some(act) = activations.iter().find(|a| a.activation == ops.kind(si)) else {
            return Ok(());
        };
        if self.absorbed[si].is_some() {
            return Ok(());
        }
        self.absorbed[si] = Some(i);
        let (mut gate, mut up) = (None, None);
        if self.model.fold_projections {
            if let Some(g) = ops.in_op(si, 0)? {
                self.absorbed[g] = Some(i);
                gate = Some(ops.slot[g]);
            }
            if let Some(u) = ops.in_op(i, 1)? {
                self.absorbed[u] = Some(i);
                up = Some(ops.slot[u]);
            }
        }
        let kernel = if self.model.fold_projections {
            act.fused
        } else {
            act.split
        };
        let activation = ops.slot[si];
        self.record(
            i,
            kernel,
            FusedShape::Gated {
                activation,
                gate,
                up,
            },
        );
        Ok(())
    }

    /// Peel a rope operand chain back from `start`, committing nothing.
    fn peel(
        &self,
        start: usize,
        gain_norm: SubOpKind,
        unit_norm: SubOpKind,
    ) -> Result<Peeled, FoldError> {
        let mut cur = start;
        let mut gain = None;
        let mut folded = Vec::new();
        while self.absorbed[cur].is_none() && self.consumers[cur] == 1 {
            let (view, kind) = (row_preserving_view(self.ops.op(cur)), self.ops.kind(cur));
            if !view && kind == gain_norm {
                gain = self.ops.weight_operand(cur);
            } else if !view && kind != unit_norm {
                break;
            }
            folded.push(cur);
            match self.ops.in_op(cur, 0)? {
                Some(p) => cur = p,
                None => break,
            }
        }
        Ok(Peeled {
            raw: cur,
            gain,
            folded,
        })
    }

    fn normed_rope(
        &mut self,
        i: usize,
        gain_norm: SubOpKind,
        unit_norm: SubOpKind,
        kernel: K,
    ) -> Result<(), FoldError> {
        let Some(rot) = self.absorbed.iter().position(|a| *a == Some(i)) else {
            return Ok(());
        };
        let ops = &self.ops;
        let (q0, k0, v0) = (ops.in_op(rot, 0)?, ops.in_op(i, 0)?, ops.in_op(i, 3)?);
        let (Some(q0), Some(k0), Some(v0)) = (q0, k0, v0) else {
            return Ok(());
        };
        let q = self.peel(q0, gain_norm, unit_norm)?;
        let k = self.peel(k0, gain_norm, unit_norm)?;
        // V is not peeled: the fold fires only when the rope's V operand IS the unit norm.
        if ops.kind(v0) != unit_norm || self.absorbed[v0].is_some() || self.consumers[v0] != 1 {
            return Ok(());
        }
        let v = ops.in_op(v0, 0)?.unwrap_or(v0);
        let shape = FusedShape::NormedRope {
            q: ops.slot[q.raw],
            k: ops.slot[k.raw],
            v: ops.slot[v],
            q_gain: q.gain,
            k_gain: k.gain,
        };
        for f in q.folded.iter().chain(&k.folded).chain([&v0]) {
            self.absorbed[*f] = Some(i);
        }
        self.record(i, kernel, shape);
        Ok(())
    }

    /// Add `i` with a norm only it reads as either operand: the norm folds into it.
    fn norm_add(&mut self, i: usize, norm: SubOpKind, kernel: K) -> Result<(), FoldError> {
        if self.absorbed[i].is_some() || !self.fusions[i].is_empty() {
            return Ok(());
        }
        let ops = &self.ops;
        for k in 0..2u8 {
            let Some(nrm) = ops.in_op(i, k)? else {
                continue;
            };
            let free = self.absorbed[nrm].is_none() && self.fusions[nrm].is_empty();
            if ops.kind(nrm) != norm || !free || self.consumers[nrm] != 1 {
                continue;
            }
            let Some(gain) = ops.weight_operand(nrm) else {
                continue;
            };
            ops.arg(i, 1 - k)?;
            let shape = FusedShape::NormAdd {
                norm: ops.slot[nrm],
                delta: ops.operand(nrm, 0),
                residual: ops.operand(i, 1 - k),
                gain,
            };
            self.absorbed[nrm] = Some(i);
            self.record(i, kernel, shape);
            return Ok(());
        }
        Ok(())
    }

    fn norm_add_scale(
        &mut self,
        i: usize,
        add: SubOpKind,
        norm: SubOpKind,
        kernel: K,
    ) -> Result<(), FoldError> {
        let ops = &self.ops;
        let Some(a) = ops.in_op(i, 0)? else {
            return Ok(());
        };
        if ops.kind(a) != add || self.absorbed[a].is_some() {
            return Ok(());
        }
        let Some(nrm) = ops.in_op(a, 0)? else {
            return Ok(());
        };
        if ops.kind(nrm) != norm || self.absorbed[nrm].is_some() {
            return Ok(());
        }
        let Some(delta) = ops.in_op(nrm, 0)? else {
            return Ok(());
        };
        let (Some(gain), Some(scale)) = (ops.weight_operand(nrm), ops.weight_operand(i)) else {
            return Ok(());
        };
        ops.arg(a, 1)?;
        let shape = FusedShape::NormAddScale {
            add: ops.slot[a],
            norm: ops.slot[nrm],
            delta: ops.slot[delta],
            residual: ops.operand(a, 1),
            gain,
            scale,
        };
        self.absorbed[a] = Some(i);
        self.absorbed[nrm] = Some(i);
        self.record(i, kernel, shape);
        Ok(())
    }

    fn finish(self) -> TapeFolds<K> {
        let slot = &self.ops.slot;
        let by_slot = |v: &[Option<usize>]| {
            v.iter()
                .enumerate()
                .filter_map(|(j, w)| w.map(|w| (slot[j], slot[w])))
                .collect()
        };
        TapeFolds {
            absorbed: by_slot(&self.absorbed),
            epilogue: by_slot(&self.epilogue),
            fusions: self
                .fusions
                .into_iter()
                .enumerate()
                .filter(|(_, f)| !f.is_empty())
                .map(|(j, f)| (slot[j], f))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{TURBOQUANT_SHAPED_CODEC, front_end_lowered, one_layer_input_shaped};
    use crate::handoff::{Expansion, ExpansionId};
    use crate::kv_codec::expand_kv_codec;
    use crate::lower::{ArchOp, GemmWeight, InputRef, LoweringInput, OpDesc};
    use crate::subtile_ir::{
        AttnMask, EwKind, GainConvention, RopeFormTag, RowScale, SourceShape, ValidatedGraph,
        lower_region,
    };
    use crate::subtile_tape::{Instr, lower_dag_to_tape};
    use InputRef::{Ext, Op};
    use ktir_superdsc::head_counts::HeadDim;

    /// A neutral kernel identity per fold.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kern {
        Rope,
        Centred,
        CentredBiased,
        Residual,
        GeluSplit,
        GeluFused,
        NormedRope,
        NormAddScale,
        ExpertGated,
        ExpertCombined,
        Route,
        Encoded,
        Normed,
        ResidualMatvec,
    }

    const SWEEPS: &[&[FoldPattern<Kern>]] = {
        use SubOpKind as K;
        &[
            &[
                FoldPattern::Sibling {
                    driver: K::RopeAppend,
                    sibling: K::RopeRotate,
                    kernel: Kern::Rope,
                },
                FoldPattern::CentredNorm {
                    norm: K::RmsNorm,
                    sub: K::Sub,
                    mean: K::Mean,
                    bias: K::BiasAdd,
                    kernel: Kern::Centred,
                    biased: Kern::CentredBiased,
                },
                FoldPattern::NormedMatvecs {
                    norm: K::RmsNorm,
                    add: K::Add,
                    matmul: K::MatmulTile,
                    weights: GemmWeightKind::Dense,
                    normed: Kern::Normed,
                    residual: Kern::ResidualMatvec,
                },
                FoldPattern::ResidualNorm {
                    norm: K::RmsNorm,
                    add: K::Add,
                    kernel: Kern::Residual,
                },
                FoldPattern::Gated {
                    mul: K::Mul,
                    activations: &[GatedKernel {
                        activation: K::Gelu,
                        split: Kern::GeluSplit,
                        fused: Kern::GeluFused,
                    }],
                },
            ],
            &[
                FoldPattern::NormedRope {
                    rope: K::RopeAppend,
                    gain_norm: K::RmsNorm,
                    unit_norm: K::RmsNormUnit,
                    kernel: Kern::NormedRope,
                },
                FoldPattern::NormAddScale {
                    scale: K::ScalarWeightMul,
                    add: K::Add,
                    norm: K::RmsNorm,
                    kernel: Kern::NormAddScale,
                },
                FoldPattern::ExpertGated {
                    act: K::ExpertGatedAct,
                    matmul: K::ExpertMatmul,
                    kernel: Kern::ExpertGated,
                },
                FoldPattern::ExpertCombined {
                    combine: K::ExpertCombine,
                    unsort: K::ExpertUnsort,
                    matmul: K::ExpertMatmul,
                    kernel: Kern::ExpertCombined,
                },
                FoldPattern::Route {
                    top_k: K::RouteTopK,
                    sort: K::RouteArgsort,
                    pre: K::RouteSoftmax,
                    gather: K::RouteGatherScores,
                    tail: &[
                        &[K::RouteScale],
                        &[K::RouteSoftmax, K::RouteRenorm],
                        &[K::RouteExpertScale],
                    ],
                    kernel: Kern::Route,
                },
            ],
            &[FoldPattern::Encoded {
                writer: K::RopeAppend,
                encode: K::KvEncode,
                kernel: Kern::Encoded,
            }],
        ]
    };

    const TABLE: FusionTable<Kern> = FusionTable {
        counted: &[],
        sweeps: SWEEPS,
    };

    const SPLIT: ModelFoldFacts = ModelFoldFacts {
        fold_projections: false,
        normed_matvecs: false,
    };

    const ADD: ArchOp = SubOp::Elementwise(EwKind::Add);
    const MUL: ArchOp = SubOp::Elementwise(EwKind::Mul);
    const RMS: ArchOp = SubOp::RmsNorm {
        eps: 1e-6,
        gain: GainConvention::Scale,
    };

    fn gemm(n: u32, inputs: Vec<InputRef>) -> OpDesc {
        let weight = GemmWeight::Dense;
        op(SubOp::MatmulTile { n, weight }, 1, inputs)
    }

    fn op(op: ArchOp, m: u32, inputs: Vec<InputRef>) -> OpDesc {
        OpDesc { op, m, inputs }
    }

    fn norm(inputs: Vec<InputRef>) -> OpDesc {
        op(RMS, 1, inputs)
    }

    fn weights(n: usize) -> Vec<SourceBinding> {
        (0..n as u32)
            .map(|id| SourceBinding::Weight { id, index: None })
            .collect()
    }

    /// Fold `ops` over `sources`; returns the slot each op's step writes, and the folds.
    fn fold(
        sources: &[(u32, u32)],
        bindings: Vec<SourceBinding>,
        ops: Vec<OpDesc>,
        tiles: &[(usize, u32)],
        table: &FusionTable<Kern>,
        model: ModelFoldFacts,
    ) -> (Vec<SlotId>, TapeFolds<Kern>) {
        let mut op_tiles = vec![None; ops.len()];
        for &(j, t) in tiles {
            op_tiles[j] = Some((t, 0));
        }
        let op_expansion = vec![None; ops.len()];
        let lowered = LoweredDecode {
            input: LoweringInput {
                sources: sources
                    .iter()
                    .map(|&(rows, cols)| SourceShape { rows, cols })
                    .collect(),
                result: ops.len() - 1,
                ops,
            },
            bindings,
            op_tiles,
            norm_gain_add_tiles: Default::default(),
            op_expansion,
        };
        fold_lowered(&lowered, table, model)
    }

    /// Fold `lowered`; returns the slot each op's step writes, and the folds.
    fn fold_lowered(
        lowered: &LoweredDecode,
        table: &FusionTable<Kern>,
        model: ModelFoldFacts,
    ) -> (Vec<SlotId>, TapeFolds<Kern>) {
        let graph = lower_region(&lowered.input, std::num::NonZeroU32::MAX);
        let tape = lower_dag_to_tape(&ValidatedGraph::new(&graph).expect("a valid fixture"));
        let slots = tape
            .instrs()
            .iter()
            .filter_map(|i| match i {
                Instr::Compute { writes, .. } => Some(*writes),
                _ => None,
            })
            .collect();
        let folds = fold_tape(&graph, &tape, lowered, table, model).expect("the fixture folds");
        (slots, folds)
    }

    #[test]
    fn a_norm_only_matvecs_read_folds_into_each_and_its_add_into_the_delta_matvec() {
        let ops = |reader: OpDesc| {
            vec![
                gemm(64, vec![Ext(0), Ext(1)]),
                op(ADD, 1, vec![Op(0), Ext(0)]),
                norm(vec![Op(1), Ext(2)]),
                gemm(64, vec![Op(2), Ext(3)]),
                reader,
                op(MUL, 1, vec![Op(3), Op(4)]),
            ]
        };
        let matvec = || gemm(64, vec![Op(2), Ext(4)]);
        let src = [(1, 64), (64, 64), (1, 64), (64, 64), (64, 64)];
        let normed = ModelFoldFacts {
            fold_projections: false,
            normed_matvecs: true,
        };
        let (s, f) = fold(&src, weights(5), ops(matvec()), &[], &TABLE, normed);
        // Absorbed into its last reader, where the input it reads is last read.
        assert_eq!(f.role(s[2]), StepRole::Absorbed { into: s[4] });
        assert_eq!(f.role(s[1]), StepRole::Epilogue { of: s[0] });
        let residual = Fusion {
            kernel: Kern::ResidualMatvec,
            shape: FusedShape::ResidualMatvec { add: s[1] },
        };
        assert_eq!(f.role(s[0]), StepRole::Drives(&residual));
        let reads = Fusion {
            kernel: Kern::Normed,
            shape: FusedShape::NormedMatvec { norm: s[2] },
        };
        assert_eq!(f.driven(s[3]), [reads]);
        assert_eq!(f.driven(s[4]), [reads]);
        // A model whose matvecs do not normalize, or a norm another kind of step reads: the add
        // folds into the norm instead.
        let residual_norm = |f: &TapeFolds<Kern>, s: &[SlotId]| {
            assert_eq!(f.role(s[1]), StepRole::Absorbed { into: s[2] });
            assert!(f.driven(s[0]).is_empty() && f.driven(s[3]).is_empty());
        };
        let (s, f) = fold(&src, weights(5), ops(matvec()), &[], &TABLE, SPLIT);
        residual_norm(&f, &s);
        let other = op(ADD, 1, vec![Op(2), Op(3)]);
        let (s, f) = fold(&src, weights(5), ops(other), &[], &TABLE, normed);
        residual_norm(&f, &s);
    }

    #[test]
    fn a_residual_add_folds_into_its_first_reader_in_canonical_order() {
        let ops = || {
            vec![
                gemm(64, vec![Ext(0), Ext(1)]),
                gemm(64, vec![Ext(0), Ext(1)]),
                op(ADD, 1, vec![Op(0), Op(1)]),
                norm(vec![Op(2), Ext(2)]),
                norm(vec![Op(2), Ext(3)]),
                op(MUL, 1, vec![Op(3), Op(4)]),
            ]
        };
        let src = [(1, 64), (64, 64), (1, 64), (1, 64)];
        // No canonical order: the first reader on the tape wins.
        let (s, f) = fold(&src, weights(4), ops(), &[], &TABLE, SPLIT);
        assert_eq!(f.role(s[2]), StepRole::Absorbed { into: s[3] });
        assert_eq!(f.role(s[4]), StepRole::Kept);
        // The later reader comes first in canonical order: it wins, and the tape order does not.
        let (s, f) = fold(&src, weights(4), ops(), &[(3, 9), (4, 4)], &TABLE, SPLIT);
        assert_eq!(f.role(s[2]), StepRole::Absorbed { into: s[4] });
        assert_eq!(f.role(s[3]), StepRole::Kept);
        let residual = Fusion {
            kernel: Kern::Residual,
            shape: FusedShape::ResidualNorm { add: s[2] },
        };
        assert_eq!(f.role(s[4]), StepRole::Drives(&residual));
    }

    #[test]
    fn a_routing_folds_into_its_top_k_with_its_scores_as_epilogues() {
        use crate::subtile_ir::{NumExperts, RouterBundle, TopK};
        use std::num::NonZeroU32;
        let experts = NumExperts::new(NonZeroU32::new(16).expect("16 experts"));
        let k = TopK::new(NonZeroU32::new(2).expect("top 2"));
        let logits = |router| SubOp::RouterLogits { experts, router };
        let top_k = SubOp::RouteTopK { k };
        let src = [(1, 64), (16, 64)];
        // Gemma: sort, top-k, gather, scale, softmax, per-expert scale.
        let gemma = RouterBundle::Gemma;
        let ops = vec![
            op(logits(gemma), 1, vec![Ext(0), Ext(1)]),
            op(SubOp::RouteArgsort, 1, vec![Op(0)]),
            op(top_k, 1, vec![Op(1)]),
            op(SubOp::RouteGatherScores, 1, vec![Op(0), Op(2)]),
            op(SubOp::RouteScale { scale: 0.125 }, 1, vec![Op(3)]),
            op(SubOp::RouteSoftmax, 1, vec![Op(4)]),
            op(
                SubOp::RouteExpertScale { router: gemma },
                1,
                vec![Op(5), Op(2), Ext(1)],
            ),
            op(ADD, 1, vec![Op(6), Op(2)]),
        ];
        let (s, f) = fold(&src, weights(2), ops, &[], &TABLE, SPLIT);
        assert_eq!(f.role(s[1]), StepRole::Absorbed { into: s[2] });
        for e in [3, 4, 5, 6] {
            assert_eq!(f.role(s[e]), StepRole::Epilogue { of: s[2] }, "step {e}");
        }
        let route = Fusion {
            kernel: Kern::Route,
            shape: FusedShape::Route {
                sort: s[1],
                pre: None,
                gather: s[3],
                tail: [Some(s[4]), Some(s[5]), Some(s[6])],
            },
        };
        assert_eq!(f.role(s[2]), StepRole::Drives(&route));
        // Qwen's shared-expert router: a softmax over every expert first, which the sort and the
        // gather alone read; the scores then renormalize.
        let shared = RouterBundle::SharedFused;
        let ops = vec![
            op(logits(shared), 1, vec![Ext(0), Ext(1)]),
            op(SubOp::RouteSoftmax, 1, vec![Op(0)]),
            op(SubOp::RouteArgsort, 1, vec![Op(1)]),
            op(top_k, 1, vec![Op(2)]),
            op(SubOp::RouteGatherScores, 1, vec![Op(1), Op(3)]),
            op(SubOp::RouteRenorm, 1, vec![Op(4)]),
            op(ADD, 1, vec![Op(5), Op(3)]),
        ];
        let (s, f) = fold(&src, weights(2), ops, &[], &TABLE, SPLIT);
        assert_eq!(f.role(s[1]), StepRole::Absorbed { into: s[3] });
        assert_eq!(f.role(s[2]), StepRole::Absorbed { into: s[3] });
        let route = Fusion {
            kernel: Kern::Route,
            shape: FusedShape::Route {
                sort: s[2],
                pre: Some(s[1]),
                gather: s[4],
                tail: [None, Some(s[5]), None],
            },
        };
        assert_eq!(f.role(s[3]), StepRole::Drives(&route));
    }

    #[test]
    fn expert_projections_fold_into_their_activation_and_combine_only_when_read_once() {
        use crate::lower::ExpertQuant;
        use crate::subtile_ir::{ExpertBundle, ExpertProj, GatedAct, SharedExpertBound, TopK};
        let matmul = |proj, n, inputs| {
            let k = TopK::new(std::num::NonZeroU32::new(2).expect("two experts"));
            let quant = ExpertQuant::declared(64, 4);
            let bundle = ExpertBundle::SwitchGlu;
            op(
                SubOp::ExpertMatmul {
                    proj,
                    n,
                    k,
                    quant,
                    bundle,
                },
                1,
                inputs,
            )
        };
        let ops = |gate_read_again: bool| {
            let (gate, up, down) = (ExpertProj::Gate, ExpertProj::Up, ExpertProj::Down);
            let act = SubOp::ExpertGatedAct {
                act: GatedAct::Gelu,
            };
            let combine = SubOp::ExpertCombine {
                hidden: 64,
                shared: SharedExpertBound(None),
            };
            let mut v = vec![
                matmul(gate, 32, vec![Ext(0), Ext(1), Ext(2)]),
                matmul(up, 32, vec![Ext(0), Ext(1), Ext(3)]),
                op(act, 1, vec![Op(0), Op(1)]),
                matmul(down, 64, vec![Op(2), Ext(1), Ext(4)]),
                op(SubOp::ExpertUnsort, 1, vec![Op(3), Ext(1)]),
                op(combine, 1, vec![Op(4), Ext(5)]),
            ];
            if gate_read_again {
                v.push(op(ADD, 1, vec![Op(5), Op(0)]));
            }
            v
        };
        let src = [(1, 64), (1, 2), (32, 64), (32, 64), (64, 32), (1, 2)];
        let (s, f) = fold(&src, weights(6), ops(false), &[], &TABLE, SPLIT);
        assert_eq!(
            f.absorbed().collect::<Vec<_>>(),
            [(s[0], s[2]), (s[1], s[2]), (s[3], s[5]), (s[4], s[5])]
        );
        let gated = Fusion {
            kernel: Kern::ExpertGated,
            shape: FusedShape::ExpertGated {
                gate: s[0],
                up: s[1],
            },
        };
        let combined = Fusion {
            kernel: Kern::ExpertCombined,
            shape: FusedShape::ExpertCombined {
                unsort: s[4],
                down: s[3],
            },
        };
        assert_eq!(f.role(s[2]), StepRole::Drives(&gated));
        assert_eq!(f.role(s[5]), StepRole::Drives(&combined));
        // A gate another step reads stays its own step, and so does the up it pairs with.
        let (s, f) = fold(&src, weights(6), ops(true), &[], &TABLE, SPLIT);
        assert_eq!(f.role(s[0]), StepRole::Kept);
        assert_eq!(f.role(s[1]), StepRole::Kept);
        assert_eq!(f.role(s[2]), StepRole::Kept);
        assert_eq!(f.role(s[5]), StepRole::Drives(&combined));
    }

    #[test]
    fn projections_fold_into_a_gated_activation_only_when_the_model_says_so() {
        let ops = || {
            vec![
                gemm(64, vec![Ext(0), Ext(1)]),
                gemm(64, vec![Ext(0), Ext(2)]),
                op(SubOp::Elementwise(EwKind::Gelu), 1, vec![Op(0)]),
                op(MUL, 1, vec![Op(2), Op(1)]),
                gemm(64, vec![Op(3), Ext(3)]),
            ]
        };
        let src = [(1, 64), (64, 64), (64, 64), (64, 64)];
        let (s, split) = fold(&src, weights(4), ops(), &[], &TABLE, SPLIT);
        let fused = ModelFoldFacts {
            fold_projections: true,
            normed_matvecs: false,
        };
        let (_, joined) = fold(&src, weights(4), ops(), &[], &TABLE, fused);
        assert_eq!(split.absorbed().collect::<Vec<_>>(), [(s[2], s[3])]);
        assert_eq!(
            joined.absorbed().collect::<Vec<_>>(),
            [(s[0], s[3]), (s[1], s[3]), (s[2], s[3])]
        );
        let kernel = |f: &TapeFolds<Kern>| match f.role(s[3]) {
            StepRole::Drives(f) => Some(f.kernel),
            _ => None,
        };
        assert_eq!(kernel(&split), Some(Kern::GeluSplit));
        assert_eq!(kernel(&joined), Some(Kern::GeluFused));
    }

    #[test]
    fn a_centred_norm_writes_its_bias_as_an_epilogue_only_when_the_norm_has_one_reader() {
        let ops = |extra_reader: bool| {
            let mut v = vec![
                gemm(64, vec![Ext(0), Ext(1)]),
                op(SubOp::Mean, 1, vec![Op(0)]),
                op(SubOp::Elementwise(EwKind::Sub), 1, vec![Op(0), Op(1)]),
                norm(vec![Op(2), Ext(2)]),
                op(SubOp::Elementwise(EwKind::BiasAdd), 1, vec![Op(3), Ext(3)]),
            ];
            let tail = if extra_reader {
                op(MUL, 1, vec![Op(4), Op(3)])
            } else {
                gemm(64, vec![Op(4), Ext(1)])
            };
            v.push(tail);
            v
        };
        let src = [(1, 64), (64, 64), (1, 64), (1, 64)];
        let (s, f) = fold(&src, weights(4), ops(false), &[], &TABLE, SPLIT);
        assert_eq!(f.role(s[4]), StepRole::Epilogue { of: s[3] });
        let centred = |kernel, bias| Fusion {
            kernel,
            shape: FusedShape::CentredNorm {
                sub: s[2],
                mean: s[1],
                bias,
            },
        };
        let biased = centred(Kern::CentredBiased, Some(s[4]));
        assert_eq!(f.role(s[3]), StepRole::Drives(&biased));
        // The colourer sees the sub and the mean absorbed, and the bias as a step of its own.
        let facts = f.fold_facts();
        assert_eq!(
            facts.absorbed.into_iter().collect::<Vec<_>>(),
            [(s[1], s[3]), (s[2], s[3])]
        );
        let (s, f) = fold(&src, weights(4), ops(true), &[], &TABLE, SPLIT);
        assert_eq!(f.role(s[4]), StepRole::Kept);
        let plain = centred(Kern::Centred, None);
        assert_eq!(f.role(s[3]), StepRole::Drives(&plain));
    }

    /// Q/K/V projections; per-head gain-normed Q and K back to rows; a unit-normed V; the query
    /// rotation and the K/V rope. `tail` reads the rope and the rotation, and the V norm when
    /// `v_read_again` holds.
    fn rope_ops(v_read_again: bool) -> Vec<OpDesc> {
        let view = |heads, m, cols, x| {
            let rows = RowScale::Times(std::num::NonZeroU32::new(heads).unwrap());
            op(SubOp::Reshape { rows, cols }, m, vec![Op(x)])
        };
        let head_dim = HeadDim::new(64);
        let mut v = vec![
            gemm(128, vec![Ext(0), Ext(1)]),
            gemm(128, vec![Ext(0), Ext(2)]),
            gemm(128, vec![Ext(0), Ext(3)]),
            view(2, 2, 64, 0),
            op(RMS, 2, vec![Op(3), Ext(4)]),
            view(1, 1, 128, 4),
            op(SubOp::rope_rotate(head_dim), 1, vec![Op(5), Ext(6), Ext(7)]),
            view(2, 2, 64, 1),
            op(RMS, 2, vec![Op(7), Ext(5)]),
            view(1, 1, 128, 8),
            op(SubOp::RmsNormUnit { eps: 1e-6 }, 1, vec![Op(2)]),
            op(
                SubOp::rope_append(head_dim, 0, AttnMask::Causal, RopeFormTag::NeoX),
                1,
                vec![Op(9), Ext(6), Ext(7), Op(10), Ext(8), Ext(9)],
            ),
            op(ADD, 1, vec![Op(6), Op(11)]),
        ];
        if v_read_again {
            v.push(op(MUL, 1, vec![Op(12), Op(10)]));
        }
        v
    }

    fn rope_fold(v_read_again: bool, table: &FusionTable<Kern>) -> (Vec<SlotId>, TapeFolds<Kern>) {
        let src = [
            (1, 128),
            (128, 128),
            (128, 128),
            (128, 128),
            (1, 64),
            (1, 64),
            (1, 64),
            (1, 64),
            (8, 128),
            (8, 128),
        ];
        let mut bindings = weights(6);
        bindings[0] = SourceBinding::EmbeddedHidden;
        bindings.extend([
            SourceBinding::Cos { local: false },
            SourceBinding::Sin { local: false },
            SourceBinding::PrefixK { layer: 0 },
            SourceBinding::PrefixV { layer: 0 },
        ]);
        fold(&src, bindings, rope_ops(v_read_again), &[], table, SPLIT)
    }

    #[test]
    fn a_rope_folds_its_norms_only_when_its_v_is_a_single_reader_unit_norm() {
        let (s, f) = rope_fold(false, &TABLE);
        let normed = Fusion {
            kernel: Kern::NormedRope,
            shape: FusedShape::NormedRope {
                // The per-head views survive as the raw operands; the gains are the norms' weights.
                q: s[3],
                k: s[7],
                v: s[2],
                q_gain: Some(StepOperand {
                    step: s[4],
                    operand: OperandIx(1),
                }),
                k_gain: Some(StepOperand {
                    step: s[8],
                    operand: OperandIx(1),
                }),
            },
        };
        assert_eq!(f.role(s[11]), StepRole::Drives(&normed));
        for folded in [4, 5, 6, 8, 9, 10] {
            assert_eq!(f.role(s[folded]), StepRole::Absorbed { into: s[11] });
        }
        assert_eq!(
            f.fold_facts().normed_rope.into_iter().collect::<Vec<_>>(),
            [s[11]]
        );

        // A second reader of the unit norm blocks the fold; the rotation still pairs with the rope.
        let (s, f) = rope_fold(true, &TABLE);
        let paired = Fusion {
            kernel: Kern::Rope,
            shape: FusedShape::Sibling { sibling: s[6] },
        };
        assert_eq!(f.role(s[11]), StepRole::Drives(&paired));
        assert_eq!(f.role(s[10]), StepRole::Kept);
        assert!(f.fold_facts().normed_rope.is_empty());

        // Unless the table declares that reader's operand is not dataflow.
        const V_NOT_READ: FusionTable<Kern> = FusionTable {
            counted: &[CountedOperands {
                kind: SubOpKind::Mul,
                before: OperandIx(1),
            }],
            sweeps: SWEEPS,
        };
        let (s, f) = rope_fold(true, &V_NOT_READ);
        assert_eq!(f.role(s[10]), StepRole::Absorbed { into: s[11] });
    }

    /// A coded KV writer folds the codec's K and V encodes after it into the command its own
    /// fold made — and only encodes of its own expansion.
    #[test]
    fn a_kv_writer_folds_its_own_codec_encodes_into_its_command() {
        let mut input = one_layer_input_shaped(256, 64, 512, 64);
        input.ops.iter_mut().for_each(|od| od.m = 1);
        let lowered = front_end_lowered(input);
        let mut l = expand_kv_codec(&lowered, &TURBOQUANT_SHAPED_CODEC).expect("expands");
        let at = |name| l.input.ops.iter().position(|od| od.op.name() == name);
        let (rotate, writer) = (at("RopeRotate").unwrap(), at("RopeAppend").unwrap());
        let (k, v) = (writer + 1, writer + 2);
        let (s, f) = fold_lowered(&l, &TABLE, SPLIT);
        let paired = Fusion {
            kernel: Kern::Rope,
            shape: FusedShape::Sibling { sibling: s[rotate] },
        };
        let encoded = Fusion {
            kernel: Kern::Encoded,
            shape: FusedShape::Encoded { k: s[k], v: s[v] },
        };
        assert_eq!(f.driven(s[writer]), [paired, encoded]);
        assert_eq!(f.role(s[writer]), StepRole::Drives(&encoded));
        for e in [k, v] {
            assert_eq!(f.role(s[e]), StepRole::Absorbed { into: s[writer] });
        }

        // A V encode some other construct expanded is not the writer's: neither encode folds.
        let elsewhere = ExpansionId(u32::MAX);
        l.op_expansion[v] = Some(Expansion {
            id: elsewhere,
            guard: None,
        });
        let (s, f) = fold_lowered(&l, &TABLE, SPLIT);
        assert_eq!(f.driven(s[writer]), [paired]);
        for e in [k, v] {
            assert_eq!(f.role(s[e]), StepRole::Kept);
        }
    }

    #[test]
    fn a_scaled_norm_add_folds_only_with_a_weight_to_scale_by() {
        let ops = || {
            vec![
                gemm(64, vec![Ext(0), Ext(1)]),
                norm(vec![Op(0), Ext(2)]),
                op(ADD, 1, vec![Op(1), Ext(0)]),
                op(SubOp::ScalarWeightMul, 1, vec![Op(2), Ext(3)]),
                gemm(64, vec![Op(3), Ext(1)]),
            ]
        };
        let src = [(1, 64), (64, 64), (1, 64), (1, 1)];
        let mut bindings = weights(4);
        bindings[0] = SourceBinding::EmbeddedHidden;
        let (s, f) = fold(&src, bindings.clone(), ops(), &[], &TABLE, SPLIT);
        let at = |j: usize, k| StepOperand {
            step: s[j],
            operand: OperandIx(k),
        };
        let scaled = Fusion {
            kernel: Kern::NormAddScale,
            shape: FusedShape::NormAddScale {
                add: s[2],
                norm: s[1],
                delta: s[0],
                residual: at(2, 1),
                gain: at(1, 1),
                scale: at(3, 1),
            },
        };
        assert_eq!(f.role(s[3]), StepRole::Drives(&scaled));
        assert_eq!(f.role(s[1]), StepRole::Absorbed { into: s[3] });
        assert_eq!(f.role(s[2]), StepRole::Absorbed { into: s[3] });
        // The scale bound to something other than a weight: nothing folds.
        bindings[3] = SourceBinding::Cos { local: false };
        let (s, f) = fold(&src, bindings, ops(), &[], &TABLE, SPLIT);
        assert_eq!(f.role(s[3]), StepRole::Kept);
        assert_eq!(f.absorbed().count(), 0);
    }
}
