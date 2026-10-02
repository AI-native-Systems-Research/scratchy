// SPDX-License-Identifier: Apache-2.0
//! THE TAPE COLOURER — which arena buffer each tape step's output lives in.
//!
//! The tape mints a fresh SSA [`SlotId`] per `Compute`. A target realises those slots in an arena
//! of COLOURS, and two slots may share a colour only if their live ranges are disjoint. A target's
//! kernels add constraints the tape cannot express: a kernel that writes over an operand needs that
//! operand's colour, a fused command writes where the step it absorbed lived, and the embedded
//! hidden IS colour 0. Those are target facts, handed in as a [`ColourFacts`] table and a
//! [`FoldFacts`] record. The linear scan that honours them is the same for every target.
//!
//! The rules:
//! - order = tape order, and a step's position is its level: a buffer whose last reader is at
//!   position `L` is reusable only from `L + 1` on;
//! - an output that shares storage with an upstream step takes that owner's colour (chains
//!   resolve), and every read of it extends the OWNER's live range;
//! - a step reading the colour-zero source at its pinned operand writes that buffer: colour 0;
//! - free colours return to a pool keyed by the output's `(rows, cols)` and are handed out
//!   smallest first;
//! - the inputs of a step that reads ACROSS its input stay live through that step;
//! - the result's colour is never reused;
//! - a two-output step's second buffer gets its own colour, never reused;
//! - the finished colouring is checked: two non-aliased holders of one colour must be live at
//!   disjoint times.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::handoff::SourceBinding;
use crate::subtile_ir::{SubOp, SubtileIR, TensorId};
use crate::subtile_tape::{SlotId, SubtileTape};
use crate::tape_steps::{Operand, OperandIx, StepPos, TapeReadError, read_steps};

/// An arena colour: the buffer a slot is realised in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Colour(u32);

impl Colour {
    /// The colour-zero source's buffer.
    pub const ZERO: Self = Self(0);

    pub const fn index(self) -> u32 {
        self.0
    }
}

/// How many colours a colouring uses: colours `0..count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColourCount(u32);

impl ColourCount {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Where a step's output buffer lives relative to its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputAlias {
    /// A fresh buffer.
    Fresh,
    /// The kernel writes over this operand, or publishes a view of it — unconditionally.
    Operand(OperandIx),
    /// A norm a fold made absorb other steps writes over the first absorbed step's
    /// [`ColourRule::delta_when_absorbed`] operand; a fresh buffer when that step declares none.
    AbsorbedDelta,
    /// The identity when the op's constant scale is exactly 1.0 — then it IS this operand.
    UnitScale(OperandIx),
    /// In place on this operand. Inside a normed-rope fold group the operand sits behind
    /// views and norms, and the alias peels back through every [`Peel::Through`] producer to
    /// the raw projection.
    PeeledWhenNormed(OperandIx),
}

/// Whether a normed rope's alias peels back through this op, and via which operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Peel {
    Stop,
    Through(OperandIx),
}

/// How long a step's inputs must stay live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reads {
    /// Until the step's position.
    Pointwise,
    /// Through the step's own def: the kernel reads across its whole input.
    Across,
}

/// How many buffers a step writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outputs {
    One,
    Two,
}

/// Where a step's output lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// In an arena colour.
    Arena,
    /// Outside the arena — the target places it itself (routing tables, expert pair rows). It
    /// takes no colour, and it CARRIES the arena values it was computed from: they stay live until
    /// every reader of it has run, since at some bake points it is only a view of them.
    OffArena,
}

/// What a target declares about one op's buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColourRule {
    pub output: OutputAlias,
    pub reads: Reads,
    pub outputs: Outputs,
    pub residency: Residency,
    pub peel: Peel,
    /// When this operand reads the colour-zero source, the kernel writes that buffer: colour 0.
    pub pinned_on: Option<OperandIx>,
    /// When a fold absorbs this step into an [`OutputAlias::AbsorbedDelta`] norm, the norm's
    /// output is this operand.
    pub delta_when_absorbed: Option<OperandIx>,
}

impl ColourRule {
    /// A fresh output buffer read pointwise: no constraint beyond liveness.
    pub const FRESH: Self = Self {
        output: OutputAlias::Fresh,
        reads: Reads::Pointwise,
        outputs: Outputs::One,
        residency: Residency::Arena,
        peel: Peel::Stop,
        pinned_on: None,
        delta_when_absorbed: None,
    };

    pub const fn output(self, output: OutputAlias) -> Self {
        Self { output, ..self }
    }

    pub const fn off_arena(self) -> Self {
        Self {
            residency: Residency::OffArena,
            ..self
        }
    }

    pub const fn reads_across(self) -> Self {
        Self {
            reads: Reads::Across,
            ..self
        }
    }

    pub const fn two_outputs(self) -> Self {
        Self {
            outputs: Outputs::Two,
            ..self
        }
    }

    pub const fn peel_through(self, k: u8) -> Self {
        Self {
            peel: Peel::Through(OperandIx(k)),
            ..self
        }
    }

    pub const fn pinned_on(self, k: u8) -> Self {
        Self {
            pinned_on: Some(OperandIx(k)),
            ..self
        }
    }

    pub const fn delta_when_absorbed(self, k: u8) -> Self {
        Self {
            delta_when_absorbed: Some(OperandIx(k)),
            ..self
        }
    }

    /// The operand the kernel writes over UNCONDITIONALLY — the one an emitter binds as both
    /// input and output.
    pub const fn in_place_operand(self) -> Option<OperandIx> {
        match self.output {
            OutputAlias::Operand(k) => Some(k),
            _ => None,
        }
    }
}

/// A target's colouring facts: its per-op rule table and the source whose buffer IS colour 0.
pub struct ColourFacts {
    pub rule: fn(&SubOp) -> ColourRule,
    pub colour_zero: SourceBinding,
}

/// A target's fold decisions, as tape facts. The colourer reads folds; it never makes them.
#[derive(Clone, Debug, Default)]
pub struct FoldFacts {
    /// `absorbed[a] = w`: step `a` is computed inside step `w`'s fused command.
    pub absorbed: BTreeMap<SlotId, SlotId>,
    /// Steps leading a normed-rope fold; with every step absorbed into them, one group each.
    pub normed_rope: BTreeSet<SlotId>,
}

/// The colour of every slot the tape writes.
#[derive(Clone, Debug)]
pub struct TapeColouring {
    by_slot: HashMap<SlotId, (Colour, Option<Colour>)>,
    count: ColourCount,
    result: Colour,
}

impl TapeColouring {
    /// The colour of the buffer `slot` is written into.
    pub fn colour_of(&self, slot: SlotId) -> Option<Colour> {
        self.by_slot.get(&slot).map(|(c, _)| *c)
    }

    /// The colour of a two-output step's second buffer.
    pub fn second_colour_of(&self, slot: SlotId) -> Option<Colour> {
        self.by_slot.get(&slot).and_then(|(_, c)| *c)
    }

    pub fn count(&self) -> ColourCount {
        self.count
    }

    /// The colour holding the forward's result.
    pub fn result(&self) -> Colour {
        self.result
    }
}

/// A holder of a colour: the step defining it, its op, and its last use.
#[derive(Clone, Copy, Debug)]
pub struct Holder {
    pub step: StepPos,
    pub op: &'static str,
    pub last: StepPos,
}

/// Why a tape could not be coloured.
#[derive(Clone, Debug)]
pub enum ColourError {
    /// The tape could not be read as steps.
    Tape(TapeReadError),
    /// A fold fact names a slot no step writes.
    FoldNamesNoStep,
    /// A unit-scale identity was declared for an op with no constant scale.
    NotAScale { step: StepPos, op: &'static str },
    /// A normed rope's peel did not reach a raw projection within its bound.
    PeelRunaway { step: StepPos, op: &'static str },
    /// Alias chains close on themselves.
    AliasCycle { step: StepPos, op: &'static str },
    /// An alias's owner comes later on the tape, so it had no colour yet.
    AliasUncoloured { step: StepPos, op: &'static str },
    /// No step writes the graph's result.
    ResultUnwritten,
    /// The forward's result is declared off the arena, where no colour can hold it.
    OffArenaResult { step: StepPos, op: &'static str },
    /// One colour held by two live values at once.
    Overlap {
        colour: Colour,
        first: Holder,
        second: Holder,
    },
}

impl std::fmt::Display for ColourError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tape(e) => write!(f, "{e}"),
            Self::FoldNamesNoStep => write!(f, "a fold fact names a slot no step writes"),
            Self::NotAScale { step, op } => {
                write!(
                    f,
                    "{step} ({op}) is declared a unit-scale identity but has no scale"
                )
            }
            Self::PeelRunaway { step, op } => write!(f, "{step} ({op}): rope alias peel runaway"),
            Self::AliasCycle { step, op } => write!(f, "{step} ({op}): alias cycle"),
            Self::AliasUncoloured { step, op } => {
                write!(f, "{step} ({op}): alias target uncolored")
            }
            Self::ResultUnwritten => write!(f, "no step writes the forward's result"),
            Self::OffArenaResult { step, op } => {
                write!(f, "{step} ({op}) writes the forward's result off the arena")
            }
            Self::Overlap {
                colour,
                first,
                second,
            } => write!(
                f,
                "colour {} is held by {} ({}, live to {}) and {} ({}, live to {}) AT THE SAME \
                 TIME — one kernel would write over a buffer the other still owns",
                colour.0, first.step, first.op, first.last, second.step, second.op, second.last,
            ),
        }
    }
}

struct Step<'a> {
    writes: SlotId,
    op: &'a SubOp,
    rule: ColourRule,
    operands: Vec<Operand>,
    shape: (u32, u32),
    output: TensorId,
}

/// The longest chain of views and norms a normed rope peels through.
const PEEL_BOUND: usize = 8;

/// Colour the unrolled `tape` of `graph` under a target's `facts` and `folds`.
///
/// The rule table is keyed on each step's node op; `bindings` is parallel to the graph's sources.
pub fn colour_tape(
    graph: &SubtileIR,
    tape: &SubtileTape,
    bindings: &[SourceBinding],
    folds: &FoldFacts,
    facts: &ColourFacts,
) -> Result<TapeColouring, ColourError> {
    let steps = steps_of(graph, tape, facts)?;
    let n = steps.len();
    let name = |p: usize| steps[p].op.name();
    let pos = |p: usize| StepPos(p as u32);
    let in_step = |p: usize, k: OperandIx| match steps[p].operands.get(k.index()) {
        Some(Operand::Step(q)) => Some(*q),
        _ => None,
    };

    let pos_of: HashMap<SlotId, usize> = (0..n).map(|p| (steps[p].writes, p)).collect();
    let at = |s: &SlotId| pos_of.get(s).copied().ok_or(ColourError::FoldNamesNoStep);
    let mut absorbed_into: Vec<Option<usize>> = vec![None; n];
    for (a, w) in &folds.absorbed {
        absorbed_into[at(a)?] = Some(at(w)?);
    }
    let mut normed_rope: HashSet<usize> = HashSet::new();
    for w in &folds.normed_rope {
        let w = at(w)?;
        normed_rope.insert(w);
        normed_rope.extend((0..n).filter(|a| absorbed_into[*a] == Some(w)));
    }

    // Storage alias per step: `Some(q)` = step p's output IS step q's buffer.
    let alias_of = (0..n)
        .map(|p| match steps[p].rule.output {
            OutputAlias::Fresh => Ok(None),
            OutputAlias::Operand(k) => Ok(in_step(p, k)),
            // The FIRST step (tape order) a fold absorbed into this one.
            OutputAlias::AbsorbedDelta => {
                Ok((0..n).find(|a| absorbed_into[*a] == Some(p)).and_then(|a| {
                    steps[a]
                        .rule
                        .delta_when_absorbed
                        .and_then(|k| in_step(a, k))
                }))
            }
            OutputAlias::UnitScale(k) => match steps[p].op {
                SubOp::ScalarMul { scale } if *scale == 1.0 => Ok(in_step(p, k)),
                SubOp::ScalarMul { .. } => Ok(None),
                _ => Err(ColourError::NotAScale {
                    step: pos(p),
                    op: name(p),
                }),
            },
            OutputAlias::PeeledWhenNormed(k) if normed_rope.contains(&p) => {
                let mut cur = in_step(p, k);
                let mut hops = 0;
                while let Some(q) = cur {
                    let Peel::Through(via) = steps[q].rule.peel else {
                        break;
                    };
                    hops += 1;
                    if hops >= PEEL_BOUND {
                        return Err(ColourError::PeelRunaway {
                            step: pos(p),
                            op: name(p),
                        });
                    }
                    match in_step(q, via) {
                        Some(r) => cur = Some(r),
                        None => break,
                    }
                }
                Ok(cur)
            }
            OutputAlias::PeeledWhenNormed(k) => Ok(in_step(p, k)),
        })
        .collect::<Result<Vec<Option<usize>>, ColourError>>()?;

    // The step whose buffer each step's output resolves to.
    let mut owner: Vec<usize> = Vec::with_capacity(n);
    for p in 0..n {
        let mut o = p;
        let mut hops = 0;
        while let Some(q) = alias_of[o] {
            o = q;
            hops += 1;
            if hops > n {
                return Err(ColourError::AliasCycle {
                    step: pos(p),
                    op: name(p),
                });
            }
        }
        owner.push(o);
    }

    let pinned: Vec<bool> = steps
        .iter()
        .map(|s| {
            s.rule.pinned_on.is_some_and(|k| {
                matches!(s.operands.get(k.index()), Some(Operand::Source(t))
                    if bindings.get(t.index()) == Some(&facts.colour_zero))
            })
        })
        .collect();

    // The arena owners each off-arena step carries: its arena operands', and what its off-arena
    // operands carry.
    let off = |p: usize| steps[p].rule.residency == Residency::OffArena;
    let mut carried: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for p in (0..n).filter(|p| off(*p)) {
        for operand in &steps[p].operands {
            if let Operand::Step(q) = operand {
                let from = if off(*q) {
                    carried[*q].clone()
                } else {
                    BTreeSet::from([owner[*q]])
                };
                carried[p].extend(from);
            }
        }
    }

    // Last use per OWNER, at step granularity; a read of an off-arena value reads what it carries.
    let mut last_use: Vec<usize> = (0..n).collect();
    for (r, s) in steps.iter().enumerate() {
        for operand in &s.operands {
            if let Operand::Step(q) = operand {
                let own = BTreeSet::from([owner[*q]]);
                for &o in if off(*q) { &carried[*q] } else { &own } {
                    last_use[o] = last_use[o].max(r);
                }
            }
        }
    }

    let result_step = steps
        .iter()
        .rposition(|s| s.output == graph.result)
        .ok_or(ColourError::ResultUnwritten)?;
    let result_owner = owner[result_step];
    if off(result_owner) {
        return Err(ColourError::OffArenaResult {
            step: pos(result_owner),
            op: name(result_owner),
        });
    }

    // Linear scan.
    let mut active: Vec<(usize, Colour, usize)> = Vec::new(); // (last use, colour, owner)
    let mut free_by_shape: HashMap<(u32, u32), BTreeSet<Colour>> = HashMap::new();
    let mut colour_shape: HashMap<Colour, (u32, u32)> = HashMap::new();
    let mut colour: Vec<Option<Colour>> = vec![None; n];
    let mut next = Colour::ZERO.0 + 1;
    for p in (0..n).filter(|p| !off(*p)) {
        let held: HashSet<usize> = match steps[p].rule.reads {
            Reads::Across => steps[p]
                .operands
                .iter()
                .filter_map(|o| match o {
                    Operand::Step(q) => Some(owner[*q]),
                    Operand::Source(_) => None,
                })
                .collect(),
            Reads::Pointwise => HashSet::new(),
        };
        active.retain(|&(lu, c, o)| {
            if lu < p && !held.contains(&o) && o != result_owner {
                free_by_shape.entry(colour_shape[&c]).or_default().insert(c);
                false
            } else {
                true
            }
        });

        if pinned[owner[p]] || pinned[p] {
            colour[p] = Some(Colour::ZERO);
            continue;
        }
        if alias_of[p].is_some() {
            colour[p] = Some(colour[owner[p]].ok_or(ColourError::AliasUncoloured {
                step: pos(p),
                op: name(p),
            })?);
            continue;
        }
        let shape = steps[p].shape;
        let c = match free_by_shape.entry(shape).or_default().pop_first() {
            Some(c) => c,
            None => {
                let c = Colour(next);
                next += 1;
                colour_shape.insert(c, shape);
                c
            }
        };
        colour[p] = Some(c);
        active.push((last_use[p], c, p));
    }

    // The invariant the arena rests on, checked on the FINISHED colouring: two values may share a
    // colour only if their live ranges are disjoint. A colouring that overlaps two live values is
    // equally wrong in every emission from it, so no downstream comparison can see it.
    let mut by_colour: BTreeMap<Colour, Vec<(usize, usize)>> = BTreeMap::new();
    for p in 0..n {
        if alias_of[p].is_some() || pinned[p] || pinned[owner[p]] {
            continue;
        }
        if let Some(c) = colour[p]
            && c != Colour::ZERO
        {
            by_colour.entry(c).or_default().push((p, last_use[p]));
        }
    }
    if let Some((c, (a, a_last), (b, b_last))) = first_overlap(by_colour) {
        let holder = |s: usize, last: usize| Holder {
            step: pos(s),
            op: name(s),
            last: pos(last),
        };
        return Err(ColourError::Overlap {
            colour: c,
            first: holder(a, a_last),
            second: holder(b, b_last),
        });
    }

    // A second output is a distinct allocation with no tracked lifetime: never reused.
    let mut by_slot = HashMap::with_capacity(n);
    for (s, c) in steps.iter().zip(&colour) {
        let Some(c) = *c else { continue };
        let second = (s.rule.outputs == Outputs::Two).then(|| {
            next += 1;
            Colour(next - 1)
        });
        by_slot.insert(s.writes, (c, second));
    }
    Ok(TapeColouring {
        by_slot,
        count: ColourCount(next),
        result: colour[result_owner].expect("the scan colours every step"),
    })
}

/// The first colour two holders `(def, last use)` hold at once, with the two holders in def order.
type Overlap = (Colour, (usize, usize), (usize, usize));

fn first_overlap(by_colour: BTreeMap<Colour, Vec<(usize, usize)>>) -> Option<Overlap> {
    by_colour.into_iter().find_map(|(c, mut v)| {
        v.sort_unstable();
        v.windows(2)
            .find(|w| w[0].1 >= w[1].0)
            .map(|w| (c, w[0], w[1]))
    })
}

/// The tape's steps in order, each with its op, rule, operands and output shape.
fn steps_of<'a>(
    graph: &'a SubtileIR,
    tape: &SubtileTape,
    facts: &ColourFacts,
) -> Result<Vec<Step<'a>>, ColourError> {
    let steps = read_steps(graph, tape).map_err(ColourError::Tape)?;
    Ok(steps
        .into_iter()
        .map(|s| {
            let shape = graph.shape(s.output);
            Step {
                writes: s.writes,
                op: s.op,
                rule: (facts.rule)(s.op),
                operands: s.operands,
                shape: (shape.rows, shape.cols),
                output: s.output,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{ArchOp, GemmWeight, InputRef, LoweringInput, OpDesc};
    use crate::subtile_ir::{
        EwKind, GainConvention, RowScale, SourceShape, ValidatedGraph, lower_region,
    };
    use crate::subtile_tape::{Instr, lower_dag_to_tape};
    use InputRef::{Ext, Op};
    use ktir_superdsc::head_counts::HeadDim;

    /// A neutral rule table covering every pattern kind.
    fn rule(op: &SubOp) -> ColourRule {
        use OutputAlias as A;
        use SubOp as O;
        const FRESH: ColourRule = ColourRule::FRESH;
        match op {
            O::Reshape { .. } => FRESH.output(A::Operand(OperandIx(0))).peel_through(0),
            O::Elementwise(EwKind::Add) => FRESH
                .output(A::Operand(OperandIx(1)))
                .pinned_on(1)
                .delta_when_absorbed(0),
            O::Elementwise(EwKind::Gelu) => FRESH.output(A::Operand(OperandIx(0))),
            O::RmsNorm { .. } => FRESH.output(A::AbsorbedDelta).peel_through(0),
            O::ScalarMul { .. } => FRESH.output(A::UnitScale(OperandIx(0))).pinned_on(0),
            O::RopeRotate { .. } => FRESH.output(A::PeeledWhenNormed(OperandIx(0))),
            O::MatmulTile { .. } => FRESH.reads_across(),
            // A MoE block keeps everything but its weighted sum off the arena.
            O::ExpertCombine { .. } => FRESH,
            _ if matches!(op, expansion_ops!()) => FRESH.off_arena(),
            _ => FRESH,
        }
    }

    const ADD: ArchOp = SubOp::Elementwise(EwKind::Add);
    const MUL: ArchOp = SubOp::Elementwise(EwKind::Mul);
    const NORM: ArchOp = SubOp::RmsNorm {
        eps: 1e-6,
        gain: GainConvention::Scale,
    };

    const FACTS: ColourFacts = ColourFacts {
        rule,
        colour_zero: SourceBinding::EmbeddedHidden,
    };

    fn gemm(n: u32, inputs: Vec<InputRef>) -> OpDesc {
        let weight = GemmWeight::Dense;
        OpDesc {
            op: SubOp::MatmulTile { n, weight },
            m: 1,
            inputs,
        }
    }

    fn op(op: ArchOp, m: u32, inputs: Vec<InputRef>) -> OpDesc {
        OpDesc { op, m, inputs }
    }

    fn weights(n: usize) -> Vec<SourceBinding> {
        (0..n as u32)
            .map(|id| SourceBinding::Weight { id, index: None })
            .collect()
    }

    /// Colour `ops` over `sources`; the folds are built from the slot each op's step writes.
    fn colour(
        sources: &[(u32, u32)],
        bindings: Vec<SourceBinding>,
        ops: Vec<OpDesc>,
        folds: impl Fn(&[SlotId]) -> FoldFacts,
    ) -> Result<(Vec<u32>, TapeColouring), ColourError> {
        let (per_op, c) = coloured(sources, bindings, ops, folds)?;
        Ok((per_op.into_iter().map(Option::unwrap).collect(), c))
    }

    /// [`colour`], with `None` for a step off the arena.
    fn coloured(
        sources: &[(u32, u32)],
        bindings: Vec<SourceBinding>,
        ops: Vec<OpDesc>,
        folds: impl Fn(&[SlotId]) -> FoldFacts,
    ) -> Result<(Vec<Option<u32>>, TapeColouring), ColourError> {
        let input = LoweringInput {
            sources: sources
                .iter()
                .map(|&(rows, cols)| SourceShape { rows, cols })
                .collect(),
            result: ops.len() - 1,
            ops,
        };
        let graph = lower_region(&input, std::num::NonZeroU32::MAX);
        let tape = lower_dag_to_tape(&ValidatedGraph::new(&graph).expect("a valid fixture"));
        let slots: Vec<SlotId> = tape
            .instrs()
            .iter()
            .filter_map(|i| match i {
                Instr::Compute { writes, .. } => Some(*writes),
                _ => None,
            })
            .collect();
        let c = colour_tape(&graph, &tape, &bindings, &folds(&slots), &FACTS)?;
        let per_op = slots
            .iter()
            .map(|s| c.colour_of(*s).map(Colour::index))
            .collect();
        Ok((per_op, c))
    }

    fn no_folds(_: &[SlotId]) -> FoldFacts {
        FoldFacts::default()
    }

    #[test]
    fn an_in_place_kernel_shares_its_operand_colour() {
        let ops = vec![
            gemm(64, vec![Ext(0), Ext(1)]),
            op(SubOp::Elementwise(EwKind::Gelu), 1, vec![Op(0)]),
            gemm(64, vec![Op(1), Ext(1)]),
        ];
        let (c, _) = colour(&[(1, 64), (64, 64)], weights(2), ops, no_folds).unwrap();
        assert_eq!(c, [1, 1, 2]);
    }

    #[test]
    fn a_fold_makes_the_norm_write_over_the_absorbed_delta() {
        let ops = || {
            vec![
                gemm(64, vec![Ext(0), Ext(1)]),
                gemm(64, vec![Ext(0), Ext(1)]),
                op(ADD, 1, vec![Op(0), Op(1)]),
                op(NORM, 1, vec![Op(2), Ext(2)]),
                op(MUL, 1, vec![Op(3), Op(0)]),
            ]
        };
        let src = [(1, 64), (64, 64), (1, 64)];
        let (unfolded, _) = colour(&src, weights(3), ops(), no_folds).unwrap();
        let (folded, _) = colour(&src, weights(3), ops(), |s| FoldFacts {
            absorbed: [(s[2], s[3])].into(),
            ..FoldFacts::default()
        })
        .unwrap();
        // The add writes over its residual either way; only the fold moves the norm.
        assert_eq!(unfolded[2], unfolded[1]);
        assert_eq!(folded[2], folded[1]);
        assert_ne!(unfolded[3], unfolded[0]);
        assert_eq!(folded[3], folded[0]);
    }

    #[test]
    fn a_scale_is_an_alias_only_at_exactly_one() {
        let ops = vec![
            gemm(64, vec![Ext(0), Ext(1)]),
            op(SubOp::ScalarMul { scale: 1.0 }, 1, vec![Op(0)]),
            op(SubOp::ScalarMul { scale: 0.5 }, 1, vec![Op(0)]),
            op(MUL, 1, vec![Op(1), Op(2)]),
        ];
        let (c, _) = colour(&[(1, 64), (64, 64)], weights(2), ops, no_folds).unwrap();
        assert_eq!(c[1], c[0]);
        assert_ne!(c[2], c[0]);
    }

    #[test]
    fn writers_over_the_colour_zero_source_are_colour_zero() {
        let ops = vec![
            gemm(64, vec![Ext(0), Ext(1)]),
            op(ADD, 1, vec![Op(0), Ext(0)]),
            gemm(64, vec![Op(1), Ext(1)]),
            op(ADD, 1, vec![Op(2), Op(1)]),
            gemm(64, vec![Op(3), Ext(1)]),
        ];
        let bindings = vec![
            SourceBinding::EmbeddedHidden,
            SourceBinding::Weight { id: 0, index: None },
        ];
        let (c, _) = colour(&[(1, 64), (64, 64)], bindings, ops, no_folds).unwrap();
        assert_eq!((c[1], c[3]), (0, 0));
        assert_ne!(c[0], 0);
    }

    #[test]
    fn a_free_colour_is_reused_only_by_its_shape_and_only_after_its_last_read() {
        let ops = vec![
            gemm(64, vec![Ext(0), Ext(1)]),
            gemm(64, vec![Op(0), Ext(1)]),
            gemm(128, vec![Op(1), Ext(2)]),
            gemm(64, vec![Op(2), Ext(3)]),
        ];
        let src = [(1, 64), (64, 64), (64, 128), (128, 64)];
        let (c, colouring) = colour(&src, weights(4), ops, no_folds).unwrap();
        // Op 1 reads op 0 last, so it cannot take op 0's colour; op 2 is a different shape; op 3
        // takes the smallest free colour of its own shape.
        assert_eq!(c, [1, 2, 3, 1]);
        assert_eq!(colouring.count().get(), 4);
        assert_eq!(colouring.result().index(), 1);
    }

    #[test]
    fn a_normed_rope_peels_to_the_raw_projection_and_an_unfused_one_does_not() {
        let ops = || {
            let k = |k| std::num::NonZeroU32::new(k).unwrap();
            let view = |rows, cols| SubOp::Reshape { rows, cols };
            vec![
                gemm(128, vec![Ext(0), Ext(1)]),
                op(view(RowScale::Times(k(2)), 64), 2, vec![Op(0)]),
                op(NORM, 2, vec![Op(1), Ext(2)]),
                op(view(RowScale::Over(k(2)), 128), 1, vec![Op(2)]),
                op(
                    SubOp::rope_rotate(HeadDim::new(64)),
                    1,
                    vec![Op(3), Ext(3), Ext(4)],
                ),
            ]
        };
        let src = [(1, 128), (128, 128), (1, 64), (1, 64), (1, 64)];
        let (direct, _) = colour(&src, weights(5), ops(), no_folds).unwrap();
        let (peeled, _) = colour(&src, weights(5), ops(), |s| FoldFacts {
            normed_rope: [s[4]].into(),
            ..FoldFacts::default()
        })
        .unwrap();
        assert_eq!(direct[4], direct[2]);
        assert_eq!(peeled[4], peeled[0]);
        assert_ne!(peeled[0], peeled[2]);
    }

    #[test]
    fn a_colour_held_by_two_live_values_is_refused() {
        let held = |v: Vec<(usize, usize)>| BTreeMap::from([(Colour(1), v)]);
        assert_eq!(first_overlap(held(vec![(0, 1), (2, 3)])), None);
        assert_eq!(
            first_overlap(held(vec![(2, 5), (0, 2)])),
            Some((Colour(1), (0, 2), (2, 5)))
        );
    }

    /// A routed block over `x = gemm(..)`: logits, top-k, scores, sorted pair rows, one expert
    /// projection, the weighted sum, then a projection of it. `x` is read by the router and the
    /// sort only — off-arena steps both.
    fn moe_block(tail: bool) -> (Vec<(u32, u32)>, Vec<OpDesc>) {
        use crate::lower::ExpertQuant;
        use crate::subtile_ir::{
            ExpertBundle, ExpertProj, NumExperts, RouterBundle, SharedExpertBound, TopK,
        };
        let nz = |n| std::num::NonZeroU32::new(n).unwrap();
        let (experts, k, bundle) = (
            NumExperts::new(nz(4)),
            TopK::new(nz(2)),
            ExpertBundle::Fused,
        );
        let quant = ExpertQuant::declared(64, 4);
        let proj = ExpertProj::Gate;
        let router = RouterBundle::Fused;
        let (sort, combine) = (
            SubOp::ExpertSort { experts, k, bundle },
            SubOp::ExpertCombine {
                hidden: 64,
                shared: SharedExpertBound(None),
            },
        );
        let mut ops = vec![
            gemm(64, vec![Ext(0), Ext(1)]),
            op(
                SubOp::RouterLogits { experts, router },
                1,
                vec![Op(0), Ext(2)],
            ),
            op(SubOp::RouteArgsort, 1, vec![Op(1)]),
            op(SubOp::RouteTopK { k }, 1, vec![Op(2)]),
            op(SubOp::RouteGatherScores, 1, vec![Op(1), Op(3)]),
            op(sort, 1, vec![Op(0), Op(3)]),
            op(
                SubOp::ExpertMatmul {
                    proj,
                    n: 64,
                    k,
                    quant,
                    bundle,
                },
                1,
                vec![Op(5), Op(5), Ext(2)],
            ),
            op(combine, 1, vec![Op(6), Op(4)]),
        ];
        if tail {
            ops.push(gemm(64, vec![Op(7), Ext(1)]));
        }
        (vec![(1, 64), (64, 64), (1, 1)], ops)
    }

    #[test]
    fn an_off_arena_value_takes_no_colour_and_keeps_what_it_was_computed_from_live() {
        let (src, ops) = moe_block(true);
        let (c, colouring) = coloured(&src, weights(3), ops, no_folds).unwrap();
        assert_eq!(c[1..7], [None; 6]);
        // `x`'s last direct reader is the sort, but the sum reads what the sort carries: without
        // the carrier the sum would take `x`'s colour while the gathered pair rows still view it.
        assert_eq!((c[0], c[7], c[8]), (Some(1), Some(2), Some(1)));
        assert_eq!(colouring.count().get(), 3);
    }

    #[test]
    fn an_off_arena_result_is_refused() {
        let (src, mut ops) = moe_block(false);
        ops.pop();
        let refused = coloured(&src, weights(3), ops, no_folds);
        assert!(matches!(refused, Err(ColourError::OffArenaResult { .. })));
    }

    #[test]
    fn an_arena_step_aliasing_an_off_arena_value_is_refused() {
        let (src, mut ops) = moe_block(false);
        // A view over the pair rows: its buffer would be one the arena never allocated.
        let view = SubOp::Reshape {
            rows: RowScale::Times(std::num::NonZeroU32::new(1).unwrap()),
            cols: 128,
        };
        ops.push(op(view, 1, vec![Op(5)]));
        ops.push(gemm(64, vec![Op(7), Ext(1)]));
        let refused = coloured(&src, weights(3), ops, no_folds);
        assert!(matches!(refused, Err(ColourError::AliasUncoloured { .. })));
    }
}
