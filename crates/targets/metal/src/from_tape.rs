// SPDX-License-Identifier: Apache-2.0
//! METAL, LOWERED FROM THE SHARED `SubtileTape`.
//!
//! ⭐ THIS IS THE FORK POINT MOVING. Metal's existing bridge consumes `LoweredDecode` — the op
//! list one stage BEFORE `SubtileIR` — and touches neither the graph nor the tape. Spyre
//! consumes the `SubtileTape`, two stages later. This module is metal reading the SAME artifact
//! spyre reads, so "one front end" names one object rather than one crate boundary.
//!
//! ⛔ AND IT LIVES IN THE TARGET CRATE, not in `compiler/macros`. Spyre's emitter is
//! `targets/spyre/src/lower_subtile_tape_to_superdsc.rs`; this is its metal counterpart. A
//! `metal_*` module inside the shared compiler is target-specific code in a common crate, which
//! is the arrangement this whole line of work exists to remove.
//!
//! ## What the tape gives
//!
//! An [`Instr::Compute`] names a node in the graph, the slot it writes, and its positional
//! inputs as either earlier slots or graph sources. So the vocabulary to match on is [`SubOp`] —
//! `lower_region` has already resolved the arch-level ops into it.
//!
//! ## Tiling is a target fact
//!
//! The graph is built with `nb = u32::MAX` — WHOLE ops. Spyre tiles to 128-column pages
//! (`decompose_rmsnorm`, `head_tile_rope`) because its substrate requires it; metal's kernels
//! take a whole `RmsNorm` and a whole rope. Same builder, same tape, different `nb`.

use scratchy_subtile::subtile_ir::{SubOp, SubtileIR, SubtileId, TensorId};
use scratchy_subtile::subtile_tape::{
    ComputeInput, ComputeInputs, Instr, LoopBound, LoopVarId, SlotId, SubtileTape,
};

/// Why a tape could not be lowered to metal.
///
/// ⛔ A REFUSAL, NEVER A FALLBACK. Dropping back to the `LoweredDecode` bridge on error would
/// mean two live lowerings again, disagreeing invisibly — the state this module exists to end.
#[derive(Debug)]
pub struct TapeLoweringError(pub String);

impl std::fmt::Display for TapeLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One lowered step: which graph node, what it writes, what it reads.
///
/// ⛔ SLOTS AND SOURCES ARE DIFFERENT THINGS AND ARE NOT INTERCHANGEABLE. A `Computed` input is
/// an earlier slot in the pool; an `External` is a graph SOURCE — a weight, a cache handle —
/// resolved through the source manifest. Collapsing both to a bare `u32` is how a weight ends up
/// aliased onto an activation, so the enum survives into the emitted step.
/// ⛔ AND THEY KEEP THEIR OWN NEWTYPES. `SlotId` and `TensorId` index different arrays; both
/// erase to `u32`, so a bare integer here would let a swap typecheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepInput {
    Slot(SlotId),
    Source(TensorId),
    /// A LOOP-INDEXED source: at iteration `v` this operand is `per_layer[v]`.
    ///
    /// ⭐ THIS IS WHAT A RE-ROLLED TAPE IS FOR. The layer loop collapses to one body only if the
    /// per-layer weights stop being N separate operands and become one operand selected by the
    /// loop variable. A target that cannot express this has to unroll, which is why metal's
    /// baked tape used to materialise 328k `LayerId` literals for five llama configs.
    PerLayerSource {
        per_layer: Vec<TensorId>,
    },
}

/// A tape `Compute`, resolved against the graph and ready for opcode emission.
///
/// `source_op` is the `LoweringInput` op this step came from — where the WEIGHTS live. The tape
/// says what to compute, in what order, into which slot; it does not say what to multiply by.
#[derive(Clone, Debug)]
pub struct TapeStep {
    pub node: SubtileId,
    pub source_op: SourceOp,
    pub writes: SlotId,
    pub inputs: Vec<StepInput>,
}

/// An index into `LoweringInput::ops` — NOT a `SubtileId` and not a slot, though all three erase
/// to a machine word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SourceOp(pub usize);

/// One entry of the ROLLED program: a step, or a loop boundary.
///
/// ⭐ THE LOOP IS DATA ON THE SHARED TAPE, NOT A PATTERN EACH TARGET RE-DISCOVERS.
/// `reroll_subtile_tape` finds the repeating body ONCE, on the `SubtileTape`, before any target
/// lowers. Both targets then read the loop off it. Metal previously re-derived the same fact
/// from its own `Instruction` stream (`apply_loop_compression` + `detect_repeating_run`) — a
/// second search, over a second representation, that could disagree with the first.
#[derive(Clone, Debug)]
pub enum TapeItem {
    Step(TapeStep),
    OpenLoop {
        var: LoopVarId,
        iters: u32,
        /// How far the LAYER advances per iteration — see [`crate::tape::lowered::TapeLoop`].
        stride: u32,
    },
    CloseLoop {
        var: LoopVarId,
    },
}

/// Build the graph metal lowers from, out of the same `LoweringInput` spyre starts from.
///
/// ⛔ NO `fuse_silu_mul`, AND NOT BECAUSE IT IS HARD. That pass DROPS ops and renumbers the
/// rest, so provenance would point at the fused list while metal's emitter indexes the original
/// — every weight off by the number of fusions before it. It is also spyre's fusion: spyre has a
/// SiluMul kernel, metal fuses gate/up in its own `fold_plan`. WHICH FUSIONS TO APPLY IS A
/// TARGET FACT; the builder underneath is the shared one.
///
/// Likewise `nb = u32::MAX` — WHOLE ops. `decompose_rmsnorm` and `head_tile_rope` exist because
/// spyre's substrate wants 128-column pages; metal's kernels take a whole RmsNorm and a whole
/// rope. Same builder, same tape, different `nb`.
pub fn graph_for_metal(input: &scratchy_subtile::lower::LoweringInput) -> SubtileIR {
    let nb = std::num::NonZeroU32::new(u32::MAX).expect("u32::MAX != 0");
    scratchy_subtile::subtile_ir::lower_region(input, nb)
}

/// Build the tape metal plays.
pub fn tape_for_metal(graph: &SubtileIR) -> Result<SubtileTape, TapeLoweringError> {
    let valid = scratchy_subtile::subtile_ir::ValidatedGraph::new(graph)
        .map_err(|e| TapeLoweringError(format!("graph is not a valid DAG: {e:?}")))?;
    Ok(scratchy_subtile::subtile_tape::lower_dag_to_tape(&valid))
}

fn resolve_inputs(inputs: &ComputeInputs) -> Result<Vec<StepInput>, TapeLoweringError> {
    inputs
        .iter()
        .map(|ci| match ci {
            // ⛔ NOT `slots[0]`. Several writers means a COLUMN-SPLIT producer, which metal has
            // no operand for — one buffer, one producer. With whole ops (nb = MAX) nothing is
            // split, so more than one writer means the graph is not what this path assumes, and
            // silently taking the first would emit a kernel reading a fraction of its input.
            ComputeInput::Computed(slots) => match slots.as_slice() {
                [one] => Ok(StepInput::Slot(*one)),
                many => Err(TapeLoweringError(format!(
                    "input has {} writers; metal binds one buffer per operand and cannot read a \
                     column-split producer",
                    many.len()
                ))),
            },
            ComputeInput::External {
                tensor, per_layer, ..
            } if per_layer.is_empty() => Ok(StepInput::Source(*tensor)),
            // Inside a re-rolled layer body: the operand is `per_layer[v]` at iteration `v`.
            // `tensor == per_layer[0]` (copy-0, the structural anchor), so a target that
            // ignored `per_layer` would silently bind LAYER 0's weights on every iteration.
            ComputeInput::External { per_layer, .. } => Ok(StepInput::PerLayerSource {
                per_layer: per_layer.clone(),
            }),
        })
        .collect()
}

/// WALK THE TAPE INTO AN ORDERED, POSSIBLY-ROLLED PROGRAM.
///
/// `AllocSlot`/`FreeSlot` carry no item — metal's slot lifetimes come from its colorer, and the
/// tape's alloc/free is the arena discipline.
///
/// ⛔ A RUNTIME-BOUND LOOP IS STILL REFUSED. `LoopBound::Runtime` is the AttnDecode KV sweep,
/// whose trip count is only known at launch; metal performs that sweep INSIDE
/// `AttentionViaCache`, so there is no metal instruction for it and emitting one per KV position
/// is not a lowering. A `Const` bound is the LAYER loop, which is exactly what metal wants.
pub fn items_of(graph: &SubtileIR, tape: &SubtileTape) -> Result<Vec<TapeItem>, TapeLoweringError> {
    // Reverse the lowering's provenance map: output tensor → the op that produced it.
    let source_of: std::collections::HashMap<TensorId, SourceOp> = graph
        .op_output
        .iter()
        .enumerate()
        .map(|(j, t)| (*t, SourceOp(j)))
        .collect();
    let cell_layers = cell_layers(graph, tape);
    let mut items = Vec::new();
    for instr in tape.instrs() {
        match instr {
            Instr::AllocSlot { .. } | Instr::FreeSlot { .. } => {}
            Instr::OpenLoop { var, bound } => match bound {
                LoopBound::Const(iters) => items.push(TapeItem::OpenLoop {
                    stride: cell_layers,
                    var: *var,
                    iters: *iters,
                }),
                LoopBound::Runtime(_) => {
                    return Err(TapeLoweringError(
                        "tape opens a RUNTIME-bound loop; that is the AttnDecode KV sweep, which \
                         metal performs inside AttentionViaCache and has no instruction for"
                            .into(),
                    ));
                }
            },
            Instr::CloseLoop { var } => items.push(TapeItem::CloseLoop { var: *var }),
            Instr::Compute {
                node,
                writes,
                inputs,
                ..
            } => {
                let out = graph.nodes[node.index()].output.tensor;
                // ⛔ NO DEFAULT. A node whose output is not any source op's output is a
                // lowering INTERMEDIATE — a decomposition's scratch tensor. Metal has no
                // weights to bind for one, and inventing a source op here would bind the
                // WRONG weights silently.
                let source_op = *source_of.get(&out).ok_or_else(|| {
                    TapeLoweringError(format!(
                        "node {} ({:?}) writes tensor {} which no source op produces — it is a \
                         lowering intermediate, and metal binds weights off the source op",
                        node.index(),
                        graph.nodes[node.index()].op,
                        out.index(),
                    ))
                })?;
                items.push(TapeItem::Step(TapeStep {
                    node: *node,
                    source_op,
                    writes: *writes,
                    inputs: resolve_inputs(inputs)?,
                }));
            }
        }
    }
    Ok(items)
}

/// The steps of a program, ignoring loop boundaries — the UNROLLED view.
///
/// Used by the callers that still want a flat order (the schedule projection below). A caller
/// that emits a loop reads [`items_of`] directly; one that flattens must say so by name.
pub fn steps_only(items: &[TapeItem]) -> Vec<&TapeStep> {
    items
        .iter()
        .filter_map(|i| match i {
            TapeItem::Step(s) => Some(s),
            _ => None,
        })
        .collect()
}

/// The layer loop, in terms of the SOURCE OPS of its first body copy.
///
/// ⭐ READ FROM THE SHARED SEARCH, NOT RE-DERIVED. `find_layer_loop` answers once, on the shared
/// tape; `reroll_subtile_tape` uses the same answer to build the rolled tape spyre lowers. This
/// translates it into the only currency metal's emitter speaks — source-op indices — so metal
/// can locate the same body in its own instruction stream WITHOUT searching that stream.
///
/// Returns `(body source ops in emission order, iterations)`.
pub fn layer_loop_source_ops(
    graph: &SubtileIR,
    tape: &SubtileTape,
) -> Result<Option<(Vec<SourceOp>, u32)>, TapeLoweringError> {
    let Some((start, period, iters)) = scratchy_subtile::subtile_tape::find_layer_loop(tape, graph)
    else {
        return Ok(None);
    };
    let source_of: std::collections::HashMap<TensorId, SourceOp> = graph
        .op_output
        .iter()
        .enumerate()
        .map(|(j, t)| (*t, SourceOp(j)))
        .collect();
    // `find_layer_loop` answers in `instrs()` positions, which include the slot bookkeeping;
    // only the `Compute`s in the first body copy carry a source op.
    let mut body = Vec::new();
    for instr in &tape.instrs()[start..start + period] {
        if let Instr::Compute { node, .. } = instr {
            let out = graph.nodes[node.index()].output.tensor;
            let op = source_of.get(&out).ok_or_else(|| {
                TapeLoweringError(format!(
                    "loop body node {} writes tensor {} which no source op produces",
                    node.index(),
                    out.index(),
                ))
            })?;
            body.push(*op);
        }
    }
    if body.is_empty() {
        return Ok(None);
    }
    Ok(Some((body, iters)))
}

/// The op each step computes, in tape order — what the opcode emitter matches on.
pub fn ops_of<'g>(graph: &'g SubtileIR, steps: &[&TapeStep]) -> Vec<&'g SubOp> {
    steps
        .iter()
        .map(|s| &graph.nodes[s.node.index()].op)
        .collect()
}

/// ⭐ THE FLIP, AS DATA: emission order and slot coloring for every source op, both READ OFF THE
/// TAPE.
///
/// These are the two things metal used to compute for itself — `wave_levels` for the order,
/// `tape_slot_map` for the coloring — over the pre-`SubtileIR` op list. Sourcing both from the
/// `SubtileTape` is what "metal is on the tape" means concretely: the schedule metal runs and
/// the schedule spyre runs are now the same object's, not two agreeing derivations.
///
/// Indexed by source-op position, because that is what metal's emitter walks.
pub struct TapeSchedule {
    /// `level[j]` = tape position of source op `j`; `None` for an op the lowering folded away
    /// (it produces no tape step, and the emitter skips it).
    pub level: Vec<Option<usize>>,
    /// `slot[j]` = the slot source op `j` writes.
    pub slot: Vec<Option<SlotId>>,
    pub num_slots: u32,
}

impl TapeSchedule {
    /// The slot holding the forward's result.
    pub fn final_slot(&self, result_op: SourceOp) -> Result<SlotId, TapeLoweringError> {
        self.slot
            .get(result_op.0)
            .copied()
            .flatten()
            .ok_or_else(|| {
                TapeLoweringError(format!("result op {} writes no tape slot", result_op.0))
            })
    }
}

/// Project the tape's order and coloring back onto the source op list.
pub fn schedule_of(num_source_ops: usize, steps: &[&TapeStep]) -> TapeSchedule {
    let mut level = vec![None; num_source_ops];
    let mut slot = vec![None; num_source_ops];
    let mut num_slots = 0u32;
    for (pos, s) in steps.iter().enumerate() {
        level[s.source_op.0] = Some(pos);
        slot[s.source_op.0] = Some(s.writes);
        num_slots = num_slots.max(s.writes.index() + 1);
    }
    TapeSchedule {
        level,
        slot,
        num_slots,
    }
}

/// Re-cut the layer loop out of the UN-rolled item list, peeling `peel` leading iterations into
/// the prologue. `rolled` supplies the loop's shape (its var, body length and trip count).
///
/// ⭐ WHY PEELING IS THE FIX AND ROTATING IS NOT. The rolled tape holds ONE copy of the body, and
/// which layer's source ops that copy is drawn from decides how a target lowers it. Metal merges
/// a layer's residual `Add` into the NEXT layer's norm — so LAYER 0's norm is not fused (nothing
/// precedes it but the embed) while every later layer's is. If the body is layer 0's ops,
/// replaying it emits an unfused `RmsNorm` for every layer: a different program.
///
/// Rotating within the body cannot fix that — it reorders the same source ops. Peeling changes
/// WHICH ops the body is, by starting the run a whole period later: layer 0 stays in the
/// prologue and lowers on its own terms, and the body becomes a representative middle layer.
///
/// The run is periodic, so `[P, B×n, E]` and `[P ++ B, B×(n-1), E]` are the same sequence.
/// How many layers one iteration of the detected run covers.
///
/// ⭐ MEASURED, NOT COUNTED. `per_layer_out` holds the same op's node id in every copy of the
/// body, so the layer a body-carried op names in copy 0 against copy 1 IS the advance. One
/// subtraction, right for any body shape.
///
/// ⛔ NOT BY COUNTING ATTENTIONS. That assumes one attention per layer, which qwen3.5 breaks: it
/// interleaves `GatedDeltaNet` layers holding no attention, so a four-layer cell counted as one
/// and the emitted stream lagged the un-rolled one by three layers per iteration
/// (`FusedAddRmsNorm(1, 0, 4, ..)` against `(1, 0, 1, ..)`).
///
/// ⛔ AND NOT `per_layer.len()` EITHER. That list is indexed by the LOOP VARIABLE, so its length
/// is the ITERATION count; reading a layer count off it yields 1 for every model.
///
/// Answers 1 on a tape with no loop — it measures the advance INSIDE a body.
pub fn cell_layers(graph: &SubtileIR, tape: &SubtileTape) -> u32 {
    let mut depth = 0usize;
    let mut found = None;
    for instr in tape.instrs() {
        match instr {
            Instr::OpenLoop { .. } => depth += 1,
            Instr::CloseLoop { .. } => depth = depth.saturating_sub(1),
            Instr::Compute { per_layer_out, .. } if depth > 0 && found.is_none() => {
                if let [c0, c1, ..] = per_layer_out.as_slice()
                    && let (Some(l0), Some(l1)) = (
                        graph.nodes[c0.index()].op.layer_index(),
                        graph.nodes[c1.index()].op.layer_index(),
                    )
                    && l1 > l0
                {
                    found = Some(l1 - l0);
                }
            }
            _ => {}
        }
    }
    found.unwrap_or(1)
}

/// How ONE cell of the detected run divides into layers: `(length in steps, class id)`, in order.
///
/// ⛔ LAYERS INSIDE A CELL ARE NOT ALL THE SAME LENGTH. gemma-3's happen to be, so dividing the
/// cell evenly worked there; gemma-4's do not — its cell is 459 instructions over six layers, five
/// sliding at 76 and one global at 79 — and an even division simply refused, leaving a body per
/// LAYER instead of per CLASS.
///
/// The sliding run is found by its own repetition: the shortest prefix that repeats back to back
/// is one layer of the leading class, and what remains after those copies is the odd one out.
/// Empty when the cell holds a single layer or no repeat is found.
pub fn cell_layer_segments(
    graph: &SubtileIR,
    tape: &SubtileTape,
    rolled: &SubtileTape,
) -> Vec<(usize, u64)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let Some((start, period, _)) = scratchy_subtile::subtile_tape::find_layer_loop(tape, graph)
    else {
        return Vec::new();
    };
    if cell_layers(graph, rolled) < 2 {
        return Vec::new();
    }
    // Fingerprints in STEP space — the split works on steps, and the tape's non-Compute
    // instructions carry no emission.
    let fp: Vec<u64> = tape
        .instrs()
        .iter()
        .zip(scratchy_subtile::subtile_tape::class_fingerprints(
            tape, graph,
        ))
        .filter(|(i, _)| matches!(i, Instr::Compute { .. }))
        .map(|(_, f)| f)
        .collect();
    let step_start = tape.instrs()[..start]
        .iter()
        .filter(|i| matches!(i, Instr::Compute { .. }))
        .count();
    let step_period = tape.instrs()[start..start + period]
        .iter()
        .filter(|i| matches!(i, Instr::Compute { .. }))
        .count();
    if step_period == 0 || step_start + step_period > fp.len() {
        return Vec::new();
    }
    let cell = &fp[step_start..step_start + step_period];
    // The shortest prefix that repeats immediately is one layer of the leading class.
    let Some(p) = (1..=step_period / 2).find(|&p| cell[..p] == cell[p..2 * p]) else {
        return Vec::new();
    };
    let mut runs = 1usize;
    while (runs + 1) * p <= step_period && cell[..p] == cell[runs * p..(runs + 1) * p] {
        runs += 1;
    }
    let key = |sl: &[u64]| -> u64 {
        let mut h = DefaultHasher::new();
        sl.hash(&mut h);
        h.finish()
    };
    let mut out: Vec<(usize, u64)> = Vec::new();
    let a = key(&cell[..p]);
    for _ in 0..runs {
        out.push((p, a));
    }
    let rest = step_period - runs * p;
    if rest > 0 {
        out.push((rest, key(&cell[runs * p..])));
    }
    out
}

pub fn roll_at(
    unrolled: &[TapeItem],
    rolled: &[TapeItem],
    peel: u32,
    // How one cell divides into layers: `(length in steps, class id)` — see
    // `cell_layer_segments`. Fewer than two disables the split rather than guessing.
    segs: &[(usize, u64)],
    // Split each cell into per-layer class runs. ⛔ TRIED, NEVER ASSUMED: it collapses gemma-3
    // from six bodies per cell to two, and it makes gemma-4-31b's colouring non-replayable
    // (`RmsNorm(9, ..)` against `RmsNorm(0, ..)`). The caller keeps whichever cut PROVES.
    split: bool,
) -> Option<Vec<TapeItem>> {
    let open = rolled
        .iter()
        .position(|i| matches!(i, TapeItem::OpenLoop { .. }))?;
    let close = rolled
        .iter()
        .position(|i| matches!(i, TapeItem::CloseLoop { .. }))?;
    let TapeItem::OpenLoop { var, iters, stride } = rolled[open] else {
        return None;
    };
    let period = close - open - 1;
    if period == 0 || iters <= peel + 1 {
        return None;
    }
    let iters_left = iters - peel;
    // The un-rolled list is all steps; the epilogue is whatever follows the run, and the run's
    // last iteration ends `period * iters_left` steps from where the peeled body begins.
    let tail = unrolled.len().checked_sub(rolled.len() - close - 1)?;
    let body_start = tail.checked_sub(period * iters_left as usize)?;
    let mut out = Vec::with_capacity(unrolled.len());
    // ⭐ THE PEELED CELLS GET THE SAME SPLIT. They run once each, so they carry no outer loop —
    // but their layers are still class runs, and leaving them straight-line is what kept a
    // peel=1 tape at six prologue bodies plus two looped ones.
    let peeled_start = body_start.saturating_sub(period * peel as usize);
    out.extend_from_slice(&unrolled[..peeled_start]);
    for c in 0..peel as usize {
        let at = peeled_start + c * period;
        let cell = &unrolled[at..at + period];
        if !split {
            out.extend_from_slice(cell);
        } else if c == 0 && segs.len() > 1 {
            // ⛔ LAYER 0 STANDS ALONE, ALWAYS. Its norm has no residual `Add` ahead of it to fuse
            // with, so metal emits `ScalarOffsetRmsNorm` where every later layer gets
            // `FusedAddRmsNormWithOffset`. That is an emission fact the TAPE cannot see — the
            // segments come off the tape and call layer 0 the same class as its neighbours — so
            // it is excluded here rather than by classification.
            let first = segs[0].0;
            out.extend_from_slice(&cell[..first]);
            out.extend(split_body_by_class(&cell[first..], var, &segs[1..]));
        } else {
            out.extend(split_body_by_class(cell, var, segs));
        }
    }
    out.push(TapeItem::OpenLoop {
        var,
        iters: iters_left,
        stride,
    });
    if split {
        out.extend(split_body_by_class(
            &unrolled[body_start..body_start + period],
            var,
            segs,
        ));
    } else {
        out.extend_from_slice(&unrolled[body_start..body_start + period]);
    }
    out.push(TapeItem::CloseLoop { var });
    // ⭐ AND THE TAIL. A layer count that is not a whole number of cells leaves a remainder —
    // gemma-3 is 26 layers over six-layer cells, so two sliding layers fall past the run. They
    // are the same class, so they are a loop too; emitting them straight left two more bodies.
    //
    // ⛔ COUNTED IN STEPS, NOT FROM THE SEGMENT LIST. `segs` describes ONE cell, so subtracting a
    // covered layer count from its length is always zero — the first version of this changed
    // nothing at all. What follows the run is whole layers of the LEADING class, then an epilogue
    // shorter than one layer.
    let lead = segs.first().map(|(l, _)| *l).unwrap_or(0);
    let tail_layers = (unrolled.len() - tail).checked_div(lead).unwrap_or(0);
    if split && tail_layers > 1 && unrolled.len() >= tail + tail_layers * lead {
        let tail_segs: Vec<(usize, u64)> = (0..tail_layers).map(|_| segs[0]).collect();
        out.extend(split_body_by_class(
            &unrolled[tail..tail + tail_layers * lead],
            var,
            &tail_segs,
        ));
        out.extend_from_slice(&unrolled[tail + tail_layers * lead..]);
    } else {
        out.extend_from_slice(&unrolled[tail..]);
    }
    Some(out)
}

/// Split one cell body into per-layer CLASS runs, looping each run that repeats.
///
/// ⭐ THIS IS WHERE THE BODY COUNT COLLAPSES. A cell rolled whole still emits one body per LAYER
/// in it — gemma-3's six-layer `SSSSSG` cell is six bodies, and with a peeled cell that is twelve.
/// Consecutive layers of the same class are the same program, so they are a loop: the cell
/// becomes ONE sliding body plus ONE global body, and the tape lands at prologue + 2 + epilogue.
///
/// ⛔ AND THE COLOURING ALREADY ALLOWS IT. Measured on gemma-3: every layer from 1 on carries an
/// identical colour sequence, ACROSS the sliding/global boundary — the arena is periodic at one
/// layer, and only layer 0 differs (which is what the cell peel already removes). Nothing here
/// forces a colour; the earlier attempt that did produced `lelelelelele` on the card.
///
/// `segs` is `(length in steps, class id)` per layer of the cell. Returns the body unchanged when
/// the cell holds one layer, when the segments do not account for it exactly, or when no class
/// repeats.
fn split_body_by_class(body: &[TapeItem], var: LoopVarId, segs: &[(usize, u64)]) -> Vec<TapeItem> {
    if segs.len() < 2 || segs.iter().map(|(l, _)| l).sum::<usize>() != body.len() {
        return body.to_vec();
    }
    let mut out = Vec::with_capacity(body.len());
    let mut at = 0usize;
    let mut k = 0usize;
    while k < segs.len() {
        let (len, cls) = segs[k];
        let mut run = 1usize;
        while k + run < segs.len() && segs[k + run] == (len, cls) {
            run += 1;
        }
        if run > 1 {
            out.push(TapeItem::OpenLoop {
                var,
                iters: run as u32,
                stride: 1,
            });
            out.extend_from_slice(&body[at..at + len]);
            out.push(TapeItem::CloseLoop { var });
        } else {
            out.extend_from_slice(&body[at..at + len * run]);
        }
        at += len * run;
        k += run;
    }
    out
}
