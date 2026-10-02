// SPDX-License-Identifier: Apache-2.0
//! METAL, LOWERED FROM THE SHARED `SubtileTape`.
//!
//! ⭐ ONE FRONT END. Metal reads the SAME artifact spyre lowers — the `SubtileTape` — so "one
//! front end" names one object rather than one crate boundary.
//!
//! ⛔ AND IT LIVES IN THE TARGET CRATE, not in `compiler/macros`. Spyre's emitter is
//! `targets/spyre/src/lower_subtile_tape_to_superdsc.rs`; this is its metal counterpart. A
//! `metal_*` module inside the shared compiler is target-specific code in a common crate, which
//! is the arrangement this whole line of work exists to remove.
//!
//! ## What the tape gives
//!
//! An [`Instr::Compute`] names a node in the graph, the slot it writes, and its positional
//! inputs as either earlier slots or graph sources. So the vocabulary to match on is `SubOp` —
//! `lower_region` has already resolved the arch-level ops into it.
//!
//! ## Tiling is a target fact
//!
//! The graph is built with `nb = u32::MAX` — WHOLE ops. Spyre tiles to 128-column pages
//! (`decompose_rmsnorm`, `head_tile_rope`) because its substrate requires it; metal's kernels
//! take a whole `RmsNorm` and a whole rope. Same builder, same tape, different `nb`.

use scratchy_subtile::subtile_ir::{SubtileIR, SubtileId, TensorId};
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
/// SiluMul kernel, metal declares its own (`op_abi::METAL_FUSIONS`). WHICH FUSIONS TO APPLY IS A
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
pub fn roll_at(unrolled: &[TapeItem], rolled: &[TapeItem], peel: u32) -> Option<Vec<TapeItem>> {
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
    out.extend_from_slice(&unrolled[..body_start]);
    out.push(TapeItem::OpenLoop {
        var,
        iters: iters_left,
        stride,
    });
    out.extend_from_slice(&unrolled[body_start..body_start + period]);
    out.push(TapeItem::CloseLoop { var });
    out.extend_from_slice(&unrolled[tail..]);
    Some(out)
}

/// Every layer rolled by its class — the shared
/// [`scratchy_subtile::subtile_tape::layer_class_plan`], rendered onto the un-rolled items.
/// `None` when the tape has no plan, so the caller keeps its other cuts.
pub fn roll_layer_classes(
    graph: &SubtileIR,
    tape: &SubtileTape,
    rolled: &[TapeItem],
    unrolled: &[TapeItem],
) -> Option<Vec<TapeItem>> {
    use scratchy_subtile::subtile_tape::RollPlan;
    fn render(plan: &[RollPlan], unrolled: &[TapeItem], var: LoopVarId, out: &mut Vec<TapeItem>) {
        for p in plan {
            match p {
                RollPlan::Steps(r) => out.extend_from_slice(&unrolled[r.clone()]),
                RollPlan::Loop {
                    iters,
                    stride,
                    body,
                } => {
                    out.push(TapeItem::OpenLoop {
                        var,
                        iters: *iters,
                        stride: *stride,
                    });
                    render(body, unrolled, var, out);
                    out.push(TapeItem::CloseLoop { var });
                }
            }
        }
    }
    let var = rolled.iter().find_map(|i| match i {
        TapeItem::OpenLoop { var, .. } => Some(*var),
        _ => None,
    })?;
    let plan = scratchy_subtile::subtile_tape::layer_class_plan(tape, graph)?;
    let mut out = Vec::with_capacity(unrolled.len());
    render(&plan, unrolled, var, &mut out);
    Some(out)
}
