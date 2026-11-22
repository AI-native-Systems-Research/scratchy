// SPDX-License-Identifier: Apache-2.0
//! SubtileIR → **KTIR**: the PRODUCER half of `SubtileIR → KTIR → SuperDSC`.
//!
//! ⭐⭐⭐ THE PROGRAM IS CONSTRUCTED, NEVER PRINTED. `ktir-core`'s `Operation` / `IRFunction` ARE the
//! interchange: the emulator executes the value directly and `#[forward]` bakes it as const data.
//! Nothing here renders MLIR and nothing anywhere parses it.
//!
//! ⭐ ONE NODE, ONE PROGRAM. Each [`SubtileNode`] becomes one KTIR `func` — inputs loaded from HBM,
//! output stored back — carried on an [`EmittedOp`] as `ktir`. Both consumers read that SAME value:
//! `-Fspyre-emu` interprets it, `-Fspyre-hw` lowers it through [`crate::superdsc_lower`].
//!
//! ⛔ NOTHING HERE EMITS A SuperDSC DESCRIPTOR. That is the consumer's half, in its own file, and a
//! node lowered straight to SuperDSC from here would be a second path to the same format.
//!
//! [`EmittedOp`]: scratchy_triton_splice::EmittedOp

use crate::lower_subtile_tape_to_superdsc::*;
use scratchy_spyre_bundle::program::ActiveCap;
use scratchy_spyre_bundle::rung::QueryExtent;
use scratchy_subtile::subtile_ir::{RopeForm, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId};
use scratchy_triton_splice::EmittedOp;

/// The stable name of a HOST-ROUTED (data-movement) SubOp, for the host-routed
/// reconnaissance set. These ops never become SuperDSC tiles — the host threads
/// activations across them — so this is a label, not a lowering.
fn host_glue_kind<F: RopeForm>(op: &SubOp<F>) -> &'static str {
    match op {
        SubOp::RopeRotate { .. } => "RopeRotate",
        SubOp::RopeAppend { .. } => "RopeAppend",
        SubOp::AttnDecode { .. } => "AttnDecode",
        SubOp::RmsNormReduce { .. } => "RmsNormReduce",
        SubOp::RmsNormApply { .. } => "RmsNormApply",
        // The pure-compute ops are never passed here.
        _ => "?",
    }
}

/// Outcome of lowering ONE SubtileIR node. Shared by the UNROLLED
/// [`lower_graph_to_ktir`] walk and the RE-ROLLED tape-driven walk so the
/// per-op `match` lives exactly once (the reroll just changes WHICH nodes are
/// walked + how many times the body runs, not how each op lowers).
pub(crate) enum NodeLowering<T: scratchy_spyre_bundle::collective::Tp> {
    Ops(Vec<EmittedOp<T>>),
    /// An op kind not yet lowerable (collected into the hard-error worklist).
    Unhandled(String),
    /// A recognized data-movement op routed host-side (not a SuperDSC tile).
    HostRouted(&'static str),
}

/// ⛔⭐⭐⭐ THE F16 FLOOR GATE — a weight fold refuses at bake when its landing cannot be
/// PROVEN inside f16's normal range. Shared by both fold recognizers, so every consumer
/// (the runtime list, the layout field, the macro census, the aliasing sites, the numeric
/// gate's `fold_scale`) sees the same refused set and the two tiers take the same shape.
///
/// The fold's runtime half multiplies the STAGED weight bytes once, elementwise
/// (`apply_weight_scale_folds`), and the card stages every float payload at f16 semantics
/// (the `cfg(spyre-hw)` narrowing), so the device stick holds `f16(ws · s)` per element.
/// Below f16's normal floor — 2⁻¹⁴ = 6.104e-5 — the subnormal grid sheds mantissa bits
/// (a folded ws at ~2.6e-5 keeps ~9 of f16's 11), and the H2D codec's re-normalization of
/// the subnormal does not give them back. Measured on granite fp8 (the card-vs-card join,
/// folds-on dump vs folds-off control): the host-staged bytes sit 0.04% off exact while
/// the o-side outputs sit 0.29% off the un-folded run — a seed the trajectory amplifies
/// ~35× to 10.2% at the logits, flipping the answer. The probe's own `fold / C` staging
/// already kept ITSELF above this floor ("C = 512 keeps BOTH sides out of subnormal
/// range"); this gate extends the same law to the real fold decision.
///
/// The per-element values are CHECKPOINT DATA — staged at load, unbounded below at bake —
/// so ANY-element safety is provable in exactly one direction: `|s| >= 1` can only move a
/// value away from the floor (a source that staged normal lands normal), while `|s| < 1`
/// can push any sufficiently small element under it and NO element can be cleared at bake.
/// The gate is that rule's conservative closure: down-scales refuse, the multiplier stays
/// on the tape (the `ScalarMul` op, the attention's own `√s` multiplies), and the fold's
/// algebra is simply not taken. An up-fold cannot create a subnormal landing the source
/// did not already carry — a source already below the floor is a staging defect, not a
/// fold's, and refuses (or not) on its own laws.
///
/// A refused fold logs once per `(target, multiplier)` per bake — the recognizers run once
/// per consumer, and the rung contract wants the refusal legible under `cargo -vv`, not
/// repeated per consumer.
fn weight_fold_clears_f16_floor(target: u32, multiplier: f32) -> bool {
    if multiplier.abs() >= 1.0 {
        return true;
    }
    static SAID: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<(u32, u32)>>> =
        std::sync::OnceLock::new();
    let said = SAID.get_or_init(Default::default);
    if said
        .lock()
        .expect("fold-gate log lock")
        .insert((target, multiplier.to_bits()))
    {
        eprintln!(
            "[spyre-folds] F16 FLOOR REFUSAL t{target} x{multiplier}: the landing \
             f16(ws·s) cannot be proven above the normal floor 6.104e-5 at bake (staged ws \
             is checkpoint data, unbounded below), and a subnormal landing sheds mantissa \
             bits the device codec does not restore — the multiplier stays on the tape"
        );
    }
    false
}

/// ⭐⭐⭐ THE SCALARMUL WEIGHT FOLD — the compile-time multipliers that never need a device op.
///
/// A `ScalarMul` whose single input is a MATMUL's whole output multiplies a linear function's
/// result by a constant, and a constant commutes with linearity: `(x·W)·s == x·(W·s)`. So the
/// multiplier can leave the runtime tape entirely — the worker stages the fold target's bytes
/// pre-scaled and the pointwise `mul` + its `[1,1]` const bind disappear. This is the constants
/// fold the hardware has no descriptor door for (no DDL template takes a `mulConst` outside
/// `quant_scale_per_token.ddl`): the weight IS the door.
///
/// The fold target is the producer's LAST input — the same slot for both arities the emitter
/// knows: dense `MatmulTile` `[A, W]` folds into `W`; fp8 W8A8 `[A, qW, ws]` folds into the
/// `ws` dequant row (`(qA·qW)·a[m]·ws[n]·s == (qA·qW)·a[m]·(ws·s)[n]` — the payload and its
/// quantization are untouched, which is why the fp8 fold is exact algebra, not a requantize).
///
/// ⛔ EVERY REFUSAL BELOW IS A CORRECTNESS FACT, NOT TIMIDITY:
///   * a SLICED read (the input region not EXACTLY one producer's whole output) — the fold
///     scales every row the matmul wrote, not just the slice read, and two producers writing
///     disjoint regions of one tensor match no whole-output producer at all;
///   * the fold target read by any OTHER node — a shared weight cannot absorb a per-consumer
///     scale (granite ties `lm_head` to the embedding table; a scale folded into a tied table
///     would scale the logits too);
///   * the producer's output read by any node other than this `ScalarMul` — that reader needs
///     the UNSCALED product, and the fold would hand it `x·W·s`;
///   * a non-matmul producer (granite's `embedding_multiplier` scales a GATHER) — the
///     multiplier does not commute with a lookup;
///   * an output region that does not MIRROR the input — the fold re-points every reader of
///     this node's output at the producer's bytes (the layout aliases the two placements),
///     which is the same window only because a pointwise map's output region equals its input
///     region;
///   * the graph RESULT — the logits ScalarMul is the splice's own lm_head-tail fold
///     (`lower_all` folds both halves to m=1), and the result's Logits placement is the
///     worker's readback window, not a window this fold may redirect.
pub(crate) fn scalar_mul_weight_folds<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Vec<(SubtileId, u32, f32)> {
    let _pc = crate::pc::Pc::start("foldscan");
    crate::pc::count("foldscan", ir.nodes.len() as u64);
    let mut folds = Vec::new();
    for node in &ir.nodes {
        let SubOp::ScalarMul { scale } = &node.op else {
            continue;
        };
        let [input] = node.inputs.as_slice() else {
            continue;
        };
        if node.output.tensor == ir.result || node.output.region != input.region {
            continue;
        }
        let Some(producer) = ir
            .nodes
            .iter()
            .find(|p| p.id != node.id && p.output == *input)
        else {
            continue;
        };
        if !matches!(producer.op, SubOp::MatmulTile { .. }) || producer.inputs.len() < 2 {
            continue;
        }
        let target = producer.inputs.last().expect("len checked above");
        // ⛔ AN fp8 WEIGHT CANNOT ABSORB A SCALE — multiplying its mantissa bytes elementwise is
        // a REQUANTIZE (the product has to re-encode into fp8's own exponent range), not the
        // exact algebra this fold is. The `ws` dequant row is the fp8 fold's target precisely
        // because it sits OUTSIDE the quantized payload. The fp8-weight test is the emitter's
        // own (`input[1]` of an arity-3 `MatmulTile`, `compute_bundle_layout::fp8_weight_tids`),
        // restated here so a matmul whose input order puts the quantized payload last is refused
        // at RECOGNITION — a build-time fact, not a load-time dtype check.
        let is_fp8_weight = |t: TensorId| {
            ir.nodes.iter().any(|n| {
                matches!(n.op, SubOp::MatmulTile { .. })
                    && n.inputs.len() == 3
                    && n.inputs[1].tensor == t
            })
        };
        if is_fp8_weight(target.tensor) {
            continue;
        }
        let shared = |tid: TensorId, skip: SubtileId| {
            ir.nodes
                .iter()
                .any(|n| n.id != skip && n.inputs.iter().any(|r| r.tensor == tid))
        };
        // The weight must be this matmul's alone, and the product must be this ScalarMul's
        // alone — see the refusals above.
        if shared(target.tensor, producer.id) || shared(input.tensor, node.id) {
            continue;
        }
        // ⛔ THE F16 FLOOR GATE — a down-scale's landing is unprovable at bake and a subnormal
        // landing corrupts the staged bytes (see the gate's doc for the measured card
        // evidence); the multiplier stays on the tape as the pointwise op it already is.
        if !weight_fold_clears_f16_floor(target.tensor.index() as u32, *scale) {
            continue;
        }
        folds.push((node.id, target.tensor.index() as u32, *scale));
    }
    folds
}

/// ⭐⭐⭐ THE ATTENTION SCALE'S OWN WEIGHT FOLD — `(q·√s)·(k·√s)ᵀ == (q·kᵀ)·s`, with each `√s`
/// commuted into the projection weight that produced its side.
///
/// torch-spyre's attention scales the scores by applying `√scale` to BOTH the query and the key
/// (`query * scaling_factor`, `key * scaling_factor`), and those two multiplies commute into
/// `W_q` and `W_k` exactly the way a residual multiplier commutes into `o_proj`: the query's
/// `√s` rides its matmul, the key's rides its matmul **and the KV cache the rope writes** — the
/// cache then holds pre-scaled K, which is consistent because the score matmul is the only thing
/// that ever reads it. The rope in between is a rotation (linear), so the scalar passes through
/// it unchanged; `V` carries no scale and is untouched.
///
/// The fold's node-side half is the mirror of the ScalarMul fold's, with ONE structural
/// difference: **the AttnDecode node STAYS on the tape at `scale = 1.0`.** The two multiplies
/// this fold removes are not tape nodes at all — they are ops the attention lowering emits
/// (`qs = Q·√s`, `new_k_scaled = K·√s`) reading the `[1,1]` registry consts — so nothing is
/// host-routed, nothing is re-pointed, and no placement aliases: the recognition rewrites the
/// node's scale to 1.0 where the node is lowered, and the lowering emits no multiplies for a
/// scale of 1.0. The registry entries for both `scale` and `√scale` leave the census with the
/// node's arm skipped.
///
/// The producer walk, and ⛔ EVERY REFUSAL IS A CORRECTNESS FACT:
///   * `inputs[0]` (Q) must be a rope's WHOLE output — its producer is the `RopeRotate` — and
///     `inputs[3]` (new K) the `RopeAppend`'s; a pre-populated cache region or any other
///     producer shape refuses (the walk is granite's chain, stated as a shape and not guessed
///     at: `matmul → rope → attention`);
///   * the rope's first input must be a MATMUL's whole output, whose LAST input is the fold
///     target (`W_q` / `W_k`) — the same slot law the ScalarMul fold uses, so fp8 W8A8 folds
///     into the `ws` dequant row (exact algebra, never a requantize);
///   * the fold target read by any OTHER node refuses — a shared weight cannot absorb a
///     per-consumer scale (a fused-QKV table would refuse here, which is why the fold is
///     granite's separate-projections shape);
///   * the MATMUL's output read by any node other than the rope, and the ROPE's output read by
///     any node other than this attention, both refuse — the fold changes those tensors'
///     VALUES (that is its whole point), so a second consumer wanting the unscaled value would
///     read pre-scaled bytes;
///   * `scale == 1.0` has nothing to fold and is skipped, not refused — it is also the value a
///     folded node carries afterward, which is what makes the rewrite idempotent.
///
/// Returns `(attn node, W_q tid, W_k tid, √scale)` — one entry per foldable attention node.
pub(crate) fn attn_scale_weight_folds<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Vec<(SubtileId, u32, u32, f32)> {
    let _pc = crate::pc::Pc::start("foldscan");
    crate::pc::count("foldscan", ir.nodes.len() as u64);
    /// The projection weight behind one side of an attention: `attn_input` must be the WHOLE
    /// output of a rope (`RopeRotate` for the query, `RopeAppend` for the key — both linear
    /// rotations of their first input, which is the algebra the fold rides), and that rope's
    /// first input must be the WHOLE output of a matmul whose last input is the target.
    fn projection_weight<F: RopeForm>(
        ir: &SubtileIR<F>,
        attn_input: &scratchy_subtile::subtile_ir::TensorRegion,
        attn_id: SubtileId,
        is_fp8_weight: &impl Fn(TensorId) -> bool,
        shared: &impl Fn(TensorId, SubtileId) -> bool,
    ) -> Option<TensorId> {
        let rope = ir
            .nodes
            .iter()
            .find(|p| p.id != attn_id && p.output == *attn_input)?;
        if !matches!(rope.op, SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. })
            || rope.inputs.is_empty()
        {
            return None;
        }
        let matmul = ir
            .nodes
            .iter()
            .find(|p| p.id != rope.id && p.output == rope.inputs[0])?;
        if !matches!(matmul.op, SubOp::MatmulTile { .. }) || matmul.inputs.len() < 2 {
            return None;
        }
        let target = matmul.inputs.last().expect("len checked above");
        // The fp8-payload and shared-weight refusals, same laws as the ScalarMul fold's —
        // see that function's doc for why each is a correctness fact and not timidity.
        if is_fp8_weight(target.tensor) {
            return None;
        }
        if shared(target.tensor, matmul.id) {
            return None;
        }
        // The matmul's product must be the rope's alone, and the rope's output this
        // attention's alone: the fold changes both tensors' values, and any other reader
        // needs the unscaled ones.
        if shared(matmul.output.tensor, rope.id) || shared(rope.output.tensor, attn_id) {
            return None;
        }
        Some(target.tensor)
    }
    let is_fp8_weight = |t: TensorId| {
        ir.nodes.iter().any(|n| {
            matches!(n.op, SubOp::MatmulTile { .. })
                && n.inputs.len() == 3
                && n.inputs[1].tensor == t
        })
    };
    let shared = |tid: TensorId, skip: SubtileId| {
        ir.nodes
            .iter()
            .any(|n| n.id != skip && n.inputs.iter().any(|r| r.tensor == tid))
    };
    let mut folds = Vec::new();
    for node in &ir.nodes {
        let SubOp::AttnDecode { attention, .. } = &node.op else {
            continue;
        };
        // inputs = [q, prefix_k, prefix_v, new_k, new_v] — the walk is the two live sides.
        let [q, _prefix_k, _prefix_v, new_k, _new_v] = node.inputs.as_slice() else {
            continue;
        };
        // The model's scale at this node's class, off the facts.
        let scale = attention.scale().get();
        if scale == 1.0 {
            continue;
        }
        let Some(w_q) = projection_weight(ir, q, node.id, &is_fp8_weight, &shared) else {
            continue;
        };
        let Some(w_k) = projection_weight(ir, new_k, node.id, &is_fp8_weight, &shared) else {
            continue;
        };
        // ⛔ THE F16 FLOOR GATE, pair-wise: the score algebra needs BOTH sides folded
        // (`(q·√s)·(k·√s)ᵀ == (q·kᵀ)·s`), so a side that cannot prove a normal landing
        // refuses the whole fold — the node keeps its own scale and lowers its multiplies.
        let sqrt_scale = scale.sqrt();
        if !weight_fold_clears_f16_floor(w_q.index() as u32, sqrt_scale)
            || !weight_fold_clears_f16_floor(w_k.index() as u32, sqrt_scale)
        {
            continue;
        }
        folds.push((node.id, w_q.index() as u32, w_k.index() as u32, sqrt_scale));
    }
    folds
}

/// ⭐ EVERY weight fold this graph admits, in the ONE list every fold consumer reads.
///
/// Every entry has passed the f16 floor gate ([`weight_fold_clears_f16_floor`]): a fold
/// whose landing cannot be proven inside f16's normal range at bake is not admitted, so
/// this list never names a fold the staged bytes cannot carry at full precision.
///
/// Both kinds are the same contract at load time — the worker multiplies the staged bytes of
/// each `(tid, multiplier)` pair once, at load — so the runtime list, the layout field, and the
/// macro's cross-program census take this union:
///   * the [`scalar_mul_weight_folds`]: `(folded ScalarMul node, weight tid, multiplier)` —
///     the node LEAVES the tape (host-routed, its output placement aliased to the producer's);
///   * the [`attn_scale_weight_folds`]: the SAME `(node, tid, multiplier)` shape, twice per
///     attention node (W_q and W_k at `√scale`) — the node STAYS, at `scale = 1.0`.
///
/// ⛔ THE TWO KINDS ARE NOT INTERCHANGEABLE DOWNSTREAM, AND THE DISTINCTION IS THE NODE'S OWN
/// OP. A consumer that treats a fold's node as gone (the placement alias, the liveness
/// extension, the golden's eval skip) must ask `scalar_mul_weight_folds` directly — only that
/// kind leaves the tape. A consumer that applies bytes (the layout field, the runtime, the
/// census, the golden's source pre-scaling) takes this union. A fold entry keyed on an
/// AttnDecode reaching the aliasing sites would point the attention's OUTPUT placement at the
/// query's bytes, so those sites keep their own recognizer rather than filtering this list.
pub(crate) fn weight_scale_folds<F: RopeForm>(ir: &SubtileIR<F>) -> Vec<(SubtileId, u32, f32)> {
    let mut folds = scalar_mul_weight_folds(ir);
    for (id, w_q, w_k, sqrt_scale) in attn_scale_weight_folds(ir) {
        folds.push((id, w_q, sqrt_scale));
        folds.push((id, w_k, sqrt_scale));
    }
    folds
}

/// ⭐ WHAT A COLLECTIVE'S SENDS MOVE — this rank's operand `x` as ONE fragment. A send is not a
/// program: it is an F-send control block (device p2p RDMA, flex-rs
/// `scheduler::build_rdma_send_cb`) the CB pipeline issues, once per peer, ahead of the consume's
/// waits, moving the fragment from `x`'s own placement into this rank's block of every peer's
/// receive window on this card's DMA engine. The consume names the operand and this fragment
/// ([`TpConsume`]), which is all the sends need.
///
/// ⛔ ONE FRAGMENT: a window block holds one fragment — `WINDOW_ROWS` rows of the operand
/// (`RecvWindow::block_bytes_for`), more than any bundle hands a collective — so every operand is
/// one, and each send is a run of messages when it passes the cap. An operand with more rows than a
/// block is refused by name.
///
/// [`TpConsume`]: scratchy_spyre_bundle::collective::TpConsume
fn send_fragment<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    node: &SubtileNode<F>,
    kind: scratchy_spyre_bundle::collective::CollectiveKind,
) -> Result<scratchy_spyre_bundle::collective::Fragment, NodeLowering<T>> {
    let x = node.inputs[0];
    let (rows, cols) = (x.region.rows.len, x.region.cols.len);
    match crate::tp_plan::consume_fragments::<T>(rows, cols).as_deref() {
        Ok([f]) => Ok(*f),
        Ok(more) => Err(NodeLowering::Unhandled(format!(
            "the {kind:?} operand [{rows}, {cols}] is {} p2p fragments — more rows than a \
             receive window block holds, and two fragments of one collective cannot share it",
            more.len()
        ))),
        Err(why) => Err(NodeLowering::Unhandled(format!(
            "the {kind:?} operand [{rows}, {cols}] cannot be cut into p2p fragments: {why:?}"
        ))),
    }
}

/// ⭐ THE WINDOW CUT `node`'S COLLECTIVE USES — [`WindowCut::of_collective`] its index in the
/// forward's own order (how many collectives precede it in `ir`), so consecutive collectives land
/// in alternate cuts. The same index on every rank and in every bundle that carries the node.
///
/// [`WindowCut::of_collective`]: scratchy_spyre_bundle::collective::WindowCut::of_collective
fn window_cut<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> scratchy_spyre_bundle::collective::WindowCut {
    let k = ir
        .nodes
        .iter()
        .take_while(|n| n.id != node.id)
        .filter(|n| matches!(n.op, SubOp::AllReduce | SubOp::AllGather { .. }))
        .count();
    scratchy_spyre_bundle::collective::WindowCut::of_collective(k)
}

/// The `[rows, cols]` region of `tensor` starting at its first byte — how a consume reads a
/// sender's fragment out of its window block (a run of whole stick columns is byte-identical to
/// such a tensor).
fn whole_block(
    tensor: TensorId,
    rows: u32,
    cols: u32,
) -> scratchy_subtile::subtile_ir::TensorRegion {
    use scratchy_subtile::subtile_ir::{Range, Region, TensorRegion};
    TensorRegion {
        tensor,
        region: Region {
            rows: Range::new(0, rows),
            cols: Range::new(0, cols),
        },
    }
}

/// The all-reduce's consumes: per rank `me` of the world, the sum of every rank's share in
/// ABSOLUTE RANK ORDER (`((x_0 + x_1) + x_2) + …`), each share read from `me`'s own operand when it
/// is `me`'s and from the sender's block of the window otherwise. A chain of `W - 1` ordinary `Add`
/// splices into the output's fragment columns — the running sum lives in the output, so the chain
/// is one program per add and no temporary. Every rank adds the same values in the same order, so
/// the replicated result is bit-identical on every card (the analysis's own law,
/// `tensor_parallel::eval_collective`). All of one rank's adds carry ONE [`TpConsume`] that plays
/// at that rank, so they fuse into its consume group behind its sends and waits.
///
/// [`TpConsume`]: scratchy_spyre_bundle::collective::TpConsume
fn all_reduce_consumes<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> NodeLowering<T> {
    use scratchy_spyre_bundle::collective::{CollectiveKind, Rank, TpConsume, ranks};
    use scratchy_subtile::subtile_ir::{EwKind, Range, Region, TensorRegion};
    let x = node.inputs[0];
    let rows = x.region.rows.len;
    let f = match send_fragment::<T, F>(node, CollectiveKind::AllReduce) {
        Ok(f) => f,
        Err(refusal) => return refusal,
    };
    let cut = window_cut(node, ir);
    let cols_at = |start: u32| Range::new(start + f.first_col(), f.cols());
    let own = TensorRegion {
        tensor: x.tensor,
        region: Region {
            rows: x.region.rows,
            cols: cols_at(x.region.cols.start),
        },
    };
    let out = TensorRegion {
        tensor: node.output.tensor,
        region: Region {
            rows: node.output.region.rows,
            cols: cols_at(node.output.region.cols.start),
        },
    };
    let mut ops = Vec::new();
    for me in ranks::<T>() {
        let consume = TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: scratchy_spyre_bundle::place::PlaceId::Act(x.tensor.index() as u32),
            fragment: f,
            plays_at: me,
            cut,
        };
        let share = |s: Rank<T>| {
            if s == me {
                own
            } else {
                whole_block(
                    TensorId::from_index(cut.block_tid(s) as usize),
                    rows,
                    f.cols(),
                )
            }
        };
        // Rank 0's share then each later rank's, each add accumulating into the output.
        let mut acc = share(Rank::of::<0>());
        for (k, s) in ranks::<T>().enumerate().skip(1) {
            let add = SubtileNode {
                id: node.id,
                op: SubOp::Elementwise(EwKind::Add),
                inputs: vec![acc, share(s)],
                output: out,
            };
            let mut e = match scratchy_triton_splice::lower(&add, ir, rows_are_requests) {
                Ok(e) => e,
                Err(reason) => return NodeLowering::Unhandled(reason),
            };
            // One name per (rank, add): they share the node's id, and a launch's name is its
            // program's identity.
            let name = format!("tpreduce_s{}_r{}_a{k}", node.id.index(), me.get());
            e.ktir.func.name = ktir_core::arena::Arena::global().str(name.clone());
            e.op_name = name;
            e.tp_consume = Some(consume);
            ops.push(e);
            acc = out;
        }
    }
    NodeLowering::Ops(ops)
}

/// The all-gather's consumes: per rank `me` of the world, one block copy
/// (`scratchy_triton_splice::lower_block_copy`) per rank `s` — `me`'s own `[rows, lc]` block from
/// its operand, every other rank's out of that sender's block of the window — to columns `s · lc`
/// of the gathered output. All of one rank's copies carry ONE [`TpConsume`] that plays at that
/// rank, so they fuse into its consume group and every other rank skips it whole.
///
/// ⛔ ONE FRAGMENT. A block copy moves a WHOLE source, and a window block holds one fragment —
/// `WINDOW_ROWS` rows of the block, which every block a bundle hands a gather fits.
///
/// [`TpConsume`]: scratchy_spyre_bundle::collective::TpConsume
fn all_gather_consumes<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    cols: u32,
) -> NodeLowering<T> {
    use scratchy_spyre_bundle::collective::{CollectiveKind, STICK_ELEMS, TpConsume, World, ranks};
    use scratchy_subtile::subtile_ir::{Range, Region, TensorRegion};
    let x = node.inputs[0];
    let (rows, lc) = (x.region.rows.len, x.region.cols.len);
    let w = <T as World>::RANKS;
    // ⭐ THE BLOCKS COVER THE WIDTH, THE LAST ONE MAYBE PADDED. Rank `r`'s block is `lc` columns
    // at column `r·lc`, and the gathered value is the first `cols` of them — every column, at an
    // even split; at a padded one (`tensor_parallel::split_block`), the last block's tail past the
    // width is padding. Each rank's block is copied up to the last stick holding a real column, so
    // nothing lands past the value's own sticks.
    let span = cols.next_multiple_of(STICK_ELEMS);
    if lc == 0 || (w - 1) * lc >= cols || cols > w * lc {
        return NodeLowering::Unhandled(format!(
            "the all-gather restores {cols} columns from a {lc}-column block — not {w} blocks \
             covering it"
        ));
    }
    let width_of = |rank: u32| lc.min(span - rank * lc);
    let fragment = match send_fragment::<T, F>(node, CollectiveKind::AllGather) {
        Ok(f) => f,
        Err(refusal) => return refusal,
    };
    let cut = window_cut(node, ir);
    let block_at = |rank: u32| TensorRegion {
        tensor: node.output.tensor,
        region: Region {
            rows: node.output.region.rows,
            cols: Range::new(node.output.region.cols.start + rank * lc, width_of(rank)),
        },
    };
    let own = |rank: u32| TensorRegion {
        tensor: x.tensor,
        region: Region {
            rows: x.region.rows,
            cols: Range::new(x.region.cols.start, width_of(rank)),
        },
    };
    let mut ops = Vec::with_capacity((w * w) as usize);
    for me in ranks::<T>() {
        let consume = TpConsume {
            kind: CollectiveKind::AllGather,
            operand: scratchy_spyre_bundle::place::PlaceId::Act(x.tensor.index() as u32),
            fragment,
            plays_at: me,
            cut,
        };
        for s in ranks::<T>() {
            let src = if s == me {
                own(s.get())
            } else {
                whole_block(
                    TensorId::from_index(cut.block_tid(s) as usize),
                    rows,
                    width_of(s.get()),
                )
            };
            let name = format!("tpgather_s{}_r{}_b{}", node.id.index(), me.get(), s.get());
            match scratchy_triton_splice::lower_block_copy(
                node,
                src,
                block_at(s.get()),
                span,
                &name,
            ) {
                Ok(mut e) => {
                    e.tp_consume = Some(consume);
                    ops.push(e);
                }
                Err(reason) => return NodeLowering::Unhandled(reason),
            }
        }
    }
    NodeLowering::Ops(ops)
}

/// Lower ONE [`SubtileNode`] to its SuperDSC op(s) — the single source of the the
/// per-`SubOp` match. `ir` is needed only for `AttnDecode`'s cache-capacity lookup.
/// ⭐⭐⭐ ONE NODE, ONE PROGRAM.
///
/// The SuperDSC lowering decomposes a node into several descriptor ops, because dxp schedules at
/// that grain. KTIR does not: a node's whole computation is one `func`, with its inputs loaded from
/// HBM and its output stored back, and the work division stated INSIDE it as the grid and its
/// loops. So this walk yields one [`EmittedOp`] per node, carrying that program.
pub(crate) fn lower_one_node<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    // Sweep KV extent for AttnDecode (paged-attn ladder rung); FULL ⇒ full cap. See main's
    // `lower_attn_node`.
    active_cap: ActiveCap,
    // See main's `lower_attn_node`: whether this bundle's rows are separate requests.
    rows_are_requests: bool,
    // ⭐ THE QUERY'S EXTENT PAIR — the rung's own `(per-request width, requests)`, minted once at
    // the bake ([`QueryExtent::of_baked_rung`]) and threaded down to the attention splice beside
    // `active_cap`. Only `AttnDecode` spends it; every other op reads the graph's rows.
    query_rows: QueryExtent,
) -> NodeLowering<T> {
    use NodeLowering::{HostRouted, Ops, Unhandled};
    // ⭐⭐⭐ A FOLDED SCALARMUL'S READERS BIND THE PRODUCER'S OUTPUT. The fold takes the node
    // out of the tape — its multiplier rides the staged weight bytes ([`scalar_mul_weight_folds`])
    // — so its output tid is written by NO program, and a consumer that binds it reads a stick
    // nothing ever stores. The card door papers over that in its LAYOUT (the placement alias
    // that points the folded tid at the producer's bytes), but the emulator door has no layout:
    // it binds programs by tid and allocates one stick per referenced tid, so the alias never
    // reaches it and the residual add reads a fresh zero stick while the projection writes a
    // tid nobody reads. Rewriting the consumer's INPUT here — at the boundary where the program
    // is minted, the same single-edit law the AttnDecode scale rewrite below follows — is the
    // one spelling that reaches every door: the minted program is self-consistent (it binds the
    // tid the producer's own op writes, holding exactly the folded product the reader wanted),
    // and on the card the producer's placement IS the bytes the layout alias pointed the folded
    // tid at, so the descriptor half does not move.
    let fold_sources: std::collections::BTreeMap<u32, _> = scalar_mul_weight_folds(ir)
        .into_iter()
        .map(|(id, _, _)| {
            let folded = &ir.nodes[id.index()];
            (folded.output.tensor.index() as u32, folded.inputs[0])
        })
        .collect();
    let rewritten;
    let node: &SubtileNode<F> = if node
        .inputs
        .iter()
        .any(|r| fold_sources.contains_key(&(r.tensor.index() as u32)))
    {
        let mut owned = node.clone();
        for r in &mut owned.inputs {
            if let Some(src) = fold_sources.get(&(r.tensor.index() as u32)) {
                *r = *src;
            }
        }
        rewritten = owned;
        &rewritten
    } else {
        node
    };
    let lowered = match &node.op {
        // The rest of the arch vocabulary. It reaches this emitter because the SHARED
        // front end expresses every op instead of asserting the unsupported ones away
        // in `lower_region` — which is the point: the IR carries the fact and the
        // TARGET says whether it has a kernel. Enumerated, never `_`, so adding a
        // SubOp is E0004 here rather than a surprise at emission.
        SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::Concat { .. }
        | SubOp::GateScale
        | SubOp::LoadRows { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        | SubOp::Mamba2 { .. }
        | SubOp::KvAppend { .. }
        | SubOp::ExpertAct { .. }
        | SubOp::Mean => Unhandled(format!("{:?} has no SuperDSC kernel", node.op)),
        // ⭐ THE ALL-REDUCE'S CONSUME. The exchange itself is not a program — the bundle's CB
        // pipeline sends this rank's operand fragment into every peer's window and waits for every
        // peer's before the consume launches (`bundle::collective::cb_pipeline`). What IS a program
        // is the sum: a chain of ordinary `Add` splices over every rank's share in absolute rank
        // order, so every rank computes bit-identical sums at any world width.
        SubOp::AllReduce => all_reduce_consumes(node, ir, rows_are_requests),
        // ⭐ THE ALL-GATHER'S CONSUME: every rank's block lands at that rank's columns, read from
        // this rank's operand or the sender's window block; the bake carries one consume per rank
        // and each session plays its own (`bundle::collective::cb_pipeline`).
        SubOp::AllGather { cols } => all_gather_consumes(node, ir, *cols),
        // ⛔ THE 2b EXPANSION KINDS, named here instead of `expansion_ops!()`: the 2a
        // routing below made the macro's patterns for THOSE kinds unreachable
        // (`unreachable_patterns` is `-D warnings`), and the remaining members are a
        // fixed list the registry's `@expansion` section states — a new expansion op
        // is E0004 here until it gets its own routing or refusal, exactly as before.
        //
        // The remaining router rows are the biased routers — the rest of the
        // Phase-2b worklist, refused BY NAME rather than silently substituted
        // with a nearest-primitive guess.
        SubOp::RouteSigmoidBias { .. } | SubOp::RouteBias { .. } => Unhandled(format!(
            "{:?} t{} needs the Phase-2b router rows — substituting a nearest primitive \
             would run a different function and report success",
            node.op,
            node.output.tensor.index() as u32,
        )),
        // ⛔ THE MoE SUB-OPS HAVE NO PER-OP CARD PATH — the MoE lowers only
        // from `kernels/moe.py` through KTIR. The splice's registry refuses
        // every MoE row and the door has no MoE arm, so a graph still
        // stating these sub-ops cannot bake on the card; the message names
        // the one path that exists.
        SubOp::RouterNorm { .. }
        | SubOp::RouterLogits { .. }
        | SubOp::RouteScale { .. }
        | SubOp::RouteSoftmax
        | SubOp::RouteRenorm
        | SubOp::RouteArgsort
        | SubOp::RouteTopK { .. }
        | SubOp::RouteGatherScores
        | SubOp::RouteExpertScale { .. }
        | SubOp::ExpertGatedAct { .. }
        | SubOp::ExpertSort { .. }
        | SubOp::ExpertUnsort
        | SubOp::ExpertCombine { .. }
        | SubOp::ExpertMatmul { .. } => {
            Unhandled("the MoE lowers only from kernels/moe.py through KTIR".to_string())
        }
        SubOp::KvEncode { .. }
        | SubOp::KvStage { .. }
        | SubOp::RotateRows { .. }
        | SubOp::AttnPackedKv
        | SubOp::SampleRowsGather
        | SubOp::SampleRowsScatter
        | SubOp::AllRowsMatmul
        | SubOp::CaptureHidden { .. }
        | SubOp::DraftConv { .. }
        | SubOp::DraftAttention { .. }
        | SubOp::DraftContextKv { .. }
        | SubOp::DraftWindowStore
        | SubOp::DraftWindowLoad
        | SubOp::DraftSelect => Unhandled(format!("{:?} has no SuperDSC kernel", node.op)),
        // ⭐ THE TOKEN DRAW — one kernel, spliced like every other: the logits rows, the per-row
        // seat words and reciprocal temperature (per-step operands the runtime binds), the
        // vocab-derived tables (tape sources staged at load), the idx stick out. `lower_all`: a
        // prefill rung's draw reads the folded tail's answer rows, `lower_all`'s machinery.
        //
        // ⭐ THE CHAINED HEAD'S PAIR — spliced like the draw: the stage decomposes the device's
        // pick into the next step's gather indices, and the gather reads the embedding row the pick
        // names out of the tied table, so the step's token never leaves the card.
        SubOp::Sample { .. } | SubOp::EmbedRowGather { .. } | SubOp::ChainStage { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests, query_rows) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        // ── THE 2a KINDS — every one through `lower_all`: the TanhSoftCap logits fold
        // and the head-norm sandwich's three seats (Reshape split / norm middle /
        // Reshape flatten) are `lower_all`'s interceptions, and every other node of
        // these kinds is the ordinary one-program splice. The Reshape arm this
        // replaces used to refuse the RESTICKIFY by name — the stick-walk kernel
        // (`reshape.py`, one `[1, 64]` identity per flat stick through the door's own
        // `lmlast` mechanism) IS that re-laying copy now. ──
        SubOp::TanhSoftCap { .. }
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::Reshape { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests, query_rows) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::MatmulTile { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests, query_rows) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        // ⛔ NO BODY, AND THAT IS THE HONEST STATE. This arm used to lower a `SubtileNode` STRAIGHT
        // TO SuperDSC descriptors — the same violation as the five `*_sdsc` bypasses that were
        // deleted, and the last one left: it was the only producer arm handing the bake an
        // `EmittedOp` with a descriptor and no KTIR program.
        //
        // It is deleted rather than ported because NOTHING IN THIS REPOSITORY CONSTRUCTS A
        // `SubOp::SumReduce` NODE. Every occurrence of the variant is the enum declaration, an
        // arity/ABI table row, a re-roll hash-class arm, the host `eval_node` reference
        // implementation, or a consumer `match` arm — there is no site that builds one, so the
        // split-K combine its doc describes is not emitted by the shared front end. `ibm/main` is
        // the same: it dispatches `lower_sumreduce_node` from `lower_one_node` (main 10698) over a
        // node kind nothing produces, so that body is unreachable there too.
        //
        // A port would therefore be main's text written to satisfy a rule, with no model able to
        // exercise it. A future arch that DOES emit one gets this loud bake error, which names where
        // the body comes from — not a second path to SuperDSC.
        SubOp::SumReduce { .. } => Unhandled(format!(
            "SubOp::SumReduce t{} has no KTIR lowering. Nothing in this repository constructs the \
             node, so no body here has ever run; the port source is `ibm/main`'s \
             `lower_sumreduce_node` (main 8432-8466), a variadic pointwise `add` over \
             `[rows, cols]`. Port it through the splice — never straight to a descriptor.",
            node.output.tensor.index() as u32,
        )),
        SubOp::Elementwise(_) | SubOp::SiluMul => {
            match scratchy_triton_splice::lower(node, ir, rows_are_requests) {
                Ok(e) => Ops(vec![e]),
                Err(reason) => Unhandled(reason),
            }
        }
        // ⭐ BOTH GAIN CONVENTIONS ARE ROWS NOW. The (1 + w) convention had its own
        // refusal here (the Scale kernel over a zero-centred gain scales every
        // normalized activation by roughly nothing — a model that loads, runs, and
        // is quietly wrong); the oneplus row landed (`rmsnorm_oneplus_fwd`, the
        // door's `rmga_` add), so the refusal arm is deleted and every convention
        // routes through the splice — which is the trade the front end's EXPRESSING
        // the convention always wanted: the IR carries the fact, and the target's
        // row states it.
        //
        // ⭐ `lower_all`, NOT `lower`: a gained rmsnorm (either convention) may be a
        // HEAD-NORM SANDWICH'S MIDDLE — its computation is the fused windowed program
        // the split's own lowering emitted (`lower_all`'s interception), so the
        // one-op path is only for the norms that are NOT a sandwich seat.
        SubOp::RmsNorm { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests, query_rows) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::RopeRotate { .. } => {
            match scratchy_triton_splice::lower(node, ir, rows_are_requests) {
                Ok(e) => Ops(vec![e]),
                Err(reason) => Unhandled(reason),
            }
        }
        // ⭐ A KV WRITER LOWERS TO ITS ROTATION AND ITS PAGED WRITE — `lower_all` emits both, the
        // scatter into the pools right after the rotation it stores.
        SubOp::RopeAppend { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests, query_rows) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        // ⭐ THE LOGITS SCALARMUL RIDES `lower_all` WITH THE MATMUL — the vocab-wide
        // tail folds BOTH halves to m=1 (main's `is_prefill_lm_head_tail` covers this
        // arm too), and the fold is the splice's `lower_all` now. Every other
        // ScalarMul is the ordinary one-op pointwise.
        //
        // ⭐⭐ EXCEPT THE ONES THAT NEVER NEED A DEVICE OP AT ALL: a multiplier whose product
        // scales a MATMUL OUTPUT commutes into the matmul's weight (`(x·W)·s == x·(W·s)`), so
        // [`scalar_mul_weight_folds`] takes it out of the tape and into the staged weight bytes
        // — the multiplier stops being a per-step `[1,1]` const + a pointwise `mul` and becomes
        // a load-time fact. See that function for why the recognition refuses shared weights.
        SubOp::ScalarMul { .. } => {
            if scalar_mul_weight_folds(ir)
                .iter()
                .any(|(id, _, _)| *id == node.id)
            {
                return HostRouted("ScalarMulWeightFold");
            }
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests, query_rows) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::AttnDecode { .. } => {
            // ⭐ THE PAGED WALK READS ITS POOLS AND ITS GEOMETRY OFF THE FACTS, and the rung's
            // walk off `active_cap` — the splice states every one as the kernel's constexprs.
            // ⭐⭐⭐ THE SCALE'S SITE, DECIDED WHERE THE FOLD COLLECTOR DECIDES. When the fold
            // applies, the program is the ONE carrier of the fact (the kernel states it as its
            // `SCALE` constexpr and the door reads that value back): `FoldedIntoWeights` states
            // 1.0 there — the door emits no `qs`/`new_k_scaled` multiplies and the emulator
            // interprets `qk · 1.0` over the pre-scaled weights the fold staged — while the
            // real multiplier rides the fold list into the staged W_q/W_k bytes, where
            // `(q·√s)·(k·√s)ᵀ` reproduces `(q·kᵀ)·s` exactly.
            let folded = attn_scale_weight_folds(ir)
                .iter()
                .any(|(id, ..)| *id == node.id);
            let scale_site = if folded {
                scratchy_subtile::subtile_ir::AttnScaleSite::FoldedIntoWeights
            } else {
                scratchy_subtile::subtile_ir::AttnScaleSite::InKernel
            };
            match scratchy_triton_splice::lower_attn(
                node,
                ir,
                scale_site,
                active_cap,
                query_rows,
                scratchy_triton_splice::PagedRows::of_requests(rows_are_requests),
            ) {
                // ⭐ THE PREFILL FAMILY — a packed rung's attention lowers to one op per window
                // (the splice's own law), so the family extends the op list whole; a solo rung's
                // family is the one op it has always extended.
                Ok(family) => Ops(family),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::RmsNormReduce { .. } | SubOp::RmsNormApply { .. } => {
            HostRouted(host_glue_kind(&node.op))
        }
    };
    // ⭐ EVERY REFUSAL NAMES ITS OP. The splice's Err says WHAT broke; this adds WHICH node was
    // breaking, through the op registry's own name (no parallel string table) — the same law the
    // door's errors state (`AttnDecode t188: ...`), so a bake refusal localizes to one op
    // instead of a reason string over an unnamed graph.
    match lowered {
        Unhandled(why) => Unhandled(format!(
            "{} t{}: {why}",
            node.op.name(),
            node.output.tensor.index()
        )),
        other => other,
    }
}
/// The PRODUCER half over a whole UNROLLED graph: one KTIR program per node, plus the bundle layout
/// every one of them resolves its addresses through.
///
pub fn lower_graph_to_ktir<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    active_cap: ActiveCap,
    // See main's `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
    // The rung's query extent pair — threaded to the attention splice ([`lower_one_node`]).
    query_rows: QueryExtent,
) -> Result<(Vec<EmittedOp<T>>, BundleLayout), SuperDscError> {
    // GLOBAL ≤7-segment memory layout (task #55) computed ONCE for the whole bundle.
    // Every op resolves each tensor's HBM address from this (by `t{id}` name) so a
    // tensor shared across ops gets the SAME address — fixing the per-op `arg_index`
    // segment-aliasing bug. Threaded as `Some(&layout)` into every node-lowering; the
    // populated layout (incl. synthetic seg3 offsets) is RETURNED for the manifest.
    // ⛔ NO LAYER CLASSES: this is the UNROLLED lowering, which has no layer loop and therefore no
    // layer boundary to split the weight segment on. An empty map is what tells the layout that
    // banking is not expressible here, leaving the tail spill as the only lever (`&Default::default()`
    // rather than a bool, so there is one spelling of "the layer structure" and not two).
    let bundle_layout = compute_bundle_layout::<T, F>(
        ir,
        weight_ids,
        scratchy_triton_splice::PagedRows::of_requests(rows_are_requests),
        &[],
        query_rows,
    )?;
    let mut ops: Vec<EmittedOp<T>> = Vec::with_capacity(ir.nodes.len());
    // Collect EVERY distinct unhandled op kind (the full worklist) in one pass —
    // reconnaissance, not a silent skip: a non-empty set is a HARD error so a
    // partial (silently-wrong) bundle is never baked (guard-every-crash rule).
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Ops explicitly ROUTED host-side (RoPE rotate / rope_append / decode attention /
    // pre-chunked RmsNorm halves) — recognized data-movement glue, not a silent skip.
    let mut host_routed: std::collections::BTreeSet<&'static str> =
        std::collections::BTreeSet::new();
    // UNROLLED walk: lower every node via the shared per-op `lower_one_node`. (The
    // RE-ROLLED tape-driven path reuses the SAME helper, walking only the loop body
    // once — see `lower_subtile_tape_to_superdsc`.) MatmulTile's Err is still a hard
    // stop; everything else collects into the worklist/host-routed sets below.
    for node in &ir.nodes {
        match lower_one_node(node, ir, active_cap, rows_are_requests, query_rows) {
            NodeLowering::Ops(v) => ops.extend(v),
            NodeLowering::Unhandled(s) => {
                // A MALFORMED matmul is an immediate hard stop (it must never bake);
                // every other op kind accumulates into the worklist for one combined Err.
                if matches!(node.op, SubOp::MatmulTile { .. }) {
                    return Err(SuperDscError(s));
                }
                unhandled.insert(s);
            }
            NodeLowering::HostRouted(s) => {
                host_routed.insert(s);
            }
        }
    }
    if !unhandled.is_empty() {
        return Err(SuperDscError(format!(
            "{} SubtileIR op kind(s) have no KTIR construction ({} program(s) built ok, \
             {} host-routed). WORKLIST: [{}]",
            unhandled.len(),
            ops.len(),
            host_routed.len(),
            unhandled.into_iter().collect::<Vec<_>>().join(", "),
        )));
    }
    Ok((ops, bundle_layout))
}

/// WHAT A LAUNCH BINDS: for every program, which SubtileIR tensor each of its parameters points
/// at, plus the graph-level facts the worker needs to place those tensors — how many are sources,
/// which one is the result, every tensor's shape, and the id of the runtime attention mask if the
/// graph has one.
///
/// ⛔ THE PAIRING IS CARRIED, NOT RE-DERIVED. A parameter is an ADDRESS; nothing in a finished
/// program says which buffer that address should be, so the only thing that can say is the
/// construction that minted the parameter. Re-deriving it downstream would be a second
/// implementation of the lowering, and the first divergence would show up as a program silently
/// reading someone else's tensor.
pub struct BundleWiring {
    pub nodes: Vec<NodeArgs>,
    pub num_sources: u32,
    pub result_tensor: u32,
    /// ⭐ THE RESULT'S OWN DTYPE — the fact that decides how the consumer reads
    /// the rung's answer back, stated by the RESULT PRODUCER'S OWN KERNEL
    /// SIGNATURE when the result is a spliced kernel's output
    /// ([`result_dtype_of_ir`]) and the f16 row every native tail emits
    /// otherwise. An INTEGER dtype states the result IS the picked token (the
    /// device argmax's idx stick): the consumer reads the raw bytes at this
    /// dtype instead of converting a `[1, vocab]` f16 row, and the row it was
    /// picked from stays readable by tid either way — the answer re-point's
    /// own contract, recorded on [`crate::wiring::Wiring::result_dtype`].
    pub result_dtype: scratchy_tensors::DType,
    /// `tensor_shapes[id] = (rows, cols)` for every tensor. When `attn_mask` is set, the LAST entry
    /// is the synthetic mask tensor `[1, capacity]`.
    pub tensor_shapes: Vec<(u32, u32)>,
    /// Tensor id of the shared attention mask-stack source, if the graph has attention. The host
    /// fills it per forward step (see [`crate::wiring::PagedStep`]); it is NOT one of
    /// `num_sources` and is written by no node.
    pub attn_mask: Option<u32>,
    /// The paged step's sources and pool — see [`crate::wiring::PagedStep`].
    pub paged: crate::wiring::ClassSteps,
    /// The weight folds this graph admits: `(tensor id, multiplier)` for each matmul weight the
    /// worker must scale at load instead of the tape applying the multiplier at runtime —
    /// EITHER a `ScalarMul` fold ([`scalar_mul_weight_folds`], whose refusal set — sliced reads,
    /// shared fold target, shared producer output, non-matmul producer — is that fold's safety
    /// envelope) or an attention-scale fold ([`attn_scale_weight_folds`], the same envelope read
    /// through the rope, `√scale` into `W_q`/`W_k`). The runtime applies both kinds identically:
    /// one multiply over the staged bytes at load.
    ///
    /// ⛔ THE UNROLLED WALK THAT PRODUCES THIS WIRING HOST-ROUTES THE FOLDED SCALARMUL NODES
    /// TOO: the same recognition gates `lower_one_node`'s ScalarMul arm, so a wiring that ran
    /// the multiplier as a program while the card path folded it would be a wiring that
    /// double-applies the scale. (An attention-scale fold's node is NOT host-routed — it still
    /// runs, at the scale 1.0 the same arm rewrote it to.)
    pub weight_scale_folds: Vec<(u32, f32)>,
    /// ⭐ THE LOAD-TIME CONSTANTS — `(tid, kind)` for every resident buffer this graph's own
    /// producer minted below the `SourceBinding` layer ([`load_consts_of_ir`], the one derivation
    /// the numeric gate reads too). The worker stages each ONCE at load through the kind's builder.
    pub load_consts: Vec<(u32, crate::wiring::LoadConst)>,
    /// ⭐ THE SAMPLER'S PER-STEP SEAT — `Some` exactly when the graph carries the minted `Sample`
    /// node ([`sample_seat_of_ir`]).
    pub sample_seat: Option<crate::wiring::SampleSeat>,
    /// ⭐ THE CHAINED HEAD'S STICK FACT — [`crate::wiring::ChainFact::Chained`] iff the mint
    /// spliced the EmbedRowGather/ChainStage pair into this graph ([`chain_of_ir`]).
    pub chain: crate::wiring::ChainFact,
}

/// One program's parameter order: `args[i]` is the tensor the i-th parameter addresses.
pub struct NodeArgs {
    pub args: Vec<usize>,
}

/// ⭐ THE HOST PAIR FOR A GEMM WEIGHT IS THE UNION OF ITS MATMUL CONSUMERS' VIEW REGIONS
/// — never the tensor's STATED shape. The pair is what the host STAGES: the wiring's
/// static `tensor_shapes` and the JSON tensors manifest are emitted from it (codegen's
/// `shapes` tokens and the per-tensor `{"rows","cols"}` entries), and the worker's
/// `stage_bound_weights` sizes everything off it — the disk guard (`rows·cols·elem`
/// against the checkpoint bytes), the re-tile, the `DeviceWidth` N-pad, and the fp8
/// K-pad rebuild. The STATED shape is what the placement RESERVES, while every matmul's
/// weight view keeps the region the lower baked from the OP DESCRIPTOR'S OWN `n`
/// (`Range::new(0, n)` — never the tensor's stated cols), a partial view over a tensor
/// stated wider.
///
/// ONE WALK, ONE MAP, every stager reads: `graph_wiring`'s `tensor_shapes` and the
/// three-width-laws check's worker law both take their pairs from HERE, so the host
/// pair cannot drift between the wiring and the check — structurally, not by
/// convention. Consumer-less tids deliberately miss the map and keep their stated
/// pairs: per-layer copies have no node of their own in the rolled graph (the class's
/// representative carries the consumer, and no copy tid is ever at a node's inputs).
///
/// THE UNION LAW — a B operand is read WHOLE, from zero: every MatmulTile consuming
/// the weight at `inputs[1]` (arity 2 or 3 — the same gate the kernel0 collection
/// walks) contributes its view's rows and cols spans, and each axis merges to ONE
/// interval that must START AT ZERO (a lone consumer at [100, 200) is refused as
/// firmly as a gap — the host pair would under-state the checkpoint and the failure
/// would move to load), carry NO GAP between consecutive consumers (an NK-split's
/// column blocks and a K-slice's row blocks tile exactly — `split_oversized_loads_pass`'s
/// contract, stated where the lower bakes its regions), and NOT EXCEED the tensor's
/// stated shape on either axis (the typed view-overrun guard; the merged end is the
/// max consumer end, so one check covers every consumer). For every landed weight the
/// One axis's consumer spans — each consumer's `(start, end)` on that axis.
type WeightSpans = Vec<(u32, u32)>;

/// union equals the stated shape — whole gemms by their single consumer, splits by
/// their tiling — byte-identical to a stated-driven pair; for the widened tied embed
/// it is the REAL vocab while the stated cols carry the pad.
pub(crate) fn gemm_weight_host_pairs<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Result<std::collections::BTreeMap<u32, (u32, u32)>, SuperDscError> {
    // Per weight tid: the consumers' (start, end) spans, rows and cols separately.
    let mut views: std::collections::BTreeMap<u32, (WeightSpans, WeightSpans)> =
        std::collections::BTreeMap::new();
    for n in &ir.nodes {
        if !matches!(n.op, SubOp::MatmulTile { .. }) || (n.inputs.len() != 2 && n.inputs.len() != 3)
        {
            continue;
        }
        let w = n.inputs[1].tensor.index() as u32;
        let r = &n.inputs[1].region;
        let entry = views.entry(w).or_default();
        entry.0.push((r.rows.start, r.rows.end()));
        entry.1.push((r.cols.start, r.cols.end()));
    }
    let mut pairs = std::collections::BTreeMap::new();
    for (w, (row_spans, col_spans)) in views {
        let stated = ir.tensors[w as usize];
        let merge = |axis: &str,
                     mut spans: Vec<(u32, u32)>,
                     stated_len: u32|
         -> Result<u32, SuperDscError> {
            spans.sort_unstable();
            if spans[0].0 != 0 {
                return Err(SuperDscError(format!(
                    "gemm weight t{w}: its matmul consumers' {axis} views start at {}, not 0 — \
                     a B operand is read whole from zero, so the host pair would under-state \
                     the checkpoint and the failure would move to load",
                    spans[0].0
                )));
            }
            let mut end = 0;
            for &(start, span_end) in &spans {
                if start > end {
                    return Err(SuperDscError(format!(
                        "gemm weight t{w}: its matmul consumers' {axis} views leave a gap \
                         {end}..{start} — an NK or K split tiles exactly, so this is a \
                         lowering defect, not a staging fact"
                    )));
                }
                end = end.max(span_end);
            }
            if end > stated_len {
                return Err(SuperDscError(format!(
                    "gemm weight t{w}: its matmul consumers' {axis} views reach {end} past the \
                     tensor's stated extent {stated_len} — a view OVERRUNNING its own tensor, \
                     which no downstream law can catch"
                )));
            }
            Ok(end)
        };
        let rows = merge("rows", row_spans, stated.rows)?;
        let cols = merge("cols", col_spans, stated.cols)?;
        pairs.insert(w, (rows, cols));
    }
    Ok(pairs)
}

/// The wiring's `tensor_shapes` — every tensor's HOST pair: the stated shape, with every
/// gemm weight's entry replaced by the union of its matmul consumers' view regions
/// ([`gemm_weight_host_pairs`]' law).
pub(crate) fn tensor_shapes_of_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Result<Vec<(u32, u32)>, SuperDscError> {
    let mut shapes: Vec<(u32, u32)> = ir.tensors.iter().map(|t| (t.rows, t.cols)).collect();
    for (&w, &(rows, cols)) in &gemm_weight_host_pairs(ir)? {
        shapes[w as usize] = (rows, cols);
    }
    Ok(shapes)
}

/// ⭐ THE PAGED STEP THIS GRAPH's ATTENTION READS — off its own `AttnDecode` nodes: the pool every
/// layer's cache tensor is (`rows / SchedulerBlockTokens::REFERENCE` pages), the forward's query rows, and the KV heads
/// one write row per (token, head) is minted for. One step serves every layer — one page table, one
/// mask stack, one write-row table — so the layers must agree; a graph whose attentions differ in
/// pool, rows or kv heads is refused rather than served one layer's tables.
///
/// `active_cap` is the rung the bundle is baked at: its walk ([`scratchy_triton_splice::walk_pages`],
/// the splice's own law) is the one fact of the step that depends on it. `paged_rows` says whether
/// the rows are a decode batch, whose chunking of that walk the batched kernel reads.
/// `query_rows` is the rung's minted extent pair — the ONE mint (the bake's codegen, off the rung
/// itself, never a division of the graph's rows), checked here against the graph it describes: the
/// pair's held product must equal the attention's `mq` rows, and a pair that does not is refused by
/// name — a bake bug, never a silently-wrong per-request width the layout would size by.
pub fn paged_step_of_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
    active_cap: ActiveCap,
    paged_rows: scratchy_triton_splice::PagedRows,
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
) -> Result<crate::wiring::ClassSteps, SuperDscError> {
    let slots = scratchy_triton_splice::SchedulerBlockTokens::REFERENCE.get();
    let mut out = crate::wiring::ClassSteps::default();
    for n in &ir.nodes {
        let SubOp::AttnDecode {
            mask, attention, ..
        } = &n.op
        else {
            continue;
        };
        let cache = n.inputs.get(1).ok_or_else(|| {
            SuperDscError(format!(
                "AttnDecode t{} states no K cache input",
                n.output.tensor.index()
            ))
        })?;
        let rows = ir.tensors[cache.tensor.index()].rows;
        if rows == 0 || !rows.is_multiple_of(slots) {
            return Err(SuperDscError(format!(
                "AttnDecode t{}: its K cache holds {rows} slots, not a whole number of \
                 {slots}-slot pages",
                n.output.tensor.index()
            )));
        }
        let (pages, mq) = (rows / slots, n.output.region.rows.len);
        if query_rows.graph_rows().get() != mq {
            return Err(SuperDscError(format!(
                "AttnDecode t{}: the graph runs {mq} query row(s) but the rung's minted extent \
                 states {} ({} per request, {} request(s)) — the mint and the graph disagree, a \
                 bake bug named here",
                n.output.tensor.index(),
                query_rows.graph_rows().get(),
                query_rows.per_request().get(),
                query_rows.requests().get(),
            )));
        }
        let walk = scratchy_triton_splice::walk_pages(pages, mq, active_cap);
        let tids = scratchy_triton_splice::PagedStepTids::of(mask.class());
        let step = crate::wiring::PagedStep {
            mask: tids.mask,
            page_table: tids.page_table,
            write_rows: tids.write_rows,
            query_rows: tids.query_rows,
            rep_rows: tids.rep_rows,
            chunk_pages: tids.chunk_pages,
            chunk_masks: tids.chunk_masks,
            pages,
            page_slots: slots,
            rows: mq,
            nkvh: attention.class(mask.class()).kv_heads().get(),
            q_width: attention.class(mask.class()).q_width(),
            head_size: attention.class(mask.class()).head_dim().get(),
            walk,
            paged_rows,
            extent: query_rows,
            attends: *mask,
        };
        let seat = match mask.class() {
            crate::wiring::AttnClass::Full => &mut out.full,
            crate::wiring::AttnClass::Sliding => &mut out.sliding,
        };
        match *seat {
            Some(prev) if prev != step => {
                return Err(SuperDscError(format!(
                    "the graph's {:?} attentions disagree about their paged step ({prev:?} vs \
                     {step:?}) — one page table, one mask stack and one write-row table serve \
                     every layer of a class",
                    mask.class()
                )));
            }
            _ => *seat = Some(step),
        }
    }
    Ok(out)
}

/// Lower `ir` for its WIRING alone — the same per-node arms a bake runs, read for the parameter
/// pairings rather than for the programs.
///
/// The pairings do not depend on the rung, so the arms lower at `ActiveCap::FULL`; `active_cap` is the
/// rung the bundle is baked at, read for the one fact that does — the paged step's walk.
pub fn graph_wiring<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    ir: &SubtileIR<F>,
    active_cap: ActiveCap,
    paged_rows: scratchy_triton_splice::PagedRows,
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
) -> Result<BundleWiring, SuperDscError> {
    let batch = paged_rows == scratchy_triton_splice::PagedRows::Batch;
    let per_node: Vec<Result<Vec<NodeArgs>, SuperDscError>> = {
        use rayon::prelude::*;
        ir.nodes
            .par_iter()
            .map(|node| {
                let ops = match lower_one_node::<T, F>(node, ir, ActiveCap::FULL, batch, query_rows)
                {
                    NodeLowering::Ops(v) => v,
                    NodeLowering::HostRouted(_) => Vec::new(),
                    NodeLowering::Unhandled(why) => return Err(SuperDscError(why)),
                };
                Ok(ops
                    .into_iter()
                    .map(|e| NodeArgs {
                        args: e.ktir.bindings.iter().map(|b| b.get() as usize).collect(),
                    })
                    .collect())
            })
            .collect()
    };
    let mut nodes = Vec::with_capacity(ir.nodes.len());
    for lowered in per_node {
        nodes.extend(lowered?);
    }
    let paged = paged_step_of_ir(ir, active_cap, paged_rows, query_rows)?;
    Ok(BundleWiring {
        nodes,
        num_sources: ir.num_sources,
        result_tensor: ir.result.index() as u32,
        result_dtype: result_dtype_of_ir(ir)?,
        tensor_shapes: tensor_shapes_of_ir(ir)?,
        // The attention reads its mask stack at the paged step's own tid.
        attn_mask: paged
            .first()
            .map(|p| if batch { p.chunk_masks } else { p.mask }),
        paged,
        weight_scale_folds: weight_scale_folds(ir)
            .into_iter()
            .map(|(_, tid, scale)| (tid, scale))
            .collect(),
        load_consts: load_consts_of_ir(ir)?,
        sample_seat: sample_seat_of_ir(ir)?,
        chain: chain_of_ir(ir)?,
    })
}
// ─────────────────────────────────────────────────────────────────────────────

/// ⭐ THE SAMPLER'S PER-STEP SEAT, read off the minted `Sample` node alone — the one derivation
/// ([`crate::wiring::SampleSeat`] names the contract; the expansion's source gate and the worker's
/// per-step bind read the class, never the node). `None` on a graph the mint did not touch; a
/// second minted node is a build error, never last-write-wins.
pub fn sample_seat_of_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Result<Option<crate::wiring::SampleSeat>, SuperDscError> {
    let Some(node) = the_sample(ir)? else {
        return Ok(None);
    };
    let SubOp::Sample { vocab, rows } = node.op else {
        unreachable!("the_sample answers a Sample")
    };
    let seed_words: [u32; 4] = std::array::from_fn(|k| {
        let word = node.inputs[1 + k].tensor;
        assert_eq!(
            ir.tensors[word.index()].cols,
            1,
            "the seed word table t{} is not one word per row — the mint and the seat disagree on \
             the seed's shape",
            word.index()
        );
        word.index() as u32
    });
    Ok(Some(crate::wiring::SampleSeat {
        seed_words,
        invt: node.inputs[5].tensor.index() as u32,
        rows,
        row: node.inputs[0].tensor.index() as u32,
        vocab,
        pad_vocab: scratchy_subtile::subtile_ir::sample_pad_vocab(vocab),
        // The idx stick is `[rows, SAMPLE_BLOCK]`, its rows dense at its own width.
        idx_row_words: ir.tensors[node.output.tensor.index()].cols,
    }))
}

/// ⭐ THE LOAD-TIME CONSTANTS — `(tid, kind)` for every resident buffer the graph's own producer
/// minted below the `SourceBinding` layer: the sampler's twenty tables (its node's inputs past the
/// row, the four seed words and the temperature, which are per-step operands and never here) and
/// the chained head's two decompose tables (its stage's inputs past the pick and the sticks). The
/// one derivation the wiring, the placement's resident set and the numeric gate read.
pub fn load_consts_of_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Result<Vec<(u32, crate::wiring::LoadConst)>, SuperDscError> {
    let mut consts = Vec::new();
    if let Some(node) = the_sample(ir)? {
        let SubOp::Sample { vocab, rows } = node.op else {
            unreachable!("the_sample answers a Sample")
        };
        let arrangements = sample_table_arrangements(node, ir)?;
        for (tr, operand) in node.inputs[6..].iter().zip(1u8..) {
            let tid = tr.tensor.index() as u32;
            consts.push((
                tid,
                crate::wiring::LoadConst::SampleTable {
                    vocab,
                    rows,
                    operand,
                    arrangement: arrangements
                        .get(&tid)
                        .copied()
                        .unwrap_or(crate::wiring::ElementArrangement::Standard),
                },
            ));
        }
    }
    if let Some(ChainPair { stage, gather }) = the_chain(ir)? {
        let SubOp::ChainStage { table_rows, .. } = stage.op else {
            unreachable!("the_chain answers a ChainStage")
        };
        let block = ir.shape(gather.inputs[2].tensor).cols;
        consts.extend(stage.inputs[3..5].iter().zip(1u8..).map(|(tr, operand)| {
            (
                tr.tensor.index() as u32,
                crate::wiring::LoadConst::ChainStageTable {
                    table_rows,
                    block,
                    operand,
                },
            )
        }));
    }
    Ok(consts)
}

/// ⭐ THE CHAINED HEAD'S STICK FACT — the two persistent sources the `ChainStage` WRITES each step
/// and the `EmbedRowGather` READS the next (the stage's inputs 1 and 2), the row the chain
/// replaced (the gather's own `replaces`) and the table columns the gather reads (its matrix
/// operand's stated width) — all the mint's own statements, read once. `Unchained` is every graph
/// the mint did not touch.
pub fn chain_of_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Result<crate::wiring::ChainFact, SuperDscError> {
    let Some(ChainPair { stage, gather }) = the_chain(ir)? else {
        return Ok(crate::wiring::ChainFact::Unchained);
    };
    let SubOp::EmbedRowGather { replaces, .. } = gather.op else {
        unreachable!("the_chain answers an EmbedRowGather")
    };
    Ok(crate::wiring::ChainFact::Chained {
        chain: crate::wiring::ChainWiring {
            s_src: stage.inputs[1].tensor.index() as u32,
            l_src: stage.inputs[2].tensor.index() as u32,
            replaces: replaces.index() as u32,
            block: ir.shape(gather.inputs[2].tensor).cols,
        },
    })
}

/// The chained head's two minted nodes.
struct ChainPair<'g, F: RopeForm> {
    stage: &'g SubtileNode<F>,
    gather: &'g SubtileNode<F>,
}

/// The graph's minted chained-head pair: `None` when the mint did not run, an error when it ran
/// twice or left one half.
fn the_chain<F: RopeForm>(ir: &SubtileIR<F>) -> Result<Option<ChainPair<'_, F>>, SuperDscError> {
    let of = |want: fn(&SubOp<F>) -> bool| -> Vec<&SubtileNode<F>> {
        ir.nodes.iter().filter(|n| want(&n.op)).collect()
    };
    let stages = of(|o| matches!(o, SubOp::ChainStage { .. }));
    let gathers = of(|o| matches!(o, SubOp::EmbedRowGather { .. }));
    match (&stages[..], &gathers[..]) {
        ([], []) => Ok(None),
        ([stage], [gather]) => Ok(Some(ChainPair { stage, gather })),
        _ => Err(SuperDscError(format!(
            "{} ChainStage and {} EmbedRowGather nodes in one graph — the mint splices exactly \
             one pair into a rung",
            stages.len(),
            gathers.len()
        ))),
    }
}

/// ⭐ THE ARRANGEMENT EACH OF THE DRAW'S TABLES IS STAGED IN — off the draw's own lowered program
/// ([`spyre_superdsc::arrangement::staged_arrangements`], which reads the device-facts table's
/// widen row): a table the program reads lane-for-lane beside a widened value is staged in the
/// widen's lane order. The program is the splice's own for this node, planned with no layouts (the
/// tables are activation-segment sources, held as their views state).
fn sample_table_arrangements<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<std::collections::BTreeMap<u32, crate::wiring::ElementArrangement>, SuperDscError> {
    let e = scratchy_triton_splice::lower::<scratchy_spyre_bundle::collective::TpWorld<1>, F>(
        node, ir, false,
    )
    .map_err(SuperDscError)?;
    let k = &e.ktir;
    let planned = spyre_superdsc::plan::<ktir_core::ddl_binds::Baked>(
        &k.func,
        &vec![None; k.func.arguments.len()],
    )
    .map_err(|err| SuperDscError(format!("the draw's program: {err:?}")))?;
    let tables: std::collections::HashSet<u32> = node.inputs[6..]
        .iter()
        .map(|tr| tr.tensor.index() as u32)
        .collect();
    let tids: Vec<u32> = k.bindings.iter().map(|b| b.get()).collect();
    let staged: Vec<bool> = tids.iter().map(|t| tables.contains(t)).collect();
    let arrangements = spyre_superdsc::arrangement::staged_arrangements::<
        ktir_core::ddl_binds::Baked,
    >(&planned.graph, &staged)
    .map_err(|err| {
        SuperDscError(format!(
            "the draw's program cannot hold one lane order per buffer: {err:?}"
        ))
    })?;
    Ok(tids.into_iter().zip(arrangements).collect())
}

/// The graph's one minted `Sample` node: `None` when the mint did not run, an error when two did.
fn the_sample<F: RopeForm>(ir: &SubtileIR<F>) -> Result<Option<&SubtileNode<F>>, SuperDscError> {
    let mut mints = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::Sample { .. }));
    match (mints.next(), mints.next()) {
        (Some(_), Some(_)) => Err(SuperDscError(
            "two Sample nodes in one graph — the mint appends exactly one to a rung; a second \
             means two mints ran over one graph"
                .to_string(),
        )),
        (node, _) => {
            if let Some(n) = node {
                assert_eq!(
                    n.inputs.len(),
                    26,
                    "the Sample node's arity is the registry's own row (logits + four seed word \
                     tables + inv_t + twenty tables)"
                );
            }
            Ok(node)
        }
    }
}

/// ⭐ THE RESULT'S OWN DTYPE, from the RESULT PRODUCER'S OWN KERNEL SIGNATURE —
/// the same `operand_dfs` authority the placement widths already read (the
/// registry's operand-order contract zips the kernel's buffer parameters with
/// the node's bindings, output LAST), asked here for the one binding that
/// names the result. A result no spliced kernel produces is the f16 row every
/// native lm-head tail emits; a spliced result carries whatever dtype its
/// kernel's own parameter states — the device argmax's `idx_ptr: *i32` makes
/// the minted rung's answer an INDEX, and the consumer's raw-read branch keys
/// on exactly that fact, never on the op's name.
pub fn result_dtype_of_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Result<scratchy_tensors::DType, SuperDscError> {
    let node = ir
        .nodes
        .iter()
        .find(|n| n.output.tensor == ir.result)
        .expect("validate demands the result is an op output, so it has a writer");
    let dfs = scratchy_triton_splice::operand_dfs(node, ir)
        .map_err(|e| SuperDscError(format!("the result's dtype: {e}")))?;
    let Some(dfs) = dfs else {
        return Ok(scratchy_tensors::DType::F16);
    };
    dfs.iter()
        .find(|(tid, _)| *tid == ir.result.index() as u32)
        .ok_or_else(|| {
            SuperDscError(
                "the result's producer binds no parameter for the result tensor — the \
                 registry's operand-order contract (output LAST) is violated"
                    .to_string(),
            )
        })
        .and_then(|(_, df)| match df {
            spyre_superdsc::format::DataFormat::Fp16 => Ok(scratchy_tensors::DType::F16),
            spyre_superdsc::format::DataFormat::Fp32 => Ok(scratchy_tensors::DType::F32),
            spyre_superdsc::format::DataFormat::Fp8 => Ok(scratchy_tensors::DType::Fp8E4m3),
            spyre_superdsc::format::DataFormat::UInt32
            | spyre_superdsc::format::DataFormat::Int32 => Ok(scratchy_tensors::DType::I32),
            spyre_superdsc::format::DataFormat::Int64 => Err(SuperDscError(
                "the result's kernel states an int64 output, which no consumer reads back — \
                 state a dtype the readback knows (f16, f32, fp8, i32)"
                    .to_string(),
            )),
        })
}

/// ⭐ ONE NODE'S OUTPUT STICK, BY THE SAME SIGNATURE AUTHORITY — `true` for
/// every native op (the f16 stick the reference legs' hop-by-hop rounding
/// law states) and for a spliced kernel whose own parameter states an f16
/// output; `false` when the signature states another stick, because the f16
/// rounding hop is the F16 STICK's law, never the value's. Measured on the
/// argmax mint: the idx stick is the kernel's own `*i32` buffer, and the
/// blanket f16 rounding certified idx 12358 as 12360 (the f16 grid's spacing
/// at [8192, 16384) is 8) against a faithful emulator answer of 12358 — a
/// WRONG TOKEN as the reference law, surfaced the moment the gate began
/// comparing the stick exactly. An integer stick holds whole f32 numbers
/// EXACTLY; the rounding hop is a corruption there, not a grid statement.
pub fn node_output_on_f16_stick<F: RopeForm>(
    node: &scratchy_subtile::subtile_ir::SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> bool {
    let dfs = match scratchy_triton_splice::operand_dfs(node, ir) {
        Ok(Some(dfs)) => dfs,
        // A native op, or a spliced ROW without a kernel signature (attention,
        // whose widths are the door's statement): the f16 stick.
        Ok(None) => return true,
        Err(e) => panic!(
            "a spliced node's own output stick is unreadable — the reference legs refuse to \
             guess the grid it rounds through: {e}"
        ),
    };
    let Some((_, df)) = dfs
        .iter()
        .find(|(tid, _)| *tid == node.output.tensor.index() as u32)
    else {
        panic!(
            "the spliced kernel for {:?} binds no parameter for its own output — the \
             registry's operand-order contract (output LAST) is violated",
            node.op
        )
    };
    matches!(df, spyre_superdsc::format::DataFormat::Fp16)
}

/// RE-ROLLED tape-driven SuperDSC lowering — the mirror of `lower_subtile_tape_to_tk_tape`
/// for SuperDSC (the fix for the 40-min unrolled-bundle compile). Walks the rerolled
/// `tape` (`subtile_tape::reroll_subtile_tape`): pre-loop Computes → `prefix`; the
/// `OpenLoop(Const(iters))`..`CloseLoop` body → `body` (lowered ONCE via the shared
/// `lower_one_node`); post-loop Computes → `suffix`. Collects the per-layer tid map
/// (`Compute::per_layer_out` for outputs + `ComputeInput::External::per_layer` for
/// weights) for the executor's per-iteration address advance.
pub fn lower_subtile_tape_to_ktir<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    tape: &scratchy_subtile::subtile_tape::SubtileTape,
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    // The driver calls this once per rung with a different `active_cap`; every rung shares the SAME
    // resident KV (storage stays `cap`) and differs only in the body's swept extents.
    active_cap: ActiveCap,
    // See main's `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
    // The rung's query extent pair — threaded to the attention splice ([`lower_one_node`]).
    query_rows: QueryExtent,
) -> Result<RolledSuperDsc<T>, SuperDscError> {
    use scratchy_subtile::subtile_tape::{ComputeInput, Instr, LoopBound};
    let _pc = crate::pc::Pc::start("producer");
    // ⛔ THE `set_rows_are_requests` / `RestoreRar` DANCE IS DELETED. It pushed the row KIND into a
    // thread-local for the duration of one bundle's lowering, with a `Drop` guard to restore it, purely
    // so the matmul splitter would not need the fact in its signature. That made the kind ambient: no
    // site was obliged to receive it, ~97 sites branched on the row COUNT instead, and a batched decode
    // was emitted as a prefill chunk everywhere the count could not tell them apart. The kind is a
    // COMPILE-TIME CONSTANT of the bundle (this crate is driven by a proc macro that knows the model and
    // the rung as literals), so it belongs in a type — `sdsc_abstract::QueryRows<ROWS_ARE_REQUESTS>` —
    // and in the signatures that need it.
    //
    // The one perf gate that outlived it here (a decode batch must not split `mb`) is gone as well: it
    // was scoped to this lowering, the descriptors are built later in `emit_bundle`, so it never reached
    // one, and the form it asked for measured slower on the card (`matmul/dims.rs`). The extent half
    // landed as the pair: [`QueryExtent`] states the rung's `(per-request width, requests)`, minted once
    // at the bake. The KIND stays the separate `rows_are_requests` bool above because the pair cannot
    // carry it alone: a decode batch mints `(1 row, m)` and a packed prefill rung will mint
    // `(w, S)` — both per-request width > 1 pairs in general, differing only in whether one launch
    // serves the requests as a batch or the rung relaunches per request.
    // ⭐ THE LAYER STRUCTURE FIRST, because the layout needs it: a weight BANK boundary may only fall
    // on a LAYER boundary, and `compute_bundle_layout` is where the weight segment is packed. Same
    // tape, same `ComputeInput::External::per_layer` the walk below reads — collected once, here.
    // ONE MAP PER ATTENTION CLASS: a class-split roll's loops carry different-length per-layer
    // tables, and the layout needs to know which class a tensor belongs to before it can pack.
    let per_layer_ext = per_layer_external_tids(tape);
    // The layout's budget passes (spill/bank) take the per-loop maps directly: the spill reasons
    // over WHICH tids are per-layer at all (flattened internally), and the BANK needs the class
    // boundaries to know which (class, iteration) launches must stay in one bank.
    let mut bundle_layout = compute_bundle_layout::<T, F>(
        ir,
        weight_ids,
        scratchy_triton_splice::PagedRows::of_requests(rows_are_requests),
        &per_layer_ext,
        query_rows,
    )?;
    // ── THE TAPE'S SEGMENTS: prefix, ONE RANGE PER LOOP (an attention class's body), suffix ──
    // The rolled tape is `prefix; [OpenLoop(body₀) CloseLoop]…[OpenLoop(bodyₙ) CloseLoop]; suffix`
    // — sibling loops, never nested (the TapeBuilder typestate forbids nesting). One `(range,
    // iters)` per loop, in tape order; the walk below lowers each separately.
    let instrs = tape.instrs();
    let mut loops: Vec<(std::ops::Range<usize>, u32)> = Vec::new();
    {
        let mut open: Option<(usize, u32)> = None;
        for (i, instr) in instrs.iter().enumerate() {
            match instr {
                Instr::OpenLoop { bound, .. } => {
                    if open.is_some() {
                        return Err(SuperDscError(
                            "reroll-superdsc: a nested layer loop — the TapeBuilder typestate \
                             forbids nesting, so the tape is not one this walk built"
                                .into(),
                        ));
                    }
                    let it = match bound {
                        LoopBound::Const(it) => *it,
                        LoopBound::Runtime(_) => {
                            return Err(SuperDscError(
                                "reroll-superdsc: a runtime-bounded layer loop is unsupported \
                                 (the decode layer count is a compile-time Const)"
                                    .into(),
                            ));
                        }
                    };
                    if it == 0 {
                        return Err(SuperDscError(
                            "reroll-superdsc: a zero-iteration layer loop — reroll found a \
                             repeating body of no layers"
                                .into(),
                        ));
                    }
                    open = Some((i, it));
                }
                Instr::CloseLoop { .. } => {
                    let Some((o, it)) = open.take() else {
                        return Err(SuperDscError(
                            "reroll-superdsc: a CloseLoop with no OpenLoop — unbalanced brackets"
                                .into(),
                        ));
                    };
                    loops.push((o + 1..i, it));
                }
                _ => {}
            }
        }
        if open.is_some() {
            return Err(SuperDscError(
                "reroll-superdsc: an OpenLoop with no CloseLoop — unbalanced brackets".into(),
            ));
        }
    }
    if loops.is_empty() {
        return Err(SuperDscError(
            "reroll-superdsc: no layer loop in the rerolled tape (no OpenLoop/CloseLoop) — \
             reroll_subtile_tape found no repeating body"
                .into(),
        ));
    }
    // The prefix needs no end marker: the walk below is a single pass whose `seg` state
    // machine routes ops to the prefix until the first OpenLoop flips it, so only the
    // suffix's start (the pre-pass below reads it) is named here.
    let suffix_start = loops[loops.len() - 1].0.end + 1;
    // ⛔ `iters` per body, from the loop bounds — NOT the count of instrs in the range.
    debug_assert_eq!(loops.len(), per_layer_ext.len()); // one map per sibling loop, by construction
    // ── The walk: prefix, each body, suffix, with per-segment lowering context ──
    // The residual-stream aliasing pre-pass first: thread the loop-carried hidden IN-PLACE, no copy.
    // Pre-scan EVERY body for hidden_in (its first node's input[0]) + hidden_out (its last node's
    // output) and alias them ALL onto the FIRST body's hidden_in placement → every launch of every
    // class reads+writes ONE resident buffer, so the residual threads with NO host round-trip and NO
    // device copy. WAR-safe: within a layer hidden_in's last read (computing h1 = h_in+attn)
    // precedes hidden_out's write (the final h_out = h1+mlp add); across launches the shared buffer
    // carries the residual. The suffix's input aliases the same buffer (the body→suffix seam).
    {
        let mut hidden_placement: Option<TensorPlacement> = None;
        let mut first_suffix_node: Option<u32> = None;
        for instr in &instrs[suffix_start..] {
            if let Instr::Compute { node, .. } = instr {
                first_suffix_node = Some(node.index() as u32);
                break;
            }
        }
        for (range, _) in &loops {
            let first = instrs[range.clone()]
                .iter()
                .find_map(|i| match i {
                    Instr::Compute { node, .. } => Some(*node),
                    _ => None,
                })
                .expect("a loop body holds at least one Compute");
            let last = instrs[range.clone()]
                .iter()
                .rev()
                .find_map(|i| match i {
                    Instr::Compute { node, .. } => Some(*node),
                    _ => None,
                })
                .expect("a loop body holds at least one Compute");
            let hin = ir.nodes[first.index()].inputs[0].tensor.index() as u32;
            let hout = ir.nodes[last.index()].output.tensor.index() as u32;
            let p = hidden_placement
                .take()
                .or_else(|| bundle_layout.placements.get(&hin).cloned())
                .expect("the first body's hidden-in has a layout placement");
            // hidden_out (last body op) → the shared hidden buffer: per-launch residual threads in
            // place; the NEXT body's hidden_in aliases the same buffer, so the cross-class seam
            // threads in place too.
            bundle_layout.placements.insert(hout, p);
            bundle_layout.placements.insert(hin, p);
            hidden_placement = Some(p);
        }
        // suffix_in (first suffix op's input[0]) → the same buffer.
        if let Some(s) = first_suffix_node {
            let sin = ir.nodes[s as usize].inputs[0].tensor.index() as u32;
            if let Some(p) = hidden_placement {
                bundle_layout.placements.insert(sin, p);
            }
        }
    }
    let mut prefix: Vec<EmittedOp<T>> = Vec::new();
    let mut suffix: Vec<EmittedOp<T>> = Vec::new();
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // ⭐ SEEDED FROM THE PRE-PASS RATHER THAN REBUILT. `per_layer_external_tids` already read every
    // per-layer WEIGHT/KV class off this same tape — it had to, because `compute_bundle_layout` needs
    // the layer boundaries to decide weight BANKS, and that runs before any op is emitted. The walk
    // below then adds only what the pre-pass cannot see (node outputs, the resident Kᵀ), and its own
    // `or_insert_with` is a no-op for anything already here.
    // ⭐ SEEDED FROM THE PRE-PASS RATHER THAN REBUILT (flattened — the walk's `or_insert`s below
    // are no-ops for anything already here). A table's KEY is its class's layer-0 tid; a table's
    // LENGTH is its class's `iters`.
    let mut per_layer: std::collections::BTreeMap<u32, Vec<u32>> = per_layer_ext
        .iter()
        .flat_map(|m| m.iter())
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    // Which weight BANKS each launch group addresses — one set per GROUP, where a group is now the
    // prefix (index 0), each body (1 + body index), or the suffix (last). A launch has ONE base per
    // segment, so a group may address exactly one; checked after the walk.
    let n_groups = loops.len() + 2;
    let mut group_weight_banks: Vec<std::collections::BTreeSet<u32>> =
        vec![Default::default(); n_groups];
    // ⭐ ONE `seg` PER GROUP: 0 = prefix, 1..=n = body i, n+1 = suffix — the same index
    // `group_weight_banks` is keyed by, so the walk's per-op bank insert needs no translation.
    let mut seg: usize = 0;
    // The BODIES, one per loop, filled as the walk crosses each OpenLoop. `bodies[bi].iters` is
    // that class's layer count; its `layer_ids` are collected from its own RopeAppend table below.
    let mut bodies: Vec<crate::lower_subtile_tape_to_superdsc::RolledBody<T>> =
        Vec::with_capacity(loops.len());
    // Each body's PER-LAYER REPRESENTATIVE tids — every External table + per-layer node output
    // this body's own instrs name, plus its kct. `body_of_tid` (the launch-table guard) uses this
    // to know which class a `per_layer` table belongs to; a table's KEY is the class's layer-0 tid,
    // and only the body that READS (or writes) that tid can own it.
    let mut body_representatives: Vec<Vec<u32>> = Vec::with_capacity(loops.len());
    // First SUFFIX Compute node — its input[0] is the residual the suffix reads (the
    // last layer's post-attn residual, e.g. t780). The rerolled body writes its output
    // to the REPRESENTATIVE-iteration tid (hidden_out_tid, e.g. t360), NOT t780, so the
    // body→suffix seam must thread hidden_out → suffix_in at runtime (else the suffix
    // reads an unwritten slot = 0 → rmsnorm(0)=inf → garbage logits).
    let mut first_suffix_node: Option<u32> = None;
    // The CURRENT body's first/last Compute node, reset at each OpenLoop and folded into that
    // body's hidden tids at its CloseLoop.
    let mut cur_first: Option<u32> = None;
    let mut cur_last: Option<u32> = None;
    for instr in instrs {
        match instr {
            Instr::OpenLoop { bound, .. } => {
                // The scan above already refused runtime bounds and zero iterations; here the
                // bound only names THIS class's layer count.
                let it = match bound {
                    LoopBound::Const(it) => *it,
                    // Unreachable in practice — the loop scan above refused runtime bounds
                    // before the walk started — but the match must be exhaustive.
                    LoopBound::Runtime(_) => {
                        return Err(SuperDscError(
                            "reroll-superdsc: a runtime-bounded layer loop is unsupported (the \
                             decode layer count is a compile-time Const)"
                                .into(),
                        ));
                    }
                };
                bodies.push(crate::lower_subtile_tape_to_superdsc::RolledBody {
                    ops: Vec::new(),
                    iters: it,
                    layer_ids: Vec::new(),
                    hidden_in_tid: u32::MAX,
                    hidden_out_tid: u32::MAX,
                });
                // In lockstep with `bodies`: `body_representatives[bi]` is body bi's own list,
                // indexed by the same `bi = seg - 1` the Compute arm below uses. `with_capacity`
                // alone leaves nothing to index — the first registration would panic.
                body_representatives.push(Vec::new());
                cur_first = None;
                cur_last = None;
                seg = bodies.len(); // 1..=n: the group index of THIS body
            }
            Instr::CloseLoop { .. } => {
                // Fold the body's first/last node into its residual-stream hidden tids (the
                // aliasing pre-pass already proved a body holds at least one Compute).
                let bi = seg - 1;
                bodies[bi].hidden_in_tid = cur_first
                    .and_then(|nid| ir.nodes[nid as usize].inputs.first())
                    .map(|r| r.tensor.index() as u32)
                    .unwrap_or(u32::MAX);
                bodies[bi].hidden_out_tid = cur_last
                    .map(|nid| ir.nodes[nid as usize].output.tensor.index() as u32)
                    .unwrap_or(u32::MAX);
                seg = n_groups - 1;
            }
            Instr::Compute {
                node,
                inputs,
                per_layer_out,
                ..
            } => {
                let n = &ir.nodes[node.index()];
                if seg >= 1 && seg <= loops.len() {
                    let bi = seg - 1;
                    let iters = bodies[bi].iters;
                    if cur_first.is_none() {
                        cur_first = Some(node.index() as u32);
                    }
                    cur_last = Some(node.index() as u32);
                    // Per-layer OUTPUT tids (the node's output tensor in each layer copy of THIS
                    // class). A table's KEY is the class's layer-0 tid, so only the body that
                    // reads (or writes) it can own it — `body_representatives` records that.
                    if iters > 1 && per_layer_out.len() as u32 == iters {
                        let outs: Vec<u32> = per_layer_out
                            .iter()
                            .map(|nid| ir.nodes[nid.index()].output.tensor.index() as u32)
                            .collect();
                        body_representatives[bi].push(n.output.tensor.index() as u32);
                        per_layer
                            .entry(n.output.tensor.index() as u32)
                            .or_insert(outs);
                    }
                    // Per-layer WEIGHT/external tids of THIS class.
                    for ci in inputs.iter() {
                        if let ComputeInput::External {
                            tensor,
                            per_layer: pl,
                            ..
                        } = ci
                            && iters > 1
                            && pl.len() as u32 == iters
                        {
                            body_representatives[bi].push(tensor.index() as u32);
                            per_layer
                                .entry(tensor.index() as u32)
                                .or_insert_with(|| pl.iter().map(|t| t.index() as u32).collect());
                        }
                    }
                    // ⭐ THE CLASS'S LAYER IDS, from its OWN rope table: the body's RopeAppend node
                    // (the KV writer — "a layer is named by its KV writer") carries `per_layer_out`
                    // naming this class's rope nodes; `layer_index()` on each gives the ABSOLUTE
                    // layer id per class-relative iteration. This is the only source of the true
                    // order, and it is the tape's own. FIRST matching table wins.
                    if bodies[bi].layer_ids.is_empty()
                        && matches!(n.op, SubOp::RopeAppend { .. })
                        && per_layer_out.len() as u32 == iters
                    {
                        bodies[bi].layer_ids = per_layer_out
                            .iter()
                            .map(|nid| {
                                ir.nodes[nid.index()]
                                    .op
                                    .layer_index()
                                    .expect("a RopeAppend names a layer")
                            })
                            .collect();
                    }
                } else if seg == n_groups - 1 && first_suffix_node.is_none() {
                    first_suffix_node = Some(node.index() as u32);
                }
                // ⭐ THE WEIGHT BANK THIS OP'S GROUP NEEDS. Recorded per group as the ops are emitted
                // because it is a property of the OPERANDS, which only the node knows; a launch binds
                // ONE base per segment, so the set for each group must end up with at most one
                // member. Empty for every op of every unbanked bundle.
                for r in &n.inputs {
                    let tid = r.tensor.index() as u32;
                    if let Some(p) = bundle_layout.placements.get(&tid)
                        && matches!(p.role, SegRole::Weight)
                    {
                        group_weight_banks[seg].insert(p.bank);
                    }
                }
                match lower_one_node(n, ir, active_cap, rows_are_requests, query_rows) {
                    NodeLowering::Ops(v) => {
                        if seg == 0 {
                            prefix.extend(v);
                        } else if seg <= loops.len() {
                            bodies[seg - 1].ops.extend(v);
                        } else {
                            suffix.extend(v);
                        }
                    }
                    NodeLowering::Unhandled(s) => {
                        if matches!(n.op, SubOp::MatmulTile { .. }) {
                            return Err(SuperDscError(s));
                        }
                        unhandled.insert(s);
                    }
                    NodeLowering::HostRouted(_) => {}
                }
            }
            Instr::AllocSlot { .. } | Instr::FreeSlot { .. } => {}
        }
    }
    if !unhandled.is_empty() {
        return Err(SuperDscError(format!(
            "reroll-superdsc: {} unhandled op kind(s): [{}]",
            unhandled.len(),
            unhandled.into_iter().collect::<Vec<_>>().join(", "),
        )));
    }
    for (bi, b) in bodies.iter().enumerate() {
        if b.layer_ids.len() != b.iters as usize {
            return Err(SuperDscError(format!(
                "reroll-superdsc: body {bi} has {} RopeAppend layer ids but its loop runs {} \
                 iterations — the layer cannot be named for every launch, so the true layer order \
                 is unknown and the launch table cannot be built",
                b.layer_ids.len(),
                b.iters
            )));
        }
    }
    // ⭐ THE LAUNCH SEQUENCE, in TRUE LAYER ORDER. Each body names its layers (absolute ids); the
    // model's layers run in ascending id order, so one sort of `(body, iter, layer_id)` triples is
    // the whole sequence. A single-class body has `layer_ids == [0..iters)`, which yields exactly
    // the `for v in 0..iters` sequence the executor ran before classes existed.
    let mut order: Vec<(u8, u32, u32)> = bodies
        .iter()
        .enumerate()
        .flat_map(|(bi, b)| {
            b.layer_ids
                .iter()
                .enumerate()
                .map(move |(it, &id)| (bi as u8, it as u32, id))
        })
        .collect();
    let total_layers = order.len();
    order.sort_by_key(|&(_, _, id)| id);
    if let Some(w) = order.windows(2).find(|w| w[0].2 == w[1].2) {
        return Err(SuperDscError(format!(
            "reroll-superdsc: two launches name layer {} (bodies {} and {}) — a layer id appears \
             in two classes' rope tables, so neither owns it",
            w[0].2, w[0].0, w[1].0
        )));
    }
    let launches: Vec<(u8, u32)> = order.iter().map(|&(b, i, _)| (b, i)).collect();
    let launch_layer: Vec<u32> = order.iter().map(|&(_, _, l)| l).collect();
    // ⭐ THE RUNG'S PLAN IS COMPLETED HERE, BEFORE ANY BUNDLE CAN SNAPSHOT IT: every program of
    // the rung (prefix, every body, suffix) is planned against the one layout, which then places
    // what the plans need — each staged weight's held layout, the constant sticks, and the scratch
    // every program's intermediates are packed into. The runtime loads ONE bundle's layout for a
    // session and launches every sibling against it, so a sibling's needs placed only in its own
    // snapshot would be addressed past the session's allocation.
    if cfg!(feature = "spyre-lower") {
        let all = prefix
            .iter()
            .chain(bodies.iter().flat_map(|b| b.ops.iter()))
            .chain(suffix.iter());
        crate::superdsc_lower::place_rung(all, &mut bundle_layout).map_err(SuperDscError)?;
        // A body's per-layer weights are its layer-0 weight's copies: each is staged in the same
        // held layout, by the same walk.
        let copies: Vec<(u32, RetileDescriptor)> = per_layer
            .iter()
            .filter_map(|(t, layers)| Some((bundle_layout.kernel_weights.get(t)?.clone(), layers)))
            .flat_map(|(k, layers)| layers.iter().map(move |&l| (l, k.clone())))
            .collect();
        bundle_layout.kernel_weights.extend(copies);
    }
    // ─────────────────────────────────────────────────────────────────────────────────────────
    //  ⭐ THE LAUNCH TABLE — one `(w_bank, w_off, kv_off)` per layer, GENERATED from the
    //  placements and PROVEN against them, so the executor never re-derives an address from a
    //  stride.
    //
    //  The old contract was arithmetic: layer `v` at `(v / layers_per_bank, (v % lpb)·weight_stride)`
    //  and `v·kv_stride`. That formula is only correct while every layer has the SAME geometry —
    //  gemma-4's two attention classes do not (sliding 8×256 KV vs global 1×512, and weight blocks
    //  that differ by class), so the formula would become three formulae (one per class, plus a
    //  class base) that could disagree with the packing that actually happened. The table replaces
    //  all of them: for the launch of (body c, iteration j), the shift is read straight off class
    //  c's own placements, and the guard below holds EVERY per-layer tensor of EVERY class to the
    //  table — strictly stronger than the pairwise stride check it replaces, because it also
    //  validates the bank and the class base, per layer, not per consecutive pair.
    // ─────────────────────────────────────────────────────────────────────────────────────────
    let w_seg = SegRole::Weight.segment();
    let kv_seg = SegRole::Kv.segment();
    // Which body owns a `per_layer` table: its key (the class's layer-0 tid) was collected from
    // that body's own instrs — see `body_representatives`. Only that body's launches shift for it.
    let body_of_tid = |tid: u32| -> Option<usize> {
        body_representatives
            .iter()
            .position(|reps| reps.contains(&tid))
    };
    // ⛔⛔⛔ A PER-LAYER **WEIGHT** OUTSIDE THE STRIDED SEGMENT IS SILENT GARBAGE: a launch shifts
    // only seg{w_seg}'s base, so layer v would read layer 0's copy of this tensor for every layer —
    // fluent output, wrong model. [`spill_weight_tail`] may only move the NON-per-layer tail, and
    // this is what holds it to that.
    //
    // ⛔ KEYED ON THE **ROLE**, NOT THE SEGMENT. A per-layer INTERMEDIATE colored into a spill slot
    // is legitimate and must keep falling through — the `WeightOverflow = 5` attempt refused
    // exactly that case (per-layer intermediate t448 in seg5) and read it as proof no segment was
    // available, when the real defect was taking a COLOR.
    //
    // A per-layer tensor must also not change SEGMENT across its class's layers: the launch shift
    // is one segment-wide byte offset, so a tensor that moved would be read at a wrong address on
    // one side of the move.
    for (t0, tids) in &per_layer {
        if tids.len() < 2 {
            continue;
        }
        let Some(p0) = bundle_layout.placements.get(t0) else {
            continue;
        };
        if p0.segment != w_seg && p0.segment != kv_seg && matches!(p0.role, SegRole::Weight) {
            return Err(SuperDscError(format!(
                "reroll-superdsc: per-layer WEIGHT t{t0} is placed in seg{}, which no launch \
                 shifts — every one of the {} layers would read layer 0's copy. Only NON-per-layer \
                 weights (the final norm, the lm_head / tied embedding) may spill; see \
                 `spill_weight_tail`.",
                p0.segment,
                tids.len(),
            )));
        }
        // Segment stability is required ONLY of the tensors a launch SHIFTS — the weight and KV
        // classes. A per-layer INTERMEDIATE may be colored differently per layer (a seg5 spill
        // slot for one, seg3 for another): intermediates are re-bound per launch from the launch's
        // own placement, so no shift ever crosses the move. Refusing those is what broke the first
        // attempt at this guard — the old pairwise check `continue`d them, and so does this one.
        let shifted =
            (p0.segment == w_seg && matches!(p0.role, SegRole::Weight)) || p0.segment == kv_seg;
        if shifted {
            for t in tids {
                if let Some(p) = bundle_layout.placements.get(t)
                    && p.segment != p0.segment
                {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: per-layer tensor t{t0} changes segment ({} → {}) across \
                         its class's layers — the launch shift is one segment-wide offset, so one \
                         side of the move would be read at a wrong address",
                        p0.segment, p.segment
                    )));
                }
            }
        }
    }
    // ── GENERATE: for each launch (body c, iter j), the weight shift from ANY of c's per-layer
    //    weight tables. A body with no per-layer weight gets (0, 0) — nothing to shift. The proof
    //    below then holds EVERY weight of the class to the entry, so which table generated it
    //    cannot matter; a class whose tables DISAGREE is refused here with both readings named.
    //
    //    The KV column is the layout's own: each launch block's pools are their own region
    //    (`place_kv_pools`), so the launch names that block's bank and shifts nothing. A body's
    //    banks are read through the tables it OWNS (`body_of_tid`, the weight column's own key),
    //    never by its position.
    use crate::lower_subtile_tape_to_superdsc::KvBank;
    /// What one body's launches hand the KV slot.
    enum BodyKv<'a> {
        /// The banks of the per-layer pool tables this body owns, one per iteration.
        Owns(&'a [KvBank]),
        /// The body owns no per-layer pool: the only pools it can read sit in bank 0.
        OwnsNone,
    }
    let body_kv: Vec<BodyKv<'_>> = (0..bodies.len())
        .map(|b| {
            let mut owned = bundle_layout
                .kv_block_banks
                .iter()
                .filter(|(t0, _)| body_of_tid(**t0) == Some(b));
            let Some((t0, first)) = owned.next() else {
                return Ok(BodyKv::OwnsNone);
            };
            match owned.find(|(_, banks)| *banks != first) {
                Some((t1, _)) => Err(SuperDscError(format!(
                    "reroll-superdsc: body {b} owns pool tables t{t0} and t{t1}, which the layout \
                     placed in different layers' regions — one launch hands the KV slot one \
                     region, so no launch addresses both"
                ))),
                None => Ok(BodyKv::Owns(first)),
            }
        })
        .collect::<Result<_, SuperDscError>>()?;
    let launch_kv: Vec<KvBank> = launches
        .iter()
        .map(|&(body, iter)| match body_kv[body as usize] {
            BodyKv::Owns(banks) => banks.get(iter as usize).copied().ok_or_else(|| {
                SuperDscError(format!(
                    "reroll-superdsc: body {body} launches layer {iter}, but its pool tables \
                     hold {} layer(s)",
                    banks.len()
                ))
            }),
            BodyKv::OwnsNone => Ok(KvBank::BASE),
        })
        .collect::<Result<_, SuperDscError>>()?;
    let mut launch_w: Vec<(u8, u64)> = Vec::with_capacity(total_layers);
    for &(body, iter) in &launches {
        let mut w_entry: Option<(u8, u64)> = None;
        for (t0, tids) in &per_layer {
            if body_of_tid(*t0) != Some(body as usize) {
                continue;
            }
            let (Some(p0), Some(pj)) = (
                bundle_layout.placements.get(t0),
                bundle_layout.placements.get(&tids[iter as usize]),
            ) else {
                continue;
            };
            if p0.segment == w_seg && matches!(p0.role, SegRole::Weight) {
                let entry = (pj.bank as u8, pj.offset.wrapping_sub(p0.offset));
                if let Some(prev) = w_entry
                    && prev != entry
                {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: body {body}'s per-layer weights DISAGREE about layer \
                         {iter}'s launch shift ({prev:?} vs {entry:?}, generated from t{t0}) — the \
                         packing did not advance the class's weights together, so no one shift \
                         addresses them all"
                    )));
                }
                w_entry = Some(entry);
            }
        }
        launch_w.push(w_entry.unwrap_or((0, 0)));
    }
    // ── PROVE, EVERY TENSOR, EVERY LAYER. The launch addresses
    //    `(bank: L.w_bank, offset: L.w_off + placement(t₀).offset)` — hold each per-layer weight
    //    to the placement its layer actually got. This is the check the generator cannot make:
    //    generation read ONE weight table per launch; this holds ALL of them.
    for (li, &(body, iter)) in launches.iter().enumerate() {
        let (wb, wo) = launch_w[li];
        for (t0, tids) in &per_layer {
            if body_of_tid(*t0) != Some(body as usize) {
                continue;
            }
            let (Some(p0), Some(pj)) = (
                bundle_layout.placements.get(t0),
                bundle_layout.placements.get(&tids[iter as usize]),
            ) else {
                continue;
            };
            if p0.segment == w_seg
                && matches!(p0.role, SegRole::Weight)
                && (pj.bank as u8 != wb || pj.offset != wo.wrapping_add(p0.offset))
            {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: per-layer weight t{tid} (layer {layer} of body {body}, \
                         class-relative {iter}) is placed at bank {bank} offset {off}, but the \
                         launch table addresses it at bank {wb} offset {want} (shift {wo} + \
                         layer-0 offset {base}). Every layer must sit where the launch looks, or \
                         that layer reads another layer's weights.",
                    tid = tids[iter as usize],
                    layer = launch_layer[li],
                    bank = pj.bank,
                    off = pj.offset,
                    want = wo.wrapping_add(p0.offset),
                    wo = wo,
                    base = p0.offset,
                    body = body,
                    iter = iter,
                    wb = wb,
                )));
            }
        }
    }
    // ⛔ AND A BODY READS ONLY PER-LAYER TENSORS IN THE KV SEGMENT. Every layer's launch hands the
    // KV slot that layer's own region, so a tensor in the segment that is not one of a layer's own
    // copies — it sits in bank 0 — is read in another layer's region by every launch but bank 0's.
    // (The prefix and suffix launch on bank 0, so a tensor only they read is read where it sits.)
    if launches.len() > 1 {
        let per_layer_tids: std::collections::HashSet<u32> =
            per_layer.values().flatten().copied().collect();
        let body_reads: std::collections::HashSet<u32> = bodies
            .iter()
            .flat_map(|b| b.ops.iter())
            .flat_map(|e| e.ktir.bindings.iter().map(|b| b.get()))
            .collect();
        if let Some(p) = bundle_layout.placements.values().find(|p| {
            p.segment == kv_seg && !per_layer_tids.contains(&p.tid) && body_reads.contains(&p.tid)
        }) {
            return Err(SuperDscError(format!(
                "reroll-superdsc: a layer's body reads t{}, which is placed in the KV segment \
                 (seg{kv_seg}) but is no layer's own tensor — each of the {} layer launches hands \
                 that segment its own layer's region, so every layer in another bank would read it \
                 elsewhere",
                p.tid,
                launches.len(),
            )));
        }
    }
    // ⛔ AND ONE BANK PER LAUNCH GROUP. A launch is handed ONE base per segment, so a program whose
    // weight operands span two banks cannot be expressed AT ALL — there is no offset that reaches
    // both. `group_weight_banks` was accumulated over the ops as they were emitted, so this is the
    // set of banks each program actually addresses.
    let group_bank = |s: usize, what: &str| -> Result<u32, SuperDscError> {
        let banks = &group_weight_banks[s];
        match banks.len() {
            0 => Ok(0), // reads no weights at all: any bank will do, so bind bank 0
            1 => Ok(*banks.iter().next().expect("len 1")),
            _ => Err(SuperDscError(format!(
                "reroll-superdsc: the {what} program's weights span weight banks {banks:?}, and a \
                 launch has ONE base per segment — no offset reaches both. The non-per-layer weights \
                 (final norm, lm_head / tied embedding) are placed in ONE bank together for exactly \
                 this reason; a weight the {what} reads from another bank would have to be \
                 REPLICATED into every bank that reads it, which `bank_weight_segment` does not do."
            ))),
        }
    };
    let prefix_weight_bank = group_bank(0, "prefix")?;
    let suffix_weight_bank = group_bank(n_groups - 1, "suffix")?;
    // Each BODY'S weight reads must all sit in the bank its OWN launches bind — the launch-table
    // proof above already held its per-layer weights there, so a disagreement here means the body
    // also reads a NON-per-layer weight from another bank: the replication case, refused with its
    // own name rather than as a launch-table mismatch.
    for bi in 0..bodies.len() {
        let launch_bank = launches
            .iter()
            .zip(&launch_w)
            .find(|(l, _)| l.0 as usize == bi)
            .map(|(_, w)| w.0 as u32)
            .unwrap_or(0);
        if group_bank(1 + bi, "body")? != launch_bank {
            return Err(SuperDscError(format!(
                "reroll-superdsc: body {bi}'s program addresses weight bank(s) {:?}, but its \
                 launches bind bank {launch_bank} — a NON-per-layer weight read inside the layer \
                 loop would need replicating into every bank.",
                group_weight_banks[1 + bi],
            )));
        }
    }
    // The suffix's residual input (first suffix node's input[0]) — the executor threads the last
    // launch's hidden_out → suffix_in once after the launch sequence (the body→suffix seam). Same
    // input-ordering convention as a body's hidden_in (rmsnorm input[0] = x = the residual).
    let suffix_in_tid = first_suffix_node
        .and_then(|nid| ir.nodes[nid as usize].inputs.first())
        .map(|r| r.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    let kv_request_stride = bundle_layout.kv_request_stride_bytes;
    Ok(RolledSuperDsc {
        prefix,
        bodies,
        suffix,
        layout: bundle_layout,
        per_layer,
        launches,
        launch_w,
        launch_kv,
        prefix_weight_bank,
        suffix_weight_bank,
        kv_request_stride,
        suffix_in_tid,
    })
}

#[cfg(test)]
mod scalar_mul_weight_fold_tests {
    use super::*;
    use scratchy_subtile::lower::GemmWeight;
    use scratchy_subtile::subtile_ir::{Region, TensorId, TensorRegion, TensorShape};

    /// granite's residual shape and its refusals, as one fixture with switches: t0 = x
    /// (activation source), t1 = W (weight source), t2 = the matmul's whole output, t3 = the
    /// ScalarMul's output, t4 = W2 (the consumer's OWN weight), t5 = the consumer's output,
    /// t6 = spare. `tied_weight` adds a second reader of W (the tied-table refusal),
    /// `shared_product` adds a second reader of t2 (the unscaled-product refusal), `sliced`
    /// makes the ScalarMul read a column slice of t2, and `as_result` makes t3 the graph result.
    ///
    /// Every caller pins an UP-multiplier: the f16 floor gate refuses a down-scale outright, so
    /// an emptiness observed at `|s| >= 1` is the switch's own refusal, never the gate's.
    fn graph(
        scale: f32,
        tied_weight: bool,
        shared_product: bool,
        sliced: bool,
        as_result: bool,
    ) -> SubtileIR {
        let tensors = vec![
            TensorShape { rows: 1, cols: 8 },  // t0 = x
            TensorShape { rows: 8, cols: 16 }, // t1 = W
            TensorShape { rows: 1, cols: 16 }, // t2 = x·W
            TensorShape { rows: 1, cols: 16 }, // t3 = (x·W)·s
            TensorShape { rows: 16, cols: 4 }, // t4 = W2 (the consumer's own weight)
            TensorShape { rows: 1, cols: 4 },  // t5 = the consumer's output
            TensorShape { rows: 1, cols: 16 }, // t6 = spare
        ];
        let whole = |t: usize| TensorRegion {
            tensor: TensorId::from_index(t),
            region: tensors[t].whole(),
        };
        let mut nodes = vec![
            SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Dense,
                },
                inputs: vec![whole(0), whole(1)],
                output: whole(2),
            },
            SubtileNode {
                id: SubtileId::from_index(1),
                op: SubOp::ScalarMul { scale },
                inputs: vec![if sliced {
                    TensorRegion {
                        tensor: TensorId::from_index(2),
                        region: Region {
                            rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
                            cols: scratchy_subtile::subtile_ir::Range::new(0, 8),
                        },
                    }
                } else {
                    whole(2)
                }],
                output: whole(3),
            },
        ];
        // The consumer every fold needs: a reader of the SCALED product over its OWN weight, so
        // the base fixture reads W exactly once (the producer).
        nodes.push(SubtileNode {
            id: SubtileId::from_index(2),
            op: SubOp::MatmulTile {
                n: 4,
                weight: GemmWeight::Dense,
            },
            inputs: vec![whole(3), whole(4)],
            output: whole(5),
        });
        if tied_weight {
            // A second reader of W — the tied lm_head/embedding shape: one weight, two matmuls.
            nodes.push(SubtileNode {
                id: SubtileId::from_index(3),
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Dense,
                },
                inputs: vec![whole(0), whole(1)],
                output: whole(6),
            });
        }
        if shared_product {
            // A second reader of the UNSCALED product t2 beside the ScalarMul — reading a
            // SLICE, so this reader is not itself a fold candidate and the refusal under test
            // is the ORIGINAL node's alone.
            nodes.insert(
                1,
                SubtileNode {
                    id: SubtileId::from_index(4),
                    op: SubOp::ScalarMul { scale: 2.0 },
                    inputs: vec![TensorRegion {
                        tensor: TensorId::from_index(2),
                        region: Region {
                            rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
                            cols: scratchy_subtile::subtile_ir::Range::new(0, 8),
                        },
                    }],
                    output: whole(6),
                },
            );
        }
        SubtileIR {
            tensors,
            num_sources: 3,
            nodes,
            result: TensorId::from_index(if as_result { 3 } else { 5 }),
            // Hand-authored fixture: there is no source op list to be the provenance of.
            op_output: Vec::new(),
        }
    }

    /// The fold fires on granite's residual shape, naming the matmul's LAST input — at an
    /// UP-multiplier, the one class the f16 floor gate permits (`|s| >= 1` can only move a
    /// staged value away from the subnormal floor; the down-scale side is pinned in
    /// `lower_subtile_tape_to_superdsc`'s refusal test).
    #[test]
    fn the_residual_multiplier_folds_into_the_matmul_weight() {
        let ir = graph(2.0, false, false, false, false);
        let folds = scalar_mul_weight_folds(&ir);
        assert_eq!(
            folds,
            vec![(SubtileId::from_index(1), 1, 2.0f32)],
            "the ScalarMul folds into W (t1), the producer's last input"
        );
    }

    /// The tied lm_head/embedding table is granite's own refusal: a weight another node reads
    /// cannot absorb a per-consumer scale.
    #[test]
    fn a_weight_another_node_reads_refuses_the_fold() {
        let ir = graph(2.0, true, false, false, false);
        assert!(
            scalar_mul_weight_folds(&ir).is_empty(),
            "a shared weight must keep its ScalarMul as an op"
        );
    }

    /// A second reader of the matmul's output needs the UNSCALED product.
    #[test]
    fn a_product_another_node_reads_refuses_the_fold() {
        let ir = graph(2.0, false, true, false, false);
        assert!(
            scalar_mul_weight_folds(&ir).is_empty(),
            "a shared product must keep its ScalarMul as an op"
        );
    }

    /// A sliced read scales rows the fold would not.
    #[test]
    fn a_sliced_read_refuses_the_fold() {
        let ir = graph(2.0, false, false, true, false);
        assert!(
            scalar_mul_weight_folds(&ir).is_empty(),
            "a sliced read must keep its ScalarMul as an op"
        );
    }

    /// The logits ScalarMul is the splice's own lm-head-tail fold, and the result's Logits
    /// placement is the worker's readback window — not this fold's to redirect.
    #[test]
    fn the_graph_result_refuses_the_fold() {
        let ir = graph(2.0, false, false, false, true);
        assert!(
            scalar_mul_weight_folds(&ir).is_empty(),
            "the result ScalarMul belongs to the splice's lm-head-tail fold"
        );
    }

    /// The fp8 W8A8 shape folds into the `ws` dequant row — the scale rides the row that is
    /// already per-column, so the quantized payload is untouched (exact algebra, no requantize)
    /// — at an UP-multiplier, the f16 floor gate's permitted class.
    #[test]
    fn the_fp8_w8a8_multiplier_folds_into_the_dequant_row() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 8 },  // t0 = qA
            TensorShape { rows: 8, cols: 16 }, // t1 = qW (fp8 payload)
            TensorShape { rows: 1, cols: 16 }, // t2 = ws (the dequant row)
            TensorShape { rows: 1, cols: 16 }, // t3 = (qA·qW)·a·ws
            TensorShape { rows: 1, cols: 16 }, // t4 = scaled
            TensorShape { rows: 16, cols: 4 }, // t5 = the consumer's own dense weight
            TensorShape { rows: 1, cols: 4 },  // t6 = consumer output
        ];
        let whole = |t: usize| TensorRegion {
            tensor: TensorId::from_index(t),
            region: tensors[t].whole(),
        };
        let nodes = vec![
            SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Fp8Dynamic,
                },
                inputs: vec![whole(0), whole(1), whole(2)],
                output: whole(3),
            },
            SubtileNode {
                id: SubtileId::from_index(1),
                op: SubOp::ScalarMul { scale: 2.0 },
                inputs: vec![whole(3)],
                output: whole(4),
            },
            SubtileNode {
                id: SubtileId::from_index(2),
                op: SubOp::MatmulTile {
                    n: 4,
                    weight: GemmWeight::Dense,
                },
                inputs: vec![whole(4), whole(5)],
                output: whole(6),
            },
        ];
        let ir: SubtileIR = SubtileIR {
            tensors,
            num_sources: 3,
            nodes,
            result: TensorId::from_index(6),
            op_output: Vec::new(),
        };
        assert_eq!(
            scalar_mul_weight_folds(&ir),
            vec![(SubtileId::from_index(1), 2, 2.0f32)],
            "the fp8 fold targets the ws dequant row (t2), never the quantized payload (t1)"
        );
    }
}

#[cfg(test)]
mod attn_scale_weight_fold_tests {
    use super::*;
    use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};

    /// granite's decode attention chain and its refusals, as one fixture with switches, built
    /// through `lower_region` — the ONLY construction that can mint the Tiled stage's
    /// `KvCacheLayout`/`KvCacheProducer`/`SoftmaxStateId` witnesses, and the SAME one the macro's
    /// KTIR path builds the real graph with (`fuse_silu_mul` + `lower_region`; the head-tiling
    /// rewrites are other targets' passes and never run here, so the ropes and the attention are
    /// WHOLE ops — the shape the recognizer's whole-region matches are pinned to).
    ///
    /// The chain under test is `matmul → rope → attention` on both live sides, exactly as
    /// `to_wavefront` lowers `rope_append(q, k, v, …)`: `x·W_q → rope_rotate`, `x·W_k →
    /// rope_append` (writing the paged cache), `AttnDecode(q_rot, prefix_k, prefix_v, k_rot, v)`
    /// with v aliased through un-roped. Geometry (2 query heads, 1 kv head, head_dim 8) fixes
    /// q_width 16 and kv_width 8, so W_q is `[8,16]` and W_k `[8,8]`.
    ///
    /// `q_unroped` / `k_unroped` point the attention at the RAW projection product on that side
    /// (a non-rope producer); `shared_w_q` adds a second matmul over W_q (the fused-QKV table
    /// shape); `shared_q_product` adds a second consumer of the q projection's product; and
    /// `shared_roped_q` adds a second consumer of the roped q. The extra consumers read over
    /// their OWN weight (a third source, `W3`), so they are never fold candidates themselves and
    /// each refusal under test is the fold's alone.
    ///
    /// Every caller pins a scale whose `√scale` clears the f16 floor gate (`|s| >= 1` — an
    /// UP-multiplier on the projection weights), so an emptiness observed here is the switch's
    /// own refusal, never the gate's; granite's real 1/64 is pinned on the refusal side in
    /// `lower_subtile_tape_to_superdsc` and the models crate.
    fn attn_graph(
        scale: f32,
        q_unroped: bool,
        k_unroped: bool,
        shared_w_q: bool,
        shared_q_product: bool,
        shared_roped_q: bool,
    ) -> SubtileIR {
        // granite's own geometry as the facts' `Attention` (uniform classes, 2 query heads
        // over 1 kv head at head_dim 8), scaled down: the fold reads only the facts' scale,
        // so the geometry is inert — it just has to exist.
        let attention = scratchy_subtile::fixtures::attention_at(2, 1, 8, scale);
        let hd = scratchy_ir::model_facts::HeadDim::new(8);
        let shape = |rows: u32, cols: u32| scratchy_subtile::subtile_ir::SourceShape { rows, cols };
        let sources = vec![
            shape(1, 8),  // 0: x (hidden 8)
            shape(8, 16), // 1: W_q [K=q_width rows? no — row-major [K, N] = [8, 16]]
            shape(8, 8),  // 2: W_k [8, kv_width]
            shape(1, 8),  // 3: v (the V projection's output, un-roped)
            shape(32, 8), // 4: prefix_k cache
            shape(32, 8), // 5: prefix_v cache
            shape(1, 16), // 6: cos
            shape(1, 16), // 7: sin
            shape(16, 4), // 8: W3 (the refusal readers' own weight)
        ];
        let q_ref = if q_unroped {
            InputRef::Op(0)
        } else {
            InputRef::Op(1)
        };
        let k_ref = if k_unroped {
            InputRef::Op(2)
        } else {
            InputRef::Op(3)
        };
        let mut ops = vec![
            // 0: q = x · W_q
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            // 1: q_rot = rope_rotate(q, cos, sin)
            OpDesc {
                op: SubOp::rope_rotate(hd),
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(6), InputRef::Ext(7)],
            },
            // 2: k = x · W_k
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 8,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(2)],
            },
            // 3: k_rot = rope_append(k, cos, sin, v, prefix_k, prefix_v) — writing the paged cache
            OpDesc {
                op: SubOp::rope_append(
                    attention,
                    scratchy_subtile::subtile_ir::RopeClass::Base,
                    0,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                    scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                ),
                m: 1,
                inputs: vec![
                    InputRef::Op(2),
                    InputRef::Ext(6),
                    InputRef::Ext(7),
                    InputRef::Ext(3),
                    InputRef::Ext(4),
                    InputRef::Ext(5),
                ],
            },
            // 4: attn = AttnDecode(q_rot, prefix_k, prefix_v, k_rot, v)
            OpDesc {
                op: SubOp::attn_decode(
                    attention,
                    33,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                ),
                m: 1,
                inputs: vec![
                    q_ref,
                    InputRef::Ext(4),
                    InputRef::Ext(5),
                    k_ref,
                    InputRef::Ext(3),
                ],
            },
        ];
        if shared_w_q {
            // A second matmul over W_q — the fused-QKV table: one projection weight, two matmuls.
            ops.push(OpDesc {
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            });
        }
        if shared_q_product {
            // A second consumer of the UNSCALED q product — over its own weight, so it is not
            // itself a fold candidate.
            ops.push(OpDesc {
                op: SubOp::MatmulTile {
                    n: 4,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(8)],
            });
        }
        if shared_roped_q {
            // A second consumer of the ROPED q — what the fold would hand pre-scaled bytes.
            ops.push(OpDesc {
                op: SubOp::MatmulTile {
                    n: 4,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(1), InputRef::Ext(8)],
            });
        }
        let input = LoweringInput {
            sources,
            ops,
            result: 4,
            exported: None,
        };
        let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
        scratchy_subtile::subtile_ir::lower_region(&input, nb)
    }

    /// The fold fires on granite's chain, naming BOTH projection weights at `√scale` — the query's
    /// and the key's, one entry each, so the runtime list holds two multiplies at load and the
    /// node's own scale is rewritten to 1.0 where it is lowered. The scale is an UP-multiplier
    /// (√4.0 = 2.0): the one class the f16 floor gate permits.
    #[test]
    fn the_attention_scale_folds_into_both_projection_weights() {
        let ir = attn_graph(4.0, false, false, false, false, false);
        assert_eq!(
            attn_scale_weight_folds(&ir),
            vec![(SubtileId::from_index(4), 1, 2, 2.0f32)],
            "the fold names W_q (source 1) and W_k (source 2), each at √scale"
        );
        assert_eq!(
            weight_scale_folds(&ir),
            vec![
                (SubtileId::from_index(4), 1, 2.0f32),
                (SubtileId::from_index(4), 2, 2.0f32),
            ],
            "the union list the runtime applies carries one (tid, multiplier) pair per side"
        );
    }

    /// The fp8 W8A8 shape folds into the `ws` dequant rows — the SAME slot law the ScalarMul fold
    /// uses, which is the whole point: the scale rides the row that is already per-column, so the
    /// quantized payloads are untouched (exact algebra, no requantize). granite's fp8 preset
    /// quantizes every projection, so both sides take it — at an UP-multiplier (√4.0 = 2.0), the
    /// f16 floor gate's permitted class.
    #[test]
    fn the_fp8_attention_folds_into_the_dequant_rows() {
        let shape = |rows: u32, cols: u32| scratchy_subtile::subtile_ir::SourceShape { rows, cols };
        let attention = scratchy_subtile::fixtures::attention_at(2, 1, 8, 4.0);
        let hd = scratchy_ir::model_facts::HeadDim::new(8);
        let input = LoweringInput {
            sources: vec![
                shape(1, 8),  // 0: x
                shape(8, 16), // 1: qW_q — the fp8 PAYLOAD
                shape(16, 1), // 2: ws_q — the dequant row
                shape(8, 8),  // 3: qW_k — the fp8 PAYLOAD
                shape(8, 1),  // 4: ws_k — the dequant row
                shape(1, 8),  // 5: v
                shape(32, 8), // 6: prefix_k cache
                shape(32, 8), // 7: prefix_v cache
                shape(1, 16), // 8: cos
                shape(1, 16), // 9: sin
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::MatmulTile {
                        n: 16,
                        weight: GemmWeight::Fp8Dynamic,
                    },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1), InputRef::Ext(2)],
                },
                OpDesc {
                    op: SubOp::rope_rotate(hd),
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(8), InputRef::Ext(9)],
                },
                OpDesc {
                    op: SubOp::MatmulTile {
                        n: 8,
                        weight: GemmWeight::Fp8Dynamic,
                    },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(3), InputRef::Ext(4)],
                },
                OpDesc {
                    op: SubOp::rope_append(
                        attention,
                        scratchy_subtile::subtile_ir::RopeClass::Base,
                        0,
                        scratchy_subtile::subtile_ir::AttnMask::Causal,
                        scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                    ),
                    m: 1,
                    inputs: vec![
                        InputRef::Op(2),
                        InputRef::Ext(8),
                        InputRef::Ext(9),
                        InputRef::Ext(5),
                        InputRef::Ext(6),
                        InputRef::Ext(7),
                    ],
                },
                OpDesc {
                    op: SubOp::attn_decode(
                        attention,
                        33,
                        scratchy_subtile::subtile_ir::AttnMask::Causal,
                    ),
                    m: 1,
                    inputs: vec![
                        InputRef::Op(1),
                        InputRef::Ext(6),
                        InputRef::Ext(7),
                        InputRef::Op(3),
                        InputRef::Ext(5),
                    ],
                },
            ],
            result: 4,
            exported: None,
        };
        let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
        let ir = scratchy_subtile::subtile_ir::lower_region(&input, nb);
        assert_eq!(
            attn_scale_weight_folds(&ir),
            vec![(SubtileId::from_index(4), 2, 4, 2.0f32)],
            "the fp8 fold targets the ws dequant rows (sources 2 and 4), never the quantized \
             payloads (1 and 3)"
        );
    }

    /// A query that skipped its rope has a MATMUL for a producer — the walk is granite's
    /// `matmul → rope → attention` chain, stated as a shape, and a non-rope producer is not it.
    #[test]
    fn an_unroped_query_refuses_the_fold() {
        let ir = attn_graph(4.0, true, false, false, false, false);
        assert!(
            attn_scale_weight_folds(&ir).is_empty(),
            "a non-rope producer of q must keep its multiplies as ops"
        );
    }

    /// The mirror refusal on the key side: the fold needs BOTH chains clean, never a half-fold —
    /// scaling only W_q would leave the scores at `q·kᵀ·(s/√s)`.
    #[test]
    fn an_unroped_key_refuses_the_whole_fold() {
        let ir = attn_graph(4.0, false, true, false, false, false);
        assert!(
            attn_scale_weight_folds(&ir).is_empty(),
            "a non-rope producer of new_k must refuse the whole fold, not fold W_q alone"
        );
    }

    /// A shared projection table is the fused-QKV shape: one weight, two matmuls, and a
    /// per-consumer scale cannot ride it.
    #[test]
    fn a_shared_projection_table_refuses_the_fold() {
        let ir = attn_graph(4.0, false, false, true, false, false);
        assert!(
            attn_scale_weight_folds(&ir).is_empty(),
            "a weight another matmul reads must keep its multiplies as ops"
        );
    }

    /// A second reader of the projection's product needs the UNSCALED product.
    #[test]
    fn a_second_reader_of_the_projection_product_refuses_the_fold() {
        let ir = attn_graph(4.0, false, false, false, true, false);
        assert!(
            attn_scale_weight_folds(&ir).is_empty(),
            "a shared projection product must keep its multiplies as ops"
        );
    }

    /// A second reader of the roped query needs the UNSCALED rotation — the fold changes that
    /// tensor's value, which is its whole point.
    #[test]
    fn a_second_reader_of_the_roped_query_refuses_the_fold() {
        let ir = attn_graph(4.0, false, false, false, false, true);
        assert!(
            attn_scale_weight_folds(&ir).is_empty(),
            "a shared roped query must keep its multiplies as ops"
        );
    }

    /// `scale == 1.0` has nothing to fold — and it is also the value a folded node carries
    /// afterward, which is what makes the recognition idempotent.
    #[test]
    fn a_scale_of_one_folds_nothing() {
        let ir = attn_graph(1.0, false, false, false, false, false);
        assert!(
            attn_scale_weight_folds(&ir).is_empty(),
            "a scale of 1.0 multiplies nothing and must fold nothing"
        );
    }
}
