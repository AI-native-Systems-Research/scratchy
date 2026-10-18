// SPDX-License-Identifier: Apache-2.0
//! SubtileIR → **KTIR**: the PRODUCER half of `SubtileIR → KTIR → SuperDSC`.
//!
//! ⭐⭐⭐ THE PROGRAM IS CONSTRUCTED, NEVER PRINTED. `ktir-core`'s `Operation` / `IRFunction` ARE the
//! interchange: the emulator executes the value directly and `#[forward]` bakes it as const data.
//! Nothing here renders MLIR and nothing anywhere parses it.
//!
//! ⭐ ONE NODE, ONE PROGRAM. Each [`SubtileNode`] becomes one KTIR `func` — inputs loaded from HBM,
//! output stored back — carried on an [`EmittedOp`] as `ktir`. Both consumers read that SAME value:
//! `-Fspyre-emu` interprets it, `-Fspyre-hw` lowers it through
//! [`crate::ktir_superdsc_door`].
//!
//! ⛔ NOTHING HERE EMITS A SuperDSC DESCRIPTOR. That is the consumer's half, in its own file, and a
//! node lowered straight to SuperDSC from here would be a second path to the same format.
//!
//! [`EmittedOp`]: crate::lower_subtile_tape_to_superdsc::EmittedOp

use crate::lower_subtile_tape_to_superdsc::*;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::argmax::{TiedLmHeadEmbedding, mint_argmax_node};
use ktir_superdsc::ktir_node::ActiveCap;
use ktir_superdsc::sdsc_abstract::POOL_STICK;
use scratchy_subtile::lower::GemmWeight;
use scratchy_subtile::subtile_ir::{RopeForm, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId};
use scratchy_subtile::superdsc_opspec::{DataFormat, DeviceTileLayout};

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
pub(crate) enum NodeLowering {
    Ops(Vec<EmittedOp>),
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
        let SubOp::AttnDecode { scale, .. } = &node.op else {
            continue;
        };
        // inputs = [q, prefix_k, prefix_v, new_k, new_v] — the walk is the two live sides.
        let [q, _prefix_k, _prefix_v, new_k, _new_v] = node.inputs.as_slice() else {
            continue;
        };
        if *scale == 1.0 {
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

/// ⭐⭐⭐ THE o_proj EDGES — every (attention output, o matmul weight, precision) triple the graph
/// states, which is the ONE dataflow fact the head-major o_proj restructure keys on.
///
/// The restructure has two doors that must agree on WHICH matmul is the o projection — the
/// attention finalize's slab-major form ([`crate::ktir_superdsc_door::BundleAttnParams`]) and the
/// o matmul's B-operand swap (`HeadmajorOproj`) — and a SHAPE cannot say: at `k == n`
/// (granite-3.1-8b's `[4096, 4096]` o_proj) the extents cannot tell the o matmul from any other
/// square projection, and a shape-keyed swap would re-point every square projection at the
/// permuted copy. The DATAFLOW can: the o matmul is the one whose `inputs[0]` is an attention's
/// whole output.
///
/// ⛔ EVERY REFUSAL IS A CORRECTNESS FACT, and the class is the fold walks' own:
///   * an attention output consumed by anything but ONE whole-reading matmul is a graph the
///     restructure cannot describe — the slab-major finalize hands EVERY consumer head-permuted
///     bytes, so a second consumer or a non-matmul reader is refused at recognition rather than
///     silently excluded while the finalize still fires for it;
///   * where the head dim is multi-slab (`hd > POOL_STICK`), every o matmul must agree on
///     fp8-ness — the handoff is ONE fact per bundle, so a mixed model would emit a slab-major
///     finalize for a dense layer whose weight was never permuted: fluent, wrong output;
///   * and every o matmul must agree on GEOMETRY (`hd`, `k`) — the permuted walk, the swap key,
///     and the per-layer family are all minted from the FIRST edge, so a hybrid-attention arch's
///     second class would keep its un-permuted weight (the swap key names only the canonical)
///     under a bundle-wide slab-major finalize: the same half a restructure, across classes
///     instead of precisions.
pub(crate) struct OprojEdge {
    /// The o matmul's weight tid — `inputs[1]`, the slot law every weight recognizer in this file
    /// already uses (dense `[A, W]` and fp8 W8A8 `[A, qW, ws]` agree on it).
    pub w: u32,
    /// The head dim of this edge's attention — the restructure's own geometry input.
    pub hd: u32,
    /// The o matmul's contraction width — `inputs[0]`'s cols, the attention output's own
    /// `nqh·hd` (the clean-shape check holds `inputs[0]` to the attention's whole output). With
    /// `hd`, the pair the single-class permuted walk is derived from: two classes may share an
    /// `hd` and differ in `nqh`, so `hd` alone cannot name a geometry.
    pub k: u32,
    /// The o matmul is fp8 W8A8 (arity-3). The permuted staging walk is derived for the fp8
    /// packed tile; a dense o_proj keeps the shipped emission byte-for-byte.
    pub fp8: bool,
}

/// The edges plus the ONE gate both restructure doors key on. See [`OprojEdge`]'s doc for the
/// walk; this carries what each consumer derives from it so none re-derives a second copy that
/// could disagree.
pub(crate) struct OprojEdges {
    /// Every edge, in node order. The FIRST is the CANONICAL one — the rolled body's
    /// representative, whose weight tid the bundle facts name and the matmul door swaps on.
    pub edges: Vec<OprojEdge>,
    /// ⭐ THE RESTRUCTURE'S GEOMETRY GATE, BUNDLE-INVARIANT BY CONSTRUCTION: every edge's
    /// attention is multi-slab (`hd > POOL_STICK` — 8b's 128; at one slab there is nothing to
    /// restructure), every o matmul is fp8 W8A8, and every edge carries ONE geometry (`hd`, `k`)
    /// — the walk, swap key, and family are minted from the first edge alone, so a
    /// hybrid-attention arch's second class would keep an un-permuted weight under a
    /// bundle-wide slab-major finalize. This is what the LAYOUT MINT keys on —
    /// never the bundle's row count, because every bundle of a model shares ONE staged weight
    /// segment: a rows-gated mint would hand the single-row body addresses no other bundle
    /// staged. Only the EMISSION gates on rows (the bundle's own `headmajor_handoff`).
    pub geometry_applies: bool,
}

pub(crate) fn oproj_edges<F: RopeForm>(ir: &SubtileIR<F>) -> Result<OprojEdges, SuperDscError> {
    let mut edges: Vec<OprojEdge> = Vec::new();
    for node in &ir.nodes {
        let SubOp::AttnDecode { geom, .. } = &node.op else {
            continue;
        };
        // The attention's consumers — every node reading any region of its output tensor. The
        // clean shape is EXACTLY ONE MatmulTile reading `inputs[0]` as the whole output, which is
        // the chain every shipped attention arch states (`attention → o_proj → residual`).
        let consumers: Vec<&SubtileNode<F>> = ir
            .nodes
            .iter()
            .filter(|c| c.id != node.id && c.inputs.iter().any(|r| r.tensor == node.output.tensor))
            .collect();
        if consumers.is_empty() {
            // An attention whose output nobody reads states no edge — nothing to swap and no
            // o_proj to companion. The graph result is not a consumer (nothing reads it), so an
            // attention that IS the result lands here too.
            continue;
        }
        let clean = consumers.len() == 1
            && matches!(consumers[0].op, SubOp::MatmulTile { .. })
            && consumers[0].inputs.len() >= 2
            && consumers[0].inputs[0].tensor == node.output.tensor
            && consumers[0].inputs[0].region == node.output.region;
        if !clean {
            let kind = |c: &SubtileNode<F>| {
                if matches!(c.op, SubOp::MatmulTile { .. }) {
                    "MatmulTile"
                } else {
                    host_glue_kind(&c.op)
                }
            };
            return Err(SuperDscError(format!(
                "AttnDecode t{}'s output is read by {} consumer(s) ({}), and the head-major o_proj \
                 restructure needs exactly ONE matmul reading it as that matmul's whole `inputs[0]` \
                 — the attention finalize's slab-major form hands EVERY reader head-permuted bytes, \
                 so a second reader or a non-matmul one is a graph the restructure cannot describe. \
                 Fix the graph, not the recognition.",
                node.output.tensor.index(),
                consumers.len(),
                consumers
                    .iter()
                    .map(|c| kind(c))
                    .collect::<Vec<_>>()
                    .join(", "),
            )));
        }
        let o = consumers[0];
        edges.push(OprojEdge {
            w: o.inputs[1].tensor.index() as u32,
            hd: geom.hd().get(),
            k: o.inputs[0].region.cols.len,
            fp8: o.inputs.len() == 3,
        });
    }
    // ⛔ WHERE THE HEAD DIM IS MULTI-SLAB, PRECISION IS ONE FACT PER MODEL. The handoff is stated
    // per bundle, so a dense o matmul under a multi-slab attention (or the reverse mix) would
    // emit a slab-major finalize whose weight was never permuted. One model is one precision; a
    // graph carrying both is refused naming the disagreement. At one slab nothing applies and
    // no precision is demanded — the 2b's dense and fp8 builds both pass through untouched.
    let multi_slab = edges.iter().any(|e| e.hd > POOL_STICK);
    if multi_slab && edges.iter().any(|e| e.fp8 != edges[0].fp8) {
        let dense = edges
            .iter()
            .find(|e| !e.fp8)
            .expect("the mix was just measured");
        let fp8 = edges
            .iter()
            .find(|e| e.fp8)
            .expect("the mix was just measured");
        return Err(SuperDscError(format!(
            "the graph's o_proj matmuls disagree on precision — t{} is dense while t{} is fp8 W8A8 \
             over a multi-slab head dim. The head-major handoff is one fact per bundle, so half a \
             restructure (a slab-major finalize whose weight was never permuted) would be fluent, \
             wrong output. One model is one precision.",
            dense.w, fp8.w,
        )));
    }
    Ok(OprojEdges {
        // ⛔ AND ONE GEOMETRY ACROSS EDGES — the third term, and the one a hybrid-attention arch
        // waits on. The permuted walk, the swap key, and the per-layer family are all minted from
        // the FIRST edge (the rolled body's layer-0 representative), so a graph whose edges
        // disagree on `(hd, k)` would stage every other class's permuted copy through the
        // canonical class's walk while the bundle-wide handoff slab-majors every attention's
        // finalize — and the other classes' o matmuls are never swapped at all (the swap key
        // names only the canonical weight), so they contract slab-major activations against
        // un-permuted weights: half a restructure, fluent and wrong, card-only. A
        // mixed-geometry model therefore keeps the SHIPPED token-stream finalize — byte-for-byte
        // the pre-restructure emission, card-proven — until every class gets its own walk and
        // swap key. `edges[0]` is safe: `multi_slab` is false on an empty edge set, and the
        // short-circuit never reaches the index otherwise. Uniform multi-slab fp8 models
        // (granite-8b's one class) are unchanged.
        geometry_applies: multi_slab && edges.iter().all(|e| e.fp8) && {
            let e0 = &edges[0];
            edges.iter().all(|e| e.hd == e0.hd && e.k == e0.k)
        },
        edges,
    })
}

/// ⭐ THE HEAD-MAJOR o_proj COPY MINT — ONE implementation, TWO consumers: the placement pass
/// (`compute_bundle_layout`) places each `to` tid one block after its companion, and
/// [`graph_wiring`] bakes the SAME list into the generated wiring so the worker materializes
/// each copy at load. A registry two consumers re-derive separately is a registry that can
/// silently disagree — and did: #299's registry rewrite of `graph_wiring` left its copy of
/// this list on `Default` (empty) while the placement pass kept minting it, so every
/// head-major copy was placed-but-never-staged and the m=1 fp8 o matmul read never-written
/// device windows (`Fr4 (oon`, granite-3.1-8b fp8, card-only — the emulator executes the KTIR
/// program, not the staging). Same shared-helper law the wiring's other registries follow.
/// Pure function of `ir`, keyed on the GEOMETRY ([`OprojEdges::geometry_applies`]), never the
/// bundle's row count — every bundle of a model shares ONE staged weight segment, so a
/// rows-gated mint would hand the single-row body addresses no other bundle staged.
pub(crate) fn oproj_perm_copies<F: RopeForm>(
    ir: &SubtileIR<F>,
) -> Result<std::collections::BTreeMap<u32, u32>, SuperDscError> {
    let oproj = oproj_edges(ir)?;
    Ok(if oproj.geometry_applies {
        oproj
            .edges
            .iter()
            .map(|e| (e.w, oproj_headmajor_tid(e.w)))
            .collect()
    } else {
        std::collections::BTreeMap::new()
    })
}

/// ⭐ THE SPLIT-KV CHUNK THIS GRAPH'S ATTENTION CAN EXPRESS — `requested` when EVERY attention
/// class's pool admits it, `None` otherwise.
///
/// ⛔ `None` IS THE PER-PAGE FOLD, WHICH IS THE SHIPPED AND CORRECT DECODE — not a silently degraded
/// one. The chunk is refused where its V gather has no slab-major order (a head dim past one page's stick
/// blocks — `ChunkScratch::of_chunk`; hd 64, 128 and 256 are admitted), and a model there keeps exactly
/// the emission it had. Asked of the GRAPH — a CONJUNCTION over every `AttnDecode` geometry, never one
/// class's answer in another's place (#554: the ladder was admitted off the sliding class alone and the
/// door refused the global class's hd=512 at bake) — so a third-party producer's geometry decides for
/// itself; a graph with no attention node has nothing to chunk.
pub fn admitted_chunk<F: RopeForm>(ir: &SubtileIR<F>, requested: Option<u32>) -> Option<u32> {
    let pools = attn_pools(ir);
    requested.filter(|&pages| admits_chunk(&pools, pages))
}

/// The paged KV pools this graph's attention reads — one per DISTINCT `AttnDecode` geometry, in
/// first-appearance order. ⛔ EVERY CLASS, NOT THE FIRST: a hybrid model (gemma-4's sliding layers
/// plus every-sixth global ones) carries two geometries in one graph, and a chunk is lowered at
/// EVERY class's attention, so an admission read off one class speaks for a class that never
/// offered it (#554). Empty for a graph with no attention node.
fn attn_pools<F: RopeForm>(ir: &SubtileIR<F>) -> Vec<ktir_superdsc::sdsc_abstract::PagedKvPool> {
    let mut pools = Vec::new();
    for n in &ir.nodes {
        if let SubOp::AttnDecode { geom, .. } = &n.op {
            let pool = ktir_superdsc::sdsc_abstract::PagedKvPool::new(
                geom.nkvh().get() as usize,
                geom.hd().get() as usize,
            );
            if !pools.contains(&pool) {
                pools.push(pool);
            }
        }
    }
    pools
}

/// ⭐ THE ONE ADMISSION PREDICATE — a chunk width this graph's attention can express is a width
/// EVERY class's pool expresses ([`ChunkScratch::of_chunk`] at each). Both doors into the bake —
/// the ladder that mints a rung ([`admitted_chunk_ladder`]) and the bundle that narrows one
/// ([`admitted_chunk`]) — ask THIS, so a rung can never name a width some class's own door
/// refuses (#554: the two answers were read off different classes, and the bake panicked between
/// them).
fn admits_chunk(pools: &[ktir_superdsc::sdsc_abstract::PagedKvPool], pages: u32) -> bool {
    !pools.is_empty()
        && pools.iter().all(|pool| {
            ktir_superdsc::sdsc_abstract::ChunkScratch::of_chunk(*pool, pages).is_some()
        })
}

/// ⭐ THE CHUNK LADDER THIS GRAPH's GEOMETRY ADMITS for a body of `requests` request rows, ascending —
/// [`ktir_superdsc::sdsc_abstract::ChunkScratch::ladder_for`] at the graph's pools (one request for a
/// body whose rows share a history, the batch width for a decode batch). The rung widths are
/// pool-independent (`ChunkScratch::LADDER`; `B × J` for a batch), so one class's ladder enumerates
/// them and [`admits_chunk`] — the ONE predicate — admits each at EVERY class's pool: a graph whose
/// classes disagree offers no rung at all, which keeps the per-page fold, as [`admitted_chunk`]'s
/// `None` does.
pub fn admitted_chunk_ladder<F: RopeForm>(ir: &SubtileIR<F>, requests: u32) -> Vec<u32> {
    let pools = attn_pools(ir);
    pools
        .first()
        .map_or_else(Vec::new, |&pool| {
            ktir_superdsc::sdsc_abstract::ChunkScratch::ladder_for(pool, requests)
        })
        .into_iter()
        .filter(|&pages| admits_chunk(&pools, pages))
        .collect()
}

/// ⭐ A LADDER CHUNK's SWEPT EXTENT PER REQUEST — the rung key the body selector compares the live
/// context against ([`ktir_superdsc::sdsc_abstract::ChunkScratch::swept_per_request`]). `pages` comes off
/// [`admitted_chunk_ladder`] at the same `requests`; `None` for a chunk some class of this graph's
/// attention refuses, which the bake turns into a build failure (a rung its own ladder never
/// offered), never a guessed extent.
pub fn chunk_swept_per_request<F: RopeForm>(
    ir: &SubtileIR<F>,
    pages: u32,
    requests: u32,
) -> Option<u32> {
    let pools = attn_pools(ir);
    if !admits_chunk(&pools, pages) {
        return None;
    }
    // `swept_per_request` counts PAGES (`pages × PAGE_SLOTS` at one request, `J × PAGE_SLOTS` for a
    // batch) — no pool's geometry enters it — so any admitting pool's scratch states it and the law
    // stays in ONE place.
    ktir_superdsc::sdsc_abstract::ChunkScratch::of_chunk(pools[0], pages)
        .map(|chunk| chunk.swept_per_request(requests))
}

/// ⭐⭐⭐ THE BUNDLE'S ATTENTION PARAMETERS, READ WHERE MAIN READ THEM.
///
/// `ibm/main`'s `lower_one_node` lowered each node during the tape walk, so `SubOp::AttnDecode`'s
/// `geom` and `scale` were in its hand and `active_cap` / `rows_are_requests` were its own walk
/// parameters. This split lowers KTIR → SuperDSC one pass later, per BUNDLE, so the same facts
/// are read HERE — off the graph's own `AttnDecode` nodes, off this walk's parameters, and off the
/// fold recognition's own node set (the `scale` main read became TWO things downstream: the value
/// the program states when there is one to apply, and the in-weights fact when the weights carry
/// it — the fold's rewrite to 1.0, or a unit scale the model itself states) —
/// and travel to the door as [`crate::ktir_superdsc_door::BundleAttnParams`], an argument of the
/// call.
///
/// ⛔ THE MODEL FACTS MUST BE THE MODEL'S, SO A DISAGREEMENT IS AN ERROR AND NOT A CHOICE — for
/// the facts that ARE one-per-bundle. The geometry is no longer one of them: a hybrid model
/// (gemma-4's sliding + every-sixth global layers) carries TWO attention classes in one graph,
/// and the door mints each program's geometry from its own views, exactly as it mints rope's
/// head dim — so this function states no geometry at all. The in-weights fact keeps the law: one
/// `#[forward]` expansion is one model, every `AttnDecode` node in one graph must agree that its
/// score scale is carried by the staged weights (folded into them, or a unit multiplier) or that
/// the program states it, and a graph carrying both would refuse at the door's first attention
/// either way — so this refuses instead, naming both nodes while both are in hand.
///
/// `None` when the graph has no attention node: there is then nothing for the facts to be facts
/// of, and an attention program arriving at the door without them is that door's own build error.
pub(crate) fn attn_bundle_params<F: RopeForm>(
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
    // The decode ladder's ONE-REQUEST rung — the bake's own call-site fact
    // (`!is_prefill && decode_rows == 1 && seq_sym.is_none()` in the `#[forward]` emission):
    // a CONCRETE single-row decode, never a symbolic batch template and never a prefill chunk.
    one_request_decode: bool,
    // The split-KV chunk this rung bakes — the ladder's own bake fact: the one-request decode rung, a
    // prompt chunk's prefix, or a decode BATCH, whose chunk holds every request's pages end to end
    // (`ktir_superdsc::sdsc_abstract::BatchChunk`; see `BundleAttnParams::chunk_pages`).
    chunk_pages: Option<u32>,
) -> Result<Option<crate::ktir_superdsc_door::BundleAttnParams>, SuperDscError> {
    // The fold's node set, taken once: a node's scale is in the staged weights iff the
    // attention-fold recognition named it, OR the node states a UNIT scale — `√1.0` leaves the
    // weights carrying the whole scale with nothing folded, which is the gemma-4 arch's own
    // default (`attention_multiplier` 1.0). The two are disjoint by the recognition's own skip
    // (a node at scale 1.0 folds nothing), which is also why the door's `scale_in_weights` is a
    // fact about what the program cannot state, not a restatement of one of its values — see
    // `BundleAttnParams::scale_in_weights`.
    let folded: std::collections::HashSet<SubtileId> = attn_scale_weight_folds(ir)
        .into_iter()
        .map(|(id, ..)| id)
        .collect();
    let mut found: Option<(u32, u32)> = None;
    let mut first_in_weights: Option<bool> = None;
    for n in &ir.nodes {
        let SubOp::AttnDecode { scale, .. } = &n.op else {
            continue;
        };
        let t = n.output.tensor.index() as u32;
        // ⭐ THE OUTPUT'S OWN ROW COUNT — the bundle's attention width, which at a decode rung IS
        // the request count (`lower_attn` reads this same `region.rows.len` as its `mq`). The
        // head-major handoff is minted exactly where this is ONE row.
        let rows = n.output.region.rows.len;
        let n_in_weights = folded.contains(&n.id) || *scale == 1.0;
        match (found, first_in_weights) {
            (None, _) => {
                found = Some((t, rows));
                first_in_weights = Some(n_in_weights);
            }
            (Some((t0, _r0)), Some(f0)) => {
                // ⛔ THE GEOMETRY CHECK IS GONE, AND THE HYBRID MODEL IS THE REASON: gemma-4's
                // sliding layers and every-sixth global layer carry TWO attention classes in one
                // graph, and the door MINTS each program's geometry from its own views (the same
                // reading rope takes for its head dim) — so no per-bundle geometry exists to agree
                // on, and two classes in one graph is legitimate. The in-weights fact below keeps
                // the one-value-per-bundle law.
                if f0 != n_in_weights {
                    return Err(SuperDscError(format!(
                        "AttnDecode t{t0} carries its attention scale in its staged weights while \
                         AttnDecode t{t} states a device score multiply. One bundle is one model, \
                         and the door takes ONE such fact for the whole bundle; a graph carrying \
                         both cannot lower one attention without mislowering the other, so this \
                         refuses instead of choosing."
                    )));
                }
            }
            // `found` and `first_in_weights` are set in the same arm and never apart.
            (Some(_), None) => unreachable!("first_in_weights tracks found"),
        }
    }
    let oproj = oproj_edges(ir)?;
    let Some((t, rows)) = found else {
        return Ok(None);
    };
    // ⛔ THE CALL SITE'S FACT AND THE GRAPH'S OWN COUNT ARE ONE FACT. `one_request_decode`
    // names THIS bake as the one-request rung, and the graph's `AttnDecode` states its own
    // width; a call site naming the rung over a wider graph is not a bundle that quietly
    // skips the handoff — it is two authorities disagreeing about one count, and it refuses,
    // naming both. (The symbolic CB batch deliberately passes `decode_rows = 1` over a
    // 96-row template; the emission's `seq_sym.is_none()` term keeps that bake off this fact
    // entirely, so the refusal below can only fire on a real disagreement.)
    if one_request_decode && rows != 1 {
        return Err(SuperDscError(format!(
            "the bake names a one-request decode rung while AttnDecode t{t} is {rows} rows \
             wide — the call site's fact and the graph's own count disagree, and a handoff \
             minted on the wrong one would slab-major a finalize whose weight no permuted \
             copy answers to",
        )));
    }
    // ⛔ WHERE THE HANDOFF APPLIES, GEOMETRY IS ONE FACT PER MODEL. The permuted walk, the swap
    // key, and the per-layer family are all minted from the FIRST edge — the rolled body's
    // layer-0 representative — so a graph whose o_proj edges disagree on `(hd, k)` (a
    // hybrid-attention arch: gemma-4's sliding 16×256 class beside its every-sixth global
    // 16×512 one) would stage every other class's permuted copy through the canonical class's
    // walk while the bundle-wide handoff slab-majors EVERY attention's finalize — and the other
    // classes' o matmuls are never swapped at all (the swap key names only the canonical
    // weight), so they contract slab-major activations against un-permuted weights. Half a
    // restructure, fluent and wrong, on the card only: the emulator executes the KTIR program,
    // not this emission. Same refusal class as the precision mix in [`oproj_edges`], across
    // classes instead of precisions — a hybrid model keeps the shipped token-stream finalize.
    if one_request_decode && oproj.geometry_applies {
        let e0 = &oproj.edges[0];
        if let Some(other) = oproj.edges.iter().find(|e| e.hd != e0.hd || e.k != e0.k) {
            return Err(SuperDscError(format!(
                "the graph's o_proj matmuls disagree on attention geometry — t{} is hd {} k {} \
                 while t{} is hd {} k {}. The head-major o_proj handoff is minted from ONE class \
                 (the first edge, the rolled body's layer-0 representative): its permuted walk, \
                 swap key, and per-layer family would restructure every other class's o_proj with \
                 the canonical class's geometry — a slab-major finalize against an un-permuted \
                 weight, fluent and wrong. One handoff is one geometry; give every class its own \
                 walk and swap key, or keep the shipped token-stream finalize for the model.",
                e0.w, e0.hd, e0.k, other.w, other.hd, other.k,
            )));
        }
    }
    // ⭐⭐⭐ THE HEAD-MAJOR o_proj HANDOFF — the restructure's bundle fact, minted ONLY where it
    // is fully expressible, and every term is this bundle's own:
    //   * `one_request_decode` — the decode ladder's ONE-REQUEST rung, where a row IS a head
    //     and o_proj's contraction axis can be permuted to meet `out` where the accumulator
    //     holds it. NOT `rows_are_requests`: that arg is FALSE at the one-request rung (it
    //     gates the batched-decode emission — the W8A8 splices, the layout's row kind, the
    //     split-mb ban — and the single-row bake keeps the shipped value of every one of
    //     them) and TRUE of rungs two rows and wider, where the handoff cannot apply anyway;
    //     a one-token PREFILL is also one row, but its rows are positions and its bundle is
    //     the prefill chain's — both are excluded by the call-site fact, not by the count;
    //   * `rows == 1` — proven by the refusal above, not assumed here;
    //   * `geometry_applies` — the head dim is multi-slab AND the o matmul is fp8 W8A8 (see
    //     [`OprojEdges::geometry_applies`]; a dense o_proj keeps the shipped emission).
    // The layout mint keys on the SAME geometry gate but NEVER on the row count — every bundle
    // of a model shares one staged weight segment, so a rows-gated mint would hand this body
    // addresses no other bundle staged.
    let headmajor_handoff = one_request_decode && oproj.geometry_applies;
    Ok(Some(crate::ktir_superdsc_door::BundleAttnParams {
        rows_are_requests,
        // ⭐ GATHERED AT THE BATCHED RUNGS (per page) AND AT A CHUNKED ONE-REQUEST RUNG — never a
        // one-request body on the per-page gather, whose per-row mask read the broadcast staging
        // cannot feed (see `BundleAttnParams::gathers_kv`).
        gathers_kv: rows_are_requests || chunk_pages.is_some(),
        chunk_pages,
        scale_in_weights: first_in_weights.unwrap_or(false),
        headmajor_handoff,
        // The CANONICAL o_proj — the first edge's weight, the rolled body's layer-0
        // representative. `Some` whenever the dataflow found the o matmul, whatever the
        // handoff: it is a fact about the graph, and the door reads it only under the
        // handoff.
        oproj_wtid: oproj.edges.first().map(|e| e.w),
    }))
}

/// Lower ONE [`SubtileNode`] to its SuperDSC op(s) — the single source of the
/// per-`SubOp` match. `ir` is needed only for `AttnDecode`'s cache-capacity lookup.
/// ⭐⭐⭐ ONE NODE, ONE PROGRAM.
///
/// The SuperDSC lowering decomposes a node into several descriptor ops, because dxp schedules at
/// that grain. KTIR does not: a node's whole computation is one `func`, with its inputs loaded from
/// HBM and its output stored back, and the work division stated INSIDE it as the grid and its
/// loops. So this walk yields one [`EmittedOp`] per node, carrying that program.
pub(crate) fn lower_one_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    // Sweep KV extent for AttnDecode (paged-attn ladder rung); FULL ⇒ full cap. See main's
    // `lower_attn_node`.
    active_cap: ActiveCap,
    // See main's `lower_attn_node`: whether this bundle's rows are separate requests.
    rows_are_requests: bool,
) -> NodeLowering {
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
        // The per-expert scale — the fourth and last of the router index
        // rows. The splice's row (`routeexpertscale.py`, the lane-free
        // one-hot column scan with the scale riding the multiply's
        // broadcast) states it: the door's own `7k − 2` select chain over
        // the same minted views, IN THE DOOR'S MINT ORDER (the splice's
        // binding row permutes the node's `[scores, idx, scale]` to
        // `[scale, idx, scores]` — `tensors[0]` IS the scale row, the
        // recorded m=8 mis-read law), the emulator's column scan.
        SubOp::RouteExpertScale { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        // THE FUSED TOP-K PAIR — the wall-11 fix. The splice's row
        // (`routetopk.py`, the ripple-mirror form over the RAW LOGITS)
        // states it: the vendor `topkindex`/`topkvalue` selection as ONE
        // kernel whose K running registers live in the OUTPUT buffers, with
        // the argsort and the score gather FOLDED INTO IT (`lower_all`
        // silences both — see the arms below). ⛔ THE STICK LAW IS THE
        // DOOR'S, NOT A REFUSAL: the measured law (the reduction dim may
        // not ride the stick; the vendor's own `_check_stick_topk`) puts
        // the `[m, W]` scores' experts ON the stick, so the door
        // (`lower_ktir_to_superdsc::route_topk`) stages to `[m_pad, W]`,
        // Kᵀ-restickifies to `topk.ddl`'s own `[W, m]`-token-sticked
        // orientation, runs the pair at slots 64, and lands ids/values
        // token-major `[m_pad, 64]`. The pair's ids are scan positions —
        // unique by construction, never a sum of duplicate ranks past the
        // bank — which is why the fold is the wall-11 CULPRIT'S CURE, not
        // an optimization: the per-row-scalar argsort it replaces could
        // DUPLICATE rank targets at ties and fault the card's FETCH.
        // THREE parameters in the door's MINT ORDER `[lg, vals, ids]`
        // (positional, never `split_out`): the logits, the gather
        // consumer's output (this kernel lands the values), the node's own
        // ids output.
        SubOp::RouteTopK { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        // The bare score gather — the fold's leftover corner, not the
        // router chain's own. `lower_all` folds the gather away when its
        // ids come from a RouteTopK whose logits are this gather's source
        // (the fused pair lands the values itself), so this row runs only
        // for a gather whose ids arrive from anywhere else — the splice's
        // row (`routegather.py`, the lane-free one-hot column scan) states
        // that function, the door's own `6k − 2` one-hot select chain over
        // the same minted views.
        SubOp::RouteGatherScores => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        // The bare stable ascending rank vector — the fold's other leftover
        // corner. `lower_all` folds the argsort away when every consumer of
        // its rank vector is a RouteTopK (the fused pair reads the LOGITS,
        // not the ranks), so this row runs only for an argsort whose ranks
        // feed anything else — the splice's row (`routeargsort.py`, the
        // lane-free count form) states that function, the door's own
        // `9W + 1` compare/reduce chain over the same minted views. The
        // kernel's header states why the form is lane-free (`tl.arange`
        // refuses the chain fixture's `E = 5`).
        SubOp::RouteArgsort => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        // The gathered expert projection — the weight row each pair reads is selected
        // by its expert index (an `indirect` access-tile dim on the weight read), the
        // one door capability the dense matmul row does not state. The splice's row
        // (`expertmatmul.py`, the one-hot-over-E ladder form) states it: the door's
        // own five-leg chain over the same minted views, the emulator's
        // arithmetic-mask form over the bank.
        SubOp::ExpertMatmul { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::KvEncode { .. }
        | SubOp::KvStage { .. }
        | SubOp::RotateRows { .. }
        | SubOp::AttnPackedKv
        | SubOp::SampleRowsGather
        | SubOp::SampleRowsScatter
        | SubOp::AllRowsMatmul => Unhandled(format!("{:?} has no SuperDSC kernel", node.op)),
        // ── THE 2a KINDS — every one through `lower_all`: the TanhSoftCap logits fold
        // and the head-norm sandwich's three seats (Reshape split / norm middle /
        // Reshape flatten) are `lower_all`'s interceptions, and every other node of
        // these kinds is the ordinary one-program splice. The Reshape arm this
        // replaces used to refuse the RESTICKIFY by name — the stick-walk kernel
        // (`reshape.py`, one `[1, 64]` identity per flat stick through the door's own
        // `lmlast` mechanism) IS that re-laying copy now. ──
        //
        // ⭐ ROUTERLOGITS' MALFORMED-NODE GUARD FIRST (branch:
        // `lower_router_logits_node`): `node_to_single_tile_op` is the REFUSAL half
        // of the matmul row — the one place a malformed node is refused before a
        // program is built, with the dims read from exactly the regions the kernel
        // states. The splice cannot call it (it lives in this crate), so the guard
        // rides the routing arm.
        SubOp::RouterLogits { .. } => {
            if let Err(reason) =
                crate::ir::bridge::subtile_tape_tile_op::node_to_single_tile_op(node)
            {
                return Unhandled(reason);
            }
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::RouterNorm { .. }
        | SubOp::RouteSoftmax
        | SubOp::RouteScale { .. }
        | SubOp::RouteRenorm
        | SubOp::ExpertGatedAct { .. }
        | SubOp::ExpertSort { .. }
        | SubOp::ExpertUnsort
        | SubOp::ExpertCombine { .. }
        | SubOp::TanhSoftCap { .. }
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::Reshape { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::MatmulTile { .. } => {
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
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
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => {
            match scratchy_triton_splice::lower(node, ir, rows_are_requests) {
                Ok(e) => Ops(vec![e]),
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
            match scratchy_triton_splice::lower_all(node, ir, rows_are_requests) {
                Ok(ops) => Ops(ops),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::AttnDecode { layout: kv, .. } => {
            // ⭐ THE CACHE CAPACITY IS THE ONE GRAPH FACT THE SPLICE NEEDS STATED — the
            // same read main's builder control makes (`ir.tensors[kv.cache_tensor()].rows`),
            // and the swept rung rides the walk's own `active_cap`. Everything else main's
            // `lower_attn_node` threaded is either read off the program by the
            // door (`attn_at`: the swept extent, the scale, the span guard, the row laws)
            // or is a constexpr the kernel states from the node's own payload.
            let cap = ir.tensors[kv.cache_tensor().index() as u32 as usize].rows;
            // ⭐⭐⭐ THE SCALE, REWRITTEN TO 1.0 WHEN ITS FOLD APPLIES. The program is the ONE
            // carrier of the scale (the kernel states it as its `SCALE` constexpr and the door
            // reads that value back), so rewriting it HERE — at the boundary where the program
            // is minted — is the single edit that reaches every consumer: the door sees 1.0 and
            // emits no `qs`/`new_k_scaled` multiplies, the emulator interprets `qk · 1.0` over
            // the pre-scaled weights the fold staged, and the registry census skips the node's
            // arm. The REAL multiplier meanwhile rides the fold list into the staged W_q/W_k
            // bytes, where `(q·√s)·(k·√s)ᵀ` reproduces `(q·kᵀ)·s` exactly.
            let mut folded_node;
            let node_ref: &SubtileNode<F> = if attn_scale_weight_folds(ir)
                .iter()
                .any(|(id, ..)| *id == node.id)
            {
                folded_node = node.clone();
                if let SubOp::AttnDecode { scale, .. } = &mut folded_node.op {
                    *scale = 1.0;
                }
                &folded_node
            } else {
                node
            };
            match scratchy_triton_splice::lower_attn(node_ref, ir, cap, active_cap) {
                Ok(e) => Ops(vec![e]),
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
/// ⛔⛔⛔ THIS USED TO BE CALLED `lower_graph_to_superdsc`, AND KEEPING THAT NAME THROUGH THE SPLIT IS
/// WHAT WENT WRONG. It is `ibm/main`'s function, text unchanged — but main's `lower_one_node` returned
/// SuperDSC descriptors and this one returns KTIR programs, so the body silently stopped producing what
/// its name, its doc and its own worklist error (`"{} SuperDSC op(s) emitted ok"`) all still claim. A
/// caller that asked it for descriptors got `EmittedOp::bare` — `op: None`, `time: 1` — and every
/// question it then asked of them ("how many descriptors?", "is this op tiled?") was answered about a
/// program instead. The WHOLE lowering is [`lower_graph_to_superdsc`] below; this is its first half.
pub fn lower_graph_to_ktir<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    active_cap: ActiveCap,
    // See main's `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
) -> Result<(Vec<EmittedOp>, BundleLayout), SuperDscError> {
    // ⛔⭐ THE ADDRESSING LOCKDOWN RUNS FIRST — the general well-formedness pass
    // (the graph half of the lockdown; the binding half is the splice's M6 walk and
    // the door half is `expert_matmul`'s bank guard). Every index→address
    // composition is proven here, over the whole graph, BEFORE a program is minted:
    // the routing's id range vs the experts its banks pack, and the closed consumer
    // table that fail-closes on any new index-consumer class. A bake that cannot
    // prove its addressing does not bake.
    scratchy_triton_splice::addressing::verify_addressing(ir).map_err(SuperDscError)?;
    // GLOBAL ≤7-segment memory layout (task #55) computed ONCE for the whole bundle.
    // Every op resolves each tensor's HBM address from this (by `t{id}` name) so a
    // tensor shared across ops gets the SAME address — fixing the per-op `arg_index`
    // segment-aliasing bug. Threaded as `Some(&layout)` into every node-lowering; the
    // populated layout (incl. synthetic seg3 offsets) is RETURNED for the manifest.
    // ⛔ NO LAYER CLASSES: this is the UNROLLED lowering, which has no layer loop and therefore no
    // layer boundary to split the weight segment on. An empty map is what tells the layout that
    // banking is not expressible here, leaving the tail spill as the only lever (`&Default::default()`
    // rather than a bool, so there is one spelling of "the layer structure" and not two).
    let bundle_layout = compute_bundle_layout(
        ir,
        weight_ids,
        rows_are_requests,
        rows_are_requests,
        &[],
        // No composite argmax on the unrolled walk: the emulator's decode twin is unrolled
        // FROM the rolled bundle (`unroll_layers`), so it inherits the composite's op
        // there; this direct entry serves prefill (never one-request) and the emu-only
        // graphs, neither of which carries the tie token.
        None,
    )?;
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(ir.nodes.len());
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
        match lower_one_node(node, ir, active_cap, rows_are_requests) {
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

/// SubtileIR → KTIR → SuperDSC for a whole UNROLLED graph — [`lower_graph_to_ktir`] followed by the
/// ONE KTIR → SuperDSC lowering over each program it built. These are the descriptors main's
/// `lower_graph_to_superdsc` returned: the same builders, in node order, threading the same two
/// bundle-scoped accumulators main's single walk threaded.
///
/// ⛔ THE TWO ACCUMULATORS BELONG TO THE CONSUMER NOW, WHICH IS WHY THEY ARE MINTED HERE AND NOT
/// UPSTREAM. main incremented the negative-symbol-id counter and inserted into the fp8
/// activation-quantize dedup set as it built each DESCRIPTOR, so both are per-bundle facts of the
/// descriptor pass — the producer's copies are threaded but never written (main's
/// `lower_matmul_node`'s `_quantized`). `ktir_groups_via_superdsc` mints them at exactly this grain for exactly this reason.
pub fn lower_graph_to_superdsc<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    active_cap: ActiveCap,
    rows_are_requests: bool,
) -> Result<(Vec<EmittedOp>, BundleLayout), SuperDscError> {
    let (programs, mut bundle_layout) =
        lower_graph_to_ktir(ir, weight_ids, active_cap, rows_are_requests)?;
    // The four facts no KTIR states, read off the graph's own `AttnDecode` nodes and this walk's own
    // parameters — the same call `lower_subtile_tape_to_ktir`'s re-rolled walk makes.
    //
    // ⛔ AND THE HEAD-MAJOR HANDOFF IS ZEROED HERE, BECAUSE THIS IS THE UNROLLED BODY. The handoff's
    // matmul half keys on ONE canonical weight tid (the rolled body's layer-0 representative); this
    // walk lowers every layer's o matmul under its OWN per-layer weight tid, so a handoff here would
    // slab-major every attention's finalize while swapping only the canonical's weight. The
    // restructure is a rolled-body fact — see the codegen's `unrolled_params`, the same seal on the
    // bake's cardless arm. `one_request_decode` is false for the same reason: the unrolled body is
    // never the bake the fact names.
    let mut attn_params = attn_bundle_params(ir, rows_are_requests, false, None)?;
    if let Some(p) = attn_params.as_mut() {
        p.headmajor_handoff = false;
    }
    let mut sym_id_base: i64 = 0;
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(programs.len());
    for e in &programs {
        let k = e.ktir.as_ref().ok_or_else(|| {
            SuperDscError(format!(
                "{}: no KTIR program — the node lowering declined this op, and a bundle short a \
                 program computes something else",
                e.op_name
            ))
        })?;
        ops.extend(
            crate::ktir_superdsc_door::lower(
                k,
                &mut sym_id_base,
                Some(&bundle_layout),
                &mut fp8_quantized,
                attn_params,
            )
            .map_err(|err| {
                SuperDscError(format!("{}: KTIR -> SuperDSC: {}", e.op_name, err.message))
            })?,
        );
    }
    // (The former negative-symbol-id disjointness guard is GONE: the concrete-unroll
    // Synthetic intermediates were assigned seg3 offsets ABOVE the colored
    // intermediates during the walk; grow the Intermediate segment's byte count to
    // cover them so the executor allocates a seg3 region large enough (task #55/#56).
    //
    // ⛔ AFTER THE CONSUMER PASS, NOT AFTER THE PRODUCER'S. `BundleLayout::synth` is called by
    // the door's descriptor builders — `assemble_rmsnorm` / `matmul_fp8` / `lmlast` — so at the
    // end of `lower_graph_to_ktir` nothing has been declared yet and this grew seg3 by zero. That is
    // the same ordering fact the re-rolled walk pays for with its explicit declare pass.
    let seg3 = SegRole::Intermediate.segment();
    let synth_high = bundle_layout.synth.borrow().next;
    if synth_high > bundle_layout.segment_bytes[seg3] {
        bundle_layout.segment_bytes[seg3] = synth_high;
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
    /// `tensor_shapes[id] = (rows, cols)` for every tensor. When `attn_mask` is set, the LAST entry
    /// is the synthetic mask tensor `[1, capacity]`.
    pub tensor_shapes: Vec<(u32, u32)>,
    /// Tensor id of the shared attention runtime length-mask source, if the graph has a maskable
    /// decode. The host fills it `[1, capacity]` per forward step (0 on valid columns, -inf past
    /// the decode position); it is NOT one of `num_sources` and is written by no node.
    pub attn_mask: Option<u32>,
    /// EVERY compile-time scalar the programs read, in registry order: entry `i` is the value the
    /// worker must bind at [`scalarmul_scale_tid`]`(i)` — the model's own multipliers and RMSNorm
    /// epsilons, exactly the set and exactly the order `subtile→superdsc` registers.
    ///
    /// ⛔ NOTHING IS SEEDED INTO IT AND NOTHING EXTRA IS PUSHED. The index IS the device tid, so an
    /// added entry moves every constant after it. Two algebraic identities were once seeded at the
    /// front for this construction's `linalg.*` `outs` seeds, and the mean-of-squares divisor was
    /// pushed beside each epsilon; both are gone — the seeds are immediates, and `1/cols` is bound at
    /// the reserved `RMS_INVCOLS_TID` the way the proven path binds it.
    ///
    /// ⛔ CARRIED HERE BECAUSE A CONSTANT A DESCRIPTOR READS IS A BOUND TENSOR. `dxp_standalone`
    /// has no immediate operand, so the programs read these off reserved tids that
    /// appear in `func.arguments` like any other buffer — which means BOTH consumers of this KTIR
    /// must fill them, the card path through `wiring::constant_steps` and the emulator through its
    /// own source binding. This is the one list they read, so they cannot disagree about it.
    pub scalarmul_scales: Vec<f32>,
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
    /// The head-major o_proj permuted copies: `(companion weight id, permuted copy id)` for each
    /// second, k-shuffled o_proj copy the layout minted. The unrolled walk's wiring carries them
    /// for the SAME reason it carries the folds — `spyre_load` materializes every listed copy at
    /// load, and a wiring that omitted one while the layout placed it would stage a copy whose
    /// bytes nothing filled. (The unrolled body never READS its permuted copies — the handoff is a
    /// rolled-body fact, see the codegen's `unrolled_params` — but the mint is bundle-invariant,
    /// so the copies exist here too and are staged and unread, like any other resident weight.)
    pub weight_copies: Vec<(u32, u32)>,
    /// ⭐ THE KERNEL-TABLE CLASS SETS — this graph's DISTINCT rope head dims (`rope_class_hds`)
    /// and attention head dims (`attn_class_hds`), sorted descending, published off the
    /// `BundleLayout` registry this walk threads: the ONE-source law `scalarmul_scales` states
    /// above, applied to the class sets. The door resolves a rope program's class index off the
    /// LAYOUT's list (`BundleLayout::rope_class_hds`, the registry mechanism
    /// `scalarmul_scales` established), so the worker's load-time table binds must be THAT list
    /// and not a second derivation free to disagree — the runtime's
    /// `Wiring::verify_class_placements` cross-checks exactly this against the baked placements.
    /// Index `i` ↔ `rope_p_class_tid(i)` / `identity_class_tid(i)`; class 0 is the widest and
    /// keeps the pre-class sentinel tid, so a uniform model's binds are byte-identical.
    /// `attn_class_hds` is EMPTY on this ladder — `AttnDecode` is refused by name and the door
    /// mints per-program geometry — but the field is carried anyway: the worker's
    /// `synthetic_constants` builds each class's identity table off it, and a door that lands
    /// attention classes states its set here the day it lands (the registry row already exists).
    pub rope_class_hds: Vec<u32>,
    pub attn_class_hds: Vec<u32>,
}

/// One program's parameter order: `args[i]` is the tensor the i-th parameter addresses.
pub struct NodeArgs {
    pub args: Vec<usize>,
}

/// ⭐ THE WIRING'S REGISTRIES — the layout facts [`graph_wiring`] rebuilds as pure functions of
/// `ir`, in the wiring's own shapes. ⛔ NO `Default`, and the literal in `graph_wiring` is
/// EXHAUSTIVE: a registry added to this struct (or to [`BundleWiring`]) is a MISSING FIELD at
/// the one construction site — a build error, not a silent empty list on the card. This is
/// the class #299's registry rewrite of `graph_wiring` dropped: `weight_copies` stayed on
/// `BundleLayout::default()` while the placement pass kept minting it, so every head-major
/// copy was placed-but-never-staged and the m=1 fp8 o matmul read never-written windows
/// (`Fr4 (oon`, granite-3.1-8b fp8, card-only). Before that rewrite the wiring ran the whole
/// placement pass for these five fields — a throwaway budget pass that refused models the
/// re-rolled bake had accepted (the 26b MoE panic that motivated the rewrite) — so the
/// registries stand alone here, exhaustively built.
pub(crate) struct WiringRegistries {
    pub scalarmul_scales: Vec<f32>,
    pub weight_scale_folds: Vec<(u32, f32)>,
    pub weight_copies: Vec<(u32, u32)>,
    pub rope_class_hds: Vec<u32>,
    pub attn_class_hds: Vec<u32>,
}

/// Lower `ir` for its WIRING alone — the same per-node arms a bake runs, read for the parameter
/// pairings rather than for the programs.
///
/// ⛔ THE LAYOUT IS NOT OPTIONAL, even though the pairing does not depend on placement. It is also
/// the REGISTRY the lowering writes constants into — the door registers its attention
/// scale in `BundleLayout.scalarmul_scales` and refuses without one ("registry desync") — so a
/// wiring pass that threads `None` does not get placement-independent answers, it gets no answers.
pub fn graph_wiring<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
) -> Result<BundleWiring, SuperDscError> {
    // ⛔ A REGISTRY, NOT A PLACEMENT — the wiring reads ONE kind of thing off the layout: the
    // registries ([`WiringRegistries`]). The placement pass [`compute_bundle_layout`] computed
    // here would be a THROWAWAY (the shipped layout is the re-rolled bake's), and it is the
    // CARD's budget pass: it refuses a weight segment past one 16 GiB region with no layer
    // classes to bank on — MEASURED on the old ladder, the 26b MoE under the fp8 preset wired
    // fine through the banked re-rolled bake and then panicked HERE on a layout nobody reads.
    // Every registry is a pure function of `ir.nodes` (the same helpers the placement pass
    // itself fills from), so that is what [`WiringRegistries`] builds — exhaustively, so a
    // registry added later is a missing field at the literal, not a silent `Default` on the
    // card.
    let _ = weight_ids;
    let registries = WiringRegistries {
        scalarmul_scales: scalar_registry(ir),
        // Same rule as the placement pass's own field: the fold list the worker applies to
        // staged bytes at load, keyed by weight tid (the `SubtileId` half is a placement-pass
        // fact and does not ship).
        weight_scale_folds: weight_scale_folds(ir)
            .into_iter()
            .map(|(_, tid, scale)| (tid, scale))
            .collect(),
        // ⭐ THE COPY MINT — [`oproj_perm_copies`], the placement pass's own helper, so the
        // wiring's list and the layout's placements cannot disagree.
        weight_copies: oproj_perm_copies(ir)?.into_iter().collect(),
        // ⭐ THE CLASS REGISTRIES, here too — the door resolves a rope/attn program's P/identity
        // class through these, and this pass lowers through the door. Same shared helper, same
        // tape, so the wiring's answer and the placement pass's cannot disagree.
        rope_class_hds: ktir_superdsc::reserved_tids::rope_p_class_hds(ir.nodes.iter().filter_map(
            |n| match &n.op {
                SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
                    Some(head_dim.get())
                }
                _ => None,
            },
        )),
        attn_class_hds: ktir_superdsc::reserved_tids::rope_p_class_hds(ir.nodes.iter().filter_map(
            |n| match &n.op {
                SubOp::AttnDecode { geom, .. } => Some(geom.hd().get()),
                _ => None,
            },
        )),
    };
    // ⭐ EVERY NODE IS LOWERED ON ITS OWN, so they are lowered IN PARALLEL and folded IN ORDER. Each
    // node's arm is a pure function of the node and the graph (the kernel memos it reaches are
    // process-wide, so a kernel two layers share still compiles once), and this walk is over the
    // UNROLLED graph: on granite-2b it was two thirds of the models build script, serially, for one
    // fact per program — its bindings — and one per mask. The fold below sees the results in node
    // order, so the first error and the mask check read exactly as the serial walk did.
    // One program's binding order, and the `(mask tid, capacity)` it reads, when it reads one.
    type WiredProgram = (NodeArgs, Option<(u32, u32)>);
    let per_node: Vec<Result<Vec<WiredProgram>, SuperDscError>> = {
        use rayon::prelude::*;
        ir.nodes
            .par_iter()
            .map(|node| {
                let ops = match lower_one_node(node, ir, ActiveCap::FULL, false) {
                    NodeLowering::Ops(v) => v,
                    // A host-routed op runs off the device, so it has no program and binds nothing. It
                    // still occupies a position in the node order, so it contributes an empty entry.
                    NodeLowering::HostRouted(_) => Vec::new(),
                    NodeLowering::Unhandled(why) => return Err(SuperDscError(why)),
                };
                let mut out = Vec::with_capacity(ops.len());
                for e in ops {
                    let Some(k) = e.ktir else { continue };
                    let mask = match k.mask {
                        // ⭐ THE CAPACITY COMES OFF THE PROGRAM, not off the node: the resident cache's
                        // own `ktdp.construct_memory_view` states `[capacity, kv_width]`, so the number
                        // this pass needs to size the host buffer is read where every other extent is
                        // read. It used to ride on `KtirNode::mask` as the second half of a pair the
                        // LOWERING never looked at.
                        //
                        // ⛔ THE CACHE'S VIEW, NOT THE MASK'S. The mask states `[1, capacity]` only at
                        // decode; a multi-row continuation binds one `[mq, capacity + mq]` tile (the
                        // prefix's length mask beside the new block's row mask), whose columns are not
                        // the capacity. The resident K is the program's third parameter in every arm
                        // (q, out, kc, kd, vc, vd — `attn_operands`'s order).
                        Some(mid) => {
                            let regs = ktir_superdsc::emit::lower_ktir_to_superdsc::regions(&k)
                                .map_err(|e| SuperDscError(e.message))?;
                            if !regs.iter().any(|r| r.tid == mid.get()) {
                                return Err(SuperDscError(format!(
                                    "the mask buffer t{mid} is not among this program's parameters \
                                     — the attention bound a mask its own program never declares"
                                )));
                            }
                            let cap = regs
                                .get(2)
                                .filter(|r| r.tid != mid.get())
                                .map(|r| r.v_rows)
                                .ok_or_else(|| {
                                    SuperDscError(format!(
                                        "the attention binding mask t{mid} states no resident-cache \
                                         parameter at position 2, so there is no `[capacity, \
                                         kv_width]` view to size the host buffer with"
                                    ))
                                })?;
                            Some((mid.get(), cap))
                        }
                        None => None,
                    };
                    out.push((
                        NodeArgs {
                            args: k.bindings.iter().map(|b| b.get() as usize).collect(),
                        },
                        mask,
                    ));
                }
                Ok(out)
            })
            .collect()
    };
    let mut nodes = Vec::with_capacity(ir.nodes.len());
    let mut mask: Option<(u32, u32)> = None;
    for lowered in per_node {
        for (args, m) in lowered? {
            if let Some(m) = m {
                if mask.is_some_and(|prev| prev != m) {
                    return Err(SuperDscError(format!(
                        "the attention mask must be uniform across layers (one shared prefix \
                         capacity), but {:?} and {m:?} were both emitted",
                        mask.unwrap()
                    )));
                }
                mask = Some(m);
            }
            nodes.push(args);
        }
    }
    let mut tensor_shapes: Vec<(u32, u32)> = ir.tensors.iter().map(|t| (t.rows, t.cols)).collect();
    let attn_mask = match mask {
        Some((mid, cap)) => {
            // The mask tensor sits at id == ir.tensors.len(); append its `[1, capacity]` shape so
            // host buffer allocation covers it.
            if mid as usize != tensor_shapes.len() {
                return Err(SuperDscError(format!(
                    "the mask's tensor id is {mid} but the next free id is {} — the mask is a \
                     synthetic source placed one past the graph, and an id that is not the next \
                     free one is an id some real tensor already answers to",
                    tensor_shapes.len()
                )));
            }
            tensor_shapes.push((1, cap));
            Some(mid)
        }
        None => None,
    };
    Ok(BundleWiring {
        nodes,
        num_sources: ir.num_sources,
        result_tensor: ir.result.index() as u32,
        tensor_shapes,
        attn_mask,
        scalarmul_scales: registries.scalarmul_scales,
        weight_scale_folds: registries.weight_scale_folds,
        weight_copies: registries.weight_copies,
        // The class sets, off the SAME registries the two lists above come from — the
        // registry the door resolves class indices against, so the worker's table binds and
        // the emitter's class tids agree by construction (see the field's doc).
        rope_class_hds: registries.rope_class_hds,
        attn_class_hds: registries.attn_class_hds,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// ⭐⭐⭐ THE COMPOSITE ARGMAX — the walk-side recognition (#603 PR3a loop closure)
//
// The suffix already ends at the logits (`ir.result`, past the lm-head tail). On a TIED
// ONE-REQUEST rung the next token needs only `argmax(logits)` — the host reads the whole
// `[1, vocab]` row back to compute what an eight-link device chain can compute in place,
// and the gathered embedding row is one `gather(table, V − max((V − iota)·(logits==max)))`
// away, first-max by arithmetic (#631's law made structural). The composite itself is
// minted by the walk below as ONE KTIR-first op appended to the suffix; this recognition
// is the fact-finding half — run BEFORE the layout (the layout must place the chain's
// two resident fp32 constants) and decided from the graph's OWN result tail, never from
// a shape:
//
//   result ← ScalarMul(logits scaling, granite) / TanhSoftCap(gemma) ← MatmulTile(Dense)
//
// The single-input column-preserving tail ops are exactly the set the splice calls
// row-local; the walk stops at the first producer that is not one, and the MATMUL it
// stops at is the lm-head by construction (it produces the logits). The TABLE is the
// authority for the composite's geometry — the gather reads its rows at width `hidden` —
// so vocab/hidden come off the table's own tensor shape, cross-checked against the
// matmul's `n`, the contraction K, and the result's cols. Arity-2 `GemmWeight::Dense`
// only: a STAGED lm-head (fp8 W8A8, arity-3, or MLX-affine) is a boundary this first cut
// refuses LOUDLY, never a silent host fallback — the model declared tied, so a quiet host
// round-trip here would hide a broken emission behind a working one.
struct ArgmaxPlan {
    /// The tie token the expansion minted from the model's own config.
    tied: TiedLmHeadEmbedding,
    /// The logits — `ir.result` itself, the tensor the composite reads (post tail).
    logits_tid: u32,
    /// The lm-head weight — the Dense matmul's B operand, the table the gather reads.
    table_tid: u32,
    /// The table's OUT count (the matmul's `n`) — the gather's bank height, and the
    /// ramp's width. The IR spells the weight `[K, N]` = `[hidden, vocab]`; the mint's
    /// `[vocab, hidden]` view is the tied bank's own OUT-MAJOR staging form (see
    /// `emit::argmax`'s module doc — token v's embedding is out-slot v, one contiguous
    /// stick range), so this is the SAME bytes the matmul reads, never a second staging.
    vocab: u32,
    /// The table's K (the contraction width) — the gather's slab, and the row the
    /// composite writes.
    hidden: u32,
    /// Every node of the recognized tail (the tail ops AND the lm-head matmul), by node
    /// id — validated as SUFFIX nodes after the walk: the lm-head runs once, after the
    /// last layer, never inside a rolled body.
    chain_nodes: Vec<u32>,
}

/// Walk the result tail backward to the lm-head, and state the composite's facts.
///
/// ⛔ NEVER A SILENT HOST FALLBACK: this returns `Err` — a build failure naming what it
/// found — for every tail it cannot recognize. The caller gates on the tie token, so an
/// `Err` here means a TIED model's emission changed shape; hiding that behind the host
/// readback would pin the regression in place.
fn recognize_argmax<F: RopeForm>(
    ir: &SubtileIR<F>,
    tied: TiedLmHeadEmbedding,
    weight_ids: &std::collections::HashSet<u32>,
) -> Result<ArgmaxPlan, SuperDscError> {
    // k=1 first cut: one request, one row of logits. The expansion gates the tie token
    // on `one_request`; a multi-row result reaching here is a wiring inconsistency, and
    // the composite has no batched form to fall back to.
    let result_rows = ir.tensors[ir.result.index()].rows;
    if result_rows != 1 {
        return Err(SuperDscError(format!(
            "reroll-superdsc: the composite argmax was handed a {result_rows}-row result — it \
             is a k=1 first cut (one request, one row of logits), so the tie token must not \
             reach a multi-row rung"
        )));
    }
    let mut chain_nodes = Vec::new();
    let mut t = ir.result;
    let node =
        loop {
            let node = ir.nodes.iter().find(|n| n.output.tensor == t).ok_or_else(|| {
            SuperDscError(format!(
                "reroll-superdsc: the composite argmax walked the result tail back to tensor \
                 {}, which no node produces — the graph is not one this walk built",
                t.index()
            ))
        })?;
            chain_nodes.push(node.id.index() as u32);
            match &node.op {
                // The single-input column-preserving tail ops — granite's logits-scaling
                // ScalarMul, gemma's TanhSoftCap. Their output still feeds the result
                // readback; the composite reads THROUGH them (a positive scaling cannot move
                // an argmax), so they stay exactly where they are.
                SubOp::ScalarMul { .. } | SubOp::TanhSoftCap { .. } if node.inputs.len() == 1 => {
                    t = node.inputs[0].tensor;
                }
                SubOp::MatmulTile { .. } => break node,
                other => {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: the composite argmax cannot recognize the lm-head tail: \
                     the result's producer chain reaches {other:?}, which is neither the \
                     single-input tail op set (ScalarMul, TanhSoftCap) nor the lm-head \
                     MatmulTile — the tied one-request emission needs a new recognizer, \
                     never a silent host fallback"
                    )));
                }
            }
        };
    let SubOp::MatmulTile { n, weight } = &node.op else {
        unreachable!("the loop broke on a MatmulTile");
    };
    if node.inputs.len() != 2 || !matches!(weight, GemmWeight::Dense) {
        return Err(SuperDscError(format!(
            "reroll-superdsc: the composite argmax found the lm-head matmul at arity {} with a \
             {:?} weight — the tied gather reads a Dense [vocab, hidden] table in place, and a \
             staged lm-head (fp8 W8A8 carries its scale as a third operand; MLX-affine packs \
             the table) is a boundary this first cut refuses loudly",
            node.inputs.len(),
            weight.kind()
        )));
    }
    let table_tid = node.inputs[1].tensor.index() as u32;
    if !weight_ids.contains(&table_tid) {
        return Err(SuperDscError(format!(
            "reroll-superdsc: the composite argmax's table tensor {table_tid} is not in the \
             weight set — the lm-head B operand must be the staged weight the gather reads"
        )));
    }
    let table = &ir.tensors[table_tid as usize];
    // The IR spells the matmul's weight `[K, N]` — rows are the contraction width,
    // cols the out count. The composite's own `[vocab, hidden]` view is the tied bank's
    // out-major staging form over the same placement (see `ArgmaxPlan::vocab`).
    let (hidden, vocab) = (table.rows, table.cols);
    // THE CROSS-CHECKS — the matmul's own `n`, the contraction K and the result's cols
    // are three more spellings of the table's geometry; a mismatch is a graph
    // inconsistency, not a recognizer miss, and it is named as one.
    let result_cols = ir.tensors[ir.result.index()].cols;
    let k = ir.tensors[node.inputs[0].tensor.index()].cols;
    if vocab != *n || vocab != result_cols || k != hidden {
        return Err(SuperDscError(format!(
            "reroll-superdsc: the composite argmax's geometry is inconsistent — the lm-head \
             table is [{vocab}, {hidden}], the matmul states n={n} over K={k}, and the result \
             is [1, {result_cols}]; four spellings of one geometry must agree"
        )));
    }
    Ok(ArgmaxPlan {
        tied,
        logits_tid: ir.result.index() as u32,
        table_tid,
        vocab,
        hidden,
        chain_nodes,
    })
}

/// What the ladder driver states about THE RUNG being lowered — the compile-time facts
/// that distinguish this bundle from its siblings on the same ladder. Grouped (not
/// positional) because the signature was at the argument-count law's ceiling the moment
/// the composite's tie token arrived, and because these three travel together: every
/// one of them is a fact ABOUT THE RUNG, stated by the driver that named it, explicit
/// at every call site (no `Default`, no silent facts — the `set_rows_are_requests`
/// ambient-state deletion's own law: a bundle fact belongs in a type).
pub struct RungFacts {
    /// The decode ladder's ONE-REQUEST rung — see [`attn_bundle_params`]. The `#[forward]`
    /// emission states it as `!is_prefill && decode_rows == 1 && seq_sym.is_none()`.
    pub one_request: bool,
    /// The split-KV chunk this rung bakes — see `BundleAttnParams::chunk_pages` (a
    /// one-request or prompt-chunk rung; `None` for the whole-cap rungs).
    pub chunk_pages: Option<u32>,
    /// ⭐⭐⭐ THE COMPOSITE ARGMAX'S TIE TOKEN — `Some` only on a TIED ONE-REQUEST rung:
    /// the expansion mints it from the model's own `tie_word_embeddings` under
    /// `one_request`, so batched rungs, prefill and every untied model keep the host
    /// readback path unchanged. The token itself is the guard
    /// (`TiedLmHeadEmbedding::from_tied_config` refuses an untied model), and a tied
    /// graph whose tail this walk cannot recognize is a LOUD error — never a silent
    /// host fallback (see `recognize_argmax`).
    pub argmax: Option<TiedLmHeadEmbedding>,
}

/// RE-ROLLED tape-driven SuperDSC lowering — the mirror of `lower_subtile_tape_to_tk_tape`
/// for SuperDSC (the fix for the 40-min unrolled-bundle compile). Walks the rerolled
/// `tape` (`subtile_tape::reroll_subtile_tape`): pre-loop Computes → `prefix`; the
/// `OpenLoop(Const(iters))`..`CloseLoop` body → `body` (lowered ONCE via the shared
/// `lower_one_node`); post-loop Computes → `suffix`. Collects the per-layer tid map
/// (`Compute::per_layer_out` for outputs + `ComputeInput::External::per_layer` for
/// weights) for the executor's per-iteration address advance.
pub fn lower_subtile_tape_to_ktir<F: RopeForm>(
    tape: &scratchy_subtile::subtile_tape::SubtileTape,
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    // The driver calls this once per rung with a different `active_cap`; every rung shares the SAME
    // resident KV (storage stays `cap`) and differs only in the body's swept extents.
    active_cap: ActiveCap,
    // See main's `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
    // The rung's own facts — the one-request flag, the chunk it bakes, and the
    // composite argmax's tie token. See [`RungFacts`].
    rung: RungFacts,
) -> Result<RolledSuperDsc, SuperDscError> {
    use scratchy_subtile::subtile_tape::{ComputeInput, Instr, LoopBound};
    let RungFacts {
        one_request: one_request_decode,
        chunk_pages,
        argmax,
    } = rung;
    let _pc = crate::pc::Pc::start("producer");
    // ⛔⭐ THE ADDRESSING LOCKDOWN RUNS FIRST — the same graph-half sweep
    // `lower_graph_to_ktir` states for the unrolled walk, run here for the tape's
    // re-rolled one: the whole graph's index→address compositions proven BEFORE the
    // segment walk mints a program. One implementation (the splice's `addressing`
    // module), two bake heads.
    scratchy_triton_splice::addressing::verify_addressing(ir).map_err(SuperDscError)?;
    // ⭐ THE CHUNK THIS GRAPH'S EVERY ATTENTION CLASS CAN EXPRESS, decided ONCE for both of its
    // consumers — the index placement (`compute_bundle_layout`'s `gathers_kv`) and the door's
    // `BundleAttnParams` — so an index is reserved exactly where a chunk gathers through it.
    let chunk_pages = admitted_chunk(ir, chunk_pages);
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
    // one, and the form it asked for measured slower on the card (`matmul/dims.rs`).
    // ⭐ THE LAYER STRUCTURE FIRST, because the layout needs it: a weight BANK boundary may only fall
    // on a LAYER boundary, and `compute_bundle_layout` is where the weight segment is packed. Same
    // tape, same `ComputeInput::External::per_layer` the walk below reads — collected once, here.
    // ONE MAP PER ATTENTION CLASS: a class-split roll's loops carry different-length per-layer
    // tables, and the layout needs to know which class a tensor belongs to before it can pack.
    let per_layer_ext = per_layer_external_tids(tape);
    // ⭐⭐⭐ THE COMPOSITE ARGMAX'S RECOGNITION — BEFORE the layout, because the layout must
    // place the chain's two resident fp32 constants (the ramp `[1, vocab]` and the
    // scalar-V stick), and from the graph's own result tail, because that is the only
    // place the fact lives. `argmax` is `Some` only on a TIED ONE-REQUEST rung, so a tied
    // graph this walk cannot recognize is a LOUD error here — never a silent host
    // fallback (the model declared tied; a quiet host round-trip would hide a broken
    // emission behind a working one).
    let argmax_plan = match argmax {
        Some(tied) => Some(recognize_argmax(ir, tied, weight_ids)?),
        None => None,
    };
    // The layout's budget passes (spill/bank) take the per-loop maps directly: the spill reasons
    // over WHICH tids are per-layer at all (flattened internally), and the BANK needs the class
    // boundaries to know which (class, iteration) launches must stay in one bank.
    let mut bundle_layout = compute_bundle_layout(
        ir,
        weight_ids,
        rows_are_requests,
        // The SAME fact `attn_bundle_params` mints `BundleAttnParams::gathers_kv` from — the batched
        // rungs' per-page gather, or a chunked one-request rung — so the index placement and the
        // emitter's admission are one predicate.
        rows_are_requests || chunk_pages.is_some(),
        &per_layer_ext,
        // The composite's ramp/V placements ride the same layout pass — placed iff the
        // recognition fired.
        argmax_plan.as_ref().map(|p| p.vocab),
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
    // ⭐ THE FIRST BODY'S hidden-in IS THE EMBEDDING-SOURCE SLOT — the graph's input
    // activation, where the runtime lands the gathered embedding row at the start of
    // every step. The composite argmax writes ITS row (the next token's embedding) into
    // the same placement, which the pre-pass below makes the shared residual buffer.
    // Declared here (not inside the pre-pass) because the mint reads it after the walk.
    let mut embed_row_tid: Option<u32> = None;
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
        // ⭐ THE FIRST BODY'S hidden-in IS THE EMBEDDING-SOURCE SLOT — captured here
        // (see the declaration above the block); the composite argmax's row binding is
        // this tid.
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
            if embed_row_tid.is_none() {
                embed_row_tid = Some(hin);
            }
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
    let mut prefix: Vec<EmittedOp> = Vec::new();
    let mut suffix: Vec<EmittedOp> = Vec::new();
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
    let mut bodies: Vec<crate::lower_subtile_tape_to_superdsc::RolledBody> =
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
    // Every SUFFIX Compute node id — the composite argmax's recognized tail must live
    // wholly in the suffix (the lm-head runs once, after the last layer, never inside a
    // rolled body). Collected during the walk because only the walk knows which nodes
    // the roll put there.
    let mut suffix_node_ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    // Matmul KERNEL weights (layer-0 tid, in=k, out=n) for the device re-tile manifest.
    let mut kernel0: Vec<(u32, u32, u32)> = Vec::new();
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
                // ⛔⛔⛔ THE GEMM-WEIGHT MANIFEST GATE — EXHAUSTIVE BY CONSTRUCTION (W29).
                //
                // The classification comes from the REGISTRY (`SubOp::gemm_weight_input` — one
                // row per op, a new gemm kind is a compile error until it answers), and this
                // collection is EXHAUSTIVE OVER IT: every op that carries a gemm weight must
                // land in one of the named arms below, and one that does not REFUSES THE BAKE.
                //
                // WHY REFUSE, AND WHY HERE: the three arms below were held together only by the
                // "every weight crosses the m=1 util floor" assumption, and the router broke
                // that silently. `RouterLogits`' `[E,hidden]` projection was EXCLUDED from the
                // kernel-weight manifest, so the shim's `None` arm staged the checkpoint's
                // row-major bytes VERBATIM into the stick-tiled slot the gemm addresses: the
                // 26b card computed `A @ perm(W)` — right ranking form, wrong experts
                // ([113,122,28,0,…] vs host [102,79,114,92,…]) → junk (root-caused + on-card
                // 2026-10-13). The same exclusion class had already fired once before as the
                // fp8 arity gap (arity-2-only collection → flat-staged fp8 weights → scrambled
                // `scr chat`, 2026-07-16). A THIRD kind would fire the same way and nothing
                // downstream would say so — the emulator reads a weight through the program's
                // row-major transpose-B `indexing_maps`, not through the card's per-core stick
                // addressing, so a staging-layout defect is numerically INVISIBLE to every emu
                // suite. THE BLIND SPOT THIS GATE COVERS: the bake-time numeric-equivalence
                // gate runs the EMULATOR build, and the emulator never executes hardware
                // staging — so the numeric gate cannot see staging defects at all. This gate
                // (exhaustiveness + the width-law agreement below) is the coverage for exactly
                // that hole: it refuses at BAKE, from the bake's own data, before any card or
                // emulator runs.
                //
                // Collect matmul KERNEL weights for the device re-tile manifest: w =
                // inputs[1] is [k,n]=[in,out] row-major; the PT array needs the device
                // tile layout [out/64, in, 64]. in=A.cols (k), out=output.cols (n).
                // Arity-3 fp8 W8A8 matmuls MUST be collected here too: their weight (inputs[1]; inputs[2] is
                // w_scale) needs the 1-byte packed RetileDescriptor from the `fp8_weight_tids` branch below.
                // in_k/out derive identically (inputs[0].cols=k, output.cols=n); the
                // KSPLIT branch (out>16384) never fires for fp8 projs.
                // ⭐ THE ROUTER'S OWN GEMM IS A KERNEL WEIGHT TOO (`RouterLogits`, arity 2, W = inputs[1] —
                // same tile dims: `matmul_tile_dims` serves both kinds).
                match n.op.gemm_weight_input() {
                    // kernel0-collected: the dense/router gemms — W = inputs[1].
                    Some(1)
                        if matches!(
                            n.op,
                            SubOp::MatmulTile { .. } | SubOp::RouterLogits { .. }
                        ) && (n.inputs.len() == 2 || n.inputs.len() == 3) =>
                    {
                        kernel0.push((
                            n.inputs[1].tensor.index() as u32,
                            n.inputs[0].region.cols.len,
                            // DEVICE out extent via the TYPE-SAFE `DeviceWidth` — the SAME rule as the
                            // worker's weight zero-pad and the slot reservation, so the RetileDescriptor
                            // device_size, the per-core address, and the staged buffer agree by
                            // construction (granite lm_head 49159→49664).
                            //
                            // ⭐ `for_pointwise` — the UNCONDITIONAL bump — NOT `for_output(m,·)`: the
                            // staged buffer is ONE per model (the worker binds it once; each bucket's
                            // session stages or borrows it), so its width must be m-INDEPENDENT, and the
                            // slot reservation already sizes every gemm weight at this same unconditional
                            // bump (`bump_sticks_to_splittable` on the cols, lower_subtile_tape_to_superdsc).
                            // For any weight crossing the util floor at m=1 — every dense weight — the two
                            // laws are IDENTICAL at every bucket (both bump), so those manifests are
                            // byte-identical to before. The router's `[E,hidden]` is the one real weight
                            // BELOW that floor at m=1 (E=128, k=2816 → 360K macs): its per-bucket matmul
                            // width varies (128 at m≤2, 512 at m≥3), and the pointwise floor (512) is the
                            // MAX any bucket's gemm reads — every bucket takes its own stick-prefix of the
                            // one buffer (the decode gemm reads 2 of 8 sticks; proven on-card by the
                            // shared-segment decode gemm at out=128 against the prefill's 8-stick slot).
                            DeviceWidth::for_pointwise(n.output.region.cols.len).get(),
                        ));
                        // KSPLIT fp32-merge (lm_head first-test, n>16384): the split weight ALSO needs B
                        // block-weight descriptors `[KB,N]` under reserved tids `ksplit_block_tid(b)`, matching
                        // the emit branch's block matmuls. Same gate + condition. The kernel0 descriptor loop
                        // below builds each as `[dev_out/64, KB, 64]` (in=KB) — exactly the block matmul's read
                        // (`ksplit_block_weight_kslice_offset_matches_read`). Worker gathers the K-slice bytes.
                    }
                    // Bank-staged: the gathered expert gemm's weight (inputs[2]) is an expert
                    // BANK — seg1 bank, its own retile, never through kernel0.
                    Some(2) if matches!(n.op, SubOp::ExpertMatmul { .. }) => {}
                    // Carries a gemm weight but has no SuperDSC kernel — the worklist below
                    // refuses the op itself; nothing to collect.
                    Some(2) if matches!(n.op, SubOp::AllRowsMatmul) => {}
                    // ⛔ A GEMM WEIGHT OUTSIDE THE COLLECTED SET — the router-defect class,
                    // refused at bake. A gemm whose weight never lands in the kernel0 manifest
                    // is staged by the shim's `None` arm (verbatim bytes into a stick-tiled
                    // slot): right ranking form, wrong elements, junk on the card, green on
                    // every emulator.
                    Some(pos) => {
                        let msg = format!(
                            "reroll-superdsc: op {} (t{}) carries a GEMM WEIGHT at input \
                             {pos} but the kernel-weight manifest does not collect its kind — \
                             its weight would be staged VERBATIM into a stick-tiled slot (the \
                             26b router-defect class: `A @ perm(W)`, junk on the card, invisible \
                             to the emulator). Add the kind to the kernel0 collection or give it \
                             its own staging arm in `SubOp::gemm_weight_input`",
                            n.op.name(),
                            n.output.tensor.index()
                        );
                        eprintln!("[spyre-gemm-weights] REFUSING TO BAKE: {msg}");
                        return Err(SuperDscError(msg));
                    }
                    None => {}
                }
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
                    // RESIDENT kct per-layer registration (kill-the-restickify residency): the score reads a
                    // per-layer RESIDENT Kᵀ kernel `kct_resident_tid(k_id)`. It's neither an External nor a node
                    // output, so the loops above never add it — insert it manually: map each layer's
                    // K-cache source tid k_Lv → kct_resident_tid(k_Lv), keyed under the layer-0 kct tid.
                    // REQUIRED so the launch-table guard validates kct's per-layer offset matches k/v — the
                    // executor shifts seg2 by the launch's kv_off for every op, so a kct packed out of step
                    // with k/v would SILENTLY read the wrong layer (runtime-errors-need-compile-
                    // time-checks). per_layer[k_id] was just inserted by the External loop above.
                    if let SubOp::AttnDecode { layout: kv, .. } = &n.op
                        && iters > 1
                        && let Some(k_layers) =
                            per_layer.get(&(kv.cache_tensor().index() as u32)).cloned()
                        && k_layers.len() as u32 == iters
                    {
                        let kct_tids: Vec<u32> = k_layers
                            .iter()
                            .map(|&k_lv| kct_resident_tid(k_lv))
                            .collect();
                        body_representatives[bi]
                            .push(kct_resident_tid(kv.cache_tensor().index() as u32));
                        per_layer
                            .entry(kct_resident_tid(kv.cache_tensor().index() as u32))
                            .or_insert(kct_tids);
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
                } else if seg == n_groups - 1 {
                    if first_suffix_node.is_none() {
                        first_suffix_node = Some(node.index() as u32);
                    }
                    suffix_node_ids.insert(node.index() as u32);
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
                match lower_one_node(n, ir, active_cap, rows_are_requests) {
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
    // ⭐⭐⭐ THE COMPOSITE ARGMAX — the loop-closure group, minted into the suffix HERE:
    // AFTER the walk (so the whole tail is collected and the residual pre-pass has named
    // the embedding-source slot) and BEFORE the declare pass below (so the chain's seven
    // scratch synthetics are declared in the shared plan's own order — the ⭐seg0-overrun
    // ordering law). KTIR-first by construction: `op: None, ktir: Some`; the descriptors
    // arrive through the door arm in the consumer, and the emulator binds `func` directly.
    //
    // THE BINDING CONVENTION (mint_argmax_node's, mirrored by the door arm): three
    // parameters over the graph's own slots — [logits = ir.result, table = the lm-head
    // weight, row = the first body's hidden-in, which IS the embedding-source slot]. The
    // composite runs LAST in the suffix, after every op that reads the residual buffer,
    // so its row write is WAR-safe within the step. (The next step's embedding staging
    // still overwrites the row on the host path — the executor flip that makes the
    // suffix end at the gather's row is PR3's stage 3, deliberately not this PR's.)
    if let Some(plan) = argmax_plan {
        // The tail must live wholly in the suffix: the lm-head runs once, after the last
        // layer. A chain node inside a rolled body would mean the recognition grabbed a
        // per-layer projection; the weight-set check already refuses that, and this
        // structural fact is checked where only the walk can see it.
        if let Some(&nid) = plan
            .chain_nodes
            .iter()
            .find(|nid| !suffix_node_ids.contains(nid))
        {
            return Err(SuperDscError(format!(
                "reroll-superdsc: the composite argmax's tail includes node {nid}, which the \
                 roll placed OUTSIDE the suffix — the lm-head must run once, after the last \
                 layer, never inside a rolled body"
            )));
        }
        // The row slot: the first body's hidden-in (the embedding source), whose placement
        // the residual pre-pass already aliased every body and the suffix input onto. Its
        // width must state the same hidden the table's rows do — the gather's slab and the
        // residual buffer are the same width or the write would not fit.
        let row_tid = embed_row_tid.expect(
            "the residual pre-pass named the first body's hidden-in — a loop body holds at \
             least one Compute, so the first body's hin exists",
        );
        let row_cols = ir.tensors[row_tid as usize].cols;
        if row_cols != plan.hidden {
            return Err(SuperDscError(format!(
                "reroll-superdsc: the composite argmax's row slot is [1, {row_cols}] but the \
                 lm-head table's rows are {} wide — the gather's slab and the residual buffer \
                 must be the same width",
                plan.hidden
            )));
        }
        let k = mint_argmax_node(
            plan.vocab,
            plan.hidden,
            plan.logits_tid,
            plan.table_tid,
            row_tid,
            plan.tied,
        );
        let mut e = EmittedOp::bare(k.func.name.to_string());
        e.ktir = Some(k);
        suffix.push(e);
        // The composite's gather READS the lm-head table — a weight — so the table's bank
        // joins the SUFFIX group's set (a launch binds ONE base per segment; the
        // single-member law is checked below with every other group's). Redundant with the
        // matmul's own read in every real graph, and stated anyway: the composite's
        // operands are its own fact, not an inference from its neighbors.
        if let Some(p) = bundle_layout.placements.get(&plan.table_tid)
            && matches!(p.role, SegRole::Weight)
        {
            group_weight_banks[n_groups - 1].insert(p.bank);
        }
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
    // ⛔⛔⛔ THE SHARED PLAN IS COMPLETED HERE, BEFORE ANY BUNDLE CAN SNAPSHOT IT.
    //
    // MEASURED, granite-3.1-2b on the card: main's decode bundle places rmsnorm scratch for tids
    // {448, 458, 1128}; ours placed {448, 458}. t1128 is the FINAL `model.norm` — the lm-head tail,
    // which lives in the SUFFIX bundle. Every other placement and every segment but seg0 was
    // byte-identical to main; seg0 was short by exactly one rmsnorm's five scratch buffers
    // (Sq16 4096 + Xn 4096 + Mean/Meps/Rinv 128 each = 8576 B at m=1, and 67 sticks × (3n − 2) across
    // the whole ladder).
    //
    // WHY, and it is an ORDERING fact, not a lowering one. `subtile→superdsc` declares a synthetic
    // intermediate during the TAPE WALK — main's `lower_rmsnorm_node` calls `assemble_rmsnorm`, which
    // calls `BundleLayout::synth`, for prefix, body AND suffix nodes alike — so its layout is complete
    // before the first `emit_bundle`. On this path the producer emits only a KTIR program, and
    // `assemble_rmsnorm` runs LATER, in the consumer (`ktir_groups_via_superdsc`), once per bundle.
    // `emit_bundle` then does `layout.map(bake_layout)` right after lowering ITS OWN ops, and codegen
    // emits prefix → body → suffix. So the BODY's snapshot — which is the layout the RUNTIME loads,
    // sizes its segments from and resolves every address through — held the prefix's and the body's
    // synths and none of the suffix's.
    //
    // The suffix's descriptors were built against the suffix's own later, complete snapshot, so they
    // address seg0 past the extent the runtime allocated from the body's. That is a DMA to an address
    // with no IOMMU translation behind it on the very first forward (the suffix runs in every one),
    // which the card answers with a response block carrying `status=Error`, reported as
    // `scheduler rejected submission` and then a bare `predict: sync rc=-1` naming no operand. It is
    // also why a descriptor-footprint audit beside the emitter was silent: at the moment the suffix's
    // ops were checked, the suffix's own snapshot did contain them.
    //
    // ⭐ THIS DECLARES, IT DOES NOT EMIT. `BundleLayout::synth_bytes` is first-one-wins, so running
    // the consumer over prefix → body → suffix here fixes every synthetic's offset in the SAME order
    // the tape walk would have, and the real pass in `ktir_groups_via_superdsc` finds each one already
    // declared and resolves the identical address. The symbol counter and the fp8-quantize set are
    // throwaways because only the layout is wanted; the descriptors are dropped.
    // The SAME four facts the real consumer pass is handed — read once here, off the graph's own
    // `AttnDecode` nodes and this walk's own parameters. See [`attn_bundle_params`].
    let attn_params = attn_bundle_params(ir, rows_are_requests, one_request_decode, chunk_pages)?;
    // The declare pass walks prefix → EVERY body → suffix, the same order the tape walk
    // declared them in, so every synthetic's first-one-wins offset is fixed identically.
    let body_op_lists: Vec<&Vec<EmittedOp>> = bodies.iter().map(|b| &b.ops).collect();
    for ops in std::iter::once(&prefix)
        .chain(body_op_lists.iter().copied())
        .chain(std::iter::once(&suffix))
    {
        let mut declare_syms: i64 = 0;
        let mut declare_fp8: std::collections::HashSet<String> = std::collections::HashSet::new();
        for e in ops.iter() {
            let Some(k) = e.ktir.as_ref() else { continue };
            crate::ktir_superdsc_door::lower(
                k,
                &mut declare_syms,
                Some(&bundle_layout),
                &mut declare_fp8,
                attn_params,
            )
            .map_err(|err| {
                SuperDscError(format!(
                    "{}: declaring the shared plan's synthetics: {}",
                    e.op_name, err.message
                ))
            })?;
        }
    }
    // Device re-tile manifest: each matmul kernel weight (layer 0) + its per-layer
    // copies → a RetileDescriptor built SOLELY from the DeviceTileLayout witness (the
    // SAME source per_core_addr uses for the stride), so the shim's host re-tile and the
    // on-card per-core address cannot diverge. KERNEL layout = [in,out] sticked on out.
    // fp8 W8A8 weights (`input[1]` of an arity-3 MatmulTile) stage 1-byte / 128-elem stick (SEN143_FP8);
    // dense weights stay fp16 2-byte / 64-stick. The shim reads `word_length` from this descriptor to
    // size the H2D re-tile, so an fp8 weight MUST carry the fp8 descriptor or it is staged as 2-byte.
    let fp8_weight_tids: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3)
        .map(|n| n.inputs[1].tensor.index() as u32)
        .collect();
    for (w_tid, in_k, out_n) in &kernel0 {
        let desc = if fp8_weight_tids.contains(w_tid) {
            // fp8 W8A8 weight = the AIU matmulfp8 PACKED tile. Each 128-byte stick holds 64 N-cols each
            // carrying its 2 K-bytes BYTE-ADJACENT — the matmulfp8 in-fold (gen_fp8_kernel_in_fold) reads the
            // 2-pack as "2 fp8 per fp16-width slot, CONTIGUOUS". Device order (outer→inner):
            // [N/64 n-sticks, K/2 k-pairs, 64 n-inner, 2 k-inner]; device[n_stick][k_outer][n_in][k_inner] =
            // host[64·n_stick + 2N·k_outer + n_in + N·k_inner] = weight[2·k_outer+k_inner][64·n_stick+n_in].
            // The [.,.,2,64] order (2 K-rows as two separate 64-N blocks) and a FLAT 128-stick layout BOTH
            // scramble the weights → garbage; only this [.,.,64,2] order runs coherent (on-card 2026-07-16,
            // `scr chat` → "Paris"). (K even + N%64==0 hold for every granite fp8 proj.)
            // ⭐ THE fp8 K-PAD (`work::fp8_k_pad`) — the whole-stick law, ONE authority: the
            // activation quantize's tail, the worker's staged rebuild, the placement's reserved
            // rows, and this packed retile's K all take the SAME pad. A dense fp8 weight whose k
            // is not a whole 128-elem stick (gemma-4-26b's dense GeGLU DOWN: k=2112 = 16.5
            // sticks) widens to 2176 — the extra 64 K-rows are ZERO on the host, real staged
            // bytes, inert to the contraction. Without the pad the retile enumerates n·k elements
            // against the worker's n·k_pad staged rebuild, and the matmul's k_pad-wide tile read
            // overruns the staged buffer. A no-op at every 128-aligned k (granite, g8b, g12b —
            // byte-identical staging).
            let k = fp8_k_pad(*in_k) as u64;
            let n = *out_n as u64;
            if !k.is_multiple_of(2) || !n.is_multiple_of(64) {
                return Err(SuperDscError(format!(
                    "fp8 W8A8 weight t{w_tid}: packed tile needs K({k})%2==0 and N({n})%64==0"
                )));
            }
            RetileDescriptor {
                device_size: vec![n / 64, k / 2, 64, 2],
                // DISK ORDER, like the fp16 tile below. The worker no longer transposes, so this map
                // reads the `[out, in]` buffer safetensors stores. Converting it is mechanical: a term
                // that stepped an IN index by `x` was `x*n` against `[in, out]` and becomes `x*1`; a
                // term that stepped an OUT index by `y` was `y*1` and becomes `y*k`. So
                // `[64, 2n, 1, n]` → `[64k, 2, k, 1]`, which resolves `host[o*k + i]` for every
                // coordinate (checked exhaustively against the transposed map's `host[i*n + o]`).
                //
                // MISSING THIS BRANCH is what made granite-3.1-8b emit garbage: it is fp8, so its
                // GEMM weights come through here and not the fp16 path, and they were still being read
                // as though something had transposed them.
                stride_map: vec![64 * k, 2, k, 1],
                stick_size: Fp8::ELEMS_PER_STICK,
                word_length: Fp8::WORD_LENGTH,
            }
        } else {
            let tile = DeviceTileLayout::<Fp16>::new(
                &["in", "out"],
                "out",
                &[*in_k as u64, *out_n as u64],
            )?;
            RetileDescriptor {
                device_size: tile.device_size(),
                // DISK ORDER: the worker binds a GEMM weight in the `[out, in]` orientation
                // safetensors stores it, so the re-tile reads it there rather than from a transposed
                // copy. Same elements, one fewer pass over the model at load.
                stride_map: tile.stride_map_disk_order(),
                stick_size: Fp16::ELEMS_PER_STICK,
                word_length: Fp16::WORD_LENGTH,
            }
        };
        // ⛔⛔⛔ THE THREE WIDTH LAWS, HELD EQUAL AT BAKE (the staging-consistency guard, W29).
        //
        // For every gemm weight the manifest declares — the entry AND each of its per-layer
        // copies — three independent laws must name the SAME device width, and each is computed
        // here from the bake's own data:
        //   1. THE MANIFEST'S: `out_n`, the width this loop declares in the RetileDescriptor
        //      (`DeviceWidth::for_pointwise` over the gemm's output cols, at the collection).
        //   2. THE WORKER'S: `DeviceWidth::for_pointwise` over the weight TENSOR's own cols —
        //      the law `spyre_load.rs`'s N-pad applies to the same source (the wiring's
        //      `[rows, cols]` pair is generated from these shapes). A gemm whose output region
        //      names a different width than the weight tensor it reads (a sliced output, a
        //      manifest/wiring shape desync) stages a buffer the manifest's tile does not
        //      cover — the shim's `prod(device_size)` byte check would fire at LOAD, on the
        //      card, after the build said yes.
        //   3. THE SLOT RESERVATION'S: the bytes this bake placed for the tensor
        //      (`compute_bundle_layout`'s `rows · bump(cols) · word_length`) must COVER
        //      the manifest's `prod(device_size) · word_length` — at least, not exactly:
        //      the reservation pads the weight's innermost stick dim to the DEVICE's
        //      whole-stick footprint (fp8 is a 128-elem stick — the 26b's dense gate/up
        //      at n=2112 = 16.5 sticks reserves 2176 cols), and over-reservation is the
        //      documented safe direction (the retile's staging guard reads the WORKER's
        //      bytes; `lower_subtile_tape_to_superdsc`'s own fp8 K-pad comment). UNDER-
        //      reservation is what aliases — the t35 class the fp8 K-pad rows law fixed
        //      (access 6,127,616 B vs placement 5,947,392 B). The K the manifest contracts
        //      (fp8 weights take `fp8_k_pad`, the same whole-stick law the reservation's rows
        //      take) times the same width. They were held together only by the "every weight
        //      crosses the m=1 util floor" assumption, which the router broke silently: its
        //      slot was reserved at the pointwise floor (512) while the excluded manifest
        //      staged the checkpoint's 128-wide bytes verbatim — over-addressed the slot by
        //      the retile, wrong experts, junk. The check is pure compile-time arithmetic on
        //      the bake's own placements and descriptors; it needs no emulator run (and the
        //      emulator could not witness it — see the gate at the collection above).
        {
            let wl = desc.word_length as u64;
            let manifest_bytes = desc.device_size.iter().product::<u64>() * wl;
            let mut tids = std::iter::once(*w_tid)
                .chain(per_layer.get(w_tid).into_iter().flatten().copied())
                .collect::<Vec<u32>>();
            tids.sort_unstable();
            tids.dedup();
            for t in tids {
                // The op that reads this weight, by name — the refusal's own provenance.
                let op_name = match ir
                    .nodes
                    .iter()
                    .find(|n| n.inputs.len() > 1 && n.inputs[1].tensor.index() as u32 == t)
                {
                    Some(n) => n.op.name(),
                    None => "gemm",
                };
                let worker_w = DeviceWidth::for_pointwise(ir.tensors[t as usize].cols).get();
                let slot = bundle_layout.placements.get(&t);
                let slot_desc = match slot {
                    Some(p) => format!("{} bytes (expected at least {manifest_bytes})", p.size),
                    None => "NO PLACEMENT".to_string(),
                };
                let refuse = |what: &str| -> SuperDscError {
                    let msg = format!(
                        "reroll-superdsc: gemm weight t{t} of {op_name}: the staging width \
                         laws disagree — {what}. The manifest declares {out_n}, the worker's \
                         N-pad law says {worker_w}, the slot reservation holds {slot_desc}. A \
                         divergence stages a buffer the device tile does not cover (or \
                         under-covers): the router-defect class, wrong elements on the card, \
                         invisible to the emulator. All three laws must be \
                         `DeviceWidth::for_pointwise` over the same logical width, and the \
                         reservation must hold at least prod(device_size)·word_length bytes"
                    );
                    eprintln!("[spyre-gemm-weights] REFUSING TO BAKE: {msg}");
                    SuperDscError(msg)
                };
                if worker_w != *out_n {
                    return Err(refuse(
                        "the gemm's output width and the weight tensor's own cols disagree",
                    ));
                }
                match slot {
                    None => {
                        return Err(refuse("the bake reserved no placement for the weight"));
                    }
                    Some(p) if p.size < manifest_bytes => {
                        return Err(refuse(
                            "the placement under-reserves the manifest's tile (aliases the next placement)",
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        bundle_layout.kernel_weights.insert(*w_tid, desc.clone());
        if let Some(layers) = per_layer.get(w_tid) {
            for &t in layers {
                bundle_layout.kernel_weights.insert(t, desc.clone());
            }
        }
    }
    // ⭐⭐⭐ THE HEAD-MAJOR PERMUTED COPIES' MANIFEST + PER-LAYER FAMILY — this walk's half of the
    //    mint `compute_bundle_layout` performed (it placed every layer's copy, one after its
    //    companion). Here the copies join the two registries the ROLLED body's runtime reads:
    //    the retile manifest — the copy's OWN walk, because the permutation is a STAGING walk,
    //    not a byte transform: the worker stages the SAME source bytes through it — and the
    //    per-layer family map (kct-style), so the executor's one-base-per-segment advance
    //    reaches layer v's copy exactly as it reaches layer v's companion, which the stride
    //    guard below then proves against the placements.
    let oproj = oproj_edges(ir)?;
    if oproj.geometry_applies {
        let canonical = oproj.edges[0].w;
        // The o_proj's own (k, n) — read from the manifest entry the walk collected for the o
        // matmul, the same program-tile pair the permuted walk is derived from.
        let Some(&(_, in_k, out_n)) = kernel0.iter().find(|(w, ..)| *w == canonical) else {
            return Err(SuperDscError(format!(
                "the head-major o_proj weight t{canonical} has no kernel manifest entry — the walk \
                 collects one for every body matmul, so its absence means the o matmul never \
                 lowered and the handoff would have nothing to swap",
            )));
        };
        let (k, n) = (in_k as u64, out_n as u64);
        let hd = oproj.edges[0].hd as u64;
        let nslab = hd / POOL_STICK as u64;
        let nqh = k / hd;
        if !k.is_multiple_of(2)
            || !n.is_multiple_of(64)
            || !hd.is_multiple_of(POOL_STICK as u64)
            || k != nqh * hd
        {
            return Err(SuperDscError(format!(
                "head-major o_proj t{canonical}: the permuted packed tile needs K({k})%2==0, \
                 N({n})%64==0, hd({hd})%64==0 and K == nqh·hd = {nqh}·{hd}",
            )));
        }
        // ⭐ THE 6-AXIS PERMUTED WALK — the fp8 packed tile's k-enumeration REFINED into
        // (slab, head, half-stick pair) so the device's contraction axis runs HEAD-MAJOR. The
        // device's logical k position is `s·nqh·64 + h·64 + 2·hp + k_in` — exactly the order the
        // slab-major finalize writes `out` in — and the host fetch for that position is
        // `host[n, h·hd + s·64 + 2·hp + k_in]`, the TRUE feature the weight column owns:
        //   device_size = [n/64, nslab, nqh, 64/2, 64, 2]
        //   stride_map  = [64·k, 64, hd, 2, k, 1]
        // which sums to `host[(64·n_stick + n_in)·k + (h·hd + s·64 + 2·hp + k_in)]` — the same
        // `[out, in]` disk-order law the fp8 tile above states, with the k axis permuted and
        // NOTHING else moved. The kernel still reads the staged buffer as the SAME
        // `[n/64, k/2, 64, 2]` packed tile (its k_pair is `s·nqh·32 + h·32 + hp`, its inner
        // stick still 64 n × 2 k), so the device program is untouched — only the staging walk
        // knows. At `nslab == 1` the walk degenerates to the fp8 tile's identity (s = 0:
        // `h·64 + 2·hp + k_in` on both sides at hd == 64), which is why the mint gates on a
        // multi-slab head dim.
        let perm_desc = RetileDescriptor {
            device_size: vec![n / 64, nslab, nqh, POOL_STICK as u64 / 2, 64, 2],
            stride_map: vec![64 * k, POOL_STICK as u64, hd, 2, k, 1],
            stick_size: Fp8::ELEMS_PER_STICK,
            word_length: Fp8::WORD_LENGTH,
        };
        let perms: Vec<u32> = oproj
            .edges
            .iter()
            .map(|e| oproj_headmajor_tid(e.w))
            .collect();
        for &p in &perms {
            bundle_layout.kernel_weights.insert(p, perm_desc.clone());
        }
        // The family: representative (layer 0's perm) → every layer's perm, in layer order —
        // the same law the kct registration above states for the resident Kᵀ.
        per_layer
            .entry(oproj_headmajor_tid(canonical))
            .or_insert(perms);
    }
    let seg3 = SegRole::Intermediate.segment();
    let synth_high = bundle_layout.synth.borrow().next;
    if synth_high > bundle_layout.segment_bytes[seg3] {
        bundle_layout.segment_bytes[seg3] = synth_high;
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
    //    weight tables and the KV shift from c's per-layer KV tables. A body with no per-layer
    //    weight/KV gets (0, 0, 0) — nothing to shift. The proof below then holds EVERY tensor of
    //    the class to the entry, so which table generated it cannot matter; a class whose tables
    //    DISAGREE is refused here with both readings named.
    let mut launch_w: Vec<(u8, u64)> = Vec::with_capacity(total_layers);
    let mut launch_kv: Vec<u64> = Vec::with_capacity(total_layers);
    for &(body, iter) in &launches {
        let mut w_entry: Option<(u8, u64)> = None;
        let mut kv_entry: Option<u64> = None;
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
            } else if p0.segment == kv_seg {
                let entry = pj.offset.wrapping_sub(p0.offset);
                if let Some(prev) = kv_entry
                    && prev != entry
                {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: body {body}'s per-layer KV tensors DISAGREE about layer \
                         {iter}'s launch shift ({prev} vs {entry} B, generated from t{t0}) — k, v \
                         and Kᵀ must advance together or the score reads one layer's keys against \
                         another's values"
                    )));
                }
                kv_entry = Some(entry);
            }
        }
        launch_w.push(w_entry.unwrap_or((0, 0)));
        launch_kv.push(kv_entry.unwrap_or(0));
    }
    // ── PROVE, EVERY TENSOR, EVERY LAYER. The launch addresses
    //    `(bank: L.w_bank, offset: L.w_off + placement(t₀).offset)` — hold each per-layer weight
    //    to the placement its layer actually got, and each per-layer KV tensor likewise. This is
    //    the check the generator cannot make: generation read ONE tensor per segment per launch;
    //    this holds ALL of them.
    for (li, &(body, iter)) in launches.iter().enumerate() {
        let (wb, wo) = launch_w[li];
        let ko = launch_kv[li];
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
                if pj.bank as u8 != wb || pj.offset != wo.wrapping_add(p0.offset) {
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
            } else if p0.segment == kv_seg && pj.offset != ko.wrapping_add(p0.offset) {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: per-layer KV tensor t{tid} (layer {layer} of body {body}) is \
                     placed at offset {off}, but the launch table addresses it at {want} (shift \
                     {ko} + layer-0 offset {base}). The score would read one layer's keys against \
                     another's values.",
                    tid = tids[iter as usize],
                    layer = launch_layer[li],
                    off = pj.offset,
                    want = ko.wrapping_add(p0.offset),
                    ko = ko,
                    base = p0.offset,
                    body = body,
                )));
            }
        }
    }
    // The PAGE stride: one pool page covers every layer of every class, so the stride is the
    // extent of the packed per-layer KV — the max end over every class's LAST layer placement
    // (classes are packed contiguously; per-class strides differ, which is exactly why this is a
    // max and not a count × stride). 0 when there is no paged KV.
    let kv_page_stride = per_layer
        .values()
        .filter_map(|tids| {
            let last = tids.last()?;
            let p = bundle_layout.placements.get(last)?;
            (p.segment == kv_seg).then(|| p.offset + p.size)
        })
        .max()
        .unwrap_or(0);
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
    // ── DISCOVERY dump (SCRATCHY_SUPERDSC_SEGDUMP) ── which segment/tensor drives the
    //    footprint. Emit runs at cargo-build (AoT bake), so this lands in the build log.
    if std::env::var_os("SCRATCHY_SUPERDSC_SEGDUMP").is_some() {
        let sb = &bundle_layout.segment_bytes;
        let tot: u64 = sb.iter().sum();
        eprintln!(
            "[SEGDUMP] layers={total_layers} total={:.3}GB segs(GB)=[{}]",
            tot as f64 / 1e9,
            sb.iter()
                .map(|b| format!("{:.3}", *b as f64 / 1e9))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let mut pls: Vec<(&u32, &TensorPlacement)> = bundle_layout.placements.iter().collect();
        pls.sort_by_key(|(_, p)| std::cmp::Reverse(p.size));
        for (tid, p) in pls.into_iter().take(15) {
            eprintln!(
                "[SEGDUMP]   t{tid} seg{} role={:?} size={:.4}GB ({} B)",
                p.segment,
                p.role,
                p.size as f64 / 1e9,
                p.size,
            );
        }
        let syn = bundle_layout.synth.borrow();
        let mut szs: Vec<(&String, &u64)> = syn.sizes.iter().collect();
        szs.sort_by_key(|(_, s)| std::cmp::Reverse(**s));
        for (name, s) in szs.into_iter().take(10) {
            eprintln!(
                "[SEGDUMP]   synth {name} size={:.4}GB ({} B)",
                *s as f64 / 1e9,
                *s
            );
        }
    }
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
        kv_page_stride,
        prefix_weight_bank,
        suffix_weight_bank,
        kv_request_stride,
        suffix_in_tid,
        attn_params,
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
        // granite's own geometry through the same mint the config path uses, scaled down: the
        // recognizer reads only `scale`, so the geometry is inert — it just has to exist.
        let geom = scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
            scratchy_subtile::model_geometry::QueryHeads::new(2),
            scratchy_subtile::model_geometry::KvHeads::new(1),
            scratchy_subtile::model_geometry::HeadDim::new(8),
        )
        .expect("2 query heads over 1 kv head, head_dim 8, mints");
        let hd = scratchy_subtile::model_geometry::HeadDim::new(8);
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
                    hd,
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
                    geom,
                    scale,
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
        let geom = scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
            scratchy_subtile::model_geometry::QueryHeads::new(2),
            scratchy_subtile::model_geometry::KvHeads::new(1),
            scratchy_subtile::model_geometry::HeadDim::new(8),
        )
        .expect("2 query heads over 1 kv head, head_dim 8, mints");
        let hd = scratchy_subtile::model_geometry::HeadDim::new(8);
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
                        hd,
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
                        geom,
                        4.0,
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

    /// ⭐ THE UNIT SCALE IS THE FOLD'S OTHER WAY TO A MULTIPLIER-LESS PROGRAM. The gemma-4 arch
    /// defaults `attention_multiplier` to 1.0, the fold correctly multiplies nothing (`√1.0`),
    /// and the splice mints at 1.0 — which the triton→ktir pipeline folds away entirely, so the
    /// program TRUTHFULLY states no multiplier and lands in the door's skip arm. The bundle fact
    /// must state that case too, or the first unit-scale model is a build error at the very door
    /// whose skip arm is its correct lowering (tiny26's bake, 2026-10-10).
    #[test]
    fn a_unit_scale_attention_carries_the_door_fact() {
        let ir = attn_graph(1.0, false, false, false, false, false);
        let p = attn_bundle_params(&ir, false, false, None)
            .expect("the bundle states its attention params")
            .expect("the graph has an attention node");
        assert!(
            p.scale_in_weights,
            "a unit scale is carried by the weights trivially — √1.0 — so the program states \
             no multiplier and the door must take the skip arm"
        );
    }

    /// The fold's own case keeps the same fact: √scale went into W_q/W_k at load, so the node is
    /// rewritten to 1.0 where it is lowered and the program states no multiplier — at an
    /// UP-multiplier (√4.0 = 2.0), the f16 floor gate's permitted class.
    #[test]
    fn a_folded_attention_carries_the_door_fact() {
        let ir = attn_graph(4.0, false, false, false, false, false);
        let p = attn_bundle_params(&ir, false, false, None)
            .expect("the bundle states its attention params")
            .expect("the graph has an attention node");
        assert!(
            p.scale_in_weights,
            "the fold puts √scale into W_q/W_k — no device multiply is left to state"
        );
    }

    /// A scale the fold refused STRUCTURALLY (here: an unroped query — the gate permits the
    /// up-multiplier, so the refusal is the chain's own) must state its multiplier in the
    /// program — the weights carry none of it, so the door fact stays false.
    #[test]
    fn an_unfolded_real_scale_states_no_door_fact() {
        let ir = attn_graph(4.0, true, false, false, false, false);
        let p = attn_bundle_params(&ir, false, false, None)
            .expect("the bundle states its attention params")
            .expect("the graph has an attention node");
        assert!(
            !p.scale_in_weights,
            "an unfolded node at a real scale must have its multiplier stated by the program"
        );
    }

    /// Two INDEPENDENT attention chains over one hidden — the disagreement fixture: each chain is
    /// `attn_graph`'s own (q projection → rope, k projection → rope_append writing its own paged
    /// cache, attention), with its own sources throughout, so the only facts the two
    /// `AttnDecode` nodes differ on are the ones handed in (the scales, and whether the second
    /// chain's query is roped — its foldability lever).
    fn two_attn_graph(a_scale: f32, b_scale: f32, b_unroped: bool) -> SubtileIR {
        let geom = scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
            scratchy_subtile::model_geometry::QueryHeads::new(2),
            scratchy_subtile::model_geometry::KvHeads::new(1),
            scratchy_subtile::model_geometry::HeadDim::new(8),
        )
        .expect("2 query heads over 1 kv head, head_dim 8, mints");
        let hd = scratchy_subtile::model_geometry::HeadDim::new(8);
        let shape = |rows: u32, cols: u32| scratchy_subtile::subtile_ir::SourceShape { rows, cols };
        let sources = vec![
            shape(1, 8),  // 0: x (hidden 8)
            shape(8, 16), // 1: W_q (chain A)
            shape(8, 8),  // 2: W_k (chain A)
            shape(1, 8),  // 3: v (chain A)
            shape(32, 8), // 4: prefix_k cache (A)
            shape(32, 8), // 5: prefix_v cache (A)
            shape(1, 16), // 6: cos (A)
            shape(1, 16), // 7: sin (A)
            shape(8, 16), // 8: W_q (chain B)
            shape(8, 8),  // 9: W_k (chain B)
            shape(1, 8),  // 10: v (chain B)
            shape(32, 8), // 11: prefix_k cache (B)
            shape(32, 8), // 12: prefix_v cache (B)
            shape(1, 16), // 13: cos (B)
            shape(1, 16), // 14: sin (B)
        ];
        let b_q_ref = if b_unroped {
            InputRef::Op(5)
        } else {
            InputRef::Op(6)
        };
        let ops = vec![
            // 0: q_a = x · W_q_a; 1: q_a_rot = rope_rotate(q_a, cos_a, sin_a)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            OpDesc {
                op: SubOp::rope_rotate(hd),
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(6), InputRef::Ext(7)],
            },
            // 2: k_a = x · W_k_a; 3: k_a_rot = rope_append(k_a, cos_a, sin_a, v_a, caches)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 8,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(2)],
            },
            OpDesc {
                op: SubOp::rope_append(
                    hd,
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
            // 4: attn_a = AttnDecode(q_a_rot, prefix_k_a, prefix_v_a, k_a_rot, v_a) at a_scale
            OpDesc {
                op: SubOp::attn_decode(
                    geom,
                    a_scale,
                    33,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                ),
                m: 1,
                inputs: vec![
                    InputRef::Op(1),
                    InputRef::Ext(4),
                    InputRef::Ext(5),
                    InputRef::Op(3),
                    InputRef::Ext(3),
                ],
            },
            // 5: q_b = x · W_q_b; 6: q_b_rot = rope_rotate(q_b, cos_b, sin_b)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(8)],
            },
            OpDesc {
                op: SubOp::rope_rotate(hd),
                m: 1,
                inputs: vec![InputRef::Op(5), InputRef::Ext(13), InputRef::Ext(14)],
            },
            // 7: k_b = x · W_k_b; 8: k_b_rot = rope_append(k_b, cos_b, sin_b, v_b, caches)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 8,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(9)],
            },
            OpDesc {
                op: SubOp::rope_append(
                    hd,
                    0,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                    scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                ),
                m: 1,
                inputs: vec![
                    InputRef::Op(7),
                    InputRef::Ext(13),
                    InputRef::Ext(14),
                    InputRef::Ext(10),
                    InputRef::Ext(11),
                    InputRef::Ext(12),
                ],
            },
            // 9: attn_b = AttnDecode(b_q_ref, prefix_k_b, prefix_v_b, k_b_rot, v_b) at b_scale
            OpDesc {
                op: SubOp::attn_decode(
                    geom,
                    b_scale,
                    33,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                ),
                m: 1,
                inputs: vec![
                    b_q_ref,
                    InputRef::Ext(11),
                    InputRef::Ext(12),
                    InputRef::Op(8),
                    InputRef::Ext(10),
                ],
            },
        ];
        let input = LoweringInput {
            sources,
            ops,
            result: 9,
            exported: None,
        };
        let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
        scratchy_subtile::subtile_ir::lower_region(&input, nb)
    }

    /// ⭐ THE DISAGREEMENT, WIDENED: one graph whose attentions disagree about WHERE the score
    /// scale lives — one carries it in its staged weights (here: a unit multiplier), the other
    /// states a device score multiply — is a build error at the recognition, naming both nodes,
    /// because the door takes ONE such fact per bundle and either choice mislowers one layer.
    #[test]
    fn a_graph_mixing_in_weights_and_stated_scales_refuses_to_choose() {
        let ir = two_attn_graph(1.0, 4.0, true);
        let err = attn_bundle_params(&ir, false, false, None).expect_err("a mixed graph refuses");
        assert!(
            err.0
                .contains("carries its attention scale in its staged weights"),
            "the refusal names the in-weights side: {err:?}"
        );
        assert!(
            err.0.contains("states a device score multiply"),
            "the refusal names the stated side: {err:?}"
        );
    }

    /// Two attention chains at DIFFERENT geometries — hd 128 beside hd 256, the gemma-4
    /// sliding/global shape scaled to fixture size — each closed by an fp8 W8A8 o projection
    /// (`attn → o_proj`, the one clean consumer the edge walk demands). Every term of
    /// [`OprojEdges::geometry_applies`] holds here EXCEPT the one-geometry term: both head
    /// dims are multi-slab (hd > POOL_STICK), both o matmuls are fp8, and the `(hd, k)`
    /// pairs disagree (256/512 vs 128/256 — hd AND k both, like the real arch). This is
    /// the exact boundary the uniformity term draws, stated as a graph.
    fn two_class_fp8_oproj_graph() -> SubtileIR {
        let geom_a = scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
            scratchy_subtile::model_geometry::QueryHeads::new(2),
            scratchy_subtile::model_geometry::KvHeads::new(1),
            scratchy_subtile::model_geometry::HeadDim::new(128),
        )
        .expect("2 query heads over 1 kv head, head_dim 128, mints");
        let geom_b = scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
            scratchy_subtile::model_geometry::QueryHeads::new(2),
            scratchy_subtile::model_geometry::KvHeads::new(1),
            scratchy_subtile::model_geometry::HeadDim::new(256),
        )
        .expect("2 query heads over 1 kv head, head_dim 256, mints");
        let hd_a = scratchy_subtile::model_geometry::HeadDim::new(128);
        let hd_b = scratchy_subtile::model_geometry::HeadDim::new(256);
        let shape = |rows: u32, cols: u32| scratchy_subtile::subtile_ir::SourceShape { rows, cols };
        let sources = vec![
            shape(1, 8),    // 0: x (hidden 8)
            shape(8, 256),  // 1: W_q_a — q width 2·128
            shape(8, 128),  // 2: W_k_a — kv width 1·128
            shape(1, 128),  // 3: v_a
            shape(32, 128), // 4: prefix_k_a
            shape(32, 128), // 5: prefix_v_a
            shape(1, 256),  // 6: cos_a
            shape(1, 256),  // 7: sin_a
            shape(256, 16), // 8: W_o_a — fp8, k = 2·128
            shape(1, 16),   // 9: o_scale_a
            shape(8, 512),  // 10: W_q_b — q width 2·256
            shape(8, 256),  // 11: W_k_b — kv width 1·256
            shape(1, 256),  // 12: v_b
            shape(32, 256), // 13: prefix_k_b
            shape(32, 256), // 14: prefix_v_b
            shape(1, 512),  // 15: cos_b
            shape(1, 512),  // 16: sin_b
            shape(512, 16), // 17: W_o_b — fp8, k = 2·256
            shape(1, 16),   // 18: o_scale_b
        ];
        let rope = |hd, k_op: InputRef, cos: InputRef, sin: InputRef| OpDesc {
            op: SubOp::rope_rotate(hd),
            m: 1,
            inputs: vec![k_op, cos, sin],
        };
        let kv_append =
            |hd, k_op: InputRef, cos: usize, sin: usize, v: usize, pk: usize, pv: usize| OpDesc {
                op: SubOp::rope_append(
                    hd,
                    0,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                    scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                ),
                m: 1,
                inputs: vec![
                    k_op,
                    InputRef::Ext(cos),
                    InputRef::Ext(sin),
                    InputRef::Ext(v),
                    InputRef::Ext(pk),
                    InputRef::Ext(pv),
                ],
            };
        let attn = |geom, q: InputRef, pk: usize, pv: usize, new_k: InputRef, v: usize| OpDesc {
            op: SubOp::attn_decode(
                geom,
                0.015625,
                33,
                scratchy_subtile::subtile_ir::AttnMask::Causal,
            ),
            m: 1,
            inputs: vec![
                q,
                InputRef::Ext(pk),
                InputRef::Ext(pv),
                new_k,
                InputRef::Ext(v),
            ],
        };
        let ops = vec![
            // 0: q_a; 1: q_a_rot; 2: k_a; 3: k_a_append; 4: attn_a; 5: o_a (fp8 W8A8)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 256,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            rope(hd_a, InputRef::Op(0), InputRef::Ext(6), InputRef::Ext(7)),
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 128,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(2)],
            },
            kv_append(hd_a, InputRef::Op(2), 6, 7, 3, 4, 5),
            attn(geom_a, InputRef::Op(1), 4, 5, InputRef::Op(3), 3),
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Fp8Dynamic,
                },
                m: 1,
                inputs: vec![InputRef::Op(4), InputRef::Ext(8), InputRef::Ext(9)],
            },
            // 6: q_b; 7: q_b_rot; 8: k_b; 9: k_b_append; 10: attn_b; 11: o_b (fp8 W8A8)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 512,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(10)],
            },
            rope(hd_b, InputRef::Op(6), InputRef::Ext(15), InputRef::Ext(16)),
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 256,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(11)],
            },
            kv_append(hd_b, InputRef::Op(8), 15, 16, 12, 13, 14),
            attn(geom_b, InputRef::Op(7), 13, 14, InputRef::Op(9), 12),
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Fp8Dynamic,
                },
                m: 1,
                inputs: vec![InputRef::Op(10), InputRef::Ext(17), InputRef::Ext(18)],
            },
        ];
        let input = LoweringInput {
            sources,
            ops,
            result: 11,
            exported: None,
        };
        let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
        scratchy_subtile::subtile_ir::lower_region(&input, nb)
    }

    /// ⭐ THE UNIFORMITY TERM, PINNED: a hybrid-geometry model — multi-slab, all-fp8, classes
    /// disagreeing on `(hd, k)` — mints NO head-major handoff at the one-request rung. The
    /// handoff's permuted walk, swap key, and per-layer family are all minted from ONE class
    /// (the first edge, the rolled body's layer-0 representative), so a bundle-wide handoff
    /// over disagreeing classes would slab-major every other class's finalize against an
    /// un-permuted weight — half a restructure, fluent and wrong, card-only (the emulator
    /// executes the KTIR program, not this emission). A mixed model therefore keeps the
    /// SHIPPED token-stream finalize, byte-for-byte the pre-restructure emission — the fix
    /// that made gemma-4-12b coherent on the card (build-6, 2026-10-14: sliding hd 256 k 4096
    /// beside global hd 512 k 8192 garbaged under the single-class handoff; uniform models
    /// like granite-8b are unchanged and still mint it). If this assertion ever fails, a
    /// per-class walk and swap key landed without this term moving with it.
    #[test]
    fn a_hybrid_geometry_model_keeps_the_shipped_finalize() {
        let ir = two_class_fp8_oproj_graph();
        let p = attn_bundle_params(&ir, false, true, None)
            .expect("a mixed-geometry model is not a refusal — it keeps the shipped emission")
            .expect("the graph has attention nodes");
        assert!(
            !p.rows_are_requests,
            "the one-request rung keeps the shipped row kind"
        );
        assert!(
            !p.headmajor_handoff,
            "one handoff is one geometry: a model whose o_proj classes disagree on (hd, k) \
             keeps the shipped token-stream finalize"
        );
        assert_eq!(
            p.oproj_wtid,
            Some(8),
            "the canonical o_proj stays stated — a graph fact the door reads only under the \
             handoff, so the mixed model's door calls are the shipped emission's own"
        );
    }

    /// The UNIFORM twin of [`two_class_fp8_oproj_graph`]: ONE attention class, multi-slab
    /// (hd 128 > `POOL_STICK`), fp8 W8A8 o projection closed by exactly one matmul —
    /// granite-8b's geometry at fixture size, and the exact shape every term of
    /// [`OprojEdges::geometry_applies`] demands.
    fn uniform_fp8_oproj_graph() -> SubtileIR {
        let geom = scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
            scratchy_subtile::model_geometry::QueryHeads::new(2),
            scratchy_subtile::model_geometry::KvHeads::new(1),
            scratchy_subtile::model_geometry::HeadDim::new(128),
        )
        .expect("2 query heads over 1 kv head, head_dim 128, mints");
        let hd = scratchy_subtile::model_geometry::HeadDim::new(128);
        let shape = |rows: u32, cols: u32| scratchy_subtile::subtile_ir::SourceShape { rows, cols };
        let sources = vec![
            shape(1, 8),    // 0: x (hidden 8)
            shape(8, 256),  // 1: W_q — q width 2·128
            shape(8, 128),  // 2: W_k — kv width 1·128
            shape(1, 128),  // 3: v
            shape(32, 128), // 4: prefix_k cache
            shape(32, 128), // 5: prefix_v cache
            shape(1, 256),  // 6: cos (q side — one per q head pair)
            shape(1, 256),  // 7: sin (q side)
            shape(256, 16), // 8: W_o — fp8, k = 2·128
            shape(1, 16),   // 9: o_scale
            shape(1, 128),  // 10: cos (k side — kv width 1·128)
            shape(1, 128),  // 11: sin (k side)
        ];
        let ops = vec![
            // 0: q = x · W_q; 1: q_rot = rope_rotate(q, cos, sin)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 256,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            OpDesc {
                op: SubOp::rope_rotate(hd),
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(6), InputRef::Ext(7)],
            },
            // 2: k = x · W_k; 3: k_rot = rope_append(k, cos_k, sin_k, v, caches)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 128,
                    weight: GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(2)],
            },
            OpDesc {
                op: SubOp::rope_append(
                    hd,
                    0,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                    scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                ),
                m: 1,
                inputs: vec![
                    InputRef::Op(2),
                    InputRef::Ext(10),
                    InputRef::Ext(11),
                    InputRef::Ext(3),
                    InputRef::Ext(4),
                    InputRef::Ext(5),
                ],
            },
            // 4: attn = AttnDecode(q_rot, prefix_k, prefix_v, k_rot, v)
            OpDesc {
                op: SubOp::attn_decode(
                    geom,
                    0.015625,
                    33,
                    scratchy_subtile::subtile_ir::AttnMask::Causal,
                ),
                m: 1,
                inputs: vec![
                    InputRef::Op(1),
                    InputRef::Ext(4),
                    InputRef::Ext(5),
                    InputRef::Op(3),
                    InputRef::Ext(3),
                ],
            },
            // 5: out = attn · W_o (fp8 W8A8, scale source 9)
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 16,
                    weight: GemmWeight::Fp8Dynamic,
                },
                m: 1,
                inputs: vec![InputRef::Op(4), InputRef::Ext(8), InputRef::Ext(9)],
            },
        ];
        let input = LoweringInput {
            sources,
            ops,
            result: 5,
            exported: None,
        };
        let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
        scratchy_subtile::subtile_ir::lower_region(&input, nb)
    }

    /// ⭐ THE COPY MINT IS THE WIRING'S REGISTRY TOO — pinned after #299's registry rewrite of
    /// [`graph_wiring`] left `weight_copies` on the layout's `Default` while the placement pass
    /// kept minting it: every head-major copy placed-but-never-staged, and the m=1 fp8 o
    /// matmul read never-written windows (`Fr4 (oon`, granite-3.1-8b fp8, card-only — the
    /// emulator executes the KTIR program, not the staging). The wiring must carry EXACTLY the
    /// placement pass's list, and a uniform multi-slab fp8 graph — granite-8b's shape — is the
    /// fixture that states it. If this assertion ever fails, one side of the mint re-derived
    /// its own list again.
    #[test]
    fn the_wiring_carries_the_oproj_copy_mint() {
        let ir = uniform_fp8_oproj_graph();
        let weight_ids: std::collections::HashSet<u32> = [1u32, 2, 8, 9].into_iter().collect();
        let wiring =
            graph_wiring(&ir, &weight_ids).expect("the uniform multi-slab fp8 graph wires");
        let layout = crate::lower_subtile_tape_to_superdsc::compute_bundle_layout(
            &ir,
            &weight_ids,
            false,
            false,
            &[],
            None,
        )
        .expect("the same graph lays out");
        assert_eq!(
            wiring.weight_copies, layout.weight_copies,
            "the wiring's copy list is the placement pass's mint — the worker materializes each \
             `to` tid from its `from` tid's bytes at load, and a list that disagrees with the \
             placements stages nothing for a window the m=1 fp8 o matmul reads"
        );
        assert_eq!(
            wiring.weight_copies,
            vec![(8, oproj_headmajor_tid(8))],
            "the uniform fixture mints exactly W_o t8's pair: one layer's o_proj weight gains \
             its head-major copy"
        );
    }
}
