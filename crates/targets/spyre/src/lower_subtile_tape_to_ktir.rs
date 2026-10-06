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
use ktir_superdsc::ktir_node::ActiveCap;
use scratchy_subtile::subtile_ir::{RopeForm, SubOp, SubtileIR, SubtileNode};
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

/// ⭐⭐⭐ THE BUNDLE'S ATTENTION PARAMETERS, READ WHERE MAIN READ THEM.
///
/// `ibm/main`'s `lower_one_node` lowered each node during the tape walk, so `SubOp::AttnDecode`'s
/// `geom` and `scale` were in its hand and `active_cap` / `rows_are_requests` were its own walk
/// parameters. This split lowers KTIR → SuperDSC one pass later, per BUNDLE, so the same four facts
/// are read HERE — off the graph's own `AttnDecode` nodes and off this walk's parameters — and travel
/// to the door as [`crate::ktir_superdsc_door::BundleAttnParams`], an argument of the call.
///
/// ⛔ THE MODEL FACTS MUST BE THE MODEL'S, SO A DISAGREEMENT IS AN ERROR AND NOT A CHOICE. One
/// `#[forward]` expansion is one model, so every `AttnDecode` node in one graph carries the same
/// geometry and the same multiplier. If two ever differed, one value per bundle could not describe
/// both, and picking the first would silently give one layer another layer's registry slot — so this
/// refuses instead, naming both.
///
/// `None` when the graph has no attention node: there is then nothing for the four to be facts of, and
/// an attention program arriving at the door without them is that door's own build error.
pub(crate) fn attn_bundle_params<F: RopeForm>(
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> Result<Option<crate::ktir_superdsc_door::BundleAttnParams>, SuperDscError> {
    let mut found: Option<(ktir_superdsc::head_counts::ModelAttnGeometry, u32)> = None;
    for n in &ir.nodes {
        let SubOp::AttnDecode { geom, .. } = &n.op else {
            continue;
        };
        let t = n.output.tensor.index() as u32;
        match found {
            None => found = Some((*geom, t)),
            Some((g0, t0)) => {
                if g0 != *geom {
                    return Err(SuperDscError(format!(
                        "AttnDecode t{t0} declares geometry ({g0}) while AttnDecode t{t} declares \
                         ({geom}). One bundle is one model, and the geometry door the lowering \
                         crosses takes ONE geometry for the whole bundle; a graph carrying two would \
                         give one layer the other's."
                    )));
                }
            }
        }
    }
    Ok(
        found.map(|(geom, _)| crate::ktir_superdsc_door::BundleAttnParams {
            geom,
            rows_are_requests,
        }),
    )
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
    // The negative-symbol-id counter and the bundle layout are the CONSUMER's facts now:
    // every spliced kind hands the door a program, and `ktir_superdsc`'s own walk mints
    // both. They remain parameters because the re-rolled walk's other consumers thread
    // them; the spliced arms state them through the splice's own mint instead.
    _sym_id_base: &mut i64,
    _layout: Option<&BundleLayout>,
) -> NodeLowering {
    use NodeLowering::{HostRouted, Ops, Unhandled};
    // ── PREFILL (m>1) LM-HEAD TAIL, FOLDED TO m=1 — the prefill bundle produces the FIRST generated
    // token's logits itself, so TTFT is ONE forward, not two. ──
    // The vocab-wide (≈49159-col) lm_head cannot run at m>1: it ALWAYS time-tiles, and per-row (m>1)
    // time-tiling is unimplemented (design-risk-4 Err). Only the LAST prompt token's logits are ever
    // read, so the tail runs at m=1 over `last_hidden[1, hidden]` = row `selector_lastrow_col(mq)` of the
    // final-norm output — which is exactly the shape the PROVEN decode path lowers. See
    // [`LAST_HIDDEN_TID`] for why the extraction is per-stick copies rather than a one-hot matmul.
    //
    // DETECT BY VOCAB-WIDTH, not `output tid == ir.result`: granite has a LOGITS ScalarMul AFTER the
    // lm_head matmul, so `ir.result` is the ScalarMul's output — the matmul's OWN tid never equals it (the
    // observed miss: the lm_head matmul t1129 kept time-tiling because ir.result was the ScalarMul t1130).
    // The lm_head matmul AND the logits ScalarMul are the ONLY ops whose output spans the result cols
    // (vocab); every intermediate is hidden/intermediate width. So BOTH re-lower at m=1.
    // The mq=1 DECODE bundle is UNAFFECTED (out_rows==1 ⇒ not the prefill tail ⇒ lowered as before).
    let result_cols = ir.tensors[ir.result.index() as u32 as usize].cols;
    let _ = result_cols; // retained: the lm-head tail shapes are now the splice's to state
    match &node.op {
        // The rest of the arch vocabulary. It reaches this emitter because the SHARED
        // front end expresses every op instead of asserting the unsupported ones away
        // in `lower_region` — which is the point: the IR carries the fact and the
        // TARGET says whether it has a kernel. Enumerated, never `_`, so adding a
        // SubOp is E0004 here rather than a surprise at emission.
        SubOp::TanhSoftCap
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::GateScale
        | SubOp::LoadPixels { .. }
        | SubOp::LoadPosEmbeds { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        | scratchy_subtile::expansion_ops!()
        | SubOp::Mean => Unhandled(format!("{:?} has no SuperDSC kernel", node.op)),
        // ⛔ THE ONE PLACE THAT MUST IMPLEMENT IT, so the refusal lives here
        // and names the required lowering rather than the op.
        SubOp::Reshape { .. } => Unhandled(
            "SubOp::Reshape reached the SuperDSC lowering. It must become a RESTICKIFY \
                 (a real re-laying copy), NOT a placement alias: `dev_off_stk` places (i,j) \
                 at (j/stk)*(a*stk)+i*stk+(j%stk) where `a` is the ROW COUNT, so two views \
                 over one buffer with different extents disagree about every element. And \
                 `declare_arrangement` will NOT catch an alias — it keys on tensor NAME, and \
                 an alias gives the two views two names."
                .to_string(),
        ),
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
        // ⛔ SPYRE'S RMSNORM MULTIPLIES BY THE STORED GAIN. The gemma-class (1 + w)
        // convention needs a different kernel, and running the Scale one over a
        // zero-centred gain scales every normalized activation by roughly nothing — a
        // model that loads, runs, and is quietly wrong. So it refuses BY NAME.
        //
        // It can reach here at all because the shared front end now EXPRESSES the
        // convention instead of asserting it away in `lower_region`. That is the trade:
        // the IR carries the fact, and the target says whether it has a kernel for it.
        SubOp::RmsNorm {
            gain: scratchy_subtile::subtile_ir::GainConvention::OnePlusScale,
            ..
        } => Unhandled(format!(
            "RmsNorm t{} uses the (1 + w) gain convention, for which this emitter has \
             no kernel — its rmsnorm multiplies by the stored gain",
            node.output.tensor.index() as u32,
        )),
        SubOp::RmsNorm { .. } => match scratchy_triton_splice::lower(node, ir, rows_are_requests) {
            Ok(e) => Ops(vec![e]),
            Err(reason) => Unhandled(reason),
        },
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
        SubOp::ScalarMul { .. } => {
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
            match scratchy_triton_splice::lower_attn(node, ir, cap, active_cap) {
                Ok(e) => Ops(vec![e]),
                Err(reason) => Unhandled(reason),
            }
        }
        SubOp::RmsNormReduce { .. } | SubOp::RmsNormApply { .. } => {
            HostRouted(host_glue_kind(&node.op))
        }
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
    // GLOBAL ≤7-segment memory layout (task #55) computed ONCE for the whole bundle.
    // Every op resolves each tensor's HBM address from this (by `t{id}` name) so a
    // tensor shared across ops gets the SAME address — fixing the per-op `arg_index`
    // segment-aliasing bug. Threaded as `Some(&layout)` into every node-lowering; the
    // populated layout (incl. synthetic seg3 offsets) is RETURNED for the manifest.
    // ⛔ NO LAYER CLASSES: this is the UNROLLED lowering, which has no layer loop and therefore no
    // layer boundary to split the weight segment on. An empty map is what tells the layout that
    // banking is not expressible here, leaving the tail spill as the only lever (`&Default::default()`
    // rather than a bool, so there is one spelling of "the layer structure" and not two).
    let bundle_layout =
        compute_bundle_layout(ir, weight_ids, rows_are_requests, &Default::default())?;
    let layout = Some(&bundle_layout);
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(ir.nodes.len());
    // The SINGLE monotonic negative-symbol-id counter for the WHOLE bundle (design
    // risk #1): every tiled tensor/core gets `-(++sym_id_base)`, threaded through
    // each node-lowering so ids never collide across ops in one bundle (torch-spyre
    // `symbol_id_offset_counter`). A per-op restart would alias addresses and
    // silently corrupt — we assert disjointness below.
    let mut sym_id_base: i64 = 0;
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
        match lower_one_node(
            node,
            ir,
            active_cap,
            rows_are_requests,
            &mut sym_id_base,
            layout,
        ) {
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
    let attn_params = attn_bundle_params(ir, rows_are_requests)?;
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
}

/// One program's parameter order: `args[i]` is the tensor the i-th parameter addresses.
pub struct NodeArgs {
    pub args: Vec<usize>,
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
    // ⛔ NO LAYER CLASSES, for the same reason [`lower_graph_to_superdsc`] passes none: this walks
    // the UNROLLED graph, which has no layer loop and so no boundary a weight BANK may fall on.
    let layout = compute_bundle_layout(ir, weight_ids, false, &Default::default())?;
    let mut sym_id_base: i64 = 0;
    let mut nodes = Vec::with_capacity(ir.nodes.len());
    let mut mask: Option<(u32, u32)> = None;
    for node in &ir.nodes {
        let lowered = lower_one_node(
            node,
            ir,
            ActiveCap::FULL,
            false,
            &mut sym_id_base,
            Some(&layout),
        );
        let ops = match lowered {
            NodeLowering::Ops(v) => v,
            // A host-routed op runs off the device, so it has no program and binds nothing. It
            // still occupies a position in the node order, so it contributes an empty entry.
            NodeLowering::HostRouted(_) => Vec::new(),
            NodeLowering::Unhandled(why) => return Err(SuperDscError(why)),
        };
        for e in ops {
            let Some(k) = e.ktir else { continue };
            if let Some(mid) = k.mask {
                // ⭐ THE CAPACITY COMES OFF THE PROGRAM, not off the node. The mask parameter's own
                // `ktdp.construct_memory_view` states `[1, capacity]`, so the number this pass needs to
                // size the host buffer is read where every other extent is read. It used to ride on
                // `KtirNode::mask` as the second half of a pair the LOWERING never looked at.
                let regs = ktir_superdsc::emit::lower_ktir_to_superdsc::regions(&k)
                    .map_err(|e| SuperDscError(e.message))?;
                let cap = regs
                    .iter()
                    .find(|r| r.tid == mid.get())
                    .map(|r| r.v_cols)
                    .ok_or_else(|| {
                        SuperDscError(format!(
                            "the mask buffer t{mid} is not among this program's parameters, so its \
                             `[1, capacity]` view states no capacity to size the host buffer with"
                        ))
                    })?;
                let m = (mid.get(), cap);
                if mask.is_some_and(|prev| prev != m) {
                    return Err(SuperDscError(format!(
                        "the attention mask must be uniform across layers (one shared prefix \
                         capacity), but {:?} and {m:?} were both emitted",
                        mask.unwrap()
                    )));
                }
                mask = Some(m);
            }
            nodes.push(NodeArgs {
                args: k.bindings.iter().map(|b| b.get() as usize).collect(),
            });
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
        scalarmul_scales: layout.scalarmul_scales.clone(),
    })
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
) -> Result<RolledSuperDsc, SuperDscError> {
    use scratchy_subtile::subtile_tape::{ComputeInput, Instr, LoopBound};
    // ⛔ THE `set_rows_are_requests` / `RestoreRar` DANCE IS DELETED. It pushed the row KIND into a
    // thread-local for the duration of one bundle's lowering, with a `Drop` guard to restore it, purely
    // so the matmul splitter would not need the fact in its signature. That made the kind ambient: no
    // site was obliged to receive it, ~97 sites branched on the row COUNT instead, and a batched decode
    // was emitted as a prefill chunk everywhere the count could not tell them apart. The kind is a
    // COMPILE-TIME CONSTANT of the bundle (this crate is driven by a proc macro that knows the model and
    // the rung as literals), so it belongs in a type — `sdsc_abstract::QueryRows<ROWS_ARE_REQUESTS>` —
    // and in the signatures that need it.
    //
    // What survives is ONE perf gate: a decode batch must not split `mb` (the PT array holds the weight
    // stationary and streams M through it, so an `mb` split reloads the weight per split and cancels the
    // amortization batching exists to buy). That is a throughput/compatibility choice, not a statement
    // about what a row means, and it is named accordingly. See `matmul/dims.rs` for the const-generic
    // fix that removes even this.
    let _prev_split_gate =
        crate::ir::bridge::tiled_op_sdsc_op::matmul::set_split_mb_forbidden(rows_are_requests);
    struct RestoreSplitGate(bool);
    impl Drop for RestoreSplitGate {
        fn drop(&mut self) {
            crate::ir::bridge::tiled_op_sdsc_op::matmul::set_split_mb_forbidden(self.0);
        }
    }
    let _restore_split_gate = RestoreSplitGate(_prev_split_gate);
    // ⭐ THE LAYER STRUCTURE FIRST, because the layout needs it: a weight BANK boundary may only fall
    // on a LAYER boundary, and `compute_bundle_layout` is where the weight segment is packed. Same
    // tape, same `ComputeInput::External::per_layer` the walk below reads — collected once, here.
    let per_layer_ext = per_layer_external_tids(tape);
    let mut bundle_layout =
        compute_bundle_layout(ir, weight_ids, rows_are_requests, &per_layer_ext)?;
    // ON-CARD RESIDUAL (unconditional): thread the loop-carried hidden IN-PLACE, no copy. Pre-scan the
    // loop body for hidden_in (first body node input[0]) + hidden_out (last body node output) and ALIAS
    // hidden_out's placement to hidden_in's → every iteration reads+writes ONE resident buffer, so the
    // residual threads with NO host round-trip and NO device copy (replaces the host thread_hidden).
    // WAR-safe: within a layer hidden_in's last read (computing h1 = h_in+attn) precedes hidden_out's
    // write (the final h_out = h1+mlp add); across iters the shared buffer carries the residual. The
    // shim's thread_hidden becomes a no-op once the placements coincide.
    {
        let (mut pseg, mut pf, mut plast, mut psuf): (u8, Option<u32>, Option<u32>, Option<u32>) =
            (0, None, None, None);
        for instr in tape.instrs() {
            match instr {
                Instr::OpenLoop { .. } => pseg = 1,
                Instr::CloseLoop { .. } => pseg = 2,
                Instr::Compute { node, .. } => {
                    if pseg == 1 {
                        if pf.is_none() {
                            pf = Some(node.index() as u32);
                        }
                        plast = Some(node.index() as u32);
                    } else if pseg == 2 && psuf.is_none() {
                        psuf = Some(node.index() as u32);
                    }
                }
                _ => {}
            }
        }
        if let (Some(f), Some(l)) = (pf, plast)
            && let Some(p) = bundle_layout
                .placements
                .get(&(ir.nodes[f as usize].inputs[0].tensor.index() as u32))
                .cloned()
        {
            let hout = ir.nodes[l as usize].output.tensor.index() as u32;
            // hidden_out (last body op) → hidden_in's buffer: per-iter residual threads in place.
            bundle_layout.placements.insert(hout, p);
            // suffix_in (first suffix op's input[0]) → the same buffer: the body→suffix seam threads
            // in place too, so BOTH host routings (thread_hidden + thread_to_suffix) are unnecessary.
            if let Some(s) = psuf {
                let sin = ir.nodes[s as usize].inputs[0].tensor.index() as u32;
                bundle_layout.placements.insert(sin, p);
            }
        }
    }
    let layout = Some(&bundle_layout);
    let mut sym_id_base: i64 = 0;
    let (mut prefix, mut body, mut suffix): (Vec<EmittedOp>, Vec<EmittedOp>, Vec<EmittedOp>) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut iters: u32 = 0;
    let mut seg: u8 = 0; // 0 = prefix, 1 = body (inside the layer loop), 2 = suffix
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // ⭐ SEEDED FROM THE PRE-PASS RATHER THAN REBUILT. `per_layer_external_tids` already read every
    // per-layer WEIGHT/KV class off this same tape — it had to, because `compute_bundle_layout` needs
    // the layer boundaries to decide weight BANKS, and that runs before any op is emitted. The walk
    // below then adds only what the pre-pass cannot see (node outputs, the resident Kᵀ), and its own
    // `or_insert_with` is a no-op for anything already here.
    //
    // ⛔ A DISAGREEMENT BETWEEN THE TWO IS CAUGHT, NOT ASSUMED AWAY: a class the pre-pass missed was
    // never banked, so its layers sit at their un-banked offsets and the per-layer FORMULA check
    // after this walk refuses the bundle naming that tensor.
    let mut per_layer: std::collections::BTreeMap<u32, Vec<u32>> = per_layer_ext.clone();
    // Which weight BANKS each launch group addresses (0 = prefix, 1 = body, 2 = suffix), accumulated
    // as the ops are emitted. A launch has ONE base per segment, so a group may address exactly one;
    // `group_bank` checks that after the walk.
    let mut group_weight_banks: [std::collections::BTreeSet<u32>; 3] = Default::default();
    // First/last body Compute node (for the residual-stream hidden in/out tids).
    let mut first_body_node: Option<u32> = None;
    let mut last_body_node: Option<u32> = None;
    // First SUFFIX Compute node — its input[0] is the residual the suffix reads (the
    // last layer's post-attn residual, e.g. t780). The rerolled body writes its output
    // to the REPRESENTATIVE-iteration tid (hidden_out_tid, e.g. t360), NOT t780, so the
    // body→suffix seam must thread hidden_out → suffix_in at runtime (else the suffix
    // reads an unwritten slot = 0 → rmsnorm(0)=inf → garbage logits).
    let mut first_suffix_node: Option<u32> = None;
    // Matmul KERNEL weights (layer-0 tid, in=k, out=n) for the device re-tile manifest.
    let mut kernel0: Vec<(u32, u32, u32)> = Vec::new();
    // LAYER blocked-f16 (SCRATCHY_LAYER_KSPLIT): collected down_proj `(dp_per_layer tids, dev_out, b_blocks)`
    // for POST-LOOP block-weight placement overlay (bundle_layout is borrowed by `layout` inside the loop).
    for instr in tape.instrs() {
        match instr {
            Instr::OpenLoop { bound, .. } => {
                iters = match bound {
                    LoopBound::Const(it) => *it,
                    LoopBound::Runtime(_) => {
                        return Err(SuperDscError(
                            "reroll-superdsc: a runtime-bounded layer loop is unsupported (the \
                             decode layer count is a compile-time Const)"
                                .into(),
                        ));
                    }
                };
                seg = 1;
            }
            Instr::CloseLoop { .. } => seg = 2,
            Instr::Compute {
                node,
                inputs,
                per_layer_out,
                ..
            } => {
                let n = &ir.nodes[node.index()];
                // Collect matmul KERNEL weights for the device re-tile manifest: w =
                // inputs[1] is [k,n]=[in,out] row-major; the PT array needs the device
                // tile layout [out/64, in, 64]. in=A.cols (k), out=output.cols (n).
                // Arity-3 fp8 W8A8 matmuls MUST be collected here too: their weight (inputs[1]; inputs[2] is
                // w_scale) needs the 1-byte packed RetileDescriptor from the `fp8_weight_tids` branch below.
                // They were EXCLUDED (arity-2 only), so the shim staged them FLAT → scrambled weights → garbage
                // (`scr chat` incoherent). in_k/out derive identically (inputs[0].cols=k, output.cols=n); the
                // KSPLIT branch (out>16384) never fires for fp8 projs. Root-caused + on-card confirmed 2026-07-16.
                if matches!(n.op, SubOp::MatmulTile { .. })
                    && (n.inputs.len() == 2 || n.inputs.len() == 3)
                {
                    kernel0.push((
                        n.inputs[1].tensor.index() as u32,
                        n.inputs[0].region.cols.len,
                        // DEVICE out extent via the TYPE-SAFE `DeviceWidth` (the SAME rule as
                        // main's `lower_matmul_node` `n_dev` and the worker's weight zero-pad): granite
                        // lm_head 49159→49664. So the RetileDescriptor device_size, the per-core
                        // address, and the staged buffer all agree by construction.
                        DeviceWidth::for_output(
                            n.output.region.rows.len,
                            n.output.region.cols.len,
                            n.inputs[0].region.cols.len,
                        )
                        .get(),
                    ));
                    // KSPLIT fp32-merge (lm_head first-test, n>16384): the split weight ALSO needs B
                    // block-weight descriptors `[KB,N]` under reserved tids `ksplit_block_tid(b)`, matching
                    // the emit branch's block matmuls. Same gate + condition. The kernel0 descriptor loop
                    // below builds each as `[dev_out/64, KB, 64]` (in=KB) — exactly the block matmul's read
                    // (`ksplit_block_weight_kslice_offset_matches_read`). Worker gathers the K-slice bytes.
                }
                if seg == 1 {
                    if first_body_node.is_none() {
                        first_body_node = Some(node.index() as u32);
                    }
                    last_body_node = Some(node.index() as u32);
                }
                if seg == 2 && first_suffix_node.is_none() {
                    first_suffix_node = Some(node.index() as u32);
                }
                // Per-layer OUTPUT tids (the node's output tensor in each layer copy).
                if iters > 1 && per_layer_out.len() as u32 == iters {
                    let outs: Vec<u32> = per_layer_out
                        .iter()
                        .map(|nid| ir.nodes[nid.index()].output.tensor.index() as u32)
                        .collect();
                    per_layer
                        .entry(n.output.tensor.index() as u32)
                        .or_insert(outs);
                }
                // Per-layer WEIGHT/external tids.
                for ci in inputs.iter() {
                    if let ComputeInput::External {
                        tensor,
                        per_layer: pl,
                        ..
                    } = ci
                        && iters > 1
                        && pl.len() as u32 == iters
                    {
                        per_layer
                            .entry(tensor.index() as u32)
                            .or_insert_with(|| pl.iter().map(|t| t.index() as u32).collect());
                    }
                }
                // RESIDENT kct per-layer registration (kill-the-restickify residency): the score reads a
                // per-layer RESIDENT Kᵀ kernel `kct_resident_tid(k_id)`. It's neither an External nor a node
                // output, so the loops above never add it — insert it manually (mirror the downproj block
                // insert below): map each layer's K-cache source tid k_Lv → kct_resident_tid(k_Lv), keyed
                // under the layer-0 kct tid. REQUIRED so the seg2 uniformity guard validates kct's per-layer
                // stride matches k/v — the executor advances kct's seg2 base by v·kv_stride automatically, so
                // a non-uniform kct packing would SILENTLY read the wrong layer (runtime-errors-need-compile-
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
                    per_layer
                        .entry(kct_resident_tid(kv.cache_tensor().index() as u32))
                        .or_insert(kct_tids);
                }
                // LAYER blocked-f16 (SCRATCHY_LAYER_KSPLIT): down_proj (large K) → B per-layer block weights.
                // Push each block's kernel0 `[KB,dev_out]` descriptor + its per-layer tid list; collect the
                // down_proj weight's per-layer list for the POST-LOOP placement overlay (block (v,b) at
                // dp_Lv.offset + b·KB·dev_out·2, so weight_stride is unchanged). All offsets Kani-proven.
                // ⭐ THE WEIGHT BANK THIS OP'S GROUP NEEDS. Recorded per group as the ops are emitted
                // because it is a property of the OPERANDS, which only the node knows; a launch binds
                // ONE base per segment, so the set for each group must end up with at most one
                // member. Empty for every op of every unbanked bundle.
                for r in &n.inputs {
                    let tid = r.tensor.index() as u32;
                    if let Some(p) = bundle_layout.placements.get(&tid)
                        && matches!(p.role, SegRole::Weight)
                    {
                        group_weight_banks[(seg as usize).min(2)].insert(p.bank);
                    }
                }
                match lower_one_node(
                    n,
                    ir,
                    active_cap,
                    rows_are_requests,
                    &mut sym_id_base,
                    layout,
                ) {
                    NodeLowering::Ops(v) => match seg {
                        0 => prefix.extend(v),
                        1 => body.extend(v),
                        _ => suffix.extend(v),
                    },
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
    if iters == 0 {
        return Err(SuperDscError(
            "reroll-superdsc: no layer loop in the rerolled tape (no OpenLoop/CloseLoop) — \
             reroll_subtile_tape found no repeating body"
                .into(),
        ));
    }
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
    let attn_params = attn_bundle_params(ir, rows_are_requests)?;
    for ops in [&prefix, &body, &suffix] {
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
            let k = *in_k as u64;
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
        bundle_layout.kernel_weights.insert(*w_tid, desc.clone());
        if let Some(layers) = per_layer.get(w_tid) {
            for &t in layers {
                bundle_layout.kernel_weights.insert(t, desc.clone());
            }
        }
    }
    let seg3 = SegRole::Intermediate.segment();
    let synth_high = bundle_layout.synth.borrow().next;
    if synth_high > bundle_layout.segment_bytes[seg3] {
        bundle_layout.segment_bytes[seg3] = synth_high;
    }
    // Per-layer SEGMENT strides for the executor: WEIGHTS (seg1) + KV (seg2) advance
    // by `v·stride` per iteration (the body's baked layer-0 offsets shift to layer-v
    // when the executor passes `seg_base + v·stride`). Verify UNIFORMITY here (a
    // build-time guard) — a non-uniform packing would make the seg-base advance read
    // the WRONG layer's weights (silent garbage). seg3 intermediates (the loop-carried
    // hidden) are EXCLUDED (host-threaded, not strided). 0 = no per-layer tensor there.
    let w_seg = SegRole::Weight.segment();
    let kv_seg = SegRole::Kv.segment();
    let mut weight_stride: u64 = 0;
    let mut kv_stride: u64 = 0;
    for tids in per_layer.values() {
        if tids.len() < 2 {
            continue;
        }
        let Some(p0) = bundle_layout.placements.get(&tids[0]) else {
            continue;
        };
        let seg = p0.segment;
        let target = if seg == w_seg {
            &mut weight_stride
        } else if seg == kv_seg {
            &mut kv_stride
        } else if matches!(p0.role, SegRole::Weight) {
            // ⛔⛔⛔ A PER-LAYER **WEIGHT** OUTSIDE THE STRIDED SEGMENT IS SILENT GARBAGE, and the
            // arm below would have `continue`d past it. The rolled body advances only seg{w_seg}'s
            // base, so layer v would read layer 0's copy of this tensor for all `iters` layers:
            // fluent output, wrong model. [`spill_weight_tail`] may only move the NON-per-layer tail,
            // and this is what holds it to that.
            //
            // ⛔ KEYED ON THE **ROLE**, NOT THE SEGMENT. A per-layer INTERMEDIATE colored into a
            // spill slot is legitimate and must keep falling through — the `WeightOverflow = 5`
            // attempt refused exactly that case (per-layer intermediate t448 in seg5) and read it as
            // proof no segment was available, when the real defect was taking a COLOR.
            return Err(SuperDscError(format!(
                "reroll-superdsc: per-layer WEIGHT t{} is placed in seg{seg}, which gets no \
                 per-layer stride — the rolled body advances only seg{w_seg}, so every one of the \
                 {} layers would read layer 0's copy. Only NON-per-layer weights (the final norm, \
                 the lm_head / tied embedding) may spill; see `spill_weight_tail`.",
                tids[0],
                tids.len(),
            )));
        } else {
            continue; // seg3 hidden / per-layer intermediate color: not stride-advanced
        };
        for w in tids.windows(2) {
            let (Some(a), Some(b)) = (
                bundle_layout.placements.get(&w[0]),
                bundle_layout.placements.get(&w[1]),
            ) else {
                return Err(SuperDscError(
                    "reroll-superdsc: a per-layer tensor is missing a layout placement".into(),
                ));
            };
            if a.segment != seg || b.segment != seg {
                return Err(SuperDscError(
                    "reroll-superdsc: a per-layer tensor changes segment across layers".into(),
                ));
            }
            // ⭐ A BANK BOUNDARY IS THE ONE PLACE THE STRIDE LEGITIMATELY DOES NOT APPLY: layer `v`
            // is `(v / lpb, (v % lpb)·stride)`, so crossing into the next bank resets the offset
            // instead of advancing it. Skipping the pair here is not a hole in the guard — the FULL
            // per-layer formula (bank AND offset, for every layer, not just consecutive pairs) is
            // checked below, which is strictly stronger than this pairwise delta ever was.
            if a.bank != b.bank {
                continue;
            }
            let d = b.offset.wrapping_sub(a.offset);
            if *target == 0 {
                *target = d;
            } else if *target != d {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: NON-UNIFORM per-layer stride in seg{seg} ({} vs {} bytes) — \
                     the executor advances the segment base by v·stride, which needs layers packed \
                     at a uniform stride. Reorder compute_bundle_layout to pack each layer's \
                     {} contiguously.",
                    *target,
                    d,
                    if seg == w_seg { "weights" } else { "KV" },
                )));
            }
        }
    }
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⭐ THE WEIGHT BANK CONTRACT — the launch-time arithmetic, proven here against the addresses
    //  that were actually baked.
    //
    //  The executor reaches layer `v` with `bank = v / layers_per_bank` and
    //  `off[SEG_WEIGHT] = (v % layers_per_bank) · weight_stride`. Every term of that is decided in
    //  `bank_weight_segment`, so all three checks below compare the FORMULA against the placements
    //  rather than re-deriving the policy — the failure mode is a layer reading another layer's
    //  weights, which is fluent, wrong output and nothing else.
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //
    // `layers_per_bank` is READ OFF the placements: how many layers share bank 0. Every layer in one
    // bank (the unbanked case) gives `iters`, which makes the division a no-op.
    let mut layers_per_bank: u32 = iters.max(1);
    for tids in per_layer.values() {
        let Some(p0) = bundle_layout.placements.get(&tids[0]) else {
            continue;
        };
        if p0.segment != w_seg || !matches!(p0.role, SegRole::Weight) || tids.len() as u32 != iters
        {
            continue;
        }
        let in_bank0 = tids
            .iter()
            .filter(|t| {
                bundle_layout
                    .placements
                    .get(*t)
                    .is_some_and(|p| p.bank == 0)
            })
            .count() as u32;
        if in_bank0 == 0 {
            return Err(SuperDscError(format!(
                "reroll-superdsc: per-layer weight class t{} has NO layer in bank 0, but the rolled \
                 body is baked at LAYER 0 — its addresses would name a bank no launch binds.",
                tids[0],
            )));
        }
        layers_per_bank = layers_per_bank.min(in_bank0);
    }
    // Now hold EVERY per-layer weight to `(v / lpb, (v % lpb)·stride + its layer-0 offset)`. This is
    // the check the pairwise delta above cannot make: it validates the bank as well as the offset,
    // and it validates every layer rather than every consecutive pair.
    for tids in per_layer.values() {
        let Some(p0) = bundle_layout.placements.get(&tids[0]) else {
            continue;
        };
        if p0.segment != w_seg || !matches!(p0.role, SegRole::Weight) || tids.len() as u32 != iters
        {
            continue;
        }
        for (v, t) in tids.iter().enumerate() {
            let Some(p) = bundle_layout.placements.get(t) else {
                continue;
            };
            let want_bank = v as u32 / layers_per_bank;
            let want_off = (v as u64 % layers_per_bank as u64) * weight_stride + p0.offset;
            if p.bank != want_bank || p.offset != want_off {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: per-layer weight t{t} (layer {v} of class t{}) is placed at \
                     bank {} offset {}, but the executor will address layer {v} at bank \
                     {want_bank} offset {want_off} ({} layer(s)/bank, stride {weight_stride} B, \
                     layer-0 offset {}). Every layer must sit where the per-layer advance looks, or \
                     that layer reads another layer's weights.",
                    tids[0], p.bank, p.offset, layers_per_bank, p0.offset,
                )));
            }
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
    let suffix_weight_bank = group_bank(2, "suffix")?;
    // The BODY is baked at layer 0, which the loop above has already proven lives in bank 0, so any
    // OTHER bank in the body means it also reads a weight that is not per-layer — the replication
    // case, refused with its own name rather than as a stride mismatch three steps later.
    if group_bank(1, "body")? != 0 {
        return Err(SuperDscError(format!(
            "reroll-superdsc: the body program addresses weight bank(s) {:?}, but it is baked at \
             LAYER 0 and every launch of it advances bank 0's base. A NON-per-layer weight read \
             inside the layer loop would need replicating into every bank.",
            group_weight_banks[1],
        )));
    }
    // Residual-stream hidden in/out tids for the executor's loop-carried threading:
    // the FIRST body node's input[0] (the layer's hidden-in, e.g. the rmsnorm x) and
    // the LAST body node's output (the layer's hidden-out, the final residual add).
    let hidden_in_tid = first_body_node
        .and_then(|nid| ir.nodes[nid as usize].inputs.first())
        .map(|r| r.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    let hidden_out_tid = last_body_node
        .map(|nid| ir.nodes[nid as usize].output.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    // The suffix's residual input (first suffix node's input[0]) — the executor threads
    // hidden_out → suffix_in once after the layer loop (the body→suffix seam). Same
    // input-ordering convention as hidden_in_tid (rmsnorm input[0] = x = the residual).
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
            "[SEGDUMP] iters={iters} total={:.3}GB segs(GB)=[{}]",
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
    // Stated only when there IS a request dimension, so an unpaged bundle refuses nothing.
    // ⛔ ALWAYS 0. There is no request dimension in a page any more, so there are no "request rows" for
    // the runtime to bound a shift by. A request is reached by its PAGE, through the host's block table.
    let kv_request_rows = 0u32;
    Ok(RolledSuperDsc {
        prefix,
        body,
        suffix,
        iters,
        layout: bundle_layout,
        per_layer,
        weight_stride,
        layers_per_bank,
        prefix_weight_bank,
        suffix_weight_bank,
        kv_stride,
        kv_request_stride,
        kv_request_rows,
        hidden_in_tid,
        hidden_out_tid,
        suffix_in_tid,
        attn_params,
    })
}
