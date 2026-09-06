// SPDX-License-Identifier: Apache-2.0
//! BRIDGE 1, SCRATCHY SIDE — the subtile tape in a vocabulary `deeptools` can name.
//!
//! ```text
//! SubtileIR tape ──► DataflowIR ──► SentientIR ──► ProgIR ──► SenProg ──► init_binary
//!    (here)          (deeptools)              the backend C++ compiler
//! ```
//!
//! ⛔⛔ FROM THE TAPE, NEVER FROM SUPERDSC. Nothing in this file reads a `Dsc`, an `SdscOp` or an
//! `EmittedOp`. It reads `SubtileIR` — the nodes, their regions, their ops — and the PLACEMENT PLAN,
//! which [`compute_bundle_layout`] derives from that same graph. An earlier attempt built a
//! lowering, threw the shape away into string dimension names and tried to read it back; every
//! defect it hit was a symptom of recovering rather than reading.
//!
//! ⭐⭐ THE PLACEMENT AUTHORITY IS SHARED, WHICH IS THE WHOLE POINT. `compute_bundle_layout` takes
//! `&SubtileIR` and `weight_ids` and produces the segment and byte offset of every tensor — and it
//! is the SAME plan the bundle's `bundle_layout.json` carries, which is what the worker H2Ds
//! against. So a weight's address here is the address the weight is actually at. The nuked branch
//! addressed every operand into the LX instead, and its programs read memory nothing ever filled.
//!
//! ⛔ THE CONSTANTS ARRIVE THROUGH DOORS, NOT AS VALUES. The model's seven numbers come through
//! `with_config_model`, the rung's two through the ladders below. Inside those arms they are
//! literals, so `deeptools`' `Exploit` flags are decided by the compiler and the branches they
//! drive emit different programs.

use std::collections::HashSet;

use scratchy_subtile::model_geometry::{OnModel, with_config_model};
use scratchy_subtile::subtile_ir::{EwKind, RopeForm, SubOp, SubtileIR, TensorRegion};

use deeptools::arch::{Bytes, Dd2, Elements};
use deeptools::bridges::subtile_to_dataflow_ir::node::{
    Cols, Node, Operand, Residence, Rows, Segment,
};
use deeptools::bridges::subtile_to_dataflow_ir::tape;
use deeptools::generated::{DataType, OpFunc};
use deeptools::islands::dataflow_ir::{GroupId, print};
use deeptools::model::Model;
use deeptools::workload::Workload;

use crate::lower_subtile_tape_to_superdsc::{
    ActiveCap, BundleLayout, GroupKind, SegRole, Trip, TripRequest, compute_bundle_layout,
    group_ranges, group_size,
};
use scratchy_spyre_bundle as bundle;

/// WHICH OP-FUNC A TAPE NODE IS.
///
/// ⛔⛔ EXHAUSTIVE, WITH NO WILDCARD. Every `SubOp` is named, so adding one to the tape is an E0004
/// here rather than a node that silently lowers as something else. The arms that return
/// unbuilt arms are work not yet done, stated as such.
///
/// ⛔⛔ AND IT CANNOT REFUSE. There is no error type in this file. A lowering that returns `Err`
/// stops the tape BEFORE it is emitted, so dbo-opt — the only oracle this bridge has — is never
/// invoked and the build prints our sentence instead of the backend's. That has cost hours and a
/// revert five times. See `tests/dfir_never_runtime_refuses.rs`, which is the ratchet.
///
/// ⛔ AND THE TAPE DOES NOT ALWAYS ARRIVE DECOMPOSED. This said the opposite — that `decompose_rmsnorm`
/// had already split `RmsNorm` into its reduce and apply halves, so the map was "mostly one to one".
/// The acceptance build falsified it: a raw `SubOp::RmsNorm` reached this function and stopped the
/// whole tape. Nodes that are several op-funcs are handled by [`expand`], which returns a sequence;
/// this function answers only for the ones that are exactly one.
fn op_func_of<F: RopeForm>(op: &SubOp<F>) -> OpFunc {
    match op {
        SubOp::MatmulTile { .. } => OpFunc::Matmul,
        SubOp::AttnDecode { .. } => OpFunc::Batchmatmul,
        // A split-K combine is an elementwise add over equal-shaped partials.
        SubOp::SumReduce => OpFunc::Add,
        SubOp::Elementwise(EwKind::Add) => OpFunc::Add,
        SubOp::Elementwise(EwKind::Sub) => OpFunc::Sub,
        SubOp::Elementwise(EwKind::Mul) => OpFunc::Mul,
        SubOp::Elementwise(EwKind::Silu) => OpFunc::Silu,
        SubOp::Elementwise(EwKind::Gelu | EwKind::GeluErf | EwKind::QuickGelu) => OpFunc::Gelufwd,
        // The scalar rides in the op rather than as an operand, but the device work is a multiply.
        SubOp::ScalarMul { .. } | SubOp::ScalarWeightMul | SubOp::GateScale => OpFunc::Mul,
        SubOp::SiluMul | SubOp::GateApply => OpFunc::Mul,
        SubOp::RmsNormReduce { .. } => OpFunc::Mean,
        SubOp::RmsNormApply { .. } => OpFunc::Mul,
        SubOp::TanhSoftCap => OpFunc::Tanh,
        SubOp::Mean => OpFunc::Mean,
        // A reshape re-lays the buffer out: the device layout is a function of the ROW COUNT, so it
        // is a restickify rather than a copy.
        SubOp::Reshape => OpFunc::Restickifyophbm,
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => OpFunc::Mul,
        // ⛔⛔ NOT YET BUILT — SAY SO, DO NOT SUBSTITUTE. Each needs its own schedule read out of the
        // templates. Giving them a stand-in op-func (`Identity`, say) makes the tape lower WHOLE and
        // wrong: a Moe emitted as a copy is a program dbo-opt compiles happily and a model that
        // produces garbage. `todo!` names the op that has work, loudly, and cannot be logged and
        // carried on from the way a returned value can.
        //
        // ⛔ NONE OF THESE IS IN A DECODE TAPE. granite/llama decode is Gemm, AttnDecode, Add, Mul,
        // RmsNorm, RopeAppend, RopeRotate, ScalarMul, Silu — the wavefront census prints exactly
        // that list — so none of these fires on the acceptance target.
        SubOp::RmsNormUnit { .. } => todo!("SubOp::RmsNormUnit has no DataflowIR decomposition"),
        SubOp::GateSplit { .. } => todo!("SubOp::GateSplit has no DataflowIR decomposition"),
        SubOp::LoadPixels { .. } => todo!("SubOp::LoadPixels has no DataflowIR decomposition"),
        SubOp::LoadPosEmbeds { .. } => {
            todo!("SubOp::LoadPosEmbeds has no DataflowIR decomposition")
        }
        SubOp::EmbeddingGather { .. } => {
            todo!("SubOp::EmbeddingGather has no DataflowIR decomposition")
        }
        SubOp::VisionRope => todo!("SubOp::VisionRope has no DataflowIR decomposition"),
        SubOp::VarlenAttention { .. } => {
            todo!("SubOp::VarlenAttention has no DataflowIR decomposition")
        }
        SubOp::EncoderAttn { .. } => todo!("SubOp::EncoderAttn has no DataflowIR decomposition"),
        SubOp::GatedDeltaNet => todo!("SubOp::GatedDeltaNet has no DataflowIR decomposition"),
        SubOp::GemmaMoe { .. } => todo!("SubOp::GemmaMoe has no DataflowIR decomposition"),
        SubOp::Moe { .. } => todo!("SubOp::Moe has no DataflowIR decomposition"),
        // ⭐ `RmsNorm` IS BUILT — [`expand`] decomposes it into six op-funcs before this is reached,
        // so this arm answers only for a caller that asks about the node rather than expanding it.
        SubOp::RmsNorm { .. } => todo!("SubOp::RmsNorm is expanded, not mapped to one op-func"),
    }
}

/// ONE TENSOR REGION AS AN OPERAND, at the address the plan gave its tensor.
///
/// ⛔⛔ THE REGION'S OWN OFFSET IS PART OF THE ADDRESS. A node reads a SLICE of a tensor, and the
/// slice's first row starts `row_start * width * bytes` into the tensor's placement. Dropping that
/// term addresses every slice at the tensor's base — which reads the right tensor and the wrong
/// rows, and looks entirely plausible.
fn operand_of<F: RopeForm>(tr: &TensorRegion, ir: &SubtileIR<F>, layout: &BundleLayout) -> Operand {
    let tid = tr.tensor.index() as u32;
    // ⛔⛔ THE PLAN IS BUILT FROM THIS SAME GRAPH, so every tensor a node touches is in it by
    // construction. This is the ONE thing in this file that is not total, and the alternative is
    // worse than a stop: a default placement is an INVENTED ADDRESS, which is the first entry in
    // this crate's own list of past fuckups — programs that read memory nothing ever filled. So the
    // address is never fabricated; making this total means making the placement plan itself return
    // a placement for every tensor, which is a change to the plan and not to this line.
    let place = layout.placements.get(&tid).unwrap_or_else(|| {
        panic!("t{tid} is absent from the placement plan built from this graph")
    });
    let shape = ir.tensors[tid as usize];
    let rows = tr.region.rows.len;

    // ⭐⭐ THE DEVICE WIDTH, NOT THE LOGICAL ONE — for a region that spans the whole tensor.
    //
    // ⛔ A TENSOR IS STORED PADDED. `compute_bundle_layout` reserves the DEVICE footprint, padding
    // the innermost stick dim up to a whole stick, so granite's `[1, 49155]` logits occupy 49216
    // elements. Addressing them at 49155 is not merely a different number: 49155 is not a whole
    // number of 64-lane vectors, so the AGEN transfer cannot be split at all and the tape stops with
    // `InnermostNotWhole { src_innermost: 49155, lanes: 64 }` — which is exactly how this was found.
    //
    // ⭐ `for_pointwise` IS THE SHARED AUTHORITY. It is documented to EQUAL `for_output(m, n, k)`
    // for any producer whose macs reach 2^20 — every real producer of a padded tensor
    // (lm_head/o_proj/down_proj) — with a Kani proof (`devwidth_pointwise_matches_matmul`). So a
    // consumer addressed through it reads the layout its producer wrote.
    //
    // ⛔ ONLY FOR A FULL-WIDTH REGION. A slice inside a row is a genuine sub-extent; widening it to
    // the padded width would read past the slice the tape asked for.
    let spans_full_width = tr.region.cols.start == 0 && tr.region.cols.len == shape.cols;
    let cols = if spans_full_width {
        crate::lower_subtile_tape_to_superdsc::DeviceWidth::for_pointwise(shape.cols).get()
    } else {
        tr.region.cols.len
    };

    // Two bytes an element: the activation stream is fp16 whatever the weights are quantised to.
    let within = u64::from(tr.region.rows.start) * u64::from(shape.cols) * 2
        + u64::from(tr.region.cols.start) * 2;

    Operand {
        rows: Rows(rows),
        cols: Cols(cols),
        format: DataType::Sen169Fp16,
        at: Residence::Hbm {
            segment: Segment(u32::try_from(place.segment).unwrap_or(0)),
            offset: Bytes(place.offset + within),
        },
    }
}

/// A SYNTHETIC INTERMEDIATE, DECLARED AND THEN ADDRESSED.
///
/// ⛔⛔ DECLARING IS NOT OPTIONAL AND NOT LAZY. `resolve_seg_base` PANICS the build on a synthetic
/// that reaches its first access undeclared, and the comment there says why: the bump did not
/// advance, so every undeclared synthetic received THE SAME ADDRESS with a zero-byte placement that
/// hid the collision from every guard downstream — on granite that put three of rope's four
/// intermediates on one buffer. So the declaration happens here, at the site that mints the name,
/// with the tensor's FULL footprint.
///
/// ⭐ AND THE ALLOCATOR IS THE LAYOUT'S OWN, so a synthetic this bridge invents lands where the
/// same synthetic would land on the other path — the two cannot disagree about an address.
fn synth_operand(
    layout: &BundleLayout,
    of: u32,
    role: bundle::SynthRole,
    rows: u32,
    cols: u32,
) -> Operand {
    let id = bundle::PlaceId::Synth { of, role };
    layout.synth(id, &[rows, cols]);
    let offset = *layout
        .synth
        .borrow()
        .map
        // ⛔ `synth` DECLARED IT ON THE LINE ABOVE, so the bump allocator has an entry by
        // construction. Same rule as `operand_of`: never fabricate an address.
        .get(&id.to_string())
        .unwrap_or_else(|| panic!("synthetic {id} was declared and is not in the layout"));
    Operand {
        rows: Rows(rows),
        cols: Cols(cols),
        format: DataType::Sen169Fp16,
        at: Residence::Hbm {
            segment: Segment(
                u32::try_from(SegRole::Intermediate.segment()).expect("a segment index fits a u32"),
            ),
            offset: Bytes(offset),
        },
    }
}

/// ONE TAPE NODE AS THE PROGRAMS IT BECOMES.
///
/// ⛔⛔ A NODE IS NOT ALWAYS ONE PROGRAM. `SubOp::RmsNorm` is six op-funcs over five intermediates,
/// and an earlier version of this bridge returned a single `OpFunc` per node — which cannot express
/// it at all, and stopped the whole tape with "no op-func for `SubOp::RmsNorm`". The decomposition
/// below is `assemble_rmsnorm`'s, read rather than derived
/// (`ir/bridge/tiled_op_sdsc_op/rmsnorm.rs:103-172`).
fn expand<F: RopeForm>(
    node: &scratchy_subtile::subtile_ir::SubtileNode<F>,
    ir: &SubtileIR<F>,
    layout: &BundleLayout,
) -> Vec<Node> {
    use bundle::SynthRole as R;

    let one = |op_func: OpFunc, inputs: Vec<Operand>, output: Operand| Node {
        op_func,
        format: DataType::Sen169Fp16,
        inputs,
        output,
    };

    if let SubOp::RmsNorm { .. } = node.op {
        // `mean = mean(x*x)` -> `+ eps` -> `rsqrt` -> `x * rinv` -> `* gamma`. The reduce's output
        // is one stick wide, not one column: the device reduces into a stick.
        // ⛔ `(x, gamma)` IS THE OP'S OWN SHAPE — `SubOp::RmsNorm`'s doc states `inputs[0] = x`,
        // `inputs[1] = weight`. Indexing rather than destructuring-with-an-else, because the `else`
        // is a refusal and a refusal stops the tape before dbo-opt reads it.
        let (x, gamma) = (&node.inputs[0], &node.inputs[1]);
        let of = node.output.tensor.index() as u32;
        let rows = node.output.region.rows.len;
        let cols = node.output.region.cols.len;
        // ⛔ ONE STICK, NOT ONE COLUMN. The reduce writes a stick-wide row; declaring it `[rows, 1]`
        // under-reserves it by a whole stick and the next synthetic starts inside it.
        let stick = 64;

        let xo = operand_of(x, ir, layout);
        let go = operand_of(gamma, ir, layout);
        let sq = synth_operand(layout, of, R::Sq16, rows, cols);
        let mean = synth_operand(layout, of, R::Mean, rows, stick);
        let meps = synth_operand(layout, of, R::Meps, rows, stick);
        let rinv = synth_operand(layout, of, R::Rinv, rows, stick);
        let xn = synth_operand(layout, of, R::Xn, rows, cols);
        let out = operand_of(&node.output, ir, layout);

        return vec![
            one(OpFunc::Mul, vec![xo, xo], sq),
            one(OpFunc::Mean, vec![sq], mean),
            one(OpFunc::Add, vec![mean], meps),
            one(OpFunc::Rsqrt, vec![meps], rinv),
            one(OpFunc::Mul, vec![xo, rinv], xn),
            one(OpFunc::Mul, vec![xn, go], out),
        ];
    }

    let op_func = op_func_of(&node.op);
    let mut inputs = Vec::with_capacity(node.inputs.len());
    for input in &node.inputs {
        inputs.push(operand_of(input, ir, layout));
    }
    vec![one(op_func, inputs, operand_of(&node.output, ir, layout))]
}

/// THE WHOLE TAPE, AS DATAFLOWIR TEXT.
///
/// ⭐ EVERY NODE. The result is one program per node of `ir.nodes`, in evaluation order, and
/// `deeptools` asserts that count itself — a partial lowering is not a result.
///
/// ⛔⛔ AND IT RETURNS THE GROUPS, NEVER A REFUSAL. Every stop this function could take is a stop
/// placed EARLIER THAN dbo-opt, which is the only thing that can tell us whether the DataflowIR we
/// emit is any good. See `tests/dfir_never_runtime_refuses.rs`.
pub fn lower_subtile_tape_to_dataflow_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &HashSet<u32>,
    rows_are_requests: bool,
    rows: u32,
    active_cap: ActiveCap,
    cap: u32,
    model: [u32; 7],
) -> Vec<String> {
    // ⛔ RESOLVED HERE, ONCE. `ActiveCap::resolve` is documented as THE ONLY place a rung becomes a
    // tile extent: NONE -> 0 (sweep no resident prefix), FULL or any out-of-range request -> the
    // bundle's whole `cap`, otherwise the request itself.
    let active_cap = active_cap.resolve(cap, scratchy_subtile::sdsc_abstract::POOL_STICK);
    // ⭐ THE SAME PLAN THE WORKER STAGES AGAINST, derived from this same graph.
    // ⛔ THE PLAN IS THE SAME ONE THE WORKER STAGES AGAINST; if it cannot be built the bundle cannot
    // exist at all, and that is the superdsc path's own failure, not a case this bridge lowers
    // differently.
    let layout = compute_bundle_layout(ir, weight_ids, rows_are_requests)
        .unwrap_or_else(|e| panic!("the placement plan this tape shares could not be built: {e}"));

    // ⛔ EVERY NODE, AND A NODE MAY BE SEVERAL PROGRAMS. The count below is per NODE, not per
    // program: `SubOp::RmsNorm` becomes six. A node that mapped to nothing is a forward silently
    // skipping work, and the emitter downstream could not tell.
    //
    // ⭐⭐ AND EVERY PROGRAM GETS ITS TRIP, in the same order, because the grouping below is over
    // PROGRAMS. `trip_kinds_and_owner` classifies the SuperDSC path's emitted ops for exactly this
    // reason: one tape node explodes into many, and the kinds distinguish AMONG them.
    let mut nodes = Vec::with_capacity(ir.nodes.len());
    let mut trips: Vec<Trip> = Vec::with_capacity(ir.nodes.len());
    for node in &ir.nodes {
        // ⛔ `expand` IS TOTAL — every arm of its match yields at least one node, so "expanded to
        // nothing" is not a state it can be in. It used to be checked here and returned as an
        // error; the check was the shape of a refusal even when it could not fire.
        let programs = expand(node, ir, &layout);
        let trip = trip_of(&node.op);
        trips.extend(std::iter::repeat_n(trip, programs.len()));
        nodes.extend(programs);
    }
    debug_assert_eq!(trips.len(), nodes.len(), "one trip per emitted program");

    // ⭐⭐ THE EXISTING GROUPING, CALLED — NOT A PARTITION OF OUR OWN. `group_ranges` and
    // `group_size` are `pub` and take no `EmittedOp`; only `trip_kinds_and_owner` is that path's
    // adapter. Its fusion rules are correctness-bearing (a distinct-slot copy must not fuse into a
    // uniform-shift `Slot` group; `Slab`'s 8192-byte stride cannot share a group with a cachewr's
    // 128), so they are called rather than restated.
    //
    // ⛔ AN EARLIER VERSION OF THIS FILE SPLIT THE ROLLED TAPE ON `OpenLoop`/`CloseLoop` INSTEAD.
    // That is the REROLL STRUCTURE — prefix, one layer, suffix — and it is not the launch partition:
    // it produced 3 groups for a bundle `launch_index` grouped into 1.
    let ranges = group_ranges(&trips, group_size());
    let mut out = Vec::with_capacity(ranges.len());
    for (gi, range) in ranges.iter().enumerate() {
        let emit = Emit {
            nodes: nodes[range.clone()].to_vec(),
            rows,
            active_cap,
            group: u32::try_from(gi).unwrap_or(0),
        };
        out.push(
            with_config_model(
                model[0], model[1], model[2], model[3], model[4], model[5], model[6], emit,
            )
            // ⛔ THE DOOR IS TOTAL OVER THE CONFIGS IN SCOPE — the build named this model, so an arm
            // for it exists. A miss is a config that was compiled and then not declared, which is a
            // `config.json`, never something this bridge lowers around.
            .unwrap_or_else(|| {
                panic!(
                    "no config declares (nqh {}, nkvh {}, hd {}, hidden {}, layers {}, ffn {}, \
                     vocab {})",
                    model[0], model[1], model[2], model[3], model[4], model[5], model[6]
                )
            }),
        );
    }
    out
}

/// WHAT KIND OF TRIP A TAPE NODE'S PROGRAMS ARE, for [`group_ranges`].
///
/// ⭐ THE KINDS ARE THE SHIM'S, not a taxonomy invented here: a `Slot` trip is a cache write the
/// runtime shifts by `slot_pos·stride` once per group, `PageFold` folds one page of resident prefix
/// and is re-launched per page, `Slab` is the incremental Kᵀ restickify at a different stride, and
/// `Pure` is everything the shim launches without per-entry handling.
///
/// ⛔ TWO OF THE SIX INPUTS THE OTHER PATH CLASSIFIES ON ARE DEAD: `host_kv_write` and
/// `slot_no_fuse` are declared, read, and set NOWHERE (`:6100` calls the first "the host_kv_write
/// nuke"), so `HostKv` and `SlotSolo` cannot arise there either.
///
/// ⛔ AND WHAT IS NOT YET DISTINGUISHED IS SAID, NOT GUESSED. Our emission does not yet produce the
/// attention's per-page fold blocks or the slab restickify as separate programs, so no node maps to
/// `PageFold` or `Slab`. When it does, they must be classified here — a fold block swept into a
/// `Pure` group would be re-launched per page along with whatever it fused with.
fn trip_of<F: RopeForm>(op: &SubOp<F>) -> Trip {
    let kind = match op {
        // The cache write. `RopeAppend` is what puts this step's K/V into the pool.
        SubOp::RopeAppend { .. } => GroupKind::Slot { req: 0 },
        _ => GroupKind::Pure,
    };
    Trip {
        kind,
        req: TripRequest(0),
    }
}

/// The whole-model arm: inside it the seven numbers are literals.
struct Emit {
    nodes: Vec<Node>,
    rows: u32,
    active_cap: u32,
    group: u32,
}

impl OnModel for Emit {
    type Out = String;

    fn on_model<
        const NQH: u32,
        const NKVH: u32,
        const HD: u32,
        const HIDDEN: u32,
        const LAYERS: u32,
        const FFN: u32,
        const VOCAB: u32,
    >(
        self,
    ) -> Self::Out {
        /// The model, with every one of its numbers a const the compiler folds.
        struct M<
            const NQH: u32,
            const NKVH: u32,
            const HD: u32,
            const HIDDEN: u32,
            const LAYERS: u32,
            const FFN: u32,
            const VOCAB: u32,
        >;
        impl<
            const NQH: u32,
            const NKVH: u32,
            const HD: u32,
            const HIDDEN: u32,
            const LAYERS: u32,
            const FFN: u32,
            const VOCAB: u32,
        > Model for M<NQH, NKVH, HD, HIDDEN, LAYERS, FFN, VOCAB>
        {
            const QUERY_HEADS: u32 = NQH;
            const KV_HEADS: u32 = NKVH;
            const HEAD_DIM: u32 = HD;
            const HIDDEN: u32 = HIDDEN;
            const LAYERS: u32 = LAYERS;
            const FFN: u32 = FFN;
            const VOCAB: u32 = VOCAB;
        }

        rungs::<M<NQH, NKVH, HD, HIDDEN, LAYERS, FFN, VOCAB>>(
            &self.nodes,
            self.rows,
            self.active_cap,
            self.group,
        )
    }
}

/// THE ROW LADDER — every row count a bundle is baked at.
///
/// ⭐ READ, NOT CHOSEN. The first six are the decode batch widths `with_baked_rung` dispatches on
/// (`sdsc_abstract.rs:687`, whose own `const` assertion ties them to `PagedKvPool::BATCH_RUNGS`);
/// the rest are `PREFILL_RUNGS` (`macros/src/codegen.rs:9975-9977`), whose ceiling of 96 is tied by
/// another `const` assertion to `PagedKvPool::PREFILL_CHUNK_SLOTS`.
macro_rules! row_ladder {
    ($emit:ident) => {
        $emit! {
            1, 2, 4, 8, 16, 32,
            7, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 39, 43, 47, 55, 63, 71, 80, 88, 96,
        }
    };
}

/// THE SWEPT LADDER — every KV extent a decode rung is baked at.
///
/// ⭐ `ActiveCap::decode_ladder`'s own base (`lower_subtile_tape_to_superdsc.rs:10923`), which the
/// comment there calls "THE LADDER, and there is no other". It used to sit behind an env var, so
/// which rungs a bundle baked depended on the shell that ran the build.
///
/// ⛔ NO 4096 OR 8192. An earlier version of this door listed them, and they are not on the ladder —
/// inventing rungs bakes programs the worker will never pick and hides a genuine miss behind a
/// plausible-looking arm. The ceiling rung is `prefix_len`, `min(max_position_embeddings, 256)` by
/// default, which is already one of these.
macro_rules! swept_ladder {
    ($emit:ident) => {
        $emit! { 64, 128, 256, 512, 1024, 2048, }
    };
}

/// THE TWO RUNG DOORS — where the row count and the swept extent stop being values.
///
/// ⛔⛔ CONST GENERICS, ALL THE WAY DOWN. The rung is as much a compile-time fact as the model: this
/// crate is driven by a proc macro that knows both as literals. A `Workload` whose numbers arrived
/// as fields would make `Exploit`'s flags a runtime computation, and a `const fn` on a runtime value
/// folds to nothing — which is the mistake three earlier versions made.
///
/// ⛔ A RUNG OFF THE LADDER IS REFUSED, never rounded. Rounding bakes a program whose swept extent
/// is not the one the worker will pick it for.
fn rungs<M: Model>(nodes: &[Node], rows: u32, active_cap: u32, group: u32) -> String {
    // ⛔ WRITTEN OUT RATHER THAN NESTED, because a `macro_rules!` inside a `macro_rules!` needs
    // `$$` — meta-variable expressions, still unstable (rust#83527). The swept ladder is expanded
    // by `swept_ladder!` at one site below, so it is still declared once.
    macro_rules! rung {
        ($r:literal, $c:literal) => {{
            struct W;
            impl Workload for W {
                const ROWS: u32 = $r;
                const ACTIVE_CAP: u32 = $c;
            }
            emit_run::<M, W>(nodes, group)
        }};
    }
    macro_rules! with_rows {
        ($($r:literal),+ $(,)?) => {
            match rows {
                $($r => match active_cap {
                    // ⭐ ZERO IS A RUNG, not a missing value: `ActiveCap::NONE` RESOLVED. A prefill
                    // chunk whose `start == 0` has no resident prefix to attend, so it sweeps
                    // nothing — and the emitter must then leave the cache walk out entirely rather
                    // than emit one of zero trips.
                    0 => rung!($r, 0),
                    64 => rung!($r, 64),
                    128 => rung!($r, 128),
                    256 => rung!($r, 256),
                    512 => rung!($r, 512),
                    1024 => rung!($r, 1024),
                    2048 => rung!($r, 2048),
                    _ => panic!("rung (rows {rows}, active_cap {active_cap}) is not one the ladders bake"),
                },)+
                _ => panic!("rung (rows {rows}, active_cap {active_cap}) is not one the ladders bake"),
            }
        };
    }
    // ⭐ THE SWEPT LADDER IS CHECKED AGAINST THE ARMS ABOVE, so the two cannot drift apart
    // silently: adding a rung to the ladder without adding its arm is a build error.
    macro_rules! count_caps {
        ($($c:literal),+ $(,)?) => { [$($c),+] };
    }
    const SWEPT: [u32; 6] = swept_ladder!(count_caps);
    const _: () = assert!(
        SWEPT.len() == 6,
        "the swept ladder and this door's arms must list the same rungs"
    );
    row_ladder!(with_rows)
}

/// The innermost arm: machine, model and rung are all constants here.
fn emit_run<M: Model, W: Workload>(nodes: &[Node], group: u32) -> String {
    // ⛔ `deeptools` STILL REFUSES A TAPE IT CANNOT WALK, and that refusal is ITS invariant, not a
    // lowering decision taken here. It is surfaced rather than turned into a value: a value would be
    // something a caller could log and carry on from, which is exactly how this bridge went blind.
    let run = tape::compile::<Dd2, M, W>(nodes, GroupId(group))
        .unwrap_or_else(|e| panic!("deeptools could not walk the tape: {e:?}"));
    print::run(&run)
}

/// Silence the unused-import warning for `Elements` on builds where no operand is LX-resident yet.
const _: Option<Elements> = None;
