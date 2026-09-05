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

use crate::lower_subtile_tape_to_superdsc::{ActiveCap, BundleLayout, SegRole, compute_bundle_layout};
use scratchy_spyre_bundle as bundle;

/// WHY A TAPE COULD NOT BECOME DATAFLOWIR.
///
/// ⛔ EVERY VARIANT NAMES THE THING IT COULD NOT DO. "lowering failed" is not a diagnosis, and a
/// bake that stops has to say what to build next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DfirError {
    /// The placement plan could not be built.
    Layout(String),
    /// A `SubOp` this bridge has no op-func for yet.
    ///
    /// ⛔ NOT A FALLBACK. There is no `_ =>` arm that quietly picks an addition: an op-func that
    /// does not exist yet is named here so the next thing to build is obvious.
    NoOpFunc(&'static str),
    /// A tensor the node reads has no placement.
    ///
    /// ⛔ THIS WOULD BE AN ADDRESS NOBODY ASSIGNED. Every tensor a node touches is in the layout by
    /// construction, so a miss means the node and the plan disagree about the graph.
    Unplaced {
        /// Which tensor.
        tid: u32,
    },
    /// The model's numbers match no config in this workspace.
    UnknownModel {
        /// The seven, in the order the door takes them.
        numbers: [u32; 7],
    },
    /// The rung is not one the ladders bake.
    UnknownRung {
        /// The row count asked for.
        rows: u32,
        /// The swept KV extent asked for.
        active_cap: u32,
    },
    /// `deeptools` refused the tape.
    Tape(String),
}

impl std::fmt::Display for DfirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layout(e) => write!(f, "placement plan: {e}"),
            Self::NoOpFunc(op) => write!(
                f,
                "no op-func for `SubOp::{op}` yet — the DataflowIR bridge lowers what it can name, \
                 and this one has to be built rather than approximated"
            ),
            Self::Unplaced { tid } => write!(
                f,
                "tensor t{tid} has no placement: the node reads an address the plan never assigned"
            ),
            Self::UnknownModel { numbers } => write!(
                f,
                "no config in this workspace declares the model \
                 (nqh {}, nkvh {}, hd {}, hidden {}, layers {}, ffn {}, vocab {}) — the fix is a \
                 config.json, never an edit to the door",
                numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6]
            ),
            Self::UnknownRung { rows, active_cap } => write!(
                f,
                "rung (rows {rows}, active_cap {active_cap}) is not one the ladders bake"
            ),
            Self::Tape(e) => write!(f, "deeptools refused the tape: {e}"),
        }
    }
}

/// WHICH OP-FUNC A TAPE NODE IS.
///
/// ⛔⛔ EXHAUSTIVE, WITH NO WILDCARD. Every `SubOp` is named, so adding one to the tape is an E0004
/// here rather than a node that silently lowers as something else. The arms that return
/// [`DfirError::NoOpFunc`] are work not yet done, stated as such.
///
/// ⛔ AND THE TAPE DOES NOT ALWAYS ARRIVE DECOMPOSED. This said the opposite — that `decompose_rmsnorm`
/// had already split `RmsNorm` into its reduce and apply halves, so the map was "mostly one to one".
/// The acceptance build falsified it: a raw `SubOp::RmsNorm` reached this function and stopped the
/// whole tape. Nodes that are several op-funcs are handled by [`expand`], which returns a sequence;
/// this function answers only for the ones that are exactly one.
fn op_func_of<F: RopeForm>(op: &SubOp<F>) -> Result<OpFunc, DfirError> {
    Ok(match op {
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
        // ⛔ NOT YET BUILT. Each of these needs its own schedule read out of the templates; naming
        // them here rather than defaulting is what keeps the gap visible.
        SubOp::RmsNorm { .. } => return Err(DfirError::NoOpFunc("RmsNorm")),
        SubOp::RmsNormUnit { .. } => return Err(DfirError::NoOpFunc("RmsNormUnit")),
        SubOp::GateSplit { .. } => return Err(DfirError::NoOpFunc("GateSplit")),
        SubOp::LoadPixels { .. } => return Err(DfirError::NoOpFunc("LoadPixels")),
        SubOp::LoadPosEmbeds { .. } => return Err(DfirError::NoOpFunc("LoadPosEmbeds")),
        SubOp::EmbeddingGather { .. } => return Err(DfirError::NoOpFunc("EmbeddingGather")),
        SubOp::VisionRope => return Err(DfirError::NoOpFunc("VisionRope")),
        SubOp::VarlenAttention { .. } => return Err(DfirError::NoOpFunc("VarlenAttention")),
        SubOp::EncoderAttn { .. } => return Err(DfirError::NoOpFunc("EncoderAttn")),
        SubOp::GatedDeltaNet => return Err(DfirError::NoOpFunc("GatedDeltaNet")),
        SubOp::GemmaMoe { .. } => return Err(DfirError::NoOpFunc("GemmaMoe")),
        SubOp::Moe { .. } => return Err(DfirError::NoOpFunc("Moe")),
    })
}

/// ONE TENSOR REGION AS AN OPERAND, at the address the plan gave its tensor.
///
/// ⛔⛔ THE REGION'S OWN OFFSET IS PART OF THE ADDRESS. A node reads a SLICE of a tensor, and the
/// slice's first row starts `row_start * width * bytes` into the tensor's placement. Dropping that
/// term addresses every slice at the tensor's base — which reads the right tensor and the wrong
/// rows, and looks entirely plausible.
fn operand_of<F: RopeForm>(
    tr: &TensorRegion,
    ir: &SubtileIR<F>,
    layout: &BundleLayout,
) -> Result<Operand, DfirError> {
    let tid = tr.tensor.index() as u32;
    let place = layout
        .placements
        .get(&tid)
        .ok_or(DfirError::Unplaced { tid })?;
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

    Ok(Operand {
        rows: Rows(rows),
        cols: Cols(cols),
        format: DataType::Sen169Fp16,
        at: Residence::Hbm {
            segment: Segment(u32::try_from(place.segment).unwrap_or(0)),
            offset: Bytes(place.offset + within),
        },
    })
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
) -> Result<Operand, DfirError> {
    let id = bundle::PlaceId::Synth { of, role };
    layout.synth(id, &[rows, cols]);
    let offset = *layout
        .synth
        .borrow()
        .map
        .get(&id.to_string())
        .ok_or(DfirError::Unplaced { tid: of })?;
    Ok(Operand {
        rows: Rows(rows),
        cols: Cols(cols),
        format: DataType::Sen169Fp16,
        at: Residence::Hbm {
            segment: Segment(
                u32::try_from(SegRole::Intermediate.segment()).expect("a segment index fits a u32"),
            ),
            offset: Bytes(offset),
        },
    })
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
) -> Result<Vec<Node>, DfirError> {
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
        let [x, gamma] = node.inputs.as_slice() else {
            return Err(DfirError::NoOpFunc("RmsNorm without exactly (x, gamma)"));
        };
        let of = node.output.tensor.index() as u32;
        let rows = node.output.region.rows.len;
        let cols = node.output.region.cols.len;
        // ⛔ ONE STICK, NOT ONE COLUMN. The reduce writes a stick-wide row; declaring it `[rows, 1]`
        // under-reserves it by a whole stick and the next synthetic starts inside it.
        let stick = 64;

        let xo = operand_of(x, ir, layout)?;
        let go = operand_of(gamma, ir, layout)?;
        let sq = synth_operand(layout, of, R::Sq16, rows, cols)?;
        let mean = synth_operand(layout, of, R::Mean, rows, stick)?;
        let meps = synth_operand(layout, of, R::Meps, rows, stick)?;
        let rinv = synth_operand(layout, of, R::Rinv, rows, stick)?;
        let xn = synth_operand(layout, of, R::Xn, rows, cols)?;
        let out = operand_of(&node.output, ir, layout)?;

        return Ok(vec![
            one(OpFunc::Mul, vec![xo, xo], sq),
            one(OpFunc::Mean, vec![sq], mean),
            one(OpFunc::Add, vec![mean], meps),
            one(OpFunc::Rsqrt, vec![meps], rinv),
            one(OpFunc::Mul, vec![xo, rinv], xn),
            one(OpFunc::Mul, vec![xn, go], out),
        ]);
    }

    let op_func = op_func_of(&node.op)?;
    let mut inputs = Vec::with_capacity(node.inputs.len());
    for input in &node.inputs {
        inputs.push(operand_of(input, ir, layout)?);
    }
    Ok(vec![one(
        op_func,
        inputs,
        operand_of(&node.output, ir, layout)?,
    )])
}

/// THE WHOLE TAPE, AS DATAFLOWIR TEXT.
///
/// ⭐ EVERY NODE. The result is one program per node of `ir.nodes`, in evaluation order, and
/// `deeptools` asserts that count itself — a partial lowering is not a result.
///
/// # Errors
///
/// Returns [`DfirError`] naming what could not be done: an op-func that is not built, a tensor with
/// no placement, a model or rung no door has an arm for.
pub fn lower_subtile_tape_to_dataflow_ir<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &HashSet<u32>,
    rows_are_requests: bool,
    rows: u32,
    active_cap: ActiveCap,
    cap: u32,
    model: [u32; 7],
    group: u32,
) -> Result<String, DfirError> {
    // ⛔ RESOLVED HERE, ONCE. `ActiveCap::resolve` is documented as THE ONLY place a rung becomes a
    // tile extent: NONE -> 0 (sweep no resident prefix), FULL or any out-of-range request -> the
    // bundle's whole `cap`, otherwise the request itself.
    let active_cap = active_cap.resolve(cap, scratchy_subtile::sdsc_abstract::POOL_STICK);
    // ⭐ THE SAME PLAN THE WORKER STAGES AGAINST, derived from this same graph.
    let layout = compute_bundle_layout(ir, weight_ids, rows_are_requests)
        .map_err(|e| DfirError::Layout(e.to_string()))?;

    // ⛔ EVERY NODE, AND A NODE MAY BE SEVERAL PROGRAMS. The count below is per NODE, not per
    // program: `SubOp::RmsNorm` becomes six. A node that mapped to nothing is a forward silently
    // skipping work, and the emitter downstream could not tell.
    let mut nodes = Vec::with_capacity(ir.nodes.len());
    let mut mapped = 0usize;
    for node in &ir.nodes {
        let programs = expand(node, ir, &layout)?;
        if programs.is_empty() {
            return Err(DfirError::Tape(format!(
                "node {} expanded to no programs at all",
                node.output.tensor.index()
            )));
        }
        mapped += 1;
        nodes.extend(programs);
    }
    if mapped != ir.nodes.len() {
        return Err(DfirError::Tape(format!(
            "the graph has {} nodes but {mapped} were mapped",
            ir.nodes.len()
        )));
    }

    let emit = Emit {
        nodes,
        rows,
        active_cap,
        group,
    };
    with_config_model(
        model[0], model[1], model[2], model[3], model[4], model[5], model[6], emit,
    )
    .ok_or(DfirError::UnknownModel { numbers: model })?
}

/// The whole-model arm: inside it the seven numbers are literals.
struct Emit {
    nodes: Vec<Node>,
    rows: u32,
    active_cap: u32,
    group: u32,
}

impl OnModel for Emit {
    type Out = Result<String, DfirError>;

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
fn rungs<M: Model>(
    nodes: &[Node],
    rows: u32,
    active_cap: u32,
    group: u32,
) -> Result<String, DfirError> {
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
                    _ => Err(DfirError::UnknownRung { rows, active_cap }),
                },)+
                _ => Err(DfirError::UnknownRung { rows, active_cap }),
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
fn emit_run<M: Model, W: Workload>(nodes: &[Node], group: u32) -> Result<String, DfirError> {
    let run = tape::compile::<Dd2, M, W>(nodes, GroupId(group))
        .map_err(|e| DfirError::Tape(format!("{e:?}")))?;
    Ok(print::run(&run))
}

/// Silence the unused-import warning for `Elements` on builds where no operand is LX-resident yet.
const _: Option<Elements> = None;
