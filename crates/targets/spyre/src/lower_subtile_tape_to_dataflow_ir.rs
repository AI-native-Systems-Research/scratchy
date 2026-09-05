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

use crate::lower_subtile_tape_to_superdsc::{BundleLayout, compute_bundle_layout};

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
/// ⭐ THE TAPE HAS ALREADY DECOMPOSED THE FUSED OPS. `RmsNorm` arrives as its reduce and apply
/// halves, rope as its rotate and append — that is what `decompose_rmsnorm` and `head_tile_rope`
/// are for — so this map is mostly one-to-one rather than a fusion table.
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
    let cols = tr.region.cols.len;

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
    active_cap: u32,
    model: [u32; 7],
    group: u32,
) -> Result<String, DfirError> {
    // ⭐ THE SAME PLAN THE WORKER STAGES AGAINST, derived from this same graph.
    let layout = compute_bundle_layout(ir, weight_ids, rows_are_requests)
        .map_err(|e| DfirError::Layout(e.to_string()))?;

    let mut nodes = Vec::with_capacity(ir.nodes.len());
    for node in &ir.nodes {
        let op_func = op_func_of(&node.op)?;
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            inputs.push(operand_of(input, ir, &layout)?);
        }
        nodes.push(Node {
            op_func,
            format: DataType::Sen169Fp16,
            inputs,
            output: operand_of(&node.output, ir, &layout)?,
        });
    }

    // ⛔ EVERY NODE OF THE TAPE IS PRESENT before anything is emitted. A short list here would be a
    // forward that silently skips work, and the emitter could not tell.
    if nodes.len() != ir.nodes.len() {
        return Err(DfirError::Tape(format!(
            "the graph has {} nodes but {} were mapped",
            ir.nodes.len(),
            nodes.len()
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
