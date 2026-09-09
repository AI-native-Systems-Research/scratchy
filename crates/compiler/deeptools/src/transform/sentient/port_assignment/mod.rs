// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `PortAssignment.cpp` — 13 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e123_clean` | 123 | 0 | 12 | `dcc/src/Transform/Sentient/PortAssignment.cpp:80` |
//! | `e124_getPortAttr` | 124 | 0 | 4 | `dcc/src/Transform/Sentient/PortAssignment.cpp:206` |
//! | `e125_getValidPorts` | 125 | 0 | 254 | `dcc/src/Transform/Sentient/PortAssignment.cpp:214` |
//! | `e126_IsSwappablePerAlgebraicReassociation` | 126 | 0 | 29 | `dcc/src/Transform/Sentient/PortAssignment.cpp:606` |
//! | `e339_addNodesToGraph` | 339 | 1 | 131 | `dcc/src/Transform/Sentient/PortAssignment.cpp:473` |
//! | `e340_updateSentientIRPorts` | 340 | 1 | 52 | `dcc/src/Transform/Sentient/PortAssignment.cpp:682` |
//! | `e341_addReuseToDummyOperands` | 341 | 1 | 36 | `dcc/src/Transform/Sentient/PortAssignment.cpp:739` |
//! | `e455_buildGraphNodes` | 455 | 2 | 6 | `dcc/src/Transform/Sentient/PortAssignment.cpp:636` |
//! | `e520_processDataID` | 520 | 3 | 67 | `dcc/src/Transform/Sentient/PortAssignment.cpp:93` |
//! | `e572_computePortLiveRange` | 572 | 4 | 40 | `dcc/src/Transform/Sentient/PortAssignment.cpp:165` |
//! | `e608_buildGraphEdges` | 608 | 5 | 37 | `dcc/src/Transform/Sentient/PortAssignment.cpp:644` |
//! | `e632_doPortAssignments` | 632 | 6 | 12 | `dcc/src/Transform/Sentient/PortAssignment.cpp:776` |
//! | `e645_runOnOperation` | 645 | 7 | 21 | `dcc/src/Transform/Sentient/PortAssignment.cpp:793` |

// ⛔ NINE OF THE THIRTEEN ANCHORS BELOW ARE UNFILLED, so the pass state has no writer and
// `getPortAttr` no caller until they land; CI runs clippy with `-D warnings`
// (the `lexical_ordering/mod.rs:83` precedent).
// ⭐ REMOVE THIS WITH `e645_runOnOperation`.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::analyses::{ColoringGraph, LiveRange};
use crate::arch::{Arch, IsaGen};
use crate::islands::sentient::dialects::sentient::{BinaryOp, Port, Precision, TernaryOp, UnaryOp};
use crate::islands::sentient::dialects::{Op, sentient};
use crate::units::DfirUnit;

/// WHICH OF THE THREE COMPUTE PORTS — the count is fixed at `performGraphColoring(3)` (`:784`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PortId {
    P0,
    P1,
    P2,
}

impl PortId {
    /// The `si32` an `opXPortID` attribute carries (`:208`).
    pub(crate) const fn get(self) -> i32 {
        match self {
            Self::P0 => 0,
            Self::P1 => 1,
            Self::P2 => 2,
        }
    }
}

/// WHICH OPERAND VALUE — `getOpADataID()` and friends, which is also a colouring node's `index_`
/// (`getOrAddNode(getOpXDataID())`, `:476-478`).
///
/// ⛔ ABSENCE IS THE REFERENCE'S `-1`: `processDataID` returns early on `data_id < 0` (`:96`), and the
/// island already spells it as `Operand::data_id: None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct DataId(pub(crate) u32);

/// WHERE AN OP SITS IN THE UNIT'S PRE-ORDER WALK — the `Operation *` identity two of the maps key on
/// (the [`super::loop_merging::InBlock`] precedent: a position, never a borrow into the tree being
/// rewritten).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OpAt(pub(crate) u32);

/// THE ORDER A COMPUTE OP TAKES ITS PORTS IN — `op_to_index_`'s value, the instruction number the
/// port live ranges are expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct InstrIndex(pub(crate) u32);

/// WHICH MAC OPERAND A REASSOCIATION MAY MOVE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MacOperand {
    /// `$opA`.
    A,
    /// `$opB`.
    B,
    /// `$opC`.
    C,
}

/// WHY NO PORT COULD BE ASSIGNED — one variant per distinct `emitError` message in `getValidPorts`,
/// each of which is followed by `signalPassFailure()` and ends the compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PortRejection {
    /// "invalid op argument in PT int" (`:259`).
    InvalidMacArgumentInPtInt,
    /// "invalid op argument in PT fp" (`:305`).
    InvalidMacArgumentInPtFp,
    /// "No SFP Ring in other components" (`:352`, `:418` — the two sites share one message).
    NoSfpRingInOtherComponents,
    /// "invalid op argument in PE/SFP" (`:361`).
    InvalidMacArgumentInPeSfp,
    /// "invalid compute unit for mac op" (`:367`).
    InvalidComputeUnitForMac,
    /// "cannot have 1.0 in binaryOp except in FMUL case" (`:410`).
    OneInBinaryOpOutsideFmul,
    /// "invalid compute unit for binary op" (`:428`).
    InvalidComputeUnitForBinary,
    /// "invalid compute unit for unary op" (`:441`).
    InvalidComputeUnitForUnary,
    /// "invalid compute unit for ternary op" (`:460`).
    InvalidComputeUnitForTernary,
    /// "unsupported op" (`:466`).
    UnsupportedOp,
}

/// THE ANSWER `getValidPorts` GIVES.
///
/// ⛔ `Ports(vec![])` AND `Rejected` ARE BOTH `return {}` IN THE REFERENCE AND ARE NOT THE SAME
/// THING: an out-of-range `lrf` is its own "unable to assign a port" and the pass carries on with an
/// unconstrained node, while every `emitError` arm fails the pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValidPorts {
    /// The ports this operand may sit on, in the reference's own order.
    Ports(Vec<PortId>),
    /// The pass is over.
    Rejected(PortRejection),
}

fn ports(list: &[PortId]) -> ValidPorts {
    ValidPorts::Ports(list.to_vec())
}

/// `PortAssignmentPass`'s eight-slot state (`:45-55`), generic over the out-of-scope colouring graph.
#[derive(Debug, Clone, Default)]
pub(crate) struct PortAssignment<G: ColoringGraph> {
    /// `port_assignment_graph_`.
    pub(crate) graph: G,
    /// `dummy_nodes_` — nodes used nowhere, which may take any port.
    pub(crate) dummy_nodes: Vec<DataId>,
    /// `reuse_nodes_` — nodes reused in another instruction.
    pub(crate) reuse_nodes: Vec<DataId>,
    /// `reuse_nodes_per_op_`.
    pub(crate) reuse_nodes_per_op: Vec<Vec<DataId>>,
    /// `port_assignment_` — the colouring's answer, and what `getPortAttr` reads.
    pub(crate) port_assignment: BTreeMap<DataId, PortId>,
    /// `op_to_index_`.
    pub(crate) op_to_index: BTreeMap<OpAt, InstrIndex>,
    /// `operand_to_owners_` — every op that names this operand value, first owner first.
    pub(crate) operand_to_owners: BTreeMap<DataId, Vec<OpAt>>,
    /// `operand_to_liverange_`.
    pub(crate) operand_to_liverange: BTreeMap<DataId, LiveRange>,
}

impl<G: ColoringGraph> PortAssignment<G> {
    /// Replaces: e123_clean
    ///
    /// Empties the eight slots between units.
    /// ⛔ NOT A RE-CONSTRUCTION: `port_assignment_graph_.clear()` deliberately leaves its
    /// `same_color_edges_` and `max_node_id_` standing (`Analyses/GraphColoring.hpp:52-60`), which is
    /// why the graph is a [`ColoringGraph`] seam rather than `G::default()`.
    /// ⭐ THE TWO INNER LOOPS (`:82`, `:84`) ARE NO-OPS: `for (auto op_vec : ...)` takes each entry by
    /// value and clears the copy, so only the outer `.clear()` has any effect.
    pub(crate) fn clean(&mut self) {
        self.op_to_index.clear();
        self.operand_to_owners.clear();
        self.reuse_nodes_per_op.clear();
        self.operand_to_liverange.clear();
        self.graph.clear();
        self.dummy_nodes.clear();
        self.reuse_nodes.clear();
        self.port_assignment.clear();
    }
}

impl<G: ColoringGraph> PortAssignment<G> {
    /// Replaces: e124_getPortAttr
    ///
    /// The port the colouring gave one operand value — the `opXPortID` attribute's whole content.
    pub(crate) fn port_attr(&self, op_id: DataId) -> PortId {
        match self.port_assignment.get(&op_id) {
            Some(port) => *port,
            // `port_assignment_.at(opID)` throws for an operand the colouring never assigned.
            None => todo!(
                "getPortAttr: port_assignment_.at({op_id:?}) — the graph colouring left this \
                 operand value uncoloured (PortAssignment.cpp:207)"
            ),
        }
    }
}

/// Replaces: e125_getValidPorts
///
/// The ISA's port constraint: which of the three ports an operand may sit on, given the op, the
/// component, the operand's own precision and the op's compute precision.
/// ⛔ THE REFERENCE DISPATCHES ON `StringRef::contains` AND THE SUBSTRING MATCHES ARE LOAD-BEARING,
/// so this keeps `Port::spelling()`: `none` contains `one`, `crossptnlink` contains `pt` (so in
/// PE/SFP it never reaches its own arm), `sfpring` contains `sfp` (so the MAC PE/SFP `sfpring` arm is
/// DEAD), and `mxint4`/`mxfp8`/`ieee_fp16` take the `int`/`fp8`/`fp16` precision branches.
/// ⛔ [`IsaGen`] MODELS TWO OF `IsaCoreGen`'s FIVE, so `arch >= RCUDD1A_ISA` always holds — the PT/int
/// `lrf0..3` `{0}` arm (`:246`) is dead and `DT_CHECK_MSG(arch < RCUDD1A_ISA)` (`:248`) for
/// `lrf4/5` can never hold. On SEN1P5 the binary `zero`/`one`/`sfpring` arms are dead behind the
/// `arch == SEN1P5_ISA`
/// branch, and `abs_min` is deliberately absent from its min/max/fcmp group.
/// ⭐ A PT MAC computing in `fp24`, `fp32` or `none` matches neither precision branch and falls out
/// to an empty answer with no error (`:470`).
pub(crate) fn valid_ports<A: Arch>(
    op: &Op,
    operand_value: Port,
    comp: DfirUnit,
    operand_precision: Precision,
    precision: Precision,
) -> ValidPorts {
    let value = operand_value.spelling();
    let prec = precision.spelling();
    match op {
        Op::Sentient(sentient::Op::VectorMac { .. }) => {
            if matches!(comp, DfirUnit::PtRow(_)) {
                if prec.contains("int") {
                    if value.contains("west") {
                        match A::GEN {
                            IsaGen::Sen1p5 => ports(&[PortId::P1, PortId::P2]),
                            IsaGen::Rcudd1a => ports(&[PortId::P2]),
                        }
                    } else if value.contains("north") {
                        ports(&[PortId::P1])
                    } else if value.contains("crossptnlink") {
                        match A::GEN {
                            IsaGen::Sen1p5 => ports(&[PortId::P1]),
                            IsaGen::Rcudd1a => todo!(
                                "getValidPorts: Cross PT north link only present in sen 1.5 \
                                 (PortAssignment.cpp:233)"
                            ),
                        }
                    } else if value.contains("zero") {
                        ports(&[PortId::P0, PortId::P1])
                    } else if value.contains("one") {
                        ports(&[PortId::P2])
                    } else if let Port::Lrf(lrf) = operand_value {
                        match lrf.get() {
                            0..=3 => ports(&[PortId::P1]),
                            4..=5 => todo!(
                                "getValidPorts: no lrf4/5 in dd1a — DT_CHECK_MSG(arch < \
                                 RCUDD1A_ISA) holds on neither modelled generation \
                                 (PortAssignment.cpp:248)"
                            ),
                            // Unable to assign a port.
                            _ => ports(&[]),
                        }
                    } else if value.contains("xrf") {
                        // xrf has only port 0.
                        ports(&[PortId::P0])
                    } else {
                        ValidPorts::Rejected(PortRejection::InvalidMacArgumentInPtInt)
                    }
                } else if prec.contains("fp16")
                    || prec.contains("bf16")
                    || prec.contains("fp8")
                    || prec.contains("fp4")
                {
                    if value.contains("west") {
                        ports(&[PortId::P1, PortId::P2])
                    } else if value.contains("north") {
                        match A::GEN {
                            IsaGen::Sen1p5 => ports(&[PortId::P1]),
                            IsaGen::Rcudd1a => ports(&[PortId::P0, PortId::P1, PortId::P2]),
                        }
                    } else if value.contains("crossptnlink") {
                        match A::GEN {
                            IsaGen::Sen1p5 => ports(&[PortId::P1]),
                            IsaGen::Rcudd1a => todo!(
                                "getValidPorts: Cross PT north link only present in sen 1.5 \
                                 (PortAssignment.cpp:275)"
                            ),
                        }
                    } else if value.contains("latch") {
                        ports(&[PortId::P0])
                    } else if value.contains("zero") {
                        ports(&[PortId::P0, PortId::P1])
                    } else if value.contains("one") {
                        ports(&[PortId::P2])
                    } else if let Port::Lrf(lrf) = operand_value {
                        match A::GEN {
                            IsaGen::Sen1p5 if lrf.get() <= 3 => ports(&[PortId::P1]),
                            IsaGen::Rcudd1a if lrf.get() <= 11 => ports(&[PortId::P0, PortId::P1]),
                            // Unable to assign a port.
                            _ => ports(&[]),
                        }
                    } else if value.contains("xrf") {
                        ports(&[PortId::P0])
                    } else {
                        ValidPorts::Rejected(PortRejection::InvalidMacArgumentInPtFp)
                    }
                } else {
                    ports(&[])
                }
            } else if matches!(comp, DfirUnit::Pe | DfirUnit::Sfp) {
                if value.contains("pt")
                    && matches!(operand_precision, Precision::Int24 | Precision::Fp24)
                {
                    ports(&[PortId::P0])
                } else if value.contains("pt")
                    || value.contains("lx")
                    || value.contains("sfp")
                    || value.contains("pe")
                {
                    ports(&[PortId::P0, PortId::P1, PortId::P2])
                } else if let Port::Lrf(lrf) = operand_value {
                    let max_lrf_num: u8 = match A::GEN {
                        IsaGen::Sen1p5 => 32,
                        IsaGen::Rcudd1a => 16,
                    };
                    if lrf.get() <= max_lrf_num - 1 {
                        ports(&[PortId::P0, PortId::P1, PortId::P2])
                    } else {
                        ports(&[])
                    }
                } else if value.contains("zero") {
                    match A::GEN {
                        IsaGen::Sen1p5 => ports(&[PortId::P0, PortId::P1, PortId::P2]),
                        IsaGen::Rcudd1a => ports(&[PortId::P1, PortId::P2]),
                    }
                } else if value.contains("one") {
                    match A::GEN {
                        IsaGen::Sen1p5 => ports(&[PortId::P0, PortId::P1, PortId::P2]),
                        IsaGen::Rcudd1a => ports(&[PortId::P1]),
                    }
                } else if value.contains("two") || value.contains("three") {
                    ports(&[PortId::P2])
                } else if value.contains("xrf") {
                    ports(&[PortId::P0])
                } else if value.contains("latch") {
                    ports(&[PortId::P0, PortId::P1, PortId::P2])
                } else if value.contains("sfpring") {
                    // ⛔ UNREACHABLE — `sfpring` contains `sfp`, matched seven arms above.
                    if matches!(comp, DfirUnit::Sfp) {
                        ports(&[PortId::P0, PortId::P1, PortId::P2])
                    } else {
                        ValidPorts::Rejected(PortRejection::NoSfpRingInOtherComponents)
                    }
                } else if value.contains("nfwd0") {
                    ports(&[PortId::P0])
                } else if value.contains("nfwd2") {
                    ports(&[PortId::P2])
                } else {
                    ValidPorts::Rejected(PortRejection::InvalidMacArgumentInPeSfp)
                }
            } else {
                ValidPorts::Rejected(PortRejection::InvalidComputeUnitForMac)
            }
        }
        Op::Sentient(sentient::Op::VectorBinary { binary_op, .. }) => {
            if matches!(comp, DfirUnit::Pe | DfirUnit::Sfp) {
                if value.contains("pt") && operand_precision == Precision::Int24 {
                    ports(&[PortId::P0])
                } else if matches!(A::GEN, IsaGen::Sen1p5) {
                    match binary_op.op() {
                        // FMA/FNMS.
                        BinaryOp::Add | BinaryOp::Sub => {
                            ports(&[PortId::P0, PortId::P1, PortId::P2])
                        }
                        // FMINMAX/FCMP — and `AbsMin` is not in the group.
                        BinaryOp::Min
                        | BinaryOp::Max
                        | BinaryOp::AbsMax
                        | BinaryOp::CompareEq
                        | BinaryOp::CompareNeq
                        | BinaryOp::CompareLe
                        | BinaryOp::CompareLt => ports(&[PortId::P0, PortId::P2]),
                        // FMUL.
                        BinaryOp::MulDiv2 | BinaryOp::Mul => ports(&[PortId::P0, PortId::P1]),
                        // GCVT/FCVT/LOGICAL/MERGE/PACK.
                        _ => {
                            if value.contains("one") {
                                ports(&[PortId::P0])
                            } else {
                                ports(&[PortId::P0, PortId::P2])
                            }
                        }
                    }
                } else if value.contains("zero") {
                    ports(&[PortId::P2])
                } else if value.contains("one") {
                    // FMUL lowers to `src0 x src1`, unlike every other binary op.
                    if matches!(binary_op.op(), BinaryOp::MulDiv2 | BinaryOp::Mul) {
                        ports(&[PortId::P1])
                    } else {
                        ValidPorts::Rejected(PortRejection::OneInBinaryOpOutsideFmul)
                    }
                } else if value.contains("sfpring") {
                    if matches!(comp, DfirUnit::Sfp) {
                        ports(&[PortId::P0, PortId::P2])
                    } else {
                        ValidPorts::Rejected(PortRejection::NoSfpRingInOtherComponents)
                    }
                } else if matches!(binary_op.op(), BinaryOp::Mul) {
                    ports(&[PortId::P0, PortId::P1])
                } else {
                    ports(&[PortId::P0, PortId::P2])
                }
            } else {
                ValidPorts::Rejected(PortRejection::InvalidComputeUnitForBinary)
            }
        }
        Op::Sentient(sentient::Op::VectorUnary { unary_op, .. }) => {
            if matches!(comp, DfirUnit::Pe | DfirUnit::Sfp) {
                if matches!(unary_op, UnaryOp::FastExp | UnaryOp::Floor) {
                    ports(&[PortId::P2])
                } else {
                    ports(&[PortId::P0])
                }
            } else {
                ValidPorts::Rejected(PortRejection::InvalidComputeUnitForUnary)
            }
        }
        Op::Sentient(sentient::Op::VectorTernary { ternary_op, .. }) => {
            if matches!(comp, DfirUnit::Pe | DfirUnit::Sfp) {
                // `DT_CHECK(ternary_op == select)` is total: `TernaryOp` has the one variant.
                let TernaryOp::Select = ternary_op;
                match A::GEN {
                    IsaGen::Sen1p5 => ports(&[PortId::P0, PortId::P2]),
                    IsaGen::Rcudd1a => {
                        if value.contains("zero") {
                            ports(&[PortId::P2])
                        } else {
                            ports(&[PortId::P0, PortId::P2])
                        }
                    }
                }
            } else {
                ValidPorts::Rejected(PortRejection::InvalidComputeUnitForTernary)
            }
        }
        _ => ValidPorts::Rejected(PortRejection::UnsupportedOp),
    }
}

/// Replaces: e126_IsSwappablePerAlgebraicReassociation
///
/// Which MAC operand carries the *value* when the other two are the constants 0.0 and 1.0 — the one
/// operand a reassociation may move to a different port. `None` is the reference's `-1`, which also
/// covers a non-MAC op and a MAC whose operands are all constant.
/// ⭐ ITS `StringRef operand_value;` (`:608`) IS DECLARED AND NEVER USED.
pub(crate) fn is_swappable_per_algebraic_reassociation(op: &Op) -> Option<MacOperand> {
    let Op::Sentient(sentient::Op::VectorMac {
        op_a, op_b, op_c, ..
    }) = op
    else {
        return None;
    };
    let operands = [op_a.port, op_b.port, op_c.port];
    // Without both a 0 and a 1 the inputs cannot validate the swap.
    if !(operands.contains(&Port::Zero) && operands.contains(&Port::One)) {
        return None;
    }
    if !matches!(op_a.port, Port::Zero | Port::One) {
        Some(MacOperand::A)
    } else if !matches!(op_b.port, Port::Zero | Port::One) {
        Some(MacOperand::B)
    } else if !matches!(op_c.port, Port::Zero | Port::One) {
        Some(MacOperand::C)
    } else {
        None
    }
}

// crustify:todo: e339_addNodesToGraph
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:473  (131 body lines, level 1)
//   original  : void PortAssignmentPass::addNodesToGraph(Operation *op, const SenComponents comp)
//   calls     : e125_getValidPorts, e126_IsSwappablePerAlgebraicReassociation

// crustify:todo: e340_updateSentientIRPorts
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:682  (52 body lines, level 1)
//   original  : void PortAssignmentPass::updateSentientIRPorts(dataflow::ProgramUnitOp &unit)
//   calls     : e124_getPortAttr, e252_size

// crustify:todo: e341_addReuseToDummyOperands
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:739  (36 body lines, level 1)
//   original  : void PortAssignmentPass::addReuseToDummyOperands()
//   calls     : e124_getPortAttr, e252_size

// crustify:todo: e455_buildGraphNodes
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:636  (6 body lines, level 2)
//   original  : void PortAssignmentPass::buildGraphNodes(dataflow::ProgramUnitOp &unit, const SenComponents comp)
//   calls     : e339_addNodesToGraph

// crustify:todo: e520_processDataID
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:93  (67 body lines, level 3)
//   original  : void PortAssignmentPass::processDataID(int data_id, StringRef operand_value, Operation *current_op)
//   calls     : e422_insert

// crustify:todo: e572_computePortLiveRange
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:165  (40 body lines, level 4)
//   original  : void PortAssignmentPass::computePortLiveRange(dataflow::ProgramUnitOp &unit)
//   calls     : e252_size, e422_insert, e520_processDataID

// crustify:todo: e608_buildGraphEdges
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:644  (37 body lines, level 5)
//   original  : void PortAssignmentPass::buildGraphEdges(dataflow::ProgramUnitOp &unit)
//   calls     : e572_computePortLiveRange

// crustify:todo: e632_doPortAssignments
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:776  (12 body lines, level 6)
//   original  : void PortAssignmentPass::doPortAssignments(dataflow::ProgramUnitOp &unit, const SenComponents comp)
//   calls     : e123_clean, e340_updateSentientIRPorts, e341_addReuseToDummyOperands, e382_performGraphColoring, e455_buildGraphNodes, e608_buildGraphEdges

// crustify:todo: e645_runOnOperation
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:793  (21 body lines, level 7)
//   original  : void PortAssignmentPass::runOnOperation()
//   calls     : e632_doPortAssignments

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::islands::sentient::dialects::Val;
    use crate::islands::sentient::dialects::sentient::{LrfIndex, Operand, RegType, ResultPorts};
    use crate::units::Row;

    /// A colouring graph that only records being cleared — `OutOfScopeColoringGraph::clear` is a
    /// `todo!`, and `clean`'s whole contract is that it asks the graph rather than replacing it.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct CountingGraph {
        clears: u32,
    }

    impl ColoringGraph for CountingGraph {
        fn clear(&mut self) {
            self.clears += 1;
        }
    }

    /// `sentient.vector_mac` reading `$opA` from `port`, at `precision`.
    fn mac(port: Port, operand_precision: Precision, precision: Precision) -> Op {
        Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: vec![Val(10)],
            op_a: Operand {
                precision: operand_precision,
                ..Operand::from(port)
            },
            op_b: Operand::from(Port::West),
            op_c: Operand::from(Port::Zero),
            result: ResultPorts::default(),
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision: precision,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: None,
        })
    }

    /// A three-operand mac, for the reassociation question.
    fn mac_of(a: Port, b: Port, c: Port) -> Op {
        let Op::Sentient(sentient::Op::VectorMac {
            mask,
            xrf_write_ptr,
            xrf_read_ptr,
            results,
            result,
            mode,
            compute_precision,
            fold_mode,
            unroll_factor,
            xrf_read_incr,
            xrf_write_incr,
            data_transfer_only,
            dbg_name,
            ..
        }) = mac(a, Precision::Fp16, Precision::Fp16)
        else {
            unreachable!("`mac` builds a vector_mac")
        };
        Op::Sentient(sentient::Op::VectorMac {
            mask,
            xrf_write_ptr,
            xrf_read_ptr,
            results,
            op_a: Operand::from(a),
            op_b: Operand::from(b),
            op_c: Operand::from(c),
            result,
            mode,
            compute_precision,
            fold_mode,
            unroll_factor,
            xrf_read_incr,
            xrf_write_incr,
            data_transfer_only,
            is_data_weight: None,
            dbg_name,
        })
    }

    /// `sentient.vector_binary` on `port` with `binary_op`.
    fn binary(port: Port, binary_op: BinaryOp) -> Op {
        Op::Sentient(sentient::Op::VectorBinary {
            mask: Val(2),
            op_a: Operand::from(port),
            op_b: Operand::from(Port::Lx),
            binary_op: sentient::Binary::Plain(binary_op),
            result: ResultPorts::default(),
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// A non-compute op, for the `unsupported op` arm.
    fn scalar_copy() -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input: Val(1),
            result: Val(2),
            reg: sentient::Reg {
                locale: RegType::Lccr,
                index: None,
            },
            program_header: false,
            element_size: None,
        })
    }

    fn pt() -> DfirUnit {
        let Some(row) = Row::checked(0) else {
            unreachable!("every arch has a PT row 0")
        };
        DfirUnit::PtRow(row)
    }

    /// e123 — all eight slots go, and the graph is asked to clear ITSELF rather than replaced.
    #[test]
    fn clean_empties_every_slot_and_asks_the_graph_to_clear_itself() {
        let mut pass = PortAssignment::<CountingGraph> {
            graph: CountingGraph::default(),
            dummy_nodes: vec![DataId(1)],
            reuse_nodes: vec![DataId(2)],
            reuse_nodes_per_op: vec![vec![DataId(2), DataId(3)]],
            port_assignment: BTreeMap::from([(DataId(1), PortId::P0)]),
            op_to_index: BTreeMap::from([(OpAt(0), InstrIndex(0))]),
            operand_to_owners: BTreeMap::from([(DataId(1), vec![OpAt(0)])]),
            operand_to_liverange: BTreeMap::from([(DataId(1), LiveRange)]),
        };
        pass.clean();
        assert_eq!(pass.graph, CountingGraph { clears: 1 });
        assert!(pass.dummy_nodes.is_empty());
        assert!(pass.reuse_nodes.is_empty());
        assert!(pass.reuse_nodes_per_op.is_empty());
        assert!(pass.port_assignment.is_empty());
        assert!(pass.op_to_index.is_empty());
        assert!(pass.operand_to_owners.is_empty());
        assert!(pass.operand_to_liverange.is_empty());
    }

    /// e124 — the attribute is exactly what the colouring recorded for that operand value.
    #[test]
    fn port_attr_answers_with_the_colouring_result() {
        let pass = PortAssignment::<CountingGraph> {
            port_assignment: BTreeMap::from([(DataId(4), PortId::P2)]),
            ..PortAssignment::default()
        };
        assert_eq!(pass.port_attr(DataId(4)), PortId::P2);
        assert_eq!(pass.port_attr(DataId(4)).get(), 2);
    }

    /// e125 — the ISA table, on the arms the vendor's own corpus reaches: a PT int mac, the same mac
    /// in fp, and a PE mac.
    #[test]
    fn valid_ports_follows_the_isa_table() {
        let int_mac = mac(Port::West, Precision::Int8, Precision::Int8);
        // West on DD2 is port 2 alone; SEN1P5 gained port 1.
        assert_eq!(
            valid_ports::<Dd2>(&int_mac, Port::West, pt(), Precision::Int8, Precision::Int8),
            ports(&[PortId::P2])
        );
        assert_eq!(
            valid_ports::<Sen1p5>(&int_mac, Port::West, pt(), Precision::Int8, Precision::Int8),
            ports(&[PortId::P1, PortId::P2])
        );
        // ⛔ `Port::None` SPELLS `none`, WHICH CONTAINS `one` — it takes the constant-1 arm.
        assert_eq!(
            valid_ports::<Dd2>(&int_mac, Port::None, pt(), Precision::Int8, Precision::Int8),
            ports(&[PortId::P2])
        );
        // `arch >= RCUDD1A_ISA` holds on DD2 too, so lrf0..3 is port 1 and the `{0}` arm is dead.
        assert_eq!(
            valid_ports::<Dd2>(
                &int_mac,
                Port::Lrf(LrfIndex::L2),
                pt(),
                Precision::Int8,
                Precision::Int8
            ),
            ports(&[PortId::P1])
        );
        // Beyond lrf5 there is no port to assign, and that is not a pass failure.
        assert_eq!(
            valid_ports::<Dd2>(
                &int_mac,
                Port::Lrf(LrfIndex::L9),
                pt(),
                Precision::Int8,
                Precision::Int8
            ),
            ports(&[])
        );
        // In fp, lrf0..11 is {0, 1} on DD2 — the same operand, a different table.
        let fp_mac = mac(Port::Lrf(LrfIndex::L9), Precision::Fp16, Precision::Fp16);
        assert_eq!(
            valid_ports::<Dd2>(
                &fp_mac,
                Port::Lrf(LrfIndex::L9),
                pt(),
                Precision::Fp16,
                Precision::Fp16
            ),
            ports(&[PortId::P0, PortId::P1])
        );
        // ⭐ A PT MAC COMPUTING IN fp32 MATCHES NEITHER PRECISION BRANCH and falls out empty.
        assert_eq!(
            valid_ports::<Dd2>(&fp_mac, Port::West, pt(), Precision::Fp32, Precision::Fp32),
            ports(&[])
        );
        // On the PE a 24-bit PT operand is pinned to port 0, and anything else from the PT is free.
        assert_eq!(
            valid_ports::<Dd2>(
                &fp_mac,
                Port::Pt,
                DfirUnit::Pe,
                Precision::Int24,
                Precision::Fp16
            ),
            ports(&[PortId::P0])
        );
        assert_eq!(
            valid_ports::<Dd2>(
                &fp_mac,
                Port::Pt,
                DfirUnit::Pe,
                Precision::Fp16,
                Precision::Fp16
            ),
            ports(&[PortId::P0, PortId::P1, PortId::P2])
        );
        // A DD2 binary `mul` reading the constant 1 is the FMUL exception: src1, not src2.
        assert_eq!(
            valid_ports::<Dd2>(
                &binary(Port::One, BinaryOp::Mul),
                Port::One,
                DfirUnit::Sfp,
                Precision::Fp16,
                Precision::Fp16
            ),
            ports(&[PortId::P1])
        );
        // On SEN1P5 the whole operator group answers first, so the same operand is an FMUL pair.
        assert_eq!(
            valid_ports::<Sen1p5>(
                &binary(Port::One, BinaryOp::Mul),
                Port::One,
                DfirUnit::Sfp,
                Precision::Fp16,
                Precision::Fp16
            ),
            ports(&[PortId::P0, PortId::P1])
        );
    }

    /// e125's negatives — the four arms that fail the pass rather than answer with no port.
    #[test]
    fn valid_ports_rejects_the_arms_that_signal_pass_failure() {
        let int_mac = mac(Port::West, Precision::Int8, Precision::Int8);
        // An `east` operand is in neither PT int list.
        assert_eq!(
            valid_ports::<Dd2>(&int_mac, Port::East, pt(), Precision::Int8, Precision::Int8),
            ValidPorts::Rejected(PortRejection::InvalidMacArgumentInPtInt)
        );
        // A mac on a component that is not PT, PE or SFP.
        assert_eq!(
            valid_ports::<Dd2>(
                &int_mac,
                Port::West,
                DfirUnit::L3lu,
                Precision::Int8,
                Precision::Int8
            ),
            ValidPorts::Rejected(PortRejection::InvalidComputeUnitForMac)
        );
        // The SFP ring outside the SFP — reachable from a binary op, whose chain has no `sfp` arm.
        assert_eq!(
            valid_ports::<Dd2>(
                &binary(Port::SfpRing, BinaryOp::Add),
                Port::SfpRing,
                DfirUnit::Pe,
                Precision::Fp16,
                Precision::Fp16
            ),
            ValidPorts::Rejected(PortRejection::NoSfpRingInOtherComponents)
        );
        // The constant 1 in a non-FMUL binary op.
        assert_eq!(
            valid_ports::<Dd2>(
                &binary(Port::One, BinaryOp::Add),
                Port::One,
                DfirUnit::Sfp,
                Precision::Fp16,
                Precision::Fp16
            ),
            ValidPorts::Rejected(PortRejection::OneInBinaryOpOutsideFmul)
        );
        // Anything that is not one of the four compute ops.
        assert_eq!(
            valid_ports::<Dd2>(
                &scalar_copy(),
                Port::West,
                DfirUnit::Pe,
                Precision::Fp16,
                Precision::Fp16
            ),
            ValidPorts::Rejected(PortRejection::UnsupportedOp)
        );
    }

    /// e126 — the value operand when the other two are 0 and 1, and the three ways there is no swap.
    #[test]
    fn is_swappable_names_the_one_operand_that_is_not_a_constant() {
        assert_eq!(
            is_swappable_per_algebraic_reassociation(&mac_of(Port::West, Port::Zero, Port::One)),
            Some(MacOperand::A)
        );
        assert_eq!(
            is_swappable_per_algebraic_reassociation(&mac_of(Port::Zero, Port::West, Port::One)),
            Some(MacOperand::B)
        );
        assert_eq!(
            is_swappable_per_algebraic_reassociation(&mac_of(Port::Zero, Port::One, Port::West)),
            Some(MacOperand::C)
        );
        // All three constant: nothing is the value, so no swap.
        assert_eq!(
            is_swappable_per_algebraic_reassociation(&mac_of(Port::Zero, Port::One, Port::Zero)),
            None
        );
        // Only a zero: not enough constants to validate the swap.
        assert_eq!(
            is_swappable_per_algebraic_reassociation(&mac_of(Port::West, Port::Zero, Port::North)),
            None
        );
        // A non-mac op is the reference's `-1` too.
        assert_eq!(is_swappable_per_algebraic_reassociation(&scalar_copy()), None);
    }
}
