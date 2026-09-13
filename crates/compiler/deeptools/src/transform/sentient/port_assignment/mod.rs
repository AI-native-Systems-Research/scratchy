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
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::sentient::{BinaryOp, Port, Precision, TernaryOp, UnaryOp};
use crate::islands::sentient::dialects::{Op, regions_ref, sentient};
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

impl<G: ColoringGraph> PortAssignment<G> {
    /// Replaces: e455_buildGraphNodes
    ///
    /// Builds the colouring graph's nodes from the ISA restrictions: every op of the unit, in
    /// pre-order, is handed to [`Self::add_nodes_to_graph`].
    ///
    /// ⛔ IT DOES NOT INITIALISE THE GRAPH ITS COMMENT (`:637-638`) CLAIMS TO: the clearing is the
    /// caller's, `doPortAssignments`' `clean()` one line earlier (`:780-781`), so calling this twice
    /// ADDS to the nodes already there.
    pub(crate) fn build_graph_nodes<A: Arch>(&mut self, unit: &ProgramUnit<A>, comp: DfirUnit) {
        self.add_nodes_in::<A>(&unit.body, comp);
    }

    /// `unit.walk<WalkOrder::PreOrder>` — the op, then its regions.
    fn add_nodes_in<A: Arch>(&mut self, scope: &[Op], comp: DfirUnit) {
        for op in scope {
            self.add_nodes_to_graph::<A>(op, comp);
            for region in regions_ref(op) {
                self.add_nodes_in::<A>(region, comp);
            }
        }
    }

    /// `PortAssignmentPass::addNodesToGraph(Operation *, SenComponents)` — e455's ONE callee and
    /// SENPASS UNIT e339, whose anchor is still open below.
    ///
    /// ⛔ e339 IS A LEVEL-1 DEPENDENCY THAT NO REMAINING SCHEDULE OWNS: sc2's port driver died on an
    /// authentication error with 12 batches unrun, `sentient.cpp: e334_mergeScalarOpIntoMac +7` among
    /// them. Isolating the call in a private seam is the `2a8195231` precedent, and e339's TODO is left
    /// untouched — filling it is not this batch's work.
    fn add_nodes_to_graph<A: Arch>(&mut self, op: &Op, comp: DfirUnit) {
        todo!(
            "PortAssignmentPass::addNodesToGraph — senpass e339 (PortAssignment.cpp:473) is not \
             ported yet, and this {comp:?} unit's {op:?} needs it"
        )
    }
}

/// WHICH LOCAL REGION AN INTERVAL BELONGS TO — `LabeledRange::label`.
///
/// ⛔ `INT_MAX` IS "GLOBAL" AND `LabeledRange::print` SPELLS IT SO (`Analyses/LiveRange.cpp:118`);
/// this is not a region index of zero, and `label_for_region` starts here (`:106`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RegionLabel {
    /// `INT_MAX`.
    Global,
    /// `op_to_index_[region_op]` for a `uniform.uniformize_regions`/`uniform.equalize_pattern`.
    Region(InstrIndex),
}

/// ONE INTERVAL OF A [`LiveRange`] — `LiveRange::LabeledRange` (`Analyses/LiveRange.hpp:19-33`).
///
/// ⭐ THIS IS THE PART OF THE ANALYSIS `e520_processDataID` ITSELF DECIDES, and it is why the union
/// below is a seam rather than a bare `todo!`: which instruction pair and which region label a data
/// ID's range is built from is in this campaign's reach; the interval arithmetic is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LabeledRange {
    /// `first` — the producer's instruction index.
    pub(crate) first: InstrIndex,
    /// `last` — the consumer's, or the terminator that stands for it.
    pub(crate) last: InstrIndex,
    /// `label`.
    pub(crate) label: RegionLabel,
}

/// THE `LiveRange` INTERVAL ALGEBRA — a trait for the same reason [`ColoringGraph`] is one: the
/// analysis is not in this campaign and a test must still be able to observe what a pass builds.
///
/// ⛔ `Analyses/LiveRange.{hpp,cpp}` IS OUT OF CAMPAIGN SCOPE — see [`LiveRange`].
pub(crate) trait LiveRanges {
    /// `range.unionWith(LiveRange(interval))` (`Analyses/LiveRange.cpp:67`).
    fn union_with(&mut self, range: &mut LiveRange, interval: LabeledRange) {
        let _ = (range, interval);
        todo!("LiveRange::unionWith (Analyses/LiveRange.cpp:67) — out of campaign scope")
    }

    /// `lhs.overlaps(rhs, half_open_range)` (`Analyses/LiveRange.hpp:44`).
    ///
    /// ⛔ THE DEFAULT IS `true` AND E608 PASSES `false` EXPLICITLY (`:647-648`), so the flag is an
    /// argument here and not a fact about the seam.
    fn overlaps(&mut self, lhs: &LiveRange, rhs: &LiveRange, half_open_range: HalfOpenRange) -> bool {
        let _ = (lhs, rhs, half_open_range);
        todo!("LiveRange::overlaps (Analyses/LiveRange.cpp:57) — out of campaign scope")
    }
}

/// WHETHER AN OVERLAP TEST TREATS THE END OF A RANGE AS OUTSIDE IT — `bool half_open_range`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HalfOpenRange {
    /// `half_open_range = true`, the seam's own default.
    Yes,
    /// `half_open_range = false`, which is what e608 asks for.
    No,
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OutOfScopeLiveRanges;

impl LiveRanges for OutOfScopeLiveRanges {}

/// WHETHER A DATA ID'S LIVE RANGE COULD BE COMPUTED AT ALL.
///
/// ⛔ `No` IS `emitError("Unable to compute live range\n")` FOLLOWED BY `signalPassFailure()`
/// (`:154-156`), which ends the compilation — the same reason [`ValidPorts::Rejected`] is its own
/// answer rather than an empty one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveRangeComputed {
    /// The interval was unioned in — or there was no data ID, which is nothing to compute.
    Yes,
    /// The ancestor sharing the producer's block is neither a loop, an `if` nor a local region.
    No,
}

/// WHERE ONE OP SITS IN THE UNIT'S PRE-ORDER WALK — its own [`OpAt`] and the BLOCK holding it, as the
/// `(enclosing op, which region of it)` steps that reach it.
///
/// ⭐ `Operation::getBlock()` AND `getParentOp()` FOR A TREE WITH NO PARENT POINTERS, which is exactly
/// the *mechanism* the campaign brief lets a port supply for itself.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Position {
    at: OpAt,
    block: Vec<(OpAt, u32)>,
}

/// `unit.walk<WalkOrder::PreOrder>`, keeping each op's block — indexed by [`OpAt`].
fn walk_of(unit: &[Op]) -> Vec<Position> {
    fn walk(scope: &[Op], block: &mut Vec<(OpAt, u32)>, next: &mut u32, out: &mut Vec<Position>) {
        for op in scope {
            let at = OpAt(*next);
            *next += 1;
            out.push(Position {
                at,
                block: block.clone(),
            });
            for (index, region) in regions_ref(op).into_iter().enumerate() {
                block.push((at, index as u32));
                walk(region, block, next, out);
                block.pop();
            }
        }
    }
    let mut out = Vec::new();
    walk(unit, &mut Vec::new(), &mut 0, &mut out);
    out
}

/// The op at one pre-order position.
fn op_at(unit: &[Op], target: OpAt) -> Option<&Op> {
    fn walk<'a>(scope: &'a [Op], next: &mut u32, target: OpAt) -> Option<&'a Op> {
        for op in scope {
            let here = OpAt(*next);
            *next += 1;
            if here == target {
                return Some(op);
            }
            for region in regions_ref(op) {
                if let Some(found) = walk(region, next, target) {
                    return Some(found);
                }
            }
        }
        None
    }
    walk(unit, &mut 0, target)
}

/// `Block::getTerminator()` for region `region` of the op at `parent` — the LAST op of that block,
/// which in pre-order is the greatest position whose block is that one.
fn terminator_of(walk: &[Position], parent: &Position, region: u32) -> Option<OpAt> {
    let mut block = parent.block.clone();
    block.push((parent.at, region));
    walk.iter()
        .filter(|position| position.block == block)
        .map(|position| position.at)
        .next_back()
}

impl<G: ColoringGraph> PortAssignment<G> {
    /// `op_to_index_[op]` — ⛔ `std::map::operator[]` DEFAULT-CONSTRUCTS, so an op the index walk never
    /// numbered reads as instruction 0 rather than throwing.
    fn index_of(&self, op: OpAt) -> InstrIndex {
        self.op_to_index.get(&op).copied().unwrap_or(InstrIndex(0))
    }

    /// Replaces: e520_processDataID
    ///
    /// Records `current_op` as an owner of this operand value and widens that value's live range to
    /// reach it from its FIRST owner — the producer.
    ///
    /// ⛔ TRAP: A CONSUMER IN A DEEPER BLOCK EXTENDS THE RANGE TO THAT BLOCK'S TERMINATOR, not to the
    /// consumer: a loop body, an `else`-less `then`, or a local region (`:135-153`).
    /// ⛔ TRAP: THE LABEL IS THE **PRODUCER'S** LOCAL REGION, not the consumer's (`:107-112`).
    /// ⭐ ITS `StringRef operand_value` (`:93`) IS DECLARED AND NEVER USED, as `e126`'s is.
    pub(crate) fn process_data_id<R: LiveRanges>(
        &mut self,
        data_id: Option<DataId>,
        current_op: OpAt,
        unit: &[Op],
        live_ranges: &mut R,
    ) -> LiveRangeComputed {
        // "Sometimes, we can data_id being -1; in that case, don't process them." (`:95-96`)
        let Some(data_id) = data_id else {
            return LiveRangeComputed::Yes;
        };
        self.operand_to_owners
            .entry(data_id)
            .or_default()
            .push(current_op);
        // `insert` KEEPS AN EXISTING RANGE — it is not an overwrite (`:102`).
        self.operand_to_liverange.entry(data_id).or_default();
        let src_op = self.operand_to_owners[&data_id][0];

        let walk = walk_of(unit);
        let (Some(src), Some(current)) =
            (walk.get(src_op.0 as usize), walk.get(current_op.0 as usize))
        else {
            // Neither op can be absent in the reference: both came from the same pre-order walk.
            return LiveRangeComputed::Yes;
        };

        // `getRegionOp(src_op)`, kept only where it is a local region (`:107-112`).
        let mut label = RegionLabel::Global;
        for (ancestor, _region) in src.block.iter().rev() {
            if matches!(op_at(unit, *ancestor), Some(Op::UniformRegions(_))) {
                label = RegionLabel::Region(self.index_of(*ancestor));
                break;
            }
        }

        let first = self.index_of(src_op);
        let last = if src.block == current.block {
            // "If the current op is from the same block --> peace!" (`:116`)
            self.index_of(current_op)
        } else {
            // The ancestor of `current_op` whose own block is the producer's (`:124-133`).
            let Some((parent_op, _)) = current.block.get(src.block.len()).copied() else {
                return LiveRangeComputed::Yes;
            };
            let Some(parent) = walk.get(parent_op.0 as usize) else {
                return LiveRangeComputed::Yes;
            };
            let terminator = match op_at(unit, parent_op) {
                // "If ancestor is for-loop, liverange is throughout." (`:136-141`)
                Some(Op::Sentient(sentient::Op::For { .. })) => terminator_of(&walk, parent, 0),
                Some(Op::Sentient(sentient::Op::If { else_body, .. })) => {
                    if else_body.is_empty() {
                        terminator_of(&walk, parent, 0)
                    } else {
                        // `DT_ERROR("yet to handle the live range case")` (`:149`) — an abort, and the
                        // reference's own comment above it is *"TODO: Check for this case."*
                        todo!(
                            "processDataID: yet to handle the live range case — a \
                             `sentient.if` WITH an else region (PortAssignment.cpp:145-150)"
                        )
                    }
                }
                Some(Op::UniformRegions(_)) => terminator_of(&walk, parent, 0),
                _ => return LiveRangeComputed::No,
            };
            match terminator {
                Some(at) => self.index_of(at),
                // An empty block has no terminator; MLIR's own `getTerminator()` requires one.
                None => return LiveRangeComputed::Yes,
            }
        };
        if let Some(range) = self.operand_to_liverange.get_mut(&data_id) {
            live_ranges.union_with(range, LabeledRange { first, last, label });
        }
        LiveRangeComputed::Yes
    }
}

/// `op<X>DataID` NARROWED — ⛔ `None` IS THE `.td`'S `-1`, which e520 declines rather than looks up.
fn data_id_of(operand: &sentient::Operand) -> Option<DataId> {
    operand
        .data_id
        .and_then(|id| u32::try_from(id).ok())
        .map(DataId)
}

/// `isa<MacOp, BinaryOp, UnaryOp, TernaryOp, YieldOp>` — which ops the index walk numbers (`:168`).
fn takes_an_instruction_index(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(
            sentient::Op::VectorMac { .. }
                | sentient::Op::VectorBinary { .. }
                | sentient::Op::VectorUnary { .. }
                | sentient::Op::VectorTernary { .. }
                | sentient::Op::Yield { .. }
        )
    )
}

impl<G: ColoringGraph> PortAssignment<G> {
    /// Replaces: e572_computePortLiveRange
    ///
    /// Numbers the unit's compute ops and yields in pre-order, then widens each compute operand's live
    /// range to every op that reads it (`:165-204`).
    ///
    /// ⛔ TWO WALKS AND NOT ONE — *"Don't merge this step with before"* (`:175`): e520 reads the index
    /// of ops it has not reached yet, and an unnumbered one would read as instruction 0.
    /// ⛔ A `select` LEAVES `$opA` UNPROCESSED (`:194-197`), and it is the only ternary operator there
    /// is, so the arm is written as a match that a second one would break.
    /// ⭐ ONE `No` LOSES: it is the reference's `signalPassFailure()`, which does not unwind.
    pub(crate) fn compute_port_live_range<R: LiveRanges>(
        &mut self,
        unit: &[Op],
        live_ranges: &mut R,
    ) -> LiveRangeComputed {
        // First fill the operations
        for position in walk_of(unit) {
            let Some(op) = op_at(unit, position.at) else {
                continue;
            };
            if takes_an_instruction_index(op) {
                let next = InstrIndex(self.op_to_index.len() as u32);
                self.op_to_index.entry(position.at).or_insert(next);
            }
        }
        // Don't merge this step with before
        let mut computed = LiveRangeComputed::Yes;
        for position in walk_of(unit) {
            let Some(op) = op_at(unit, position.at) else {
                continue;
            };
            let operands: Vec<Option<DataId>> = match op {
                Op::Sentient(sentient::Op::VectorMac {
                    op_a, op_b, op_c, ..
                }) => vec![data_id_of(op_a), data_id_of(op_b), data_id_of(op_c)],
                Op::Sentient(sentient::Op::VectorBinary { op_a, op_b, .. }) => {
                    vec![data_id_of(op_a), data_id_of(op_b)]
                }
                Op::Sentient(sentient::Op::VectorUnary { op_a, .. }) => vec![data_id_of(op_a)],
                Op::Sentient(sentient::Op::VectorTernary {
                    op_a,
                    op_b,
                    op_c,
                    ternary_op,
                    ..
                }) => {
                    let mut ids = Vec::new();
                    match ternary_op {
                        // A `select` leaves `$opA` unprocessed (`:194-197`).
                        TernaryOp::Select => {
                            let _ = op_a;
                        }
                    }
                    ids.push(data_id_of(op_b));
                    ids.push(data_id_of(op_c));
                    ids
                }
                _ => Vec::new(),
            };
            for data_id in operands {
                if self.process_data_id(data_id, position.at, unit, live_ranges)
                    == LiveRangeComputed::No
                {
                    computed = LiveRangeComputed::No;
                }
            }
        }
        computed
    }
}

impl<G: ColoringGraph> PortAssignment<G> {
    /// Replaces: e608_buildGraphEdges
    ///
    /// Every ordered pair of operand values whose port live ranges overlap becomes a node pair on the
    /// colouring graph, with an interference edge between the distinct ones (`:644-679`).
    ///
    /// ⛔ TRAP: THE PAIRS ARE ORDERED AND INCLUDE THE DIAGONAL. A range overlaps itself, so every
    /// operand gets `getOrAddNode` called on it, and each unordered pair is visited TWICE — which is
    /// what fills `reuse_nodes_per_op_` with `{a,b}` and `{b,a}` both.
    /// ⛔ TRAP: THE REUSE PAIR IS PUSHED ONCE PER OWNER OF `node1`, not once per shared owner: the
    /// `break` leaves the inner loop only (`:668`), so two ops both naming `node1` and `node2` record
    /// the pair twice.
    /// ⚠️ THE ITERATION IS `BTreeMap`'s AND THE REFERENCE'S IS `DenseMap`'s, so the pairs come out in
    /// a different order; nothing downstream of `reuse_nodes_per_op_` or the graph reads that order.
    pub(crate) fn build_graph_edges<R: LiveRanges>(
        &mut self,
        unit: &[Op],
        live_ranges: &mut R,
    ) -> LiveRangeComputed {
        let computed = self.compute_port_live_range(unit, live_ranges);
        // The overlaps are read out before the graph is written, because both are `self`.
        let mut overlapping = Vec::new();
        for (node1_idx, range1) in &self.operand_to_liverange {
            for (node2_idx, range2) in &self.operand_to_liverange {
                if live_ranges.overlaps(range1, range2, HalfOpenRange::No) {
                    overlapping.push((*node1_idx, *node2_idx));
                }
            }
        }
        for (node1_idx, node2_idx) in overlapping {
            if self.dummy_nodes.contains(&node1_idx) && self.dummy_nodes.contains(&node2_idx) {
                continue;
            }
            if node1_idx != node2_idx
                && self.reuse_nodes.contains(&node1_idx)
                && self.reuse_nodes.contains(&node2_idx)
            {
                let owners1 = self.operand_to_owners.get(&node1_idx).cloned().unwrap_or_default();
                let owners2 = self.operand_to_owners.get(&node2_idx).cloned().unwrap_or_default();
                for op1 in &owners1 {
                    if owners2.contains(op1) {
                        self.reuse_nodes_per_op.push(vec![node1_idx, node2_idx]);
                    }
                }
            }
            self.graph.get_or_add_node(node1_idx);
            self.graph.get_or_add_node(node2_idx);
            if node1_idx != node2_idx {
                self.graph.add_bidirectional_edge(node1_idx, node2_idx);
            }
        }
        computed
    }
}

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
    use crate::islands::dataflow_ir::Units;
    use crate::islands::sentient::dialects::Val;
    use crate::islands::sentient::dialects::sentient::{
        Binary, LrfIndex, Operand, RegType, ResultPorts,
    };
    use crate::units::Row;

    /// A colouring graph that only records what it was asked — `GraphColoring` itself is out of
    /// scope, and `clean`'s whole contract is that it asks the graph rather than replacing it.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct CountingGraph {
        clears: u32,
        nodes: Vec<DataId>,
        edges: Vec<(DataId, DataId)>,
    }

    impl ColoringGraph for CountingGraph {
        fn clear(&mut self) {
            self.clears += 1;
        }

        fn get_or_add_node(&mut self, index: DataId) {
            self.nodes.push(index);
        }

        fn add_bidirectional_edge(&mut self, node1: DataId, node2: DataId) {
            self.edges.push((node1, node2));
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
        assert_eq!(
            pass.graph,
            CountingGraph {
                clears: 1,
                ..CountingGraph::default()
            }
        );
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
        assert_eq!(
            is_swappable_per_algebraic_reassociation(&scalar_copy()),
            None
        );
    }

    /// A `dataflow.program_unit` on a PE holding `body`.
    fn pe_unit(body: Vec<Op>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(DfirUnit::Pe, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// e455 — an EMPTY unit asks nothing and adds nothing, which is why `doPortAssignments` tests the
    /// node count before building any edge.
    #[test]
    fn e455_adds_nothing_of_its_own_for_a_unit_with_no_ops() {
        let mut pass = PortAssignment::<CountingGraph>::default();
        pass.build_graph_nodes::<Dd2>(&pe_unit(Vec::new()), DfirUnit::Pe);
        assert_eq!(pass.graph, CountingGraph::default());
        assert!(pass.dummy_nodes.is_empty() && pass.reuse_nodes.is_empty());
    }

    /// e455's positive — an op of the unit IS handed on, and the hand-off is e339, which is not ported.
    #[test]
    #[should_panic(expected = "senpass e339")]
    fn e455_hands_each_op_of_the_unit_to_the_unported_e339() {
        let unit = pe_unit(vec![mac(Port::West, Precision::Fp16, Precision::Fp16)]);
        PortAssignment::<CountingGraph>::default().build_graph_nodes::<Dd2>(&unit, DfirUnit::Pe);
    }

    /// A [`LiveRanges`] that records what a pass asks it to union — the interval algebra itself is
    /// `Analyses/LiveRange.cpp` and out of scope.
    #[derive(Debug, Default)]
    struct RecordingLiveRanges(Vec<LabeledRange>);

    impl LiveRanges for RecordingLiveRanges {
        fn union_with(&mut self, _range: &mut LiveRange, interval: LabeledRange) {
            self.0.push(interval);
        }
    }

    /// e520 — the producer's own use is one instruction wide, and a use inside a `sentient.for` runs to
    /// that body's LAST op rather than to the use itself.
    #[test]
    fn e520_extends_a_deeper_use_to_the_enclosing_blocks_terminator() {
        let unit = vec![
            scalar_copy(),
            Op::Sentient(sentient::Op::For {
                iv: Val(3),
                bound: Val(4),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![scalar_copy(), scalar_copy()],
            }),
        ];
        let mut pass = PortAssignment::<CountingGraph>::default();
        pass.op_to_index = (0..4)
            .map(|index| (OpAt(index), InstrIndex(index)))
            .collect();
        let mut ranges = RecordingLiveRanges::default();

        // ⛔ `data_id == -1` IS NOTHING TO COMPUTE, not a failure (`:95-96`).
        assert_eq!(
            pass.process_data_id(None, OpAt(0), &unit, &mut ranges),
            LiveRangeComputed::Yes
        );
        pass.process_data_id(Some(DataId(7)), OpAt(0), &unit, &mut ranges);
        pass.process_data_id(Some(DataId(7)), OpAt(2), &unit, &mut ranges);

        assert_eq!(
            ranges.0,
            vec![
                LabeledRange {
                    first: InstrIndex(0),
                    last: InstrIndex(0),
                    label: RegionLabel::Global,
                },
                LabeledRange {
                    first: InstrIndex(0),
                    last: InstrIndex(3),
                    label: RegionLabel::Global,
                },
            ]
        );
        assert_eq!(pass.operand_to_owners[&DataId(7)], vec![OpAt(0), OpAt(2)]);
    }

    /// A `sentient.vector_unary` whose `$opA` carries `data_id` — `None` being the `.td`'s `-1`.
    fn unary_on(data_id: Option<i32>) -> Op {
        Op::Sentient(sentient::Op::VectorUnary {
            mask: Val(2),
            op_a: Operand {
                data_id,
                ..Operand::from(Port::West)
            },
            unary_op: UnaryOp::Rec,
            result: ResultPorts::default(),
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// e572 — only the compute ops are numbered, and the two ops sharing a data ID both widen its
    /// range while the unassigned one is nothing to compute.
    #[test]
    fn e572_numbers_the_compute_ops_then_widens_each_operands_range() {
        let unit = vec![
            scalar_copy(),
            unary_on(Some(7)),
            unary_on(None),
            unary_on(Some(7)),
        ];
        let mut pass = PortAssignment::<CountingGraph>::default();
        let mut ranges = RecordingLiveRanges::default();

        assert_eq!(
            pass.compute_port_live_range(&unit, &mut ranges),
            LiveRangeComputed::Yes
        );

        // The `sentient.scalar_copy` at position 0 takes no instruction index.
        assert_eq!(
            pass.op_to_index,
            BTreeMap::from([
                (OpAt(1), InstrIndex(0)),
                (OpAt(2), InstrIndex(1)),
                (OpAt(3), InstrIndex(2)),
            ])
        );
        assert_eq!(pass.operand_to_owners[&DataId(7)], vec![OpAt(1), OpAt(3)]);
        assert_eq!(pass.operand_to_owners.len(), 1);
        assert_eq!(
            ranges.0,
            vec![
                LabeledRange {
                    first: InstrIndex(0),
                    last: InstrIndex(0),
                    label: RegionLabel::Global,
                },
                LabeledRange {
                    first: InstrIndex(0),
                    last: InstrIndex(2),
                    label: RegionLabel::Global,
                },
            ]
        );
    }

    /// A `sentient.vector_binary` whose `$opA` and `$opB` carry `a` and `b`.
    fn binary_on(a: i32, b: i32) -> Op {
        Op::Sentient(sentient::Op::VectorBinary {
            mask: Val(2),
            op_a: Operand {
                data_id: Some(a),
                ..Operand::from(Port::West)
            },
            op_b: Operand {
                data_id: Some(b),
                ..Operand::from(Port::East)
            },
            binary_op: Binary::Plain(BinaryOp::Min),
            result: ResultPorts::default(),
            compute_precision: Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// A [`LiveRanges`] that answers EVERY pair overlapping and records the flag it was asked with —
    /// the interval algebra itself is `Analyses/LiveRange.cpp` and out of scope.
    #[derive(Debug, Default)]
    struct AlwaysOverlapping(Vec<HalfOpenRange>);

    impl LiveRanges for AlwaysOverlapping {
        fn union_with(&mut self, _range: &mut LiveRange, _interval: LabeledRange) {}

        fn overlaps(
            &mut self,
            _lhs: &LiveRange,
            _rhs: &LiveRange,
            half_open_range: HalfOpenRange,
        ) -> bool {
            self.0.push(half_open_range);
            true
        }
    }

    /// e608 — every ordered pair of the three ranges is asked, a pair of dummies is skipped whole,
    /// the diagonal adds its node but no edge, and the reuse pair is pushed once per owner.
    #[test]
    fn e608_walks_every_ordered_pair_and_skips_the_pair_of_dummies() {
        let unit = vec![binary_on(7, 8), unary_on(Some(9))];
        let mut pass = PortAssignment::<CountingGraph>::default();
        pass.dummy_nodes = vec![DataId(7), DataId(9)];
        pass.reuse_nodes = vec![DataId(7), DataId(8)];
        let mut ranges = AlwaysOverlapping::default();

        assert_eq!(
            pass.build_graph_edges(&unit, &mut ranges),
            LiveRangeComputed::Yes
        );

        // Nine ordered pairs, every one asked with `half_open_range = false`.
        assert_eq!(ranges.0, vec![HalfOpenRange::No; 9]);
        assert_eq!(
            pass.graph.edges,
            vec![
                (DataId(7), DataId(8)),
                (DataId(8), DataId(7)),
                (DataId(8), DataId(9)),
                (DataId(9), DataId(8)),
            ]
        );
        // `(8, 8)` adds its node twice and no edge; `(7, 7)`, `(7, 9)`, `(9, 7)` and `(9, 9)` are
        // two dummies and add nothing at all.
        assert_eq!(
            pass.graph.nodes,
            vec![
                DataId(7),
                DataId(8),
                DataId(8),
                DataId(7),
                DataId(8),
                DataId(8),
                DataId(8),
                DataId(9),
                DataId(9),
                DataId(8),
            ]
        );
        assert_eq!(
            pass.reuse_nodes_per_op,
            vec![vec![DataId(7), DataId(8)], vec![DataId(8), DataId(7)]]
        );
    }
}
