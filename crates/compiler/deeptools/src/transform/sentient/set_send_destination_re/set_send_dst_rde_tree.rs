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

//! `SetSendDestinationRE.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e196_isOperationAUse` | 196 | 0 | 5 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:225` |
//! | `e197_isSimplifiable` | 197 | 0 | 10 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:308` |
//! | `e379_initializeDataflowInfoForLXLU` | 379 | 1 | 32 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:244` |
//! | `e380_initializeDataflowInfoForSFP` | 380 | 1 | 30 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:277` |
//! | `e476_initializeDataflowInfo` | 476 | 2 | 8 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:235` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so both items below are reachable only from this
// file's own tests until `e583_runOnOperation` (level 4) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e583: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::sentient::dialects::{Definitions, Op, dataflow, sentient, uniform};
use crate::transform::sentient::analyses::RdeNode;
use crate::transform::sentient::set_send_destination_re::set_dst_gen_value_lxlu::SetDstGenValueLxlu;
use crate::transform::sentient::set_send_destination_re::set_dst_gen_value_sfp::SetDstGenValueSfp;
use crate::transform::sentient::set_send_destination_re::simple_set_dst_gen_value_lxlu::SendDestination;
use crate::transform::sentient::set_send_destination_re::{
    QueryMapOp, SetDestReOptimizationMode, SetSendDstReCount,
};
use crate::units::{Core, Corelet, DfirUnit, Residency};

/// `RDETreeOptimizer<SetSendDstRDETree>` AS e475 REACHES IT
/// (`Analyses/RedundantDefinitionEliminationTree.hpp:345`) — this tree built over one program unit,
/// computed, simplified, then walked bottom-up to remove, common up and hoist its redundant
/// `set_send_dst`s.
///
/// ⛔ TREE AND OPTIMIZER ARE BOTH `Analyses/` WORK AND OUT OF CAMPAIGN SCOPE, so this is a trait for
/// the reason [`Liveness`](crate::transform::sentient::analyses::Liveness) is one: e475's whole tail
/// is gated on the count, and a test must be able to observe both sides of that gate.
pub(crate) trait SetSendDstRdeTreeOptimizer {
    /// `tree.compute(unit)`, `tree.simplify()` and `optimizer.optimize()` AS ONE CALL (`:121-136`) —
    /// rewrites `unit` and answers how many nodes were removed, commoned up or hoisted.
    ///
    /// ⭐ THE CONSTRUCTION ARGUMENTS COME WITH IT: `mode` is e195's (`:121`), and
    /// `EnableDynamicLoopHoisting` (`:92-96`) is a `dcc-opt` flag this crate has not got.
    fn optimize(
        &mut self,
        unit: &mut Vec<Op>,
        mode: SetDestReOptimizationMode,
    ) -> SetSendDstReCount;
}

/// THE ONE CRATE IMPLEMENTATION: the tree is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OutOfScopeSetSendDstRdeTree;

impl SetSendDstRdeTreeOptimizer for OutOfScopeSetSendDstRdeTree {
    fn optimize(
        &mut self,
        _unit: &mut Vec<Op>,
        _mode: SetDestReOptimizationMode,
    ) -> SetSendDstReCount {
        todo!(
            "RDETreeOptimizer<SetSendDstRDETree>::optimize \
             (Analyses/RedundantDefinitionEliminationTree.hpp:361) — out of campaign scope"
        )
    }
}

/// Replaces: e196_isOperationAUse
///
/// The six ops that READ a send destination: the two sends and the four vector computes.
#[must_use]
pub(crate) fn is_operation_a_use(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(
            sentient::Op::LoadAndSend { .. }
                | sentient::Op::LoadComputeAndSend { .. }
                | sentient::Op::VectorMac { .. }
                | sentient::Op::VectorUnary { .. }
                | sentient::Op::VectorBinary { .. }
                | sentient::Op::VectorTernary { .. }
        )
    )
}

/// Replaces: e197_isSimplifiable
///
/// *"We would like to remove subtrees that do not contain any set_send_dst or load_and_send, mac,
/// unary, binary, ternary operations"* (`:309-310`) — and that are a leaf, or a statically dead loop.
///
/// ⭐ [`RdeNode`] IS THE CAMPAIGN-WIDE SEAM: the tree class is `Analyses/` work and out of campaign
/// scope, and all four RDE passes ask a node the same questions.
///
/// ⛔ TRAP: `sentient.load_compute_and_send` IS A USE (e196) AND STILL SIMPLIFIABLE HERE — the two
/// `isa<>` lists differ by exactly that op (`:226-228` against `:312-314`). `simplify()` clears every
/// simplifiable node (`Analyses/RedundantDefinitionEliminationTree.cpp:325-340`), so such a use leaves
/// the tree and `deadDefOptimization`'s `sib->isUse()` guard
/// (`Analyses/RedundantDefinitionEliminationTreeImpl.cpp:228`) can no longer fire for it. Ported as
/// written — dbo-opt is this pipeline's oracle, so a "corrected" list would be the divergence.
///
/// ⛔ `isStaticallyDeadLoop` is `Analyses/` work and OUT OF CAMPAIGN SCOPE; the `||` short-circuits,
/// so a leaf never reaches it.
#[must_use]
pub(crate) fn is_simplifiable(node: &RdeNode<'_>) -> bool {
    let RdeNode::At { op, leaf } = node else {
        // `getRoot() == &node` — the root is the only node without an operation.
        return false;
    };
    if matches!(
        op,
        Op::Sentient(
            sentient::Op::SetSendDst { .. }
                | sentient::Op::LoadAndSend { .. }
                | sentient::Op::VectorMac { .. }
                | sentient::Op::VectorUnary { .. }
                | sentient::Op::VectorBinary { .. }
                | sentient::Op::VectorTernary { .. }
        )
    ) {
        return false;
    }
    if *leaf {
        return true;
    }
    todo!(
        "RedundantDefinitionEliminationTree::isStaticallyDeadLoop \
         (Analyses/RedundantDefinitionEliminationTree.hpp:247) — out of campaign scope"
    )
}

/// `SetDstGenValueLXLU` AND `SetDstGenValueSFP` AS ONE `setDataflowGen` ARGUMENT — WHICH of the two a
/// node generates is not the node's business but the program unit's, decided once by e195.
///
/// ⭐ THE C++ NEEDS NO SUCH TYPE because both derive from `DataFlowDefinitionBase` and `RDENode` holds
/// the base pointer; that class is `Analyses/` work and out of campaign scope, so the union of what
/// THIS tree's hooks generate is spelled here instead.
#[derive(Debug, Clone)]
pub(crate) enum SetDstGenValue {
    /// `kOptimizeForLXLU`'s definition.
    Lxlu(SetDstGenValueLxlu),
    /// `kOptimizeForSFP`'s definition.
    Sfp(SetDstGenValueSfp),
}

/// `consumer_unit.getType().lower()` NARROWED BY THE THREE COMPARISONS (`:262-271`) — `None` is the
/// `gen_mode` that stays `kUnknown`.
///
/// ⭐ THE `find("pt") != npos` COLLAPSE, EVALUATED OVER EVERY [`DfirUnit::spelling`]: `ptrow0`…`ptrow7`
/// and `crossptnlink` are the spellings that contain `"pt"` — *"it covers crossptlink as well"* — and
/// no other one does (`sfpstate` and `constant` are the near misses).
fn send_destination(unit: DfirUnit) -> Option<SendDestination> {
    match unit {
        DfirUnit::PtRow(_) | DfirUnit::CrossPtnLink => Some(SendDestination::Pt),
        DfirUnit::Sfp => Some(SendDestination::Sfp),
        DfirUnit::L0su => Some(SendDestination::L0su),
        _ => None,
    }
}

/// `dcc::getCoreId(unit)` AND `dcc::getCoreletId(unit)` AS ONE PAIR (`Utils/DccExtContext.cpp:78,116`)
/// — `None` is the `-1` either one returns for an absent attribute.
///
/// ⛔ `CoreWide` IS `corelet = 0`, NOT ABSENT (`UnitMaterializer.cpp:62-80`), which is the same reading
/// [`crate::transform::sentient::specialized_canonicalization`]'s `corelet_of` takes.
fn core_and_corelet(residency: Residency) -> Option<(Core, Corelet)> {
    let corelet = match residency {
        Residency::Global | Residency::Scratchpad { .. } => None,
        Residency::CoreWide { .. } => Corelet::checked(0),
        Residency::Corelet { corelet, .. } => Some(corelet),
    };
    residency.core().zip(corelet)
}

/// `set_dst_op.getUnits().getDefiningOp()` (`:252`, `:285`) — the op that binds the destination this
/// node's `sentient.set_send_dst` names, or `None` for a node that is not one.
fn set_send_dst_destination<'a>(op: &Op, defs: Definitions<'a>) -> Option<&'a Op> {
    let Op::Sentient(sentient::Op::SetSendDst { units }) = op else {
        return None;
    };
    defs.of(units.val())
}

/// Replaces: e379_initializeDataflowInfoForLXLU
///
/// The LXLU gen value a node starts with: none for a use, a query-map composite when the destination is
/// uniformized, else the kind of unit being sent to.
///
/// ⛔ TRAP: `None` IS NOT `unknown()` — the early return leaves `df_gen_` NULL, which generates nothing
/// at all (`Analyses/RedundantDefinitionEliminationTree.cpp:198`), while an unknown value IS a
/// definition. ⛔ TRAP: A DESTINATION THAT IS NEITHER `sfp`, `pt` NOR `l0su` — an operand that is no
/// `dataflow.get_unit` included — leaves `gen_mode` at `kUnknown` and aborts at `.hpp:98`.
#[must_use]
pub(crate) fn initialize_dataflow_info_for_lxlu(
    node: &RdeNode<'_>,
    defs: Definitions<'_>,
) -> Option<SetDstGenValueLxlu> {
    let RdeNode::At { op, .. } = node else {
        // `isOperationAUse(*nullptr)` — the root never reaches a `setDataflowGen` hook
        // (`Analyses/RedundantDefinitionEliminationTree.cpp:217,244-294`); e045 answers it the same way.
        return Some(SetDstGenValueLxlu::unknown());
    };
    // "Uses cannot generate definitions" (`:245`).
    if is_operation_a_use(op) {
        return None;
    }
    // "Only set_send_dst operations generate definitions, initially" (`:247`).
    let Some(destination) = set_send_dst_destination(op, defs) else {
        return Some(SetDstGenValueLxlu::unknown());
    };
    if let Op::Uniform(uniform::Op::QueryMap { result, map, key }) = destination {
        let qmap = QueryMapOp::of(*result, *map, *key);
        return Some(SetDstGenValueLxlu::composite(qmap, (*op).clone()));
    }
    let mode = match destination {
        Op::Dataflow(dataflow::Op::GetUnit { unit, .. }) => send_destination(*unit),
        _ => None,
    };
    match mode {
        Some(mode) => Some(SetDstGenValueLxlu::simple(mode, (*op).clone())),
        None => panic!(
            "DT_CHECK(mode != SimpleSetDstGenValueLXLU::Mode::kUnknown) \
             (`SetSendDestinationRE.hpp:98`)"
        ),
    }
}

/// Replaces: e380_initializeDataflowInfoForSFP
///
/// The SFP gen value a node starts with: none for a use, a query-map composite when the destination is
/// uniformized, else the core and corelet of the SFP being sent to.
///
/// ⛔ TRAP: THE DESTINATION MUST BE ANOTHER SFP AND MUST SIT ON A CORE. A non-`sfp` consumer aborts at
/// `:302-304`, and one whose `dataflow.get_unit` is global carries no `core` — so `core_id` stays `-1`
/// and it aborts at `.hpp:194` instead. ⭐ `None` IS NOT `unknown()`, for the reason
/// [`initialize_dataflow_info_for_lxlu`] records.
#[must_use]
pub(crate) fn initialize_dataflow_info_for_sfp(
    node: &RdeNode<'_>,
    defs: Definitions<'_>,
) -> Option<SetDstGenValueSfp> {
    let RdeNode::At { op, .. } = node else {
        return Some(SetDstGenValueSfp::unknown());
    };
    if is_operation_a_use(op) {
        return None;
    }
    let Some(destination) = set_send_dst_destination(op, defs) else {
        return Some(SetDstGenValueSfp::unknown());
    };
    if let Op::Uniform(uniform::Op::QueryMap { result, map, key }) = destination {
        let qmap = QueryMapOp::of(*result, *map, *key);
        return Some(SetDstGenValueSfp::composite(qmap, (*op).clone()));
    }
    let ids = match destination {
        Op::Dataflow(dataflow::Op::GetUnit {
            unit, residency, ..
        }) => {
            // `DT_CHECK_MSG((consumer_str.empty() || consumer_str == "sfp"), ...)` — a spelling is
            // never empty here, so the `||`'s first arm is unreachable.
            if *unit != DfirUnit::Sfp {
                panic!(
                    "DT_CHECK_MSG: expected the set_send_dst destination to be another SFP unit \
                     (`SetSendDestinationRE.cpp:302`)"
                );
            }
            core_and_corelet(*residency)
        }
        _ => None,
    };
    match ids {
        Some((core, corelet)) => Some(SetDstGenValueSfp::simple(core, corelet, (*op).clone())),
        None => {
            panic!("DT_CHECK(core_id >= 0 && corelet_id >= 0) (`SetSendDestinationRE.hpp:194`)")
        }
    }
}

/// Replaces: e476_initializeDataflowInfo
///
/// Gives one node its initial gen value, from whichever setting the unit is being optimised for.
///
/// ⭐ `opt_mode_` IS A PARAMETER, not a field: the tree is `Analyses/` work and out of campaign scope,
/// so what a ported hook can hold is the mode e195 handed the constructor (`.hpp:222-226`).
/// ⛔ THE `kUnknown` ARM IS REACHABLE ONLY BY MISUSE: e475 skips a unit that is neither all-LXLU nor
/// all-SFP before building a tree (`:110-120`), which is what makes the reference's third arm an
/// `llvm_unreachable` rather than a case.
#[must_use]
pub(crate) fn initialize_dataflow_info(
    mode: SetDestReOptimizationMode,
    node: &RdeNode<'_>,
    defs: Definitions<'_>,
) -> Option<SetDstGenValue> {
    match mode {
        SetDestReOptimizationMode::OptimizeForLxlu => {
            initialize_dataflow_info_for_lxlu(node, defs).map(SetDstGenValue::Lxlu)
        }
        SetDestReOptimizationMode::OptimizeForSfp => {
            initialize_dataflow_info_for_sfp(node, defs).map(SetDstGenValue::Sfp)
        }
        SetDestReOptimizationMode::Unknown => panic!(
            "llvm_unreachable(\"invalid optimization mode\") (`SetSendDestinationRE.cpp:241`)"
        ),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::link::{L0su as L0suUnit, Link, Lxlu as LxluUnit};
    use crate::islands::sentient::dialects::Val;
    use crate::transform::sentient::set_send_destination_re::composite_set_dst_gen_value_lxlu::CompositeSetDstGenValueLxlu;
    use crate::transform::sentient::set_send_destination_re::composite_set_dst_gen_value_sfp::CompositeSetDstGenValueSfp;
    use crate::transform::sentient::set_send_destination_re::set_dst_gen_value_lxlu::GenValue as LxluGenValue;
    use crate::transform::sentient::set_send_destination_re::set_dst_gen_value_sfp::GenValue as SfpGenValue;
    use crate::transform::sentient::set_send_destination_re::simple_set_dst_gen_value_lxlu::SimpleSetDstGenValueLxlu;
    use crate::transform::sentient::set_send_destination_re::simple_set_dst_gen_value_sfp::SimpleSetDstGenValueSfp;

    /// `sentient.load_compute_and_send` — the op the two `isa<>` lists disagree about.
    fn load_compute_and_send() -> Op {
        Op::Sentient(sentient::Op::LoadComputeAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            element_index: Val(4),
            scale_index: Val(5),
            consumer: Link::<LxluUnit, L0suUnit>::between(Val(6), Val(7)).ends().0,
            result: Val(8),
            src_total_elements: Elements(32),
            dst_total_elements: Elements(32),
            src_element_size: Bits(16),
            dst_element_size: Bits(16),
            dir: None,
            shuffle_mode: sentient::ShuffleMode::NoShuffle,
            reg: sentient::Reg {
                locale: sentient::RegType::Unknown,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// `sentient.set_send_dst` — the definition the tree exists to place.
    fn set_send_dst() -> Op {
        Op::Sentient(sentient::Op::SetSendDst {
            units: Link::<LxluUnit, L0suUnit>::between(Val(6), Val(7)).ends().0,
        })
    }

    /// `sentient.nop` — in neither list.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// e196 — a send is a use; the definition it places, and an op in neither list, are not.
    #[test]
    fn e196_only_the_sends_and_the_computes_are_uses() {
        assert!(is_operation_a_use(&load_compute_and_send()));
        assert!(!is_operation_a_use(&set_send_dst()));
        assert!(!is_operation_a_use(&nop()));
    }

    /// e197 — the root and a `set_send_dst` stay; a leaf in neither list goes; and so does a
    /// `load_compute_and_send`, which e196 calls a USE. That last one is the asymmetry, witnessed.
    #[test]
    fn e197_a_load_compute_and_send_leaf_is_simplifiable_though_it_is_a_use() {
        assert!(!is_simplifiable(&RdeNode::Root));
        let def = set_send_dst();
        assert!(!is_simplifiable(&RdeNode::At {
            op: &def,
            leaf: true
        }));
        let other = nop();
        assert!(is_simplifiable(&RdeNode::At {
            op: &other,
            leaf: true
        }));
        let use_op = load_compute_and_send();
        assert!(is_operation_a_use(&use_op));
        assert!(is_simplifiable(&RdeNode::At {
            op: &use_op,
            leaf: true
        }));
    }

    /// `dataflow.get_unit` binding `result` as one unit of `unit`'s kind.
    fn get_unit(result: Val, unit: DfirUnit, residency: Residency) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result,
            residency,
            unit,
            num_folds: None,
            reg_locale: None,
        })
    }

    /// `uniform.query_map` binding `result` — a uniformized destination.
    fn query_map(result: Val) -> Op {
        Op::Uniform(uniform::Op::QueryMap {
            result,
            map: Val(90),
            key: Val(91),
        })
    }

    /// `sentient.set_send_dst` whose `$units` is `dst`.
    fn set_send_dst_to(dst: Val) -> Op {
        Op::Sentient(sentient::Op::SetSendDst {
            units: Link::<LxluUnit, L0suUnit>::between(Val(0), dst).ends().0,
        })
    }

    /// A node over `op`, as the tree hands one to a `setDataflowGen` hook.
    fn node(op: &Op) -> RdeNode<'_> {
        RdeNode::At { op, leaf: true }
    }

    /// e379 — a use generates NOTHING, an op that is no `set_send_dst` the unknown value, a
    /// uniformized destination the composite, and each consumer kind its own send destination.
    #[test]
    fn e379_gives_each_consumer_kind_its_destination_and_a_use_no_definition() {
        let region = vec![
            get_unit(Val(21), DfirUnit::L0su, Residency::Global),
            get_unit(Val(22), DfirUnit::Sfp, Residency::Global),
            get_unit(Val(23), DfirUnit::CrossPtnLink, Residency::Global),
            query_map(Val(24)),
        ];
        let regions: [&[Op]; 1] = [region.as_slice()];
        let defs = Definitions::from_innermost(&regions);

        let use_op = load_compute_and_send();
        assert!(
            initialize_dataflow_info_for_lxlu(&node(&use_op), defs).is_none(),
            "a use leaves `df_gen_` null, which is not the unknown value"
        );

        let other = nop();
        let unknown = initialize_dataflow_info_for_lxlu(&node(&other), defs)
            .expect("a non-use generates a definition");
        assert!(unknown.is_unknown_value());

        for (dst, destination) in [
            (Val(21), SendDestination::L0su),
            (Val(22), SendDestination::Sfp),
            // "It covers crossptlink as well" — `find("pt")`, evaluated.
            (Val(23), SendDestination::Pt),
        ] {
            let op = set_send_dst_to(dst);
            let value = initialize_dataflow_info_for_lxlu(&node(&op), defs)
                .expect("a `set_send_dst` generates a definition");
            assert_eq!(
                value.gen_value(),
                LxluGenValue::Simple(SimpleSetDstGenValueLxlu::of(destination))
            );
            assert!(
                value.op().is_some(),
                "the generating operation travels with it"
            );
        }

        let uniformized = set_send_dst_to(Val(24));
        let value = initialize_dataflow_info_for_lxlu(&node(&uniformized), defs)
            .expect("a `set_send_dst` generates a definition");
        assert_eq!(
            value.gen_value(),
            LxluGenValue::Composite(CompositeSetDstGenValueLxlu::of(QueryMapOp::of(
                Val(24),
                Val(90),
                Val(91)
            )))
        );
    }

    /// e380 — the same four cases for the SFP, whose simple value is the destination SFP's core and
    /// corelet rather than a unit kind.
    #[test]
    fn e380_gives_the_destination_sfps_core_and_corelet_and_a_use_no_definition() {
        let core = Core::checked(1).expect("every arch this crate builds for has core 1");
        let corelet = Corelet::checked(0).expect("every core has corelet 0");
        let region = vec![
            get_unit(Val(31), DfirUnit::Sfp, Residency::Corelet { core, corelet }),
            get_unit(Val(32), DfirUnit::Sfp, Residency::CoreWide { core }),
            query_map(Val(33)),
        ];
        let regions: [&[Op]; 1] = [region.as_slice()];
        let defs = Definitions::from_innermost(&regions);

        let use_op = load_compute_and_send();
        assert!(initialize_dataflow_info_for_sfp(&node(&use_op), defs).is_none());

        let other = nop();
        let unknown = initialize_dataflow_info_for_sfp(&node(&other), defs)
            .expect("a non-use generates a definition");
        assert!(unknown.is_unknown_value());

        // ⭐ AND `CoreWide` READS BACK AS CORELET 0, not as an absent corelet.
        for dst in [Val(31), Val(32)] {
            let op = set_send_dst_to(dst);
            let value = initialize_dataflow_info_for_sfp(&node(&op), defs)
                .expect("a `set_send_dst` generates a definition");
            assert_eq!(
                value.gen_value(),
                SfpGenValue::Simple(SimpleSetDstGenValueSfp::of(core, corelet))
            );
        }

        let uniformized = set_send_dst_to(Val(33));
        let value = initialize_dataflow_info_for_sfp(&node(&uniformized), defs)
            .expect("a `set_send_dst` generates a definition");
        assert_eq!(
            value.gen_value(),
            SfpGenValue::Composite(CompositeSetDstGenValueSfp::of(QueryMapOp::of(
                Val(33),
                Val(90),
                Val(91)
            )))
        );
    }

    /// e476 — each mode reaches its own initialiser and answers with THAT mode's definition, and the
    /// mode e475 never builds a tree with reaches the reference's `llvm_unreachable`.
    #[test]
    fn e476_dispatches_on_the_optimization_mode() {
        let region = vec![get_unit(Val(41), DfirUnit::L0su, Residency::Global)];
        let regions: [&[Op]; 1] = [region.as_slice()];
        let defs = Definitions::from_innermost(&regions);
        let op = set_send_dst_to(Val(41));

        let lxlu =
            initialize_dataflow_info(SetDestReOptimizationMode::OptimizeForLxlu, &node(&op), defs);
        assert!(
            matches!(lxlu, Some(SetDstGenValue::Lxlu(_))),
            "the LXLU initialiser"
        );

        // ⭐ NOT THE SAME NODE: the SFP hook aborts on an `l0su` destination, so this arm is asked
        // with an op that is no `set_send_dst` at all — still that mode's own definition.
        let nothing = nop();
        let sfp = initialize_dataflow_info(
            SetDestReOptimizationMode::OptimizeForSfp,
            &node(&nothing),
            defs,
        );
        assert!(
            matches!(sfp, Some(SetDstGenValue::Sfp(_))),
            "the SFP initialiser"
        );

        let caught = std::panic::catch_unwind(|| {
            initialize_dataflow_info(SetDestReOptimizationMode::Unknown, &RdeNode::Root, defs)
        });
        let payload = caught.expect_err("the third arm is the reference's llvm_unreachable");
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|msg| (*msg).to_string()))
            .unwrap_or_default();
        assert!(
            message.contains("invalid optimization mode"),
            "the unreachable third arm"
        );
    }
}
