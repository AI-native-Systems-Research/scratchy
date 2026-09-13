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

//! `SetMaskRE.cpp` — 3 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e185_isOperationAUse` | 185 | 0 | 4 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:194` |
//! | `e186_isSimplifiable` | 186 | 0 | 10 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:221` |
//! | `e375_initializeDataflowInfo` | 375 | 1 | 17 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:203` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// module's own tests until something calls it. CI runs clippy with `-D warnings`, so without this the
// first ported leaf fails the gate.
// ⭐ REMOVE THIS WITH e536: `e474_runOn` has landed, but its body is blocked on the out-of-scope
// `SetMaskRDETree` and on e378, so it reaches nothing here — e536 is the one that wires the pass up.
#![allow(dead_code)]

use super::incr_mask_gen_value::IncrMaskGenValue;
use super::set_mask_gen_value::SetMaskGenValue;
use super::set_mask_rde_tree_optimizer::DataflowGen;
use crate::islands::sentient::dialects::{Op, sentient};
use crate::transform::sentient::analyses::RdeNode;
use crate::transform::sentient::cfg_simplification_sentient_level::pattern_simplification_manager::OpPath;

/// Replaces: e185_isOperationAUse
///
/// The four compute ops read the mask, and so does `sentient.incrmask` — which the pass counts as a
/// use AND a definition, which is what makes a use-free `incrmask` removable at all.
#[must_use]
pub(crate) fn is_operation_a_use(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(
            sentient::Op::VectorMac { .. }
                | sentient::Op::VectorUnary { .. }
                | sentient::Op::VectorBinary { .. }
                | sentient::Op::VectorTernary { .. }
                | sentient::Op::IncrMask { .. }
        )
    )
}

/// Replaces: e186_isSimplifiable
///
/// A subtree may go when it is neither the root nor one of the six mask-relevant ops, and is a leaf
/// — or a statically dead loop.
///
/// ⛔ TRAP: `isStaticallyDeadLoop` is `Analyses/` work and OUT OF CAMPAIGN SCOPE. The `||`
/// short-circuits, so a leaf never reaches it.
#[must_use]
pub(crate) fn is_simplifiable(node: &RdeNode<'_>) -> bool {
    let RdeNode::At { op, leaf } = node else {
        // `getRoot() == &node` — the root is the only node without an operation.
        return false;
    };
    if matches!(
        op,
        Op::Sentient(
            sentient::Op::SetMask { .. }
                | sentient::Op::IncrMask { .. }
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

/// `SetMaskRDETree::isOperationADef` (`SetMaskRE.cpp:199`) — a `set_mask` or an `incrmask`.
///
/// ⛔ NOT e044, WHICH IS `ImplicitSyncRDETree`'S OVERRIDE OF THE SAME NAME: this one is a two-op
/// `isa<>` that the campaign never scheduled, so it carries no anchor and is a private helper of the
/// one unit that reads it.
#[must_use]
fn is_operation_a_def(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(sentient::Op::SetMask { .. } | sentient::Op::IncrMask { .. })
    )
}

/// Replaces: e375_initializeDataflowInfo
///
/// The node's `setDataflowGen`: a `sentient.set_mask` generates its mask value, a `sentient.incrmask`
/// its increment, and any other selected node the unknown `set_mask` value.
///
/// ⛔ `None` IS THE EARLY `return`, WHICH SETS NOTHING — a node that is a use and not a definition
/// keeps whatever gen it had, which is not the same as being given the unknown value.
/// ⛔ AN `incrmask` PASSES THAT GATE THOUGH IT IS A USE: it is a def too (`SetMaskRE.cpp:199`), which
/// is what makes a use-free one removable at all. Only the four compute ops are turned away.
/// ⛔ `assert(setmask_op.getMaskValue())` IS THE ISLAND'S TYPE — `$mask_value` is a mandatory operand,
/// so there is no null to check.
/// ⛔ THE ROOT NEVER ARRIVES: the reference dereferences its null operation here, and `compute()`
/// calls this on selected nodes only (`Analyses/RedundantDefinitionEliminationTree.cpp:244-294`).
/// ⭐ `at` IS `node->getOperation()`'S IDENTITY, which both gens store as an [`OpPath`] rather than a
/// borrow — see [`SetMaskGenValue::op`].
#[must_use]
pub(crate) fn initialize_dataflow_info(node: &RdeNode<'_>, at: OpPath) -> Option<DataflowGen> {
    let RdeNode::At { op, .. } = node else {
        return None;
    };
    if is_operation_a_use(op) && !is_operation_a_def(op) {
        return None;
    }
    if let Op::Sentient(sentient::Op::SetMask { mask_value, .. }) = op {
        return Some(DataflowGen::SetMask(SetMaskGenValue::of(*mask_value, at)));
    }
    if let Some(incrmask) = IncrMaskGenValue::of_incr_mask(op, at) {
        return Some(DataflowGen::IncrMask(incrmask));
    }
    Some(DataflowGen::SetMask(SetMaskGenValue::unknown()))
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::sentient::dialects::Val;

    /// `sentient.set_mask %mask`.
    fn set_mask(mask: u32) -> Op {
        Op::Sentient(sentient::Op::SetMask {
            mask_value: Val(mask),
            dbg_name: None,
        })
    }

    /// `sentient.incrmask`.
    fn incrmask() -> Op {
        Op::Sentient(sentient::Op::IncrMask { dbg_name: None })
    }

    /// `sentient.nop` — an op that is neither a use nor a definition.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// A selected leaf node over `op`.
    fn node(op: &Op) -> RdeNode<'_> {
        RdeNode::At { op, leaf: true }
    }

    /// `sentient.vector_unary` — a compute op, i.e. a use that is not a definition.
    fn vector_unary() -> Op {
        Op::Sentient(sentient::Op::VectorUnary {
            mask: Val(0),
            op_a: sentient::Operand::from(sentient::Port::North),
            unary_op: sentient::UnaryOp::ReductionAdd,
            result: sentient::ResultPorts::default(),
            compute_precision: sentient::Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// e185 — `incrmask` is a use, `set_mask` is not, and neither is anything else.
    #[test]
    fn the_compute_ops_and_incrmask_are_uses() {
        assert!(is_operation_a_use(&incrmask()));
        assert!(!is_operation_a_use(&set_mask(1)));
        assert!(!is_operation_a_use(&nop()));
    }

    /// e375 — a `set_mask` and an `incrmask` each generate their own definition, a plain op the
    /// unknown one, and a compute op (a use that is not a def) nothing at all.
    #[test]
    fn only_set_mask_and_incrmask_generate_their_own_definition() {
        let at = OpPath::at(&[(0, 2)]);

        let set_mask_op = set_mask(9);
        let Some(DataflowGen::SetMask(value)) =
            initialize_dataflow_info(&node(&set_mask_op), at.clone())
        else {
            unreachable!("a set_mask generates a SetMaskGenValue")
        };
        assert_eq!(value.mask_value(), Some(Val(9)));
        assert_eq!(value.op(), Some(&at));

        let incrmask_op = incrmask();
        let Some(DataflowGen::IncrMask(value)) =
            initialize_dataflow_info(&node(&incrmask_op), at.clone())
        else {
            unreachable!("an incrmask generates an IncrMaskGenValue")
        };
        assert!(!value.is_unknown_value());
        assert_eq!(value.op(), Some(&at));

        let nop_op = nop();
        let Some(DataflowGen::SetMask(value)) =
            initialize_dataflow_info(&node(&nop_op), at.clone())
        else {
            unreachable!("any other selected op generates the unknown value")
        };
        assert!(value.is_unknown_value());

        let compute = vector_unary();
        assert!(
            initialize_dataflow_info(&node(&compute), at.clone()).is_none(),
            "a use that is not a def is left alone"
        );
        assert!(
            initialize_dataflow_info(&RdeNode::Root, at).is_none(),
            "the root has no operation"
        );
    }

    /// e186 — the root and the six mask-relevant ops stay; any other leaf goes.
    #[test]
    fn neither_the_root_nor_a_mask_relevant_op_is_simplifiable_but_another_leaf_is() {
        assert!(!is_simplifiable(&RdeNode::Root));
        for op in [set_mask(1), incrmask()] {
            assert!(!is_simplifiable(&RdeNode::At {
                op: &op,
                leaf: true
            }));
        }
        let other = nop();
        assert!(is_simplifiable(&RdeNode::At {
            op: &other,
            leaf: true
        }));
    }
}
