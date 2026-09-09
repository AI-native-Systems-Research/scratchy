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
// module's own tests until `e474_runOn`/`e536_runOnOperation` (levels 2/3) land and something calls
// it. CI runs clippy with `-D warnings`, so without this the first ported leaf fails the gate.
// ⭐ REMOVE THIS WITH e474: at that point an unused item here is a real defect again.
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

/// `SetMaskRDETree::isOperationADef` (`SetMaskRE.cpp:199`) — the two ops that generate a definition.
///
/// ⛔ NOT `e044_isOperationADef`, WHICH THE WORKLIST NAMES BY COLLISION: that unit is
/// `ImplicitSyncRDETree`'s override (`ImplicitSyncRE.cpp:94`, a `sentient.sync` with a positive
/// boundary). This pass's override is a different predicate on a different set of ops and is no
/// scheduled unit of its own, so it lives here as a helper rather than being borrowed from that pass.
fn is_operation_a_def(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(sentient::Op::SetMask { .. } | sentient::Op::IncrMask { .. })
    )
}

/// Replaces: e375_initializeDataflowInfo
///
/// The node's `setDataflowGen`: a `sentient.set_mask` generates its mask value, a `sentient.incrmask`
/// its increment, and every other selected node the unknown set-mask value.
///
/// ⛔ `None` IS THE EARLY `return`, AND IT IS NOT THE UNKNOWN VALUE: a use that is not also a
/// definition — the four compute ops — is left with NO gen at all, which is what makes e193's
/// `if (!sib_gen) break` and e194's `else { break }` stop at one.
///
/// ⛔ `assert(setmask_op.getMaskValue())` IS THE ISLAND: `sentient.set_mask` carries a non-optional
/// `mask_value`, so the abort has no input left to fire on.
#[must_use]
pub(crate) fn initialize_dataflow_info(node: &RdeNode<'_>, at: OpPath) -> Option<DataflowGen> {
    let RdeNode::At { op, .. } = node else {
        // The root has no operation, so it is neither a use nor a set_mask nor an incrmask.
        return Some(DataflowGen::SetMask(SetMaskGenValue::unknown()));
    };
    if is_operation_a_use(op) && !is_operation_a_def(op) {
        return None;
    }
    match op {
        Op::Sentient(sentient::Op::SetMask { mask_value, .. }) => Some(DataflowGen::SetMask(
            SetMaskGenValue::of(*mask_value, at),
        )),
        Op::Sentient(sentient::Op::IncrMask { .. }) => IncrMaskGenValue::of_incr_mask(op, at)
            .map(DataflowGen::IncrMask)
            .or_else(|| Some(DataflowGen::SetMask(SetMaskGenValue::unknown()))),
        _ => Some(DataflowGen::SetMask(SetMaskGenValue::unknown())),
    }
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

    /// `sentient.vector_unary` — a use of the mask that is not a definition of it.
    fn vector_unary() -> Op {
        Op::Sentient(sentient::Op::VectorUnary {
            mask: Val(90),
            op_a: sentient::Operand::from(sentient::Port::North),
            unary_op: sentient::UnaryOp::Floor,
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

    /// e375 — a `set_mask` generates its mask value, an `incrmask` an increment, a use-only op NO gen
    /// at all, and anything else the unknown set-mask value.
    #[test]
    fn only_a_set_mask_or_an_incrmask_generates_a_real_definition() {
        let at = OpPath::at(&[(0, 2)]);
        fn node(op: &Op) -> RdeNode<'_> {
            RdeNode::At { op, leaf: true }
        }

        let mask = set_mask(4);
        let Some(DataflowGen::SetMask(mask_gen)) =
            initialize_dataflow_info(&node(&mask), at.clone())
        else {
            unreachable!("a set_mask generates a set-mask definition")
        };
        assert_eq!(mask_gen.mask_value(), Some(Val(4)));
        assert_eq!(mask_gen.op(), Some(&at));

        let incr = incrmask();
        let Some(DataflowGen::IncrMask(incr_gen)) =
            initialize_dataflow_info(&node(&incr), at.clone())
        else {
            unreachable!("an incrmask is both a use and a definition")
        };
        assert!(!incr_gen.is_unknown_value());
        assert_eq!(incr_gen.op(), Some(&at));

        let unary = vector_unary();
        assert!(
            initialize_dataflow_info(&node(&unary), at.clone()).is_none(),
            "a use that is not a definition gets no gen, which is what ends a sibling search"
        );

        let other = nop();
        let Some(DataflowGen::SetMask(unknown)) =
            initialize_dataflow_info(&node(&other), at.clone())
        else {
            unreachable!("every other selected node gets the unknown set-mask value")
        };
        assert!(unknown.is_unknown_value());
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
