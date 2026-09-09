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

//! `SetActiveMaskValueRE.cpp` — 2 of the campaign's 656 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e180_initializeDataflowInfo` | 180 | 0 | 16 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.cpp:99` |
//! | `e181_isSimplifiable` | 181 | 0 | 9 | `dcc/src/Transform/Sentient/SetActiveMaskValueRE.cpp:116` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so both units below are reachable only from this
// file's own tests until `e582_runOnOperation` (level 4) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e582: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use super::SetActiveMaskValueGenValue;
use crate::islands::sentient::dialects::{Op, sentient};
use crate::transform::sentient::analyses::RdeNode;

/// `SetActiveMaskValueRDETree::isOperationAUse` (`SetActiveMaskValueRE.cpp:91`) — a `load_and_send`
/// reads the SAMV state and generates nothing.
///
/// ⭐ NOT A SCHEDULED UNIT of its own (unlike `SetMaskRDETree`'s e185), but e180's first line is a
/// call to it, so the predicate is spelled here rather than dropped.
#[must_use]
fn is_operation_a_use(op: &Op) -> bool {
    matches!(op, Op::Sentient(sentient::Op::LoadAndSend { .. }))
}

/// Replaces: e180_initializeDataflowInfo
///
/// The node's `setDataflowGen`: a `sentient.samv` generates its mask value and attribute dictionary,
/// every other non-use node the unknown value.
///
/// ⛔ `None` IS "`df_gen_` LEFT NULL", NOT AN UNKNOWN VALUE. The use arm returns before any
/// `setDataflowGen`, so a `load_and_send` node keeps the null the tree constructed it with — a
/// distinction `SetActiveMaskValueGenValue::unknown()` would erase.
///
/// ⛔ TRAP: `DT_CHECK(samv_op.getMaskValue())` (`:110`) IS NOW THE TYPE — `$mask_value` is a required
/// operand (`SentientOps.td:1019`), so the island cannot hold a `samv` without one.
///
/// ⛔ TRAP: THE REFERENCE DEREFERENCES A NULL OPERATION FOR THE ROOT (`*node->getOperation()` at
/// `:101`), which `compute()` never reaches because it calls this on selected nodes only
/// (`Analyses/RedundantDefinitionEliminationTree.cpp:217,247-294`). The root takes the "not a samv"
/// arm here and generates the unknown value.
#[must_use]
pub(crate) fn initialize_dataflow_info(node: &RdeNode<'_>) -> Option<SetActiveMaskValueGenValue> {
    let RdeNode::At { op, .. } = node else {
        return Some(SetActiveMaskValueGenValue::unknown());
    };
    if is_operation_a_use(op) {
        return None;
    }
    Some(
        SetActiveMaskValueGenValue::of_samv(op).unwrap_or_else(SetActiveMaskValueGenValue::unknown),
    )
}

/// Replaces: e181_isSimplifiable
///
/// A subtree may go when it is neither the root nor a `sentient.samv` nor a `load_and_send`, and is a
/// leaf — or a statically dead loop.
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
        Op::Sentient(sentient::Op::Samv { .. } | sentient::Op::LoadAndSend { .. })
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

#[cfg(test)]
mod unit_tests {
    use super::{initialize_dataflow_info, is_simplifiable};
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{
        RawPrecision, SliceId, ValidEntries, WslLen,
    };
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::transform::sentient::analyses::RdeNode;

    /// `sentient.samv mask_value(%7) {...}`.
    fn samv() -> Op {
        Op::Sentient(sentient::Op::Samv {
            mask_value: Val(7),
            mask_all: false,
            num_valid_entry: ValidEntries(5),
            slice_id_xsl: SliceId(4),
            xsl_inner: false,
            wsl_len: WslLen(2),
            precision: RawPrecision(16),
            dbg_name: None,
        })
    }

    /// `sentient.load_and_send` — the one operation this tree calls a USE.
    fn load_and_send() -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: SendEnd::to_self(Val(4)),
            result: Val(5),
            extent: sentient::Extent {
                total_elements: Elements(64),
                element_size: Bits(32),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(1),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: sentient::ShuffleMode::NoShuffle,
            reg: sentient::Reg {
                locale: sentient::RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// `sentient.nop` — neither a def nor a use.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// e180 — a samv generates its value and dictionary, a `load_and_send` generates NO definition at
    /// all, and everything else the unknown value.
    #[test]
    fn only_a_samv_generates_a_definition_and_a_use_generates_none() {
        let samv_op = samv();
        let generated = initialize_dataflow_info(&RdeNode::At {
            op: &samv_op,
            leaf: true,
        })
        .expect("a samv sets its dataflow gen");
        assert_eq!(generated.mask_value(), Some(Val(7)));
        assert!(!generated.is_unknown_value());

        let other = nop();
        assert!(
            initialize_dataflow_info(&RdeNode::At {
                op: &other,
                leaf: true
            })
            .expect("a non-use sets an unknown gen")
            .is_unknown_value()
        );
        assert!(
            initialize_dataflow_info(&RdeNode::Root)
                .expect("the root takes the not-a-samv arm")
                .is_unknown_value()
        );

        // The use arm returns BEFORE `setDataflowGen`, so `df_gen_` stays null.
        let use_op = load_and_send();
        assert!(
            initialize_dataflow_info(&RdeNode::At {
                op: &use_op,
                leaf: true
            })
            .is_none(),
            "a load_and_send generates no definition at all"
        );
    }

    /// e181 — the root, a samv and a `load_and_send` all hold their subtree; any other leaf goes.
    #[test]
    fn neither_the_root_nor_a_samv_is_simplifiable_but_another_leaf_is() {
        let samv_op = samv();
        assert!(!is_simplifiable(&RdeNode::Root));
        assert!(!is_simplifiable(&RdeNode::At {
            op: &samv_op,
            leaf: true
        }));
        let use_op = load_and_send();
        assert!(!is_simplifiable(&RdeNode::At {
            op: &use_op,
            leaf: true
        }));
        let other = nop();
        assert!(is_simplifiable(&RdeNode::At {
            op: &other,
            leaf: true
        }));
    }
}
