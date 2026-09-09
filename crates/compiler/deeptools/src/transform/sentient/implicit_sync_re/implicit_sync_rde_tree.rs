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

//! `ImplicitSyncRE.cpp` — 3 of the campaign's 656 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e044_isOperationADef` | 044 | 0 | 6 | `dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:94` |
//! | `e045_initializeDataflowInfo` | 045 | 0 | 12 | `dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:101` |
//! | `e046_isSimplifiable` | 046 | 0 | 7 | `dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:114` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e559_runOnOperation` (level 4) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e559: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use core::num::NonZeroU32;

use super::{ImplicitSyncGenValue, TileSize};
use crate::islands::sentient::dialects::{Op, sentient};

/// ⭐ THE RDE NODE IS THE CAMPAIGN-WIDE OUT-OF-SCOPE SEAM, not this pass's own type: four RDE passes
/// override the same two hooks and ask a node the same two questions.
pub(crate) use crate::transform::sentient::analyses::RdeNode;

/// A `sentient.sync` WHOSE `implicit_sync_memory_boundary` IS POSITIVE.
///
/// ⭐ ONE WITNESS DISCHARGES TWO UNITS: it is e044's `true`, and it is also e045's `DT_CHECK`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DefiningSync(TileSize);

impl DefiningSync {
    /// Replaces: e044_isOperationADef
    ///
    /// A `sentient.sync` is a definition exactly when its `implicit_sync_memory_boundary` is present
    /// and positive — `Some` here IS the reference's `true`.
    ///
    /// ⛔ TRAP: `None` is the `.td`'s `-1` (see [`sentient::Op::Sync`]), so an absent boundary and a
    /// non-positive one give the same answer.
    #[must_use]
    pub(crate) fn of(op: &Op) -> Option<DefiningSync> {
        let Op::Sentient(sentient::Op::Sync {
            implicit_sync_memory_boundary,
            ..
        }) = op
        else {
            return None;
        };
        u32::try_from(implicit_sync_memory_boundary.unwrap_or(0))
            .ok()
            .and_then(NonZeroU32::new)
            .map(|size| DefiningSync(TileSize::of(size)))
    }

    /// The boundary that made it a definition.
    #[must_use]
    pub(crate) const fn tile_size(self) -> TileSize {
        self.0
    }
}

/// Replaces: e045_initializeDataflowInfo
///
/// The node's `setDataflowGen`: a defining `sentient.sync` generates its boundary as a tile size,
/// every other node the unknown value.
///
/// ⛔ TRAP: `DT_CHECK(has_value())` (`:107`) IS UNREACHABLE AND IS NOW A TYPE. `isOperationSelected`
/// admits a `sentient.sync` only through `isOperationADef` and `compute()` calls this on selected
/// nodes alone (`Analyses/RedundantDefinitionEliminationTree.cpp:217,247-294`), so [`DefiningSync`]
/// is the check — and unlike the abort it has an answer for the input the check cannot get.
#[must_use]
pub(crate) fn initialize_dataflow_info(node: &RdeNode<'_>) -> ImplicitSyncGenValue {
    let RdeNode::At { op, .. } = node else {
        return ImplicitSyncGenValue::unknown();
    };
    DefiningSync::of(op).map_or_else(ImplicitSyncGenValue::unknown, |sync| {
        ImplicitSyncGenValue::of(sync.tile_size(), (*op).clone())
    })
}

/// Replaces: e046_isSimplifiable
///
/// A subtree may go when it is neither the root nor a `sentient.sync`, and is a leaf — or a
/// statically dead loop.
///
/// ⛔ TRAP: `isStaticallyDeadLoop` is `Analyses/` work and OUT OF CAMPAIGN SCOPE. The `||`
/// short-circuits, so a leaf never reaches it.
#[must_use]
pub(crate) fn is_simplifiable(node: &RdeNode<'_>) -> bool {
    let RdeNode::At { op, leaf } = node else {
        // `getRoot() == &node` — the root is the only node without an operation.
        return false;
    };
    if matches!(op, Op::Sentient(sentient::Op::Sync { .. })) {
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
    use super::*;

    /// `sentient.sync` with the given `implicit_sync_memory_boundary`.
    fn sync(boundary: Option<i32>) -> Op {
        Op::Sentient(sentient::Op::Sync {
            mode: sentient::SyncMode::Send,
            peers: Vec::new(),
            soft: false,
            implicit_sync_memory_boundary: boundary,
            dbg_name: None,
        })
    }

    /// `sentient.nop` — an op that is not a sync.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    #[test]
    fn only_a_sync_with_a_positive_boundary_is_a_definition() {
        assert!(DefiningSync::of(&sync(Some(4096))).is_some());
        assert!(DefiningSync::of(&sync(Some(0))).is_none());
        assert!(DefiningSync::of(&sync(Some(-1))).is_none());
        assert!(DefiningSync::of(&sync(None)).is_none());
        assert!(DefiningSync::of(&nop()).is_none());
    }

    #[test]
    fn a_defining_sync_generates_its_boundary_and_every_other_node_the_unknown_value() {
        let defining = sync(Some(4096));
        let generated = initialize_dataflow_info(&RdeNode::At {
            op: &defining,
            leaf: true,
        });
        assert_eq!(
            generated.tile_size().map(|size| size.get().get()),
            Some(4096)
        );
        let other = nop();
        assert!(
            initialize_dataflow_info(&RdeNode::At {
                op: &other,
                leaf: true
            })
            .is_unknown_value()
        );
        assert!(initialize_dataflow_info(&RdeNode::Root).is_unknown_value());
    }

    #[test]
    fn neither_the_root_nor_a_sync_is_simplifiable_but_a_leaf_is() {
        let defining = sync(Some(4096));
        assert!(!is_simplifiable(&RdeNode::Root));
        assert!(!is_simplifiable(&RdeNode::At {
            op: &defining,
            leaf: true
        }));
        let other = nop();
        assert!(is_simplifiable(&RdeNode::At {
            op: &other,
            leaf: true
        }));
    }
}
