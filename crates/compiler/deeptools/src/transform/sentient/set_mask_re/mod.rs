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

//! `SetMaskRE.cpp` — 2 of the campaign's 656 units (dependency level(s) [2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e474_runOn` | 474 | 2 | 26 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:151` |
//! | `e536_runOnOperation` | 536 | 3 | 6 | `dcc/src/Transform/Sentient/SetMaskRE.cpp:144` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e536_runOnOperation` (level 3) is what calls
// [`run_on_unit`], and nothing but this file's own tests reaches it until that lands. CI runs clippy
// with `-D warnings`. ⭐ REMOVE THIS WITH e536.
#![allow(dead_code)]

pub(crate) mod incr_mask_gen_value;
pub(crate) mod set_mask_gen_value;
pub(crate) mod set_mask_rde_tree;
pub(crate) mod set_mask_rde_tree_optimizer;

use crate::arch::Arch;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::sentient::ProgramUnit;


/// `Statistic<"set_mask_re_count", "num-setmask-eliminated", "Number of times `set_mask` or
/// `incrmask` operations were removed or hoisted">` (`Transform/Sentient/Passes.td:174`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SetMaskReCount(pub(crate) u32);

/// Replaces: e474_runOn
///
/// Runs `set_mask`/`incrmask` redundancy elimination over ONE PT unit, and banks how many ops it
/// removed or hoisted.
///
/// ⛔ ANYTHING BUT THE PT IS SKIPPED AND ITS COUNT LEFT ALONE (`:153-155`) — the pass is *"removes
/// redundant set_mask operations in the PT unit"* (`Passes.td:169`), and `set_mask_re_count` is a
/// per-module statistic, so a skipped unit must not zero what a PT unit before it banked.
/// ⛔ THE REST OF THE BODY IS BLOCKED BY TWO DIFFERENT THINGS: `SetMaskRDETree`'s construction,
/// `compute` and `simplify` are `RedundantDefinitionEliminationTree`'s and OUT OF CAMPAIGN SCOPE,
/// while `SetMaskRDETreeOptimizer::optimize()` is this campaign's own e378.
pub(crate) fn run_on_unit<A: Arch>(
    unit: &mut ProgramUnit<A>,
    set_mask_re_count: &mut SetMaskReCount,
) {
    if unit.on.kind().generic() != GenericComp::Pt {
        return;
    }
    let _ = set_mask_re_count;
    todo!(
        "e474_runOn: SetMaskRDETree's construction, compute() and simplify() \
         (Analyses/RedundantDefinitionEliminationTree.hpp) are out of campaign scope, as is the \
         DT_CHECK over dcc_ext_ctx_.isa_per_unit_, and SetMaskRDETreeOptimizer::optimize() is not \
         ported yet (senpass e378, SetMaskRE.cpp:311) — together they are SetMaskRE.cpp:157-175"
    )
}

// crustify:todo: e536_runOnOperation
//   authority : dcc/src/Transform/Sentient/SetMaskRE.cpp:144  (6 body lines, level 3)
//   original  : void runOnOperation()
//   calls     : e474_runOn

#[cfg(test)]
mod unit_tests {
    use super::{SetMaskReCount, run_on_unit};
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::sentient::ProgramUnit;
    use crate::islands::sentient::dialects::Val;
    use crate::units::{DfirUnit, Row};

    /// One empty unit on `kind`.
    fn unit_on(kind: DfirUnit) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(kind, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        }
    }

    /// e474 — anything but the PT is skipped, and the module-wide count it did not touch stays put.
    #[test]
    fn a_unit_that_is_not_the_pt_is_skipped_and_the_count_kept() {
        let mut count = SetMaskReCount(3);
        run_on_unit(&mut unit_on(DfirUnit::Lxlu), &mut count);
        assert_eq!(count, SetMaskReCount(3));
    }

    /// e474 — a PT unit runs the tree, whose optimizer is e378 and not ported.
    #[test]
    #[should_panic(expected = "senpass e378")]
    fn the_pt_reaches_the_unported_rde_tree_optimizer() {
        run_on_unit(
            &mut unit_on(DfirUnit::PtRow(Row::checked(0).expect("PT row 0 exists"))),
            &mut SetMaskReCount(0),
        );
    }
}
