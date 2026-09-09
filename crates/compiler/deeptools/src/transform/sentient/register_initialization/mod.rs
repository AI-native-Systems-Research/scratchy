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

//! `RegisterInitialization.cpp` — 6 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e131_runLocalAnalysis` | 131 | 0 | 21 | `dcc/src/Transform/Sentient/RegisterInitialization.cpp:136` |
//! | `e345_run` | 345 | 1 | 14 | `dcc/src/Transform/Sentient/RegisterInitialization.cpp:121` |
//! | `e346_runGlobalAnalysis` | 346 | 1 | 5 | `dcc/src/Transform/Sentient/RegisterInitialization.cpp:158` |
//! | `e347_runTransformation` | 347 | 1 | 3 | `dcc/src/Transform/Sentient/RegisterInitialization.cpp:164` |
//! | `e458_runOn` | 458 | 2 | 68 | `dcc/src/Transform/Sentient/RegisterInitialization.cpp:182` |
//! | `e521_runOnOperation` | 521 | 3 | 16 | `dcc/src/Transform/Sentient/RegisterInitialization.cpp:261` |

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e521_runOnOperation` (level 3) is what reaches
// this file's driver, and every unit below is reachable only from the tests until it lands. CI runs
// clippy with `-D warnings`. ⭐ REMOVE THIS WITH e521.

use crate::transform::sentient::analyses::{
    Candidate, CandidateCollector, CandidateEvaluator, CandidateSelector, UniformGroups,
};

/// Replaces: e131_runLocalAnalysis
///
/// Phase 1 of the `Driver`: the global candidates are collected and evaluated, then every group
/// leader's local candidates are collected, evaluated, selected against the globals and merged.
///
/// ⛔ THE ORDER IS THE PORT — the globals are merged into `result` LAST, after every local
/// selection, because `selectLocally` may purge that list (`RegisterInitialization.cpp:149-155`).
/// ⛔ EVERY COLLABORATOR IS OUT OF CAMPAIGN SCOPE (`RegisterInitialization/`), so what is ported is
/// the sequence; each call lands on a seam whose only crate implementation is a `todo!`.
pub fn run_local_analysis(
    result: &mut Vec<Candidate>,
    collector: &mut dyn CandidateCollector,
    evaluator: &mut dyn CandidateEvaluator,
    selector: &mut dyn CandidateSelector,
    groups: &dyn UniformGroups,
) {
    let mut global_candidates: Vec<Candidate> = Vec::new();
    collector.collect_global_candidates(&mut global_candidates);
    evaluator.evaluate_locally(&mut global_candidates);

    for core in groups.group_leaders() {
        let mut local_candidates: Vec<Candidate> = Vec::new();
        collector.collect_local_candidates(core, &mut local_candidates);
        evaluator.evaluate_locally(&mut local_candidates);
        selector.select_locally(&mut local_candidates, &mut global_candidates, core);
        evaluator.merge_into(result, &local_candidates);
    }
    evaluator.merge_into(result, &global_candidates);
}

// crustify:todo: e345_run
//   authority : dcc/src/Transform/Sentient/RegisterInitialization.cpp:121  (14 body lines, level 1)
//   original  : void run()
//   calls     : e131_runLocalAnalysis, e252_size, e346_runGlobalAnalysis, e347_runTransformation

// crustify:todo: e346_runGlobalAnalysis
//   authority : dcc/src/Transform/Sentient/RegisterInitialization.cpp:158  (5 body lines, level 1)
//   original  : void runGlobalAnalysis(ListOfCandidatesRef candidates)
//   calls     : e345_run

// crustify:todo: e347_runTransformation
//   authority : dcc/src/Transform/Sentient/RegisterInitialization.cpp:164  (3 body lines, level 1)
//   original  : void runTransformation(ConstListOfCandidatesRef candidates)
//   calls     : e345_run

// crustify:todo: e458_runOn
//   authority : dcc/src/Transform/Sentient/RegisterInitialization.cpp:182  (68 body lines, level 2)
//   original  : void runOn(dataflow::ProgramUnitOp prog_unit, Liveness &liveness)
//   calls     : e345_run

// crustify:todo: e521_runOnOperation
//   authority : dcc/src/Transform/Sentient/RegisterInitialization.cpp:261  (16 body lines, level 3)
//   original  : void RegisterInitializationPass::runOnOperation()
//   calls     : e458_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::sentient::dialects::Val;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// WHICH COLLABORATOR WAS ASKED WHAT, IN ORDER — the only observable a driver phase has.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Call {
        CollectGlobal,
        EvaluateLocally(Vec<Candidate>),
        CollectLocal(Val),
        SelectLocally(Val),
        MergeInto(Vec<Candidate>),
    }

    /// The one log the four seams write to — the reference's four references are four distinct
    /// objects, so each fake is its own value sharing this.
    type Log = Rc<RefCell<Vec<Call>>>;

    struct Collector(Log);
    struct Evaluator(Log);
    struct Selector(Log);
    struct Groups(Vec<Val>);

    impl CandidateCollector for Collector {
        fn collect_global_candidates(&mut self, results: &mut Vec<Candidate>) {
            self.0.borrow_mut().push(Call::CollectGlobal);
            results.push(Candidate(100));
        }

        fn collect_local_candidates(&mut self, core: Val, results: &mut Vec<Candidate>) {
            self.0.borrow_mut().push(Call::CollectLocal(core));
            results.push(Candidate(core.0));
        }
    }

    impl CandidateEvaluator for Evaluator {
        fn evaluate_locally(&mut self, candidates: &mut Vec<Candidate>) {
            self.0
                .borrow_mut()
                .push(Call::EvaluateLocally(candidates.clone()));
        }

        fn merge_into(&mut self, result: &mut Vec<Candidate>, sublist: &[Candidate]) {
            self.0.borrow_mut().push(Call::MergeInto(sublist.to_vec()));
            result.extend_from_slice(sublist);
        }
    }

    impl CandidateSelector for Selector {
        fn select_locally(
            &mut self,
            _local: &mut Vec<Candidate>,
            global: &mut Vec<Candidate>,
            core: Val,
        ) {
            self.0.borrow_mut().push(Call::SelectLocally(core));
            // The purge the reference's note is about: this selector keeps no global candidate.
            global.clear();
        }
    }

    impl UniformGroups for Groups {
        fn group_leaders(&self) -> Vec<Val> {
            self.0.clone()
        }
    }

    /// e131_runLocalAnalysis — the sequence, and the globals merged LAST and already purged.
    #[test]
    fn e131_run_local_analysis() {
        let log: Log = Rc::new(RefCell::new(Vec::new()));
        let mut collector = Collector(Rc::clone(&log));
        let mut evaluator = Evaluator(Rc::clone(&log));
        let mut selector = Selector(Rc::clone(&log));
        let groups = Groups(vec![Val(1), Val(2)]);
        let mut result: Vec<Candidate> = Vec::new();

        run_local_analysis(
            &mut result,
            &mut collector,
            &mut evaluator,
            &mut selector,
            &groups,
        );

        assert_eq!(
            *log.borrow(),
            vec![
                Call::CollectGlobal,
                Call::EvaluateLocally(vec![Candidate(100)]),
                Call::CollectLocal(Val(1)),
                Call::EvaluateLocally(vec![Candidate(1)]),
                Call::SelectLocally(Val(1)),
                Call::MergeInto(vec![Candidate(1)]),
                Call::CollectLocal(Val(2)),
                Call::EvaluateLocally(vec![Candidate(2)]),
                Call::SelectLocally(Val(2)),
                Call::MergeInto(vec![Candidate(2)]),
                // ⭐ THE POINT: the globals arrive last, and by then the selector has purged them.
                Call::MergeInto(Vec::new()),
            ]
        );
        assert_eq!(result, vec![Candidate(1), Candidate(2)]);
    }
}
