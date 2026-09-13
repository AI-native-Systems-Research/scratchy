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

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET. `e521_runOnOperation` — the pass entry — has landed
// as [`run_on_operation`], and nothing in this crate calls it, so every item below is still reachable
// only from the tests. CI runs clippy with `-D warnings`.
// ⭐ REMOVE THIS WHEN A PIPELINE CALLS `run_on_operation`: from then on an unused item here is a real
// defect.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::transform::sentient::analyses::{
    Candidate, CandidateCollector, CandidateEvaluator, CandidateSelector, InstructionEstimator,
    Liveness, OutOfScopeCandidateCollector, OutOfScopeCandidateEvaluator,
    OutOfScopeCandidateSelector, OutOfScopeTransformer, OutOfScopeUniformGrouper,
    OutOfScopeUniformGroups, Transformer, UniformGrouper, UniformGroups,
};
use crate::workload::Workload;

/// THE `-dcc-register-initialization-collector` VALUES — `dcc::reginit::CollectorKind`
/// (`RegisterInitialization/Collector.h:28`) less the `kUnknown` the option never offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CollectorKind {
    /// `simple` — the only value declared, and the `cl::init` (`RegisterInitialization.cpp:36-41`).
    #[default]
    Simple,
}

/// THE `-dcc-register-initialization-evaluator` VALUES — `dcc::reginit::EvaluatorKind`
/// (`RegisterInitialization/Evaluator.h:23`) less `kUnknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvaluatorKind {
    /// `simple` — the only value declared, and the `cl::init` (`RegisterInitialization.cpp:43-48`).
    #[default]
    Simple,
}

/// THE `-dcc-register-initialization-selector` VALUES — `dcc::reginit::SelectorKind`
/// (`RegisterInitialization/Selector.h:32-40`) less `kUnknown`.
///
/// ⛔ THE DEFAULT IS `kBisect`, NOT the greedy algorithm the option's own help text calls "old"
/// (`RegisterInitialization.cpp:50-67`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectorKind {
    /// `greedy` — colourability checked one candidate at a time.
    Greedy,
    /// `const-decay`, constructed with `/* backtrack */ true`.
    ConstantDecay,
    /// `linear-decay`, constructed with `/* backtrack */ true`.
    LinearDecay,
    /// `polynomial-decay`, constructed with `/* backtrack */ true`.
    PolynomialDecay,
    /// `bisect` — the `cl::init`.
    #[default]
    Bisect,
}

/// `collectorKind`, `evaluatorKind`, `selectorKind` — the three `cl::opt`s' inits
/// (`RegisterInitialization.cpp:36-67`). ⛔ NOT PASS OPTIONS: nothing in `dcc/src` assigns them, so
/// these are the kinds every build runs.
const COLLECTOR_KIND: CollectorKind = CollectorKind::Simple;
const EVALUATOR_KIND: EvaluatorKind = EvaluatorKind::Simple;
const SELECTOR_KIND: SelectorKind = SelectorKind::Bisect;

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

/// Replaces: e345_run
///
/// The `Driver`'s whole flow over ONE candidate list: local analysis, then global analysis, then the
/// transformation.
///
/// ⛔ ONE LIST THREADED THROUGH ALL THREE PHASES — [`run_local_analysis`] fills it,
/// [`run_global_analysis`] rewrites it in place (the grouper may replace locals with globals), and
/// [`run_transformation`] only reads it.
/// ⭐ THE TWO `LLVM_DEBUG` DUMPS ARE DROPPED: `candidates.size()` and `dcc::reginit::dump` have no
/// effect on the IR, which is why `e252_size` is a call this port does not make.
pub fn run(
    collector: &mut dyn CandidateCollector,
    evaluator: &mut dyn CandidateEvaluator,
    selector: &mut dyn CandidateSelector,
    grouper: &mut dyn UniformGrouper,
    transformer: &mut dyn Transformer,
    groups: &dyn UniformGroups,
) {
    let mut candidates: Vec<Candidate> = Vec::new();
    run_local_analysis(&mut candidates, collector, evaluator, selector, groups);
    run_global_analysis(&mut candidates, grouper, evaluator, selector);
    run_transformation(&candidates, transformer);
}

/// Replaces: e346_runGlobalAnalysis
///
/// Phase 2: the grouper regroups the merged list, then it is evaluated and selected globally.
///
/// ⛔ THE ORDER IS THE PORT, and the grouping comes FIRST because it is what puts the global
/// candidates in the list the other two then judge.
pub fn run_global_analysis(
    candidates: &mut Vec<Candidate>,
    grouper: &mut dyn UniformGrouper,
    evaluator: &mut dyn CandidateEvaluator,
    selector: &mut dyn CandidateSelector,
) {
    grouper.run(candidates);
    evaluator.evaluate_globally(candidates);
    selector.select_globally(candidates);
}

/// Replaces: e347_runTransformation
///
/// Phase 3: the transformer applies the surviving candidates to the IR and to the liveness object.
///
/// ⭐ THE LIST IS READ-ONLY HERE — `ConstListOfCandidatesRef` (`RegisterInitialization.cpp:164`), so
/// nothing this phase does can change which candidates there were.
pub fn run_transformation(candidates: &[Candidate], transformer: &mut dyn Transformer) {
    transformer.run(candidates);
}

/// Replaces: e458_runOn
///
/// One program unit's whole register initialisation: run the uniform-group analysis, build the
/// collaborators the three options choose between, and hand the chosen three to the [`run`] driver.
///
/// ⛔ THE THREE `DT_ERROR` DEFAULTS ARE INEXPRESSIBLE — each option is a closed `enum` WITHOUT the
/// `kUnknown` the `cl::opt` never offers, so "unexpected kind" is not a state this can be in.
/// ⭐ EVERY COLLABORATOR IS CONSTRUCTED HERE AND EVERY ONE IS OUT OF CAMPAIGN SCOPE
/// (`RegisterInitialization/`, `Analyses/UniformGroupAnalysis`): the crate has one implementation of
/// each interface, so the ctor arguments — `prog_unit`, `liveness`, `dccExtContext()` and
/// `/* backtrack */ true` — reach seams that keep nothing, and the unit stops at the first of them.
pub fn run_on<A: Arch>(unit: &mut ProgramUnit<A>, liveness: &mut dyn Liveness) {
    let mut uga = OutOfScopeUniformGroups;
    uga.collect_exclusive_group_leaders();

    // `SimpleCollector(prog_unit, liveness, uga)` and the eight collaborators after it: one crate
    // implementation per interface, and not one of them keeps what the reference hands its ctor.
    let _ = (unit, liveness);
    let mut simple_collector = OutOfScopeCandidateCollector;
    let mut simple_evaluator = OutOfScopeCandidateEvaluator;
    let mut greedy_selector = OutOfScopeCandidateSelector;
    let mut cd_selector = OutOfScopeCandidateSelector;
    let mut ld_selector = OutOfScopeCandidateSelector;
    let mut pd_selector = OutOfScopeCandidateSelector;
    let mut bisect_selector = OutOfScopeCandidateSelector;
    let mut uniform_grouper = OutOfScopeUniformGrouper;
    let mut transformer = OutOfScopeTransformer;

    let chosen_collector: &mut dyn CandidateCollector = match COLLECTOR_KIND {
        CollectorKind::Simple => &mut simple_collector,
    };
    let chosen_evaluator: &mut dyn CandidateEvaluator = match EVALUATOR_KIND {
        EvaluatorKind::Simple => &mut simple_evaluator,
    };
    let chosen_selector: &mut dyn CandidateSelector = match SELECTOR_KIND {
        SelectorKind::Greedy => &mut greedy_selector,
        SelectorKind::ConstantDecay => &mut cd_selector,
        SelectorKind::LinearDecay => &mut ld_selector,
        SelectorKind::PolynomialDecay => &mut pd_selector,
        SelectorKind::Bisect => &mut bisect_selector,
    };

    run(
        chosen_collector,
        chosen_evaluator,
        chosen_selector,
        &mut uniform_grouper,
        &mut transformer,
        &uga,
    );
}

/// `-dcc-register-initialization-disable`, `cl::init(false)` (`:77-80`) — a `dcc-opt` command-line
/// flag, not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `opts_.OptLevel == 0` (`:266`) — ⛔ NOT A `dcc-opt` FLAG BUT A BUILD OPTION:
/// `CommonPassOptions::OptLevel` defaults to `-1` (`dcc/tools/Options/dcc-pass-option.h:117`), so an
/// ordinary build never asks the estimator and initialises every unit.
const OPT_LEVEL_ZERO: bool = false;

/// Replaces: e521_runOnOperation
///
/// The pass entry: register-initialises every program unit, unless the flag turned the pass off or
/// `-O0` found the unit already fits its instruction buffer.
///
/// ⭐ `markAnalysesPreserved<Liveness>()` HAS NO EXPRESSION HERE — a `Liveness` this crate hands in as
/// a seam is owned by the caller, so there is no pass-manager cache to keep or invalidate.
/// ⚠️ TRAP: THE ESTIMATOR IS ONLY ASKED AT `-O0`, and it is out of campaign scope — the `&&` short
/// circuit is what keeps [`InstructionEstimator::have_ibuff_space`]'s `todo!` out of a default build.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    liveness: &mut dyn Liveness,
    instruction_estimator: &mut impl InstructionEstimator,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    for unit in program.units.iter_mut() {
        if OPT_LEVEL_ZERO && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        run_on(unit, liveness);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::ProgramUnits;
    use crate::islands::sentient::dialects::Val;
    use crate::transform::sentient::analyses::OutOfScopeInstructionEstimator;
    use crate::transform::sentient::analyses::OutOfScopeLiveness;
    use crate::units::DfirUnit;
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
        Group(Vec<Candidate>),
        EvaluateGlobally(Vec<Candidate>),
        SelectGlobally(Vec<Candidate>),
        Transform(Vec<Candidate>),
    }

    /// The one log the four seams write to — the reference's four references are four distinct
    /// objects, so each fake is its own value sharing this.
    type Log = Rc<RefCell<Vec<Call>>>;

    struct Collector(Log);
    struct Evaluator(Log);
    struct Selector(Log);
    struct Groups(Vec<Val>);
    struct Grouper(Log);
    struct Transform(Log);

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

        fn evaluate_globally(&mut self, candidates: &mut Vec<Candidate>) {
            self.0
                .borrow_mut()
                .push(Call::EvaluateGlobally(candidates.clone()));
        }

        fn merge_into(&mut self, result: &mut Vec<Candidate>, sublist: &[Candidate]) {
            self.0.borrow_mut().push(Call::MergeInto(sublist.to_vec()));
            result.extend_from_slice(sublist);
        }
    }

    impl UniformGrouper for Grouper {
        fn run(&mut self, candidates: &mut Vec<Candidate>) {
            self.0.borrow_mut().push(Call::Group(candidates.clone()));
            // The one phase that may grow the list: a grouped global candidate goes in.
            candidates.push(Candidate(900));
        }
    }

    impl Transformer for Transform {
        fn run(&mut self, candidates: &[Candidate]) {
            self.0
                .borrow_mut()
                .push(Call::Transform(candidates.to_vec()));
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

        fn select_globally(&mut self, candidates: &mut Vec<Candidate>) {
            self.0
                .borrow_mut()
                .push(Call::SelectGlobally(candidates.clone()));
        }
    }

    impl UniformGroups for Groups {
        fn group_leaders(&self) -> Vec<Val> {
            self.0.clone()
        }

        fn is_group_leader(&self, unit: Val) -> bool {
            self.0.contains(&unit)
        }

        fn group_members_led_by(&self, _leader: Val) -> Vec<Val> {
            Vec::new()
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

    /// e345_run — the three phases in order over ONE list, and what phase 2 added is what phase 3 sees.
    #[test]
    fn e345_run() {
        let log: Log = Rc::new(RefCell::new(Vec::new()));
        let mut collector = Collector(Rc::clone(&log));
        let mut evaluator = Evaluator(Rc::clone(&log));
        let mut selector = Selector(Rc::clone(&log));
        let mut grouper = Grouper(Rc::clone(&log));
        let mut transformer = Transform(Rc::clone(&log));
        let groups = Groups(vec![Val(1)]);

        run(
            &mut collector,
            &mut evaluator,
            &mut selector,
            &mut grouper,
            &mut transformer,
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
                Call::MergeInto(Vec::new()),
                // Phase 2 over what phase 1 merged, and phase 3 over what phase 2 left.
                Call::Group(vec![Candidate(1)]),
                Call::EvaluateGlobally(vec![Candidate(1), Candidate(900)]),
                Call::SelectGlobally(vec![Candidate(1), Candidate(900)]),
                Call::Transform(vec![Candidate(1), Candidate(900)]),
            ]
        );
    }

    /// e346_runGlobalAnalysis — the grouper first, so the candidate it adds is judged by the other two.
    #[test]
    fn e346_run_global_analysis() {
        let log: Log = Rc::new(RefCell::new(Vec::new()));
        let mut evaluator = Evaluator(Rc::clone(&log));
        let mut selector = Selector(Rc::clone(&log));
        let mut grouper = Grouper(Rc::clone(&log));
        let mut candidates = vec![Candidate(5)];

        run_global_analysis(&mut candidates, &mut grouper, &mut evaluator, &mut selector);

        assert_eq!(
            *log.borrow(),
            vec![
                Call::Group(vec![Candidate(5)]),
                Call::EvaluateGlobally(vec![Candidate(5), Candidate(900)]),
                Call::SelectGlobally(vec![Candidate(5), Candidate(900)]),
            ]
        );
        assert_eq!(candidates, vec![Candidate(5), Candidate(900)]);
    }

    /// e458_runOn — the analysis it runs FIRST is out of campaign scope, so a unit gets no further:
    /// the leaders it collects are what every phase after it iterates.
    #[test]
    #[should_panic(expected = "UniformGroupAnalyzer::collectExclusiveGroupLeaders")]
    fn e458_run_on_starts_at_the_uniform_group_analysis() {
        let mut unit: ProgramUnit<Dd2> = ProgramUnit {
            on: Units::one(DfirUnit::Pe, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        };
        let mut liveness = OutOfScopeLiveness;
        run_on(&mut unit, &mut liveness);
    }

    /// e347_runTransformation — the whole list reaches the transformer, and only the transformer.
    #[test]
    fn e347_run_transformation() {
        let log: Log = Rc::new(RefCell::new(Vec::new()));
        let mut transformer = Transform(Rc::clone(&log));
        run_transformation(&[Candidate(7), Candidate(8)], &mut transformer);
        assert_eq!(
            *log.borrow(),
            vec![Call::Transform(vec![Candidate(7), Candidate(8)])]
        );
    }
    /// A model and a rung, so the program is typed; nothing here reads either.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// e521_runOnOperation — the entry reaches [`run_on`] for the unit, and ⚠️ THE ESTIMATOR IS NOT
    /// ASKED: at any optimisation level but `-O0` the `&&` short circuits before its `todo!`, so the
    /// panic that arrives names the uniform-group analysis and not `haveIbuffSpace`.
    #[test]
    #[should_panic(expected = "UniformGroupAnalyzer::collectExclusiveGroupLeaders")]
    fn e521_run_on_operation_reaches_the_driver_without_asking_the_estimator() {
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Pe, Val(0)),
                    precision: None,
                    body: Vec::new(),
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };
        let mut liveness = OutOfScopeLiveness;
        let mut estimator = OutOfScopeInstructionEstimator;
        run_on_operation(&mut program, &mut liveness, &mut estimator);
    }
}
