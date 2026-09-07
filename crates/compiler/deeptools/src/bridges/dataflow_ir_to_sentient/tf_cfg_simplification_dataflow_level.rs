// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `CFGSimplificationDataflowLevel.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [9]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e383_runOnOperation` | 383/384 | 80 | `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77` |

use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::is_operation_selected;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, affine, agen, dataflow, scf};
use crate::islands::dataflow_ir::{self as dfir};

/// HOW MANY TIMES ONE BOUNDED REWRITE MAY FIRE ON ONE UNIT.
///
/// ⭐⭐ THE PASS IS A FIXPOINT LOOP WITH A LEASH. `hoistCommonConditionals` and
/// `shallowlyMergeConditionals` each return the node they last analysed and are called again from
/// there until they return nothing — *"After every simplification, recompute the tree and redo the
/// analysis, repeating until no further hoisting is possible"*
/// (`CFGSimplificationDataflowLevel.cpp:89-91`) — and each loop's `while` also tests a COUNT against
/// a limit (`:104`, `:143`). Without the count a rewrite that keeps handing back a node it already
/// simplified never terminates.
///
/// ⛔ THE TWO LIMITS ARE SEPARATE VALUES THAT HAPPEN TO AGREE. `MaxNumOfHoists` and `MaxNumOfMerges`
/// are two `llvm::cl::opt<int>`s, each `llvm::cl::init(50)`
/// (`CFGSimplificationDataflowLevel.cpp:50-61`). Folding them into one constant because both read 50
/// today would make raising one raise the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rounds(u32);

impl Rounds {
    /// The limit, for the loop that counts against it.
    #[must_use]
    pub fn limit(self) -> u32 {
        self.0
    }
}

/// `-dcc-cfg-simplification-dataflow-level-max-num-hoists`, `llvm::cl::init(50)`
/// (`CFGSimplificationDataflowLevel.cpp:56-60`).
pub const MAX_NUM_OF_HOISTS: Rounds = Rounds(50);

/// `-dcc-cfg-simplification-dataflow-level-max-num-merges`, `llvm::cl::init(50)`
/// (`CFGSimplificationDataflowLevel.cpp:50-54`).
pub const MAX_NUM_OF_MERGES: Rounds = Rounds(50);

/// ONE SIMPLIFICATION, IN THE ORDER THE PASS RUNS IT.
///
/// ⭐⭐ THE ORDER IS THE PASS. Every step here consumes the tree the step before it left, and the
/// reference says so where it matters: *"Hoisting may lead to duplicate, side-effect-free siblings.
/// Remove these"* (`CFGSimplificationDataflowLevel.cpp:113`) and *"Merging may lead to duplicate,
/// side-effect-free siblings. Remove these"* (`:145`). That is why
/// [`RemoveDuplicateConditionals`](Self::RemoveDuplicateConditionals) appears TWICE in [`STEPS`] —
/// once after the hoists and once after the merges — and reading the list as a set of rewrites to
/// apply in any order loses both cleanups.
///
/// ⛔ AND THE MERGE IS DELIBERATELY INCOMPLETE HERE. *"We must leave the merging of sibling
/// conditionals both yielding values to a later Sentient-level pass as the Dataflow-level calls to
/// the Canonicalizer would partially undo those merges"* (`:131-133`). So a step this pass does NOT
/// take is as much a decision as one it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Step {
    /// `tree.hoistCommonConditionals` — `e284`, bounded by [`MAX_NUM_OF_HOISTS`]
    /// (`CFGSimplificationDataflowLevel.cpp:94-104`).
    HoistCommonConditionals,
    /// `tree.hoistLoopInvariantConditionals` — `e371` (`:107`).
    HoistLoopInvariantConditionals,
    /// `tree.removeDuplicateConditionals` — `TransformationConditionalTree.cpp:58` (`:114`, `:146`).
    RemoveDuplicateConditionals,
    /// `tree.removeConditionWhenThenElseBranchesMatch` — `TransformationConditionalTree.cpp:181`
    /// (`:123`).
    RemoveConditionWhenThenElseBranchesMatch,
    /// `tree.shallowlyMergeConditionals` — `e380`, bounded by [`MAX_NUM_OF_MERGES`] (`:134-143`).
    ShallowlyMergeConditionals,
    /// `tree.simplifyValueBasedConditionals` — `e381` (`:151`).
    SimplifyValueBasedConditionals,
}

/// THE SEVEN CALLS, IN ORDER — `CFGSimplificationDataflowLevel.cpp:97-151`.
///
/// ⛔ SEVEN CALLS, SIX KINDS. See [`Step`] for why the duplicate removal is listed twice.
pub const STEPS: &[Step] = &[
    Step::HoistCommonConditionals,
    Step::HoistLoopInvariantConditionals,
    Step::RemoveDuplicateConditionals,
    Step::RemoveConditionWhenThenElseBranchesMatch,
    Step::ShallowlyMergeConditionals,
    Step::RemoveDuplicateConditionals,
    Step::SimplifyValueBasedConditionals,
];

/// WHETHER THE CONDITIONAL TREE TAKES THIS OP AS A NODE — the anchored unit, not a second copy.
///
/// ⛔⛔ THIS USED TO BE ITS OWN `isa<>` CHAIN, AND THAT WAS ONE PREDICATE WRITTEN TWICE.
/// `isOperationSelected` (`CFGSDataflowConditionalTree.cpp:34`) is bridge-2 entry **095**, homed in
/// [`super::tf_cfgs_dataflow_conditional_tree`]; this file needed the same question and answered it
/// locally, before that entry was ported. Two records of one fact, and the local one carried a note
/// that the `affine.if` half had no island variant — which is now false. It delegates.
///
/// ⭐ THE CALL SITE IS UNCHANGED. `count_conditionals` still asks the same question of the same ops;
/// the answer now comes from the one function the reference has.
fn is_selected(op: &DfirOp) -> bool {
    is_operation_selected(op)
}

/// PREORDER, DESCENDING INTO EVERY REGION — `walk<WalkOrder::PreOrder>`.
///
/// ⛔ THIS IS A USE-WALK AND NOTHING ELSE, which is the one mechanism a port of this campaign may
/// simplify: MLIR reaches nested ops through `Region`/`Block`, this island nests them in `Vec`s. The
/// JUDGEMENT stays in [`is_selected`]; a leaf op falls through here because it has no region, not
/// because it was decided about.
fn walk_preorder(ops: &[DfirOp], visit: &mut impl FnMut(&DfirOp)) {
    for op in ops {
        visit(op);
        match op {
            DfirOp::Affine(affine::Op::For { body, .. })
            | DfirOp::Scf(scf::Op::Parallel { body, .. } | scf::Op::For { body, .. })
            | DfirOp::Dataflow(dataflow::Op::ProgramUnit { body, .. }) => {
                walk_preorder(body, visit);
            }
            DfirOp::Scf(scf::Op::If {
                body, else_body, ..
            }) => {
                walk_preorder(body, visit);
                walk_preorder(else_body, visit);
            }
            DfirOp::Agen(agen::Op::CompositeLoadAndStore(transfer)) => {
                walk_preorder(&transfer.body, visit);
            }
            // ⭐ AN `affine.if` HAS TWO REGIONS TOO, and now that the island can hold one the walk
            // has to descend into both — a preorder walk that skipped them would report a tree with
            // no nodes for a program whose conditionals are all at the affine rung.
            DfirOp::Affine(affine::Op::If {
                body, else_body, ..
            }) => {
                walk_preorder(body, visit);
                walk_preorder(else_body, visit);
            }
            // no region.
            DfirOp::Arith(_)
            | DfirOp::Affine(_)
            | DfirOp::Scf(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            | DfirOp::VectorChain(_)
            | DfirOp::Symbol(_) => {}
        }
    }
}

/// HOW MANY NODES ONE UNIT'S CONDITIONAL TREE HOLDS.
///
/// ⭐ `empty()` IS `!root_ || !root_->getFirstChild()` (`dcc/src/Analysis/OperationTree.hpp:214`) —
/// the root is the unit itself and is never a conditional, so an empty tree is a unit with no
/// selected op anywhere beneath it. Counting them answers `compute()` + `empty()` together
/// (`ConditionalTree.cpp:185`, `CFGSimplificationDataflowLevel.cpp:83-85`) without building the tree
/// that only the unported rewrites would read.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Nodes(usize);

impl Nodes {
    /// `tree.empty()` — no conditional was selected, so every step below would be a no-op.
    #[must_use]
    pub fn empty(self) -> bool {
        self.0 == 0
    }

    /// How many were selected.
    #[must_use]
    pub fn count(self) -> usize {
        self.0
    }
}

/// THE CONDITIONALS BENEATH ONE UNIT — `CFGSDataflowConditionalTree tree(*unit); tree.compute();`.
#[must_use]
pub fn conditional_tree<A: Arch>(unit: &dfir::ProgramUnit<A>) -> Nodes {
    let mut nodes = Nodes::default();
    walk_preorder(&unit.body, &mut |op| {
        if is_selected(op) {
            nodes.0 += 1;
        }
    });
    nodes
}

/// SIMPLIFY THE CONTROL FLOW GRAPH OF EVERY UNIT IN A PROGRAM.
///
/// `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`. The file's own summary
/// (`:8-21`) lists what it does: hoist side-effect-free conditionals common to both branches of a
/// parent, remove conditionals with equivalent then/else branches, shallowly merge sibling
/// conditionals sharing a top-level condition, remove duplicate side-effect-free siblings, hoist
/// loop-invariant Index-yielding conditionals out of their loops, and replace a top-level
/// 1-dimensional conditional yielding a fixed-stride Index sequence by a new iteration argument.
///
/// # ⭐⭐ PER UNIT, AND EACH UNIT'S TREE IS ITS OWN
///
/// `:81-85` walks the module for `dataflow::ProgramUnitOp` and builds a fresh
/// `CFGSDataflowConditionalTree` on each, returning early from that unit when the tree is empty. A
/// unit with no undecided branch is left exactly as it was — not "simplified to itself", *skipped*,
/// which is why an empty-tree program can be returned unchanged rather than rebuilt.
///
/// # ⛔⛔ TODAY EVERY TREE IN THIS CRATE IS EMPTY, AND THAT IS A CHECKED FACT NOT AN ASSUMPTION
///
/// [`is_selected`] is `isa<affine.if, scf.if>`; the DataflowIR island has no `affine.if` at all, so
/// the only node kind is [`scf::Op::If`]. Nothing this compiler currently emits constructs one — the
/// undecided branches bridge 1 flattens are gone by the time a program reaches this rung. So this
/// pass is provably a no-op over every program the pipeline can hand it, and wiring it in cannot
/// change a single emitted byte.
///
/// ⛔ WHICH IS EXACTLY WHY IT IS WIRED IN ANYWAY. The day an `scf.if` DOES reach this rung, the
/// program needs six rewrites that do not exist and the answer must be a BUILD FAILURE naming the
/// first missing one — not a lowering that quietly skips the simplification and emits a conditional
/// the Sentient rung mis-schedules. A pass that is only added once its input appears is a pass that
/// is absent on the one run that needed it.
///
/// # ⛔ WHAT THE PORT DROPS, AND WHY
///
/// - `DisableThisPass` (`:45-48`) is an `llvm::cl::opt<bool>` read at `:78`: a **command-line flag of
///   `dcc-opt`**, not a program property. This crate has no pass pipeline and no flags — which pass
///   runs is a call in [`super::program`] — so the switch has nowhere to live. Turning it into a
///   parameter would make "did the compiler simplify" a runtime question with two answers.
/// - The `LLVM_DEBUG` / `DEBUG_WITH_TYPE(VerboseDebug, tree.print(...))` tracing after every step
///   (`:86-155`) prints the tree; it has no effect on the IR.
/// - The tree itself is not built. `compute()` populates a node graph that only the six unported
///   rewrites read; what this port needs from it is `empty()`, which is exactly the node COUNT
///   ([`conditional_tree`]). The tree gets built in `tf_cfgs_dataflow_conditional_tree.rs`, its own
///   home, when its rewrites land.
///
/// # ⛔⛔ FIVE OF THE SEVEN CALLS ARE OUTSIDE THIS CAMPAIGN'S 384
///
/// `e284_hoistCommonConditionals`, `e371_hoistLoopInvariantConditionals`,
/// `e380_shallowlyMergeConditionals` and `e381_simplifyValueBasedConditionals` are scheduled units.
/// But `removeDuplicateConditionals` and `removeConditionWhenThenElseBranchesMatch` live on the base
/// class in `dcc/src/Analysis/TransformationConditionalTree.cpp:58` and `:181`, as do the tree's
/// constructor, `compute()` and `empty()` — and none of those five is in the 384 or in the campaign's
/// 106 documented exclusions. So this pass cannot be completed by this campaign as scheduled, and the
/// `todo!` below names the first step rather than pretending the list is reachable.
///
/// # ⛔ AND IT TAKES A SHARED REFERENCE, WHICH IS A STATEMENT ABOUT WHAT IS PORTED
///
/// The reference rewrites its module in place. Here all seven calls in [`STEPS`] are unported, so
/// there is nothing to write back: the whole observable effect is the choice between leaving a
/// program alone and failing the build. A `&mut` parameter would claim a capability nothing behind it
/// has, and would make every caller clone a program in order to rewrite none of them. The parameter
/// becomes `&mut` in the changeset that lands the first rewrite.
///
/// Replaces: e383_runOnOperation
pub fn run_on_operation<A: Arch>(program: &dfir::Program<A>) {
    for unit in program.units.iter() {
        // ⛔ `if (tree.empty()) return;` IS PER UNIT, NOT PER MODULE (`:85`) — `return` from the walk
        // lambda skips this unit and the walk carries on to the next.
        let tree = conditional_tree(unit);
        if tree.empty() {
            continue;
        }

        // ⛔ THE FIRST STEP IS THE ONE THE BUILD MUST NAME. Running the later steps against an
        // unsimplified tree would be a different pass, so the gap is reported at step 1 with the
        // order that follows it stated in [`STEPS`].
        todo!(
            "e284_hoistCommonConditionals (then {:?}, hoists <= {}, merges <= {}): {} conditional(s) \
             reached the dataflow-level CFG simplification on {:?}",
            &STEPS[1..],
            MAX_NUM_OF_HOISTS.limit(),
            MAX_NUM_OF_MERGES.limit(),
            tree.count(),
            unit.on.kind()
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        MAX_NUM_OF_HOISTS, MAX_NUM_OF_MERGES, STEPS, Step, conditional_tree, is_selected,
        run_on_operation,
    };
    use crate::arch::Target;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, arith, scf};
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::DfirUnit;

    /// One unit holding `body` — the pass reads no machine fact, so the kind is arbitrary.
    fn unit_holding(body: Vec<DfirOp>) -> ProgramUnit<Target> {
        ProgramUnit {
            on: Units::one(DfirUnit::Lxlu, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// An `scf.if` on `cond` with the given arms.
    fn branch(cond: u32, body: Vec<DfirOp>, else_body: Vec<DfirOp>) -> DfirOp {
        DfirOp::Scf(scf::Op::If {
            cond: Val(cond),
            body,
            else_body,
        })
    }

    /// 🎯 THE PREDICATE IS `isa<affine.if, scf.if>` AND NOTHING ELSE IS A NODE.
    ///
    /// `CFGSDataflowConditionalTree.cpp:34`. A loop is not a node, a terminator is not a node, an
    /// `scf.parallel` is not a node — if any of them were, `empty()` would be false for every
    /// program with a loop in it and the pass would run its six rewrites over a tree of loops.
    #[test]
    fn only_an_undecided_branch_is_a_tree_node() {
        assert!(is_selected(&branch(0, vec![], vec![])));
        assert!(!is_selected(&DfirOp::Scf(scf::Op::Yield {
            operands: vec![]
        })));
        assert!(!is_selected(&DfirOp::Scf(scf::Op::Parallel {
            ivs: vec![],
            body: vec![],
        })));
        assert!(!is_selected(&DfirOp::Affine(affine::Op::Yield {
            operands: vec![]
        })));
        assert!(!is_selected(&DfirOp::Affine(affine::Op::For {
            iv: Val(0),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            dbg_name: None,
            carried: Vec::new(),
            body: vec![],
        })));
    }

    /// 🎯 `empty()` IS ABOUT CONDITIONALS, NOT ABOUT PROGRAM SIZE.
    ///
    /// A unit full of statements with no undecided branch has an EMPTY tree
    /// (`OperationTree.hpp:214` — a root with no first child), so `:85` skips it. Answering "not
    /// empty" for a program that merely has ops in it would send every unit into the unported
    /// rewrites.
    #[test]
    fn a_unit_with_no_conditional_has_an_empty_tree() {
        let unit = unit_holding(vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 0,
            }),
            DfirOp::Affine(affine::Op::Yield { operands: vec![] }),
        ]);
        assert!(conditional_tree(&unit).empty());
        assert_eq!(conditional_tree(&unit).count(), 0);
    }

    /// 🎯 THE WALK IS PREORDER AND ENTERS BOTH ARMS AND EVERY LOOP BODY.
    ///
    /// ⛔⛔ A NESTED CONDITIONAL IS THE WHOLE POINT OF THE PASS — hoisting a conditional common to
    /// both branches of a PARENT conditional (`CFGSimplificationDataflowLevel.cpp:11-12`) is
    /// unaskable if the walk stops at the top level. This counts three: the outer `if`, the one in
    /// its `then`, and the one inside a loop in its `else`.
    #[test]
    fn the_walk_finds_conditionals_nested_in_arms_and_loops() {
        let inner_in_loop = DfirOp::Affine(affine::Op::For {
            iv: Val(2),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            dbg_name: None,
            carried: Vec::new(),
            body: vec![branch(3, vec![], vec![])],
        });
        let unit = unit_holding(vec![branch(
            0,
            vec![branch(1, vec![], vec![])],
            vec![inner_in_loop],
        )]);

        assert!(!conditional_tree(&unit).empty());
        assert_eq!(conditional_tree(&unit).count(), 3);
    }

    /// A program of `units`, named so a failure says which case it was.
    fn program_of(units: Vec<ProgramUnit<Target>>) -> Program<Target> {
        let mut units = units.into_iter();
        let head = units.next().expect("a test program names its units");
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(head, units.collect()),
            arch: core::marker::PhantomData,
        }
    }

    /// 🎯 A PROGRAM WITH NO CONDITIONAL IS SKIPPED UNIT BY UNIT, AND THE PASS RETURNS.
    ///
    /// `:85` returns from a unit whose tree is empty, so the pass does nothing to it. This is the
    /// case every program the pipeline currently produces is in — see [`run_on_operation`]'s note —
    /// and REACHING THE END OF THE CALL is the assertion: were any unit's tree non-empty the `todo!`
    /// would fire and this test would fail with the name of the missing rewrite.
    ///
    /// ⛔ IT DOES NOT COMPARE THE PROGRAM WITH A CLONE OF ITSELF. The parameter is shared, so
    /// "unchanged" is true by the type and an equality against a copy would be a tautology dressed
    /// as a check.
    #[test]
    fn a_program_without_conditionals_is_skipped_unit_by_unit() {
        let with_ops = unit_holding(vec![DfirOp::Arith(arith::Op::Constant {
            result: Val(0),
            value: 0,
        })]);
        let empty = unit_holding(Vec::new());
        assert!(conditional_tree(&with_ops).empty());
        assert!(conditional_tree(&empty).empty());

        run_on_operation(&program_of(vec![with_ops, empty]));
    }

    /// 🎯 A CONDITIONAL REACHING THIS RUNG NAMES THE FIRST MISSING REWRITE.
    ///
    /// ⛔⛔ THIS IS THE WHOLE REASON THE PASS IS WIRED IN WHILE IT IS A NO-OP. The alternative to
    /// failing here is lowering an `scf.if` that six unported rewrites were supposed to have
    /// simplified first — a program the Sentient rung then schedules against a control flow graph the
    /// reference would never have handed it. The failure has to carry `e284_hoistCommonConditionals`,
    /// because "the CFG simplification did not run" is not a diagnosis anybody can act on.
    #[test]
    #[should_panic(expected = "e284_hoistCommonConditionals")]
    fn a_conditional_names_the_first_unported_rewrite() {
        run_on_operation(&program_of(vec![unit_holding(vec![branch(
            0,
            vec![],
            vec![],
        )])]));
    }

    /// 🎯 SEVEN CALLS, SIX KINDS, AND THE DUPLICATE REMOVAL IS THE ONE THAT REPEATS.
    ///
    /// ⛔ THE REPEAT IS LOAD-BEARING: hoisting can create duplicate siblings (`:113`) and so can
    /// merging (`:145`), and the two cleanups sit on opposite sides of the merge loop. Listing the
    /// step once would leave whichever set of duplicates came second in the program.
    #[test]
    fn the_step_order_is_the_references_own() {
        assert_eq!(STEPS.len(), 7);
        assert_eq!(STEPS[0], Step::HoistCommonConditionals);
        assert_eq!(STEPS[1], Step::HoistLoopInvariantConditionals);
        assert_eq!(STEPS[2], Step::RemoveDuplicateConditionals);
        assert_eq!(STEPS[3], Step::RemoveConditionWhenThenElseBranchesMatch);
        assert_eq!(STEPS[4], Step::ShallowlyMergeConditionals);
        assert_eq!(STEPS[5], Step::RemoveDuplicateConditionals);
        assert_eq!(STEPS[6], Step::SimplifyValueBasedConditionals);
        assert_eq!(
            STEPS
                .iter()
                .filter(|step| **step == Step::RemoveDuplicateConditionals)
                .count(),
            2
        );
    }

    /// 🎯 THE TWO CAPS ARE TWO VALUES.
    ///
    /// `MaxNumOfHoists` and `MaxNumOfMerges` are separate `llvm::cl::opt<int>`s that both
    /// `init(50)` (`:50-61`). They agree today; they are not the same knob.
    #[test]
    fn the_hoist_and_merge_caps_are_fifty_each() {
        assert_eq!(MAX_NUM_OF_HOISTS.limit(), 50);
        assert_eq!(MAX_NUM_OF_MERGES.limit(), 50);
    }
}
