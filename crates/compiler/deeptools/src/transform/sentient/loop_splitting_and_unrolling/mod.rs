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

//! `LoopSplittingAndUnrolling.cpp` — 19 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e085_replaceIterArgsAndYieldResults` | 085 | 0 | 13 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:132` |
//! | `e086_isOkToUnroll` | 086 | 0 | 47 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:159` |
//! | `e087_cleanupAndRecalculate` | 087 | 0 | 37 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:295` |
//! | `e088_ifOpUsesIV` | 088 | 0 | 10 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:455` |
//! | `e089_hasIfOps` | 089 | 0 | 12 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:466` |
//! | `e090_evaluatePredicate` | 090 | 0 | 33 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:552` |
//! | `e091_findAllSplitVals` | 091 | 0 | 22 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:596` |
//! | `e092_computeReducedCost` | 092 | 0 | 39 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:649` |
//! | `e318_promoteForLoopBodyAndDelete` | 318 | 1 | 10 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:147` |
//! | `e319_getAllTrueIndexOfIfOpsList` | 319 | 1 | 7 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:587` |
//! | `e320_subsetsImpl` | 320 | 1 | 9 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:622` |
//! | `e446_subsets` | 446 | 2 | 12 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:635` |
//! | `e447_splitLoopNWay` | 447 | 2 | 129 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:858` |
//! | `e513_unrollLoop` | 513 | 3 | 86 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:207` |
//! | `e514_findOptimalNWaySplits` | 514 | 3 | 162 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:691` |
//! | `e566_getSavedCycleAndIbuffCost` | 566 | 4 | 117 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:335` |
//! | `e567_doSplitOrUnroll` | 567 | 4 | 23 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:993` |
//! | `e603_findBestLoopNodeToOptimize` | 603 | 5 | 69 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:479` |
//! | `e629_runOnOperation` | 629 | 6 | 61 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:1019` |

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e629_runOnOperation` (level 6) is what calls this
// file's driver, and every unit below is reachable only from the tests until it lands. CI runs clippy
// with `-D warnings`. ⭐ REMOVE THIS WITH e629.

use std::collections::{BTreeMap, VecDeque};

use super::ForRef;
use super::analyses::{InstructionCount, InstructionEstimator};
use super::cfg_simplification_sentient_level::pattern_simplification_manager::OpPath;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{
    Op, Val, clone_ops, defining_op, replace_all_uses_with, use_count,
};

/// THE PASS'S OWN STATE (`LoopSplittingAndUnrolling.cpp:125-129`) — the candidate list and its ibuff
/// costs, which is everything [`cleanup_and_recalculate`] touches.
///
/// ⭐ `can_not_split_`/`can_not_unroll_` land with the units that read them (e566, e603).
#[derive(Debug, Clone, Default)]
pub(crate) struct LoopSplittingAndUnrolling {
    /// `unrolling_or_splitting_candidates`, in the order the reverse-BFS walk pushed them.
    pub(crate) unrolling_or_splitting_candidates: Vec<LoopNodeId>,
    /// `loopnode_to_ibuff`.
    pub(crate) loopnode_to_ibuff: BTreeMap<LoopNodeId, InstructionCount>,
}

/// `dyn_cast<sentient::ForOp>(op)` AS A TYPE — the four things this pass reads off a loop.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ForOp<'a> {
    /// `getInductionVar()`.
    pub(crate) iv: Val,
    /// `getBound()`.
    pub(crate) bound: Val,
    /// `getIterOperands()` zipped with `getRegionIterArgs()` — one entry each.
    pub(crate) carried: &'a [ops::Carried],
    /// `getBody()`.
    pub(crate) body: &'a [Op],
}

impl<'a> ForOp<'a> {
    /// The loop, or `None` for any other op.
    #[must_use]
    pub(crate) fn of(op: &'a Op) -> Option<ForOp<'a>> {
        let Op::Sentient(ops::Op::For {
            iv,
            bound,
            carried,
            body,
            ..
        }) = op
        else {
            return None;
        };
        Some(ForOp {
            iv: *iv,
            bound: *bound,
            carried,
            body,
        })
    }
}

/// ONE ITERATION BOUND OF A LOOP — the reference's `int bound`, walked from the loop's constant bound
/// DOWN TO 1 (`LoopSplittingAndUnrolling.cpp:697-707`).
///
/// ⛔ TRAP: bound `b` is the induction variable value `b - 1` — `splitLoopNWay` gives a split the
/// bound `kv.first - kv.second + 1` and shifts its IV by `kv.second - 1` (`:858-986`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct IterBound(pub(crate) i64);

/// ONE SPLIT — `std::pair<int, int>`: the highest and the lowest iteration bound that behave alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitRange {
    /// `.first` — the highest bound in the group.
    pub(crate) high: IterBound,
    /// `.second` — the lowest.
    pub(crate) low: IterBound,
}

/// `std::map<int, std::vector<bool>, std::greater<>> &bound_to_all_ifops_predval` — every iteration
/// bound with each if-op's predicate value at it, HIGHEST BOUND FIRST.
///
/// ⛔⛔ NON-EMPTY BY CONSTRUCTION, AND THAT IS A FIX: `findAllSplitVals` dereferences `begin()`
/// before ever comparing it with `end()` (`:601-604`), so an empty map is undefined behaviour there.
/// Its one caller fills the map from `loop_bound` down to 1 (`:697-707`), which is what this states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundPredVals {
    /// The highest bound and its predicate values.
    head: (IterBound, Vec<bool>),
    /// The rest, descending.
    tail: Vec<(IterBound, Vec<bool>)>,
}

impl BoundPredVals {
    /// The map read highest key first — `std::greater<>` — and `None` when it holds nothing.
    #[must_use]
    pub(crate) fn from_map(bounds: &BTreeMap<IterBound, Vec<bool>>) -> Option<BoundPredVals> {
        let mut descending = bounds
            .iter()
            .rev()
            .map(|(bound, predvals)| (*bound, predvals.clone()));
        let head = descending.next()?;
        Some(BoundPredVals {
            head,
            tail: descending.collect(),
        })
    }

    /// The entries, highest bound first.
    fn entries(&self) -> Vec<&(IterBound, Vec<bool>)> {
        core::iter::once(&self.head)
            .chain(self.tail.iter())
            .collect()
    }
}

/// WHICH NODE OF A [`LoopForest`] — the reference's `dcc::LoopNode *`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LoopNodeId(pub(crate) u32);

/// ONE `dcc::LoopNode` — the loop it names, and the loops nested in that one.
#[derive(Debug, Clone, Default)]
pub(crate) struct LoopNode {
    /// `getOperation()`; `None` is the synthetic root, the only node without one
    /// (`src/Analysis/LoopTree.hpp:151`).
    pub(crate) op: Option<ForRef>,
    /// The children, in program order.
    pub(crate) children: Vec<LoopNodeId>,
}

/// `dcc::LoopTree<sentient::ForOp, dataflow::ProgramUnitOp>` — one node per `sentient.for`, parented
/// by the innermost `sentient.for` enclosing it (`src/Analysis/LoopTree.hpp:149-168`).
///
/// ⭐ SUPPLIED, NOT INVENTED: `compute` is a pre-order structural walk with no cost model and no
/// analysis in it, i.e. the mechanism for reaching operands that this campaign permits supplying.
/// ⛔ AN `affine.for` IS NOT A NODE — `LoopOp` is `sentient::ForOp` — but the walk descends into one.
#[derive(Debug, Clone)]
pub(crate) struct LoopForest {
    /// Node 0 is the root.
    nodes: Vec<LoopNode>,
}

impl Default for LoopForest {
    fn default() -> LoopForest {
        LoopForest {
            nodes: vec![LoopNode::default()],
        }
    }
}

impl LoopForest {
    /// `compute(unit)`.
    #[must_use]
    pub(crate) fn of(unit: &[Op]) -> LoopForest {
        let mut forest = LoopForest::default();
        forest.compute(unit, LoopNodeId(0));
        forest
    }

    /// `recompute(unit)` — `clear(); compute(unit);` (`src/Analysis/LoopTree.hpp:105-108`).
    pub(crate) fn recompute(&mut self, unit: &[Op]) {
        *self = LoopForest::of(unit);
    }

    /// `getRoot()`.
    #[must_use]
    pub(crate) const fn root(&self) -> LoopNodeId {
        LoopNodeId(0)
    }

    /// One node.
    #[must_use]
    pub(crate) fn node(&self, id: LoopNodeId) -> Option<&LoopNode> {
        self.nodes.get(id.0 as usize)
    }

    /// `walk(action, kReverseBFS)` — BFS from the root onto a stack, then popped, so the deepest and
    /// rightmost node is visited first and the root last (`src/Analysis/OperationTree.cpp:200-237`).
    #[must_use]
    pub(crate) fn reverse_bfs(&self) -> Vec<LoopNodeId> {
        let mut queue = VecDeque::from([self.root()]);
        let mut visited = Vec::new();
        while let Some(id) = queue.pop_front() {
            visited.push(id);
            if let Some(node) = self.node(id) {
                queue.extend(node.children.iter().copied());
            }
        }
        visited.reverse();
        visited
    }

    /// A new node under `parent`.
    fn push(&mut self, op: Option<ForRef>, parent: LoopNodeId) -> LoopNodeId {
        let id = LoopNodeId(self.nodes.len() as u32);
        self.nodes.push(LoopNode {
            op,
            children: Vec::new(),
        });
        if let Some(node) = self.nodes.get_mut(parent.0 as usize) {
            node.children.push(id);
        }
        id
    }

    /// The pre-order walk: `parent` is the innermost enclosing loop's node, which is what
    /// `getParentOfType<LoopOp>()` finds — a non-loop op between two loops changes nothing.
    fn compute(&mut self, body: &[Op], parent: LoopNodeId) {
        for op in body {
            match op {
                Op::Sentient(inner) => {
                    let here = match inner {
                        ops::Op::For { iv, .. } => self.push(Some(ForRef(*iv)), parent),
                        _ => parent,
                    };
                    for region in ops::regions(inner) {
                        self.compute(region, here);
                    }
                }
                Op::AffineFor(loop_op) => self.compute(&loop_op.body, parent),
                // ⭐ A LOWER-RUNG REGION CANNOT HOLD A `sentient.for`: it is typed with the rung
                // below's `Op`, which has no `Sentient` arm.
                _ => {}
            }
        }
    }
}

/// THE `sentient.if` UNDER AN IF-[`CondNode`] — the two regions, which is all e092 costs.
///
/// ⛔⛔ THIS TYPE IS `DT_CHECK_MSG(if_op, "Expect valid IfNode")` (`:661`). `isIfNode()` admits
/// `affine.if` and `scf.if` as well (`src/Analysis/ConditionalTree.cpp:99-102`), so that abort is
/// reachable in the reference; here an if-node that is not a `sentient.if` has no representation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IfNode<'a> {
    /// `getThenRegion()`.
    pub(crate) then_body: &'a [Op],
    /// `getElseRegion()` — empty is the reference's absent else region.
    pub(crate) else_body: &'a [Op],
}

impl<'a> IfNode<'a> {
    /// The `sentient.if`, or `None` for any other op.
    #[must_use]
    pub(crate) fn of(op: &'a Op) -> Option<IfNode<'a>> {
        let Op::Sentient(ops::Op::If {
            then_body,
            else_body,
            ..
        }) = op
        else {
            return None;
        };
        Some(IfNode {
            then_body,
            else_body,
        })
    }

    /// `getThenRegion().front().getTerminator()->getNumOperands()` — a body with no `sentient.yield`
    /// hands nothing back.
    #[must_use]
    fn yielded_operands(&self) -> i32 {
        match self.then_body.last() {
            Some(Op::Sentient(ops::Op::Yield { results })) => results.len() as i32,
            _ => 0,
        }
    }
}

/// `dcc::CondNode` AS e092 READS IT — the conditional tree itself is out of campaign scope
/// (`src/Analysis/ConditionalTree.hpp:32`), so this is the projection its caller fills.
#[derive(Debug, Clone)]
pub(crate) enum CondNode<'a> {
    /// The root, a then-node, an else-node, or any node whose op is not an if: all of them only
    /// accumulate their children (`:653-659`).
    Branches(Vec<CondNode<'a>>),
    /// An if-node over a `sentient.if`.
    If {
        /// Which op — and the key `if_op_to_pred_val` is looked up by.
        at: OpPath,
        /// The op itself.
        if_op: IfNode<'a>,
        /// `getThenNode()`, the first child.
        then_node: Box<CondNode<'a>>,
        /// `getElseNode()`: the last child when there are two, `nullptr` when there is one.
        else_node: Option<Box<CondNode<'a>>>,
    },
}

/// THE GREEDY CANONICALISER RUN OVER THE COLLECTED `sentient.if`s — a trait, because the pattern
/// behind it is the DIALECT's (`RemoveStaticCondition`,
/// `dcc/src/Dialect/Sentient/SentientOps.cpp:1509`) and `dcc/src/Dialect/` is not in this campaign.
pub(crate) trait IfOpCanonicalizer {
    /// `applyOpPatternsGreedily(if_ops, frozen, config{strictness = ExistingOps})` — in place, over
    /// exactly the ops named, which is what `ExistingOps` means.
    fn canonicalize_if_ops(&mut self, unit: &mut Vec<Op>, if_ops: &[OpPath]);
}

/// THE ONE CRATE IMPLEMENTATION: the pattern is not ported, so running it is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OutOfScopeIfOpCanonicalizer;

impl IfOpCanonicalizer for OutOfScopeIfOpCanonicalizer {
    fn canonicalize_if_ops(&mut self, _unit: &mut Vec<Op>, _if_ops: &[OpPath]) {
        todo!(
            "sentient::IfOp::getCanonicalizationPatterns / RemoveStaticCondition \
             (Dialect/Sentient/SentientOps.cpp:1509) — out of campaign scope"
        )
    }
}

/// `Operation::walk`, pre-order — every op of the body and everything nested in it.
fn for_each_op<'a>(body: &'a [Op], visit: &mut impl FnMut(&'a Op)) {
    for op in body {
        visit(op);
        match op {
            Op::Sentient(inner) => {
                for region in ops::regions(inner) {
                    for_each_op(region, visit);
                }
            }
            Op::AffineFor(loop_op) => for_each_op(&loop_op.body, visit),
            _ => {}
        }
    }
}

/// `unit_op->walk([&](sentient::IfOp if_op) { ifOps.push_back(if_op); })` — the if ops as the
/// [`OpPath`]s that name them, which is the identity an op has in this island.
fn if_op_paths(unit: &[Op]) -> Vec<OpPath> {
    fn descend(body: &[Op], base: &[(u32, u32)], region: u32, out: &mut Vec<OpPath>) {
        for (index, op) in body.iter().enumerate() {
            let mut here = base.to_vec();
            here.push((region, index as u32));
            if matches!(op, Op::Sentient(ops::Op::If { .. })) {
                out.push(OpPath::at(&here));
            }
            match op {
                Op::Sentient(inner) => {
                    for (sub, nested) in ops::regions(inner).into_iter().enumerate() {
                        descend(nested, &here, sub as u32, out);
                    }
                }
                Op::AffineFor(loop_op) => descend(&loop_op.body, &here, 0, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    descend(unit, &[], 0, &mut out);
    out
}

/// The `sentient.for` a [`ForRef`] names, wherever in the unit it sits.
fn for_op_at(unit: &[Op], loop_ref: ForRef) -> Option<&Op> {
    let mut found = None;
    for_each_op(unit, &mut |op| {
        if found.is_none()
            && let Op::Sentient(ops::Op::For { iv, .. }) = op
            && *iv == loop_ref.0
        {
            found = Some(op);
        }
    });
    found
}

/// `value.getDefiningOp<sentient::ConstantOp>().getValue()`.
fn constant_value(value: Val, scope: &[Op]) -> Option<i64> {
    match defining_op(value, scope) {
        Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => Some(*value),
        _ => None,
    }
}

/// Replaces: e085_replaceIterArgsAndYieldResults
///
/// Rewires every reader of the loop's iter args onto its inits, then every reader of the loop's
/// results onto the values its body yields.
///
/// ⛔ TRAP: the yield operands are read AFTER all the arg replacements, so an arg rewrite changes
/// what a result is rewired to — `sentient.yield %arg1` hands back the init.
/// ⭐ `llvm::zip` stops at the shorter list, so a body with no `sentient.yield` rewires no result.
pub(crate) fn replace_iter_args_and_yield_results(scope: &mut [Op], at: usize) {
    let Some(carried) = scope
        .get(at)
        .and_then(ForOp::of)
        .map(|for_op| for_op.carried.to_vec())
    else {
        return;
    };
    for entry in &carried {
        replace_all_uses_with(scope, entry.arg, entry.init);
    }
    let yielded = match scope
        .get(at)
        .and_then(ForOp::of)
        .and_then(|for_op| for_op.body.last())
    {
        Some(Op::Sentient(ops::Op::Yield { results })) => results.clone(),
        _ => Vec::new(),
    };
    for (entry, inner) in carried.iter().zip(yielded) {
        replace_all_uses_with(scope, entry.result, inner);
    }
}

/// Replaces: e086_isOkToUnroll
///
/// Refuses a loop whose constant-initialised iter arg reaches the MUTABLE address operand of a
/// memory op: unrolling makes that address constant, and the copies it then needs grow live ranges
/// until register allocation fails.
///
/// ⛔ TRAP: only the mutable operand counts — the same value as an immutable address is fine.
/// ⭐ `DisableLoopUnroll` (`:46-48`) IS DROPPED: an `llvm::cl::opt<bool>` with `cl::init(false)` is a
/// command-line flag of `dcc-opt`, not a program property, and this crate has no flags.
#[must_use]
pub(crate) fn is_ok_to_unroll(for_op: ForOp<'_>, scope: &[Op]) -> bool {
    let mut is_ok_to_unroll = true;
    for entry in for_op.carried {
        if !matches!(
            defining_op(entry.init, scope),
            Some(Op::Sentient(ops::Op::ScalarConstant { .. }))
        ) {
            continue;
        }
        let iter_arg = entry.arg;
        for_each_op(for_op.body, &mut |op| {
            let mutable_use = match op {
                Op::Sentient(ops::Op::LoadAndStore {
                    src_mutable_addr,
                    dst_mutable_addr,
                    ..
                }) => *src_mutable_addr == iter_arg || *dst_mutable_addr == iter_arg,
                Op::Sentient(
                    ops::Op::LoadAndSend { mutable_addr, .. }
                    | ops::Op::ReceiveAndStore { mutable_addr, .. }
                    | ops::Op::LoadAndExtractScalar { mutable_addr, .. }
                    | ops::Op::LoadComputeAndSend { mutable_addr, .. },
                ) => *mutable_addr == iter_arg,
                _ => false,
            };
            if mutable_use {
                is_ok_to_unroll = false;
            }
        });
    }
    is_ok_to_unroll
}

/// Replaces: e087_cleanupAndRecalculate
///
/// Canonicalises the unit's `sentient.if`s, re-counts instructions, rebuilds the loop forest, then
/// records every loop's ibuff cost and collects the constant-bound loops as candidates.
///
/// ⛔ TRAP: BOTH walks are reverse-BFS, so `unrolling_or_splitting_candidates` comes out
/// deepest-and-rightmost first — the order `findBestLoopNodeToOptimize` (e603) then iterates.
/// ⭐ `erased` feeds only an `LLVM_DEBUG` (`:311`) and is dropped; the rewrite itself is not.
pub(crate) fn cleanup_and_recalculate<E: InstructionEstimator, C: IfOpCanonicalizer>(
    pass: &mut LoopSplittingAndUnrolling,
    tree: &mut LoopForest,
    ie: &mut E,
    canonicalizer: &mut C,
    unit_op: &mut Vec<Op>,
) {
    pass.unrolling_or_splitting_candidates.clear();
    pass.loopnode_to_ibuff.clear();
    let if_ops = if_op_paths(unit_op);
    canonicalizer.canonicalize_if_ops(unit_op, &if_ops);
    ie.recalculate(unit_op);
    tree.recompute(unit_op);
    // Calculate the ibuff of every loop node in the loop tree.
    for id in tree.reverse_bfs() {
        let Some(loop_ref) = tree.node(id).and_then(|node| node.op) else {
            continue;
        };
        let Some(op) = for_op_at(unit_op, loop_ref) else {
            continue;
        };
        let ibuff = ie.estimated_instruction_count_of_op(op);
        pass.loopnode_to_ibuff.insert(id, ibuff);
    }
    // Collect all the possible candidates for split/unroll.
    for id in tree.reverse_bfs() {
        let Some(loop_ref) = tree.node(id).and_then(|node| node.op) else {
            continue;
        };
        let Some(for_op) = for_op_at(unit_op, loop_ref).and_then(ForOp::of) else {
            continue;
        };
        if constant_value(for_op.bound, unit_op).is_some() {
            pass.unrolling_or_splitting_candidates.push(id);
        }
    }
}

/// Replaces: e088_ifOpUsesIV
///
/// True when a `sentient.if` compares against the loop's induction variable.
///
/// ⭐ THE INDUCTION VARIABLE'S USERS ARE ITS OWN BODY: it is the region's first argument, so no user
/// outside the loop can exist, and `lhs`/`rhs` are a `sentient.if`'s only operands.
/// ⭐ THE ROOT NODE'S NULL OPERATION IS DISCHARGED BY THE CALLER: [`LoopNode::op`] is an `Option`, so
/// the reference's `dyn_cast` of a null operation has no case here.
#[must_use]
pub(crate) fn if_op_uses_iv(loop_ref: ForRef, unit: &[Op]) -> bool {
    let Some(for_op) = for_op_at(unit, loop_ref).and_then(ForOp::of) else {
        return false;
    };
    let mut uses_iv = false;
    for_each_op(for_op.body, &mut |op| {
        if let Op::Sentient(ops::Op::If { lhs, rhs, .. }) = op
            && (*lhs == for_op.iv || *rhs == for_op.iv)
        {
            uses_iv = true;
        }
    });
    uses_iv
}

/// Replaces: e089_hasIfOps
///
/// True when the loop holds a `sentient.if` anywhere inside it.
///
/// ⭐ `op->walk` VISITS THE OP ITSELF, but a `sentient.for` is never a `sentient.if`, so this is the
/// body. `WalkResult::interrupt()` only stops the walk early — it does not change the answer.
#[must_use]
pub(crate) fn has_if_ops(loop_ref: ForRef, unit: &[Op]) -> bool {
    let Some(for_op) = for_op_at(unit, loop_ref).and_then(ForOp::of) else {
        return false;
    };
    let mut contain_if_ops = false;
    for_each_op(for_op.body, &mut |op| {
        if matches!(op, Op::Sentient(ops::Op::If { .. })) {
            contain_if_ops = true;
        }
    });
    contain_if_ops
}

/// Replaces: e090_evaluatePredicate
///
/// The if op's predicate evaluated at one iteration bound, comparing whichever side is a constant
/// against the bound — `false` when neither side is.
///
/// ⛔ TRAP: A CONSTANT LHS IS TRIED FIRST, and the comparison keeps the operand order, so `slt`
/// reads `lhs < bound` on the left and `bound < rhs` on the right. ⭐ The trailing `else` is `ne`:
/// sentient declares its own six-case `CmpIPredicate` (`SentientTypes.td:481-484`), not `arith`'s.
#[must_use]
pub(crate) fn evaluate_predicate(bound: IterBound, if_op: &Op, scope: &[Op]) -> bool {
    let Op::Sentient(ops::Op::If {
        predicate,
        lhs,
        rhs,
        ..
    }) = if_op
    else {
        return false;
    };
    if let Some(lhs) = constant_value(*lhs, scope) {
        return match predicate {
            ops::CmpPredicate::Sge => lhs >= bound.0,
            ops::CmpPredicate::Sgt => lhs > bound.0,
            ops::CmpPredicate::Slt => lhs < bound.0,
            ops::CmpPredicate::Sle => lhs <= bound.0,
            ops::CmpPredicate::Eq => lhs == bound.0,
            ops::CmpPredicate::Ne => lhs != bound.0,
        };
    }
    if let Some(rhs) = constant_value(*rhs, scope) {
        return match predicate {
            ops::CmpPredicate::Sge => bound.0 >= rhs,
            ops::CmpPredicate::Sgt => bound.0 > rhs,
            ops::CmpPredicate::Slt => bound.0 < rhs,
            ops::CmpPredicate::Sle => bound.0 <= rhs,
            ops::CmpPredicate::Eq => rhs == bound.0,
            ops::CmpPredicate::Ne => rhs != bound.0,
        };
    }
    false
}

/// Replaces: e091_findAllSplitVals
///
/// Groups the consecutive descending bounds whose if-op predicate values are all equal, emitting one
/// `(highest, lowest)` range per group.
///
/// ⛔ TRAP: the comparison is against the group's FIRST entry, not the previous one, so a group is a
/// run of bounds identical to where it started.
/// ⭐ The final push is outside the loop: the last group is always emitted.
#[must_use]
pub(crate) fn find_all_split_vals(bound_to_all_ifops_predval: &BoundPredVals) -> Vec<SplitRange> {
    let entries = bound_to_all_ifops_predval.entries();
    let mut split_vals = Vec::new();
    let (mut first, mut second) = (0usize, 0usize);
    let mut p_first = bound_to_all_ifops_predval.head.0;
    let mut p_second = bound_to_all_ifops_predval.head.0;
    while let (Some(at_first), Some(at_second)) = (entries.get(first), entries.get(second)) {
        if at_first.1 == at_second.1 {
            p_second = at_second.0;
            second += 1;
        } else {
            split_vals.push(SplitRange {
                high: p_first,
                low: p_second,
            });
            first = second;
            p_first = at_second.0;
            p_second = at_second.0;
        }
    }
    split_vals.push(SplitRange {
        high: p_first,
        low: p_second,
    });
    split_vals
}

/// Replaces: e092_computeReducedCost
///
/// Accumulates what the splitting saves: per if op whose predicate is statically known, the op and
/// its yield operands, plus the whole cost of the branch that dies — then on down the live branch.
///
/// ⛔ TRAP: an if op that is NOT in the map still costs a recursion into BOTH branches; only a known
/// predicate prunes one.
/// ⭐ THEN/ELSE/ROOT NODES AND ANY NON-IF NODE ONLY ACCUMULATE THEIR CHILDREN (`:653-659`).
pub(crate) fn compute_reduced_cost<E: InstructionEstimator>(
    node: Option<&CondNode<'_>>,
    ie: &mut E,
    cost_reduced: &mut InstructionCount,
    if_op_to_pred_val: &BTreeMap<OpPath, bool>,
) {
    let Some(node) = node else {
        return;
    };
    let (at, if_op, then_node, else_node) = match node {
        // For Then/Else/Root nodes simply accumulate reduced costs of the children.
        CondNode::Branches(children) => {
            for child in children {
                compute_reduced_cost(Some(child), ie, cost_reduced, if_op_to_pred_val);
            }
            return;
        }
        CondNode::If {
            at,
            if_op,
            then_node,
            else_node,
        } => (at, if_op, then_node, else_node),
    };
    let Some(&pred_eval) = if_op_to_pred_val.get(at) else {
        compute_reduced_cost(Some(then_node), ie, cost_reduced, if_op_to_pred_val);
        compute_reduced_cost(else_node.as_deref(), ie, cost_reduced, if_op_to_pred_val);
        return;
    };
    cost_reduced.0 += 1 + if_op.yielded_operands();
    if pred_eval {
        // Else-branch is dead. Accumulate its cost, recurse on Then-branch.
        cost_reduced.0 += ie.estimated_instruction_count_of_region(if_op.else_body).0;
        compute_reduced_cost(Some(then_node), ie, cost_reduced, if_op_to_pred_val);
    } else {
        // Then-branch is dead. Accumulate its cost, recurse on Else-branch.
        cost_reduced.0 += ie.estimated_instruction_count_of_region(if_op.then_body).0;
        compute_reduced_cost(else_node.as_deref(), ie, cost_reduced, if_op_to_pred_val);
    }
}

/// Replaces: e318_promoteForLoopBodyAndDelete
///
/// Rewires the loop's iter args and results ([`replace_iter_args_and_yield_results`]), then hoists
/// its body — minus the terminator — into the loop's own place and erases the loop.
///
/// ⛔ TRAP: THE SPLICE IS AT THE LOOP'S POSITION, not the end of the block, so the hoisted ops keep
/// their order against the loop's siblings — a later sibling still reads them.
/// ⭐ `getBody()->back().erase()` erases the terminator unconditionally; this island materialises a
/// `sentient.yield` only when the loop carries something, so the drop is guarded.
pub(crate) fn promote_for_loop_body_and_delete(scope: &mut Vec<Op>, at: usize) {
    if scope.get(at).and_then(ForOp::of).is_none() {
        return;
    }
    replace_iter_args_and_yield_results(scope, at);
    let Some(Op::Sentient(ops::Op::For { body, .. })) = scope.get_mut(at) else {
        return;
    };
    let mut hoisted = core::mem::take(body);
    if matches!(hoisted.last(), Some(Op::Sentient(ops::Op::Yield { .. }))) {
        hoisted.pop();
    }
    let tail = scope.split_off(at + 1);
    scope.truncate(at);
    scope.extend(hoisted);
    scope.extend(tail);
}

/// Replaces: e319_getAllTrueIndexOfIfOpsList
///
/// Each if op's predicate ([`evaluate_predicate`]) at one iteration bound, in the list's order.
///
/// ⛔ TRAP: THE NAME IS A MISNOMER — nothing is filtered and no index is returned: the result is one
/// `bool` per if op, positionally, and its callers compare whole vectors across bounds.
#[must_use]
pub(crate) fn get_all_true_index_of_if_ops_list(
    bound: IterBound,
    if_list: &[&Op],
    scope: &[Op],
) -> Vec<bool> {
    if_list
        .iter()
        .map(|if_op| evaluate_predicate(bound, if_op, scope))
        .collect()
}

/// Replaces: e320_subsetsImpl
///
/// Every subset of the split ranges, emitted in the recursion's own order: the running subset first,
/// then one descent per remaining range.
///
/// ⭐ THE EMPTY SUBSET IS THE FIRST ENTRY — `res.push_back(subset)` runs before the loop, so the
/// caller's initially empty `subset` is recorded, and each range is pushed then popped in place.
fn subsets_impl(
    split_vals: &[SplitRange],
    res: &mut Vec<Vec<SplitRange>>,
    subset: &mut Vec<SplitRange>,
    index: usize,
) {
    res.push(subset.clone());
    for at in index..split_vals.len() {
        subset.push(split_vals[at]);
        subsets_impl(split_vals, res, subset, at + 1);
        subset.pop();
    }
}

/// Replaces: e446_subsets
///
/// Every subset of the split ranges ([`subsets_impl`]), ordered by DECREASING SIZE, which is the order
/// `findOptimalNWaySplits` tries them in.
///
/// ⭐ STABLE, WHERE `std::sort` IS NOT: equal-sized subsets keep the recursion's own enumeration order
/// here and have no defined order there, so this is the version whose candidate list is the same
/// twice. ⭐ The empty subset is enumerated first and therefore sorts LAST.
#[must_use]
pub(crate) fn subsets(split_vals: &[SplitRange]) -> Vec<Vec<SplitRange>> {
    let mut res = Vec::new();
    let mut subset = Vec::new();
    subsets_impl(split_vals, &mut res, &mut subset, 0);
    res.sort_by(|first, second| second.len().cmp(&first.len()));
    res
}

/// WHETHER THE LOOP WAS SPLIT — `splitLoopNWay`'s `LogicalResult`, whose failure has one cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopSplit {
    /// `success()`: the loop is gone and one loop per split range stands in its place.
    Split,
    /// `failure()`: *"Unable to Perform Loop Splitting, due to dynamic bound"*.
    ///
    /// ⭐ A NON-POSITIVE CONSTANT BOUND IS ALSO HERE: the reference fills its map with
    /// `while (loop_bound_copy)` from the bound down to 1, so a bound of 0 leaves it empty and
    /// `findAllSplitVals` then dereferences `begin()` — undefined behaviour there, refused here.
    BoundIsNotAPositiveConstant,
}

/// ⛔ `getDefiningOp()` IS GLOBAL, so a value resolves in whichever enclosing scope defines it: the
/// clone's own body for a constant it carries, the block above for one hoisted out of the loop.
fn constant_value_in(value: Val, scopes: &[&[Op]]) -> Option<i64> {
    scopes.iter().find_map(|scope| constant_value(value, scope))
}

/// `(getLhs() or getRhs() is a sentient.constant) && (iv == getLhs() || iv == getRhs())` — the filter
/// both of `splitLoopNWay`'s walks apply (`:875-881`, `:923-929`).
fn is_if_op_on_iv(op: &Op, iv: Val, scopes: &[&[Op]]) -> bool {
    let Op::Sentient(ops::Op::If { lhs, rhs, .. }) = op else {
        return false;
    };
    let pred_can_be_evaluated =
        constant_value_in(*lhs, scopes).is_some() || constant_value_in(*rhs, scopes).is_some();
    pred_can_be_evaluated && (iv == *lhs || iv == *rhs)
}

/// The `sentient.if`s of `body` on `iv`, in the pre-order `for_op->walk` visits them.
fn if_ops_on_iv<'a>(body: &'a [Op], iv: Val, scopes: &[&[Op]]) -> Vec<&'a Op> {
    let mut if_list = Vec::new();
    for_each_op(body, &mut |op| {
        if is_if_op_on_iv(op, iv, scopes) {
            if_list.push(op);
        }
    });
    if_list
}

/// Splices each `sentient.if` on `iv` down to the branch `predvals` names positionally, walking
/// `if_list1`'s pre-order, and collects the rewires the reference performs immediately per if op.
///
/// ⛔⛔ BOTH REGIONS ARE DESCENDED INTO EVEN THOUGH ONE IS DISCARDED, because that is what fixes
/// `predvals`' positions: `if_list1` holds the DISCARDED branch's if ops too, at their own pre-order
/// slots. The reference processes them inside the dying region and then double-frees them in its
/// second erase loop (`:961-963`); here the folding is real and the region is simply dropped after.
///
/// ⭐ AN EMPTY ELSE REGION SPLICES NOTHING AND REWIRES NOTHING (`:943`), so the if op's results lose
/// their definition and keep their readers — the reference's own outcome.
fn fold_if_ops_on_iv(
    block: Vec<Op>,
    iv: Val,
    scopes: &[&[Op]],
    predvals: &[bool],
    index: &mut usize,
    rewires: &mut Vec<Vec<(Val, Val)>>,
) -> Vec<Op> {
    let mut out = Vec::new();
    for op in block {
        if !is_if_op_on_iv(&op, iv, scopes) {
            let mut op = op;
            // ⛔ THE REGION TREE IS `walk`'S, NOT ONE DIALECT'S: an if op on the iv inside an
            // `affine.for` is in `if_list1` at its own slot, so stopping here would both leave it
            // standing and shift every later if op's `predvals` position by one.
            match &mut op {
                Op::Sentient(inner) => {
                    for region in ops::regions_mut(inner) {
                        *region = fold_if_ops_on_iv(
                            core::mem::take(region),
                            iv,
                            scopes,
                            predvals,
                            index,
                            rewires,
                        );
                    }
                }
                Op::AffineFor(loop_op) => {
                    loop_op.body = fold_if_ops_on_iv(
                        core::mem::take(&mut loop_op.body),
                        iv,
                        scopes,
                        predvals,
                        index,
                        rewires,
                    );
                }
                _ => {}
            }
            out.push(op);
            continue;
        }
        // `bound_to_all_ifops_predval[kv.first][index]` — a vector shorter than the list cannot
        // happen (both walks apply the same filter to the same structure), and reads as the else.
        let take_then = predvals.get(*index).copied().unwrap_or(false);
        *index += 1;
        // The outer if op's rewire is performed BEFORE its regions' are, so its slot is reserved
        // here: a result rewired onto an inner if's result must be redirected again by that inner
        // rewire, which only holds when the two are applied in this order.
        let slot = rewires.len();
        rewires.push(Vec::new());
        let Op::Sentient(ops::Op::If {
            yielded,
            then_body,
            else_body,
            ..
        }) = op
        else {
            continue;
        };
        let produced: Vec<Val> = yielded.iter().map(|entry| entry.result).collect();
        let else_was_empty = else_body.is_empty();
        let then_out = fold_if_ops_on_iv(then_body, iv, scopes, predvals, index, rewires);
        let else_out = fold_if_ops_on_iv(else_body, iv, scopes, predvals, index, rewires);
        if !take_then && else_was_empty {
            continue;
        }
        let mut region = if take_then { then_out } else { else_out };
        // `cloned_if_op.getResult(i).replaceAllUsesWith(cloned_if_yield_op->getOperand(i))`, then
        // `cloned_if_yield_op->erase()`.
        if let Some(Op::Sentient(ops::Op::Yield { results })) = region.last() {
            rewires[slot] = produced.into_iter().zip(results.iter().copied()).collect();
            region.pop();
        }
        out.extend(region);
    }
    out
}

/// `dcc::utils::updateDbgName(prefix, op, suffix)` (`dcc/src/Utils/Utils.cpp:491-515`) — ⛔ IT WRITES
/// ONLY WHERE A NAME ALREADY EXISTS, because `getNewDbgNameFromOp` returns null without one.
fn update_dbg_name(op: &mut Op, prefix: &str, suffix: &str) {
    let Op::Sentient(inner) = op else {
        return;
    };
    if let Some(Some(name)) = ops::dbg_name_mut(inner) {
        *name = format!("{prefix}{name}{suffix}");
    }
}

/// `%c = sentient.scalar_constant {value} : index`, as `const_builder_` creates one.
fn index_constant(value: i64, result: Val) -> Op {
    Op::Sentient(ops::Op::ScalarConstant {
        value,
        result,
        reg_locale: ops::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    })
}

/// Replaces: e447_splitLoopNWay
///
/// Replaces the loop at `at` with ONE LOOP PER SPLIT RANGE, each with the range's own trip count, its
/// induction variable shifted to the range's base, and every `sentient.if` on that variable already
/// folded to the branch the range decided.
///
/// ⛔ TRAP: `builder.clone` RUNS BEFORE `updateDbgName`, so clone *k* carries the name the original
/// had after *k-1* renamings and the fully accumulated name is discarded with the original.
/// ⛔ TRAP: THE TAIL REWIRE USES THE LAST CLONE ALONE — the loops are chained by
/// `newiter_args`, so only the final one's results stand for the original's.
/// ⚠️ DIVERGENCE: each range's constants land immediately before their clone rather than at
/// `const_builder_`'s program-unit top, the divergence `e322_copyOneIter` already records.
pub(crate) fn split_loop_n_way(scope: &mut Vec<Op>, at: usize, values: &mut Values) -> LoopSplit {
    let Some(for_op) = scope.get(at).and_then(ForOp::of) else {
        return LoopSplit::BoundIsNotAPositiveConstant;
    };
    let Some(loop_bound) = constant_value(for_op.bound, scope) else {
        return LoopSplit::BoundIsNotAPositiveConstant;
    };
    if loop_bound <= 0 {
        return LoopSplit::BoundIsNotAPositiveConstant;
    }
    let inits: Vec<Val> = for_op.carried.iter().map(|entry| entry.init).collect();
    let original_results: Vec<Val> = for_op.carried.iter().map(|entry| entry.result).collect();
    // `getAllTrueIndexOfIfOpsList(loop_bound_copy, if_list)` for every bound from `loop_bound` down
    // to 1; the map is `std::greater<>`-ordered and [`BoundPredVals`] reads it back that way.
    let if_list = if_ops_on_iv(for_op.body, for_op.iv, &[scope]);
    let mut bound_to_all_ifops_predval = BTreeMap::new();
    for bound in 1..=loop_bound {
        let bound = IterBound(bound);
        bound_to_all_ifops_predval.insert(
            bound,
            get_all_true_index_of_if_ops_list(bound, &if_list, scope),
        );
    }
    drop(if_list);
    let Some(pred_vals) = BoundPredVals::from_map(&bound_to_all_ifops_predval) else {
        return LoopSplit::BoundIsNotAPositiveConstant;
    };
    let split_vals = find_all_split_vals(&pred_vals);

    let mut newiter_args: Vec<Val> = Vec::new();
    // `OpBuilder builder(for_op->getNextNode())` — every clone is inserted just before the loop's
    // next sibling, so the clones end up after the loop in `split_vals` order.
    let mut cursor = at + 1;
    for range in &split_vals {
        let mut minted = Vec::new();
        let bound_val = values.mint();
        minted.push(index_constant(range.high.0 - range.low.0 + 1, bound_val));
        let template = scope[at].clone();
        let mut mapping = ValueMapping::new();
        let mut cloned = clone_ops(&[template], values, &mut mapping);
        update_dbg_name(&mut scope[at], "LS(", &format!(", sv {})", range.high.0));
        let Some(Op::Sentient(ops::Op::For {
            iv: cloned_iv,
            bound,
            carried,
            body,
            ..
        })) = cloned.first_mut()
        else {
            return LoopSplit::BoundIsNotAPositiveConstant;
        };
        let cloned_iv = *cloned_iv;
        *bound = bound_val;
        // `getInitArgsMutable().assign(newiter_args.size() > 0 ? newiter_args : getIterOperands())`.
        let assigned = if newiter_args.is_empty() {
            &inits
        } else {
            &newiter_args
        };
        for (entry, init) in carried.iter_mut().zip(assigned) {
            entry.init = *init;
        }
        // ⛔ THE FILTER NEEDS BOTH SCOPES: the clone's body for the constants it carries — remapped,
        // so only this snapshot defines them — and the block the loop sits in for the ones hoisted
        // above it, which the clone reads unchanged. A body-only scope answers "not on the induction
        // variable" for every hoisted comparison and folds nothing.
        let snapshot = body.clone();
        let scopes: [&[Op]; 2] = [&snapshot, scope];
        let predvals = bound_to_all_ifops_predval
            .get(&range.high)
            .cloned()
            .unwrap_or_default();
        let mut index = 0;
        let mut rewires = Vec::new();
        *body = fold_if_ops_on_iv(
            core::mem::take(body),
            cloned_iv,
            &scopes,
            &predvals,
            &mut index,
            &mut rewires,
        );
        for (of, with) in rewires.into_iter().flatten() {
            replace_all_uses_with(body, of, with);
        }
        // `if (!getInductionVar().use_empty())` — the range's iterations start at `kv.second - 1`,
        // so every surviving reader of the clone's variable reads `iv + (low - 1)` instead.
        if use_count(cloned_iv, body) > 0 {
            let shift_val = values.mint();
            minted.push(index_constant(range.low.0 - 1, shift_val));
            let sum = values.mint();
            body.insert(
                0,
                Op::Sentient(ops::Op::ScalarAdd {
                    lhs: cloned_iv,
                    rhs: shift_val,
                    result: sum,
                    reg: None,
                    ty: ScalarTy::Index,
                    element_size: None,
                }),
            );
            // `replaceAllUsesExcept(add_op.getResult(), {add_op})` — the addition itself keeps
            // reading the variable, which is why the rewrite starts past it.
            replace_all_uses_with(&mut body[1..], cloned_iv, sum);
        }
        newiter_args = carried.iter().map(|entry| entry.result).collect();
        let inserted = minted.len() + cloned.len();
        minted.append(&mut cloned);
        for (offset, op) in minted.into_iter().enumerate() {
            scope.insert(cursor + offset, op);
        }
        cursor += inserted;
    }
    for (original, last) in original_results.iter().zip(&newiter_args) {
        replace_all_uses_with(scope, *original, *last);
    }
    scope.remove(at);
    LoopSplit::Split
}

/// WHETHER THE LOOP WAS UNROLLED — `unrollLoop`'s `LogicalResult`, whose failure has one cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopUnroll {
    /// `success()`: the loop is gone and its iterations stand in its place — ⛔ OR IT WAS DYNAMIC and
    /// the pass only remarked *"Dynamic loops cannot be unrolled"* (`:290-291`), which is also
    /// `success()` and leaves the loop exactly where it was.
    Done,
    /// `failure()`: [`is_ok_to_unroll`] refused, because unrolling would freeze an iter arg into a
    /// constant address and register allocation would then fail.
    WouldFailRegisterAllocation,
}

/// Replaces: e513_unrollLoop
///
/// Replaces a constant-bound loop with its `loop_size` iterations laid out flat, each carrying the
/// previous one's yielded values and its own induction-variable constant.
///
/// ⛔ TRAP: THE ORIGINAL BODY IS ITERATION `loop_size` AND STAYS FIRST — the clones run `loop_size-1`
/// down to 1 after it — and its readers of the variable are finally rewired to the BOUND (`:286`),
/// not to a constant of their own iteration.
/// ⛔ TRAP: A CARRY-FREE EMPTY BODY ERASES THE LOOP (`:217-220`) and a bound of 0 hands every result
/// back to its initialiser first (`:224-232`) — two erasures that unroll nothing.
/// ⚠️ DIVERGENCE: `const_builder`'s constants land just before the loop rather than at the program
/// unit's top, the divergence [`split_loop_n_way`] already records.
pub(crate) fn unroll_loop(scope: &mut Vec<Op>, at: usize, values: &mut Values) -> LoopUnroll {
    let Some(for_op) = scope.get(at).and_then(ForOp::of) else {
        return LoopUnroll::WouldFailRegisterAllocation;
    };
    if !is_ok_to_unroll(for_op, scope) {
        return LoopUnroll::WouldFailRegisterAllocation;
    }
    let (iv, bound) = (for_op.iv, for_op.bound);
    let carried: Vec<ops::Carried> = for_op.carried.to_vec();
    let body: Vec<Op> = for_op.body.to_vec();

    // "Nothing in the loop body other than the terminator" — this island materialises a
    // `sentient.yield` only for a loop that carries something, so a carry-free empty body is that.
    if body.is_empty() && carried.is_empty() {
        scope.remove(at);
        return LoopUnroll::Done;
    }
    let Some(loop_size) = constant_value(bound, scope) else {
        // `for_op->emitRemark("Dynamic loops cannot be unrolled")` — a diagnostic and nothing else.
        return LoopUnroll::Done;
    };
    if loop_size == 0 {
        for entry in &carried {
            replace_all_uses_with(scope, entry.result, entry.init);
        }
        scope.remove(at);
        return LoopUnroll::Done;
    }

    let has_terminator = matches!(body.last(), Some(Op::Sentient(ops::Op::Yield { .. })));
    // `std::prev(loop_body->end(), 2)` — the body without its terminator, which is what each
    // unrolled instance clones.
    let template = if has_terminator {
        &body[..body.len() - 1]
    } else {
        &body[..]
    };
    let original_yield_operands: Vec<Val> = match body.last() {
        Some(Op::Sentient(ops::Op::Yield { results })) => results.clone(),
        _ => Vec::new(),
    };
    let mut modified_yield_operands = original_yield_operands.clone();
    let iv_is_used = use_count(iv, &body) > 0;

    let mut consts: Vec<Op> = Vec::new();
    let mut clones: Vec<Op> = Vec::new();
    for iteration in (1..loop_size).rev() {
        let mut mapping = ValueMapping::new();
        // `operandMap.map(loop_iter_args, modified_yield_operands)` — this instance reads what the
        // PREVIOUS one yielded, which for the first clone is the original body's own results.
        for (entry, yielded) in carried.iter().zip(&modified_yield_operands) {
            mapping.map(entry.arg, *yielded);
        }
        if iv_is_used {
            let iv_value = values.mint();
            consts.push(index_constant(iteration, iv_value));
            mapping.map(iv, iv_value);
        }
        let mut cloned = clone_ops(template, values, &mut mapping);
        for op in &mut cloned {
            update_dbg_name(op, "LU(", &format!(", i={iteration})"));
        }
        clones.append(&mut cloned);
        // ⛔ THE LOOKUP IS OF THE **ORIGINAL** OPERAND, not of the running one: the map for this
        // instance is keyed by the body as written. A yield operand defined outside the body is
        // mapped by nothing and stands for itself — `IRMapping::lookup` would abort there.
        for (slot, original) in modified_yield_operands
            .iter_mut()
            .zip(&original_yield_operands)
        {
            *slot = mapping.lookup_or_default(*original);
        }
    }

    let Some(Op::Sentient(ops::Op::For { body, .. })) = scope.get_mut(at) else {
        return LoopUnroll::Done;
    };
    if has_terminator {
        body.pop();
    }
    for op in body.iter_mut() {
        update_dbg_name(op, "LU(", &format!(", i={loop_size})"));
    }
    body.append(&mut clones);
    if has_terminator {
        body.push(Op::Sentient(ops::Op::Yield {
            results: modified_yield_operands,
        }));
    }
    // `getInductionVar().replaceAllUsesWith(const_bound)` — only the original body still reads it;
    // every clone was remapped to its own constant.
    replace_all_uses_with(body, iv, bound);

    let inserted = consts.len();
    for (offset, op) in consts.into_iter().enumerate() {
        scope.insert(at + offset, op);
    }
    promote_for_loop_body_and_delete(scope, at + inserted);
    LoopUnroll::Done
}

/// `dcc::TransformationConditionalTree` — a trait for the same reason [`IfOpCanonicalizer`] is one:
/// the tree is not in this campaign and a test must still be able to state its shape.
///
/// ⛔ `dcc/src/Analysis/ConditionalTree.hpp` IS OUTSIDE THE CAMPAIGN'S SCOPE (which is the TOP LEVEL
/// of `Transform/Sentient/`), so the crate's only implementation is [`OutOfScopeConditionalTree`].
/// ⭐ THE PATHS ITS NODES CARRY ARE READ RELATIVE TO THE LOOP'S BODY, the same convention
/// [`if_op_paths`] uses for a unit — that is what makes `if_op_to_pred_val`'s keys match.
pub(crate) trait ConditionalTree {
    /// `TransformationConditionalTree tree(*for_op); tree.compute(); tree.getRoot()` (`:740-741`).
    fn root<'a>(&mut self, for_op: &'a Op) -> CondNode<'a> {
        let _ = for_op;
        todo!(
            "dcc::TransformationConditionalTree::{{compute,getRoot}} \
             (dcc/src/Analysis/ConditionalTree.hpp) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION: the tree is not ported, so asking it for a root is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OutOfScopeConditionalTree;

impl ConditionalTree for OutOfScopeConditionalTree {}

/// `findOptimalNWaySplits`' FOUR-TUPLE (`:766-767`), whose first two entries are what
/// `isSplittingProfitable` reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NWaySplits {
    /// `index` — ⛔ ALWAYS 0: the window search that would move it is commented out (`:770-853`).
    pub(crate) index: usize,
    /// `nway_split` — how many split ranges [`find_all_split_vals`] found.
    pub(crate) nway_split: usize,
    /// `if_list.size()` — how many `sentient.if`s of the loop are on its induction variable.
    pub(crate) if_ops_on_iv: usize,
    /// `final_ibuff_cost` — the extra loops' cost less what the folded conditionals save.
    pub(crate) ibuff_cost: InstructionCount,
}

/// The `sentient.if`s of `body` on `iv` as the [`OpPath`]s naming them, in the same pre-order
/// [`if_ops_on_iv`] walks — so the two lists are positionally one list.
fn if_op_paths_on_iv(body: &[Op], iv: Val, scopes: &[&[Op]]) -> Vec<OpPath> {
    fn descend(body: &[Op], base: &[(u32, u32)], region: u32, iv: Val, scopes: &[&[Op]], out: &mut Vec<OpPath>) {
        for (index, op) in body.iter().enumerate() {
            let mut here = base.to_vec();
            here.push((region, index as u32));
            if is_if_op_on_iv(op, iv, scopes) {
                out.push(OpPath::at(&here));
            }
            match op {
                Op::Sentient(inner) => {
                    for (sub, nested) in ops::regions(inner).into_iter().enumerate() {
                        descend(nested, &here, sub as u32, iv, scopes, out);
                    }
                }
                Op::AffineFor(loop_op) => descend(&loop_op.body, &here, 0, iv, scopes, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    descend(body, &[], 0, iv, scopes, &mut out);
    out
}

/// Replaces: e514_findOptimalNWaySplits
///
/// Prices the FULL n-way split of one loop: one extra copy of the loop per split range beyond the
/// first, less every conditional those ranges fold away.
///
/// ⛔ TRAP: THE SEARCH IS COMMENTED OUT (`:770-853`), so `index` is 0 and `nway_split` is always every
/// range — and the ibuff comparison this ends with only picks a DEBUG MESSAGE (`:764-773`); the cost
/// it returns is the same whether the split fits or not.
/// ⛔ TRAP: ONE RUNNING `cost_reduced` ACROSS ALL RANGES (`:743-752`), so an if op folded by several
/// ranges is credited once per range.
/// ⭐ A bound of 0 leaves the map empty, where the reference reads `begin()` of nothing.
pub(crate) fn find_optimal_n_way_splits<E: InstructionEstimator, T: ConditionalTree>(
    scope: &[Op],
    at: usize,
    ie: &mut E,
    tree: &mut T,
    unit: &[Op],
) -> NWaySplits {
    let Some((for_op, loop_bound)) = scope
        .get(at)
        .and_then(ForOp::of)
        .and_then(|for_op| constant_value(for_op.bound, scope).map(|bound| (for_op, bound)))
    else {
        // `DT_CHECK(isa<sentient::ConstantOp>(for_op.getBound().getDefiningOp()))` (`:698`) — an
        // abort, and a loop with a dynamic bound is the one thing that reaches it.
        panic!(
            "findOptimalNWaySplits expects a constant-bound sentient.for \
             (LoopSplittingAndUnrolling.cpp:698)"
        )
    };
    let if_list = if_ops_on_iv(for_op.body, for_op.iv, &[scope]);
    let if_paths = if_op_paths_on_iv(for_op.body, for_op.iv, &[scope]);
    let mut bound_to_all_ifops_predval = BTreeMap::new();
    for bound in 1..=loop_bound {
        let bound = IterBound(bound);
        bound_to_all_ifops_predval.insert(
            bound,
            get_all_true_index_of_if_ops_list(bound, &if_list, scope),
        );
    }
    let split_vals = match BoundPredVals::from_map(&bound_to_all_ifops_predval) {
        Some(pred_vals) => find_all_split_vals(&pred_vals),
        // `while (loop_bound_copy)` leaves the map empty for a bound of 0 and `findAllSplitVals` then
        // reads `begin()` of nothing; no range is the fixed point that reads nothing.
        None => Vec::new(),
    };
    let nway_split = split_vals.len();
    // "number of loops generated multiplied by ibuff of forop" (`:735-736`).
    let per_loop = ie.estimated_instruction_count_of_op(&scope[at]).0;
    let ibuff_cost = per_loop * (nway_split as i32 - 1);

    let root = tree.root(&scope[at]);
    let mut cost_reduced = InstructionCount(0);
    for range in &split_vals {
        let predvals = bound_to_all_ifops_predval
            .get(&range.high)
            .cloned()
            .unwrap_or_default();
        // `std::map::insert` of the zipped lists — `llvm::zip` stops at the shorter one.
        let if_op_to_pred_eval: BTreeMap<OpPath, bool> =
            if_paths.iter().cloned().zip(predvals).collect();
        compute_reduced_cost(Some(&root), ie, &mut cost_reduced, &if_op_to_pred_eval);
    }
    // ⛔ ASKED FOR AND THEN ONLY LOGGED (`:762-773`) — the two arms differ by their message alone.
    let _available_ibuff_space = ie.remaining_ibuff_space(unit);
    NWaySplits {
        index: 0,
        nway_split,
        if_ops_on_iv: if_list.len(),
        ibuff_cost: InstructionCount(ibuff_cost - cost_reduced.0),
    }
}

// crustify:todo: e566_getSavedCycleAndIbuffCost
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:335  (117 body lines, level 4)
//   original  : std::pair<int, int> LoopSplittingAndUnrollingPass::getSavedCycleAndIbuffCost( dcc::LoopNode *loop, InstructionEstimatorImpl &ie, dataflow::ProgramUnitOp &unit, std::pair<int, int> &is_splitting)
//   calls     : e088_ifOpUsesIV, e514_findOptimalNWaySplits

// crustify:todo: e567_doSplitOrUnroll
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:993  (23 body lines, level 4)
//   original  : void LoopSplittingAndUnrollingPass::doSplitOrUnroll( OpBuilder &builder, dcc::LoopNode *n, std::pair<int, int> index_and_nwaysplit)
//   calls     : e447_splitLoopNWay, e513_unrollLoop

// crustify:todo: e603_findBestLoopNodeToOptimize
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:479  (69 body lines, level 5)
//   original  : std::pair<dcc::LoopNode *, std::pair<int, int>> LoopSplittingAndUnrollingPass::findBestLoopNodeToOptimize( InstructionEstimatorImpl &ie, dataflow::ProgramUnitOp &unit_op, bool disable_unrolling)
//   calls     : e088_ifOpUsesIV, e089_hasIfOps, e566_getSavedCycleAndIbuffCost

// crustify:todo: e629_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:1019  (61 body lines, level 6)
//   original  : void LoopSplittingAndUnrollingPass::runOnOperation()
//   calls     : e087_cleanupAndRecalculate, e567_doSplitOrUnroll, e597_runLightWeightSimplifications, e603_findBestLoopNodeToOptimize

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType, Yielded};

    /// AN ESTIMATOR THAT ANSWERS WHAT THE TEST SAYS. ⛔ `InstructionEstimatorImpl` is out of campaign
    /// scope, so a test STATES its counts rather than deriving them — the effect under test is what
    /// the unit does with a count, not the count.
    #[derive(Debug, Default)]
    struct StatedEstimator {
        /// How many times `recalculate` was asked for.
        recalculated: usize,
        /// What every op costs.
        per_op: i32,
        /// What is left of the ibuff, and how many times it was asked for.
        ibuff_space: i32,
        ibuff_asks: usize,
    }

    impl InstructionEstimator for StatedEstimator {
        fn recalculate(&mut self, _unit: &[Op]) {
            self.recalculated += 1;
        }

        fn estimated_instruction_count_of_op(&mut self, _op: &Op) -> InstructionCount {
            InstructionCount(self.per_op)
        }

        fn estimated_instruction_count_of_region(&mut self, region: &[Op]) -> InstructionCount {
            InstructionCount(region.len() as i32)
        }

        fn have_ibuff_space(&mut self, _unit: &[Op]) -> bool {
            todo!("no unit here asks this fake whether the ibuff has space")
        }

        fn remaining_ibuff_space(&mut self, _unit: &[Op]) -> InstructionCount {
            self.ibuff_asks += 1;
            InstructionCount(self.ibuff_space)
        }
    }

    /// A canonicaliser that records the ops it was handed. ⛔ `RemoveStaticCondition` is the
    /// dialect's and out of campaign scope; what e087 owns is WHICH ops reach it.
    #[derive(Debug, Default)]
    struct RecordingCanonicalizer {
        saw: Vec<OpPath>,
    }

    impl IfOpCanonicalizer for RecordingCanonicalizer {
        fn canonicalize_if_ops(&mut self, _unit: &mut Vec<Op>, if_ops: &[OpPath]) {
            self.saw = if_ops.to_vec();
        }
    }

    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    fn carried(init: Val, arg: Val, result: Val) -> Carried {
        Carried {
            init,
            arg,
            result,
            reg: Reg {
                locale: RegType::Lrf,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    fn for_loop(iv: Val, bound: Val, carried: Vec<Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(ops::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried,
            dbg_name: None,
            body,
        })
    }

    fn if_op(lhs: Val, rhs: Val, then_body: Vec<Op>, else_body: Vec<Op>) -> Op {
        Op::Sentient(ops::Op::If {
            predicate: ops::CmpPredicate::Sge,
            lhs,
            rhs,
            yielded: Vec::new(),
            dbg_name: None,
            then_body,
            else_body,
        })
    }

    fn compare(predicate: ops::CmpPredicate, lhs: Val, rhs: Val) -> Op {
        Op::Sentient(ops::Op::If {
            predicate,
            lhs,
            rhs,
            yielded: Vec::new(),
            dbg_name: None,
            then_body: vec![yields(Vec::new())],
            else_body: Vec::new(),
        })
    }

    fn yields(results: Vec<Val>) -> Op {
        Op::Sentient(ops::Op::Yield { results })
    }

    /// A `sentient.load_and_store` reading `src_mutable_addr` and writing `dst_mutable_addr`.
    fn load_and_store(src_mutable_addr: Val, dst_mutable_addr: Val) -> Op {
        Op::Sentient(ops::Op::LoadAndStore {
            src: Val(90),
            dst: Val(91),
            src_mutable_addr,
            src_immutable_addr: Val(92),
            src_inc: Val(93),
            dst_mutable_addr,
            dst_immutable_addr: Val(94),
            dst_inc: Val(95),
            multicast_info: None,
            results: (Val(96), Val(97)),
            extent: ops::Extent::of(Elements(64), Bits(16)),
            stride: 1,
            rotate_val: None,
            shuffle_mode: ops::ShuffleMode::NoShuffle,
            src_reg: Reg {
                locale: RegType::Lrf,
                index: None,
            },
            dst_reg: Reg {
                locale: RegType::Lrf,
                index: None,
            },
            dir: None,
            is_ibr_write: false,
            dbg_name: None,
        })
    }

    /// The iter arg's readers move onto the init, the loop result's readers onto the yielded value.
    #[test]
    fn the_iter_args_become_their_inits_and_the_results_become_what_is_yielded() {
        let mut scope = vec![
            constant(0, Val(0)),
            for_loop(
                Val(1),
                Val(9),
                vec![carried(Val(0), Val(2), Val(3))],
                vec![add(Val(2), Val(2), Val(4)), yields(vec![Val(4)])],
            ),
            add(Val(3), Val(3), Val(5)),
        ];
        replace_iter_args_and_yield_results(&mut scope, 1);
        let Some(for_op) = ForOp::of(&scope[1]) else {
            panic!("the loop is still a sentient.for")
        };
        assert_eq!(for_op.body.first(), Some(&add(Val(0), Val(0), Val(4))));
        assert_eq!(scope[2], add(Val(4), Val(4), Val(5)));
    }

    /// ⛔ THE CASE THE REFUSAL EXISTS FOR: a constant-initialised iter arg used as a MUTABLE address
    /// blocks unrolling; the same value as an immutable address does not.
    #[test]
    fn a_constant_iter_arg_on_a_mutable_address_blocks_unrolling() {
        let blocked = vec![
            constant(4096, Val(0)),
            for_loop(
                Val(1),
                Val(9),
                vec![carried(Val(0), Val(2), Val(3))],
                vec![load_and_store(Val(2), Val(7)), yields(vec![Val(2)])],
            ),
        ];
        let Some(for_op) = ForOp::of(&blocked[1]) else {
            panic!("the loop is a sentient.for")
        };
        assert!(!is_ok_to_unroll(for_op, &blocked));

        let allowed = vec![
            constant(4096, Val(0)),
            for_loop(
                Val(1),
                Val(9),
                vec![carried(Val(0), Val(2), Val(3))],
                vec![load_and_store(Val(7), Val(8)), yields(vec![Val(2)])],
            ),
        ];
        let Some(for_op) = ForOp::of(&allowed[1]) else {
            panic!("the loop is a sentient.for")
        };
        assert!(is_ok_to_unroll(for_op, &allowed));
    }

    /// The if ops reach the canonicaliser, every loop gets an ibuff cost, and only the constant-bound
    /// loop becomes a candidate — reverse-BFS order, so the inner loop is visited first.
    #[test]
    fn cleanup_costs_every_loop_and_collects_only_the_constant_bound_ones() {
        let mut unit = vec![
            constant(8, Val(0)),
            for_loop(
                Val(1),
                Val(0),
                Vec::new(),
                vec![
                    if_op(Val(1), Val(0), vec![yields(Vec::new())], Vec::new()),
                    for_loop(Val(4), Val(5), Vec::new(), vec![yields(Vec::new())]),
                    yields(Vec::new()),
                ],
            ),
        ];
        let mut pass = LoopSplittingAndUnrolling::default();
        let mut tree = LoopForest::default();
        let mut ie = StatedEstimator {
            per_op: 7,
            ..StatedEstimator::default()
        };
        let mut canonicalizer = RecordingCanonicalizer::default();
        cleanup_and_recalculate(&mut pass, &mut tree, &mut ie, &mut canonicalizer, &mut unit);
        assert_eq!(canonicalizer.saw, vec![OpPath::at(&[(0, 1), (0, 0)])]);
        assert_eq!(ie.recalculated, 1);
        assert_eq!(
            tree.reverse_bfs(),
            vec![LoopNodeId(2), LoopNodeId(1), LoopNodeId(0)]
        );
        assert_eq!(
            pass.loopnode_to_ibuff,
            BTreeMap::from([
                (LoopNodeId(1), InstructionCount(7)),
                (LoopNodeId(2), InstructionCount(7))
            ])
        );
        // Only the outer loop's bound is a `sentient.scalar_constant`.
        assert_eq!(pass.unrolling_or_splitting_candidates, vec![LoopNodeId(1)]);
    }

    /// An if op comparing the induction variable, against one that does not.
    #[test]
    fn only_an_if_op_reading_the_induction_variable_counts() {
        let uses = vec![for_loop(
            Val(1),
            Val(0),
            Vec::new(),
            vec![
                compare(ops::CmpPredicate::Slt, Val(1), Val(6)),
                yields(Vec::new()),
            ],
        )];
        assert!(if_op_uses_iv(ForRef(Val(1)), &uses));

        let elsewhere = vec![for_loop(
            Val(1),
            Val(0),
            Vec::new(),
            vec![
                compare(ops::CmpPredicate::Slt, Val(5), Val(6)),
                yields(Vec::new()),
            ],
        )];
        assert!(!if_op_uses_iv(ForRef(Val(1)), &elsewhere));
    }

    /// An if op NESTED below the loop still counts; a loop without one does not.
    #[test]
    fn an_if_op_anywhere_inside_the_loop_counts() {
        let nested = vec![for_loop(
            Val(1),
            Val(0),
            Vec::new(),
            vec![
                for_loop(
                    Val(4),
                    Val(5),
                    Vec::new(),
                    vec![
                        compare(ops::CmpPredicate::Slt, Val(5), Val(6)),
                        yields(Vec::new()),
                    ],
                ),
                yields(Vec::new()),
            ],
        )];
        assert!(has_if_ops(ForRef(Val(1)), &nested));

        let plain = vec![for_loop(
            Val(1),
            Val(0),
            Vec::new(),
            vec![add(Val(5), Val(6), Val(7)), yields(Vec::new())],
        )];
        assert!(!has_if_ops(ForRef(Val(1)), &plain));
    }

    /// A constant LHS compares as written, a constant RHS compares the other way round, and neither
    /// side constant is `false`.
    #[test]
    fn the_predicate_is_evaluated_against_whichever_side_is_constant() {
        let scope = vec![
            constant(5, Val(0)),
            compare(ops::CmpPredicate::Sge, Val(0), Val(9)),
            compare(ops::CmpPredicate::Slt, Val(9), Val(0)),
            compare(ops::CmpPredicate::Sge, Val(8), Val(9)),
        ];
        // 5 >= 3
        assert!(evaluate_predicate(IterBound(3), &scope[1], &scope));
        // !(5 >= 7)
        assert!(!evaluate_predicate(IterBound(7), &scope[1], &scope));
        // 3 < 5, the bound on the left
        assert!(evaluate_predicate(IterBound(3), &scope[2], &scope));
        assert!(!evaluate_predicate(IterBound(3), &scope[3], &scope));
    }

    /// Bounds 4 and 3 agree, 2 and 1 agree: two ranges, each `(highest, lowest)`.
    #[test]
    fn equal_predicate_vectors_over_consecutive_bounds_become_one_range() {
        let bounds = BTreeMap::from([
            (IterBound(4), vec![true]),
            (IterBound(3), vec![true]),
            (IterBound(2), vec![false]),
            (IterBound(1), vec![false]),
        ]);
        let Some(bounds) = BoundPredVals::from_map(&bounds) else {
            panic!("the map holds four bounds")
        };
        assert_eq!(
            find_all_split_vals(&bounds),
            vec![
                SplitRange {
                    high: IterBound(4),
                    low: IterBound(3)
                },
                SplitRange {
                    high: IterBound(2),
                    low: IterBound(1)
                }
            ]
        );
    }

    /// A known-true predicate charges the op, its yield operands and the whole dead ELSE region.
    #[test]
    fn a_known_predicate_reduces_by_the_op_its_yields_and_the_dead_branch() {
        let op = if_op(
            Val(1),
            Val(0),
            vec![yields(vec![Val(7), Val(8)])],
            vec![
                constant(1, Val(9)),
                constant(2, Val(10)),
                yields(vec![Val(9), Val(10)]),
            ],
        );
        let Some(if_node) = IfNode::of(&op) else {
            panic!("the op is a sentient.if")
        };
        let at = OpPath::at(&[(0, 0)]);
        let node = CondNode::Branches(vec![CondNode::If {
            at: at.clone(),
            if_op: if_node,
            then_node: Box::new(CondNode::Branches(Vec::new())),
            else_node: None,
        }]);
        let mut ie = StatedEstimator::default();
        let mut cost_reduced = InstructionCount(0);
        compute_reduced_cost(
            Some(&node),
            &mut ie,
            &mut cost_reduced,
            &BTreeMap::from([(at.clone(), true)]),
        );
        // 1 for the op + 2 yield operands + the else region's 3 ops.
        assert_eq!(cost_reduced, InstructionCount(6));

        // ⛔ AN IF OP THAT IS NOT IN THE MAP COSTS NOTHING.
        let mut cost_reduced = InstructionCount(0);
        compute_reduced_cost(Some(&node), &mut ie, &mut cost_reduced, &BTreeMap::new());
        assert_eq!(cost_reduced, InstructionCount(0));
    }

    /// e318 — the body replaces the loop in place, the iter arg reads the init and the sibling that
    /// read the loop's result reads the yielded value.
    #[test]
    fn e318_hoists_the_body_over_the_loop_and_rewires_it() {
        let mut scope = vec![
            constant(0, Val(0)),
            for_loop(
                Val(1),
                Val(9),
                vec![carried(Val(0), Val(2), Val(3))],
                vec![add(Val(2), Val(2), Val(4)), yields(vec![Val(4)])],
            ),
            add(Val(3), Val(3), Val(5)),
        ];
        promote_for_loop_body_and_delete(&mut scope, 1);
        assert_eq!(
            scope,
            vec![
                constant(0, Val(0)),
                add(Val(0), Val(0), Val(4)),
                add(Val(4), Val(4), Val(5)),
            ]
        );
    }

    /// e319 — one predicate value per if op at bound 5, including a `false` for the if op whose
    /// operands are both non-constant.
    #[test]
    fn e319_answers_one_predicate_value_per_if_op() {
        let scope = vec![constant(4, Val(0))];
        let if_list = [
            compare(ops::CmpPredicate::Sge, Val(0), Val(2)),
            compare(ops::CmpPredicate::Slt, Val(0), Val(2)),
            compare(ops::CmpPredicate::Eq, Val(7), Val(8)),
        ];
        let if_list: Vec<&Op> = if_list.iter().collect();
        assert_eq!(
            get_all_true_index_of_if_ops_list(IterBound(5), &if_list, &scope),
            vec![false, true, false]
        );
    }

    /// e320 — two ranges give four subsets, empty one first, in the recursion's order.
    #[test]
    fn e320_enumerates_every_subset_starting_with_the_empty_one() {
        let high = SplitRange {
            high: IterBound(9),
            low: IterBound(5),
        };
        let low = SplitRange {
            high: IterBound(4),
            low: IterBound(1),
        };
        let mut res = Vec::new();
        subsets_impl(&[high, low], &mut res, &mut Vec::new(), 0);
        assert_eq!(
            res,
            vec![Vec::new(), vec![high], vec![high, low], vec![low]]
        );
    }

    /// The loop `op` is, which every one of these collected by matching on.
    fn for_op_of(op: &Op) -> ForOp<'_> {
        ForOp::of(op).expect("collected by matching `sentient.for`")
    }

    /// e446 — the sort is by DECREASING SIZE and it is stable, so ties keep [`subsets_impl`]'s own
    /// enumeration order and the empty subset lands last.
    #[test]
    fn e446_orders_every_subset_longest_first_and_keeps_the_ties_stable() {
        let range = |high: i64, low: i64| SplitRange {
            high: IterBound(high),
            low: IterBound(low),
        };
        let (first, second, third) = (range(9, 8), range(7, 5), range(4, 1));
        assert_eq!(
            subsets(&[first, second, third]),
            vec![
                vec![first, second, third],
                vec![first, second],
                vec![first, third],
                vec![second, third],
                vec![first],
                vec![second],
                vec![third],
                Vec::new(),
            ]
        );
    }

    /// e447 — a loop over `%iv >= 2` splits into the bounds that behave alike: bounds 3 and 2 take
    /// the `then` branch, bound 1 the `else`, so two loops stand in its place with no conditional
    /// left, chained through the iter args, and only the shifted one carries the `iv + (low - 1)`.
    #[test]
    fn e447_replaces_the_loop_with_one_folded_loop_per_split_range() {
        let mut values = Values::default();
        let bound_c = values.mint();
        let two = values.mint();
        let init_c = values.mint();
        let iv = values.mint();
        let arg = values.mint();
        let result = values.mint();
        let if_result = values.mint();
        let then_val = values.mint();
        let else_val = values.mint();
        let reader = values.mint();

        let mut scope = vec![
            constant(3, bound_c),
            constant(2, two),
            constant(7, init_c),
            for_loop(
                iv,
                bound_c,
                vec![carried(init_c, arg, result)],
                vec![
                    Op::Sentient(ops::Op::If {
                        predicate: ops::CmpPredicate::Sge,
                        lhs: iv,
                        rhs: two,
                        yielded: vec![Yielded {
                            result: if_result,
                            reg: Reg {
                                locale: RegType::Lrf,
                                index: None,
                            },
                            element_size: None,
                        }],
                        dbg_name: None,
                        // The `then` branch reads the induction variable and the `else` branch does
                        // not, which is what decides whether each clone needs the shift.
                        then_body: vec![add(iv, two, then_val), yields(vec![then_val])],
                        else_body: vec![add(arg, bound_c, else_val), yields(vec![else_val])],
                    }),
                    yields(vec![if_result]),
                ],
            ),
            add(result, bound_c, reader),
        ];

        assert_eq!(
            split_loop_n_way(&mut scope, 3, &mut values),
            LoopSplit::Split
        );

        // The original loop is gone and no `sentient.if` survives: each clone's predicate was
        // decided by its own range.
        let mut for_ops = Vec::new();
        let mut if_ops = 0;
        for_each_op(&scope, &mut |op| match op {
            Op::Sentient(ops::Op::If { .. }) => if_ops += 1,
            Op::Sentient(ops::Op::For { .. }) => for_ops.push(op),
            _ => {}
        });
        assert_eq!(if_ops, 0);
        assert_eq!(for_ops.len(), 2);

        let bounds: Vec<Option<i64>> = for_ops
            .iter()
            .map(|op| constant_value(for_op_of(op).bound, &scope))
            .collect();
        // `kv.first - kv.second + 1` for the ranges `(3, 2)` and `(1, 1)`.
        assert_eq!(bounds, vec![Some(2), Some(1)]);

        let split = for_op_of(for_ops[0]);
        let tail = for_op_of(for_ops[1]);
        // The first clone keeps the original inits; the second reads the first's results.
        assert_eq!(split.carried[0].init, init_c);
        assert_eq!(tail.carried[0].init, split.carried[0].result);
        // `%sum = %iv + 1` heads the shifted clone, and the folded `then` branch reads `%sum`.
        let Op::Sentient(ops::Op::ScalarAdd {
            lhs,
            rhs,
            result: sum,
            ..
        }) = &split.body[0]
        else {
            panic!("the shifted clone must open with the induction variable's offset");
        };
        assert_eq!(*lhs, split.iv);
        assert_eq!(constant_value(*rhs, &scope), Some(1));
        assert!(
            matches!(&split.body[1], Op::Sentient(ops::Op::ScalarAdd { lhs, .. }) if lhs == sum)
        );
        assert_eq!(split.body.len(), 3);
        // The `else` branch never read the variable, so that clone gets no offset at all.
        assert_eq!(tail.body.len(), 2);
        assert!(matches!(
            &tail.body[0],
            Op::Sentient(ops::Op::ScalarAdd { lhs, .. }) if *lhs == tail.carried[0].arg
        ));
        // The original loop's readers moved onto the LAST clone alone.
        assert!(matches!(
            scope.last(),
            Some(Op::Sentient(ops::Op::ScalarAdd { lhs, .. })) if *lhs == tail.carried[0].result
        ));
    }
    /// e447 — an if op on the induction variable nested in an `affine.for` is folded like any other
    /// and takes its OWN `predvals` slot: `if_list1` is a `walk`, so skipping it would leave it
    /// standing AND read the next if op's branch off the wrong bound.
    #[test]
    fn e447_folds_the_if_ops_inside_an_affine_for_at_their_own_predval_slots() {
        use crate::islands::sentient::dialects::{AffineFor, affine};

        let (iv, two) = (Val(1), Val(2));
        let scope = vec![constant(2, two)];
        let block = vec![
            Op::AffineFor(AffineFor {
                iv: Val(3),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: vec![if_op(iv, two, vec![nop("nested-then")], vec![nop("nested-else")])],
                dbg_name: None,
            }),
            if_op(iv, two, vec![nop("after-then")], vec![nop("after-else")]),
        ];

        let mut index = 0;
        let mut rewires = Vec::new();
        let out = fold_if_ops_on_iv(
            block,
            iv,
            &[scope.as_slice()],
            &[true, false],
            &mut index,
            &mut rewires,
        );

        // Two if ops, two slots: the nested one is FIRST in pre-order.
        assert_eq!(index, 2);
        let Op::AffineFor(time_loop) = &out[0] else {
            panic!("the time loop stays, holding only what its if op resolved to")
        };
        assert_eq!(time_loop.body, vec![nop("nested-then")]);
        // Slot 1 is `false`, so the following if op keeps its `else` — reading slot 0 would take the
        // `then` instead.
        assert_eq!(out[1], nop("after-else"));
        assert_eq!(out.len(), 2);
    }

    /// A `sentient.nop` carrying a debug name — the one op a body can hold that `updateDbgName`
    /// writes to without also being a loop.
    fn nop(dbg_name: &str) -> Op {
        Op::Sentient(ops::Op::Nop {
            dbg_name: Some(dbg_name.to_string()),
        })
    }

    /// e513 — a two-iteration loop comes out as its ORIGINAL body, named `i=2`, followed by the clone
    /// for `i=1` reading the original's yielded value and its own induction-variable constant.
    #[test]
    fn e513_lays_the_iterations_out_flat_with_the_original_body_first() {
        let mut values = Values::default();
        let bound_c = values.mint();
        let init_c = values.mint();
        let iv = values.mint();
        let arg = values.mint();
        let result = values.mint();
        let sum = values.mint();
        let reader = values.mint();

        let mut scope = vec![
            constant(2, bound_c),
            constant(7, init_c),
            for_loop(
                iv,
                bound_c,
                vec![carried(init_c, arg, result)],
                vec![nop("n"), add(arg, iv, sum), yields(vec![sum])],
            ),
            add(result, bound_c, reader),
        ];

        assert_eq!(unroll_loop(&mut scope, 2, &mut values), LoopUnroll::Done);

        let mut adds = Vec::new();
        let mut names = Vec::new();
        let mut for_ops = 0;
        for_each_op(&scope, &mut |op| match op {
            Op::Sentient(ops::Op::ScalarAdd {
                lhs, rhs, result, ..
            }) => adds.push((*lhs, *rhs, *result)),
            Op::Sentient(ops::Op::Nop { dbg_name: Some(name) }) => names.push(name.clone()),
            Op::Sentient(ops::Op::For { .. }) => for_ops += 1,
            _ => {}
        });

        assert_eq!(for_ops, 0);
        assert_eq!(names, vec!["LU(n, i=2)".to_string(), "LU(n, i=1)".to_string()]);
        assert_eq!(adds.len(), 3);
        // The original body reads the loop's INIT and — the trap — the loop's BOUND for its variable.
        assert_eq!(adds[0], (init_c, bound_c, sum));
        // The clone reads what the original yielded, and a constant of its own iteration.
        assert_eq!(adds[1].0, sum);
        assert_eq!(constant_value(adds[1].1, &scope), Some(1));
        // The loop's reader moved onto the LAST iteration's result.
        assert_eq!(adds[2], (adds[1].2, bound_c, reader));
    }
    /// A conditional tree the test STATES: one if-node over the loop's own `sentient.if`.
    /// ⛔ `dcc::TransformationConditionalTree` is outside the campaign, so what e514 owns is what it
    /// does with a tree, not the tree.
    #[derive(Debug, Default)]
    struct StatedTree {
        at: OpPath,
    }

    impl ConditionalTree for StatedTree {
        fn root<'a>(&mut self, for_op: &'a Op) -> CondNode<'a> {
            let Op::Sentient(ops::Op::For { body, .. }) = for_op else {
                return CondNode::Branches(Vec::new());
            };
            let Some(if_op) = body.first().and_then(IfNode::of) else {
                return CondNode::Branches(Vec::new());
            };
            CondNode::Branches(vec![CondNode::If {
                at: self.at.clone(),
                if_op,
                then_node: Box::new(CondNode::Branches(Vec::new())),
                else_node: None,
            }])
        }
    }

    /// e514 — a bound-3 loop whose one conditional is true for iterations 3 and 2 splits two ways, and
    /// the price is ONE extra copy of the loop less what each range's dead branch saves.
    #[test]
    fn e514_prices_the_full_split_against_what_the_folded_conditionals_save() {
        let scope = vec![
            constant(3, Val(0)),
            constant(2, Val(1)),
            for_loop(
                Val(2),
                Val(0),
                Vec::new(),
                vec![if_op(
                    Val(2),
                    Val(1),
                    vec![yields(vec![Val(3)])],
                    vec![constant(5, Val(4)), yields(vec![Val(4)])],
                )],
            ),
        ];
        let mut ie = StatedEstimator {
            per_op: 10,
            ibuff_space: 4,
            ..StatedEstimator::default()
        };
        let mut tree = StatedTree {
            at: OpPath::at(&[(0, 0)]),
        };

        let answer = find_optimal_n_way_splits(&scope, 2, &mut ie, &mut tree, &scope);

        assert_eq!(
            answer,
            NWaySplits {
                index: 0,
                nway_split: 2,
                if_ops_on_iv: 1,
                // `10 * (2 - 1)` for the extra loop, less `(1 + 1 + 2)` for the range that folds the
                // else away and `(1 + 1 + 1)` for the one that folds the then away.
                ibuff_cost: InstructionCount(3),
            }
        );
        // ⛔ THE IBUFF IS ASKED FOR AND THE ANSWER ONLY LOGGED: 3 fits in 4, and a cost that did not
        // fit would come back the same.
        assert_eq!(ie.ibuff_asks, 1);
    }
}
