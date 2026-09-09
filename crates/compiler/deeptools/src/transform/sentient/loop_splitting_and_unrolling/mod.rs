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
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{Op, Val, defining_op, replace_all_uses_with};

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

// crustify:todo: e446_subsets
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:635  (12 body lines, level 2)
//   original  : static std::vector<std::vector<std::pair<int, int>>> subsets( std::vector<std::pair<int, int>> &split_vals)
//   calls     : e252_size, e320_subsetsImpl

// crustify:todo: e447_splitLoopNWay
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:858  (129 body lines, level 2)
//   original  : LogicalResult LoopSplittingAndUnrollingPass::splitLoopNWay( dcc::LoopNode *n, OpBuilder &const_builder)
//   calls     : e091_findAllSplitVals, e252_size, e319_getAllTrueIndexOfIfOpsList

// crustify:todo: e513_unrollLoop
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:207  (86 body lines, level 3)
//   original  : LogicalResult LoopSplittingAndUnrollingPass::unrollLoop( dcc::LoopNode *n, OpBuilder &const_builder)
//   calls     : e086_isOkToUnroll, e252_size, e318_promoteForLoopBodyAndDelete, e423_lookup

// crustify:todo: e514_findOptimalNWaySplits
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:691  (162 body lines, level 3)
//   original  : std::tuple<int, int, int, int> LoopSplittingAndUnrollingPass::findOptimalNWaySplits( dcc::LoopNode *n, InstructionEstimatorImpl &ie, dataflow::ProgramUnitOp &unit_op)
//   calls     : e091_findAllSplitVals, e092_computeReducedCost, e252_size, e319_getAllTrueIndexOfIfOpsList, e422_insert

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
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType};

    /// AN ESTIMATOR THAT ANSWERS WHAT THE TEST SAYS. ⛔ `InstructionEstimatorImpl` is out of campaign
    /// scope, so a test STATES its counts rather than deriving them — the effect under test is what
    /// the unit does with a count, not the count.
    #[derive(Debug, Default)]
    struct StatedEstimator {
        /// How many times `recalculate` was asked for.
        recalculated: usize,
        /// What every op costs.
        per_op: i32,
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
            recalculated: 0,
            per_op: 7,
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
}
