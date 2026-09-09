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

//! `LoopAbsorption.cpp` — 10 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e066_LoopAbsorptionManager` | 066 | 0 | 6 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:37` |
//! | `e067_updateLoopBound` | 067 | 0 | 48 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:58` |
//! | `e068_doChildrenLeftSiblingSubtreeHeightsMatch` | 068 | 0 | 12 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:122` |
//! | `e069_doChildrenRightSiblingSubtreeHeightsMatch` | 069 | 0 | 12 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:269` |
//! | `e070_removeNodesFromWorklistAndTree` | 070 | 0 | 7 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:410` |
//! | `e309_absorbIntoAnchorFromLeft` | 309 | 1 | 15 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:236` |
//! | `e310_absorbIntoAnchorFromRight` | 310 | 1 | 17 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:387` |
//! | `e506_canAbsorbToTheLeft` | 506 | 3 | 99 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:135` |
//! | `e507_canAbsorbToTheRight` | 507 | 3 | 103 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:282` |
//! | `e562_absorptionAnalysis` | 562 | 4 | 65 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:424` |

// ⛔ NOTHING IN THIS MODULE HAS A CALLER UNTIL `e600_runLoopAbsorption` (level 5) LANDS, and CI runs
// clippy with `-D warnings`. ⭐ REMOVE THIS WITH e600.
#![allow(dead_code)]

use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    BlockArgEquivalence, EquivalenceTag, OperationEquivalence, SubregionCompare,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient::RegType;
use crate::islands::sentient::dialects::{self as dialects, Op, Val, sentient, uniform};
use crate::transform::sentient::ForRef;
use crate::transform::sentient::loop_tree::{LoopNodeId, LoopTree};

/// WHERE THE PASS MAY INSERT — the `dataflow.program_unit` the anchor loop sits in.
///
/// ⛔⛔ `DT_CHECK_MSG(unit, "expected loop to appear in dataflow.program_unit")`
/// (`LoopAbsorption.cpp:72`) IS THIS TYPE. The check cannot be asked at run time because naming the
/// unit is what builds one. ⭐ `preamble` IS THE REGION HOLDING the unit, which is where
/// `OpBuilder const_builder(dyn_cast<dataflow::ProgramUnitOp>(*unit))` (`:73`) inserts: an
/// `OpBuilder` built from an operation inserts BEFORE it, not into it — contrast
/// `LoopMerging.cpp:205`, which passes `.getRegion()` and so starts inside.
#[derive(Debug)]
pub(crate) struct UnitSite<'u> {
    /// `crate::islands::sentient::Program::preamble` — the ops before the unit.
    pub preamble: &'u mut Vec<Op>,
    /// `crate::islands::sentient::ProgramUnit::body` — the unit's own block.
    pub body: &'u mut Vec<Op>,
}

/// `anchor_bb_`, `body_bb_` and `terminator_` (`LoopAbsorption.hpp:36-40`) — DERIVED, NOT STORED: a
/// raw `Block *` into the IR being rewritten is the *positioning* mechanism a port supplies itself.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AnchorPositions<'s> {
    /// `anchor_bb_` — the block containing the anchor loop.
    pub block: &'s [Op],
    /// `body_bb_` — the anchor loop's body block.
    pub body: &'s [Op],
    /// `terminator_` — the `sentient.yield` closing that body; `None` for a body with no ops.
    pub terminator: Option<&'s sentient::Op>,
}

/// HOW MUCH `updateLoopBound` ADDS TO THE ANCHOR'S TRIP COUNT — its `int64_t increment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BoundIncrement(pub i64);

/// `LoopAbsorptionManager` (`LoopAbsorption.hpp:22`) — one anchor loop and the absorptions into it.
///
/// ⚠️ `iter_arg_num_to_new_start_val_` and `to_update_` (`:43-51`) are NOT fields yet: they are what
/// `e506_canAbsorbToTheLeft`/`e507_canAbsorbToTheRight` populate, and they arrive with those units.
#[derive(Debug)]
pub(crate) struct LoopAbsorptionManager<'a> {
    /// `oe_` — already ported, and already tagged `"loop-absorption"`.
    oe: OperationEquivalence,
    /// `tree_` — ⭐ `LoopTree<true>` because this pass is the one that asks for subtree heights.
    tree: &'a mut LoopTree<true>,
    /// `sorted_worklist_`.
    sorted_worklist: &'a mut Vec<LoopNodeId>,
    /// `anchor_`.
    anchor: LoopNodeId,
    /// `anchor_for_op_`.
    anchor_for_op: ForRef,
    /// The unit the anchor lives in — what the three `Block *` fields point into.
    site: UnitSite<'a>,
}

impl<'a> LoopAbsorptionManager<'a> {
    /// Replaces: e066_LoopAbsorptionManager
    ///
    /// The anchor is the worklist's LAST node, and the manager is its loop plus the site it sits in.
    ///
    /// ⛔ `None` IS THE CONSTRUCTOR'S THREE UNSTATED PRECONDITIONS: `sorted_worklist.back()` on an
    /// empty vector is undefined; `anchor_->getOpAs<sentient::ForOp>()` `DT_CHECK`s on the synthetic
    /// root; and the loop must be in `site`. ⭐ ASKING THEM HERE IS WHAT MAKES THEM UNASKABLE LATER.
    pub(crate) fn new(
        sorted_worklist: &'a mut Vec<LoopNodeId>,
        tree: &'a mut LoopTree<true>,
        site: UnitSite<'a>,
    ) -> Option<Self> {
        // `anchor_(sorted_worklist.back())` (`LoopAbsorption.cpp:41`).
        let anchor = *sorted_worklist.last()?;
        // `anchor_for_op_ = anchor_->getOpAs<sentient::ForOp>();` (`:47`).
        let anchor_for_op = tree.loop_of(anchor)?;
        // `anchor_bb_`/`body_bb_`/`terminator_` (`:48-50`) — derived, so all this needs is that the
        // loop is findable at all.
        positions_of(anchor_for_op, site.body)?;
        Some(Self {
            // `oe_(dcc::OperationEquivalence(nullptr, nullptr, DEBUG_TYPE, true, false, true))`
            // (`:42-45`) — `use_equiv_classes` is `tagged`'s default.
            oe: OperationEquivalence::tagged(
                EquivalenceTag::LoopAbsorption,
                SubregionCompare::Recursive,
                BlockArgEquivalence::SameOwnerAndIndex,
            ),
            tree,
            sorted_worklist,
            anchor,
            anchor_for_op,
            site,
        })
    }

    /// The constructor's last three statements (`LoopAbsorption.cpp:48-50`).
    ///
    /// ⛔ `None` IS UNREACHABLE AFTER [`Self::new`], which found the anchor loop in this site.
    pub(crate) fn positions(&self) -> Option<AnchorPositions<'_>> {
        positions_of(self.anchor_for_op, self.site.body)
    }

    /// Replaces: e067_updateLoopBound
    ///
    /// Raise the anchor's bound by `increment`: fold it when it is a constant, otherwise mint a
    /// `sentient.scalar_add` beside whatever defines it, then drop a bound left with no uses.
    ///
    /// ⛔ THE INCREMENT CONSTANT IS MINTED BEFORE THE BRANCH AND ERASED AGAIN IN THE FOLDED ARM
    /// (`:73-80`) — it is the operand of a `scalar_add` that only two of the four arms emit.
    /// ⛔ TRAP: `bound.getType()` is always `index` — `Index:$bound` (`SentientOps.td:47`).
    pub(crate) fn update_loop_bound(&mut self, vals: &mut Values, increment: BoundIncrement) {
        // `Value bound = anchor_for_op_.getBound();` (`:59`).
        let Some(bound) = bound_of(self.anchor_for_op, self.site.body) else {
            return;
        };
        // `dyn_cast_or_null<sentient::ConstantOp>(bound.getDefiningOp())` (`:60-61`) and its
        // `uniform::QueryMapOp` twin (`:62-63`), both read before anything is inserted.
        let definer = dialects::defining_op(bound, self.site.body);
        let const_bound = match definer {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => Some(*value),
            _ => None,
        };
        let query_map = matches!(definer, Some(Op::Uniform(uniform::Op::QueryMap { .. })));
        let has_definer = definer.is_some();
        let bound_type = ScalarTy::Index;
        // `auto const_increm = sentient::ConstantOp::create(const_builder, …, increment);` (`:73-74`).
        let const_increm = vals.mint();
        self.site
            .preamble
            .push(constant(const_increm, increment.0, bound_type));

        if let Some(value) = const_bound {
            // `setBound(ConstantOp::create(const_builder, loc, bound_type,
            //  const_bound.getValue() + increment))` (`:77-78`).
            let folded = vals.mint();
            self.site.preamble.push(constant(
                folded,
                value.saturating_add(increment.0),
                bound_type,
            ));
            set_bound(self.anchor_for_op, self.site.body, folded);
            // `const_increm.erase();` (`:79`) — the add was never emitted, so nothing reads it.
            dialects::erase_defining_op(&mut *self.site.preamble, const_increm);
        } else if query_map {
            // `is_query_map_constant` (`:64-66`) chooses between two `updateBoundToValuePlusMap`
            // calls (`:81-88`); the predicate AND the rewrite are both in
            // `Transform/Sentient/Analyses/Utils`, which is outside this campaign.
            todo!(
                "dcc::utils::isConstant / dcc::utils::updateBoundToValuePlusMap — out of campaign scope"
            )
        } else {
            // `sentient::AddOp::create(builder, loc, bound_type, bound, const_increm)` (`:106-107`),
            // placed *"as to avoid preventing absorption opportunities"* (`:90-93`).
            let sum = vals.mint();
            let add = Op::Sentient(sentient::Op::ScalarAdd {
                lhs: bound,
                rhs: const_increm,
                result: sum,
                reg: None,
                ty: bound_type,
            });
            if has_definer {
                // `builder.setInsertionPointAfter(bound.getDefiningOp());` (`:97`).
                insert_after_definition(&mut *self.site.body, bound, add);
            } else {
                // `setInsertionPointToStart(cast<BlockArgument>(bound).getOwner());` (`:99-101`).
                insert_at_start_of_binder(&mut *self.site.body, bound, add);
            }
            set_bound(self.anchor_for_op, self.site.body, sum);
        }

        // `if (bound.getDefiningOp() && bound.getDefiningOp()->use_empty())
        //  bound.getDefiningOp()->erase();` (`:110-111`).
        if has_definer && !is_used(&mut *self.site.body, bound) {
            dialects::erase_defining_op(&mut *self.site.body, bound);
        }
    }

    /// Replaces: e068_doChildrenLeftSiblingSubtreeHeightsMatch
    ///
    /// Walk the anchor's children right-to-left against its own left siblings: every child must have
    /// a sibling of the same subtree height, or nothing to the left can be absorbed.
    ///
    /// ⛔ THE HEIGHTS ARE THE ONES `compute` MEASURED, deliberately not refreshed after an absorption
    /// removed a node — see [`LoopTree::subtree_height_of`].
    pub(crate) fn do_children_left_sibling_subtree_heights_match(&self) -> bool {
        let mut next_child = self.tree.last_child(self.anchor);
        let mut next_sibling = self.tree.prev_sibling(self.anchor);
        while let Some(child) = next_child {
            // `if (!next_sibling || getSubtreeHeightOf(next_child) != getSubtreeHeightOf(next_sibling))`
            // (`:126-129`) — the anchor ran out of siblings before it ran out of children.
            let Some(sibling) = next_sibling else {
                return false;
            };
            if self.tree.subtree_height_of(child) != self.tree.subtree_height_of(sibling) {
                return false;
            }
            next_child = self.tree.prev_sibling(child);
            next_sibling = self.tree.prev_sibling(sibling);
        }
        true
    }

    /// Replaces: e069_doChildrenRightSiblingSubtreeHeightsMatch
    ///
    /// [`Self::do_children_left_sibling_subtree_heights_match`] mirrored: the anchor's children
    /// left-to-right against its own right siblings.
    pub(crate) fn do_children_right_sibling_subtree_heights_match(&self) -> bool {
        let mut next_child = self.tree.first_child(self.anchor);
        let mut next_sibling = self.tree.next_sibling(self.anchor);
        while let Some(child) = next_child {
            let Some(sibling) = next_sibling else {
                return false;
            };
            if self.tree.subtree_height_of(child) != self.tree.subtree_height_of(sibling) {
                return false;
            }
            next_child = self.tree.next_sibling(child);
            next_sibling = self.tree.next_sibling(sibling);
        }
        true
    }

    /// Replaces: e070_removeNodesFromWorklistAndTree
    ///
    /// Drop each absorbed loop from the worklist and unlink its node from the tree.
    ///
    /// ⛔ THE HEIGHT MAP KEEPS ITS ENTRY, as the reference's does: `remove` frees the node while
    /// `loop_node_to_subtree_height_` still keys on the freed pointer, and e068/e069 run again after
    /// this. See [`crate::bridges::dataflow_ir_to_sentient::vc_loop_mask_tree::OperationTreeBase::remove`].
    pub(crate) fn remove_nodes_from_worklist_and_tree(&mut self, nodes_to_delete: &[LoopNodeId]) {
        for n in nodes_to_delete {
            // `llvm::erase_if(sorted_worklist_, [n](LoopNode *node) { return node == n; })` (`:412-413`).
            self.sorted_worklist.retain(|node| node != n);
            // `tree_.remove(n);` (`:414`).
            self.tree.remove(*n);
        }
    }
}

/// `sentient.scalar_constant` holding `value` — the op both `ConstantOp::create` calls mint.
fn constant(result: Val, value: i64, ty: ScalarTy) -> Op {
    Op::Sentient(sentient::Op::ScalarConstant {
        value,
        result,
        reg_locale: RegType::Imm,
        ty,
        is_symbol: false,
    })
}

/// The sentient-typed regions of one op — where a `sentient.for` or a `sentient.scalar_add` can sit.
///
/// ⛔ A SHARED-DIALECT OP'S REGION HOLDS THE RUNG BELOW (`Vec<lower::Op>`) and can hold neither.
fn regions_of(op: &Op) -> Vec<&[Op]> {
    match op {
        Op::Sentient(inner) => sentient::regions(inner),
        Op::AffineFor(loop_op) => vec![loop_op.body.as_slice()],
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => Vec::new(),
    }
}

/// [`regions_of`] exclusively.
fn regions_mut_of(op: &mut Op) -> Vec<&mut Vec<Op>> {
    match op {
        Op::Sentient(inner) => sentient::regions_mut(inner),
        Op::AffineFor(loop_op) => vec![&mut loop_op.body],
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => Vec::new(),
    }
}

/// `anchor_for_op_->getBlock()`, `&anchor_for_op_.getRegion().front()` and
/// `body_bb_->getTerminator()` in one lookup (`LoopAbsorption.cpp:48-50`).
fn positions_of(for_op: ForRef, scope: &[Op]) -> Option<AnchorPositions<'_>> {
    for op in scope {
        if let Op::Sentient(sentient::Op::For { iv, body, .. }) = op
            && *iv == for_op.0
        {
            return Some(AnchorPositions {
                block: scope,
                body: body.as_slice(),
                terminator: match body.last() {
                    Some(Op::Sentient(last)) => Some(last),
                    _ => None,
                },
            });
        }
        for region in regions_of(op) {
            if let Some(found) = positions_of(for_op, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `anchor_for_op_.getBound()` — `$bound`, the loop's one non-variadic operand.
fn bound_of(for_op: ForRef, scope: &[Op]) -> Option<Val> {
    for op in scope {
        if let Op::Sentient(sentient::Op::For { iv, bound, .. }) = op
            && *iv == for_op.0
        {
            return Some(*bound);
        }
        for region in regions_of(op) {
            if let Some(found) = bound_of(for_op, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `anchor_for_op_.setBound(v)` — the setter half of [`bound_of`].
fn set_bound(for_op: ForRef, scope: &mut [Op], to: Val) {
    for op in scope.iter_mut() {
        if let Op::Sentient(sentient::Op::For { iv, bound, .. }) = op
            && *iv == for_op.0
        {
            *bound = to;
            return;
        }
        for region in regions_mut_of(op) {
            set_bound(for_op, region, to);
        }
    }
}

/// `OpBuilder::setInsertionPointAfter(val.getDefiningOp())`.
fn insert_after_definition(scope: &mut Vec<Op>, val: Val, op: Op) {
    let mut pending = Some(op);
    insert_after(scope, val, &mut pending);
}

/// One step of [`insert_after_definition`]; `pending` is taken once the definition is found.
fn insert_after(scope: &mut Vec<Op>, val: Val, pending: &mut Option<Op>) {
    if let Some(at) = scope
        .iter()
        .position(|op| dialects::results(op).contains(&val))
    {
        if let Some(op) = pending.take() {
            scope.insert(at + 1, op);
        }
        return;
    }
    for op in scope.iter_mut() {
        for region in regions_mut_of(op) {
            insert_after(region, val, pending);
        }
    }
}

/// `OpBuilder::setInsertionPointToStart(cast<BlockArgument>(val).getOwner())`.
///
/// ⭐ FALLING BACK TO `scope` ITSELF IS THE UNIT'S OWN ARGUMENT: nothing in the unit binds it, and the
/// block that does is the unit's, which is this vector.
fn insert_at_start_of_binder(scope: &mut Vec<Op>, val: Val, op: Op) {
    let mut pending = Some(op);
    insert_at_binder(scope, val, &mut pending);
    if let Some(op) = pending {
        scope.insert(0, op);
    }
}

/// One step of [`insert_at_start_of_binder`].
fn insert_at_binder(scope: &mut Vec<Op>, val: Val, pending: &mut Option<Op>) {
    for op in scope.iter_mut() {
        let binds = match &op {
            Op::Sentient(inner) => sentient::block_args(inner).contains(&val),
            _ => false,
        };
        let mut regions = regions_mut_of(op);
        if binds {
            if let Some(region) = regions.first_mut()
                && let Some(op) = pending.take()
            {
                region.insert(0, op);
            }
            return;
        }
        for region in regions {
            insert_at_binder(region, val, pending);
            if pending.is_none() {
                return;
            }
        }
    }
}

/// `Value::use_empty()`, negated — whether anything in `scope` still reads `val`.
///
/// ⛔ TAKES `&mut` BECAUSE THIS RUNG HAS NO IMMUTABLE OPERAND READER: `sentient::operands_mut` is the
/// only accessor there is, and an immutable twin would be a second description of one op's operand
/// list. Nothing is written.
fn is_used(scope: &mut [Op], val: Val) -> bool {
    for op in scope.iter_mut() {
        // ⭐ `&mut *op` REBORROWS so the shared-dialect arm's binding ends with the match.
        let reads = match &mut *op {
            Op::Sentient(inner) => sentient::operands_mut(inner).into_iter().any(|o| *o == val),
            Op::AffineFor(loop_op) => {
                loop_op.carried.iter().any(|carried| carried.init == val)
                    || [&loop_op.lo, &loop_op.hi]
                        .iter()
                        .any(|bound| **bound == lower::affine::Bound::Val(val))
            }
            // ⛔ A SHARED-DIALECT OP IS READ THROUGH ITS LOWER-RUNG FORM, regions included.
            other => dialects::lowered(other)
                .is_some_and(|low| !lower::uses(val, core::slice::from_ref(&low)).is_empty()),
        };
        if reads {
            return true;
        }
        for region in regions_mut_of(op) {
            if is_used(region, val) {
                return true;
            }
        }
    }
    false
}

// crustify:todo: e309_absorbIntoAnchorFromLeft
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:236  (15 body lines, level 1)
//   original  : void LoopAbsorptionManager::absorbIntoAnchorFromLeft()
//   calls     : e252_size

// crustify:todo: e310_absorbIntoAnchorFromRight
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:387  (17 body lines, level 1)
//   original  : void LoopAbsorptionManager::absorbIntoAnchorFromRight()
//   calls     : e252_size

// crustify:todo: e506_canAbsorbToTheLeft
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:135  (99 body lines, level 3)
//   original  : bool LoopAbsorptionManager::canAbsorbToTheLeft( std::vector<dcc::LoopNode *> &nodes_to_delete)
//   calls     : e068_doChildrenLeftSiblingSubtreeHeightsMatch, e422_insert

// crustify:todo: e507_canAbsorbToTheRight
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:282  (103 body lines, level 3)
//   original  : bool LoopAbsorptionManager::canAbsorbToTheRight( std::vector<dcc::LoopNode *> &nodes_to_delete)
//   calls     : e069_doChildrenRightSiblingSubtreeHeightsMatch, e422_insert

// crustify:todo: e562_absorptionAnalysis
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:424  (65 body lines, level 4)
//   original  : void LoopAbsorptionManager::absorptionAnalysis()
//   calls     : e067_updateLoopBound, e070_removeNodesFromWorklistAndTree, e309_absorbIntoAnchorFromLeft, e310_absorbIntoAnchorFromRight, e506_canAbsorbToTheLeft, e507_canAbsorbToTheRight

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// One `sentient.for` with no carried values.
    fn for_op(iv: Val, bound: Val, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

    /// A unit body of top-level loops, `children[i]` leaf children each, `children[anchor]` being the
    /// anchor's — so a childless sibling has subtree height 1 and one with children has 2.
    fn scenario(children: &[usize], anchor: usize) -> (Vec<Op>, Val) {
        let mut vals = Values::default();
        let bound = vals.mint();
        let mut body = Vec::new();
        let mut anchor_iv = bound;
        for (i, count) in children.iter().enumerate() {
            let iv = vals.mint();
            let kids = (0..*count)
                .map(|_| for_op(vals.mint(), bound, Vec::new()))
                .collect();
            if i == anchor {
                anchor_iv = iv;
            }
            body.push(for_op(iv, bound, kids));
        }
        (body, anchor_iv)
    }

    /// e068 over one [`scenario`].
    fn left_matches(children: &[usize], anchor: usize) -> bool {
        let (mut body, iv) = scenario(children, anchor);
        let mut tree = LoopTree::<true>::of(&body);
        let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
        let mut preamble = Vec::new();
        let site = UnitSite {
            preamble: &mut preamble,
            body: &mut body,
        };
        LoopAbsorptionManager::new(&mut worklist, &mut tree, site)
            .unwrap()
            .do_children_left_sibling_subtree_heights_match()
    }

    /// e069 over one [`scenario`].
    fn right_matches(children: &[usize], anchor: usize) -> bool {
        let (mut body, iv) = scenario(children, anchor);
        let mut tree = LoopTree::<true>::of(&body);
        let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
        let mut preamble = Vec::new();
        let site = UnitSite {
            preamble: &mut preamble,
            body: &mut body,
        };
        LoopAbsorptionManager::new(&mut worklist, &mut tree, site)
            .unwrap()
            .do_children_right_sibling_subtree_heights_match()
    }

    /// e066: the anchor is the worklist's last node, and the two shapes that have no anchor at all.
    #[test]
    fn new_anchors_on_the_worklists_last_loop() {
        let (mut body, iv) = scenario(&[0, 0], 1);
        let mut tree = LoopTree::<true>::of(&body);
        let anchor = tree.node_of(ForRef(iv)).unwrap();
        let root = tree.root();
        let mut preamble = Vec::new();

        let mut empty: Vec<LoopNodeId> = Vec::new();
        assert!(
            LoopAbsorptionManager::new(
                &mut empty,
                &mut tree,
                UnitSite {
                    preamble: &mut preamble,
                    body: &mut body,
                },
            )
            .is_none()
        );
        let mut just_the_root = vec![root];
        assert!(
            LoopAbsorptionManager::new(
                &mut just_the_root,
                &mut tree,
                UnitSite {
                    preamble: &mut preamble,
                    body: &mut body,
                },
            )
            .is_none()
        );

        let mut worklist = vec![tree.root(), anchor];
        let manager = LoopAbsorptionManager::new(
            &mut worklist,
            &mut tree,
            UnitSite {
                preamble: &mut preamble,
                body: &mut body,
            },
        )
        .unwrap();
        assert_eq!(manager.anchor, anchor);
        assert_eq!(manager.anchor_for_op, ForRef(iv));
        assert!(manager.positions().is_some());
    }

    /// e067's folded arm: `%c8` + 4 becomes one `%c12` in the preamble, and the old bound is dropped.
    #[test]
    fn update_loop_bound_folds_a_constant_bound() {
        let mut vals = Values::default();
        let bound = vals.mint();
        let iv = vals.mint();
        let mut body = vec![
            constant(bound, 8, ScalarTy::Index),
            for_op(iv, bound, Vec::new()),
        ];
        let mut tree = LoopTree::<true>::of(&body);
        let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
        let mut preamble = Vec::new();
        let site = UnitSite {
            preamble: &mut preamble,
            body: &mut body,
        };
        let mut manager = LoopAbsorptionManager::new(&mut worklist, &mut tree, site).unwrap();

        manager.update_loop_bound(&mut vals, BoundIncrement(4));

        let new_bound = bound_of(manager.anchor_for_op, manager.site.body.as_slice()).unwrap();
        assert_ne!(new_bound, bound);
        assert_eq!(manager.site.preamble.len(), 1);
        assert!(matches!(
            manager.site.preamble[0],
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 12,
                result,
                ..
            }) if result == new_bound
        ));
        // The increment constant was erased with the fold, and so was the bound left with no uses.
        assert_eq!(manager.site.body.len(), 1);
    }

    /// e068: two childless left siblings answer two leaf children; a taller sibling does not.
    #[test]
    fn left_sibling_heights_must_match_child_heights() {
        assert!(left_matches(&[0, 0, 2], 2));
        assert!(!left_matches(&[0, 1, 2], 2));
        // Fewer siblings than children.
        assert!(!left_matches(&[0, 2], 1));
    }

    /// e069: the same, walking right.
    #[test]
    fn right_sibling_heights_must_match_child_heights() {
        assert!(right_matches(&[2, 0, 0], 0));
        assert!(!right_matches(&[2, 1, 0], 0));
        assert!(!right_matches(&[2, 0], 0));
    }

    /// e070: the removed node leaves the worklist and the sibling chain closes over it.
    #[test]
    fn remove_nodes_drops_them_from_the_worklist_and_the_tree() {
        let (mut body, iv) = scenario(&[0, 0, 0], 2);
        let mut tree = LoopTree::<true>::of(&body);
        let anchor = tree.node_of(ForRef(iv)).unwrap();
        let first = tree.first_child(tree.root()).unwrap();
        let middle = tree.next_sibling(first).unwrap();
        let mut worklist = vec![first, middle, anchor];
        let mut preamble = Vec::new();
        let site = UnitSite {
            preamble: &mut preamble,
            body: &mut body,
        };
        let mut manager = LoopAbsorptionManager::new(&mut worklist, &mut tree, site).unwrap();

        manager.remove_nodes_from_worklist_and_tree(&[middle]);

        assert_eq!(*manager.sorted_worklist, vec![first, anchor]);
        assert_eq!(manager.tree.next_sibling(first), Some(anchor));
        assert_eq!(manager.tree.parent_loop(middle), None);
    }
}
