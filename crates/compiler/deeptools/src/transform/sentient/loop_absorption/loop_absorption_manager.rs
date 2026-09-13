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

// ⛔ NOTHING IN THIS MODULE IS REACHED FROM THE PIPELINE UNTIL `e640_runOnOperation` (level 7) LANDS,
// and CI runs clippy with `-D warnings`. ⭐ REMOVE THIS WITH e640: e600 gives the manager its caller,
// but that caller has none of its own until the pass is wired in.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    BlockArgEquivalence, EquivalenceTag, OperationEquivalence, SubregionCompare,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient::RegType;
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, Op, Val, sentient, uniform,
};
use crate::transform::sentient::loop_coalescing::TripLimit;
use crate::transform::sentient::loop_tree::{LoopNodeId, LoopTree};
use crate::transform::sentient::utils::{ConstKind, is_constant};
use crate::transform::sentient::{ForRef, ops_are_equivalent};

/// WHERE THE PASS MAY INSERT — the `dataflow.program_unit` the anchor loop sits in.
///
/// ⛔⛔ `DT_CHECK_MSG(unit, "expected loop to appear in dataflow.program_unit")`
/// (`LoopAbsorption.cpp:70`) IS THIS TYPE. The check cannot be asked at run time because naming the
/// unit is what builds one. ⭐ `preamble` IS THE REGION HOLDING the unit, which is where
/// `OpBuilder const_builder(dyn_cast<dataflow::ProgramUnitOp>(*unit))` (`:71`) inserts: an
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

/// WHICH ARGUMENT OF THE ANCHOR LOOP A NEW STARTING VALUE BELONGS TO — the `int` key of
/// `iter_arg_num_to_new_start_val_` (`LoopAbsorption.hpp:43`).
///
/// ⛔⛔ IT IS A BLOCK-ARGUMENT NUMBER USED DIRECTLY AS AN OPERAND INDEX. `canAbsorbToTheLeft` keys on
/// `cast<BlockArgument>(b_operand).getArgNumber()` (`:169`) and `absorbIntoAnchorFromLeft` hands it
/// straight to `setOperand` (`:238`); a `sentient.for` BINDS `[iv, iterArgs..]` and TAKES
/// `[bound, inits..]`, so `AnchorArgNumber(i)` for `i >= 1` really is `inits[i - 1]` — but
/// `AnchorArgNumber(0)` is the induction variable, and writes the loop's `$bound`.
/// ⭐ A TRAP KEPT, NOT A TYPO FIXED: the mirror image in `canAbsorbToTheRight` subtracts the 1 and
/// says so — *"getArgNumber() counts the inductive variable as argument #0"* (`:334-336`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct AnchorArgNumber(pub usize);

/// HOW MUCH `updateLoopBound` ADDS TO THE ANCHOR'S TRIP COUNT — its `int64_t increment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BoundIncrement(pub i64);

/// `LoopAbsorptionManager` (`LoopAbsorption.hpp:22`) — one anchor loop and the absorptions into it.
///
/// ⭐ `iter_arg_num_to_new_start_val_` AND `to_update_` (`LoopAbsorption.hpp:43-51`) ARE FIELDS: `e506_canAbsorbToTheLeft`
/// and `e507_canAbsorbToTheRight` fill them while checking equivalence, and the two absorptions here
/// apply and clear them.
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
    /// `iter_arg_num_to_new_start_val_` (`LoopAbsorption.hpp:43`) — drained by
    /// [`Self::absorb_into_anchor_from_left`]. See [`AnchorArgNumber`] for what the key really is.
    iter_arg_num_to_new_start_val: BTreeMap<AnchorArgNumber, Val>,
    /// `to_update_` (`LoopAbsorption.hpp:51`) — the old→new result rewrites, drained by
    /// [`Self::absorb_into_anchor_from_right`].
    to_update: BTreeMap<Val, Val>,
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
        // `anchor_(sorted_worklist.back())` (`LoopAbsorption.cpp:42`).
        let anchor = *sorted_worklist.last()?;
        // `anchor_for_op_ = anchor_->getOpAs<sentient::ForOp>();` (`:47`).
        let anchor_for_op = tree.loop_of(anchor)?;
        // `anchor_bb_`/`body_bb_`/`terminator_` (`:48-50`) — derived, so all this needs is that the
        // loop is findable at all.
        positions_of(anchor_for_op, site.body)?;
        Some(Self {
            // `oe_(dcc::OperationEquivalence(nullptr, nullptr, DEBUG_TYPE, true, false, true))`
            // (`:43-46`) — `use_equiv_classes` is `tagged`'s default.
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
            iter_arg_num_to_new_start_val: BTreeMap::new(),
            to_update: BTreeMap::new(),
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
    /// (`:72-78`) — it is the operand of a `scalar_add` that only two of the four arms emit.
    /// ⛔ TRAP: `bound.getType()` is always `index` — `Index:$bound` (`SentientOps.td:62`).
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
        // `auto const_increm = sentient::ConstantOp::create(const_builder, …, increment);` (`:72-73`).
        let const_increm = vals.mint();
        self.site
            .preamble
            .push(constant(const_increm, increment.0, bound_type));

        if let Some(value) = const_bound {
            // `setBound(ConstantOp::create(const_builder, loc, bound_type,
            //  const_bound.getValue() + increment))` (`:76-77`).
            let folded = vals.mint();
            self.site.preamble.push(constant(
                folded,
                value.saturating_add(increment.0),
                bound_type,
            ));
            set_bound(self.anchor_for_op, self.site.body, folded);
            // `const_increm.erase();` (`:78`) — the add was never emitted, so nothing reads it.
            dialects::erase_defining_op(&mut *self.site.preamble, const_increm);
        } else if query_map {
            // `is_query_map_constant` (`:64-66`) chooses between two `updateBoundToValuePlusMap`
            // calls (`:80-86`); the predicate AND the rewrite are both in
            // `Transform/Sentient/Analyses/Utils`, which is outside this campaign.
            todo!(
                "dcc::utils::isConstant / dcc::utils::updateBoundToValuePlusMap — out of campaign scope"
            )
        } else {
            // `sentient::AddOp::create(builder, loc, bound_type, bound, const_increm)` (`:99-100`),
            // placed *"as to avoid preventing absorption opportunities"* (`:88-91`).
            let sum = vals.mint();
            let add = Op::Sentient(sentient::Op::ScalarAdd {
                lhs: bound,
                rhs: const_increm,
                result: sum,
                reg: None,
                ty: bound_type,
                element_size: None,
            });
            if has_definer {
                // `builder.setInsertionPointAfter(bound.getDefiningOp());` (`:94`).
                insert_after_definition(&mut *self.site.body, bound, add);
            } else {
                // `setInsertionPointToStart(cast<BlockArgument>(bound).getOwner());` (`:96-97`).
                insert_at_start_of_binder(&mut *self.site.body, bound, add);
            }
            set_bound(self.anchor_for_op, self.site.body, sum);
        }

        // `if (bound.getDefiningOp() && bound.getDefiningOp()->use_empty())
        //  bound.getDefiningOp()->erase();` (`:103-104`).
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
            // (`:127-130`) — the anchor ran out of siblings before it ran out of children.
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
            // `llvm::erase_if(sorted_worklist_, [n](LoopNode *node) { return node == n; })` (`:413-414`).
            self.sorted_worklist.retain(|node| node != n);
            // `tree_.remove(n);` (`:415`).
            self.tree.remove(*n);
        }
    }

    /// Replaces: e309_absorbIntoAnchorFromLeft
    ///
    /// Rewrite the anchor's starting values, then delete the ops immediately to its LEFT — one per op
    /// of its body, the yield excluded.
    ///
    /// ⛔ THE KEY IS A BLOCK-ARGUMENT NUMBER USED AS AN OPERAND INDEX — see [`AnchorArgNumber`].
    /// ⭐ SATURATING WHERE `std::prev` WALKS OFF `begin()` (`:245-247`): it is `canAbsorbToTheLeft`'s
    /// `rend()` test that keeps that walk in range (`:154`), so a short block absorbs what is there.
    pub(crate) fn absorb_into_anchor_from_left(&mut self) {
        // `for (pair : ..) setOperand(..)` then `clear()` (`:237-240`) — cleared even if the anchor
        // has since moved, as the reference's `clear()` is unconditional too.
        let updates = core::mem::take(&mut self.iter_arg_num_to_new_start_val);
        let Some((block, at)) = anchor_slot_mut(self.anchor_for_op, &mut *self.site.body) else {
            return;
        };
        for (arg, val) in updates {
            dialects::set_operand(&mut block[at], arg.0, val);
        }
        // `int count = body_bb_->getOperations().size() - 1;` (`:244`) — the yield is not counted.
        let count = body_len(&block[at]).saturating_sub(1);
        block.drain(at.saturating_sub(count)..at);
    }

    /// Replaces: e310_absorbIntoAnchorFromRight
    ///
    /// Move every reader of an absorbed op's result onto the anchor's own result, then delete the ops
    /// immediately to its RIGHT — one per op of its body, the yield excluded.
    ///
    /// ⭐ THE UNIT IS THE REWRITE SCOPE: `Value::replaceAllUsesWith` is module-wide and this manager's
    /// widest reach is the `dataflow.program_unit` the anchor sits in.
    /// ⭐ ERASING IN REVERSE (`:400-401`) IS ONE CONTIGUOUS REMOVAL HERE, so the order is kept for free.
    pub(crate) fn absorb_into_anchor_from_right(&mut self) {
        // `for (pair : to_update_) get<0>(pair).replaceAllUsesWith(get<1>(pair));` then `clear()`
        // (`:388-391`).
        let updates = core::mem::take(&mut self.to_update);
        for (of, with) in updates {
            dialects::replace_all_uses_with(&mut *self.site.body, of, with);
        }
        let Some((block, at)) = anchor_slot_mut(self.anchor_for_op, &mut *self.site.body) else {
            return;
        };
        let count = body_len(&block[at]).saturating_sub(1);
        // `std::next(Block::iterator(anchor_for_op_), 1)` walked forward `count` times (`:397-399`).
        let end = block.len().min(at + 1 + count);
        block.drain(at + 1..end);
    }

    /// Replaces: e506_canAbsorbToTheLeft
    ///
    /// Walk the anchor's body backwards from its second-last op against the ops immediately to its
    /// LEFT: each pair must be the same computation, with the loop side's operands reaching the
    /// anchor's own iter args, and every result pair must have the same number of readers.
    ///
    /// ⛔ THE CHECKER IS WHERE THE EFFECT IS — `iter_arg_num_to_new_start_val` is FILLED here and
    /// spent by [`Self::absorb_into_anchor_from_left`]; a mismatch leaves whatever it filled, as the
    /// reference does.
    /// ⭐ AN EMPTY BODY ANSWERS `true`: the reference's `std::next(rbegin(), 1)` on a body of one op
    /// is already `rend()`, and on an empty one is undefined.
    pub(crate) fn can_absorb_to_the_left(&mut self, nodes_to_delete: &mut Vec<LoopNodeId>) -> bool {
        // `if (!doChildrenLeftSiblingSubtreeHeightsMatch()) return false;` (`:137-144`).
        if !self.do_children_left_sibling_subtree_heights_match() {
            return false;
        }
        let oe = self.oe;
        let LoopAbsorptionManager {
            tree,
            anchor,
            anchor_for_op,
            site,
            iter_arg_num_to_new_start_val,
            ..
        } = self;
        let anchor_for_op = *anchor_for_op;
        let unit_body: &[Op] = &*site.body;
        let Some(AnchorPositions {
            block,
            body,
            terminator,
        }) = positions_of(anchor_for_op, unit_body)
        else {
            return false;
        };
        // ⭐ INNERMOST FIRST, and harmless when the anchor is top-level and the two are one region.
        let scope = [block, unit_body];
        let defs = Definitions::from_innermost(&scope);
        let Some(anchor_at) = anchor_index(anchor_for_op, block) else {
            return false;
        };
        // `anchor_for_op_.getInits()` and `terminator_->getOperands()`, indexed alike.
        let inits: Vec<Val> = carried_of(anchor_for_op, unit_body)
            .iter()
            .flat_map(|carried| carried.iter().map(|carried| carried.init))
            .collect();
        let yielded = yielded_by(terminator);

        // `dcc::LoopNode *prev_sibling = anchor_->getPrevSibling();` (`:148`).
        let mut prev_sibling = tree.prev_sibling(*anchor);
        let mut left = anchor_at;
        for body_at in (0..body.len().saturating_sub(1)).rev() {
            // `if (block_it == anchor_bb_->getOperations().rend()) return false;` (`:154`).
            let Some(next_left) = left.checked_sub(1) else {
                return false;
            };
            left = next_left;
            let op_in_loop = &body[body_at];
            let op_to_left_of_loop = &block[left];

            // `check_operand_equivalence` (`:158-188`) — `b_operand` is the loop side.
            let mut check = |a_operand: Val, b_operand: Val| -> bool {
                // `if (!isa<BlockArgument>(b_operand)) return false;` and
                // `if (b_owner != anchor_for_op_) return false;` (`:163-172`).
                let Some((b_owner, b_arg_idx)) = defs.for_arg_of(b_operand) else {
                    return false;
                };
                if !is_anchor(b_owner, anchor_for_op) {
                    return false;
                }
                let Some(assigned) = iter_arg_num_to_new_start_val
                    .get(&AnchorArgNumber(b_arg_idx))
                    .copied()
                else {
                    // `iter_arg_num_to_new_start_val_.insert({b_arg_idx, a_operand}); return true;`
                    iter_arg_num_to_new_start_val.insert(AnchorArgNumber(b_arg_idx), a_operand);
                    return true;
                };
                // `return a_operand_op ? oe_.operationsAreEquivalent(*it_op, *a_operand_op)
                //                      : (it->second == a_operand);` (`:178-182`).
                match (defs.of(a_operand), defs.of(assigned)) {
                    (None, _) => assigned == a_operand,
                    (Some(a_operand_op), Some(it_op)) => {
                        ops_are_equivalent(it_op, a_operand_op, defs, oe.block_args, &mut None)
                    }
                    // ⛔ THE REFERENCE DEREFERENCES A NULL `it_op` HERE (`:178-181`): the assigned
                    // start value is itself a region argument while `a_operand` is a result.
                    (Some(_), None) => todo!(
                        "LoopAbsorption.cpp:178-181 dereferences a null defining op for the \\
                         already-assigned starting value"
                    ),
                }
            };
            let mut checker: Option<&mut dyn FnMut(Val, Val) -> bool> = Some(&mut check);
            if !ops_are_equivalent(
                op_to_left_of_loop,
                op_in_loop,
                defs,
                oe.block_args,
                &mut checker,
            ) {
                return false;
            }

            // `zip(op_to_left_of_loop.getResults(), op_in_loop.getResults())` (`:198-222`).
            for (left_result, loop_result) in dialects::results(op_to_left_of_loop)
                .into_iter()
                .zip(dialects::results(op_in_loop))
            {
                if dialects::use_count(left_result, unit_body)
                    != dialects::use_count(loop_result, unit_body)
                {
                    return false;
                }
                // Whatever the loop yields must already start from the op to its left.
                for (i, yielded_val) in yielded.iter().enumerate() {
                    if *yielded_val == loop_result && inits.get(i) != Some(&left_result) {
                        return false;
                    }
                }
            }

            // `if (isa<sentient::ForOp>(op_in_loop)) { nodes_to_delete.push_back(prev_sibling); .. }`
            // (`:228-231`). ⭐ e068 PROVED THE SIBLING IS THERE for every child, so `None` is a
            // refusal rather than the reference's null push.
            if matches!(op_in_loop, Op::Sentient(sentient::Op::For { .. })) {
                let Some(sibling) = prev_sibling else {
                    return false;
                };
                nodes_to_delete.push(sibling);
                prev_sibling = tree.prev_sibling(sibling);
            }
        }
        true
    }

    /// Replaces: e507_canAbsorbToTheRight
    ///
    /// [`Self::can_absorb_to_the_left`] mirrored: the anchor's body forwards from its FIRST op
    /// against the ops immediately to its RIGHT, with each iter arg's reader count required to match
    /// its result's up front.
    ///
    /// ⛔ THE EFFECT IS `to_update` — the old→new result rewrites [`Self::absorb_into_anchor_from_right`]
    /// spends.
    /// ⭐ THE `- 1` ON THE ARGUMENT NUMBER IS THE REFERENCE'S OWN (`:334-336`), and argument #0 — the
    /// induction variable — has no carried slot to name, so it never matches.
    pub(crate) fn can_absorb_to_the_right(
        &mut self,
        nodes_to_delete: &mut Vec<LoopNodeId>,
    ) -> bool {
        // `if (!doChildrenRightSiblingSubtreeHeightsMatch()) return false;` (`:284-291`).
        if !self.do_children_right_sibling_subtree_heights_match() {
            return false;
        }
        let oe = self.oe;
        let LoopAbsorptionManager {
            tree,
            anchor,
            anchor_for_op,
            site,
            to_update,
            ..
        } = self;
        let anchor_for_op = *anchor_for_op;
        let unit_body: &[Op] = &*site.body;
        let Some(AnchorPositions {
            block,
            body,
            terminator,
        }) = positions_of(anchor_for_op, unit_body)
        else {
            return false;
        };
        let scope = [block, unit_body];
        let defs = Definitions::from_innermost(&scope);
        let Some(anchor_at) = anchor_index(anchor_for_op, block) else {
            return false;
        };
        let carried = carried_of(anchor_for_op, unit_body);
        let yielded = yielded_by(terminator);

        // `zip(getRegionIterArgs(), getResults())`: an iter arg read more often than its result is
        // read is one whose absorption would drop a reader (`:293-307`).
        for slot in carried.into_iter().flatten() {
            if dialects::use_count(slot.arg, unit_body)
                != dialects::use_count(slot.result, unit_body)
            {
                return false;
            }
        }
        let results: Vec<Val> = carried
            .iter()
            .flat_map(|carried| carried.iter().map(|carried| carried.result))
            .collect();

        // `dcc::LoopNode *next_sibling = anchor_->getNextSibling();` (`:309`).
        let mut next_sibling = tree.next_sibling(*anchor);
        let mut right = anchor_at;
        for op_in_loop in body.iter().take(body.len().saturating_sub(1)) {
            right += 1;
            // `if (block_it == anchor_bb_->getOperations().end()) return false;` (`:317`).
            let Some(op_to_right_of_loop) = block.get(right) else {
                return false;
            };

            // `check_operand_equivalence` (`:322-341`) — here `a_operand` is the loop side.
            let mut check = |a_operand: Val, b_operand: Val| -> bool {
                let (Some((a_owner, a_arg_num)), Some(b_operand_op)) =
                    (defs.for_arg_of(a_operand), defs.of(b_operand))
                else {
                    return false;
                };
                // `getArgNumber() - 1` — argument #0 is the induction variable and names no slot.
                let Some(a_arg_idx) = a_arg_num.checked_sub(1) else {
                    return false;
                };
                let Some(b_res_idx) = dialects::results(b_operand_op)
                    .iter()
                    .position(|result| *result == b_operand)
                else {
                    return false;
                };
                is_anchor(a_owner, anchor_for_op)
                    && is_anchor(b_operand_op, anchor_for_op)
                    && a_arg_idx == b_res_idx
            };
            let mut checker: Option<&mut dyn FnMut(Val, Val) -> bool> = Some(&mut check);
            if !ops_are_equivalent(
                op_in_loop,
                op_to_right_of_loop,
                defs,
                oe.block_args,
                &mut checker,
            ) {
                return false;
            }

            // `zip(op_in_loop.getResults(), op_to_right_of_loop.getResults())` (`:350-374`).
            for (loop_result, right_result) in dialects::results(op_in_loop)
                .into_iter()
                .zip(dialects::results(op_to_right_of_loop))
            {
                match yielded.iter().position(|yielded| *yielded == loop_result) {
                    // A yielded result already leaves the loop, so the op to the right can read it.
                    Some(yield_idx) => {
                        if let Some(result) = results.get(yield_idx) {
                            to_update.insert(right_result, *result);
                        }
                    }
                    None => {
                        if dialects::use_count(loop_result, unit_body)
                            != dialects::use_count(right_result, unit_body)
                        {
                            return false;
                        }
                    }
                }
            }

            if matches!(op_in_loop, Op::Sentient(sentient::Op::For { .. })) {
                let Some(sibling) = next_sibling else {
                    return false;
                };
                nodes_to_delete.push(sibling);
                next_sibling = tree.next_sibling(sibling);
            }
        }
        true
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
        Op::UniformRegions(regions) => regions
            .regions()
            .iter()
            .map(|region| region.body.as_slice())
            .collect(),
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
        Op::UniformRegions(regions) => regions
            .regions_mut()
            .iter_mut()
            .map(|region| &mut region.body)
            .collect(),
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

/// [`positions_of`]'S MUTABLE TWIN — the block holding the anchor loop, and its index in it.
///
/// ⭐ THE TWO ABSORPTIONS NEED THE BLOCK ITSELF, not a view of it: they `erase()` the ops beside the
/// anchor, which `AnchorPositions`' shared slices cannot express.
fn anchor_slot_mut(for_op: ForRef, scope: &mut Vec<Op>) -> Option<(&mut Vec<Op>, usize)> {
    if let Some(at) = scope.iter().position(
        |op| matches!(op, Op::Sentient(sentient::Op::For { iv, .. }) if *iv == for_op.0),
    ) {
        return Some((scope, at));
    }
    for op in scope.iter_mut() {
        for region in regions_mut_of(op) {
            if let Some(found) = anchor_slot_mut(for_op, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `body_bb_->getOperations().size()` — the anchor loop's body ops, its `sentient.yield` INCLUDED,
/// which is why both absorptions subtract one.
fn body_len(anchor: &Op) -> usize {
    match anchor {
        Op::Sentient(sentient::Op::For { body, .. }) => body.len(),
        _ => 0,
    }
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

/// `op == anchor_for_op_` — the identity comparison both checkers make on a block argument's owner.
fn is_anchor(op: &Op, anchor_for_op: ForRef) -> bool {
    matches!(op, Op::Sentient(sentient::Op::For { iv, .. }) if *iv == anchor_for_op.0)
}

/// `Block::iterator(anchor_for_op_)` — where the anchor sits in the block holding it.
fn anchor_index(for_op: ForRef, block: &[Op]) -> Option<usize> {
    block.iter().position(|op| is_anchor(op, for_op))
}

/// `anchor_for_op_.getInits()`, `getRegionIterArgs()` and `getResults()` in one lookup — the anchor's
/// `$initArgs` slots, which hold all three lists side by side.
fn carried_of(for_op: ForRef, scope: &[Op]) -> Option<&[sentient::Carried]> {
    for op in scope {
        if let Op::Sentient(sentient::Op::For { iv, carried, .. }) = op
            && *iv == for_op.0
        {
            return Some(carried.as_slice());
        }
        for region in regions_of(op) {
            if let Some(found) = carried_of(for_op, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `terminator_->getOperands()` — what the anchor's body hands back, and nothing for a body whose
/// last op is not a `sentient.yield`.
fn yielded_by(terminator: Option<&sentient::Op>) -> &[Val] {
    match terminator {
        Some(sentient::Op::Yield { results }) => results.as_slice(),
        _ => &[],
    }
}

/// `cl::opt<bool> EnableDynamicLoopAbsorption("dcc-loop-absorption-dynamic-loops", .., cl::init(false))`
/// (`LoopAbsorption.cpp:30-33`) — a `dcc-opt` command-line flag, not a program property.
pub(crate) const ENABLE_DYNAMIC_LOOP_ABSORPTION: bool = false;

/// `dcc::utils::getMaxConstValAcrossUnits(op)` (`Analyses/Utils.cpp:400-419`) — a
/// `sentient.scalar_constant`'s own value, or the LARGEST constant a `uniform.query_map` maps.
///
/// ⛔ THE TWO `DT_CHECK_MSG`s AND `max_val.value()` ARE THE THREE `panic!`s: another defining op, a
/// mapped value that is not a constant, and an empty immutable mapping.
fn max_const_val_across_units(val: Val, defs: Definitions<'_>) -> i64 {
    match defs.of(val) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => *value,
        Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
            let mut max_val: Option<i64> = None;
            for mapped in dialects::uniform_mapping_values(*map, *key, defs) {
                let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) = defs.of(mapped)
                else {
                    panic!(
                        "Expect query map values to be constants. (Analyses/Utils.cpp:412-413): \
                         {mapped:?}"
                    )
                };
                max_val = Some(max_val.map_or(*value, |seen: i64| seen.max(*value)));
            }
            let Some(max_val) = max_val else {
                panic!(
                    "getMaxConstValAcrossUnits reached `max_val.value()` on an immutable mapping \
                     with no values, which throws (Analyses/Utils.cpp:418)"
                )
            };
            max_val
        }
        _ => panic!(
            "Should only be called on constant operation (Analyses/Utils.cpp:405-406): {val:?}"
        ),
    }
}

/// `dcc::utils::updateDbgName(prefix, op, suffix)` (`dcc/src/Utils/Utils.cpp:491-515`) — ⛔ IT WRITES
/// ONLY WHERE A NAME ALREADY EXISTS, because `getNewDbgNameFromOp` returns null without one.
fn update_dbg_name(op: &mut Op, prefix: &str, suffix: &str) {
    let Op::Sentient(inner) = op else {
        return;
    };
    if let Some(Some(name)) = sentient::dbg_name_mut(inner) {
        *name = format!("{prefix}{name}{suffix}");
    }
}

impl LoopAbsorptionManager<'_> {
    /// Replaces: e562_absorptionAnalysis
    ///
    /// Absorbs as many right siblings into the anchor as the LCCR still has room for, then as many
    /// left ones, and finally raises the anchor's trip count by the total and tags its debug name.
    ///
    /// ⛔ THE LEFT LOOP NEVER CLEARS `to_update_` (`:461-478`) — the reference clears it once between
    /// the two loops (`:459`) and the right loop clears it per round (`:452`), a difference kept as is.
    /// ⭐ `max_bound_across_units += num_right_absorptions` BEFORE THE LEFT LOOP (`:463`) charges the
    /// right absorptions against the same budget, which is why one bound serves both directions.
    pub(crate) fn absorption_analysis(&mut self, vals: &mut Values) {
        // `isConstant<sentient::ConstantOp>(anchor_for_op_.getBound())` (`:426-427`).
        let bound = bound_of(self.anchor_for_op, self.site.body);
        // ⭐ `getDefiningOp()` IS MODULE-WIDE, so the bound's definition may sit in the region holding
        // the unit as well as in the unit itself — [`UnitSite::preamble`] is that outer region.
        let scope: [&[Op]; 2] = [self.site.body, self.site.preamble];
        let defs = Definitions::from_innermost(&scope);
        let loop_bound_constant =
            bound.is_some_and(|bound| is_constant(bound, ConstKind::ScalarConstant, defs));
        if !loop_bound_constant && !ENABLE_DYNAMIC_LOOP_ABSORPTION {
            return;
        }
        // `loop_bound_constant ? getMaxConstValAcrossUnits(..) : -1` (`:437-440`).
        let mut max_bound_across_units = if loop_bound_constant {
            bound.map_or(-1, |bound| max_const_val_across_units(bound, defs))
        } else {
            -1
        };

        // ── right absorptions (`:442-459`) ──
        let mut num_right_absorptions: i64 = 0;
        loop {
            if loop_bound_constant
                && max_bound_across_units + num_right_absorptions >= TripLimit::LCCR.0
            {
                break;
            }
            let mut nodes_to_delete = Vec::new();
            self.to_update.clear();
            if !self.can_absorb_to_the_right(&mut nodes_to_delete) {
                break;
            }
            self.absorb_into_anchor_from_right();
            num_right_absorptions += 1;
            self.remove_nodes_from_worklist_and_tree(&nodes_to_delete);
        }
        self.to_update.clear();

        // ── left absorptions (`:461-478`) ──
        let mut num_left_absorptions: i64 = 0;
        max_bound_across_units += num_right_absorptions;
        loop {
            if loop_bound_constant
                && max_bound_across_units + num_left_absorptions >= TripLimit::LCCR.0
            {
                break;
            }
            let mut nodes_to_delete = Vec::new();
            if !self.can_absorb_to_the_left(&mut nodes_to_delete) {
                break;
            }
            self.absorb_into_anchor_from_left();
            num_left_absorptions += 1;
            self.remove_nodes_from_worklist_and_tree(&nodes_to_delete);
        }

        if num_left_absorptions + num_right_absorptions > 0 {
            self.update_loop_bound(
                vals,
                BoundIncrement(num_left_absorptions + num_right_absorptions),
            );
            if let Some((block, at)) = anchor_slot_mut(self.anchor_for_op, &mut *self.site.body) {
                update_dbg_name(&mut block[at], "LA(", ")");
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// One `sentient.for` with no carried values.
    fn for_op(iv: Val, bound: Val, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

    /// One `$initArgs` slot of a loop — the five facts a [`sentient::Carried`] keeps together.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// One `sentient.for` carrying one value, `body` then the `sentient.yield` that closes it.
    fn carrying_for(iv: Val, bound: Val, carried: sentient::Carried, body: Vec<Op>) -> Op {
        let mut body = body;
        body.push(Op::Sentient(sentient::Op::Yield {
            results: vec![carried.arg],
        }));
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: vec![carried],
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

    /// One `sentient.for` carrying one value, `body` closed by a `sentient.yield` of `yielded`.
    fn for_yielding(
        iv: Val,
        bound: Val,
        carried: sentient::Carried,
        body: Vec<Op>,
        yielded: Val,
    ) -> Op {
        let mut body = body;
        body.push(Op::Sentient(sentient::Op::Yield {
            results: vec![yielded],
        }));
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: vec![carried],
            dbg_name: None,
            body,
        })
    }

    /// `sentient.scalar_add` — a two-operand op, so a checker is asked twice about one pair of ops.
    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    /// The manager over `body`, anchored on the loop binding `iv`.
    fn anchored<'a>(
        body: &'a mut Vec<Op>,
        preamble: &'a mut Vec<Op>,
        tree: &'a mut LoopTree<true>,
        worklist: &'a mut Vec<LoopNodeId>,
    ) -> LoopAbsorptionManager<'a> {
        LoopAbsorptionManager::new(worklist, tree, UnitSite { preamble, body }).unwrap()
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

    /// e309: the new starting value lands on the anchor's own init, and exactly ONE op to the left is
    /// absorbed — one per body op with the yield not counted.
    #[test]
    fn absorb_from_left_rewrites_the_inits_and_absorbs_that_many_ops() {
        let mut vals = Values::default();
        let bound = vals.mint();
        let init = vals.mint();
        let new_start = vals.mint();
        let left = vals.mint();
        let iv = vals.mint();
        let arg = vals.mint();
        let result = vals.mint();
        let inner = vals.mint();
        let mut body = vec![
            constant(bound, 4, ScalarTy::Index),
            constant(init, 0, ScalarTy::Index),
            constant(new_start, 7, ScalarTy::Index),
            constant(left, 1, ScalarTy::Index),
            carrying_for(
                iv,
                bound,
                carried(init, arg, result),
                vec![constant(inner, 1, ScalarTy::Index)],
            ),
        ];
        let mut tree = LoopTree::<true>::of(&body);
        let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
        let mut preamble = Vec::new();
        let mut manager = LoopAbsorptionManager::new(
            &mut worklist,
            &mut tree,
            UnitSite {
                preamble: &mut preamble,
                body: &mut body,
            },
        )
        .unwrap();
        // ⭐ ARGUMENT #1, so operand #1 — which really is `inits[0]`. See [`AnchorArgNumber`].
        manager
            .iter_arg_num_to_new_start_val
            .insert(AnchorArgNumber(1), new_start);

        manager.absorb_into_anchor_from_left();

        assert!(manager.iter_arg_num_to_new_start_val.is_empty());
        assert_eq!(manager.site.body.len(), 4);
        let Op::Sentient(sentient::Op::For { carried, .. }) = &manager.site.body[3] else {
            panic!("the anchor is the last op of the block")
        };
        assert_eq!(carried[0].init, new_start);
    }

    /// e310: the reader to the right moves onto the anchor's result, and the op it read is absorbed.
    #[test]
    fn absorb_from_right_repoints_uses_and_absorbs_that_many_ops() {
        let mut vals = Values::default();
        let bound = vals.mint();
        let init = vals.mint();
        let iv = vals.mint();
        let arg = vals.mint();
        let result = vals.mint();
        let inner = vals.mint();
        let right = vals.mint();
        let sum = vals.mint();
        let mut body = vec![
            constant(bound, 4, ScalarTy::Index),
            constant(init, 0, ScalarTy::Index),
            carrying_for(
                iv,
                bound,
                carried(init, arg, result),
                vec![constant(inner, 1, ScalarTy::Index)],
            ),
            constant(right, 1, ScalarTy::Index),
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs: right,
                rhs: right,
                result: sum,
                reg: None,
                ty: ScalarTy::Index,
                element_size: None,
            }),
        ];
        let mut tree = LoopTree::<true>::of(&body);
        let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
        let mut preamble = Vec::new();
        let mut manager = LoopAbsorptionManager::new(
            &mut worklist,
            &mut tree,
            UnitSite {
                preamble: &mut preamble,
                body: &mut body,
            },
        )
        .unwrap();
        manager.to_update.insert(right, result);

        manager.absorb_into_anchor_from_right();

        assert!(manager.to_update.is_empty());
        assert_eq!(manager.site.body.len(), 4);
        assert!(matches!(
            &manager.site.body[3],
            Op::Sentient(sentient::Op::ScalarAdd { lhs, rhs, .. })
                if *lhs == result && *rhs == result
        ));
    }
    /// e506, on the vendor's own left case (`LoopAbsorption.cpp:108-120`): the op to the left reads
    /// `%zero` where the loop's own op reads its iter arg, and that iter arg starts from the left
    /// op's result — so `%zero` becomes the new starting value. The negative is a second operand the
    /// already-assigned start value contradicts.
    #[test]
    fn can_absorb_to_the_left_records_the_new_starting_value() {
        fn absorb(second_operand_differs: bool) -> (bool, BTreeMap<AnchorArgNumber, Val>) {
            let mut vals = Values::default();
            let bound = vals.mint();
            let zero = vals.mint();
            let other = vals.mint();
            let left = vals.mint();
            let iv = vals.mint();
            let arg = vals.mint();
            let result = vals.mint();
            let inner = vals.mint();
            let left_rhs = if second_operand_differs { other } else { zero };
            let mut body = vec![
                constant(bound, 5, ScalarTy::Index),
                constant(zero, 0, ScalarTy::Index),
                constant(other, 3, ScalarTy::Index),
                add(zero, left_rhs, left),
                for_yielding(
                    iv,
                    bound,
                    carried(left, arg, result),
                    vec![add(arg, arg, inner)],
                    inner,
                ),
            ];
            let mut tree = LoopTree::<true>::of(&body);
            let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
            let mut preamble = Vec::new();
            let mut manager = anchored(&mut body, &mut preamble, &mut tree, &mut worklist);
            let mut nodes_to_delete = Vec::new();
            let answer = manager.can_absorb_to_the_left(&mut nodes_to_delete);
            assert!(nodes_to_delete.is_empty(), "the absorbed op is not a loop");
            (answer, manager.iter_arg_num_to_new_start_val.clone())
        }

        let (matched, assigned) = absorb(false);
        assert!(matched);
        // ⭐ ARGUMENT #1, so `inits[0]` — see [`AnchorArgNumber`].
        assert_eq!(assigned.len(), 1);
        assert_eq!(assigned.keys().next(), Some(&AnchorArgNumber(1)));

        assert!(!absorb(true).0);
    }

    /// e507, on the vendor's own right case (`LoopAbsorption.cpp:252-267`): the op to the right reads
    /// the loop's result where the loop's own op reads the matching iter arg, so the reader is
    /// repointed at the anchor. The negative is an iter arg read more often than its result.
    #[test]
    fn can_absorb_to_the_right_repoints_the_reader_at_the_anchor() {
        fn absorb(result_read_once: bool) -> (bool, BTreeMap<Val, Val>, Val, Val) {
            let mut vals = Values::default();
            let bound = vals.mint();
            let zero = vals.mint();
            let other = vals.mint();
            let iv = vals.mint();
            let arg = vals.mint();
            let result = vals.mint();
            let inner = vals.mint();
            let right = vals.mint();
            let right_rhs = if result_read_once { other } else { result };
            let mut body = vec![
                constant(bound, 5, ScalarTy::Index),
                constant(zero, 0, ScalarTy::Index),
                constant(other, 3, ScalarTy::Index),
                for_yielding(
                    iv,
                    bound,
                    carried(zero, arg, result),
                    vec![add(arg, arg, inner)],
                    inner,
                ),
                add(result, right_rhs, right),
            ];
            let mut tree = LoopTree::<true>::of(&body);
            let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
            let mut preamble = Vec::new();
            let mut manager = anchored(&mut body, &mut preamble, &mut tree, &mut worklist);
            let mut nodes_to_delete = Vec::new();
            let answer = manager.can_absorb_to_the_right(&mut nodes_to_delete);
            assert!(nodes_to_delete.is_empty(), "the absorbed op is not a loop");
            (answer, manager.to_update.clone(), right, result)
        }

        let (matched, to_update, right, result) = absorb(false);
        assert!(matched);
        assert_eq!(to_update.get(&right), Some(&result));

        assert!(!absorb(true).0);
    }

    /// e562, on e507's own right case: one right absorption happens, the second round finds nothing
    /// to absorb, no left absorption matches a constant, and the trip count ends up one higher.
    #[test]
    fn absorption_analysis_absorbs_the_one_right_sibling_and_raises_the_bound() {
        let mut vals = Values::default();
        let bound = vals.mint();
        let zero = vals.mint();
        let other = vals.mint();
        let iv = vals.mint();
        let arg = vals.mint();
        let result = vals.mint();
        let inner = vals.mint();
        let right = vals.mint();
        let mut body = vec![
            constant(bound, 5, ScalarTy::Index),
            constant(zero, 0, ScalarTy::Index),
            constant(other, 3, ScalarTy::Index),
            for_yielding(
                iv,
                bound,
                carried(zero, arg, result),
                vec![add(arg, arg, inner)],
                inner,
            ),
            add(result, result, right),
        ];
        let mut tree = LoopTree::<true>::of(&body);
        let mut worklist = vec![tree.node_of(ForRef(iv)).unwrap()];
        let mut preamble = Vec::new();
        let mut manager = anchored(&mut body, &mut preamble, &mut tree, &mut worklist);

        manager.absorption_analysis(&mut vals);

        // The absorbed `add` is gone and the old bound with it.
        assert_eq!(manager.site.body.len(), 3);
        let new_bound = bound_of(manager.anchor_for_op, manager.site.body.as_slice()).unwrap();
        assert!(manager.site.preamble.iter().any(|op| matches!(
            op,
            Op::Sentient(sentient::Op::ScalarConstant { value: 6, result, .. }) if *result == new_bound
        )));
    }
}
