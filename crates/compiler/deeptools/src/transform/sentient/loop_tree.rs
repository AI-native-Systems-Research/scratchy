// SPDX-License-Identifier: Apache-2.0
//! `dcc/src/Analysis/LoopTree.hpp` — the dialect-neutral abstraction over a program's loop nests.
//!
//! ⚠️ NOT A SCHEDULED UNIT, DELIBERATELY WRITTEN HERE. `src/Analysis/` is outside the campaign's 656
//! definitions, so no entry number can be spent on it — and `e068`/`e069` ARE a walk of this tree
//! (`getLastChild`/`getPrevSibling` over the anchor's children) while `e070` is `tree_.remove(n)`.
//! There is no way to port those without the thing they walk. Six passes build one
//! (`LoopAbsorption`, `LoopMerging`, `LoopCoalescing`, `LoopRolling`, `MultiDimLoopPeeling`,
//! `TransformForExposedPipeline`), so it is homed beside them rather than inside the first.
//!
//! ⭐ THE ARENA IS ALREADY PORTED. `dcc::LoopTree` derives from `mlir::OperationTreeBase`, and so does
//! bridge 2's `LoopMaskTree` — [`OperationTreeBase`] is that base, generic in its payload exactly so a
//! second derived family instantiates it instead of writing a second arena.

// ⛔ NOTHING CALLS THIS UNTIL `e600_runLoopAbsorption` (level 5) LANDS, and CI runs clippy with
// `-D warnings`. ⭐ REMOVE THIS WITH e600.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use crate::bridges::dataflow_ir_to_sentient::vc_loop_mask_tree::{
    OperationNodeId, OperationTreeBase,
};
use crate::islands::sentient::dialects::{Op, sentient};
use crate::transform::sentient::ForRef;

/// A NODE'S IDENTITY WITHIN ONE LOOP TREE — what a `dcc::LoopNode *` is in the C++.
///
/// ⛔ AN INDEX, NOT A REFERENCE: the C++ tree is intrusive and cyclic, and `operator==` on a node is
/// `this == &n` (`OperationTree.hpp:33`) — pointer identity, which is exactly `==` on two of these.
/// See [`OperationNodeId`] for the whole argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopNodeId(OperationNodeId);

/// WHAT A `dcc::LoopNode` NAMES — `LoopTree.hpp:27`.
///
/// ⛔⛔ THE ROOT IS SYNTHETIC AND NAMES NO LOOP. *"Conceptually a program consists of a forest of loop
/// trees … The LoopTree class introduces a synthetic root node and collects all such trees under that
/// single root"* (`LoopTree.hpp:69-72`) — `new LoopNode(nullptr)` (`:150`). So the outermost loops of a
/// program unit are SIBLINGS under a node that stands for no operation, and an enum is what makes
/// `getOpAs<sentient::ForOp>()` on the root unaskable rather than null. Bridge 2's `LoopMaskNode` is
/// the same shape for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopNode {
    /// `new LoopNode(nullptr)` — the forest's collector.
    SyntheticRoot,
    /// One `sentient.for`, named by its induction variable. See [`ForRef`].
    Loop(ForRef),
}

/// `loop_node_to_subtree_height_`'s value — `unsigned short` (`LoopTree.hpp:174`).
///
/// ⛔ A LEAF IS **1**, NOT 0 (`LoopTree.hpp:182-183`), so there is no height that means "no node": that
/// is what [`LoopTree::subtree_height_of`]'s `Option` is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubtreeHeight(pub u16);

/// `dcc::LoopTree<LoopOp, ProgramUnit>` — one program unit's loop forest.
///
/// # ⭐⭐ `compute_subtree_heights_` IS A CONST GENERIC, WHICH DELETES ITS `DT_CHECK`
///
/// `getSubtreeHeightOf` opens with `DT_CHECK_MSG(compute_subtree_heights_, "Cannot compute subtree
/// heights if the corresponding LoopTree feature is not enabled.")` (`LoopTree.hpp:96-99`) — a flag
/// decided once at construction and then re-asked at every read. This crate's rule is that a flag
/// deciding which methods EXIST is a const generic, so [`Self::subtree_height_of`] is implemented on
/// `LoopTree<true>` alone and the refusal becomes an E0599 at the call site. `LoopAbsorption` builds
/// `LoopTree tree(op, /*compute_subtree_heights*/ true)` (`LoopAbsorption.cpp:495-496`); the other five
/// passes take the `false` default and cannot reach the getter at all.
#[derive(Debug, Clone)]
pub struct LoopTree<const HEIGHTS: bool> {
    /// `mlir::OperationTreeBase`, the base class.
    tree: OperationTreeBase<LoopNode>,
    /// `loop_node_to_subtree_height_` (`LoopTree.hpp:174`) — empty unless `HEIGHTS`.
    heights: HashMap<LoopNodeId, SubtreeHeight>,
}

impl<const HEIGHTS: bool> LoopTree<HEIGHTS> {
    /// `LoopTree(ProgramUnit unit, bool compute_subtree_heights)` and its `compute(unit)`
    /// (`LoopTree.hpp:74-78`, `:146-164`).
    ///
    /// ⭐ PRE-ORDER, AND THE RECURSION IS `getParentOfType<LoopOp>()`: a `sentient.for` reached from
    /// inside another one becomes its child, and one reached through a non-loop region (a
    /// `sentient.if`) keeps the enclosing loop as its parent, which is what the C++ `op_to_loop`
    /// lookup computes.
    #[must_use]
    pub fn of(unit_body: &[Op]) -> LoopTree<HEIGHTS> {
        let mut tree = OperationTreeBase::with_root(LoopNode::SyntheticRoot);
        let root = LoopNodeId(tree.root());
        collect(&mut tree, root, unit_body);
        let mut out = LoopTree {
            tree,
            heights: HashMap::new(),
        };
        if HEIGHTS {
            out.compute_subtree_heights();
        }
        out
    }

    /// `getRoot()` — `LoopTree.hpp:80-86`.
    #[must_use]
    pub fn root(&self) -> LoopNodeId {
        LoopNodeId(self.tree.root())
    }

    /// `OperationTreeBase::empty()` — `OperationTree.hpp:214`, a forest with no trees in it.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.tree.first_child(self.tree.root()).is_none()
    }

    /// `LoopNode::getOpAs<sentient::ForOp>()` — ⭐ `None` FOR THE SYNTHETIC ROOT AND ONLY FOR IT.
    #[must_use]
    pub fn loop_of(&self, n: LoopNodeId) -> Option<ForRef> {
        match self.tree.payload(n.0) {
            LoopNode::SyntheticRoot => None,
            LoopNode::Loop(for_op) => Some(*for_op),
        }
    }

    /// The node naming one loop, absent when this forest does not contain it.
    #[must_use]
    pub fn node_of(&self, for_op: ForRef) -> Option<LoopNodeId> {
        self.walk_preorder()
            .into_iter()
            .find(|n| self.loop_of(*n) == Some(for_op))
    }

    /// `LoopNode::getParentLoop()` — `LoopTree.hpp:33-35`.
    #[must_use]
    pub fn parent_loop(&self, n: LoopNodeId) -> Option<LoopNodeId> {
        self.tree.parent_node(n.0).map(LoopNodeId)
    }

    /// `LoopNode::getFirstChild()` — `LoopTree.hpp:36-38`.
    #[must_use]
    pub fn first_child(&self, n: LoopNodeId) -> Option<LoopNodeId> {
        self.tree.first_child(n.0).map(LoopNodeId)
    }

    /// `LoopNode::getNextSibling()` — `LoopTree.hpp:39-41`.
    #[must_use]
    pub fn next_sibling(&self, n: LoopNodeId) -> Option<LoopNodeId> {
        self.tree.next_sibling(n.0).map(LoopNodeId)
    }

    /// `LoopNode::getPrevSibling()` — `LoopTree.hpp:42-44`, a scan of the parent's chain.
    #[must_use]
    pub fn prev_sibling(&self, n: LoopNodeId) -> Option<LoopNodeId> {
        self.tree.prev_sibling(n.0).map(LoopNodeId)
    }

    /// `LoopNode::getLastChild()` — `LoopTree.hpp:46-48`, *"the last child in syntactic order"*.
    #[must_use]
    pub fn last_child(&self, n: LoopNodeId) -> Option<LoopNodeId> {
        self.tree.last_child(n.0).map(LoopNodeId)
    }

    /// `OperationTreeBase::remove(start)` — `OperationTree.hpp:216`. See [`OperationTreeBase::remove`]
    /// for what "does not free the storage" buys.
    pub fn remove(&mut self, n: LoopNodeId) {
        self.tree.remove(n.0);
    }

    /// `OperationNode::reverseBreadthFirstWalk(n, action)` — `Analysis/OperationTree.cpp:200-236`, the
    /// order `WalkOrder::kReverseBFS` walks in (`:100-102`, so `keep_order` is `false`).
    ///
    /// ⛔⛔ THE VISIT LIST IS PRECOMPUTED AND THE ACTION'S RETURN IS DISCARDED — `stack.push_back` runs
    /// the whole BFS first and then `(void)action(stack.back())` unwinds it. So an action of this walk
    /// may rewrite the IR and [`Self::remove`] nodes as it goes, and a `LoopNode *` it returns to steer
    /// the walk is DEAD CODE, which is exactly what `LoopMerging`'s `checkAndMerge` returns.
    #[must_use]
    pub fn walk_reverse_bfs(&self) -> Vec<LoopNodeId> {
        let mut queue = VecDeque::from([self.root()]);
        let mut out = Vec::new();
        while let Some(n) = queue.pop_front() {
            out.push(n);
            let mut child = self.first_child(n);
            while let Some(c) = child {
                queue.push_back(c);
                child = self.next_sibling(c);
            }
        }
        out.reverse();
        out
    }

    /// Every node, root first — the pre-order walk the two total helpers here need.
    #[must_use]
    fn walk_preorder(&self) -> Vec<LoopNodeId> {
        let mut out = Vec::new();
        let mut stack = vec![self.root()];
        while let Some(n) = stack.pop() {
            out.push(n);
            let mut child = self.first_child(n);
            let mut kids = Vec::new();
            while let Some(c) = child {
                kids.push(c);
                child = self.next_sibling(c);
            }
            // ⭐ REVERSED ONTO THE STACK so siblings come off it in syntactic order.
            kids.reverse();
            stack.extend(kids);
        }
        out
    }

    /// `computeSubtreeHeights()` — `LoopTree.hpp:177-201`, bottom-up, a leaf being 1.
    fn compute_subtree_heights(&mut self) {
        self.heights.clear();
        let root = self.root();
        self.record_subtree_height(root);
    }

    /// One post-order step of [`Self::compute_subtree_heights`].
    fn record_subtree_height(&mut self, n: LoopNodeId) -> SubtreeHeight {
        let mut tallest = 0;
        let mut child = self.first_child(n);
        let leaf = child.is_none();
        while let Some(c) = child {
            tallest = tallest.max(self.record_subtree_height(c).0);
            child = self.next_sibling(c);
        }
        // ⛔ `saturating_add` BECAUSE `unsigned short` WRAPS AND THIS MUST NOT: a nest 65,535 deep is
        // not a program, and a wrapped height would compare EQUAL to a shallow one in e068/e069.
        let height = SubtreeHeight(if leaf { 1 } else { tallest.saturating_add(1) });
        self.heights.insert(n, height);
        height
    }
}

impl LoopTree<true> {
    /// `getSubtreeHeightOf(n)` — `LoopTree.hpp:96-100`, `loop_node_to_subtree_height_.at(n)`.
    ///
    /// ⛔ `None` IS "NOT A NODE OF THIS FOREST", NOT A REFUSAL: the map is filled for every node the
    /// walk reached, so a `LoopNodeId` this tree minted always answers. The C++ `.at()` throws
    /// instead, and its key is a raw pointer that `remove()` has already `delete`d.
    ///
    /// ⭐ THE HEIGHTS ARE THE ONES `compute` MEASURED, deliberately not recomputed on
    /// [`Self::remove`] — `absorptionAnalysis` absorbs siblings and removes their nodes inside its own
    /// loop (`LoopAbsorption.cpp:455`, `:477`) and then asks e068/e069 again, so the answers it acts on
    /// are the pre-absorption ones.
    #[must_use]
    pub fn subtree_height_of(&self, n: LoopNodeId) -> Option<SubtreeHeight> {
        self.heights.get(&n).copied()
    }
}

/// `unit->walk<PreOrder>([&](LoopOp op) { .. })` — `LoopTree.hpp:149-163`.
fn collect(tree: &mut OperationTreeBase<LoopNode>, parent: LoopNodeId, scope: &[Op]) {
    for op in scope {
        match op {
            Op::Sentient(sentient::Op::For { iv, body, .. }) => {
                let node = LoopNodeId(tree.push_child(parent.0, LoopNode::Loop(ForRef(*iv))));
                collect(tree, node, body);
            }
            // ⭐ A NON-LOOP REGION DOES NOT DEEPEN THE TREE: `getParentOfType<LoopOp>()` skips it, so a
            // loop inside a `sentient.if` is a child of the loop AROUND the `if`.
            Op::Sentient(inner) => {
                for region in sentient::regions(inner) {
                    collect(tree, parent, region);
                }
            }
            Op::AffineFor(loop_op) => collect(tree, parent, &loop_op.body),
            // ⭐ A LOCAL REGION AT THIS RUNG CAN HOLD ONE, and it is not a loop itself.
            Op::UniformRegions(regions) => {
                for region in regions.regions() {
                    collect(tree, parent, &region.body);
                }
            }
            // ⛔ A SHARED-DIALECT REGION CANNOT HOLD A `sentient.for` AT ALL: its body is typed
            // `Vec<`[`crate::islands::dataflow_ir::dialects::Op`]`>`, the rung below, which has no
            // sentient arm.
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
    }
}
