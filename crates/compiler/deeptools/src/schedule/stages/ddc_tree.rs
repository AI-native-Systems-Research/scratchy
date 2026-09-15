// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ `currDsc->scheduleTree_` AS STAGE 2B WALKS IT — the EXCLUSIVE half
//! [`v1::Dsc2Store::split`] hands out, over the SAME [`super::DscTree`] stage 2a grew.
//!
//! ⛔ THE TREE IS NOT COPIED. [`Dsc2Tree`] holds `&'s Dsc2State`, and the state's tree is
//! [`super::DscState`]'s — so a node stage 2a minted is a node stage 2b walks, and a node stage 2b
//! mints is one a caller measuring [`super::DscState::kinds`] afterwards counts. Giving this view a
//! tree of its own would compile and silently drop every one of stage 2a's twenty-two nodes.
//!
//! ⭐ MOST OF THIS SURFACE IS ANSWERED FROM REAL DATA, because [`super::tree::TreeData`] already
//! models `nodeType_`, `prev_`, `next_`, `name_`, the `LOOP` staging pair, the transfer bodies and
//! the `(AllocId, NodeId)` pairing. What is NOT answered is every method whose fact lives on a
//! `dsc2::` node field this crate's tree does not carry — `isParametricLoop()`,
//! `parametricStride`, `getNextView`, `isNodeRelevant`, `repetitionWithOffset_` — and each of those
//! is a `todo!` naming the field and its `dsc/dsc2.h` line.

use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::SenComponent;

use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::schedule::ddc::fold::{AllocId, BlockId, NodeId, NodeKind, PadType};
use crate::schedule::ddc::metadata::MetaDimKind;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{ComputeNode, ConditionNode, NodeName, TransferNode};
use crate::units::Corelet;

use super::ddc_state::Dsc2State;
use super::state::DscTree;

/// ⭐ ONE DSC'S SCHEDULE TREE, AS STAGE 2B'S EXCLUSIVE VIEW OF IT.
#[derive(Debug)]
pub struct Dsc2Tree<'s, 'l> {
    state: &'s Dsc2State<'l>,
    dsc: crate::schedule::l3::dsc::DscIdx,
}

impl<'s, 'l> Dsc2Tree<'s, 'l> {
    /// The view over that DSC's tree.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>, dsc: crate::schedule::l3::dsc::DscIdx) -> Self {
        Self { state, dsc }
    }

    /// The tree itself — ⛔ TOTAL BY CONSTRUCTION: [`super::Dsc2Sites::carriers`] answers [`None`]
    /// for a `dscs_` position the state holds no tree for, so a view that exists names one.
    fn tree(&self) -> &'s DscTree {
        self.state
            .tree(self.dsc)
            .expect("a Dsc2Tree is only built for a DSC the state holds a tree for")
    }
}

/// ⭐ `nodeType_`, `getPrev()` AND `getNextView(ALL)` — the three-method view [`BlockId::of`] is
/// keyed by, so a `BLOCK` this tree hands out is one the tree itself proved.
impl crate::schedule::ddc::fold::ScheduleTree for Dsc2Tree<'_, '_> {
    /// `nodeType_` — ⛔ TOTAL by the trait's own signature; a node this tree does not hold is the
    /// reference's dereference of a dangling pointer, and [`NodeKind::Block`] is what the root is.
    fn kind(&self, node: NodeId) -> NodeKind {
        self.tree().with(|tree| {
            tree.node_kind(node).unwrap_or_else(|| {
                panic!("fold::ScheduleTree::kind: node {node:?} is not in this DSC's scheduleTree_")
            })
        })
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.tree().with(|tree| tree.parent(node))
    }

    fn children(&self, block: BlockId) -> Vec<NodeId> {
        self.tree().with(|tree| tree.children(block.node()))
    }
}

impl v1::ScheduleWalk for Dsc2Tree<'_, '_> {
    /// `traverseTreeDFS(from, {LOOP})`.
    fn loops_under(&self, from: BlockId) -> Vec<LoopId> {
        self.tree().with(|tree| {
            let mut found = Vec::new();
            let mut stack = vec![from.node()];
            while let Some(at) = stack.pop() {
                if tree.node_kind(at) == Some(NodeKind::Loop) {
                    found.push(LoopId(at));
                }
                stack.extend(tree.children(at).into_iter().rev());
            }
            found
        })
    }

    /// `traverseTreeDFS(nullptr, {kind})`.
    fn nodes_of_kind(&self, kind: NodeKind) -> Vec<NodeId> {
        self.tree().with(|tree| {
            tree.dfs()
                .into_iter()
                .filter(|node| tree.node_kind(*node) == Some(kind))
                .collect()
        })
    }

    /// `traverseTreeDFSMutable(nullptr, {ALLOCATE})` as the allocations those nodes name.
    fn allocates(&self) -> Vec<AllocId> {
        self.tree().with(|tree| {
            tree.dfs()
                .into_iter()
                .filter_map(|node| tree.allocate(node).map(|(alloc, _)| alloc))
                .collect()
        })
    }

    /// `traverseTreeDFS(from, {kind}, unit)` — ⛔ THE COMPONENT FILTER IS `isNodeRelevant(unit)`
    /// (`dsc/dsc2.h:470`), which reads `relevantComps_`, a field this tree does not carry; see
    /// [`v1::ScheduleNodes::is_relevant`].
    fn nodes_of_kind_under(
        &self,
        _from: Option<NodeId>,
        _kind: NodeKind,
        _unit: SenComponent,
    ) -> Vec<NodeId> {
        todo!(
            "v1::ScheduleWalk::nodes_of_kind_under: wants traverseTreeDFS(from, {{kind}}, unit), \
             whose component filter is isNodeRelevant(unit) reading relevantComps_ \
             (dsc/dsc2.h:516) — the map dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647) fills, and \
             which this tree does not carry"
        )
    }

    /// `allocNode->getPrev()` — the block, loop or region CONTAINING that ALLOCATE.
    fn prev_of_alloc(&self, alloc: AllocId) -> Option<NodeId> {
        self.tree()
            .with(|tree| tree.node_of_alloc(alloc).and_then(|node| tree.parent(node)))
    }

    /// `src_`/`dstVias_` as the node's own pairing.
    fn transfer(&self, node: NodeId) -> Option<TransferNode> {
        self.tree().with(|tree| tree.transfer(node))
    }

    /// `nodeType_ == LOOP` as the loop it then is.
    fn as_loop(&self, node: NodeId) -> Option<LoopId> {
        self.tree()
            .with(|tree| (tree.node_kind(node) == Some(NodeKind::Loop)).then_some(LoopId(node)))
    }

    /// `getPrev()` — ⚠️ THE PARENT, not the preceding sibling.
    fn prev(&self, node: NodeId) -> Option<NodeId> {
        self.tree().with(|tree| tree.parent(node))
    }

    /// `getOwnerLoop()`.
    fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
        self.tree().with(|tree| tree.owner_loop(node))
    }

    /// `LoopNode::dims_` (`dsc/dsc2.h:570`) — EMPTY for a node that is not a loop, which is what
    /// the reference's downcast then reads as nothing.
    fn loop_dims(&self, at: LoopId) -> Vec<(PrimaryDim, MetaDimKind)> {
        self.tree().with(|tree| {
            tree.loop_node(at)
                .map(|held| held.dims.iter().map(|pair| (pair.dim, pair.kind)).collect())
                .unwrap_or_default()
        })
    }
}

impl v1::ExploreTree for Dsc2Tree<'_, '_> {
    /// `traverseTreeDFSMutable()` — EVERY node, in DFS order.
    fn all_nodes(&self) -> Vec<NodeId> {
        self.tree().with(super::tree::TreeData::dfs)
    }

    /// `nodeType_` as the arm it is.
    fn tree_node(&self, node: NodeId) -> Option<v1::TreeNode> {
        self.tree().with(|tree| match tree.allocate(node) {
            Some((alloc, _)) => Some(v1::TreeNode::Allocate(alloc)),
            None => tree.node_kind(node).map(v1::TreeNode::Other),
        })
    }

    /// `nodeType_ == BLOCK` as the block it then is.
    fn as_block(&self, node: NodeId) -> Option<BlockId> {
        (self.tree().with(|tree| tree.node_kind(node)) == Some(NodeKind::Block))
            .then(|| BlockId::of(self, node))
            .flatten()
    }

    /// `node->name_`.
    fn node_name(&self, node: NodeId) -> Option<NodeName> {
        self.tree().with(|tree| tree.name(node))
    }

    /// `numId_` and `denId_` — ⭐ BOTH PRESENT, because [`super::tree::TreeData`] holds a
    /// [`crate::schedule::ddc::transformation_util::LoopNode`] whose pair is non-optional; the
    /// reference's `-1` is a state an unbuilt loop carries and no loop of this tree is unbuilt.
    fn loop_staging(&self, at: LoopId) -> Option<v1::LoopStaging> {
        self.tree().with(|tree| {
            tree.loop_node(at).map(|held| v1::LoopStaging {
                num: Some(held.num),
                den: Some(held.den),
            })
        })
    }

    /// ⛔ `isParametricLoop()` (`dsc/dsc2.h:596`) reads `LoopNode::parametricLdsIdx_` and its
    /// `numChunks_`, neither of which [`crate::schedule::ddc::transformation_util::LoopNode`]
    /// carries — a loop this tree mints is never parametric and a `false` here would state that as
    /// a FACT rather than as an absence.
    fn is_parametric_loop(&self, _at: LoopId) -> bool {
        todo!(
            "v1::ExploreTree::is_parametric_loop: wants isParametricLoop() (dsc/dsc2.h:596), which \
             reads LoopNode::parametricLdsIdx_ — a field transformation_util::LoopNode does not \
             carry"
        )
    }

    /// `dims_ = dims` — entry 307's `std::stable_sort` writes the whole list back.
    ///
    /// ⛔ A LOOP HAS AT LEAST ONE DIM ([`crate::schedule::ddc::transformation_util::LoopDims`] makes
    /// *"Cannot construct loop with no dimensions"* unspellable), so an EMPTY write is dropped
    /// rather than made — which is the one state the reference's `dims_` can hold and this type
    /// cannot.
    fn set_loop_dims(&mut self, at: LoopId, dims: Vec<(PrimaryDim, MetaDimKind)>) {
        use crate::schedule::ddc::transformation_util::{LoopDims, PrimaryDimAndKind};
        let mut stated = dims
            .into_iter()
            .map(|(dim, kind)| PrimaryDimAndKind { dim, kind });
        let Some(first) = stated.next() else {
            return;
        };
        let held = LoopDims::new(first, stated.collect());
        self.tree().with_mut(|tree| tree.set_loop_dims(at, held));
    }

    /// The allocation's node.
    fn alloc_node(&self, alloc: AllocId) -> Option<NodeId> {
        self.tree().with(|tree| tree.node_of_alloc(alloc))
    }

    /// ⛔ `allocNode->padding_.getPadding(dim)` (`dsc/dims.cpp:806`) — ⚠️ A DIFFERENT FIELD FROM
    /// `paddingSizes_`. [`crate::schedule::l3::dl_ops::L3AllocateNode::padding`] holds a
    /// [`crate::schedule::ddc::transformation_util::PaddingForm`], and reading a [`PadType`] off it
    /// is `getPadding`, which is `dsc/dims.cpp` and not this batch's.
    fn alloc_padding(&self, _alloc: AllocId, _dim: PrimaryDim) -> PadType {
        todo!(
            "v1::ExploreTree::alloc_padding: wants allocNode->padding_.getPadding(dim) \
             (dsc/dims.cpp:806) — PaddingForm's own accessor, a `dsc/` seam, and NOT paddingSizes_"
        )
    }

    /// `traverseTreeDFSMutable(from, {TRANSFER, COMPUTE}, ALL, -1, -1)` — ⛔ ONE DFS OVER BOTH
    /// KINDS AND NOT TWO, because `updateSizePerDim`'s multiple check is order-dependent.
    fn transfers_and_computes_under(&self, from: NodeId) -> Vec<NodeId> {
        self.tree().with(|tree| {
            let mut found = Vec::new();
            let mut stack = vec![from];
            while let Some(at) = stack.pop() {
                if matches!(
                    tree.node_kind(at),
                    Some(NodeKind::Transfer | NodeKind::Compute)
                ) {
                    found.push(at);
                }
                stack.extend(tree.children(at).into_iter().rev());
            }
            found
        })
    }

    /// ⛔ `computeNode->isOpaqueOp_` (`dsc/dsc2.h:531`) — a COMPUTE-node field. This tree holds no
    /// COMPUTE nodes at all yet: stage 2a mints none (the census over 24,363 programs reads
    /// `compute: 0`) and entry 345's parse is what mints them, so there is nothing here to read the
    /// flag off.
    fn is_opaque_compute(&self, _node: NodeId) -> bool {
        todo!(
            "v1::ExploreTree::is_opaque_compute: wants computeNode->isOpaqueOp_ (dsc/dsc2.h:531) — \
             a COMPUTE-node field, and super::tree::Kind has no Compute arm"
        )
    }

    /// ⛔ `getBlockTransferSizePerDim(transfer, unit, 0, false, false, true)` — a
    /// `DesignSpaceConfig` accessor over the transfer's own layout and the DSC's stick sizes
    /// (`dsc/dsc2.cpp`), not a field of the node.
    fn block_transfer_sizes(
        &self,
        _node: NodeId,
        _unit: SenComponent,
    ) -> BTreeMap<PrimaryDim, Extent> {
        todo!(
            "v1::ExploreTree::block_transfer_sizes: wants \
             getBlockTransferSizePerDim(transfer, unit, 0, false, false, true) — a \
             DesignSpaceConfig accessor over the live super-DSC's stick sizes, a `dsc/` seam"
        )
    }

    /// ⛔ `computeOp_`'s entry for a COMPUTE node — the pairing between a schedule node and a
    /// `computeOp_` index, which the reference holds on the node (`ComputeNode::opIdx_`) and this
    /// tree does not carry.
    fn compute_op(&self, _node: NodeId) -> Option<v1::DscComputeOp> {
        todo!(
            "v1::ExploreTree::compute_op: wants computeOp_'s entry for that COMPUTE node — the \
             node-to-op-index pairing, which super::tree::Kind has no Compute arm to hold"
        )
    }

    /// The transfer written back after entry 307 has shrunk its unit-time chunks.
    fn set_transfer(&mut self, node: NodeId, transfer: TransferNode) -> Option<()> {
        self.tree().with_mut(|tree| {
            tree.node_kind(node)?;
            tree.set_transfer(node, transfer);
            Some(())
        })
    }
}

impl v1::ScheduleNodes for Dsc2Tree<'_, '_> {
    fn nodes(&self) -> Vec<NodeId> {
        self.tree().with(super::tree::TreeData::dfs)
    }

    fn kind(&self, node: NodeId) -> Option<NodeKind> {
        self.tree().with(|tree| tree.node_kind(node))
    }

    /// ⛔ THE SAME FIELD [`v1::ExploreTree::is_parametric_loop`] WANTS.
    fn is_parametric(&self, _at: LoopId) -> bool {
        todo!(
            "v1::ScheduleNodes::is_parametric: wants isParametricLoop() (dsc/dsc2.h:596) reading \
             LoopNode::parametricLdsIdx_, which transformation_util::LoopNode does not carry"
        )
    }

    /// `numId_` and `denId_` — ⛔ TOTAL by the trait's own statement, and every [`LoopId`] a ported
    /// unit holds came out of THIS tree's own walk.
    fn loop_stages(&self, at: LoopId) -> v1::LoopStages {
        self.tree().with(|tree| {
            let held = tree
                .loop_node(at)
                .expect("a LoopId this tree's own walk produced names a loop of this tree");
            v1::LoopStages {
                num: held.num,
                den: held.den,
            }
        })
    }

    /// ⛔ `parametricStride(dsc)` — the parametric-loop arithmetic behind the field above.
    fn parametric_stride(&self, _at: LoopId) -> v1::LoopEleOffset {
        todo!(
            "v1::ScheduleNodes::parametric_stride: wants parametricStride(dsc) on that loop — \
             reads LoopNode::parametricLdsIdx_ against the DSC's layout, a `dsc/` seam"
        )
    }

    /// ⛔ `parametricIterCount(dsc, corelet, unit)` — likewise.
    fn parametric_iter_count(
        &self,
        _at: LoopId,
        _corelet: Corelet,
        _unit: SenComponent,
    ) -> v1::IterCount {
        todo!(
            "v1::ScheduleNodes::parametric_iter_count: wants parametricIterCount(dsc, corelet, \
             unit) on that loop — a `dsc/` seam over LoopNode::parametricLdsIdx_"
        )
    }

    /// ⛔ `isNodeRelevant(unit)` (`dsc/dsc2.h:470`) reads `relevantComps_`, which
    /// `dsc.setRelevantCompCoreCl()` (`dsc/dsc2.cpp:2647`) fills — a `dsc/` seam this tree does not
    /// hold, and the SAME map [`v1::ConditionSimplification::relevant_comps`] wants.
    fn is_relevant(&self, _node: NodeId, _unit: SenComponent) -> bool {
        todo!(
            "v1::ScheduleNodes::is_relevant: wants isNodeRelevant(unit) (dsc/dsc2.h:470) reading \
             relevantComps_ (dsc/dsc2.h:516), which dsc.setRelevantCompCoreCl() \
             (dsc/dsc2.cpp:2647) fills"
        )
    }

    /// ⛔ `getNextView(unit).size()` — the per-component child view, which is `next_` FILTERED by
    /// `isNodeRelevant`, so it wants the same `relevantComps_`.
    fn next_view_len(&self, _node: NodeId, _unit: SenComponent) -> usize {
        todo!(
            "v1::ScheduleNodes::next_view_len: wants getNextView(unit).size() — next_ filtered by \
             isNodeRelevant(unit), so it wants relevantComps_ (dsc/dsc2.h:516)"
        )
    }

    /// `allocNode->getOwnerLoop()`.
    fn alloc_owner_loop(&self, alloc: AllocId) -> Option<LoopId> {
        self.tree()
            .with(|tree| tree.owner_loop(tree.node_of_alloc(alloc)?))
    }

    /// `TransferNode::paddingInfo_.isEmpty() == false` (`dsc/dsc2.h:838`) — the transfer's OWN
    /// padding, which [`crate::schedule::dsc2::TransferNode::padding`] carries.
    fn transfer_has_padding(&self, node: NodeId) -> bool {
        self.tree().with(|tree| {
            tree.transfer(node)
                .is_some_and(|held| held.padding.dims().next().is_some())
        })
    }

    /// ⛔ NO COMPUTE NODES IN THIS TREE — see [`v1::ExploreTree::is_opaque_compute`].
    fn compute(&self, _node: NodeId) -> Option<ComputeNode> {
        todo!(
            "v1::ScheduleNodes::compute: wants type_, exUnit_, inputs_ and outputs_ each zipped \
             with its offsets vector off that COMPUTE node — super::tree::Kind has no Compute arm"
        )
    }

    /// ⛔ `repetitionWithOffset_.forOutputs_.size()` (`dsc/dsc2.h:950-953`) — a COMPUTE-node field.
    fn repetition_with_offset_outputs(&self, _node: NodeId) -> usize {
        todo!(
            "v1::ScheduleNodes::repetition_with_offset_outputs: wants \
             repetitionWithOffset_.forOutputs_.size() (dsc/dsc2.h:950-953) on that COMPUTE node"
        )
    }
}

impl v1::MaskInsertion for Dsc2Tree<'_, '_> {
    /// ⛔ `before->prev_->addChildNode(cond, true, before)` — the CONDITION node itself is a
    /// [`ConditionNode`] with its two regions, and [`super::tree::Kind::Condition`] holds a
    /// `loopCond_`/`coreClCond_` pair rather than the whole `dsc2::ConditionNode`; splicing one in
    /// means translating a `StickMaskNode` subtree this tree has no arm for.
    fn insert_condition_before(&mut self, _before: NodeId, _cond: ConditionNode) -> Option<()> {
        todo!(
            "v1::MaskInsertion::insert_condition_before: wants \
             before->prev_->addChildNode(cond, true, before) with a whole dsc2::ConditionNode — \
             its regions carry a StickMaskNode, which super::tree::Kind has no arm for"
        )
    }
}
