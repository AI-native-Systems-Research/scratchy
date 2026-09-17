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
//! models `nodeType_`, `prev_`, `next_`, `name_`, the `LOOP` staging pair, the transfer bodies, the
//! COMPUTE and STICKMASK nodes, the two condition regions and the `(AllocId, NodeId)` pairing.
//!
//! ⛔⛔ WHAT IS LEFT IS EXACTLY TWO FACTS, AND NEITHER HAS A HOME ANYWHERE IN THIS CRATE:
//!
//! 1. **`relevantComps_`** (`dsc/dsc2.h:516`) — the per-node `comp -> core -> corelets` map
//!    `dsc.setRelevantCompCoreCl()` (`dsc/dsc2.cpp:2647-2729`) fills in ONE whole-tree propagation.
//!    That call is itself a `todo!` in [`super::Dsc2Store`], and it has nowhere to write: the map is
//!    a FIELD OF EVERY NODE and [`super::tree::TreeData`]'s entries carry no slot for it. Three
//!    methods here want it — [`v1::ScheduleWalk::nodes_of_kind_under`],
//!    [`v1::ScheduleNodes::is_relevant`] and [`v1::ScheduleNodes::next_view_len`] — and so does
//!    [`v1::ConditionSimplification::relevant_comps`] in [`super::Dsc2Store`].
//! 2. **`parametricLdsIdx_`** (`dsc/dsc2.h:618`) with the DSC-side arithmetic over it
//!    (`parametricStride`, `parametricIterCount`, `dsc/dsc2.cpp:4126-4220`), which reads
//!    `labeledDs_`, `getCumulativeStickSizes` and `dataStageParam_` — a `dsc/` seam this TREE view
//!    holds none of.
//!
//! ⭐ AND TWO MORE THAT ARE ANSWERABLE ONLY BY THEIR CALLER, NOT BY A TREE:
//! [`v1::ExploreTree::is_opaque_compute`] (this crate models `isOpaqueOp_` as membership in
//! `Metadata::opaque_ops`, which the caller already holds one line away) and
//! [`v1::ExploreTree::compute_op`], whose return type describes `computeOp_` and not a node.
//! [`v1::ScheduleNodes::repetition_with_offset_outputs`] wants a field the DDL conversion drops.
//! Each `todo!` below names its field, its authority line and what would answer it.

use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::SenComponent;

use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::schedule::ddc::fold::{AllocId, BlockId, NodeId, NodeKind, PadType};
use crate::schedule::ddc::metadata::MetaDimKind;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::InsertionPoint;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{
    ComputeNode, CondRegions, ConditionNode, NodeName, SchedNode, TransferNode,
};
use crate::units::Corelet;

use super::ddc_state::Dsc2State;
use super::state::DscTree;
use super::tree::{Cond, Kind, TreeData};

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

    /// `traverseTreeDFS(from, {kind}, unit)` — ⛔ THE COMPONENT FILTER IS LOAD-BEARING AND CANNOT BE
    /// DROPPED. `traverseTreeDFS` calls `isNodeRelevant(comp, -1, -1)` on EVERY node it reaches and
    /// `continue`s past a node that answers false WITHOUT descending into it (`dsc/dsc2.cpp:2245-2247`),
    /// so an unfiltered walk is a different set and not a superset-with-noise.
    ///
    /// ⛔ AT `clId = coreId = -1` THAT PREDICATE REDUCES TO `relevantComps_.count(comp) != 0`
    /// (`dsc/dsc2.cpp:1923-1926`), and the ONE live caller passes a real component — `L0SU`
    /// (`ddc/v1.rs:3326`) — so the `comp == ALL` short-circuit that would make this total never fires.
    /// `relevantComps_` is fact 1 of this file's header.
    fn nodes_of_kind_under(
        &self,
        _from: Option<NodeId>,
        _kind: NodeKind,
        _unit: SenComponent,
    ) -> Vec<NodeId> {
        todo!(
            "v1::ScheduleWalk::nodes_of_kind_under: wants traverseTreeDFS(from, {{kind}}, unit), \
             whose per-node filter is isNodeRelevant(comp, -1, -1) == relevantComps_.count(comp) \
             (dsc/dsc2.cpp:1923-1926, field dsc/dsc2.h:516) — the map \
             dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647-2729) fills, which nothing fills and no \
             node of this tree carries a slot for"
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

    /// `LoopNode::dims_` (`dsc/dsc2.h:575` — ⚠️ NOT `:570`, which is inside the COMMENTED-OUT
    /// `PrimaryDimAndKind` block above it) — EMPTY for a node that is not a loop, which is what the
    /// reference's downcast then reads as nothing.
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

    /// ⭐⭐ `isParametricLoop()` — ANSWERED `false`, AND THAT IS THE AUTHORITY'S OWN VALUE, NOT AN
    /// ABSENCE DRESSED AS ONE.
    ///
    /// ⛔ THE CITATION THIS DOC USED TO CARRY WAS WRONG TWICE. `isParametricLoop()` is
    /// `dsc/dsc2.h:599` and it returns `isParametricLoop_` — a PLAIN BOOL whose member initializer is
    /// `false` (`dsc/dsc2.h:617`). It does not read `parametricLdsIdx_` (that is
    /// `parametricLdsIdx()`, `:603`) and `numChunks_` appears nowhere in `LoopNode` at all.
    ///
    /// ⭐ AND EXACTLY TWO THINGS IN THE WHOLE REFERENCE EVER SET IT — `markAsParametricLoop()`
    /// (`:600`), called ONLY from `ParametricLoopOp`'s arm of the DDL conversion
    /// (`ddc/ddl/ddl_conversion.cpp:1128`), and the JSON importer reading a serialised super-DSC
    /// back (`dsc/dsc2.cpp:1412`). Every loop the ddc stages mint goes through
    /// `Ddc::constructLoopNode`, which is `new dsc2::LoopNode()` (`ddc/ddc_transformation_util.cpp:144`)
    /// and so takes the `false`.
    ///
    /// ⭐⭐ THE PRODUCER PROVES IT ON THIS SIDE TOO: a parametric loop carries `numId_ = denId_ = -1`
    /// (`ddl_conversion.cpp:1129-1130`), which [`crate::schedule::ddc::transformation_util::LoopNode`]
    /// cannot spell — its pair is non-optional — and [`super::Dsc2Store`]'s `add_loop` REFUSES a
    /// `parametric_lds.is_some()` loop rather than dropping the flag. So no `Kind::Loop` of this tree
    /// can be a parametric one. This is the same fact [`super::Dsc2Store`]'s `schedule_head_block`
    /// already states and `stages.rs`' *"a loop stage 2a minted is not parametric"* already asserts.
    fn is_parametric_loop(&self, _at: LoopId) -> bool {
        false
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

    /// ⭐⭐ `allocNode->padding_.getPadding(dim)` (`dsc/dims.cpp:806-812`) — ANSWERED, AND THE
    /// OBJECTION THIS DOC USED TO CARRY NAMED THE WRONG SEAM.
    ///
    /// ⭐ `getPadding` IS ALREADY PORTED, on that same citation:
    /// [`crate::schedule::ddc::transformation_util::PaddingForm::padding`], and the SECOND Rust
    /// spelling of `padding_` this doc used to name is GONE — [`super::Dsc2Sites`]' `AllocArena` seed
    /// carries the form whole now, so the arena answers this rather than refusing it.
    ///
    /// ⭐ AND THE NODE HERE IS THE NODE THE REFERENCE READS: `ddcv1.cpp:677-681` takes `component_`,
    /// `ldsIdx_` and `padding_` off the SAME `an` that the `{ALLOCATE}` walk yielded, and
    /// [`super::tree::Kind::Allocate`] carries exactly that node as an
    /// [`crate::schedule::l3::dl_ops::L3AllocateNode`], whose `padding` IS `padding_`.
    ///
    /// ⚠️ STILL A DIFFERENT FIELD FROM `paddingSizes_`, which is the DATASTAGE's padding —
    /// [`v1::StageSizes::stage_padding_sizes`] — and which the one caller (`ddc/v1.rs:4427-4436`)
    /// reads BESIDE this one, exactly as `ddcv1.cpp:679-683` does.
    ///
    /// ⛔ THE OLD LINK HERE NAMED `v1::ExploreStages::alloc_padding_sizes`, WHICH WAS WRONG TWICE:
    /// that trait never declared the method (it was on `StageSizes`), and the method itself named a
    /// field that does not exist — `grep paddingSizes_ dsc/dsc2.h` is zero hits. It is now deleted.
    fn alloc_padding(&self, alloc: AllocId, dim: PrimaryDim) -> PadType {
        self.tree().with(|tree| alloc_padding_of(tree, alloc, dim))
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

    /// ⛔⛔ `computeNode->isOpaqueOp_` (`dsc/dsc2.h:941`) — AND THE BLOCKER IS NO LONGER THE TREE.
    /// [`super::tree::Kind::Compute`] holds the whole [`ComputeNode`] now, but
    /// [`crate::schedule::dsc2::ComputeNode`] carries NO SUCH FIELD, deliberately: **this crate models
    /// `isOpaqueOp_` as membership in `Metadata::opaque_ops`** — `schedule/ddl/conversion.rs:3708`
    /// says so in as many words, and it is faithful, because the reference sets the flag at exactly
    /// one place (`ddc/ddl/ddl_conversion.cpp:1575`) and in the same breath inserts
    /// `metadata_.opaqueOps_[newNode]` (`:1578`).
    ///
    /// ⛔ SO A TREE CANNOT ANSWER IT AND ITS CALLER ALREADY CAN. `ddc/v1.rs:4914-4915` asks this and
    /// then, on the very next line, reads `self.metadata.opaque_ops.get(&child)` — the same map, one
    /// line away. The reference's own reader does both off `metadata` too
    /// (`ddc/ddcv1.cpp:1101-1102`: `if (compute->isOpaqueOp_) { … metadata.opaqueOps_.at(compute) }`).
    /// ⭐ ANSWERING THIS MEANS THE CALLER TESTING `opaque_ops.contains_key(&child)` — a change in
    /// `ddc/v1.rs`, not here — or `Metadata` being handed to this carrier, which stage 2b's own
    /// signature does not do.
    fn is_opaque_compute(&self, _node: NodeId) -> bool {
        todo!(
            "v1::ExploreTree::is_opaque_compute: wants computeNode->isOpaqueOp_ (dsc/dsc2.h:941), \
             whose ported home is membership in Metadata::opaque_ops (ddl/conversion.rs:3708, \
             metadata.rs:945) — a map this TREE view does not hold and the caller at ddc/v1.rs:4915 \
             already reads one line later"
        )
    }

    /// ⛔ `getBlockTransferSizePerDim(transfer, unit, 0, false, false, true)` (`dsc/dsc2.cpp:3452`)
    /// — a `DesignSpaceConfig` METHOD, not a field of the node, and one that forwards to
    /// `getBlockTransferSizePerDimCustomLocation` (`:3472-3600`): it branches on
    /// `getTransferType()`, then reads `constantInfo_`, `labeledDs_`, the stick sizes and the buffer
    /// capacities off the DSC. A TREE view holds none of that, which is why the seam is unanswerable
    /// HERE rather than merely unported — it belongs on the DSC-side carrier beside the other
    /// `getBlockTransferSize*` reads.
    fn block_transfer_sizes(
        &self,
        _node: NodeId,
        _unit: SenComponent,
    ) -> BTreeMap<PrimaryDim, Extent> {
        todo!(
            "v1::ExploreTree::block_transfer_sizes: wants \
             getBlockTransferSizePerDim(transfer, unit, 0, false, false, true) \
             (dsc/dsc2.cpp:3452 -> :3472-3600) — a DesignSpaceConfig method over constantInfo_, \
             labeledDs_ and the DSC's stick sizes, none of which a tree view holds"
        )
    }

    /// ⛔⛔ THERE IS NO SUCH PAIRING, AND THIS DOC USED TO INVENT ONE. It claimed the reference
    /// *"holds the pairing on the node (`ComputeNode::opIdx_`)"*. **`opIdx_` appears ZERO times in
    /// `dsc/dsc2.h`**, and `computeOp_` is a `DesignSpaceConfig` member, not a field of `ComputeNode`
    /// (`dsc2.h:900-962`) — the node has no back-pointer at all. The reference's only reader of what
    /// this seam is for (`ddc/ddcv1.cpp:1092-1113`) takes `compute->exUnit_` and
    /// `compute->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_` STRAIGHT OFF THE NODE.
    ///
    /// ⛔ SO THE BLOCKER IS NOW THE RETURN TYPE, NOT THE TREE. [`super::tree::Kind::Compute`] holds the
    /// whole [`ComputeNode`], so BOTH facts the reference reads are in hand:
    /// `compute->exUnit_` is `ex_unit`, and `outputsLdsAndLoopOffsets_` filtered by
    /// `if (outputInfo.myLdsIdx_ < 0) continue` (`ddc/ddcv1.cpp:1107-1108`) is
    /// `outputs.iter().filter_map(|o| o.data.my_lds_idx)`.
    ///
    /// ⛔⛔ WHAT CANNOT BE ANSWERED IS THE OTHER THREE FIELDS, BECAUSE [`v1::DscComputeOp`] DESCRIBES A
    /// DIFFERENT C++ OBJECT: `opFuncName`, `attributes_.dataFormat_`, `inputLabeledDs` and
    /// `outputLabeledDs` are `DesignSpaceConfig::computeOp_` entries — which is why the OTHER provider
    /// of this type ([`v1::ExploreTree::compute_ops`], answered in [`super::Dsc2Store`]) reads the DSC's
    /// op list and not a node. `ComputeNode` has no `opFuncName`; its `op` is `type_`, a
    /// [`crate::schedule::ddl::ops::DdlComputeType`] and a different closed set. A `None`
    /// `op_func` here would say *"this compute has no op func"* about a compute that has one, and that
    /// is the wrong opcode the backend lowers happily.
    ///
    /// ⭐ ANSWERING THIS MEANS NARROWING THE RETURN in `ddc/v1.rs` to the two fields the reference and
    /// both live readers (`:4912`, `:4917`) actually touch — not synthesising the other three here.
    fn compute_op(&self, _node: NodeId) -> Option<v1::DscComputeOp> {
        todo!(
            "v1::ExploreTree::compute_op: wants `exUnit_` and `outputsLdsAndLoopOffsets_` off the \
             COMPUTE node (ddc/ddcv1.cpp:1092-1113) — both now held by tree::Kind::Compute, but \
             DscComputeOp models DesignSpaceConfig::computeOp_ and also carries \
             op_func/format/inputs, which a dsc2::ComputeNode does not have; narrow the return in \
             ddc/v1.rs before answering this"
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

    /// ⭐ THE SAME `isParametricLoop_` [`v1::ExploreTree::is_parametric_loop`] ANSWERS, and answered
    /// the same way and for the same reasons — one fact, one value, read that doc.
    fn is_parametric(&self, at: LoopId) -> bool {
        v1::ExploreTree::is_parametric_loop(self, at)
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

    /// ⛔⛔ `parametricStride(dsc)` (`dsc/dsc2.cpp:4197-4221`) — THE ONE FIELD THIS FILE GENUINELY DOES
    /// NOT HAVE, `parametricLdsIdx_` (`dsc/dsc2.h:618`), read against `labeledDs_.at(ldsIdx).dsType_`
    /// and `getCumulativeStickSizes` — a DSC the TREE view does not hold, on top of a field
    /// [`crate::schedule::ddc::transformation_util::LoopNode`] does not carry. Its own
    /// `DT_CHECK_MSG(ldsIdx != -1)` is what a fabricated stride would walk past.
    ///
    /// ⭐ AND IT IS UNREACHABLE WHILE [`Self::is_parametric`] ANSWERS `false`: every reader guards on
    /// that first (`ddc/v1.rs:2567-2570`, `l3/dl_ops.rs:17928-17929`), exactly as the reference does
    /// (`ddc/ddcv1.cpp:2467-2472`), so this `todo!` is a stop no program of this tree can reach — and
    /// the one that WOULD reach it is a program whose loop the tree could not have minted.
    fn parametric_stride(&self, _at: LoopId) -> v1::LoopEleOffset {
        todo!(
            "v1::ScheduleNodes::parametric_stride: wants parametricStride(dsc) \
             (dsc/dsc2.cpp:4197-4221) — LoopNode::parametricLdsIdx_ (dsc/dsc2.h:618) against \
             labeledDs_ and getCumulativeStickSizes, neither of which a tree view holds; \
             unreachable while is_parametric answers false"
        )
    }

    /// ⛔⛔ `parametricIterCount(dsc, corelet, unit)` (`dsc/dsc2.cpp:4126-4193`) — LIKEWISE, and
    /// deeper: it opens with `DT_ERROR` on a non-parametric loop or one with more than one dim, then
    /// walks `getOwnerLoop()` for a reference datastage and reads that stage's `paddingSizes_`,
    /// `symbolicDimInfo_` and `primaryDimToVal_st`, before dividing by
    /// [`Self::parametric_stride`]. Same two absences, same guarded callers.
    fn parametric_iter_count(
        &self,
        _at: LoopId,
        _corelet: Corelet,
        _unit: SenComponent,
    ) -> v1::IterCount {
        todo!(
            "v1::ScheduleNodes::parametric_iter_count: wants parametricIterCount(dsc, corelet, \
             unit) (dsc/dsc2.cpp:4126-4193) — LoopNode::parametricLdsIdx_ plus the reference \
             datastage's paddingSizes_/primaryDimToVal_st, a `dsc/` seam; unreachable while \
             is_parametric answers false"
        )
    }

    /// ⛔ `isNodeRelevant(unit)` (`dsc/dsc2.h:470`, defined `dsc/dsc2.cpp:1916-1932`). At the
    /// `clId = coreId = -1` this seam is always called with, its whole body reduces to
    /// `comp == ALL || relevantComps_.count(comp) != 0` — so it is a ONE-LINE read of fact 1 of this
    /// file's header, and the only thing missing is the map. The SAME map
    /// [`v1::ConditionSimplification::relevant_comps`] wants in [`super::Dsc2Store`].
    ///
    /// ⛔ AND `true` IS NOT A SAFE DEFAULT. The reference's own default is `false` for an absent
    /// component, and the map is EMPTY until `setRelevantCompCoreCl()` runs — so either constant
    /// states a fact about every node, and `last_fusable_loop` (`ddc/v1.rs:3144`) turns that constant
    /// straight into which loop a transfer fuses under.
    fn is_relevant(&self, _node: NodeId, _unit: SenComponent) -> bool {
        todo!(
            "v1::ScheduleNodes::is_relevant: wants isNodeRelevant(unit) (dsc/dsc2.h:470, \
             dsc/dsc2.cpp:1916-1932), which at clId=coreId=-1 is \
             relevantComps_.count(comp) != 0 (dsc/dsc2.h:516) — the map \
             dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647-2729) fills, which nothing fills"
        )
    }

    /// ⛔ `getNextView(unit).size()` (`dsc/dsc2.cpp:1984-1992`) — `next_` FILTERED by
    /// `isNodeRelevant(comp, clId, coreId)`, so it wants the same `relevantComps_`. ⚠️ AND THE
    /// CONDITION OVERRIDE COMES WITH IT: `ConditionNode::getNextView` (`:1994-2001`) returns BOTH
    /// children unfiltered whenever `!hasCoreClCond()`, which is a second reader of the guard
    /// [`super::tree::Cond`] already holds.
    fn next_view_len(&self, _node: NodeId, _unit: SenComponent) -> usize {
        todo!(
            "v1::ScheduleNodes::next_view_len: wants getNextView(unit).size() \
             (dsc/dsc2.cpp:1984-1992) — next_ filtered by isNodeRelevant(unit), so it wants \
             relevantComps_ (dsc/dsc2.h:516), plus ConditionNode::getNextView's unfiltered override \
             for a loop-guarded condition (dsc/dsc2.cpp:1994-2001)"
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

    /// ⭐⭐ `type_`, `exUnit_`, `inputs_` AND `outputs_` EACH ZIPPED WITH ITS OFFSETS VECTOR —
    /// ANSWERED, because [`super::tree::Kind::Compute`] carries the whole
    /// [`ComputeNode`] and that node IS the zip. ⛔ THE ARM IS THE REFERENCE'S OWN
    /// `static_cast<dsc2::ComputeNode*>(node)` GUARDED BY `nodeType_ == COMPUTE`
    /// (`ddc/ddcv1.cpp:3038-3039`), so [`None`] for every other kind is that guard and not a refusal.
    /// ⭐ AND THE `"Compute node input/output missing information"` `DT_ERROR` THE REFERENCE RAISES ONE
    /// LINE LATER (`:3040-3043`) IS UNSPELLABLE THROUGH THIS TYPE: it fires when
    /// `outputsLdsAndLoopOffsets_.size() != outputs_.size()`, and an [`crate::schedule::dsc2::Operand`]
    /// is that pair.
    ///
    /// ⚠️ WHAT THIS ANSWERS AND `is_opaque_compute` DOES NOT: the flag has no field on the node here,
    /// only a home in `Metadata::opaque_ops`. The zip does — see that method's doc.
    ///
    /// ⛔ NOTHING MINTS A `Kind::Compute` INTO THIS TREE YET, so today this answers [`None`] for every
    /// node the corpus holds ([`super::tree::Kind`]'s own doc: `compute: 0` over 24,363 programs, and
    /// the DDL walk writes into a detached copy). That is a producer gap in
    /// [`super::Dsc2Store`]'s `schedule_head_block` seam, NOT a reason to fabricate a node here.
    fn compute(&self, node: NodeId) -> Option<ComputeNode> {
        self.tree().with(|tree| compute_of(tree, node))
    }

    /// ⛔⛔ `repetitionWithOffset_.forOutputs_.size()` (`dsc/dsc2.h:950-954`) — A FIELD WITH NO PORTED
    /// HOME ANYWHERE IN THIS CRATE, and the gap is upstream of this file.
    ///
    /// ⭐ THE REFERENCE FILLS IT AT EXACTLY ONE PLACE: `ComputeOp`'s arm of the DDL conversion pushes
    /// `getRepetitionIfExists(entry)` ONCE PER OUTPUT, in the same loop that pushes `outputs_`
    /// (`ddc/ddl/ddl_conversion.cpp:1395-1407`, the helper at `:858-869` reading a
    /// `ddl.allocate replication=`). ⛔ OUR `op_compute` (`schedule/ddl/conversion.rs:3426-3520`)
    /// pushes the operand and DROPS the repetition, and
    /// [`crate::schedule::dsc2::ComputeNode`] has no slot for it — so there is nothing to read, on
    /// the node or beside it.
    ///
    /// ⛔ AND `outputs.len()` IS NOT IT. The two are equal at fill time, but
    /// `insertComputeBetweenTransferAndReg` (`ddc/ddc_transformation_util.cpp:1074-1086`) resizes
    /// `outputs_` and `outputsLdsAndLoopOffsets_` to ONE and leaves `forOutputs_` alone, so answering
    /// one with the other states a length the reference can contradict — and this count drives which
    /// clone gets which const element offset (`ddc/ddcv1.cpp:3057-3081`).
    ///
    /// ⭐ WHAT ANSWERS IT: a `ReplicationFactor` per operand on
    /// [`crate::schedule::dsc2::ComputeNode`] — the newtype the TRANSFER side already carries for the
    /// same `getRepetitionIfExists` (`schedule/ddl/conversion.rs:3278`) — filled in that same output
    /// loop. Then this reads it off [`super::tree::Kind::Compute`].
    fn repetition_with_offset_outputs(&self, _node: NodeId) -> usize {
        todo!(
            "v1::ScheduleNodes::repetition_with_offset_outputs: wants \
             repetitionWithOffset_.forOutputs_.size() (dsc/dsc2.h:950-954) on that COMPUTE node — a \
             field dsc2::ComputeNode does not carry and ddl/conversion.rs's op_compute drops, whose \
             only filler is ddl_conversion.cpp:1405; NOT outputs.len(), which \
             ddc_transformation_util.cpp:1074-1086 can shrink out from under it"
        )
    }
}

impl v1::MaskInsertion for Dsc2Tree<'_, '_> {
    /// ⭐⭐ `lastSeenNode->prev_->addChildNode(cond, true, lastSeenNode)` (`ddc/ddcv1.cpp:3633-3634`)
    /// — ANSWERED, now that [`super::tree::Kind::StickMask`] exists to hold what the two regions carry.
    ///
    /// ⛔ `before` STAYS A SIBLING AND IS NOT MOVED INSIDE. The reference splices the condition in
    /// AHEAD of it under the same parent and then fills the regions with two freshly minted blocks
    /// (`:3651-3664`); a port that re-parented `before` under the condition would guard the whole
    /// subtree the mask must sit beside.
    ///
    /// ⛔ [`None`] IS `before->prev_ == nullptr` — the reference reaches this only through
    /// `insertionNodes.insert(lastSeenNode->prev_)` (`:3624`) and its loop stops before the root, so a
    /// parentless `before` is a state it never presents. Checking it FIRST is why no half-spliced
    /// condition can be left behind.
    fn insert_condition_before(&mut self, before: NodeId, cond: ConditionNode) -> Option<()> {
        self.tree()
            .with_mut(|tree| insert_condition_before_of(tree, before, &cond))
    }
}

/// `before->prev_->addChildNode(cond, /*addBefore*/ true, before)` (`ddc/ddcv1.cpp:3634`) — the whole
/// body of [`v1::MaskInsertion::insert_condition_before`], over the tree itself.
///
/// ⛔ THE PARENT IS CHECKED BEFORE ANYTHING IS MINTED, so a `before` at the root leaves no orphaned
/// condition behind — see that method's doc for why the reference never presents one.
fn insert_condition_before_of(
    tree: &mut TreeData,
    before: NodeId,
    cond: &ConditionNode,
) -> Option<()> {
    tree.parent(before)?;
    mint_condition(tree, cond, InsertionPoint::Before(before))?;
    Some(())
}

/// ⭐ `allocNode->padding_.getPadding(dim)` OFF THE ALLOCATE NODE THAT ALLOCATION NAMES —
/// [`crate::schedule::ddc::transformation_util::PaddingForm::padding`] on the node's own `padding_`.
///
/// ⛔ TOTAL EXACTLY WHERE THE REFERENCE IS: `getPadding` answers `NOPAD` for a dim the form does not
/// state (*"Default is NOPAD"*, `dsc/dims.cpp:808-810`), and an [`AllocId`] no ALLOCATE node holds
/// cannot arise — the reference reaches `an->padding_` only through a node the `{ALLOCATE}` walk
/// yielded (`ddc/ddcv1.cpp:673-681`), and every [`AllocId`] on this side comes out of
/// [`super::tree::TreeData`]'s own `(alloc, node)` pairing.
fn alloc_padding_of(tree: &TreeData, alloc: AllocId, dim: PrimaryDim) -> PadType {
    tree.node_of_alloc(alloc)
        .and_then(|node| tree.allocate(node))
        .map_or(PadType::NoPad, |(_, held)| held.padding.padding(dim))
}

/// `static_cast<dsc2::ComputeNode*>(node)` UNDER `nodeType_ == COMPUTE` (`ddc/ddcv1.cpp:3038-3039`) —
/// the node itself, [`None`] for every other kind.
fn compute_of(tree: &TreeData, node: NodeId) -> Option<ComputeNode> {
    match tree.kind_of(node)? {
        Kind::Compute(held) => Some(held.clone()),
        _ => None,
    }
}

/// ⭐ `new dsc2::ConditionNode()` SPLICED AT `at`, THEN ITS TWO REGIONS — `addThenRegion(thenBlock)`
/// and `addElseRegion(elseBlock)` with each block's children under it (`ddc/ddcv1.cpp:3633-3664`).
///
/// ⛔ THE GUARD IS ONE OF TWO AND NEVER BOTH, which is `hasCoreClCond()` — *"`loopCond_`'s empty
/// `twoLevelOrOfAnds_` answers 'the core/corelet set is what guards this'"*
/// ([`crate::schedule::dsc2::ConditionNode::has_core_cl_cond`]). This is the exact inverse of the read
/// [`super::Dsc2Store`]'s `sched_node_of` performs, so a condition minted here and read back reports
/// the same guard.
///
/// ⛔ A REGION THE CONDITION NEVER GOT IS SKIPPED, not minted empty: [`CondRegions::then_branch`] and
/// [`CondRegions::else_branch`] are the reference's own `nullptr`-returning getters (`dsc/dsc2.h:707`,
/// `:713`), and *"ConditionNode only accepts 2 BlockNodes as children"* (`dsc/dsc2.cpp:2143`) is
/// discharged by that type rather than refused here.
fn mint_condition(tree: &mut TreeData, held: &ConditionNode, at: InsertionPoint) -> Option<NodeId> {
    let (loop_cond, cores) = if held.has_core_cl_cond() {
        (None, Some(v1::CoreClSet(held.core_cl_cond.clone())))
    } else {
        (Some(held.loop_cond.clone()), None)
    };
    let condition = tree.add(
        held.base.name.clone(),
        Kind::Condition(Cond {
            loop_cond,
            cores,
            then_region: Vec::new(),
            else_region: Vec::new(),
        }),
        None,
    );
    tree.link(condition, at);
    for (region, then_region) in [
        (held.next.then_branch(), true),
        (held.next.else_branch(), false),
    ] {
        let Some(block) = region else {
            continue;
        };
        let node = tree.add(block.base.name.clone(), Kind::Block, None);
        tree.add_region(condition, node, then_region);
        for child in &block.children {
            mint_sched_node(tree, child, InsertionPoint::LastIn(node))?;
        }
    }
    Some(condition)
}

/// ⭐ `new dsc2::XNode(*node)` FOLLOWED BY `addChildNode` — ONE node of a minted subtree, with its own
/// children under it, spliced at `at`.
///
/// ⛔ TWO ARMS ARE REFUSED AND NEITHER IS A GAP IN THE TREE:
///
/// * [`SchedNode::Leaf`] carries ONLY a [`NodeName`]. An ALLOCATE, COMPUTE or TRANSFER node needs its
///   payload — an `(AllocId, L3AllocateNode)` pair, a [`ComputeNode`], a [`TransferNode`] — and
///   minting one from a name alone is fabricating the node this crate ranks worse than a stop.
/// * [`SchedNode::Condition`] carries a bare child vector and NO guard at all, so materialising one
///   means inventing a `loopCond_`/`coreClCond_`. Nothing produces it out of this tree either —
///   [`super::Dsc2Store`]'s `sched_node_of` reads a `Kind::Condition` back as
///   [`SchedNode::Guarded`], which IS answered above.
fn mint_sched_node(tree: &mut TreeData, held: &SchedNode, at: InsertionPoint) -> Option<NodeId> {
    use crate::schedule::ddc::transformation_util::{
        LoopDims, LoopNode as HeldLoop, PrimaryDimAndKind,
    };

    let (name, kind, children) = match held {
        SchedNode::Block(block) => (block.base.name.clone(), Kind::Block, block.children.as_slice()),
        SchedNode::StickMask(mask) => (
            mask.base.name.clone(),
            Kind::StickMask((**mask).clone()),
            [].as_slice(),
        ),
        SchedNode::Sync(sync) => (sync.base.name.clone(), Kind::Sync(sync.clone()), [].as_slice()),
        // ⛔ THE SAME THREE REFUSALS [`super::Dsc2Store`]'s `add_loop` STANDS ON, and for the same
        // reasons: [`HeldLoop`]'s `numId_`/`denId_` pair is non-optional, a parametric loop carries
        // `-1` for both (`ddc/ddl/ddl_conversion.cpp:1129-1130`), and [`LoopDims`] makes *"Cannot
        // construct loop with no dimensions"* unspellable.
        SchedNode::Loop(node) => {
            let (num, den) = (node.num?, node.den?);
            if node.parametric_lds.is_some() {
                return None;
            }
            let kinded = |dim: &crate::schedule::dsc2::LoopDim| PrimaryDimAndKind {
                dim: dim.dim,
                kind: dim.kind,
            };
            let (first, rest) = node.dims.split_first()?;
            (
                node.block.base.name.clone(),
                Kind::Loop(HeldLoop {
                    name: node.block.base.name.clone(),
                    num,
                    den,
                    dims: LoopDims::new(kinded(first), rest.iter().map(kinded).collect()),
                }),
                node.block.children.as_slice(),
            )
        }
        SchedNode::Guarded(cond) => return mint_condition(tree, cond, at),
        SchedNode::Condition(_) | SchedNode::Leaf(_) => return None,
    };
    let node = tree.add(name, kind, None);
    tree.link(node, at);
    for child in children {
        mint_sched_node(tree, child, InsertionPoint::LastIn(node))?;
    }
    Some(node)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sys_arch_spec::arch_enums::SenComponent;

    use crate::arch::Elements;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
    use crate::schedule::ddc::fold::{NodeId, NodeKind, PadType};
    use crate::schedule::ddc::transformation::LoopId;
    use crate::schedule::ddc::transformation_util::{InsertionPoint, PaddingForm};
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        BlockNode, ComputeNode, CondOp, CondRegions, ConditionNode, Coordinate, DataInfo,
        InstrAttribute, LayoutDims, LdsIdx, LeafKind, LeafNode, LoopBound, LoopCond,
        LoopCondComposite, NodeBase, NodeName, Operand, RepetitionWithOffset, SchedNode,
        StickMaskNode,
    };
    use crate::units::{Core, Corelet, NumFolds};

    use super::super::tree::{Kind, TreeData, seed_allocate_node};
    use super::{
        alloc_padding_of, compute_of, insert_condition_before_of,
        mint_condition, mint_sched_node,
    };

    /// A tree with just its `root_level_operations` head, which is the one node
    /// [`crate::schedule::stages::DscState`] seeds every DSC with.
    fn a_rooted_tree() -> (TreeData, NodeId) {
        let mut tree = TreeData::default();
        let root = tree.add(
            NodeName("root_level_operations".to_owned()),
            Kind::Block,
            None,
        );
        tree.set_head(root);
        (tree, root)
    }

    /// ⭐ `allocNode->padding_.getPadding(dim)` CARRIES THE STATED [`PadType`] AND `NOPAD` FOR AN
    /// UNSTATED DIM — the reference's *"Default is NOPAD"* (`dsc/dims.cpp:808-810`), read back through
    /// the tree by the [`crate::schedule::ddc::fold::AllocId`] the node was minted under.
    #[test]
    fn alloc_padding_reads_the_stated_pad_type_and_nopad_for_a_dim_the_form_omits() {
        let (mut tree, root) = a_rooted_tree();
        let mut held = seed_allocate_node(
            NodeName("allocate_lds0_lx".to_owned()),
            LdsIdx(0),
            SenComponent::Lx,
            &LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::In]),
        );
        let mut padding = PaddingForm::default();
        // Two DIFFERENT non-default values, so a reader that answered "some padding" would pass and a
        // reader that answered the wrong dim's would not.
        padding.set_padding(PrimaryDim::In, PadType::PaddedFullSpan);
        padding.set_padding(PrimaryDim::Out, PadType::LoweredPadded);
        held.padding = padding;
        let alloc = tree.fresh_alloc();
        tree.add(held.name.clone(), Kind::Allocate(alloc, held), Some(root));

        assert_eq!(
            alloc_padding_of(&tree, alloc, PrimaryDim::In),
            PadType::PaddedFullSpan
        );
        assert_eq!(
            alloc_padding_of(&tree, alloc, PrimaryDim::Out),
            PadType::LoweredPadded
        );
        // ⛔ THE UNSTATED DIM IS `NOPAD` AND NOT A REFUSAL — `getPadding`'s own default.
        assert_eq!(
            alloc_padding_of(&tree, alloc, PrimaryDim::Mb),
            PadType::NoPad
        );
    }

    /// ⭐ `static_cast<dsc2::ComputeNode*>(node)` CARRIES `exUnit_` AND THE `outputs_`/offsets ZIP, and
    /// answers [`None`] for a node whose `nodeType_` is not `COMPUTE` — the reference's own downcast
    /// guard (`ddc/ddcv1.cpp:3038-3039`).
    #[test]
    fn compute_carries_the_ex_unit_and_the_operand_zip_and_refuses_every_other_kind() {
        let (mut tree, root) = a_rooted_tree();
        let operand = |lds: u32| Operand {
            unit: SenComponent::Pe,
            storage: SenComponent::Pelrf,
            data: DataInfo {
                my_lds_idx: Some(LdsIdx(lds)),
                ..DataInfo::default()
            },
        };
        let held = ComputeNode {
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: Coordinate::default(),
            repetition_with_offset: RepetitionWithOffset::default(),
            name: NodeName("compute_pe_fma16".to_owned()),
            op: DdlComputeType::Fma16,
            ex_unit: SenComponent::Pe,
            inputs: vec![operand(0), operand(1)],
            outputs: vec![operand(2)],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        };
        let node = tree.add(held.name.clone(), Kind::Compute(held), Some(root));

        let read = compute_of(&tree, node).expect("a COMPUTE node answers its own node");
        assert_eq!(read.ex_unit, SenComponent::Pe);
        assert_eq!(read.op, DdlComputeType::Fma16);
        // ⭐ THE ZIP, CARRYING THE `myLdsIdx_` VALUES — which is what `compute_op`'s caller wants and
        // what makes the `"Compute node input/output missing information"` DT_ERROR unspellable.
        assert_eq!(
            read.inputs
                .iter()
                .map(|held| held.data.my_lds_idx)
                .collect::<Vec<_>>(),
            vec![Some(LdsIdx(0)), Some(LdsIdx(1))]
        );
        assert_eq!(
            read.outputs.first().map(|held| held.data.my_lds_idx),
            Some(Some(LdsIdx(2)))
        );
        // ⛔ THE NEGATIVE CONTROL: the head block is not a compute.
        assert_eq!(compute_of(&tree, root), None);
    }

    /// One SAMV mask node, and its `firstStickCoordToMaskPerDim_`-cleared reset twin.
    fn a_mask(name: &str, masked: bool) -> StickMaskNode {
        StickMaskNode {
            base: NodeBase::named(NodeName(name.to_owned())),
            mask_val_const_id: None,
            data_format: None,
            stick_layout: Vec::new(),
            first_stick_coord_to_mask_per_dim: if masked {
                BTreeMap::from([(PrimaryDim::Out, Elements(7))])
            } else {
                BTreeMap::new()
            },
            affected_transfers: Vec::new(),
        }
    }

    /// Entry 262's condition, exactly as `coordinate_masking` builds it (`ddc/v1.rs:3539-3560`).
    fn a_samv_condition(at: LoopId) -> ConditionNode {
        ConditionNode {
            base: NodeBase::named(NodeName("condition_SAMV_dim_out".to_owned())),
            loop_cond: LoopCondComposite {
                two_level_or_of_ands: vec![vec![LoopCond {
                    loop_comp: at,
                    dim: PrimaryDim::Out,
                    op: CondOp::Eq,
                    bound: LoopBound::Last,
                }]],
                negated: false,
            },
            core_cl_cond: BTreeMap::new(),
            next: CondRegions::ThenElse([
                BlockNode {
                    base: NodeBase::named(NodeName("block_SAMV_dim_out".to_owned())),
                    children: vec![SchedNode::StickMask(Box::new(a_mask("SAMV_out", true)))],
                },
                BlockNode {
                    base: NodeBase::named(NodeName("block_SAMV_reset".to_owned())),
                    children: vec![SchedNode::StickMask(Box::new(a_mask("SAMV_reset", false)))],
                },
            ]),
        }
    }

    /// ⭐⭐ `lastSeenNode->prev_->addChildNode(cond, true, lastSeenNode)` AND THE TWO REGIONS — read
    /// back THROUGH THE TREE, which is the only reading that can tell a splice from a local.
    ///
    /// ⛔ THE ORDER IS THE ASSERTION: the condition sits BEFORE `before` among the parent's children and
    /// `before` is still that parent's child, because the reference never moves it (`:3600-3665`).
    #[test]
    fn insert_condition_before_splices_the_guard_ahead_of_its_sibling_and_fills_both_regions() {
        let (mut tree, root) = a_rooted_tree();
        let earlier = tree.add(
            NodeName("transfer_first".to_owned()),
            Kind::Block,
            Some(root),
        );
        let before = tree.add(
            NodeName("transfer_lds0".to_owned()),
            Kind::Block,
            Some(root),
        );

        insert_condition_before_of(&mut tree, before, &a_samv_condition(LoopId(root)))
            .expect("a block region of one stick mask is spliceable");
        // The condition is the child the parent gained; naming it by position is the point of the
        // next assertion, so it is found by KIND and not by index.
        let condition = *tree
            .children(root)
            .iter()
            .find(|node| tree.node_kind(**node) == Some(NodeKind::Condition))
            .expect("the splice minted a CONDITION under the parent");

        // ⛔ POSITION, not just membership.
        assert_eq!(tree.children(root), vec![earlier, condition, before]);
        assert_eq!(tree.parent(before), Some(root));
        assert_eq!(tree.node_kind(condition), Some(NodeKind::Condition));
        assert_eq!(
            tree.name(condition),
            Some(NodeName("condition_SAMV_dim_out".to_owned()))
        );

        // ⭐ `hasCoreClCond()` IS FALSE HERE, so the guard is the loop predicate and the core set is
        // absent — the two states [`super::Cond`] keeps apart.
        let cond = tree
            .loop_cond(condition)
            .expect("a loop-guarded condition carries its loopCond_");
        assert_eq!(
            cond.two_level_or_of_ands,
            vec![vec![LoopCond {
                loop_comp: LoopId(root),
                dim: PrimaryDim::Out,
                op: CondOp::Eq,
                bound: LoopBound::Last,
            }]]
        );
        assert_eq!(tree.core_cl_cond(condition), None);

        // ⭐ `getThenBranchNode()` IS THE FIRST BLOCK OF THE THEN REGION, and both regions are linked
        // children in order.
        let regions = tree.children(condition);
        assert_eq!(regions.len(), 2, "addThenRegion then addElseRegion");
        assert_eq!(tree.then_branch(condition), regions.first().copied());
        // ⛔ AND THE TWO REGIONS ARE DISJOINT AND IN THE RIGHT SLOTS — `getThenBranchNode()` alone
        // cannot tell a condition that pushed BOTH blocks into `then_region` from a correct one.
        let (then_region, else_region) = match tree.kind_of(condition) {
            Some(Kind::Condition(held)) => {
                Some((held.then_region.clone(), held.else_region.clone()))
            }
            _ => None,
        }
        .expect("a CONDITION node holds a Cond");
        assert_eq!(then_region, vec![regions[0]]);
        assert_eq!(else_region, vec![regions[1]]);
        assert_eq!(
            regions
                .iter()
                .filter_map(|node| tree.name(*node))
                .collect::<Vec<_>>(),
            vec![
                NodeName("block_SAMV_dim_out".to_owned()),
                NodeName("block_SAMV_reset".to_owned()),
            ]
        );

        // ⭐⭐ AND THE MASKS THEMSELVES, CARRYING THE VALUE THAT DISTINGUISHES THEM: the then-branch
        // masks `out` at 7 and the else-branch reset masks nothing (`ddc/ddcv1.cpp:3663-3664`).
        let mask_of = |block: NodeId| match tree.kind_of(*tree.children(block).first()?)? {
            Kind::StickMask(held) => Some(held.clone()),
            _ => None,
        };
        let then_mask = mask_of(regions[0]).expect("the then region holds a STICKMASK");
        let else_mask = mask_of(regions[1]).expect("the else region holds a STICKMASK");
        assert_eq!(then_mask.base.name, NodeName("SAMV_out".to_owned()));
        assert_eq!(
            then_mask.first_stick_coord_to_mask_per_dim,
            BTreeMap::from([(PrimaryDim::Out, Elements(7))])
        );
        assert_eq!(else_mask.base.name, NodeName("SAMV_reset".to_owned()));
        assert!(
            else_mask.first_stick_coord_to_mask_per_dim.is_empty(),
            "samvReset->firstStickCoordToMaskPerDim_.clear()"
        );

        // ⛔ AND `before->prev_ == nullptr` MINTS NOTHING — the count is read back off the tree, so a
        // refusal that had already spliced a condition would fail here.
        let held = tree.len();
        assert_eq!(
            insert_condition_before_of(&mut tree, root, &a_samv_condition(LoopId(root))),
            None
        );
        assert_eq!(tree.len(), held, "a refusal leaves no orphaned condition");
    }

    /// ⭐ A CORE/CORELET-GUARDED CONDITION LANDS IN THE OTHER SLOT — `hasCoreClCond()` is
    /// `loopCond_.twoLevelOrOfAnds_.empty()` (`dsc/dsc2.h:685`), so an empty predicate means the set is
    /// the guard, and a minted condition read back reports exactly one of the two.
    #[test]
    fn a_condition_with_no_loop_predicate_is_guarded_by_its_core_corelet_set() {
        let (mut tree, root) = a_rooted_tree();
        let before = tree.add(
            NodeName("transfer_lds0".to_owned()),
            Kind::Block,
            Some(root),
        );
        let core = Core::checked(3).expect("core 3 exists on every arch this crate targets");
        let cores = BTreeMap::from([(core, [Corelet::at::<1>()].into_iter().collect())]);
        let condition = mint_condition(
            &mut tree,
            &ConditionNode {
                base: NodeBase::named(NodeName("condition_core3".to_owned())),
                loop_cond: LoopCondComposite::default(),
                core_cl_cond: cores.clone(),
                next: CondRegions::Then(BlockNode {
                    base: NodeBase::named(NodeName("block_core3".to_owned())),
                    children: Vec::new(),
                }),
            },
            InsertionPoint::Before(before),
        )
        .expect("an empty then-block is still a block");

        assert_eq!(tree.loop_cond(condition), None);
        assert_eq!(
            tree.core_cl_cond(condition).map(|set| set.0),
            Some(cores),
            "coreClCond_ carried across, and loopCond_ left absent"
        );
        assert_eq!(tree.children(condition).len(), 1, "only a then region");
    }

    /// ⛔ THE THREE THINGS A MINTED SUBTREE REFUSES RATHER THAN INVENTS — and each is a NODE that would
    /// otherwise be fabricated: a `Leaf` carries only a name, a bare `Condition` carries no guard, and a
    /// parametric loop carries `numId_ = denId_ = -1`.
    #[test]
    fn a_subtree_refuses_the_arms_whose_payload_it_would_have_to_invent() {
        let (mut tree, root) = a_rooted_tree();
        let before = mint_sched_node(
            &mut tree,
            &SchedNode::Block(BlockNode {
                base: NodeBase::named(NodeName("host".to_owned())),
                children: Vec::new(),
            }),
            InsertionPoint::LastIn(root),
        )
        .expect("a block is spliceable");

        for arm in [
            SchedNode::Leaf(LeafNode::new(
                LeafKind::Allocate,
                NodeName("allocate_lds0_lx".to_owned()),
            )),
            SchedNode::Condition(BlockNode {
                base: NodeBase::named(NodeName("condition_unguarded".to_owned())),
                children: Vec::new(),
            }),
        ] {
            assert_eq!(
                mint_sched_node(&mut tree, &arm, InsertionPoint::LastIn(before)),
                None,
                "{arm:?} names a node whose payload this tree would have to invent"
            );
        }
    }

    /// ⭐ `isParametricLoop()` IS `false` FOR EVERY LOOP THIS TREE CAN HOLD, and the reason is a TYPE and
    /// not a default: a parametric loop's `numId_`/`denId_` are both `-1`
    /// (`ddc/ddl/ddl_conversion.cpp:1129-1130`), which
    /// [`crate::schedule::ddc::transformation_util::LoopNode`] cannot spell — so the materialiser
    /// refuses one rather than dropping the flag, exactly as `Dsc2Store`'s `add_loop` does.
    #[test]
    fn a_parametric_loop_is_refused_rather_than_minted_with_its_flag_dropped() {
        use crate::schedule::dsc2::{LoopDim, LoopNode, NodeBase};

        let (mut tree, root) = a_rooted_tree();
        let dims = vec![LoopDim {
            dim: PrimaryDim::Out,
            kind: crate::schedule::ddc::metadata::MetaDimKind::Unpadded,
        }];
        let block = BlockNode {
            base: NodeBase::named(NodeName("loop_ds0_ds1_out".to_owned())),
            children: Vec::new(),
        };
        let plain = LoopNode {
            dims: dims.clone(),
            num: Some(crate::schedule::ddc::metadata::DatastageId(0)),
            den: Some(crate::schedule::ddc::metadata::DatastageId(1)),
            ..LoopNode::bare(block)
        };
        let minted = mint_sched_node(
            &mut tree,
            &SchedNode::Loop(Box::new(plain.clone())),
            InsertionPoint::LastIn(root),
        )
        .expect("a loop with a real numId_/denId_ pair is spliceable");
        assert_eq!(tree.node_kind(minted), Some(NodeKind::Loop));

        // ⛔ THE PARAMETRIC TWIN IS REFUSED — both for its `parametricLdsIdx_` and, on the other side,
        // for the `-1` pair it would carry.
        assert_eq!(
            mint_sched_node(
                &mut tree,
                &SchedNode::Loop(Box::new(LoopNode {
                    parametric_lds: Some(LdsIdx(0)),
                    ..plain.clone()
                })),
                InsertionPoint::LastIn(root),
            ),
            None
        );
        assert_eq!(
            mint_sched_node(
                &mut tree,
                &SchedNode::Loop(Box::new(LoopNode {
                    num: None,
                    den: None,
                    ..plain
                })),
                InsertionPoint::LastIn(root),
            ),
            None
        );
    }
}
