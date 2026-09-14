// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE SURGERY CARRIER — `E` in [`crate::schedule::l3::dl_ops::run`], which is fifteen traits
//! once every supertrait is closed: `DscTrees`, `DscTreeSurgery`, `DscTransferWrites`,
//! `DscGtrSurgery`, `DscL3Surgery`, `TransferNodes`, `DscSyncSurgery`, `DscTransfers`,
//! `DscTransferSizes`, `ChunkLoopNest`, `DscPagedTrees`, `CoordPropTrees`, `MemOrgs`,
//! `AllocationSites` and `LxZeroPadTransform`.
//!
//! ⭐ IT HOLDS `&'s DscState` — A SHARED REFERENCE — so [`super::Reads`] names the same trees and the
//! same `memOrg_` nodes. Every write below goes through that state's cells, which is what restores
//! the reference's single `this` across the port's severed signature.
//!
//! ⛔ THE TWO `allocation` METHODS COLLIDE: [`DscTrees::allocation`] answers a [`NodeId`] and
//! [`AllocationSites::allocation`] answers an [`AllocateNode`], both on `E` with the same arguments.
//! Every call inside the port is already disambiguated; a caller reading this file must be too.

use std::cell::RefCell;
use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::{Bytes, Elements};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::schedule::ddc::fold::{
    AllocId, DistributedLoop, ElemArrDistribution, FoldParamInfo, NodeId, PadType,
    TemporalLoopDistribution,
};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::{
    DataStages, InsertionPoint, LoopBands, LoopDims, LoopNode, PrimaryDimAndKind, ScheduleSurgery,
};
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{
    AllocLayout, AllocPlacement, AllocateNode, BlockNode, Coordinate, Dsc, LayoutDims, LdsIdx,
    LoopCondComposite, MaxDimSize, NodeName, NumBuffers, Padding, StartAddress, SyncNode,
    TransferNode,
};
use crate::schedule::l3::dl_ops::{
    AllocCoordinateSeam, ChunkLoopNest, CoordPropTree, CoordPropTrees, CoreWindowDims,
    CoreletSliceDims, CoreletSliceSeam, DscCoordProp, DscGtrSurgery, DscL3Surgery, DscPagedTrees,
    DscSyncSurgery, DscTransferSizes, DscTransferWrites, DscTransfers, DscTreeSurgery, DscTrees,
    GtrGroupId, L3AllocateNode, L3Sync, L3TreeSurgery, L3WalkNode, LoopAndDim, LxZeroPadTransform,
    PropagatedAllocation,
};
use crate::schedule::l3::dsc::{
    Buffering, DimPadding, DscIdx, L3Transfer, MemOrgs, SuperChunkStage, SuperDsc, TransferNodes,
};

use super::state::{DscState, DscTree};
use super::tree::{Cond, Kind, Org, TreeData};

/// ⭐ THE `E` CARRIER — one shared reference to the state, plus the two cursors the port's
/// `&mut Self::Tree` accessors need and the CORE window dims [`ChunkLoopNest`] takes its loop dims
/// from.
#[derive(Debug)]
pub struct Env<'s> {
    state: &'s DscState,
    /// `for (dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx)`.
    dscs: Vec<DscIdx>,
    /// ⭐ [`CoreWindowDims::of_l3`] PER DSC, SNAPSHOT BEFORE THE STAGE RUNS AND PROVABLY FRESH: it
    /// reads the CORE data stage's `ki`/`kj` (`dsc/dims.h:183`), and stage 2a writes only the chunk,
    /// super-chunk, one-page and IBR stages — never the core one. The snapshot is how a carrier that
    /// cannot see `&mut SuperDsc` answers a question about it without a second answer.
    core_windows: Vec<CoreWindowDims>,
    /// The cursor [`DscPagedTrees::tree_mut`] hands out.
    paged: PagedCursor<'s>,
    /// The cursor [`CoordPropTrees::coord_prop`] would hand out.
    coord: CoordCursor<'s>,
    /// The distribution and corelet-slice seam that same accessor hands out.
    seam: CoordSeam,
    /// `distributeElemArrToTemporalLoops`' accumulated loop parameters.
    loop_params: (),
    /// `getLayoutDims(ldsIdx)` on the DSC the cursor names.
    layout: Layout,
}

impl<'s> Env<'s> {
    /// The carrier over one seeded state and the super-DSC it was seeded from.
    ///
    /// ⭐ THE SUPER-DSC IS READ HERE AND NOT HELD: `run` takes it as `&mut`, so a carrier that kept a
    /// borrow of it could not be handed to `run` at all. What is taken is the per-DSC CORE window
    /// dims, which stage 2a does not rewrite.
    #[must_use]
    pub fn new(state: &'s DscState, sdsc: &SuperDsc) -> Self {
        Self {
            state,
            dscs: (0u32..)
                .map(DscIdx)
                .zip(sdsc.dscs().iter())
                .map(|(at, _)| at)
                .collect(),
            core_windows: sdsc.dscs().iter().map(CoreWindowDims::of_l3).collect(),
            paged: PagedCursor {
                state,
                at: DscIdx(0),
            },
            coord: CoordCursor {
                state,
                at: DscIdx(0),
            },
            seam: CoordSeam::default(),
            loop_params: (),
            layout: Layout,
        }
    }

    /// That DSC's tree, [`None`] for a `dscs_` position the state holds none for.
    fn dsc(&self, at: DscIdx) -> Option<&'s DscTree> {
        self.state.dsc(at)
    }
}

impl DscTrees for Env<'_> {
    type Tree = DscTree;

    fn tree(&self, dsc: DscIdx) -> Option<&Self::Tree> {
        self.dsc(dsc)
    }

    fn root(&self, dsc: DscIdx) -> Option<NodeId> {
        self.dsc(dsc)?.with(TreeData::head)
    }

    fn lx_below_block(&self, dsc: DscIdx) -> Option<NodeId> {
        self.dsc(dsc)?.with(TreeData::lx_below_block)
    }

    fn allocation(&self, dsc: DscIdx, lds: LdsIdx, storage: SenComponent) -> Option<NodeId> {
        self.dsc(dsc)?.org(lds)?.node(storage)
    }

    fn transfer_src_lds(&self, dsc: DscIdx, node: NodeId) -> Option<LdsIdx> {
        self.dsc(dsc)?
            .with(|tree| tree.transfer(node))?
            .src
            .data
            .my_lds_idx
    }

    fn transfer_dst_is_lds(&self, dsc: DscIdx, node: NodeId) -> bool {
        self.dsc(dsc)
            .and_then(|dsc| dsc.with(|tree| tree.transfer(node)))
            .is_some_and(|held| held.dsts.first().data.my_lds_idx.is_some())
    }
}

impl DscTreeSurgery for Env<'_> {
    fn insert_sync(&mut self, dsc: DscIdx, sync: SyncNode, at: InsertionPoint) -> NodeId {
        let Some(held) = self.dsc(dsc) else {
            return NodeId(u32::MAX);
        };
        held.with_mut(|tree| {
            let node = tree.add(sync.name.clone(), Kind::Sync(sync), None);
            tree.link(node, at);
            node
        })
    }

    fn move_node(&mut self, dsc: DscIdx, node: NodeId, at: InsertionPoint) {
        if let Some(held) = self.dsc(dsc) {
            held.with_mut(|tree| {
                tree.unlink(node);
                tree.link(node, at);
            });
        }
    }
}

impl DscTransferWrites for Env<'_> {
    fn transfer(&self, dsc: DscIdx, node: NodeId) -> Option<TransferNode> {
        self.dsc(dsc)?.with(|tree| tree.transfer(node))
    }

    fn set_transfer(&mut self, dsc: DscIdx, node: NodeId, transfer: TransferNode) {
        if let Some(held) = self.dsc(dsc) {
            held.with_mut(|tree| tree.set_transfer(node, transfer));
        }
    }
}

impl DscGtrSurgery for Env<'_> {
    fn node_name(&self, dsc: DscIdx, node: NodeId) -> Option<NodeName> {
        self.dsc(dsc)?.with(|tree| tree.name(node))
    }

    fn new_transfer(&mut self, dsc: DscIdx, transfer: TransferNode) -> NodeId {
        let Some(held) = self.dsc(dsc) else {
            return NodeId(u32::MAX);
        };
        held.with_mut(|tree| tree.add(transfer.name.clone(), Kind::Transfer(transfer), None))
    }

    fn new_block(&mut self, dsc: DscIdx, name: NodeName) -> NodeId {
        let Some(held) = self.dsc(dsc) else {
            return NodeId(u32::MAX);
        };
        held.with_mut(|tree| tree.add(name, Kind::Block, None))
    }

    fn new_condition(&mut self, dsc: DscIdx, name: NodeName, cond: LoopCondComposite) -> NodeId {
        let Some(held) = self.dsc(dsc) else {
            return NodeId(u32::MAX);
        };
        held.with_mut(|tree| {
            tree.add(
                name,
                Kind::Condition(Cond {
                    loop_cond: Some(cond),
                    ..Cond::default()
                }),
                None,
            )
        })
    }

    fn add_then_region(&mut self, dsc: DscIdx, condition: NodeId, block: NodeId) {
        if let Some(held) = self.dsc(dsc) {
            held.with_mut(|tree| tree.add_region(condition, block, true));
        }
    }

    fn add_else_region(&mut self, dsc: DscIdx, condition: NodeId, block: NodeId) {
        if let Some(held) = self.dsc(dsc) {
            held.with_mut(|tree| tree.add_region(condition, block, false));
        }
    }

    fn add_child_node(&mut self, dsc: DscIdx, node: NodeId, at: InsertionPoint) {
        if let Some(held) = self.dsc(dsc) {
            held.with_mut(|tree| tree.link(node, at));
        }
    }

    fn add_alloc_user(&mut self, dsc: DscIdx, alloc: NodeId, user: NodeId) {
        let Some(held) = self.dsc(dsc) else {
            return;
        };
        // `allocUsers_` lives on the node, which is reached through the `memOrg_` entry naming it.
        if let Some((lds, storage)) = held.home_of(alloc)
            && let Some(org) = held.org(lds)
        {
            org.add_user(storage, user);
        }
    }

    fn insert_gtr_id(&mut self, dsc: DscIdx, group: GtrGroupId) {
        if let Some(held) = self.dsc(dsc) {
            held.with_mut(|tree| tree.insert_gtr_id(group));
        }
    }
}

impl DscL3Surgery for Env<'_> {
    fn fresh_alloc(&mut self, dsc: DscIdx) -> AllocId {
        let Some(held) = self.dsc(dsc) else {
            return AllocId(u32::MAX);
        };
        held.with_mut(TreeData::fresh_alloc)
    }

    fn new_allocate(&mut self, dsc: DscIdx, alloc: AllocId, node: L3AllocateNode) -> NodeId {
        let Some(held) = self.dsc(dsc) else {
            return NodeId(u32::MAX);
        };
        held.with_mut(|tree| tree.add(node.name.clone(), Kind::Allocate(alloc, node), None))
    }

    fn set_mem_org_allocation(
        &mut self,
        dsc: DscIdx,
        lds: LdsIdx,
        storage: SenComponent,
        node: NodeId,
    ) {
        let Some(held) = self.dsc(dsc) else {
            return;
        };
        let Some((_, minted)) = held.with(|tree| tree.allocate(node)) else {
            return;
        };
        if let Some(org) = held.org(lds) {
            org.set_node(storage, node, minted.clone());
        }
        // ⭐ ONE NODE, TWO VOCABULARIES: `memOrg_.at(storage).allocateNode_` is a single
        // `dsc2::AllocateNode`, and entries 219/220/292 place an address on the SAME node entry 353
        // just minted. Deriving the ddc view from the L3 view is what keeps the two from disagreeing
        // — a second, independently-built node is a second answer.
        if let Some(view) = ddc_view(&minted) {
            held.set_placed(node, view);
        }
    }

    fn new_core_condition(&mut self, dsc: DscIdx, name: NodeName, cores: v1::CoreClSet) -> NodeId {
        let Some(held) = self.dsc(dsc) else {
            return NodeId(u32::MAX);
        };
        held.with_mut(|tree| {
            tree.add(
                name,
                Kind::Condition(Cond {
                    cores: Some(cores),
                    ..Cond::default()
                }),
                None,
            )
        })
    }
}

impl TransferNodes for Env<'_> {
    fn transfers(&self, dsc: DscIdx) -> Vec<L3Transfer> {
        self.dsc(dsc)
            .map(|held| held.with(TreeData::transfers))
            .unwrap_or_default()
    }
}

impl DscSyncSurgery for Env<'_> {
    fn syncs(&self, dsc: DscIdx) -> Vec<L3Sync> {
        self.dsc(dsc)
            .map(|held| held.with(TreeData::syncs))
            .unwrap_or_default()
    }

    fn delete_node(&mut self, dsc: DscIdx, node: NodeId) {
        if let Some(held) = self.dsc(dsc) {
            held.with_mut(|tree| tree.delete(node));
        }
    }
}

impl MemOrgs for Env<'_> {
    type Org = Org;

    fn mem_org(&self, dsc: DscIdx, lds: LdsIdx) -> Option<&Self::Org> {
        self.dsc(dsc)?.org(lds)
    }
}

impl crate::schedule::l3::dl_ops::AllocationSites for Env<'_> {
    fn allocation(&self, dsc: DscIdx, lds: LdsIdx, storage: SenComponent) -> Option<AllocateNode> {
        let held = self.dsc(dsc)?;
        held.org(lds)?.placed(storage)
    }

    fn set_allocation(
        &mut self,
        dsc: DscIdx,
        lds: LdsIdx,
        storage: SenComponent,
        node: AllocateNode,
    ) {
        let Some(held) = self.dsc(dsc) else {
            return;
        };
        if let Some(org) = held.org(lds) {
            org.set_placed(storage, node.clone());
            if let Some(at) = org.node(storage) {
                held.set_placed(at, node);
            }
        }
    }
}

impl DscTransfers for Env<'_> {
    type Org = Org;

    fn mem_orgs(&self, dsc: DscIdx) -> Vec<&Self::Org> {
        self.dsc(dsc)
            .map(|held| held.orgs().iter().collect())
            .unwrap_or_default()
    }

    fn lx_alloc_transfer_users(&self, dsc: DscIdx, lds: LdsIdx) -> Vec<NodeId> {
        let Some(held) = self.dsc(dsc) else {
            return Vec::new();
        };
        let Some(org) = held.org(lds) else {
            return Vec::new();
        };
        let Some(users) = crate::schedule::l3::dsc::MemOrg::lx_alloc_users(org) else {
            return Vec::new();
        };
        held.with(|tree| {
            users
                .into_iter()
                .filter(|user| tree.transfer(*user).is_some())
                .collect()
        })
    }
}

impl DscTransferSizes for Env<'_> {
    /// ⛔ NOT ANSWERABLE FROM THIS STATE, AND NOT A FACT THE TREE HOLDS:
    /// `dsc.getBlockTransferSizePerDim(*transNode, storage, clId)` is `dsc/dsc2.cpp:3474`, declared
    /// OUTSIDE this campaign's file list, and it reads the DSC's data stages and stick sizes — which
    /// live in the `&mut SuperDsc` this carrier cannot see. Its one reader
    /// (`fill_explicit_transfer_size`) SKIPS every DSC with `numCoreletsUsed_DSC2_ <= 1`, and
    /// scratchy emits `1` in all 313 sampled super-DSCs, so this is unreached on the path that
    /// matters.
    fn block_transfer_size_per_dim(
        &self,
        _dsc: DscIdx,
        _node: NodeId,
        _storage: SenComponent,
        _corelet: crate::units::Corelet,
    ) -> Option<BTreeMap<PrimaryDim, Elements>> {
        todo!(
            "DscTransferSizes::block_transfer_size_per_dim: wants \
             DesignSpaceConfig::getBlockTransferSizePerDim (dsc/dsc2.cpp:3474) over the DSC's data \
             stages, which `&'a F`/`&'a mut E` cannot reach through `run`'s `&mut SuperDsc`"
        )
    }
}

impl ChunkLoopNest for Env<'_> {
    fn dscs(&self) -> Vec<DscIdx> {
        self.dscs.clone()
    }

    fn core_window_dims(&self, dsc: DscIdx) -> Option<CoreWindowDims> {
        self.core_windows
            .get(usize::try_from(dsc.0).ok()?)
            .cloned()
    }

    fn set_head_den(&mut self, dsc: DscIdx, den: DatastageId) -> Option<()> {
        let held = self.dsc(dsc)?;
        held.with_mut(|tree| tree.set_head_den(den));
        Some(())
    }

    fn head(&self, dsc: DscIdx) -> Option<NodeId> {
        self.dsc(dsc)?.with(TreeData::head)
    }

    /// ⛔ REFUSES, AND WHAT IT WANTS: `getNewDataStageIndex(mySDsc, dsc)` followed by a DEFAULT
    /// INSERT into `dsc.dataStageParam_`. Both the search and the insert are on the super-DSC, which
    /// `run` holds as `&mut` and no carrier may alias — so a caller cannot mint a data stage from
    /// here. It is reached only for [`crate::schedule::l3::dl_ops::LxBufferChoice::SpatialDouble`];
    /// the reference chose DOUBLE for every program of scratchy's own staged bundle (one `(0,1)`
    /// loop band per order dim in `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_*/sdsc.json`).
    fn mint_super_chunk_stage(&mut self, _dsc: DscIdx) -> Option<SuperChunkStage> {
        self.state
            .refuse("ChunkLoopNest::mint_super_chunk_stage: wants a fresh dataStageParam_ entry on the &mut SuperDsc")
    }

    fn add_loop(&mut self, dsc: DscIdx, parent: NodeId, node: LoopNode) -> Option<NodeId> {
        let held = self.dsc(dsc)?;
        held.with_mut(|tree| {
            tree.name(parent)?;
            Some(tree.add(node.name.clone(), Kind::Loop(node), Some(parent)))
        })
    }

    fn add_block(&mut self, dsc: DscIdx, parent: NodeId, node: BlockNode) -> Option<NodeId> {
        let held = self.dsc(dsc)?;
        held.with_mut(|tree| {
            tree.name(parent)?;
            Some(tree.add(node.name, Kind::Block, Some(parent)))
        })
    }
}

impl LxZeroPadTransform for Env<'_> {
    /// ⛔ REFUSES, AND WHAT IT WANTS: `dsc2::transformLxZeroPadInfoInScheduleTree(mySDsc)` is a whole
    /// pass in `dsc/dsc2.cpp`, OUTSIDE this campaign's file list — one seam call and NOT a fact.
    /// Answering `Some(())` would state that the pass ran.
    fn transform_lx_zero_pad_info(&mut self, _sdsc: &SuperDsc) -> Option<()> {
        self.state.refuse(
            "LxZeroPadTransform::transform_lx_zero_pad_info: wants the unported \
             dsc2::transformLxZeroPadInfoInScheduleTree pass",
        )
    }
}

impl<'s> DscPagedTrees for Env<'s> {
    type Tree = PagedCursor<'s>;

    fn tree_mut(&mut self, dsc: DscIdx) -> Option<&mut Self::Tree> {
        self.state.dsc(dsc)?;
        self.paged.at = dsc;
        Some(&mut self.paged)
    }
}

impl<'s> CoordPropTrees for Env<'s> {
    type Layout = Layout;
    type Org = Org;
    type Tree = CoordCursor<'s>;
    type Env = CoordSeam;

    /// ⛔ REFUSES, AND WHAT IT WANTS: the tree half of this surface is answerable (see
    /// [`CoordCursor`]), but entry 374 also distributes through
    /// `distributeElemArrToTemporalLoops` and `dsc2::loopRelevantForDim` and reads
    /// `dataStageParam_` — the [`AllocCoordinateSeam`]/[`CoreletSliceSeam`] halves, all `dsc/`
    /// functions declared outside this campaign and all reading the `&mut SuperDsc`. `None` here is
    /// the trait's own *"a `dscs_` position the caller holds no surface for"*, which is literally
    /// this caller's state.
    fn coord_prop(
        &mut self,
        _dsc: DscIdx,
    ) -> Option<DscCoordProp<'_, Self::Layout, Self::Org, Self::Tree, Self::Env>> {
        self.state.refuse(
            "CoordPropTrees::coord_prop: wants the distributeElemArrToTemporalLoops / \
             loopRelevantForDim / dataStageParam_ seam entry 374 propagates through",
        )
    }
}

/// ⭐ ONE DSC'S TREE AS THE PAGED LOOP NEST WRITES IT — a CURSOR over [`DscState`] rather than an
/// owned tree, because [`DscPagedTrees::tree_mut`] hands out `&mut Self::Tree` and the tree it must
/// name is the shared one.
#[derive(Debug)]
pub struct PagedCursor<'s> {
    state: &'s DscState,
    at: DscIdx,
}

impl PagedCursor<'_> {
    /// The tree this cursor names, for one question.
    fn with<T>(&self, ask: impl FnOnce(&TreeData) -> T) -> Option<T> {
        Some(self.state.dsc(self.at)?.with(ask))
    }

    /// The same, for one write.
    fn with_mut<T>(&self, write: impl FnOnce(&mut TreeData) -> T) -> Option<T> {
        Some(self.state.dsc(self.at)?.with_mut(write))
    }
}

impl ScheduleSurgery for PagedCursor<'_> {
    fn node_name(&self, node: NodeId) -> NodeName {
        self.with(|tree| tree.name(node))
            .flatten()
            .unwrap_or_else(|| NodeName(String::new()))
    }

    fn set_node_name(&mut self, node: NodeId, name: NodeName) {
        self.with_mut(|tree| tree.set_name(node, name));
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.with(|tree| tree.parent(node)).flatten()
    }

    fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
        self.with(|tree| tree.owner_loop(node)).flatten()
    }

    fn transfer(&self, node: NodeId) -> TransferNode {
        self.with(|tree| tree.transfer(node))
            .flatten()
            .expect("a node id this walk produced names a transfer of this tree")
    }

    fn loop_num(&self, loop_node: LoopId) -> DatastageId {
        self.minted(loop_node).num
    }

    fn loop_den(&self, loop_node: LoopId) -> DatastageId {
        self.minted(loop_node).den
    }

    fn loop_dims(&self, loop_node: LoopId) -> LoopDims {
        self.minted(loop_node).dims
    }

    /// ⭐ FALSE FOR EVERY LOOP THIS ARENA HOLDS, AND NOT A DEFAULT: `isParametricLoop()` is
    /// `parametricLds_ != -1` on a `dsc2::LoopNode`, and every loop in this tree was minted by
    /// [`crate::schedule::l3::dl_ops::create_loop_node`] or
    /// [`crate::schedule::ddc::transformation_util::construct_loop_node`], neither of which states
    /// one — which is why [`LoopNode`] does not carry the field at all.
    fn is_parametric(&self, _loop_node: LoopId) -> bool {
        false
    }

    fn conditions_under(&self, root: NodeId) -> Vec<NodeId> {
        self.with(|tree| tree.conditions_under(root))
            .unwrap_or_default()
    }

    /// ⛔⛔ TWO RUST TYPES FOR ONE C++ FIELD, AND THIS IS WHERE THEY MEET.
    /// [`crate::schedule::dsc2::LoopCondComposite`] (what [`DscGtrSurgery::new_condition`] mints a
    /// condition WITH) and [`crate::schedule::ddc::transformation_util::LoopCondComposite`] (what
    /// [`ScheduleSurgery`] reads one BACK as) are both `dsc/dsc2.h:675-677`. Converting between them
    /// here would give this tree a SECOND answer for `loopCond_` that could disagree with the first,
    /// so the duplication is named rather than papered over. Unreached on scratchy's path: only the
    /// paged loop-split reads it, and no sampled super-DSC pages.
    fn loop_cond(
        &self,
        _condition: NodeId,
    ) -> crate::schedule::ddc::transformation_util::LoopCondComposite {
        todo!(
            "ScheduleSurgery::loop_cond: dsc2::LoopCondComposite and \
             transformation_util::LoopCondComposite are two Rust types for one dsc/dsc2.h:675-677 \
             field, and this tree holds the dsc2 one that new_condition minted"
        )
    }

    /// ⛔ THE SAME COLLISION, on the write side.
    fn set_loop_cond(
        &mut self,
        _condition: NodeId,
        _cond: crate::schedule::ddc::transformation_util::LoopCondComposite,
    ) {
        todo!(
            "ScheduleSurgery::set_loop_cond: would write a transformation_util::LoopCondComposite \
             over the dsc2::LoopCondComposite new_condition minted"
        )
    }

    fn new_loop(&mut self, loop_node: LoopNode) -> LoopId {
        self.with_mut(|tree| {
            LoopId(tree.add(loop_node.name.clone(), Kind::Loop(loop_node), None))
        })
        .unwrap_or(LoopId(NodeId(u32::MAX)))
    }

    fn new_block(&mut self, name: NodeName) -> NodeId {
        self.with_mut(|tree| tree.add(name, Kind::Block, None))
            .unwrap_or(NodeId(u32::MAX))
    }

    fn add_child_node(&mut self, node: NodeId, at: InsertionPoint) {
        self.with_mut(|tree| tree.link(node, at));
    }

    fn move_node(&mut self, node: NodeId, at: InsertionPoint) {
        self.with_mut(|tree| {
            tree.unlink(node);
            tree.link(node, at);
        });
    }
}

impl PagedCursor<'_> {
    /// The loop minted at that id — total for the same reason [`DscTree`]'s is.
    fn minted(&self, at: LoopId) -> LoopNode {
        self.with(|tree| tree.loop_node(at).cloned())
            .flatten()
            .expect("a LoopId this tree's own walk produced names a loop of this tree")
    }
}

impl LoopBands for PagedCursor<'_> {
    fn set_loop_den(&mut self, loop_node: LoopId, den: DatastageId) {
        self.with_mut(|tree| tree.set_loop_den(loop_node, den));
    }

    fn set_loop_dims(&mut self, loop_node: LoopId, dims: LoopDims) {
        self.with_mut(|tree| tree.set_loop_dims(loop_node, dims));
    }

    fn move_children(&mut self, from: NodeId, to: NodeId) {
        self.with_mut(|tree| tree.move_children(from, to));
    }

    fn insert_perfectly_nested(&mut self, base: LoopId, nested: LoopId) {
        self.with_mut(|tree| {
            tree.move_children(base.0, nested.0);
            tree.link(nested.0, InsertionPoint::LastIn(base.0));
        });
    }

    /// ⛔ ASKED FOR AND NOT PORTED — `loopCond_.adjustConditionForSplitLoop` is a `dsc/` function
    /// outside this campaign's file list, and it is where the `EQ`/`NE`/`(GT,FIRST)`/`(LT,LAST)`
    /// rewrite and its four aborts live.
    fn adjust_condition_for_split_loop(
        &mut self,
        _condition: NodeId,
        _orig: LoopId,
        _new_loops: &[LoopId],
    ) {
        todo!(
            "LoopBands::adjust_condition_for_split_loop: wants the unported \
             dsc2::LoopCondComposite::adjustConditionForSplitLoop (dsc/dsc2.cpp:2061)"
        )
    }
}

impl L3TreeSurgery for PagedCursor<'_> {
    fn delete_child_node(&mut self, node: NodeId) {
        self.with_mut(|tree| tree.delete(node));
    }

    fn fresh_alloc(&mut self) -> AllocId {
        self.with_mut(TreeData::fresh_alloc)
            .unwrap_or(AllocId(u32::MAX))
    }

    fn new_allocate(&mut self, alloc: AllocId, node: L3AllocateNode) -> NodeId {
        self.with_mut(|tree| tree.add(node.name.clone(), Kind::Allocate(alloc, node), None))
            .unwrap_or(NodeId(u32::MAX))
    }

    fn new_transfer(&mut self, node: TransferNode) -> NodeId {
        self.with_mut(|tree| tree.add(node.name.clone(), Kind::Transfer(node), None))
            .unwrap_or(NodeId(u32::MAX))
    }

    fn new_sync(&mut self, node: SyncNode) -> NodeId {
        self.with_mut(|tree| tree.add(node.name.clone(), Kind::Sync(node), None))
            .unwrap_or(NodeId(u32::MAX))
    }

    fn add_sync_other_end(&mut self, sync: NodeId, other: NodeId) {
        self.with_mut(|tree| {
            if let Some(name) = tree.name(other) {
                tree.add_sync_other_end(sync, name);
            }
        });
    }

    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
        let Some(Some(node)) = self.with(|tree| tree.node_of_alloc(alloc)) else {
            return;
        };
        let Some(held) = self.state.dsc(self.at) else {
            return;
        };
        if let Some((lds, storage)) = held.home_of(node)
            && let Some(org) = held.org(lds)
        {
            org.add_user(storage, user);
        }
    }

    fn set_mem_org_allocation(&mut self, lds: LdsIdx, storage: SenComponent, alloc: AllocId) {
        let Some(held) = self.state.dsc(self.at) else {
            return;
        };
        let Some(Some(node)) = self.with(|tree| tree.node_of_alloc(alloc)) else {
            return;
        };
        let Some((_, minted)) = held.with(|tree| tree.allocate(node)) else {
            return;
        };
        if let Some(org) = held.org(lds) {
            org.set_node(storage, node, minted.clone());
        }
        if let Some(view) = ddc_view(&minted) {
            held.set_placed(node, view);
        }
    }

    fn set_buffering(&mut self, alloc: AllocId, buffering: Buffering) {
        self.with_mut(|tree| {
            if let Some(node) = tree.node_of_alloc(alloc) {
                tree.set_buffering(node, buffering);
            }
        });
    }

    fn set_transfer(&mut self, node: NodeId, transfer: TransferNode) {
        self.with_mut(|tree| tree.set_transfer(node, transfer));
    }

    fn loops_transfers_and_allocates(&self) -> Vec<L3WalkNode> {
        self.with(TreeData::loops_transfers_and_allocates)
            .unwrap_or_default()
    }

    fn allocate_node(&self, alloc: AllocId) -> Option<(NodeId, L3AllocateNode)> {
        let node = self.with(|tree| tree.node_of_alloc(alloc)).flatten()?;
        let (_, held) = self.with(|tree| tree.allocate(node)).flatten()?;
        Some((node, held))
    }

    fn mem_org_allocation(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId> {
        let held = self.state.dsc(self.at)?;
        let node = held.org(lds)?.node(storage)?;
        let (alloc, _) = held.with(|tree| tree.allocate(node))?;
        Some(alloc)
    }
}

/// ⭐ ONE DSC'S TREE AS THE COORDINATE PROPAGATION WALKS IT — the tree half of entry 374's surface,
/// which THIS STATE CAN ANSWER; the distribution seam beside it cannot (see
/// [`CoordPropTrees::coord_prop`]).
#[derive(Debug)]
pub struct CoordCursor<'s> {
    state: &'s DscState,
    at: DscIdx,
}

impl CoordPropTree for CoordCursor<'_> {
    fn allocation(&self, name: &NodeName) -> Option<PropagatedAllocation> {
        let held = self.state.dsc(self.at)?;
        let node = held.with(|tree| tree.find_named(&name.0))?;
        let (_, minted) = held.with(|tree| tree.allocate(node))?;
        let mut alloc = match held.placed(node) {
            Some(placed) => placed,
            None => ddc_view(&minted)?,
        };
        if let Some((lds, storage)) = held.home_of(node)
            && let Some(org) = held.org(lds)
            && let Some(users) = users_at(org, storage)
        {
            alloc.alloc_users = users;
        }
        let loops = held.with(|tree| {
            let mut loops = Vec::new();
            let mut at = tree.owner_loop(node);
            while let Some(owner) = at {
                if let Some(held) = tree.loop_node(owner) {
                    loops.push(held.clone());
                }
                at = tree.owner_loop(owner.0);
            }
            loops
        });
        let coordinate = held
            .with(|tree| tree.coordinate(node))
            .unwrap_or_else(Coordinate::default);
        Some(PropagatedAllocation {
            node,
            alloc,
            loops,
            coordinate,
        })
    }

    fn mem_org_allocation(&self, lds: LdsIdx, storage: SenComponent) -> Option<NodeName> {
        let held = self.state.dsc(self.at)?;
        let node = held.org(lds)?.node(storage)?;
        held.with(|tree| tree.name(node))
    }

    fn transfer(&self, node: NodeId) -> Option<TransferNode> {
        self.state.dsc(self.at)?.with(|tree| tree.transfer(node))
    }

    fn set_allocate_coordinate(&mut self, node: NodeId, coordinate: Coordinate) {
        if let Some(held) = self.state.dsc(self.at) {
            held.with_mut(|tree| tree.set_coordinate(node, coordinate));
        }
    }
}

/// `allocUsers_` at one storage of one organisation.
fn users_at(org: &Org, storage: SenComponent) -> Option<Vec<NodeId>> {
    match storage {
        SenComponent::Hbm => crate::schedule::l3::dsc::MemOrg::hbm_alloc_users(org),
        SenComponent::Lx => crate::schedule::l3::dsc::MemOrg::lx_alloc_users(org),
        _ => None,
    }
}

/// `getLayoutDims(ldsIdx)` on the DSC the coordinate cursor names.
#[derive(Debug, Clone, Copy)]
pub struct Layout;

impl Dsc for Layout {
    /// ⛔ WANTS `DesignSpaceConfig::getLayoutDims(ldsIdx)`, which is a field of the `&mut SuperDsc`
    /// no carrier may alias. Unreachable while [`CoordPropTrees::coord_prop`] refuses.
    fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
        todo!(
            "Dsc::layout_dims: wants DesignSpaceConfig::layout_dims, a field of the &mut SuperDsc \
             that `&'a mut E` cannot alias"
        )
    }
}

/// THE DISTRIBUTION AND CORELET-SLICE SEAM entry 374 propagates through — every method a `dsc/`
/// function outside this campaign's file list, and unreachable while
/// [`CoordPropTrees::coord_prop`] refuses.
#[derive(Debug, Default)]
pub struct CoordSeam {
    stages: DataStages<CoordDims>,
}

impl TemporalLoopDistribution for CoordSeam {
    type LoopParams = ();

    fn related_loops<'l>(
        &self,
        _dim: PrimaryDimAndKind,
        _chain: &[LoopAndDim<'l>],
        _pad: PadType,
    ) -> Vec<LoopAndDim<'l>> {
        todo!(
            "TemporalLoopDistribution::related_loops: wants dsc2::collectRelatedLoops \
             (dsc/dsc2.cpp:6575)"
        )
    }

    fn distribute(
        &self,
        _request: &ElemArrDistribution<'_>,
        _loop_params: &mut Self::LoopParams,
    ) -> Vec<FoldParamInfo> {
        todo!(
            "TemporalLoopDistribution::distribute: wants \
             dsc2::distributeElemArrToTemporalLoops (dsc/dsc2.cpp:5934)"
        )
    }

    fn distributed(
        &self,
        _loop_params: &Self::LoopParams,
        _loop_node: &LoopNode,
        _dim: PrimaryDim,
    ) -> Option<DistributedLoop> {
        todo!(
            "TemporalLoopDistribution::distributed: wants \
             loopParamsAfterDistribution.at(loopNode).at(dim)"
        )
    }
}

impl AllocCoordinateSeam for CoordSeam {
    fn parametric_iter_count(
        &self,
        _loop_node: &LoopNode,
    ) -> Option<crate::schedule::dsc2::FoldCardinality> {
        todo!(
            "AllocCoordinateSeam::parametric_iter_count: wants LoopNode::parametricIterCount(dsc, \
             0, NO_COMPONENT, -1)"
        )
    }

    fn comp_view(&self, _stage: DatastageId, _dim: PrimaryDim) -> Option<Extent> {
        todo!(
            "AllocCoordinateSeam::comp_view: wants dataStageParam_.at(stage).ss_\
             .dataStageDimToVal_compView_st(dim, L3LU, 0) on the &mut SuperDsc"
        )
    }

    fn stage_padding(&self, _stage: DatastageId) -> Option<&BTreeMap<PrimaryDim, DimPadding>> {
        todo!(
            "AllocCoordinateSeam::stage_padding: wants dataStageParam_.at(stage).ss_.paddingSizes_ \
             on the &mut SuperDsc"
        )
    }
}

impl CoreletSliceSeam for CoordSeam {
    type Dims = CoordDims;

    fn stages(&self) -> &DataStages<Self::Dims> {
        &self.stages
    }

    fn stages_mut(&mut self) -> &mut DataStages<Self::Dims> {
        &mut self.stages
    }

    fn loop_relevant(
        &self,
        _dim: PrimaryDimAndKind,
        _loop_node: &LoopNode,
        _pad: PadType,
    ) -> bool {
        todo!(
            "CoreletSliceSeam::loop_relevant: wants dsc2::loopRelevantForDim (dsc/dsc2.cpp:6550)"
        )
    }

    fn lx_below_chunk_loops(&self) -> Option<Vec<&LoopNode>> {
        todo!(
            "CoreletSliceSeam::lx_below_chunk_loops: wants the lx_below block's owner-loop chain, \
             innermost first and without the outermost loop"
        )
    }
}

/// The extents payload one corelet-sliced data stage carries.
#[derive(Debug, Clone, Default)]
pub struct CoordDims;

impl CoreletSliceDims for CoordDims {
    fn first_corelet_split_dim(&self) -> Option<PrimaryDim> {
        todo!("CoreletSliceDims::first_corelet_split_dim: wants coreletSplit_.begin()->first")
    }

    fn divide_for_corelets(
        &mut self,
        _dim: PrimaryDim,
        _corelets: crate::schedule::l3::dsc::CoreletsUsed,
    ) {
        todo!("CoreletSliceDims::divide_for_corelets: wants primaryDimToValHandler_st(dim) /= n")
    }

    fn comp_view_extent(&self, _dim: PrimaryDim, _comp: SenComponent) -> Option<Extent> {
        todo!("CoreletSliceDims::comp_view_extent: wants dataStageDimToVal_compView_st(dim, comp)")
    }
}

/// ⭐ THE SAME ALLOCATE NODE IN THE ddc VOCABULARY — `memOrg_.at(storage).allocateNode_` is ONE
/// `dsc2::AllocateNode`, and this is the L3 view of it read as that node.
///
/// ⛔ NOTHING IS PLACED HERE. `startAddressCoreCorelet_` and `bufferOffsetCoreCorelet_` come back
/// EMPTY, which is the truth about a freshly minted allocation: entries 219/220/222/292 are what
/// place them, and a value invented here would be a fabricated address.
///
/// ⛔ AND `maxDimSizes_` CROSSES UNCHANGED: [`crate::schedule::ddc::fold::AllocLayout`] holds a
/// RESOLVED extent or nothing, and nothing IS the reference's `-1`, which is [`MaxDimSize::Unset`].
/// [`None`] is a layout naming NO dim, which `dsc2::AllocLayout`'s own non-emptiness forbids —
/// `layoutDimOrder_.at(0)` (`ddc/ddcv1.cpp:1704`) discharged by the type.
fn ddc_view(minted: &L3AllocateNode) -> Option<AllocateNode> {
    let max = |size: Option<Elements>| match size {
        Some(size) => MaxDimSize::Resolved(size),
        None => MaxDimSize::Unset,
    };
    let mut layout = minted.layout.0.iter().map(|(dim, size)| (*dim, max(*size)));
    let first = layout.next()?;
    Some(AllocateNode {
        name: minted.name.clone(),
        component: minted.component,
        lds: Some(minted.lds),
        const_idx: None,
        temp_storage_for_compute: None,
        layout: AllocLayout::new(first, layout.collect()),
        start_address: StartAddress::default(),
        placement: AllocPlacement {
            num_buffers: match minted.buffering {
                Buffering::None => NumBuffers::Single,
                Buffering::Double => NumBuffers::Double,
                Buffering::Streaming => NumBuffers::Streaming,
            },
            padding: Padding::default(),
            buffer_offset: BTreeMap::new(),
            is_start_addr_symbolic: false,
        },
        gap_stick_spread: BTreeMap::new(),
        alloc_users: Vec::new(),
    })
}

/// The unused-field silencer for the seam's own loop parameters, which
/// [`CoordPropTrees::coord_prop`] would hand out.
const _: fn(&Env<'_>) = |env| {
    let _ = &env.coord;
    let _ = &env.seam;
    let _ = &env.loop_params;
    let _ = &env.layout;
    let _: &RefCell<()> = &RefCell::new(());
    let _: Bytes = Bytes(0);
};
