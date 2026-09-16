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
    TemporalLoopDistribution, comp_row_id,
};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::{
    DataStages, InsertionPoint, LoopBands, LoopDims, LoopNode, PaddingForm, PrimaryDimAndKind,
    ScheduleSurgery,
};
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{
    AllocLayout, AllocPlacement, AllocateNode, BlockNode, Coordinate, Dsc, LayoutDims, LdsIdx,
    LoopCondComposite, MaxDimSize, NodeName, NumBuffers, Padding, StartAddress, SyncNode,
    TransferNode,
};
use crate::schedule::l3::dl_ops::{
    AllocCoordinateSeam, AllocationReads, AllocationSites, AllocationView, ChunkLoopNest,
    CoordPropTree, CoordPropTrees, CoreWindowDims,
    CoreletSliceDims, CoreletSliceSeam, DscCoordProp, DscGtrSurgery, DscL3Surgery, DscPagedTrees,
    DscSyncSurgery, DscTransferSizes, DscTransferWrites, DscTransfers, DscTreeSurgery, DscTrees,
    GtrGroupId, L3AllocateNode, L3Sync, L3TreeSurgery, L3WalkNode, LoopAndDim, LxZeroPadTransform,
    PropagatedAllocation,
};
use crate::schedule::l3::dsc::{
    Buffering, CoreletsUsed, DataStages as L3DataStages, DimPadding, DscIdx, FilledDims,
    L3Transfer, MemOrgs, StageDims, SuperChunkStage, SuperDsc, TransferNodes,
};
use crate::units::{Corelet, Row};

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
    /// ⭐ `dataStageParam_` PER DSC, SNAPSHOT BEFORE THE STAGE RUNS AND PROVABLY FRESH for the same
    /// reason [`Self::core_windows`] is: the one 2a unit that would ADD an entry is
    /// [`ChunkLoopNest::mint_super_chunk_stage`], which refuses, and the one that WRITES an entry it
    /// is handed is entry 229 — which mints its denominator stage into the map it was given and
    /// `currDsc->dataStageParam_.erase(denId)`s it again before returning
    /// (`L3DlOpsScheduler.cpp:7673`), so no write of stage 2a's crosses back into the super-DSC.
    stage_seeds: Vec<DataStages<CoordDims>>,
    /// `getLayoutDims(ldsIdx)` PER DSC — the answer the walk landed on, which
    /// [`crate::schedule::l3::dsc::DesignSpaceConfig::layout_dims`] already states per labelled DS.
    /// Stage 2a mints allocate nodes but never rewrites `layoutDimOrder_` on one the wire stated.
    layouts: Vec<BTreeMap<LdsIdx, LayoutDims>>,
    /// The cursor [`DscPagedTrees::tree_mut`] hands out.
    paged: PagedCursor<'s>,
    /// The cursor [`CoordPropTrees::coord_prop`] would hand out.
    coord: CoordCursor<'s>,
    /// The distribution and corelet-slice seam that same accessor hands out, positioned by
    /// [`Env::position_coord`].
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
            stage_seeds: sdsc
                .dscs()
                .iter()
                .map(|dsc| coord_stages(&dsc.data_stages))
                .collect(),
            layouts: sdsc
                .dscs()
                .iter()
                .map(|dsc| dsc.layout_dims.clone())
                .collect(),
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
            layout: Layout::default(),
        }
    }

    /// That DSC's tree, [`None`] for a `dscs_` position the state holds none for.
    fn dsc(&self, at: DscIdx) -> Option<&'s DscTree> {
        self.state.dsc(at)
    }

    /// ⭐ THE COORDINATE-PROPAGATION CURSORS AND SEAM MOVED ONTO ONE DSC — `currDsc` for entry 374,
    /// which is per-DSC where this carrier holds ONE of each. The seam takes a FRESH COPY of that
    /// DSC's data stages every time it is positioned, so entry 229's mint-and-erase of a denominator
    /// stage cannot leak from one DSC's propagation into the next.
    ///
    /// ⛔ [`None`] IS *"a `dscs_` position this carrier holds no surface for"* — the same absence
    /// [`CoordPropTrees::coord_prop`] answers with.
    fn position_coord(&mut self, at: DscIdx) -> Option<()> {
        let index = usize::try_from(at.0).ok()?;
        let stages = self.stage_seeds.get(index)?.clone();
        let chunk_loops = self.state.dsc(at)?.with(chunk_loop_chain);
        self.seam = CoordSeam {
            stages,
            chunk_loops,
        };
        self.layout = Layout::of(self.layouts.get(index)?.clone());
        self.coord.at = at;
        Some(())
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

impl AllocationReads for Env<'_> {
    fn allocation(
        &self,
        dsc: DscIdx,
        lds: LdsIdx,
        storage: SenComponent,
    ) -> Option<AllocationView> {
        Some(AllocationView::of(self.dsc(dsc)?.org(lds)?.placed(storage)?))
    }
}

impl AllocationSites for Env<'_> {
    /// ⛔⛔ THE WRITE-BACK IS UNCONDITIONAL AND THAT IS THE REFERENCE: `allocNode` is a POINTER
    /// (`L3DlOpsScheduler.cpp:1642`, `:5687`), so a placement that refuses part-way leaves what it had
    /// already written. A `place` that returned before touching the node writes the value it read,
    /// which is the same cell — so the only observable effect is `place`'s own.
    fn place_allocation(
        &mut self,
        dsc: DscIdx,
        lds: LdsIdx,
        storage: SenComponent,
        place: &mut dyn FnMut(&mut AllocateNode) -> Option<()>,
    ) -> Option<Option<()>> {
        let held = self.dsc(dsc)?;
        let org = held.org(lds)?;
        let mut node = org.placed(storage)?;
        let placed = place(&mut node);
        org.set_placed(storage, node);
        Some(placed)
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
        self.core_windows.get(usize::try_from(dsc.0).ok()?).cloned()
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

    /// ⛔ STILL REFUSES, AND WHAT IS LEFT IS THREE `dsc/dsc2.cpp` FUNCTIONS AND NOTHING ELSE.
    /// The tree half is answerable ([`CoordCursor`]), and so now are every `dataStageParam_` read
    /// and the layout order: [`Dsc::layout_dims`], [`AllocCoordinateSeam::comp_view`],
    /// [`AllocCoordinateSeam::stage_padding`], [`AllocCoordinateSeam::parametric_iter_count`],
    /// [`CoreletSliceSeam::lx_below_chunk_loops`] and all three [`CoreletSliceDims`] methods are
    /// answered off [`Env::position_coord`]'s snapshot. What remains is
    /// `dsc2::distributeElemArrToTemporalLoops` (`dsc/dsc2.cpp:5934`), `dsc2::collectRelatedLoops`
    /// (`:6575`) and `dsc2::loopRelevantForDim` (`:6550`) — declared OUTSIDE this campaign's file
    /// list, so they are crustify units and not carrier facts. Handing the surface out before they
    /// land would turn this refusal into a `todo!` inside the distributor.
    ///
    /// ⭐ THE POSITIONING RUNS ANYWAY, so the `?` below is the trait's own *"a `dscs_` position the
    /// caller holds no surface for"* and the refusal beside it is the seam's; when the three
    /// functions land, the refusal is the only line that goes.
    fn coord_prop(
        &mut self,
        dsc: DscIdx,
    ) -> Option<DscCoordProp<'_, Self::Layout, Self::Org, Self::Tree, Self::Env>> {
        self.position_coord(dsc)?;
        self.state.refuse(
            "CoordPropTrees::coord_prop: wants dsc2::distributeElemArrToTemporalLoops / \
             collectRelatedLoops / loopRelevantForDim, the three unported dsc2 functions entry 374 \
             distributes through",
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
        self.with_mut(|tree| LoopId(tree.add(loop_node.name.clone(), Kind::Loop(loop_node), None)))
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

/// `getLayoutDims(ldsIdx)` on the DSC the coordinate cursor names — the ANSWER that walk landed on,
/// per labelled DS, as [`DesignSpaceConfig::layout_dims`] states it.
#[derive(Debug, Clone, Default)]
pub struct Layout {
    orders: BTreeMap<LdsIdx, LayoutDims>,
}

impl Layout {
    /// One DSC's stated layout orders.
    #[must_use]
    pub const fn of(orders: BTreeMap<LdsIdx, LayoutDims>) -> Self {
        Self { orders }
    }
}

impl Dsc for Layout {
    /// ⭐ A READ AND NOT A RE-DERIVATION, which is what the trait's own note asks for: the walk over
    /// `memOrg_` and `referenceLdsIdx_` is [`crate::schedule::dsc2::layout_dims`] (entry 006), the
    /// wire runs it, and `DesignSpaceConfig::layout_dims` carries what it landed on. Walking it a
    /// second time here would be a second answer that could disagree.
    ///
    /// ⛔ THE `todo!` IS `getLayoutDims`' OWN `DT_CHECK(allocNode)` (`dsc/dsc2.cpp:4022`) AND NOTHING
    /// ELSE — an lds this DSC resolves NO allocate node for. It is reached for no lds the wire stated
    /// an order for, and a fabricated order here would be an invented layout: `LayoutDims` is
    /// non-empty by construction, so there is no absence to answer with.
    fn layout_dims(&self, lds: LdsIdx) -> LayoutDims {
        match self.orders.get(&lds) {
            Some(order) => order.clone(),
            None => todo!(
                "Dsc::layout_dims: this DSC states no getLayoutDims answer for {lds:?}, which is \
                 the reference's own DT_CHECK(allocNode) (dsc/dsc2.cpp:4022) — a labelled DS whose \
                 memOrg_ chain resolves to no allocate node"
            ),
        }
    }
}

/// THE DISTRIBUTION AND CORELET-SLICE SEAM entry 374 propagates through, POSITIONED ON ONE DSC by
/// [`Env::position_coord`] — its `dataStageParam_` snapshot and the lx_below loop chain, which is
/// every fact of this surface that is not one of the three unported `dsc2` functions.
#[derive(Debug, Default)]
pub struct CoordSeam {
    /// `currDsc->dataStageParam_` — see [`Env::stage_seeds`] for why a copy is faithful.
    stages: DataStages<CoordDims>,
    /// [`chunk_loop_chain`] on this DSC's tree, [`None`] where it holds no lx_below block.
    chunk_loops: Option<Vec<LoopNode>>,
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

    /// ⛔ PAIRED WITH [`TemporalLoopDistribution::distribute`] AND NOT SEPARATELY ANSWERABLE:
    /// `loopParamsAfterDistribution` is that call's OUT-PARAMETER, and [`Self::LoopParams`] is `()`
    /// because nothing fills it here. ⛔ [`None`] WOULD BE A FALSE ANSWER, not a refusal — the
    /// reference's distribution states an entry for EVERY loop it was handed
    /// (`L3DlOpsScheduler.cpp:7484`, `:7691`), so *"no such loop"* is a sentence about the
    /// distribution that this seam has not run, and its callers would build a coordinate with a fold
    /// silently missing.
    fn distributed(
        &self,
        _loop_params: &Self::LoopParams,
        _loop_node: &LoopNode,
        _dim: PrimaryDim,
    ) -> Option<DistributedLoop> {
        todo!(
            "TemporalLoopDistribution::distributed: wants \
             loopParamsAfterDistribution.at(loopNode).at(dim), which \
             dsc2::distributeElemArrToTemporalLoops (dsc/dsc2.cpp:5934) is what fills"
        )
    }
}

impl AllocCoordinateSeam for CoordSeam {
    /// ⭐⭐ [`None`] FOR EVERY LOOP THIS ARENA CAN HOLD, AND THAT IS THE REFERENCE'S OWN ARM RATHER
    /// THAN A REFUSAL. The callsite is `if (loop->isParametricLoop()) { loop->parametricIterCount(
    /// &dsc, 0, NO_COMPONENT, -1); } else { <the two `dataStageDimToVal_compView_st` reads> }`
    /// (`L3DlOpsScheduler.cpp:7464-7482`), and the function itself opens with `DT_ERROR("Requested
    /// parametricIterCount on a non-parametric loop")` (`dsc/dsc2.cpp:4126-4131`) — so a loop that is
    /// not parametric has no count to answer, and the caller's other arm is
    /// [`AllocCoordinateSeam::comp_view`], which this carrier answers.
    ///
    /// ⛔ AND NO LOOP HERE IS PARAMETRIC, BY THE TYPE: `isParametricLoop()` is `parametricLds_ != -1`
    /// on a `dsc2::LoopNode`, and [`LoopNode`] does not carry that field at all — the same fact
    /// [`ScheduleSurgery::is_parametric`] answers `false` with, for the same reason.
    fn parametric_iter_count(
        &self,
        _loop_node: &LoopNode,
    ) -> Option<crate::schedule::dsc2::FoldCardinality> {
        None
    }

    /// `dataStageParam_.at(stage).ss_.dataStageDimToVal_compView_st(dim, L3LU, 0)` — one corelet's
    /// view, on the snapshot [`Env::position_coord`] took.
    ///
    /// ⛔ [`None`] IS `primaryDimToVal_st`'S OWN `-1` AND ITS `.at()` STOPS (see [`comp_view_of`]),
    /// plus a stage this DSC states no dims for — which is the extent-less
    /// [`crate::schedule::l3::dsc::EmptyStage`] `ddl.datastage` mints, whose every slot the reference
    /// reads as that same `-1`.
    fn comp_view(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Extent> {
        comp_view_of(
            self.stages.0.get(&stage)?.ss.dims.stated()?,
            dim,
            SenComponent::L3lu,
            Some(Corelet::at::<0>()),
        )
    }

    /// `dataStageParam_.at(stage).ss_.paddingSizes_` — ⛔ [`None`] IS THAT `.at()`'S THROW AND
    /// NOTHING ELSE: an extent-less stage HOLDS an (empty) `paddingSizes_` in the reference, so it
    /// answers the empty map here rather than an absence (see [`CoordDims::padding`]).
    fn stage_padding(&self, stage: DatastageId) -> Option<&BTreeMap<PrimaryDim, DimPadding>> {
        Some(self.stages.0.get(&stage)?.ss.dims.padding())
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

    fn loop_relevant(&self, _dim: PrimaryDimAndKind, _loop_node: &LoopNode, _pad: PadType) -> bool {
        todo!("CoreletSliceSeam::loop_relevant: wants dsc2::loopRelevantForDim (dsc/dsc2.cpp:6550)")
    }

    /// `getLxBelowBlockNode(scheduleTree_)->getMutableOwnerLoop()` walked upward — the chain
    /// [`chunk_loop_chain`] gathers, INNERMOST FIRST and WITHOUT the root loop.
    ///
    /// ⛔ [`None`] IS `DT_CHECK_MSG(lxBelowBlockNode, "Expect a valid lx_below block node.")`
    /// (`L3DlOpsScheduler.cpp:7599`); an lx_below block with NO owner loop answers the EMPTY chain,
    /// which is the reference's `while (currLoop)` never entering.
    fn lx_below_chunk_loops(&self) -> Option<Vec<&LoopNode>> {
        Some(self.chunk_loops.as_ref()?.iter().collect())
    }
}

/// ONE DATA-STAGE HALF'S EXTENTS — `DataStructDims` as `dataStageParam_` holds it, or NO EXTENTS AT
/// ALL for the stage `ddl.datastage` leaves extent-less.
///
/// ⭐ [`None`] IS [`crate::schedule::l3::dsc::EmptyStage`], WHICH IS AN ENTRY AND NOT AN ABSENCE:
/// `dataStageParam_.count(id)` is 1 for it, so `paddingSizes_` answers the EMPTY MAP while every
/// `primaryDimToVal_st` answers the `-1` an unwritten slot carries. Folding the two into one absence
/// would make [`AllocCoordinateSeam::stage_padding`] refuse for the IBR and one-page stages every
/// chunk loop nest names as a denominator.
#[derive(Debug, Clone, Default)]
pub struct CoordDims(Option<FilledDims>);

impl CoordDims {
    /// A stage half that states dims.
    #[must_use]
    pub const fn of(dims: FilledDims) -> Self {
        Self(Some(dims))
    }

    /// The extents themselves, [`None`] for the extent-less stage.
    #[must_use]
    pub fn stated(&self) -> Option<&StageDims> {
        Some(self.0.as_ref()?.dims())
    }

    /// `paddingSizes_` — the extent-less stage's is EMPTY, which is what the reference's own
    /// default-inserted entry carries.
    #[must_use]
    pub fn padding(&self) -> &BTreeMap<PrimaryDim, DimPadding> {
        /// `dataStageParam_[id].ss_.paddingSizes_` on an entry nothing has written.
        static NONE_STATED: BTreeMap<PrimaryDim, DimPadding> = BTreeMap::new();
        match &self.0 {
            Some(dims) => &dims.dims().padding,
            None => &NONE_STATED,
        }
    }
}

impl CoreletSliceDims for CoordDims {
    /// `coreletSplit_.begin()->first` — `coreletSplit_` is a `std::map<PrimaryDimTypes, ..>`
    /// (`dsc/dims.h:206`) and [`PrimaryDim`]'s [`Ord`] is that same numbering, so the FIRST key of
    /// the [`BTreeMap`] is the reference's `begin()`.
    fn first_corelet_split_dim(&self) -> Option<PrimaryDim> {
        self.stated()?.corelet_split.keys().next().copied()
    }

    /// `primaryDimToValHandler_st(dim) /= n`, AND every `coreletSplit_.at(dim)` share alike
    /// (`L3DlOpsScheduler.cpp:7552-7573`) — the chunk half-and-half slicing strategy on this half.
    ///
    /// ⛔ A SLOT THE STAGE DOES NOT STATE IS LEFT UNSTATED. `primaryDimToValHandler_st` hands back the
    /// raw `double&`, so the reference divides the `-1` an unwritten slot carries and leaves `-1/n`
    /// there; keeping the slot ABSENT keeps `primaryDimToVal_st`'s answer the same `-1` instead of the
    /// `0` that truncation would make of it. The dim asked for is the chunk stage's own
    /// `coreletSplit_` key, so the slot is stated on every path that reaches here.
    ///
    /// ⛔ DIVERGENCE, AS EVERYWHERE ELSE IN THIS CRATE: INTEGER DIVISION where the reference divides a
    /// `double` and truncates on the READ. Composed divisions agree for positive extents; a value read
    /// back before a second divide differs by less than one element.
    fn divide_for_corelets(&mut self, dim: PrimaryDim, corelets: CoreletsUsed) {
        let Some(dims) = self.0.as_mut() else {
            return;
        };
        let corelets = i64::from(corelets.get());
        if let Some(extent) = dims.dims().extent(dim) {
            dims.set_extent(dim, Extent(extent.0 / corelets));
        }
        if let Some(shares) = dims.corelet_split_mut().get_mut(&dim) {
            for share in shares.iter_mut() {
                *share = Extent(share.0 / corelets);
            }
        }
    }

    /// `dataStageDimToVal_compView_st(dim, comp)` — every corelet, no padding, full density.
    fn comp_view_extent(&self, dim: PrimaryDim, comp: SenComponent) -> Option<Extent> {
        comp_view_of(self.stated()?, dim, comp, None)
    }
}

/// Replaces: `DataStructDims::dataStageDimToVal_compView_st` (`dsc/dims.cpp:708-717`)
///
/// ⭐ THE WHOLE OF IT IS A ROW LOOKUP AND `primaryDimToVal_st`, WHICH IS PORTED: the reference derives
/// `ptrowId` from the component through `EnumsConversion::senCompToRowId` — [`comp_row_id`], entry 082
/// — and hands the SAME component on as `peOrSfp`, which is [`v1::sampled_as`]. So the three
/// arguments are one [`v1::DimSample`] and the answer is [`StageDims::sampled_extent`] (entry 012).
///
/// ⛔ [`None`] IS `primaryDimToVal_st`'S OWN `-1` AND ITS `.at()` STOPS, plus ONE OF OUR OWN: a
/// component on PT row 7 where this arch has four rows is [`Row::checked`]'s absence, and the
/// reference indexes `rowSplit_.at(clId).at(7)` there, which throws. Folding it into `row = -1` would
/// answer the SUM over the rows for a component that names one.
///
/// ⚠️ ITS HOME SHOULD BE [`StageDims`] ITSELF, beside `sampled_extent`: `stages/ddc_reads.rs` and
/// `stages/offsets.rs` each hold a `todo!` for this same wrapper, and one method on the type would
/// answer all three. It is written here because this campaign owns this file alone.
fn comp_view_of(
    dims: &StageDims,
    dim: PrimaryDim,
    comp: SenComponent,
    corelet: Option<Corelet>,
) -> Option<Extent> {
    let row = match comp_row_id(comp) {
        Some(held) => Some(Row::checked(u32::from(held.ordinal()))?),
        None => None,
    };
    dims.sampled_extent(
        dim,
        v1::DimSample {
            comp: v1::sampled_as(comp),
            row,
            corelet,
        },
        &PaddingForm::default(),
        None,
        false,
    )
}

/// ONE DSC'S `dataStageParam_` IN THE VOCABULARY THE SEAM'S ACCESSORS TRAFFIC IN — every stage the
/// DSC holds, the extent-less ones included, projected onto [`CoordDims`].
///
/// ⚠️ THE IDS SEEDED ARE `0..next_index()`: [`L3DataStages`] holds its core and chunk stages apart
/// from its minted ones and exposes `at`, `holds` and `next_index` but NO iterator over the ids it
/// holds, so a stage minted at an id at or past the first free one is not seeded —
/// [`AllocCoordinateSeam::comp_view`] and [`AllocCoordinateSeam::stage_padding`] then REFUSE for it
/// rather than answer a wrong extent. `next_index()` is the count-or-later, so a densely numbered
/// map — which is what `ddl.datastage` produces — is seeded whole.
fn coord_stages(held: &L3DataStages) -> DataStages<CoordDims> {
    use crate::schedule::ddc::transformation_util::{
        DataStage as UtilStage, StageDims as UtilHalf,
    };
    let mut stages: DataStages<CoordDims> = DataStages::default();
    for id in (0..=held.next_index().0).map(DatastageId) {
        let stage = match (held.at(id), held.empty_stage(id)) {
            (Some(held), _) => UtilStage {
                ss: UtilHalf {
                    name: held.ss.name.clone(),
                    dims: CoordDims::of(held.ss.dims.clone()),
                },
                el: UtilHalf {
                    name: held.el.name.clone(),
                    dims: CoordDims::of(held.el.dims.clone()),
                },
            },
            // `dataStageParam_[id]` as `ddl.datastage` leaves it: an entry that states no dim, whose
            // two halves carry the one name the reference writes onto both.
            (None, Some(empty)) => UtilStage {
                ss: UtilHalf {
                    name: empty.name.clone(),
                    dims: CoordDims::default(),
                },
                el: UtilHalf {
                    name: empty.name.clone(),
                    dims: CoordDims::default(),
                },
            },
            (None, None) => continue,
        };
        stages.0.insert(id, stage);
    }
    stages
}

/// `getLxBelowBlockNode(scheduleTree_)`, then `getMutableOwnerLoop()` upward while the loop reached
/// HAS an owner loop — *"Exclude the root loop"* (`L3DlOpsScheduler.cpp:7600-7614`), which breaks on a
/// loop with no OWNER LOOP and not on one with no parent block.
///
/// ⛔ [`None`] IS THE `DT_CHECK_MSG` FOR THE BLOCK, or a loop id this tree holds no loop at — the
/// reference's own pointer being null, which cannot happen in a well-formed tree.
fn chunk_loop_chain(tree: &TreeData) -> Option<Vec<LoopNode>> {
    let block = tree.lx_below_block()?;
    let mut chain = Vec::new();
    let mut at = tree.owner_loop(block);
    while let Some(current) = at {
        let owner = tree.owner_loop(current.0);
        if owner.is_none() {
            break;
        }
        chain.push(tree.loop_node(current)?.clone());
        at = owner;
    }
    Some(chain)
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

#[cfg(test)]
mod tests {
    //! ⭐⭐ THE COORDINATE-PROPAGATION SEAM DRIVEN DIRECTLY, WHICH IS THE ONLY WAY IT CAN BE TESTED
    //! AT ALL: [`CoordPropTrees::coord_prop`] still refuses (three unported `dsc2` functions), so a
    //! test that ran a stage would exercise none of this. Every case below asks the seam, the dims
    //! payload or the chain walk itself.
    //!
    //! ⛔ AND EVERY ONE CARRIES VALUES, NOT VERDICTS. The two that could pass by accident are called
    //! out where they are: `comp_view` must read CORELET ZERO's share (a `clId = -1` spelling answers
    //! the whole core's extent instead), and `comp_view_extent` must DERIVE the PT row from the
    //! component (a `row = -1` spelling answers the whole dim instead).

    use super::*;

    use crate::schedule::ddc::fold::{Dilation, Stride};
    use crate::schedule::ddc::metadata::MetaDimKind;
    use crate::schedule::l3::dsc::{DataStage as L3DataStage, NamedDims, PadSizes, UnneededPad};

    /// A stage half stating those extents and nothing else.
    fn extents(stated: &[(PrimaryDim, i64)]) -> FilledDims {
        let mut dims = StageDims::default();
        for &(dim, extent) in stated {
            dims.extents.insert(dim, Extent(extent));
        }
        FilledDims::of(dims).expect("a stage half that states a dim")
    }

    /// `dataStageParam_[id]` with both halves stating the same extents.
    fn stage(name: &str, stated: &[(PrimaryDim, i64)]) -> L3DataStage {
        let name = crate::schedule::ddc::transformation_util::StageName(name.to_owned());
        L3DataStage {
            ss: NamedDims {
                name: name.clone(),
                dims: extents(stated),
            },
            el: NamedDims {
                name,
                dims: extents(stated),
            },
        }
    }

    /// One `paddingSizes_` entry, its window dim the fact the assertions read back.
    fn padding(window: PrimaryDim) -> DimPadding {
        DimPadding {
            sizes: PadSizes::Voided,
            window_dim: Some(window),
            unneeded: UnneededPad::default(),
            stride: Stride::ONE,
            dilation: Dilation(1),
        }
    }

    /// A seam over those stages, positioned on no tree.
    fn seam(stages: DataStages<CoordDims>) -> CoordSeam {
        CoordSeam {
            stages,
            chunk_loops: None,
        }
    }

    /// `loop_ds<num>_ds<den>_<dim>` over one dim.
    fn loop_node(num: u32, den: u32, dim: PrimaryDim) -> LoopNode {
        crate::schedule::ddc::transformation_util::construct_loop_node(
            DatastageId(num),
            DatastageId(den),
            LoopDims::new(
                PrimaryDimAndKind {
                    dim,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        )
    }

    /// ⭐ `coreletSplit_.begin()->first` IS THE LOWEST KEY, and the slice divides BOTH the dim's own
    /// slot and every corelet share of it — `L3DlOpsScheduler.cpp:7552-7573`, carried as values.
    #[test]
    fn the_slice_divides_the_dims_slot_and_every_corelet_share_of_it() {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::Out, Extent(64));
        dims.extents.insert(PrimaryDim::Y, Extent(8));
        // Two split dims: `Out` (ordinal 1) precedes `Y` (ordinal 11), so `begin()` is `Out`.
        dims.corelet_split
            .insert(PrimaryDim::Y, vec![Extent(4), Extent(4)]);
        dims.corelet_split
            .insert(PrimaryDim::Out, vec![Extent(32), Extent(32)]);
        let mut held = CoordDims::of(FilledDims::of(dims).expect("a stated half"));

        assert_eq!(
            held.first_corelet_split_dim(),
            Some(PrimaryDim::Out),
            "the reference reads `coreletSplit_.begin()->first`, which is the lowest dim key"
        );

        held.divide_for_corelets(PrimaryDim::Out, CoreletsUsed::ONE);
        assert_eq!(
            held.stated().and_then(|dims| dims.extent(PrimaryDim::Out)),
            Some(Extent(64)),
            "one corelet divides by one"
        );

        let two = CoreletsUsed::new(std::num::NonZeroU32::new(2).expect("two"));
        held.divide_for_corelets(PrimaryDim::Out, two);
        let stated = held.stated().expect("the half still states its dims");
        assert_eq!(
            stated.extent(PrimaryDim::Out),
            Some(Extent(32)),
            "`primaryDimToValHandler_st(dim) /= n` — the slot itself is halved"
        );
        assert_eq!(
            stated.corelet_split.get(&PrimaryDim::Out),
            Some(&vec![Extent(16), Extent(16)]),
            "and every `coreletSplit_.at(dim)` share alike — a write that only touched the slot \
             leaves 32 here"
        );
        assert_eq!(
            stated.extent(PrimaryDim::Y),
            Some(Extent(8)),
            "the dim NOT asked for is untouched"
        );
        assert_eq!(
            stated.corelet_split.get(&PrimaryDim::Y),
            Some(&vec![Extent(4), Extent(4)]),
            "and so are its shares"
        );
    }

    /// ⭐ THE EXTENT-LESS STAGE IS AN ENTRY AND NOT AN ABSENCE — `ddl.datastage`'s bare
    /// `dataStageParam_[id]`: every `primaryDimToVal_st` answers the `-1` this crate spells [`None`],
    /// while `paddingSizes_` is an EMPTY MAP the reference can and does read.
    #[test]
    fn an_extent_less_stage_states_no_dim_and_an_empty_padding_map() {
        let mut held = CoordDims::default();
        assert_eq!(held.stated(), None);
        assert_eq!(held.first_corelet_split_dim(), None);
        assert_eq!(
            held.comp_view_extent(PrimaryDim::Out, SenComponent::Lx),
            None
        );
        assert!(
            held.padding().is_empty(),
            "an entry nothing wrote carries an empty paddingSizes_, which is not the same as no entry"
        );
        // A divide on a stage that states nothing writes nothing — the reference divides the `-1` a
        // slot carries and this keeps the slot UNSTATED rather than truncating it to zero.
        held.divide_for_corelets(PrimaryDim::Out, CoreletsUsed::ONE);
        assert_eq!(held.stated(), None);
    }

    /// ⭐⭐ THE PT ROW COMES FROM THE COMPONENT — `dataStageDimToVal_compView_st`'s whole body
    /// (`dsc/dims.cpp:708-717`): `senCompToRowId` then `primaryDimToVal_st`. ⛔ A SPELLING THAT PASSED
    /// `row = -1` ANSWERS 64 FOR EVERY ROW, which is what this test discriminates.
    #[test]
    fn the_comp_view_derives_the_pt_row_from_the_component() {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::Y, Extent(64));
        dims.row_split.insert(
            PrimaryDim::Y,
            BTreeMap::from([(
                Corelet::at::<0>(),
                vec![Extent(10), Extent(20), Extent(30), Extent(40)],
            )]),
        );
        let held = CoordDims::of(FilledDims::of(dims).expect("a stated half"));

        assert_eq!(
            held.comp_view_extent(PrimaryDim::Y, SenComponent::Ptrow2),
            Some(Extent(30)),
            "`ptrowId` 2 reads `rowSplit_.at(dim).begin()->second.at(2)`"
        );
        assert_eq!(
            held.comp_view_extent(PrimaryDim::Y, SenComponent::Ptrow0),
            Some(Extent(10)),
            "and row 0 reads the first share, not the sum"
        );
        assert_eq!(
            held.comp_view_extent(PrimaryDim::Y, SenComponent::Lx),
            Some(Extent(64)),
            "a component `senCompToRowId` does not name is the reference's `-1` row, which falls \
             through to the whole-core extent"
        );
    }

    /// ⭐⭐ `comp_view` READS CORELET ZERO — `dataStageDimToVal_compView_st(dim, L3LU, 0)`
    /// (`L3DlOpsScheduler.cpp:7477-7481`). ⛔ A `clId = -1` SPELLING ANSWERS 64 HERE.
    #[test]
    fn the_seams_comp_view_reads_corelet_zeros_share_and_the_stage_asked_for() {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::Out, Extent(64));
        dims.corelet_split
            .insert(PrimaryDim::Out, vec![Extent(16), Extent(48)]);
        dims.padding
            .insert(PrimaryDim::Out, padding(PrimaryDim::Ki));
        let stated = CoordDims::of(FilledDims::of(dims).expect("a stated half"));

        let mut stages: DataStages<CoordDims> = DataStages::default();
        stages.0.insert(
            DatastageId(1),
            crate::schedule::ddc::transformation_util::DataStage {
                ss: crate::schedule::ddc::transformation_util::StageDims {
                    name: crate::schedule::ddc::transformation_util::StageName("1".to_owned()),
                    dims: stated,
                },
                el: crate::schedule::ddc::transformation_util::StageDims::default(),
            },
        );
        stages.0.insert(DatastageId(2), Default::default());
        let held = seam(stages);

        assert_eq!(
            held.comp_view(DatastageId(1), PrimaryDim::Out),
            Some(Extent(16)),
            "corelet ZERO's `coreletSplit_` share — the whole core's 64 means the corelet was lost"
        );
        assert_eq!(
            held.comp_view(DatastageId(2), PrimaryDim::Out),
            None,
            "an extent-less stage answers the reference's `-1`"
        );
        assert_eq!(
            held.comp_view(DatastageId(7), PrimaryDim::Out),
            None,
            "and a stage id this DSC holds nothing at is `dataStageParam_.at()`'s throw"
        );

        assert_eq!(
            held.stage_padding(DatastageId(1))
                .and_then(|padding| padding.get(&PrimaryDim::Out))
                .and_then(|held| held.window_dim),
            Some(PrimaryDim::Ki),
            "`paddingSizes_` crosses whole — the window dim is what entry 228's PADDED arm reads"
        );
        assert_eq!(
            held.stage_padding(DatastageId(2)).map(BTreeMap::is_empty),
            Some(true),
            "the extent-less stage HAS an (empty) paddingSizes_, so entry 228 does not refuse on it"
        );
        assert_eq!(held.stage_padding(DatastageId(7)), None);
    }

    /// ⭐ THE CHAIN IS INNERMOST FIRST AND THE ROOT LOOP IS EXCLUDED —
    /// `L3DlOpsScheduler.cpp:7598-7614`.
    #[test]
    fn the_lx_below_chain_is_innermost_first_and_excludes_the_root_loop() {
        let mut tree = TreeData::default();
        let root = tree.add(
            NodeName("loop_root".to_owned()),
            Kind::Loop(loop_node(0, 1, PrimaryDim::Mb)),
            None,
        );
        // `getLxBelowBlockNode` walks the tree FROM ITS HEAD, so a tree whose head is unset holds no
        // node at all as far as the walk is concerned.
        tree.set_head(root);
        let middle = tree.add(
            NodeName("loop_middle".to_owned()),
            Kind::Loop(loop_node(1, 2, PrimaryDim::Out)),
            Some(root),
        );
        let inner = tree.add(
            NodeName("loop_inner".to_owned()),
            Kind::Loop(loop_node(2, 3, PrimaryDim::Y)),
            Some(middle),
        );
        tree.add(
            NodeName(crate::schedule::l3::dl_ops::LX_BELOW_BLOCK_NODE_NAME.to_owned()),
            Kind::Block,
            Some(inner),
        );

        let chain = chunk_loop_chain(&tree).expect("this tree holds an lx_below block");
        assert_eq!(
            chain
                .iter()
                .map(|held| held.name.0.clone())
                .collect::<Vec<_>>(),
            vec!["loop_ds2_ds3_y".to_owned(), "loop_ds1_ds2_out".to_owned()],
            "inner first, and `loop_ds0_ds1_mb` is the root loop the walk breaks on"
        );
        assert_eq!(
            seam(DataStages::default())
                .lx_below_chunk_loops()
                .map(|held| held.len()),
            None,
            "no lx_below block is the DT_CHECK_MSG, which is a refusal and not an empty chain"
        );

        // An lx_below block directly under the root loop: the reference's `while (currLoop)` breaks
        // on the first lap, which is the EMPTY chain and not a refusal.
        let mut shallow = TreeData::default();
        let only = shallow.add(
            NodeName("loop_root".to_owned()),
            Kind::Loop(loop_node(0, 1, PrimaryDim::Mb)),
            None,
        );
        shallow.set_head(only);
        shallow.add(
            NodeName(crate::schedule::l3::dl_ops::LX_BELOW_BLOCK_NODE_NAME.to_owned()),
            Kind::Block,
            Some(only),
        );
        assert_eq!(chunk_loop_chain(&shallow), Some(Vec::new()));
    }

    /// ⭐ EVERY STAGE THE DSC HOLDS IS SEEDED, the extent-less ones included.
    #[test]
    fn the_seed_carries_every_stage_the_dsc_holds() {
        let mut held = L3DataStages::new(
            stage("core", &[(PrimaryDim::Out, 64)]),
            stage("chunk", &[(PrimaryDim::Out, 32)]),
        );
        held.set(DatastageId(2), stage("2", &[(PrimaryDim::Out, 16)]));
        let ibr = held.next_index();
        held.mint_ibr(ibr);

        let stages = coord_stages(&held);
        assert_eq!(
            stages.0.keys().copied().collect::<Vec<_>>(),
            vec![DatastageId(0), DatastageId(1), DatastageId(2), ibr],
            "the core, the chunk, the minted stage and the extent-less IBR entry"
        );
        assert_eq!(
            stages
                .0
                .get(&DatastageId(2))
                .and_then(|stage| stage.ss.dims.stated())
                .and_then(|dims| dims.extent(PrimaryDim::Out)),
            Some(Extent(16)),
            "the minted stage's own extent, not the core's"
        );
        assert_eq!(
            stages
                .0
                .get(&ibr)
                .map(|stage| stage.ss.dims.stated().is_none()),
            Some(true),
            "and the IBR entry states no dim, which is what `ddl.datastage` leaves"
        );
    }

    /// ⭐ THE LAYOUT ORDER IS THE WALK'S STATED ANSWER, read and not re-derived.
    #[test]
    fn the_layout_order_is_the_stated_answer_per_labelled_ds() {
        let held = Layout::of(BTreeMap::from([
            (
                LdsIdx(0),
                LayoutDims::new(PrimaryDim::Mb, vec![PrimaryDim::Out, PrimaryDim::Y]),
            ),
            (LdsIdx(1), LayoutDims::new(PrimaryDim::Out, Vec::new())),
        ]));
        assert_eq!(
            held.layout_dims(LdsIdx(0)).to_vec(),
            vec![PrimaryDim::Mb, PrimaryDim::Out, PrimaryDim::Y]
        );
        assert_eq!(held.layout_dims(LdsIdx(1)).first(), PrimaryDim::Out);
    }

    /// ⭐ NO LOOP OF THIS ARENA IS PARAMETRIC, so entry 228 takes its datastage arm — which is
    /// [`AllocCoordinateSeam::comp_view`] — for every loop it is handed.
    #[test]
    fn no_loop_of_this_arena_is_parametric() {
        assert_eq!(
            seam(DataStages::default()).parametric_iter_count(&loop_node(0, 1, PrimaryDim::Out)),
            None,
            "`isParametricLoop()` is `parametricLds_ != -1` and `LoopNode` carries no such field"
        );
    }
}
