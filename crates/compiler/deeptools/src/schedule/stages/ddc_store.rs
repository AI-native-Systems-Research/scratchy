// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE WHOLE OF ONE `currDsc` — [`v1::Dsc2Store`], the `E` carrier entry 379 drives, with its
//! thirty-five supertraits.
//!
//! ⛔⛔ THIS IS WHERE [`v1::Dsc2Store::split`] IS SATISFIED, AND IT IS SATISFIED THE WAY ITS OWN DOC
//! SAYS (`ddc/v1.rs:6040-6047`): [`Dsc2Store`] holds `reads` and `tree` as FIELDS, so
//! `(&self.reads, &mut self.tree)` is one shared and one exclusive borrow of two disjoint fields —
//! and BOTH fields hold `&'s Dsc2State`, so the two views name the SAME tree, the SAME `labeledDs_`
//! and the SAME `computeOp_`. Two carriers over one `currDsc` is the defect that doc warns about, and
//! this is not it: entry 308's writes go through the cell every entry 307 read comes out of.
//!
//! # ⭐ WHAT THIS CARRIER ANSWERS
//!
//! Everything the schedule tree is — the DFS walks, the surgery
//! ([`crate::schedule::ddc::transformation_util::ScheduleSurgery`] is answered whole),
//! the loop bands, the condition regions, `computeOp_` and every DSC fact
//! [`super::Dsc2Reads`] already answers, DELEGATED to it so the two halves cannot disagree.
//!
//! # ⛔ WHAT IT DOES NOT, AND THE THREE REASONS
//!
//! 1. **THIS TREE HOLDS NO COMPUTE NODES — AND THAT IS A MISSING WRITER, NOT A MISSING ARM.**
//!
//!    ⛔⛔ A RECORDED CORRECTION. This note used to say *"[`super::tree::Kind`] has `Block`, `Loop`,
//!    `Transfer`, `Allocate`, `Sync` and `Condition` — and no `Compute`"*, and every stub below cited
//!    it as *"no Compute arm"*. **[`super::tree::Kind::Compute`] EXISTS** and holds the whole
//!    [`ComputeNode`] (`stages/tree.rs:96`, beside a `StickMask` arm the same sentence omitted), and
//!    [`super::tree::TreeData::kind_of`] is `pub`, so a compute node is READABLE from here.
//!
//!    What is true is that NOTHING CONSTRUCTS ONE: `Kind::Compute(..)` appears at no callsite in the
//!    crate, because stage 2a mints none (the 24,363-program census reads `compute: 0`) and entry
//!    345's DDL parse is what mints them. So the walks below find none — a READING, not an absence.
//!
//!    Each remaining compute stub therefore names its OWN blocker rather than the arm:
//!    [`tr::ComputeWalk::compute_op`] wants a `ComputeOpType` our [`ComputeNode`] does not project,
//!    [`v1::ComputeMasks::computes_under_mask`] wants an [`crate::schedule::dsc2::InstrAttribute`]
//!    field, and the two whole-node getters want one `TreeData` accessor. Three that cited the arm
//!    and never needed it — `compute_name`, `set_compute_name` and `parent_dim_loop` — are answered,
//!    because all three read `ScheduleNode`'s BASE fields and not `ComputeNode`'s at all.
//! 2. **`l3::dsc` DROPS SEVEN `DesignSpaceConfig` FIELDS** — `dsName_`, `dataFormat_`,
//!    `wordLength`, `constantInfo_`, `scaledLdsCategory_`'s non-scale arms, `dimToSymbolMapping_`
//!    and `l0TetheredMode_`. A conversion gap, not a data gap; see [`super::Dsc2Reads`].
//! 3. **`dsc/` AND `util/foldManager/` SEAMS** — `getBufferCapacityForNode`,
//!    `getBlockTransferSize*`, `getPadding`, `buildAndPropagateFold`, `setRelevantCompCoreCl` and
//!    `finalizeScheduleTree` are all outside this campaign's file list and are REACHED, not
//!    reimplemented, by the reference itself.

use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{PrimaryDim, StickDims};
use crate::formats::DataFormat;
use crate::schedule::ddc::fold::{
    AllocId, ConstIdx, DataOrigin, NodeId, NodeKind, ScaledLds, StoredStream,
};
use crate::schedule::ddc::metadata::{DatastageId, DdcMemory};
use crate::schedule::ddc::transformation as tr;
use crate::schedule::ddc::transformation_util as tu;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{
    AllocateNode, ComputeNode, LayoutDims, LdsIdx, LoopCondComposite, NodeName, TransferNode,
    WordLength,
};
use crate::schedule::l3::dsc::DscIdx;
use crate::units::Corelet;

use super::ddc_reads::Dsc2Reads;
use super::ddc_state::{self, Dsc2State};
use super::ddc_tree::Dsc2Tree;
use super::tree::{Kind, TreeData};

/// ⭐ ONE `currDsc` — the two halves [`v1::Dsc2Store::split`] hands out, side by side, over one state.
#[derive(Debug)]
pub struct Dsc2Store<'s, 'l> {
    /// The shared half.
    reads: Dsc2Reads<'s, 'l>,
    /// The exclusive half — `currDsc->scheduleTree_`.
    tree: Dsc2Tree<'s, 'l>,
    state: &'s Dsc2State<'l>,
    dsc: DscIdx,
}

impl<'s, 'l> Dsc2Store<'s, 'l> {
    /// One DSC's store, along the fold manager's own address coordinates.
    #[must_use]
    pub fn new(
        state: &'s Dsc2State<'l>,
        dsc: DscIdx,
        coords: crate::schedule::l3::dl_ops::AddressFoldCoords,
    ) -> Self {
        Self {
            reads: Dsc2Reads::new(state, dsc, coords),
            tree: Dsc2Tree::new(state, dsc),
            state,
            dsc,
        }
    }

    /// That DSC's facts — ⛔ TOTAL BY CONSTRUCTION.
    fn facts(&self) -> &'s super::ddc_state::Dsc2Facts {
        self.state
            .facts(self.dsc)
            .expect("a Dsc2Store is only built for a DSC the state holds facts for")
    }

    /// The tree, for one read.
    pub(super) fn with_tree<T>(&self, ask: impl FnOnce(&TreeData) -> T) -> T {
        self.state
            .tree(self.dsc)
            .expect("a Dsc2Store is only built for a DSC the state holds a tree for")
            .with(ask)
    }

    /// The tree, for one write.
    fn with_tree_mut<T>(&self, write: impl FnOnce(&mut TreeData) -> T) -> T {
        self.state
            .tree(self.dsc)
            .expect("a Dsc2Store is only built for a DSC the state holds a tree for")
            .with_mut(write)
    }

    // ⭐ THE HELPERS [`super::ddc_store2`]'S IMPLS READ THROUGH — one per FACT, so the two files
    // cannot reach the state two different ways.

    /// That DSC's facts.
    pub(super) fn dsc_facts(&self) -> &'s super::ddc_state::Dsc2Facts {
        self.facts()
    }

    /// ⭐⭐ `split`'s OWN BODY — two disjoint fields, both naming one state.
    pub(super) fn halves(&mut self) -> (&Dsc2Reads<'s, 'l>, &mut Dsc2Tree<'s, 'l>) {
        (&self.reads, &mut self.tree)
    }

    /// `0 .. numCoreletsUsed_DSC2_` as corelets.
    pub(super) fn dsc2_corelets(&self) -> Vec<Corelet> {
        v1::Placement::corelets_used(&self.reads)
    }

    /// `coreIdsUsed_`.
    pub(super) fn cores_used_of(&self) -> v1::CoresUsed {
        v1::Placement::cores_used(&self.reads)
    }

    /// `node->nodeType_`.
    pub(super) fn node_kind_of(&self, node: NodeId) -> Option<NodeKind> {
        self.with_tree(|tree| tree.node_kind(node))
    }

    /// ⭐ ONE READ-MODIFY-WRITE OF A TRANSFER BODY — a NO-OP on a node that is not a transfer, which
    /// is the reference's own downcast of one.
    pub(super) fn edit_transfer(&self, node: NodeId, edit: impl FnOnce(&mut TransferNode)) {
        self.with_tree_mut(|tree| {
            if let Some(mut held) = tree.transfer(node) {
                edit(&mut held);
                tree.set_transfer(node, held);
            }
        });
    }

    /// ⭐ ONE DESTINATION OF A TRANSFER, READ-MODIFY-WRITTEN — [`crate::schedule::dsc2::Dsts`] keeps
    /// its NON-EMPTINESS by holding `first` and `rest` privately, so a per-destination write goes
    /// through the constructor and the routes are carried over unchanged.
    pub(super) fn edit_dst(
        &self,
        node: NodeId,
        dst: usize,
        edit: impl FnOnce(&mut crate::schedule::dsc2::Operand),
    ) {
        self.edit_transfer(node, |held| {
            let hops: Vec<crate::schedule::dsc2::Hops> = (0..held.dsts.iter().count())
                .map(|at| crate::schedule::dsc2::Hops(held.dsts.hops(at).to_vec()))
                .collect();
            let mut operands: Vec<crate::schedule::dsc2::Operand> =
                held.dsts.iter().cloned().collect();
            let Some(target) = operands.get_mut(dst) else {
                return;
            };
            edit(target);
            let (first, rest) = operands.split_first().expect("Dsts is non-empty by type");
            held.dsts = crate::schedule::dsc2::Dsts::new(first.clone(), rest.to_vec())
                .with_hops(hops);
        });
    }

    /// `allocNode->addAllocUser(user)` — filed on the labelled DS's `memOrg_`, where the users list
    /// lives; a NO-OP for an allocation no `memOrg_` names, which is the reference's null node.
    pub(super) fn add_user_to(&self, alloc: AllocId, user: NodeId) {
        let Some(tree) = self.state.tree(self.dsc) else {
            return;
        };
        let Some(node) = tree.with(|held| held.node_of_alloc(alloc)) else {
            return;
        };
        if let Some((lds, storage)) = tree.home_of(node)
            && let Some(org) = tree.org(lds)
        {
            org.add_user(storage, user);
        }
    }

    /// `condNode->coreClCond_`.
    pub(super) fn core_cl_cond_of(&self, condition: NodeId) -> Option<v1::CoreClSet> {
        self.with_tree(|tree| tree.core_cl_cond(condition))
    }

    /// `condNode->getThenBranchNode()`.
    pub(super) fn then_region_of(&self, condition: NodeId) -> Option<NodeId> {
        self.with_tree(|tree| tree.then_branch(condition))
    }

    /// `new dsc2::ConditionNode()` with its `name_` and `coreClCond_`, and no loop condition.
    pub(super) fn new_core_cl_condition(&self, name: NodeName, core_cl: v1::CoreClSet) -> NodeId {
        self.with_tree_mut(|tree| {
            tree.add(
                name,
                Kind::Condition(super::tree::Cond {
                    loop_cond: None,
                    cores: Some(core_cl),
                    then_region: Vec::new(),
                    else_region: Vec::new(),
                }),
                None,
            )
        })
    }

    /// `condNode->addThenRegion(block)` / `addElseRegion(block)`.
    pub(super) fn add_region_to(&self, condition: NodeId, block: NodeId, then_region: bool) {
        self.with_tree_mut(|tree| tree.add_region(condition, block, then_region));
    }

    /// `dataStageParam_.at(core).ss_.paddingSizes_.at(dim)` — ⭐ THE SHARED MAP.
    pub(super) fn core_stage_padding(
        &self,
        dim: PrimaryDim,
    ) -> Option<crate::schedule::l3::dsc::DimPadding> {
        self.facts().with_stages(|stages| {
            stages
                .0
                .get(&crate::schedule::ddc::metadata::Metadata::CORE_DSTGID)?
                .ss
                .dims
                .dims
                .padding
                .get(&dim)
                .copied()
        })
    }

    /// `dataStageParam_.at(core).ss_.peSfpSplit_.empty()` negated — ⭐ THE SHARED MAP.
    pub(super) fn core_stage_has_pe_sfp_split(&self) -> bool {
        self.facts().with_stages(|stages| {
            stages
                .0
                .get(&crate::schedule::ddc::metadata::Metadata::CORE_DSTGID)
                .is_some_and(|held| !held.ss.dims.dims.pe_sfp_split.is_empty())
        })
    }

    /// `traverseTreeDFSMutable(base, {SYNC}, .., excludeList)` reduced to each SYNC's `units_`.
    pub(super) fn sync_units_below(
        &self,
        base: tr::LoopId,
        exclude: Option<tr::LoopId>,
    ) -> Vec<crate::schedule::dsc2::SyncUnits> {
        self.with_tree(|tree| {
            let mut found = Vec::new();
            let mut stack = vec![base.0];
            while let Some(at) = stack.pop() {
                if exclude.is_some_and(|skip| skip.0 == at) {
                    continue;
                }
                if let Some(units) = tree.sync_units(at) {
                    found.push(units);
                }
                stack.extend(tree.children(at).into_iter().rev());
            }
            found
        })
    }

    /// The identity a freshly minted `dsc2::AllocateNode` would take.
    pub(super) fn next_free_alloc(&self) -> AllocId {
        self.with_tree(super::tree::TreeData::peek_alloc)
    }

    /// `labeledDs_.at(lds).memOrg_`'s FIRST entry with an `allocateNode_`, in `std::map` order.
    pub(super) fn first_alloc_of(&self, lds: LdsIdx) -> Option<AllocId> {
        let tree = self.state.tree(self.dsc)?;
        let org = tree.org(lds)?;
        tree.with(|held| {
            [SenComponent::Hbm, SenComponent::Lx]
                .into_iter()
                .filter_map(|storage| org.node(storage))
                .find_map(|node| held.allocate(node).map(|(alloc, _)| alloc))
        })
    }

    /// `alloc->layoutDimOrder_.at(0)` — the OUTERMOST layout dim.
    pub(super) fn alloc_layout_first(&self, alloc: AllocId) -> Option<PrimaryDim> {
        self.with_tree(|tree| tree.node_of_alloc(alloc).and_then(|node| tree.allocate(node)))
            .and_then(|(_, held)| held.layout.0.first().map(|(dim, _)| *dim))
    }

    /// `allocNode->getPrev()` — whether that allocate node already has a parent.
    pub(super) fn alloc_has_parent(&self, alloc: AllocId) -> bool {
        self.with_tree(|tree| {
            tree.node_of_alloc(alloc)
                .is_some_and(|node| tree.parent(node).is_some())
        })
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE FOUR TRAITS BOTH HALVES CARRY — DELEGATED, so the shared and the exclusive view of one
// `currDsc` cannot give two answers for one field.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl crate::schedule::dsc2::Dsc for Dsc2Store<'_, '_> {
    fn layout_dims(&self, lds: LdsIdx) -> LayoutDims {
        crate::schedule::dsc2::Dsc::layout_dims(&self.reads, lds)
    }
}

impl v1::LabeledDsIndices for Dsc2Store<'_, '_> {
    fn labeled_ds_indices(&self) -> Vec<LdsIdx> {
        v1::LabeledDsIndices::labeled_ds_indices(&self.reads)
    }
}

impl v1::LdsSticks for Dsc2Store<'_, '_> {
    fn stick_dims(&self, lds: LdsIdx) -> StickDims {
        v1::LdsSticks::stick_dims(&self.reads, lds)
    }
}

impl v1::Masking for Dsc2Store<'_, '_> {
    fn coordinate_masking(&self) -> BTreeMap<PrimaryDim, Vec<v1::MaskRun>> {
        v1::Masking::coordinate_masking(&self.reads)
    }
    fn declares_samv_wsllen(&self) -> bool {
        v1::Masking::declares_samv_wsllen(&self.reads)
    }
    fn masking_constant(&self) -> Option<ConstIdx> {
        v1::Masking::masking_constant(&self.reads)
    }
    fn splits_across_cores(&self, dim: PrimaryDim) -> bool {
        v1::Masking::splits_across_cores(&self.reads, dim)
    }
    fn is_symbolic(&self, dim: PrimaryDim) -> bool {
        v1::Masking::is_symbolic(&self.reads, dim)
    }
    fn non_broadcast_dims(&self, lds: LdsIdx) -> BTreeSet<PrimaryDim> {
        v1::Masking::non_broadcast_dims(&self.reads, lds)
    }
    fn lds_format(&self, lds: LdsIdx) -> Option<DataFormat> {
        v1::Masking::lds_format(&self.reads, lds)
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE DSC'S OWN SHAPE AND OP LIST — answered from the state.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl v1::CoreletShapes for Dsc2Store<'_, '_> {
    /// `numCoreletsUsed_`.
    fn corelets_used(&self) -> u32 {
        self.facts().with_dsc(|dsc| dsc.corelets_used.get())
    }

    /// ⛔ `CoreD_.primaryDimToVal_st(dim)` — `CoreD_` is a `DataStructDims` member of
    /// `DesignSpaceConfig` beside `dataStageParam_` (`dsc/designSpaceConfig.h:102`), and
    /// [`crate::schedule::l3::dsc::DesignSpaceConfig`] projects only the CORE DATA STAGE, which is a
    /// different object: `corelet_shares` is derived from `dataStageParam_.at(0)` *where the core
    /// data stage exists, else `CoreletD_` against `CoreD_`* — a RATIO, not either extent.
    fn core_extent(&self, _dim: PrimaryDim) -> crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent {
        todo!(
            "v1::CoreletShapes::core_extent: wants CoreD_.primaryDimToVal_st(dim) \
             (dsc/designSpaceConfig.h:102) — l3::dsc::DesignSpaceConfig projects only the RATIO \
             CoreletD_/CoreD_ as corelet_shares, never either extent"
        )
    }

    /// ⛔ `CoreletD_.primaryDimToVal_st(dim)` — the same gap.
    fn corelet_extent(&self, _dim: PrimaryDim) -> crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent {
        todo!(
            "v1::CoreletShapes::corelet_extent: wants CoreletD_.primaryDimToVal_st(dim) \
             (dsc/designSpaceConfig.h:102) — see core_extent"
        )
    }
}

impl v1::DataStages for Dsc2Store<'_, '_> {
    /// `dataStageParam_.at(stage).ss_.coreletSplit_`'s keys — ⭐ THE SHARED MAP; EMPTY where the
    /// reference's `.at()` finds no such data stage, which is its own arm.
    fn corelet_split_dims(&self, stage: DatastageId) -> BTreeSet<PrimaryDim> {
        self.facts().with_stages(|stages| {
            stages
                .0
                .get(&stage)
                .map(|held| held.ss.dims.dims.corelet_split.keys().copied().collect())
                .unwrap_or_default()
        })
    }
}

impl v1::ComputeOps for Dsc2Store<'_, '_> {
    /// `computeOp_`, one `opFuncName` per entry.
    fn op_funcs(&self) -> v1::OpFuncs {
        let ops = self.facts().ops();
        let mut funcs = ops.iter().map(|op| op.op_func);
        let first = funcs.next().flatten();
        v1::OpFuncs::new(first, funcs.collect())
    }

    /// `computeOp_.at(0).opFuncName = op_func`.
    fn set_first_op_func(&mut self, op_func: OpFunc) {
        self.facts().with_ops_mut(|ops| {
            if let Some(first) = ops.first_mut() {
                first.op_func = Some(op_func);
            }
        });
    }
}

impl tr::ComputeWalk for Dsc2Store<'_, '_> {
    /// Every `COMPUTE` node — ⭐ EMPTY, AND THAT IS A READING AND NOT AN ABSENCE: the tree holds
    /// none, so the walk finds none, exactly as the reference's walk over a tree with no computes
    /// does.
    fn computes(&self) -> Vec<NodeId> {
        self.with_tree(|tree| {
            tree.dfs()
                .into_iter()
                .filter(|node| tree.node_kind(*node) == Some(NodeKind::Compute))
                .collect()
        })
    }

    /// ⛔ `type_` ON A COMPUTE NODE — AND THE ARM IS NOT WHAT BLOCKS IT. `Kind::Compute` holds the
    /// whole [`ComputeNode`], but the C++ field is `ComputeOpType type_ = ComputeOpType::COUNT`
    /// (`dsc/dsc2.h:933`) and [`ComputeNode::op`] projects a
    /// [`crate::schedule::ddl::ops::DdlComputeType`] — the DDL's `computetype=` set, whose 34 arms
    /// are `Macc`/`Fma16`/…/`And` and include NEITHER value [`tr::ComputeOp`] classifies. Answering
    /// `Other` for a `RECIPROCAL` or a `LAYERNORMSCALE` is the wrong branch of entry 105, not a
    /// default.
    fn compute_op(&self, _node: NodeId) -> tr::ComputeOp {
        todo!(
            "tr::ComputeWalk::compute_op: wants ComputeNode::type_, a ComputeOpType \
             (dsc/dsc2.h:933) — dsc2::ComputeNode::op projects DdlComputeType, which cannot spell \
             RECIPROCAL or LAYERNORMSCALE, so tr::ComputeOp's two named arms are unreachable"
        )
    }
}

impl tr::ComputeNodes for Dsc2Store<'_, '_> {
    /// ⛔ THE WHOLE `dsc2::ComputeNode` — READABLE, awaiting ONE accessor.
    /// [`super::tree::Kind::Compute`] holds it and [`super::tree::TreeData::kind_of`] is `pub`, so
    /// this is `tree.kind_of(node)` matched on that arm. What is missing is the pairing
    /// [`super::tree::TreeData::transfer`] has: a by-kind reader plus this trait's TOTAL return,
    /// which for a non-COMPUTE node is the reference's own `static_cast` and therefore a stop. Left
    /// as one stub rather than a second `panic!` for a node no writer mints.
    fn compute(&self, _compute: NodeId) -> ComputeNode {
        todo!(
            "tr::ComputeNodes::compute: wants the whole dsc2::ComputeNode — Kind::Compute HOLDS it \
             (stages/tree.rs:96); needs a TreeData::compute reader beside TreeData::transfer and a \
             stop for the non-COMPUTE node the reference static_casts"
        )
    }
}

impl v1::ComputeMasks for Dsc2Store<'_, '_> {
    /// ⛔ `instrAttribute_.computeMaskLoopOffsets_` — A FIELD, and the arm is not what blocks it.
    /// The C++ `InstrAttribute` carries `std::map<int, unordered_map<const LoopNode*,
    /// unordered_map<PrimaryDimTypes, int>>> computeMaskLoopOffsets_` (`dsc/dsc2.h:925`), and
    /// [`crate::schedule::dsc2::InstrAttribute`] projects `compute_mask_` (`:916`) — a DIFFERENT
    /// field — and not this one. Its absence is what both this and
    /// [`tr::ComputeMasking::set_compute_mask_loop_offset`] want.
    fn computes_under_mask(&self, _node: NodeId) -> bool {
        todo!(
            "v1::ComputeMasks::computes_under_mask: wants \
             instrAttribute_.computeMaskLoopOffsets_ (dsc/dsc2.h:925) — dsc2::InstrAttribute \
             projects compute_mask_ (:916) and not this map; Kind::Compute itself is present"
        )
    }
}

impl tr::ComputeMasking for Dsc2Store<'_, '_> {
    /// `compNode->name_` — ⭐ ANSWERED, AND IT NEVER NEEDED A COMPUTE NODE. `name_` is declared on
    /// the BASE class: `std::string name_;` sits beside `const NodeType nodeType_` in `ScheduleNode`
    /// (`dsc/dsc2.h:459`), which `ComputeNode` inherits through
    /// `InheritWithClone<ScheduleNode, ComputeNode>` (`dsc/dsc2.h:900`). So `compNode->name_` and
    /// `node->name_` are ONE field, and [`tu::ScheduleSurgery::node_name`] already answers it —
    /// delegated rather than re-read so the two cannot give two names for one node.
    fn compute_name(&self, compute: NodeId) -> NodeName {
        tu::ScheduleSurgery::node_name(self, compute)
    }
    /// `compNode->name_ = name` — the same base field, written.
    fn set_compute_name(&mut self, compute: NodeId, name: NodeName) {
        self.with_tree_mut(|tree| tree.set_name(compute, name));
    }
    /// `compNode->getParentDimLoop(dim)` — ⭐ ANSWERED, AND ALSO A BASE-CLASS METHOD:
    /// `getParentDimLoop` is declared on `ScheduleNode` (`dsc/dsc2.h:467`) and its body reads no
    /// `ComputeNode` state at all (`dsc/dsc2.cpp:1906-1914`):
    ///
    /// ```text
    /// const LoopNode* loop = getOwnerLoop();
    /// while (loop != nullptr && !loop->hasLoopDim(dim)) loop = loop->getOwnerLoop();
    /// return loop;
    /// ```
    ///
    /// ⭐ `getOwnerLoop()` STARTS AT `prev_`, NOT AT THE NODE — `auto parent = prev_;` then walks
    /// `prev_` for the nearest `LOOP` (`dsc/dsc2.cpp:1896-1900`) — so the walk below re-asks
    /// [`super::tree::TreeData::owner_loop`] from the loop's OWN id and can never return that loop
    /// itself. `hasLoopDim(dim)` is `dims_` scanned for the dim and nothing more
    /// (`dsc/dsc2.cpp:4223-4228`).
    fn parent_dim_loop(&self, compute: NodeId, dim: PrimaryDim) -> Option<tr::LoopId> {
        self.with_tree(|tree| {
            let mut at = tree.owner_loop(compute);
            while let Some(held) = at {
                if tree
                    .loop_node(held)
                    .is_some_and(|node| node.dims.iter().any(|pair| pair.dim == dim))
                {
                    return Some(held);
                }
                at = tree.owner_loop(held.0);
            }
            None
        })
    }
    /// `0 .. numCoreletsUsed_DSC2_` as corelets — ⭐ ANSWERED, and it is the DSC2 count and not the
    /// total.
    fn corelets(&self) -> Vec<Corelet> {
        v1::Placement::corelets_used(&self.reads)
    }
    fn loop_dims(&self, dim_loop: tr::LoopId) -> Vec<PrimaryDim> {
        self.with_tree(|tree| {
            tree.loop_node(dim_loop)
                .map(|held| held.dims.iter().map(|pair| pair.dim).collect())
                .unwrap_or_default()
        })
    }
    fn set_compute_mask_loop_offset(
        &mut self,
        _compute: NodeId,
        _corelet: Corelet,
        _dim_loop: tr::LoopId,
        _dim: PrimaryDim,
        _offset: tr::MaskLoopOffset,
    ) {
        todo!(
            "tr::ComputeMasking::set_compute_mask_loop_offset: wants \
             instrAttribute_.computeMaskLoopOffsets_[cl][loop][dim] = offset (dsc/dsc2.h:925) — \
             dsc2::InstrAttribute does not project that map, and TreeData has no by-kind writer for \
             a Kind::Compute body; the arm itself is present"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE SCHEDULE TREE AS SURGERY — ⭐ ANSWERED WHOLE, off `super::tree::TreeData`.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl tu::ScheduleSurgery for Dsc2Store<'_, '_> {
    /// `node->name_` — ⛔ TOTAL: the reference dereferences the node.
    fn node_name(&self, node: NodeId) -> NodeName {
        self.with_tree(|tree| {
            tree.name(node).unwrap_or_else(|| {
                panic!("ScheduleSurgery::node_name: {node:?} is not in this DSC's scheduleTree_")
            })
        })
    }

    fn set_node_name(&mut self, node: NodeId, name: NodeName) {
        self.with_tree_mut(|tree| tree.set_name(node, name));
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.with_tree(|tree| tree.parent(node))
    }

    fn owner_loop(&self, node: NodeId) -> Option<tr::LoopId> {
        self.with_tree(|tree| tree.owner_loop(node))
    }

    /// `transferNode->..` — ⛔ TOTAL: the reference downcasts, so a non-TRANSFER node is its own UB.
    fn transfer(&self, node: NodeId) -> TransferNode {
        self.with_tree(|tree| {
            tree.transfer(node).unwrap_or_else(|| {
                panic!(
                    "ScheduleSurgery::transfer: {node:?} is not a TRANSFER of this DSC's \
                     scheduleTree_ — the reference's static_cast there is undefined"
                )
            })
        })
    }

    fn loop_num(&self, loop_node: tr::LoopId) -> DatastageId {
        self.with_tree(|tree| {
            tree.loop_node(loop_node)
                .map(|held| held.num)
                .expect("a LoopId this tree's own walk produced names a loop of this tree")
        })
    }

    fn loop_den(&self, loop_node: tr::LoopId) -> DatastageId {
        self.with_tree(|tree| {
            tree.loop_node(loop_node)
                .map(|held| held.den)
                .expect("a LoopId this tree's own walk produced names a loop of this tree")
        })
    }

    fn loop_dims(&self, loop_node: tr::LoopId) -> tu::LoopDims {
        self.with_tree(|tree| {
            tree.loop_node(loop_node)
                .map(|held| held.dims.clone())
                .expect("a LoopId this tree's own walk produced names a loop of this tree")
        })
    }

    /// ⛔ `isParametricLoop()` — A ONE-`bool` FIELD GAP, AND NOT THE ONE THIS USED TO CITE.
    ///
    /// ⛔⛔ A RECORDED CORRECTION. The message here said it *"reads `LoopNode::parametricLdsIdx_`"*.
    /// It does not: `bool isParametricLoop() const { return isParametricLoop_; }` (`dsc/dsc2.h:599`)
    /// returns `bool isParametricLoop_ = false` (`dsc/dsc2.h:617`), which is a SEPARATE member from
    /// `int parametricLdsIdx_ = -1` (`:618`) — the two are set by different writers
    /// (`markAsParametricLoop()` at `:600` against `setParametricLdsIdx(idx)` at `:604`), so a loop
    /// can be parametric with no parametric lds and vice versa. [`tu::LoopNode`] carries `name`,
    /// `num`, `den` and `dims` and neither of them, so this is ONE `bool` to project, not the index.
    fn is_parametric(&self, _loop_node: tr::LoopId) -> bool {
        todo!(
            "tu::ScheduleSurgery::is_parametric: wants isParametricLoop() (dsc/dsc2.h:599) reading \
             LoopNode::isParametricLoop_ (dsc/dsc2.h:617) — a bool member distinct from \
             parametricLdsIdx_ (:618), and transformation_util::LoopNode carries neither"
        )
    }

    /// `new dsc2::LoopNode(..)` — an UNPARENTED loop.
    fn new_loop(&mut self, loop_node: tu::LoopNode) -> tr::LoopId {
        self.with_tree_mut(|tree| {
            let name = loop_node.name.clone();
            tr::LoopId(tree.add(name, Kind::Loop(loop_node), None))
        })
    }

    /// `new dsc2::BlockNode()` — a fresh block has NO children.
    fn new_block(&mut self, name: NodeName) -> NodeId {
        self.with_tree_mut(|tree| tree.add(name, Kind::Block, None))
    }

    fn add_child_node(&mut self, node: NodeId, at: tu::InsertionPoint) {
        self.with_tree_mut(|tree| tree.link(node, at));
    }

    /// `node->moveNode(currDsc, ..)` — unlinked from its old parent first.
    fn move_node(&mut self, node: NodeId, at: tu::InsertionPoint) {
        self.with_tree_mut(|tree| {
            tree.unlink(node);
            tree.link(node, at);
        });
    }

    fn conditions_under(&self, root: NodeId) -> Vec<NodeId> {
        self.with_tree(|tree| tree.conditions_under(root))
    }

    /// `condNode->loopCond_` — ⛔ TOTAL, and an EMPTY composite is what a core/corelet-guarded
    /// condition carries (`hasCoreClCond()` is `twoLevelOrOfAnds_.empty()`, `dsc/dsc2.h:693`), so the
    /// default is the reference's own value and not a substitute.
    ///
    /// ⚠️ TWO RUST SPELLINGS OF ONE C++ TYPE, AND THIS IS WHERE THEY MEET. `dsc/dsc2.h:675-677` is
    /// projected BOTH as [`crate::schedule::dsc2::LoopCondComposite`] (which
    /// [`super::tree::Cond`] holds and [`super::DscTree::loop_cond`] answers) AND as
    /// [`tu::LoopCondComposite`] (which this trait wants). The two are ISOMORPHIC field for field —
    /// `two_level_or_of_ands`/`or_of_ands`, `negated`, and per term `loop_comp`/`loop_node` over the
    /// same [`NodeId`] with `bound`/`against` over the same `condValType_`+`condValInt_` pair — so the
    /// translation below reads one C++ field through the other spelling and invents nothing. It is
    /// recorded here because a DUPLICATED type is a place two answers can drift apart.
    fn loop_cond(&self, condition: NodeId) -> tu::LoopCondComposite {
        let held = self.with_tree(|tree| tree.loop_cond(condition));
        held.map(as_util_cond).unwrap_or_default()
    }

    fn set_loop_cond(&mut self, condition: NodeId, cond: tu::LoopCondComposite) {
        let held = as_dsc2_cond(&cond);
        self.with_tree_mut(|tree| tree.set_loop_cond(condition, held));
    }
}

/// `dsc2::LoopCondComposite` READ THROUGH `transformation_util`'S SPELLING — see
/// [`tu::ScheduleSurgery::loop_cond`]'s note. ⛔ `Index(u32)` WIDENS to `Int(i64)` losslessly.
fn as_util_cond(held: LoopCondComposite) -> tu::LoopCondComposite {
    use crate::bridges::superdsc_to_dataflow_ir::control_flow::CondValType;
    use crate::schedule::dsc2::LoopBound;
    // ⚠️ AND `condOp_` IS SPELLED TWICE TOO — the same eight arms in both, so the mapping is total
    // both ways and states no arm the other lacks.
    tu::LoopCondComposite {
        or_of_ands: held
            .two_level_or_of_ands
            .into_iter()
            .map(|ands| {
                ands.into_iter()
                    .map(|term| tu::LoopCond {
                        loop_node: term.loop_comp.0,
                        dim: term.dim,
                        op: as_util_op(term.op),
                        against: match term.bound {
                            LoopBound::Index(at) => CondValType::Int(i64::from(at)),
                            LoopBound::First => CondValType::First,
                            LoopBound::Last => CondValType::Last,
                        },
                    })
                    .collect()
            })
            .collect(),
        negated: held.negated,
    }
}

/// `condOp_` (`dsc/dsc2.h:656`) read through `control_flow`'s spelling — the same eight arms.
const fn as_util_op(
    op: crate::schedule::dsc2::CondOp,
) -> crate::bridges::superdsc_to_dataflow_ir::control_flow::CondOp {
    use crate::bridges::superdsc_to_dataflow_ir::control_flow::CondOp as Cf;
    use crate::schedule::dsc2::CondOp as D2;
    match op {
        D2::Eq => Cf::Eq,
        D2::Ne => Cf::Ne,
        D2::Lt => Cf::Lt,
        D2::Le => Cf::Le,
        D2::Gt => Cf::Gt,
        D2::Ge => Cf::Ge,
        D2::Toggle => Cf::Toggle,
        D2::Always => Cf::Always,
        D2::Never => Cf::Never,
        D2::Const => Cf::Const,
        D2::Default => Cf::Default,
    }
}

/// The same the other way.
const fn as_dsc2_op(
    op: crate::bridges::superdsc_to_dataflow_ir::control_flow::CondOp,
) -> crate::schedule::dsc2::CondOp {
    use crate::bridges::superdsc_to_dataflow_ir::control_flow::CondOp as Cf;
    use crate::schedule::dsc2::CondOp as D2;
    match op {
        Cf::Eq => D2::Eq,
        Cf::Ne => D2::Ne,
        Cf::Lt => D2::Lt,
        Cf::Le => D2::Le,
        Cf::Gt => D2::Gt,
        Cf::Ge => D2::Ge,
        Cf::Toggle => D2::Toggle,
        Cf::Always => D2::Always,
        Cf::Never => D2::Never,
        Cf::Const => D2::Const,
        Cf::Default => D2::Default,
    }
}

/// The same the other way — ⛔ [`None`] IS UNSPELLABLE, so a negative `Int` becomes the reference's
/// own `condValInt_` unsigned read; the reference stores that field as an `int` and every writer in
/// the tree writes a loop index, which is non-negative.
fn as_dsc2_cond(held: &tu::LoopCondComposite) -> LoopCondComposite {
    use crate::bridges::superdsc_to_dataflow_ir::control_flow::CondValType;
    use crate::schedule::dsc2::{LoopBound, LoopCond as Dsc2LoopCond};
    LoopCondComposite {
        two_level_or_of_ands: held
            .or_of_ands
            .iter()
            .map(|ands| {
                ands.iter()
                    .map(|term| Dsc2LoopCond {
                        loop_comp: tr::LoopId(term.loop_node),
                        dim: term.dim,
                        op: as_dsc2_op(term.op),
                        bound: match term.against {
                            CondValType::Int(at) => {
                                LoopBound::Index(u32::try_from(at).unwrap_or_default())
                            }
                            CondValType::First => LoopBound::First,
                            CondValType::Last => LoopBound::Last,
                        },
                    })
                    .collect()
            })
            .collect(),
        negated: held.negated,
    }
}

impl tu::LoopBands for Dsc2Store<'_, '_> {
    fn set_loop_den(&mut self, loop_node: tr::LoopId, den: DatastageId) {
        self.with_tree_mut(|tree| tree.set_loop_den(loop_node, den));
    }

    fn set_loop_dims(&mut self, loop_node: tr::LoopId, dims: tu::LoopDims) {
        self.with_tree_mut(|tree| tree.set_loop_dims(loop_node, dims));
    }

    fn move_children(&mut self, from: NodeId, to: NodeId) {
        self.with_tree_mut(|tree| tree.move_children(from, to));
    }

    /// `base->insertPerfectlyNestedBlockNode(nested)` — [`Self::move_children`] and then the one
    /// child.
    fn insert_perfectly_nested(&mut self, base: tr::LoopId, nested: tr::LoopId) {
        self.with_tree_mut(|tree| {
            tree.move_children(base.0, nested.0);
            tree.link(nested.0, tu::InsertionPoint::LastIn(base.0));
        });
    }

    /// ⛔ `condNode->loopCond_.adjustConditionForSplitLoop(orig, new_loops)`
    /// (`dsc/dsc2.cpp:2061`) — a `dsc/` function outside this campaign's file list, and where the
    /// `EQ`/`NE`/`(GT,FIRST)`/`(LT,LAST)` rewrite and its four aborts live.
    fn adjust_condition_for_split_loop(
        &mut self,
        _condition: NodeId,
        _orig: tr::LoopId,
        _new_loops: &[tr::LoopId],
    ) {
        todo!(
            "tu::LoopBands::adjust_condition_for_split_loop: wants \
             loopCond_.adjustConditionForSplitLoop(orig, new_loops) (dsc/dsc2.cpp:2061) — a `dsc/` \
             function outside this campaign's file list"
        )
    }
}

impl tr::TransferWalk for Dsc2Store<'_, '_> {
    fn transfers(&self) -> Vec<NodeId> {
        self.with_tree(|tree| {
            tree.dfs()
                .into_iter()
                .filter(|node| tree.node_kind(*node) == Some(NodeKind::Transfer))
                .collect()
        })
    }

    fn transfer(&self, node: NodeId) -> TransferNode {
        tu::ScheduleSurgery::transfer(self, node)
    }
}

impl tr::ScopeTree for Dsc2Store<'_, '_> {
    /// ⛔ `getInnermostCommonAncestor(a, b, pathToA, pathToB)` — the two PATHS are what
    /// [`tr::Ancestry`] carries, and building them is a `dsc/dsc2.cpp` walk that also classifies
    /// every node on each path.
    fn ancestry(&self, _transfer: NodeId, _consumer: NodeId) -> tr::Ancestry {
        todo!(
            "tr::ScopeTree::ancestry: wants getInnermostCommonAncestor(transfer, consumer, \
             pathToTransfer, pathToConsumer) — a `dsc/dsc2.cpp` walk that also classifies every node \
             on both paths"
        )
    }

    /// ⛔ `getNextView(ALL)` PLUS the node's own classification as a [`tr::ScopeNode`], whose
    /// COMPUTE arm this tree has no node for.
    fn scope_node(&self, _node: NodeId) -> tr::ScopeNode {
        todo!(
            "tr::ScopeTree::scope_node: wants the node classified as a ScopeNode with its \
             getNextView(ALL) children — its Compute arm needs a COMPUTE node, which \
             super::tree::Kind has no arm for"
        )
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.with_tree(|tree| tree.parent(node))
    }

    /// `getNonBroadcastLdsDimSet(myLdsIdx_)` — EMPTY for an absent index, which is the reference's
    /// own `ldsIdx < 0` arm.
    fn non_broadcast_lds_dims(&self, lds: Option<LdsIdx>) -> BTreeSet<PrimaryDim> {
        lds.map(|lds| {
            self.facts()
                .with_dsc(|dsc| ddc_state::non_broadcast_dims_of(dsc, lds))
        })
        .unwrap_or_default()
    }
}

impl v1::ConditionSimplification for Dsc2Store<'_, '_> {
    /// `scheduleTree_.getHead()` — ⛔ TOTAL, and the reference's head IS a `LoopNode`
    /// (`dsc/dsc2.h:637`) even though every tree scratchy seeds has a `BLOCK` there.
    fn head(&self) -> tr::LoopId {
        self.with_tree(|tree| {
            tr::LoopId(
                tree.head()
                    .expect("every seeded scheduleTree_ has a head block"),
            )
        })
    }

    /// ⛔ `relevantComps_`'s keys (`dsc/dsc2.h:516`) — the map
    /// `dsc.setRelevantCompCoreCl()` (`dsc/dsc2.cpp:2647`) fills, which this tree does not carry;
    /// the SAME gap [`v1::ScheduleNodes::is_relevant`] names.
    fn relevant_comps(&self, _node: tr::LoopId) -> Vec<SenComponent> {
        todo!(
            "v1::ConditionSimplification::relevant_comps: wants relevantComps_'s keys \
             (dsc/dsc2.h:516), which dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647) fills"
        )
    }

    fn conditions(&self, head: tr::LoopId) -> Vec<NodeId> {
        self.with_tree(|tree| tree.conditions_under(head.0))
    }

    /// ⛔ `node->getRelevantCoreCl()` — the same seam.
    fn relevant_core_cl(&self, _node: NodeId) -> v1::CoreClSet {
        todo!(
            "v1::ConditionSimplification::relevant_core_cl: wants getRelevantCoreCl() \
             (dsc/dsc2.h:471), which dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647) fills"
        )
    }

    /// `cn->hasCoreClCond()` — TRUE when `loopCond_.twoLevelOrOfAnds_` is EMPTY
    /// (`dsc/dsc2.h:693`), which is exactly what [`super::tree::Cond`] distinguishes.
    fn has_core_cl_cond(&self, condition: NodeId) -> bool {
        self.with_tree(|tree| tree.core_cl_cond(condition).is_some())
    }

    /// `cn->next_` — the "then" and "else" regions, at most two `BLOCK`s.
    fn branches(&self, condition: NodeId) -> Vec<NodeId> {
        self.with_tree(|tree| tree.children(condition))
    }

    /// `node->moveNode(currDsc, sibling->getMutableParent(), false, sibling)`.
    fn move_after(&mut self, node: NodeId, sibling: NodeId) {
        self.with_tree_mut(|tree| {
            tree.unlink(node);
            tree.link(node, tu::InsertionPoint::After(sibling));
        });
    }

    /// `node->prev_->deleteChildNode(currDsc, node)` — the node AND its subtree.
    fn delete_node(&mut self, node: NodeId) {
        self.with_tree_mut(|tree| tree.delete(node));
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ALLOCATIONS — the `memOrg_` half is answered off the tree's own `Org`s; every MINT names the
// `dsc2::AllocateNode` state this tree does not model.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl tu::DscAllocations for Dsc2Store<'_, '_> {
    /// `labeledDs_.at(lds).ldsIdx_` — ⛔ THE ENTRY'S OWN self-index, and NOT `referenceLdsIdx_`.
    fn own_lds_idx(&self, lds: LdsIdx) -> LdsIdx {
        self.facts()
            .with_lds(lds, crate::schedule::l3::dsc::LabeledDs::recorded)
            .unwrap_or(lds)
    }

    /// `getAllocation(di, storage, allowMissingAlloc=true)` (`dsc/dsc2.cpp:2586-2625`) — ⭐ ANSWERED
    /// ON BOTH ARMS, the labelled-DS one off the tree's own `memOrg_` and the CONSTANT one off
    /// [`crate::schedule::l3::dsc::ConstantInfo::allocations`].
    ///
    /// ⛔⛔ `allowMissingAlloc = true` MAKES EVERY ONE OF ITS FOUR ERROR PATHS `return nullptr`, so
    /// [`None`] here IS the reference's answer and not a refusal: neither index set (`:2589-2596`), a
    /// `storage` that is not a memory (`:2597-2603` — unspellable, [`DdcMemory`] is a memory by type),
    /// a labelled DS with no `memOrg_` entry or a null node in it (`:2605-2615`), and a constant with
    /// no `allocations_` entry or a null one in it (`:2617-2625`).
    fn allocation_in(&self, origin: DataOrigin, storage: DdcMemory) -> Option<AllocId> {
        let component = tu::memory_component(storage);
        match origin {
            // `labeledDs_.at(myLdsIdx_).memOrg_.at(storage).allocateNode_`.
            DataOrigin::LabeledDs(lds) => {
                tu::ComponentAllocations::mem_org_allocation(self, lds, component)
            }
            // `constantInfo_.at(constantId_).allocations_.at(storage)`.
            DataOrigin::Constant(constant) => self.facts().with_dsc(|dsc| {
                dsc.ddc
                    .constants
                    .get(&constant)?
                    .allocations
                    .get(&component)
                    .copied()
            }),
        }
    }

    /// `labeledDs_.at(lds).memOrg_[storage] = { isPresent, allocateNode_ }` — ⛔ AN ORDERING GAP, AND
    /// LOOKING THE NODE UP IS THE WRONG FIX.
    ///
    /// ⛔⛔ THE NODE DOES NOT EXIST YET AT THIS CALL, which is why an impl reading
    /// `node_of_alloc(alloc)` would answer `None` and SILENTLY DROP the registration. Its one caller
    /// is [`tu::construct_allocation`], which calls this and then RETURNS the allocate-node body for
    /// its own caller to splice: at `ddc/transformation.rs:3316-3325` the id comes from
    /// `free_alloc()` — [`Self::next_free_alloc`], *"the identity a freshly minted node WOULD
    /// take"* — and the node is only minted by the `insert_allocate` / `held.insert` that follows
    /// (`ddc/transformation_util.rs:3308-3315`). In the reference the id IS the node (`new
    /// dsc2::AllocateNode()`'s pointer), so `memOrg_[storage].allocateNode_ = alloc` is total there
    /// and unspellable here.
    ///
    /// ⭐ THE REPAIR IS IN [`super::tree`], NOT HERE: either [`super::tree::Org`] files the pending
    /// [`AllocId`] per storage (splitting `isPresent` off the node identity, which
    /// [`tu::ComponentAllocations::copy_mem_org_without_allocation`] needs anyway), or
    /// [`super::tree::TreeData::fresh_alloc`] registers the id before the node. Answering it from
    /// here without one of those inserts an allocation nothing filled.
    fn set_allocation_in(&mut self, _lds: LdsIdx, _storage: DdcMemory, _alloc: AllocId) {
        todo!(
            "tu::DscAllocations::set_allocation_in: wants memOrg_[storage] = {{isPresent=true, \
             allocateNode_=alloc}} — the node the AllocId names is minted AFTER this call \
             (ddc/transformation_util.rs:661 then :3315), so super::tree::Org must file the pending \
             AllocId; a node lookup here answers None and drops the registration"
        )
    }

    /// The nodes in `allocUsers_` — ⭐ ANSWERED off the labelled DS's own `memOrg_`.
    fn alloc_users(&self, alloc: AllocId) -> Vec<NodeId> {
        let Some(tree) = self.state.tree(self.dsc) else {
            return Vec::new();
        };
        let Some(node) = tree.with(|held| held.node_of_alloc(alloc)) else {
            return Vec::new();
        };
        let Some((lds, storage)) = tree.home_of(node) else {
            return Vec::new();
        };
        match storage {
            SenComponent::Hbm => tree
                .org(lds)
                .and_then(crate::schedule::l3::dsc::MemOrg::hbm_alloc_users),
            _ => tree
                .org(lds)
                .and_then(crate::schedule::l3::dsc::MemOrg::lx_alloc_users),
        }
        .unwrap_or_default()
    }

    /// `allocNode->component_` — ⭐ ANSWERED off the L3 view the tree holds, through the one closed
    /// mapping `SenComponent` has onto [`DdcMemory`].
    ///
    /// ⛔⛔ A RECORDED DIVERGENCE FROM THIS PORT, RESOLVED IN THE AUTHORITY'S FAVOUR. This method used
    /// to `todo!` on `L0` saying *"`component_` alone cannot say which [of L0 / L0_SCALE] — it needs
    /// the allocation's `scaledLdsCategory_`"*. IT DOES SAY WHICH, and no `scaledLdsCategory_` is
    /// consulted anywhere:
    ///
    ///   * `L0` and `L0_SCALE` ARE TWO DISTINCT `SenComponents` VALUES — the reference tests them side
    ///     by side as such, `comp == SenComponents::LX || comp == SenComponents::L0 || comp ==
    ///     SenComponents::L0_SCALE` (`ddc/ddcv1.cpp:191-192`) — and this crate spells both:
    ///     [`SenComponent::L0`] and [`SenComponent::L0Scale`]
    ///     (`sys-arch-spec/src/arch_enums.rs:1275`, `L0Scale = 99`).
    ///   * The map that loop walks is `metadata.newAllocations_`, keyed by `allocNode->component_`
    ///     ITSELF (`ddc/ddc_transformation_util.cpp:59`, `:1339`, `:1506`;
    ///     `ddc/ddl/ddl_conversion.cpp:820`, `:844`, `:1642`) — so the component IS the tracker key,
    ///     and the two L0 halves are already distinct keys before any category is read.
    ///
    /// So the mapping below is total on both, and the `todo!` was over-refusing.
    ///
    /// ⚠️ AND ONE THING THE AUTHORITY DOES *NOT* SAY, LEFT AS IT IS: `ddc::memories`
    /// (`ddc/ddc_metadata.h:20-21`) is the EIGHT-element set `{LX, L0, PELRF, SFPLRF, PTARF, PTXRF,
    /// PTIRF, HBM}` and `L0_SCALE` is NOT in it, while [`DdcMemory`] has nine arms including
    /// [`DdcMemory::L0Scale`]. The two are different objects — that set is a membership test, this enum
    /// is `newAllocations_`'s key — and nothing here needs them reconciled.
    fn alloc_component(&self, alloc: AllocId) -> DdcMemory {
        let component = self
            .with_tree(|tree| tree.node_of_alloc(alloc).and_then(|node| tree.allocate(node)))
            .map(|(_, held)| held.component)
            .unwrap_or_else(|| {
                panic!(
                    "tu::DscAllocations::alloc_component: {alloc:?} names no ALLOCATE of this DSC's \
                     scheduleTree_"
                )
            });
        match component {
            SenComponent::Hbm => DdcMemory::Hbm,
            SenComponent::Lx => DdcMemory::Lx,
            SenComponent::Pelrf => DdcMemory::PeLrf,
            SenComponent::Sfplrf => DdcMemory::SfpLrf,
            SenComponent::Ptarf => DdcMemory::PtaRf,
            SenComponent::Ptxrf => DdcMemory::PtxRf,
            SenComponent::Ptirf => DdcMemory::PtiRf,
            SenComponent::L0 => DdcMemory::L0,
            SenComponent::L0Scale => DdcMemory::L0Scale,
            other => todo!(
                "tu::DscAllocations::alloc_component: {other:?} has no DdcMemory arm — ddc tracks \
                 nine memories (ddc/metadata.rs:840-861) and this component is not one of them"
            ),
        }
    }

    /// ⛔ `allocNode->ldsIdx_`/`constIdx_` — [`DataOrigin`]'s constant arm again.
    fn alloc_origin(&self, _alloc: AllocId) -> DataOrigin {
        todo!(
            "tu::DscAllocations::alloc_origin: wants allocNode->ldsIdx_/constIdx_ as a DataOrigin — \
             its constant arm reads constIdx_, which l3::dl_ops::L3AllocateNode does not carry"
        )
    }

    /// THE ALLOCATION AS A SCHEDULE NODE — ⛔ TOTAL: one pointer in the reference.
    fn alloc_node(&self, alloc: AllocId) -> NodeId {
        self.with_tree(|tree| {
            tree.node_of_alloc(alloc).unwrap_or_else(|| {
                panic!(
                    "tu::DscAllocations::alloc_node: {alloc:?} names no ALLOCATE of this DSC's \
                     scheduleTree_"
                )
            })
        })
    }

    /// ⛔ `reduceUsersOrDeleteAllocation(di, storage, userNode, canDelete)`
    /// (`dsc/dsc2.cpp:2495`) — a `dsc/` function that drops one reference AND may unlink the node
    /// and clear the `memOrg_`/`constantInfo_` slot. Its `bool` DECIDES whether entry 111 acts.
    fn reduce_users_or_delete(
        &mut self,
        _alloc_use: tu::AllocationUse,
        _can_delete: tu::CanDelete,
    ) -> bool {
        todo!(
            "tu::DscAllocations::reduce_users_or_delete: wants \
             reduceUsersOrDeleteAllocation(di, storage, userNode, canDelete) \
             (dsc/dsc2.cpp:2495) — a `dsc/` function outside this campaign's file list, whose bool \
             decides whether entry 111 acts"
        )
    }
}

impl tu::AllocationsByNode for Dsc2Store<'_, '_> {
    /// The allocation at that node.
    fn allocation_of(&self, node: NodeId) -> Option<AllocId> {
        self.with_tree(|tree| tree.allocate(node).map(|(alloc, _)| alloc))
    }
}

impl crate::schedule::ddc::fold::Allocations for Dsc2Store<'_, '_> {
    /// ⛔⛔ `constantInfo_` IS NO LONGER WHAT BLOCKS THIS — [`tu::DscAllocations::allocation_in`] makes
    /// the SAME `getAllocation` call on both arms now. What is left is the KEY:
    /// [`StoredStream::storage`] is a [`crate::units::DfirUnit`] and `getAllocation`'s `storage` is a
    /// `SenComponents` (`dsc/dsc2.cpp:2587`), and the two are not the same set —
    /// [`crate::units::DfirUnit`] is *"the `SenComponents` subset DataflowIR binds"* with a
    /// `PtRow(Row)` arm that aggregates several of them and no reverse mapping. Picking one
    /// `SenComponents` per `DfirUnit` here would be a new closed table, and a WRONG row silently looks
    /// the stream up in a memory it does not live in — `getAllocation` then answers `nullptr` and the
    /// fold drops the coordinate.
    fn allocation(&self, _stored: StoredStream) -> Option<AllocId> {
        todo!(
            "fold::Allocations::allocation: wants getAllocation(stream, memory) keyed by a \
             SenComponents (dsc/dsc2.cpp:2587), and StoredStream::storage is a DfirUnit — a subset \
             with a PtRow(Row) arm and no reverse mapping. constantInfo_ IS carried now; see \
             tu::DscAllocations::allocation_in for the same call on both arms"
        )
    }

    /// ⛔ `dsc2::getValueAllocation(dsc, allocNode)` (`dsc/dsc2.cpp:5819`).
    fn value_allocation(&self, _scale: AllocId) -> Option<AllocId> {
        todo!(
            "fold::Allocations::value_allocation: wants dsc2::getValueAllocation(dsc, allocNode) \
             (dsc/dsc2.cpp:5819) — a `dsc/` function outside this campaign's file list"
        )
    }
}

impl tu::AllocationPaddings for Dsc2Store<'_, '_> {
    /// `allocNode->padding_` — ⭐ ANSWERED: it is exactly the [`tu::PaddingForm`] the L3 allocate
    /// node carries.
    fn padding(&self, alloc: AllocId) -> tu::PaddingForm {
        self.with_tree(|tree| tree.node_of_alloc(alloc).and_then(|node| tree.allocate(node)))
            .map(|(_, held)| held.padding)
            .unwrap_or_default()
    }
}

impl tu::ComponentAllocations for Dsc2Store<'_, '_> {
    /// `labeledDs_.at(lds).memOrg_.count(storage)` — ⭐ ANSWERED off the tree's `Org`.
    fn has_mem_org(&self, lds: LdsIdx, storage: SenComponent) -> bool {
        self.state
            .tree(self.dsc)
            .and_then(|tree| tree.org(lds))
            .is_some_and(|org| org.node(storage).is_some())
    }

    /// ⛔ `memOrg_[to] = memOrg_[from]` with the copy's `allocateNode_` CLEARED — [`super::tree::Org`]
    /// FUSES `isPresent` with the node identity (`memOrg_[storage].isPresent` AND its
    /// `allocateNode_` are one entry), so a present-but-unallocated organisation is unspellable.
    fn copy_mem_org_without_allocation(
        &mut self,
        _lds: LdsIdx,
        _from: SenComponent,
        _to: SenComponent,
    ) {
        todo!(
            "tu::ComponentAllocations::copy_mem_org_without_allocation: wants memOrg_[to] = \
             memOrg_[from] with allocateNode_ CLEARED — super::tree::Org fuses isPresent with the \
             node identity, so a present-but-unallocated entry is unspellable there"
        )
    }

    /// `memOrg_.at(storage).allocateNode_` — ⭐ ANSWERED.
    fn mem_org_allocation(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId> {
        let tree = self.state.tree(self.dsc)?;
        let node = tree.org(lds)?.node(storage)?;
        tree.with(|held| held.allocate(node).map(|(alloc, _)| alloc))
    }

    /// `memOrg_.at(storage).allocateNode_ = alloc` — ⭐ ANSWERED, and the pair
    /// [`super::tree::Org::set_node`] wants is READ OFF THE ALLOCATION rather than asked of the
    /// caller: the [`NodeId`] is [`super::tree::TreeData::node_of_alloc`] and the L3 view is that
    /// node's own [`super::tree::TreeData::allocate`], so nothing here is supplied and nothing is
    /// invented.
    ///
    /// ⛔ THIS IS NOT [`tu::DscAllocations::set_allocation_in`], AND THE DIFFERENCE IS THE ORDER. Its
    /// one caller reaches it with `clone`, the id
    /// [`tu::AllocateCloning::clone_allocate_after`] returned for a node it has already SPLICED into
    /// the tree (`ddc/transformation_util.rs:1069` then `:1090`) — so the lookup is total here where
    /// on `set_allocation_in`'s caller it would find nothing. An id naming no ALLOCATE is therefore a
    /// defect and not a state, and it is RECORDED as a refusal rather than dropped: the write is the
    /// only thing that files the clone in `memOrg_`, and a silent no-op would leave the clone
    /// unreachable from the labelled DS that owns it.
    fn set_mem_org_allocation(&mut self, lds: LdsIdx, storage: SenComponent, alloc: AllocId) {
        let Some(tree) = self.state.tree(self.dsc) else {
            return;
        };
        let held = tree.with(|held| {
            let node = held.node_of_alloc(alloc)?;
            let (_, minted) = held.allocate(node)?;
            Some((node, minted))
        });
        let Some((node, minted)) = held else {
            let _: Option<()> = self.state.refuse(
                "ComponentAllocations::set_mem_org_allocation: the AllocId names no ALLOCATE of \
                 this DSC's scheduleTree_, so memOrg_ cannot name its node",
            );
            return;
        };
        if let Some(org) = tree.org(lds) {
            org.set_node(storage, node, minted);
        }
    }

    /// `constantInfo_.at(constant).allocations_.count(storage)` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::ConstantInfo::allocations`].
    ///
    /// ⛔ AN ID THE TABLE DOES NOT HOLD IS `false` AND NOT A THROW, and that is the trait's own return:
    /// it asks whether an allocation EXISTS, and a constant that does not exist has none.
    fn has_constant_allocation(&self, constant: ConstIdx, storage: SenComponent) -> bool {
        self.facts().with_dsc(|dsc| {
            dsc.ddc
                .constants
                .get(&constant)
                .is_some_and(|held| held.allocations.contains_key(&storage))
        })
    }

    /// `constantInfo_.at(constant).allocations_[storage] = alloc` — ⭐ ANSWERED through the shared
    /// `currDsc` cell.
    ///
    /// ⛔ A NO-OP FOR A CONSTANT THE TABLE DOES NOT HOLD, which is `constantInfo_.at()`'s throw:
    /// `operator[]` fills a slot of an entry that EXISTS, and inventing the entry would put an
    /// allocation on a constant no DSC declared.
    fn set_constant_allocation(
        &mut self,
        constant: ConstIdx,
        storage: SenComponent,
        alloc: AllocId,
    ) {
        self.facts().with_dsc_mut(|dsc| {
            if let Some(held) = dsc.ddc.constants.get_mut(&constant) {
                held.allocations.insert(storage, alloc);
            }
        });
    }
}

impl tu::AllocateCloning for Dsc2Store<'_, '_> {
    /// ⛔ `allocNode` AS THE `ddc` VIEW OF IT — [`AllocateNode`] carries the PLACED state
    /// (`startAddressCoreCorelet_`, `bufferOffsetCoreCorelet_`, `paddingSizes_`,
    /// `gapStickSpread_`), and this tree holds the L3 view
    /// ([`crate::schedule::l3::dl_ops::L3AllocateNode`]) plus whatever stage 2a placed. An allocation
    /// stage 2a did not place has no `ddc` view at all, and inventing one is inventing an address.
    fn allocate(&self, _alloc: AllocId) -> AllocateNode {
        todo!(
            "tu::AllocateCloning::allocate: wants the whole dsc2::AllocateNode — its placed state \
             (startAddressCoreCorelet_, bufferOffsetCoreCorelet_, paddingSizes_, gapStickSpread_) \
             exists only for an allocation the memory tracker placed, and stage 2a stops at that \
             tracker"
        )
    }

    fn clone_allocate_after(&mut self, _alloc: AllocId, _body: AllocateNode) -> AllocId {
        todo!(
            "tu::AllocateCloning::clone_allocate_after: wants node->clone() with a whole \
             dsc2::AllocateNode body placed after alloc's node — see AllocateCloning::allocate"
        )
    }

    fn set_temp_storage_for_compute(&mut self, _alloc: AllocId, _compute: NodeName) {
        todo!(
            "tu::AllocateCloning::set_temp_storage_for_compute: wants \
             allocNode->tempStorageForCompute_ = compute (dsc/dsc2.h:978) — a field \
             l3::dl_ops::L3AllocateNode does not carry"
        )
    }
}

impl tu::ComputeCloning for Dsc2Store<'_, '_> {
    /// ⛔ THE SAME ONE ACCESSOR [`tr::ComputeNodes::compute`] names — the arm is present, the by-kind
    /// reader is not.
    fn compute(&self, _node: NodeId) -> ComputeNode {
        todo!(
            "tu::ComputeCloning::compute: wants the whole dsc2::ComputeNode — Kind::Compute HOLDS \
             it (stages/tree.rs:96); see tr::ComputeNodes::compute for the reader it needs"
        )
    }

    /// `node->clone()` with `body`, placed IMMEDIATELY AFTER `node` — ⭐ ANSWERED, and it needs no
    /// reader: the body is HANDED IN, so this only mints and links, exactly as
    /// [`tu::NodeCloning::clone_transfer_after`] does for a transfer.
    ///
    /// ⛔ NOTHING ELSE IN THE CRATE CONSTRUCTS A [`super::tree::Kind::Compute`], so this is the first
    /// writer of that arm. That is why the census reads `compute: 0` and why every compute WALK in
    /// this file answers empty: the arm was always spellable and never spelled.
    fn clone_compute_after(&mut self, node: NodeId, body: ComputeNode) -> NodeId {
        self.with_tree_mut(|tree| {
            let name = body.name.clone();
            let clone = tree.add(name, Kind::Compute(body), None);
            tree.link(clone, tu::InsertionPoint::After(node));
            clone
        })
    }
}

impl tu::NodeCloning for Dsc2Store<'_, '_> {
    /// `node->clone()` with `body`, placed IMMEDIATELY AFTER `node` — ⭐ ANSWERED: a transfer is a
    /// kind this tree holds.
    fn clone_transfer_after(&mut self, node: NodeId, body: TransferNode) -> NodeId {
        self.with_tree_mut(|tree| {
            let name = body.name.clone();
            let clone = tree.add(name, Kind::Transfer(body), None);
            tree.link(clone, tu::InsertionPoint::After(node));
            clone
        })
    }
}

/// ⭐ ONE `labeledDs_` ENTRY AS THIS CARRIER HOLDS IT — [`crate::schedule::l3::dsc::LabeledDs`] with
/// the two indices [`tu::LabeledDsEntry`] writes beside it.
///
/// ⛔ `referenceLdsIdx_` IS NOT A `l3::dsc::LabeledDs` FIELD, and it is the field entry 308 stamps on
/// every internal tensor it mints (`dsc/dscdefn.h:324`) — so it is carried HERE rather than left
/// unwritable, which is what makes [`tu::LabeledDsEntry`] satisfiable at all.
#[derive(Debug, Clone)]
pub struct Dsc2LdsEntry {
    /// The projected entry.
    pub held: crate::schedule::l3::dsc::LabeledDs,
    /// `ldsIdx_` — the entry's OWN self-index.
    pub recorded: LdsIdx,
    /// `referenceLdsIdx_` — which tensor an internal one is derived from, [`None`] for the
    /// reference's `-1`.
    pub reference: Option<LdsIdx>,
}

impl tu::LabeledDsEntry for Dsc2LdsEntry {
    fn recorded_lds_idx(&self) -> LdsIdx {
        self.recorded
    }

    fn set_recorded_lds_idx(&mut self, recorded: LdsIdx) {
        self.recorded = recorded;
    }

    fn set_reference_lds_idx(&mut self, reference: LdsIdx) {
        self.reference = Some(reference);
    }
}

impl tu::NewLabeledDs for Dsc2Store<'_, '_> {
    type Entry = Dsc2LdsEntry;

    /// `labeledDs_.size() - 1` — ⛔ TOTAL because [`crate::schedule::l3::dsc::LabeledDsList`] is
    /// non-empty by type.
    fn last_lds_pos(&self) -> LdsIdx {
        self.facts()
            .with_dsc(|dsc| ddc_state::lds_positions(dsc).last().copied())
            .unwrap_or_else(|| panic!("LabeledDsList is non-empty by type"))
    }

    /// `labeledDs_.back().ldsIdx_`.
    fn last_recorded_lds_idx(&self) -> LdsIdx {
        self.facts()
            .with_dsc(|dsc| dsc.labeled_ds.back().recorded())
    }

    /// `labeledDs_.back().ldsIdx_ = recorded` — ⭐ ANSWERED through the shared `currDsc` cell, and it
    /// is the LAST entry's OWN index, which [`tu::add_new_lds`] bumps by one before the insert.
    fn set_last_recorded_lds_idx(&mut self, recorded: LdsIdx) {
        self.facts().with_dsc_mut(|dsc| {
            dsc.labeled_ds.back_mut().set_recorded(recorded);
        });
    }

    /// ⛔⛔ `labeledDs_.insert(end() - 1, entry)` — AND THE CELL IS NO LONGER WHAT BLOCKS IT.
    /// `referenceLdsIdx_` (`dsc/dscdefn.h:324`) is: [`Dsc2LdsEntry`] carries it because
    /// [`crate::schedule::l3::dsc::LabeledDs`] does not, and `add_new_lds` STAMPS it on the entry it
    /// inserts (`tu::LabeledDsEntry::set_reference_lds_idx`). Pushing the projected half alone would
    /// drop which tensor this internal one is derived from — the field the reference itself writes
    /// here and nowhere else — and it would compile.
    fn insert_lds_before_last(&mut self, _entry: Self::Entry) {
        todo!(
            "tu::NewLabeledDs::insert_lds_before_last: wants labeledDs_.insert(end() - 1, entry) \
             carrying its referenceLdsIdx_ (dsc/dscdefn.h:324), which l3::dsc::LabeledDs does not \
             project — add_new_lds stamps it on exactly the entry this inserts, so pushing the \
             projected half alone drops it"
        )
    }

    /// ⛔ `labeledDs_.at(pos).memOrg_.clear()` — [`super::tree::Org`] has no clear, and clearing it
    /// would ORPHAN the allocate nodes stage 2a filed there.
    fn clear_mem_org(&mut self, _pos: LdsIdx) {
        todo!(
            "tu::NewLabeledDs::clear_mem_org: wants labeledDs_.at(pos).memOrg_.clear() — \
             super::tree::Org has no clear, and clearing it orphans the allocate nodes stage 2a \
             filed in it"
        )
    }

    /// The `allocateNode_`s of `labeledDs_.at(pos).memOrg_` — ⭐ ANSWERED off the tree's `Org`.
    fn mem_org_allocations(&self, pos: LdsIdx) -> Vec<AllocId> {
        let Some(tree) = self.state.tree(self.dsc) else {
            return Vec::new();
        };
        let Some(org) = tree.org(pos) else {
            return Vec::new();
        };
        // Every storage this organisation names, as the allocations those nodes are.
        tree.with(|held| {
            [SenComponent::Hbm, SenComponent::Lx]
                .into_iter()
                .filter_map(|storage| org.node(storage))
                .filter_map(|node| held.allocate(node).map(|(alloc, _)| alloc))
                .collect()
        })
    }

    /// `allocNode->ldsIdx_` — ⭐ ANSWERED off the L3 view.
    fn alloc_lds_idx(&self, alloc: AllocId) -> Option<LdsIdx> {
        self.with_tree(|tree| tree.node_of_alloc(alloc).and_then(|node| tree.allocate(node)))
            .map(|(_, held)| held.lds)
    }

    /// ⛔ `allocNode->ldsIdx_ = lds` — [`crate::schedule::l3::dl_ops::L3AllocateNode::lds`] is
    /// carried inside [`super::tree::Kind::Allocate`], and [`super::tree::TreeData`] has no setter
    /// for it; adding one is tree surgery this carrier does not own.
    fn set_alloc_lds_idx(&mut self, _alloc: AllocId, _lds: LdsIdx) {
        todo!(
            "tu::NewLabeledDs::set_alloc_lds_idx: wants allocNode->ldsIdx_ = lds — \
             super::tree::TreeData has no setter for L3AllocateNode::lds"
        )
    }

    /// ⛔ Every `{ALLOCATE, TRANSFER, COMPUTE, LOOP}` slot WITH its lds index — the COMPUTE and the
    /// parametric-LOOP arms are both unspellable here (no Compute arm, no `parametricLdsIdx_`).
    fn tree_lds_slots(&self) -> Vec<(tu::TreeLdsSlot, Option<LdsIdx>)> {
        todo!(
            "tu::NewLabeledDs::tree_lds_slots: wants every ALLOCATE/TRANSFER/COMPUTE/LOOP slot \
             naming a labelled DS — its Compute arm needs a COMPUTE node and its ParametricLoop arm \
             needs LoopNode::parametricLdsIdx_, neither of which this tree carries. A PARTIAL walk \
             would silently leave entry 308's repointing half done."
        )
    }

    /// ⛔ The write half of the same walk.
    fn set_tree_lds(&mut self, _slot: tu::TreeLdsSlot, _lds: LdsIdx) {
        todo!(
            "tu::NewLabeledDs::set_tree_lds: wants that slot's myLdsIdx_/ldsIdx_/\
             parametricLdsIdx_ = lds — see tree_lds_slots"
        )
    }

    /// The `isOpaqueOp_` computes that walk reaches — ⭐ EMPTY, because there are no COMPUTE nodes.
    fn opaque_computes(&self) -> Vec<NodeId> {
        Vec::new()
    }
}

impl v1::PrepDsc for Dsc2Store<'_, '_> {
    /// `numCoreletsUsed_DSC2_ = corelets` — ⭐ ANSWERED, into the state's own cell.
    fn set_corelets_used_dsc2(&mut self, corelets: u32) {
        self.facts().set_corelets_dsc2(corelets);
    }

    /// `computeOp_`, in order — ⭐ ANSWERED. This is `run_v1`'s FIRST provider call and its EMPTY
    /// answer skips the DSC, so it is a construction argument of the state rather than a refusal.
    fn compute_ops(&self) -> Vec<v1::DscComputeOp> {
        self.facts().ops()
    }

    /// `computeOp_.emplace(begin() + at)` filled with `op` — ⛔ [`None`] past the end.
    fn insert_compute_op(&mut self, at: usize, op: v1::DscComputeOp) -> Option<()> {
        self.facts().with_ops_mut(|ops| {
            if at > ops.len() {
                return None;
            }
            ops.insert(at, op);
            Some(())
        })
    }

    /// `computeOp_.at(at).opFuncName = op_func`.
    fn set_op_func(&mut self, at: usize, op_func: OpFunc) {
        self.facts().with_ops_mut(|ops| {
            if let Some(op) = ops.get_mut(at) {
                op.op_func = Some(op_func);
            }
        });
    }

    /// `computeOp_.at(at).inputLabeledDs.push_back(&labeledDs_.at(lds))`.
    fn push_op_input(&mut self, at: usize, lds: LdsIdx) {
        self.facts().with_ops_mut(|ops| {
            if let Some(op) = ops.get_mut(at) {
                op.inputs.push(lds);
            }
        });
    }

    /// SOME `constantInfo_` entry named `useZeroMean` whose single datum is `1` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::DdcFacts::constants`].
    ///
    /// ⛔ IT IS `getSingleDataStrict(constinfo.data_).at(0) == 1` AND NOT *"has a datum"*
    /// (`ddc/ddcv1.cpp:2066-2068`): an entry called `useZeroMean` whose datum is `0` is the SWITCH
    /// TURNED OFF, and answering `true` for it would swap `EXX2` for `EXX2_ZEROMEAN` on a program that
    /// asked for the opposite. `.at(0)` on an EMPTY data vector is that vector's own throw, which is
    /// [`Option::is_some_and`]'s `false` here — no scratchy constant is empty (all twelve of `g0/`
    /// carry one datum).
    fn declares_zero_mean_constant(&self) -> bool {
        self.facts().with_dsc(|dsc| {
            dsc.ddc.constants.values().any(|held| {
                held.name.0 == "useZeroMean" && held.data.first().is_some_and(|datum| *datum == 1)
            })
        })
    }

    /// ⛔ `computeOp_.at(at).opConsts.at("useZeroMean")[0] == 1`.
    fn op_declares_zero_mean(&self, _at: usize) -> bool {
        todo!(
            "v1::PrepDsc::op_declares_zero_mean: wants computeOp_.at(at).opConsts.at(\
             \"useZeroMean\")[0] == 1 (ddc/ddcv1.cpp:2074-2078) — opConsts is not a field of \
             v1::DscComputeOp, and this DECIDES an op-func swap"
        )
    }

    /// `labeledDs_.at(lds)` — ⭐ ANSWERED: the entry [`tu::add_new_lds`] clones.
    fn lds_entry(&self, lds: LdsIdx) -> Option<Self::Entry> {
        let held = self.facts().with_lds(lds, Clone::clone)?;
        Some(Dsc2LdsEntry {
            recorded: held.recorded(),
            held,
            reference: None,
        })
    }

    /// `dsName_ += suffix` — ⭐ ANSWERED, and it is `+=` and NOT `=`: the reference APPENDS
    /// `_internalInput`/`_internalKernel` to the name the cloned entry already carries
    /// (`ddc/ddcv1.cpp:2098`), so an assignment would lose which tensor the internal one was derived
    /// from in the one place a human reads it.
    ///
    /// ⛔ A NO-OP FOR AN INDEX THE LIST DOES NOT HOLD, which is `labeledDs_.at()`'s throw.
    fn append_lds_name(&mut self, lds: LdsIdx, suffix: v1::InternalLds) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.append_name(suffix.suffix()));
    }

    /// `dsType_ = DsTypes::INTERNAL` — ⭐ ANSWERED through the shared `currDsc` cell.
    fn set_lds_internal(&mut self, lds: LdsIdx) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_ds_type(tr::DsType::Internal));
    }

    /// `wordLength = length` — ⭐ ANSWERED.
    fn set_lds_word_length(&mut self, lds: LdsIdx, length: WordLength) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_word_length(length));
    }

    /// `dataFormat_ = format` — ⭐ ANSWERED.
    fn set_lds_format(&mut self, lds: LdsIdx, format: DataFormat) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_data_format(format));
    }

    /// `scaledLdsCategory_ = category` — ⭐ ANSWERED as the closed three-way, which is why the
    /// `REGULAR_TENSOR`/`VALUE_TENSOR` write entry 308 makes is now expressible.
    fn set_lds_scaled_category(&mut self, lds: LdsIdx, category: ScaledLds) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_scaled_category(category));
    }

    /// `mxInfo_ = labeledDs_.at(from).mxInfo_` — ⭐ ANSWERED.
    ///
    /// ⛔ THE COPY IS OF THE RAW FIELD, so it is read back through `from`'s own category exactly as the
    /// reference's `labeledDs_.at(from).mxInfo_` is: a `from` that is not a `SCALE_TENSOR` copies the
    /// `mxInfo_` it holds, which is `{blkSize: 0, relatedLdsIdx: -1}` — [`None`] here.
    fn copy_mx_info(&mut self, to: LdsIdx, from: LdsIdx) {
        let Some(mx_info) = self
            .facts()
            .with_lds(from, crate::schedule::l3::dsc::LabeledDs::scale_tensor)
        else {
            return;
        };
        let _: Option<()> = self
            .facts()
            .with_lds_mut(to, |held| held.set_mx_info(mx_info));
    }

    /// `primaryDsInfo_.at(lds's dsType_).stickDimOrder_` — ⭐ ANSWERED.
    ///
    /// ⛔⛔ ITS EMPTY VECTOR IS NOW RECORDED, AND THAT IS THE POINT. `.at(dsType_)` THROWS where the
    /// map holds no such type (`ddc/ddcv1.cpp:2112`), and this used to answer `unwrap_or_default()` —
    /// an EMPTY dim order indistinguishable, at the callsite, from a labelled DS that genuinely has
    /// no stick dims. Its caller `extend`s two vectors with the answer (`ddc/v1.rs:5868-5877`), so a
    /// silent empty builds an internal tensor with NO sticks and reports nothing. The vector stays
    /// empty — the trait cannot spell absence and must not invent a dim — but the refusal is filed on
    /// the state, so `Dsc2State::refusals` names it instead of the count looking healthy.
    fn stick_order(&self, lds: LdsIdx) -> Vec<PrimaryDim> {
        match self.facts().with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds)) {
            Some(dims) => dims.0.iter().map(|(dim, _)| *dim).collect(),
            None => self
                .state
                .refuse(
                    "PrepDsc::stick_order: primaryDsInfo_ holds no entry for that labelled DS's \
                     dsType_, so its stickDimOrder_ is UNKNOWN and not empty",
                )
                .unwrap_or_default(),
        }
    }

    /// `primaryDsInfo_.at(lds's dsType_).stickSize_` — ⭐ ANSWERED, and its empty is recorded for the
    /// same reason [`Self::stick_order`]'s is.
    fn stick_sizes_of(&self, lds: LdsIdx) -> Vec<Elements> {
        match self.facts().with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds)) {
            Some(dims) => dims.0.iter().map(|(_, size)| *size).collect(),
            None => self
                .state
                .refuse(
                    "PrepDsc::stick_sizes_of: primaryDsInfo_ holds no entry for that labelled DS's \
                     dsType_, so its stickSize_ is UNKNOWN and not empty",
                )
                .unwrap_or_default(),
        }
    }

    /// `internalDsInfo.layoutDimOrder_ = inputDsInfo.layoutDimOrder_` (`ddc/ddcv1.cpp:2115`) — ⭐
    /// ANSWERED THROUGH THE SHARED `currDsc` CELL.
    ///
    /// ⛔⛔ A RECORDED CORRECTION: this said *"`DesignSpaceConfig::primary_ds_info` is not behind a
    /// cell in `Dsc2Facts`"*. The WHOLE `DesignSpaceConfig` is — `Dsc2Facts::dsc` is a
    /// `RefCell<DesignSpaceConfig>` and [`super::ddc_state::Dsc2Facts::with_dsc_mut`] hands out
    /// `&mut` to it off a `&self` (`stages/ddc_state.rs:335`, `:359`), which is the same cell every
    /// `set_lds_*` on this trait already writes through. `primary_ds_info` is a plain field of it.
    ///
    /// ⭐ THE `from` IS READ THROUGH ITS OWN `dsType_`, not as a position: the reference indexes
    /// `primaryDsInfo_.at(inputLds->dsType_)` (`ddc/ddcv1.cpp:2112`), so the source layout is the one
    /// filed under the SOURCE'S TYPE and two labelled DSes of one type share it.
    ///
    /// ⛔ THE TWO ARMS ARE THE REFERENCE'S TWO OPERATORS AND NOT A DEFAULT. `.at(dsType)` on the
    /// source THROWS where no such type is filed — recorded here as a refusal, because writing
    /// nothing would leave `INTERNAL` carrying whatever layout a previous op left — while
    /// `primaryDsInfo_[INTERNAL]` is `operator[]`, which DEFAULT-CONSTRUCTS the entry, so the vacant
    /// arm inserts the copied layout with the EMPTY stick order that entry 308 then pushes into.
    fn set_internal_layout_from(&mut self, from: LdsIdx) {
        let recorded = self.facts().with_dsc_mut(|dsc| {
            let Some(ds_type) = ddc_state::ds_type_of(dsc, from) else {
                return false;
            };
            let Some(layout) = dsc
                .primary_ds_info
                .get(&ds_type)
                .map(|held| held.layout.clone())
            else {
                return false;
            };
            match dsc
                .primary_ds_info
                .entry(crate::schedule::ddc::transformation::DsType::Internal)
            {
                std::collections::btree_map::Entry::Occupied(mut held) => {
                    held.get_mut().layout = layout;
                }
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(crate::schedule::l3::dsc::PrimaryDsInfo {
                        layout,
                        stick: StickDims(Vec::new()),
                    });
                }
            }
            true
        });
        if !recorded {
            let _: Option<()> = self.state.refuse(
                "PrepDsc::set_internal_layout_from: the source labelled DS names no dsType_ with a \
                 primaryDsInfo_ entry, so INTERNAL's layoutDimOrder_ has nothing to copy",
            );
        }
    }

    /// ⛔ `primaryDsInfo_[INTERNAL].stickDimOrder_.push_back(dim)` WITH
    /// `stickRepl_.push_back(repl)` — TWO FIELD GAPS, and the cell is not one of them (see
    /// [`Self::set_internal_layout_from`], which writes the same map).
    ///
    /// 1. **`stickRepl_` IS NOT PROJECTED.** The reference pushes one per dim and always `1` at this
    ///    site (`ddc/ddcv1.cpp:2117-2122`), but [`crate::schedule::l3::dsc::PrimaryDsInfo`] carries
    ///    `layout` and `stick` and nothing else, so [`v1::StickRepl`] has nowhere to land. Pushing
    ///    the dim and discarding the replication is an effect DROPPED silently.
    /// 2. **`stickDimOrder_` AND `stickSize_` ARE FUSED HERE AND SEPARATE THERE.** `PrimaryDsInfo`
    ///    holds one [`StickDims`] — a `Vec<(PrimaryDim, Elements)>` — while the reference pushes the
    ///    dims in one loop and the sizes in another, over DIFFERENT lengths in the general case
    ///    (`:2117-2135`: two dim loops, then two size loops each cutting its first entry by the slice
    ///    count). A `(dim, size)` pair cannot be formed from either call alone, so the arrival of a
    ///    dim with no size yet is unspellable rather than merely unwritten.
    ///
    /// ⭐ THE REPAIR IS IN `l3/dsc.rs`: `PrimaryDsInfo` splitting `stick` into `stick_dim_order`,
    /// `stick_size` and `stick_repl` the way `dsc/dscdefn.h:474` holds them. Until then BOTH halves
    /// stop, so entry 308 cannot half-fill the internal tensor's sticks.
    fn push_internal_stick_dim(&mut self, _dim: PrimaryDim, _repl: v1::StickRepl) {
        todo!(
            "v1::PrepDsc::push_internal_stick_dim: wants \
             primaryDsInfo_[INTERNAL].stickDimOrder_.push_back(dim) with stickRepl_.push_back(repl) \
             (ddc/ddcv1.cpp:2117-2122) — l3::dsc::PrimaryDsInfo projects no stickRepl_ and fuses \
             stickDimOrder_ with stickSize_ as one StickDims, so a dim cannot be pushed alone"
        )
    }

    /// ⛔ `primaryDsInfo_[INTERNAL].stickSize_.push_back(size)` — the other half of the same fusion.
    fn push_internal_stick_size(&mut self, _size: Elements) {
        todo!(
            "v1::PrepDsc::push_internal_stick_size: wants \
             primaryDsInfo_[INTERNAL].stickSize_.push_back(size) (ddc/ddcv1.cpp:2124-2135) — see \
             push_internal_stick_dim: StickDims pairs the size with a dim this call does not carry"
        )
    }

    /// `labeledDs_.at(to).memOrg_[LX] = labeledDs_.at(from).memOrg_.at(LX)` then
    /// `memOrg_.at(LX).allocateNode_` read back (`ddc/ddcv1.cpp:2137-2138`) — ⭐ ANSWERED, and it is
    /// the SOURCE's node copied BY IDENTITY, which is what the reference's pointer copy is.
    ///
    /// ⛔ THE COPY ALIASES ONE NODE ACROSS TWO `memOrg_` ENTRIES, AND THAT IS THE REFERENCE'S OWN
    /// STATE — `newLds->memOrg_[LX]` holds the very `allocInput0` pointer `inputLds` holds, which is
    /// why the next thing it does is clone that node and overwrite the slot
    /// (`ddc/ddcv1.cpp:2141-2156`, then [`Self::set_lx_alloc`]). ⚠️ WHILE THE ALIAS STANDS,
    /// [`super::state::DscTree::home_of`] — a reverse scan of the organisations, so FIRST match wins
    /// — can name either labelled DS as the node's home, and with it either one's `placed` and
    /// `allocUsers_`. Nothing in entry 308 reads those between the two calls (the users it needs come
    /// from the caller's own `allocs` map), so this is recorded rather than worked around; a caller
    /// that DID read them would need `Org` keyed by node identity.
    ///
    /// ⛔ `isZeroPadded` IS NOT COPIED AND CANNOT BE: [`super::tree::Org`] files it as a
    /// `SenComponent` set with no writer anywhere in the crate, so `lx_zero_padded()` is constantly
    /// `Some(false)` for every organisation this tree builds and the copy of it is the identity.
    fn copy_lx_mem_org(&mut self, to: LdsIdx, from: LdsIdx) -> Option<AllocId> {
        let tree = self.state.tree(self.dsc)?;
        let node = tree.org(from)?.node(SenComponent::Lx)?;
        let (alloc, minted) = tree.with(|held| held.allocate(node))?;
        tree.org(to)?.set_node(SenComponent::Lx, node, minted);
        Some(alloc)
    }

    /// `memOrg_.at(LX).allocateNode_ = alloc` (`ddc/ddcv1.cpp:2156`) — ⭐ ANSWERED the way
    /// [`tu::ComponentAllocations::set_mem_org_allocation`] is, and for the same reason: its caller
    /// reaches it with the id [`Self::insert_alloc_before`] returned for a node ALREADY spliced in
    /// (`ddc/v1.rs:5931`), so the `(NodeId, L3AllocateNode)` pair is read off the allocation itself.
    fn set_lx_alloc(&mut self, lds: LdsIdx, alloc: AllocId) {
        tu::ComponentAllocations::set_mem_org_allocation(self, lds, SenComponent::Lx, alloc);
    }

    /// ⛔ `memOrg_.at(LX).isPresent = present` — [`super::tree::Org`] FUSES presence with the node,
    /// so `isPresent = false` on an organisation that names a node is unspellable.
    fn set_lx_present(&mut self, _lds: LdsIdx, _present: bool) {
        todo!(
            "v1::PrepDsc::set_lx_present: wants memOrg_.at(LX).isPresent = present — \
             super::tree::Org fuses isPresent with the allocateNode_ identity"
        )
    }

    /// ⛔ `sibling->getMutableParent()->addChildNode(node, true, sibling)` with a whole
    /// `dsc2::AllocateNode` — see [`tu::AllocateCloning::allocate`].
    fn insert_alloc_before(&mut self, _sibling: AllocId, _node: AllocateNode) -> Option<AllocId> {
        todo!(
            "v1::PrepDsc::insert_alloc_before: wants a whole dsc2::AllocateNode spliced before \
             sibling — super::tree::Kind::Allocate holds the L3 view, and the ddc view carries \
             placed state no unplaced allocation has"
        )
    }

    /// `under->getMutableParent()->addChildNode(node, false, sibling)` — ⭐ ANSWERED: a transfer is
    /// a kind this tree holds, and both [`v1::InsertUnder`] arms resolve to a parent it knows.
    fn insert_transfer_after(
        &mut self,
        under: v1::InsertUnder,
        sibling: NodeId,
        node: TransferNode,
    ) -> Option<NodeId> {
        self.with_tree_mut(|tree| {
            let anchor = match under {
                v1::InsertUnder::ParentOfTransfer(at) => at,
                v1::InsertUnder::ParentOfAlloc(alloc) => tree.node_of_alloc(alloc)?,
            };
            let parent = tree.parent(anchor)?;
            let name = node.name.clone();
            let minted = tree.add(name, Kind::Transfer(node), None);
            // `addChildNode(node, /*addBefore=*/false, sibling)` under `anchor`'s parent, which is
            // where the sibling itself sits when the two agree and where the reference puts it when
            // they do not.
            let at = if tree.parent(sibling) == Some(parent) {
                tu::InsertionPoint::After(sibling)
            } else {
                tu::InsertionPoint::LastIn(parent)
            };
            tree.link(minted, at);
            Some(minted)
        })
    }
}
