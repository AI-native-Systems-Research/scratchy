// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE THREE SURFACES ENTRY 333 READS — INHABITED, AND EVERY METHOD UNREACHABLE.
//!
//! `fill_loop_offsets_and_addresses` (`e333`) reaches its inputs through `DscOffsetFacts`'s three
//! getters on the `F` carrier, and all three of those getters REFUSE in `super::reads` — they answer
//! [`None`]. So nothing in this file is ever called. The types exist because
//! [`crate::schedule::l3::dl_ops::run`]'s where-clause names `F::Sizes`, `F::Nodes` and `F::Facts`,
//! and an associated type has to be inhabited by SOMETHING before the carrier compiles at all. Each
//! body is a `todo!` that NAMES the single fact that method wants, so the record of what is missing
//! lives in the code rather than in a note beside it.
//!
//! # ⛔ WHY THE GETTERS REFUSE — THE ONE REASON THIS FILE IS NOTHING BUT `todo!`s
//!
//! `run` takes `sdsc: &mut SuperDsc` while the read carrier arrives as `&'a F`, so no `F` may alias
//! the super-DSC. `L3OffsetInputs` wants precisely what lives there: the datastage extents
//! ([`v1::StageSizes`]), the address-granularity table with the padding, symbolic and PE/SFP facts
//! beside it ([`v1::OffsetSizes`]), and the whole [`v1::ScheduleWalk`] / [`v1::ScheduleNodes`] walk of
//! `dscs_.at(dsc).scheduleTree_`. A snapshot would not rescue it either — entries 380/351 rewrite the
//! chunk stage while the stage runs. Unifying the two carriers is review 382's own cross-entry work
//! over entries 050/219/220/222/292/333; until that lands, the honest shape is a refusal at the
//! getter and a named `todo!` below it.
//!
//! ⛔ AND NEVER A PLAUSIBLE CONSTANT. Every method here hands back an extent, an address scale, a
//! stride or a page size that a PLACEMENT is computed from, and a fabricated placement is the failure
//! this crate ranks worse than a stop.

use std::collections::BTreeMap;
use std::num::NonZeroU64;

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{AllocId, BlockId, ConstIdx, NodeId, NodeKind, PadType, Stride};
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::PaddingForm;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{ComputeNode, LdsIdx, LdsScale, TransferNode, WordLength};
use crate::schedule::l3::dl_ops::{L3OffsetFacts, VariableSymbol};
use crate::schedule::l3::dsc::{DimStage, DscIdx, MemOrgs};
use crate::units::Corelet;

use super::tree::Org as TreeOrg;

/// THE DATASTAGE EXTENTS AND THE ADDRESS-GRANULARITY TABLE ENTRY 333 ASKS FOR — every answer lives in
/// `dataStageParam_` or `dscGlobal.sysDef`, neither of which `&'a F` can reach.
#[derive(Debug, Clone, Copy, Default)]
pub struct OffsetSizes;

/// THE DSC'S OWN `scheduleTree_` AS ENTRY 333 WALKS IT — a wider walk than the L3 tree arena answers,
/// and this surface is not wired to it.
#[derive(Debug, Clone, Copy, Default)]
pub struct OffsetNodes;

/// THE FOUR `dsc2`/`DesignSpaceConfig` SEAMS AND THE TWO DATA STAGES ENTRY 333 NEEDS BESIDE
/// [`v1::OffsetSizes`].
#[derive(Debug, Clone, Copy, Default)]
pub struct OffsetFacts;

/// ONE `DataStructDims` [`OffsetFacts`] WOULD HAND BACK — [`L3OffsetFacts::size_stage`] and
/// [`L3OffsetFacts::chunk_stage`] both stop before any method here is asked.
#[derive(Debug, Clone, Copy, Default)]
pub struct OffsetStage;

impl v1::StageSizes for OffsetSizes {
    fn dim_extent(
        &self,
        _stage: DatastageId,
        _dim: PrimaryDim,
        _unit: SenComponent,
        _corelet: Option<Corelet>,
        _padding: PadType,
        _density: v1::Density,
    ) -> Extent {
        todo!(
            "v1::StageSizes::dim_extent: wants dataStageParam_.at(stage).ss_.primaryDimToVal_st(dim, \
             comp, -1, corelet, padding, density), a field of the &mut SuperDsc that `&'a F` cannot \
             alias"
        )
    }

    fn comp_view_scaled(
        &self,
        _stage: DatastageId,
        _dim: PrimaryDim,
        _unit: SenComponent,
        _corelet: Option<Corelet>,
        _padding: PadType,
        _density: v1::Density,
    ) -> Extent {
        todo!(
            "v1::StageSizes::comp_view_scaled: wants \
             dataStageParam_.at(stage).ss_.dataStageDimToVal_compView_st(dim, unit, corelet, \
             padding, density) on the LIVE data stage, which `&'a F` cannot alias"
        )
    }

    fn lds_scale(&self, _lds: LdsIdx, _dim: PrimaryDim) -> Option<LdsScale> {
        todo!(
            "v1::StageSizes::lds_scale: wants \
             labeledDs_.at(lds).scale_.at(getDimIndexInLayoutOrder(dsType_, dim)) — a DSC \
             labelled-DS layout fact on the super-DSC"
        )
    }

    fn dim_density(&self, _lds: LdsIdx, _dim: PrimaryDim) -> v1::Density {
        todo!(
            "v1::StageSizes::dim_density: wants 1.0 / labeledDs_.at(lds).mxInfo_.blkSize on a scale \
             tensor's mx dim, a DSC labelled-DS fact on the super-DSC"
        )
    }

    fn corelet_split(&self, _stage: DatastageId, _dim: PrimaryDim) -> Option<Vec<Elements>> {
        todo!(
            "v1::StageSizes::corelet_split: wants \
             dataStageParam_.at(stage).ss_.coreletSplit_.at(dim) on the LIVE data stage"
        )
    }

    fn alloc_padding_sizes(&self, _alloc: AllocId, _dim: PrimaryDim) -> Option<v1::PaddingSizes> {
        todo!(
            "v1::StageSizes::alloc_padding_sizes: wants allocNode->paddingSizes_.at(dim) on the \
             ALLOCATE node — the allocation's map, NOT the stage's"
        )
    }

    fn stage_padding_sizes(
        &self,
        _stage: DatastageId,
        _dim: PrimaryDim,
    ) -> Option<v1::PaddingSizes> {
        todo!(
            "v1::StageSizes::stage_padding_sizes: wants \
             dataStageParam_.at(stage).ss_.paddingSizes_.at(dim) — the STAGE's map, a different one \
             from the allocation's, and on the LIVE data stage"
        )
    }

    fn size_stage(&self, _alloc: AllocId) -> DatastageId {
        todo!(
            "v1::StageSizes::size_stage: wants getSizeDataStageForNode(node, node) — which \
             dataStageParam_ entry sizes this allocation"
        )
    }

    fn is_sole_partial_reduction_input(&self, _lds: LdsIdx) -> bool {
        todo!(
            "v1::StageSizes::is_sole_partial_reduction_input: wants a sweep of currDsc->computeOps_ \
             for a GENERIC_PARTIAL_REDUCTION opFuncName whose single inputLabeledDs is this lds"
        )
    }
}

impl v1::OffsetSizes for OffsetSizes {
    fn lds_alloc(&self, _lds: LdsIdx, _storage: SenComponent) -> Option<AllocId> {
        todo!(
            "v1::OffsetSizes::lds_alloc: wants \
             labeledDs_.at(lds).memOrg_.at(storage).allocateNode_ on the super-DSC's DSC"
        )
    }

    fn const_alloc(&self, _constant: ConstIdx, _storage: SenComponent) -> Option<AllocId> {
        todo!(
            "v1::OffsetSizes::const_alloc: wants \
             constantInfo_.at(constant).allocations_.at(storage) on the super-DSC's DSC"
        )
    }

    fn address_scale(&self, _unit: GenericComp, _storage: SenComponent) -> Option<NonZeroU64> {
        todo!(
            "v1::OffsetSizes::address_scale: wants \
             dscGlobal.sysDef.addressGranularityScalePerUnit.at({{generic, storage}}) — a \
             system-definition table handed to the scheduler's constructor, not a super-DSC fact"
        )
    }

    fn stage_padding_dims(&self, _stage: DatastageId) -> Vec<(PrimaryDim, v1::PaddingSizes)> {
        todo!(
            "v1::OffsetSizes::stage_padding_dims: wants every entry of \
             dataStageParam_.at(stage).ss_.paddingSizes_ on the LIVE data stage"
        )
    }

    fn has_symbolic_dim(&self, _stage: DatastageId, _dim: PrimaryDim) -> bool {
        todo!(
            "v1::OffsetSizes::has_symbolic_dim: wants \
             dataStageParam_.at(stage).ss_.symbolicDimInfo_.count(dim) on the LIVE data stage"
        )
    }

    fn pe_sfp_split_dims(&self, _stage: DatastageId) -> Vec<PrimaryDim> {
        todo!(
            "v1::OffsetSizes::pe_sfp_split_dims: wants \
             dataStageParam_.at(stage).ss_.peSfpSplit_ on the LIVE data stage"
        )
    }

    fn block_transfer_size(
        &self,
        _node: NodeId,
        _unit: SenComponent,
        _corelet: Corelet,
        _dim: PrimaryDim,
    ) -> Elements {
        todo!(
            "v1::OffsetSizes::block_transfer_size: wants \
             getBlockTransferSizePerDim(transfer, unit, corelet)[dim] — a DesignSpaceConfig \
             accessor over that TRANSFER node of the DSC's scheduleTree_"
        )
    }

    fn temporal_stride(
        &self,
        _node: NodeId,
        _alloc: AllocId,
        _at: LoopId,
        _dim: PrimaryDim,
    ) -> Option<v1::LoopEleOffset> {
        todo!(
            "v1::OffsetSizes::temporal_stride: wants \
             loopDistributionParamInfo.at(node).at(alloc).at(loop).at(dim)\
             .temporalStridePostDistribution — the metadata's fold-distribution table"
        )
    }
}

impl v1::ScheduleWalk for OffsetNodes {
    fn loops_under(&self, _from: BlockId) -> Vec<LoopId> {
        todo!(
            "v1::ScheduleWalk::loops_under: wants the DSC's scheduleTree_ traverseTreeDFS(from, \
             {{LOOP}}) walk, which this surface is not wired to"
        )
    }

    fn nodes_of_kind(&self, _kind: NodeKind) -> Vec<NodeId> {
        todo!(
            "v1::ScheduleWalk::nodes_of_kind: wants the DSC's scheduleTree_ \
             traverseTreeDFS(nullptr, {{kind}}) walk, which this surface is not wired to"
        )
    }

    fn allocates(&self) -> Vec<AllocId> {
        todo!(
            "v1::ScheduleWalk::allocates: wants the DSC's scheduleTree_ \
             traverseTreeDFSMutable(nullptr, {{ALLOCATE}}) walk, which this surface is not wired to"
        )
    }

    fn nodes_of_kind_under(
        &self,
        _from: Option<NodeId>,
        _kind: NodeKind,
        _unit: SenComponent,
    ) -> Vec<NodeId> {
        todo!(
            "v1::ScheduleWalk::nodes_of_kind_under: wants the DSC's scheduleTree_ \
             traverseTreeDFS(from, {{kind}}, unit) walk, which this surface is not wired to"
        )
    }

    fn prev_of_alloc(&self, _alloc: AllocId) -> Option<NodeId> {
        todo!(
            "v1::ScheduleWalk::prev_of_alloc: wants allocNode->getPrev() — the block, loop or region \
             CONTAINING that ALLOCATE in the DSC's scheduleTree_"
        )
    }

    fn transfer(&self, _node: NodeId) -> Option<TransferNode> {
        todo!(
            "v1::ScheduleWalk::transfer: wants that node's src_ and dstVias_ off the DSC's \
             scheduleTree_, which this surface is not wired to"
        )
    }

    fn as_loop(&self, _node: NodeId) -> Option<LoopId> {
        todo!(
            "v1::ScheduleWalk::as_loop: wants nodeType_ == LOOP on that node of the DSC's \
             scheduleTree_"
        )
    }

    fn prev(&self, _node: NodeId) -> Option<NodeId> {
        todo!(
            "v1::ScheduleWalk::prev: wants getPrev() — the PARENT block up the DSC's scheduleTree_, \
             not the preceding sibling"
        )
    }

    fn owner_loop(&self, _node: NodeId) -> Option<LoopId> {
        todo!(
            "v1::ScheduleWalk::owner_loop: wants getOwnerLoop() — the innermost LOOP enclosing that \
             node in the DSC's scheduleTree_"
        )
    }

    fn loop_dims(&self, _at: LoopId) -> Vec<(PrimaryDim, MetaDimKind)> {
        todo!(
            "v1::ScheduleWalk::loop_dims: wants LoopNode::dims_ on that loop of the DSC's \
             scheduleTree_"
        )
    }
}

impl v1::ScheduleNodes for OffsetNodes {
    fn nodes(&self) -> Vec<NodeId> {
        todo!(
            "v1::ScheduleNodes::nodes: wants the DSC's scheduleTree_ traverseTreeDFSMutable() walk \
             with no filter, which this surface is not wired to"
        )
    }

    fn kind(&self, _node: NodeId) -> Option<NodeKind> {
        todo!("v1::ScheduleNodes::kind: wants nodeType_ on that node of the DSC's scheduleTree_")
    }

    fn is_parametric(&self, _at: LoopId) -> bool {
        todo!(
            "v1::ScheduleNodes::is_parametric: wants isParametricLoop() on that loop of the DSC's \
             scheduleTree_"
        )
    }

    fn loop_stages(&self, _at: LoopId) -> v1::LoopStages {
        todo!(
            "v1::ScheduleNodes::loop_stages: wants numId_ and denId_ on that loop of the DSC's \
             scheduleTree_"
        )
    }

    fn parametric_stride(&self, _at: LoopId) -> v1::LoopEleOffset {
        todo!(
            "v1::ScheduleNodes::parametric_stride: wants parametricStride(dsc) on that loop of the \
             DSC's scheduleTree_"
        )
    }

    fn parametric_iter_count(
        &self,
        _at: LoopId,
        _corelet: Corelet,
        _unit: SenComponent,
    ) -> v1::IterCount {
        todo!(
            "v1::ScheduleNodes::parametric_iter_count: wants parametricIterCount(dsc, corelet, \
             unit) on that loop of the DSC's scheduleTree_"
        )
    }

    fn is_relevant(&self, _node: NodeId, _unit: SenComponent) -> bool {
        todo!(
            "v1::ScheduleNodes::is_relevant: wants isNodeRelevant(unit) on that node of the DSC's \
             scheduleTree_"
        )
    }

    fn next_view_len(&self, _node: NodeId, _unit: SenComponent) -> usize {
        todo!(
            "v1::ScheduleNodes::next_view_len: wants getNextView(unit).size() on that node of the \
             DSC's scheduleTree_"
        )
    }

    fn alloc_owner_loop(&self, _alloc: AllocId) -> Option<LoopId> {
        todo!(
            "v1::ScheduleNodes::alloc_owner_loop: wants allocNode->getOwnerLoop() — the innermost \
             LOOP that ALLOCATE sits under in the DSC's scheduleTree_"
        )
    }

    fn transfer_has_padding(&self, _node: NodeId) -> bool {
        todo!(
            "v1::ScheduleNodes::transfer_has_padding: wants \
             TransferNode::paddingInfo_.isEmpty() == false on that node of the DSC's scheduleTree_"
        )
    }

    fn compute(&self, _node: NodeId) -> Option<ComputeNode> {
        todo!(
            "v1::ScheduleNodes::compute: wants type_, exUnit_, inputs_ and outputs_ each zipped \
             with its offsets vector, off that COMPUTE node of the DSC's scheduleTree_"
        )
    }

    fn repetition_with_offset_outputs(&self, _node: NodeId) -> usize {
        todo!(
            "v1::ScheduleNodes::repetition_with_offset_outputs: wants \
             repetitionWithOffset_.forOutputs_.size() on that node of the DSC's scheduleTree_"
        )
    }
}

impl MemOrgs for OffsetFacts {
    type Org = TreeOrg;

    fn mem_org(&self, _dsc: DscIdx, _lds: LdsIdx) -> Option<&Self::Org> {
        todo!(
            "MemOrgs::mem_org: wants dscs_.at(dsc).labeledDs_.at(lds).memOrg_ — the same \
             organisation `super::tree::Org` holds, reached through a carrier that may not alias the \
             &mut SuperDsc"
        )
    }
}

impl L3OffsetFacts for OffsetFacts {
    type Stage = OffsetStage;

    fn page_sizes(&self, _alloc: AllocId) -> BTreeMap<PrimaryDim, NonZeroU64> {
        todo!(
            "L3OffsetFacts::page_sizes: wants allocNode->getPageSize(), which dispatches on \
             indirectAllocType_ and reads relatedIndirectAccessAlloc_ — fields only L3AllocateNode \
             carries, and a seam rather than a derivation"
        )
    }

    fn ibr_sizes_no_rounding(
        &self,
        _alloc: AllocId,
        _lds: LdsIdx,
        _storage: SenComponent,
    ) -> Option<BTreeMap<PrimaryDim, Elements>> {
        todo!(
            "L3OffsetFacts::ibr_sizes_no_rounding: wants \
             getBufferCapacityForNodePerDim(alloc, lds, storage, -1, -1, noRounding=true) — a \
             DesignSpaceConfig accessor on the super-DSC"
        )
    }

    fn word_length(&self, _lds: LdsIdx) -> Option<WordLength> {
        todo!(
            "L3OffsetFacts::word_length: wants labeledDs_.at(lds).wordLength on the super-DSC's DSC"
        )
    }

    fn dim_symbols(&self, _dim: PrimaryDim) -> Vec<VariableSymbol> {
        todo!(
            "L3OffsetFacts::dim_symbols: wants dimToSymbolMapping_.at(dim), where the EMPTY vector \
             is count(dim) == 0 and a skip is not a refusal"
        )
    }

    fn size_stage(&self, _alloc: AllocId) -> Option<&Self::Stage> {
        todo!(
            "L3OffsetFacts::size_stage: wants getSizeDataStageForNode(alloc, alloc).ss_ — the LIVE \
             dataStageParam_ entry that sizes this allocation"
        )
    }

    fn chunk_stage(&self) -> &Self::Stage {
        todo!(
            "L3OffsetFacts::chunk_stage: wants dataStageParam_.at(dataStageChunkIdx).ss_ — \
             mandatory, and rewritten by entries 380/351 while the stage runs, so no snapshot serves"
        )
    }
}

impl DimStage for OffsetStage {
    fn corelet_dim_val(
        &self,
        _dim: PrimaryDim,
        _comp: SenComponent,
        _corelet: Corelet,
        _padded: &PaddingForm,
    ) -> Option<Extent> {
        todo!(
            "DimStage::corelet_dim_val: wants primaryDimToVal_st(dim, comp, -1, corelet, padded) on \
             the LIVE dataStageParam_ entry"
        )
    }

    fn is_corelet_split(&self, _dim: PrimaryDim) -> bool {
        todo!(
            "DimStage::is_corelet_split: wants coreletSplit_.count(dim) on the LIVE \
             dataStageParam_ entry"
        )
    }

    fn corelet_split(&self, _dim: PrimaryDim, _corelet: Corelet) -> Option<Extent> {
        todo!(
            "DimStage::corelet_split: wants coreletSplit_.at(dim).at(corelet) — that corelet's RAW \
             unpadded share on the LIVE dataStageParam_ entry"
        )
    }

    fn pad_stride(&self, _dim: PrimaryDim) -> Option<Stride> {
        todo!(
            "DimStage::pad_stride: wants paddingSizes_.at(dim).stride_ on the LIVE dataStageParam_ \
             entry"
        )
    }
}
