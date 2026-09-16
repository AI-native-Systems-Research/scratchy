// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE READ CARRIER — `F` in [`crate::schedule::l3::dl_ops::run`], which is ten traits:
//! `MemOrgs`, `TransferNodes`, `ScheduleTrees`, `DscStages`, `DscLoopStages`, `DscTrees`,
//! `SysFlopsPerByte`, `ComputeOps`, `AllocationReads` and `DscOffsetFacts`.
//!
//! ⭐ IT HOLDS THE SAME `&'s DscState` [`super::Env`] DOES, which is the whole point: entries
//! 289/295/332 read the tree through `reads` while 290/353/368 write it through `env`, and they name
//! ONE tree.
//!
//! ⭐⭐ AND THAT IS WHY [`AllocationReads`] IS HERE AS WELL AS ON `Env`: entry 222's PROBE only reads
//! `memOrg_.at(storage).allocateNode_`, and the paged chain (entries 368/354/336/294) probes with
//! `env` exclusively borrowed for the tree it is rewriting — so the probe reads the allocate nodes
//! through THIS carrier, which is the same cell `Env`'s write seam places into.
//!
//! ⛔⛔ WHAT THIS CARRIER CANNOT SEE, AND WHY. `run` takes `sdsc: &mut SuperDsc` AND
//! `inputs: &L3RunInputs<'_, F, P>`, so `F` may not alias the super-DSC — every fact that lives in
//! `dscs_.at(i).dataStageParam_` is out of reach here, and stage 2a REWRITES the chunk stage
//! (entries 380/351), so a snapshot taken before the call would be stale by the time entry 292 reads
//! it. That is [`DscStages`] and half of [`DscOffsetFacts`], and it is review 382's cross-entry work
//! over entries 050/219/220/222/292/333 — not a fact scratchy fails to write.

use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::schedule::ddc::fold::Stride;
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation_util::PaddingForm;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::LdsIdx;
use crate::schedule::l3::dl_ops::{
    AllocationReads, AllocationView, DscLoopStages, DscOffsetFacts, DscStages, FlopPerByte,
    OpFuncDataFormat, SysFlopsPerByte,
};
use crate::schedule::l3::dsc::{
    DimStage, DscIdx, L3Transfer, MemOrgs, PlacedAllocation, ScheduleTrees, TransferNodes,
};
use crate::units::Corelet;

use super::offsets::{OffsetFacts, OffsetNodes, OffsetSizes};
use super::state::{DscState, DscTree};
use super::tree::{Org, TreeData};

/// ⭐ THE `F` CARRIER — one shared reference to the state, plus the ONE construction argument the
/// scheduler is handed rather than reading off the super-DSC.
#[derive(Debug)]
pub struct Reads<'s> {
    state: &'s DscState,
    /// `computeOp_`, one `opFuncName` per entry.
    ///
    /// ⛔⛔ A CONSTRUCTION ARGUMENT AND NOT A DERIVATION. `computeOp_` is a `DesignSpaceConfig`
    /// field, and [`crate::schedule::l3::dsc::DesignSpaceConfig`] models only its
    /// `indirectAccessIndexLabeledDs` — which is exactly why [`v1::ComputeOps`] is a SEPARATE trait
    /// on `F` rather than a method on the DSC. A caller that has the op func (scratchy's SuperDSC
    /// states `sfp_fma16` for `rmsq_o728`) passes it here; [`OpFuncs::new(None, ..)`](v1::OpFuncs)
    /// is *"a `computeOp_` entry whose `opFuncName` is unset"*, on which the ported min-param units
    /// refuse exactly as the reference's `.at(0)` throws.
    ops: v1::OpFuncs,
}

impl<'s> Reads<'s> {
    /// The carrier over one seeded state and the compute ops the caller holds.
    #[must_use]
    pub const fn new(state: &'s DscState, ops: v1::OpFuncs) -> Self {
        Self { state, ops }
    }

    /// That DSC's tree, [`None`] for a `dscs_` position the state holds none for.
    fn dsc(&self, at: DscIdx) -> Option<&'s DscTree> {
        self.state.dsc(at)
    }
}

impl MemOrgs for Reads<'_> {
    type Org = Org;

    fn mem_org(&self, dsc: DscIdx, lds: LdsIdx) -> Option<&Self::Org> {
        self.dsc(dsc)?.org(lds)
    }
}

impl TransferNodes for Reads<'_> {
    fn transfers(&self, dsc: DscIdx) -> Vec<L3Transfer> {
        self.dsc(dsc)
            .map(|held| held.with(TreeData::transfers))
            .unwrap_or_default()
    }
}

impl AllocationReads for Reads<'_> {
    fn allocation(
        &self,
        dsc: DscIdx,
        lds: LdsIdx,
        storage: sys_arch_spec::arch_enums::SenComponent,
    ) -> Option<AllocationView> {
        Some(AllocationView::of(self.dsc(dsc)?.org(lds)?.placed(storage)?))
    }
}

impl ScheduleTrees for Reads<'_> {
    fn allocations(&self, dsc: DscIdx) -> Vec<PlacedAllocation> {
        self.dsc(dsc)
            .map(|held| held.with_placed(|placed| held.with(|tree| tree.allocations(placed))))
            .unwrap_or_default()
    }
}

impl crate::schedule::l3::dl_ops::DscTrees for Reads<'_> {
    type Tree = DscTree;

    fn tree(&self, dsc: DscIdx) -> Option<&Self::Tree> {
        self.dsc(dsc)
    }

    fn root(&self, dsc: DscIdx) -> Option<crate::schedule::ddc::fold::NodeId> {
        self.dsc(dsc)?.with(TreeData::head)
    }

    fn lx_below_block(&self, dsc: DscIdx) -> Option<crate::schedule::ddc::fold::NodeId> {
        self.dsc(dsc)?.with(TreeData::lx_below_block)
    }

    fn allocation(
        &self,
        dsc: DscIdx,
        lds: LdsIdx,
        storage: sys_arch_spec::arch_enums::SenComponent,
    ) -> Option<crate::schedule::ddc::fold::NodeId> {
        self.dsc(dsc)?.org(lds)?.node(storage)
    }

    fn transfer_src_lds(
        &self,
        dsc: DscIdx,
        node: crate::schedule::ddc::fold::NodeId,
    ) -> Option<LdsIdx> {
        self.dsc(dsc)?
            .with(|tree| tree.transfer(node))?
            .src
            .data
            .my_lds_idx
    }

    fn transfer_dst_is_lds(&self, dsc: DscIdx, node: crate::schedule::ddc::fold::NodeId) -> bool {
        self.dsc(dsc)
            .and_then(|held| held.with(|tree| tree.transfer(node)))
            .is_some_and(|held| held.dsts.first().data.my_lds_idx.is_some())
    }
}

impl DscLoopStages for Reads<'_> {
    type Stages = DscTree;

    fn loop_stages(&self, dsc: DscIdx) -> Option<&Self::Stages> {
        self.dsc(dsc)
    }
}

impl v1::ComputeOps for Reads<'_> {
    fn op_funcs(&self) -> v1::OpFuncs {
        self.ops.clone()
    }

    /// ⭐ UNREACHABLE THROUGH [`crate::schedule::l3::dl_ops::run`], WHICH IS WHY THE BODY IS EMPTY:
    /// `F` is only ever held as `&'a F`, so nothing in stage 2a can call a `&mut self` method on it.
    /// `computeOp_.at(0).opFuncName` is assigned by the DDC stage, not this one.
    fn set_first_op_func(&mut self, _op_func: sys_arch_spec::arch_enums::OpFunc) {}
}

impl SysFlopsPerByte for Reads<'_> {
    /// ⛔ NOT A SUPER-DSC FACT AT ALL: `sysFlopsPerByte` is a `dscGlobal.sysDef` table — a SYSTEM
    /// DEFINITION handed to the scheduler's constructor, per data format. It is read only under
    /// `CHUNK_EXPLORE`, by `find_best_params_for_arithmetic_intensity`
    /// ([`crate::schedule::l3::dl_ops::run`] reaches it at `dl_ops.rs:20947`), and every
    /// [`super::run_l3`] call site in this crate passes `CHUNK_EXPLORE = false` — so this stop is
    /// unreachable today by a const generic that REMOVES the op, not merely unexercised.
    ///
    /// ⛔ AND IT IS NOT A LOOKUP BUT AN ARITHMETIC IDENTITY THIS FILE CANNOT SPELL YET —
    /// `sysFlopsPerByte[fmt] = numCores * numCoreletsPerCore * numPTRows * numPTCols * numSimdPerPT *
    /// numSubSimdPerPT.at(fmt) * coreFreq * 2 / hbmBw` (`sys-arch-spec/sysdef.cpp:297-307`,
    /// *"1 MAC = 2 FLOPs"*). Five of those eight are already [`crate::arch::Arch`] consts
    /// ([`crate::arch::Arch::CORES`], [`crate::arch::Arch::CORELETS_PER_CORE`],
    /// [`crate::arch::Arch::PT_ROWS`], [`crate::arch::Arch::PT_COLS`],
    /// [`crate::arch::Arch::SIMD_PER_PT`]); the THREE that are missing are `numSubSimdPerPT` per
    /// format (`:231-234`), `coreFreq = 1.5` (`:222`) and `hbmBw` (`:242-244`, which is `32` for a
    /// one-core system and otherwise `170`/`1024` by generation). ⭐ SO THE FACT IS DECLARED DATA AND
    /// NOT A CONSTRUCTION ARGUMENT — three consts on `crate::arch::Arch`, and this method becomes the
    /// product. That edit is `arch.rs`', which this file does not own.
    fn sys_flops_per_byte(&self, _format: OpFuncDataFormat) -> FlopPerByte {
        todo!(
            "SysFlopsPerByte::sys_flops_per_byte: wants dscGlobal.sysDef.sysFlopsPerByte.at(format) \
             = cores * corelets * ptRows * ptCols * simdPerPt * subSimdPerPt(format) * coreFreq * 2 \
             / hbmBw (sys-arch-spec/sysdef.cpp:297-307) — three of those (numSubSimdPerPT, coreFreq, \
             hbmBw) are not yet crate::arch::Arch consts"
        )
    }
}

impl DscStages for Reads<'_> {
    type Stage = SeveredStage;

    /// ⛔ REFUSES, AND WHAT IT WANTS: `dscs_.at(dsc).dataStageParam_.at(stage)`. `run` holds the
    /// super-DSC as `&mut` and this carrier is `&'a F`, so it cannot alias that field — AND a
    /// snapshot would not do: entries 380 and 351 REWRITE the chunk stage before entry 292 reads it
    /// here, so the answer must come from the live super-DSC. Unifying the two carriers is review
    /// 382's own cross-entry work over entries 050/219/220/222/292/333.
    fn dim_stage(&self, _dsc: DscIdx, _stage: DatastageId) -> Option<&Self::Stage> {
        self.state.refuse(
            "DscStages::dim_stage: wants dscs_.at(dsc).dataStageParam_.at(stage) LIVE — a snapshot \
             is stale once entries 380/351 rewrite the chunk stage",
        )
    }
}

/// ONE `DataStructDims` THIS CARRIER CANNOT REACH — [`DscStages::dim_stage`] refuses before any
/// method here can be asked.
///
/// ⭐⭐ UNINHABITED, AND THAT IS THE SEVERING AS A TYPE RATHER THAN AS FOUR STOPS. `DscStages::Stage`
/// is bounded by `DimStage + ?Sized` and NOTHING MORE — an associated type has to be NAMED before
/// the carrier compiles, not INHABITED — so an empty enum satisfies it while making *"no method here
/// is ever called"* a fact the compiler checks instead of one a reader has to confirm by going and
/// reading [`DscStages::dim_stage`]. Four `todo!`s that could only be reached by minting a value of
/// this type are four stops that cannot exist at all.
///
/// ⛔ THE RECORD OF WHAT IS MISSING STAYS, on each method below: an uninhabited body is
/// `match *self {}`, and the doc above it still names the one field that method wants. What is gone
/// is the stop, not the note.
#[derive(Debug, Clone, Copy)]
pub enum SeveredStage {}

impl DimStage for SeveredStage {
    /// ⛔ Wants `primaryDimToVal_st(dim, comp, -1, corelet, padded)` on the live data stage.
    fn corelet_dim_val(
        &self,
        _dim: PrimaryDim,
        _comp: sys_arch_spec::arch_enums::SenComponent,
        _corelet: Corelet,
        _padded: &PaddingForm,
    ) -> Option<Extent> {
        match *self {}
    }

    /// ⛔ Wants `coreletSplit_.count(dim)` on the live data stage.
    fn is_corelet_split(&self, _dim: PrimaryDim) -> bool {
        match *self {}
    }

    /// ⛔ Wants `coreletSplit_.at(dim).at(corelet)` on the live data stage.
    fn corelet_split(&self, _dim: PrimaryDim, _corelet: Corelet) -> Option<Extent> {
        match *self {}
    }

    /// ⛔ Wants `paddingSizes_.at(dim).stride_` on the live data stage.
    fn pad_stride(&self, _dim: PrimaryDim) -> Option<Stride> {
        match *self {}
    }
}

impl DscOffsetFacts for Reads<'_> {
    type Sizes = OffsetSizes;
    type Nodes = OffsetNodes;
    type Facts = OffsetFacts;

    /// ⛔ REFUSES: [`v1::StageSizes`] and [`v1::OffsetSizes`] are the datastage extents and the
    /// address-granularity table, both severed with [`DscStages::dim_stage`].
    fn offset_sizes(&self, _dsc: DscIdx) -> Option<&Self::Sizes> {
        self.state.refuse(
            "DscOffsetFacts::offset_sizes: wants v1::StageSizes + v1::OffsetSizes — the live \
             dataStageParam_ extents and dscGlobal.sysDef.addressGranularityScalePerUnit",
        )
    }

    /// ⛔ REFUSES: [`v1::ScheduleNodes`] is a WIDER walk than this state answers — parametric
    /// strides, per-unit relevance and the compute node's operands, none of which the L3 tree arena
    /// holds.
    fn offset_nodes(&self, _dsc: DscIdx) -> Option<&Self::Nodes> {
        self.state.refuse(
            "DscOffsetFacts::offset_nodes: wants v1::ScheduleNodes (parametricStride, \
             isNodeRelevant, getNextView, the ComputeNode operands) over the DSC's own tree",
        )
    }

    /// ⛔ REFUSES: [`crate::schedule::l3::dl_ops::L3OffsetFacts`] is `getPageSize`,
    /// `getBufferCapacityForNodePerDim`, `wordLength`, `dimToSymbolMapping_` and the two size
    /// stages — the `dsc2` seams plus the live data stages.
    fn offset_facts(&self, _dsc: DscIdx) -> Option<&Self::Facts> {
        self.state.refuse(
            "DscOffsetFacts::offset_facts: wants L3OffsetFacts — getPageSize, \
             getBufferCapacityForNodePerDim, labeledDs_.wordLength, dimToSymbolMapping_ and the live \
             size/chunk data stages",
        )
    }
}
