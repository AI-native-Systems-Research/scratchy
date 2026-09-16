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
//! ⭐⭐ AND THE SUPER-DSC IS NOT A FIELD OF IT BUT AN **ARGUMENT** TO FOUR OF ITS METHODS. `run` takes
//! `sdsc: &mut SuperDsc` AND `inputs: &L3RunInputs<'_, F, P>`, so `F` may not alias the super-DSC and
//! every fact in `dscs_.at(i).dataStageParam_` was out of reach while [`DscStages::dim_stage`] and
//! [`DscOffsetFacts`]' three getters answered `Option<&Self::Sizes>` — a borrow OF THE CARRIER, which
//! is the one thing this carrier cannot own. ⛔ AND A SNAPSHOT WAS NEVER THE ANSWER EITHER: stage 2a
//! REWRITES the chunk stage (entries 380/351) and GROWS the schedule tree (290/353/368), so a copy
//! taken at construction is stale by the time entry 292 or 333 reads it. Both now take the super-DSC
//! the caller is already holding — `run` builds `L3OffsetInputs` from a shared reborrow of its own
//! `&mut` (`l3/dl_ops.rs`), and [`crate::schedule::l3::dl_ops::fill_allocation_start_addr_and_offset`]
//! takes `sdsc: &SuperDsc` outright — and hand back a projection BY VALUE, so the DSC borrow lives at
//! the call site and never in `F`.

use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::LdsIdx;
use crate::schedule::l3::dl_ops::{
    AllocationReads, AllocationView, DscLoopStages, DscOffsetFacts, DscStages, FlopPerByte,
    OpFuncDataFormat, SysFlopsPerByte,
};
use crate::schedule::l3::dsc::{
    DscIdx, L3Transfer, MemOrgs, PlacedAllocation, ScheduleTrees, SuperDsc, TransferNodes,
};

use super::offsets::{OffsetFactsOf, OffsetNodesOf, OffsetSizesOf, OffsetStageOf};
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
    /// `computeOp_` IN FULL — the five fields [`v1::DscComputeOp`] projects, which is what
    /// [`super::offsets::OffsetSizesOf`] needs and what [`Self::ops`] cannot supply:
    /// `v1::StageSizes::is_sole_partial_reduction_input` sweeps `inputLabeledDs`, and [`v1::OpFuncs`]
    /// carries `opFuncName` per entry and nothing else.
    ///
    /// ⛔⛔ EMPTY IS *"NOT HANDED TO STAGE 2A"* AND NOT A DSC WITH NO COMPUTE, which is why
    /// [`DscOffsetFacts::offset_sizes`] REFUSES on it rather than sweeping it: [`v1::OpFuncs`]' own
    /// non-emptiness states that `computeOp_.at(0)` throws on an op-less DSC, so an empty list is a
    /// state the reference cannot be in and a `false` swept out of it would be invented. This is the
    /// same min-param refusal [`Self::ops`] documents, on the one argument
    /// [`super::run_l3`] does not yet take.
    computes: Vec<v1::DscComputeOp>,
}

impl<'s> Reads<'s> {
    /// The carrier over one seeded state and the compute ops the caller holds.
    #[must_use]
    pub const fn new(state: &'s DscState, ops: v1::OpFuncs) -> Self {
        Self {
            state,
            ops,
            computes: Vec::new(),
        }
    }

    /// THE SAME CARRIER WITH `computeOp_` IN FULL — the one construction argument entry 333's size
    /// surface needs beyond [`v1::OpFuncs`], for a caller that holds it (the `rmsq_o728` fixture
    /// states one `SFP_FMA16` op over `lds0`/`lds1`).
    #[must_use]
    pub fn with_compute_ops(mut self, computes: Vec<v1::DscComputeOp>) -> Self {
        self.computes = computes;
        self
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
    type Stage<'x>
        = OffsetStageOf<'x>
    where
        Self: 'x;

    /// `dscs_.at(dsc).dataStageParam_.at(stage).ss_`, READ THROUGH THE CALLER'S OWN BORROW — so
    /// entries 380 and 351 rewriting the chunk stage above are seen here, which a snapshot taken when
    /// this carrier was built could not be.
    ///
    /// ⛔ [`None`] IS THAT `.at()`'s THROW AND NOTHING ELSE: a `dscs_` position the super-DSC holds no
    /// DSC for, and a `dataStageParam_` index that DSC states no stage under — which
    /// [`crate::schedule::l3::dsc::DataStages::at`] answers for its own
    /// [`crate::schedule::l3::dsc::EmptyStage`] too, an entry the reference holds but states no extent
    /// in. ⭐ THE TWO IDS ENTRY 292 ASKS FOR ARE FIELDS of that type, so neither of them can be it.
    fn dim_stage<'x>(
        &'x self,
        sdsc: &'x SuperDsc,
        dsc: DscIdx,
        stage: DatastageId,
    ) -> Option<Self::Stage<'x>> {
        Some(OffsetStageOf::new(
            sdsc.dscs().at(dsc)?.data_stages.at(stage)?.ss.dims.dims(),
        ))
    }
}

impl DscOffsetFacts for Reads<'_> {
    type Sizes<'x>
        = OffsetSizesOf<'x>
    where
        Self: 'x;
    type Nodes<'x>
        = OffsetNodesOf<'x>
    where
        Self: 'x;
    type Facts<'x>
        = OffsetFactsOf<'x>
    where
        Self: 'x;

    /// [`super::offsets::OffsetSizesOf`] over this DSC's live `dataStageParam_`/`labeledDs_` and the
    /// tree the growers filed every allocation in.
    ///
    /// ⛔ REFUSES ON ONE MISSING CONSTRUCTION ARGUMENT AND NOTHING ELSE — `computeOp_` in the
    /// [`v1::DscComputeOp`] shape, which [`Self::with_compute_ops`] takes and [`super::run_l3`] does
    /// not yet pass. ⛔ NOT A SUBSTITUTED EMPTY SWEEP: see this carrier's own `computes` field.
    fn offset_sizes<'x>(&'x self, sdsc: &'x SuperDsc, dsc: DscIdx) -> Option<Self::Sizes<'x>> {
        if self.computes.is_empty() {
            return self.state.refuse(
                "DscOffsetFacts::offset_sizes: wants computeOp_ as v1::DscComputeOp (opFuncName + \
                 exUnit + dataFormat_ + inputLabeledDs + outputLabeledDs), which run_l3 does not \
                 take — v1::OpFuncs carries only opFuncName, so the inputLabeledDs sweep of \
                 is_sole_partial_reduction_input (ddc/ddcv1.cpp:1915-1921) has nothing to read",
            );
        }
        Some(OffsetSizesOf::new(
            sdsc.dscs().at(dsc)?,
            self.dsc(dsc)?,
            &self.computes,
        ))
    }

    /// [`super::offsets::OffsetNodesOf`] over `dscs_.at(dsc).scheduleTree_` — ⭐ THE STATE'S OWN TREE
    /// AND NOT THE SUPER-DSC'S, which is where entries 290/353/368 minted every node this walk visits;
    /// the super-DSC carries no `scheduleTree_` of its own for this stage to read.
    fn offset_nodes<'x>(&'x self, _sdsc: &'x SuperDsc, dsc: DscIdx) -> Option<Self::Nodes<'x>> {
        Some(OffsetNodesOf::new(self.dsc(dsc)?))
    }

    /// [`super::offsets::OffsetFactsOf`] over every DSC's `memOrg_`s, this DSC's `labeledDs_` and
    /// `dimToSymbolMapping_`, its tree's ALLOCATE nodes, and its LIVE chunk data stage.
    fn offset_facts<'x>(&'x self, sdsc: &'x SuperDsc, dsc: DscIdx) -> Option<Self::Facts<'x>> {
        OffsetFactsOf::new(self.state, dsc, sdsc.dscs().at(dsc)?)
    }
}
