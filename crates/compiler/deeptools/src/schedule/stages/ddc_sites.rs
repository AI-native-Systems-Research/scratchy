// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ WHERE ONE DSC'S CARRIERS COME FROM — [`v1::Dsc2Sites`], the `P` [`v1::run_v1`] is handed, plus
//! the two carriers that are types of their own: `currDsc->dataStageParam_` and the DDL match site.
//!
//! # ⭐ `Dsc2Stages` — THE SHARED DATASTAGE MAP, AND THE ONE METHOD THAT CANNOT SHARE IT
//!
//! Twenty-six of [`v1::ExploreStages`]'s twenty-eight methods read or write the map in
//! [`Dsc2State`] — the SAME map [`super::Dsc2Reads`]'s [`v1::StageSizes`] reads, which is what makes
//! a datastage entry 338 mints the one entry 128 lays out.
//!
//! ⛔⛔ [`v1::Dsc2Stages::as_util`] IS THE EXCEPTION AND IT IS A `todo!`. It returns
//! `&mut UtilDataStages<Self::Dims>` — a REAL exclusive borrow, which no [`RefCell`] can hand out —
//! and its own doc demands it be *"THE MAP THIS TYPE'S OWN `ExploreStages` READS"*, because a freshly
//! built map would drop every effect entries 302/303/338 have and still compile. Those two
//! requirements are incompatible for a carrier whose map must ALSO be visible to the `dsc` half that
//! `Dsc2Carriers` borrows disjointly from it. So the method NAMES the conflict rather than resolving
//! it with a copy.
//!
//! ⭐ ITS THREE CALLERS ALL SIT AFTER THE DDL STEP (`ddc/v1.rs:6503`, `:6549`, `:6580`) and are all
//! `let _ = ..`, so this is not what stops the stage.
//!
//! # ⛔ `Dsc2Ddl` — TWENTY-FOUR METHODS, AND `l3::dsc` CARRIES SIX OF THEM
//!
//! The DDL match reads `dataFormat_`, `wordLength` and `dsName_` on every operand and WRITES the
//! first two (`MatchSite::set_lds_word_length`, `set_lds_format`) — the three fields
//! [`crate::schedule::l3::dsc::LabeledDs`] does not project. What it can answer is the work-slice
//! ring, the layout-order test and the stick dims.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickDims,
};
use crate::formats::DataFormat;
use crate::schedule::ddc::fold::{AllocId, ConstIdx, NodeId, PadType};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation as tr;
use crate::schedule::ddc::transformation_util as tu;
use crate::schedule::ddc::v1;
use crate::schedule::ddl::conversion as conv;
use crate::schedule::dsc2::{LdsIdx, WordLength};
use crate::schedule::l3::dl_ops::AddressFoldCoords;
use crate::schedule::l3::dsc::{DscIdx, SymbolicDimInfo, WkSlice};
use crate::units::{Core, Corelet};

use super::ddc_state::{self, Dsc2Dims, Dsc2State};
use super::ddc_store::Dsc2Store;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// `currDsc->dataStageParam_` THROUGH ITS THREE VOCABULARIES.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ ONE DSC'S `dataStageParam_` — the `S` carrier, over the SHARED map in [`Dsc2State`].
#[derive(Debug)]
pub struct Dsc2Stages<'s, 'l> {
    state: &'s Dsc2State<'l>,
    dsc: DscIdx,
}

impl<'s, 'l> Dsc2Stages<'s, 'l> {
    /// The map of that DSC.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>, dsc: DscIdx) -> Self {
        Self { state, dsc }
    }

    fn facts(&self) -> &'s super::ddc_state::Dsc2Facts {
        self.state
            .facts(self.dsc)
            .expect("a Dsc2Stages is only built for a DSC the state holds facts for")
    }

    /// One half, cloned for the length of one read.
    fn half(&self, at: v1::StageSite) -> Option<Dsc2Dims> {
        self.facts().with_stages(|stages| {
            let held = stages.0.get(&at.stage)?;
            Some(match at.half {
                v1::StageHalf::Ss => held.ss.dims.clone(),
                v1::StageHalf::El => held.el.dims.clone(),
            })
        })
    }

    /// One half, for the length of one write — a NO-OP for a datastage the map does not hold, which
    /// is the reference's `.at()` throw.
    fn edit(&self, at: v1::StageSite, write: impl FnOnce(&mut Dsc2Dims)) {
        self.facts().with_stages_mut(|stages| {
            if let Some(held) = stages.0.get_mut(&at.stage) {
                match at.half {
                    v1::StageHalf::Ss => write(&mut held.ss.dims),
                    v1::StageHalf::El => write(&mut held.el.dims),
                }
            }
        });
    }
}

impl v1::ExploreStages for Dsc2Stages<'_, '_> {
    type Dims = Dsc2Dims;

    /// `dataStageParam_`'s keys, in `std::map` order.
    fn stages(&self) -> Vec<DatastageId> {
        self.facts()
            .with_stages(|stages| stages.0.keys().copied().collect())
    }

    /// `dataStageParam_.at(stage).name()` as the closed three-way [`v1::StageLabel`] is.
    fn label(&self, stage: DatastageId) -> Option<v1::StageLabel> {
        let held = self.half(v1::StageSite::ss(stage))?;
        Some(match held.name.0.as_str() {
            "core" => v1::StageLabel::Core,
            "chunk" => v1::StageLabel::Chunk,
            _ => v1::StageLabel::Other,
        })
    }

    /// ⛔ `isExternalDs(stage)` — a `DesignSpaceConfig` predicate over the whole DSC (it compares the
    /// stage's sizes against the TENSOR's rather than a tile's), not a field of the stage.
    fn is_external(&self, _stage: DatastageId) -> bool {
        todo!(
            "v1::ExploreStages::is_external: wants isExternalDs(stage) — a DesignSpaceConfig \
             predicate comparing the stage's sizes against the whole tensor's, and it DECIDES \
             whether finalize_external_stage runs on that stage"
        )
    }

    /// The snapshot — ⭐ AN OWNED CLONE, which is what the trait's own doc asks for.
    fn dims(&self, at: v1::StageSite) -> Option<Self::Dims> {
        self.half(at)
    }

    /// `primaryDimToVal_st(dim, comp, row, cl, padded, density, symbolic)` — ⭐ THE PLAIN SLOT where
    /// the authority's own control flow is a field read; see [`Dsc2Dims::raw_slot`].
    fn extent(
        &self,
        at: v1::StageSite,
        dim: PrimaryDim,
        sample: v1::DimSample,
        padding: PadType,
        symbolic: v1::SymbolicRead,
    ) -> Extent {
        if sample.comp.is_none()
            && sample.row.is_none()
            && sample.corelet.is_none()
            && let Some(half) = self.half(at)
            && let Some(extent) = half.raw_slot(dim, padding, symbolic)
        {
            return extent;
        }
        todo!(
            "v1::ExploreStages::extent: wants primaryDimToVal_st (dsc/dims.cpp:653-704) to fold a \
             SAMPLED axis, or calculate_padded (:563-616) for {padding:?}, on stage {at:?} dim \
             {dim:?} — a hand-rolled fold is a fabricated extent"
        )
    }

    /// `primaryDimToValHandler_st(dim)` READ — ⭐ THE RAW PER-DIM SLOT, before any split is folded in,
    /// which is exactly what [`crate::schedule::l3::dsc::StageDims::extents`] holds.
    ///
    /// ⭐ THE UNSTATED SLOT IS THE REFERENCE'S `-1`, entry 207's recorded divergence.
    fn dim_value(&self, at: v1::StageSite, dim: PrimaryDim) -> Extent {
        self.half(at)
            .and_then(|half| half.dims.extents.get(&dim).copied())
            .unwrap_or(crate::schedule::l3::dl_ops::UNSTATED_EXTENT)
    }

    /// `primaryDimToValHandler_st(dim) = value`.
    fn set_dim_value(&mut self, at: v1::StageSite, dim: PrimaryDim, value: Extent) {
        self.edit(at, |half| {
            half.dims.extents.insert(dim, value);
        });
    }

    /// `coreletSplit_.at(dim)`.
    fn corelet_shares(&self, at: v1::StageSite, dim: PrimaryDim) -> Option<Vec<Extent>> {
        self.half(at)?.dims.corelet_split.get(&dim).cloned()
    }

    /// `coreletSplit_[dim] = shares`.
    fn set_corelet_shares(&mut self, at: v1::StageSite, dim: PrimaryDim, shares: Vec<Extent>) {
        self.edit(at, |half| {
            half.dims.corelet_split.insert(dim, shares);
        });
    }

    /// `rowSplit_.at(dim)`.
    fn row_shares(
        &self,
        at: v1::StageSite,
        dim: PrimaryDim,
    ) -> Option<BTreeMap<Corelet, Vec<Extent>>> {
        self.half(at)?.row_split.get(&dim).cloned()
    }

    /// `rowSplit_[dim] = shares`.
    fn set_row_shares(
        &mut self,
        at: v1::StageSite,
        dim: PrimaryDim,
        shares: BTreeMap<Corelet, Vec<Extent>>,
    ) {
        self.edit(at, |half| {
            half.row_split.insert(dim, shares);
        });
    }

    /// `peSfpSplit_.at(dim)`.
    fn pe_sfp_shares(
        &self,
        at: v1::StageSite,
        dim: PrimaryDim,
    ) -> Option<BTreeMap<Corelet, v1::PeSfpShares>> {
        self.half(at)?.pe_sfp_split.get(&dim).cloned()
    }

    /// `peSfpSplit_[dim] = shares`.
    fn set_pe_sfp_shares(
        &mut self,
        at: v1::StageSite,
        dim: PrimaryDim,
        shares: BTreeMap<Corelet, v1::PeSfpShares>,
    ) {
        self.edit(at, |half| {
            half.pe_sfp_split.insert(dim, shares);
        });
    }

    /// `paddingSizes_` — every padded dim with its sizes, in `std::map` order.
    fn padding_dims(
        &self,
        at: v1::StageSite,
    ) -> Vec<(PrimaryDim, crate::schedule::l3::dsc::DimPadding)> {
        self.half(at)
            .map(|half| half.dims.padding.into_iter().collect())
            .unwrap_or_default()
    }

    /// `paddingSizes_[dim] = padding` — the final value, as the trait's own doc says.
    fn set_padding(
        &mut self,
        at: v1::StageSite,
        dim: PrimaryDim,
        padding: crate::schedule::l3::dsc::DimPadding,
    ) {
        self.edit(at, |half| {
            half.dims.padding.insert(dim, padding);
        });
    }

    /// `symbolicDimInfo_` — every symbolic dim, in `std::map` order.
    fn symbolic_dims(&self, at: v1::StageSite) -> Vec<PrimaryDim> {
        self.half(at)
            .map(|half| half.dims.symbolic.info().keys().copied().collect())
            .unwrap_or_default()
    }

    /// `symbolicDimInfo_.at(dim)`.
    fn symbolic_info(&self, at: v1::StageSite, dim: PrimaryDim) -> Option<SymbolicDimInfo> {
        self.half(at)?.dims.symbolic.info().get(&dim).copied()
    }

    /// `symbolicDimInfo_.insert_or_assign(dim, info)`.
    fn set_symbolic_info(&mut self, at: v1::StageSite, dim: PrimaryDim, info: SymbolicDimInfo) {
        self.edit(at, |half| half.dims.symbolic.add_dim(dim, info));
    }

    /// ⛔ `makeDimSymbolic(refDs, dim)` (`dsc/dims.cpp:764-780`) is NOT an insert: after the
    /// `emplace` it OVERWRITES `primaryDimToValHandler_st(dim)` with the reference stage's value AND
    /// re-copies whichever of `coreletSplit_`/`rowSplit_`/`peSfpSplit_` already named the dim. An
    /// insert alone leaves this stage's own extent in place beside a symbol that does not describe it.
    fn make_dim_symbolic(&mut self, _at: v1::StageSite, _from: v1::StageSite, _dim: PrimaryDim) {
        todo!(
            "v1::ExploreStages::make_dim_symbolic: wants makeDimSymbolic(refDs, dim) \
             (dsc/dims.cpp:764-780) — it also overwrites primaryDimToValHandler_st(dim) from the \
             reference stage and re-copies every split that named the dim"
        )
    }

    /// ⛔ `makeDimNotSymbolic(dim)` (`dsc/dims.cpp:781-803`) — the DIVIDE-BY-FACTOR that follows the
    /// erase; see [`Dsc2Dims`]'s own note.
    fn make_dim_not_symbolic(&mut self, _at: v1::StageSite, _dim: PrimaryDim) {
        todo!(
            "v1::ExploreStages::make_dim_not_symbolic: wants makeDimNotSymbolic(dim) \
             (dsc/dims.cpp:781-803) — it erases the entry AND divides the dim value and every \
             corelet/row/PE-SFP share by maxSize_/granularity_"
        )
    }

    /// `pruneMaxSymbolicVolumes(refDstg)` (`dsc/dims.cpp:729-760`) — ⭐ ALREADY PORTED, as
    /// [`crate::schedule::l3::dsc::Symbolic::prune_volumes_from`], which FUSES the
    /// `maxSymbolicVolume_ = ref.maxSymbolicVolume_` assignment the reference makes before the prune.
    fn prune_max_symbolic_volumes(&mut self, at: v1::StageSite, from: v1::StageSite) {
        let Some(reference) = self.half(from) else {
            return;
        };
        self.edit(at, |half| {
            half.dims.symbolic.prune_volumes_from(&reference.dims.symbolic);
        });
    }

    /// `compound()` — ⭐ ALREADY PORTED, as
    /// [`crate::schedule::l3::dsc::StageDims::compound`].
    fn compound(&mut self, at: v1::StageSite) {
        self.edit(at, |half| half.dims.compound());
    }

    /// `denDs.el_ = denDs.ss_` — the whole half copied over.
    fn copy_ss_to_el(&mut self, stage: DatastageId) {
        self.facts().with_stages_mut(|stages| {
            if let Some(held) = stages.0.get_mut(&stage) {
                held.el.dims = held.ss.dims.clone();
                held.el.name = held.ss.name.clone();
            }
        });
    }

    /// `el_.name_ += "el"`.
    fn mark_epilogue_name(&mut self, stage: DatastageId) {
        self.facts().with_stages_mut(|stages| {
            if let Some(held) = stages.0.get_mut(&stage) {
                held.el.name.0.push_str("el");
                held.el.dims.name = held.el.name.clone();
            }
        });
    }

    /// `ds.r_ = ds.c_ = … = -1` — ⚠️ A NO-OP IN THIS MODEL, AND STATED ANYWAY, exactly as the trait's
    /// own doc says: those nine fields are all outside the closed twelve [`PrimaryDim`], so nothing
    /// reachable from here can observe them.
    fn clear_deprecated_dims(&mut self, _at: v1::StageSite) {}

    /// ⭐⭐ `finalizeExternalDataStage(dsc, stage, clSplitDims, numPTRows, usePt, rowSplitDim,
    /// peSfpSplitDims)` — `ddc/ddcv1.cpp:1748-1797`, READ, AND ITS THREE GUARDS ANSWERED.
    ///
    /// ⛔⛔ IT IS **NOT** GATED ON `isExternalDs` — THE NAME MISLEADS. The whole body is three
    /// independently guarded blocks and nothing else:
    ///
    /// 1. `if (dsc.numCoreletsUsed_DSC2_ > 1)` (`:1755`) — the corelet split, `ceil(extent / 2)` into
    ///    both halves of `coreletSplit_`;
    /// 2. `if (doPTSplit)` (`:1767`) — the PT-row split, `ceil(extent / numPTRows)` per corelet into
    ///    `rowSplit_`;
    /// 3. `for (auto dim : peSfpSplitDims)` (`:1778`) — `ceil`/`floor` halves into `peSfpSplit_`.
    ///
    /// ⭐ SO WITH ALL THREE GUARDS FALSE THE FUNCTION IS A COMPLETE NO-OP, and that is the authority's
    /// own control flow rather than a guess: one corelet, `usePt == No`, and an empty PE/SFP split set
    /// leave no statement in the body reachable. `clSplitDims` is read ONLY inside guard 1, so it
    /// cannot matter when that guard is false.
    ///
    /// ⭐ ALL THREE ARE FACTS THIS CARRIER HOLDS: `numCoreletsUsed_DSC2_` is the state's own cell
    /// (`prep_dsc` sets it from `numCoreletsUsed_`, and scratchy emits `numCoreletsUsed_ = 1` on every
    /// one of its 313 sampled SuperDSCs — `crustify-ddc/EXCLUSIONS.tsv`), and the other two are
    /// ARGUMENTS.
    ///
    /// ⛔ AND EVERY OTHER CASE IS A `todo!` NAMING THE WRITES. Each of the three blocks computes an
    /// EXTENT — a corelet's share, a PT row's share, a PE/SFP half — and every one of those becomes a
    /// buffer size and an address. A hand-rolled `ceil(extent / 2)` here would be a fabricated extent,
    /// and it would also need `primaryDimToVal_st` at a SAMPLED corelet, which
    /// [`Dsc2Dims::raw_slot`] refuses for the same reason.
    fn finalize_external_stage(
        &mut self,
        stage: DatastageId,
        _cl_split: &BTreeSet<PrimaryDim>,
        use_pt: v1::UsePt,
        _row_split: Option<PrimaryDim>,
        pe_sfp_split: &BTreeSet<PrimaryDim>,
    ) -> Option<()> {
        // `numCoreletsUsed_DSC2_` — [`None`] is the `-1` a DSC is built with, which the reference
        // compares as `> 1` and so takes as "no corelet split" too.
        let corelets = self.facts().corelets_dsc2().unwrap_or(1);
        if corelets <= 1 && use_pt == v1::UsePt::No && pe_sfp_split.is_empty() {
            return Some(());
        }
        todo!(
            "v1::ExploreStages::finalize_external_stage: finalizeExternalDataStage \
             (ddc/ddcv1.cpp:1748-1797) reaches a live block on stage {stage:?} — \
             numCoreletsUsed_DSC2_={corelets}, usePt={use_pt:?}, peSfpSplitDims={pe_sfp_split:?}. \
             Each block writes an EXTENT (ceil(extent/2) into coreletSplit_, ceil(extent/numPTRows) \
             into rowSplit_, ceil/floor halves into peSfpSplit_) off primaryDimToVal_st at a SAMPLED \
             corelet, which is a placement input and not a carrier's to invent."
        )
    }
}

impl tr::StageExtents for Dsc2Stages<'_, '_> {
    /// One stage's stick-view extent for one dim — ⭐ THE PLAIN SLOT of its steady-state half.
    fn stage_extent(&self, stage: DatastageId, dim: PrimaryDim) -> Extent {
        if let Some(half) = self.half(v1::StageSite::ss(stage))
            && let Some(extent) = half.raw_slot(dim, PadType::NoPad, v1::SymbolicRead::Max)
        {
            return extent;
        }
        todo!(
            "tr::StageExtents::stage_extent: wants \
             dataStageParam_.at(stage).ss_.primaryDimToVal_st(dim) (dsc/dims.cpp:647) folded over a \
             split this stage states on {dim:?}"
        )
    }
}

impl v1::Dsc2Stages for Dsc2Stages<'_, '_> {
    /// ⛔⛔ THE ONE METHOD THE CELL CANNOT SERVE — see this module's header. A `&mut` borrow cannot
    /// come out of a [`RefCell`], and handing back an owned copy is the exact defect this method's own
    /// doc warns about: *"A freshly built map here would DROP every effect entries 302, 303 and 338
    /// have, and it would still compile."*
    fn as_util(&mut self) -> &mut tu::DataStages<Self::Dims> {
        todo!(
            "v1::Dsc2Stages::as_util: wants a &mut to THE MAP THIS TYPE'S OWN ExploreStages READS. \
             That map lives in Dsc2State behind a RefCell, because Dsc2Carriers borrows `dsc` and \
             `stages` disjointly and v1::StageSizes on the `dsc` half reads the same dataStageParam_ \
             — so no owner can hand out a &mut. Returning an owned copy would drop every effect \
             entries 302/303/338 have, which is what this method's doc forbids."
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE DDL MATCH AND EXPORT SITE.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ ONE DSC AS THE DDL MATCH AND THE EXPORT READ AND WRITE IT — the `Y` carrier.
#[derive(Debug)]
pub struct Dsc2Ddl<'s, 'l> {
    state: &'s Dsc2State<'l>,
    dsc: DscIdx,
}

impl<'s, 'l> Dsc2Ddl<'s, 'l> {
    /// The DDL site of that DSC.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>, dsc: DscIdx) -> Self {
        Self { state, dsc }
    }

    fn facts(&self) -> &'s super::ddc_state::Dsc2Facts {
        self.state
            .facts(self.dsc)
            .expect("a Dsc2Ddl is only built for a DSC the state holds facts for")
    }
}

impl conv::AllocationSite for Dsc2Ddl<'_, '_> {
    /// `labeledDs_.at(lds).memOrg_.at(unit).allocateNode_` — ⭐ ANSWERED off the tree's own `memOrg_`.
    fn lds_allocation(&self, lds: LdsIdx, unit: SenComponent) -> Option<AllocId> {
        let tree = self.state.tree(self.dsc)?;
        let node = tree.org(lds)?.node(unit)?;
        tree.with(|held| held.allocate(node).map(|(alloc, _)| alloc))
    }

    /// `constantInfo_.at(constant).allocations_.at(unit)` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::ConstantInfo::allocations`], with [`None`] for either `.at()`'s
    /// throw exactly as the trait's own [`Option`] spells it.
    fn constant_allocation(&self, constant: ConstIdx, unit: SenComponent) -> Option<AllocId> {
        self.facts().with_dsc(|dsc| {
            dsc.ddc
                .constants
                .get(&constant)?
                .allocations
                .get(&unit)
                .copied()
        })
    }
}

impl conv::InternalTensorSite for Dsc2Ddl<'_, '_> {
    /// `dsc.labeledDs_`'s tail — ⭐ ANSWERED.
    fn labeled_ds_tail(&self) -> Option<conv::LabeledDsTail> {
        self.facts().with_dsc(|dsc| {
            let positions = ddc_state::lds_positions(dsc);
            Some(conv::LabeledDsTail {
                insert_position: *positions.last()?,
                last_lds: dsc.labeled_ds.back().recorded(),
            })
        })
    }

    /// `++dsc.labeledDs_.back().ldsIdx_` — ⭐ ANSWERED through the shared `currDsc` cell, the same
    /// write [`tu::NewLabeledDs::set_last_recorded_lds_idx`] makes.
    fn set_last_lds_idx(&mut self, lds: LdsIdx) {
        self.facts().with_dsc_mut(|dsc| {
            dsc.labeled_ds.back_mut().set_recorded(lds);
        });
    }

    /// ⛔⛔ `dsc.labeledDs_.insert(end() - 1, newLds)` — AND WHAT BLOCKS IT IS NOW `density_` AND
    /// `referenceLdsIdx_`, NOT THE CELL OR THE RECORD. `addInternalTensor`
    /// (`ddc/ddl/ddl_conversion.cpp:482-498`) copies EIGHT fields off the reference entry —
    /// `dsType_`, `segment_ = STACK`, `isFirstUse_`, `scale_`, `density_`, `wordLength`,
    /// `dataFormat_`, `referenceLdsIdx_` — and [`crate::schedule::l3::dsc::LabeledDs`] projects
    /// neither `density_` (`dsc/dscdefn.h:333`), `segment_` (`:328`), `isFirstUse_` (`:331`) nor
    /// `referenceLdsIdx_` (`:324`).
    ///
    /// ⛔ AND THE ONE THAT DECIDES A SIZE IS `density_`: it is the per-layout-dim occupancy
    /// `getBufferCapacityForNode` multiplies an extent by, so a copy that dropped it would size the
    /// internal tensor's buffer as if it were dense.
    fn insert_internal_tensor(&mut self, _new: conv::InternalTensor) {
        todo!(
            "conv::InternalTensorSite::insert_internal_tensor: wants \
             labeledDs_.insert(end() - 1, newLds) copying dsType_/segment_/isFirstUse_/scale_/\
             density_/wordLength/dataFormat_/referenceLdsIdx_ off the reference \
             (ddc/ddl/ddl_conversion.cpp:482-498). wordLength and dataFormat_ ARE now carried; \
             density_ (dsc/dscdefn.h:333), segment_, isFirstUse_ and referenceLdsIdx_ are not, and \
             density_ is what getBufferCapacityForNode multiplies an extent by"
        )
    }

    /// ⛔ `computeOp_.at(compute_op).interimLabeledDs.push_back(&newLds)` — `interimLabeledDs` is not
    /// a field of [`v1::DscComputeOp`].
    fn add_interim_lds(&mut self, _compute_op: conv::ComputeOpIdx, _lds: LdsIdx) {
        todo!(
            "conv::InternalTensorSite::add_interim_lds: wants \
             computeOp_.at(op).interimLabeledDs.push_back(&newLds) — interimLabeledDs is not a field \
             of v1::DscComputeOp"
        )
    }

    /// ⛔ The `{ALLOCATE, TRANSFER, COMPUTE}` walk PROJECTED ONTO EVERY LDS SLOT — its
    /// [`conv::LdsSlot::ComputeInput`]/`ComputeOutput`/`OpaqueCompute` arms all need a COMPUTE node,
    /// and a PARTIAL walk would leave entry 173's retagging half done.
    fn lds_slots(&self) -> Vec<conv::LdsSlot> {
        todo!(
            "conv::InternalTensorSite::lds_slots: wants the {{ALLOCATE, TRANSFER, COMPUTE}} walk \
             projected onto every slot naming a labeled DS — three of its six arms need a COMPUTE \
             node, which super::tree::Kind has no arm for, and a partial walk leaves entry 173's \
             retagging half done"
        )
    }

    /// ⛔ The read half of the same walk.
    fn slot_lds(&self, _slot: conv::LdsSlot) -> Option<LdsIdx> {
        todo!("conv::InternalTensorSite::slot_lds: see lds_slots")
    }

    /// ⛔ And its write half.
    fn set_slot_lds(&mut self, _slot: conv::LdsSlot, _lds: LdsIdx) {
        todo!("conv::InternalTensorSite::set_slot_lds: see lds_slots")
    }

    /// `compAndAllocNode`'s own `allocNode->ldsIdx_` — ⭐ ANSWERED off the L3 view.
    fn allocation_lds(&self, alloc: AllocId) -> Option<LdsIdx> {
        let tree = self.state.tree(self.dsc)?;
        tree.with(|held| {
            held.node_of_alloc(alloc)
                .and_then(|node| held.allocate(node))
                .map(|(_, node)| node.lds)
        })
    }

    /// ⛔ `allocNode->ldsIdx_ = lds` — the same missing setter
    /// [`tu::NewLabeledDs::set_alloc_lds_idx`] names.
    fn set_allocation_lds(&mut self, _alloc: AllocId, _lds: LdsIdx) {
        todo!(
            "conv::InternalTensorSite::set_allocation_lds: wants allocNode->ldsIdx_ = lds — \
             super::tree::TreeData has no setter for L3AllocateNode::lds"
        )
    }
}

impl conv::DdlSite for Dsc2Ddl<'_, '_> {
    /// ⛔ `dsc.computeOp_.front().attributes_.dataFormat_` — ⭐ ANSWERED, because
    /// [`v1::DscComputeOp::format`] IS that field.
    fn fused_format(&self) -> Option<DataFormat> {
        self.facts().ops().first().and_then(|op| op.format)
    }

    /// `dsc.labeledDs_.at(lds).dataFormat_` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::LdsRecord::data_format`]. This is what the DDL match BINDS each
    /// operand's type by, and it VARIES on real data: `SEN143_FP8` on 7 of `g0/`'s 580 labelled DSs and
    /// `SEN169_FP16` on the other 573.
    fn lds_format(&self, lds: LdsIdx) -> Option<DataFormat> {
        self.facts()
            .with_lds(lds, |held| held.record().data_format)
            .flatten()
    }

    /// ⛔ `addressGranularityScalePerUnit.count({senCompToGenericComp.at(unit), storage})` — the
    /// system-definition table [`v1::OffsetSizes::address_scale`] also wants.
    fn unit_reaches(&self, _unit: SenComponent, _storage: SenComponent) -> bool {
        todo!(
            "conv::DdlSite::unit_reaches: wants \
             addressGranularityScalePerUnit.count({{generic, storage}}) — the scheduler's own \
             sysDef construction argument, not a super-DSC fact"
        )
    }

    /// `dsc.getDimIndexInLayoutOrder(labeledDs_.at(lds).dsType_, dim) >= 0` — ⭐ ANSWERED off
    /// `layoutDimOrder_`.
    fn dim_in_layout_order(&self, lds: LdsIdx, dim: PrimaryDim) -> bool {
        self.facts().with_dsc(|dsc| {
            dsc.layout_dims
                .get(&lds)
                .is_some_and(|layout| layout.iter().any(|named| named == dim))
        })
    }

    /// `sdsc.numWkSlicesPerDim_.at(dim)` — ⭐ ANSWERED off the super-DSC's map, copied into the state.
    fn wk_slices(&self, dim: PrimaryDim) -> Option<u32> {
        self.state.wk_slices(dim).map(|count| count.get())
    }

    /// `sdsc.coreIdToWkSlice_` — ⭐ ANSWERED, likewise; the ring's neighbour lookup needs the WHOLE
    /// slice vector per core, which is what this hands back.
    fn core_work_slices(&self) -> BTreeMap<Core, WkSlice> {
        self.state.core_wk_slices()
    }

    /// `dsc.primaryDsInfo_.at(labeledDs_.at(lds).dsType_)`'s stick pair — ⭐ ANSWERED.
    fn stick_dims(&self, lds: LdsIdx) -> Option<StickDims> {
        self.facts()
            .with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds))
    }
}

impl conv::MatchSite for Dsc2Ddl<'_, '_> {
    /// `dsc.computeOp_`, IN ORDER — ⭐ ANSWERED from `computeOp_`.
    ///
    /// ⛔ `coreExclude`/`coreClExclude` ARE EMPTY, and that is a projection gap rather than a
    /// reading: [`v1::DscComputeOp`] carries the five fields entries 307/308 read and neither exclude
    /// list. ⚠️ An op whose exclusions the DDL match should have honoured will bind on every core.
    /// ⛔ AND AN OP WITH NO `opFuncName` IS DROPPED, not defaulted: [`conv::BindableOp::op_func`] has
    /// no absent state and the reference's `NONE` cannot bind a template.
    fn bindable_ops(&self) -> Vec<conv::BindableOp> {
        self.facts()
            .ops()
            .into_iter()
            .filter_map(|op| {
                Some(conv::BindableOp {
                    op_func: op.op_func?,
                    format: op.format,
                    inputs: op.inputs,
                    outputs: op.outputs,
                    core_exclude: BTreeSet::new(),
                    core_cl_exclude: BTreeSet::new(),
                })
            })
            .collect()
    }

    /// `dsc.labeledDs_.at(lds).wordLength` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::LdsRecord::word_length`]. ⛔ [`None`] IS THAT `.at()`'s THROW and
    /// NOT the declared `0`: the DDL match compares this width against the template's own
    /// `bitSize_ / 8`, and a `0` for an absent entry would report a mismatch instead of a missing
    /// operand.
    fn lds_word_length(&self, lds: LdsIdx) -> Option<WordLength> {
        self.facts().with_lds(lds, |held| held.record().word_length)
    }

    /// `newLds.wordLength = type->bitSize_ / 8` — ⭐ ANSWERED through the shared `currDsc` cell.
    fn set_lds_word_length(&mut self, lds: LdsIdx, length: WordLength) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_word_length(length));
    }

    /// `newLds.dataFormat_ = type->dataFormat_` — ⭐ ANSWERED, and it is the SAME `labeledDs_` entry
    /// [`Self::lds_word_length`] reads, so the width and the format a template binds cannot drift.
    fn set_lds_format(&mut self, lds: LdsIdx, format: DataFormat) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_data_format(format));
    }
}

impl conv::DdlSizes for Dsc2Ddl<'_, '_> {
    /// ⛔ `getBlockTransferSize(node, src_.unit_, 0, false, true)` FOLDED WITH the
    /// `!getRelevantCoreCl().empty()` that gates it — the first is a `dsc/` accessor and the second
    /// wants the `relevantCoreCl_` map `setRelevantCompCoreCl` fills.
    fn block_transfer_size(&self, _transfer: NodeId) -> Option<Elements> {
        todo!(
            "conv::DdlSizes::block_transfer_size: wants getBlockTransferSize(node, src_.unit_, 0, \
             false, true) gated on !getRelevantCoreCl().empty() — a `dsc/` accessor plus the \
             relevantCoreCl_ map dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647) fills"
        )
    }

    /// ⛔ `getBufferCapacityForNode(allocatenode, ldsIdx_, component_, 0, 0)` — the SAME capacity
    /// call [`v1::Placement::buffer_capacity`] names.
    fn buffer_capacity(&self, _alloc: AllocId) -> Option<Elements> {
        todo!(
            "conv::DdlSizes::buffer_capacity: wants getBufferCapacityForNode(alloc, ldsIdx_, \
             component_, 0, 0) (dsc/dsc2.cpp:3977) over the live super-DSC's stick sizes"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ THE DDL TEMPLATE SET — THE GENERATED MODULES, WHICH IS WHERE THIS STAGE USED TO STOP.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// THE `T` CARRIER — [`conv::DdlTemplateSet`], whose one method hands back a
/// [`conv::StatedTemplate`] with SIX parts: `source`, `program`, `binds`, `padded`, `constraints`,
/// `root`.
///
/// ⭐⭐ ALL SIX NOW COME OUT OF THE `.ddl`. `build.rs` used to emit exactly one — `PROGRAMS` served
/// `program` and the other four were mentioned ZERO times each — so this carrier's only honest answer
/// was [`None`], and stage 2b's `run_v1` stopped here. It now walks each template MODULE-WIDE beside
/// the per-bind walk and emits `crate::generated::MODULES`;
/// [`crate::schedule::ddl::templates::DdlTemplates`] is the one implementor over it, and this carrier
/// delegates.
///
/// ⛔ THE DELEGATION IS TOTAL, so nothing is recorded here any more. `Template`'s variants and
/// `MODULES`' rows are minted from the same census of `ddl_templates/*.ddl`, so a template a candidate
/// list can name is a template this set holds.
///
/// ⛔ AND IT DOES NOT SHORT-CIRCUIT. `select_and_parse_ddl_template` reaches this only AFTER
/// `ddl_templates(opFunc, A::GEN)` answered [`Some`] (`ddl/conversion.rs:6011`), so an op-func with no
/// candidate templates never asks — it returns *"no DDL available for op …"* and stage 2b answers
/// [`v1::DscFilled::No`] for the super-DSC, which is a COMPLETE run of stage 2b and not a stop.
#[derive(Debug)]
pub struct DdcTemplates<'s, 'l> {
    /// Kept so a future refusal of this carrier's own has somewhere to go; the set itself has none.
    #[expect(
        dead_code,
        reason = "the state is the refusal sink every other carrier of this module holds, and this \
                  one no longer refuses — see the type's note"
    )]
    state: &'s Dsc2State<'l>,
    /// The generated set this carrier delegates to. A FIELD and not a temporary, because
    /// [`conv::DdlTemplateSet::stated`] hands back a [`conv::StatedTemplate`] borrowed from the set.
    set: crate::schedule::ddl::templates::DdlTemplates,
}

impl<'s, 'l> DdcTemplates<'s, 'l> {
    /// The template set.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>) -> Self {
        Self {
            state,
            set: crate::schedule::ddl::templates::DdlTemplates::new(),
        }
    }
}

impl conv::DdlTemplateSet for DdcTemplates<'_, '_> {
    type Source = crate::schedule::ddl::templates::TemplateSource;

    /// ⭐ THE GENERATED MODULES, verbatim — see [`crate::schedule::ddl::templates::DdlTemplates`].
    fn stated(
        &self,
        template: crate::generated::Template,
    ) -> Option<conv::StatedTemplate<'_, Self::Source>> {
        conv::DdlTemplateSet::stated(&self.set, template)
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ THE PROVIDER — every store of one DSC, borrowed together.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE DSC'S CARRIERS, OWNED SO THAT [`v1::Dsc2Sites::carriers`] CAN HAND OUT TEN DISJOINT `&mut`s.
#[derive(Debug)]
struct PerDsc<'s, 'l> {
    store: Dsc2Store<'s, 'l>,
    stages: Dsc2Stages<'s, 'l>,
    ddl: Dsc2Ddl<'s, 'l>,
    allocs: v1::AllocArena,
    computes: v1::ComputeArena,
    syncs: BTreeMap<NodeId, crate::schedule::dsc2::SyncNode>,
}

/// ⭐⭐ WHERE ONE DSC'S CARRIERS COME FROM — `sdsc.dscs_.at(idx)` and everything hung off it.
///
/// ⛔ EVERY CARRIER IS OWNED HERE because [`v1::Dsc2Carriers`] is TEN `&'a mut`s taken from
/// `&mut self` at once: they must be ten disjoint FIELDS, which is what this is.
#[derive(Debug)]
pub struct Dsc2Provider<'s, 'l> {
    state: &'s Dsc2State<'l>,
    per: Vec<PerDsc<'s, 'l>>,
    trackers: ddc_state::DdcTrackers,
    sink: ddc_state::DdcSink,
    symbols: ddc_state::DdcSymbols,
    coords: ddc_state::DdcCoords,
}

impl<'s, 'l> Dsc2Provider<'s, 'l> {
    /// The provider over every DSC the state holds, along the fold manager's own address coordinates.
    #[must_use]
    pub fn new(state: &'s Dsc2State<'l>, coords: &AddressFoldCoords) -> Self {
        let per = (0..state.len())
            .filter_map(|at| u32::try_from(at).ok().map(DscIdx))
            .map(|dsc| PerDsc {
                store: Dsc2Store::new(state, dsc, coords.clone()),
                stages: Dsc2Stages::new(state, dsc),
                ddl: Dsc2Ddl::new(state, dsc),
                allocs: seed_arena(state, dsc),
                computes: v1::ComputeArena::new(),
                syncs: BTreeMap::new(),
            })
            .collect();
        Self {
            state,
            per,
            trackers: ddc_state::DdcTrackers,
            sink: ddc_state::DdcSink::default(),
            symbols: ddc_state::DdcSymbols,
            coords: ddc_state::DdcCoords,
        }
    }

    /// Where entry 260's fills landed, for a caller that wants to read them back.
    #[must_use]
    pub const fn sink(&self) -> &ddc_state::DdcSink {
        &self.sink
    }
}

impl<'s, 'l> v1::Dsc2Sites for Dsc2Provider<'s, 'l> {
    type Dsc = Dsc2Store<'s, 'l>;
    type Stages = Dsc2Stages<'s, 'l>;
    type Ddl = Dsc2Ddl<'s, 'l>;
    type Trackers = ddc_state::DdcTrackers;
    type Sink = ddc_state::DdcSink;
    type Symbols = ddc_state::DdcSymbols;
    type Coords = ddc_state::DdcCoords;

    /// `dsc.name_` — the construction argument [`Dsc2State::seeded`] takes, which only the two verbose
    /// lines read.
    fn dsc_name(&self, dsc: DscIdx) -> v1::StorageName {
        self.state
            .facts(dsc)
            .map_or_else(|| v1::StorageName(format!("dsc{}", dsc.0)), |facts| facts.name())
    }

    /// ⛔ [`None`] IS `dscs_.at(idx)`'s OWN THROW — a `dscs_` position this provider holds no stores
    /// for, which is what ends `run_v1`'s loop.
    fn carriers(
        &mut self,
        dsc: DscIdx,
    ) -> Option<
        v1::Dsc2Carriers<
            '_,
            Self::Dsc,
            Self::Stages,
            Self::Ddl,
            Self::Trackers,
            Self::Sink,
            Self::Symbols,
            Self::Coords,
        >,
    > {
        let per = self.per.get_mut(usize::try_from(dsc.0).ok()?)?;
        Some(v1::Dsc2Carriers {
            dsc: &mut per.store,
            stages: &mut per.stages,
            ddl: &mut per.ddl,
            allocs: &mut per.allocs,
            computes: &mut per.computes,
            syncs: &mut per.syncs,
            trackers: &mut self.trackers,
            sink: &mut self.sink,
            symbols: &mut self.symbols,
            coords: &self.coords,
        })
    }
}

/// ⭐⭐ THE `ddc` VIEW OF EVERY ALLOCATE NODE THE TREE ALREADY HOLDS — the arena
/// [`v1::Dsc2Carriers::allocs`] is, seeded so that IT AND THE TREE CANNOT DISAGREE.
///
/// ⛔⛔ WITHOUT THIS, STAGE 2B STOPS ON ITS FIRST ALLOCATE. `attach_to_prefilled_schedule` reads
/// `allocs.get(&alloc)?` for every `ALLOCATE` of the tree (`ddc/v1.rs:5964-5967`) and answers [`None`]
/// for one the arena has no entry for — and an EMPTY arena beside a tree with three HBM allocations is
/// exactly the state `AllocArena::new()` leaves.
///
/// ⭐ IT IS NOT A FABRICATION, AND [`super::tree::Org`]'S OWN DOC IS WHY: *"ITS `allocateNode_` IS ONE
/// NODE READ THROUGH TWO VOCABULARIES"* — [`crate::schedule::l3::dl_ops::L3AllocateNode`] is what
/// entry 353 mints and [`crate::schedule::dsc2::AllocateNode`] is what entries 219/220/292 place.
/// Every field below is COPIED from the L3 view of the same node; nothing is derived.
///
/// ⛔⛔ AND THE PLACED STATE IS LEFT AT THE FRESH-NODE DEFAULT, WHICH IS A STATEMENT AND NOT A
/// SHORTCUT. `startAddressCoreCorelet_`, `numBuffers_`, `padding_`, `bufferOffsetCoreCorelet_` and
/// `isStartAddrSymbolic_` are what the MEMORY TRACKER and entries 258/259 write, and
/// [`crate::schedule::dsc2::AllocPlacement`]'s own doc says *"every construction site of a fresh
/// allocate node wants all four at their defaults"*. Stage 2a stops AT that tracker, so an unplaced
/// node is the true state of every allocation this seed can see.
///
/// ⭐ `numBuffers_` IS THE ONE PLACEMENT FIELD THAT IS CARRIED, because entry 353 — a GROWER, not the
/// tracker — is what chooses it: [`crate::schedule::l3::dl_ops::set_lx_buffer_type`] picks
/// `LxBufferChoice::Double` and `create_allocation_and_transfer` stamps it on every LX allocation
/// BEFORE any address exists. `numBuffers_` is a placement INPUT (the double-buffer stride is
/// `numBuffers x bufferOffset`), so silently writing [`crate::schedule::dsc2::NumBuffers::Single`] over
/// a `Double` the L3 scheduler chose would halve every buffer — the copy below is what prevents that,
/// and the mapping is the closed three-way the field's own comment names
/// (*"1:no buffering, 2:double-buffer, -1:streaming buffer"*).
///
/// ⛔ `bufferOffsetCoreCorelet_` IS NOT CARRIED AND MUST NOT BE: it is the tracker's own output
/// (`g0/debug/sdsc_0/sdsc.json` puts it at 256 on every LX node and every core), and the tracker has
/// not run.
///
/// ⛔ AND A NON-DEFAULT `padding_` IS REFUSED RATHER THAN DROPPED. [`crate::schedule::dsc2::Padding`]
/// is a per-dim [`PadType`] map and the L3 view carries a
/// [`crate::schedule::ddc::transformation_util::PaddingForm`] — TWO Rust spellings of `padding_`
/// (`dsc/dsc2.h:981`) whose relation is `getPadding(dim)` (`dsc/dims.cpp:806`), a `dsc/` seam. Reading
/// one out of the other is that seam and not a copy, so a stated padding stops here instead.
///
/// ⛔ AN ALLOCATION WHOSE `layoutDimOrder_` IS EMPTY IS LIKEWISE LEFT OUT:
/// [`crate::schedule::dsc2::AllocLayout`] is non-empty by type, which is the reference's own
/// `layoutDimOrder_.at(0)` (`ddc/ddcv1.cpp:1704`).
fn seed_arena(state: &Dsc2State<'_>, dsc: DscIdx) -> v1::AllocArena {
    use crate::schedule::dsc2::{AllocLayout, AllocateNode, MaxDimSize, NumBuffers};
    use crate::schedule::l3::dsc::Buffering;

    let mut arena = v1::AllocArena::new();
    let Some(tree) = state.tree(dsc) else {
        return arena;
    };
    tree.with(|held| {
        for node in held.dfs() {
            let Some((alloc, minted)) = held.allocate(node) else {
                continue;
            };
            // `numBuffers_` — the closed three-way the field's own comment names, carried across.
            let num_buffers = match minted.buffering {
                Buffering::None => NumBuffers::Single,
                Buffering::Double => NumBuffers::Double,
                Buffering::Streaming => NumBuffers::Streaming,
            };
            // ⛔ `padding_` READ THROUGH THE OTHER SPELLING IS `getPadding(dim)`, a `dsc/` seam.
            if minted.padding != tu::PaddingForm::default() {
                let _: Option<()> = state.refuse(
                    "Dsc2Provider: an allocation whose L3 padding_ (a PaddingForm) is non-default is \
                     left out of the AllocArena — dsc2::Padding is a per-dim PadType map, and \
                     reading one out of the other is getPadding(dim) (dsc/dims.cpp:806), a `dsc/` \
                     seam and not a copy",
                );
                continue;
            }
            // `layoutDimOrder_` zipped with `maxDimSizes_`, innermost first, and an UNBOUNDED entry is
            // the reference's own `resize(n, -1)` (`dsc/dsc2.h:982`) — which is `MaxDimSize::Unset`.
            let mut layout = minted.layout.0.iter().map(|(dim, max)| {
                (
                    *dim,
                    max.map_or(MaxDimSize::Unset, MaxDimSize::Resolved),
                )
            });
            let Some(first) = layout.next() else {
                let _: Option<()> = state.refuse(
                    "Dsc2Provider: an allocation whose layoutDimOrder_ is EMPTY is left out of the \
                     AllocArena — dsc2::AllocLayout is non-empty by type, which is the reference's \
                     own layoutDimOrder_.at(0) (ddc/ddcv1.cpp:1704)",
                );
                continue;
            };
            arena.insert(
                alloc,
                AllocateNode {
                    name: minted.name.clone(),
                    component: minted.component,
                    lds: Some(minted.lds),
                    // `constIdx_` — this node is a labelled DS's, not a constant's.
                    const_idx: None,
                    temp_storage_for_compute: None,
                    layout: AllocLayout::new(first, layout.collect()),
                    start_address: crate::schedule::dsc2::StartAddress::default(),
                    placement: crate::schedule::dsc2::AllocPlacement {
                        num_buffers,
                        ..crate::schedule::dsc2::AllocPlacement::default()
                    },
                    gap_stick_spread: BTreeMap::new(),
                    // `allocUsers_` — the users list lives on `super::tree::Org`, and stage 2b's own
                    // `add_alloc_user` writes it there; an arena copy would be a second answer.
                    alloc_users: Vec::new(),
                },
            );
        }
    });
    arena
}

/// ⛔ A `RefCell` IS NAMED IN THIS MODULE'S HEADER — kept referenced so the doc link resolves.
const _: fn(&RefCell<()>) = |_| ();
