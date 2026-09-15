// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE SHARED HALF OF ONE `currDsc` — [`v1::Dsc2Reads`], which [`v1::Dsc2Store::split`] hands out
//! BESIDE the exclusive tree, and which reads the SAME [`super::ddc_state::Dsc2State`] the tree does.
//!
//! ⛔⛔ THE SECOND SEVERING, AND IT IS `Dsc2Carriers`' OWN — SAY IT HERE, ONCE.
//!
//! `Dsc2Carriers` (`ddc/v1.rs:6180-6201`) borrows `dsc: &'a mut E` and `stages: &'a mut S` from the
//! provider at the same time, so the `Dsc2Store` half may not hold a reference to the `Dsc2Stages`
//! half. But [`v1::StageSizes`] and [`v1::OffsetSizes`] — nine of this carrier's forty-one methods —
//! read `dataStageParam_`, which is exactly what the `Stages` half owns. ⭐ THE REPAIR IS THE SAME
//! ONE STAGE 2A USED: the map lives in the STATE, behind a cell, and both halves name it. So every
//! datastage read below is a REAL read of the map [`super::Dsc2Stages`] writes, not a snapshot.
//!
//! ⛔ THE ONE THING THAT CANNOT BE REPAIRED THAT WAY is [`v1::Dsc2Stages::as_util`], which returns a
//! `&mut UtilDataStages<..>` — a real exclusive borrow, which no cell can hand out. It is the single
//! `todo!` of [`super::Dsc2Stages`] and it names this.
//!
//! # ⛔ WHAT IS ANSWERED AND WHAT IS NOT
//!
//! Answered from the DSC scratchy writes: the cores and corelets, the layout and stick orders, the
//! non-broadcast dim sets, the work-slice splits, the scale-tensor facts, the `memOrg_` allocations,
//! and every read of the shared datastage map in the ONE case the authority's own control flow makes
//! a plain field read (see [`super::ddc_state::Dsc2Dims`]).
//!
//! Not answered, each naming its field: `dsName_`, `dataFormat_`, `constantInfo_`,
//! `coordinateMasking_`, `dimToSymbolMapping_`, `l0TetheredMode_` and `opConsts` — SEVEN fields of
//! `DesignSpaceConfig` that [`crate::schedule::l3::dsc::DesignSpaceConfig`] does not project. ⛔ THAT
//! IS A CONVERSION GAP AND NOT A DATA GAP: `g0/sdsc_0.json` carries `dataFormat_` and `dsName_` on
//! every `labeledDs_` entry, so the facts exist in what scratchy writes and the l3 projection is
//! where they are dropped.

use core::num::NonZeroU64;
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Bytes, Elements};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickDims,
};
use crate::formats::DataFormat;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{
    AllocId, Cardinality, ConstIdx, NodeId, PadType, ScaleBlock, ScaledLds,
};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::PaddingForm;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{LayoutDims, LdsIdx, LdsScale};
use crate::schedule::l3::dl_ops::AddressFoldCoords;
use crate::schedule::l3::dsc::DscIdx;
use crate::units::{Corelet, Row};

use super::ddc_state::{self, Dsc2Dims, Dsc2State};

/// ⭐ THE SHARED VIEW OF ONE `currDsc`.
#[derive(Debug)]
pub struct Dsc2Reads<'s, 'l> {
    state: &'s Dsc2State<'l>,
    dsc: DscIdx,
    /// `{coreFoldProp_, coreletFoldProp_} ++ sdscFoldProps_` — a CONSTRUCTION ARGUMENT of the
    /// reference's fold manager and not a super-DSC field, exactly as stage 2a takes it
    /// ([`super::run_l3`]'s `coords`).
    coords: AddressFoldCoords,
}

impl<'s, 'l> Dsc2Reads<'s, 'l> {
    /// The shared view over that DSC, along the fold manager's own address coordinates.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>, dsc: DscIdx, coords: AddressFoldCoords) -> Self {
        Self {
            state,
            dsc,
            coords,
        }
    }

    /// That DSC's facts — ⛔ TOTAL BY CONSTRUCTION, as [`super::Dsc2Tree`]'s tree is.
    fn facts(&self) -> &'s super::ddc_state::Dsc2Facts {
        self.state
            .facts(self.dsc)
            .expect("a Dsc2Reads is only built for a DSC the state holds facts for")
    }

    /// One half of one datastage of the SHARED map, cloned for the length of one read.
    fn half(&self, at: v1::StageSite) -> Option<Dsc2Dims> {
        self.facts().with_stages(|stages| {
            let held = stages.0.get(&at.stage)?;
            Some(match at.half {
                v1::StageHalf::Ss => held.ss.dims.clone(),
                v1::StageHalf::El => held.el.dims.clone(),
            })
        })
    }

    /// `dataStageParam_.at(stage).ss_` — the half every [`v1::StageSizes`] and [`v1::OffsetSizes`]
    /// method is keyed by, since both take a bare [`DatastageId`].
    fn steady(&self, stage: DatastageId) -> Option<Dsc2Dims> {
        self.half(v1::StageSite::ss(stage))
    }
}

impl crate::schedule::dsc2::Dsc for Dsc2Reads<'_, '_> {
    /// `getLayoutDims(ldsIdx)` — ⛔ TOTAL by the trait's signature, and the reference's own
    /// `getLayoutDims` ABORTS for an index it holds no order for (`dsc/dsc2.cpp:4007`).
    fn layout_dims(&self, lds: LdsIdx) -> LayoutDims {
        self.facts()
            .with_dsc(|dsc| dsc.layout_dims.get(&lds).cloned())
            .unwrap_or_else(|| {
                panic!(
                    "Dsc::layout_dims: getLayoutDims({lds:?}) aborts for an lds this DSC states no \
                     layoutDimOrder_ for (dsc/dsc2.cpp:4007)"
                )
            })
    }
}

impl v1::LabeledDsIndices for Dsc2Reads<'_, '_> {
    /// `labeledDs_.at(i).ldsIdx_` for every entry, in order — ⭐ THE ENTRY'S OWN RECORDED INDEX and
    /// not its position, which is what [`crate::schedule::l3::dsc::LabeledDs::recorded`] holds.
    fn labeled_ds_indices(&self) -> Vec<LdsIdx> {
        self.facts().with_dsc(|dsc| {
            dsc.labeled_ds
                .iter()
                .map(crate::schedule::l3::dsc::LabeledDs::recorded)
                .collect()
        })
    }
}

impl v1::LdsSticks for Dsc2Reads<'_, '_> {
    /// `primaryDsInfo_.at(labeledDs_.at(lds).dsType_)`'s stick order zipped with its sizes.
    fn stick_dims(&self, lds: LdsIdx) -> StickDims {
        self.facts()
            .with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds))
            .unwrap_or_default()
    }
}

impl v1::StorageNames for Dsc2Reads<'_, '_> {
    /// `labeledDs_.at(lds).dsName_` (`dsc/dscdefn.h:326`) — ⭐ ANSWERED off [`LdsRecord::name`].
    ///
    /// ⛔ TOTAL BY THE TRAIT'S SIGNATURE, and the reference's `.at()` THROWS for an index the list
    /// does not hold — `currDsc->labeledDs_.at(anode->ldsIdx_).dsName_`
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5498`) over a `std::vector`. So an absent index
    /// panics rather than answering the empty name, which would key the memory tracker by a name the
    /// reference never used.
    fn lds_name(&self, lds: LdsIdx) -> v1::StorageName {
        self.facts()
            .with_lds(lds, |held| held.record().name.clone())
            .unwrap_or_else(|| {
                panic!("v1::StorageNames::lds_name: labeledDs_.at({lds:?}) throws for an absent index")
            })
    }

    /// `constantInfo_.at(constant).name_` (`dsc/dsc2.h:48`) — ⭐ ANSWERED off [`DdcFacts::constants`].
    ///
    /// ⛔ TOTAL LIKEWISE, and `constantInfo_` is reached with `.at()` —
    /// `currDsc->constantInfo_.at(anode->constIdx_).name_` (`L3DlOpsScheduler.cpp:5500`): a constant
    /// id the table does not hold is that throw. The empty name is a real value of the field (a constant
    /// nothing named), so it cannot double as the missing-entry answer.
    fn constant_name(&self, constant: ConstIdx) -> v1::StorageName {
        self.facts()
            .with_dsc(|dsc| dsc.ddc.constants.get(&constant).map(|held| held.name.clone()))
            .unwrap_or_else(|| {
                panic!(
                    "v1::StorageNames::constant_name: constantInfo_.at({constant:?}) throws for an \
                     id this DSC's table does not hold"
                )
            })
    }
}

impl v1::ExploreDsc for Dsc2Reads<'_, '_> {
    /// `lds < labeledDs_.size()`.
    fn holds_lds(&self, lds: LdsIdx) -> bool {
        self.facts().with_lds(lds, |_| ()).is_some()
    }

    /// `labeledDs_.at(lds).scaledLdsCategory_` (`dsc/dscdefn.h:352-356`) — ⭐ ANSWERED AS THE CLOSED
    /// THREE-WAY IT IS, off [`crate::schedule::l3::dsc::LabeledDs::scaled_category`].
    ///
    /// ⛔ AN ABSENT INDEX IS THE `.at()` THROW and not `REGULAR_TENSOR`: the trait is total,
    /// `REGULAR_TENSOR` is a real category entry 307 branches on, and the reference reads it as
    /// `currDsc->labeledDs_.at(allocNodeToMatch->ldsIdx_).scaledLdsCategory_`
    /// (`ddc/ddc_fold.cpp:463`).
    fn scaled_category(&self, lds: LdsIdx) -> ScaledLds {
        self.facts()
            .with_lds(lds, crate::schedule::l3::dsc::LabeledDs::scaled_category)
            .unwrap_or_else(|| {
                panic!(
                    "v1::ExploreDsc::scaled_category: labeledDs_.at({lds:?}) throws for an absent \
                     index"
                )
            })
    }

    /// `labeledDs_.at(lds).mxInfo_`'s dim and block size.
    fn mx_info(&self, lds: LdsIdx) -> Option<(PrimaryDim, ScaleBlock)> {
        self.facts()
            .with_lds(lds, crate::schedule::l3::dsc::LabeledDs::scale_tensor)
            .flatten()
            .map(|held| (held.dim, held.blk_size))
    }
}

impl v1::Masking for Dsc2Reads<'_, '_> {
    /// ⛔ Wants `coordinateMasking_` (`dsc/designSpaceConfig.h:100`).
    fn coordinate_masking(&self) -> BTreeMap<PrimaryDim, Vec<v1::MaskRun>> {
        todo!(
            "v1::Masking::coordinate_masking: wants coordinateMasking_ \
             (dsc/designSpaceConfig.h:100), not projected onto l3::dsc::DesignSpaceConfig. An EMPTY \
             map is a real state of the reference's field (entry 262's first arm), so answering \
             empty would pin 'no masking' as a FACT for a DSC that may well request some."
        )
    }

    /// ⛔ Wants `computeOp_.at(0).opConsts.count("samv-wsllen")` — `opConsts` is not carried by
    /// [`v1::DscComputeOp`], which projects the five fields entries 307/308 read.
    fn declares_samv_wsllen(&self) -> bool {
        todo!(
            "v1::Masking::declares_samv_wsllen: wants computeOp_.at(0).opConsts.count(\
             \"samv-wsllen\") — opConsts is not a field of v1::DscComputeOp"
        )
    }

    /// `constantInfo_.count(maskingConstId_)` as the constant it then names (`:101`) — ⭐ ANSWERED
    /// off [`DdcFacts`], and the `count` is the whole of it: an id the table does NOT hold is
    /// [`None`], exactly as a `maskingConstId_` of `-1` is.
    fn masking_constant(&self) -> Option<ConstIdx> {
        self.facts().with_dsc(|dsc| {
            let named = dsc.ddc.masking_const?;
            dsc.ddc.constants.contains_key(&named).then_some(named)
        })
    }

    /// `sdsc_->numWkSlicesPerDim_.at(dim) > 1` (`dsc/superdsc.h:69`) — ⭐ THE SUPER-DSC'S OWN MAP,
    /// copied into the state because `run_v1` holds the super-DSC as `&mut`.
    fn splits_across_cores(&self, dim: PrimaryDim) -> bool {
        self.state
            .wk_slices(dim)
            .is_some_and(|count| count.get() > 1)
    }

    /// `dimToSymbolMapping_.count(dim)` (`dsc/designSpaceConfig.h:77`) — ⭐ ANSWERED off
    /// [`DdcFacts::dim_to_symbol`], and it is the `count` and not the entry: a dim the map NAMES is
    /// symbolic whatever its symbol list holds.
    fn is_symbolic(&self, dim: PrimaryDim) -> bool {
        self.facts()
            .with_dsc(|dsc| dsc.ddc.dim_to_symbol.contains_key(&dim))
    }

    /// `getNonBroadcastLdsDimSet(lds)` (`dsc/dsc2.cpp:4050`).
    fn non_broadcast_dims(&self, lds: LdsIdx) -> BTreeSet<PrimaryDim> {
        self.facts()
            .with_dsc(|dsc| ddc_state::non_broadcast_dims_of(dsc, lds))
    }

    /// `labeledDs_.at(lds).dataFormat_` — ⭐ ANSWERED off [`LdsRecord::data_format`].
    ///
    /// ⛔ THE TWO ABSENCES ARE ONE HERE AND THE TRAIT SAYS SO: its return is already
    /// `Option<DataFormat>`, whose [`None`] is `DataFormats::INVALID` — and a labelled DS the list does
    /// not hold has no format either.
    fn lds_format(&self, lds: LdsIdx) -> Option<DataFormat> {
        self.facts()
            .with_lds(lds, |held| held.record().data_format)
            .flatten()
    }
}

impl v1::Placement for Dsc2Reads<'_, '_> {
    /// `coreIdsUsed_`.
    fn cores_used(&self) -> v1::CoresUsed {
        self.facts().with_dsc(|dsc| {
            let used = &dsc.core_ids_used;
            v1::CoresUsed::new(used.first(), used.iter().skip(1).collect())
        })
    }

    /// `numCoreletsUsed_DSC2_` as the corelets it names — ⛔ EMPTY BEFORE ENTRY 308 STATES IT, which
    /// is the `-1` a DSC is built with; the reference SIZES A VECTOR with that `-1`
    /// (`L3DlOpsScheduler.cpp:4844`), so an empty list is the honest reading of an unprepared DSC.
    fn corelets_used(&self) -> Vec<Corelet> {
        self.facts()
            .corelets_dsc2()
            .map(ddc_state::corelets_of)
            .unwrap_or_default()
    }

    /// `numCoreletsUsed_` — ⚠️ A DIFFERENT COUNT, and entry 258 reads both.
    fn corelets_used_total(&self) -> Vec<Corelet> {
        ddc_state::corelets_of(self.facts().with_dsc(|dsc| dsc.corelets_used.get()))
    }

    /// ⛔ NEVER A CONSTANT. `getBufferCapacityForNode(node, ldsIdx, comp, corelet, row)`
    /// (`dsc/dsc2.cpp:3977`) walks the allocate node's layout against the DSC's stick sizes and
    /// rounds to an even stick count. [`crate::schedule::l3::dsc::DesignSpaceConfig::lx_chunk_capacity`]
    /// is that SAME call already made for the CHUNK stage and one specific site, so reusing it here
    /// would answer a different question with the same number.
    fn buffer_capacity(
        &self,
        _alloc: AllocId,
        _lds: Option<LdsIdx>,
        _at: Option<(Corelet, Row)>,
    ) -> Bytes {
        todo!(
            "v1::Placement::buffer_capacity: wants getBufferCapacityForNode(node, ldsIdx, comp, \
             corelet, row) (dsc/dsc2.cpp:3977) over the live super-DSC's stick sizes — \
             lx_chunk_capacity is the same call at ONE site and answers a different question"
        )
    }

    /// `labeledDs_.at(lds).scaledLdsCategory_ == SCALE_TENSOR` — ⭐ THE ONE ARM
    /// [`crate::schedule::ddc::fold::MxScaleTensor`] PROVES, so this is answerable where
    /// [`v1::ExploreDsc::scaled_category`]'s three-way is not.
    fn is_scale_tensor(&self, lds: LdsIdx) -> bool {
        self.facts()
            .with_lds(lds, crate::schedule::l3::dsc::LabeledDs::scale_tensor)
            .flatten()
            .is_some()
    }

    /// `l0TetheredMode_` (`dsc/designSpaceConfig.h:117`) — ⭐ ANSWERED off
    /// [`DdcFacts::l0_tethered`], which the conversion states as the field's own declared `false`
    /// because scratchy emits no `l0TetheredMode_` at all.
    fn l0_tethered(&self) -> v1::L0Tethered {
        self.facts().with_dsc(|dsc| dsc.ddc.l0_tethered)
    }

    /// ⛔ Wants `coreIdToTetheredCoreCoord(core).subcoreId`.
    fn subcore(&self, _core: crate::units::Core) -> u32 {
        todo!(
            "v1::Placement::subcore: wants coreIdToTetheredCoreCoord(core).subcoreId — a \
             sys-arch-spec topology lookup, not a super-DSC field, and it multiplies into an L0 \
             address"
        )
    }

    /// `{coreFoldProp_, coreletFoldProp_} ++ sdscFoldProps_`'s size, which is
    /// [`AddressFoldCoords::depth`] — ⭐ THE SAME ONE VALUE stage 2a's
    /// [`super::Placement`](super::carriers::Placement) answers from, so the fold space this places
    /// along and the one entry 292 walks cannot disagree.
    fn address_fold_depth(&self) -> usize {
        self.coords.depth()
    }
}

impl v1::StageSizes for Dsc2Reads<'_, '_> {
    /// `primaryDimToVal_st(dim, comp, -1, corelet, padding, density)` —
    /// [`StageDims::sampled_extent`](crate::schedule::l3::dsc::StageDims::sampled_extent) (entry
    /// 012) with `ptrowId = -1`, this [`PadType`] as the dim's own padding form and the density as
    /// its divisor.
    ///
    /// ⛔ THE UNSTATED SLOT IS THE REFERENCE'S OWN `-1` AND NOT A REFUSAL, so `plain_slot` still
    /// answers it. Only what the FOLD stops on — a corelet the split does not name, an aborted
    /// `calculate_padded` — is a `todo!`, because a substituted extent is a fabricated one.
    fn dim_extent(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
        unit: SenComponent,
        corelet: Option<Corelet>,
        padding: PadType,
        density: v1::Density,
    ) -> Extent {
        if let Some(half) = self.steady(stage) {
            let mut form = PaddingForm::default();
            form.set_padding(dim, padding);
            let at = v1::DimSample {
                comp: v1::sampled_as(unit),
                row: None,
                corelet,
            };
            // `dimDensity` is `1.0` or `1.0 / mxInfo_.blkSize`, so the block is the divisor and a
            // block of one is the identity the reference's `1.0` multiply is.
            let density = ScaleBlock::of(Cardinality(density.get().get()));
            if let Some(extent) = half.dims.sampled_extent(dim, at, &form, density, false) {
                return extent;
            }
            if let Some(extent) = plain_slot(&half, dim) {
                return extent;
            }
        }
        todo!(
            "v1::StageSizes::dim_extent: primaryDimToVal_st (dsc/dims.cpp:653-704) STOPS for stage \
             {stage:?} dim {dim:?} unit {unit:?} corelet {corelet:?} padding {padding:?} — either \
             dataStageParam_ states no such stage, or the split names that corelet nowhere, or \
             calculate_padded (:563-616) aborted"
        )
    }

    /// `dataStageDimToVal_compView_st(dim, unit, corelet, padding, density)` — ⛔ IT DERIVES THE PT
    /// ROW FROM THE COMPONENT (`EnumsConversion::senCompToRowId`, `dsc/dims.cpp:707-714`) and then
    /// calls `primaryDimToVal_st` with it, so a sampled row is always in play.
    fn comp_view_scaled(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
        unit: SenComponent,
        corelet: Option<Corelet>,
        padding: PadType,
        density: v1::Density,
    ) -> Extent {
        let plain = corelet.is_none()
            && padding == PadType::NoPad
            && density == v1::Density::FULL;
        if plain
            && let Some(half) = self.steady(stage)
            && half.dims.row_split.is_empty()
            && !matches!(unit, SenComponent::Pe | SenComponent::Sfp | SenComponent::Pelrf | SenComponent::Sfplrf)
            && let Some(extent) = plain_slot(&half, dim)
        {
            return extent;
        }
        todo!(
            "v1::StageSizes::comp_view_scaled: wants dataStageDimToVal_compView_st \
             (dsc/dims.cpp:707-714), which derives ptrowId from {unit:?} via \
             EnumsConversion::senCompToRowId and then folds rowSplit_/peSfpSplit_/coreletSplit_ for \
             stage {stage:?} dim {dim:?}"
        )
    }

    /// `lds.scale_.at(getDimIndexInLayoutOrder(dsType_, dim))` — ⛔ [`None`] IS
    /// `getDimIndexInLayoutOrder < 0`, a dim this ds type's layout order does not name, which is
    /// what the trait's own doc distinguishes entry 259's throw from entry 260's `1`.
    fn lds_scale(&self, lds: LdsIdx, dim: PrimaryDim) -> Option<LdsScale> {
        self.facts()
            .with_dsc(|dsc| {
                let held = dsc.labeled_ds.at(lds)?;
                dsc.layout_dims
                    .get(&lds)?
                    .iter()
                    .any(|named| named == dim)
                    .then(|| held.scale(dim))
                    .flatten()
            })
            .map(|scale| match scale {
                crate::schedule::ddc::transformation::Scale::StickDim => LdsScale::StickBroadcast,
                crate::schedule::ddc::transformation::Scale::UnitStick => LdsScale::Broadcast,
                crate::schedule::ddc::transformation::Scale::Sized(scale) if scale <= 0.0 => {
                    LdsScale::Broadcast
                }
                crate::schedule::ddc::transformation::Scale::Sized(scale)
                    if (scale - 1.0).abs() < f64::EPSILON =>
                {
                    LdsScale::Unscaled
                }
                crate::schedule::ddc::transformation::Scale::Sized(_) => LdsScale::Scaled,
            })
    }

    /// `1.0 / mxInfo_.blkSize` on a scale tensor's mx dim, [`v1::Density::FULL`] everywhere else.
    fn dim_density(&self, lds: LdsIdx, dim: PrimaryDim) -> v1::Density {
        match self
            .facts()
            .with_lds(lds, crate::schedule::l3::dsc::LabeledDs::scale_tensor)
            .flatten()
        {
            Some(held) if held.dim == dim => NonZeroU64::new(held.blk_size.count().0)
                .map_or(v1::Density::FULL, v1::Density::per_block),
            _ => v1::Density::FULL,
        }
    }

    /// `dsNode.coreletSplit_.at(dim)` — ⭐ THE SHARED MAP, read raw.
    fn corelet_split(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Vec<Elements>> {
        self.steady(stage)?
            .dims
            .corelet_split
            .get(&dim)
            .map(|shares| {
                shares
                    .iter()
                    .map(|extent| Elements(extent.0.unsigned_abs()))
                    .collect()
            })
    }

    /// ⛔ Wants `allocNode->paddingSizes_.at(dim)` (`dsc/dsc2.h:1000`) — ⚠️ NOT the `padding_`
    /// [`crate::schedule::l3::dl_ops::L3AllocateNode`] carries: `paddingSizes_` is the five-number
    /// per-dim record and `padding_` is the [`PadType`] form.
    fn alloc_padding_sizes(&self, _alloc: AllocId, _dim: PrimaryDim) -> Option<v1::PaddingSizes> {
        todo!(
            "v1::StageSizes::alloc_padding_sizes: wants allocNode->paddingSizes_.at(dim) \
             (dsc/dsc2.h:1000) — the ALLOCATION's five-number record, which \
             l3::dl_ops::L3AllocateNode does not carry (it holds padding_, the PadType form)"
        )
    }

    /// `dataStageParam_.at(stage).ss_.paddingSizes_.at(dim)` — ⭐ THE SHARED MAP, converted field for
    /// field from [`crate::schedule::l3::dsc::DimPadding`].
    ///
    /// ⛔ [`None`] ALSO WHERE THE ENTRY NAMES NO WINDOW DIM: [`v1::PaddingSizes::window_dim`] is a
    /// bare [`PrimaryDim`] with no absent state, and the reference's `PrimaryDimTypesCount` IS the
    /// absence (`dsc/dims.cpp:583`), so a substituted dim would name a window that does not exist.
    fn stage_padding_sizes(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
    ) -> Option<v1::PaddingSizes> {
        let held = *self.steady(stage)?.dims.padding.get(&dim)?;
        // ⛔ `Voided` IS THE REFERENCE'S `padFront_ = padBack_ = -1` — a chunked dim whose padding
        // belongs to no chunk. [`v1::PaddingSizes`] has no such state, and `calculate_padded` reads
        // that `-1` as *"Padded access is not valid in datastage"* (`dsc/dims.cpp:582-584`), so it
        // is an absence here rather than a substituted zero.
        let (front, back) = match held.sizes {
            crate::schedule::l3::dsc::PadSizes::Unpadded => (Elements(0), Elements(0)),
            crate::schedule::l3::dsc::PadSizes::Sized { front, back } => {
                (Elements(u64::from(front.0)), Elements(u64::from(back.0)))
            }
            crate::schedule::l3::dsc::PadSizes::Voided => return None,
        };
        Some(v1::PaddingSizes {
            window_dim: held.window_dim?,
            stride: NonZeroU64::new(held.stride.get().unsigned_abs())?,
            dilation: NonZeroU64::new(held.dilation.0.unsigned_abs())?,
            pad_front: front,
            pad_back: back,
        })
    }

    /// ⛔ Wants `getSizeDataStageForNode(node, node)` (`dsc/designSpaceConfig.h:264`).
    fn size_stage(&self, _alloc: AllocId) -> DatastageId {
        todo!(
            "v1::StageSizes::size_stage: wants getSizeDataStageForNode(node, node) \
             (dsc/designSpaceConfig.h:264) — WHICH dataStageParam_ entry sizes this allocation; a \
             substituted id would size every buffer off the wrong stage"
        )
    }

    /// SOME compute op whose `opFuncName` is `GENERIC_PARTIAL_REDUCTION` takes exactly this lds as
    /// its only input — ⭐ ANSWERED FROM `computeOp_`, which the state carries.
    ///
    /// ⚠️ THE SWEEP'S OWN `DT_CHECK(inputLabeledDs.size() == 1)` IS SWALLOWED, exactly as the
    /// trait's doc says it is: a `bool` has nowhere to say a partial-reduction op had a different
    /// input count.
    fn is_sole_partial_reduction_input(&self, lds: LdsIdx) -> bool {
        self.facts().ops().iter().any(|op| {
            op.op_func == Some(OpFunc::GenericPartialReduction)
                && op.inputs.len() == 1
                && op.inputs.first() == Some(&lds)
        })
    }
}

impl v1::OffsetSizes for Dsc2Reads<'_, '_> {
    /// `labeledDs_.at(lds).memOrg_.at(storage).allocateNode_` — ⭐ ANSWERED OFF THE TREE'S OWN
    /// `memOrg_`, which is where stage 2a filed every allocation it minted.
    ///
    /// ⛔ [`None`] IS THE `DT_ERROR` *"does not have memOrg_ entry for DataLocation storge"*.
    fn lds_alloc(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId> {
        let tree = self.state.tree(self.dsc)?;
        let node = tree.org(lds)?.node(storage)?;
        tree.with(|held| held.allocate(node).map(|(alloc, _)| alloc))
    }

    /// `constantInfo_.at(constant).allocations_.at(storage)` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::ConstantInfo::allocations`].
    ///
    /// ⛔ [`None`] IS EITHER `.at()`'s THROW, AND THE TRAIT'S RETURN ALREADY IS ONE: an id no
    /// `constantInfo_` entry is filed under, or a component no allocation of that constant names. It is
    /// EMPTY on every constant scratchy writes (`"allocations_": {}` on all twelve of `g0/`), and
    /// `ComponentAllocations::set_constant_allocation` is what fills it — so absence here is the
    /// unplaced state and not a lost fact.
    fn const_alloc(&self, constant: ConstIdx, storage: SenComponent) -> Option<AllocId> {
        self.facts().with_dsc(|dsc| {
            dsc.ddc
                .constants
                .get(&constant)?
                .allocations
                .get(&storage)
                .copied()
        })
    }

    /// ⛔ Wants `dscGlobal.sysDef.addressGranularityScalePerUnit.at({generic, storage})` — a
    /// SYSTEM-DEFINITION table handed to the scheduler's constructor
    /// (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29`), not a super-DSC fact.
    fn address_scale(&self, _unit: GenericComp, _storage: SenComponent) -> Option<NonZeroU64> {
        todo!(
            "v1::OffsetSizes::address_scale: wants \
             dscGlobal.sysDef.addressGranularityScalePerUnit.at({{generic, storage}}) — the \
             scheduler's own construction argument, and the DIVISOR of every placed address \
             (DT_CHECK(addrScale > 0), ddc/ddcv1.cpp:2415)"
        )
    }

    /// `dataStageParam_.at(stage).ss_.paddingSizes_` — ⭐ THE SHARED MAP, every padded dim it holds.
    fn stage_padding_dims(&self, stage: DatastageId) -> Vec<(PrimaryDim, v1::PaddingSizes)> {
        let Some(half) = self.steady(stage) else {
            return Vec::new();
        };
        half.dims
            .padding
            .keys()
            .filter_map(|dim| {
                v1::StageSizes::stage_padding_sizes(self, stage, *dim).map(|sizes| (*dim, sizes))
            })
            .collect()
    }

    /// `dataStageParam_.at(stage).ss_.symbolicDimInfo_.count(dim)` — ⭐ THE SHARED MAP.
    fn has_symbolic_dim(&self, stage: DatastageId, dim: PrimaryDim) -> bool {
        self.steady(stage)
            .is_some_and(|half| half.dims.symbolic.info().contains_key(&dim))
    }

    /// `dataStageParam_.at(stage).ss_.peSfpSplit_` — ⭐ THE SHARED MAP.
    fn pe_sfp_split_dims(&self, stage: DatastageId) -> Vec<PrimaryDim> {
        self.steady(stage)
            .map(|half| half.dims.pe_sfp_split.keys().copied().collect())
            .unwrap_or_default()
    }

    /// ⛔ Wants `getBlockTransferSizePerDim(transfer, unit, corelet)[dim]`.
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
             accessor over that TRANSFER node and the DSC's stick sizes, a `dsc/` seam"
        )
    }

    /// ⛔ Wants `loopDistributionParamInfo.at(node).at(alloc).at(loop).at(dim)`'s
    /// `temporalStridePostDistribution` — the METADATA's fold-distribution table, which entry 292
    /// fills and which no carrier owns.
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

/// ⭐ THE PLAIN FIELD READ, AT FULL DENSITY AND NO PADDING — `primaryDimToVal_base_st`
/// (`dsc/dims.cpp:516-560`), the ONE case the authority provably returns the stored slot for. Kept as
/// a free function beside its two callers so both stand on the same citation.
fn plain_slot(half: &Dsc2Dims, dim: PrimaryDim) -> Option<Extent> {
    half.raw_slot(dim, PadType::NoPad, v1::SymbolicRead::Max)
}
