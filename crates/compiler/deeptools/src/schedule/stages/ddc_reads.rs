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
use crate::schedule::ddc::fold::{AllocId, ConstIdx, NodeId, PadType, ScaleBlock, ScaledLds};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
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
            .dsc()
            .layout_dims
            .get(&lds)
            .cloned()
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
        self.facts()
            .dsc()
            .labeled_ds
            .iter()
            .map(crate::schedule::l3::dsc::LabeledDs::recorded)
            .collect()
    }
}

impl v1::LdsSticks for Dsc2Reads<'_, '_> {
    /// `primaryDsInfo_.at(labeledDs_.at(lds).dsType_)`'s stick order zipped with its sizes.
    fn stick_dims(&self, lds: LdsIdx) -> StickDims {
        ddc_state::stick_dims_of(self.facts().dsc(), lds).unwrap_or_default()
    }
}

impl v1::StorageNames for Dsc2Reads<'_, '_> {
    /// ⛔ Wants `labeledDs_.at(lds).dsName_` — the SAME gap stage 2a's
    /// [`super::Placement`](super::carriers::Placement) records, and the reason
    /// [`super::DscState::seeded`] names its own allocate nodes positionally.
    fn lds_name(&self, _lds: LdsIdx) -> v1::StorageName {
        todo!(
            "v1::StorageNames::lds_name: wants labeledDs_.at(lds).dsName_, which \
             l3::dsc::LabeledDs does not carry — present in g0/sdsc_0.json, dropped by the l3 \
             projection. It NAMES a tracker entry, so a positional stand-in would key the memory \
             tracker by a name the reference never used."
        )
    }

    /// ⛔ Wants `constantInfo_.at(constant).name_` — `constantInfo_` is not a field of
    /// [`crate::schedule::l3::dsc::DesignSpaceConfig`] at all.
    fn constant_name(&self, _constant: ConstIdx) -> v1::StorageName {
        todo!(
            "v1::StorageNames::constant_name: wants constantInfo_.at(constant).name_ — \
             constantInfo_ is not projected onto l3::dsc::DesignSpaceConfig"
        )
    }
}

impl v1::ExploreDsc for Dsc2Reads<'_, '_> {
    /// `lds < labeledDs_.size()`.
    fn holds_lds(&self, lds: LdsIdx) -> bool {
        self.facts().dsc().labeled_ds.at(lds).is_some()
    }

    /// `labeledDs_.at(lds).scaledLdsCategory_` — ⭐ THE `SCALE_TENSOR` ARM IS PROVED:
    /// [`crate::schedule::ddc::fold::MxScaleTensor::of`] answers [`Some`] for that category ALONE
    /// (`ddc/fold.rs:2707-2712`).
    ///
    /// ⛔ ITS ABSENCE COLLAPSES THE OTHER TWO. `REGULAR_TENSOR` and `VALUE_TENSOR` are both a `None`
    /// scale tensor, and entry 307 branches on which — so the absence is named rather than guessed.
    fn scaled_category(&self, lds: LdsIdx) -> ScaledLds {
        match self
            .facts()
            .dsc()
            .labeled_ds
            .at(lds)
            .and_then(crate::schedule::l3::dsc::LabeledDs::scale_tensor)
        {
            Some(_) => ScaledLds::Scale,
            None => todo!(
                "v1::ExploreDsc::scaled_category: l3::dsc::LabeledDs projects \
                 scaledLdsCategory_ ONLY as Option<MxScaleTensor>, which collapses REGULAR_TENSOR \
                 and VALUE_TENSOR into one absence — and entry 307 branches on which"
            ),
        }
    }

    /// `labeledDs_.at(lds).mxInfo_`'s dim and block size.
    fn mx_info(&self, lds: LdsIdx) -> Option<(PrimaryDim, ScaleBlock)> {
        self.facts()
            .dsc()
            .labeled_ds
            .at(lds)
            .and_then(crate::schedule::l3::dsc::LabeledDs::scale_tensor)
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

    /// ⛔ Wants `constantInfo_.count(maskingConstId_)`.
    fn masking_constant(&self) -> Option<ConstIdx> {
        todo!(
            "v1::Masking::masking_constant: wants constantInfo_.count(maskingConstId_) \
             (dsc/designSpaceConfig.h:101) — neither field is projected onto l3::dsc"
        )
    }

    /// `sdsc_->numWkSlicesPerDim_.at(dim) > 1` (`dsc/superdsc.h:69`) — ⭐ THE SUPER-DSC'S OWN MAP,
    /// copied into the state because `run_v1` holds the super-DSC as `&mut`.
    fn splits_across_cores(&self, dim: PrimaryDim) -> bool {
        self.state
            .wk_slices(dim)
            .is_some_and(|count| count.get() > 1)
    }

    /// ⛔ Wants `dimToSymbolMapping_.count(dim)` (`dsc/designSpaceConfig.h:77`).
    fn is_symbolic(&self, _dim: PrimaryDim) -> bool {
        todo!(
            "v1::Masking::is_symbolic: wants dimToSymbolMapping_.count(dim) \
             (dsc/designSpaceConfig.h:77) — not projected onto l3::dsc::DesignSpaceConfig, and it \
             is one of the fields stage 2b WRITES (dimToSymbolMapping_ in g0/debug/sdsc_0/sdsc.json)"
        )
    }

    /// `getNonBroadcastLdsDimSet(lds)` (`dsc/dsc2.cpp:4050`).
    fn non_broadcast_dims(&self, lds: LdsIdx) -> BTreeSet<PrimaryDim> {
        ddc_state::non_broadcast_dims_of(self.facts().dsc(), lds)
    }

    /// ⛔ Wants `labeledDs_.at(lds).dataFormat_`.
    fn lds_format(&self, _lds: LdsIdx) -> Option<DataFormat> {
        todo!(
            "v1::Masking::lds_format: wants labeledDs_.at(lds).dataFormat_, which l3::dsc::LabeledDs \
             does not carry — present in g0/sdsc_0.json, dropped by the l3 projection"
        )
    }
}

impl v1::Placement for Dsc2Reads<'_, '_> {
    /// `coreIdsUsed_`.
    fn cores_used(&self) -> v1::CoresUsed {
        let used = &self.facts().dsc().core_ids_used;
        v1::CoresUsed::new(used.first(), used.iter().skip(1).collect())
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
        ddc_state::corelets_of(self.facts().dsc().corelets_used.get())
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
            .dsc()
            .labeled_ds
            .at(lds)
            .and_then(crate::schedule::l3::dsc::LabeledDs::scale_tensor)
            .is_some()
    }

    /// ⛔ Wants `l0TetheredMode_` (`dsc/designSpaceConfig.h`).
    fn l0_tethered(&self) -> v1::L0Tethered {
        todo!(
            "v1::Placement::l0_tethered: wants l0TetheredMode_, not projected onto \
             l3::dsc::DesignSpaceConfig — it DECIDES whether the L0 address is per-subcore, so \
             either arm chosen here is a placement decision"
        )
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
    /// `primaryDimToVal_st(dim, comp, -1, corelet, padding, density)` — ⭐ THE SHARED MAP, read in
    /// the ONE case the authority's own control flow is a plain field read; see
    /// [`super::ddc_state::Dsc2Dims`].
    fn dim_extent(
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
            && density == v1::Density::FULL
            && !matches!(unit, SenComponent::Pe | SenComponent::Sfp | SenComponent::Pelrf | SenComponent::Sfplrf);
        if plain
            && let Some(half) = self.steady(stage)
            && let Some(extent) = plain_slot(&half, dim)
        {
            return extent;
        }
        todo!(
            "v1::StageSizes::dim_extent: wants primaryDimToVal_st (dsc/dims.cpp:653-704) to fold \
             coreletSplit_/rowSplit_/peSfpSplit_, calculate_padded (:563-616) and the density \
             divide for stage {stage:?} dim {dim:?} unit {unit:?} — a hand-rolled fold is a \
             fabricated extent"
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
            && half.row_split.is_empty()
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
        let dsc = self.facts().dsc();
        let held = dsc.labeled_ds.at(lds)?;
        dsc.layout_dims
            .get(&lds)?
            .iter()
            .any(|named| named == dim)
            .then(|| held.scale(dim))
            .flatten()
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
            .dsc()
            .labeled_ds
            .at(lds)
            .and_then(crate::schedule::l3::dsc::LabeledDs::scale_tensor)
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

    /// ⛔ Wants `constantInfo_.at(constant).allocations_.at(storage)`.
    fn const_alloc(&self, _constant: ConstIdx, _storage: SenComponent) -> Option<AllocId> {
        todo!(
            "v1::OffsetSizes::const_alloc: wants constantInfo_.at(constant).allocations_.at(\
             storage) — constantInfo_ is not projected onto l3::dsc::DesignSpaceConfig"
        )
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
            .map(|half| half.pe_sfp_split.keys().copied().collect())
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
