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
//! ⚠️ AND THE LIST OF WHAT IS NOT ANSWERED IS **TWO** FIELDS, NOT SEVEN — the seven this header used
//! to name include `dsName_`, `dataFormat_`, `constantInfo_`, `dimToSymbolMapping_` and
//! `l0TetheredMode_`, every one of which [`v1::StorageNames`], [`v1::Masking`] and
//! [`v1::Placement`] below now read off [`crate::schedule::l3::dsc::LdsRecord`] and
//! [`crate::schedule::l3::dsc::DdcFacts`]. A stale refusal list is what sends the next reader
//! looking for a fact that is already in hand, so it is corrected rather than annotated.
//!
//! Not answered, each naming its field:
//!
//! - `coordinateMasking_` (`dsc/designSpaceConfig.h:100`) — ⛔ A CONVERSION GAP AND NOT A DATA GAP,
//!   and this one is checkable: the SuperDSC wire type CARRIES it
//!   (`lower_subtile_tape_to_superdsc.rs:1249`, `coordinateMasking_: BTreeMap<String, Vec<[i64; 2]>>`)
//!   and the l3 projection drops it (`superdsc_to_l3_sdsc.rs:614`). ⛔ SO THE EMPTY MAP IS NOT OURS
//!   TO ASSUME: it is a real state of the field (entry 262's first arm) *and* the state today's
//!   emitter writes (`:5294`), and the emitter's own guard already says the day it stops being so —
//!   *"emit masking or pad K"* (`:8304-8310`). Answering empty from here pins "no masking" for a DSC
//!   that fills it, which emits a program reading the unmasked tail.
//! - `opConsts` — the `DscComputeOp` map (wire: `lower_subtile_tape_to_superdsc.rs:1083`,
//!   `Option<serde_json::Value>`), which [`v1::DscComputeOp`] does not project either.
//!
//! ⭐ BOTH ARE ONE FIELD EACH ON A TYPE OUTSIDE THIS FILE — [`crate::schedule::l3::dsc::DdcFacts`]
//! and [`v1::DscComputeOp`] — plus the conversion arm that fills it. Neither is a scheduler fact
//! this carrier could derive.

use core::num::NonZeroU64;
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Arch, Bytes, Elements, Target};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickDims,
};
use crate::formats::DataFormat;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{
    AllocId, Cardinality, ConstIdx, NodeId, PadType, ScaleBlock, ScaledLds, comp_row_id,
};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::PaddingForm;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{LayoutDims, LdsIdx, LdsScale};
use crate::schedule::l3::dl_ops::{AddressFoldCoords, UNSTATED_EXTENT};
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

    /// `coreIdToTetheredCoreCoord(core).subcoreId` — ⭐ ANSWERED, AND IT IS ONE `%`:
    /// `coreCoord.subcoreId = flattenedSubcore % tetheredCoreUnitSize`
    /// (`sys-arch-spec/sysdef.cpp:573-578`), whose divisor is the arch's own
    /// [`Arch::TETHERED_CORE_UNIT`] — `1` before SEN1P5 and `2` from it (`sysdef.cpp:200-205`).
    ///
    /// ⭐ NOT A CONSTRUCTION ARGUMENT AFTER ALL, WHICH IS WHY IT IS NOT SEVERED LIKE
    /// [`v1::OffsetSizes::address_scale`]: the other two `SenSystemDef` reads this file wants are
    /// TABLES keyed by things the arch does not state, while this one is a pure function of the core
    /// id and a number [`crate::arch`] already declares. [`crate::units::Core`] is
    /// `CoreId<{ Target::CORES }>`, so the id in hand is already THIS arch's — reading the divisor
    /// off any other arch would be reading it for a core that does not exist.
    ///
    /// ⛔ `1` IS THE UNTETHERED ANSWER AND NOT AN ABSENCE: every core on DD2 is subcore `0`, which
    /// is what `tetheredCoreUnitSize = 1` says, and entry 258 reads that as *"the top half of L0"*.
    /// ⛔ THE UNTETHERED ARM IS SPELLED OUT RATHER THAN LEFT TO `% 1`, and not for style: on an arch
    /// whose `TETHERED_CORE_UNIT` is `1` — DD2, per `sys-arch-spec/sysdef.cpp:200-205` — the modulo is
    /// a constant `0`, which clippy's `modulo_one` denies at the crate's `-D warnings` gate. Writing
    /// the arm keeps the answer IDENTICAL for every arch (`x % 1 == 0` for all `x`) while leaving the
    /// divisor a real one from SEN1P5 on, and it does it without an `#[allow]`, which this crate bans.
    fn subcore(&self, core: crate::units::Core) -> u32 {
        let unit = Target::TETHERED_CORE_UNIT;
        if unit <= 1 { 0 } else { core.get() % unit }
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
    ///
    /// ⛔⛔ AND THAT FALL-THROUGH IS GATED, WHICH IT WAS NOT: `plain_slot` reads at
    /// [`PadType::NoPad`], so taking it for a PADDED read answered the caller's padded extent with
    /// the dim's UNPADDED one. `calculate_padded` aborts for a padded dim `paddingSizes_` names no
    /// entry for (*"Padded dimension without padding sizes information in datastage"*,
    /// `dsc/dims.cpp:575-578`), and that stop was reachable straight through here. What DOES fall
    /// through under a pad type is the reference's own `-1`: `calculate_padded` short-circuits on
    /// `val < 0` BEFORE any abort (`:567-568`), so an UNSTATED slot answers `-1` whatever the
    /// padding — which is exactly [`UNSTATED_EXTENT`] and nothing else.
    ///
    /// ⚠️ RECORDED DIVERGENCE ON THE UNSTATED SLOT AT A DENSITY BELOW ONE: the reference scales the
    /// `-1` as a `double` and truncates, so `primaryDimToVal_base_st` reaches `calculate_padded`
    /// with `0` rather than `-1` (`:557-559`) and then either returns `0` or aborts on the padded
    /// arm. `plain_slot` ignores the density and answers `-1`, which is the absence sentinel every
    /// consumer skips (`isValidDimParam` is `param > 0.0`) where `0` is a zero-sized buffer.
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
            if let Some(extent) = plain_slot(&half, dim)
                && (padding == PadType::NoPad || extent == UNSTATED_EXTENT)
            {
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
    ///
    /// ⭐⭐ WHICH IS THE WHOLE OF IT, AND IT IS [`v1::StageSizes::dim_extent`]'S BODY WITH THE ROW
    /// FILLED IN: `rowId = senCompToRowId.count(comp) ? .at(comp) : -1` and then the SAME
    /// `primaryDimToVal_st` — already ported as
    /// [`StageDims::sampled_extent`](crate::schedule::l3::dsc::StageDims::sampled_extent) (entry
    /// 012) — so this reads the shared datastage map rather than deferring. The narrow
    /// no-row-split/no-padding fast path that used to stand here answered a STRICT SUBSET of the
    /// same question and refused the `rowSplit_` fold that is this method's entire reason to exist.
    ///
    /// ⛔ THE ROW IS [`sampled_row`]'S, NOT `Row::checked`'S DIRECTLY, and its outer [`None`] is a
    /// stop rather than the reference's `-1`; see its own doc for why the two cannot be folded.
    ///
    /// ⛔ THE `plain_slot` FALL-THROUGH IS GATED EXACTLY AS `dim_extent`'S IS — the reference's own
    /// `-1`, and a padded read may not be answered with the unpadded slot.
    fn comp_view_scaled(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
        unit: SenComponent,
        corelet: Option<Corelet>,
        padding: PadType,
        density: v1::Density,
    ) -> Extent {
        if let Some(half) = self.steady(stage)
            && let Some(row) = sampled_row(&half, dim, unit)
        {
            let mut form = PaddingForm::default();
            form.set_padding(dim, padding);
            let at = v1::DimSample {
                comp: v1::sampled_as(unit),
                row,
                corelet,
            };
            // The same divisor `dim_extent` states: `dimDensity` is `1.0` or `1.0 / mxInfo_.blkSize`.
            let density = ScaleBlock::of(Cardinality(density.get().get()));
            if let Some(extent) = half.dims.sampled_extent(dim, at, &form, density, false) {
                return extent;
            }
            if let Some(extent) = plain_slot(&half, dim)
                && (padding == PadType::NoPad || extent == UNSTATED_EXTENT)
            {
                return extent;
            }
        }
        todo!(
            "v1::StageSizes::comp_view_scaled: dataStageDimToVal_compView_st \
             (dsc/dims.cpp:707-714) STOPS for stage {stage:?} dim {dim:?} unit {unit:?} corelet \
             {corelet:?} padding {padding:?} — either dataStageParam_ states no such stage, or \
             senCompToRowId puts that unit on a row this arch does not have while rowSplit_ names \
             the dim, or the split names that corelet nowhere, or calculate_padded (:563-616) \
             aborted"
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

    // ⛔ `alloc_padding_sizes` WAS HERE AND THE TRAIT METHOD IS DELETED — it named a field that does
    // not exist. `grep paddingSizes_ dsc/dsc2.h` is ZERO hits, and `:1000` — the line the old citation
    // named — is `relatedIndirectAccessAlloc_`.
    // ⭐ `paddingSizes_` is declared on exactly ONE type, `DataStructDims` (`dsc/dims.h:219`), so every
    // read of it is a stage read BY TYPE whatever the local is named. `stage_padding_sizes` below is
    // the whole of that fact, and this method had zero production callers.

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

    /// ⛔⛔ NOT A `dataStageParam_` INDEX AT ALL, WHICH IS WHY NO CARRIER CAN ANSWER IT.
    /// `getSizeDataStageForNode(node, alloc)` (`dsc/designSpaceConfig.h:264`, defined
    /// `dsc/dsc2.cpp:3611-3616`) SYNTHESISES A FRESH `dsc2::DataStage` and returns it BY VALUE: it
    /// walks the allocation's owner loops outward, records a SEPARATE denominator stage id per
    /// relevant dim (`denDsForDim`, `:3646-3679`), and then copies each dim's extent and its
    /// `coreletSplit_`/`rowSplit_`/`peSfpSplit_` share out of THAT dim's own stage (`:3681-3700`) —
    /// plus a parametric loop's stride written straight in (`:3661-3668`) and an unconditional
    /// `ss_ = el_ = N_` for a unified HBM allocation (`:3624-3633`). The reference then reads
    /// `dsNode.primaryDimToVal_st(..)` off that VALUE (`ddc/ddcv1.cpp:1923-1924`, `:1983`).
    ///
    /// ⛔ SO THE SIGNATURE IS THE DEFECT AND NOT THE CARRIER: a [`DatastageId`] can only name one
    /// EXISTING entry, and this stage is composed from several. Every caller —
    /// [`v1::StageSizes::dim_extent`] and [`v1::StageSizes::corelet_split`] keyed by the returned id
    /// (`ddc/v1.rs:1935-1972`) — reads a per-dim answer that only the synthesised value has, so no
    /// value returned here is right for more than one dim. The fix is a `Stage` VALUE on
    /// [`v1::StageSizes`], and that is `ddc/v1.rs`' edit.
    fn size_stage(&self, _alloc: AllocId) -> DatastageId {
        todo!(
            "v1::StageSizes::size_stage: getSizeDataStageForNode(node, alloc) \
             (dsc/dsc2.cpp:3611-3700) SYNTHESISES a dsc2::DataStage per-dim from several denominator \
             stages and returns it BY VALUE — a DatastageId cannot name it, so a substituted id \
             would size every dim off one stage. The trait wants a Stage value, not an index"
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

    /// ⛔ Wants `getBlockTransferSizePerDim(transfer, unit, corelet)[dim]` — the real body is
    /// `getBlockTransferSizePerDimCustomLocation` (`dsc/dsc2.cpp:3474`, some 130 lines), which picks
    /// a REFERENCE data location by which end is the smaller unit, walks four fallbacks to find a
    /// `memOrg_` entry, and then reads the layout off `getSizeDataStageForNode(nodeForLocation,
    /// allocation)` (`:3539-3541`).
    ///
    /// ⛔ SO IT IS BLOCKED BEHIND THE SAME SEAM [`v1::StageSizes::size_stage`] IS, and the fix is the
    /// same one: that call returns a SYNTHESISED `dsc2::DataStage` by value.
    /// ⚠️ AND THE SAME FACT HAS A SECOND TRAIT SPELLING —
    /// [`crate::schedule::l3::dl_ops::DscTransferSizes::block_transfer_size_per_dim`] stands on the
    /// same citation; whichever port lands should land once.
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
    /// `temporalStridePostDistribution` — the fold-distribution table, which NO carrier owns.
    ///
    /// ⛔⛔ AND ENTRY 292 DOES NOT FILL IT, WHICH THIS DOC USED TO SAY. Entry 292 READS it
    /// (`ddc/ddcv1.cpp:2455-2460`, straight into `di.loopEleOffsets_`); the WRITER is
    /// `dsc2::computeLoopElemOffsetsFromCoordinates` (`dsc/dsc2.cpp:6694-6790`,
    /// `temporalStridePostDistribution = innerCard * temporalAlpha / allocAlpha`), called from the
    /// FOLD — `ddc/ddc_fold.cpp:1937`, `:3779`, `:4640`. So the fact is produced a whole stage
    /// earlier, by the entries this file's peer carriers drive.
    ///
    /// ⛔⛔ AND IT IS PRODUCED NOWHERE IN THIS TREE. That call is
    /// [`crate::schedule::ddc::fold::LoopElemOffsets`], declared as a CAPABILITY *"and not a drop"*
    /// (`ddc/fold.rs:5128-5145`) — and its ONLY implementor is a `#[cfg(test)]` double inside
    /// `mod tests_e356_e358` (`ddc/fold.rs:10732`). Entries 356-358's ALLOCATE arms therefore perform
    /// their second effect in tests and in no real run, so there is no table for this method to read
    /// and answering a stride here would be answering one nothing computed. A production implementor
    /// of that trait is the prerequisite, and it is not this file's.
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

/// ⭐ WHICH PT ROW A COMPONENT SAMPLES — `rowId = senCompToRowId.count(comp) ? .at(comp) : -1`
/// (`dsc/dims.cpp:712-714`), which [`comp_row_id`] already ports, narrowed to a row THIS arch has.
///
/// ⛔⛔ THE OUTER [`None`] IS A STOP AND NOT THE REFERENCE'S `-1`, AND THAT IS THE WHOLE POINT OF
/// THE TWO LAYERS. `senCompToRowId` names rows `0..7` whatever the arch's row count — the table is
/// arch-blind, which is why [`crate::schedule::ddc::fold::PtRowId`] is not a
/// [`Row`](crate::units::Row) — so a `PTROW5` on a four-row arch has an id the table states and no
/// row to be. `rowSplit_.at(d).at(ptrowId)` then indexes a per-corelet vector sized by the arch's
/// own row count and THROWS (`dsc/dims.cpp:666-673`). `Some(None)` is the reference's `-1`, which
/// SUMS every row, so answering that here would hand ONE row the whole core's extent.
///
/// ⭐ AND THE STOP IS NARROWED TO A ROW-SPLIT DIM, because `ptrowId >= 0 && rowSplit_.count(d) > 0`
/// (`:664`) is the only place the id is read: for a dim no `rowSplit_` names, the reference carries
/// that same out-of-range id straight past into the PE/SFP and corelet views without ever indexing
/// with it, and `Some(None)` reaches the identical answer there.
fn sampled_row(half: &Dsc2Dims, dim: PrimaryDim, unit: SenComponent) -> Option<Option<Row>> {
    match comp_row_id(unit) {
        // `senCompToRowId.count(comp) == 0` — the reference's own `-1`.
        None => Some(None),
        Some(named) => match Row::checked(u32::from(named.ordinal())) {
            Some(row) => Some(Some(row)),
            None if half.dims.row_split.contains_key(&dim) => None,
            None => Some(None),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{Dsc2Dims, PadType, PrimaryDim, Row, Target, plain_slot, sampled_row};
    use crate::arch::Arch;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent;
    use crate::schedule::ddc::transformation_util::PaddingForm;
    use crate::schedule::l3::dl_ops::UNSTATED_EXTENT;
    use crate::units::Corelet;
    use sys_arch_spec::arch_enums::SenComponent;

    /// THE DIM EVERY CASE BELOW READS.
    const DIM: PrimaryDim = PrimaryDim::In;

    /// One steady-state half stating `in_ = 64` and NOTHING else — no split, no padding entry.
    fn a_stated_slot() -> Dsc2Dims {
        let mut half = Dsc2Dims::default();
        half.dims.extents.insert(DIM, Extent(64));
        half
    }

    /// The same, with `rowSplit_.at(IN)` naming corelet 0's four rows.
    fn a_row_split() -> Dsc2Dims {
        let mut half = a_stated_slot();
        half.dims.row_split.insert(
            DIM,
            BTreeMap::from([(
                Corelet::checked(0).expect("every arch has corelet 0"),
                vec![Extent(16), Extent(16), Extent(16), Extent(16)],
            )]),
        );
        half
    }

    /// ⭐ WHY [`super::v1::StageSizes::dim_extent`]'S AND
    /// [`super::v1::StageSizes::comp_view_scaled`]'S `plain_slot` FALL-THROUGH IS GATED, carrying both
    /// values rather than the rule: the SAME stage answers the stored `64` to `plain_slot` while
    /// `primaryDimToVal_st` REFUSES the padded read of the same dim, so an ungated fall-through hands
    /// a padded caller the unpadded extent.
    #[test]
    fn the_unpadded_slot_is_not_an_answer_to_a_padded_read_the_reference_aborts() {
        let half = a_stated_slot();

        // `primaryDimToVal_base_st` with `NOPAD`: the stored `in_` (`dsc/dims.cpp:529-530`).
        assert_eq!(
            plain_slot(&half, DIM),
            Some(Extent(64)),
            "the stored slot, which is what `plain_slot` is for"
        );

        // ⛔ AND THE PADDED READ OF THAT SAME DIM IS THE REFERENCE'S `DT_ERROR` — *"Padded dimension
        // without padding sizes information in datastage"* (`dsc/dims.cpp:575-578`), because
        // `paddingSizes_` names no entry for it.
        let mut padded = PaddingForm::default();
        padded.set_padding(DIM, PadType::PaddedFullSpanWUnneeded);
        assert_eq!(
            half.dims.sampled_extent(DIM, super::v1::DimSample::WHOLE, &padded, None, false),
            None,
            "calculate_padded aborts for a padded dim paddingSizes_ does not name"
        );

        // ⭐ WHAT *DOES* FALL THROUGH UNDER A PAD TYPE IS THE REFERENCE'S OWN `-1`: `calculate_padded`
        // short-circuits on `val < 0` BEFORE any abort (`:567-568`), and an unstated slot is that.
        assert_eq!(
            plain_slot(&Dsc2Dims::default(), DIM),
            Some(UNSTATED_EXTENT),
            "an unstated slot is the reference's -1 under every pad type"
        );
    }

    /// ⭐ THE ROW A COMPONENT SAMPLES, AND THE STOP THAT IS NOT THE REFERENCE'S `-1`.
    #[test]
    fn a_row_this_arch_lacks_stops_only_where_the_split_would_index_with_it() {
        let split = a_row_split();
        let plain = a_stated_slot();

        // `senCompToRowId.count(LXLU) == 0` (`sys-arch-spec/arch_enums.cpp:296-320` names only the PT
        // and L0LU row spellings) — the reference's `-1`, on either stage.
        assert_eq!(
            sampled_row(&split, DIM, SenComponent::Lxlu),
            Some(None),
            "a component on no row at all is ptrowId = -1"
        );

        // `senCompToRowId.at(PTROW3) = 3` (`:299`), and row 3 exists on BOTH generations —
        // `numPTRows` is 8 then 4 (`sys-arch-spec/sysdef.cpp:224`).
        assert_eq!(
            sampled_row(&split, DIM, SenComponent::Ptrow3).map(|row| row.map(Row::get)),
            Some(Some(3)),
            "PTROW3 samples row 3 on every arch this crate emits for"
        );

        // ⛔⛔ `senCompToRowId.at(PTROW5) = 5` (`:301`) WHATEVER THE ARCH'S ROW COUNT — the table is
        // arch-blind. The two stages then part company, and that is the whole narrowing.
        let at_split = sampled_row(&split, DIM, SenComponent::Ptrow5);
        let at_plain = sampled_row(&plain, DIM, SenComponent::Ptrow5);
        if Target::PT_ROWS > 5 {
            assert_eq!(
                at_split.map(|row| row.map(Row::get)),
                Some(Some(5)),
                "an arch with 8 rows has row 5, so rowSplit_.at(IN).at(5) is a real share"
            );
            assert_eq!(
                at_plain.map(|row| row.map(Row::get)),
                Some(Some(5)),
                "and the id is carried past a dim rowSplit_ does not name, as the reference carries it"
            );
        } else {
            assert_eq!(
                at_split, None,
                "rowSplit_.at(IN).at(5) indexes a 4-row vector and THROWS (dsc/dims.cpp:666-673); \
                 `Some(None)` here would sum every row and hand one row the whole core"
            );
            assert_eq!(
                at_plain.map(|row| row.map(Row::get)),
                Some(None),
                "`ptrowId >= 0 && rowSplit_.count(d) > 0` (:664) never reads the id for this dim, so \
                 it is the reference's -1 and not a stop"
            );
        }
    }

    // ⛔ NO TEST FOR [`super::v1::Placement::subcore`], AND THE REASON IS THAT ONE WOULD BE A
    // TAUTOLOGY FROM HERE. Reaching it means a `Dsc2Reads`, which means a `Dsc2State`, which means a
    // whole `l3::dsc::SuperDsc` — `DesignSpaceConfig` has no `Default` and `super::super::stages`'
    // own `#[cfg(test)]` fixture is what builds one. Re-deriving `core.get() %
    // Target::TETHERED_CORE_UNIT` beside the method that computes it would assert the expression
    // against itself. What DOES stand behind it is checked one compilation earlier: `crate::arch`'s
    // const block asserts `Dd2::TETHERED_CORE_UNIT == 1` and
    // `Sen1p5::CORES % Sen1p5::TETHERED_CORE_UNIT == 0`, which is `sysdef.cpp:200-205` verbatim, so
    // the divisor is both non-zero and the reference's.
}
