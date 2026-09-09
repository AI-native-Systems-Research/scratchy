// SPDX-License-Identifier: Apache-2.0

//! WHAT THE L3 SCHEDULER'S UNITS TRAFFIC IN — the reduced l3 view of `SuperDsc`,
//! `DesignSpaceConfig`, `DataStructDims` and `LabeledDsInfo`.
//!
//! ⛔ NOT A SCHEDULED UNIT of the campaign, but the vocabulary its units need, on the precedent of
//! [`crate::schedule::dsc2`]: a reduced per-module projection of one C++ class is this crate's
//! idiom, and each module states the fields its own units touch and nothing else.
//!
//! ⭐⭐ EVERY `DT_CHECK` OF THIS BATCH IS A CONSTRUCTOR HERE, not a refusal later. `dscs_.size() >= 1`,
//! `!dscIndices.empty()`, `coreIdsUsed_[0]`, `!DataStructDims::empty()`, the `scale_` index bounds
//! and `maxSymbolicVolume_`'s keys being named by `symbolicDimInfo_` are each discharged once, where
//! the value is built.

use crate::arch::{Bytes, Elements};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    self as shape_constraints, Extent, PrimaryDim, StickDims, StickPart,
};
use crate::schedule::ddc::fold::Stride;
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::{DsType, Scale};
use crate::schedule::ddc::transformation_util::{PaddingForm, StageName};
use crate::schedule::dsc2::{LayoutDims, LdsIdx, NodeName};
use crate::units::{Core, Corelet};
use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU32, NonZeroU64};
use sys_arch_spec::arch_enums::SenComponent;

/// WHERE ONE LABELLED DATA STRUCTURE LIVES — `memOrg_` (`dsc/dscdefn.h:337`) reduced to the three
/// questions this stage asks of it, each already the reference's own predicate over that map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pinning {
    /// `isHbmPinned()` (`:369`) — `memOrg_.at(HBM).isPresent`.
    pub hbm: bool,
    /// `isLxPinned()` (`:424`) — an LX organisation that is neither HBM- nor XRF-pinned nor a ring.
    pub lx: bool,
    /// `memOrg_.count(LX) && memOrg_.at(LX).isPadded`.
    pub lx_padded: bool,
}

/// ONE LABELLED DATA STRUCTURE — `LabeledDsInfo` (`dsc/dscdefn.h:321`) with its `scale_` ZIPPED onto
/// the `layoutDimOrder_` of `primaryDsInfo_[dsType_]` that `getDimIndexInLayoutOrder`
/// (`dsc/designSpaceConfig.cpp:429`) indexes it by.
///
/// ⭐ THE ZIP IS THE `DT_CHECK("Invalid layoutDimOrder_ index.")`: `scaleIdx >= 0 && scaleIdx <
/// scale_.size()` cannot be asked once a dim and its scale are one entry.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledDs {
    ds_type: DsType,
    scales: Vec<(PrimaryDim, Scale)>,
    recorded: LdsIdx,
    pinning: Pinning,
}

impl LabeledDs {
    /// A labelled data structure's layout order paired with its scales, outermost first.
    #[must_use]
    pub fn new(
        ds_type: DsType,
        scales: Vec<(PrimaryDim, Scale)>,
        recorded: LdsIdx,
        pinning: Pinning,
    ) -> Self {
        Self {
            ds_type,
            scales,
            recorded,
            pinning,
        }
    }

    /// `ldsIdx_` — the entry's OWN self-index, which need NOT equal the position it sits at in
    /// `labeledDs_`: it defaults to `183` (`dsc/dscdefn.h:323`) and is written independently.
    #[must_use]
    pub fn recorded(&self) -> LdsIdx {
        self.recorded
    }

    /// `memOrg_`, as the three questions asked of it.
    #[must_use]
    pub fn pinning(&self) -> Pinning {
        self.pinning
    }

    /// `dsType_`.
    #[must_use]
    pub fn ds_type(&self) -> DsType {
        self.ds_type
    }

    /// `scale_.at(getDimIndexInLayoutOrder(dsType_, dim))`, `None` where the layout order does not
    /// name `dim` — the reference's `-1` index.
    #[must_use]
    pub fn scale(&self, dim: PrimaryDim) -> Option<Scale> {
        self.scales
            .iter()
            .find(|(entry, _)| *entry == dim)
            .map(|(_, scale)| *scale)
    }
}

/// HOW MANY CORELETS OF A CORE A DSC USES — `numCoreletsUsed_` (`dsc/designSpaceConfig.h:74`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreletsUsed(NonZeroU32);

impl CoreletsUsed {
    /// The one-corelet case scratchy emits for every sampled SuperDSC.
    pub const ONE: Self = Self(NonZeroU32::MIN);

    /// A corelet count; a DSC that uses none has no work.
    #[must_use]
    pub const fn new(count: NonZeroU32) -> Self {
        Self(count)
    }

    /// The count itself.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// Whether more than one corelet is in play — the `numCoreletsUsed_ <= 1` early-out.
    #[must_use]
    pub const fn splits(self) -> bool {
        self.0.get() > 1
    }
}

/// ONE DIM'S CORELET-0 SHARE AGAINST THE WHOLE — the pair of `primaryDimToVal_st` results
/// `isDimensionCoreletSplit` compares, whichever carrier states them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreletShare {
    /// The core data stage's `(dim, NO_COMPONENT, -1, 0)`, or `CoreletD_`'s value.
    pub corelet0: Extent,
    /// The core data stage's `(dim, NO_COMPONENT, -1, -1)`, or `CoreD_`'s value.
    pub whole: Extent,
}

impl CoreletShare {
    /// Whether corelet 0 holds less than the whole of the dim.
    #[must_use]
    pub const fn splits(self) -> bool {
        self.corelet0.0 < self.whole.0
    }
}

/// THE CORES A DSC USES, NON-EMPTY — `coreIdsUsed_`, whose first entry
/// `getLabeledDsWkSliceMulticastDegree` reaches with a bare `[0]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreIdsUsed {
    first: Core,
    rest: Vec<Core>,
}

impl CoreIdsUsed {
    /// A DSC runs on at least one core, and this is how that is stated.
    #[must_use]
    pub const fn new(first: Core, rest: Vec<Core>) -> Self {
        Self { first, rest }
    }

    /// `coreIdsUsed_[0]`.
    #[must_use]
    pub const fn first(&self) -> Core {
        self.first
    }

    /// Every core, first one first.
    pub fn iter(&self) -> impl Iterator<Item = Core> + '_ {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }
}

/// ONE PRIMARY DATA STRUCTURE — `PrimaryDsInfo` (`dsc/dscdefn.h:474`) reduced to its two dim orders,
/// ONE value because `primaryDsInfo_.at(dsType)` hands both back together and a `DsTypes` present in
/// one order and absent from the other is not a state the reference can hold.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimaryDsInfo {
    /// `layoutDimOrder_`.
    pub layout: LayoutDims,
    /// `stickDimOrder_` zipped with `stickSize_`, as [`StickDims`] carries them.
    pub stick: StickDims,
}

/// A DSC'S LABELLED DATA STRUCTURES, NON-EMPTY — `labeledDs_` (`dsc/designSpaceConfig.h:86`).
///
/// ⭐ NON-EMPTY BECAUSE THE REFERENCE NEVER GUARDS IT: the min-param units reach `.front()`/`.back()`
/// with no check, and `isLastLds` compares against the UNSIGNED `labeledDs_.size() - 1`
/// (`L3DlOpsScheduler.h:229`), which on an empty list is `SIZE_MAX`.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledDsList {
    first: LabeledDs,
    rest: Vec<LabeledDs>,
}

impl LabeledDsList {
    /// A DSC labels at least one data structure, and this is how that is stated.
    #[must_use]
    pub const fn new(first: LabeledDs, rest: Vec<LabeledDs>) -> Self {
        Self { first, rest }
    }

    /// `labeledDs_.front()`.
    #[must_use]
    pub const fn front(&self) -> &LabeledDs {
        &self.first
    }

    /// `labeledDs_.back()`.
    #[must_use]
    pub fn back(&self) -> &LabeledDs {
        self.rest.last().unwrap_or(&self.first)
    }

    /// Every entry, in `labeledDs_` order.
    pub fn iter(&self) -> impl Iterator<Item = &LabeledDs> + '_ {
        core::iter::once(&self.first).chain(self.rest.iter())
    }

    /// Every entry WITH THE POSITION IT SITS AT — the index every `labeledDs_.at(idx)` uses, which
    /// is not the [`LabeledDs::recorded`] index the entry itself carries.
    pub fn indexed(&self) -> impl Iterator<Item = (LdsIdx, &LabeledDs)> + '_ {
        (0u32..).map(LdsIdx).zip(self.iter())
    }

    /// `labeledDs_.at(idx)`, `None` past the end — that `.at()`'s throw.
    #[must_use]
    pub fn at(&self, idx: LdsIdx) -> Option<&LabeledDs> {
        self.indexed()
            .find(|(at, _)| *at == idx)
            .map(|(_, lds)| lds)
    }

    /// `isOutputLabeledDs(ldsIdx, dsc)` (`L3DlOpsScheduler.h:228`) — `ldsIdx == labeledDs_.size()
    /// - 1`, so the LAST entry is the output and a non-empty list always has one.
    ///
    /// ⛔ TRAP: IT IS ASKED OF `lds.ldsIdx_`, THE ENTRY'S OWN [`LabeledDs::recorded`] INDEX, and not
    /// of the position that entry sits at — so a recorded index that drifted from its position
    /// answers for whichever position the index names.
    #[must_use]
    pub fn is_output(&self, idx: LdsIdx) -> bool {
        idx.0 as usize + 1 == self.iter().count()
    }
}

/// ONE DESIGN SPACE CONFIG — `DesignSpaceConfig` (`dsc/designSpaceConfig.h:74`) reduced to the
/// fields this batch reads.
#[derive(Debug, Clone, PartialEq)]
pub struct DesignSpaceConfig {
    /// `numCoreletsUsed_`.
    pub corelets_used: CoreletsUsed,
    /// `numCoreletsUsed_DSC2_` (`dsc/designSpaceConfig.h:118`).
    ///
    /// ⛔⛔ [`None`] IS THE `-1` A DSC IS BUILT WITH, and `prepDsc` (entry 054) is the only thing
    /// that replaces it. The reference SIZES A `std::vector` WITH IT — `std::vector<int64_t>
    /// coreletOffsets(dsc.numCoreletsUsed_DSC2_, 0)` (`L3DlOpsScheduler.cpp:4844`) — so the
    /// unprepared state is undefined behaviour there and absence is the honest answer here.
    pub corelets_used_dsc2: Option<CoreletsUsed>,
    /// Per dim, corelet 0's share against the whole — `dataStageParam_.at(0).ss_` where the core
    /// data stage exists, else `CoreletD_` against `CoreD_`.
    pub corelet_shares: BTreeMap<PrimaryDim, CoreletShare>,
    /// `primaryDsInfo_` (`dsc/dscdefn.h:474`).
    pub primary_ds_info: BTreeMap<DsType, PrimaryDsInfo>,
    /// `coreIdsUsed_`.
    pub core_ids_used: CoreIdsUsed,
    /// `getLayoutDims(ldsIdx)` (`dsc/dsc2.cpp:4007`), per labelled data structure.
    pub layout_dims: BTreeMap<LdsIdx, LayoutDims>,
    /// `labeledDs_`.
    pub labeled_ds: LabeledDsList,
    /// `dataStageParam_` (`dsc/designSpaceConfig.h:105`).
    pub data_stages: DataStages,
    /// `computeOp_.at(0).indirectAccessIndexLabeledDs` (`dsc/dscdefn.h:511`), by index.
    pub indirect_access_index_lds: BTreeSet<LdsIdx>,
    /// `getBufferCapacityForNode(lds.memOrg_.at(LX).allocateNode_, ldsIdx, LX, -1, -1, bytesPerStick)`
    /// (`dsc/dsc2.cpp:3977`) per labelled data structure, in BYTES as the reference computes it.
    ///
    /// ⭐ ABSENCE IS THE TWO `DT_CHECK`s: `memOrg_` naming no `LX`, or its entry carrying no allocate
    /// node (`L3DlOpsScheduler.cpp:1705-1709`), are one missing entry here.
    pub lx_chunk_capacity: BTreeMap<LdsIdx, Bytes>,
}

impl DesignSpaceConfig {
    /// `dataStageParam_.at(dataStageCoreIdx).ss_` — `dataStageCoreIdx` is `0`
    /// (`L3DlOpsScheduler.cpp:275`).
    ///
    /// ⭐ MANDATORY, WHICH IS `DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx), "Expect
    /// dataStageParam_ entry for the core data stage.")` (`:353`, `:1186`, `:1197`, `:1385`, `:1630`)
    /// DISCHARGED HERE: every min-param unit reaches it with a bare `.at()`.
    /// `isDimensionCoreletSplit`'s defensive `count` (`:77`) is then a constant, and
    /// [`Self::corelet_shares`] keeps its own answer because the split it reports may come from
    /// `CoreletD_`/`CoreD_` instead.
    ///
    /// ⭐ A READ OF [`Self::data_stages`] AND NOT A FIELD OF ITS OWN: `dataStageParam_.at(0).ss_` is
    /// one fact, and a second field holding it is a second answer that can disagree.
    #[must_use]
    pub const fn core_stage(&self) -> &FilledDims {
        &self.data_stages.core().ss.dims
    }

    /// `getNonBroadcastLdsDims(ldsIdx)` (`dsc/dsc2.cpp:4039`) — `getLayoutDims`' order, filtered to
    /// the dims whose `scale_` is strictly positive.
    ///
    /// ⛔ TRAP: THIS IS NOT THE COMPLEMENT OF `isLabeledDsDimensionBroadcast`. That predicate
    /// broadcasts on `scale_ < 1` (`L3DlOpsScheduler.cpp:71`); this set keeps every `scale_ > 0`, so
    /// a fractional scale is broadcast to one and non-broadcast to the other.
    ///
    /// ⛔ [`None`] IS `getLayoutDims`' OWN `DT_CHECK` (`dsc/dsc2.cpp:4009`, `:4022`): an index past
    /// `labeledDs_`, or a labelled data structure this DSC states no allocate node for. ⛔ AND THE
    /// EMPTY ANSWER PRECEDES IT — `if (nbDimSet.empty()) return nbDims;` (`dsc/dsc2.cpp:4043`) runs
    /// BEFORE `getLayoutDims` is ever called, so a wholly broadcast structure cannot reach the abort.
    ///
    /// ⭐ THE TWO ORDERS STAY TWO. The reference builds its set from `primaryDsInfo_`'s layout order
    /// and then filters `getLayoutDims(ldsIdx)`, a DIFFERENT list; [`LabeledDs`] carries the first
    /// zipped, so the membership test is a scale lookup and a dim named by neither drops out.
    #[must_use]
    pub fn non_broadcast_lds_dims(&self, lds: LdsIdx) -> Option<Vec<PrimaryDim>> {
        let entry = self.labeled_ds.at(lds)?;
        let non_broadcast = |scale: &Scale| matches!(scale, Scale::Sized(scale) if *scale > 0.0);
        if !entry.scales.iter().any(|(_, scale)| non_broadcast(scale)) {
            return Some(Vec::new());
        }
        let layout = self.layout_dims.get(&lds)?;
        Some(
            layout
                .iter()
                .filter(|dim| entry.scale(*dim).as_ref().is_some_and(non_broadcast))
                .collect(),
        )
    }

    /// `getCumulativeStickSizes(dsType)` (`dsc/dsc2.cpp:4108`) with all four flags at their defaults,
    /// which is [`StickPart::Whole`], delegating to the ported fold.
    ///
    /// ⛔ `None` IS `primaryDsInfo_.at(dsType)` THROWING, or the `int` product the fold refuses to
    /// wrap. ⛔ AND THE REFERENCE'S `DT_CHECK(elemInSlice > 0 && elemInSlice % 8 == 0)`
    /// (`dsc/dsc2.cpp:4083`) RUNS EVEN ON THE WHOLE-STICK PATH, where it constrains nothing the
    /// answer depends on; [`StickPart::Whole`] carries no slice, so that abort is not reproduced.
    #[must_use]
    pub fn cumulative_stick_sizes(&self, ds_type: DsType) -> Option<Vec<(PrimaryDim, Elements)>> {
        let info = self.primary_ds_info.get(&ds_type)?;
        shape_constraints::cumulative_stick_sizes(&info.stick, StickPart::Whole)
    }
}

/// WHICH DSC OF THE SUPER-DSC — an index into `dscs_` (`dsc/superdsc.h:67`), which is also the key
/// a `DscScheduleStep` names its DSCs by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DscIdx(pub u32);

/// A SUPER-DSC'S DSCs, NON-EMPTY — `dscs_.size() >= 1` is the whole body of `isSameDscGroup`.
#[derive(Debug, Clone, PartialEq)]
pub struct DscList {
    first: DesignSpaceConfig,
    rest: Vec<DesignSpaceConfig>,
}

impl DscList {
    /// A super-DSC schedules at least one DSC.
    #[must_use]
    pub const fn new(first: DesignSpaceConfig, rest: Vec<DesignSpaceConfig>) -> Self {
        Self { first, rest }
    }

    /// `dscs_.at(idx)`, `None` past the end — that `.at()`'s throw.
    #[must_use]
    pub fn at(&self, idx: DscIdx) -> Option<&DesignSpaceConfig> {
        match idx.0 {
            0 => Some(&self.first),
            n => self.rest.get(usize::try_from(n).ok()? - 1),
        }
    }

    /// `dscs_.at(0)`.
    #[must_use]
    pub const fn first(&self) -> &DesignSpaceConfig {
        &self.first
    }

    /// Every DSC, in `dscs_` order.
    pub fn iter(&self) -> impl Iterator<Item = &DesignSpaceConfig> + '_ {
        core::iter::once(&self.first).chain(self.rest.iter())
    }

    /// Every DSC, in `dscs_` order, to be WRITTEN — `for (auto &dsc : mySDsc.dscs_)`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut DesignSpaceConfig> + '_ {
        core::iter::once(&mut self.first).chain(self.rest.iter_mut())
    }
}

/// A GROUP OF DSCs SELECTED OUT OF A SUPER-DSC, NON-EMPTY — the reference's `dscIndices` with the
/// `dscs_.at()` lookups already done, which is `DT_CHECK_MSG(!dscIndices.empty(), "Expect valid
/// DSCs.")` plus its two `.at` throws discharged at once.
#[derive(Debug, Clone, PartialEq)]
pub struct DscGroup<'a> {
    main: &'a DesignSpaceConfig,
    rest: Vec<&'a DesignSpaceConfig>,
}

impl<'a> DscGroup<'a> {
    /// A group of DSCs led by the one the reference calls `dscMain`.
    #[must_use]
    pub const fn new(main: &'a DesignSpaceConfig, rest: Vec<&'a DesignSpaceConfig>) -> Self {
        Self { main, rest }
    }

    /// `dscs_.at(dscIndices[0])` — the DSC every per-group fact is read from.
    #[must_use]
    pub const fn main(&self) -> &'a DesignSpaceConfig {
        self.main
    }

    /// Every DSC of the group, `dscMain` first.
    pub fn iter(&self) -> impl Iterator<Item = &'a DesignSpaceConfig> + '_ {
        core::iter::once(self.main).chain(self.rest.iter().copied())
    }
}

/// WHICH SLICE OF THE WORK a core takes along one dim — `coreIdToWkSlice_`'s inner value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WkSliceId(pub i32);

/// ONE CORE'S WORK SLICE — `coreIdToWkSlice_`'s value (`dsc/superdsc.h:70`), read only by dim.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WkSlice(pub BTreeMap<PrimaryDim, WkSliceId>);

impl WkSlice {
    /// `at(dim)`, `None` where this slice does not state the dim — the reference's throw.
    #[must_use]
    pub fn at(&self, dim: PrimaryDim) -> Option<WkSliceId> {
        self.0.get(&dim).copied()
    }
}

/// HOW MANY CORES SHARE ONE TRANSFER'S DATA — `getLabeledDsWkSliceMulticastDegree`'s `unsigned`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MulticastDegree(pub u32);

/// ONE STEP OF A CORE'S DSC SCHEDULE — `DscScheduleStep` (`dsc/superdsc.h:30`) reduced to its two
/// DSC indices, whose `-1` default is *no DSC* and so is an [`Option`] here. "If both, then we
/// assume inpNeighborFetch" is the reference's own comment on the pair (`:31`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DscScheduleStep {
    /// `datadsc_idx`.
    pub data_dsc: Option<DscIdx>,
    /// `dldsc_idx`.
    pub dl_dsc: Option<DscIdx>,
}

/// THE SUPER-DSC THIS STAGE SCHEDULES — `SuperDsc` (`dsc/superdsc.h:67`) reduced to the two fields
/// this batch reads.
#[derive(Debug, Clone, PartialEq)]
pub struct SuperDsc {
    dscs: DscList,
    /// `coreIdToWkSlice_`.
    pub core_id_to_wk_slice: BTreeMap<Core, WkSlice>,
    /// `coreIdToDscSchedule` (`dsc/superdsc.h:77`), absent for a core the super-DSC states no
    /// schedule for — that `.at()`'s throw.
    pub core_id_to_dsc_schedule: BTreeMap<Core, Vec<DscScheduleStep>>,
}

impl SuperDsc {
    /// A super-DSC over a non-empty DSC list.
    #[must_use]
    pub const fn new(
        dscs: DscList,
        core_id_to_wk_slice: BTreeMap<Core, WkSlice>,
        core_id_to_dsc_schedule: BTreeMap<Core, Vec<DscScheduleStep>>,
    ) -> Self {
        Self {
            dscs,
            core_id_to_wk_slice,
            core_id_to_dsc_schedule,
        }
    }

    /// `dscs_`.
    #[must_use]
    pub const fn dscs(&self) -> &DscList {
        &self.dscs
    }

    /// `dscs_`, to be WRITTEN — entry 054 fills each DSC's `numCoreletsUsed_DSC2_` through it.
    pub const fn dscs_mut(&mut self) -> &mut DscList {
        &mut self.dscs
    }
}

/// A COUNT OF PADDING ELEMENTS ALONG ONE EDGE OF A DIM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PadElems(pub u32);

/// A DIM'S FRONT AND BACK PADDING — `padFront_`/`padBack_` (`dsc/dims.h:100`), whose paired `-1` is
/// not a size but the statement that CHUNKING VOIDED THEM (`L3DlOpsScheduler.cpp:137`).
///
/// ⭐ `if (padBack_ != 0 || padFront_ != 0)` IS THIS ENUM: voiding an unpadded dim is not a case the
/// scheduler has to test for, because [`PadSizes::Unpadded`] has nothing to void.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PadSizes {
    /// `padFront_ == 0 && padBack_ == 0`.
    #[default]
    Unpadded,
    /// Real padding on at least one edge.
    Sized {
        /// `padFront_`.
        front: PadElems,
        /// `padBack_`.
        back: PadElems,
    },
    /// `padBack_ = padFront_ = -1` — the dim is chunked, so its padding belongs to no chunk.
    Voided,
}

impl PadSizes {
    /// `padFront_`/`padBack_` as the reference stores them.
    #[must_use]
    pub const fn of(front: PadElems, back: PadElems) -> Self {
        if front.0 == 0 && back.0 == 0 {
            Self::Unpadded
        } else {
            Self::Sized { front, back }
        }
    }

    /// `padBack_ = padFront_ = -1`, guarded as the reference guards it.
    #[must_use]
    pub const fn voided(self) -> Self {
        match self {
            Self::Unpadded => Self::Unpadded,
            Self::Sized { .. } | Self::Voided => Self::Voided,
        }
    }
}

/// PADDING THE OP DOES NOT NEED — `unneededPad_`, `unneededPadFront_`, `unneededPadBack_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnneededPad {
    /// `unneededPad_`.
    pub total: PadElems,
    /// `unneededPadFront_`.
    pub front: PadElems,
    /// `unneededPadBack_`.
    pub back: PadElems,
}

impl UnneededPad {
    /// All three zero.
    pub const NONE: Self = Self {
        total: PadElems(0),
        front: PadElems(0),
        back: PadElems(0),
    };
}

/// ONE DIM'S PADDING — `DimPaddingSizes` (`dsc/dims.h:134`) reduced to what `voidPaddingIfChunking`
/// and `calculate_padded`'s full-span arm touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimPadding {
    /// `padFront_`/`padBack_`.
    pub sizes: PadSizes,
    /// `windowDim_`, whose `PrimaryDimTypesCount` default is the absence of a window dim.
    pub window_dim: Option<PrimaryDim>,
    /// The three `unneededPad` counts.
    pub unneeded: UnneededPad,
    /// `stride_`, whose declared default is `1` and not `0`.
    pub stride: Stride,
}

impl Default for DimPadding {
    /// The reference's own field initialisers (`dsc/dims.h:135-142`) — every count zero and the
    /// stride ONE.
    fn default() -> Self {
        Self {
            sizes: PadSizes::Unpadded,
            window_dim: None,
            unneeded: UnneededPad::NONE,
            stride: Stride::ONE,
        }
    }
}

/// A SYMBOLIC DIM'S BOUNDS — `SymbolicDimInfo` (`dsc/dims.h:148`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicDimInfo {
    /// `maxSize_`.
    pub max_size: MaxSize,
    /// `granularity_`.
    pub granularity: Granularity,
}

/// A SYMBOLIC DIM'S LARGEST SIZE — `maxSize_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaxSize(pub u32);

/// A SYMBOLIC DIM'S STEP — `granularity_` (`dsc/dims.h:150`), which the pruner DIVIDES a volume
/// limit by.
///
/// ⛔ DIVERGENCE, AND THE FIELD'S DEFAULT IS A REFERENCE DEFECT: `granularity_ = -1` makes
/// `myVolumeLimit % dimGranularity == 0` pass for every value and `myVolumeLimit /= dimGranularity`
/// NEGATE the limit (`dsc/dims.cpp:746-748`). A non-zero positive step cannot spell that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Granularity(NonZeroU32);

impl Granularity {
    /// A step of at least one element.
    #[must_use]
    pub const fn new(step: NonZeroU32) -> Self {
        Self(step)
    }

    /// The step.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// THE LARGEST VOLUME A SET OF SYMBOLIC DIMS MAY REACH — `maxSymbolicVolume_`'s value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VolumeLimit(pub u32);

impl VolumeLimit {
    /// The pruner's `mulOfMaxes = 1` seed.
    pub const ONE: Self = Self(1);

    /// `mulOfMaxes *= symbolicDimInfo_.at(symDim).maxSize_`.
    #[must_use]
    pub const fn times(self, max: MaxSize) -> Self {
        Self(self.0.saturating_mul(max.0))
    }

    /// `myVolumeLimit /= dimGranularity`, `None` where the reference's
    /// `DT_CHECK(myVolumeLimit % dimGranularity == 0)` does not hold.
    #[must_use]
    pub const fn divided_exactly_by(self, step: Granularity) -> Option<Self> {
        if self.0 % step.get() == 0 {
            Some(Self(self.0 / step.get()))
        } else {
            None
        }
    }
}

/// A STAGE'S SYMBOLIC DIMS AND THE VOLUMES THEY MAY REACH — `symbolicDimInfo_` (`dsc/dims.h:197`)
/// with `maxSymbolicVolume_` (`:202`), ONE value because the pruner reads the second against the
/// first and neither is well formed without the other.
///
/// ⭐ THE INVARIANT IS `DT_CHECK(refDstg.symbolicDimInfo_.count(symDim))` (`dsc/dims.cpp:745`):
/// every dim a volume limit is keyed on is named by `info`, so a stage handed to the pruner as the
/// reference always has the granularity the pruner asks it for.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Symbolic {
    info: BTreeMap<PrimaryDim, SymbolicDimInfo>,
    volumes: BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit>,
}

impl Symbolic {
    /// A stage's symbolic state; volume limits keyed on a dim `info` does not name are DROPPED, the
    /// one place the subset invariant is established.
    #[must_use]
    pub fn new(
        info: BTreeMap<PrimaryDim, SymbolicDimInfo>,
        volumes: BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit>,
    ) -> Self {
        let volumes = volumes
            .into_iter()
            .filter(|(dims, _)| dims.iter().all(|dim| info.contains_key(dim)))
            .collect();
        Self { info, volumes }
    }

    /// `symbolicDimInfo_`.
    #[must_use]
    pub const fn info(&self) -> &BTreeMap<PrimaryDim, SymbolicDimInfo> {
        &self.info
    }

    /// `maxSymbolicVolume_`.
    #[must_use]
    pub const fn volumes(&self) -> &BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit> {
        &self.volumes
    }

    /// `symbolicDimInfo_[dim] = symbolicInfo` — an unchunked dim carried in from another stage.
    pub fn add_dim(&mut self, dim: PrimaryDim, info: SymbolicDimInfo) {
        self.info.insert(dim, info);
    }

    /// `DataStructDims::pruneMaxSymbolicVolumes` (`dsc/dims.cpp:729`) FUSED WITH THE ASSIGNMENT THAT
    /// PRECEDES ITS CALL: adopts `reference`'s volume limits, re-keyed onto the symbolic dims THIS
    /// stage still has, and divided down by the granularity of each dim it lost.
    ///
    /// ⭐ THE FUSION IS WHAT KEEPS THE TYPE HONEST — `maxSymbolicVolume_ = ref.maxSymbolicVolume_`
    /// followed by a prune passes through the one state a well-formed [`Symbolic`] cannot hold, and
    /// this is the only call shape the reference ever uses (`L3DlOpsScheduler.cpp:171-172`).
    ///
    /// ⭐ Every write the reference makes is `min`-guarded and no erased key is ever a write target,
    /// so rebuilding the map with a `min`-insert is its in-place erase-and-insert walk exactly.
    ///
    /// ⛔ DIVERGENCE: where the limit does not divide by a lost dim's granularity — the reference's
    /// `DT_CHECK` at `dsc/dims.cpp:748` — the entry is kept UNTOUCHED, which is the reference's own
    /// `!needPruning` arm rather than an invented divisibility rule.
    pub fn prune_volumes_from(&mut self, reference: &Symbolic) {
        let mut pruned: BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit> = BTreeMap::new();
        let mut keep_min = |dims: BTreeSet<PrimaryDim>, limit: VolumeLimit| {
            let entry = pruned.entry(dims).or_insert(limit);
            *entry = (*entry).min(limit);
        };
        for (sym_dims, limit) in &reference.volumes {
            if sym_dims.iter().all(|dim| self.info.contains_key(dim)) {
                keep_min(sym_dims.clone(), *limit);
                continue;
            }
            let mut mine = BTreeSet::new();
            let mut my_limit = *limit;
            let mut mul_of_maxes = VolumeLimit::ONE;
            let mut exact = true;
            for &sym_dim in sym_dims {
                match self.info.get(&sym_dim) {
                    Some(info) => {
                        mine.insert(sym_dim);
                        mul_of_maxes = mul_of_maxes.times(info.max_size);
                    }
                    None => match reference
                        .info
                        .get(&sym_dim)
                        .and_then(|info| my_limit.divided_exactly_by(info.granularity))
                    {
                        Some(reduced) => my_limit = reduced,
                        None => {
                            exact = false;
                            break;
                        }
                    },
                }
            }
            if !exact {
                keep_min(sym_dims.clone(), *limit);
                continue;
            }
            if !mine.is_empty() {
                keep_min(mine, my_limit.min(mul_of_maxes));
            }
        }
        self.volumes = pruned;
    }
}

/// A DATA STAGE'S DIMS — `DataStructDims` (`dsc/dims.h:158`) reduced to what this batch reads.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StageDims {
    /// The primary dims this stage states, as `primaryDimToVal_st(dim)` answers them.
    pub extents: BTreeMap<PrimaryDim, Extent>,
    /// `paddingSizes_`.
    pub padding: BTreeMap<PrimaryDim, DimPadding>,
    /// `symbolicDimInfo_` with `maxSymbolicVolume_`.
    pub symbolic: Symbolic,
}

impl StageDims {
    /// `primaryDimToVal_st(dim)`, `None` for a dim the stage does not state — its `-1` default,
    /// which compares equal only to another absence.
    #[must_use]
    pub fn extent(&self, dim: PrimaryDim) -> Option<Extent> {
        self.extents.get(&dim).copied()
    }

    /// `hasPadding` (`L3DlOpsScheduler.cpp:1071-1075`) — the dim has a `paddingSizes_` entry AND its
    /// `PADDED_FULLSPAN_WUNNEEDED` span (`calculate_padded`, `dsc/dims.cpp:563`) differs from its
    /// plain extent.
    ///
    /// ⛔ `None` IS `calculate_padded`'s THREE REACHABLE ABORTS: *"Cannot calculate padded version of
    /// compound dim"* for [`PrimaryDim::Ij`]/[`PrimaryDim::Kij`], *"Padded access is not valid in
    /// datastage"* for a [`PadSizes::Voided`] non-window dim, and *"Missing window size"* where the
    /// window dim's own extent is absent or below one.
    ///
    /// ⭐ AN UNSTATED DIM IS `Some(false)`, NOT AN ABORT — `val < 0` short-circuits to `-1` on both
    /// sides of the comparison. Spans saturate rather than wrap: a saturated span still differs from
    /// the extent, which is the only question asked.
    #[must_use]
    pub fn has_padding(&self, dim: PrimaryDim) -> Option<bool> {
        let Some(pad) = self.padding.get(&dim) else {
            return Some(false);
        };
        let Some(plain) = self.extent(dim) else {
            return Some(false);
        };
        if matches!(dim, PrimaryDim::Ij | PrimaryDim::Kij) {
            return None;
        }
        let unneeded = i64::from(pad.unneeded.total.0);
        let padded = match pad.window_dim {
            None => match pad.sizes {
                PadSizes::Voided => return None,
                PadSizes::Unpadded => plain.0.saturating_add(unneeded),
                PadSizes::Sized { front, back } => plain
                    .0
                    .saturating_add(i64::from(front.0))
                    .saturating_add(i64::from(back.0))
                    .saturating_add(unneeded),
            },
            Some(window) => {
                let window_size = self.extent(window).filter(|size| size.0 >= 1)?;
                window_size
                    .0
                    .saturating_add(plain.0.saturating_sub(1).saturating_mul(pad.stride.0))
                    .saturating_add(unneeded)
            }
        };
        Some(padded != plain.0)
    }

    /// `DataStructDims::compound` (`dsc/dims.cpp:84`) — `IJ = I·J` and `KIJ = KI·KJ`, and a product
    /// with an absent or negative operand is the compound dim's OWN absence, which is the `-1` the
    /// reference writes.
    ///
    /// ⛔ THE REFERENCE ALSO WRITES `zij_`, `sij_` AND `rc_`: no `PrimaryDimTypes` value names those
    /// six operands or their three products (`dsc/dims.cpp:485`), so they are unspellable here — and
    /// no unit of this file reads them.
    pub fn compound(&mut self) {
        for (product, left, right) in [
            (PrimaryDim::Ij, PrimaryDim::I, PrimaryDim::J),
            (PrimaryDim::Kij, PrimaryDim::Ki, PrimaryDim::Kj),
        ] {
            match (self.extent(left), self.extent(right)) {
                (Some(Extent(left)), Some(Extent(right))) if left >= 0 && right >= 0 => {
                    self.extents.insert(product, Extent(left * right));
                }
                _ => {
                    self.extents.remove(&product);
                }
            }
        }
    }
}

/// A DATA STAGE WITH DIMS IN IT — `!DataStructDims::empty()` (`dsc/dims.cpp:112`) as a type, so
/// "Expect non-empty data-stage parameters" is discharged where the stage is built.
#[derive(Debug, Clone, PartialEq)]
pub struct FilledDims(StageDims);

impl FilledDims {
    /// A stage that states at least one dim, or `None`.
    #[must_use]
    pub fn of(dims: StageDims) -> Option<Self> {
        (!dims.extents.is_empty()).then_some(Self(dims))
    }

    /// The dims.
    #[must_use]
    pub const fn dims(&self) -> &StageDims {
        &self.0
    }

    /// `paddingSizes_`, for the scheduler to void — reaching the padding cannot empty the extents,
    /// so the non-emptiness survives every mutation this stage makes.
    pub const fn padding_mut(&mut self) -> &mut BTreeMap<PrimaryDim, DimPadding> {
        &mut self.0.padding
    }

    /// `symbolicDimInfo_`/`maxSymbolicVolume_`, for the scheduler to carry forward.
    pub const fn symbolic_mut(&mut self) -> &mut Symbolic {
        &mut self.0.symbolic
    }

    /// `primaryDimToValHandler_st(dim) = extent` — the scheduler's one way to write a dim. Adding an
    /// extent cannot empty the map, so the non-emptiness survives it.
    pub fn set_extent(&mut self, dim: PrimaryDim, extent: Extent) {
        self.0.extents.insert(dim, extent);
    }

    /// `DataStructDims::compound()`, which only ever writes a COMPOUND dim and so cannot empty a
    /// stage that states any other one.
    ///
    /// ⛔ DIVERGENCE, IN THE ONE CASE THE REFERENCE CAN EMPTY A STAGE: a stage stating NOTHING BUT
    /// `IJ`/`KIJ` would lose them to the `-1` write, and keeps them here instead. Every stage this
    /// file compounds is a copy of a data stage's `ss_`, which states its layout dims.
    pub fn compound(&mut self) {
        let before = self.0.extents.clone();
        self.0.compound();
        if self.0.extents.is_empty() {
            self.0.extents = before;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// PLACEMENT — what entries 049, 050, 053, 055 and 056 read off an allocation and its schedule tree.
// ⭐ REACHING AN `AllocateNode` THROUGH `memOrg_` OR A TREE WALK IS THE MECHANISM, the one part the
// campaign statement names as droppable; the placed address, the corelet share and the indirection
// are the facts, and they are what these seams state.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// A PLACED BYTE ADDRESS — `AllocateNode::startAddressCoreCorelet_`'s `int64_t`
/// (`dsc/dsc2.h:985-986`).
///
/// ⛔⛔ UNSIGNED, WHICH IS `DT_CHECK_MSG(startAddr >= 0 && bufferOffset >= 0, "Invalid start address
/// or buffer offset.")` (`L3DlOpsScheduler.cpp:4954-4955`) DISCHARGED HERE: the reference seeds both
/// at `-1` and that check is what rules the sentinel out. Absence carries the sentinel instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteAddress(pub u64);

/// HOW FAR INTO AN ALLOCATION ONE CORE'S BUFFER STARTS — `bufferOffsetCoreCorelet_`'s `int64_t`
/// (`dsc/dsc2.h:988`). A DISPLACEMENT, not an address, and the reference adds the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferOffset(pub u64);

/// HOW MANY BUFFERS AN ALLOCATION GETS — `AllocateNode::numBuffers_` (`dsc/dsc2.h:984`), an enum
/// because that field's own comment states the three values it takes: "1:no buffering,
/// 2:double-buffer, -1:streaming buffer".
///
/// ⛔ ENTRY 050 ACCEPTS ONLY TWO OF THEM — `DT_CHECK_MSG(numBuffers_ == 1 || numBuffers_ == 2,
/// "Expect no buffering or double buffering.")` — so [`Streaming`](Buffering::Streaming) is a state
/// entry 016 can MINT and a placement cannot read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffering {
    /// `1`.
    None,
    /// `2`.
    Double,
    /// `-1`.
    Streaming,
}

impl Buffering {
    /// `numBuffers_`, `None` for a count the field's own comment does not name.
    #[must_use]
    pub const fn of(buffers: i32) -> Option<Self> {
        match buffers {
            1 => Some(Self::None),
            2 => Some(Self::Double),
            -1 => Some(Self::Streaming),
            _ => Option::None,
        }
    }
}

/// A COORDINATE INTO AN ALLOCATION'S PLACED ADDRESSES — `startAddressCoreCorelet_`'s
/// `std::deque<int64_t>`, "per core, corelet, and sdsc folds" (`dsc/dsc2.h:982-984`).
///
/// ⭐ THE CORE AND THE CORELET ARE ENTRIES 0 AND 1, which is what lets entry 050 write
/// `coord.at(1) = 0`; everything behind them belongs to the super-DSC's own folds and passes through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressCoord {
    /// `coord.at(0)`.
    pub core: Core,
    /// `coord.at(1)`.
    pub corelet: Corelet,
    /// `coord` from index 2 on.
    pub sdsc_folds: Vec<i64>,
}

/// WHAT AN ALLOCATION INDIRECTS THROUGH — `AllocateNode::IndirectAllocType` (`dsc/dsc2.h:990`) with
/// the `INDEX_TENSOR` arm carrying the `IndexTensorType` (`:995`) that is only read under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndirectAlloc {
    /// `VALUE_TENSOR` — a paged tensor's values.
    ValueTensor,
    /// `INDEX_TENSOR`.
    IndexTensor(IndexTensor),
}

/// WHAT AN INDEX TENSOR HOLDS — `AllocateNode::IndexTensorType` (`dsc/dsc2.h:995`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexTensor {
    /// `ADDRESS` — the only kind the L3 scheduler supports.
    Address,
    /// `INDEX`.
    Index,
}

/// WHAT A LABELLED DS'S `memOrg_` ANSWERS — `LabeledDsInfo::memOrg_` (`dsc/dscdefn.h:330`) reduced to
/// the reads this batch makes of it.
pub trait MemOrg {
    /// `isHbmPinned()` (`dsc/dscdefn.h:369`) — `memOrg_.at(HBM).isPresent`, false with no HBM entry.
    ///
    /// ⭐ THE SAME PREDICATE [`Pinning::hbm`] STATES, reached from the allocation side rather than
    /// from a [`LabeledDs`]; entry 050 holds the LX node this seam answers for, not the labelled DS.
    fn hbm_pinned(&self) -> bool;

    /// `memOrg_.at(SenComponents::LX).allocateNode_->numBuffers_`.
    ///
    /// ⛔ [`None`] COVERS THREE OF ENTRY 050'S REFUSALS AT ONCE: `DT_CHECK_MSG(memOrg_.count(LX),
    /// "Expect LX in memOrg_.")`, `DT_ERROR("Expect a valid LX allocate node.")`, and a `numBuffers_`
    /// outside the two [`Buffering`] names.
    fn lx_buffering(&self) -> Option<Buffering>;

    /// `startAddressCoreCorelet_.getData(coord)` on that node, `None` where it holds no such entry.
    fn lx_start_address(&self, at: &AddressCoord) -> Option<ByteAddress>;

    /// `bufferOffsetCoreCorelet_.at(coord.at(0)).at(corelet)` on that node — both `.at()`s.
    fn lx_buffer_offset(&self, core: Core, corelet: Corelet) -> Option<BufferOffset>;

    /// `memOrg_.at(SenComponents::HBM).allocateNode_->indirectAllocType_` with its
    /// `indexTensorType_`; [`None`] is no HBM entry, a null allocate node, or `NO_INDIRECTION`.
    fn hbm_indirection(&self) -> Option<IndirectAlloc>;

    /// That allocate node's `name_`, [`None`] where there is no HBM entry or it holds no node —
    /// which is *"Expect HBM in memOrg_."* and *"Expect a valid allocate node."* both
    /// (`L3DlOpsScheduler.cpp:7157`, `:7161`).
    fn hbm_allocation(&self) -> Option<NodeName>;

    /// That node's `layoutDimOrder_` (`dsc/dsc2.h:982`), whose non-emptiness is [`LayoutDims`]' own
    /// — which is *"Expect valid layoutDimOrder_."* (`:6717`).
    fn hbm_layout_dims(&self) -> Option<LayoutDims>;

    /// `getPageSize()`'s KEY SET on that node (`dsc/dsc2.cpp:4479`), EMPTY where nothing pages.
    ///
    /// ⛔ THE SIZES ARE DELIBERATELY NOT ASKED FOR: every reader in scope asks `pageSize.count(dim)`
    /// and nothing more, and `maxDimSizes_` entries are datastage KEYS as often as element counts.
    /// ⭐ ONE UNBOUNDED ENTRY WINS WHEREVER IT SITS — the reference ERASES an already-multiplied dim
    /// on meeting a negative `maxSize` and skips every later entry naming it, so the key set is
    /// "bounded by the layout and never left unbounded", whatever order the layout states them in.
    fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim>;
}

/// WHAT ENTRY 049 READS OFF ONE DATA STAGE — `DataStructDims` (`dsc/dims.h:268-300`) reduced to the
/// four lookups a corelet offset is built from, so the node stage and the chunk stage are ONE type.
pub trait DimStage {
    /// `primaryDimToVal_st(dim, comp, /*ptrowId=*/-1, corelet, padded)` (`dsc/dims.cpp:651`) — that
    /// corelet's padded extent along `dim`, [`None`] for any of its `.at()` throws.
    fn corelet_dim_val(
        &self,
        dim: PrimaryDim,
        comp: SenComponent,
        corelet: Corelet,
        padded: &PaddingForm,
    ) -> Option<Extent>;

    /// `coreletSplit_.count(dim)` (`dsc/dims.h:236`).
    fn is_corelet_split(&self, dim: PrimaryDim) -> bool;

    /// `coreletSplit_.at(dim).at(corelet)` — that corelet's RAW share, unpadded.
    fn corelet_split(&self, dim: PrimaryDim, corelet: Corelet) -> Option<Extent>;

    /// `paddingSizes_.at(dim).stride_` (`dsc/dims.h:219,140`).
    ///
    /// ⛔ [`None`] IS BOTH OF ENTRY 049'S CHECKS ON IT — `paddingSizes_.count(dim)` and
    /// `stride_ > 0` — because a non-positive stride is not a stride this offset can use.
    fn pad_stride(&self, dim: PrimaryDim) -> Option<Stride>;
}

/// ONE DSC'S SCHEDULE TREE AS ENTRY 053 READS IT — `scheduleTree_.traverseTreeDFS()`.
pub trait ScheduleNodes {
    /// Every node's `name_` in DFS order; the EMPTY vector is `scheduleTree_.empty()`.
    fn node_names(&self) -> Vec<NodeName>;
}

/// ONE ALLOCATION AS ENTRY 055 READS IT — an `ALLOCATE` node's `ldsIdx_`, `component_` and the
/// `(core, address)` pairs its `startAddressCoreCorelet_` states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedAllocation {
    /// `ldsIdx_`, [`None`] for the reference's `-1`.
    pub lds: Option<LdsIdx>,
    /// `component_`.
    pub component: SenComponent,
    /// `getDataAndFoldCoordinates()` reduced to `coord[0]` and the value, in the fold manager's own
    /// order — `if (!coord.empty())` skips an entry that names no core at all.
    pub addresses: Vec<(Core, ByteAddress)>,
}

/// EVERY DSC'S ALLOCATIONS IN ONE SUPER-DSC — `dscs_.at(i).scheduleTree_.traverseTreeDFS(nullptr,
/// {dsc2::ScheduleNode::ALLOCATE})`.
///
/// ⭐ `DT_CHECK_MSG(allocNode, "Expect an allocate node.")` IS THE FILTER THAT PRODUCED THE LIST: the
/// reference `dynamic_cast`s each node back down and checks the cast it just asked the walk for.
pub trait ScheduleTrees {
    /// That DSC's allocations in DFS order; EMPTY for an index the super-DSC does not have.
    fn allocations(&self, dsc: DscIdx) -> Vec<PlacedAllocation>;
}

/// `dataStageCoreIdx`, `0` (`L3DlOpsScheduler.cpp:275`).
pub const DATA_STAGE_CORE: DatastageId = DatastageId(0);

/// `dataStageChunkIdx`, `1` (`L3DlOpsScheduler.cpp:276`).
pub const DATA_STAGE_CHUNK: DatastageId = DatastageId(1);

/// ONE DSC'S SCHEDULER METADATA — `L3DlOpsScheduler::Metadata` (`L3DlOpsScheduler.h:111`) reduced to
/// the two fields this batch writes.
///
/// ⭐⭐ NEITHER ID IS OPTIONAL, AND THAT IS THE REFERENCE'S `-1`s GONE. `core_dstgid` and
/// `chunk_dstgid` are declared `-1` (`:193-194`) and `prepDsc` is the ONLY thing that mints an
/// entry, always writing both, so a `Metadata` without them is a state the reference cannot reach.
/// ⛔ IT IS THE SCHEDULER'S OWN `Metadata`, NOT [`crate::schedule::ddc::metadata::Metadata`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerMetadata {
    /// `core_dstgid`.
    pub core_dstg: DatastageId,
    /// `chunk_dstgid`.
    pub chunk_dstg: DatastageId,
}

/// HOW FAR INTO AN ALLOCATION ONE CORELET'S SHARE STARTS — entry 049's `int64_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CoreletOffset(pub Bytes);

/// WHERE ONE LABELLED DS'S LX DATA STARTS — entry 050's `std::pair<int64_t, int64_t>`, whose two
/// halves are different units and so different types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialPlacement {
    /// `startAddressCoreCorelet_.getData(coord)`.
    pub start: ByteAddress,
    /// `bufferOffsetCoreCorelet_.at(core).at(0)`, or `0` where there is only ever one buffer.
    pub buffer_offset: BufferOffset,
}
/// THE TWO STAGE NAMES THE L3 SCHEDULER WRITES.
///
/// ⭐ AN INHERENT IMPL ON [`crate::schedule::ddc::transformation_util::StageName`] AND NOT A SECOND
/// NEWTYPE: `DataStructDims::name_` is one field, and the ddc view already states it. The two names
/// live here because `"superchunk"` is the L3 scheduler's word, not the transformer's.
impl StageName {
    /// `"superchunk"` (`L3DlOpsScheduler.cpp:2814`).
    #[must_use]
    pub fn super_chunk() -> Self {
        Self("superchunk".to_owned())
    }

    /// `"chunk"` (`L3DlOpsScheduler.cpp:1412`).
    #[must_use]
    pub fn chunk() -> Self {
        Self("chunk".to_owned())
    }
}

/// ONE HALF OF A DATA STAGE — a `DataStructDims` with its `name_`, which is the field the two halves
/// of a `dsc2::DataStage` differ in.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedDims {
    /// `name_`.
    pub name: StageName,
    /// The dims themselves.
    pub dims: FilledDims,
}

/// ONE DATA STAGE — `dsc2::DataStage` (`dsc/dsc2.h:40`): the stick-space dims and the element-space
/// dims, each carrying its own name.
#[derive(Debug, Clone, PartialEq)]
pub struct DataStage {
    /// `ss_`.
    pub ss: NamedDims,
    /// `el_`.
    pub el: NamedDims,
}

impl DataStage {
    /// `name()` — `ss_.name_` (`dsc/dsc2.h:43`), so the stage's name is the stick side's.
    #[must_use]
    pub const fn name(&self) -> &StageName {
        &self.ss.name
    }

    /// `ds.ss_.name_ = ds.el_.name_ = name` — a rename touches BOTH halves, which is the only write
    /// `addSuperChunkDataStage` makes to the stage it copies.
    pub fn rename(&mut self, name: StageName) {
        self.ss.name = name.clone();
        self.el.name = name;
    }

    /// `ss_.primaryDimToVal_st(dim)`, `None` for a dim the stick side does not state.
    #[must_use]
    pub fn ss_extent(&self, dim: PrimaryDim) -> Option<Extent> {
        self.ss.dims.dims().extent(dim)
    }
}

/// EVERY DATA STAGE OF A DSC — `dataStageParam_`, with the core and chunk stages HELD APART.
///
/// ⭐⭐ THAT SPLIT IS "Core data stage parameters are unavailable." AND "Expect chunk data stage."
/// DISCHARGED ONCE: eight units of this file open with one or both of those `DT_CHECK`s and not one
/// of them has an arm for the failure, so the two fixed stages are fields and the minted ones are a
/// map.
#[derive(Debug, Clone, PartialEq)]
pub struct DataStages {
    core: DataStage,
    chunk: DataStage,
    minted: BTreeMap<DatastageId, DataStage>,
}

impl DataStages {
    /// A DSC whose core and chunk stages are both stated.
    #[must_use]
    pub const fn new(core: DataStage, chunk: DataStage) -> Self {
        Self {
            core,
            chunk,
            minted: BTreeMap::new(),
        }
    }

    /// `dataStageParam_.at(dataStageCoreIdx)`.
    #[must_use]
    pub const fn core(&self) -> &DataStage {
        &self.core
    }

    /// `dataStageParam_.at(dataStageChunkIdx)`.
    #[must_use]
    pub const fn chunk(&self) -> &DataStage {
        &self.chunk
    }

    /// `dataStageParam_.at(index)`, `None` for an index no stage was created for.
    #[must_use]
    pub fn at(&self, index: DatastageId) -> Option<&DataStage> {
        match index {
            DATA_STAGE_CORE => Some(&self.core),
            DATA_STAGE_CHUNK => Some(&self.chunk),
            other => self.minted.get(&other),
        }
    }

    /// `dataStageParam_[index] = stage`.
    pub fn set(&mut self, index: DatastageId, stage: DataStage) {
        match index {
            DATA_STAGE_CORE => self.core = stage,
            DATA_STAGE_CHUNK => self.chunk = stage,
            other => {
                self.minted.insert(other, stage);
            }
        }
    }

    /// `dataStageSuperChunkIdx >= 0 && dataStageParam_.count(dataStageSuperChunkIdx)` — the witness
    /// that a superchunk index NAMES AN ENTRY THAT ALREADY EXISTS.
    ///
    /// ⭐ IT HOLDS BECAUSE `getNewDataStageIndex` DEFAULT-INSERTS: its last statement is the bare
    /// subscript `dsc.dataStageParam_[newIdx];` (`L3DlOpsScheduler.cpp:6622`), so the entry is present
    /// and empty before `addSuperChunkDataStage` ever runs. The `>= 0` half is [`DatastageId`]'s own.
    #[must_use]
    pub fn super_chunk(&self, index: DatastageId) -> Option<SuperChunkStage> {
        self.at(index).map(|_| SuperChunkStage(index))
    }
}

/// A SUPERCHUNK DATA-STAGE INDEX THAT NAMES AN EXISTING ENTRY — minted by
/// [`DataStages::super_chunk`] and by nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuperChunkStage(DatastageId);

impl SuperChunkStage {
    /// `dataStageSuperChunkIdx` (`L3DlOpsScheduler.h:224`).
    #[must_use]
    pub const fn index(self) -> DatastageId {
        self.0
    }
}

/// ONE DIM'S CANDIDATE CHUNK EXTENTS WITH THE INDEX THE SEARCH SELECTED — `DscParamCandidatesType`'s
/// inner vector ZIPPED ONTO `DscParamCandidateIndicesType`'s index (`L3DlOpsScheduler.h:100-103`).
///
/// ⭐ THE ZIP IS `DT_CHECK_MSG(selectedIdx < dscCandidates[dscIdx].at(dim).size(), "Index is out of
/// range.")`: once the list and the choice into it are one value, the question cannot be asked, and
/// the pair of `.at(dim)` lookups the reference does on two separate maps becomes one.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedCandidate {
    candidates: Vec<Extent>,
    selected: usize,
}

impl SelectedCandidate {
    /// A candidate list with the index the search chose, or `None` where the index is past its end.
    #[must_use]
    pub fn new(candidates: Vec<Extent>, selected: u32) -> Option<Self> {
        let selected = selected as usize;
        (selected < candidates.len()).then_some(Self {
            candidates,
            selected,
        })
    }

    /// `dscCandidates[dscIdx].at(dim).at(selectedIdx)`.
    #[must_use]
    pub fn extent(&self) -> Extent {
        self.candidates[self.selected]
    }

    /// Every candidate the search generated for this dim.
    #[must_use]
    pub fn candidates(&self) -> &[Extent] {
        &self.candidates
    }
}

/// ONE DSC'S SELECTED CHUNK PARAMETERS — `dscCandidates[dscIdx]` and `selectedIndices[dscIdx]`
/// narrowed to the `primaryDims` the caller asks to be written, which is the whole mechanism for
/// reaching this unit's operands.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DscParamCandidates(pub BTreeMap<PrimaryDim, SelectedCandidate>);

/// HOW MANY STICKS ONE STICK VOLUME SPANS — `stickVolume`, POSITIVE BY TYPE, which is
/// `DT_CHECK_MSG(stickVolume > 0, "Invalid stick volume.")` (`L3DlOpsScheduler.cpp:1703`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StickVolume(NonZeroU64);

impl StickVolume {
    /// A stick volume of at least one stick.
    #[must_use]
    pub const fn new(sticks: NonZeroU64) -> Self {
        Self(sticks)
    }

    /// The span, in sticks.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// HOW MANY STICK VOLUMES — the count `getLabeledDsNumOfStickVolumesInCore` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StickVolumes(pub u64);
