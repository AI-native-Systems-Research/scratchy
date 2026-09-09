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

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    self as shape_constraints, Extent, PrimaryDim, StickDims, StickPart,
};
use crate::schedule::ddc::fold::Stride;
use crate::schedule::ddc::transformation::{DsType, Scale};
use crate::schedule::dsc2::{LayoutDims, LdsIdx};
use crate::units::Core;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

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
}

/// ONE DESIGN SPACE CONFIG — `DesignSpaceConfig` (`dsc/designSpaceConfig.h:74`) reduced to the
/// fields this batch reads.
#[derive(Debug, Clone, PartialEq)]
pub struct DesignSpaceConfig {
    /// `numCoreletsUsed_`.
    pub corelets_used: CoreletsUsed,
    /// Per dim, corelet 0's share against the whole — `dataStageParam_.at(0).ss_` where the core
    /// data stage exists, else `CoreletD_` against `CoreD_`.
    pub corelet_shares: BTreeMap<PrimaryDim, CoreletShare>,
    /// `primaryDsInfo_` (`dsc/dscdefn.h:474`).
    pub primary_ds_info: BTreeMap<DsType, PrimaryDsInfo>,
    /// `coreIdsUsed_`.
    pub core_ids_used: CoreIdsUsed,
    /// `getLayoutDims(ldsIdx)` (`dsc/dsc2.cpp:4007`), per labelled data structure.
    pub layout_dims: BTreeMap<LdsIdx, LayoutDims>,
    /// `dataStageParam_.at(dataStageCoreIdx).ss_` — `dataStageCoreIdx` is `0`
    /// (`L3DlOpsScheduler.cpp:275`).
    ///
    /// ⭐ MANDATORY, WHICH IS `DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx), "Expect
    /// dataStageParam_ entry for the core data stage.")` (`:353`, `:1185`, `:1191`) DISCHARGED HERE:
    /// every min-param unit reaches it with a bare `.at()`. `isDimensionCoreletSplit`'s defensive
    /// `count` (`:77`) is then a constant, and [`Self::corelet_shares`] keeps its own answer because
    /// the split it reports may come from `CoreletD_`/`CoreD_` instead.
    pub core_stage: FilledDims,
    /// `labeledDs_`.
    pub labeled_ds: LabeledDsList,
}

impl DesignSpaceConfig {
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

    /// `DataStructDims::pruneMaxSymbolicVolumes` (`dsc/dims.cpp:728`) FUSED WITH THE ASSIGNMENT THAT
    /// PRECEDES ITS CALL: adopts `reference`'s volume limits, re-keyed onto the symbolic dims THIS
    /// stage still has, and divided down by the granularity of each dim it lost.
    ///
    /// ⭐ THE FUSION IS WHAT KEEPS THE TYPE HONEST — `maxSymbolicVolume_ = ref.maxSymbolicVolume_`
    /// followed by a prune passes through the one state a well-formed [`Symbolic`] cannot hold, and
    /// this is the only call shape the reference ever uses (`L3DlOpsScheduler.cpp:170-171`).
    ///
    /// ⭐ Every write the reference makes is `min`-guarded and no erased key is ever a write target,
    /// so rebuilding the map with a `min`-insert is its in-place erase-and-insert walk exactly.
    ///
    /// ⛔ DIVERGENCE: where the limit does not divide by a lost dim's granularity — the reference's
    /// `DT_CHECK` at `dsc/dims.cpp:747` — the entry is kept UNTOUCHED, which is the reference's own
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

    /// `hasPadding` (`L3DlOpsScheduler.cpp:1070-1075`) — the dim has a `paddingSizes_` entry AND its
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
}
