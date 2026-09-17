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
    self as shape_constraints, Extent, PrimaryDim, StickDims, StickPart, VectorComp,
};
use crate::formats::DataFormat;
use crate::schedule::ddc::fold::{
    AllocId, ConstIdx, Dilation, MxScaleTensor, NodeId, PadType, ScaleBlock, ScaledLds, Stride,
};
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind};
use crate::schedule::ddc::transformation::{DsType, Scale};
use crate::schedule::ddc::transformation_util::{PaddingForm, StageName};
use crate::schedule::ddc::v1::{DimSample, L0Tethered, PeSfpShares, StorageName};
use crate::schedule::dsc2::{LayoutDims, LdsIdx, NodeName, WordLength};
use crate::schedule::l3::dl_ops::{GtrGroupId, VariableSymbol};
use crate::units::{Core, Corelet, Row};
use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU32, NonZeroU64};
/// ⭐ RE-EXPORTED BECAUSE [`Pinning::mem_org`] IS A PUBLIC FIELD KEYED BY IT — a caller outside this
/// crate that builds a [`Pinning`] has to be able to NAME the component, and `sys-arch-spec` is not
/// its dependency. One path, so the key type cannot be reached through two.
pub use sys_arch_spec::arch_enums::SenComponent;

/// WHERE ONE LABELLED DATA STRUCTURE LIVES — `memOrg_` (`dsc/dscdefn.h:337`): every component the
/// map names with that entry's `isPresent`, plus the two LX questions a component set cannot answer.
///
/// ⛔ THE KEY AND THE FLAG ARE DIFFERENT QUESTIONS, AND THE REFERENCE ASKS BOTH: `memOrg_.count(comp)`
/// is [`Self::names`] while `memOrg_.at(comp).isPresent` is the value — an entry present as a key with
/// `isPresent = false` answers the first and not the second.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pinning {
    /// Each component `memOrg_` names, with its `isPresent`.
    pub mem_org: BTreeMap<SenComponent, bool>,
    /// `isLxPinned()` (`:424`) — an LX organisation that is neither HBM- nor XRF-pinned nor a ring.
    pub lx: bool,
    /// `memOrg_.count(LX) && memOrg_.at(LX).isPadded`.
    pub lx_padded: bool,
}

impl Pinning {
    /// `pinnedComponent()`'s own check order (`dsc/dscdefn.h:443-445`) — the pinning predicates first,
    /// then every other memory.
    const CHECK_ORDER: [SenComponent; 11] = [
        SenComponent::Hbm,
        SenComponent::Ring,
        SenComponent::Sfpring,
        SenComponent::Lx,
        SenComponent::Pt,
        SenComponent::Ptxrf,
        SenComponent::Ptarf,
        SenComponent::Sfplrf,
        SenComponent::Pelrf,
        SenComponent::L0,
        SenComponent::Ptirf,
    ];

    /// `isHbmPinned()` (`:369-375`).
    #[must_use]
    pub fn hbm(&self) -> bool {
        self.mem_org.get(&SenComponent::Hbm) == Some(&true)
    }

    /// `memOrg_.count(component)` — whether the map NAMES it, which an absent `isPresent` still does.
    #[must_use]
    pub fn names(&self, component: SenComponent) -> bool {
        self.mem_org.contains_key(&component)
    }

    /// `pinnedComponent()` (`:442-454`) — the FIRST present component in that order, and [`None`] for
    /// its `NO_COMPONENT` fallthrough.
    #[must_use]
    pub fn pinned_component(&self) -> Option<SenComponent> {
        Self::CHECK_ORDER
            .into_iter()
            .find(|component| self.mem_org.get(component) == Some(&true))
    }
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
    scale_tensor: Option<MxScaleTensor>,
    record: LdsRecord,
    scaled_category: ScaledLds,
}

/// ⭐⭐ THE THREE `LabeledDsInfo` FIELDS THE DSM *DESCRIBES* AN OPERAND WITH — `dsName_`
/// (`dsc/dscdefn.h:326`), `wordLength` (`:334`) and `dataFormat_` (`:335`).
///
/// ⭐ ONE VALUE BECAUSE THEY ARE WRITTEN AS ONE: `insert_internal_tensor` copies all three off a
/// reference entry in one statement (`ddc/ddl/ddl_conversion.cpp`'s `newLds`), and
/// `prep_dsc`'s three writers set the three of one entry in a row (`ddc/v1.rs:4032`, `:4036`,
/// `:4038`).
///
/// ⛔ THE DEFAULTS ARE THE AUTHORITY'S OWN MEMBER INITIALIZERS, VERBATIM, and that is why
/// [`Default`] is derivable: `dsName_` is a default-constructed `std::string` (EMPTY, `:326` states
/// no initializer), `wordLength = 0` (`:334`) and `dataFormat_ = DataFormats::INVALID` (`:335`) —
/// which [`None`] is, because [`DataFormat`] spells no `INVALID` variant and an absent format is
/// exactly *"no precision information yet"*.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LdsRecord {
    /// `dsName_` (`dsc/dscdefn.h:326`) — EMPTY on a DS nothing named.
    pub name: StorageName,
    /// `wordLength` (`:334`) — ⛔ A `double` THERE AND AN ELEMENT WIDTH IN BYTES HERE: scratchy
    /// writes `2` for `SEN169_FP16` and `1` for `SEN143_FP8` over `g0/`'s 580 labelled DSs, never a
    /// fraction. `0` is the field's own initializer, which is *"nobody stated a width"*.
    pub word_length: WordLength,
    /// `dataFormat_` (`:335`) — ⛔ [`None`] IS `DataFormats::INVALID`, the field's own initializer.
    pub data_format: Option<DataFormat>,
}

impl LabeledDs {
    /// A labelled data structure's layout order paired with its scales, innermost first.
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
            scale_tensor: None,
            record: LdsRecord::default(),
            scaled_category: ScaledLds::Regular,
        }
    }

    /// `scaledLdsCategory_ == SCALE_TENSOR` WITH ITS `mxInfo_` — a BUILDER and not a `new` argument
    /// because `SCALE_TENSOR` is the rare category and every other site states `REGULAR`.
    ///
    /// ⛔ IT SETS `scaledLdsCategory_` TOO, and it must: [`Self::scale_tensor`]'s own doc is *"the
    /// ONE pair of conditions every reader of it tests"*, so a `mxInfo_` beside a `REGULAR_TENSOR`
    /// category would be a state the reference cannot hold and a second answer to
    /// [`Self::scaled_category`].
    #[must_use]
    pub fn with_scale_tensor(mut self, scale_tensor: MxScaleTensor) -> Self {
        self.scale_tensor = Some(scale_tensor);
        self.scaled_category = ScaledLds::Scale;
        self
    }

    /// `{dsName_, wordLength, dataFormat_}` — a BUILDER for the same reason
    /// [`Self::with_scale_tensor`] is: every existing construction site of this type states the
    /// authority's own initializers, which is what [`LdsRecord::default`] is.
    #[must_use]
    pub fn with_record(mut self, record: LdsRecord) -> Self {
        self.record = record;
        self
    }

    /// `mxInfo_` where `scaledLdsCategory_ == SCALE_TENSOR`, which is the ONE pair of conditions
    /// every reader of it tests — [`None`] for any other category.
    ///
    /// ⭐ THE CATEGORY GATE IS IN THE READER, so `mxInfo_` and `scaledLdsCategory_` stay the two
    /// independent fields the reference has (`copy_mx_info` writes one, `set_lds_scaled_category` the
    /// other) and the PAIR is still one answer — the same shape [`MxScaleTensor::of`] states.
    #[must_use]
    pub const fn scale_tensor(&self) -> Option<MxScaleTensor> {
        match self.scaled_category {
            ScaledLds::Scale => self.scale_tensor,
            ScaledLds::Regular | ScaledLds::Value => None,
        }
    }

    /// `scaledLdsCategory_` (`dsc/dscdefn.h:352-356`) as the CLOSED THREE-WAY it is.
    ///
    /// ⛔ NOT DERIVABLE FROM [`Self::scale_tensor`]: that answers `SCALE_TENSOR` against everything
    /// else, and `REGULAR_TENSOR` vs `VALUE_TENSOR` is the pair entry 307 branches on.
    #[must_use]
    pub const fn scaled_category(&self) -> ScaledLds {
        self.scaled_category
    }

    /// `dsName_`, `wordLength` and `dataFormat_` together.
    #[must_use]
    pub const fn record(&self) -> &LdsRecord {
        &self.record
    }

    /// `dsName_ = name` — `AutoShuffling::set_lds_name` (`ddc/transformation.rs:3083`).
    pub fn set_name(&mut self, name: StorageName) {
        self.record.name = name;
    }

    /// `dsName_ += suffix` — `PrepDsc::append_lds_name` (`ddc/v1.rs:4032`).
    pub fn append_name(&mut self, suffix: &str) {
        self.record.name.0.push_str(suffix);
    }

    /// `wordLength = length`.
    pub const fn set_word_length(&mut self, length: WordLength) {
        self.record.word_length = length;
    }

    /// `dataFormat_ = format`.
    pub const fn set_data_format(&mut self, format: DataFormat) {
        self.record.data_format = Some(format);
    }

    /// `scaledLdsCategory_ = category`.
    pub const fn set_scaled_category(&mut self, category: ScaledLds) {
        self.scaled_category = category;
    }

    /// `mxInfo_ = labeledDs_.at(from).mxInfo_` — `PrepDsc::copy_mx_info`'s write half.
    pub const fn set_mx_info(&mut self, mx_info: Option<MxScaleTensor>) {
        self.scale_tensor = mx_info;
    }

    /// `dsType_ = ds_type` — `PrepDsc::set_lds_internal` writes `DsTypes::INTERNAL` here.
    pub const fn set_ds_type(&mut self, ds_type: DsType) {
        self.ds_type = ds_type;
    }

    /// `ldsIdx_ = recorded` — `NewLabeledDs::set_last_recorded_lds_idx`'s `++back().ldsIdx_`.
    ///
    /// ⛔ THE ENTRY'S OWN INDEX AND NOT ITS POSITION, which is the drift [`Self::recorded`] documents:
    /// writing it does NOT move the entry.
    pub const fn set_recorded(&mut self, recorded: LdsIdx) {
        self.recorded = recorded;
    }

    /// `ldsIdx_` — the entry's OWN self-index, which need NOT equal the position it sits at in
    /// `labeledDs_`: it defaults to `183` (`dsc/dscdefn.h:323`) and is written independently.
    #[must_use]
    pub fn recorded(&self) -> LdsIdx {
        self.recorded
    }

    /// `memOrg_`, as the questions asked of it.
    #[must_use]
    pub const fn pinning(&self) -> &Pinning {
        &self.pinning
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

/// HOW MANY CORES A DSC USES — `numCoresUsed_` (`dsc/designSpaceConfig.h:73`).
///
/// ⛔ NOT A FIELD, BECAUSE IT IS NOT A SECOND FACT: `DT_CHECK(coreIdsUsed_.size() == numCoresUsed_)`
/// (`dsc/designSpaceConfig.cpp:1033`, `dsc/dsc2Pcfg.cpp:21`) says the two agree, so
/// [`CoreIdsUsed::count`] derives it and the pair cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreCount(pub u32);

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

    /// `numCoresUsed_`, which is `coreIdsUsed_.size()` — ONE at minimum, by construction.
    #[must_use]
    pub fn count(&self) -> CoreCount {
        CoreCount(
            u32::try_from(self.rest.len())
                .unwrap_or(u32::MAX)
                .saturating_add(1),
        )
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

    /// `labeledDs_.at(idx)` AS THE REFERENCE'S NON-CONST `.at()` — the same position, writable, for the
    /// five `prep_dsc`/`AutoShuffling` setters that rewrite one entry's record.
    ///
    /// ⛔ THE POSITION AND NOT [`LabeledDs::recorded`], exactly as [`Self::at`] is: the two can drift.
    pub fn at_mut(&mut self, idx: LdsIdx) -> Option<&mut LabeledDs> {
        match idx.0 {
            0 => Some(&mut self.first),
            at => self.rest.get_mut(usize::try_from(at).ok()? - 1),
        }
    }

    /// `labeledDs_.back()`, writable — TOTAL for the same reason [`Self::back`] is.
    pub fn back_mut(&mut self) -> &mut LabeledDs {
        self.rest.last_mut().unwrap_or(&mut self.first)
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

/// ONE CONSTANT OF A DSC — `dsc2::ConstantInfo` (`dsc/dsc2.h:46-62`).
///
/// ⛔ `data_` IS A `FoldManager<std::vector<int64_t>>` AND THIS IS ITS SINGLE FOLD, which is the ONE
/// reading the ported units make: `FoldInfraUtils::getSingleDataStrict(constinfo.data_)`
/// (`ddc/ddcv1.cpp:2067`) is the only accessor of it in the batch, and `getSingleDataStrict` is by its
/// own name the assertion that there is exactly one. All twelve constants scratchy emits across
/// `g0/`'s 187 programs carry one datum (`"data_": [10240]` / `[15872]`).
///
/// ⛔ `allocations_` IS A `map<SenComponents, dsc2::AllocateNode*>` THERE AND AN [`AllocId`] HERE —
/// the arena handle, which is how every other allocate-node reference in this crate is spelled. It
/// is EMPTY on every constant scratchy writes (`"allocations_": {}`), and
/// `ComponentAllocations::set_constant_allocation` is what fills it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConstantInfo {
    /// `name_` (`dsc/dsc2.h:48`).
    pub name: StorageName,
    /// `dataFormat_` (`:47`) — ⛔ [`None`] IS `DataFormats::INVALID`, the field's own initializer.
    pub data_format: Option<DataFormat>,
    /// `getSingleDataStrict(data_)` (`:49`) — the one fold's values, in the stated format.
    pub data: Vec<i64>,
    /// `isDataSymbolic_` (`:50`).
    pub is_data_symbolic: bool,
    /// `allocations_` (`:51`) as arena handles.
    pub allocations: BTreeMap<SenComponent, AllocId>,
}

/// ⭐⭐ THE FOUR `DesignSpaceConfig` FIELDS **ONLY STAGE 2B** READS — `constantInfo_`
/// (`dsc/designSpaceConfig.h:90`), `maskingConstId_` (`:101`), `dimToSymbolMapping_` (`:76-77`) and
/// `l0TetheredMode_` (`:117`).
///
/// ⛔ ONE VALUE AND NOT FOUR FIELDS OF [`DesignSpaceConfig`], AND THAT IS WHAT KEEPS THE MODULE'S OWN
/// RULE VISIBLE: `l3::dsc` is *"a reduced per-module projection … each module states the fields its own
/// units touch and nothing else"*, and no L3 unit reads any of these — [`crate::schedule::ddc::v1`]'s
/// do. Grouping them says which stage they belong to, and it means every stage-2a construction site
/// states [`Default::default`], which is each field's OWN C++ initializer and not a value chosen here.
///
/// ⛔⛔ AND EVERY ONE OF THOSE INITIALIZERS IS THE STATE SCRATCHY'S SuperDSC ACTUALLY LEAVES.
/// `dimToSymbolMapping_`, `gtrIdsUsed_` and `l0TetheredMode_` are *"all scheduler outputs — DROPPED"*
/// by the emitter's own note (`lower_subtile_tape_to_superdsc.rs:1240-1241`), so absence there is the
/// declared default here (`{}` and `false`) rather than a fact the conversion lost.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DdcFacts {
    /// `constantInfo_` (`dsc/designSpaceConfig.h:90`), keyed by the `int` id it is filed under.
    pub constants: BTreeMap<ConstIdx, ConstantInfo>,
    /// `maskingConstId_` (`:101`) — ⛔ [`None`] IS THE DECLARED `-1`, and *"assuming all tensors use
    /// same masking constant"* is the field's own comment.
    pub masking_const: Option<ConstIdx>,
    /// `dimToSymbolMapping_` (`:76-77`) — *"single value for pure symbolic and pivot dims, multiple
    /// entries (max-pivot) for irregular dims"*, so a dim's entry is a LIST and an ABSENT dim is a
    /// `count(dim) == 0`.
    pub dim_to_symbol: BTreeMap<PrimaryDim, Vec<VariableSymbol>>,
    /// `l0TetheredMode_` (`:117`) — ⛔ [`L0Tethered::Split`] IS THE DECLARED `false`.
    pub l0_tethered: L0Tethered,
}

/// ONE DESIGN SPACE CONFIG — `DesignSpaceConfig` (`dsc/designSpaceConfig.h:74`) reduced to the
/// fields this batch reads.
#[derive(Debug, Clone, PartialEq)]
pub struct DesignSpaceConfig {
    /// The four fields only stage 2b reads — see [`DdcFacts`].
    pub ddc: DdcFacts,
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
    /// `N_.paddingSizes_` (`dsc/designSpaceConfig.h:103`) — the WHOLE data structure's padding, which
    /// is where the window a padded dim belongs to is stated. EMPTY where nothing is padded.
    ///
    /// ⭐ THE PADDING ALONE AND NOT THE `N_` STAGE: entries 215 and 221 read this map and nothing else
    /// of it, and a second copy of the extents is a second answer that can disagree with
    /// [`Self::data_stages`].
    pub full_padding: BTreeMap<PrimaryDim, DimPadding>,
    /// `gtrIdsUsed_` (`dsc/designSpaceConfig.h:116`) — which group tag registers this DSC's
    /// multicast transfers claim. EMPTY on a DSC the L3 scheduler has not reached; entries 218 and
    /// 291 are what fill it, and only for a group with more than one sharer.
    pub gtr_ids_used: BTreeSet<GtrGroupId>,
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

    /// `getNonBroadcastLdsDimSet(ldsIdx)` (`dsc/dsc2.cpp:4050`) — the dims of the labelled DS's OWN
    /// layout order whose `scale_` is strictly positive, NOT filtered through `getLayoutDims`.
    ///
    /// ⛔ TRAP: THIS IS NOT [`Self::non_broadcast_lds_dims`], which intersects this set with the
    /// ALLOCATE node's order (`dsc/dsc2.cpp:4044`). Entries 201 and 202 ask THIS one, so a dim the
    /// allocate node does not name is non-broadcast here and absent there.
    ///
    /// ⛔ [`None`] IS `labeledDs_.at(ldsIdx)` THROWING; `if (ldsIdx < 0) return {}` beside it is
    /// unspellable because [`LdsIdx`] is unsigned, and `layout.at(i)`'s own throw is unreachable
    /// because [`LabeledDs`] carries the layout order and the scales ZIPPED.
    #[must_use]
    pub fn non_broadcast_lds_dim_set(&self, lds: LdsIdx) -> Option<BTreeSet<PrimaryDim>> {
        let entry = self.labeled_ds.at(lds)?;
        Some(
            entry
                .scales
                .iter()
                .filter(|(_, scale)| matches!(scale, Scale::Sized(scale) if *scale > 0.0))
                .map(|(dim, _)| *dim)
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

    /// `getStickDims(ldsIdx)` (`dsc/designSpaceConfig.h:244`) — `primaryDsInfo_.at(dsType_)
    /// .stickDimOrder_` of the labelled DS sitting at that POSITION, in `stickDimOrder_` order.
    ///
    /// ⛔ [`None`] IS EITHER `.at()` THROWING: a position past `labeledDs_`, or a DS type this DSC
    /// states no `primaryDsInfo_` entry for.
    #[must_use]
    pub fn stick_dims(&self, lds: LdsIdx) -> Option<Vec<PrimaryDim>> {
        let info = self
            .primary_ds_info
            .get(&self.labeled_ds.at(lds)?.ds_type())?;
        Some(info.stick.0.iter().map(|&(dim, _)| dim).collect())
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

    /// `dscs_.at(idx)` TO BE WRITTEN, `None` past the end — `auto& dsc = mySDsc.dscs_.at(dscIdx)`.
    pub fn at_mut(&mut self, idx: DscIdx) -> Option<&mut DesignSpaceConfig> {
        match idx.0 {
            0 => Some(&mut self.first),
            n => self.rest.get_mut(usize::try_from(n).ok()? - 1),
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

/// HOW MANY WORK SLICES ONE DIM IS CUT INTO — `numWkSlicesPerDim_`'s value (`dsc/superdsc.h:69`).
///
/// ⭐ NON-ZERO, WHICH IS WHAT MAKES THE PRODUCT A COUNT: every writer states `1` or `numCoresUsed`
/// (`dbo/src/Transforms/sdsc_bundle/ProgramCorrection.cpp:1094`, `:1396`, `:1399`;
/// `GatherIndexConversion.cpp:100`; `dsc/dsm.cpp:22041`), so the zero that would silently zero the
/// reference's `numWkSlices` is unspellable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WkSliceCount(NonZeroU32);

impl WkSliceCount {
    /// ONE SLICE — `unsigned numWkSlices = 1`, which is the product's identity.
    pub const ONE: Self = Self(NonZeroU32::MIN);

    /// A stated count.
    #[must_use]
    pub const fn new(count: NonZeroU32) -> Self {
        Self(count)
    }

    /// The count.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// `numWkSlices *= count`, [`None`] where the reference's `unsigned` product would WRAP.
    #[must_use]
    pub const fn times(self, count: Self) -> Option<Self> {
        match self.0.checked_mul(count.0) {
            Some(product) => Some(Self(product)),
            None => None,
        }
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

/// THE SUPER-DSC THIS STAGE SCHEDULES — `SuperDsc` (`dsc/superdsc.h:67`) reduced to the fields this
/// batch reads.
#[derive(Debug, Clone, PartialEq)]
pub struct SuperDsc {
    dscs: DscList,
    /// `numWkSlicesPerDim_` (`dsc/superdsc.h:69`), absent for a dim nothing sliced — that `.at()`'s
    /// throw, which entry 199 multiplies straight into its product and entry 210 reaches with no
    /// guard at all.
    pub num_wk_slices_per_dim: BTreeMap<PrimaryDim, WkSliceCount>,
    /// `coreIdToWkSlice_`.
    pub core_id_to_wk_slice: BTreeMap<Core, WkSlice>,
    /// `coreIdToDscSchedule` (`dsc/superdsc.h:77`), absent for a core the super-DSC states no
    /// schedule for — that `.at()`'s throw.
    pub core_id_to_dsc_schedule: BTreeMap<Core, Vec<DscScheduleStep>>,
    /// `coreIdToDsc_` (`dsc/superdsc.h:68`) — EVERY core the whole super-DSC schedules, which is
    /// wider than any one DSC's [`DesignSpaceConfig::core_ids_used`].
    ///
    /// ⛔ AN INPUT AND NOT A DERIVED UNION: dbo fills it (`ProgramCorrection.cpp:1064`,
    /// `SdscRelayoutInsertion.cpp:538`), and [`Self::new`] leaves it EMPTY exactly as the
    /// reference's default construction does. Entry 291 reads it to widen a conditional-GTR group
    /// beyond one DSC's cores, and an empty map makes that arm REFUSE rather than answer wrongly.
    pub core_id_to_dsc: BTreeMap<Core, DscIdx>,
    /// `datastageBasedElemOff` (`dsc/superdsc.h:116`), read at `dsc/dsc2.cpp:3034`.
    ///
    /// 🛑 A LATCH AND NOT A CHOICE: entry 379 only ever SETS it, from the first DSC carrying a
    /// `ReStickifyOpLx`/`ReStickifyOpHBM` op onwards, and nothing in the reference tree clears it.
    pub datastage_based_elem_off: bool,
}

impl SuperDsc {
    /// A super-DSC over a non-empty DSC list.
    #[must_use]
    pub const fn new(
        dscs: DscList,
        num_wk_slices_per_dim: BTreeMap<PrimaryDim, WkSliceCount>,
        core_id_to_wk_slice: BTreeMap<Core, WkSlice>,
        core_id_to_dsc_schedule: BTreeMap<Core, Vec<DscScheduleStep>>,
    ) -> Self {
        Self {
            dscs,
            num_wk_slices_per_dim,
            core_id_to_wk_slice,
            core_id_to_dsc_schedule,
            core_id_to_dsc: BTreeMap::new(),
            datastage_based_elem_off: false,
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

/// A DIM'S FRONT AND BACK PADDING — `padFront_`/`padBack_` (`dsc/dims.h:135-136`), whose paired `-1`
/// is not a size but the statement that THE PADDING BELONGS TO NO CHUNK
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:138`, `ddc/ddcv1.cpp:1172`).
///
/// ⛔ THE TWO WRITERS OF THAT `-1` DISAGREE ON A GUARD, so they are TWO operations here:
/// `voidPaddingIfChunking` tests `if (padBack_ != 0 || padFront_ != 0)` first
/// ([`Self::voided_if_padded`]); `exploreAssignDataStages` does not ([`Self::voided`]).
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

    /// `padBack_ = padFront_ = -1` UNCONDITIONALLY — `ddc/ddcv1.cpp:1172` voids the entry it just
    /// emplaced whatever that entry held, and a DEFAULT-CONSTRUCTED one reaches it
    /// (`dsc/dsc2.cpp:3665` and `ddc/ddl/ddl_conversion.cpp:2520` both mint one with `operator[]`).
    /// The `-1` is then `calculate_padded`'s *"Padded access is not valid in datastage"*
    /// (`dsc/dims.cpp:581-582`), so voiding an unpadded dim is OBSERVABLE.
    #[must_use]
    pub const fn voided(self) -> Self {
        match self {
            Self::Unpadded | Self::Sized { .. } | Self::Voided => Self::Voided,
        }
    }

    /// `if (padBack_ != 0 || padFront_ != 0) padBack_ = padFront_ = -1` — the GUARDED void, whose
    /// guard belongs to `voidPaddingIfChunking` alone
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:137-138`).
    #[must_use]
    pub const fn voided_if_padded(self) -> Self {
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

/// Replaces: e002_DimPaddingSizes
///
/// ONE DIM'S PADDING — `DimPaddingSizes` (`dsc/dims.h:134`), ALL EIGHT of its declared fields, as
/// `DataStructDims::paddingSizes_` (`dsc/dims.h:219`) stores one of them per dim.
///
/// ⚠ [`crate::schedule::ddc::v1::PaddingSizes`] IS A SECOND, LOSSY SPELLING of this same C++ struct
/// — five fields, no `unneededPad` counts — and it is reachable ONLY through the carrier-trait
/// methods `stage_padding_sizes` (`ddc/v1.rs:1625`) and `stage_padding_dims` (`:2319`), both of
/// which BUILD one out of THIS type. It retires when that layer does, which is why the anchor sits
/// here, on the type that STORES the fields, and not where the scheduler filed the TODO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimPadding {
    /// Field: e002_DimPaddingSizes.padFront_
    ///
    /// Field: e002_DimPaddingSizes.padBack_
    ///
    /// `padFront_` (`dsc/dims.h:135`) AND `padBack_` (`:136`) AS ONE VALUE, because BOTH writers
    /// that are not a constructor set them in a single
    /// `padInfo.padBack_ = padInfo.padFront_ = -1`
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:138` and `ddc/ddcv1.cpp:1172`), and a `-1` edge
    /// is not a size on its own. Only the first tests `!= 0` first — see [`PadSizes::voided`].
    pub sizes: PadSizes,
    /// Field: e002_DimPaddingSizes.windowDim_
    ///
    /// `windowDim_` (`dsc/dims.h:142`), whose `PrimaryDimTypesCount` default is the absence of a
    /// window dim — the sentinel that same writer tests for as
    /// `padInfo.windowDim_ != PrimaryDimTypesCount` (`L3DlOpsScheduler.cpp:134`).
    pub window_dim: Option<PrimaryDim>,
    /// Field: e002_DimPaddingSizes.unneededPad_
    ///
    /// Field: e002_DimPaddingSizes.unneededPadFront_
    ///
    /// Field: e002_DimPaddingSizes.unneededPadBack_
    ///
    /// `unneededPad_` (`dsc/dims.h:137`), `unneededPadFront_` (`:138`) and `unneededPadBack_`
    /// (`:139`) AS ONE TRIPLE, because every writer clears all three in a single assignment.
    ///
    /// ⛔ ONE OF THE TWO CLEARS IS LIVE: `L3DlOpsScheduler.cpp:144` sits behind a
    /// `carryUnneededPadToChunk` whose value is `true` (`:48`), but `ddc/ddcv1.cpp:1170-1171`
    /// clears the triple with NO such guard, so a chunk stage's counts do not always arrive from
    /// the core stage's intact.
    pub unneeded: UnneededPad,
    /// Field: e002_DimPaddingSizes.stride_
    ///
    /// `stride_` (`dsc/dims.h:140`), whose declared default is `1` and not `0`.
    pub stride: Stride,
    /// Field: e002_DimPaddingSizes.dilation_
    ///
    /// `dilation_` (`dsc/dims.h:141`), whose declared default is likewise `1`.
    pub dilation: Dilation,
}

impl DimPadding {
    /// `getMetaDimVal(kind)` (`dsc/dims.cpp:59-72`) — the one stored number a meta dim kind names
    /// directly.
    ///
    /// ⛔ [`None`] IS *"Impossible to get the direct value of MetaDimKind"*: only four of the eight
    /// kinds name a field. A [`PadSizes::Voided`] edge answers the reference's `-1`, which is a
    /// statement and not a size.
    #[must_use]
    pub const fn meta_dim_val(&self, kind: MetaDimKind) -> Option<i64> {
        let (front, back) = match self.sizes {
            PadSizes::Unpadded => (0, 0),
            PadSizes::Sized { front, back } => (front.0 as i64, back.0 as i64),
            PadSizes::Voided => (-1, -1),
        };
        match kind {
            MetaDimKind::Dilation => Some(self.dilation.0),
            MetaDimKind::Stride => Some(self.stride.get()),
            MetaDimKind::PadFront => Some(front),
            MetaDimKind::PadBack => Some(back),
            _ => None,
        }
    }
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
            dilation: Dilation::ONE,
        }
    }
}

/// Replaces: e003_SymbolicDimInfo
///
/// A SYMBOLIC DIM'S BOUNDS — `SymbolicDimInfo` (`dsc/dims.h:148`), both of its declared fields, as
/// `DataStructDims::symbolicDimInfo_` stores one of them per symbolic dim.
///
/// ⛔ NEITHER FIELD CAN SPELL ITS OWN `-1` INITIALISER, AND THAT IS A GUARD AND NOT A GAP: a `-1`
/// here is not a bound, it is a field nobody set, and SIX read sites in `dsc/dims.cpp` consume it
/// as though it were one — `:524` and `:526` as the dim's own value, `:744` and `:747` in the
/// volume pruner, `:623` and `:784` as the `maxSize_ / granularity_` factor. See [`MaxSize`] and
/// [`Granularity`] for the reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicDimInfo {
    /// Field: e003_SymbolicDimInfo.maxSize_
    ///
    /// `maxSize_` (`dsc/dims.h:149`) — the largest extent the dim may take.
    pub max_size: MaxSize,
    /// Field: e003_SymbolicDimInfo.granularity_
    ///
    /// `granularity_` (`dsc/dims.h:150`) — the step it takes them in.
    pub granularity: Granularity,
}

/// A SYMBOLIC DIM'S LARGEST SIZE — `maxSize_` (`dsc/dims.h:149`).
///
/// ⛔ DIVERGENCE, AND THE FIELD'S DEFAULT IS A REFERENCE DEFECT, EXACTLY AS FOR [`Granularity`]:
/// `primaryDimToVal_base_st` returns `maxSize_` AS the dim's value (`dsc/dims.cpp:526`), so an unset
/// field reaches the caller as a NEGATIVE EXTENT; and `pruneMaxSymbolicVolumes` folds it into
/// `mulOfMaxes *= symbolicDimInfo_.at(symDim).maxSize_` (`dsc/dims.cpp:744`), where one unset dim
/// NEGATES the product every other dim contributed to. An unsigned count cannot spell either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaxSize(pub u32);

/// A SYMBOLIC DIM'S STEP — `granularity_` (`dsc/dims.h:150`), which the pruner DIVIDES a volume
/// limit by.
///
/// ⛔ DIVERGENCE, AND THE FIELD'S DEFAULT IS A REFERENCE DEFECT: `granularity_ = -1` makes
/// `myVolumeLimit % dimGranularity == 0` pass for every value and `myVolumeLimit /= dimGranularity`
/// NEGATE the limit (`dsc/dims.cpp:748-749`). A non-zero positive step cannot spell that.
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
/// ⭐ THE INVARIANT IS `DT_CHECK(refDstg.symbolicDimInfo_.count(symDim))` (`dsc/dims.cpp:746`):
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

    /// Replaces: e003_scaleFromMaxToGranularity
    ///
    /// RE-EXPRESSES A SIZE FROM MAX UNITS IN GRANULARITY UNITS — a symbolic dim's corelet and row
    /// splits are filled against `maxSize_`, so a granularity-unit reader divides by
    /// `maxSize_ / granularity_`. A dim `symbolicDimInfo_` does not name passes through UNTOUCHED.
    ///
    /// ⛔ [`None`] IS BOTH `DT_CHECK`s AT ONCE (`dsc/dims.cpp:623`, `:625`): a `maxSize_` the
    /// granularity does not divide, a zero factor, and a `val` the factor does not divide.
    #[must_use]
    pub fn scale_from_max_to_granularity(&self, dim: PrimaryDim, val: Extent) -> Option<Extent> {
        let Some(info) = self.info.get(&dim) else {
            return Some(val);
        };
        if info.max_size.0 % info.granularity.get() != 0 {
            return None;
        }
        let factor = i64::from(info.max_size.0 / info.granularity.get());
        if factor == 0 || val.0 % factor != 0 {
            return None;
        }
        Some(Extent(val.0 / factor))
    }

    /// `maxSymbolicVolume_ = ref.maxSymbolicVolume_` THEN `pruneMaxSymbolicVolumes(ref)` — the FUSED
    /// call shape, which is ONE of the reference's four
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:171-172`); the other three are UNFUSED and take
    /// [`Self::prune_volumes`].
    ///
    /// ⭐ `DT_CHECK(refDstg.symbolicDimInfo_.count(symDim))` (`dsc/dims.cpp:746`) CANNOT FIRE HERE:
    /// the limits are `reference`'s own, and its own `info` names their keys by construction.
    pub fn prune_volumes_from(&mut self, reference: &Symbolic) {
        let stated = StatedVolumes::new(reference.volumes.clone());
        let pruned = stated.pruned_against(self, reference);
        self.volumes = pruned;
    }

    /// `pruneMaxSymbolicVolumes(refDstg)` UNFUSED — THIS stage's OWN limits re-keyed onto the dims it
    /// still calls symbolic, which is what `ddc/ddcv1.cpp:1424`, `:1425` and `dsc/dsc2.cpp:3719` ask
    /// for; adopting `reference`'s map there DISCARDS the stage's own limits for the core's.
    ///
    /// ⛔ AND IT IS THE IDENTITY UNDER [`Symbolic`]'S OWN INVARIANT. `needPruning` is
    /// `any_of(symDims, !symbolicDimInfo_.count(dim))` (`dsc/dims.cpp:732-734`), [`Symbolic::new`]
    /// DROPS exactly those keys, and no mutator here can re-create one — so at the one production
    /// caller (`stages/ddc_sites.rs:663`) every entry short-circuits and the map comes back unchanged.
    /// The reference reaches the pre-prune state through `makeDimNotSymbolic`, which erases from
    /// `symbolicDimInfo_` and leaves `maxSymbolicVolume_` alone (`dsc/dims.cpp:781-787`); ours STOPS
    /// there instead. The walk does real work only through [`Self::prune_volumes_from`] and entry
    /// 015's direct [`StatedVolumes::pruned_against`] (`l3/capacity.rs:425`).
    pub fn prune_volumes(&mut self, reference: &Symbolic) {
        let stated = StatedVolumes::new(std::mem::take(&mut self.volumes));
        let pruned = stated.pruned_against(self, reference);
        self.volumes = pruned;
    }
}

/// Replaces: e006_DataStructDims
///
/// A DATA STAGE'S DIMS — `DataStructDims` (`dsc/dims.h:158`), 19 of its 28 declared members; the
/// remaining nine are its DEPRECATED block and they are absent BY MEASUREMENT, not by narrowing.
///
/// ⛔⛔ `r_` `c_` `rc_` `si_` `sj_` `sij_` `zi_` `zj_` `zij_` (`dsc/dims.h:166`, `:175-176`,
/// `:181-182`, `:185-186`, `:192-193`) ARE CLEARED TO `-1` FOR EVERY DATA STAGE BEFORE DDC READS
/// ONE — `clearDeprecatedFields` assigns all nine in one statement over `dataStageParam_`'s `ss_`
/// and `el_` (`ddc/ddcv1.cpp:2081-2090`), and the header marks two of them *"to be removed in
/// future.."* (`:174`). Nothing on this campaign's path names any of the nine: neither the L3
/// scheduler (`dcg/dcg_fe/scheduler/`) nor `ddc/ddl/` mentions one, and the only writers tree-wide
/// are `dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp:314-346`, off this path. So [`Self::compound`] computing
/// TWO of the reference's five products is complete, not short.
///
/// ⭐ `name_` (`dsc/dims.h:160`) IS CARRIED, by [`NamedDims`] — the half of this same C++ class that
/// pairs the name with these dims.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StageDims {
    /// Field: e006_DataStructDims.in_
    ///
    /// Field: e006_DataStructDims.out_
    ///
    /// Field: e006_DataStructDims.mb_
    ///
    /// Field: e006_DataStructDims.ij_
    ///
    /// Field: e006_DataStructDims.kij_
    ///
    /// Field: e006_DataStructDims.y_
    ///
    /// Field: e006_DataStructDims.x_
    ///
    /// Field: e006_DataStructDims.x1_
    ///
    /// Field: e006_DataStructDims.i_
    ///
    /// Field: e006_DataStructDims.j_
    ///
    /// Field: e006_DataStructDims.ki_
    ///
    /// Field: e006_DataStructDims.kj_
    ///
    /// THE TWELVE SLOTS `PrimaryDimTypes` NAMES, AS ONE MAP — `primaryDimToValHandler_st`
    /// (`dsc/dims.cpp:485-514`) is the dim-to-field switch, and its twelve arms are exactly these
    /// twelve members with `DT_ERROR("Invalid PrimaryDim")` for every other. So a key here IS the
    /// field, and no `DataStructDims` member reachable by a `PrimaryDimTypes` is outside this map.
    ///
    /// [`StageDims::extent`] reads one RAW; [`StageDims::whole_extent`] is the
    /// `primaryDimToVal_st(dim)` reading of it.
    pub extents: BTreeMap<PrimaryDim, Extent>,
    /// Field: e006_DataStructDims.paddingSizes_
    ///
    /// `paddingSizes_` (`dsc/dims.h:219`).
    pub padding: BTreeMap<PrimaryDim, DimPadding>,
    /// Field: e006_DataStructDims.symbolicDimInfo_
    ///
    /// Field: e006_DataStructDims.maxSymbolicVolume_
    ///
    /// `symbolicDimInfo_` (`:197`) with `maxSymbolicVolume_` (`:202`) — see [`Symbolic`] for why the
    /// pruner makes them one value.
    pub symbolic: Symbolic,
    /// Field: e006_DataStructDims.coreletSplit_
    ///
    /// `coreletSplit_` (`:206`) — per corelet-split dim, one extent per corelet of the core.
    pub corelet_split: BTreeMap<PrimaryDim, Vec<Extent>>,
    /// Field: e006_DataStructDims.rowSplit_
    ///
    /// `rowSplit_` (`dsc/dims.h:209`) — per corelet, that corelet's share of the dim broken up
    /// across the PT's rows, indexed BY ROW.
    ///
    /// ⛔ THE OUTER KEY IS THE CORELET AND ONLY THE CORELETS THE SPLIT NAMES ARE PRESENT, which is
    /// what makes `primaryDimToVal_st`'s `clId = -1` arm read the FIRST corelet and not the core
    /// (`dsc/dims.cpp:673-675`).
    pub row_split: BTreeMap<PrimaryDim, BTreeMap<Corelet, Vec<Extent>>>,
    /// Field: e006_DataStructDims.peSfpSplit_
    ///
    /// `peSfpSplit_` (`dsc/dims.h:210-214`) — per corelet, the PE's and the SFP's shares.
    ///
    /// ⭐ BOTH SIDES ARE ALWAYS PRESENT, so `.at(peOrSfp)`'s throw (`dsc/dims.cpp:687`, `:691`, `:694`) is
    /// discharged by [`PeSfpShares`] rather than checked; its own doc names the three writers.
    pub pe_sfp_split: BTreeMap<PrimaryDim, BTreeMap<Corelet, PeSfpShares>>,
}

impl StageDims {
    /// THE RAW SLOT of one primary dim — the `i_`/`j_`/… field naming `dim`, [`None`] for the `-1`
    /// an unwritten field carries. `compound()` (`dsc/dims.cpp:84`) and [`Self::scaled_extent`]'s
    /// fall-through each read the FIELD, which is why this stays.
    ///
    /// ⛔ NOT `primaryDimToVal_st(dim)` — that is [`Self::whole_extent`], which answers a symbolic
    /// dim's `maxSize_` instead of its slot and refuses a negative one.
    #[must_use]
    pub fn extent(&self, dim: PrimaryDim) -> Option<Extent> {
        self.extents.get(&dim).copied()
    }

    /// Replaces: e010_primaryDimToVal_base_st
    ///
    /// One dim's stated extent — `symbolicDimInfo_`'s `maxSize_` or `granularity_` where the dim is
    /// symbolic, else the stored slot — scaled by `dimDensity` and then rewritten by
    /// [`Self::calculate_padded`]. This is also `primaryDimToVal_st(dim, NO_COMPONENT, -1, -1, ..)`
    /// (`dsc/dims.h:269-273`): no component and a negative row/corelet id fall through to here.
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S `-1` — an unstated slot, plus every abort of
    /// [`Self::calculate_padded`]. `DT_CHECK(dimDensity > 0.0 && dimDensity <= 1.0)`
    /// (`dsc/dims.cpp:558`) is discharged by [`ScaleBlock`], whose reciprocal cannot leave that
    /// range.
    ///
    /// ⭐ AND THE BODY'S THIRD ABORT, `DT_ERROR("Invalid PrimaryDim")` (`dsc/dims.cpp:553-556`), IS
    /// DISCHARGED BY THE ENUM AS IT ALREADY IS IN THE REFERENCE: the twelve arms cover every
    /// `PrimaryDimTypes` but the `PrimaryDimTypesCount` terminator (`dsc/dims.h:34-48`).
    ///
    /// ⛔ DIVERGENCE: INTEGER DIVISION where the reference multiplies by the `double` `1.0/blkSize`
    /// and truncates — equal for every power-of-two block, one short for a block of three.
    #[must_use]
    pub fn scaled_extent(
        &self,
        dim: PrimaryDim,
        padded: &PaddingForm,
        density: Option<ScaleBlock>,
        granularity: bool,
    ) -> Option<Extent> {
        let stated = match self.symbolic.info().get(&dim) {
            Some(info) if granularity => i64::from(info.granularity.get()),
            Some(info) => i64::from(info.max_size.0),
            None => self.extent(dim)?.0,
        };
        let val = match density {
            Some(block) => stated / i64::try_from(block.count().0).unwrap_or(i64::MAX),
            None => stated,
        };
        self.calculate_padded(dim, Extent(val), padded, granularity)
    }

    /// Replaces: e009_calculate_padded
    ///
    /// `val` rewritten by `dim`'s padding: the front and back edges for a non-window dim, or the
    /// window span `wSize + (val - 1) * stride_` for a windowed one, across the six `PadType`s.
    ///
    /// ⭐ `val` IS A PARAMETER, NOT A READ. `ddc/ddc_fold.cpp:367` and `primaryDimToVal_st`
    /// (`dsc/dims.cpp:701`) each pass a val this function did not read, so it cannot be fused into
    /// [`Self::scaled_extent`]; `granularity` only reaches the recursive window read.
    ///
    /// ⛔ [`None`] IS THE `-1` AND EVERY ABORT AT ONCE: a negative `val` (short-circuited BEFORE any
    /// abort, `:567-568`), *"Cannot calculate padded version of compound dim"*, a missing
    /// `paddingSizes_` entry, *"Padded access is not valid in datastage"* for a
    /// [`PadSizes::Voided`] edge, *"Missing window size"*, and each *"Unsupported padding type"*.
    /// Spans SATURATE rather than wrap.
    #[must_use]
    pub fn calculate_padded(
        &self,
        dim: PrimaryDim,
        val: Extent,
        padded: &PaddingForm,
        granularity: bool,
    ) -> Option<Extent> {
        let val = val.0;
        if val < 0 {
            return None;
        }
        let pad_type = padded.padding(dim);
        if pad_type == PadType::NoPad {
            return Some(Extent(val));
        }
        if matches!(dim, PrimaryDim::Ij | PrimaryDim::Kij) {
            return None;
        }
        let pad = self.padding.get(&dim)?;
        let unneeded = i64::from(pad.unneeded.total.0);
        let span = match pad.window_dim {
            None => {
                let (front, back) = match pad.sizes {
                    PadSizes::Voided => return None,
                    PadSizes::Unpadded => (0, 0),
                    PadSizes::Sized { front, back } => (i64::from(front.0), i64::from(back.0)),
                };
                let edges = val.saturating_add(front).saturating_add(back);
                match pad_type {
                    PadType::PaddedFullSpanWUnneeded => edges.saturating_add(unneeded),
                    PadType::PaddedFullSpan => edges,
                    _ => return None,
                }
            }
            Some(window) => {
                let window_size = self
                    .scaled_extent(window, &PaddingForm::default(), None, granularity)
                    .filter(|size| size.0 >= 1)?;
                let strided = window_size
                    .0
                    .saturating_add(val.saturating_sub(1).saturating_mul(pad.stride.get()));
                match pad_type {
                    PadType::PaddedFullSpanWUnneeded => strided.saturating_add(unneeded),
                    PadType::PaddedWZeroPad => strided,
                    PadType::PaddedNoZeroPad => {
                        let (front, back) = match pad.sizes {
                            PadSizes::Voided => return None,
                            PadSizes::Unpadded => (0, 0),
                            PadSizes::Sized { front, back } => {
                                (i64::from(front.0), i64::from(back.0))
                            }
                        };
                        strided
                            .saturating_add(unneeded)
                            .saturating_sub(i64::from(pad.unneeded.front.0))
                            .saturating_sub(i64::from(pad.unneeded.back.0))
                            .saturating_sub(front)
                            .saturating_sub(back)
                    }
                    PadType::LoweredPadded => window_size.0.saturating_mul(val),
                    _ => return None,
                }
            }
        };
        Some(Extent(span))
    }

    /// Replaces: e011_primaryDimToVal_clView_st
    ///
    /// ONE CORELET'S VIEW OF A DIM — that corelet's `coreletSplit_` share, re-expressed in
    /// granularity units where asked, density-scaled, then rewritten by [`Self::calculate_padded`]. A
    /// dim the split does not name, or no corelet at all, is [`Self::scaled_extent`]'s whole-core
    /// answer.
    ///
    /// ⛔ A SHORT SPLIT IS A STOP, NOT A FALL-THROUGH: once `clId >= 0 && coreletSplit_.count(d)` both
    /// hold the reference is committed to `.at(clId)` (`dsc/dims.cpp:635`) and that throw is
    /// [`None`] — answering the base extent there hands one corelet the WHOLE core's extent.
    ///
    /// ⛔ DIVERGENCE, AS IN [`Self::scaled_extent`]: INTEGER DIVISION for the `double` density.
    #[must_use]
    pub fn corelet_extent(
        &self,
        dim: PrimaryDim,
        corelet: Option<Corelet>,
        padded: &PaddingForm,
        density: Option<ScaleBlock>,
        granularity: bool,
    ) -> Option<Extent> {
        let (Some(corelet), Some(split)) = (corelet, self.corelet_split.get(&dim)) else {
            return self.scaled_extent(dim, padded, density, granularity);
        };
        let share = *split.get(usize::try_from(corelet.get()).ok()?)?;
        let share = if granularity {
            self.symbolic.scale_from_max_to_granularity(dim, share)?
        } else {
            share
        };
        let scaled = match density {
            Some(block) => share.0 / i64::try_from(block.count().0).unwrap_or(i64::MAX),
            None => share.0,
        };
        self.calculate_padded(dim, Extent(scaled), padded, granularity)
    }

    /// `rowSplit_.at(dim)` READ AT ONE ROW — one named corelet's share, else the SUM over the
    /// corelets the split names when `coreletSplit_` names the dim too, else the FIRST corelet's
    /// share alone (`dsc/dims.cpp:666-676`).
    fn row_share(
        &self,
        dim: PrimaryDim,
        per_corelet: &BTreeMap<Corelet, Vec<Extent>>,
        row: Row,
        corelet: Option<Corelet>,
    ) -> Option<Extent> {
        let index = usize::try_from(row.get()).ok()?;
        let at_row = |shares: &[Extent]| shares.get(index).copied();
        match corelet {
            Some(corelet) => at_row(per_corelet.get(&corelet)?),
            None if self.corelet_split.contains_key(&dim) => per_corelet
                .values()
                .try_fold(0i64, |sum, shares| {
                    Some(sum.saturating_add(at_row(shares)?.0))
                })
                .map(Extent),
            None => at_row(per_corelet.values().next()?),
        }
    }

    /// `peSfpSplit_.at(dim)` READ ON ONE SIDE — the same three-way over the PE's or the SFP's half
    /// (`dsc/dims.cpp:686-696`).
    fn pe_sfp_share(
        &self,
        dim: PrimaryDim,
        per_corelet: &BTreeMap<Corelet, PeSfpShares>,
        comp: VectorComp,
        corelet: Option<Corelet>,
    ) -> Option<Extent> {
        match corelet {
            Some(corelet) => Some(per_corelet.get(&corelet)?.get(comp)),
            None if self.corelet_split.contains_key(&dim) => {
                Some(Extent(per_corelet.values().fold(0i64, |sum, shares| {
                    sum.saturating_add(shares.get(comp).0)
                })))
            }
            None => Some(per_corelet.values().next()?.get(comp)),
        }
    }

    /// Replaces: e012_primaryDimToVal_st
    ///
    /// ONE SAMPLE'S VIEW OF A DIM — the PT row's `rowSplit_` share where the sample names a row AND
    /// the stage splits that dim across rows, else the PE's or the SFP's `peSfpSplit_` share where it
    /// names a vector component, else [`Self::corelet_extent`]'s corelet view. The share is
    /// re-expressed in granularity units where asked, density-scaled, then rewritten by
    /// [`Self::calculate_padded`] — the same tail as the corelet view.
    ///
    /// ⛔ `clId = -1` IS A SUM ONLY WHERE THE DIM IS ALSO CORELET-SPLIT (`dsc/dims.cpp:669-675`);
    /// otherwise it is the FIRST corelet's share alone. Summing unconditionally would multiply a
    /// single-corelet stage's extent by the corelet count.
    ///
    /// ⛔ EVERY `.at` IS A STOP, NOT A FALL-THROUGH, exactly as in [`Self::corelet_extent`]: a
    /// corelet the split does not name (`:667`) and a row past the end of its vector both throw, and
    /// answering the whole core there hands one row the core's extent.
    ///
    /// ⛔ AND `rowSplit_.at(d).begin()` ON AN EMPTY INNER MAP (`:674`) IS [`None`] — the reference
    /// dereferences its end iterator there.
    ///
    /// ⭐ `PELRF -> PE` AND `SFPLRF -> SFP` (`:659-663`) ARE DISCHARGED BY THE TYPE: [`VectorComp`]
    /// spells only the two compute components, and `v1::sampled_as` is where a [`SenComponent`] is
    /// mapped onto it.
    ///
    /// ⭐ THE PE/SFP ARM IS PORTED, NOT DEFERRED, though `peSfpSplit_` is empty on all 187 g0
    /// reference exports: [`PeSfpShares`] already spells both halves, so the arm costs a `todo!` that
    /// a bundle outside g0 could walk into.
    ///
    /// ⛔ DIVERGENCE, AS IN [`Self::scaled_extent`]: INTEGER DIVISION where the reference multiplies
    /// by the `double` `1.0/blkSize` and truncates.
    #[must_use]
    pub fn sampled_extent(
        &self,
        dim: PrimaryDim,
        at: DimSample,
        padded: &PaddingForm,
        density: Option<ScaleBlock>,
        granularity: bool,
    ) -> Option<Extent> {
        let share = if let (Some(row), Some(per_corelet)) = (at.row, self.row_split.get(&dim)) {
            self.row_share(dim, per_corelet, row, at.corelet)?
        } else if let (Some(comp), Some(per_corelet)) = (at.comp, self.pe_sfp_split.get(&dim)) {
            self.pe_sfp_share(dim, per_corelet, comp, at.corelet)?
        } else {
            return self.corelet_extent(dim, at.corelet, padded, density, granularity);
        };
        let share = if granularity {
            self.symbolic.scale_from_max_to_granularity(dim, share)?
        } else {
            share
        };
        let scaled = match density {
            Some(block) => share.0 / i64::try_from(block.count().0).unwrap_or(i64::MAX),
            None => share.0,
        };
        self.calculate_padded(dim, Extent(scaled), padded, granularity)
    }

    /// Replaces: e013_primaryDimToVal_st_1arg
    ///
    /// ONE DIM AS THE STAGE STATES IT — [`Self::sampled_extent`] at [`DimSample::WHOLE`]: unpadded,
    /// full density, the max symbolic size, and no row, corelet or component named.
    ///
    /// ⛔ NOT [`Self::extent`], WHICH IS THE RAW SLOT: a row-split dim answers the WHOLE here because
    /// `ptrowId`/`clId` default to `-1` (`dsc/dims.cpp:647-649`), a symbolic dim answers `maxSize_`
    /// rather than its slot, and a negative slot is [`None`].
    #[must_use]
    pub fn whole_extent(&self, dim: PrimaryDim) -> Option<Extent> {
        self.sampled_extent(dim, DimSample::WHOLE, &PaddingForm::default(), None, false)
    }

    /// `primaryDimToVal_st(dim, NO_COMPONENT, -1, -1, {dim, pad})` — [`Self::scaled_extent`] with the
    /// default density and the max symbolic size, which is what every caller that names one padding
    /// type asks for.
    #[must_use]
    pub fn padded_extent(&self, dim: PrimaryDim, pad: PadType) -> Option<Extent> {
        let mut form = PaddingForm::default();
        form.set_padding(dim, pad);
        self.scaled_extent(dim, &form, None, false)
    }

    /// `hasPadding` (`L3DlOpsScheduler.cpp:1071-1075`) — the dim has a `paddingSizes_` entry AND its
    /// `PADDED_FULLSPAN_WUNNEEDED` span differs from its plain extent.
    ///
    /// ⛔ [`None`] IS [`Self::padded_extent`]'s ABORT SET. ⭐ AN UNSTATED DIM IS `Some(false)`, NOT an
    /// abort — `val < 0` short-circuits to `-1` on both sides of the comparison.
    ///
    /// ⛔ THE PLAIN SIDE IS `primaryDimToVal_st(dim)` ([`Self::whole_extent`]), NOT THE RAW SLOT
    /// (`L3DlOpsScheduler.cpp:1075`): a symbolic dim compares its `maxSize_` on BOTH sides, and a
    /// negative slot is the `-1` both sides short-circuit to rather than an abort.
    #[must_use]
    pub fn has_padding(&self, dim: PrimaryDim) -> Option<bool> {
        if !self.padding.contains_key(&dim) {
            return Some(false);
        }
        let Some(plain) = self.whole_extent(dim) else {
            return Some(false);
        };
        Some(self.padded_extent(dim, PadType::PaddedFullSpanWUnneeded)? != plain)
    }

    /// `DataStructDims::compound` (`dsc/dims.cpp:84`) — `IJ = I·J` and `KIJ = KI·KJ`, and a product
    /// with an absent or negative operand is the compound dim's OWN absence, which is the `-1` the
    /// reference writes.
    ///
    /// ⛔ THE REFERENCE ALSO WRITES `zij_`, `sij_` AND `rc_`, AND ALL THREE ARE ALWAYS `-1` HERE.
    /// Their six operands are the deprecated block `clearDeprecatedFields` sets to `-1` for every
    /// data stage before DDC reads one (`ddc/ddcv1.cpp:2081-2090`), so each product's guard fails on
    /// both sides and each lands on the reference's own `-1`. No `PrimaryDimTypes` value names any of
    /// the nine either (`dsc/dims.cpp:485-514`). TWO PRODUCTS IS THE WHOLE OF IT ON THIS PATH.
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

    /// `coreletSplit_`, for the scheduler to state a corelet split with — writing it cannot empty
    /// the extents, so the non-emptiness survives it.
    pub const fn corelet_split_mut(&mut self) -> &mut BTreeMap<PrimaryDim, Vec<Extent>> {
        &mut self.0.corelet_split
    }

    /// Replaces: e002_primaryDimToValHandler_st
    ///
    /// THE ADDRESSABLE SLOT OF ONE PRIMARY DIM — `double&` for the field naming `d`. Adding an extent
    /// cannot empty the map, so [`FilledDims`]' non-emptiness survives it; the READ half is
    /// [`StageDims::extent`], whose [`None`] is the `-1` an unwritten field carries.
    ///
    /// ⛔ A BARE SETTER IS FAITHFUL ONLY BECAUSE THIS CLOSURE'S ONLY CALLER ASSIGNS. Entry 015's four
    /// calls are plain assignments (`dsc/dsc2.cpp:3661-3662`, `:3685-3688`), as is
    /// `ddc/ddc_transformation_util.cpp:1270`. REPO-WIDE THE `double&` IS ALSO READ AND
    /// READ-MODIFY-WRITTEN: `dsDim *= numCoreletsPerCore` (`ddc/ddcv1.cpp:1141`), `std::ceil(dsDim …)`
    /// (`:956`), `if (wDim < 0)` (`:1182`), `int origValue = dsDim` then `dsDim == origValue` (`:1187`,
    /// `:1195`), and it is handed to `setValueInDs` as an out-parameter (`:1176`). A caller from
    /// outside this closure needs [`StageDims::extent`] alongside it, not a second setter.
    ///
    /// ⭐ TOTAL: its 12 arms are exactly the 12 `PrimaryDimTypes` (`dsc/dims.h:34-47`), spelled without
    /// `PrimaryDimTypesCount`, so `DT_ERROR("Invalid PrimaryDim")` is unspellable.
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
/// `std::deque<int64_t>`, "per core, corelet, and sdsc folds" (`dsc/dsc2.h:985-986`).
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

/// WHAT A LABELLED DS'S `memOrg_` ANSWERS — `LabeledDsInfo::memOrg_` (`dsc/dscdefn.h:337`) reduced to
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
    /// outside the THREE [`Buffering`] names. WHICH of the three an arm admits is entry 050's own
    /// check, not this seam's — [`Buffering::Streaming`] reaches here and only the pinned arm may.
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
    /// (`L3DlOpsScheduler.cpp:7157`, `:7160`).
    fn hbm_allocation(&self) -> Option<NodeName>;

    /// That node's `layoutDimOrder_` (`dsc/dsc2.h:982`), whose non-emptiness is [`LayoutDims`]' own
    /// — which is *"Expect valid layoutDimOrder_."* (`:6719`).
    fn hbm_layout_dims(&self) -> Option<LayoutDims>;

    /// `memOrg_.at(LX).isZeroPadded != ZpType::NOZEROPAD` FUSED WITH the `DT_CHECK_MSG(isPadded,
    /// "Expect memOrg_ LX isPadded is true.")` beside it (`L3DlOpsScheduler.cpp:5320`).
    ///
    /// ⛔ [`None`] IS THAT CHECK: a zero-padded LX organisation that is not padded. `Some(false)` is
    /// no LX entry and an LX entry that zero-pads nothing, which are one answer to every reader.
    fn lx_zero_padded(&self) -> Option<bool>;

    /// `getPageSize()`'s KEY SET on that node (`dsc/dsc2.cpp:4480`), EMPTY where nothing pages.
    ///
    /// ⛔ THE SIZES ARE DELIBERATELY NOT ASKED FOR: every reader in scope asks `pageSize.count(dim)`
    /// and nothing more, and `maxDimSizes_` entries are datastage KEYS as often as element counts.
    /// ⭐ ONE UNBOUNDED ENTRY WINS WHEREVER IT SITS IN THE REFERENCE — it ERASES an already-multiplied
    /// dim on meeting a negative `maxSize` and skips every later entry naming it
    /// (`dsc/dsc2.cpp:4498-4511`), so the set IT returns is "bounded by the layout and never left
    /// unbounded", whatever order the layout states them in.
    /// ⛔ THAT IS THE REFERENCE AND NOT THIS SEAM. See [`Self::hbm_page_sizes`] for the three ways
    /// the one production implementation of it departs from `getPageSize()`.
    fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim> {
        self.hbm_page_sizes().unwrap_or_default().into_keys().collect()
    }

    /// ONE PAGE'S ELEMENTS PER LAYOUT DIM OF `memOrg_.at(HBM).allocateNode_`, [`None`] where
    /// `memOrg_` names no `HBM` or its entry holds no node — *"Exepect HBM in memOrg_."* and
    /// *"Expect a valid HBM allocate node."* both (`L3DlOpsScheduler.cpp:6686-6691`).
    ///
    /// ⛔ IT IS NOT `dsc2::AllocateNode::getPageSize()` ITSELF. That body is `dsc/dsc2.cpp:4480-4513`
    /// and it is ported, once, as
    /// [`AllocateNode::page_sizes`](crate::schedule::dsc2::AllocateNode::page_sizes). The only
    /// production implementation of THIS seam is `Org::page_sizes` (`schedule/stages/tree.rs:785`),
    /// which reads the layout alone and so departs from the reference three ways:
    /// 1. NO `indirectAllocType_` GATE. The reference reads that field FIRST and returns an EMPTY
    ///    map for `NO_INDIRECTION` (`dsc/dsc2.cpp:4483-4485`) — the arm every program we compile
    ///    takes, by CONSTRUCTION: one emitter site, the literal `"no_indirection"`
    ///    (`crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs:5212`). The seam answers from
    ///    the layout whether the allocation indirects or not.
    /// 2. NO `INDEX_TENSOR` REDIRECTION. The reference answers an index tensor from
    ///    `relatedIndirectAccessAlloc_`'s layout and not its own (`dsc/dsc2.cpp:4489-4492`).
    /// 3. A REPEATED DIM IS OVERWRITTEN, NOT MULTIPLIED, and an unbounded one is SKIPPED rather than
    ///    ERASING the size already accumulated for it (`dsc/dsc2.cpp:4498-4511`).
    ///
    /// ⭐ ALL THREE ARE INERT TODAY, AND ONLY BECAUSE OF THE DATA: every production mint of the
    /// layout writes `None` for every max size (`stages/tree.rs:910`,
    /// `ddc/transformation_util.rs:319`, `l3/dl_ops.rs:953`), which is the reference's
    /// `maxDimSizes_ == [-1,…]` measured on all 187 g0 programs, so the map comes back empty and no
    /// reader can tell the two apart. The first mint that writes a real max size makes departure 1
    /// visible to every reader at once.
    ///
    /// ⭐ THE ONE FACT [`MemOrg::hbm_page_dims`] IS A VIEW OF, so a paged dim and its page size can
    /// never disagree about whether the dim pages at all.
    fn hbm_page_sizes(&self) -> Option<BTreeMap<PrimaryDim, Extent>>;

    /// `memOrg_.at(SenComponents::LX).allocateNode_->padding_` (`dsc/dsc2.h:983`), [`None`] where
    /// `memOrg_` names no `LX` or its entry holds no node — *"Expect LX in memOrg_."* and *"Expect a
    /// valid allocate node."* both.
    fn lx_padding(&self) -> Option<PaddingForm>;

    /// ONE PAGE'S ELEMENTS PER LAYOUT DIM ON THAT LX NODE, WITH ITS SIZES, empty where nothing pages.
    ///
    /// ⛔ A DIFFERENT NODE FROM [`Self::hbm_page_dims`], AND THE SIZES ARE WHY BOTH EXIST: entry 209
    /// CAPS a chunk parameter at `pageSizes.at(dim)` (`l3/dl_ops.rs:6640`), so the value it reads is
    /// load-bearing, where every reader of the HBM node's page set asks only `count(dim)`.
    /// ⛔ NOR IS IT `getPageSize()` ON THAT NODE: the one production implementation is that same
    /// `Org::page_sizes` handed `SenComponent::Lx` (`schedule/stages/tree.rs:870`), so all three
    /// departures [`Self::hbm_page_sizes`] names hold here too — and entry 209 is the ONE reader not
    /// itself gated on an indirection, so it is where departure 1 would surface first.
    fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent>;

    /// `memOrg_.at(HBM).allocateNode_->allocUsers_` (`dsc/dsc2.h:1000`) by node id — the nodes
    /// `hasAllocUser(node)` answers for. [`None`] is no HBM entry or no node; the EMPTY vector is
    /// `!hasAllocUsers()`, which is a REFUSAL of its own and not this seam's.
    fn hbm_alloc_users(&self) -> Option<Vec<NodeId>>;

    /// The same list on the LX node, which is the branch an input neighbour fetch takes.
    fn lx_alloc_users(&self) -> Option<Vec<NodeId>>;
}

/// EVERY LABELLED DS'S MEMORY ORGANISATION IN ONE SUPER-DSC — `mySDsc.dscs_.at(i).labeledDs_.at(j)
/// .memOrg_`, so a unit that walks two nested lists can reach each entry's allocate nodes.
pub trait MemOrgs {
    /// One labelled DS's organisation, however the caller stores it.
    type Org: MemOrg + ?Sized;

    /// `dscs_.at(dsc).labeledDs_.at(lds).memOrg_`, [`None`] for either `.at()`'s throw.
    fn mem_org(&self, dsc: DscIdx, lds: LdsIdx) -> Option<&Self::Org>;
}

/// ONE TRANSFER NODE AS ENTRY 208 FILTERS IT — a `dsc2::TransferNode` reduced to its identity and
/// the two storages the filter reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3Transfer {
    /// The node itself, which is what an alloc-user list names and what a later insertion point is
    /// computed from.
    pub node: NodeId,
    /// `name_`.
    pub name: NodeName,
    /// `src_.storage_`.
    pub src: SenComponent,
    /// `dstVias_.front().loc_.storage_` — the FRONT destination, which is the only one the filter
    /// reads however many the transfer has.
    pub dst: SenComponent,
}

/// ONE DSC'S TRANSFER NODES — `dsc.scheduleTree_.traverseTreeDFS(nullptr, {TRANSFER})`.
///
/// ⭐ `DT_CHECK_MSG(node->nodeType_ == TRANSFER, "Expect a transfer node.")` IS THE FILTER THAT
/// PRODUCED THE LIST, exactly as [`ScheduleTrees`]' allocate walk re-checks its own cast.
pub trait TransferNodes {
    /// That DSC's transfers in DFS order; EMPTY for an index the super-DSC does not have.
    fn transfers(&self, dsc: DscIdx) -> Vec<L3Transfer>;
}

/// A SCHEDULE TREE AS ENTRY 211 WALKS IT — `getMutableParent()` and a parent's children, both by
/// node id.
///
/// ⭐ EVERY NODE'S PARENT IS A BLOCK AND SO ALWAYS HAS CHILDREN: `prev_` is declared
/// `BlockNode *prev_` (`dsc/dsc2.h:515`) and `getPrev()` and `getMutableParent()` (`:463-464`) are
/// the SAME link, which is what makes *"Parent node must be a block node."* unreachable.
///
/// ⛔ NOT [`fold::ScheduleTree`](crate::schedule::ddc::fold::ScheduleTree), whose `children` takes a
/// BLOCK-only id and so cannot be asked for an arbitrary node's siblings, and NOT
/// [`ScopeTree`](crate::schedule::ddc::transformation::ScopeTree), which carries an ancestry and a
/// scope classification this walk never asks for.
pub trait NodeParents {
    /// `getMutableParent()`, [`None`] at the root.
    fn parent(&self, node: NodeId) -> Option<NodeId>;

    /// `next_` of `parent`, in order; EMPTY for a node that is not a block.
    fn children(&self, parent: NodeId) -> Vec<NodeId>;
}

/// WHICH SIDE OF A REFERENCE NODE SET AN INSERTION LANDS ON — `insertBefore`
/// (`L3DlOpsScheduler.cpp:3376`), an enum because the flag sits beside the node set it selects into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertSide {
    /// `insertBefore == true` — the FIRST child of the common parent in the set.
    Before,
    /// `insertBefore == false` — the LAST.
    After,
}

/// ONE DIM'S CANDIDATE CHUNK EXTENTS, NON-EMPTY — `DscParamCandidatesType`'s inner vector
/// (`L3DlOpsScheduler.h:100`).
///
/// ⭐ THE NON-EMPTINESS IS `DT_CHECK_MSG(!dscCandidates[dscIdx][dim].empty(), "There must be at least
/// one valid candidate.")` — entry 207's closing check, discharged by the type it builds.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidates(Vec<Extent>);

impl Candidates {
    /// The candidates, or [`None`] for an empty list.
    #[must_use]
    pub fn of(extents: Vec<Extent>) -> Option<Self> {
        (!extents.is_empty()).then_some(Self(extents))
    }

    /// The candidates, in ascending order as the search generates them.
    #[must_use]
    pub fn extents(&self) -> &[Extent] {
        &self.0
    }
}

/// ONE DSC'S CANDIDATES, PER DIM — `DscParamCandidatesType`'s inner map.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DimCandidates(BTreeMap<PrimaryDim, Candidates>);

impl DimCandidates {
    /// Every dim's candidates, or [`None`] where any dim reached the end with none.
    #[must_use]
    pub fn of(per_dim: BTreeMap<PrimaryDim, Vec<Extent>>) -> Option<Self> {
        per_dim
            .into_iter()
            .map(|(dim, extents)| Candidates::of(extents).map(|found| (dim, found)))
            .collect::<Option<BTreeMap<_, _>>>()
            .map(Self)
    }

    /// `.at(dim)`, [`None`] for a dim no candidate was generated for.
    #[must_use]
    pub fn get(&self, dim: PrimaryDim) -> Option<&Candidates> {
        self.0.get(&dim)
    }

    /// Every dim and its candidates, in `PrimaryDimTypes` order.
    pub fn iter(&self) -> impl Iterator<Item = (PrimaryDim, &Candidates)> + '_ {
        self.0.iter().map(|(dim, found)| (*dim, found))
    }
}

/// EVERY DSC'S CANDIDATES — `DscParamCandidatesType` (`L3DlOpsScheduler.h:100`), sized by
/// `mySDsc.dscs_.size()`.
#[derive(Debug, Clone, PartialEq)]
pub struct DscCandidates(Vec<DimCandidates>);

impl DscCandidates {
    /// One entry per DSC, in `dscs_` order.
    #[must_use]
    pub const fn new(per_dsc: Vec<DimCandidates>) -> Self {
        Self(per_dsc)
    }

    /// `dscCandidates[dscIdx]`, [`None`] past the end.
    #[must_use]
    pub fn at(&self, dsc: DscIdx) -> Option<&DimCandidates> {
        self.0.get(usize::try_from(dsc.0).ok()?)
    }

    /// Every DSC's candidates, in `dscs_` order.
    pub fn iter(&self) -> impl Iterator<Item = &DimCandidates> + '_ {
        self.0.iter()
    }
}

/// THE TWO DATA STAGES A PAGED DIM IS CHECKED AGAINST — `dataStageOnePageIdx` and
/// `dataStageIbrIdx` (`L3DlOpsScheduler.h:222-223`), whose `-1` default is this type's absence.
///
/// ⭐ NAMED FIELDS AND NOT A PAIR: the two indices are both `DatastageId` and transposing them would
/// check a page size against an index-tensor stick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedStages {
    /// `dataStageOnePageIdx`, assigned at `L3DlOpsScheduler.cpp:6680-6681`.
    pub one_page: DatastageId,
    /// `dataStageIbrIdx`, assigned at `:6630-6631` — the index-tensor stick's stage, read on BOTH its
    /// `ss_` (steady state) and its `el_` (epilogue).
    pub ibr: DatastageId,
}

/// WHAT ENTRY 049 READS OFF ONE DATA STAGE — `DataStructDims` (`dsc/dims.h:158-303`) reduced to the
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

    /// `coreletSplit_.count(dim)` (`dsc/dims.h:206`).
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

    /// `"core"` — the name the stage at [`DATA_STAGE_CORE`] carries
    /// (`ddc/ddl/ddl_conversion.cpp:2976`: `loopnode->numId_ == metadata_.chunk_dstgid ? "chunk" :
    /// "core"`).
    ///
    /// ⭐ IT IS THE INDEX THAT IDENTIFIES THE CORE STAGE, NOT THE NAME: `dataStageCoreIdx` is `0`
    /// (`L3DlOpsScheduler.cpp:275`), so a converter reading `dataStageParam_[0]` builds the name
    /// rather than carrying whatever string its input spelled — which is what keeps a stage name out
    /// of the closed set of things an input can decide.
    #[must_use]
    pub fn core() -> Self {
        Self("core".to_owned())
    }

    /// `"ibr"` (`L3DlOpsScheduler.cpp:6664`).
    #[must_use]
    pub fn ibr() -> Self {
        Self("ibr".to_owned())
    }

    /// `"1page"` (`L3DlOpsScheduler.cpp:6701`).
    #[must_use]
    pub fn one_page() -> Self {
        Self("1page".to_owned())
    }
}

/// ONE HALF OF A DATA STAGE — a `DataStructDims` (`dsc/dims.h:158`) with its `name_`, which is the
/// member the two halves of a `dsc2::DataStage` differ in.
///
/// ⭐ THIS AND [`StageDims`] ARE ONE C++ CLASS SPELLED IN TWO LAYERS, NOT TWO HOMES: `name_` sits
/// here, the other 18 carried members sit on [`StageDims`], and [`FilledDims`] between them is the
/// `!empty()` witness. ⚠️ `ddc::transformation_util::StageDims<D>` is a THIRD spelling — the SAME
/// class kept generic in its extents payload for the ddc side — so it carries `name_` too. The
/// collapse is ~70 sites across twelve files and belongs with the carrier deletion, not here.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedDims {
    /// Field: e006_DataStructDims.name_
    ///
    /// `name_` (`dsc/dims.h:160`).
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

    /// `ss_.primaryDimToVal_st(dim)` ([`StageDims::whole_extent`]), `None` for a dim the stick side
    /// does not state — the spelling of its three readers
    /// (`L3DlOpsScheduler.h:490-492`, `ddc/ddl/ddl_conversion.cpp:3070-3071`).
    #[must_use]
    pub fn ss_extent(&self, dim: PrimaryDim) -> Option<Extent> {
        self.ss.dims.dims().whole_extent(dim)
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
    empty: BTreeMap<DatastageId, EmptyStage>,
}

/// A DATA STAGE THAT STATES NO DIM AT ALL — `dataStageParam_[id]` as `ddl.datastage` leaves it: a
/// bare `operator[]` insert, then a NAME and the core stage's volume ceiling copied onto both halves
/// (`ddc/ddl/ddl_conversion.cpp:2585-2591`).
///
/// ⭐⭐ A SECOND TYPE AND NOT A RELAXED [`FilledDims`]. Eight units of this file discharge *"Core data
/// stage parameters are unavailable."* against a stage that states dims; admitting an extent-less
/// [`DataStage`] would reopen every one of them. This stage HAS no extents, so [`DataStages::at`]
/// answers [`None`] for it — which is what `primaryDimToVal_st` answers for each of its dims — and
/// the two things the reference does write are reachable by name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmptyStage {
    /// `ss_.name_`, which is the stage's index spelled out.
    pub name: StageName,
    /// `ss_.maxSymbolicVolume_` and `el_.maxSymbolicVolume_`, which are one value here because the
    /// reference assigns the core stage's to both.
    pub volumes: BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit>,
}

impl DataStages {
    /// A DSC whose core and chunk stages are both stated.
    #[must_use]
    pub const fn new(core: DataStage, chunk: DataStage) -> Self {
        Self {
            core,
            chunk,
            minted: BTreeMap::new(),
            empty: BTreeMap::new(),
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

    /// `for (auto& [dsIdx, ds] : dsc.dataStageParam_) { ds.ss_.paddingSizes_[dim]; ds.el_.paddingSizes_[dim]; }`
    /// — every stage gains a ZERO padding entry for `dim` where it had none.
    ///
    /// ⭐ ASKING IS A MUTATION: the reference's statement is a bare `operator[]` and its whole effect
    /// is that default insert, which is what a dimension mapping needs the stages to carry.
    ///
    /// ⚠️ [`EmptyStage`] states no `paddingSizes_` to write, so the stages minted by `ddl.datastage`
    /// are not reached — a divergence from a reference map that holds one value type.
    pub fn ensure_padding(&mut self, dim: PrimaryDim) {
        for stage in [&mut self.core, &mut self.chunk]
            .into_iter()
            .chain(self.minted.values_mut())
        {
            stage.ss.dims.padding_mut().entry(dim).or_default();
            stage.el.dims.padding_mut().entry(dim).or_default();
        }
    }

    /// `dataStageParam_[index] = stage`.
    pub fn set(&mut self, index: DatastageId, stage: DataStage) {
        // ⭐ THE BARE ENTRY [`Self::mint_one_page`] LEFT IS REPLACED AND NOT SHADOWED: the reference
        // holds ONE `dataStageParam_`, so filling an index cannot leave an empty stage beside it.
        self.empty.remove(&index);
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

    /// `dataStageIbrIdx` NAMING AN ENTRY THAT EXISTS — the same witness as [`Self::super_chunk`], for
    /// the indirect-buffer-register stage entries 225 and 226 loop against.
    #[must_use]
    pub fn ibr(&self, index: DatastageId) -> Option<IbrStage> {
        self.at(index).map(|_| IbrStage(index))
    }

    /// `auto id = dataStageParam_.size(); while (dataStageParam_.count(id)) id++;` — the index
    /// `ddl.datastage` mints at, which is the first free one AT OR AFTER the current size.
    #[must_use]
    pub fn next_index(&self) -> DatastageId {
        let occupied = 2 + self.minted.len() + self.empty.len();
        let mut id = DatastageId(u32::try_from(occupied).unwrap_or(u32::MAX));
        while self.at(id).is_some() || self.empty.contains_key(&id) {
            id = DatastageId(id.0.saturating_add(1));
        }
        id
    }

    /// `dataStageParam_[id] = <an extent-less stage>` — see [`EmptyStage`].
    pub fn mint_empty(&mut self, index: DatastageId, stage: EmptyStage) {
        self.empty.insert(index, stage);
    }

    /// That stage back, absent for an index that states dims or none at all.
    #[must_use]
    pub fn empty_stage(&self, index: DatastageId) -> Option<&EmptyStage> {
        self.empty.get(&index)
    }

    /// `dataStageParam_.at(index).name()` WHICHEVER KIND OF STAGE THAT INDEX IS — the test entry 325
    /// makes to tell an external stage from a minted one.
    #[must_use]
    pub fn stage_name(&self, index: DatastageId) -> Option<&StageName> {
        self.at(index)
            .map(DataStage::name)
            .or_else(|| self.empty.get(&index).map(|stage| &stage.name))
    }

    /// `dataStageOnePageIdx` NAMING AN ENTRY THAT EXISTS — the one-page stage entry 227 loops against.
    #[must_use]
    pub fn one_page(&self, index: DatastageId) -> Option<OnePageStage> {
        self.at(index).map(|_| OnePageStage(index))
    }

    /// `dataStageParam_.count(index)` — whether this DSC holds a stage under that index AT ALL, the
    /// extent-less ones included, which is the test `getNewDataStageIndex` searches on.
    #[must_use]
    pub fn holds(&self, index: DatastageId) -> bool {
        self.at(index).is_some() || self.empty.contains_key(&index)
    }

    /// `dsc.dataStageParam_[dataStageOnePageIdx];` — the BARE DEFAULT INSERT `getNewDataStageIndex`
    /// ends on (`L3DlOpsScheduler.cpp:6622`), and the witness that insert makes true.
    ///
    /// ⭐⭐ A MINT AND NOT [`Self::one_page`], WHICH CANNOT ANSWER YET: entry 335 needs the witness
    /// in order to WRITE the stage, and `getNewDataStageIndex` is what makes the index name an entry
    /// before it does. The name is default-constructed exactly as the reference's `operator[]` leaves
    /// it; entry 335 writes `"1page"` over it.
    pub fn mint_one_page(&mut self, index: DatastageId) -> OnePageStage {
        self.empty.entry(index).or_default();
        OnePageStage(index)
    }

    /// The same bare insert for `dataStageIbrIdx` (`:6630`), and its witness — entry 334 writes
    /// `"ibr"` over the default name.
    pub fn mint_ibr(&mut self, index: DatastageId) -> IbrStage {
        self.empty.entry(index).or_default();
        IbrStage(index)
    }
}

/// AN IBR DATA-STAGE INDEX THAT NAMES AN EXISTING ENTRY — minted by [`DataStages::ibr`] and
/// [`DataStages::mint_ibr`] and by nothing else, as [`SuperChunkStage`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IbrStage(DatastageId);

impl IbrStage {
    /// `dataStageIbrIdx` (`L3DlOpsScheduler.h:226`).
    #[must_use]
    pub const fn index(self) -> DatastageId {
        self.0
    }
}

/// A ONE-PAGE DATA-STAGE INDEX THAT NAMES AN EXISTING ENTRY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnePageStage(DatastageId);

impl OnePageStage {
    /// `dataStageOnePageIdx` (`L3DlOpsScheduler.h:228`).
    #[must_use]
    pub const fn index(self) -> DatastageId {
        self.0
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

    /// `selectedIndices[dscIdx].at(dim)` — the index itself, for the search's `startIdx + 1` walk.
    #[must_use]
    pub const fn selected_index(&self) -> usize {
        self.selected
    }

    /// The same candidates under a different choice, or `None` where the index is past their end.
    #[must_use]
    pub fn with_index(&self, selected: usize) -> Option<Self> {
        (selected < self.candidates.len()).then(|| Self {
            candidates: self.candidates.clone(),
            selected,
        })
    }
}

/// ONE DSC'S SELECTED CHUNK PARAMETERS — `dscCandidates[dscIdx]` and `selectedIndices[dscIdx]`
/// narrowed to the `primaryDims` the caller asks to be written, which is the whole mechanism for
/// reaching this unit's operands.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DscParamCandidates(pub BTreeMap<PrimaryDim, SelectedCandidate>);

/// EVERY DSC'S SELECTED CHUNK PARAMETERS — `DscParamCandidatesType` AND
/// `DscParamCandidateIndicesType` (`L3DlOpsScheduler.h:100-103`) AS ONE VALUE, one entry per DSC in
/// `dscs_` order.
///
/// ⭐ ONE VALUE AND NOT TWO PARALLEL VECTORS: the searches advance an index and read the list it
/// indexes in the same step, and `DT_CHECK_MSG(idx < dscCandidates[dscIdx].at(dim).size(), "Index is
/// out of range.")` is unspellable once the two travel together.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedDscCandidates(Vec<DscParamCandidates>);

impl SelectedDscCandidates {
    /// One entry per DSC, in `dscs_` order.
    #[must_use]
    pub const fn new(per_dsc: Vec<DscParamCandidates>) -> Self {
        Self(per_dsc)
    }

    /// Every DSC's candidates over `primaryDims` with every index at zero — the
    /// `entry.emplace(dim, 0)` seeding (`L3DlOpsScheduler.cpp:1520`); `None` where a listed dim has
    /// no candidates on some DSC.
    #[must_use]
    pub fn starting(candidates: &DscCandidates, primary_dims: &[PrimaryDim]) -> Option<Self> {
        candidates
            .iter()
            .map(|dims| {
                primary_dims
                    .iter()
                    .map(|dim| {
                        let found = dims.get(*dim)?;
                        Some((*dim, SelectedCandidate::new(found.extents().to_vec(), 0)?))
                    })
                    .collect::<Option<BTreeMap<_, _>>>()
                    .map(DscParamCandidates)
            })
            .collect::<Option<Vec<_>>>()
            .map(Self)
    }

    /// `selectedIndices[dscIdx]`, [`None`] past the end.
    #[must_use]
    pub fn at(&self, dsc: DscIdx) -> Option<&DscParamCandidates> {
        self.0.get(usize::try_from(dsc.0).ok()?)
    }

    /// `selectedIndices[dscIdx].at(dim)`, [`None`] where the DSC or the dim is absent.
    #[must_use]
    pub fn selected_index(&self, dsc: DscIdx, dim: PrimaryDim) -> Option<usize> {
        Some(self.at(dsc)?.0.get(&dim)?.selected_index())
    }

    /// `dscCandidates[dscIdx].at(dim).size()`, [`None`] where the DSC or the dim is absent.
    #[must_use]
    pub fn candidate_count(&self, dsc: DscIdx, dim: PrimaryDim) -> Option<usize> {
        Some(self.at(dsc)?.0.get(&dim)?.candidates().len())
    }

    /// The same selection with ONE DSC's dim moved to `idx` — the core-split arm's per-DSC advance;
    /// [`None`] where the DSC, the dim or the index is out of range.
    #[must_use]
    pub fn with_selection(&self, dsc: DscIdx, dim: PrimaryDim, idx: usize) -> Option<Self> {
        let at = usize::try_from(dsc.0).ok()?;
        let mut moved = self.0.clone();
        let selected = moved.get_mut(at)?.0.get_mut(&dim)?;
        *selected = selected.with_index(idx)?;
        Some(Self(moved))
    }

    /// The same selection with EVERY DSC's dim moved to `idx` — the non-core-split arm's shared
    /// advance; [`None`] where any DSC lacks the dim or the index is past its candidates.
    #[must_use]
    pub fn with_selection_on_every_dsc(&self, dim: PrimaryDim, idx: usize) -> Option<Self> {
        let mut moved = self.0.clone();
        for entry in &mut moved {
            let selected = entry.0.get_mut(&dim)?;
            *selected = selected.with_index(idx)?;
        }
        Some(Self(moved))
    }
}

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

// ────────────────────────────────────────────────────────────────────────────────────────────────
// THE TWO QUESTIONS ASKED OF A LAYOUT ORDER — both over `primaryDsInfo_.at(dsType).layoutDimOrder_`,
// which is a DIFFERENT list from the allocate node's order [`crate::schedule::dsc2::layout_dims`]
// walks to. The two orders stay two.
// ────────────────────────────────────────────────────────────────────────────────────────────────

/// WHERE A DIM SITS IN A LAYOUT ORDER — the `int` index `getDimIndexInLayoutOrder`
/// (`dsc/designSpaceConfig.cpp:429`) returns, which is what subscripts `LabeledDsInfo::scale_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LayoutPos(pub usize);

impl LayoutDims {
    /// Replaces: e004_getDimIndexInLayoutOrder
    ///
    /// THE POSITION OF ONE DIM IN THE LAYOUT ORDER — a linear scan of
    /// `primaryDsInfo_.at(dstype).layoutDimOrder_` for `dim`, first match winning.
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S `-1`, A REAL ANSWER AND NOT AN ABORT — and the evidence is the
    /// `int scale = dimIdx < 0 ? 1 : lds.scale_.at(dimIdx);` idiom, at SIXTEEN sites
    /// (`ddc/ddc_fold.cpp:1391`, `:2298`, `:2536`, `:3001`, `:3161`, `:3366`, `:3965`, `:3981`,
    /// `:4280`, `:4290`, `ddc/ddcv1.cpp:2451`, `:2632`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6033`, `:6203`, `:7400`, `:7542`), plus the
    /// `dimIdx < 0 ||` guard at `ddc/ddc_transformation.cpp:1793` and the `>= 0` test at
    /// `ddc/ddl/ddl_conversion.cpp:1305`: there an ABSENT dim reads as an ORDINARY dim of scale `1`.
    ///
    /// ⛔⛔ BUT EIGHT OTHER SUBSCRIPTS TAKE THE ANSWER UNGUARDED, so THERE the `-1` THROWS out of
    /// `scale_.at()` and [`None`] IS A STOP AND NOT A `1`: `dsc/dsc2.cpp:3559` and `:3821` — ENTRY
    /// 017'S OWN READ, one line below its call at `:3820` — `ddc/ddcv1.cpp:500`, `:1502`, `:1528`,
    /// `:1904`, and `ddc/ddc_transformation.cpp:929` with `:931`.
    /// `isLabeledDsDimensionBroadcast` refuses it outright instead, with a
    /// `DT_CHECK_MSG(scaleIdx >= 0 && scaleIdx < lds.scale_.size())`
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:69-70`, and the same check again at `:4272`).
    /// ⛔ SO THE `1` IS NOT THIS FUNCTION'S ANSWER AND IT IS NOT UNIFORM: a port of a SUBSCRIPTING
    /// site must NOT default an absent dim to `1`.
    ///
    /// ⛔ AND A THIRD CLASS, NEITHER GUARDED NOR THROWING: at `ddc/ddcv1.cpp:1588-1591` the answer is
    /// an ARITHMETIC OFFSET, `sizeIdx += tensorSizes.size()`, into `unitTimeTransferChunkSize_` and
    /// not a subscript of `scale_` at all — a `-1` there silently becomes `tensorSizes.size() - 1`, a
    /// valid index onto the wrong dim. A port of THAT site must treat [`None`] as neither a `1` nor a
    /// stop, but as its own case.
    #[must_use]
    pub fn index_of(&self, dim: PrimaryDim) -> Option<LayoutPos> {
        self.iter().position(|named| named == dim).map(LayoutPos)
    }

    /// Replaces: e005_getLayoutDimSet
    ///
    /// THE LAYOUT ORDER AS A SET — `std::set<PrimaryDimTypes>(ldov.begin(), ldov.end())`, for the
    /// readers that ask `count(dim)` and never a position.
    ///
    /// ⭐ THE `int ldsIdx` OVERLOAD (`dsc/designSpaceConfig.h:238-240`) IS THIS SAME CALL through
    /// `labeledDs_.at(ldsIdx).dsType_`, so [`DesignSpaceConfig::primary_ds_info`] keyed by
    /// [`LabeledDs::ds_type`] spells it and there is no second method.
    #[must_use]
    pub fn to_set(&self) -> BTreeSet<PrimaryDim> {
        self.iter().collect()
    }
}

#[cfg(test)]
mod tests_e002_e005 {
    //! ⭐⭐ THE FOUR DIMS-AND-LAYOUT UNITS AGAINST THE REFERENCE'S OWN EXPORT.
    //!
    //! # WHERE THE NUMBERS COME FROM
    //!
    //! `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_<N>/sdsc.json` — the SCHEDULED output of the
    //! reference's own run over `g0/sdsc_<N>.json`, i.e. what its L3/ddc/dcg wrote. Two programs:
    //!
    //!   * `sdsc_15`, DSC `t729_fq_mm` — `dataStageParam_["0"].ss_` (`name_: "core"`) states
    //!     `in_: 2048`, `out_: 64`, `mb_: 1` and `-1` for the other seven, and `["7"].ss_`
    //!     (`name_: "7"`) states `in_: 16` for the SAME dim. Three DISTINCT values, so a test that
    //!     transposed two of them would go red.
    //!   * `sdsc_2`, DSC `rmeps_o728` — `primaryDsInfo_["OUTPUT"].layoutDimOrder_` is
    //!     `["mb", "out", "y"]` and `labeledDs_[1].scale_` is `[-1, -2, 1]`, three DISTINCT scales.
    //!     Each names a different `getBufferCapacityForNodePerDim` arm (`dsc/dsc2.cpp:3824-3830`), so
    //!     an off-by-one index answers a different arm rather than a different number.
    //!
    //! ⛔ WHAT THE CORPUS CANNOT SHOW, MEASURED: `symbolicDimInfo_` is EMPTY on all 2,232 stage
    //! halves of the 187 programs, so `scaleFromMaxToGranularity`'s divide is not reached by any g0
    //! program and its expected values below are computed from `dsc/dims.cpp:618-628` by hand and
    //! labelled as such. Only its PASSTHROUGH is corpus-grounded.

    use super::{
        FilledDims, Granularity, LabeledDs, LayoutPos, MaxSize, Pinning, StageDims, Symbolic,
        SymbolicDimInfo,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
    use crate::schedule::ddc::transformation::{DsType, Scale};
    use crate::schedule::dsc2::{LayoutDims, LdsIdx};
    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU32;

    /// `sdsc_2`'s `primaryDsInfo_["OUTPUT"].layoutDimOrder_`, VERBATIM.
    fn rmeps_layout() -> LayoutDims {
        LayoutDims::new(PrimaryDim::Mb, vec![PrimaryDim::Out, PrimaryDim::Y])
    }

    /// `sdsc_2`'s `labeledDs_[1].scale_`, VERBATIM and positional against [`rmeps_layout`].
    const RMEPS_SCALE: [f64; 3] = [-1.0, -2.0, 1.0];

    fn granularity(step: u32) -> Granularity {
        Granularity::new(NonZeroU32::new(step).expect("a positive step"))
    }

    fn symbolic(max: u32, step: u32) -> Symbolic {
        Symbolic::new(
            BTreeMap::from([(
                PrimaryDim::In,
                SymbolicDimInfo {
                    max_size: MaxSize(max),
                    granularity: granularity(step),
                },
            )]),
            BTreeMap::new(),
        )
    }

    /// e002 — the extents `sdsc_15`'s `core` stage states, written through the slot and read back,
    /// with the reference's `-1` for `y_` reading as an ABSENCE and its `["7"]` value OVERWRITING the
    /// `["0"]` one through the same slot.
    #[test]
    fn one_stages_extents_are_the_reference_s() {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::In, Extent(2048));
        let mut filled = FilledDims::of(dims).expect("a stage that states a dim");
        filled.set_extent(PrimaryDim::Out, Extent(64));
        filled.set_extent(PrimaryDim::Mb, Extent(1));

        assert_eq!(filled.dims().extent(PrimaryDim::In), Some(Extent(2048)));
        assert_eq!(filled.dims().extent(PrimaryDim::Out), Some(Extent(64)));
        assert_eq!(filled.dims().extent(PrimaryDim::Mb), Some(Extent(1)));
        // `y_: -1` in the export — an unwritten field, which is an absence and not a zero.
        assert_eq!(filled.dims().extent(PrimaryDim::Y), None);
        assert_eq!(filled.dims().extent(PrimaryDim::Kij), None);

        // `dataStageParam_["7"].ss_.in_` is 16: the handler hands back the SAME `double&`, so the
        // second assignment through it replaces the first rather than adding a dim.
        filled.set_extent(PrimaryDim::In, Extent(16));
        assert_eq!(filled.dims().extent(PrimaryDim::In), Some(Extent(16)));
        assert_eq!(filled.dims().extents.len(), 3);
    }

    /// e003 — the passthrough every g0 stage takes, then the divide and both `DT_CHECK`s computed
    /// from `dsc/dims.cpp:618-628`.
    #[test]
    fn a_max_unit_size_re_expressed_in_granularity_units() {
        // MEASURED: `symbolicDimInfo_` is `{}` on all 2,232 stage halves, so THIS is the arm every
        // g0 program takes, and the value is `sdsc_15`'s own `in_`.
        let plain = Symbolic::new(BTreeMap::new(), BTreeMap::new());
        assert_eq!(
            plain.scale_from_max_to_granularity(PrimaryDim::In, Extent(2048)),
            Some(Extent(2048))
        );

        // CONSTRUCTED, not corpus-measured: `maxSize_ / granularity_` = 2048 / 64 = 32, and
        // 2048 / 32 = 64.
        assert_eq!(
            symbolic(2048, 64).scale_from_max_to_granularity(PrimaryDim::In, Extent(2048)),
            Some(Extent(64))
        );
        // A dim the map does not name is UNTOUCHED even when another dim is symbolic.
        assert_eq!(
            symbolic(2048, 64).scale_from_max_to_granularity(PrimaryDim::Out, Extent(2048)),
            Some(Extent(2048))
        );
        // `DT_CHECK(maxSize_ % granularity_ == 0)` (`:623`).
        assert_eq!(
            symbolic(2048, 3).scale_from_max_to_granularity(PrimaryDim::In, Extent(2048)),
            None
        );
        // `DT_CHECK(factor != 0 ...)` (`:625`) — a zero `maxSize_`, which a positive granularity
        // does NOT discharge.
        assert_eq!(
            symbolic(0, 64).scale_from_max_to_granularity(PrimaryDim::In, Extent(2048)),
            None
        );
        // `DT_CHECK(... && val % factor == 0)` (`:625`).
        assert_eq!(
            symbolic(2048, 64).scale_from_max_to_granularity(PrimaryDim::In, Extent(33)),
            None
        );
    }

    /// e004 — every position `sdsc_2`'s layout order gives, checked by the `scale_` each one selects.
    #[test]
    fn a_dim_s_position_selects_the_scale_the_reference_wrote() {
        let layout = rmeps_layout();
        assert_eq!(layout.index_of(PrimaryDim::Mb), Some(LayoutPos(0)));
        assert_eq!(layout.index_of(PrimaryDim::Out), Some(LayoutPos(1)));
        assert_eq!(layout.index_of(PrimaryDim::Y), Some(LayoutPos(2)));
        // The reference's `-1`: a REAL answer, which `isLabeledDsDimensionBroadcast` tests with
        // `>= 0` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:68`).
        assert_eq!(layout.index_of(PrimaryDim::In), None);
        assert_eq!(layout.index_of(PrimaryDim::Kij), None);

        // `scale_[getDimIndexInLayoutOrder(dsType_, dim)]`, which is the read every capacity arm
        // makes of it (`dsc/dsc2.cpp:3820-3830`).
        let scale_at = |dim| RMEPS_SCALE[layout.index_of(dim).expect("a dim in the order").0];
        assert_eq!(scale_at(PrimaryDim::Mb), -1.0);
        assert_eq!(scale_at(PrimaryDim::Out), -2.0);
        assert_eq!(scale_at(PrimaryDim::Y), 1.0);

        // The same three scales as scratchy states them, reached through the zip that replaced the
        // subscript — `Scale::UnitStick` is the `-1` and `Scale::StickDim` the `-2`.
        let lds = LabeledDs::new(
            DsType::Output,
            vec![
                (PrimaryDim::Mb, Scale::UnitStick),
                (PrimaryDim::Out, Scale::StickDim),
                (PrimaryDim::Y, Scale::Sized(1.0)),
            ],
            LdsIdx(1),
            Pinning::default(),
        );
        assert_eq!(lds.scale(PrimaryDim::Out), Some(Scale::StickDim));
        assert_eq!(lds.scale(PrimaryDim::In), None);
    }

    /// e005 — `sdsc_2`'s layout order as the set its `count(dim)` readers ask.
    #[test]
    fn the_layout_order_as_a_set_is_the_dims_the_reference_named() {
        let set = rmeps_layout().to_set();
        assert_eq!(
            set,
            BTreeSet::from([PrimaryDim::Mb, PrimaryDim::Out, PrimaryDim::Y])
        );
        assert!(set.contains(&PrimaryDim::Y));
        // `count(IN) == 0` — the question `FreshAllocation::of` and the broadcast readers ask.
        assert!(!set.contains(&PrimaryDim::In));
        // `sdsc_15`'s KERNEL order `["in", "out"]` is a DIFFERENT set, and the two-dim order's set
        // has two members and not three.
        let kernel = LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Out]).to_set();
        assert_eq!(kernel, BTreeSet::from([PrimaryDim::In, PrimaryDim::Out]));
        assert_ne!(kernel, set);
    }
}

/// A STAGE'S `maxSymbolicVolume_` AS IT STANDS BEFORE THE PRUNE — volume limits that MAY be keyed on
/// dims the stage's own `symbolicDimInfo_` does not name, which is the one state a well-formed
/// [`Symbolic`] cannot hold and is exactly the input `pruneMaxSymbolicVolumes` exists to consume.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StatedVolumes(BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit>);

impl StatedVolumes {
    /// The limits as a stage states them, before any key is read against a `symbolicDimInfo_`.
    #[must_use]
    pub const fn new(volumes: BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit>) -> Self {
        Self(volumes)
    }

    /// Replaces: e008_pruneMaxSymbolicVolumes
    ///
    /// RE-KEYS EACH LIMIT ONTO THE DIMS `mine` STILL CALLS SYMBOLIC — divided by the granularity
    /// `reference` gives every dim that was lost, capped at the product of the survivors' `maxSize_`,
    /// and DROPPED where no survivor is left. A key `mine` names in full passes through untouched.
    ///
    /// ⭐ Every write the reference makes is `min`-guarded and no erased key is ever a write target,
    /// so rebuilding the map with a `min`-insert is its in-place erase-and-insert walk exactly.
    ///
    /// ⛔ BOTH `DT_CHECK`s THROW (`dsc/dims.cpp:746`, `:748`) AND SO DO THESE: keeping an entry the
    /// prune could not reduce would state a volume limit no stage asked for.
    #[must_use]
    pub fn pruned_against(
        self,
        mine: &Symbolic,
        reference: &Symbolic,
    ) -> BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit> {
        let mut pruned: BTreeMap<BTreeSet<PrimaryDim>, VolumeLimit> = BTreeMap::new();
        let mut keep_min = |dims: BTreeSet<PrimaryDim>, limit: VolumeLimit| {
            let entry = pruned.entry(dims).or_insert(limit);
            *entry = (*entry).min(limit);
        };
        for (sym_dims, limit) in self.0 {
            if sym_dims.iter().all(|dim| mine.info.contains_key(dim)) {
                keep_min(sym_dims, limit);
                continue;
            }
            let mut survivors = BTreeSet::new();
            let mut my_limit = limit;
            let mut mul_of_maxes = VolumeLimit::ONE;
            for &sym_dim in &sym_dims {
                if let Some(info) = mine.info.get(&sym_dim) {
                    survivors.insert(sym_dim);
                    mul_of_maxes = mul_of_maxes.times(info.max_size);
                    continue;
                }
                let Some(lost) = reference.info.get(&sym_dim) else {
                    panic!(
                        "StatedVolumes::pruned_against: \
                         DT_CHECK(refDstg.symbolicDimInfo_.count({sym_dim:?})) throws for a lost dim \
                         the reference stage does not call symbolic"
                    )
                };
                match my_limit.divided_exactly_by(lost.granularity) {
                    Some(reduced) => my_limit = reduced,
                    None => panic!(
                        "StatedVolumes::pruned_against: \
                         DT_CHECK(myVolumeLimit % dimGranularity == 0) throws for {my_limit:?} over \
                         {sym_dim:?}'s {step:?}",
                        step = lost.granularity
                    ),
                }
            }
            if !survivors.is_empty() {
                keep_min(survivors, my_limit.min(mul_of_maxes));
            }
        }
        pruned
    }
}

#[cfg(test)]
mod tests_e008 {
    //! ⭐ THE PRUNER AGAINST THE REFERENCE'S OWN WORKED EXAMPLE (`dsc/dims.cpp:719-728`).
    //!
    //! ⛔ WHAT THE CORPUS CANNOT SHOW, MEASURED: `maxSymbolicVolume_` is EMPTY on every data stage of
    //! all 187 g0 programs, so no g0 program reaches this walk. The numbers are the reference's own
    //! comment — `a,b,c -> 2048`, `a` lost at `gr=4 max=64`, `b` and `c` symbolic at `gr=16 max=64`,
    //! and `min(64*64, 2048/4) = 512`.

    use super::{Granularity, MaxSize, StatedVolumes, Symbolic, SymbolicDimInfo, VolumeLimit};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU32;

    /// `a`, `b` and `c` of the reference's comment, as three members of the closed dim set.
    const A: PrimaryDim = PrimaryDim::Mb;
    const B: PrimaryDim = PrimaryDim::Out;
    const C: PrimaryDim = PrimaryDim::Y;

    fn info(max: u32, step: u32) -> SymbolicDimInfo {
        SymbolicDimInfo {
            max_size: MaxSize(max),
            granularity: Granularity::new(NonZeroU32::new(step).expect("a positive step")),
        }
    }

    /// The reference stage of the comment: all three dims still symbolic.
    fn core() -> Symbolic {
        Symbolic::new(
            BTreeMap::from([(A, info(64, 4)), (B, info(64, 16)), (C, info(64, 16))]),
            BTreeMap::from([(BTreeSet::from([A, B, C]), VolumeLimit(2048))]),
        )
    }

    /// e008 — the reference's own worked example, then its `mulOfMaxes` cap winning instead.
    #[test]
    fn a_lost_dim_divides_the_limit_and_re_keys_it_onto_the_survivors() {
        let chunk = Symbolic::new(
            BTreeMap::from([(B, info(64, 16)), (C, info(64, 16))]),
            BTreeMap::new(),
        );
        let stated = StatedVolumes::new(BTreeMap::from([(
            BTreeSet::from([A, B, C]),
            VolumeLimit(2048),
        )]));
        // `min(max(b) * max(c), 2048 / gr(a)) = min(4096, 512) = 512`, keyed on `{b, c}` and not on
        // `{a, b, c}` — the erase and the insert are one step.
        assert_eq!(
            stated.pruned_against(&chunk, &core()),
            BTreeMap::from([(BTreeSet::from([B, C]), VolumeLimit(512))])
        );

        // `mulOfMaxes` the other way round: with `b` the only survivor, 2048 / gr(a) = 512 is capped
        // at max(b) = 64.
        let one = Symbolic::new(BTreeMap::from([(B, info(64, 16))]), BTreeMap::new());
        let stated = StatedVolumes::new(BTreeMap::from([(
            BTreeSet::from([A, B]),
            VolumeLimit(2048),
        )]));
        assert_eq!(
            stated.pruned_against(&one, &core()),
            BTreeMap::from([(BTreeSet::from([B]), VolumeLimit(64))])
        );
    }

    /// e008 — THE NEGATIVE THAT PINS THE FUSION: the two call shapes give DIFFERENT maps, so the one
    /// that adopts the reference's limits cannot stand in for the one that keeps this stage's.
    #[test]
    fn the_unfused_prune_keeps_this_stage_s_own_limits() {
        let mine = || {
            Symbolic::new(
                BTreeMap::from([(B, info(64, 16)), (C, info(64, 16))]),
                BTreeMap::from([(BTreeSet::from([B, C]), VolumeLimit(256))]),
            )
        };

        // `ddc/ddcv1.cpp:1424` — every dim this stage's own limit names is still symbolic here, so
        // `needPruning` is false and the 256 stands.
        let mut unfused = mine();
        unfused.prune_volumes(&core());
        assert_eq!(
            *unfused.volumes(),
            BTreeMap::from([(BTreeSet::from([B, C]), VolumeLimit(256))])
        );

        // `L3DlOpsScheduler.cpp:171-172` — the assignment first, so the core's 2048 becomes 512 and
        // this stage's own 256 is gone.
        let mut fused = mine();
        fused.prune_volumes_from(&core());
        assert_eq!(
            *fused.volumes(),
            BTreeMap::from([(BTreeSet::from([B, C]), VolumeLimit(512))])
        );
    }

    /// e008 — ⛔ THE UNFUSED WALK IS THE IDENTITY FOR EVERY [`Symbolic`] THAT EXISTS, so
    /// `stages/ddc_sites.rs:663` prunes nothing. `needPruning` (`dsc/dims.cpp:732-734`) needs a key
    /// naming a dim `info` does NOT; [`Symbolic::new`] drops exactly those, and `add_dim` only ever
    /// WIDENS `info`. The reference arms the walk through `makeDimNotSymbolic`, which erases from
    /// `symbolicDimInfo_` and leaves `maxSymbolicVolume_` alone (`dsc/dims.cpp:781-787`) — a state
    /// this type has no method to reach.
    #[test]
    fn the_unfused_prune_has_no_reachable_input_it_reduces() {
        // (1) the constructor refuses to hold the very limit the walk exists to re-key.
        let dropped = Symbolic::new(
            BTreeMap::from([(B, info(64, 16))]),
            BTreeMap::from([(BTreeSet::from([A, B]), VolumeLimit(2048))]),
        );
        assert!(dropped.volumes().is_empty());

        // (2) the one mutator WIDENS `info`, so a surviving key stays fully named and the walk
        // `ddc/ddcv1.cpp:1424` asks for hands back the map it was given.
        let mut stage = Symbolic::new(
            BTreeMap::from([(B, info(64, 16)), (C, info(64, 16))]),
            BTreeMap::from([(BTreeSet::from([B, C]), VolumeLimit(256))]),
        );
        stage.add_dim(A, info(64, 4));
        let before = stage.volumes().clone();
        stage.prune_volumes(&core());
        assert_eq!(*stage.volumes(), before);
        assert_eq!(
            before,
            BTreeMap::from([(BTreeSet::from([B, C]), VolumeLimit(256))])
        );
    }
}

#[cfg(test)]
mod tests_e009_e010 {
    //! ⭐⭐ THE REFERENCE PUBLISHES THIS PAIR'S ANSWER. `paddingSizes_[dim].totalSize_` in a
    //! reference export IS `primaryDimToVal_base_st(dim, {dim, PADDED_FULLSPAN_WUNNEEDED}, 1.0,
    //! false)` (`dsc/dims.cpp:298-299`), so every number below is the reference's own, read from
    //! `g0/debug/sdsc_14/sdsc.json` — the ONLY one of the 187 g0 programs that states any
    //! `paddingSizes_` at all (18 entries, 4 of them `"N/A"`).
    //!
    //! ⛔ WHAT THE CORPUS CANNOT SHOW, MEASURED OVER ALL 187: no entry anywhere states a non-zero
    //! `padFront_`/`padBack_`/`unneededPad_`/`stride_`, and none states a `windowDim_`. The window
    //! and unneeded arms below are therefore CONSTRUCTED from the reference's own formulae, and are
    //! marked as such.

    use super::{DimPadding, PadElems, PadSizes, StageDims, UnneededPad};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
    use crate::schedule::ddc::fold::{PadType, Stride};
    use crate::schedule::ddc::transformation_util::PaddingForm;
    use std::collections::BTreeMap;

    /// `sdsc_14`'s `dataStageParam_["0"].ss_` — `name_: "core"`, `out_: 128`, `mb_: 1`, `y_: 1`, with
    /// an all-zero `paddingSizes_` entry on `out` and on `mb`.
    fn core_stage() -> StageDims {
        StageDims {
            extents: BTreeMap::from([
                (PrimaryDim::Out, Extent(128)),
                (PrimaryDim::Mb, Extent(1)),
                (PrimaryDim::Y, Extent(1)),
            ]),
            padding: BTreeMap::from([
                (PrimaryDim::Out, DimPadding::default()),
                (PrimaryDim::Mb, DimPadding::default()),
            ]),
            ..StageDims::default()
        }
    }

    /// `getPadding(dim) == PADDED_FULLSPAN_WUNNEEDED` on one dim, which is the form the export's
    /// `totalSize_` is written through.
    fn full_span(dim: PrimaryDim) -> PaddingForm {
        let mut form = PaddingForm::default();
        form.set_padding(dim, PadType::PaddedFullSpanWUnneeded);
        form
    }

    /// e009 + e010 — the four `totalSize_` values `sdsc_14` exports, and the plain read 186 of the
    /// 187 g0 programs take instead.
    #[test]
    fn the_reference_s_own_total_size_export_is_what_this_pair_computes() {
        let core = core_stage();

        // `dataStageParam_["0"].ss_.paddingSizes_`: `out` -> `totalSize_: 128`, `mb` -> `1`. All six
        // pad counts are zero, so the full-span arm adds nothing to the stated extent.
        assert_eq!(
            core.padded_extent(PrimaryDim::Out, PadType::PaddedFullSpanWUnneeded),
            Some(Extent(128))
        );
        assert_eq!(
            core.padded_extent(PrimaryDim::Mb, PadType::PaddedFullSpanWUnneeded),
            Some(Extent(1))
        );

        // `N_.paddingSizes_`: the same two dims of the same program at `out_: 2048`, `totalSize_:
        // 2048`, which pins the arm on the extent and not on the stage.
        let mut whole = core.clone();
        whole.extents.insert(PrimaryDim::Out, Extent(2048));
        assert_eq!(
            whole.padded_extent(PrimaryDim::Out, PadType::PaddedFullSpanWUnneeded),
            Some(Extent(2048))
        );

        // e010 ALONE — `getPadding(d) == NOPAD`, the arm 186 of 187 g0 programs take, which returns
        // the stated slot before any `paddingSizes_` lookup.
        assert_eq!(
            core.scaled_extent(PrimaryDim::Out, &PaddingForm::default(), None, false),
            Some(Extent(128))
        );
        // `in_: -1` in the same export — the unstated slot, which `calculate_padded` short-circuits
        // to `-1` (`dsc/dims.cpp:567-568`).
        assert_eq!(
            core.scaled_extent(PrimaryDim::In, &PaddingForm::default(), None, false),
            None
        );

        // e009 ALONE, WITH A VAL IT DID NOT READ — the `ddc/ddc_fold.cpp:367` call shape, whose
        // `innerCard` is a fold cardinality and not this stage's extent.
        assert_eq!(
            core.calculate_padded(
                PrimaryDim::Out,
                Extent(64),
                &full_span(PrimaryDim::Out),
                false
            ),
            Some(Extent(64))
        );
    }

    /// e009 — the abort set, led by the one the reference EXPORTS as `"N/A"`.
    #[test]
    fn a_voided_edge_a_missing_entry_and_a_wrong_pad_type_are_each_the_refusal() {
        // `dataStageParam_["2"].ss_.paddingSizes_["out"]`: `padFront_: -1, padBack_: -1` and
        // `totalSize_: "N/A"` — the export's own guard (`dsc/dims.cpp:295-296`) standing in for
        // *"Padded access is not valid in datastage"* (`:581-582`). 4 of the corpus's 18 entries.
        let mut voided = core_stage();
        voided.extents.insert(PrimaryDim::Out, Extent(64));
        voided.padding.insert(
            PrimaryDim::Out,
            DimPadding {
                sizes: PadSizes::Voided,
                ..DimPadding::default()
            },
        );
        assert_eq!(
            voided.padded_extent(PrimaryDim::Out, PadType::PaddedFullSpanWUnneeded),
            None
        );
        // ⭐ AND THE SAME STAGE STILL ANSWERS ITS PLAIN READ: `out_: 64` is what that export states,
        // and NOPAD returns before the voided edge is ever reached.
        assert_eq!(
            voided.scaled_extent(PrimaryDim::Out, &PaddingForm::default(), None, false),
            Some(Extent(64))
        );

        // *"Padded dimension without padding sizes information in datastage"* (`:575-577`) — the dim
        // is stated and positive, and `paddingSizes_` does not name it.
        let core = core_stage();
        assert_eq!(
            core.padded_extent(PrimaryDim::Y, PadType::PaddedFullSpanWUnneeded),
            None
        );

        // *"Cannot calculate padded version of compound dim"* (`:572-573`), which needs a POSITIVE
        // val to be reached at all — a compound dim left at `-1` takes the short-circuit instead.
        let mut compound = core.clone();
        compound.extents.insert(PrimaryDim::Ij, Extent(16));
        compound
            .padding
            .insert(PrimaryDim::Ij, DimPadding::default());
        assert_eq!(
            compound.padded_extent(PrimaryDim::Ij, PadType::PaddedFullSpanWUnneeded),
            None
        );

        // *"Unsupported padding type requested for padded non-window operation"* (`:589-592`): a
        // window pad type on a dim with no `windowDim_`.
        assert_eq!(
            core.padded_extent(PrimaryDim::Out, PadType::PaddedWZeroPad),
            None
        );
    }

    /// e009 — CONSTRUCTED, no g0 program states a `windowDim_`: the window span and the unneeded
    /// counts, from `dsc/dims.cpp:593-613`.
    #[test]
    fn the_window_span_is_the_kernel_plus_the_strided_tail() {
        let mut windowed = core_stage();
        // A 3-wide kernel on `ki`, stride 2, over 5 outputs: `wSize + (val - 1) * stride_`.
        windowed.extents.insert(PrimaryDim::Ki, Extent(3));
        windowed.extents.insert(PrimaryDim::Out, Extent(5));
        windowed.padding.insert(
            PrimaryDim::Out,
            DimPadding {
                window_dim: Some(PrimaryDim::Ki),
                stride: Stride::new(2).expect("a moving stride"),
                unneeded: UnneededPad {
                    total: PadElems(4),
                    front: PadElems(1),
                    back: PadElems(2),
                },
                sizes: PadSizes::of(PadElems(1), PadElems(1)),
                ..DimPadding::default()
            },
        );

        // `PADDED_WZEROPAD`: 3 + (5 - 1) * 2 = 11.
        assert_eq!(
            windowed.padded_extent(PrimaryDim::Out, PadType::PaddedWZeroPad),
            Some(Extent(11))
        );
        // `PADDED_FULLSPAN_WUNNEEDED` adds `unneededPad_` alone: 11 + 4 = 15.
        assert_eq!(
            windowed.padded_extent(PrimaryDim::Out, PadType::PaddedFullSpanWUnneeded),
            Some(Extent(15))
        );
        // `PADDED_NOZEROPAD` subtracts the two unneeded edges and both pads: 15 - 1 - 2 - 1 - 1 = 10.
        assert_eq!(
            windowed.padded_extent(PrimaryDim::Out, PadType::PaddedNoZeroPad),
            Some(Extent(10))
        );
        // `LOWERED_PADDED` is `wSize * val` and ignores the stride entirely: 3 * 5 = 15.
        assert_eq!(
            windowed.padded_extent(PrimaryDim::Out, PadType::LoweredPadded),
            Some(Extent(15))
        );
        // *"Missing window size"* (`:596-597`) — `ki` unstated makes the recursive read `-1`.
        windowed.extents.remove(&PrimaryDim::Ki);
        assert_eq!(
            windowed.padded_extent(PrimaryDim::Out, PadType::PaddedWZeroPad),
            None
        );
    }

    /// e009 — CONSTRUCTED, no g0 entry states a non-zero pad count: the NON-WINDOW span
    /// `val + padFront_ + padBack_ (+ unneededPad_)` (`dsc/dims.cpp:584-588`), and its two-arm
    /// `else` (`:589-592`).
    ///
    /// ⛔ WITHOUT THIS THE FORMULA IS UNPINNED. `sdsc_14`'s 18 `paddingSizes_` entries are all-zero
    /// on every count, so the reference's own `totalSize_` export equals the bare extent, and
    /// reducing this arm to `val` leaves every other assertion in this module green.
    #[test]
    fn the_non_window_span_is_the_extent_plus_both_edges_and_then_the_unneeded_pad() {
        let mut padded = core_stage();
        padded.extents.insert(PrimaryDim::Out, Extent(100));
        padded.padding.insert(
            PrimaryDim::Out,
            DimPadding {
                sizes: PadSizes::of(PadElems(3), PadElems(5)),
                unneeded: UnneededPad {
                    total: PadElems(7),
                    front: PadElems(2),
                    back: PadElems(5),
                },
                ..DimPadding::default()
            },
        );

        // `PADDED_FULLSPAN` is the two edges alone (`:587-588`): 100 + 3 + 5 = 108.
        assert_eq!(
            padded.padded_extent(PrimaryDim::Out, PadType::PaddedFullSpan),
            Some(Extent(108))
        );
        // `PADDED_FULLSPAN_WUNNEEDED` adds `unneededPad_` and NOT its two halves (`:584-586`):
        // 108 + 7 = 115, not 108 + 2 + 5.
        assert_eq!(
            padded.padded_extent(PrimaryDim::Out, PadType::PaddedFullSpanWUnneeded),
            Some(Extent(115))
        );
        // *"Unsupported padding type requested for padded non-window operation"* (`:589-592`) — the
        // three window spellings have no non-window formula at all.
        for pad in [
            PadType::PaddedWZeroPad,
            PadType::PaddedNoZeroPad,
            PadType::LoweredPadded,
        ] {
            assert_eq!(padded.padded_extent(PrimaryDim::Out, pad), None);
        }

        // ⭐ AND `stride_` IS NOT READ HERE (`:584-588` against `:599-609`): a stride of 4 on the
        // same dim leaves both spans exactly where they were.
        let edges = padded.padding[&PrimaryDim::Out];
        padded.padding.insert(
            PrimaryDim::Out,
            DimPadding {
                stride: Stride::new(4).expect("a moving stride"),
                ..edges
            },
        );
        assert_eq!(
            padded.padded_extent(PrimaryDim::Out, PadType::PaddedFullSpan),
            Some(Extent(108))
        );
    }
}

#[cfg(test)]
mod tests_e011 {
    //! ⭐⭐ THE REFERENCE PUBLISHES BOTH SIDES OF THIS FUNCTION. MEASURED over all 187 g0 reference
    //! exports: exactly 8 of their 4,850 dim blocks state a non-empty `coreletSplit_` and all 8 are
    //! `g0/debug/sdsc_49/sdsc.json`, the ONLY program with `numCoreletsUsed_ > 1` — and each states
    //! the split BESIDE the whole-core slot, so the corelet view's answer and the base view's answer
    //! are both the reference's own numbers rather than ours.
    //!
    //! ⛔ WHAT THE CORPUS CANNOT SHOW, SAME MEASUREMENT: 0 of those 4,850 blocks states any
    //! `symbolicDimInfo_`, so the granularity arm and the density divide are CONSTRUCTED from the
    //! reference's own formulae and are marked as such.

    use super::{Granularity, MaxSize, StageDims, Symbolic, SymbolicDimInfo};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
    use crate::schedule::ddc::fold::{Cardinality, ScaleBlock};
    use crate::schedule::ddc::transformation_util::PaddingForm;
    use crate::units::Corelet;
    use std::collections::BTreeMap;
    use std::num::NonZeroU32;

    /// `sdsc_49`'s `dataStageParam_["0"].ss_` — `name_: "core"`, `out_: 512`, `mb_: 2`, `y_: 1`,
    /// `in_: -1`, `coreletSplit_: {"out": [256, 256]}` and an EMPTY `paddingSizes_`.
    fn core_stage() -> StageDims {
        StageDims {
            extents: BTreeMap::from([
                (PrimaryDim::Out, Extent(512)),
                (PrimaryDim::Mb, Extent(2)),
                (PrimaryDim::Y, Extent(1)),
            ]),
            corelet_split: BTreeMap::from([(PrimaryDim::Out, vec![Extent(256), Extent(256)])]),
            ..StageDims::default()
        }
    }

    /// e011 — the corelet view answers the SHARE and the base view the WHOLE, on the one g0 program
    /// that splits a dim across corelets.
    #[test]
    fn the_reference_s_own_corelet_split_export_is_what_the_corelet_view_answers() {
        let core = core_stage();
        let plain = PaddingForm::default();

        // `coreletSplit_["out"] = [256, 256]` — each corelet's own share.
        assert_eq!(
            core.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<0>()),
                &plain,
                None,
                false
            ),
            Some(Extent(256))
        );
        assert_eq!(
            core.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<1>()),
                &plain,
                None,
                false
            ),
            Some(Extent(256))
        );
        // `clId = -1` on the SAME dim is `out_: 512`, the whole core — this is the pair the reference
        // exports together, and reading either for the other is the defect this asserts against.
        assert_eq!(
            core.corelet_extent(PrimaryDim::Out, None, &plain, None, false),
            Some(Extent(512))
        );

        // ⭐ A CORELET ID WITH NO SPLIT ON THE DIM IS STILL THE BASE READ: `mb_: 2` and `y_: 1` are
        // whole-core values in that export, and `coreletSplit_` names only `out`.
        assert_eq!(
            core.corelet_extent(
                PrimaryDim::Mb,
                Some(Corelet::at::<1>()),
                &plain,
                None,
                false
            ),
            Some(Extent(2))
        );
        assert_eq!(
            core.corelet_extent(PrimaryDim::Y, Some(Corelet::at::<0>()), &plain, None, false),
            Some(Extent(1))
        );
        // `in_: -1` in the same export — the unstated slot, reached through the base arm.
        assert_eq!(
            core.corelet_extent(
                PrimaryDim::In,
                Some(Corelet::at::<0>()),
                &plain,
                None,
                false
            ),
            None
        );

        // The same program's `dataStageParam_["2"].ss_`: `out_: 128`, `coreletSplit_: {"out": [64,
        // 64]}`, `mb_: 1`, `y_: -1` — a second published pair, which pins the answer to the stage.
        let mut inner = core_stage();
        inner.extents.insert(PrimaryDim::Out, Extent(128));
        inner.extents.insert(PrimaryDim::Mb, Extent(1));
        inner.extents.remove(&PrimaryDim::Y);
        inner
            .corelet_split
            .insert(PrimaryDim::Out, vec![Extent(64), Extent(64)]);
        assert_eq!(
            inner.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<0>()),
                &plain,
                None,
                false
            ),
            Some(Extent(64))
        );
        assert_eq!(
            inner.corelet_extent(PrimaryDim::Out, None, &plain, None, false),
            Some(Extent(128))
        );

        // ⛔ THE SHORT SPLIT — `coreletSplit_.at("out")` holding ONE share while corelet 1 is asked
        // for. The reference is already committed to `.at(1)` and throws (`dsc/dims.cpp:635`); it does
        // NOT come back with `out_`.
        let mut short = core_stage();
        short
            .corelet_split
            .insert(PrimaryDim::Out, vec![Extent(256)]);
        assert_eq!(
            short.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<1>()),
                &plain,
                None,
                false
            ),
            None
        );
        assert_eq!(
            short.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<0>()),
                &plain,
                None,
                false
            ),
            Some(Extent(256))
        );
    }

    /// e011 — CONSTRUCTED, no g0 stage states a `symbolicDimInfo_`: the granularity re-expression
    /// (`dsc/dims.cpp:618-628`) and the density divide, both applied to the SHARE and not the whole.
    #[test]
    fn the_share_is_re_expressed_in_granularity_units_and_then_density_scaled() {
        let mut symbolic = core_stage();
        symbolic.symbolic = Symbolic::new(
            BTreeMap::from([(
                PrimaryDim::Out,
                SymbolicDimInfo {
                    max_size: MaxSize(512),
                    granularity: Granularity::new(NonZeroU32::new(128).expect("a positive step")),
                },
            )]),
            BTreeMap::new(),
        );
        let plain = PaddingForm::default();

        // `maxSize_ / granularity_ = 4`, and the share is filled against the max — so 256 max units
        // is 64 granularity units. ⭐ THE BASE ARM ANSWERS THE WHOLE CORE'S `granularity_` INSTEAD,
        // which is what makes the two reads different questions.
        assert_eq!(
            symbolic.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<0>()),
                &plain,
                None,
                true
            ),
            Some(Extent(64))
        );
        assert_eq!(
            symbolic.corelet_extent(PrimaryDim::Out, None, &plain, None, true),
            Some(Extent(128))
        );
        // The same share unscaled, so the granularity flag is the only difference.
        assert_eq!(
            symbolic.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<0>()),
                &plain,
                None,
                false
            ),
            Some(Extent(256))
        );

        // `val % factor != 0` (`:625`) reached through the SHARE: 250 is not a multiple of 4.
        let mut ragged = symbolic.clone();
        ragged
            .corelet_split
            .insert(PrimaryDim::Out, vec![Extent(256), Extent(250)]);
        assert_eq!(
            ragged.corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<1>()),
                &plain,
                None,
                true
            ),
            None
        );

        // `size *= dimDensity` with `dimDensity = 1.0 / 4`: the share divided, not the whole.
        let block = ScaleBlock::of(Cardinality(4)).expect("four elements is a scale block");
        assert_eq!(
            core_stage().corelet_extent(
                PrimaryDim::Out,
                Some(Corelet::at::<0>()),
                &plain,
                Some(block),
                false
            ),
            Some(Extent(64))
        );
    }
}

#[cfg(test)]
mod tests_e012 {
    //! ⭐⭐ THE REFERENCE PUBLISHES BOTH SIDES OF THIS FUNCTION TOO. MEASURED over all 187 g0
    //! reference exports: 1,270 of their 4,850 dim blocks state a non-empty `rowSplit_` (93 of 187
    //! programs), and in EVERY one `sum(rowSplit_[d][cl]) == the whole-core slot` with each row's
    //! share `slot / 8` — so the row view's answer and the base view's answer are both the
    //! reference's own numbers rather than ours.
    //!
    //! ⛔ WHAT THE CORPUS CANNOT SHOW, SAME MEASUREMENT: every one of those 1,270 inner maps is
    //! keyed on corelet `"0"` ALONE, none of their dims is in `coreletSplit_` too, and `peSfpSplit_`
    //! is empty on all 4,850 — so the SUM arm and the PE/SFP arm are CONSTRUCTED from the
    //! reference's own code and marked as such.

    use super::{Granularity, MaxSize, StageDims, Symbolic, SymbolicDimInfo};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PrimaryDim, VectorComp,
    };
    use crate::schedule::ddc::transformation_util::PaddingForm;
    use crate::schedule::ddc::v1::{DimSample, PeSfpShares};
    use crate::units::{Corelet, Row};
    use std::collections::BTreeMap;
    use std::num::NonZeroU32;

    /// `g0/debug/sdsc_100/sdsc.json`'s `dataStageParam_["0"].ss_` — `name_: "core"`, `in_: 64`,
    /// `out_: 64`, `mb_: 1`, `y_: 1`, `i_: -1`, `rowSplit_: {"in": {"0": [8, 8, 8, 8, 8, 8, 8, 8]}}`
    /// and EMPTY `coreletSplit_`/`peSfpSplit_`/`paddingSizes_`/`symbolicDimInfo_`.
    fn row_split_stage() -> StageDims {
        StageDims {
            extents: BTreeMap::from([
                (PrimaryDim::In, Extent(64)),
                (PrimaryDim::Out, Extent(64)),
                (PrimaryDim::Mb, Extent(1)),
                (PrimaryDim::Y, Extent(1)),
                (PrimaryDim::I, Extent(-1)),
            ]),
            row_split: BTreeMap::from([(
                PrimaryDim::In,
                BTreeMap::from([(Corelet::at::<0>(), vec![Extent(8); 8])]),
            )]),
            ..StageDims::default()
        }
    }

    /// `ptrowId = row`, and `clId` where one is named.
    fn at_row(row: u32, corelet: Option<Corelet>) -> DimSample {
        DimSample {
            comp: None,
            row: Some(Row::checked(row).expect("a row of this PT")),
            corelet,
        }
    }

    /// e012 — the row view answers the SHARE and the base view the WHOLE, on the reference's own
    /// row-split exports.
    #[test]
    fn the_reference_s_own_row_split_export_is_what_the_row_view_answers() {
        let stage = row_split_stage();
        let plain = PaddingForm::default();

        // `rowSplit_["in"]["0"] = [8; 8]` read with `clId = -1` — the `begin()` arm, since
        // `coreletSplit_` is empty in that export. Each of the eight rows answers its own share.
        for row in 0..8 {
            assert_eq!(
                stage.sampled_extent(PrimaryDim::In, at_row(row, None), &plain, None, false),
                Some(Extent(8))
            );
        }
        // The same share through `.at(0).at(row)`.
        assert_eq!(
            stage.sampled_extent(
                PrimaryDim::In,
                at_row(3, Some(Corelet::at::<0>())),
                &plain,
                None,
                false
            ),
            Some(Extent(8))
        );
        // `in_: 64` with no row named — and 8 shares of 8 summing to 64 is the pair that export
        // states together, so reading either for the other is the defect this asserts against.
        assert_eq!(
            stage.sampled_extent(PrimaryDim::In, DimSample::WHOLE, &plain, None, false),
            Some(Extent(64))
        );

        // ⛔ THE NEGATIVE CONTROL — corelet 1 is not a key of that inner map, and the reference is
        // already committed to `.at(clId)` (`dsc/dims.cpp:667`). It does NOT come back with `in_`.
        assert_eq!(
            stage.sampled_extent(
                PrimaryDim::In,
                at_row(0, Some(Corelet::at::<1>())),
                &plain,
                None,
                false
            ),
            None
        );

        // `rowSplit_` names only `in`, so a row id on `out` is still the whole `out_: 64` ...
        assert_eq!(
            stage.sampled_extent(PrimaryDim::Out, at_row(0, None), &plain, None, false),
            Some(Extent(64))
        );
        // ... and `i_: -1` is the unstated slot, reached through that same base arm.
        assert_eq!(
            stage.sampled_extent(PrimaryDim::I, at_row(0, None), &plain, None, false),
            None
        );

        // `g0/debug/sdsc_15/sdsc.json`'s `dataStageParam_["0"].ss_`: `in_: 2048` with
        // `rowSplit_: {"in": {"0": [256 x 8]}}` — a second published pair, at another scale.
        let mut wide = row_split_stage();
        wide.extents.insert(PrimaryDim::In, Extent(2048));
        wide.row_split.insert(
            PrimaryDim::In,
            BTreeMap::from([(Corelet::at::<0>(), vec![Extent(256); 8])]),
        );
        assert_eq!(
            wide.sampled_extent(PrimaryDim::In, at_row(7, None), &plain, None, false),
            Some(Extent(256))
        );
        assert_eq!(
            wide.sampled_extent(PrimaryDim::In, DimSample::WHOLE, &plain, None, false),
            Some(Extent(2048))
        );
    }

    /// e012 — CONSTRUCTED, no g0 export states a `peSfpSplit_` or a row-split dim that
    /// `coreletSplit_` names too: the PE/SFP side, the conditional `clId = -1` sum and the arm order
    /// (`dsc/dims.cpp:664-702`).
    #[test]
    fn the_pe_sfp_side_and_the_conditional_corelet_sum_follow_the_reference_s_arms() {
        let plain = PaddingForm::default();
        let at_comp = |comp, corelet| DimSample {
            comp: Some(comp),
            row: None,
            corelet,
        };

        // `peSfpSplit_["in"]["0"] = {PE: 32, SFP: 16}` with an empty `coreletSplit_`: `.begin()` for
        // `clId = -1`, `.at(0)` for corelet 0, and each side reads its own half.
        let mut vector = row_split_stage();
        vector.row_split.clear();
        vector.pe_sfp_split = BTreeMap::from([(
            PrimaryDim::In,
            BTreeMap::from([(
                Corelet::at::<0>(),
                PeSfpShares {
                    pe: Extent(32),
                    sfp: Extent(16),
                },
            )]),
        )]);
        for (comp, share) in [(VectorComp::Pe, 32), (VectorComp::Sfp, 16)] {
            assert_eq!(
                vector.sampled_extent(PrimaryDim::In, at_comp(comp, None), &plain, None, false),
                Some(Extent(share))
            );
            assert_eq!(
                vector.sampled_extent(
                    PrimaryDim::In,
                    at_comp(comp, Some(Corelet::at::<0>())),
                    &plain,
                    None,
                    false
                ),
                Some(Extent(share))
            );
        }
        // The same stop as the row arm — corelet 1 is not a key (`dsc/dims.cpp:687`).
        assert_eq!(
            vector.sampled_extent(
                PrimaryDim::In,
                at_comp(VectorComp::Pe, Some(Corelet::at::<1>())),
                &plain,
                None,
                false
            ),
            None
        );

        // ⛔ THE SUM IS CONDITIONAL: `clId = -1` adds the corelets' row shares only where
        // `coreletSplit_` names the dim as well (`:669-675`); otherwise it is the FIRST corelet's
        // share alone, and summing unconditionally doubles this stage's answer.
        let mut two = row_split_stage();
        two.row_split.insert(
            PrimaryDim::In,
            BTreeMap::from([
                (Corelet::at::<0>(), vec![Extent(4); 8]),
                (Corelet::at::<1>(), vec![Extent(4); 8]),
            ]),
        );
        assert_eq!(
            two.sampled_extent(PrimaryDim::In, at_row(0, None), &plain, None, false),
            Some(Extent(4))
        );
        two.corelet_split
            .insert(PrimaryDim::In, vec![Extent(32), Extent(32)]);
        assert_eq!(
            two.sampled_extent(PrimaryDim::In, at_row(0, None), &plain, None, false),
            Some(Extent(8))
        );
        // A named corelet is still that corelet's share, corelet-split or not.
        assert_eq!(
            two.sampled_extent(
                PrimaryDim::In,
                at_row(0, Some(Corelet::at::<1>())),
                &plain,
                None,
                false
            ),
            Some(Extent(4))
        );

        // ⭐ THE ROW ARM IS TESTED FIRST (`:664` before `:683`), so a sample naming both a row and a
        // component reads `rowSplit_`.
        let mut both = two.clone();
        both.pe_sfp_split = BTreeMap::from([(
            PrimaryDim::In,
            BTreeMap::from([(Corelet::at::<0>(), PeSfpShares::both(Extent(99)))]),
        )]);
        assert_eq!(
            both.sampled_extent(
                PrimaryDim::In,
                DimSample {
                    comp: Some(VectorComp::Pe),
                    row: Some(Row::at::<0>()),
                    corelet: Some(Corelet::at::<0>()),
                },
                &plain,
                None,
                false
            ),
            Some(Extent(4))
        );
    }

    /// e013 — the one-argument spelling answers `in_: 64` on the very export whose `rowSplit_` makes
    /// the row view answer 8, `i_: -1` as an absence and a symbolic `out` as its `maxSize_`.
    #[test]
    fn the_one_argument_spelling_answers_the_reference_s_whole_dim_not_its_row_share() {
        let stage = row_split_stage();

        // `g0/debug/sdsc_100/sdsc.json` states `in_: 64` AND `rowSplit_["in"]["0"] = [8; 8]` together.
        // ⛔ THE NEGATIVE CONTROL IS THE 8: `ptrowId` defaults to `-1`, so this must NOT be a share.
        assert_eq!(stage.whole_extent(PrimaryDim::In), Some(Extent(64)));
        assert_eq!(stage.whole_extent(PrimaryDim::Out), Some(Extent(64)));
        assert_eq!(stage.whole_extent(PrimaryDim::Mb), Some(Extent(1)));
        assert_eq!(stage.whole_extent(PrimaryDim::Y), Some(Extent(1)));
        // `i_: -1` in that export, STATED as such above — `calculate_padded`'s `val < 0` short-circuit
        // is the `-1` read back as an absence, not as a size.
        assert_eq!(stage.whole_extent(PrimaryDim::I), None);
        // ⛔ AND THE RAW SLOT IS NOT THAT ANSWER: `extent` hands the `-1` straight back, which is
        // what makes these two functions distinguishable at all.
        assert_eq!(stage.extent(PrimaryDim::I), Some(Extent(-1)));

        // ⛔ THE SECOND DISTINCTION, CONSTRUCTED — `symbolicDimInfo_` is empty on all 4,850 g0 dim
        // blocks, so only the reference's own code states this: a symbolic dim answers `maxSize_`
        // (`dsc/dims.cpp:521-527`) and NOT the `out_: 64` slot sitting beside it.
        let mut symbolic = row_split_stage();
        symbolic.symbolic = Symbolic::new(
            BTreeMap::from([(
                PrimaryDim::Out,
                SymbolicDimInfo {
                    max_size: MaxSize(256),
                    granularity: Granularity::new(NonZeroU32::new(32).expect("a positive step")),
                },
            )]),
            BTreeMap::new(),
        );
        assert_eq!(symbolic.whole_extent(PrimaryDim::Out), Some(Extent(256)));
        assert_eq!(symbolic.extent(PrimaryDim::Out), Some(Extent(64)));
    }
}

#[cfg(test)]
mod tests_e006_data_struct_dims {
    use super::*;

    /// `compound()` (`dsc/dims.cpp:84-111`) OVER ITS FIVE PAIRS: `IJ = I·J` and `KIJ = KI·KJ` land,
    /// a pair with one operand missing removes its product, and the three deprecated products
    /// (`rc_`, `sij_`, `zij_`) stay absent because `clearDeprecatedFields` (`ddc/ddcv1.cpp:2081`)
    /// has already put `-1` in all six of their operands.
    #[test]
    fn compound_computes_the_two_live_products_and_no_deprecated_one() {
        let mut dims = StageDims::default();
        for (dim, extent) in [
            (PrimaryDim::I, 4),
            (PrimaryDim::J, 5),
            (PrimaryDim::Ki, 3),
            (PrimaryDim::Kj, 7),
        ] {
            dims.extents.insert(dim, Extent(extent));
        }
        dims.compound();
        assert_eq!(dims.extent(PrimaryDim::Ij), Some(Extent(20)));
        assert_eq!(dims.extent(PrimaryDim::Kij), Some(Extent(21)));
        // Every dim the map now names is one of `primaryDimToValHandler_st`'s twelve, so no
        // deprecated product can have been written.
        assert_eq!(dims.extents.len(), 6);

        // `else { ij_ = -1; }` — the product's own absence.
        dims.extents.remove(&PrimaryDim::J);
        dims.compound();
        assert_eq!(dims.extent(PrimaryDim::Ij), None);
        assert_eq!(dims.extent(PrimaryDim::Kij), Some(Extent(21)));
    }
}

#[cfg(test)]
mod tests_e002_dim_padding_sizes {
    use super::*;

    /// THE TWO WRITERS OF `padBack_ = padFront_ = -1` PART ON AN UNPADDED DIM:
    /// `ddc/ddcv1.cpp:1172` voids the entry it emplaced whatever it held, while
    /// `L3DlOpsScheduler.cpp:137` first tests `padBack_ != 0 || padFront_ != 0` — and a
    /// default-constructed `paddingSizes_` entry (`dsc/dsc2.cpp:3665`) is `0`/`0`.
    #[test]
    fn only_the_guarded_void_leaves_an_unpadded_dim_alone() {
        assert_eq!(PadSizes::Unpadded.voided(), PadSizes::Voided);
        assert_eq!(PadSizes::Unpadded.voided_if_padded(), PadSizes::Unpadded);
        let sized = PadSizes::of(PadElems(1), PadElems(0));
        assert_eq!(sized.voided(), PadSizes::Voided);
        assert_eq!(sized.voided_if_padded(), PadSizes::Voided);
    }
}
