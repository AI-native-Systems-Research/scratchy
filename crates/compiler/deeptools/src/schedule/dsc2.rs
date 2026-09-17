// SPDX-License-Identifier: Apache-2.0

//! THE `dsc2` VOCABULARY THE ddc FOLD UNITS TRAFFIC IN — a reduced view of `dsc/dsc2.h`.
//!
//! ⭐ REDUCED, NOT INVENTED. Every type here is one C++ declaration's fields narrowed to what the
//! ported units read and write; the citation is on each. `src/bridges/superdsc_to_dataflow_ir/
//! shape_constraints.rs` carries a DIFFERENT reduced view of `ScheduleNode` — that one projects a
//! node onto its `data_connect=` reads and writes and carries no unit, lds or fold, so the two are
//! separate projections of one C++ class rather than one fact spelled twice.
//!
//! ⛔ THE `DT_ERROR`/`DT_CHECK` ARMS ARE GONE BY CONSTRUCTION, not by a runtime refusal:
//!   * [`Node`] has exactly the three arms `dbgPrint`/`getComponent` accept, so
//!     `"Unsupported node type"` is unspellable.
//!   * [`Dsts`] is non-empty, so `dstVias_.at(0)` cannot throw.
//!   * [`CoordinateCategory`] omits `UNKNOWN_COORD`, so `addFold`'s `default:` arm is unspellable.
//!   * [`LayoutDims`] is non-empty, so `getLayoutDims`' `DT_CHECK(!layoutDimOrder_.empty())` holds.
//!   * An operand and its [`DataInfo`] are ONE value, so the length mismatch `dbgPrint` walks into
//!     (it bounds the loop by `inputsLdsAndLoopOffsets_.size()` and indexes `inputs_`) cannot occur.

use core::num::NonZeroU64;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use sys_arch_spec::arch_enums::{DataLocation, SenComponent};

use crate::arch::{Bytes, Elements, Sticks};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickPart, cumulative_stick_sizes,
};
use crate::formats::DataFormat;
use crate::generated::{DataConnect, Mode, ParamKey, ParamValue, RegName};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{ConstIdx, NodeId, PadType};
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind, Metadata};
use crate::schedule::ddc::transformation::{LoopId, MaskLoopOffset};
use crate::schedule::ddc::transformation_util::PaddingForm;
use crate::schedule::ddc::v1::{ConstEleOffset, LdsSticks, LoopEleOffset};
use crate::schedule::ddl::ops::DdlComputeType;
use crate::schedule::l3::dl_ops::{GtrGroupId, Shares, VariableSymbol};
use crate::schedule::l3::dsc::{IndirectAlloc, WkSlice};
use crate::units::{Core, Corelet, NumFolds};

impl PrimaryDim {
    /// Every layout dim IN `PrimaryDimTypes`' OWN ORDINAL ORDER (`dsc/dims.h:34`), which is what a
    /// `std::map<PrimaryDimTypes, ..>` iterates in — `rowSplit_.begin()->first` is the first of
    /// these that the stage splits.
    pub const ALL: [Self; 12] = [
        Self::In,
        Self::Out,
        Self::Ij,
        Self::Mb,
        Self::X,
        Self::Y,
        Self::Kij,
        Self::I,
        Self::J,
        Self::Ki,
        Self::Kj,
        Self::X1,
    ];

    /// `EnumsConversion::primaryDimToString` (`dsc/dims.cpp:22`) — total over the 12 dims, so the
    /// reference's `.at()` cannot throw for any of them.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::Ij => "ij",
            Self::Mb => "mb",
            Self::X => "x",
            Self::Y => "y",
            Self::Kij => "kij",
            Self::I => "i",
            Self::J => "j",
            Self::Ki => "ki",
            Self::Kj => "kj",
            Self::X1 => "x1",
        }
    }
}

/// WHICH LABELLED DATA STRUCTURE — `DataInfo::myLdsIdx_` (`dsc/dsc2.h:722`) once its `-1` is an
/// [`Option`]. Issued by the [`Dsc`] that owns the `labeledDs_` list; the range `DT_CHECK` in
/// `getLayoutDims` is that owner's, not a caller's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LdsIdx(pub u32);

/// HOW MANY OF THE FOLD'S OWN STEPS — `FoldDimProp::factor_`, a `uint32_t`
/// (`util/foldManager/foldInfrastructure.h:119`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldCardinality(pub u32);

/// A FOLD'S AFFINE COEFFICIENT — `alpha`/`beta`, whose `Dtype` is `CoordinateBaseType`, i.e.
/// `int64_t` (`dsc/dsc2.h:442`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldCoeff(pub i64);

/// A FOLD'S NAME — `FoldDimProp::label_`. Not a closed set: the reference builds it by
/// concatenation (`"rowsplit_fold_" + primaryDimToString.at(dim)`), so it is a name, not an enum.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldLabel(pub String);

/// WHAT KIND OF FOLD — `CoordinateCategory` (`dsc/dsc2.h:66`) less `UNKNOWN_COORD`, which is only
/// ever the value `addFold`'s `default:` arm raises *"Unsupported coordinate category"* on
/// (`dsc/dsc2.h:139-140`). Dropping it removes that refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoordinateCategory {
    /// `SPATIAL_COORD` — a fold across components.
    Spatial,
    /// `TEMPORAL_COORD` — a fold across loop iterations.
    Temporal,
    /// `ELEM_ARR_COORD` — the element arrangement within a stick.
    ElemArr,
}

/// WHERE IN A DIM'S FOLD LIST — `CoordinateFoldPosition` (`dsc/dsc2.h:73`). The discriminants are
/// load-bearing: they are the `pos` a `FoldManager` is indexed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum FoldPosition {
    /// The core work-slice fold, outermost.
    Core = 0,
    /// The corelet fold.
    Corelet = 1,
    /// The row-split fold.
    RowSplit = 2,
}

/// ONE FOLD OF ONE DIM — a `FoldDimProp` plus the affine pair `insertAlphaBeta` puts on it
/// (`foldInfrastructure.h:1328`, `:2394`). Only `Affine` folds carry alpha/beta, and `addFold`
/// builds nothing else, so the base function type is not a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// `FoldDimProp::factor_`.
    pub cardinality: FoldCardinality,
    /// `FoldDimProp::label_`.
    pub label: FoldLabel,
    /// `insertAlpha`.
    pub alpha: FoldCoeff,
    /// `insertBeta`.
    pub beta: FoldCoeff,
}

/// ONE DIM'S FOLDS, OUTERMOST FIRST — `FoldManager::dim_prop_` plus the three per-category counts
/// `CoordinateType` keeps beside it (`numOfSpatialFolds_` and friends, `dsc/dsc2.h:120-141`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FoldDim {
    folds: VecDeque<Fold>,
    /// Field: e005_CoordinateType.numOfSpatialFolds_
    ///
    /// THIS DIM'S ENTRY in `numOfSpatialFolds_` (`dsc/dsc2.h:436`). Absent from the reference's map
    /// IS zero — `getNumOfSpatialFolds` (`:144`) says so outright — so the count is total here.
    spatial: u32,
    /// Field: e005_CoordinateType.numOfTemporalFolds_
    ///
    /// This dim's entry in `numOfTemporalFolds_` (`dsc/dsc2.h:437`), zero when absent (`:147`).
    temporal: u32,
    /// Field: e005_CoordinateType.numOfElemArrFolds_
    ///
    /// This dim's entry in `numOfElemArrFolds_` (`dsc/dsc2.h:438`), zero when absent (`:150`).
    elem_arr: u32,
}

impl FoldDim {
    /// A DIM WITH NO FOLDS AND NO COUNTS — the entry a `CoordinateType` has not been given yet, and
    /// the fold space a default-constructed `FoldManager` starts as.
    pub const EMPTY: Self = Self {
        folds: VecDeque::new(),
        spatial: 0,
        temporal: 0,
        elem_arr: 0,
    };

    /// This dim's folds, position 0 first — the order `getFoldDimSize(pos)` indexes.
    pub fn folds(&self) -> impl Iterator<Item = &Fold> {
        self.folds.iter()
    }

    /// `getNumOfSpatialFolds`.
    #[must_use]
    pub const fn spatial_folds(&self) -> u32 {
        self.spatial
    }

    /// `getNumOfTemporalFolds`.
    #[must_use]
    pub const fn temporal_folds(&self) -> u32 {
        self.temporal
    }

    /// `getNumOfElemArrFolds`.
    #[must_use]
    pub const fn elem_arr_folds(&self) -> u32 {
        self.elem_arr
    }

    /// `getFoldDimSize(pos)` — the cardinality at a named fold position, absent where the dim has
    /// not been folded that far.
    #[must_use]
    pub fn cardinality_at(&self, pos: FoldPosition) -> Option<FoldCardinality> {
        self.folds.get(pos as usize).map(|fold| fold.cardinality)
    }
}

/// Replaces: e005_CoordinateType
///
/// A COORDINATE — `CoordinateType<CoordinateBaseType>` (`dsc/dsc2.h:76`), carrying all SEVEN of its
/// declared members: `coordinates_` with the three per-dim fold counts fused into it, plus
/// `padding_`, `foldConstructed_` and `coreIdToWkSlice_`.
///
/// ⭐ THE THREE COUNT MAPS LIVE ON THE PER-DIM VALUE, NOT BESIDE IT. The reference keeps four
/// independent `std::map`s keyed by the same dim, and `getCoordinateCategoryOfPos` (`:153`) reads
/// three of them beside a `coordinates_.at(dim)` — so a dim named in one map and absent from another
/// is a state it has to survive. Fusing them into [`FoldDim`] makes that state unspellable.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Coordinate {
    /// Field: e005_CoordinateType.coordinates_
    ///
    /// `coordinates_` (`dsc/dsc2.h:431`), a `map<PrimaryDimTypes, FoldManager<Dtype>>`.
    dims: BTreeMap<PrimaryDim, FoldDim>,
    /// Field: e005_CoordinateType.padding_
    ///
    /// `padding_` (`dsc/dsc2.h:439`) — how this coordinate READS each dim, which `clearFoldForDim`
    /// deliberately leaves standing: *"Note: Padding is not cleared."* (`:100-102`).
    padding: PaddingForm,
    /// Field: e005_CoordinateType.foldConstructed_
    ///
    /// `foldConstructed_` (`dsc/dsc2.h:435`), which `completeFoldConstruction` (`:118`) sets and only
    /// `clear` (`:97`) puts back.
    fold_constructed: bool,
    /// Field: e005_CoordinateType.coreIdToWkSlice_
    ///
    /// `coreIdToWkSlice_` (`dsc/dsc2.h:432`) — WHICH WORK SLICE EACH CORE TAKES per dim as THIS
    /// coordinate reads it, which entry 355 rewrites away from the super-DSC's own answer.
    core_id_to_wk_slice: BTreeMap<Core, WkSlice>,
}

impl Coordinate {
    /// `hasCoordForDim` / `coordinates_.count(dim)` (`dsc/dsc2.h:254`).
    #[must_use]
    pub fn covers(&self, dim: PrimaryDim) -> bool {
        self.dims.contains_key(&dim)
    }

    /// This dim's folds, absent where the coordinate does not cover it.
    #[must_use]
    pub fn fold_dim(&self, dim: PrimaryDim) -> Option<&FoldDim> {
        self.dims.get(&dim)
    }

    /// `coordinates_`, in dim order — what a walk over every dim this coordinate covers reads.
    pub fn iter(&self) -> impl Iterator<Item = (PrimaryDim, &FoldDim)> {
        self.dims.iter().map(|(&dim, folds)| (dim, folds))
    }

    /// `CoordinateType::addFold(dim, cat, card, label, alpha, beta, 0)` (`dsc/dsc2.h:120`) — the
    /// `pos == 0` case, which inserts at the FRONT of the dim's fold list
    /// (`foldInfrastructure.h:1348`, `:1371`) and is the only position the fold builders here use.
    ///
    /// ⛔ Front insertion is TOTAL; a positional insert is not (`buildDim` `DT_CHECK`s `pos == 0`
    /// on an empty list and `insertAlphaBeta` `DT_CHECK`s `pos <= size - 1`), so the units that
    /// need one own that surface rather than this one growing a refusal.
    pub fn add_fold_front(
        &mut self,
        dim: PrimaryDim,
        category: CoordinateCategory,
        cardinality: FoldCardinality,
        label: FoldLabel,
        alpha: FoldCoeff,
        beta: FoldCoeff,
    ) {
        let entry = self.dims.entry(dim).or_default();
        entry.folds.push_front(Fold {
            cardinality,
            label,
            alpha,
            beta,
        });
        match category {
            CoordinateCategory::Spatial => entry.spatial += 1,
            CoordinateCategory::Temporal => entry.temporal += 1,
            CoordinateCategory::ElemArr => entry.elem_arr += 1,
        }
    }

    /// `clearFoldForDim(dim)` (`dsc/dsc2.h:103-116`) — empties the dim's folds and zeroes its three
    /// counts.
    ///
    /// ⛔⛔ THE DIM STAYS COVERED. The reference `reset()`s the fold manager in place and leaves the
    /// `coordinates_` entry (and the padding) where it was, so [`Self::covers`] still answers true
    /// afterwards; the fold builders rely on that to re-add a rebuilt fold list to a live dim.
    pub fn clear_fold_for_dim(&mut self, dim: PrimaryDim) {
        if let Some(entry) = self.dims.get_mut(&dim) {
            *entry = FoldDim::default();
        }
    }

    /// `getPadding(dim)` (`dsc/dsc2.h:242`) — how this coordinate READS the dim, which is a
    /// different fact from the allocation's own [`AllocPlacement::padding`] and survives
    /// [`Self::clear_fold_for_dim`] as the reference's own note says (`:100-102`).
    #[must_use]
    pub fn padding(&self, dim: PrimaryDim) -> PadType {
        self.padding.padding(dim)
    }

    /// `setPadding(dim, pad)` (`dsc/dsc2.h:236`).
    pub fn set_padding(&mut self, dim: PrimaryDim, pad: PadType) {
        self.padding.set_padding(dim, pad);
    }

    /// `setPadding(const PaddingFormType)` (`dsc/dsc2.h:240`) — REPLACES the whole form, which is a
    /// DIFFERENT operation from the per-dim [`Self::set_padding`]: a dim the form has no entry for
    /// goes back to reading `NOPAD`.
    pub fn set_padding_form(&mut self, form: PaddingForm) {
        self.padding = form;
    }

    /// `coreIdToWkSlice_.at(core)` TO BE WRITTEN, [`None`] where the coordinate states no slice for
    /// the core — *"Core ID not found."*.
    pub fn wk_slice_mut(&mut self, core: Core) -> Option<&mut WkSlice> {
        self.core_id_to_wk_slice.get_mut(&core)
    }

    /// `coreIdToWkSlice_[core] = slice`.
    pub fn set_wk_slice(&mut self, core: Core, slice: WkSlice) {
        self.core_id_to_wk_slice.insert(core, slice);
    }

    /// `coreIdToWkSlice_`, in core order.
    pub fn wk_slices(&self) -> impl Iterator<Item = (Core, &WkSlice)> {
        self.core_id_to_wk_slice.iter().map(|(&core, at)| (core, at))
    }

    /// `getPadding()` (`dsc/dsc2.h:245`) — the WHOLE padding form, which entry 356 hands straight to
    /// another coordinate's [`Self::set_padding_form`]. Only the dims carrying a style are named;
    /// [`Self::padding`] already reads every other one as `NOPAD`.
    #[must_use]
    pub const fn padding_form(&self) -> &PaddingForm {
        &self.padding
    }

    /// `foldConstructed()` (`dsc/dsc2.h:119`).
    #[must_use]
    pub const fn fold_constructed(&self) -> bool {
        self.fold_constructed
    }

    /// `completeFoldConstruction()` — a one-way latch, as the reference's own setter is.
    pub const fn complete_fold_construction(&mut self) {
        self.fold_constructed = true;
    }

    /// `fm.insertBeta(offset + fm.getBeta(fm.getNumDims() - 1), fm.getNumDims() - 1)` — the ONE
    /// positional beta write in the fold builders (`ddc/ddc_fold.cpp:4714-4716`), which adds an
    /// offset to the INNERMOST level of a dim's fold list.
    ///
    /// ⛔ THE READ AND THE WRITE ARE ONE OPERATION, which is what makes this total: the reference's
    /// `getNumDims() - 1` underflows on an empty fold list and its `insertBeta` then `DT_CHECK`s the
    /// position, so a dim carrying no fold has no innermost level and this does nothing.
    pub fn add_to_innermost_beta(&mut self, dim: PrimaryDim, offset: FoldCoeff) {
        if let Some(fold) = self
            .dims
            .get_mut(&dim)
            .and_then(|entry| entry.folds.back_mut())
        {
            fold.beta = FoldCoeff(fold.beta.0 + offset.0);
        }
    }
}

/// A NODE'S NAME — `ScheduleNode::name_`, whose default is the empty string.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeName(pub String);

/// Replaces: e007_DataInfo
///
/// WHAT AN OPERAND KNOWS ABOUT ITS DATA — `DataInfo` (`dsc/dsc2.h:721-739`), ALL TEN of its declared
/// fields. `dataConnect_` is a closed set, so it is [`DataConnect`] and not a string; a
/// default-constructed `DataInfo` leaves it EMPTY, which is the [`None`].
///
/// ⭐ THE SIX OFFSET AND ADDRESS FIELDS USED TO LIVE ONLY ON [`super::ddc::v1::DataInfoFill`], a
/// payload handed to a carrier trait's `fill`, so no later unit could READ one back: four `todo!` stubs
/// in `schedule/stages/ddc_store2.rs` state that against themselves. They are declared HERE, where
/// the reference declares them, and `DataInfoFill` is the same six values in flight.
///
/// ⛔ NOT [`Copy`], AND THAT IS THE POINT OF DECLARING THEM. `constEleOffsets_`, `loopEleOffsets_`
/// and `bufferAddrOffset_` are nested `std::map`s and `startAddr_` is a `FoldManager`; the C++ struct
/// is not trivially copyable either, and a per-operand clone of four maps is not something a `Copy`
/// bound should hide at the use site.
///
/// ⚠️ [`super::ddc::fold::DataStream`] carries `myLdsIdx_` and `constantId_` as ONE
/// [`super::ddc::fold::DataOrigin`], which is the stronger statement: `isLabeledDs` and `isConstant`
/// both `DT_CHECK` that the two are not set together (`dsc/dsc2.h:742-743`, `:747-748`). Converging
/// the spellings rewrites 162 call sites of these two fields, so it is its own unit, not a doc edit.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataInfo {
    /// Field: e007_DataInfo.dataConnect_
    ///
    /// `dataConnect_` (`dsc/dsc2.h:739`).
    pub data_connect: Option<DataConnect>,
    /// Field: e007_DataInfo.myLdsIdx_
    ///
    /// `myLdsIdx_` (`dsc/dsc2.h:722`), once its `-1` is an [`Option`].
    pub my_lds_idx: Option<LdsIdx>,
    /// Field: e007_DataInfo.constantId_
    ///
    /// `constantId_` (`dsc/dsc2.h:726`), once its `-1` is an [`Option`].
    pub constant_id: Option<ConstIdx>,
    /// Field: e007_DataInfo.latchDataId_
    ///
    /// `latchDataId_` (`dsc/dsc2.h:725`), *"used to link producer and consumer when using LATCH"*,
    /// once its `-1` is an [`Option`]. Entry 339 is the one unit that mints one.
    pub latch_data_id: Option<LatchDataId>,
    /// Field: e007_DataInfo.startAddr_
    ///
    /// `startAddr_` (`dsc/dsc2.h:723`), *"per core, corelet, and sdsc folds"* — the same
    /// `FoldManager<int64_t>` an allocation's `startAddressCoreCorelet_` is, so it is the same
    /// [`StartAddress`] and not a second spelling of one.
    pub start_addr: StartAddress,
    /// Field: e007_DataInfo.isStartAddrSymbolic_
    ///
    /// `isStartAddrSymbolic_` (`dsc/dsc2.h:724`), copied off the allocation.
    pub is_start_addr_symbolic: bool,
    /// Field: e007_DataInfo.constEleOffsets_
    ///
    /// `constEleOffsets_` (`dsc/dsc2.h:727-729`) — *"constant offset on top of the start address. Per
    /// core and corelet"*.
    pub const_ele_offsets: BTreeMap<Core, BTreeMap<Corelet, BTreeMap<PrimaryDim, ConstEleOffset>>>,
    /// Field: e007_DataInfo.loopEleOffsets_
    ///
    /// `loopEleOffsets_` (`dsc/dsc2.h:730-734`), *"Per corelet"*. The reference keys the middle map
    /// by `const LoopNode*`; a [`LoopId`] is that pointer with no way to dangle.
    pub loop_ele_offsets: BTreeMap<Corelet, BTreeMap<LoopId, BTreeMap<PrimaryDim, LoopEleOffset>>>,
    /// Field: e007_DataInfo.bufferAddrOffset_
    ///
    /// `bufferAddrOffset_` (`dsc/dsc2.h:735-737`) — *"address offset to move to the next buffer (per
    /// core and corelet)"*.
    pub buffer_addr_offset: BTreeMap<Core, BTreeMap<Corelet, Bytes>>,
    /// Field: e007_DataInfo.bufferSwitchPosition_
    ///
    /// `bufferSwitchPosition_` (`dsc/dsc2.h:738`) — the loop a multiply-buffered allocation switches
    /// buffers at, whose `nullptr` default is the [`None`].
    pub buffer_switch_position: Option<LoopId>,
}

/// WHICH LATCHED RESULT AN OPERAND IS LINKED TO — one `latchDataId_` (`dsc/dsc2.h:725`). Producer
/// and consumers carry the SAME id, which is the whole point of the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LatchDataId(pub u32);

/// A NODE OPERAND — its component and its [`DataInfo`] AS ONE VALUE.
///
/// ⭐ THIS PAIRING IS THE POINT. The C++ keeps `inputs_` (components) and
/// `inputsLdsAndLoopOffsets_` (data) in two vectors of independent length, and `dbgPrint` bounds
/// its loop by the second while indexing the first with `.at()`. Pairing them makes that throw
/// unspellable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    /// `inputs_`/`outputs_` entry, `src_.unit_`, or `dstVias_.at(i).loc_.unit_`.
    pub unit: SenComponent,
    /// `src_.storage_` / `dstVias_.at(i).loc_.storage_` — the memory the unit reaches, which is the
    /// other half of the `DataLocation` the reference keeps this operand in.
    pub storage: SenComponent,
    /// The matching `..LdsAndLoopOffsets_` entry.
    pub data: DataInfo,
}

/// THE COMPONENTS ONE DESTINATION IS REACHED THROUGH — `DstVia::via_` (`dsc/dsc2.h:818`), in hop
/// order, EMPTY where the transfer reaches its destination straight from `src_`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Hops(pub Vec<SenComponent>);

/// A TRANSFER'S DESTINATIONS — `dstVias_` zipped with `dstLdsAndLoopOffsets_`, NON-EMPTY so that
/// `getComponent`'s `dstVias_.at(0)` is total.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dsts {
    first: Operand,
    rest: Vec<Operand>,
    /// Field: e014_TransferNode.via_
    ///
    /// `dstVias_.at(i).via_` (`dsc/dsc2.h:818`), per destination.
    ///
    /// ⭐ SHORT OR ABSENT IS EMPTY, AND THAT IS THE TRUTH RATHER THAN A GAP: a freshly minted
    /// `DstVia` carries an empty `via_`, so a site that states no route for a destination has stated
    /// the route it has.
    hops: Vec<Hops>,
}

impl Dsts {
    /// A transfer has at least one destination, and this is how that is stated.
    #[must_use]
    pub const fn new(first: Operand, rest: Vec<Operand>) -> Self {
        Self {
            first,
            rest,
            hops: Vec::new(),
        }
    }

    /// The same destinations with the routes that reach them, in destination order.
    #[must_use]
    pub fn with_hops(mut self, hops: Vec<Hops>) -> Self {
        self.hops = hops;
        self
    }

    /// `dstVias_.at(index).via_`.
    #[must_use]
    pub fn hops(&self, index: usize) -> &[SenComponent] {
        self.hops.get(index).map_or(&[], |hops| &hops.0)
    }

    /// Every destination WITH its route — what a hop walk over `dstVias_` reads off each entry.
    pub fn routes(&self) -> impl Iterator<Item = (&Operand, &[SenComponent])> {
        self.iter()
            .enumerate()
            .map(|(index, dst)| (dst, self.hops(index)))
    }

    /// `dstVias_.at(0)` — total.
    #[must_use]
    pub const fn first(&self) -> &Operand {
        &self.first
    }

    /// `dstVias_.front()` AS AN L-VALUE — the end entries 221 and 227 rewrite in place.
    pub const fn first_mut(&mut self) -> &mut Operand {
        &mut self.first
    }

    /// `dstVias_.size()`.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rest.len() + 1
    }

    /// Every destination in order.
    pub fn iter(&self) -> impl Iterator<Item = &Operand> {
        core::iter::once(&self.first).chain(self.rest.iter())
    }

    /// `dstLdsAndLoopOffsets_.at(i)`, absent past the end.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Operand> {
        match index {
            0 => Some(&self.first),
            n => self.rest.get(n - 1),
        }
    }
}

/// ONE LAYOUT DIM'S MAX SIZE — one entry of `AllocateNode::maxDimSizes_` (`dsc/dsc2.h:983`), whose
/// `int` is THREE things in sequence.
///
/// ⛔⛔ A TRI-STATE, NOT A NUMBER, AND THAT IS WHAT STOPS A DOUBLE RESOLVE. Every writer but entry
/// 128 only ever `resize(.., -1)`s it or stores a `dataStageParam_` KEY in it; entry 128 overwrites
/// that key with an ELEMENT COUNT in place, so a second run of entry 128 would read the count back
/// as a datastage index. Here it cannot: [`Self::Resolved`] is not a [`Self::Stage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxDimSize {
    /// `-1` — no datastage bounds this dim.
    Unset,
    /// A `dataStageParam_` index, as it stands BEFORE entry 128.
    Stage(DatastageId),
    /// The extent entry 128 resolved it to, in stick-normalised elements.
    Resolved(Elements),
}

/// AN ALLOCATION'S LAYOUT — `layoutDimOrder_` (`dsc/dsc2.h:982`) zipped with `maxDimSizes_` (`:983`),
/// NON-EMPTY.
///
/// ⛔⛔ TWO REFERENCE ABORTS GONE BY CONSTRUCTION. Zipping the two vectors is
/// `DT_ERROR("Mismatch in allocate layout vectors")` (`ddc/ddcv1.cpp:1715-1716`); being non-empty is
/// `layoutDimOrder_.at(0)` (`:1704`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocLayout {
    first: (PrimaryDim, MaxDimSize),
    rest: Vec<(PrimaryDim, MaxDimSize)>,
}

impl AllocLayout {
    /// An allocation has at least one layout dim, and this is how that is stated.
    #[must_use]
    pub const fn new(first: (PrimaryDim, MaxDimSize), rest: Vec<(PrimaryDim, MaxDimSize)>) -> Self {
        Self { first, rest }
    }

    /// `layoutDimOrder_.at(0)` — total.
    ///
    /// ⚠️ INDEX 0 IS THE INNERMOST LAYOUT DIM, NOT THE OUTERMOST: `dsc/dataOpDsc.h:347` says so of
    /// the same field by name, `dsc/dsc2.cpp:2792-2816` appends the layout dims AFTER the stick dims
    /// into `sizesNoGaps_`, and `dsc/dsc2.cpp:2961-2963` takes a dim's stride to be the product of
    /// everything BEFORE its index there — so index 0 has stride one. The reference's own comment at
    /// `ddc/ddcv1.cpp:1937-1938` walks it "from the innermost to outer dimensions".
    #[must_use]
    pub const fn innermost_dim(&self) -> PrimaryDim {
        self.first.0
    }

    /// The dims with their max sizes, innermost first.
    pub fn iter(&self) -> impl Iterator<Item = (PrimaryDim, MaxDimSize)> + '_ {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }

    /// The same, writable — the `maxDimSizes_[i] = ..` entry 128 performs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (PrimaryDim, MaxDimSize)> {
        core::iter::once(&mut self.first).chain(self.rest.iter_mut())
    }

    /// `layoutDimOrder_` alone, as [`LayoutDims`].
    #[must_use]
    pub fn dims(&self) -> LayoutDims {
        LayoutDims::new(
            self.first.0,
            self.rest.iter().map(|&(dim, _)| dim).collect(),
        )
    }
}

/// WHETHER AN ADDRESS FOLD VARIES ACROSS ITS AXIS — `BaseFuncType`
/// (`util/foldManager/foldInfrastructure.h:39`) narrowed to the two arms `buildFoldSpace` is ever
/// handed and `getFuncType` ever compared against (`ddc/ddcv1.cpp:279`, `:2001`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum AddressFold {
    /// `BaseFuncType::Constant` — the same address on every step of that axis.
    #[default]
    Constant,
    /// `BaseFuncType::Map` — a per-step address, which is what `insertData` then fills.
    Map,
}

/// AN ALLOCATION'S PER-CORE, PER-CORELET START ADDRESS — `startAddressCoreCorelet_`, a
/// `FoldManager<int64_t>` (`dsc/dsc2.h:985-986`): the fold space `buildFoldSpace` lays out, the
/// per-axis [`AddressFold`] it is laid out with, and the addresses `insertData` places into it.
///
/// ⭐ THE PLACED ADDRESSES ARE KEYED BY `(core, corelet)` AND NOT BY A COORDINATE DEQUE. Every write
/// in the authority tree sets `coord[0] = core`, `coord[1] = cl` and leaves the rest at zero
/// (`ddc/ddcv1.cpp:344-350`, `:2005-2011`), so those two axes ARE the key and the deque is the
/// mechanism for reaching it.
///
/// ⭐ ONE `(core, corelet)` MAY HOLD SEVERAL ADDRESSES, and that is entry 222: the L3 scheduler places
/// each execution phase separately and spreads the resulting list over the REMAINING fold coordinates
/// of that `(core, corelet)` (`L3DlOpsScheduler.cpp:4249-4262`), one address per coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StartAddress {
    folds: FoldDim,
    func_types: Vec<AddressFold>,
    placed: BTreeMap<Core, BTreeMap<Corelet, Vec<Bytes>>>,
}

impl StartAddress {
    /// The `FoldManager<int64_t>` A DEFAULT-CONSTRUCTED `DataInfo` OR `AllocateNode` CARRIES — no fold
    /// space laid out, which is `hasZeroFoldDim()`, and nothing placed.
    pub const EMPTY: Self = Self {
        folds: FoldDim::EMPTY,
        func_types: Vec::new(),
        placed: BTreeMap::new(),
    };

    /// An address whose fold space is one dim's folds and nothing placed in it yet.
    #[must_use]
    pub const fn new(folds: FoldDim) -> Self {
        Self {
            folds,
            func_types: Vec::new(),
            placed: BTreeMap::new(),
        }
    }

    /// The fold space's own dim folds.
    #[must_use]
    pub const fn folds(&self) -> &FoldDim {
        &self.folds
    }

    /// `hasZeroFoldDim()` — TRUE before `buildFoldSpace` has laid any axis out.
    #[must_use]
    pub fn has_zero_fold_dim(&self) -> bool {
        self.func_types.is_empty()
    }

    /// `buildFoldSpace(foldProps, foldTypes)` (`ddc/ddcv1.cpp:339`) — `depth` axes, of which only the
    /// core and the corelet are anything but [`AddressFold::Constant`].
    pub fn build_fold_space(&mut self, depth: usize, core: AddressFold, corelet: AddressFold) {
        self.build_fold_space_spread(depth, AddressFold::Constant, core, corelet);
    }

    /// `buildFoldSpace(foldProps, foldTypes(size, defaultFold))` (`l3/l3.cpp:4249`) — the same axes,
    /// laid out over a `default` the L3 sets to [`AddressFold::Map`] when one allocation's addresses
    /// differ from coordinate to coordinate.
    ///
    /// ⭐ THE DEFAULT IS NOT ALWAYS `Constant`, WHICH IS WHY THIS IS THE GENERAL SPELLING. A spread
    /// address has a distinct value at every fold coordinate, so every axis maps.
    pub fn build_fold_space_spread(
        &mut self,
        depth: usize,
        default: AddressFold,
        core: AddressFold,
        corelet: AddressFold,
    ) {
        self.func_types = vec![default; depth];
        if let Some(slot) = self.func_types.get_mut(FoldPosition::Core as usize) {
            *slot = core;
        }
        if let Some(slot) = self.func_types.get_mut(FoldPosition::Corelet as usize) {
            *slot = corelet;
        }
    }

    /// `getFuncType(pos)`, absent where the fold space does not reach that far.
    #[must_use]
    pub fn func_type(&self, pos: FoldPosition) -> Option<AddressFold> {
        self.func_types.get(pos as usize).copied()
    }

    /// `insertData(addr, {{0, core}, {1, corelet}})` — ONE address over every remaining coordinate.
    pub fn insert(&mut self, core: Core, corelet: Corelet, address: Bytes) {
        self.insert_spread(core, corelet, vec![address]);
    }

    /// The same write with ONE ADDRESS PER REMAINING FOLD COORDINATE, innermost coordinate first —
    /// entry 222's `addrIt++` walk over `getFlattenedCoordinates({{0, 0}, {1, 0}})`.
    pub fn insert_spread(&mut self, core: Core, corelet: Corelet, addresses: Vec<Bytes>) {
        self.placed
            .entry(core)
            .or_default()
            .insert(corelet, addresses);
    }

    /// `getSingleData({{0, core}, {1, corelet}})`, absent where nothing was placed there.
    ///
    /// ⛔ ALSO ABSENT FOR A SPREAD, which is `getSingleData`'s own *"more than one data"* refusal:
    /// [`Self::spread`] is what reads that shape.
    #[must_use]
    pub fn at(&self, core: Core, corelet: Corelet) -> Option<Bytes> {
        match self.placed.get(&core)?.get(&corelet)?.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

    /// Every address placed at that `(core, corelet)`, in fold-coordinate order — EMPTY where nothing
    /// was placed there at all.
    #[must_use]
    pub fn spread(&self, core: Core, corelet: Corelet) -> &[Bytes] {
        self.placed
            .get(&core)
            .and_then(|per_cl| per_cl.get(&corelet))
            .map_or(&[], Vec::as_slice)
    }

    /// `getDataAndFoldCoordinates({{1, corelet}})` — every core's address at ONE corelet, which is
    /// the frontier entry 259 copies onto the others.
    #[must_use]
    pub fn at_corelet(&self, corelet: Corelet) -> Vec<(Core, Bytes)> {
        self.placed
            .iter()
            .flat_map(|(&core, per_cl)| {
                per_cl
                    .get(&corelet)
                    .map_or(&[][..], Vec::as_slice)
                    .iter()
                    .map(move |&addr| (core, addr))
            })
            .collect()
    }

    /// `getAllData()`, in core-then-corelet order.
    #[must_use]
    pub fn all(&self) -> Vec<Bytes> {
        self.placed
            .values()
            .flat_map(|per_cl| per_cl.values().flatten().copied())
            .collect()
    }

    /// THE ONE ADDRESS EVERY CORE AND CORELET SHARES — `getAllData()` FUSED WITH the `std::equal`
    /// beside it and the `allAddr.at(0)` after it (`ddc/ddcv1.cpp:3369-3377`).
    ///
    /// ⛔ [`None`] IS BOTH OF THAT SITE'S FAILURES: addresses that differ, and an address nothing has
    /// been placed into at all — the reference throws on the first and indexes past the end on the
    /// second, and neither is an answer.
    #[must_use]
    pub fn uniform(&self) -> Option<Bytes> {
        let all = self.all();
        let (&first, rest) = all.split_first()?;
        rest.iter().all(|&addr| addr == first).then_some(first)
    }

    /// `getFuncType()` RE-LAID with `foldTypes.front() = Map` and the placed addresses copied over,
    /// which the reference spells as a `buildFoldSpace` followed by an `apply(copy)`
    /// (`L3DlOpsScheduler.cpp:5959-5967`) — an indirect address varies from core to core.
    ///
    /// ⛔ [`None`] IS `DT_CHECK(foldTypes.size() >= 2)`: a fold space without a corelet axis has no
    /// core axis to map either.
    #[must_use]
    pub fn with_mapped_core(&self) -> Option<Self> {
        (self.func_types.len() >= 2).then_some(())?;
        let mut mapped = self.clone();
        *mapped.func_types.first_mut()? = AddressFold::Map;
        Some(mapped)
    }

    /// `getDataAndFoldCoordinates()` FUSED WITH the `insertData` that rewrites each address it yields
    /// (`L3DlOpsScheduler.cpp:5892-5901`) — the ONE step, since a half-rewritten fold space is not a
    /// state the reference can be in.
    ///
    /// ⛔ [`None`] IS THE CLOSURE'S OWN REFUSAL. An address the caller means to leave alone is
    /// returned unchanged.
    pub fn map_addresses<F>(&mut self, mut rewrite: F) -> Option<()>
    where
        F: FnMut(Core, Corelet, Bytes) -> Option<Bytes>,
    {
        for (&core, per_cl) in &mut self.placed {
            for (&cl, addrs) in per_cl.iter_mut() {
                for addr in addrs.iter_mut() {
                    *addr = rewrite(core, cl, *addr)?;
                }
            }
        }
        Some(())
    }

    /// `apply({}, std::divides<int64_t>(), scale)` (`ddc/ddcv1.cpp:2426`) — every placed address at
    /// the unit's address granularity.
    #[must_use]
    pub fn divided_by(&self, scale: NonZeroU64) -> Self {
        Self {
            folds: self.folds.clone(),
            func_types: self.func_types.clone(),
            placed: self
                .placed
                .iter()
                .map(|(&core, per_cl)| {
                    (
                        core,
                        per_cl
                            .iter()
                            .map(|(&cl, addrs)| {
                                (
                                    cl,
                                    addrs
                                        .iter()
                                        .map(|addr| Bytes(addr.0 / scale.get()))
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

/// HOW MANY BUFFERS ONE ALLOCATION HOLDS — `numBuffers_` (`dsc/dsc2.h:984`), whose own comment names
/// the closed set: *"1:no buffering, 2:double-buffer, -1:streaming buffer"*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum NumBuffers {
    /// `1` — the default, and the value every `numBuffers_ != 1` test is asking about.
    #[default]
    Single,
    /// `2` — double buffered.
    Double,
    /// `-1` — a streaming buffer, which reserves the WHOLE memory rather than a size.
    Streaming,
}

impl NumBuffers {
    /// The divisor the reference computes a buffer offset with, INCLUDING its `if (numBuffers == -1)
    /// numBuffers = 2;  // reserve at least 2 buffers` (`ddc/ddcv1.cpp:239`).
    #[must_use]
    pub const fn reserved(self) -> NonZeroU64 {
        match self {
            Self::Single => NonZeroU64::new(1).expect("one is not zero"),
            Self::Double | Self::Streaming => NonZeroU64::new(2).expect("two is not zero"),
        }
    }

    /// `numBuffers_ != 1` — whether the allocation switches buffers at all.
    #[must_use]
    pub const fn switches(self) -> bool {
        !matches!(self, Self::Single)
    }

    /// `numBuffers_ == -1` — whether the whole memory is reserved for it.
    #[must_use]
    pub const fn is_streaming(self) -> bool {
        matches!(self, Self::Streaming)
    }
}

/// WHERE AN ALLOCATION LANDED AND HOW IT IS BUFFERED — the four `dsc2::AllocateNode` fields the
/// placement units read and write (`dsc/dsc2.h:981-988`), as ONE value.
///
/// ⭐ ONE FIELD RATHER THAN FOUR because they are filled TOGETHER: entry 258 writes the buffer offset
/// beside the start address, entry 259 reads the padding beside both, and every construction site of
/// a fresh allocate node wants all four at their defaults.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AllocPlacement {
    /// `numBuffers_` (`:984`).
    pub num_buffers: NumBuffers,
    /// `padding_` (`:981`).
    pub padding: PaddingForm,
    /// `bufferOffsetCoreCorelet_` (`:988`) — how far apart this allocation's buffers are.
    pub buffer_offset: BTreeMap<Core, BTreeMap<Corelet, Bytes>>,
    /// `isStartAddrSymbolic_` (`:987`).
    pub is_start_addr_symbolic: bool,
}

/// ONE END OF A TRANSFER AS ITS MINTING SITE IS HANDED IT — a [`DataLocation`] (`src_`, or a
/// `DstVia::loc_`) TOGETHER WITH the matching `..LdsAndLoopOffsets_.myLdsIdx_`.
///
/// ⭐ THE PAIRING IS THE POINT, AGAIN: `createTransferNode` is given the units, the storages and the
/// lds indices as three independently-sized vectors, and it fills `dstVias_` from the first two and
/// `dstLdsAndLoopOffsets_` from the third. An end that cannot arrive without its lds index is what
/// makes *"Destination unit and storage numbers do not match."* unspellable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Via {
    /// `src_` / `dstVias_.at(i).loc_`.
    pub loc: DataLocation,
    /// The matching `myLdsIdx_`, once its `-1` is an [`Option`].
    pub lds: Option<LdsIdx>,
}

impl Via {
    /// The operand this end becomes, with `dataConnect_` and `constantId_` EMPTY — the
    /// default-constructed `DataInfo` a freshly minted node carries (`dsc/dsc2.h:722`).
    #[must_use]
    pub fn operand(self) -> Operand {
        Operand {
            unit: self.loc.unit,
            storage: self.loc.storage,
            data: DataInfo {
                my_lds_idx: self.lds,
                ..DataInfo::EMPTY
            },
        }
    }

    /// The end an operand already states, which is what duplicating a transfer node copies out.
    #[must_use]
    pub const fn of(operand: &Operand) -> Self {
        Self {
            loc: DataLocation {
                unit: operand.unit,
                storage: operand.storage,
            },
            lds: operand.data.my_lds_idx,
        }
    }
}

/// `dsc2::AllocateNode` (`dsc/dsc2.h:974`) narrowed to what the ported units read and write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocateNode {
    /// `name_`.
    pub name: NodeName,
    /// `component_` (`:980`).
    pub component: SenComponent,
    /// `ldsIdx_` (`:976`), once its `-1` is an [`Option`].
    pub lds: Option<LdsIdx>,
    /// `constIdx_` (`:977`), once its `-1` is an [`Option`].
    pub const_idx: Option<ConstIdx>,
    /// `tempStorageForCompute_` (`:978`) — every reader takes its `name_`, so that is what is kept.
    pub temp_storage_for_compute: Option<NodeName>,
    /// `layoutDimOrder_` and `maxDimSizes_` AS ONE VALUE — see [`AllocLayout`].
    pub layout: AllocLayout,
    /// `startAddressCoreCorelet_` (`:985`).
    pub start_address: StartAddress,
    /// `numBuffers_`, `padding_`, `bufferOffsetCoreCorelet_` and `isStartAddrSymbolic_` AS ONE
    /// VALUE — see [`AllocPlacement`].
    pub placement: AllocPlacement,
    /// `gapStickSpread_` (`:1006`) — per layout dim, how many sticks of gap the data is spread over.
    pub gap_stick_spread: BTreeMap<PrimaryDim, Sticks>,
    /// `allocUsers_` (`:1007`), less the `int` beside each entry, which is `addAllocUser`'s REFERENCE
    /// COUNT (`:1015-1021`) and is not read by any unit ported so far.
    pub alloc_users: Vec<NodeId>,
}

/// HOW MANY TIMES AN OPAQUE OP'S BODY IS UNROLLED — `param_map_["unroll"]`, which the reference
/// keeps as the DECIMAL SPELLING of a power of two (`ddc/ddcv1.cpp:3352`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unroll(pub u32);

impl Unroll {
    /// The `1` an opaque op with no internal register allocation gets.
    pub const ONE: Self = Self(1);
}

/// AN OPAQUE OP'S ARITHMETIC PRECISION — `param_map_["prec"]`, whose two spellings are the whole of
/// the closed set (`ddc/ddcv1.cpp:3400-3403`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    /// `"fp32"` — `dataFormat_ == IEEE_FP32`.
    Fp32,
    /// `"fp16"` — every other format, including the unset one.
    Fp16,
}

/// WHICH PHYSICAL REGISTER — the `n` of the reference's `"R" + std::to_string(n)`
/// (`ddc/ddcv1.cpp:3358`), which is a stick index into the allocation's own memory and not a byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegSlot(pub u64);

impl RegSlot {
    /// `"R" + std::to_string(startAddress)` — the spelling the islands read back.
    #[must_use]
    pub fn spelling(self) -> String {
        format!("R{}", self.0)
    }
}

/// HOW MANY SLICES A COMPUTE INSTRUCTION REPEATS OVER — `InstrAttribute::repetition_`
/// (`dsc/dsc2.h:923`), whose default EIGHT is *"default 8 slices works the same"* and not zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Repetition(pub u32);

impl Repetition {
    /// `repetition_ = 8`.
    pub const ALL_SLICES: Self = Self(8);
}

impl Default for Repetition {
    /// A fresh `InstrAttribute` carries eight, and every reader of it wants that rather than none.
    fn default() -> Self {
        Self::ALL_SLICES
    }
}

/// ONE ENTRY OF A PACK/MERGE MAPPING — one `InstrAttribute::indices_` element (`dsc/dsc2.h:922`),
/// whose `-1` is the reference's own *"zero/sign extend"* marker and NOT a slice position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackIndex {
    /// `-1` — zero- or sign-extend rather than take a slice.
    Extend,
    /// A slice of the source stick.
    Slice(u32),
}

/// WHICH LANES OF A COMPUTE INSTRUCTION ARE LIVE — `InstrAttribute::compute_mask_`
/// (`dsc/dsc2.h:916`), whose initialiser 255 is *all eight* and not *none*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeMask(pub u32);

impl ComputeMask {
    /// `compute_mask_ = 255`.
    pub const ALL: Self = Self(255);
}

impl Default for ComputeMask {
    /// A fresh `InstrAttribute` states every lane, and every reader of it wants that rather than none.
    fn default() -> Self {
        Self::ALL
    }
}

/// AN OPAQUE OP'S BOUND INSTRUCTION STATE — `ComputeNode::instrAttribute_` (`dsc/dsc2.h:905-939`)
/// narrowed to the four things entry 261 writes, plus the three `ddl.compute` states entry 323 sets.
///
/// ⛔ NO CENSUSED `ddl.compute` STATES A `mask=`, so [`Self::compute_mask`] is invariably the
/// reference's own 255 initialiser — but entry 323 writes the slot and entry 325 reads it back out,
/// so the slot itself is not droppable.
///
/// ⭐ THE PARAM MAP IS TWO TYPED FIELDS AND NOT A `map<string, string>`: the reference stores an
/// unroll factor and a precision under fixed keys, and both are closed values.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InstrAttribute {
    /// `param_map_["unroll"]`.
    pub unroll: Unroll,
    /// `param_map_["prec"]`, absent until entry 261 decides it.
    pub precision: Option<Precision>,
    /// `read_write_reg_map_` — the internal registers.
    pub read_write_regs: BTreeMap<RegName, RegSlot>,
    /// `read_only_reg_map_` — the input/output registers.
    pub read_only_regs: BTreeMap<RegName, RegSlot>,
    /// Field: e015_ComputeNode.mode_
    ///
    /// `mode_` (`dsc/dsc2.h:915`) — the general SRC1/IMM field, once its `-1` default is an [`Option`].
    pub mode: Option<Mode>,
    /// `compute_mask_` (`dsc/dsc2.h:916`) — `mask=` where a template states one.
    pub compute_mask: ComputeMask,
    /// Field: e015_ComputeNode.repetition_
    ///
    /// `repetition_` (`dsc/dsc2.h:907`), *"default 8 slices works the same"*.
    pub repetition: Repetition,
    /// Field: e015_ComputeNode.indices_
    ///
    /// `indices_` (`dsc/dsc2.h:906`) — the PACK/MERGE mapping, empty where the op states none.
    pub indices: Vec<PackIndex>,
    /// `param_map_` MINUS its two fixed keys — a `ddl.opaque`'s `params=`, copied through
    /// (`ddc/ddl/ddl_conversion.cpp:2674-2690`).
    ///
    /// ⭐ NO OVERLAP WITH [`Self::unroll`] OR [`Self::precision`]: those two are `"unroll"` and
    /// `"prec"`, which entry 261 writes from the compute's own state, and no vendored template
    /// spells either as a `params=` key.
    pub params: BTreeMap<ParamKey, ParamValue>,
    /// Field: e015_ComputeNode.computeMaskLoopOffsets_
    ///
    /// Field: e015_ComputeNode.loopEleOffsets_
    ///
    /// `computeMaskLoopOffsets_` (`dsc/dsc2.h:923-925`), *"Per corelet"* — how many elements of each
    /// dim one trip of that loop steps the compute's MASK by. EMPTY is the state entry 242 tests
    /// (`ddc/ddcv1.cpp:1696-1698`), so an absent entry is static masking and not a missing value.
    ///
    /// ⛔ `loopEleOffsets_` (`:917-922`) IS THIS FIELD AND NOT [`DataInfo::loop_ele_offsets`]: the
    /// reference declares it COMMENTED OUT immediately above the live member, which carries its inner
    /// two maps under a corelet key. The `loopEleOffsets_` that IS live belongs to [`DataInfo`]
    /// (`:730-734`) — a different type's field of the same name.
    pub compute_mask_loop_offsets:
        BTreeMap<Corelet, BTreeMap<LoopId, BTreeMap<PrimaryDim, MaskLoopOffset>>>,
    /// `input_data_connects_` (`dsc/dsc2.h:927`) — which ports a spliced opaque body reads.
    pub input_data_connects: Vec<DataConnect>,
    /// `output_data_connects_` (`dsc/dsc2.h:929`).
    pub output_data_connects: Vec<DataConnect>,
}

impl Default for Unroll {
    /// `param_map_` carries no `"unroll"` key on a fresh node, and every reader of it wants one.
    fn default() -> Self {
        Self::ONE
    }
}

/// WHAT ONE CORELET SEES OF A COMPUTE'S OPERANDS — `ComputeNode::CoreletView`
/// (`dsc/dsc2.h:943-946`), one [`UnitView`] per input and one per output.
///
/// ⭐ ONE ENTRY PER OPERAND IN OPERAND ORDER: both lists are `push_back`ed while walking `inputs_`
/// and `outputs_` (`dsc/dsc2.cpp:3021-3031`), so position `i` here is operand `i` of
/// [`ComputeNode::inputs`] / [`ComputeNode::outputs`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ComputeCoreletView {
    /// Field: e015_ComputeNode.inputsLoopsAndSizes_
    ///
    /// `inputsLoopsAndSizes_` (`dsc/dsc2.h:944`).
    pub inputs_loops_and_sizes: Vec<UnitView>,
    /// Field: e015_ComputeNode.outputsLoopsAndSizes_
    ///
    /// `outputsLoopsAndSizes_` (`dsc/dsc2.h:945`).
    pub outputs_loops_and_sizes: Vec<UnitView>,
}

/// WHICH OF A CLONED COMPUTE'S OPERANDS TAKE A REPETITION OFFSET —
/// `ComputeNode::RepetitionWithOffset` (`dsc/dsc2.h:950-953`), whose two lists are BOTH declared
/// empty.
///
/// ⭐ EMPTY IS THE ANSWER "NONE DO", and it is the answer entry 108 acts on: it reads
/// `forOutputs_.size()` to decide how many clones to mint and then writes `forOutputs_.at(idx)`
/// (`ddc/ddc_transformation.cpp:1373`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RepetitionWithOffset {
    /// Field: e015_ComputeNode.forInputs_
    ///
    /// `forInputs_ = {}` (`dsc/dsc2.h:951`), in [`ComputeNode::inputs`] order.
    pub for_inputs: Vec<Repetition>,
    /// Field: e015_ComputeNode.forOutputs_
    ///
    /// `forOutputs_ = {}` (`dsc/dsc2.h:952`), in [`ComputeNode::outputs`] order.
    pub for_outputs: Vec<Repetition>,
}

/// Replaces: e015_ComputeNode
///
/// `dsc2::ComputeNode` (`dsc/dsc2.h:900-962`) carrying ALL FOURTEEN of its declared fields, plus the
/// `name_` this projection needs from the `ScheduleNode` base.
///
/// ⭐ AN OPERAND AND ITS OFFSETS ARE ONE VALUE: `inputs_`/`inputsLdsAndLoopOffsets_` and
/// `outputs_`/`outputsLdsAndLoopOffsets_` are four parallel vectors in the reference, and fusing each
/// pair into an [`Operand`] is what makes `dbgPrint`'s length mismatch unspellable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeNode {
    /// `name_`.
    pub name: NodeName,
    /// Field: e015_ComputeNode.type_
    ///
    /// `type_` (`dsc/dsc2.h:933`) — the dialect's whole recognised compute set, because the MACC
    /// resolution entry 323 performs names `IMA4`/`IMA8`/`FMA8`/`FMA4`, which no `computetype=`
    /// spells and the census therefore cannot express.
    ///
    /// ⛔ NO `ComputeOpType::COUNT` ARM. The reference's initialiser is that past-the-end sentinel
    /// (`:933`), so a node minted and never given an op is unspellable here rather than carrying a
    /// value no reader handles.
    pub op: DdlComputeType,
    /// Field: e015_ComputeNode.exUnit_
    ///
    /// `exUnit_` (`dsc/dsc2.h:932`).
    pub ex_unit: SenComponent,
    /// Field: e015_ComputeNode.inputs_
    ///
    /// Field: e015_ComputeNode.inputsLdsAndLoopOffsets_
    ///
    /// `inputs_` (`:935`) zipped with `inputsLdsAndLoopOffsets_` (`:937`).
    pub inputs: Vec<Operand>,
    /// Field: e015_ComputeNode.outputs_
    ///
    /// Field: e015_ComputeNode.outputsLdsAndLoopOffsets_
    ///
    /// `outputs_` (`:936`) zipped with `outputsLdsAndLoopOffsets_` (`:938`).
    pub outputs: Vec<Operand>,
    /// `numFoldsEngaged` (`dsc/dsc2.h:940`), whose default is ONE and not zero.
    pub num_folds_engaged: NumFolds,
    /// Field: e015_ComputeNode.dataFormat_
    ///
    /// `dataFormat_` (`dsc/dsc2.h:934`), whose reference initialiser is `SEN169_FP16` — so [`None`]
    /// is *"nobody chose one"* and [`ComputeNode::effective_data_format`] is what turns that back
    /// into the format the reference would have read.
    pub data_format: Option<DataFormat>,
    /// Field: e015_ComputeNode.instrAttribute_
    ///
    /// `instrAttribute_` (`dsc/dsc2.h:939`).
    pub instr_attribute: InstrAttribute,
    /// Field: e015_ComputeNode.isOpaqueOp_
    ///
    /// `isOpaqueOp_` (`dsc/dsc2.h:941`) — whether the compute is a `ddl.opaque` whose body is
    /// spliced in rather than a recognised op. FALSE on a fresh node; the DDL conversion sets it
    /// (`ddc/ddl/ddl_conversion.cpp:1575-1578`).
    pub is_opaque_op: bool,
    /// Field: e015_ComputeNode.coreletViews_
    ///
    /// `coreletViews_` (`dsc/dsc2.h:947`), EMPTY until the tree is finalised: `dsc/dsc2.cpp:3020`
    /// fills one entry per corelet in `0 .. numCoreletsUsed_DSC2_`, so no `-1` ever keys it.
    pub corelet_views: BTreeMap<Corelet, ComputeCoreletView>,
    /// Field: e015_ComputeNode.inputCoordinates_
    ///
    /// `inputCoordinates_` (`dsc/dsc2.h:948`), which may be SHORTER than an opaque op's input list.
    pub input_coordinates: Vec<Coordinate>,
    /// Field: e015_ComputeNode.outputCoordinate_
    ///
    /// `outputCoordinate_` (`dsc/dsc2.h:949`) — ONE coordinate however many outputs, which is the
    /// reference's own declaration and not a narrowing.
    pub output_coordinate: Coordinate,
    /// Field: e015_ComputeNode.repetitionWithOffset_
    ///
    /// `repetitionWithOffset_` (`dsc/dsc2.h:954`).
    pub repetition_with_offset: RepetitionWithOffset,
}

impl ComputeNode {
    /// The format this compute reads and writes, which a fresh node HAS rather than lacks:
    /// `dataFormat_`'s reference initialiser is `DataFormats::SEN169_FP16` (`dsc/dsc2.h:934`).
    #[must_use]
    pub fn effective_data_format(&self) -> DataFormat {
        self.data_format.unwrap_or(DataFormat::Sen169Fp16)
    }

    /// `coreletViews_.at(corelet_id_)`, or `coreletViews_.begin()->second` where the executing unit
    /// has no corelet of its own (`SNComputeLowering.cpp:549-571`) — [`None`] reads the FIRST entry,
    /// which is what the reference's `-1` does, and absent where nothing has been finalised yet.
    #[must_use]
    pub fn corelet_view(&self, corelet: Option<Corelet>) -> Option<&ComputeCoreletView> {
        match corelet {
            Some(corelet) => self.corelet_views.get(&corelet),
            None => self.corelet_views.values().next(),
        }
    }
}

/// A POSITION IN A STICK'S DIM ORDER — `srcSizeIdx_`/`dstSizeIdx_` (`dsc/dsc2.h:821`), an index into
/// the `getStickSizes` list and NOT a size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickDimIdx(pub u32);

/// ONE DIM AND HOW MUCH OF IT — `dsc2::Size` (`dsc/dsc2.h:486`), whose `int size_ = -1` default means
/// UNSET. Every size entry 126 pushes is a positive element count, so this one is not optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    /// Field: e009_ScheduleNode.dim_
    ///
    /// `dim_` (`dsc/dsc2.h:487`).
    pub dim: PrimaryDim,
    /// Field: e009_ScheduleNode.size_
    ///
    /// `size_` (`:488`).
    ///
    /// ⚠️ ONE LIST, TWO DENOMINATIONS in the `sizesNoGaps_` `buildUnitView` fills: the leading
    /// `getStickSizes` entries are element counts (`dsc/dsc2.cpp:2776-2780`), and each layout entry
    /// after them is divided by `getCumulativeStickSizes` (`:2803-2805`), so it counts STICKS of its
    /// own dim. [`Elements`] is exact for [`SizeAndIndex::size_dim`] and a conflation for that tail.
    pub size: Elements,
}

/// A POSITION IN A UNIT VIEW'S SIZE LIST — the `sizeIdx_` a [`LoopInfo`] holds (`dsc/dsc2.h:503`),
/// an index into [`UnitView::sizes_no_gaps`] and NOT a [`StickDimIdx`] into the stick order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ViewSizeIdx(pub usize);

/// A LOOP'S STEP IN ONE VIEW-SIZE ENTRY — the `elemOffset_` of a [`LoopInfo`] (`dsc/dsc2.h:504`),
/// COUNTED IN STEPS OF `sizes_no_gaps[size_idx]` AND NOT IN [`Elements`].
///
/// ⛔ `calculateSizeIdxAndOffset` DIVIDES. The raw element offset `DataInfo::loopEleOffsets_` holds
/// goes in and `offset / dimSizeSoFar` comes out, over the same-dim entries inner to `size_idx`
/// (`dsc/dsc2.cpp:2745`). The gap fix-up pins that denomination: rescaling `sizesNoGaps_[i].size_`
/// rescales every offset whose `sizeIdx_` is `i` by the same `gapStickSpread` (`:2885-2894`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ViewElemOffset(pub u32);

/// ONE ENCLOSING LOOP OF A UNIT VIEW — `ScheduleNode::UnitView::LoopInfo` (`dsc/dsc2.h:500-505`).
///
/// ⭐ `loop_` IS A NAME HERE. A view is STORED on its node (`TransferNode::srcLoopsAndSize_`,
/// `dsc/dsc2.h:848`; `ComputeNode::inputsLoopsAndSizes_`, `:944`), so its `const LoopNode*` crosses
/// ownership — and a name is the link a tree of owned nodes can hold, as [`SyncNode::other_ends`]
/// already does for a sync's other end.
///
/// ⛔ BOTH OF THOSE HOLD FOR THE VIEWS `buildUnitView` MINTS, WHOSE LOOPS ARE TREE ANCESTORS
/// (`dsc/dsc2.cpp:2848-2868`), AND NOT FOR THE C++ FIELD.
/// `constructImplicitLoopsForContiguousTransfer` builds `LoopInfo`s over `new LoopNode`s that are in
/// no tree and so carry no name, the outer one of them leaving `dim_` at `PrimaryDimTypesCount`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:733-737`, `:765-769`). Those reach
/// the bridge as `transfer::CompositeLoop` and `control_flow::SamvLoop`, which model both states;
/// this type is the tree's own list and models neither.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopInfo {
    /// Field: e009_ScheduleNode.loop_
    ///
    /// `loop_` (`:501`), by that [`LoopNode`]'s own `name_`.
    pub loop_name: NodeName,
    /// `dim_` (`:502`) — a dim of the enclosing loop, so never the unset default.
    pub dim: PrimaryDim,
    /// Field: e009_ScheduleNode.sizeIdx_
    ///
    /// `sizeIdx_` — [`None`] for the `-1` a loop unrelated to the lds dims gets
    /// (`dsc/dsc2.cpp:2872`).
    pub size_idx: Option<ViewSizeIdx>,
    /// Field: e009_ScheduleNode.elemOffset_
    ///
    /// `elemOffset_` — one iteration's step in [`ViewElemOffset`]'s own unit, [`None`] for the same
    /// `-1`. INDEPENDENT OF `size_idx`: the dummy outer loop at `SNTransferLowering.cpp:765-769`
    /// pairs `sizeIdx_ = -1` with `elemOffset_ = 1`, so the two are not one optional pair.
    pub elem_offset: Option<ViewElemOffset>,
}

/// WHAT ONE UNIT SEES OF A LABELLED DS — `ScheduleNode::UnitView` (`dsc/dsc2.h:499-511`): the sizes
/// a transfer or a compute steps through, and the loops that step them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnitView {
    /// Field: e009_ScheduleNode.sizesNoGaps_
    ///
    /// `sizesNoGaps_` — the stick dims first, then the layout dims (`dsc/dsc2.cpp:2775`, `:2791`).
    pub sizes_no_gaps: Vec<Size>,
    /// Field: e009_ScheduleNode.compositeLoops_
    ///
    /// `compositeLoops_` — the loops fusable into the unit's own stepping, innermost first.
    pub composite_loops: Vec<LoopInfo>,
    /// Field: e009_ScheduleNode.outerLoops_
    ///
    /// `outerLoops_` — the rest of the nest above those.
    pub outer_loops: Vec<LoopInfo>,
    /// Field: e009_ScheduleNode.sizesWithGaps_
    ///
    /// `sizesWithGaps_`, *"per core"*, whose `-1` key is the entry standing for every core.
    pub sizes_with_gaps: BTreeMap<Option<Core>, Vec<Size>>,
}

impl UnitView {
    /// `getSizesForCoreId(coreId)` (`dsc/dsc2.cpp:2398-2405`) — this core's gapped sizes, else the
    /// every-core entry, else the ungapped ones.
    #[must_use]
    pub fn sizes_for_core(&self, core: Option<Core>) -> &[Size] {
        self.sizes_with_gaps
            .get(&core)
            .or_else(|| self.sizes_with_gaps.get(&None))
            .unwrap_or(&self.sizes_no_gaps)
    }
}

/// ONE CHUNK OF A UNIT-TIME TRANSFER — `TransferNode::SizeAndIndex` (`dsc/dsc2.h:820`). Both of its
/// `-1` indices are filled with the same stick-dim position at the one site that pushes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeAndIndex {
    /// Field: e014_TransferNode.sizeDim_
    ///
    /// `sizeDim_` (`dsc/dsc2.h:821`).
    pub size_dim: Size,
    /// Field: e014_TransferNode.srcSizeIdx_
    ///
    /// `srcSizeIdx_` (`dsc/dsc2.h:822`).
    pub src_size_idx: StickDimIdx,
    /// Field: e014_TransferNode.dstSizeIdx_
    ///
    /// `dstSizeIdx_` (`dsc/dsc2.h:822`).
    pub dst_size_idx: StickDimIdx,
}

/// ONE POSITION OF A TRANSFER'S ZERO-PAD FOLD — the `(size, alpha, beta)` triple that ONE
/// `TransferPadInfo::FoldDimPosition` slot of `buildPadFrontSizes`' and `buildPadBackSizes`' three
/// parallel `std::vector<int>`s holds (`dsc/dsc2.h:777-782`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadFold {
    /// `sizes[pos]` — how many steps that fold axis walks.
    pub cardinality: FoldCardinality,
    /// `alphas[pos]`.
    pub alpha: FoldCoeff,
    /// `betas[pos]`.
    pub beta: FoldCoeff,
}

/// ONE DIM'S ZERO-PAD FOLD SPACE — both `TransferPadInfo::FoldDimPosition` slots,
/// `WORK_SLICE_FOLDDIM = 0` then `CHUNK_FOLDDIM = 1` (`dsc/dsc2.h:768-772`), which
/// `buildTransferFoldDim` folds *"from outer to inner"* in that order (`dsc/dsc2.cpp:4652-4654`).
///
/// ⭐ `TOTAL_FOLDDIM_NUM` IS THE COUNT, NOT A THIRD SLOT: it is `2`, it is the length
/// `buildTransferFoldDim` resizes each dim's `FoldDimProp` arena to (`dsc/dsc2.cpp:4664`), and it is
/// what `buildPadSizes` `DT_CHECK`s all three `std::vector<int>` lengths against (`:4612-4615`). A
/// pair carries the whole fold space, so that length check is what this shape makes unspellable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroPadFolds {
    /// `WORK_SLICE_FOLDDIM`.
    pub work_slice: PadFold,
    /// `CHUNK_FOLDDIM`.
    pub chunk: PadFold,
}

/// Replaces: e008_TransferPadInfo
///
/// A TRANSFER'S ZERO-PAD INFO — `TransferNode::paddingInfo_` (`dsc/dsc2.h:845`), a `TransferPadInfo`
/// (`:755`), as the two per-dim fold spaces its builders fill.
///
/// ⭐ FOUR OF THE SIX DECLARED MEMBERS ARE C++ POINTER PLUMBING WITH NOTHING TO CARRY.
/// `transferPadFrontSizeHelper`/`transferPadBackSizeHelper` (`:808-809`) are `MapWithFMHelper` VIEWS
/// the constructor binds to the two `*Size_` maps (`:757-759`), and the one thing read through either
/// is `getAllKeys()` (`:787`) — [`Self::dims`]. `transferPadFrontFoldProps`/`transferPadBackFoldProps`
/// (`:806-807`) are the `FoldDimProp` ARENAS those fold managers point into, which [`ZeroPadFolds`]
/// owns inline. That plumbing is why the reference's copy constructor *"Do[es] nothing on purpose"*
/// (`:761-764`) and its assignment is `= delete` (`:766`) — a hazard this type does not have.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransferPadding {
    /// Field: e008_TransferPadInfo.transferPadFrontSize_
    ///
    /// `transferPadFrontSize_` (`dsc/dsc2.h:810`), one `FoldManager<int>` per dim.
    front: BTreeMap<PrimaryDim, ZeroPadFolds>,
    /// Field: e008_TransferPadInfo.transferPadBackSize_
    ///
    /// `transferPadBackSize_` (`dsc/dsc2.h:811`).
    back: BTreeMap<PrimaryDim, ZeroPadFolds>,
}

impl TransferPadding {
    /// `buildPadFrontSizes(dim, sizes, alphas, betas)` (`dsc/dsc2.h:777-779`), the three parallel
    /// vectors carried as the one [`ZeroPadFolds`] their positions index.
    ///
    /// ⚠️ A SECOND BUILD OF ONE DIM REPLACES THE FIRST WHERE THE REFERENCE ABORTS:
    /// `buildTransferFoldDim` `DT_CHECK`s `!foldProps.count(dim)`, *"Expect empty fold properties."*
    /// (`dsc/dsc2.cpp:4662`). Entry 221 reaches each padded dim exactly once, from a per-dim loop
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5380`), so no reachable path distinguishes the two
    /// and refusing here would be a runtime abort standing in for an unreachable state.
    pub fn build_pad_front(&mut self, dim: PrimaryDim, folds: ZeroPadFolds) {
        self.front.insert(dim, folds);
    }

    /// `buildPadBackSizes(dim, sizes, alphas, betas)` (`dsc/dsc2.h:780-782`), under the same
    /// single-build contract as [`Self::build_pad_front`].
    pub fn build_pad_back(&mut self, dim: PrimaryDim, folds: ZeroPadFolds) {
        self.back.insert(dim, folds);
    }

    /// The front fold space of one dim, absent where nothing built it.
    #[must_use]
    pub fn pad_front(&self, dim: PrimaryDim) -> Option<ZeroPadFolds> {
        self.front.get(&dim).copied()
    }

    /// The back fold space of one dim, absent where nothing built it.
    #[must_use]
    pub fn pad_back(&self, dim: PrimaryDim) -> Option<ZeroPadFolds> {
        self.back.get(&dim).copied()
    }

    /// `getPadFrontOrBackDimsSet` (`dsc/dsc2.h:783-788`) with both ends unioned — each dim ONCE and
    /// in dim order, as the `std::set<PrimaryDimTypes>` its `getAllKeys()` returns.
    pub fn dims(&self) -> impl Iterator<Item = PrimaryDim> + '_ {
        self.front
            .keys()
            .chain(self.back.keys())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    /// `isEmpty()` (`dsc/dsc2.h:774-776`) — neither end has been handed any dim.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.front.is_empty() && self.back.is_empty()
    }
}

/// HOW MANY TIMES ONE UNIT-TIME TRANSFER REPEATS — `replicationFactor_` (`dsc/dsc2.h:834`), whose
/// default is ONE and not zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReplicationFactor(pub u64);

impl ReplicationFactor {
    /// `replicationFactor_ = 1` — no replication.
    pub const ONE: Self = Self(1);
}

/// HOW MANY UNIT-TIME CHUNKS ONE TRANSFER MOVES — `unitTimeTransferNumChunks_` (`dsc/dsc2.h:835`),
/// whose default is ONE and not zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NumChunks(pub u32);

impl NumChunks {
    /// `unitTimeTransferNumChunks_ = 1` — one chunk per unit time.
    pub const ONE: Self = Self(1);
}

/// HOW MUCH OF A LAYOUT DIM ONE LDS ACTUALLY HOLDS — `LabeledDsInfo::scale_`
/// (`dsc/dscdefn.h:332`), one entry per layout dim, as the arms the ported code distinguishes.
///
/// ⛔ THE REFERENCE FIELD IS A `double` AND IS NOT A COUNT: `dsc/dsc_standalone.cpp:377` pushes
/// `1 / kij`. Nothing in this campaign reads its magnitude — entry 259 tests `== 1`
/// (`ddc/ddcv1.cpp:1899`), entry 260 tests `> 0` (`:2452`, `:2629`) and entry 307 tests `== -2`
/// (`:1058`, `:1505`) — so a fractional scale cannot silently truncate into an arm below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdsScale {
    /// `scale_[i] == -2` — the dim is broadcast ALONG THE STICK, which entry 307 replicates over
    /// rather than skips.
    StickBroadcast,
    /// `scale_[i] <= 0` otherwise — this lds broadcasts the dim, so it spans nothing of it.
    Broadcast,
    /// `0 < scale_[i] != 1` — the dim is present at a scale that is not one.
    Scaled,
    /// `scale_[i] == 1` — the dim is present, unscaled.
    Unscaled,
}

impl LdsScale {
    /// `scale_[i] > 0` INVERTED — the reference's own test, spelled as the question it answers.
    ///
    /// ⛔ `-2` IS ALSO NOT `> 0`, so the stick-broadcast arm answers TRUE here: it is a spelling of
    /// which broadcast it is, never a fourth non-broadcast state.
    #[must_use]
    pub const fn is_broadcast(self) -> bool {
        matches!(self, Self::Broadcast | Self::StickBroadcast)
    }
}

/// WHICH GROUP TAG REGISTER ONE CORE'S END OF A MULTICAST TRANSFER USES — `dsc2::GroupTagRegInfo`
/// (`dsc/dsc2.h:34`), whose two `-1` defaults are "unshared" and "unfilled" and NOT counts.
///
/// ⭐ `groupId_ = -1` IS NOT A GROUP: entries 218 and 291 both write the id ONLY when
/// `numSharers_ > 1`, so the sentinel is [`None`] here and an unshared end cannot name a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupTagRegInfo {
    /// `numSharers_` — how many cores share this end. Never the `-1` default: a filled entry
    /// carries a real count, and an unfilled one is an absent map entry.
    pub num_sharers: Shares,
    /// `groupId_`, with the reference's `-1` as [`None`].
    pub group: Option<GtrGroupId>,
}

/// HOW MANY TIMES ONE END OF A TRANSFER READS ITS ALLOCATION — `repetition_.srcRep_` or one
/// `dstReps_` entry (`dsc/dsc2.h:827-828`), which is a `ddl.allocate`'s `replication=`.
///
/// ⛔ NOT [`Repetition`], WHOSE DEFAULT IS EIGHT. This one's only writer hands back `1` at all four
/// of its early exits and where the allocation states no `replication=`
/// (`ddc/ddl/ddl_conversion.cpp:858-870`) — and `srcRep_` is declared with NO initialiser at all, so
/// ONE is the value that cannot be the reference's uninitialised read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperandRepetition(pub u32);

impl OperandRepetition {
    /// `getRepetitionIfExists`' own `return 1`.
    pub const ONE: Self = Self(1);
}

impl Default for OperandRepetition {
    /// The writer's fallback, and the only value a reader of an unwritten `srcRep_` could soundly
    /// see.
    fn default() -> Self {
        Self::ONE
    }
}

/// HOW OFTEN EACH END OF A TRANSFER REPEATS — the anonymous `repetition_` struct
/// (`dsc/dsc2.h:826-829`), one factor for the source and one per destination.
///
/// ⛔ WRITTEN AND NEVER READ AT `a0d29abbed`: `ddc/ddl/ddl_conversion.cpp:1171` and `:1189` fill it,
/// no reader anywhere in the reference tree consults it, and the schedule serialiser does not emit
/// it. It is carried because dropping a declared field is not this port's call to make.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransferRepetition {
    /// Field: e014_TransferNode.srcRep_
    ///
    /// `srcRep_` (`dsc/dsc2.h:827`).
    pub src: OperandRepetition,
    /// Field: e014_TransferNode.dstReps_
    ///
    /// `dstReps_` (`dsc/dsc2.h:828`), in [`Dsts`] order.
    pub dsts: Vec<OperandRepetition>,
}

/// WHAT ONE CORELET SEES OF A TRANSFER'S ENDS — `TransferNode::CoreletView` (`dsc/dsc2.h:847-850`).
///
/// ⭐ AN INDIRECT VIEW IS EMPTY UNLESS THAT END IS INDIRECT: `dsc/dsc2.cpp:3066` and `:3075` guard
/// both fills with `isSrcIndirect()` / `isDstIndirect()`, so a default [`UnitView`] here says
/// *"direct"* rather than *"not built yet"*.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransferCoreletView {
    /// `srcLoopsAndSize_` (`dsc/dsc2.h:848`).
    pub src_loops_and_size: UnitView,
    /// Field: e014_TransferNode.srcIndirectLoopsAndSize_
    ///
    /// `srcIndirectLoopsAndSize_` (`dsc/dsc2.h:848`).
    pub src_indirect_loops_and_size: UnitView,
    /// `dstLoopsAndSizes_` (`dsc/dsc2.h:849`), in [`Dsts`] order.
    pub dst_loops_and_sizes: Vec<UnitView>,
    /// Field: e014_TransferNode.dstIndirectLoopsAndSizes_
    ///
    /// `dstIndirectLoopsAndSizes_` (`dsc/dsc2.h:849`), pushed only for an indirect destination.
    pub dst_indirect_loops_and_sizes: Vec<UnitView>,
}

/// Replaces: e014_TransferNode
///
/// `dsc2::TransferNode` (`dsc/dsc2.h:814-898`) carrying ALL TWENTY of its declared members, plus the
/// `name_` this projection needs from the `ScheduleNode` base.
///
/// ⭐ AN END AND ITS OFFSETS ARE ONE VALUE: `src_`/`srcLdsAndLoopOffsets_` and
/// `dstVias_`/`dstLdsAndLoopOffsets_` are parallel in the reference, and `ddc/ddcv1.cpp:2867` is the
/// `DT_ERROR` that fires when the two destination vectors disagree — fusing each pair into an
/// [`Operand`] is what makes that error unspellable.
///
/// ⛔ AND AN INDIRECT END IS A WHOLE `DataInfo` AND NOT AN LDS INDEX. `fillDataInfo` writes
/// `startAddr_` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5879-5880`, `:5986`),
/// `isStartAddrSymbolic_` (`:5916`) and `loopEleOffsets_` (`:6182`) into
/// `srcIndirectLdsAndLoopOffsets_`/`dstIndirectLdsAndLoopOffsets_`, and both readers take the whole
/// record: `buildUnitView(tn->srcIndirect_, tn->srcIndirectLdsAndLoopOffsets_, ..)`
/// (`dsc/dsc2.cpp:3067-3069`, `:3076-3079`) and `createTransferLocInfo(.., srcIndDtInfo, ..)`
/// (`dsc/dsc2Pcfg.cpp:1182-1186`, `:1190-1195`). So each indirect end is an [`Operand`] like the
/// direct one, and only its `myLdsIdx_` being carried would leave those three writes nowhere to land.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferNode {
    /// `name_`.
    pub name: NodeName,
    /// `src_.unit_` zipped with `srcLdsAndLoopOffsets_` (`dsc/dsc2.h:824`, `:832`).
    pub src: Operand,
    /// Field: e014_TransferNode.dstVias_
    ///
    /// `dstVias_` (`dsc/dsc2.h:825`) zipped with `dstLdsAndLoopOffsets_` (`:833`).
    pub dsts: Dsts,
    /// Field: e014_TransferNode.replicationFactor_
    ///
    /// `replicationFactor_` (`dsc/dsc2.h:834`).
    pub replication_factor: ReplicationFactor,
    /// Field: e014_TransferNode.unitTimeTransferChunkSize_
    ///
    /// `unitTimeTransferChunkSize_` (`dsc/dsc2.h:836`) — the continuous elements within a stick
    /// boundary.
    pub unit_time_transfer_chunk_size: Vec<SizeAndIndex>,
    /// Field: e014_TransferNode.unitTimeTransferNumChunks_
    ///
    /// `unitTimeTransferNumChunks_` (`dsc/dsc2.h:837`) — how many of those chunks one unit time
    /// moves.
    pub unit_time_transfer_num_chunks: NumChunks,
    /// Field: e014_TransferNode.unitTimeTransferChunkStride_
    ///
    /// `unitTimeTransferChunkStride_` (`dsc/dsc2.h:838`) — the dims a chunk-strided load skips over,
    /// EMPTY for a contiguous one.
    ///
    /// ⛔ A `Vec` AND NOT ONE ENTRY, BECAUSE THE REFERENCE DOES NOT GUARANTEE ONE: its reader
    /// `DT_CHECK`s `size() <= 1` (`dsc/dsc2.cpp:3555`) while the writer `push_back`s once per
    /// remaining dim (`ddc/ddcv1.cpp:1636-1641`). Narrowing it to a single entry would assert a
    /// contract the writer does not hold to. ⭐ That writer is a lambda whose only call site is
    /// COMMENTED OUT (`ddc/ddcv1.cpp:1649-1650`), so on this path only a deserialised schedule
    /// (`dsc/dsc2.cpp:1573`) fills it at all.
    pub unit_time_transfer_chunk_stride: Vec<SizeAndIndex>,
    /// Field: e014_TransferNode.rotateNumElements_
    ///
    /// `rotateNumElements_` (`dsc/dsc2.h:839`) — how far the LXLU rotates what it moves, absent for
    /// the reference's `0`.
    ///
    /// ⭐ ZERO AND ABSENT ARE THE SAME STATE HERE, not a lost distinction: every reader guards on
    /// `rotateNumElements_ > 0` (`SNTransferLowering.cpp:991`, `:1097`, `:2239`, `:2277`) and the one
    /// consumer that reads it unguarded substitutes `0` for a non-LXLU unit (`dsc/dsc2Pcfg.cpp:1233`).
    pub rotate_num_elements: Option<Elements>,
    /// Field: e014_TransferNode.paddingInfo_
    ///
    /// `paddingInfo_` (`dsc/dsc2.h:845`) — EMPTY on a fresh node; entry 221 is what fills it.
    pub padding: TransferPadding,
    /// Field: e014_TransferNode.repetition_
    ///
    /// `repetition_` (`dsc/dsc2.h:826-829`).
    pub repetition: TransferRepetition,
    /// Field: e014_TransferNode.srcIndirect_
    ///
    /// Field: e014_TransferNode.srcIndirectLdsAndLoopOffsets_
    ///
    /// `srcIndirect_` (`dsc/dsc2.h:824`) FUSED WITH THE WHOLE `srcIndirectLdsAndLoopOffsets_` (`:832`)
    /// — the index tensor this transfer gathers its SOURCE addresses through, [`None`] for a direct
    /// transfer, which is `isSrcIndirect()`'s own `unit_ != NO_COMPONENT` test (`:877`).
    pub src_indirect: Option<Operand>,
    /// Field: e014_TransferNode.locIndirect_
    ///
    /// Field: e014_TransferNode.dstIndirectLdsAndLoopOffsets_
    ///
    /// `dstVias_.front().locIndirect_` (`dsc/dsc2.h:817`) FUSED WITH THE WHOLE
    /// `dstIndirectLdsAndLoopOffsets_.front()` (`:833`).
    ///
    /// ⛔ ONE END AND NOT A VECTOR, WHICH IS THE REFERENCE'S OWN THREE `DT_CHECK`s: entry 227 demands
    /// `dstVias_.size() == 1` before it writes (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7138`), and
    /// that `dstIndirectLdsAndLoopOffsets_` was empty (`:7143`) and holds exactly one entry after
    /// (`:7146`).
    pub dst_indirect: Option<Operand>,
    /// Field: e014_TransferNode.lastFusableParentLoopSrc_
    ///
    /// `lastFusableParentLoopSrc_` (`dsc/dsc2.h:830`) — the outermost enclosing loop the SOURCE end
    /// can fuse into its own stepping, absent for the reference's `nullptr`, which is the state
    /// `compLoop = lastFusableParentLoop != nullptr` reads (`dsc/dsc2.cpp:2843`).
    pub last_fusable_parent_loop_src: Option<LoopId>,
    /// Field: e014_TransferNode.lastFusableParentLoopDst_
    ///
    /// `lastFusableParentLoopDst_` (`dsc/dsc2.h:831`), ONE ENTRY PER DESTINATION in [`Dsts`] order —
    /// `ddc/ddcv1.cpp:2871-2878` clears it and pushes once per `dstVias_` entry, and `dsc/dsc2.cpp
    /// :3074` indexes it by the same `i`. Each entry is absent for its own `nullptr`.
    pub last_fusable_parent_loop_dst: Vec<Option<LoopId>>,
    /// Field: e014_TransferNode.coreIdToGTRInfo_
    ///
    /// `coreIdToGTRInfo_` (`dsc/dsc2.h:840`) — the multicast group each transferring core belongs to,
    /// EMPTY on a fresh node. Entry 291 is what fills it and entry 218 is what fills a condition
    /// arm's.
    pub core_id_to_gtr_info: BTreeMap<Core, GroupTagRegInfo>,
    /// Field: e014_TransferNode.transferSize_
    ///
    /// `transferSize_` (`dsc/dsc2.h:843`) — an EXPLICIT per-dim size that overrides the one derived
    /// from the data stage (`dsc/dsc2.cpp:3474` reads it), EMPTY on a fresh node. Entry 295 fills it.
    pub transfer_size: BTreeMap<PrimaryDim, Elements>,
    /// Field: e014_TransferNode.coreletViews_
    ///
    /// `coreletViews_` (`dsc/dsc2.h:851`), EMPTY until the tree is finalised: `dsc/dsc2.cpp:3062`
    /// fills one entry per corelet in `0 .. numCoreletsUsed_DSC2_`, so no `-1` ever keys it.
    pub corelet_views: BTreeMap<Corelet, TransferCoreletView>,
    /// Field: e014_TransferNode.transferCoordinates_
    ///
    /// `transferCoordinates_` (`dsc/dsc2.h:852`) — ONE coordinate shared by every end of the
    /// transfer, which `fillDataInfo` threads through both the source and each destination
    /// (`ddc/ddcv1.cpp:2863-2875`).
    pub transfer_coordinates: Coordinate,
}

/// WHAT A TRANSFER MOVES BETWEEN — `dsc2::TransferNode::getTransferType()` (`dsc/dsc2.h:882-900`),
/// decided entirely by whether each end names a labeled ds (`myLdsIdx_ >= 0`) or a constant
/// (`constantId_ >= 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransferKind {
    /// `CONSTANT_TO_CONSTANT`.
    ConstantToConstant,
    /// `CONSTANT_TO_TENSOR`.
    ConstantToTensor,
    /// `TENSOR_TO_TENSOR`.
    TensorToTensor,
    /// `NO_TRANSFER_TO_TENSOR` — the destination is a tensor and the source is neither.
    NoTransferToTensor,
    /// `NO_TRANSFER_FROM_TENSOR` — the source is a tensor and the destination is neither.
    NoTransferFromTensor,
    /// `INVALID_TRANSFER_TYPE` — neither end names a tensor.
    Invalid,
}

impl DataInfo {
    /// A DEFAULT-CONSTRUCTED `DataInfo` (`dsc/dsc2.h:721-739`) — every `-1` and the one `nullptr`
    /// absent, every map empty, and the start address a fold space nothing is laid out in.
    pub const EMPTY: Self = Self {
        data_connect: None,
        my_lds_idx: None,
        constant_id: None,
        latch_data_id: None,
        start_addr: StartAddress::EMPTY,
        is_start_addr_symbolic: false,
        const_ele_offsets: BTreeMap::new(),
        loop_ele_offsets: BTreeMap::new(),
        buffer_addr_offset: BTreeMap::new(),
        buffer_switch_position: None,
    };

    /// `isLabeledDs()` (`dsc/dsc2.h:741`) — `myLdsIdx_ >= 0`.
    ///
    /// ⛔ THE REFERENCE'S `DT_CHECK_MSG(!(labeledDs && constant), "Cannot be both labeledDs and
    /// constant.")` IS UNSPELLABLE HERE and needs no runtime guard: `my_lds_idx` and `constant_id`
    /// are separate [`Option`]s, so a node carrying both answers `true` here exactly as the
    /// reference's field test would — the reference stops, we do not, and no caller distinguishes.
    #[must_use]
    pub const fn is_labeled_ds(&self) -> bool {
        self.my_lds_idx.is_some()
    }

    /// `isConstant()` (`dsc/dsc2.h:746`) — `constantId_ >= 0`.
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        self.constant_id.is_some()
    }
}

impl TransferNode {
    /// `getTransferType()` (`dsc/dsc2.h:882`).
    ///
    /// ⛔ THE DESTINATION IS `dstLdsAndLoopOffsets_.front()` AND [`Dsts`] IS NON-EMPTY, so the
    /// reference's `!dstLdsAndLoopOffsets_.empty() &&` guard is discharged by the type.
    #[must_use]
    pub const fn transfer_kind(&self) -> TransferKind {
        let src = &self.src.data;
        let dst = &self.dsts.first().data;
        match (
            src.is_labeled_ds(),
            src.is_constant(),
            dst.is_labeled_ds(),
            dst.is_constant(),
        ) {
            (_, true, _, true) => TransferKind::ConstantToConstant,
            (_, true, true, _) => TransferKind::ConstantToTensor,
            (true, _, true, _) => TransferKind::TensorToTensor,
            (false, false, true, _) => TransferKind::NoTransferToTensor,
            (true, _, false, false) => TransferKind::NoTransferFromTensor,
            _ => TransferKind::Invalid,
        }
    }

    /// `coreletViews_.at(corelet_id_)`, or `coreletViews_.begin()->second` where the transferring
    /// unit has no corelet of its own (`SNTransferLowering.cpp:861-863`) — [`None`] reads the FIRST
    /// entry, which is what the reference's `-1` does, and absent where nothing has been finalised.
    #[must_use]
    pub fn corelet_view(&self, corelet: Option<Corelet>) -> Option<&TransferCoreletView> {
        match corelet {
            Some(corelet) => self.corelet_views.get(&corelet),
            None => self.corelet_views.values().next(),
        }
    }
}

/// AN ELEMENT WIDTH IN BYTES — `labeledDs_::wordLength_` (`dsc/dsc2.h:296`).
///
/// ⛔ A WIDTH AND NOT A COUNT: entry 308 writes `2` for a BFLOAT16 internal kernel, and an element
/// count of two would be a different tensor.
///
/// ⛔ [`Default`] IS THE FIELD'S OWN `= 0` (`dsc/dscdefn.h:334`) — *"nobody stated a width"*, not a
/// zero-byte element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct WordLength(pub u32);

/// WHICH KIND OF NODE A TREE POSITION IS — `ScheduleNode::NodeType` (`dsc/dsc2.h:446-456`), the set
/// `traverseTreeDFS`'s `nodeTypes` filter is drawn from.
///
/// ⛔ NO `INVALID` ARM. It is the base class's own `nodeType_ = INVALID` default (`:460`), and every
/// subclass hands its own kind to the constructor at `:481`, so no node in a tree carries it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeType {
    /// `BLOCK`.
    Block,
    /// `LOOP`.
    Loop,
    /// `TRANSFER`.
    Transfer,
    /// `COMPUTE`.
    Compute,
    /// `SYNC`.
    Sync,
    /// `CONDITION`.
    Condition,
    /// `ALLOCATE`.
    Allocate,
    /// `STICKMASK`.
    StickMask,
}

impl NodeType {
    /// `isBlockNode()` (`dsc/dsc2.h:479`) — whether a traversal descends into a node of this kind.
    #[must_use]
    pub const fn is_block_node(self) -> bool {
        matches!(self, Self::Block | Self::Loop | Self::Condition)
    }
}

/// WHICH OF THE THREE CHILDLESS KINDS A [`SchedNode::Leaf`] IS, so that a node the tree holds only
/// as a position still answers `nodeType_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LeafKind {
    /// `ALLOCATE`.
    Allocate,
    /// `COMPUTE`.
    Compute,
    /// `TRANSFER`.
    Transfer,
}

impl LeafKind {
    /// The `nodeType_` a leaf of this kind carries.
    #[must_use]
    pub const fn node_type(self) -> NodeType {
        match self {
            Self::Allocate => NodeType::Allocate,
            Self::Compute => NodeType::Compute,
            Self::Transfer => NodeType::Transfer,
        }
    }
}

/// WHAT `isNodeRelevant` IS ASKED — the four legal shapes of its `(comp, clId, coreId)`
/// (`dsc/dsc2.cpp:1916-1932`). Its two `DT_ERROR`s are the shapes this enum cannot spell: a corelet
/// with no core, and a core or corelet filter under `comp == ALL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relevance {
    /// `(ALL, -1, -1)`.
    Any,
    /// `(comp, -1, -1)`.
    Comp(SenComponent),
    /// `(comp, -1, coreId)`.
    CompCore(SenComponent, Core),
    /// `(comp, clId, coreId)`.
    CompCoreCl(SenComponent, Core, Corelet),
}

/// HOW FAR A COMPONENT QUERY IS NARROWED — the `(coreId, clId)` `getRelevantComps` takes
/// (`dsc/dsc2.cpp:1949-1975`), whose one `DT_ERROR` is again a corelet with no core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreSel {
    /// `(-1, -1)`.
    Any,
    /// `(coreId, -1)`.
    Core(Core),
    /// `(coreId, clId)`.
    Corelet(Core, Corelet),
}

/// WHICH CORELETS OF WHICH CORES OF WHICH COMPONENTS A NODE IS FOR — `relevantComps_`
/// (`dsc/dsc2.h:516`), the filter `getNextView` selects a node by.
///
/// ⛔ A COMPONENT PRESENT WITH AN EMPTY CORE MAP IS NOT AN ABSENT ONE: [`Self::relevant_comps`]
/// skips it (`dsc/dsc2.cpp:1963`, `:1972`) while [`Self::is_node_relevant`] accepts it as long as no
/// core is named (`:1923-1926`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RelevantComps(BTreeMap<SenComponent, BTreeMap<Core, BTreeSet<Corelet>>>);

impl RelevantComps {
    /// `relevantComps_[comp][coreId].insert(clId)` — the write a sequence makes when it decides a
    /// node belongs to a unit (`dsc/dsc2.cpp:2692`).
    pub fn mark(&mut self, comp: SenComponent, core: Core, corelet: Corelet) {
        self.0
            .entry(comp)
            .or_default()
            .entry(core)
            .or_default()
            .insert(corelet);
    }

    /// `relevantComps_.erase(comp)` (`dsc/dsc2.cpp:2977-2980`) — dropping `NO_COMPONENT` once the
    /// per-component marking it seeded is in place.
    pub fn forget(&mut self, comp: SenComponent) {
        self.0.remove(&comp);
    }

    /// `relevantComps_.empty()`, which `DT_CHECK(!..)` reads on the head (`ddc/ddcv1.cpp:3458`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `isNodeRelevant(comp, clId, coreId)` (`dsc/dsc2.cpp:1916-1932`).
    #[must_use]
    pub fn is_node_relevant(&self, of: Relevance) -> bool {
        let (comp, core, corelet) = match of {
            Relevance::Any => return true,
            Relevance::Comp(comp) => (comp, None, None),
            Relevance::CompCore(comp, core) => (comp, Some(core), None),
            Relevance::CompCoreCl(comp, core, corelet) => (comp, Some(core), Some(corelet)),
        };
        let Some(cores) = self.0.get(&comp) else {
            return false;
        };
        let Some(core) = core else {
            return true;
        };
        let Some(corelets) = cores.get(&core) else {
            return false;
        };
        corelet.is_none_or(|corelet| corelets.contains(&corelet))
    }

    /// `getRelevantCoreCl(comp)` (`dsc/dsc2.cpp:1934-1947`) — [`None`] is its `ALL` default, which
    /// unions every component's cores; a core with no corelets contributes nothing either way.
    #[must_use]
    pub fn relevant_core_cl(
        &self,
        comp: Option<SenComponent>,
    ) -> BTreeMap<Core, BTreeSet<Corelet>> {
        let mut all: BTreeMap<Core, BTreeSet<Corelet>> = BTreeMap::new();
        for (relevant, cores) in &self.0 {
            if comp.is_some_and(|comp| comp != *relevant) {
                continue;
            }
            for (core, corelets) in cores {
                if !corelets.is_empty() {
                    all.entry(*core).or_default().extend(corelets.iter().copied());
                }
            }
        }
        all
    }

    /// `getRelevantComps(coreId, clId)` (`dsc/dsc2.cpp:1949-1975`) — which components reach the core
    /// and corelet `at` names.
    #[must_use]
    pub fn relevant_comps(&self, at: CoreSel) -> BTreeSet<SenComponent> {
        let mut all = BTreeSet::new();
        for (comp, cores) in &self.0 {
            let reaches = match at {
                CoreSel::Any => !cores.is_empty(),
                CoreSel::Core(core) => cores.get(&core).is_some_and(|cls| !cls.is_empty()),
                CoreSel::Corelet(core, corelet) => {
                    cores.get(&core).is_some_and(|cls| cls.contains(&corelet))
                }
            };
            if reaches {
                all.insert(*comp);
            }
        }
        all
    }
}

/// Replaces: e009_ScheduleNode
///
/// WHAT EVERY SCHEDULE-TREE NODE CARRIES — `dsc2::ScheduleNode`'s own state (`dsc/dsc2.h:444-517`),
/// held BY COMPOSITION on each node kind exactly as [`LoopNode`] holds its [`BlockNode`] part.
///
/// ⭐ FOUR DECLARED FIELDS, TWO OF THEM HERE. `nodeType_` (`:460`) IS the [`SchedNode`] discriminant,
/// spelled back by [`SchedNode::node_type`]; `prev_` (`:515`) is a parent POINTER, which an owned
/// tree answers by walking DOWN instead ([`ScheduleTree::ancestors`]). Its two nested structs are
/// types rather than fields: [`Size`] and [`UnitView`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodeBase {
    /// Field: e009_ScheduleNode.name_
    ///
    /// `name_` (`dsc/dsc2.h:461`).
    pub name: NodeName,
    /// Field: e009_ScheduleNode.relevantComps_
    ///
    /// `relevantComps_` (`:516`), empty until a sequence marks the node.
    pub relevant_comps: RelevantComps,
}

impl NodeBase {
    /// A node known only by its name, which is every mint site — `relevantComps_` is filled by a
    /// later pass over the tree (`dsc/dsc2.cpp:2656-2692`).
    #[must_use]
    pub const fn named(name: NodeName) -> Self {
        Self {
            name,
            relevant_comps: RelevantComps(BTreeMap::new()),
        }
    }
}

/// AN ALLOCATE, COMPUTE OR TRANSFER NODE'S TREE POSITION — `isBlockNode()` is false and it has no
/// children, so a walk neither yields nor descends into it; its payload lives with its own unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafNode {
    /// The `ScheduleNode` part.
    pub base: NodeBase,
    /// Which of the three kinds it is.
    pub kind: LeafKind,
}

impl LeafNode {
    /// A named leaf of kind `kind`.
    #[must_use]
    pub const fn new(kind: LeafKind, name: NodeName) -> Self {
        Self {
            base: NodeBase::named(name),
            kind,
        }
    }
}

/// Replaces: e010_BlockNode
///
/// `dsc2::BlockNode` (`dsc/dsc2.h:526-561`) carrying its ONE declared field, plus the `name_` a block
/// is looked up by.
///
/// ⛔ A FRESH BLOCK HAS NO CHILDREN: `new dsc2::BlockNode()` leaves the child vector empty, and
/// `addChildNode`/`moveChildNode` are the units that fill it.
///
/// ⛔⛔ `Clone` IS A DEEP COPY AND IS NOT THE REFERENCE'S `clone()`, WHICH IS CHILDLESS.
/// `VectorOfChildren(const VectorOfChildren&) {}` is *"do nothing on purpose"*, leaving it *"up to the
/// caller to manually insert copies of the children"* (`dsc/dsc2.h:533-535`), and every Block-derived
/// `clone()` is `new Derived(*this)` (`util/utils.h:105-107`), so it runs that copy constructor.
///
/// ⛔ AND THAT COPY IS LIVE, TWICE, BOTH TIMES RE-INSERTING THE CHILDREN BY HAND: entry 300 clones a
/// loop band and re-adds the inner loop (`ddc/ddc_transformation.cpp:984-986`), and entry 249 clones a
/// condition and adds two fresh regions (`ddc/ddc_transformation_util.cpp:580`, `:587-588`).
///
/// ⭐ SO A PORT OF `clone()` MINTS THE CHILDLESS NODE AND DOES NOT REACH FOR THIS `Clone` — that is
/// what `PackStickDim::clone_loop` (`ddc/transformation.rs:1489`) and `Dsc2Store::clone_condition_node`
/// (`stages/ddc_store.rs:257`) do. This `Clone` is the SNAPSHOT of an owned subtree, which is what
/// `v1::Dsc2Store::schedule_head_block` (`ddc/v1.rs:6318`) answers with.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockNode {
    /// The `ScheduleNode` part.
    pub base: NodeBase,
    /// Field: e010_BlockNode.next_
    ///
    /// `next_` (`dsc/dsc2.h:538`), in order — OWNED INLINE rather than held by index, because the
    /// reference's `VectorOfChildren` is a `vector<unique_ptr<ScheduleNode>>` (`:529`) and owns them
    /// too.
    pub children: Vec<SchedNode>,
}

impl BlockNode {
    /// The position of the child named `name`, which is how `addChildNode`'s `siblingRefNode` is
    /// named in the sequences that insert beside a node they already found.
    #[must_use]
    pub fn child_pos(&self, name: &NodeName) -> Option<ChildPos> {
        self.children
            .iter()
            .position(|child| child.name() == name)
            .map(ChildPos)
    }

    /// `addChildNode(node, /*addBefore*/ false, sibling)` (`dsc/dsc2.cpp:2013`) — insert `node`
    /// IMMEDIATELY AFTER `at`, answering its own position so a run of nodes chains in order.
    pub fn insert_after(&mut self, at: ChildPos, node: SchedNode) -> ChildPos {
        let pos = (at.0 + 1).min(self.children.len());
        self.children.insert(pos, node);
        ChildPos(pos)
    }

    /// `addChildNode(node, /*addBefore=*/false)` — append, which is where a walk in program order puts
    /// every node it mints.
    pub fn add_child(&mut self, node: SchedNode) -> ChildPos {
        self.children.push(node);
        ChildPos(self.children.len() - 1)
    }

    /// `addChildNode(node, /*addBefore=*/true)` — the front insert, whose caller is the allocation
    /// case: `allocParent->addChildNode(myAllocNode, allocParent != currParent)`
    /// (`ddc/ddl/ddl_conversion.cpp:782`, and `:831` for an external constant's), which goes in FRONT
    /// exactly when the allocation's region block is an ANCESTOR of the current parent rather than the
    /// current parent itself.
    pub fn add_child_front(&mut self, node: SchedNode) -> ChildPos {
        self.children.insert(0, node);
        ChildPos(0)
    }
}

/// ONE DIM A LOOP ITERATES — `PrimaryDimAndKind` (`dsc/dims.h:76`), the element type of
/// `LoopNode::dims_`: a primary dim paired with the META KIND the DDL dim it came from carried,
/// because that pair is what `convertDsc2Ddl` matches a loop back to its DDL dims by.
///
/// ⭐ ONE C++ DECLARATION, ONE RUST TYPE — `transformation_util::PrimaryDimAndKind` IS this type
/// under its C++ name, so a pair crosses the `ddc` and `dsc2` halves of the port unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopDim {
    /// Field: e011_LoopNode.dim_
    ///
    /// `dim_` (`dsc/dims.h:77`), whose `PrimaryDimTypesCount` default this type does not admit.
    pub dim: PrimaryDim,
    /// Field: e011_LoopNode.kind_
    ///
    /// `kind_` (`dsc/dims.h:78`) — the meta kind of the DDL dim that named this one.
    pub kind: MetaDimKind,
}

/// Replaces: e011_LoopNode
///
/// A LOOP NODE — `dsc2::LoopNode` (`dsc/dsc2.h:563`): a block, the dims it divides, the two
/// datastages whose extents give its trip count, and the symbols a SYMBOLIC dim counts by.
///
/// ⭐ THE PARAMETRIC FLAG AND ITS INDEX ARE ONE FIELD. `markAsParametricLoop()` and
/// `setParametricLdsIdx(lds)` are called together and only together (`ddl_conversion.cpp:1093-1101`),
/// so a parametric loop with no index — and an index on a non-parametric loop — are both unspellable.
/// ⛔ `numId_`/`denId_` ABSENT IS THE REFERENCE'S `-1`, which a parametric loop sets for both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopNode {
    /// The loop's own block part: the `name_` and `next_` it inherits (`dsc/dsc2.h:526`).
    pub block: BlockNode,
    /// Field: e011_LoopNode.dims_
    ///
    /// `dims_` (`dsc/dsc2.h:575`) — *"ordered inner to outer"*.
    pub dims: Vec<LoopDim>,
    /// Field: e011_LoopNode.numId_
    ///
    /// `numId_` (`:573`) — the datastage the trip count divides.
    pub num: Option<DatastageId>,
    /// Field: e011_LoopNode.denId_
    ///
    /// `denId_` (`:574`) — the datastage it divides by.
    pub den: Option<DatastageId>,
    /// Field: e011_LoopNode.isParametricLoop_
    ///
    /// Field: e011_LoopNode.parametricLdsIdx_
    ///
    /// `isParametricLoop_` (`:617`) TOGETHER WITH `parametricLdsIdx_` (`:618`).
    pub parametric_lds: Option<LdsIdx>,
    /// Field: e011_LoopNode.loopCountSymbolIds_
    ///
    /// `loopCountSymbolIds_` (`:576-578`) — *"single value for pure symbolic and pivot dims, multiple
    /// entries (max-pivot) for irregular dims"*, so a dim's entry is a LIST and an absent dim is the
    /// `count(dim) == 0` [`Self::is_dim_symbolic`] reads.
    pub loop_count_symbol_ids: BTreeMap<PrimaryDim, Vec<VariableSymbol>>,
}

impl LoopNode {
    /// A freshly minted loop over `block`, before an arm fills its tiling.
    #[must_use]
    pub const fn bare(block: BlockNode) -> Self {
        Self {
            block,
            dims: Vec::new(),
            num: None,
            den: None,
            parametric_lds: None,
            loop_count_symbol_ids: BTreeMap::new(),
        }
    }

    /// The dims it iterates, without their meta kinds.
    #[must_use]
    pub fn primary_dims(&self) -> Vec<PrimaryDim> {
        self.dims.iter().map(|entry| entry.dim).collect()
    }

    /// `isDimSymbolic(dim)` (`dsc/dsc2.h:595-597`) — `loopCountSymbolIds_.count(dim) != 0`, which is
    /// what decides whether the V3 lowering gives the `scf.for` a symbolic upper bound instead of a
    /// counted one (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:918-937`).
    #[must_use]
    pub fn is_dim_symbolic(&self, dim: PrimaryDim) -> bool {
        self.loop_count_symbol_ids.contains_key(&dim)
    }

    /// `hasLoopDim(dim)` (`dsc/dsc2.cpp:4223-4228`) — a linear scan of `dims_` that compares the
    /// `dim_` half of each pair only, so a dim named at another `kind_` still matches.
    #[must_use]
    pub fn has_loop_dim(&self, dim: PrimaryDim) -> bool {
        self.dims.iter().any(|entry| entry.dim == dim)
    }
}

/// WHICH END OF A SIGNAL PAIR A SYNC NODE IS — `SyncNode::isReceive_` (`dsc/dsc2.h:967`), whose
/// `false` default is the sending end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    /// `isReceive_ == false`.
    Send,
    /// `isReceive_ == true`.
    Receive,
}

/// HOW A SYNC NODE SIGNALS — `SyncNode::isSoft_` (`dsc/dsc2.h:967`), whose `false` default is the
/// hardware signal.
///
/// ⛔ A SEPARATE ENUM AND NOT A SECOND `bool` ON THE SAME CALL: the reference takes `isReceive` and
/// `isSoft` adjacently, and two `bool`s in that position transpose silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrength {
    /// `isSoft_ == false`.
    Hard,
    /// `isSoft_ == true`.
    Soft,
}

/// THE UNITS A SYNC NODE SIGNALS BETWEEN — `SyncNode::units_` (`dsc/dsc2.h:966`), NON-EMPTY because
/// a sync with no unit in it signals to nobody, and a set because the reference's is an
/// `unordered_set`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncUnits(BTreeSet<SenComponent>);

impl SyncUnits {
    /// A sync signals between at least one unit, and this is how that is stated.
    #[must_use]
    pub fn new(first: SenComponent, rest: impl IntoIterator<Item = SenComponent>) -> Self {
        let mut units: BTreeSet<SenComponent> = rest.into_iter().collect();
        units.insert(first);
        Self(units)
    }

    /// `units_`, in the set's order.
    pub fn iter(&self) -> impl Iterator<Item = SenComponent> + '_ {
        self.0.iter().copied()
    }
}

/// Replaces: e013_SyncNode
///
/// `dsc2::SyncNode` (`dsc/dsc2.h:964-972`) carrying ALL FIVE of its declared fields, plus the
/// `name_` this projection needs from the `ScheduleNode` base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncNode {
    /// The `ScheduleNode` part.
    pub base: NodeBase,
    /// Field: e013_SyncNode.units_
    ///
    /// `units_` (`dsc/dsc2.h:966`), *"all to all signals"*.
    pub units: SyncUnits,
    /// Field: e013_SyncNode.isReceive_
    ///
    /// `isReceive_` (`dsc/dsc2.h:967`).
    pub direction: SyncDirection,
    /// Field: e013_SyncNode.isSoft_
    ///
    /// `isSoft_` (`dsc/dsc2.h:967`).
    pub strength: SyncStrength,
    /// Field: e013_SyncNode.implicitSyncRefTransfer_
    ///
    /// `implicitSyncRefTransfer_` (`dsc/dsc2.h:968`) — `nullptr` until entry 261 picks the transfer
    /// this sync stands in for.
    pub implicit_sync_ref_transfer: Option<NodeId>,
    /// Field: e013_SyncNode.otherEndOfTheSignals_
    ///
    /// `otherEndOfTheSignals_` (`dsc/dsc2.h:969`) BY NAME, empty on a fresh node: a sequence that
    /// mints both ends pairs them, and a name is the link a tree of owned nodes can hold.
    ///
    /// ⛔ AND THAT LINK IS EXACT ONLY WHILE THE NAMES ARE. The reference holds
    /// `std::vector<dsc2::SyncNode*>`, so a duplicate name costs it nothing — and duplicates ARE
    /// minted: entry 300 puts `sync_lxsu_send_lxlu`/`sync_lxlu_recv_lxsu` around BOTH halves of the
    /// band it splits, cross-linking each pair by POINTER (`ddc/ddc_transformation.cpp:1199-1208` and
    /// `:1268-1277`), and the port mints the same two names (`ddc/transformation.rs:2329`). Tree names
    /// become unique only in `finalizeScheduleTree`, which appends `__N` (`dsc/dsc2.cpp:2976-2991`),
    /// AFTER those transformations. Until it runs, a resolver keyed on this name answers with the
    /// FIRST node carrying it (`stages/tree.rs:538`), so entry 121 can be asked of the wrong end
    /// (`ddc/transformation_util.rs:1359`). `ddl/conversion.rs:6062` records the same narrowing from
    /// the DDL side.
    pub other_ends: Vec<NodeName>,
}

/// HOW A LOOP CONDITION COMPARES — `dsc2::CondOp` (`dsc/dscdefn.h:95`), verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CondOp {
    /// `EQ`.
    Eq,
    /// `NE`.
    Ne,
    /// `LT`.
    Lt,
    /// `LE`.
    Le,
    /// `GT`.
    Gt,
    /// `GE`.
    Ge,
    /// `TOGGLE`.
    Toggle,
    /// `ALWAYS`.
    Always,
    /// `NEVER`.
    Never,
    /// `CONST`.
    Const,
    /// `DEFAULT`.
    Default,
}

/// WHICH ITERATION A LOOP CONDITION NAMES — `LoopCond::CondValType` FUSED WITH the `condValInt_`
/// only its `INT` arm reads (`dsc/dsc2.h:654-672`), so the `-1` sentinel is unspellable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopBound {
    /// `INT` with `condValInt_ = n`.
    Index(u32),
    /// `FIRST`.
    First,
    /// `LAST`.
    Last,
}

/// ONE CLAUSE OF A CONDITION — `dsc2::LoopCond` (`dsc/dsc2.h:654`): this loop's index on this dim,
/// compared against this bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopCond {
    /// `loopComp_`.
    pub loop_comp: LoopId,
    /// `dim_`.
    pub dim: PrimaryDim,
    /// `condOp_`.
    pub op: CondOp,
    /// `condValType_` with `condValInt_`.
    pub bound: LoopBound,
}

/// A CONDITION'S PREDICATE — `dsc2::LoopCondComposite` (`dsc/dsc2.h:675`), an OR of ANDs with an
/// optional negation over the whole thing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoopCondComposite {
    /// `twoLevelOrOfAnds_` — the outer vector is the OR, each inner vector an AND.
    pub two_level_or_of_ands: Vec<Vec<LoopCond>>,
    /// `negated_`.
    pub negated: bool,
}

/// THE TWO REGIONS A CONDITION HOLDS — `ConditionNode`'s inherited `next_` (`dsc/dsc2.h:529`) under
/// this node's own documented assumption, *"max 2 children in next_, of type BLOCK: the 'then' and
/// 'else' regions"* (`:688`).
///
/// ⛔ ALL THREE `DT_ERROR` ARMS OF `ConditionNode::addChildNode` (`dsc/dsc2.cpp:2143-2166`) ARE
/// UNSPELLABLE HERE, not refused: a child that is not a `BLOCK`, a third child, and an
/// `addElseRegion` with `next_` still empty.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CondRegions {
    /// `next_` EMPTY — `new dsc2::ConditionNode()` (`dsc/dsc2.h:686`) before either region is added.
    #[default]
    Empty,
    /// `next_.size() == 1` — a `getThenBranchNode()` with no else branch beside it.
    Then(BlockNode),
    /// `next_.size() == 2` — `next_[0]` and `next_[1]` (`dsc/dsc2.h:707`, `:713`).
    ThenElse([BlockNode; 2]),
}

impl CondRegions {
    /// `next_` IN ORDER, which is what `ConditionNode::getNextView` returns UNFILTERED for a
    /// loop-guarded condition (`dsc/dsc2.cpp:1995-2011`).
    #[must_use]
    pub fn regions(&self) -> &[BlockNode] {
        match self {
            Self::Empty => &[],
            Self::Then(then) => core::slice::from_ref(then),
            Self::ThenElse(both) => both,
        }
    }

    /// The same list, borrowed for the in-place edits a splice into a region performs.
    pub fn regions_mut(&mut self) -> &mut [BlockNode] {
        match self {
            Self::Empty => &mut [],
            Self::Then(then) => core::slice::from_mut(then),
            Self::ThenElse(both) => both,
        }
    }

    /// `getThenBranchNode()` (`dsc/dsc2.h:707`) — `next_[0]`, and its `nullptr` for an empty `next_`.
    #[must_use]
    pub const fn then_branch(&self) -> Option<&BlockNode> {
        match self {
            Self::Empty => None,
            Self::Then(then) | Self::ThenElse([then, _]) => Some(then),
        }
    }

    /// `getElseBranchNode()` (`:713`) — `next_[1]`, and its `nullptr` for `next_.size() < 2`.
    #[must_use]
    pub const fn else_branch(&self) -> Option<&BlockNode> {
        match self {
            Self::Empty | Self::Then(_) => None,
            Self::ThenElse([_, otherwise]) => Some(otherwise),
        }
    }
}

/// Replaces: e012_ConditionNode
///
/// `dsc2::ConditionNode` (`dsc/dsc2.h:685`) — a block whose children ARE its then-region and its
/// else-region, guarded by a loop predicate or by a core/corelet set.
///
/// ⛔ `hasCoreClCond()` IS `loopCond_.twoLevelOrOfAnds_.empty()` IN THE REFERENCE (`:693-695`), i.e.
/// it answers "the core/corelet set is what guards this", not "the set is non-empty".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConditionNode {
    /// The `ScheduleNode` part, reached through the `BlockNode` base (`dsc/dsc2.h:526`).
    pub base: NodeBase,
    /// Field: e012_ConditionNode.loopCond_
    ///
    /// `loopCond_` (`dsc/dsc2.h:690`).
    pub loop_cond: LoopCondComposite,
    /// Field: e012_ConditionNode.coreClCond_
    ///
    /// `coreClCond_` (`:691-692`) — *"list of core/corelets the 'then' region applies to"*.
    pub core_cl_cond: BTreeMap<Core, BTreeSet<Corelet>>,
    /// Field: e012_ConditionNode.next_
    ///
    /// `next_` (`:529`) — the base's child vector, under this node's two-`BLOCK` assumption (`:688`).
    pub next: CondRegions,
}

impl ConditionNode {
    /// `ConditionNode::addChildNode` (`dsc/dsc2.cpp:2143`) — the THEN region while none is stated and
    /// the ELSE region after, which is how the two blocks a `ddl.if` opens land in order.
    ///
    /// ⛔ [`None`] IS *"ConditionNode only accepts 2 BlockNodes as children"* AND IS NOW ITS ONLY
    /// SPELLABLE ARM — the other two are discharged by the argument type and by [`CondRegions`].
    pub fn add_region(&mut self, block: BlockNode) -> Option<()> {
        match core::mem::take(&mut self.next) {
            CondRegions::Empty => self.next = CondRegions::Then(block),
            CondRegions::Then(then) => self.next = CondRegions::ThenElse([then, block]),
            both @ CondRegions::ThenElse(_) => {
                self.next = both;
                return None;
            }
        }
        Some(())
    }

    /// `hasCoreClCond()` — TRUE when no loop predicate was given, so `coreClCond_` is the guard.
    #[must_use]
    pub fn has_core_cl_cond(&self) -> bool {
        self.loop_cond.two_level_or_of_ands.is_empty()
    }
}

/// `dsc2::StickMaskNode` (`dsc/dsc2.h:1058`) — a mask applied to the trailing elements of a stick,
/// naming the constant that holds the mask value and the transfers it applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StickMaskNode {
    /// The `ScheduleNode` part.
    pub base: NodeBase,
    /// `maskValConstId_` (`:1063`), whose `-1` default is [`None`].
    pub mask_val_const_id: Option<ConstIdx>,
    /// `dataFormat_` (`:1064`), whose `INVALID` default is [`None`].
    pub data_format: Option<DataFormat>,
    /// `stickLayout_` (`:1065`) — the stick's dim sizes, which every affected transfer must agree on.
    pub stick_layout: Vec<Size>,
    /// `firstStickCoordToMaskPerDim_` (`:1066`) — the first masked coordinate per dim. EMPTY is the
    /// reset node, which masks nothing.
    pub first_stick_coord_to_mask_per_dim: BTreeMap<PrimaryDim, Elements>,
    /// `affectedTransfers_` (`:1067`).
    pub affected_transfers: Vec<NodeId>,
}

/// A SCHEDULE NODE, AS THE FOLD UNITS SEE IT — `nodeType_`'s `ALLOCATE`, `COMPUTE` and `TRANSFER`
/// and nothing else, so `dbgPrint`'s and `getComponent`'s `"Unsupported node type"` cannot be
/// reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Node<'a> {
    /// `ScheduleNode::ALLOCATE`.
    Allocate(&'a AllocateNode),
    /// `ScheduleNode::COMPUTE`.
    Compute(&'a ComputeNode),
    /// `ScheduleNode::TRANSFER`.
    Transfer(&'a TransferNode),
}

impl Node<'_> {
    /// `ScheduleNode::name_`.
    #[must_use]
    pub const fn name(&self) -> &NodeName {
        match self {
            Self::Allocate(node) => &node.name,
            Self::Compute(node) => &node.name,
            Self::Transfer(node) => &node.name,
        }
    }
}

/// WHICH SIDE OF A TRANSFER — `getComponent`'s `getSrc` flag, which is ignored for every other
/// node kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferSide {
    /// `getSrc == true`, the default.
    Src,
    /// `getSrc == false` — the FIRST destination.
    Dst,
}

/// WHICH OPERAND — `getLayoutDimsFromNode`'s `(inputPos, outputPos)` pair of `-1` sentinels.
///
/// ⛔ "Neither source nor destination was specified" is unspellable: there is no third arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandPos {
    /// `inputPos` — for a transfer this names the source, which is the reference's
    /// `DT_CHECK(inputPos == 0)`: a transfer has exactly one, so the index cannot select wrongly.
    Input(usize),
    /// `outputPos` — for a transfer, the destination at that index.
    Output(usize),
}

/// A LABELLED DATA STRUCTURE'S LAYOUT ORDER — `AllocateNode::layoutDimOrder_` as
/// `getLayoutDims` returns it, NON-EMPTY because that function `DT_CHECK`s it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutDims {
    first: PrimaryDim,
    rest: Vec<PrimaryDim>,
}

impl LayoutDims {
    /// A layout order has at least one dim, and this is how that is stated.
    #[must_use]
    pub const fn new(first: PrimaryDim, rest: Vec<PrimaryDim>) -> Self {
        Self { first, rest }
    }

    /// THE INNERMOST DIM — `getLayoutDims(...).at(0)`, total because the order is NON-EMPTY.
    ///
    /// ⛔ INDEX 0 IS THE INNERMOST, NOT THE OUTERMOST. `getNonBroadcastLdsDims` walks
    /// `getLayoutDims(ldsIdx)` FORWARD (`dsc/dsc2.cpp:4044`) and both callers of that walk call it
    /// "from the innermost to outer dimensions in `layoutDimOrder_`"
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5038-5039`; `ddc/ddcv1.cpp:1937-1938` says the same
    /// over `allocNode->layoutDimOrder_` directly), and the `DataOpDsc` twin field spells it out —
    /// `// idx 0 means innermost` (`dsc/dataOpDsc.h:347`). `primaryDsInfo_`'s order shares the
    /// direction, being copied onto the node UNREVERSED
    /// (`dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1234`), so this holds for both of the lists
    /// [`Self::index_of`]'s banner keeps apart. Agrees with [`Self::iter`] and with
    /// [`AllocLayout::innermost_dim`].
    #[must_use]
    pub const fn first(&self) -> PrimaryDim {
        self.first
    }

    /// The dims, innermost first — the order `layoutDimOrder_` itself is in
    /// (`dsc/dataOpDsc.h:347`, `ddc/ddcv1.cpp:1937-1938`).
    pub fn iter(&self) -> impl Iterator<Item = PrimaryDim> + '_ {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }

    /// The dims as the reference's `std::vector<PrimaryDimTypes>`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<PrimaryDim> {
        self.iter().collect()
    }
}

/// WHAT THE FOLD UNITS ASK OF A `DesignSpaceConfig`.
///
/// ⭐ REACHING THE LAYOUT ORDER IS THE MECHANISM, NOT THE FACT. `getLayoutDims`
/// (`dsc/dsc2.cpp:4007`) walks `referenceLdsIdx_` until it finds a `memOrg_` entry with an
/// allocate node, preferring `LX`/`HBM`; what every caller wants is the order it lands on.
pub trait Dsc {
    /// `DesignSpaceConfig::getLayoutDims(ldsIdx)` — the ANSWER, whose walk is [`layout_dims`]. A
    /// carrier states what that walk landed on; it does not re-derive it.
    fn layout_dims(&self, lds: LdsIdx) -> LayoutDims;
}

/// WHICH GENERIC COMPONENT A `SenComponents` IS — `EnumsConversion::senCompToGenericComp`
/// (`sys-arch-spec/arch_enums.cpp:124-211`), the map every predicate over a node's units consults.
///
/// ⭐ IT IS A FREE FUNCTION AND NOT A METHOD because [`SenComponent`] is another crate's type;
/// [`crate::generated::Unit::generic`] is the same map over the DDL-censused subset, and the two are
/// one table read through two vocabularies.
///
/// ⛔ [`None`] IS "NO NAME FOR ITS IMAGE", NOT "NO IMAGE". Nine components map to register-file and
/// buffer images ([`SenComponent::Lrfreg`] and its three per-unit spellings, `SFPLRF`, `PELRF`,
/// `PTARF`, `PTXRF`, `LXLUSCALEREG`, `LXLUVALUE`, `L3IBR`, `QGI`) that [`GenericComp`] does not name,
/// and the remaining components are not keys of the reference map at all — `senCompToGenericComp
/// .at(comp)` THROWS for them. Every rule built on this asks `== <a named component>`, so "no name"
/// and "a different component" are one answer and the throw never has to be reproduced.
///
/// ⛔ `L0`, `CONSTANT` AND `SFPRING` ANSWER THEMSELVES rather than [`None`]: they are not keys of the
/// reference map either, but [`GenericComp`] already names them for exactly that reason (see its
/// own note), so this stays consistent with the crate's other reading of the table.
/// ⛔ AND `PTNORTH`/`PTWEST`/`PTSOUTH` ARE *NOT* KEYS. [`crate::generated::Unit::generic`] folds the
/// north and south links into `Pt`; the reference table has no entry for any of the three, so they
/// answer [`None`] here.
#[must_use]
pub const fn generic_comp(unit: SenComponent) -> Option<GenericComp> {
    match unit {
        // `:125-127` — the SFP and its two per-corelet copies.
        SenComponent::Sfp | SenComponent::Sfp0 | SenComponent::Sfp1 => Some(GenericComp::Sfp),
        // `:128-152` — EVERY PT row, in the bare spelling and in both per-fold copies.
        SenComponent::Pt
        | SenComponent::Ptrow0
        | SenComponent::Ptrow1
        | SenComponent::Ptrow2
        | SenComponent::Ptrow3
        | SenComponent::Ptrow4
        | SenComponent::Ptrow5
        | SenComponent::Ptrow6
        | SenComponent::Ptrow7
        | SenComponent::Ptrow0_0
        | SenComponent::Ptrow1_0
        | SenComponent::Ptrow2_0
        | SenComponent::Ptrow3_0
        | SenComponent::Ptrow4_0
        | SenComponent::Ptrow5_0
        | SenComponent::Ptrow6_0
        | SenComponent::Ptrow7_0
        | SenComponent::Ptrow0_1
        | SenComponent::Ptrow1_1
        | SenComponent::Ptrow2_1
        | SenComponent::Ptrow3_1
        | SenComponent::Ptrow4_1
        | SenComponent::Ptrow5_1
        | SenComponent::Ptrow6_1
        | SenComponent::Ptrow7_1 => Some(GenericComp::Pt),
        // `:160-162` — the state components are their own images.
        SenComponent::Sfpstate => Some(GenericComp::SfpState),
        SenComponent::Pestate => Some(GenericComp::PeState),
        // `:165-167` — the PE and its two per-corelet copies.
        SenComponent::Pe | SenComponent::Pe0 | SenComponent::Pe1 => Some(GenericComp::Pe),
        // `:168-176` — ⛔ THE LOAD HALF AND THE STORE HALF ARE DIFFERENT IMAGES.
        SenComponent::Lxlu | SenComponent::Lxlu0 | SenComponent::Lxlu1 => Some(GenericComp::Lxlu),
        SenComponent::Lxsu | SenComponent::Lxsu0 | SenComponent::Lxsu1 => Some(GenericComp::Lxsu),
        SenComponent::Lx => Some(GenericComp::Lx),
        // `:177-204` — the L0 load unit, every row spelling of it, and the store half.
        SenComponent::L0lu
        | SenComponent::L0lu0
        | SenComponent::L0lu1
        | SenComponent::L0lurow0
        | SenComponent::L0lurow1
        | SenComponent::L0lurow2
        | SenComponent::L0lurow3
        | SenComponent::L0lurow4
        | SenComponent::L0lurow5
        | SenComponent::L0lurow6
        | SenComponent::L0lurow7
        | SenComponent::L0lurow0_0
        | SenComponent::L0lurow1_0
        | SenComponent::L0lurow2_0
        | SenComponent::L0lurow3_0
        | SenComponent::L0lurow4_0
        | SenComponent::L0lurow5_0
        | SenComponent::L0lurow6_0
        | SenComponent::L0lurow7_0
        | SenComponent::L0lurow0_1
        | SenComponent::L0lurow1_1
        | SenComponent::L0lurow2_1
        | SenComponent::L0lurow3_1
        | SenComponent::L0lurow4_1
        | SenComponent::L0lurow5_1
        | SenComponent::L0lurow6_1
        | SenComponent::L0lurow7_1 => Some(GenericComp::L0lu),
        SenComponent::L0su | SenComponent::L0su0 | SenComponent::L0su1 => Some(GenericComp::L0su),
        // `:205-209` — the L3 halves, the HBM and the virtual IBR.
        SenComponent::L3lu => Some(GenericComp::L3lu),
        SenComponent::L3su => Some(GenericComp::L3su),
        SenComponent::Hbm => Some(GenericComp::Hbm),
        SenComponent::Lxvirtualibr => Some(GenericComp::LxVirtualIbr),
        // `:210` — the link out of this partition.
        SenComponent::Crossptnlink => Some(GenericComp::CrossPtnLink),
        // ⛔ NOT KEYS, but [`GenericComp`] names them; see this function's note.
        SenComponent::L0 => Some(GenericComp::L0),
        SenComponent::Constant => Some(GenericComp::Constant),
        SenComponent::Sfpring => Some(GenericComp::SfpRing),
        // The register-file and buffer images [`GenericComp`] does not name, and every component the
        // reference map has no key for.
        SenComponent::Lrfreg
        | SenComponent::PeLrfreg
        | SenComponent::SfpLrfreg
        | SenComponent::PtLrfreg
        | SenComponent::Sfplrf
        | SenComponent::Pelrf
        | SenComponent::Ptarf
        | SenComponent::Ptxrf
        | SenComponent::Lxluscalereg
        | SenComponent::Lxluvalue
        | SenComponent::L3ibr
        | SenComponent::Qgi
        | SenComponent::NoComponent
        | SenComponent::Ring
        | SenComponent::Zero
        | SenComponent::L3
        | SenComponent::L3luibr
        | SenComponent::L3suibr
        | SenComponent::Lxlusufifo
        | SenComponent::Ptirf
        | SenComponent::Ptnorth
        | SenComponent::Ptwest
        | SenComponent::Ptsouth
        | SenComponent::All
        | SenComponent::One
        | SenComponent::Latch
        | SenComponent::Nfwd0
        | SenComponent::Nfwd2
        | SenComponent::L0Scale => None,
    }
}

/// ONE SCHEDULE-TREE NODE — `dsc2::ScheduleNode`'s subclasses, reduced to the distinction
/// `traverseTreeDFS`'s `nodeTypes` filter makes (`dsc/dsc2.cpp:2222`): which `nodeType_` a node
/// carries, and whether `isBlockNode()` (`dsc/dsc2.h:479`) lets the walk descend into it.
///
/// ⭐ `Loop` AND `Condition` HOLD A [`BlockNode`] BECAUSE THAT IS THE C++ INHERITANCE: `LoopNode`
/// derives from `BlockNode` (`dsc/dsc2.h:563`), so each is its block part narrowed to what the
/// traversal reads. Their own fields belong to the units that read them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedNode {
    /// `nodeType_ == BLOCK` — the only kind a `{BLOCK}` filter yields.
    Block(BlockNode),
    /// `nodeType_ == LOOP`: descended into, never yielded by a `{BLOCK}` filter. Boxed with its own
    /// tiling, because a loop is the one block kind that carries state of its own.
    Loop(Box<LoopNode>),
    /// `nodeType_ == CONDITION`: likewise a block kind that a `{BLOCK}` filter passes over.
    Condition(BlockNode),
    /// `nodeType_ == CONDITION` for a condition MINTED with both its regions — the shape entry 262
    /// splices in, kept apart from [`SchedNode::Condition`] because a two-region condition's children
    /// are not one child vector.
    Guarded(Box<ConditionNode>),
    /// `nodeType_ == STICK_MASK` — a leaf that carries its mask.
    StickMask(Box<StickMaskNode>),
    /// `nodeType_ == SYNC`: a leaf that carries its own node, because the sequences that mint syncs
    /// cross-link the pair they minted and a bare name cannot be linked.
    Sync(SyncNode),
    /// An allocate, compute or transfer node — `isBlockNode()` is false and it has no children, so the
    /// walk neither yields nor descends.
    Leaf(LeafNode),
}

impl SchedNode {
    /// The `ScheduleNode` part, whichever kind of node this is.
    #[must_use]
    pub const fn base(&self) -> &NodeBase {
        match self {
            Self::Block(block) | Self::Condition(block) => &block.base,
            Self::Loop(node) => &node.block.base,
            Self::Guarded(cond) => &cond.base,
            Self::StickMask(mask) => &mask.base,
            Self::Sync(sync) => &sync.base,
            Self::Leaf(leaf) => &leaf.base,
        }
    }

    /// The same part, exclusively, which is how a pass over the tree marks `relevantComps_`
    /// (`dsc/dsc2.cpp:2656-2692`).
    pub const fn base_mut(&mut self) -> &mut NodeBase {
        match self {
            Self::Block(block) | Self::Condition(block) => &mut block.base,
            Self::Loop(node) => &mut node.block.base,
            Self::Guarded(cond) => &mut cond.base,
            Self::StickMask(mask) => &mut mask.base,
            Self::Sync(sync) => &mut sync.base,
            Self::Leaf(leaf) => &mut leaf.base,
        }
    }

    /// `ScheduleNode::name_`, whichever kind of node this is.
    #[must_use]
    pub const fn name(&self) -> &NodeName {
        &self.base().name
    }

    /// Field: e009_ScheduleNode.nodeType_
    ///
    /// `nodeType_` (`dsc/dsc2.h:460`) — the discriminant this enum already is, spelled back out for
    /// the `nodeTypes` filter `traverseTreeDFS` takes (`dsc/dsc2.cpp:2222`).
    #[must_use]
    pub const fn node_type(&self) -> NodeType {
        match self {
            Self::Block(_) => NodeType::Block,
            Self::Loop(_) => NodeType::Loop,
            Self::Condition(_) | Self::Guarded(_) => NodeType::Condition,
            Self::StickMask(_) => NodeType::StickMask,
            Self::Sync(_) => NodeType::Sync,
            Self::Leaf(leaf) => leaf.kind.node_type(),
        }
    }
}

/// WHERE IN A BLOCK'S CHILDREN A NODE SITS — the `siblingRefNode` position `addChildNode`
/// (`dsc/dsc2.cpp:2013`) inserts relative to, minted only by [`BlockNode::child_pos`] so an index
/// into one block cannot be applied to another block's shorter child vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildPos(usize);

/// Replaces: e016_ScheduleTree
///
/// A DSC'S SCHEDULE TREE — `dsc2::ScheduleTree` (`dsc/dsc2.h:621`) carrying its ONE declared field.
///
/// ⭐ `head_` IS NEVER *YIELDED* BY A TRAVERSAL AND IS OBSERVED ANYWAY. `traverseTreeDFS(nullptr, ..)`
/// seeds the queue with `head_.next_` (`dsc/dsc2.cpp:2233`), yet `writeToJson` serialises the
/// sentinel's `denId_` as `"scheduleTreeHeadDenId_"` (`:368`) and the parser reads it straight back
/// (`:1159`), `setRelevantCompCoreCl` writes its `relevantComps_` (`:2656`) and erases it (`:2977`),
/// and `getOwnerLoop()` hands it out as the owner loop of every top-level node (`:5035`).
///
/// ⛔ `Clone` DEEP-COPIES WHERE `copyFrom` REFUSES. `ScheduleTree::copyFrom` (`dsc/dsc2.cpp:2267`)
/// raises *"Not yet able to deep copy a schedule tree"* for any non-empty tree, with the deep copy it
/// means commented out directly above the raise; nothing in the reference tree copies a filled one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleTree {
    /// Field: e016_ScheduleTree.head_
    ///
    /// `head_` (`dsc/dsc2.h:623`) — the UNNAMED `LoopNode` sentinel whose children are the frontier
    /// every traversal starts from, and which no `name_` lookup can reach.
    head: LoopNode,
}

impl Default for ScheduleTree {
    /// `ScheduleTree()` (`dsc/dsc2.h:629`) — an empty tree over a sentinel naming the core datastage.
    fn default() -> Self {
        Self::new(BlockNode::default())
    }
}

impl ScheduleTree {
    /// A tree over `head_`'s children, with the `head_.denId_ = 0` the constructor states.
    #[must_use]
    pub fn new(head: BlockNode) -> Self {
        let mut sentinel = LoopNode::bare(head);
        sentinel.den = Some(Metadata::CORE_DSTGID);
        Self { head: sentinel }
    }

    /// `clear()` (`dsc/dsc2.h:626`) — `head_.next_.clear()`, which drops every node and leaves the
    /// sentinel's own fields alone.
    pub fn clear(&mut self) {
        self.head.block.children.clear();
    }

    /// `empty()` (`dsc/dsc2.h:627`) — `head_.next_.empty()`, which is also what `isDSC2()` reads to
    /// decide a DSC carries a DSC2 schedule at all (`dsc/designSpaceConfig.cpp:31`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.head.block.children.is_empty()
    }

    /// Field: e016_ScheduleTree.denId_
    ///
    /// `getHead()->denId_` — `0`, the core datastage, from construction (`dsc/dsc2.h:629`), and
    /// [`None`] only where an imported `scheduleTreeHeadDenId_` states the reference's `-1`.
    #[must_use]
    pub const fn head_den(&self) -> Option<DatastageId> {
        self.head.den
    }

    /// `getHeadMutable()->denId_ = den` — the write entry 217 makes onto the root before it chains
    /// the chunk loops under it, and the one the importer makes (`dsc/dsc2.cpp:1159`).
    pub fn set_head_den(&mut self, den: DatastageId) {
        self.head.den = Some(den);
    }

    /// `getHead()` (`dsc/dsc2.h:637`) — the sentinel, which IS a `LoopNode`.
    #[must_use]
    pub const fn head(&self) -> &LoopNode {
        &self.head
    }

    /// `getHeadMutable()` (`:638`).
    pub const fn head_mut(&mut self) -> &mut LoopNode {
        &mut self.head
    }

    /// `traverseTreeDFS(nullptr, {BLOCK})` (`dsc/dsc2.cpp:2222`) — every `BLOCK` node in pre-order
    /// DFS, descending through the loop and condition blocks it does not yield.
    #[must_use]
    pub fn blocks_dfs(&self) -> Vec<&BlockNode> {
        let mut found = Vec::new();
        collect_blocks(&self.head.block, &mut found);
        found
    }

    /// `traverseTreeDFSMutable(nullptr, {BLOCK})` FUSED WITH THE `break` ITS CALLERS WRITE: the FIRST
    /// `BLOCK` node the pre-order walk accepts, exclusively borrowed so the caller can edit it.
    ///
    /// ⭐ THE FUSION IS WHAT RUST ADMITS. The reference hands back a `std::vector<ScheduleNode*>`
    /// holding a block AND its descendants at once; every `{BLOCK}` caller in the scheduler stops at
    /// its first match, so the search — not the vector — is the primitive.
    pub fn find_block_mut(
        &mut self,
        accepts: impl Fn(&BlockNode) -> bool + Copy,
    ) -> Option<&mut BlockNode> {
        find_block_mut(&mut self.head.block, accepts)
    }

    /// The same search over the `CONDITION` nodes, which [`Self::find_block_mut`] descends THROUGH
    /// and never yields — `dsc2::ConditionNode` is a `BlockNode` in the reference and this type holds
    /// its two children in [`ConditionNode::next`], so reaching one by name is its own search.
    pub fn find_guarded_mut(
        &mut self,
        accepts: impl Fn(&ConditionNode) -> bool + Copy,
    ) -> Option<&mut ConditionNode> {
        find_guarded_mut(&mut self.head.block, accepts)
    }

    /// The same search over the `SYNC` nodes, which neither [`Self::find_block_mut`] nor
    /// [`Self::find_guarded_mut`] yields — `otherEndOfTheSignals_` is bound onto the sync nodes
    /// THEMSELVES (`ddc/ddl/ddl_conversion.cpp:2813`), and a signal names its two ends by name.
    pub fn find_sync_mut(
        &mut self,
        accepts: impl Fn(&SyncNode) -> bool + Copy,
    ) -> Option<&mut SyncNode> {
        find_sync_mut(&mut self.head.block, accepts)
    }

    /// Field: e009_ScheduleNode.prev_
    ///
    /// `prev_` (`dsc/dsc2.h:515`) FOR THE NODE NAMED `name`, WALKED DOWN: an owned tree cannot hold a
    /// parent pointer, and a chain rebuilt by the search that reached the node cannot go stale.
    ///
    /// ⭐ THE SENTINEL TERMINATES THE CHAIN. `head_` is a `LoopNode`, so `getOwnerLoop()` on a
    /// top-level node hands the head back rather than `nullptr`, which is why the reference's own
    /// outward walks stop on `currNode != getHead()` (`dsc/dsc2.cpp:5035`).
    ///
    /// ⛔ THE NAME MUST BE UNIQUE, AND ONLY `finalizeScheduleTree` MAKES IT SO — it appends `__<i>`
    /// to each repeat, REWRITING `name_` in place (`dsc/dsc2.cpp:2986-2991`). Before that pass a
    /// duplicate resolves to whichever node this pre-order walk reaches first.
    #[must_use]
    pub fn ancestors(&self, name: &NodeName) -> Option<Ancestors<'_>> {
        let mut loops = Vec::new();
        let (parent, loops) = ancestors_in(&self.head.block, &mut loops, name)?;
        Some(Ancestors {
            parent,
            loops,
            head: &self.head,
        })
    }
}

/// WHAT A NODE'S `prev_` CHAIN ANSWERS — the block whose child list holds it, and the loops that
/// enclose it, innermost first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ancestors<'a> {
    /// `getPrev()` (`dsc/dsc2.h:462`).
    pub parent: &'a BlockNode,
    /// The same chain filtered to `LOOP`, innermost first — the loops BELOW the sentinel, which is
    /// the chain the reference's outward walks take (`while (currNode != getHead())`).
    pub loops: Vec<&'a LoopNode>,
    /// `ScheduleTree::getHead()` — the sentinel every `prev_` chain ends at, and so the owner loop of
    /// a top-level node.
    pub head: &'a LoopNode,
}

impl<'a> Ancestors<'a> {
    /// `getOwnerLoop()` (`dsc/dsc2.cpp:1896-1900`) — the innermost enclosing loop.
    ///
    /// ⛔ TOTAL, AND THE REFERENCE'S IS TOO FOR EVERY NAMED NODE: the walk stops at the first `LOOP`
    /// in the `prev_` chain, and every chain ends at the sentinel head, which is one. Only `head_`
    /// itself has `prev_ == nullptr`, and no name reaches it.
    #[must_use]
    pub fn owner_loop(&self) -> &'a LoopNode {
        self.loops.first().copied().unwrap_or(self.head)
    }

    /// `getParentDimLoop(dim)` (`dsc/dsc2.cpp:1906-1914`) — the innermost enclosing loop that tiles
    /// `dim`, which is [`Self::owner_loop`] continued outwards. The sentinel's `dims_` is empty, so
    /// the walk runs out at it exactly where the reference returns `nullptr`.
    #[must_use]
    pub fn parent_dim_loop(&self, dim: PrimaryDim) -> Option<&'a LoopNode> {
        self.loops
            .iter()
            .copied()
            .chain(std::iter::once(self.head))
            .find(|enclosing| enclosing.has_loop_dim(dim))
    }
}

/// The parent and enclosing loops of the node named `name` below `parent`, with `loops` the enclosing
/// loops so far — the sentinel is [`ScheduleTree::ancestors`]'s to add.
type Ancestry<'a> = (&'a BlockNode, Vec<&'a LoopNode>);

fn ancestors_in<'a>(
    parent: &'a BlockNode,
    loops: &mut Vec<&'a LoopNode>,
    name: &NodeName,
) -> Option<Ancestry<'a>> {
    for child in &parent.children {
        if child.name() == name {
            return Some((parent, loops.clone()));
        }
        let found = match child {
            SchedNode::Block(inner) | SchedNode::Condition(inner) => {
                ancestors_in(inner, loops, name)
            }
            SchedNode::Loop(node) => {
                loops.insert(0, node);
                let found = ancestors_in(&node.block, loops, name);
                loops.remove(0);
                found
            }
            SchedNode::Guarded(cond) => cond
                .next
                .regions()
                .into_iter()
                .find_map(|region| ancestors_in(region, loops, name)),
            SchedNode::StickMask(_) | SchedNode::Sync(_) | SchedNode::Leaf(_) => None,
        };
        if found.is_some() {
            return found;
        }
    }
    None
}

/// Pre-order DFS over the `BLOCK` nodes below `block`, which is itself never yielded.
fn collect_blocks<'a>(block: &'a BlockNode, found: &mut Vec<&'a BlockNode>) {
    collect_blocks_in(&block.children, found);
}

/// The same walk over a child list, which is what a two-region condition has instead of a block.
fn collect_blocks_in<'a>(children: &'a [SchedNode], found: &mut Vec<&'a BlockNode>) {
    for child in children {
        match child {
            SchedNode::Block(inner) => {
                found.push(inner);
                collect_blocks(inner, found);
            }
            SchedNode::Loop(node) => collect_blocks(&node.block, found),
            SchedNode::Condition(inner) => collect_blocks(inner, found),
            SchedNode::Guarded(cond) => {
                for region in cond.next.regions() {
                    found.push(region);
                    collect_blocks(region, found);
                }
            }
            SchedNode::StickMask(_) | SchedNode::Sync(_) | SchedNode::Leaf(_) => {}
        }
    }
}

/// The first accepted `BLOCK` node below `block`, in the same pre-order.
fn find_block_mut(
    block: &mut BlockNode,
    accepts: impl Fn(&BlockNode) -> bool + Copy,
) -> Option<&mut BlockNode> {
    find_block_mut_in(&mut block.children, accepts)
}

/// The same search over a child list.
fn find_block_mut_in(
    children: &mut [SchedNode],
    accepts: impl Fn(&BlockNode) -> bool + Copy,
) -> Option<&mut BlockNode> {
    for child in children {
        match child {
            SchedNode::Block(inner) => {
                if accepts(inner) {
                    return Some(inner);
                }
                if let Some(found) = find_block_mut(inner, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Loop(node) => {
                if let Some(found) = find_block_mut(&mut node.block, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Condition(inner) => {
                if let Some(found) = find_block_mut(inner, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Guarded(cond) => {
                for region in cond.next.regions_mut() {
                    if accepts(region) {
                        return Some(region);
                    }
                    if let Some(found) = find_block_mut(region, accepts) {
                        return Some(found);
                    }
                }
            }
            SchedNode::StickMask(_) | SchedNode::Sync(_) | SchedNode::Leaf(_) => {}
        }
    }
    None
}

/// The first accepted `CONDITION` node below `block`, in the same pre-order.
fn find_guarded_mut(
    block: &mut BlockNode,
    accepts: impl Fn(&ConditionNode) -> bool + Copy,
) -> Option<&mut ConditionNode> {
    find_guarded_mut_in(&mut block.children, accepts)
}

/// The same search over a child list.
fn find_guarded_mut_in(
    children: &mut [SchedNode],
    accepts: impl Fn(&ConditionNode) -> bool + Copy,
) -> Option<&mut ConditionNode> {
    for child in children {
        match child {
            SchedNode::Block(inner) | SchedNode::Condition(inner) => {
                if let Some(found) = find_guarded_mut(inner, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Loop(node) => {
                if let Some(found) = find_guarded_mut(&mut node.block, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Guarded(cond) => {
                if accepts(cond) {
                    return Some(cond.as_mut());
                }
                for region in cond.next.regions_mut() {
                    if let Some(found) = find_guarded_mut(region, accepts) {
                        return Some(found);
                    }
                }
            }
            SchedNode::StickMask(_) | SchedNode::Sync(_) | SchedNode::Leaf(_) => {}
        }
    }
    None
}

/// The first accepted `SYNC` node below `block`, in the same pre-order.
fn find_sync_mut(
    block: &mut BlockNode,
    accepts: impl Fn(&SyncNode) -> bool + Copy,
) -> Option<&mut SyncNode> {
    find_sync_mut_in(&mut block.children, accepts)
}

/// The same search over a child list.
fn find_sync_mut_in(
    children: &mut [SchedNode],
    accepts: impl Fn(&SyncNode) -> bool + Copy,
) -> Option<&mut SyncNode> {
    for child in children {
        match child {
            SchedNode::Sync(sync) => {
                if accepts(sync) {
                    return Some(sync);
                }
            }
            SchedNode::Block(inner) | SchedNode::Condition(inner) => {
                if let Some(found) = find_sync_mut(inner, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Loop(node) => {
                if let Some(found) = find_sync_mut(&mut node.block, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Guarded(cond) => {
                for region in cond.next.regions_mut() {
                    if let Some(found) = find_sync_mut(region, accepts) {
                        return Some(found);
                    }
                }
            }
            SchedNode::StickMask(_) | SchedNode::Leaf(_) => {}
        }
    }
    None
}

/// WHAT `getLayoutDims` WALKS — one labelled DS's `memOrg_` allocate nodes and its
/// `referenceLdsIdx_` (`dsc/dscdefn.h:324`, `:337`), neither of which a [`LayoutDims`] carrier holds.
pub trait LabeledDsAllocations {
    /// `labeledDs_.at(lds).memOrg_`'s entries THAT HOLD AN ALLOCATE NODE, in component order, each
    /// with that node's `layoutDimOrder_` (`dsc/dsc2.h:982`) — an entry with a null
    /// `allocateNode_` contributes nothing to the walk, so it is filtered here rather than there.
    ///
    /// ⛔ [`None`] IS `labeledDs_.at(ldsIdx)`'s THROW, which is the one `DT_CHECK(ldsIdx >= 0 &&
    /// ldsIdx < labeledDs_.size())` guards for the FIRST index and nothing guards for a referenced one.
    fn alloc_layout_orders(&self, lds: LdsIdx) -> Option<Vec<(SenComponent, LayoutDims)>>;

    /// `labeledDs_.at(lds).referenceLdsIdx_`, [`None`] for the `-1` that ends the walk.
    fn reference_lds(&self, lds: LdsIdx) -> Option<LdsIdx>;
}

/// Replaces: e006_getLayoutDims
///
/// THE LAYOUT ORDER OF THE ALLOCATE NODE A LABELLED DS RESOLVES TO — follows `referenceLdsIdx_`
/// until some `memOrg_` entry holds an allocate node, keeping the FIRST `LX`/`HBM` entry it meets and
/// otherwise the LAST entry with a node, and answers THAT node's `layoutDimOrder_`.
///
/// ⛔⛔ NOT `primaryDsInfo_.at(dsType_).layoutDimOrder_`, which [`LayoutDims::index_of`] and
/// [`LayoutDims::to_set`] read: THE TWO ORDERS STAY TWO — a second labelled DS of one `dsType_` has
/// its own allocate node, and `primaryDsInfo_` holds one entry for both. ⭐ THEY AGREE ON THE WHOLE
/// CORPUS, MEASURED: 807 of 807 labelled DSs of `g0/debug/sdsc_*/sdsc.json` resolve to a node whose
/// order IS their `dsType_`'s entry, so this is the derivation and not a defect fix.
///
/// ⭐ THE `LX`/`HBM` PREFERENCE CANNOT BE REORDERED AWAY, AND THE `break` IS STILL LOAD-BEARING.
/// `memOrg_` is a `std::map<SenComponents, MemOrg>` (`dsc/dscdefn.h:337`), so it iterates in ordinal
/// order, and `HBM = 0`/`LX = 1` are the two lowest keys a NAMED component can hold — only
/// `NO_COMPONENT = -1` sorts below them (`sys-arch-spec/arch_enums.h:13-17`), and an entry keyed on
/// it is OVERWRITTEN by the `hbm`/`lx` hit rather than kept, because the scan REASSIGNS on every
/// entry that holds a node. What can FOLLOW them is a register file, and the `break` is what stops
/// it overwriting them (`dsc/dsc2.cpp:4016`).
/// ⛔ SO IT IS NOT "ONLY AN EARLY EXIT". MEASURED over all 807 labelled DSs of
/// `g0/debug/sdsc_*/sdsc.json`: 238 hold BOTH an `hbm`/`lx` node and a register-file node, and in all
/// 238 dropping the `break` selects a DIFFERENT node. It reaches the same ANSWER — the two nodes'
/// `layoutDimOrder_`s are equal in all 238 — but that is a corpus measurement and not the reference's
/// guarantee, and g0's 187 programs are one of 134 bundles.
/// ⭐ [`BTreeMap<SenComponent, _>`] IS THAT SAME ORDER, because [`SenComponent`]'s derived [`Ord`] is
/// its C++ numbering — which is why the seam states its entries "in component order".
///
/// ⭐ THE LAST-ENTRY-WINS ARM DECIDES AMONG REGISTER FILES ALONE, and the corpus does reach it: 219 of
/// the 807 resolve to a `pelrf`/`ptarf`/`sfplrf` node (580 take `hbm` and 8 take `lx`, so 227 is every
/// NON-HBM resolution and not this arm's count). 12 of the 219 name TWO register files, and in all 12
/// their orders are equal — an arm the corpus ENTERS, not a choice the corpus EXERCISES.
/// ⛔ NEITHER THAT ARM NOR THE `referenceLdsIdx_` HOP IS REACHABLE THROUGH THE ONE PRODUCTION CARRIER.
/// `WireAllocations` (`crates/targets/spyre/src/superdsc_to_l3_sdsc.rs:580`, its seam impl at `:607`)
/// REFUSES a `component_` outside `{hbm, lx}` and answers [`LabeledDsAllocations::reference_lds`]
/// with [`None`], so on our wire this walk stops inside its first lap every time.
///
/// ⛔ [`None`] IS `DT_CHECK(allocNode)` (`dsc/dsc2.cpp:4022`), an unheld index, and a
/// `referenceLdsIdx_` CYCLE — which the reference does not terminate on at all.
/// ⭐ `DT_CHECK(!allocNode->layoutDimOrder_.empty())` (`:4023`) IS DISCHARGED BY [`LayoutDims`]' OWN
/// SHAPE and is not re-checked here: 85 of the corpus's 1899 allocate nodes DO carry an empty order,
/// and no labelled DS resolves to one of them.
#[must_use]
pub fn layout_dims(dsc: &(impl LabeledDsAllocations + ?Sized), lds: LdsIdx) -> Option<LayoutDims> {
    let mut visited = BTreeSet::new();
    let mut at = Some(lds);
    while let Some(idx) = at {
        if !visited.insert(idx) {
            return None;
        }
        let mut found = None;
        for (component, order) in dsc.alloc_layout_orders(idx)? {
            let preferred = matches!(component, SenComponent::Lx | SenComponent::Hbm);
            found = Some(order);
            if preferred {
                break;
            }
        }
        if found.is_some() {
            return found;
        }
        at = dsc.reference_lds(idx);
    }
    None
}

#[cfg(test)]
mod tests_e006 {
    //! ⭐⭐ `getLayoutDims`' WALK OVER THE REFERENCE'S OWN `memOrg_` TABLES.
    //!
    //! # WHERE THE NUMBERS COME FROM
    //!
    //! `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_<N>/sdsc.json` — the reference's export after
    //! its own L3/ddc/dcg ran. Each case below is one `labeledDs_[i].memOrg_`, with every component
    //! it names, whether that entry holds an `allocateNode_`, and that node's own `layoutDimOrder_`.
    //!
    //! ⭐ THE EXPECTED ORDER IS A VALUE THE REFERENCE WROTE, not a re-derivation: `constructAllocation`
    //! assigns `allocNode->layoutDimOrder_ = currDsc->getLayoutDims(labelledDS.ldsIdx_)`
    //! (`ddc/ddc_transformation_util.cpp:20-60`), so every node the scheduler MINTED carries this
    //! function's own answer. MEASURED across the 187 programs: 1,197 of 1,213 `allocate_lds<N>_*`
    //! nodes carry exactly the order this walk answers for `<N>` over the exported `memOrg_`. The 16
    //! exceptions are all in the eight `attn_newkt*` transposes, where lds 1's `lx` entry points at
    //! `allocate_lds0_lx_internalInput` and the node still NAMED `allocate_lds1_lx` is the stale one —
    //! the name is stale, not the walk.
    //!
    //! ⛔ WHAT THE CORPUS CANNOT DISCRIMINATE, MEASURED: within one labelled DS every component's
    //! node carries the SAME order (0 of 807 records disagree), so no g0 program can tell which entry
    //! was picked — only whether SOME entry answered. And `referenceLdsIdx_` is absent from all 807
    //! records. The cases marked CONSTRUCTED below cover those two, from `dsc/dsc2.cpp:4007-4025`.

    use super::{LabeledDsAllocations, LayoutDims, LdsIdx, layout_dims};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
    use std::collections::BTreeMap;
    use sys_arch_spec::arch_enums::SenComponent;

    /// ONE PROGRAM'S `labeledDs_` AS THE WALK SEES IT — per index, the components whose entry holds an
    /// allocate node with that node's order, and the `referenceLdsIdx_`.
    struct Exported(BTreeMap<LdsIdx, (Vec<(SenComponent, LayoutDims)>, Option<LdsIdx>)>);

    impl LabeledDsAllocations for Exported {
        fn alloc_layout_orders(&self, lds: LdsIdx) -> Option<Vec<(SenComponent, LayoutDims)>> {
            self.0.get(&lds).map(|(orders, _)| orders.clone())
        }

        fn reference_lds(&self, lds: LdsIdx) -> Option<LdsIdx> {
            self.0.get(&lds).and_then(|(_, reference)| *reference)
        }
    }

    fn order(dims: &[PrimaryDim]) -> LayoutDims {
        let (first, rest) = dims.split_first().expect("a non-empty layout order");
        LayoutDims::new(*first, rest.to_vec())
    }

    fn exported(entries: Vec<(u32, Vec<(SenComponent, &[PrimaryDim])>, Option<u32>)>) -> Exported {
        Exported(
            entries
                .into_iter()
                .map(|(lds, orders, reference)| {
                    (
                        LdsIdx(lds),
                        (
                            orders
                                .into_iter()
                                .map(|(component, dims)| (component, order(dims)))
                                .collect(),
                            reference.map(LdsIdx),
                        ),
                    )
                })
                .collect(),
        )
    }

    const MB_OUT_Y: [PrimaryDim; 3] = [PrimaryDim::Mb, PrimaryDim::Out, PrimaryDim::Y];
    const IN_OUT: [PrimaryDim; 2] = [PrimaryDim::In, PrimaryDim::Out];
    const MB_OUT: [PrimaryDim; 2] = [PrimaryDim::Mb, PrimaryDim::Out];
    const Y_OUT: [PrimaryDim; 2] = [PrimaryDim::Y, PrimaryDim::Out];

    /// e006 — six exported `memOrg_` tables, then the arms the corpus does not reach.
    #[test]
    fn a_labelled_ds_resolves_to_the_order_the_reference_recorded() {
        // `sdsc_0` — lds 0 `{hbm, lx}`, both with a node; lds 1 `{hbm, lx, sfplrf}` where the
        // `sfplrf` entry's `allocateNode_` is the EMPTY STRING, so the seam does not list it (48 of
        // the 807 records have such an entry).
        let sdsc_0 = exported(vec![
            (
                0,
                vec![
                    (SenComponent::Hbm, &MB_OUT_Y),
                    (SenComponent::Lx, &MB_OUT_Y),
                ],
                None,
            ),
            (
                1,
                vec![
                    (SenComponent::Hbm, &MB_OUT_Y),
                    (SenComponent::Lx, &MB_OUT_Y),
                ],
                None,
            ),
        ]);
        assert_eq!(layout_dims(&sdsc_0, LdsIdx(0)), Some(order(&MB_OUT_Y)));
        assert_eq!(layout_dims(&sdsc_0, LdsIdx(1)), Some(order(&MB_OUT_Y)));
        // `DT_CHECK(ldsIdx >= 0 && ldsIdx < labeledDs_.size())` — an index the program does not hold.
        assert_eq!(layout_dims(&sdsc_0, LdsIdx(2)), None);

        // `sdsc_1` lds 1 — `{pelrf, sfplrf}`, NO `hbm`/`lx`: the scan never breaks and the LAST entry
        // with a node is the answer.
        let sdsc_1 = exported(vec![(
            1,
            vec![
                (SenComponent::Pelrf, &MB_OUT_Y),
                (SenComponent::Sfplrf, &MB_OUT_Y),
            ],
            None,
        )]);
        assert_eq!(layout_dims(&sdsc_1, LdsIdx(1)), Some(order(&MB_OUT_Y)));

        // `sdsc_15` — lds 1 is the KERNEL `{hbm, lx, ptxrf}` on `["in", "out"]`, lds 2 an OUTPUT held
        // ONLY in `{ptarf}`, and lds 3 the same in `{pelrf}`. A DIFFERENT order from `sdsc_0`'s.
        let sdsc_15 = exported(vec![
            (
                1,
                vec![
                    (SenComponent::Hbm, &IN_OUT),
                    (SenComponent::Lx, &IN_OUT),
                    (SenComponent::Ptxrf, &IN_OUT),
                ],
                None,
            ),
            (2, vec![(SenComponent::Ptarf, &MB_OUT)], None),
            (3, vec![(SenComponent::Pelrf, &MB_OUT)], None),
        ]);
        assert_eq!(layout_dims(&sdsc_15, LdsIdx(1)), Some(order(&IN_OUT)));
        assert_eq!(layout_dims(&sdsc_15, LdsIdx(2)), Some(order(&MB_OUT)));
        assert_eq!(layout_dims(&sdsc_15, LdsIdx(3)), Some(order(&MB_OUT)));

        // `sdsc_50` lds 1 — `{lx, l0, sfplrf}`, an INTERNAL tensor with no HBM at all.
        let sdsc_50 = exported(vec![(
            1,
            vec![
                (SenComponent::Lx, &Y_OUT),
                (SenComponent::L0, &Y_OUT),
                (SenComponent::Sfplrf, &Y_OUT),
            ],
            None,
        )]);
        assert_eq!(layout_dims(&sdsc_50, LdsIdx(1)), Some(order(&Y_OUT)));

        // CONSTRUCTED — WHAT THE CORPUS CANNOT DISCRIMINATE, AND THE NEGATIVE CONTROL FOR `:4016`'s
        // `break`: an `hbm`/`lx` entry and a REGISTER FILE whose orders DISAGREE. 238 of the 807
        // records hold both kinds of node and dropping the `break` picks the register file in every
        // one of them; g0 only hides that because the two orders are equal in all 238. Here they are
        // not, so a lost `break` answers `Y_OUT`/`MB_OUT` instead.
        let disagreeing = exported(vec![
            (
                0,
                vec![
                    (SenComponent::Hbm, &MB_OUT_Y),
                    (SenComponent::Sfplrf, &Y_OUT),
                ],
                None,
            ),
            (
                1,
                vec![(SenComponent::Lx, &IN_OUT), (SenComponent::Pelrf, &MB_OUT)],
                None,
            ),
        ]);
        assert_eq!(layout_dims(&disagreeing, LdsIdx(0)), Some(order(&MB_OUT_Y)));
        assert_eq!(layout_dims(&disagreeing, LdsIdx(1)), Some(order(&IN_OUT)));

        // CONSTRUCTED — `DT_CHECK(allocNode)` (`:4022`): every entry of the index the walk lands on
        // has a null `allocateNode_`, and `referenceLdsIdx_` is `-1`, so the walk ends with nothing.
        let starved = exported(vec![(0, vec![], None)]);
        assert_eq!(layout_dims(&starved, LdsIdx(0)), None);

        // CONSTRUCTED — the `referenceLdsIdx_` hop, which no g0 record takes: lds 1 holds no node and
        // names lds 0 as its reference, so the answer is lds 0's node.
        let referenced = exported(vec![
            (0, vec![(SenComponent::Hbm, &MB_OUT_Y)], None),
            (1, vec![], Some(0)),
        ]);
        assert_eq!(layout_dims(&referenced, LdsIdx(1)), Some(order(&MB_OUT_Y)));

        // CONSTRUCTED — a `referenceLdsIdx_` CYCLE, which the reference spins on forever.
        let cycle = exported(vec![(0, vec![], Some(1)), (1, vec![], Some(0))]);
        assert_eq!(layout_dims(&cycle, LdsIdx(0)), None);
    }
}

impl AllocateNode {
    /// Replaces: e007_getPageSize
    ///
    /// ONE PAGE'S ELEMENTS PER LAYOUT DIM — EMPTY for an allocation that indirects through nothing
    /// (`dsc/dsc2.cpp:4483-4485`), which is what every program we compile states: `indirectAllocType_`
    /// has ONE emitter construction site and it writes `"no_indirection"`.
    ///
    /// ⛔ THE INDIRECTION IS A PARAMETER, NOT A FIELD: `indirectAllocType_` (`dsc/dsc2.h:990`) and
    /// `relatedIndirectAccessAlloc_` are typed on
    /// [`L3AllocateNode`](crate::schedule::l3::dl_ops::L3AllocateNode), whose file another agent owns
    /// this wave, so the caller hands over what it reads — as e015 takes its ancestor loop chain.
    #[must_use]
    pub fn page_sizes(&self, indirect: Option<IndirectAlloc>) -> BTreeMap<PrimaryDim, Extent> {
        match indirect {
            None => BTreeMap::new(),
            Some(through) => todo!(
                "dsc2::AllocateNode::page_sizes: dsc/dsc2.cpp:4486-4511 — {through:?} wants the \
                 REFERENCE allocation (this node for VALUE_TENSOR, relatedIndirectAccessAlloc_ for \
                 INDEX_TENSOR, whose DT_CHECK at :4491 is its presence) and then that node's \
                 layoutDimOrder_/maxDimSizes_ walk, where a NEGATIVE max size UNBOUNDS the dim and \
                 ERASES the page size it had"
            ),
        }
    }
}

#[cfg(test)]
mod tests_e007 {
    //! ⭐ EXHAUSTIVE BY CONSTRUCTION AND NOT SAMPLED: `indirectAllocType_` has ONE emitter
    //! construction site (`crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs:5212`) and it
    //! writes `"no_indirection"`, so `NO_INDIRECTION` is the arm all 24,363 programs take.

    use super::{
        AllocLayout, AllocPlacement, AllocateNode, BTreeMap, Elements, LdsIdx, MaxDimSize,
        NodeName, NumBuffers, PaddingForm, PrimaryDim, SenComponent, StartAddress,
    };

    /// An HBM allocation with a BOUNDED two-dim layout, so an answer keyed on a layout dim would show.
    fn hbm_alloc() -> AllocateNode {
        AllocateNode {
            name: NodeName("kv_cache".to_owned()),
            component: SenComponent::Hbm,
            lds: Some(LdsIdx(0)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new(
                (PrimaryDim::Out, MaxDimSize::Resolved(Elements(64))),
                vec![(PrimaryDim::Y, MaxDimSize::Resolved(Elements(16)))],
            ),
            start_address: StartAddress::default(),
            placement: AllocPlacement {
                num_buffers: NumBuffers::Single,
                padding: PaddingForm::default(),
                buffer_offset: BTreeMap::new(),
                is_start_addr_symbolic: false,
            },
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        }
    }

    /// e007 — the arm every program takes, and it reads no layout at all.
    #[test]
    fn an_allocation_that_indirects_through_nothing_has_no_page_size() {
        // `NO_INDIRECTION` returns BEFORE `layoutDimOrder_` is touched (`dsc/dsc2.cpp:4483-4485`), so
        // a bounded 64x16 layout still pages nothing — 64 and 16 must not appear.
        assert_eq!(hbm_alloc().page_sizes(None), BTreeMap::new());
    }
}

impl LoopNode {
    /// Replaces: e014_parametricStride
    ///
    /// A PARAMETRIC LOOP'S STRIDE — the CUMULATIVE STICK SIZE of its one dim in the labelled DS
    /// `parametricLdsIdx_` names, or `1` for a dim that is not a stick dim but IS in that DS's layout
    /// order.
    ///
    /// ⛔ [`None`] IS A STOP, NOT A STRIDE OF `1`: `DT_CHECK_MSG(ldsIdx != -1)` (`dsc/dsc2.cpp:4199`)
    /// and the `DT_ERROR` for a dim in NEITHER set (`:4213`). `getSizeDataStageForNode` writes this
    /// straight into a datastage extent (`:3660-3662`), so a fabricated `1` sizes the buffer wrong.
    #[must_use]
    pub fn parametric_stride(&self, dsc: &(impl LdsSticks + Dsc + ?Sized)) -> Option<Elements> {
        let lds = self.parametric_lds?;
        let stick_sizes = cumulative_stick_sizes(&dsc.stick_dims(lds), StickPart::Whole)?;
        // `dims_[0]`, which this body indexes unguarded — `parametricIterCount` is where the
        // reference states the arity, `DT_ERROR`ing on `dims_.size() != 1` (`dsc/dsc2.cpp:4129`).
        let loop_dim = self.dims.first()?.dim;
        if let Some(&(_, stride)) = stick_sizes.iter().find(|&&(dim, _)| dim == loop_dim) {
            return Some(stride);
        }
        // Check if the dim at least exists in case this is not in stick dims.
        dsc.layout_dims(lds)
            .iter()
            .any(|dim| dim == loop_dim)
            .then_some(Elements(1))
    }
}

#[cfg(test)]
mod tests_e014 {
    //! ⭐⭐ THE TWO STRIDES THE REFERENCE ITSELF WROTE — `g0/debug/sdsc_14/sdsc.json`, the ONE g0
    //! program of 187 that holds a parametric loop.
    //!
    //! `ddc/ddcv1.cpp:2468` assigns `loopEleOffs = loopPtr->parametricStride(currDsc)`, so the export
    //! carries this function's own answer: the `transfer_lds1_src:sfp_dst:lxsu` node's
    //! `dstLdsAndLoopOffsets_[0].loopEleOffsets_["0"]` reads
    //! `"parametric_loop_out(padded)__2": {"out": 128}` and `"parametric_loop_mb(padded)": {"mb": 1}`.
    //!
    //! Both of those loops carry `parametricLdsIdx_ = 1`; lds 1 is `dsType_ = OUTPUT`, whose
    //! `primaryDsInfo_` states `stickDimOrder_ = ["out"]` with `stickSize_ = [128]` and
    //! `layoutDimOrder_ = ["mb", "out", "y"]` — the order `allocate-Tensor1_hbm` and
    //! `allocate_lds1_lx` BOTH carry, which is what `getLayoutDims(1)` answers.
    //!
    //! ⭐ SO ONE PROGRAM EXERCISES BOTH LIVE ARMS: `out` is the stick dim and takes 128, while `mb` is
    //! not a stick dim at all and takes the `1` its place in the layout order earns it.

    use super::{
        BlockNode, Dsc, Elements, LayoutDims, LdsIdx, LdsSticks, LoopDim, LoopNode, MetaDimKind,
        NodeBase, NodeName, PrimaryDim,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims;

    /// `sdsc_14`'s lds 1, through the two seams this reads a DSC by.
    struct Sdsc14;

    impl LdsSticks for Sdsc14 {
        fn stick_dims(&self, _lds: LdsIdx) -> StickDims {
            StickDims(vec![(PrimaryDim::Out, Elements(128))])
        }
    }

    impl Dsc for Sdsc14 {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            LayoutDims::new(PrimaryDim::Mb, vec![PrimaryDim::Out, PrimaryDim::Y])
        }
    }

    /// One `parametric_loop_<dim>(padded)`, with `numId_ = denId_ = -1` as its producer writes them.
    fn parametric(dim: PrimaryDim, lds: Option<u32>) -> LoopNode {
        LoopNode {
            dims: vec![LoopDim {
                dim,
                kind: MetaDimKind::Padded,
            }],
            parametric_lds: lds.map(LdsIdx),
            ..LoopNode::bare(BlockNode {
                base: NodeBase::named(NodeName(format!("parametric_loop_{}(padded)", dim.spelling()))),
                children: Vec::new(),
            })
        }
    }

    /// e014 — the two strides `sdsc_14`'s own `loopEleOffsets_` records.
    #[test]
    fn a_parametric_loops_stride_is_its_dims_cumulative_stick_size() {
        // `"parametric_loop_out(padded)__2": {"out": 128}`.
        assert_eq!(
            parametric(PrimaryDim::Out, Some(1)).parametric_stride(&Sdsc14),
            Some(Elements(128))
        );
        // `"parametric_loop_mb(padded)": {"mb": 1}` — the layout arm, so 128 must NOT appear here.
        assert_eq!(
            parametric(PrimaryDim::Mb, Some(1)).parametric_stride(&Sdsc14),
            Some(Elements(1))
        );
    }

    /// e014 — the two stops, and neither of them is a stride of `1`.
    #[test]
    fn a_dim_outside_both_sets_and_an_unset_index_are_stops() {
        // `DT_ERROR` (`dsc/dsc2.cpp:4213-4216`): `x` is neither lds 1's stick dim nor in its
        // `["mb", "out", "y"]` layout order, so the `int loopStride = 1` initializer never returns.
        assert_eq!(
            parametric(PrimaryDim::X, Some(1)).parametric_stride(&Sdsc14),
            None
        );
        // `DT_CHECK_MSG(ldsIdx != -1)` (`:4199-4202`) — every other loop in every other g0 program.
        assert_eq!(
            parametric(PrimaryDim::Out, None).parametric_stride(&Sdsc14),
            None
        );
    }
}

#[cfg(test)]
mod tests_e011 {
    //! ⭐ A LOOP'S COUNT IS SYMBOLIC PER DIM — `loopCountSymbolIds_` (`dsc/dsc2.h:576-578`) is a map
    //! from dim to a LIST of symbols, *"single value for pure symbolic and pivot dims, multiple
    //! entries (max-pivot) for irregular dims"*.
    //!
    //! ⭐ AND THE TWO QUESTIONS THE REFERENCE ASKS READ DIFFERENT HALVES: `isDimSymbolic`
    //! (`dsc/dsc2.h:595-597`) reads that map, while `hasLoopDim` (`dsc/dsc2.cpp:4223-4228`) scans
    //! `dims_` comparing the `dim_` half of each pair only.

    use super::{
        BlockNode, LoopDim, LoopNode, MetaDimKind, NodeBase, NodeName, PrimaryDim, VariableSymbol,
    };

    /// e011 — the map answers per dim, and `hasLoopDim` is blind to the `kind_` half of the pair.
    #[test]
    fn a_loops_count_is_symbolic_only_on_the_dims_its_symbol_map_names() {
        let mut held = LoopNode::bare(BlockNode {
            base: NodeBase::named(NodeName("loop_ds0_ds1_out_y".to_owned())),
            children: Vec::new(),
        });
        held.dims = vec![
            LoopDim {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::Unpadded,
            },
            LoopDim {
                dim: PrimaryDim::Y,
                kind: MetaDimKind::Padded,
            },
        ];
        // A max-pivot dim carries MORE THAN ONE symbol, which is why the value is a list.
        held.loop_count_symbol_ids
            .insert(PrimaryDim::Out, vec![VariableSymbol(7), VariableSymbol(8)]);

        assert!(held.is_dim_symbolic(PrimaryDim::Out));
        assert_eq!(
            held.loop_count_symbol_ids
                .get(&PrimaryDim::Out)
                .map(Vec::len),
            Some(2)
        );
        // ⛔ AN ABSENT DIM IS `count(dim) == 0`, i.e. a COUNTED loop, not a missing fact: `y` is a dim
        // of this loop and still has no symbolic bound.
        assert!(!held.is_dim_symbolic(PrimaryDim::Y));
        assert!(
            held.has_loop_dim(PrimaryDim::Y),
            "padded, and still `dims_.first == y`"
        );
        assert!(!held.has_loop_dim(PrimaryDim::X));
    }
}

#[cfg(test)]
mod tests_e012 {
    //! ⭐ `next_` IS THE THEN REGION THEN THE ELSE REGION — `addThenRegion` `DT_ERROR`s on a non-empty
    //! `next_`, `addElseRegion` on an empty one *"ConditionNode does not have a 'then' region"* and on
    //! a full one (`dsc/dsc2.cpp:2153-2167`), so ORDER IS THE ONLY WAY IN.

    use std::collections::BTreeMap;

    use super::{BlockNode, CondRegions, ConditionNode, LoopCondComposite, NodeBase, NodeName};

    /// One region block, named as `coordinate_masking` names them (`ddc/ddcv1.cpp:3539-3560`).
    fn region(name: &str) -> BlockNode {
        BlockNode {
            base: NodeBase::named(NodeName(name.to_owned())),
            children: Vec::new(),
        }
    }

    /// e012 — two regions land in order, and the third is the one refusal left.
    #[test]
    fn a_condition_takes_a_then_region_then_an_else_and_refuses_a_third() {
        let mut held = ConditionNode {
            base: NodeBase::named(NodeName("condition_SAMV_dim_out".to_owned())),
            loop_cond: LoopCondComposite::default(),
            core_cl_cond: BTreeMap::new(),
            next: CondRegions::Empty,
        };
        assert_eq!(
            held.next.then_branch(),
            None,
            "`new ConditionNode()` has no region"
        );

        assert_eq!(held.add_region(region("block_SAMV_dim_out")), Some(()));
        assert_eq!(
            held.next.then_branch(),
            Some(&region("block_SAMV_dim_out")),
            "the first region added IS `getThenBranchNode()`"
        );
        assert_eq!(held.next.else_branch(), None);

        assert_eq!(held.add_region(region("block_SAMV_reset")), Some(()));
        assert_eq!(held.next.else_branch(), Some(&region("block_SAMV_reset")));
        assert_eq!(held.next.regions().len(), 2);

        // ⛔ *"ConditionNode only accepts 2 BlockNodes as children"* (`dsc/dsc2.cpp:2143-2146`).
        assert_eq!(held.add_region(region("block_third")), None);
        assert_eq!(
            held.next.regions().len(),
            2,
            "the refusal left both regions where they were"
        );
    }
}

#[cfg(test)]
mod tests_e007_data_info {
    //! WHAT AN OPERAND KNOWS ABOUT ITS DATA — [`DataInfo`] (`dsc/dsc2.h:721-739`), whose ten declared
    //! fields this batch put where the reference declares them. Two facts here a field list cannot
    //! state on its own: a default-constructed one is absent in every field, and the three nested maps
    //! keep the reference's OWN key ladders, which are not the same ladder — `constEleOffsets_` is
    //! core-outermost (`dsc/dsc2.h:727-729`) and `loopEleOffsets_` is corelet-outermost (`:730-734`),
    //! the one asymmetry a reader is likely to normalise away.

    use super::{
        Bytes, ConstEleOffset, Core, Corelet, DataInfo, DataLocation, LdsIdx, LoopEleOffset,
        LoopId, NodeId, PrimaryDim, SenComponent, StartAddress, Via,
    };

    fn core0() -> Core {
        Core::checked(0).expect("core 0")
    }

    fn cl0() -> Corelet {
        Corelet::at::<0>()
    }

    /// e007 — `DataInfo()` leaves all ten fields at the reference's own default.
    #[test]
    fn a_default_data_info_is_absent_in_all_ten_fields() {
        let fresh = DataInfo::default();
        assert_eq!(
            fresh,
            DataInfo::EMPTY,
            "`Default` and `EMPTY` are one value"
        );

        assert_eq!(
            fresh.data_connect, None,
            "`dataConnect_` is the empty string"
        );
        assert_eq!(fresh.my_lds_idx, None, "`myLdsIdx_ = -1`");
        assert_eq!(fresh.constant_id, None, "`constantId_ = -1`");
        assert_eq!(fresh.latch_data_id, None, "`latchDataId_ = -1`");
        assert_eq!(
            fresh.start_addr,
            StartAddress::EMPTY,
            "`startAddr_` folds nothing"
        );
        assert!(
            !fresh.is_start_addr_symbolic,
            "`isStartAddrSymbolic_ = false`"
        );
        assert!(fresh.const_ele_offsets.is_empty(), "`constEleOffsets_`");
        assert!(fresh.loop_ele_offsets.is_empty(), "`loopEleOffsets_`");
        assert!(fresh.buffer_addr_offset.is_empty(), "`bufferAddrOffset_`");
        assert_eq!(
            fresh.buffer_switch_position, None,
            "`bufferSwitchPosition_ = nullptr`"
        );

        // ⛔ NEITHER, so the reference's *"Cannot be both labeledDs and constant."* never arises.
        assert!(
            !fresh.is_labeled_ds(),
            "`isLabeledDs()` on `myLdsIdx_ = -1`"
        );
        assert!(!fresh.is_constant(), "`isConstant()` on `constantId_ = -1`");
    }

    /// e007 — the three nested maps keep the reference's own key ladders, one of which skips the core.
    #[test]
    fn the_offset_maps_keep_the_references_own_key_ladders() {
        let switch_at = LoopId(NodeId(7));
        let mut data = DataInfo::EMPTY;

        data.const_ele_offsets
            .entry(core0())
            .or_default()
            .entry(cl0())
            .or_default()
            .insert(PrimaryDim::Mb, ConstEleOffset(4));
        data.loop_ele_offsets
            .entry(cl0())
            .or_default()
            .entry(switch_at)
            .or_default()
            .insert(PrimaryDim::Mb, LoopEleOffset(1));
        data.buffer_addr_offset
            .entry(core0())
            .or_default()
            .insert(cl0(), Bytes(4096));

        // `constEleOffsets_` (`dsc/dsc2.h:727-729`) — core OUTERMOST, then corelet, then the dim.
        assert_eq!(
            data.const_ele_offsets[&core0()][&cl0()][&PrimaryDim::Mb],
            ConstEleOffset(4)
        );
        assert_eq!(
            data.const_ele_offsets[&core0()].len(),
            1,
            "one corelet stated under core 0"
        );

        // `loopEleOffsets_` (`:730-734`) — CORELET outermost, with NO core level to state at all.
        assert_eq!(
            data.loop_ele_offsets[&cl0()][&switch_at][&PrimaryDim::Mb],
            LoopEleOffset(1),
            "*\"put 1 if popping next element\"*"
        );
        assert_eq!(
            data.loop_ele_offsets[&cl0()].len(),
            1,
            "one loop stated under corelet 0"
        );

        // `bufferAddrOffset_` (`:735-737`) — core then corelet, and the leaf is the address step.
        assert_eq!(data.buffer_addr_offset[&core0()][&cl0()], Bytes(4096));

        // A dim no offset is stated for is ABSENT, which is not the same answer as an offset of zero.
        assert!(
            !data.const_ele_offsets[&core0()][&cl0()].contains_key(&PrimaryDim::I),
            "`unordered_map::count` on a dim the stage never offset"
        );
    }

    /// e007 — a freshly minted end states `myLdsIdx_` and nothing else, and reading it back is the
    /// same end.
    #[test]
    fn a_via_states_only_its_lds_and_round_trips_through_an_operand() {
        let via = Via {
            loc: DataLocation {
                unit: SenComponent::Lxlu,
                storage: SenComponent::Lx,
            },
            lds: Some(LdsIdx(3)),
        };

        let operand = via.operand();
        assert_eq!(operand.unit, SenComponent::Lxlu);
        assert_eq!(operand.storage, SenComponent::Lx);
        assert!(
            operand.data.is_labeled_ds(),
            "an lds-carrying end IS a labeled ds"
        );
        assert_eq!(
            operand.data,
            DataInfo {
                my_lds_idx: Some(LdsIdx(3)),
                ..DataInfo::EMPTY
            },
            "a minted end states the lds and leaves the other nine fields default"
        );

        assert_eq!(
            Via::of(&operand),
            via,
            "a `dstVias_` entry read back off an operand is the end it was minted from"
        );
    }
}

#[cfg(test)]
mod tests_e009_schedule_node {
    use super::*;

    /// `getOwnerLoop()` (`dsc/dsc2.cpp:1896`) AND `getParentDimLoop()` (`:1906`) WALKED DOWN over a
    /// nest whose inner loop tiles `Y` and whose outer one tiles `X`: a leaf's parent is the block it
    /// hangs off, its owner loop is the innermost enclosing one, and asking for `X` continues past
    /// that loop to the one that tiles it.
    #[test]
    fn a_prev_chain_answers_the_owner_loop_and_the_dim_loop() {
        let leaf = SchedNode::Leaf(LeafNode::new(
            LeafKind::Transfer,
            NodeName("transfer".to_owned()),
        ));
        let mut inner = LoopNode::bare(BlockNode {
            base: NodeBase::named(NodeName("loop_y".to_owned())),
            children: vec![SchedNode::Block(BlockNode {
                base: NodeBase::named(NodeName("block".to_owned())),
                children: vec![leaf],
            })],
        });
        inner.dims.push(LoopDim {
            dim: PrimaryDim::Y,
            kind: MetaDimKind::Unpadded,
        });
        let mut outer = LoopNode::bare(BlockNode {
            base: NodeBase::named(NodeName("loop_x".to_owned())),
            children: vec![SchedNode::Loop(Box::new(inner))],
        });
        outer.dims.push(LoopDim {
            dim: PrimaryDim::X,
            kind: MetaDimKind::Unpadded,
        });
        let tree = ScheduleTree::new(BlockNode {
            base: NodeBase::named(NodeName("head".to_owned())),
            children: vec![SchedNode::Loop(Box::new(outer))],
        });

        let found = tree
            .ancestors(&NodeName("transfer".to_owned()))
            .expect("the transfer is in the tree");
        assert_eq!(found.parent.base.name, NodeName("block".to_owned()));
        assert_eq!(
            found.owner_loop().block.base.name,
            NodeName("loop_y".to_owned())
        );
        assert_eq!(
            found.parent_dim_loop(PrimaryDim::X).map(|held| &held.block.base.name),
            Some(&NodeName("loop_x".to_owned()))
        );
        // `getParentDimLoop` runs out of loops rather than refusing.
        assert!(found.parent_dim_loop(PrimaryDim::I).is_none());
        // ⭐ A top-level child's owner loop is the SENTINEL, which this tree named "head".
        assert_eq!(
            tree.ancestors(&NodeName("loop_x".to_owned()))
                .expect("the outer loop is in the tree")
                .owner_loop()
                .block
                .base
                .name,
            NodeName("head".to_owned())
        );
    }

    /// `getSizesForCoreId` (`dsc/dsc2.cpp:2398-2405`) OVER ALL THREE OF ITS ARMS: a core with its
    /// own gapped list gets it, a core without one falls back to the shared `-1` entry, and a view
    /// that never ran the gap pass falls back to `sizesNoGaps_`.
    #[test]
    fn a_cores_view_sizes_fall_back_through_the_shared_entry_to_the_gapless_list() {
        let zero = Core::checked(0).expect("core 0");
        let one = Core::checked(1).expect("core 1");
        let entry = |size| {
            [Size {
                dim: PrimaryDim::X,
                size: Elements(size),
            }]
        };

        let mut view = UnitView {
            sizes_no_gaps: entry(4).to_vec(),
            ..UnitView::default()
        };
        // `sizesWithGaps_` is empty until the gap pass runs, and every core reads the gapless list.
        assert_eq!(view.sizes_for_core(Some(zero)), entry(4).as_slice());

        view.sizes_with_gaps.insert(None, entry(8).to_vec());
        assert_eq!(
            view.sizes_for_core(Some(one)),
            entry(8).as_slice(),
            "the shared `-1` entry answers for a core with none of its own"
        );

        view.sizes_with_gaps.insert(Some(zero), entry(12).to_vec());
        assert_eq!(view.sizes_for_core(Some(zero)), entry(12).as_slice());
        // ⭐ One core's own list does NOT displace the shared one for the others.
        assert_eq!(view.sizes_for_core(Some(one)), entry(8).as_slice());
    }

    /// `isNodeRelevant` (`dsc/dsc2.cpp:1916`), `getRelevantCoreCl` (`:1934`) and `getRelevantComps`
    /// (`:1949`) OVER ONE MARKED CORELET: the component, core and corelet that were marked answer
    /// yes, a neighbouring corelet answers no, and each of the two summaries names what was marked.
    #[test]
    fn a_marked_corelet_is_the_only_one_the_three_relevance_reads_report() {
        let core = Core::checked(0).expect("core 0");
        let corelet = Corelet::at::<0>();
        let mut comps = RelevantComps::default();
        assert!(comps.is_empty());
        comps.mark(SenComponent::L3lu, core, corelet);

        assert!(comps.is_node_relevant(Relevance::Any));
        assert!(comps.is_node_relevant(Relevance::CompCoreCl(
            SenComponent::L3lu,
            core,
            corelet
        )));
        assert!(!comps.is_node_relevant(Relevance::Comp(SenComponent::Lxlu0)));
        assert!(
            !comps.is_node_relevant(Relevance::CompCore(
                SenComponent::L3lu,
                Core::checked(1).expect("core 1")
            ))
        );

        assert_eq!(
            comps.relevant_core_cl(None),
            BTreeMap::from([(core, BTreeSet::from([corelet]))])
        );
        assert_eq!(
            comps.relevant_core_cl(Some(SenComponent::Lxlu0)),
            BTreeMap::new()
        );
        assert_eq!(
            comps.relevant_comps(CoreSel::Core(core)),
            BTreeSet::from([SenComponent::L3lu])
        );
        assert_eq!(comps.relevant_comps(CoreSel::Any).len(), 1);

        comps.forget(SenComponent::L3lu);
        assert!(comps.is_empty());
    }
}

#[cfg(test)]
mod tests_e010_block_node {
    //! `dsc2::BlockNode` (`dsc/dsc2.h:526-561`) and its ONE declared field. What a field list cannot
    //! state on its own: `next_` is EMPTY on a fresh block and keeps insertion order, which is the
    //! order every `getNextView` and `siblingRefNode` read is answered off.

    use super::*;

    #[test]
    fn a_block_owns_its_children_in_the_order_they_were_added() {
        let mut block = BlockNode::default();
        assert!(
            block.children.is_empty(),
            "`new dsc2::BlockNode()` leaves `next_` empty (`dsc/dsc2.h:554`)"
        );

        for name in ["t0", "c0", "t1"] {
            block.add_child(SchedNode::Leaf(LeafNode::new(
                LeafKind::Transfer,
                NodeName(name.to_owned()),
            )));
        }
        assert_eq!(
            block
                .children
                .iter()
                .map(|child| child.name().0.as_str())
                .collect::<Vec<_>>(),
            ["t0", "c0", "t1"],
            "`addChildNode(node, /*addBefore=*/false)` appends"
        );

        let at = block
            .child_pos(&NodeName("c0".to_owned()))
            .expect("`c0` is a child");
        block.insert_after(
            at,
            SchedNode::Leaf(LeafNode::new(
                LeafKind::Compute,
                NodeName("c1".to_owned()),
            )),
        );
        assert_eq!(
            block
                .children
                .iter()
                .map(|child| child.name().0.as_str())
                .collect::<Vec<_>>(),
            ["t0", "c0", "c1", "t1"],
            "`siblingRefNode` lands the new node immediately after the one it names"
        );
        assert!(
            block.child_pos(&NodeName("absent".to_owned())).is_none(),
            "a name no child carries runs out rather than refusing"
        );
    }

    /// ⭐ THE FRONT INSERT IS THE ALLOCATION CASE — `allocParent->addChildNode(myAllocNode,
    /// allocParent != currParent)` (`ddc/ddl/ddl_conversion.cpp:782`) — and the deep `Clone` beside it
    /// is a SNAPSHOT of an owned subtree, ⛔ NOT the reference's `clone()`, whose copy constructor
    /// drops every child (`dsc/dsc2.h:533-535`, `util/utils.h:105-107`).
    #[test]
    fn a_front_insert_precedes_every_child_already_there_and_clone_keeps_them_all() {
        let leaf = |name: &str| {
            SchedNode::Leaf(LeafNode::new(
                LeafKind::Transfer,
                NodeName(name.to_owned()),
            ))
        };
        let mut block = BlockNode::default();
        for name in ["t0", "t1"] {
            block.add_child(leaf(name));
        }
        assert_eq!(
            block.add_child_front(leaf("alloc")),
            ChildPos(0),
            "`addChildNode(node, /*addBefore=*/true)` lands in front of the transfers already there"
        );

        let names = |block: &BlockNode| {
            block
                .children
                .iter()
                .map(|child| child.name().0.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&block), ["alloc", "t0", "t1"]);

        let snapshot = block.clone();
        block.add_child(leaf("t2"));
        assert_eq!(
            names(&snapshot),
            ["alloc", "t0", "t1"],
            "`Clone` copied `next_` and the copy is INDEPENDENT of the block it came from"
        );
        assert_eq!(names(&block), ["alloc", "t0", "t1", "t2"]);
    }
}

#[cfg(test)]
mod tests_e013_sync_node {
    //! `dsc2::SyncNode` (`dsc/dsc2.h:964-972`) and its five declared fields. What the field list
    //! cannot state: a fresh sync is a HARD SEND standing in for no transfer with no other end bound,
    //! and `units_` cannot be empty at all — [`SyncUnits`] takes a first component by value, so the
    //! reference's *"all to all signals"* set with nobody in it is unspellable.

    use super::*;

    #[test]
    fn a_fresh_sync_is_a_hard_send_with_no_partner_bound() {
        let mut sync = SyncNode {
            base: NodeBase::named(NodeName("sync0".to_owned())),
            units: SyncUnits::new(SenComponent::L0lu, [SenComponent::Ptrow0]),
            direction: SyncDirection::Send,
            strength: SyncStrength::Hard,
            implicit_sync_ref_transfer: None,
            other_ends: Vec::new(),
        };
        assert_eq!(
            sync.units.iter().collect::<Vec<_>>().len(),
            2,
            "`units_` holds every component the sync signals between"
        );
        assert!(
            sync.implicit_sync_ref_transfer.is_none(),
            "`implicitSyncRefTransfer_ = nullptr` (`dsc/dsc2.h:968`)"
        );
        assert!(
            sync.other_ends.is_empty(),
            "`otherEndOfTheSignals_` is empty until a sequence pairs both ends"
        );

        // Entry 261 binds the transfer this sync stands in for, and pairs the two ends by name.
        sync.implicit_sync_ref_transfer = Some(NodeId(4));
        sync.other_ends.push(NodeName("sync1".to_owned()));
        let receive = SyncNode {
            base: NodeBase::named(NodeName("sync1".to_owned())),
            direction: SyncDirection::Receive,
            strength: SyncStrength::Soft,
            other_ends: vec![sync.base.name.clone()],
            ..sync.clone()
        };
        assert_eq!(receive.other_ends, vec![NodeName("sync0".to_owned())]);
        assert_ne!(
            receive.direction, sync.direction,
            "`isReceive_` is the one flag that distinguishes the two ends"
        );
        assert_eq!(receive.implicit_sync_ref_transfer, Some(NodeId(4)));
    }
}

#[cfg(test)]
mod tests_e014_transfer_node {
    //! `dsc2::TransferNode` (`dsc/dsc2.h:814-898`) and its twenty declared fields. What the field
    //! list cannot state: the reference's own non-zero initialisers (`replicationFactor_ = 1`,
    //! `unitTimeTransferNumChunks_ = 1`, `srcRep_` with no initialiser at all), that
    //! `rotateNumElements_ = 0` and *"no rotation"* are ONE state, and that `coreletViews_` answers a
    //! corelet-less unit off its FIRST entry rather than off a `-1` key.

    use super::*;

    fn operand(unit: SenComponent, storage: SenComponent) -> Operand {
        Operand {
            unit,
            storage,
            data: DataInfo::EMPTY,
        }
    }

    fn fresh() -> TransferNode {
        TransferNode {
            name: NodeName("t0".to_owned()),
            src: operand(SenComponent::L0lu, SenComponent::L0),
            dsts: Dsts::new(operand(SenComponent::Ptrow0, SenComponent::NoComponent), Vec::new()),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: None,
            padding: TransferPadding::default(),
            repetition: TransferRepetition::default(),
            src_indirect: None,
            dst_indirect: None,
            last_fusable_parent_loop_src: None,
            last_fusable_parent_loop_dst: Vec::new(),
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            corelet_views: BTreeMap::new(),
            transfer_coordinates: Coordinate::default(),
        }
    }

    #[test]
    fn a_fresh_transfer_carries_the_references_own_initialisers() {
        let mut node = fresh();
        assert_eq!(
            node.replication_factor,
            ReplicationFactor::ONE,
            "`replicationFactor_ = 1` (`dsc/dsc2.h:834`)"
        );
        assert_eq!(
            node.unit_time_transfer_num_chunks,
            NumChunks::ONE,
            "`unitTimeTransferNumChunks_ = 1` (`:837`)"
        );
        assert_eq!(
            node.repetition.src,
            OperandRepetition::ONE,
            "`srcRep_` has NO initialiser, and its writer's fallback is 1"
        );
        assert!(
            node.repetition.dsts.is_empty(),
            "`dstReps_` gains one entry per destination the DDL conversion walks"
        );
        assert!(
            node.rotate_num_elements.is_none(),
            "`rotateNumElements_ = 0` (`:839`) IS *no rotation*"
        );
        assert!(
            node.unit_time_transfer_chunk_stride.is_empty(),
            "a contiguous unit-time transfer skips nothing (`:838`)"
        );
        assert!(
            node.last_fusable_parent_loop_src.is_none(),
            "`lastFusableParentLoopSrc_ = nullptr` (`:830`)"
        );
        assert!(
            node.corelet_view(None).is_none(),
            "`coreletViews_` is empty until the tree is finalised"
        );

        // `dsc/dsc2.cpp:3061-3082` fills one view per corelet, source-then-destinations, and the
        // indirect halves stay default because neither end is indirect.
        for (idx, corelet) in [Corelet::at::<0>(), Corelet::at::<1>()].into_iter().enumerate() {
            node.corelet_views.insert(
                corelet,
                TransferCoreletView {
                    src_loops_and_size: UnitView {
                        sizes_no_gaps: vec![Size {
                            dim: PrimaryDim::Mb,
                            size: Elements(4 + idx as u64),
                        }],
                        ..UnitView::default()
                    },
                    dst_loops_and_sizes: vec![UnitView::default()],
                    ..TransferCoreletView::default()
                },
            );
        }
        node.last_fusable_parent_loop_dst
            .push(Some(LoopId(NodeId(9))));
        assert_eq!(
            node.last_fusable_parent_loop_dst.len(),
            1,
            "one entry per destination, which `dsc/dsc2.cpp:3074` indexes by the same `i`"
        );
        assert_eq!(
            node.corelet_view(Some(Corelet::at::<1>()))
                .expect("corelet 1 was filled")
                .src_loops_and_size
                .sizes_no_gaps[0]
                .size,
            Elements(5),
            "`coreletViews_.at(corelet_id_)`"
        );
        // ⭐ A CORELET-LESS UNIT READS THE FIRST ENTRY, not a `-1` key — `coreletViews_.begin()`
        // (`SNTransferLowering.cpp:861`).
        assert_eq!(
            node.corelet_view(None).expect("some view was filled"),
            node.corelet_view(Some(Corelet::at::<0>()))
                .expect("corelet 0 was filled")
        );
        assert!(
            node.corelet_view(None)
                .is_some_and(|view| view.src_indirect_loops_and_size == UnitView::default()
                    && view.dst_indirect_loops_and_sizes.is_empty()),
            "an indirect view is filled only for an indirect end (`dsc/dsc2.cpp:3066`, `:3075`)"
        );
    }

    /// ⛔ AN INDIRECT END IS A WHOLE `DataInfo` AND NOT AN LDS INDEX: `fillDataInfo` writes
    /// `isStartAddrSymbolic_` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5916`) and `loopEleOffsets_`
    /// (`:6182`) into `srcIndirectLdsAndLoopOffsets_` beside the `myLdsIdx_` entry 227 set, and
    /// `buildUnitView(tn->srcIndirect_, tn->srcIndirectLdsAndLoopOffsets_, ..)` (`dsc/dsc2.cpp
    /// :3067-3069`) reads the record back WHOLE.
    #[test]
    fn an_indirect_end_carries_every_field_fill_data_info_writes_into_it() {
        let mut node = fresh();
        assert!(
            node.src_indirect.is_none() && node.dst_indirect.is_none(),
            "`srcIndirect_.unit_ == NO_COMPONENT` IS *not indirect* (`dsc/dsc2.h:877`)"
        );

        // Entry 227's own three writes (`L3DlOpsScheduler.cpp:7135-7137`), then entry 333's two.
        let mut indirect = operand(SenComponent::L3lu, SenComponent::L3luibr);
        indirect.data.my_lds_idx = Some(LdsIdx(5));
        indirect.data.is_start_addr_symbolic = true;
        indirect
            .data
            .loop_ele_offsets
            .entry(Corelet::at::<0>())
            .or_default()
            .entry(LoopId(NodeId(9)))
            .or_default()
            .insert(PrimaryDim::Mb, LoopEleOffset(1));
        node.src_indirect = Some(indirect);

        let held = node.src_indirect.as_ref().expect("the source is indirect");
        assert_eq!(
            Via::of(held),
            Via {
                loc: DataLocation {
                    unit: SenComponent::L3lu,
                    storage: SenComponent::L3luibr,
                },
                lds: Some(LdsIdx(5)),
            },
            "`srcIndirect_` and its `myLdsIdx_` are still the pair a minting site hands in"
        );
        assert!(
            held.data.is_start_addr_symbolic,
            "`indirectDi->isStartAddrSymbolic_ = true` (`:5916`) has somewhere to land"
        );
        assert_eq!(
            held.data.loop_ele_offsets[&Corelet::at::<0>()][&LoopId(NodeId(9))][&PrimaryDim::Mb],
            LoopEleOffset(1),
            "`indirectDi->loopEleOffsets_[clId][loopPtr][dim] = loopEleOffs` (`:6182`)"
        );
    }
}

#[cfg(test)]
mod tests_e015_compute_node {
    //! `dsc2::ComputeNode` (`dsc/dsc2.h:900-962`) and its fourteen declared fields. What the field
    //! list cannot state: `dataFormat_`'s initialiser is `SEN169_FP16` and NOT `INVALID`, so an
    //! unstated format still has a value; `repetitionWithOffset_`'s two lists are BOTH empty, which is
    //! the *"no operand takes one"* entry 108 counts; and `computeMaskLoopOffsets_` empty is STATIC
    //! MASKING rather than a missing value.

    use super::*;

    fn fresh() -> ComputeNode {
        ComputeNode {
            name: NodeName("c0".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Ptrow0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: Coordinate::default(),
            repetition_with_offset: RepetitionWithOffset::default(),
        }
    }

    #[test]
    fn a_fresh_compute_carries_the_references_own_initialisers() {
        let mut node = fresh();
        assert!(
            !node.is_opaque_op,
            "`isOpaqueOp_ = false` (`dsc/dsc2.h:941`)"
        );
        assert_eq!(
            node.effective_data_format(),
            DataFormat::Sen169Fp16,
            "`dataFormat_ = DataFormats::SEN169_FP16` (`:934`) — an unstated format is FP16, not absent"
        );
        assert_eq!(
            node.instr_attribute.repetition,
            Repetition::ALL_SLICES,
            "`repetition_ = 8` (`:907`), *default 8 slices works the same*"
        );
        assert!(
            node.instr_attribute.compute_mask_loop_offsets.is_empty(),
            "empty `computeMaskLoopOffsets_` is STATIC masking (`ddc/ddcv1.cpp:1696-1698`)"
        );
        assert!(
            node.repetition_with_offset.for_inputs.is_empty()
                && node.repetition_with_offset.for_outputs.is_empty(),
            "`forInputs_ = {{}}` and `forOutputs_ = {{}}` (`:951-952`)"
        );
        assert!(
            node.input_coordinates.is_empty()
                && node.output_coordinate == Coordinate::default(),
            "no fold has been constructed on either side yet"
        );
        assert!(node.corelet_view(None).is_none());

        // Entry 109 writes one mask offset per corelet, loop and dim; entry 108 gives one output a
        // repetition offset of its own.
        node.instr_attribute
            .compute_mask_loop_offsets
            .entry(Corelet::at::<0>())
            .or_default()
            .entry(LoopId(NodeId(3)))
            .or_default()
            .insert(PrimaryDim::Mb, MaskLoopOffset::ADVANCE);
        node.repetition_with_offset
            .for_outputs
            .push(Repetition::ALL_SLICES);
        node.data_format = Some(DataFormat::IeeeFp32);
        node.corelet_views.insert(
            Corelet::at::<1>(),
            ComputeCoreletView {
                inputs_loops_and_sizes: vec![UnitView::default(), UnitView::default()],
                outputs_loops_and_sizes: vec![UnitView::default()],
            },
        );

        assert_eq!(
            node.instr_attribute.compute_mask_loop_offsets[&Corelet::at::<0>()]
                [&LoopId(NodeId(3))][&PrimaryDim::Mb],
            MaskLoopOffset::ADVANCE,
            "`computeMaskLoopOffsets_[corelet][loop][dim]` — the reference's own key ladder"
        );
        assert_eq!(
            node.repetition_with_offset.for_outputs.len(),
            1,
            "`forOutputs_.size()` is what entry 108 counts clones by"
        );
        assert_eq!(
            node.effective_data_format(),
            DataFormat::IeeeFp32,
            "a stated format is the one that is read back"
        );
        // ⭐ CORELET 1 IS THE ONLY ENTRY, so a corelet-less unit reads IT — `coreletViews_.begin()`
        // (`SNComputeLowering.cpp:549`).
        assert_eq!(
            node.corelet_view(None)
                .expect("corelet 1 was filled")
                .inputs_loops_and_sizes
                .len(),
            2
        );
        assert!(
            node.corelet_view(Some(Corelet::at::<0>())).is_none(),
            "an unfilled corelet is absent rather than empty"
        );
    }
}

#[cfg(test)]
mod tests_e016_schedule_tree {
    use super::*;

    /// `ScheduleTree()`, `empty()`, `clear()` and the sentinel head: a fresh tree is empty with
    /// `head_.denId_ = 0`, a top-level loop makes it non-empty, THAT loop's `getOwnerLoop()` is the
    /// sentinel and not `nullptr`, and `clear()` drops the children while keeping the head's `denId_`.
    #[test]
    fn a_fresh_tree_is_empty_and_its_sentinel_head_owns_every_top_level_node() {
        let mut tree = ScheduleTree::default();
        assert!(tree.is_empty());
        assert_eq!(tree.head_den(), Some(Metadata::CORE_DSTGID));
        assert_eq!(tree.head().block.base.name, NodeName::default());

        tree.head_mut()
            .block
            .add_child(SchedNode::Loop(Box::new(LoopNode::bare(BlockNode {
                base: NodeBase::named(NodeName("loop_ds0_ds1_y".to_owned())),
                children: Vec::new(),
            }))));
        assert!(!tree.is_empty());

        let found = tree
            .ancestors(&NodeName("loop_ds0_ds1_y".to_owned()))
            .expect("the loop just added");
        assert!(found.loops.is_empty(), "nothing encloses a top-level node");
        assert_eq!(found.owner_loop().den, Some(Metadata::CORE_DSTGID));
        // The sentinel tiles no dim, so the outward walk runs out at it.
        assert!(found.parent_dim_loop(PrimaryDim::Y).is_none());

        tree.set_head_den(DatastageId(3));
        tree.clear();
        assert!(tree.is_empty());
        assert_eq!(tree.head_den(), Some(DatastageId(3)));
    }
}

#[cfg(test)]
mod tests_e005_coordinate {
    //! HOW A COORDINATE IS BUILT AND UNBUILT — [`Coordinate`] (`dsc/dsc2.h:76`). Three facts here a
    //! field list cannot state on its own: `addFold(.., pos = 0)` inserts at the FRONT, so the fold
    //! added LAST is the outermost (`foldInfrastructure.h:1348`, `:1371`); `clearFoldForDim` leaves
    //! the dim COVERED and its padding standing — *"Note: Padding is not cleared."*
    //! (`dsc/dsc2.h:100-102`); and the one positional beta write is TOTAL where the reference's
    //! `getNumDims() - 1` underflows (`ddc/ddc_fold.cpp:4714-4716`).

    use super::{
        Coordinate, CoordinateCategory, Core, FoldCardinality, FoldCoeff, FoldDim, FoldLabel,
        FoldPosition, PadType, PaddingForm, PrimaryDim, WkSlice,
    };
    use crate::schedule::l3::dsc::WkSliceId;

    fn label(name: &str) -> FoldLabel {
        FoldLabel(name.to_owned())
    }

    /// `addFold(OUT, cat, card, name, 1, 0, 0)` — one fold onto the same dim every time.
    fn add(coord: &mut Coordinate, category: CoordinateCategory, card: u32, name: &str) {
        coord.add_fold_front(
            PrimaryDim::Out,
            category,
            FoldCardinality(card),
            label(name),
            FoldCoeff(1),
            FoldCoeff(0),
        );
    }

    /// e005 — front insertion, and each category counting ONLY its own arm.
    #[test]
    fn add_fold_front_puts_the_last_fold_added_outermost() {
        let mut coord = Coordinate::default();
        assert!(
            !coord.covers(PrimaryDim::Out),
            "`coordinates_.count(dim)` before the first fold"
        );

        add(&mut coord, CoordinateCategory::Temporal, 3, "inner");
        add(&mut coord, CoordinateCategory::Spatial, 2, "outer");

        let dim = coord.fold_dim(PrimaryDim::Out).expect("the dim is covered");
        assert_eq!(
            dim.folds().map(|one| one.label.clone()).collect::<Vec<_>>(),
            vec![label("outer"), label("inner")],
            "position 0 is the fold added LAST"
        );
        assert_eq!(
            dim.cardinality_at(FoldPosition::Core),
            Some(FoldCardinality(2)),
            "`getFoldDimSize(0)`"
        );
        assert_eq!(
            dim.cardinality_at(FoldPosition::Corelet),
            Some(FoldCardinality(3)),
            "`getFoldDimSize(1)`"
        );
        assert_eq!(
            dim.cardinality_at(FoldPosition::RowSplit),
            None,
            "the dim is only two folds deep"
        );

        assert_eq!(dim.spatial_folds(), 1, "`numOfSpatialFolds_`");
        assert_eq!(dim.temporal_folds(), 1, "`numOfTemporalFolds_`");
        assert_eq!(
            dim.elem_arr_folds(),
            0,
            "`numOfElemArrFolds_` is absent-is-zero, never bumped by another arm"
        );
    }

    /// e005 — `clearFoldForDim` resets the fold space in place; the dim and its padding survive.
    #[test]
    fn clear_fold_for_dim_keeps_the_dim_and_its_padding() {
        let mut coord = Coordinate::default();
        coord.set_padding(PrimaryDim::Out, PadType::PaddedWZeroPad);
        add(&mut coord, CoordinateCategory::ElemArr, 4, "elem");

        coord.clear_fold_for_dim(PrimaryDim::Out);

        assert!(
            coord.covers(PrimaryDim::Out),
            "the `coordinates_` entry is reset, not erased"
        );
        let dim = coord.fold_dim(PrimaryDim::Out).expect("still covered");
        assert_eq!(dim, &FoldDim::EMPTY, "`reset()` in place");
        assert_eq!(
            dim.elem_arr_folds(),
            0,
            "the count goes back with the folds it counted"
        );
        assert_eq!(
            coord.padding(PrimaryDim::Out),
            PadType::PaddedWZeroPad,
            "*\"Note: Padding is not cleared.\"*"
        );

        // ⛔ Clearing a dim the coordinate does not cover does not mint one.
        coord.clear_fold_for_dim(PrimaryDim::Mb);
        assert!(!coord.covers(PrimaryDim::Mb));
    }

    /// e005 — the positional beta write lands on the INNERMOST fold, and on none where there is none.
    #[test]
    fn add_to_innermost_beta_lands_on_the_last_fold_only() {
        let mut coord = Coordinate::default();
        add(&mut coord, CoordinateCategory::Temporal, 3, "inner");
        add(&mut coord, CoordinateCategory::Spatial, 2, "outer");

        coord.add_to_innermost_beta(PrimaryDim::Out, FoldCoeff(5));

        assert_eq!(
            coord
                .fold_dim(PrimaryDim::Out)
                .expect("covered")
                .folds()
                .map(|one| one.beta)
                .collect::<Vec<_>>(),
            vec![FoldCoeff(0), FoldCoeff(5)],
            "`getNumDims() - 1` is the innermost level, not position 0"
        );

        // ⛔ NO INNERMOST LEVEL: nothing to add to, and nothing to refuse either.
        coord.clear_fold_for_dim(PrimaryDim::Out);
        coord.add_to_innermost_beta(PrimaryDim::Out, FoldCoeff(7));
        coord.add_to_innermost_beta(PrimaryDim::Mb, FoldCoeff(7));
        assert_eq!(
            coord
                .fold_dim(PrimaryDim::Out)
                .expect("covered")
                .folds()
                .count(),
            0
        );
        assert!(!coord.covers(PrimaryDim::Mb));
    }

    /// e005 — `setPadding(const PaddingFormType&)` (`dsc/dsc2.h:240`) REPLACES the form, which the
    /// per-dim `setPadding` (`:236`) does not.
    #[test]
    fn setting_the_whole_padding_form_replaces_it() {
        let mut coord = Coordinate::default();
        coord.set_padding(PrimaryDim::Out, PadType::PaddedFullSpan);
        coord.set_padding(PrimaryDim::In, PadType::LoweredPadded);

        let mut replacement = PaddingForm::default();
        replacement.set_padding(PrimaryDim::In, PadType::PaddedNoZeroPad);
        coord.set_padding_form(replacement);

        assert_eq!(coord.padding(PrimaryDim::In), PadType::PaddedNoZeroPad);
        assert_eq!(
            coord.padding(PrimaryDim::Out),
            PadType::NoPad,
            "the old form is gone, not merged into the new one"
        );
        assert_eq!(
            coord.padding_form().stated().collect::<Vec<_>>(),
            vec![(PrimaryDim::In, PadType::PaddedNoZeroPad)],
            "`getPadding()` (`dsc/dsc2.h:245`) names only the dims the form states"
        );
    }

    /// e005 — `foldConstructed_` is the one-way latch its only setter makes it, and
    /// `coreIdToWkSlice_` is absent for a core it does not name.
    #[test]
    fn fold_construction_latches_and_wk_slices_are_stated_per_core() {
        let mut coord = Coordinate::default();
        assert!(!coord.fold_constructed(), "`foldConstructed_ = false`");
        coord.complete_fold_construction();
        coord.complete_fold_construction();
        assert!(coord.fold_constructed(), "the latch does not toggle back");

        let core = Core::checked(0).expect("core 0");
        assert!(
            coord.wk_slice_mut(core).is_none(),
            "*\"Core ID not found.\"*"
        );

        coord.set_wk_slice(core, WkSlice::default());
        coord
            .wk_slice_mut(core)
            .expect("stated now")
            .0
            .insert(PrimaryDim::Out, WkSliceId(2));

        assert_eq!(
            coord
                .wk_slices()
                .map(|(at, slice)| (at, slice.at(PrimaryDim::Out)))
                .collect::<Vec<_>>(),
            vec![(core, Some(WkSliceId(2)))],
            "the mutable borrow writes THIS coordinate's own map"
        );
    }
}

#[cfg(test)]
mod tests_e008_transfer_padding {
    //! A TRANSFER'S ZERO-PAD INFO — [`TransferPadding`] (`dsc/dsc2.h:755`). What its shape asserts:
    //! `isEmpty()` is the AND of both maps (`:774-776`), `getPadFrontOrBackDimsSet` returns a
    //! `std::set` so a dim built at BOTH ends is named once (`:783-788`), and the two
    //! `FoldDimPosition` slots are the whole fold space because `TOTAL_FOLDDIM_NUM` is the count `2`
    //! (`:768-772`).

    use super::{FoldCardinality, FoldCoeff, PadFold, PrimaryDim, TransferPadding, ZeroPadFolds};

    /// The `(sizes, alphas, betas)` triple `buildPadSizes` is handed, as the two slots index it.
    fn folds(work_slice: u32, chunk: u32) -> ZeroPadFolds {
        ZeroPadFolds {
            work_slice: PadFold {
                cardinality: FoldCardinality(work_slice),
                alpha: FoldCoeff(1),
                beta: FoldCoeff(0),
            },
            chunk: PadFold {
                cardinality: FoldCardinality(chunk),
                alpha: FoldCoeff(1),
                beta: FoldCoeff(0),
            },
        }
    }

    /// e008 — a default-constructed one states nothing at either end.
    #[test]
    fn a_default_transfer_padding_is_empty_at_both_ends() {
        let fresh = TransferPadding::default();
        assert!(fresh.is_empty(), "`isEmpty()` on two empty maps");
        assert_eq!(fresh.dims().count(), 0, "`getAllKeys()` on both is empty");
        assert_eq!(fresh.pad_front(PrimaryDim::Out), None);
        assert_eq!(fresh.pad_back(PrimaryDim::Out), None);
    }

    /// e008 — `isEmpty()` is an AND, so ONE built end already makes it false.
    #[test]
    fn one_built_end_is_enough_to_be_non_empty() {
        let mut front_only = TransferPadding::default();
        front_only.build_pad_front(PrimaryDim::Out, folds(2, 3));
        assert!(!front_only.is_empty(), "a front-only pad is not empty");

        let mut back_only = TransferPadding::default();
        back_only.build_pad_back(PrimaryDim::Out, folds(2, 3));
        assert!(!back_only.is_empty(), "nor is a back-only one");
    }

    /// e008 — the union names each dim ONCE, in `PrimaryDimTypes` order, and the two ends stay
    /// independent maps.
    #[test]
    fn dims_unions_both_ends_once_and_in_dim_order() {
        let mut padding = TransferPadding::default();
        padding.build_pad_front(PrimaryDim::Mb, folds(2, 3));
        padding.build_pad_front(PrimaryDim::In, folds(4, 5));
        padding.build_pad_back(PrimaryDim::In, folds(6, 7));
        padding.build_pad_back(PrimaryDim::Out, folds(8, 9));

        assert_eq!(
            padding.dims().collect::<Vec<_>>(),
            vec![PrimaryDim::In, PrimaryDim::Out, PrimaryDim::Mb],
            "`In` is built at BOTH ends and is still named once"
        );

        assert_eq!(
            padding.pad_back(PrimaryDim::Mb),
            None,
            "building the front end never states the back"
        );
        assert_eq!(padding.pad_front(PrimaryDim::Out), None);
        assert_eq!(padding.pad_front(PrimaryDim::In), Some(folds(4, 5)));
        assert_eq!(padding.pad_back(PrimaryDim::In), Some(folds(6, 7)));
    }

    /// e008 — both `FoldDimPosition` slots round-trip, work slice OUTER and chunk INNER, the order
    /// `buildTransferFoldDim` builds them in (`dsc/dsc2.cpp:4652-4654`).
    #[test]
    fn both_folddim_slots_round_trip_outer_to_inner() {
        let mut padding = TransferPadding::default();
        padding.build_pad_front(
            PrimaryDim::Out,
            ZeroPadFolds {
                work_slice: PadFold {
                    cardinality: FoldCardinality(2),
                    alpha: FoldCoeff(16),
                    beta: FoldCoeff(-3),
                },
                chunk: PadFold {
                    cardinality: FoldCardinality(5),
                    alpha: FoldCoeff(1),
                    beta: FoldCoeff(7),
                },
            },
        );

        let built = padding.pad_front(PrimaryDim::Out).expect("just built");
        assert_eq!(
            built.work_slice,
            PadFold {
                cardinality: FoldCardinality(2),
                alpha: FoldCoeff(16),
                beta: FoldCoeff(-3),
            },
            "`WORK_SLICE_FOLDDIM = 0`, the outer fold"
        );
        assert_eq!(
            built.chunk,
            PadFold {
                cardinality: FoldCardinality(5),
                alpha: FoldCoeff(1),
                beta: FoldCoeff(7),
            },
            "`CHUNK_FOLDDIM = 1`, the inner fold"
        );
    }

    /// e008 — a second build of one dim REPLACES the first, where the reference `DT_CHECK`s
    /// *"Expect empty fold properties."* (`dsc/dsc2.cpp:4662`) on a path entry 221 cannot reach
    /// twice.
    #[test]
    fn a_second_build_of_one_dim_replaces_the_first() {
        let mut padding = TransferPadding::default();
        padding.build_pad_front(PrimaryDim::Out, folds(2, 3));
        padding.build_pad_front(PrimaryDim::Out, folds(4, 5));

        assert_eq!(padding.pad_front(PrimaryDim::Out), Some(folds(4, 5)));
        assert_eq!(
            padding.dims().collect::<Vec<_>>(),
            vec![PrimaryDim::Out],
            "one entry per dim, never a second"
        );
    }
}
