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
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::v1::LdsSticks;
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
    spatial: u32,
    temporal: u32,
    elem_arr: u32,
}

impl FoldDim {
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

/// A COORDINATE — `CoordinateType<CoordinateBaseType>` (`dsc/dsc2.h:76`) narrowed to its
/// `coordinates_` map and the fold counts, which is what the fold builders read and write.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Coordinate {
    dims: BTreeMap<PrimaryDim, FoldDim>,
    padding: Padding,
    fold_constructed: bool,
    /// `coreIdToWkSlice_` (`dsc/dsc2.h:81`) — WHICH WORK SLICE EACH CORE TAKES per dim as THIS
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
    /// (`foldInfrastructure.h:1349`) and is the only position the fold builders in this module use.
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

    /// `clearFoldForDim(dim)` (`dsc/dsc2.h:99-114`) — empties the dim's folds and zeroes its three
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

    /// `getPadding(dim)` (`dsc/dsc2.h:241`) — how this coordinate READS the dim, which is a
    /// different fact from the allocation's own [`AllocPlacement::padding`] and survives
    /// [`Self::clear_fold_for_dim`] as the reference's own note says (`:100`).
    #[must_use]
    pub fn padding(&self, dim: PrimaryDim) -> PadType {
        self.padding.get(dim)
    }

    /// `setPadding(dim, pad)` (`dsc/dsc2.h:236`).
    pub fn set_padding(&mut self, dim: PrimaryDim, pad: PadType) {
        self.padding.set(dim, pad);
    }

    /// `setPadding(const PaddingFormType)` (`dsc/dsc2.h:240`) — REPLACES the whole form, which is a
    /// DIFFERENT operation from the per-dim [`Self::set_padding`]: a dim the form has no entry for
    /// goes back to reading `NOPAD`.
    ///
    /// ⛔ TAKES THE ENTRIES AND NOT A [`Padding`] so that the fold builders can hand over an
    /// allocation's `padding_` without this module depending on theirs.
    pub fn set_padding_form(&mut self, form: impl IntoIterator<Item = (PrimaryDim, PadType)>) {
        self.padding = Padding::default();
        for (dim, pad) in form {
            self.padding.set(dim, pad);
        }
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

    /// `getPadding()` (`dsc/dsc2.h:239`) — the WHOLE padding form, which entry 356 hands straight to
    /// another coordinate's [`Self::set_padding_form`]. Only the dims carrying a style are named;
    /// [`Self::padding`] already reads every other one as `NOPAD`.
    pub fn padding_form(&self) -> impl Iterator<Item = (PrimaryDim, PadType)> + '_ {
        self.padding.dims().map(|dim| (dim, self.padding.get(dim)))
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
    /// positional beta write in the fold builders (`ddc/ddc_fold.cpp:4576-4578`), which adds an
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

/// WHAT AN OPERAND KNOWS ABOUT ITS DATA — `DataInfo` (`dsc/dsc2.h:721`) narrowed to the three fields
/// the ported units read. `dataConnect_` is a closed set, so it is [`DataConnect`] and not a string;
/// a default-constructed `DataInfo` leaves it EMPTY, which is the [`None`].
///
/// ⚠️ [`super::ddc::fold::DataStream`] carries `myLdsIdx_` and `constantId_` as ONE
/// [`super::ddc::fold::DataOrigin`], which is the stronger statement (`isLabeledDs`/`isConstant`
/// `DT_CHECK` that both are not set). Converging the two spellings is a review pass's, not this
/// batch's: `fold.rs` already carries the same note about the fold vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DataInfo {
    /// `dataConnect_`.
    pub data_connect: Option<DataConnect>,
    /// `myLdsIdx_`, once its `-1` is an [`Option`].
    pub my_lds_idx: Option<LdsIdx>,
    /// `constantId_` (`dsc/dsc2.h:726`), once its `-1` is an [`Option`].
    pub constant_id: Option<ConstIdx>,
    /// `latchDataId_` (`dsc/dsc2.h:725`), *"used to link producer and consumer when using LATCH"*,
    /// once its `-1` is an [`Option`]. Entry 339 is the one unit that mints one.
    pub latch_data_id: Option<LatchDataId>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// `dstVias_.at(i).via_`, per destination.
    ///
    /// ⭐ SHORT OR ABSENT IS EMPTY, AND THAT IS THE TRUTH RATHER THAN A GAP: a freshly minted
    /// `DstVia` carries an empty `via_` (`dsc/dsc2.h:818`), so a site that states no route for a
    /// destination has stated the route it has.
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

/// AN ALLOCATION'S PER-DIM PADDING STYLE — `padding_`, a `PaddingFormType` (`dsc/dsc2.h:981`) whose
/// `getPadding(dim)` answers `NOPAD` for every dim it has no entry for.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Padding(BTreeMap<PrimaryDim, PadType>);

impl Padding {
    /// `getPadding(dim)` — total, because the absent case IS `NOPAD`.
    #[must_use]
    pub fn get(&self, dim: PrimaryDim) -> PadType {
        self.0.get(&dim).copied().unwrap_or(PadType::NoPad)
    }

    /// `setPadding(dim, pad)`.
    pub fn set(&mut self, dim: PrimaryDim, pad: PadType) {
        self.0.insert(dim, pad);
    }

    /// The dims that carry a style at all, in `std::map`'s order.
    pub fn dims(&self) -> impl Iterator<Item = PrimaryDim> + '_ {
        self.0.keys().copied()
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
    pub padding: Padding,
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
    pub const fn operand(self) -> Operand {
        Operand {
            unit: self.loc.unit,
            storage: self.loc.storage,
            data: DataInfo {
                data_connect: None,
                my_lds_idx: self.lds,
                constant_id: None,
                latch_data_id: None,
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
    /// `mode_` (`dsc/dsc2.h:930`) — the general SRC1/IMM field, once its `-1` default is an [`Option`].
    pub mode: Option<Mode>,
    /// `compute_mask_` (`dsc/dsc2.h:916`) — `mask=` where a template states one.
    pub compute_mask: ComputeMask,
    /// `repetition_` (`dsc/dsc2.h:923`).
    pub repetition: Repetition,
    /// `indices_` (`dsc/dsc2.h:922`) — the PACK/MERGE mapping, empty where the op states none.
    pub indices: Vec<PackIndex>,
    /// `param_map_` MINUS its two fixed keys — a `ddl.opaque`'s `params=`, copied through
    /// (`ddc/ddl/ddl_conversion.cpp:2674-2690`).
    ///
    /// ⭐ NO OVERLAP WITH [`Self::unroll`] OR [`Self::precision`]: those two are `"unroll"` and
    /// `"prec"`, which entry 261 writes from the compute's own state, and no vendored template
    /// spells either as a `params=` key.
    pub params: BTreeMap<ParamKey, ParamValue>,
    /// `input_data_connects_` (`dsc/dsc2.h:936`) — which ports a spliced opaque body reads.
    pub input_data_connects: Vec<DataConnect>,
    /// `output_data_connects_` (`dsc/dsc2.h:938`).
    pub output_data_connects: Vec<DataConnect>,
}

impl Default for Unroll {
    /// `param_map_` carries no `"unroll"` key on a fresh node, and every reader of it wants one.
    fn default() -> Self {
        Self::ONE
    }
}

/// `dsc2::ComputeNode` (`dsc/dsc2.h:900`) narrowed to what the fold units read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeNode {
    /// `name_`.
    pub name: NodeName,
    /// `type_` — the dialect's whole recognised compute set, because the MACC resolution entry 323
    /// performs names `IMA4`/`IMA8`/`FMA8`/`FMA4`, which no `computetype=` spells and the census
    /// therefore cannot express.
    pub op: DdlComputeType,
    /// `exUnit_`.
    pub ex_unit: SenComponent,
    /// `inputs_` zipped with `inputsLdsAndLoopOffsets_`.
    pub inputs: Vec<Operand>,
    /// `outputs_` zipped with `outputsLdsAndLoopOffsets_`.
    pub outputs: Vec<Operand>,
    /// `numFoldsEngaged` (`dsc/dsc2.h:940`), whose default is ONE and not zero.
    pub num_folds_engaged: NumFolds,
    /// `dataFormat_` (`dsc/dsc2.h:945`), once its `DataFormats::INVALID` default is an [`Option`].
    pub data_format: Option<DataFormat>,
    /// `instrAttribute_` (`dsc/dsc2.h:939`).
    pub instr_attribute: InstrAttribute,
}

/// A POSITION IN A STICK'S DIM ORDER — `srcSizeIdx_`/`dstSizeIdx_` (`dsc/dsc2.h:821`), an index into
/// the `getStickSizes` list and NOT a size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickDimIdx(pub u32);

/// ONE DIM AND HOW MUCH OF IT — `dsc2::Size` (`dsc/dsc2.h:486`), whose `int size_ = -1` default means
/// UNSET. Every size entry 126 pushes is a positive element count, so this one is not optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    /// `dim_`.
    pub dim: PrimaryDim,
    /// `size_`.
    pub size: Elements,
}

/// ONE CHUNK OF A UNIT-TIME TRANSFER — `TransferNode::SizeAndIndex` (`dsc/dsc2.h:820`). Both of its
/// `-1` indices are filled with the same stick-dim position at the one site that pushes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeAndIndex {
    /// `sizeDim_`.
    pub size_dim: Size,
    /// `srcSizeIdx_`.
    pub src_size_idx: StickDimIdx,
    /// `dstSizeIdx_`.
    pub dst_size_idx: StickDimIdx,
}

/// ONE POSITION OF A TRANSFER'S ZERO-PAD FOLD — the `(size, alpha, beta)` triple that ONE
/// `TransferPadInfo::FoldDimPosition` slot of `buildPadFrontSizes`' and `buildPadBackSizes`' three
/// parallel `std::vector<int>`s holds (`dsc/dsc2.h:849`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadFold {
    /// `sizes[pos]` — how many steps that fold axis walks.
    pub cardinality: FoldCardinality,
    /// `alphas[pos]`.
    pub alpha: FoldCoeff,
    /// `betas[pos]`.
    pub beta: FoldCoeff,
}

/// ONE DIM'S ZERO-PAD FOLD SPACE — the two `TransferPadInfo::FoldDimPosition` slots that are ever
/// written, `WORK_SLICE_FOLDDIM` and `CHUNK_FOLDDIM`.
///
/// ⛔ THE REMAINING `TOTAL_FOLDDIM_NUM` SLOTS ARE UNSPELLABLE, AND THAT IS THE TRUTH: entry 221 is the
/// only writer of `paddingInfo_`, it sets exactly these two, and every other slot keeps the zero its
/// `std::vector<int>` was sized with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroPadFolds {
    /// `WORK_SLICE_FOLDDIM`.
    pub work_slice: PadFold,
    /// `CHUNK_FOLDDIM`.
    pub chunk: PadFold,
}

/// A TRANSFER'S ZERO-PAD INFO — `TransferNode::paddingInfo_`, a `TransferPadInfo` (`dsc/dsc2.h:836`),
/// reduced to the two per-dim fold spaces its builders fill.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransferPadding {
    front: BTreeMap<PrimaryDim, ZeroPadFolds>,
    back: BTreeMap<PrimaryDim, ZeroPadFolds>,
}

impl TransferPadding {
    /// `buildPadFrontSizes(dim, sizes, alphas, betas)`.
    pub fn build_pad_front(&mut self, dim: PrimaryDim, folds: ZeroPadFolds) {
        self.front.insert(dim, folds);
    }

    /// `buildPadBackSizes(dim, sizes, alphas, betas)`.
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

    /// Every dim either builder has been handed, in dim order.
    pub fn dims(&self) -> impl Iterator<Item = PrimaryDim> + '_ {
        self.front.keys().chain(self.back.keys()).copied()
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

/// `dsc2::TransferNode` (`dsc/dsc2.h:814`) narrowed to what the ported units read and write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferNode {
    /// `name_`.
    pub name: NodeName,
    /// `src_.unit_` zipped with `srcLdsAndLoopOffsets_`.
    pub src: Operand,
    /// `dstVias_` zipped with `dstLdsAndLoopOffsets_`.
    pub dsts: Dsts,
    /// `replicationFactor_` (`:834`).
    pub replication_factor: ReplicationFactor,
    /// `unitTimeTransferChunkSize_` (`:836`) — the continuous elements within a stick boundary.
    pub unit_time_transfer_chunk_size: Vec<SizeAndIndex>,
    /// `unitTimeTransferNumChunks_` (`:835`) — how many of those chunks one unit time moves.
    pub unit_time_transfer_num_chunks: NumChunks,
    /// `paddingInfo_` (`:836`) — EMPTY on a fresh node; entry 221 is what fills it.
    pub padding: TransferPadding,
    /// `srcIndirect_` FUSED WITH `srcIndirectLdsAndLoopOffsets_.myLdsIdx_` — the index tensor this
    /// transfer gathers its SOURCE addresses through, [`None`] for a direct transfer.
    pub src_indirect: Option<Via>,
    /// `dstVias_.front().locIndirect_` FUSED WITH `dstIndirectLdsAndLoopOffsets_.front().myLdsIdx_`.
    ///
    /// ⛔ ONE END AND NOT A VECTOR, WHICH IS THE REFERENCE'S OWN TWO `DT_CHECK`s: entry 227 demands
    /// `dstVias_.size() == 1` before it writes, and then that `dstIndirectLdsAndLoopOffsets_` was
    /// empty and holds exactly one entry after.
    pub dst_indirect: Option<Via>,
    /// `coreIdToGTRInfo_` (`:840`) — the multicast group each transferring core belongs to, EMPTY
    /// on a fresh node. Entry 291 is what fills it and entry 218 is what fills a condition arm's.
    pub core_id_to_gtr_info: BTreeMap<Core, GroupTagRegInfo>,
    /// `transferSize_` (`:843`) — an EXPLICIT per-dim size that overrides the one derived from the
    /// data stage (`dsc/dsc2.cpp:3474` reads it), EMPTY on a fresh node. Entry 295 fills it.
    pub transfer_size: BTreeMap<PrimaryDim, Elements>,
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
    /// `isLabeledDs()` (`dsc/dsc2.h:737`) — `myLdsIdx_ >= 0`.
    ///
    /// ⛔ THE REFERENCE'S `DT_CHECK_MSG(!(labeledDs && constant), "Cannot be both labeledDs and
    /// constant.")` IS UNSPELLABLE HERE and needs no runtime guard: `my_lds_idx` and `constant_id`
    /// are separate [`Option`]s, so a node carrying both answers `true` here exactly as the
    /// reference's field test would — the reference stops, we do not, and no caller distinguishes.
    #[must_use]
    pub const fn is_labeled_ds(&self) -> bool {
        self.my_lds_idx.is_some()
    }

    /// `isConstant()` (`dsc/dsc2.h:742`) — `constantId_ >= 0`.
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

/// `dsc2::BlockNode` (`dsc/dsc2.h:526`) narrowed to the `name_` a block is looked up by and the
/// `next_` children a traversal descends into.
///
/// ⛔ A FRESH BLOCK HAS NO CHILDREN: `new dsc2::BlockNode()` leaves the child vector empty, and
/// `addChildNode`/`moveChildNode` are the units that fill it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockNode {
    /// `name_`.
    pub name: NodeName,
    /// `next_`, in order.
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

    /// `addChildNode(node, /*addFront=*/false)` — append, which is where a walk in program order puts
    /// every node it mints.
    pub fn add_child(&mut self, node: SchedNode) -> ChildPos {
        self.children.push(node);
        ChildPos(self.children.len() - 1)
    }

    /// `addChildNode(node, /*addFront=*/true)` — the allocation case, whose node has to precede the
    /// transfers already in the block it lands in (`ddl_conversion.cpp:713-716`).
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

/// `dsc2::SyncNode` (`dsc/dsc2.h:964`) narrowed to what minting one writes and what entry 261 then
/// binds onto it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncNode {
    /// `name_`.
    pub name: NodeName,
    /// `units_`.
    pub units: SyncUnits,
    /// `isReceive_`.
    pub direction: SyncDirection,
    /// `isSoft_`.
    pub strength: SyncStrength,
    /// `implicitSyncRefTransfer_` (`dsc/dsc2.h:969`) — `nullptr` until entry 261 picks the transfer
    /// this sync stands in for.
    pub implicit_sync_ref_transfer: Option<NodeId>,
    /// `otherEndOfTheSignals_` (`dsc/dsc2.h:970`) BY NAME, empty on a fresh node: a sequence that
    /// mints both ends pairs them, and a name is the link a tree of owned nodes can hold.
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
    /// `name_` of the `BlockNode` base (`dsc/dsc2.h:526`).
    pub name: NodeName,
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
    /// `name_`.
    pub name: NodeName,
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
    Leaf(NodeName),
}

impl SchedNode {
    /// `ScheduleNode::name_`, whichever kind of node this is.
    #[must_use]
    pub const fn name(&self) -> &NodeName {
        match self {
            Self::Block(block) | Self::Condition(block) => &block.name,
            Self::Loop(node) => &node.block.name,
            Self::Guarded(cond) => &cond.name,
            Self::StickMask(mask) => &mask.name,
            Self::Sync(sync) => &sync.name,
            Self::Leaf(name) => name,
        }
    }
}

/// WHERE IN A BLOCK'S CHILDREN A NODE SITS — the `siblingRefNode` position `addChildNode`
/// (`dsc/dsc2.cpp:2013`) inserts relative to, minted only by [`BlockNode::child_pos`] so an index
/// into one block cannot be applied to another block's shorter child vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildPos(usize);

/// A DSC'S SCHEDULE TREE — `dsc2::ScheduleTree` (`dsc/dsc2.h:621`) reduced to `head_`, whose
/// children are the frontier every traversal starts from.
///
/// ⭐ `head_` IS NEVER VISITED BY A TRAVERSAL. `traverseTreeDFS(nullptr, ..)` seeds the queue with
/// `head_.next_` (`dsc/dsc2.cpp:2233`), and `head_` is a `LoopNode`, so a `{BLOCK}` filter could not
/// name it — but its `denId_` IS OBSERVED: `writeToJson` serialises it as `"scheduleTreeHeadDenId_"`
/// (`dsc/dsc2.cpp:368`) and the parser reads it straight back onto the head (`:1159`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScheduleTree {
    head: BlockNode,
    head_den: Option<DatastageId>,
}

impl ScheduleTree {
    /// A tree over `head_`'s children, whose head names no denominator data stage yet.
    #[must_use]
    pub const fn new(head: BlockNode) -> Self {
        Self {
            head,
            head_den: None,
        }
    }

    /// `getHead()->denId_`, [`None`] for the `-1` an unwritten head carries.
    #[must_use]
    pub const fn head_den(&self) -> Option<DatastageId> {
        self.head_den
    }

    /// `getHeadMutable()->denId_ = den` — the write entry 217 makes onto the root before it chains
    /// the chunk loops under it.
    pub fn set_head_den(&mut self, den: DatastageId) {
        self.head_den = Some(den);
    }

    /// `getHead()` — the root's block part.
    #[must_use]
    pub const fn head(&self) -> &BlockNode {
        &self.head
    }

    /// `getHeadMutable()`.
    pub const fn head_mut(&mut self) -> &mut BlockNode {
        &mut self.head
    }

    /// `traverseTreeDFS(nullptr, {BLOCK})` (`dsc/dsc2.cpp:2222`) — every `BLOCK` node in pre-order
    /// DFS, descending through the loop and condition blocks it does not yield.
    #[must_use]
    pub fn blocks_dfs(&self) -> Vec<&BlockNode> {
        let mut found = Vec::new();
        collect_blocks(&self.head, &mut found);
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
        find_block_mut(&mut self.head, accepts)
    }

    /// The same search over the `CONDITION` nodes, which [`Self::find_block_mut`] descends THROUGH
    /// and never yields — `dsc2::ConditionNode` is a `BlockNode` in the reference and this type holds
    /// its two children in [`ConditionNode::next`], so reaching one by name is its own search.
    pub fn find_guarded_mut(
        &mut self,
        accepts: impl Fn(&ConditionNode) -> bool + Copy,
    ) -> Option<&mut ConditionNode> {
        find_guarded_mut(&mut self.head, accepts)
    }

    /// The same search over the `SYNC` nodes, which neither [`Self::find_block_mut`] nor
    /// [`Self::find_guarded_mut`] yields — `otherEndOfTheSignals_` is bound onto the sync nodes
    /// THEMSELVES (`ddc/ddl/ddl_conversion.cpp:2813`), and a signal names its two ends by name.
    pub fn find_sync_mut(
        &mut self,
        accepts: impl Fn(&SyncNode) -> bool + Copy,
    ) -> Option<&mut SyncNode> {
        find_sync_mut(&mut self.head, accepts)
    }
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
        NodeName, NumBuffers, Padding, PrimaryDim, SenComponent, StartAddress,
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
                padding: Padding::default(),
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
        NodeName, PrimaryDim,
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
                name: NodeName(format!("parametric_loop_{}(padded)", dim.spelling())),
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

    use super::{BlockNode, LoopDim, LoopNode, MetaDimKind, NodeName, PrimaryDim, VariableSymbol};

    /// e011 — the map answers per dim, and `hasLoopDim` is blind to the `kind_` half of the pair.
    #[test]
    fn a_loops_count_is_symbolic_only_on_the_dims_its_symbol_map_names() {
        let mut held = LoopNode::bare(BlockNode {
            name: NodeName("loop_ds0_ds1_out_y".to_owned()),
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

    use super::{BlockNode, CondRegions, ConditionNode, LoopCondComposite, NodeName};

    /// One region block, named as `coordinate_masking` names them (`ddc/ddcv1.cpp:3539-3560`).
    fn region(name: &str) -> BlockNode {
        BlockNode {
            name: NodeName(name.to_owned()),
            children: Vec::new(),
        }
    }

    /// e012 — two regions land in order, and the third is the one refusal left.
    #[test]
    fn a_condition_takes_a_then_region_then_an_else_and_refuses_a_third() {
        let mut held = ConditionNode {
            name: NodeName("condition_SAMV_dim_out".to_owned()),
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
