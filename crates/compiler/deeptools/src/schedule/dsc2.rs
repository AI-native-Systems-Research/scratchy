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
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
use crate::formats::DataFormat;
use crate::generated::{ComputeType, DataConnect, RegName};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{ConstIdx, NodeId, PadType};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
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

impl ComputeType {
    /// `EnumsConversion::computeTypeToString` (`dsc/dscdefn.cpp:34`), which is what the dsc-side debug
    /// prints and JSON carry.
    ///
    /// ⛔ NOT the generated [`ComputeType::spelling`] — that one is the DDL template's own spelling,
    /// which is UPPERCASE (`MACC`), and the two are different strings for one op.
    /// ⛔ NO `_` ARM: a compute type entering the census has to be given its C++ spelling here. That
    /// is also why the reference's `computeTypeToString.at(type_)` throw is unreachable rather than
    /// mapped — the one op with no entry, `FCVT`, is not in the census.
    #[must_use]
    pub const fn cpp_spelling(self) -> &'static str {
        match self {
            Self::Assign => "assign",
            Self::Equal => "equal",
            Self::Fabsmax => "fabsmax",
            Self::Fest => "fest",
            Self::Floor => "floor",
            Self::Fma16 => "fma16",
            Self::Fma32 => "fma32",
            Self::Fmax => "fmax",
            Self::Fmin => "fmin",
            Self::Fmul => "fmul",
            Self::Fnms => "fnms",
            Self::Greaterequal => "greaterequal",
            Self::Greaterthan => "greaterthan",
            Self::Icvt => "icvt",
            Self::Lesserequal => "lesserequal",
            Self::Lesserthan => "lesserthan",
            Self::Macc => "macc",
            Self::Notequal => "notequal",
            Self::Or => "or",
            Self::Packmerge => "packmerge",
            Self::Reduce => "reduce",
            Self::Select => "select",
            Self::Shuffle => "shuffle",
            Self::Splat => "splat",
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

    /// `foldConstructed()` (`dsc/dsc2.h:119`).
    #[must_use]
    pub const fn fold_constructed(&self) -> bool {
        self.fold_constructed
    }

    /// `completeFoldConstruction()` — a one-way latch, as the reference's own setter is.
    pub const fn complete_fold_construction(&mut self) {
        self.fold_constructed = true;
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
}

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
    #[must_use]
    pub const fn outermost_dim(&self) -> PrimaryDim {
        self.first.0
    }

    /// The dims with their max sizes, outermost first.
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
        self.placed.entry(core).or_default().insert(corelet, addresses);
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

/// AN OPAQUE OP'S BOUND INSTRUCTION STATE — `ComputeNode::instrAttribute_` (`dsc/dsc2.h:905-939`)
/// narrowed to the four things entry 261 writes.
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
}

impl Default for Unroll {
    /// `param_map_` carries no `"unroll"` key on a fresh node, and every reader of it wants one.
    fn default() -> Self {
        Self::ONE
    }
}

/// `dsc2::ComputeNode` (`dsc/dsc2.h:948`) narrowed to what the fold units read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeNode {
    /// `name_`.
    pub name: NodeName,
    /// `type_` — the crate's censused compute set, whose `spelling()` is
    /// `EnumsConversion::computeTypeToString` for every member of it.
    pub op: ComputeType,
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
}

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

/// `dsc2::ConditionNode` (`dsc/dsc2.h:685`) — a block whose children are split into a then-region and
/// an else-region, guarded by a loop predicate or a core/corelet set.
///
/// ⛔ `hasCoreClCond()` IS `loopCond_.twoLevelOrOfAnds_.empty()` IN THE REFERENCE, i.e. it answers
/// "the core/corelet set is what guards this", not "the set is non-empty".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConditionNode {
    /// `name_`.
    pub name: NodeName,
    /// `loopCond_`.
    pub loop_cond: LoopCondComposite,
    /// `coreClCond_`.
    pub core_cl_cond: BTreeMap<Core, BTreeSet<Corelet>>,
    /// `addThenRegion(block)` — the children taken when the predicate holds.
    pub then_region: Vec<SchedNode>,
    /// `addElseRegion(block)` — the children taken otherwise.
    pub else_region: Vec<SchedNode>,
}

impl ConditionNode {
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

    /// The dims, outermost first.
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
    /// `DesignSpaceConfig::getLayoutDims(ldsIdx)`.
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
    /// `nodeType_ == LOOP`: descended into, never yielded by a `{BLOCK}` filter.
    Loop(BlockNode),
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
            Self::Block(block) | Self::Loop(block) | Self::Condition(block) => &block.name,
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
/// ⭐ `head_` IS NEVER VISITED. `traverseTreeDFS(nullptr, ..)` seeds the queue with `head_.next_`
/// (`dsc/dsc2.cpp:2233`), and `head_` is a `LoopNode` with `denId_ = 0` in any case, so a `{BLOCK}`
/// filter could not name it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScheduleTree {
    head: BlockNode,
}

impl ScheduleTree {
    /// A tree over `head_`'s children.
    #[must_use]
    pub const fn new(head: BlockNode) -> Self {
        Self { head }
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
            SchedNode::Loop(inner) | SchedNode::Condition(inner) => collect_blocks(inner, found),
            SchedNode::Guarded(cond) => {
                collect_blocks_in(&cond.then_region, found);
                collect_blocks_in(&cond.else_region, found);
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
            SchedNode::Loop(inner) | SchedNode::Condition(inner) => {
                if let Some(found) = find_block_mut(inner, accepts) {
                    return Some(found);
                }
            }
            SchedNode::Guarded(cond) => {
                if let Some(found) = find_block_mut_in(&mut cond.then_region, accepts) {
                    return Some(found);
                }
                if let Some(found) = find_block_mut_in(&mut cond.else_region, accepts) {
                    return Some(found);
                }
            }
            SchedNode::StickMask(_) | SchedNode::Sync(_) | SchedNode::Leaf(_) => {}
        }
    }
    None
}
