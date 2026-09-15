// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE ONE `currDsc` STAGE 2B'S FOUR VIEWS NAME — [`Dsc2State`] holds every DSC's design space,
//! its `computeOp_` list and its `dataStageParam_`, and [`super::Dsc2Store`], [`super::Dsc2Reads`],
//! [`super::Dsc2Tree`] and [`super::Dsc2Stages`] each hold a SHARED reference to it.
//!
//! ⛔⛔ THE STRUCTURAL FACT A CALLER MUST WORK AROUND — [`v1::Dsc2Store::split`]. Its signature is
//! `fn split(&mut self) -> (&Self::Reads, &mut Self::Tree)`, and its own doc (`ddc/v1.rs:6040-6047`)
//! says a caller holding `struct { reads, tree }` satisfies it and warns that TWO CARRIERS OVER ONE
//! `currDsc` would make entry 308's writes invisible to entry 307's reads. So [`super::Dsc2Store`]
//! OWNS its two halves as fields — and both of those halves hold `&'s Dsc2State`, so the shared and
//! the exclusive view name the SAME tree, the SAME `labeledDs_` and the SAME compute-op list.
//!
//! ⭐ IT IS THE PATTERN STAGE 2A ALREADY PROVED, NOT A SECOND ONE. [`super::DscState`] is the same
//! shape for the L3 stage, and it is sound for the same reason: every tree and design-space trait
//! answers BY VALUE, so no [`RefCell`] borrow outlives the call that took it and none can overlap a
//! write.
//!
//! # ⛔⛔ THREE FACTS `l3::dsc` DOES NOT CARRY, AND WHY THEY ARE CONSTRUCTION ARGUMENTS
//!
//! 1. **`computeOp_`** — [`v1::PrepDsc::compute_ops`] is the FIRST provider call `run_v1` makes
//!    (`ddc/v1.rs:6437`), and an EMPTY answer makes it `continue` past the DSC. So a carrier that
//!    could not state the op list would silently skip every DSC and answer [`v1::DscFilled::Yes`]
//!    having done nothing — a false green. [`DesignSpaceConfig`] does not hold `computeOp_`, so the
//!    list is handed in, exactly as stage 2a is handed [`v1::OpFuncs`] ([`super::run_l3`]).
//! 2. **`dsc.name_`** — `dsc/designSpaceConfig.h:60`, which [`DesignSpaceConfig`] does not carry
//!    either, and which only [`v1::Dsc2Fill::said`]'s two verbose lines read.
//! 3. **`numWkSlicesPerDim_`** — a [`SuperDsc`] field, and `run_v1` takes the super-DSC as
//!    `&mut`, so no provider may alias it. It is copied in at construction.
//!
//! # ⛔⛔ AND THE ONE `run_v1` ITSELF SEVERS
//!
//! `run_v1(sdsc: &mut SuperDsc, sites: &mut P, ..)` takes TWO independent `&mut`s, so `P` may not
//! borrow the super-DSC. `currDsc` in the reference IS `sdsc.dscs_.at(idx)`; here it is a CLONE taken
//! at construction. `select_and_parse_ddl_template` writes through `sdsc.dscs_mut().at_mut(idx)`
//! (`ddc/v1.rs:6489`) and those writes do NOT reach this clone. That is a third cut of the same kind
//! review 382 recorded on `L3RunInputs`, it is in stage 2b's own signature, and no caller can close
//! it — see [`Dsc2Facts::dsc`].

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PaddedExtent, PrimaryDim, Sample,
};
use crate::schedule::ddc::fold::PadType;
use crate::schedule::ddc::transformation_util::{
    DataStage as UtilDataStage, DataStages as UtilDataStages, DimSplit,
    StageDims as UtilStageDims, StageExtents as UtilStageExtents, StageName,
};
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::LdsIdx;
use crate::schedule::l3::dsc::{DesignSpaceConfig, DscIdx, StageDims, SuperDsc, WkSliceCount};
use crate::units::Corelet;

use super::state::{DscState, DscTree};

/// ⭐ ONE HALF OF ONE `dataStageParam_` ENTRY — `DataStructDims` (`dsc/dims.h`) as
/// [`v1::ExploreStages::Dims`], which must be `Stage + UtilStageExtents + Default + Clone`
/// (`ddc/v1.rs:6420`).
///
/// ⭐ THE FOUR MAPS `l3::dsc::StageDims` ALREADY IS, PLUS THE TWO IT LACKS. `StageDims` carries
/// `primaryDimToValHandler_st`'s slots, `paddingSizes_`, `symbolicDimInfo_`/`maxSymbolicVolume_` and
/// `coreletSplit_`; `rowSplit_` and `peSfpSplit_` are stated here beside it rather than in a second
/// extents type, so [`UtilStageExtents::states`] answers all four splits off one value.
#[derive(Debug, Clone, Default)]
pub struct Dsc2Dims {
    /// `name_`.
    pub name: StageName,
    /// The four maps `l3::dsc` already models.
    pub dims: StageDims,
    /// `rowSplit_` — per corelet, each PT row's share.
    pub row_split: BTreeMap<PrimaryDim, BTreeMap<Corelet, Vec<Extent>>>,
    /// `peSfpSplit_` — per corelet, the PE's and the SFP's shares.
    pub pe_sfp_split: BTreeMap<PrimaryDim, BTreeMap<Corelet, v1::PeSfpShares>>,
}

impl Dsc2Dims {
    /// ⭐⭐ `primaryDimToVal_base_st`'S PLAIN FIELD READ, AND NOTHING ELSE —
    /// `dsc/dims.cpp:516-560`, the same citation entry 207 already stands on in this crate.
    ///
    /// ⛔⛔ [`None`] WHERE THE AUTHORITY'S CONTROL FLOW LEAVES THAT READ, and that is deliberate.
    /// `primaryDimToVal_st` (`dsc/dims.cpp:653-704`) folds in `rowSplit_`, then `peSfpSplit_`, then
    /// `primaryDimToVal_clView_st`'s `coreletSplit_` (`:631-645`), and `calculate_padded`
    /// (`:563-616`) rewrites the result through the window dim, the stride and six pad types. Every
    /// one of those is an EXTENT a placement is computed from, and a hand-rolled fold here would be
    /// a fabricated extent — the failure this crate ranks worse than a stop. So the reader answers
    /// only where the authority provably returns the stored slot: the dim is not symbolic, no split
    /// names it, the padding is `NOPAD`, and the sample asks for the whole of every axis.
    ///
    /// ⭐ THE SYMBOLIC ARM IS ANSWERED, because it too is a plain field read: `symbolicDimInfo_`'s
    /// `maxSize_` under [`v1::SymbolicRead::Max`] and its `granularity_` under
    /// [`v1::SymbolicRead::Granularity`] (`:522-527`).
    pub(super) fn raw_slot(
        &self,
        dim: PrimaryDim,
        padding: PadType,
        symbolic: v1::SymbolicRead,
    ) -> Option<Extent> {
        if padding != PadType::NoPad {
            return None;
        }
        if self.row_split.contains_key(&dim)
            || self.pe_sfp_split.contains_key(&dim)
            || self.dims.corelet_split.contains_key(&dim)
        {
            return None;
        }
        if let Some(info) = self.dims.symbolic.info().get(&dim) {
            return Some(match symbolic {
                v1::SymbolicRead::Max => Extent(i64::from(info.max_size.0)),
                v1::SymbolicRead::Granularity => Extent(i64::from(info.granularity.get())),
            });
        }
        Some(
            self.dims
                .extents
                .get(&dim)
                .copied()
                // ⭐ THE UNSTATED SLOT IS THE REFERENCE'S OWN `-1`, not a refusal — entry 207's
                // recorded divergence, and the same constant it stands on.
                .unwrap_or(crate::schedule::l3::dl_ops::UNSTATED_EXTENT),
        )
    }

    /// The one split map [`DimSplit`] names, as the dims it holds.
    fn split_dims(&self, split: DimSplit) -> BTreeSet<PrimaryDim> {
        match split {
            DimSplit::Corelet => self.dims.corelet_split.keys().copied().collect(),
            DimSplit::Row => self.row_split.keys().copied().collect(),
            DimSplit::PeSfp => self.pe_sfp_split.keys().copied().collect(),
            DimSplit::Padding => self.dims.padding.keys().copied().collect(),
        }
    }
}

impl crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Stage for Dsc2Dims {
    fn is_symbolic(&self, dim: PrimaryDim) -> bool {
        self.dims.symbolic.info().contains_key(&dim)
    }

    fn is_corelet_split(&self, dim: PrimaryDim) -> bool {
        self.dims.corelet_split.contains_key(&dim)
    }

    fn is_row_split(&self, dim: PrimaryDim) -> bool {
        self.row_split.contains_key(&dim)
    }

    fn is_pe_sfp_split(&self, dim: PrimaryDim) -> bool {
        self.pe_sfp_split.contains_key(&dim)
    }

    fn splits_any_row(&self) -> bool {
        !self.row_split.is_empty()
    }

    /// `primaryDimToVal_st(dim, comp, row, cl)` — ⛔ [`Self::raw_slot`]'s refusal is a `todo!` here,
    /// because the trait's return type is TOTAL and a substituted extent is a fabricated one.
    fn extent(&self, dim: PrimaryDim, at: Sample) -> Extent {
        // The whole of every axis is what `primaryDimToVal_st`'s own defaults ask for
        // (`dsc/dims.cpp:647`); a sampled axis reaches one of the three fold arms.
        if at.row.is_none()
            && at.comp.is_none()
            && let Some(extent) = self.raw_slot(dim, PadType::NoPad, v1::SymbolicRead::Max)
        {
            return extent;
        }
        todo!(
            "Stage::extent: wants primaryDimToVal_st (dsc/dims.cpp:653-704) to fold rowSplit_/\
             peSfpSplit_/coreletSplit_ for a SAMPLED axis — this stage states a split on {dim:?} or \
             the sample names a row/vector component. A hand-rolled fold would be a fabricated \
             extent."
        )
    }

    /// The same with `PADDED_WZEROPAD` on `dim` — ⛔ ALWAYS `calculate_padded`
    /// (`dsc/dims.cpp:563-616`), so there is no plain-read case to answer.
    fn padded_extent(&self, _dim: PrimaryDim, _at: Sample) -> Option<PaddedExtent> {
        todo!(
            "Stage::padded_extent: wants calculate_padded (dsc/dims.cpp:563-616) — the window dim, \
             the stride and the six PadTypes. Never a plain read, so never answerable by a carrier."
        )
    }
}

impl UtilStageExtents for Dsc2Dims {
    fn copy_dim_value_from(&mut self, other: &Self, dim: PrimaryDim) {
        match other.dims.extents.get(&dim) {
            Some(extent) => {
                self.dims.extents.insert(dim, *extent);
            }
            None => {
                self.dims.extents.remove(&dim);
            }
        }
    }

    fn states(&self, split: DimSplit, dim: PrimaryDim) -> bool {
        self.split_dims(split).contains(&dim)
    }

    fn copy_split_from(&mut self, other: &Self, split: DimSplit, dim: PrimaryDim) {
        match split {
            DimSplit::Corelet => {
                if let Some(held) = other.dims.corelet_split.get(&dim) {
                    self.dims.corelet_split.insert(dim, held.clone());
                }
            }
            DimSplit::Row => {
                if let Some(held) = other.row_split.get(&dim) {
                    self.row_split.insert(dim, held.clone());
                }
            }
            DimSplit::PeSfp => {
                if let Some(held) = other.pe_sfp_split.get(&dim) {
                    self.pe_sfp_split.insert(dim, held.clone());
                }
            }
            DimSplit::Padding => {
                if let Some(held) = other.dims.padding.get(&dim) {
                    self.dims.padding.insert(dim, *held);
                }
            }
        }
    }

    /// ⛔⛔ NOT AN ERASE. `makeDimNotSymbolic` (`dsc/dims.cpp:781-803`) erases the entry AND THEN
    /// DIVIDES every stated size for that dim — `primaryDimToValHandler_st(dim)`, every
    /// `coreletSplit_` share, every `rowSplit_` share and every `peSfpSplit_` side — by
    /// `maxSize_ / granularity_`. Dropping the divide leaves every extent that factor too LARGE, and
    /// an extent that large becomes an oversized buffer and a wrong address. So the divide is named
    /// rather than skipped.
    fn make_dim_not_symbolic(&mut self, _dim: PrimaryDim) {
        todo!(
            "UtilStageExtents::make_dim_not_symbolic: wants makeDimNotSymbolic \
             (dsc/dims.cpp:781-803) — it ERASES symbolicDimInfo_[dim] and then divides the dim \
             value and every corelet/row/PE-SFP share by maxSize_/granularity_. A bare erase leaves \
             every extent that factor too large."
        )
    }

    fn clear_split(&mut self, split: DimSplit) {
        match split {
            DimSplit::Corelet => self.dims.corelet_split.clear(),
            DimSplit::Row => self.row_split.clear(),
            DimSplit::PeSfp => self.pe_sfp_split.clear(),
            DimSplit::Padding => self.dims.padding.clear(),
        }
    }

    /// `<split>_.at(dim)` — ⛔ THE PE/SFP MAP IS PER CORELET AND PER SIDE, so a flat list of sizes
    /// cannot state it; that arm names the shape it wants rather than flattening two keys into one.
    fn split_sizes(&self, split: DimSplit, dim: PrimaryDim) -> Vec<Elements> {
        match split {
            DimSplit::Corelet => self
                .dims
                .corelet_split
                .get(&dim)
                .map(|shares| {
                    shares
                        .iter()
                        .map(|extent| Elements(extent.0.unsigned_abs()))
                        .collect()
                })
                .unwrap_or_default(),
            DimSplit::Row => self
                .row_split
                .get(&dim)
                .map(|per_corelet| {
                    per_corelet
                        .values()
                        .flatten()
                        .map(|extent| Elements(extent.0.unsigned_abs()))
                        .collect()
                })
                .unwrap_or_default(),
            DimSplit::PeSfp => todo!(
                "UtilStageExtents::split_sizes: peSfpSplit_.at(dim) is keyed by corelet AND by \
                 VectorComp (dsc/dims.h:208); a flat Vec<Elements> cannot say which side a size \
                 belongs to, and entry 300 reads .at(0)/.at(1) positionally"
            ),
            DimSplit::Padding => Vec::new(),
        }
    }
}

/// ⭐ ONE DSC'S FACTS BEYOND ITS TREE — what `currDsc` answers that [`DscTree`] does not.
#[derive(Debug)]
pub struct Dsc2Facts {
    /// `dsc.name_` (`dsc/designSpaceConfig.h:60`) — a construction argument; see the module note.
    name: v1::StorageName,
    /// ⛔⛔ A CLONE OF `sdsc.dscs_.at(idx)`, AND THAT IS `run_v1`'S OWN CUT. The reference's
    /// `currDsc` IS that entry; `run_v1` takes `sdsc: &mut SuperDsc` beside `sites: &mut P`, so no
    /// provider may borrow it. `select_and_parse_ddl_template` writes the DSC through the super-DSC
    /// path (`ddc/v1.rs:6489`) and this clone does not see those writes.
    dsc: DesignSpaceConfig,
    /// `computeOp_` — a construction argument; see the module note.
    ops: RefCell<Vec<v1::DscComputeOp>>,
    /// `numCoreletsUsed_DSC2_` as `prep_dsc` (entry 308) writes it, [`None`] before it runs, which
    /// is the `-1` a DSC is built with (`l3::dsc::DesignSpaceConfig::corelets_used_dsc2`).
    corelets_dsc2: Cell<Option<u32>>,
    /// `dataStageParam_` — the map [`super::Dsc2Stages`] hands back from `as_util`, seeded from the
    /// DSC's own core and chunk stages.
    stages: RefCell<UtilDataStages<Dsc2Dims>>,
}

impl Dsc2Facts {
    /// `dsc.name_`.
    pub(super) fn name(&self) -> v1::StorageName {
        self.name.clone()
    }

    /// The design space, for the length of one read.
    pub(super) const fn dsc(&self) -> &DesignSpaceConfig {
        &self.dsc
    }

    /// `computeOp_`, in order.
    pub(super) fn ops(&self) -> Vec<v1::DscComputeOp> {
        self.ops.borrow().clone()
    }

    /// `computeOp_`, for the length of one write.
    pub(super) fn with_ops_mut<T>(&self, write: impl FnOnce(&mut Vec<v1::DscComputeOp>) -> T) -> T {
        write(&mut self.ops.borrow_mut())
    }

    /// `numCoreletsUsed_DSC2_`, [`None`] before entry 308 states it.
    pub(super) fn corelets_dsc2(&self) -> Option<u32> {
        self.corelets_dsc2.get()
    }

    /// `numCoreletsUsed_DSC2_ = corelets`.
    pub(super) fn set_corelets_dsc2(&self, corelets: u32) {
        self.corelets_dsc2.set(Some(corelets));
    }

    /// `dataStageParam_`, for one read.
    pub(super) fn with_stages<T>(&self, ask: impl FnOnce(&UtilDataStages<Dsc2Dims>) -> T) -> T {
        ask(&self.stages.borrow())
    }

    /// `dataStageParam_`, for one write.
    pub(super) fn with_stages_mut<T>(
        &self,
        write: impl FnOnce(&mut UtilDataStages<Dsc2Dims>) -> T,
    ) -> T {
        write(&mut self.stages.borrow_mut())
    }
}

/// ⭐ EVERY DSC'S FACTS AND THE TREES STAGE 2A LEFT — the ONE state stage 2b's four views name.
#[derive(Debug)]
pub struct Dsc2State<'l> {
    /// The schedule trees, which are stage 2a's own state: composing the two stages means handing
    /// [`Self::seeded`] the SAME [`DscState`] [`super::run_l3`] grew.
    l3: &'l DscState,
    dscs: Vec<Dsc2Facts>,
    /// `sdsc.numWkSlicesPerDim_` — a [`SuperDsc`] field, copied in because `run_v1` holds the
    /// super-DSC as `&mut`.
    wk_slices: BTreeMap<PrimaryDim, WkSliceCount>,
    /// `sdsc.coreIdToWkSlice_` — the same, and what the DDL ring's neighbour lookup walks.
    core_wk_slices: BTreeMap<crate::units::Core, crate::schedule::l3::dsc::WkSlice>,
    /// ⭐ EVERY PROVIDER METHOD THAT REFUSED, in the order it was asked — an OBSERVER and not a
    /// behaviour, exactly as [`DscState::refusals`] is.
    refusals: RefCell<Vec<&'static str>>,
}

impl<'l> Dsc2State<'l> {
    /// ⭐⭐ THE SEED — every DSC's design space cloned out of the super-DSC, its `computeOp_` handed
    /// in per DSC, and its `dataStageParam_` seeded from the DSC's own core and chunk stages.
    ///
    /// ⛔ `ops` IS POSITIONAL BESIDE `sdsc.dscs()`. A DSC the caller states no ops for gets an EMPTY
    /// list, which is `run_v1`'s own `continue` (`ddc/v1.rs:6438`) — the DSC is skipped, exactly as
    /// the reference skips a DSC with no compute op.
    ///
    /// ⛔ AND `names` LIKEWISE: `dsc.name_` is not a [`DesignSpaceConfig`] field, so a caller that
    /// has none gets the positional spelling, which only the two verbose lines read.
    #[must_use]
    pub fn seeded(
        sdsc: &SuperDsc,
        l3: &'l DscState,
        ops: &[Vec<v1::DscComputeOp>],
        names: &[v1::StorageName],
    ) -> Self {
        let dscs = sdsc
            .dscs()
            .iter()
            .enumerate()
            .map(|(at, dsc)| Dsc2Facts {
                name: names.get(at).cloned().unwrap_or_else(|| {
                    v1::StorageName(format!("dsc{at}"))
                }),
                dsc: dsc.clone(),
                ops: RefCell::new(ops.get(at).cloned().unwrap_or_default()),
                corelets_dsc2: Cell::new(dsc.corelets_used_dsc2.map(|used| used.get())),
                stages: RefCell::new(seed_stages(dsc)),
            })
            .collect();
        Self {
            l3,
            dscs,
            wk_slices: sdsc.num_wk_slices_per_dim.clone(),
            core_wk_slices: sdsc.core_id_to_wk_slice.clone(),
            refusals: RefCell::new(Vec::new()),
        }
    }

    /// That DSC's facts, [`None`] for a `dscs_` position this state holds none for.
    pub(super) fn facts(&self, at: DscIdx) -> Option<&Dsc2Facts> {
        self.dscs.get(usize::try_from(at.0).ok()?)
    }

    /// That DSC's schedule tree — the SAME tree stage 2a grew.
    pub(super) fn tree(&self, at: DscIdx) -> Option<&DscTree> {
        self.l3.dsc(at)
    }

    /// `sdsc.numWkSlicesPerDim_.at(dim)`.
    pub(super) fn wk_slices(&self, dim: PrimaryDim) -> Option<WkSliceCount> {
        self.wk_slices.get(&dim).copied()
    }

    /// `sdsc.coreIdToWkSlice_` — every core the super-DSC states, with its work slice per dim.
    pub(super) fn core_wk_slices(
        &self,
    ) -> BTreeMap<crate::units::Core, crate::schedule::l3::dsc::WkSlice> {
        self.core_wk_slices.clone()
    }

    /// ⭐ A PROVIDER METHOD'S OWN REFUSAL, recorded and then propagated.
    pub(super) fn refuse<T>(&self, what: &'static str) -> Option<T> {
        self.refusals.borrow_mut().push(what);
        None
    }

    /// Every refusal so far, in the order it was made.
    #[must_use]
    pub fn refusals(&self) -> Vec<&'static str> {
        self.refusals.borrow().clone()
    }

    /// The FIRST refusal — the one fact that decides what happens next.
    #[must_use]
    pub fn first_refusal(&self) -> Option<&'static str> {
        self.refusals.borrow().first().copied()
    }

    /// How many DSCs this state holds facts for.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dscs.len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dscs.is_empty()
    }
}

/// ⭐ `dataStageParam_` SEEDED FROM THE DSC'S OWN CORE AND CHUNK STAGES — the two
/// [`crate::schedule::l3::dsc::DataStages`] holds out of the map, filed under the two ids stage 2b
/// reads them by (`Metadata::CORE_DSTGID`, `Metadata::CHUNK_DSTGID`).
///
/// ⛔ TWO ENTRIES AND NOT MORE. `l3::dsc::DataStages` states exactly the core and the chunk stage;
/// every further `dataStageParam_` entry is one entry 338 MINTS while stage 2b runs, so seeding any
/// other id would be inventing a stage nobody stated.
fn seed_stages(dsc: &DesignSpaceConfig) -> UtilDataStages<Dsc2Dims> {
    let mut map = UtilDataStages::default();
    for (id, stage) in [
        (
            crate::schedule::ddc::metadata::Metadata::CORE_DSTGID,
            dsc.data_stages.core(),
        ),
        (
            crate::schedule::ddc::metadata::Metadata::CHUNK_DSTGID,
            dsc.data_stages.chunk(),
        ),
    ] {
        map.0.insert(
            id,
            UtilDataStage {
                ss: UtilStageDims {
                    name: stage.ss.name.clone(),
                    dims: Dsc2Dims {
                        name: stage.ss.name.clone(),
                        dims: stage.ss.dims.dims().clone(),
                        row_split: BTreeMap::new(),
                        pe_sfp_split: BTreeMap::new(),
                    },
                },
                el: UtilStageDims {
                    name: stage.el.name.clone(),
                    dims: Dsc2Dims {
                        name: stage.el.name.clone(),
                        dims: stage.el.dims.dims().clone(),
                        row_split: BTreeMap::new(),
                        pe_sfp_split: BTreeMap::new(),
                    },
                },
            },
        );
    }
    map
}

/// `M` — `memTrackers` as stage 2b reaches them, which is a DIFFERENT trait from stage 2a's
/// [`crate::schedule::l3::dl_ops::ExPhaseTrackers`] over the SAME unported C++ allocator.
///
/// ⛔⛔ WANTS `ddc::DsTrackInMem`, AN UNPORTED 703-LINE C++ ALLOCATOR — `util/memtracker/
/// mem_track.{h,cpp}`, outside every campaign's file list, being ported separately. Every method
/// here is a `todo!` NAMING its call, and that is the whole point: `check_and_add` and
/// `check_and_add_at` DECIDE the byte offset of every allocation, and an invented [`v1::Placed`] is
/// a fabricated address.
///
/// ⭐ THE ORACLE IS ALREADY RECORDED, on stage 2a's own tracker
/// ([`super::Trackers`](super::carriers::Trackers)): `g0/debug/sdsc_0/sdsc.json`'s three LX
/// allocations sit at 1_625_344 / 1_625_856 / 1_626_368, and `bufferOffsetCoreCorelet_` is the
/// DOUBLE-BUFFER STRIDE (256 on every node and every core), NOT the address.
#[derive(Debug, Clone, Copy, Default)]
pub struct DdcTrackers;

impl v1::MemTrackers for DdcTrackers {
    fn capacity(&self, _at: v1::TrackerSite) -> crate::arch::Bytes {
        todo!(
            "v1::MemTrackers::capacity: wants memCapacity off ddc::DsTrackInMem — an UNPORTED \
             703-line C++ allocator (util/memtracker/mem_track.{{h,cpp}})"
        )
    }

    fn backup(&mut self, _at: v1::TrackerSite) {
        todo!("v1::MemTrackers::backup: wants backupEps(exphase) on ddc::DsTrackInMem")
    }

    fn restore_all(&mut self) {
        todo!(
            "v1::MemTrackers::restore_all: wants restoreEps(exphase, backupInfo) on \
             ddc::DsTrackInMem"
        )
    }

    fn remove(&mut self, _at: v1::TrackerSite, _name: &v1::StorageName) {
        todo!("v1::MemTrackers::remove: wants removeDs(name, seps) on ddc::DsTrackInMem")
    }

    fn add_at(
        &mut self,
        _at: v1::TrackerSite,
        _name: &v1::StorageName,
        _size: crate::arch::Bytes,
        _address: crate::arch::Bytes,
    ) {
        todo!(
            "v1::MemTrackers::add_at: wants addDsAtStartAddr(name, size, seps, addr) on \
             ddc::DsTrackInMem"
        )
    }

    fn check_and_add(
        &mut self,
        _at: v1::TrackerSite,
        _name: &v1::StorageName,
        _size: crate::arch::Bytes,
    ) -> Option<v1::Placed> {
        todo!(
            "v1::MemTrackers::check_and_add: wants checkAndAddDs(name, size, seps) on \
             ddc::DsTrackInMem — THIS CALL IS THE PLACEMENT AUTHORITY; an invented Placed is a \
             fabricated address"
        )
    }

    fn check_and_add_at(
        &mut self,
        _at: v1::TrackerSite,
        _name: &v1::StorageName,
        _size: crate::arch::Bytes,
        _address: crate::arch::Bytes,
    ) -> Option<v1::Placed> {
        todo!(
            "v1::MemTrackers::check_and_add_at: wants checkAndAddDsAtAddr(name, size, seps, addr) \
             on ddc::DsTrackInMem"
        )
    }
}

/// `K` — where entry 260's `DataInfo` fills land, kept by the operand they landed on.
#[derive(Debug, Default)]
pub struct DdcSink {
    fills: BTreeMap<v1::OperandSite, v1::DataInfoFill>,
    const_ele_offsets:
        BTreeMap<v1::OperandSite, BTreeMap<(crate::units::Core, Corelet, PrimaryDim), v1::ConstEleOffset>>,
    last_fusable_src: BTreeMap<crate::schedule::ddc::fold::NodeId, Option<crate::schedule::ddc::transformation::LoopId>>,
    last_fusable_dsts:
        BTreeMap<crate::schedule::ddc::fold::NodeId, Vec<Option<crate::schedule::ddc::transformation::LoopId>>>,
}

impl DdcSink {
    /// Every fill entry 260 wrote, by the operand it landed on.
    #[must_use]
    pub const fn fills(&self) -> &BTreeMap<v1::OperandSite, v1::DataInfoFill> {
        &self.fills
    }
}

impl v1::DataInfoSink for DdcSink {
    fn fill(&mut self, at: v1::OperandSite, fill: v1::DataInfoFill) -> Option<()> {
        self.fills.insert(at, fill);
        Some(())
    }

    fn set_const_ele_offset(
        &mut self,
        at: v1::OperandSite,
        core: crate::units::Core,
        corelet: Corelet,
        dim: PrimaryDim,
        offset: v1::ConstEleOffset,
    ) -> Option<()> {
        self.const_ele_offsets
            .entry(at)
            .or_default()
            .insert((core, corelet, dim), offset);
        Some(())
    }

    /// `constEleOffsets_.empty()` — ⛔ [`None`] IS AN OPERAND WITH NO FILL, which is the reference's
    /// own `.at()` on an operand entry 260 has not reached.
    fn const_ele_offsets_empty(&self, at: v1::OperandSite) -> Option<bool> {
        self.fills.contains_key(&at).then(|| {
            self.const_ele_offsets
                .get(&at)
                .is_none_or(BTreeMap::is_empty)
        })
    }

    fn set_last_fusable_src(
        &mut self,
        node: crate::schedule::ddc::fold::NodeId,
        at: Option<crate::schedule::ddc::transformation::LoopId>,
    ) -> Option<()> {
        self.last_fusable_src.insert(node, at);
        Some(())
    }

    fn set_last_fusable_dsts(
        &mut self,
        node: crate::schedule::ddc::fold::NodeId,
        at: Vec<Option<crate::schedule::ddc::transformation::LoopId>>,
    ) -> Option<()> {
        self.last_fusable_dsts.insert(node, at);
        Some(())
    }
}

/// `X` — `sdsc_->symbolDefinitions_`, the table entry 260 divides placed addresses in.
///
/// ⭐ EVERY ADDRESS THIS CALLER PLACES IS A BYTE COUNT, so the reference's symbolic arm is never
/// entered: [`v1::Symbols::divide_symbols`] is the identity the reference performs on a
/// non-symbolic address, which [`crate::schedule::dsc2::StartAddress::divided_by`] already is.
#[derive(Debug, Clone, Copy, Default)]
pub struct DdcSymbols;

impl v1::Symbols for DdcSymbols {
    fn divide_symbols(
        &mut self,
        address: &crate::schedule::dsc2::StartAddress,
        by: core::num::NonZeroU64,
    ) -> crate::schedule::dsc2::StartAddress {
        address.divided_by(by)
    }
}

/// `C` — the coordinate and work-slice tables entry 260's coordinate-based constant offset reaches
/// through.
///
/// ⛔⛔ THE LAST TWO METHODS ARE `util/foldManager/foldInfrastructure.h`, OUTSIDE THIS CAMPAIGN'S
/// FILE LIST, and both are fold ALGEBRA over an address: `single_beta` evaluates a fold dim at one
/// `(core, corelet, row)` and `distance_in_steps` solves a lexicographic affine distance. A guessed
/// coefficient is a wrong address, so neither is answered.
#[derive(Debug, Clone, Copy, Default)]
pub struct DdcCoords;

impl v1::CoordinateOffsets for DdcCoords {
    fn node_work_slices(
        &self,
        _at: v1::OperandSite,
        _core: crate::units::Core,
    ) -> Option<BTreeMap<PrimaryDim, v1::WorkSlice>> {
        todo!(
            "v1::CoordinateOffsets::node_work_slices: wants coordinates.coreIdToWkSlice_ on the \
             node's own Coordinate, falling back to sdsc_->coreIdToWkSlice_ — a per-node table \
             entry 375's coordinate capture is what fills"
        )
    }

    fn alloc_work_slices(
        &self,
        _alloc: crate::schedule::ddc::fold::AllocId,
        _core: crate::units::Core,
    ) -> Option<BTreeMap<PrimaryDim, v1::WorkSlice>> {
        todo!(
            "v1::CoordinateOffsets::alloc_work_slices: wants the same off the allocation's \
             sliceViewCoordinates_/allocateCoordinates_"
        )
    }

    fn node_coordinate(
        &self,
        _at: v1::OperandSite,
        _offsets: v1::ElemOffsets,
    ) -> crate::schedule::dsc2::Coordinate {
        todo!(
            "v1::CoordinateOffsets::node_coordinate: wants transferCoordinates_/\
             inputCoordinates_.at(i)/outputCoordinate_ on that node — the coordinates entry 375 \
             captures"
        )
    }

    fn alloc_coordinate(
        &self,
        _alloc: crate::schedule::ddc::fold::AllocId,
    ) -> crate::schedule::dsc2::Coordinate {
        todo!(
            "v1::CoordinateOffsets::alloc_coordinate: wants sliceViewCoordinates_ where its \
             coordinates_ is non-empty, else allocateCoordinates_"
        )
    }

    fn relevant_core_cl(&self, _node: crate::schedule::ddc::fold::NodeId) -> v1::CoreClSet {
        todo!(
            "v1::CoordinateOffsets::relevant_core_cl: wants getRelevantCoreCl() (dsc/dsc2.h:471), \
             which dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647) is what fills — a `dsc/` seam"
        )
    }

    fn single_beta(
        &self,
        _folds: &crate::schedule::dsc2::FoldDim,
        _core: i64,
        _corelet: i64,
        _row: i64,
    ) -> crate::schedule::dsc2::FoldCoeff {
        todo!(
            "v1::CoordinateOffsets::single_beta: wants getSingleData({{Core, core}}, {{Corelet, \
             corelet}}, {{RowSplit, row}}) — fold algebra in util/foldManager/\
             foldInfrastructure.h, outside this campaign's file list"
        )
    }

    fn distance_in_steps(
        &self,
        _folds: &crate::schedule::dsc2::FoldDim,
        _beta: crate::schedule::dsc2::FoldCoeff,
        _fixed: &BTreeMap<usize, i64>,
    ) -> v1::ConstEleOffset {
        todo!(
            "v1::CoordinateOffsets::distance_in_steps: wants \
             FoldInfraUtils::lexiAffineSolveDistanceInSteps(folds, beta, fixed) — \
             util/foldManager/foldInfrastructure.h, outside this campaign's file list"
        )
    }
}

/// ⛔ HELPERS THE STATE HANDS ITS VIEWS — a labelled DS's `dsType_`, which several traits key
/// `primaryDsInfo_` by.
pub(super) fn ds_type_of(dsc: &DesignSpaceConfig, lds: LdsIdx) -> Option<
    crate::schedule::ddc::transformation::DsType,
> {
    dsc.labeled_ds
        .indexed()
        .find(|(at, _)| *at == lds)
        .map(|(_, held)| held.ds_type())
}

/// ⛔ AND THE STICK DIMS THAT `dsType_` NAMES — `primaryDsInfo_.at(dsType_)`'s `stickDimOrder_`
/// zipped with its `stickSize_`, which is exactly what [`crate::schedule::l3::dsc::PrimaryDsInfo`]
/// already holds.
pub(super) fn stick_dims_of(
    dsc: &DesignSpaceConfig,
    lds: LdsIdx,
) -> Option<crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims> {
    let ds_type = ds_type_of(dsc, lds)?;
    dsc.primary_ds_info
        .get(&ds_type)
        .map(|info| info.stick.clone())
}

/// ⛔ `getNonBroadcastLdsDimSet(lds)` (`dsc/dsc2.cpp:4050`) as
/// [`DesignSpaceConfig::non_broadcast_lds_dims`] answers it — EMPTY for an index the DSC does not
/// hold, which is the reference's own `ldsIdx < 0` arm.
pub(super) fn non_broadcast_dims_of(
    dsc: &DesignSpaceConfig,
    lds: LdsIdx,
) -> BTreeSet<PrimaryDim> {
    dsc.non_broadcast_lds_dims(lds)
        .map(|dims| dims.into_iter().collect())
        .unwrap_or_default()
}

/// Every `labeledDs_` position, in order — `LabeledDsList::indexed`'s keys.
pub(super) fn lds_positions(dsc: &DesignSpaceConfig) -> Vec<LdsIdx> {
    dsc.labeled_ds.indexed().map(|(at, _)| at).collect()
}

/// The corelets `numCoreletsUsed_` names, as corelets rather than a count.
pub(super) fn corelets_of(count: u32) -> Vec<Corelet> {
    (0..count).filter_map(Corelet::checked).collect()
}

