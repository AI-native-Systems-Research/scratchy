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
use crate::schedule::ddc::fold::{self, PadType};
use crate::schedule::ddc::transformation_util::{
    DataStage as UtilDataStage, DataStages as UtilDataStages, DimSplit, PaddingForm,
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
/// ⭐ EVERY MAP IS `l3::dsc::StageDims`' OWN — `primaryDimToValHandler_st`'s slots,
/// `paddingSizes_`, `symbolicDimInfo_`/`maxSymbolicVolume_`, `coreletSplit_`, `rowSplit_` and
/// `peSfpSplit_`.
///
/// ⛔ `rowSplit_` AND `peSfpSplit_` WERE STATED HERE BESIDE IT, AND A SECOND HOME IS WHY THEY MOVED:
/// entry 012 ([`StageDims::sampled_extent`]) is the fold that reads them, so a duplicate here would
/// leave that fold looking at an EMPTY map at this call site and answering the whole core for one
/// row.
#[derive(Debug, Clone, Default)]
pub struct Dsc2Dims {
    /// `name_`.
    pub name: StageName,
    /// Every map of `DataStructDims` this crate models.
    pub dims: StageDims,
}

/// [`Sample`]'s corelet is always one corelet; `primaryDimToVal_st`'s `clId` is `-1` for the whole
/// core (`dsc/dims.h:250-255`), which is what [`v1::DimSample`] spells.
const fn dim_sample(at: Sample) -> v1::DimSample {
    v1::DimSample {
        comp: at.comp,
        row: at.row,
        corelet: Some(at.corelet),
    }
}

impl Dsc2Dims {
    /// ⭐⭐ `primaryDimToVal_base_st`'S PLAIN FIELD READ, AND NOTHING ELSE —
    /// `dsc/dims.cpp:516-560`, the same citation entry 207 already stands on in this crate.
    ///
    /// ⛔⛔ [`None`] WHERE THE AUTHORITY'S CONTROL FLOW LEAVES THAT READ, and that is deliberate:
    /// this reader answers only where the authority provably returns the stored slot — the dim is
    /// not symbolic, no split names it, and the padding is `NOPAD`.
    ///
    /// ⭐ THE FOLD ABOVE IT IS [`StageDims::sampled_extent`] (entry 012, `dsc/dims.cpp:653-704`),
    /// which folds `rowSplit_`, then `peSfpSplit_`, then `coreletSplit_` (`:631-645`) and
    /// `calculate_padded` (`:563-616`). ⛔ THIS READER IS STILL WHAT ANSWERS THE UNSTATED SLOT, since
    /// that fold's [`None`] cannot tell the reference's own `-1` from the reference aborting, and
    /// answering a stop with a number is the failure this crate ranks worse than a stop.
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
        if self.dims.row_split.contains_key(&dim)
            || self.dims.pe_sfp_split.contains_key(&dim)
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
            DimSplit::Row => self.dims.row_split.keys().copied().collect(),
            DimSplit::PeSfp => self.dims.pe_sfp_split.keys().copied().collect(),
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
        self.dims.row_split.contains_key(&dim)
    }

    fn is_pe_sfp_split(&self, dim: PrimaryDim) -> bool {
        self.dims.pe_sfp_split.contains_key(&dim)
    }

    fn splits_any_row(&self) -> bool {
        !self.dims.row_split.is_empty()
    }

    /// `primaryDimToVal_st(dim, comp, row, cl)` — [`StageDims::sampled_extent`] (entry 012) at this
    /// sample, with the reference's own defaults for the padding form, the density and the symbolic
    /// read (`dsc/dims.cpp:653-704`).
    ///
    /// ⛔ THE UNSTATED SLOT IS THE REFERENCE'S OWN `-1` AND NOT A REFUSAL, so [`Self::raw_slot`]
    /// still answers it — 187 of 187 g0 programs read a dim their stage leaves at `-1`. Only what the
    /// FOLD stops on (a corelet or row the split does not name, an aborted `calculate_padded`) is a
    /// `todo!`, because the trait's return type is TOTAL and a substituted extent is a fabricated
    /// one.
    ///
    /// ⚠️⚠️ ONE RESIDUAL DIVERGENCE, RECORDED AND NOT PAPERED OVER — [`Self::raw_slot`] guards on
    /// *"no split names the dim"*, but the authority's guard is *"this SAMPLE did not enter that
    /// split's arm"*. `primaryDimToVal_st` takes the row arm only when `ptrowId >= 0 &&
    /// rowSplit_.count(d)` (`dsc/dims.cpp:665`) and the PE/SFP arm only when the component is one of
    /// the two (`:682-683`); otherwise it falls through to `primaryDimToVal_clView_st` and, with the
    /// dim absent from `coreletSplit_`, to `primaryDimToVal_base_st`'s `-1`. So a dim that IS
    /// row- or PE/SFP-split, is NOT corelet-split, has NO stated extent and is not symbolic, read at
    /// a sample naming neither a row nor a component, `todo!`s here where the authority answers
    /// `-1`.
    ///
    /// ⛔ IT IS LEFT AS A STOP DELIBERATELY, on two grounds. Closing it means re-stating
    /// [`StageDims::sampled_extent`]'s three-way dispatch at this call site — a SECOND HOME for the
    /// decision, which is the defect this file's own header records against `rowSplit_`/
    /// `peSfpSplit_` — and the case is unproven on the corpus: a stage that splits a dim across rows
    /// but states no size for it. A loud stop is recoverable; a substituted extent is the
    /// fabricated extent this crate ranks worse than a stop. ⛔ AND [`Self::raw_slot`] MUST NOT BE
    /// WIDENED to fix it: three other carriers read it as the *"provably the stored slot"*
    /// predicate.
    fn extent(&self, dim: PrimaryDim, at: Sample) -> Extent {
        if let Some(extent) =
            self.dims
                .sampled_extent(dim, dim_sample(at), &PaddingForm::default(), None, false)
        {
            return extent;
        }
        // The whole of every axis, and the dim unstated: `primaryDimToVal_base_st` answers its `-1`
        // slot (`dsc/dims.cpp:516-560`) and every arm above passes that through.
        if let Some(extent) = self.raw_slot(dim, PadType::NoPad, v1::SymbolicRead::Max) {
            return extent;
        }
        todo!(
            "Stage::extent: primaryDimToVal_st (dsc/dims.cpp:653-704) STOPS on {dim:?} at {at:?} — \
             the split names neither that corelet nor that row, or calculate_padded (:563-616) \
             aborted. Substituting the whole core's extent there would be a fabricated extent."
        )
    }

    /// The same with `PADDED_WZEROPAD` on `dim` — one `PaddingFormType` handed to the same fold, so
    /// `calculate_padded`'s window span (`dsc/dims.cpp:596-602`) is what comes back.
    ///
    /// ⛔ [`None`] IS *"dim not relevant"* AND EVERY ABORT OF THAT REWRITE AT ONCE — see
    /// [`PaddedExtent`], where the `DT_ERROR` this feeds is argued unreachable from a positive span.
    fn padded_extent(&self, dim: PrimaryDim, at: Sample) -> Option<PaddedExtent> {
        let mut padded = PaddingForm::default();
        padded.set_padding(dim, PadType::PaddedWZeroPad);
        PaddedExtent::of(
            self.dims
                .sampled_extent(dim, dim_sample(at), &padded, None, false)?
                .0,
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
                if let Some(held) = other.dims.row_split.get(&dim) {
                    self.dims.row_split.insert(dim, held.clone());
                }
            }
            DimSplit::PeSfp => {
                if let Some(held) = other.dims.pe_sfp_split.get(&dim) {
                    self.dims.pe_sfp_split.insert(dim, held.clone());
                }
            }
            DimSplit::Padding => {
                if let Some(held) = other.dims.padding.get(&dim) {
                    self.dims.padding.insert(dim, *held);
                }
            }
        }
    }

    /// ⛔⛔ NOT AN ERASE. `makeDimNotSymbolic` (`dsc/dims.cpp:781-804`) erases the entry AND THEN
    /// DIVIDES every stated size for that dim — `primaryDimToValHandler_st(dim)`, every
    /// `coreletSplit_` share, every `rowSplit_` share and every `peSfpSplit_` side — by
    /// `maxSize_ / granularity_`. Dropping the divide leaves every extent that factor too LARGE, and
    /// an extent that large becomes an oversized buffer and a wrong address. So the divide is named
    /// rather than skipped.
    ///
    /// ⭐ EVERY MAP IT WRITES IS ALREADY HERE — [`StageDims`]'s `symbolic`, `extents`,
    /// `corelet_split`, `row_split` and `pe_sfp_split`. What is missing is the 22-line
    /// TRANSFORMATION, which is a port and not a field read, so it is named rather than written
    /// here: three `DT_CHECK`s (`maxSize_ % granularity_ == 0`, `factor != 0`, and `val % factor == 0`
    /// per divide) are refusals a `&mut self` returning `()` has nowhere to put.
    ///
    /// ⭐ AND THIS IMPL IS ITS ONE HOME, beside its four sibling map writers. The same question at
    /// `v1::ExploreStages::make_dim_not_symbolic` (`stages/ddc_sites.rs`) reaches this very
    /// [`Dsc2Dims`] through [`Dsc2Facts::with_stages_mut`], so it delegates here rather than holding
    /// a second copy — which is what this file's own header says about `rowSplit_`/`peSfpSplit_`.
    /// ⚠️ `transformation_util::StageExtents::make_dim_not_symbolic` cites `dsc/dims.cpp:717`; the
    /// function is at `:781`.
    fn make_dim_not_symbolic(&mut self, _dim: PrimaryDim) {
        todo!(
            "UtilStageExtents::make_dim_not_symbolic: wants makeDimNotSymbolic \
             (dsc/dims.cpp:781-804), UNPORTED — it ERASES symbolicDimInfo_[dim] and then divides the \
             dim value and every corelet/row/PE-SFP share by maxSize_/granularity_. Every map it \
             writes is on this StageDims already; the transformation is not. A bare erase leaves \
             every extent that factor too large."
        )
    }

    fn clear_split(&mut self, split: DimSplit) {
        match split {
            DimSplit::Corelet => self.dims.corelet_split.clear(),
            DimSplit::Row => self.dims.row_split.clear(),
            DimSplit::PeSfp => self.dims.pe_sfp_split.clear(),
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
                .dims
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
                 VectorComp (dsc/dims.h:212-214); a flat Vec<Elements> cannot say which side a size \
                 belongs to, and entry 300 reads .at(0)/.at(1) positionally. NOT a missing fact — \
                 pe_sfp_split holds both sides; the RETURN TYPE cannot spell them, and no caller \
                 asks: entry 300's only split_sizes call is DimSplit::Corelet \
                 (schedule/ddc/transformation.rs:1759, ddc/ddc_transformation.cpp:920-922). \
                 An empty Vec here would read as `not split`, which pe_sfp_split refutes"
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
    ///
    /// ⭐⭐ BEHIND THE SAME CELL [`Self::stages`] AND [`Self::ops`] ARE, AND FOR THE SAME REASON:
    /// entry 308 (`prep_dsc`) and entry 372 (`AutoShuffling`) REWRITE `labeledDs_` — `dsName_`,
    /// `wordLength`, `dataFormat_`, `scaledLdsCategory_`, `dsType_` — through `&mut self` methods on
    /// [`super::Dsc2Store`], while [`super::Dsc2Reads`] reads the SAME entries through `&self`. The
    /// reference has one `currDsc`, so a second copy for the writers would make entry 308's writes
    /// invisible to entry 307's reads — the defect review 382 recorded on `L3RunInputs`.
    ///
    /// ⭐ SOUND FOR THE REASON THE MODULE HEADER GIVES: every design-space trait answers BY VALUE, so
    /// no borrow of the interior outlives the call that took it — which is why the accessors below are
    /// CLOSURES and not a [`std::cell::Ref`] a caller could hold across a write.
    dsc: RefCell<DesignSpaceConfig>,
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

    /// The design space, for the length of ONE read.
    pub(super) fn with_dsc<T>(&self, ask: impl FnOnce(&DesignSpaceConfig) -> T) -> T {
        ask(&self.dsc.borrow())
    }

    /// The design space, for the length of ONE write — `currDsc->...= ` as entries 308 and 372 make
    /// it.
    pub(super) fn with_dsc_mut<T>(&self, write: impl FnOnce(&mut DesignSpaceConfig) -> T) -> T {
        write(&mut self.dsc.borrow_mut())
    }

    /// `labeledDs_.at(lds)`, for one read — ⛔ [`None`] IS that `.at()`'s own throw.
    pub(super) fn with_lds<T>(
        &self,
        lds: LdsIdx,
        ask: impl FnOnce(&crate::schedule::l3::dsc::LabeledDs) -> T,
    ) -> Option<T> {
        self.with_dsc(|dsc| dsc.labeled_ds.at(lds).map(ask))
    }

    /// `labeledDs_.at(lds)`, for one write — the five setters entry 308 and entry 372 rewrite an
    /// entry's record with. ⛔ [`None`] IS THE `.at()` THROW, so a write to an absent index is
    /// reported and not silently dropped.
    pub(super) fn with_lds_mut<T>(
        &self,
        lds: LdsIdx,
        write: impl FnOnce(&mut crate::schedule::l3::dsc::LabeledDs) -> T,
    ) -> Option<T> {
        self.with_dsc_mut(|dsc| dsc.labeled_ds.at_mut(lds).map(write))
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
                dsc: RefCell::new(dsc.clone()),
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
                    },
                },
                el: UtilStageDims {
                    name: stage.el.name.clone(),
                    dims: Dsc2Dims {
                        name: stage.el.name.clone(),
                        dims: stage.el.dims.dims().clone(),
                    },
                },
            },
        );
    }
    map
}

/// `M` — `memTrackers` as stage 2b reaches them, which is a DIFFERENT trait from stage 2a's
/// [`crate::schedule::l3::dl_ops::ExPhaseTrackers`] over the SAME allocator.
///
/// ⛔⛔ THE ALLOCATOR IS PORTED. THIS TYPE'S OWN `todo!`s SAID OTHERWISE FOR SEVEN METHODS AND THAT
/// WAS FALSE — `util/memtracker/mem_track.{h,cpp}` is
/// [`crate::schedule::memtrack::tracker::DsTrackInMem`], ported unit by unit (`e003`..`e038`) by the
/// `crustify-memtrack` campaign, and `sys-arch-spec/memtracker/mem_track_bundle.{h,cpp}` is
/// [`crate::schedule::memtrack::bundle::MemTrackBundle`], which IS
/// `getTracker(comp, core, corelet, row)`. All seven of this trait's questions have a ported answer:
/// `memCapacity` is the tracker's own field (`e032_initMemTrack`), `backupEps` is `e023`,
/// `restoreEps` `e035`, `removeDs` `e019`, `addDsAtStartAddr` `e031`, `checkAndAddDs` `e037` and
/// `checkAndAddDsAtAddr` `e038`. ⚠️ `v1::MemTrackers`' own doc still says *"outside this campaign's
/// file list"*; that is the same stale claim one file up.
///
/// ⛔⛔ SO EVERY `todo!` BELOW IS A WIRING GAP, AND ALL SEVEN HAVE ONE CAUSE: this is a UNIT struct.
/// [`super::Dsc2Provider`] holds it as a FIELD and states `type Trackers = ddc_state::DdcTrackers`
/// (`stages/ddc_sites.rs`), and `Dsc2Provider::new` has no `A: Arch` — so nothing there can seed a
/// bundle, and giving this type the `MemTrackBundle` field it needs stops that file compiling.
/// [`super::Trackers`](super::carriers::Trackers) is the shape to reach: it already owns a real
/// `MemTrackBundle<DsTrackInMem>` and answers the SAME seven questions for stage 2a.
///
/// ⛔⛔ AND A SECOND FACT NO FIELD ON THIS TYPE FIXES: `run_l3` builds its
/// [`super::Trackers`](super::carriers::Trackers) as a LOCAL and drops it (`schedule/stages.rs`,
/// `run_l3`). One bundle must be threaded from stage 2a into stage 2b, because a fresh bundle here
/// would place stage 2b's allocations OVER stage 2a's — the same LX bytes handed out twice.
///
/// ⚠️ AND EVEN WIRED, A NON-LX SITE STILL STOPS: the bundle holds the LX family only, because
/// `initializeMemoryTrackers` reads `regInfoPerUnit.at(LXLU).at(RegType::SCALE)` and
/// `sys_arch_spec::regfile::RegType` has no `SCALE` arm — see
/// [`super::Trackers`](super::carriers::Trackers). Stage 2b is the stage that places the
/// register-file and L0 nodes, so that gap is on its path, not stage 2a's.
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
            "v1::MemTrackers::capacity: memCapacity IS PORTED — DsTrackInMem::mem_capacity, set by \
             e032_initMemTrack (schedule/memtrack/tracker.rs). This unit struct holds no \
             MemTrackBundle to read it from; see DdcTrackers"
        )
    }

    fn backup(&mut self, _at: v1::TrackerSite) {
        todo!(
            "v1::MemTrackers::backup: backupEps(exphase) IS PORTED — e023_backupEps \
             (schedule/memtrack/tracker.rs); this unit struct holds no bundle. See DdcTrackers"
        )
    }

    fn restore_all(&mut self) {
        todo!(
            "v1::MemTrackers::restore_all: restoreEps(exphase, backupInfo) IS PORTED — \
             e035_restoreEps; and the snapshot map this replays is state a unit struct cannot keep. \
             See DdcTrackers"
        )
    }

    fn remove(&mut self, _at: v1::TrackerSite, _name: &v1::StorageName) {
        todo!(
            "v1::MemTrackers::remove: removeDs(name, seps) IS PORTED — e019_removeDs; this unit \
             struct holds no bundle. See DdcTrackers"
        )
    }

    fn add_at(
        &mut self,
        _at: v1::TrackerSite,
        _name: &v1::StorageName,
        _size: crate::arch::Bytes,
        _address: crate::arch::Bytes,
    ) {
        todo!(
            "v1::MemTrackers::add_at: addDsAtStartAddr(name, size, seps, addr) IS PORTED — \
             e031_addDsAtStartAddr; this unit struct holds no bundle. See DdcTrackers"
        )
    }

    fn check_and_add(
        &mut self,
        _at: v1::TrackerSite,
        _name: &v1::StorageName,
        _size: crate::arch::Bytes,
    ) -> Option<v1::Placed> {
        todo!(
            "v1::MemTrackers::check_and_add: checkAndAddDs(name, size, seps) IS PORTED — \
             e037_checkAndAddDs, and stage 2a already drives it (carriers::Trackers, whose \
             lx_oracle test reproduces all 588 LX addresses of the 187 reference programs). THIS \
             CALL IS THE PLACEMENT AUTHORITY and it COMMITS, so it must run on the SAME bundle \
             stage 2a placed into — run_l3 drops its own. See DdcTrackers"
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
            "v1::MemTrackers::check_and_add_at: checkAndAddDsAtAddr(name, size, seps, addr) IS \
             PORTED — e038_checkAndAddDsAtAddr; this unit struct holds no bundle, and it commits on \
             the same bundle stage 2a placed into. See DdcTrackers"
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
/// ⛔⛔ THE FOUR TABLE READERS ARE BLOCKED BY THIS TYPE'S OWN SHAPE, NOT BY A MISSING PORT, and the
/// distinction decides who fixes them. This is a UNIT struct: `Dsc2Provider` holds it as
/// `trackers`/`coords` FIELDS and states `type Coords = ddc_state::DdcCoords`
/// (`stages/ddc_sites.rs`, `Dsc2Provider::new` and its `impl v1::Dsc2Sites`), so it owns nothing and
/// borrows nothing, while every one of those four reads needs the state:
/// * `sdsc_->coreIdToWkSlice_`, the fallback BOTH work-slice readers take — already held, by
///   [`Dsc2State::core_wk_slices`].
/// * `allocNode->allocateCoordinates_` — already held, by `TreeData::coordinate` at
///   `TreeData::node_of_alloc`'s node (`stages/tree.rs`).
/// So `alloc_work_slices` and `alloc_coordinate` are a CONSTRUCTION-SITE change (hand this carrier
/// `&Dsc2State` and the [`DscIdx`] whose tree it reads, exactly as [`super::Dsc2Store`] and
/// [`super::Dsc2Stages`] are handed them), and adding a field here without that change stops
/// `ddc_sites.rs` compiling. ⛔ IT IS NOT DONE HERE: that file is not this file.
///
/// ⛔ AND TWO FACTS ARE GENUINELY ABSENT, not merely unreachable — `sliceViewCoordinates_`, which
/// `alloc_coordinate` prefers over `allocateCoordinates_` when non-empty, and the per-node
/// `transferCoordinates_`/`inputCoordinates_`/`outputCoordinate_` `node_coordinate` wants. The tree
/// carries ONE coordinate map and only `clear_allocate_coordinates`/`clear_transfer_coordinates`
/// (`schedule/ddc/mod.rs:178-180`) and `resize_input_coordinates` (`schedule/ddc/fold.rs:7507`) name
/// the others, so nothing writes them yet.
///
/// ⛔ ONE METHOD IS STILL FOLD ALGEBRA OUTSIDE THIS CAMPAIGN'S FILE LIST — see
/// [`Self::distance_in_steps`]. ⭐ `single_beta` IS NOT, and the prose that grouped the two was
/// wrong: [`fold::single_data`] is `getSingleData` and is already in this crate.
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
             node's own Coordinate (ddc/ddcv1.cpp:2690-2692), falling back to \
             sdsc_->coreIdToWkSlice_. THE FALLBACK IS HELD (Dsc2State::core_wk_slices); the node's \
             own Coordinate is NOT — see Self::node_coordinate — and this carrier is a unit struct \
             that cannot reach either. Wiring it is a change in stages/ddc_sites.rs"
        )
    }

    fn alloc_work_slices(
        &self,
        _alloc: crate::schedule::ddc::fold::AllocId,
        _core: crate::units::Core,
    ) -> Option<BTreeMap<PrimaryDim, v1::WorkSlice>> {
        todo!(
            "v1::CoordinateOffsets::alloc_work_slices: wants the same off the allocation's \
             sliceViewCoordinates_/allocateCoordinates_ (ddc/ddcv1.cpp:2696-2700) — i.e. \
             Self::alloc_coordinate's core_id_to_wk_slice, else Dsc2State::core_wk_slices. BOTH \
             HALVES ARE HELD once this carrier can reach the state; it is a unit struct, so wiring \
             it is a change in stages/ddc_sites.rs"
        )
    }

    fn node_coordinate(
        &self,
        _at: v1::OperandSite,
        _offsets: v1::ElemOffsets,
    ) -> crate::schedule::dsc2::Coordinate {
        todo!(
            "v1::CoordinateOffsets::node_coordinate: wants transferCoordinates_/\
             inputCoordinates_.at(i)/outputCoordinate_ on that node — a MISSING FIELD, not a \
             borrow: TreeData holds one coordinate map (allocateCoordinates_) and only \
             clear_transfer_coordinates (schedule/ddc/mod.rs:180) and resize_input_coordinates \
             (schedule/ddc/fold.rs:7507) name these, so nothing writes them. The field belongs on \
             the tree's node record (stages/tree.rs)"
        )
    }

    fn alloc_coordinate(
        &self,
        _alloc: crate::schedule::ddc::fold::AllocId,
    ) -> crate::schedule::dsc2::Coordinate {
        todo!(
            "v1::CoordinateOffsets::alloc_coordinate: wants sliceViewCoordinates_ where its \
             coordinates_ is non-empty, else allocateCoordinates_ (ddc/ddcv1.cpp:2696-2698). \
             allocateCoordinates_ IS HELD (TreeData::coordinate at node_of_alloc(alloc)) and \
             sliceViewCoordinates_ is NOT MODELLED AT ALL, so answering with the one we have would \
             silently prefer the wrong table wherever a slice view exists"
        )
    }

    /// ⛔ AN UNPORTED PASS AND A MISSING NODE FIELD, AND BOTH ARE ONE THING. `getRelevantCoreCl()`
    /// (`dsc/dsc2.h:471`) reads `ScheduleNode::relevantComps_`, and the only writer is
    /// `DesignSpaceConfig::setRelevantCompCoreCl()` (`dsc/dsc2.cpp:2647-2712`) — a whole DFS over
    /// `scheduleTree_` that seeds the head with `{coreIdsUsed_} x {0..numCoreletsUsed_DSC2_}`, then
    /// INTERSECTS each `CONDITION`'s then-region with its `coreClCond_` and SUBTRACTS it from the
    /// else-region, then unions the per-component views upward. Nothing in this crate carries
    /// `relevantComps_`, so the answer cannot be read off a field.
    ///
    /// ⛔ THE WHOLE-CORE SET IS NOT THE ANSWER, which is why nothing is substituted: the caller uses
    /// it to SKIP sites (`schedule/ddc/v1.rs:2868`), so widening it writes a constant offset at a
    /// `(core, corelet)` the conditionals exclude.
    ///
    /// ⚠️ THE SAME QUESTION STANDS TWICE — `v1::ConditionSimplification::relevant_core_cl`
    /// (`stages/ddc_store.rs`) is the same `getRelevantCoreCl()` on the tree carrier, so ONE port of
    /// that pass answers both.
    fn relevant_core_cl(&self, _node: crate::schedule::ddc::fold::NodeId) -> v1::CoreClSet {
        todo!(
            "v1::CoordinateOffsets::relevant_core_cl: wants getRelevantCoreCl() (dsc/dsc2.h:471) \
             off ScheduleNode::relevantComps_, which ONLY setRelevantCompCoreCl \
             (dsc/dsc2.cpp:2647-2712) fills — an unported tree pass AND a node field this crate \
             does not carry. The whole-core set is not a substitute: the caller SKIPS sites by it"
        )
    }

    /// `dimCoord.getSingleData({{Core, core}, {Corelet, corelet}, {RowSplit, row}})`
    /// (`ddc/ddcv1.cpp:2761-2768`) — the node's own fold value at ONE spatial site, which is the
    /// `beta` `lexiAffineSolve` is then asked to hit.
    ///
    /// ⭐⭐ IT IS ALREADY IN THIS CRATE, AND THE PROSE THAT SAID OTHERWISE WAS WRONG.
    /// [`fold::single_data`] IS `getSingleData` (`util/foldManager/foldInfrastructure.h:1934`) — a
    /// `getData` over a deque holding the named positions and zero everywhere else — written out as
    /// the all-affine sum `Σ_i (α_i·idx_i + β_i)`, and entry 237 already reads it exactly this way
    /// (`schedule/ddc/fold.rs:3300`). [`crate::schedule::dsc2::FoldDim`] implements
    /// [`fold::AffineFoldDims`] (`schedule/ddc/fold.rs:1877`), so no walk and no new type is needed.
    ///
    /// ⭐ THE THREE ARGUMENTS ARRIVE ALREADY SELECTED. The reference fixes `Core` at
    /// `useCoreIdNode ? sliceIdNode : 0`, `Corelet` at `useClIdNode ? clId : 0` and `RowSplit` at
    /// `rowIdNode`; the caller performs those three choices (`schedule/ddc/v1.rs:2894-2900`) and
    /// hands the results down, so this reader fixes the three positions and nothing else.
    ///
    /// ⛔ TOTAL WHERE `fold_dim_indices.at(dim)` THROWS — a coordinate with fewer than three fold
    /// levels. That is not a decision made here: the sum runs over the levels the dim HAS, so a
    /// fixed position past the last one contributes nothing, which is
    /// [`fold::AffineFoldDims`]'s own recorded stance for this type (*"an index past the last level
    /// answers `FoldParamInfoType`'s own defaults"*).
    fn single_beta(
        &self,
        folds: &crate::schedule::dsc2::FoldDim,
        core: i64,
        corelet: i64,
        row: i64,
    ) -> crate::schedule::dsc2::FoldCoeff {
        crate::schedule::dsc2::FoldCoeff(
            fold::single_data(
                folds,
                &[
                    (fold::FoldPosition::Core, core),
                    (fold::FoldPosition::Corelet, corelet),
                    (fold::FoldPosition::RowSplit, row),
                ],
            )
            .0,
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
             FoldInfraUtils::lexiAffineSolveDistanceInSteps (foldInfrastructure.h:3023) — THREE \
             unported units, and a SOLVER rather than an evaluation, which is why single_beta's \
             fix does not reach it: lexiAffineSolve (:2965) calls \
             LexiAffineSolver(alphas, factors, beta).solve(target) (util/utils.h:184), a pruned DFS \
             for the lexicographically smallest coordinate hitting the target, then \
             coordDistanceInSteps (:3001) walks it back to a step count. Both headers are outside \
             this campaign's file list"
        )
    }
}

/// ⛔ HELPERS THE STATE HANDS ITS VIEWS — a labelled DS's `dsType_`, which several traits key
/// `primaryDsInfo_` by.
pub(super) fn ds_type_of(dsc: &DesignSpaceConfig, lds: LdsIdx) -> Option<
    crate::schedule::ddc::transformation::DsType,
> {
    // `labeledDs_.at(lds)` — ⛔ THE POSITION AND NOT `LabeledDs::recorded`, which is what
    // `LabeledDsList::at` already spells; the two can drift.
    dsc.labeled_ds.at(lds).map(|held| held.ds_type())
}

/// ⛔ AND THE STICK DIMS THAT `dsType_` NAMES — `primaryDsInfo_.at(dsType_)`'s `stickDimOrder_`
/// zipped with its `stickSize_`, which is exactly what [`crate::schedule::l3::dsc::PrimaryDsInfo`]
/// already holds.
///
/// ⭐ ITS TWO EMPTIES ARE DISTINGUISHABLE, AND THAT WAS CHECKED RATHER THAN ASSUMED. [`None`] is a
/// THROW and only a throw — `labeledDs_.at(lds)`'s (through [`ds_type_of`]) or
/// `primaryDsInfo_.at(dsType_)`'s — while `Some(`empty [`StickDims`]`)` is a `dsType_` whose
/// `stickDimOrder_` is genuinely empty. So a caller CAN tell the two apart, and neither is a
/// swallowed refusal. Contrast [`non_broadcast_dims_of`], whose total return type cannot.
///
/// [`StickDims`]: crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims
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
/// [`DesignSpaceConfig::non_broadcast_lds_dim_set`] answers it — the labelled DS's OWN
/// `primaryDsInfo_.at(dsType_).layoutDimOrder_` filtered to `scale_ > 0`.
///
/// ⛔⛔ IT USED TO CALL [`DesignSpaceConfig::non_broadcast_lds_dims`], WHICH IS A DIFFERENT FUNCTION
/// AND A SMALLER SET. `getNonBroadcastLdsDims` (`dsc/dsc2.cpp:4039-4048`) is this set INTERSECTED
/// with `getLayoutDims(ldsIdx)` — the ALLOCATE node's `layoutDimOrder_` (`:4007-4025`) — so every
/// non-broadcast dim the allocate node does not name was silently DROPPED. Both callers ask for the
/// set: `v1::…::non_broadcast_dims` (`schedule/ddc/v1.rs:3370`) and
/// `tr::ScopeTree::non_broadcast_lds_dims` (`schedule/ddc/transformation.rs:958`) each cite
/// `getNonBroadcastLdsDimSet`, and so do the reference sites they stand on
/// (`ddc/ddc_transformation.cpp:238`, `:1457`, both of which take a `std::set`).
/// ⭐ `l3::dsc` ALREADY NAMED THIS TRAP — [`DesignSpaceConfig::non_broadcast_lds_dim_set`]'s own doc
/// says *"THIS IS NOT `non_broadcast_lds_dims`, which intersects this set with the ALLOCATE node's
/// order"*. Seven `dl_ops` sites already ask for the right one.
///
/// ⚠️ THE EMPTY IS STILL INDISTINGUISHABLE FOR ONE CASE, AND THAT IS RECORDED RATHER THAN HIDDEN:
/// [`None`] here is `labeledDs_.at(ldsIdx)` THROWING, and the reference guards only `ldsIdx < 0` —
/// which [`LdsIdx`]'s unsignedness makes unspellable. Both trait returns are a bare
/// `BTreeSet<PrimaryDim>`, so there is nowhere to put that throw; a caller cannot tell it from a
/// wholly broadcast structure's genuine `{}`. ⛔ THE ONE THING THAT WOULD FIX IT is those two trait
/// returns becoming [`Option`], which is not this file's to change.
pub(super) fn non_broadcast_dims_of(
    dsc: &DesignSpaceConfig,
    lds: LdsIdx,
) -> BTreeSet<PrimaryDim> {
    dsc.non_broadcast_lds_dim_set(lds).unwrap_or_default()
}

/// Every `labeledDs_` position, in order — `LabeledDsList::indexed`'s keys.
pub(super) fn lds_positions(dsc: &DesignSpaceConfig) -> Vec<LdsIdx> {
    dsc.labeled_ds.indexed().map(|(at, _)| at).collect()
}

/// The corelets `numCoreletsUsed_` names, as corelets rather than a count.
pub(super) fn corelets_of(count: u32) -> Vec<Corelet> {
    (0..count).filter_map(Corelet::checked).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::schedule::ddc::transformation::{DsType, Scale};
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, FilledDims, LabeledDs, LabeledDsList,
        NamedDims, Pinning,
    };

    /// One stated dim, which is all [`FilledDims::of`] asks for.
    fn stage(name: &str) -> DataStage {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::I, Extent(1));
        let half = || NamedDims {
            name: StageName(name.to_owned()),
            dims: FilledDims::of(dims.clone()).expect("a stage that states a dim"),
        };
        DataStage {
            ss: half(),
            el: half(),
        }
    }

    /// A DSC holding exactly the two labelled structures the assertions below read, and NO
    /// `layout_dims` entry for either — which is `getLayoutDims`' `DT_CHECK(allocNode)`
    /// (`dsc/dsc2.cpp:4022`), the abort that separates the two reference functions.
    fn dsc_without_layouts() -> DesignSpaceConfig {
        // `scale_ = 0` is broadcast, `scale_ > 0` is not (`dsc/dsc2.cpp:4058`).
        let broadcast = LabeledDs::new(
            DsType::Input,
            vec![
                (PrimaryDim::I, Scale::Sized(0.0)),
                (PrimaryDim::J, Scale::Sized(0.0)),
            ],
            LdsIdx(0),
            Pinning::default(),
        );
        let mixed = LabeledDs::new(
            DsType::Input,
            vec![
                (PrimaryDim::I, Scale::Sized(1.0)),
                (PrimaryDim::J, Scale::Sized(0.0)),
            ],
            LdsIdx(1),
            Pinning::default(),
        );
        DesignSpaceConfig {
            ddc: crate::schedule::l3::dsc::DdcFacts::default(),
            gtr_ids_used: BTreeSet::new(),
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: Some(CoreletsUsed::ONE),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(
                crate::units::Core::checked(0).expect("core 0"),
                vec![],
            ),
            layout_dims: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(broadcast, vec![mixed]),
            data_stages: DataStages::new(stage("core"), stage("chunk")),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        }
    }

    /// ⭐⭐ THE TWO REFERENCE FUNCTIONS ARE NOT INTERCHANGEABLE, AND THIS CARRIES THE DIM RATHER THAN
    /// A COUNT. `getNonBroadcastLdsDimSet(1)` (`dsc/dsc2.cpp:4050-4065`) reads the labelled DS's OWN
    /// `layoutDimOrder_` filtered to `scale_ > 0` and NEVER calls `getLayoutDims`, so its answer is
    /// `{I}`. `getNonBroadcastLdsDims(1)` (`:4039-4048`) intersects that with `getLayoutDims(1)`,
    /// which on this DSC hits `DT_CHECK(allocNode)` — so the WRONG choice presents a structure that
    /// does carry a non-broadcast dim as one that carries none.
    ///
    /// ⛔ THIS IS THE ASSERTION THAT WOULD HAVE CAUGHT IT. [`non_broadcast_dims_of`] called the
    /// intersecting flavour while both of its callers cite the set
    /// (`schedule/ddc/v1.rs:3370`, `schedule/ddc/transformation.rs:958`); every test stayed green,
    /// because nothing named the dim that went missing.
    #[test]
    fn the_non_broadcast_set_is_the_lds_own_order_and_not_the_allocate_nodes() {
        let dsc = dsc_without_layouts();

        // The reference's own two answers for the SAME index, side by side.
        assert_eq!(
            dsc.non_broadcast_lds_dim_set(LdsIdx(1)),
            Some(BTreeSet::from([PrimaryDim::I])),
            "getNonBroadcastLdsDimSet reads the lds' own order and needs no allocate node"
        );
        assert_eq!(
            dsc.non_broadcast_lds_dims(LdsIdx(1)),
            None,
            "getNonBroadcastLdsDims stops at getLayoutDims' DT_CHECK(allocNode)"
        );

        // ⭐ THE CARRIER ANSWERS THE SET — the dim itself, not its absence.
        assert_eq!(
            non_broadcast_dims_of(&dsc, LdsIdx(1)),
            BTreeSet::from([PrimaryDim::I])
        );

        // ⭐ AND A WHOLLY BROADCAST STRUCTURE IS STILL GENUINELY EMPTY, so the fix widened the
        // answer only where a dim was being dropped: every `scale_` is 0 here.
        assert_eq!(non_broadcast_dims_of(&dsc, LdsIdx(0)), BTreeSet::new());
    }

    /// ⚠️ THE ONE EMPTY A CALLER CANNOT TELL APART, asserted so the gap is measured rather than
    /// believed: position 2 is past `labeledDs_`, which is `labeledDs_.at(ldsIdx)`'s THROW — and it
    /// reads back identical to position 0's genuine `{}` above, because both trait returns are a bare
    /// `BTreeSet`.
    #[test]
    fn a_position_the_dsc_does_not_hold_reads_back_as_a_genuine_empty() {
        let dsc = dsc_without_layouts();
        assert_eq!(dsc.non_broadcast_lds_dim_set(LdsIdx(2)), None);
        assert_eq!(non_broadcast_dims_of(&dsc, LdsIdx(2)), BTreeSet::new());
    }
}

