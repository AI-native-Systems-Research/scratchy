// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ WHERE ONE DSC'S CARRIERS COME FROM — [`v1::Dsc2Sites`], the `P` [`v1::run_v1`] is handed, plus
//! the two carriers that are types of their own: `currDsc->dataStageParam_` and the DDL match site.
//!
//! # ⭐ `Dsc2Stages` — THE SHARED DATASTAGE MAP, AND THE ONE METHOD THAT CANNOT SHARE IT
//!
//! Twenty-six of [`v1::ExploreStages`]'s twenty-eight methods read or write the map in
//! [`Dsc2State`] — the SAME map [`super::Dsc2Reads`]'s [`v1::StageSizes`] reads, which is what makes
//! a datastage entry 338 mints the one entry 128 lays out.
//!
//! ⛔⛔ [`v1::Dsc2Stages::as_util`] IS THE EXCEPTION AND IT IS A `todo!`. It returns
//! `&mut UtilDataStages<Self::Dims>` — a REAL exclusive borrow, which no [`RefCell`] can hand out —
//! and its own doc demands it be *"THE MAP THIS TYPE'S OWN `ExploreStages` READS"*, because a freshly
//! built map would drop every effect entries 302/303/338 have and still compile. Those two
//! requirements are incompatible for a carrier whose map must ALSO be visible to the `dsc` half that
//! `Dsc2Carriers` borrows disjointly from it. So the method NAMES the conflict rather than resolving
//! it with a copy.
//!
//! ⭐ ITS THREE CALLERS ALL SIT AFTER THE DDL STEP (`ddc/v1.rs:6503`, `:6549`, `:6580`) and are all
//! `let _ = ..`, so this is not what stops the stage.
//!
//! # ⛔ `Dsc2Ddl` — TWENTY-FOUR METHODS, AND `l3::dsc` CARRIES SIX OF THEM
//!
//! The DDL match reads `dataFormat_`, `wordLength` and `dsName_` on every operand and WRITES the
//! first two (`MatchSite::set_lds_word_length`, `set_lds_format`) — the three fields
//! [`crate::schedule::l3::dsc::LabeledDs`] does not project. What it can answer is the work-slice
//! ring, the layout-order test and the stick dims.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::dsc_lowering::DataLocation;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickDims,
};
use crate::formats::DataFormat;
use crate::schedule::ddc::fold::{AllocId, ConstIdx, NodeId, PadType};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation as tr;
use crate::schedule::ddc::transformation_util as tu;
use crate::schedule::ddc::v1;
use crate::schedule::ddl::conversion as conv;
use crate::schedule::dsc2::{
    BlockNode, ComputeNode, CondRegions, ConditionNode, LdsIdx, LoopNode, NodeName, SyncNode,
    SyncUnits, TransferNode, WordLength,
};
use crate::schedule::l3::dl_ops::AddressFoldCoords;
use crate::schedule::l3::dsc::{DscIdx, Symbolic, SymbolicDimInfo, WkSlice};
use crate::units::{Core, Corelet};

use super::ddc_state::{self, Dsc2Dims, Dsc2State};
use super::ddc_store::Dsc2Store;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// `currDsc->dataStageParam_` THROUGH ITS THREE VOCABULARIES.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ ONE DSC'S `dataStageParam_` — the `S` carrier, over the SHARED map in [`Dsc2State`].
#[derive(Debug)]
pub struct Dsc2Stages<'s, 'l> {
    state: &'s Dsc2State<'l>,
    dsc: DscIdx,
}

/// ⭐⭐ WHICH ROW OF `addressGranularityScalePerUnit` A `{unit, storage}` PAIR NAMES —
/// `sys-arch-spec/sysdef.cpp:531-554`, all twenty-three, as the [`DataLocation`] that already holds
/// their granularities.
///
/// ⛔⛔ THE POINT IS THAT THE KEY SET AND THE VALUES STAY ONE FACT. [`DataLocation`]'s own doc says
/// *"the table's keys are the whole domain, and that is why this is an enum"* — so the honest answer
/// to `count({unit, storage})` is *which variant that pair is*, not a second listing of the pairs
/// beside the granularities. ⚠️ REPORTED: this belongs as `DataLocation::of(unit, storage)` in
/// `bridges/superdsc_to_dataflow_ir/dsc_lowering.rs`, where the granularities live and where
/// `v1::OffsetSizes::address_scale` (`stages/offsets.rs:171`, `stages/ddc_reads.rs:580`) would read
/// it too; it is written here because that file is not this agent's to edit.
///
/// ⛔ `LRFREG` IS `SenComponent::Lrfreg` AND NOT `PeLrfreg`/`SfpLrfreg`/`PtLrfreg`. Those three are
/// separate components (`sys-arch-spec/arch_enums.rs`, 102-104) and the table names none of them:
/// `{SFP, LRFREG}` and `{SFP, SFPLRF}` are the two keys assigned in one statement (`:546-547`).
///
/// ⛔ AND THE PAIRS ARE THE **GENERIC** COMPONENTS: the reference keys on
/// `senCompToGenericComp.at(unit)`, so `LXLU0`/`LXLU1` arrive as `LXLU` and every PT row as `PT`.
fn address_location(unit: SenComponent, storage: SenComponent) -> Option<DataLocation> {
    Some(match (unit, storage) {
        (SenComponent::L3lu, SenComponent::Hbm) => DataLocation::L3luHbm,
        (SenComponent::L3lu, SenComponent::Lx) => DataLocation::L3luLx,
        (SenComponent::L3lu, SenComponent::L3luibr) => DataLocation::L3luIbr,
        (SenComponent::L3su, SenComponent::Hbm) => DataLocation::L3suHbm,
        (SenComponent::L3su, SenComponent::Lx) => DataLocation::L3suLx,
        (SenComponent::L3su, SenComponent::L3suibr) => DataLocation::L3suIbr,
        (SenComponent::Lxlu, SenComponent::Lx) => DataLocation::LxluLx,
        (SenComponent::Lxlu, SenComponent::Lxluscalereg) => DataLocation::LxluScaleReg,
        (SenComponent::Lxlu, SenComponent::Lxluvalue) => DataLocation::LxluValue,
        (SenComponent::Lxsu, SenComponent::Lx) => DataLocation::LxsuLx,
        (SenComponent::L0lu, SenComponent::L0) => DataLocation::L0luL0,
        (SenComponent::L0su, SenComponent::L0) => DataLocation::L0suL0,
        (SenComponent::L0lu, SenComponent::L0Scale) => DataLocation::L0luL0Scale,
        (SenComponent::L0su, SenComponent::L0Scale) => DataLocation::L0suL0Scale,
        (SenComponent::Sfp, SenComponent::Lrfreg) => DataLocation::SfpLrfReg,
        (SenComponent::Sfp, SenComponent::Sfplrf) => DataLocation::SfpLrf,
        (SenComponent::Sfp, SenComponent::Sfpstate) => DataLocation::SfpState,
        (SenComponent::Pe, SenComponent::Lrfreg) => DataLocation::PeLrfReg,
        (SenComponent::Pe, SenComponent::Pelrf) => DataLocation::PeLrf,
        (SenComponent::Pe, SenComponent::Pestate) => DataLocation::PeState,
        (SenComponent::Pt, SenComponent::Lrfreg) => DataLocation::PtLrfReg,
        (SenComponent::Pt, SenComponent::Ptarf) => DataLocation::PtArf,
        (SenComponent::Pt, SenComponent::Ptxrf) => DataLocation::PtXrf,
        _ => return None,
    })
}

/// WHAT `makeDimNotSymbolic` WILL WRITE ONTO ONE HALF — computed WHOLE before the first write, so
/// that its one stop cannot leave a half with the symbol erased and the shares still in `maxSize_`
/// units.
///
/// ⛔ EACH SPLIT IS AN [`Option`] BECAUSE ABSENT AND EMPTY ARE DIFFERENT: `coreletSplit_.find(dim) ==
/// end()` skips the divide entirely (`dsc/dims.cpp:793`), while a stated split with no shares is a
/// map the reference walks and leaves alone.
struct NotSymbolic {
    /// `symbolicDimInfo_` with the dim erased, `maxSymbolicVolume_` untouched.
    symbolic: Symbolic,
    /// `primaryDimToValHandler_st(dim) = divideByFactor(primaryDimToVal_st(dim))`.
    value: Extent,
    /// `coreletSplit_.at(dim)`, each share divided.
    corelet: Option<Vec<Extent>>,
    /// `rowSplit_.at(dim)`, likewise.
    rows: Option<BTreeMap<Corelet, Vec<Extent>>>,
    /// `peSfpSplit_.at(dim)`, likewise, both halves.
    pe_sfp: Option<BTreeMap<Corelet, v1::PeSfpShares>>,
}

/// `ceil(a / double(b))` — the FLOATING-POINT ceiling the reference reaches through
/// `float(numPTRows)` and `double totalWork`, over integers.
///
/// ⛔ NOT `i64::div_ceil`, WHICH PANICS ON A NEGATIVE OPERAND AND WOULD ROUND IT THE OTHER WAY. An
/// unstated dim is `-1` (`dsc/dims.h:161-193`) and the reference happily halves it: `ceil(-0.5)` is
/// `0`, which `(-1 + 1).div_euclid(2)` answers and `(-1i64).div_ceil(2)` does not.
const fn div_ceil_i64(value: i64, by: i64) -> i64 {
    (value + by - 1).div_euclid(by)
}

/// `floor(a / double(b))` — the other half of the PE/SFP split, and `-1` for `floor(-0.5)`.
const fn div_floor_i64(value: i64, by: i64) -> i64 {
    value.div_euclid(by)
}

/// ONE DIM'S PADDING AS A `PaddingFormType` — the argument `primaryDimToVal_st` takes, built from
/// the one dim and one [`PadType`] every caller of [`v1::ExploreStages::extent`] states.
///
/// ⛔ [`PadType::NoPad`] IS STILL WRITTEN RATHER THAN LEFT ABSENT, and the two agree: `getPadding`
/// answers `NOPAD` for a dim the form does not name (`dsc/dims.cpp:806`), which is what
/// [`tu::PaddingForm::padding`] does.
fn padding_form(dim: PrimaryDim, padding: PadType) -> tu::PaddingForm {
    let mut form = tu::PaddingForm::default();
    form.set_padding(dim, padding);
    form
}

impl<'s, 'l> Dsc2Stages<'s, 'l> {
    /// The map of that DSC.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>, dsc: DscIdx) -> Self {
        Self { state, dsc }
    }

    fn facts(&self) -> &'s super::ddc_state::Dsc2Facts {
        self.state
            .facts(self.dsc)
            .expect("a Dsc2Stages is only built for a DSC the state holds facts for")
    }

    /// One half, cloned for the length of one read.
    fn half(&self, at: v1::StageSite) -> Option<Dsc2Dims> {
        self.facts().with_stages(|stages| {
            let held = stages.0.get(&at.stage)?;
            Some(match at.half {
                v1::StageHalf::Ss => held.ss.dims.clone(),
                v1::StageHalf::El => held.el.dims.clone(),
            })
        })
    }

    /// One half, for the length of one write — a NO-OP for a datastage the map does not hold, which
    /// is the reference's `.at()` throw.
    fn edit(&self, at: v1::StageSite, write: impl FnOnce(&mut Dsc2Dims)) {
        self.facts().with_stages_mut(|stages| {
            if let Some(held) = stages.0.get_mut(&at.stage) {
                match at.half {
                    v1::StageHalf::Ss => write(&mut held.ss.dims),
                    v1::StageHalf::El => write(&mut held.el.dims),
                }
            }
        });
    }
}

impl v1::ExploreStages for Dsc2Stages<'_, '_> {
    type Dims = Dsc2Dims;

    /// `dataStageParam_`'s keys, in `std::map` order.
    fn stages(&self) -> Vec<DatastageId> {
        self.facts()
            .with_stages(|stages| stages.0.keys().copied().collect())
    }

    /// `dataStageParam_.at(stage).name()` as the closed three-way [`v1::StageLabel`] is.
    fn label(&self, stage: DatastageId) -> Option<v1::StageLabel> {
        let held = self.half(v1::StageSite::ss(stage))?;
        Some(match held.name.0.as_str() {
            "core" => v1::StageLabel::Core,
            "chunk" => v1::StageLabel::Chunk,
            _ => v1::StageLabel::Other,
        })
    }

    /// ⛔⛔ `isExternalDs` DOES NOT EXIST IN THE AUTHORITY — grepped over the whole of
    /// `/Users/nickm/git/deeptools-src` and the identifier appears ZERO times. So the sentence this
    /// stub used to carry (*"a `DesignSpaceConfig` predicate comparing the stage's sizes against the
    /// whole tensor's, and it DECIDES whether finalize_external_stage runs"*) named a function that is
    /// not there, and its second half is refuted twice over: [`Self::finalize_external_stage`] is
    /// called on EVERY stage of `dataStageParam_` (`ddc/ddcv1.cpp:2085-2088`), and its own body is
    /// three independently guarded blocks with no external test in any of them.
    ///
    /// ⛔ WHAT THE REFERENCE ACTUALLY MEANS BY "EXTERNAL" IS `!metadata.datastages_.count(id)` — a
    /// stage the DDL walk did NOT mint (`ddc/ddcv1.cpp:934-937`, and the two `continue // external`
    /// lines `run_v1` already carries at `ddc/v1.rs:5084`, `:5087`). That is a [`Metadata`] fact, and
    /// [`Metadata`] is a borrow DISJOINT from this carrier: the caller holds it, which is why both of
    /// those `continue`s are written at the CALL SITE and not asked of the stages.
    ///
    /// ⭐ AND NOTHING CALLS THIS. `v1::ExploreStages::is_external` has no caller anywhere in the crate
    /// — only the trait declaration (`ddc/v1.rs:3880`) and a test double (`:8566`) — so this `todo!`
    /// is unreachable rather than blocking. It is left NAMED rather than answered `false`, because a
    /// carrier answering a predicate it cannot see is how a fabricated fact gets in.
    fn is_external(&self, _stage: DatastageId) -> bool {
        todo!(
            "v1::ExploreStages::is_external: `isExternalDs` is NOT a symbol of the authority (zero \
             hits in deeptools-src); the reference's own test is !metadata.datastages_.count(id) \
             (ddc/ddcv1.cpp:934-937), a Metadata fact this carrier does not borrow. No caller exists \
             — see this method's doc"
        )
    }

    /// The snapshot — ⭐ AN OWNED CLONE, which is what the trait's own doc asks for.
    fn dims(&self, at: v1::StageSite) -> Option<Self::Dims> {
        self.half(at)
    }

    /// ⭐⭐ `primaryDimToVal_st(dim, comp, row, cl, padded, density, symbolic)` — THE PORTED FOLD,
    /// [`crate::schedule::l3::dsc::StageDims::sampled_extent`] (entry 012, `dsc/dims.cpp:653-704`),
    /// which is the SAME map this carrier holds: `rowSplit_` first, then `peSfpSplit_`, then the
    /// corelet view (`:631-645`) and `calculate_padded` (`:563-616`).
    ///
    /// ⛔⛔ THIS USED TO REACH ONLY [`Dsc2Dims::raw_slot`] AND `todo!` ON EVERY SAMPLED AXIS, while
    /// the fold sat ported one module away — the same composition
    /// [`Dsc2Dims`]'s own `Stage::extent` (`stages/ddc_state.rs`) already makes. So every read at a
    /// corelet, a PT row or a PE/SFP half stopped the stage, and `finalize_external_stage` below
    /// could not compute a share at all.
    ///
    /// ⛔ `dimDensity` IS THE REFERENCE'S OWN DEFAULT `1.0` and not a dropped argument: the trait
    /// states no density, so [`None`] is what every caller of this method asks for
    /// (`dsc/dims.h:269-273`).
    ///
    /// ⛔ AND [`Dsc2Dims::raw_slot`] STAYS THE SECOND ARM, because the fold's [`None`] cannot tell
    /// the reference's own `-1` from the reference ABORTING: an unstated slot is `-1` on 187 of 187
    /// g0 programs, and only the abort is a stop.
    fn extent(
        &self,
        at: v1::StageSite,
        dim: PrimaryDim,
        sample: v1::DimSample,
        padding: PadType,
        symbolic: v1::SymbolicRead,
    ) -> Extent {
        if let Some(half) = self.half(at) {
            if let Some(extent) = half.dims.sampled_extent(
                dim,
                sample,
                &padding_form(dim, padding),
                None,
                symbolic == v1::SymbolicRead::Granularity,
            ) {
                return extent;
            }
            if let Some(extent) = half.raw_slot(dim, padding, symbolic) {
                return extent;
            }
        }
        todo!(
            "v1::ExploreStages::extent: primaryDimToVal_st (dsc/dims.cpp:653-704) STOPS on stage \
             {at:?} dim {dim:?} at {sample:?} — the split names neither that corelet nor that row, \
             or calculate_padded (:563-616) aborted on {padding:?}. Substituting the whole core's \
             extent there would be a fabricated extent."
        )
    }

    /// `primaryDimToValHandler_st(dim)` READ — ⭐ THE RAW PER-DIM SLOT, before any split is folded in,
    /// which is exactly what [`crate::schedule::l3::dsc::StageDims::extents`] holds.
    ///
    /// ⭐ THE UNSTATED SLOT IS THE REFERENCE'S `-1`, entry 207's recorded divergence.
    fn dim_value(&self, at: v1::StageSite, dim: PrimaryDim) -> Extent {
        self.half(at)
            .and_then(|half| half.dims.extents.get(&dim).copied())
            .unwrap_or(crate::schedule::l3::dl_ops::UNSTATED_EXTENT)
    }

    /// `primaryDimToValHandler_st(dim) = value`.
    fn set_dim_value(&mut self, at: v1::StageSite, dim: PrimaryDim, value: Extent) {
        self.edit(at, |half| {
            half.dims.extents.insert(dim, value);
        });
    }

    /// `coreletSplit_.at(dim)`.
    fn corelet_shares(&self, at: v1::StageSite, dim: PrimaryDim) -> Option<Vec<Extent>> {
        self.half(at)?.dims.corelet_split.get(&dim).cloned()
    }

    /// `coreletSplit_[dim] = shares`.
    fn set_corelet_shares(&mut self, at: v1::StageSite, dim: PrimaryDim, shares: Vec<Extent>) {
        self.edit(at, |half| {
            half.dims.corelet_split.insert(dim, shares);
        });
    }

    /// `rowSplit_.at(dim)`.
    fn row_shares(
        &self,
        at: v1::StageSite,
        dim: PrimaryDim,
    ) -> Option<BTreeMap<Corelet, Vec<Extent>>> {
        self.half(at)?.dims.row_split.get(&dim).cloned()
    }

    /// `rowSplit_[dim] = shares`.
    fn set_row_shares(
        &mut self,
        at: v1::StageSite,
        dim: PrimaryDim,
        shares: BTreeMap<Corelet, Vec<Extent>>,
    ) {
        self.edit(at, |half| {
            half.dims.row_split.insert(dim, shares);
        });
    }

    /// `peSfpSplit_.at(dim)`.
    fn pe_sfp_shares(
        &self,
        at: v1::StageSite,
        dim: PrimaryDim,
    ) -> Option<BTreeMap<Corelet, v1::PeSfpShares>> {
        self.half(at)?.dims.pe_sfp_split.get(&dim).cloned()
    }

    /// `peSfpSplit_[dim] = shares`.
    fn set_pe_sfp_shares(
        &mut self,
        at: v1::StageSite,
        dim: PrimaryDim,
        shares: BTreeMap<Corelet, v1::PeSfpShares>,
    ) {
        self.edit(at, |half| {
            half.dims.pe_sfp_split.insert(dim, shares);
        });
    }

    /// `paddingSizes_` — every padded dim with its sizes, in `std::map` order.
    fn padding_dims(
        &self,
        at: v1::StageSite,
    ) -> Vec<(PrimaryDim, crate::schedule::l3::dsc::DimPadding)> {
        self.half(at)
            .map(|half| half.dims.padding.into_iter().collect())
            .unwrap_or_default()
    }

    /// `paddingSizes_[dim] = padding` — the final value, as the trait's own doc says.
    fn set_padding(
        &mut self,
        at: v1::StageSite,
        dim: PrimaryDim,
        padding: crate::schedule::l3::dsc::DimPadding,
    ) {
        self.edit(at, |half| {
            half.dims.padding.insert(dim, padding);
        });
    }

    /// `symbolicDimInfo_` — every symbolic dim, in `std::map` order.
    fn symbolic_dims(&self, at: v1::StageSite) -> Vec<PrimaryDim> {
        self.half(at)
            .map(|half| half.dims.symbolic.info().keys().copied().collect())
            .unwrap_or_default()
    }

    /// `symbolicDimInfo_.at(dim)`.
    fn symbolic_info(&self, at: v1::StageSite, dim: PrimaryDim) -> Option<SymbolicDimInfo> {
        self.half(at)?.dims.symbolic.info().get(&dim).copied()
    }

    /// `symbolicDimInfo_.insert_or_assign(dim, info)`.
    fn set_symbolic_info(&mut self, at: v1::StageSite, dim: PrimaryDim, info: SymbolicDimInfo) {
        self.edit(at, |half| half.dims.symbolic.add_dim(dim, info));
    }

    /// ⭐⭐ `makeDimSymbolic(refDs, dim)` (`dsc/dims.cpp:764-780`) WHOLE — NOT an insert: after the
    /// `emplace` it OVERWRITES `primaryDimToValHandler_st(dim)` with the REFERENCE stage's value and
    /// re-copies whichever of `coreletSplit_`/`rowSplit_`/`peSfpSplit_` already named the dim. An
    /// insert alone leaves this stage's own extent in place beside a symbol that does not describe it.
    ///
    /// ⛔⛔ `emplace(*symIt).second` IS A NO-OP RETURN, NOT AN OVERWRITE (`:768`): a dim this stage
    /// already calls symbolic keeps its OWN info and its own extent, and every statement below is
    /// skipped. [`crate::schedule::l3::dsc::Symbolic::add_dim`] is an `insert`, so the
    /// already-symbolic test has to be made here.
    ///
    /// ⛔ THE VALUE COPIED IS `refDs.primaryDimToVal_st(dim)` — read on the REFERENCE half and read
    /// with the symbol in place, so it is that stage's `maxSize_` and not its stored slot
    /// (`dsc/dims.cpp:522-527`). Recomputing it from this stage would be a fabricated extent.
    ///
    /// ⛔ `DT_CHECK(refDs.symbolicDimInfo_.count(dim))` (`:766`) IS NOT DISCHARGED BY THE CALL SITE:
    /// `calculate_epilogues` guards on `symbolic_dims(den_ss)` (`ddc/v1.rs:5095`) — the stage's own
    /// STEADY-STATE half — while the check is on `core`, the reference half. So it stays a stop, and
    /// so do the three `.at(dim)` throws (`:771`, `:774`, `:777`), each of which is a split THIS
    /// stage names and the reference does not.
    ///
    /// ⛔ EVERY ABORT IS DECIDED **BEFORE** THE FIRST WRITE — one stop for the method, and a half that
    /// is either wholly rewritten or wholly untouched. A `todo!` reached mid-edit would leave the
    /// symbol added and the splits stale.
    fn make_dim_symbolic(&mut self, at: v1::StageSite, from: v1::StageSite, dim: PrimaryDim) {
        // ⛔ THE ALREADY-SYMBOLIC TEST FIRST, WHICH IS THE `emplace(..).second` RETURN — and it comes
        // before every abort because the reference returns there without reading `refDs` again.
        if self
            .half(at)
            .is_some_and(|half| half.dims.symbolic.info().contains_key(&dim))
        {
            return;
        }
        // `refDs` READ WHOLE — the reference half is a different datastage in every caller, and all
        // of these are reads of its pre-existing state.
        let planned = self.half(from).and_then(|reference| {
            let info = reference.dims.symbolic.info().get(&dim).copied()?;
            // `refDs.primaryDimToVal_st(dim)`, with the symbol in place: `maxSize_`.
            let value = reference.dims.sampled_extent(
                dim,
                v1::DimSample::WHOLE,
                &tu::PaddingForm::default(),
                None,
                false,
            )?;
            // ⛔ EACH SPLIT IS `refDs.<map>.at(dim)` AND ONLY WHERE **THIS** HALF NAMES THE DIM, so a
            // reference that does not name it is the `.at()` throw and not an absent copy.
            let mine = self.half(at)?;
            let corelet = if mine.dims.corelet_split.contains_key(&dim) {
                Some(reference.dims.corelet_split.get(&dim).cloned()?)
            } else {
                None
            };
            let rows = if mine.dims.row_split.contains_key(&dim) {
                Some(reference.dims.row_split.get(&dim).cloned()?)
            } else {
                None
            };
            let pe_sfp = if mine.dims.pe_sfp_split.contains_key(&dim) {
                Some(reference.dims.pe_sfp_split.get(&dim).cloned()?)
            } else {
                None
            };
            Some((info, value, corelet, rows, pe_sfp))
        });
        let Some((info, value, corelet, rows, pe_sfp)) = planned else {
            todo!(
                "v1::ExploreStages::make_dim_symbolic: makeDimSymbolic (dsc/dims.cpp:764-780) \
                 ABORTED writing {dim:?} onto {at:?} from {from:?} — one of \
                 dataStageParam_.at({from:?}), DT_CHECK(refDs.symbolicDimInfo_.count({dim:?})) \
                 (:766), refDs.primaryDimToVal_st({dim:?}) (:769), or a \
                 refDs.coreletSplit_/rowSplit_/peSfpSplit_.at({dim:?}) throw (:771-778) for a split \
                 {at:?} states and {from:?} does not. Every one of those carries a VALUE the \
                 reference owns, and none of them has a substitute"
            )
        };
        self.edit(at, |half| {
            half.dims.symbolic.add_dim(dim, info);
            half.dims.extents.insert(dim, value);
            if let Some(shares) = &corelet {
                half.dims.corelet_split.insert(dim, shares.clone());
            }
            if let Some(shares) = &rows {
                half.dims.row_split.insert(dim, shares.clone());
            }
            if let Some(shares) = &pe_sfp {
                half.dims.pe_sfp_split.insert(dim, shares.clone());
            }
        });
    }

    /// ⭐⭐ `makeDimNotSymbolic(dim)` (`dsc/dims.cpp:781-803`) — the erase AND the DIVIDE-BY-FACTOR
    /// that follows it, over the dim value and every corelet, row and PE/SFP share.
    ///
    /// ⛔⛔ THE DIVIDE IS THE WHOLE POINT AND IT IS NOT `scaleFromMaxToGranularity`. That helper is a
    /// no-op for a dim `symbolicDimInfo_` does not name (`:619-621`), and the erase happens FIRST —
    /// so the factor has to be taken from the info before it goes, and the value read back AFTER, off
    /// the now-plain slot. Dropping the divide leaves every share expressed in `maxSize_` units while
    /// the dim is no longer symbolic, which multiplies every one of them by `maxSize_/granularity_`.
    ///
    /// ⛔ `symIt == end` IS A NO-OP RETURN (`:783`) and is what the size search's own guard already
    /// tests (`ddc/v1.rs:5325`), so it is answered here rather than stopped.
    ///
    /// ⛔ WHAT STOPS: the two `DT_CHECK`s on the factor (`:784`, `:786`) and every
    /// `DT_CHECK(val % factor == 0)` inside `divideByFactor` (`:789`) — a share the factor does not
    /// divide, which is an extent this cannot round.
    ///
    /// ⛔ AND A VOLUME LIMIT KEYED ON THE DIM STOPS TOO, because the reference's plain
    /// `symbolicDimInfo_.erase` leaves `maxSymbolicVolume_` untouched while
    /// [`crate::schedule::l3::dsc::Symbolic`] holds *"every dim a volume limit is keyed on is named
    /// by `info`"* as a type invariant — a state the reference reaches and this type cannot spell.
    /// `Symbolic` needs its own `remove_dim` to decide that; rebuilding through `Symbolic::new` here
    /// would silently DROP the limit.
    /// ⛔ AND EVERY ABORT IS DECIDED BEFORE THE FIRST WRITE, as in [`Self::make_dim_symbolic`]: one
    /// stop, and a half either wholly rewritten or wholly untouched.
    fn make_dim_not_symbolic(&mut self, at: v1::StageSite, dim: PrimaryDim) {
        let Some(half) = self.half(at) else {
            return;
        };
        // `if (symIt == symbolicDimInfo_.end()) return;` — already not symbolic.
        let Some(info) = half.dims.symbolic.info().get(&dim).copied() else {
            return;
        };
        let granularity = i64::from(info.granularity.get());
        let max = i64::from(info.max_size.0);
        let planned = (|| -> Option<NotSymbolic> {
            // `DT_CHECK(maxSize_ % granularity_ == 0)` (`:784`) then `DT_CHECK(factor != 0)` (`:786`).
            if granularity == 0 || max % granularity != 0 {
                return None;
            }
            let factor = max / granularity;
            if factor == 0 {
                return None;
            }
            // ⛔ A VOLUME LIMIT KEYED ON THE DIM — see this method's own note.
            if half
                .dims
                .symbolic
                .volumes()
                .keys()
                .any(|dims| dims.contains(&dim))
            {
                return None;
            }
            // `divideByFactor`, whose `DT_CHECK(val % factor == 0)` is the [`None`].
            let divided = |val: Extent| -> Option<Extent> {
                (val.0 % factor == 0).then(|| Extent(val.0 / factor))
            };
            // The erase, exactly — every volume limit survives because none names this dim.
            let mut info_after = half.dims.symbolic.info().clone();
            info_after.remove(&dim);
            let symbolic = Symbolic::new(info_after, half.dims.symbolic.volumes().clone());
            // `primaryDimToValHandler_st(dim) = divideByFactor(primaryDimToVal_st(dim))`, read AFTER
            // the erase — so off the plain slot and not off `maxSize_`.
            let mut plain = half.dims.clone();
            plain.symbolic = symbolic.clone();
            let value = divided(plain.sampled_extent(
                dim,
                v1::DimSample::WHOLE,
                &tu::PaddingForm::default(),
                None,
                false,
            )?)?;
            let corelet = match half.dims.corelet_split.get(&dim) {
                Some(shares) => Some(
                    shares
                        .iter()
                        .map(|share| divided(*share))
                        .collect::<Option<Vec<Extent>>>()?,
                ),
                None => None,
            };
            let rows = match half.dims.row_split.get(&dim) {
                Some(per_corelet) => Some(
                    per_corelet
                        .iter()
                        .map(|(corelet, rows)| {
                            Some((
                                *corelet,
                                rows.iter()
                                    .map(|share| divided(*share))
                                    .collect::<Option<Vec<Extent>>>()?,
                            ))
                        })
                        .collect::<Option<BTreeMap<Corelet, Vec<Extent>>>>()?,
                ),
                None => None,
            };
            let pe_sfp = match half.dims.pe_sfp_split.get(&dim) {
                Some(per_corelet) => Some(
                    per_corelet
                        .iter()
                        .map(|(corelet, shares)| {
                            Some((
                                *corelet,
                                v1::PeSfpShares {
                                    pe: divided(shares.pe)?,
                                    sfp: divided(shares.sfp)?,
                                },
                            ))
                        })
                        .collect::<Option<BTreeMap<Corelet, v1::PeSfpShares>>>()?,
                ),
                None => None,
            };
            Some(NotSymbolic {
                symbolic,
                value,
                corelet,
                rows,
                pe_sfp,
            })
        })();
        let Some(NotSymbolic {
            symbolic,
            value,
            corelet,
            rows,
            pe_sfp,
        }) = planned
        else {
            todo!(
                "v1::ExploreStages::make_dim_not_symbolic: makeDimNotSymbolic (dsc/dims.cpp:781-803) \
                 ABORTED on {at:?} {dim:?} (maxSize_={max}, granularity_={granularity}) — one of \
                 DT_CHECK(maxSize_ % granularity_ == 0) (:784), DT_CHECK(factor != 0) (:786), a \
                 DT_CHECK(val % factor == 0) inside divideByFactor (:789) on the slot or on a \
                 corelet/row/PE-SFP share, or a maxSymbolicVolume_ keyed on {dim:?} — which \
                 symbolicDimInfo_.erase (:788) leaves in place and l3::dsc::Symbolic holds as a type \
                 invariant it has no remove_dim to break. A rounded share is a fabricated extent"
            )
        };
        self.edit(at, |half| {
            half.dims.symbolic = symbolic.clone();
            half.dims.extents.insert(dim, value);
            if let Some(shares) = &corelet {
                half.dims.corelet_split.insert(dim, shares.clone());
            }
            if let Some(shares) = &rows {
                half.dims.row_split.insert(dim, shares.clone());
            }
            if let Some(shares) = &pe_sfp {
                half.dims.pe_sfp_split.insert(dim, shares.clone());
            }
        });
    }

    /// `pruneMaxSymbolicVolumes(refDstg)` (`dsc/dims.cpp:729-762`) — ⭐ ALREADY PORTED, as
    /// [`crate::schedule::l3::dsc::Symbolic::prune_volumes`], the UNFUSED shape `ddc/ddcv1.cpp:1424`
    /// and `:1425` call: THIS stage keeps its own `maxSymbolicVolume_`, re-keyed onto the dims it still
    /// calls symbolic, and does not adopt the core's.
    fn prune_max_symbolic_volumes(&mut self, at: v1::StageSite, from: v1::StageSite) {
        let Some(reference) = self.half(from) else {
            return;
        };
        self.edit(at, |half| {
            half.dims.symbolic.prune_volumes(&reference.dims.symbolic);
        });
    }

    /// `compound()` — ⭐ ALREADY PORTED, as
    /// [`crate::schedule::l3::dsc::StageDims::compound`].
    fn compound(&mut self, at: v1::StageSite) {
        self.edit(at, |half| half.dims.compound());
    }

    /// `denDs.el_ = denDs.ss_` — the whole half copied over.
    fn copy_ss_to_el(&mut self, stage: DatastageId) {
        self.facts().with_stages_mut(|stages| {
            if let Some(held) = stages.0.get_mut(&stage) {
                held.el.dims = held.ss.dims.clone();
                held.el.name = held.ss.name.clone();
            }
        });
    }

    /// `el_.name_ += "el"`.
    fn mark_epilogue_name(&mut self, stage: DatastageId) {
        self.facts().with_stages_mut(|stages| {
            if let Some(held) = stages.0.get_mut(&stage) {
                held.el.name.0.push_str("el");
                held.el.dims.name = held.el.name.clone();
            }
        });
    }

    /// `ds.r_ = ds.c_ = … = -1` — ⚠️ A NO-OP IN THIS MODEL, AND STATED ANYWAY, exactly as the trait's
    /// own doc says: those nine fields are all outside the closed twelve [`PrimaryDim`], so nothing
    /// reachable from here can observe them.
    fn clear_deprecated_dims(&mut self, _at: v1::StageSite) {}

    /// ⭐⭐ `finalizeExternalDataStage(dsc, stage, clSplitDims, numPTRows, usePt, rowSplitDim,
    /// peSfpSplitDims)` — `ddc/ddcv1.cpp:1748-1797`, READ, AND ITS THREE GUARDS ANSWERED.
    ///
    /// ⛔⛔ IT IS **NOT** GATED ON `isExternalDs` — THE NAME MISLEADS. The whole body is three
    /// independently guarded blocks and nothing else:
    ///
    /// 1. `if (dsc.numCoreletsUsed_DSC2_ > 1)` (`:1755`) — the corelet split, `ceil(extent / 2)` into
    ///    both halves of `coreletSplit_`;
    /// 2. `if (doPTSplit)` (`:1767`) — the PT-row split, `ceil(extent / numPTRows)` per corelet into
    ///    `rowSplit_`;
    /// 3. `for (auto dim : peSfpSplitDims)` (`:1778`) — `ceil`/`floor` halves into `peSfpSplit_`.
    ///
    /// ⭐ SO WITH ALL THREE GUARDS FALSE THE FUNCTION IS A COMPLETE NO-OP, and that is the authority's
    /// own control flow rather than a guess: one corelet, `usePt == No`, and an empty PE/SFP split set
    /// leave no statement in the body reachable. `clSplitDims` is read ONLY inside guard 1, so it
    /// cannot matter when that guard is false.
    ///
    /// ⭐ ALL THREE ARE FACTS THIS CARRIER HOLDS: `numCoreletsUsed_DSC2_` is the state's own cell
    /// (`prep_dsc` sets it from `numCoreletsUsed_`, and scratchy emits `numCoreletsUsed_ = 1` on every
    /// one of its 313 sampled SuperDSCs — `crustify-ddc/EXCLUSIONS.tsv`), and the other two are
    /// ARGUMENTS.
    ///
    /// ⭐⭐ ALL THREE BLOCKS ARE NOW PORTED, because the extent each one halves is
    /// `primaryDimToVal_st` AT A SAMPLED CORELET and [`Self::extent`] above now folds it — that read
    /// is what used to make this a `todo!`, not the arithmetic.
    ///
    /// ⛔⛔ BLOCK 1'S `ceil` IS A NO-OP AND REPRODUCING THAT IS THE POINT.
    /// `ceil(ds.ss_.primaryDimToVal_st(dim) / 2)` (`:1758`) divides an `int` by the `int` `2`, so C++
    /// has already truncated toward zero before `ceil` ever sees the value — the corelet share is
    /// `extent / 2` ROUNDED DOWN. Rust's `/` on `i64` truncates the same way, so the expression is
    /// written the same way and NOT "corrected" to a `div_ceil`, which would hand each corelet one
    /// element more than the reference gives it on every odd extent.
    ///
    /// ⭐ BLOCKS 2 AND 3 ARE REAL ROUNDING, and for the opposite reason: `float(numPTRows)` (`:1771`)
    /// and the `double totalWork` (`:1783`) make those divisions floating point. So the PT row share
    /// is a true `ceil` and the PE/SFP halves are a true `ceil`/`floor` — including on a NEGATIVE
    /// extent, where `ceil(-0.5)` is `0` and `floor(-0.5)` is `-1`. [`div_ceil_i64`]/[`div_floor_i64`]
    /// are those two, exactly.
    ///
    /// ⛔ `{clWork, clWork}` IS TWO ENTRIES WHATEVER `numCoreletsUsed_DSC2_` SAYS (`:1759`) — the
    /// reference hard-codes the pair — while blocks 2 and 3 loop `0 .. numCoreletsUsed_DSC2_`. That
    /// asymmetry is the authority's and is carried.
    ///
    /// ⛔ `numPTRows` IS `Target::PT_ROWS`, the same identity the landed
    /// [`DataLocation::granularity`] already stands on for the `{L0SU, L0}` row: the arch is a cargo
    /// feature, so `dscGlobal.sysDef.numPTRows` is a literal here and not a threaded argument.
    ///
    /// ⛔ WHAT STILL STOPS: a `doPTSplit` with NO `rowSplitDim`, which the reference spells as the
    /// out-of-range `PrimaryDimTypesCount` default (`ddc/ddcv1.cpp:1753`) and writes
    /// `rowSplit_[PrimaryDimTypesCount]` under — a key outside the closed twelve [`PrimaryDim`] that
    /// this type cannot spell; a corelet index past the arch's own corelet count; and either half's
    /// extent read aborting.
    fn finalize_external_stage(
        &mut self,
        stage: DatastageId,
        cl_split: &BTreeSet<PrimaryDim>,
        use_pt: v1::UsePt,
        row_split: Option<PrimaryDim>,
        pe_sfp_split: &BTreeSet<PrimaryDim>,
    ) -> Option<()> {
        use crate::arch::{Arch, Target};

        // `numCoreletsUsed_DSC2_` — [`None`] is the `-1` a DSC is built with, which the reference
        // compares as `> 1` and so takes as "no corelet split" too.
        let corelets = self.facts().corelets_dsc2().unwrap_or(1);
        if corelets <= 1 && use_pt == v1::UsePt::No && pe_sfp_split.is_empty() {
            return Some(());
        }
        let ss = v1::StageSite::ss(stage);
        let el = v1::StageSite::el(stage);
        // `0 .. numCoreletsUsed_DSC2_`, as corelets, AND the row-split dim where a PT split is asked
        // for — the two facts every live block below needs, both decided before the first write.
        let planned: Option<(Vec<Corelet>, Option<PrimaryDim>)> = (0..corelets)
            .map(Corelet::checked)
            .collect::<Option<Vec<Corelet>>>()
            .and_then(|used| match use_pt {
                v1::UsePt::Yes => Some((used, Some(row_split?))),
                v1::UsePt::No => Some((used, None)),
            });
        let Some((used, pt_dim)) = planned else {
            todo!(
                "v1::ExploreStages::finalize_external_stage: finalizeExternalDataStage \
                 (ddc/ddcv1.cpp:1748-1797) cannot name what it must write on {stage:?} — either \
                 numCoreletsUsed_DSC2_={corelets} names a corelet this arch does not have \
                 (Target::CORELETS_PER_CORE), and every share below is written per corelet (:1768, \
                 :1780); or doPTSplit is set with no rowSplitDim, whose reference default is the \
                 out-of-range PrimaryDimTypesCount (:1753) that it writes rowSplit_ under and that \
                 the closed twelve PrimaryDim cannot spell"
            )
        };

        // ── 1. `if (dsc.numCoreletsUsed_DSC2_ > 1)` (`:1755`) ─────────────────────────────────────
        if corelets > 1 {
            for dim in cl_split {
                // ⛔ THE WHOLE-CORE EXTENT, TRUNCATED — see this method's own note on the `ceil`.
                let shares = if self
                    .half(ss)
                    .is_some_and(|half| half.dims.corelet_split.contains_key(dim))
                {
                    None
                } else {
                    let work = v1::ExploreStages::extent(
                        self,
                        ss,
                        *dim,
                        v1::DimSample::WHOLE,
                        PadType::NoPad,
                        v1::SymbolicRead::Max,
                    );
                    Some(vec![Extent(work.0 / 2), Extent(work.0 / 2)])
                };
                if let Some(shares) = shares {
                    self.edit(ss, |half| {
                        half.dims.corelet_split.insert(*dim, shares.clone());
                    });
                }
                // `ds.el_.coreletSplit_[dim] = ds.ss_.coreletSplit_[dim]` — the STEADY STATE'S final
                // value, whether this call wrote it or it was already there.
                let from_ss = self
                    .half(ss)
                    .and_then(|half| half.dims.corelet_split.get(dim).cloned())
                    .unwrap_or_default();
                if self
                    .half(el)
                    .is_some_and(|half| !half.dims.corelet_split.contains_key(dim))
                {
                    self.edit(el, |half| {
                        half.dims.corelet_split.insert(*dim, from_ss.clone());
                    });
                }
            }
        }

        // ── 2. `if (doPTSplit)` (`:1767`) ─────────────────────────────────────────────────────────
        if let Some(dim) = pt_dim {
            let rows = i64::from(Target::PT_ROWS);
            let mut per_corelet: BTreeMap<Corelet, Vec<Extent>> = BTreeMap::new();
            for corelet in &used {
                // `ceil(primaryDimToVal_st(dim, NO_COMPONENT, -1, cl) / float(numPTRows))`.
                let work = v1::ExploreStages::extent(
                    self,
                    ss,
                    dim,
                    v1::DimSample::at_corelet(*corelet),
                    PadType::NoPad,
                    v1::SymbolicRead::Max,
                );
                let share = Extent(div_ceil_i64(work.0, rows));
                per_corelet.insert(
                    *corelet,
                    std::iter::repeat_n(share, Target::PT_ROWS as usize).collect(),
                );
            }
            self.edit(ss, |half| {
                half.dims.row_split.insert(dim, per_corelet.clone());
            });
            // `ds.el_.rowSplit_[dim] = ds.ss_.rowSplit_[dim]` — UNGUARDED in the reference, unlike
            // the corelet and PE/SFP blocks.
            let from_ss = self
                .half(ss)
                .and_then(|half| half.dims.row_split.get(&dim).cloned())
                .unwrap_or_default();
            self.edit(el, |half| {
                half.dims.row_split.insert(dim, from_ss.clone());
            });
        }

        // ── 3. `for (auto dim : peSfpSplitDims)` (`:1778`) ────────────────────────────────────────
        for dim in pe_sfp_split {
            for at in [ss, el] {
                if self
                    .half(at)
                    .is_some_and(|half| half.dims.pe_sfp_split.contains_key(dim))
                {
                    continue;
                }
                let mut per_corelet: BTreeMap<Corelet, v1::PeSfpShares> = BTreeMap::new();
                for corelet in &used {
                    let work = v1::ExploreStages::extent(
                        self,
                        at,
                        *dim,
                        v1::DimSample::at_corelet(*corelet),
                        PadType::NoPad,
                        v1::SymbolicRead::Max,
                    );
                    per_corelet.insert(
                        *corelet,
                        v1::PeSfpShares {
                            pe: Extent(div_ceil_i64(work.0, 2)),
                            sfp: Extent(div_floor_i64(work.0, 2)),
                        },
                    );
                }
                self.edit(at, |half| {
                    half.dims.pe_sfp_split.insert(*dim, per_corelet.clone());
                });
            }
        }
        Some(())
    }
}

impl tr::StageExtents for Dsc2Stages<'_, '_> {
    /// ⭐ One stage's stick-view extent for one dim — `primaryDimToVal_st(dim)`
    /// (`dsc/dims.cpp:647`), whose five defaults are all the WHOLE of their axis, folded by the same
    /// [`crate::schedule::l3::dsc::StageDims::sampled_extent`] [`v1::ExploreStages::extent`] uses.
    ///
    /// ⛔ SO A STAGE THAT STATES A SPLIT ON `dim` NO LONGER STOPS ENTRY 242: `clId = -1` is the first
    /// corelet's share where the dim is row-split and the SUM over corelets where it is also
    /// corelet-split (`:668-674`), which is a distinction only the ported fold makes.
    fn stage_extent(&self, stage: DatastageId, dim: PrimaryDim) -> Extent {
        let at = v1::StageSite::ss(stage);
        if let Some(half) = self.half(at) {
            if let Some(extent) = half.dims.sampled_extent(
                dim,
                v1::DimSample::WHOLE,
                &tu::PaddingForm::default(),
                None,
                false,
            ) {
                return extent;
            }
            if let Some(extent) = half.raw_slot(dim, PadType::NoPad, v1::SymbolicRead::Max) {
                return extent;
            }
        }
        todo!(
            "tr::StageExtents::stage_extent: dataStageParam_.at({stage:?}).ss_\
             .primaryDimToVal_st({dim:?}) (dsc/dims.cpp:647) ABORTED — an empty inner rowSplit_ map \
             (:672) or calculate_padded's own stop, neither of which is a number to substitute"
        )
    }
}

impl v1::Dsc2Stages for Dsc2Stages<'_, '_> {
    /// ⛔⛔ THE ONE METHOD THE CELL CANNOT SERVE — see this module's header. A `&mut` borrow cannot
    /// come out of a [`RefCell`], and handing back an owned copy is the exact defect this method's own
    /// doc warns about: *"A freshly built map here would DROP every effect entries 302, 303 and 338
    /// have, and it would still compile."*
    fn as_util(&mut self) -> &mut tu::DataStages<Self::Dims> {
        todo!(
            "v1::Dsc2Stages::as_util: wants a &mut to THE MAP THIS TYPE'S OWN ExploreStages READS. \
             That map lives in Dsc2State behind a RefCell, because Dsc2Carriers borrows `dsc` and \
             `stages` disjointly and v1::StageSizes on the `dsc` half reads the same dataStageParam_ \
             — so no owner can hand out a &mut. Returning an owned copy would drop every effect \
             entries 302/303/338 have, which is what this method's doc forbids."
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE DDL MATCH AND EXPORT SITE.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ ONE DSC AS THE DDL MATCH AND THE EXPORT READ AND WRITE IT — the `Y` carrier.
#[derive(Debug)]
pub struct Dsc2Ddl<'s, 'l> {
    state: &'s Dsc2State<'l>,
    dsc: DscIdx,
}

impl<'s, 'l> Dsc2Ddl<'s, 'l> {
    /// The DDL site of that DSC.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>, dsc: DscIdx) -> Self {
        Self { state, dsc }
    }

    fn facts(&self) -> &'s super::ddc_state::Dsc2Facts {
        self.state
            .facts(self.dsc)
            .expect("a Dsc2Ddl is only built for a DSC the state holds facts for")
    }
}

impl conv::AllocationSite for Dsc2Ddl<'_, '_> {
    /// `labeledDs_.at(lds).memOrg_.at(unit).allocateNode_` — ⭐ ANSWERED off the tree's own `memOrg_`.
    fn lds_allocation(&self, lds: LdsIdx, unit: SenComponent) -> Option<AllocId> {
        let tree = self.state.tree(self.dsc)?;
        let node = tree.org(lds)?.node(unit)?;
        tree.with(|held| held.allocate(node).map(|(alloc, _)| alloc))
    }

    /// `constantInfo_.at(constant).allocations_.at(unit)` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::ConstantInfo::allocations`], with [`None`] for either `.at()`'s
    /// throw exactly as the trait's own [`Option`] spells it.
    fn constant_allocation(&self, constant: ConstIdx, unit: SenComponent) -> Option<AllocId> {
        self.facts().with_dsc(|dsc| {
            dsc.ddc
                .constants
                .get(&constant)?
                .allocations
                .get(&unit)
                .copied()
        })
    }
}

impl conv::InternalTensorSite for Dsc2Ddl<'_, '_> {
    /// `dsc.labeledDs_`'s tail — ⭐ ANSWERED.
    fn labeled_ds_tail(&self) -> Option<conv::LabeledDsTail> {
        self.facts().with_dsc(|dsc| {
            let positions = ddc_state::lds_positions(dsc);
            Some(conv::LabeledDsTail {
                insert_position: *positions.last()?,
                last_lds: dsc.labeled_ds.back().recorded(),
            })
        })
    }

    /// `++dsc.labeledDs_.back().ldsIdx_` — ⭐ ANSWERED through the shared `currDsc` cell, the same
    /// write [`tu::NewLabeledDs::set_last_recorded_lds_idx`] makes.
    fn set_last_lds_idx(&mut self, lds: LdsIdx) {
        self.facts().with_dsc_mut(|dsc| {
            dsc.labeled_ds.back_mut().set_recorded(lds);
        });
    }

    /// ⛔⛔ `dsc.labeledDs_.insert(end() - 1, newLds)` — AND WHAT BLOCKS IT IS NOW `density_` AND
    /// `referenceLdsIdx_`, NOT THE CELL OR THE RECORD. `addInternalTensor`
    /// (`ddc/ddl/ddl_conversion.cpp:482-498`) copies EIGHT fields off the reference entry —
    /// `dsType_`, `segment_ = STACK`, `isFirstUse_`, `scale_`, `density_`, `wordLength`,
    /// `dataFormat_`, `referenceLdsIdx_` — and [`crate::schedule::l3::dsc::LabeledDs`] projects
    /// neither `density_` (`dsc/dscdefn.h:333`), `segment_` (`:328`), `isFirstUse_` (`:331`) nor
    /// `referenceLdsIdx_` (`:324`).
    ///
    /// ⛔ AND THE ONE THAT DECIDES A SIZE IS `density_`: it is the per-layout-dim occupancy
    /// `getBufferCapacityForNode` multiplies an extent by, so a copy that dropped it would size the
    /// internal tensor's buffer as if it were dense.
    fn insert_internal_tensor(&mut self, _new: conv::InternalTensor) {
        todo!(
            "conv::InternalTensorSite::insert_internal_tensor: wants \
             labeledDs_.insert(end() - 1, newLds) copying dsType_/segment_/isFirstUse_/scale_/\
             density_/wordLength/dataFormat_/referenceLdsIdx_ off the reference \
             (ddc/ddl/ddl_conversion.cpp:482-498). wordLength and dataFormat_ ARE now carried; \
             density_ (dsc/dscdefn.h:333), segment_, isFirstUse_ and referenceLdsIdx_ are not, and \
             density_ is what getBufferCapacityForNode multiplies an extent by"
        )
    }

    /// ⛔ `computeOp_.at(compute_op).interimLabeledDs.push_back(&newLds)` — `interimLabeledDs` is not
    /// a field of [`v1::DscComputeOp`].
    fn add_interim_lds(&mut self, _compute_op: conv::ComputeOpIdx, _lds: LdsIdx) {
        todo!(
            "conv::InternalTensorSite::add_interim_lds: wants \
             computeOp_.at(op).interimLabeledDs.push_back(&newLds) — interimLabeledDs is not a field \
             of v1::DscComputeOp"
        )
    }

    /// ⭐⭐ `traverseTreeDFSMutable(nullptr, {ALLOCATE, TRANSFER, COMPUTE})` PROJECTED ONTO EVERY LDS
    /// SLOT (`ddc/ddl/ddl_conversion.cpp:531-556`), IN THE WALK'S OWN ORDER — ANSWERED.
    ///
    /// ⛔⛔ THE DOC THAT USED TO STAND HERE SAID *"three of its six arms need a COMPUTE node, which
    /// `super::tree::Kind` has no arm for"* AND THAT IS NO LONGER TRUE: [`super::tree::Kind::Compute`]
    /// holds a whole [`ComputeNode`], and [`conv::ScheduleWrites::add_compute`]/`mint_compute` on this
    /// very type are what put them there. The `DT_ERROR("the node has to be either compute, transfer
    /// or allocate.")` (`:557`) is unspellable because the walk is filtered to those three and every
    /// other [`super::tree::Kind`] is simply not projected.
    ///
    /// ⛔ [`conv::LdsSlot::OpaqueCompute`] IS EMITTED FOR EVERY COMPUTE, AND THAT IS NOT A WIDENING.
    /// `isOpaqueOp_` is not projected onto [`ComputeNode`] — membership in `Metadata::opaque_ops` IS
    /// that flag ([`conv::op_opaque`]'s own doc) — and the [`Metadata`] is a borrow DISJOINT from this
    /// site. But [`conv::add_internal_tensor`] guards that arm with
    /// `metadata.opaque_ops.get_mut(&node)`, which is the reference's own `opaqueOps_.at(cn)` and is
    /// [`None`] for exactly the computes whose `isOpaqueOp_` is clear. So the pair of them applies the
    /// reference's predicate; a site that tried to apply it alone would have to guess.
    ///
    /// ⛔ AND THE OPAQUE SLOT COMES FIRST, PER NODE, BESIDE the inputs and outputs and not instead of
    /// them: the reference retags `opaqueOps_.at(cn).ldsIdx_` and THEN walks
    /// `inputsLdsAndLoopOffsets_` and `outputsLdsAndLoopOffsets_` on the same node (`:534-544`).
    fn lds_slots(&self) -> Vec<conv::LdsSlot> {
        let Some(tree) = self.state.tree(self.dsc) else {
            return Vec::new();
        };
        tree.with(|held| {
            let mut slots = Vec::new();
            for node in held.dfs() {
                match held.kind_of(node) {
                    Some(super::tree::Kind::Compute(compute)) => {
                        slots.push(conv::LdsSlot::OpaqueCompute(node));
                        slots.extend(
                            (0..compute.inputs.len())
                                .map(|at| conv::LdsSlot::ComputeInput(node, at)),
                        );
                        slots.extend(
                            (0..compute.outputs.len())
                                .map(|at| conv::LdsSlot::ComputeOutput(node, at)),
                        );
                    }
                    Some(super::tree::Kind::Transfer(transfer)) => {
                        slots.push(conv::LdsSlot::TransferSrc(node));
                        slots.extend(
                            (0..transfer.dsts.len())
                                .map(|at| conv::LdsSlot::TransferDst(node, at)),
                        );
                    }
                    Some(super::tree::Kind::Allocate(..)) => {
                        slots.push(conv::LdsSlot::Allocate(node));
                    }
                    _ => {}
                }
            }
            slots
        })
    }

    /// ⭐ The read half of the same walk — `myLdsIdx_` of that slot, [`None`] for the reference's `-1`.
    ///
    /// ⛔ [`conv::LdsSlot::OpaqueCompute`] IS [`None`] BY CONTRACT, not by omission: the trait's own
    /// doc says *"that slot lives in the [`Metadata`], and the port reads and writes it there"*, and
    /// [`conv::add_internal_tensor`] never asks this method for one.
    fn slot_lds(&self, slot: conv::LdsSlot) -> Option<LdsIdx> {
        let tree = self.state.tree(self.dsc)?;
        tree.with(|held| match slot {
            conv::LdsSlot::OpaqueCompute(_) => None,
            conv::LdsSlot::ComputeInput(node, at) => match held.kind_of(node)? {
                super::tree::Kind::Compute(compute) => compute.inputs.get(at)?.data.my_lds_idx,
                _ => None,
            },
            conv::LdsSlot::ComputeOutput(node, at) => match held.kind_of(node)? {
                super::tree::Kind::Compute(compute) => compute.outputs.get(at)?.data.my_lds_idx,
                _ => None,
            },
            conv::LdsSlot::TransferSrc(node) => match held.kind_of(node)? {
                super::tree::Kind::Transfer(transfer) => transfer.src.data.my_lds_idx,
                _ => None,
            },
            conv::LdsSlot::TransferDst(node, at) => match held.kind_of(node)? {
                super::tree::Kind::Transfer(transfer) => {
                    transfer.dsts.get(at)?.data.my_lds_idx
                }
                _ => None,
            },
            conv::LdsSlot::Allocate(node) => match held.kind_of(node)? {
                super::tree::Kind::Allocate(_, minted) => Some(minted.lds),
                _ => None,
            },
        })
    }

    /// ⛔ And its write half — the one of the three that STILL STOPS.
    ///
    /// ⛔⛔ [`super::tree::TreeData`] HAS NO PER-SLOT LDS SETTER. `set_transfer` replaces a whole
    /// transfer node, but a COMPUTE's `inputsLdsAndLoopOffsets_.at(i).myLdsIdx_` and an ALLOCATE's
    /// `ldsIdx_` have no mutator at all — and every mutator of that type is `pub(super)`, so the
    /// setter belongs there and not here.
    ///
    /// ⛔ AND A PARTIAL WRITE IS WORSE THAN THIS STOP. Retagging only the transfer slots would leave
    /// every compute still naming the OLD last index while the LDS list has already been renumbered,
    /// which is a silently wrong operand and not a missing one.
    fn set_slot_lds(&mut self, slot: conv::LdsSlot, lds: LdsIdx) {
        todo!(
            "conv::InternalTensorSite::set_slot_lds: {slot:?} := {lds:?} — \
             super::tree::TreeData has no per-slot LDS setter (a COMPUTE's \
             inputsLdsAndLoopOffsets_.at(i).myLdsIdx_ and an ALLOCATE's ldsIdx_ have no mutator), and \
             writing only the TRANSFER slots would leave every compute naming the old last index. \
             lds_slots and slot_lds above are ANSWERED; this is the one half that needs a `stages` \
             mutator"
        )
    }

    /// `compAndAllocNode`'s own `allocNode->ldsIdx_` — ⭐ ANSWERED off the L3 view.
    fn allocation_lds(&self, alloc: AllocId) -> Option<LdsIdx> {
        let tree = self.state.tree(self.dsc)?;
        tree.with(|held| {
            held.node_of_alloc(alloc)
                .and_then(|node| held.allocate(node))
                .map(|(_, node)| node.lds)
        })
    }

    /// ⛔ `allocNode->ldsIdx_ = lds` — the same missing setter
    /// [`tu::NewLabeledDs::set_alloc_lds_idx`] names.
    fn set_allocation_lds(&mut self, _alloc: AllocId, _lds: LdsIdx) {
        todo!(
            "conv::InternalTensorSite::set_allocation_lds: wants allocNode->ldsIdx_ = lds — \
             super::tree::TreeData has no setter for L3AllocateNode::lds"
        )
    }
}

impl conv::DdlSite for Dsc2Ddl<'_, '_> {
    /// ⛔ `dsc.computeOp_.front().attributes_.dataFormat_` — ⭐ ANSWERED, because
    /// [`v1::DscComputeOp::format`] IS that field.
    fn fused_format(&self) -> Option<DataFormat> {
        self.facts().ops().first().and_then(|op| op.format)
    }

    /// `dsc.labeledDs_.at(lds).dataFormat_` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::LdsRecord::data_format`]. This is what the DDL match BINDS each
    /// operand's type by, and it VARIES on real data: `SEN143_FP8` on 7 of `g0/`'s 580 labelled DSs and
    /// `SEN169_FP16` on the other 573.
    fn lds_format(&self, lds: LdsIdx) -> Option<DataFormat> {
        self.facts()
            .with_lds(lds, |held| held.record().data_format)
            .flatten()
    }

    /// ⭐⭐ `addressGranularityScalePerUnit.count({senCompToGenericComp.at(unit), storage})` —
    /// ANSWERED, and answered as the VENDOR'S OWN TYPE: [`address_location`] hands back the
    /// [`DataLocation`] row that pair names, so this is `count() == 1` and not a re-listed table.
    ///
    /// ⛔⛔ IT IS ON THE DDL EXPANSION'S CRITICAL PATH AND A `todo!` HERE STOPPED IT.
    /// `setDataLocAndInfo` returns [`None`] the moment this answers false for a memory storage
    /// (`ddl/conversion.rs:2913`), so every `ddl.data_transfer` and every `ddl.compute` end that
    /// names an LX, an L0 or the HBM asks this — the four transfers and the one compute
    /// `g0/debug/sdsc_0/sdsc.json` still owes below `lx_below_schedule` among them.
    ///
    /// ⭐ THE `senCompToGenericComp` HALF IS THE CALLER'S: `conv::generic_component(own)` is applied
    /// at the call site (`ddl/conversion.rs:2911`), so what arrives here is already generic and
    /// mapping it again would be a second answer.
    fn unit_reaches(&self, unit: SenComponent, storage: SenComponent) -> bool {
        address_location(unit, storage).is_some()
    }

    /// `dsc.getDimIndexInLayoutOrder(labeledDs_.at(lds).dsType_, dim) >= 0` — ⭐ ANSWERED off
    /// `layoutDimOrder_`.
    fn dim_in_layout_order(&self, lds: LdsIdx, dim: PrimaryDim) -> bool {
        self.facts().with_dsc(|dsc| {
            dsc.layout_dims
                .get(&lds)
                .is_some_and(|layout| layout.index_of(dim).is_some())
        })
    }

    /// `sdsc.numWkSlicesPerDim_.at(dim)` — ⭐ ANSWERED off the super-DSC's map, copied into the state.
    fn wk_slices(&self, dim: PrimaryDim) -> Option<u32> {
        self.state.wk_slices(dim).map(|count| count.get())
    }

    /// `sdsc.coreIdToWkSlice_` — ⭐ ANSWERED, likewise; the ring's neighbour lookup needs the WHOLE
    /// slice vector per core, which is what this hands back.
    fn core_work_slices(&self) -> BTreeMap<Core, WkSlice> {
        self.state.core_wk_slices()
    }

    /// `dsc.primaryDsInfo_.at(labeledDs_.at(lds).dsType_)`'s stick pair — ⭐ ANSWERED.
    fn stick_dims(&self, lds: LdsIdx) -> Option<StickDims> {
        self.facts()
            .with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds))
    }
}

impl conv::MatchSite for Dsc2Ddl<'_, '_> {
    /// `dsc.computeOp_`, IN ORDER — ⭐ ANSWERED from `computeOp_`.
    ///
    /// ⛔ `coreExclude`/`coreClExclude` ARE EMPTY, and that is a projection gap rather than a
    /// reading: [`v1::DscComputeOp`] carries the five fields entries 307/308 read and neither exclude
    /// list. ⚠️ An op whose exclusions the DDL match should have honoured will bind on every core.
    /// ⛔ AND AN OP WITH NO `opFuncName` IS DROPPED, not defaulted: [`conv::BindableOp::op_func`] has
    /// no absent state and the reference's `NONE` cannot bind a template.
    fn bindable_ops(&self) -> Vec<conv::BindableOp> {
        self.facts()
            .ops()
            .into_iter()
            .filter_map(|op| {
                Some(conv::BindableOp {
                    op_func: op.op_func?,
                    format: op.format,
                    inputs: op.inputs,
                    outputs: op.outputs,
                    core_exclude: BTreeSet::new(),
                    core_cl_exclude: BTreeSet::new(),
                })
            })
            .collect()
    }

    /// `dsc.labeledDs_.at(lds).wordLength` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::LdsRecord::word_length`]. ⛔ [`None`] IS THAT `.at()`'s THROW and
    /// NOT the declared `0`: the DDL match compares this width against the template's own
    /// `bitSize_ / 8`, and a `0` for an absent entry would report a mismatch instead of a missing
    /// operand.
    fn lds_word_length(&self, lds: LdsIdx) -> Option<WordLength> {
        self.facts().with_lds(lds, |held| held.record().word_length)
    }

    /// `newLds.wordLength = type->bitSize_ / 8` — ⭐ ANSWERED through the shared `currDsc` cell.
    fn set_lds_word_length(&mut self, lds: LdsIdx, length: WordLength) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_word_length(length));
    }

    /// `newLds.dataFormat_ = type->dataFormat_` — ⭐ ANSWERED, and it is the SAME `labeledDs_` entry
    /// [`Self::lds_word_length`] reads, so the width and the format a template binds cannot drift.
    fn set_lds_format(&mut self, lds: LdsIdx, format: DataFormat) {
        let _: Option<()> = self
            .facts()
            .with_lds_mut(lds, |held| held.set_data_format(format));
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐⭐ `dsc.scheduleTree_` — THE LIVE TREE, WHICH IS WHAT THE DDL CONVERSION NOW MINTS INTO.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐⭐ THE SEAM THAT REPLACED A DROPPED DEEP COPY. `run_v1` used to open the conversion with
/// `DdlConversion::new(store.schedule_head_block())` — a `dsc2::BlockNode` **by value** — so every
/// compute, transfer, loop, sync and condition the DDL walk minted was written into a temporary and
/// discarded, and the census could only ever show the `allocate` nodes stage 2a had seeded. The
/// reference holds `DesignSpaceConfig& dsc` (`ddc/ddl/ddl_conversion.h:511`) and starts
/// `parseDdl2Dsc` from `dsc.scheduleTree_.getHeadMutable()` (`ddl_conversion.cpp:2774`), which is why
/// `performAutomaticShuffling`'s `traverseTreeDFSMutable(nullptr, {COMPUTE})`
/// (`ddc_transformation.cpp:1999`) finds those computes three statements later.
///
/// ⭐ NO DISJOINT-BORROW PROBLEM AND NO STUBBED ACCESSOR. [`super::state::DscTree`] keeps its
/// [`super::tree::TreeData`] behind a [`RefCell`] and hands no borrow of the interior out
/// ([`super::state::DscTree::with_mut`]), so a `&mut self` write here and the `&self` reads
/// [`Dsc2Carriers`]' other nine carriers make cannot alias. That is the same property the `dsc`/
/// `stages` split relies on.
impl conv::ScheduleReads for Dsc2Ddl<'_, '_> {
    /// `scheduleTree_.getHeadMutable()`.
    fn head(&self) -> Option<NodeId> {
        self.state.tree(self.dsc)?.with(|tree| tree.head())
    }

    fn node_name(&self, node: NodeId) -> Option<NodeName> {
        self.state.tree(self.dsc)?.with(|tree| tree.name(node))
    }

    /// `scheduleTree_.getHead()` MATERIALISED — the same walk
    /// [`v1::Dsc2Store::schedule_head_block`] answers with, over the LIVE tree.
    fn head_block(&self) -> Option<BlockNode> {
        self.state.tree(self.dsc)?.with(|tree| {
            let head = tree.head()?;
            Some(super::ddc_store2::head_block_of(tree, head))
        })
    }

    fn transfer(&self, node: NodeId) -> Option<TransferNode> {
        self.state.tree(self.dsc)?.with(|tree| tree.transfer(node))
    }

    fn compute(&self, node: NodeId) -> Option<ComputeNode> {
        self.state
            .tree(self.dsc)?
            .with(|tree| match tree.kind_of(node) {
                Some(super::tree::Kind::Compute(held)) => Some(held.clone()),
                _ => None,
            })
    }

    /// The `SYNC` this tree carries by that name — the sync pairing holds its ends by NAME
    /// ([`crate::schedule::dsc2::SyncNode::other_ends`]) where the reference holds live pointers.
    fn sync_units(&self, name: &NodeName) -> Option<SyncUnits> {
        self.state.tree(self.dsc)?.with(|tree| {
            let node = tree.find_named(&name.0)?;
            tree.sync_units(node)
        })
    }
}

impl conv::ScheduleWrites for Dsc2Ddl<'_, '_> {
    /// `getHeadMutable()->addChildNode(new dsc2::BlockNode(), /*addFront=*/true)`.
    fn add_root_level_block(&mut self, name: NodeName) -> Option<NodeId> {
        let held = self.state.tree(self.dsc)?;
        held.with_mut(|tree| {
            let head = tree.head()?;
            let node = tree.add(name, super::tree::Kind::Block, None);
            tree.link(node, tu::InsertionPoint::FirstIn(head));
            Some(node)
        })
    }

    /// `currParent->addChildNode(new dsc2::BlockNode())`, DISPATCHED ON THE PARENT — a `CONDITION`
    /// takes it as its next region, which is `ConditionNode::addChildNode`'s override.
    fn add_block(&mut self, parent: NodeId, name: NodeName) -> Option<NodeId> {
        let held = self.state.tree(self.dsc)?;
        held.with_mut(|tree| {
            let then_region = match tree.kind_of(parent) {
                Some(super::tree::Kind::Condition(cond)) => {
                    // `addThenRegion` while that region is empty, then `addElseRegion`, then
                    // `ConditionNode::add_region`'s own refusal.
                    if cond.then_region.is_empty() {
                        Some(true)
                    } else if cond.else_region.is_empty() {
                        Some(false)
                    } else {
                        return None;
                    }
                }
                _ => None,
            };
            let node = tree.add(name, super::tree::Kind::Block, None);
            match then_region {
                Some(then) => tree.add_region(parent, node, then),
                None => tree.link(node, tu::InsertionPoint::LastIn(parent)),
            }
            Some(node)
        })
    }

    /// `currParent->addChildNode(new dsc2::LoopNode())`.
    ///
    /// ⛔ `numId_`/`denId_` MUST BOTH BE STATED, which is [`tu::LoopNode`]'s own contract — *"every
    /// callsite of the constructor passes a real pair"*. A parametric loop carries `-1` for both and
    /// is the arm [`conv::op_parametric_loop`] refuses rather than inventing a stage for; the
    /// [`None`] here is that same fact, reached from the other side.
    fn add_loop(&mut self, parent: NodeId, held: LoopNode) -> Option<NodeId> {
        let (num, den) = (held.num?, held.den?);
        if held.parametric_lds.is_some() {
            return None;
        }
        let name = held.block.base.name.clone();
        // `dims_`, IN THE LOOP'S OWN ORDER — non-empty by [`tu::LoopDims`]' construction, which is
        // entry 114's *"Cannot construct loop with no dimensions"* made unspellable. The DDL arm has
        // already refused an empty band before reaching here.
        let kinded = |dim: &crate::schedule::dsc2::LoopDim| tu::PrimaryDimAndKind {
            dim: dim.dim,
            kind: dim.kind,
        };
        let (first, rest) = held.dims.split_first()?;
        let minted = tu::LoopNode {
            name: name.clone(),
            num,
            den,
            dims: tu::LoopDims::new(kinded(first), rest.iter().map(kinded).collect()),
        };
        let tree = self.state.tree(self.dsc)?;
        Some(tree.with_mut(|tree| tree.add(name, super::tree::Kind::Loop(minted), Some(parent))))
    }

    /// `currParent->addChildNode(new dsc2::TransferNode())`.
    fn add_transfer(&mut self, parent: NodeId, held: TransferNode) -> Option<NodeId> {
        let tree = self.state.tree(self.dsc)?;
        let name = held.name.clone();
        Some(tree.with_mut(|tree| tree.add(name, super::tree::Kind::Transfer(held), Some(parent))))
    }

    fn set_transfer(&mut self, node: NodeId, held: TransferNode) -> Option<()> {
        let tree = self.state.tree(self.dsc)?;
        tree.with_mut(|tree| tree.set_transfer(node, held));
        Some(())
    }

    /// `currParent->addChildNode(new dsc2::ComputeNode())`.
    fn add_compute(&mut self, parent: NodeId, held: ComputeNode) -> Option<NodeId> {
        let tree = self.state.tree(self.dsc)?;
        let name = held.name.clone();
        Some(tree.with_mut(|tree| tree.add(name, super::tree::Kind::Compute(held), Some(parent))))
    }

    /// `new dsc2::ComputeNode()` alone — `TreeData::add` with no parent, which is a node the tree
    /// holds and no block yet lists.
    fn mint_compute(&mut self, held: ComputeNode) -> Option<NodeId> {
        let tree = self.state.tree(self.dsc)?;
        let name = held.name.clone();
        Some(tree.with_mut(|tree| tree.add(name, super::tree::Kind::Compute(held), None)))
    }

    /// `currParent->addChildNode(node)`.
    fn link_child(&mut self, parent: NodeId, node: NodeId) -> Option<()> {
        let tree = self.state.tree(self.dsc)?;
        tree.with_mut(|tree| tree.link(node, tu::InsertionPoint::LastIn(parent)));
        Some(())
    }

    /// `currParent->addChildNode(new dsc2::SyncNode())`.
    fn add_sync(&mut self, parent: NodeId, held: SyncNode) -> Option<NodeId> {
        let tree = self.state.tree(self.dsc)?;
        let name = held.base.name.clone();
        Some(tree.with_mut(|tree| tree.add(name, super::tree::Kind::Sync(held), Some(parent))))
    }

    /// `currParent->addChildNode(new dsc2::ConditionNode())`.
    ///
    /// ⛔ THE TWO REGIONS ARE DROPPED HERE AND THAT IS NOT A LOSS: they are EMPTY at every mint site
    /// ([`conv::op_if`], [`conv::op_sync`]'s corelet split), and the blocks that fill them are minted
    /// under this node by [`Self::add_block`] — which is where the tree records them.
    fn add_condition(&mut self, parent: NodeId, held: ConditionNode) -> Option<NodeId> {
        if !matches!(held.next, CondRegions::Empty) {
            return None;
        }
        let tree = self.state.tree(self.dsc)?;
        let cond = super::tree::Cond {
            // ⭐ `hasCoreClCond()` DECIDES WHICH HALF IS STATED — an empty `twoLevelOrOfAnds_` IS the
            // core/corelet-guarded case (`dsc/dsc2.h:693-695`), so the empty composite is [`None`]
            // here rather than a stated-empty guard.
            loop_cond: (!held.loop_cond.two_level_or_of_ands.is_empty())
                .then(|| held.loop_cond.clone()),
            cores: (!held.core_cl_cond.is_empty())
                .then(|| v1::CoreClSet(held.core_cl_cond.clone())),
            then_region: Vec::new(),
            else_region: Vec::new(),
        };
        let name = held.base.name.clone();
        Some(tree.with_mut(|tree| tree.add(name, super::tree::Kind::Condition(cond), Some(parent))))
    }

    fn add_sync_other_ends(&mut self, name: &NodeName, others: &[NodeName]) -> Option<()> {
        let tree = self.state.tree(self.dsc)?;
        tree.with_mut(|tree| {
            let node = tree.find_named(&name.0)?;
            for other in others {
                tree.add_sync_other_end(node, other.clone());
            }
            Some(())
        })
    }
}

impl conv::DdlSizes for Dsc2Ddl<'_, '_> {
    /// ⛔ `getBlockTransferSize(node, src_.unit_, 0, false, true)` FOLDED WITH the
    /// `!getRelevantCoreCl().empty()` that gates it — the first is a `dsc/` accessor and the second
    /// wants the `relevantCoreCl_` map `setRelevantCompCoreCl` fills.
    fn block_transfer_size(&self, _transfer: NodeId) -> Option<Elements> {
        todo!(
            "conv::DdlSizes::block_transfer_size: wants getBlockTransferSize(node, src_.unit_, 0, \
             false, true) gated on !getRelevantCoreCl().empty() — a `dsc/` accessor plus the \
             relevantCoreCl_ map dsc.setRelevantCompCoreCl() (dsc/dsc2.cpp:2647) fills"
        )
    }

    /// ⛔ `getBufferCapacityForNode(allocatenode, ldsIdx_, component_, 0, 0)` — the SAME capacity
    /// call [`v1::Placement::buffer_capacity`] names.
    ///
    /// ⭐ THE CALLEE IS PORTED — [`crate::schedule::l3::capacity::buffer_capacity`] with
    /// [`crate::schedule::l3::capacity::DscSizing`] over this site's own `currDsc`, and THIS carrier
    /// can reach both halves the shared stage-2b one cannot: `TreeData::node_of_alloc(alloc)` for the
    /// allocate node and `owner_loop`/`loop_node` for the `AncestorLoops` chain, exactly as
    /// [`conv::AllocationSite::lds_allocation`] above already walks the tree.
    /// ⛔⛔ TWO THINGS STILL BLOCK IT, AND BOTH ARE DECISIONS RATHER THAN PORTS.
    /// 1. **THE UNIT IS WRONG IN THE TRAIT.** `getBufferCapacityForNode` returns `int64_t` **BYTES**
    ///    (`dsc/dsc2.cpp:3988-4004`: it multiplies `myLds.wordLength` in and compares against
    ///    `bytesPerStick`), and [`conv::DdlSizes::buffer_capacity`] declares [`Elements`]. Handing
    ///    bytes back as elements would stamp a DDL `allocation_size=` too large by `wordLength`.
    /// 2. **THE THREE `dsc2::AllocateNode` MEMBERS** [`crate::schedule::l3::capacity::AllocSizing`]
    ///    needs — `ignoreSymbolicVolumeLimits_` (`dsc/dsc2.h:1002`), `backGapCore_` (`:989`) and
    ///    `indirectAllocType_` (`:994`) — have no slot on our [`crate::schedule::dsc2::AllocateNode`].
    /// ⛔ AND [`None`] IS NOT AVAILABLE AS THE REFUSAL: `convert_dsc2_ddl` writes
    /// `allocation_size: node.lds.and_then(|_| self.site.buffer_capacity(alloc))`
    /// (`schedule/ddl/conversion.rs:5231`), so [`None`] EMITS A DDL ALLOCATE OP WITH NO
    /// `allocation_size=` AT ALL — a silent omission where the reference always states one. The
    /// `todo!` is louder and therefore correct.
    fn buffer_capacity(&self, _alloc: AllocId) -> Option<Elements> {
        todo!(
            "conv::DdlSizes::buffer_capacity: l3::capacity::buffer_capacity (dsc/dsc2.cpp:3977) IS \
             PORTED and this site can reach the node and its loop chain — blocked on (1) this trait \
             declaring Elements where the reference returns BYTES (it multiplies wordLength in at \
             :3999) and (2) the three AllocateNode members AllocSizing needs: \
             ignoreSymbolicVolumeLimits_ (dsc/dsc2.h:1002), backGapCore_ (:989), indirectAllocType_ \
             (:994). None is not the refusal here: conversion.rs:5231 turns it into a DDL allocate \
             op stating no allocation_size at all"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ THE DDL TEMPLATE SET — THE GENERATED MODULES, WHICH IS WHERE THIS STAGE USED TO STOP.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// THE `T` CARRIER — [`conv::DdlTemplateSet`], whose one method hands back a
/// [`conv::StatedTemplate`] with SIX parts: `source`, `program`, `binds`, `padded`, `constraints`,
/// `root`.
///
/// ⭐⭐ ALL SIX NOW COME OUT OF THE `.ddl`. `build.rs` used to emit exactly one — `PROGRAMS` served
/// `program` and the other four were mentioned ZERO times each — so this carrier's only honest answer
/// was [`None`], and stage 2b's `run_v1` stopped here. It now walks each template MODULE-WIDE beside
/// the per-bind walk and emits `crate::generated::MODULES`;
/// [`crate::schedule::ddl::templates::DdlTemplates`] is the one implementor over it, and this carrier
/// delegates.
///
/// ⛔ THE DELEGATION IS TOTAL, so nothing is recorded here any more. `Template`'s variants and
/// `MODULES`' rows are minted from the same census of `ddl_templates/*.ddl`, so a template a candidate
/// list can name is a template this set holds.
///
/// ⛔ AND IT DOES NOT SHORT-CIRCUIT. `select_and_parse_ddl_template` reaches this only AFTER
/// `ddl_templates(opFunc, A::GEN)` answered [`Some`] (`ddl/conversion.rs:6011`), so an op-func with no
/// candidate templates never asks — it returns *"no DDL available for op …"* and stage 2b answers
/// [`v1::DscFilled::No`] for the super-DSC, which is a COMPLETE run of stage 2b and not a stop.
#[derive(Debug)]
pub struct DdcTemplates<'s, 'l> {
    /// Kept so a future refusal of this carrier's own has somewhere to go; the set itself has none.
    #[expect(
        dead_code,
        reason = "the state is the refusal sink every other carrier of this module holds, and this \
                  one no longer refuses — see the type's note"
    )]
    state: &'s Dsc2State<'l>,
    /// The generated set this carrier delegates to. A FIELD and not a temporary, because
    /// [`conv::DdlTemplateSet::stated`] hands back a [`conv::StatedTemplate`] borrowed from the set.
    set: crate::schedule::ddl::templates::DdlTemplates,
}

impl<'s, 'l> DdcTemplates<'s, 'l> {
    /// The template set.
    #[must_use]
    pub const fn new(state: &'s Dsc2State<'l>) -> Self {
        Self {
            state,
            set: crate::schedule::ddl::templates::DdlTemplates::new(),
        }
    }
}

impl conv::DdlTemplateSet for DdcTemplates<'_, '_> {
    type Source = crate::schedule::ddl::templates::TemplateSource;

    /// ⭐ THE GENERATED MODULES, verbatim — see [`crate::schedule::ddl::templates::DdlTemplates`].
    fn stated(
        &self,
        template: crate::generated::Template,
    ) -> Option<conv::StatedTemplate<'_, Self::Source>> {
        conv::DdlTemplateSet::stated(&self.set, template)
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ THE PROVIDER — every store of one DSC, borrowed together.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE DSC'S CARRIERS, OWNED SO THAT [`v1::Dsc2Sites::carriers`] CAN HAND OUT TEN DISJOINT `&mut`s.
#[derive(Debug)]
struct PerDsc<'s, 'l> {
    store: Dsc2Store<'s, 'l>,
    stages: Dsc2Stages<'s, 'l>,
    ddl: Dsc2Ddl<'s, 'l>,
    allocs: v1::AllocArena,
    computes: v1::ComputeArena,
    syncs: BTreeMap<NodeId, crate::schedule::dsc2::SyncNode>,
}

/// ⭐⭐ WHERE ONE DSC'S CARRIERS COME FROM — `sdsc.dscs_.at(idx)` and everything hung off it.
///
/// ⛔ EVERY CARRIER IS OWNED HERE because [`v1::Dsc2Carriers`] is TEN `&'a mut`s taken from
/// `&mut self` at once: they must be ten disjoint FIELDS, which is what this is.
#[derive(Debug)]
pub struct Dsc2Provider<'s, 'l> {
    state: &'s Dsc2State<'l>,
    per: Vec<PerDsc<'s, 'l>>,
    trackers: ddc_state::DdcTrackers,
    sink: ddc_state::DdcSink,
    symbols: ddc_state::DdcSymbols,
    coords: ddc_state::DdcCoords,
}

impl<'s, 'l> Dsc2Provider<'s, 'l> {
    /// The provider over every DSC the state holds, along the fold manager's own address coordinates.
    #[must_use]
    pub fn new(state: &'s Dsc2State<'l>, coords: &AddressFoldCoords) -> Self {
        let per = (0..state.len())
            .filter_map(|at| u32::try_from(at).ok().map(DscIdx))
            .map(|dsc| PerDsc {
                store: Dsc2Store::new(state, dsc, coords.clone()),
                stages: Dsc2Stages::new(state, dsc),
                ddl: Dsc2Ddl::new(state, dsc),
                allocs: seed_arena(state, dsc),
                computes: v1::ComputeArena::new(),
                syncs: BTreeMap::new(),
            })
            .collect();
        Self {
            state,
            per,
            trackers: ddc_state::DdcTrackers,
            sink: ddc_state::DdcSink::default(),
            symbols: ddc_state::DdcSymbols,
            coords: ddc_state::DdcCoords,
        }
    }

    /// Where entry 260's fills landed, for a caller that wants to read them back.
    #[must_use]
    pub const fn sink(&self) -> &ddc_state::DdcSink {
        &self.sink
    }
}

impl<'s, 'l> v1::Dsc2Sites for Dsc2Provider<'s, 'l> {
    type Dsc = Dsc2Store<'s, 'l>;
    type Stages = Dsc2Stages<'s, 'l>;
    type Ddl = Dsc2Ddl<'s, 'l>;
    type Trackers = ddc_state::DdcTrackers;
    type Sink = ddc_state::DdcSink;
    type Symbols = ddc_state::DdcSymbols;
    type Coords = ddc_state::DdcCoords;

    /// `dsc.name_` — the construction argument [`Dsc2State::seeded`] takes, which only the two verbose
    /// lines read.
    fn dsc_name(&self, dsc: DscIdx) -> v1::StorageName {
        self.state
            .facts(dsc)
            .map_or_else(|| v1::StorageName(format!("dsc{}", dsc.0)), |facts| facts.name())
    }

    /// ⛔ [`None`] IS `dscs_.at(idx)`'s OWN THROW — a `dscs_` position this provider holds no stores
    /// for, which is what ends `run_v1`'s loop.
    fn carriers(
        &mut self,
        dsc: DscIdx,
    ) -> Option<
        v1::Dsc2Carriers<
            '_,
            Self::Dsc,
            Self::Stages,
            Self::Ddl,
            Self::Trackers,
            Self::Sink,
            Self::Symbols,
            Self::Coords,
        >,
    > {
        let per = self.per.get_mut(usize::try_from(dsc.0).ok()?)?;
        Some(v1::Dsc2Carriers {
            dsc: &mut per.store,
            stages: &mut per.stages,
            ddl: &mut per.ddl,
            allocs: &mut per.allocs,
            computes: &mut per.computes,
            syncs: &mut per.syncs,
            trackers: &mut self.trackers,
            sink: &mut self.sink,
            symbols: &mut self.symbols,
            coords: &self.coords,
        })
    }
}

/// ⭐⭐ THE `ddc` VIEW OF EVERY ALLOCATE NODE THE TREE ALREADY HOLDS — the arena
/// [`v1::Dsc2Carriers::allocs`] is, seeded so that IT AND THE TREE CANNOT DISAGREE.
///
/// ⛔⛔ WITHOUT THIS, STAGE 2B STOPS ON ITS FIRST ALLOCATE. `attach_to_prefilled_schedule` reads
/// `allocs.get(&alloc)?` for every `ALLOCATE` of the tree (`ddc/v1.rs:5964-5967`) and answers [`None`]
/// for one the arena has no entry for — and an EMPTY arena beside a tree with three HBM allocations is
/// exactly the state `AllocArena::new()` leaves.
///
/// ⭐ IT IS NOT A FABRICATION, AND [`super::tree::Org`]'S OWN DOC IS WHY: *"ITS `allocateNode_` IS ONE
/// NODE READ THROUGH TWO VOCABULARIES"* — [`crate::schedule::l3::dl_ops::L3AllocateNode`] is what
/// entry 353 mints and [`crate::schedule::dsc2::AllocateNode`] is what entries 219/220/292 place.
/// Every field below is COPIED from the L3 view of the same node; nothing is derived.
///
/// ⛔⛔ AND THE PLACED STATE IS LEFT AT THE FRESH-NODE DEFAULT, WHICH IS A STATEMENT AND NOT A
/// SHORTCUT. `startAddressCoreCorelet_`, `numBuffers_`, `padding_`, `bufferOffsetCoreCorelet_` and
/// `isStartAddrSymbolic_` are what the MEMORY TRACKER and entries 258/259 write, and
/// [`crate::schedule::dsc2::AllocPlacement`]'s own doc says *"every construction site of a fresh
/// allocate node wants all four at their defaults"*. Stage 2a stops AT that tracker, so an unplaced
/// node is the true state of every allocation this seed can see.
///
/// ⭐ `numBuffers_` IS THE ONE PLACEMENT FIELD THAT IS CARRIED, because entry 353 — a GROWER, not the
/// tracker — is what chooses it: [`crate::schedule::l3::dl_ops::set_lx_buffer_type`] picks
/// `LxBufferChoice::Double` and `create_allocation_and_transfer` stamps it on every LX allocation
/// BEFORE any address exists. `numBuffers_` is a placement INPUT (the double-buffer stride is
/// `numBuffers x bufferOffset`), so silently writing [`crate::schedule::dsc2::NumBuffers::Single`] over
/// a `Double` the L3 scheduler chose would halve every buffer — the copy below is what prevents that,
/// and the mapping is the closed three-way the field's own comment names
/// (*"1:no buffering, 2:double-buffer, -1:streaming buffer"*).
///
/// ⛔ `bufferOffsetCoreCorelet_` IS NOT CARRIED AND MUST NOT BE: it is the tracker's own output
/// (`g0/debug/sdsc_0/sdsc.json` puts it at 256 on every LX node and every core), and the tracker has
/// not run.
///
/// ⭐ `padding_` IS CARRIED, AND THE OBJECTION THIS DOC USED TO HOLD NAMED A DUPLICATION THAT IS GONE.
/// The allocate node's `padding_` (`dsc/dsc2.h:981`) and the L3 view's are now ONE type,
/// [`crate::schedule::ddc::transformation_util::PaddingForm`], so carrying it across is a move and
/// not the `getPadding(dim)` (`dsc/dims.cpp:806`) seam a second spelling would have made of it.
///
/// ⛔ AN ALLOCATION WHOSE `layoutDimOrder_` IS EMPTY IS LIKEWISE LEFT OUT:
/// [`crate::schedule::dsc2::AllocLayout`] is non-empty by type, which is the reference's own
/// `layoutDimOrder_.at(0)` (`ddc/ddcv1.cpp:1704`).
fn seed_arena(state: &Dsc2State<'_>, dsc: DscIdx) -> v1::AllocArena {
    use crate::schedule::dsc2::{AllocLayout, AllocateNode, MaxDimSize, NumBuffers};
    use crate::schedule::l3::dsc::Buffering;

    let mut arena = v1::AllocArena::new();
    let Some(tree) = state.tree(dsc) else {
        return arena;
    };
    tree.with(|held| {
        for node in held.dfs() {
            let Some((alloc, minted)) = held.allocate(node) else {
                continue;
            };
            // `numBuffers_` — the closed three-way the field's own comment names, carried across.
            let num_buffers = match minted.buffering {
                Buffering::None => NumBuffers::Single,
                Buffering::Double => NumBuffers::Double,
                Buffering::Streaming => NumBuffers::Streaming,
            };
            // `padding_` (`dsc/dsc2.h:981`) — ONE `PaddingFormType` on both sides, carried whole.
            // `layoutDimOrder_` zipped with `maxDimSizes_`, innermost first, and an UNBOUNDED entry is
            // the reference's own `resize(n, -1)` (`dsc/dsc2.h:982`) — which is `MaxDimSize::Unset`.
            let mut layout = minted
                .layout
                .0
                .iter()
                .map(|(dim, max)| (*dim, max.map_or(MaxDimSize::Unset, MaxDimSize::Resolved)));
            let Some(first) = layout.next() else {
                let _: Option<()> = state.refuse(
                    "Dsc2Provider: an allocation whose layoutDimOrder_ is EMPTY is left out of the \
                     AllocArena — dsc2::AllocLayout is non-empty by type, which is the reference's \
                     own layoutDimOrder_.at(0) (ddc/ddcv1.cpp:1704)",
                );
                continue;
            };
            arena.insert(
                alloc,
                AllocateNode {
                    name: minted.name.clone(),
                    component: minted.component,
                    lds: Some(minted.lds),
                    // `constIdx_` — this node is a labelled DS's, not a constant's.
                    const_idx: None,
                    temp_storage_for_compute: None,
                    layout: AllocLayout::new(first, layout.collect()),
                    start_address: crate::schedule::dsc2::StartAddress::default(),
                    placement: crate::schedule::dsc2::AllocPlacement {
                        num_buffers,
                        padding: minted.padding.clone(),
                        ..crate::schedule::dsc2::AllocPlacement::default()
                    },
                    gap_stick_spread: BTreeMap::new(),
                    // `allocUsers_` — the users list lives on `super::tree::Org`, and stage 2b's own
                    // `add_alloc_user` writes it there; an arena copy would be a second answer.
                    alloc_users: Vec::new(),
                },
            );
        }
    });
    arena
}

/// ⛔ A `RefCell` IS NAMED IN THIS MODULE'S HEADER — kept referenced so the doc link resolves.
const _: fn(&RefCell<()>) = |_| ();

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐⭐ THE DDL EXPANSION MINTS INTO THE **LIVE** TREE — the seam this module's `ScheduleWrites`
// impl exists to be, tested by READING THE CALLER'S OWN `TreeData` BACK AFTER THE CALL RETURNS.
// ════════════════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod live_tree_tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU32;

    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
    use crate::generated::{Attrs, NameId, PROGRAMS, Program, Stmt, StmtKind};
    use crate::schedule::ddc::fold::{BlockId, NodeId, NodeKind};
    use crate::schedule::ddc::metadata::Metadata;
    use crate::schedule::ddc::transformation::DsType;
    use crate::schedule::ddc::transformation_util::StageName;
    use crate::schedule::ddc::v1;
    use crate::schedule::ddl::conversion::{
        CondProp, DdlConversion, DdlInterface, DdlOp, DdlRoot, RegionId, RegionOp, RegionTree,
        parse_ddl2_dsc,
    };
    use crate::schedule::dsc2::{LdsIdx, NodeName};
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DesignSpaceConfig, DscIdx, DscList,
        FilledDims, LabeledDs, LabeledDsList, NamedDims, Pinning, StageDims, SuperDsc,
    };
    use crate::units::Core;

    use super::super::ddc_state::Dsc2State;
    use super::super::state::DscState;
    use super::Dsc2Ddl;

    /// One core by index.
    fn core(index: u32) -> Core {
        Core::checked(index).expect("a core in range")
    }

    /// THE BAREST DSC THIS WALK NEEDS — no HBM-pinned tensor, so [`DscState::seeded`] leaves exactly
    /// the `root_level_operations` head block and nothing else, and every node counted afterwards is
    /// one the DDL expansion minted.
    fn a_bare_dsc() -> DesignSpaceConfig {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::X, Extent(4));
        dims.extents.insert(PrimaryDim::Y, Extent(2));
        let named = NamedDims {
            name: StageName::default(),
            dims: FilledDims::of(dims).expect("a stage that states a dim"),
        };
        let stage = DataStage {
            ss: named.clone(),
            el: named,
        };
        let two = CoreletsUsed::new(NonZeroU32::new(2).expect("two corelets"));
        DesignSpaceConfig {
            ddc: crate::schedule::l3::dsc::DdcFacts::default(),
            gtr_ids_used: BTreeSet::new(),
            corelets_used: two,
            corelets_used_dsc2: Some(two),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(core(0), vec![core(1)]),
            layout_dims: BTreeMap::new(),
            data_stages: DataStages::new(stage.clone(), stage),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(DsType::Input, vec![], LdsIdx(0), Pinning::default()),
                vec![],
            ),
        }
    }

    /// A program with hand-written statements, wearing the first vendored program's template.
    fn synthetic(names: &'static [&'static str], stmts: &'static [Stmt]) -> Program {
        Program {
            template: PROGRAMS[0].template,
            op_func: "",
            bind: "",
            stmts,
            roles: &[],
            names,
        }
    }

    /// ⭐⭐⭐ EVERY NODE `parse_ddl2_dsc` MINTS REACHES THE TREE THE CALLER STILL OWNS.
    ///
    /// ⛔⛔ THIS IS THE TEST THE DROPPED DEEP COPY COULD NOT FAIL. `run_v1` opened the conversion with
    /// `DdlConversion::new(store.schedule_head_block())` — a `dsc2::BlockNode` BY VALUE — so the walk
    /// wrote every compute, loop, transfer, sync, condition and block into a temporary that was
    /// discarded when the DSC's turn ended, and the live tree kept only the `allocate` nodes stage 2a
    /// had seeded. The reference holds `DesignSpaceConfig& dsc` (`ddc/ddl/ddl_conversion.h:511`) and
    /// starts from `dsc.scheduleTree_.getHeadMutable()` (`ddl_conversion.cpp:2774`).
    ///
    /// ⛔ SO THE ASSERTIONS READ THE `TreeData` OUT OF THE `DscState` THIS TEST OWNS, **AFTER**
    /// `parse_ddl2_dsc` HAS RETURNED, BY NAME AND BY `nodeType_`. Nothing is asserted off a local this
    /// test held before the call, and nothing is asserted off `DdlConversion` — which no longer carries
    /// a tree to assert against.
    ///
    /// ⭐ THE NEGATIVE CONTROL IS THE FIRST ASSERTION, taken on the SAME state before the call: the
    /// seeded tree is ONE node. Reverting `Dsc2Ddl`'s writes to a by-value copy leaves that one node
    /// in place and every assertion after it fails.
    #[test]
    fn every_node_the_ddl_expansion_mints_reaches_the_live_tree() {
        /// One `ddl.operation_bind`, so `processCondition`'s first arm has something to `dyn_cast`.
        static BIND: &[Stmt] = &[Stmt {
            kind: StmtKind::OperationBind,
            depth: 0,
            attrs: Attrs::Bare(StmtKind::OperationBind),
            results: &[NameId(0)],
            operands: &[],
            path: &[],
        }];
        let program = synthetic(&["%the_op"], BIND);
        let sdsc = SuperDsc::new(
            DscList::new(a_bare_dsc(), Vec::new()),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let l3_state = DscState::seeded(&sdsc);

        // ⭐ THE NEGATIVE CONTROL'S BASELINE: the seed is the root block alone, since this DSC pins
        // nothing in HBM. A conversion that minted into a copy leaves this number unchanged.
        assert_eq!(
            l3_state.node_count(),
            1,
            "the seed is `root_level_operations` and nothing else"
        );
        let head = l3_state
            .dsc(DscIdx(0))
            .expect("the one DSC's tree")
            .with(|tree| tree.head())
            .expect("a seeded tree has a head");

        let state2 = Dsc2State::seeded(
            &sdsc,
            &l3_state,
            &[Vec::new()],
            &[v1::StorageName("bare".to_owned())],
        );
        let mut site = Dsc2Ddl::new(&state2, DscIdx(0));
        // ⛔ `belowLxScheduleInsertBlock` IS NEVER THE HEAD — `traverseTreeDFSMutable` seeds from
        // `head_.next_` (`dsc/dsc2.cpp:2233`) — so the reference's `!=` holds and
        // `root_level_operations` gets minted. `NodeId(1)` is the next identity this tree will issue,
        // which no node holds yet, and the comparison only needs it to differ from the head.
        let mut metadata = Metadata {
            below_lx_schedule_insert_block: BlockId::of(&OneBlock, NodeId(1)),
            ..Metadata::default()
        };

        // An UNRESOLVED `ddl.if` — `resolved_conditions` seeded with the default `CondProp`, whose
        // `resolved` is absent — so `op_if` mints a CONDITION and a block per region.
        let mut interface = DdlInterface::default();
        interface
            .resolved_conditions
            .insert(NameId(0), CondProp::default());
        let mut dsc = a_bare_dsc();
        let mut conversion = DdlConversion::new();
        let said = parse_ddl2_dsc(
            &program,
            &mut conversion,
            &mut interface,
            &mut metadata,
            &mut dsc,
            &mut site,
            &DdlRoot {
                regions: RegionTree {
                    ops: BTreeMap::from([(
                        RegionId(0),
                        vec![RegionOp {
                            op: DdlOp::If {
                                condition: NameId(0),
                                then_region: RegionId(1),
                                else_region: RegionId(2),
                            },
                            regions: vec![RegionId(1), RegionId(2)],
                        }],
                    )]),
                },
                dataflows: vec![RegionId(0)],
                transformations: Vec::new(),
            },
        );
        assert_eq!(said, Some(Vec::new()), "the walk ran and said nothing");

        // ⭐⭐ READ BACK OUT OF THE TREE THIS TEST STILL OWNS — four nodes the expansion minted, each
        // by NAME and by `nodeType_`, plus the nesting that says the condition's regions hang off it.
        let tree = l3_state.dsc(DscIdx(0)).expect("the one DSC's tree");
        assert_eq!(
            tree.node_count(),
            5,
            "the seeded root block plus `root_level_operations`, the condition and its two regions: \
             {:?}",
            tree.names()
        );
        assert_eq!(
            tree.kinds(),
            BTreeMap::from([(NodeKind::Block, 4), (NodeKind::Condition, 1)]),
            "one CONDITION and four BLOCKs — a census that still read `Block: 1` would mean the mint \
             went into a copy"
        );
        // ⛔ RESOLVED BY IDENTITY AND NOT BY NAME. The SEED's own head block is called
        // `root_level_operations` too (`state::ROOT_BLOCK_NAME`), so a name lookup finds the head and
        // says nothing about what was minted — which is exactly the confusion `OpOutcome::parent`
        // carrying a `NodeName` used to be made of. The minted block is the head's FIRST child, which
        // is `addChildNode(.., /*addFront=*/true)`.
        let (root, root_name, condition, regions) = tree.with(|held| {
            let root = *held
                .children(head)
                .first()
                .expect("`root_level_operations` reached the LIVE tree as the head's first child");
            let condition = held
                .children(root)
                .into_iter()
                .find(|node| held.node_kind(*node) == Some(NodeKind::Condition))
                .expect("the CONDITION the unresolved `ddl.if` minted reached the LIVE tree");
            let regions: Vec<NodeName> = held
                .children(condition)
                .into_iter()
                .filter_map(|node| held.name(node))
                .collect();
            (root, held.name(root), condition, regions)
        });
        // ⛔ THE MINTED ROOT BLOCK IS NOT THE SEEDED HEAD, which is what `addChildNode(.., front)`
        // means: the head keeps its own identity and gains a child.
        assert_ne!(root, head, "a fresh block, not the seeded head");
        assert_eq!(
            root_name,
            Some(NodeName("root_level_operations".to_owned())),
            "and it carries the reference's own name for it"
        );
        assert_eq!(
            tree.with(|held| held.parent(root)),
            Some(head),
            "`root_level_operations` hangs off `scheduleTree_.getHeadMutable()`"
        );
        assert_eq!(
            tree.with(|held| held.parent(condition)),
            Some(root),
            "the condition hangs off the block the dataflow region was entered with"
        );
        assert_eq!(
            regions,
            vec![
                NodeName("condition_region0".to_owned()),
                NodeName("condition_region1".to_owned()),
            ],
            "the THEN and ELSE region blocks reached the LIVE tree, under the condition"
        );
        // ⛔ AND THE REGIONS ARE RECORDED BY THE LIVE IDENTITY, not by a name that 570 `ddl.if`s share.
        assert_eq!(
            interface.region2blocks.get(&RegionId(0)),
            Some(&root),
            "region2blocks_ holds the block the region was opened with, by identity"
        );
        assert_eq!(
            tree.with(|held| held
                .children(condition)
                .into_iter()
                .filter_map(|node| interface
                    .region2blocks
                    .iter()
                    .find_map(|(region, held)| (*held == node).then_some(*region)))
                .collect::<Vec<_>>()),
            vec![RegionId(1), RegionId(2)],
            "and each arm's region names its OWN minted block"
        );
    }

    /// ⭐⭐⭐ WHAT STOPS THE DDL EXPANSION ON THE VENDOR'S OWN `broadcast_ops.ddl` — AND IT IS NOT A
    /// CARRIER. `metadata_.core_dstgid` and `metadata_.chunk_dstgid` are `const int` MEMBERS of the
    /// reference's `Metadata`, fixed at `0` and `1` (`ddc/ddc_metadata.h:210-211`), and this crate
    /// already carries them as [`Metadata::CORE_DSTGID`]/[`Metadata::CHUNK_DSTGID`]. But
    /// [`DdlConversion`] states them as its OWN `Option<DatastageId>` fields
    /// (`ddl/conversion.rs:2014-2016`) and **nothing in the crate ever writes either one** — grepped:
    /// the only mentions outside `conversion.rs` are none at all.
    ///
    /// ⛔⛔ SO `op_get_external_datastage` REFUSES ON EVERY TEMPLATE, and it is the **first op of
    /// `broadcast_ops.ddl`'s `ddl.dataflow`** (`MODULES`' `BroadcastOps` region 1 opens with two
    /// `ddl.get_external_datastage` statements and then two `ddl.datastage`s). That is why the whole
    /// expansion mints exactly ONE node — `root_level_operations` — and stops: the very next statement
    /// asks for the core datastage id and gets [`None`].
    ///
    /// ⭐ THE TEST CARRIES BOTH ARMS, so it is a value and not an observation: the same walk over the
    /// same template and the same DSC is run twice, and the ONLY difference is the two ids. Setting
    /// them is what moves the census.
    ///
    /// ⚠️ NOT ESTABLISHED: that the second arm's node count is the reference's. This DSC pins nothing
    /// and states no compute, so `g0/debug/sdsc_N/sdsc.json` has no counterpart to diff against —
    /// what is measured is which STATEMENT the walk reaches, not that its output is right. The fix
    /// itself belongs in `ddl/conversion.rs`, which this file's agent does not own.
    #[test]
    fn the_ddl_dataflow_walk_stops_on_the_unset_core_and_chunk_datastage_ids() {
        use crate::generated::Template;
        use crate::schedule::ddl::conversion::DdlTemplateSet;
        use crate::schedule::ddl::templates::DdlTemplates;

        /// `dim_association_` AS `matchDdl2Dsc` LEAVES IT — the six `ddl.dimension` results
        /// `broadcast_ops.ddl`'s loops name, each bound to a dim. That map is filled by the MATCH
        /// (`ddl_conversion.cpp:2110`), which these walks do not run, so seeding it is how the loop's
        /// own refusal is told apart from the datastage one.
        fn seed_dim_association(interface: &mut DdlInterface) {
            for (at, dim) in [
                PrimaryDim::Mb,
                PrimaryDim::Y,
                PrimaryDim::X,
                PrimaryDim::In,
                PrimaryDim::Out,
                PrimaryDim::I,
            ]
            .into_iter()
            .enumerate()
            {
                let prop = interface
                    .dim_association
                    .entry(NameId(u16::try_from(at).expect("six dims")))
                    .or_default();
                prop.dim = Some(dim);
            }
        }

        /// One run of the vendored `broadcast_ops.ddl` dataflow over a fresh tree, with the two ids
        /// as given — the node count the LIVE tree ends with, and whether the walk refused.
        fn walk(
            ids: Option<(
                crate::schedule::ddc::metadata::DatastageId,
                crate::schedule::ddc::metadata::DatastageId,
            )>,
            dims: bool,
        ) -> (usize, bool) {
            let sdsc = SuperDsc::new(
                DscList::new(a_bare_dsc(), Vec::new()),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
            );
            let l3_state = DscState::seeded(&sdsc);
            let state2 = Dsc2State::seeded(
                &sdsc,
                &l3_state,
                &[Vec::new()],
                &[v1::StorageName("bare".to_owned())],
            );
            let mut site = Dsc2Ddl::new(&state2, DscIdx(0));
            let templates = DdlTemplates::new();
            let stated = DdlTemplateSet::stated(&templates, Template::BroadcastOps)
                .expect("`broadcast_ops.ddl` is one of the 32 vendored modules");
            let mut metadata = Metadata {
                below_lx_schedule_insert_block: BlockId::of(&OneBlock, NodeId(1)),
                ..Metadata::default()
            };
            let mut interface = DdlInterface::default();
            let mut dsc = a_bare_dsc();
            let mut conversion = DdlConversion::new();
            // ⭐ BOTH ARMS ARE NOW EXPLICIT, because `DdlConversion::new` no longer leaves these
            // [`None`]: it states the authority's own constants (`ddc_metadata.h:210-211`), which is
            // what this test's second arm proved was the fix. So the refusal arm has to WRITE `None`
            // rather than rely on the default — otherwise it silently stops testing the refusal and
            // becomes a second copy of the other arm.
            match ids {
                Some((core, chunk)) => {
                    conversion.core_datastage = Some(core);
                    conversion.chunk_datastage = Some(chunk);
                }
                None => {
                    conversion.core_datastage = None;
                    conversion.chunk_datastage = None;
                }
            }
            if dims {
                seed_dim_association(&mut interface);
            }
            let said = parse_ddl2_dsc(
                stated.program,
                &mut conversion,
                &mut interface,
                &mut metadata,
                &mut dsc,
                &mut site,
                &stated.root,
            );
            (l3_state.node_count(), said.is_none())
        }

        // ⛔ THE STATE AS IT SHIPS: both ids [`None`], so the first `ddl.get_external_datastage` of
        // the dataflow refuses and the tree keeps only `root_level_operations`.
        let (unset_nodes, unset_refused) = walk(None, true);
        assert!(
            unset_refused,
            "the walk refuses with the ids unset — `op_get_external_datastage` reads \
             `ctx.state.core_datastage?`"
        );
        assert_eq!(
            unset_nodes, 2,
            "the seeded root block plus `root_level_operations`, and NOTHING the dataflow region \
             states — which is the 22-to-23 delta stage 2b measures on the real `rmsq_o728` tree"
        );

        // ⛔⛔ AND THE COUNT ALONE CANNOT SHOW THE FIX, WHICH IS WHY THIS TEST WALKS THE OPS. The
        // first node-MINTING op of `broadcast_ops.ddl`'s dataflow is its `ddl.loop`, and everything
        // before it — two `ddl.get_external_datastage`, two `ddl.datastage` and five `ddl.if` — mints
        // nothing: a resolved `ddl.if` descends one region and creates no `CONDITION`. So both walks
        // above end at 2 nodes and the DIFFERENCE is *which statement they die on*.
        let (set_nodes, set_refused) =
            walk(Some((Metadata::CORE_DSTGID, Metadata::CHUNK_DSTGID)), true);
        assert_eq!(
            set_nodes, 3,
            "with the two ids set the RECURSIVE walk reaches and mints the `ddl.loop` — \
             `root_level_operations` plus one LOOP on top of the seeded root block, against \
             {unset_nodes} without them"
        );
        // ⚠️ IT STILL REFUSES BELOW THAT LOOP, and this test does not claim otherwise — this DSC
        // states no compute op and pins nothing, so the nested regions have no operand to resolve.
        // ⛔ AND 3 IS NOT THE REFERENCE'S COUNT FOR ANYTHING: this fixture has no
        // `g0/debug/sdsc_N/sdsc.json` counterpart. What is pinned is WHICH STATEMENT each walk dies
        // on, and that the two ids move it from the first one to the tenth.
        assert!(
            set_refused,
            "and it still refuses further down on a DSC this bare — see this assertion's own note"
        );

        // ⭐⭐ THE TRAIL, OP BY OP, WITH THE IDS SET — `process_op` per row of region 1.
        let sdsc = SuperDsc::new(
            DscList::new(a_bare_dsc(), Vec::new()),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let l3_state = DscState::seeded(&sdsc);
        let state2 = Dsc2State::seeded(
            &sdsc,
            &l3_state,
            &[Vec::new()],
            &[v1::StorageName("bare".to_owned())],
        );
        let mut site = Dsc2Ddl::new(&state2, DscIdx(0));
        let templates = DdlTemplates::new();
        let stated = DdlTemplateSet::stated(&templates, Template::BroadcastOps)
            .expect("`broadcast_ops.ddl` is one of the 32 vendored modules");
        let head = l3_state
            .dsc(DscIdx(0))
            .expect("the one DSC's tree")
            .with(|tree| tree.head())
            .expect("a seeded tree has a head");
        let region = stated
            .root
            .regions
            .ops
            .get(&RegionId(1))
            .expect("`ddl.dataflow`'s own region");

        /// How far down `region` `process_op` gets before one of them answers [`None`], with the two
        /// datastage ids as given.
        fn reached(
            stated: &crate::schedule::ddl::conversion::StatedTemplate<
                '_,
                crate::schedule::ddl::templates::TemplateSource,
            >,
            region: &[RegionOp],
            site: &mut Dsc2Ddl<'_, '_>,
            head: NodeId,
            ids: bool,
            dims: bool,
        ) -> usize {
            let mut metadata = Metadata::default();
            let mut interface = DdlInterface::default();
            let mut dsc = a_bare_dsc();
            let mut conversion = DdlConversion::new();
            // ⭐ BOTH ARMS EXPLICIT — `DdlConversion::new` now states the authority's constants, so the
            // refusal arm must WRITE `None` instead of relying on the default. Relying on it would make
            // this arm silently agree with the other one and stop testing the refusal at all.
            if ids {
                conversion.core_datastage = Some(Metadata::CORE_DSTGID);
                conversion.chunk_datastage = Some(Metadata::CHUNK_DSTGID);
            } else {
                conversion.core_datastage = None;
                conversion.chunk_datastage = None;
            }
            if dims {
                seed_dim_association(&mut interface);
            }
            for (at, held) in region.iter().enumerate() {
                if crate::schedule::ddl::conversion::process_op(
                    stated.program,
                    &mut conversion,
                    &mut interface,
                    &mut metadata,
                    &mut dsc,
                    site,
                    &held.op,
                    head,
                )
                .is_none()
                {
                    return at;
                }
            }
            region.len()
        }

        // ⛔⛔ WITH THE IDS UNSET THE WALK DIES ON OP 0, WHICH IS `ddl.get_external_datastage`.
        assert_eq!(
            reached(&stated, region, &mut site, head, false, true),
            0,
            "the FIRST op of the vendored dataflow refuses — `op_get_external_datastage` reads \
             `ctx.state.core_datastage?` and nothing in the crate ever writes it"
        );
        assert!(
            matches!(
                region.first().map(|held| &held.op),
                Some(DdlOp::GetExternalDatastage(_))
            ),
            "and that op IS a `ddl.get_external_datastage`, so the refusal is that read and not \
             some other arm's: {:?}",
            region.first().map(|held| &held.op)
        );

        // ⭐⭐ WITH THEM SET IT WALKS NINE OPS AND STOPS AT THE `ddl.loop` — the first op that would
        // mint a node.
        let with_ids = reached(&stated, region, &mut site, head, true, false);
        assert_eq!(
            with_ids, 9,
            "both `ddl.get_external_datastage`s, both `ddl.datastage`s and all five `ddl.if`s answer"
        );
        assert!(
            matches!(region.get(with_ids).map(|held| &held.op), Some(DdlOp::Loop(_))),
            "and op 9 is the `ddl.loop`: {:?}",
            region.get(with_ids).map(|held| &held.op)
        );

        // ⛔ THAT LAST STOP IS **THIS TEST'S** OWN MISSING `matchDdl2Dsc` AND NOT A CARRIER GAP, and
        // this is the assertion that proves it rather than asserting it: `op_loop` reads
        // `dim_association.get(&dim)?` for each of the six `ddl.dimension` operands
        // (`ddl/conversion.rs:3057`), a map the MATCH fills; seed it and the loop answers too.
        assert_eq!(
            reached(&stated, region, &mut site, head, true, true),
            region.len(),
            "with `dim_association` seeded as the match leaves it, every op of the dataflow region \
             answers — so the datastage ids are the only gap this walk has"
        );
        assert!(
            state2.refusals().is_empty(),
            "and no carrier refused at any point: {:?}",
            state2.refusals()
        );
    }

    /// A tree that calls every node a BLOCK — what `BlockId::of` needs to accept an identity that is
    /// not yet in the live tree.
    struct OneBlock;

    impl crate::schedule::ddc::fold::ScheduleTree for OneBlock {
        fn kind(&self, _node: NodeId) -> NodeKind {
            NodeKind::Block
        }
        fn parent(&self, _node: NodeId) -> Option<NodeId> {
            None
        }
        fn children(&self, _block: BlockId) -> Vec<NodeId> {
            Vec::new()
        }
    }
}
