// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE THREE SURFACES ENTRY 333 READS — EACH ONE A PROJECTION OF THE LIVE `currDsc` AND ITS LIVE
//! `scheduleTree_`, not a stub.
//!
//! `fill_loop_offsets_and_addresses` (`e333`) reaches its inputs through [`DscOffsetFacts`]'s three
//! getters, and each of the three answers with one of the types below: the datastage extents and the
//! address-granularity table ([`v1::StageSizes`] + [`v1::OffsetSizes`]), the whole
//! `dscs_.at(dsc).scheduleTree_` walk ([`v1::ScheduleWalk`] + [`v1::ScheduleNodes`]), and the
//! `dsc2`/`DesignSpaceConfig` seams beside them ([`L3OffsetFacts`]).
//!
//! [`DscOffsetFacts`]: crate::schedule::l3::dl_ops::DscOffsetFacts
//!
//! # ⭐ WHERE EACH SURFACE'S FACTS COME FROM
//!
//! Every one of them is a BORROW and never a snapshot: `dataStageParam_` is REWRITTEN while stage 2a
//! runs (entries 380/351 write `sdsc.dscs_mut()`), the schedule tree is GROWN while it runs (entries
//! 290/353/368), and a copy taken before either would be stale by the time entry 333 reads it. So
//! [`OffsetSizesOf`], [`OffsetNodesOf`] and [`OffsetFactsOf`] hold `&'a DesignSpaceConfig`,
//! `&'a DscTree` and `&'a DscState` — the same objects `run` is holding — and every method below is a
//! read THROUGH that borrow, at the moment it is asked.
//!
//! # ⭐ HOW THE BORROW REACHES THEM, SINCE THE CARRIER MAY NOT HOLD IT
//!
//! `run` takes `sdsc: &mut SuperDsc` while the read carrier arrives as `&'a F`, so [`super::Reads`]
//! may not hold the `&DesignSpaceConfig` these surfaces want — a getter answering `Option<&Self::Sizes>`
//! forced exactly that. So `DscOffsetFacts`' three getters TAKE the super-DSC (entry 382's own shared
//! reborrow, the same one it puts in `L3OffsetInputs`) and hand a surface back BY VALUE; all four
//! surfaces are [`Copy`] projections of that borrow, so the DSC borrow lives at the call site and the
//! carrier holds none.
//!
//! ⛔⛔ ONE CONSTRUCTION ARGUMENT IS STILL MISSING, AND IT IS NOT A BORROW: [`OffsetSizesOf`]'s `ops` is
//! `computeOp_` in the [`v1::DscComputeOp`] shape, which [`super::run_l3`] does not take —
//! [`super::Reads`] holds [`v1::OpFuncs`], and that carries `opFuncName` per entry and nothing else.
//! `DscOffsetFacts::offset_sizes` refuses on it by name, so entry 333 is not entered until it is
//! threaded ([`super::Reads::with_compute_ops`] is the seam it lands on).
//!
//! ⛔ AND NEVER A PLAUSIBLE CONSTANT. Every method here hands back an extent, an address scale, a
//! stride or a page size that a PLACEMENT is computed from, and a fabricated placement is the failure
//! this crate ranks worse than a stop. What is still missing is named on the method that wants it —
//! `getSizeDataStageForNode` (e015/e016), `getBufferCapacityForNodePerDim` (e018),
//! `getBlockTransferSizePerDim`, `addressGranularityScalePerUnit`, `loopDistributionParamInfo`, and
//! the one `dsc2` FIELD our ported nodes drop (`LoopNode::isParametricLoop_`'s two accessors).
//!
//! ⛔⛔ AND ONE OF THEM IS NOT A DROPPED FIELD BUT AN INVENTED ONE. `AllocateNode::paddingSizes_` was
//! listed here as a sixth gap; **there is no such field**. `dsc2::AllocateNode` (`dsc/dsc2.h:974-1011`)
//! carries `PaddingFormType padding_` (`:981`) and nothing else padding-shaped, and `paddingSizes_` is
//! declared ONLY on `DataStructDims` (`dsc/dims.h:219`,
//! `std::map<PrimaryDimTypes, DimPaddingSizes>`) — so EVERY ONE of its 24 reads in `ddc/ddcv1.cpp` is
//! a read of a datastage BY TYPE, whatever the local is called (`coreDs.ss_` at `:580`, `dsChunk` at
//! `:1961-1968`, `dataStageParam_.at(denId_).ss_` at `:2437`, `ds` at `:2512-2674`). That map is
//! [`v1::StageSizes::stage_padding_sizes`], which is answered. So the gap was never a missing borrow
//! or a missing field: [`v1::StageSizes::alloc_padding_sizes`] describes nothing, has zero production
//! callers, and is to be DELETED from the trait in `ddc/v1.rs` rather than answered.

use std::collections::BTreeMap;
use std::num::NonZeroU64;

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{
    AllocId, BlockId, Cardinality, ConstIdx, NodeId, NodeKind, PadType, ScaleBlock, Stride,
    comp_row_id,
};
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind};
use crate::schedule::ddc::transformation::{LoopId, Scale};
use crate::schedule::ddc::transformation_util::PaddingForm;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{ComputeNode, LdsIdx, LdsScale, TransferNode, WordLength};
use crate::schedule::l3::dl_ops::{L3OffsetFacts, VariableSymbol};
use crate::schedule::l3::dsc::{
    DesignSpaceConfig, DimStage, DscIdx, LabeledDs, MemOrgs, PadSizes, StageDims,
};
use crate::units::{Corelet, Row};

use super::state::{DscState, DscTree};
use super::tree::{Kind, Org as TreeOrg, TreeData};

/// `primaryDimToVal_st`'s ANSWER FOR A DIM THE STAGE STATES NOTHING FOR — every `DataStructDims`
/// extent field is initialised `-1` (`dsc/dims.h:162-193`) and `calculate_padded` returns that `-1`
/// straight back out of its `if (val < 0)` short-circuit (`dsc/dims.cpp:566-568`), whatever padding
/// was asked for.
///
/// ⛔ NOT A SUBSTITUTED ZERO AND NOT AN ABORT: the reference's readers COMPARE against `-1`
/// (`L3DlOpsScheduler.cpp:1071-1075`'s `hasPadding` is one), so folding it into `0` would make an
/// unstated dim look like a stated empty one.
const UNSTATED: Extent = Extent(-1);

/// THE DATASTAGE EXTENTS AND THE ADDRESS-GRANULARITY TABLE ENTRY 333 ASKS FOR — one DSC's live
/// `dataStageParam_`, `labeledDs_` and `constantInfo_`, its live `memOrg_`s, and the `computeOp_` the
/// scheduler was constructed with.
#[derive(Debug, Clone, Copy)]
pub struct OffsetSizesOf<'a> {
    /// `mySDsc.dscs_.at(dscIdx)` — `dataStageParam_`, `labeledDs_`, `constantInfo_`.
    dsc: &'a DesignSpaceConfig,
    /// `currDsc->scheduleTree_` and the `memOrg_`s beside it, which is where every allocation this
    /// stage minted is filed.
    tree: &'a DscTree,
    /// `computeOp_` — ⛔ A CONSTRUCTION ARGUMENT AND NOT A DERIVATION, for the reason
    /// [`super::Reads`]' own field states: [`DesignSpaceConfig`] models only that field's
    /// `indirectAccessIndexLabeledDs`.
    ///
    /// ⛔ AND IT IS [`v1::DscComputeOp`] AND NOT [`v1::OpFuncs`], WHICH IS WHAT `Reads` HOLDS TODAY:
    /// [`Self::is_sole_partial_reduction_input`] sweeps `inputLabeledDs`, and `OpFuncs` carries the
    /// `opFuncName` per entry and nothing else. [`super::Dsc2State`] already keeps `computeOp_` in
    /// this shape for stage 2b.
    ops: &'a [v1::DscComputeOp],
}

/// THE DSC'S OWN `scheduleTree_` AS ENTRY 333 WALKS IT.
#[derive(Debug, Clone, Copy)]
pub struct OffsetNodesOf<'a> {
    tree: &'a DscTree,
}

/// THE FOUR `dsc2`/`DesignSpaceConfig` SEAMS AND THE TWO DATA STAGES ENTRY 333 NEEDS BESIDE
/// [`v1::OffsetSizes`].
#[derive(Debug, Clone, Copy)]
pub struct OffsetFactsOf<'a> {
    /// EVERY DSC'S `memOrg_`s, because [`MemOrgs::mem_org`] is keyed by [`DscIdx`] and entry 333 may
    /// ask about a DSC other than its own.
    state: &'a DscState,
    /// `mySDsc.dscs_.at(dscIdx)` — `labeledDs_.wordLength` and `dimToSymbolMapping_`.
    dsc: &'a DesignSpaceConfig,
    /// This DSC's tree, for the ALLOCATE nodes `getPageSize` reads.
    tree: &'a DscTree,
    /// `dataStageParam_.at(dataStageChunkIdx).ss_` — a FIELD because [`L3OffsetFacts::chunk_stage`]
    /// hands back a borrow, and *"Expect chunk data stage."* is discharged by
    /// [`crate::schedule::l3::dsc::DataStages`] holding it as a field of its own. ⭐ IT IS THE LIVE
    /// ONE: the borrow is of the DSC entry 382 is holding, so entries 380/351 rewriting it are seen
    /// here.
    chunk: OffsetStageOf<'a>,
}

/// ONE `DataStructDims` — `dataStageParam_.at(id).ss_`, borrowed live.
#[derive(Debug, Clone, Copy)]
pub struct OffsetStageOf<'a> {
    dims: &'a StageDims,
}

// ⭐ ALL FOUR ARE `Copy` PROJECTIONS OF ONE BORROW, AND THAT IS WHAT LETS THEM BE HANDED BACK BY
// VALUE: `DscOffsetFacts::Sizes<'x>`/`Nodes<'x>`/`Facts<'x>` and `DscStages::Stage<'x>` are these
// types over the super-DSC borrow their getter was ASKED with, so no carrier has to own one.

impl<'a> OffsetSizesOf<'a> {
    /// The extents and tables of ONE DSC, borrowed from the super-DSC entry 382 is holding.
    #[must_use]
    pub const fn new(
        dsc: &'a DesignSpaceConfig,
        tree: &'a DscTree,
        ops: &'a [v1::DscComputeOp],
    ) -> Self {
        Self { dsc, tree, ops }
    }
}

impl<'a> OffsetNodesOf<'a> {
    /// One DSC's schedule tree, borrowed live.
    #[must_use]
    pub const fn new(tree: &'a DscTree) -> Self {
        Self { tree }
    }
}

impl<'a> OffsetFactsOf<'a> {
    /// The seams of ONE DSC, [`None`] for a `dscs_` position the state holds no tree for.
    #[must_use]
    pub fn new(state: &'a DscState, at: DscIdx, dsc: &'a DesignSpaceConfig) -> Option<Self> {
        Some(Self {
            state,
            dsc,
            tree: state.dsc(at)?,
            chunk: OffsetStageOf::new(dsc.data_stages.chunk().ss.dims.dims()),
        })
    }
}

impl<'a> OffsetStageOf<'a> {
    /// One data stage's stick-space dims.
    #[must_use]
    pub const fn new(dims: &'a StageDims) -> Self {
        Self { dims }
    }
}

/// `dataStageParam_.at(stage).ss_` — [`None`] for an index this DSC holds NO stage under (that
/// `.at()`'s throw) AND for one it holds an extent-less stage under, which
/// [`crate::schedule::l3::dsc::EmptyStage`] states no dim at all.
fn steady(dsc: &DesignSpaceConfig, stage: DatastageId) -> Option<&StageDims> {
    Some(dsc.data_stages.at(stage)?.ss.dims.dims())
}

/// `dataStageParam_.count(stage)` WITH NO DIMS IN IT — the one `.at()` that does NOT throw and yet
/// states nothing, so every dim of it reads [`UNSTATED`].
fn extent_less(dsc: &DesignSpaceConfig, stage: DatastageId) -> bool {
    dsc.data_stages.empty_stage(stage).is_some()
}

/// WHETHER `primaryDimToVal_st` FALLS ALL THE WAY THROUGH TO `primaryDimToVal_base_st` for this
/// sample (`dsc/dims.cpp:653-704`) — the row arm needs a row AND `rowSplit_.count(d)`, the PE/SFP arm
/// a vector component AND `peSfpSplit_.count(d)`, and `primaryDimToVal_clView_st` a corelet AND
/// `coreletSplit_.count(d)` (`:635`).
fn falls_through_to_base(dims: &StageDims, dim: PrimaryDim, at: v1::DimSample) -> bool {
    let row = at.row.is_some() && dims.row_split.contains_key(&dim);
    let pe_sfp = at.comp.is_some() && dims.pe_sfp_split.contains_key(&dim);
    let corelet = at.corelet.is_some() && dims.corelet_split.contains_key(&dim);
    !(row || pe_sfp || corelet)
}

/// WHETHER THE STAGE STATES THIS DIM AT ALL — neither a `symbolicDimInfo_` entry (whose `maxSize_`
/// the base read takes instead) nor an extent slot, which is the `-1` [`UNSTATED`] names.
fn states_nothing(dims: &StageDims, dim: PrimaryDim) -> bool {
    !dims.symbolic.info().contains_key(&dim) && dims.extent(dim).is_none()
}

/// `primaryDimToVal_st(d, peOrSfp, ptrowId, clId, padded, dimDensity)` (`dsc/dims.cpp:647-704`) —
/// entry 012 with this padding form and this density, and the reference's `-1` where the base arm
/// reads a slot the stage never wrote.
///
/// ⛔ [`None`] IS THE REFERENCE'S ABORT SET AND NOTHING ELSE: `coreletSplit_.at(clId)` (`:635`),
/// `rowSplit_.at(clId).at(ptrowId)` (`:668`), `peSfpSplit_.at(clId)` (`:686`) and every `DT_ERROR` of
/// `calculate_padded` (`:563-616`).
///
/// ⛔ AND THE `-1` IS ONLY CARRIED AT FULL DENSITY. The reference multiplies its `-1` by the `double`
/// density BEFORE `calculate_padded`, so an unstated dim under a `1/blkSize` density truncates to `0`
/// and comes back PADDED, not `-1`; entry 010 refuses that case (it reads the slot before dividing),
/// so it stays a stop here rather than a guess.
fn dim_val(
    dims: &StageDims,
    dim: PrimaryDim,
    at: v1::DimSample,
    padding: PadType,
    density: v1::Density,
) -> Option<Extent> {
    let mut form = PaddingForm::default();
    form.set_padding(dim, padding);
    // `dimDensity` is `1.0` or `1.0 / mxInfo_.blkSize`, so the block is the divisor and a block of
    // one is the identity the reference's `1.0` multiply is.
    let block = ScaleBlock::of(Cardinality(density.get().get()));
    if let Some(extent) = dims.sampled_extent(dim, at, &form, block, false) {
        return Some(extent);
    }
    (falls_through_to_base(dims, dim, at)
        && states_nothing(dims, dim)
        && density == v1::Density::FULL)
        .then_some(UNSTATED)
}

impl v1::StageSizes for OffsetSizesOf<'_> {
    /// `primaryDimToVal_st(dim, comp, -1, corelet, padding, density)` — entry 012 at `ptrowId = -1`,
    /// so the `rowSplit_` arm is not entered however the unit is spelled.
    fn dim_extent(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
        unit: SenComponent,
        corelet: Option<Corelet>,
        padding: PadType,
        density: v1::Density,
    ) -> Extent {
        let at = v1::DimSample {
            comp: v1::sampled_as(unit),
            row: None,
            corelet,
        };
        if let Some(dims) = steady(self.dsc, stage) {
            if let Some(extent) = dim_val(dims, dim, at, padding, density) {
                return extent;
            }
        } else if extent_less(self.dsc, stage) {
            return UNSTATED;
        }
        todo!(
            "v1::StageSizes::dim_extent: primaryDimToVal_st (dsc/dims.cpp:647-704) STOPS for stage \
             {stage:?} dim {dim:?} unit {unit:?} corelet {corelet:?} padding {padding:?} density \
             {density:?} — either dataStageParam_ holds no entry under that index at all (`.at()`'s \
             throw), or a split does not name that slot (:635, :686), or calculate_padded aborted \
             (:563-616)"
        )
    }

    /// `dataStageDimToVal_compView_st(dim, unit, corelet, padding, density)` (`dsc/dims.cpp:707-714`)
    /// — the SAME entry 012 read with the PT row DERIVED FROM THE COMPONENT
    /// (`EnumsConversion::senCompToRowId`, `sys-arch-spec/arch_enums.cpp:296`), which is entry 082's
    /// own table, so a row-split dim is sampled at that unit's row rather than summed.
    fn comp_view_scaled(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
        unit: SenComponent,
        corelet: Option<Corelet>,
        padding: PadType,
        density: v1::Density,
    ) -> Extent {
        // ⛔ `senCompToRowId` NAMES ROWS 0..7 WHATEVER THE ARCH'S ROW COUNT, and `units::Row` is
        // bounded by it — so an off-arch ordinal is NOT `-1`: `ptrowId >= 0` still holds on the
        // reference side, which enters the row arm and throws at `.at(ptrowId)`. That is the stop
        // below, not a fall-through to the whole core.
        let ordinal = comp_row_id(unit);
        let row = ordinal.and_then(|id| Row::checked(u32::from(id.ordinal())));
        let at = v1::DimSample {
            comp: v1::sampled_as(unit),
            row,
            corelet,
        };
        if let Some(dims) = steady(self.dsc, stage) {
            let off_arch =
                ordinal.is_some() && row.is_none() && dims.row_split.contains_key(&dim);
            if !off_arch
                && let Some(extent) = dim_val(dims, dim, at, padding, density)
            {
                return extent;
            }
        } else if extent_less(self.dsc, stage) {
            return UNSTATED;
        }
        todo!(
            "v1::StageSizes::comp_view_scaled: dataStageDimToVal_compView_st \
             (dsc/dims.cpp:707-714) STOPS for stage {stage:?} dim {dim:?} unit {unit:?} row \
             {ordinal:?} corelet {corelet:?} padding {padding:?} density {density:?} — either \
             dataStageParam_ holds no entry under that index, or rowSplit_/peSfpSplit_/coreletSplit_ \
             does not name that slot (:635, :668, :686), or the row is one this arch does not have, \
             or calculate_padded aborted (:563-616)"
        )
    }

    /// `lds.scale_.at(getDimIndexInLayoutOrder(dsType_, dim))` — ⛔ [`None`] IS
    /// `getDimIndexInLayoutOrder < 0`, a dim this ds type's LAYOUT ORDER does not name, which is the
    /// difference the trait's own doc draws between entry 259's throw and entry 260's `1`.
    fn lds_scale(&self, lds: LdsIdx, dim: PrimaryDim) -> Option<LdsScale> {
        let held = self.dsc.labeled_ds.at(lds)?;
        let named = self
            .dsc
            .layout_dims
            .get(&lds)?
            .iter()
            .any(|walked| walked == dim);
        let scale = named.then(|| held.scale(dim)).flatten()?;
        Some(match scale {
            Scale::StickDim => LdsScale::StickBroadcast,
            Scale::UnitStick => LdsScale::Broadcast,
            Scale::Sized(scale) if scale <= 0.0 => LdsScale::Broadcast,
            Scale::Sized(scale) if (scale - 1.0).abs() < f64::EPSILON => LdsScale::Unscaled,
            Scale::Sized(_) => LdsScale::Scaled,
        })
    }

    /// `1.0 / mxInfo_.blkSize` on a scale tensor's OWN mx dim, [`v1::Density::FULL`] everywhere else.
    fn dim_density(&self, lds: LdsIdx, dim: PrimaryDim) -> v1::Density {
        match self.dsc.labeled_ds.at(lds).and_then(LabeledDs::scale_tensor) {
            Some(held) if held.dim == dim => NonZeroU64::new(held.blk_size.count().0)
                .map_or(v1::Density::FULL, v1::Density::per_block),
            _ => v1::Density::FULL,
        }
    }

    /// `dsNode.coreletSplit_.at(dim)` — how many elements each corelet takes of that dim, read raw.
    fn corelet_split(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Vec<Elements>> {
        steady(self.dsc, stage)?
            .corelet_split
            .get(&dim)
            .map(|shares| {
                shares
                    .iter()
                    .map(|extent| Elements(extent.0.unsigned_abs()))
                    .collect()
            })
    }

    /// ⛔⛔ THE FIELD THIS METHOD NAMES DOES NOT EXIST, AND THE METHOD HAS NO CALLER. Read
    /// `dsc/dsc2.h:974-1011`: `dsc2::AllocateNode` carries `padding_` (a `PaddingFormType`, i.e. one
    /// [`PadType`] per dim) and NOTHING called `paddingSizes_`; `dsc2.h:1000` is
    /// `relatedIndirectAccessAlloc_`. Every `paddingSizes_` read in `ddc/ddcv1.cpp` is on a
    /// `DataStructDims` — `coreDs.ss_` (`:580`), `dsChunk` (`:1961`),
    /// `dataStageParam_.at(loopPtr->denId_).ss_` (`:2437`, `:2512`) — which is a STAGE map and is
    /// [`Self::stage_padding_sizes`]. So the trait's *"a DIFFERENT map from the one above, the
    /// ALLOCATION's"* is a distinction the authority does not draw, and `v1::StageSizes` declares this
    /// method with ZERO production callers (`ddc/v1.rs:1551`; the only other mentions are two
    /// `#[cfg(test)]` doubles).
    ///
    /// ⛔ IT THEREFORE STAYS A STOP ON PURPOSE. Answering it would mean inventing a seam — either a
    /// five-number record no allocate node holds, or a silent re-read of the stage map under a name
    /// that says otherwise. The work is to DELETE the method from [`v1::StageSizes`], which is
    /// `ddc/v1.rs`'s to do.
    fn alloc_padding_sizes(&self, _alloc: AllocId, _dim: PrimaryDim) -> Option<v1::PaddingSizes> {
        todo!(
            "v1::StageSizes::alloc_padding_sizes: NO SUCH FIELD and NO CALLER — dsc2::AllocateNode \
             (dsc/dsc2.h:974-1011) has padding_ (a PaddingFormType) and no paddingSizes_; every \
             paddingSizes_ in ddc/ddcv1.cpp is a DataStructDims (stage) map, which is \
             stage_padding_sizes. Delete this method from v1::StageSizes rather than answer it"
        )
    }

    /// `dataStageParam_.at(stage).ss_.paddingSizes_.at(dim)` (`dsc/dims.h:219`) — converted field for
    /// field from [`crate::schedule::l3::dsc::DimPadding`].
    ///
    /// ⭐⭐ THE **ONLY** `paddingSizes_` MAP THERE IS, which is why [`Self::alloc_padding_sizes`] above
    /// is not "a different map" but no map at all: `paddingSizes_` is declared solely on
    /// `DataStructDims`, so every read of it in `ddc/ddcv1.cpp` is a read of a STAGE by type. ⛔ AND
    /// BOTH READERS READ A STAGE'S — the trait's doc used to draw a stage-vs-allocation split between
    /// entries 259 and 260 and there is none, so no stop here turns on it.
    ///
    /// ⛔ [`None`] ALSO WHERE THE ENTRY NAMES NO WINDOW DIM: [`v1::PaddingSizes::window_dim`] is a bare
    /// [`PrimaryDim`] with no absent state, and the reference's `PrimaryDimTypesCount` IS the absence
    /// (`dsc/dims.cpp:583`), so a substituted dim would name a window that does not exist. ⛔ AND
    /// [`PadSizes::Voided`] IS THE `padFront_ = padBack_ = -1` `calculate_padded` reads as *"Padded
    /// access is not valid in datastage"* (`:582-584`) — an absence here rather than a substituted
    /// zero.
    fn stage_padding_sizes(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
    ) -> Option<v1::PaddingSizes> {
        let held = *steady(self.dsc, stage)?.padding.get(&dim)?;
        let (front, back) = match held.sizes {
            PadSizes::Unpadded => (Elements(0), Elements(0)),
            PadSizes::Sized { front, back } => {
                (Elements(u64::from(front.0)), Elements(u64::from(back.0)))
            }
            PadSizes::Voided => return None,
        };
        Some(v1::PaddingSizes {
            window_dim: held.window_dim?,
            stride: NonZeroU64::new(held.stride.get().unsigned_abs())?,
            dilation: NonZeroU64::new(held.dilation.0.unsigned_abs())?,
            pad_front: front,
            pad_back: back,
        })
    }

    /// ⛔ Wants `getSizeDataStageForNode(node, node)` (`dsc/designSpaceConfig.h:264`) — entries
    /// e015/e016 (`dsc/dsc2.cpp:3611-3752`), which belong to the capacity campaign
    /// ([`crate::schedule::l3::capacity`]) and are NOT ported.
    fn size_stage(&self, _alloc: AllocId) -> DatastageId {
        todo!(
            "v1::StageSizes::size_stage: wants getSizeDataStageForNode(node, node) \
             (dsc/dsc2.cpp:3611-3752, e015/e016 — unported) — WHICH dataStageParam_ entry sizes this \
             allocation; a substituted id would size every buffer off the wrong stage"
        )
    }

    /// SOME compute op whose `opFuncName` is `GENERIC_PARTIAL_REDUCTION` takes exactly this lds as its
    /// only input (`ddc/ddcv1.cpp:1915-1921`) — ⭐ ANSWERED FROM `computeOp_`, which this carrier holds
    /// as its construction argument.
    ///
    /// ⚠️ THE SWEEP'S OWN `DT_CHECK(inputLabeledDs.size() == 1)` IS SWALLOWED, exactly as the trait's
    /// doc says: a `bool` has nowhere to say a partial-reduction op had a different input count.
    fn is_sole_partial_reduction_input(&self, lds: LdsIdx) -> bool {
        self.ops.iter().any(|op| {
            op.op_func == Some(sys_arch_spec::arch_enums::OpFunc::GenericPartialReduction)
                && op.inputs.len() == 1
                && op.inputs.first() == Some(&lds)
        })
    }
}

impl v1::OffsetSizes for OffsetSizesOf<'_> {
    /// `labeledDs_.at(lds).memOrg_.at(storage).allocateNode_` — ⭐ ANSWERED OFF THE TREE'S OWN
    /// `memOrg_`, which is where stage 2a filed every allocation it minted.
    ///
    /// ⛔ [`None`] IS THE `DT_ERROR` *"does not have memOrg_ entry for DataLocation storge"*
    /// (`ddc/ddcv1.cpp:2400-2408`).
    fn lds_alloc(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId> {
        let node = self.tree.org(lds)?.node(storage)?;
        self.tree
            .with(|tree| tree.allocate(node).map(|(alloc, _)| alloc))
    }

    /// `constantInfo_.at(constant).allocations_.at(storage)` — ⛔ [`None`] IS EITHER `.at()`'s THROW,
    /// and the trait's return already is one.
    fn const_alloc(&self, constant: ConstIdx, storage: SenComponent) -> Option<AllocId> {
        self.dsc
            .ddc
            .constants
            .get(&constant)?
            .allocations
            .get(&storage)
            .copied()
    }

    /// ⛔ Wants `dscGlobal.sysDef.addressGranularityScalePerUnit.at({generic, storage})` — a
    /// SYSTEM-DEFINITION table handed to the scheduler's constructor
    /// (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29`), not a super-DSC fact, and THE DIVISOR OF
    /// EVERY PLACED ADDRESS (`DT_CHECK(addrScale > 0)`, `ddc/ddcv1.cpp:2415`).
    fn address_scale(&self, _unit: GenericComp, _storage: SenComponent) -> Option<NonZeroU64> {
        todo!(
            "v1::OffsetSizes::address_scale: wants \
             dscGlobal.sysDef.addressGranularityScalePerUnit.at({{generic, storage}}) — the \
             scheduler's own construction argument, and the DIVISOR of every placed address \
             (DT_CHECK(addrScale > 0), ddc/ddcv1.cpp:2415)"
        )
    }

    /// `dataStageParam_.at(stage).ss_.paddingSizes_` — every padded dim with its sizes, which is what
    /// the window-dim rescue searches (`ddc/ddcv1.cpp:2461-2474`).
    fn stage_padding_dims(&self, stage: DatastageId) -> Vec<(PrimaryDim, v1::PaddingSizes)> {
        let Some(dims) = steady(self.dsc, stage) else {
            return Vec::new();
        };
        dims.padding
            .keys()
            .filter_map(|dim| {
                v1::StageSizes::stage_padding_sizes(self, stage, *dim).map(|sizes| (*dim, sizes))
            })
            .collect()
    }

    /// `dataStageParam_.at(stage).ss_.symbolicDimInfo_.count(dim)`.
    fn has_symbolic_dim(&self, stage: DatastageId, dim: PrimaryDim) -> bool {
        steady(self.dsc, stage).is_some_and(|dims| dims.symbolic.info().contains_key(&dim))
    }

    /// `dataStageParam_.at(stage).ss_.peSfpSplit_` — the dims split between the PE and the SFP.
    fn pe_sfp_split_dims(&self, stage: DatastageId) -> Vec<PrimaryDim> {
        steady(self.dsc, stage)
            .map(|dims| dims.pe_sfp_split.keys().copied().collect())
            .unwrap_or_default()
    }

    /// ⛔ Wants `getBlockTransferSizePerDim(transfer, unit, corelet)[dim]` — a `DesignSpaceConfig`
    /// accessor over that TRANSFER node and the DSC's stick sizes (`dsc/dsc2.cpp:3474`), which is a
    /// `dsc/` seam and is not ported.
    fn block_transfer_size(
        &self,
        _node: NodeId,
        _unit: SenComponent,
        _corelet: Corelet,
        _dim: PrimaryDim,
    ) -> Elements {
        todo!(
            "v1::OffsetSizes::block_transfer_size: wants \
             getBlockTransferSizePerDim(transfer, unit, corelet)[dim] (dsc/dsc2.cpp:3474) — a \
             DesignSpaceConfig accessor over that TRANSFER node and the DSC's stick sizes, unported"
        )
    }

    /// ⛔ Wants `loopDistributionParamInfo.at(node).at(alloc).at(loop).at(dim)`'s
    /// `temporalStridePostDistribution` — the METADATA's fold-distribution table, which entry 292
    /// fills and which no carrier owns; [`crate::schedule::l3::dl_ops::DscMetadata`] models
    /// `newAllocations_` and `externalNodes_` only.
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
             .temporalStridePostDistribution — the metadata's fold-distribution table, which \
             L3DlOpsScheduler.h:111's ported DscMetadata does not model"
        )
    }
}

/// `isNodeRelevant(comp, -1, -1)` (`dsc/dsc2.cpp:1916-1933`) OVER A TREE STAGE 2A ITSELF MINTED.
///
/// ⭐⭐ `ALL` IS THE WHOLE OF IT, AND THAT IS A FOUND FACT RATHER THAN A GUESS. `relevantComps_`
/// (`dsc/dsc2.h:518`) has exactly two fillers: `importJsonObj` (`dsc/dsc2.cpp:1373`), which our seed
/// does not reach — scratchy's wire `scheduleTree_` is a `Vec<AllocNode>` and
/// [`super::DscState::seeded`] mints the head block and the HBM allocations itself — and
/// `DesignSpaceConfig::setRelevantCompCoreCl` (`:2647`), whose ONLY caller is `ddc/ddcv1.cpp:3779`,
/// i.e. stage 2b, which runs AFTER this one (`SchedulerStages.cpp:29-57`). So every node entry 333
/// walks has an EMPTY map, `relevantComps_.find(comp) == end()` for every component, and the
/// reference answers `false` for all of them and `true` only for `ALL`.
///
/// ⛔ IT IS THEREFORE A STATEMENT ABOUT WHEN THIS STAGE RUNS, not about the node: a caller that runs
/// entry 333 over a tree stage 2b has already touched must fill the map first, and this is where that
/// shows.
fn relevant(unit: SenComponent) -> bool {
    unit == SenComponent::All
}

impl v1::ScheduleWalk for OffsetNodesOf<'_> {
    /// `traverseTreeDFS(from, {LOOP})` — every loop at or below that block, in DFS order.
    fn loops_under(&self, from: BlockId) -> Vec<LoopId> {
        self.tree.with(|tree| {
            walk(tree, Some(from.node()), SenComponent::All)
                .into_iter()
                .filter(|node| tree.node_kind(*node) == Some(NodeKind::Loop))
                .map(LoopId)
                .collect()
        })
    }

    /// `traverseTreeDFS(nullptr, {kind})` — every node of one kind in the whole tree.
    fn nodes_of_kind(&self, kind: NodeKind) -> Vec<NodeId> {
        self.tree.with(|tree| {
            walk(tree, None, SenComponent::All)
                .into_iter()
                .filter(|node| tree.node_kind(*node) == Some(kind))
                .collect()
        })
    }

    /// `traverseTreeDFSMutable(nullptr, {ALLOCATE})` — the same for ALLOCATE nodes, whose identity is
    /// the [`AllocId`] the metadata already holds.
    fn allocates(&self) -> Vec<AllocId> {
        self.tree.with(|tree| {
            walk(tree, None, SenComponent::All)
                .into_iter()
                .filter_map(|node| match tree.kind_of(node)? {
                    Kind::Allocate(alloc, _) => Some(*alloc),
                    _ => None,
                })
                .collect()
        })
    }

    /// `traverseTreeDFS(from, {kind}, unit)` — the same below one node, filtered by component, where
    /// [`None`] is the reference's `nullptr` start and so the whole tree.
    ///
    /// ⛔ A COMPONENT OTHER THAN `ALL` VISITS NOTHING, and that is the walk's own control flow rather
    /// than a refusal: `if (!currNode->isNodeRelevant(comp, ..)) continue;` (`dsc/dsc2.cpp:2245`)
    /// skips the node AND its children, and [`relevant`] states why every node's map is empty here.
    fn nodes_of_kind_under(
        &self,
        from: Option<NodeId>,
        kind: NodeKind,
        unit: SenComponent,
    ) -> Vec<NodeId> {
        self.tree.with(|tree| {
            walk(tree, from, unit)
                .into_iter()
                .filter(|node| tree.node_kind(*node) == Some(kind))
                .collect()
        })
    }

    /// `allocNode->getPrev()` — the block, loop or region CONTAINING the ALLOCATE, and [`None`] for an
    /// allocation this tree does not hold.
    fn prev_of_alloc(&self, alloc: AllocId) -> Option<NodeId> {
        self.tree
            .with(|tree| tree.parent(tree.node_of_alloc(alloc)?))
    }

    /// `src_`/`dstVias_` as the node's own pairing, and [`None`] for a node that is not a TRANSFER.
    fn transfer(&self, node: NodeId) -> Option<TransferNode> {
        self.tree.with(|tree| tree.transfer(node))
    }

    /// `nodeType_ == LOOP` as the loop it then is.
    fn as_loop(&self, node: NodeId) -> Option<LoopId> {
        (self.tree.node_kind(node) == Some(NodeKind::Loop)).then_some(LoopId(node))
    }

    /// `getPrev()` — ⚠️ THE PARENT, NOT THE PRECEDING SIBLING (`dsc/dsc2.h:463,515`), absent only at
    /// the root.
    fn prev(&self, node: NodeId) -> Option<NodeId> {
        self.tree.with(|tree| tree.parent(node))
    }

    /// `getOwnerLoop()` — the innermost LOOP enclosing this node.
    fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
        crate::schedule::l3::dl_ops::LoopNesting::owner_loop(self.tree, node)
    }

    /// `LoopNode::dims_` (`dsc/dsc2.h:570`), innermost first — EMPTY for an id that names no loop of
    /// this tree, which is the reference's own null dereference and not an answer.
    fn loop_dims(&self, at: LoopId) -> Vec<(PrimaryDim, MetaDimKind)> {
        self.tree.with(|tree| {
            tree.loop_node(at)
                .map(|held| held.dims.iter().map(|held| (held.dim, held.kind)).collect())
                .unwrap_or_default()
        })
    }
}

impl v1::ScheduleNodes for OffsetNodesOf<'_> {
    /// `traverseTreeDFSMutable()` with no filter — every node, in DFS order.
    fn nodes(&self) -> Vec<NodeId> {
        self.tree.with(|tree| walk(tree, None, SenComponent::All))
    }

    /// `nodeType_`, absent for a node this tree does not hold.
    fn kind(&self, node: NodeId) -> Option<NodeKind> {
        self.tree.node_kind(node)
    }

    /// `isParametricLoop()` (`dsc/dsc2.h:599`) — ⭐ `false` OVER A TREE STAGE 2A ITSELF MINTED, on the
    /// same footing as [`relevant`]: `isParametricLoop_` (`:617`) is written by
    /// `markAsParametricLoop()` at exactly two sites, `ddc/ddl/ddl_conversion.cpp:1128` (the
    /// `ParametricLoopOp` arm of the DDL walk, which is stage 2b) and `importJsonObj`
    /// (`dsc/dsc2.cpp:1410`, for a `parametricLoop_` field on a wire loop node our seed builds none
    /// of).
    ///
    /// ⛔ SO THE FIELD IS ABSENT FROM [`crate::schedule::ddc::transformation_util::LoopNode`] BY
    /// AGREEMENT WITH THE DATA, not by omission: no loop in a stage-2a tree can be parametric. A
    /// caller that runs this walk over a tree the DDL expansion has minted into must add the field
    /// and the two accessors below it — see [`Self::parametric_stride`].
    fn is_parametric(&self, _at: LoopId) -> bool {
        false
    }

    /// `numId_` and `denId_` (`dsc/dsc2.h:573-574`) — TOTAL: every loop node carries both.
    fn loop_stages(&self, at: LoopId) -> v1::LoopStages {
        let stages = self.tree.with(|tree| {
            tree.loop_node(at).map(|held| v1::LoopStages {
                num: held.num,
                den: held.den,
            })
        });
        match stages {
            Some(stages) => stages,
            None => todo!(
                "v1::ScheduleNodes::loop_stages: {at:?} names no LOOP of this DSC's scheduleTree_ — \
                 numId_/denId_ (dsc/dsc2.h:573-574) are TOTAL on a loop node, so an id this tree's \
                 own walk produced cannot miss, and a substituted pair would take every offset's \
                 step from the wrong datastage"
            ),
        }
    }

    /// ⛔ Wants `LoopNode::parametricStride(currDsc)` (`dsc/dsc2.h:606`) — UNREACHABLE while
    /// [`Self::is_parametric`] answers `false`, and unported either way.
    fn parametric_stride(&self, at: LoopId) -> v1::LoopEleOffset {
        todo!(
            "v1::ScheduleNodes::parametric_stride: wants LoopNode::parametricStride(currDsc) \
             (dsc/dsc2.h:606) for {at:?} — unported, and reachable only once isParametricLoop_ \
             (dsc/dsc2.h:617) is carried by the ported loop node"
        )
    }

    /// ⛔ Wants `LoopNode::parametricIterCount(currDsc, clId, comp, -1)` (`dsc/dsc2.h:603`) — the same
    /// pair of reasons as [`Self::parametric_stride`].
    fn parametric_iter_count(
        &self,
        at: LoopId,
        corelet: Corelet,
        unit: SenComponent,
    ) -> v1::IterCount {
        todo!(
            "v1::ScheduleNodes::parametric_iter_count: wants \
             LoopNode::parametricIterCount(currDsc, {corelet:?}, {unit:?}, -1) (dsc/dsc2.h:603) for \
             {at:?} — unported, and reachable only once isParametricLoop_ is carried"
        )
    }

    /// `isNodeRelevant(unit)` (`dsc/dsc2.h:470`) — see [`relevant`] for why `ALL` is the whole answer
    /// over a stage-2a tree.
    fn is_relevant(&self, _node: NodeId, unit: SenComponent) -> bool {
        relevant(unit)
    }

    /// `getNextView(unit).size()` (`dsc/dsc2.cpp:1984-1993`) — how many children of this node that
    /// unit sees, which is the children filtered by `isNodeRelevant(unit)`.
    ///
    /// ⭐ AND THE CONDITION-NODE OVERRIDE IS PORTED (`:1995-2001`): a CONDITION that carries no
    /// `coreClCond_` hands back BOTH children whatever the component was, because it delegates as
    /// `BlockNode::getNextView(ALL)`.
    fn next_view_len(&self, node: NodeId, unit: SenComponent) -> usize {
        self.tree.with(|tree| {
            let loop_cond = matches!(tree.kind_of(node), Some(Kind::Condition(cond)) if cond.cores.is_none());
            if relevant(unit) || loop_cond {
                tree.children(node).len()
            } else {
                0
            }
        })
    }

    /// `allocNode->getOwnerLoop()` — the innermost LOOP that ALLOCATE sits under, [`None`] at the root
    /// and for an allocation this tree does not hold.
    fn alloc_owner_loop(&self, alloc: AllocId) -> Option<LoopId> {
        let node = self.tree.with(|tree| tree.node_of_alloc(alloc))?;
        crate::schedule::l3::dl_ops::LoopNesting::owner_loop(self.tree, node)
    }

    /// `TransferNode::paddingInfo_.isEmpty() == false` (`dsc/dsc2.h:845`), whose `isEmpty()` is
    /// `transferPadFrontSize_.empty() && transferPadBackSize_.empty()` (`:774-776`) — the two maps
    /// [`crate::schedule::dsc2::TransferPadding::dims`] walks.
    fn transfer_has_padding(&self, node: NodeId) -> bool {
        self.tree.with(|tree| {
            tree.transfer(node)
                .is_some_and(|held| held.padding.dims().next().is_some())
        })
    }

    /// `type_`, `exUnit_`, `inputs_` and `outputs_` each zipped with its offsets vector — the whole
    /// COMPUTE node, and [`None`] for a node that is not one.
    fn compute(&self, node: NodeId) -> Option<ComputeNode> {
        self.tree.with(|tree| match tree.kind_of(node)? {
            Kind::Compute(held) => Some(held.clone()),
            _ => None,
        })
    }

    /// `repetitionWithOffset_.forOutputs_.size()` (`dsc/dsc2.h:950-954`) — ⭐ `0` OVER A STAGE-2A
    /// TREE, on the same footing as [`relevant`]: the vector is default-EMPTY and its only writers
    /// are `ddl_conversion.cpp:1405` and `ddc_transformation.cpp:1366`, both stage 2b.
    ///
    /// ⛔ THE FIELD IS ABSENT FROM [`ComputeNode`] BY AGREEMENT WITH THAT, so a caller that runs this
    /// walk over a tree stage 2b has minted into must add it.
    fn repetition_with_offset_outputs(&self, _node: NodeId) -> usize {
        0
    }
}

/// `traverseTreeDFS(startNode, {}, comp)` (`dsc/dsc2.cpp:2222-2265`) — pre-order, parent before
/// children.
///
/// ⛔ THE HEAD IS NOT VISITED. A `nullptr` start (and a start node that IS the head, whose `prev_` is
/// null) begins at the head's CHILDREN (`:2232-2234`); any other start node is itself the first node
/// visited. ⛔ AND AN IRRELEVANT NODE TAKES ITS WHOLE SUBTREE WITH IT (`:2245-2248`), which is a
/// `continue` before the children are pushed.
fn walk(tree: &TreeData, from: Option<NodeId>, comp: SenComponent) -> Vec<NodeId> {
    let head = tree.head();
    let mut stack: Vec<NodeId> = match from {
        Some(node) if Some(node) != head => vec![node],
        _ => {
            let Some(head) = head else {
                return Vec::new();
            };
            tree.children(head).into_iter().rev().collect()
        }
    };
    let mut order = Vec::new();
    while let Some(at) = stack.pop() {
        if !relevant(comp) {
            continue;
        }
        order.push(at);
        stack.extend(tree.children(at).into_iter().rev());
    }
    order
}

impl MemOrgs for OffsetFactsOf<'_> {
    type Org = TreeOrg;

    /// `dscs_.at(dsc).labeledDs_.at(lds).memOrg_`, [`None`] for either `.at()`'s throw.
    fn mem_org(&self, dsc: DscIdx, lds: LdsIdx) -> Option<&Self::Org> {
        self.state.dsc(dsc)?.org(lds)
    }
}

impl<'a> L3OffsetFacts for OffsetFactsOf<'a> {
    type Stage = OffsetStageOf<'a>;

    /// `allocNode->getPageSize()` (`dsc/dsc2.cpp:4480-4512`) — ⭐ DELEGATED TO ENTRY 007, which is the
    /// one home of that dispatch, with the `indirectAllocType_` this allocation's L3 node carries.
    ///
    /// ⭐ EMPTY WHERE NOTHING PAGES, and that is the arm every program we compile takes: the reference
    /// returns BEFORE `layoutDimOrder_` is touched for `NO_INDIRECTION` (`:4483-4485`), so a bounded
    /// layout still pages nothing. ⛔ WHICH IS WHY THIS IS NOT [`MemOrg::hbm_page_sizes`]: that read is
    /// the `maxDimSizes_` bound with no dispatch at all.
    ///
    /// ⛔ AN ALLOCATION THIS TREE DOES NOT HOLD IS EMPTY RATHER THAN A STOP — the reference
    /// dereferences a null pointer there, and entry 333 asks only about allocations its own walk
    /// produced. ⛔ AND A NON-POSITIVE PAGE IS DROPPED, which is [`NonZeroU64`] stating the divisor
    /// guard the page size is read under (`L3DlOpsScheduler.cpp:6156-6164`).
    ///
    /// [`MemOrg::hbm_page_sizes`]: crate::schedule::l3::dsc::MemOrg::hbm_page_sizes
    fn page_sizes(&self, alloc: AllocId) -> BTreeMap<PrimaryDim, NonZeroU64> {
        let held = self.tree.with(|tree| {
            let node = tree.node_of_alloc(alloc)?;
            let (_, minted) = tree.allocate(node)?;
            Some((node, minted.indirect))
        });
        let Some((node, indirect)) = held else {
            return BTreeMap::new();
        };
        if indirect.is_none() {
            return BTreeMap::new();
        }
        match self.tree.placed(node) {
            Some(placed) => placed
                .page_sizes(indirect)
                .into_iter()
                .filter_map(|(dim, elems)| {
                    Some((dim, NonZeroU64::new(u64::try_from(elems.0).ok()?)?))
                })
                .collect(),
            None => todo!(
                "L3OffsetFacts::page_sizes: {alloc:?} indirects through {indirect:?} and \
                 dsc2::AllocateNode::page_sizes (e007, dsc/dsc2.cpp:4486-4511) reads \
                 layoutDimOrder_/maxDimSizes_ off the REFERENCE allocation's ddc node — which this \
                 memOrg_ entry has none of yet"
            ),
        }
    }

    /// ⛔ Wants `getBufferCapacityForNodePerDim(alloc, lds, storage, -1, -1, noRounding=true)`
    /// (`dsc/dsc2.cpp:3966-3975`, e018 — unported, and its ~210-line
    /// `getBufferCapacityForNodePerDimCustomLocation` beneath it) — the index tensor's IBR extent per
    /// dim without rounding up to a whole stick. ⛔ A FABRICATED CAPACITY COMMITS A FABRICATED
    /// PLACEMENT ([`crate::schedule::l3::capacity`]).
    fn ibr_sizes_no_rounding(
        &self,
        _alloc: AllocId,
        _lds: LdsIdx,
        _storage: SenComponent,
    ) -> Option<BTreeMap<PrimaryDim, Elements>> {
        todo!(
            "L3OffsetFacts::ibr_sizes_no_rounding: wants \
             getBufferCapacityForNodePerDim(alloc, lds, storage, -1, -1, noRounding=true) \
             (dsc/dsc2.cpp:3966-3975, e018 — unported, see schedule::l3::capacity)"
        )
    }

    /// `labeledDs_.at(lds).wordLength` (`dsc/dscdefn.h:334`) — ⛔ [`None`] IS THE `.at()`'s THROW
    /// ALONE: the field's own `0` initializer is *"nobody stated a width"* and is a value, which is
    /// why it is not folded into the absence.
    fn word_length(&self, lds: LdsIdx) -> Option<WordLength> {
        Some(self.dsc.labeled_ds.at(lds)?.record().word_length)
    }

    /// `dimToSymbolMapping_.at(dim)`, where the EMPTY vector is `count(dim) == 0` and a skip is not a
    /// refusal.
    fn dim_symbols(&self, dim: PrimaryDim) -> Vec<VariableSymbol> {
        self.dsc
            .ddc
            .dim_to_symbol
            .get(&dim)
            .cloned()
            .unwrap_or_default()
    }

    /// ⛔ Wants `getSizeDataStageForNode(alloc, alloc).ss_` (`dsc/dsc2.cpp:3611-3752`, e015/e016 —
    /// unported): WHICH `dataStageParam_` entry sizes this allocation. The stage itself is reachable
    /// ([`Self::chunk_stage`] is one), so what is missing is the CHOICE and not the data.
    fn size_stage(&self, _alloc: AllocId) -> Option<&Self::Stage> {
        todo!(
            "L3OffsetFacts::size_stage: wants getSizeDataStageForNode(alloc, alloc).ss_ \
             (dsc/dsc2.cpp:3611-3752, e015/e016 — unported) — a substituted stage would size this \
             allocation's corelet offset off the wrong extents"
        )
    }

    /// `dataStageParam_.at(dataStageChunkIdx).ss_` (`L3DlOpsScheduler.cpp:4851`) — ⭐ MANDATORY, AND
    /// *"Expect chunk data stage."* IS DISCHARGED BY [`crate::schedule::l3::dsc::DataStages`], whose
    /// chunk stage is a FIELD rather than a map entry.
    ///
    /// ⭐ READ LIVE, WHICH IS THE WHOLE REASON THIS SURFACE BORROWS: entries 380 and 351 REWRITE this
    /// stage while stage 2a runs, so a copy taken at construction time would be stale here.
    fn chunk_stage(&self) -> &Self::Stage {
        &self.chunk
    }
}

impl DimStage for OffsetStageOf<'_> {
    /// `primaryDimToVal_st(dim, comp, -1, corelet, padded)` (`dsc/dims.cpp:651`) — that corelet's
    /// padded extent along `dim`, [`None`] for any of its `.at()` throws.
    ///
    /// ⛔ NO DENSITY AND NO GRANULARITY HERE, which is the reference's own two defaults: entry 049
    /// passes neither.
    fn corelet_dim_val(
        &self,
        dim: PrimaryDim,
        comp: SenComponent,
        corelet: Corelet,
        padded: &PaddingForm,
    ) -> Option<Extent> {
        let at = v1::DimSample {
            comp: v1::sampled_as(comp),
            row: None,
            corelet: Some(corelet),
        };
        self.dims.sampled_extent(dim, at, padded, None, false)
    }

    /// `coreletSplit_.count(dim)` (`dsc/dims.h:206`).
    fn is_corelet_split(&self, dim: PrimaryDim) -> bool {
        self.dims.corelet_split.contains_key(&dim)
    }

    /// `coreletSplit_.at(dim).at(corelet)` — that corelet's RAW share, unpadded, and [`None`] for
    /// either `.at()`'s throw.
    fn corelet_split(&self, dim: PrimaryDim, corelet: Corelet) -> Option<Extent> {
        self.dims
            .corelet_split
            .get(&dim)?
            .get(usize::try_from(corelet.get()).ok()?)
            .copied()
    }

    /// `paddingSizes_.at(dim).stride_` (`dsc/dims.h:219,140`) — ⛔ [`None`] IS BOTH OF ENTRY 049'S
    /// CHECKS ON IT, `paddingSizes_.count(dim)` and `stride_ > 0`, because a non-positive stride is
    /// not a stride this offset can use.
    fn pad_stride(&self, dim: PrimaryDim) -> Option<Stride> {
        let stride = self.dims.padding.get(&dim)?.stride;
        (stride.get() > 0).then_some(stride)
    }
}

#[cfg(test)]
mod tests {
    //! ⭐ EVERY ASSERTION HERE CARRIES A VALUE OFF THE C++ AND NOT OFF THIS FILE. The two answers
    //! these surfaces DERIVE rather than delegate — `primaryDimToVal_st`'s `-1` for a dim the stage
    //! never wrote, and `isNodeRelevant`'s `ALL`-only `true` over a stage-2a tree — are exactly the
    //! two things measured, each against a construction where a plausible wrong answer (`0`, the
    //! stated slot, "every child") would show.

    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU32;

    use sys_arch_spec::arch_enums::SenComponent;

    use super::{
        AllocId, DesignSpaceConfig, DimStage, Extent, Kind, L3OffsetFacts, NodeKind, PadType,
        PrimaryDim, v1,
    };
    use crate::arch::Elements;
    use crate::schedule::ddc::fold::AllocLayout;
    use crate::schedule::ddc::transformation::DsType;
    use crate::schedule::ddc::transformation_util::{PaddingForm, StageName};
    use crate::schedule::dsc2::{LdsIdx, NodeName, WordLength};
    use crate::schedule::l3::dl_ops::{DscOffsetFacts, DscStages, L3AllocateNode};
    use crate::schedule::l3::dsc::{
        Buffering, CoreIdsUsed, CoreletsUsed, DATA_STAGE_CHUNK, DATA_STAGE_CORE, DataStage,
        DataStages, DdcFacts, DscIdx, DscList, EmptyStage, FilledDims, LabeledDs, LabeledDsList,
        LdsRecord, NamedDims, Pinning, StageDims, SuperDsc,
    };
    use crate::schedule::stages::state::DscState;
    use crate::units::Core;

    /// A DSC whose core and chunk stages state `Y = 2` and NOTHING ELSE, so a dim they do not name is
    /// a dim `primaryDimToVal_st` reads its `-1` initializer for.
    fn a_dsc() -> DesignSpaceConfig {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::Y, Extent(2));
        let named = NamedDims {
            name: StageName::default(),
            dims: FilledDims::of(dims).expect("a stage that states a dim"),
        };
        let stage = DataStage {
            ss: named.clone(),
            el: named,
        };
        let one = CoreletsUsed::new(NonZeroU32::new(1).expect("one corelet"));
        DesignSpaceConfig {
            ddc: DdcFacts::default(),
            corelets_used: one,
            corelets_used_dsc2: Some(one),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(
                Core::checked(0).expect("a core in range"),
                Vec::new(),
            ),
            layout_dims: BTreeMap::new(),
            data_stages: DataStages::new(stage.clone(), stage),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            gtr_ids_used: BTreeSet::new(),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(DsType::Input, vec![], LdsIdx(0), Pinning::default())
                    .with_record(LdsRecord {
                        word_length: WordLength(2),
                        ..LdsRecord::default()
                    }),
                vec![],
            ),
        }
    }

    /// That DSC in a one-DSC super-DSC, so [`DscState::seeded`] holds a tree for it.
    fn a_sdsc() -> SuperDsc {
        SuperDsc::new(
            DscList::new(a_dsc(), Vec::new()),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        )
    }

    /// ⭐⭐ `primaryDimToVal_st` ANSWERS THE FIELD'S OWN `-1` FOR A DIM THE STAGE NEVER WROTE, and the
    /// stated slot for one it did.
    ///
    /// ⛔ THE `-1` IS THE C++'s, NOT OURS: every `DataStructDims` extent is initialised `-1`
    /// (`dsc/dims.h:162-193`) and `calculate_padded` hands that `-1` straight back through
    /// `if (val < 0) return -1;` (`dsc/dims.cpp:566-568`). A `0` here — the plausible wrong answer,
    /// and what an `unwrap_or_default` would give — would make an unstated dim compare equal to a
    /// stated empty one in `hasPadding` (`L3DlOpsScheduler.cpp:1071-1075`).
    #[test]
    fn an_unstated_dim_reads_minus_one_and_a_stated_dim_reads_its_slot() {
        let dsc = a_dsc();
        let tree = super::DscTree::default();
        let sizes = super::OffsetSizesOf::new(&dsc, &tree, &[]);
        let read = |dim| {
            v1::StageSizes::dim_extent(
                &sizes,
                DATA_STAGE_CORE,
                dim,
                SenComponent::NoComponent,
                None,
                PadType::NoPad,
                v1::Density::FULL,
            )
        };
        assert_eq!(read(PrimaryDim::Y), Extent(2), "the slot the stage states");
        assert_eq!(
            read(PrimaryDim::X),
            Extent(-1),
            "dsc/dims.h:162-193's initializer carried out through dsc/dims.cpp:566-568"
        );
    }

    /// ⭐ AN EXTENT-LESS STAGE IS NOT AN ABSENT ONE — `dataStageParam_[id]`'s bare `operator[]` insert
    /// (`ddc/ddl/ddl_conversion.cpp:2585-2591`) leaves an entry whose every dim reads that same `-1`,
    /// so it must not be confused with the `.at()` throw of an index no stage was created for.
    #[test]
    fn an_extent_less_stage_states_minus_one_for_every_dim() {
        let mut dsc = a_dsc();
        dsc.data_stages
            .mint_empty(super::DatastageId(3), EmptyStage::default());
        let tree = super::DscTree::default();
        let sizes = super::OffsetSizesOf::new(&dsc, &tree, &[]);
        assert_eq!(
            v1::StageSizes::dim_extent(
                &sizes,
                super::DatastageId(3),
                PrimaryDim::Y,
                SenComponent::NoComponent,
                None,
                PadType::NoPad,
                v1::Density::FULL,
            ),
            Extent(-1),
            "the entry exists and states nothing, which is not the same as no entry at all"
        );
    }

    /// ⭐⭐ ONLY `ALL` SEES A CHILD, WHICH IS WHAT AN EMPTY `relevantComps_` MEANS — the reference's
    /// `isNodeRelevant` returns `true` for `ALL` and `relevantComps_.find(comp) == end()` for
    /// everything else (`dsc/dsc2.cpp:1918-1924`), and the map is filled by
    /// `setRelevantCompCoreCl` from `ddc/ddcv1.cpp:3779` — stage 2b, AFTER this stage.
    ///
    /// ⛔ THE WRONG ANSWER WOULD BE "EVERY CHILD, WHATEVER THE UNIT": that is what ignoring the filter
    /// gives, and `l3_last_fusable_loop` (`L3DlOpsScheduler.cpp:6283-6295`) would then fuse a transfer
    /// through loops its own unit cannot see.
    #[test]
    fn a_component_other_than_all_sees_no_child_and_the_head_is_never_visited() {
        let held = super::DscTree::default();
        let root = held.with_mut(|tree| {
            let root = tree.add(NodeName("root_level_operations".to_owned()), Kind::Block, None);
            tree.set_head(root);
            tree.add(NodeName("block_a".to_owned()), Kind::Block, Some(root));
            tree.add(NodeName("block_b".to_owned()), Kind::Block, Some(root));
            root
        });
        let nodes = super::OffsetNodesOf::new(&held);
        assert_eq!(
            v1::ScheduleNodes::next_view_len(&nodes, root, SenComponent::All),
            2,
            "getNextView(ALL) filters nothing at all"
        );
        assert_eq!(
            v1::ScheduleNodes::next_view_len(&nodes, root, SenComponent::Lx),
            0,
            "no node of a stage-2a tree names LX in relevantComps_"
        );
        assert_eq!(
            v1::ScheduleNodes::nodes(&nodes).len(),
            2,
            "traverseTreeDFS starts at the head's CHILDREN (dsc/dsc2.cpp:2232-2234)"
        );
        assert!(
            v1::ScheduleWalk::nodes_of_kind_under(&nodes, None, NodeKind::Block, SenComponent::Lx)
                .is_empty(),
            "an irrelevant node takes its whole subtree with it (dsc/dsc2.cpp:2245-2248)"
        );
    }

    /// ⭐⭐ AN ALLOCATION THAT INDIRECTS THROUGH NOTHING PAGES NOTHING — `getPageSize` returns BEFORE
    /// `layoutDimOrder_` is touched for `NO_INDIRECTION` (`dsc/dsc2.cpp:4483-4485`), which is the arm
    /// every program we compile takes.
    ///
    /// ⛔ THE LAYOUT IS BOUNDED ON PURPOSE, so the plausible wrong answer SHOWS: reading
    /// `maxDimSizes_` without the dispatch — which is what
    /// [`MemOrg::hbm_page_sizes`](crate::schedule::l3::dsc::MemOrg::hbm_page_sizes) does — would
    /// answer `{OUT: 64}` here and page an offset nothing pages.
    #[test]
    fn an_allocation_that_indirects_through_nothing_pages_nothing_from_a_bounded_layout() {
        let sdsc = a_sdsc();
        let state = DscState::seeded(&sdsc);
        let dsc = a_dsc();
        let held = state.dsc(DscIdx(0)).expect("the seeded DSC");
        let alloc = held.with_mut(|tree| {
            let alloc = tree.fresh_alloc();
            let node = L3AllocateNode {
                name: NodeName("allocate_lds0_lx".to_owned()),
                lds: LdsIdx(0),
                component: SenComponent::Lx,
                buffering: Buffering::None,
                layout: AllocLayout(vec![(PrimaryDim::Out, Some(Elements(64)))]),
                padding: PaddingForm::default(),
                indirect: None,
                related_indirect: None,
            };
            tree.add(
                NodeName("allocate_lds0_lx".to_owned()),
                Kind::Allocate(alloc, node),
                None,
            );
            alloc
        });
        let facts = super::OffsetFactsOf::new(&state, DscIdx(0), &dsc).expect("a seeded DSC");
        assert_eq!(
            L3OffsetFacts::page_sizes(&facts, alloc),
            BTreeMap::new(),
            "NO_INDIRECTION returns before layoutDimOrder_ is read, so 64 must not appear"
        );
        assert_eq!(
            L3OffsetFacts::page_sizes(&facts, AllocId(999)),
            BTreeMap::new(),
            "an allocation this tree does not hold pages nothing rather than stopping"
        );
    }

    /// ⭐ THE TWO SEAMS THAT ARE PLAIN FIELD READS, read through the facts surface: `wordLength`
    /// (`dsc/dscdefn.h:334`) and `dimToSymbolMapping_`, whose EMPTY vector is `count(dim) == 0` and
    /// not a refusal.
    #[test]
    fn the_word_length_and_the_symbol_mapping_are_the_dscs_own_fields() {
        let sdsc = a_sdsc();
        let state = DscState::seeded(&sdsc);
        let dsc = a_dsc();
        let facts = super::OffsetFactsOf::new(&state, DscIdx(0), &dsc).expect("a seeded DSC");
        assert_eq!(
            L3OffsetFacts::word_length(&facts, LdsIdx(0)),
            Some(WordLength(2))
        );
        assert_eq!(
            L3OffsetFacts::word_length(&facts, LdsIdx(7)),
            None,
            "labeledDs_.at(7) throws, and that throw is the absence"
        );
        assert!(L3OffsetFacts::dim_symbols(&facts, PrimaryDim::Y).is_empty());
        assert_eq!(
            DimStage::corelet_split(L3OffsetFacts::chunk_stage(&facts), PrimaryDim::Y, corelet()),
            None,
            "coreletSplit_ names no dim on this stage, so .at(dim) throws"
        );
        assert!(
            !DimStage::is_corelet_split(L3OffsetFacts::chunk_stage(&facts), PrimaryDim::Y),
            "coreletSplit_.count(Y) == 0"
        );
        assert_eq!(
            dsc.data_stages.at(DATA_STAGE_CHUNK).map(DataStage::name),
            Some(&StageName::default()),
            "the chunk stage the facts surface borrows is this DSC's own"
        );
    }

    /// ⭐⭐⭐ THE SEAM, MEASURED ON THE ONE VALUE A SNAPSHOT CANNOT CARRY. [`DscStages::dim_stage`] used
    /// to answer `state.refuse(..)` because it returned `Option<&Self::Stage>` — a borrow OF THE
    /// CARRIER — while `run` holds the super-DSC as `&mut`, so no `F` may own that borrow. It now TAKES
    /// the super-DSC the caller is already holding and hands a [`Copy`] projection back BY VALUE.
    ///
    /// ⛔⛔ AND WHY A SNAPSHOT WAS NEVER THE ANSWER EITHER: the rewrite below is entries 380/351's own
    /// `dataStageParam_[dataStageChunkIdx] = ..` through `sdsc.dscs_mut()`, made AFTER this carrier
    /// exists. A copy taken at construction would still read `Y = 2` here; the borrow reads `Y = 5`.
    /// ⭐ SO THE TWO ANSWERS DIFFER BY THE REWRITE ALONE, which is the whole property the severing was
    /// recorded for.
    #[test]
    fn the_carrier_reads_the_chunk_stage_a_later_rewrite_left_and_not_the_one_it_was_built_over() {
        let mut sdsc = a_sdsc();
        let state = DscState::seeded(&sdsc);
        let reads = crate::schedule::stages::Reads::new(&state, v1::OpFuncs::new(None, Vec::new()));

        let chunk_y = |held: &SuperDsc| {
            let stage = DscStages::dim_stage(&reads, held, DscIdx(0), DATA_STAGE_CHUNK)
                .expect("the chunk data stage is a FIELD of DataStages");
            DimStage::corelet_dim_val(
                &stage,
                PrimaryDim::Y,
                SenComponent::NoComponent,
                corelet(),
                &PaddingForm::default(),
            )
        };
        assert_eq!(
            chunk_y(&sdsc),
            Some(Extent(2)),
            "the slot this DSC's chunk stage states"
        );

        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::Y, Extent(5));
        let named = NamedDims {
            name: StageName::default(),
            dims: FilledDims::of(dims).expect("a stage that states a dim"),
        };
        sdsc.dscs_mut()
            .at_mut(DscIdx(0))
            .expect("the one DSC")
            .data_stages
            .set(
                DATA_STAGE_CHUNK,
                DataStage {
                    ss: named.clone(),
                    el: named,
                },
            );
        assert_eq!(
            chunk_y(&sdsc),
            Some(Extent(5)),
            "read THROUGH the borrow, so entries 380/351's rewrite is seen"
        );

        assert_eq!(
            DscStages::dim_stage(&reads, &sdsc, DscIdx(0), super::DatastageId(9)).map(|_| ()),
            None,
            "an index this DSC states no stage under is dataStageParam_.at()'s throw"
        );
        assert_eq!(
            DscStages::dim_stage(&reads, &sdsc, DscIdx(3), DATA_STAGE_CORE).map(|_| ()),
            None,
            "and so is a dscs_ position the super-DSC holds no DSC for"
        );
    }

    /// ⭐⭐ THE THREE SURFACES ENTRY 333 IS HANDED, ANSWERED OFF THE SAME BORROW — two of them used to
    /// refuse for the identical reason [`DscStages::dim_stage`] did, and now project the DSC and the
    /// tree the growers filed into.
    ///
    /// ⛔⛔ THE THIRD STILL REFUSES, AND ON A CONSTRUCTION ARGUMENT RATHER THAN A BORROW: `computeOp_`
    /// in the [`v1::DscComputeOp`] shape, which [`crate::schedule::stages::run_l3`] does not take. An
    /// empty sweep would answer `false` for `is_sole_partial_reduction_input`
    /// (`ddc/ddcv1.cpp:1915-1921`) off a `computeOp_` the reference cannot have, and that `false`
    /// cancels a corelet split — so it refuses by name instead.
    #[test]
    fn the_offset_surfaces_answer_off_the_super_dsc_and_only_the_compute_ops_are_missing() {
        let sdsc = a_sdsc();
        let state = DscState::seeded(&sdsc);
        let reads = crate::schedule::stages::Reads::new(&state, v1::OpFuncs::new(None, Vec::new()));

        let nodes = DscOffsetFacts::offset_nodes(&reads, &sdsc, DscIdx(0))
            .expect("the state holds a tree for dscs_[0]");
        assert_eq!(
            v1::ScheduleNodes::nodes(&nodes),
            Vec::new(),
            "the seed's head is never visited (dsc/dsc2.cpp:2232-2234), so an unheaded tree walks to \
             nothing"
        );

        let facts = DscOffsetFacts::offset_facts(&reads, &sdsc, DscIdx(0))
            .expect("the state holds a tree for dscs_[0]");
        assert_eq!(
            L3OffsetFacts::word_length(&facts, LdsIdx(0)),
            Some(WordLength(2)),
            "labeledDs_.at(0).wordLength, read through the projection rather than refused"
        );
        assert_eq!(
            DimStage::corelet_dim_val(
                L3OffsetFacts::chunk_stage(&facts),
                PrimaryDim::Y,
                SenComponent::NoComponent,
                corelet(),
                &PaddingForm::default(),
            ),
            Some(Extent(2)),
            "and *\"Expect chunk data stage.\"* is discharged by the field it borrows"
        );
        assert_eq!(
            DscOffsetFacts::offset_facts(&reads, &sdsc, DscIdx(3)).map(|_| ()),
            None,
            "a dscs_ position with no tree is that .at()'s throw"
        );

        assert!(
            DscOffsetFacts::offset_sizes(&reads, &sdsc, DscIdx(0)).is_none(),
            "computeOp_ in the DscComputeOp shape is not a construction argument of run_l3"
        );
        assert!(
            state
                .refusals()
                .iter()
                .any(|said| said.contains("v1::DscComputeOp")),
            "and the refusal NAMES that one argument rather than the borrow: {:?}",
            state.refusals()
        );
    }

    /// Corelet 0, which is the literal the reference passes.
    fn corelet() -> crate::units::Corelet {
        crate::units::Corelet::checked(0).expect("a corelet in range")
    }
}
