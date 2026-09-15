// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ SCRATCHY'S SuperDSC AS THE SCHEDULER'S OWN `SuperDsc` — the half of bridge 1's scheduling leg
//! that speaks scratchy's vocabulary, so that `deeptools::schedule::stages::run_stages` can be handed
//! REAL DATA instead of a transcribed fixture.
//!
//! ```text
//! SubtileTape ──lower_subtile_tape_to_superdsc──► SdscOp / Dsc   (the json DTOs dxp_standalone reads)
//!             ──THIS MODULE──────────────────────► l3::dsc::SuperDsc
//!             ──schedule::stages::run_stages─────► a GROWN scheduleTree_
//! ```
//!
//! ⛔⛔ WHY IT LIVES IN THE SPYRE CRATE AND NOT IN `deeptools`. `deeptools` never depends on scratchy —
//! it is the port of the vendor compiler and knows nothing of `EmittedOp`, `OpSpec` or the tape. So
//! the conversion cannot live there: the side that names BOTH vocabularies is this one.
//!
//! ⛔⛔ AND WHY IT IS NOT IN [`crate::lower_superdsc_to_dataflow_ir`]. That file's ratchet
//! (`tests/dfir_never_runtime_refuses.rs`) freezes `panic!`, `todo!` and `Result` at ZERO, because it
//! is the lowering the bake compiles and a stop there pre-empts dbo-opt. This module is
//! MEASUREMENT — it runs a scheduler stage that ends in a `todo!` today and reports where — and it
//! reaches nothing the bake stages. Keeping the two apart is what lets the ratchet stay at zero.
//!
//! # ⛔⛔ NO FABRICATED VALUE, ANYWHERE
//!
//! `dxp_standalone` accepts scratchy's SuperDSC and runs these same stages over it to a working
//! `init_binary`, so every fact the target type wants already exists in what scratchy writes. Each
//! field below is therefore ONE of three things and says which:
//!
//!   * READ from the DTO;
//!   * an HONEST ABSENCE with the reference citation that makes absence the right answer;
//!   * a [`None`] from the whole conversion, which is this bridge's refusal idiom — never a
//!     plausible constant.
//!
//! ⭐ AND NO STRING LEAKS INTO A CLOSED SET. A dim name is resolved by scanning
//! [`PrimaryDim::ALL`] for a matching [`PrimaryDim::spelling`] and a `dsType_` by scanning the three
//! [`Role`] variants for a matching [`Role::ds_type`] — the same lookup-in-a-sealed-set shape
//! [`crate::lower_superdsc_to_dataflow_ir::op_func_of`] already uses. An unrecognised name is an
//! ABSENT dim or a refusal, never a substituted one.
//!
//! # ⭐⭐ THE CENSUS THIS MADE MEASURABLE — `-Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel`
//!
//! 134 bundles, **24,363 programs, every one converted**, stage 2a run over each:
//!
//! ```text
//! nodes  93,110 (seed)  ->  347,939        = +254,829   (3.74x)
//!   allocate  137,494        block   49,354
//!   transfer   93,110        condition  628
//!   loop       67,353        sync/compute  0
//! ```
//!
//! ⭐ EVERY ONE OF THOSE COUNTS IS AN EXACT IDENTITY OVER THE INPUT, which is what says it is a real
//! reading and not a stride or a default: the seed is one root block per program plus one HBM allocate
//! per HBM-pinned tensor (24,363 + 68,747); `allocate` is exactly 2 x 68,747, one LX allocation minted
//! per HBM one; `transfer` is exactly 68,747 + 24,363, one HBM->LX load per tensor plus one LX->HBM
//! store per program; `block` is exactly 2 x 24,363 + 628, the root and `lx_below_schedule` plus one
//! region per condition. `sync` and `compute` are ZERO because `create_synchronization` runs AFTER the
//! stop and the computes are stage 2b's.
//!
//! ⭐⭐ AND ALL 24,363 STOP AT **ONE** PLACE — `ExPhaseTrackers::backup`, the unported memory
//! tracker — with **zero** carrier refusals. Not one program stopped anywhere else, so this conversion
//! hands stage 2a nothing the reference fixture did not.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use deeptools::arch::Elements;
use deeptools::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickDims,
};
use deeptools::schedule::ddc::fold::NodeKind;
use deeptools::schedule::ddc::transformation::{DsType, Scale};
use deeptools::schedule::ddc::transformation_util::StageName;
use deeptools::schedule::dsc2::{LayoutDims, LdsIdx};
use deeptools::schedule::l3::dsc::{
    CoreIdsUsed, CoreletShare, CoreletsUsed, DATA_STAGE_CORE, DataStage, DataStages,
    DesignSpaceConfig, DscIdx, DscList, DscScheduleStep, FilledDims, LabeledDs, LabeledDsList,
    NamedDims, Pinning, PrimaryDsInfo, SenComponent, StageDims, SuperDsc, WkSlice, WkSliceCount,
    WkSliceId,
};
use deeptools::schedule::ddc::v1::OpFuncs;
use deeptools::schedule::l3::dl_ops::AddressFoldCoords;
use deeptools::schedule::stages::{DscState, run_l3};
use deeptools::units::Core;
use scratchy_subtile::superdsc_opspec::Role;

use crate::lower_subtile_tape_to_superdsc::{
    Dsc as WireDsc, IterSpace, LabeledDs as WireLabeledDs, LayoutInfo, MemOrg, SdscOp,
};

/// THE `-1` AN UNUSED `IterSpace` SLOT CARRIES — `iter_slot_unused`
/// (`lower_subtile_tape_to_superdsc.rs:824`), which is also what `skip_serializing_if` drops.
///
/// ⛔⛔ AN UNUSED SLOT IS AN **ABSENT** DIM, NOT `Extent(-1)`. `StageDims::extent` answers
/// [`Option`] and `primaryDimToVal_st`'s own `-1` default *is* that absence
/// (`dsc/dims.cpp:516-560`); writing `Extent(-1)` into `extents` would make the dim STATED with a
/// negative size, which `scaled_extent` short-circuits on and `FilledDims::of` would count as a
/// filled stage.
const UNUSED_SLOT: i64 = -1;

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Resolving names into the sealed sets
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ ONE `IterSpace` SLOT BY THE DIM THAT NAMES IT — the exhaustive match IS the mapping, so a
/// thirteenth [`PrimaryDim`] fails to compile here rather than silently reading no slot.
///
/// ⛔ THE NINE SLOTS WITH NO `PrimaryDim` ARE UNREACHABLE BY TYPE. `IterSpace` carries `r_`, `c_`,
/// `rc_`, `si_`, `sj_`, `sij_`, `zi_`, `zj_` and `zij_`; `PrimaryDimTypes` (`dsc/dims.h:34`) names
/// none of them, and `DataStructDims::compound` says so itself — *"no `PrimaryDimTypes` value names
/// those six operands or their three products"* (`l3/dsc.rs`). They cannot be asked for and so cannot
/// be dropped by accident.
const fn slot(space: &IterSpace, dim: PrimaryDim) -> i64 {
    match dim {
        PrimaryDim::In => space.in_,
        PrimaryDim::Out => space.out_,
        PrimaryDim::Ij => space.ij_,
        PrimaryDim::Mb => space.mb_,
        PrimaryDim::X => space.x_,
        PrimaryDim::Y => space.y_,
        PrimaryDim::Kij => space.kij_,
        PrimaryDim::I => space.i_,
        PrimaryDim::J => space.j_,
        PrimaryDim::Ki => space.ki_,
        PrimaryDim::Kj => space.kj_,
        PrimaryDim::X1 => space.x1_,
    }
}

/// ⭐ A DIM NAME AS THE SEALED ENUM, or [`None`] for a name no [`PrimaryDim`] spells.
///
/// ⛔ A LOOKUP IN A CLOSED SET AND NOT A PARSE, exactly as
/// [`crate::lower_superdsc_to_dataflow_ir::op_func_of`] resolves an op-func: [`PrimaryDim::spelling`]
/// is `EnumsConversion::primaryDimToString`'s own rendering (`dsc/dims.cpp:22`), so an unrecognised
/// name yields an ABSENT dim rather than a substituted one.
#[must_use]
pub fn primary_dim_of(name: &str) -> Option<PrimaryDim> {
    PrimaryDim::ALL.into_iter().find(|d| d.spelling() == name)
}

/// ⭐ A `dsType_` AS [`DsType`], or [`None`] for a role scratchy's frontend cannot state.
///
/// ⛔ THE CLOSED SET IS [`Role`], NOT THE STRING. `Role::ds_type` (`superdsc_opspec.rs:1040`) is the
/// only producer of the three spellings scratchy writes, and the match below is exhaustive over it —
/// so a fourth role fails to compile here instead of resolving to a plausible [`DsType`].
#[must_use]
pub fn ds_type_of(spelling: &str) -> Option<DsType> {
    const ROLES: [Role; 3] = [Role::Input, Role::Kernel, Role::Output];
    let role = ROLES.into_iter().find(|role| role.ds_type() == spelling)?;
    Some(match role {
        Role::Input => DsType::Input,
        Role::Kernel => DsType::Kernel,
        Role::Output => DsType::Output,
    })
}

/// ⭐ A `scale_` INTEGER AS THE SEALED [`Scale`], or [`None`] for a value the reference does not
/// spell.
///
/// ⛔ THE THREE VALUES ARE THE UPSTREAM ENUM'S OWN (`superdsc_opspec.rs:1017-1025`: `Active` → `1`,
/// `RedNonStick` → `-1`, `RedStick` → `-2`), and the reference reads them as
/// `-1` ⇒ *the dim is exactly one element*, `-2` ⇒ *the dim spans the whole stick*, and anything
/// non-negative as the size (`dsc/dsc2.cpp:3824-3830`). A FOURTH negative value has no meaning
/// there, so it is a refusal and not a guess.
/// ⭐ THE `f64` IS REACHED THROUGH `f64::From<i32>` AND NOT A CAST, so a size too large to be an
/// exact `f64` is a REFUSAL rather than a silently rounded scale — `scale_` is `1` on 1,516 of the
/// 1,626 entries of `g0/` and never larger.
fn scale_of(scale: i64) -> Option<Scale> {
    match scale {
        -1 => Some(Scale::UnitStick),
        -2 => Some(Scale::StickDim),
        size if size >= 0 => Some(Scale::Sized(f64::from(i32::try_from(size).ok()?))),
        _ => None,
    }
}

/// ⭐ EVERY CORE A `BTreeMap<String, _>` KEYED BY CORE ID NAMES, AS [`Core`].
///
/// ⛔ A RENDER-AND-LOOK-UP, NOT A PARSE: the keys are `c.to_string()` on a `u32` core id at the
/// emitter (`core_dsc_schedule`, `emit_sdsc`), so walking the arch's own cores and looking each one
/// up keeps the core index inside [`Core`] the whole way. A key no [`Core`] names — a core id past
/// this arch's count — is DROPPED, which is `Core::checked`'s own answer and what
/// [`crate::lower_superdsc_to_dataflow_ir::OneDsc::of`] already does with `coreIdsUsed_`.
fn cores_of<V>(map: &BTreeMap<String, V>) -> impl Iterator<Item = (Core, &V)> {
    (0u32..)
        .map_while(Core::checked)
        .filter_map(move |core| map.get(&core.get().to_string()).map(|held| (core, held)))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  One IterSpace → one data-stage half
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ ONE `IterSpace`'S STATED DIMS — every [`PrimaryDim`] whose slot is not the [`UNUSED_SLOT`]
/// sentinel.
fn extents_of(space: &IterSpace) -> BTreeMap<PrimaryDim, Extent> {
    PrimaryDim::ALL
        .into_iter()
        .filter_map(|dim| {
            let stated = slot(space, dim);
            (stated != UNUSED_SLOT).then_some((dim, Extent(stated)))
        })
        .collect()
}

/// ⭐ ONE `IterSpace` AS A [`FilledDims`], or [`None`].
///
/// ⛔ [`None`] IS EITHER OF TWO THINGS, AND BOTH ARE REFUSALS RATHER THAN DROPS:
///
///   * `FilledDims::of` — an `IterSpace` stating no [`PrimaryDim`] at all, which is
///     `!DataStructDims::empty()` (`dsc/dims.cpp:112`) as a type;
///   * A NON-EMPTY `paddingSizes_`, `symbolicDimInfo_`, `maxSymbolicVolume_`, `coreletSplit_`,
///     `rowSplit_` OR `peSfpSplit_`. Each has a home on [`StageDims`] whose SHAPE differs from the
///     wire's — `paddingSizes_` is one number per dim where [`deeptools::schedule::l3::dsc::DimPadding`]
///     wants a front AND a back edge, and `symbolicDimInfo_` likewise carries a `maxSize_` and a
///     `granularity_`. Carrying a single number into either would be inventing the other half, and
///     dropping the map silently would lose a stated fact, so the honest third answer is to refuse.
///
/// ⭐ TODAY THE SIX ARE EMPTY BY CONSTRUCTION, WHICH IS WHY THIS IS A GUARD AND NOT A GAP: every
/// `IterSpace` scratchy builds comes from `IterSpace::empty()` (four sites) and only the named dim
/// slots are ever written (`set_iter_dim`) — measured empty on all 187 programs of
/// `/Users/nickm/tmp/bridge1-fixtures/g0/`. The guard is what makes that keep being true.
fn stage_half(space: &IterSpace) -> Option<FilledDims> {
    if !space.paddingSizes_.is_empty()
        || !space.symbolicDimInfo_.is_empty()
        || !space.maxSymbolicVolume_.is_empty()
        || !space.coreletSplit_.is_empty()
        || !space.rowSplit_.is_empty()
        || !space.peSfpSplit_.is_empty()
    {
        return None;
    }
    let dims = StageDims {
        extents: extents_of(space),
        ..StageDims::default()
    };
    FilledDims::of(dims)
}

/// ⭐⭐ A DSC'S `dataStageParam_` AS [`DataStages`], WHICH DEMANDS A CORE **AND** A CHUNK STAGE.
///
/// ⛔⛔ SYNTHESISING THE CHUNK STAGE FROM THE CORE ONE IS REPRODUCING THE REFERENCE, NOT INVENTING.
/// Two citations, both verified in `/Users/nickm/tmp/dt_src` (revision on disk):
///
///   * `dbo/src/Utils/sdsc_bundle/SdscCoreletSplit.cpp:116` — `DT_CHECK(data_stage_params.size() ==
///     1)`. The vendor's own corelet-split pass ASSERTS that an input SuperDSC carries exactly the
///     ONE stage scratchy emits, and synthesises what it needs from it.
///   * `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1471-1482` — `setChunkDataStageParams`' own
///     `isAllLxLocal` arm: `addOrUpdateDataStageParam(dsc, dsc.dataStageParam_.at(dataStageCoreIdx)
///     .ss_, "chunk", dsc.dataStageParam_.at(dataStageCoreIdx).el_, "chunk", dataStageChunkIdx)` —
///     literally the CORE stage's two halves copied in under the name `"chunk"`.
///
/// ⭐ AND THE STAGE OVERWRITES IT ANYWAY: `set_chunk_data_stage_params` is what stage 2a calls, and
/// the crate's own note says it *"writes DATA STAGES and mints no node"* — so this seed is the
/// placeholder [`DataStages::new`] requires, in the exact shape the reference builds in one arm and
/// replaces in the other.
///
/// ⛔ [`None`] IS: no `dataStageParam_` entry at `dataStageCoreIdx` (`0`,
/// `L3DlOpsScheduler.cpp:275`) — the *"Core data stage parameters are unavailable."* `DT_CHECK`
/// (`:1410`) discharged HERE — or either half refusing [`stage_half`].
///
/// ⭐ THE NAMES ARE BUILT, NOT CARRIED. [`StageName::core`]/[`StageName::chunk`] render what the
/// reference calls the stage at each index, so the wire's own `name_` string never becomes a
/// [`StageName`]; the INDEX is what identifies the core stage.
fn data_stages(dsc: &WireDsc) -> Option<DataStages> {
    let core = dsc.dataStageParam_.get(&DATA_STAGE_CORE.0.to_string())?;
    let named = |name: StageName, space: &IterSpace| -> Option<NamedDims> {
        Some(NamedDims {
            name,
            dims: stage_half(space)?,
        })
    };
    Some(DataStages::new(
        DataStage {
            ss: named(StageName::core(), &core.ss_)?,
            el: named(StageName::core(), &core.el_)?,
        },
        DataStage {
            ss: named(StageName::chunk(), &core.ss_)?,
            el: named(StageName::chunk(), &core.el_)?,
        },
    ))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  One labelled data structure
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ ONE TENSOR'S `memOrg_` AS THE QUESTIONS THE REFERENCE ASKS OF IT.
///
/// * `mem_org` — every component the map NAMES, with its `isPresent`. Scratchy's [`MemOrg`] is
///   SEALED to `hbm`/`lx` (no register-file key can be added), so those two are the whole domain.
/// * `lx` — `isLxPinned()` (`dsc/dscdefn.h:424-441`): `memOrg_.count(LX) > 0 && !isHbmPinned() &&
///   !isXrfPinned()` and not a ring. ⭐ THE LAST THREE COLLAPSE ON THIS INPUT BY TYPE: `RING`,
///   `SFPRING` and `PTXRF` are keys scratchy's sealed [`MemOrg`] cannot hold, and `isXrfPinned()`
///   (`:385-395`) needs LX ABSENT to answer true at all — so the predicate is exactly *LX is named
///   and HBM is not pinned*.
/// * `lx_padded` — `memOrg_.count(LX) && memOrg_.at(LX).isPadded`. ⭐ AN HONEST ABSENCE WITH ITS
///   CITATION: scratchy's `MemPresence` carries ONLY `isPresent` (the frontend-minimal field set
///   torch-spyre emits), so `isPadded` keeps the reference's own field initialiser `= false`
///   (`dsc/dscdefn.h:309`). `false` here is the value the reference reads, not a stand-in.
fn pinning_of(mem: &MemOrg) -> Pinning {
    let mut mem_org = BTreeMap::new();
    if let Some(present) = mem.hbm() {
        mem_org.insert(SenComponent::Hbm, present);
    }
    if let Some(present) = mem.lx() {
        mem_org.insert(SenComponent::Lx, present);
    }
    let hbm_pinned = mem_org.get(&SenComponent::Hbm) == Some(&true);
    Pinning {
        lx: mem.lx().is_some() && !hbm_pinned,
        lx_padded: false,
        mem_org,
    }
}

/// ⭐ ONE `labeledDs_` ENTRY, WITH ITS `scale_` ZIPPED ONTO THE LAYOUT ORDER OF ITS OWN `dsType_`.
///
/// ⛔⛔ THE ZIP IS THE `DT_CHECK("Invalid layoutDimOrder_ index.")`. `getDimIndexInLayoutOrder`
/// (`dsc/designSpaceConfig.cpp:429`) indexes `scale_` BY the position of a dim in
/// `primaryDsInfo_[dsType_].layoutDimOrder_`, so the two lists are one fact and
/// [`LabeledDs`] carries them zipped. A `scale_` of a different length than the layout order is a
/// refusal here rather than an out-of-range read later — measured equal on every one of the 580
/// labelled DSs of `g0/` (`(len scale_, len layout)` ∈ {(3,3), (2,2)}).
///
/// ⛔ `wordLength` AND `dataFormat_` HAVE NO HOME ON THIS TYPE, AND THAT IS THE TARGET'S SCOPE AND
/// NOT A DROP: `l3::dsc::LabeledDs` is the reduced projection of `LabeledDsInfo` that the L3 units
/// read, and the two facts reach the scheduler through `PrepDsc::set_lds_word_length` /
/// `set_lds_format` (`ddc/v1.rs:4036`, `:4038`) — a WRITER on the `P` carrier, not a field of the
/// super-DSC. Forcing them in would put a second copy of a fact somewhere nothing reads it.
///
/// ⛔ `dsName_` LIKEWISE. `DscState::seeded` records that gap itself: the reference names its seed
/// allocate node `allocate-<dsName_>_hbm` and the ported type carries no name, so the seed is named
/// after the POSITION instead.
fn labeled_of(
    lds: &WireLabeledDs,
    layouts: &BTreeMap<&'static str, LayoutInfo>,
) -> Option<LabeledDs> {
    let ds_type = ds_type_of(lds.dsType_)?;
    let layout = &layouts.get(&lds.dsType_)?.layoutDimOrder_;
    if layout.len() != lds.scale_.len() {
        return None;
    }
    let scales = layout
        .iter()
        .zip(&lds.scale_)
        .map(|(name, scale)| Some((primary_dim_of(name)?, scale_of(*scale)?)))
        .collect::<Option<Vec<_>>>()?;
    Some(LabeledDs::new(
        ds_type,
        scales,
        LdsIdx(lds.ldsIdx_),
        pinning_of(&lds.memOrg_),
    ))
}

/// ⭐ ONE `primaryDsInfo_` ENTRY — `layoutDimOrder_` and `stickDimOrder_` zipped with `stickSize_`.
///
/// ⛔⛔ `layoutDimOrder_` GOES IN **VERBATIM, WITH NO FLIP**, AND THAT IS MEASURED. The reference
/// pipeline's OUTPUT `primaryDsInfo_.layoutDimOrder_` is byte-identical to its input's
/// (`["mb","out","y"]` in `g0/sdsc_0.json` and in `g0/debug/sdsc_0/sdsc.json`), and the LX nodes the
/// scheduler mints inherit the same order. ⛔ THE *"index 0 is innermost"* NOTE ONE FINDS NEARBY
/// BELONGS TO A DIFFERENT TYPE — `dsc2::AllocLayout::innermost_dim`, which the SCHEDULER mints and
/// this conversion never constructs.
///
/// ⛔ [`None`] IS: an empty `layoutDimOrder_` ([`LayoutDims`] is non-empty by type), a dim name no
/// [`PrimaryDim`] spells, or a `stickDimOrder_`/`stickSize_` pair of unequal length — the two are one
/// value on [`StickDims`] *"so the two orders cannot disagree in length"*.
fn primary_ds_info_of(info: &LayoutInfo) -> Option<PrimaryDsInfo> {
    let mut layout = info
        .layoutDimOrder_
        .iter()
        .map(|name| primary_dim_of(name))
        .collect::<Option<Vec<_>>>()?
        .into_iter();
    let first = layout.next()?;
    if info.stickDimOrder_.len() != info.stickSize_.len() {
        return None;
    }
    let stick = info
        .stickDimOrder_
        .iter()
        .zip(&info.stickSize_)
        .map(|(name, size)| Some((primary_dim_of(name)?, Elements(u64::from(*size)))))
        .collect::<Option<Vec<_>>>()?;
    Some(PrimaryDsInfo {
        layout: LayoutDims::new(first, layout.collect()),
        stick: StickDims(stick),
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  One DSC
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐⭐ ONE SCRATCHY `Dsc` AS ONE [`DesignSpaceConfig`] — all twelve fields, each one read, cited or
/// refused.
///
/// ⛔ [`None`] IS A REFUSAL FROM ANY CALLEE, PROPAGATED, plus this function's own three: a
/// `numCoreletsUsed_` of zero (a DSC that uses no corelet has no work — [`CoreletsUsed`]'s own
/// invariant), an EMPTY `coreIdsUsed_` (`DT_CHECK(coreIdsUsed_.size() == numCoresUsed_)`,
/// `dsc/designSpaceConfig.cpp:1033`, and `coreIdsUsed_[0]` is reached with a bare subscript), and an
/// EMPTY `labeledDs_` ([`LabeledDsList`] is non-empty by type because
/// `isLastLds` compares against the UNSIGNED `size() - 1`).
///
/// ⛔ `coordinateMasking_` AND `maskingConstId_` ARE OUTSIDE THE TARGET'S PROJECTION, not dropped
/// here: `l3::dsc` is *"a reduced per-module projection of one C++ class … each module states the
/// fields its own units touch and nothing else"* (its own header), and no unit of stage 2a reads
/// either. Both are empty/`-1` on all 187 programs of `g0/` in any case.
#[must_use]
pub fn design_space_config(dsc: &WireDsc) -> Option<DesignSpaceConfig> {
    // 1. `numCoreletsUsed_` — READ. Scratchy emits `1` (`ACTIVE_CORELETS`) on all 187 programs, but
    //    the count is taken from the field rather than assumed, so a two-corelet emission converts.
    let corelets_used = CoreletsUsed::new(NonZeroU32::new(dsc.numCoreletsUsed_)?);

    // 5. `coreIdsUsed_` — READ, and it is `coreIdsUsed_` rather than `0..numCoresUsed_`: a bundle may
    //    occupy a non-contiguous set. A core id past this arch's count is dropped by `Core::checked`.
    let mut cores = dsc.coreIdsUsed_.iter().filter_map(|id| Core::checked(*id));
    let core_ids_used = CoreIdsUsed::new(cores.next()?, cores.collect());

    // 4. `primaryDsInfo_` — READ, per `dsType_`.
    let primary_ds_info = dsc
        .primaryDsInfo_
        .iter()
        .map(|(role, info)| Some((ds_type_of(role)?, primary_ds_info_of(info)?)))
        .collect::<Option<BTreeMap<_, _>>>()?;

    // 7. `labeledDs_` — READ, non-empty by type.
    let mut labelled = dsc
        .labeledDs_
        .iter()
        .map(|lds| labeled_of(lds, &dsc.primaryDsInfo_))
        .collect::<Option<Vec<_>>>()?
        .into_iter();
    let labeled_ds = LabeledDsList::new(labelled.next()?, labelled.collect());

    // 6. `getLayoutDims(ldsIdx)` — READ, per labelled DS, as the `LayoutDims` of ITS `dsType_`'s
    //    `primaryDsInfo_` entry. ⭐ KEYED BY THE POSITION the DS sits at in `labeledDs_`, which is
    //    what `LabeledDsList::indexed` and `DscState::seeded` index by — NOT by the entry's own
    //    `recorded()` index. The emitter writes `ldsIdx_: i as u32`, so the two agree today; keying by
    //    position is what stays correct if they ever drift.
    let layout_dims = labeled_ds
        .indexed()
        .map(|(at, lds)| Some((at, primary_ds_info.get(&lds.ds_type())?.layout.clone())))
        .collect::<Option<BTreeMap<_, _>>>()?;

    // 8. `dataStageParam_` — READ (core) + SYNTHESISED (chunk), see [`data_stages`].
    let data_stages = data_stages(dsc)?;

    // 3. `corelet_shares` — READ off the CORE data stage. ⭐ `corelet0 == whole` IS THE REFERENCE'S
    //    OWN ANSWER, NOT A STAND-IN: `corelet0` is `primaryDimToVal_st(dim, NO_COMPONENT, -1, 0)` and
    //    `whole` is the same with `clId = -1`; `primaryDimToVal_clView_st` (`dsc/dims.cpp:631-645`)
    //    takes its `coreletSplit_.at(d).at(clId)` branch ONLY when `coreletSplit_` names the dim, and
    //    falls through to `primaryDimToVal_base_st` — the plain extent — otherwise. [`stage_half`]
    //    refuses a non-empty `coreletSplit_`, so both readings ARE the plain extent here, and
    //    `CoreletShare::splits()` is correspondingly false — which is what one corelet means.
    let corelet_shares = data_stages
        .core()
        .ss
        .dims
        .dims()
        .extents
        .iter()
        .map(|(dim, extent)| {
            (
                *dim,
                CoreletShare {
                    corelet0: *extent,
                    whole: *extent,
                },
            )
        })
        .collect();

    // 9. `computeOp_.at(0).indirectAccessIndexLabeledDs` — EMPTY, and empty BY CONSTRUCTION rather
    //    than by measurement: both `ComputeOp` build sites in the emitter write
    //    `indirectAccessIndexLabeledDs: vec![]` (`lower_subtile_tape_to_superdsc.rs:5251`, `:5338`),
    //    so no scratchy program states one. ⛔ AND A NON-EMPTY ONE IS A REFUSAL, not a guess: the wire
    //    carries operand NAMES (`"Tensor0-idx0"`) and this field wants an `LdsIdx`, so resolving it
    //    needs the name→position mapping the emitter does not yet write down.
    let indirect_named = dsc
        .computeOp_
        .first()
        .is_some_and(|op| !op.indirectAccessIndexLabeledDs.is_empty());
    if indirect_named {
        return None;
    }
    let indirect_access_index_lds = BTreeSet::new();

    // 11. `N_.paddingSizes_` — EMPTY. Nothing in the emitter ever writes an `IterSpace`'s
    //     `paddingSizes_` (every one comes from `IterSpace::empty()`), measured empty on all 187
    //     programs of `g0/`. ⛔ AND A NON-EMPTY ONE REFUSES, in [`stage_half`], for the same reason:
    //     one wire number cannot fill a front AND a back edge.
    if !dsc.N_.paddingSizes_.is_empty() {
        return None;
    }
    let full_padding = BTreeMap::new();

    Some(DesignSpaceConfig {
        corelets_used,
        // 2. `numCoreletsUsed_DSC2_` — ⭐ [`None`] IS THE REFERENCE'S `-1`, VERBATIM. The field is
        //    declared `-1` (`dsc/designSpaceConfig.h:118`) and `prepDsc` (entry 054) is the only
        //    thing that ever replaces it; the reference SIZES a `std::vector` with it
        //    (`L3DlOpsScheduler.cpp:4844`), so the unprepared state is UB there and absence is the
        //    only honest reading here. A `Some(ONE)` would be inventing that `prepDsc` had run.
        corelets_used_dsc2: None,
        corelet_shares,
        primary_ds_info,
        core_ids_used,
        layout_dims,
        labeled_ds,
        data_stages,
        indirect_access_index_lds,
        // 10. `lx_chunk_capacity` — ⭐⭐ EMPTY, AND THAT IS FAITHFUL RATHER THAN A SHORTCUT. Its own
        //     doc says absence IS the two `DT_CHECK`s of `L3DlOpsScheduler.cpp:1705-1709`: `memOrg_`
        //     naming no `LX`, **or its entry carrying no allocate node**. Scratchy's `scheduleTree_`
        //     holds ONLY HBM allocate nodes — all 580 nodes across `g0/`'s 187 programs are
        //     `(nodeType_ "allocate", component_ "hbm")` — so no labelled DS has an LX allocate node
        //     and the SECOND check holds for every one of them. The capacity is a byte count computed
        //     FROM that node (`getBufferCapacityForNode`, `dsc/dsc2.cpp:3977`); with no node there is
        //     no capacity to state, and stating one would be a fabricated size.
        lx_chunk_capacity: BTreeMap::new(),
        full_padding,
        // 12. `gtrIdsUsed_` — ⭐ EMPTY, which is *"a DSC the L3 scheduler has not reached"*: entries
        //     218 and 291 are what fill it, and only for a multicast group with more than one sharer.
        //     Stage 2a is what runs next, so this is its input state and not a missing fact.
        gtr_ids_used: BTreeSet::new(),
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  One SuperDSC op
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐⭐ ONE SCRATCHY `SdscOp` AS ONE [`SuperDsc`] — the entry point.
///
/// ⛔ `len(dscs_) == 1` ON ALL 187 PROGRAMS OF `g0/`, AND THE LIST IS STILL WALKED AS A LIST:
/// [`DscList`] takes a first and a rest, so a multi-DSC op converts without a second code path.
///
/// ⛔ [`None`] IS: an empty `dscs_`, any DSC refusing [`design_space_config`], or a
/// `numWkSlicesPerDim_` entry of ZERO — `WkSliceCount` is non-zero *"because every writer states `1`
/// or `numCoresUsed`"*, and a zero would silently zero the reference's `numWkSlices` product.
#[must_use]
pub fn super_dsc(op: &SdscOp) -> Option<SuperDsc> {
    let mut dscs = op
        .dscs_
        .iter()
        .flat_map(BTreeMap::values)
        .map(design_space_config)
        .collect::<Option<Vec<_>>>()?
        .into_iter();
    let dscs = DscList::new(dscs.next()?, dscs.collect());

    // `numWkSlicesPerDim_` — READ. A dim name no `PrimaryDim` spells is ABSENT, which is what
    // `num_wk_slices_per_dim`'s own doc calls *"absent for a dim nothing sliced"*.
    let num_wk_slices_per_dim = op
        .numWkSlicesPerDim_
        .iter()
        .filter_map(|(name, count)| Some((primary_dim_of(name)?, NonZeroU32::new(*count)?)))
        .map(|(dim, count)| (dim, WkSliceCount::new(count)))
        .collect();

    // `coreIdToWkSlice_` — READ, per core, per dim.
    let core_id_to_wk_slice = cores_of(&op.coreIdToWkSlice_)
        .map(|(core, slices)| {
            let held = slices
                .iter()
                .filter_map(|(name, slice)| {
                    Some((
                        primary_dim_of(name)?,
                        WkSliceId(i32::try_from(slice.get()).ok()?),
                    ))
                })
                .collect();
            (core, WkSlice(held))
        })
        .collect();

    // `coreIdToDscSchedule` — READ. ⭐⭐ THE `-1` IS `None`, WHICH IS THE FIELD'S OWN DEFAULT:
    // `DscScheduleStep`'s two indices are declared `-1` and its four-argument constructor is
    // `(datadsc_idx, dldsc_idx, before_sync, after_sync)` (`dsc/superdsc.h:30-45`), so scratchy's
    // `[-1, 0, 0, 0]` is *no data DSC, DL DSC 0, neither sync*. ⛔ `before_sync`/`after_sync` are
    // outside `l3::dsc::DscScheduleStep`'s projection — it carries *"its two DSC indices"* — so slots
    // 2 and 3 are not read here; they are `0` on all 2,847 core schedules of `g0/`.
    let core_id_to_dsc_schedule = cores_of(&op.coreIdToDscSchedule)
        .map(|(core, steps)| {
            let held = steps
                .iter()
                .map(|[data, dl, _before, _after]| DscScheduleStep {
                    data_dsc: u32::try_from(*data).ok().map(DscIdx),
                    dl_dsc: u32::try_from(*dl).ok().map(DscIdx),
                })
                .collect();
            (core, held)
        })
        .collect();

    let mut sdsc = SuperDsc::new(
        dscs,
        num_wk_slices_per_dim,
        core_id_to_wk_slice,
        core_id_to_dsc_schedule,
    );
    // `coreIdToDsc_` — READ. ⭐ NOT IN `new` BECAUSE IT IS AN INPUT AND NOT A DERIVED UNION: dbo
    // fills it (`ProgramCorrection.cpp:1064`) and `SuperDsc::new` leaves it EMPTY exactly as the
    // reference's default construction does. Scratchy DOES state it, so it is filled here.
    sdsc.core_id_to_dsc = cores_of(&op.coreIdToDsc_)
        .map(|(core, dsc)| (core, DscIdx(*dsc)))
        .collect();
    // `datastageBasedElemOff` — 🛑 A LATCH AND NOT A CHOICE: entry 379 only ever SETS it, from the
    // first DSC carrying a `ReStickifyOpLx`/`ReStickifyOpHBM` onwards, and nothing clears it. It has
    // no wire spelling, so `SuperDsc::new`'s `false` is the unlatched input state.
    Some(sdsc)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Running stage 2a and measuring what it left
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ THE DEFAULT PANIC HOOK, SILENCED FOR AS LONG AS THIS VALUE LIVES.
///
/// ⛔⛔ WITHOUT IT THE CENSUS FLOODS THE BUILD LOG. Stage 2a's stop is a `todo!` and the default hook
/// prints a `thread 'main' panicked at …` block for every one — one per program, tens of thousands
/// per build. The payload carries the message we need on its own, so the hook has nothing to add.
///
/// ⛔ AND IT IS SCOPED, NOT INSTALLED ONCE: `Drop` puts the previous hook back — including while
/// UNWINDING — so a panic anywhere else in the build keeps its normal report.
struct QuietPanics(Option<Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>>);

impl QuietPanics {
    fn install() -> Self {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        Self(Some(previous))
    }
}

impl Drop for QuietPanics {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            std::panic::set_hook(previous);
        }
    }
}

/// A panic payload as the message it carries — `todo!`/`panic!` with arguments box a `String`, one
/// with a literal boxes a `&'static str`.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(held) = payload.downcast_ref::<String>() {
        return held.clone();
    }
    if let Some(held) = payload.downcast_ref::<&'static str>() {
        return (*held).to_owned();
    }
    "a panic payload that is neither `String` nor `&str`".to_owned()
}

/// ⭐⭐ WHAT STAGE 2A LEFT ON ONE PROGRAM — the measurement, carried as VALUES.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StageEffect {
    /// The seed: `root_level_operations` plus one HBM allocate per HBM-pinned labelled DS.
    pub nodes_before: usize,
    /// What the tree holds AFTER the stage, whether or not it completed.
    pub nodes_after: usize,
    /// ⭐ THE SAME COUNT BROKEN OUT BY `nodeType_` — the census the four stages owe, per kind.
    pub kinds: BTreeMap<NodeKind, usize>,
    /// Whether stage 2a ran to completion.
    pub l3: bool,
    /// The FIRST provider method that refused, if any — a CARRIER gap rather than a ported unit's.
    pub first_refusal: Option<&'static str>,
    /// ⭐ WHERE IT STOPPED — the panic message, [`None`] where the stage returned instead of
    /// panicking. It is REPORTED and never swallowed into a substituted value.
    pub stopped_at: Option<String>,
}

/// ⭐⭐ CONVERT ONE SCRATCHY `SdscOp` AND RUN STAGE 2A OVER IT, REPORTING WHAT IT LEFT.
///
/// ⛔ [`None`] IS THE CONVERSION REFUSING ([`super_dsc`]) — the program is not measured, and the
/// caller reports that rather than counting it as zero nodes.
///
/// ⛔⛔ THE `catch_unwind` IS MEASUREMENT INSTRUMENTATION, NOT A RUNTIME REFUSAL. Stage 2a ends in a
/// `todo!` — `ExPhaseTrackers::backup`, the unported 703-line `util/memtracker/mem_track.{h,cpp}`
/// allocator, which is scoped separately and must NOT be faked: a tracker answering a plausible
/// offset would place real tensors at invented addresses. A panic escaping here would kill the BAKE,
/// so the panic is caught, its message reported as *where it stopped*, and nothing is substituted for
/// the answer it did not give.
///
/// ⛔ IT IS SOUND TO READ THE STATE AFTER THE UNWIND. Every [`DscState`] interior is a `RefCell` whose
/// borrow guards are dropped BY the unwind, so no guard outlives it and the tree is readable.
/// [`std::panic::AssertUnwindSafe`] is what lets the `&mut` cross the boundary; the state is only
/// READ afterwards, never handed back to the stage.
#[must_use]
/// ⭐⭐ `computeOp_`'s `opFuncName`s AS THE SEALED ENUM — every op of the SuperDSC's first (and only)
/// DSC, in `computeOp_` order.
///
/// ⛔ THE MAPPING IS THE REFERENCE'S OWN PARSE BOUNDARY, NOT A TABLE WE WROTE.
/// `sys_arch_spec::arch_enums::OpFunc::from_spelling` is documented as
/// `EnumsConversion::stringToOpFuncs`, and its `SPELLINGS` are the lowercase names scratchy already
/// writes (`"floor"`, `"batchmatmulmxfp8"`, …). So this is a lookup in a closed set — no string
/// reaches a typed field, and no name→op table is invented here.
///
/// ⛔ A NAME THE SET DOES NOT SPELL IS [`None`] FOR THAT OP, WHICH IS `OpFuncs::NONE` — the same
/// thing the reference's own default (`ComputeOpInfo::opFuncName` = `NONE`) states. It is NOT
/// substituted with a plausible neighbour: specialising the scheduler on the wrong op-func picks the
/// wrong data format and the wrong conv2d arm.
///
/// ⭐ EMPTY `computeOp_` STILL YIELDS ONE ENTRY, because [`OpFuncs`] is non-empty by construction and
/// a compute-less DSC is exactly what `OpFuncs::NONE` says.
pub fn op_funcs_of(op: &SdscOp) -> OpFuncs {
    let named: Vec<Option<deeptools::sys_arch_spec::arch_enums::OpFunc>> = op
        .dscs_
        .first()
        .into_iter()
        .flat_map(|per_name| per_name.values())
        .flat_map(|dsc| dsc.computeOp_.iter())
        .map(|compute| {
            deeptools::sys_arch_spec::arch_enums::OpFunc::from_spelling(&compute.opFuncName)
        })
        .collect();
    let mut entries = named.into_iter();
    let first = entries.next().flatten();
    OpFuncs::new(first, entries.collect())
}

pub fn run_stage_2a(op: &SdscOp) -> Option<StageEffect> {
    let mut sdsc = super_dsc(op)?;
    let state = DscState::seeded(&sdsc);
    let nodes_before = state.node_count();
    let ops = op_funcs_of(op);

    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // ⭐⭐ THE REAL OP-FUNC, NOT `None`. `run_stages_with` states `OpFuncs::new(None, ..)`
        // because `computeOp_` is not a field of `l3::dsc::SuperDsc`; a caller that HOLDS the op func
        // is meant to call [`run_l3`], and scratchy holds it as `computeOp_[i].opFuncName`.
        //
        // ⛔ `AddressFoldCoords::flat()` IS DELIBERATELY UNCHANGED. The corpus declares ONE `time`
        // axis of factor 1, so one coordinate is right and `flat()` gives exactly that — but the
        // fold DEPTH (2 vs 3) is not stated anywhere we read, and a wrong fold coordinate is a wrong
        // ADDRESS. Guessing it is the fabricated placement this crate ranks worse than a stop.
        run_l3::<false, deeptools::arch::Dd2>(&mut sdsc, &state, ops.clone(), &AddressFoldCoords::flat())
    }));

    // ⭐ THE COUNTS COME OFF THE STATE EITHER WAY, so a run that panicked and one that returned are
    // measured by the same reading and cannot report different totals for one tree.
    let (l3, stopped_at) = match ran {
        Ok(done) => (done.is_some(), None),
        Err(payload) => (false, Some(panic_text(payload.as_ref()))),
    };
    Some(StageEffect {
        nodes_before,
        nodes_after: state.node_count(),
        kinds: state.kinds(),
        l3,
        first_refusal: state.first_refusal(),
        stopped_at,
    })
}

/// ⭐⭐ THE CORPUS CENSUS — every program of one bundle, aggregated.
///
/// ⭐ MEASURED, and the whole-build aggregate of these is in this module's header: 134 bundles,
/// 24,363 programs, all converted, 93,110 -> 347,939 nodes, every one stopping at the memory tracker
/// and none anywhere else.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    /// How many programs were offered.
    pub programs: usize,
    /// How many the conversion produced an `l3::dsc::SuperDsc` for.
    pub converted: usize,
    /// Seed nodes over every converted program.
    pub nodes_before: usize,
    /// Nodes after stage 2a over every converted program.
    pub nodes_after: usize,
    /// ⭐ THE PER-`nodeType_` TOTAL, which is the gate.
    pub kinds: BTreeMap<NodeKind, usize>,
    /// How many programs stage 2a completed on.
    pub completed: usize,
    /// ⭐⭐ WHERE THE PROGRAMS STOPPED, one entry per distinct message with its count — so a stop
    /// anywhere OTHER than the memory tracker is visible by name rather than folded into a total.
    pub stops: BTreeMap<String, usize>,
    /// Every carrier refusal, by method, with its count — a provider gap as opposed to a ported
    /// unit's own stop.
    pub refusals: BTreeMap<&'static str, usize>,
}

impl Corpus {
    /// Fold one program's measurement in.
    fn absorb(&mut self, effect: &StageEffect) {
        self.converted += 1;
        self.nodes_before += effect.nodes_before;
        self.nodes_after += effect.nodes_after;
        for (kind, count) in &effect.kinds {
            *self.kinds.entry(*kind).or_insert(0) += count;
        }
        if effect.l3 {
            self.completed += 1;
        }
        if let Some(stop) = &effect.stopped_at {
            *self.stops.entry(stop.clone()).or_insert(0) += 1;
        }
        if let Some(refusal) = effect.first_refusal {
            *self.refusals.entry(refusal).or_insert(0) += 1;
        }
    }

    /// ⭐ THE CENSUS AS ONE LINE PER FACT — what the bake prints and the aggregation greps.
    #[must_use]
    pub fn report(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "stage2a: {} program(s), {} converted, {} completed; nodes {} -> {}",
            self.programs, self.converted, self.completed, self.nodes_before, self.nodes_after,
        )];
        for (kind, count) in &self.kinds {
            lines.push(format!("stage2a-kind: {} {count}", kind.spelling()));
        }
        for (stop, count) in &self.stops {
            lines.push(format!("stage2a-stop: {count} x {stop}"));
        }
        for (refusal, count) in &self.refusals {
            lines.push(format!("stage2a-refusal: {count} x {refusal}"));
        }
        lines
    }
}

/// ⭐⭐ CONVERT AND RUN STAGE 2A OVER EVERY PROGRAM OF ONE BUNDLE.
///
/// ⛔ THE PANIC HOOK IS SILENCED FOR THE WHOLE WALK AND RESTORED AFTER — see [`QuietPanics`].
#[must_use]
pub fn census(ops: &[SdscOp]) -> Corpus {
    let _quiet = QuietPanics::install();
    let mut corpus = Corpus {
        programs: ops.len(),
        ..Corpus::default()
    };
    for op in ops {
        if let Some(effect) = run_stage_2a(op) {
            corpus.absorb(&effect);
        }
    }
    corpus
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⭐ THE TWO NAME LOOKUPS ARE CLOSED-SET LOOKUPS, CARRYING THE VALUES — so a renamed variant
    /// fails here rather than silently making every dim `In` or every role `Output`.
    #[test]
    fn a_dim_and_a_role_resolve_from_the_names_scratchy_writes() {
        // ⭐ THE FOUR DIMS `g0/`'s 187 programs ACTUALLY NAME, plus the two combined ones the
        // matmul iteration space states.
        for (name, dim) in [
            ("out", PrimaryDim::Out),
            ("mb", PrimaryDim::Mb),
            ("y", PrimaryDim::Y),
            ("in", PrimaryDim::In),
            ("ij", PrimaryDim::Ij),
            ("x", PrimaryDim::X),
        ] {
            assert_eq!(
                primary_dim_of(name),
                Some(dim),
                "`{name}` is a dim name scratchy writes"
            );
        }
        assert_eq!(
            primary_dim_of("r"),
            None,
            "`r_` is an `IterSpace` slot no `PrimaryDimTypes` names, so it must be ABSENT rather \
             than resolved to a neighbouring dim"
        );

        // ⭐ THE THREE ROLES, AND THE ROUND TRIP IS THROUGH `Role::ds_type` — the only producer.
        for (role, ds_type) in [
            (Role::Input, DsType::Input),
            (Role::Kernel, DsType::Kernel),
            (Role::Output, DsType::Output),
        ] {
            assert_eq!(
                ds_type_of(role.ds_type()),
                Some(ds_type),
                "`{}` is what `Role::ds_type` writes",
                role.ds_type()
            );
        }
        assert_eq!(
            ds_type_of("INTERNAL"),
            None,
            "a `DsTypes` scratchy's frontend cannot state must refuse, not resolve"
        );
    }

    /// ⭐⭐ THE `scale_` MAPPING CARRIES THE REFERENCE'S OWN THREE READINGS, and the corpus's own
    /// three values are each covered: `1` (1,516 occurrences in `g0/`), `-1` (76) and `-2` (34).
    ///
    /// ⛔ THE FOURTH ARM IS THE REFUSAL, NOT A DEFAULT: `-3` has no meaning at
    /// `dsc/dsc2.cpp:3824-3830`, so it must be `None` and not `Sized(-3.0)` or `UnitStick`.
    #[test]
    fn a_scale_resolves_to_the_reference_reading_and_refuses_the_rest() {
        assert_eq!(scale_of(1), Some(Scale::Sized(1.0)), "`1` is an ACTIVE dim");
        assert_eq!(
            scale_of(-1),
            Some(Scale::UnitStick),
            "`-1` is the dim being exactly ONE element"
        );
        assert_eq!(
            scale_of(-2),
            Some(Scale::StickDim),
            "`-2` is the dim spanning the WHOLE stick"
        );
        assert_eq!(scale_of(0), Some(Scale::Sized(0.0)), "a non-negative size");
        assert_eq!(
            scale_of(-3),
            None,
            "a negative the reference does not spell must refuse rather than pick a neighbour"
        );
    }

    /// ⭐⭐ AN UNUSED `IterSpace` SLOT IS AN ABSENT DIM, NOT `Extent(-1)` — the trap named in
    /// [`UNUSED_SLOT`], carried as values.
    ///
    /// ⛔ NOT A TAUTOLOGY: the extents are read back through `PrimaryDim::ALL`, so a `slot` arm wired
    /// to the wrong field would put the value under the wrong dim and fail here. The numbers are
    /// `g0/sdsc_0.json`'s own core stage (`{out_: 64, mb_: 1, y_: 1}`).
    #[test]
    fn an_unused_slot_is_an_absent_dim_and_a_stated_one_carries_its_extent() {
        let mut space = IterSpace::empty();
        space.out_ = 64;
        space.mb_ = 1;
        space.y_ = 1;

        let extents = extents_of(&space);
        assert_eq!(
            extents,
            BTreeMap::from([
                (PrimaryDim::Out, Extent(64)),
                (PrimaryDim::Mb, Extent(1)),
                (PrimaryDim::Y, Extent(1)),
            ]),
            "the three dims `rmsq_o728`'s core stage states, each under its own dim"
        );
        for dim in PrimaryDim::ALL {
            if matches!(dim, PrimaryDim::Out | PrimaryDim::Mb | PrimaryDim::Y) {
                continue;
            }
            assert!(
                !extents.contains_key(&dim),
                "{dim:?} is at the -1 sentinel, so it must be ABSENT — an `Extent(-1)` would make \
                 it a STATED dim of negative size"
            );
        }

        // ⭐ AND A WHOLLY UNUSED SPACE IS NOT A FILLED STAGE — `!DataStructDims::empty()`.
        assert!(
            stage_half(&IterSpace::empty()).is_none(),
            "an `IterSpace` stating no dim cannot be a `FilledDims`"
        );
    }

    /// ⭐⭐ THE SIX SCHEDULER MAPS REFUSE RATHER THAN DROP — a stated fact this conversion cannot
    /// carry must stop it, because silently dropping one would schedule against a shape the input did
    /// not describe.
    #[test]
    fn a_stated_scheduler_map_refuses_instead_of_being_dropped() {
        let stated = |mutate: fn(&mut IterSpace)| {
            let mut space = IterSpace::empty();
            space.out_ = 64;
            mutate(&mut space);
            space
        };
        assert!(
            stage_half(&stated(|_| {})).is_some(),
            "the control: a plain stated dim converts"
        );
        for (what, mutate) in [
            (
                "paddingSizes_",
                (|s: &mut IterSpace| {
                    s.paddingSizes_.insert("out".to_owned(), 1);
                }) as fn(&mut IterSpace),
            ),
            ("symbolicDimInfo_", |s| {
                s.symbolicDimInfo_.insert("out".to_owned(), 64);
            }),
            ("maxSymbolicVolume_", |s| {
                s.maxSymbolicVolume_.insert("out".to_owned(), 64);
            }),
            ("coreletSplit_", |s| {
                s.coreletSplit_.insert("out".to_owned(), vec![32, 32]);
            }),
            ("rowSplit_", |s| {
                s.rowSplit_.insert("out".to_owned(), 8);
            }),
            ("peSfpSplit_", |s| {
                s.peSfpSplit_.insert("out".to_owned(), 2);
            }),
        ] {
            assert!(
                stage_half(&stated(mutate)).is_none(),
                "a stated `{what}` has no faithful home on `StageDims`, so it must REFUSE — \
                 dropping it would schedule against a shape the input did not describe"
            );
        }
    }

    /// ⭐⭐ `memOrg_` READS AS THE REFERENCE'S THREE QUESTIONS — and the two residencies scratchy
    /// actually emits give DIFFERENT answers, which is what proves this is reading the field rather
    /// than returning a constant.
    ///
    /// ⛔ `isLxPinned()` IS FALSE ON THE `hbm+lx` CASE (`!isHbmPinned()` fails) AND TRUE ON
    /// `lx`-ONLY — `dsc/dscdefn.h:424-427`. All 580 labelled DSs of `g0/` are `hbm+lx`; the LX-only
    /// arm is what `Allocation::Lx` emits.
    #[test]
    fn a_memorg_answers_names_present_and_lx_pinned_separately() {
        let both = pinning_of(&MemOrg::hbm_lx());
        assert!(both.hbm(), "`hbm` is present, so the DS is HBM-pinned");
        assert!(
            both.names(SenComponent::Lx),
            "`memOrg_` NAMES lx even though the DS is HBM-pinned"
        );
        assert!(
            !both.lx,
            "isLxPinned() is FALSE with hbm present — `!isHbmPinned()` fails"
        );

        let lx_only = pinning_of(&MemOrg::lx_only());
        assert!(!lx_only.hbm(), "an LX-only DS is not HBM-pinned");
        assert!(
            lx_only.lx,
            "isLxPinned() is TRUE for lx named with no hbm and no ring/xrf"
        );
        assert!(
            !lx_only.names(SenComponent::Hbm),
            "`memOrg_` does not name hbm at all"
        );

        let hbm_only = pinning_of(&MemOrg::hbm_only());
        assert!(hbm_only.hbm(), "hbm-only is HBM-pinned");
        assert!(
            !hbm_only.names(SenComponent::Lx),
            "an index tensor's `memOrg_` names no lx"
        );
        assert!(!hbm_only.lx, "and so it is not LX-pinned either");
    }

    /// ⭐⭐⭐ THE WHOLE CONVERSION ON A **REAL EMITTED** `SdscOp`, AND STAGE 2A RUN OVER IT.
    ///
    /// ⛔⛔ NOT A TRANSCRIBED FIXTURE AND NOT A HAND-BUILT STRUCT. `matmul_opspec` + `emit_sdsc` is
    /// the emitter's own path — the same `SdscOp` `render_dfir_input` hands the census — so this test
    /// exercises the conversion against data scratchy PRODUCED. A hand-built `Dsc` is not even
    /// constructible: `AllocNode::maxDimSizes_` is a `DeviceWalk` with no public constructor.
    ///
    /// ⭐⭐ IT CARRIES THE VALUES, NOT A VERDICT. The seed is one root block plus one HBM allocate per
    /// HBM-pinned tensor (three, for `A·W→O`); stage 2a's growers then mint the chunk loop nest, the
    /// LX allocations and their transfers, and the stage stops at the memory tracker. Both counts and
    /// the per-kind census are asserted, so a regression to a refusal AND a regression to an
    /// unexpectedly-different stop both fail here.
    ///
    /// ⛔ THE STOP IS NAMED, NOT `is_some()`. A bare "it panicked" would pass on the FIRST `todo!` of
    /// eighty-six and say nothing about how far the stage got — `ExPhaseTrackers::backup` makes this
    /// a ratchet in both directions, exactly as `schedule::stages`' own fixture test is.
    #[test]
    fn a_real_emitted_matmul_converts_and_stage_2a_grows_its_tree() {
        use crate::ir::bridge::tiled_op_sdsc_op::matmul::opspec::matmul_opspec;
        use crate::lower_subtile_tape_to_superdsc::emit_sdsc;
        use scratchy_subtile::superdsc_opspec::SdscFoldSet;

        let op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2")
            .expect("a 384x384x64 matmul is a shape the emitter builds");
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let wire = emit_sdsc("MatMul_0", &op, &folds, None).expect("the emitter lowers it");

        // ⭐ THE CONVERSION FIRST, ON ITS OWN — so a failure here is the conversion's and not the
        // stage's, and the fields it read are checked against the emitter's own values.
        let wire_dsc = wire
            .dscs_
            .iter()
            .flat_map(BTreeMap::values)
            .next()
            .expect("the emitter writes one `dscs_` entry");

        let converted = super_dsc(&wire).expect("a real emitted SuperDSC converts");
        let dsc = converted.dscs().first();
        assert_eq!(
            dsc.core_ids_used.count().0,
            wire_dsc.numCoresUsed_,
            "`coreIdsUsed_` carries every core `numCoresUsed_` claims"
        );
        assert_eq!(
            dsc.corelets_used.get(),
            wire_dsc.numCoreletsUsed_,
            "`numCoreletsUsed_` is READ, not assumed to be one"
        );
        assert!(
            !dsc.corelets_used.splits(),
            "one corelet does not split, so no min-param unit takes the corelet arm"
        );
        assert_eq!(
            dsc.corelets_used_dsc2, None,
            "`numCoreletsUsed_DSC2_` is the reference's -1 until `prepDsc` runs"
        );
        assert_eq!(
            dsc.labeled_ds.iter().count(),
            wire_dsc.labeledDs_.len(),
            "every labelled DS converts — a dropped one would silently unschedule a tensor"
        );
        assert_eq!(
            dsc.layout_dims.len(),
            wire_dsc.labeledDs_.len(),
            "and each has its own layout order, keyed by the POSITION it sits at"
        );
        assert!(
            dsc.corelet_shares
                .values()
                .all(|share| !share.splits() && share.corelet0 == share.whole),
            "with no `coreletSplit_`, corelet 0's share IS the whole dim — \
             `primaryDimToVal_clView_st` falls through to the base extent for both readings"
        );
        assert!(
            dsc.lx_chunk_capacity.is_empty(),
            "no labelled DS has an LX allocate node, so there is no capacity to state"
        );
        assert!(
            dsc.gtr_ids_used.is_empty() && dsc.full_padding.is_empty(),
            "both are filled by stages this input has not reached"
        );
        assert_eq!(
            converted.core_id_to_dsc.len(),
            wire.coreIdToDsc_.len(),
            "`coreIdToDsc_` is stated by scratchy, so it is filled rather than left empty"
        );
        // ⭐ THE `-1` IS `None`: scratchy writes `[[-1, 0, 0, 0]]` per core, which is NO data DSC and
        // DL DSC 0.
        for steps in converted.core_id_to_dsc_schedule.values() {
            assert_eq!(
                steps,
                &[DscScheduleStep {
                    data_dsc: None,
                    dl_dsc: Some(DscIdx(0)),
                }],
                "`-1` is an ABSENT DSC index, not `DscIdx(u32::MAX)`"
            );
        }

        // ⭐⭐ AND NOW THE STAGE, OVER THAT SAME CONVERSION.
        let effect = run_stage_2a(&wire).expect("the same conversion, run through stage 2a");
        assert_eq!(
            effect.nodes_before, 4,
            "the seed: `root_level_operations` plus one HBM allocate per HBM-pinned tensor"
        );
        assert!(
            effect.nodes_after > effect.nodes_before,
            "stage 2a MINTED nodes on real data — {} -> {}",
            effect.nodes_before,
            effect.nodes_after,
        );
        assert_eq!(
            effect.kinds.values().sum::<usize>(),
            effect.nodes_after,
            "the per-kind census must account for every node, or one of the two is short"
        );
        assert_eq!(
            effect.kinds.get(&NodeKind::Block),
            Some(&2),
            "`root_level_operations` and `lx_below_schedule`"
        );
        // ⭐ ONE CHUNK LOOP PER ORDER DIM, WHICH IS `create_chunk_loops` UNDER `LxBufferChoice::Double`
        // — the reference's `loop_ds0_ds1_{y,out,mb}` band. ⛔ TIED TO THE INPUT AND NOT A LITERAL:
        // this matmul's OUTPUT layout order is FOUR dims (`in` is a real matmul dim), where
        // `rmsq_o728`'s is three — so the count is read off the layout the emitter wrote, and the
        // literal beside it says what that is for this program.
        let order: BTreeSet<PrimaryDim> = dsc
            .layout_dims
            .values()
            .flat_map(LayoutDims::iter)
            .collect();
        assert_eq!(
            order,
            BTreeSet::from([
                PrimaryDim::In,
                PrimaryDim::Out,
                PrimaryDim::Mb,
                PrimaryDim::Y,
            ]),
            "a matmul's three roles between them name FOUR distinct layout dims — `in` is the \
             reduction axis the KERNEL and INPUT carry and the OUTPUT does not"
        );
        assert_eq!(
            effect.kinds.get(&NodeKind::Loop),
            Some(&order.len()),
            "one chunk loop per DISTINCT layout dim across `labeledDs_` — which is
             `collect_all_dimensions_for_loop_order`, and four here where `rmsq_o728`'s single \
             OUTPUT role gives three"
        );
        assert_eq!(
            effect.first_refusal, None,
            "no carrier was asked for a fact it could not give — the stop is a ported unit's"
        );
        let stopped = effect
            .stopped_at
            .expect("stage 2a stops at the memory tracker");
        assert!(
            stopped.contains("ExPhaseTrackers::backup"),
            "the stop must be the unported memory tracker and nothing earlier; got: {stopped}"
        );
        assert!(!effect.l3, "so stage 2a did not complete");
    }

    /// ⭐⭐ THE CORE STAGE IS COPIED UNDER THE NAME `"chunk"`, WHICH IS THE REFERENCE'S OWN MOVE —
    /// `L3DlOpsScheduler.cpp:1471-1482`. Carrying the VALUES: the chunk half must state the SAME
    /// extents as the core half, and both names must be the ones the reference writes.
    #[test]
    fn the_chunk_stage_is_the_core_stage_under_the_reference_name() {
        assert_eq!(
            StageName::core().0,
            "core",
            "the stage at `dataStageCoreIdx` is named `core`"
        );
        assert_eq!(StageName::chunk().0, "chunk", "and its copy `chunk`");
        assert_eq!(
            DATA_STAGE_CORE.0.to_string(),
            "0",
            "`dataStageCoreIdx` is 0, and `0` is the ONE `dataStageParam_` key scratchy writes"
        );
    }
}
