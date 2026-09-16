// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ SCRATCHY'S SuperDSC AS THE SCHEDULER'S OWN `SuperDsc` — the half of bridge 1's scheduling leg
//! that speaks scratchy's vocabulary, so that `deeptools::sdsc::run_stages_2a_2b` can be handed REAL
//! DATA instead of a transcribed fixture.
//!
//! ```text
//! SubtileTape ──lower_subtile_tape_to_superdsc──► SdscOp / Dsc   (the json DTOs dxp_standalone reads)
//!             ──THIS MODULE──────────────────────► SuperDsc + computeOp_ + dsc.name_
//!             ──sdsc::run_stages_2a_2b───────────► a GROWN tree, HELD (`sdsc::Scheduling`)
//! ```
//!
//! ⛔⛔ WHY IT LIVES IN THE SPYRE CRATE AND NOT IN `deeptools`. `deeptools` never depends on scratchy —
//! it is the port of the vendor compiler and knows nothing of `EmittedOp`, `OpSpec` or the tape. So
//! the conversion cannot live there: the side that names BOTH vocabularies is this one.
//!
//! # ⛔⛔ THE LAYERING, AND WHAT IS STILL OWED — *"scratchy knows nothing about l3"*
//!
//! Two separate things had to come out of this file, and only ONE of them is done:
//!
//! 1. ✅ **THE COMPOSITION**, which is a target-neutral pass and now lives in
//!    `deeptools::schedule::stages` (`run_stages_2a_2b`) — the root `CLAUDE.md`'s *"one implementation
//!    of every target-neutral pass"*. Nothing here names `DscState`, `run_l3`, `run_ddc`,
//!    `AddressFoldCoords`, `ddc_defaults` or `Dsc2State` any more.
//! 2. ✅ **THE VOCABULARY REACH-IN**. This file used to name NINE `deeptools::schedule` internals
//!    directly — `l3::dsc`, `dsc2`, `ddc::fold`, `ddc::v1`, `ddc::transformation`,
//!    `transformation_util`, `stages`. It now names ONE declared seam, [`deeptools::sdsc`], and
//!    `grep deeptools::schedule crates/targets/` answers ZERO in code. That is what lets the
//!    scheduler's internals be rearranged without touching a target.
//! 3. ⛔ **THE FILE ITSELF IS STILL HERE, AND THAT IS THE OWED MOVE.** The wire→[`SuperDsc`]
//!    conversion belongs inside `deeptools`, and moving it needs the SuperDSC WIRE SCHEMA to move with
//!    it — blocked twice today: the DTOs are `serde`/`serde_json::Value` types and `deeptools` has NO
//!    serde dependency (its `Cargo.toml` lists only `sys-arch-spec`), and `SdscOp::coreIdToWkSlice_` /
//!    `AllocNode::maxDimSizes_` are typed by `scratchy_subtile` (`SliceIndex`, `DeviceWalk`), which
//!    `deeptools` may never depend on. ⭐ THE SEAM IS WHAT MAKES THAT A MOVE RATHER THAN A REWRITE:
//!    relocating this file deletes one `use` line instead of re-pointing nine.
//!
//! ⛔⛔ AND WHY IT IS NOT IN [`crate::lower_superdsc_to_dataflow_ir`]. That file's ratchet
//! (`tests/dfir_never_runtime_refuses.rs`) freezes `panic!`, `todo!` and `Result` at ZERO, because it
//! is the lowering the bake compiles and a stop there pre-empts dbo-opt. This module RUNS the stages
//! and reports where they stop, catching a panic rather than letting one kill the bake; it produces
//! the schedule the lowering is to take but decides nothing about what the bake emits. Keeping the two
//! apart is what lets the ratchet stay at zero.
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
//! ⭐⭐ AND ALL 24,363 NOW RUN **THROUGH** THE MEMORY TRACKER, with **zero** panics and **zero**
//! carrier refusals: re-measured after `stages::Trackers` was wired over the ported LX allocator, the
//! census above is UNCHANGED to the node (the stop moved three statements, and nothing is minted
//! between them) and not one `stage2a-stop:` or `stage2a-refusal:` line is printed for any program.
//!
//! ⛔⛔ THE ARENA IS GONE, AND THE STOP MOVED ONE STATEMENT — READ THIS BEFORE TRUSTING ANY OLDER
//! ACCOUNT. The stop used to be a bare `None` at `try_alloc_l3`'s `allocs.get(&alloc)?`: entry 222
//! looked the allocate node it was about to place up in a `v1::AllocArena` that NO UNIT OF STAGE 2A
//! EVER WROTE, because the reference has ONE `dsc2::AllocateNode *` per
//! `labeledDs_.at(lds).memOrg_.at(storage)` and the port had split it into a tree node AND an arena
//! entry, writing only the tree half. Entry 222 now reads that node through
//! `deeptools`' `AllocationReads` seam — the one cell entry 353's mint fills — and writes its
//! placements back through `AllocationSites::place_allocation`, so the arena has no production reader
//! or writer on the placement path at all.
//!
//! ⭐⭐⭐ THE CAPACITY IS ANSWERED AND THE LX PLACEMENTS ARE COMMITTED — READ THIS BEFORE ANY OLDER
//! ACCOUNT OF THE STOP. `L3Placement::buffer_capacity_even_sticks` (`deeptools stages/carriers.rs`)
//! used to REFUSE, on every one of the 24,363 programs, for want of the `&DesignSpaceConfig` the
//! reference calls `getBufferCapacityForNode` ON. It now takes that borrow as an argument — the one
//! `try_alloc_l3` already holds — and walks `deeptools`' `l3::capacity::buffer_capacity`
//! (`dsc/dsc2.cpp:3977` and its ~210-line `getBufferCapacityForNodePerDimCustomLocation`), so entry
//! 222's `alloc_all_mem` — the ONE committing call of the whole stage — runs and writes every LX start
//! address and buffer offset onto `memOrg_.allocateNode_`.
//!
//! ⛔⛔ AND THE STOP MOVED AGAIN, TO A DEFECT OF EXACTLY THE SHAPE ENTRY 222 JUST SHED. On the real
//! emitted matmul this file's own test converts, stage 2a now runs `set_lx_buffer_type` through
//! `alloc_all_mem`, `fill_transfer_zero_padding_info`, `fill_transfer_multicast_info`,
//! `fill_allocation_start_addr_and_offset` and entry 333's `offset_sizes`/`offset_nodes`/`offset_facts`
//! — 4 -> 23 nodes, **zero** carrier refusals — and stops inside entry 333's `fillDataInfo`
//! (`deeptools`' `schedule/l3/dl_ops.rs:17872`) at `inputs.allocs.get(&alloc)?`: the `v1::AllocArena`
//! that `deeptools`' `schedule/stages.rs` hands `run` EMPTY, whose own comment says *"the only reader
//! left is entry 333's constant-allocation conflation, which is unreachable while
//! `Reads::offset_sizes` refuses"*. `offset_sizes` no longer refuses, so that reader IS reached. ⭐ THE
//! ONE-LINE CONSEQUENCE: entry 333 must read the placed allocation off the ONE
//! `memOrg_.allocateNode_` cell (`AllocationReads`/`L3OffsetFacts`), exactly as entry 222 now does;
//! nothing else has to change here.
//!
//! ⛔⛔ AND THAT STOP IS WHY STAGE 2B IS NOT **REACHED** ON THE CORPUS, WHICH IS NOT THE SAME FACT AS
//! STAGE 2B BEING UNCALLED. [`run_stages`] composes 2a then 2b through `stages::run_stages_2a_2b`,
//! which gates 2b on 2a completing — an abandoned stage 2a leaves entry 333's operands UNFILLED and 2b
//! computes offsets FROM them, so running it there would compute addresses from half a fill. So
//! `stage2b:` reports `0 reached` for as long as `completed` is 0.
//!
//! ⛔⛔ NOT VERIFIED, AND HERE IS EXACTLY WHAT A RE-MEASUREMENT IS EXPECTED TO CHANGE. The whole-build
//! census has NOT been re-run since the capacity landed — the acceptance bake needs a card and
//! `dbo-opt`, neither of which this worktree has. The node counts above are now a FLOOR and not the
//! reading: `sync` and `compute` were ZERO *"because `create_synchronization` runs AFTER the stop"*,
//! and it no longer does — the two programs measured directly grew a full sync band
//! (`sync_{send,receive}_{l3lu,lxlu,l3su,lxsu}...`, eight nodes) and a `lx_below_schedule` block, so
//! `sync` should become non-zero and `nodes` should rise well past 347,939. ⭐ THE TWO DIRECT
//! MEASUREMENTS, BOTH IN-REPO: this file's own
//! [`tests::a_real_emitted_matmul_converts_and_stage_2a_grows_its_tree`] on a real emitted matmul wire
//! SDSC (4 -> 23 nodes, zero refusals, stopping at entry 333's arena), and `deeptools`' own
//! `schedule/stages.rs` fixture for `0_rmsq_o728` (4 -> 22 nodes — which is
//! `REFERENCE_STAGE_2A_TREE.len()`, the reference's whole stage-2a tree for that program, node for
//! node). ⚠️ THE FIXTURE STOPS EARLIER THAN THE WIRE DOES AND THE REASON IS THE FIXTURE: its
//! `SuperDsc::new(.., BTreeMap::new(), BTreeMap::new())` states an EMPTY `coreIdToWkSlice_`, which
//! entry 291's `sdsc.core_id_to_wk_slice.get(&core)?` refuses — a map [`superdsc_to_l3_sdsc`] fills
//! from the wire, which is why the wire program gets past it and the fixture does not.
//! ⛔ `stage2a-refusal:` should now be ABSENT from the corpus print where the previous account expected
//! `24363 x L3Placement::buffer_capacity_even_sticks`, and `stage-stop:` should stay absent too — the
//! new stop returns rather than panics.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use deeptools::arch::Elements;
use deeptools::formats::DataFormat;
// ⛔⛔ ONE deeptools PATH, AND IT IS THE DECLARED SEAM. `deeptools::sdsc` re-exports bridge 1's whole
// scheduling vocabulary; NOTHING here may name `deeptools::schedule::*`, because *"scratchy knows
// nothing about l3"* and the scheduler is a target-neutral pass. This file used to reach into nine
// separate `schedule` internals — `l3::dsc`, `dsc2`, `ddc::fold`, `ddc::v1`, `ddc::transformation`,
// `ddc::transformation_util` and `stages` — which is the boundary being absent rather than declared.
// ⭐ THE AUDIT IS `grep deeptools::schedule crates/targets/` AND ITS ANSWER IS ZERO.
use deeptools::sdsc::{
    ConstIdx, ConstantInfo, CoreIdsUsed, CoreletShare, CoreletsUsed, DATA_STAGE_CORE, DataStage,
    DataStages, DdcFacts, DesignSpaceConfig, DscComputeOp, DscFilled, DscIdx, DscList,
    DscScheduleStep, DscState, DsType, Extent, FilledDims, L0Tethered, LabeledDs,
    LabeledDsAllocations, LabeledDsList, LayoutDims, LdsIdx, LdsRecord, NamedDims, NodeKind, OpFunc,
    OpFuncs, Pinning, PrimaryDim, PrimaryDsInfo, Scale, Scheduling, SenComponent, StageDims,
    StageName, StickDims, StorageName, SuperDsc, WkSlice, WkSliceCount, WkSliceId, WordLength,
    layout_dims, run_stages_2a_2b,
};
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

/// ⭐ AN `exUnit` SPELLING AS THE SEALED [`SenComponent`], or [`None`] for one the wire cannot write.
///
/// ⛔ THE CLOSED SET IS THE TWO COMPONENTS THE EMITTER'S OWN PRODUCER CAN NAME, and both producers
/// are exhaustive matches with a two-value range: `OpFunc::ex_unit` answers `"pt"` for
/// `Matmul`/`BatchMatmul`/`Transpose` and `"sfp"` for everything else
/// (`scratchy_subtile::superdsc_opspec` `:1210-1221`), and
/// [`crate::lower_subtile_tape_to_superdsc::ex_unit`] (`:937-942`) is the same two. So no third
/// spelling reaches this, and the lookup goes through [`SenComponent::spelling`] rather than a table
/// written here — a renamed variant fails at the round trip instead of resolving to a neighbour.
///
/// ⛔ AND A SPELLING NEITHER NAMES IS A STOP, NOT A DEFAULT. `DscComputeOp::ex_unit` is what
/// `usePt` (`ddc/ddcv1.cpp:2026`) reads to decide whether the DSC's first compute runs on the PT,
/// which is what makes a row split meaningful — answering `Sfp` for an unknown unit would split the
/// wrong dim.
#[must_use]
pub fn ex_unit_of(spelling: &str) -> Option<SenComponent> {
    const UNITS: [SenComponent; 2] = [SenComponent::Pt, SenComponent::Sfp];
    UNITS.into_iter().find(|unit| unit.spelling() == spelling)
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
///     wire's — `paddingSizes_` is one number per dim where [`deeptools::sdsc::DimPadding`]
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
/// ⭐⭐ `dsName_`, `wordLength` AND `dataFormat_` ARE READ, and all three are on [`LdsRecord`]
/// because the authority puts them on `LabeledDsInfo` and not on `DesignSpaceConfig` —
/// `dsc/dscdefn.h:326`, `:334`, `:335`.
///
/// ⛔ AND THEY VARY THE WAY THEY MUST, which is what says this is a reading and not a constant: over
/// `g0/`'s 580 labelled DSs `dataFormat_` is `SEN169_FP16` on 573 and `SEN143_FP8` on 7, and
/// `wordLength` is `2` on exactly those 573 and `1` on exactly those 7 — the byte width of the format
/// beside it. `dsName_` is `Tensor{position}` on all 580, which is the name the reference's own seed
/// allocate node carries (`allocate-Tensor0_hbm` in `g0/debug/sdsc_0/sdsc.json`).
///
/// ⛔ A `dataFormat_` NO [`crate::formats::DataFormat`] SPELLS IS A REFUSAL AND NOT AN ABSENCE.
/// `LdsRecord::data_format`'s [`None`] means `DataFormats::INVALID`, which is *"nobody stated a
/// precision"*; a stated-but-unrecognised spelling is a different fact and folding the two together
/// would silently hand the DDL match an INVALID operand type it would then bind by.
///
/// ⛔ `scaledLdsCategory_` IS NOT WRITTEN BY SCRATCHY AT ALL — absent on all 580 — so it stays at the
/// declared `REGULAR_TENSOR` (`dsc/dscdefn.h:356`), which [`LabeledDs::new`] already is.
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
    let record = LdsRecord {
        name: StorageName(lds.dsName_.clone()),
        word_length: WordLength(lds.wordLength),
        data_format: Some(DataFormat::from_spelling(lds.dataFormat_)?),
    };
    Some(
        LabeledDs::new(
            ds_type,
            scales,
            LdsIdx(lds.ldsIdx_),
            pinning_of(&lds.memOrg_),
        )
        .with_record(record),
    )
}

/// ⭐⭐ `constantInfo_` AS THE MAP THE SCHEDULER READS — `std::map<int, dsc2::ConstantInfo>`
/// (`dsc/designSpaceConfig.h:90`).
///
/// ⛔⛔ THE WIRE CARRIES TWO SHAPES AND BOTH ARE THE SAME FACT: the emitter writes the JSON *string*
/// `"{}"` for an empty table and a real object otherwise, and its own note says why —
/// *"the empty OBJECT {} is falsy in dxp's Python so the SFP constant-table / NR-refine setup is
/// skipped"* (`lower_subtile_tape_to_superdsc.rs:1257-1262`). 175 of `g0/`'s 187 DSCs carry the
/// string; the other 12 carry `{"0": {"allocations_": {}, "dataFormat_": "SEN169_FP16",
/// "data_": [10240 | 15872], "name_": "scaling_factor"}}`.
///
/// ⛔ [`None`] IS A STATED-BUT-UNREADABLE TABLE: a key that is not an integer id, an entry that is not
/// an object, a `data_` element that is not an integer, or a `dataFormat_` no [`DataFormat`] spells.
/// Every one of those is a constant the scheduler WOULD read and we cannot state, and
/// `declares_zero_mean_constant` turns exactly such a table into an op-func swap
/// (`ddc/ddcv1.cpp:2064-2072`) — so a dropped entry is a silently different program.
///
/// ⛔ `allocations_` IS NOT READ EVEN WHERE IT IS PRESENT, and it is `{}` on every constant scratchy
/// writes. It holds `dsc2::AllocateNode*`s the ALLOCATOR fills; `ComponentAllocations::
/// set_constant_allocation` is what writes it, and a non-empty one here would be a placement nobody
/// made.
fn constant_info_of(value: &serde_json::Value) -> Option<BTreeMap<ConstIdx, ConstantInfo>> {
    // The emitter's own empty spelling — a JSON string, not an object.
    if value.as_str() == Some("{}") {
        return Some(BTreeMap::new());
    }
    let table = value.as_object()?;
    table
        .iter()
        .map(|(id, entry)| {
            let entry = entry.as_object()?;
            let data = entry
                .get("data_")
                .map_or_else(|| Some(Vec::new()), |data| {
                    data.as_array()?.iter().map(serde_json::Value::as_i64).collect()
                })?;
            let data_format = match entry.get("dataFormat_") {
                Some(format) => Some(DataFormat::from_spelling(format.as_str()?)?),
                // ⛔ AN ENTRY THAT NAMES NO FORMAT IS `DataFormats::INVALID`, the field's own
                // initializer (`dsc/dsc2.h:47`).
                None => None,
            };
            Some((
                ConstIdx(id.parse::<u32>().ok()?),
                ConstantInfo {
                    name: StorageName(
                        entry.get("name_").and_then(serde_json::Value::as_str)?.to_owned(),
                    ),
                    data_format,
                    data,
                    // `isDataSymbolic_` (`dsc/dsc2.h:50`) — the emitter writes no such key on any of
                    // the twelve constants of `g0/`, which is the declared `false`.
                    is_data_symbolic: entry
                        .get("isDataSymbolic_")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                    allocations: BTreeMap::new(),
                },
            ))
        })
        .collect()
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

/// ⭐⭐ THE WIRE'S ALLOCATE NODES AS `getLayoutDims` WALKS THEM — `scheduleTree_` keyed by `ldsIdx_`
/// and `component_`, which IS `memOrg_[component].allocateNode_->layoutDimOrder_`: the emitter writes
/// one node per view (`lower_subtile_tape_to_superdsc.rs:5150-5161`) beside the `labeledDs_` entry for
/// the same view (`:4988`), both with `ldsIdx_: i as u32`, so every labelled DS position answers.
/// MEASURED over `g0/sdsc_*.json`: 580 allocate nodes, one per labelled DS, every `component_`
/// `"hbm"` — `scheduleTree_` carries nothing else (`nodeType_` is `"allocate"` on all of them).
///
/// ⛔ `referenceLdsIdx_` IS ABSENT FROM THE WIRE RECORD (`:1043-1051`), which is the reference's `-1`:
/// the walk resolves on the labelled DS's own `memOrg_` and never takes a second lap. The field is
/// absent from all 580 wire records AND from all 807 records of the reference's own export.
struct WireAllocations(BTreeMap<LdsIdx, BTreeMap<SenComponent, LayoutDims>>);

impl WireAllocations {
    /// ⛔ [`None`] IS A `component_` NO [`SenComponent`] SPELLS, or a layout order naming a dim no
    /// [`PrimaryDim`] spells, or an EMPTY one — the refusals [`primary_ds_info_of`] makes of the
    /// other order, over the allocate node's.
    fn of(dsc: &WireDsc) -> Option<Self> {
        let mut held: BTreeMap<LdsIdx, BTreeMap<SenComponent, LayoutDims>> = BTreeMap::new();
        for node in &dsc.scheduleTree_ {
            let component = [SenComponent::Hbm, SenComponent::Lx]
                .into_iter()
                .find(|component| component.spelling() == node.component_)?;
            let mut order = node
                .layoutDimOrder_
                .iter()
                .map(|name| primary_dim_of(name))
                .collect::<Option<Vec<_>>>()?
                .into_iter();
            let first = order.next()?;
            held.entry(LdsIdx(node.ldsIdx_))
                .or_default()
                .insert(component, LayoutDims::new(first, order.collect()));
        }
        Some(Self(held))
    }
}

impl LabeledDsAllocations for WireAllocations {
    fn alloc_layout_orders(&self, lds: LdsIdx) -> Option<Vec<(SenComponent, LayoutDims)>> {
        Some(
            self.0
                .get(&lds)?
                .iter()
                .map(|(component, order)| (*component, order.clone()))
                .collect(),
        )
    }

    fn reference_lds(&self, _lds: LdsIdx) -> Option<LdsIdx> {
        None
    }
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
/// ⛔ `coordinateMasking_` IS OUTSIDE THE TARGET'S PROJECTION, not dropped here: `l3::dsc` is *"a
/// reduced per-module projection of one C++ class … each module states the fields its own units touch
/// and nothing else"* (its own header), and stage 2b reaches it through `v1::Masking::
/// coordinate_masking` rather than off the DSC. It is empty on all 187 programs of `g0/`.
///
/// ⭐ `maskingConstId_` AND `constantInfo_` **ARE** READ, ONTO [`DdcFacts`] — see [`constant_info_of`].
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

    // 6. `getLayoutDims(ldsIdx)` — READ, per labelled DS, THROUGH THE WALK ITSELF; see
    //    [`WireAllocations`]. ⭐ KEYED BY THE POSITION the DS sits at in `labeledDs_`, which is what
    //    `LabeledDsList::indexed` and `DscState::seeded` index by — NOT by the entry's own
    //    `recorded()` index. The emitter writes `ldsIdx_: i as u32` on BOTH records, so the position
    //    keys the allocate nodes too.
    //
    // ⭐⭐ IT USED TO ANSWER `primary_ds_info.get(&lds.ds_type())?.layout`, WHICH IS THE OTHER LIST —
    //    AND THE TWO AGREE ON THE WHOLE CORPUS, 580 of 580 labelled DSs of `g0/sdsc_*.json` and 807 of
    //    807 of the reference's own export. This is NOT a defect fix; it is the derivation.
    //    `primaryDsInfo_` holds ONE entry per `dsType_` and the emitter's
    //    `primary.entry(v.role.ds_type()).or_insert_with(..)` (`:5021`) lets the FIRST view of a role
    //    fix it, while `getLayoutDims` answers the ALLOCATE NODE's own `layoutDimOrder_`
    //    (`dsc/dsc2.cpp:4007-4025`), which the emitter writes PER VIEW (`:5161`). The day two views of
    //    one role carry different orders, only this spelling stays right.
    let allocations = WireAllocations::of(dsc)?;
    let layout_dims = labeled_ds
        .indexed()
        .map(|(at, _)| Some((at, layout_dims(&allocations, at)?)))
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

    // 9. `computeOp_.at(0).indirectAccessIndexLabeledDs` — READ, THROUGH [`lds_by_operand_name`].
    //    ⭐⭐ EMPTY ON EVERY SCRATCHY PROGRAM TODAY, BY CONSTRUCTION: both `ComputeOp` build sites in
    //    the emitter write `indirectAccessIndexLabeledDs: vec![]`
    //    (`lower_subtile_tape_to_superdsc.rs:5275`, `:5362`). ⛔⛔ IT USED TO REFUSE A NON-EMPTY ONE,
    //    CITING A MAPPING THAT DOES NOT EXIST — *"the wire carries operand NAMES (`"Tensor0-idx0"`)
    //    and this field wants an `LdsIdx`, so resolving it needs the name→position mapping the emitter
    //    does not yet write down"*. THAT WAS FALSE. The wire `LabeledDs`
    //    (`lower_subtile_tape_to_superdsc.rs:1043-1045`) carries `ldsIdx_: u32` AND `dsName_: String`
    //    on the SAME record, and the operand spelling is those two composed — see
    //    [`lds_by_operand_name`]. So the field is resolved rather than refused, and a name the map
    //    does not hold is a STOP.
    let by_name = lds_by_operand_name(dsc);
    let indirect_access_index_lds = dsc
        .computeOp_
        .first()
        .into_iter()
        .flat_map(|op| op.indirectAccessIndexLabeledDs.iter())
        .map(|name| by_name.get(name.as_str()).copied())
        .collect::<Option<BTreeSet<_>>>()?;

    // 11. `N_.paddingSizes_` — EMPTY. Nothing in the emitter ever writes an `IterSpace`'s
    //     `paddingSizes_` (every one comes from `IterSpace::empty()`), measured empty on all 187
    //     programs of `g0/`. ⛔ AND A NON-EMPTY ONE REFUSES, in [`stage_half`], for the same reason:
    //     one wire number cannot fill a front AND a back edge.
    if !dsc.N_.paddingSizes_.is_empty() {
        return None;
    }
    let full_padding = BTreeMap::new();

    // 13. `constantInfo_` — READ, both wire spellings; see [`constant_info_of`].
    let constants = constant_info_of(&dsc.constantInfo_)?;

    // 14. `maskingConstId_` — READ. ⭐ [`None`] IS THE DECLARED `-1` (`dsc/designSpaceConfig.h:101`)
    //     AND A NEGATIVE ID IS THAT AND NOTHING ELSE: scratchy writes `-1` on all 187 programs, and
    //     `masking_constant` is `constantInfo_.count(maskingConstId_)`, which `-1` never satisfies.
    //     ⛔ AN ID THAT DOES NOT FIT A `u32` IS A REFUSAL, not a `-1`: it is a stated constant we
    //     cannot name, and answering "no masking constant" for it would drop entry 262's whole guard.
    let masking_const = match dsc.maskingConstId_ {
        -1 => None,
        id => Some(ConstIdx(u32::try_from(id).ok()?)),
    };

    Some(DesignSpaceConfig {
        // 15/16. `dimToSymbolMapping_` AND `l0TetheredMode_` — ⭐ THE DECLARED `{}` AND `false`,
        //        because scratchy EMITS NEITHER: the emitter's own field list calls both scheduler
        //        OUTPUTS and drops them (`lower_subtile_tape_to_superdsc.rs:1240-1241`). ⛔ THAT IS AN
        //        INPUT STATE AND NOT A MISSING FACT — `dimToSymbolMapping_` is one of the fields stage
        //        2b itself WRITES (it appears in `g0/debug/sdsc_0/sdsc.json`, the OUTPUT), and
        //        `l0TetheredMode_` is `ddc/ddcv1.cpp:271`'s read of a mode nothing has set.
        ddc: DdcFacts {
            constants,
            masking_const,
            dim_to_symbol: BTreeMap::new(),
            l0_tethered: L0Tethered::Split,
        },
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
//  `computeOp_` — stage 2b's own construction argument
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐⭐ EVERY LABELLED DS BY THE NAME A `computeOp_` OPERAND CALLS IT — the mapping a stale comment
/// in this very file claimed did not exist.
///
/// ⛔⛔ THE OPERAND SPELLING IS `{dsName_}-idx{ldsIdx_}`, NOT `dsName_`. Both facts are on the SAME
/// wire record (`lower_subtile_tape_to_superdsc.rs:1043-1045`: `ldsIdx_: u32` beside
/// `dsName_: String`), and the emitter composes the operand reference out of exactly those two at
/// both `ComputeOp` build sites — `format!("Tensor{i}-idx{i}")` (`:5049-5057`, `:5248-5249`) beside
/// `ldsIdx_: i as u32` / `dsName_: format!("Tensor{i}")` (`:4989-4990`). So the suffix is not
/// decoration to be stripped: it is the INDEX restated, and keying by the composed spelling checks
/// the two agree instead of trusting either alone.
///
/// ⭐⭐ AND THE AUTHORITY AGREES, ON BOTH SIDES OF THE STAGES. `g0/sdsc_0.json` — the input
/// `dxp_standalone` compiles to a working `init_binary` — carries `dsName_` `"Tensor0"`/`"Tensor1"`/
/// `"Tensor2"` with `computeOp_[0].inputLabeledDs` `["Tensor0-idx0", "Tensor1-idx1"]`, and
/// `g0/debug/sdsc_0/sdsc.json` — the same programs AFTER both stages — carries those same operand
/// names unchanged. So the spelling is stable across the stages and is what stage 2b reads.
///
/// ⛔ A DUPLICATE SPELLING IS NOT MERGED SILENTLY: the map is keyed by the composed name, so two
/// entries claiming one spelling collapse to the LAST, and [`dsc_compute_ops`] would then resolve an
/// operand to the wrong DS. That cannot happen while `ldsIdx_` is the entry's own position — the
/// suffix makes every key distinct by construction — and it is the reason the index is in the key.
#[must_use]
fn lds_by_operand_name(dsc: &WireDsc) -> BTreeMap<String, LdsIdx> {
    dsc.labeledDs_
        .iter()
        .map(|lds| {
            (
                format!("{}-idx{}", lds.dsName_, lds.ldsIdx_),
                LdsIdx(lds.ldsIdx_),
            )
        })
        .collect()
}

/// ⭐⭐⭐ ONE DSC'S `computeOp_` AS STAGE 2B'S OWN [`DscComputeOp`] LIST — the construction argument
/// [`run_stages_2a_2b`] cannot be called without.
///
/// ⛔⛔ AN EMPTY LIST IS A FALSE GREEN AND NOT A CHEAP ANSWER. `v1::PrepDsc::compute_ops` is the
/// FIRST provider call `run_v1` makes (`ddc/v1.rs:6437`) and an empty answer makes it `continue` past
/// the DSC — so a caller stating no ops gets [`DscFilled::Yes`] having placed no address and
/// minted no node. That is why every one of the five fields below is READ, and why a field that
/// cannot be read stops the whole conversion.
///
/// ⛔⛔ A NAME THAT DOES NOT RESOLVE IS A STOP, NOT A GUESS. No index 0, no positional fallback, no
/// skipped operand: `inputLabeledDs`/`outputLabeledDs` are what entries 307 and 308 sweep to decide
/// the reduction axis and the tensor sizes, so an operand pointing at the wrong labelled DS is a
/// wrong extent and then a wrong address. [`None`] here means the composition is not called at all.
///
/// ⭐ THE FIVE FIELDS, EACH ONE READ:
///
/// * `op_func` — `opFuncName` through `OpFunc::from_spelling`, which is
///   `EnumsConversion::stringToOpFuncs`. ⛔ A NAME THE SET DOES NOT SPELL IS [`None`] FOR THAT OP,
///   which is `OpFuncs::NONE` — the reference's own `ComputeOpInfo::opFuncName` default — exactly as
///   [`op_funcs_of`] already reads the same field. Never a plausible neighbour.
/// * `ex_unit` — `exUnit` through [`ex_unit_of`], a STOP for a spelling the emitter cannot write.
/// * `format` — `attributes_.dataFormat_` through [`DataFormat::from_spelling`], whose [`None`] is
///   `DataFormats::INVALID`, *"nobody stated a precision"*. ⛔ AND A STATED-BUT-UNRECOGNISED
///   SPELLING IS A STOP RATHER THAN THAT ABSENCE — the same distinction [`labeled_of`] already draws
///   on `LdsRecord::data_format`: folding the two together would hand the DDL match an INVALID
///   operand type it would then bind by. The emitter writes only `DataFormat` spellings
///   (`lower_subtile_tape_to_superdsc.rs:5326-5348`), so the stop is unreachable on scratchy's own
///   programs and is here because the wire type is a `&'static str`.
/// * `inputs` / `interim` / `outputs` — `inputLabeledDs` / `interimLabeledDs` / `outputLabeledDs`,
///   all three through [`lds_by_operand_name`], because all three are the same
///   `std::vector<LabeledDsInfo*>` (`dsc/dscdefn.h:506-510`) and the writer spells all three
///   identically as `{dsName_}-idx{ldsIdx_}` (`dsc/designSpaceConfig.cpp:6730-6753`, interim at
///   `:6738-6741`).
///
/// ⛔ `indirectAccessIndexLabeledDs` IS STILL NOT ON [`DscComputeOp`] — the indirect list reaches the
/// stage through `DesignSpaceConfig::indirect_access_index_lds` instead, which
/// [`design_space_config`] fills off the same map.
#[must_use]
pub fn dsc_compute_ops(dsc: &WireDsc) -> Option<Vec<DscComputeOp>> {
    let by_name = lds_by_operand_name(dsc);
    let operands = |named: &[String]| {
        named
            .iter()
            .map(|name| by_name.get(name.as_str()).copied())
            .collect::<Option<Vec<_>>>()
    };
    dsc.computeOp_
        .iter()
        .map(|compute| {
            let format = match compute.attributes_.dataFormat_ {
                // `DataFormats::INVALID`'s own spelling — the one absence, and the only one.
                "INVALID" => None,
                stated => Some(DataFormat::from_spelling(stated)?),
            };
            Some(DscComputeOp {
                op_func: OpFunc::from_spelling(&compute.opFuncName),
                ex_unit: ex_unit_of(compute.exUnit)?,
                format,
                inputs: operands(&compute.inputLabeledDs)?,
                interim: operands(&compute.interimLabeledDs)?,
                outputs: operands(&compute.outputLabeledDs)?,
            })
        })
        .collect()
}

/// ⭐⭐ EVERY DSC'S `computeOp_` LIST, POSITIONALLY BESIDE `dscs_` — what
/// `Dsc2State::seeded` indexes by.
///
/// ⛔ ONE `Vec` PER DSC AND NO GAPS: a position the caller states no list for gets an EMPTY one from
/// `seeded`, which is `run_v1`'s own `continue`, so a DROPPED entry would silently skip that DSC.
/// [`None`] from any DSC therefore refuses the whole list rather than shortening it.
#[must_use]
pub fn compute_ops_of(op: &SdscOp) -> Option<Vec<Vec<DscComputeOp>>> {
    op.dscs_
        .iter()
        .flat_map(BTreeMap::values)
        .map(dsc_compute_ops)
        .collect()
}

/// ⭐⭐ EVERY DSC'S `name_`, POSITIONALLY BESIDE `dscs_` — READ, and read from the one place scratchy
/// writes it.
///
/// ⭐ `dsc.name_` IS THE `dscs_` MAP **KEY**. The wire's `dscs_` is a `Vec<BTreeMap<String, Dsc>>`
/// (`lower_subtile_tape_to_superdsc.rs:1300`) whose key is the program name the emitter states
/// (`"MatMul_0"`, `"rmsq_o728"` in `g0/sdsc_0.json`), and `dsc/designSpaceConfig.h:60`'s `name_` is
/// the same fact — so this is a READ and not the positional `dsc{at}` spelling
/// `Dsc2State::seeded` falls back to.
///
/// ⛔ IT IS NOT LOAD-BEARING BEYOND DIAGNOSTICS — only `v1::Dsc2Fill::said`'s two verbose lines read
/// it — which is why an absent key is not a refusal here; there is no absent key, because a
/// `BTreeMap` entry has one by construction.
#[must_use]
pub fn dsc_names(op: &SdscOp) -> Vec<StorageName> {
    op.dscs_
        .iter()
        .flat_map(BTreeMap::keys)
        .map(|name| StorageName(name.clone()))
        .collect()
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
#[must_use]
pub fn op_funcs_of(op: &SdscOp) -> OpFuncs {
    let named: Vec<Option<OpFunc>> = op
        .dscs_
        .first()
        .into_iter()
        .flat_map(|per_name| per_name.values())
        .flat_map(|dsc| dsc.computeOp_.iter())
        .map(|compute| OpFunc::from_spelling(&compute.opFuncName))
        .collect();
    let mut entries = named.into_iter();
    let first = entries.next().flatten();
    OpFuncs::new(first, entries.collect())
}

/// ⭐⭐ ONE PROGRAM, SCHEDULED — the artifacts the LOWERING takes, plus the two readings
/// [`deeptools::sdsc::StagesRan`] does not carry.
///
/// ⛔⛔ THE ARTIFACTS ARE THE POINT, AND THEY USED TO BE DROPPED. The old `census` ran the stages,
/// absorbed the numbers into a [`Corpus`] and let the scheduled super-DSC fall off the end of the loop
/// — so the DSC handed to the lowering was the UNSCHEDULED wire one, which is why the bake reported
/// 134 bundles and 0 launch groups. Carrying [`Scheduling`] out is what closes that.
#[derive(Debug)]
pub struct Scheduled {
    /// ⭐ THE SCHEDULED SUPER-DSC AND ITS TREE — see [`Scheduling::state`], which is where every node
    /// the stages minted lives.
    pub scheduling: Scheduling,
    /// ⭐ THE PER-`nodeType_` CENSUS of that tree — the count the four stages owe, per kind.
    pub kinds: BTreeMap<NodeKind, usize>,
    /// ⭐⭐ HOW MANY [`DscComputeOp`]s WERE HANDED TO STAGE 2B, over every DSC.
    ///
    /// ⛔ A ZERO HERE IS THE FALSE GREEN, NAMED. `v1::PrepDsc::compute_ops` is the first provider
    /// call `run_v1` makes and an empty answer makes it `continue` past the DSC, so a `0` beside a
    /// [`DscFilled::Yes`] is a stage that did nothing — which is why the count is carried out
    /// rather than left implicit in the wire.
    pub compute_ops: usize,
}

impl Scheduled {
    /// ⭐⭐⭐ THE SCHEDULE TREE, FOR THE LOWERING TO WALK — every node stages 2a and 2b minted.
    ///
    /// ⛔⛔ THIS IS THE HAND-OFF THE WHOLE TASK EXISTS FOR, AND IT IS THE TYPED TREE RATHER THAN THE
    /// WIRE'S `nodeType_` STRING. The lowering reads `crate::lower_subtile_tape_to_superdsc::Dsc`'s
    /// `scheduleTree_`, which is a `Vec<AllocNode>` whose every `nodeType_` is `"allocate"` — so it
    /// walks no statement and lowers no program. [`DscState::dscs`] hands out one `DscTree` per DSC,
    /// positionally beside [`Scheduling::sdsc`]'s own `dscs()`, and THAT is what carries the loops,
    /// transfers, syncs and computes.
    ///
    /// ⛔ IT IS BORROWED FROM THE [`Scheduling`] THIS VALUE OWNS, so the lowering's `'c` must not
    /// outlive the [`Bundle`] the walk holds — which is why `render_dfir_input` keeps the bundle alive
    /// across the whole group loop rather than per group.
    #[must_use]
    pub const fn state(&self) -> &DscState {
        self.scheduling.state()
    }

    /// ⭐⭐ HOW MANY **STATEMENT** NODES THE SCHEDULED TREE HOLDS — every node that is not an
    /// `ALLOCATE`, which is exactly what a lowering walks.
    ///
    /// ⛔ THE NUMBER THAT DECIDES WHETHER THE LOWERING PRODUCES A PROGRAM. `ScheduleNode::NodeType`
    /// is `{BLOCK, LOOP, TRANSFER, COMPUTE, SYNC, CONDITION, ALLOCATE, STICKMASK}` and the port's
    /// `Statement` is those less `ALLOCATE` (`superdsc_to_dataflow_ir/driver.rs:467`), so a tree of
    /// nothing but allocations lowers to nothing — which is what scratchy's own unscheduled
    /// `scheduleTree_` is, and what these two stages exist to change.
    #[must_use]
    pub fn statements(&self) -> usize {
        self.kinds
            .iter()
            .filter(|(kind, _)| **kind != NodeKind::Allocate)
            .map(|(_, count)| count)
            .sum()
    }
}

/// ⭐⭐ WHAT ONE PROGRAM DID — the three outcomes, told apart by TYPE rather than by a `None` that
/// folds two of them together.
#[derive(Debug)]
pub enum Ran {
    /// ⛔ THE CONVERSION REFUSED — [`super_dsc`] or [`compute_ops_of`]. NO STAGE WAS CALLED, and the
    /// program is reported as unconverted rather than counted as zero nodes.
    NotConverted,
    /// ⛔⛔ A STAGE PANICKED, and the message is carried so the loop is not blind.
    ///
    /// ⛔ NO ARTIFACTS, AND NOT BY OVERSIGHT: the super-DSC is MOVED into
    /// [`run_stages_2a_2b`], so the unwind takes it and the tree with
    /// it. Re-seeding an empty one here would report a program that *scheduled to nothing*, which is
    /// a different fact from one that stopped.
    Stopped(String),
    /// ⭐ BOTH STAGES RETURNED — the artifacts, and which of them completed
    /// ([`Scheduling::ran`]).
    Scheduled(Scheduled),
}

impl Ran {
    /// ⭐ THE SCHEDULED ARTIFACTS, [`None`] for a program that did not convert or that stopped.
    ///
    /// ⛔ THE TWO ABSENCES ARE FOLDED **ONLY HERE**, for a caller that needs the schedule and cannot
    /// act on the difference; [`Ran`] itself keeps them apart so the census can report each by name.
    #[must_use]
    pub const fn scheduled(&self) -> Option<&Scheduled> {
        match self {
            Self::Scheduled(held) => Some(held),
            Self::NotConverted | Self::Stopped(_) => None,
        }
    }
}

/// ⭐⭐⭐ CONVERT ONE SCRATCHY `SdscOp` AND RUN **BOTH** SCHEDULER STAGES OVER IT, KEEPING WHAT THEY
/// LEFT.
///
/// ⛔⛔ [`Ran::NotConverted`] IS A CONVERSION REFUSING, and there are now TWO of them —
/// [`super_dsc`] and [`compute_ops_of`]. The second is the one that matters: an operand name no
/// labelled DS answers to stops the program HERE rather than handing stage 2b a shortened list,
/// because `v1::PrepDsc::compute_ops` is `run_v1`'s first call and an empty answer makes it
/// `continue` past the DSC — [`DscFilled::Yes`] having done nothing at all. If the ops cannot be
/// built, the stage must not be called.
///
/// ⛔⛔ THE `catch_unwind` IS MEASUREMENT INSTRUMENTATION, NOT A RUNTIME REFUSAL. Both stages hold
/// `todo!`s — a fact they cannot answer must NOT be faked, because a carrier answering a plausible
/// offset would place real tensors at invented addresses. A panic escaping here would kill the BAKE,
/// so the panic is caught, its message reported as *where it stopped*, and nothing is substituted for
/// the answer it did not give.
///
/// ⛔ IT IS SOUND TO READ THE ARTIFACTS AFTER THE UNWIND — and they SURVIVE it, which is why the
/// `Option` is unwrapped outside the closure. `sdsc`, `ops` and `names` are owned by THIS frame, so an
/// unwind inside [`run_stages_2a_2b`] would drop the [`Scheduling`] it
/// was building; a caught panic therefore reports where it stopped and carries no artifacts, which is
/// the honest answer rather than a half-scheduled tree.
///
/// ⭐ THE REAL OP-FUNC, NOT `None`: `run_stages` states `OpFuncs::new(None, ..)` because `computeOp_`
/// is not a field of `l3::dsc::SuperDsc`, and scratchy holds it as `computeOp_[i].opFuncName`.
#[must_use]
pub fn run_stages(op: &SdscOp) -> Ran {
    let (Some(sdsc), Some(dsc_ops)) = (super_dsc(op), compute_ops_of(op)) else {
        return Ran::NotConverted;
    };
    let ops = op_funcs_of(op);
    let names = dsc_names(op);
    let compute_ops = dsc_ops.iter().map(Vec::len).sum();

    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_stages_2a_2b::<deeptools::arch::Dd2>(sdsc, ops, &dsc_ops, &names)
    }));
    match ran {
        Ok(scheduling) => Ran::Scheduled(Scheduled {
            kinds: scheduling.state().kinds(),
            compute_ops,
            scheduling,
        }),
        Err(payload) => Ran::Stopped(panic_text(payload.as_ref())),
    }
}

/// ⭐⭐ THE CORPUS CENSUS — every program of one bundle, aggregated.
///
/// ⭐ MEASURED, and the whole-build aggregate of these is in this module's header: 134 bundles,
/// 24,363 programs, all converted, 93,110 -> 347,939 nodes, 0 completed — every one running THROUGH
/// the memory tracker and stopping at entry 222's arena lookup, with no `stops` entry and no
/// `refusals` entry anywhere in the corpus.
///
/// ⛔ THE STAGE-2A AND STAGE-2B COUNTS ARE SEPARATE FIELDS AND ARE NEVER ADDED. Reporting one total
/// for two stages is how a scope's completion gets read as the stage's; `completed` is 2a's and
/// `ddc_ran`/`ddc_filled` are 2b's, and 2b is only ever REACHED on a program `completed` counts.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    /// How many programs were offered.
    pub programs: usize,
    /// How many the conversion produced an `l3::dsc::SuperDsc` AND a `computeOp_` list for.
    pub converted: usize,
    /// Seed nodes over every converted program.
    pub nodes_before: usize,
    /// Nodes after both stages over every converted program.
    pub nodes_after: usize,
    /// ⭐ THE PER-`nodeType_` TOTAL, which is the gate.
    pub kinds: BTreeMap<NodeKind, usize>,
    /// ⭐ HOW MANY COMPUTE OPS RESOLVED, over every converted program — the number that says the list
    /// handed to stage 2b is not empty. An empty list is `run_v1`'s own `continue`.
    pub compute_ops: usize,
    /// How many programs stage 2a completed on — and therefore how many stage 2b was REACHED on.
    pub completed: usize,
    /// How many programs stage 2b RETURNED on, whether it filled them or not.
    pub ddc_ran: usize,
    /// How many of those it answered `DscFilled::Yes` for.
    pub ddc_filled: usize,
    /// ⭐⭐ WHERE THE PROGRAMS STOPPED, one entry per distinct message with its count — so a stop
    /// anywhere OTHER than the memory tracker is visible by name rather than folded into a total.
    pub stops: BTreeMap<String, usize>,
    /// Every stage-2a carrier refusal, by method, with its count — a provider gap as opposed to a
    /// ported unit's own stop.
    pub refusals: BTreeMap<&'static str, usize>,
    /// The same for stage 2b's own carriers, kept apart because they are DIFFERENT traits over the
    /// same state.
    pub ddc_refusals: BTreeMap<&'static str, usize>,
}

impl Corpus {
    /// Fold one program's measurement in.
    fn absorb(&mut self, scheduled: &Scheduled) {
        let ran = scheduled.scheduling.ran();
        self.converted += 1;
        self.nodes_before += ran.nodes_before;
        self.nodes_after += ran.nodes_after;
        for (kind, count) in &scheduled.kinds {
            *self.kinds.entry(*kind).or_insert(0) += count;
        }
        self.compute_ops += scheduled.compute_ops;
        if ran.l3 {
            self.completed += 1;
        }
        if ran.ddc {
            self.ddc_ran += 1;
        }
        if scheduled.scheduling.filled() == Some(DscFilled::Yes) {
            self.ddc_filled += 1;
        }
        if let Some(refusal) = ran.first_refusal {
            *self.refusals.entry(refusal).or_insert(0) += 1;
        }
        if let Some(refusal) = scheduled.scheduling.ddc_refusal() {
            *self.ddc_refusals.entry(refusal).or_insert(0) += 1;
        }
    }

    /// ⭐ THE CENSUS AS ONE LINE PER FACT — what the bake prints and the aggregation greps.
    #[must_use]
    pub fn report(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "stage2a: {} program(s), {} converted, {} completed; nodes {} -> {}",
                self.programs,
                self.converted,
                self.completed,
                self.nodes_before,
                self.nodes_after,
            ),
            // ⛔ REPORTED PER STAGE. `reached` is `completed` restated on purpose: stage 2b runs only
            // over a program stage 2a finished, so a zero there is the whole explanation for a zero
            // here and the two must be readable on one line.
            format!(
                "stage2b: {} reached, {} returned, {} filled; {} compute op(s) offered",
                self.completed, self.ddc_ran, self.ddc_filled, self.compute_ops,
            ),
        ];
        for (kind, count) in &self.kinds {
            lines.push(format!("stage-kind: {} {count}", kind.spelling()));
        }
        for (stop, count) in &self.stops {
            lines.push(format!("stage-stop: {count} x {stop}"));
        }
        for (refusal, count) in &self.refusals {
            lines.push(format!("stage2a-refusal: {count} x {refusal}"));
        }
        for (refusal, count) in &self.ddc_refusals {
            lines.push(format!("stage2b-refusal: {count} x {refusal}"));
        }
        lines
    }
}

/// ⭐⭐⭐ CONVERT AND RUN BOTH STAGES OVER EVERY PROGRAM OF ONE BUNDLE, **KEEPING WHAT THEY LEFT** —
/// the whole correction, in the return type.
///
/// ⛔⛔ THE OLD `census` RETURNED ONLY THE [`Corpus`], AND THAT WAS THE DEFECT. It ran the stages over
/// every program, absorbed the node counts, and let each scheduled super-DSC drop at the end of the
/// loop iteration; the DSC then handed to the lowering was the UNSCHEDULED wire one. So the bake
/// reported 134 bundles, 0 launch groups and 0.0 MB of device code from a scheduler that had run
/// 24,363 times. [`Self::programs`] is what closes it — and the function is named
/// [`schedule_bundle`] rather than `census` because a census is what it used to be and a name that
/// says *statistics* is how the artifacts came to be droppable in the first place.
#[derive(Debug)]
pub struct Bundle {
    /// The aggregate reading, which is what the bake prints.
    pub corpus: Corpus,
    /// ⭐⭐ EVERY PROGRAM'S OUTCOME, **POSITIONALLY BESIDE THE INPUT** — so the lowering can take the
    /// scheduled artifacts for the very program it is lowering. A dropped entry would misalign every
    /// later program with its schedule, which is why the walk pushes one per input unconditionally.
    pub programs: Vec<Ran>,
}

impl Bundle {
    /// ⭐⭐⭐ THE SCHEDULE TREE OF THE PROGRAM AT ONE INPUT POSITION — **the hand-off the lowering
    /// takes**, and the reason [`Self::programs`] is positional.
    ///
    /// ⛔⛔ THE INDEX IS INTO THE `&[SdscOp]` [`schedule_bundle`] WAS GIVEN, not into the group or the
    /// launch order. `render_dfir_input` partitions those same programs into groups with
    /// `group_ranges`, so a group's `i`-th program is input `r.start + i` — reading it by the group's
    /// own offset would pair every program past group 0 with ANOTHER program's addresses, which
    /// compiles and lowers and is silently wrong.
    ///
    /// ⛔ [`None`] IS A PROGRAM THAT DID NOT CONVERT OR THAT STOPPED, and a lowering must treat that as
    /// *no schedule* rather than an empty one — the two are the difference between a stop and a program
    /// that scheduled to nothing.
    #[must_use]
    pub fn state_of(&self, at: usize) -> Option<&DscState> {
        Some(self.programs.get(at)?.scheduled()?.state())
    }
}

/// ⭐⭐ CONVERT AND RUN BOTH STAGES OVER EVERY PROGRAM OF ONE BUNDLE — see [`Bundle`].
///
/// ⛔ THE PANIC HOOK IS SILENCED FOR THE WHOLE WALK AND RESTORED AFTER — see [`QuietPanics`].
#[must_use]
pub fn schedule_bundle(ops: &[SdscOp]) -> Bundle {
    let _quiet = QuietPanics::install();
    let mut corpus = Corpus {
        programs: ops.len(),
        ..Corpus::default()
    };
    let mut programs = Vec::with_capacity(ops.len());
    for op in ops {
        let ran = run_stages(op);
        match &ran {
            Ran::NotConverted => {}
            Ran::Stopped(stop) => *corpus.stops.entry(stop.clone()).or_insert(0) += 1,
            Ran::Scheduled(scheduled) => corpus.absorb(scheduled),
        }
        programs.push(ran);
    }
    Bundle { corpus, programs }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scaledLdsCategory_`'s declared `REGULAR_TENSOR` — read only by the assertion that scratchy
    /// emits no such field, so it is imported HERE and not beside the lib's own seam row.
    use deeptools::sdsc::ScaledLds;

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
    /// LX allocations and their transfers, and the stage runs THROUGH the memory tracker and stops at
    /// entry 222's arena lookup. Both counts and the per-kind census are asserted, so a regression to
    /// a refusal AND a regression to an unexpectedly-different stop both fail here.
    ///
    /// ⛔ THE ABSENCE OF A PANIC IS ASSERTED BY VALUE — `stopped_at == None`. A `todo!` reappearing
    /// anywhere on this path fails here, which is the ratchet in the other direction now that the
    /// tracker no longer stops the stage, exactly as `schedule::stages`' own fixture test is.
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
        // ⭐⭐ `dsName_`, `wordLength` AND `dataFormat_` AS AN IDENTITY OVER THE WIRE ENTRY BESIDE IT —
        // not against a transcribed constant, so a conversion that read the WRONG entry (or the same
        // entry three times) fails here.
        for (converted, wire_lds) in dsc.labeled_ds.iter().zip(&wire_dsc.labeledDs_) {
            assert_eq!(
                converted.record().name,
                StorageName(wire_lds.dsName_.clone()),
                "`dsName_` verbatim — this is the name the reference's own seed allocate node carries \
                 (`allocate-Tensor0_hbm`)"
            );
            assert_eq!(
                converted.record().word_length,
                WordLength(wire_lds.wordLength),
                "`wordLength` verbatim — an ELEMENT WIDTH the DDL match compares against a template's \
                 own `bitSize_ / 8`"
            );
            assert_eq!(
                converted.record().data_format,
                DataFormat::from_spelling(wire_lds.dataFormat_),
                "`dataFormat_` through the closed-set lookup — what the DDL match BINDS each operand \
                 by"
            );
            // ⛔ AND `scaledLdsCategory_` IS THE DECLARED `REGULAR_TENSOR` (`dsc/dscdefn.h:356`): the
            // emitter writes no such field, so this is the authority's initializer and not a
            // `SCALE_TENSOR` an `Option<MxScaleTensor>` used to collapse it with.
            assert_eq!(
                converted.scaled_category(),
                ScaledLds::Regular,
                "`scaledLdsCategory_` is emitted by nothing, which is REGULAR_TENSOR"
            );
        }
        // ⭐ AND THE THREE PER-DSC ONES. `constantInfo_` is the wire's own table (the emitter writes the
        // string `"{}"` for this op), `maskingConstId_` is its declared `-1`, and neither
        // `dimToSymbolMapping_` nor `l0TetheredMode_` is emitted at all.
        assert_eq!(
            dsc.ddc.constants.len(),
            constant_info_of(&wire_dsc.constantInfo_)
                .expect("the emitter's own table reads")
                .len(),
            "`constantInfo_` converts whichever of its two wire spellings this op wrote"
        );
        assert_eq!(
            dsc.ddc.masking_const,
            match wire_dsc.maskingConstId_ {
                -1 => None,
                id => Some(ConstIdx(u32::try_from(id).expect("a non-negative id"))),
            },
            "`maskingConstId_` — `None` IS the declared -1"
        );
        assert!(
            dsc.ddc.dim_to_symbol.is_empty(),
            "`dimToSymbolMapping_` is a scheduler OUTPUT the emitter drops \
             (lower_subtile_tape_to_superdsc.rs:1240-1241), so `{{}}` is its input state"
        );
        assert_eq!(
            dsc.ddc.l0_tethered,
            L0Tethered::Split,
            "`l0TetheredMode_` likewise — `false` is what nothing set"
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

        // ⭐⭐ AND NOW THE STAGES, OVER THAT SAME CONVERSION.
        let ran = run_stages(&wire);
        let scheduled = ran
            .scheduled()
            .expect("the same conversion, run through both scheduler stages");
        let effect = scheduled.scheduling.ran();
        // ⭐⭐⭐ THE COMPUTE-OP LIST IS NON-EMPTY AND CARRIES REAL RESOLVED VALUES — the whole point,
        // asserted BEFORE the node counts because it is the fact the counts depend on. A zero here
        // beside a `DscFilled::Yes` is stage 2b's `continue`, which is a stage that did nothing.
        assert_eq!(
            scheduled.compute_ops, wire_dsc.computeOp_.len(),
            "every `computeOp_` entry the emitter wrote reached stage 2b — a SHORT list would skip \
             ops and an EMPTY one would skip the DSC entirely (`ddc/v1.rs:6437`)"
        );
        assert_eq!(
            effect.nodes_before, 4,
            "the seed: `root_level_operations` plus one HBM allocate per HBM-pinned tensor"
        );
        // ⛔⛔ THE CENSUS IS STRICTLY GREATER THAN THE SEED, AND THAT IS THE TEST THIS DEFECT NEEDED.
        // The stages ran over the tree this `Scheduling` HOLDS, so a composition that scheduled a
        // throwaway copy — the defect: `census` absorbed the numbers and dropped the artifacts —
        // would leave `state()` at the seed and fail here. "It ran" would not.
        assert!(
            effect.nodes_after > effect.nodes_before,
            "the stages MINTED nodes on the tree this value carries — {} -> {}",
            effect.nodes_before,
            effect.nodes_after,
        );
        assert_eq!(
            scheduled.scheduling.state().node_count(),
            effect.nodes_after,
            "and the count is read off the SAME state the caller now holds, not off a copy the \
             composition kept to itself"
        );
        assert_eq!(
            scheduled.kinds.values().sum::<usize>(),
            effect.nodes_after,
            "the per-kind census must account for every node, or one of the two is short"
        );
        assert_eq!(
            scheduled.kinds.get(&NodeKind::Block),
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
            scheduled.kinds.get(&NodeKind::Loop),
            Some(&order.len()),
            "one chunk loop per DISTINCT layout dim across `labeledDs_` — which is
             `collect_all_dimensions_for_loop_order`, and four here where `rmsq_o728`'s single \
             OUTPUT role gives three"
        );
        // ⭐⭐ NOTHING PANICS. The memory tracker is REAL — `stages::Trackers` owns a `MemTrackBundle`
        // of ported `DsTrackInMem`s, gated against the reference's own addresses for all 187 programs
        // (`deeptools`'s `schedule/stages/carriers/lx_oracle.rs`) — so this program runs THROUGH
        // `backup`/`remove`/`check_and_add` and reaches no `todo!` at all. A `Ran::Stopped` would have
        // failed the `let` above, which is that assertion carried by the type.
        //
        // ⭐⭐⭐ THE CAPACITY QUESTION IS ANSWERED AND THE LX PLACEMENTS ARE COMMITTED. `L3Placement::
        // buffer_capacity_even_sticks` now walks `DesignSpaceConfig::getBufferCapacityForNode`
        // (`dsc/dsc2.cpp:3977`) through `deeptools`' `l3::capacity::buffer_capacity` on the very
        // `&DesignSpaceConfig` the reference calls it on, so entry 222's `alloc_all_mem` — *the one
        // committing call of the whole stage* — RAN and wrote every LX start address and buffer offset
        // onto `memOrg_.allocateNode_`. `refusals()` is EMPTY: NO carrier of `deeptools`' `stages` was
        // asked for a fact it could not give, where the whole corpus used to stop on this one.
        assert_eq!(
            scheduled.scheduling.state().refusals(),
            Vec::<&str>::new(),
            "no stage-2a carrier refuses on this program any more — the capacity carrier was the \
             last one that did, on every program of the corpus"
        );
        // ⛔⛔ SO THE STOP MOVED, AND WHERE IT MOVED TO IS A DROPPED-EFFECT DEFECT OF THE SAME SHAPE
        // ENTRY 222 JUST SHED. The frontier is now entry 333's `fillDataInfo`
        // (`deeptools`' `schedule/l3/dl_ops.rs:17872`, `l3_fill_data_info`), whose
        // `inputs.allocs.get(&alloc)?` looks the allocation up in the `v1::AllocArena` that
        // `deeptools`' `schedule/stages.rs` hands `run` EMPTY — the map whose own comment says *"the
        // only reader left is entry 333's constant-allocation conflation, which is unreachable while
        // `Reads::offset_sizes` refuses"*. `offset_sizes` no longer refuses, so that reader IS reached,
        // on `AllocId(1)`, for the src operand of `transfer_lds0_src:hbm_dst:lx`. ⭐ THE FIX IS THE ONE
        // ALREADY APPLIED TO ENTRY 222: read the allocation off the ONE `memOrg_.allocateNode_` cell
        // (`AllocationReads` / `L3OffsetFacts`) instead of a second map keyed by identity.
        //
        // ⛔ EVERY STEP BEFORE IT RAN, MEASURED BY TRACING `run` STATEMENT BY STATEMENT:
        // `set_lx_buffer_type`, `create_chunk_loops`, `create_allocation_and_transfer`,
        // `set_chunk_data_stage_params`, `set_super_chunk_data_stage_params`,
        // `optimize_hbm_lds_output_in_schedule_tree`, `optimize_hbm_transfers`,
        // `create_synchronization`, `alloc_all_mem`, `fill_transfer_zero_padding_info`,
        // `fill_transfer_multicast_info`, `fill_allocation_start_addr_and_offset`, and entry 333's own
        // `offset_sizes`/`offset_nodes`/`offset_facts`.
        assert!(
            !effect.l3,
            "so stage 2a does not complete YET — see above for where it stops"
        );
        assert_eq!(
            effect.first_refusal, None,
            "and the stop is a PORTED unit's own `None`, not a carrier refusal — which is what makes \
             the next work item entry 333's arena rather than a vendor unit to port"
        );
        // ⛔⛔ SO STAGE 2B IS NOT **REACHED** ON THIS PROGRAM, AND THAT IS STATED AS A CONSEQUENCE
        // RATHER THAN AS THE PORT'S BEHAVIOUR. `run_stages_2a_2b` gates stage 2b on stage 2a
        // completing, because `run_l3`'s `None` leaves entry 333's operands unfilled and stage 2b
        // computes offsets FROM them — running it over an abandoned tree would compute addresses from
        // half a fill. When entry 333 reads the placed allocation off `memOrg_`, `l3` becomes true and
        // stage 2b runs with the list asserted above; nothing else has to change.
        assert_eq!(
            (effect.ddc, scheduled.scheduling.filled()),
            (false, None),
            "stage 2b is gated on stage 2a completing, and `l3` is false — so this is NOT REACHED, \
             which `filled() == None` says and a `DscFilled::Yes` would contradict"
        );
        assert_eq!(
            scheduled.scheduling.ddc_refusal(),
            None,
            "and no stage-2b carrier was asked anything at all"
        );
    }

    /// ⭐⭐⭐ THE `computeOp_` OPERAND NAMES RESOLVE TO `LdsIdx` VALUES, AND THE VALUES ARE THE
    /// ANSWER — the seam whose empty answer is stage 2b's `continue` (`ddc/v1.rs:6437-6438`).
    ///
    /// ⛔⛔ A STALE COMMENT IN THIS FILE CLAIMED THIS MAPPING DID NOT EXIST — *"resolving it needs the
    /// name→position mapping the emitter does not yet write down"*. It does: `ldsIdx_` and `dsName_`
    /// are on the SAME wire record (`lower_subtile_tape_to_superdsc.rs:1043-1045`) and the operand
    /// spelling is those two composed, `{dsName_}-idx{ldsIdx_}`.
    ///
    /// ⭐⭐ THE ANSWER KEY IS THE AUTHORITY'S, NOT OURS. `g0/sdsc_0.json` — the input
    /// `dxp_standalone` compiles to a working `init_binary` — pairs `dsName_` `"Tensor0"` with
    /// `inputLabeledDs` `"Tensor0-idx0"`, and `g0/debug/sdsc_0/sdsc.json` carries the SAME operand
    /// names after both stages have run. So the composed spelling is what stage 2b reads.
    ///
    /// ⛔ AND THE NEGATIVE CONTROL IS THE POINT OF THE TEST, not decoration: a resolver that fell back
    /// to index 0, to positional order, or that skipped the operand would pass an "it resolved" check
    /// and fail these value assertions.
    #[test]
    fn an_operand_name_resolves_to_its_own_lds_index_and_an_unknown_one_stops() {
        use crate::ir::bridge::tiled_op_sdsc_op::matmul::opspec::matmul_opspec;
        use crate::lower_subtile_tape_to_superdsc::emit_sdsc;
        use scratchy_subtile::superdsc_opspec::SdscFoldSet;

        let op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2")
            .expect("a 384x384x64 matmul is a shape the emitter builds");
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let wire = emit_sdsc("MatMul_0", &op, &folds, None).expect("the emitter lowers it");
        let wire_dsc = wire
            .dscs_
            .iter()
            .flat_map(BTreeMap::values)
            .next()
            .expect("the emitter writes one `dscs_` entry");

        let ops = dsc_compute_ops(wire_dsc).expect("every operand name resolves");
        assert_eq!(
            ops.len(),
            wire_dsc.computeOp_.len(),
            "one `DscComputeOp` per `computeOp_` entry — a dropped one is an op stage 2b never sees"
        );
        assert!(
            !ops.is_empty(),
            "an EMPTY list makes `run_v1` `continue` past the DSC and answer `DscFilled::Yes` \
             having done nothing — the false green this whole conversion exists to avoid"
        );

        // ⭐⭐ THE VALUES, AGAINST THE WIRE'S OWN `ldsIdx_` — an IDENTITY over the record beside it,
        // so a resolver keyed on the wrong field fails here rather than resolving plausibly.
        let by_name: BTreeMap<&str, u32> = wire_dsc
            .labeledDs_
            .iter()
            .map(|lds| (lds.dsName_.as_str(), lds.ldsIdx_))
            .collect();
        for (converted, wire_op) in ops.iter().zip(&wire_dsc.computeOp_) {
            for (named, resolved) in [
                (&wire_op.inputLabeledDs, &converted.inputs),
                (&wire_op.outputLabeledDs, &converted.outputs),
            ] {
                assert_eq!(
                    named.len(),
                    resolved.len(),
                    "every operand resolves — a SKIPPED one changes the reduction sweep's arity"
                );
                for (name, lds) in named.iter().zip(resolved) {
                    // The spelling the emitter wrote, split back into the two facts it composes.
                    let (base, idx) = name
                        .rsplit_once("-idx")
                        .expect("the emitter writes `{dsName_}-idx{ldsIdx_}`");
                    let expected = *by_name
                        .get(base)
                        .expect("the base name is a `labeledDs_` entry's own `dsName_`");
                    assert_eq!(
                        idx.parse::<u32>().ok(),
                        Some(expected),
                        "`{name}`'s own suffix IS `{base}`'s `ldsIdx_` — the suffix is the index \
                         restated, which is why the key carries both"
                    );
                    assert_eq!(
                        *lds,
                        LdsIdx(expected),
                        "`{name}` resolved to the labelled DS that answers to it — NOT index 0, not \
                         the operand's position in the list"
                    );
                }
            }
            assert_eq!(
                converted.op_func,
                OpFunc::from_spelling(&wire_op.opFuncName),
                "`opFuncName` through the closed set — `None` is `OpFuncs::NONE`, never a neighbour"
            );
            assert_eq!(
                converted.ex_unit,
                ex_unit_of(wire_op.exUnit).expect("`pt` or `sfp`"),
                "`exUnit` through `SenComponent::spelling` — what `usePt` reads"
            );
            assert_eq!(
                converted.format,
                DataFormat::from_spelling(wire_op.attributes_.dataFormat_),
                "`attributes_.dataFormat_` through the same closed-set lookup `labeled_of` uses"
            );
        }

        // ⛔⛔ THE NEGATIVE CONTROL: A NAME NOTHING ANSWERS TO IS A STOP. Not index 0, not a skip.
        let mut broken = wire_dsc.clone();
        broken.computeOp_[0].inputLabeledDs[0] = "Tensor0-idx7".to_owned();
        assert_eq!(
            dsc_compute_ops(&broken),
            None,
            "`Tensor0-idx7` names no labelled DS — resolving it to `Tensor0`'s real index would put \
             the operand on the wrong DS, which is a wrong extent and then a wrong address"
        );
        let mut renamed = wire_dsc.clone();
        renamed.computeOp_[0].exUnit = "pe";
        assert_eq!(
            dsc_compute_ops(&renamed),
            None,
            "`pe` is a real `SenComponent` but not one the emitter's own `ex_unit` producer can \
             write, so it must STOP rather than resolve — a wrong `usePt` splits the wrong dim"
        );

        // ⭐ AND THE PER-DSC NAME IS THE `dscs_` MAP KEY, READ — not the positional `dsc0` fallback.
        assert_eq!(
            dsc_names(&wire),
            vec![StorageName("MatMul_0".to_owned())],
            "`dsc.name_` is the key the emitter states, which is what `Dsc2Fill::said` prints"
        );
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
