// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE FOUR SCHEDULER STAGES, COMPOSED AND CALLABLE — `dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-57`.
//!
//! Campaign 6 ported 382 units of these stages and NOTHING CALLED ANY OF THEM. Every entry point
//! takes its state through trait parameters — `L3RunInputs<F, P>` / `L3RunSurgery<E, M, K, S>` for
//! stage 2a, `Dsc2Sites` for stage 2b — and no concrete type implemented them, so `F` and `E` were
//! uninhabited and the stages were dead code. THIS MODULE IS THOSE CONCRETE TYPES. Writing them is
//! the integration, not a second port: the bodies exist, this is what reaches them.
//!
//! # THE ORDER, AND WHY EACH STAGE IS OR IS NOT HERE
//!
//! 1. `sbf::doCoreletSplitSdsc` — ⛔ EXCLUDED, and not by us: `SchedulerStages.cpp:25` returns early
//!    unless `numCoreletsPerCore == 2`, and scratchy emits `numCoreletsUsed_ = 1` in all 313 sampled
//!    SuperDSCs. Recorded in `crustify-ddc/EXCLUSIONS.tsv`.
//! 2. `L3DlOpsScheduler::run` — [`l3::dl_ops::run`], `e382_run`.
//! 3. `ddc::Ddc::run_v1` — [`ddc::v1::run_v1`], `e379_run_v1`. THE STAGE THAT PLACES ADDRESSES.
//! 4. `DcgManager::runDcgForDlOpsStandalone` — ⛔ NOT WIRED, ON PURPOSE. Its body delegates to
//!    `dcg_fe/pcfg_gen/` and `dcg_be/`, both declared out of campaign 6's scope, and it fills
//!    `pcfg_[core][L3LU]/[L3SU]` — the PCFG path this project has ruled off, because our DL path
//!    emits those units directly (`bridges/superdsc_to_dataflow_ir/transfer.rs` carries 61 L3LU/L3SU
//!    references). Porting `pcfg_gen` to reach it is forbidden, so it stays a `todo!` inside
//!    `dcg::manager` and this module does not call it.
//!
//! # ⛔⛔ THE ONE STRUCTURAL FACT A CALLER MUST WORK AROUND — REVIEW 382'S OWN NOTE
//!
//! `L3RunInputs::reads` is `&'a F` and `L3RunSurgery::env` is `&'a mut E`. A shared and an exclusive
//! borrow cannot view ONE object, so the reference's single `this` is SEVERED by the port's own
//! signature — and its review says so at `l3/dl_ops.rs:20556-20560`: the placed address (`:5687`
//! writes, `:4971`/`:5034` rewrite, `:5816` reads back) and the minted tree (290/353/368 write
//! `env`; 289/295/332 read `reads`) are both chains that cross the cut.
//!
//! ⭐ SO THE TWO CARRIERS SHARE ONE STATE THROUGH A CELL, WHICH IS THE HONEST REPAIR. [`Reads`] and
//! [`Env`] are two objects, as the signature demands, each holding `&'s DscState` — and the state's
//! mutable interior is what restores the reference's aliasing. It is NOT a workaround that changes
//! behaviour: it makes the two views name the same tree and the same placed address, which is what
//! the reference does and what any faithful caller must do. Unifying the carriers in the port itself
//! is the cross-entry work review 382 names over entries 050/219/220/222/292/333; until that lands,
//! the cell is what a caller has.
//!
//! # ⛔ NO FABRICATION, ANYWHERE
//!
//! Every provider method answers from the SuperDSC it was given. `dxp_standalone` and `dbo-opt` run
//! THESE SAME FOUR STAGES over scratchy's SuperDSC today and produce a working `init_binary`
//! (`RunSchedulerOnSdsc.cpp:144-170`), so every fact these methods are asked for is already present
//! in what scratchy writes — there is no data gap, by construction. A method that cannot yet be
//! answered is a `todo!` NAMING THE TRAIT AND METHOD, never a plausible constant: a fabricated
//! placement is the failure this crate ranks worse than a stop.

mod carriers;
mod ddc_reads;
mod ddc_sites;
mod ddc_state;
mod ddc_store;
mod ddc_store2;
mod ddc_tree;
mod env;
mod offsets;
mod reads;
mod state;
mod tree;

pub use carriers::{ExecutionStep, LxAvailFraction, Placement, Sink, Symbols, Trackers};
pub use ddc_reads::Dsc2Reads;
pub use ddc_sites::{DdcTemplates, Dsc2Ddl, Dsc2Provider, Dsc2Stages};
pub use ddc_state::{DdcCoords, DdcSink, DdcSymbols, DdcTrackers, Dsc2Dims, Dsc2State};
pub use ddc_store::{Dsc2LdsEntry, Dsc2Store};
pub use ddc_tree::Dsc2Tree;
pub use env::Env;
pub use reads::Reads;
pub use state::{DscState, DscTree};

use super::ddc::v1;
use super::l3;
use crate::arch::Dd2;

/// ⭐ WHAT STAGE 2A LEFT — the answer the caller measures, and the reason this returns a value rather
/// than printing one.
///
/// ⛔ [`None`] IS A CALLEE'S OWN STOP, propagated, not an error we invent: `l3::dl_ops::run` answers
/// `Option<()>` and every `None` in it is a ported unit's refusal or *"Memory allocation must be
/// valid to commit."*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StagesRan {
    /// Whether stage 2a completed.
    pub l3: bool,
    /// Whether stage 2b completed.
    pub ddc: bool,
    /// How many schedule-tree nodes every DSC held BEFORE the stage — the seed.
    pub nodes_before: usize,
    /// How many they hold AFTER it, whether or not it completed. ⭐ THE EFFECT, WHICH IS A VALUE AND
    /// NOT A RATIO: a stage that refuses part-way still leaves the nodes its growers minted.
    pub nodes_after: usize,
    /// The FIRST provider method that refused, [`None`] where none did — the one fact that decides
    /// what happens next.
    pub first_refusal: Option<&'static str>,
}

/// ⭐⭐ STAGE 2A, COMPOSED AND CALLED — builds the seed state, the six carriers, and hands them to
/// [`l3::dl_ops::run`].
///
/// ⛔ `CHUNK_EXPLORE` IS `false` HERE, which is the reference with `DISABLE_ABOVE_LX_CHUNK_EXPLORE`
/// SET: with it the datastage exploration of entries 366/367 is skipped, and so is the
/// [`l3::dl_ops::SysFlopsPerByte`] system table only it reads. Call [`run_l3`] with `true` to explore.
///
/// ⛔ `LxBufferTypeMode::Auto` IS THE SCHEDULER'S OWN DEFAULT and is what the heuristic decides
/// through; every program of scratchy's own staged bundle came out DOUBLE.
#[must_use]
pub fn run_stages(sdsc: &mut l3::dsc::SuperDsc) -> StagesRan {
    let state = DscState::seeded(sdsc);
    run_stages_with(sdsc, &state)
}

/// ⭐⭐ [`run_stages`]' OWN BODY WITH THE SEED STATE HANDED IN — the same composition, the same
/// `computeOp_`-less [`v1::OpFuncs`] and the same flat [`l3::dl_ops::AddressFoldCoords`].
///
/// ⛔⛔ IT EXISTS SO A CALLER CAN STILL READ THE TREE AFTER THE STAGE STOPS. Stage 2a's stop today is
/// the memory tracker's `todo!` ([`carriers`]'s `ExPhaseTrackers::backup`), and a [`DscState`] built
/// INSIDE [`run_stages`] is dropped by that unwind — so the nodes the growers minted, which are the
/// whole measurement, are unreachable. A caller that owns the state measures
/// [`DscState::kinds`] afterwards.
///
/// ⭐ [`StagesRan`] IS THE SAME ANSWER EITHER WAY: it is read off the state this takes, so the two
/// entry points cannot report different numbers for one run.
#[must_use]
pub fn run_stages_with(sdsc: &mut l3::dsc::SuperDsc, state: &DscState) -> StagesRan {
    let nodes_before = state.node_count();
    let l3 = run_l3::<false, Dd2>(
        sdsc,
        state,
        // ⛔ `computeOp_` IS NOT A FIELD OF [`l3::dsc::DesignSpaceConfig`] — see
        // [`Reads::new`]'s own note. A `computeOp_` entry whose `opFuncName` is unset is what a
        // caller holding no op func can state, and the ported min-param units refuse on it exactly
        // as the reference's `.at(0)` throws. A caller that HAS the op func calls [`run_l3`].
        v1::OpFuncs::new(None, Vec::new()),
        // ⛔ `sdscFoldProps_` IS NOT A FIELD OF [`l3::dsc::SuperDsc`] EITHER, and
        // [`l3::dl_ops::AddressFoldCoords::flat`] is *"the ONE coordinate a super-DSC declaring no
        // folds of its own has"* — which is literally true of the value in hand. A caller holding
        // the SDSC's own fold props (`0_rmsq_o728` declares one `time` axis of factor 1) builds them
        // with [`l3::dl_ops::AddressFoldCoords::of`] and calls [`run_l3`].
        &l3::dl_ops::AddressFoldCoords::flat(),
    )
    .is_some();
    StagesRan {
        l3,
        ddc: false,
        nodes_before,
        nodes_after: state.node_count(),
        first_refusal: state.first_refusal(),
    }
}

/// ⭐ THE SAME STAGE WITH ITS TWO CONSTRUCTION ARGUMENTS EXPOSED — the compute ops and the fold
/// manager's own address coordinates, both of which the reference scheduler is HANDED rather than
/// reading off the super-DSC.
///
/// ⛔ [`None`] IS A CALLEE'S REFUSAL, propagated; `state.refusals()` says which provider made it.
pub fn run_l3<const CHUNK_EXPLORE: bool, A: crate::arch::Arch>(
    sdsc: &mut l3::dsc::SuperDsc,
    state: &DscState,
    ops: v1::OpFuncs,
    coords: &l3::dl_ops::AddressFoldCoords,
) -> Option<()> {
    let reads = Reads::new(state, ops);
    let placement = Placement::of(coords.clone());
    let inputs = l3::dl_ops::L3RunInputs {
        reads: &reads,
        placement: &placement,
        lx_buffer_mode: l3::dl_ops::LxBufferTypeMode::Auto,
        coords,
    };
    let mut env = Env::new(state, sdsc);
    let mut allocs = v1::AllocArena::new();
    // ⭐⭐ THE REAL LX ALLOCATOR, ONE TRACKER PER CORE OVER `A::LX_CAPACITY` AT `A::BYTES_PER_STICK`,
    // with the front end's share pinned at address 0 — see [`Trackers::at_step`]. `ExecutionStep`'s
    // default is `dbo::execStepOf`'s own `0` for a bundle that stamped no `sbf.exec_step`, which is
    // what a single-phase scratchy bake is.
    let mut trackers = Trackers::at_step::<A>(ExecutionStep::default());
    let mut sink = Sink::default();
    let mut symbols = Symbols::default();
    let mut surgery = l3::dl_ops::L3RunSurgery {
        env: &mut env,
        allocs: &mut allocs,
        trackers: &mut trackers,
        sink: &mut sink,
        symbols: &mut symbols,
    };
    l3::dl_ops::run::<CHUNK_EXPLORE, A, _, _, _, _, _, _>(sdsc, &inputs, &mut surgery)
}

/// ⭐⭐ STAGE 2B, COMPOSED AND CALLED — [`run_l3`]'s PEER: it builds the provider over a
/// [`Dsc2State`] and hands it to [`v1::run_v1`].
///
/// ⛔⛔ THE `computeOp_` LIST IS A CONSTRUCTION ARGUMENT AND THAT IS NOT OPTIONAL.
/// [`v1::PrepDsc::compute_ops`] is the FIRST provider call `run_v1` makes (`ddc/v1.rs:6437`) and an
/// EMPTY answer makes it `continue` past the DSC — so a caller that stated no ops would get
/// [`v1::DscFilled::Yes`] having done nothing at all, which is a false green. `computeOp_` is not a
/// field of [`l3::dsc::DesignSpaceConfig`], so the caller states it, exactly as [`run_l3`] is handed
/// [`v1::OpFuncs`].
///
/// ⛔ `DSC_TO_DDL` IS `false`: entry 345's export is behind `dscToDdl_`, which only
/// `ddc/ddc_standalone.cpp:40` sets.
///
/// ⛔ [`None`] IS EVERY CALLEE'S OWN STOP plus a `dscs_.at()` the provider does not hold;
/// [`Dsc2State::first_refusal`] says whether a CARRIER made it.
pub fn run_ddc<A: crate::arch::Arch>(
    sdsc: &mut l3::dsc::SuperDsc,
    state: &Dsc2State<'_>,
    options: v1::Dsc2Options,
    coords: &l3::dl_ops::AddressFoldCoords,
) -> Option<v1::Dsc2Fill> {
    let mut sites = Dsc2Provider::new(state, coords);
    let templates = DdcTemplates::new(state);
    let mut shuffle_names = super::ddc::transformation::AutoShuffleNames::default();
    v1::run_v1::<A, _, _, false>(sdsc, &mut sites, &templates, &mut shuffle_names, options)
}

/// ⭐ STAGE 2B'S OWN CONSTRUCTION ARGUMENTS AT THEIR REFERENCE DEFAULTS — the five ambient reads
/// entry 379's callees take, each cited on [`v1::Dsc2Options`]'s own fields.
///
/// ⛔ EVERY ONE IS THE `dtGetEnv`/`value_or` DEFAULT AND NOT A CHOICE MADE HERE: `ENABLE_LN32` unset
/// is [`v1::Ln32::Off`], `trueLXTracker_` false is [`v1::LxTrackers::Ephemeral`],
/// `allowUnpaddedIndexingAtPaddedNoZeroPad` false is [`v1::UnpaddedIndexing::Forbidden`],
/// `verifyCoordinateBasedLoopElemOff` unset leaves the latch at [`v1::ElemOffsets::Distribution`],
/// and `coordFoldReportLevel_` `0` is [`crate::schedule::ddc::fold::CoordFoldReport::Off`].
#[must_use]
pub const fn ddc_defaults() -> v1::Dsc2Options {
    v1::Dsc2Options {
        ln32: v1::Ln32::Off,
        lx: v1::LxTrackers::Ephemeral,
        unpadded: v1::UnpaddedIndexing::Forbidden,
        offsets: v1::ElemOffsets::Distribution,
        report: crate::schedule::ddc::fold::CoordFoldReport::Off,
    }
}

#[cfg(test)]
mod tests {
    // ⭐⭐ THE ANSWER KEY IS THE REFERENCE PIPELINE'S OWN OUTPUT ON SCRATCHY'S OWN STAGED BUNDLE:
    // `/Users/nickm/tmp/bridge1-fixtures/g0/sdsc_0.json` in, `.../debug/sdsc_0/sdsc.json` out. This
    // fixture is `0_rmsq_o728`'s ONE DSC transcribed field for field, and every count and name
    // asserted below is READ OFF THAT OUTPUT — not off ours.
    //
    // ⛔ WHICH OF THE OUTPUT'S 30 NODES ARE STAGE 2A'S. Twenty-two: the four the input already has
    // (`root_level_operations` and `allocate-Tensor{0,1,2}_hbm`), the three chunk loops
    // (`loop_ds0_ds1_{y,out,mb}`), the `lx_below_schedule` block, `allocate_lds{0,1,2}_lx`,
    // `transfer_lds{0,1}_src:hbm_dst:lx`, `transfer_lds2_src:lx_dst:hbm` and the eight
    // `sync_*_l3{lu,su}_*_lx{lu,su}` nodes. The other eight — `loop_ds1_ds2_out_mb_y`, the three
    // `loop_ds2_ds3_out_mb_y*`, the three `src:sfp`/`dst:sfp` transfers and `compute_sfp_fma16` —
    // are stage 2b's, which this module does not yet compose.

    use super::*;

    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU32;

    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PrimaryDim, StickDims,
    };
    use crate::schedule::ddc::transformation::{DsType, Scale};
    use crate::schedule::ddc::transformation_util::StageName;
    use crate::schedule::dsc2::{LayoutDims, LdsIdx, WordLength};
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DesignSpaceConfig, DscList, FilledDims, LabeledDs,
        LabeledDsList, LdsRecord, NamedDims, Pinning, PrimaryDsInfo, StageDims, SuperDsc,
        WkSliceCount,
    };
    use crate::units::Core;
    use sys_arch_spec::arch_enums::SenComponent;

    /// `numCoresUsed_ = 32`, `coreIdsUsed_ = 0..31`.
    const CORES: u32 = 32;

    /// `primaryDsInfo_.at(OUTPUT).layoutDimOrder_` — `["mb", "out", "y"]`.
    const LAYOUT: [PrimaryDim; 3] = [PrimaryDim::Mb, PrimaryDim::Out, PrimaryDim::Y];

    /// `dataStageParam_.at("0").ss_` — `{out_: 64, mb_: 1, y_: 1}`.
    const CORE_EXTENTS: [(PrimaryDim, i64); 3] = [
        (PrimaryDim::Out, 64),
        (PrimaryDim::Mb, 1),
        (PrimaryDim::Y, 1),
    ];

    fn dims(extents: &[(PrimaryDim, i64)]) -> FilledDims {
        let mut stage = StageDims::default();
        for &(dim, extent) in extents {
            stage.extents.insert(dim, Extent(extent));
        }
        FilledDims::of(stage).expect("a stage that states a dim")
    }

    fn stage(name: &str, extents: &[(PrimaryDim, i64)]) -> DataStage {
        let name = StageName(name.to_owned());
        DataStage {
            ss: NamedDims {
                name: name.clone(),
                dims: dims(extents),
            },
            el: NamedDims {
                name,
                dims: dims(extents),
            },
        }
    }

    /// `memOrg_` on every one of `rmsq_o728`'s three tensors — `hbm` AND `lx`, both present.
    fn pinning() -> Pinning {
        Pinning {
            mem_org: BTreeMap::from([(SenComponent::Hbm, true), (SenComponent::Lx, true)]),
            lx: false,
            lx_padded: false,
        }
    }

    /// One `labeledDs_` entry — `dsType_ = OUTPUT`, `scale_ = [1, 1, 1]` over the layout order.
    ///
    /// ⭐ `dsName_`, `wordLength` AND `dataFormat_` ARE THE FIXTURE'S OWN, TRANSCRIBED:
    /// `g0/sdsc_0.json`'s three entries are `Tensor{0,1,2}` / `2` / `SEN169_FP16`, and the reference's
    /// own output names its seed allocate nodes `allocate-Tensor{N}_hbm` off exactly that `dsName_`.
    ///
    /// ⛔ `scaledLdsCategory_` IS ABSENT FROM ALL 580 LABELLED DSs OF `g0/`, which is the declared
    /// `REGULAR_TENSOR` (`dsc/dscdefn.h:356`) — [`LabeledDs::new`]'s own state.
    fn labeled(recorded: LdsIdx) -> LabeledDs {
        LabeledDs::new(
            DsType::Output,
            LAYOUT.iter().map(|&dim| (dim, Scale::Sized(1.0))).collect(),
            recorded,
            pinning(),
        )
        .with_record(LdsRecord {
            name: v1::StorageName(format!("Tensor{}", recorded.0)),
            word_length: WordLength(2),
            data_format: Some(crate::formats::DataFormat::Sen169Fp16),
        })
    }

    /// ⭐ `0_rmsq_o728`'s ONE DSC, transcribed from `g0/sdsc_0.json`.
    ///
    /// ⛔ THE CHUNK STAGE IS SYNTHESISED FROM THE CORE ONE, AND THAT IS THE REFERENCE'S OWN MOVE:
    /// scratchy emits a single `dataStageParam_["0"]`, `dxp`'s `SdscCoreletSplit.cpp:70` asserts that
    /// size and synthesises the chunk stage from it, and `ddl_conversion.cpp:2585-2591` is how.
    /// [`crate::schedule::l3::dsc::DataStages::new`] demands both, so the synthesis happens here.
    fn a_rmsq_dsc() -> DesignSpaceConfig {
        let cores: Vec<Core> = (1..CORES).filter_map(Core::checked).collect();
        DesignSpaceConfig {
            // ⭐ THE FIXTURE'S OWN FOUR, AND EVERY ONE IS AT THE AUTHORITY'S INITIALIZER:
            // `"constantInfo_": "{}"` (EMPTY), `"maskingConstId_": -1` ([`None`]), and neither
            // `dimToSymbolMapping_` nor `l0TetheredMode_` is emitted at all — the emitter's own note
            // calls both scheduler OUTPUTS (`lower_subtile_tape_to_superdsc.rs:1240-1241`).
            ddc: crate::schedule::l3::dsc::DdcFacts::default(),
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: Some(CoreletsUsed::ONE),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::from([(
                DsType::Output,
                PrimaryDsInfo {
                    layout: LayoutDims::new(LAYOUT[0], LAYOUT[1..].to_vec()),
                    stick: StickDims::default(),
                },
            )]),
            core_ids_used: CoreIdsUsed::new(Core::checked(0).expect("core 0"), cores),
            layout_dims: (0u32..3)
                .map(|at| (LdsIdx(at), LayoutDims::new(LAYOUT[0], LAYOUT[1..].to_vec())))
                .collect(),
            labeled_ds: LabeledDsList::new(
                labeled(LdsIdx(0)),
                vec![labeled(LdsIdx(1)), labeled(LdsIdx(2))],
            ),
            data_stages: crate::schedule::l3::dsc::DataStages::new(
                stage("core", &CORE_EXTENTS),
                stage("chunk", &CORE_EXTENTS),
            ),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            gtr_ids_used: BTreeSet::new(),
        }
    }

    /// ⭐ `0_rmsq_o728` — `numWkSlicesPerDim_ = {mb: 1, out: 32, y: 1}`.
    fn a_rmsq_super_dsc() -> SuperDsc {
        SuperDsc::new(
            DscList::new(a_rmsq_dsc(), Vec::new()),
            BTreeMap::from([
                (PrimaryDim::Mb, WkSliceCount::ONE),
                (
                    PrimaryDim::Out,
                    WkSliceCount::new(NonZeroU32::new(CORES).expect("32 slices")),
                ),
                (PrimaryDim::Y, WkSliceCount::ONE),
            ]),
            BTreeMap::new(),
            BTreeMap::new(),
        )
    }

    /// ⭐ THE REFERENCE'S OWN STAGE-2A TREE FOR `rmsq_o728`, IN ITS OWN DFS ORDER — the twenty-two of
    /// `debug/sdsc_0/sdsc.json`'s thirty nodes that stage 2a owns.
    ///
    /// ⛔ THE FIRST THREE ALLOCATES ARE SPELLED `allocate-Tensor{N}_hbm` THERE. `Tensor{N}` is
    /// `labeledDs_.at(N).dsName_`, which [`DscState::seeded`] does not reach: the seed is built off the
    /// SUPER-DSC alone, and `dsName_` arrives at the scheduler through
    /// [`v1::StorageNames`] — a method on the `P` carrier that takes no
    /// [`l3::dsc::DscIdx`], so it cannot say which DSC's `labeledDs_` an index names
    /// ([`carriers::Placement`]'s own note). So the names below are positional.
    const REFERENCE_STAGE_2A_TREE: [&str; 22] = [
        "root_level_operations",
        "allocate_lds0_hbm",
        "allocate_lds1_hbm",
        "allocate_lds2_hbm",
        "loop_ds0_ds1_y",
        "loop_ds0_ds1_out",
        "loop_ds0_ds1_mb",
        "allocate_lds0_lx",
        "transfer_lds0_src:hbm_dst:lx",
        "allocate_lds1_lx",
        "transfer_lds1_src:hbm_dst:lx",
        "sync_send_l3lu_to_lxlu",
        "sync_receive_lxlu_from_l3lu",
        "sync_send_lxlu_to_l3lu",
        "sync_receive_l3lu_from_lxlu",
        "allocate_lds2_lx",
        "lx_below_schedule",
        "sync_send_lxsu_to_l3su",
        "sync_receive_l3su_from_lxsu",
        "sync_send_l3su_to_lxsu",
        "sync_receive_lxsu_from_l3su",
        "transfer_lds2_src:lx_dst:hbm",
    ];

    /// ⭐⭐ THE ONE THING THAT STOPS STAGE 2A ON SCRATCHY'S OWN SUPER-DSC, ISOLATED: entry 207's own
    /// RECORDED DIVERGENCE. `generate_dsc_param_candidates` refuses *"a non-chunk dim the core stage
    /// does not state"* where the reference records its single candidate of `-1`; `explored_primary_dims`
    /// is `[In, Out, Mb, X, Y, I, J, Ki, Kj, X1]` and `rmsq_o728`'s core stage states three of them
    /// (`out`, `mb`, `y`), so the other SEVEN each refuse.
    ///
    /// ⛔ NOT A PROVIDER GAP: `DscState::refusals()` is EMPTY on this run — no carrier of
    /// [`super::stages`] was asked for a fact it could not give.
    /// ⭐ THE GROWERS OF ENTRY 382, IN ITS OWN ORDER, UP TO THE STEP NAMED — every unit
    /// [`l3::dl_ops::run`] calls before it, with the same arguments.
    fn grow<'s>(sdsc: &l3::dsc::SuperDsc, state: &'s DscState) -> (Reads<'s>, Env<'s>) {
        use crate::schedule::l3::dl_ops as ops;
        use crate::schedule::l3::dsc::{DscIdx, MemOrgs};

        let reads = Reads::new(state, an_op_func());
        let mut env = Env::new(state, sdsc);
        let choice = ops::set_lx_buffer_type::<Dd2, _>(sdsc, ops::LxBufferTypeMode::Auto, &reads)
            .expect("the heuristic picks a buffering");
        assert_eq!(
            choice,
            ops::LxBufferChoice::Double,
            "the reference chose DOUBLE: one `(ds0, ds1)` loop band per order dim"
        );
        let orgs: Vec<&tree::Org> = sdsc
            .dscs()
            .first()
            .labeled_ds
            .indexed()
            .map(|(at, _)| {
                MemOrgs::mem_org(&reads, DscIdx(0), at).expect("every labelled DS has an org")
            })
            .collect();
        let buffering = ops::create_chunk_loops(sdsc, &orgs, &mut env, choice)
            .expect("the chunk loop nest is chained");
        let mut metadata = BTreeMap::from([(DscIdx(0), ops::DscMetadata::default())]);
        ops::create_allocation_and_transfer(sdsc, &mut metadata, buffering, &reads, &mut env)
            .expect("every LX allocation and its transfer");
        (reads, env)
    }

    /// ⭐⭐ ENTRY 207 RECORDS THE REFERENCE'S `-1` FOR A DIM THE CORE STAGE DOES NOT STATE.
    ///
    /// ⛔ THIS TEST USED TO ASSERT THE OPPOSITE, and it was pinning a PORT DIVERGENCE as if it were
    /// the reference's behaviour — which is how the divergence survived: it stopped stage 2a at
    /// `set_chunk_data_stage_params` on every program scratchy emits, and a green test said that was
    /// correct. The authority has ONE `DT_CHECK` there — that `dataStageParam_` holds the core stage
    /// (`L3DlOpsScheduler.cpp:1186`) — and `primaryDimToVal_base_st` then reads the field raw
    /// (`dsc/dims.cpp:516-560`), returning the `-1` default with no throw.
    ///
    /// ⭐ IT CARRIES THE VALUES, not just the verdict: the ten explored dims all get exactly one
    /// candidate, the three the core stage states are positive, and the seven it does not are
    /// [`ops::UNSTATED_EXTENT`] — so a regression to a refusal, or to a fabricated extent, both fail.
    #[test]
    fn entry_207_records_the_unstated_extent_rather_than_refusing() {
        use crate::schedule::l3::dl_ops as ops;
        let sdsc = a_rmsq_super_dsc();
        let state = DscState::seeded(&sdsc);
        // ⭐ THE GROWERS RUN FIRST, exactly as in entry 382: entry 207 reads each labelled DS's LX
        // `memOrg_`, which entry 353 is what fills.
        let (reads, _env) = grow(&sdsc, &state);
        let main = sdsc.dscs().first();

        let chunk_dims: BTreeSet<PrimaryDim> = main
            .labeled_ds
            .iter()
            .filter_map(|lds| main.non_broadcast_lds_dims(lds.recorded()))
            .flatten()
            .collect();
        let params = vec![
            ops::initial_chunk_params::<Dd2, _>(&sdsc, main, &reads, &chunk_dims)
                .expect("every chunk dim has a positive core extent"),
        ];
        // `explored_primary_dims()` — every [`PrimaryDim`] but the two combined ones.
        let explored: Vec<PrimaryDim> = PrimaryDim::ALL
            .into_iter()
            .filter(|dim| !matches!(dim, PrimaryDim::Ij | PrimaryDim::Kij))
            .collect();
        let candidates = ops::generate_dsc_param_candidates(
            &sdsc,
            &params,
            &explored,
            &chunk_dims,
            &BTreeSet::new(),
            &reads,
            None,
        )
        .expect("entry 207 answers for every explored dim, stated or not");
        let per_dim = candidates
            .at(crate::schedule::l3::dsc::DscIdx(0))
            .expect("the one DSC's candidates");

        // ⭐ EVERY EXPLORED DIM IS ANSWERED FOR — the count, not just "it did not refuse".
        let answered = per_dim.iter().count();
        assert_eq!(
            answered,
            explored.len(),
            "entry 207 must answer for all {} explored dims; it answered {answered}",
            explored.len(),
        );

        // ⭐ AND THE ANSWER IS THE RIGHT ONE PER DIM: positive where the core stage states it,
        // exactly the reference's `-1` where it does not. A fabricated extent fails both arms.
        let stated: BTreeSet<PrimaryDim> = chunk_dims.iter().copied().collect();
        for &dim in &explored {
            let got = per_dim
                .get(dim)
                .unwrap_or_else(|| panic!("no candidate list for {dim:?}"));
            if stated.contains(&dim) {
                assert!(
                    got.extents().iter().all(|extent| extent.0 > 0),
                    "{dim:?} is stated by the core stage, so every candidate must be a real \
                     extent; got {:?}",
                    got.extents()
                );
            } else {
                assert_eq!(
                    got.extents(),
                    [ops::UNSTATED_EXTENT],
                    "{dim:?} is not stated by the core stage, so entry 207 must record the \
                     reference's single candidate of {:?}",
                    ops::UNSTATED_EXTENT
                );
            }
        }
        assert!(state.refusals().is_empty(), "no carrier was asked anything");
    }

    /// `computeOp_.at(0).opFuncName` on `rmsq_o728` — `"mul"` on the `sfp`, per `g0/sdsc_0.json`.
    fn an_op_func() -> v1::OpFuncs {
        v1::OpFuncs::new(Some(sys_arch_spec::arch_enums::OpFunc::Mul), Vec::new())
    }

    /// ⭐⭐ `rmsq_o728`'s WHOLE `computeOp_`, TRANSCRIBED FROM `g0/sdsc_0.json`:
    /// `[{exUnit: "sfp", opFuncName: "mul", attributes_.dataFormat_: "SEN169_FP16",
    /// inputLabeledDs: ["Tensor0-idx0", "Tensor1-idx1"], outputLabeledDs: ["Tensor2-idx2"]}]`.
    ///
    /// ⛔ IT IS A CONSTRUCTION ARGUMENT AND NOT A FIELD OF [`l3::dsc::DesignSpaceConfig`] — see
    /// [`run_ddc`]. ⭐ AND IT IS A REAL FIELD, NOT A CONSTANT: over the first forty fixtures the
    /// `(opFuncName, exUnit, #inputs, #outputs)` signature takes TWELVE distinct values —
    /// `mul`/`mean`/`reciprocal`/`add`/`max`/`maximum`/`sub`/`exp`/`identity`/`minimum`/`sum` on the
    /// `sfp`, plus one two-op DSC (`batchmatmul` on the `pt` then `stridedadd` on the `sfp`) — so a
    /// carrier that answered one shape for every program would be wrong on 35 of them.
    fn the_rmsq_compute_ops() -> Vec<v1::DscComputeOp> {
        vec![v1::DscComputeOp {
            op_func: Some(sys_arch_spec::arch_enums::OpFunc::Mul),
            ex_unit: SenComponent::Sfp,
            format: Some(crate::formats::DataFormat::Sen169Fp16),
            inputs: vec![LdsIdx(0), LdsIdx(1)],
            outputs: vec![LdsIdx(2)],
        }]
    }

    /// ⭐⭐ STAGE 2B RUN ON `rmsq_o728`, AND WHERE IT STOPS — the one fact this composition was built
    /// to produce.
    ///
    /// ⛔⛔ THIS IS NOT THE REFERENCE'S COMPOSITION AND THE TEST SAYS SO. The reference runs stage 2a
    /// to completion and THEN stage 2b over the tree 2a left (`SchedulerStages.cpp:29-57`). Stage 2a
    /// stops at the memory tracker on every program scratchy emits
    /// ([`stage_2a_runs_past_entry_207_and_reaches_the_memory_tracker`]), so what stage 2b is handed
    /// here is the FOUR-NODE SEED and not a grown tree. That is deliberate: stage 2b has its OWN
    /// [`v1::MemTrackers`] carrier, so running it on the seed exercises stage 2b's own units and says
    /// where **2b** stops, which is a different question from where 2a stops.
    ///
    /// ⭐ WHAT IT PROVES RAN. `run_v1` reaches this seam only after `sites.carriers(0)` handed out all
    /// ten borrows, [`v1::PrepDsc::compute_ops`] answered the op list (a non-empty answer, or the DSC
    /// would have been skipped), [`v1::CoreletShapes::corelets_used`] and
    /// [`v1::PrepDsc::set_corelets_used_dsc2`] ran, [`v1::ExploreStages::dims`] handed back the chunk
    /// snapshot, `get_pe_sfp_split_dim` walked it, the reduction sweep read every operand's
    /// non-broadcast dims and `numWkSlicesPerDim_`, and [`v1::ExploreStages::stages`] listed the map.
    ///
    /// ⛔ THE EXPECTED MESSAGE IS THE SPECIFIC SEAM, NOT ANY PANIC — the same ratchet
    /// [`stage_2a_runs_past_entry_207_and_reaches_the_memory_tracker`] is: a bare `should_panic` would
    /// pass on the first of this module's `todo!`s and so would say nothing about how far 2b got.
    #[test]
    fn stage_2b_on_the_seed_tree_stops_on_the_missing_below_lx_block() {
        let mut sdsc = a_rmsq_super_dsc();
        let l3_state = DscState::seeded(&sdsc);
        let state = Dsc2State::seeded(
            &sdsc,
            &l3_state,
            &[the_rmsq_compute_ops()],
            &[v1::StorageName("rmsq_o728".to_owned())],
        );
        let ran = run_ddc::<Dd2>(
            &mut sdsc,
            &state,
            ddc_defaults(),
            &l3::dl_ops::AddressFoldCoords::flat(),
        );

        // ⭐ THE STOP IS A PORTED UNIT'S OWN REFUSAL AND NOT A CARRIER'S, and not a `todo!` either:
        // nothing panicked, and no provider method was asked for a fact it could not give.
        assert!(ran.is_none(), "stage 2b refuses on the seed tree");
        assert!(
            state.refusals().is_empty(),
            "no carrier refused: {:?}",
            state.refusals()
        );
        assert_eq!(
            l3_state.node_count(),
            4,
            "the tree is untouched — stage 2b refused before minting anything"
        );

        // ⭐⭐ AND THIS IS *WHICH* REFUSAL, ISOLATED. `attach_to_prefilled_schedule` inserts every
        // node it VISITS into `metadata.external_nodes` before testing it, so a full set proves the
        // per-node walk refused on NONE of them — every ALLOCATE passed `holds_lds`, `is_dsc_memory`
        // and the `lds_alloc(lds, component) == alloc` round-trip through the tree's own `memOrg_`.
        // What is [`None`] is `below_lx_schedule_insert_block`, and the LAST line of that unit is
        // `metadata.below_lx_schedule_insert_block.map(|_| ())` — the reference's
        // *"Missing below-lx schedule insert block"*.
        let mut sites = Dsc2Provider::new(&state, &l3::dl_ops::AddressFoldCoords::flat());
        let mut metadata = crate::schedule::ddc::metadata::Metadata::default();
        let (attached, arena) = {
            let carriers = v1::Dsc2Sites::carriers(&mut sites, l3::dsc::DscIdx(0))
                .expect("the one DSC's carriers");
            let arena = carriers.allocs.len();
            let (reads, tree) = v1::Dsc2Store::split(carriers.dsc);
            (
                v1::attach_to_prefilled_schedule::<Dd2, _, _, _>(
                    reads,
                    &*tree,
                    &*carriers.stages,
                    &mut metadata,
                    carriers.allocs,
                    ddc_defaults().offsets,
                ),
                arena,
            )
        };
        assert_eq!(arena, 3, "the arena holds the ddc view of all three HBM allocations");
        assert_eq!(attached, None, "the same refusal, reached directly");
        assert_eq!(
            metadata.external_nodes.len(),
            4,
            "the walk visited every node of the seed and refused on none of them"
        );
        assert_eq!(
            metadata.below_lx_schedule_insert_block, None,
            "and the one thing missing is the `lx_below_schedule` block, which stage 2a mints"
        );
    }

    /// ⭐⭐⭐ STAGE 2B ON THE TREE STAGE 2A'S GROWERS ACTUALLY BUILD — the deepest run available today,
    /// and the one that says where **stage 2b** stops on a real prefilled schedule.
    ///
    /// ⛔⛔ THE COMPOSITION IS 2A-MINUS-TWO-STEPS AND THIS TEST SAYS SO LOUDLY. The reference is
    /// `L3DlOpsScheduler::run` then `Ddc::run_v1` (`SchedulerStages.cpp:29-57`). [`run_l3`] cannot
    /// complete: it stops at the memory tracker's `todo!`. So the tree here is built by calling stage
    /// 2a's growers DIRECTLY, in entry 382's own order, skipping exactly the two steps that are not
    /// growers — `set_chunk_data_stage_params` (entry 207's recorded divergence, which mints no node)
    /// and the memory tracker (which places addresses and mints no node). That is the SAME tree
    /// [`the_growers_reproduce_the_references_whole_stage_2a_tree`] checks node for node against
    /// `g0/debug/sdsc_0/sdsc.json`, so it is the reference's own stage-2a tree and not an invented one.
    ///
    /// ⛔ WHAT IS THEREFORE *NOT* TRUE OF THIS RUN: no allocation has been PLACED, because the memory
    /// tracker never ran. Every `startAddressCoreCorelet_` is the fresh-node default. So any stage-2b
    /// unit that reads a placed address is reading an unplaced one, and this test asserts only where
    /// the stage stops — never a value derived from an address.
    #[test]
    fn stage_2b_on_the_grown_stage_2a_tree_reaches_the_ddl_conversion() {
        use crate::schedule::l3::dl_ops as ops;

        let mut sdsc = a_rmsq_super_dsc();
        let l3_state = DscState::seeded(&sdsc);
        // ⭐ ENTRY 382'S GROWERS, IN ITS OWN ORDER — the same six calls
        // `the_growers_reproduce_the_references_whole_stage_2a_tree` makes.
        {
            let (reads, mut env) = grow(&sdsc, &l3_state);
            ops::optimize_hbm_lds_output_in_schedule_tree(&sdsc, &mut env)
                .expect("the output tensor's HBM load is dropped");
            ops::optimize_hbm_transfers(&sdsc, &reads, &mut env)
                .expect("the HBM transfers are hoisted");
            ops::create_synchronization(&sdsc, ops::LxBuffering::Double, &reads, &reads, &mut env)
                .expect("the L3LU/LXLU and LXSU/L3SU sync pairs");
        }
        assert_eq!(
            l3_state.node_count(),
            22,
            "the reference's own stage-2a tree for `rmsq_o728`"
        );

        let state = Dsc2State::seeded(
            &sdsc,
            &l3_state,
            &[the_rmsq_compute_ops()],
            &[v1::StorageName("rmsq_o728".to_owned())],
        );
        // ⭐⭐⭐ AND STAGE 2B NOW RUNS *THROUGH* `Dsc2Store::schedule_head_block` — `run_v1`'s line
        // 6483 — INTO `select_and_parse_ddl_template` (`:6489`), which is the DDL step itself.
        //
        // ⭐ WHAT THAT PROVES RAN: `prep_dsc` complete (including `finalize_external_stage` on BOTH
        // datastages, `get_pe_sfp_split_dim` over the chunk snapshot, and the cross-core reduction
        // sweep), then `init_global_data`, then the whole of `attach_to_prefilled_schedule` — its walk
        // over all 22 nodes, every ALLOCATE's `holds_lds`/`is_dsc_memory`/`lds_alloc` round-trip, every
        // TRANSFER's `transfer_kind` and bound checks, every LOOP's `loop_dims`, and the
        // `lx_below_schedule` block found BY NAME — then the head block materialised whole, then
        // `DdlConversion::new` over it, then the template selection.
        //
        let ran = run_ddc::<Dd2>(
            &mut sdsc,
            &state,
            ddc_defaults(),
            &l3::dl_ops::AddressFoldCoords::flat(),
        );

        // ⛔⛔ NOTHING PANICKED — stage 2b reached the DDL step and came back with a callee's own
        // [`None`], which is the first time this stage has ended in anything but a `todo!`.
        assert!(
            ran.is_none(),
            "the DDL step's own refusal, propagated — not a carrier `todo!`: {ran:?}"
        );

        // ⭐⭐ AND *WHICH* REFUSAL, EXACTLY ONE, NAMED. `select_and_parse_ddl_template` asked
        // [`DdcTemplates::stated`] — which it reaches only AFTER `ddl_templates(opFunc, A::GEN)`
        // answered [`Some`] (`ddl/conversion.rs:6011`), so this is the template set being ASKED and
        // not an early exit. ⛔ IT IS A `build.rs` GAP AND NOT A CARRIER'S: four of a
        // `StatedTemplate`'s six parts are never emitted.
        assert_eq!(
            state.refusals().len(),
            1,
            "one refusal, so nothing else along the way could not answer: {:?}",
            state.refusals()
        );
        assert!(
            state
                .first_refusal()
                .is_some_and(|what| what.starts_with("DdlTemplateSet::stated:")),
            "the stop is the template set, not a store: {:?}",
            state.first_refusal()
        );

        // ⛔⛔ AND THE TREE DID NOT MOVE. The DDL step is where stage 2b MINTS its computes and
        // transfers, and it minted nothing — so *"stage 2b ran"* must never be read as *"stage 2b did
        // something"*. The corpus census cannot move until a template is stated AND the allocator
        // lands.
        assert_eq!(
            l3_state.node_count(),
            22,
            "still the reference's stage-2a tree, node for node"
        );
    }

    /// ⭐⭐⭐ `scheduleTree_.getHead()` MATERIALISED IS THE REFERENCE'S OWN TREE, NODE FOR NODE — the
    /// one thing [`v1::Dsc2Store::schedule_head_block`]'s doc says must not go wrong.
    ///
    /// ⛔⛔ AN EMPTY-CHILDREN BLOCK IS WHAT THIS TEST EXISTS TO CATCH. The DDL conversion splices every
    /// parsed template node into this block by POSITION (`ddl/conversion.rs:1914-1916`), so a head
    /// block with no children would attach the whole template to nothing — and it would compile, and
    /// `run_v1` would still return the same [`None`] it does today. So the assertion is on the SHAPE and
    /// the NAMES, against `REFERENCE_STAGE_2A_TREE`, and not on "it is a block".
    ///
    /// ⭐ AND THE LOOP NEST IS CHECKED AS A NEST, not as a count: `loop_ds0_ds1_y` OWNS
    /// `loop_ds0_ds1_out`, which owns `loop_ds0_ds1_mb` — the three chunk loops
    /// `debug/sdsc_0/sdsc.json` nests in that order — so a walk that flattened the tree fails here even
    /// though it would carry all 22 names.
    #[test]
    fn the_materialised_head_block_is_the_references_own_tree_nested() {
        use crate::schedule::dsc2::{BlockNode, SchedNode};
        use crate::schedule::l3::dl_ops as ops;

        let sdsc = a_rmsq_super_dsc();
        let l3_state = DscState::seeded(&sdsc);
        {
            let (reads, mut env) = grow(&sdsc, &l3_state);
            ops::optimize_hbm_lds_output_in_schedule_tree(&sdsc, &mut env)
                .expect("the output tensor's HBM load is dropped");
            ops::optimize_hbm_transfers(&sdsc, &reads, &mut env)
                .expect("the HBM transfers are hoisted");
            ops::create_synchronization(&sdsc, ops::LxBuffering::Double, &reads, &reads, &mut env)
                .expect("the L3LU/LXLU and LXSU/L3SU sync pairs");
        }
        let state = Dsc2State::seeded(
            &sdsc,
            &l3_state,
            &[the_rmsq_compute_ops()],
            &[v1::StorageName("rmsq_o728".to_owned())],
        );
        let mut sites = Dsc2Provider::new(&state, &l3::dl_ops::AddressFoldCoords::flat());
        let carriers = v1::Dsc2Sites::carriers(&mut sites, l3::dsc::DscIdx(0))
            .expect("the one DSC's carriers");
        let head = v1::Dsc2Store::schedule_head_block(&*carriers.dsc);

        /// Every name under one block, in the order its `next_` holds them, depth first.
        fn names(block: &BlockNode, into: &mut Vec<String>) {
            into.push(block.name.0.clone());
            for child in &block.children {
                match child {
                    SchedNode::Block(held) | SchedNode::Condition(held) => names(held, into),
                    SchedNode::Loop(held) => names(&held.block, into),
                    other => into.push(other.name().0.clone()),
                }
            }
        }

        let mut flat = Vec::new();
        names(&head, &mut flat);
        assert_eq!(
            flat, REFERENCE_STAGE_2A_TREE,
            "the materialised head block IS the reference's own stage-2a tree, in its own DFS order"
        );
        // ⭐ THE NEST, WHICH A FLAT WALK WOULD PASS THE ABOVE WITHOUT: `loop_ds0_ds1_y` owns
        // `loop_ds0_ds1_out` owns `loop_ds0_ds1_mb`, and the innermost carries the whole LX body.
        let outer = head
            .children
            .iter()
            .find_map(|child| match child {
                SchedNode::Loop(held) if held.block.name.0 == "loop_ds0_ds1_y" => Some(held),
                _ => None,
            })
            .expect("`loop_ds0_ds1_y` is a child of the root block");
        // ⛔ AND ITS `numId_`/`denId_` ARE THE REAL PAIR, so a materialiser that dropped them fails.
        assert!(
            outer.num.is_some() && outer.den.is_some(),
            "numId_/denId_ carried across: {:?}/{:?}",
            outer.num,
            outer.den
        );
        // ⛔ `isParametricLoop_` IS `false` — the member initializer of `dsc/dsc2.h:617-618`, and the
        // only two writers of it (`ddl_conversion.cpp:1126-1161`, `dsc/dsc2.cpp:1409-1416`) are the DDL
        // conversion this call OPENS and the JSON importer, neither of which has run.
        assert_eq!(
            outer.parametric_lds, None,
            "a loop stage 2a minted is not parametric"
        );
        let mid = outer
            .block
            .children
            .iter()
            .find_map(|child| match child {
                SchedNode::Loop(held) if held.block.name.0 == "loop_ds0_ds1_out" => Some(held),
                _ => None,
            })
            .expect("`loop_ds0_ds1_out` is INSIDE `loop_ds0_ds1_y` and not beside it");
        let inner = mid
            .block
            .children
            .iter()
            .find_map(|child| match child {
                SchedNode::Loop(held) if held.block.name.0 == "loop_ds0_ds1_mb" => Some(held),
                _ => None,
            })
            .expect("`loop_ds0_ds1_mb` is INSIDE `loop_ds0_ds1_out`");
        assert_eq!(
            inner.block.children.len(),
            15,
            "the innermost chunk loop carries the whole LX body — the fifteen nodes of \
             `REFERENCE_STAGE_2A_TREE` from `allocate_lds0_lx` on, which is what an empty-children \
             block would have lost"
        );
    }

    /// ⭐⭐ WHY THE DDL STEP ASKS THE TEMPLATE SET AT ALL — `select_and_parse_ddl_template` asks
    /// `ddl_templates(opFunc.spelling(), A::GEN)` BEFORE it touches the set
    /// (`ddl/conversion.rs:6011`), and that is a build-time table this test can read directly.
    ///
    /// ⭐ SO THE STOP [`stage_2b_on_the_grown_stage_2a_tree_reaches_the_ddl_conversion`] MEASURES IS
    /// EXPLAINED HERE: `mul` HAS candidate templates on `Dd2`, so the DDL step does NOT take the *"no
    /// DDL available for op"* early exit — it ASKS [`DdcTemplates::stated`], which answers [`None`]
    /// because `build.rs` emits one of a
    /// [`crate::schedule::ddl::conversion::StatedTemplate`]'s six parts. That is stage 2b's LAST
    /// reachable seam, and it is a `build.rs` gap and not a carrier's.
    ///
    /// ⛔ AND IT IS NOT VACUOUS: the same table has op funcs with NO candidates, and for one of those
    /// stage 2b would COMPLETE with [`v1::DscFilled::No`] rather than stop. The test carries both
    /// arms, so a change that emptied the table for `mul` fails it.
    #[test]
    fn mul_has_ddl_candidates_so_the_ddl_step_reaches_the_template_set() {
        use crate::generated::ddl_templates;
        use sys_arch_spec::arch_enums::OpFunc;

        let candidates = ddl_templates(OpFunc::Mul.spelling(), <Dd2 as crate::arch::Arch>::GEN)
            .expect("`mul` is one of the op funcs ddl_templates/*.ddl states a template for");
        assert!(
            !candidates.is_empty(),
            "a stated op func has at least one candidate template"
        );
        // ⭐ THE OTHER ARM, so this is a value and not a tautology: `GenericPartialReduction` is the
        // op func `prep_dsc` itself MINTS for a cross-core reduction, and no vendored template states
        // one — for a DSC whose first op were that, stage 2b would answer DscFilled::No and COMPLETE.
        assert_eq!(
            ddl_templates(
                OpFunc::GenericPartialReduction.spelling(),
                <Dd2 as crate::arch::Arch>::GEN
            ),
            None,
            "no ddl_templates/*.ddl states a template for GENERIC_PARTIAL_REDUCTION"
        );
    }

    /// ⭐ AND A DSC WITH NO `computeOp_` IS SKIPPED WHOLE, exactly as the reference skips it
    /// (`ddc/v1.rs:6438`) — so stage 2b COMPLETES and answers [`v1::DscFilled::Yes`] having minted
    /// nothing.
    ///
    /// ⛔⛔ THIS IS THE FALSE GREEN [`run_ddc`]'S DOC WARNS ABOUT, PINNED AS SUCH: it is what a caller
    /// that could not state `computeOp_` would get for EVERY program, and the tree is untouched. The
    /// test asserts the node count did not move, so "stage 2b completed" can never be read as "stage
    /// 2b did something".
    #[test]
    fn a_dsc_with_no_compute_op_is_skipped_and_the_tree_is_untouched() {
        let mut sdsc = a_rmsq_super_dsc();
        let l3_state = DscState::seeded(&sdsc);
        let before = l3_state.node_count();
        let state = Dsc2State::seeded(&sdsc, &l3_state, &[], &[]);
        let fill = run_ddc::<Dd2>(
            &mut sdsc,
            &state,
            ddc_defaults(),
            &l3::dl_ops::AddressFoldCoords::flat(),
        )
        .expect("a super-DSC whose every DSC states no compute op runs to the end of the loop");
        assert_eq!(
            fill.filled,
            v1::DscFilled::Yes,
            "no DSC found no DDL template, because no DSC was reached at all"
        );
        assert!(
            fill.said.is_empty(),
            "not even the `[DDC] start working on DSC` line was printed: {:?}",
            fill.said
        );
        assert_eq!(
            l3_state.node_count(),
            before,
            "the tree did not move — DscFilled::Yes here means NOTHING RAN"
        );
        assert!(
            state.refusals().is_empty(),
            "no carrier was asked anything: {:?}",
            state.refusals()
        );
    }

    /// ⭐⭐ [`l3::dl_ops::run`] IS CALLED, AND WHAT IT LEAVES IS COUNTED — the fifteen nodes stage 2a's
    /// growers reach before entry 207's divergence stops it.
    ///
    /// ⛔ NOT A TAUTOLOGY. The seed this test builds is FOUR nodes; every one of the eleven above it
    /// was minted by a ported unit, and each is checked BY THE REFERENCE'S OWN NAME from
    /// `debug/sdsc_0/sdsc.json` — the chunk loop nest in the reference's own order (`y` outermost),
    /// the per-tensor LX allocations and the HBM loads.
    ///
    /// ⛔ `transfer_lds2_src:hbm_dst:lx` IS MINTED HERE AND IS NOT IN THE REFERENCE'S OUTPUT — that is
    /// the output tensor's HBM load, which entry 214 (`optimize_hbm_lds_output_in_schedule_tree`)
    /// DROPS. It sits AFTER the stop, so this run has not reached it yet.
    ///
    /// ⭐⭐ THE FRONTIER MOVED PAST THE MEMORY TRACKER AND THIS TEST NOW PINS WHERE IT IS. The
    /// tracker is REAL — [`Trackers`] owns a [`crate::schedule::memtrack::bundle::MemTrackBundle`] of
    /// ported `DsTrackInMem`s, checked against the reference's own addresses for all 187 programs in
    /// `carriers/lx_oracle.rs` — so `backup`, `remove`, `capacity`, `check_and_add` and `restore_all`
    /// no longer stop anything. THE STOP IS NOW `try_alloc_l3`'s `allocs.get(&alloc)?`
    /// (`l3/dl_ops.rs:9261`): entry 222 looks the allocate node it is about to place up in the
    /// [`v1::AllocArena`], and NO UNIT OF STAGE 2A EVER WRITES THAT ARENA. The minting units call
    /// `L3TreeSurgery::fresh_alloc` + `new_allocate` (`:16607`, `:16624`, `:9587`, `:9603`), which put
    /// an `L3AllocateNode` in the TREE; the reference has ONE `dsc2::AllocateNode *` and the port
    /// split it in two, writing only the tree half. Filling the arena is a cross-entry port change
    /// (the two projections carry different fields), not carrier wiring.
    ///
    /// ⛔ NO `should_panic` ANY MORE, WHICH IS THE RATCHET IN BOTH DIRECTIONS: a `todo!` anywhere in
    /// this path now FAILS this test, and the fifteen node names below pin how far the growers got, so
    /// a regression to an earlier stop fails too. The stop itself is a ported unit's `None` and no
    /// carrier refused, which is what the two assertions beside it say.
    #[test]
    fn stage_2a_runs_past_the_memory_tracker_and_stops_at_the_unwritten_alloc_arena() {
        let mut sdsc = a_rmsq_super_dsc();
        let state = DscState::seeded(&sdsc);
        assert_eq!(
            state.node_count(),
            4,
            "the seed is `root_level_operations` plus one HBM allocate per HBM-pinned tensor"
        );

        let ran = run_l3::<false, Dd2>(
            &mut sdsc,
            &state,
            an_op_func(),
            &l3::dl_ops::AddressFoldCoords::flat(),
        );

        let names: Vec<String> = state.dscs()[0]
            .names()
            .into_iter()
            .map(|name| name.0)
            .collect();
        // ⛔ THE STOP IS A PORTED UNIT'S OWN REFUSAL AND NOT A CARRIER'S: nothing here was asked for a
        // fact it could not give.
        assert_eq!(
            ran, None,
            "entry 222 finds no allocate node in the arena and returns"
        );
        assert!(
            state.refusals().is_empty(),
            "the stop is entry 222's, not a carrier's: {:?}",
            state.refusals()
        );
        assert_eq!(
            names,
            [
                "root_level_operations",
                "allocate_lds0_hbm",
                "allocate_lds1_hbm",
                "allocate_lds2_hbm",
                "loop_ds0_ds1_y",
                "loop_ds0_ds1_out",
                "loop_ds0_ds1_mb",
                "allocate_lds0_lx",
                "transfer_lds0_src:hbm_dst:lx",
                "allocate_lds1_lx",
                "transfer_lds1_src:hbm_dst:lx",
                "allocate_lds2_lx",
                "transfer_lds2_src:hbm_dst:lx",
                "lx_below_schedule",
                "transfer_lds2_src:lx_dst:hbm",
            ],
            "the tree stage 2a left"
        );
    }

    /// ⭐ THE FIXED-SIGNATURE ENTRY POINT REACHES THE SAME SEAM — `run_stages` states no compute op
    /// and no fold props of its own (neither is a `SuperDsc` field), so it stops exactly where
    /// [`stage_2a_runs_past_the_memory_tracker_and_stops_at_the_unwritten_alloc_arena`] does, and the
    /// node count is what pins that.
    #[test]
    fn run_stages_reaches_the_alloc_arena_too() {
        let mut sdsc = a_rmsq_super_dsc();
        let ran = run_stages(&mut sdsc);
        assert_eq!(ran.nodes_before, 4, "the seed");
        assert_eq!(
            ran.nodes_after, 15,
            "what entry 382 left before entry 222's arena lookup stopped it"
        );
        assert!(!ran.l3, "stage 2a did not complete");
        assert!(!ran.ddc, "stage 2b is not composed yet");
        assert_eq!(
            ran.first_refusal, None,
            "no carrier refused — the stop is entry 222's own `None`"
        );
    }

    /// ⭐⭐ AND WHAT THE GROWERS REACH ONCE ENTRY 207 IS ANSWERED — the reference's whole stage-2a
    /// tree, TWENTY-TWO NODES, node for node and in DFS order.
    ///
    /// ⛔ THIS IS ENTRY 382'S OWN ORDER WITH ONE STEP SKIPPED, and it exists for exactly one reason:
    /// to say what the divergence costs. `set_chunk_data_stage_params` writes DATA STAGES and mints no
    /// node, so skipping it does not skip a grower — every node below is minted by the same units
    /// [`l3::dl_ops::run`] calls, in the order it calls them.
    ///
    /// ⛔ NOT A TAUTOLOGY: [`REFERENCE_STAGE_2A_TREE`] is transcribed from
    /// `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_0/sdsc.json`, whose thirty nodes the reference
    /// pipeline wrote for THIS program — not from anything this crate emitted.
    #[test]
    fn the_growers_reproduce_the_references_whole_stage_2a_tree() {
        use crate::schedule::l3::dl_ops as ops;

        let sdsc = a_rmsq_super_dsc();
        let state = DscState::seeded(&sdsc);
        let (reads, mut env) = grow(&sdsc, &state);
        // ⭐ SKIPPED: `set_chunk_data_stage_params` — entry 207's divergence, isolated in the test
        // above. It writes data stages and mints no schedule node.
        ops::optimize_hbm_lds_output_in_schedule_tree(&sdsc, &mut env)
            .expect("the output tensor's HBM load is dropped");
        ops::optimize_hbm_transfers(&sdsc, &reads, &mut env)
            .expect("the HBM transfers are hoisted");
        ops::create_synchronization(&sdsc, ops::LxBuffering::Double, &reads, &reads, &mut env)
            .expect("the L3LU/LXLU and LXSU/L3SU sync pairs");

        let names: Vec<String> = state.dscs()[0]
            .names()
            .into_iter()
            .map(|name| name.0)
            .collect();
        assert_eq!(
            names, REFERENCE_STAGE_2A_TREE,
            "the reference's own stage-2a tree for `rmsq_o728`"
        );
        assert_eq!(
            state.node_count(),
            22,
            "twenty-two nodes from a four-node seed"
        );
        assert!(state.refusals().is_empty(), "no carrier refused");
    }

    /// ⭐⭐ THE SEVEN FIELDS `l3::dsc` USED TO DROP, ANSWERED WITH `g0/sdsc_0.json`'S OWN VALUES —
    /// `dsName_`, `wordLength`, `dataFormat_`, `scaledLdsCategory_`, `constantInfo_`,
    /// `dimToSymbolMapping_` and `l0TetheredMode_`.
    ///
    /// ⛔ EVERY EXPECTATION IS A VALUE READ OFF THE FIXTURE, not a shape: `Tensor{0,1,2}` /
    /// `wordLength: 2` / `dataFormat_: "SEN169_FP16"` on each of the three `labeledDs_` entries,
    /// `constantInfo_: "{}"` (so no constant and `masking_constant` is [`None`]), `maskingConstId_: -1`,
    /// and NEITHER `dimToSymbolMapping_` NOR `l0TetheredMode_` emitted — the declared `{}` and `false`.
    ///
    /// ⛔ AND `holds_lds(3)` IS THE OTHER ARM, so `lds_name` is not satisfied by any total function: the
    /// list holds exactly three entries and the fourth index is `labeledDs_.at()`'s throw.
    #[test]
    fn the_seven_dropped_fields_answer_with_the_fixtures_own_values() {
        use crate::schedule::ddc::fold::ScaledLds;
        use crate::schedule::ddc::v1::{ExploreDsc, Masking, Placement, StorageNames};

        let sdsc = a_rmsq_super_dsc();
        let l3_state = DscState::seeded(&sdsc);
        let state = Dsc2State::seeded(
            &sdsc,
            &l3_state,
            &[the_rmsq_compute_ops()],
            &[v1::StorageName("rmsq_o728".to_owned())],
        );
        let mut sites = Dsc2Provider::new(&state, &l3::dl_ops::AddressFoldCoords::flat());
        let carriers = v1::Dsc2Sites::carriers(&mut sites, l3::dsc::DscIdx(0))
            .expect("the one DSC's carriers");
        let (reads, _) = v1::Dsc2Store::split(carriers.dsc);

        // `dsName_` — `allocate-Tensor{N}_hbm` in `debug/sdsc_0/sdsc.json` is named off THIS.
        for at in 0u32..3 {
            assert!(reads.holds_lds(LdsIdx(at)), "labeledDs_ holds position {at}");
            assert_eq!(
                reads.lds_name(LdsIdx(at)),
                v1::StorageName(format!("Tensor{at}")),
                "labeledDs_.at({at}).dsName_"
            );
            // `dataFormat_` — what the DDL match binds each operand's type by.
            assert_eq!(
                Masking::lds_format(reads, LdsIdx(at)),
                Some(crate::formats::DataFormat::Sen169Fp16),
                "labeledDs_.at({at}).dataFormat_"
            );
            // `scaledLdsCategory_` — ABSENT on all 580 labelled DSs of `g0/`, which is the declared
            // `REGULAR_TENSOR` (`dsc/dscdefn.h:356`) and NOT the `SCALE_TENSOR` an
            // `Option<MxScaleTensor>` used to collapse it with.
            assert_eq!(
                reads.scaled_category(LdsIdx(at)),
                ScaledLds::Regular,
                "labeledDs_.at({at}).scaledLdsCategory_"
            );
            assert!(
                !Placement::is_scale_tensor(reads, LdsIdx(at)),
                "and REGULAR_TENSOR is not a scale tensor"
            );
        }
        // ⛔ THE OTHER ARM: three entries, so the fourth index is the `.at()` throw and `lds_name` is
        // not a total function over `LdsIdx`.
        assert!(
            !reads.holds_lds(LdsIdx(3)),
            "labeledDs_ has exactly three entries"
        );

        // `constantInfo_` is the string `"{}"` and `maskingConstId_` is `-1`, so there is no masking
        // constant — and `masking_constant` is `constantInfo_.count(maskingConstId_)`, which `-1` never
        // satisfies.
        assert_eq!(
            reads.masking_constant(),
            None,
            "constantInfo_ is empty and maskingConstId_ is -1"
        );
        // ⛔ AND NEITHER IS A SUBSTITUTED DEFAULT: `dimToSymbolMapping_` is emitted by nothing, which is
        // `count(dim) == 0` for every dim, and `l0TetheredMode_` likewise is the declared `false`.
        for dim in [PrimaryDim::Mb, PrimaryDim::Out, PrimaryDim::Y] {
            assert!(
                !Masking::is_symbolic(reads, dim),
                "dimToSymbolMapping_.count({dim:?}) == 0"
            );
        }
        assert_eq!(
            Placement::l0_tethered(reads),
            v1::L0Tethered::Split,
            "l0TetheredMode_ == false"
        );

        // `wordLength` — read through the DDL match site, which is what compares it against a
        // template's own `bitSize_ / 8`.
        for at in 0u32..3 {
            assert_eq!(
                crate::schedule::ddl::conversion::MatchSite::lds_word_length(
                    &*carriers.ddl,
                    LdsIdx(at)
                ),
                Some(WordLength(2)),
                "labeledDs_.at({at}).wordLength"
            );
        }
    }

    /// ⭐⭐⭐ AND THE WRITES REACH THE READS — the ONE property [`v1::Dsc2Store::split`]'s own doc
    /// (`ddc/v1.rs:6040-6047`) says a caller must not break: *"two carriers over one `currDsc` would
    /// let entry 308's writes be invisible to entry 307's reads"*.
    ///
    /// ⛔⛔ THIS TEST IS THE ONE THAT WOULD FAIL ON A SECOND COPY, AND NOTHING ELSE WOULD. Every setter
    /// below is a `&mut self` method on [`super::Dsc2Store`] and every read is through the `&Reads` half
    /// [`v1::Dsc2Store::split`] hands out; a `Dsc2Facts` that gave the writers their own
    /// `DesignSpaceConfig` would compile, silently drop all five writes, and every other test in this
    /// module would still pass.
    ///
    /// ⛔ AND IT CARRIES VALUES: each assertion names the value the setter wrote, so a setter that
    /// wrote *something* to the right field does not satisfy it.
    #[test]
    fn entry_308_and_372_writes_are_visible_to_the_shared_reads_half() {
        use crate::schedule::ddc::fold::ScaledLds;
        use crate::schedule::ddc::transformation::AutoShuffling;
        use crate::schedule::ddc::v1::{ExploreDsc, Masking, PrepDsc, StorageNames};

        let sdsc = a_rmsq_super_dsc();
        let l3_state = DscState::seeded(&sdsc);
        let state = Dsc2State::seeded(
            &sdsc,
            &l3_state,
            &[the_rmsq_compute_ops()],
            &[v1::StorageName("rmsq_o728".to_owned())],
        );
        let mut sites = Dsc2Provider::new(&state, &l3::dl_ops::AddressFoldCoords::flat());
        let carriers = v1::Dsc2Sites::carriers(&mut sites, l3::dsc::DscIdx(0))
            .expect("the one DSC's carriers");
        let store = carriers.dsc;

        // ⭐ ENTRY 308'S OWN WRITES ON THE MINTED INTERNAL INPUT (`ddc/ddcv1.cpp:2098-2162`):
        // `dsName_ += "_internalInput"`, `dsType_ = INTERNAL`, `wordLength = 2`,
        // `dataFormat_ = SEN143_FP8`, `scaledLdsCategory_ = VALUE_TENSOR`.
        PrepDsc::append_lds_name(store, LdsIdx(1), v1::InternalLds::Input);
        PrepDsc::set_lds_internal(store, LdsIdx(1));
        PrepDsc::set_lds_word_length(store, LdsIdx(1), WordLength(1));
        PrepDsc::set_lds_format(store, LdsIdx(1), crate::formats::DataFormat::Sen143Fp8);
        PrepDsc::set_lds_scaled_category(store, LdsIdx(1), ScaledLds::Value);
        // ⭐ AND ENTRY 372'S, WHICH *ASSIGNS* THE NAME RATHER THAN APPENDING IT
        // (`ddc/transformation.rs:3295`).
        AutoShuffling::set_lds_name(
            store,
            LdsIdx(2),
            v1::StorageName("autoshuffle_reg_0".to_owned()),
        );

        // `dsType_` is on the STORE's own [`crate::schedule::ddc::transformation::LabeledDs`], read
        // before the split so the two halves are not both borrowed at once.
        assert_eq!(
            crate::schedule::ddc::transformation::LabeledDs::ds_type(&*store, LdsIdx(1)),
            DsType::Internal,
            "dsType_ = INTERNAL"
        );

        let (reads, _) = v1::Dsc2Store::split(store);
        assert_eq!(
            reads.lds_name(LdsIdx(1)),
            v1::StorageName("Tensor1_internalInput".to_owned()),
            "append_lds_name APPENDED to the fixture's own `Tensor1`, it did not replace it"
        );
        assert_eq!(
            Masking::lds_format(reads, LdsIdx(1)),
            Some(crate::formats::DataFormat::Sen143Fp8),
            "set_lds_format is visible through the shared currDsc"
        );
        assert_eq!(
            reads.scaled_category(LdsIdx(1)),
            ScaledLds::Value,
            "VALUE_TENSOR — the arm an Option<MxScaleTensor> could not state at all"
        );
        assert_eq!(
            reads.lds_name(LdsIdx(2)),
            v1::StorageName("autoshuffle_reg_0".to_owned()),
            "set_lds_name ASSIGNED, so `Tensor2` is gone"
        );
        // ⛔ AND THE UNTOUCHED ENTRY IS UNTOUCHED, so none of the above is a write to every entry.
        assert_eq!(
            reads.lds_name(LdsIdx(0)),
            v1::StorageName("Tensor0".to_owned()),
            "labeledDs_.at(0) was not written"
        );
        assert_eq!(
            Masking::lds_format(reads, LdsIdx(0)),
            Some(crate::formats::DataFormat::Sen169Fp16),
            "and it keeps the fixture's own format"
        );
    }

    /// ⭐⭐ `constantInfo_` WITH A REAL CONSTANT IN IT — `g0/sdsc_1.json`'s `1_rmmean_o728`, whose table
    /// is `{"0": {"name_": "scaling_factor", "dataFormat_": "SEN169_FP16", "data_": [10240],
    /// "allocations_": {}}}`.
    ///
    /// ⛔ THE `useZeroMean` PREDICATE IS ASSERTED ON BOTH ARMS, and its datum is what decides:
    /// `getSingleDataStrict(constinfo.data_).at(0) == 1` (`ddc/ddcv1.cpp:2066-2068`) swaps `EXX2` for
    /// `EXX2_ZEROMEAN`, so a constant of that NAME whose datum is `0` must answer `false` — otherwise
    /// the predicate is satisfied by any wrong behaviour of the right shape.
    #[test]
    fn a_real_constant_info_table_answers_its_name_its_id_and_the_zero_mean_swap() {
        use crate::schedule::ddc::fold::ConstIdx;
        use crate::schedule::ddc::v1::{Masking, PrepDsc, StorageNames};
        use crate::schedule::l3::dsc::{ConstantInfo, DdcFacts};

        /// One super-DSC whose one DSC carries that `constantInfo_` table and `maskingConstId_`.
        fn with_constants(constants: DdcFacts) -> SuperDsc {
            let mut dsc = a_rmsq_dsc();
            dsc.ddc = constants;
            SuperDsc::new(
                DscList::new(dsc, Vec::new()),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
            )
        }

        /// `sdsc_1.json`'s own entry, with the datum the caller states.
        fn scaling_factor(datum: i64) -> ConstantInfo {
            ConstantInfo {
                name: v1::StorageName("scaling_factor".to_owned()),
                data_format: Some(crate::formats::DataFormat::Sen169Fp16),
                data: vec![datum],
                is_data_symbolic: false,
                allocations: BTreeMap::new(),
            }
        }

        let ask = |facts: DdcFacts, check: &dyn Fn(&mut super::Dsc2Store<'_, '_>)| {
            let sdsc = with_constants(facts);
            let l3_state = DscState::seeded(&sdsc);
            let state = Dsc2State::seeded(
                &sdsc,
                &l3_state,
                &[the_rmsq_compute_ops()],
                &[v1::StorageName("rmmean_o728".to_owned())],
            );
            let mut sites = Dsc2Provider::new(&state, &l3::dl_ops::AddressFoldCoords::flat());
            let carriers = v1::Dsc2Sites::carriers(&mut sites, l3::dsc::DscIdx(0))
                .expect("the one DSC's carriers");
            check(carriers.dsc);
        };

        // `constantInfo_.at(0).name_` — the value the reference keys its memory tracker by.
        ask(
            DdcFacts {
                constants: BTreeMap::from([(ConstIdx(0), scaling_factor(10240))]),
                ..DdcFacts::default()
            },
            &|store| {
                // ⛔ AND THE NAME IS NOT `useZeroMean`, so no swap.
                assert!(
                    !PrepDsc::declares_zero_mean_constant(&*store),
                    "`scaling_factor` is not `useZeroMean`"
                );
                let (reads, _) = v1::Dsc2Store::split(store);
                assert_eq!(
                    reads.constant_name(ConstIdx(0)),
                    v1::StorageName("scaling_factor".to_owned()),
                    "constantInfo_.at(0).name_ off `g0/sdsc_1.json`"
                );
                // ⛔ `maskingConstId_` IS STILL `-1`, so a table WITH a constant in it does not by
                // itself make a masking constant.
                assert_eq!(
                    reads.masking_constant(),
                    None,
                    "constantInfo_.count(-1) is zero however full the table is"
                );
            },
        );

        // ⭐ `maskingConstId_ = 0` WITH THE TABLE HOLDING 0 — `constantInfo_.count(maskingConstId_)`.
        ask(
            DdcFacts {
                constants: BTreeMap::from([(ConstIdx(0), scaling_factor(10240))]),
                masking_const: Some(ConstIdx(0)),
                ..DdcFacts::default()
            },
            &|store| {
                let (reads, _) = v1::Dsc2Store::split(store);
                assert_eq!(
                    reads.masking_constant(),
                    Some(ConstIdx(0)),
                    "the table holds the id maskingConstId_ names"
                );
            },
        );

        // ⛔ AND `maskingConstId_ = 7` WITH THE TABLE HOLDING ONLY 0 IS [`None`] — the `count` is the
        // whole of the reference's test, so a bare `Some(maskingConstId_)` would fail here.
        ask(
            DdcFacts {
                constants: BTreeMap::from([(ConstIdx(0), scaling_factor(10240))]),
                masking_const: Some(ConstIdx(7)),
                ..DdcFacts::default()
            },
            &|store| {
                let (reads, _) = v1::Dsc2Store::split(store);
                assert_eq!(
                    reads.masking_constant(),
                    None,
                    "constantInfo_.count(7) is zero, so entry 262 has no masking constant"
                );
            },
        );

        // ⭐ THE ZERO-MEAN SWAP, BOTH ARMS — the DATUM decides, not the name.
        ask(
            DdcFacts {
                constants: BTreeMap::from([(
                    ConstIdx(0),
                    ConstantInfo {
                        name: v1::StorageName("useZeroMean".to_owned()),
                        data: vec![1],
                        ..scaling_factor(1)
                    },
                )]),
                ..DdcFacts::default()
            },
            &|store| {
                assert!(
                    PrepDsc::declares_zero_mean_constant(&*store),
                    "`useZeroMean` with datum 1 swaps EXX2 for EXX2_ZEROMEAN"
                );
            },
        );
        ask(
            DdcFacts {
                constants: BTreeMap::from([(
                    ConstIdx(0),
                    ConstantInfo {
                        name: v1::StorageName("useZeroMean".to_owned()),
                        data: vec![0],
                        ..scaling_factor(0)
                    },
                )]),
                ..DdcFacts::default()
            },
            &|store| {
                assert!(
                    !PrepDsc::declares_zero_mean_constant(&*store),
                    "`useZeroMean` with datum 0 is the switch turned OFF — this arm is what a \
                     `has a datum` predicate would get wrong"
                );
            },
        );
    }
}
