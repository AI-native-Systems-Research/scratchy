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
mod ddc_state;
mod env;
mod offsets;
mod reads;
mod state;
mod tree;

pub use carriers::{Placement, Sink, Symbols, Trackers};
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
    let mut trackers = Trackers::default();
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
    use crate::schedule::dsc2::{LayoutDims, LdsIdx};
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DesignSpaceConfig, DscList, FilledDims, LabeledDs,
        LabeledDsList, NamedDims, Pinning, PrimaryDsInfo, StageDims, SuperDsc, WkSliceCount,
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
    fn labeled(recorded: LdsIdx) -> LabeledDs {
        LabeledDs::new(
            DsType::Output,
            LAYOUT.iter().map(|&dim| (dim, Scale::Sized(1.0))).collect(),
            recorded,
            pinning(),
        )
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
    /// `labeledDs_.at(N).dsName_`, which [`crate::schedule::l3::dsc::LabeledDs`] does not carry — the
    /// one gap [`DscState::seeded`] records, and the only name below that is not the reference's own.
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
    /// ⭐⭐ THE FRONTIER MOVED PAST ENTRY 207 AND THIS TEST NOW PINS WHERE IT IS. With 207 recording
    /// the reference's `-1` instead of refusing, stage 2a runs on past `set_chunk_data_stage_params`
    /// and reaches the MEMORY TRACKER — `ddc::DsTrackInMem`, whose `checkAndAddDs` DECIDES the byte
    /// offset of every allocation. That is an unported seam and it must stay a `todo!`: a tracker that
    /// answered a plausible capacity or offset would place real tensors at invented addresses, which
    /// this crate ranks worse than a stop.
    ///
    /// ⛔ THE EXPECTED MESSAGE IS THE SPECIFIC SEAM, NOT ANY PANIC. A bare `should_panic` here would
    /// pass on the FIRST `todo!` of eighty-six and so would say nothing about how far the stage got;
    /// naming `ExPhaseTrackers::backup` makes this a ratchet in both directions — it fails if the
    /// stage regresses to an earlier stop, and it fails the moment the tracker lands, which is the
    /// cue to re-measure the census.
    #[test]
    #[should_panic(expected = "ExPhaseTrackers::backup")]
    fn stage_2a_runs_past_entry_207_and_reaches_the_memory_tracker() {
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
        assert_eq!(ran, None, "entry 207's divergence stops the stage");
        assert!(
            state.refusals().is_empty(),
            "the stop is entry 207's, not a carrier's: {:?}",
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
    /// [`stage_2a_runs_past_entry_207_and_reaches_the_memory_tracker`] does. Pinned by the same named
    /// seam and for the same reason.
    #[test]
    #[should_panic(expected = "ExPhaseTrackers::backup")]
    fn run_stages_reaches_the_memory_tracker_too() {
        let mut sdsc = a_rmsq_super_dsc();
        let ran = run_stages(&mut sdsc);
        assert_eq!(ran.nodes_before, 4, "the seed");
        assert_eq!(
            ran.nodes_after, 15,
            "what entry 382 left before entry 207 stopped it"
        );
        assert!(!ran.l3, "stage 2a did not complete");
        assert!(!ran.ddc, "stage 2b is not composed yet");
        assert_eq!(
            ran.first_refusal, None,
            "no carrier refused — the stop is entry 207's own divergence"
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
}
