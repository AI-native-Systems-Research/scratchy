// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY ddc / L3-SCHEDULER CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.     ║
// ║ Campaign statement: crustify-ddc/TASK.md   ·   worklist: crustify-ddc/UNITS.tsv              ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (revision a0d29abbed)
//    Every citation below resolves against that revision. `crustify-ddc/cpp/{l3,ddc,ddl,dcg}.cpp`
//    say WHICH functions are in scope and IN WHAT ORDER; their bodies were verified byte-identical
//    to the authority (382/382, 1,069,773 bytes, 2 of 2 negative controls DETECTED), so either
//    may be read — but the authority file is the one that carries the surrounding declarations you
//    will need. ⛔ The other local deeptools checkout is a DIFFERENT revision; the pod is not
//    reachable from here.
//
// 2. WHAT THIS STAGE IS. `dbo-opt` is the binary scratchy shells out to; its per-program pipeline
//    runs `runDdc` for every program (dbo/src/Transforms/sdsc_bundle/RunSchedulerOnSdsc.cpp:145).
//    ⭐ `dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp` — 60 lines — IS THE SPEC FOR THIS WHOLE
//    JOB. READ IT FIRST. Four stages:
//      1.  sbf::doCoreletSplitSdsc     ⛔ EXCLUDED: SchedulerStages.cpp:25 returns early unless
//          numCoreletsPerCore == 2, and scratchy emits numCoreletsUsed_ = 1 in all 313 sampled
//          SuperDSCs. See crustify-ddc/EXCLUSIONS.tsv.
//      2a. L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, verbose).run(sdsc)
//      2b. ddc::Ddc(dscGlobal, ..).run_v1(sdsc)      entry: ddc/ddcv1.cpp:3695
//      3.  DcgManager::runDcgForDlOpsStandalone(sdsc)   dcg/dcg_manager/dcg_manager.cpp:449
//          ⛔ THAT branch, NOT `runDcg`: SchedulerStages.cpp:53-57 picks it whenever `dscs_` is
//          non-empty, always true for scratchy's input. ⚠️ Its body delegates almost entirely to
//          `dcg_fe/pcfg_gen/` and `dcg_be/`, both OUT of scope — `todo!` NAMING the missing
//          translator is correct there; ⛔ do NOT invent a PCFG.
//    `runDdc` raises when DDC finds no mapping ("Scheduler failed to find a suitable op mapping"),
//    so the mapping is not optional.
//
// 3. WHY IT MATTERS. `ddc.run_v1` is what PLACES ADDRESSES, and the L3 scheduler's `run` commits
//    LX allocations then calls `fillAllocationStartAddrAndOffset` — literally "Set start address,
//    offset in allocations" (dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:8000). Our emitted views
//    have printed `start_address = 0` where the reference states a placed base; the backend has
//    refused with `Register initialization out of boundary`; and `src/reginit.rs` (1,569 lines, on
//    the integ branch) HAND-COMPUTES placement from ddc/ddcv1.cpp:132-360. This campaign replaces
//    that guesswork with the real thing.
//
// 4. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For this stage the effect is WHAT IS
//    WRITTEN INTO THE `SuperDsc`: addresses, mappings, symbol definitions, fold state, schedule
//    steps. A hand attempt on bridge 2 extracted each function's decision rule into a documented
//    predicate, omitted the part that changed the IR, and reported it done — nothing called any of
//    it. A PREDICATE IS NOT A PORT. Droppable: only the mechanism for REACHING operands (walking
//    uses, memoising, positioning a builder). ⛔ If a TYPE cannot express a result, EXTEND THE
//    TYPE. Deciding a function is unnecessary is NOT the porter's call.
//
// 5. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 6. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no sanitizers, ❌ no C-vs-Rust harness.
//
// 7. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    NEWTYPES, NEVER RAW SCALARS — an address, an offset, a core index and an element count must
//    not be interchangeable `u64`s; transposing two must be E0308. A closed set is an `enum`, never
//    a string. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED here
//    — and ⛔ never substitute a stand-in op, or a fabricated address, to dodge one. A fabricated
//    placement to avoid a stop is worse than the stop.
//
// 8. ⚠️ THE L3 SCHEDULER MAY OR MAY NOT APPLY TO US — DO NOT DECIDE THIS, PORT IT. Scratchy states
//    its own schedule (our SuperDSC writes `coreIdToDscSchedule`) and L3DlOpsScheduler.cpp:415-425
//    READS that field. That question is the USER'S, not the porter's. Measured while scoping: the
//    field occurs in that file ONLY as a read, never a write, so it is an INPUT to both stages —
//    what the L3 scheduler PRODUCES is the dsc2 schedule tree, the LX buffer type, the committed
//    LX allocations and their start addresses.
//
// 9. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 10. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//     the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 11. ⭐⭐ THE ACCEPTANCE CRITERION — WHAT THESE STAGES PRODUCE. They mutate the `SuperDsc` IN
//     PLACE, and the observable result is that THE `ScheduleNode` TREE GAINS ITS LOOP, TRANSFER,
//     SYNC AND CONDITION NODES. Scratchy's SuperDSC today has only ALLOCATE nodes (one construction
//     site: `crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs:5127`) plus a flat
//     `computeOp_` list — which matches torch-spyre's own `generate_sdsc`, and is exactly why
//     `dxp_standalone` works and the Rust `sdscToDataflowIR` port yields nothing.
//     The REPRESENTATIVE MINTING SITE to model is `ddc/ddl/ddl_conversion.cpp:1065`: it mints a
//     `dsc2::LoopNode`, takes its dims from the DDL, names it `loop_ds<num>_ds<den>`, and registers
//     it in `ddlInterface.loop_labels_`.
//     ⛔ A PORT THAT DOCUMENTS THE SCHEDULING DECISION WITHOUT ADDING NODES TO THAT TREE IS NOT A
//     PORT.
//
// 12. ⭐⭐ THE TARGET VOCABULARY IS ALREADY TYPED — a wrong shape must be a COMPILE ERROR, not a
//     judgement call. Emit into these EXISTING types; do not invent any:
//       `src/bridges/superdsc_to_dataflow_ir/driver.rs:467`        `Statement`
//       `driver.rs:1152`                                           `Scheduled`
//       `driver.rs:1756-1790`                                      `Viewed` / `Viewing`
//       `driver.rs:1539`                                           `ScheduleView::roots`
//       `driver.rs:1788`                                           `Dsc`
//     The single function it all plugs into is `Schedule::roots` in
//     `crates/targets/spyre/src/lower_superdsc_to_dataflow_ir.rs` — committed, compiles, and
//     currently yields nothing. Both the component and the DSC are already in hand there.
//
// 13. ALREADY PORTED, DO NOT DUPLICATE — all four in
//     `src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs`, following the same
//     `/// Replaces: eNNN_name` convention so cross-referencing works:
//       `e001_checkConstraints` (ddc/ddcv1.cpp:792)   `e002_createDataConnectMetadata` (:3283)
//       `e041_getStickSizes`    `e071_getCumulativeStickSizes`  (both `DesignSpaceConfig` methods
//       in `dsc/dsc2.cpp`, OUTSIDE this campaign's file list — your units CALL them.)
//     `checkConstraints` is a LAMBDA inside `Ddc::exploreAssignDataStages`, so porting that unit
//     means CALLING the existing port, not writing a second constraint checker. Units carrying such
//     a constraint have a ⛔ NOTE on the anchor itself.

//! `ddc/ddc_fold.cpp` — 36 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e078_dbgPrint` | 078 | 0 | 25 | — | `ddc/ddc_fold.cpp:20` |
//! | `e079_dbgPrint` | 079 | 0 | 16 | — | `ddc/ddc_fold.cpp:46` |
//! | `e080_allDimsCovered` | 080 | 0 | 9 | — | `ddc/ddc_fold.cpp:75` |
//! | `e081_buildFoldForBroadcastDim` | 081 | 0 | 19 | — | `ddc/ddc_fold.cpp:87` |
//! | `e082_getCompRowId` | 082 | 0 | 7 | — | `ddc/ddc_fold.cpp:110` |
//! | `e083_getComponent` | 083 | 0 | 17 | — | `ddc/ddc_fold.cpp:118` |
//! | `e084_getLayoutDimsFromNode` | 084 | 0 | 51 | — | `ddc/ddc_fold.cpp:136` |
//! | `e085_needToConsiderRowBundling` | 085 | 0 | 7 | — | `ddc/ddc_fold.cpp:190` |
//! | `e086_isAllocateIncoming` | 086 | 0 | 28 | — | `ddc/ddc_fold.cpp:201` |
//! | `e087_findCommonAncestor` | 087 | 0 | 32 | — | `ddc/ddc_fold.cpp:232` |
//! | `e088_orderDescendants` | 088 | 0 | 25 | — | `ddc/ddc_fold.cpp:267` |
//! | `e089_constructAllocElemArrLayout` | 089 | 0 | 87 | — | `ddc/ddc_fold.cpp:296` |
//! | `e090_combineContigousLevels` | 090 | 0 | 34 | — | `ddc/ddc_fold.cpp:395` |
//! | `e091_matchDataStream` | 091 | 0 | 42 | — | `ddc/ddc_fold.cpp:430` |
//! | `e092_getNumElementsInPTSlice` | 092 | 0 | 18 | `Ddc` | `ddc/ddc_fold.cpp:1519` |
//! | `e093_buildSpatialFold` | 093 | 0 | 41 | `Ddc` | `ddc/ddc_fold.cpp:2109` |
//! | `e094_getDefaultRowSplitFold` | 094 | 0 | 6 | — | `ddc/ddc_fold.cpp:2154` |
//! | `e095_gatherFoldParams` | 095 | 0 | 12 | `Ddc` | `ddc/ddc_fold.cpp:2224` |
//! | `e233_dbgPrint` | 233 | 1 | 11 | — | `ddc/ddc_fold.cpp:63` |
//! | `e234_scaleUpCoord` | 234 | 1 | 118 | `Ddc` | `ddc/ddc_fold.cpp:476` |
//! | `e235_scaleDownCoord` | 235 | 1 | 87 | `Ddc` | `ddc/ddc_fold.cpp:598` |
//! | `e236_needNonRowBundling` | 236 | 1 | 50 | `Ddc` | `ddc/ddc_fold.cpp:688` |
//! | `e237_sameCoordinateRange` | 237 | 1 | 252 | `Ddc` | `ddc/ddc_fold.cpp:1260` |
//! | `e238_getRelatedComputeCoord` | 238 | 1 | 129 | `Ddc` | `ddc/ddc_fold.cpp:1976` |
//! | `e239_computeParamsForRowSplitFold` | 239 | 1 | 60 | `Ddc` | `ddc/ddc_fold.cpp:2161` |
//! | `e240_buildFoldForExternalAllocation` | 240 | 1 | 156 | `Ddc` | `ddc/ddc_fold.cpp:2238` |
//! | `e241_relateLoopsToAllocElemArr` | 241 | 1 | 166 | `Ddc` | `ddc/ddc_fold.cpp:2872` |
//! | `e297_gatherRelatedPTRowsBase` | 297 | 2 | 217 | `Ddc` | `ddc/ddc_fold.cpp:740` |
//! | `e298_buildFoldFromAllocation` | 298 | 2 | 245 | `Ddc` | `ddc/ddc_fold.cpp:3043` |
//! | `e299_buildFoldFromNonAllocRef` | 299 | 2 | 355 | `Ddc` | `ddc/ddc_fold.cpp:3294` |
//! | `e337_gatherRelatedPTRows` | 337 | 3 | 298 | `Ddc` | `ddc/ddc_fold.cpp:959` |
//! | `e356_buildFoldForAllocation` | 356 | 4 | 473 | `Ddc` | `ddc/ddc_fold.cpp:2395` |
//! | `e357_buildFoldForCompute` | 357 | 4 | 773 | `Ddc` | `ddc/ddc_fold.cpp:3656` |
//! | `e358_buildFoldForTransfer` | 358 | 4 | 289 | `Ddc` | `ddc/ddc_fold.cpp:4433` |
//! | `e370_buildAndPropagateFold` | 370 | 5 | 350 | `Ddc` | `ddc/ddc_fold.cpp:1625` |
//! | `e375_coordinateCapture` | 375 | 6 | 86 | `Ddc` | `ddc/ddc_fold.cpp:1538` |

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroI64;

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::{Arch, Elements};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, SliceElems, Stage, StickDims, StickPart, cumulative_stick_sizes,
};
use crate::generated::DataConnect;
use crate::schedule::ddc::RefRole;
use crate::schedule::ddc::metadata::MetaDimKind;
use crate::schedule::ddc::transformation::Scale;
use crate::schedule::ddc::transformation_util::{LoopNode, PaddingForm, PrimaryDimAndKind};
use crate::schedule::dsc2;
use crate::schedule::dsc2::{
    ComputeNode, CoordinateCategory, Dsc, FoldCardinality, FoldCoeff, LdsIdx, Node, Operand,
    OperandPos, TransferNode, TransferSide,
};
use crate::schedule::l3::dl_ops::{LoopAndDim, LoopDistribution};
use crate::units::DfirUnit;

/// ONE OPERAND AS `dbgPrint` SPELLS IT — `'<component>(<data_connect>)'`, and an unset
/// `dataConnect_` is the empty string the default-constructed `DataInfo` prints.
fn operand_text(operand: &Operand) -> String {
    let connect = operand.data.data_connect.map_or("", |dc| dc.spelling());
    format!(" '{}({})'", operand.unit.spelling(), connect)
}

/// Replaces: e078_dbgPrint
///
/// A compute node's debug line: name, address, op spelling, then its inputs and its outputs.
///
/// ⛔ TRAP CLOSED: the reference bounds both loops by the `..LdsAndLoopOffsets_` length while
/// indexing `inputs_`/`outputs_` with `.at(i)`, so a length mismatch throws. `Operand` pairs the
/// component with its data, and `DdlComputeType::cpp_spelling` is total where
/// `computeTypeToString.at(type_)` has no entry for `FCVT`.
#[must_use]
pub fn dbg_print_compute(node: &ComputeNode) -> String {
    let mut out = format!(
        " ComputeNode: {}({:p}) {} [",
        node.name.0,
        node,
        node.op.cpp_spelling()
    );
    for operand in &node.inputs {
        out.push_str(&operand_text(operand));
    }
    out.push_str("] -> [");
    for operand in &node.outputs {
        out.push_str(&operand_text(operand));
    }
    out.push(']');
    out
}

/// Replaces: e079_dbgPrint
///
/// A transfer node's debug line: name, address, its one source, then every destination.
///
/// ⛔ TRAP CLOSED: the reference bounds its loop by `dstLdsAndLoopOffsets_.size()` while indexing
/// `dstVias_` with `.at(i)`. `Dsts` holds the pair as one list.
#[must_use]
pub fn dbg_print_transfer(node: &TransferNode) -> String {
    let mut out = format!(
        " TransferNode: {}({:p}) [{}] -> [",
        node.name.0,
        node,
        operand_text(&node.src).trim_start()
    );
    for operand in node.dsts.iter() {
        out.push_str(&operand_text(operand));
    }
    out.push(']');
    out
}

/// Replaces: e080_allDimsCovered
///
/// Whether the coordinate already has a fold for every dim of that labelled data structure's
/// layout order.
#[must_use]
pub fn all_dims_covered(dsc: &impl Dsc, coord: &dsc2::Coordinate, lds: LdsIdx) -> bool {
    dsc.layout_dims(lds).iter().all(|dim| coord.covers(dim))
}

/// WHAT A BROADCAST DIM'S ELEMENT ARRANGEMENT COSTS — the reference's `scale`, which reaches
/// `buildFoldForBroadcastDim` only when NEGATIVE (`ddc/ddc_fold.cpp:2301`, `:3975`, `:4284`) and
/// there distinguishes exactly one value, `-2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastScale {
    /// Any negative scale other than `-2`: the arrangement is one element.
    Unit,
    /// `scale == -2`: the dim's own cumulative stick size (`ddc/ddc_fold.cpp:94`).
    CumulativeStickSize(FoldCardinality),
}

/// Replaces: e081_buildFoldForBroadcastDim
///
/// Gives a broadcast dim its element-arrangement fold plus cardinality-1 row-split, corelet and
/// core-workslice folds, each pushed to the FRONT — which is why the dim ends up with
/// `FoldPosition::Core`, `Corelet` and `RowSplit` at positions 0, 1 and 2.
///
/// ⛔ TRAP CLOSED: `getCumulativeStickSizes(lds.dsType_).at(dim)` throws for a dim those sizes omit.
/// The cardinality arrives resolved in `BroadcastScale`, which is the only fact the reference's
/// `currDsc` and `lds` arguments were reached through.
pub fn build_fold_for_broadcast_dim(
    coord: &mut dsc2::Coordinate,
    dim: PrimaryDim,
    scale: BroadcastScale,
) {
    let elem_arr_card = match scale {
        BroadcastScale::Unit => FoldCardinality(1),
        BroadcastScale::CumulativeStickSize(card) => card,
    };
    let alpha = FoldCoeff(i64::from(elem_arr_card.0));
    let fold_dim_str = dim.spelling();

    coord.add_fold_front(
        dim,
        CoordinateCategory::ElemArr,
        elem_arr_card,
        dsc2::FoldLabel("elem_arr_0".to_owned()),
        FoldCoeff(0),
        FoldCoeff(0),
    );
    for label in [
        format!("rowsplit_fold_{fold_dim_str}"),
        format!("corelet_fold_{fold_dim_str}"),
        format!("core_workslice_fold_{fold_dim_str}"),
    ] {
        coord.add_fold_front(
            dim,
            CoordinateCategory::Spatial,
            FoldCardinality(1),
            dsc2::FoldLabel(label),
            alpha,
            FoldCoeff(0),
        );
    }
}

/// WHICH PT ROW A COMPONENT SITS ON — the ordinal `senCompToRowId`
/// (`sys-arch-spec/arch_enums.cpp:296`) gives it, which is the digit in the component's own name.
///
/// ⛔ ARCH-BLIND ON PURPOSE, so it is NOT `units::PtRow`. That table names rows 0..7 whatever the
/// arch's row count, and callers compare row ids for equality and adjacency
/// (`ddc/ddc_fold.cpp:768`, `:1184`); folding "row 7 on a 4-row arch" into the same absence as "not
/// a row component" would make those comparisons lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PtRowId(u8);

impl PtRowId {
    /// The row ordinal, 0..=7.
    #[must_use]
    pub const fn ordinal(self) -> u8 {
        self.0
    }

    /// `abs(refRowId - getCompRowId(propDestUnit)) <= 1` (`ddc/ddc_fold.cpp:1184`).
    #[must_use]
    pub const fn is_adjacent_or_same(self, other: Self) -> bool {
        self.0.abs_diff(other.0) <= 1
    }
}

/// Replaces: e082_getCompRowId
///
/// The PT row a component is on, absent where it is on none — the reference's `-1`.
#[must_use]
pub fn comp_row_id(comp: SenComponent) -> Option<PtRowId> {
    let row = match comp {
        SenComponent::Ptrow0
        | SenComponent::Ptrow0_0
        | SenComponent::Ptrow0_1
        | SenComponent::L0lurow0
        | SenComponent::L0lurow0_0
        | SenComponent::L0lurow0_1 => 0,
        SenComponent::Ptrow1
        | SenComponent::Ptrow1_0
        | SenComponent::Ptrow1_1
        | SenComponent::L0lurow1
        | SenComponent::L0lurow1_0
        | SenComponent::L0lurow1_1 => 1,
        SenComponent::Ptrow2
        | SenComponent::Ptrow2_0
        | SenComponent::Ptrow2_1
        | SenComponent::L0lurow2
        | SenComponent::L0lurow2_0
        | SenComponent::L0lurow2_1 => 2,
        SenComponent::Ptrow3
        | SenComponent::Ptrow3_0
        | SenComponent::Ptrow3_1
        | SenComponent::L0lurow3
        | SenComponent::L0lurow3_0
        | SenComponent::L0lurow3_1 => 3,
        SenComponent::Ptrow4
        | SenComponent::Ptrow4_0
        | SenComponent::Ptrow4_1
        | SenComponent::L0lurow4
        | SenComponent::L0lurow4_0
        | SenComponent::L0lurow4_1 => 4,
        SenComponent::Ptrow5
        | SenComponent::Ptrow5_0
        | SenComponent::Ptrow5_1
        | SenComponent::L0lurow5
        | SenComponent::L0lurow5_0
        | SenComponent::L0lurow5_1 => 5,
        SenComponent::Ptrow6
        | SenComponent::Ptrow6_0
        | SenComponent::Ptrow6_1
        | SenComponent::L0lurow6
        | SenComponent::L0lurow6_0
        | SenComponent::L0lurow6_1 => 6,
        SenComponent::Ptrow7
        | SenComponent::Ptrow7_0
        | SenComponent::Ptrow7_1
        | SenComponent::L0lurow7
        | SenComponent::L0lurow7_0
        | SenComponent::L0lurow7_1 => 7,
        // The other 59 components are not on a row, which is what `senCompToRowId.count(comp) == 0`
        // says, and a component added to the enum is not on one either.
        _ => return None,
    };
    Some(PtRowId(row))
}

/// Replaces: e083_getComponent
///
/// The component a node is at: an allocation's own, a compute's execution unit, or a transfer's
/// source or FIRST destination. The side is read for a transfer and ignored otherwise.
///
/// ⛔ TRAP CLOSED: `"Unsupported node type"` is unspellable over `Node`, and `dstVias_.at(0)` cannot
/// throw because `Dsts` is non-empty.
#[must_use]
pub fn component(node: Node<'_>, side: TransferSide) -> SenComponent {
    match node {
        Node::Allocate(alloc) => alloc.component,
        Node::Compute(compute) => compute.ex_unit,
        Node::Transfer(transfer) => match side {
            TransferSide::Src => transfer.src.unit,
            TransferSide::Dst => transfer.dsts.first().unit,
        },
    }
}

/// Replaces: e084_getLayoutDimsFromNode
///
/// The layout order of the labelled data structure one named operand of a node reads or writes, and
/// no dims where that operand has none — the reference's `myLdsIdx_ == -1`.
///
/// ⛔ The reference's `DT_ERROR` for "neither input nor output specified" is unspellable, and its
/// `DT_CHECK(inputPos == 0)` on a transfer cannot fail because its one transfer caller passes a
/// literal 0 (`ddc/ddc_fold.cpp:802`); an input position past 0 here reads the src regardless.
/// ⛔ TRAP: a position past the operand list gives no dims where the reference's `.at()` throws.
#[must_use]
pub fn layout_dims_from_node(dsc: &impl Dsc, node: Node<'_>, pos: OperandPos) -> Vec<PrimaryDim> {
    let lds = match (node, pos) {
        (Node::Allocate(alloc), _) => alloc.lds,
        (Node::Transfer(transfer), OperandPos::Input(_)) => transfer.src.data.my_lds_idx,
        (Node::Transfer(transfer), OperandPos::Output(at)) => {
            transfer.dsts.get(at).and_then(|dst| dst.data.my_lds_idx)
        }
        (Node::Compute(compute), OperandPos::Input(at)) => compute
            .inputs
            .get(at)
            .and_then(|input| input.data.my_lds_idx),
        (Node::Compute(compute), OperandPos::Output(at)) => compute
            .outputs
            .get(at)
            .and_then(|output| output.data.my_lds_idx),
    };
    lds.map(|lds| dsc.layout_dims(lds).to_vec())
        .unwrap_or_default()
}

/// Replaces: e085_needToConsiderRowBundling
///
/// Whether the core stage's FIRST row-split dim — lowest `PrimaryDimTypes` ordinal, which is what
/// `rowSplit_.begin()` yields — is one of the dims being worked on. No row split at all is `false`.
///
/// ⛔ The reference's `refCoord` argument is never read (`ddc/ddc_fold.cpp:190-198`).
#[must_use]
pub fn need_to_consider_row_bundling<S: Stage>(core_ds: &S, working_dims: &[PrimaryDim]) -> bool {
    PrimaryDim::ALL
        .into_iter()
        .find(|&dim| core_ds.is_row_split(dim))
        .is_some_and(|row_split_dim| working_dims.contains(&row_split_dim))
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⛔ THIS FILE CARRIES THE FOLD VOCABULARY TWICE — entries 078-085 landed on
// `crate::schedule::dsc2`, entries 086-095 on the block below. THE REVIEW WALKED ALL FIVE PAIRS AND
// ONLY ONE WAS ONE C++ DECLARATION TYPED TWICE; the premise that they all were is withdrawn.
// `LdsIdx` was identical and is now `dsc2`'s alone:
//   `dsc2::CoordinateCategory` / `CoordCategory` — CONVERGED BY THIS REVIEW: one declaration
//     (`dsc/dsc2.h:66`), same three variants, same dropped `UNKNOWN_COORD`, so `CoordCategory` is
//     gone and this file uses `dsc2::CoordinateCategory`.
//   `dsc2::FoldCardinality(u32)` / `Cardinality(u64)` — NOT ONE DECLARATION: `FoldDimProp::factor_`
//     is `uint32_t` (`foldInfrastructure.h:153`) and `FoldParamInfoType::cardinality` is `int64_t`
//     (`dsc/dsc2.h:1083`). Two widths, two types; converging them would erase one.
//   `dsc2::FoldCoeff` / `Alpha` + `Beta` — both are `CoordinateBaseType`, but one newtype for both
//     coefficients lets a caller transpose them and two do not, so the SPLIT pair should win.
//   `dsc2::FoldLabel(String)` / `FoldLabel` (4 variants) — the enum should win, because the crate
//     rule is that a closed set is an enum. Entry 081 mints `"rowsplit_fold_" + dim`, a DIFFERENT
//     label from entry 094's bare `"rowsplit_fold"` (`ddc/ddc_fold.cpp:100` vs `:2158`), so the set
//     covering both needs a dim-carrying variant first.
//   `dsc2::Coordinate` / `Coordinate` — NOT A DUPLICATE: one is the concrete map, the other is this
//     file's two-operation seam. ⭐ THE GAP IS CLOSED: `impl Coordinate for dsc2::Coordinate` below
//     is what lets entry 093 be pointed at a real coordinate.
// ════════════════════════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE SCHEDULE TREE, ITS ALLOCATIONS AND ITS FOLDS — as entries 086-093 read them.
//
// ⭐ THE TRAITS ARE THE MECHANISM FOR REACHING OPERANDS, which the campaign statement names as the
// one droppable part: `getPrev`, `getNextView`, `getAllocation`, `calculate_padded` and
// `dataStageDimToVal_compView_st` are all `dsc/dsc2.cpp` / `dsc/dims.cpp` — outside this campaign's
// file list (`crustify-ddc/OUTSIDE-DEPS.tsv`, "dsc2 tree utilities"). What these units OWN is the
// decision and the mutation, and both are below.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// `dsc2::ScheduleNode::NodeType` (`dsc/dsc2.h:446`), less `INVALID` — that is the field's unset
/// sentinel, never a node in a built tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeKind {
    /// `BLOCK`.
    Block,
    /// `LOOP`.
    Loop,
    /// `TRANSFER`.
    Transfer,
    /// `COMPUTE`.
    Compute,
    /// `SYNC`.
    Sync,
    /// `CONDITION`.
    Condition,
    /// `ALLOCATE`.
    Allocate,
    /// `STICKMASK`.
    StickMask,
}

/// ONE SCHEDULE NODE'S IDENTITY — the reference compares `ScheduleNode*`, and every question these
/// units put to a node is asked of the tree that owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// ONE ALLOCATION'S IDENTITY — the reference compares `AllocateNode*`, and entries 086 and 091 do
/// nothing with one but compare it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocId(pub u32);

/// A `constantInfo_` index, positive by type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstIdx(pub u32);

/// WHERE A DATASTREAM'S DATA COMES FROM — `DataInfo::myLdsIdx_` / `constantId_`
/// (`dsc/dsc2.h:722,726`), exactly one of which is set.
///
/// ⛔⛔ TWO `DT_ERROR`s COLLAPSE INTO THIS ENUM. `getAllocation` aborts when NEITHER is set
/// (`dsc/dsc2.cpp:2589-2594`) and `isLabeledDs`/`isConstant` `DT_CHECK` that BOTH are not
/// (`dsc/dsc2.h:741-751`); an enum can be in neither state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataOrigin {
    /// `myLdsIdx_ >= 0`.
    LabeledDs(LdsIdx),
    /// `constantId_ >= 0`.
    Constant(ConstIdx),
}

/// ONE DATASTREAM END — `dsc2::DataInfo` (`dsc/dsc2.h:721`) reduced to the two fields entries 086
/// and 091 read: the allocation key and the `data_connect=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataStream {
    /// `myLdsIdx_` / `constantId_` — what `getAllocation` keys on.
    pub origin: DataOrigin,
    /// `dataConnect_`, absent for the reference's `""`. ⭐ THE PORT IDENTITY IS THE `data_connect=`,
    /// so this is a generated closed set and not a string.
    pub data_connect: Option<DataConnect>,
}

/// A DATASTREAM END TOGETHER WITH THE MEMORY IT LIVES IN — the reference's parallel
/// `inputsLdsAndLoopOffsets_.at(i)` / `inputs_.at(i)` pair (`ddc/ddc_fold.cpp:221-222`), zipped so
/// the two cannot disagree in length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredStream {
    /// The stream.
    pub stream: DataStream,
    /// `src_.storage_` / `inputs_.at(i)` — the memory `getAllocation` looks the stream up in.
    pub storage: DfirUnit,
}

/// `dsc2::ScheduleNode` reduced to the three questions entries 087 and 088 put to a tree.
pub trait ScheduleTree {
    /// `nodeType_`.
    fn kind(&self, node: NodeId) -> NodeKind;
    /// `getPrev()` (`dsc/dsc2.h:463`) — the enclosing block, loop or condition, absent at the root.
    fn parent(&self, node: NodeId) -> Option<NodeId>;
    /// `BlockNode::getNextView(SenComponents::ALL)` (`dsc/dsc2.cpp`) — EVERY child in schedule
    /// order, because the `ALL` arm of `isNodeRelevant` filters nothing at all.
    fn children(&self, block: BlockId) -> Vec<NodeId>;
}

/// A NODE PROVED TO BE A `BLOCK` — `DT_CHECK_MSG(commonAncestor->nodeType_ == BLOCK, "... is not a
/// block node")` (`ddc/ddc_fold.cpp:271-273`) made unconstructible.
///
/// ⛔ `BLOCK` ONLY, NOT `isBlockNode()`: a `LOOP` and a `CONDITION` also have children, and entry
/// 088's check refuses both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(NodeId);

impl BlockId {
    /// The block, or [`None`] for the node kind entry 088 refuses.
    #[must_use]
    pub fn of<T: ScheduleTree + ?Sized>(tree: &T, node: NodeId) -> Option<Self> {
        (tree.kind(node) == NodeKind::Block).then_some(BlockId(node))
    }

    /// The node this block is.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.0
    }
}

/// `DesignSpaceConfig`'s ALLOCATION LOOKUP — `getAllocation(di, storage, /*allowMissingAlloc=*/
/// true)` (`dsc/dsc2.cpp:2586`) and the scale-to-value redirect beside it.
pub trait Allocations {
    /// The allocation this stream has in that memory, absent where the reference returns `nullptr`.
    fn allocation(&self, stored: StoredStream) -> Option<AllocId>;

    /// `dsc2::getValueAllocation(dsc, allocNode)` (`dsc/dsc2.cpp:5819`) — the VALUE allocation a
    /// scale allocation is implicitly connected through, absent for the two lookups that abort
    /// there (an unsupported `component_`, or a value labeled DS with no `memOrg_` for it).
    fn value_allocation(&self, scale: AllocId) -> Option<AllocId>;
}

/// A NODE PROVED TO BE AN ALLOCATION'S USER *AND* TO CARRY INCOMING DATASTREAMS.
///
/// ⛔⛔ BOTH OF ENTRY 086'S `DT_ERROR`s ARE THIS TYPE. `hasAllocUser(node)` false gives *"ScheduleNode
/// <n> is not a user of the allocateNode <a>"* (`ddc/ddc_fold.cpp:205-207`) and any `nodeType_` but
/// `TRANSFER`/`COMPUTE` gives *"Unsupported nodeType for scheduleNode <n>"* (`:229`). Neither is
/// reachable from a value of this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Incoming<'n>(IncomingStreams<'n>);

/// WHAT ONE USER READS ITS DATA THROUGH — the two `nodeType_` arms entry 086 supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncomingStreams<'n> {
    /// `TRANSFER` — `srcLdsAndLoopOffsets_` in `src_.storage_`, the ONE incoming end. ⛔ THE
    /// DESTINATIONS ARE NOT LOOKED AT: a transfer whose *destination* is this allocation is not
    /// "incoming".
    Transfer(StoredStream),
    /// `COMPUTE` — every input, and the outputs likewise unread.
    Compute(&'n [StoredStream]),
}

impl<'n> Incoming<'n> {
    /// The witness, or [`None`] for either refusal: a node absent from `allocUsers_`, or one whose
    /// kind carries no incoming datastreams.
    #[must_use]
    pub fn of(
        alloc_users: &[NodeId],
        node: NodeId,
        streams: Option<IncomingStreams<'n>>,
    ) -> Option<Self> {
        alloc_users.contains(&node).then_some(())?;
        streams.map(Incoming)
    }
}

/// A FOLD'S STRIDE — `FoldParamInfoType::alpha` / `addFold`'s `alpha`, a `CoordinateBaseType`
/// (`#define CoordinateBaseType int64_t`, `dsc/dsc2.h:442`). Signed: an alpha of 0 is a broadcast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Alpha(pub i64);

/// A FOLD'S OFFSET — `FoldParamInfoType::beta`, likewise signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Beta(pub i64);

/// A FOLD'S TRIP COUNT — `FoldParamInfoType::cardinality` / `addFold`'s `foldCardinality`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cardinality(pub u64);

impl Cardinality {
    /// A LAYOUT DIM'S CAPPED EXTENT AS A FOLD'S TRIP COUNT — the reference's
    /// `newFoldParam.cardinality = maxDimSize` (`ddc/ddc_fold.cpp:366`): the split fold steps once
    /// per capped chunk, so the cap is the trip count.
    #[must_use]
    pub const fn of_capped_extent(extent: Elements) -> Self {
        Self(extent.0)
    }
}

/// A PADDED DIMENSION'S STRIDE — `DimPaddingSizes::stride_` (`dsc/dims.h:140`), whose default is 1
/// and not 0, so an unpadded dim strides by one element rather than standing still.
///
/// ⛔ NON-ZERO BY TYPE BECAUSE ENTRY 328 DIVIDES BY IT: `totPadding / padInfo.stride_`
/// (`L3DlOpsScheduler.cpp:879`) has no guard at all, so a stride of zero is a DIVIDE-BY-ZERO in the
/// reference. A window that never advances is not a stride, and this is where that is stated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stride(NonZeroI64);

impl Stride {
    /// The unpadded stride the reference initialises `stride` to (`ddc/ddc_fold.cpp:2115`).
    pub const ONE: Stride = Stride(NonZeroI64::new(1).unwrap());

    /// A stride as the reference stores it, [`None`] for the standing-still zero.
    #[must_use]
    pub const fn new(stride: i64) -> Option<Self> {
        match NonZeroI64::new(stride) {
            Some(stride) => Some(Self(stride)),
            None => None,
        }
    }

    /// `stride_` itself.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0.get()
    }
}

/// A PADDED DIMENSION'S DILATION — `DimPaddingSizes::dilation_` (`dsc/dims.h:141`), whose default is
/// likewise 1: a window with no dilation steps by one element between taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Dilation(pub i64);

impl Dilation {
    /// The undilated step the reference initialises `dilation_` to.
    pub const ONE: Dilation = Dilation(1);
}

/// WHICH PADDING FORM A DIM CARRIES — `PadType` (`dsc/dims.h:50`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PadType {
    /// `NOPAD`.
    NoPad,
    /// `LOWERED_PADDED`.
    LoweredPadded,
    /// `PADDED_NOZEROPAD`.
    PaddedNoZeroPad,
    /// `PADDED_WZEROPAD`.
    PaddedWZeroPad,
    /// `PADDED_FULLSPAN`.
    PaddedFullSpan,
    /// `PADDED_FULLSPAN_WUNNEEDED`.
    PaddedFullSpanWUnneeded,
}

/// A FOLD'S LABEL — `addFold`'s `foldLabel` / `FoldParamInfoType::foldDimLabel`, the closed set of
/// spellings this file mints. A closed set is an enum here, never a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FoldLabel {
    /// `"corelet_fold_dim"` (`ddc/ddc_fold.cpp:2135`).
    CoreletFoldDim,
    /// `"core_workslice_fold_dim"` (`:2149`).
    CoreWorksliceFoldDim,
    /// `"elem_arr_layout_split"` (`:370`).
    ElemArrLayoutSplit,
    /// `"rowsplit_fold"` (`:2158,2172`).
    RowSplitFold,
    /// `"elem_arr_scaleup"` (`:519`) — the level entry 234 inserts for the MX scale block.
    ElemArrScaleUp,
}

impl FoldLabel {
    /// The reference's spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::CoreletFoldDim => "corelet_fold_dim",
            Self::CoreWorksliceFoldDim => "core_workslice_fold_dim",
            Self::ElemArrLayoutSplit => "elem_arr_layout_split",
            Self::RowSplitFold => "rowsplit_fold",
            Self::ElemArrScaleUp => "elem_arr_scaleup",
        }
    }
}

/// ONE FOLD LEVEL — `dsc2::FoldParamInfoType` (`dsc/dsc2.h:1081`).
///
/// ⭐ INDEX 0 IS THE OUTERMOST LEVEL, throughout: `combineContigousLevels`' own diagram says so
/// (`ddc/ddc_fold.cpp:391-392`) and `FoldManager::buildDim` inserts *before* `pos`
/// (`util/foldManager/foldInfrastructure.h:1350`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldParamInfo {
    /// `alpha` — the stride between consecutive trips of this level.
    pub alpha: Alpha,
    /// `beta` — the offset added at this level.
    pub beta: Beta,
    /// `cardinality` — this level's trip count.
    pub cardinality: Cardinality,
    /// `foldDimLabel`.
    pub label: Option<FoldLabel>,
}

/// A FOLD ABOUT TO BE ADDED TO A COORDINATE — `addFold`'s arguments other than the dim and the
/// category (`dsc/dsc2.h:120`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold {
    /// `foldCardinality`.
    pub cardinality: Cardinality,
    /// `foldLabel`, absent for the reference's `""` — which is what a re-added [`FoldParamInfo`]
    /// carries whenever the fold manager it was gathered off had no label for the level.
    pub label: Option<FoldLabel>,
    /// `alpha`.
    pub alpha: Alpha,
    /// `beta`.
    pub beta: Beta,
}

/// `dsc2::CoordinateType<CoordinateBaseType>` reduced to the one operation entry 093 performs.
pub trait Coordinate {
    /// `addFold(dim, coordCat, cardinality, label, alpha, beta, /*pos=*/0)` (`dsc/dsc2.h:120`).
    ///
    /// ⛔⛔ `pos = 0` IS THE OUTERMOST POSITION, so the fold added SECOND ends up OUTSIDE the one
    /// added first (`foldInfrastructure.h:1350`, "insert new node before pos").
    fn add_fold(&mut self, dim: PrimaryDim, category: CoordinateCategory, fold: Fold);

    /// `clearFoldForDim(dim)` (`dsc/dsc2.h:99`) — drops the dim's folds and zeroes its three counts,
    /// WITHOUT dropping the dim: entries 234 and 235 clear then re-add a rebuilt list.
    fn clear_fold_for_dim(&mut self, dim: PrimaryDim);
}

/// ⭐ THE SEAM MEETS THE CONCRETE COORDINATE. `dsc2::Coordinate` is the coordinate the schedule tree
/// actually carries; this impl is what lets entry 093 and everything above it be pointed at one,
/// which is the gap this file's own vocabulary note recorded.
///
/// ⚠️ `dsc2::FoldCardinality` IS A `u32` AND `addFold`'s own parameter an `int` (`dsc/dsc2.h:118`),
/// so a trip count with no spelling on the other side saturates rather than refusing.
impl Coordinate for dsc2::Coordinate {
    fn add_fold(&mut self, dim: PrimaryDim, category: CoordinateCategory, fold: Fold) {
        let label = fold
            .label
            .map_or_else(String::new, |label| label.spelling().to_owned());
        self.add_fold_front(
            dim,
            category,
            FoldCardinality(u32::try_from(fold.cardinality.0).unwrap_or(u32::MAX)),
            dsc2::FoldLabel(label),
            FoldCoeff(fold.alpha.0),
            FoldCoeff(fold.beta.0),
        );
    }

    fn clear_fold_for_dim(&mut self, dim: PrimaryDim) {
        dsc2::Coordinate::clear_fold_for_dim(self, dim);
    }
}

/// HOW MANY ELEMENT-ARRANGEMENT LEVELS OF THE FOLD LIST BELONG TO ONE ALLOCATE NODE — the
/// reference's `numElemArrFoldsOfAllocNode`, counted from the INNERMOST end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElemArrFolds(pub u32);

/// WHICH ELEMENT-ARRANGEMENT LEVEL A LOOP WAS RELATED TO — `LoopDistributionParamType::
/// relatedElemArrLevel` (`dsc/dsc2.h:1113`), counted 1-based from the INNERMOST level:
/// `currElemArrLevel = foldParams.size() - currElemArrIndex` (`ddc/ddc_fold.cpp:341`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElemArrLevel(pub u32);

/// `dsc2::LoopDistributionParamPerNodeType` (`dsc/dsc2.h:1119`) reduced to the ONE rewrite entry
/// 089 performs on it — the whole map stays with whoever owns it.
pub trait LoopLevels {
    /// `++loopInfo.at(dim).relatedElemArrLevel` for EVERY loop whose `dim` entry sits at or inside
    /// `level` (`ddc/ddc_fold.cpp:376-381`) — one insert shifted them all by one.
    fn bump_levels_at_or_inside(&mut self, dim: PrimaryDim, level: ElemArrLevel);
}

/// AN ALLOCATE NODE'S LAYOUT — `layoutDimOrder_` zipped with `maxDimSizes_`
/// (`dsc/dsc2.h:982-983`), so the reference's parallel `.at(i)` pair cannot disagree in length.
///
/// ⛔ THE CAP IS AN `Option`, NOT A NUMBER: the reference's `maxDimSize > 0` test
/// (`ddc/ddc_fold.cpp:358`) folds zero and every negative into one absence, and absence stops the
/// split rather than capping at nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocLayout(pub Vec<(PrimaryDim, Option<Elements>)>);

/// THE ALLOCATION'S OWN SIZE DATA STAGE — `getSizeDataStageForNode(allocNode, allocNode).ss_`
/// (`dsc/designSpaceConfig.h:264`) with the allocation's own `padding_` already bound to it.
pub trait SizeStage {
    /// `calculate_padded(dim, val, allocNode->padding_, /*getSymbolicGranularity=*/false)`
    /// (`dsc/dims.cpp`) — the padded span of `val` unpadded elements along `dim`.
    fn padded(&self, dim: PrimaryDim, val: i64) -> Alpha;
}

/// THE CORE DATA STAGE ENTRY 093 FOLDS OVER — `dataStageParam_.at(metadata.core_dstgid).ss_`, whose
/// id is fixed at 0 (`ddc/ddc_metadata.h:211`), plus the two counts beside it.
pub trait CoreStage {
    /// `paddingSizes_.at(dim).stride_` (`dsc/dims.h:219,140`), absent where the stage records no
    /// padding sizes for the dim.
    fn pad_stride(&self, dim: PrimaryDim) -> Option<Stride>;
    /// `coreletSplit_.at(dim).at(0)` — the FIRST corelet's share, absent where the dim is not
    /// corelet-split at all, and also where its split list is EMPTY, which the reference `.at(0)`s.
    fn first_corelet_share(&self, dim: PrimaryDim) -> Option<Extent>;
    /// `dataStageDimToVal_compView_st(dim, SenComponents::NO_COMPONENT, -1, {})`
    /// (`dsc/dims.h:277`) — one core's whole extent for the dim, with no component, corelet or
    /// padding view.
    fn core_extent(&self, dim: PrimaryDim) -> Extent;
    /// `currDsc->numCoreletsUsed_` (`dsc/designSpaceConfig.h:74`).
    fn corelets_used(&self) -> Cardinality;
    /// `sdsc_->numWkSlicesPerDim_.at(dim)` (`dsc/superdsc.h:69`).
    fn work_slices(&self, dim: PrimaryDim) -> Cardinality;
}

/// `LabeledDsInfo::ScaledLdsCategory` (`dsc/dscdefn.h:352`) — whether a labeled DS holds values,
/// MX scales, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScaledLds {
    /// `REGULAR_TENSOR`.
    Regular,
    /// `VALUE_TENSOR`.
    Value,
    /// `SCALE_TENSOR` — an MX scale tensor.
    Scale,
}

/// ONE END OF A COORDINATE PROPAGATION — `CoordPropInfoType`'s `refNode` / `nodeToFold`
/// (`dsc/dsc2.h:1089-1090`) reduced to the two kinds entry 091 distinguishes.
///
/// ⛔⛔ AND THAT REMOVES A `static_cast` THE REFERENCE PERFORMS UNCONDITIONALLY. `comp =
/// static_cast<const ComputeNode*>(nodeToFold)->exUnit_` (`ddc/ddc_fold.cpp:447-448`) runs whether
/// or not the node is a compute; the value is only READ under `otherNodeIsCompute`, so the read is
/// harmless but the cast is not. Here `exUnit_` exists only where there is a compute to read it off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropEnd {
    /// `ALLOCATE` — the allocation, and the labeled-DS category that decides whether a match is
    /// redirected to its value tensor. [`None`] covers the reference's `ldsIdx_ == -1` (`:462`).
    Allocate {
        /// The allocation itself.
        alloc: AllocId,
        /// `labeledDs_.at(ldsIdx_).scaledLdsCategory_`.
        scaled: Option<ScaledLds>,
    },
    /// `COMPUTE` — `exUnit_`, the unit the compute runs on.
    Compute {
        /// `exUnit_`.
        ex_unit: DfirUnit,
    },
    /// Any other `nodeType_`: entry 091 treats it as neither an allocation nor a compute.
    Other,
}

/// ONE PENDING COORDINATE PROPAGATION — `dsc2::CoordPropInfoType` (`dsc/dsc2.h:1087`) reduced to
/// the three fields entry 091 reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordPropInfo {
    /// `dataConnect`, absent for the reference's `""`.
    pub data_connect: Option<DataConnect>,
    /// `refNode`.
    pub ref_node: PropEnd,
    /// `nodeToFold`.
    pub node_to_fold: PropEnd,
}

/// Replaces: e086_isAllocateIncoming
///
/// Whether the allocation is on the INCOMING side of this user — a transfer's source, or any one of
/// a compute's inputs.
///
/// ⛔ `getAllocation` IS CALLED WITH `allowMissingAlloc`, so a stream with no allocation in its own
/// storage is a MISMATCH and not a stop.
#[must_use]
pub fn is_allocate_incoming<A: Allocations + ?Sized>(
    dsc: &A,
    alloc: AllocId,
    user: &Incoming<'_>,
) -> bool {
    match user.0 {
        IncomingStreams::Transfer(src) => dsc.allocation(src) == Some(alloc),
        IncomingStreams::Compute(inputs) => inputs
            .iter()
            .any(|&input| dsc.allocation(input) == Some(alloc)),
    }
}

/// Replaces: e087_findCommonAncestor
///
/// The INNERMOST enclosing node of one of the given kinds that encloses EVERY node in the list.
///
/// ⛔ CANDIDATES COME FROM `nodes[0]`'s CHAIN ALONE (`:240-246`), collected walking OUTWARD, so the
/// answer always sits on that node's spine and the first with a full count is the innermost.
/// ⛔ THE VOTE IS PER LIST ENTRY, NOT PER DISTINCT NODE (`:249-257`): a node named twice votes twice,
/// so a duplicated list still answers with that node's own innermost ancestor.
#[must_use]
pub fn find_common_ancestor<T: ScheduleTree + ?Sized>(
    tree: &T,
    nodes: &[NodeId],
    ancestor_kinds: &[NodeKind],
) -> Option<NodeId> {
    let mut candidates: Vec<(NodeId, usize)> = Vec::new();
    let mut curr = tree.parent(*nodes.first()?);
    while let Some(parent) = curr {
        if ancestor_kinds.contains(&tree.kind(parent)) {
            candidates.push((parent, 0));
        }
        curr = tree.parent(parent);
    }
    for &node in nodes {
        let mut curr = tree.parent(node);
        while let Some(parent) = curr {
            if let Some(seen) = candidates.iter_mut().find(|(cand, _)| *cand == parent) {
                seen.1 += 1;
            }
            curr = tree.parent(parent);
        }
    }
    candidates
        .iter()
        .find(|(_, count)| *count == nodes.len())
        .map(|&(cand, _)| cand)
}

/// Replaces: e088_orderDescendants
///
/// The ancestor's OWN CHILDREN that enclose the given nodes, in schedule order.
///
/// ⛔⛔ CHILDREN, NOT THE NODES THEMSELVES: two entries under one child yield that child ONCE, so the
/// result is shorter than the input, and a node not under this ancestor contributes nothing.
/// ⭐ The reference APPENDS to a caller-owned vector without clearing it (`:270`); returning the list
/// makes that the caller's decision.
#[must_use]
pub fn order_descendants<T: ScheduleTree + ?Sized>(
    tree: &T,
    ancestor: BlockId,
    nodes: &[NodeId],
) -> Vec<NodeId> {
    let mut ancestor_children: Vec<NodeId> = Vec::new();
    for &node in nodes {
        // Find the child of the ancestor that contains node.
        let mut curr = Some(node);
        while let Some(walked) = curr {
            if tree.parent(walked) == Some(ancestor.node()) {
                ancestor_children.push(walked);
                break;
            }
            curr = tree.parent(walked);
        }
    }
    tree.children(ancestor)
        .into_iter()
        .filter(|child| ancestor_children.contains(child))
        .collect()
}

/// Replaces: e089_constructAllocElemArrLayout
///
/// Splits the innermost non-unit elem-arr level into `maxDimSizes_` chunks along `coord_dim`.
///
/// ⛔⛔ HALF THE BODY IS DEAD AND IS NOT REPRODUCED: the whole `stickDimSize` map (`:310-327`) is read
/// only by an `if` whose body is COMMENTED OUT (`:355-357`), and `layoutDims` (`:344-345`) not at all.
/// ⛔⛔ `remainingDimSizes` IS A REFERENCE INTO THE VECTOR BEING INSERTED INTO (`:343`, `:371-372`);
/// every `DT_CHECK` (`:362`) is decided HERE first, so [`None`] leaves the fold list untouched.
pub fn construct_alloc_elem_arr_layout<S: SizeStage + ?Sized, L: LoopLevels + ?Sized>(
    stage: &S,
    layout: &AllocLayout,
    coord_dim: PrimaryDim,
    fold_params: &mut Vec<FoldParamInfo>,
    elem_arr_folds: &mut ElemArrFolds,
    loop_levels: &mut L,
) -> Option<ElemArrFolds> {
    let levels = i64::try_from(fold_params.len()).ok()?;
    let outermost = levels - i64::from(elem_arr_folds.0);
    let mut curr = levels - 1;
    while curr >= outermost
        && curr >= 0
        && fold_params[usize::try_from(curr).ok()?].cardinality == Cardinality(1)
    {
        curr -= 1;
    }
    if curr < outermost || curr < 0 {
        // All element arrangement levels have cardinality 1. Can not split any
        // element arrangement.
        return Some(ElemArrFolds(0));
    }
    let curr = usize::try_from(curr).ok()?;
    let curr_level = ElemArrLevel(u32::try_from(fold_params.len() - curr).ok()?);

    // Decide the whole chunk chain before touching the fold list.
    let mut remaining = fold_params[curr].cardinality;
    let mut chunks: Vec<Cardinality> = Vec::new();
    for &(dim, max_size) in &layout.0 {
        if dim != coord_dim {
            continue;
        }
        // In case that the dim size is larger than the max allowed size specified by user, set dim
        // size to maxDimSize and reset the remaining sizes.
        let Some(max_size) = max_size.filter(|max| remaining.0 > max.0) else {
            break;
        };
        if max_size.0 == 0 || remaining.0 % max_size.0 != 0 {
            return None;
        }
        remaining = Cardinality(remaining.0 / max_size.0);
        chunks.push(Cardinality::of_capped_extent(max_size));
    }

    let mut inner_card: u64 = 1;
    for chunk in &chunks {
        fold_params.insert(
            curr + 1,
            FoldParamInfo {
                alpha: stage.padded(coord_dim, i64::try_from(inner_card).ok()?),
                beta: Beta(0),
                cardinality: *chunk,
                label: Some(FoldLabel::ElemArrLayoutSplit),
            },
        );
        elem_arr_folds.0 += 1;
        inner_card = inner_card.wrapping_mul(chunk.0);
        // Adjust elemArr indices from previous iterations.
        loop_levels.bump_levels_at_or_inside(coord_dim, curr_level);
    }
    fold_params[curr].cardinality = remaining;
    fold_params[curr].alpha = Alpha(fold_params[curr].alpha.0.wrapping_mul(inner_card as i64));
    Some(ElemArrFolds(u32::try_from(chunks.len()).ok()?))
}

/// Replaces: e090_combineContigousLevels
///
/// Merges each adjacent pair of levels whose strides already compose into one and drops
/// cardinality-1 levels, scanning inner to outer over `start..=end` (`end` = innermost by default).
///
/// ⛔⛔ THE MERGED ALPHA IS THE INNER ONE, NOT THE COMMENT'S: the diagram (`:409-412`) writes the value
/// the outer level ALREADY had, while the code assigns `alpha_i` (`:414`). The code is right.
/// ⛔ AND THE INNERMOST LEVEL IS NEVER DROPPED for cardinality 1 — that arm only ever erases `i - 1`.
/// ⚠️ AN `end` PAST THE LAST LEVEL IS CLAMPED to it, where the reference's `.at(end)` throws.
pub fn combine_contigous_levels(
    fold_params: &mut Vec<FoldParamInfo>,
    start: usize,
    end: Option<usize>,
) {
    //                     Outermost   innermost
    //                        |         |
    //   foldParams: 0 1 ... start ... end
    //                       <-----------|
    //                       Scan from inner to outer.
    let Some(end) = end.or_else(|| fold_params.len().checked_sub(1)) else {
        return;
    };
    let mut i = end.min(fold_params.len().saturating_sub(1));
    while i > start {
        let outer = fold_params[i - 1];
        let inner = fold_params[i];
        if i128::from(outer.alpha.0) == i128::from(inner.alpha.0) * i128::from(inner.cardinality.0)
        {
            //   [Outer] elemArr(i-1) : ( alpha_(i), card_(i-1) * card_(i))
            //   [Inner] elemArr(i)   : Removed
            fold_params[i - 1].alpha = inner.alpha;
            fold_params[i - 1].cardinality =
                Cardinality(outer.cardinality.0.wrapping_mul(inner.cardinality.0));
            fold_params[i - 1].beta = Beta(outer.beta.0.wrapping_add(inner.beta.0));
            // Remove element arrangement fold at fold index i.
            fold_params.remove(i);
        } else if outer.cardinality == Cardinality(1) {
            // Remove folds of cardinality 1.
            fold_params.remove(i - 1);
        }
        // Move to the next outer fold level.
        i -= 1;
    }
}

/// Replaces: e091_matchDataStream
///
/// Whether this datastream is the propagation's — by `data_connect=` if it names one, else by whether
/// the stream's allocation in `storage` IS the propagation's allocate end.
///
/// ⛔ A COMPUTE ON THE OTHER END REDIRECTS A SCALE TENSOR TO ITS VALUE TENSOR (`:461-471`): computes
/// consume the values and the MX scales ride along. `LXLU` is exempt — an FMUL there converts FP4 to
/// bf16 and touches the scale allocation directly.
/// ⚠️ AN ABSENT VALUE ALLOCATION ANSWERS `false`; the reference aborts inside `getValueAllocation`
/// (`dsc/dsc2.cpp:5824,5842`).
#[must_use]
pub fn match_data_stream<A: Allocations + ?Sized>(
    dsc: &A,
    prop: &CoordPropInfo,
    stored: StoredStream,
    check_ref_for_allocate: bool,
) -> bool {
    if prop.data_connect.is_some() && prop.data_connect == stored.stream.data_connect {
        return true;
    }
    let (alloc_end, other_end) = if check_ref_for_allocate {
        (prop.ref_node, prop.node_to_fold)
    } else {
        (prop.node_to_fold, prop.ref_node)
    };
    let PropEnd::Allocate { alloc, scaled } = alloc_end else {
        return false;
    };
    let mut to_match = alloc;
    if let PropEnd::Compute { ex_unit } = other_end
        && ex_unit != DfirUnit::Lxlu
        && scaled == Some(ScaledLds::Scale)
    {
        // Compute nodes consume/produce the value tensors. The scale tensors are implicitly
        // connected through the corresponding value tensors.
        let Some(value) = dsc.value_allocation(to_match) else {
            return false;
        };
        to_match = value;
    }
    dsc.allocation(stored) == Some(to_match)
}

/// Replaces: e092_getNumElementsInPTSlice
///
/// How many elements of `dim` one PT slice holds — the cumulative stick sizes over one L0 slice, and
/// one element for a dim the stick does not name.
///
/// ⛔ The element arrangement inside a slice is NOT scaled by the row count (`:1520-1522`).
/// ⛔ AND THE REFERENCE'S BOUNDS CHECK IS OFF BY ONE (`ldsIdx > labeledDs_.size()`, `:1523`) — taking
/// the stick's dims resolves the labeled DS before the call, so no index is left to get wrong.
#[must_use]
pub fn num_elements_in_pt_slice<A: Arch>(dims: &StickDims, dim: PrimaryDim) -> Option<Elements> {
    let slice = SliceElems::per_l0_row::<A>(dims)?;
    let sizes = cumulative_stick_sizes(dims, StickPart::WithinSlice(slice))?;
    Some(
        sizes
            .iter()
            .find(|(walked, _)| *walked == dim)
            .map_or(Elements(1), |&(_, elems)| elems),
    )
}

/// Replaces: e093_buildSpatialFold
///
/// Adds this dim's two spatial folds to a node's coordinate — the corelet fold, then the core
/// work-slice fold outside it.
///
/// ⛔⛔ THE CORELET FOLD IS ADDED EVEN FOR A DIM NO CORELET SPLITS, with alpha 0 (`:2129-2131`) — a
/// zero stride over `numCoreletsUsed_` trips is a BROADCAST, not an absent fold. Both go in at
/// `pos = 0`, so the CORE fold ends up OUTERMOST. ⛔ The `node` argument is never read at all.
pub fn build_spatial_fold<S: CoreStage + ?Sized, C: Coordinate + ?Sized>(
    stage: &S,
    dim: PrimaryDim,
    pad: PadType,
    coord: &mut C,
) {
    let stride = match pad {
        PadType::PaddedWZeroPad | PadType::PaddedFullSpan | PadType::PaddedFullSpanWUnneeded => {
            stage.pad_stride(dim).unwrap_or(Stride::ONE)
        }
        PadType::NoPad | PadType::LoweredPadded | PadType::PaddedNoZeroPad => Stride::ONE,
    };

    // Corelet spatial fold
    let corelet_alpha = stage
        .first_corelet_share(dim)
        .map_or(Alpha(0), |share| Alpha(share.0.wrapping_mul(stride.get())));
    coord.add_fold(
        dim,
        CoordinateCategory::Spatial,
        Fold {
            cardinality: stage.corelets_used(),
            label: Some(FoldLabel::CoreletFoldDim),
            alpha: corelet_alpha,
            beta: Beta(0),
        },
    );

    // Core workslice fold. Assumption: each core is assigned the same number of elements.
    coord.add_fold(
        dim,
        CoordinateCategory::Spatial,
        Fold {
            cardinality: stage.work_slices(dim),
            label: Some(FoldLabel::CoreWorksliceFoldDim),
            alpha: Alpha(stage.core_extent(dim).0.wrapping_mul(stride.get())),
            beta: Beta(0),
        },
    );
}

// ⭐ TESTS FOR ENTRIES 086-093. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e086_e093 {
    use super::{
        AllocId, AllocLayout, Allocations, Alpha, Beta, BlockId, Cardinality, CoordPropInfo,
        Coordinate, CoordinateCategory, CoreStage, DataOrigin, DataStream, ElemArrFolds,
        ElemArrLevel, Fold, FoldLabel, FoldParamInfo, Incoming, IncomingStreams, LdsIdx,
        LoopLevels, NodeId, NodeKind, PadType, PropEnd, ScaledLds, ScheduleTree, SizeStage,
        StoredStream, Stride, build_spatial_fold, combine_contigous_levels,
        construct_alloc_elem_arr_layout, find_common_ancestor, is_allocate_incoming,
        match_data_stream, num_elements_in_pt_slice, order_descendants,
    };
    use crate::arch::{Dd2, Elements};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PrimaryDim, StickDims,
    };
    use crate::generated::DataConnect;
    use crate::units::DfirUnit;

    /// A LOOKUP TABLE FOR `getAllocation` — the reference's own map keyed on `(myLdsIdx_, storage)`
    /// (`dsc/dsc2.cpp:2605-2607`), plus the scale-to-value edge beside it.
    #[derive(Default)]
    struct Allocs {
        placed: Vec<(LdsIdx, DfirUnit, AllocId)>,
        values: Vec<(AllocId, AllocId)>,
    }

    impl Allocations for Allocs {
        fn allocation(&self, stored: StoredStream) -> Option<AllocId> {
            let DataOrigin::LabeledDs(lds) = stored.stream.origin else {
                return None;
            };
            self.placed
                .iter()
                .find(|(named, unit, _)| *named == lds && *unit == stored.storage)
                .map(|&(_, _, alloc)| alloc)
        }
        fn value_allocation(&self, scale: AllocId) -> Option<AllocId> {
            self.values
                .iter()
                .find(|(named, _)| *named == scale)
                .map(|&(_, value)| value)
        }
    }

    fn stream(lds: u32, storage: DfirUnit) -> StoredStream {
        StoredStream {
            stream: DataStream {
                origin: DataOrigin::LabeledDs(LdsIdx(lds)),
                data_connect: None,
            },
            storage,
        }
    }

    /// A TREE ANSWERED FROM A PARENT TABLE — `getPrev()` is the only edge entries 087 and 088 walk,
    /// and `getNextView(ALL)` is that table read the other way round in index order.
    struct Tree {
        kinds: Vec<NodeKind>,
        parents: Vec<Option<u32>>,
    }

    impl ScheduleTree for Tree {
        fn kind(&self, node: NodeId) -> NodeKind {
            self.kinds[node.0 as usize]
        }
        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.parents[node.0 as usize].map(NodeId)
        }
        fn children(&self, block: BlockId) -> Vec<NodeId> {
            (0..u32::try_from(self.parents.len()).unwrap_or(0))
                .map(NodeId)
                .filter(|&node| self.parent(node) == Some(block.node()))
                .collect()
        }
    }

    /// TWO LOOPS UNDER ONE BLOCK, TWO COMPUTES UNDER THE FIRST:
    ///   `0 Block { 1 Loop { 3 Compute, 4 Compute }, 2 Loop { 5 Compute } }`
    fn two_loops_under_a_block() -> Tree {
        Tree {
            kinds: vec![
                NodeKind::Block,
                NodeKind::Loop,
                NodeKind::Loop,
                NodeKind::Compute,
                NodeKind::Compute,
                NodeKind::Compute,
            ],
            parents: vec![None, Some(0), Some(0), Some(1), Some(1), Some(2)],
        }
    }

    /// The allocation's own padded span, tabulated — `calculate_padded` is `dsc/dims.cpp` and the
    /// mechanism for reaching it is not this unit's.
    struct Padded(i64);

    impl SizeStage for Padded {
        fn padded(&self, _dim: PrimaryDim, val: i64) -> Alpha {
            Alpha(val.wrapping_mul(self.0))
        }
    }

    /// Every `relatedElemArrLevel` the reference would have bumped, recorded rather than applied.
    #[derive(Default)]
    struct Bumps(Vec<(PrimaryDim, ElemArrLevel)>);

    impl LoopLevels for Bumps {
        fn bump_levels_at_or_inside(&mut self, dim: PrimaryDim, level: ElemArrLevel) {
            self.0.push((dim, level));
        }
    }

    /// The core data stage, tabulated.
    struct Core {
        pad_stride: Option<Stride>,
        first_corelet: Option<Extent>,
        core_extent: Extent,
        corelets: Cardinality,
        slices: Cardinality,
    }

    impl CoreStage for Core {
        fn pad_stride(&self, _dim: PrimaryDim) -> Option<Stride> {
            self.pad_stride
        }
        fn first_corelet_share(&self, _dim: PrimaryDim) -> Option<Extent> {
            self.first_corelet
        }
        fn core_extent(&self, _dim: PrimaryDim) -> Extent {
            self.core_extent
        }
        fn corelets_used(&self) -> Cardinality {
            self.corelets
        }
        fn work_slices(&self, _dim: PrimaryDim) -> Cardinality {
            self.slices
        }
    }

    /// Every `addFold` in the order it was made.
    #[derive(Default)]
    struct Coord(Vec<(PrimaryDim, CoordinateCategory, Fold)>);

    impl Coordinate for Coord {
        fn add_fold(&mut self, dim: PrimaryDim, category: CoordinateCategory, fold: Fold) {
            self.0.push((dim, category, fold));
        }
        fn clear_fold_for_dim(&mut self, dim: PrimaryDim) {
            self.0.retain(|(walked, _, _)| *walked != dim);
        }
    }

    #[test]
    fn a_computes_second_input_is_incoming_and_a_transfers_other_source_is_not() {
        let dsc = Allocs {
            placed: vec![
                (LdsIdx(7), DfirUnit::L0, AllocId(1)),
                (LdsIdx(9), DfirUnit::Lx, AllocId(2)),
            ],
            ..Allocs::default()
        };
        let inputs = [stream(4, DfirUnit::L0), stream(7, DfirUnit::L0)];
        let compute = Incoming::of(
            &[NodeId(3)],
            NodeId(3),
            Some(IncomingStreams::Compute(&inputs)),
        )
        .expect("node 3 is a user and a compute");
        assert!(is_allocate_incoming(&dsc, AllocId(1), &compute));
        // The transfer reads allocation 2, so allocation 1 is not on its incoming side.
        let transfer = Incoming::of(
            &[NodeId(5)],
            NodeId(5),
            Some(IncomingStreams::Transfer(stream(9, DfirUnit::Lx))),
        )
        .expect("node 5 is a user and a transfer");
        assert!(!is_allocate_incoming(&dsc, AllocId(1), &transfer));
        // ⛔ AND NEITHER REFUSAL IS CONSTRUCTIBLE: a non-user, and a kind with no incoming streams.
        assert!(
            Incoming::of(
                &[NodeId(3)],
                NodeId(5),
                Some(IncomingStreams::Compute(&inputs))
            )
            .is_none()
        );
        assert!(Incoming::of(&[NodeId(5)], NodeId(5), None).is_none());
    }

    #[test]
    fn the_common_ancestor_is_the_innermost_and_two_sibling_loops_have_none() {
        let tree = two_loops_under_a_block();
        // Both computes sit under loop 1, which is inside block 0 — the INNERMOST wins.
        assert_eq!(
            find_common_ancestor(
                &tree,
                &[NodeId(3), NodeId(4)],
                &[NodeKind::Block, NodeKind::Loop],
            ),
            Some(NodeId(1)),
        );
        // Nodes 3 and 5 share only the block, and no loop encloses both.
        assert_eq!(
            find_common_ancestor(&tree, &[NodeId(3), NodeId(5)], &[NodeKind::Loop]),
            None,
        );
        // ⛔ AND THE VOTE IS PER LIST ENTRY, NOT PER DISTINCT NODE: node 3 named twice votes twice,
        // so the count is reached and its own loop comes back.
        assert_eq!(
            find_common_ancestor(&tree, &[NodeId(3), NodeId(3)], &[NodeKind::Loop]),
            Some(NodeId(1)),
        );
    }

    #[test]
    fn ordering_yields_the_ancestors_own_children_once_each_in_schedule_order() {
        let tree = two_loops_under_a_block();
        let block = BlockId::of(&tree, NodeId(0)).expect("node 0 is a block");
        // Three nodes named out of order, under two children: loop 1 (twice) and loop 2.
        assert_eq!(
            order_descendants(&tree, block, &[NodeId(5), NodeId(3), NodeId(4)]),
            vec![NodeId(1), NodeId(2)],
        );
        // ⛔ A LOOP IS NOT A BLOCK, whichever way the reference's `isBlockNode` would answer.
        assert!(BlockId::of(&tree, NodeId(1)).is_none());
    }

    #[test]
    fn a_layout_splits_the_innermost_non_unit_level_into_capped_chunks() {
        let mut folds = vec![
            FoldParamInfo {
                alpha: Alpha(1000),
                beta: Beta(0),
                cardinality: Cardinality(5),
                label: None,
            },
            FoldParamInfo {
                alpha: Alpha(100),
                beta: Beta(0),
                cardinality: Cardinality(64),
                label: None,
            },
            FoldParamInfo {
                alpha: Alpha(1),
                beta: Beta(0),
                cardinality: Cardinality(1),
                label: None,
            },
        ];
        let mut elem_arr = ElemArrFolds(2);
        let mut bumps = Bumps::default();
        let layout = AllocLayout(vec![
            (PrimaryDim::Out, Some(Elements(16))),
            (PrimaryDim::Out, Some(Elements(2))),
        ]);
        assert_eq!(
            construct_alloc_elem_arr_layout(
                &Padded(3),
                &layout,
                PrimaryDim::Out,
                &mut folds,
                &mut elem_arr,
                &mut bumps,
            ),
            Some(ElemArrFolds(2)),
        );
        // 64 = 2 * 16 * 2: level 1 keeps 2 and its alpha is scaled by the 32 now inside it, and the
        // two inserted levels land INSIDE it, innermost first inserted.
        assert_eq!(folds[1].cardinality, Cardinality(2));
        assert_eq!(folds[1].alpha, Alpha(100 * 32));
        assert_eq!(folds[2].cardinality, Cardinality(2));
        assert_eq!(folds[2].alpha, Alpha(16 * 3));
        assert_eq!(folds[3].cardinality, Cardinality(16));
        assert_eq!(folds[3].alpha, Alpha(1 * 3));
        assert_eq!(folds[2].label, Some(FoldLabel::ElemArrLayoutSplit));
        assert_eq!(elem_arr, ElemArrFolds(4));
        assert_eq!(bumps.0, vec![(PrimaryDim::Out, ElemArrLevel(2)); 2]);
        // ⛔ AND A CAP THAT DOES NOT DIVIDE LEAVES THE LIST UNTOUCHED rather than half-split.
        let before = folds.clone();
        let indivisible = AllocLayout(vec![(PrimaryDim::Out, Some(Elements(3)))]);
        let mut inner = ElemArrFolds(4);
        assert_eq!(
            construct_alloc_elem_arr_layout(
                &Padded(3),
                &indivisible,
                PrimaryDim::Out,
                &mut folds,
                &mut inner,
                &mut Bumps::default(),
            ),
            None,
        );
        assert_eq!(folds, before);
    }

    #[test]
    fn contigous_levels_merge_and_unit_levels_vanish() {
        let level = |alpha: i64, beta: i64, card: u64| FoldParamInfo {
            alpha: Alpha(alpha),
            beta: Beta(beta),
            cardinality: Cardinality(card),
            label: None,
        };
        // Levels 1 and 2 compose (32 == 8 * 4); level 0 is a cardinality-1 level that goes.
        let mut folds = vec![level(999, 7, 1), level(32, 2, 3), level(8, 5, 4)];
        combine_contigous_levels(&mut folds, 0, None);
        assert_eq!(folds, vec![level(8, 7, 12)]);
        // ⛔ THE MERGED ALPHA IS THE INNER STRIDE — 8, not the comment's 32.
        assert_eq!(folds[0].alpha, Alpha(8));
        // ⚠️ AN `end` PAST THE LAST LEVEL IS CLAMPED to it, where the reference's `.at(end)` throws.
        let mut clamped = vec![level(32, 2, 3), level(8, 5, 4)];
        combine_contigous_levels(&mut clamped, 0, Some(99));
        assert_eq!(clamped, vec![level(8, 7, 12)]);
    }

    #[test]
    fn a_scale_tensor_matches_through_its_value_allocation_except_on_lxlu() {
        let dsc = Allocs {
            placed: vec![
                (LdsIdx(1), DfirUnit::L0, AllocId(10)),
                (LdsIdx(2), DfirUnit::L0, AllocId(20)),
            ],
            values: vec![(AllocId(20), AllocId(10))],
        };
        let scale = PropEnd::Allocate {
            alloc: AllocId(20),
            scaled: Some(ScaledLds::Scale),
        };
        let via_compute = CoordPropInfo {
            data_connect: None,
            ref_node: scale,
            node_to_fold: PropEnd::Compute {
                ex_unit: DfirUnit::Pe,
            },
        };
        // The PE consumes the VALUE tensor, allocation 10, so the value stream matches the scale end.
        assert!(match_data_stream(
            &dsc,
            &via_compute,
            stream(1, DfirUnit::L0),
            true
        ));
        assert!(!match_data_stream(
            &dsc,
            &via_compute,
            stream(2, DfirUnit::L0),
            true
        ));
        // ⛔ AN LXLU FMUL TOUCHES THE SCALE ALLOCATION DIRECTLY — no redirection.
        let via_lxlu = CoordPropInfo {
            node_to_fold: PropEnd::Compute {
                ex_unit: DfirUnit::Lxlu,
            },
            ..via_compute
        };
        assert!(match_data_stream(
            &dsc,
            &via_lxlu,
            stream(2, DfirUnit::L0),
            true
        ));
        // A named `data_connect=` short-circuits, and an end that is not an allocation never matches.
        let named = CoordPropInfo {
            data_connect: Some(DataConnect::PeHtOut),
            ref_node: PropEnd::Other,
            node_to_fold: PropEnd::Other,
        };
        assert!(!match_data_stream(
            &dsc,
            &named,
            stream(1, DfirUnit::L0),
            true
        ));
        let mut carried = stream(1, DfirUnit::L0);
        carried.stream.data_connect = Some(DataConnect::PeHtOut);
        assert!(match_data_stream(&dsc, &named, carried, true));
    }

    #[test]
    fn a_pt_slice_holds_the_within_slice_share_and_one_of_a_dim_the_stick_omits() {
        // 2 * 64 = 128 elements over Dd2's 8 PT rows: 16 to a slice, which is the whole `in` (2)
        // and 8 of the 64 `out`.
        let dims = StickDims(vec![
            (PrimaryDim::In, Elements(2)),
            (PrimaryDim::Out, Elements(64)),
        ]);
        assert_eq!(
            num_elements_in_pt_slice::<Dd2>(&dims, PrimaryDim::Out),
            Some(Elements(8)),
        );
        assert_eq!(
            num_elements_in_pt_slice::<Dd2>(&dims, PrimaryDim::In),
            Some(Elements(2)),
        );
        // A dim the stick does not name holds ONE element, not none.
        assert_eq!(
            num_elements_in_pt_slice::<Dd2>(&dims, PrimaryDim::Mb),
            Some(Elements(1)),
        );
    }

    #[test]
    fn the_corelet_fold_goes_in_first_and_broadcasts_when_no_corelet_splits_the_dim() {
        let mut coord = Coord::default();
        build_spatial_fold(
            &Core {
                pad_stride: Stride::new(4),
                first_corelet: Some(Extent(6)),
                core_extent: Extent(12),
                corelets: Cardinality(2),
                slices: Cardinality(3),
            },
            PrimaryDim::Out,
            PadType::PaddedWZeroPad,
            &mut coord,
        );
        assert_eq!(
            coord.0,
            vec![
                (
                    PrimaryDim::Out,
                    CoordinateCategory::Spatial,
                    Fold {
                        cardinality: Cardinality(2),
                        label: Some(FoldLabel::CoreletFoldDim),
                        alpha: Alpha(24),
                        beta: Beta(0),
                    },
                ),
                (
                    PrimaryDim::Out,
                    CoordinateCategory::Spatial,
                    Fold {
                        cardinality: Cardinality(3),
                        label: Some(FoldLabel::CoreWorksliceFoldDim),
                        alpha: Alpha(48),
                        beta: Beta(0),
                    },
                ),
            ],
        );
        // ⛔ NO CORELET SPLIT AND NO PADDING STRIDE STILL YIELDS THE CORELET FOLD, alpha 0.
        let mut unsplit = Coord::default();
        build_spatial_fold(
            &Core {
                pad_stride: Stride::new(4),
                first_corelet: None,
                core_extent: Extent(12),
                corelets: Cardinality(2),
                slices: Cardinality(3),
            },
            PrimaryDim::Out,
            PadType::NoPad,
            &mut unsplit,
        );
        assert_eq!(unsplit.0[0].2.alpha, Alpha(0));
        // NOPAD ignores the padding stride entirely.
        assert_eq!(unsplit.0[1].2.alpha, Alpha(12));
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The two units that produce a fold-params list.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// WHICH SLOT OF A FOLD-PARAMS LIST — `dsc2::CoordinateFoldPosition` (`dsc/dsc2.h:73`).
///
/// ⛔ THE FIRST THREE POSITIONS ARE ROLES, NOT ARRIVAL ORDER. ddc reads them back as
/// `FOLD_POS_CORE` = 0, `FOLD_POS_CORELET` = 1, `FOLD_POS_ROWSPLIT` = 2 (`ddc/ddc_fold.cpp:12-16`)
/// and assigns that slot directly — `foldParams.at(FOLD_POS_ROWSPLIT) = ..` (`:2589,3425`). With the
/// role in a type, a literal `2` is not what selects the rowsplit fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldPosition {
    /// `Core = 0`.
    Core,
    /// `Corelet = 1`.
    Corelet,
    /// `RowSplit = 2`.
    RowSplit,
}

impl FoldPosition {
    /// The slot's index in a [`gather_fold_params`] list.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            FoldPosition::Core => 0,
            FoldPosition::Corelet => 1,
            FoldPosition::RowSplit => 2,
        }
    }
}

/// `FoldManager<CoordinateBaseType>` (`util/foldManager/foldInfrastructure.h:898`) reduced to what
/// entry 095 reads out of it.
///
/// ⛔ AFFINE IN THE NAME BECAUSE THAT RETIRES A `DT_ERROR`. `getAlphaBeta` refuses any dimension
/// whose `BaseFuncType` is not `Affine` — *"Cannot query beta in non affine fold func"* (`:2431`) —
/// so a `Map`, `Constant` or `WkSplit` dimension aborts the compiler there. A holder whose dims are
/// not all affine cannot implement this trait; the classification is
/// [`crate::bridges::superdsc_to_dataflow_ir::driver::FoldDimFunc`].
///
/// ⚠️ THE FOLD MANAGER ITSELF IS OUT OF SCOPE (`util/foldManager/`, a recorded outside dependency of
/// 14 units), so the walk that reaches one belongs to the caller.
pub trait AffineFoldDims {
    /// `getNumDims()` (`:2270`).
    fn num_dims(&self) -> usize;

    /// `getAlphaBeta(alpha, beta, dim)` (`:2422`).
    ///
    /// ⛔ ONE α/β PAIR PER LEVEL, AND IT IS THE FIRST CHILD'S: the reference collects every fold
    /// function at the level and reads `*ffs_at_pos.begin()` (`:2437-2440`), so a level whose
    /// children disagree reports only the first — its answer, not an approximation of it.
    fn alpha_beta(&self, dim: usize) -> (Alpha, Beta);

    /// `getFoldDimSize(dim)` — `FoldDimProp::factor_` (`:126,2631`).
    fn dim_size(&self, dim: usize) -> Cardinality;

    /// `getFoldDimProp(dim)->Label()` (`:127,2624`); `None` is the empty label.
    fn dim_label(&self, dim: usize) -> Option<FoldLabel>;
}

/// Replaces: e094_getDefaultRowSplitFold
///
/// THE IDENTITY ROWSPLIT FOLD — one fold index, contributing no coordinate at all
/// (`ddc/ddc_fold.cpp:2154`).
///
/// ⛔ IT RETURNS A VALUE RATHER THAN TAKING `&mut`, because the reference overwrites all four fields
/// of its out-parameter and its three callers read nothing out of the old value: one hands it a
/// default-constructed local (`:2169`) and two hand it an existing slot (`:2593,3429`). Assigning
/// [`FoldPosition::RowSplit`]'s slot is the whole effect.
///
/// ⛔ `alpha = 0` IS THE NO-OP, and `FoldParamInfoType`'s own `alpha = 1` default is not it.
#[must_use]
pub const fn default_row_split_fold() -> FoldParamInfo {
    FoldParamInfo {
        alpha: Alpha(0),
        beta: Beta(0),
        cardinality: Cardinality(1),
        label: Some(FoldLabel::RowSplitFold),
    }
}

/// Replaces: e095_gatherFoldParams
///
/// A FOLD MANAGER'S DIMENSIONS AS A FOLD-PARAMS LIST, in ascending dimension order
/// (`ddc/ddc_fold.cpp:2224`).
///
/// ⛔ THE ORDER IS LOAD-BEARING, NOT INCIDENTAL. Callers index the result by role —
/// `foldParams.at(FOLD_POS_ROWSPLIT)` (`:3146`) — and re-add folds walking it BACKWARDS because
/// `addFold(.., pos = 0)` inserts at the front (`:3267-3279`). See [`FoldPosition`].
///
/// ⛔ IT RETURNS THE LIST INSTEAD OF APPENDING TO ONE. The reference `push_back`s into an
/// out-parameter, and all ten call sites pass a vector that is empty at that instant — eight freshly
/// declared (`:499,621,2531,2997,3142,3413`, `L3DlOpsScheduler.cpp:7396,7617`) and two `clear()`ed
/// on the two lines above (`ddc/ddc_fold.cpp:1345-1346`). The appending mode has no caller, so it
/// is not offered.
///
/// ⚠️ ITS L3 TWIN IS A SEPARATE UNIT. `L3DlOpsScheduler::gatherFoldParams`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7248`) is this function character for character and
/// is scheduled as `e061_gatherFoldParams`; when it lands it delegates here.
#[must_use]
pub fn gather_fold_params<F: AffineFoldDims + ?Sized>(fold_manager: &F) -> Vec<FoldParamInfo> {
    (0..fold_manager.num_dims())
        .map(|dim| {
            let (alpha, beta) = fold_manager.alpha_beta(dim);
            FoldParamInfo {
                alpha,
                beta,
                cardinality: fold_manager.dim_size(dim),
                label: fold_manager.dim_label(dim),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests_e094_e095 {
    use super::{
        AffineFoldDims, Alpha, Beta, Cardinality, FoldLabel, FoldParamInfo, FoldPosition,
        default_row_split_fold, gather_fold_params,
    };

    /// A fold manager's dimensions, outermost first, as its four getters report them.
    struct Dims(Vec<(Alpha, Beta, Cardinality, Option<FoldLabel>)>);

    impl AffineFoldDims for Dims {
        fn num_dims(&self) -> usize {
            self.0.len()
        }
        fn alpha_beta(&self, dim: usize) -> (Alpha, Beta) {
            (self.0[dim].0, self.0[dim].1)
        }
        fn dim_size(&self, dim: usize) -> Cardinality {
            self.0[dim].2
        }
        fn dim_label(&self, dim: usize) -> Option<FoldLabel> {
            self.0[dim].3
        }
    }

    /// e094: the identity fold the reference's own comment claims (`ddc/ddc_fold.cpp:2164-2165`) —
    /// one fold index, no coordinate, which `FoldParamInfoType`'s `alpha = 1` default is not.
    #[test]
    fn the_default_rowsplit_fold_is_the_identity_fold() {
        assert_eq!(
            default_row_split_fold(),
            FoldParamInfo {
                alpha: Alpha(0),
                beta: Beta(0),
                cardinality: Cardinality(1),
                label: Some(FoldLabel::RowSplitFold),
            }
        );
    }

    /// e095: the list comes back in ascending dimension order, so the roles ddc reads it back by
    /// (`FOLD_POS_CORE`/`CORELET`/`ROWSPLIT`, `ddc/ddc_fold.cpp:12-16`) land in those slots.
    #[test]
    fn gathered_params_keep_the_fold_managers_dimension_order() {
        let fold_manager = Dims(vec![
            (
                Alpha(64),
                Beta(0),
                Cardinality(32),
                Some(FoldLabel::CoreWorksliceFoldDim),
            ),
            (
                Alpha(-8),
                Beta(4),
                Cardinality(2),
                Some(FoldLabel::CoreletFoldDim),
            ),
            (
                Alpha(0),
                Beta(0),
                Cardinality(1),
                Some(FoldLabel::RowSplitFold),
            ),
        ]);

        let params = gather_fold_params(&fold_manager);

        assert_eq!(params.len(), 3);
        assert_eq!(
            params[FoldPosition::Core.index()].cardinality,
            Cardinality(32)
        );
        // A negative alpha survives as a step rather than being taken for a size.
        assert_eq!(params[FoldPosition::Corelet.index()].alpha, Alpha(-8));
        assert_eq!(params[FoldPosition::Corelet.index()].beta, Beta(4));
        // The outermost slot ddc assigns by role is the one e094 mints.
        assert_eq!(
            params[FoldPosition::RowSplit.index()],
            default_row_split_fold()
        );
    }
}

/// HOW MANY OF A DIM'S FOLDS ARE SPATIAL — `getNumOfSpatialFolds(dim)` (`dsc/dsc2.h:144`), which is
/// TOTAL: a dim the coordinate does not cover answers 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpatialFolds(pub u32);

/// HOW MANY OF A DIM'S FOLDS ARE TEMPORAL — `getNumOfTemporalFolds(dim)` (`:147`), likewise total. A
/// type of its own, so the two counts cannot be transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemporalFolds(pub u32);

/// ONE DIM OF A COORDINATE — its fold manager, the two fold counts and `getPadding(dim)` (`:242`).
///
/// ⛔ AN UNCOVERED DIM IS AN ABSENT FOLD MANAGER, NOT A THROW: `coordinates_.at(dim)` aborts where
/// the two counts answer 0, and [`gather_fold_params`] over nothing is the empty list.
pub struct CoordDim<'a, F: AffineFoldDims + ?Sized> {
    /// `coordinates_.at(dim)`.
    pub folds: Option<&'a F>,
    /// `getNumOfSpatialFolds(dim)`.
    pub spatial: SpatialFolds,
    /// `getNumOfTemporalFolds(dim)`.
    pub temporal: TemporalFolds,
    /// `getPadding(dim)`.
    pub pad: PadType,
}

/// AN ALLOCATE NODE'S TWO COORDINATES FOR ONE DIM — `allocateCoordinates_` and
/// `sliceViewCoordinates_` (`dsc/dsc2.h:1008-1009`), the second absent where `foldConstructed()`
/// (`:119`) is false.
pub struct AllocCoordinates<'a, F: AffineFoldDims + ?Sized> {
    /// `allocateCoordinates_`.
    pub allocate: CoordDim<'a, F>,
    /// `sliceViewCoordinates_`, once its folds are constructed.
    pub slice_view: Option<CoordDim<'a, F>>,
}

impl<'a, F: AffineFoldDims + ?Sized> AllocCoordinates<'a, F> {
    /// `effectiveRefCoord` (`ddc/ddc_fold.cpp:2988-2991`) — the slice view only when the size
    /// component is an individual PT row AND the slice view's folds were constructed.
    #[must_use]
    pub fn effective(&self, row: Option<PtRowId>) -> &CoordDim<'a, F> {
        match (row, &self.slice_view) {
            (Some(_), Some(slice_view)) => slice_view,
            _ => &self.allocate,
        }
    }
}

/// THE PROPAGATION'S ALLOCATE END — `coordPropInfo.nodeToFold` as an `AllocateNode`, with the
/// labeled DS's scale for the dim being related already in hand.
///
/// ⛔ BOTH `DT_CHECK_MSG`s ARE DISCHARGED BY TYPE: that the node IS an `ALLOCATE` (`:2883`), and that
/// its `ldsIdx_ >= 0` (`:2910`).
pub struct BaseAllocation<'a, F: AffineFoldDims + ?Sized> {
    /// `component_`.
    pub component: SenComponent,
    /// `ldsIdx_`.
    pub lds: LdsIdx,
    /// `allocateCoordinates_` / `sliceViewCoordinates_`, for the dim being related.
    pub coordinates: AllocCoordinates<'a, F>,
    /// `labeledDs_.at(ldsIdx_).scale_.at(getDimIndexInLayoutOrder(dsType_, dim))`, absent where the
    /// layout order does not name the dim — which the reference reads as a scale of 1 (`:3002`).
    pub dim_scale: Option<Scale>,
}

/// ONE TRANSFER DESTINATION — `dstVias_.at(i).loc_` zipped with `dstLdsAndLoopOffsets_.at(i)`.
///
/// ⚠️ NOT [`crate::schedule::dsc2::Operand`], whose `storage` is a [`SenComponent`] and whose data is
/// a [`crate::schedule::dsc2::DataInfo`]: [`match_data_stream`] reads this file's spellings, and
/// converging the two is the review pass's, as `dsc2.rs` already notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferDst {
    /// `loc_.unit_`.
    pub unit: SenComponent,
    /// The destination stream, in `loc_.storage_`.
    pub stream: StoredStream,
}

/// A TRANSFER'S DESTINATIONS, NON-EMPTY — so neither `dstVias_.at(0)` override (`:2960`, `:2977`)
/// can throw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferDsts {
    first: TransferDst,
    rest: Vec<TransferDst>,
}

impl TransferDsts {
    /// A transfer has at least one destination, and this is how that is stated.
    #[must_use]
    pub const fn new(first: TransferDst, rest: Vec<TransferDst>) -> Self {
        Self { first, rest }
    }

    /// `dstVias_.at(0)` — total.
    #[must_use]
    pub const fn first(&self) -> &TransferDst {
        &self.first
    }

    /// Every destination in order.
    pub fn iter(&self) -> impl Iterator<Item = &TransferDst> {
        core::iter::once(&self.first).chain(self.rest.iter())
    }
}

/// WHICH WAY A TRANSFER MOVES RELATIVE TO THE ALLOCATION — `coordPropInfo.refIsProducer`
/// (`dsc/dsc2.h:1091`), which is a direction and so an enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// `refIsProducer` — transfer -> allocate, so the MATCHING destination carries the components.
    IntoAllocation,
    /// allocate -> transfer, so `src_.unit_` does.
    OutOfAllocation,
}

/// THE PROPAGATION'S NON-ALLOC END — `coordPropInfo.refNode`'s `nodeType_` reduced to the three arms
/// entry 241 distinguishes.
pub enum RefNode<'a> {
    /// `COMPUTE`.
    Compute {
        /// `exUnit_`.
        ex_unit: SenComponent,
    },
    /// `TRANSFER`.
    Transfer {
        /// `src_.unit_`.
        src: SenComponent,
        /// `dstVias_` zipped with `dstLdsAndLoopOffsets_`.
        dsts: &'a TransferDsts,
        /// `refIsProducer`.
        direction: TransferDirection,
    },
    /// Any other `nodeType_`: neither arm runs, and both components stay as the allocation set them.
    Other,
}

/// THE COMPONENT PAIR A PROPAGATION IS SIZED AND PROPAGATED FOR — `sizeRefComp` and `propRefComp`.
///
/// ⛔ THEY MOVE INDEPENDENTLY: both transfer overrides (`:2952-2963`, `:2966-2981`) move the SIZE
/// alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefComponents {
    /// `sizeRefComp`, whose start is `SenComponents::ALL`.
    pub size: SenComponent,
    /// `propRefComp`, whose start is the allocation's own `component_`.
    pub prop: SenComponent,
}

/// ONE COORDINATE PROPAGATION OFF AN ALLOCATION'S ELEMENT ARRANGEMENT — `coordPropInfo` with both of
/// its ends resolved.
pub struct AllocPropagation<'a, F: AffineFoldDims + ?Sized> {
    /// `nodeToFold`.
    pub alloc: BaseAllocation<'a, F>,
    /// `refNode` — the node the distributor records the loop params against.
    pub ref_node: NodeId,
    /// `refNode`'s kind.
    pub ref_kind: RefNode<'a>,
    /// The propagation as [`match_data_stream`] reads it.
    pub streams: &'a CoordPropInfo,
}

/// `sizeRefComp` and `propRefComp` as entry 241 derives them (`ddc/ddc_fold.cpp:2914-2981`).
fn ref_components<A: Allocations + ?Sized, F: AffineFoldDims + ?Sized>(
    dsc: &A,
    prop: &AllocPropagation<'_, F>,
) -> RefComponents {
    let is_pe_or_sfp = |comp| matches!(comp, SenComponent::Pe | SenComponent::Sfp);
    let mut comps = RefComponents {
        size: SenComponent::All,
        prop: prop.alloc.component,
    };
    match prop.ref_kind {
        RefNode::Compute { ex_unit } => {
            comps.size = ex_unit;
            if comps.prop == SenComponent::Ptxrf || comps.prop == SenComponent::Ptarf {
                // *** This is a kludge. FIND A GENERAL SOLUTION ***
                // PTXRF and PTARF represent the allocation for all PT rows; a computeNode on the PT
                // represents computation on an individual PT row.
                comps.prop = ex_unit;
            }
        }
        RefNode::Transfer {
            src,
            dsts,
            direction,
        } => {
            match direction {
                // Transfer -> allocate.
                TransferDirection::IntoAllocation => {
                    if let Some(dst) = dsts
                        .iter()
                        .find(|dst| match_data_stream(dsc, prop.streams, dst.stream, false))
                    {
                        comps.size = dst.unit;
                        comps.prop = dst.unit;
                    }
                }
                // Allocate -> transfer.
                TransferDirection::OutOfAllocation => {
                    comps.size = src;
                    comps.prop = src;
                }
            }
            // Override the comp in case transfer is for a single PT row: for a transfer from LXLU to
            // PTRow4 the coordinate should be for row 4 only.
            if comp_row_id(comps.size).is_none() {
                if comp_row_id(src).is_some() {
                    comps.size = src;
                } else if comp_row_id(dsts.first().unit).is_some() {
                    comps.size = dsts.first().unit;
                }
            }
            // Override the comp in case transfer involves PE or SFP, to account for PE-SFP
            // splitting. A component that already represents an individual PT row has higher
            // priority.
            if comp_row_id(comps.size).is_none() && !is_pe_or_sfp(comps.size) {
                if is_pe_or_sfp(src) {
                    comps.size = src;
                } else if is_pe_or_sfp(dsts.first().unit) {
                    comps.size = dsts.first().unit;
                }
            }
        }
        RefNode::Other => {}
    }
    comps
}

/// THE DISTRIBUTOR'S REQUEST — `distributeElemArrToTemporalLoops`' arguments (`dsc/dsc2.h:1174`)
/// other than the DSC and its two out-params.
pub struct ElemArrDistribution<'l> {
    /// `currDim`.
    pub dim: PrimaryDimAndKind,
    /// `foldOwnerNode` — the REF node, not the allocation.
    pub fold_owner: NodeId,
    /// `targetLdxIdx` — the ALLOCATION's labeled DS.
    pub target_lds: LdsIdx,
    /// `refPadType` — the effective coordinate's padding for the dim.
    pub ref_pad: PadType,
    /// `targetPadType` — the propagated coordinate's padding for the dim.
    pub target_pad: PadType,
    /// `sizeRefComp` and `propRefComp`.
    pub components: RefComponents,
    /// `loopChain` — the surplus loops, INNERMOST FIRST.
    pub loops_to_distribute: Vec<LoopAndDim<'l>>,
    /// `elemArr` — the fold levels inside the reference's own folds, INNERMOST FIRST.
    pub elem_arr: Vec<FoldParamInfo>,
}

/// THE TWO `dsc/dsc2.cpp` HELPERS ENTRY 241 REACHES OUT TO — both OUTSIDE this campaign's file list
/// (`crustify-ddc/OUTSIDE-DEPS.tsv`, "dsc2 tree utilities"), so they are seams here and not ports.
pub trait TemporalLoopDistribution {
    /// `dsc2::LoopDistributionParamPerNodeType` — the out-param entry 241 threads through, whose
    /// contents are the distributor's own business.
    type LoopParams;

    /// `collectRelatedLoops(currDsc, dimToFind, allEnclosingLoops, relatedLoops, accessPadType)`
    /// (`dsc/dsc2.cpp:6575`) — the enclosing chain filtered to the loops that walk this dim, and
    /// INNERMOST FIRST because the chain is.
    fn related_loops<'l>(
        &self,
        dim: PrimaryDimAndKind,
        chain: &[LoopAndDim<'l>],
        pad: PadType,
    ) -> Vec<LoopAndDim<'l>>;

    /// `distributeElemArrToTemporalLoops(.., targetCoreletId = 0, ..)` (`dsc/dsc2.cpp:5934`) — for
    /// ONE corelet, as entry 241's own argument comment says (`ddc/ddc_fold.cpp:3038`).
    ///
    /// ⚠️ ENTRY 229 PASSES `-1` *"for both corelets"* (`L3DlOpsScheduler.cpp:7676`) AND IT IS INERT
    /// THERE: `targetCoreletId` is read on the `BELOW_CHUNK` arm alone (`dsc/dsc2.cpp:6054`) and
    /// every loop that unit hands over is tagged `ABOVE_CHUNK`, so the seam takes no such argument.
    ///
    /// ⛔ THE REQUEST'S LIFETIME IS THE CALL'S, not the implementor's: entry 229 distributes over a
    /// loop it MINTS ON THE SPOT, which no `'l` outliving the seam could name.
    ///
    /// ⛔ IT RETURNS `elemArrParamsAfterDistribution`, the redistributed element-arrangement levels
    /// OUTERMOST FIRST. Entry 241 drops them (`:3029`); entries 234 and 240 append them to a fold
    /// list, so the out-param cannot be dropped from the seam.
    fn distribute(
        &self,
        request: &ElemArrDistribution<'_>,
        loop_params: &mut Self::LoopParams,
    ) -> Vec<FoldParamInfo>;

    /// `loopParamsAfterDistribution.at(loopNode).at(dim)` (`ddc/ddc_fold.cpp:3348-3351`, `:4172`) —
    /// what the distributor decided for one loop and one dim, absent where it recorded nothing.
    /// Both reference reads are `.at()` on a map the distributor may not have filled.
    fn distributed(
        &self,
        loop_params: &Self::LoopParams,
        loop_node: &LoopNode,
        dim: PrimaryDim,
    ) -> Option<DistributedLoop>;
}

/// WHAT THE DISTRIBUTOR DECIDED FOR ONE TEMPORAL LOOP — `LoopDistributionParamType`'s `alpha` and
/// `beta` (`dsc/dsc2.h:1109`), the only two of its four fields any caller reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedLoop {
    /// `alpha`, whose own default is 1.
    pub alpha: Alpha,
    /// `beta`, whose own default is 0.
    pub beta: Beta,
}

/// Replaces: e241_relateLoopsToAllocElemArr
///
/// Hands the loops around the REF node that the allocation's own temporal folds do NOT account for,
/// together with the fold levels inside them, to the temporal distributor — for one dim.
///
/// ⛔ THE `coordinate` ARGUMENT IS READ ONLY FOR ITS PADDING OF THIS DIM (`:3005`, `:3037`).
/// ⚠️ `allocPosInDest` is computed and never read (`:2932`, `:2941`), and the report-level prints
/// (`:2891-2908`) are diagnostics; neither survives.
pub fn relate_loops_to_alloc_elem_arr<'l, A, T, F>(
    dsc: &A,
    distribution: &T,
    prop: &AllocPropagation<'_, F>,
    dim: PrimaryDim,
    target_pad: PadType,
    ref_loop_chain: &[LoopAndDim<'l>],
    loop_params: &mut T::LoopParams,
) where
    A: Allocations + ?Sized,
    T: TemporalLoopDistribution + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    let components = ref_components(dsc, prop);
    let effective = prop
        .alloc
        .coordinates
        .effective(comp_row_id(components.size));
    let fold_params = effective.folds.map(gather_fold_params).unwrap_or_default();

    let scale_is_non_broadcast = |scale| matches!(scale, Scale::Sized(scale) if scale > 0.0);
    // Note: dim is of type PrimaryDimTypes, metaDimKind is missing here.
    let dim = PrimaryDimAndKind {
        dim,
        kind: MetaDimKind::Unpadded,
    };
    let related_loops = if prop.alloc.dim_scale.is_none_or(scale_is_non_broadcast) {
        // The dimension is a non-broadcast dimension.
        distribution.related_loops(dim, ref_loop_chain, target_pad)
    } else {
        Vec::new()
    };

    // The node has more enclosing loops than the reference allocateNode: distribute the additional
    // loops over the element arrangement levels.
    let temporal_diff = related_loops
        .len()
        .saturating_sub(effective.temporal.0 as usize);
    if temporal_diff == 0 {
        return;
    }
    // `temporalFoldEnds + 1` is `spatialFoldEnds + refTemporalCount + 1`, so the levels inside the
    // reference's own folds start at spatial + temporal — with no negative index to clamp.
    let inside = (effective.spatial.0 + effective.temporal.0) as usize;
    let _dropped = distribution.distribute(
        &ElemArrDistribution {
            dim,
            fold_owner: prop.ref_node,
            target_lds: prop.alloc.lds,
            ref_pad: effective.pad,
            target_pad,
            components,
            // Scan order is from the innermost loop towards the outermost loop.
            loops_to_distribute: related_loops[..temporal_diff].to_vec(),
            elem_arr: fold_params
                .get(inside..)
                .unwrap_or_default()
                .iter()
                .rev()
                .copied()
                .collect(),
        },
        loop_params,
    );
}

#[cfg(test)]
mod tests_e241 {
    use super::{
        AffineFoldDims, AllocCoordinates, AllocId, AllocPropagation, Allocations, Alpha,
        BaseAllocation, Beta, Cardinality, CoordDim, CoordPropInfo, DataOrigin, DataStream,
        ElemArrDistribution, FoldLabel, FoldParamInfo, LdsIdx, NodeId, PadType, PropEnd,
        RefComponents, RefNode, SpatialFolds, StoredStream, TemporalFolds,
        TemporalLoopDistribution, TransferDirection, TransferDst, TransferDsts,
        relate_loops_to_alloc_elem_arr,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
    use crate::generated::DataConnect;
    use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind};
    use crate::schedule::ddc::transformation::Scale;
    use crate::schedule::ddc::transformation_util::{LoopDims, LoopNode, PrimaryDimAndKind};
    use crate::schedule::dsc2::NodeName;
    use crate::schedule::l3::dl_ops::{LoopAndDim, LoopDistribution};
    use crate::units::DfirUnit;
    use sys_arch_spec::arch_enums::SenComponent;

    /// A fold manager's dims by trip count, OUTERMOST FIRST.
    struct Dims(Vec<u64>);

    impl AffineFoldDims for Dims {
        fn num_dims(&self) -> usize {
            self.0.len()
        }
        fn alpha_beta(&self, _dim: usize) -> (Alpha, Beta) {
            (Alpha(1), Beta(0))
        }
        fn dim_size(&self, dim: usize) -> Cardinality {
            Cardinality(self.0[dim])
        }
        fn dim_label(&self, _dim: usize) -> Option<FoldLabel> {
            None
        }
    }

    /// The propagation names a `data_connect=`, so [`super::match_data_stream`] short-circuits and
    /// no allocation is ever looked up.
    struct NoAllocs;

    impl Allocations for NoAllocs {
        fn allocation(&self, _stored: StoredStream) -> Option<AllocId> {
            None
        }
        fn value_allocation(&self, _scale: AllocId) -> Option<AllocId> {
            None
        }
    }

    /// What the distributor was asked to do.
    #[derive(Debug, PartialEq)]
    struct Recorded {
        dim: PrimaryDimAndKind,
        fold_owner: NodeId,
        target_lds: LdsIdx,
        ref_pad: PadType,
        target_pad: PadType,
        components: RefComponents,
        loops: Vec<String>,
        elem_arr: Vec<Cardinality>,
    }

    /// The two dsc2 seams: `collectRelatedLoops` keeps the chain entries that walk the dim, and the
    /// distributor records its request in the out-param.
    struct Seams;

    impl TemporalLoopDistribution for Seams {
        type LoopParams = Option<Recorded>;

        fn related_loops<'l>(
            &self,
            dim: PrimaryDimAndKind,
            chain: &[LoopAndDim<'l>],
            _pad: PadType,
        ) -> Vec<LoopAndDim<'l>> {
            chain
                .iter()
                .copied()
                .filter(|entry| entry.dim == dim)
                .collect()
        }

        fn distribute(
            &self,
            request: &ElemArrDistribution<'_>,
            loop_params: &mut Self::LoopParams,
        ) -> Vec<FoldParamInfo> {
            *loop_params = Some(Recorded {
                dim: request.dim,
                fold_owner: request.fold_owner,
                target_lds: request.target_lds,
                ref_pad: request.ref_pad,
                target_pad: request.target_pad,
                components: request.components,
                loops: request
                    .loops_to_distribute
                    .iter()
                    .map(|entry| entry.loop_node.name.0.clone())
                    .collect(),
                elem_arr: request
                    .elem_arr
                    .iter()
                    .map(|fold| fold.cardinality)
                    .collect(),
            });
            Vec::new()
        }

        fn distributed(
            &self,
            _loop_params: &Self::LoopParams,
            _loop_node: &LoopNode,
            _dim: PrimaryDim,
        ) -> Option<super::DistributedLoop> {
            None
        }
    }

    fn loop_named(name: &str, dim: PrimaryDim) -> LoopNode {
        LoopNode {
            name: NodeName(name.to_owned()),
            num: DatastageId(0),
            den: DatastageId(1),
            dims: LoopDims::new(
                PrimaryDimAndKind {
                    dim,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        }
    }

    /// THE REFERENCE'S OWN EXAMPLE — `transfer_lds4_src:ptrow6_dst:pe` in resnet's `Conv_0`
    /// (`ddc/ddc_fold.cpp:2969-2971`): the matched destination sets both components to `PE`, then the
    /// PT-row override moves the SIZE to row 6 alone and leaves the propagation on `PE`. Row 6 then
    /// selects the SLICE-VIEW coordinate, whose two counts decide both slices.
    #[test]
    fn a_transfer_off_a_pt_row_sizes_for_the_row_and_hands_over_the_surplus_loops() {
        let allocate_folds = Dims(vec![7, 7]);
        let slice_view_folds = Dims(vec![2, 3, 4, 5]);
        let dsts = TransferDsts::new(
            TransferDst {
                unit: SenComponent::Pe,
                stream: StoredStream {
                    stream: DataStream {
                        origin: DataOrigin::LabeledDs(LdsIdx(4)),
                        data_connect: Some(DataConnect::PeHtOut),
                    },
                    storage: DfirUnit::L0,
                },
            },
            Vec::new(),
        );
        let streams = CoordPropInfo {
            data_connect: Some(DataConnect::PeHtOut),
            ref_node: PropEnd::Other,
            node_to_fold: PropEnd::Other,
        };
        let prop = AllocPropagation {
            alloc: BaseAllocation {
                component: SenComponent::Ptxrf,
                lds: LdsIdx(4),
                coordinates: AllocCoordinates {
                    allocate: CoordDim {
                        folds: Some(&allocate_folds),
                        spatial: SpatialFolds(0),
                        temporal: TemporalFolds(0),
                        pad: PadType::NoPad,
                    },
                    slice_view: Some(CoordDim {
                        folds: Some(&slice_view_folds),
                        spatial: SpatialFolds(1),
                        temporal: TemporalFolds(1),
                        pad: PadType::PaddedWZeroPad,
                    }),
                },
                dim_scale: Some(Scale::Sized(1.0)),
            },
            ref_node: NodeId(9),
            ref_kind: RefNode::Transfer {
                src: SenComponent::Ptrow6,
                dsts: &dsts,
                direction: TransferDirection::IntoAllocation,
            },
            streams: &streams,
        };
        let (inner, middle, outer, other) = (
            loop_named("inner", PrimaryDim::In),
            loop_named("middle", PrimaryDim::In),
            loop_named("outer", PrimaryDim::In),
            loop_named("other", PrimaryDim::Out),
        );
        let chain: Vec<LoopAndDim<'_>> = [&inner, &middle, &outer, &other]
            .into_iter()
            .map(|loop_node| LoopAndDim {
                loop_node,
                dim: PrimaryDimAndKind {
                    dim: loop_node.dims.iter().next().expect("one dim").dim,
                    kind: MetaDimKind::Unpadded,
                },
                distribution: LoopDistribution::AboveChunk,
            })
            .collect();

        let mut loop_params = None;
        relate_loops_to_alloc_elem_arr(
            &NoAllocs,
            &Seams,
            &prop,
            PrimaryDim::In,
            PadType::LoweredPadded,
            &chain,
            &mut loop_params,
        );

        assert_eq!(
            loop_params,
            Some(Recorded {
                dim: PrimaryDimAndKind {
                    dim: PrimaryDim::In,
                    kind: MetaDimKind::Unpadded,
                },
                fold_owner: NodeId(9),
                target_lds: LdsIdx(4),
                // The REFERENCE's padding is the effective coordinate's, the TARGET's is the
                // propagated coordinate's, and they are not interchangeable.
                ref_pad: PadType::PaddedWZeroPad,
                target_pad: PadType::LoweredPadded,
                components: RefComponents {
                    size: SenComponent::Ptrow6,
                    prop: SenComponent::Pe,
                },
                // Three loops walk `In`, one temporal fold covers one of them, and the two that are
                // left are handed over innermost first.
                loops: vec!["inner".to_owned(), "middle".to_owned()],
                // Spatial 1 + temporal 1 of the slice view's four levels are the reference's own, so
                // the last two are the element arrangement — innermost first.
                elem_arr: vec![Cardinality(5), Cardinality(4)],
            })
        );
    }
}

/// Replaces: e233_dbgPrint
///
/// A schedule node's debug line, dispatched on its kind.
///
/// ⛔ TRAP CLOSED: `"Unsupported node type."` is unspellable over [`Node`], whose three variants are
/// exactly the three arms the reference supports.
#[must_use]
pub fn dbg_print(node: Node<'_>) -> String {
    match node {
        Node::Allocate(alloc) => format!(" AllocateNode: {}({alloc:p})", alloc.name.0),
        Node::Compute(compute) => dbg_print_compute(compute),
        Node::Transfer(transfer) => dbg_print_transfer(transfer),
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE MX SCALE BLOCK, AND THE TAIL THE TWO SCALING UNITS SHARE — as entries 234 and 235 read them.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// HOW MANY ELEMENTS ONE MX SCALE COVERS — `LabeledDsInfo::mxInfo_.blkSize` (`dsc/dscdefn.h`).
///
/// ⛔⛔ NON-ZERO BY CONSTRUCTION, AND THAT IS WHAT MAKES ENTRY 235 TOTAL: its own arithmetic divides
/// by the block size twice (`accumulatedCardinality / scaleBlkSize` and the `1.0/scaleBlkSize`
/// factor, `ddc/ddc_fold.cpp:601`, `:645`), and a block of zero elements is not a scale block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScaleBlock(u64);

impl ScaleBlock {
    /// The block, or [`None`] for a size of zero.
    #[must_use]
    pub const fn of(elems: Cardinality) -> Option<Self> {
        if elems.0 == 0 {
            None
        } else {
            Some(Self(elems.0))
        }
    }

    /// How many elements the block covers.
    #[must_use]
    pub const fn count(self) -> Cardinality {
        Cardinality(self.0)
    }
}

/// AN MX **VALUE** TENSOR — the labeled DS entry 234 scales up for, with its `mxInfo_` resolved.
///
/// ⛔⛔ BOTH OF ENTRY 234'S `DT_ERROR`s ARE THIS TYPE: `ldsIdx == -1` (`:479`) is unspellable over
/// [`LdsIdx`], and *"Invalid labeledDs category"* (`:485-488`) is what [`Self::of`] refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxValueTensor {
    /// `ldsIdx`.
    pub lds: LdsIdx,
    /// `mxInfo_.dim`.
    pub dim: PrimaryDim,
    /// `mxInfo_.blkSize`.
    pub blk_size: ScaleBlock,
}

impl MxValueTensor {
    /// The value tensor, or [`None`] for any other `scaledLdsCategory_`.
    #[must_use]
    pub const fn of(
        category: ScaledLds,
        lds: LdsIdx,
        dim: PrimaryDim,
        blk_size: ScaleBlock,
    ) -> Option<Self> {
        match category {
            ScaledLds::Value => Some(Self { lds, dim, blk_size }),
            ScaledLds::Regular | ScaledLds::Scale => None,
        }
    }
}

/// AN MX **SCALE** TENSOR — the labeled DS entry 235 compresses for.
///
/// ⛔ ITS `ldsIdx` IS NOT A FIELD: entry 235 reads the labeled DS for its `mxInfo_` alone (`:605`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxScaleTensor {
    /// `mxInfo_.dim`.
    pub dim: PrimaryDim,
    /// `mxInfo_.blkSize`.
    pub blk_size: ScaleBlock,
}

impl MxScaleTensor {
    /// The scale tensor, or [`None`] for any other `scaledLdsCategory_`.
    #[must_use]
    pub const fn of(category: ScaledLds, dim: PrimaryDim, blk_size: ScaleBlock) -> Option<Self> {
        match category {
            ScaledLds::Scale => Some(Self { dim, blk_size }),
            ScaledLds::Regular | ScaledLds::Value => None,
        }
    }
}

/// WHETHER ENTRY 235 COMPRESSED THE COORDINATE — its refusal and its nothing-to-do arm as ANSWERS
/// rather than an abort and a silent return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDown {
    /// The fold list was rebuilt on the target coordinate.
    Applied,
    /// `!rhsCoord.hasCoordForDim(dim)` (`:606`) — the reference returns having done nothing.
    NoCoordForDim,
    /// *"Non-contiguous element arrangment within a scale block is not supported"* (`:625-627`),
    /// which also covers the `foldParams.at(i - 1)` at `i == 0` that throws before it can say so.
    NonContiguousInScaleBlock,
}

/// The tail entries 234 and 235 share (`:553-569`, `:653-669`): clear the dim, then re-add every
/// level from the INNERMOST outwards, because `addFold(.., pos = 0)` inserts at the front.
///
/// ⛔ THE CATEGORY IS DECIDED BY POSITION AGAINST THE **REFERENCE** COORDINATE'S TWO COUNTS, not by
/// what the rebuilt list looks like — a level that moved inside the temporal end becomes temporal.
fn re_add_rebuilt_folds<C: Coordinate + ?Sized>(
    coord: &mut C,
    dim: PrimaryDim,
    fold_params: &[FoldParamInfo],
    spatial_ends: i64,
    temporal_ends: i64,
) {
    coord.clear_fold_for_dim(dim);
    for (at, level) in fold_params.iter().enumerate().rev() {
        let at = i64::try_from(at).unwrap_or(i64::MAX);
        let category = if at > temporal_ends {
            CoordinateCategory::ElemArr
        } else if at > spatial_ends {
            CoordinateCategory::Temporal
        } else {
            CoordinateCategory::Spatial
        };
        coord.add_fold(
            dim,
            category,
            Fold {
                cardinality: level.cardinality,
                label: level.label,
                alpha: level.alpha,
                beta: level.beta,
            },
        );
    }
}

/// Replaces: e234_scaleUpCoord
///
/// Rewrites the target's fold list for an MX value tensor's dim: every level scaled by the scale-block
/// size, a new element arrangement for the block, the inner temporal levels redistributed over it.
///
/// ⛔ DELIBERATE DIVERGENCE: the reference's `for (auto loop : loopChain)` never reads `loop`, so it
/// appends |loopChain| duplicates of ONE `collectRelatedLoops` answer (`:516-520`); this calls it
/// once and clamps the take, which is the same list whenever the take fits. `relatedDims` is dead.
pub fn scale_up_coord<'l, T, F, C>(
    value: &MxValueTensor,
    rhs: &CoordDim<'_, F>,
    lhs: &mut C,
    fold_owner: NodeId,
    comp: SenComponent,
    distribution: &T,
    loop_chain: &[LoopAndDim<'l>],
    loop_params: &mut T::LoopParams,
) where
    T: TemporalLoopDistribution + ?Sized,
    F: AffineFoldDims + ?Sized,
    C: Coordinate + ?Sized,
{
    let dim = value.dim;
    let Some(folds) = rhs.folds else {
        return;
    };
    let scale_up = i64::try_from(value.blk_size.count().0).unwrap_or(i64::MAX);
    let mut fold_params = gather_fold_params(folds);
    let spatial_ends = i64::from(rhs.spatial.0) - 1;
    let temporal_ends = spatial_ends + i64::from(rhs.temporal.0);

    for level in &mut fold_params {
        level.alpha = Alpha(level.alpha.0.wrapping_mul(scale_up));
        level.beta = Beta(level.beta.0.wrapping_mul(scale_up));
    }
    // Add a new element-arangement fold for scale block.
    fold_params.push(FoldParamInfo {
        alpha: Alpha(1),
        beta: Beta(0),
        cardinality: value.blk_size.count(),
        label: Some(FoldLabel::ElemArrScaleUp),
    });

    // Figure out all innermost levels that use the same scale element. Exclude the newly inserted
    // element arrangement. ⚠️ A list whose every level is a broadcast or a single trip walks off the
    // FRONT, where the reference reaches `.at(-1)` and throws; -1 still enters the branch below.
    let mut scale_change = i64::try_from(fold_params.len()).unwrap_or(i64::MAX) - 2;
    while let Some(level) = usize::try_from(scale_change)
        .ok()
        .and_then(|at| fold_params.get(at))
    {
        if level.cardinality != Cardinality(1) && level.alpha != Alpha(0) {
            break;
        }
        scale_change -= 1;
    }
    if scale_change <= temporal_ends {
        let inside = usize::try_from(temporal_ends + 1).unwrap_or(0);
        // Assumption: the mx-scale dimension does not have padding.
        let dim_and_kind = PrimaryDimAndKind {
            dim,
            kind: MetaDimKind::Unpadded,
        };
        let related = distribution.related_loops(dim_and_kind, loop_chain, PadType::NoPad);
        let take = usize::try_from(temporal_ends - scale_change).unwrap_or(0);
        let to_distribute = related.get(..take).unwrap_or(&related).to_vec();
        // Distribute temporal loops over element arrangements and compute the fold parameters for
        // temporal and element arrangement folds.
        let redistributed = distribution.distribute(
            &ElemArrDistribution {
                dim: dim_and_kind,
                fold_owner,
                target_lds: value.lds,
                ref_pad: PadType::NoPad,
                target_pad: PadType::NoPad,
                components: RefComponents {
                    size: comp,
                    prop: comp,
                },
                loops_to_distribute: to_distribute.clone(),
                // The element arrangement is scanned innermost first.
                elem_arr: fold_params
                    .get(inside..)
                    .unwrap_or_default()
                    .iter()
                    .rev()
                    .copied()
                    .collect(),
            },
            loop_params,
        );

        // Update the relevant foldParams levels with the newly computed values. ⚠️ The reference's
        // `i <= temporalFoldEnds, j >= 0` binds only `j >= 0` through the comma operator (`:543`);
        // with this take the two bounds coincide exactly, so that defect is not one to repair.
        for (offset, related_loop) in to_distribute.iter().rev().enumerate() {
            let at = usize::try_from(scale_change + 1).unwrap_or(0) + offset;
            let Some(level) = fold_params.get_mut(at) else {
                break;
            };
            // ⚠️ The reference's two `.at()`s throw where the distributor recorded nothing for the
            // loop; a level it decided nothing about keeps the stride it already had.
            if let Some(decided) =
                distribution.distributed(loop_params, related_loop.loop_node, dim)
            {
                level.alpha = decided.alpha;
                level.beta = decided.beta;
            }
        }
        fold_params.truncate(inside);
        fold_params.extend(redistributed);
    }

    re_add_rebuilt_folds(lhs, dim, &fold_params, spatial_ends, temporal_ends);
}

/// Replaces: e235_scaleDownCoord
///
/// Compresses the target's fold list for an MX scale tensor's dim: the innermost levels that fill one
/// scale block collapse into it, and every outer stride and offset divides by the block size.
///
/// ⛔ THE DIVISION IS THE REFERENCE'S `double` ONE (`:601`, `:651`) — a `1.0/blkSize` factor
/// multiplied into an `int64_t`, so truncation toward zero is part of the contract.
/// ⚠️ The scan is decided before any mutation, so the target is untouched on the two refusals.
pub fn scale_down_coord<F, C>(
    scale: &MxScaleTensor,
    rhs: &CoordDim<'_, F>,
    lhs: &mut C,
) -> ScaleDown
where
    F: AffineFoldDims + ?Sized,
    C: Coordinate + ?Sized,
{
    let dim = scale.dim;
    let Some(folds) = rhs.folds else {
        return ScaleDown::NoCoordForDim;
    };
    let blk = scale.blk_size.count().0;
    let mut fold_params = gather_fold_params(folds);
    let spatial_ends = i64::from(rhs.spatial.0) - 1;
    let temporal_ends = spatial_ends + i64::from(rhs.temporal.0);

    // Decide the whole scan before touching the fold list.
    let mut accumulated: u64 = 1;
    let mut at = i64::try_from(fold_params.len()).unwrap_or(i64::MAX) - 1;
    let mut reached: Option<(usize, u64)> = None;
    while at > temporal_ends && at >= 0 {
        let level = usize::try_from(at).unwrap_or(0);
        accumulated = accumulated.saturating_mul(fold_params[level].cardinality.0);
        if accumulated < blk {
            // Check if the next level is part of a contiguous element arrangement, then accumulate
            // more folds to reach the scale block size.
            let outer = level
                .checked_sub(1)
                .and_then(|outer| fold_params.get(outer));
            let contiguous = Alpha(i64::try_from(accumulated).unwrap_or(i64::MAX));
            if outer.is_none_or(|outer| outer.alpha != contiguous) {
                return ScaleDown::NonContiguousInScaleBlock;
            }
            at -= 1;
            continue;
        }
        reached = Some((level, accumulated));
        break;
    }

    if let Some((level, accumulated)) = reached {
        if accumulated > blk {
            fold_params[level].cardinality = Cardinality(accumulated / blk);
            fold_params[level].alpha = Alpha(1);
            fold_params.truncate(level + 1);
        } else {
            // accumulatedCardinality == scaleBlkSize
            fold_params.truncate(level);
        }
        let factor = 1.0_f64 / blk as f64;
        for outer in fold_params.iter_mut().take(level) {
            if outer.alpha.0 > 1 {
                outer.alpha = Alpha((outer.alpha.0 as f64 * factor) as i64);
                outer.beta = Beta((outer.beta.0 as f64 * factor) as i64);
            }
        }
    }

    re_add_rebuilt_folds(lhs, dim, &fold_params, spatial_ends, temporal_ends);
    ScaleDown::Applied
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ONE `data_connect=`'s NODE GROUP — as entry 236 reads it.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH END OF A `data_connect=` — `checkProducers` (`ddc/ddc_fold.cpp:688`), which selects both the
/// node set AND the side of a transfer whose component is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectEnd {
    /// `producers_`.
    Producers,
    /// `consumers_`.
    Consumers,
}

impl ConnectEnd {
    /// `getComponent(node, /*getSrc=*/!checkProducers)` (`:695`) — a PRODUCER of the connect is asked
    /// for its DESTINATION component, because that is the end the data leaves by.
    #[must_use]
    pub const fn node_side(self) -> TransferSide {
        match self {
            Self::Producers => TransferSide::Dst,
            Self::Consumers => TransferSide::Src,
        }
    }
}

/// ONE MEMBER OF A `data_connect=`'s NODE SET — its identity in the tree plus the node itself, which
/// is how the reference's one `ScheduleNode*` answers both `getComponent` and `getPrev`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectedNode<'a> {
    /// The node's place in the tree.
    pub id: NodeId,
    /// The node.
    pub node: Node<'a>,
}

/// Replaces: e236_needNonRowBundling
///
/// Whether a group of non-PT-row nodes on one `data_connect=` has to be bundled — true unless each
/// of them is under a conditional of its own.
///
/// ⛔ TRAP CLOSED: the reference's `while (currNode != commonAncestor)` walk dereferences null once
/// it runs off the root (`:708-718`); a node whose chain never reaches the ancestor contributes
/// nothing here. ⭐ The count is keyed on the node whose PARENT is the condition, not the condition.
#[must_use]
pub fn need_non_row_bundling<T: ScheduleTree + ?Sized>(
    tree: &T,
    nodes: &[ConnectedNode<'_>],
    end: ConnectEnd,
) -> bool {
    // The reference's `nodeSet` is a set, so these are distinct nodes.
    if nodes.len() <= 1 {
        return false;
    }
    if nodes
        .iter()
        .any(|member| comp_row_id(component(member.node, end.node_side())).is_some())
    {
        return false;
    }
    let ids: Vec<NodeId> = nodes.iter().map(|member| member.id).collect();
    let Some(common_ancestor) = find_common_ancestor(tree, &ids, &[NodeKind::Loop]) else {
        // TO DO: Is this an error condition?
        return false;
    };
    let mut descendant_count: BTreeMap<NodeId, u32> = BTreeMap::new();
    for member in nodes {
        let mut curr = Some(member.id);
        while let Some(walked) = curr.filter(|walked| *walked != common_ancestor) {
            if tree
                .parent(walked)
                .is_some_and(|prev| tree.kind(prev) == NodeKind::Condition)
            {
                // Non-row operation is enclosed in conditionals. Keep track of the number of
                // descendants per conditional branch.
                *descendant_count.entry(walked).or_default() += 1;
                break;
            }
            curr = tree.parent(walked);
        }
    }
    if descendant_count.is_empty() {
        // None of the nodes in nodeSet are enclosed in any conditional up to the common ancestor.
        return true;
    }
    // Multiple nodes under the same conditional is a non-PT-row bundling situation; each under a
    // separate conditional needs no bundling.
    descendant_count.values().any(|&count| count > 1)
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// TWO COORDINATES SIDE BY SIDE — as entry 237 compares them.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE COORDINATE VALUE — what `getSingleData` answers, a `CoordinateBaseType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoordValue(pub i64);

/// `FoldManager::getSingleData(posToFixCoord)` — the coordinate reached by holding the named fold
/// positions at the given indices and every other position at index 0.
///
/// ⚠️ NOT A PORT: `getSingleData` lives in `util/foldManager/` (a recorded outside dependency), and
/// this is its all-affine evaluation — `Σ_i (α_i · idx_i + β_i)` over the dim's levels.
#[must_use]
pub fn single_data<F: AffineFoldDims + ?Sized>(
    folds: &F,
    fixed: &[(FoldPosition, i64)],
) -> CoordValue {
    let mut total: i64 = 0;
    for level in 0..folds.num_dims() {
        let (alpha, beta) = folds.alpha_beta(level);
        let index = fixed
            .iter()
            .find(|(pos, _)| pos.index() == level)
            .map_or(0, |&(_, index)| index);
        total = total
            .wrapping_add(alpha.0.wrapping_mul(index))
            .wrapping_add(beta.0);
    }
    CoordValue(total)
}

/// WHICH CORE — `coreId`, the key of `coreIdToWkSlice_` and of `getRelevantCoreCl`'s answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreOrdinal(pub u32);

/// WHICH CORELET WITHIN A CORE — the `std::set<int>` beside each core in `getRelevantCoreCl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreletOrdinal(pub u32);

/// WHICH WORK SLICE OF A DIM A CORE OWNS — `coreIdToWkSlice_.at(coreId).at(dim)`, the index entry 237
/// holds [`FoldPosition::Core`] at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkSlice(pub i64);

/// THE WORK-SLICE ASSIGNMENT — `coreIdToWkSlice_` (`dsc/dsc2.h`), a coordinate's own or the SuperDSC's.
///
/// ⛔ EMPTY IS A STATE, NOT AN ABSENCE: entry 237 reads emptiness to decide whether a coordinate
/// slices work specially at all (`:1631-1636`), and separately falls back to the SuperDSC's map.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WorkSlices(pub BTreeMap<CoreOrdinal, BTreeMap<PrimaryDim, WorkSlice>>);

/// ONE SIDE OF THE COMPARISON — a coordinate together with the node facts entry 237 asks of it.
pub struct CoordSide<'a, F: AffineFoldDims + ?Sized> {
    /// `coordinates_` — this side's fold manager per dim.
    pub dims: BTreeMap<PrimaryDim, &'a F>,
    /// `coreIdToWkSlice_`.
    pub work_slices: &'a WorkSlices,
    /// `node->getRelevantCoreCl(comp)`, with the reference's `NO_COMPONENT` fallback (`:1649-1653`)
    /// already applied — which is the only fact the `comp` argument was ever reached through.
    pub cores: BTreeMap<CoreOrdinal, BTreeSet<CoreletOrdinal>>,
}

/// WHETHER A DIM ONE SIDE LACKS IS A MISMATCH — `commonDimsOnly` (`ddc/ddc_fold.cpp:1266`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimCoverage {
    /// `commonDimsOnly == false` — a differing dim count, or a dim the RHS lacks, answers `false`.
    EveryDim,
    /// `commonDimsOnly == true` — only the dims both sides cover are compared.
    CommonDimsOnly,
}

/// `isConsistentCoord` (`:1274-1329`), entry 237's own lambda: level for level, the same trip counts
/// and the same strides wherever a level trips more than once, and the same total offset.
///
/// ⛔ THE STRIDE IS UNCHECKED AT CARDINALITY 1 (`:1299-1301`) — a level taken once reaches the same
/// element whatever its stride, so only its offset can disagree, and the offsets are summed.
fn is_consistent_coord(lhs: &[FoldParamInfo], rhs: &[FoldParamInfo]) -> bool {
    if lhs.len() != rhs.len() {
        // LHS and RHS do not have the same number of coordinate levels after combining.
        return false;
    }
    let mut lhs_beta: i64 = 0;
    let mut rhs_beta: i64 = 0;
    for (lhs_level, rhs_level) in lhs.iter().zip(rhs) {
        if lhs_level.cardinality != rhs_level.cardinality
            || (lhs_level.cardinality > Cardinality(1) && lhs_level.alpha != rhs_level.alpha)
        {
            return false;
        }
        lhs_beta = lhs_beta.wrapping_add(lhs_level.beta.0);
        rhs_beta = rhs_beta.wrapping_add(rhs_level.beta.0);
    }
    lhs_beta == rhs_beta
}

/// `fillCoreIdToWkSlice` (`:1645-1667`), entry 237's second lambda: `<coreId, <work-slice, set of
/// corelets>>` for the cores this side is relevant on.
///
/// ⚠️ A CORE WHOSE MAP DOES NOT NAME THE DIM IS SKIPPED, where the reference's `.at(coordDim)`
/// throws; a missing CORE is already the reference's own `continue`.
fn core_slices<'a, F: AffineFoldDims + ?Sized>(
    side: &'a CoordSide<'_, F>,
    sdsc_slices: &'a WorkSlices,
    dim: PrimaryDim,
) -> BTreeMap<CoreOrdinal, (WorkSlice, &'a BTreeSet<CoreletOrdinal>)> {
    let slices = if side.work_slices.0.is_empty() {
        sdsc_slices
    } else {
        side.work_slices
    };
    side.cores
        .iter()
        .filter_map(|(&core, corelets)| {
            let slice = *slices.0.get(&core)?.get(&dim)?;
            Some((core, (slice, corelets)))
        })
        .collect()
}

/// Replaces: e237_sameCoordinateRange
///
/// Whether two coordinates cover the same range: level for level where neither side slices work
/// specially, else start coordinate for start coordinate at each core's own work slice.
///
/// ⛔⛔ HALF THE BODY IS DEAD AND IS NOT REPRODUCED — the `else` of `if (true)` (`:1497-1505`), and
/// with it both working copies, whose one surviving read is a corelet cardinality their own mutation
/// cannot reach. Three `DT_ERROR`/`DT_CHECK` arms answer `false` rather than aborting.
#[must_use]
pub fn same_coordinate_range<F: AffineFoldDims + ?Sized>(
    lhs: &CoordSide<'_, F>,
    rhs: &CoordSide<'_, F>,
    sdsc_slices: &WorkSlices,
    broadcast_dims: &BTreeSet<PrimaryDim>,
    coverage: DimCoverage,
) -> bool {
    let common_dims_only = coverage == DimCoverage::CommonDimsOnly;
    if !common_dims_only && lhs.dims.len() != rhs.dims.len() {
        return false;
    }
    let lhs_has_custom_slicing = !lhs.work_slices.0.is_empty() && lhs.work_slices != sdsc_slices;
    let rhs_has_custom_slicing = !rhs.work_slices.0.is_empty() && rhs.work_slices != sdsc_slices;
    let corelet = FoldPosition::Corelet.index();

    for (&coord_dim, &lhs_folds) in &lhs.dims {
        let Some(&rhs_folds) = rhs.dims.get(&coord_dim) else {
            // RHS does not have coordinates for this dim.
            if common_dims_only {
                continue;
            }
            return false;
        };
        let mut lhs_fold_params = gather_fold_params(lhs_folds);
        let mut rhs_fold_params = gather_fold_params(rhs_folds);

        if (!lhs_has_custom_slicing && !rhs_has_custom_slicing)
            || lhs.work_slices == rhs.work_slices
        {
            combine_contigous_levels(&mut lhs_fold_params, 0, None);
            combine_contigous_levels(&mut rhs_fold_params, 0, None);
            if !is_consistent_coord(&lhs_fold_params, &rhs_fold_params) {
                return false;
            }
            continue;
        }

        // Skip broadcast dimensions.
        if broadcast_dims.contains(&coord_dim) {
            continue;
        }
        let lhs_cores = core_slices(lhs, sdsc_slices, coord_dim);
        let rhs_cores = core_slices(rhs, sdsc_slices, coord_dim);
        // First, check that the size of the two lists are same.
        if lhs_cores.len() != rhs_cores.len() {
            return false;
        }
        // ⚠️ The two working copies fix the CORE level, which no read below reaches; a fold list too
        // short to have a corelet level answers 0 trips, where the reference's `.at()` throws.
        let lhs_corelet_trips = lhs_fold_params
            .get(corelet)
            .map_or(Cardinality(0), |level| level.cardinality);
        let rhs_corelet_trips = rhs_fold_params
            .get(corelet)
            .map_or(Cardinality(0), |level| level.cardinality);

        for (core, &(lhs_slice, lhs_corelets)) in &lhs_cores {
            let Some(&(rhs_slice, rhs_corelets)) = rhs_cores.get(core) else {
                // Workslice information for this coreId was not found on the RHS.
                return false;
            };
            // Check if need to fix corelets. Assumption: only one corelet is active.
            let mut fixed_lhs_corelet = None;
            let mut fixed_rhs_corelet = None;
            if Cardinality(lhs_corelets.len() as u64) < lhs_corelet_trips {
                fixed_lhs_corelet = lhs_corelets.first().copied();
            } else if Cardinality(rhs_corelets.len() as u64) < rhs_corelet_trips {
                fixed_rhs_corelet = rhs_corelets.first().copied();
            } else if lhs_corelet_trips != rhs_corelet_trips {
                // Unexpected corelet cardinality mismatch.
                return false;
            }

            // Verifying the full range of coordinates requires both the steady state and the
            // epilogue, and the coordinate framework does not keep the epilogue; as a temporary
            // workaround for PSUM, check the START coordinate only.
            let mut lhs_fixed = vec![(FoldPosition::Core, lhs_slice.0)];
            if let Some(fixed) = fixed_lhs_corelet {
                lhs_fixed.push((FoldPosition::Corelet, i64::from(fixed.0)));
            }
            let mut rhs_fixed = vec![(FoldPosition::Core, rhs_slice.0)];
            if let Some(fixed) = fixed_rhs_corelet {
                rhs_fixed.push((FoldPosition::Corelet, i64::from(fixed.0)));
            }
            if single_data(lhs_folds, &lhs_fixed) != single_data(rhs_folds, &rhs_fixed) {
                return false;
            }
        }
    }

    true
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// A COMPUTE NODE'S DATASTREAMS — as entry 238 searches them.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// AN OPAQUE OP'S DATASTREAMS — `instrAttribute_`'s two `data_connects_` lists plus the labeled DS
/// `metadata.opaqueOps_` records for the node (`ddc/ddc_metadata.h`).
///
/// ⛔ ONLY AN OPAQUE OP MATCHES BY `data_connect=` ALONE, which is why this is its own value: for a
/// non-opaque compute the connect lives on each stream, and an allocate reference is matched by
/// allocation identity instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpaqueStreams<'a> {
    /// `input_data_connects_`.
    pub inputs: &'a [DataConnect],
    /// `output_data_connects_`.
    pub outputs: &'a [DataConnect],
    /// `metadata.opaqueOps_.at(computeNode).ldsIdx_`, absent for its `-1`.
    pub lds: Option<LdsIdx>,
}

/// A COMPUTE NODE AS ENTRY 238 READS IT — its unit, its stored streams, and its opaque form if it
/// has one (`isOpaqueOp_`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeStreams<'a> {
    /// `exUnit_` — read only for the scale-to-value redirect's `LXLU` exemption.
    pub ex_unit: DfirUnit,
    /// `inputsLdsAndLoopOffsets_` zipped with `inputs_`.
    pub inputs: &'a [StoredStream],
    /// `outputsLdsAndLoopOffsets_` zipped with `outputs_`.
    pub outputs: &'a [StoredStream],
    /// `isOpaqueOp_`, with the attribute lists it makes readable.
    pub opaque: Option<OpaqueStreams<'a>>,
}

/// THE COORDINATE REFERENCE'S NODE — `refFoldInfo.refNode`'s `nodeType_` reduced to the two arms
/// entry 238 supports.
///
/// ⛔⛔ THE UNSUPPORTED-KIND `DT_ERROR` (`:2060-2067`) IS UNSPELLABLE OVER THIS ENUM: `TRANSFER` and
/// `COMPUTE` are ONE arm, because the reference treats them identically once it has excluded
/// `ALLOCATE`, and no third kind can be named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeRefNode {
    /// `ALLOCATE` — matched by allocation identity, with the scale-to-value redirect.
    Allocation {
        /// `refAllocNode`.
        alloc: AllocId,
        /// `labeledDs_.at(ldsIdx_).scaledLdsCategory_`, absent for `ldsIdx_ == -1`.
        scaled: Option<ScaledLds>,
    },
    /// `TRANSFER` or `COMPUTE` — matched by `data_connect=`.
    Streamed,
}

/// THE COORDINATE REFERENCE — `dsc2::CoordPropInfoType` as entry 238 reads one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeCoordRef {
    /// `dataConnect`, absent for the reference's `""`.
    pub data_connect: Option<DataConnect>,
    /// `refNode`.
    pub ref_node: ComputeRefNode,
}

/// WHICH OF A COMPUTE'S COORDINATES — `inputCoordinates_.at(i)` or the single `outputCoordinate_`,
/// with the index the reference pushes into `constructedInput`/`OutputCoords`.
///
/// ⛔ THE OUTPUT INDEX DOES NOT SELECT A COORDINATE: a compute has ONE `outputCoordinate_` however
/// many outputs it has (`:2008`, `:2047`, `:2093`), and the index is what the caller records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelatedComputeCoord {
    /// `inputCoordinates_.at(i)`.
    Input(usize),
    /// `outputCoordinate_`, reached through output `i`.
    Output(usize),
}

/// THE COORDINATE ENTRY 238 SELECTED, WITH THE LABELED DS IT CAME THROUGH — `selectedLdsIdx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelatedCoord {
    /// Which coordinate.
    pub coord: RelatedComputeCoord,
    /// `selectedLdsIdx`, absent for its `-1` — which a constant-backed stream leaves it at.
    pub lds: Option<LdsIdx>,
}

/// `selectedLdsIdx = ..myLdsIdx_` — a constant-backed stream has none.
const fn selected_lds(stored: StoredStream) -> Option<LdsIdx> {
    match stored.stream.origin {
        DataOrigin::LabeledDs(lds) => Some(lds),
        DataOrigin::Constant(_) => None,
    }
}

/// Replaces: e238_getRelatedComputeCoord
///
/// Which of a compute's coordinates the propagation refers to — by `data_connect=` for an opaque op
/// or a streamed reference, and by allocation identity for an allocate reference.
///
/// ⭐ THE ALLOCATE ARM *IS* [`match_data_stream`] WITH THE CONNECT SUPPRESSED: its scale-to-value
/// redirect (`:2018-2027`) is character for character `:461-471`. ⛔ The four `DT_ERROR`s become
/// [`None`], and the unsupported-`nodeType_` one is unspellable over [`ComputeRefNode`].
#[must_use]
pub fn get_related_compute_coord<A: Allocations + ?Sized>(
    dsc: &A,
    compute: &ComputeStreams<'_>,
    reference: &ComputeCoordRef,
) -> Option<RelatedCoord> {
    // Future work: Should we associate allocateNodes with data_connects?
    if let Some(opaque) = compute.opaque {
        let connect = reference.data_connect?;
        if let Some(at) = opaque.inputs.iter().position(|&dc| dc == connect) {
            return Some(RelatedCoord {
                coord: RelatedComputeCoord::Input(at),
                lds: opaque.lds,
            });
        }
        let at = opaque.outputs.iter().position(|&dc| dc == connect)?;
        return Some(RelatedCoord {
            coord: RelatedComputeCoord::Output(at),
            lds: opaque.lds,
        });
    }

    match reference.ref_node {
        // Assumption: all datastreams related to a given allocate node have the same coordinate
        // arrangement, so the first matching datastream is sufficient.
        ComputeRefNode::Allocation { alloc, scaled } => {
            let prop = CoordPropInfo {
                data_connect: None,
                ref_node: PropEnd::Allocate { alloc, scaled },
                node_to_fold: PropEnd::Compute {
                    ex_unit: compute.ex_unit,
                },
            };
            let matches = |&stored: &StoredStream| match_data_stream(dsc, &prop, stored, true);
            if let Some(at) = compute.inputs.iter().position(matches) {
                return Some(RelatedCoord {
                    coord: RelatedComputeCoord::Input(at),
                    lds: selected_lds(compute.inputs[at]),
                });
            }
            let at = compute.outputs.iter().position(matches)?;
            Some(RelatedCoord {
                coord: RelatedComputeCoord::Output(at),
                lds: selected_lds(compute.outputs[at]),
            })
        }
        // Assumption: all datastreams related to a given data_connect have the same coordinate
        // arrangement, so the first matching datastream is sufficient.
        ComputeRefNode::Streamed => {
            let connect = reference.data_connect?;
            let names = |stored: &StoredStream| stored.stream.data_connect == Some(connect);
            if let Some(at) = compute.inputs.iter().position(names) {
                return Some(RelatedCoord {
                    coord: RelatedComputeCoord::Input(at),
                    lds: selected_lds(compute.inputs[at]),
                });
            }
            let at = compute.outputs.iter().position(names)?;
            Some(RelatedCoord {
                coord: RelatedComputeCoord::Output(at),
                lds: selected_lds(compute.outputs[at]),
            })
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// A ROW-SPLIT GROUP — as entry 239 reads one.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT KIND OF ROW GROUP — `RowGroupInfo::Category` (`ddc/ddc.h:556-562`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowGroupCategory {
    /// `ROW_TO_SAME_ROW`.
    RowToSameRow,
    /// `NONROW_TO_ROW`.
    NonRowToRow,
    /// `ROW_TO_NONROW`.
    RowToNonRow,
    /// `ROW_NORTH_SOUTH`.
    RowNorthSouth,
    /// `NO_BUNDLING`, the field's own default.
    NoBundling,
}

/// ONE NODE OF A ROW GROUP — `RowGroupInfo::RowGroupNodeInfo` (`ddc/ddc.h:566-573`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGroupNode {
    /// `node`.
    pub node: NodeId,
    /// `row`, absent for its `-1` default. ⭐ [`None`] SORTS BELOW EVERY ROW, exactly as `-1` does,
    /// so the monotonic scan reads the same either way.
    pub row: Option<PtRowId>,
    /// `beta` — the offset this node's coordinate uses for row grouping.
    pub beta: Beta,
}

/// A ROW GROUP — `RowGroupInfo` (`ddc/ddc.h:555`) reduced to the category and the node list.
///
/// ⛔⛔ NON-EMPTY BY CONSTRUCTION, and that is what retires the `nodeInfo.empty()` arm: entry 239's
/// `nodeInfo.at(0)` runs before any category test (`:2168-2175`), so an empty group and an absent
/// group are ONE absence. `commonGroupAncestor`, `activeRow` and `ascendingOrder` are not read here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGroup {
    cat: RowGroupCategory,
    first: RowGroupNode,
    rest: Vec<RowGroupNode>,
}

impl RowGroup {
    /// A group of at least one node, and this is how that is stated.
    #[must_use]
    pub const fn new(cat: RowGroupCategory, first: RowGroupNode, rest: Vec<RowGroupNode>) -> Self {
        Self { cat, first, rest }
    }

    /// `cat`.
    #[must_use]
    pub const fn category(&self) -> RowGroupCategory {
        self.cat
    }

    /// `nodeInfo.at(0)` — total.
    #[must_use]
    pub const fn first(&self) -> &RowGroupNode {
        &self.first
    }

    /// `nodeInfo.at(1)`, absent for the one-node group where the reference throws.
    #[must_use]
    pub fn second(&self) -> Option<&RowGroupNode> {
        self.rest.first()
    }

    /// Every node of the group, in order.
    pub fn iter(&self) -> impl Iterator<Item = &RowGroupNode> {
        std::iter::once(&self.first).chain(self.rest.iter())
    }

    /// `nodeInfo.size()`.
    #[must_use]
    pub fn count(&self) -> usize {
        1 + self.rest.len()
    }
}

/// Replaces: e239_computeParamsForRowSplitFold
///
/// The row-split fold's parameters for one dim: the identity fold where the core stage does not row
/// split it, a single-row fold, or one level spanning every PT row.
///
/// ⛔ [`None`] IS THE REFERENCE'S THREE ABORTS: `nodeInfo.at(1)` on a one-node group of any category
/// but `ROW_TO_SAME_ROW` (`:2172`), a break in the all-rows monotonic sequence (`:2186`, `:2196`),
/// and a node count that is neither one row nor every row (`:2215`).
#[must_use]
pub fn compute_params_for_row_split_fold<A: Arch, S: Stage + ?Sized>(
    core_stage: &S,
    dim: PrimaryDim,
    row_group: Option<&RowGroup>,
) -> Option<FoldParamInfo> {
    // Setup parameters for the rowsplit fold. The default parameters represent an identity fold.
    let Some(row_group) = row_group.filter(|_| core_stage.is_row_split(dim)) else {
        return Some(default_row_split_fold());
    };
    let first = *row_group.first();
    // Jump in coordinate from one PT-row to the next.
    let (alpha, beta) = if row_group.category() == RowGroupCategory::RowToSameRow {
        (Alpha(0), first.beta)
    } else {
        let second = *row_group.second()?;
        (Alpha(second.beta.0.wrapping_sub(first.beta.0)), first.beta)
    };
    let num_nodes = u32::try_from(row_group.count()).unwrap_or(u32::MAX);

    match row_group.category() {
        // Rowsplit fold is a single PT row.
        RowGroupCategory::RowToSameRow | RowGroupCategory::RowNorthSouth => Some(FoldParamInfo {
            alpha,
            beta,
            cardinality: Cardinality(1),
            label: Some(FoldLabel::RowSplitFold),
        }),
        // Rowsplit fold is for all PT rows, whose row numbers must be monotonic. ⚠️ A break in the
        // expected sequence names two schedule nodes in the reference's message; that pair is the
        // caller's to report, because a fold builder that cannot state one is still correct.
        _ if num_nodes == A::PT_ROWS => {
            let rows: Vec<Option<PtRowId>> = row_group.iter().map(|node| node.row).collect();
            let ascending = rows.first() < rows.get(1);
            if rows.windows(2).any(|pair| {
                if ascending {
                    pair[0] >= pair[1]
                } else {
                    pair[0] <= pair[1]
                }
            }) {
                return None;
            }
            Some(FoldParamInfo {
                alpha,
                beta: first.beta,
                cardinality: Cardinality(u64::from(A::PT_ROWS)),
                label: Some(FoldLabel::RowSplitFold),
            })
        }
        // Unsupported number of rows.
        _ => None,
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// AN EXTERNAL ALLOCATION AND THE LOOPS AROUND IT — as entry 240 reads them.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// HOW ONE LAYOUT DIM IS SCALED — `labeledDs_.scale_.at(getDimIndexInLayoutOrder(dsType_, dim))`
/// resolved, with the reference's "no index answers 1" (`:2288`) already applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimScale {
    /// A non-negative scale: the dim carries real data.
    Sized,
    /// A negative scale: the dim is broadcast, and this is what its arrangement costs.
    Broadcast(BroadcastScale),
}

/// AN EXTERNAL ALLOCATION — `dsc2::AllocateNode` reduced to what entry 240 reads off one.
///
/// ⛔⛔ `ldsIdx_ < 0` IS NOT A STATE HERE (`:2249`): the reference returns having done nothing, so an
/// allocation with no labeled DS is simply one this value cannot be built for.
/// ⭐ `setPadding(allocNode->padding_)` (`:2268`) STAMPS THE FORM ONTO THE COORDINATE and it STAYS
/// THERE: other units read the coordinate's own form afterwards (`:2718`, `:2821`, `:3774`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7581`), so it is state, not a re-spelling of `padding_`.
pub struct ExternalAllocation<'a> {
    /// `ldsIdx_`.
    pub lds: LdsIdx,
    /// `component_`.
    pub component: SenComponent,
    /// `layoutDimOrder_`, each dim with its resolved scale.
    pub layout: &'a [(PrimaryDimAndKind, DimScale)],
    /// `padding_`.
    pub padding: &'a PaddingForm,
}

/// HOW MANY TIMES ONE TEMPORAL LOOP TRIPS FOR A DIM, AND BY HOW MUCH — the two arms of
/// `isParametricLoop()` (`:2331-2355`).
///
/// ⛔⛔ THE ARM IS LOAD-BEARING, NOT DECORATION: the distributor's decision overrides `alpha` on the
/// STAGED arm ONLY (`:2352-2355`), so collapsing the two would apply it to parametric loops too.
/// ⚠️ The staged arm's `numDs / denDs` division lives behind this seam, and with it its divide by zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopTrip {
    /// `parametricIterCount(currDsc, 0, NO_COMPONENT, -1)` and `parametricStride(currDsc)`.
    Parametric {
        /// The trip count.
        iterations: Cardinality,
        /// The stride.
        alpha: Alpha,
    },
    /// `numDs.primaryDimToVal_st(..) / denDs.primaryDimToVal_st(..)`, and the denominator as the
    /// stride — which the distributor may override.
    Staged {
        /// The trip count.
        iterations: Cardinality,
        /// The stride.
        alpha: Alpha,
    },
}

/// THE CORE STAGE PLUS THE TWO EXTRA QUESTIONS ENTRY 240 PUTS TO IT.
pub trait ExternalFoldStage: CoreStage {
    /// `dataStageDimToVal_compView_st(dim, NO_COMPONENT, 0, {dim, pad})` (`:2279-2283`) — ONE
    /// CORELET's extent for the dim, which is the element arrangement's trip count.
    ///
    /// ⛔ NOT [`CoreStage::core_extent`], whose corelet view is `-1`: that one is the whole core.
    fn corelet_extent(&self, dim: PrimaryDim, pad: PadType) -> Extent;

    /// One temporal loop's trip count and stride for one dim (`:2331-2355`).
    fn loop_trip(&self, loop_node: &LoopNode, dim: PrimaryDimAndKind) -> LoopTrip;
}

/// `addFold(dim, cat, card, label, alpha, beta, /*pos=*/0)` with a label the reference builds by
/// CONCATENATION — `"elem_arr_" + i`, `loopNode->name_ + " " + dim`, `"rowsplit_fold_" + dim`.
///
/// ⚠️ NOT [`FoldLabel`]: that is the closed set of spellings this file mints as LITERALS, and a
/// per-index or per-loop name is not in it. [`dsc2::FoldLabel`] is the open one, as its own doc says.
fn add_labelled_fold(
    coord: &mut dsc2::Coordinate,
    dim: PrimaryDim,
    category: CoordinateCategory,
    label: String,
    level: FoldParamInfo,
) {
    coord.add_fold_front(
        dim,
        category,
        FoldCardinality(u32::try_from(level.cardinality.0).unwrap_or(u32::MAX)),
        dsc2::FoldLabel(label),
        FoldCoeff(level.alpha.0),
        FoldCoeff(level.beta.0),
    );
}

/// Replaces: e240_buildFoldForExternalAllocation
///
/// Builds an external allocation's coordinate from scratch — for each layout dim it does not already
/// cover: the element arrangement, one temporal level per related loop, the row split, then the
/// spatial pair. The whole coordinate latches as constructed at the end.
///
/// ⚠️ `coordinates_[currDim.dim_]` (`:2276`) is a DEAD `operator[]` INSERTION: every arm below adds
/// at least one fold to the dim, so the dim is covered on the next pass either way.
pub fn build_fold_for_external_allocation<'l, S, T>(
    alloc: &ExternalAllocation<'_>,
    fold_owner: NodeId,
    stage: &S,
    distribution: &T,
    loop_chain: &[LoopAndDim<'l>],
    coord: &mut dsc2::Coordinate,
    loop_params: &mut T::LoopParams,
) where
    S: ExternalFoldStage + ?Sized,
    T: TemporalLoopDistribution + ?Sized,
{
    if coord.fold_constructed() {
        // Upstream-filled coordinates of an external allocate node.
        return;
    }
    if loop_chain.is_empty() {
        // TO DO: Determine a robust response.
        return;
    }

    coord.set_padding_form(alloc.padding.stated());

    // Construct a foldManager for each relevant dimension.
    for &(curr_dim, dim_scale) in alloc.layout {
        let dim = curr_dim.dim;
        if coord.covers(dim) {
            continue;
        }
        let fold_dim_str = dim.spelling();
        let alloc_padding_for_curr_dim = coord.padding(dim);

        // Build element arrangement info along with the temporal fold. TO DO: this is a placeholder
        // and includes only the monotonically increasing contiguous case.
        let elem_arr = vec![FoldParamInfo {
            alpha: Alpha(1),
            beta: Beta(0),
            cardinality: Cardinality(
                u64::try_from(stage.corelet_extent(dim, alloc_padding_for_curr_dim).0)
                    .unwrap_or_default(),
            ),
            label: None,
        }];

        if let DimScale::Broadcast(scale) = dim_scale {
            // This is a broadcast dimension.
            build_fold_for_broadcast_dim(coord, dim, scale);
            continue;
        }

        // The dimension is a non-broadcast dimension.
        let related_loops =
            distribution.related_loops(curr_dim, loop_chain, alloc_padding_for_curr_dim);
        // Distribute temporal loops over element arrangements and compute the fold parameters for
        // temporal and element arrangement folds.
        let after_distribution = distribution.distribute(
            &ElemArrDistribution {
                dim: curr_dim,
                fold_owner,
                target_lds: alloc.lds,
                ref_pad: alloc_padding_for_curr_dim,
                target_pad: alloc_padding_for_curr_dim,
                components: RefComponents {
                    size: alloc.component,
                    prop: alloc.component,
                },
                loops_to_distribute: related_loops.clone(),
                elem_arr,
            },
            loop_params,
        );

        // Add the element arrangement folds.
        for (at, level) in after_distribution.iter().enumerate() {
            add_labelled_fold(
                coord,
                dim,
                CoordinateCategory::ElemArr,
                format!("elem_arr_{at}"),
                *level,
            );
        }

        // Temporal folds for the loops. Order: innermost to outermost.
        for related_loop in &related_loops {
            let (iterations, alpha) =
                match stage.loop_trip(related_loop.loop_node, related_loop.dim) {
                    LoopTrip::Parametric { iterations, alpha } => (iterations, alpha),
                    LoopTrip::Staged { iterations, alpha } => (
                        iterations,
                        distribution
                            .distributed(loop_params, related_loop.loop_node, dim)
                            .map_or(alpha, |decided| decided.alpha),
                    ),
                };
            add_labelled_fold(
                coord,
                dim,
                CoordinateCategory::Temporal,
                format!("{} {fold_dim_str}", related_loop.loop_node.name.0),
                FoldParamInfo {
                    alpha,
                    beta: Beta(0),
                    cardinality: iterations,
                    label: None,
                },
            );
        }

        // Build rowsplit fold.
        add_labelled_fold(
            coord,
            dim,
            CoordinateCategory::Spatial,
            format!("rowsplit_fold_{fold_dim_str}"),
            FoldParamInfo {
                alpha: Alpha(0),
                beta: Beta(0),
                cardinality: Cardinality(1),
                label: None,
            },
        );
        // Add spatial folds for core and corelet.
        build_spatial_fold(stage, dim, alloc_padding_for_curr_dim, coord);
    }

    coord.complete_fold_construction();
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// PT-ROW GROUPING'S BASE WALK — entry 297 and the candidate vocabulary it reads.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE COORDINATE WITHOUT ITS NODE — `coordinates_` and `coreIdToWkSlice_`.
///
/// ⛔ APART FROM [`CoordSide`] BECAUSE THE REFERENCE COMPARES TWO NODES' COORDINATES AGAINST *ONE*
/// NODE: `sameCoordinateRange(node, *rowNode[rowId], node, *coord, …)` (`ddc/ddc_fold.cpp:915`) hands
/// the CURRENT node in on both sides, so the core/corelet facts belong to the candidate and not to
/// the coordinate recorded for its row.
pub struct RowCoordinate<'a, F: AffineFoldDims + ?Sized> {
    /// `coordinates_` — the fold manager per dim.
    pub dims: BTreeMap<PrimaryDim, &'a F>,
    /// `coreIdToWkSlice_`.
    pub work_slices: &'a WorkSlices,
}

/// ONE OPERAND OF A ROW CANDIDATE, in the order the reference scans that side.
pub struct RowOperand<'a, F: AffineFoldDims + ?Sized> {
    /// Which operand it is, as entry 084 names it: a transfer's `Input(0)` source or its `Output(i)`
    /// destination, a compute's `Input(i)` or `Output(i)`.
    pub pos: OperandPos,
    /// The datastream and the memory it lives in, as entry 091 matches it.
    pub stream: StoredStream,
    /// Where this operand's fold is recorded — `transferCoordinates_`, shared by every destination of
    /// a transfer; `outputCoordinate_`, shared by every output of a compute; `inputCoordinates_.at(i)`
    /// per input.
    pub coord: &'a RowCoordinate<'a, F>,
    /// `labeledDs_.at(myLdsIdx_)` AS AN MX VALUE TENSOR, absent for any other `scaledLdsCategory_` —
    /// which is what turns the reference's category test into a type.
    pub mx: Option<MxValueTensor>,
}

/// A NODE THE REFERENCE'S DFS OFFERED — `traverseTreeDFSMutable(nullptr, nodeTypes)` (`:8163`).
///
/// ⛔ THE FILTER IS THE REFERENCE NODE'S OWN KIND, AND AN EMPTY FILTER FILTERS NOTHING
/// (`dsc/dsc2.cpp:2247`): a reference node that is neither a compute nor a transfer offers BOTH, so
/// the walk is not narrower than the two arms below.
pub struct RowCandidate<'a, F: AffineFoldDims + ?Sized> {
    /// The node's tree identity.
    pub id: NodeId,
    /// The node, for entry 084's layout read and to pick the arm.
    pub node: Node<'a>,
    /// Its PT row — a transfer's `src_.unit_` when that is a row unit else its FIRST destination's, a
    /// compute's `exUnit_`. [`None`] is the reference's `continue`: nothing here touches PT.
    pub row: Option<PtRowId>,
    /// `node->getRelevantCoreCl(SenComponents::ALL)`.
    pub cores: BTreeMap<CoreOrdinal, BTreeSet<CoreletOrdinal>>,
    /// The side `refIsProducer` selects, in scan order — a transfer's destinations or its one source,
    /// a compute's outputs or its inputs. THE CALLER RESOLVES THE ROLE, because it only decides which
    /// side is scanned.
    pub operands: Vec<RowOperand<'a, F>>,
}

/// WHAT ENTRY 297 FOUND — its `bool` return fused with what it wrote through its `RowGroupInfo &`,
/// which are one answer. EVERY variant leaves `nodeInfo` and `commonGroupAncestor` replaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelatedPtRows {
    /// `cat = NO_BUNDLING`, `true` — no dim is split across rows, or the row-split dim is absent from
    /// a matched operand's layout.
    NoBundling,
    /// `cat = NO_BUNDLING`, `false` — a matched operand's coordinate has no fold for the row-split
    /// dim yet, so the group cannot be read off it.
    NotFolded,
    /// `true` WITH THE CALLER'S CATEGORY UNTOUCHED (`:8290`) — no node matched the datastream, which
    /// is why entry 337 sets the category before calling and calls this "can be overwritten".
    NoCandidates,
    /// The pruned group and its `commonGroupAncestor`. ⭐ NO CATEGORY: this unit never sets one on
    /// success, so the group cannot yet be a [`RowGroup`].
    Group {
        /// `nodeInfo` — the first node found for each row, with that row's `ROWSPLIT` offset.
        nodes: Vec<RowGroupNode>,
        /// `commonGroupAncestor`.
        common_ancestor: NodeId,
    },
}

/// Replaces: e297_gatherRelatedPTRowsBase
///
/// The first node on each PT row whose datastream is this propagation's, with that row's `ROWSPLIT`
/// offset, plus the block enclosing them all.
///
/// ⛔ THE TWO REFUSALS ARE READ IN OPPOSITE ORDERS — a transfer's layout before its one shared
/// `transferCoordinates_`, a compute's coordinate first — so the kind decides `NoBundling` vs
/// `NotFolded`; `None` is the four aborts, one of which (`nodeInfo.at(1)`, `:8341`) throws.
#[must_use]
pub fn gather_related_pt_rows_base<A, D, S, T, F>(
    dsc: &D,
    core_ds: &S,
    tree: &T,
    prop: &CoordPropInfo,
    scale_down: crate::schedule::ddc::ScaleDown,
    candidates: &[RowCandidate<'_, F>],
    sdsc_slices: &WorkSlices,
) -> Option<RelatedPtRows>
where
    A: Arch,
    D: Dsc + Allocations,
    S: Stage + ?Sized,
    T: ScheduleTree + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    let Some(row_split_dim) = PrimaryDim::ALL
        .into_iter()
        .find(|&dim| core_ds.is_row_split(dim))
    else {
        // No dimension is split across PT rows.
        return Some(RelatedPtRows::NoBundling);
    };

    // The node, its row, the fold manager for the row-split dim and the operand it came out of.
    let mut primary: Vec<(&RowCandidate<'_, F>, PtRowId, &F, &RowOperand<'_, F>)> = Vec::new();
    for candidate in candidates {
        let Some(row) = candidate.row else {
            // The node is not related to PT.
            continue;
        };
        let Some(operand) = candidate
            .operands
            .iter()
            .find(|operand| match_data_stream(dsc, prop, operand.stream, false))
        else {
            // Skip this scheduleNode.
            continue;
        };
        let in_layout =
            || layout_dims_from_node(dsc, candidate.node, operand.pos).contains(&row_split_dim);
        let folds = match candidate.node {
            Node::Transfer(_) => {
                if !in_layout() {
                    // Rowsplit dimension does not appear in the working node's layout.
                    return Some(RelatedPtRows::NoBundling);
                }
                match operand.coord.dims.get(&row_split_dim) {
                    Some(&folds) => folds,
                    None => return Some(RelatedPtRows::NotFolded),
                }
            }
            Node::Compute(_) => {
                let Some(&folds) = operand.coord.dims.get(&row_split_dim) else {
                    return Some(RelatedPtRows::NotFolded);
                };
                if !in_layout() {
                    // Rowsplit dimension does not appear in the working node's layout.
                    return Some(RelatedPtRows::NoBundling);
                }
                folds
            }
            // The reference's `if`/`else if` (`:8164`, `:8225`) has no arm for an allocation, so the
            // node the unfiltered walk offered is dropped.
            Node::Allocate(_) => continue,
        };
        // The node's output is consistent with the target group. Record it for further processing.
        primary.push((candidate, row, folds, operand));
    }

    if primary.is_empty() {
        return Some(RelatedPtRows::NoCandidates);
    }

    // Prune the candidate list to pick only the first node for each PT row.
    let mut nodes: Vec<RowGroupNode> = Vec::new();
    let mut per_row: BTreeMap<PtRowId, &RowCoordinate<'_, F>> = BTreeMap::new();
    for (candidate, row, folds, operand) in primary {
        if let Some(&recorded) = per_row.get(&row) {
            let recorded_side = CoordSide {
                dims: recorded.dims.clone(),
                work_slices: recorded.work_slices,
                cores: candidate.cores.clone(),
            };
            let operand_side = CoordSide {
                dims: operand.coord.dims.clone(),
                work_slices: operand.coord.work_slices,
                cores: candidate.cores.clone(),
            };
            if !same_coordinate_range(
                &recorded_side,
                &operand_side,
                sdsc_slices,
                // No need to check for broadcast dims.
                &BTreeSet::new(),
                DimCoverage::CommonDimsOnly,
            ) {
                // Mismatched coordinates found while constructing groups for PT rows.
                return None;
            }
            continue;
        }
        per_row.insert(row, operand.coord);
        let (_, mut beta) = folds.alpha_beta(FoldPosition::RowSplit.index());
        if scale_down == crate::schedule::ddc::ScaleDown::Yes {
            if let Some(mx) = operand.mx {
                if mx.dim == row_split_dim {
                    let block = i64::try_from(mx.blk_size.count().0).unwrap_or(i64::MAX);
                    beta = Beta(beta.0 / block);
                }
            }
        }
        nodes.push(RowGroupNode {
            node: candidate.id,
            row: Some(row),
            beta,
        });
    }

    // Find common ancestor. `nodeInfo.at(1)` throws on a one-node group, so a group of one never
    // reaches the count check below.
    let first = *nodes.first()?;
    let second = *nodes.get(1)?;
    let ref_path: Vec<NodeId> =
        core::iter::successors(tree.parent(first.node), |&walked| tree.parent(walked)).collect();
    let common_ancestor =
        core::iter::successors(tree.parent(second.node), |&walked| tree.parent(walked))
            // Row scheduleNodes are not enclosed in a common ancestor Node.
            .find(|walked| ref_path.contains(walked))?;

    // Only allow either one PTrow or all PTrows, nothing in between.
    if u32::try_from(nodes.len()).unwrap_or(u32::MAX) != A::PT_ROWS {
        return None;
    }
    Some(RelatedPtRows::Group {
        nodes,
        common_ancestor,
    })
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// PROPAGATING A FOLD ONTO A NODE — entries 298 and 299, and the four things they share.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHETHER A PROPAGATED FOLD WAS BUILT — what entries 298 and 299 do, which their `void` and `bool`
/// returns state only part of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldPropagation {
    /// The coordinate was extended for every propagated dim it did not already cover.
    Built,
    /// `coordinate.foldConstructed()` — entry 299's `false` (`:8652`), which is nothing to do.
    AlreadyConstructed,
    /// The corelet-split dim is propagated off an allocation with a custom `coreIdToWkSlice_`
    /// (`:8433-8440`).
    CoreletSplitVaries,
    /// No loop enclosing the row group's common ancestor walks the dim (`:8785-8789`).
    NoRowBundlingLoop,
    /// ⛔ ONE ABSENCE FOR EVERY `.at()` THROW AND UNCHECKED DIVIDE THESE TWO REACH: a fold list with
    /// no `ROWSPLIT` or `CORELET` slot, entry 239's three aborts, an absent or zero pad stride, a
    /// loop the distributor recorded nothing for, and a loop chain too short for the levels a
    /// transfer size demands be redistributed.
    ReferenceAborts,
}

/// ONE FOLD LEVEL CARRYING THE OPEN LABEL THE CONSTRUCTION TAIL READS BACK OFF IT.
///
/// ⛔ [`FoldParamInfo::label`] CANNOT SERVE: it is the CLOSED [`FoldLabel`] set, and these two units
/// write `"elem_arr_<j>"`, `"<loop> <dim>"` and `"corelet_fold"` INTO the list and read them at the
/// add site. Which is also why [`re_add_rebuilt_folds`] is not the tail here — it passes the closed
/// label through instead.
struct PropagatedLevel {
    level: FoldParamInfo,
    label: String,
}

/// A level straight out of [`gather_fold_params`], under the fold manager's own label.
fn propagated_level(level: FoldParamInfo) -> PropagatedLevel {
    PropagatedLevel {
        level,
        label: level
            .label
            .map(FoldLabel::spelling)
            .unwrap_or_default()
            .to_owned(),
    }
}

/// `for (i = size - 1; i > ends; --i) elemArr.push_back(at(i))` (`:8511-8513`, `:8854-8856`) — the
/// levels inside `ends`, INNERMOST FIRST, as the distributor takes them.
fn levels_inside(levels: &[PropagatedLevel], ends: i64) -> Vec<FoldParamInfo> {
    let inside = usize::try_from(ends.saturating_add(1)).unwrap_or(0);
    levels
        .get(inside..)
        .unwrap_or_default()
        .iter()
        .rev()
        .map(|level| level.level)
        .collect()
}

/// `foldParams.erase(begin + ends + 1, end)` then the redistribution's own levels appended
/// (`:8525-8536`, `:8908-8918`) — the distributor returns them OUTERMOST FIRST and `elem_arr_<j>`
/// counts from the INNERMOST, so the walk is reversed and the index is `j`.
fn replace_elem_arr(levels: &mut Vec<PropagatedLevel>, ends: i64, after: &[FoldParamInfo]) {
    levels.truncate(usize::try_from(ends.saturating_add(1)).unwrap_or(0));
    for (from_inner, level) in after.iter().enumerate().rev() {
        levels.push(PropagatedLevel {
            level: *level,
            label: format!("elem_arr_{from_inner}"),
        });
    }
}

/// The construction tail both units end each dim with (`:8590-8605`, `:8957-8972`): every level
/// re-added at the FRONT so the list lands outermost first, categorised by where it sits, and the
/// element arrangement RELABELLED by its distance from the innermost level.
fn add_propagated_folds(
    coord: &mut dsc2::Coordinate,
    dim: PrimaryDim,
    levels: &[PropagatedLevel],
    spatial_ends: i64,
    temporal_ends: i64,
) {
    for (at, level) in levels.iter().enumerate().rev() {
        let pos = i64::try_from(at).unwrap_or(i64::MAX);
        let (category, label) = if pos > temporal_ends {
            let from_inner = levels.len() - 1 - at;
            (
                CoordinateCategory::ElemArr,
                format!("elem_arr_{from_inner}"),
            )
        } else if pos > spatial_ends {
            (CoordinateCategory::Temporal, level.label.clone())
        } else {
            (CoordinateCategory::Spatial, level.label.clone())
        };
        add_labelled_fold(coord, dim, category, label, level.level);
    }
}

/// THE TWO `dsc/dsc2.cpp` QUESTIONS THESE UNITS ASK THAT [`TemporalLoopDistribution`] DOES NOT —
/// likewise outside this campaign's file list, so likewise seams and not ports.
pub trait LoopRelevance: TemporalLoopDistribution {
    /// `dsc2::loopRelevantForDim(currDsc, dim, loopNode, accessPadType)` —
    /// [`TemporalLoopDistribution::related_loops`]'s per-loop kernel, which entry 299 applies to a
    /// chain it walks itself.
    fn loop_relevant_for_dim(&self, dim: PrimaryDim, loop_node: &LoopNode, pad: PadType) -> bool;

    /// `loopParamsAfterDistribution.at(nullptr).at(dim)` (`:8549-8552`) — what corelet-slicing
    /// distribution decided, which is recorded under NO loop at all.
    fn distributed_corelet_slice(
        &self,
        loop_params: &Self::LoopParams,
        dim: PrimaryDim,
    ) -> Option<DistributedLoop>;
}

/// THE CORE STAGE PLUS THE ONE QUESTION THESE UNITS PUT TO A TEMPORAL LOOP.
pub trait PropagatedFoldStage: CoreStage + Stage {
    /// How many times one temporal loop trips for a dim: `parametricIterCount(currDsc, 0,
    /// NO_COMPONENT, ptRowId)` on a parametric loop, else `numDs / denDs` read through
    /// `dataStageDimToVal_compView_st(dim, propRefComp, 0)` (`:8553-8571`, `:8925-8946`).
    ///
    /// ⛔ NOT [`ExternalFoldStage::loop_trip`], WHICH CARRIES NEITHER THE PROPAGATION COMPONENT NOR
    /// THE PT ROW — and both of those choose the number here.
    /// ⚠️ The staged arm's divide by zero lives behind this seam, as `loop_trip`'s does.
    fn loop_iterations(
        &self,
        loop_node: &LoopNode,
        dim: PrimaryDimAndKind,
        prop: SenComponent,
        row: Option<PtRowId>,
    ) -> Cardinality;
}

/// `coreIdToWkSliceVariesOnDim` (`:8408-8428`) — whether two cores that BOTH record a work slice for
/// the dim record different ones. A core that records none is skipped, not counted as a difference.
fn work_slice_varies_on_dim(slices: &WorkSlices, dim: PrimaryDim) -> bool {
    let mut first: Option<WorkSlice> = None;
    for per_dim in slices.0.values() {
        let Some(&slice) = per_dim.get(&dim) else {
            continue;
        };
        match first {
            None => first = Some(slice),
            Some(recorded) if recorded != slice => return true,
            Some(_) => {}
        }
    }
    false
}

/// THE REFERENCE ALLOCATION AS ENTRY 298 READS IT — [`BaseAllocation`] over EVERY dim it propagates,
/// because this unit loops over `effectiveRefCoord.coordinates_` and that one names a single dim.
///
/// ⛔ BOTH `DT_CHECK_MSG`s ARE DISCHARGED BY TYPE, as [`BaseAllocation`]'s are: that the reference
/// node IS an `ALLOCATE` (`:8373`), and that its `ldsIdx_ >= 0` (`:8405`).
pub struct AllocationFold<'a, F: AffineFoldDims + ?Sized> {
    /// `ldsIdx_`.
    pub lds: LdsIdx,
    /// `effectiveRefCoord.coordinates_` ∩ `dimsToPropagate`, each dim with BOTH of the allocation's
    /// coordinates — [`AllocCoordinates::effective`] is this unit's own choice (`:8455-8459`).
    pub dims: Vec<(PrimaryDim, AllocCoordinates<'a, F>)>,
    /// `allocateCoordinates_.coreIdToWkSlice_`.
    pub work_slices: &'a WorkSlices,
    /// `labeledDs_.at(ldsIdx_).scale_` per dim, absent where the layout order does not name the dim —
    /// which the reference reads as a scale of 1 (`:8477`).
    pub dim_scales: BTreeMap<PrimaryDim, Scale>,
    /// `coreletSplitDim`, absent for its `PrimaryDimTypesCount` (`:8432`).
    pub corelet_split_dim: Option<PrimaryDim>,
    /// `sizeRefComp` and `propRefComp`, which this unit receives rather than derives.
    pub components: RefComponents,
    /// `refRowGroup`.
    pub row_group: Option<&'a RowGroup>,
}

/// THE NODE ENTRY 298 BUILDS THE FOLD FOR — `node` and the loops around it.
pub struct AllocFoldTarget<'a, 'l> {
    /// The node the distributor records its loop params against.
    pub node: NodeId,
    /// `loopChain` from `getEnclosingLoopsAndRelatedDims(currDsc, node, .., buildCoreletFold)`
    /// (`:8444-8450`), INNERMOST FIRST — `isExternalNode` and the corelet-fold request are the
    /// caller's, because together they only decide whether the chain carries a `CORELET_SLICE` entry.
    pub loops: &'a [LoopAndDim<'l>],
}

/// Replaces: e298_buildFoldFromAllocation
///
/// Propagates an allocation's coordinate onto a node: per dim its own levels with the row split
/// recomputed, then a temporal level per surplus enclosing loop with the arrangement redistributed.
///
/// ⛔ THE `covers` SKIP IS THE LAST THING CHECKED (`:8581`), NOT THE FIRST — the distribution above it
/// has already recorded loop params that a later dim of the same call reads back.
/// ⚠️ The corelet arm decrements the loop differential (`:8554`): a corelet slice OVERWRITES a spatial.
#[must_use]
pub fn build_fold_from_allocation<A, D, S, T, F>(
    dsc: &D,
    core_ds: &S,
    distribution: &T,
    alloc: &AllocationFold<'_, F>,
    target: &AllocFoldTarget<'_, '_>,
    coord: &mut dsc2::Coordinate,
    loop_params: &mut T::LoopParams,
) -> FoldPropagation
where
    A: Arch,
    D: Dsc,
    S: PropagatedFoldStage + ?Sized,
    T: LoopRelevance + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    if let Some(split) = alloc.corelet_split_dim {
        if alloc.dims.iter().any(|&(dim, _)| dim == split)
            && work_slice_varies_on_dim(alloc.work_slices, split)
        {
            // Can not propagate coordinates for a coreletSplit dimension from an allocateNode with a
            // custom coreIdToWkSlice.
            return FoldPropagation::CoreletSplitVaries;
        }
    }

    let row = comp_row_id(alloc.components.size);
    let scale_is_non_broadcast = |scale| matches!(scale, Scale::Sized(scale) if scale > 0.0);
    for (dim, coordinates) in &alloc.dims {
        let dim = *dim;
        let effective = coordinates.effective(row);
        let mut fold_params: Vec<PropagatedLevel> = effective
            .folds
            .map(gather_fold_params)
            .unwrap_or_default()
            .into_iter()
            .map(propagated_level)
            .collect();

        // Propagation from an allocation does not involve row-to-nonrow bundling. It may involve
        // nonrow-to-row UN-bundling, which the rowsplit fold captures directly.
        let Some(mut row_split) =
            compute_params_for_row_split_fold::<A, _>(core_ds, dim, alloc.row_group)
        else {
            return FoldPropagation::ReferenceAborts;
        };
        // Experimental, and the reference says so: this belongs inside entry 239.
        if effective.pad != PadType::NoPad && coord.padding(dim) == PadType::NoPad {
            // Divide by the stride.
            let Some(stride) = core_ds.pad_stride(dim).map(Stride::get) else {
                return FoldPropagation::ReferenceAborts;
            };
            row_split.alpha = Alpha(row_split.alpha.0 / stride);
            row_split.beta = Beta(row_split.beta.0 / stride);
        }
        match fold_params.get_mut(FoldPosition::RowSplit.index()) {
            Some(slot) => *slot = propagated_level(row_split),
            None => return FoldPropagation::ReferenceAborts,
        }

        // Note: dim is of type PrimaryDimTypes, metaDimKind is missing here.
        let curr_dim = PrimaryDimAndKind {
            dim,
            kind: MetaDimKind::Unpadded,
        };
        let related_loops = if alloc
            .dim_scales
            .get(&dim)
            .copied()
            .is_none_or(scale_is_non_broadcast)
        {
            // The dimension is a non-broadcast dimension.
            distribution.related_loops(curr_dim, target.loops, coord.padding(dim))
        } else {
            Vec::new()
        };

        let spatial_ends = i64::from(effective.spatial.0) - 1;
        let mut temporal_ends = spatial_ends + i64::from(effective.temporal.0);
        // The node has more enclosing loops than the reference allocateNode: distribute the
        // additional loops over the element arrangement levels.
        let surplus = related_loops
            .len()
            .saturating_sub(effective.temporal.0 as usize);
        if surplus > 0 {
            // Scan order is from the innermost loop towards the outermost loop.
            let to_distribute = related_loops.get(..surplus).unwrap_or_default().to_vec();
            let after = distribution.distribute(
                &ElemArrDistribution {
                    dim: curr_dim,
                    fold_owner: target.node,
                    target_lds: alloc.lds,
                    ref_pad: effective.pad,
                    target_pad: coord.padding(dim),
                    components: alloc.components,
                    loops_to_distribute: to_distribute.clone(),
                    elem_arr: levels_inside(&fold_params, temporal_ends),
                },
                loop_params,
            );
            // Distribution may remove some original element arrangement levels.
            replace_elem_arr(&mut fold_params, temporal_ends, &after);

            // The insertion point is the beginning of the element arrangement folds, and it does NOT
            // advance: each inserted level pushes the previous one further in, so the innermost loop
            // of the scan ends up innermost.
            let at = usize::try_from(temporal_ends.saturating_add(1)).unwrap_or(0);
            let fold_dim_str = dim.spelling();
            let mut surplus = i64::try_from(surplus).unwrap_or(i64::MAX);
            for related in &to_distribute {
                if related.distribution == LoopDistribution::CoreletSlice {
                    let Some(decided) = distribution.distributed_corelet_slice(loop_params, dim)
                    else {
                        return FoldPropagation::ReferenceAborts;
                    };
                    let corelet = PropagatedLevel {
                        level: FoldParamInfo {
                            alpha: decided.alpha,
                            beta: decided.beta,
                            cardinality: core_ds.corelets_used(),
                            label: None,
                        },
                        label: "corelet_fold".to_owned(),
                    };
                    match fold_params.get_mut(FoldPosition::Corelet.index()) {
                        Some(slot) => *slot = corelet,
                        None => return FoldPropagation::ReferenceAborts,
                    }
                    surplus -= 1;
                    continue;
                }
                let Some(decided) = distribution.distributed(loop_params, related.loop_node, dim)
                else {
                    return FoldPropagation::ReferenceAborts;
                };
                fold_params.insert(
                    at.min(fold_params.len()),
                    PropagatedLevel {
                        level: FoldParamInfo {
                            alpha: decided.alpha,
                            beta: decided.beta,
                            cardinality: core_ds.loop_iterations(
                                related.loop_node,
                                related.dim,
                                alloc.components.prop,
                                None,
                            ),
                            label: None,
                        },
                        label: format!("{} {fold_dim_str}", related.loop_node.name.0),
                    },
                );
            }
            temporal_ends += surplus;
        }

        if coord.covers(dim) {
            // The scheduleNode's coordinates already include a fold for this dimension.
            continue;
        }
        add_propagated_folds(coord, dim, &fold_params, spatial_ends, temporal_ends);
    }
    if all_dims_covered(dsc, coord, alloc.lds) {
        coord.complete_fold_construction();
    }
    FoldPropagation::Built
}

/// WHICH KIND THE FOLD IS BEING BUILT FOR — `nodeToFold`'s two `DT_ERROR`-checked kinds
/// (`:8644-8651`), reduced to the one thing the kind decides.
pub enum FoldTargetKind<'a> {
    /// `TRANSFER` — its `transferSize_`, which may demand more element arrangement than the
    /// reference's own folds hold (`:8859`).
    Transfer(&'a BTreeMap<PrimaryDim, Cardinality>),
    /// `COMPUTE` — nothing to override.
    Compute,
}

/// THE REFERENCE SIDE OF A NON-ALLOCATION PROPAGATION — `refNode` and `refCoordinate` (`:8629-8631`).
///
/// ⛔ THE FIRST `DT_ERROR` IS DISCHARGED BY CONSTRUCTION (`:8637-8643`): only a transfer or a compute
/// has an enclosing loop chain and a coordinate to hand over, so a reference of any other kind cannot
/// be spelled here.
pub struct NonAllocReference<'a, 'l, F: AffineFoldDims + ?Sized> {
    /// `refLdsIdx`.
    pub lds: LdsIdx,
    /// `dimList` ∩ `dimsToPropagate`, each with its counts and padding — `foldSingleDim`
    /// (`:8668-8677`) is the caller handing over exactly one entry.
    pub dims: Vec<(PrimaryDim, CoordDim<'a, F>)>,
    /// `refLds.scale_` per dim, absent where the layout order does not name the dim.
    pub dim_scales: BTreeMap<PrimaryDim, Scale>,
    /// `refNode`'s enclosing loops, INNERMOST FIRST and UNFILTERED — this unit filters them itself,
    /// per dim, under the reference coordinate's own padding.
    pub loops: &'a [&'l LoopNode],
    /// `sizeRefComp` and `propRefComp`.
    pub components: RefComponents,
    /// `refRowGroup`.
    pub row_group: Option<&'a RowGroup>,
}

/// THE NODE ENTRY 299 BUILDS THE FOLD FOR — `nodeToFold` (`:8635`).
pub struct NonAllocTarget<'a, 'l> {
    /// The node the distributor records its loop params against.
    pub node: NodeId,
    /// Its kind.
    pub kind: FoldTargetKind<'a>,
    /// Its enclosing loops with their `loopsBelowChunkBoundary` tags, INNERMOST FIRST and unfiltered.
    pub loops: &'a [(&'l LoopNode, LoopDistribution)],
    /// `refRowGroup.commonGroupAncestor`'s own enclosing chain, THE ANCESTOR ITSELF FIRST when it is
    /// a loop (`:8768-8778`) — empty where the group has no common ancestor at all.
    pub row_bundling: &'a [&'l LoopNode],
}

/// Replaces: e299_buildFoldFromNonAllocRef
///
/// Propagates a transfer's or compute's coordinate onto another such node: the reference's levels,
/// the row group's bundling, and the working node's unshared loops distributed over what is left.
///
/// ⛔ "UNSHARED" IS THE COMMON ANCESTOR'S POSITION IN THE REFERENCE'S CHAIN; its `DT_ERROR` (`:8737`)
/// is tautological, and `back()` on an EMPTY chain (`:8724`) reads past the end — no position at all.
/// ⚠️ `refElemArrFoldCount` is written three times and never read (`:8748`, `:8790`, `:8823`).
#[must_use]
pub fn build_fold_from_non_alloc_ref<A, D, S, T, F>(
    dsc: &D,
    core_ds: &S,
    distribution: &T,
    reference: &NonAllocReference<'_, '_, F>,
    target: &NonAllocTarget<'_, '_>,
    coord: &mut dsc2::Coordinate,
    loop_params: &mut T::LoopParams,
) -> FoldPropagation
where
    A: Arch,
    D: Dsc,
    S: PropagatedFoldStage + ?Sized,
    T: LoopRelevance + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    if coord.fold_constructed() {
        return FoldPropagation::AlreadyConstructed;
    }
    let scale_is_non_broadcast = |scale| matches!(scale, Scale::Sized(scale) if scale > 0.0);
    for (dim, ref_dim) in &reference.dims {
        let dim = *dim;
        if coord.covers(dim) {
            // The scheduleNode's coordinates already include a fold for this dimension.
            continue;
        }
        let target_pad = coord.padding(dim);
        let relevant =
            |loop_node: &LoopNode, pad| distribution.loop_relevant_for_dim(dim, loop_node, pad);

        // Find the common ancestor.
        let mut ref_chain: Vec<&LoopNode> = Vec::new();
        let mut working_chain: Vec<(&LoopNode, LoopDistribution)> = Vec::new();
        let mut common_ancestor_pos: Option<usize> = None;
        let non_broadcast = reference
            .dim_scales
            .get(&dim)
            .copied()
            .is_none_or(scale_is_non_broadcast);
        if non_broadcast {
            // The dimension is a non-broadcast dimension.
            ref_chain = reference
                .loops
                .iter()
                .copied()
                .filter(|&loop_node| relevant(loop_node, ref_dim.pad))
                .collect();
            for &(loop_node, tag) in target.loops {
                if relevant(loop_node, target_pad) {
                    working_chain.push((loop_node, tag));
                }
                if ref_chain
                    .iter()
                    .any(|&walked| core::ptr::eq(walked, loop_node))
                {
                    break;
                }
            }
            common_ancestor_pos = working_chain.last().and_then(|&(innermost, _)| {
                ref_chain
                    .iter()
                    .position(|&walked| core::ptr::eq(walked, innermost))
            });
        }

        // Build the fold.
        let mut fold_params: Vec<PropagatedLevel> = ref_dim
            .folds
            .map(gather_fold_params)
            .unwrap_or_default()
            .into_iter()
            .map(propagated_level)
            .collect();
        let ref_spatial = i64::from(ref_dim.spatial.0);
        let mut ref_temporal = i64::from(ref_dim.temporal.0);
        let Some(row_split) =
            compute_params_for_row_split_fold::<A, _>(core_ds, dim, reference.row_group)
        else {
            return FoldPropagation::ReferenceAborts;
        };

        if core_ds.is_row_split(dim) {
            match reference.row_group.map(RowGroup::category) {
                Some(RowGroupCategory::RowToSameRow) => {
                    match fold_params.get_mut(FoldPosition::RowSplit.index()) {
                        Some(slot) => *slot = propagated_level(row_split),
                        None => return FoldPropagation::ReferenceAborts,
                    }
                }
                Some(RowGroupCategory::RowToNonRow) => {
                    // Bundle coordinates from PT rows. 1. The spatial fold for PT rows becomes a noop.
                    match fold_params.get_mut(FoldPosition::RowSplit.index()) {
                        Some(slot) => *slot = propagated_level(default_row_split_fold()),
                        None => return FoldPropagation::ReferenceAborts,
                    }
                    // 2. Place the row fold after the innermost common loop holding the reference node
                    // and its row siblings.
                    let Some(ancestor) = target
                        .row_bundling
                        .iter()
                        .copied()
                        .find(|&loop_node| relevant(loop_node, target_pad))
                    else {
                        return FoldPropagation::NoRowBundlingLoop;
                    };
                    if let Some(at) = ref_chain
                        .iter()
                        .position(|&walked| core::ptr::eq(walked, ancestor))
                    {
                        // A virtual temporal fold immediately inside that ancestor's own temporal fold.
                        let at = ref_spatial + ref_temporal - i64::try_from(at).unwrap_or(i64::MAX);
                        let at = usize::try_from(at).unwrap_or(0).min(fold_params.len());
                        fold_params.insert(at, propagated_level(row_split));
                    }
                }
                Some(RowGroupCategory::NonRowToRow) => {
                    // Read every temporal and element arrangement fold of the refNode as element
                    // arrangement, and distribute every loop enclosing the nodeToFold.
                    common_ancestor_pos = ref_chain.len().checked_sub(1);
                    working_chain = target
                        .loops
                        .iter()
                        .copied()
                        .filter(|&(loop_node, _)| relevant(loop_node, target_pad))
                        .collect();
                }
                Some(RowGroupCategory::RowNorthSouth | RowGroupCategory::NoBundling) | None => {}
            }
        }

        if let Some(at) = common_ancestor_pos.filter(|&at| at > 0) {
            // Some innermost loops enclosing the reference node are absent from the working node's
            // loop nest. Read the diff down to the common ancestor — which is itself excluded — as
            // part of the reference's element arrangement.
            ref_temporal -= i64::try_from(at).unwrap_or(i64::MAX);
        }

        let mut ref_temporal_ends = ref_spatial + ref_temporal - 1;
        let working_spatial_ends = ref_spatial - 1;
        let mut working_temporal_ends = working_spatial_ends + ref_temporal;
        if non_broadcast {
            working_temporal_ends += i64::try_from(working_chain.len()).unwrap_or(i64::MAX) - 1;
        }

        let curr_dim = PrimaryDimAndKind {
            dim,
            kind: MetaDimKind::Unpadded,
        };
        let mut to_distribute: Vec<LoopAndDim<'_>> = Vec::new();
        let mut elem_arr: Vec<FoldParamInfo> = Vec::new();
        let mut needs_distribution = false;
        if working_chain.len() > 1 {
            // There are loops between the common ancestor and the working node. Distribute them over
            // the reference element arrangement from the previous step; the common ancestor is not one.
            needs_distribution = true;
            to_distribute = working_chain
                .iter()
                .map(|&(loop_node, distribution)| LoopAndDim {
                    loop_node,
                    dim: curr_dim,
                    distribution,
                })
                .collect();
            to_distribute.pop();
            elem_arr = levels_inside(&fold_params, ref_temporal_ends);
        } else if let FoldTargetKind::Transfer(sizes) = target.kind {
            if let Some(&size) = sizes.get(&dim) {
                // Reference and working node share the loop nest. Check whether overriding the
                // transfer size needs re-distribution over some of those loops.
                let mut elem_arr_card: u64 = 1;
                let mut accumulated: u64 = 1;
                let mut size_pos: Option<usize> = None;
                for at in (usize::try_from(ref_spatial).unwrap_or(0)..fold_params.len()).rev() {
                    let card = fold_params[at].level.cardinality.0;
                    accumulated = accumulated.saturating_mul(card);
                    if i64::try_from(at).unwrap_or(i64::MAX) > ref_temporal_ends {
                        elem_arr_card = elem_arr_card.saturating_mul(card);
                    }
                    if accumulated >= size.0 {
                        size_pos = Some(at);
                        break;
                    }
                }
                if elem_arr_card < size.0 {
                    needs_distribution = true;
                    // ⛔ NO POSITION IS THE REFERENCE'S `-1`, and the chain slice below is what
                    // `loopsToWorkingNode.at(i)` throws on when it asks for more loops than there are.
                    let pos = size_pos.map_or(-1, |at| i64::try_from(at).unwrap_or(i64::MAX));
                    let wanted = usize::try_from(ref_temporal_ends - pos + 1).unwrap_or(0);
                    let Some(chain) = working_chain.get(..wanted) else {
                        return FoldPropagation::ReferenceAborts;
                    };
                    to_distribute = chain
                        .iter()
                        .map(|&(loop_node, _)| LoopAndDim {
                            loop_node,
                            dim: curr_dim,
                            distribution: LoopDistribution::BelowChunk,
                        })
                        .collect();
                    elem_arr = levels_inside(&fold_params, pos - 1);
                    // Re-distribution happens from the level immediately after refTemporalFoldEnds.
                    ref_temporal_ends = pos - 1;
                }
            }
        }

        if needs_distribution {
            let after = distribution.distribute(
                &ElemArrDistribution {
                    dim: curr_dim,
                    fold_owner: target.node,
                    target_lds: reference.lds,
                    ref_pad: ref_dim.pad,
                    target_pad,
                    components: reference.components,
                    loops_to_distribute: to_distribute.clone(),
                    elem_arr,
                },
                loop_params,
            );
            // Distribution may remove some original element arrangement levels.
            replace_elem_arr(&mut fold_params, ref_temporal_ends, &after);

            // The working node's additional temporal folds go before the element arrangement, at an
            // insertion point that does not advance.
            let at = usize::try_from(ref_temporal_ends.saturating_add(1)).unwrap_or(0);
            let fold_dim_str = dim.spelling();
            let row = if core_ds.is_row_split(dim) {
                comp_row_id(reference.components.size)
            } else {
                None
            };
            for related in &to_distribute {
                let Some(decided) = distribution.distributed(loop_params, related.loop_node, dim)
                else {
                    return FoldPropagation::ReferenceAborts;
                };
                fold_params.insert(
                    at.min(fold_params.len()),
                    PropagatedLevel {
                        level: FoldParamInfo {
                            alpha: decided.alpha,
                            beta: decided.beta,
                            cardinality: core_ds.loop_iterations(
                                related.loop_node,
                                related.dim,
                                reference.components.prop,
                                row,
                            ),
                            label: None,
                        },
                        label: format!("{} {fold_dim_str}", related.loop_node.name.0),
                    },
                );
            }
        }

        add_propagated_folds(
            coord,
            dim,
            &fold_params,
            working_spatial_ends,
            working_temporal_ends,
        );
    }
    if all_dims_covered(dsc, coord, reference.lds) {
        coord.complete_fold_construction();
    }
    FoldPropagation::Built
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// WHICH PT ROWS ONE PROPAGATION SPANS — entry 337 and the vocabulary it reads its two units off.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE OPERAND AS ENTRY 337 READS IT — an `inputs_`/`outputs_` component beside the
/// `inputsLdsAndLoopOffsets_`/`outputsLdsAndLoopOffsets_` entry the reference indexes in lockstep
/// with it (`ddc/ddc_fold.cpp:1046-1054`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitStream {
    /// `inputs_.at(i)` / `outputs_.at(i)` — a UNIT, not a storage.
    pub unit: SenComponent,
    /// `..LdsAndLoopOffsets_.at(i)`, in that operand's storage.
    pub stream: StoredStream,
}

/// ONE END OF ENTRY 337'S PROPAGATION — `refNode`/`nodeToFold`'s `nodeType_` with the units it
/// reads off each kind. [`PropEnd`] is the same pair as [`match_data_stream`] needs them; this is
/// the same pair as the ROW question needs them.
pub enum PropUnits<'a> {
    /// `TRANSFER` — `src_.unit_`, and `dstVias_` zipped with `dstLdsAndLoopOffsets_`.
    Transfer {
        /// `src_.unit_`.
        src: SenComponent,
        /// `dstVias_`, non-empty so neither `dstVias_.at(0)` read can throw.
        dsts: &'a TransferDsts,
    },
    /// `COMPUTE` — `exUnit_` and both operand vectors.
    Compute {
        /// `exUnit_`.
        ex_unit: SenComponent,
        /// `inputs_`.
        inputs: &'a [UnitStream],
        /// `outputs_`.
        outputs: &'a [UnitStream],
    },
    /// `ALLOCATE` — `component_`, the only field this unit reads off one.
    Allocate {
        /// `component_`.
        component: SenComponent,
    },
    /// Any other `nodeType_`.
    Other,
}

/// ENTRY 337'S PROPAGATION WITH BOTH PROPAGATION UNITS RESOLVED — `propSrcUnit` and `propDestUnit`.
///
/// ⛔⛔ EVERY ABORT OF THE 100-LINE RESOLUTION IS THIS WITNESS MISSING: all five
/// `DT_CHECK(prop*Unit != NO_COMPONENT)` (`:1010`, `:1060`, `:1084`, `:1105`, `:1156`), the
/// `abs(refRowId - getCompRowId(propDestUnit)) <= 1` check (`:1184`), and *"[gatherRelatedPTRows]
/// Unsupported reference node for propagation"* for an allocation folded against neither a transfer
/// nor a compute (`:1162`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtRowPropagation {
    src: SenComponent,
    dest: SenComponent,
}

impl PtRowPropagation {
    /// Both units, or [`None`] for any of those seven stops.
    ///
    /// ⚠️ TRAP: the two ends are read with OPPOSITE `checkRefForAllocate` — `true` where the node to
    /// fold is the transfer or compute, `false` where it is the allocation — because the flag names
    /// which end of the propagation the allocation is.
    #[must_use]
    pub fn of<A: Allocations + ?Sized>(
        dsc: &A,
        prop: &CoordPropInfo,
        role: RefRole,
        ref_node: &PropUnits<'_>,
        node_to_fold: &PropUnits<'_>,
    ) -> Option<Self> {
        let none = SenComponent::NoComponent;
        let matched = |operands: &[UnitStream], check_ref: bool| {
            operands
                .iter()
                .find(|operand| match_data_stream(dsc, prop, operand.stream, check_ref))
                .map(|operand| operand.unit)
        };
        let matched_dst = |dsts: &TransferDsts, check_ref: bool| {
            dsts.iter()
                .find(|dst| match_data_stream(dsc, prop, dst.stream, check_ref))
                .map(|dst| dst.unit)
        };
        let mut src = none;
        let mut dest = none;
        match *node_to_fold {
            PropUnits::Transfer { src: from, dsts } => {
                // The destination unit for propagation is the transfer node's own; where neither end
                // of it is a row unit that stays the source.
                dest = from;
                if comp_row_id(dest).is_none() {
                    if let Some(row) = dsts.iter().find(|dst| comp_row_id(dst.unit).is_some()) {
                        dest = row.unit;
                    }
                }
                src = match role {
                    RefRole::Producer => from,
                    RefRole::Consumer => matched_dst(dsts, true)?,
                };
            }
            PropUnits::Compute {
                ex_unit,
                inputs,
                outputs,
            } => {
                dest = ex_unit;
                match role {
                    // A compute's `inputs_` names the GENERIC unit (`l0lu`), so where the reference
                    // node is a transfer the row-specific unit comes off its matching destination.
                    RefRole::Producer => {
                        src = match *ref_node {
                            PropUnits::Transfer { dsts, .. } => matched_dst(dsts, true)?,
                            _ => matched(inputs, true)?,
                        };
                    }
                    RefRole::Consumer => {
                        src = matched(outputs, true).unwrap_or(none);
                        // Override the comp in case the transfer is for a single PT row: for LXLU to
                        // PTRow4 the coordinate is for row 4 only.
                        if let PropUnits::Transfer { src: from, dsts } = *ref_node {
                            if comp_row_id(src).is_none() {
                                if comp_row_id(from).is_some() {
                                    src = from;
                                } else if comp_row_id(dsts.first().unit).is_some() {
                                    src = dsts.first().unit;
                                }
                            }
                        }
                        if src == none {
                            return None;
                        }
                    }
                }
            }
            PropUnits::Allocate { component } => match *ref_node {
                PropUnits::Transfer { src: from, dsts } => {
                    match role {
                        RefRole::Producer => {
                            dest = matched_dst(dsts, false)?;
                            src = dest;
                        }
                        RefRole::Consumer => {
                            src = dsts.first().unit;
                            if comp_row_id(src).is_none() {
                                if let Some(row) = dsts
                                    .iter()
                                    .skip(1)
                                    .find(|dst| comp_row_id(dst.unit).is_some())
                                {
                                    src = row.unit;
                                }
                            }
                            dest = from;
                        }
                    }
                    if comp_row_id(src).is_none() && comp_row_id(from).is_some() {
                        src = from;
                    }
                    // PTARF and PTXRF hold the allocation for ALL rows, so the allocation's own
                    // component outranks the transfer end.
                    if matches!(component, SenComponent::Ptarf | SenComponent::Ptxrf) {
                        dest = component;
                    }
                }
                PropUnits::Compute {
                    ex_unit,
                    inputs,
                    outputs,
                } => {
                    src = ex_unit;
                    dest = match role {
                        RefRole::Producer => matched(outputs, false)?,
                        RefRole::Consumer => matched(inputs, false)?,
                    };
                }
                // *"Unsupported reference node for propagation"*.
                PropUnits::Allocate { .. } | PropUnits::Other => return None,
            },
            // The reference's chain has no `else`, so both units stay `NO_COMPONENT`.
            PropUnits::Other => {}
        }
        if let (Some(src_row), Some(dest_row)) = (comp_row_id(src), comp_row_id(dest)) {
            if !src_row.is_adjacent_or_same(dest_row) {
                return None;
            }
        }
        Some(Self { src, dest })
    }
}

/// THE SAME NODES SCANNED FROM BOTH SIDES — entry 337's unbundling arm hands the base a propagation
/// with `refIsProducer` NEGATED and both ends swapped (`:12053-12057`), and the role is exactly what
/// decides which side of a candidate is scanned.
pub struct RowCandidates<'a, F: AffineFoldDims + ?Sized> {
    /// Scanned for the propagation as given.
    pub forward: &'a [RowCandidate<'a, F>],
    /// Scanned for the reversed propagation.
    pub reverse: &'a [RowCandidate<'a, F>],
}

/// WHAT ENTRY 337 FOUND — the `RowGroupInfo` it writes through its `&` fused with its `bool`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtRowGrouping {
    /// `cat = NO_BUNDLING` AND NOTHING ELSE WRITTEN — neither unit is on a PT row, or no dim is row
    /// split. ⭐ The caller's `nodeInfo` is left standing, which is why this is not an empty group.
    NoBundling,
    /// `cat = ROW_TO_SAME_ROW`, `activeRow` and the ONE `nodeInfo` entry this unit pushes itself.
    SameRow(PtRowId, RowGroupNode),
    /// The category set BEFORE the base call, and what the base then found —
    /// [`RelatedPtRows::NoBundling`] overwrites it and every other variant leaves it standing, which
    /// is what the reference's *"Category can be overwritten by the base function"* says.
    Bundled(RowGroupCategory, RelatedPtRows),
}

/// The base's answer as this unit's. Its [`RelatedPtRows::NotFolded`] IS its own `false`, so it
/// propagates as this unit's `false` and never reaches a category.
fn bundled(cat: RowGroupCategory, found: Option<RelatedPtRows>) -> Option<PtRowGrouping> {
    match found? {
        RelatedPtRows::NotFolded => None,
        other => Some(PtRowGrouping::Bundled(cat, other)),
    }
}

/// Replaces: e337_gatherRelatedPTRows
///
/// WHICH PT ROWS ONE COORDINATE PROPAGATION SPANS, from the two units it runs between: no bundling,
/// one row, or a group the base collects (`ddc/ddc_fold.cpp:959`).
///
/// ⛔ THE ROW-TO-SAME-ROW ARMS ARE THE ONLY ONES THIS UNIT ANSWERS ITSELF; the other three set a
/// category and hand the work to entry 297, whose own answer may overwrite it.
/// ⚠️ TRAP: PTXRF/PTARF TO A ROW UNIT IS `ROW_TO_SAME_ROW` AND TAKES `alpha * row` UNCONDITIONALLY,
/// where the row-to-same-row arm proper takes `beta` unless the rowsplit fold steps more than once.
/// ⚠️ `refCoordinate.coordinates_.at(rowSplitDim)` throws where the reference coordinate has no fold
/// for that dim, and both single-row arms read it; that is one of this [`None`].
#[must_use]
pub fn gather_related_pt_rows<A, D, S, T, F>(
    dsc: &D,
    core_ds: &S,
    tree: &T,
    prop: &CoordPropInfo,
    units: PtRowPropagation,
    ref_node: NodeId,
    ref_coord: &RowCoordinate<'_, F>,
    scale_down: crate::schedule::ddc::ScaleDown,
    candidates: &RowCandidates<'_, F>,
    sdsc_slices: &WorkSlices,
) -> Option<PtRowGrouping>
where
    A: Arch,
    D: Dsc + Allocations,
    S: Stage + ?Sized,
    T: ScheduleTree + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    let src_row = comp_row_id(units.src);
    let dest_row = comp_row_id(units.dest);
    if src_row.is_none() && dest_row.is_none() {
        // Neither reference nor working node corresponds to a PT row.
        return Some(PtRowGrouping::NoBundling);
    }
    let Some(row_split_dim) = PrimaryDim::ALL
        .into_iter()
        .find(|&dim| core_ds.is_row_split(dim))
    else {
        // No dimension is split across PT rows.
        return Some(PtRowGrouping::NoBundling);
    };
    let base = |prop: &CoordPropInfo, candidates: &[RowCandidate<'_, F>]| {
        gather_related_pt_rows_base::<A, D, S, T, F>(
            dsc, core_ds, tree, prop, scale_down, candidates, sdsc_slices,
        )
    };
    let single_row = |row: PtRowId, scaled: bool| {
        let folds = *ref_coord.dims.get(&row_split_dim)?;
        let (alpha, beta) = folds.alpha_beta(FoldPosition::RowSplit.index());
        let beta = if scaled || folds.dim_size(FoldPosition::RowSplit.index()).0 > 1 {
            Beta(alpha.0 * i64::from(row.ordinal()))
        } else {
            beta
        };
        Some(PtRowGrouping::SameRow(
            row,
            RowGroupNode {
                node: ref_node,
                row: Some(row),
                beta,
            },
        ))
    };

    match (src_row, dest_row) {
        (Some(_), None) => {
            bundled(RowGroupCategory::RowToNonRow, base(prop, candidates.forward))
        }
        // No need to collect nodes for all rows: the reference node's own row is the group.
        (Some(row), Some(dest)) if row == dest => single_row(row, false),
        (Some(_), Some(_)) => {
            bundled(RowGroupCategory::RowNorthSouth, base(prop, candidates.forward))
        }
        // Propagation from PTXRF/PTARF to a row unit counts as ROW_TO_SAME_ROW.
        (None, Some(dest))
            if matches!(units.src, SenComponent::Ptxrf | SenComponent::Ptarf) =>
        {
            single_row(dest, true)
        }
        // UN-bundling is needed in this scenario.
        (None, Some(_)) => {
            let reversed = CoordPropInfo {
                data_connect: prop.data_connect,
                ref_node: prop.node_to_fold,
                node_to_fold: prop.ref_node,
            };
            bundled(
                RowGroupCategory::NonRowToRow,
                base(&reversed, candidates.reverse),
            )
        }
        // Both absent is the early answer above, so the reference's chain has no fallthrough.
        (None, None) => Some(PtRowGrouping::NoBundling),
    }
}

// crustify:todo: e356_buildFoldForAllocation
//   authority : ddc/ddc_fold.cpp:2395  (473 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::buildFoldForAllocation( dsc2::CoordPropInfoType &coordPropInfo, const dsc2::CoordinateType<CoordinateBaseType> &inputRefCoord, dsc2::AllocateNode *allocNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:12521-12997
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e074_retry, e078_dbgPrint, e079_dbgPrint, e080_allDimsCovered, e085_needToConsiderRowBundling, e089_constructAllocElemArrLayout, e092_getNumElementsInPTSlice, e094_getDefaultRowSplitFold, e095_gatherFoldParams, e233_dbgPrint, e234_scaleUpCoord, e235_scaleDownCoord, e237_sameCoordinateRange …

// crustify:todo: e357_buildFoldForCompute
//   authority : ddc/ddc_fold.cpp:3656  (773 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::buildFoldForCompute(dsc2::ComputeNode *computeNode, dsc2::CoordPropInfoType &coordPropInfo, std::vector<int> &constructedInputCoords, std::vector<int> &constructedOutputCoords)
//   extract   : crustify-ddc/cpp/ddc.cpp:13007-13783
//   calls     : e074_retry, e078_dbgPrint, e079_dbgPrint, e080_allDimsCovered, e081_buildFoldForBroadcastDim, e082_getCompRowId, e085_needToConsiderRowBundling, e104_clear, e233_dbgPrint, e234_scaleUpCoord, e235_scaleDownCoord, e236_needNonRowBundling, e238_getRelatedComputeCoord, e298_buildFoldFromAllocation …

// crustify:todo: e358_buildFoldForTransfer
//   authority : ddc/ddc_fold.cpp:4433  (289 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::buildFoldForTransfer(dsc2::TransferNode *transferNode, dsc2::CoordPropInfoType &coordPropInfo)
//   extract   : crustify-ddc/cpp/ddc.cpp:13793-14083
//   calls     : e074_retry, e078_dbgPrint, e079_dbgPrint, e082_getCompRowId, e085_needToConsiderRowBundling, e122_isNodeRelatedToComps, e178_getAccessPattern, e233_dbgPrint, e238_getRelatedComputeCoord, e298_buildFoldFromAllocation, e299_buildFoldFromNonAllocRef, e337_gatherRelatedPTRows

// crustify:todo: e370_buildAndPropagateFold
//   authority : ddc/ddc_fold.cpp:1625  (350 body lines, level 5)
//   class     : Ddc
//   original  : void Ddc::buildAndPropagateFold()
//   extract   : crustify-ddc/cpp/ddc.cpp:14384-14734
//   calls     : e073_addPropInfo, e075_getCurrItem, e078_dbgPrint, e079_dbgPrint, e082_getCompRowId, e086_isAllocateIncoming, e230_addPropInfo, e232_reset, e233_dbgPrint, e238_getRelatedComputeCoord, e240_buildFoldForExternalAllocation, e356_buildFoldForAllocation, e357_buildFoldForCompute, e358_buildFoldForTransfer

// crustify:todo: e375_coordinateCapture
//   authority : ddc/ddc_fold.cpp:1538  (86 body lines, level 6)
//   class     : Ddc
//   original  : void Ddc::coordinateCapture()
//   extract   : crustify-ddc/cpp/ddc.cpp:14875-14961
//   calls     : e076_print, e102_print, e370_buildAndPropagateFold

// ⭐ TESTS FOR ENTRIES 078-085. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e078_e085 {
    use super::*;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PaddedExtent, Sample,
    };
    use crate::generated::DataConnect;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, AllocateNode, DataInfo, Dsts, FoldPosition, InstrAttribute,
        LayoutDims, MaxDimSize, NodeName, NumChunks, ReplicationFactor, StartAddress,
        TransferPadding,
    };
    use crate::units::NumFolds;

    /// One labelled data structure whose layout order is `[out, in]`, which is all
    /// `getLayoutDims` is ever asked for here.
    struct OneLds(LayoutDims);

    impl Dsc for OneLds {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            self.0.clone()
        }
    }

    fn out_then_in() -> OneLds {
        OneLds(LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::In]))
    }

    /// A core stage that splits exactly the dims it is told to. Only [`Stage::is_row_split`] is
    /// read by entry 085; the extents are not part of the question.
    struct RowSplitOn(Vec<PrimaryDim>);

    impl Stage for RowSplitOn {
        fn is_symbolic(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_corelet_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_row_split(&self, dim: PrimaryDim) -> bool {
            self.0.contains(&dim)
        }
        fn is_pe_sfp_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn splits_any_row(&self) -> bool {
            !self.0.is_empty()
        }
        fn extent(&self, _dim: PrimaryDim, _at: Sample) -> Extent {
            Extent(0)
        }
        fn padded_extent(&self, _dim: PrimaryDim, _at: Sample) -> Option<PaddedExtent> {
            None
        }
    }

    fn operand(unit: SenComponent, connect: Option<DataConnect>, lds: Option<u32>) -> Operand {
        Operand {
            unit,
            storage: SenComponent::NoComponent,
            data: DataInfo {
                data_connect: connect,
                my_lds_idx: lds.map(LdsIdx),
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    #[test]
    fn a_compute_line_names_every_operand_with_its_own_connect() {
        let node = ComputeNode {
            name: NodeName("mac0".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Ptrow3,
            inputs: vec![
                operand(SenComponent::Ptrow3, Some(DataConnect::AconstConnect), None),
                operand(SenComponent::Lx, None, None),
            ],
            outputs: vec![operand(SenComponent::Ptsouth, None, None)],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        };
        assert_eq!(
            dbg_print_compute(&node),
            format!(
                " ComputeNode: mac0({:p}) macc [ 'ptrow3(aconst_connect)' 'lx()'] -> \
                 [ 'ptsouth()']",
                &node
            )
        );
    }

    #[test]
    fn a_transfer_line_carries_one_source_and_every_destination() {
        let node = TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            name: NodeName("t0".to_owned()),
            src: operand(SenComponent::Hbm, None, None),
            dsts: Dsts::new(
                operand(SenComponent::Lx, None, None),
                vec![operand(SenComponent::L0, None, None)],
            ),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
        };
        assert_eq!(
            dbg_print_transfer(&node),
            format!(
                " TransferNode: t0({:p}) ['hbm()'] -> [ 'lx()' 'l0()']",
                &node
            )
        );
    }

    #[test]
    fn a_coordinate_covers_a_layout_order_only_once_it_holds_every_dim_of_it() {
        let dsc = out_then_in();
        let mut coord = dsc2::Coordinate::default();
        build_fold_for_broadcast_dim(&mut coord, PrimaryDim::Out, BroadcastScale::Unit);
        assert!(!all_dims_covered(&dsc, &coord, LdsIdx(0)));
        build_fold_for_broadcast_dim(&mut coord, PrimaryDim::In, BroadcastScale::Unit);
        assert!(all_dims_covered(&dsc, &coord, LdsIdx(0)));
    }

    #[test]
    fn a_broadcast_dim_lands_its_four_folds_with_core_corelet_and_rowsplit_at_their_positions() {
        let mut coord = dsc2::Coordinate::default();
        build_fold_for_broadcast_dim(
            &mut coord,
            PrimaryDim::Mb,
            BroadcastScale::CumulativeStickSize(FoldCardinality(64)),
        );
        let dim = coord.fold_dim(PrimaryDim::Mb).expect("the dim was folded");
        assert_eq!(
            dim.folds()
                .map(|fold| fold.label.0.as_str())
                .collect::<Vec<_>>(),
            vec![
                "core_workslice_fold_mb",
                "corelet_fold_mb",
                "rowsplit_fold_mb",
                "elem_arr_0",
            ]
        );
        assert_eq!(
            dim.cardinality_at(FoldPosition::Core),
            Some(FoldCardinality(1))
        );
        assert_eq!(
            dim.cardinality_at(FoldPosition::RowSplit),
            Some(FoldCardinality(1))
        );
        // The element arrangement is the one that took the scale, and it is the three spatial
        // folds' alpha.
        assert_eq!(
            dim.folds().nth(3).map(|f| f.cardinality),
            Some(FoldCardinality(64))
        );
        assert!(dim.folds().take(3).all(|fold| fold.alpha == FoldCoeff(64)));
        assert_eq!((dim.spatial_folds(), dim.elem_arr_folds()), (3, 1));
    }

    #[test]
    fn a_row_id_is_the_digit_in_the_name_and_absent_for_anything_not_on_a_row() {
        assert_eq!(
            comp_row_id(SenComponent::Ptrow7_1).map(PtRowId::ordinal),
            Some(7)
        );
        assert_eq!(
            comp_row_id(SenComponent::L0lurow0).map(PtRowId::ordinal),
            Some(0)
        );
        assert_eq!(comp_row_id(SenComponent::Lx), None);
        let (row3, row4) = (
            comp_row_id(SenComponent::Ptrow3).expect("a row"),
            comp_row_id(SenComponent::L0lurow4_0).expect("a row"),
        );
        assert!(row3.is_adjacent_or_same(row4));
        assert!(!row3.is_adjacent_or_same(comp_row_id(SenComponent::Ptrow5).expect("a row")));
    }

    #[test]
    fn a_nodes_component_is_its_own_unit_and_a_transfer_answers_for_both_sides() {
        let alloc = AllocateNode {
            name: NodeName("a0".to_owned()),
            component: SenComponent::Lx,
            lds: Some(LdsIdx(0)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new((PrimaryDim::Out, MaxDimSize::Unset), Vec::new()),
            start_address: StartAddress::default(),
            placement: AllocPlacement::default(),
            gap_stick_spread: Default::default(),
            alloc_users: Vec::new(),
        };
        let compute = ComputeNode {
            name: NodeName("c0".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Ptrow2,
            inputs: vec![],
            outputs: vec![],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        };
        let transfer = TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            name: NodeName("t0".to_owned()),
            src: operand(SenComponent::Hbm, None, None),
            dsts: Dsts::new(operand(SenComponent::L0, None, None), vec![]),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
        };
        assert_eq!(
            component(Node::Allocate(&alloc), TransferSide::Src),
            SenComponent::Lx
        );
        assert_eq!(
            component(Node::Compute(&compute), TransferSide::Dst),
            SenComponent::Ptrow2
        );
        assert_eq!(
            component(Node::Transfer(&transfer), TransferSide::Src),
            SenComponent::Hbm
        );
        assert_eq!(
            component(Node::Transfer(&transfer), TransferSide::Dst),
            SenComponent::L0
        );
    }

    #[test]
    fn an_operand_without_a_labelled_ds_has_no_layout_dims_and_neither_has_one_past_the_end() {
        let dsc = out_then_in();
        let compute = ComputeNode {
            name: NodeName("c0".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Ptrow0,
            inputs: vec![
                operand(SenComponent::Lx, None, Some(0)),
                operand(SenComponent::Lx, None, None),
            ],
            outputs: vec![],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        };
        let node = Node::Compute(&compute);
        assert_eq!(
            layout_dims_from_node(&dsc, node, OperandPos::Input(0)),
            vec![PrimaryDim::Out, PrimaryDim::In]
        );
        assert!(layout_dims_from_node(&dsc, node, OperandPos::Input(1)).is_empty());
        assert!(layout_dims_from_node(&dsc, node, OperandPos::Input(9)).is_empty());
        assert!(layout_dims_from_node(&dsc, node, OperandPos::Output(0)).is_empty());
    }

    #[test]
    fn row_bundling_asks_only_about_the_lowest_ordinal_row_split_dim() {
        assert!(!need_to_consider_row_bundling(
            &RowSplitOn(vec![]),
            &[PrimaryDim::Out]
        ));
        // `Out` is ordinal 1 and `Mb` is 3, so `rowSplit_.begin()` is `Out` — a working set naming
        // only the LATER dim does not bundle.
        let both = RowSplitOn(vec![PrimaryDim::Mb, PrimaryDim::Out]);
        assert!(need_to_consider_row_bundling(&both, &[PrimaryDim::Out]));
        assert!(!need_to_consider_row_bundling(&both, &[PrimaryDim::Mb]));
    }
}

// ⭐ TESTS FOR ENTRIES 233-240. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e233_e240 {
    use super::*;
    use crate::arch::Target;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{PaddedExtent, Sample};
    use crate::schedule::ddc::metadata::DatastageId;
    use crate::schedule::ddc::transformation_util::LoopDims;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::InstrAttribute;
    use crate::schedule::l3::dl_ops::LoopDistribution;
    use crate::units::NumFolds;

    /// A fold manager's levels, OUTERMOST FIRST, as its four getters report them.
    struct Levels(Vec<FoldParamInfo>);

    impl AffineFoldDims for Levels {
        fn num_dims(&self) -> usize {
            self.0.len()
        }
        fn alpha_beta(&self, dim: usize) -> (Alpha, Beta) {
            (self.0[dim].alpha, self.0[dim].beta)
        }
        fn dim_size(&self, dim: usize) -> Cardinality {
            self.0[dim].cardinality
        }
        fn dim_label(&self, dim: usize) -> Option<FoldLabel> {
            self.0[dim].label
        }
    }

    const fn level(alpha: i64, beta: i64, cardinality: u64) -> FoldParamInfo {
        FoldParamInfo {
            alpha: Alpha(alpha),
            beta: Beta(beta),
            cardinality: Cardinality(cardinality),
            label: None,
        }
    }

    fn coord_dim(levels: &Levels, spatial: u32, temporal: u32) -> CoordDim<'_, Levels> {
        CoordDim {
            folds: Some(levels),
            spatial: SpatialFolds(spatial),
            temporal: TemporalFolds(temporal),
            pad: PadType::NoPad,
        }
    }

    /// One dim's folds as `(alpha, beta, cardinality, label)`, OUTERMOST FIRST.
    fn folds_of(coord: &dsc2::Coordinate, dim: PrimaryDim) -> Vec<(i64, i64, u32, String)> {
        coord.fold_dim(dim).map_or_else(Vec::new, |folds| {
            folds
                .folds()
                .map(|fold| {
                    (
                        fold.alpha.0,
                        fold.beta.0,
                        fold.cardinality.0,
                        fold.label.0.clone(),
                    )
                })
                .collect()
        })
    }

    /// The two dsc2 seams, tabulated: every enclosing loop is related to the dim, the distributor
    /// answers one fixed element arrangement, and it decided nothing for any loop.
    struct Distributor(Vec<FoldParamInfo>);

    impl TemporalLoopDistribution for Distributor {
        type LoopParams = ();

        fn related_loops<'l>(
            &self,
            _dim: PrimaryDimAndKind,
            chain: &[LoopAndDim<'l>],
            _pad: PadType,
        ) -> Vec<LoopAndDim<'l>> {
            chain.to_vec()
        }

        fn distribute(
            &self,
            _request: &ElemArrDistribution<'_>,
            _loop_params: &mut Self::LoopParams,
        ) -> Vec<FoldParamInfo> {
            self.0.clone()
        }

        fn distributed(
            &self,
            _loop_params: &Self::LoopParams,
            _loop_node: &LoopNode,
            _dim: PrimaryDim,
        ) -> Option<DistributedLoop> {
            None
        }
    }

    /// A tree answered from a parent table, as entries 087 and 088's own fixture is.
    struct Tree {
        kinds: Vec<NodeKind>,
        parents: Vec<Option<u32>>,
    }

    impl ScheduleTree for Tree {
        fn kind(&self, node: NodeId) -> NodeKind {
            self.kinds[node.0 as usize]
        }
        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.parents[node.0 as usize].map(NodeId)
        }
        fn children(&self, block: BlockId) -> Vec<NodeId> {
            (0..u32::try_from(self.parents.len()).unwrap_or(0))
                .map(NodeId)
                .filter(|&node| self.parent(node) == Some(block.node()))
                .collect()
        }
    }

    fn compute_on(name: &str, unit: SenComponent) -> ComputeNode {
        ComputeNode {
            name: dsc2::NodeName(name.to_owned()),
            op: DdlComputeType::Macc,
            ex_unit: unit,
            inputs: Vec::new(),
            outputs: Vec::new(),
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        }
    }

    /// `getAllocation` and the scale-to-value edge, tabulated.
    struct Allocs {
        placed: Vec<(LdsIdx, DfirUnit, AllocId)>,
        values: Vec<(AllocId, AllocId)>,
    }

    impl Allocations for Allocs {
        fn allocation(&self, stored: StoredStream) -> Option<AllocId> {
            let DataOrigin::LabeledDs(lds) = stored.stream.origin else {
                return None;
            };
            self.placed
                .iter()
                .find(|(named, unit, _)| *named == lds && *unit == stored.storage)
                .map(|&(_, _, alloc)| alloc)
        }
        fn value_allocation(&self, scale: AllocId) -> Option<AllocId> {
            self.values
                .iter()
                .find(|(named, _)| *named == scale)
                .map(|&(_, value)| value)
        }
    }

    /// A core stage that row splits exactly the dims it is told to; entry 239 asks it nothing else.
    struct RowSplitOn(Vec<PrimaryDim>);

    impl Stage for RowSplitOn {
        fn is_symbolic(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_corelet_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_row_split(&self, dim: PrimaryDim) -> bool {
            self.0.contains(&dim)
        }
        fn is_pe_sfp_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn splits_any_row(&self) -> bool {
            !self.0.is_empty()
        }
        fn extent(&self, _dim: PrimaryDim, _at: Sample) -> Extent {
            Extent(0)
        }
        fn padded_extent(&self, _dim: PrimaryDim, _at: Sample) -> Option<PaddedExtent> {
            None
        }
    }

    /// The eight PT-row components, whose ordinals are the rows a full row group covers.
    const PT_ROW_COMPONENTS: [SenComponent; 8] = [
        SenComponent::Ptrow0,
        SenComponent::Ptrow1,
        SenComponent::Ptrow2,
        SenComponent::Ptrow3,
        SenComponent::Ptrow4,
        SenComponent::Ptrow5,
        SenComponent::Ptrow6,
        SenComponent::Ptrow7,
    ];

    /// The core stage entry 240 folds over, plus its two extra answers.
    struct ExternalStage {
        corelet_extent: Extent,
        core_extent: Extent,
        corelets: Cardinality,
        slices: Cardinality,
        trip: LoopTrip,
    }

    impl CoreStage for ExternalStage {
        fn pad_stride(&self, _dim: PrimaryDim) -> Option<Stride> {
            None
        }
        fn first_corelet_share(&self, _dim: PrimaryDim) -> Option<Extent> {
            None
        }
        fn core_extent(&self, _dim: PrimaryDim) -> Extent {
            self.core_extent
        }
        fn corelets_used(&self) -> Cardinality {
            self.corelets
        }
        fn work_slices(&self, _dim: PrimaryDim) -> Cardinality {
            self.slices
        }
    }

    impl ExternalFoldStage for ExternalStage {
        fn corelet_extent(&self, _dim: PrimaryDim, _pad: PadType) -> Extent {
            self.corelet_extent
        }
        fn loop_trip(&self, _loop_node: &LoopNode, _dim: PrimaryDimAndKind) -> LoopTrip {
            self.trip
        }
    }

    fn loop_named(name: &str, dim: PrimaryDim) -> LoopNode {
        LoopNode {
            name: dsc2::NodeName(name.to_owned()),
            num: DatastageId(0),
            den: DatastageId(1),
            dims: LoopDims::new(
                PrimaryDimAndKind {
                    dim,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        }
    }

    #[test]
    fn a_node_line_is_its_kind_its_name_and_its_own_address() {
        let alloc = dsc2::AllocateNode {
            name: dsc2::NodeName("lx_alloc0".to_owned()),
            component: SenComponent::Lx,
            lds: Some(LdsIdx(0)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: dsc2::AllocLayout::new((PrimaryDim::Out, dsc2::MaxDimSize::Unset), Vec::new()),
            start_address: dsc2::StartAddress::default(),
            placement: dsc2::AllocPlacement::default(),
            gap_stick_spread: Default::default(),
            alloc_users: Vec::new(),
        };
        assert_eq!(
            dbg_print(Node::Allocate(&alloc)),
            format!(" AllocateNode: lx_alloc0({:p})", &alloc)
        );
        let compute = compute_on("mac0", SenComponent::Ptrow3);
        assert_eq!(
            dbg_print(Node::Compute(&compute)),
            dbg_print_compute(&compute)
        );
    }

    #[test]
    fn a_scale_block_multiplies_every_level_and_takes_an_element_arrangement_of_its_own() {
        // Three spatial levels and no temporal one, so the scale block's own levels sit outside the
        // redistribution window and the distributor is never reached.
        let levels = Levels(vec![
            level(100, 7, 2),
            level(0, 0, 1),
            level(0, 3, 1),
            level(1, 0, 8),
        ]);
        let rhs = coord_dim(&levels, 3, 0);
        let value = MxValueTensor::of(
            ScaledLds::Value,
            LdsIdx(4),
            PrimaryDim::Out,
            ScaleBlock::of(Cardinality(4)).expect("four elements is a scale block"),
        )
        .expect("a value tensor");
        let mut lhs = dsc2::Coordinate::default();

        scale_up_coord(
            &value,
            &rhs,
            &mut lhs,
            NodeId(1),
            SenComponent::Lx,
            &Distributor(Vec::new()),
            &[],
            &mut (),
        );

        assert_eq!(
            folds_of(&lhs, PrimaryDim::Out),
            vec![
                (400, 28, 2, String::new()),
                (0, 0, 1, String::new()),
                (0, 12, 1, String::new()),
                (4, 0, 8, String::new()),
                (1, 0, 4, "elem_arr_scaleup".to_owned()),
            ]
        );
        let dim = lhs.fold_dim(PrimaryDim::Out).expect("the dim was rebuilt");
        assert_eq!(
            (
                dim.spatial_folds(),
                dim.temporal_folds(),
                dim.elem_arr_folds()
            ),
            (3, 0, 2)
        );
    }

    #[test]
    fn the_levels_that_fill_one_scale_block_collapse_and_every_outer_stride_divides() {
        // 8 x 4 elements inside one spatial level, and a scale block of 4: the innermost level IS
        // the block, so it goes, and the two above it divide.
        let levels = Levels(vec![level(32, 0, 3), level(4, 0, 8), level(1, 0, 4)]);
        let rhs = coord_dim(&levels, 1, 0);
        let scale = MxScaleTensor::of(
            ScaledLds::Scale,
            PrimaryDim::Out,
            ScaleBlock::of(Cardinality(4)).expect("four elements is a scale block"),
        )
        .expect("a scale tensor");
        let mut lhs = dsc2::Coordinate::default();

        assert_eq!(scale_down_coord(&scale, &rhs, &mut lhs), ScaleDown::Applied);
        assert_eq!(
            folds_of(&lhs, PrimaryDim::Out),
            vec![(8, 0, 3, String::new()), (1, 0, 8, String::new())]
        );
    }

    #[test]
    fn a_gap_inside_the_scale_block_refuses_and_leaves_the_target_as_it_was() {
        // The innermost level covers 2 of the 8 elements the block needs, and the level above it
        // strides 999 rather than the 2 a contiguous arrangement would.
        let levels = Levels(vec![level(100, 0, 5), level(999, 0, 2), level(1, 0, 2)]);
        let rhs = coord_dim(&levels, 1, 0);
        let scale = MxScaleTensor::of(
            ScaledLds::Scale,
            PrimaryDim::Out,
            ScaleBlock::of(Cardinality(8)).expect("eight elements is a scale block"),
        )
        .expect("a scale tensor");
        let mut lhs = dsc2::Coordinate::default();

        assert_eq!(
            scale_down_coord(&scale, &rhs, &mut lhs),
            ScaleDown::NonContiguousInScaleBlock
        );
        assert_eq!(lhs, dsc2::Coordinate::default());
    }

    #[test]
    fn two_non_row_nodes_under_one_conditional_bundle_and_one_each_under_its_own_does_not() {
        let (lx, sfp) = (
            compute_on("c0", SenComponent::Lx),
            compute_on("c1", SenComponent::Sfp),
        );
        let nodes = [
            ConnectedNode {
                id: NodeId(3),
                node: Node::Compute(&lx),
            },
            ConnectedNode {
                id: NodeId(4),
                node: Node::Compute(&sfp),
            },
        ];
        //   `0 Loop { 1 Condition { 2 Block { 3 Compute, 4 Compute } } }`
        let shared = Tree {
            kinds: vec![
                NodeKind::Loop,
                NodeKind::Condition,
                NodeKind::Block,
                NodeKind::Compute,
                NodeKind::Compute,
            ],
            parents: vec![None, Some(0), Some(1), Some(2), Some(2)],
        };
        assert!(need_non_row_bundling(
            &shared,
            &nodes,
            ConnectEnd::Producers
        ));
        //   `0 Loop { 1 Condition { 3 Compute }, 2 Condition { 4 Compute } }`
        let separate = Tree {
            kinds: vec![
                NodeKind::Loop,
                NodeKind::Condition,
                NodeKind::Condition,
                NodeKind::Compute,
                NodeKind::Compute,
            ],
            parents: vec![None, Some(0), Some(0), Some(1), Some(2)],
        };
        assert!(!need_non_row_bundling(
            &separate,
            &nodes,
            ConnectEnd::Consumers
        ));
        // One node is not a group at all.
        assert!(!need_non_row_bundling(
            &shared,
            &nodes[..1],
            ConnectEnd::Producers
        ));
    }

    /// One side of the comparison over the `In` dim, on the cores it is relevant to.
    fn side<'a>(
        levels: &'a Levels,
        slices: &'a WorkSlices,
        cores: &BTreeMap<CoreOrdinal, BTreeSet<CoreletOrdinal>>,
    ) -> CoordSide<'a, Levels> {
        CoordSide {
            dims: BTreeMap::from([(PrimaryDim::In, levels)]),
            work_slices: slices,
            cores: cores.clone(),
        }
    }

    #[test]
    fn two_coordinates_agree_when_each_core_starts_its_own_work_slice_at_the_same_element() {
        let sdsc_slices = WorkSlices(BTreeMap::from([(
            CoreOrdinal(0),
            BTreeMap::from([(PrimaryDim::In, WorkSlice(0))]),
        )]));
        let lhs_slices = WorkSlices(BTreeMap::from([(
            CoreOrdinal(0),
            BTreeMap::from([(PrimaryDim::In, WorkSlice(5))]),
        )]));
        let rhs_slices = WorkSlices(BTreeMap::from([(
            CoreOrdinal(0),
            BTreeMap::from([(PrimaryDim::In, WorkSlice(3))]),
        )]));
        let cores = BTreeMap::from([(CoreOrdinal(0), BTreeSet::from([CoreletOrdinal(0)]))]);
        // Work slice 5 at stride 2, against work slice 3 at stride 3 with an offset of 1: both
        // coordinates start at element 10.
        let lhs_levels = Levels(vec![level(2, 0, 4), level(0, 0, 1)]);
        let rhs_levels = Levels(vec![level(3, 1, 4), level(0, 0, 1)]);
        let lhs = side(&lhs_levels, &lhs_slices, &cores);
        let rhs = side(&rhs_levels, &rhs_slices, &cores);

        assert!(same_coordinate_range(
            &lhs,
            &rhs,
            &sdsc_slices,
            &BTreeSet::new(),
            DimCoverage::EveryDim
        ));

        // The same coordinate one element along does not agree.
        let moved_levels = Levels(vec![level(3, 0, 4), level(0, 0, 1)]);
        let moved = side(&moved_levels, &rhs_slices, &cores);
        assert!(!same_coordinate_range(
            &lhs,
            &moved,
            &sdsc_slices,
            &BTreeSet::new(),
            DimCoverage::EveryDim
        ));
    }

    #[test]
    fn an_allocate_reference_reaches_a_computes_input_through_its_value_allocation() {
        let inputs = [StoredStream {
            stream: DataStream {
                origin: DataOrigin::LabeledDs(LdsIdx(4)),
                data_connect: None,
            },
            storage: DfirUnit::L0,
        }];
        let compute = ComputeStreams {
            ex_unit: DfirUnit::Pe,
            inputs: &inputs,
            outputs: &[],
            opaque: None,
        };
        // The reference is the SCALE allocation, and a PE compute consumes the value tensor it is
        // implicitly connected through.
        let reference = ComputeCoordRef {
            data_connect: None,
            ref_node: ComputeRefNode::Allocation {
                alloc: AllocId(7),
                scaled: Some(ScaledLds::Scale),
            },
        };
        let dsc = Allocs {
            placed: vec![(LdsIdx(4), DfirUnit::L0, AllocId(3))],
            values: vec![(AllocId(7), AllocId(3))],
        };

        assert_eq!(
            get_related_compute_coord(&dsc, &compute, &reference),
            Some(RelatedCoord {
                coord: RelatedComputeCoord::Input(0),
                lds: Some(LdsIdx(4)),
            })
        );
    }

    #[test]
    fn a_full_row_group_folds_every_pt_row_at_the_step_between_the_first_two() {
        let nodes: Vec<RowGroupNode> = (0..Target::PT_ROWS)
            .map(|row| RowGroupNode {
                node: NodeId(row),
                row: comp_row_id(PT_ROW_COMPONENTS[row as usize]),
                beta: Beta(i64::from(row) * 16),
            })
            .collect();
        let group = RowGroup::new(RowGroupCategory::NonRowToRow, nodes[0], nodes[1..].to_vec());
        let stage = RowSplitOn(vec![PrimaryDim::Out]);

        assert_eq!(
            compute_params_for_row_split_fold::<Target, _>(&stage, PrimaryDim::Out, Some(&group)),
            Some(FoldParamInfo {
                alpha: Alpha(16),
                beta: Beta(0),
                cardinality: Cardinality(u64::from(Target::PT_ROWS)),
                label: Some(FoldLabel::RowSplitFold),
            })
        );
        // A dim the core stage does not row split takes the identity fold, group or no group.
        assert_eq!(
            compute_params_for_row_split_fold::<Target, _>(&stage, PrimaryDim::In, Some(&group)),
            Some(default_row_split_fold())
        );
    }

    #[test]
    fn an_external_dim_gets_its_element_arrangement_its_loop_its_rowsplit_and_its_spatial_pair() {
        let stage = ExternalStage {
            corelet_extent: Extent(16),
            core_extent: Extent(64),
            corelets: Cardinality(1),
            slices: Cardinality(2),
            trip: LoopTrip::Staged {
                iterations: Cardinality(4),
                alpha: Alpha(16),
            },
        };
        let padding = PaddingForm::default();
        let sized = [(
            PrimaryDimAndKind {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::Unpadded,
            },
            DimScale::Sized,
        )];
        let alloc = ExternalAllocation {
            lds: LdsIdx(0),
            component: SenComponent::Lx,
            layout: &sized,
            padding: &padding,
        };
        let loop_node = loop_named("loop_ds0_ds1", PrimaryDim::Out);
        let chain = [LoopAndDim {
            loop_node: &loop_node,
            dim: PrimaryDimAndKind {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::Unpadded,
            },
            distribution: LoopDistribution::AboveChunk,
        }];
        let distribution = Distributor(vec![level(1, 0, 16)]);
        let mut coord = dsc2::Coordinate::default();

        build_fold_for_external_allocation(
            &alloc,
            NodeId(9),
            &stage,
            &distribution,
            &chain,
            &mut coord,
            &mut (),
        );

        assert_eq!(
            folds_of(&coord, PrimaryDim::Out),
            vec![
                (64, 0, 2, "core_workslice_fold_dim".to_owned()),
                (0, 0, 1, "corelet_fold_dim".to_owned()),
                (0, 0, 1, "rowsplit_fold_out".to_owned()),
                (16, 0, 4, "loop_ds0_ds1 out".to_owned()),
                (1, 0, 16, "elem_arr_0".to_owned()),
            ]
        );
        assert!(coord.fold_constructed());

        // A broadcast dim takes the four folds entry 081 mints instead, and no loop is distributed
        // over it at all.
        let broadcast = [(
            PrimaryDimAndKind {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::Unpadded,
            },
            DimScale::Broadcast(BroadcastScale::Unit),
        )];
        let mut coord = dsc2::Coordinate::default();
        build_fold_for_external_allocation(
            &ExternalAllocation {
                layout: &broadcast,
                ..alloc
            },
            NodeId(9),
            &stage,
            &distribution,
            &chain,
            &mut coord,
            &mut (),
        );
        assert_eq!(
            folds_of(&coord, PrimaryDim::Out),
            vec![
                (1, 0, 1, "core_workslice_fold_out".to_owned()),
                (1, 0, 1, "corelet_fold_out".to_owned()),
                (1, 0, 1, "rowsplit_fold_out".to_owned()),
                (0, 0, 1, "elem_arr_0".to_owned()),
            ]
        );
    }

    /// REGRESSION, entry 240: the port used to read `allocNode->padding_` directly and never run
    /// `setPadding(allocNode->padding_)` (`ddc/ddc_fold.cpp:2268`), so the coordinate went out
    /// carrying no form and every later `getPadding()` of it (`:2718`, `:2821`, `:3774`) read `NOPAD`.
    #[test]
    fn an_external_allocations_form_is_stamped_onto_its_coordinate_and_only_past_the_early_returns() {
        let stage = ExternalStage {
            corelet_extent: Extent(16),
            core_extent: Extent(64),
            corelets: Cardinality(1),
            slices: Cardinality(2),
            trip: LoopTrip::Staged {
                iterations: Cardinality(4),
                alpha: Alpha(16),
            },
        };
        let mut padding = PaddingForm::default();
        padding.set_padding(PrimaryDim::Out, PadType::PaddedWZeroPad);
        let sized = [(
            PrimaryDimAndKind {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::Unpadded,
            },
            DimScale::Sized,
        )];
        let alloc = ExternalAllocation {
            lds: LdsIdx(0),
            component: SenComponent::Lx,
            layout: &sized,
            padding: &padding,
        };
        let loop_node = loop_named("loop_ds0_ds1", PrimaryDim::Out);
        let chain = [LoopAndDim {
            loop_node: &loop_node,
            dim: PrimaryDimAndKind {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::Unpadded,
            },
            distribution: LoopDistribution::AboveChunk,
        }];
        let distribution = Distributor(vec![level(1, 0, 16)]);

        let mut coord = dsc2::Coordinate::default();
        build_fold_for_external_allocation(
            &alloc,
            NodeId(9),
            &stage,
            &distribution,
            &chain,
            &mut coord,
            &mut (),
        );
        assert_eq!(coord.padding(PrimaryDim::Out), PadType::PaddedWZeroPad);

        // `setPadding` sits BELOW the empty-loop-chain return (`:2258-2268`), so a coordinate that
        // never reaches the dim loop is left exactly as it came in.
        let mut untouched = dsc2::Coordinate::default();
        build_fold_for_external_allocation(
            &alloc,
            NodeId(9),
            &stage,
            &distribution,
            &[],
            &mut untouched,
            &mut (),
        );
        assert_eq!(untouched.padding(PrimaryDim::Out), PadType::NoPad);
    }
}

/// ⭐ TESTS FOR ENTRIES 297-299, AND FOR ENTRY 337, whose two single-row arms read the same fold
/// managers and whose other three arms are entry 297's own. Union this module with this file's other
/// test modules when they land.
#[cfg(test)]
mod tests_e297_e299 {
    use super::*;
    use crate::arch::Target;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{PaddedExtent, Sample};
    use crate::schedule::ddc::metadata::DatastageId;
    use crate::schedule::ddc::transformation_util::LoopDims;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        DataInfo, Dsts, InstrAttribute, LayoutDims, NodeName, NumChunks, Operand,
        ReplicationFactor, TransferPadding,
    };
    use crate::units::NumFolds;

    /// A fold manager's levels, OUTERMOST FIRST, as its four getters report them.
    struct Levels(Vec<FoldParamInfo>);

    impl AffineFoldDims for Levels {
        fn num_dims(&self) -> usize {
            self.0.len()
        }
        fn alpha_beta(&self, dim: usize) -> (Alpha, Beta) {
            (self.0[dim].alpha, self.0[dim].beta)
        }
        fn dim_size(&self, dim: usize) -> Cardinality {
            self.0[dim].cardinality
        }
        fn dim_label(&self, dim: usize) -> Option<FoldLabel> {
            self.0[dim].label
        }
    }

    const fn level(
        alpha: i64,
        beta: i64,
        cardinality: u64,
        label: Option<FoldLabel>,
    ) -> FoldParamInfo {
        FoldParamInfo {
            alpha: Alpha(alpha),
            beta: Beta(beta),
            cardinality: Cardinality(cardinality),
            label,
        }
    }

    /// The four levels a propagated dim starts from: the core work slice, the corelet, the row split
    /// and one element arrangement.
    fn ref_levels() -> Levels {
        Levels(vec![
            level(100, 0, 2, Some(FoldLabel::CoreWorksliceFoldDim)),
            level(50, 0, 2, Some(FoldLabel::CoreletFoldDim)),
            level(0, 0, 1, Some(FoldLabel::RowSplitFold)),
            level(1, 0, 8, None),
        ])
    }

    /// One labelled data structure whose layout order this fixture states outright.
    struct OneLds(LayoutDims);

    impl Dsc for OneLds {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            self.0.clone()
        }
    }

    impl Allocations for OneLds {
        fn allocation(&self, _stored: StoredStream) -> Option<AllocId> {
            None
        }
        fn value_allocation(&self, _scale: AllocId) -> Option<AllocId> {
            None
        }
    }

    /// A tree answered from a parent table.
    struct Tree(Vec<Option<u32>>);

    impl ScheduleTree for Tree {
        fn kind(&self, node: NodeId) -> NodeKind {
            if self.0[node.0 as usize].is_none() {
                NodeKind::Block
            } else {
                NodeKind::Transfer
            }
        }
        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.0[node.0 as usize].map(NodeId)
        }
        fn children(&self, block: BlockId) -> Vec<NodeId> {
            (0..u32::try_from(self.0.len()).unwrap_or(0))
                .map(NodeId)
                .filter(|&node| self.parent(node) == Some(block.node()))
                .collect()
        }
    }

    /// A core stage that row splits exactly the dims it is told to and answers one fixed trip count.
    struct RowSplitOn(Vec<PrimaryDim>);

    impl Stage for RowSplitOn {
        fn is_symbolic(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_corelet_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_row_split(&self, dim: PrimaryDim) -> bool {
            self.0.contains(&dim)
        }
        fn is_pe_sfp_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn splits_any_row(&self) -> bool {
            !self.0.is_empty()
        }
        fn extent(&self, _dim: PrimaryDim, _at: Sample) -> Extent {
            Extent(0)
        }
        fn padded_extent(&self, _dim: PrimaryDim, _at: Sample) -> Option<PaddedExtent> {
            None
        }
    }

    impl CoreStage for RowSplitOn {
        fn pad_stride(&self, _dim: PrimaryDim) -> Option<Stride> {
            None
        }
        fn first_corelet_share(&self, _dim: PrimaryDim) -> Option<Extent> {
            None
        }
        fn core_extent(&self, _dim: PrimaryDim) -> Extent {
            Extent(0)
        }
        fn corelets_used(&self) -> Cardinality {
            Cardinality(2)
        }
        fn work_slices(&self, _dim: PrimaryDim) -> Cardinality {
            Cardinality(1)
        }
    }

    impl PropagatedFoldStage for RowSplitOn {
        fn loop_iterations(
            &self,
            _loop_node: &LoopNode,
            _dim: PrimaryDimAndKind,
            _prop: SenComponent,
            _row: Option<PtRowId>,
        ) -> Cardinality {
            Cardinality(4)
        }
    }

    /// The dsc2 seams, tabulated: every enclosing loop walks the dim, the distributor answers one
    /// fixed element arrangement and one fixed stride for every loop.
    struct Seams(Vec<FoldParamInfo>);

    impl TemporalLoopDistribution for Seams {
        type LoopParams = ();

        fn related_loops<'l>(
            &self,
            _dim: PrimaryDimAndKind,
            chain: &[LoopAndDim<'l>],
            _pad: PadType,
        ) -> Vec<LoopAndDim<'l>> {
            chain.to_vec()
        }

        fn distribute(
            &self,
            _request: &ElemArrDistribution<'_>,
            _loop_params: &mut Self::LoopParams,
        ) -> Vec<FoldParamInfo> {
            self.0.clone()
        }

        fn distributed(
            &self,
            _loop_params: &Self::LoopParams,
            _loop_node: &LoopNode,
            _dim: PrimaryDim,
        ) -> Option<DistributedLoop> {
            Some(DistributedLoop {
                alpha: Alpha(16),
                beta: Beta(0),
            })
        }
    }

    impl LoopRelevance for Seams {
        fn loop_relevant_for_dim(
            &self,
            _dim: PrimaryDim,
            _loop_node: &LoopNode,
            _pad: PadType,
        ) -> bool {
            true
        }
        fn distributed_corelet_slice(
            &self,
            _loop_params: &Self::LoopParams,
            _dim: PrimaryDim,
        ) -> Option<DistributedLoop> {
            None
        }
    }

    fn loop_named(name: &str) -> LoopNode {
        LoopNode {
            name: NodeName(name.to_owned()),
            num: DatastageId(0),
            den: DatastageId(1),
            dims: LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::Out,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        }
    }

    fn operand(lds: Option<u32>) -> Operand {
        Operand {
            unit: SenComponent::Lx,
            storage: SenComponent::NoComponent,
            data: DataInfo {
                data_connect: Some(DataConnect::AconstConnect),
                my_lds_idx: lds.map(LdsIdx),
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    fn transfer_from_lds0() -> TransferNode {
        TransferNode {
            unit_time_transfer_num_chunks: NumChunks::ONE,
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            name: NodeName("t0".to_owned()),
            src: operand(Some(0)),
            dsts: Dsts::new(operand(None), Vec::new()),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
        }
    }

    fn compute_from_lds0() -> ComputeNode {
        ComputeNode {
            name: NodeName("c0".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Ptrow0,
            inputs: vec![operand(Some(0))],
            outputs: Vec::new(),
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        }
    }

    /// The propagation matched on its `data_connect=` alone, so no allocation lookup is reached.
    fn propagation() -> CoordPropInfo {
        CoordPropInfo {
            data_connect: Some(DataConnect::AconstConnect),
            ref_node: PropEnd::Other,
            node_to_fold: PropEnd::Other,
        }
    }

    fn matched_stream() -> StoredStream {
        StoredStream {
            stream: DataStream {
                origin: DataOrigin::LabeledDs(LdsIdx(0)),
                data_connect: Some(DataConnect::AconstConnect),
            },
            storage: DfirUnit::L0,
        }
    }

    /// One dim's folds as `(label, alpha, cardinality)`, OUTERMOST FIRST.
    fn folds_of(coord: &dsc2::Coordinate, dim: PrimaryDim) -> Vec<(String, i64, u32)> {
        coord.fold_dim(dim).map_or_else(Vec::new, |folds| {
            folds
                .folds()
                .map(|fold| (fold.label.0.clone(), fold.alpha.0, fold.cardinality.0))
                .collect()
        })
    }

    const PT_ROWS: [SenComponent; 8] = [
        SenComponent::Ptrow0,
        SenComponent::Ptrow1,
        SenComponent::Ptrow2,
        SenComponent::Ptrow3,
        SenComponent::Ptrow4,
        SenComponent::Ptrow5,
        SenComponent::Ptrow6,
        SenComponent::Ptrow7,
    ];

    #[test]
    fn every_pt_row_reporting_the_same_datastream_groups_under_the_block_that_holds_them_all() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::In]));
        let core_ds = RowSplitOn(vec![PrimaryDim::Out]);
        // Node 0 is the block; every candidate hangs directly off it.
        let tree = Tree(
            std::iter::once(None)
                .chain(std::iter::repeat_n(Some(0), Target::PT_ROWS as usize))
                .collect(),
        );
        let levels = ref_levels();
        let slices = WorkSlices::default();
        let coord = RowCoordinate {
            dims: BTreeMap::from([(PrimaryDim::Out, &levels)]),
            work_slices: &slices,
        };
        let transfer = transfer_from_lds0();
        let candidates: Vec<RowCandidate<'_, Levels>> = (0..Target::PT_ROWS)
            .map(|row| RowCandidate {
                id: NodeId(row + 1),
                node: Node::Transfer(&transfer),
                row: comp_row_id(PT_ROWS[row as usize]),
                cores: BTreeMap::new(),
                operands: vec![RowOperand {
                    pos: OperandPos::Input(0),
                    stream: matched_stream(),
                    coord: &coord,
                    mx: None,
                }],
            })
            .collect();

        let found = gather_related_pt_rows_base::<Target, _, _, _, _>(
            &dsc,
            &core_ds,
            &tree,
            &propagation(),
            crate::schedule::ddc::ScaleDown::No,
            &candidates,
            &slices,
        );
        assert_eq!(
            found,
            Some(RelatedPtRows::Group {
                nodes: (0..Target::PT_ROWS)
                    .map(|row| RowGroupNode {
                        node: NodeId(row + 1),
                        row: comp_row_id(PT_ROWS[row as usize]),
                        // The rowsplit slot's beta, which every candidate reads off the same manager.
                        beta: Beta(0),
                    })
                    .collect(),
                common_ancestor: NodeId(0),
            })
        );
    }

    #[test]
    fn the_two_refusals_are_read_in_opposite_orders_on_a_transfer_and_on_a_compute() {
        // The row-split dim is absent from the layout AND the coordinate has no fold for it, so the
        // kind alone decides which refusal answers.
        let dsc = OneLds(LayoutDims::new(PrimaryDim::In, Vec::new()));
        let core_ds = RowSplitOn(vec![PrimaryDim::Out]);
        let tree = Tree(vec![None, Some(0)]);
        let slices = WorkSlices::default();
        let coord: RowCoordinate<'_, Levels> = RowCoordinate {
            dims: BTreeMap::new(),
            work_slices: &slices,
        };
        let transfer = transfer_from_lds0();
        let compute = compute_from_lds0();
        let candidate = |node| {
            vec![RowCandidate {
                id: NodeId(1),
                node,
                row: comp_row_id(SenComponent::Ptrow0),
                cores: BTreeMap::new(),
                operands: vec![RowOperand {
                    pos: OperandPos::Input(0),
                    stream: matched_stream(),
                    coord: &coord,
                    mx: None,
                }],
            }]
        };
        let gather = |candidates: &[RowCandidate<'_, Levels>]| {
            gather_related_pt_rows_base::<Target, _, _, _, _>(
                &dsc,
                &core_ds,
                &tree,
                &propagation(),
                crate::schedule::ddc::ScaleDown::No,
                candidates,
                &slices,
            )
        };
        assert_eq!(
            gather(&candidate(Node::Transfer(&transfer))),
            Some(RelatedPtRows::NoBundling)
        );
        assert_eq!(
            gather(&candidate(Node::Compute(&compute))),
            Some(RelatedPtRows::NotFolded)
        );
    }

    #[test]
    fn a_surplus_loop_around_the_node_takes_a_temporal_level_off_the_allocations_arrangement() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, Vec::new()));
        let core_ds = RowSplitOn(Vec::new());
        let distribution = Seams(vec![level(2, 0, 4, None)]);
        let levels = ref_levels();
        let slices = WorkSlices::default();
        let alloc = AllocationFold {
            lds: LdsIdx(0),
            dims: vec![(
                PrimaryDim::Out,
                AllocCoordinates {
                    allocate: CoordDim {
                        folds: Some(&levels),
                        spatial: SpatialFolds(3),
                        temporal: TemporalFolds(0),
                        pad: PadType::NoPad,
                    },
                    slice_view: None,
                },
            )],
            work_slices: &slices,
            dim_scales: BTreeMap::new(),
            corelet_split_dim: None,
            components: RefComponents {
                size: SenComponent::All,
                prop: SenComponent::Lx,
            },
            row_group: None,
        };
        let loop_node = loop_named("loop0");
        let chain = [LoopAndDim {
            loop_node: &loop_node,
            dim: PrimaryDimAndKind {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::Unpadded,
            },
            distribution: LoopDistribution::AboveChunk,
        }];
        let mut coord = dsc2::Coordinate::default();

        assert_eq!(
            build_fold_from_allocation::<Target, _, _, _, _>(
                &dsc,
                &core_ds,
                &distribution,
                &alloc,
                &AllocFoldTarget {
                    node: NodeId(1),
                    loops: &chain,
                },
                &mut coord,
                &mut (),
            ),
            FoldPropagation::Built
        );
        assert_eq!(
            folds_of(&coord, PrimaryDim::Out),
            vec![
                ("core_workslice_fold_dim".to_owned(), 100, 2),
                ("corelet_fold_dim".to_owned(), 50, 2),
                ("rowsplit_fold".to_owned(), 0, 1),
                // The surplus loop's own temporal level, at the stride the distributor decided.
                ("loop0 out".to_owned(), 16, 4),
                ("elem_arr_0".to_owned(), 2, 4),
            ]
        );
        assert!(coord.fold_constructed());
    }

    #[test]
    fn a_loop_the_reference_does_not_share_is_distributed_and_the_common_ancestor_is_not() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, Vec::new()));
        let core_ds = RowSplitOn(Vec::new());
        let distribution = Seams(vec![level(2, 0, 4, None)]);
        let levels = ref_levels();
        let (outer, inner) = (loop_named("loopA"), loop_named("loopB"));
        let reference = NonAllocReference {
            lds: LdsIdx(0),
            dims: vec![(
                PrimaryDim::Out,
                CoordDim {
                    folds: Some(&levels),
                    spatial: SpatialFolds(3),
                    temporal: TemporalFolds(0),
                    pad: PadType::NoPad,
                },
            )],
            dim_scales: BTreeMap::new(),
            // The reference sits directly under the shared ancestor.
            loops: &[&outer],
            components: RefComponents {
                size: SenComponent::All,
                prop: SenComponent::Lx,
            },
            row_group: None,
        };
        let working = [
            (&inner, LoopDistribution::AboveChunk),
            (&outer, LoopDistribution::AboveChunk),
        ];
        let mut coord = dsc2::Coordinate::default();

        assert_eq!(
            build_fold_from_non_alloc_ref::<Target, _, _, _, _>(
                &dsc,
                &core_ds,
                &distribution,
                &reference,
                &NonAllocTarget {
                    node: NodeId(1),
                    kind: FoldTargetKind::Compute,
                    loops: &working,
                    row_bundling: &[],
                },
                &mut coord,
                &mut (),
            ),
            FoldPropagation::Built
        );
        assert_eq!(
            folds_of(&coord, PrimaryDim::Out),
            vec![
                ("core_workslice_fold_dim".to_owned(), 100, 2),
                ("corelet_fold_dim".to_owned(), 50, 2),
                ("rowsplit_fold".to_owned(), 0, 1),
                // Only the loop below the common ancestor was distributed.
                ("loopB out".to_owned(), 16, 4),
                ("elem_arr_0".to_owned(), 2, 4),
            ]
        );
    }
    /// The three rowsplit levels entry 337's two single-row arms read: alpha 32 and beta 7 at
    /// [`FoldPosition::RowSplit`], stepping ONCE so the row-to-same-row arm takes the beta.
    fn one_step_row_split() -> Levels {
        Levels(vec![
            level(100, 0, 2, Some(FoldLabel::CoreWorksliceFoldDim)),
            level(50, 0, 2, Some(FoldLabel::CoreletFoldDim)),
            level(32, 7, 1, Some(FoldLabel::RowSplitFold)),
        ])
    }

    fn on_row(unit: SenComponent) -> Vec<UnitStream> {
        vec![UnitStream {
            unit,
            stream: matched_stream(),
        }]
    }

    #[test]
    fn ptxrf_to_a_row_offsets_by_alpha_where_the_same_row_arm_takes_the_rowsplit_beta() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::In]));
        let core_ds = RowSplitOn(vec![PrimaryDim::Out]);
        let tree = Tree(vec![None]);
        let levels = one_step_row_split();
        let slices = WorkSlices::default();
        let coord = RowCoordinate {
            dims: BTreeMap::from([(PrimaryDim::Out, &levels)]),
            work_slices: &slices,
        };
        let no_candidates: RowCandidates<'_, Levels> = RowCandidates {
            forward: &[],
            reverse: &[],
        };
        let gather = |units| {
            gather_related_pt_rows::<Target, _, _, _, Levels>(
                &dsc,
                &core_ds,
                &tree,
                &propagation(),
                units,
                NodeId(0),
                &coord,
                crate::schedule::ddc::ScaleDown::No,
                &no_candidates,
                &slices,
            )
        };
        let row3 = comp_row_id(SenComponent::Ptrow3);
        let on_ptrow3 = on_row(SenComponent::Ptrow3);
        let on_ptxrf = on_row(SenComponent::Ptxrf);

        // Row 3 producing for row 3: the rowsplit fold steps once, so its own beta stands.
        let same_row = PtRowPropagation::of(
            &dsc,
            &propagation(),
            RefRole::Producer,
            &PropUnits::Other,
            &PropUnits::Compute {
                ex_unit: SenComponent::Ptrow3,
                inputs: &on_ptrow3,
                outputs: &[],
            },
        );
        assert_eq!(
            gather(same_row.expect("both units resolve to row 3")),
            Some(PtRowGrouping::SameRow(
                row3.expect("ptrow3 is a row"),
                RowGroupNode {
                    node: NodeId(0),
                    row: row3,
                    beta: Beta(7),
                },
            ))
        );

        // PTXRF consumed by row 3: the same category, but alpha times the row REGARDLESS.
        let from_ptxrf = PtRowPropagation::of(
            &dsc,
            &propagation(),
            RefRole::Consumer,
            &PropUnits::Other,
            &PropUnits::Compute {
                ex_unit: SenComponent::Ptrow3,
                inputs: &[],
                outputs: &on_ptxrf,
            },
        );
        assert_eq!(
            gather(from_ptxrf.expect("PTXRF is the source unit and row 3 the destination")),
            Some(PtRowGrouping::SameRow(
                row3.expect("ptrow3 is a row"),
                RowGroupNode {
                    node: NodeId(0),
                    row: row3,
                    beta: Beta(32 * 3),
                },
            ))
        );
    }
}
