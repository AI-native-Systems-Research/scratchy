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
use crate::schedule::ddc::{CoordPropTracker, NodeNames, Propagation, QueuedProp, RefRole};
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind, Metadata, NodeIndex};
use crate::schedule::ddc::transformation::{DsType, Scale};
use crate::schedule::ddc::transformation_util::{
    LoopNode, PaddingForm, PrimaryDimAndKind, is_memory,
};
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
pub fn need_to_consider_row_bundling<S: Stage + ?Sized>(
    core_ds: &S,
    working_dims: &[PrimaryDim],
) -> bool {
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

impl NodeKind {
    /// `ScheduleNode::nodeTypeToString.at(nodeType_)` (`dsc/dsc2.cpp:1879-1888`) — this enum's name
    /// lowered, for every kind it spells.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Loop => "loop",
            Self::Transfer => "transfer",
            Self::Compute => "compute",
            Self::Sync => "sync",
            Self::Condition => "condition",
            Self::Allocate => "allocate",
            Self::StickMask => "stickmask",
        }
    }
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

    /// The read-back of [`Self::spelling`] — `None` for the open spellings the propagation units
    /// write (`"elem_arr_<n>"`, `"<loop> <dim>"`), which no closed label can carry.
    #[must_use]
    pub fn of_spelling(text: &str) -> Option<Self> {
        match text {
            "corelet_fold_dim" => Some(Self::CoreletFoldDim),
            "core_workslice_fold_dim" => Some(Self::CoreWorksliceFoldDim),
            "elem_arr_layout_split" => Some(Self::ElemArrLayoutSplit),
            "rowsplit_fold" => Some(Self::RowSplitFold),
            "elem_arr_scaleup" => Some(Self::ElemArrScaleUp),
            _ => None,
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

    /// `loopParamInfo[loopNode][dim] = {alpha, beta, relatedElemArrLevel}`
    /// (`ddc/ddc_fold.cpp:2811-2817`) — the link entry 356 records ITSELF, because no distribution
    /// runs for a propagation onto an allocation and so nothing else would record one.
    fn record_loop_level(
        &mut self,
        loop_node: &LoopNode,
        dim: PrimaryDim,
        decided: DistributedLoop,
        level: ElemArrLevel,
    );
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
    /// `currDsc->numCoreletsUsed_DSC2_` (`dsc/designSpaceConfig.h:104`) — the DM-filled copy entry
    /// 298 reads (`ddc_fold.cpp:3228`), which `prepDsc` assigns from `numCoreletsUsed_`
    /// (`ddcv1.cpp:2024`), so the two agree today but only one of them is this reference's.
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
mod tests_node_kind {
    use super::NodeKind;

    /// ⭐⭐ EVERY `nodeType_` SPELLING IS THE REFERENCE'S OWN TABLE — `ScheduleNode::nodeTypeToString`
    /// (`dsc/dsc2.cpp:1879-1888`), transcribed as VALUES.
    ///
    /// ⛔ THE EIGHT ARE ALSO A CLOSED SET IN `NodeType`'s DECLARATION ORDER (`dsc/dsc2.h:446-456`:
    /// `INVALID`, `BLOCK`, `LOOP`, `TRANSFER`, `COMPUTE`, `SYNC`, `CONDITION`, `ALLOCATE`, `STICKMASK`)
    /// less `INVALID`, which the reference calls a sentinel and which has no variant here. So the
    /// ORDER is asserted too, not just the membership: a variant inserted in the wrong place would
    /// still spell correctly and still be wrong to anything comparing `as u32`.
    ///
    /// ⛔ WHY THIS EXISTS NOW: `Compute` and `StickMask` were the two kinds
    /// [`crate::schedule::stages::Kind`] could not hold, and nothing pinned their spellings — a
    /// `"stick_mask"` or a `"COMPUTE"` here would reach the emitted json and nothing would say so.
    #[test]
    fn every_node_kind_spells_what_the_reference_spells_in_the_references_own_order() {
        assert_eq!(
            [
                NodeKind::Block,
                NodeKind::Loop,
                NodeKind::Transfer,
                NodeKind::Compute,
                NodeKind::Sync,
                NodeKind::Condition,
                NodeKind::Allocate,
                NodeKind::StickMask,
            ]
            .map(NodeKind::spelling),
            [
                "block",
                "loop",
                "transfer",
                "compute",
                "sync",
                "condition",
                "allocate",
                "stickmask",
            ],
            "`ScheduleNode::nodeTypeToString` (`dsc/dsc2.cpp:1879-1888`), in `NodeType`'s own \
             declaration order less the `INVALID` sentinel"
        );
    }
}

#[cfg(test)]
mod tests_e086_e093 {
    use super::{
        AllocId, AllocLayout, Allocations, Alpha, Beta, BlockId, Cardinality, CoordPropInfo,
        Coordinate, CoordinateCategory, CoreStage, DataOrigin, DataStream, DistributedLoop,
        ElemArrFolds, ElemArrLevel, Fold, FoldLabel, FoldParamInfo, Incoming, IncomingStreams,
        LdsIdx, LoopLevels, LoopNode, NodeId, NodeKind, PadType, PropEnd, ScaledLds, ScheduleTree,
        SizeStage, StoredStream, Stride, build_spatial_fold, combine_contigous_levels,
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

        fn record_loop_level(
            &mut self,
            _loop_node: &LoopNode,
            _dim: PrimaryDim,
            _decided: DistributedLoop,
            _level: ElemArrLevel,
        ) {
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

/// ⭐ THE READ SIDE OF THE SEAM MEETS THE CONCRETE COORDINATE, as `impl Coordinate for
/// dsc2::Coordinate` is the write side: the fold builders gather their params off the coordinate the
/// schedule tree actually carries.
///
/// ⛔ TOTAL WHERE `getFoldDimSize(pos)` THROWS: an index past the last level answers
/// `FoldParamInfoType`'s own defaults, and [`AffineFoldDims::num_dims`] is the only bound any caller
/// here derives an index from.
/// ⚠️ AN OPEN LABEL READS BACK AS [`None`] — the closed set cannot spell `"elem_arr_<c>"` or
/// `"corelet_fold"`, which is why [`PropagatedLevel`] carries the string beside it.
impl AffineFoldDims for dsc2::FoldDim {
    fn num_dims(&self) -> usize {
        self.folds().count()
    }

    fn alpha_beta(&self, dim: usize) -> (Alpha, Beta) {
        self.folds().nth(dim).map_or((Alpha(1), Beta(0)), |fold| {
            (Alpha(fold.alpha.0), Beta(fold.beta.0))
        })
    }

    fn dim_size(&self, dim: usize) -> Cardinality {
        self.folds()
            .nth(dim)
            .map_or(Cardinality(1), |fold| Cardinality(fold.cardinality.0.into()))
    }

    fn dim_label(&self, dim: usize) -> Option<FoldLabel> {
        self.folds()
            .nth(dim)
            .and_then(|fold| FoldLabel::of_spelling(&fold.label.0))
    }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    coord.set_padding_form(alloc.padding.clone());

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
    /// `coordPropInfo.dimsToPropagate` IN FULL (`dsc/dsc2.h:1094`) — ⛔ NOT the dims of
    /// [`Self::dims`], which is that list INTERSECTED with the coordinate: the corelet-split stop
    /// (`:8429`) asks the REQUEST, so a split dim asked for but carrying no fold still stops it.
    pub propagated_dims: &'a [PrimaryDim],
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
        if alloc.propagated_dims.contains(&split)
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
/// with it (`ddc/ddc_fold.cpp:1024-1033`, `:1038-1047`).
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
/// ⛔⛔ EVERY ABORT OF THE 100-LINE RESOLUTION IS THIS WITNESS MISSING: all six
/// `DT_CHECK(prop*Unit != NO_COMPONENT)` (`:999`, `:1035`, `:1066`, `:1084`, `:1137`, `:1151`) and
/// *"[gatherRelatedPTRows] Unsupported reference node for propagation"* for an allocation folded
/// against neither a transfer nor a compute (`:1154`). ⛔ THE ADJACENCY CHECK (`:1184`) IS NOT ONE OF
/// THEM: the reference asks it inside its both-rows arm, past the `rowSplit_` early return.
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
        // ⛔ THE FIRST MATCH'S UNIT AND NOTHING FURTHER: the reference `break`s on the match and
        // THEN checks the unit, so a matched operand sitting on `NO_COMPONENT` is one of the six
        // `DT_CHECK`s and not a reason to keep scanning.
        let matched = |operands: &[UnitStream], check_ref: bool| {
            operands
                .iter()
                .find(|operand| match_data_stream(dsc, prop, operand.stream, check_ref))
                .map(|operand| operand.unit)
                .filter(|&unit| unit != none)
        };
        let matched_dst = |dsts: &TransferDsts, check_ref: bool| {
            dsts.iter()
                .find(|dst| match_data_stream(dsc, prop, dst.stream, check_ref))
                .map(|dst| dst.unit)
                .filter(|&unit| unit != none)
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
        Some(Self { src, dest })
    }
}

/// THE SAME NODES SCANNED FROM BOTH SIDES — entry 337's unbundling arm hands the base a propagation
/// with `refIsProducer` NEGATED and both ends swapped (`ddc/ddc_fold.cpp:1222-1226`), and the role
/// is exactly what decides which side of a candidate is scanned.
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
    let base = |prop: &CoordPropInfo,
                scaled: crate::schedule::ddc::ScaleDown,
                candidates: &[RowCandidate<'_, F>]| {
        gather_related_pt_rows_base::<A, D, S, T, F>(
            dsc, core_ds, tree, prop, scaled, candidates, sdsc_slices,
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
        (Some(_), None) => bundled(
            RowGroupCategory::RowToNonRow,
            base(prop, scale_down, candidates.forward),
        ),
        // No need to collect nodes for all rows: the reference node's own row is the group.
        (Some(row), Some(dest)) if row == dest => single_row(row, false),
        // `DT_CHECK(abs(refRowId - getCompRowId(propDestUnit)) <= 1)` (`:1184`) — asked ONLY here,
        // so a propagation between two non-adjacent rows with nothing row split answers above.
        (Some(row), Some(dest)) => {
            if !row.is_adjacent_or_same(dest) {
                return None;
            }
            bundled(
                RowGroupCategory::RowNorthSouth,
                base(prop, scale_down, candidates.forward),
            )
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
            // ⛔ A FRESH `CoordPropInfoType`, so the base reads the DEFAULT `scaleDown`
            // (`dsc/dsc2.h:1095`) and never this propagation's.
            bundled(
                RowGroupCategory::NonRowToRow,
                base(
                    &reversed,
                    crate::schedule::ddc::ScaleDown::No,
                    candidates.reverse,
                ),
            )
        }
        // Both absent is the early answer above, so the reference's chain has no fallthrough.
        (None, None) => Some(PtRowGrouping::NoBundling),
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE THREE FOLD BUILDERS' SHARED VOCABULARY — entries 356, 357 and 358 each dispatch on the
// reference node's kind, and every one of the nine arms opens with the SAME row-bundling gate.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// A NODE'S ENCLOSING LOOPS — `getEnclosingLoops(node)`, INNERMOST FIRST and THE NODE ITSELF FIRST
/// when it is a loop, which is what [`NonAllocTarget::row_bundling`] reads (`:8768-8778`).
///
/// ⛔ A CAPABILITY AND NOT A WITNESS FIELD: the node asked about is
/// `refRowGroup.commonGroupAncestor`, which the row-bundling gate only discovers mid-call.
pub trait EnclosingLoops {
    /// The loops enclosing a node, innermost first.
    fn enclosing_loops(&self, node: NodeId) -> Vec<&LoopNode>;

    /// `getEnclosingLoopsAndRelatedDims(currDsc, node, chain, loopsBelowChunkBoundary,
    /// buildCoreletFold)` — the same walk with each loop's dim and distribution tag, which is what
    /// [`TemporalLoopDistribution::related_loops`] then filters per dim.
    ///
    /// ⛔ THE CORELET-FOLD REQUEST IS AN ARGUMENT AND NOT A FIELD: entry 356 decides it from the
    /// allocation's own external-node state and work slices, and it is the only thing that puts a
    /// [`LoopDistribution::CoreletSlice`] entry in the chain.
    fn enclosing_loop_chain(&self, node: NodeId, build_corelet_fold: bool) -> Vec<LoopAndDim<'_>>;
}

/// `dsc2::computeLoopElemOffsetsFromCoordinates(currDsc, node, coord, refAllocNode, loopParams,
/// dimsToPropagate, ..)` — the one effect entries 356-358's ALLOCATE arms perform besides the fold.
///
/// ⛔ A CAPABILITY AND NOT A DROP. It writes each loop's element offset back onto the node, so a
/// port that omitted it would be the documented-predicate failure this campaign forbids; it is not a
/// unit of this campaign, so the effect is declared here and supplied from outside.
pub trait LoopElemOffsets {
    /// Records the loop element offsets a node's freshly built coordinate implies, against the
    /// allocation it was propagated from.
    fn compute_loop_elem_offsets(
        &mut self,
        node: NodeId,
        coord: &dsc2::Coordinate,
        alloc: NodeId,
        dims: &[PrimaryDim],
    );
}

/// THE AMBIENT STATE THE THREE FOLD BUILDERS READ — `Ddc`'s own members, which the reference reaches
/// through `this` and a port has to be handed.
pub struct FoldContext<'a, D, S: ?Sized, T: ?Sized, L: ?Sized, E: ?Sized, N: ?Sized> {
    /// `currDsc`.
    pub dsc: &'a D,
    /// `currDsc->dataStageParam_.at(metadata.core_dstgid)`.
    pub core_ds: &'a S,
    /// The schedule tree the row walk descends.
    pub tree: &'a T,
    /// What decides whether a loop is relevant to a dim, and how a slice is distributed.
    pub distribution: &'a L,
    /// A node's enclosing loop chain.
    pub ancestors: &'a E,
    /// `name_`, which only the retry's threshold message reads.
    pub names: &'a N,
    /// `metadata`.
    pub meta: &'a Metadata,
}

/// `refRowGroup` AS THE TWO PROPAGATORS TAKE IT — the group entry 337 found with the
/// `commonGroupAncestor` beside it, which [`NonAllocTarget::row_bundling`] is derived from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefRowGroup {
    /// `nodeInfo` with its category, absent for the empty group [`RowGroup`] cannot spell.
    pub group: Option<RowGroup>,
    /// `commonGroupAncestor`, absent where no group was collected at all.
    pub ancestor: Option<NodeId>,
}

/// `refRowGroup` from what entry 337 answered.
///
/// ⛔ AN EMPTY GROUP IS AN ABSENT ONE, as [`RowGroup`]'s own note says: the reference's
/// default-constructed `RowGroupInfo` and one whose `nodeInfo` was replaced by an empty list reach
/// the two propagators identically, and both read `nodeInfo.at(0)` before any category test.
fn ref_row_group(found: PtRowGrouping) -> RefRowGroup {
    match found {
        PtRowGrouping::NoBundling => RefRowGroup::default(),
        PtRowGrouping::SameRow(_, node) => RefRowGroup {
            group: Some(RowGroup::new(
                RowGroupCategory::RowToSameRow,
                node,
                Vec::new(),
            )),
            ancestor: None,
        },
        PtRowGrouping::Bundled(
            cat,
            RelatedPtRows::Group {
                nodes,
                common_ancestor,
            },
        ) => {
            let mut nodes = nodes.into_iter();
            RefRowGroup {
                group: nodes
                    .next()
                    .map(|first| RowGroup::new(cat, first, nodes.collect())),
                ancestor: Some(common_ancestor),
            }
        }
        // `NO_BUNDLING` overwrites the category and the other two leave `nodeInfo` empty.
        PtRowGrouping::Bundled(..) => RefRowGroup::default(),
    }
}

/// THE ROW-BUNDLING SCAN'S ARGUMENTS THAT DO NOT DEPEND ON A COORDINATE — everything
/// `gatherRelatedPTRows(refRowGroup, coordPropInfo, <a coordinate>)` needs besides that coordinate.
///
/// ⛔ APART FROM [`RowBundling`] BECAUSE ENTRY 356 PRODUCES ITS OWN GATE COORDINATE: its reference
/// coordinate is a local copy the scale-down rewrites, so no caller can hand one over.
pub struct RowBundlingScan<'a, F: AffineFoldDims + ?Sized> {
    /// `propSrcUnit`/`propDestUnit`, resolved off both ends by [`PtRowPropagation::of`].
    pub units: PtRowPropagation,
    /// `coordPropInfo.refNode`'s tree identity.
    pub ref_node: NodeId,
    /// The nodes the reference's DFS offered, scanned from both sides.
    pub candidates: RowCandidates<'a, F>,
    /// `sdscWkSlices`.
    pub sdsc_slices: &'a WorkSlices,
}

/// THE ROW-BUNDLING GATE'S ARGUMENTS — `needToConsiderRowBundling(coreDs, <a coordinate>,
/// dimsToPropagate) && !gatherRelatedPTRows(refRowGroup, coordPropInfo, <that coordinate>)`, spelled
/// identically in all nine of entries 356-358's dispatch arms.
///
/// ⛔ TWO COORDINATES, NOT ONE, AND A PORT CANNOT FUSE THEM: the gate asks `effectiveRefCoord` while
/// the `retry` beside it charges `refAllocNode->allocateCoordinates_.getTensorDims()`
/// (`:13972-13981`). ⭐ Entry 085 never reads the coordinate it is handed, so only the RETRY's dims
/// are a field here.
pub struct RowBundling<'a, F: AffineFoldDims + ?Sized> {
    /// Everything the scan needs that the coordinate does not decide.
    pub scan: RowBundlingScan<'a, F>,
    /// `effectiveRefCoord` / `refTransferNode->transferCoordinates_` / `refCoordinate`, as the arm
    /// that has already chosen states it — for entry 358's ALLOCATE arm, `allocateCoordinates_`.
    pub ref_coord: RowCoordinate<'a, F>,
    /// `refAllocNode->sliceViewCoordinates_` where it `foldConstructed()` — ⛔ ENTRY 358'S ALLOCATE
    /// ARM SELECTS `effectiveRefCoord` OFF ITS OWN `sizeRefComp` (`:13966-13970`), which the caller
    /// cannot derive without redoing that unit's two overrides, so there the choice is the unit's
    /// and this is the alternative. Every other arm has chosen already and leaves this [`None`].
    pub ref_slice_view: Option<RowCoordinate<'a, F>>,
    /// `getTensorDims()` OF THE COORDINATE THE RETRY NAMES, which is not always the gate's.
    pub retry_dims: &'a [PrimaryDim],
}

/// The gate, and the retry it charges where the group cannot be collected. [`None`] IS THE CALLER'S
/// `return false` — the propagation has already been re-queued by then.
fn gated_row_group<A, D, S, T, N, F>(
    dsc: &D,
    core_ds: &S,
    tree: &T,
    names: &N,
    tracker: &mut CoordPropTracker,
    prop: &Propagation,
    dims_to_propagate: &[PrimaryDim],
    scan: &RowBundlingScan<'_, F>,
    ref_coord: &RowCoordinate<'_, F>,
    retry_dims: &[PrimaryDim],
) -> Option<RefRowGroup>
where
    A: Arch,
    D: Dsc + Allocations,
    S: Stage + ?Sized,
    T: ScheduleTree + ?Sized,
    N: NodeNames + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    if !need_to_consider_row_bundling(core_ds, dims_to_propagate) {
        // `refRowGroup` stays default-constructed; the gate never writes it.
        return Some(RefRowGroup::default());
    }
    match gather_related_pt_rows::<A, D, S, T, F>(
        dsc,
        core_ds,
        tree,
        &prop.ends,
        scan.units,
        scan.ref_node,
        ref_coord,
        prop.scale_down,
        &scan.candidates,
        scan.sdsc_slices,
    ) {
        Some(found) => Some(ref_row_group(found)),
        None => {
            // Could not collect PT-row group. Schedule the propagation to retry later.
            tracker.retry(names, *prop, retry_dims);
            None
        }
    }
}

/// `propRefComp` FOR A NON-ALLOCATION REFERENCE — ⛔ THE TWO KINDS SPELL IT DIFFERENTLY, which is
/// the only thing past the coordinate selection that distinguishes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonAllocPropComp {
    /// `TRANSFER` — BOTH ends of the reference transfer. Entry 358 names its OWN source where the
    /// reference produces and `src` only where it consumes (`:14025-14026`); entry 357, having no
    /// source of its own, names `dst` when the reference produces and reads both to find a PT row
    /// (`:13155-13170`).
    TransferEnds {
        /// `refTransferNode->src_.unit_`.
        src: SenComponent,
        /// `refTransferNode->dstVias_.at(0).loc_.unit_`.
        dst: SenComponent,
    },
    /// `COMPUTE` — `refComputeNode->exUnit_`, whatever the role (`:14052`).
    ComputeUnit(SenComponent),
}

/// WHICH KIND OF REFERENCE THE PROPAGATION COMES FROM, pre-resolved except for `sizeRefComp`,
/// `propRefComp` and `refRowGroup`, which the fold builder itself derives.
pub enum FoldReferenceKind<'a, 'l, F: AffineFoldDims + ?Sized> {
    /// `ALLOCATE`.
    Allocate {
        /// `refAllocNode`'s tree identity.
        node: NodeId,
        /// `refAllocNode->ldsIdx_`.
        lds: LdsIdx,
        /// `refAllocNode->component_`, which entry 357 kludges to the compute node's own unit for
        /// a PT-row reference (`:13117-13125`).
        component: SenComponent,
        /// `effectiveRefCoord.coordinates_` ∩ `dimsToPropagate` with BOTH of the allocation's
        /// coordinates per dim — [`AllocCoordinates::effective`] is entry 298's own choice off
        /// `sizeRefComp`, so nothing here selects the slice view.
        dims: Vec<(PrimaryDim, AllocCoordinates<'a, F>)>,
        /// `allocateCoordinates_.coreIdToWkSlice_`.
        work_slices: &'a WorkSlices,
        /// `labeledDs_.at(ldsIdx_).scale_` per dim.
        dim_scales: BTreeMap<PrimaryDim, Scale>,
        /// `coreletSplitDim`.
        corelet_split_dim: Option<PrimaryDim>,
    },
    /// `TRANSFER` or `COMPUTE` — ONE arm, because everything the propagator needs past the
    /// coordinate selection is shared and [`NonAllocPropComp`] carries the one difference.
    ///
    /// ⭐ `getRelatedComputeCoord` (entry 238) IS THE CALLER'S: in this unit its `refInputPos`,
    /// `refOutputPos` and `selectedLdsIdx` are all written and never read, so it serves only to
    /// reach the reference coordinate, and reaching an operand is the droppable mechanism.
    NonAlloc {
        /// `refNode`'s tree identity.
        node: NodeId,
        /// `refCoordinate.coordinates_` ∩ `dimsToPropagate`, each with its counts and padding.
        dims: Vec<(PrimaryDim, CoordDim<'a, F>)>,
        /// `refLds.scale_` per dim.
        dim_scales: BTreeMap<PrimaryDim, Scale>,
        /// `refNode`'s enclosing loops, INNERMOST FIRST and unfiltered.
        loops: &'a [&'l LoopNode],
        /// `propRefComp` as this kind spells it.
        prop_comp: NonAllocPropComp,
    },
}

/// THE REFERENCE SIDE OF ONE PROPAGATION — its kind and the gate its arm opens with.
pub struct FoldReference<'a, 'l, F: AffineFoldDims + ?Sized> {
    /// The row-bundling gate.
    pub gate: RowBundling<'a, F>,
    /// The kind.
    pub kind: FoldReferenceKind<'a, 'l, F>,
    /// `refCoordinate.getPadding()` — the reference coordinate's WHOLE padding form, which entry
    /// 357 copies onto the coordinate it folds before dispatching (`:13126`, `:13154`, `:13202`).
    ///
    /// ⛔ THE ALLOCATE ARM NAMES `allocateCoordinates_.getPadding()` and NOT the slice view its own
    /// gate may have selected.
    pub ref_padding: &'a PaddingForm,
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ENTRY 356 — BUILDING AN ALLOCATION'S FOLD.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE DIM OF A CONCRETE COORDINATE AS THE FOLD SEAMS TAKE IT — the read side of
/// `impl Coordinate for dsc2::Coordinate`, for the unit that holds a real coordinate rather than a
/// caller-supplied witness. An uncovered dim is the absent fold manager [`CoordDim`] already spells.
fn coord_dim_of(coord: &dsc2::Coordinate, dim: PrimaryDim) -> CoordDim<'_, dsc2::FoldDim> {
    let folds = coord.fold_dim(dim);
    CoordDim {
        folds,
        spatial: SpatialFolds(folds.map_or(0, dsc2::FoldDim::spatial_folds)),
        temporal: TemporalFolds(folds.map_or(0, dsc2::FoldDim::temporal_folds)),
        pad: coord.padding(dim),
    }
}

/// [`gather_fold_params`] BESIDE THE OPEN LABEL EACH LEVEL ACTUALLY CARRIES — entry 095's answer
/// zipped with the `foldDimLabel` strings, because entry 356 hands the label it read to `addFold`
/// verbatim and the closed [`FoldLabel`] set cannot spell `"elem_arr_<c>"` or `"corelet_fold"`.
fn gather_propagated_levels(folds: &dsc2::FoldDim) -> Vec<PropagatedLevel> {
    gather_fold_params(folds)
        .into_iter()
        .zip(folds.folds())
        .map(|(level, fold)| PropagatedLevel {
            level,
            label: fold.label.0.clone(),
        })
        .collect()
}

/// The construction tail entry 356 ends each dim with (`:12883-12895`, `:12934-12946`): every level
/// re-added at the FRONT so the list lands outermost first, categorised by where it sits.
///
/// ⛔ NOT [`add_propagated_folds`], WHICH RELABELS THE ELEMENT ARRANGEMENT: this unit passes each
/// level's OWN label to the add site, having relabelled the elem-arr levels earlier and on ONE
/// branch only (`:12823-12826`), so relabelling again here would rename the levels it did not.
fn add_carried_folds(
    coord: &mut dsc2::Coordinate,
    dim: PrimaryDim,
    levels: &[PropagatedLevel],
    spatial_ends: i64,
    temporal_ends: i64,
) {
    for (at, level) in levels.iter().enumerate().rev() {
        let pos = i64::try_from(at).unwrap_or(i64::MAX);
        let category = if pos > temporal_ends {
            CoordinateCategory::ElemArr
        } else if pos > spatial_ends {
            CoordinateCategory::Temporal
        } else {
            CoordinateCategory::Spatial
        };
        add_labelled_fold(coord, dim, category, level.label.clone(), level.level);
    }
}

/// THE MX SCALE PAIR AN L0 VALUE ALLOCATION IS BUILT FROM — the value tensor and BOTH coordinates of
/// the allocation `getScaleAllocation(currDsc, allocNode)` finds behind it (`:12578`).
///
/// ⭐ THE LOOKUP IS DROPPED, NOT ITS ANSWER; [`MxValueTensor::of`] already carries the `VALUE_TENSOR`
/// test, so only the `component_ == L0` half of the arm is left to spell in code.
pub struct AllocScaleUp<'a> {
    /// `allocLds.mxInfo_` as entry 234 takes it.
    pub value: MxValueTensor,
    /// `scaleAlloc->allocateCoordinates_`.
    pub allocate: &'a dsc2::Coordinate,
    /// `scaleAlloc->sliceViewCoordinates_`.
    pub slice_view: &'a dsc2::Coordinate,
}

/// THE CONSISTENCY CHECK ENTRY 356 RUNS FOR A DIM THE ALLOCATION ALREADY COVERS (`:12952-12974`).
///
/// ⛔ PRESENT EXACTLY WHEN `refNode->nodeType_ == TRANSFER` AND ITS `transferCoordinates_
/// .foldConstructed()` — the reference's two guards reach the same skip, and neither can be asked of
/// a transfer coordinate the caller has not resolved.
pub struct AllocFoldConsistency<'a> {
    /// `transferNode` with `transferCoordinates_`, its work slices, and the cores of the unit
    /// `refIsProducer` picks — `dstVias_.at(0).loc_.unit_` or `src_.unit_` (`:12960-12962`).
    pub transfer: CoordSide<'a, dsc2::FoldDim>,
    /// `allocNode->getRelevantCoreCl(comp)` for that same unit.
    pub alloc_cores: BTreeMap<CoreOrdinal, BTreeSet<CoreletOrdinal>>,
    /// `sdscWkSlices`.
    pub sdsc_slices: &'a WorkSlices,
    /// The broadcast dims of `allocNode->ldsIdx_`, which entry 237 excuses.
    pub broadcast_dims: &'a BTreeSet<PrimaryDim>,
}

/// THE ALLOCATION ENTRY 356 BUILDS THE FOLD FOR — `allocNode` plus the `Ddc` members keyed on it.
///
/// ⛔ ITS `ldsIdx_ < 0` RETURN (`:12568`) IS DISCHARGED BY TYPE, as [`BaseAllocation`]'s is.
pub struct AllocationFoldTarget<'a, Z: SizeStage + ?Sized> {
    /// `allocNode`'s tree identity.
    pub node: NodeId,
    /// `ldsIdx_`.
    pub lds: LdsIdx,
    /// `component_`.
    pub component: SenComponent,
    /// `layoutDimOrder_` zipped with `maxDimSizes_` — BOTH the walk order (`:12638`) and entry 089's
    /// own cap list, which is why one value serves.
    pub layout: &'a AllocLayout,
    /// `padding_`.
    pub padding: &'a PaddingForm,
    /// `allocateCoordinates_.coreIdToWkSlice_`.
    pub work_slices: &'a WorkSlices,
    /// `labeledDs_.at(ldsIdx_).scale_` per dim, absent where the layout order does not name the dim
    /// — which the reference reads as a scale of 1 (`:12663`).
    pub dim_scales: BTreeMap<PrimaryDim, Scale>,
    /// The scale pair, present only for an MX value tensor.
    pub scale_up: Option<AllocScaleUp<'a>>,
    /// `labeledDs_.at(ldsIdx_).mxInfo_` where the allocation is an MX SCALE tensor, the only case
    /// `scaleDownCoord(.., allocNode->ldsIdx_)` (`:12530`) does anything for.
    pub scale_down: Option<MxScaleTensor>,
    /// The stick dims of `ldsIdx_`, which entry 092 measures one PT slice over.
    pub stick_dims: &'a StickDims,
    /// `coreletSplitDim`, absent for its `PrimaryDimTypesCount`.
    pub corelet_split_dim: Option<PrimaryDim>,
    /// `getSizeDataStageForNode(allocNode, allocNode).ss_`, as entry 089 reads it.
    pub size_stage: &'a Z,
    /// `refNode`'s kind, which entry 241 derives both components from.
    pub ref_kind: RefNode<'a>,
    /// The consistency check, where there is one to run.
    pub consistency: Option<AllocFoldConsistency<'a>>,
}

/// Replaces: e356_buildFoldForAllocation
///
/// Propagates a transfer's or a compute's coordinate ONTO an allocation: per layout dim the
/// reference's own levels with the row split recomputed and the surplus temporal levels turned into
/// element arrangement, added to both the allocate and the slice-view coordinate.
///
/// ⛔ THE REFERENCE COORDINATE IS A LOCAL COPY THE SCALE-DOWN REWRITES (`:12527-12534`), so this
/// takes the coordinate itself and not a witness over it — a borrow taken before the rewrite reads
/// the stale folds. ⛔ FIVE `panic!`s CARRY FIVE REFERENCE ABORTS; none of them is a refusal.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "the reference reads eleven `Ddc` members and a port is handed each one"
)]
pub fn build_fold_for_allocation<A, D, S, T, L, E, N, Z>(
    ctx: &FoldContext<'_, D, S, T, L, E, N>,
    alloc: &AllocationFoldTarget<'_, Z>,
    prop: &Propagation,
    dims_to_propagate: &[PrimaryDim],
    gate: &RowBundlingScan<'_, dsc2::FoldDim>,
    input_ref_coord: &dsc2::Coordinate,
    input_ref_slices: &WorkSlices,
    coord: &mut dsc2::Coordinate,
    slice_view: &mut dsc2::Coordinate,
    tracker: &mut CoordPropTracker,
    loop_params: &mut L::LoopParams,
) -> bool
where
    A: Arch,
    D: Dsc + Allocations,
    S: Stage + ?Sized,
    T: ScheduleTree + ?Sized,
    L: LoopRelevance + ?Sized,
    L::LoopParams: LoopLevels,
    E: EnclosingLoops + ?Sized,
    N: NodeNames + ?Sized,
    Z: SizeStage + ?Sized,
{
    let mut ref_coord = input_ref_coord.clone();
    if prop.scale_down == crate::schedule::ddc::ScaleDown::Yes {
        if let Some(scale) = alloc.scale_down {
            // Compress the coordinates of the scale tensor along the MX dimension.
            let _compressed = scale_down_coord(
                &scale,
                &coord_dim_of(input_ref_coord, scale.dim),
                &mut ref_coord,
            );
        }
    }

    // `refNodeCoordinates.getTensorDims()` OF THE LOCAL COPY, which is what both retries charge.
    let retry_dims: Vec<PrimaryDim> = ref_coord.iter().map(|(dim, _)| dim).collect();
    let Some(row_group) = gated_row_group::<A, D, S, T, N, dsc2::FoldDim>(
        ctx.dsc,
        ctx.core_ds,
        ctx.tree,
        ctx.names,
        tracker,
        prop,
        dims_to_propagate,
        gate,
        &RowCoordinate {
            dims: ref_coord.iter().collect(),
            work_slices: input_ref_slices,
        },
        &retry_dims,
    ) else {
        return false;
    };

    let alloc_chain = ctx.ancestors.enclosing_loop_chain(alloc.node, false);
    if alloc.component == SenComponent::L0 {
        if let Some(up) = &alloc.scale_up {
            if !coord.covers(up.value.dim) {
                // The value tensor's coordinates on L0 are constructed from the coordinates of the
                // corresponding scale tensor allocation on L0_SCALE.
                if !up.allocate.covers(up.value.dim) {
                    // The scale coordinates are not built yet. Retry the propagation later.
                    tracker.retry(ctx.names, *prop, &retry_dims);
                    return false;
                }
                scale_up_coord(
                    &up.value,
                    &coord_dim_of(up.allocate, up.value.dim),
                    coord,
                    alloc.node,
                    alloc.component,
                    ctx.distribution,
                    &alloc_chain,
                    loop_params,
                );
                scale_up_coord(
                    &up.value,
                    &coord_dim_of(up.slice_view, up.value.dim),
                    slice_view,
                    alloc.node,
                    alloc.component,
                    ctx.distribution,
                    &alloc_chain,
                    loop_params,
                );
            }
        }
    }

    // An external allocation with the default work slicing takes the corelet loop one level lower.
    let lower_corelet_loop = ctx.meta.external_nodes.contains(&alloc.node)
        && alloc.corelet_split_dim.is_some()
        && alloc.work_slices.0.is_empty();
    let ref_chain = ctx
        .ancestors
        .enclosing_loop_chain(prop.ref_node, lower_corelet_loop);

    if alloc.padding.stated().next().is_some() {
        coord.set_padding_form(alloc.padding.clone());
    } else {
        let carried = ref_coord.padding_form().clone();
        coord.set_padding_form(carried);
    }
    // `loopDistributionParamInfo[refNode][allocNode]` (`:12621`) is the one `loop_params` the caller
    // hands over for this very pair, so the reference's bare touch is already discharged.

    let mut processed: BTreeSet<PrimaryDim> = BTreeSet::new();
    for &(dim, _) in &alloc.layout.0 {
        if !dims_to_propagate.contains(&dim) || !ref_coord.covers(dim) {
            continue;
        }
        // A dim may appear more than once in the allocation's layout order.
        if !processed.insert(dim) {
            continue;
        }
        let alloc_fold_already_exists = coord.covers(dim);
        let ref_folds = ref_coord.fold_dim(dim);
        let mut levels = ref_folds.map(gather_propagated_levels).unwrap_or_default();
        // ⛔ `foldParams.at(FOLD_POS_CORELET)` and `.at(FOLD_POS_ROWSPLIT)` index the role slots
        // directly (`:12690`, `:12716`), so a fold list too short for them throws in the reference.
        if levels.len() <= FoldPosition::RowSplit.index() {
            panic!(
                "[buildFoldForAllocation] the coordinate of {} carries no role folds for {}.",
                ctx.names.name(prop.ref_node),
                dim.spelling(),
            );
        }
        let count = |folds: u32| usize::try_from(folds).unwrap_or(usize::MAX);
        let num_spatial_ref = count(ref_folds.map_or(0, dsc2::FoldDim::spatial_folds));
        let num_temporal_ref = count(ref_folds.map_or(0, dsc2::FoldDim::temporal_folds))
            + usize::from(lower_corelet_loop);
        let mut orig_elem_arr_ref = count(ref_folds.map_or(0, dsc2::FoldDim::elem_arr_folds));

        // Note: dim is of type PrimaryDimTypes, metaDimKind is missing here.
        let curr_dim = PrimaryDimAndKind {
            dim,
            kind: MetaDimKind::Unpadded,
        };
        let scale_is_non_broadcast = |scale| matches!(scale, Scale::Sized(scale) if scale > 0.0);
        let non_broadcast = alloc
            .dim_scales
            .get(&dim)
            .copied()
            .is_none_or(scale_is_non_broadcast);
        let mut related_for_alloc: Vec<LoopAndDim<'_>> = Vec::new();
        let mut related_for_ref: Vec<LoopAndDim<'_>> = Vec::new();
        if non_broadcast {
            related_for_alloc =
                ctx.distribution
                    .related_loops(curr_dim, &alloc_chain, alloc.padding.padding(dim));
            related_for_ref =
                ctx.distribution
                    .related_loops(curr_dim, &ref_chain, ref_coord.padding(dim));
            if lower_corelet_loop && alloc.corelet_split_dim == Some(dim) {
                let Some(corelet_pos) = related_for_ref
                    .iter()
                    .position(|walked| walked.distribution == LoopDistribution::CoreletSlice)
                else {
                    panic!(
                        "[buildFoldForAllocation] Could not find corelet_slice loop in the \
                         loop-chain of {}.",
                        ctx.names.name(prop.ref_node),
                    );
                };
                // Move the corelet fold to a new temporal position just inside the chunk loops, and
                // leave the corelet slot itself a single no-op trip.
                let corelet = levels[FoldPosition::Corelet.index()].level;
                let at = (num_spatial_ref + corelet_pos).min(levels.len());
                levels.insert(
                    at,
                    PropagatedLevel {
                        level: corelet,
                        label: "corelet_fold".to_owned(),
                    },
                );
                levels[FoldPosition::Corelet.index()] = PropagatedLevel {
                    level: FoldParamInfo {
                        alpha: Alpha(0),
                        beta: Beta(0),
                        cardinality: Cardinality(1),
                        label: None,
                    },
                    label: "corelet_fold".to_owned(),
                };
            }
        }

        // The allocateNode sits in the same loop as the refNode or above it.
        if num_temporal_ref < related_for_alloc.len() {
            panic!(
                "[buildFoldForAllocation] {} has more loops related to {} than {} has temporal \
                 folds.",
                ctx.names.name(alloc.node),
                dim.spelling(),
                ctx.names.name(prop.ref_node),
            );
        }
        let mut num_temporal_alloc = num_temporal_ref;

        let Some(row_fold) =
            compute_params_for_row_split_fold::<A, S>(ctx.core_ds, dim, row_group.group.as_ref())
        else {
            panic!(
                "[buildFoldForAllocation] the PT-row group of {} cannot be folded over {}.",
                ctx.names.name(prop.ref_node),
                dim.spelling(),
            );
        };
        let row_bundling_needed = ctx.core_ds.is_row_split(dim)
            && row_group.group.as_ref().map_or(0, RowGroup::count)
                == usize::try_from(A::PT_ROWS).unwrap_or(usize::MAX);
        if !row_bundling_needed
            || matches!(alloc.component, SenComponent::Ptxrf | SenComponent::Ptarf)
        {
            levels[FoldPosition::RowSplit.index()] = propagated_level(row_fold);
        } else {
            // Bundling coordinates from PT-rows: the spatial row fold becomes the no-op, and a
            // virtual row fold goes just inside the temporal fold of the innermost loop common to
            // the reference node and its row siblings.
            levels[FoldPosition::RowSplit.index()] = propagated_level(default_row_split_fold());
            let ancestor_loop = row_group.ancestor.and_then(|ancestor| {
                ctx.ancestors
                    .enclosing_loops(ancestor)
                    .into_iter()
                    .find(|walked| {
                        ctx.distribution
                            .loop_relevant_for_dim(dim, walked, ref_coord.padding(dim))
                    })
            });
            let Some(ancestor_loop) = ancestor_loop else {
                panic!(
                    "[buildFoldForAllocation] (RefNode={}, workingNode={}Failed to find common \
                     ancestor loop for PT-row bundling.",
                    ctx.names.name(prop.ref_node),
                    ctx.names.name(alloc.node),
                );
            };
            // ⛔ THE REFERENCE COMPARES LOOP POINTERS (`:12746`); [`LoopNode`] carries no identity of
            // its own, so this is value equality, as every other loop lookup in this file is.
            if let Some(found) = related_for_ref
                .iter()
                .position(|walked| walked.loop_node == ancestor_loop)
            {
                // ⚠️ The reference's `begin() + ..` can run past the end, which this clamps.
                let at = (num_spatial_ref + num_temporal_ref)
                    .saturating_sub(found)
                    .min(levels.len());
                levels.insert(at, propagated_level(row_fold));
                orig_elem_arr_ref += 1;
            }
        }

        // If the allocateNode is above the refNode, the temporal folds between them become element
        // arrangement. This assumes stores at the destination are linear, which holds below LX.
        if num_temporal_ref > related_for_alloc.len() {
            num_temporal_alloc = related_for_alloc.len();
            let mut num_elem_arr_alloc =
                orig_elem_arr_ref + num_temporal_ref - num_temporal_alloc;
            if alloc_fold_already_exists {
                // The existing element arrangement cannot change — other nodes' loops may already
                // be tied to specific levels of it — so simulate a distribution over it instead.
                let allocate = coord_dim_of(coord, dim);
                let slice = slice_view.fold_constructed();
                let slice_view: &dsc2::Coordinate = slice_view;
                let propagation = AllocPropagation {
                    alloc: BaseAllocation {
                        component: alloc.component,
                        lds: alloc.lds,
                        coordinates: AllocCoordinates {
                            allocate,
                            slice_view: slice.then(|| coord_dim_of(slice_view, dim)),
                        },
                        dim_scale: alloc.dim_scales.get(&dim).copied(),
                    },
                    ref_node: prop.ref_node,
                    ref_kind: alloc.ref_kind,
                    streams: &prop.ends,
                };
                relate_loops_to_alloc_elem_arr::<D, L, dsc2::FoldDim>(
                    ctx.dsc,
                    ctx.distribution,
                    &propagation,
                    dim,
                    ref_coord.padding(dim),
                    &ref_chain,
                    loop_params,
                );
            } else {
                // First-time construction: relate the loops enclosing the reference node to the
                // element arrangement levels of the allocate node.
                //
                //   Ref Fold  :  0 1 2 3 4 5 6
                //                S S T T T T E
                //   Alloc Fold:  S S T T E E E
                //     For loops related to positions [4-5]
                //       loop_elem_arr_level = foldNumDims - i - 1
                let fold_dim_count = i64::try_from(levels.len()).unwrap_or(i64::MAX);
                let spatial = i64::try_from(num_spatial_ref).unwrap_or(i64::MAX);
                let related_count = i64::try_from(related_for_ref.len()).unwrap_or(i64::MAX);
                let mut curr_elem_arr_level = orig_elem_arr_ref;
                let mut at = fold_dim_count - i64::try_from(orig_elem_arr_ref).unwrap_or(0) - 1;
                let ends = fold_dim_count - i64::try_from(num_elem_arr_alloc).unwrap_or(0);
                while at >= ends {
                    // The related chain runs INNER to OUTER where the fold list runs outer to inner.
                    let related_index = related_count - (at - spatial) - 1;
                    let level = usize::try_from(at)
                        .ok()
                        .and_then(|at| levels.get(at))
                        .map(|level| level.level);
                    let related = usize::try_from(related_index)
                        .ok()
                        .and_then(|at| related_for_ref.get(at));
                    if let (Some(level), Some(related)) = (level, related) {
                        loop_params.record_loop_level(
                            related.loop_node,
                            dim,
                            DistributedLoop {
                                alpha: level.alpha,
                                beta: level.beta,
                            },
                            ElemArrLevel(u32::try_from(curr_elem_arr_level).unwrap_or(u32::MAX)),
                        );
                    }
                    curr_elem_arr_level += 1;
                    at -= 1;
                }
                let mut params: Vec<FoldParamInfo> =
                    levels.iter().map(|level| level.level).collect();
                let mut elem_arr =
                    ElemArrFolds(u32::try_from(num_elem_arr_alloc).unwrap_or(u32::MAX));
                let _split = construct_alloc_elem_arr_layout(
                    alloc.size_stage,
                    alloc.layout,
                    dim,
                    &mut params,
                    &mut elem_arr,
                    loop_params,
                );
                num_elem_arr_alloc = usize::try_from(elem_arr.0).unwrap_or(usize::MAX);
                // ⭐ ENTRY 089 ONLY INSERTS INSIDE THE ELEMENT ARRANGEMENT, so every level outside
                // it keeps both its position and its label; the rest are relabelled `elem_arr_<c>`
                // from the innermost end, which is exactly what the reference does next.
                let outer = params.len().saturating_sub(num_elem_arr_alloc);
                levels = params
                    .iter()
                    .enumerate()
                    .map(|(at, &level)| PropagatedLevel {
                        level,
                        label: if at < outer {
                            levels
                                .get(at)
                                .map_or_else(String::new, |old| old.label.clone())
                        } else {
                            format!("elem_arr_{}", params.len() - 1 - at)
                        },
                    })
                    .collect();
            }
        } else if non_broadcast {
            // Otherwise the two nodes must share their innermost loop relevant to the dim.
            let innermost_ref = ctx
                .ancestors
                .enclosing_loops(prop.ref_node)
                .into_iter()
                .find(|walked| {
                    ctx.distribution
                        .loop_relevant_for_dim(dim, walked, ref_coord.padding(dim))
                })
                .cloned();
            let innermost_alloc = ctx
                .ancestors
                .enclosing_loops(alloc.node)
                .into_iter()
                .find(|walked| {
                    ctx.distribution
                        .loop_relevant_for_dim(dim, walked, coord.padding(dim))
                })
                .cloned();
            // ⛔ THE REFERENCE COMPARES LOOP POINTERS HERE TOO (`:12874`), so two distinct loops that
            // state the same name, tiling and dims abort there and pass here; [`LoopNode`] carries no
            // identity of its own. The two chain prints before it (`:12851-12872`) are debug output.
            if innermost_alloc != innermost_ref {
                panic!(
                    "[buildFoldForAllocation] {} and {} do not share their innermost loop relevant \
                     to {}.",
                    ctx.names.name(alloc.node),
                    ctx.names.name(prop.ref_node),
                    dim.spelling(),
                );
            }
        }

        if alloc_fold_already_exists {
            // Verification: an allocation with custom work slicing must span the same range as the
            // transfer it was propagated from.
            if let Some(check) = &alloc.consistency {
                if !alloc.work_slices.0.is_empty() {
                    let alloc_side = CoordSide {
                        dims: coord.iter().collect(),
                        work_slices: alloc.work_slices,
                        cores: check.alloc_cores.clone(),
                    };
                    if !same_coordinate_range(
                        &alloc_side,
                        &check.transfer,
                        check.sdsc_slices,
                        check.broadcast_dims,
                        DimCoverage::EveryDim,
                    ) {
                        panic!(
                            "Coordinates of transfer {} and allocateNode {} are not consistent.",
                            ctx.names.name(prop.ref_node),
                            ctx.names.name(alloc.node),
                        );
                    }
                }
            }
            continue;
        }

        // Construct the folds for the allocateNode.
        let spatial_ends = i64::try_from(num_spatial_ref).unwrap_or(i64::MAX) - 1;
        let temporal_ends = spatial_ends + i64::try_from(num_temporal_alloc).unwrap_or(i64::MAX);
        add_carried_folds(coord, dim, &levels, spatial_ends, temporal_ends);

        // Where the allocation needs a sliced view, build a separate fold for the sliced
        // arrangement.
        if matches!(alloc.component, SenComponent::L0 | SenComponent::L0Scale) {
            if ctx.core_ds.is_row_split(dim) {
                // Find the element arrangement level a single PT slice chunk ends in.
                let target = num_elements_in_pt_slice::<A>(alloc.stick_dims, dim)
                    .map_or(1, |elems| i64::try_from(elems.0).unwrap_or(i64::MAX));
                let levels_count = i64::try_from(levels.len()).unwrap_or(i64::MAX);
                let ends = levels_count
                    - 1
                    - i64::from(coord.fold_dim(dim).map_or(0, dsc2::FoldDim::elem_arr_folds));
                let mut inner_card: i64 = 1;
                let mut elem_arr_index = levels_count - 1;
                while elem_arr_index > ends {
                    let card = usize::try_from(elem_arr_index)
                        .ok()
                        .and_then(|at| levels.get(at))
                        .map_or(1, |level| {
                            i64::try_from(level.level.cardinality.0).unwrap_or(i64::MAX)
                        });
                    if inner_card.saturating_mul(card) > target {
                        break;
                    }
                    inner_card = inner_card.saturating_mul(card);
                    elem_arr_index -= 1;
                }
                // ⚠️ A list whose element arrangement never fills one slice chunk walks off the
                // FRONT, where the reference reaches `.at(-1)` and throws.
                let chunk = usize::try_from(elem_arr_index).ok();
                let alpha = chunk
                    .and_then(|at| levels.get(at))
                    .map_or(Alpha(0), |level| level.level.alpha);
                let row_split = FoldPosition::RowSplit.index();
                levels[row_split].level.alpha = Alpha(if inner_card == 0 {
                    0
                } else {
                    target.saturating_mul(alpha.0) / inner_card
                });
                levels[row_split].level.beta = Beta(0);
                levels[row_split].level.cardinality = Cardinality(u64::from(A::PT_ROWS));
                if let Some(level) = chunk.and_then(|at| levels.get_mut(at)) {
                    let rows = u64::from(A::PT_ROWS).max(1);
                    level.level.cardinality = Cardinality(level.level.cardinality.0.div_ceil(rows));
                }
            }
            add_carried_folds(slice_view, dim, &levels, spatial_ends, temporal_ends);
            slice_view.set_padding(dim, coord.padding(dim));
        }
    }

    if all_dims_covered(ctx.dsc, coord, alloc.lds) {
        coord.complete_fold_construction();
        if slice_view.iter().next().is_some() {
            slice_view.complete_fold_construction();
        }
    }
    true
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ENTRY 358 — BUILDING A TRANSFER'S FOLD.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE END OF A TRANSFER AS ENTRY 358 READS IT — the operand with the allocation
/// `getAllocation(di, storage, /*allowMissingAlloc=*/true)` finds behind it.
///
/// ⭐ THE LOOKUP IS DROPPED, NOT ITS ANSWER, and the allocation's `padding_` comes WITH it so that an
/// allocation without its padding form is unspellable.
pub struct TransferFoldEnd<'a> {
    /// `src_` / `dstVias_.at(i)` zipped with its `..LdsAndLoopOffsets_` entry.
    pub operand: Operand,
    /// The allocation found behind it and that allocation's `padding_`, absent for `nullptr`.
    pub alloc: Option<(AllocId, &'a PaddingForm)>,
}

/// A TRANSFER'S DESTINATIONS AS ENTRY 358 READS THEM — NON-EMPTY, so that neither of its two
/// `dstVias_.at(0)` reads and neither `dstLdsAndLoopOffsets_.at(0)` read can throw.
pub struct TransferFoldDsts<'a> {
    /// `dstVias_.at(0)` — total.
    pub first: TransferFoldEnd<'a>,
    /// The remaining destinations, in `dstVias_` order.
    pub rest: Vec<TransferFoldEnd<'a>>,
}

impl<'a> TransferFoldDsts<'a> {
    /// Every destination in order.
    pub fn iter(&self) -> impl Iterator<Item = &TransferFoldEnd<'a>> {
        std::iter::once(&self.first).chain(self.rest.iter())
    }
}

/// THE TRANSFER ENTRY 358 BUILDS THE FOLD FOR.
pub struct TransferFold<'a, 'l> {
    /// `transferNode`'s tree identity.
    pub node: NodeId,
    /// `src_` zipped with `srcLdsAndLoopOffsets_`, and its allocation.
    pub src: TransferFoldEnd<'a>,
    /// `dstVias_` zipped with `dstLdsAndLoopOffsets_`, and their allocations.
    pub dsts: TransferFoldDsts<'a>,
    /// `getNonBroadcastLdsDims(srcLdsAndLoopOffsets_.myLdsIdx_)`.
    pub transfer_dims: &'a [PrimaryDim],
    /// `transferSize_`, which the non-allocation propagator may need to override its element
    /// arrangement from.
    pub transfer_size: &'a BTreeMap<PrimaryDim, Cardinality>,
    /// This transfer's own enclosing loops as [`AllocFoldTarget`] takes them — `loopChain` from
    /// `getEnclosingLoopsAndRelatedDims`, INNERMOST FIRST.
    pub alloc_loops: &'a [LoopAndDim<'l>],
    /// The same loops as [`NonAllocTarget`] takes them, with their `loopsBelowChunkBoundary` tags.
    pub non_alloc_loops: &'a [(&'l LoopNode, LoopDistribution)],
    /// `getBlockTransferSizePerDim(*transferNode, transferComp, 0)`, ABSENT WHERE
    /// `isNodeRelatedToComps(transferNode, {PE, SFP}, transferComp)` answered false.
    ///
    /// ⭐ ENTRY 122 FUSED INTO THE WITNESS: its `transferComp` is read only to key this one lookup,
    /// so an absent map IS that `false` and the size needs no second seam.
    pub pe_sfp_transfer_size: Option<&'a BTreeMap<PrimaryDim, Elements>>,
}

/// Replaces: e358_buildFoldForTransfer
///
/// Propagates one coordinate onto a transfer: its destination padding, then the reference's kind
/// picks the propagator, then the PE-SFP split offset lands on the innermost element arrangement.
///
/// ⛔ FIVE OF THE SIX `false`s ARE "NOTHING TO PROPAGATE HERE" and one is the charged retry.
/// ⚠️ AN EMPTY `dataConnect` MATCHES AN EMPTY ONE (`:13854-13856`), so a transfer whose source states
/// no connect, against a propagation that states none either, is read as REFERENCE-IS-SOURCE.
#[must_use]
pub fn build_fold_for_transfer<A, D, S, T, L, E, N, O, F>(
    ctx: &FoldContext<'_, D, S, T, L, E, N>,
    transfer: &TransferFold<'_, '_>,
    prop: &Propagation,
    dims_to_propagate: &[PrimaryDim],
    reference: FoldReference<'_, '_, F>,
    coord: &mut dsc2::Coordinate,
    tracker: &mut CoordPropTracker,
    loop_params: &mut L::LoopParams,
    offsets: &mut O,
) -> bool
where
    A: Arch,
    D: Dsc + Allocations,
    S: PropagatedFoldStage + ?Sized,
    T: ScheduleTree + ?Sized,
    L: LoopRelevance + ?Sized,
    E: EnclosingLoops + ?Sized,
    N: NodeNames + ?Sized,
    O: LoopElemOffsets + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    let FoldReference { gate, kind, .. } = reference;
    // Skipping external transfer.
    if ctx.meta.external_nodes.contains(&transfer.node) {
        return false;
    }
    // Skipping transfer from HBM, and skipping transfer to HBM.
    if transfer.src.operand.storage == SenComponent::Hbm
        || transfer
            .dsts
            .iter()
            .any(|dst| dst.operand.storage == SenComponent::Hbm)
    {
        return false;
    }

    let meta_info = ctx.meta.datatransfers.get(&transfer.node);
    let access = |dim| {
        meta_info.and_then(|info| info.access_pattern_per_dim.get(&dim).copied())
    };

    let mut src_pad = PaddingForm::default();
    if let Some((_, padding)) = transfer.src.alloc {
        src_pad = padding.clone();
    } else if meta_info.is_some() {
        for &dim in transfer.transfer_dims {
            if let Some(pattern) = access(dim) {
                src_pad.set_padding(dim, pattern.from);
            }
        }
    }

    // Determine if reference is source or destination.
    let ref_alloc = match prop.ends.ref_node {
        PropEnd::Allocate { alloc, .. } => Some(alloc),
        PropEnd::Compute { .. } | PropEnd::Other => None,
    };
    let names_ref = |end: &TransferFoldEnd<'_>| {
        (ref_alloc.is_some() && end.alloc.map(|(alloc, _)| alloc) == ref_alloc)
            || end.operand.data.data_connect == prop.ends.data_connect
    };
    let ref_is_src = names_ref(&transfer.src);

    let mut dest_pad = PaddingForm::default();
    let dest_pad_computed = transfer.dsts.iter().any(|dst| match dst.alloc {
        Some((_, padding)) if padding.stated().next().is_some() => {
            dest_pad = padding.clone();
            true
        }
        _ => false,
    });
    if !dest_pad_computed && meta_info.is_some() {
        for &dim in transfer.transfer_dims {
            let pad = access(dim).map_or_else(|| src_pad.padding(dim), |pattern| pattern.to);
            dest_pad.set_padding(dim, pad);
        }
    }
    coord.set_padding_form(dest_pad.clone());

    // Record whether coordinates for the pe-sfp split dimensions are already constructed.
    let existing_pe_sfp: Vec<PrimaryDim> = ctx
        .meta
        .pe_sfp_split_dims
        .iter()
        .copied()
        .filter(|&dim| coord.covers(dim))
        .collect();

    let Some(lds_for_prop) = transfer.dsts.first.operand.data.my_lds_idx.or_else(|| {
        transfer
            .src
            .operand
            .data
            .my_lds_idx
            .filter(|_| meta_info.is_some_and(|info| info.access_pattern_per_dim.is_empty()))
    }) else {
        // Most likely a transfer of constant. No coordinate is needed in that case.
        return false;
    };

    let mut size_ref_comp = if ref_is_src {
        transfer.src.operand.unit
    } else {
        match transfer.dsts.iter().find(|dst| names_ref(dst)) {
            Some(dst) => dst.operand.unit,
            // ⛔ `dstVias_.at(refPosInDest)` with `refPosInDest` still `-1`: no end of the transfer
            // names the reference at all, and the reference indexes the vector with `size_t(-1)`.
            // Reproduced as the abort it is, as `CoordPropTracker::retry`'s threshold is.
            None => panic!(
                "[buildFoldForTransfer] No end of the transfer names the reference {}.",
                ctx.names.name(prop.ref_node),
            ),
        }
    };
    // Override the comp in case transfer is for a single PT row. Example: transfer from LXLU to
    // PTRow4 — the coordinate should be for PT row 4 only.
    let is_row = |comp| comp_row_id(comp).is_some();
    if !is_row(size_ref_comp) {
        if is_row(transfer.src.operand.unit) {
            size_ref_comp = transfer.src.operand.unit;
        } else if is_row(transfer.dsts.first.operand.unit) {
            size_ref_comp = transfer.dsts.first.operand.unit;
        }
    }
    // Override the comp in case transfer involves PE or SFP, to account for PE-SFP splitting. A comp
    // already representing an individual PT row has higher priority than PE or SFP.
    let is_pe_sfp = |comp| matches!(comp, SenComponent::Pe | SenComponent::Sfp);
    if !is_row(size_ref_comp) && !is_pe_sfp(size_ref_comp) {
        if is_pe_sfp(transfer.src.operand.unit) {
            size_ref_comp = transfer.src.operand.unit;
        } else if is_pe_sfp(transfer.dsts.first.operand.unit) {
            size_ref_comp = transfer.dsts.first.operand.unit;
        }
    }

    // ⛔ `effectiveRefCoord` (`:13966-13970`) KEYS OFF `sizeRefComp`, WHICH IS DERIVED ABOVE, so the
    // slice view is selected here and not by the caller.
    let gate_coord = match &gate.ref_slice_view {
        Some(slice_view) if is_row(size_ref_comp) => slice_view,
        _ => &gate.ref_coord,
    };
    let Some(row_group) = gated_row_group::<A, D, S, T, N, F>(
        ctx.dsc,
        ctx.core_ds,
        ctx.tree,
        ctx.names,
        tracker,
        prop,
        dims_to_propagate,
        &gate.scan,
        gate_coord,
        gate.retry_dims,
    ) else {
        return false;
    };

    match kind {
        FoldReferenceKind::Allocate {
            node,
            lds,
            dims,
            work_slices,
            dim_scales,
            corelet_split_dim,
            ..
        } => {
            // TO DO: Try to find a more general checking. Skipping forward coordinate propagation
            // for transfer to lxluScaledValue.
            if transfer.dsts.first.operand.unit == SenComponent::Lxluvalue
                && prop.ref_role == RefRole::Producer
            {
                return false;
            }
            let prop_comp = match prop.ref_role {
                RefRole::Producer => transfer.src.operand.unit,
                RefRole::Consumer => transfer.dsts.first.operand.unit,
            };
            let alloc = AllocationFold {
                lds,
                dims,
                work_slices,
                dim_scales,
                propagated_dims: dims_to_propagate,
                corelet_split_dim,
                components: RefComponents {
                    size: size_ref_comp,
                    prop: prop_comp,
                },
                row_group: row_group.group.as_ref(),
            };
            let target = AllocFoldTarget {
                node: transfer.node,
                loops: transfer.alloc_loops,
            };
            let _ = build_fold_from_allocation::<A, D, S, L, F>(
                ctx.dsc,
                ctx.core_ds,
                ctx.distribution,
                &alloc,
                &target,
                coord,
                loop_params,
            );
            offsets.compute_loop_elem_offsets(transfer.node, coord, node, dims_to_propagate);
        }
        FoldReferenceKind::NonAlloc {
            dims,
            dim_scales,
            loops,
            prop_comp,
            ..
        } => {
            let prop_comp = match prop_comp {
                NonAllocPropComp::TransferEnds { src, .. } => match prop.ref_role {
                    RefRole::Producer => transfer.src.operand.unit,
                    RefRole::Consumer => src,
                },
                NonAllocPropComp::ComputeUnit(unit) => unit,
            };
            let ref_fold = NonAllocReference {
                lds: lds_for_prop,
                dims,
                dim_scales,
                loops,
                components: RefComponents {
                    size: size_ref_comp,
                    prop: prop_comp,
                },
                row_group: row_group.group.as_ref(),
            };
            let bundling = row_group
                .ancestor
                .map(|ancestor| ctx.ancestors.enclosing_loops(ancestor))
                .unwrap_or_default();
            let target = NonAllocTarget {
                node: transfer.node,
                kind: FoldTargetKind::Transfer(transfer.transfer_size),
                loops: transfer.non_alloc_loops,
                row_bundling: &bundling,
            };
            let _ = build_fold_from_non_alloc_ref::<A, D, S, L, F>(
                ctx.dsc,
                ctx.core_ds,
                ctx.distribution,
                &ref_fold,
                &target,
                coord,
                loop_params,
            );
        }
    }

    // Add offset for PE-SFP split to the innermost element-arrangement. The fold parameter for
    // PE-SFP split differs from the rowsplit, which has a dedicated fold level built further down.
    let splitting = meta_info.is_some_and(|info| {
        info.apply_pe_sfp_split_offset_src || !info.apply_pe_sfp_split_offset_dest.is_empty()
    });
    if splitting {
        if let Some(sizes) = transfer.pe_sfp_transfer_size {
            let split_dims: Vec<PrimaryDim> = coord
                .iter()
                .map(|(dim, _)| dim)
                .filter(|dim| {
                    ctx.meta.pe_sfp_split_dims.contains(dim) && !existing_pe_sfp.contains(dim)
                })
                .collect();
            for dim in split_dims {
                // ⛔ `transferSizePerDim.at(dim)` throws for a dim the block size does not name;
                // there is nothing to offset that level by, so the level keeps its own beta.
                if let Some(&size) = sizes.get(&dim) {
                    coord.add_to_innermost_beta(dim, FoldCoeff(size.0.cast_signed()));
                }
            }
        }
    }
    true
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ENTRY 357 — BUILDING A COMPUTE'S FOLD, AND SPREADING IT ACROSS ITS DATASTREAMS.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// `dataStageParam_.at(loop->denId_).ss_.paddingSizes_.at(dim).windowDim_` — which of a loop's dims
/// walks the window of a padded dim, absent where that stage records no padding sizes for the dim.
///
/// ⛔ KEYED ON THE **LOOP'S OWN** DENOMINATOR STAGE (`:13701`), which is why this is neither a
/// [`CoreStage`] question nor the per-dim `window_dim` the hoisting units ask.
pub trait WindowDims {
    /// The window dim a data stage pairs with a padded dim.
    fn window_dim(&self, stage: DatastageId, dim: PrimaryDim) -> Option<PrimaryDim>;
}

/// WHICH SIDE OF A COMPUTE ONE PROPAGATION PASS WALKS — the lambda's `forInput`, which selects the
/// stream list, the coordinate it writes and the constructed-index list in one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamSide {
    /// `inputsLdsAndLoopOffsets_` / `inputCoordinates_` / `constructedInputCoords`.
    Inputs,
    /// `outputsLdsAndLoopOffsets_` / `outputCoordinate_` / `constructedOutputCoords`.
    Outputs,
}

/// A COMPUTE'S COORDINATES, MUTABLY — `inputCoordinates_` and the single `outputCoordinate_`.
///
/// ⛔ ONE OUTPUT COORDINATE HOWEVER MANY OUTPUTS, as [`RelatedComputeCoord`]'s own note says.
pub struct ComputeCoordinates<'a> {
    /// `inputCoordinates_`, which may be SHORTER than an opaque op's attribute list.
    pub inputs: &'a mut [dsc2::Coordinate],
    /// `outputCoordinate_`.
    pub output: &'a mut dsc2::Coordinate,
}

/// `constructedInputCoords` AND `constructedOutputCoords` — the two out-vectors entry 357 clears,
/// pushes into and hands back, and which its caller reads to know what it touched.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstructedCoords {
    /// `constructedInputCoords` — input positions.
    pub inputs: Vec<usize>,
    /// `constructedOutputCoords` — output positions, and every push past the first is a literal `0`.
    pub outputs: Vec<usize>,
}

/// A LABELED DS'S `mxInfo_` AS ENTRY 357 READS IT (`dsc/dscdefn.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxFold {
    /// `mxInfo_.dim`.
    pub dim: PrimaryDim,
    /// `mxInfo_.relatedLdsIdx`, absent for its `-1`.
    pub related_lds: Option<LdsIdx>,
    /// `mxInfo_.blkSize`.
    pub blk_size: ScaleBlock,
}

/// ONE LABELED DS AS THE PER-DIM PROPAGATION READS IT — `labeledDs_.at(ldsIdx)` reduced to the facts
/// the copy / scale-up / scale-down choice turns on.
pub struct LabeledDsFold<'a> {
    /// `scale_` per dim, absent where the layout order does not name the dim — a scale of 1
    /// (`:13317`, `:13333`).
    pub dim_scales: BTreeMap<PrimaryDim, Scale>,
    /// `getCumulativeStickSizes(dsType_)`, which only a `StickDim` scale is charged.
    pub stick_sizes: BTreeMap<PrimaryDim, FoldCardinality>,
    /// `memOrg_.begin()->second.allocateNode_->padding_` — ⛔ AN EMPTY `memOrg_` IS THE DEFAULT FORM
    /// the reference leaves under its bare *"TO DO: Any action needed?"* (`:13296-13300`).
    pub padding: &'a PaddingForm,
    /// `scaledLdsCategory_`.
    pub category: ScaledLds,
    /// `mxInfo_`.
    pub mx: Option<MxFold>,
}

/// THE COMPUTE ENTRY 357 BUILDS THE FOLD FOR.
pub struct ComputeFold<'a, 'l> {
    /// `computeNode`'s tree identity.
    pub node: NodeId,
    /// `exUnit_` — ⛔ A [`SenComponent`] AND NOT [`ComputeStreams`]'s [`DfirUnit`]: it is both
    /// `sizeRefComp` and the PTXRF/PTARF kludge's replacement, and both of those are row-aware.
    pub ex_unit: SenComponent,
    /// `inputsLdsAndLoopOffsets_` zipped with `inputs_`.
    pub inputs: &'a [StoredStream],
    /// `outputsLdsAndLoopOffsets_` zipped with `outputs_`.
    pub outputs: &'a [StoredStream],
    /// `isOpaqueOp_`, with the attribute lists it makes readable.
    pub opaque: Option<OpaqueStreams<'a>>,
    /// `labeledDs_`, for the indices this unit reaches — an absent key is that map's own `.at` throw.
    pub labeled_ds: &'a BTreeMap<LdsIdx, LabeledDsFold<'a>>,
    /// `dataConnects_.at(inputsLdsAndLoopOffsets_.at(constructedInputCoords.at(0)).dataConnect_)
    /// .producers_` (`:13054-13057`) — the ONE connect entry 236 is asked about, resolved off the
    /// index entry 238 selected.
    pub bundling_producers: &'a [ConnectedNode<'a>],
    /// The output padding form the reference hunts for at `:13551-13592`: the first `ALLOCATE`
    /// consumer's `padding_`, else the output transfer's own once its folds are constructed.
    ///
    /// ⛔ [`None`] IS `!outputPaddingAvailable`, the reference's `return true`. Both `DT_CHECK`s and
    /// the `dataConnects_` walk are the caller's, and `DT_CHECK(outputNode)` (`:13596`) is DEAD:
    /// the only path leaving it null returns first.
    pub output_padding: Option<&'a PaddingForm>,
    /// This compute's own enclosing loops as [`AllocFoldTarget`] takes them, INNERMOST FIRST.
    pub alloc_loops: &'a [LoopAndDim<'l>],
    /// The same loops as [`NonAllocTarget`] takes them, with their `loopsBelowChunkBoundary` tags.
    pub non_alloc_loops: &'a [(&'l LoopNode, LoopDistribution)],
}

/// `workingAllocNode`, `workingDc` and `rhsLdsIdx` — the three keys one propagation pass matches a
/// datastream against, any one of which makes it a whole-coordinate copy (`:13248-13250`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkingStream {
    /// `workingAllocNode`, whose null is what makes the allocation test not fire.
    alloc: Option<AllocId>,
    /// `workingDc` — ⚠️ AN EMPTY CONNECT MATCHES AN EMPTY ONE, as entry 358's own note says.
    data_connect: Option<DataConnect>,
    /// `rhsLdsIdx`.
    lds: LdsIdx,
}

/// `inputCoordinates_.at(i)` — ⛔ its throw, which an opaque op whose attribute list outruns its
/// coordinate list reaches.
fn input_at<'c>(coords: &'c mut ComputeCoordinates<'_>, at: usize) -> &'c mut dsc2::Coordinate {
    match coords.inputs.get_mut(at) {
        Some(coord) => coord,
        None => panic!("[buildFoldForCompute] input {at} has no coordinate."),
    }
}

/// The coordinate one side names at a position, mutably.
fn side_coord_mut<'c>(
    coords: &'c mut ComputeCoordinates<'_>,
    side: StreamSide,
    at: usize,
) -> &'c mut dsc2::Coordinate {
    match side {
        StreamSide::Inputs => input_at(coords, at),
        StreamSide::Outputs => &mut *coords.output,
    }
}

/// The coordinate entry 238 selected, mutably.
fn selected_coord_mut<'c>(
    coords: &'c mut ComputeCoordinates<'_>,
    selected: RelatedComputeCoord,
) -> &'c mut dsc2::Coordinate {
    match selected {
        RelatedComputeCoord::Input(at) => side_coord_mut(coords, StreamSide::Inputs, at),
        RelatedComputeCoord::Output(at) => side_coord_mut(coords, StreamSide::Outputs, at),
    }
}

/// `computeCoord` AS THE CATEGORISING SNAPSHOT — ⛔ THE REFERENCE'S IS A LIVE REFERENCE, and by the
/// last propagation pass (`:13773`) the dim loop above has rewritten what it points at, so it is
/// re-read at every call site rather than snapshotted once.
fn categorising_coord(
    coords: &ComputeCoordinates<'_>,
    selected: RelatedComputeCoord,
) -> dsc2::Coordinate {
    match selected {
        RelatedComputeCoord::Input(at) => match coords.inputs.get(at) {
            Some(coord) => coord.clone(),
            None => panic!("[buildFoldForCompute] input {at} has no coordinate."),
        },
        RelatedComputeCoord::Output(_) => coords.output.clone(),
    }
}

/// `labeledDs_.at(ldsIdx)` — ⛔ its throw, for an index the caller resolved no labeled DS for.
fn lds_fold<'a>(compute: &ComputeFold<'a, '_>, lds: LdsIdx) -> &'a LabeledDsFold<'a> {
    match compute.labeled_ds.get(&lds) {
        Some(found) => found,
        None => panic!("[buildFoldForCompute] labeledDs {} is not resolved.", lds.0),
    }
}

/// `scale_.at(dimIdx)` with the reference's `dimIdx < 0 ⇒ 1` (`:13317`, `:13333`, `:13632`, `:13642`).
fn dim_scale_of(lds: &LabeledDsFold<'_>, dim: PrimaryDim) -> Scale {
    lds.dim_scales
        .get(&dim)
        .copied()
        .unwrap_or(Scale::Sized(1.0))
}

/// `scale < 0` WITH THE COST IT CARRIES — ⛔ [`Scale::Sized`] IS THE NON-NEGATIVE ARM BY
/// CONSTRUCTION, so the two sentinels are exactly the reference's broadcast test, and
/// `getCumulativeStickSizes(dsType_).at(dim)` throws for a stick dim those sizes do not name.
fn broadcast_of(lds: &LabeledDsFold<'_>, dim: PrimaryDim, scale: Scale) -> Option<BroadcastScale> {
    match scale {
        Scale::Sized(_) => None,
        Scale::UnitStick => Some(BroadcastScale::Unit),
        Scale::StickDim => match lds.stick_sizes.get(&dim) {
            Some(&card) => Some(BroadcastScale::CumulativeStickSize(card)),
            None => panic!(
                "[buildFoldForCompute] no cumulative stick size for {}.",
                dim.spelling(),
            ),
        },
    }
}

/// `computeCoord.getFoldCategory(dim, pos)` (`dsc/dsc2.h:223-234`) — which band a level sits in, read
/// off the CATEGORISING coordinate while the level itself comes from the reference's.
///
/// ⛔ ONE `panic!` FOR TWO REFERENCE ABORTS: `coordinates_.at(dim)` throws for a dim the categorising
/// coordinate does not cover, and a `pos` past its fold list answers `UNKNOWN_COORD`, which `addFold`
/// then aborts on. ⭐ The element arrangement is the UNBOUNDED TAIL BAND, not a third count.
fn fold_category(
    categories: &dsc2::Coordinate,
    dim: PrimaryDim,
    pos: usize,
    node: &str,
) -> CoordinateCategory {
    let count = |folds: u32| usize::try_from(folds).unwrap_or(usize::MAX);
    let Some(folds) = categories
        .fold_dim(dim)
        .filter(|folds| pos < folds.folds().count())
    else {
        panic!(
            "[buildFoldForCompute] {node} has no fold category for {} at level {pos}.",
            dim.spelling(),
        );
    };
    let spatial = count(folds.spatial_folds());
    if pos < spatial {
        CoordinateCategory::Spatial
    } else if pos < spatial + count(folds.temporal_folds()) {
        CoordinateCategory::Temporal
    } else {
        CoordinateCategory::ElemArr
    }
}

/// `for (i = numDims-1; i >= 0; --i) lhsCoord.addFold(dim, computeCoord.getFoldCategory(dim, i), ..)`
/// (`:13368-13377`, `:13652-13661`) — every level of one dim re-added at the FRONT, outermost last.
fn copy_levels(
    target: &mut dsc2::Coordinate,
    categories: &dsc2::Coordinate,
    source: &dsc2::Coordinate,
    dim: PrimaryDim,
    node: &str,
) {
    let levels = source
        .fold_dim(dim)
        .map(gather_propagated_levels)
        .unwrap_or_default();
    for (at, level) in levels.iter().enumerate().rev() {
        let category = fold_category(categories, dim, at, node);
        add_labelled_fold(target, dim, category, level.label.clone(), level.level);
    }
}

/// `propagateToDataStream(workingAllocNode, workingDc, rhsLdsIdx, rhsCoord, forInput)`
/// (`:13219-13437`) — one coordinate spread over one side of a compute: a whole-coordinate copy for
/// every stream sharing the working allocation, connect or labeled DS, and a per-dim copy, scale-up or
/// scale-down for the rest.
///
/// ⛔ `rhsCoord` AND `computeCoord` ARE TWO COORDINATES, NOT ONE: the levels come off the first and
/// the categories off the second, and by the last pass they hold different folds.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's lambda captures nine of the enclosing scope's bindings"
)]
fn propagate_to_data_stream<D, S, T, L, E, N>(
    ctx: &FoldContext<'_, D, S, T, L, E, N>,
    compute: &ComputeFold<'_, '_>,
    working: WorkingStream,
    rhs: &dsc2::Coordinate,
    categories: &dsc2::Coordinate,
    side: StreamSide,
    coords: &mut ComputeCoordinates<'_>,
    constructed: &mut ConstructedCoords,
    loop_params: &mut L::LoopParams,
) where
    D: Dsc + Allocations,
    S: ?Sized,
    T: ?Sized,
    L: LoopRelevance + ?Sized,
    E: EnclosingLoops + ?Sized,
    N: NodeNames + ?Sized,
{
    let streams = match side {
        StreamSide::Inputs => compute.inputs,
        StreamSide::Outputs => compute.outputs,
    };
    let name = ctx.names.name(compute.node);
    let rhs_lds = lds_fold(compute, working.lds);
    for at in 0..streams.len() {
        let stored = streams[at];
        let first_recorded = match side {
            StreamSide::Inputs => constructed.inputs.first().copied(),
            StreamSide::Outputs => constructed.outputs.first().copied(),
        };
        if first_recorded == Some(at) {
            continue;
        }
        let Some(my_lds) = selected_lds(stored) else {
            continue;
        };
        if my_lds == working.lds
            || (working.alloc.is_some() && ctx.dsc.allocation(stored) == working.alloc)
            || working.data_connect == stored.stream.data_connect
        {
            // Copying the coordinate over: this stream shares the working ldsIdx, allocation or
            // data_connect.
            match side {
                StreamSide::Inputs => {
                    if !input_at(coords, at).fold_constructed() {
                        *input_at(coords, at) = rhs.clone();
                        if all_dims_covered(ctx.dsc, input_at(coords, at), my_lds) {
                            input_at(coords, at).complete_fold_construction();
                        }
                        if !constructed.inputs.contains(&at) {
                            constructed.inputs.push(at);
                        }
                    }
                }
                // ⚠️ NO `foldConstructed` GUARD ON THE OUTPUT (`:13274-13288`), and the push is
                // unconditional where the input's is not.
                StreamSide::Outputs => {
                    *coords.output = rhs.clone();
                    if all_dims_covered(ctx.dsc, coords.output, my_lds) {
                        coords.output.complete_fold_construction();
                    }
                    constructed.outputs.push(0);
                }
            }
            continue;
        }

        // Try to propagate the coordinate of individual dimensions.
        let lhs_lds = lds_fold(compute, my_lds);
        let lhs_dims = ctx.dsc.layout_dims(my_lds).to_vec();
        let mut added_new_dim = false;
        // ⛔ `rowSplitDim` (`:13085-13091`) AND `rowSplitDimPropagated` (`:13302`, `:13337-13339`) ARE
        // DROPPED: the flag is written and never read, so the row-split dim is not a fact this unit
        // needs at all.
        for &lhs_dim in &lhs_dims {
            if side_coord_mut(coords, side, at).covers(lhs_dim) {
                continue;
            }
            if !rhs.covers(lhs_dim) {
                continue;
            }
            if rhs.padding(lhs_dim) != lhs_lds.padding.padding(lhs_dim) {
                continue;
            }
            let lhs_scale = dim_scale_of(lhs_lds, lhs_dim);
            if let Some(broadcast) = broadcast_of(lhs_lds, lhs_dim, lhs_scale) {
                // Constructing the broadcast coordinate for the dim.
                build_fold_for_broadcast_dim(side_coord_mut(coords, side, at), lhs_dim, broadcast);
                continue;
            }
            if lhs_scale != dim_scale_of(rhs_lds, lhs_dim) {
                continue;
            }
            let lhs_mx = lhs_lds.mx.filter(|mx| mx.dim == lhs_dim);
            let scaled = lhs_lds.category != rhs_lds.category;
            if scaled {
                if let Some(mx) = lhs_mx {
                    // The labeledDs must be related in the context of mxScale. Assumption:
                    // verifying one direction of the relation is sufficient.
                    if mx.related_lds != Some(working.lds) {
                        continue;
                    }
                } else if rhs_lds.mx.is_some_and(|mx| mx.dim == lhs_dim) {
                    continue;
                }
            }
            let pad = lhs_lds.padding.padding(lhs_dim);
            side_coord_mut(coords, side, at).set_padding(lhs_dim, pad);
            match lhs_mx.filter(|_| scaled) {
                None => copy_levels(
                    side_coord_mut(coords, side, at),
                    categories,
                    rhs,
                    lhs_dim,
                    name,
                ),
                Some(mx) => match lhs_lds.category {
                    // Scale-tensor coordinate to value-tensor coordinate.
                    ScaledLds::Value => scale_up_coord(
                        &MxValueTensor {
                            lds: my_lds,
                            dim: lhs_dim,
                            blk_size: mx.blk_size,
                        },
                        &coord_dim_of(rhs, lhs_dim),
                        side_coord_mut(coords, side, at),
                        compute.node,
                        compute.ex_unit,
                        ctx.distribution,
                        &ctx.ancestors.enclosing_loop_chain(compute.node, false),
                        loop_params,
                    ),
                    // Value-tensor coordinate to scale-tensor coordinate.
                    ScaledLds::Scale => {
                        let _compressed = scale_down_coord(
                            &MxScaleTensor {
                                dim: lhs_dim,
                                blk_size: mx.blk_size,
                            },
                            &coord_dim_of(rhs, lhs_dim),
                            side_coord_mut(coords, side, at),
                        );
                    }
                    ScaledLds::Regular => panic!(
                        "[buildfoldForCompute] Unhandled ScaledLdsCategory found for labeledDs {}.",
                        my_lds.0,
                    ),
                },
            }
            added_new_dim = true;
        }
        if added_new_dim {
            match side {
                StreamSide::Inputs => {
                    if all_dims_covered(ctx.dsc, input_at(coords, at), my_lds) {
                        input_at(coords, at).complete_fold_construction();
                    }
                    if !constructed.inputs.contains(&at) {
                        constructed.inputs.push(at);
                    }
                }
                StreamSide::Outputs => constructed.outputs.push(0),
            }
        }
    }
}

/// `output_data_connects_.at(at)` / `input_data_connects_.at(at)` — ⛔ their throws.
fn attribute_connect(connects: &[DataConnect], at: usize) -> DataConnect {
    match connects.get(at) {
        Some(&connect) => connect,
        None => panic!("[buildFoldForCompute] no data_connect at attribute position {at}."),
    }
}

/// `for inpIndex : input_data_connects_` — an opaque op's copy onto every non-constructed input
/// naming the working connect (`:13445-13462`, `:13500-13518`, `:13751-13764`).
///
/// ⚠️ THE LAST SITE LATCHES AND PUSHES UNCONDITIONALLY; the other two latch only where the source is
/// itself constructed and push only where the index is absent.
fn copy_to_opaque_inputs(
    opaque: &OpaqueStreams<'_>,
    working_dc: DataConnect,
    source: &dsc2::Coordinate,
    always_latch: bool,
    coords: &mut ComputeCoordinates<'_>,
    constructed: &mut ConstructedCoords,
) {
    for at in 0..opaque.inputs.len() {
        if input_at(coords, at).fold_constructed() {
            continue;
        }
        if attribute_connect(opaque.inputs, at) != working_dc {
            continue;
        }
        *input_at(coords, at) = source.clone();
        if always_latch || source.fold_constructed() {
            input_at(coords, at).complete_fold_construction();
        }
        if always_latch || !constructed.inputs.contains(&at) {
            constructed.inputs.push(at);
        }
    }
}

/// `workingAllocNode`/`workingDc` off one of the compute's own streams (`:13464-13470`,
/// `:13520-13526`, `:13768-13772`) — ⛔ `..LdsAndLoopOffsets_.at(i)`'s throw for a recorded index
/// with no stream behind it.
fn working_stream<D: Allocations + ?Sized>(
    dsc: &D,
    streams: &[StoredStream],
    at: usize,
    lds: LdsIdx,
) -> WorkingStream {
    let Some(&stored) = streams.get(at) else {
        panic!("[buildFoldForCompute] no datastream at position {at}.");
    };
    WorkingStream {
        alloc: dsc.allocation(stored),
        data_connect: stored.stream.data_connect,
        lds,
    }
}

/// Replaces: e357_buildFoldForCompute
///
/// Propagates one coordinate onto a compute: the reference's kind picks the propagator, the result is
/// spread to every input and output sharing its allocation, connect or labeled DS, and the input folds
/// are converted onto the output where the two read the dim's padding differently.
///
/// ⛔ THE `refNode == computeNode` RETURN PRECEDES EVERY CLEAR (`:13017`), so the caller's two
/// out-vectors SURVIVE it and are emptied on all five later `false`s. ⛔ `computeDims` (`:13092`),
/// `loopParamsAfterDistribution` (`:13598` — never read again, and its `operator[]` mints only an
/// empty entry no reader can tell from absent) and the nested *"Unsupported padtype conversion"*
/// `DT_ERROR` (`:13725`, unreachable behind its own two guards) are dead and dropped as such.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "the reference reads twelve `Ddc` members and a port is handed each one"
)]
pub fn build_fold_for_compute<A, D, S, T, L, E, N, O, W, F>(
    ctx: &FoldContext<'_, D, S, T, L, E, N>,
    compute: &ComputeFold<'_, '_>,
    windows: &W,
    prop: &Propagation,
    dims_to_propagate: &[PrimaryDim],
    reference: FoldReference<'_, '_, F>,
    selected: RelatedCoord,
    coords: &mut ComputeCoordinates<'_>,
    constructed: &mut ConstructedCoords,
    tracker: &mut CoordPropTracker,
    loop_params: &mut L::LoopParams,
    offsets: &mut O,
) -> bool
where
    A: Arch,
    D: Dsc + Allocations,
    S: PropagatedFoldStage + ?Sized,
    T: ScheduleTree + ?Sized,
    L: LoopRelevance + ?Sized,
    E: EnclosingLoops + ?Sized,
    N: NodeNames + ?Sized,
    O: LoopElemOffsets + ?Sized,
    W: WindowDims + ?Sized,
    F: AffineFoldDims + ?Sized,
{
    let FoldReference {
        gate,
        kind,
        ref_padding,
    } = reference;
    // A computeNode can be both a consumer and a producer of the same data_connect.
    if prop.ref_node == compute.node {
        return false;
    }
    // `clearWorkDone()`, then the one position entry 238 selected — that unit clears both lists
    // itself before recording it (`:3818-3819`).
    let clear = |constructed: &mut ConstructedCoords| {
        constructed.inputs.clear();
        constructed.outputs.clear();
    };
    clear(constructed);
    match selected.coord {
        RelatedComputeCoord::Input(at) => constructed.inputs.push(at),
        RelatedComputeCoord::Output(at) => constructed.outputs.push(at),
    }

    if selected_coord_mut(coords, selected.coord).fold_constructed() {
        // The fold is already constructed for the coordinate.
        clear(constructed);
        return false;
    }
    if compute.opaque.is_none()
        && !constructed.inputs.is_empty()
        && matches!(kind, FoldReferenceKind::NonAlloc { .. })
        && need_non_row_bundling(ctx.tree, compute.bundling_producers, ConnectEnd::Producers)
    {
        // Non-row bundling is not supported for the compute node.
        clear(constructed);
        return false;
    }
    let Some(compute_lds) = selected.lds else {
        panic!(
            "[buildFoldForCompute] Can not propagate the coordinate to computeNode {} as its ldsIdx \
             is not set.",
            ctx.names.name(compute.node),
        );
    };
    // ⛔ HOISTED WITH ITS ARM'S GATE BUT AHEAD OF IT: the reference asks this at `:13177`, BEFORE the
    // `:13190` gate the port lifts out.
    if matches!(
        kind,
        FoldReferenceKind::NonAlloc {
            prop_comp: NonAllocPropComp::ComputeUnit(_),
            ..
        }
    ) && prop.ends.data_connect.is_none()
    {
        panic!(
            "[buildFoldForCompute] RefNode= {}, computeNode= {}: data_connect is not set.",
            ctx.names.name(prop.ref_node),
            ctx.names.name(compute.node),
        );
    }

    // ⭐ ONE GATE FOR ALL THREE ARMS: they spell it identically, and entry 085 never reads the
    // coordinate it is handed. ⛔ ITS `false` CLEARS, where entry 358's does not.
    let Some(row_group) = gated_row_group::<A, D, S, T, N, F>(
        ctx.dsc,
        ctx.core_ds,
        ctx.tree,
        ctx.names,
        tracker,
        prop,
        dims_to_propagate,
        &gate.scan,
        &gate.ref_coord,
        gate.retry_dims,
    ) else {
        clear(constructed);
        return false;
    };

    let mut can_proceed = true;
    {
        let compute_coord = selected_coord_mut(coords, selected.coord);
        compute_coord.set_padding_form(ref_padding.clone());
        match kind {
            FoldReferenceKind::Allocate {
                node,
                lds,
                component,
                dims,
                work_slices,
                dim_scales,
                corelet_split_dim,
            } => {
                // *** This is a kludge. FIND A GENERAL SOLUTION *** PTXRF and PTARF represent the
                // allocation for all PT rows, while a computeNode on the PT represents the
                // computation on an individual PT row.
                let prop_comp = match component {
                    SenComponent::Ptxrf | SenComponent::Ptarf => compute.ex_unit,
                    other => other,
                };
                let alloc = AllocationFold {
                    lds,
                    dims,
                    work_slices,
                    dim_scales,
                    propagated_dims: dims_to_propagate,
                    corelet_split_dim,
                    components: RefComponents {
                        size: compute.ex_unit,
                        prop: prop_comp,
                    },
                    row_group: row_group.group.as_ref(),
                };
                let target = AllocFoldTarget {
                    node: compute.node,
                    loops: compute.alloc_loops,
                };
                // ⚠️ THE ANSWER IS DISCARDED HERE (`:13128`), so an allocation reference never fails
                // this unit: `canProceed` stays `true` whatever entry 298 reports.
                let _propagated = build_fold_from_allocation::<A, D, S, L, F>(
                    ctx.dsc,
                    ctx.core_ds,
                    ctx.distribution,
                    &alloc,
                    &target,
                    compute_coord,
                    loop_params,
                );
                offsets.compute_loop_elem_offsets(
                    compute.node,
                    compute_coord,
                    node,
                    dims_to_propagate,
                );
            }
            FoldReferenceKind::NonAlloc {
                dims,
                dim_scales,
                loops,
                prop_comp,
                ..
            } => {
                let mut components = RefComponents {
                    size: compute.ex_unit,
                    prop: match prop_comp {
                        NonAllocPropComp::TransferEnds { src, dst } => match prop.ref_role {
                            RefRole::Producer => dst,
                            RefRole::Consumer => src,
                        },
                        // A computeNode corresponding to an individual PT row is expected to carry
                        // that row's unit.
                        NonAllocPropComp::ComputeUnit(unit) => unit,
                    },
                };
                if let NonAllocPropComp::TransferEnds { src, dst } = prop_comp {
                    // Override the components in case one end of the transfer is a single PT row.
                    // ⛔ A DEFAULT `RowGroupInfo` IS `NO_BUNDLING`, so an absent group RUNS this.
                    let row_to_non_row = row_group
                        .group
                        .as_ref()
                        .is_some_and(|group| group.category() == RowGroupCategory::RowToNonRow);
                    if !row_to_non_row {
                        if comp_row_id(src).is_some() {
                            components = RefComponents { size: src, prop: src };
                        } else if comp_row_id(dst).is_some() {
                            components = RefComponents { size: dst, prop: dst };
                        }
                    }
                }
                let ref_fold = NonAllocReference {
                    lds: compute_lds,
                    dims,
                    dim_scales,
                    loops,
                    components,
                    row_group: row_group.group.as_ref(),
                };
                let bundling = row_group
                    .ancestor
                    .map(|ancestor| ctx.ancestors.enclosing_loops(ancestor))
                    .unwrap_or_default();
                let target = NonAllocTarget {
                    node: compute.node,
                    kind: FoldTargetKind::Compute,
                    loops: compute.non_alloc_loops,
                    row_bundling: &bundling,
                };
                // ⛔ ENTRY 299 ANSWERS `false` FOR EXACTLY ONE STATE (`:8653`), so this is the whole
                // of `canProceed`.
                can_proceed = build_fold_from_non_alloc_ref::<A, D, S, L, F>(
                    ctx.dsc,
                    ctx.core_ds,
                    ctx.distribution,
                    &ref_fold,
                    &target,
                    compute_coord,
                    loop_params,
                ) != FoldPropagation::AlreadyConstructed;
            }
        }
    }
    if !can_proceed {
        clear(constructed);
        return false;
    }
    if all_dims_covered(
        ctx.dsc,
        selected_coord_mut(coords, selected.coord),
        compute_lds,
    ) {
        selected_coord_mut(coords, selected.coord).complete_fold_construction();
    }

    if let Some(&first_input) = constructed.inputs.first() {
        match &compute.opaque {
            Some(opaque) => {
                let working_dc = attribute_connect(opaque.inputs, first_input);
                let source = categorising_coord(coords, selected.coord);
                copy_to_opaque_inputs(opaque, working_dc, &source, false, coords, constructed);
            }
            None => {
                let working = working_stream(ctx.dsc, compute.inputs, first_input, compute_lds);
                // ⭐ ONE SNAPSHOT SERVES AS BOTH LEVELS AND CATEGORIES ACROSS BOTH CALLS: the
                // selected input is protected from the first by its own recorded position, and
                // `constructedOutputCoords` is provably empty through it because only the OUTPUT arms
                // push there.
                let source = categorising_coord(coords, selected.coord);
                propagate_to_data_stream(
                    ctx,
                    compute,
                    working,
                    &source,
                    &source,
                    StreamSide::Inputs,
                    coords,
                    constructed,
                    loop_params,
                );
                let source = categorising_coord(coords, selected.coord);
                propagate_to_data_stream(
                    ctx,
                    compute,
                    working,
                    &source,
                    &source,
                    StreamSide::Outputs,
                    coords,
                    constructed,
                    loop_params,
                );
            }
        }
    }

    // Propagate the coordinate to the output(s) of the compute node. The output may not carry all the
    // dimensions of the inputs; pooling is the example.
    if let Some(&first_output) = constructed.outputs.first() {
        match &compute.opaque {
            Some(opaque) => {
                let working_dc = attribute_connect(opaque.outputs, first_output);
                let source = coords.output.clone();
                copy_to_opaque_inputs(opaque, working_dc, &source, false, coords, constructed);
            }
            None => {
                let working = working_stream(ctx.dsc, compute.outputs, first_output, compute_lds);
                let source = coords.output.clone();
                let categories = categorising_coord(coords, selected.coord);
                propagate_to_data_stream(
                    ctx,
                    compute,
                    working,
                    &source,
                    &categories,
                    StreamSide::Inputs,
                    coords,
                    constructed,
                    loop_params,
                );
            }
        }
    }

    if coords.output.fold_constructed() {
        return true;
    }

    // Find the output ldsIdx.
    let output_lds_idx = compute.outputs.iter().copied().find_map(selected_lds);
    let Some(output_padding) = compute.output_padding else {
        return true;
    };
    coords.output.set_padding_form(output_padding.clone());

    // ⭐ THE OPAQUE `outputDims` LDS *IS* `computeLdsIdx`: entry 238's opaque arm answers
    // `opaqueOps_.at(node).ldsIdx_` itself, and an absent one has already panicked above.
    let output_dims: Vec<PrimaryDim> = if compute.opaque.is_some() {
        ctx.dsc.layout_dims(compute_lds).to_vec()
    } else {
        output_lds_idx.map_or_else(Vec::new, |lds| ctx.dsc.layout_dims(lds).to_vec())
    };
    let output_lds = output_lds_idx.map(|lds| lds_fold(compute, lds));
    let compute_lds_fold = lds_fold(compute, compute_lds);
    let name = ctx.names.name(compute.node);
    let categories = categorising_coord(coords, selected.coord);

    // Propagate the newly constructed input folds to the output.
    for &dim in &output_dims {
        if coords.output.covers(dim) {
            continue;
        }
        // The coordinate of all the dimensions may not be available in the base coordinate.
        if !categories.covers(dim) {
            continue;
        }
        // A negative scale indicates the broadcast property of the dimension.
        let dim_scale = output_lds.map_or(Scale::Sized(1.0), |lds| dim_scale_of(lds, dim));
        if let Some(broadcast) = output_lds.and_then(|lds| broadcast_of(lds, dim, dim_scale)) {
            build_fold_for_broadcast_dim(&mut *coords.output, dim, broadcast);
            continue;
        }
        // The reference dim is a broadcast dim while the current dim is not.
        let ref_scale = dim_scale_of(compute_lds_fold, dim);
        let non_broadcast = matches!(dim_scale, Scale::Sized(scale) if scale > 0.0);
        if non_broadcast && matches!(ref_scale, Scale::UnitStick | Scale::StickDim) {
            continue;
        }

        let input_padding = categories.padding(dim);
        if input_padding == coords.output.padding(dim) {
            copy_levels(&mut *coords.output, &categories, &categories, dim, name);
            continue;
        }
        // ⛔ THE `else` UNDER THIS IS THE UNREACHABLE `DT_ERROR`: an output padding of `NOPAD` that
        // differs from the input's leaves the input NOT `NOPAD` by construction.
        if coords.output.padding(dim) != PadType::NoPad {
            continue;
        }
        // Unpadded access to a padded arrangement: the loops relevant to the dim decide which
        // temporal levels survive, and the core datastage's stride divides what is left.
        let curr_loop_chain: Vec<&LoopNode> = ctx
            .ancestors
            .enclosing_loops(compute.node)
            .into_iter()
            .filter(|walked| {
                ctx.distribution
                    .loop_relevant_for_dim(dim, walked, input_padding)
            })
            .collect();
        let levels = categories
            .fold_dim(dim)
            .map(gather_propagated_levels)
            .unwrap_or_default();
        let num_elem_arr = usize::try_from(
            categories
                .fold_dim(dim)
                .map_or(0, dsc2::FoldDim::elem_arr_folds),
        )
        .unwrap_or(usize::MAX);
        for (at, level) in levels.iter().enumerate().rev() {
            let category = fold_category(&categories, dim, at, name);
            // Do we need to get the stride from the loop denominator? For now, using the stride from
            // the core datastage.
            let mut factor = 1;
            if matches!(
                category,
                CoordinateCategory::Temporal | CoordinateCategory::Spatial
            ) {
                let Some(stride) = ctx.core_ds.pad_stride(dim) else {
                    panic!(
                        "[buildFoldForCompute] {name} states no padding stride for {}.",
                        dim.spelling(),
                    );
                };
                factor = if stride.get() == -1 { 1 } else { stride.get() };
            }
            // TO DO: We need to reduce the cardinality of the window loop to 1.
            if category == CoordinateCategory::Temporal {
                // No fold is needed for a window loop while accessing an unpadded data structure.
                // ⛔ THE INDEX GOES NEGATIVE FOR A CHAIN SHORTER THAN THE FOLD LIST, where the
                // reference's `.at(size_t(-1))` throws.
                let Some(loop_node) = (levels.len().checked_sub(1))
                    .and_then(|last| last.checked_sub(at))
                    .and_then(|rest| rest.checked_sub(num_elem_arr))
                    .and_then(|position| curr_loop_chain.get(position))
                else {
                    panic!(
                        "[buildFoldForCompute] {name} has fewer loops related to {} than temporal \
                         folds.",
                        dim.spelling(),
                    );
                };
                let window = windows.window_dim(loop_node.den, dim);
                let is_window_loop = loop_node
                    .dims
                    .iter()
                    .any(|walked| walked.kind == MetaDimKind::WindowDim && window == Some(walked.dim));
                if is_window_loop {
                    continue;
                }
            }
            // Should we scale beta for the temporal folds as well?
            let beta = if category == CoordinateCategory::Spatial {
                Beta(level.level.beta.0 / factor)
            } else {
                level.level.beta
            };
            add_labelled_fold(
                &mut *coords.output,
                dim,
                category,
                level.label.clone(),
                FoldParamInfo {
                    alpha: Alpha(level.level.alpha.0 / factor),
                    beta,
                    ..level.level
                },
            );
        }
    }

    // ⚠️ THIS COVERAGE TEST IS TAKEN BEFORE the copies below, which the reference's own ordering
    // (`:13736-13745` then `:13748`) puts first.
    let all_covered = output_dims.iter().all(|&dim| coords.output.covers(dim));
    // Propagate/copy to the inputs associated with the same allocation or data_connect as any of the
    // outputs.
    match &compute.opaque {
        Some(opaque) => {
            let working_dc = attribute_connect(opaque.outputs, 0);
            let source = coords.output.clone();
            copy_to_opaque_inputs(opaque, working_dc, &source, true, coords, constructed);
        }
        None => {
            if let Some(output_lds_idx) = output_lds_idx {
                for at in 0..compute.outputs.len() {
                    let working = working_stream(ctx.dsc, compute.outputs, at, output_lds_idx);
                    let source = coords.output.clone();
                    let categories = categorising_coord(coords, selected.coord);
                    propagate_to_data_stream(
                        ctx,
                        compute,
                        working,
                        &source,
                        &categories,
                        StreamSide::Inputs,
                        coords,
                        constructed,
                        loop_params,
                    );
                }
            }
        }
    }

    constructed.outputs.push(0);
    if all_covered {
        coords.output.complete_fold_construction();
    }
    true
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ ENTRY 370 — THE DRIVER THAT BUILDS EVERY FOLD AND PROPAGATES IT.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// `coordPropReportLevel_` AS ITS TWO THRESHOLDS — `> 0` and `> 2` is the whole of what the level
/// means to entry 370.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CoordPropReport {
    /// `0` — no report.
    #[default]
    Off,
    /// `1` and `2` — every propagation step's header. ⭐ LEVEL 2 ADDS NOTHING AT THIS ENTRY.
    Steps,
    /// `3` and above — the same, plus every labeled DS and every unsupported node kind.
    Labels,
}

impl CoordPropReport {
    /// `coordPropReportLevel_ > 0`.
    const fn reports(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// `coordPropReportLevel_ > 2`.
    const fn labels(self) -> bool {
        matches!(self, Self::Labels)
    }
}

/// ONE LABELED DATA STRUCTURE AS THE EXTERNAL SCAN READS IT — a `currDsc->labeledDs_` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropLds {
    /// Which entry this is.
    pub idx: LdsIdx,
    /// `dsType_`.
    pub ds_type: DsType,
    /// `memOrg_.at(SenComponents::LX).allocateNode_`, absent where `memOrg_.count(LX)` is zero —
    /// which is the reference's `extAllocNode == nullptr`.
    pub external: Option<NodeId>,
}

/// AN ALLOCATE NODE'S COORDINATE AS THE PROPAGATION READS IT — `allocateCoordinates_`'s tensor dims
/// with the work slices that filter them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocateCoord {
    /// `getTensorDims()`.
    pub dims: Vec<PrimaryDim>,
    /// `coreIdToWkSlice_` — empty is a state and not an absence, as [`WorkSlices`] says.
    pub work_slices: WorkSlices,
}

/// AN ALLOCATE NODE AS THE PROPAGATION KEYS ON ONE — its two identities plus the labeled-DS facts
/// the MX value-to-scale redirect reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropAlloc {
    /// The node — `refsAdded_`'s key and `dimsToPropagate`'s owner.
    pub node: NodeId,
    /// The allocation the node holds, which is what entry 086 compares.
    pub alloc: AllocId,
    /// `component_`.
    pub component: SenComponent,
    /// `ldsIdx_`, absent for its `-1`.
    pub lds: Option<LdsIdx>,
    /// `labeledDs_.at(ldsIdx_).scaledLdsCategory_`, absent for that same `-1`.
    pub scaled: Option<ScaledLds>,
}

/// ONE DATASTREAM END AS THE PROPAGATION READS IT — the reference's `(dataInfo, comp)` argument pair.
///
/// ⚠️ THE STORAGE IS HERE TWICE BECAUSE THE CRATE SPELLS IT TWICE: `getAllocation` is keyed on
/// [`StoredStream`]'s [`DfirUnit`], while the PT_NORTH/PT_SOUTH test compares the reference's own
/// `SenComponents`. Converging the two spellings is the note on [`dsc2::DataInfo`]'s, not this
/// entry's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropStream {
    /// `dataInfo` with the memory it is looked up in.
    pub stored: StoredStream,
    /// `comp` — the same storage as a `SenComponents`.
    pub component: SenComponent,
}

/// AN OPAQUE OP'S CONNECT LISTS — `instrAttribute_`'s two `data_connects_` vectors, which are the
/// only handle on an opaque op's operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueConnects {
    /// `input_data_connects_`.
    pub inputs: Vec<DataConnect>,
    /// `output_data_connects_`.
    pub outputs: Vec<DataConnect>,
}

/// A COMPUTE NODE AS ENTRY 370 READS IT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropCompute {
    /// `exUnit_` in the fold vocabulary, for [`PropEnd::Compute`].
    pub ex_unit: DfirUnit,
    /// `senCompToRowId.count(exUnit_) ? .at(exUnit_) : -1` — the INLINE table read, which is why it
    /// is not [`comp_row_id`]'s call.
    pub row: Option<PtRowId>,
    /// `isOpaqueOp_`, with the connect lists it makes readable.
    pub opaque: Option<OpaqueConnects>,
    /// `inputsLdsAndLoopOffsets_` paired with `inputs_`.
    pub inputs: Vec<PropStream>,
    /// `outputsLdsAndLoopOffsets_` paired with `outputs_`.
    pub outputs: Vec<PropStream>,
}

/// ONE END OF A TRANSFER — a `..LdsAndLoopOffsets_`/`DataLocation` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropTransferEnd {
    /// The stream, in the storage it is looked up in.
    pub stream: PropStream,
    /// `loc_.unit_` — what [`comp_row_id`] is asked about.
    pub unit: SenComponent,
}

/// A TRANSFER NODE AS ENTRY 370 READS IT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropTransfer {
    /// `srcLdsAndLoopOffsets_` with `src_`.
    pub src: PropTransferEnd,
    /// `dstLdsAndLoopOffsets_` with `dstVias_`, in destination order.
    pub dsts: Vec<PropTransferEnd>,
}

/// ONE USER'S INCOMING DATASTREAMS, OWNED — what entry 086's borrowed [`IncomingStreams`] reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserStreams {
    /// `TRANSFER` — `srcLdsAndLoopOffsets_` in `src_.storage_`.
    Transfer(StoredStream),
    /// `COMPUTE` — every input.
    Compute(Vec<StoredStream>),
}

impl UserStreams {
    /// The same streams as entry 086 reads them.
    #[must_use]
    pub fn borrow(&self) -> IncomingStreams<'_> {
        match self {
            Self::Transfer(src) => IncomingStreams::Transfer(*src),
            Self::Compute(inputs) => IncomingStreams::Compute(inputs),
        }
    }
}

/// WHICH COORDINATE THE ALLOCATE ARM PROPAGATES FROM — `refCoordinate`, NAMED and not lent because
/// both of the calls it feeds mutate the node it lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropRefCoord {
    /// `static_cast<TransferNode *>(refNode)->transferCoordinates_`.
    Transfer(NodeId),
    /// `getRelatedComputeCoord(refCompute, reverseCoordPropInfo, ..)`'s pick on that compute.
    Compute(NodeId, RelatedCoord),
}

/// `SenComponents::L0_SCALE` / `PTXRF` — the memory a value tensor's scale tensor is allocated in,
/// and [`None`] for the reference's `NO_COMPONENT` `DT_ERROR` (`ddc/ddc_fold.cpp:1737-1744`).
const fn scale_memory(component: SenComponent) -> Option<SenComponent> {
    match component {
        SenComponent::L0 => Some(SenComponent::L0Scale),
        SenComponent::Ptxrf => Some(SenComponent::Ptxrf),
        _ => None,
    }
}

/// WHAT ENTRY 370 READS AND MUTATES, AND THE SEAM IT REACHES THE FOUR FOLD BUILDERS THROUGH.
///
/// ⛔⛔ THE SEAM, AND NOT A STAND-IN: the five builder methods below MUST call the landed
/// [`build_fold_for_external_allocation`], [`build_fold_for_compute`], [`build_fold_for_transfer`],
/// [`build_fold_for_allocation`] and [`get_related_compute_coord`]. Their argument lists take eight
/// to twelve witnesses each — `FoldContext`, `ComputeFold`, `FoldReference`, `RowBundlingScan`,
/// `WorkSlices` and the rest — and only the whole `Ddc` can assemble those, which is the mechanism
/// for reaching operands and the one part of this entry that is not stated here. A stand-in fold
/// leaves every coordinate below LX unbuilt and this entry has nothing left to propagate.
/// ⛔ THE COORDINATES STAY IN THE DSC. `refCoordinate` is named by [`PropRefCoord`] rather than lent
/// because entry 356 and `computeLoopElemOffsetsFromCoordinates` both write through it.
pub trait FoldDriver: ScheduleTree + Allocations {
    /// `scheduleTree_.traverseTreeDFSMutable(nullptr, {COMPUTE})`.
    fn computes(&self) -> Vec<NodeId>;

    /// `computeNode->inputCoordinates_.resize(slots)`.
    fn resize_input_coordinates(&mut self, compute: NodeId, slots: usize);

    /// `currDsc->labeledDs_`, in order.
    fn labeled_ds(&self) -> Vec<PropLds>;

    /// `currDsc->printLabeledDs(std::cout, lds, "  ")`.
    fn labeled_ds_text(&self, lds: LdsIdx) -> String;

    /// `isExternalNode(node)`.
    fn is_external_node(&self, node: NodeId) -> bool;

    /// `allocNode->allocateCoordinates_`.
    fn allocate_coord(&self, alloc: NodeId) -> AllocateCoord;

    /// `sdsc_->coreIdToWkSlice_`.
    fn sdsc_work_slices(&self) -> WorkSlices;

    /// `allocNode->allocUsers_`'s keys, in the map's order. ⛔ THE REF COUNT BESIDE EACH IS UNREAD.
    fn allocation_users(&self, alloc: NodeId) -> Vec<NodeId>;

    /// The incoming datastreams entry 086 reads off one user.
    fn incoming_streams(&self, node: NodeId) -> Option<UserStreams>;

    /// An allocate node's identities and labeled-DS facts, absent for a node that is not one.
    fn prop_allocation(&self, node: NodeId) -> Option<PropAlloc>;

    /// The node an allocation is held by — what `getMutableAllocation`'s answer is keyed back to.
    fn allocation_node(&self, alloc: AllocId) -> Option<NodeId>;

    /// `labeledDs_.at(labeledDs_.at(lds).mxInfo_.relatedLdsIdx).memOrg_.at(memory).allocateNode_`,
    /// and [`None`] where either of those two `.at()` throws.
    fn related_scale_allocation(&self, lds: LdsIdx, memory: SenComponent) -> Option<NodeId>;

    /// A compute node as this entry reads it, absent for a node that is not one.
    fn prop_compute(&self, compute: NodeId) -> Option<PropCompute>;

    /// A transfer node as this entry reads it, absent for a node that is not one.
    fn prop_transfer(&self, transfer: NodeId) -> Option<PropTransfer>;

    /// `inputCoordinates_.at(i).getTensorDims()` / `outputCoordinate_.getTensorDims()`. ⛔ AN OUTPUT
    /// POSITION NAMES THE SINGLE `outputCoordinate_` whatever its index, as [`RelatedComputeCoord`]
    /// says.
    fn compute_coord_dims(&self, compute: NodeId, at: OperandPos) -> Vec<PrimaryDim>;

    /// `transferCoordinates_.getTensorDims()`, and [`None`] where `coordinates_` is EMPTY — which is
    /// the emptiness the TRANSFER arm tests before it walks the ends.
    fn transfer_coord_dims(&self, transfer: NodeId) -> Option<Vec<PrimaryDim>>;

    /// The node the census indexed at that position — `metadata.dataConnects_` holds its ends
    /// positionally, and the propagation queue keys on the tree.
    fn node_at(&self, index: NodeIndex) -> Option<NodeId>;

    /// The node itself, for entry 233's debug line.
    fn schedule_node(&self, node: NodeId) -> Option<Node<'_>>;

    /// `buildFoldForExternalAllocation(extAllocNode)` — entry 240 with its witnesses assembled.
    fn build_fold_for_external_allocation(&mut self, alloc: NodeId);

    /// `buildFoldForCompute(computeNode, coordPropInfo, constructedInputCoords,
    /// constructedOutputCoords)` — entry 357, whose two out-vectors are the answer.
    fn build_fold_for_compute(
        &mut self,
        compute: NodeId,
        item: &QueuedProp,
        tracker: &mut CoordPropTracker,
    ) -> ConstructedCoords;

    /// `buildFoldForTransfer(transferNode, coordPropInfo)` — entry 358.
    fn build_fold_for_transfer(
        &mut self,
        transfer: NodeId,
        item: &QueuedProp,
        tracker: &mut CoordPropTracker,
    ) -> bool;

    /// `buildFoldForAllocation(coordPropInfo, refCoordinate, allocNode)` — entry 356.
    fn build_fold_for_allocation(
        &mut self,
        alloc: NodeId,
        item: &QueuedProp,
        reference: PropRefCoord,
        tracker: &mut CoordPropTracker,
    ) -> bool;

    /// `getRelatedComputeCoord(refCompute, reverseCoordPropInfo, inputCoords, outputCoords, ldsIdx)`
    /// — entry 238 over the compute's own streams, which only the DSC can lend.
    fn related_compute_coord(
        &self,
        compute: NodeId,
        reference: &ComputeCoordRef,
    ) -> Option<RelatedCoord>;

    /// `dsc2::computeLoopElemOffsetsFromCoordinates(currDsc, refNode, refCoordinate, allocNode,
    /// loopDistributionParamInfo.at(refNode).at(allocNode), dimsToPropagate, coordPropReportLevel_)`
    /// — the [`LoopElemOffsets`] capability, reached over the coordinate this entry named.
    fn loop_elem_offsets_from_coordinates(
        &mut self,
        reference: PropRefCoord,
        alloc: NodeId,
        dims: &[PrimaryDim],
        report: CoordPropReport,
    );
}

/// `coordPropInfo.refNode` / `.nodeToFold` as [`PropEnd`] spells one — the ALLOCATE and COMPUTE arms
/// carry what entry 091's match reads and every other kind is [`PropEnd::Other`].
fn prop_end<S: FoldDriver + ?Sized>(dsc: &S, node: NodeId) -> PropEnd {
    match dsc.kind(node) {
        NodeKind::Allocate => match dsc.prop_allocation(node) {
            Some(alloc) => PropEnd::Allocate {
                alloc: alloc.alloc,
                scaled: alloc.scaled,
            },
            None => PropEnd::Other,
        },
        NodeKind::Compute => match dsc.prop_compute(node) {
            Some(compute) => PropEnd::Compute {
                ex_unit: compute.ex_unit,
            },
            None => PropEnd::Other,
        },
        _ => PropEnd::Other,
    }
}

/// `for (auto &[userNode, refCount] : allocNode->allocUsers_)` — the block the external scan and the
/// ALLOCATE arm run WORD FOR WORD (`ddc/ddc_fold.cpp:1690-1710`, `:1901-1921`): every COMPUTE or
/// TRANSFER user that is not an opaque compute is queued against this allocation.
fn queue_allocation_users<S: FoldDriver + ?Sized>(
    dsc: &S,
    tracker: &mut CoordPropTracker,
    alloc: NodeId,
    dims: &[PrimaryDim],
) {
    let Some(owner) = dsc.prop_allocation(alloc) else {
        return;
    };
    let users = dsc.allocation_users(alloc);
    for &user in &users {
        let end = match dsc.kind(user) {
            NodeKind::Compute => {
                let Some(body) = dsc.prop_compute(user) else {
                    continue;
                };
                // Opaque operations do not maintain dataInfos. Currently, it is not possible to
                // determine which input or output of an opaque op corresponds to an allocateNode
                // unless the associated data_connect is also specified.
                if body.opaque.is_some() {
                    continue;
                }
                PropEnd::Compute {
                    ex_unit: body.ex_unit,
                }
            }
            NodeKind::Transfer => PropEnd::Other,
            _ => continue,
        };
        // ⭐ NEITHER OF ENTRY 086'S REFUSALS IS REACHABLE HERE: the list IS `allocUsers_`, and the
        // kind is already one of the two that carry incoming datastreams.
        let streams = dsc.incoming_streams(user);
        let incoming = Incoming::of(&users, user, streams.as_ref().map(UserStreams::borrow));
        let ref_role = match incoming {
            Some(user) if is_allocate_incoming(dsc, owner.alloc, &user) => RefRole::Producer,
            _ => RefRole::Consumer,
        };
        tracker.add_prop_info(
            Propagation {
                ends: CoordPropInfo {
                    data_connect: None,
                    ref_node: PropEnd::Allocate {
                        alloc: owner.alloc,
                        scaled: owner.scaled,
                    },
                    node_to_fold: end,
                },
                ref_node: alloc,
                node_to_fold: user,
                ref_role,
                scale_down: super::ScaleDown::No,
            },
            dims,
        );
    }
}

/// `collectAndProcessDatastreamNodes` (`ddc/ddc_fold.cpp:1714-1830`) — entry 370's own lambda: one
/// datastream end either reaches an allocation, which is queued (with its MX scale tensor behind it),
/// or it is a FIFO, whose peers on that `data_connect=` are queued after the PT-row filter.
fn collect_and_process_datastream_nodes<S: FoldDriver + ?Sized>(
    dsc: &S,
    metadata: &Metadata,
    tracker: &mut CoordPropTracker,
    curr: NodeId,
    ref_for_curr: NodeId,
    stream: PropStream,
    ref_dims: &[PrimaryDim],
    row: Option<PtRowId>,
    side: StreamSide,
) {
    let curr_end = prop_end(dsc, curr);
    let ref_role = match side {
        StreamSide::Inputs => RefRole::Consumer,
        StreamSide::Outputs => RefRole::Producer,
    };
    if let Some(alloc) = dsc.allocation(stream.stored) {
        let Some(owner) = dsc
            .allocation_node(alloc)
            .and_then(|node| dsc.prop_allocation(node))
        else {
            return;
        };
        if owner.node == ref_for_curr {
            return;
        }
        let mut prop = Propagation {
            ends: CoordPropInfo {
                data_connect: None,
                ref_node: curr_end,
                node_to_fold: PropEnd::Allocate {
                    alloc: owner.alloc,
                    scaled: owner.scaled,
                },
            },
            ref_node: curr,
            node_to_fold: owner.node,
            ref_role,
            scale_down: super::ScaleDown::No,
        };
        tracker.add_prop_info(prop, ref_dims);
        // In case a computeNode consumes from or produces to a value tensor allocation in MX-scale
        // mode, include the corresponding scale tensor allocation in the propagation queue.
        if let PropEnd::Compute { .. } = curr_end
            && let Some(lds) = owner.lds
            && owner.scaled == Some(ScaledLds::Value)
        {
            let memory = scale_memory(owner.component).unwrap_or_else(|| {
                panic!(
                    "Unsupported component {} for value tensor allocation.",
                    owner.component.spelling()
                )
            });
            let Some(scale) = dsc
                .related_scale_allocation(lds, memory)
                .and_then(|node| dsc.prop_allocation(node))
            else {
                return;
            };
            prop.ends.node_to_fold = PropEnd::Allocate {
                alloc: scale.alloc,
                scaled: scale.scaled,
            };
            prop.node_to_fold = scale.node;
            prop.scale_down = super::ScaleDown::Yes;
            tracker.add_prop_info(prop, ref_dims);
        }
        return;
    }
    // The datastream is a FIFO.
    if is_memory(stream.component) || stream.component == SenComponent::NoComponent {
        return;
    }
    let Some(dc) = stream.stored.stream.data_connect else {
        return;
    };
    let Some(ends) = metadata.data_connects.get(&dc) else {
        return;
    };
    let peers = match side {
        StreamSide::Inputs => ends.producers(),
        StreamSide::Outputs => ends.consumers(),
    };
    for &index in peers {
        let Some(peer) = dsc.node_at(index) else {
            continue;
        };
        let peer_end = match dsc.kind(peer) {
            NodeKind::Compute => {
                let Some(body) = dsc.prop_compute(peer) else {
                    continue;
                };
                if let Some(row) = row
                    && !row_matches_compute(row, body.row, curr_end, stream.component)
                {
                    continue;
                }
                PropEnd::Compute {
                    ex_unit: body.ex_unit,
                }
            }
            NodeKind::Transfer => {
                let Some(body) = dsc.prop_transfer(peer) else {
                    continue;
                };
                if let Some(row) = row {
                    // Transfer's destination corresponds to the input dataInfo, and its source to
                    // the output one.
                    let peer_row = match side {
                        StreamSide::Inputs => body.dsts.iter().find_map(|dst| {
                            (dst.stream.stored.stream.data_connect == Some(dc))
                                .then(|| comp_row_id(dst.unit))
                                .flatten()
                        }),
                        StreamSide::Outputs => comp_row_id(body.src.unit),
                    };
                    if peer_row != Some(row) {
                        continue;
                    }
                }
                PropEnd::Other
            }
            _ => continue,
        };
        tracker.add_prop_info(
            Propagation {
                ends: CoordPropInfo {
                    data_connect: Some(dc),
                    ref_node: curr_end,
                    node_to_fold: peer_end,
                },
                ref_node: curr,
                node_to_fold: peer,
                ref_role,
                scale_down: super::ScaleDown::No,
            },
            ref_dims,
        );
    }
}

/// THE PT-ROW FILTER FOR A COMPUTE PEER (`ddc/ddc_fold.cpp:1782-1800`) — the same row, or one row
/// north or south of it when both ends are computes.
///
/// ⚠️⚠️ TRAP, AND IT IS THE REFERENCE'S ARITHMETIC: `dcNodePtRowId == ptRowId - 1` HOLDS FOR A PEER
/// ON NO ROW AT ALL when `ptRowId` is 0, because both sides are then `-1`. The comparison is kept
/// signed for exactly that arm; `ptRowId + 1` can never be `-1`, so PT_SOUTH has no twin case.
fn row_matches_compute(
    row: PtRowId,
    peer: Option<PtRowId>,
    curr_end: PropEnd,
    component: SenComponent,
) -> bool {
    let ref_row = i16::from(row.ordinal());
    let peer_row = peer.map_or(-1, |peer| i16::from(peer.ordinal()));
    if let PropEnd::Compute { .. } = curr_end {
        // Allow ComputeNode -> ComputeNode with row difference of 1 to cover for PT_NORTH and
        // PT_SOUTH dataflow.
        (component == SenComponent::Ptnorth && peer_row == ref_row - 1)
            || (component == SenComponent::Ptsouth && peer_row == ref_row + 1)
            || peer_row == ref_row
    } else {
        peer_row == ref_row
    }
}

/// Replaces: e370_buildAndPropagateFold
///
/// Builds every external LX allocation's fold, then drains the propagation queue: each step builds
/// the fold of one compute, transfer or allocation and queues the datastream nodes behind it.
///
/// ⚠️ TRAP: `sdsc_->coreIdToWkSlice_.at(customCore).at(refDim)` throws where the SuperDSC has no
/// slice for that core or dim; a missing slice is a MISMATCH here, which drops the dim from `refDims`
/// rather than ending the run.
/// ⚠️ TRAP: the reference `static_cast<ComputeNode *>`s the ALLOCATE arm's ref node whatever its kind
/// (`:1892-1897`); a ref node that is neither a transfer nor a compute is skipped here.
/// ⚠️ TRAP: entry 238's [`None`] is the reference's `DT_ERROR`, which ENDS the run — the step is
/// dropped here instead. The reversed record's negated `refIsProducer` is unread either way:
/// [`ComputeCoordRef`] carries only the two fields entry 238 asks for.
pub fn build_and_propagate_fold<S: FoldDriver + ?Sized>(
    dsc: &mut S,
    metadata: &Metadata,
    tracker: &mut CoordPropTracker,
    report: CoordPropReport,
) -> String {
    let mut out = String::new();
    // Reserve placeholders for coordinates in computeNodes.
    for compute in dsc.computes() {
        let Some(body) = dsc.prop_compute(compute) else {
            continue;
        };
        let slots = match &body.opaque {
            Some(opaque) => opaque.inputs.len(),
            None => body.inputs.len(),
        };
        dsc.resize_input_coordinates(compute, slots);
    }
    tracker.reset();
    // TO DO: Consider moving the external allocations to prepDsc to simulate the computation in
    // upstream components. First, construct folds for the external allocateNodes.
    for lds in dsc.labeled_ds() {
        if report.labels() {
            out.push_str("\nLDS: ");
            out.push_str(&dsc.labeled_ds_text(lds.idx));
            out.push_str("\n  extAllocNode: ");
            match lds
                .external
                .and_then(|node| dsc.schedule_node(node).map(|node| node.name().0.clone()))
            {
                Some(name) => out.push_str(&name),
                None => out.push_str("<not found>"),
            }
        }
        let Some(external) = lds.external else {
            continue;
        };
        if lds.ds_type == DsType::Internal || !dsc.is_external_node(external) {
            continue;
        }
        dsc.build_fold_for_external_allocation(external);
        let coord = dsc.allocate_coord(external);
        let ref_dims: Vec<PrimaryDim> = if coord.work_slices.0.is_empty() {
            coord.dims
        } else {
            let sdsc = dsc.sdsc_work_slices();
            coord
                .dims
                .iter()
                .copied()
                .filter(|ref_dim| {
                    coord.work_slices.0.iter().all(|(core, per_dim)| {
                        per_dim.get(ref_dim).is_none_or(|slice| {
                            sdsc.0.get(core).and_then(|dims| dims.get(ref_dim)) == Some(slice)
                        })
                    })
                })
                .collect()
        };
        if !ref_dims.is_empty() {
            queue_allocation_users(dsc, tracker, external, &ref_dims);
        }
    }

    // Construct folds for below-LX scheduleNodes based on the folds of the external allocateNodes.
    while let Some(item) = tracker.next_item() {
        let node = item.prop.node_to_fold;
        if report.reports() {
            out.push_str("\n>>> CoordinatePropagationInfo:\n      refNode=");
            if let Some(ref_node) = dsc.schedule_node(item.prop.ref_node) {
                out.push_str(&dbg_print(ref_node));
            }
            out.push_str("\n      nodeTofold= ");
            if let Some(to_fold) = dsc.schedule_node(node) {
                out.push_str(&dbg_print(to_fold));
            }
            out.push_str(&format!(
                "\n      data_connect= {}, refIsProducer= {}\n      Dims:",
                item.prop
                    .ends
                    .data_connect
                    .map_or("", |connect| connect.spelling()),
                match item.prop.ref_role {
                    RefRole::Producer => "T",
                    RefRole::Consumer => "F",
                }
            ));
            for dim in &item.dims {
                out.push(' ');
                out.push_str(dim.spelling());
            }
            out.push('\n');
        }
        // Check if a fold is constructed already.
        match dsc.kind(node) {
            NodeKind::Compute => {
                let constructed = dsc.build_fold_for_compute(node, &item, tracker);
                let Some(body) = dsc.prop_compute(node) else {
                    continue;
                };
                if let Some(opaque) = &body.opaque {
                    for &at in &constructed.inputs {
                        let Some(&working) = opaque.inputs.get(at) else {
                            continue;
                        };
                        let dims = dsc.compute_coord_dims(node, OperandPos::Input(at));
                        queue_opaque_peers(
                            dsc,
                            metadata,
                            tracker,
                            node,
                            working,
                            &dims,
                            StreamSide::Inputs,
                        );
                    }
                    if !constructed.outputs.is_empty() {
                        // Include all outputs for propagation.
                        let dims = dsc.compute_coord_dims(node, OperandPos::Output(0));
                        for &working in &opaque.outputs {
                            queue_opaque_peers(
                                dsc,
                                metadata,
                                tracker,
                                node,
                                working,
                                &dims,
                                StreamSide::Outputs,
                            );
                        }
                    }
                    continue;
                }
                for (side, positions) in [
                    (StreamSide::Inputs, &constructed.inputs),
                    (StreamSide::Outputs, &constructed.outputs),
                ] {
                    for &at in positions {
                        let (streams, coord) = match side {
                            StreamSide::Inputs => (&body.inputs, OperandPos::Input(at)),
                            StreamSide::Outputs => (&body.outputs, OperandPos::Output(at)),
                        };
                        let Some(&stream) = streams.get(at) else {
                            continue;
                        };
                        let dims = dsc.compute_coord_dims(node, coord);
                        collect_and_process_datastream_nodes(
                            dsc,
                            metadata,
                            tracker,
                            node,
                            item.prop.ref_node,
                            stream,
                            &dims,
                            body.row,
                            side,
                        );
                    }
                }
            }
            NodeKind::Transfer => {
                if !dsc.build_fold_for_transfer(node, &item, tracker) {
                    continue;
                }
                let Some(dims) = dsc.transfer_coord_dims(node) else {
                    continue;
                };
                let Some(body) = dsc.prop_transfer(node) else {
                    continue;
                };
                for (end, side) in std::iter::once((body.src, StreamSide::Inputs))
                    .chain(body.dsts.iter().map(|&dst| (dst, StreamSide::Outputs)))
                {
                    collect_and_process_datastream_nodes(
                        dsc,
                        metadata,
                        tracker,
                        node,
                        item.prop.ref_node,
                        end.stream,
                        &dims,
                        comp_row_id(end.unit),
                        side,
                    );
                }
            }
            NodeKind::Allocate => {
                let reverse = ComputeCoordRef {
                    data_connect: item.prop.ends.data_connect,
                    ref_node: match item.prop.ends.node_to_fold {
                        PropEnd::Allocate { alloc, scaled } => {
                            ComputeRefNode::Allocation { alloc, scaled }
                        }
                        PropEnd::Compute { .. } | PropEnd::Other => ComputeRefNode::Streamed,
                    },
                };
                let ref_node = item.prop.ref_node;
                let reference = match dsc.kind(ref_node) {
                    NodeKind::Transfer => PropRefCoord::Transfer(ref_node),
                    NodeKind::Compute => match dsc.related_compute_coord(ref_node, &reverse) {
                        Some(selected) => PropRefCoord::Compute(ref_node, selected),
                        None => continue,
                    },
                    _ => continue,
                };
                if !dsc.build_fold_for_allocation(node, &item, reference, tracker) {
                    continue;
                }
                dsc.loop_elem_offsets_from_coordinates(reference, node, &item.dims, report);
                let coord = dsc.allocate_coord(node);
                if coord.work_slices.0.is_empty() {
                    queue_allocation_users(dsc, tracker, node, &coord.dims);
                }
            }
            kind => {
                if report.labels() {
                    out.push_str(&format!(
                        "[buildAndPropagateFold] Unsupported user node type {}",
                        kind.spelling()
                    ));
                }
            }
        }
    }
    out
}

/// `for (auto producer : metadata.dataConnects_.at(workingDc).producers_)` and its consumer twin
/// (`ddc/ddc_fold.cpp:1849-1873`) — an opaque op's peers are queued by `data_connect=` alone, with
/// no allocation lookup and no row filter.
fn queue_opaque_peers<S: FoldDriver + ?Sized>(
    dsc: &S,
    metadata: &Metadata,
    tracker: &mut CoordPropTracker,
    compute: NodeId,
    working: DataConnect,
    dims: &[PrimaryDim],
    side: StreamSide,
) {
    let Some(ends) = metadata.data_connects.get(&working) else {
        return;
    };
    let (peers, ref_role) = match side {
        StreamSide::Inputs => (ends.producers(), RefRole::Consumer),
        StreamSide::Outputs => (ends.consumers(), RefRole::Producer),
    };
    let curr_end = prop_end(dsc, compute);
    for &index in peers {
        let Some(peer) = dsc.node_at(index) else {
            continue;
        };
        tracker.add_prop_info(
            Propagation {
                ends: CoordPropInfo {
                    data_connect: Some(working),
                    ref_node: curr_end,
                    node_to_fold: prop_end(dsc, peer),
                },
                ref_node: compute,
                node_to_fold: peer,
                ref_role,
                scale_down: super::ScaleDown::No,
            },
            dims,
        );
    }
}

// ⭐ TESTS FOR ENTRY 370. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e370 {
    use super::*;
    use crate::schedule::ddc::CoordPropTracker;

    /// The three-node schedule the walk runs over — an external LX allocation on labeled DS 0, the
    /// compute that consumes it, and the allocation on labeled DS 1 that it writes — recording every
    /// fold the driver was asked to build.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Walk {
        resized: Vec<(NodeId, usize)>,
        externals: Vec<NodeId>,
        computes: Vec<NodeId>,
        allocs: Vec<NodeId>,
        offsets: Vec<(NodeId, Vec<PrimaryDim>)>,
    }

    /// The stream backed by labeled DS `lds`, which is how this double keys its allocations.
    fn stream(lds: u32) -> StoredStream {
        StoredStream {
            stream: DataStream {
                origin: DataOrigin::LabeledDs(LdsIdx(lds)),
                data_connect: None,
            },
            storage: DfirUnit::Lx,
        }
    }

    impl ScheduleTree for Walk {
        fn kind(&self, node: NodeId) -> NodeKind {
            if node == NodeId(2) {
                NodeKind::Compute
            } else {
                NodeKind::Allocate
            }
        }
        fn parent(&self, _node: NodeId) -> Option<NodeId> {
            None
        }
        fn children(&self, _block: BlockId) -> Vec<NodeId> {
            Vec::new()
        }
    }

    impl Allocations for Walk {
        fn allocation(&self, stored: StoredStream) -> Option<AllocId> {
            match stored.stream.origin {
                DataOrigin::LabeledDs(lds) => Some(AllocId(lds.0)),
                DataOrigin::Constant(_) => None,
            }
        }
        fn value_allocation(&self, _scale: AllocId) -> Option<AllocId> {
            None
        }
    }

    impl FoldDriver for Walk {
        fn computes(&self) -> Vec<NodeId> {
            vec![NodeId(2)]
        }
        fn resize_input_coordinates(&mut self, compute: NodeId, slots: usize) {
            self.resized.push((compute, slots));
        }
        fn labeled_ds(&self) -> Vec<PropLds> {
            vec![PropLds {
                idx: LdsIdx(0),
                ds_type: DsType::Input,
                external: Some(NodeId(0)),
            }]
        }
        fn labeled_ds_text(&self, _lds: LdsIdx) -> String {
            String::new()
        }
        fn is_external_node(&self, node: NodeId) -> bool {
            node == NodeId(0)
        }
        fn allocate_coord(&self, alloc: NodeId) -> AllocateCoord {
            let dim = if alloc == NodeId(0) {
                PrimaryDim::In
            } else {
                PrimaryDim::Out
            };
            AllocateCoord {
                dims: vec![dim],
                work_slices: WorkSlices::default(),
            }
        }
        fn sdsc_work_slices(&self) -> WorkSlices {
            WorkSlices::default()
        }
        fn allocation_users(&self, _alloc: NodeId) -> Vec<NodeId> {
            vec![NodeId(2)]
        }
        fn incoming_streams(&self, node: NodeId) -> Option<UserStreams> {
            (node == NodeId(2)).then(|| UserStreams::Compute(vec![stream(0)]))
        }
        fn prop_allocation(&self, node: NodeId) -> Option<PropAlloc> {
            (node != NodeId(2)).then(|| PropAlloc {
                node,
                alloc: AllocId(node.0),
                component: SenComponent::L0,
                lds: Some(LdsIdx(node.0)),
                scaled: None,
            })
        }
        fn allocation_node(&self, alloc: AllocId) -> Option<NodeId> {
            Some(NodeId(alloc.0))
        }
        fn related_scale_allocation(&self, _lds: LdsIdx, _memory: SenComponent) -> Option<NodeId> {
            None
        }
        fn prop_compute(&self, compute: NodeId) -> Option<PropCompute> {
            (compute == NodeId(2)).then(|| PropCompute {
                ex_unit: DfirUnit::Sfp,
                row: None,
                opaque: None,
                inputs: vec![PropStream {
                    stored: stream(0),
                    component: SenComponent::L0,
                }],
                outputs: vec![PropStream {
                    stored: stream(1),
                    component: SenComponent::L0,
                }],
            })
        }
        fn prop_transfer(&self, _transfer: NodeId) -> Option<PropTransfer> {
            None
        }
        fn compute_coord_dims(&self, _compute: NodeId, at: OperandPos) -> Vec<PrimaryDim> {
            match at {
                OperandPos::Input(_) => vec![PrimaryDim::In],
                OperandPos::Output(_) => vec![PrimaryDim::Out],
            }
        }
        fn transfer_coord_dims(&self, _transfer: NodeId) -> Option<Vec<PrimaryDim>> {
            None
        }
        fn node_at(&self, _index: NodeIndex) -> Option<NodeId> {
            None
        }
        fn schedule_node(&self, _node: NodeId) -> Option<Node<'_>> {
            None
        }
        fn build_fold_for_external_allocation(&mut self, alloc: NodeId) {
            self.externals.push(alloc);
        }
        fn build_fold_for_compute(
            &mut self,
            compute: NodeId,
            _item: &QueuedProp,
            _tracker: &mut CoordPropTracker,
        ) -> ConstructedCoords {
            self.computes.push(compute);
            ConstructedCoords {
                inputs: vec![0],
                outputs: vec![0],
            }
        }
        fn build_fold_for_transfer(
            &mut self,
            _transfer: NodeId,
            _item: &QueuedProp,
            _tracker: &mut CoordPropTracker,
        ) -> bool {
            false
        }
        fn build_fold_for_allocation(
            &mut self,
            alloc: NodeId,
            _item: &QueuedProp,
            _reference: PropRefCoord,
            _tracker: &mut CoordPropTracker,
        ) -> bool {
            self.allocs.push(alloc);
            true
        }
        fn related_compute_coord(
            &self,
            _compute: NodeId,
            _reference: &ComputeCoordRef,
        ) -> Option<RelatedCoord> {
            Some(RelatedCoord {
                coord: RelatedComputeCoord::Input(0),
                lds: Some(LdsIdx(0)),
            })
        }
        fn loop_elem_offsets_from_coordinates(
            &mut self,
            _reference: PropRefCoord,
            alloc: NodeId,
            dims: &[PrimaryDim],
            _report: CoordPropReport,
        ) {
            self.offsets.push((alloc, dims.to_vec()));
        }
    }

    /// e370: the external allocation's fold is built first, its compute user is propagated to, and the
    /// compute's own output allocation is reached behind it — then the walk comes BACK through the
    /// compute to the input allocation and stops, because the ledger already holds that pair's dim.
    #[test]
    fn the_external_fold_is_built_then_the_walk_propagates_both_ways_and_terminates() {
        let mut walk = Walk::default();
        let mut tracker = CoordPropTracker::default();

        let out = build_and_propagate_fold(
            &mut walk,
            &Metadata::default(),
            &mut tracker,
            CoordPropReport::Labels,
        );

        // One placeholder per non-opaque input, and the external LX allocation's fold before any
        // propagation step.
        assert_eq!(walk.resized, vec![(NodeId(2), 1)]);
        assert_eq!(walk.externals, vec![NodeId(0)]);
        // Allocation 0 -> compute -> allocation 1 -> compute -> allocation 0, and no sixth step.
        assert_eq!(walk.computes, vec![NodeId(2), NodeId(2)]);
        assert_eq!(walk.allocs, vec![NodeId(1), NodeId(0)]);
        assert_eq!(
            walk.offsets,
            vec![
                (NodeId(1), vec![PrimaryDim::Out]),
                (NodeId(0), vec![PrimaryDim::In]),
            ]
        );
        assert!(out.starts_with("\nLDS: \n  extAllocNode: <not found>"));
        assert_eq!(out.matches(">>> CoordinatePropagationInfo:").count(), 4);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ USES FOR ENTRY 375.
// ════════════════════════════════════════════════════════════════════════════════════════════════

use crate::schedule::dsc2::NodeName;

/// HOW ENTRY 375 REACHES `buildAndPropagateFold()` (`ddc/ddc_fold.cpp:1625`) — a DSC's own call into
/// the landed [`build_and_propagate_fold`], which is where the fold this report reads off every node
/// is built. Only the report itself is entry 375's.
///
/// ⛔ THE SEAM, AND NOT A STAND-IN: an implementation that does anything but call
/// [`build_and_propagate_fold`] over its own [`FoldDriver`] leaves every coordinate unbuilt and
/// the report empty. The call is not made here because the driver takes the `Metadata`, the
/// [`CoordPropTracker`] and the report level, which are the `Ddc`'s and not this trait's.
pub trait FoldConstruction {
    /// `buildAndPropagateFold()`.
    fn build_and_propagate_fold(&mut self);
}

/// `debugPrint`'s `printContent` (`dsc/dsc2.h:361`) as entry 375 sets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordContent {
    /// `printContent = true`.
    Included,
    /// `printContent = false`.
    Omitted,
}

/// `coordFoldReportLevel_` AS ITS THREE THRESHOLDS — `> 0`, `> 1` and `> 2` is the whole of what the
/// level means to entry 375.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CoordFoldReport {
    /// `0` — no report.
    #[default]
    Off,
    /// `1` — every node's header and every coordinate.
    Coordinates,
    /// `2` — the same, with `printContent` set.
    CoordinateContent,
    /// `3` and above — the same, plus each node's own `print`.
    NodeContent,
}

impl CoordFoldReport {
    /// `coordFoldReportLevel_ > 0`.
    const fn reports(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// `coordFoldReportLevel_ > 1`.
    const fn content(self) -> CoordContent {
        match self {
            Self::Off | Self::Coordinates => CoordContent::Omitted,
            Self::CoordinateContent | Self::NodeContent => CoordContent::Included,
        }
    }

    /// `coordFoldReportLevel_ > 2`.
    const fn prints_nodes(self) -> bool {
        matches!(self, Self::NodeContent)
    }
}

/// ONE NODE OF THE CAPTURE WALK WITH THE COORDINATES THE FOLD GAVE IT —
/// `traverseTreeDFSMutable(nullptr, {ALLOCATE, COMPUTE, TRANSFER})` projected onto exactly the
/// fields entry 375 reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturedNode {
    /// `ALLOCATE` — `name_`, `component_` and `allocateCoordinates_`.
    Allocate {
        /// The node itself, for `allocNode->print`.
        node: NodeId,
        /// `name_`.
        name: NodeName,
        /// `component_`.
        memory: SenComponent,
        /// `allocateCoordinates_`.
        coordinates: dsc2::Coordinate,
    },
    /// `TRANSFER` — its two ends and `transferCoordinates_`.
    Transfer {
        /// The node itself, for `transferNode->print`.
        node: NodeId,
        /// `name_`, `src_` and `dstVias_`.
        body: TransferNode,
        /// `transferCoordinates_`.
        coordinates: dsc2::Coordinate,
    },
    /// `COMPUTE` — its op and operands, `inputCoordinates_` and `outputCoordinate_`.
    Compute {
        /// The node itself, for `computeNode->print`.
        node: NodeId,
        /// `name_`, `type_`, `inputs_` and `outputs_`.
        body: ComputeNode,
        /// `inputCoordinates_`, in order.
        inputs: Vec<dsc2::Coordinate>,
        /// `outputCoordinate_`.
        output: dsc2::Coordinate,
    },
}

/// WHAT ENTRY 375 READS TO WRITE ITS REPORT — the capture walk, plus the two `dsc/dsc2.h` printers
/// that sit outside this campaign's file census.
pub trait CoordinateCapture: FoldConstruction {
    /// `traverseTreeDFSMutable(nullptr, {ALLOCATE, COMPUTE, TRANSFER})`, in the traversal's order.
    fn captured_nodes(&self) -> Vec<CapturedNode>;

    /// `coord.debugPrint(std::cout, printContent)` (`dsc/dsc2.h:361`).
    fn coordinate_text(&self, coord: &dsc2::Coordinate, content: CoordContent) -> String;

    /// `node->print(std::cout)` — `AllocateNode`, `TransferNode` and `ComputeNode` each override it.
    fn node_text(&self, node: NodeId) -> String;
}

/// `"\n================================"` — 32 columns, per node.
const CAPTURE_RULE: &str = "\n================================";

/// `"\n--------------------------"` — 26 columns, around each input's number.
const INPUT_RULE: &str = "\n--------------------------";

/// Replaces: e375_coordinateCapture
///
/// Builds and propagates the fold, then reports every allocate, transfer and compute node together
/// with the coordinates that fold gave it.
///
/// ⚠️ TRAP: LEVEL 2 PRINTS EXACTLY WHAT LEVEL 1 PRINTS — `debugPrint` never reads its `printContent`
/// or `ps` parameters (`dsc/dsc2.h:361-429`), so the `> 1` argument is inert.
/// ⚠️ TRAP: a compute's `"(storage=..)"` labels read `inputs_`/`outputs_` (`dsc/dsc2.h:935-936`),
/// which are the COMPONENT vectors carried by [`Operand::unit`] here — the reference pairs
/// `inputs_[i]` as the *storage* half of `DataLocation{exUnit_, inputs_[i]}` when it fills the
/// operand's data (`ddc/ddcv1.cpp:3045-3056`), which is why the label spells "storage=".
/// ⚠️ TRAP: the reference bounds these loops by `inputs_`/`outputs_` and indexes
/// `..LdsAndLoopOffsets_` with the same `i`; its own `DT_ERROR` at `ddc/ddcv1.cpp:3040-3043` rejects
/// any compute whose two vectors disagree, so the fused [`Operand`] cannot spell that `.at` throw.
pub fn coordinate_capture<S: CoordinateCapture + ?Sized>(
    dsc: &mut S,
    report: CoordFoldReport,
) -> String {
    dsc.build_and_propagate_fold();
    if !report.reports() {
        return String::new();
    }

    let content = report.content();
    let mut out = String::new();
    for captured in dsc.captured_nodes() {
        out.push_str(CAPTURE_RULE);
        match captured {
            CapturedNode::Allocate {
                node,
                name,
                memory,
                coordinates,
            } => {
                out.push_str(&format!("\nAllocateNode: {}", name.0));
                out.push_str(&format!("\nMemory: {}", memory.spelling()));
                if report.prints_nodes() {
                    out.push_str(&dsc.node_text(node));
                }
                out.push_str(&dsc.coordinate_text(&coordinates, content));
            }
            CapturedNode::Transfer {
                node,
                body,
                coordinates,
            } => {
                out.push_str(&format!("\nTransferNode: {}", body.name.0));
                let dst = body.dsts.first();
                out.push_str(&format!(
                    "\nsrc storage: {}, src unit : {}, dst storage: {}, dst unit: {}",
                    body.src.storage.spelling(),
                    body.src.unit.spelling(),
                    dst.storage.spelling(),
                    dst.unit.spelling()
                ));
                if report.prints_nodes() {
                    out.push_str(&dsc.node_text(node));
                }
                out.push_str(&dsc.coordinate_text(&coordinates, content));
            }
            CapturedNode::Compute {
                node,
                body,
                inputs,
                output,
            } => {
                out.push_str(&format!("\nComputeNode: {}", body.name.0));
                out.push_str(&format!("\nOp: {}", body.op.cpp_spelling()));
                out.push_str(", inputs:[ ");
                for input in &body.inputs {
                    out.push_str(&format!("(storage={}) ", input.unit.spelling()));
                }
                out.push_str("], outputs:[ ");
                for output in &body.outputs {
                    out.push_str(&format!("(storage={}) ", output.unit.spelling()));
                }
                out.push(']');
                if report.prints_nodes() {
                    out.push_str(&dsc.node_text(node));
                }
                out.push_str("\n\nInput coordinates:");
                for (index, coord) in inputs.iter().enumerate() {
                    // The reference's `std::endl` is the newline that closes the second rule.
                    out.push_str(&format!("{INPUT_RULE}\n  input= {index}{INPUT_RULE}\n"));
                    out.push_str(&dsc.coordinate_text(coord, content));
                }
                out.push_str("\n\nOutput coordinates:");
                out.push_str(&dsc.coordinate_text(&output, content));
            }
        }
    }
    out
}

// ⭐ TESTS FOR ENTRY 375.
#[cfg(test)]
mod tests_e375 {
    use super::*;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        DataInfo, Dsts, InstrAttribute, NodeName, NumChunks, ReplicationFactor, TransferPadding,
    };
    use crate::units::NumFolds;
    use std::cell::Cell;

    /// A DSC WHOSE THREE CAPTURED NODES COVER ALL THREE ARMS, and whose two printers TAG what they
    /// were handed so the report's order and its `printContent` argument are both readable.
    struct Report {
        built: Cell<u32>,
        coords: Cell<u32>,
        nodes: Vec<CapturedNode>,
    }

    impl FoldConstruction for Report {
        fn build_and_propagate_fold(&mut self) {
            self.built.set(self.built.get() + 1);
        }
    }

    impl CoordinateCapture for Report {
        fn captured_nodes(&self) -> Vec<CapturedNode> {
            self.nodes.clone()
        }

        fn coordinate_text(&self, _coord: &dsc2::Coordinate, content: CoordContent) -> String {
            let index = self.coords.get();
            self.coords.set(index + 1);
            format!("\n<coord {index} {content:?}>")
        }

        fn node_text(&self, node: NodeId) -> String {
            format!("\n<print {}>", node.0)
        }
    }

    fn operand(unit: SenComponent, storage: SenComponent) -> Operand {
        Operand {
            unit,
            storage,
            data: DataInfo {
                data_connect: None,
                my_lds_idx: None,
                constant_id: None,
                latch_data_id: None,
                ..DataInfo::EMPTY
            },
        }
    }

    fn report() -> Report {
        let transfer = TransferNode {
            repetition: dsc2::TransferRepetition::default(),
            last_fusable_parent_loop_src: None,
            last_fusable_parent_loop_dst: Vec::new(),
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: None,
            corelet_views: BTreeMap::new(),
            transfer_coordinates: dsc2::Coordinate::default(),
            name: NodeName("t0".to_owned()),
            src: operand(SenComponent::L3lu, SenComponent::Hbm),
            dsts: Dsts::new(operand(SenComponent::Lxlu0, SenComponent::Lx), Vec::new()),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
        };
        let compute = ComputeNode {
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: dsc2::Coordinate::default(),
            repetition_with_offset: dsc2::RepetitionWithOffset::default(),
            name: NodeName("c0".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Pe,
            // A compute operand's component is `inputs_[i]` / `outputs_[i]`, which this tree carries
            // in [`Operand::unit`]; `storage` is unset so the expected text pins the field read.
            inputs: vec![
                operand(SenComponent::Pelrf, SenComponent::NoComponent),
                operand(SenComponent::Sfplrf, SenComponent::NoComponent),
            ],
            outputs: vec![operand(SenComponent::Lx, SenComponent::NoComponent)],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        };
        Report {
            built: Cell::new(0),
            coords: Cell::new(0),
            nodes: vec![
                CapturedNode::Allocate {
                    node: NodeId(1),
                    name: NodeName("allocate_lds0_lx".to_owned()),
                    memory: SenComponent::Lx,
                    coordinates: dsc2::Coordinate::default(),
                },
                CapturedNode::Transfer {
                    node: NodeId(2),
                    body: transfer,
                    coordinates: dsc2::Coordinate::default(),
                },
                CapturedNode::Compute {
                    node: NodeId(3),
                    body: compute,
                    inputs: vec![dsc2::Coordinate::default(), dsc2::Coordinate::default()],
                    output: dsc2::Coordinate::default(),
                },
            ],
        }
    }

    #[test]
    fn the_top_level_reports_every_node_with_the_coordinates_the_fold_gave_it() {
        let mut dsc = report();
        let text = coordinate_capture(&mut dsc, CoordFoldReport::NodeContent);
        assert_eq!(dsc.built.get(), 1);
        // The two rules are 32 and 26 columns wide, counted rather than transcribed.
        assert_eq!(text.matches(&"=".repeat(32)).count(), 3);
        assert!(!text.contains(&"=".repeat(33)));
        assert_eq!(text.matches(&"-".repeat(26)).count(), 4);
        assert!(!text.contains(&"-".repeat(27)));
        assert_eq!(
            text,
            "\n================================\
             \nAllocateNode: allocate_lds0_lx\
             \nMemory: lx\
             \n<print 1>\
             \n<coord 0 Included>\
             \n================================\
             \nTransferNode: t0\
             \nsrc storage: hbm, src unit : l3lu, dst storage: lx, dst unit: lxlu0\
             \n<print 2>\
             \n<coord 1 Included>\
             \n================================\
             \nComputeNode: c0\
             \nOp: macc, inputs:[ (storage=pelrf) (storage=sfplrf) ], outputs:[ (storage=lx) ]\
             \n<print 3>\
             \n\nInput coordinates:\
             \n--------------------------\
             \n  input= 0\
             \n--------------------------\
             \n\n<coord 2 Included>\
             \n--------------------------\
             \n  input= 1\
             \n--------------------------\
             \n\n<coord 3 Included>\
             \n\nOutput coordinates:\
             \n<coord 4 Included>"
        );
    }

    #[test]
    fn level_zero_still_builds_the_fold_and_reports_nothing() {
        let mut dsc = report();
        assert!(coordinate_capture(&mut dsc, CoordFoldReport::Off).is_empty());
        assert_eq!(dsc.built.get(), 1);
    }
}

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
                ..DataInfo::EMPTY
            },
        }
    }

    #[test]
    fn a_compute_line_names_every_operand_with_its_own_connect() {
        let node = ComputeNode {
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: dsc2::Coordinate::default(),
            repetition_with_offset: dsc2::RepetitionWithOffset::default(),
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
            repetition: dsc2::TransferRepetition::default(),
            last_fusable_parent_loop_src: None,
            last_fusable_parent_loop_dst: Vec::new(),
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: None,
            corelet_views: BTreeMap::new(),
            transfer_coordinates: dsc2::Coordinate::default(),
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
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: dsc2::Coordinate::default(),
            repetition_with_offset: dsc2::RepetitionWithOffset::default(),
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
            repetition: dsc2::TransferRepetition::default(),
            last_fusable_parent_loop_src: None,
            last_fusable_parent_loop_dst: Vec::new(),
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: None,
            corelet_views: BTreeMap::new(),
            transfer_coordinates: dsc2::Coordinate::default(),
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
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: dsc2::Coordinate::default(),
            repetition_with_offset: dsc2::RepetitionWithOffset::default(),
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
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: dsc2::Coordinate::default(),
            repetition_with_offset: dsc2::RepetitionWithOffset::default(),
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
                ..DataInfo::EMPTY
            },
        }
    }

    fn transfer_from_lds0() -> TransferNode {
        TransferNode {
            repetition: dsc2::TransferRepetition::default(),
            last_fusable_parent_loop_src: None,
            last_fusable_parent_loop_dst: Vec::new(),
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: None,
            corelet_views: BTreeMap::new(),
            transfer_coordinates: dsc2::Coordinate::default(),
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
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            input_coordinates: Vec::new(),
            output_coordinate: dsc2::Coordinate::default(),
            repetition_with_offset: dsc2::RepetitionWithOffset::default(),
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
            propagated_dims: &[PrimaryDim::Out],
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
    fn a_corelet_split_dim_that_is_requested_but_carries_no_fold_still_stops_the_propagation() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, Vec::new()));
        let core_ds = RowSplitOn(Vec::new());
        let distribution = Seams(Vec::new());
        let slices = WorkSlices(BTreeMap::from([
            (
                CoreOrdinal(0),
                BTreeMap::from([(PrimaryDim::Out, WorkSlice(0))]),
            ),
            (
                CoreOrdinal(1),
                BTreeMap::from([(PrimaryDim::Out, WorkSlice(1))]),
            ),
        ]));
        // `dims` is EMPTY — the split dim is asked for and the coordinate has no fold manager for it,
        // so only `propagated_dims` can see the request the reference refuses on (`:8429`).
        let mut alloc: AllocationFold<'_, Levels> = AllocationFold {
            lds: LdsIdx(0),
            dims: Vec::new(),
            work_slices: &slices,
            dim_scales: BTreeMap::new(),
            propagated_dims: &[PrimaryDim::Out],
            corelet_split_dim: Some(PrimaryDim::Out),
            components: RefComponents {
                size: SenComponent::All,
                prop: SenComponent::Lx,
            },
            row_group: None,
        };
        let target = AllocFoldTarget {
            node: NodeId(1),
            loops: &[],
        };
        let mut coord = dsc2::Coordinate::default();
        assert_eq!(
            build_fold_from_allocation::<Target, _, _, _, _>(
                &dsc,
                &core_ds,
                &distribution,
                &alloc,
                &target,
                &mut coord,
                &mut (),
            ),
            FoldPropagation::CoreletSplitVaries
        );
        // The negative: the same varying slice map is no obstacle to a dim nobody asked to propagate.
        alloc.propagated_dims = &[];
        let mut coord = dsc2::Coordinate::default();
        assert_eq!(
            build_fold_from_allocation::<Target, _, _, _, _>(
                &dsc,
                &core_ds,
                &distribution,
                &alloc,
                &target,
                &mut coord,
                &mut (),
            ),
            FoldPropagation::Built
        );
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

    /// ⭐ THE TWO STOPS THAT ARE NOT WHERE THEY LOOK: the adjacency `DT_CHECK` (`:1184`) sits PAST
    /// the `rowSplit_` return, and a matched operand ON `NO_COMPONENT` is a stop and not a miss.
    #[test]
    fn non_adjacent_rows_stop_only_past_the_rowsplit_return_and_a_no_component_match_stops_too() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::In]));
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
        let on_ptrow3 = on_row(SenComponent::Ptrow3);
        let unresolved = on_row(SenComponent::NoComponent);

        // Row 3 feeding a compute on row 5: both units resolve, non-adjacent as they are.
        let north_south = PtRowPropagation::of(
            &dsc,
            &propagation(),
            RefRole::Producer,
            &PropUnits::Other,
            &PropUnits::Compute {
                ex_unit: SenComponent::Ptrow5,
                inputs: &on_ptrow3,
                outputs: &[],
            },
        )
        .expect("row 3 and row 5 are both units");
        let gather = |core_ds: &RowSplitOn| {
            gather_related_pt_rows::<Target, _, _, _, Levels>(
                &dsc,
                core_ds,
                &tree,
                &propagation(),
                north_south,
                NodeId(0),
                &coord,
                crate::schedule::ddc::ScaleDown::No,
                &no_candidates,
                &slices,
            )
        };

        // Nothing row split: the reference answers before it ever asks about adjacency.
        assert_eq!(gather(&RowSplitOn(vec![])), Some(PtRowGrouping::NoBundling));
        // Past that answer, rows 3 and 5 are two apart and the `DT_CHECK` fires.
        assert_eq!(gather(&RowSplitOn(vec![PrimaryDim::Out])), None);

        // A matched input sitting on `NO_COMPONENT` is the `DT_CHECK` on `propSrcUnit`.
        assert_eq!(
            PtRowPropagation::of(
                &dsc,
                &propagation(),
                RefRole::Producer,
                &PropUnits::Other,
                &PropUnits::Compute {
                    ex_unit: SenComponent::Ptrow3,
                    inputs: &unresolved,
                    outputs: &[],
                },
            ),
            None
        );
    }
}

#[cfg(test)]
mod tests_e356_e358 {
    use super::*;
    use crate::arch::Target;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{PaddedExtent, Sample};
    use crate::schedule::ddc::transformation_util::LoopDims;
    use crate::schedule::dsc2::{DataInfo, LayoutDims, NodeName};

    /// One labelled data structure whose layout order is a single dim.
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

    /// A core data stage that splits the dims it names across PT rows and no corelet; the EMPTY set
    /// is what leaves the row-bundling gate answering the default group without reading the scan.
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

    impl SizeStage for RowSplitOn {
        fn padded(&self, _dim: PrimaryDim, val: i64) -> Alpha {
            Alpha(val)
        }
    }

    /// A tree answered from a parent table; node 0 is the block that holds the rest.
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

    /// `loopParamInfo` as the two levels-recording seams write it.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Params(Vec<(PrimaryDim, ElemArrLevel)>);

    impl LoopLevels for Params {
        fn bump_levels_at_or_inside(&mut self, dim: PrimaryDim, level: ElemArrLevel) {
            self.0.push((dim, level));
        }
        fn record_loop_level(
            &mut self,
            _loop_node: &LoopNode,
            dim: PrimaryDim,
            _decided: DistributedLoop,
            level: ElemArrLevel,
        ) {
            self.0.push((dim, level));
        }
    }

    /// The dsc2 seams, tabulated: every loop walks every dim and the distributor answers one fixed
    /// stride.
    struct Seams;

    impl TemporalLoopDistribution for Seams {
        type LoopParams = Params;

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
            Vec::new()
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

    /// No node has an enclosing loop of its own; each unit's loop chain is handed over as a witness
    /// field instead, so this seam is only reached for the row-bundling ancestor.
    struct NoLoops;

    impl EnclosingLoops for NoLoops {
        fn enclosing_loops(&self, _node: NodeId) -> Vec<&LoopNode> {
            Vec::new()
        }
        fn enclosing_loop_chain(
            &self,
            _node: NodeId,
            _build_corelet_fold: bool,
        ) -> Vec<LoopAndDim<'_>> {
            Vec::new()
        }
    }

    struct Named;

    impl NodeNames for Named {
        fn name(&self, _node: NodeId) -> &str {
            "n"
        }
    }

    /// The loop element offsets, recorded per node — the ALLOCATE arms' second effect.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Offsets(Vec<NodeId>);

    impl LoopElemOffsets for Offsets {
        fn compute_loop_elem_offsets(
            &mut self,
            node: NodeId,
            _coord: &dsc2::Coordinate,
            _alloc: NodeId,
            _dims: &[PrimaryDim],
        ) {
            self.0.push(node);
        }
    }

    /// No data stage pairs a window dim with a padded one.
    struct NoWindows;

    impl WindowDims for NoWindows {
        fn window_dim(&self, _stage: DatastageId, _dim: PrimaryDim) -> Option<PrimaryDim> {
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

    /// The reference's coordinate for one dim: the three role folds and one element arrangement.
    fn ref_coordinate(dim: PrimaryDim) -> dsc2::Coordinate {
        let mut coord = dsc2::Coordinate::default();
        let mut front = |category, cardinality, label: &str, alpha| {
            coord.add_fold_front(
                dim,
                category,
                FoldCardinality(cardinality),
                dsc2::FoldLabel(label.to_owned()),
                FoldCoeff(alpha),
                FoldCoeff(0),
            );
        };
        // Added at the FRONT, so the innermost level goes on first.
        front(CoordinateCategory::ElemArr, 8, "elem_arr_0", 1);
        front(CoordinateCategory::Spatial, 1, "rowsplit_fold", 0);
        front(CoordinateCategory::Spatial, 2, "corelet_fold_dim", 50);
        front(CoordinateCategory::Spatial, 2, "core_workslice_fold_dim", 100);
        coord
    }

    /// What [`ref_coordinate`] propagates to: the same four levels, the row split recomputed to the
    /// identity fold it already was.
    fn propagated() -> Vec<(String, i64, u32)> {
        vec![
            ("core_workslice_fold_dim".to_owned(), 100, 2),
            ("corelet_fold_dim".to_owned(), 50, 2),
            ("rowsplit_fold".to_owned(), 0, 1),
            ("elem_arr_0".to_owned(), 1, 8),
        ]
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

    /// A step matched on its `data_connect=` alone, so no allocation lookup is reached.
    fn propagation(ref_node: NodeId, node_to_fold: NodeId) -> Propagation {
        Propagation {
            ends: CoordPropInfo {
                data_connect: Some(DataConnect::AconstConnect),
                ref_node: PropEnd::Other,
                node_to_fold: PropEnd::Other,
            },
            ref_node,
            node_to_fold,
            ref_role: RefRole::Producer,
            scale_down: crate::schedule::ddc::ScaleDown::No,
        }
    }

    fn scan<'a>(slices: &'a WorkSlices) -> RowBundlingScan<'a, dsc2::FoldDim> {
        RowBundlingScan {
            units: PtRowPropagation {
                src: SenComponent::Lx,
                dest: SenComponent::L0,
            },
            ref_node: NodeId(2),
            candidates: RowCandidates {
                forward: &[],
                reverse: &[],
            },
            sdsc_slices: slices,
        }
    }

    fn stored(lds: u32) -> StoredStream {
        StoredStream {
            stream: DataStream {
                origin: DataOrigin::LabeledDs(LdsIdx(lds)),
                data_connect: Some(DataConnect::AconstConnect),
            },
            storage: DfirUnit::L0,
        }
    }

    fn end(unit: SenComponent, connect: Option<DataConnect>) -> TransferFoldEnd<'static> {
        TransferFoldEnd {
            operand: Operand {
                unit,
                storage: SenComponent::NoComponent,
                data: DataInfo {
                    data_connect: connect,
                    my_lds_idx: Some(LdsIdx(0)),
                    constant_id: None,
                    latch_data_id: None,
                    ..DataInfo::EMPTY
                },
            },
            alloc: None,
        }
    }

    #[test]
    fn an_l0_allocation_below_its_reference_gains_both_its_own_folds_and_the_slice_views() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, Vec::new()));
        let core_ds = RowSplitOn(Vec::new());
        let tree = Tree(vec![None, Some(0), Some(0)]);
        let meta = Metadata::default();
        let ctx = FoldContext {
            dsc: &dsc,
            core_ds: &core_ds,
            tree: &tree,
            distribution: &Seams,
            ancestors: &NoLoops,
            names: &Named,
            meta: &meta,
        };
        let layout = AllocLayout(vec![(PrimaryDim::Out, None)]);
        let padding = PaddingForm::default();
        let slices = WorkSlices::default();
        let sticks = StickDims::default();
        let alloc = AllocationFoldTarget {
            node: NodeId(1),
            lds: LdsIdx(0),
            component: SenComponent::L0,
            layout: &layout,
            padding: &padding,
            work_slices: &slices,
            dim_scales: BTreeMap::new(),
            scale_up: None,
            scale_down: None,
            stick_dims: &sticks,
            corelet_split_dim: None,
            size_stage: &core_ds,
            ref_kind: RefNode::Other,
            consistency: None,
        };
        let ref_coord = ref_coordinate(PrimaryDim::Out);
        let mut coord = dsc2::Coordinate::default();
        let mut slice_view = dsc2::Coordinate::default();
        let mut tracker = CoordPropTracker::default();
        let mut params = Params::default();

        assert!(build_fold_for_allocation::<Target, _, _, _, _, _, _, _>(
            &ctx,
            &alloc,
            &propagation(NodeId(2), NodeId(1)),
            &[PrimaryDim::Out],
            &scan(&slices),
            &ref_coord,
            &slices,
            &mut coord,
            &mut slice_view,
            &mut tracker,
            &mut params,
        ));
        assert_eq!(folds_of(&coord, PrimaryDim::Out), propagated());
        // ⭐ THE L0 ARM BUILDS A SECOND COORDINATE over the same levels — a port that wrote only the
        // allocate one would pass every assertion above it.
        assert_eq!(folds_of(&slice_view, PrimaryDim::Out), propagated());
        assert!(coord.fold_constructed() && slice_view.fold_constructed());
    }

    #[test]
    fn a_computes_second_input_on_the_same_labeled_ds_takes_the_whole_coordinate_over() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, Vec::new()));
        let core_ds = RowSplitOn(Vec::new());
        let tree = Tree(vec![None, Some(0), Some(0)]);
        let meta = Metadata::default();
        let ctx = FoldContext {
            dsc: &dsc,
            core_ds: &core_ds,
            tree: &tree,
            distribution: &Seams,
            ancestors: &NoLoops,
            names: &Named,
            meta: &meta,
        };
        let padding = PaddingForm::default();
        let labeled_ds = BTreeMap::from([(
            LdsIdx(0),
            LabeledDsFold {
                dim_scales: BTreeMap::new(),
                stick_sizes: BTreeMap::new(),
                padding: &padding,
                category: ScaledLds::Regular,
                mx: None,
            },
        )]);
        let inputs_streams = [stored(0), stored(0)];
        let loop_node = loop_named("loop0");
        let non_alloc_loops = [(&loop_node, LoopDistribution::AboveChunk)];
        let compute = ComputeFold {
            node: NodeId(1),
            ex_unit: SenComponent::Ptrow0,
            inputs: &inputs_streams,
            outputs: &[],
            opaque: None,
            labeled_ds: &labeled_ds,
            bundling_producers: &[],
            output_padding: None,
            alloc_loops: &[],
            non_alloc_loops: &non_alloc_loops,
        };
        let ref_coord = ref_coordinate(PrimaryDim::Out);
        let ref_loops = [&loop_node];
        let slices = WorkSlices::default();
        let dims = [PrimaryDim::Out];
        let reference = || FoldReference {
            gate: RowBundling {
                scan: scan(&slices),
                ref_coord: RowCoordinate {
                    dims: ref_coord.iter().collect(),
                    work_slices: &slices,
                },
                ref_slice_view: None,
                retry_dims: &dims,
            },
            kind: FoldReferenceKind::NonAlloc {
                node: NodeId(2),
                dims: vec![(PrimaryDim::Out, coord_dim_of(&ref_coord, PrimaryDim::Out))],
                dim_scales: BTreeMap::new(),
                loops: &ref_loops,
                prop_comp: NonAllocPropComp::TransferEnds {
                    src: SenComponent::Lx,
                    dst: SenComponent::L0,
                },
            },
            ref_padding: &padding,
        };
        let selected = RelatedCoord {
            coord: RelatedComputeCoord::Input(0),
            lds: Some(LdsIdx(0)),
        };
        let prop = propagation(NodeId(2), NodeId(1));
        let mut inputs = vec![dsc2::Coordinate::default(), dsc2::Coordinate::default()];
        let mut output = dsc2::Coordinate::default();
        let mut constructed = ConstructedCoords::default();
        let mut tracker = CoordPropTracker::default();
        let mut params = Params::default();
        let mut offsets = Offsets::default();

        let built = build_fold_for_compute::<Target, _, _, _, _, _, _, _, _, _>(
            &ctx,
            &compute,
            &NoWindows,
            &prop,
            &dims,
            reference(),
            selected,
            &mut ComputeCoordinates {
                inputs: &mut inputs,
                output: &mut output,
            },
            &mut constructed,
            &mut tracker,
            &mut params,
            &mut offsets,
        );
        assert!(built);
        assert_eq!(folds_of(&inputs[0], PrimaryDim::Out), propagated());
        // The second input shares the working labeled DS, so it takes the whole coordinate.
        assert_eq!(inputs[1], inputs[0]);
        assert_eq!(constructed.inputs, vec![0, 1]);

        // ⛔ THE TRAP: a compute that is its own reference answers BEFORE either out-vector is
        // cleared, so a caller's earlier record survives.
        let mut untouched = ConstructedCoords {
            inputs: vec![7],
            outputs: vec![9],
        };
        assert!(!build_fold_for_compute::<Target, _, _, _, _, _, _, _, _, _>(
            &ctx,
            &compute,
            &NoWindows,
            &Propagation {
                ref_node: compute.node,
                ..prop
            },
            &dims,
            reference(),
            selected,
            &mut ComputeCoordinates {
                inputs: &mut inputs,
                output: &mut output,
            },
            &mut untouched,
            &mut tracker,
            &mut params,
            &mut offsets,
        ));
        assert_eq!(
            untouched,
            ConstructedCoords {
                inputs: vec![7],
                outputs: vec![9],
            }
        );
    }

    #[test]
    fn a_transfer_naming_its_reference_on_its_own_source_takes_that_references_levels() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, Vec::new()));
        let core_ds = RowSplitOn(Vec::new());
        let tree = Tree(vec![None, Some(0), Some(0)]);
        let meta = Metadata::default();
        let ctx = FoldContext {
            dsc: &dsc,
            core_ds: &core_ds,
            tree: &tree,
            distribution: &Seams,
            ancestors: &NoLoops,
            names: &Named,
            meta: &meta,
        };
        let loop_node = loop_named("loop0");
        let non_alloc_loops = [(&loop_node, LoopDistribution::AboveChunk)];
        let transfer_dims = [PrimaryDim::Out];
        let transfer_size = BTreeMap::new();
        let transfer = TransferFold {
            node: NodeId(1),
            // The source states the propagation's own `data_connect=`, which is what makes the
            // reference the SOURCE end.
            src: end(SenComponent::Lx, Some(DataConnect::AconstConnect)),
            dsts: TransferFoldDsts {
                first: end(SenComponent::L0, None),
                rest: Vec::new(),
            },
            transfer_dims: &transfer_dims,
            transfer_size: &transfer_size,
            alloc_loops: &[],
            non_alloc_loops: &non_alloc_loops,
            pe_sfp_transfer_size: None,
        };
        let padding = PaddingForm::default();
        let ref_coord = ref_coordinate(PrimaryDim::Out);
        let ref_loops = [&loop_node];
        let slices = WorkSlices::default();
        let dims = [PrimaryDim::Out];
        let reference = FoldReference {
            gate: RowBundling {
                scan: scan(&slices),
                ref_coord: RowCoordinate {
                    dims: ref_coord.iter().collect(),
                    work_slices: &slices,
                },
                ref_slice_view: None,
                retry_dims: &dims,
            },
            kind: FoldReferenceKind::NonAlloc {
                node: NodeId(2),
                dims: vec![(PrimaryDim::Out, coord_dim_of(&ref_coord, PrimaryDim::Out))],
                dim_scales: BTreeMap::new(),
                loops: &ref_loops,
                prop_comp: NonAllocPropComp::TransferEnds {
                    src: SenComponent::Lx,
                    dst: SenComponent::L0,
                },
            },
            ref_padding: &padding,
        };
        let mut coord = dsc2::Coordinate::default();
        let mut tracker = CoordPropTracker::default();
        let mut params = Params::default();
        let mut offsets = Offsets::default();

        assert!(build_fold_for_transfer::<Target, _, _, _, _, _, _, _, _>(
            &ctx,
            &transfer,
            &propagation(NodeId(2), NodeId(1)),
            &dims,
            reference,
            &mut coord,
            &mut tracker,
            &mut params,
            &mut offsets,
        ));
        assert_eq!(folds_of(&coord, PrimaryDim::Out), propagated());
        // A non-allocation reference reaches no allocation, so the offsets seam is not one of this
        // arm's effects.
        assert_eq!(offsets, Offsets::default());
    }

    /// ⭐ THE GATE'S COORDINATE IS THIS UNIT'S CHOICE AND NOT THE CALLER'S: `effectiveRefCoord`
    /// (`:13966-13970`) keys off `sizeRefComp`, which entry 358 derives off its own transfer.
    #[test]
    fn the_gate_takes_the_slice_view_only_where_this_units_own_size_comp_is_a_row() {
        let dsc = OneLds(LayoutDims::new(PrimaryDim::Out, Vec::new()));
        let core_ds = RowSplitOn(vec![PrimaryDim::Out]);
        let tree = Tree(vec![None, Some(0), Some(0)]);
        let meta = Metadata::default();
        let ctx = FoldContext {
            dsc: &dsc,
            core_ds: &core_ds,
            tree: &tree,
            distribution: &Seams,
            ancestors: &NoLoops,
            names: &Named,
            meta: &meta,
        };
        let loop_node = loop_named("loop0");
        let non_alloc_loops = [(&loop_node, LoopDistribution::AboveChunk)];
        let transfer_dims = [PrimaryDim::Out];
        let transfer_size = BTreeMap::new();
        let padding = PaddingForm::default();
        let ref_coord = ref_coordinate(PrimaryDim::Out);
        let ref_loops = [&loop_node];
        let slices = WorkSlices::default();
        let dims = [PrimaryDim::Out];
        // `allocateCoordinates_`, stating no fold for the row-split dim, so the same-row arm can only
        // collect a group off the slice view.
        let unfolded = dsc2::Coordinate::default();
        let slice_view = || {
            Some(RowCoordinate {
                dims: ref_coord.iter().collect(),
                work_slices: &slices,
            })
        };

        let built = |src, slice_view| {
            let transfer = TransferFold {
                node: NodeId(1),
                src: end(src, Some(DataConnect::AconstConnect)),
                dsts: TransferFoldDsts {
                    first: end(SenComponent::L0, None),
                    rest: Vec::new(),
                },
                transfer_dims: &transfer_dims,
                transfer_size: &transfer_size,
                alloc_loops: &[],
                non_alloc_loops: &non_alloc_loops,
                pe_sfp_transfer_size: None,
            };
            let reference = FoldReference {
                gate: RowBundling {
                    scan: RowBundlingScan {
                        units: PtRowPropagation {
                            src: SenComponent::Ptrow3,
                            dest: SenComponent::Ptrow3,
                        },
                        ref_node: NodeId(2),
                        candidates: RowCandidates {
                            forward: &[],
                            reverse: &[],
                        },
                        sdsc_slices: &slices,
                    },
                    ref_coord: RowCoordinate {
                        dims: unfolded.iter().collect(),
                        work_slices: &slices,
                    },
                    ref_slice_view: slice_view,
                    retry_dims: &dims,
                },
                kind: FoldReferenceKind::NonAlloc {
                    node: NodeId(2),
                    dims: vec![(PrimaryDim::Out, coord_dim_of(&ref_coord, PrimaryDim::Out))],
                    dim_scales: BTreeMap::new(),
                    loops: &ref_loops,
                    prop_comp: NonAllocPropComp::TransferEnds {
                        src: SenComponent::Lx,
                        dst: SenComponent::L0,
                    },
                },
                ref_padding: &padding,
            };
            let mut coord = dsc2::Coordinate::default();
            let mut tracker = CoordPropTracker::default();
            let mut params = Params::default();
            let mut offsets = Offsets::default();
            build_fold_for_transfer::<Target, _, _, _, _, _, _, _, _>(
                &ctx,
                &transfer,
                &propagation(NodeId(2), NodeId(1)),
                &dims,
                reference,
                &mut coord,
                &mut tracker,
                &mut params,
                &mut offsets,
            )
        };

        // A PT-row source makes `sizeRefComp` a row, so the slice view is what the gate asks.
        assert!(built(SenComponent::Ptrow3, slice_view()));
        // Without one the gate falls back to the coordinate the caller states and collects nothing.
        assert!(!built(SenComponent::Ptrow3, None));
        // ⛔ AND THE GUARD IS `sizeRefComp`, NOT THE SCAN'S UNITS: an LX source leaves it a non-row,
        // so the slice view goes unread even though it is right there.
        assert!(!built(SenComponent::Lx, slice_view()));
    }
}
