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

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::{Arch, Elements};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, SliceElems, Stage, StickDims, StickPart, cumulative_stick_sizes,
};
use crate::generated::DataConnect;
use crate::schedule::dsc2;
use crate::schedule::dsc2::{
    ComputeNode, CoordinateCategory, Dsc, FoldCardinality, FoldCoeff, LdsIdx, Node, Operand,
    OperandPos, TransferNode, TransferSide,
};
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
/// component with its data, and `ComputeType::cpp_spelling` is total where
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
//     file's one-operation seam. The real gap is that the concrete type does not IMPLEMENT the seam,
//     so entry 093 cannot be pointed at a real coordinate yet.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stride(pub i64);

impl Stride {
    /// The unpadded stride the reference initialises `stride` to (`ddc/ddc_fold.cpp:2115`).
    pub const ONE: Stride = Stride(1);
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
    /// `foldLabel`.
    pub label: FoldLabel,
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
        .map_or(Alpha(0), |share| Alpha(share.0.wrapping_mul(stride.0)));
    coord.add_fold(
        dim,
        CoordinateCategory::Spatial,
        Fold {
            cardinality: stage.corelets_used(),
            label: FoldLabel::CoreletFoldDim,
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
            label: FoldLabel::CoreWorksliceFoldDim,
            alpha: Alpha(stage.core_extent(dim).0.wrapping_mul(stride.0)),
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
                pad_stride: Some(Stride(4)),
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
                        label: FoldLabel::CoreletFoldDim,
                        alpha: Alpha(24),
                        beta: Beta(0),
                    },
                ),
                (
                    PrimaryDim::Out,
                    CoordinateCategory::Spatial,
                    Fold {
                        cardinality: Cardinality(3),
                        label: FoldLabel::CoreWorksliceFoldDim,
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
                pad_stride: Some(Stride(4)),
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

// crustify:todo: e233_dbgPrint
//   authority : ddc/ddc_fold.cpp:63  (11 body lines, level 1)
//   original  : void dbgPrint(const dsc2::ScheduleNode *node)
//   extract   : crustify-ddc/cpp/ddc.cpp:3233-3244
//   calls     : e078_dbgPrint, e079_dbgPrint

// crustify:todo: e234_scaleUpCoord
//   authority : ddc/ddc_fold.cpp:476  (118 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::scaleUpCoord(dsc2::ScheduleNode *lhsNode, dsc2::CoordinateType<CoordinateBaseType> &lhsCoord, dsc2::CoordinateType<CoordinateBaseType> &rhsCoord, const int ldsIdx, const SenComponents comp)
//   extract   : crustify-ddc/cpp/ddc.cpp:3254-3375
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e095_gatherFoldParams

// crustify:todo: e235_scaleDownCoord
//   authority : ddc/ddc_fold.cpp:598  (87 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::scaleDownCoord(dsc2::CoordinateType<CoordinateBaseType> &lhsCoord, dsc2::CoordinateType<CoordinateBaseType> &rhsCoord, const int ldsIdx)
//   extract   : crustify-ddc/cpp/ddc.cpp:3385-3474
//   calls     : e095_gatherFoldParams

// crustify:todo: e236_needNonRowBundling
//   authority : ddc/ddc_fold.cpp:688  (50 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::needNonRowBundling(std::string &dataConnect, bool checkProducers) const
//   extract   : crustify-ddc/cpp/ddc.cpp:3484-3535
//   calls     : e082_getCompRowId, e083_getComponent, e087_findCommonAncestor

// crustify:todo: e237_sameCoordinateRange
//   authority : ddc/ddc_fold.cpp:1260  (252 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::sameCoordinateRange( const dsc2::ScheduleNode *lhsNode, const dsc2::CoordinateType<CoordinateBaseType> &lhs, const dsc2::ScheduleNode *rhsNode, const dsc2::CoordinateType<CoordinateBaseType> &rhs, SenComponents comp /* = SenComponents::ALL*/, int ldsIdx /*= -1*/, bool commonDimsOnly /* = f
//   extract   : crustify-ddc/cpp/ddc.cpp:3545-3803
//   calls     : e090_combineContigousLevels, e095_gatherFoldParams, e104_clear

// crustify:todo: e238_getRelatedComputeCoord
//   authority : ddc/ddc_fold.cpp:1976  (129 body lines, level 1)
//   class     : Ddc
//   original  : dsc2::CoordinateType<CoordinateBaseType> &Ddc::getRelatedComputeCoord( dsc2::ComputeNode *computeNode, dsc2::CoordPropInfoType &refFoldInfo, std::vector<int> &constructedInputCoords, std::vector<int> &constructedOutputCoords, int &selectedLdsIdx)
//   extract   : crustify-ddc/cpp/ddc.cpp:3813-3945
//   calls     : e104_clear

// crustify:todo: e239_computeParamsForRowSplitFold
//   authority : ddc/ddc_fold.cpp:2161  (60 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::computeParamsForRowSplitFold( const PrimaryDimTypes &currDim, dsc2::FoldParamInfoType &resultFoldParams, RowGroupInfo &rowGroup)
//   extract   : crustify-ddc/cpp/ddc.cpp:3955-4017
//   calls     : e094_getDefaultRowSplitFold

// crustify:todo: e240_buildFoldForExternalAllocation
//   authority : ddc/ddc_fold.cpp:2238  (156 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::buildFoldForExternalAllocation(dsc2::AllocateNode *allocNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:4027-4183
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e081_buildFoldForBroadcastDim, e093_buildSpatialFold

// crustify:todo: e241_relateLoopsToAllocElemArr
//   authority : ddc/ddc_fold.cpp:2872  (166 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::relateLoopsToAllocElemArr( dsc2::CoordPropInfoType &coordPropInfo, const PrimaryDimTypes dim, const dsc2::CoordinateType<CoordinateBaseType> &coordinate, const dsc2::VectorOfLoopAndDim &refLoopChain, dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution)
//   extract   : crustify-ddc/cpp/ddc.cpp:4193-4363
//   calls     : e091_matchDataStream, e095_gatherFoldParams

// crustify:todo: e297_gatherRelatedPTRowsBase
//   authority : ddc/ddc_fold.cpp:740  (217 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::gatherRelatedPTRowsBase(const dsc2::CoordPropInfoType &coordPropInfo, RowGroupInfo &rowGroup)
//   extract   : crustify-ddc/cpp/ddc.cpp:8139-8357
//   calls     : e082_getCompRowId, e084_getLayoutDimsFromNode, e091_matchDataStream, e104_clear, e237_sameCoordinateRange

// crustify:todo: e298_buildFoldFromAllocation
//   authority : ddc/ddc_fold.cpp:3043  (245 body lines, level 2)
//   class     : Ddc
//   original  : void Ddc::buildFoldFromAllocation( dsc2::CoordPropInfoType &coordPropInfo, dsc2::ScheduleNode *node, dsc2::CoordinateType<CoordinateBaseType> &coordinate, SenComponents sizeRefComp, SenComponents propRefComp, dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution, RowGroupInfo &refRowGr
//   extract   : crustify-ddc/cpp/ddc.cpp:8367-8617
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e078_dbgPrint, e079_dbgPrint, e080_allDimsCovered, e095_gatherFoldParams, e233_dbgPrint, e239_computeParamsForRowSplitFold

// crustify:todo: e299_buildFoldFromNonAllocRef
//   authority : ddc/ddc_fold.cpp:3294  (355 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::buildFoldFromNonAllocRef( dsc2::CoordPropInfoType &coordPropInfo, const int refLdsIdx, const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate, dsc2::CoordinateType<CoordinateBaseType> &coordinate, const SenComponents sizeRefComp, SenComponents propRefComp, dsc2::LoopDistributionPara
//   extract   : crustify-ddc/cpp/ddc.cpp:8627-8988
//   calls     : e080_allDimsCovered, e094_getDefaultRowSplitFold, e095_gatherFoldParams, e104_clear, e239_computeParamsForRowSplitFold

// crustify:todo: e337_gatherRelatedPTRows
//   authority : ddc/ddc_fold.cpp:959  (298 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::gatherRelatedPTRows( RowGroupInfo &refRowGroup, const dsc2::CoordPropInfoType &coordPropInfo, const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate)
//   extract   : crustify-ddc/cpp/ddc.cpp:11783-12083
//   calls     : e076_print, e082_getCompRowId, e091_matchDataStream, e102_print, e297_gatherRelatedPTRowsBase

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
    use crate::generated::{ComputeType, DataConnect};
    use crate::schedule::dsc2::{
        AllocLayout, AllocateNode, DataInfo, Dsts, FoldPosition, LayoutDims, MaxDimSize, NodeName,
        ReplicationFactor, StartAddress,
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
            },
        }
    }

    #[test]
    fn a_compute_line_names_every_operand_with_its_own_connect() {
        let node = ComputeNode {
            name: NodeName("mac0".to_owned()),
            op: ComputeType::Macc,
            ex_unit: SenComponent::Ptrow3,
            inputs: vec![
                operand(SenComponent::Ptrow3, Some(DataConnect::AconstConnect), None),
                operand(SenComponent::Lx, None, None),
            ],
            outputs: vec![operand(SenComponent::Ptsouth, None, None)],
            num_folds_engaged: NumFolds::ONE,
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
            name: NodeName("t0".to_owned()),
            src: operand(SenComponent::Hbm, None, None),
            dsts: Dsts::new(
                operand(SenComponent::Lx, None, None),
                vec![operand(SenComponent::L0, None, None)],
            ),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
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
            gap_stick_spread: Default::default(),
            alloc_users: Vec::new(),
        };
        let compute = ComputeNode {
            name: NodeName("c0".to_owned()),
            op: ComputeType::Macc,
            ex_unit: SenComponent::Ptrow2,
            inputs: vec![],
            outputs: vec![],
            num_folds_engaged: NumFolds::ONE,
        };
        let transfer = TransferNode {
            name: NodeName("t0".to_owned()),
            src: operand(SenComponent::Hbm, None, None),
            dsts: Dsts::new(operand(SenComponent::L0, None, None), vec![]),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
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
            op: ComputeType::Macc,
            ex_unit: SenComponent::Ptrow0,
            inputs: vec![
                operand(SenComponent::Lx, None, Some(0)),
                operand(SenComponent::Lx, None, None),
            ],
            outputs: vec![],
            num_folds_engaged: NumFolds::ONE,
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
