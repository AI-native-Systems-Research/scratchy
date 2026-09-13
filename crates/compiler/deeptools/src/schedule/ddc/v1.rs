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

//! `ddc/ddcv1.cpp` — 24 of the campaign's 382 units (dependency level(s) [0, 1, 2, 7, 8]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e124_getLdsOrConstNameOfAllocNode` | 124 | 0 | 10 | `Ddc` | `ddc/ddcv1.cpp:20` |
//! | `e125_minimizeAllocations` | 125 | 0 | 101 | `Ddc` | `ddc/ddcv1.cpp:30` |
//! | `e126_populateUnitTimeTransfers` | 126 | 0 | 115 | `Ddc` | `ddc/ddcv1.cpp:439` |
//! | `e127_spreadDataInAllocate` | 127 | 0 | 27 | `Ddc` | `ddc/ddcv1.cpp:1682` |
//! | `e128_finalizeAllocateLayouts` | 128 | 0 | 30 | `Ddc` | `ddc/ddcv1.cpp:1712` |
//! | `e129_getClSplitDim` | 129 | 0 | 19 | — | `ddc/ddcv1.cpp:1799` |
//! | `e130_getPeSfpSplitDim` | 130 | 0 | 47 | — | `ddc/ddcv1.cpp:1819` |
//! | `e131_setPeFoldsIfPtInteraction` | 131 | 0 | 22 | — | `ddc/ddcv1.cpp:1868` |
//! | `e132_restoreDsc` | 132 | 0 | 5 | `Ddc` | `ddc/ddcv1.cpp:2272` |
//! | `e133_adjustLoopOffsetsAndAddresses` | 133 | 0 | 81 | `Ddc` | `ddc/ddcv1.cpp:3201` |
//! | `e134_simplifyScheduleTree` | 134 | 0 | 27 | `Ddc` | `ddc/ddcv1.cpp:3457` |
//! | `e135_updateLdsIdxMetadata` | 135 | 0 | 5 | `Ddc` | `ddc/ddcv1.cpp:3667` |
//! | `e136_initGlobalData` | 136 | 0 | 9 | `Ddc` | `ddc/ddcv1.cpp:3673` |
//! | `e258_allocAllMem` | 258 | 1 | 306 | `Ddc` | `ddc/ddcv1.cpp:132` |
//! | `e259_calculateClStartAddress` | 259 | 1 | 123 | `Ddc` | `ddc/ddcv1.cpp:1895` |
//! | `e260_fillLoopOffsetsAndAddresses` | 260 | 1 | 844 | `Ddc` | `ddc/ddcv1.cpp:2355` |
//! | `e261_finalizeOps` | 261 | 1 | 127 | `Ddc` | `ddc/ddcv1.cpp:3329` |
//! | `e262_coordinateMasking` | 262 | 1 | 181 | `Ddc` | `ddc/ddcv1.cpp:3485` |
//! | `e263_identifyBelowChunkBoundaryLoops` | 263 | 1 | 8 | `Ddc` | `ddc/ddcv1.cpp:3683` |
//! | `e307_exploreAssignDataStages` | 307 | 2 | 1126 | `Ddc` | `ddc/ddcv1.cpp:555` |
//! | `e308_prepDsc` | 308 | 2 | 252 | `Ddc` | `ddc/ddcv1.cpp:2019` |
//! | `e309_attachToPrefilledSchedule` | 309 | 2 | 71 | `Ddc` | `ddc/ddcv1.cpp:2280` |
//! | `e379_run_v1` | 379 | 7 | 109 | `Ddc` | `ddc/ddcv1.cpp:3692` |
//! | `e381_run` | 381 | 8 | 15 | `Ddc` | `ddc/ddcv1.cpp:3802` |

use core::num::{NonZeroI64, NonZeroU64};
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Arch, Bytes, Elements, FoldedUnit, IsaGen, Sticks};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, Sample, SliceElems, Stage, StickDims, StickPart, VectorComp,
    cumulative_stick_sizes, stick_sizes,
};
use crate::formats::DataFormat;
use crate::generated::{ComputeType, RegName};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{
    AllocId, BlockId, ConstIdx, NodeId, NodeKind, PadType, comp_row_id,
};
use crate::schedule::ddc::metadata::{DatastageId, DdcMemory, DestIdx, MetaDimKind, Metadata};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::dsc2::{
    AddressFold, AllocateNode, BlockNode, CondOp, ComputeNode, ConditionNode, Coordinate, Dsc,
    FoldCoeff, FoldDim, FoldPosition, LdsIdx, LoopBound, LoopCond, LoopCondComposite, MaxDimSize,
    NodeName, NumBuffers, Operand, Padding, Precision, RegSlot, ReplicationFactor, SchedNode, Size,
    SizeAndIndex, StartAddress, StickDimIdx, StickMaskNode, SyncNode, TransferNode, Unroll,
    generic_comp,
};
use crate::units::{Core, Corelet, Row};

/// AN ALLOCATION ARENA — the `dsc2::AllocateNode*`s the metadata's maps name.
///
/// ⭐ WALKING THE SCHEDULE TREE TO REACH THEM IS THE MECHANISM; the identity is the [`AllocId`], and
/// that is what `newAllocations_` and `shadowAllocations_` already hold.
pub type AllocArena = BTreeMap<AllocId, AllocateNode>;

/// A COMPUTE ARENA — `currDsc->computeOps_` as the nodes `opaqueOps_` keys into.
pub type ComputeArena = BTreeMap<NodeId, ComputeNode>;

/// WHAT A LABELLED DS OR A CONSTANT IS CALLED — `LabeledDs::dsName_` / `ConstantInfo::name_`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StorageName(pub String);

/// THE TWO NAME TABLES — `currDsc->labeledDs_` and `currDsc->constantInfo_`. Both are indexed by an
/// id the DSC itself issued, so neither `.at()` is a caller's obligation.
pub trait StorageNames {
    /// `labeledDs_.at(lds).dsName_`.
    fn lds_name(&self, lds: LdsIdx) -> StorageName;
    /// `constantInfo_.at(constant).name_`.
    fn constant_name(&self, constant: ConstIdx) -> StorageName;
}

/// WHAT ONE LABELLED DS'S STICK IS — `labeledDs_.at(i).dsType_` as `getStickSizes` reads it.
pub trait LdsSticks {
    /// The stick dims and their sizes.
    fn stick_dims(&self, lds: LdsIdx) -> StickDims;
}

/// WHETHER ONE ALLOC USER IS A MASKED COMPUTE — `nodeType_ == COMPUTE` AND
/// `instrAttribute_.computeMaskLoopOffsets_` non-empty (`ddc/ddcv1.cpp:1696-1698`), which is ONE
/// question about one user rather than two.
pub trait ComputeMasks {
    /// True only for a COMPUTE node carrying mask loop offsets.
    fn computes_under_mask(&self, node: NodeId) -> bool;
}

/// THE CORE-VERSUS-CORELET EXTENTS entry 129 COMPARES — `numCoreletsUsed_`, `CoreD_`, `CoreletD_`.
pub trait CoreletShapes {
    /// `numCoreletsUsed_`.
    fn corelets_used(&self) -> u32;
    /// `CoreD_.primaryDimToVal_st(dim)`.
    fn core_extent(&self, dim: PrimaryDim) -> Extent;
    /// `CoreletD_.primaryDimToVal_st(dim)`.
    fn corelet_extent(&self, dim: PrimaryDim) -> Extent;
}

/// ONE ENTRY OF `computeOp_` — `opFuncName` with `attributes_.dataFormat_`, the two fields entries
/// 127 and 130 read off it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeOp {
    /// `opFuncName`.
    pub op_func: OpFunc,
    /// `attributes_.dataFormat_`.
    pub format: DataFormat,
}

/// `dtGetEnv<bool>("ENABLE_LN32")` (`ddc/ddcv1.cpp:1844`) AS AN ARGUMENT.
///
/// ⛔ NOT AN AMBIENT READ. A placement decision taken from the process environment can be neither
/// reproduced nor tested, so the caller states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln32 {
    /// `ENABLE_LN32` unset or false — the `value_or(false)` default.
    Off,
    /// `ENABLE_LN32` true, which withdraws `LAYERNORM_SCALE` from the PE/SFP split set.
    On,
}

/// `primaryDimToVal_st`'s `comp` ARGUMENT for an allocation's component (`dsc/dims.cpp:659-663`) —
/// only the PE and the SFP, with their register files, name a vector component; every other
/// component falls through as `NO_COMPONENT` (`:683`).
fn sampled_as(component: SenComponent) -> Option<VectorComp> {
    match component {
        SenComponent::Pelrf | SenComponent::Pe => Some(VectorComp::Pe),
        SenComponent::Sfplrf | SenComponent::Sfp => Some(VectorComp::Sfp),
        _ => None,
    }
}

/// `stickSizePerDim.count(dim) ? .at(dim) : 1` — and the reference's DIVISION BY ZERO where a stick
/// dim measures nought, which has no [`NonZeroU64`].
pub(crate) fn stick_divisor(
    sizes: &[(PrimaryDim, Elements)],
    dim: PrimaryDim,
) -> Option<NonZeroU64> {
    match sizes.iter().find(|(walked, _)| *walked == dim) {
        Some(&(_, size)) => NonZeroU64::new(size.0),
        None => NonZeroU64::new(1),
    }
}

/// Replaces: e124_getLdsOrConstNameOfAllocNode
///
/// WHAT AN ALLOCATION'S DATA IS CALLED — its temp-storage compute's name, else its labelled DS's,
/// else its constant's.
///
/// ⭐ [`None`] IS THE REFERENCE'S `""`, which is not a name: its three live callers pass the result
/// STRAIGHT TO THE MEMORY TRACKER as a DS key (`ddc/ddcv1.cpp:280`, `:336`, `:341`) and none of them
/// tests it, so `None` is where the reference would key the tracker on the empty string.
#[must_use]
pub fn get_lds_or_const_name_of_alloc_node(
    anode: &AllocateNode,
    names: &(impl StorageNames + ?Sized),
) -> Option<StorageName> {
    if let Some(compute) = &anode.temp_storage_for_compute {
        return Some(StorageName(compute.0.clone()));
    }
    if let Some(lds) = anode.lds {
        return Some(names.lds_name(lds));
    }
    anode
        .const_idx
        .map(|constant| names.constant_name(constant))
}

/// A POSITION IN THE SCHEDULE TREE'S DFS ORDER — `node_to_index`'s `int` (`ddc/ddcv1.cpp:39-46`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DfsIndex(pub u32);

/// AN ALLOCATION'S LIVE RANGE — entry 125's local `Interval`, half-open `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    /// `start` — the earliest user's DFS index.
    pub start: DfsIndex,
    /// `end` — the latest user's.
    pub end: DfsIndex,
}

impl LiveRange {
    /// `{INT_MAX, 0}` — what a USER-LESS allocation gets (`ddc/ddcv1.cpp:50-51`).
    ///
    /// ⭐ INVERTED ON PURPOSE, and it is load-bearing: it is false against every interval in both
    /// directions, so an allocation nothing reads always lands in a shadow group of its own.
    pub const NONE: Self = Self {
        start: DfsIndex(u32::MAX),
        end: DfsIndex(0),
    };

    /// The range spanning every user (`ddc/ddcv1.cpp:52-58`).
    #[must_use]
    pub fn of(users: &[DfsIndex]) -> Self {
        let mut range = Self::NONE;
        for &user in users {
            range.start = range.start.min(user);
            range.end = range.end.max(user);
        }
        range
    }

    /// `overlapInterval` (`ddc/ddcv1.cpp:80-85`) — transcribed INCLUDING its asymmetry, so `self` is
    /// the allocation being placed and `other` a group member already there.
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        (self.end > other.start && self.end <= other.end)
            || (self.start >= other.start && self.start < other.end)
    }
}

/// ONE ALLOCATE NODE AS entry 125 GROUPS IT — its identity, the two fields it filters on, and its
/// [`LiveRange`].
///
/// ⭐ CARRYING THE RANGE IS WHAT MAKES `live_range.at(node)` TOTAL: the reference builds a side map
/// keyed by node and then indexes it three times, and every one of those is a throw here impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocLive {
    /// Which allocation.
    pub alloc: AllocId,
    /// `component_`.
    pub component: SenComponent,
    /// `ldsIdx_`, whose `-1` the auto-shuffling arm SKIPS ENTIRELY — not even a group of its own.
    pub lds: Option<LdsIdx>,
    /// Its live range over the DFS order.
    pub range: LiveRange,
}

/// Replaces: e125_minimizeAllocations
///
/// GROUPS ALLOCATIONS THAT MAY SHARE ONE ADDRESS into `shadowAllocations_`: with auto shuffling a
/// register-file allocation joins the first group of its own component that no member's live range
/// overlaps; everything else gets a group of its own.
///
/// ⭐ ASSIGNED, NOT APPENDED, AND THAT IS EXACT: entry 104 clears the field and this is its ONLY
/// writer in the authority tree (`ddc/ddcv1.cpp:118`), so it is empty at every entry.
pub fn minimize_allocations(
    allocs: &[AllocLive],
    has_auto_shuffling: bool,
    metadata: &mut Metadata,
) {
    let mut ordered: Vec<&AllocLive> = allocs.iter().collect();
    // ⛔ STABLE WHERE THE REFERENCE'S `std::sort` IS NOT (`ddc/ddcv1.cpp:63-67`): it orders on `start`
    // alone, so equal starts may come out in any order there and the group each joins can differ.
    // `allocs` arrives in the reference's ALLOCATE-DFS order, so this is one permitted outcome, fixed.
    ordered.sort_by_key(|live| live.range.start);
    let mut groups: Vec<Vec<&AllocLive>> = Vec::new();
    for live in ordered {
        let mut node_added = false;
        if has_auto_shuffling
            && matches!(
                live.component,
                SenComponent::Sfplrf | SenComponent::Pelrf | SenComponent::Ptxrf
            )
        {
            if live.lds.is_none() {
                continue;
            }
            for group in &mut groups {
                if node_added {
                    break;
                }
                let Some(component) = group.first().map(|member| member.component) else {
                    continue;
                };
                if live.component != component {
                    continue;
                }
                if group.iter().any(|member| live.range.overlaps(member.range)) {
                    continue;
                }
                if !group.iter().any(|member| member.alloc == live.alloc) {
                    group.push(live);
                }
                node_added = true;
            }
        }
        if !node_added {
            groups.push(vec![live]);
        }
    }
    metadata.shadow_allocations = groups
        .iter()
        .map(|group| group.iter().map(|member| member.alloc).collect())
        .collect();
}

/// WHICH LAYOUT DIMS ARE SPLAT — the `-2` positions of `LabeledDs::scale_`, as dims.
///
/// ⭐ `is_any_of(-2, scale_)` IS `!is_empty()` HERE, because `scale_` is indexed by a dim's position
/// in layout order (`getDimIndexInLayoutOrder`), so every entry of it names a layout dim and the two
/// spellings of the same test cannot disagree.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SplatDims(pub BTreeSet<PrimaryDim>);

/// A CONSTANT AS A CONSTANT-TO-CONSTANT TRANSFER READS IT — `data_.getSingleData().size()` and
/// `dataFormat_`.
///
/// ⛔ BOTH DIVISORS NON-ZERO BY TYPE: the element count is a [`NonZeroU64`], and [`DataFormat`] has
/// no `INVALID` arm, so `dataFormatsToBitWidth.at(INVALID) == -1` cannot reach the division.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantData {
    /// `data_.getSingleData().size()`.
    pub elements: NonZeroU64,
    /// `dataFormat_`.
    pub format: DataFormat,
}

/// THE LABELLED DS A TENSOR TRANSFER MEASURES — `labeledDs_.at(ldsIdx)` narrowed to the three fields
/// entry 126 reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferLds {
    /// `dsType_`, as stick dims.
    pub stick: StickDims,
    /// `scale_`'s `-2` positions — see [`SplatDims`].
    pub splat_dims: SplatDims,
    /// `dataFormat_`.
    pub format: DataFormat,
}

/// WHAT `getTransferType()` SAID, AND THE LABELLED DS THE MATCHING SIDE NAMES.
///
/// ⛔ FIVE ARMS AND NO SIXTH, so `DT_CHECK_MSG("Unexpected transfer type.")` is unspellable —
/// `INVALID_TRANSFER_TYPE` (`dsc/dsc2.h:854`) has no variant here. The [`None`] lds is the
/// reference's `ldsIdx < 0` skip; which SIDE supplies it is the arm's own answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferOperands {
    /// `CONSTANT_TO_CONSTANT` — `constantInfo_.at(srcLdsAndLoopOffsets_.constantId_)`.
    ConstantToConstant(ConstantData),
    /// `CONSTANT_TO_TENSOR` — the DESTINATION's `myLdsIdx_`.
    ConstantToTensor(Option<TransferLds>),
    /// `TENSOR_TO_TENSOR` — the SOURCE's.
    TensorToTensor(Option<TransferLds>),
    /// `NO_TRANSFER_TO_TENSOR` — the DESTINATION's.
    NoTransferToTensor(Option<TransferLds>),
    /// `NO_TRANSFER_FROM_TENSOR` — the SOURCE's.
    NoTransferFromTensor(Option<TransferLds>),
}

/// Replaces: e126_populateUnitTimeTransfers
///
/// FILLS ONE TRANSFER'S `replicationFactor_` AND `unitTimeTransferChunkSize_` — the elements it moves
/// per unit time, chunked along the stick's dims, capped by the metadata's forced element count, and
/// collapsed to a single-element splat where the source is a constant or a `-2`-scaled LX load.
///
/// ⛔ [`None`] IS A REFERENCE ABORT: an element count indivisible by the constant or by the stick
/// composition, a multi-dim splat that is not constant-to-tensor, a splat format narrower than 16b or
/// a replication factor not divisible by 4, and `senCompToGenericComp.at()` on a component that map
/// has no key for.
pub fn populate_unit_time_transfers<A: Arch>(
    transfer: &mut TransferNode,
    operands: &TransferOperands,
    num_elem_limit: Option<Elements>,
) -> Option<()> {
    let (lds, unit) = match operands {
        TransferOperands::ConstantToConstant(constant) => {
            let mut factor = 8 * A::BYTES_PER_STICK.get()
                / constant.elements.get()
                / u64::from(constant.format.bits().0);
            if let Some(limit) = num_elem_limit.filter(|limit| limit.0 > 0) {
                if limit.0 % constant.elements.get() != 0 {
                    return None;
                }
                factor = factor.min(limit.0 / constant.elements.get());
            }
            transfer.replication_factor = ReplicationFactor(factor);
            return Some(());
        }
        TransferOperands::ConstantToTensor(lds) | TransferOperands::NoTransferToTensor(lds) => {
            (lds, transfer.dsts.first().unit)
        }
        TransferOperands::TensorToTensor(lds) | TransferOperands::NoTransferFromTensor(lds) => {
            (lds, transfer.src.unit)
        }
    };
    if unit == SenComponent::NoComponent {
        return Some(());
    }
    let Some(lds) = lds.as_ref() else {
        return Some(());
    };
    let constant_to_tensor = matches!(operands, TransferOperands::ConstantToTensor(_));
    let src_is_constant = transfer.src.unit == SenComponent::Constant;
    let part = if unit.generic()? == SenComponent::L0lu {
        StickPart::WithinSlice(SliceElems::per_l0_row::<A>(&lds.stick)?)
    } else {
        StickPart::Whole
    };
    let sizes = stick_sizes(&lds.stick, part);
    let mut do_2b_splat =
        (!lds.splat_dims.0.is_empty() && unit == SenComponent::Lxlu && sizes.len() == 1)
            || src_is_constant;
    if !src_is_constant && unit == SenComponent::Lxlu && sizes.len() > 1 {
        do_2b_splat = sizes.iter().all(|(dim, _)| lds.splat_dims.0.contains(dim));
    }
    let mut elem_so_far = 1u64;
    let mut chunks: Vec<SizeAndIndex> = Vec::new();
    for (index, &(dim, extent)) in sizes.iter().enumerate() {
        // ⛔ THE LOOP CONDITION IS WHY `% numElemLimit` BELOW CANNOT DIVIDE BY ZERO: a limit of nought
        // fails `elemSoFar < numElemLimit` at the first test, so the body is unreachable for it.
        if num_elem_limit.is_some_and(|limit| elem_so_far >= limit.0) {
            break;
        }
        let mut size = extent.0;
        elem_so_far *= size;
        if let Some(limit) = num_elem_limit.filter(|limit| elem_so_far > limit.0) {
            if elem_so_far % limit.0 != 0 {
                return None;
            }
            size /= elem_so_far / limit.0;
        }
        chunks.push(SizeAndIndex {
            size_dim: Size {
                dim,
                size: Elements(if do_2b_splat { 1 } else { size }),
            },
            src_size_idx: StickDimIdx(index as u32),
            dst_size_idx: StickDimIdx(index as u32),
        });
    }
    if do_2b_splat {
        if sizes.len() > 1 && !constant_to_tensor {
            return None;
        }
        if let Some(limit) = num_elem_limit {
            transfer.replication_factor = ReplicationFactor(limit.0);
        } else {
            let mut factor = 1u64;
            for &(_, size) in &sizes {
                factor *= size.0;
            }
            if !src_is_constant && lds.format != DataFormat::Sen169Fp16 {
                if !matches!(
                    lds.format,
                    DataFormat::IeeeFp32 | DataFormat::IeeeInt32 | DataFormat::Senuint32
                ) {
                    return None;
                }
                // ⭐ `unitTimeTransferChunkSize_[0].sizeDim_.size_ == 1` HOLDS BY CONSTRUCTION: every
                // chunk pushed under `do2BSplat` above carries a size of exactly one. The empty case
                // the reference indexes into regardless is the [`None`].
                let first = chunks.first_mut()?;
                first.size_dim.size = Elements(first.size_dim.size.0 * 4);
                if factor % 4 != 0 {
                    return None;
                }
                factor /= 4;
            }
            transfer.replication_factor = ReplicationFactor(factor);
        }
    }
    // ⛔ ASSIGNED WHERE THE REFERENCE `push_back`s ONTO WHATEVER IS ALREADY THERE
    // (`ddc/ddcv1.cpp:524`): it never clears the field, and a JSON-imported DSC can arrive with
    // entries (`dsc/dsc2.cpp:1560`). Its own `[0].sizeDim_.size_ == 1` check (`:544`) only holds when
    // the field started empty, so assigning is what that check already assumes.
    transfer.unit_time_transfer_chunk_size = chunks;
    Some(())
}

/// Replaces: e127_spreadDataInAllocate
///
/// SPREADS EVERY PTXRF ALLOCATION OVER EIGHT STICKS OF GAP in its outermost layout dim — on SEN1P5
/// only, only when the DSC restickifies, and only for an allocation some MASKED compute reads.
pub fn spread_data_in_allocate<A: Arch>(
    compute_ops: &[ComputeOp],
    metadata: &Metadata,
    allocs: &mut AllocArena,
    masks: &impl ComputeMasks,
) {
    if A::GEN != IsaGen::Sen1p5 {
        return;
    }
    if !compute_ops
        .iter()
        .any(|op| matches!(op.op_func, OpFunc::ReStickifyOpLx | OpFunc::ReStickifyOpHbm))
    {
        return;
    }
    let Some(allocation) = metadata.new_allocations.get(&DdcMemory::PtxRf) else {
        return;
    };
    for &id in allocation.lds_idx_and_alloc_node.values() {
        let Some(alloc) = allocs.get_mut(&id) else {
            continue;
        };
        if !alloc
            .alloc_users
            .iter()
            .any(|&user| masks.computes_under_mask(user))
        {
            continue;
        }
        let dim = alloc.layout.outermost_dim();
        alloc.gap_stick_spread.insert(dim, Sticks(8));
    }
}

/// Replaces: e128_finalizeAllocateLayouts
///
/// RESOLVES EVERY ALLOCATION'S `maxDimSizes_` from a datastage index to a stick-normalised element
/// count, then gives every shadow group its FIRST member's start address.
///
/// ⛔ [`None`] IS A REFERENCE ABORT: a `dataStageParam_` or `getCumulativeStickSizes` `.at()` throw,
/// a zero stick size, or a negative extent — which has no [`Elements`].
pub fn finalize_allocate_layouts<S: Stage>(
    metadata: &Metadata,
    allocs: &mut AllocArena,
    stages: &BTreeMap<DatastageId, S>,
    sticks: &impl LdsSticks,
) -> Option<()> {
    for allocation in metadata.new_allocations.values() {
        for (&lds, &id) in &allocation.lds_idx_and_alloc_node {
            let Some(alloc) = allocs.get_mut(&id) else {
                continue;
            };
            // ⭐ THE LDS COMES FROM THE MAP KEY, which is what makes `labeledDs_.at(alloc->ldsIdx_)`
            // total: `newAllocations_` is keyed BY the index the allocation carries.
            let cumulative = cumulative_stick_sizes(&sticks.stick_dims(lds), StickPart::Whole)?;
            let at = Sample {
                corelet: Corelet::at::<0>(),
                row: Some(Row::at::<0>()),
                comp: sampled_as(alloc.component),
            };
            for entry in alloc.layout.iter_mut() {
                let MaxDimSize::Stage(stage) = entry.1 else {
                    continue;
                };
                let extent = u64::try_from(stages.get(&stage)?.extent(entry.0, at).0).ok()?;
                entry.1 = MaxDimSize::Resolved(Elements(
                    extent / stick_divisor(&cumulative, entry.0)?.get(),
                ));
            }
        }
    }
    for group in &metadata.shadow_allocations {
        let Some((&first, rest)) = group.split_first() else {
            continue;
        };
        let Some(address) = allocs.get(&first).map(|alloc| alloc.start_address.clone()) else {
            continue;
        };
        for &other in rest {
            if let Some(alloc) = allocs.get_mut(&other) {
                alloc.start_address = address.clone();
            }
        }
    }
    Some(())
}

/// Replaces: e129_getClSplitDim
///
/// WHICH DIMS A MULTI-CORELET DSC SPLITS ACROSS CORELETS — every dim whose per-core extent differs
/// from its per-corelet one, and NONE at all when the DSC uses one corelet.
///
/// ⛔ [`None`] IS THE `DT_ERROR`: more than one corelet and not a single dim to explain it. An empty
/// [`Some`] is the legitimate single-corelet answer, so the two are not the same value.
pub fn get_cl_split_dim(dsc: &impl CoreletShapes) -> Option<BTreeSet<PrimaryDim>> {
    let mut split = BTreeSet::new();
    if dsc.corelets_used() > 1 {
        for dim in PrimaryDim::ALL {
            // `is_any_of(dim, IJ, KIJ, PrimaryDimTypesCount)` — the terminator has no variant here.
            if matches!(dim, PrimaryDim::Ij | PrimaryDim::Kij) {
                continue;
            }
            if dsc.core_extent(dim) != dsc.corelet_extent(dim) {
                split.insert(dim);
            }
        }
        if split.is_empty() {
            return None;
        }
    }
    Some(split)
}

/// Replaces: e130_getPeSfpSplitDim
///
/// WHICH DIM THE PE AND THE SFP SPLIT — the chunk stage's own request where it made one, else the
/// innermost layout dim of the input whose chunk extent is an even multiple of its stick size, and
/// only for the nine op-funcs the split is implemented for.
///
/// ⛔ [`None`] IS `DT_CHECK_MSG("PE/SFP split requested, but hardware cannot perform it")`: an fp32
/// compute before SEN1P5. An empty [`Some`] is "no split", which is not the same answer.
pub fn get_pe_sfp_split_dim<A: Arch, S: Stage>(
    compute_op: &ComputeOp,
    chunk_stage: &S,
    input_lds: LdsIdx,
    dsc: &impl Dsc,
    sticks: &impl LdsSticks,
    ln32: Ln32,
) -> Option<BTreeSet<PrimaryDim>> {
    let split_not_possible = compute_op.format == DataFormat::IeeeFp32 && A::GEN <= IsaGen::Rcudd1a;
    // `dataStageParam_.at(1).ss_.peSfpSplit_` — datastage 1 is `Metadata::CHUNK_DSTGID`, and
    // `PrimaryDim::ALL` walks the dims in the `std::map` order its key set is iterated in.
    let requested: BTreeSet<PrimaryDim> = PrimaryDim::ALL
        .into_iter()
        .filter(|&dim| chunk_stage.is_pe_sfp_split(dim))
        .collect();
    if !requested.is_empty() {
        if split_not_possible {
            return None;
        }
        return Some(requested);
    }
    if split_not_possible {
        return Some(BTreeSet::new());
    }
    let splits = matches!(
        compute_op.op_func,
        OpFunc::Reciprocal
            | OpFunc::SqrtFwd
            | OpFunc::Rsqrt
            | OpFunc::GeluFwd
            | OpFunc::TanhFwd
            | OpFunc::Int32Idxtoaddr
            | OpFunc::SigmoidFwd
            | OpFunc::SiluFwd
    ) || (compute_op.op_func == OpFunc::LayernormScale && ln32 == Ln32::Off);
    if !splits {
        return Some(BTreeSet::new());
    }
    // ⭐ ONE `input_lds` FOR BOTH READS: the reference takes the stick sizes from
    // `labeledDs_.at(0).dsType_` and the layout order from that same `inpLds.ldsIdx_`
    // (`ddc/ddcv1.cpp:1852-1854`), and a labelled DS's position in `labeledDs_` IS its `ldsIdx_`.
    let per_dim = cumulative_stick_sizes(&sticks.stick_dims(input_lds), StickPart::Whole)?;
    let mut split = BTreeSet::new();
    for dim in dsc.layout_dims(input_lds).to_vec().into_iter().rev() {
        let at = Sample {
            corelet: Corelet::at::<0>(),
            row: None,
            comp: None,
        };
        // ⛔ A NEGATIVE EXTENT IS `DataStructDims`' UNSET `-1` (`dsc/dims.h:162-193`), NOT A REFUSAL:
        // the reference's signed `extent / stickSize` is never `> 1` then, so it tries the next dim.
        let Ok(extent) = u64::try_from(chunk_stage.extent(dim, at).0) else {
            continue;
        };
        let for_split = extent / stick_divisor(&per_dim, dim)?.get();
        if for_split > 1 && for_split % 2 == 0 {
            split.insert(dim);
            break;
        }
    }
    Some(split)
}

/// Replaces: e131_setPeFoldsIfPtInteraction
///
/// GIVES EVERY PE COMPUTE of a SEN1P5 DSC THE ARCH'S PE FOLD COUNT, and only when the DSC has a PT
/// compute at all — the PT's presence is what makes the PE fold, not the PE's own op.
///
/// ⛔ [`None`] IS `senCompToGenericComp.at()` ON A COMPONENT THAT MAP HAS NO KEY FOR, which is 20 of
/// the 107 (`sys-arch-spec/arch_enums.cpp:124-211`).
pub fn set_pe_folds_if_pt_interaction<A: Arch>(computes: &mut ComputeArena) -> Option<()> {
    if A::GEN < IsaGen::Sen1p5 {
        return Some(());
    }
    let mut do_change = false;
    for node in computes.values() {
        if node.ex_unit.generic()? == SenComponent::Pt {
            do_change = true;
            break;
        }
    }
    if !do_change {
        return Some(());
    }
    for node in computes.values_mut() {
        if node.ex_unit.generic()? == SenComponent::Pe {
            node.num_folds_engaged = A::GEN.folds_per_unit(FoldedUnit::Pe);
        }
    }
    Some(())
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE `run_v1` FIX-UP VOCABULARY — as entries 132-136 read it.
//
// ⭐ THE TRAITS ARE THE MECHANISM FOR REACHING OPERANDS, the one part the campaign statement names
// as droppable: `traverseTreeDFSMutable`, `getOwnerLoop`, `getRelevantCoreCl`, `moveNode`,
// `deleteChildNode` and `getStickDims` are all `dsc/dsc2.cpp` and `dsc/designSpaceConfig.h` —
// outside this campaign's file list. What these five units OWN is the decision and the mutation.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE `loopEleOffsets_` ENTRY — how many elements of that dim to step per trip of that loop, an
/// `int` in `DataInfo::loopEleOffsets_` (`dsc/dsc2.h:730-732`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopEleOffset(pub i32);

/// WHICH INPUT OF A COMPUTE — an index into `inputs_`/`inputsLdsAndLoopOffsets_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputIdx(pub usize);

/// `computeOp_` (`dsc/designSpaceConfig.h:89`) PROJECTED ONTO `opFuncName`, and NON-EMPTY.
///
/// ⛔ THE NON-EMPTINESS IS WHAT MAKES ENTRY 132 TOTAL: `computeOp_.at(0)` throws on an op-less DSC,
/// and a DSC with no compute op has nothing for DDC to schedule. `OpFuncs::NONE` is [`None`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpFuncs {
    first: Option<OpFunc>,
    rest: Vec<Option<OpFunc>>,
}

impl OpFuncs {
    /// A DSC has at least one compute op, and this is how that is stated.
    #[must_use]
    pub const fn new(first: Option<OpFunc>, rest: Vec<Option<OpFunc>>) -> Self {
        Self { first, rest }
    }

    /// `computeOp_.at(0).opFuncName` — total.
    #[must_use]
    pub const fn first(&self) -> Option<OpFunc> {
        self.first
    }

    /// Every op's `opFuncName`, in `computeOp_`'s order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<Option<OpFunc>> {
        core::iter::once(self.first)
            .chain(self.rest.iter().copied())
            .collect()
    }
}

/// `DesignSpaceConfig::computeOp_` — the flat op list entries 132 and 133 read `opFuncName` off.
pub trait ComputeOps {
    /// `computeOp_`, one `opFuncName` per entry.
    fn op_funcs(&self) -> OpFuncs;
    /// `computeOp_.at(0).opFuncName = op_func`.
    fn set_first_op_func(&mut self, op_func: OpFunc);
}

/// WHAT ENTRY 133 REACHES THROUGH — the two schedule walks, the loop nesting above a node, the
/// input's stick dims, and the loop element offsets it writes.
pub trait LoopOffsets {
    /// `0 .. numCoreletsUsed_DSC2_` as corelets, so the loop index cannot be an out-of-range `int`.
    fn corelets(&self) -> Vec<Corelet>;
    /// `traverseTreeDFSMutable(nullptr, {TRANSFER})`, in the traversal's order.
    fn transfers(&self) -> Vec<NodeId>;
    /// `src_`/`srcLdsAndLoopOffsets_` and `dstVias_`/`dstLdsAndLoopOffsets_`, each pair zipped so
    /// they cannot disagree in length.
    fn transfer(&self, node: NodeId) -> TransferNode;
    /// `traverseTreeDFSMutable(nullptr, {COMPUTE})`, in the traversal's order.
    fn computes(&self) -> Vec<NodeId>;
    /// `type_`, `exUnit_` and `inputs_` zipped with `inputsLdsAndLoopOffsets_`.
    fn compute(&self, node: NodeId) -> ComputeNode;
    /// `getStickDims(lds)` — `primaryDsInfo_.at(labeledDs_.at(lds).dsType_).stickDimOrder_`
    /// (`dsc/designSpaceConfig.h:241-246`).
    fn stick_dims(&self, lds: LdsIdx) -> Vec<PrimaryDim>;
    /// `node->getOwnerLoop()` (`dsc/dsc2.h:465`), absent for its `nullptr`.
    fn owner_loop(&self, node: NodeId) -> Option<LoopId>;
    /// `srcLdsAndLoopOffsets_.loopEleOffsets_.at(corelet)` — the loops it keys and the dims each of
    /// those holds; EMPTY where the reference's `.at()` finds no entry for that corelet.
    fn src_loop_ele_offsets(
        &self,
        node: NodeId,
        corelet: Corelet,
    ) -> Vec<(LoopId, Vec<PrimaryDim>)>;
    /// `srcLdsAndLoopOffsets_.loopEleOffsets_.at(cl).at(dim_loop).at(dim) = offset` — a write over
    /// three keys [`RestickifySite`] proved present.
    fn set_src_loop_ele_offset(
        &mut self,
        node: NodeId,
        corelet: Corelet,
        dim_loop: LoopId,
        dim: PrimaryDim,
        offset: LoopEleOffset,
    );
    /// `inputsLdsAndLoopOffsets_.at(input).loopEleOffsets_[cl][dim_loop][dim] = offset`.
    ///
    /// ⛔ `operator[]`, SO IT INSERTS all three keys, and ⛔ `dim_loop` MAY BE ABSENT: the reference
    /// keys this map by `getOwnerLoop()` without checking it, and a null `LoopNode*` is a legitimate
    /// key of an `unordered_map<const LoopNode*, ..>` (`ddc/ddcv1.cpp:3257-3261`).
    fn set_input_loop_ele_offset(
        &mut self,
        node: NodeId,
        input: InputIdx,
        corelet: Corelet,
        dim_loop: Option<LoopId>,
        dim: PrimaryDim,
        offset: LoopEleOffset,
    );
}

/// A CORE/CORELET SET — `getRelevantCoreCl`'s `std::map<int, std::set<int>>` (`dsc/dsc2.h:471`),
/// whose EQUALITY is the whole of entry 134's branch test.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoreClSet(pub BTreeMap<Core, BTreeSet<Corelet>>);

/// A RESTICKIFY TRANSFER WHOSE FOUR `DT_CHECK`S ALREADY HELD — the `L0LU`→`PT` transfer, the two
/// loops around it, and the one stick dim to step.
///
/// ⛔ THE REFERENCE DEREFERENCES `innerLoop` UNGUARDED (`ddc/ddcv1.cpp:3236-3237`) and then throws
/// four ways: the corelet's `loopEleOffsets_` must key EXACTLY those two loops, each must carry the
/// dim, and `getStickDims` must have returned exactly one. Constructing this proves all of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestickifySite {
    /// The `dsc2::TransferNode*` whose source offsets get stepped.
    pub transfer: NodeId,
    /// `transfer->getOwnerLoop()` — steps 2 elements.
    pub inner: LoopId,
    /// `innerLoop->getOwnerLoop()` — steps 1.
    pub outer: LoopId,
    /// `inputStickDim.at(0)`, the sole stick dim of the LXLU-sourced input.
    pub dim: PrimaryDim,
}

impl RestickifySite {
    /// The site, or [`None`] wherever the reference would dereference a null loop or throw.
    #[must_use]
    pub fn of<T: LoopOffsets + ?Sized>(dsc: &T, transfer: NodeId, dim: PrimaryDim) -> Option<Self> {
        let node = dsc.transfer(transfer);
        if is_skipped(node.src.unit) || generic_comp(node.src.unit) != Some(GenericComp::L0lu) {
            return None;
        }
        let to_pt = node
            .dsts
            .iter()
            .any(|dst| !is_skipped(dst.unit) && generic_comp(dst.unit) == Some(GenericComp::Pt));
        if !to_pt {
            return None;
        }
        let inner = dsc.owner_loop(transfer)?;
        let outer = dsc.owner_loop(inner.0)?;
        for corelet in dsc.corelets() {
            let offsets = dsc.src_loop_ele_offsets(transfer, corelet);
            if offsets.len() != 2 {
                return None;
            }
            let steps = |dim_loop: LoopId| {
                offsets
                    .iter()
                    .any(|(keyed, dims)| *keyed == dim_loop && dims.contains(&dim))
            };
            if !steps(inner) || !steps(outer) {
                return None;
            }
        }
        Some(Self {
            transfer,
            inner,
            outer,
            dim,
        })
    }
}

/// AN LXLU `fmul` WHOSE TWO INPUTS ENTRY 133 HAS SEEN, AND WHICH OF THEM CARRIES THE STEP.
///
/// ⛔ `DT_CHECK(compute->inputs_.size() == 2)` (`ddc/ddcv1.cpp:3252`) is what this makes a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LxluScaleSite {
    /// The `dsc2::ComputeNode*` to step.
    pub compute: NodeId,
    /// `idx` — input 1 when the pair is exactly (`LXLUSCALEREG`, `LATCH`), input 0 otherwise.
    pub input: InputIdx,
}

impl LxluScaleSite {
    /// The site, or [`None`] where the compute does not read exactly two inputs.
    #[must_use]
    pub fn of(compute: NodeId, node: &ComputeNode) -> Option<Self> {
        let [scale, latch] = node.inputs.as_slice() else {
            return None;
        };
        let latched = scale.unit == SenComponent::Lxluscalereg && latch.unit == SenComponent::Latch;
        Some(Self {
            compute,
            input: InputIdx(usize::from(latched)),
        })
    }
}

/// WHAT ENTRY 134 REACHES THROUGH — the condition walk, each node's core/corelet census, and the
/// two tree edits it makes.
pub trait ConditionSimplification {
    /// `scheduleTree_.getHead()` (`dsc/dsc2.h:637`), which is a `LoopNode`.
    fn head(&self) -> LoopId;
    /// `relevantComps_`'s keys (`dsc/dsc2.h:516`) — entry 134 reads only whether there are any.
    fn relevant_comps(&self, node: LoopId) -> Vec<SenComponent>;
    /// `traverseTreeDFSMutable(head, {CONDITION})`, in the traversal's order.
    fn conditions(&self, head: LoopId) -> Vec<NodeId>;
    /// `node->getRelevantCoreCl()` with its default `comp = ALL`.
    fn relevant_core_cl(&self, node: NodeId) -> CoreClSet;
    /// `cn->hasCoreClCond()` — TRUE when `loopCond_.twoLevelOrOfAnds_` is EMPTY (`dsc/dsc2.h:693`).
    fn has_core_cl_cond(&self, condition: NodeId) -> bool;
    /// `cn->next_` — the "then" and "else" regions, at most two `BLOCK`s (`dsc/dsc2.h:685-688`).
    fn branches(&self, condition: NodeId) -> Vec<NodeId>;
    /// `node->moveNode(currDsc, sibling->getMutableParent(), false, sibling)` — into the sibling's
    /// own parent, AFTER the sibling.
    fn move_after(&mut self, node: NodeId, sibling: NodeId);
    /// `node->prev_->deleteChildNode(currDsc, node)`.
    fn delete_node(&mut self, node: NodeId);
}

/// THE SCHEDULE HEAD WITH ITS RELEVANT COMPONENTS ALREADY FILLED — entry 134's
/// `DT_CHECK(!scheduleTree_.getHead()->relevantComps_.empty())` (`ddc/ddcv1.cpp:3458`) as a value.
///
/// ⛔ IT IS ALSO WHERE THE WALK STARTS: `traverseTreeDFSMutable(nullptr, ..)` means "from the head",
/// so the node whose census the reference asserts is the same node it then traverses — carrying the
/// head makes the assertion and the traversal one fact rather than two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleHead {
    /// `scheduleTree_.getHead()`.
    pub head: LoopId,
}

impl ScheduleHead {
    /// The head, or [`None`] where its `relevantComps_` is still empty.
    #[must_use]
    pub fn of<T: ConditionSimplification + ?Sized>(tree: &T) -> Option<Self> {
        let head = tree.head();
        (!tree.relevant_comps(head).is_empty()).then_some(Self { head })
    }
}

/// WHAT ENTRY 135 READS OFF THE DSC — `labeledDs_`'s own `ldsIdx_` field, entry by entry.
pub trait LabeledDsIndices {
    /// `labeledDs_.at(i).ldsIdx_` for every entry, in order.
    fn labeled_ds_indices(&self) -> Vec<LdsIdx>;
}

/// THE PER-DATASTAGE SPLIT STRATEGY ENTRY 136 CONSULTS.
pub trait DataStages {
    /// `dataStageParam_.at(stage).ss_.coreletSplit_`'s keys (`dsc/dims.h:206`) in `std::map`'s own
    /// order — EMPTY where the reference's `.at()` finds no such data stage.
    fn corelet_split_dims(&self, stage: DatastageId) -> BTreeSet<PrimaryDim>;
}

/// `Ddc`'s OWN PER-DSC STATE (`ddc/ddc.h:105-110`), narrowed to what entry 136 writes.
///
/// ⛔ NOT PART OF [`Metadata`]: `coreletSplitDim` is a member of `Ddc` itself, so it survives
/// `Metadata::clear` and is reset by `initGlobalData` instead.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GlobalData {
    /// `coreletSplitDim` (`ddc/ddc.h:109`) — `PrimaryDimTypesCount` is UNSET, not dim zero.
    pub corelet_split_dim: Option<PrimaryDim>,
    /// `loopsBelowChunkBoundary` (`ddc/ddc.h:110`) — entry 263 fills it and entry 260 reads it.
    ///
    /// ⛔ ITS PRESENCE IS WHY THIS TYPE IS NOT `Copy`: a set is not a scalar, and entry 260 holds a
    /// shared borrow of it while writing offsets.
    pub loops_below_chunk_boundary: BTreeSet<LoopId>,
}

/// `is_any_of(unit, skip_units)` with entry 133's `skip_units = {NO_COMPONENT, CONSTANT}`.
const fn is_skipped(unit: SenComponent) -> bool {
    matches!(unit, SenComponent::NoComponent | SenComponent::Constant)
}

/// Replaces: e132_restoreDsc
///
/// Puts the backed-up `opFuncName` back on the first compute op, undoing the `EXX2` →
/// `EXX2_ZEROMEAN` swap `prepDsc` made there (`ddc/ddcv1.cpp:2064-2078`).
///
/// ⛔ ENTRY 0 IS BOTH ENDS OF THE PAIR: the backup is a single slot taken from `computeOp_.at(0)`,
/// so the restore cannot land on the wrong op — and ⛔ it is NOT cleared, so a second call restores
/// the same name again over whatever the first left.
pub fn restore_dsc<T: ComputeOps + ?Sized>(metadata: &Metadata, dsc: &mut T) {
    if let Some(op_func) = metadata.op_func_backup {
        dsc.set_first_op_func(op_func);
    }
}

/// The FIRST LXLU-sourced transfer's stick dim, where it has exactly one — entry 133's
/// `inputStickDim` and the `DT_CHECK(inputStickDim.size() == 1)` that guards its `.at(0)`.
///
/// ⛔ THE REFERENCE `break`s AT THE FIRST LXLU TRANSFER, so a second one with a single stick dim
/// does not rescue a first one with two: this answers about that first transfer only.
fn restickify_stick_dim<T: LoopOffsets + ?Sized>(dsc: &T) -> Option<PrimaryDim> {
    for node in dsc.transfers() {
        let transfer = dsc.transfer(node);
        if is_skipped(transfer.src.unit) {
            continue;
        }
        if generic_comp(transfer.src.unit) != Some(GenericComp::Lxlu) {
            continue;
        }
        let lds = transfer.src.data.my_lds_idx?;
        return match dsc.stick_dims(lds).as_slice() {
            [dim] => Some(*dim),
            _ => None,
        };
    }
    None
}

/// `adjustLoopOffsetsForRestickify` (`ddc/ddcv1.cpp:3207-3249`) — 2 elements of the input's stick
/// dim per trip of the inner loop, 1 per trip of the outer, on every corelet in use.
fn adjust_loop_offsets_for_restickify<T: LoopOffsets + ?Sized>(dsc: &mut T) {
    let Some(dim) = restickify_stick_dim(dsc) else {
        return;
    };
    let sites: Vec<RestickifySite> = dsc
        .transfers()
        .into_iter()
        .filter_map(|node| RestickifySite::of(dsc, node, dim))
        .collect();
    for site in sites {
        for corelet in dsc.corelets() {
            dsc.set_src_loop_ele_offset(
                site.transfer,
                corelet,
                site.inner,
                site.dim,
                LoopEleOffset(2),
            );
            dsc.set_src_loop_ele_offset(
                site.transfer,
                corelet,
                site.outer,
                site.dim,
                LoopEleOffset(1),
            );
        }
    }
}

/// `adjustLoopOffsetsForLXLUCompute` (`ddc/ddcv1.cpp:3251-3263`) — one `in` element per trip of the
/// compute's own loop, on the input that is not the latched scale register.
fn adjust_loop_offsets_for_lxlu_compute<T: LoopOffsets + ?Sized>(dsc: &mut T, site: LxluScaleSite) {
    let parent_loop = dsc.owner_loop(site.compute);
    for corelet in dsc.corelets() {
        dsc.set_input_loop_ele_offset(
            site.compute,
            site.input,
            corelet,
            parent_loop,
            PrimaryDim::In,
            LoopEleOffset(1),
        );
    }
}

/// Replaces: e133_adjustLoopOffsetsAndAddresses
///
/// On SEN1P5 and later only: steps the two loops around every `L0LU`→`PT` transfer by 2 and 1
/// elements of the restickified input's stick dim once per restickify op in `computeOp_`, then gives
/// every LXLU `fmul` a one-element `in` step on its own loop.
///
/// ⛔ IT ADJUSTS NO ADDRESSES despite the name — the body writes loop element offsets only.
/// ⛔ THE RESTICKIFY PASS RE-RUNS ONCE PER MATCHING OP (`ddc/ddcv1.cpp:3266-3271`); its writes are
/// absolute, so the repeats land on the same values.
pub fn adjust_loop_offsets_and_addresses<A, T>(dsc: &mut T)
where
    A: Arch,
    T: LoopOffsets + ComputeOps + ?Sized,
{
    if A::GEN < IsaGen::Sen1p5 {
        return;
    }
    for op_func in dsc.op_funcs().to_vec() {
        if matches!(
            op_func,
            Some(OpFunc::ReStickifyOpLx | OpFunc::ReStickifyOpHbm)
        ) {
            adjust_loop_offsets_for_restickify(dsc);
        }
    }
    let latched: Vec<LxluScaleSite> = dsc
        .computes()
        .into_iter()
        .filter_map(|node| {
            let compute = dsc.compute(node);
            (compute.op == ComputeType::Fmul && compute.ex_unit == SenComponent::Lxlu)
                .then(|| LxluScaleSite::of(node, &compute))
                .flatten()
        })
        .collect();
    for site in latched {
        adjust_loop_offsets_for_lxlu_compute(dsc, site);
    }
}

/// Replaces: e134_simplifyScheduleTree
///
/// Drops every core/corelet condition that decides nothing: one relevant to no core/corelet at all
/// goes away, and one whose "then" or "else" region covers exactly the condition's own core/corelet
/// set is hoisted into the condition's place. Walked in REVERSE so a deletion cannot invalidate the
/// rest of the traversal, and external conditions are left to the DSC that owns them.
///
/// ⛔ `hasCoreClCond()` IS A QUESTION ABOUT `loopCond_`, NOT `coreClCond_` (`dsc/dsc2.h:693`): a
/// node with no loop condition answers TRUE even with an empty `coreClCond_`.
pub fn simplify_schedule_tree<T: ConditionSimplification + ?Sized>(
    tree: &mut T,
    metadata: &Metadata,
    head: ScheduleHead,
) {
    let mut conditions = tree.conditions(head.head);
    conditions.reverse();
    for condition in conditions {
        if metadata.external_nodes.contains(&condition) {
            continue;
        }
        let relevant = tree.relevant_core_cl(condition);
        if relevant.0.is_empty() {
            tree.delete_node(condition);
            continue;
        }
        if !tree.has_core_cl_cond(condition) {
            continue;
        }
        for child in tree.branches(condition) {
            if tree.relevant_core_cl(child) == relevant {
                tree.move_after(child, condition);
                tree.delete_node(condition);
                break;
            }
        }
    }
}

/// Replaces: e135_updateLdsIdxMetadata
///
/// Seeds `ldsIdxAfterDdc` with the IDENTITY over the DSC's labelled data structures — the map DDC
/// then rewrites as it renumbers them.
///
/// ⛔ THE KEY IS THE ENTRY'S OWN `ldsIdx_`, NOT ITS POSITION in `labeledDs_`, and ⛔ the seeding
/// ADDS to whatever is already there rather than replacing it.
pub fn update_lds_idx_metadata<T: LabeledDsIndices + ?Sized>(metadata: &mut Metadata, dsc: &T) {
    for lds in dsc.labeled_ds_indices() {
        metadata.lds_idx_after_ddc.insert(lds, lds);
    }
}

/// Replaces: e136_initGlobalData
///
/// Resets the corelet split dim to the FIRST dim the core data stage splits across corelets, or to
/// unset where it splits none.
///
/// ⛔ "FIRST" IS `std::map`'s ORDER, i.e. `PrimaryDimTypes`' ordinal order, not the DDL's; and ⛔ the
/// reset happens either way, so a stale dim from the previous DSC cannot survive.
pub fn init_global_data<T: DataStages + ?Sized>(global: &mut GlobalData, dsc: &T) {
    global.corelet_split_dim = dsc
        .corelet_split_dims(Metadata::CORE_DSTGID)
        .into_iter()
        .next();
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE PLACEMENT, OFFSET AND MASKING VOCABULARY — as entries 258-263 read it.
//
// ⭐ THE TRAITS ARE AGAIN THE MECHANISM FOR REACHING OPERANDS, the one part the campaign statement
// names as droppable: `traverseTreeDFS`, `getOwnerLoop`, `primaryDimToVal_st`,
// `dataStageDimToVal_compView_st`, `getBufferCapacityForNode` and the memory trackers'
// `checkAndAddDs` all live outside this campaign's file list. What these six units OWN is the
// placement decision, the offset arithmetic, and what both write into the DSC.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// A DIM'S ELEMENT DENSITY — `dimDensity` (`ddc/ddcv1.cpp:1958`), a `double` that is only ever `1.0`
/// or `1.0 / mxInfo_.blkSize`.
///
/// ⛔ A DIVISOR AND NOT A FLOAT: it is handed to `primaryDimToVal_st`, which folds it into an integer
/// extent, so a `double` here is a rounding decision the reference never made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Density(NonZeroU64);

impl Density {
    /// `1.0` — every dim that is not a scale tensor's mx dim.
    pub const FULL: Self = Self(NonZeroU64::new(1).expect("one is not zero"));

    /// `1.0 / mxInfo_.blkSize`.
    #[must_use]
    pub const fn per_block(block: NonZeroU64) -> Self {
        Self(block)
    }

    /// The divisor itself.
    #[must_use]
    pub const fn get(self) -> NonZeroU64 {
        self.0
    }
}

/// ONE DIM'S PADDING PARAMETERS — `AllocateNode::paddingSizes_.at(dim)` (`dsc/dsc2.h:1000`) narrowed
/// to the five fields entries 259 and 260 read.
///
/// ⛔ `stride_` IS NON-ZERO BY TYPE: the reference's `if (stride_ > 0)` (`ddc/ddcv1.cpp:1996`) is the
/// whole of its handling, and a zero stride multiplies an offset to nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddingSizes {
    /// `windowDim_`.
    pub window_dim: PrimaryDim,
    /// `stride_`.
    pub stride: NonZeroU64,
    /// `dilation_`.
    pub dilation: NonZeroU64,
    /// `padFront_`.
    pub pad_front: Elements,
    /// `padBack_`.
    pub pad_back: Elements,
}

/// THE CORES A DSC USES — `coreIdsUsed_` (`dsc/designSpaceConfig.h:120`), NON-EMPTY because
/// `coreIdsUsed_.front()` is entry 258's proxy site for every per-core memory and a DSC on no core
/// has nothing to place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoresUsed {
    first: Core,
    rest: Vec<Core>,
}

impl CoresUsed {
    /// A DSC runs on at least one core, and this is how that is stated.
    #[must_use]
    pub const fn new(first: Core, rest: Vec<Core>) -> Self {
        Self { first, rest }
    }

    /// `coreIdsUsed_.front()` — total.
    #[must_use]
    pub const fn head(&self) -> Core {
        self.first
    }

    /// `coreIdsUsed_`, in order.
    pub fn iter(&self) -> impl Iterator<Item = Core> + '_ {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }
}

/// WHETHER THE MEMORY TRACKERS ARE THE REAL ONES — `trueLXTracker_` (`ddc/ddcv1.h`).
///
/// ⛔ AN ARGUMENT AND NOT A FIELD BECAUSE `DT_CHECK_MSG(trueLXTracker_ || comp != LX)`
/// (`ddc/ddcv1.cpp:171`) MAKES IT A PRECONDITION: under [`LxTrackers::Ephemeral`] an LX allocation is
/// not placeable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LxTrackers {
    /// `trueLXTracker_ == false`.
    Ephemeral,
    /// `trueLXTracker_ == true`.
    True,
}

/// WHETHER A SUCCESSFUL PLACEMENT IS KEPT — entry 258's own `commitIfValid` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Commit {
    /// Probe only: the trackers are restored and nothing is written onto the allocations.
    No,
    /// Keep the addresses and buffer offsets if every allocation fitted.
    IfValid,
}

/// HOW THE MODEL'S L0 IS SHARED — `l0TetheredMode_` (`ddc/ddcv1.cpp:271`): on SEN1P5 two subcores
/// share one L0, either whole or split half and half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L0Tethered {
    /// `l0TetheredMode_ == false` — L0 is split, and each subcore's half of it is blocked off.
    Split,
    /// `l0TetheredMode_ == true` — the buffers are shared between the left and right corelets.
    Whole,
}

/// WHERE A MEMORY TRACKER PUT AN ALLOCATION — `checkAndAddDs`'s answer, whose `DOESNT_FIT` sentinel
/// is an arm and not an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placed {
    /// The byte address it was placed at.
    At(Bytes),
    /// `DOESNT_FIT` — this memory cannot hold the set.
    DoesntFit,
}

/// ONE MEMORY TRACKER — `memTrackers->getTracker(comp, core, corelet, row)` (`ddc/ddcv1.cpp:187`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrackerSite {
    /// `comp`.
    pub memory: DdcMemory,
    /// `core`.
    pub core: Core,
    /// `corelet`.
    pub corelet: Corelet,
    /// `row`.
    pub row: Row,
}

/// THE MEMORY TRACKERS — `DsTrackInMem` (`ddc/memTracker.h`), outside this campaign's file list.
///
/// ⛔ [`None`] FROM EITHER `check_and_add` IS THE `EXISTS` ANSWER, which entry 258 `DT_CHECK`s: a
/// name already in the tracker means this set is being placed twice over itself.
pub trait MemTrackers {
    /// `memCapacity`.
    fn capacity(&self, at: TrackerSite) -> Bytes;
    /// `backupEps(exphase)`, IDEMPOTENT — the reference's `trackerBackups.try_emplace` is that.
    fn backup(&mut self, at: TrackerSite);
    /// `restoreEps(exphase, backupInfo)` for every tracker backed up since.
    fn restore_all(&mut self);
    /// `removeDs(name, seps)`.
    fn remove(&mut self, at: TrackerSite, name: &StorageName);
    /// `addDsAtStartAddr(name, size, seps, addr)`.
    fn add_at(&mut self, at: TrackerSite, name: &StorageName, size: Bytes, address: Bytes);
    /// `checkAndAddDs(name, size, seps)`.
    fn check_and_add(&mut self, at: TrackerSite, name: &StorageName, size: Bytes)
    -> Option<Placed>;
    /// `checkAndAddDsAtAddr(name, size, seps, addr)`.
    fn check_and_add_at(
        &mut self,
        at: TrackerSite,
        name: &StorageName,
        size: Bytes,
        address: Bytes,
    ) -> Option<Placed>;
}

/// WHAT THE DESIGN SPACE TELLS ENTRY 258 — all of it `dsc/designSpaceConfig.h`, outside this
/// campaign's file list.
pub trait Placement {
    /// `coreIdsUsed_`.
    fn cores_used(&self) -> CoresUsed;
    /// `numCoreletsUsed_DSC2_` as the corelets it names.
    fn corelets_used(&self) -> Vec<Corelet>;
    /// `numCoreletsUsed_` — ⚠️ A DIFFERENT COUNT, AND ENTRY 258 READS BOTH: the placement walks
    /// `numCoreletsUsed_DSC2_` (`ddc/ddcv1.cpp:184`) and the buffer-offset copy walks this one
    /// (`:352`).
    fn corelets_used_total(&self) -> Vec<Corelet>;
    /// `getBufferCapacityForNode(node, ldsIdx, comp, corelet, row)`, with [`None`] for the reference's
    /// `-1` lds and its `-1, -1` site — both of which it passes verbatim.
    fn buffer_capacity(
        &self,
        alloc: AllocId,
        lds: Option<LdsIdx>,
        at: Option<(Corelet, Row)>,
    ) -> Bytes;
    /// `labeledDs_.at(lds).scaledLdsCategory_ == SCALE_TENSOR`.
    fn is_scale_tensor(&self, lds: LdsIdx) -> bool;
    /// `l0TetheredMode_`.
    fn l0_tethered(&self) -> L0Tethered;
    /// `coreIdToTetheredCoreCoord(core).subcoreId`.
    fn subcore(&self, core: Core) -> u32;
    /// `{coreFoldProp_, coreletFoldProp_} ++ sdscFoldProps_`'s size — how many axes the address fold
    /// space has (`ddc/ddcv1.cpp:326-338`).
    fn address_fold_depth(&self) -> usize;
}

/// THE DATASTAGE EXTENTS ENTRIES 259, 260 AND 262 READ — `primaryDimToVal_st`,
/// `dataStageDimToVal_compView_st` and the layout facts beside them, all
/// `dsc/designSpaceConfig.h`.
pub trait StageSizes {
    /// `primaryDimToVal_st(dim, comp, -1, corelet, padding, density)`, where [`None`] is the
    /// reference's `-1` corelet.
    fn dim_extent(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
        unit: SenComponent,
        corelet: Option<Corelet>,
        padding: PadType,
        density: Density,
    ) -> Extent;
    /// `dataStageDimToVal_compView_st(dim, unit, corelet, padding)`.
    fn comp_view(
        &self,
        stage: DatastageId,
        dim: PrimaryDim,
        unit: SenComponent,
        corelet: Option<Corelet>,
        padding: PadType,
    ) -> Extent;
    /// `lds.scale_.at(getDimIndexInLayoutOrder(dsType_, dim))` — `ONE` is a dim this lds does NOT
    /// broadcast.
    fn lds_replication(&self, lds: LdsIdx, dim: PrimaryDim) -> ReplicationFactor;
    /// `1.0 / mxInfo_.blkSize` on a scale tensor's mx dim, and [`Density::FULL`] everywhere else.
    fn dim_density(&self, lds: LdsIdx, dim: PrimaryDim) -> Density;
    /// `dsNode.coreletSplit_.at(dim)` — how many elements each corelet takes of that dim.
    fn corelet_split(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Vec<Elements>>;
    /// `allocNode->paddingSizes_.at(dim)`.
    fn alloc_padding_sizes(&self, alloc: AllocId, dim: PrimaryDim) -> Option<PaddingSizes>;
    /// `dataStageParam_.at(stage).ss_.paddingSizes_.at(dim)` — ⚠️ A DIFFERENT MAP FROM THE ONE ABOVE,
    /// and entry 259 reads the stage's while entry 260 reads the allocation's.
    fn stage_padding_sizes(&self, stage: DatastageId, dim: PrimaryDim) -> Option<PaddingSizes>;
    /// `getSizeDataStageForNode(node, node)` — which datastage sizes this allocation.
    fn size_stage(&self, alloc: AllocId) -> DatastageId;
    /// The one compute op whose `opFuncName` is `GENERIC_PARTIAL_REDUCTION` takes exactly this lds as
    /// its only input (`ddc/ddcv1.cpp:1949-1955`).
    fn is_sole_partial_reduction_input(&self, lds: LdsIdx) -> bool;
}

/// WHETHER ONE SITE STOOD PROXY FOR ALL OF THEM — `copyToCoreCl.at(alloc)` (`ddc/ddcv1.cpp:322`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proxy {
    /// The reference's `false` — every core (or corelet) was placed in its own tracker.
    Each,
    /// The reference's `true` — the first stood in for all of them, so its placement is copied out.
    First,
}

impl Proxy {
    /// `foldTypes[i] = copy ? Constant : Map` — a constant fold is what an address that does not vary
    /// over that axis gets.
    const fn fold(self) -> AddressFold {
        match self {
            Self::First => AddressFold::Constant,
            Self::Each => AddressFold::Map,
        }
    }
}

/// `copyToCoreCl.at(alloc)`, whose two `bool`s a pair transposes silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Proxies {
    core: Proxy,
    corelet: Proxy,
}

/// WHAT `tryAlloc` COLLECTED — held off the allocations until every set has fitted, because a probe
/// that did not fit must leave them as they were.
#[derive(Debug, Default)]
struct Placements {
    /// `startAddressCoreCorelet_`.
    start: BTreeMap<AllocId, BTreeMap<Core, BTreeMap<Corelet, Bytes>>>,
    /// `bufferOffsetCoreCorelet_`.
    offsets: BTreeMap<AllocId, BTreeMap<Core, BTreeMap<Corelet, Bytes>>>,
    /// `copyToCoreCl`.
    copied_from: BTreeMap<AllocId, Proxies>,
}

/// `skip_allocate_mem` (`ddc/ddcv1.cpp:243-259`).
///
/// ⚠️ ONLY THE HEAD OF A SHADOW GROUP IS EVER PLACED, so a DSC whose `shadowAllocations_` is empty
/// places NO labelled DS at all — the flag starts `true` and only a group's own first entry clears it.
fn heads_a_shadow_group(metadata: &Metadata, alloc: AllocId) -> bool {
    metadata
        .shadow_allocations
        .iter()
        .any(|group| group.first() == Some(&alloc))
}

/// `tryAlloc` (`ddc/ddcv1.cpp:169-321`) — [`None`] is a `DT_CHECK`, `Some(false)` its `return false`.
fn try_alloc<A: Arch, P, M>(
    dsc: &P,
    metadata: &Metadata,
    allocs: &AllocArena,
    trackers: &mut M,
    lx: LxTrackers,
    commit: Commit,
    placed: &mut Placements,
) -> Option<bool>
where
    P: Placement + StorageNames + ?Sized,
    M: MemTrackers + ?Sized,
{
    for (&memory, alloc_metadata) in &metadata.new_allocations {
        if lx == LxTrackers::Ephemeral && memory == DdcMemory::Lx {
            return None;
        }
        let l0 = matches!(memory, DdcMemory::L0 | DdcMemory::L0Scale);
        // For non-LX memories the first core stands proxy for all of them.
        let (cores, copy_core) = if l0 || memory == DdcMemory::Lx {
            (dsc.cores_used().iter().collect(), Proxy::Each)
        } else {
            (vec![dsc.cores_used().head()], Proxy::First)
        };
        let split_l0 = A::GEN > IsaGen::Rcudd1a && l0;
        let copy_corelet = if split_l0 { Proxy::Each } else { Proxy::First };
        let corelets = match copy_corelet {
            Proxy::First => vec![Corelet::at::<0>()],
            Proxy::Each => dsc.corelets_used(),
        };
        let cores: Vec<Core> = cores;
        for &core in &cores {
            for &corelet in &corelets {
                // Rows use 0 as proxy.
                for row in [Row::at::<0>()] {
                    let at = TrackerSite {
                        memory,
                        core,
                        corelet,
                        row,
                    };
                    trackers.backup(at);
                    let mut node_and_size: Vec<(AllocId, Bytes)> = Vec::new();
                    for (&lds, &alloc) in &alloc_metadata.lds_idx_and_alloc_node {
                        if !heads_a_shadow_group(metadata, alloc) {
                            continue;
                        }
                        let buffers = allocs.get(&alloc)?.placement.num_buffers.reserved();
                        let size = dsc
                            .buffer_capacity(alloc, Some(lds), Some((corelet, row)))
                            .0
                            .checked_mul(buffers.get())?;
                        node_and_size.push((alloc, Bytes(size)));
                    }
                    for &alloc in alloc_metadata.cons_id_and_alloc_node.values() {
                        node_and_size.push((alloc, A::sticks_to_bytes(Sticks(1))));
                    }
                    for (compute, &alloc) in &alloc_metadata.comp_and_alloc_node {
                        let node = allocs.get(&alloc)?;
                        if node.placement.num_buffers != NumBuffers::Single {
                            return None;
                        }
                        let opaque = metadata.opaque_ops.get(compute)?;
                        let size = dsc.buffer_capacity(alloc, node.lds, Some((corelet, row)));
                        let unroll = size.0 / A::BYTES_PER_STICK;
                        // More than the opaque op can handle, or not a power of two.
                        if unroll > u64::from(opaque.max_unroll.0)
                            || unroll == 0
                            || !unroll.is_power_of_two()
                        {
                            return Some(false);
                        }
                        let regs = u64::try_from(opaque.internal_regs.len()).ok()?;
                        let unrolled = u64::try_from(opaque.internal_regs_with_unroll()).ok()?;
                        let sticks = regs.checked_add(unroll.checked_sub(1)?.checked_mul(unrolled)?)?;
                        node_and_size.push((alloc, A::sticks_to_bytes(Sticks(sticks))));
                    }
                    for &(alloc, _) in &node_and_size {
                        let name = get_lds_or_const_name_of_alloc_node(allocs.get(&alloc)?, dsc)?;
                        trackers.remove(at, &name);
                    }
                    if split_l0 && dsc.l0_tethered() == L0Tethered::Split {
                        // L0 is half-half split between the left and right corelets, so this subcore's
                        // half is blocked off: the top half for subcore 0, the bottom half otherwise.
                        let half = Bytes(trackers.capacity(at).0 / 2);
                        let (blocked, address) = if dsc.subcore(core) == 0 {
                            ("Reg-not-available-to-subcore0", half)
                        } else {
                            ("Reg-not-available-to-subcore1", Bytes(0))
                        };
                        trackers.add_at(at, &StorageName(blocked.to_owned()), half, address);
                    }
                    let mut all_ds_fit = true;
                    for &(alloc, size) in &node_and_size {
                        let node = allocs.get(&alloc)?;
                        let mut my_size = size;
                        if node.placement.num_buffers.is_streaming() {
                            // Full capacity reserved for a circular buffer.
                            let capacity = trackers.capacity(at);
                            my_size = my_size.max(
                                if split_l0 && dsc.l0_tethered() == L0Tethered::Split {
                                    Bytes(capacity.0 / 2)
                                } else {
                                    capacity
                                },
                            );
                        }
                        let name = get_lds_or_const_name_of_alloc_node(node, dsc)?;
                        let scale_xrf = node.lds.is_some_and(|lds| dsc.is_scale_tensor(lds))
                            && node.component == SenComponent::Ptxrf;
                        let site = if scale_xrf {
                            let start = A::sticks_to_bytes(A::XRF_SCALE_START);
                            trackers.check_and_add_at(at, &name, my_size, start)?
                        } else {
                            trackers.check_and_add(at, &name, my_size)?
                        };
                        let Placed::At(address) = site else {
                            all_ds_fit = false;
                            break;
                        };
                        if commit == Commit::IfValid {
                            placed
                                .start
                                .entry(alloc)
                                .or_default()
                                .entry(core)
                                .or_default()
                                .insert(corelet, address);
                            // ⚠️ THE SIZE ASKED FOR, not the capacity a streaming buffer widened it to.
                            placed
                                .offsets
                                .entry(alloc)
                                .or_default()
                                .entry(core)
                                .or_default()
                                .insert(corelet, Bytes(size.0 / node.placement.num_buffers.reserved()));
                            placed.copied_from.insert(
                                alloc,
                                Proxies {
                                    core: copy_core,
                                    corelet: copy_corelet,
                                },
                            );
                        }
                    }
                    if !all_ds_fit {
                        return Some(false);
                    }
                }
            }
        }
    }
    Some(true)
}

/// Replaces: e258_allocAllMem
///
/// Places every new allocation in its memory tracker at each (core, corelet, row) site that owns one,
/// and — when it all fitted and the caller asked to commit — writes the addresses into each
/// allocation's fold space, its buffer offsets beside them, and copies both out of every proxy site.
///
/// ⛔ [`None`] IS EVERY `DT_CHECK`: ephemeral trackers under an LX allocation, an empty shadow group
/// (`.at(0)`), an opaque op's allocation that is not single-buffered, a name the tracker already holds,
/// a fold space already dimensioned, and a proxied axis placed at more than one coordinate.
/// ⚠️ THE `coreArch <= MPW4_ISA` PTARF PREFILL IS OMITTED: [`IsaGen`] has no MPW4, so it is dead here.
pub fn alloc_all_mem<A: Arch, P, T, M>(
    dsc: &P,
    tree: &T,
    metadata: &mut Metadata,
    allocs: &mut AllocArena,
    trackers: &mut M,
    lx: LxTrackers,
    commit: Commit,
) -> Option<bool>
where
    P: Placement + StorageNames + ?Sized,
    T: ScheduleWalk + ?Sized,
    M: MemTrackers + ?Sized,
{
    let mut allocate_size: BTreeMap<AllocId, Option<Bytes>> = BTreeMap::new();
    for alloc in tree.allocates() {
        let lds = allocs.get(&alloc)?.lds;
        allocate_size.insert(
            alloc,
            lds.map(|lds| dsc.buffer_capacity(alloc, Some(lds), None)),
        );
    }
    // The first entry of each shadow group is made the largest, since only it is given memory.
    for group in &mut metadata.shadow_allocations {
        let mut max_size = Bytes(0);
        let mut switch_index = 0;
        for (index, alloc) in group.iter().enumerate() {
            if let Some(size) = *allocate_size.get(alloc)? {
                if size > max_size {
                    max_size = size;
                    switch_index = index;
                }
            }
        }
        if group.is_empty() {
            return None;
        }
        group.swap(0, switch_index);
    }

    let mut placed = Placements::default();
    let success = try_alloc::<A, P, M>(dsc, metadata, allocs, trackers, lx, commit, &mut placed)?;
    if !success || commit == Commit::No {
        trackers.restore_all();
        return Some(success);
    }

    let depth = dsc.address_fold_depth();
    for (&alloc, addresses) in &placed.start {
        let proxies = *placed.copied_from.get(&alloc)?;
        let node = allocs.get_mut(&alloc)?;
        if !node.start_address.has_zero_fold_dim()
            || (proxies.core == Proxy::First && addresses.len() != 1)
            || (proxies.corelet == Proxy::First && addresses.values().next()?.len() != 1)
        {
            return None;
        }
        node.start_address
            .build_fold_space(depth, proxies.core.fold(), proxies.corelet.fold());
        for (&core, per_corelet) in addresses {
            for (&corelet, &address) in per_corelet {
                node.start_address.insert(core, corelet, address);
            }
        }
    }
    for (&alloc, offsets) in &placed.offsets {
        allocs.get_mut(&alloc)?.placement.buffer_offset = offsets.clone();
    }
    for (&alloc, proxies) in &placed.copied_from {
        if proxies.corelet == Proxy::First {
            // ⚠️ `numCoreletsUsed_`, and NOT the `_DSC2_` count the placement above walked.
            let corelets = dsc.corelets_used_total();
            let node = allocs.get_mut(&alloc)?;
            for per_corelet in node.placement.buffer_offset.values_mut() {
                let head = *per_corelet.get(&Corelet::at::<0>())?;
                for &corelet in corelets.iter().skip(1) {
                    per_corelet.insert(corelet, head);
                }
            }
        }
        if proxies.core == Proxy::First {
            let cores = dsc.cores_used();
            let node = allocs.get_mut(&alloc)?;
            if node.placement.num_buffers.switches() {
                let at_head = node.placement.buffer_offset.get(&cores.head())?.clone();
                for core in cores.iter().skip(1) {
                    node.placement.buffer_offset.insert(core, at_head.clone());
                }
            }
        }
    }
    Some(true)
}

/// `stickSizePerDim.count(dim) ? stickSizePerDim.at(dim) : 1` — total, and NON-ZERO because the
/// reference divides an offset by it.
fn cumulative_stick_size(sizes: &[(PrimaryDim, Elements)], dim: PrimaryDim) -> Option<NonZeroU64> {
    sizes
        .iter()
        .find(|(walked, _)| *walked == dim)
        .map_or(NonZeroU64::new(1), |(_, size)| NonZeroU64::new(size.0))
}

/// Replaces: e259_calculateClStartAddress
///
/// Offsets every corelet's copy of an LX allocation from corelet 0's by the bytes its non-broadcast
/// dims span up to and including the corelet-split dim, and copies its buffer offset across.
///
/// ⛔ [`None`] IS EVERY `DT_CHECK`: no such allocation, a component other than `LX`, two split dims, a
/// split dim the size datastage does not split, a padded split dim that is not
/// `PADDED_FULLSPAN_WUNNEEDED` on `I` with a chunk stride, a non-zero offset over a corelet fold axis
/// that is `Constant`, and a zero stick size — which the reference divides by.
/// ⚠️ THE OFFSET ACCUMULATES ACROSS CORELETS: `overallOffset` is declared OUTSIDE the corelet loop.
pub fn calculate_cl_start_address<A: Arch, P>(
    dsc: &P,
    metadata: &Metadata,
    allocs: &mut AllocArena,
    alloc: AllocId,
) -> Option<()>
where
    P: Placement + StageSizes + LdsSticks + ?Sized,
{
    let corelets = dsc.corelets_used();
    if corelets.len() < 2 {
        return Some(());
    }
    let node = allocs.get(&alloc)?;
    if node.component != SenComponent::Lx {
        return None;
    }
    let lds = node.lds?;
    let unit = node.component;
    let padding = node.placement.padding.clone();
    let buffered = node.placement.num_buffers.switches();
    let cl_fold = node.start_address.func_type(FoldPosition::Corelet);

    let mut non_broadcast: Vec<PrimaryDim> = Vec::new();
    let mut split: Option<PrimaryDim> = None;
    for dim in node.layout.dims().iter() {
        if dsc.lds_replication(lds, dim) != ReplicationFactor::ONE {
            continue;
        }
        non_broadcast.push(dim);
        if metadata.cl_split_dims.contains(&dim) && split.replace(dim).is_some() {
            return None;
        }
    }
    if dsc.is_sole_partial_reduction_input(lds) {
        split = None;
    }

    let size_stage = dsc.size_stage(alloc);
    let sticks = cumulative_stick_sizes(&dsc.stick_dims(lds), StickPart::Whole)?;
    let stick_bytes = i64::try_from(A::BYTES_PER_STICK.get()).ok()?;

    let mut overall: i64 = 0;
    for corelet in corelets.iter().copied().skip(1) {
        let previous = Corelet::checked(corelet.get() - 1)?;
        if let Some(split_dim) = split {
            dsc.corelet_split(size_stage, split_dim)?;
            let mut offset = stick_bytes;
            for &dim in &non_broadcast {
                let density = dsc.dim_density(lds, dim);
                let stick = i64::try_from(cumulative_stick_size(&sticks, dim)?.get()).ok()?;
                if dim != split_dim {
                    let extent =
                        dsc.dim_extent(size_stage, dim, unit, Some(previous), padding.get(dim), density);
                    offset *= extent.0 / stick;
                    continue;
                }
                // Use the chunk data stage where the dim is corelet-split.
                let span = match padding.get(dim) {
                    PadType::NoPad => {
                        dsc.dim_extent(
                            Metadata::CHUNK_DSTGID,
                            dim,
                            unit,
                            Some(previous),
                            padding.get(dim),
                            density,
                        )
                        .0
                    }
                    // offset_in_element = size_of_i * stride, and `I` is the only dim supported.
                    PadType::PaddedFullSpanWUnneeded if dim == PrimaryDim::I => {
                        let sizes = dsc.stage_padding_sizes(Metadata::CHUNK_DSTGID, dim)?;
                        let taken = dsc
                            .corelet_split(Metadata::CHUNK_DSTGID, dim)?
                            .get(previous.get() as usize)
                            .copied()?;
                        i64::try_from(taken.0.checked_mul(sizes.stride.get())?).ok()?
                    }
                    _ => return None,
                };
                offset *= span / stick;
                break;
            }
            overall += offset;
        }
        if overall != 0 && cl_fold != Some(AddressFold::Map) {
            return None;
        }
        let node = allocs.get_mut(&alloc)?;
        // Corelet 0 is the only corelet with a placed address at this point.
        for (core, address) in node.start_address.at_corelet(Corelet::at::<0>()) {
            let placed = Bytes(address.0.checked_add_signed(overall)?);
            node.start_address.insert(core, corelet, placed);
        }
        if buffered {
            for per_corelet in node.placement.buffer_offset.values_mut() {
                let at_zero = per_corelet.get(&Corelet::at::<0>()).copied()?;
                per_corelet.insert(corelet, at_zero);
            }
        }
    }
    Some(())
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
//    LOOP OFFSETS AND ADDRESSES — ENTRY 260'S VOCABULARY
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OUTPUT OF A COMPUTE — an index into `outputs_`/`outputsLdsAndLoopOffsets_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputIdx(pub usize);

/// WHICH OPERAND ONE `DataInfo` IS — the four `fillDataInfo` call sites (`ddc/ddcv1.cpp:2861-2905`,
/// `:3010-3025`) AS A VALUE.
///
/// ⭐ THIS IS WHAT MAKES THE FILL ADDRESSABLE. The reference hands the lambda a `dsc2::DataInfo&`
/// taken out of one of four parallel vectors; naming the site instead means the later constant-offset
/// fixups cannot land on a different operand than the one they were computed for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperandSite {
    /// `srcLdsAndLoopOffsets_`.
    TransferSrc(NodeId),
    /// `dstLdsAndLoopOffsets_.at(index)`.
    TransferDst(NodeId, DestIdx),
    /// `inputsLdsAndLoopOffsets_.at(index)`.
    ComputeInput(NodeId, InputIdx),
    /// `outputsLdsAndLoopOffsets_.at(index)`.
    ComputeOutput(NodeId, OutputIdx),
}

impl OperandSite {
    /// The `processorNode` this operand hangs off.
    #[must_use]
    pub const fn node(self) -> NodeId {
        match self {
            Self::TransferSrc(node)
            | Self::TransferDst(node, _)
            | Self::ComputeInput(node, _)
            | Self::ComputeOutput(node, _) => node,
        }
    }
}

/// WHERE THE LOOP ELEMENT OFFSETS COME FROM — the `ddcGlobal` pair `datastageBasedElemOff` and
/// `verifyCoordinateBasedLoopElemOff` as the three states they have.
///
/// ⛔ AN ENUM AND NOT TWO `bool`s: their fourth combination is `datastageBasedElemOff` with a verify
/// that can no longer fail, because that arm writes the very value the check compares against
/// (`ddc/ddcv1.cpp:2585-2589`), so it is not a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemOffsets {
    /// Neither switch — `loopDistributionParamInfo` alone, and the constant offsets come from the
    /// coordinates.
    Distribution,
    /// `verifyCoordinateBasedLoopElemOff` — the same answer, cross-checked against the datastage
    /// ladder's.
    DistributionVerified,
    /// `datastageBasedElemOff` — the datastage ladder's answer, and the constant offsets come from the
    /// padding alone.
    Datastage,
}

impl ElemOffsets {
    /// `datastageBasedElemOff`.
    #[must_use]
    pub const fn is_datastage(self) -> bool {
        matches!(self, Self::Datastage)
    }

    /// `verifyCoordinateBasedLoopElemOff || datastageBasedElemOff` — whether the datastage ladder runs
    /// at all.
    #[must_use]
    pub const fn runs_ladder(self) -> bool {
        !matches!(self, Self::Distribution)
    }
}

/// WHETHER AN UNPADDED INDEX MAY REACH A `PADDED_NOZEROPAD` ALLOCATION — entry 260's own
/// `allowUnpaddedIndexingAtPaddedNoZeroPad` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpaddedIndexing {
    /// `false` — *"Unsupported access into a dimension that is in padded_nozeropad form."*
    Forbidden,
    /// `true` — an `Unpadded` or `WindowDim` index is scaled by the window stride instead.
    Allowed,
}

/// ONE `constEleOffsets_` ENTRY — how many elements of that dim to skip ONCE, and SIGNED because the
/// pad-back arm subtracts (`ddc/ddcv1.cpp:2648`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstEleOffset(pub i64);

impl ConstEleOffset {
    /// The value `di.constEleOffsets_[core][cl][dim]` takes when `operator[]` default-constructs it.
    pub const ZERO: Self = Self(0);
}

/// HOW MANY TRIPS A LOOP MAKES — `parametricIterCount`, or the numerator datastage's extent over the
/// denominator's step (`ddc/ddcv1.cpp:2498-2502`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IterCount(pub i64);

/// A LOOP'S TWO DATASTAGES — `numId_` and `denId_` (`dsc/dsc2.h:572-573`) AS ONE VALUE: the offset
/// ladder takes its step from the denominator and its trip count from the numerator, and swapping the
/// two is silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopStages {
    /// `numId_`.
    pub num: DatastageId,
    /// `denId_`.
    pub den: DatastageId,
}

/// WHICH WORK SLICE ONE CORE TAKES OF ONE DIM — `coreIdToWkSlice_.at(core).at(dim)`
/// (`dsc/superdsc.h:69`), a fold COORDINATE and not an element count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkSlice(pub i64);

/// WHAT ONE OPERAND'S `DataInfo` GAINS — the six fields `fillDataInfo` writes, computed TOGETHER and
/// installed in one move so that a half-filled `DataInfo` is not a state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataInfoFill {
    /// `startAddr_`, already at this unit's address granularity.
    pub start_address: StartAddress,
    /// `isStartAddrSymbolic_`, copied off the allocation.
    pub is_start_addr_symbolic: bool,
    /// `loopEleOffsets_[corelet][loop][dim]`.
    pub loop_ele_offsets: BTreeMap<Corelet, BTreeMap<LoopId, BTreeMap<PrimaryDim, LoopEleOffset>>>,
    /// `constEleOffsets_[core][corelet][dim]`.
    pub const_ele_offsets: BTreeMap<Core, BTreeMap<Corelet, BTreeMap<PrimaryDim, ConstEleOffset>>>,
    /// `bufferSwitchPosition_` — the loop a multiply-buffered allocation switches buffers at.
    pub buffer_switch_position: Option<LoopId>,
    /// `bufferAddrOffset_`, also at this unit's address granularity.
    pub buffer_addr_offset: BTreeMap<Core, BTreeMap<Corelet, Bytes>>,
}

/// `dsc2::memories.count(storage)` (`dsc/dscdefn.cpp:142-144`) — the storages a `DataLocation` may
/// name and `fillDataInfo` will fill for.
///
/// ⛔ NOT [`DdcMemory`], WHICH IS `ddc::memories`: that set has neither `LRFREG`, `L3LUIBR`,
/// `L3SUIBR`, `PESTATE`, `SFPSTATE`, `LXLUSCALEREG` nor `QGI`, so it cannot answer this question.
#[must_use]
pub const fn is_dsc_memory(storage: SenComponent) -> bool {
    matches!(
        storage,
        SenComponent::Lx
            | SenComponent::L0
            | SenComponent::L0Scale
            | SenComponent::Lrfreg
            | SenComponent::Pelrf
            | SenComponent::Sfplrf
            | SenComponent::Ptarf
            | SenComponent::Ptxrf
            | SenComponent::Ptirf
            | SenComponent::Hbm
            | SenComponent::L3luibr
            | SenComponent::L3suibr
            | SenComponent::Pestate
            | SenComponent::Sfpstate
            | SenComponent::Lxluscalereg
            | SenComponent::Qgi
    )
}

/// `dsc2::nonCoreletMemories.count(component)` (`dsc/dscdefn.cpp:145-146`) — a memory that is NOT
/// per-corelet, and so one whose offsets may have to be read across both corelets at once.
#[must_use]
pub const fn is_non_corelet_memory(component: SenComponent) -> bool {
    matches!(
        component,
        SenComponent::Lx
            | SenComponent::Hbm
            | SenComponent::L3luibr
            | SenComponent::L3suibr
            | SenComponent::Qgi
    )
}

/// THE `SuperDsc`'S SYMBOL TABLE — `sdsc_->symbolDefinitions_`, outside this campaign's file list.
///
/// ⭐ THE ARM IS THE TRAIT'S, NOT THE REPRESENTATION'S. Under `isStartAddrSymbolic_` every value in a
/// [`StartAddress`] is a SYMBOL ID rather than a byte count, so the division cannot be performed —
/// it has to be *defined*, and the table that defines it is the only thing that can spell the result.
pub trait Symbols {
    /// `startAddr_.apply({}, [&](sym) { return addVar(DIV, {{true, sym}, {false, scale}}); })`
    /// (`ddc/ddcv1.cpp:2419-2424`) — one new symbol per placed value, and the rewritten address.
    fn divide_symbols(&mut self, address: &StartAddress, by: NonZeroU64) -> StartAddress;
}

/// WHAT ENTRY 260 READS OFF THE SCHEDULE TREE ON TOP OF [`ScheduleWalk`] — the unfiltered DFS, each
/// node's kind, and the loop facts the offset ladder needs.
pub trait ScheduleNodes: ScheduleWalk {
    /// `traverseTreeDFSMutable()` with no filter — every node, in DFS order.
    fn nodes(&self) -> Vec<NodeId>;
    /// `nodeType_`, absent for a node this tree does not hold.
    fn kind(&self, node: NodeId) -> Option<NodeKind>;
    /// `isParametricLoop()` (`dsc/dsc2.h:596`).
    fn is_parametric(&self, at: LoopId) -> bool;
    /// `numId_` and `denId_` — TOTAL: every loop node carries both.
    fn loop_stages(&self, at: LoopId) -> LoopStages;
    /// `parametricStride(dsc)`.
    fn parametric_stride(&self, at: LoopId) -> LoopEleOffset;
    /// `parametricIterCount(dsc, corelet, unit)`.
    fn parametric_iter_count(&self, at: LoopId, corelet: Corelet, unit: SenComponent) -> IterCount;
    /// `isNodeRelevant(unit)` (`dsc/dsc2.h:470`).
    fn is_relevant(&self, node: NodeId, unit: SenComponent) -> bool;
    /// `getNextView(unit).size()` — how many children of this node that unit sees.
    fn next_view_len(&self, node: NodeId, unit: SenComponent) -> usize;
    /// `allocNode->getOwnerLoop()` — the innermost LOOP the ALLOCATE sits under, [`None`] at the root.
    fn alloc_owner_loop(&self, alloc: AllocId) -> Option<LoopId>;
    /// `TransferNode::paddingInfo_.isEmpty() == false` (`dsc/dsc2.h:838`).
    fn transfer_has_padding(&self, node: NodeId) -> bool;
    /// `type_`, `exUnit_`, `inputs_` and `outputs_`, each zipped with its offsets vector.
    fn compute(&self, node: NodeId) -> Option<ComputeNode>;
    /// `repetitionWithOffset_.forOutputs_.size()` (`dsc/dsc2.h:950-953`) — how many outputs of a
    /// cloned compute take a repetition offset.
    fn repetition_with_offset_outputs(&self, node: NodeId) -> usize;
}

/// WHAT ENTRY 260 READS OFF THE DESIGN SPACE — the allocation behind each storage, the address
/// granularity table, and the four datastage facts the offset ladder and its fixups need.
pub trait OffsetSizes {
    /// `labeledDs_.at(lds).memOrg_.at(storage).allocateNode_` — ⛔ [`None`] IS THE `DT_ERROR`
    /// *"does not have memOrg_ entry for DataLocation storge"* (`ddc/ddcv1.cpp:2400-2408`).
    fn lds_alloc(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId>;
    /// `constantInfo_.at(constant).allocations_.at(storage)`.
    fn const_alloc(&self, constant: ConstIdx, storage: SenComponent) -> Option<AllocId>;
    /// `dscGlobal.sysDef.addressGranularityScalePerUnit.at({generic, storage})`, NON-ZERO because
    /// `DT_CHECK(addrScale > 0)` guards a division by it (`:2415`).
    fn address_scale(&self, unit: GenericComp, storage: SenComponent) -> Option<NonZeroU64>;
    /// `dataStageParam_.at(stage).ss_.paddingSizes_` — every padded dim with its sizes, which is what
    /// the window-dim rescue searches (`:2461-2474`).
    fn stage_padding_dims(&self, stage: DatastageId) -> Vec<(PrimaryDim, PaddingSizes)>;
    /// `dataStageParam_.at(stage).ss_.symbolicDimInfo_.count(dim)`.
    fn has_symbolic_dim(&self, stage: DatastageId, dim: PrimaryDim) -> bool;
    /// `dataStageParam_.at(stage).ss_.peSfpSplit_` — the dims split between the PE and the SFP.
    fn pe_sfp_split_dims(&self, stage: DatastageId) -> Vec<PrimaryDim>;
    /// `getBlockTransferSizePerDim(transfer, unit, corelet)[dim]` — ZERO for a dim the map does not
    /// hold, because the reference indexes it with `operator[]`.
    fn block_transfer_size(
        &self,
        node: NodeId,
        unit: SenComponent,
        corelet: Corelet,
        dim: PrimaryDim,
    ) -> Elements;
    /// `loopDistributionParamInfo.at(node).at(alloc).at(loop).at(dim)
    /// .temporalStridePostDistribution` — four chained `.at()`s, so [`None`] is any of them.
    fn temporal_stride(
        &self,
        node: NodeId,
        alloc: AllocId,
        at: LoopId,
        dim: PrimaryDim,
    ) -> Option<LoopEleOffset>;
}

/// WHAT THE COORDINATE-BASED CONSTANT OFFSET REACHES THROUGH — the work-slice tables, the two
/// coordinates being compared, and the two fold-algebra primitives, which live in
/// `util/foldManager/foldInfrastructure.h` and are outside this campaign's file list.
pub trait CoordinateOffsets {
    /// `coordinates.coreIdToWkSlice_`, falling back to `sdsc_->coreIdToWkSlice_` where it is empty, at
    /// one core — ⛔ [`None`] IS `count(coreId) == 0`, which is the walk's own `continue`.
    fn node_work_slices(&self, at: OperandSite, core: Core)
    -> Option<BTreeMap<PrimaryDim, WorkSlice>>;
    /// The same off the allocation's `sliceViewCoordinates_`/`allocateCoordinates_`.
    fn alloc_work_slices(
        &self,
        alloc: AllocId,
        core: Core,
    ) -> Option<BTreeMap<PrimaryDim, WorkSlice>>;
    /// `transferCoordinates_`, `inputCoordinates_.at(i)` or `outputCoordinate_` — and under
    /// [`ElemOffsets::Datastage`] a compute input reads the OUTPUT's coordinate (`:3016-3018`).
    fn node_coordinate(&self, at: OperandSite, offsets: ElemOffsets) -> Coordinate;
    /// `sliceViewCoordinates_` where its `coordinates_` is non-empty, else `allocateCoordinates_`.
    fn alloc_coordinate(&self, alloc: AllocId) -> Coordinate;
    /// `getRelevantCoreCl()` (`dsc/dsc2.h:471`).
    fn relevant_core_cl(&self, node: NodeId) -> CoreClSet;
    /// `getSingleData({{Core, core}, {Corelet, corelet}, {RowSplit, row}})` — this dim's affine value
    /// at one spatial coordinate.
    fn single_beta(&self, folds: &FoldDim, core: i64, corelet: i64, row: i64) -> FoldCoeff;
    /// `FoldInfraUtils::lexiAffineSolveDistanceInSteps(folds, beta, fixed)` — how far `beta` is from
    /// the fold space's origin, in that dim's own steps.
    fn distance_in_steps(
        &self,
        folds: &FoldDim,
        beta: FoldCoeff,
        fixed: &BTreeMap<usize, i64>,
    ) -> ConstEleOffset;
}

/// WHERE THE FILLED `DataInfo`S GO — the four operand slots and the two `lastFusableParentLoop`
/// fields entry 260 writes.
pub trait DataInfoSink {
    /// `di = <the fill>` at one operand.
    fn fill(&mut self, at: OperandSite, fill: DataInfoFill) -> Option<()>;
    /// `constEleOffsets_[core][corelet][dim] = offset` — the after-the-fact fixups, which reach an
    /// operand whose fill has already been installed.
    fn set_const_ele_offset(
        &mut self,
        at: OperandSite,
        core: Core,
        corelet: Corelet,
        dim: PrimaryDim,
        offset: ConstEleOffset,
    ) -> Option<()>;
    /// `constEleOffsets_.empty()` — the guard every fixup but the replication one carries.
    fn const_ele_offsets_empty(&self, at: OperandSite) -> Option<bool>;
    /// `lastFusableParentLoopSrc_`.
    fn set_last_fusable_src(&mut self, node: NodeId, at: Option<LoopId>) -> Option<()>;
    /// `lastFusableParentLoopDst_`, CLEARED AND REFILLED, one entry per destination.
    fn set_last_fusable_dsts(&mut self, node: NodeId, at: Vec<Option<LoopId>>) -> Option<()>;
}

/// EVERYTHING ENTRY 260 READS — one value, so the eight things the reference reaches for through
/// `currDsc`, `sdsc_`, `metadata` and `ddcGlobal` arrive together and in one lifetime.
pub struct OffsetInputs<'a, P: ?Sized, T: ?Sized, C: ?Sized> {
    /// The design space.
    pub dsc: &'a P,
    /// The schedule tree.
    pub tree: &'a T,
    /// The coordinate and work-slice tables.
    pub coords: &'a C,
    /// `metadata`.
    pub metadata: &'a Metadata,
    /// The allocate nodes the tree's ALLOCATEs name.
    pub allocs: &'a AllocArena,
    /// `loopsBelowChunkBoundary`.
    pub global: &'a GlobalData,
    /// `datastageBasedElemOff` / `verifyCoordinateBasedLoopElemOff`.
    pub offsets: ElemOffsets,
    /// `allowUnpaddedIndexingAtPaddedNoZeroPad`.
    pub unpadded: UnpaddedIndexing,
}

/// Replaces: e260_fillLoopOffsetsAndAddresses
///
/// Fills every transfer and compute operand with its allocation's start address at that unit's address
/// granularity, one element offset per enclosing loop and dim, the padding's and the coordinates'
/// constant offsets, its buffer switch position, and the metadata's cloned-transfer offsets on top.
///
/// ⛔ [`None`] IS EVERY `DT_ERROR`/`DT_CHECK`: a storage the lds has no `memOrg_` for, an allocation
/// no parent loop holds, a zero address granularity, and two offset methods that disagree.
pub fn fill_loop_offsets_and_addresses<A, P, T, C, K, S>(
    inputs: &OffsetInputs<'_, P, T, C>,
    sink: &mut K,
    symbols: &mut S,
) -> Option<()>
where
    A: Arch,
    P: Placement + StageSizes + OffsetSizes + LdsSticks + ?Sized,
    T: ScheduleNodes + ?Sized,
    C: CoordinateOffsets + ?Sized,
    K: DataInfoSink + ?Sized,
    S: Symbols + ?Sized,
{
    let dsc = inputs.dsc;
    let tree = inputs.tree;
    let coords = inputs.coords;
    let metadata = inputs.metadata;
    let cores = dsc.cores_used();
    let corelets = dsc.corelets_used();

    for node in tree.nodes() {
        if metadata.external_nodes.contains(&node) {
            continue;
        }
        let owner_loop = tree.owner_loop(node);
        match tree.kind(node) {
            Some(NodeKind::Transfer) => {
                let transfer = tree.transfer(node)?;
                if transfer.src.unit == SenComponent::NoComponent
                    && !tree.transfer_has_padding(node)
                {
                    continue;
                }
                let src_site = OperandSite::TransferSrc(node);
                let src_coord = coords.node_coordinate(src_site, inputs.offsets);
                if let Some(filled) = fill_data_info::<A, _, _, _, _>(
                    inputs,
                    symbols,
                    src_site,
                    &transfer.src,
                    owner_loop,
                    &src_coord,
                )? {
                    sink.fill(src_site, filled)?;
                }
                sink.set_last_fusable_src(
                    node,
                    last_fusable_loop(dsc, tree, node, transfer.src.unit),
                )?;

                // ⛔ THE `dstLdsAndLoopOffsets_.size() != dstVias_.size()` `DT_ERROR` IS UNSPELLABLE:
                // [`Dsts`] holds each destination's location and its offsets as ONE entry, so the two
                // cannot be different lengths.
                let mut fusable_dsts = Vec::new();
                for (index, dst) in transfer.dsts.iter().enumerate() {
                    let site = OperandSite::TransferDst(node, DestIdx(u32::try_from(index).ok()?));
                    let coord = coords.node_coordinate(site, inputs.offsets);
                    if let Some(filled) = fill_data_info::<A, _, _, _, _>(
                        inputs, symbols, site, dst, owner_loop, &coord,
                    )? {
                        sink.fill(site, filled)?;
                    }
                    fusable_dsts.push(last_fusable_loop(dsc, tree, node, dst.unit));
                }
                sink.set_last_fusable_dsts(node, fusable_dsts)?;
                transfer_metadata_offsets(inputs, sink, node, &transfer, &cores, &corelets)?;
            }
            Some(NodeKind::Compute) => {
                let compute = tree.compute(node)?;
                // ⛔ THE `"Compute node input/output missing information"` `DT_ERROR` IS UNSPELLABLE:
                // [`ComputeNode`] zips `inputs_`/`outputs_` with their offsets, one [`Operand`] each.
                for (index, input) in compute.inputs.iter().enumerate() {
                    let site = OperandSite::ComputeInput(node, InputIdx(index));
                    let coord = coords.node_coordinate(site, inputs.offsets);
                    if let Some(filled) = fill_data_info::<A, _, _, _, _>(
                        inputs, symbols, site, input, owner_loop, &coord,
                    )? {
                        sink.fill(site, filled)?;
                    }
                }
                for (index, output) in compute.outputs.iter().enumerate() {
                    let site = OperandSite::ComputeOutput(node, OutputIdx(index));
                    let coord = coords.node_coordinate(site, inputs.offsets);
                    if let Some(filled) = fill_data_info::<A, _, _, _, _>(
                        inputs, symbols, site, output, owner_loop, &coord,
                    )? {
                        sink.fill(site, filled)?;
                    }
                }
                if metadata.node_cloning_map.contains_key(&node) {
                    clone_repetition_offsets(inputs, sink, node, &compute, &cores, &corelets)?;
                }
            }
            _ => {}
        }
    }
    Some(())
}

/// `fillDataInfo(di, loc, loopLocation, processorNode, isProducer, coordinates)`
/// (`ddc/ddcv1.cpp:2359-2673`) as the value it computes rather than the reference it mutates.
///
/// [`None`] is every abort; `Some(None)` is the lambda's two early returns, which leave the operand
/// exactly as it was.
/// ⚠️ `isProducer` IS DROPPED: its only reader is the `dataConnectLoops` block, which is commented out
/// at `ddc/ddcv1.cpp:2424-2427`.
fn fill_data_info<A, P, T, C, S>(
    inputs: &OffsetInputs<'_, P, T, C>,
    symbols: &mut S,
    at: OperandSite,
    loc: &Operand,
    loop_location: Option<LoopId>,
    coordinates: &Coordinate,
) -> Option<Option<DataInfoFill>>
where
    A: Arch,
    P: Placement + StageSizes + OffsetSizes + ?Sized,
    T: ScheduleNodes + ?Sized,
    C: CoordinateOffsets + ?Sized,
    S: Symbols + ?Sized,
{
    let dsc = inputs.dsc;
    let tree = inputs.tree;
    if loc.data.my_lds_idx.is_none() && loc.data.constant_id.is_none() {
        return Some(None);
    }
    if !is_dsc_memory(loc.storage) {
        return Some(None);
    }

    // ⛔ THE `memOrg_` LOOKUP AND ITS `DT_ERROR` ARE THE SAME QUESTION: an lds with no entry for this
    // storage has no allocation to take an address from, which is exactly what the reference stops on.
    let alloc = match (loc.data.my_lds_idx, loc.data.constant_id) {
        (Some(lds), _) => dsc.lds_alloc(lds, loc.storage)?,
        (None, Some(constant)) => dsc.const_alloc(constant, loc.storage)?,
        (None, None) => return Some(None),
    };
    let allocation = inputs.allocs.get(&alloc)?;

    let generic = generic_comp(loc.unit)?;
    let mut scale = dsc.address_scale(generic, loc.storage)?.get();
    if generic == GenericComp::L0lu {
        scale = scale.checked_mul(u64::from(A::PT_ROWS))?;
    }
    let scale = NonZeroU64::new(scale)?;
    let is_start_addr_symbolic = allocation.placement.is_start_addr_symbolic;
    let start_address = if scale.get() == 1 {
        allocation.start_address.clone()
    } else if is_start_addr_symbolic {
        symbols.divide_symbols(&allocation.start_address, scale)
    } else {
        allocation.start_address.divided_by(scale)
    };

    let mut fill = DataInfoFill {
        start_address,
        is_start_addr_symbolic,
        loop_ele_offsets: BTreeMap::new(),
        const_ele_offsets: BTreeMap::new(),
        buffer_switch_position: None,
        buffer_addr_offset: BTreeMap::new(),
    };

    // Constants get an address and nothing else.
    let Some(lds) = loc.data.my_lds_idx else {
        return Some(Some(fill));
    };
    let alloc_owner = tree.alloc_owner_loop(alloc);
    let non_corelet = is_non_corelet_memory(allocation.component);
    let below_chunk = |at: Option<LoopId>| {
        at.is_some_and(|at| inputs.global.loops_below_chunk_boundary.contains(&at))
    };
    let mut both_corelets = non_corelet && !below_chunk(loop_location);
    let cores = dsc.cores_used();
    let corelets = dsc.corelets_used();
    let layout: Vec<PrimaryDim> = allocation.layout.dims().iter().collect();
    let padding = allocation.placement.padding.clone();

    // ── the element offsets, one climb from the node's loop up to the allocation's ──────────────
    let mut walked = loop_location;
    while walked != alloc_owner {
        // ⛔ A CLIMB THAT RAN OUT OF TREE IS THE `DT_ERROR`: the reference tests `prev_ == nullptr`
        // and, at the root, dereferences a null `loopPtr` on the next trip.
        let here = walked?;
        tree.prev(here.0)?;
        if !both_corelets && non_corelet && !below_chunk(Some(here)) {
            both_corelets = true;
        }
        let stages = tree.loop_stages(here);
        for (dim, kind) in tree.loop_dims(here) {
            let mut alloc_padding = padding.get(dim);
            let mut relevant = layout.contains(&dim);
            let mut related_pad_dim = None;
            if !relevant && !tree.is_parametric(here) {
                // Accessing a padded dim through the window dim that walks it — iterating within one
                // window.
                for (pad_dim, pad_info) in dsc.stage_padding_dims(stages.den) {
                    if pad_info.window_dim == dim && padding.get(pad_dim) != PadType::NoPad {
                        relevant = true;
                        alloc_padding = padding.get(pad_dim);
                        related_pad_dim = Some(pad_dim);
                        break;
                    }
                }
            }
            if !relevant || dsc.lds_replication(lds, dim) == ReplicationFactor(0) {
                continue;
            }
            for &corelet in &corelets {
                let previous = if inputs.offsets.is_datastage() {
                    LoopEleOffset(0)
                } else {
                    dsc.temporal_stride(at.node(), alloc, here, dim)?
                };
                let mut stored = previous;
                if inputs.offsets.runs_ladder() {
                    let (offset, iterations) = if tree.is_parametric(here) {
                        (
                            tree.parametric_stride(here),
                            tree.parametric_iter_count(here, corelet, loc.unit),
                        )
                    } else {
                        let view = if both_corelets { None } else { Some(corelet) };
                        let step = NonZeroI64::new(
                            dsc.comp_view(stages.den, dim, loc.unit, view, PadType::NoPad).0,
                        )?;
                        let iterations = IterCount(
                            dsc.comp_view(stages.num, dim, loc.unit, view, PadType::NoPad).0
                                / step.get(),
                        );
                        let offset = padded_loop_offset(
                            dsc,
                            stages.den,
                            dim,
                            kind,
                            loc.unit,
                            view,
                            &padding,
                            alloc_padding,
                            related_pad_dim,
                            step.get(),
                            inputs.unpadded,
                        )?;
                        (LoopEleOffset(i32::try_from(offset).ok()?), iterations)
                    };
                    if inputs.offsets.is_datastage() {
                        stored = offset;
                    }
                    // ⛔ THE MISMATCH `DT_ERROR`: under [`ElemOffsets::Datastage`] the two are the same
                    // write, so this can only bite under [`ElemOffsets::DistributionVerified`].
                    if iterations > IterCount(1) && stored != offset {
                        return None;
                    }
                }
                fill.loop_ele_offsets
                    .entry(corelet)
                    .or_default()
                    .entry(here)
                    .or_default()
                    .insert(dim, stored);
            }
        }
        walked = tree.owner_loop(here.0);
    }

    // ── the constant offsets the padding itself contributes, over the same climb ────────────────
    let mut walked = loop_location;
    while walked != alloc_owner {
        let here = walked?;
        tree.prev(here.0)?;
        let stages = tree.loop_stages(here);
        for (dim, kind) in tree.loop_dims(here) {
            if !layout.contains(&dim) || dsc.lds_replication(lds, dim) == ReplicationFactor(0) {
                continue;
            }
            let alloc_padding = padding.get(dim);
            if !is_zero_padded(alloc_padding) {
                continue;
            }
            let stage = if tree.is_parametric(here) {
                Metadata::CORE_DSTGID
            } else {
                stages.den
            };
            // ⛔ THE TWO `padFront_ < 0` / `padBack_ < 0` `DT_ERROR`S ARE UNSPELLABLE: [`Elements`] is
            // unsigned, so a negative pad is not a value [`PaddingSizes`] can hold.
            let offset = match kind {
                MetaDimKind::PadValid => {
                    // The zero-pad front, which a later stage adds on top of the element offset.
                    let sizes = dsc.stage_padding_sizes(stage, dim)?;
                    ConstEleOffset(i64::try_from(sizes.pad_front.0).ok()?)
                }
                MetaDimKind::PadBack => {
                    // The zero-pad front PLUS the valid span, which is the padded extent less the back.
                    let sizes = dsc.stage_padding_sizes(stage, dim)?;
                    let span = dsc
                        .dim_extent(
                            stage,
                            dim,
                            SenComponent::NoComponent,
                            None,
                            padding.get(dim),
                            Density::FULL,
                        )
                        .0;
                    ConstEleOffset(span - i64::try_from(sizes.pad_back.0).ok()?)
                }
                _ => continue,
            };
            for core in cores.iter() {
                for &corelet in &corelets {
                    fill.const_ele_offsets
                        .entry(core)
                        .or_default()
                        .entry(corelet)
                        .or_default()
                        .insert(dim, offset);
                }
            }
        }
        walked = tree.owner_loop(here.0);
    }

    if !inputs.offsets.is_datastage() {
        coordinate_const_offsets(inputs, &mut fill, at, alloc, coordinates, loc.unit)?;
    }

    if allocation.placement.num_buffers.switches() {
        // ⛔ `DT_CHECK_MSG(loopPtr->getOwnerLoop() != nullptr, "Do not expect the root node.")`, where
        // `loopPtr` has walked all the way up to the allocation's own owner loop.
        let switch_at = alloc_owner?;
        tree.owner_loop(switch_at.0)?;
        fill.buffer_switch_position = Some(switch_at);
        fill.buffer_addr_offset = allocation
            .placement
            .buffer_offset
            .iter()
            .map(|(&core, per_cl)| {
                (
                    core,
                    per_cl
                        .iter()
                        .map(|(&cl, &offset)| (cl, Bytes(offset.0 / scale.get())))
                        .collect(),
                )
            })
            .collect();
    }
    Some(Some(fill))
}

/// `is_any_of(allocPadding, PADDED_WZEROPAD, PADDED_FULLSPAN, PADDED_FULLSPAN_WUNNEEDED)` — the three
/// forms that carry a zero-pad region an index has to be offset past.
const fn is_zero_padded(padding: PadType) -> bool {
    matches!(
        padding,
        PadType::PaddedWZeroPad | PadType::PaddedFullSpan | PadType::PaddedFullSpanWUnneeded
    )
}

/// The `PadType` ladder at `ddc/ddcv1.cpp:2497-2578` — how one trip of this loop moves through an
/// allocation that is padded that way, given the denominator datastage's own step.
///
/// ⛔ `DT_CHECK_MSG(windowDim_ != PrimaryDimTypesCount, "Expect a window-based dimension.")` IS
/// UNSPELLABLE: [`PaddingSizes::window_dim`] is a [`PrimaryDim`], so "no window dim" is not a value it
/// can hold — the reference's sentinel is the absence of the entry, which is already the [`Option`].
#[expect(clippy::too_many_arguments, reason = "the ladder reads eight independent facts")]
fn padded_loop_offset<P: StageSizes + OffsetSizes + ?Sized>(
    dsc: &P,
    stage: DatastageId,
    dim: PrimaryDim,
    kind: MetaDimKind,
    unit: SenComponent,
    view: Option<Corelet>,
    padding: &Padding,
    alloc_padding: PadType,
    related_pad_dim: Option<PrimaryDim>,
    step: i64,
    unpadded: UnpaddedIndexing,
) -> Option<i64> {
    match alloc_padding {
        // Indexing a padded dim from a parametric loop that reaches the valid region only — the step
        // above already IS the offset.
        PadType::PaddedNoZeroPad => {
            if kind == MetaDimKind::PadValid {
                return Some(step);
            }
            if !matches!(kind, MetaDimKind::Unpadded | MetaDimKind::WindowDim)
                || unpadded == UnpaddedIndexing::Forbidden
            {
                return None;
            }
            match dsc.stage_padding_sizes(stage, dim) {
                Some(sizes) => step.checked_mul(i64::try_from(sizes.stride.get()).ok()?),
                None => Some(step),
            }
        }
        _ if is_zero_padded(alloc_padding) => {
            if matches!(kind, MetaDimKind::Padded | MetaDimKind::PadValid) {
                // Indexing the padded dim directly, or its valid part — the zero-pad front is added
                // as a constant offset by the climb above.
                return Some(dsc.comp_view(stage, dim, unit, view, padding.get(dim)).0);
            }
            if let Some(sizes) = dsc.stage_padding_sizes(stage, dim) {
                // Indexing the unpadded dim of a window-based op's result.
                return step.checked_mul(i64::try_from(sizes.stride.get()).ok()?);
            }
            match related_pad_dim {
                // Iterating along a window dim, inside one window.
                Some(related) => {
                    let sizes = dsc.stage_padding_sizes(stage, related)?;
                    step.checked_mul(i64::try_from(sizes.dilation.get()).ok()?)
                }
                None => Some(step),
            }
        }
        PadType::LoweredPadded => {
            if let Some(sizes) = dsc.stage_padding_sizes(stage, dim) {
                if kind != MetaDimKind::Unpadded {
                    return None;
                }
                let window = dsc
                    .dim_extent(
                        stage,
                        sizes.window_dim,
                        SenComponent::NoComponent,
                        None,
                        PadType::NoPad,
                        Density::FULL,
                    )
                    .0;
                return step.checked_mul(window);
            }
            match related_pad_dim {
                Some(related) => {
                    let sizes = dsc.stage_padding_sizes(stage, related)?;
                    step.checked_mul(i64::try_from(sizes.dilation.get()).ok()?)
                }
                None => Some(step),
            }
        }
        // NOPAD: only an unpadded index reaches this dim, and its offset is the step itself.
        _ => Some(step),
    }
}

/// The coordinate-based constant offsets at `ddc/ddcv1.cpp:2588-2664` — how far the node's own fold
/// coordinate sits from the allocation's, in that dim's steps, at each relevant core and corelet.
fn coordinate_const_offsets<P, T, C>(
    inputs: &OffsetInputs<'_, P, T, C>,
    fill: &mut DataInfoFill,
    at: OperandSite,
    alloc: AllocId,
    coordinates: &Coordinate,
    unit: SenComponent,
) -> Option<()>
where
    P: Placement + ?Sized,
    T: ScheduleNodes + ?Sized,
    C: CoordinateOffsets + ?Sized,
{
    let coords = inputs.coords;
    let cores = inputs.dsc.cores_used();
    let corelets = inputs.dsc.corelets_used();
    let head = cores.head();
    let mut any_offset = !fill.const_ele_offsets.is_empty();
    let alloc_coordinates = coords.alloc_coordinate(alloc);
    let relevant = coords.relevant_core_cl(at.node());
    // `senCompToRowId` covers the PT and L0LU row spellings only, and a unit that is not one of them
    // reads row zero.
    let row = comp_row_id(unit).map_or(0, |row| i64::from(row.ordinal()));

    for (dim, node_folds) in coordinates.iter() {
        let Some(alloc_folds) = alloc_coordinates.fold_dim(dim) else {
            continue;
        };
        let folded = |folds: &FoldDim, pos: FoldPosition| {
            folds.cardinality_at(pos).is_some_and(|card| card.0 > 1)
        };
        let use_core_node = folded(node_folds, FoldPosition::Core);
        let use_cl_node = folded(node_folds, FoldPosition::Corelet);
        let row_node = if folded(node_folds, FoldPosition::RowSplit) { row } else { 0 };
        let use_core_alloc = folded(alloc_folds, FoldPosition::Core);
        let use_cl_alloc = folded(alloc_folds, FoldPosition::Corelet);
        let row_alloc = if folded(alloc_folds, FoldPosition::RowSplit) { row } else { 0 };

        // Every temporal fold is pinned to zero: only the element-arrangement folds carry a distance.
        let mut fixed = BTreeMap::from([(FoldPosition::RowSplit as usize, row_alloc)]);
        let spatial = alloc_folds.spatial_folds();
        for axis in spatial..spatial.saturating_add(alloc_folds.temporal_folds()) {
            fixed.entry(axis as usize).or_insert(0);
        }

        for core in cores.iter() {
            for &corelet in &corelets {
                // `operator[]` default-constructs the field before ANY of the guards below reads it,
                // so a skipped site still holds an explicit zero.
                fill.const_ele_offsets
                    .entry(core)
                    .or_default()
                    .entry(corelet)
                    .or_default()
                    .entry(dim)
                    .or_insert(ConstEleOffset::ZERO);
                if !relevant.0.get(&core).is_some_and(|cls| cls.contains(&corelet)) {
                    continue;
                }
                let Some(node_slices) = coords.node_work_slices(at, core) else {
                    continue;
                };
                let Some(alloc_slices) = coords.alloc_work_slices(alloc, core) else {
                    continue;
                };
                // An axis neither side folds is the same offset everywhere along it, so the proxy
                // site's answer is copied rather than solved again.
                if !use_core_node && !use_core_alloc && core != head {
                    let proxy = offset_at(fill, head, corelet, dim);
                    fill.const_ele_offsets
                        .get_mut(&core)?
                        .get_mut(&corelet)?
                        .insert(dim, proxy);
                    continue;
                }
                if !use_cl_node && !use_cl_alloc && corelet != Corelet::at::<0>() {
                    let proxy = offset_at(fill, core, Corelet::at::<0>(), dim);
                    fill.const_ele_offsets
                        .get_mut(&core)?
                        .get_mut(&corelet)?
                        .insert(dim, proxy);
                    continue;
                }
                let slice_node = node_slices.get(&dim)?;
                let beta = coords.single_beta(
                    node_folds,
                    if use_core_node { slice_node.0 } else { 0 },
                    if use_cl_node { i64::from(corelet.get()) } else { 0 },
                    row_node,
                );
                let slice_alloc = alloc_slices.get(&dim)?;
                fixed.insert(
                    FoldPosition::Core as usize,
                    if use_core_alloc { slice_alloc.0 } else { 0 },
                );
                fixed.insert(
                    FoldPosition::Corelet as usize,
                    if use_cl_alloc { i64::from(corelet.get()) } else { 0 },
                );
                let mine = coords.distance_in_steps(alloc_folds, beta, &fixed);
                if mine != ConstEleOffset::ZERO {
                    any_offset = true;
                    // ⛔ `DT_CHECK_MSG(.., "Constant offset from coordinates in conflict with existing
                    // constant offset")` — the padding climb above may already have written here.
                    let existing = offset_at(fill, core, corelet, dim);
                    if existing != ConstEleOffset::ZERO && existing != mine {
                        return None;
                    }
                    fill.const_ele_offsets
                        .get_mut(&core)?
                        .get_mut(&corelet)?
                        .insert(dim, mine);
                }
            }
        }
    }
    // An offset that was zero everywhere is no offset at all.
    if !any_offset {
        fill.const_ele_offsets.clear();
    }
    Some(())
}

/// `di.constEleOffsets_[core][corelet][dim]` READ — zero where nothing was written, which is what the
/// reference's `operator[]` default-constructs.
fn offset_at(
    fill: &DataInfoFill,
    core: Core,
    corelet: Corelet,
    dim: PrimaryDim,
) -> ConstEleOffset {
    fill.const_ele_offsets
        .get(&core)
        .and_then(|per_cl| per_cl.get(&corelet))
        .and_then(|per_dim| per_dim.get(&dim))
        .copied()
        .unwrap_or(ConstEleOffset::ZERO)
}

/// The five `metadata.datatransfers_` fixups at `ddc/ddcv1.cpp:2884-3005`, which lay a constant offset
/// over an already-filled transfer operand.
///
/// ⚠️ AN `else if` CHAIN IN THE REFERENCE, SO AT MOST ONE OF THE FIVE RUNS.
fn transfer_metadata_offsets<P, T, C, K>(
    inputs: &OffsetInputs<'_, P, T, C>,
    sink: &mut K,
    node: NodeId,
    transfer: &TransferNode,
    cores: &CoresUsed,
    corelets: &[Corelet],
) -> Option<()>
where
    P: OffsetSizes + ?Sized,
    T: ScheduleNodes + ?Sized,
    C: ?Sized,
    K: DataInfoSink + ?Sized,
{
    let dsc = inputs.dsc;
    let metadata = inputs.metadata;
    let Some(meta) = metadata.datatransfers.get(&node) else {
        return Some(());
    };
    let src_site = OperandSite::TransferSrc(node);
    let first_dst = OperandSite::TransferDst(node, DestIdx(0));
    let src_empty = sink.const_ele_offsets_empty(src_site)?;
    let dst_empty = sink.const_ele_offsets_empty(first_dst)?;
    let sites = |index: usize| -> Option<OperandSite> {
        Some(OperandSite::TransferDst(
            node,
            DestIdx(u32::try_from(index).ok()?),
        ))
    };

    if meta.apply_row_offset_src && is_dsc_memory(transfer.src.storage) && src_empty {
        // ⛔ [`None`] WHERE THERE IS NO ROW SPLIT DIM: the reference keys the offset by
        // `metadata.rowSplitDim`, and an unset one names no dim to offset along.
        let dim = metadata.row_split_dim?;
        let rows = i64::from(comp_row_id(transfer.dsts.first().unit)?.ordinal());
        for core in cores.iter() {
            for &corelet in corelets {
                let size = dsc.block_transfer_size(node, transfer.src.unit, corelet, dim);
                let offset = ConstEleOffset(i64::try_from(size.0).ok()?.checked_mul(rows)?);
                sink.set_const_ele_offset(src_site, core, corelet, dim, offset)?;
            }
        }
    } else if meta.apply_row_offset_dst && dst_empty {
        let dim = metadata.row_split_dim?;
        let rows = i64::from(comp_row_id(transfer.src.unit)?.ordinal());
        for (index, dst) in transfer.dsts.iter().enumerate() {
            if !is_dsc_memory(dst.storage) {
                continue;
            }
            let site = sites(index)?;
            for core in cores.iter() {
                for &corelet in corelets {
                    let size = dsc.block_transfer_size(node, dst.unit, corelet, dim);
                    let offset = ConstEleOffset(i64::try_from(size.0).ok()?.checked_mul(rows)?);
                    sink.set_const_ele_offset(site, core, corelet, dim, offset)?;
                }
            }
        }
    } else if meta.replicated {
        // ⛔ `DT_CHECK_MSG(offset_src_ > 0 || offset_dest_.size() > 0, "At least one of the src or dst
        // should be replicated")`.
        if meta.offset_src == Elements(0) && meta.offset_dest.is_empty() {
            return None;
        }
        // ⛔ `throw`: a replicated transfer that is not in the cloning map has no original to take its
        // block size from.
        let original = cloned_from(metadata, node)?;
        let src_unit = inputs.tree.transfer(original)?.src.unit;
        let dims: Vec<PrimaryDim> = transfer
            .unit_time_transfer_chunk_size
            .iter()
            .map(|chunk| chunk.size_dim.dim)
            .collect();
        if meta.offset_src > Elements(0) {
            for core in cores.iter() {
                for &corelet in corelets {
                    for &dim in &dims {
                        let size = dsc.block_transfer_size(original, src_unit, corelet, dim);
                        let offset = ConstEleOffset(i64::try_from(size.0).ok()?);
                        sink.set_const_ele_offset(src_site, core, corelet, dim, offset)?;
                    }
                }
            }
        }
        for (&dst, &offset) in &meta.offset_dest {
            if offset == Elements(0) {
                continue;
            }
            let site = OperandSite::TransferDst(node, dst);
            for core in cores.iter() {
                for &corelet in corelets {
                    for &dim in &dims {
                        let size = dsc.block_transfer_size(original, src_unit, corelet, dim);
                        let value = ConstEleOffset(i64::try_from(size.0).ok()?);
                        sink.set_const_ele_offset(site, core, corelet, dim, value)?;
                    }
                }
            }
        }
    } else if meta.apply_pe_sfp_split_offset_src && src_empty {
        // The clone's offset is the ORIGINAL node's transfer size.
        let original = sole_cloned_from(metadata, node)?;
        let src_unit = inputs.tree.transfer(original)?.src.unit;
        for core in cores.iter() {
            for &corelet in corelets {
                for dim in dsc.pe_sfp_split_dims(Metadata::CORE_DSTGID) {
                    let size = dsc.block_transfer_size(original, src_unit, corelet, dim);
                    let offset = ConstEleOffset(i64::try_from(size.0).ok()?);
                    sink.set_const_ele_offset(src_site, core, corelet, dim, offset)?;
                }
            }
        }
    } else if !meta.apply_pe_sfp_split_offset_dest.is_empty() && dst_empty {
        let original = sole_cloned_from(metadata, node)?;
        let src_unit = inputs.tree.transfer(original)?.src.unit;
        for &dst in &meta.apply_pe_sfp_split_offset_dest {
            let site = OperandSite::TransferDst(node, dst);
            for core in cores.iter() {
                for &corelet in corelets {
                    for dim in dsc.pe_sfp_split_dims(Metadata::CORE_DSTGID) {
                        let size = dsc.block_transfer_size(original, src_unit, corelet, dim);
                        let offset = ConstEleOffset(i64::try_from(size.0).ok()?);
                        sink.set_const_ele_offset(site, core, corelet, dim, offset)?;
                    }
                }
            }
        }
    }
    Some(())
}

/// The cloned-compute repetition offsets at `ddc/ddcv1.cpp:3027-3054` — each clone's output is pushed
/// one further along the allocation's outermost dim, the FIRST clone furthest.
fn clone_repetition_offsets<P, T, C, K>(
    inputs: &OffsetInputs<'_, P, T, C>,
    sink: &mut K,
    node: NodeId,
    compute: &ComputeNode,
    cores: &CoresUsed,
    corelets: &[Corelet],
) -> Option<()>
where
    P: OffsetSizes + LdsSticks + ?Sized,
    T: ScheduleNodes + ?Sized,
    C: ?Sized,
    K: DataInfoSink + ?Sized,
{
    let dsc = inputs.dsc;
    let clones = inputs.metadata.node_cloning_map.get(&node)?;
    for index in 0..inputs.tree.repetition_with_offset_outputs(node) {
        let output = compute.outputs.get(index)?;
        let lds = output.data.my_lds_idx?;
        // A store unit writes the memory behind it, and that is where the allocation lives.
        let comp = if output.storage == SenComponent::Lxsu {
            SenComponent::Lx
        } else {
            output.storage
        };
        let alloc = dsc.lds_alloc(lds, comp)?;
        let dim = inputs.allocs.get(&alloc)?.layout.outermost_dim();
        let sticks = cumulative_stick_sizes(&dsc.stick_dims(lds), StickPart::Whole)?;
        let size = i64::try_from(cumulative_stick_size(&sticks, dim)?.get()).ok()?;
        let mut factor = i64::try_from(clones.len()).ok()?;
        for &clone in clones {
            let site = OperandSite::ComputeOutput(clone, OutputIdx(index));
            let offset = ConstEleOffset(factor.checked_mul(size)?);
            for core in cores.iter() {
                for &corelet in corelets {
                    sink.set_const_ele_offset(site, core, corelet, dim, offset)?;
                }
            }
            factor -= 1;
        }
    }
    Some(())
}

/// `findLastFusableLoop(unit)` (`ddc/ddcv1.cpp:2818-2857`) — the outermost enclosing loop this unit
/// can still see a single child through.
///
/// ⭐ [`None`] IS `nullptr` AND NOT AN ABORT: a unit the transfer is not relevant to, or one whose
/// first parent already forks, genuinely has no fusable loop.
fn last_fusable_loop<P, T>(dsc: &P, tree: &T, node: NodeId, unit: SenComponent) -> Option<LoopId>
where
    P: OffsetSizes + ?Sized,
    T: ScheduleNodes + ?Sized,
{
    if !tree.is_relevant(node, unit) {
        return None;
    }
    let mut last = None;
    let mut parent = tree.prev(node);
    while let Some(here) = parent {
        if tree.prev(here).is_none() || tree.next_view_len(here, unit) != 1 {
            break;
        }
        if tree.kind(here) == Some(NodeKind::Condition) {
            break;
        }
        if let Some(at) = tree.as_loop(here) {
            if is_symbolic_loop(dsc, tree, at) {
                break;
            }
            last = Some(at);
        }
        parent = tree.prev(here);
    }
    last
}

/// `isSymbolicLoop(loop)` (`ddc/ddcv1.cpp:2828-2846`) — a loop over a dim the numerator datastage
/// makes symbolic and the denominator does not needs a correction, and so cannot be fused.
fn is_symbolic_loop<P: OffsetSizes + ?Sized, T: ScheduleNodes + ?Sized>(
    dsc: &P,
    tree: &T,
    at: LoopId,
) -> bool {
    if tree.is_parametric(at) {
        // Parametric loops cannot yet be symbolic.
        return false;
    }
    let stages = tree.loop_stages(at);
    tree.loop_dims(at).into_iter().any(|(dim, _)| {
        dsc.has_symbolic_dim(stages.num, dim) && !dsc.has_symbolic_dim(stages.den, dim)
    })
}

/// `nodeCloningMap_` SEARCHED BY CLONE — the original whose clone list holds this node.
fn cloned_from(metadata: &Metadata, clone: NodeId) -> Option<NodeId> {
    metadata
        .node_cloning_map
        .iter()
        .find(|(_, clones)| clones.contains(&clone))
        .map(|(&original, _)| original)
}

/// The same search under `DT_CHECK(clones.size() == 1)` — the two PE/SFP-split arms require every
/// entry in the map to name exactly one clone, and stop on the first that does not.
fn sole_cloned_from(metadata: &Metadata, clone: NodeId) -> Option<NodeId> {
    let mut original = None;
    for (&node, clones) in &metadata.node_cloning_map {
        if clones.len() != 1 {
            return None;
        }
        if clones.first() == Some(&clone) {
            original = Some(node);
        }
    }
    original
}

/// `insertReg(name, startAddress, regMap, unrollIdx)` (`ddc/ddcv1.cpp:3349-3357`).
///
/// ⭐ [`RegName::at_slice`] IS THE `name.find("_unroll")` SPLIT, generated from the same census that
/// produced both the template's names and the body's, so the two cannot drift.
fn insert_reg(
    regs: &mut BTreeMap<RegName, RegSlot>,
    name: RegName,
    start: u64,
    unroll: Unroll,
    unroll_idx: bool,
) -> Option<()> {
    if name.at_slice(0).is_none() {
        regs.insert(name, RegSlot(start));
        return Some(());
    }
    for i in 0..unroll.0 {
        let step = u64::from(if unroll_idx { i } else { 0 });
        regs.insert(name.at_slice(i)?, RegSlot(start.checked_add(step)?));
    }
    Some(())
}

/// `getBufferCapacityForNode(node, node->ldsIdx_, node->component_, 0, 0) / bytesPerStick`, and ONE
/// for the reference's `ldsIdx_ < 0` (`ddc/ddcv1.cpp:3336-3341`, `:3565-3571`).
fn reg_unroll_factor<A: Arch, P: Placement + ?Sized>(
    dsc: &P,
    node: &AllocateNode,
    alloc: AllocId,
) -> Option<u32> {
    let Some(lds) = node.lds else {
        return Some(1);
    };
    let at = Some((Corelet::at::<0>(), Row::at::<0>()));
    u32::try_from(dsc.buffer_capacity(alloc, Some(lds), at).0 / A::BYTES_PER_STICK).ok()
}

/// Replaces: e261_finalizeOps
///
/// Binds every opaque compute's register names to the start STICK of the allocation behind them and
/// fills its `unroll`/`prec`, then points each implicit sync at the transfer it stands in for.
///
/// ⛔ [`None`] IS EVERY `DT_ERROR`: an unroll factor that is zero or not a power of two, an opaque op
/// placed at differing addresses per core/corelet, a register alloc whose own unroll is neither 1 nor
/// the op's, two implicit syncs on one memory, an implicit sync outside `L0`, an arch before `RCUDD1A`
/// (⚠️ UNSPELLABLE — [`IsaGen`] starts there), and two transfers into one L0 storage.
pub fn finalize_ops<A: Arch, P, T>(
    dsc: &P,
    tree: &T,
    metadata: &Metadata,
    allocs: &AllocArena,
    computes: &mut ComputeArena,
    syncs: &mut BTreeMap<NodeId, SyncNode>,
) -> Option<()>
where
    P: Placement + ?Sized,
    T: ScheduleWalk + ?Sized,
{
    for (compute, opaque) in &metadata.opaque_ops {
        let unroll = match opaque.internal_reg_alloc {
            Some(alloc) => reg_unroll_factor::<A, P>(dsc, allocs.get(&alloc)?, alloc)?,
            None => 1,
        };
        if !unroll.is_power_of_two() {
            return None;
        }
        let unroll = Unroll(unroll);
        let node = computes.get_mut(compute)?;
        node.instr_attribute.unroll = unroll;
        if !opaque.internal_regs.is_empty() {
            let alloc = opaque.internal_reg_alloc?;
            let start = allocs.get(&alloc)?.start_address.uniform()?.0 / A::BYTES_PER_STICK;
            for reg in &opaque.internal_regs {
                // `read_write_reg_map_.size()` is re-read per register, so each one starts past the last.
                let taken = u64::try_from(node.instr_attribute.read_write_regs.len()).ok()?;
                let at = start.checked_add(taken)?;
                insert_reg(
                    &mut node.instr_attribute.read_write_regs,
                    reg.name,
                    at,
                    unroll,
                    true,
                )?;
            }
        }
        for (&reg, &alloc) in &opaque.in_out_reg_allocs {
            let reg_alloc = allocs.get(&alloc)?;
            let reg_unroll = reg_unroll_factor::<A, P>(dsc, reg_alloc, alloc)?;
            if reg_unroll != unroll.0 && reg_unroll != 1 {
                return None;
            }
            let start = reg_alloc.start_address.uniform()?.0 / A::BYTES_PER_STICK;
            insert_reg(
                &mut node.instr_attribute.read_only_regs,
                reg,
                start,
                unroll,
                reg_unroll > 1,
            )?;
        }
        let fp32 = node.data_format == Some(DataFormat::IeeeFp32);
        node.instr_attribute.precision = Some(if fp32 {
            Precision::Fp32
        } else {
            Precision::Fp16
        });
    }

    let mut synced: BTreeSet<SenComponent> = BTreeSet::new();
    for (sync, &alloc) in &metadata.implicit_syncs {
        let memory = allocs.get(&alloc)?.component;
        if !synced.insert(memory) || memory != SenComponent::L0 {
            return None;
        }
        let dest = SenComponent::L0su;
        if A::GEN < IsaGen::Rcudd1a {
            return None;
        }
        let transfers =
            tree.nodes_of_kind_under(tree.prev_of_alloc(alloc), NodeKind::Transfer, dest);
        let mut reference = *transfers.first()?;
        if transfers.len() != 1 {
            // An Mx-matmul needs two LXLU->L0SU transfers, one per L0 storage; the `L0` one is the ref.
            let mut storages: BTreeSet<SenComponent> = BTreeSet::new();
            for &transfer in &transfers {
                let storage = tree.transfer(transfer)?.dsts.first().storage;
                if !storages.insert(storage) {
                    return None;
                }
                if storage == SenComponent::L0 {
                    reference = transfer;
                }
            }
        }
        syncs.get_mut(sync)?.implicit_sync_ref_transfer = Some(reference);
    }
    set_pe_folds_if_pt_interaction::<A>(computes)
}

/// ONE COORDINATE-MASKING RUN — `coordinateMasking_.at(dim).at(i)`, whose own comment names the pair
/// `pair<unmasked, masked>` (`dsc/designSpaceConfig.h:100`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskRun {
    /// `first` — the elements this run leaves alone.
    pub unmasked: Elements,
    /// `second` — the elements it masks off the end.
    pub masked: Elements,
}

/// WHAT ENTRY 262 ASKS THE DSC — the masking request and the facts about the data it masks, all
/// `dsc/designSpaceConfig.h`.
pub trait Masking {
    /// `coordinateMasking_` (`:100`).
    fn coordinate_masking(&self) -> BTreeMap<PrimaryDim, Vec<MaskRun>>;
    /// `computeOp_.at(0).opConsts.count("samv-wsllen")` — the op const whose presence means an
    /// upstream tool owed us a masking request.
    fn declares_samv_wsllen(&self) -> bool;
    /// `constantInfo_.count(maskingConstId_)` as the constant it then names (`:101`).
    fn masking_constant(&self) -> Option<ConstIdx>;
    /// `sdsc_->numWkSlicesPerDim_.at(dim) > 1` (`dsc/superdsc.h:69`).
    fn splits_across_cores(&self, dim: PrimaryDim) -> bool;
    /// `dimToSymbolMapping_.count(dim)` (`:77`).
    fn is_symbolic(&self, dim: PrimaryDim) -> bool;
    /// `getNonBroadcastLdsDimSet(lds)` (`:253`).
    fn non_broadcast_dims(&self, lds: LdsIdx) -> BTreeSet<PrimaryDim>;
    /// `labeledDs_.at(lds).dataFormat_`.
    fn lds_format(&self, lds: LdsIdx) -> Option<DataFormat>;
}

/// HOW ENTRY 262 SPLICES ITS GUARD IN — `addChildNode(cond, true, before)` (`dsc/dsc2.h:688-717`).
pub trait MaskInsertion {
    /// `before->prev_->addChildNode(cond, true, before)` — the condition takes `before`'s place among
    /// its parent's children, with `before` following it, and [`None`] where no parent holds it.
    fn insert_condition_before(&mut self, before: NodeId, cond: ConditionNode) -> Option<()>;
}

/// Replaces: e262_coordinateMasking
///
/// Mints the SAMV stick-mask node for a masked tail, guards it with a condition on the last iteration
/// of every loop over the masked dim, and splices both — plus a SAMV reset in the else region — in
/// above the highest transfer that is not already inside a loop over that dim.
///
/// ⛔ [`None`] IS EVERY `DT_ERROR`: a masking request the upstream tool owed but did not fill, a run
/// that is not at the end, a dim split over corelets or cores or symbolic, a missing masking constant,
/// a constant transfer out of LXLU, disagreeing stick layouts, a transfer masking does not affect, a
/// stick SAMV cannot reach, masking past one stick, two insertion points, and two masked dims.
pub fn coordinate_masking<A: Arch, P, T>(
    dsc: &P,
    metadata: &Metadata,
    tree: &mut T,
) -> Option<()>
where
    P: Masking + LdsSticks + ?Sized,
    T: ScheduleWalk + MaskInsertion + ?Sized,
{
    let masking = dsc.coordinate_masking();
    if masking.is_empty() {
        // The op const says upstream should have filled `coordinateMasking_`.
        return if dsc.declares_samv_wsllen() {
            None
        } else {
            Some(())
        };
    }
    let mut any_dim_valid = false;
    for runs in masking.values() {
        if runs.len() != 1 {
            return None;
        }
        if runs.first()?.masked > Elements(0) {
            any_dim_valid = true;
        }
    }
    // No dim actually masks anything, so no stickMask node is constructed.
    if !any_dim_valid {
        return Some(());
    }
    for &dim in masking.keys() {
        if metadata.cl_split_dims.contains(&dim)
            || dsc.splits_across_cores(dim)
            || dsc.is_symbolic(dim)
        {
            return None;
        }
    }
    let mut mask = StickMaskNode {
        name: NodeName("SAMV".to_owned()),
        mask_val_const_id: Some(dsc.masking_constant()?),
        data_format: None,
        stick_layout: Vec::new(),
        first_stick_coord_to_mask_per_dim: BTreeMap::new(),
        affected_transfers: Vec::new(),
    };
    for node in tree.nodes_of_kind(NodeKind::Transfer) {
        let transfer = tree.transfer(node)?;
        if transfer.src.unit != SenComponent::Lxlu {
            continue;
        }
        // A constant transfer out of LXLU would be one masking cannot be applied to.
        let lds = transfer.src.data.my_lds_idx?;
        let dims = dsc.stick_dims(lds);
        let stick: Vec<Size> = stick_sizes(&dims, StickPart::Whole)
            .into_iter()
            .map(|(dim, size)| Size { dim, size })
            .collect();
        let no_bcast = dsc.non_broadcast_dims(lds);
        if !stick.iter().any(|size| no_bcast.contains(&size.dim)) {
            continue;
        }
        if !mask.stick_layout.is_empty() {
            if mask.stick_layout != stick {
                return None;
            }
            mask.affected_transfers.push(node);
            continue;
        }
        mask.stick_layout = stick.clone();
        if !stick.iter().any(|size| masking.contains_key(&size.dim)) {
            return None;
        }
        mask.affected_transfers.push(node);
        mask.data_format = dsc.lds_format(lds);
        let slice = SliceElems::per_stick::<A>(&dims)?;
        let cross = stick_sizes(&dims, StickPart::CrossSlice(slice));
        let within = stick_sizes(&dims, StickPart::WithinSlice(slice));
        let crossing = cross.first()?.0;
        if cross.len() != 1
            || within.len() > 2
            || (crossing != within.first()?.0
                && (within.len() == 1 || crossing != within.get(1)?.0))
        {
            return None;
        }
        for (dim, size) in cumulative_stick_sizes(&dims, StickPart::Whole)? {
            let Some(runs) = masking.get(&dim) else {
                continue;
            };
            let masked = runs.first()?.masked;
            if masked > size {
                return None;
            }
            mask.first_stick_coord_to_mask_per_dim
                .entry(dim)
                .or_insert(Elements(size.0 - masked.0));
        }
    }

    let mut insertion_points: BTreeSet<NodeId> = BTreeSet::new();
    for &transfer in &mask.affected_transfers {
        // The highest ancestor still outside every loop over a masked dim.
        let mut last_seen = transfer;
        while let Some(parent) = tree.prev(last_seen) {
            if tree.prev(parent).is_none() {
                break;
            }
            if let Some(at) = tree.as_loop(parent) {
                if tree.loop_dims(at).iter().any(|(dim, _)| {
                    mask.first_stick_coord_to_mask_per_dim.contains_key(dim)
                }) {
                    break;
                }
            }
            last_seen = parent;
        }
        if !insertion_points.insert(tree.prev(last_seen)?) {
            continue;
        }
        if insertion_points.len() != 1 || mask.first_stick_coord_to_mask_per_dim.len() != 1 {
            return None;
        }
        let (&dim, _) = mask.first_stick_coord_to_mask_per_dim.first_key_value()?;
        let mut ands: Vec<LoopCond> = Vec::new();
        let mut climb = tree.owner_loop(last_seen);
        while let Some(at) = climb {
            if tree.prev(at.0).is_none() {
                break;
            }
            if tree.loop_dims(at).iter().any(|(loop_dim, _)| *loop_dim == dim) {
                ands.push(LoopCond {
                    loop_comp: at,
                    dim,
                    op: CondOp::Eq,
                    bound: LoopBound::Last,
                });
            }
            climb = tree.owner_loop(at.0);
        }
        let spelling = dim.spelling();
        let reset = StickMaskNode {
            name: NodeName("SAMV_reset".to_owned()),
            first_stick_coord_to_mask_per_dim: BTreeMap::new(),
            ..mask.clone()
        };
        tree.insert_condition_before(
            last_seen,
            ConditionNode {
                name: NodeName(format!("condition_SAMV_dim_{spelling}")),
                loop_cond: LoopCondComposite {
                    two_level_or_of_ands: vec![ands],
                    negated: false,
                },
                core_cl_cond: BTreeMap::new(),
                then_region: vec![SchedNode::Block(BlockNode {
                    name: NodeName(format!("block_SAMV_dim_{spelling}")),
                    children: vec![SchedNode::StickMask(Box::new(mask.clone()))],
                })],
                else_region: vec![SchedNode::Block(BlockNode {
                    name: NodeName("block_SAMV_reset".to_owned()),
                    children: vec![SchedNode::StickMask(Box::new(reset))],
                })],
            },
        )?;
    }
    Some(())
}

/// THE SCHEDULE TREE AS ENTRIES 258-263 WALK IT — `traverseTreeDFS`, `getPrev` and `getOwnerLoop`
/// (`dsc/dsc2.cpp:2222`, `dsc/dsc2.h:474-500`), all outside this campaign's file list.
pub trait ScheduleWalk {
    /// `traverseTreeDFS(from, {LOOP})` — every loop below a block, in DFS order.
    fn loops_under(&self, from: BlockId) -> Vec<LoopId>;
    /// `traverseTreeDFS(nullptr, {kind})` — every node of one kind in the whole tree.
    fn nodes_of_kind(&self, kind: NodeKind) -> Vec<NodeId>;
    /// `traverseTreeDFSMutable(nullptr, {ALLOCATE})` — the same for ALLOCATE nodes, whose identity is
    /// the [`AllocId`] the metadata already holds.
    fn allocates(&self) -> Vec<AllocId>;
    /// `traverseTreeDFS(from, {kind}, unit)` — the same below one node, filtered by component, where
    /// [`None`] is the reference's `nullptr` start and so the whole tree.
    fn nodes_of_kind_under(
        &self,
        from: Option<NodeId>,
        kind: NodeKind,
        unit: SenComponent,
    ) -> Vec<NodeId>;
    /// `allocNode->getPrev()` — the block, loop or region CONTAINING the ALLOCATE, which is the
    /// subtree entry 261 searches, and [`None`] for an allocation this tree does not hold.
    fn prev_of_alloc(&self, alloc: AllocId) -> Option<NodeId>;
    /// `src_`/`dstVias_` as the node's own pairing, and [`None`] for a node that is not a TRANSFER.
    fn transfer(&self, node: NodeId) -> Option<TransferNode>;
    /// `nodeType_ == LOOP` as the loop it then is.
    fn as_loop(&self, node: NodeId) -> Option<LoopId>;
    /// `getPrev()` — ⚠️ THE PARENT, NOT THE PRECEDING SIBLING: `prev_` is a `BlockNode*` up the tree
    /// (`dsc/dsc2.h:463,515`), absent only at the root.
    fn prev(&self, node: NodeId) -> Option<NodeId>;
    /// `getOwnerLoop()` — the innermost LOOP enclosing this node.
    fn owner_loop(&self, node: NodeId) -> Option<LoopId>;
    /// `LoopNode::dims_` (`dsc/dsc2.h:570`).
    fn loop_dims(&self, at: LoopId) -> Vec<(PrimaryDim, MetaDimKind)>;
}

/// Replaces: e263_identifyBelowChunkBoundaryLoops
///
/// Refills `loopsBelowChunkBoundary` with every LOOP under the below-LX insertion block
/// (`ddc/ddcv1.cpp:3683-3691`).
///
/// ⛔ [`None`] IS THE `DT_CHECK(metadata.belowLxScheduleInsertBlock)`: with no insertion block there
/// is no chunk boundary to be below, and the reference stops rather than answering "none".
/// ⭐ THE SET IS CLEARED FIRST, so a second call does not accumulate.
pub fn identify_below_chunk_boundary_loops<T: ScheduleWalk + ?Sized>(
    tree: &T,
    metadata: &Metadata,
    global: &mut GlobalData,
) -> Option<()> {
    let block = metadata.below_lx_schedule_insert_block?;
    global.loops_below_chunk_boundary = tree.loops_under(block).into_iter().collect();
    Some(())
}

// crustify:todo: e307_exploreAssignDataStages
//   authority : ddc/ddcv1.cpp:555  (1126 body lines, level 2)
//   class     : Ddc
//   original  : void Ddc::exploreAssignDataStages()
//   extract   : crustify-ddc/cpp/ddc.cpp:10042-11168
//   calls     : e096_updateMin, e097_updateMax, e098_updateValues, e104_clear, e258_allocAllMem
//   ⛔ NOTE   : REUSE, DO NOT RE-IMPLEMENT: this body CONTAINS the `checkConstraints` lambda at
//               ddc/ddcv1.cpp:792 and its inner `checkConstraintsImpl` at :825, both ALREADY PORTED on
//               bridge1-campaign as `e001_checkConstraints` in
//               crates/compiler/deeptools/src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs:439
//               (the impl at that file's :518). CALL THOSE. A second copy of the constraint check is the
//               diverging-implementation failure this campaign's exclusion list exists to prevent.

// crustify:todo: e308_prepDsc
//   authority : ddc/ddcv1.cpp:2019  (252 body lines, level 2)
//   class     : Ddc
//   original  : void Ddc::prepDsc()
//   extract   : crustify-ddc/cpp/ddc.cpp:11178-11430
//   calls     : e104_clear, e130_getPeSfpSplitDim, e257_addNewLds

// crustify:todo: e309_attachToPrefilledSchedule
//   authority : ddc/ddcv1.cpp:2280  (71 body lines, level 2)
//   class     : Ddc
//   original  : void Ddc::attachToPrefilledSchedule()
//   extract   : crustify-ddc/cpp/ddc.cpp:11440-11511
//   calls     : e259_calculateClStartAddress

// crustify:todo: e379_run_v1
//   authority : ddc/ddcv1.cpp:3692  (109 body lines, level 7)
//   class     : Ddc
//   original  : bool Ddc::run_v1(SuperDsc& sdsc)
//   extract   : crustify-ddc/cpp/ddc.cpp:15148-15257
//   calls     : e104_clear, e125_minimizeAllocations, e126_populateUnitTimeTransfers, e127_spreadDataInAllocate, e128_finalizeAllocateLayouts, e132_restoreDsc, e133_adjustLoopOffsetsAndAddresses, e134_simplifyScheduleTree, e135_updateLdsIdxMetadata, e136_initGlobalData, e244_cloneForOffsetAdjustment, e245_setSizeForFixedSizeTransfers, e246_transformForInterSliceRestickify, e260_fillLoopOffsetsAndAddresses …

// crustify:todo: e381_run
//   authority : ddc/ddcv1.cpp:3802  (15 body lines, level 8)
//   class     : Ddc
//   original  : void Ddc::run(SuperDsc& sdsc)
//   extract   : crustify-ddc/cpp/ddc.cpp:15267-15282
//   calls     : e379_run_v1

#[cfg(test)]
mod tests_e132_e136 {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::schedule::dsc2::{
        DataInfo, Dsts, InstrAttribute, NodeName, Operand, ReplicationFactor, TransferPadding,
    };
    use crate::units::NumFolds;

    /// The one corelet every build has, which is all these fixtures need.
    fn corelet0() -> Corelet {
        Corelet::checked(0).expect("every core has a corelet 0")
    }

    fn operand(unit: SenComponent, lds: Option<u32>) -> Operand {
        Operand {
            unit,
            storage: SenComponent::NoComponent,
            data: DataInfo {
                data_connect: None,
                my_lds_idx: lds.map(LdsIdx),
                constant_id: None,
            },
        }
    }

    /// `computeOp_` with one entry.
    struct Ops(Option<OpFunc>);

    impl ComputeOps for Ops {
        fn op_funcs(&self) -> OpFuncs {
            OpFuncs::new(self.0, Vec::new())
        }
        fn set_first_op_func(&mut self, op_func: OpFunc) {
            self.0 = Some(op_func);
        }
    }

    #[test]
    fn the_backup_is_restored_onto_the_first_op_and_only_when_one_was_taken() {
        let mut metadata = Metadata::default();
        let mut dsc = Ops(Some(OpFunc::Exx2Zeromean));
        // `OpFuncs::NONE` — `prepDsc` never swapped, so nothing is put back.
        restore_dsc(&metadata, &mut dsc);
        assert_eq!(dsc.0, Some(OpFunc::Exx2Zeromean));
        metadata.op_func_backup = Some(OpFunc::Exx2);
        restore_dsc(&metadata, &mut dsc);
        assert_eq!(dsc.0, Some(OpFunc::Exx2));
    }

    /// The inner loop around the restickify transfer, and the outer loop around that.
    const INNER: LoopId = LoopId(NodeId(10));
    const OUTER: LoopId = LoopId(NodeId(11));
    /// The `fmul`'s own loop.
    const FMUL_LOOP: LoopId = LoopId(NodeId(12));

    /// One offset write, as entry 133 made it.
    #[derive(Debug, PartialEq, Eq)]
    enum Write {
        Src(NodeId, LoopId, PrimaryDim, LoopEleOffset),
        Input(NodeId, InputIdx, Option<LoopId>, PrimaryDim, LoopEleOffset),
    }

    /// An LXLU-sourced transfer stating one stick dim, an `L0LU`→`PT` transfer under two loops, and
    /// an LXLU `fmul` reading (`LXLUSCALEREG`, `LATCH`).
    #[derive(Default)]
    struct Offsets(Vec<Write>);

    impl ComputeOps for Offsets {
        fn op_funcs(&self) -> OpFuncs {
            OpFuncs::new(Some(OpFunc::ReStickifyOpLx), Vec::new())
        }
        fn set_first_op_func(&mut self, _op_func: OpFunc) {}
    }

    impl LoopOffsets for Offsets {
        fn corelets(&self) -> Vec<Corelet> {
            vec![corelet0()]
        }
        fn transfers(&self) -> Vec<NodeId> {
            vec![NodeId(0), NodeId(1)]
        }
        fn transfer(&self, node: NodeId) -> TransferNode {
            let (src, dst) = if node == NodeId(0) {
                (SenComponent::Lxlu, SenComponent::Lx)
            } else {
                (SenComponent::L0lu, SenComponent::Ptrow3)
            };
            TransferNode {
                padding: TransferPadding::default(),
                src_indirect: None,
                dst_indirect: None,
                name: NodeName("t".to_owned()),
                src: operand(src, Some(0)),
                dsts: Dsts::new(operand(dst, Some(0)), Vec::new()),
                replication_factor: ReplicationFactor::ONE,
                unit_time_transfer_chunk_size: Vec::new(),
            }
        }
        fn computes(&self) -> Vec<NodeId> {
            vec![NodeId(2)]
        }
        fn compute(&self, _node: NodeId) -> ComputeNode {
            ComputeNode {
                name: NodeName("mul".to_owned()),
                op: ComputeType::Fmul,
                ex_unit: SenComponent::Lxlu,
                inputs: vec![
                    operand(SenComponent::Lxluscalereg, None),
                    operand(SenComponent::Latch, None),
                ],
                outputs: Vec::new(),
                num_folds_engaged: NumFolds::ONE,
                data_format: None,
                instr_attribute: InstrAttribute::default(),
            }
        }
        fn stick_dims(&self, _lds: LdsIdx) -> Vec<PrimaryDim> {
            vec![PrimaryDim::Out]
        }
        fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
            match node {
                NodeId(1) => Some(INNER),
                NodeId(10) => Some(OUTER),
                NodeId(2) => Some(FMUL_LOOP),
                _ => None,
            }
        }
        fn src_loop_ele_offsets(
            &self,
            node: NodeId,
            _corelet: Corelet,
        ) -> Vec<(LoopId, Vec<PrimaryDim>)> {
            if node == NodeId(1) {
                vec![
                    (INNER, vec![PrimaryDim::Out]),
                    (OUTER, vec![PrimaryDim::Out]),
                ]
            } else {
                Vec::new()
            }
        }
        fn set_src_loop_ele_offset(
            &mut self,
            node: NodeId,
            _corelet: Corelet,
            dim_loop: LoopId,
            dim: PrimaryDim,
            offset: LoopEleOffset,
        ) {
            self.0.push(Write::Src(node, dim_loop, dim, offset));
        }
        fn set_input_loop_ele_offset(
            &mut self,
            node: NodeId,
            input: InputIdx,
            _corelet: Corelet,
            dim_loop: Option<LoopId>,
            dim: PrimaryDim,
            offset: LoopEleOffset,
        ) {
            self.0
                .push(Write::Input(node, input, dim_loop, dim, offset));
        }
    }

    #[test]
    fn restickify_steps_two_then_one_and_the_latched_fmul_steps_its_second_input() {
        // ⛔ RCUDD1A IS BELOW SEN1P5: the whole body is skipped, and that is a compile-time fact.
        let mut before = Offsets::default();
        adjust_loop_offsets_and_addresses::<Dd2, _>(&mut before);
        assert_eq!(before.0, Vec::new());

        let mut dsc = Offsets::default();
        adjust_loop_offsets_and_addresses::<Sen1p5, _>(&mut dsc);
        assert_eq!(
            dsc.0,
            vec![
                Write::Src(NodeId(1), INNER, PrimaryDim::Out, LoopEleOffset(2)),
                Write::Src(NodeId(1), OUTER, PrimaryDim::Out, LoopEleOffset(1)),
                Write::Input(
                    NodeId(2),
                    InputIdx(1),
                    Some(FMUL_LOOP),
                    PrimaryDim::In,
                    LoopEleOffset(1),
                ),
            ]
        );
    }

    /// What entry 134 did to the tree, in the order it did it.
    #[derive(Debug, PartialEq, Eq)]
    enum Edit {
        Moved(NodeId, NodeId),
        Deleted(NodeId),
    }

    /// Three conditions: 3 is external, 4 is relevant to nobody, 5 has a "then" region matching its
    /// own core/corelet set.
    #[derive(Default)]
    struct Conditions(Vec<Edit>);

    fn core_cl(corelets: &[u32]) -> CoreClSet {
        let core = Core::checked(0).expect("core 0");
        CoreClSet(BTreeMap::from([(
            core,
            corelets
                .iter()
                .filter_map(|cl| Corelet::checked(*cl))
                .collect(),
        )]))
    }

    impl ConditionSimplification for Conditions {
        fn head(&self) -> LoopId {
            LoopId(NodeId(0))
        }
        fn relevant_comps(&self, _node: LoopId) -> Vec<SenComponent> {
            vec![SenComponent::Pe]
        }
        fn conditions(&self, _head: LoopId) -> Vec<NodeId> {
            vec![NodeId(3), NodeId(4), NodeId(5)]
        }
        fn relevant_core_cl(&self, node: NodeId) -> CoreClSet {
            match node {
                NodeId(4) => CoreClSet::default(),
                _ => core_cl(&[0]),
            }
        }
        fn has_core_cl_cond(&self, _condition: NodeId) -> bool {
            true
        }
        fn branches(&self, condition: NodeId) -> Vec<NodeId> {
            if condition == NodeId(5) {
                vec![NodeId(6)]
            } else {
                Vec::new()
            }
        }
        fn move_after(&mut self, node: NodeId, sibling: NodeId) {
            self.0.push(Edit::Moved(node, sibling));
        }
        fn delete_node(&mut self, node: NodeId) {
            self.0.push(Edit::Deleted(node));
        }
    }

    #[test]
    fn a_matching_branch_is_hoisted_an_empty_condition_is_dropped_and_an_external_one_is_left() {
        let mut metadata = Metadata::default();
        metadata.external_nodes.insert(NodeId(3));
        let mut tree = Conditions::default();
        let head = ScheduleHead::of(&tree).expect("the head's relevantComps_ was filled");
        simplify_schedule_tree(&mut tree, &metadata, head);
        // Reverse order: 5 hoists its branch, 4 goes away, 3 is external and untouched.
        assert_eq!(
            tree.0,
            vec![
                Edit::Moved(NodeId(6), NodeId(5)),
                Edit::Deleted(NodeId(5)),
                Edit::Deleted(NodeId(4)),
            ]
        );
    }

    /// `labeledDs_` whose own `ldsIdx_` fields are NOT its positions.
    struct Labeled(Vec<LdsIdx>);

    impl LabeledDsIndices for Labeled {
        fn labeled_ds_indices(&self) -> Vec<LdsIdx> {
            self.0.clone()
        }
    }

    #[test]
    fn the_seeded_map_is_the_identity_over_each_entrys_own_lds_index() {
        let mut metadata = Metadata::default();
        update_lds_idx_metadata(&mut metadata, &Labeled(vec![LdsIdx(4), LdsIdx(2)]));
        assert_eq!(
            metadata.lds_idx_after_ddc,
            BTreeMap::from([(LdsIdx(4), LdsIdx(4)), (LdsIdx(2), LdsIdx(2))])
        );
    }

    /// One data stage's `coreletSplit_`, keyed by its id.
    struct Stages(BTreeMap<DatastageId, BTreeSet<PrimaryDim>>);

    impl DataStages for Stages {
        fn corelet_split_dims(&self, stage: DatastageId) -> BTreeSet<PrimaryDim> {
            self.0.get(&stage).cloned().unwrap_or_default()
        }
    }

    #[test]
    fn the_corelet_split_dim_is_the_lowest_dim_the_core_stage_splits() {
        let mut global = GlobalData {
            corelet_split_dim: Some(PrimaryDim::X1),
            ..GlobalData::default()
        };
        // `Y` is declared after `Out` in `PrimaryDimTypes`, so `begin()` lands on `Out`.
        let split = BTreeSet::from([PrimaryDim::Y, PrimaryDim::Out]);
        init_global_data(
            &mut global,
            &Stages(BTreeMap::from([(Metadata::CORE_DSTGID, split)])),
        );
        assert_eq!(global.corelet_split_dim, Some(PrimaryDim::Out));
        // No such data stage: the stale dim is still cleared.
        init_global_data(&mut global, &Stages(BTreeMap::new()));
        assert_eq!(global.corelet_split_dim, None);
    }
}

#[cfg(test)]
mod tests_e124_e131 {
    use super::{
        AllocArena, AllocLive, ComputeArena, ComputeMasks, ComputeOp, ConstantData, CoreletShapes,
        DfsIndex,
        LdsSticks, LiveRange, Ln32, SplatDims, StorageName, StorageNames, TransferLds,
        TransferOperands, finalize_allocate_layouts, get_cl_split_dim,
        get_lds_or_const_name_of_alloc_node, get_pe_sfp_split_dim, minimize_allocations,
        populate_unit_time_transfers, set_pe_folds_if_pt_interaction, spread_data_in_allocate,
    };
    use core::num::NonZeroU64;
    use std::collections::{BTreeMap, BTreeSet};

    use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

    use crate::arch::{Elements, Sen1p5, Sticks};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        PaddedExtent, PrimaryDim, Sample, Stage, StickDims,
    };
    use crate::formats::DataFormat;
    use crate::schedule::ddc::fold::{AllocId, ConstIdx, NodeId};
    use crate::schedule::ddc::metadata::{Allocation, DdcMemory, Metadata};
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, AllocateNode, ComputeNode, Coordinate, CoordinateCategory,
        DataInfo, Dsc, Dsts, FoldCardinality, FoldCoeff, FoldLabel, InstrAttribute, LayoutDims,
        LdsIdx, MaxDimSize, NodeName, Operand, ReplicationFactor, StartAddress, TransferNode,
        TransferPadding,
    };
    use crate::units::NumFolds;

    /// A NAME TABLE that answers with the id it was asked about, so a test can tell the two apart.
    struct Names;
    impl StorageNames for Names {
        fn lds_name(&self, lds: LdsIdx) -> StorageName {
            StorageName(format!("lds{}", lds.0))
        }
        fn constant_name(&self, constant: ConstIdx) -> StorageName {
            StorageName(format!("const{}", constant.0))
        }
    }

    /// ONE DIM'S EXTENTS, and the PE/SFP split the chunk stage requests.
    struct Chunk {
        extents: BTreeMap<PrimaryDim, i64>,
        pe_sfp_split: BTreeSet<PrimaryDim>,
    }

    impl Stage for Chunk {
        fn is_symbolic(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_corelet_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_row_split(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_pe_sfp_split(&self, dim: PrimaryDim) -> bool {
            self.pe_sfp_split.contains(&dim)
        }
        fn splits_any_row(&self) -> bool {
            false
        }
        fn extent(
            &self,
            dim: PrimaryDim,
            _at: Sample,
        ) -> crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent {
            crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent(
                self.extents.get(&dim).copied().unwrap_or(-1),
            )
        }
        fn padded_extent(&self, _dim: PrimaryDim, _at: Sample) -> Option<PaddedExtent> {
            None
        }
    }

    /// ONE STICK for every labelled DS.
    struct Sticked(StickDims);
    impl LdsSticks for Sticked {
        fn stick_dims(&self, _lds: LdsIdx) -> StickDims {
            self.0.clone()
        }
    }

    /// ONE LAYOUT ORDER for every labelled DS.
    struct Layout(LayoutDims);
    impl Dsc for Layout {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            self.0.clone()
        }
    }

    /// THE ONE MASKED COMPUTE.
    struct Masked(Option<NodeId>);
    impl ComputeMasks for Masked {
        fn computes_under_mask(&self, node: NodeId) -> bool {
            self.0 == Some(node)
        }
    }

    /// THE CORE/CORELET EXTENTS entry 129 compares, with `Out` the only dim that may differ.
    struct Corelets {
        used: u32,
        out_per_corelet: i64,
    }
    impl CoreletShapes for Corelets {
        fn corelets_used(&self) -> u32 {
            self.used
        }
        fn core_extent(
            &self,
            dim: PrimaryDim,
        ) -> crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent {
            crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent(match dim {
                PrimaryDim::Out => 128,
                _ => 1,
            })
        }
        fn corelet_extent(
            &self,
            dim: PrimaryDim,
        ) -> crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent {
            crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent(match dim {
                PrimaryDim::Out => self.out_per_corelet,
                _ => 1,
            })
        }
    }

    fn allocate(name: &str) -> AllocateNode {
        AllocateNode {
            name: NodeName(name.to_owned()),
            component: SenComponent::Ptxrf,
            lds: Some(LdsIdx(0)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new((PrimaryDim::Out, MaxDimSize::Unset), Vec::new()),
            start_address: StartAddress::default(),
            placement: AllocPlacement::default(),
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        }
    }

    fn operand(unit: SenComponent) -> Operand {
        Operand {
            unit,
            storage: SenComponent::NoComponent,
            data: DataInfo::default(),
        }
    }

    #[test]
    fn e124_names_the_temp_storage_then_the_lds_then_the_constant() {
        let mut node = allocate("a");
        assert_eq!(
            get_lds_or_const_name_of_alloc_node(&node, &Names),
            Some(StorageName("lds0".to_owned()))
        );
        node.temp_storage_for_compute = Some(NodeName("compute".to_owned()));
        assert_eq!(
            get_lds_or_const_name_of_alloc_node(&node, &Names),
            Some(StorageName("compute".to_owned()))
        );
        node.temp_storage_for_compute = None;
        node.lds = None;
        node.const_idx = Some(ConstIdx(3));
        assert_eq!(
            get_lds_or_const_name_of_alloc_node(&node, &Names),
            Some(StorageName("const3".to_owned()))
        );
        node.const_idx = None;
        assert_eq!(get_lds_or_const_name_of_alloc_node(&node, &Names), None);
    }

    #[test]
    fn e125_shares_one_shadow_group_only_between_disjoint_live_ranges() {
        let live = |alloc: u32, users: &[u32]| AllocLive {
            alloc: AllocId(alloc),
            component: SenComponent::Sfplrf,
            lds: Some(LdsIdx(0)),
            range: LiveRange::of(&users.iter().map(|&u| DfsIndex(u)).collect::<Vec<_>>()),
        };
        let allocs = [live(0, &[0, 1]), live(1, &[4, 5]), live(2, &[0, 1])];
        let mut metadata = Metadata::default();
        minimize_allocations(&allocs, true, &mut metadata);
        assert_eq!(
            metadata.shadow_allocations,
            vec![vec![AllocId(0), AllocId(1)], vec![AllocId(2)]]
        );
        // Without auto shuffling every allocation stands alone.
        minimize_allocations(&allocs, false, &mut metadata);
        assert_eq!(
            metadata.shadow_allocations,
            vec![vec![AllocId(0)], vec![AllocId(2)], vec![AllocId(1)]]
        );
    }

    #[test]
    fn e126_splats_a_constant_source_one_element_at_a_time() {
        let mut transfer = TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            name: NodeName("t".to_owned()),
            src: operand(SenComponent::Constant),
            dsts: Dsts::new(operand(SenComponent::Lxlu), Vec::new()),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
        };
        let operands = TransferOperands::ConstantToTensor(Some(TransferLds {
            stick: StickDims(vec![(PrimaryDim::Out, Elements(64))]),
            splat_dims: SplatDims::default(),
            format: DataFormat::IeeeFp32,
        }));
        assert_eq!(
            populate_unit_time_transfers::<Sen1p5>(&mut transfer, &operands, None),
            Some(())
        );
        assert_eq!(transfer.replication_factor, ReplicationFactor(64));
        assert_eq!(
            transfer
                .unit_time_transfer_chunk_size
                .iter()
                .map(|chunk| chunk.size_dim.size)
                .collect::<Vec<_>>(),
            vec![Elements(1)]
        );
    }

    #[test]
    fn e126_replicates_a_constant_to_constant_transfer_across_the_stick() {
        let mut transfer = TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            name: NodeName("t".to_owned()),
            src: operand(SenComponent::Constant),
            dsts: Dsts::new(operand(SenComponent::Constant), Vec::new()),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
        };
        let operands = TransferOperands::ConstantToConstant(ConstantData {
            elements: NonZeroU64::new(8).expect("8 is not zero"),
            format: DataFormat::Sen169Fp16,
        });
        // 8 * 128 bytes per stick / 8 elements / 16 bits.
        assert_eq!(
            populate_unit_time_transfers::<Sen1p5>(&mut transfer, &operands, None),
            Some(())
        );
        assert_eq!(transfer.replication_factor, ReplicationFactor(8));
        // An element count the constant does not divide is the `DT_ERROR`.
        assert_eq!(
            populate_unit_time_transfers::<Sen1p5>(&mut transfer, &operands, Some(Elements(12))),
            None
        );
    }

    #[test]
    fn e127_spreads_only_a_masked_ptxrf_allocation() {
        let mut metadata = Metadata::default();
        metadata.new_allocations.insert(
            DdcMemory::PtxRf,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(LdsIdx(0), AllocId(7))]),
                cons_id_and_alloc_node: BTreeMap::new(),
                comp_and_alloc_node: BTreeMap::new(),
            },
        );
        let mut node = allocate("ptxrf");
        node.alloc_users = vec![NodeId(3)];
        let ops = [ComputeOp {
            op_func: OpFunc::ReStickifyOpHbm,
            format: DataFormat::Sen169Fp16,
        }];

        let mut allocs: AllocArena = BTreeMap::from([(AllocId(7), node.clone())]);
        spread_data_in_allocate::<Sen1p5>(&ops, &metadata, &mut allocs, &Masked(None));
        assert!(allocs[&AllocId(7)].gap_stick_spread.is_empty());

        let mut allocs: AllocArena = BTreeMap::from([(AllocId(7), node)]);
        spread_data_in_allocate::<Sen1p5>(&ops, &metadata, &mut allocs, &Masked(Some(NodeId(3))));
        assert_eq!(
            allocs[&AllocId(7)].gap_stick_spread,
            BTreeMap::from([(PrimaryDim::Out, Sticks(8))])
        );
    }

    #[test]
    fn e128_normalises_a_max_dim_size_by_its_stick_and_shares_the_shadow_address() {
        let mut placed = Coordinate::default();
        placed.add_fold_front(
            PrimaryDim::Out,
            CoordinateCategory::Spatial,
            FoldCardinality(4),
            FoldLabel("cl_fold_out".to_owned()),
            FoldCoeff(1),
            FoldCoeff(0),
        );
        let address = StartAddress::new(
            placed
                .fold_dim(PrimaryDim::Out)
                .cloned()
                .expect("the fold was just added"),
        );

        let mut first = allocate("first");
        first.layout = AllocLayout::new(
            (PrimaryDim::Out, MaxDimSize::Stage(Metadata::CHUNK_DSTGID)),
            Vec::new(),
        );
        first.start_address = address.clone();
        let mut second = allocate("second");
        second.layout = first.layout.clone();

        let mut allocs: AllocArena = BTreeMap::from([(AllocId(0), first), (AllocId(1), second)]);
        let mut metadata = Metadata::default();
        metadata.new_allocations.insert(
            DdcMemory::PtxRf,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(LdsIdx(0), AllocId(0))]),
                cons_id_and_alloc_node: BTreeMap::new(),
                comp_and_alloc_node: BTreeMap::new(),
            },
        );
        metadata.shadow_allocations = vec![vec![AllocId(0), AllocId(1)]];
        let stages = BTreeMap::from([(
            Metadata::CHUNK_DSTGID,
            Chunk {
                extents: BTreeMap::from([(PrimaryDim::Out, 128)]),
                pe_sfp_split: BTreeSet::new(),
            },
        )]);
        let sticks = Sticked(StickDims(vec![(PrimaryDim::Out, Elements(64))]));

        assert_eq!(
            finalize_allocate_layouts(&metadata, &mut allocs, &stages, &sticks),
            Some(())
        );
        assert_eq!(
            allocs[&AllocId(0)].layout.iter().collect::<Vec<_>>(),
            vec![(PrimaryDim::Out, MaxDimSize::Resolved(Elements(2)))]
        );
        assert_eq!(allocs[&AllocId(1)].start_address, address);
    }

    #[test]
    fn e129_names_the_corelet_split_dim_and_refuses_a_split_it_cannot_find() {
        assert_eq!(
            get_cl_split_dim(&Corelets {
                used: 2,
                out_per_corelet: 64
            }),
            Some(BTreeSet::from([PrimaryDim::Out]))
        );
        assert_eq!(
            get_cl_split_dim(&Corelets {
                used: 2,
                out_per_corelet: 128
            }),
            None
        );
        assert_eq!(
            get_cl_split_dim(&Corelets {
                used: 1,
                out_per_corelet: 128
            }),
            Some(BTreeSet::new())
        );
    }

    #[test]
    fn e130_splits_the_innermost_even_dim_for_a_reciprocal() {
        let chunk = Chunk {
            extents: BTreeMap::from([(PrimaryDim::In, 8), (PrimaryDim::Out, 32)]),
            pe_sfp_split: BTreeSet::new(),
        };
        let sticks = Sticked(StickDims(vec![(PrimaryDim::Out, Elements(8))]));
        let layout = Layout(LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Out]));
        let op = ComputeOp {
            op_func: OpFunc::Reciprocal,
            format: DataFormat::Sen169Fp16,
        };
        assert_eq!(
            get_pe_sfp_split_dim::<Sen1p5, Chunk>(
                &op,
                &chunk,
                LdsIdx(0),
                &layout,
                &sticks,
                Ln32::Off
            ),
            Some(BTreeSet::from([PrimaryDim::Out]))
        );
        // A layout dim the chunk stage states no extent for is SKIPPED, not a refusal.
        let unset = Layout(LayoutDims::new(
            PrimaryDim::In,
            vec![PrimaryDim::Out, PrimaryDim::Y],
        ));
        assert_eq!(
            get_pe_sfp_split_dim::<Sen1p5, Chunk>(
                &op,
                &chunk,
                LdsIdx(0),
                &unset,
                &sticks,
                Ln32::Off
            ),
            Some(BTreeSet::from([PrimaryDim::Out]))
        );
        // An op-func the split is not implemented for asks for none of it.
        let add = ComputeOp {
            op_func: OpFunc::Add,
            format: DataFormat::Sen169Fp16,
        };
        assert_eq!(
            get_pe_sfp_split_dim::<Sen1p5, Chunk>(
                &add,
                &chunk,
                LdsIdx(0),
                &layout,
                &sticks,
                Ln32::Off
            ),
            Some(BTreeSet::new())
        );
    }

    #[test]
    fn e131_folds_the_pe_only_when_a_pt_compute_is_present() {
        let compute = |unit: SenComponent| ComputeNode {
            name: NodeName("c".to_owned()),
            op: crate::generated::ComputeType::Fmul,
            ex_unit: unit,
            inputs: Vec::new(),
            outputs: Vec::new(),
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        };
        let mut alone: ComputeArena = BTreeMap::from([(NodeId(0), compute(SenComponent::Pe0))]);
        assert_eq!(
            set_pe_folds_if_pt_interaction::<Sen1p5>(&mut alone),
            Some(())
        );
        assert_eq!(alone[&NodeId(0)].num_folds_engaged, NumFolds::ONE);

        let mut with_pt: ComputeArena = BTreeMap::from([
            (NodeId(0), compute(SenComponent::Pe0)),
            (NodeId(1), compute(SenComponent::Ptrow0)),
        ]);
        assert_eq!(
            set_pe_folds_if_pt_interaction::<Sen1p5>(&mut with_pt),
            Some(())
        );
        assert_eq!(with_pt[&NodeId(0)].num_folds_engaged, NumFolds(2));
        assert_eq!(with_pt[&NodeId(1)].num_folds_engaged, NumFolds::ONE);
    }
}

#[cfg(test)]
mod tests_e258_e263 {
    use super::*;
    use crate::arch::Dd2;
    use crate::schedule::ddc::fold::ScheduleTree;
    use crate::schedule::ddc::metadata::Allocation;
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, DataInfo, Dsts, SyncDirection, SyncStrength, SyncUnits,
        TransferPadding,
    };

    /// The one core and the two corelets a `Dd2` core has.
    fn core0() -> Core {
        Core::checked(0).expect("core 0")
    }
    fn cl0() -> Corelet {
        Corelet::at::<0>()
    }
    fn cl1() -> Corelet {
        Corelet::checked(1).expect("a Dd2 core has two corelets")
    }

    /// The loop the fixtures hang their transfer under, and the block above it.
    const LOOP: LoopId = LoopId(NodeId(1));
    const ROOT: NodeId = NodeId(9);
    /// The denominator and the numerator datastage every fixture loop names.
    const DEN: DatastageId = DatastageId(0);
    const NUM: DatastageId = DatastageId(1);

    fn operand(unit: SenComponent, storage: SenComponent, lds: Option<u32>) -> Operand {
        Operand {
            unit,
            storage,
            data: DataInfo {
                data_connect: None,
                my_lds_idx: lds.map(LdsIdx),
                constant_id: None,
            },
        }
    }

    /// One allocation of `I`, since [`AllocateNode`] has no `Default`.
    fn alloc_node(component: SenComponent, lds: Option<u32>) -> AllocateNode {
        AllocateNode {
            name: NodeName("alloc".to_owned()),
            component,
            lds: lds.map(LdsIdx),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new(
                (PrimaryDim::I, MaxDimSize::Resolved(Elements(64))),
                Vec::new(),
            ),
            start_address: StartAddress::new(FoldDim::default()),
            placement: AllocPlacement::default(),
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        }
    }

    /// ONE SCHEDULE TREE, stated as the tables the walks read off it.
    #[derive(Default)]
    struct Tree {
        nodes: Vec<NodeId>,
        kinds: BTreeMap<NodeId, NodeKind>,
        parents: BTreeMap<NodeId, NodeId>,
        owners: BTreeMap<NodeId, LoopId>,
        dims: BTreeMap<LoopId, Vec<(PrimaryDim, MetaDimKind)>>,
        transfers: BTreeMap<NodeId, TransferNode>,
        alloc_prev: BTreeMap<AllocId, NodeId>,
        under_block: Vec<LoopId>,
        inserted: Vec<NodeId>,
    }

    impl ScheduleWalk for Tree {
        fn loops_under(&self, _from: BlockId) -> Vec<LoopId> {
            self.under_block.clone()
        }
        fn nodes_of_kind(&self, kind: NodeKind) -> Vec<NodeId> {
            self.nodes
                .iter()
                .copied()
                .filter(|node| self.kinds.get(node) == Some(&kind))
                .collect()
        }
        fn allocates(&self) -> Vec<AllocId> {
            self.alloc_prev.keys().copied().collect()
        }
        fn nodes_of_kind_under(
            &self,
            _from: Option<NodeId>,
            kind: NodeKind,
            _unit: SenComponent,
        ) -> Vec<NodeId> {
            self.nodes_of_kind(kind)
        }
        fn prev_of_alloc(&self, alloc: AllocId) -> Option<NodeId> {
            self.alloc_prev.get(&alloc).copied()
        }
        fn transfer(&self, node: NodeId) -> Option<TransferNode> {
            self.transfers.get(&node).cloned()
        }
        fn as_loop(&self, node: NodeId) -> Option<LoopId> {
            (self.kinds.get(&node) == Some(&NodeKind::Loop)).then_some(LoopId(node))
        }
        fn prev(&self, node: NodeId) -> Option<NodeId> {
            self.parents.get(&node).copied()
        }
        fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
            self.owners.get(&node).copied()
        }
        fn loop_dims(&self, at: LoopId) -> Vec<(PrimaryDim, MetaDimKind)> {
            self.dims.get(&at).cloned().unwrap_or_default()
        }
    }

    impl ScheduleNodes for Tree {
        fn nodes(&self) -> Vec<NodeId> {
            self.nodes.clone()
        }
        fn kind(&self, node: NodeId) -> Option<NodeKind> {
            self.kinds.get(&node).copied()
        }
        fn is_parametric(&self, _at: LoopId) -> bool {
            false
        }
        fn loop_stages(&self, _at: LoopId) -> LoopStages {
            LoopStages { num: NUM, den: DEN }
        }
        fn parametric_stride(&self, _at: LoopId) -> LoopEleOffset {
            LoopEleOffset(0)
        }
        fn parametric_iter_count(
            &self,
            _at: LoopId,
            _corelet: Corelet,
            _unit: SenComponent,
        ) -> IterCount {
            IterCount(1)
        }
        fn is_relevant(&self, _node: NodeId, _unit: SenComponent) -> bool {
            true
        }
        fn next_view_len(&self, _node: NodeId, _unit: SenComponent) -> usize {
            1
        }
        fn alloc_owner_loop(&self, _alloc: AllocId) -> Option<LoopId> {
            None
        }
        fn transfer_has_padding(&self, _node: NodeId) -> bool {
            false
        }
        fn compute(&self, _node: NodeId) -> Option<ComputeNode> {
            None
        }
        fn repetition_with_offset_outputs(&self, _node: NodeId) -> usize {
            0
        }
    }

    impl ScheduleTree for Tree {
        fn kind(&self, node: NodeId) -> NodeKind {
            // The fixture's root is the block everything else hangs under.
            self.kinds.get(&node).copied().unwrap_or(NodeKind::Block)
        }
        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.parents.get(&node).copied()
        }
        fn children(&self, _block: BlockId) -> Vec<NodeId> {
            Vec::new()
        }
    }

    impl MaskInsertion for Tree {
        fn insert_condition_before(&mut self, before: NodeId, _cond: ConditionNode) -> Option<()> {
            self.inserted.push(before);
            Some(())
        }
    }

    /// ONE DESIGN SPACE — every fact entries 258-262 ask a DSC for.
    struct Space {
        corelets: Vec<Corelet>,
        allocs: BTreeMap<(LdsIdx, SenComponent), AllocId>,
        /// `addressGranularityScalePerUnit`.
        scale: NonZeroU64,
        /// The denominator datastage's own step, and the numerator's whole span.
        step: Extent,
        span: Extent,
        masking: BTreeMap<PrimaryDim, Vec<MaskRun>>,
        declares_samv: bool,
    }

    impl Default for Space {
        fn default() -> Self {
            Self {
                corelets: vec![cl0()],
                allocs: BTreeMap::new(),
                scale: NonZeroU64::new(1).expect("one is not zero"),
                step: Extent(4),
                span: Extent(16),
                masking: BTreeMap::new(),
                declares_samv: false,
            }
        }
    }

    impl Placement for Space {
        fn cores_used(&self) -> CoresUsed {
            CoresUsed::new(core0(), Vec::new())
        }
        fn corelets_used(&self) -> Vec<Corelet> {
            self.corelets.clone()
        }
        fn corelets_used_total(&self) -> Vec<Corelet> {
            self.corelets.clone()
        }
        fn buffer_capacity(
            &self,
            _alloc: AllocId,
            _lds: Option<LdsIdx>,
            _at: Option<(Corelet, Row)>,
        ) -> Bytes {
            Bytes(256)
        }
        fn is_scale_tensor(&self, _lds: LdsIdx) -> bool {
            false
        }
        fn l0_tethered(&self) -> L0Tethered {
            L0Tethered::Whole
        }
        fn subcore(&self, _core: Core) -> u32 {
            0
        }
        fn address_fold_depth(&self) -> usize {
            3
        }
    }

    impl StorageNames for Space {
        fn lds_name(&self, lds: LdsIdx) -> StorageName {
            StorageName(format!("lds{}", lds.0))
        }
        fn constant_name(&self, _constant: ConstIdx) -> StorageName {
            StorageName("const".to_owned())
        }
    }

    impl LdsSticks for Space {
        fn stick_dims(&self, _lds: LdsIdx) -> StickDims {
            StickDims(vec![(PrimaryDim::I, Elements(1))])
        }
    }

    impl StageSizes for Space {
        fn dim_extent(
            &self,
            _stage: DatastageId,
            _dim: PrimaryDim,
            _unit: SenComponent,
            _corelet: Option<Corelet>,
            _padding: PadType,
            _density: Density,
        ) -> Extent {
            self.span
        }
        fn comp_view(
            &self,
            stage: DatastageId,
            _dim: PrimaryDim,
            _unit: SenComponent,
            _corelet: Option<Corelet>,
            _padding: PadType,
        ) -> Extent {
            if stage == NUM { self.span } else { self.step }
        }
        fn lds_replication(&self, _lds: LdsIdx, _dim: PrimaryDim) -> ReplicationFactor {
            ReplicationFactor::ONE
        }
        fn dim_density(&self, _lds: LdsIdx, _dim: PrimaryDim) -> Density {
            Density::FULL
        }
        fn corelet_split(&self, _stage: DatastageId, _dim: PrimaryDim) -> Option<Vec<Elements>> {
            None
        }
        fn alloc_padding_sizes(&self, _alloc: AllocId, _dim: PrimaryDim) -> Option<PaddingSizes> {
            None
        }
        fn stage_padding_sizes(
            &self,
            _stage: DatastageId,
            _dim: PrimaryDim,
        ) -> Option<PaddingSizes> {
            None
        }
        fn size_stage(&self, _alloc: AllocId) -> DatastageId {
            DEN
        }
        fn is_sole_partial_reduction_input(&self, _lds: LdsIdx) -> bool {
            false
        }
    }

    impl OffsetSizes for Space {
        fn lds_alloc(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId> {
            self.allocs.get(&(lds, storage)).copied()
        }
        fn const_alloc(&self, _constant: ConstIdx, _storage: SenComponent) -> Option<AllocId> {
            None
        }
        fn address_scale(&self, _unit: GenericComp, _storage: SenComponent) -> Option<NonZeroU64> {
            Some(self.scale)
        }
        fn stage_padding_dims(&self, _stage: DatastageId) -> Vec<(PrimaryDim, PaddingSizes)> {
            Vec::new()
        }
        fn has_symbolic_dim(&self, _stage: DatastageId, _dim: PrimaryDim) -> bool {
            false
        }
        fn pe_sfp_split_dims(&self, _stage: DatastageId) -> Vec<PrimaryDim> {
            Vec::new()
        }
        fn block_transfer_size(
            &self,
            _node: NodeId,
            _unit: SenComponent,
            _corelet: Corelet,
            _dim: PrimaryDim,
        ) -> Elements {
            Elements(0)
        }
        fn temporal_stride(
            &self,
            _node: NodeId,
            _alloc: AllocId,
            _at: LoopId,
            _dim: PrimaryDim,
        ) -> Option<LoopEleOffset> {
            Some(LoopEleOffset(0))
        }
    }

    impl Masking for Space {
        fn coordinate_masking(&self) -> BTreeMap<PrimaryDim, Vec<MaskRun>> {
            self.masking.clone()
        }
        fn declares_samv_wsllen(&self) -> bool {
            self.declares_samv
        }
        fn masking_constant(&self) -> Option<ConstIdx> {
            None
        }
        fn splits_across_cores(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn is_symbolic(&self, _dim: PrimaryDim) -> bool {
            false
        }
        fn non_broadcast_dims(&self, _lds: LdsIdx) -> BTreeSet<PrimaryDim> {
            BTreeSet::new()
        }
        fn lds_format(&self, _lds: LdsIdx) -> Option<DataFormat> {
            None
        }
    }

    /// A tracker that places everything at the one address it was built with.
    struct Trackers(Bytes);

    impl MemTrackers for Trackers {
        fn capacity(&self, _at: TrackerSite) -> Bytes {
            Bytes(4096)
        }
        fn backup(&mut self, _at: TrackerSite) {}
        fn restore_all(&mut self) {}
        fn remove(&mut self, _at: TrackerSite, _name: &StorageName) {}
        fn add_at(&mut self, _at: TrackerSite, _name: &StorageName, _size: Bytes, _addr: Bytes) {}
        fn check_and_add(
            &mut self,
            _at: TrackerSite,
            _name: &StorageName,
            _size: Bytes,
        ) -> Option<Placed> {
            Some(Placed::At(self.0))
        }
        fn check_and_add_at(
            &mut self,
            _at: TrackerSite,
            _name: &StorageName,
            _size: Bytes,
            address: Bytes,
        ) -> Option<Placed> {
            Some(Placed::At(address))
        }
    }

    /// No coordinate anywhere, which is what leaves entry 260's coordinate arm a no-op.
    struct NoCoords;

    impl CoordinateOffsets for NoCoords {
        fn node_work_slices(
            &self,
            _at: OperandSite,
            _core: Core,
        ) -> Option<BTreeMap<PrimaryDim, WorkSlice>> {
            None
        }
        fn alloc_work_slices(
            &self,
            _alloc: AllocId,
            _core: Core,
        ) -> Option<BTreeMap<PrimaryDim, WorkSlice>> {
            None
        }
        fn node_coordinate(&self, _at: OperandSite, _offsets: ElemOffsets) -> Coordinate {
            Coordinate::default()
        }
        fn alloc_coordinate(&self, _alloc: AllocId) -> Coordinate {
            Coordinate::default()
        }
        fn relevant_core_cl(&self, _node: NodeId) -> CoreClSet {
            CoreClSet(BTreeMap::new())
        }
        fn single_beta(&self, _folds: &FoldDim, _core: i64, _corelet: i64, _row: i64) -> FoldCoeff {
            FoldCoeff(0)
        }
        fn distance_in_steps(
            &self,
            _folds: &FoldDim,
            _beta: FoldCoeff,
            _fixed: &BTreeMap<usize, i64>,
        ) -> ConstEleOffset {
            ConstEleOffset::ZERO
        }
    }

    /// Every fill entry 260 wrote, kept by the operand it landed on.
    #[derive(Default)]
    struct Sink {
        fills: BTreeMap<OperandSite, DataInfoFill>,
        fusable_src: BTreeMap<NodeId, Option<LoopId>>,
    }

    impl DataInfoSink for Sink {
        fn fill(&mut self, at: OperandSite, fill: DataInfoFill) -> Option<()> {
            self.fills.insert(at, fill);
            Some(())
        }
        fn set_const_ele_offset(
            &mut self,
            at: OperandSite,
            core: Core,
            corelet: Corelet,
            dim: PrimaryDim,
            offset: ConstEleOffset,
        ) -> Option<()> {
            self.fills
                .get_mut(&at)?
                .const_ele_offsets
                .entry(core)
                .or_default()
                .entry(corelet)
                .or_default()
                .insert(dim, offset);
            Some(())
        }
        fn const_ele_offsets_empty(&self, at: OperandSite) -> Option<bool> {
            Some(
                self.fills
                    .get(&at)
                    .is_none_or(|fill| fill.const_ele_offsets.is_empty()),
            )
        }
        fn set_last_fusable_src(&mut self, node: NodeId, at: Option<LoopId>) -> Option<()> {
            self.fusable_src.insert(node, at);
            Some(())
        }
        fn set_last_fusable_dsts(&mut self, _node: NodeId, _at: Vec<Option<LoopId>>) -> Option<()> {
            Some(())
        }
    }

    /// The symbol table, never reached while the address is a byte count.
    struct NoSymbols;

    impl Symbols for NoSymbols {
        fn divide_symbols(&mut self, address: &StartAddress, by: NonZeroU64) -> StartAddress {
            address.divided_by(by)
        }
    }

    #[test]
    fn an_lx_allocation_takes_the_trackers_address_and_an_ephemeral_tracker_refuses_lx() {
        let dsc = Space::default();
        let tree = Tree {
            alloc_prev: BTreeMap::from([(AllocId(0), ROOT)]),
            ..Tree::default()
        };
        let mut allocs = AllocArena::from([(AllocId(0), alloc_node(SenComponent::Lx, Some(0)))]);
        let mut metadata = Metadata::default();
        metadata.new_allocations.insert(
            DdcMemory::Lx,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(LdsIdx(0), AllocId(0))]),
                ..Allocation::default()
            },
        );
        metadata.shadow_allocations = vec![vec![AllocId(0)]];
        let mut trackers = Trackers(Bytes(0x200));
        assert_eq!(
            alloc_all_mem::<Dd2, _, _, _>(
                &dsc,
                &tree,
                &mut metadata,
                &mut allocs,
                &mut trackers,
                LxTrackers::True,
                Commit::IfValid,
            ),
            Some(true)
        );
        let placed = &allocs[&AllocId(0)];
        assert_eq!(placed.start_address.at(core0(), cl0()), Some(Bytes(0x200)));
        // The whole 256-byte buffer, since a single-buffered allocation does not divide it.
        assert_eq!(placed.placement.buffer_offset[&core0()][&cl0()], Bytes(256));
        // ⛔ An LX allocation onto the ephemeral trackers is the `DT_CHECK`.
        assert_eq!(
            alloc_all_mem::<Dd2, _, _, _>(
                &dsc,
                &tree,
                &mut metadata,
                &mut allocs,
                &mut trackers,
                LxTrackers::Ephemeral,
                Commit::IfValid,
            ),
            None
        );
    }

    #[test]
    fn an_unsplit_lx_allocation_gives_every_corelet_corelet_zeros_address_and_buffer_offset() {
        let dsc = Space {
            corelets: vec![cl0(), cl1()],
            ..Space::default()
        };
        let mut node = alloc_node(SenComponent::Lx, Some(0));
        node.start_address.insert(core0(), cl0(), Bytes(0x80));
        node.placement.num_buffers = NumBuffers::Double;
        node.placement
            .buffer_offset
            .insert(core0(), BTreeMap::from([(cl0(), Bytes(0x40))]));
        let mut allocs = AllocArena::from([(AllocId(0), node)]);
        let metadata = Metadata::default();
        assert_eq!(
            calculate_cl_start_address::<Dd2, _>(&dsc, &metadata, &mut allocs, AllocId(0)),
            Some(())
        );
        // No corelet-split dim, so corelet 1 sits at corelet 0's own address.
        let placed = &allocs[&AllocId(0)];
        assert_eq!(placed.start_address.at(core0(), cl1()), Some(Bytes(0x80)));
        assert_eq!(placed.placement.buffer_offset[&core0()][&cl1()], Bytes(0x40));
        // ⛔ A component other than `LX` is the `DT_CHECK`.
        let mut elsewhere = AllocArena::from([(AllocId(0), alloc_node(SenComponent::L0, Some(0)))]);
        assert_eq!(
            calculate_cl_start_address::<Dd2, _>(&dsc, &metadata, &mut elsewhere, AllocId(0)),
            None
        );
    }

    #[test]
    fn a_transfer_source_takes_the_scaled_start_address_the_datastage_step_and_its_fusable_loop() {
        let dsc = Space {
            allocs: BTreeMap::from([((LdsIdx(0), SenComponent::Lx), AllocId(0))]),
            scale: NonZeroU64::new(2).expect("two is not zero"),
            ..Space::default()
        };
        let mut node = alloc_node(SenComponent::Lx, Some(0));
        node.start_address.insert(core0(), cl0(), Bytes(0x200));
        let allocs = AllocArena::from([(AllocId(0), node)]);
        let transfer = TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            name: NodeName("t".to_owned()),
            src: operand(SenComponent::Lxlu, SenComponent::Lx, Some(0)),
            dsts: Dsts::new(
                operand(SenComponent::Lxsu, SenComponent::NoComponent, None),
                Vec::new(),
            ),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
        };
        let tree = Tree {
            nodes: vec![NodeId(0)],
            kinds: BTreeMap::from([(NodeId(0), NodeKind::Transfer), (LOOP.0, NodeKind::Loop)]),
            parents: BTreeMap::from([(NodeId(0), LOOP.0), (LOOP.0, ROOT)]),
            owners: BTreeMap::from([(NodeId(0), LOOP)]),
            dims: BTreeMap::from([(LOOP, vec![(PrimaryDim::I, MetaDimKind::Unpadded)])]),
            transfers: BTreeMap::from([(NodeId(0), transfer)]),
            ..Tree::default()
        };
        let metadata = Metadata::default();
        let global = GlobalData::default();
        let inputs = OffsetInputs {
            dsc: &dsc,
            tree: &tree,
            coords: &NoCoords,
            metadata: &metadata,
            allocs: &allocs,
            global: &global,
            offsets: ElemOffsets::Datastage,
            unpadded: UnpaddedIndexing::Forbidden,
        };
        let mut sink = Sink::default();
        assert_eq!(
            fill_loop_offsets_and_addresses::<Dd2, _, _, _, _, _>(
                &inputs,
                &mut sink,
                &mut NoSymbols
            ),
            Some(())
        );
        let src = &sink.fills[&OperandSite::TransferSrc(NodeId(0))];
        // 0x200 bytes at an address granularity of two.
        assert_eq!(src.start_address.at(core0(), cl0()), Some(Bytes(0x100)));
        // The denominator datastage's own step along `I`, and no padding to offset past.
        assert_eq!(
            src.loop_ele_offsets[&cl0()][&LOOP][&PrimaryDim::I],
            LoopEleOffset(4)
        );
        assert!(src.const_ele_offsets.is_empty());
        assert_eq!(sink.fusable_src[&NodeId(0)], Some(LOOP));
        // The destination names no lds, so it is left exactly as it was.
        assert!(
            !sink
                .fills
                .contains_key(&OperandSite::TransferDst(NodeId(0), DestIdx(0)))
        );
    }

    #[test]
    fn an_implicit_sync_points_at_the_sole_transfer_below_it_and_only_inside_l0() {
        let dsc = Space::default();
        let tree = Tree {
            nodes: vec![NodeId(0)],
            kinds: BTreeMap::from([(NodeId(0), NodeKind::Transfer)]),
            alloc_prev: BTreeMap::from([(AllocId(0), ROOT)]),
            transfers: BTreeMap::from([(
                NodeId(0),
                TransferNode {
                    padding: TransferPadding::default(),
                    src_indirect: None,
                    dst_indirect: None,
                    name: NodeName("t".to_owned()),
                    src: operand(SenComponent::Lxlu, SenComponent::Lx, Some(0)),
                    dsts: Dsts::new(
                        operand(SenComponent::L0su, SenComponent::L0, Some(0)),
                        Vec::new(),
                    ),
                    replication_factor: ReplicationFactor::ONE,
                    unit_time_transfer_chunk_size: Vec::new(),
                },
            )]),
            ..Tree::default()
        };
        let mut metadata = Metadata::default();
        metadata.implicit_syncs.insert(NodeId(3), AllocId(0));
        let mut syncs = BTreeMap::from([(
            NodeId(3),
            SyncNode {
                name: NodeName("s".to_owned()),
                units: SyncUnits::new(SenComponent::L0su, []),
                direction: SyncDirection::Send,
                strength: SyncStrength::Hard,
                implicit_sync_ref_transfer: None,
                other_ends: Vec::new(),
            },
        )]);
        let mut computes = ComputeArena::new();
        let allocs = AllocArena::from([(AllocId(0), alloc_node(SenComponent::L0, Some(0)))]);
        assert_eq!(
            finalize_ops::<Dd2, _, _>(&dsc, &tree, &metadata, &allocs, &mut computes, &mut syncs),
            Some(())
        );
        assert_eq!(syncs[&NodeId(3)].implicit_sync_ref_transfer, Some(NodeId(0)));
        // ⛔ An implicit sync on a memory other than `L0` is the `DT_ERROR`.
        let elsewhere = AllocArena::from([(AllocId(0), alloc_node(SenComponent::Lx, Some(0)))]);
        assert_eq!(
            finalize_ops::<Dd2, _, _>(
                &dsc,
                &tree,
                &metadata,
                &elsewhere,
                &mut computes,
                &mut syncs
            ),
            None
        );
    }

    #[test]
    fn masking_refuses_a_request_the_dsc_declared_but_left_unfilled_and_masks_nothing_for_an_idle_run()
    {
        let metadata = Metadata::default();
        let mut tree = Tree::default();
        // ⛔ The `samv-wsllen` op const says an upstream tool owed us a request.
        let declared = Space {
            declares_samv: true,
            ..Space::default()
        };
        assert_eq!(
            coordinate_masking::<Dd2, _, _>(&declared, &metadata, &mut tree),
            None
        );
        // No op const and no request, so there is nothing to mask.
        assert_eq!(
            coordinate_masking::<Dd2, _, _>(&Space::default(), &metadata, &mut tree),
            Some(())
        );
        // A dim whose only run masks zero elements mints no SAMV node either.
        let idle = Space {
            masking: BTreeMap::from([(
                PrimaryDim::I,
                vec![MaskRun {
                    unmasked: Elements(64),
                    masked: Elements(0),
                }],
            )]),
            ..Space::default()
        };
        assert_eq!(
            coordinate_masking::<Dd2, _, _>(&idle, &metadata, &mut tree),
            Some(())
        );
        assert!(tree.inserted.is_empty());
    }

    #[test]
    fn the_below_chunk_boundary_set_is_the_loops_under_the_insertion_block_and_needs_one() {
        let tree = Tree {
            under_block: vec![LOOP, LoopId(NodeId(2))],
            ..Tree::default()
        };
        let mut metadata = Metadata::default();
        let mut global = GlobalData::default();
        // ⛔ `DT_CHECK(metadata.belowLxScheduleInsertBlock)`.
        assert_eq!(
            identify_below_chunk_boundary_loops(&tree, &metadata, &mut global),
            None
        );
        metadata.below_lx_schedule_insert_block = BlockId::of(&tree, ROOT);
        assert_eq!(
            identify_below_chunk_boundary_loops(&tree, &metadata, &mut global),
            Some(())
        );
        assert_eq!(
            global.loops_below_chunk_boundary,
            BTreeSet::from([LOOP, LoopId(NodeId(2))])
        );
    }
}
