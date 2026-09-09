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

use core::num::NonZeroU64;
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Arch, Elements, FoldedUnit, IsaGen, Sticks};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, Sample, SliceElems, Stage, StickDims, StickPart, VectorComp,
    cumulative_stick_sizes, stick_sizes,
};
use crate::formats::DataFormat;
use crate::generated::ComputeType;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{AllocId, ConstIdx, NodeId};
use crate::schedule::ddc::metadata::{DatastageId, DdcMemory, Metadata};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::dsc2::{
    AllocateNode, ComputeNode, Dsc, LdsIdx, MaxDimSize, ReplicationFactor, Size, SizeAndIndex,
    StickDimIdx, TransferNode, generic_comp,
};
use crate::units::{Core, Corelet, Row};

/// AN ALLOCATION ARENA — the `dsc2::AllocateNode*`s the metadata's maps name.
///
/// ⭐ WALKING THE SCHEDULE TREE TO REACH THEM IS THE MECHANISM; the identity is the [`AllocId`], and
/// that is what `newAllocations_` and `shadowAllocations_` already hold.
pub type AllocArena = BTreeMap<AllocId, AllocateNode>;

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
/// `instrAttribute_.computeMaskLoopOffsets_` non-empty (`ddc/ddcv1.cpp:1698-1704`), which is ONE
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

/// `dtGetEnv<bool>("ENABLE_LN32")` (`ddc/ddcv1.cpp:1849`) AS AN ARGUMENT.
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
/// ⭐ [`None`] IS THE REFERENCE'S `""`, which is not a name: every caller compares the result
/// against the empty string rather than using it.
#[must_use]
pub fn get_lds_or_const_name_of_alloc_node(
    anode: &AllocateNode,
    names: &impl StorageNames,
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

/// A POSITION IN THE SCHEDULE TREE'S DFS ORDER — `node_to_index`'s `int` (`ddc/ddcv1.cpp:44-48`).
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
    /// `{INT_MAX, 0}` — what a USER-LESS allocation gets (`ddc/ddcv1.cpp:52-53`).
    ///
    /// ⭐ INVERTED ON PURPOSE, and it is load-bearing: it is false against every interval in both
    /// directions, so an allocation nothing reads always lands in a shadow group of its own.
    pub const NONE: Self = Self {
        start: DfsIndex(u32::MAX),
        end: DfsIndex(0),
    };

    /// The range spanning every user (`ddc/ddcv1.cpp:54-60`).
    #[must_use]
    pub fn of(users: &[DfsIndex]) -> Self {
        let mut range = Self::NONE;
        for &user in users {
            range.start = range.start.min(user);
            range.end = range.end.max(user);
        }
        range
    }

    /// `overlapInterval` (`ddc/ddcv1.cpp:88-93`) — transcribed INCLUDING its asymmetry, so `self` is
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
    let per_dim = cumulative_stick_sizes(&sticks.stick_dims(input_lds), StickPart::Whole)?;
    let mut split = BTreeSet::new();
    for dim in dsc.layout_dims(input_lds).to_vec().into_iter().rev() {
        let at = Sample {
            corelet: Corelet::at::<0>(),
            row: None,
            comp: None,
        };
        let extent = u64::try_from(chunk_stage.extent(dim, at).0).ok()?;
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
/// ENGAGES THE PE'S SECOND FOLD ON EVERY COMPUTE of a SEN1P5 DSC that has any PT compute at all —
/// the PT's presence is what makes the PE fold, not the PE's own op.
///
/// ⛔ [`None`] IS `senCompToGenericComp.at()` ON A COMPONENT THAT MAP HAS NO KEY FOR, which is 20 of
/// the 107 (`sys-arch-spec/arch_enums.cpp:124-211`).
pub fn set_pe_folds_if_pt_interaction<A: Arch>(computes: &mut [ComputeNode]) -> Option<()> {
    if A::GEN < IsaGen::Sen1p5 {
        return Some(());
    }
    let mut do_change = false;
    for node in computes.iter() {
        if node.ex_unit.generic()? == SenComponent::Pt {
            do_change = true;
            break;
        }
    }
    if !do_change {
        return Some(());
    }
    for node in computes.iter_mut() {
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
/// `int` in `DataInfo::loopEleOffsets_` (`dsc/dsc2.h:729`).
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
    /// key of an `unordered_map<const LoopNode*, ..>` (`ddc/ddcv1.cpp:3260-3263`).
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlobalData {
    /// `coreletSplitDim` (`ddc/ddc.h:109`) — `PrimaryDimTypesCount` is UNSET, not dim zero.
    pub corelet_split_dim: Option<PrimaryDim>,
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

/// `adjustLoopOffsetsForLXLUCompute` (`ddc/ddcv1.cpp:3251-3264`) — one `in` element per trip of the
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
/// ⛔ THE RESTICKIFY PASS RE-RUNS ONCE PER MATCHING OP (`ddc/ddcv1.cpp:3267-3272`); its writes are
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

// crustify:todo: e258_allocAllMem
//   authority : ddc/ddcv1.cpp:132  (306 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::allocAllMem(bool commitIfValid)
//   extract   : crustify-ddc/cpp/ddc.cpp:6204-6510
//   calls     : e124_getLdsOrConstNameOfAllocNode
//   ⛔ NOTE   : SUPERSEDES HAND-TRANSCRIBED GUESSWORK: crates/compiler/deeptools/src/reginit.rs (1,569
//               lines, on the integ branch) hand-computes placement from ddc/ddcv1.cpp:132-360 -- THIS
//               function. Port what the authority does, not what reginit.rs guessed; the two will be
//               reconciled when this campaign and integ meet, and the authority wins.

// crustify:todo: e259_calculateClStartAddress
//   authority : ddc/ddcv1.cpp:1895  (123 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::calculateClStartAddress(dsc2::AllocateNode* allocNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:6521-6644
//   calls     : e104_clear
//   ⛔ NOTE   : PLACES ADDRESSES. Part of the span reginit.rs hand-transcribes.

// crustify:todo: e260_fillLoopOffsetsAndAddresses
//   authority : ddc/ddcv1.cpp:2355  (844 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::fillLoopOffsetsAndAddresses( const bool allowUnpaddedIndexingAtPaddedNoZeroPad)
//   extract   : crustify-ddc/cpp/ddc.cpp:6654-7499
//   calls     : e076_print, e102_print, e104_clear

// crustify:todo: e261_finalizeOps
//   authority : ddc/ddcv1.cpp:3329  (127 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::finalizeOps()
//   extract   : crustify-ddc/cpp/ddc.cpp:7511-7638
//   calls     : e131_setPeFoldsIfPtInteraction
//   ⛔ NOTE   : Binds register names to allocation start addresses (ddcv1.cpp:3345-3392) -- the `R +
//               std::to_string(startAddress)` convention our islands already read back.

// crustify:todo: e262_coordinateMasking
//   authority : ddc/ddcv1.cpp:3485  (181 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::coordinateMasking()
//   extract   : crustify-ddc/cpp/ddc.cpp:7648-7829
//   calls     : e104_clear

// crustify:todo: e263_identifyBelowChunkBoundaryLoops
//   authority : ddc/ddcv1.cpp:3683  (8 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::identifyBelowChunkBoundaryLoops()
//   extract   : crustify-ddc/cpp/ddc.cpp:7839-7847
//   calls     : e104_clear

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
    use crate::schedule::dsc2::{DataInfo, Dsts, NodeName, Operand, ReplicationFactor};
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
        AllocArena, AllocLive, ComputeMasks, ComputeOp, ConstantData, CoreletShapes, DfsIndex,
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
        AllocLayout, AllocateNode, ComputeNode, Coordinate, CoordinateCategory, DataInfo, Dsc,
        Dsts, FoldCardinality, FoldCoeff, FoldLabel, LayoutDims, LdsIdx, MaxDimSize, NodeName,
        Operand, ReplicationFactor, StartAddress, TransferNode,
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
        let address = StartAddress(
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
        };
        let mut alone = [compute(SenComponent::Pe0)];
        assert_eq!(
            set_pe_folds_if_pt_interaction::<Sen1p5>(&mut alone),
            Some(())
        );
        assert_eq!(alone[0].num_folds_engaged, NumFolds::ONE);

        let mut with_pt = [compute(SenComponent::Pe0), compute(SenComponent::Ptrow0)];
        assert_eq!(
            set_pe_folds_if_pt_interaction::<Sen1p5>(&mut with_pt),
            Some(())
        );
        assert_eq!(with_pt[0].num_folds_engaged, NumFolds(2));
        assert_eq!(with_pt[1].num_folds_engaged, NumFolds::ONE);
    }
}
