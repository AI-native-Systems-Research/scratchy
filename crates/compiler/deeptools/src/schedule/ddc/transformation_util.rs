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

//! `ddc/ddc_transformation_util.cpp` — 32 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e110_constructAllocation` | 110 | 0 | 50 | `Ddc` | `ddc/ddc_transformation_util.cpp:20` |
//! | `e111_reduceUsersOrDeleteAllocationAndMetadata` | 111 | 0 | 41 | `Ddc` | `ddc/ddc_transformation_util.cpp:73` |
//! | `e112_constructDatastage` | 112 | 0 | 8 | `Ddc` | `ddc/ddc_transformation_util.cpp:117` |
//! | `e113_constructDatastage` | 113 | 0 | 11 | `Ddc` | `ddc/ddc_transformation_util.cpp:126` |
//! | `e114_constructLoopNode` | 114 | 0 | 18 | `Ddc` | `ddc/ddc_transformation_util.cpp:138` |
//! | `e115_collectLoopReferences` | 115 | 0 | 14 | `Ddc` | `ddc/ddc_transformation_util.cpp:318` |
//! | `e116_getPaddingPerDim` | 116 | 0 | 49 | `Ddc` | `ddc/ddc_transformation_util.cpp:709` |
//! | `e117_getNodeDescription` | 117 | 0 | 25 | `Ddc` | `ddc/ddc_transformation_util.cpp:1094` |
//! | `e118_cloneForPeSfpWorkSplit` | 118 | 0 | 114 | `Ddc` | `ddc/ddc_transformation_util.cpp:1306` |
//! | `e119_cloneForPeSfpWorkSplit` | 119 | 0 | 115 | `Ddc` | `ddc/ddc_transformation_util.cpp:1422` |
//! | `e120_storageOrDatastreamIsExternal` | 120 | 0 | 32 | `Ddc` | `ddc/ddc_transformation_util.cpp:1652` |
//! | `e121_relatedToExternalNodes` | 121 | 0 | 8 | `Ddc` | `ddc/ddc_transformation_util.cpp:1748` |
//! | `e122_isNodeRelatedToComps` | 122 | 0 | 30 | `Ddc` | `ddc/ddc_transformation_util.cpp:1778` |
//! | `e123_updateNodesWithNewLds` | 123 | 0 | 57 | `Ddc` | `ddc/ddc_transformation_util.cpp:1919` |
//! | `e247_splitLoopBandOnDim` | 247 | 1 | 123 | `Ddc` | `ddc/ddc_transformation_util.cpp:159` |
//! | `e248_splitLoopBandOnDatastage` | 248 | 1 | 30 | `Ddc` | `ddc/ddc_transformation_util.cpp:287` |
//! | `e249_moveTransferNode` | 249 | 1 | 372 | `Ddc` | `ddc/ddc_transformation_util.cpp:335` |
//! | `e250_convertResultFromFIFOtoReg` | 250 | 1 | 148 | `Ddc` | `ddc/ddc_transformation_util.cpp:760` |
//! | `e251_unrollTransfer` | 251 | 1 | 72 | `Ddc` | `ddc/ddc_transformation_util.cpp:1120` |
//! | `e252_unrollTransferForSymbolicDims` | 252 | 1 | 110 | `Ddc` | `ddc/ddc_transformation_util.cpp:1193` |
//! | `e253_srcRelatedToExternalNodes` | 253 | 1 | 4 | `Ddc` | `ddc/ddc_transformation_util.cpp:1687` |
//! | `e254_destRelatedToExternalNodes` | 254 | 1 | 5 | `Ddc` | `ddc/ddc_transformation_util.cpp:1693` |
//! | `e255_inputRelatedToExternalNodes` | 255 | 1 | 11 | `Ddc` | `ddc/ddc_transformation_util.cpp:1717` |
//! | `e256_outputRelatedToExternalNodes` | 256 | 1 | 11 | `Ddc` | `ddc/ddc_transformation_util.cpp:1730` |
//! | `e257_addNewLds` | 257 | 1 | 107 | `Ddc` | `ddc/ddc_transformation_util.cpp:1811` |
//! | `e304_cloneForPeSfpWorkSplit` | 304 | 2 | 113 | `Ddc` | `ddc/ddc_transformation_util.cpp:1538` |
//! | `e305_destRelatedToExternalNodes` | 305 | 2 | 10 | `Ddc` | `ddc/ddc_transformation_util.cpp:1700` |
//! | `e306_relatedToExternalNodes` | 306 | 2 | 4 | `Ddc` | `ddc/ddc_transformation_util.cpp:1743` |
//! | `e339_convertResultToSkipReg` | 339 | 3 | 110 | `Ddc` | `ddc/ddc_transformation_util.cpp:909` |
//! | `e340_insertComputeBetweenTransferAndReg` | 340 | 3 | 69 | `Ddc` | `ddc/ddc_transformation_util.cpp:1021` |
//! | `e341_relatedToExternalNodes` | 341 | 3 | 4 | `Ddc` | `ddc/ddc_transformation_util.cpp:1712` |
//! | `e361_relatedToExternalNodes` | 361 | 4 | 20 | `Ddc` | `ddc/ddc_transformation_util.cpp:1757` |

// ⭐ USES FOR ENTRIES 110-117. Union these into this file's top block when its other entries land.
use std::collections::{BTreeMap, BTreeSet};

use super::fold::{AllocId, AllocLayout, Allocations, DataOrigin, NodeId, PadType, StoredStream};
use super::metadata::{DatastageId, DdcMemory, MetaDimKind, Metadata};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
use crate::generated::DataConnect;
use crate::schedule::dsc2::{ComputeNode, Dsc, LdsIdx, NodeName, TransferNode, TransferSide};

// ⭐ TYPES FOR ENTRIES 110-117. Union this section with this file's other vocabulary when its other
// entries land; each declaration is one C++ one narrowed to what the ported entries read and write.

/// HOW EACH DIM OF AN ALLOCATION IS PADDED — `PaddingFormType` (`dsc/dims.h:94`), a
/// `map<PrimaryDimTypes, PadType>` whose `getPadding` answers `NOPAD` for a dim it has no entry for
/// (`dsc/dims.cpp:806`) and whose `setPadding` is an `insert_or_assign`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PaddingForm(BTreeMap<PrimaryDim, PadType>);

impl PaddingForm {
    /// `getPadding(dim)` — total, because absent IS [`PadType::NoPad`].
    #[must_use]
    pub fn padding(&self, dim: PrimaryDim) -> PadType {
        self.0.get(&dim).copied().unwrap_or(PadType::NoPad)
    }

    /// `setPadding(dim, pad)`.
    pub fn set_padding(&mut self, dim: PrimaryDim, pad: PadType) {
        self.0.insert(dim, pad);
    }

    /// Every dim this form states a padding for, in `PrimaryDimTypes` order.
    pub fn stated(&self) -> impl Iterator<Item = (PrimaryDim, PadType)> + '_ {
        self.0.iter().map(|(dim, pad)| (*dim, *pad))
    }
}

/// `EnumsConversion::senComponentsToString.at(comp)` for the eight `ddc::memories`
/// (`sys-arch-spec/arch_enums.cpp:11-113`, `ddc/ddc_metadata.h:19`).
const fn memory_spelling(memory: DdcMemory) -> &'static str {
    match memory {
        DdcMemory::Lx => "lx",
        DdcMemory::L0 => "l0",
        DdcMemory::PeLrf => "pelrf",
        DdcMemory::SfpLrf => "sfplrf",
        DdcMemory::PtaRf => "ptarf",
        DdcMemory::PtxRf => "ptxrf",
        DdcMemory::PtiRf => "ptirf",
        DdcMemory::Hbm => "hbm",
    }
}

/// `dsc2::AllocateNode` (`dsc/dsc2.h:974`) AS ENTRIES 110 AND 111 BUILD AND UNBUILD ONE.
///
/// ⛔ A DIFFERENT PROJECTION FROM [`crate::schedule::dsc2::AllocateNode`], which narrows the same
/// C++ struct to the component and lds the fold units read; this one carries the layout order, the
/// padding form and the user refcounts, which those never touch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DdcAllocateNode {
    /// `name_`, minted `allocate_lds<idx>_<memory>`.
    pub name: NodeName,
    /// `ldsIdx_`/`constIdx_`, exactly one of which is set.
    pub origin: DataOrigin,
    /// `component_`, a MEMORY BY TYPE — `ddc::memories` (`ddc/ddc_metadata.h:19`), which is what
    /// makes *"Requested storage <c> is not a memory"* unspellable.
    pub component: DdcMemory,
    /// `layoutDimOrder_` zipped with `maxDimSizes_`, whose fresh entries are the reference's `-1`.
    pub layout: AllocLayout,
    /// `padding_`.
    pub padding: PaddingForm,
    /// `allocUsers_` (`dsc/dsc2.h:1010`) — each user to its reference count, a map because the
    /// reference's vector of pairs is only ever searched by node.
    pub alloc_users: BTreeMap<NodeId, u32>,
}

impl DdcAllocateNode {
    /// `addAllocUser(userNode)` (`dsc/dsc2.h:1012`) — a repeat user bumps its count.
    pub fn add_alloc_user(&mut self, user: NodeId) {
        *self.alloc_users.entry(user).or_insert(0) += 1;
    }
}

/// WHETHER AN UNUSED ALLOCATION MAY LEAVE THE TREE — `reduceUsersOrDeleteAllocation`'s `canDelete`
/// (`dsc/dsc2.cpp:2495`), which entry 111 sets from `!isExternalNode(allocNode)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanDelete {
    /// The node is not external, so an allocation with no users left is deleted.
    Yes,
    /// The node is external and stays in the tree however few users it has.
    No,
}

/// A LABELED DS PROVED READY FOR A NEW ALLOCATION IN ONE MEMORY, WITH THE LAYOUT ORDER IT GETS.
///
/// ⛔⛔ THREE OF ENTRY 110'S REFUSALS COLLAPSE INTO THIS TYPE: *"LabeldDs[i] already has an
/// allocation for memOrg_ entry"*, `DT_CHECK(allocMeta.find(ldsIdx) == allocMeta.end())`, and
/// *"Handling of external allocations with repeated dimensions is not yet implemented"*. Its other
/// two are gone by the argument types instead — [`LdsIdx`] and [`DdcMemory`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshAllocation {
    lds: LdsIdx,
    storage: DdcMemory,
    layout: AllocLayout,
}

impl FreshAllocation {
    /// The witness, or [`None`] for any of the three refusals.
    #[must_use]
    pub fn of<D: DscAllocations + ?Sized>(
        dsc: &D,
        metadata: &Metadata,
        lds: LdsIdx,
        storage: DdcMemory,
    ) -> Option<Self> {
        if dsc
            .allocation_in(DataOrigin::LabeledDs(lds), storage)
            .is_some()
        {
            return None;
        }
        if metadata
            .new_allocations
            .get(&storage)
            .is_some_and(|allocated| allocated.lds_idx_and_alloc_node.contains_key(&lds))
        {
            return None;
        }
        let dims = dsc.layout_dims(dsc.reference_lds(lds)).to_vec();
        let distinct: BTreeSet<PrimaryDim> = dims.iter().copied().collect();
        (distinct.len() == dims.len()).then(|| Self {
            lds,
            storage,
            layout: AllocLayout(dims.into_iter().map(|dim| (dim, None)).collect()),
        })
    }
}

/// A DATASTREAM PROVED TO HAVE AN ALLOCATION IN ONE MEMORY *AND* A NAMED USER OF IT.
///
/// ⛔⛔ TWO REFUSALS COLLAPSE INTO THIS TYPE. Entry 111's *"LabeldDs[i]'s memOrg_ entry for
/// DataLocation storge <c> does not have any allocation"* is the missing allocation, and
/// `removeAllocUser`'s *"Schedule node <n> is not in the user list of allocate node <a>"*
/// (`dsc/dsc2.h:1032`) is the absent user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationUse {
    origin: DataOrigin,
    storage: DdcMemory,
    alloc: AllocId,
    user: NodeId,
}

impl AllocationUse {
    /// The witness, or [`None`] for either refusal.
    #[must_use]
    pub fn of<D: DscAllocations + ?Sized>(
        dsc: &D,
        origin: DataOrigin,
        storage: DdcMemory,
        user: NodeId,
    ) -> Option<Self> {
        let alloc = dsc.allocation_in(origin, storage)?;
        dsc.alloc_users(alloc).contains(&user).then_some(Self {
            origin,
            storage,
            alloc,
            user,
        })
    }

    /// The allocation the use names.
    #[must_use]
    pub const fn alloc(self) -> AllocId {
        self.alloc
    }

    /// Which labeled DS or constant reached it.
    #[must_use]
    pub const fn origin(self) -> DataOrigin {
        self.origin
    }

    /// Which memory.
    #[must_use]
    pub const fn storage(self) -> DdcMemory {
        self.storage
    }

    /// The user whose reference is being dropped.
    #[must_use]
    pub const fn user(self) -> NodeId {
        self.user
    }
}

/// WHAT ENTRIES 110 AND 111 ASK OF A `DesignSpaceConfig` — the labeled-DS side of an allocation.
///
/// ⭐ REACHING THE ALLOCATION IS THE MECHANISM, NOT THE FACT. `getAllocation`
/// (`dsc/dsc2.cpp:2586`) walks `memOrg_`, `getMutableAllocation` is its `const_cast` (`:2635`) and
/// `reduceUsersOrDeleteAllocation` (`:2495`) does the tree-side unlink; what these two entries want
/// is the allocation, and whether it left the tree.
pub trait DscAllocations: Dsc {
    /// `labeledDs_.at(lds).ldsIdx_` (`dsc/dscdefn.h:321`) — the ENTRY'S OWN self-index, which is
    /// what `getLayoutDims` is keyed by and need not be the index it was looked up under.
    fn reference_lds(&self, lds: LdsIdx) -> LdsIdx;

    /// `getAllocation(di, storage, /*allowMissingAlloc=*/true)`.
    fn allocation_in(&self, origin: DataOrigin, storage: DdcMemory) -> Option<AllocId>;

    /// `labeledDs_.at(lds).memOrg_[storage] = { isPresent = true, allocateNode_ = alloc }`
    /// (`dsc/dscdefn.h:305`).
    fn set_allocation_in(&mut self, lds: LdsIdx, storage: DdcMemory, alloc: AllocId);

    /// The nodes in `allocUsers_` (`dsc/dsc2.h:1010`).
    fn alloc_users(&self, alloc: AllocId) -> Vec<NodeId>;

    /// `allocNode->component_`.
    fn alloc_component(&self, alloc: AllocId) -> DdcMemory;

    /// `allocNode->ldsIdx_`/`constIdx_`.
    fn alloc_origin(&self, alloc: AllocId) -> DataOrigin;

    /// THE ALLOCATION AS A SCHEDULE NODE — one pointer in the reference, which is what lets
    /// `isExternalNode(allocNode)` (`ddc/ddc.h:298`) take it; [`AllocId`] and [`NodeId`] are two
    /// newtypes over that single identity.
    fn alloc_node(&self, alloc: AllocId) -> NodeId;

    /// `reduceUsersOrDeleteAllocation(di, storage, userNode, canDelete)` (`dsc/dsc2.cpp:2495`) —
    /// drops one of the user's references and, once none remain and deletion is permitted, unlinks
    /// the node from the schedule tree and clears the `memOrg_`/`constantInfo_` slot holding it.
    /// `true` when the node LEFT THE TREE, which is the only case entry 111 acts on.
    fn reduce_users_or_delete(&mut self, alloc_use: AllocationUse, can_delete: CanDelete) -> bool;
}

/// ONE `DataStructDims` (`dsc/dims.h:158`) — its `name_` and its extents, whatever those are.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StageDims<D> {
    /// `name_`.
    pub name: StageName,
    /// Every extent the stage states; the payload entries 112 and 113 carry through untouched.
    pub dims: D,
}

/// A DATA STAGE'S NAME — `DataStructDims::name_`, which entries 112 and 113 mint by concatenation
/// (`<id>` and `<id>el`), so it is a name and not a closed set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StageName(pub String);

/// `dsc2::DataStage` (`dsc/dsc2.h:40`) — the steady-state and epilogue halves of one staging
/// parameter set.
///
/// ⭐ GENERIC IN THE EXTENTS PAYLOAD ON PURPOSE: entry 113 mints a stage by COPYING a reference
/// one, and a projection that dropped the extents would make it entry 112 spelled twice.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataStage<D> {
    /// `ss_` — the steady-state dims.
    pub ss: StageDims<D>,
    /// `el_` — the epilogue dims.
    pub el: StageDims<D>,
}

impl<D> DataStage<D> {
    /// `ss_.name_ = to_string(id); el_.name_ = to_string(id) + "el";`
    pub fn rename_for(&mut self, id: DatastageId) {
        self.ss.name = StageName(id.0.to_string());
        self.el.name = StageName(format!("{}el", id.0));
    }
}

/// `DesignSpaceConfig::dataStageParam_` (`dsc/designSpaceConfig.h:105`) — the DSC's data stages by
/// id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataStages<D>(pub BTreeMap<DatastageId, DataStage<D>>);

impl<D> Default for DataStages<D> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<D> DataStages<D> {
    /// `auto id = dataStageParam_.size(); while (dataStageParam_.count(id)) id++;`
    ///
    /// ⚠️ NOT `max + 1`, AND NOT THE LOWEST FREE ID: the search STARTS at the map's size, so over
    /// keys {0, 2} it answers 3 and leaves 1 free for good.
    #[must_use]
    pub fn free_id(&self) -> DatastageId {
        let mut id = DatastageId(self.0.len() as u32);
        while self.0.contains_key(&id) {
            id.0 += 1;
        }
        id
    }
}

/// ONE LOOP DIM AND WHAT KIND OF EXTENT IT WALKS — `PrimaryDimAndKind` (`dsc/dims.h:76`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimaryDimAndKind {
    /// `dim_`, whose `PrimaryDimTypesCount` default this type does not admit.
    pub dim: PrimaryDim,
    /// `kind_`.
    pub kind: MetaDimKind,
}

/// A LOOP'S DIMS, NON-EMPTY — entry 114's *"Cannot construct loop with no dimensions"*
/// (`ddc/ddc_transformation_util.cpp:141`) made unspellable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopDims {
    first: PrimaryDimAndKind,
    rest: Vec<PrimaryDimAndKind>,
}

impl LoopDims {
    /// A loop has at least one dim, and this is how that is stated.
    #[must_use]
    pub const fn new(first: PrimaryDimAndKind, rest: Vec<PrimaryDimAndKind>) -> Self {
        Self { first, rest }
    }

    /// The dims in order — which is the order the loop's name spells them in.
    pub fn iter(&self) -> impl Iterator<Item = PrimaryDimAndKind> + '_ {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }
}

/// `dsc2::LoopNode` (`dsc/dsc2.h:563`) as entry 114 mints one.
///
/// ⛔ `numId_`/`denId_` ARE NOT OPTIONAL HERE: their `-1` is what an unbuilt loop carries, and every
/// callsite of the constructor passes a real pair (`ddc/ddc_transformation_util.cpp:233`, `:299`,
/// `:1186`, `:1299`; `ddc/ddc_transformation.cpp:980`, `:1047`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopNode {
    /// `name_`, minted `loop_ds<num>_ds<den>` then `_<dim>` per dim.
    pub name: NodeName,
    /// `numId_` — the numerator data stage.
    pub num: DatastageId,
    /// `denId_` — the denominator data stage.
    pub den: DatastageId,
    /// `dims_`.
    pub dims: LoopDims,
}

/// ONE AND-CLAUSE TERM OF A LOOP CONDITION — `dsc2::LoopCond` (`dsc/dsc2.h:654`) reduced to the
/// loop it compares against, which is all entry 115 reads.
///
/// ⛔ `loopComp_`'s `nullptr` DEFAULT IS NOT ADMITTED: entry 115 inserts it straight into
/// `referencedLoops`, so a null there is a defect and not a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopCond {
    /// `loopComp_`.
    pub loop_node: NodeId,
}

/// A CONDITION NODE'S LOOP CONDITION — `LoopCondComposite::twoLevelOrOfAnds_` (`dsc/dsc2.h:675`),
/// an OR of ANDs of per-loop terms.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoopCondComposite {
    /// `twoLevelOrOfAnds_`, EMPTY exactly when the node carries the core/corelet condition instead
    /// — that is the whole of `hasCoreClCond()` (`dsc/dsc2.h:692`).
    pub or_of_ands: Vec<Vec<LoopCond>>,
}

/// A SCHEDULE NODE AS ENTRIES 115 AND 117 SEE ONE — the kinds either looks inside, and
/// [`UtilNode::Other`] for every kind neither does.
///
/// ⛔ NOT [`crate::schedule::dsc2::Node`]: that enum has exactly the three arms the fold units
/// accept and cannot spell a LOOP, SYNC, BLOCK or CONDITION node at all, and these two entries are
/// called across the whole tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilNode<'a> {
    /// `TRANSFER`.
    Transfer(&'a TransferNode),
    /// `COMPUTE`.
    Compute(&'a ComputeNode),
    /// `CONDITION` — its name and its loop condition.
    Condition(&'a NodeName, &'a LoopCondComposite),
    /// Every other `nodeType_`, which both entries reduce to its `name_`.
    Other(&'a NodeName),
}

impl UtilNode<'_> {
    /// `ScheduleNode::name_`.
    #[must_use]
    pub const fn name(&self) -> &NodeName {
        match self {
            Self::Transfer(node) => &node.name,
            Self::Compute(node) => &node.name,
            Self::Condition(name, _) | Self::Other(name) => name,
        }
    }
}

/// `DataInfo::dataConnect_` AS THE REFERENCE CONCATENATES IT — the EMPTY STRING where it is unset,
/// which is what a default-constructed `DataInfo` carries.
fn connect_spelling(connect: Option<DataConnect>) -> &'static str {
    connect.map_or("", DataConnect::spelling)
}

/// A TRANSFER'S ENDS AS ALLOCATION LOOKUPS — `srcLdsAndLoopOffsets_` with `src_.storage_`, and each
/// `dstLdsAndLoopOffsets_.at(i)` with `dstVias_.at(i).loc_.storage_`, ZIPPED so the reference's two
/// independently-sized vectors cannot disagree and its `.at(i)` cannot throw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferEnds {
    /// The source end.
    pub src: StoredStream,
    /// The destination ends, in order.
    pub dsts: Vec<StoredStream>,
}

/// `getAllocation` PLUS THE PADDING FORM ON WHAT IT FINDS — `AllocateNode::padding_`
/// (`dsc/dsc2.h:981`), which [`Allocations`] alone does not carry.
pub trait AllocationPaddings: Allocations {
    /// `allocNode->padding_`.
    fn padding(&self, alloc: AllocId) -> PaddingForm;
}

/// Replaces: e110_constructAllocation
///
/// MINTS THE ALLOCATE NODE for one labeled DS in one memory: names it `allocate_lds<idx>_<memory>`,
/// registers it in `metadata.new_allocations[memory]` and in the labeled DS's own `memOrg_`, and
/// records `user_node` as its first user (`ddc/ddc_transformation_util.cpp:20`).
///
/// ⚠️ TRAP: the layout order comes from the labeled DS entry's OWN `ldsIdx_`
/// ([`DscAllocations::reference_lds`]), NOT from the index it was looked up under.
/// ⭐ `alloc` IS THE IDENTITY ITS OWNER ISSUES — `new dsc2::AllocateNode()` in the reference, whose
/// pointer both names the node and is the key both registries hold it under.
pub fn construct_allocation<D: DscAllocations + ?Sized>(
    dsc: &mut D,
    metadata: &mut Metadata,
    fresh: FreshAllocation,
    padding_per_dim: PaddingForm,
    user_node: NodeId,
    alloc: AllocId,
) -> DdcAllocateNode {
    let FreshAllocation {
        lds,
        storage,
        layout,
    } = fresh;

    let mut node = DdcAllocateNode {
        name: NodeName(format!(
            "allocate_lds{}_{}",
            lds.0,
            memory_spelling(storage)
        )),
        origin: DataOrigin::LabeledDs(lds),
        component: storage,
        layout,
        padding: padding_per_dim,
        alloc_users: BTreeMap::new(),
    };

    metadata
        .new_allocations
        .entry(storage)
        .or_default()
        .lds_idx_and_alloc_node
        .insert(lds, alloc);
    dsc.set_allocation_in(lds, storage, alloc);
    node.add_alloc_user(user_node);

    node
}

/// Replaces: e111_reduceUsersOrDeleteAllocationAndMetadata
///
/// Drops one user's reference to an allocation and, when that took the node out of the schedule
/// tree, erases it from `metadata.new_allocations` too (`:73`).
///
/// ⚠️ TRAP: the metadata keys are the ALLOCATION'S OWN component and origin, not the caller's `di`.
/// ⛔ DELIBERATE DIVERGENCE, and the defect is the reference's: it folds both indices into one
/// `int` and erases that from `ldsIdxAndAllocNode`, so a CONSTANT allocation either aborts on
/// *"Missing metadata for allocateNode"* or erases an unrelated labeled DS. `consIdAndAllocNode`
/// exists for exactly those (`:1356`), so a constant is erased from it; the mixed-key state cannot
/// be spelled here. Its two *"Missing metadata"* aborts are then a registry with nothing to erase,
/// which entry 110 — the only writer of `newAllocations_`, and it always registers — rules out.
pub fn reduce_users_or_delete_allocation_and_metadata<D: DscAllocations + ?Sized>(
    dsc: &mut D,
    metadata: &mut Metadata,
    alloc_use: AllocationUse,
) {
    let alloc = alloc_use.alloc();
    let component = dsc.alloc_component(alloc);
    let origin = dsc.alloc_origin(alloc);
    let can_delete = if metadata.external_nodes.contains(&dsc.alloc_node(alloc)) {
        CanDelete::No
    } else {
        CanDelete::Yes
    };

    if !dsc.reduce_users_or_delete(alloc_use, can_delete) {
        return;
    }

    if let Some(allocated) = metadata.new_allocations.get_mut(&component) {
        match origin {
            DataOrigin::LabeledDs(lds) => allocated.lds_idx_and_alloc_node.remove(&lds),
            DataOrigin::Constant(cons) => allocated.cons_id_and_alloc_node.remove(&cons),
        };
    }
}

/// Replaces: e112_constructDatastage
///
/// Mints an EMPTY data stage under the first free id and names its two halves `<id>` and `<id>el`
/// (`:117`).
///
/// ⚠️ TRAP: the id is [`DataStages::free_id`] — the search starts at the map's SIZE, so it is
/// neither `max + 1` nor the lowest free id.
pub fn construct_datastage<D: Default>(stages: &mut DataStages<D>) -> DatastageId {
    let id = stages.free_id();
    stages.0.entry(id).or_default().rename_for(id);
    id
}

/// Replaces: e113_constructDatastage
///
/// Mints a data stage that is a COPY of a reference one under the first free id, then renames its
/// two halves (`:126`).
///
/// ⚠️ ONLY THE TWO NAMES ARE OVERWRITTEN — the reference's `emplace` copies, so every extent of
/// `reference` survives into the new stage. Same id trap as [`construct_datastage`].
pub fn construct_datastage_from<D: Clone>(
    stages: &mut DataStages<D>,
    reference: &DataStage<D>,
) -> DatastageId {
    let id = stages.free_id();
    stages
        .0
        .entry(id)
        .or_insert_with(|| reference.clone())
        .rename_for(id);
    id
}

/// Replaces: e114_constructLoopNode
///
/// MINTS THE LOOP NODE for a numerator/denominator data-stage pair, named `loop_ds<num>_ds<den>`
/// followed by `_<dim>` over its dims in order (`:138`).
///
/// ⚠️ TRAP: `baseNode` IS NEVER READ. The reference takes it and neither links the new loop to it
/// nor consults it, so the mint is unparented and placing it is the caller's own step.
#[must_use]
pub fn construct_loop_node(num: DatastageId, den: DatastageId, dims: LoopDims) -> LoopNode {
    let mut name = format!("loop_ds{}_ds{}", num.0, den.0);
    for entry in dims.iter() {
        name.push('_');
        name.push_str(entry.dim.spelling());
    }

    LoopNode {
        name: NodeName(name),
        num,
        den,
        dims,
    }
}

/// Replaces: e115_collectLoopReferences
///
/// Adds every loop that any and-clause of a CONDITION node's loop condition compares against; every
/// other node kind contributes nothing (`:318`).
///
/// ⚠️ TRAP: the `hasCoreClCond()` early return IS `twoLevelOrOfAnds_.empty()` (`dsc/dsc2.h:692`),
/// so all it can ever skip is an empty walk — the guard filters no clause.
pub fn collect_loop_references(node: UtilNode<'_>, referenced_loops: &mut BTreeSet<NodeId>) {
    if let UtilNode::Condition(_, cond) = node {
        for and_clause in &cond.or_of_ands {
            for term in and_clause {
                referenced_loops.insert(term.loop_node);
            }
        }
    }
}

/// Replaces: e116_getPaddingPerDim
///
/// The padding form for one side of a transfer: the padding of the first allocation found on that
/// side, falling back to the other side, then overwritten per dim by whatever access pattern the
/// metadata states for the transfer (`:709`).
///
/// ⚠️ THE DESTINATION SCAN TAKES THE FIRST DESTINATION THAT HAS AN ALLOCATION, not one that matches
/// anything; with no allocation on either side the form is empty, which reads `NOPAD` everywhere.
#[must_use]
pub fn get_padding_per_dim<A: AllocationPaddings + ?Sized>(
    allocs: &A,
    metadata: &Metadata,
    transfer: NodeId,
    ends: &TransferEnds,
    side: TransferSide,
) -> PaddingForm {
    let dst_alloc = || ends.dsts.iter().find_map(|dst| allocs.allocation(*dst));
    let reference = match side {
        TransferSide::Src => allocs.allocation(ends.src).or_else(dst_alloc),
        TransferSide::Dst => dst_alloc().or_else(|| allocs.allocation(ends.src)),
    };

    let mut result = reference.map_or_else(PaddingForm::default, |alloc| allocs.padding(alloc));

    if let Some(transfer_meta) = metadata.datatransfers.get(&transfer) {
        for (dim, pattern) in &transfer_meta.access_pattern_per_dim {
            result.set_padding(
                *dim,
                match side {
                    TransferSide::Src => pattern.from,
                    TransferSide::Dst => pattern.to,
                },
            );
        }
    }

    result
}

/// Replaces: e117_getNodeDescription
///
/// The node's name, plus its data connects in brackets: ` [ src: <dc> dst: <dc>.. ]` for a TRANSFER
/// and ` [ input: <dc>.. output: <dc>..]` for a COMPUTE (`:1094`).
///
/// ⚠️ THE SPACING IS THE REFERENCE'S: every listed connect is preceded by a space, the transfer arm
/// closes with `" ]"` and the compute arm with `"]"`, and an unset `dataConnect_` contributes only
/// its separator.
#[must_use]
pub fn get_node_description(node: UtilNode<'_>) -> String {
    let mut description = node.name().0.clone();

    match node {
        UtilNode::Transfer(transfer) => {
            description.push_str(" [ src: ");
            description.push_str(connect_spelling(transfer.src.data.data_connect));
            description.push_str(" dst: ");
            for dst in transfer.dsts.iter() {
                description.push(' ');
                description.push_str(connect_spelling(dst.data.data_connect));
            }
            description.push_str(" ]");
        }
        UtilNode::Compute(compute) => {
            description.push_str(" [ input: ");
            for input in &compute.inputs {
                description.push(' ');
                description.push_str(connect_spelling(input.data.data_connect));
            }
            description.push_str(" output: ");
            for output in &compute.outputs {
                description.push(' ');
                description.push_str(connect_spelling(output.data.data_connect));
            }
            description.push(']');
        }
        UtilNode::Condition(..) | UtilNode::Other(_) => {}
    }

    description
}

// crustify:todo: e118_cloneForPeSfpWorkSplit
//   authority : ddc/ddc_transformation_util.cpp:1306  (114 body lines, level 0)
//   class     : Ddc
//   original  : dsc2::AllocateNode *Ddc::cloneForPeSfpWorkSplit( dsc2::AllocateNode *node, bool skipMetadataUpdate /* = false */)
//   extract   : crustify-ddc/cpp/ddc.cpp:1523-1638

// crustify:todo: e119_cloneForPeSfpWorkSplit
//   authority : ddc/ddc_transformation_util.cpp:1422  (115 body lines, level 0)
//   class     : Ddc
//   original  : dsc2::ComputeNode *Ddc::cloneForPeSfpWorkSplit(dsc2::ComputeNode *node)
//   extract   : crustify-ddc/cpp/ddc.cpp:1648-1763

// crustify:todo: e120_storageOrDatastreamIsExternal
//   authority : ddc/ddc_transformation_util.cpp:1652  (32 body lines, level 0)
//   class     : Ddc
//   original  : bool Ddc::storageOrDatastreamIsExternal(const dsc2::DataInfo &dataInfo, SenComponents storage, bool isIncoming) const
//   extract   : crustify-ddc/cpp/ddc.cpp:1773-1807

// crustify:todo: e121_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1748  (8 body lines, level 0)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::SyncNode *node) const
//   extract   : crustify-ddc/cpp/ddc.cpp:1817-1825

// crustify:todo: e122_isNodeRelatedToComps
//   authority : ddc/ddc_transformation_util.cpp:1778  (30 body lines, level 0)
//   class     : Ddc
//   original  : bool Ddc::isNodeRelatedToComps(const dsc2::ScheduleNode *node, const std::vector<SenComponents> &comps, SenComponents &nodeComp)
//   extract   : crustify-ddc/cpp/ddc.cpp:1835-1867

// crustify:todo: e123_updateNodesWithNewLds
//   authority : ddc/ddc_transformation_util.cpp:1919  (57 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::updateNodesWithNewLds(int newLdsIdx, int oldLdsIdx, dsc2::ScheduleNode *startNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:1877-1935

// crustify:todo: e247_splitLoopBandOnDim
//   authority : ddc/ddc_transformation_util.cpp:159  (123 body lines, level 1)
//   class     : Ddc
//   original  : dsc2::LoopNode *Ddc::splitLoopBandOnDim( dsc2::LoopNode *baseLoop, const std::vector<std::vector<PrimaryDimAndKind>> &inputSplitDimSetsOuterToInner, bool unspecifiedDimsInnermost)
//   extract   : crustify-ddc/cpp/ddc.cpp:5086-5213
//   calls     : e114_constructLoopNode

// crustify:todo: e248_splitLoopBandOnDatastage
//   authority : ddc/ddc_transformation_util.cpp:287  (30 body lines, level 1)
//   class     : Ddc
//   original  : dsc2::LoopNode *Ddc::splitLoopBandOnDatastage(dsc2::LoopNode *baseLoop)
//   extract   : crustify-ddc/cpp/ddc.cpp:5223-5253
//   calls     : e112_constructDatastage, e113_constructDatastage, e114_constructLoopNode

// crustify:todo: e249_moveTransferNode
//   authority : ddc/ddc_transformation_util.cpp:335  (372 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::moveTransferNode(dsc2::TransferNode *transferNode, dsc2::LoopNode *newParentLoop)
//   extract   : crustify-ddc/cpp/ddc.cpp:5263-5636
//   calls     : e104_clear

// crustify:todo: e250_convertResultFromFIFOtoReg
//   authority : ddc/ddc_transformation_util.cpp:760  (148 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::convertResultFromFIFOtoReg(dsc2::TransferNode *transferNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:5646-5794
//   calls     : e110_constructAllocation, e116_getPaddingPerDim

// crustify:todo: e251_unrollTransfer
//   authority : ddc/ddc_transformation_util.cpp:1120  (72 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::unrollTransfer(dsc2::TransferNode *transferNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:5804-5876
//   calls     : e104_clear, e112_constructDatastage, e113_constructDatastage, e114_constructLoopNode

// crustify:todo: e252_unrollTransferForSymbolicDims
//   authority : ddc/ddc_transformation_util.cpp:1193  (110 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::unrollTransferForSymbolicDims( dsc2::TransferNode *transferNode, const std::map<PrimaryDimTypes, SymbolicDimInfo> &symbolicDims)
//   extract   : crustify-ddc/cpp/ddc.cpp:5886-5998
//   calls     : e112_constructDatastage, e113_constructDatastage, e114_constructLoopNode

// crustify:todo: e253_srcRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1687  (4 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::srcRelatedToExternalNodes( const dsc2::TransferNode *transferNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6008-6013
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e254_destRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1693  (5 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::destRelatedToExternalNodes(const dsc2::TransferNode *transferNode, int dstIndex) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6023-6029
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e255_inputRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1717  (11 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::inputRelatedToExternalNodes( const dsc2::ComputeNode *computeNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6039-6051
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e256_outputRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1730  (11 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::outputRelatedToExternalNodes( const dsc2::ComputeNode *computeNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6061-6073
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e257_addNewLds
//   authority : ddc/ddc_transformation_util.cpp:1811  (107 body lines, level 1)
//   class     : Ddc
//   original  : int Ddc::addNewLds(LabeledDsInfo *refLds)
//   extract   : crustify-ddc/cpp/ddc.cpp:6083-6190
//   calls     : e104_clear

// crustify:todo: e304_cloneForPeSfpWorkSplit
//   authority : ddc/ddc_transformation_util.cpp:1538  (113 body lines, level 2)
//   class     : Ddc
//   original  : dsc2::TransferNode *Ddc::cloneForPeSfpWorkSplit(dsc2::TransferNode *node)
//   extract   : crustify-ddc/cpp/ddc.cpp:9878-9991
//   calls     : e251_unrollTransfer

// crustify:todo: e305_destRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1700  (10 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::destRelatedToExternalNodes( const dsc2::TransferNode *transferNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:10001-10012
//   calls     : e254_destRelatedToExternalNodes

// crustify:todo: e306_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1743  (4 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::ComputeNode *computeNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:10022-10026
//   calls     : e255_inputRelatedToExternalNodes, e256_outputRelatedToExternalNodes

// crustify:todo: e339_convertResultToSkipReg
//   authority : ddc/ddc_transformation_util.cpp:909  (110 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::convertResultToSkipReg(dsc2::TransferNode *transferNode, int destIndex, bool useLatch)
//   extract   : crustify-ddc/cpp/ddc.cpp:12139-12250
//   calls     : e111_reduceUsersOrDeleteAllocationAndMetadata, e254_destRelatedToExternalNodes, e305_destRelatedToExternalNodes

// crustify:todo: e340_insertComputeBetweenTransferAndReg
//   authority : ddc/ddc_transformation_util.cpp:1021  (69 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::insertComputeBetweenTransferAndReg(dsc2::TransferNode *transferNode, int transferDestIndex, dsc2::ComputeNode *computeNode, int computeInputIndex)
//   extract   : crustify-ddc/cpp/ddc.cpp:12260-12332
//   calls     : e254_destRelatedToExternalNodes, e305_destRelatedToExternalNodes

// crustify:todo: e341_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1712  (4 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::TransferNode *transferNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:12342-12346
//   calls     : e253_srcRelatedToExternalNodes, e254_destRelatedToExternalNodes, e305_destRelatedToExternalNodes

// crustify:todo: e361_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1757  (20 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::BlockNode *root) const
//   extract   : crustify-ddc/cpp/ddc.cpp:14282-14302
//   calls     : e121_relatedToExternalNodes, e306_relatedToExternalNodes, e341_relatedToExternalNodes

#[cfg(test)]
mod tests_e110_e117 {
    // ⭐ TESTS FOR ENTRIES 110-117. Union this module with this file's other test modules when they
    // land.
    use super::*;

    use sys_arch_spec::arch_enums::SenComponent;

    use super::super::metadata::{Allocation, DataTransfer, TransferAccessPattern};
    use crate::generated::ComputeType;
    use crate::schedule::ddc::fold::{ConstIdx, DataStream};
    use crate::schedule::dsc2::{DataInfo, Dsts, LayoutDims, Operand, ReplicationFactor};
    use crate::units::{DfirUnit, NumFolds};

    /// A DSC WITH ONE LABELED DS. Its own `ldsIdx_` is deliberately NOT the index it is looked up
    /// under, which is what puts entry 110's layout trap under test.
    #[derive(Debug)]
    struct TestDsc {
        lookup: LdsIdx,
        reference: LdsIdx,
        layout: LayoutDims,
        allocations: BTreeMap<(DataOrigin, DdcMemory), AllocId>,
        users: Vec<NodeId>,
        component: DdcMemory,
        origin: DataOrigin,
        left_tree: bool,
        mem_org: BTreeMap<(LdsIdx, DdcMemory), AllocId>,
        reduce_seen: Option<(DataOrigin, DdcMemory, NodeId, CanDelete)>,
    }

    impl TestDsc {
        fn new() -> Self {
            Self {
                lookup: LdsIdx(4),
                reference: LdsIdx(7),
                layout: LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Out, PrimaryDim::Mb]),
                allocations: BTreeMap::new(),
                users: Vec::new(),
                component: DdcMemory::PeLrf,
                origin: DataOrigin::LabeledDs(LdsIdx(4)),
                left_tree: true,
                mem_org: BTreeMap::new(),
                reduce_seen: None,
            }
        }
    }

    impl Dsc for TestDsc {
        fn layout_dims(&self, lds: LdsIdx) -> LayoutDims {
            if lds == self.reference {
                self.layout.clone()
            } else {
                // A repeated dim, so a lookup by the WRONG index yields no witness at all.
                LayoutDims::new(PrimaryDim::X1, vec![PrimaryDim::X1])
            }
        }
    }

    impl DscAllocations for TestDsc {
        fn reference_lds(&self, _lds: LdsIdx) -> LdsIdx {
            self.reference
        }

        fn allocation_in(&self, origin: DataOrigin, storage: DdcMemory) -> Option<AllocId> {
            self.allocations.get(&(origin, storage)).copied()
        }

        fn set_allocation_in(&mut self, lds: LdsIdx, storage: DdcMemory, alloc: AllocId) {
            self.mem_org.insert((lds, storage), alloc);
        }

        fn alloc_users(&self, _alloc: AllocId) -> Vec<NodeId> {
            self.users.clone()
        }

        fn alloc_component(&self, _alloc: AllocId) -> DdcMemory {
            self.component
        }

        fn alloc_origin(&self, _alloc: AllocId) -> DataOrigin {
            self.origin
        }

        fn alloc_node(&self, alloc: AllocId) -> NodeId {
            NodeId(alloc.0)
        }

        fn reduce_users_or_delete(
            &mut self,
            alloc_use: AllocationUse,
            can_delete: CanDelete,
        ) -> bool {
            self.reduce_seen = Some((
                alloc_use.origin(),
                alloc_use.storage(),
                alloc_use.user(),
                can_delete,
            ));
            self.left_tree
        }
    }

    /// The allocation lookup and the padding form on what it finds, for entry 116.
    #[derive(Debug, Default)]
    struct TestAllocs {
        by_stream: Vec<(StoredStream, AllocId)>,
        paddings: BTreeMap<AllocId, PaddingForm>,
    }

    impl Allocations for TestAllocs {
        fn allocation(&self, stored: StoredStream) -> Option<AllocId> {
            self.by_stream
                .iter()
                .find(|(candidate, _)| *candidate == stored)
                .map(|(_, alloc)| *alloc)
        }

        fn value_allocation(&self, _scale: AllocId) -> Option<AllocId> {
            None
        }
    }

    impl AllocationPaddings for TestAllocs {
        fn padding(&self, alloc: AllocId) -> PaddingForm {
            self.paddings.get(&alloc).cloned().unwrap_or_default()
        }
    }

    fn stored(lds: u32, storage: DfirUnit) -> StoredStream {
        StoredStream {
            stream: DataStream {
                origin: DataOrigin::LabeledDs(LdsIdx(lds)),
                data_connect: None,
            },
            storage,
        }
    }

    fn operand(connect: Option<DataConnect>) -> Operand {
        Operand {
            unit: SenComponent::Pe,
            data: DataInfo {
                data_connect: connect,
                my_lds_idx: None,
                constant_id: None,
            },
        }
    }

    #[test]
    fn an_allocation_is_named_from_its_lds_and_memory_and_reaches_both_registries() {
        let mut dsc = TestDsc::new();
        let mut metadata = Metadata::default();
        let lds = dsc.lookup;

        let fresh = FreshAllocation::of(&dsc, &metadata, lds, DdcMemory::SfpLrf)
            .expect("a free memOrg slot, a free registry slot and a distinct layout");
        let node = construct_allocation(
            &mut dsc,
            &mut metadata,
            fresh,
            PaddingForm::default(),
            NodeId(11),
            AllocId(3),
        );

        assert_eq!(node.name, NodeName("allocate_lds4_sfplrf".to_string()));
        assert_eq!(node.origin, DataOrigin::LabeledDs(lds));
        assert_eq!(node.component, DdcMemory::SfpLrf);
        // The layout came from the entry's OWN index 7, and its sizes start unset.
        assert_eq!(
            node.layout,
            AllocLayout(vec![
                (PrimaryDim::In, None),
                (PrimaryDim::Out, None),
                (PrimaryDim::Mb, None),
            ])
        );
        assert_eq!(node.alloc_users, BTreeMap::from([(NodeId(11), 1)]));
        assert_eq!(
            metadata.new_allocations[&DdcMemory::SfpLrf].lds_idx_and_alloc_node,
            BTreeMap::from([(lds, AllocId(3))])
        );
        assert_eq!(
            dsc.mem_org,
            BTreeMap::from([((lds, DdcMemory::SfpLrf), AllocId(3))])
        );

        // And a second allocation in the same memory has no witness at all.
        assert!(FreshAllocation::of(&dsc, &metadata, lds, DdcMemory::SfpLrf).is_none());
    }

    #[test]
    fn the_last_user_of_a_labeled_ds_allocation_erases_it_from_the_metadata() {
        let mut dsc = TestDsc::new();
        let lds = LdsIdx(4);
        dsc.allocations
            .insert((DataOrigin::LabeledDs(lds), DdcMemory::PeLrf), AllocId(3));
        dsc.users = vec![NodeId(11)];

        let mut metadata = Metadata::default();
        metadata.new_allocations.insert(
            DdcMemory::PeLrf,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(lds, AllocId(3))]),
                ..Allocation::default()
            },
        );

        let alloc_use = AllocationUse::of(
            &dsc,
            DataOrigin::LabeledDs(lds),
            DdcMemory::PeLrf,
            NodeId(11),
        )
        .expect("an allocation in that memory, and a node in its user list");
        reduce_users_or_delete_allocation_and_metadata(&mut dsc, &mut metadata, alloc_use);

        assert_eq!(
            dsc.reduce_seen,
            Some((
                DataOrigin::LabeledDs(lds),
                DdcMemory::PeLrf,
                NodeId(11),
                CanDelete::Yes
            ))
        );
        assert!(
            metadata.new_allocations[&DdcMemory::PeLrf]
                .lds_idx_and_alloc_node
                .is_empty()
        );
    }

    #[test]
    fn a_constant_allocation_is_erased_from_the_constant_registry_and_an_external_one_survives() {
        let mut dsc = TestDsc::new();
        let cons = ConstIdx(2);
        dsc.origin = DataOrigin::Constant(cons);
        dsc.allocations
            .insert((DataOrigin::Constant(cons), DdcMemory::PeLrf), AllocId(3));
        dsc.users = vec![NodeId(11)];

        let mut metadata = Metadata::default();
        metadata.new_allocations.insert(
            DdcMemory::PeLrf,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(LdsIdx(2), AllocId(9))]),
                cons_id_and_alloc_node: BTreeMap::from([(cons, AllocId(3))]),
                ..Allocation::default()
            },
        );

        let alloc_use = AllocationUse::of(
            &dsc,
            DataOrigin::Constant(cons),
            DdcMemory::PeLrf,
            NodeId(11),
        )
        .expect("an allocation in that memory, and a node in its user list");
        reduce_users_or_delete_allocation_and_metadata(&mut dsc, &mut metadata, alloc_use);

        // THE DIVERGENCE: the constant leaves `consIdAndAllocNode`, and the labeled DS the
        // reference would have indexed with the constant's own index is untouched.
        let allocated = &metadata.new_allocations[&DdcMemory::PeLrf];
        assert!(allocated.cons_id_and_alloc_node.is_empty());
        assert_eq!(
            allocated.lds_idx_and_alloc_node,
            BTreeMap::from([(LdsIdx(2), AllocId(9))])
        );

        // An external allocation that stayed in the tree keeps its metadata entry.
        let mut dsc = TestDsc::new();
        dsc.origin = DataOrigin::Constant(cons);
        dsc.allocations
            .insert((DataOrigin::Constant(cons), DdcMemory::PeLrf), AllocId(3));
        dsc.users = vec![NodeId(11)];
        dsc.left_tree = false;
        metadata.new_allocations.insert(
            DdcMemory::PeLrf,
            Allocation {
                cons_id_and_alloc_node: BTreeMap::from([(cons, AllocId(3))]),
                ..Allocation::default()
            },
        );
        metadata.external_nodes.insert(NodeId(3));

        let alloc_use = AllocationUse::of(
            &dsc,
            DataOrigin::Constant(cons),
            DdcMemory::PeLrf,
            NodeId(11),
        )
        .expect("an allocation in that memory, and a node in its user list");
        reduce_users_or_delete_allocation_and_metadata(&mut dsc, &mut metadata, alloc_use);

        assert_eq!(dsc.reduce_seen.expect("the delegate ran").3, CanDelete::No);
        assert_eq!(
            metadata.new_allocations[&DdcMemory::PeLrf].cons_id_and_alloc_node,
            BTreeMap::from([(cons, AllocId(3))])
        );
    }

    #[test]
    fn a_fresh_datastage_takes_the_first_free_id_at_or_above_the_map_size() {
        let mut stages: DataStages<Vec<u32>> = DataStages::default();
        stages.0.insert(DatastageId(0), DataStage::default());
        stages.0.insert(DatastageId(2), DataStage::default());

        let id = construct_datastage(&mut stages);

        // Size 2 is taken, so the search lands on 3 and the hole at 1 stays free.
        assert_eq!(id, DatastageId(3));
        assert_eq!(stages.0[&id].ss.name, StageName("3".to_string()));
        assert_eq!(stages.0[&id].el.name, StageName("3el".to_string()));
        assert!(stages.0[&id].ss.dims.is_empty());
        assert!(!stages.0.contains_key(&DatastageId(1)));
    }

    #[test]
    fn a_copied_datastage_keeps_the_reference_extents_and_takes_only_the_new_names() {
        let mut stages: DataStages<Vec<u32>> = DataStages::default();
        let reference = DataStage {
            ss: StageDims {
                name: StageName("ref".to_string()),
                dims: vec![8, 16],
            },
            el: StageDims {
                name: StageName("refel".to_string()),
                dims: vec![1],
            },
        };

        let id = construct_datastage_from(&mut stages, &reference);

        assert_eq!(id, DatastageId(0));
        assert_eq!(stages.0[&id].ss.name, StageName("0".to_string()));
        assert_eq!(stages.0[&id].el.name, StageName("0el".to_string()));
        assert_eq!(stages.0[&id].ss.dims, vec![8, 16]);
        assert_eq!(stages.0[&id].el.dims, vec![1]);
    }

    #[test]
    fn a_loop_is_named_from_its_datastage_pair_and_then_each_of_its_dims() {
        let dims = LoopDims::new(
            PrimaryDimAndKind {
                dim: PrimaryDim::Mb,
                kind: MetaDimKind::Unpadded,
            },
            vec![PrimaryDimAndKind {
                dim: PrimaryDim::Kij,
                kind: MetaDimKind::WindowDim,
            }],
        );

        let node = construct_loop_node(DatastageId(1), DatastageId(5), dims);

        assert_eq!(node.name, NodeName("loop_ds1_ds5_mb_kij".to_string()));
        assert_eq!(node.num, DatastageId(1));
        assert_eq!(node.den, DatastageId(5));
        assert_eq!(node.dims.iter().count(), 2);
    }

    #[test]
    fn every_and_clause_term_of_a_condition_contributes_its_loop_and_other_kinds_contribute_none() {
        let cond = LoopCondComposite {
            or_of_ands: vec![
                vec![
                    LoopCond {
                        loop_node: NodeId(1),
                    },
                    LoopCond {
                        loop_node: NodeId(2),
                    },
                ],
                vec![LoopCond {
                    loop_node: NodeId(1),
                }],
            ],
        };
        let name = NodeName("cond0".to_string());
        let mut referenced = BTreeSet::new();

        collect_loop_references(UtilNode::Condition(&name, &cond), &mut referenced);
        assert_eq!(referenced, BTreeSet::from([NodeId(1), NodeId(2)]));

        collect_loop_references(UtilNode::Other(&name), &mut referenced);
        assert_eq!(referenced, BTreeSet::from([NodeId(1), NodeId(2)]));
    }

    #[test]
    fn the_source_side_falls_back_to_a_destination_and_the_access_pattern_overwrites_per_dim() {
        let src = stored(4, DfirUnit::Pe);
        let dst = stored(5, DfirUnit::Sfp);

        let mut padding = PaddingForm::default();
        padding.set_padding(PrimaryDim::In, PadType::PaddedFullSpan);
        padding.set_padding(PrimaryDim::Out, PadType::LoweredPadded);

        let allocs = TestAllocs {
            // ONLY the destination has an allocation, so the source side falls back to it.
            by_stream: vec![(dst, AllocId(3))],
            paddings: BTreeMap::from([(AllocId(3), padding)]),
        };

        let mut metadata = Metadata::default();
        metadata.datatransfers.insert(
            NodeId(11),
            DataTransfer {
                access_pattern_per_dim: BTreeMap::from([(
                    PrimaryDim::Out,
                    TransferAccessPattern {
                        from: PadType::PaddedNoZeroPad,
                        to: PadType::PaddedWZeroPad,
                    },
                )]),
                ..DataTransfer::default()
            },
        );

        let ends = TransferEnds {
            src,
            dsts: vec![dst],
        };
        let result = get_padding_per_dim(&allocs, &metadata, NodeId(11), &ends, TransferSide::Src);

        assert_eq!(result.padding(PrimaryDim::In), PadType::PaddedFullSpan);
        // The access pattern's SOURCE half overwrote the allocation's own form for that dim.
        assert_eq!(result.padding(PrimaryDim::Out), PadType::PaddedNoZeroPad);
        // And an unstated dim reads NOPAD.
        assert_eq!(result.padding(PrimaryDim::Mb), PadType::NoPad);

        let result = get_padding_per_dim(&allocs, &metadata, NodeId(11), &ends, TransferSide::Dst);
        assert_eq!(result.padding(PrimaryDim::Out), PadType::PaddedWZeroPad);
    }

    #[test]
    fn a_transfer_and_a_compute_carry_their_data_connects_and_any_other_kind_is_just_its_name() {
        let transfer = TransferNode {
            name: NodeName("t0".to_string()),
            src: operand(Some(DataConnect::ArfPt)),
            dsts: Dsts::new(operand(Some(DataConnect::ArfPtsum)), vec![operand(None)]),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
        };
        assert_eq!(
            get_node_description(UtilNode::Transfer(&transfer)),
            "t0 [ src: arf_pt dst:  arf_ptsum  ]"
        );

        let compute = ComputeNode {
            name: NodeName("c0".to_string()),
            op: ComputeType::Macc,
            ex_unit: SenComponent::Pe,
            inputs: vec![operand(Some(DataConnect::ArfPt))],
            outputs: vec![operand(Some(DataConnect::ArfPtsum))],
            num_folds_engaged: NumFolds::ONE,
        };
        assert_eq!(
            get_node_description(UtilNode::Compute(&compute)),
            "c0 [ input:  arf_pt output:  arf_ptsum]"
        );

        let name = NodeName("block0".to_string());
        assert_eq!(get_node_description(UtilNode::Other(&name)), "block0");
    }
}
