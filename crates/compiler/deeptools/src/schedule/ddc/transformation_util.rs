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

use sys_arch_spec::arch_enums::SenComponent;

use super::fold::{
    AllocId, AllocLayout, Allocations, BlockId, ConstIdx, DataOrigin, NodeId, PadType, StoredStream,
};
use super::metadata::{DatastageId, DdcMemory, DestIdx, MetaDimKind, Metadata, NodeIndex};
use super::transformation::{LoopId, SkipMetadataUpdate};
use super::v1::CoreClSet;
use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::control_flow::{CondOp, CondValType};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
use crate::generated::{DataConnect, Strategy};
use crate::schedule::dsc2::{
    AllocateNode, ComputeNode, DataInfo, Dsc, Dsts, Hops, LatchDataId, LdsIdx, NodeName, Operand,
    OperandPos, SyncNode, TransferNode, TransferSide,
};
use crate::schedule::l3::dsc::SymbolicDimInfo;
use crate::units::{Core, Corelet};

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

/// `EnumsConversion::senComponentsToString.at(comp)` for the `ddc::memories`
/// (`sys-arch-spec/arch_enums.cpp:10-118`, `ddc/ddc_metadata.h:20-21`).
const fn memory_spelling(memory: DdcMemory) -> &'static str {
    match memory {
        DdcMemory::Lx => "lx",
        DdcMemory::L0 => "l0",
        DdcMemory::L0Scale => "l0_scale",
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
    /// `component_`, a MEMORY BY TYPE, which is what makes *"Requested storage <c> is not a memory"*
    /// unspellable. ⛔ THAT GUARD READS `dsc2::memories` — 16 members (`dsc/dscdefn.cpp:142-144`) —
    /// and NOT [`DdcMemory`]'s eight (`ddc::memories`, `ddc/ddc_metadata.h:20-21`); the narrowing
    /// holds because both callsites pass PELRF or SFPLRF only
    /// (`ddc/ddc_transformation_util.cpp:834-837`, `ddc/ddc_transformation.cpp:1876-1878`).
    pub component: DdcMemory,
    /// `layoutDimOrder_` zipped with `maxDimSizes_`, whose fresh entries are the reference's `-1`.
    pub layout: AllocLayout,
    /// `padding_`.
    pub padding: PaddingForm,
    /// `allocUsers_` (`dsc/dsc2.h:1007`) — each user to its reference count, a map because the
    /// reference's vector of pairs is only ever searched by node.
    pub alloc_users: BTreeMap<NodeId, u32>,
}

impl DdcAllocateNode {
    /// `addAllocUser(userNode)` (`dsc/dsc2.h:1012`) — a repeat user bumps its count.
    pub fn add_alloc_user(&mut self, user: NodeId) {
        *self.alloc_users.entry(user).or_insert(0) += 1;
    }

    /// `removeAllocUser(userNode)` (`dsc/dsc2.h:1022`) — decrements, and erases at zero.
    ///
    /// ⛔ ITS *"Schedule node <n> is not in the user list of allocate node <a>"* IS AN ABSENT ENTRY,
    /// which here is simply no write: a node that was never a user cannot lose a reference.
    pub fn remove_alloc_user(&mut self, user: NodeId) {
        if let Some(count) = self.alloc_users.get_mut(&user) {
            *count -= 1;
            if *count == 0 {
                self.alloc_users.remove(&user);
            }
        }
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
        let dims = dsc.layout_dims(dsc.own_lds_idx(lds)).to_vec();
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
    /// `labeledDs_.at(lds).ldsIdx_` (`dsc/dscdefn.h:323`) — the ENTRY'S OWN self-index, which is
    /// what `getLayoutDims` is keyed by and need not be the index it was looked up under. ⛔ NOT
    /// `referenceLdsIdx_` (`dsc/dscdefn.h:324`), the neighbouring field naming another tensor.
    fn own_lds_idx(&self, lds: LdsIdx) -> LdsIdx;

    /// `getAllocation(di, storage, /*allowMissingAlloc=*/true)`.
    fn allocation_in(&self, origin: DataOrigin, storage: DdcMemory) -> Option<AllocId>;

    /// `labeledDs_.at(lds).memOrg_[storage] = { isPresent = true, allocateNode_ = alloc }`
    /// (`dsc/dscdefn.h:305`).
    fn set_allocation_in(&mut self, lds: LdsIdx, storage: DdcMemory, alloc: AllocId);

    /// The nodes in `allocUsers_` (`dsc/dsc2.h:1007`).
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

/// `dsc2::DataStage` (`dsc/dsc2.h:39`) — the steady-state and epilogue halves of one staging
/// parameter set.
///
/// ⭐ GENERIC IN THE EXTENTS PAYLOAD ON PURPOSE: entry 113 mints a stage by COPYING a reference
/// one, and a projection that dropped the extents would make it entry 112 spelled twice.
/// ⚠️ `schedule::l3::dsc::DataStage` PROJECTS THE SAME C++ STRUCT for the L3 side, with the core and
/// chunk stages held out of the map; the mint is NOT spelled twice — `e064_constructDatastage` calls
/// [`construct_datastage_from`].
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// ONE AND-CLAUSE TERM OF A LOOP CONDITION — `dsc2::LoopCond` (`dsc/dsc2.h:654-673`): which loop,
/// which of its dims, which comparison, and what the iterator is compared against.
///
/// ⛔ `loopComp_`'s `nullptr` DEFAULT IS NOT ADMITTED: entry 115 inserts it straight into
/// `referencedLoops`, so a null there is a defect and not a state. Nor is `dim_`'s
/// `PrimaryDimTypesCount`: `loopComp_` and `dim_` are the LOOKUP `mlir_loop_from_sn_loop_node`
/// performs (`SNControlFlowLowering.cpp:92-94`), and an unnamed dim matches no iterator.
/// ⛔ `condValInt_` LIVES IN [`CondValType::Int`], so the `-1` it holds for a `FIRST`/`LAST` term —
/// a value the reference reads only for `INT` (`dsc/dsc2.h:663`) — is unspellable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopCond {
    /// `loopComp_`.
    pub loop_node: NodeId,
    /// `dim_`, which is what entry 247 re-points the term by.
    pub dim: PrimaryDim,
    /// `condOp_`.
    pub op: CondOp,
    /// `condValType_` with its `condValInt_`.
    pub against: CondValType,
}

/// A CONDITION NODE'S LOOP CONDITION — `LoopCondComposite` (`dsc/dsc2.h:675-677`), an OR of ANDs of
/// per-loop terms with ONE negation over the lot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoopCondComposite {
    /// `twoLevelOrOfAnds_`, EMPTY exactly when the node carries the core/corelet condition instead
    /// — that is the whole of `hasCoreClCond()` (`dsc/dsc2.h:693-695`).
    pub or_of_ands: Vec<Vec<LoopCond>>,
    /// `negated_` (`dsc/dsc2.h:677`), which entry 249 flips when it reproduces an `else` branch.
    pub negated: bool,
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
/// ([`DscAllocations::own_lds_idx`]), NOT from the index it was looked up under.
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
/// exists for exactly those (`ddc/ddc_metadata.h:123`), so a constant is erased from it; the
/// mixed-key state cannot be spelled here. Its two *"Missing metadata"* aborts become a no-op
/// instead: EVERY site that mints an allocation registers it under the node's own `component_`
/// (`:59`, `:1339`, `:1506`; `ddc/ddc_transformation.cpp:2126`, `:2355`), so a live allocation with
/// no registry entry is unreachable — entry 110 is one such writer, not the only one.
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
/// ⚠️ TRAP: the `hasCoreClCond()` early return IS `twoLevelOrOfAnds_.empty()` (`dsc/dsc2.h:693-695`),
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

// ⭐ TYPES FOR ENTRIES 118-123. Union this section with this file's other vocabulary when its
// remaining entries land.

/// THE ALLOCATION ONE SCHEDULE NODE IS — [`DscAllocations::alloc_node`] READ THE OTHER WAY, which is
/// what turns a `nodeCloningMap_` entry back into an allocation: the reference holds ONE
/// `AllocateNode*` in both and `static_cast`s it back (`:1728`).
pub trait AllocationsByNode {
    /// The allocation at `node`, [`None`] where the node is not an `ALLOCATE`.
    fn allocation_of(&self, node: NodeId) -> Option<AllocId>;
}

/// WHAT ENTRIES 118 AND 119 DO TO AN ALLOCATION — its body, the clone, the placement and the one
/// field entry 119 writes on an allocation it did not mint.
pub trait AllocateCloning: DscAllocations + AllocationsByNode {
    /// `allocNode` as the carrier holds it.
    fn allocate(&self, alloc: AllocId) -> AllocateNode;

    /// `node->clone()` with `body` for its state, placed IMMEDIATELY AFTER `alloc`'s node among its
    /// parent's children (`parent->addChildNode(newNode, false, node)`). Yields the clone.
    fn clone_allocate_after(&mut self, alloc: AllocId, body: AllocateNode) -> AllocId;

    /// `allocNode->tempStorageForCompute_ = compute` (`dsc/dsc2.h:978`).
    fn set_temp_storage_for_compute(&mut self, alloc: AllocId, compute: NodeName);
}

/// WHAT ENTRIES 118 AND 119 DO TO A COMPUTE — its body, the clone and the placement.
pub trait ComputeCloning {
    /// `computeNode` as the carrier holds it.
    fn compute(&self, node: NodeId) -> ComputeNode;

    /// `node->clone()` with `body` for its state, placed immediately after `node` among its parent's
    /// children. Yields the clone.
    fn clone_compute_after(&mut self, node: NodeId, body: ComputeNode) -> NodeId;
}

/// THE `memOrg_` AND `constantInfo_` SLOTS KEYED BY A FULL `SenComponents` — the ones entry 118
/// writes.
///
/// ⛔ NOT [`DscAllocations::allocation_in`], WHICH IS KEYED BY [`DdcMemory`]: `toggleMap` names `PE`,
/// `SFP`, `PESTATE` and `SFPSTATE`, none of which has a [`DdcMemory`] spelling, and this is the same
/// `memOrg_` reached under the vocabulary that can state them.
/// ⛔ PRESENCE AND A SET ALLOCATION ARE TWO QUESTIONS: the reference tests `memOrg_.count(c)` and
/// then `memOrg_.at(c).allocateNode_` separately, and an entry whose allocation is still null is
/// exactly what its own copy-then-fill sequence creates.
pub trait ComponentAllocations {
    /// `labeledDs_.at(lds).memOrg_.count(storage)`.
    fn has_mem_org(&self, lds: LdsIdx, storage: SenComponent) -> bool;

    /// `memOrg_[to] = memOrg_[from]` with the copy's `allocateNode_` CLEARED (`:1399`).
    fn copy_mem_org_without_allocation(
        &mut self,
        lds: LdsIdx,
        from: SenComponent,
        to: SenComponent,
    );

    /// `memOrg_.at(storage).allocateNode_`, [`None`] for an absent entry and a null one alike.
    fn mem_org_allocation(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId>;

    /// `memOrg_.at(storage).allocateNode_ = alloc`.
    fn set_mem_org_allocation(&mut self, lds: LdsIdx, storage: SenComponent, alloc: AllocId);

    /// `constantInfo_.at(constant).allocations_.count(storage)`.
    fn has_constant_allocation(&self, constant: ConstIdx, storage: SenComponent) -> bool;

    /// `constantInfo_.at(constant).allocations_[storage] = alloc`.
    fn set_constant_allocation(
        &mut self,
        constant: ConstIdx,
        storage: SenComponent,
        alloc: AllocId,
    );
}

/// AN ALLOCATION PROVED CLONEABLE FOR THE PE/SFP WORK SPLIT, CARRYING THE SUFFIX ITS CLONE TAKES.
///
/// ⛔⛔ TWO ABORTS AND THE `nullptr` ANSWER COLLAPSE INTO THIS TYPE: *"has already been cloned"*,
/// *"as none of ldsIdx_ and constIdx_ is set"*, and the component `toggleMap` has no entry for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeSfpAllocateSplit {
    alloc: AllocId,
    body: AllocateNode,
    origin: DataOrigin,
    toggled: SenComponent,
    suffix: ParallelSuffix,
}

impl PeSfpAllocateSplit {
    /// The witness, or [`None`] for either abort and for the `nullptr`.
    ///
    /// ⚠️⚠️ TRAP, AND IT IS THE REFERENCE'S: THE SUFFIX IS NOT THE TOGGLE. It is `"_sfp_parallel"`
    /// for `PELRF` and `PESTATE` ALONE (`:1332`), so a `PE` allocation's clone runs on SFP and is
    /// still named `"_pe_parallel"` — the opposite of entry 119's rule for that same pair.
    /// ⛔ `ldsIdx_` WINS: the reference tests it first and the constant branch is its `else`.
    #[must_use]
    pub fn of<D: AllocateCloning + ?Sized>(
        dsc: &D,
        metadata: &Metadata,
        alloc: AllocId,
    ) -> Option<Self> {
        let body = dsc.allocate(alloc);
        let toggled = toggle_pe_sfp(body.component)?;
        if metadata
            .node_cloning_map
            .contains_key(&dsc.alloc_node(alloc))
        {
            return None;
        }
        let origin = match (body.lds, body.const_idx) {
            (Some(lds), _) => DataOrigin::LabeledDs(lds),
            (None, Some(constant)) => DataOrigin::Constant(constant),
            (None, None) => return None,
        };
        let suffix = match body.component {
            SenComponent::Pelrf | SenComponent::Pestate => ParallelSuffix::Sfp,
            _ => ParallelSuffix::Pe,
        };
        Some(Self {
            alloc,
            body,
            origin,
            toggled,
            suffix,
        })
    }

    /// Which labeled DS or constant the allocation is for.
    #[must_use]
    pub const fn origin(&self) -> DataOrigin {
        self.origin
    }

    /// The component the clone is placed on — `toggleMap.at(node->component_)`.
    #[must_use]
    pub const fn toggled(&self) -> SenComponent {
        self.toggled
    }
}

/// Replaces: e118_cloneForPeSfpWorkSplit
///
/// CLONES ONE ALLOCATION ONTO THE OTHER HALF OF THE PE/SFP PAIR (`:1306`): a copy right after it with
/// the component toggled, the suffix on its name and no users or temp storage, registered under the
/// new component in `newAllocations_` and filling that component's `memOrg_` or `allocations_` slot.
///
/// ⛔ THE [`None`]s ARE FOUR MORE ABORTS: the original missing under its own component, either target
/// slot already taken, or the labeled DS holding no `memOrg_` entry for the original component.
pub fn clone_allocate_for_pe_sfp_work_split<D>(
    dsc: &mut D,
    metadata: &mut Metadata,
    split: PeSfpAllocateSplit,
    update: SkipMetadataUpdate,
) -> Option<AllocId>
where
    D: AllocateCloning + ComponentAllocations + ?Sized,
{
    let PeSfpAllocateSplit {
        alloc,
        body,
        origin,
        toggled,
        suffix,
    } = split;
    let node = dsc.alloc_node(alloc);

    // ⚠️ EVERY REFUSAL IS ANSWERED BEFORE THE CLONE IS MINTED, so a `None` leaves nothing behind.
    // The reference instead places the node between its metadata writes and its `memOrg_` writes.
    // `newAllocations_[newNode->component_]` and `.at(node->component_)` — a component with no
    // [`DdcMemory`] spelling is not a key of that map, which is what its `.at()` throws on.
    let registration = match update {
        SkipMetadataUpdate::No => {
            let (from, to) = component_memory(body.component).zip(component_memory(toggled))?;
            if registered_allocation(metadata, from, origin) != Some(alloc) {
                return None;
            }
            if registered_allocation(metadata, to, origin).is_some() {
                return None;
            }
            Some(to)
        }
        SkipMetadataUpdate::Yes => None,
    };

    match origin {
        DataOrigin::LabeledDs(lds) => {
            if !dsc.has_mem_org(lds, body.component) {
                return None;
            }
            if dsc.mem_org_allocation(lds, toggled).is_some() {
                return None;
            }
        }
        DataOrigin::Constant(constant) => {
            if dsc.has_constant_allocation(constant, toggled) {
                return None;
            }
        }
    }

    let mut clone_body = body.clone();
    clone_body.name = NodeName(format!("{}{}", body.name.0, suffix.spelling()));
    clone_body.component = toggled;
    clone_body.alloc_users.clear();
    clone_body.temp_storage_for_compute = None;
    let clone = dsc.clone_allocate_after(alloc, clone_body);

    if let Some(to) = registration {
        let allocated = metadata.new_allocations.entry(to).or_default();
        match origin {
            DataOrigin::LabeledDs(lds) => {
                allocated.lds_idx_and_alloc_node.insert(lds, clone);
            }
            DataOrigin::Constant(constant) => {
                allocated.cons_id_and_alloc_node.insert(constant, clone);
            }
        }
    }
    let clone_node = dsc.alloc_node(clone);
    metadata
        .node_cloning_map
        .entry(node)
        .or_default()
        .push(clone_node);

    match origin {
        DataOrigin::LabeledDs(lds) => {
            if !dsc.has_mem_org(lds, toggled) {
                dsc.copy_mem_org_without_allocation(lds, body.component, toggled);
            }
            dsc.set_mem_org_allocation(lds, toggled, clone);
        }
        DataOrigin::Constant(constant) => {
            dsc.set_constant_allocation(constant, toggled, clone);
        }
    }

    Some(clone)
}

/// `newAllocations_.at(memory).ldsIdxAndAllocNode`, else `.consIdAndAllocNode`, keyed by the origin —
/// the two branches entry 118 spells twice over.
fn registered_allocation(
    metadata: &Metadata,
    memory: DdcMemory,
    origin: DataOrigin,
) -> Option<AllocId> {
    let allocated = metadata.new_allocations.get(&memory)?;
    match origin {
        DataOrigin::LabeledDs(lds) => allocated.lds_idx_and_alloc_node.get(&lds).copied(),
        DataOrigin::Constant(constant) => allocated.cons_id_and_alloc_node.get(&constant).copied(),
    }
}

/// A COMPUTE PROVED CLONEABLE FOR THE PE/SFP WORK SPLIT, CARRYING THE SUFFIX ITS CLONE TAKES.
///
/// ⛔⛔ THREE ABORTS AND THE `nullptr` ANSWER COLLAPSE INTO THIS TYPE: an input sent from the other
/// half of the pair, an output sent to it, *"has already been cloned"*, and an `exUnit_` that is
/// neither PE nor SFP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeSfpComputeSplit {
    node: NodeId,
    toggled: SenComponent,
    suffix: ParallelSuffix,
}

impl PeSfpComputeSplit {
    /// The witness, or [`None`] for any abort and for the `nullptr`.
    ///
    /// ⚠️ THE SUFFIX IS THE HALF THE CLONE RUNS ON here — `PE` becomes `"_sfp_parallel"` (`:1685`),
    /// which is NOT what [`PeSfpAllocateSplit`] does with the same component.
    #[must_use]
    pub fn of(metadata: &Metadata, node: NodeId, compute: &ComputeNode) -> Option<Self> {
        let ex_unit = pe_or_sfp(compute.ex_unit)?;
        let operands = compute.inputs.iter().chain(compute.outputs.iter());
        if operands
            .clone()
            .any(|operand| pe_or_sfp(operand.unit).is_some())
        {
            return None;
        }
        if metadata.node_cloning_map.contains_key(&node) {
            return None;
        }
        let (toggled, suffix) = if ex_unit == SenComponent::Pe {
            (SenComponent::Sfp, ParallelSuffix::Sfp)
        } else {
            (SenComponent::Pe, ParallelSuffix::Pe)
        };
        Some(Self {
            node,
            toggled,
            suffix,
        })
    }
}

/// Replaces: e119_cloneForPeSfpWorkSplit
///
/// CLONES ONE COMPUTE ONTO THE OTHER HALF OF THE PE/SFP PAIR (`:1422`): a copy right after it with
/// `exUnit_` flipped and the suffix on its name, whose named operand connects take the suffix and
/// whose operand units toggle — or, for an opaque op, whose register allocations follow their clones.
///
/// ⚠️⚠️ TRAP: A NAMED CONNECT TAKES THE SUFFIX UNCONDITIONALLY, whether or not that operand's unit
/// toggled — the opposite of [`toggle_for_split`], which entry 304 uses on a transfer.
pub fn clone_compute_for_pe_sfp_work_split<D>(
    dsc: &mut D,
    metadata: &mut Metadata,
    split: PeSfpComputeSplit,
) -> Option<NodeId>
where
    D: ComputeCloning + AllocateCloning + FifoResults + MintedConnects + ?Sized,
{
    let PeSfpComputeSplit {
        node,
        toggled,
        suffix,
    } = split;
    let mut body = dsc.compute(node);
    body.name = NodeName(format!("{}{}", body.name.0, suffix.spelling()));
    body.ex_unit = toggled;

    // ⛔ THE OPAQUE BRANCH'S `None`s ARE THE REFERENCE'S `DT_CHECK`s: no metadata entry, no internal
    // register allocation, or a register allocation that was not cloned exactly once.
    let mut opaque = None;
    if dsc.is_opaque(node) {
        let mut op = metadata.opaque_ops.get(&node)?.clone();
        let internal = sole_allocation_clone(dsc, metadata, op.internal_reg_alloc?)?;
        op.internal_reg_alloc = Some(internal);
        for alloc in op.in_out_reg_allocs.values_mut() {
            *alloc = sole_allocation_clone(dsc, metadata, *alloc)?;
        }
        let connects = body
            .instr_attribute
            .input_data_connects
            .iter_mut()
            .chain(body.instr_attribute.output_data_connects.iter_mut());
        for connect in connects {
            *connect = dsc.intern_connect(MintedConnect::Parallel(*connect, suffix));
        }
        opaque = Some((op, internal));
    } else {
        for operand in body.inputs.iter_mut().chain(body.outputs.iter_mut()) {
            if let Some(base) = operand.data.data_connect {
                operand.data.data_connect =
                    Some(dsc.intern_connect(MintedConnect::Parallel(base, suffix)));
            }
            if let Some(unit) = toggle_pe_sfp(operand.unit) {
                operand.unit = unit;
            }
        }
    }

    // `getMutableAllocation(offsets.at(i), newNode-><in|out>puts_.at(i), true)` for each operand
    // whose unit moved — and `toggleMap` is its own inverse, so the toggled unit answers that.
    let users: Vec<(DataOrigin, DdcMemory)> = if opaque.is_none() {
        body.inputs
            .iter()
            .chain(body.outputs.iter())
            .filter(|operand| toggle_pe_sfp(operand.unit).is_some())
            .filter_map(|operand| data_origin(operand.data).zip(component_memory(operand.unit)))
            .collect()
    } else {
        Vec::new()
    };
    let name = body.name.clone();
    let clone = dsc.clone_compute_after(node, body);

    for (origin, memory) in users {
        if let Some(alloc) = dsc.allocation_in(origin, memory) {
            dsc.add_alloc_user(alloc, clone);
        }
    }
    if let Some((op, internal)) = opaque {
        let memory = dsc.alloc_component(internal);
        metadata
            .new_allocations
            .entry(memory)
            .or_default()
            .comp_and_alloc_node
            .insert(clone, internal);
        dsc.set_temp_storage_for_compute(internal, name);
        dsc.add_alloc_user(internal, clone);
        for &alloc in op.in_out_reg_allocs.values() {
            dsc.add_alloc_user(alloc, clone);
        }
        metadata.opaque_ops.insert(clone, op);
    }
    metadata
        .node_cloning_map
        .entry(node)
        .or_default()
        .push(clone);

    Some(clone)
}

/// THE ONE CLONE OF AN ALLOCATION — `DT_CHECK(nodeCloningMap_.at(alloc).size() == 1)` and the `.at(0)`
/// that follows it (`:1725`, `:1740`), [`None`] where the allocation was not cloned exactly once.
fn sole_allocation_clone<D>(dsc: &D, metadata: &Metadata, alloc: AllocId) -> Option<AllocId>
where
    D: DscAllocations + AllocationsByNode + ?Sized,
{
    let clones = metadata.node_cloning_map.get(&dsc.alloc_node(alloc))?;
    let &[only] = clones.as_slice() else {
        return None;
    };
    dsc.allocation_of(only)
}

/// THE CENSUS' POSITIONAL IDENTITY AS A SCHEDULE NODE — `dataConnects_`' ends are [`NodeIndex`] while
/// `externalNodes_` holds [`NodeId`], and the reference holds ONE `ScheduleNode*` in both.
///
/// ⛔ THE SEAM IS THE IDENTITY RESOLUTION ALONE: the census read and the external test both stay in
/// entry 120. `metadata.rs`' own note says unifying the two spellings belongs to whichever batch first
/// owns a schedule-tree type, and this is not it.
pub trait CensusNodes {
    /// The schedule node the census recorded at that position.
    fn census_node(&self, node: NodeIndex) -> NodeId;
}

/// Replaces: e120_storageOrDatastreamIsExternal
///
/// Whether the datastream's allocation in `storage`, or ANY node on the connect's producer side —
/// consumer side, outgoing — is an `externalNodes_` member (`:1652`).
///
/// ⚠️ TRAP: the allocation is looked up under `storage`, which entries 253/254 reach with an operand's
/// `storage_` and entries 255/256 with its `unit_`. ⛔ An unset `dataConnect_` is the reference's
/// `dataConnects_.count("")`, which no census inserts; its `> 2` `std::cerr` notes are droppable.
#[must_use]
pub fn storage_or_datastream_is_external<D>(
    dsc: &D,
    metadata: &Metadata,
    data: DataInfo,
    storage: SenComponent,
    direction: StreamDirection,
) -> bool
where
    D: DscAllocations + CensusNodes + ?Sized,
{
    let allocation = data_origin(data)
        .zip(component_memory(storage))
        .and_then(|(origin, memory)| dsc.allocation_in(origin, memory));
    if let Some(alloc) = allocation {
        if metadata.external_nodes.contains(&dsc.alloc_node(alloc)) {
            return true;
        }
    }

    let Some(ends) = data
        .data_connect
        .and_then(|connect| metadata.data_connects.get(&connect))
    else {
        return false;
    };
    let nodes = match direction {
        StreamDirection::Incoming => ends.producers(),
        StreamDirection::Outgoing => ends.consumers(),
    };
    nodes
        .iter()
        .any(|&node| metadata.external_nodes.contains(&dsc.census_node(node)))
}

/// THE SCHEDULE NODE ONE `name_` NAMES — [`SyncNode::other_ends`] holds names where the reference
/// holds `SyncNode*`, and `isExternalNode` is asked of the node itself.
pub trait SyncNodesByName {
    /// The node named `name`, [`None`] where the tree holds none.
    fn node_named(&self, name: &NodeName) -> Option<NodeId>;
}

/// Replaces: e121_relatedToExternalNodes
///
/// Whether ANY node at the other end of one of this sync's signals is external (`:1748`).
///
/// ⛔ A DISTINCT NAME: five C++ `relatedToExternalNodes` overloads differ only in their argument
/// type — entries 306, 341 and 361 are the others — and Rust must name each of them.
/// ⚠️ A NAME THE TREE HOLDS NO NODE FOR IS NOT EXTERNAL, which is also the reference's answer for a
/// signal end that is not an `externalNodes_` member.
#[must_use]
pub fn sync_related_to_external_nodes<N: SyncNodesByName + ?Sized>(
    nodes: &N,
    metadata: &Metadata,
    node: &SyncNode,
) -> bool {
    node.other_ends.iter().any(|name| {
        nodes
            .node_named(name)
            .is_some_and(|end| metadata.external_nodes.contains(&end))
    })
}

/// A SCHEDULE NODE AS ENTRY 122 READS ONE — the three `nodeType_`s that carry a component, and every
/// other kind, which the reference's `if`/`else if` chain leaves unrelated.
///
/// ⛔ NEITHER [`UtilNode`] NOR [`crate::schedule::dsc2::Node`] SPELLS THIS SET: the first has no
/// `ALLOCATE` arm and the second has no fourth arm, and entry 122 is asked of a whole tree's nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentNode<'a> {
    /// `ALLOCATE` — its `component_`.
    Allocate(&'a AllocateNode),
    /// `COMPUTE` — its `exUnit_`.
    Compute(&'a ComputeNode),
    /// `TRANSFER` — its `src_.unit_`, then each `dstVias_.at(i).loc_.unit_`.
    Transfer(&'a TransferNode),
    /// Every other `nodeType_`, which carries no component of its own.
    Other,
}

/// Replaces: e122_isNodeRelatedToComps
///
/// WHICH OF `comps` THE NODE IS ON, if any (`:1778`) — an allocation's component, a compute's
/// execution unit, or a transfer's source unit and then each destination's, first match winning.
///
/// ⭐ THE `nodeComp` OUT-PARAMETER AND THE `bool` ARE ONE [`Option`]: the reference writes
/// `NO_COMPONENT` and answers `false` together, and no caller can read one without the other.
#[must_use]
pub fn is_node_related_to_comps(
    node: ComponentNode<'_>,
    comps: &[SenComponent],
) -> Option<SenComponent> {
    let related = |comp: SenComponent| comps.contains(&comp).then_some(comp);
    match node {
        ComponentNode::Allocate(alloc) => related(alloc.component),
        ComponentNode::Compute(compute) => related(compute.ex_unit),
        ComponentNode::Transfer(transfer) => related(transfer.src.unit)
            .or_else(|| transfer.dsts.iter().find_map(|dst| related(dst.unit))),
        ComponentNode::Other => None,
    }
}

/// WHAT ENTRY 123 ASKS OF THE DSC AND ITS SCHEDULE TREE — the same walk [`NewLabeledDs`] states,
/// SCOPED TO ONE SUBTREE, plus the `memOrg_` move the allocate branch performs.
///
/// ⛔ [`TreeLdsSlot::ParametricLoop`] NEVER APPEARS HERE: entry 123's filter is `{ALLOCATE, TRANSFER,
/// COMPUTE}` and entry 257's adds `LOOP`, so the loop arm is unreachable under this walk.
pub trait SubtreeLds: NewLabeledDs + AllocationsByNode {
    /// Every lds slot the `{ALLOCATE, TRANSFER, COMPUTE}` walk from `start` reaches WITH the index it
    /// holds, in DFS order — `traverseTreeDFSMutable(startNode, .., ALL, -1, -1)`.
    fn tree_lds_slots_under(&self, start: NodeId) -> Vec<(TreeLdsSlot, Option<LdsIdx>)>;

    /// The `isOpaqueOp_` computes that same walk reaches.
    fn opaque_computes_under(&self, start: NodeId) -> Vec<NodeId>;

    /// The `memOrg_` storages of `labeledDs_.at(lds)` whose `allocateNode_` IS `alloc`.
    fn mem_org_storages_of(&self, lds: LdsIdx, alloc: AllocId) -> Vec<SenComponent>;

    /// `memorgNew[storage] = memorgOld.at(storage)` with the SOURCE's `allocateNode_` cleared.
    fn move_mem_org(&mut self, from: LdsIdx, to: LdsIdx, storage: SenComponent);
}

/// Replaces: e123_updateNodesWithNewLds
///
/// REPOINTS EVERY DATASTREAM UNDER `start` FROM ONE LABELED DS INDEX TO ANOTHER (`:1919`): each
/// compute operand's and transfer end's `myLdsIdx_`, each opaque op's `ldsIdx_`, and each allocation's
/// `ldsIdx_` — whose `memOrg_` entry moves across and whose `newAllocations_` key rekeys to `new_lds`.
///
/// ⛔ DELIBERATE DIVERGENCE: the reference inserts `ldsToAlloc[newLdsIdx]` INTO the `std::map` its own
/// range-`for` walks, so a `newLdsIdx` above `oldLdsIdx` is revisited and erased; here the key lives.
pub fn update_nodes_with_new_lds<D: SubtreeLds + ?Sized>(
    dsc: &mut D,
    metadata: &mut Metadata,
    new_lds: LdsIdx,
    old_lds: LdsIdx,
    start: NodeId,
) {
    for compute in dsc.opaque_computes_under(start) {
        if let Some(opaque) = metadata.opaque_ops.get_mut(&compute) {
            if opaque.lds_idx == Some(old_lds) {
                opaque.lds_idx = Some(new_lds);
            }
        }
    }

    // ⛔ *"the node has to be either compute, transfer or allocate."* is unspellable: the walk's own
    // filter admits exactly those three kinds.
    for (slot, held) in dsc.tree_lds_slots_under(start) {
        if held != Some(old_lds) {
            continue;
        }
        dsc.set_tree_lds(slot, new_lds);
        let TreeLdsSlot::Allocate(node) = slot else {
            continue;
        };
        let Some(alloc) = dsc.allocation_of(node) else {
            continue;
        };
        for storage in dsc.mem_org_storages_of(old_lds, alloc) {
            dsc.move_mem_org(old_lds, new_lds, storage);
        }
        for allocated in metadata.new_allocations.values_mut() {
            let held_under: Vec<LdsIdx> = allocated
                .lds_idx_and_alloc_node
                .iter()
                .filter(|&(_, &at)| at == alloc)
                .map(|(&lds, _)| lds)
                .collect();
            if held_under.is_empty() {
                continue;
            }
            allocated.lds_idx_and_alloc_node.insert(new_lds, alloc);
            for lds in held_under {
                if lds != new_lds {
                    allocated.lds_idx_and_alloc_node.remove(&lds);
                }
            }
        }
    }
}

// ⭐ TYPES FOR ENTRIES 304-306. Union this section with this file's other vocabulary when its
// remaining entries land.

/// WHICH INPUT OF A COMPUTE — the `i` entries 300 and 304 spell into a minted name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputIdx(pub usize);

/// WHICH HALF OF THE PE/SFP PAIR A CLONE RUNS ON — entry 304's suffix, `"_sfp_parallel"` where the
/// original ran on PE and `"_pe_parallel"` otherwise (`ddc/ddc_transformation_util.cpp:1583`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParallelSuffix {
    /// `"_pe_parallel"`.
    Pe,
    /// `"_sfp_parallel"`.
    Sfp,
}

impl ParallelSuffix {
    /// The suffix as the reference concatenates it.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Pe => "_pe_parallel",
            Self::Sfp => "_sfp_parallel",
        }
    }
}

/// A `dataConnect_` THE DDC MINTS RATHER THAN THE DDL CENSUS NAMING — every fresh connect entries
/// 300 and 304 write into a `DataInfo` (`ddc/ddc_transformation.cpp:1152-1310`, `:1583-1608`).
///
/// ⛔⛔ [`DataConnect`] IS A BUILD-TIME CENSUS OF THE VENDORED DDL TEMPLATES, so it has no member for
/// any of these and never will; a `String` would break the crate's closed-set rule. THIS ENUM IS THE
/// CLOSED SET, and [`MintedConnects`] is what turns one into the carrier's own [`DataConnect`], the
/// way `ddlInterface.loop_labels_` interns a minted loop's label (`ddc/ddl/ddl_conversion.cpp:1065`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MintedConnect {
    /// `"sfp_compress_input<i>_comp"` — entry 300's LX-to-SFP compression transfer destination.
    SfpCompressInputComp(InputIdx),
    /// `"sfp_compress_output<i>_comp"` — that input's dummy FMA output, and the SFP-to-LX source.
    SfpCompressOutputComp(InputIdx),
    /// `"lx_compress_input<i>_comp"` — the write back into LX, taken over by the original transfer.
    LxCompressInputComp(InputIdx),
    /// `"lx_compress_output_exp"` — entry 300's expansion read out of the internal output.
    LxCompressOutputExp,
    /// `"sfp_compress_output_exp"` — the expansion FMA's output, read again by its write-back.
    SfpCompressOutputExp,
    /// `<base> + <suffix>` — entry 304's suffix on a connect that is already named.
    Parallel(DataConnect, ParallelSuffix),
    /// `"autoshuffle_edge_<n>"` — entry 376's edge into one freshly allocated shuffle register.
    AutoshuffleEdge(AutoShuffleName),
}

impl MintedConnect {
    /// The name as the reference concatenates it.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            Self::SfpCompressInputComp(i) => format!("sfp_compress_input{}_comp", i.0),
            Self::SfpCompressOutputComp(i) => format!("sfp_compress_output{}_comp", i.0),
            Self::LxCompressInputComp(i) => format!("lx_compress_input{}_comp", i.0),
            Self::LxCompressOutputExp => "lx_compress_output_exp".to_string(),
            Self::SfpCompressOutputExp => "sfp_compress_output_exp".to_string(),
            Self::Parallel(base, suffix) => format!("{}{}", base.spelling(), suffix.spelling()),
            Self::AutoshuffleEdge(name) => format!("autoshuffle_edge_{}", name.0),
        }
    }
}

/// ONE AUTOMATIC-SHUFFLE NAME SUFFIX — entry 376's `name_counter` (`ddc/ddc_transformation.cpp:1854`),
/// which suffixes BOTH an intermediate register's `dsName_` and its `dataConnect_` so the two agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AutoShuffleName(pub u32);

/// HOW A MINTED `data_connect=` REACHES THE CRATE'S CENSUS.
pub trait MintedConnects {
    /// The [`DataConnect`] naming `connect`, interned if the carrier has not seen it before.
    fn intern_connect(&mut self, connect: MintedConnect) -> DataConnect;
}

/// WHAT ENTRY 300 CANNOT REACH YET — `updateNodesWithNewLds(newLdsIdx, oldLdsIdx, startNode)`
/// (`ddc/ddc_transformation_util.cpp:1919`), which repoints every datastream under `start` from one
/// labeled DS index to another.
///
/// ⛔ THE SEAM ENTRY 300 REACHES IT THROUGH, AND NOT A STAND-IN: it walks the whole subtree rewriting
/// transfer, compute and allocate operands at once, which is a carrier's job however it is spelled.
/// ⭐ [`update_nodes_with_new_lds`] IS THE PORT, and it is the one implementation of this method: a
/// carrier states [`SubtreeLds`] and delegates.
pub trait NewLdsRewrite {
    /// `updateNodesWithNewLds(new_lds, old_lds, start)`.
    fn update_nodes_with_new_lds(&mut self, new_lds: LdsIdx, old_lds: LdsIdx, start: NodeId);
}

/// `toggleMap` (`ddc/ddc_transformation_util.cpp:12-18`) — the PE/SFP swap over the THREE pairs it
/// holds, [`None`] for a component it has no entry for, which is what leaves that field alone.
#[must_use]
pub const fn toggle_pe_sfp(comp: SenComponent) -> Option<SenComponent> {
    match comp {
        SenComponent::Pe => Some(SenComponent::Sfp),
        SenComponent::Sfp => Some(SenComponent::Pe),
        SenComponent::Pelrf => Some(SenComponent::Sfplrf),
        SenComponent::Sfplrf => Some(SenComponent::Pelrf),
        SenComponent::Sfpstate => Some(SenComponent::Pestate),
        SenComponent::Pestate => Some(SenComponent::Sfpstate),
        _ => None,
    }
}

/// A TRANSFER PROVED CLONEABLE FOR THE PE/SFP WORK SPLIT, CARRYING THE SUFFIX ITS CLONE TAKES.
///
/// ⛔⛔ TWO ABORTS AND THE `nullptr` ANSWER COLLAPSE INTO THIS TYPE: *"as the transfer writes to
/// multiple destinations"*, *"has already been cloned"*, and the transfer that involves neither PE
/// nor SFP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeSfpTransferSplit {
    transfer: NodeId,
    suffix: ParallelSuffix,
}

impl PeSfpTransferSplit {
    /// The witness, or [`None`] for either abort and for the `nullptr`.
    ///
    /// ⛔ TRAP, AND IT IS THE REFERENCE'S: the *"dstVias.via() is not empty"* abort is gated on
    /// `transformationReportLevel_ > 0` (`:1548-1552`), so at the DEFAULT report level a transfer
    /// routed to PE through other components IS still cloned; that default is what holds here.
    /// ⚠️ And the multi-destination abort is only reached through the DESTINATION scan — a transfer
    /// whose `src_.unit_` is PE or SFP clones however many destinations it has.
    #[must_use]
    pub fn of(metadata: &Metadata, transfer: NodeId, node: &TransferNode) -> Option<Self> {
        let mut existing = pe_or_sfp(node.src.unit);
        if existing.is_none() {
            existing = node.dsts.iter().filter_map(|dst| pe_or_sfp(dst.unit)).last();
            if existing.is_some() && node.dsts.len() > 1 {
                return None;
            }
        }
        let existing = existing?;
        if metadata.node_cloning_map.contains_key(&transfer) {
            return None;
        }
        Some(Self {
            transfer,
            suffix: if existing == SenComponent::Pe {
                ParallelSuffix::Sfp
            } else {
                ParallelSuffix::Pe
            },
        })
    }
}

/// `is_any_of(comp, {PE, SFP})` — `comps` (`ddc/ddc_transformation_util.cpp:1543`).
const fn pe_or_sfp(comp: SenComponent) -> Option<SenComponent> {
    match comp {
        SenComponent::Pe => Some(SenComponent::Pe),
        SenComponent::Sfp => Some(SenComponent::Sfp),
        _ => None,
    }
}

/// WHAT ENTRY 304 DOES TO THE TREE — the clone and its placement.
pub trait NodeCloning: ScheduleSurgery {
    /// `node->clone()` with `body` for its transfer state, placed IMMEDIATELY AFTER `node` among its
    /// parent's children (`parent->addChildNode(newNode, false, node)`). Yields the clone.
    fn clone_transfer_after(&mut self, node: NodeId, body: TransferNode) -> NodeId;
}

/// Replaces: e304_cloneForPeSfpWorkSplit
///
/// CLONES ONE TRANSFER ONTO THE OTHER HALF OF THE PE/SFP PAIR: fully unrolls the original, mints a
/// copy right after it whose source and destination units, storages and `data_connect=` are swapped
/// by [`toggle_pe_sfp`], registers the copy as a user of both ends' allocations, records which ends
/// then need the split's constant offset, and records the clone (`:1538`).
///
/// ⛔ DELIBERATE DIVERGENCE: *"could not be unrolled"* is an abort in the reference; here it is the
/// [`None`] answer, which is the ONLY thing this [`Option`] means once [`PeSfpTransferSplit`] holds.
/// ⚠️ A CONNECT TAKES THE SUFFIX ONLY WHERE THE UNIT TOGGLED *AND* THE CONNECT IS ALREADY NAMED;
/// the STORAGE toggle is independent of both.
pub fn clone_transfer_for_pe_sfp_work_split<S, D>(
    tree: &mut S,
    stages: &mut DataStages<D>,
    metadata: &mut Metadata,
    split: PeSfpTransferSplit,
) -> Option<NodeId>
where
    S: NodeCloning + FifoResults + Allocations + TransferUnrolling + MintedConnects + ?Sized,
    D: Default,
{
    let PeSfpTransferSplit { transfer, suffix } = split;
    let site = InternalNode::of(metadata, transfer)?;
    if !unroll_transfer(tree, stages, metadata, site) {
        return None;
    }

    let mut body = tree.transfer(transfer);
    body.name = NodeName(format!("{}{}", body.name.0, suffix.spelling()));
    toggle_for_split(tree, suffix, &mut body.src);
    let hops: Vec<Hops> = body
        .dsts
        .routes()
        .map(|(_, hops)| Hops(hops.to_vec()))
        .collect();
    let mut dsts: Vec<Operand> = body.dsts.iter().copied().collect();
    for dst in &mut dsts {
        toggle_for_split(tree, suffix, dst);
    }
    let (first, rest) = dsts.split_first()?;
    body.dsts = Dsts::new(*first, rest.to_vec()).with_hops(hops);

    let clone = tree.clone_transfer_after(transfer, body.clone());

    let ends = tree.transfer_ends(clone);
    for end in std::iter::once(ends.src).chain(ends.dsts) {
        if let Some(alloc) = tree.allocation(end) {
            tree.add_alloc_user(alloc, clone);
        }
    }

    let cloned = metadata.datatransfers.entry(clone).or_default();
    if pe_or_sfp(body.src.unit).is_none() && is_memory(body.src.storage) {
        cloned.apply_pe_sfp_split_offset_src = true;
    }
    for (index, dst) in body.dsts.iter().enumerate() {
        if pe_or_sfp(dst.unit).is_none() && is_memory(dst.storage) {
            cloned
                .apply_pe_sfp_split_offset_dest
                .push(DestIdx(index as u32));
        }
    }
    metadata
        .node_cloning_map
        .entry(transfer)
        .or_default()
        .push(clone);

    Some(clone)
}

/// One operand through `toggleMap`, with the suffix on a connect whose unit moved.
fn toggle_for_split<S: MintedConnects + ?Sized>(
    tree: &mut S,
    suffix: ParallelSuffix,
    operand: &mut Operand,
) {
    if let Some(unit) = toggle_pe_sfp(operand.unit) {
        operand.unit = unit;
        if let Some(base) = operand.data.data_connect {
            operand.data.data_connect =
                Some(tree.intern_connect(MintedConnect::Parallel(base, suffix)));
        }
    }
    if let Some(storage) = toggle_pe_sfp(operand.storage) {
        operand.storage = storage;
    }
}

/// Replaces: e305_destRelatedToExternalNodes
///
/// Whether ANY destination datastream of the transfer is external (`:1700`) — entry 254 over every
/// destination.
///
/// ⚠️ TRAP: the reference declares `const dsc2::AllocateNode *allocNode;` and never reads it.
/// ⛔ A DISTINCT NAME because entry 254 is the same C++ name at a different arity.
#[must_use]
pub fn any_dest_related_to_external_nodes<E: ExternalStreams + ?Sized>(
    streams: &E,
    transfer: &TransferNode,
) -> bool {
    transfer
        .dsts
        .iter()
        .any(|dst| dest_related_to_external_nodes(streams, dst))
}

/// Replaces: e306_relatedToExternalNodes
///
/// Whether ANY input OR output datastream of the compute is external (`:1743`).
///
/// ⛔ A DISTINCT NAME: five C++ `relatedToExternalNodes` overloads differ only in their argument
/// type — entries 121, 306, 341 and 361 — and Rust must name each of them.
#[must_use]
pub fn compute_related_to_external_nodes<E: ExternalStreams + ?Sized>(
    streams: &E,
    compute: &ComputeNode,
) -> bool {
    input_related_to_external_nodes(streams, compute)
        || output_related_to_external_nodes(streams, compute)
}

// ⭐ TYPES FOR ENTRIES 339 AND 340 — the destination witness, the latch-link counter and the writes
// they add to [`FifoResults`]. Union them with this file's other vocabulary.

/// WHERE A TRANSFER'S RESULT IS SKIPPED TO — `convertResultToSkipReg`'s `useLatch` (`:909`).
///
/// ⭐ AN ENUM AND NOT A `bool`: the two arms test the SAME storage for OPPOSITE things when deciding
/// the result is already converted, so transposing a callsite must be a type error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipRegTarget {
    /// `useLatch == true` — the result becomes a `LATCH`, linked by a [`LatchDataId`].
    Latch,
    /// `useLatch == false` — the result becomes a FIFO on the component it is reached from.
    Fifo,
}

/// THE SOURCE OF EVERY LATCH LINK — `Ddc::latchDataIdCounter_` (`ddc/ddc.h:39`), bumped once per
/// latched result so that the producer and all of its consumers carry ONE id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LatchDataIds(u32);

impl LatchDataIds {
    /// `latchDataIdCounter_++` — this latch's id, and the counter advanced past it.
    pub const fn next(&mut self) -> LatchDataId {
        let id = LatchDataId(self.0);
        self.0 += 1;
        id
    }
}

/// ONE DESTINATION OF A TRANSFER ENTRIES 339 AND 340 MAY REPOINT AT ALL.
///
/// ⛔⛔ FOUR ABORTS COLLAPSE INTO THIS WITNESS — both entries open with *"Can not convert result of
/// external transferNode"* ([`InternalNode`]) and then
/// `DT_CHECK(destIndex < dstLdsAndLoopOffsets_.size())`. Their `return false` paths are NOT here:
/// those are each function's own answer and stay in the port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferDest {
    transfer: NodeId,
    dest: DestIdx,
    operand: Operand,
    reached_from: SenComponent,
}

impl TransferDest {
    /// The witness, or [`None`] for either abort.
    #[must_use]
    pub fn of(
        metadata: &Metadata,
        node: NodeId,
        transfer: &TransferNode,
        dest: DestIdx,
    ) -> Option<Self> {
        let internal = InternalNode::of(metadata, node)?;
        let index = dest.0 as usize;
        let operand = *transfer.dsts.get(index)?;
        Some(Self {
            transfer: internal.node(),
            dest,
            operand,
            reached_from: transfer
                .dsts
                .hops(index)
                .last()
                .copied()
                .unwrap_or(transfer.src.unit),
        })
    }

    /// The transfer.
    #[must_use]
    pub const fn transfer(self) -> NodeId {
        self.transfer
    }

    /// Which destination, as the reference's `dstVias_`/`dstLdsAndLoopOffsets_` index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.dest.0 as usize
    }

    /// `dstVias_.at(i).loc_` zipped with `dstLdsAndLoopOffsets_.at(i)`.
    #[must_use]
    pub const fn operand(self) -> Operand {
        self.operand
    }

    /// `dstVias_.at(i).via_.back()`, else `src_.unit_` — the component this destination is REACHED
    /// FROM, which is what both entries repoint a skipped result at.
    #[must_use]
    pub const fn reached_from(self) -> SenComponent {
        self.reached_from
    }
}

/// A NODE PROVED NOT TO BE IN THE SCHEDULE TREE YET — `DT_CHECK(computeNode->getPrev() == nullptr)`
/// (`:1035`), which is entry 340's one remaining abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnplacedNode(NodeId);

impl UnplacedNode {
    /// The witness, or [`None`] where the node already has a parent.
    #[must_use]
    pub fn of<S: ScheduleSurgery + ?Sized>(surgery: &S, node: NodeId) -> Option<Self> {
        surgery.parent(node).is_none().then_some(Self(node))
    }

    /// The node.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.0
    }
}

/// WHAT ENTRIES 339 AND 340 ADDITIONALLY WRITE — the latch links and the compute's operand vectors.
pub trait SkipRegResults: FifoResults {
    /// `transferNode->dstLdsAndLoopOffsets_.at(i).latchDataId_ = id`.
    fn set_dst_latch_data_id(&mut self, transfer: NodeId, dst: usize, id: LatchDataId);

    /// `transferConsumer->srcLdsAndLoopOffsets_.latchDataId_ = id`.
    fn set_src_latch_data_id(&mut self, transfer: NodeId, id: LatchDataId);

    /// `computeConsumer->inputsLdsAndLoopOffsets_.at(i).latchDataId_ = id`.
    fn set_compute_input_latch_data_id(&mut self, compute: NodeId, input: usize, id: LatchDataId);

    /// `allocNode->removeAllocUser(user)` (`dsc/dsc2.h:1024`) — its *"Schedule node <n> is not in
    /// the user list of allocate node <a>"* abort is [`AllocationUse`] missing. ⛔ NOT
    /// [`DscAllocations::reduce_users_or_delete`], which drops a reference AND may unlink the node.
    fn remove_alloc_use(&mut self, alloc_use: AllocationUse);

    /// `computeNode->outputs_.resize(1)` with `outputsLdsAndLoopOffsets_`, then `at(0) = (unit,
    /// data)`. ⚠️ EVERY OTHER OUTPUT IS DROPPED. `unit` is an `outputs_` entry and so a component,
    /// which is why the reference writes a transfer's `storage_` into it.
    fn set_sole_compute_output(&mut self, compute: NodeId, unit: SenComponent, data: DataInfo);

    /// `computeNode->inputs_.resize(i + 1)` with `inputsLdsAndLoopOffsets_`, then `at(i) = (unit,
    /// data)`. ⚠️ THE RESIZE GOES BOTH WAYS: inputs past `i` are DROPPED, and a gap below it is
    /// `NO_COMPONENT` against a default-constructed `DataInfo`.
    fn resize_compute_inputs_to(
        &mut self,
        compute: NodeId,
        input: InputIdx,
        unit: SenComponent,
        data: DataInfo,
    );
}

/// One `reduceUsersOrDeleteAllocationAndMetadata(di, storage, user)` — a no-op where the operand
/// keys no allocation, which is the abort inside its own `getMutableAllocation`.
fn reduce_users_of<D: SkipRegResults + DscAllocations + ?Sized>(
    dsc: &mut D,
    metadata: &mut Metadata,
    data: DataInfo,
    storage: SenComponent,
    user: NodeId,
) {
    if let Some(alloc_use) = data_origin(data)
        .zip(component_memory(storage))
        .and_then(|(origin, memory)| AllocationUse::of(dsc, origin, memory, user))
    {
        reduce_users_or_delete_allocation_and_metadata(dsc, metadata, alloc_use);
    }
}

/// Replaces: e339_convertResultToSkipReg
///
/// MOVES ONE TRANSFER DESTINATION OUT OF ITS REGISTER FILE into a latch or a FIFO (`:909`): drops
/// the register allocation, repoints the destination at `LATCH` or at the component it is reached
/// from, and repoints every consumer of its connect under one shared fresh [`LatchDataId`].
///
/// ⛔ `true` AND NOTHING DONE where the destination ALREADY skips the register file; any other
/// non-register, non-`LXLUVALUE` storage is `false`. ⚠️ TRAP: an `LXLUVALUE` destination keeps its
/// own allocation and its COMPUTE consumers' inputs' (`:999`) but NEVER a transfer consumer's source,
/// which is dropped either way (`:985`); *"Unsupported consumer type"* becomes a skipped consumer.
pub fn convert_result_to_skip_reg<D>(
    dsc: &mut D,
    metadata: &mut Metadata,
    latch_ids: &mut LatchDataIds,
    dest: TransferDest,
    target: SkipRegTarget,
) -> bool
where
    D: SkipRegResults + DscAllocations + ExternalStreams + ?Sized,
{
    let dst = dest.operand();
    let is_lxlu_value = dst.unit == SenComponent::Lxluvalue;

    if !is_register(dst.storage) && !is_lxlu_value {
        return match target {
            SkipRegTarget::Latch => dst.storage == SenComponent::Latch,
            SkipRegTarget::Fifo => {
                dst.storage != SenComponent::Latch && !is_memory(dst.storage)
            }
        };
    }
    let connect = dst.data.data_connect;

    if !is_lxlu_value {
        let keyed = data_origin(dst.data).zip(component_memory(dst.storage));
        let Some((origin, memory)) = keyed else {
            return false;
        };
        if dsc.allocation_in(origin, memory).is_none() {
            return false;
        }
        if dest_related_to_external_nodes(dsc, &dst) {
            return false;
        }
        reduce_users_of(dsc, metadata, dst.data, dst.storage, dest.transfer());
    }

    let (new_component, latch_id) = match target {
        SkipRegTarget::Latch => {
            let id = latch_ids.next();
            dsc.set_dst_latch_data_id(dest.transfer(), dest.index(), id);
            (SenComponent::Latch, Some(id))
        }
        SkipRegTarget::Fifo => (dest.reached_from(), None),
    };
    dsc.set_dst_storage(dest.transfer(), dest.index(), new_component);

    for consumer in dsc.connect_consumers(connect) {
        match consumer {
            FifoConsumer::Transfer(node) => {
                let src = dsc.transfer(node).src;
                reduce_users_of(dsc, metadata, src.data, src.storage, node);
                dsc.set_src_storage(node, new_component);
                if let Some(id) = latch_id {
                    dsc.set_src_latch_data_id(node, id);
                }
            }
            FifoConsumer::Compute(node, compute) => {
                for (input, operand) in compute.inputs.iter().enumerate() {
                    if operand.data.data_connect != connect {
                        continue;
                    }
                    if !is_lxlu_value {
                        reduce_users_of(dsc, metadata, operand.data, operand.unit, node);
                    }
                    dsc.set_compute_input_unit(node, input, new_component);
                    if let Some(id) = latch_id {
                        dsc.set_compute_input_latch_data_id(node, input, id);
                    }
                }
            }
            FifoConsumer::Other(_) => {}
        }
    }

    true
}

/// Replaces: e340_insertComputeBetweenTransferAndReg
///
/// PUTS A COMPUTE BETWEEN A TRANSFER AND ITS REGISTER DESTINATION (`:1021`): places it right after
/// the transfer, hands it the register allocation as its SOLE output, and gives it the component the
/// destination is reached from as `input`, which is what the transfer now writes instead.
///
/// ⚠️ TRAP: THE PLACEMENT HAPPENS FIRST, so every `false` — a non-register destination, a missing
/// allocation, an external destination — leaves the compute already in the tree beside the transfer.
/// ⚠️ TRAP: the reference resizes, so outputs past the first and inputs past `input` are DROPPED.
pub fn insert_compute_between_transfer_and_reg<D>(
    dsc: &mut D,
    dest: TransferDest,
    compute: UnplacedNode,
    input: InputIdx,
) -> bool
where
    D: SkipRegResults + DscAllocations + ExternalStreams + ?Sized,
{
    let compute = compute.node();
    dsc.add_child_node(compute, InsertionPoint::After(dest.transfer()));

    let dst = dest.operand();
    if !is_register(dst.storage) {
        return false;
    }
    let keyed = data_origin(dst.data).zip(component_memory(dst.storage));
    let Some((origin, memory)) = keyed else {
        return false;
    };
    let Some(alloc) = dsc.allocation_in(origin, memory) else {
        return false;
    };
    if dest_related_to_external_nodes(dsc, &dst) {
        return false;
    }

    dsc.add_alloc_user(alloc, compute);
    if let Some(alloc_use) = AllocationUse::of(dsc, origin, memory, dest.transfer()) {
        dsc.remove_alloc_use(alloc_use);
    }

    dsc.set_sole_compute_output(compute, dst.storage, dst.data);
    let new_component = dest.reached_from();
    dsc.set_dst_storage(dest.transfer(), dest.index(), new_component);
    dsc.resize_compute_inputs_to(compute, input, new_component, dst.data);

    true
}

/// Replaces: e341_relatedToExternalNodes
///
/// Whether EITHER END of the transfer touches an external datastream (`:1712`) — entry 253 on the
/// source, entry 305 across every destination.
///
/// ⛔ A DISTINCT NAME: five C++ `relatedToExternalNodes` overloads differ only in their argument
/// type — entries 121, 306, 341 and 361 — and Rust must name each of them.
#[must_use]
pub fn transfer_related_to_external_nodes<E: ExternalStreams + ?Sized>(
    streams: &E,
    transfer: &TransferNode,
) -> bool {
    src_related_to_external_nodes(streams, transfer)
        || any_dest_related_to_external_nodes(streams, transfer)
}

/// ONE NODE OF THE SUBTREE ENTRY 361 WALKS — the three `nodeType_` arms of its one
/// `{ALLOCATE, COMPUTE, TRANSFER}` traversal, each carrying what its own arm reads.
///
/// ⛔ THE BODY TRAVELS WITH THE NODE because the two live arms ask entries 306 and 341, which take
/// the compute and the transfer themselves; an allocate is asked nothing but `isExternalNode`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubtreeNode {
    /// `ALLOCATE`.
    Allocate(NodeId),
    /// `COMPUTE`, whose datastreams entry 306 reads.
    Compute(NodeId, ComputeNode),
    /// `TRANSFER`, whose ends entry 341 reads.
    Transfer(NodeId, TransferNode),
}

impl SubtreeNode {
    /// The schedule node itself, which is what `isExternalNode` is asked about.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        match self {
            Self::Allocate(node) | Self::Compute(node, _) | Self::Transfer(node, _) => *node,
        }
    }
}

/// WHAT ENTRY 361 ASKS OF THE SCHEDULE TREE.
pub trait ExternalSubtree: ExternalStreams {
    /// `scheduleTree_.traverseTreeDFS(root, {ALLOCATE, COMPUTE, TRANSFER})` (`:1758`).
    fn subtree_nodes(&self, root: BlockId) -> Vec<SubtreeNode>;
}

/// Replaces: e361_relatedToExternalNodes
///
/// Whether ANYTHING UNDER THIS BLOCK touches an external datastream (`:1757`) — `isExternalNode` on
/// every allocate, compute and transfer under `root`, then entry 306 on the computes and entry 341
/// on the transfers.
///
/// ⛔ A DISTINCT NAME, as entries 121/306/341 each needed one. ⚠️ TRAP: ENTRY 121 IS UNREACHABLE
/// FROM HERE — the traversal filter admits no `SYNC` node, however the call graph reads.
#[must_use]
pub fn subtree_related_to_external_nodes<S: ExternalSubtree + ?Sized>(
    tree: &S,
    metadata: &Metadata,
    root: BlockId,
) -> bool {
    tree.subtree_nodes(root).into_iter().any(|node| {
        metadata.external_nodes.contains(&node.node())
            || match &node {
                SubtreeNode::Allocate(_) => false,
                SubtreeNode::Compute(_, compute) => compute_related_to_external_nodes(tree, compute),
                SubtreeNode::Transfer(_, transfer) => {
                    transfer_related_to_external_nodes(tree, transfer)
                }
            }
    })
}

// ⭐ USES FOR ENTRIES 255-257: `DataInfo`, `OperandPos` and `SenComponent`, added to this file's top
// block. TYPES FOR ENTRIES 255-257 follow; union them with this file's other vocabulary as its
// remaining entries land.

/// WHICH WAY A DATASTREAM FLOWS PAST ITS NODE — `storageOrDatastreamIsExternal`'s `isIncoming`
/// (`ddc/ddc_transformation_util.cpp:1652`), which selects the data connect's `producers_` or its
/// `consumers_` and spells itself `"Incoming"`/`"Outgoing"` in the transformation report.
///
/// ⭐ AN ENUM AND NOT A `bool`: a direction is a closed set, and each of entries 253-256 passes one
/// literal, so transposing two callsites must be a type error rather than an inverted predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StreamDirection {
    /// `isIncoming == true` — the operand READS, so the connect's `producers_` are checked.
    Incoming,
    /// `isIncoming == false` — the operand WRITES, so its `consumers_` are.
    Outgoing,
}

/// WHAT ENTRIES 253-256 ASK OF ENTRY 120 — `storageOrDatastreamIsExternal(dataInfo, storage,
/// isIncoming)` (`ddc/ddc_transformation_util.cpp:1652`): whether that datastream's allocation in
/// `storage`, or any node on its `data_connect=`, is an `externalNodes_` member (`ddc/ddc.h:298`).
///
/// ⛔ THE SEAM ENTRIES 253-256 REACH IT THROUGH, AND NOT A STAND-IN PREDICATE: entry 120 reads
/// `currDsc`'s allocations, `metadata.dataConnects_` and `metadata.externalNodes_` at once, which is
/// a carrier's question however it is spelled, and entries 305/306/341/361 ask through it too.
/// ⭐ [`storage_or_datastream_is_external`] IS THE PORT, and it is the one implementation of this
/// method: a carrier states [`CensusNodes`] beside its allocations and delegates.
pub trait ExternalStreams {
    /// `storageOrDatastreamIsExternal(dataInfo, storage, isIncoming)`.
    fn storage_or_datastream_is_external(
        &self,
        data: DataInfo,
        storage: SenComponent,
        direction: StreamDirection,
    ) -> bool;
}

/// Replaces: e255_inputRelatedToExternalNodes
///
/// Whether ANY input datastream of the compute is external (`:1717`).
///
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: THE OPERAND'S COMPONENT — [`Operand::unit`], the `inputs_`
/// entry — IS PASSED AS ENTRY 120'S `storage`, so `Operand::storage` is NOT what is asked about.
/// ⛔ The reference bounds the loop by `inputsLdsAndLoopOffsets_.size()` while indexing
/// `inputs_.at(i)`; [`ComputeNode::inputs`] zips the two, so that throw is unspellable.
#[must_use]
pub fn input_related_to_external_nodes<E: ExternalStreams + ?Sized>(
    streams: &E,
    compute: &ComputeNode,
) -> bool {
    compute.inputs.iter().any(|input| {
        streams.storage_or_datastream_is_external(input.data, input.unit, StreamDirection::Incoming)
    })
}

/// Replaces: e256_outputRelatedToExternalNodes
///
/// Whether ANY output datastream of the compute is external (`:1730`) — entry 255 over `outputs_`,
/// asked of the connect's `consumers_` instead of its `producers_`.
///
/// ⛔ THE SAME TRAP: `outputs_.at(i)` is the operand's component and it is passed as the `storage`.
#[must_use]
pub fn output_related_to_external_nodes<E: ExternalStreams + ?Sized>(
    streams: &E,
    compute: &ComputeNode,
) -> bool {
    compute.outputs.iter().any(|output| {
        streams.storage_or_datastream_is_external(
            output.data,
            output.unit,
            StreamDirection::Outgoing,
        )
    })
}

/// ONE `labeledDs_` ENTRY AS ENTRY 257 TOUCHES ONE — `LabeledDsInfo` (`dsc/dscdefn.h:321`), whose
/// body is COPIED WHOLESALE and whose `ldsIdx_` is the only field read.
///
/// ⭐ OPAQUE ON PURPOSE. `addNewLds` clones `*refLds` and writes two indices into the clone; one of
/// its callsites builds the source locally rather than taking it out of `labeledDs_`
/// (`ddc/ddc_transformation.cpp:1907`), so the source cannot be narrowed to a position.
pub trait LabeledDsEntry: Clone {
    /// `ldsIdx_` (`dsc/dscdefn.h:323`) — the entry's OWN self-index, whose default is `183` and which
    /// need not be the position the entry sits at.
    fn recorded_lds_idx(&self) -> LdsIdx;

    /// `ldsIdx_ = recorded`.
    fn set_recorded_lds_idx(&mut self, recorded: LdsIdx);

    /// `referenceLdsIdx_ = reference` (`:324`) — which tensor an internal one is derived from.
    fn set_reference_lds_idx(&mut self, reference: LdsIdx);
}

/// A PLACE IN THE SCHEDULE TREE THAT NAMES A LABELLED DATA STRUCTURE — every lds index the
/// `{ALLOCATE, TRANSFER, COMPUTE, LOOP}` walk of entry 257 reaches (`:1878-1911`).
///
/// ⛔ `DT_ERROR("Invalid block node.")` (`:1911`) IS UNSPELLABLE: the walk's filter admits exactly
/// these four `nodeType_`s, so there is no fifth arm left to refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeLdsSlot {
    /// `allocNode->ldsIdx_`.
    Allocate(NodeId),
    /// `computeNode->inputsLdsAndLoopOffsets_.at(i).myLdsIdx_` for [`OperandPos::Input`], and
    /// `outputsLdsAndLoopOffsets_.at(i).myLdsIdx_` for [`OperandPos::Output`].
    Compute(NodeId, OperandPos),
    /// `transferNode->srcLdsAndLoopOffsets_.myLdsIdx_` for [`OperandPos::Input`] — a transfer has
    /// exactly one source — and `dstLdsAndLoopOffsets_.at(i).myLdsIdx_` for [`OperandPos::Output`].
    Transfer(NodeId, OperandPos),
    /// `loopNode->parametricLdsIdx_` (`dsc/dsc2.h:618`), reached ONLY where `isParametricLoop()`
    /// (`:599`) holds — a non-parametric loop's index is left alone.
    ParametricLoop(NodeId),
}

/// WHAT ENTRY 257 DOES TO THE DSC AND ITS SCHEDULE TREE — the insert, the index writes and the walk
/// that finds every slot naming a labelled data structure.
///
/// ⭐ EVERY MUTATION THE REFERENCE PERFORMS IS ONE METHOD HERE; the DECISION of which to call, and
/// with what value, stays in the port.
pub trait NewLabeledDs {
    /// One `labeledDs_` entry as the carrier holds it.
    type Entry: LabeledDsEntry;

    /// `labeledDs_.size() - 1`, TOTAL because the list is non-empty — the reference reaches
    /// `labeledDs_.at(labeledDs_.size() - 1)` and `.back()` unguarded (`:1817-1818`).
    fn last_lds_pos(&self) -> LdsIdx;

    /// `labeledDs_.back().ldsIdx_`, likewise total.
    fn last_recorded_lds_idx(&self) -> LdsIdx;

    /// `labeledDs_.back().ldsIdx_ = recorded`.
    fn set_last_recorded_lds_idx(&mut self, recorded: LdsIdx);

    /// `labeledDs_.insert(labeledDs_.end() - 1, std::move(entry))` — the entry lands at position
    /// `size() - 1` and the entry that was last moves up one place.
    ///
    /// ⛔ REPOINTING IS THIS METHOD'S JOB, NOT THE PORT'S. The reference rebuilds every
    /// `LabeledDsInfo*` held by `labeledDs_.dataTransfers_.myDsInfo` and by `computeOp_`'s
    /// `inputLabeledDs`/`interimLabeledDs`/`outputLabeledDs` (`:1826-1845`) only because the vector
    /// reallocated; positions `0 .. size()-2` keep the entries they had, so a carrier that holds
    /// those references BY POSITION owes exactly the one shift of the entry that was last.
    /// ⛔ NOTHING MAY POINT AT THE NEW ENTRY: its callers push it on themselves
    /// (`ddc/ddcv1.cpp:2104-2105`).
    fn insert_lds_before_last(&mut self, entry: Self::Entry);

    /// `labeledDs_.at(pos).memOrg_.clear()` — a NO-OP past the end, which is that `.at()`'s throw.
    fn clear_mem_org(&mut self, pos: LdsIdx);

    /// The `allocateNode_`s of `labeledDs_.at(pos).memOrg_` (`dsc/dscdefn.h:314`) — EMPTY past the
    /// end, and empty for an organisation whose allocation is still null.
    fn mem_org_allocations(&self, pos: LdsIdx) -> Vec<AllocId>;

    /// `allocNode->ldsIdx_`, absent for the reference's `-1`.
    fn alloc_lds_idx(&self, alloc: AllocId) -> Option<LdsIdx>;

    /// `allocNode->ldsIdx_ = lds`.
    fn set_alloc_lds_idx(&mut self, alloc: AllocId, lds: LdsIdx);

    /// Every slot the `{ALLOCATE, TRANSFER, COMPUTE, LOOP}` walk reaches WITH the lds index it holds,
    /// absent for the reference's `-1`.
    fn tree_lds_slots(&self) -> Vec<(TreeLdsSlot, Option<LdsIdx>)>;

    /// That slot's `myLdsIdx_`/`ldsIdx_`/`parametricLdsIdx_` set to `lds`.
    fn set_tree_lds(&mut self, slot: TreeLdsSlot, lds: LdsIdx);

    /// The `isOpaqueOp_` computes that same walk reaches — the `metadata.opaqueOps_` keys entry 257
    /// looks up (`:1884-1887`). ⛔ One with NO entry in that map is `opaqueOps_.at(cn)`'s throw, and
    /// nothing to rename here.
    fn opaque_computes(&self) -> Vec<NodeId>;
}

/// Replaces: e257_addNewLds
///
/// INSERTS A COPY OF `ref_lds` BEFORE THE LAST LABELLED DATA STRUCTURE (`:1811`): the copy takes the
/// index the last one had, that one is bumped by one, and the vacated index is RENAMED wherever it is
/// held — allocations, tree lds slots, opaque ops, `newAllocations_`, `ldsIdxAfterDdc`. Yields the new
/// entry's POSITION (`ddc/ddcv1.cpp:2103-2104`). ⛔⛔ TRAP, AND IT IS THE REFERENCE'S: A RECORDED INDEX
/// IS USED AS A POSITION — `labeledDs_.at(newLastLdsIdx).memOrg_` (`:1873`) indexes the vector BY the
/// bumped recorded index, and is cleared AFTER the rename it feeds, which on a drifted list decides.
#[must_use]
pub fn add_new_lds<D: NewLabeledDs + ?Sized>(
    dsc: &mut D,
    metadata: &mut Metadata,
    ref_lds: &D::Entry,
) -> LdsIdx {
    // `newLds.ldsIdx_ = currDsc->labeledDs_.size() - 1` — the POSITION the insert puts it at.
    let new_recorded = dsc.last_lds_pos();
    let old_last = dsc.last_recorded_lds_idx();
    // `currDsc->labeledDs_.back().ldsIdx_++`.
    dsc.set_last_recorded_lds_idx(LdsIdx(old_last.0 + 1));

    let mut new_lds = ref_lds.clone();
    new_lds.set_recorded_lds_idx(new_recorded);
    new_lds.set_reference_lds_idx(ref_lds.recorded_lds_idx());
    dsc.insert_lds_before_last(new_lds);

    // `newLastLdsIdx = currDsc->labeledDs_.at(currDsc->labeledDs_.size() - 1).ldsIdx_`, read back off
    // the entry the insert shifted up.
    let new_last = dsc.last_recorded_lds_idx();

    for alloc in dsc.mem_org_allocations(new_last) {
        dsc.set_alloc_lds_idx(alloc, new_last);
    }
    dsc.clear_mem_org(new_recorded);

    for (slot, held) in dsc.tree_lds_slots() {
        if held == Some(old_last) {
            dsc.set_tree_lds(slot, new_last);
        }
    }
    for compute in dsc.opaque_computes() {
        if let Some(opaque) = metadata.opaque_ops.get_mut(&compute) {
            if opaque.lds_idx == Some(old_last) {
                opaque.lds_idx = Some(new_last);
            }
        }
    }

    for allocated in metadata.new_allocations.values_mut() {
        if let Some(alloc) = allocated.lds_idx_and_alloc_node.remove(&old_last) {
            allocated.lds_idx_and_alloc_node.insert(new_last, alloc);
        }
        for &alloc in allocated.comp_and_alloc_node.values() {
            if dsc.alloc_lds_idx(alloc) == Some(old_last) {
                dsc.set_alloc_lds_idx(alloc, new_last);
            }
        }
    }

    // Update DDL interface: the entry KEYED by the old index takes the new one as its value, and
    // failing that every entry whose value was the old index is remapped (`:1904-1911`).
    if let Some(after) = metadata.lds_idx_after_ddc.get_mut(&old_last) {
        *after = new_last;
    } else {
        for after in metadata.lds_idx_after_ddc.values_mut() {
            if *after == old_last {
                *after = new_last;
            }
        }
    }

    new_recorded
}

// ⭐ USES FOR ENTRIES 247-254: `Strategy`, `Operand`, `Core`/`Corelet`, `LoopId`, `CoreClSet` and
// `SymbolicDimInfo`, added to this file's top block. TYPES FOR ENTRIES 247-254 follow; union them
// with this file's other vocabulary as its remaining entries land.

/// `dsc2::memories` — the SIXTEEN storages a result may live in (`dsc/dscdefn.cpp:142-144`).
///
/// ⭐ ONE TABLE, TWO READERS: `hasNonMemoryResult()` (`dsc/dsc2.cpp:4368`) and entry 250's FIFO scan
/// both ask it, and a destination storage OUTSIDE it is exactly what "a FIFO destination" means.
/// ⛔ NOT [`DdcMemory`]'s eight (`ddc::memories`, `ddc/ddc_metadata.h:20-21`), a different and
/// smaller set — [`DdcAllocateNode::component`] carries that one and says so.
#[must_use]
pub const fn is_memory(storage: SenComponent) -> bool {
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

/// The `SenComponents` one [`DdcMemory`] IS — `ddc::memories`' own spelling read the other way, which
/// is what lets entry 250 write an allocation's `component_` back into a `DataLocation`.
pub const fn memory_component(memory: DdcMemory) -> SenComponent {
    match memory {
        DdcMemory::Lx => SenComponent::Lx,
        DdcMemory::L0 => SenComponent::L0,
        DdcMemory::L0Scale => SenComponent::L0Scale,
        DdcMemory::PeLrf => SenComponent::Pelrf,
        DdcMemory::SfpLrf => SenComponent::Sfplrf,
        DdcMemory::PtaRf => SenComponent::Ptarf,
        DdcMemory::PtxRf => SenComponent::Ptxrf,
        DdcMemory::PtiRf => SenComponent::Ptirf,
        DdcMemory::Hbm => SenComponent::Hbm,
    }
}

/// `Ddc::registerComponents` — the SEVEN storages a result is held in AS A REGISTER
/// (`ddc/ddcv1.cpp:17-18`).
///
/// ⭐ NEITHER A SUBSET NOR A SUPERSET OF [`is_memory`]: PELRF, SFPLRF, PTARF, PTXRF, PESTATE,
/// SFPSTATE and LXLUSCALEREG are all `dsc2::memories` too, and PTIRF is a memory that is not a
/// register — so entries 339 and 340 ask this one and not that one.
#[must_use]
pub const fn is_register(storage: SenComponent) -> bool {
    matches!(
        storage,
        SenComponent::Pelrf
            | SenComponent::Sfplrf
            | SenComponent::Ptarf
            | SenComponent::Ptxrf
            | SenComponent::Pestate
            | SenComponent::Sfpstate
            | SenComponent::Lxluscalereg
    )
}

/// The [`DdcMemory`] one storage IS — [`memory_component`] read the other way, which is how a
/// `DataLocation`'s storage becomes an allocation key.
///
/// ⛔ NARROWER THAN `getAllocation`'S OWN MEMORY TEST, which admits all sixteen of `dsc2::memories`
/// ([`is_memory`]) — the seven with no [`DdcMemory`] spelling cannot key
/// [`DscAllocations::allocation_in`], so each answers exactly as an absent `memOrg_` entry does:
/// *"missing allocation"*, which is `getAllocation(.., allowMissingAlloc=true)`'s own `nullptr`.
#[must_use]
pub const fn component_memory(storage: SenComponent) -> Option<DdcMemory> {
    match storage {
        SenComponent::Lx => Some(DdcMemory::Lx),
        SenComponent::L0 => Some(DdcMemory::L0),
        SenComponent::L0Scale => Some(DdcMemory::L0Scale),
        SenComponent::Pelrf => Some(DdcMemory::PeLrf),
        SenComponent::Sfplrf => Some(DdcMemory::SfpLrf),
        SenComponent::Ptarf => Some(DdcMemory::PtaRf),
        SenComponent::Ptxrf => Some(DdcMemory::PtxRf),
        SenComponent::Ptirf => Some(DdcMemory::PtiRf),
        SenComponent::Hbm => Some(DdcMemory::Hbm),
        _ => None,
    }
}

/// THE KEY `getAllocation(di, ..)` READS OFF ONE OPERAND — `myLdsIdx_`, else `constantId_`
/// (`dsc/dsc2.cpp:2589`), [`None`] where the reference has neither and answers *"One of myLdsIdx or
/// constantId must be set"*.
#[must_use]
pub const fn data_origin(data: DataInfo) -> Option<DataOrigin> {
    match (data.my_lds_idx, data.constant_id) {
        (Some(lds), _) => Some(DataOrigin::LabeledDs(lds)),
        (None, Some(cons)) => Some(DataOrigin::Constant(cons)),
        (None, None) => None,
    }
}

/// A NODE PROVED NOT TO BE IN `externalNodes_` — `isExternalNode(node)` (`ddc/ddc.h:298`).
///
/// ⛔⛔ ONE TYPE FOR SIX ABORTS: every one of entries 247-252 opens with
/// `if (isExternalNode(x)) DT_ERROR("Can not <verb> external <x>")`, and each of those is this
/// witness missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InternalNode(NodeId);

impl InternalNode {
    /// The witness, or [`None`] where the node is external.
    #[must_use]
    pub fn of(metadata: &Metadata, node: NodeId) -> Option<Self> {
        (!metadata.external_nodes.contains(&node)).then_some(Self(node))
    }

    /// The node.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.0
    }
}

/// WHETHER DATASTAGE EXPLORATION HAS FINISHED — `dataStageExplorationDone_` (`ddc/ddc.h:352`), after
/// which no unit may mint a datastage or an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatastageExploration {
    /// `false` — datastages may still be minted.
    Open,
    /// `true`.
    Done,
}

/// WHERE A NODE IS PLACED AMONG A PARENT'S CHILDREN — `addChildNode`'s `(addBefore,
/// siblingRefNode)` pair (`dsc/dsc2.cpp:2010`).
///
/// ⛔ *"Sibling reference node not found in parent node"* IS UNSPELLABLE: naming a sibling names its
/// parent too, so the two cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertionPoint {
    /// `(true, sibling)` — immediately before `sibling`, among its own parent's children.
    Before(NodeId),
    /// `(false, sibling)` — immediately after it.
    After(NodeId),
    /// `(true, nullptr)` — the FRONT of that parent's children.
    FirstIn(NodeId),
    /// `(false, nullptr)` — the BACK, which is `addChildNode`'s and `moveNode`'s own default.
    LastIn(NodeId),
}

/// WHAT ENTRIES 247-252 DO TO THE SCHEDULE TREE — the mint, the placement and the move.
///
/// ⭐ EVERY MUTATION THE REFERENCE PERFORMS IS ONE METHOD HERE; the DECISION of which to call, in
/// what order and with what value stays in the port.
pub trait ScheduleSurgery {
    /// `node->name_`.
    fn node_name(&self, node: NodeId) -> NodeName;

    /// `node->name_ = name`.
    fn set_node_name(&mut self, node: NodeId, name: NodeName);

    /// `node->getPrev()` (`dsc/dsc2.h:463`) — the node's PARENT, absent for the schedule head.
    fn parent(&self, node: NodeId) -> Option<NodeId>;

    /// `node->getOwnerLoop()` (`dsc/dsc2.cpp:1892`) — the nearest `LOOP` ancestor, walking `prev_`.
    fn owner_loop(&self, node: NodeId) -> Option<LoopId>;

    /// `transferNode->..` — the transfer at that node, as [`super::transformation::TransferWalk`]
    /// also reads one.
    fn transfer(&self, node: NodeId) -> TransferNode;

    /// `loopNode->numId_`.
    fn loop_num(&self, loop_node: LoopId) -> DatastageId;

    /// `loopNode->denId_`.
    fn loop_den(&self, loop_node: LoopId) -> DatastageId;

    /// `loopNode->dims_` (`dsc/dsc2.h:601`), ordered inner to outer.
    fn loop_dims(&self, loop_node: LoopId) -> LoopDims;

    /// `loopNode->isParametricLoop()` (`dsc/dsc2.h:599`).
    fn is_parametric(&self, loop_node: LoopId) -> bool;

    /// `new dsc2::LoopNode(..)` — an UNPARENTED loop, which is all [`construct_loop_node`] mints.
    fn new_loop(&mut self, loop_node: LoopNode) -> LoopId;

    /// `new dsc2::BlockNode()` with its `name_` — a fresh block has NO children.
    fn new_block(&mut self, name: NodeName) -> NodeId;

    /// `addChildNode(node, ..)` (`dsc/dsc2.cpp:2010`).
    fn add_child_node(&mut self, node: NodeId, at: InsertionPoint);

    /// `node->moveNode(currDsc, ..)` (`dsc/dsc2.cpp:1977`) — unlinked from its old parent first.
    fn move_node(&mut self, node: NodeId, at: InsertionPoint);

    /// `scheduleTree_.traverseTreeDFSMutable(root, {CONDITION})`.
    fn conditions_under(&self, root: NodeId) -> Vec<NodeId>;

    /// `condNode->loopCond_`.
    fn loop_cond(&self, condition: NodeId) -> LoopCondComposite;

    /// `condNode->loopCond_ = cond`.
    fn set_loop_cond(&mut self, condition: NodeId, cond: LoopCondComposite);
}

/// WHAT ENTRIES 247 AND 248 ADDITIONALLY DO — rewriting one loop band in place.
pub trait LoopBands: ScheduleSurgery {
    /// `loopNode->denId_ = den`.
    fn set_loop_den(&mut self, loop_node: LoopId, den: DatastageId);

    /// `loopNode->dims_ = dims`.
    fn set_loop_dims(&mut self, loop_node: LoopId, dims: LoopDims);

    /// `from->moveChildren(to)` (`dsc/dsc2.cpp:2043`) — every child re-parented, in order.
    fn move_children(&mut self, from: NodeId, to: NodeId);

    /// `base->insertPerfectlyNestedBlockNode(nested)` — [`Self::move_children`] and then the one
    /// child. ⛔ *"Nested node must not have any children."* cannot hold for a freshly minted loop.
    fn insert_perfectly_nested(&mut self, base: LoopId, nested: LoopId);

    /// `condNode->loopCond_.adjustConditionForSplitLoop(orig, new_loops)` (`dsc/dsc2.cpp:2061`).
    ///
    /// ⛔ ASKED FOR AND NOT PORTED HERE: it is a `dsc/` function, outside this campaign's file list,
    /// and it is where the `EQ`/`NE`/`(GT,FIRST)`/`(LT,LAST)` rewrite and its four aborts live.
    fn adjust_condition_for_split_loop(
        &mut self,
        condition: NodeId,
        orig: LoopId,
        new_loops: &[LoopId],
    );
}

/// WHERE THE BASE LOOP'S DIMS THAT NO SPLIT SET NAMED GO — `unspecifiedDimsInnermost`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnspecifiedDims {
    /// `true` — they become the innermost loop of the new nest.
    Innermost,
    /// `false` — the outermost.
    Outermost,
}

/// A LOOP BAND SPLIT PROVED WELL FORMED — the dim sets outermost to innermost, each drawn from the
/// base loop's own dims, no dim in two of them, plus whatever dims no set named.
///
/// ⛔⛔ THREE ABORTS AND TWO OUT-OF-BOUNDS READS GONE. The three are the external-loop refusal,
/// *"Base loop is not associated with (d, k)"* and *"Overlapping dimension sets specified for
/// loop-splitting."*; requiring at least one set closes `newLoops[1]` and
/// `baseLoop->moveChildren(baseLoop)` on the empty input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopBandSplit {
    base: LoopId,
    sets: Vec<LoopDims>,
}

impl LoopBandSplit {
    /// The split, or [`None`] for any of the three refusals; `sets` is outermost to innermost, and
    /// the base loop's dims no set names become one more set placed per `unspecified`.
    ///
    /// ⚠️ TRAP: the reference reads those remaining dims out of an `unordered_map`, so their order is
    /// unspecified there; here it is the base loop's own dims order.
    #[must_use]
    pub fn of<S: ScheduleSurgery + ?Sized>(
        tree: &S,
        metadata: &Metadata,
        base: LoopId,
        sets: &[LoopDims],
        unspecified: UnspecifiedDims,
    ) -> Option<Self> {
        InternalNode::of(metadata, base.0)?;
        let Some((first_set, other_sets)) = sets.split_first() else {
            return None;
        };
        let base_dims = tree.loop_dims(base);
        let mut named: BTreeSet<PrimaryDimAndKind> = BTreeSet::new();
        for set in core::iter::once(first_set).chain(other_sets) {
            for dim in set.iter() {
                if !base_dims.iter().any(|base| base == dim) {
                    return None;
                }
                if !named.insert(dim) {
                    return None;
                }
            }
        }

        let mut sets = sets.to_vec();
        let remaining: Vec<PrimaryDimAndKind> = base_dims
            .iter()
            .filter(|dim| !named.contains(dim))
            .collect();
        if let Some((first, rest)) = remaining.split_first() {
            let dims = LoopDims::new(*first, rest.to_vec());
            match unspecified {
                UnspecifiedDims::Innermost => sets.push(dims),
                UnspecifiedDims::Outermost => sets.insert(0, dims),
            }
        }

        Some(Self { base, sets })
    }
}

/// Replaces: e247_splitLoopBandOnDim
///
/// SPLITS ONE LOOP INTO A PERFECT NEST, one loop per dim set: the base loop keeps the outermost set
/// and gains `__split` in its name, each further set becomes a fresh loop over the SAME
/// numerator/denominator pair nested inside the last, the base loop's children move into the
/// innermost one, and every CONDITION now under it that compared against the base loop is
/// re-pointed at whichever new loop carries its dim (`:159`). Yields the INNERMOST new loop.
///
/// ⚠️ TRAP: the reference's inner `break` leaves only the DIMS scan, so where several new loops
/// carry the condition's dim the LAST one wins; that is reproduced.
pub fn split_loop_band_on_dim<S: LoopBands + ?Sized>(tree: &mut S, split: LoopBandSplit) -> LoopId {
    let LoopBandSplit { base, sets } = split;
    let Some((outermost, inner_sets)) = sets.split_first() else {
        return base;
    };
    if inner_sets.is_empty() {
        // One dim set is left and it is the base loop's own: no transformation is needed.
        return base;
    }

    let num = tree.loop_num(base);
    let den = tree.loop_den(base);
    let mut new_loops = vec![base];
    let mut first_new = None;
    let mut previous = None;
    for dims in inner_sets {
        let minted = tree.new_loop(construct_loop_node(num, den, dims.clone()));
        if let Some(previous) = previous {
            tree.add_child_node(minted.0, InsertionPoint::LastIn(previous));
        } else {
            first_new = Some(minted.0);
        }
        previous = Some(minted.0);
        new_loops.push(minted);
    }

    let innermost = new_loops.last().copied().unwrap_or(base);
    tree.move_children(base.0, innermost.0);
    if let Some(first_new) = first_new {
        tree.add_child_node(first_new, InsertionPoint::LastIn(base.0));
    }

    // The base loop keeps the outermost set, and its name is entry 114's naming plus `__split`.
    tree.set_loop_dims(base, outermost.clone());
    let renamed = construct_loop_node(num, den, outermost.clone()).name;
    tree.set_node_name(base.0, NodeName(format!("{}__split", renamed.0)));

    for condition in tree.conditions_under(innermost.0) {
        let mut cond = tree.loop_cond(condition);
        for clause in &mut cond.or_of_ands {
            for term in clause.iter_mut() {
                if term.loop_node != base.0 {
                    continue;
                }
                for candidate in &new_loops {
                    if tree.loop_dims(*candidate).iter().any(|d| d.dim == term.dim) {
                        term.loop_node = candidate.0;
                    }
                }
            }
        }
        tree.set_loop_cond(condition, cond);
    }

    innermost
}

/// A DATASTAGE SPLIT PROVED PERFORMABLE, CARRYING THE NUMERATOR STAGE IT WILL COPY.
///
/// ⛔⛔ THREE REFUSALS GONE: the external-loop abort, *"Loop splitting after datastage exploration is
/// not supported."* and `dataStageParam_.at(baseLoop->numId_)` throwing. ⚠️ THE REFERENCE MINTS THE
/// COPY *BEFORE* IT CHECKS THE EXPLORATION FLAG, so an aborting run has already grown the map; here
/// nothing is minted until the witness holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatastageSplitSite<D> {
    base: LoopId,
    reference: DataStage<D>,
}

impl<D: Clone> DatastageSplitSite<D> {
    /// The site, or [`None`] for any of the three refusals.
    #[must_use]
    pub fn of<S: ScheduleSurgery + ?Sized>(
        tree: &S,
        metadata: &Metadata,
        stages: &DataStages<D>,
        base: LoopId,
        exploration: DatastageExploration,
    ) -> Option<Self> {
        InternalNode::of(metadata, base.0)?;
        if exploration == DatastageExploration::Done {
            return None;
        }
        let reference = stages.0.get(&tree.loop_num(base))?.clone();
        Some(Self { base, reference })
    }
}

/// Replaces: e248_splitLoopBandOnDatastage
///
/// SPLITS ONE LOOP ON ITS DATASTAGE: a copy of the loop's numerator stage becomes a new stage, a
/// fresh loop over the SAME dims is nested perfectly inside the base loop with that new stage as its
/// numerator, the base loop's DENOMINATOR becomes it too, and every CONDITION now under the new loop
/// has its loop references adjusted for the split (`:287`). Yields the new inner loop.
///
/// ⚠️ TRAP: the base loop keeps its own numerator, so the pair `num/new` and `new/oldDen` compose to
/// the one band the single loop walked.
pub fn split_loop_band_on_datastage<S, D>(
    tree: &mut S,
    stages: &mut DataStages<D>,
    site: DatastageSplitSite<D>,
) -> LoopId
where
    S: LoopBands + ?Sized,
    D: Clone,
{
    let DatastageSplitSite { base, reference } = site;
    let stage = construct_datastage_from(stages, &reference);
    let dims = tree.loop_dims(base);
    let den = tree.loop_den(base);
    let minted = tree.new_loop(construct_loop_node(stage, den, dims));
    tree.insert_perfectly_nested(base, minted);
    tree.set_loop_den(base, stage);

    for condition in tree.conditions_under(minted.0) {
        tree.adjust_condition_for_split_loop(condition, base, &[base, minted]);
    }

    minted
}

/// WHICH BRANCH OF AN ENCLOSING CONDITION THE MOVED NODE SAT IN — `condNode->getThenBranchNode() ==
/// lastSeenNode` (`ddc/ddc_transformation_util.cpp:493`), which decides whether the reproduced
/// condition is combined or negated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionBranch {
    /// The `then` region — a core/corelet condition INTERSECTS, and a loop condition is copied.
    Then,
    /// The `else` region — a core/corelet condition is SUBTRACTED, and a loop condition's
    /// `negated_` flips.
    Else,
}

/// `coreClCond_` after one more `then`-branch condition intersects into it — `set_intersect` per
/// core, with a core that empties or is unnamed dropped.
fn intersect_core_cl(accumulated: &CoreClSet, condition: &CoreClSet) -> CoreClSet {
    CoreClSet(
        accumulated
            .0
            .iter()
            .filter_map(|(core, corelets)| {
                let other = condition.0.get(core)?;
                let both: BTreeSet<Corelet> = corelets.intersection(other).copied().collect();
                (!both.is_empty()).then_some((*core, both))
            })
            .collect(),
    )
}

/// `coreClCond_` after one more `else`-branch condition is subtracted from it — `set_diff` per core
/// the condition names, with a core that empties dropped.
///
/// ⛔ DELIBERATE DIVERGENCE, AND THE DEFECT IS THE REFERENCE'S: both of its accumulate steps
/// `erase` from the very map their range-`for` is walking (`:503`, `:544`), which invalidates the
/// iterator. Rebuilding states the intended result without the undefined behaviour.
fn subtract_core_cl(accumulated: &CoreClSet, condition: &CoreClSet) -> CoreClSet {
    let mut result = accumulated.clone();
    for (core, corelets) in &condition.0 {
        let Some(left) = result.0.get(core) else {
            continue;
        };
        let rest: BTreeSet<Corelet> = left.difference(corelets).copied().collect();
        if rest.is_empty() {
            result.0.remove(core);
        } else {
            result.0.insert(*core, rest);
        }
    }
    result
}

/// THE UNIVERSE MINUS ONE `else`-branch condition — every `coreIdsUsed_` core mapped to all
/// `numCoreletsUsed_DSC2_` corelets the condition did not name.
///
/// ⚠️ TRAP: a core the condition maps to EXACTLY the universe contributes NO entry at all rather
/// than an empty one, which is the reference's `if (clsIt->second != allCorelets)`.
fn negated_core_cl(condition: &CoreClSet, cores: &[Core], all: &BTreeSet<Corelet>) -> CoreClSet {
    let mut result = BTreeMap::new();
    for core in cores {
        match condition.0.get(core) {
            None => {
                result.insert(*core, all.clone());
            }
            Some(corelets) if corelets != all => {
                result.insert(*core, all.difference(corelets).copied().collect());
            }
            Some(_) => {}
        }
    }
    CoreClSet(result)
}

/// A TRANSFER MOVE PROVED LEGAL, CARRYING EVERYTHING THE REFERENCE VALIDATED BEFORE IT MUTATED.
///
/// ⛔⛔ SEVEN ABORTS COLLAPSE HERE: the external transfer, the FIFO destination, hoisting above the
/// source's producer loop, hoisting above a parametric loop, a loop conditional naming a chain loop,
/// an external allocate node, and moving allocations after datastage exploration.
/// ⛔⛔ AND THE SINKING PATH IS DEAD CODE, SO THIS ONLY EVER HOISTS. The reference rebuilds its chain
/// under `while (loop != transferOwnerLoop && loop != nullptr)`, whose guard already excludes the
/// terminator the following `if (loopChain.back() != transferOwnerLoop) DT_ERROR(..)` demands — so
/// that abort fires for EVERY destination that is not an ancestor, and the consumer-loop check, the
/// `start`/`end` swap and both `if (!insertBefore)` blocks below are unreachable. A destination that
/// is not an ancestor is [`None`] here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferMove {
    transfer: NodeId,
    new_parent: LoopId,
    /// `loopChain` — the transfer's owner loop first, `new_parent` last, always two or more.
    chain: Vec<LoopId>,
    /// `allocateNodesToMove`.
    allocations: Vec<NodeId>,
    /// `coreConditionToMove->coreClCond_`.
    core_condition: Option<CoreClSet>,
    /// `loopConditions`, in the walk's own innermost-to-outermost order.
    loop_conditions: Vec<(NodeId, ConditionBranch)>,
}

impl TransferMove {
    /// The move, or [`None`] for any of the seven refusals — and also where there is nothing to do,
    /// which is the reference's early `return` for a transfer already owned by `new_parent`.
    ///
    /// ⚠️ TRAP: the producer-loop AND parametric-loop checks are BOTH gated on the source's
    /// `dataConnect_` having a metadata entry, so an unset connect skips the parametric check too.
    #[must_use]
    pub fn of<S: TransferMoves + ?Sized>(
        tree: &S,
        metadata: &Metadata,
        transfer: NodeId,
        new_parent: LoopId,
        exploration: DatastageExploration,
    ) -> Option<Self> {
        InternalNode::of(metadata, transfer)?;
        // A transfer with no owner loop leaves `loopChain` empty and `loopChain.back()` undefined.
        let owner = tree.owner_loop(transfer)?;
        if owner == new_parent {
            return None;
        }

        let mut chain = Vec::new();
        let mut current = Some(owner);
        while let Some(loop_node) = current {
            chain.push(loop_node);
            if loop_node == new_parent {
                break;
            }
            current = tree.owner_loop(loop_node.0);
        }
        if chain.last() != Some(&new_parent) {
            return None;
        }

        let moved = tree.transfer(transfer);
        // `hasNonMemoryResult()` (`dsc/dsc2.cpp:4368`).
        if moved.dsts.iter().any(|dst| !is_memory(dst.storage)) {
            return None;
        }
        if let Some(producers) = moved
            .src
            .data
            .data_connect
            .and_then(|connect| tree.producer_loops(connect))
        {
            for loop_node in chain.iter().take(chain.len().saturating_sub(1)) {
                if producers.contains(loop_node) || tree.is_parametric(*loop_node) {
                    return None;
                }
            }
        }

        let mut core_condition: Option<CoreClSet> = None;
        let mut loop_conditions = Vec::new();
        let mut last_seen = transfer;
        let mut curr = transfer;
        while curr != new_parent.0 {
            if tree.is_condition(curr) {
                let branch = if tree.then_branch(curr) == Some(last_seen) {
                    ConditionBranch::Then
                } else {
                    ConditionBranch::Else
                };
                let cond = tree.loop_cond(curr);
                // `hasCoreClCond()` IS `twoLevelOrOfAnds_.empty()` (`dsc/dsc2.h:693-695`).
                if cond.or_of_ands.is_empty() {
                    let core_cl = tree.core_cl_cond(curr);
                    let combined = match (&core_condition, branch) {
                        (None, ConditionBranch::Then) => core_cl,
                        (None, ConditionBranch::Else) => {
                            negated_core_cl(&core_cl, &tree.core_ids_used(), &tree.corelets_used())
                        }
                        (Some(accumulated), ConditionBranch::Then) => {
                            intersect_core_cl(accumulated, &core_cl)
                        }
                        (Some(accumulated), ConditionBranch::Else) => {
                            subtract_core_cl(accumulated, &core_cl)
                        }
                    };
                    core_condition = Some(combined);
                } else {
                    for clause in &cond.or_of_ands {
                        for term in clause {
                            if chain
                                .iter()
                                .any(|l| *l != new_parent && l.0 == term.loop_node)
                            {
                                return None;
                            }
                        }
                    }
                    loop_conditions.push((curr, branch));
                }
            }
            last_seen = curr;
            curr = tree.parent(curr)?;
        }

        let mut allocations = Vec::new();
        for dst in moved.dsts.iter() {
            let Some(alloc) = tree.destination_allocation(dst) else {
                continue;
            };
            let alloc_owner = tree.owner_loop(alloc);
            if alloc_owner == Some(new_parent) {
                continue;
            }
            if chain
                .iter()
                .any(|loop_node| alloc_owner == Some(*loop_node))
            {
                InternalNode::of(metadata, alloc)?;
                allocations.push(alloc);
            }
        }
        if !allocations.is_empty() && exploration == DatastageExploration::Done {
            return None;
        }

        Some(Self {
            transfer,
            new_parent,
            chain,
            allocations,
            core_condition,
            loop_conditions,
        })
    }

    /// The loop the transfer is hoisted to.
    #[must_use]
    pub const fn new_parent(&self) -> LoopId {
        self.new_parent
    }
}

/// WHAT ENTRY 249 ADDITIONALLY ASKS OF THE TREE — the conditions it reproduces and the allocations
/// that travel with the transfer.
pub trait TransferMoves: ScheduleSurgery {
    /// `node->nodeType_ == CONDITION`.
    fn is_condition(&self, node: NodeId) -> bool;

    /// `condNode->getThenBranchNode()` (`dsc/dsc2.h:685`).
    fn then_branch(&self, condition: NodeId) -> Option<NodeId>;

    /// `condNode->coreClCond_` (`dsc/dsc2.h:683`).
    fn core_cl_cond(&self, condition: NodeId) -> CoreClSet;

    /// `0 .. currDsc->numCoreletsUsed_DSC2_` — the `allCorelets` universe.
    fn corelets_used(&self) -> BTreeSet<Corelet>;

    /// `currDsc->coreIdsUsed_`.
    fn core_ids_used(&self) -> Vec<Core>;

    /// `metadata.dataConnects_.at(connect).getProducerLoops()` (`ddc/ddc_metadata.h:159`), absent
    /// where that connect has NO metadata entry — which is the reference's `count()` gate.
    ///
    /// ⛔ ASKED THROUGH THE SEAM BECAUSE THE TWO NODE SPELLINGS ARE STILL TWO: [`Ends`] keys its
    /// producers by [`super::metadata::NodeIndex`] while a loop chain is [`LoopId`] over [`NodeId`],
    /// and unifying them is the refactor `metadata.rs` reserves for a batch that owns a
    /// schedule-tree type outright (`ddc/metadata.rs:532-534`).
    fn producer_loops(&self, connect: DataConnect) -> Option<Vec<LoopId>>;

    /// `currDsc->getMutableAllocation(dstLdsAndLoopOffsets_.at(i), dstVias_.at(i).loc_.storage_)` as
    /// a schedule node, absent where it is null.
    fn destination_allocation(&self, dst: &Operand) -> Option<NodeId>;

    /// `condNode->clone()` — a detached copy, with its regions still to be added.
    fn clone_condition(&mut self, condition: NodeId) -> NodeId;

    /// `new dsc2::ConditionNode()` with its `name_` and `coreClCond_`, and no loop condition.
    fn new_condition(&mut self, name: NodeName, core_cl: CoreClSet) -> NodeId;

    /// `condNode->addThenRegion(block)` (`dsc/dsc2.h:686`).
    fn add_then_region(&mut self, condition: NodeId, block: NodeId);

    /// `condNode->addElseRegion(block)` (`dsc/dsc2.h:687`).
    fn add_else_region(&mut self, condition: NodeId, block: NodeId);
}

/// Replaces: e249_moveTransferNode
///
/// HOISTS ONE TRANSFER TO AN ENCLOSING LOOP, immediately BEFORE the chain loop it leaves: the
/// destination allocations that lived in the loops it leaves move with it, a
/// `core_corelet_reuse_<transfer>` condition reproduces the core/corelet conditions it was under,
/// each enclosing loop condition is cloned as `<cond>_reuse_<transfer>` with fresh
/// `_then_region`/`_else_region` blocks and chained inside the previous one's `then`, and the
/// transfer lands at the FRONT of the innermost of those (`:335`).
///
/// ⚠️ TRAP: `insertBefore` is `true` on every reachable path, so the reference's `lastLoopCond` is
/// assigned and never read and both of its `if (!insertBefore)` blocks are dead.
pub fn move_transfer_node<S: TransferMoves + ?Sized>(tree: &mut S, move_: TransferMove) {
    let TransferMove {
        transfer,
        new_parent: _,
        chain,
        allocations,
        core_condition,
        loop_conditions,
    } = move_;
    let moved_name = tree.node_name(transfer).0;

    // `reverse(loopConditions)` — the chain is built outermost first.
    let mut first_loop_cond = None;
    let mut last_then: Option<NodeId> = None;
    for (condition, branch) in loop_conditions.iter().rev() {
        let name = format!("{}_reuse_{}", tree.node_name(*condition).0, moved_name);
        let cloned = tree.clone_condition(*condition);
        tree.set_node_name(cloned, NodeName(name.clone()));
        let then_branch = tree.new_block(NodeName(format!("{name}_then_region")));
        let else_branch = tree.new_block(NodeName(format!("{name}_else_region")));
        tree.add_then_region(cloned, then_branch);
        tree.add_else_region(cloned, else_branch);
        if *branch == ConditionBranch::Else {
            let mut cond = tree.loop_cond(cloned);
            cond.negated ^= true;
            tree.set_loop_cond(cloned, cond);
        }
        if first_loop_cond.is_none() {
            first_loop_cond = Some(cloned);
        }
        if let Some(last_then) = last_then {
            tree.add_child_node(cloned, InsertionPoint::LastIn(last_then));
        }
        last_then = Some(then_branch);
    }

    // `refNodeForInsertion = loopChain[loopChain.size() - 2]`, whose own parent is the effective new
    // parent — which is what naming the SIBLING already says.
    let Some(reference) = chain.iter().rev().nth(1).copied() else {
        return;
    };
    let mut at = InsertionPoint::Before(reference.0);

    for allocation in &allocations {
        tree.move_node(*allocation, at);
    }

    if let Some(core_cl) = core_condition {
        let name = format!("core_corelet_reuse_{moved_name}");
        let condition = tree.new_condition(NodeName(name.clone()), core_cl);
        tree.add_child_node(condition, at);
        let then_branch = tree.new_block(NodeName(format!("{name}_then_region")));
        let else_branch = tree.new_block(NodeName(format!("{name}_else_region")));
        tree.add_then_region(condition, then_branch);
        tree.add_else_region(condition, else_branch);
        at = InsertionPoint::FirstIn(then_branch);
    }

    if let Some(first_loop_cond) = first_loop_cond {
        tree.add_child_node(first_loop_cond, at);
        if let Some(last_then) = last_then {
            at = InsertionPoint::FirstIn(last_then);
        }
    }

    tree.move_node(transfer, at);
}

/// ONE CONSUMER OF A FIFO RESULT — the `dataConnects_[dc].consumers_` entries entry 250 rewrites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FifoConsumer {
    /// `TRANSFER` — its `src_.storage_` is repointed.
    Transfer(NodeId),
    /// `COMPUTE` — every `inputs_` entry whose `dataConnect_` matches is repointed.
    Compute(NodeId, ComputeNode),
    /// Every other `nodeType_`, which is *"Unsupported consumer type"*.
    Other(NodeId),
}

impl FifoConsumer {
    /// The consumer's node.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        match self {
            Self::Transfer(node) | Self::Compute(node, _) | Self::Other(node) => *node,
        }
    }
}

/// WHAT ENTRY 250 ADDITIONALLY ASKS — the FIFO result's consumers and the writes that repoint them.
pub trait FifoResults: ScheduleSurgery {
    /// `metadata.dataConnects_[connect].consumers_`. ⛔ An UNSET `dataConnect_` is the reference's
    /// `dataConnects_[""]`, which DEFAULT-CONSTRUCTS an empty entry — so it has no consumers.
    /// ⭐ Entry 339 reads the same map with `.at` (`:980`) and still cannot throw: entry 002 keys
    /// every transfer destination's connect (`ddc/ddcv1.cpp:3296-3297`).
    fn connect_consumers(&self, connect: Option<DataConnect>) -> Vec<FifoConsumer>;

    /// `computeNode->isOpaqueOp_` (`dsc/dsc2.h:531`).
    fn is_opaque(&self, compute: NodeId) -> bool;

    /// The transfer's ends as allocation lookups, for [`get_padding_per_dim`].
    fn transfer_ends(&self, transfer: NodeId) -> TransferEnds;

    /// The minted allocate node, placed and registered under `alloc`.
    fn insert_allocate(&mut self, alloc: AllocId, node: DdcAllocateNode, at: InsertionPoint);

    /// `transferNode->dstVias_[i].loc_.storage_ = storage`.
    fn set_dst_storage(&mut self, transfer: NodeId, dst: usize, storage: SenComponent);

    /// `transferConsumer->src_.storage_ = storage`.
    fn set_src_storage(&mut self, transfer: NodeId, storage: SenComponent);

    /// `computeConsumer->inputs_[i] = unit` — the UNIT vector (`dsc/dsc2.h:906`), not a storage.
    fn set_compute_input_unit(&mut self, compute: NodeId, input: usize, unit: SenComponent);

    /// `allocNode->addAllocUser(user)` (`dsc/dsc2.h:1012`) — the count-bumping one,
    /// [`DdcAllocateNode::add_alloc_user`].
    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId);
}

/// A TRANSFER WHOSE FIFO RESULT MAY BE CONVERTED AT ALL — entry 250's two aborts.
///
/// ⛔⛔ BOTH ARE THIS WITNESS MISSING: *"Conversion of FIFO result to register after datastage
/// exploration is not supported."* and the external-transfer refusal. Its `return false` paths are
/// NOT here — those are the function's own answer and stay in the port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FifoConversionSite {
    transfer: NodeId,
}

impl FifoConversionSite {
    /// The site, or [`None`] for either abort.
    #[must_use]
    pub fn of(
        metadata: &Metadata,
        transfer: NodeId,
        exploration: DatastageExploration,
    ) -> Option<Self> {
        if exploration == DatastageExploration::Done {
            return None;
        }
        InternalNode::of(metadata, transfer).map(|internal| Self {
            transfer: internal.node(),
        })
    }
}

/// Replaces: e250_convertResultFromFIFOtoReg
///
/// MOVES A TRANSFER'S ONE FIFO RESULT INTO A REGISTER FILE: mints an
/// `allocate_lds<i>_<mem>_<dc>_fifo_to_reg` allocation in the destination unit's own LRF right
/// before the transfer, repoints that destination's storage at it, and repoints every consumer of
/// the connect — a transfer's source storage, a compute's matching input UNITS — adding each as an
/// alloc user (`:760`). `true` also where there was NO FIFO result at all.
///
/// ⛔ THE `false` ANSWERS ARE THE REFERENCE'S: more than one FIFO result, a consumer that is opaque
/// or external, a destination unit that is neither `SFP` nor `PE`, and an allocation that already
/// exists. ⛔ DELIBERATE DIVERGENCE: an unset destination lds index and repeated layout dims are
/// aborts inside the reference's `constructAllocation`; here they answer `false` instead.
/// ⚠️ *"Unsupported consumer type"* becomes a skipped consumer, since a refusal is not available.
pub fn convert_result_from_fifo_to_reg<D>(
    dsc: &mut D,
    metadata: &mut Metadata,
    site: FifoConversionSite,
    alloc: AllocId,
) -> bool
where
    D: FifoResults + DscAllocations + AllocationPaddings + ?Sized,
{
    let transfer = site.transfer;
    let node = dsc.transfer(transfer);

    let mut fifo = None;
    for (index, dst) in node.dsts.iter().enumerate() {
        if is_memory(dst.storage) {
            continue;
        }
        if fifo.is_some() {
            return false;
        }
        fifo = Some((index, *dst));
    }
    let Some((index, dst)) = fifo else {
        return true;
    };
    let connect = dst.data.data_connect;

    for consumer in dsc.connect_consumers(connect) {
        if matches!(consumer, FifoConsumer::Compute(compute, _) if dsc.is_opaque(compute)) {
            return false;
        }
        if metadata.external_nodes.contains(&consumer.node()) {
            return false;
        }
    }

    let register_file = match dst.unit {
        SenComponent::Sfp => DdcMemory::SfpLrf,
        SenComponent::Pe => DdcMemory::PeLrf,
        _ => return false,
    };
    let Some(lds) = dst.data.my_lds_idx else {
        return false;
    };
    let Some(fresh) = FreshAllocation::of(dsc, metadata, lds, register_file) else {
        return false;
    };

    let ends = dsc.transfer_ends(transfer);
    let padding = get_padding_per_dim(dsc, metadata, transfer, &ends, TransferSide::Dst);
    let mut alloc_node = construct_allocation(dsc, metadata, fresh, padding, transfer, alloc);
    alloc_node.name = NodeName(format!(
        "{}_{}_fifo_to_reg",
        alloc_node.name.0,
        connect_spelling(connect)
    ));
    let component = memory_component(alloc_node.component);
    dsc.insert_allocate(alloc, alloc_node, InsertionPoint::Before(transfer));

    dsc.set_dst_storage(transfer, index, component);
    for consumer in dsc.connect_consumers(connect) {
        match consumer {
            FifoConsumer::Transfer(node) => {
                dsc.set_src_storage(node, component);
                dsc.add_alloc_user(alloc, node);
            }
            FifoConsumer::Compute(node, compute) => {
                for (input, operand) in compute.inputs.iter().enumerate() {
                    if operand.data.data_connect != connect {
                        continue;
                    }
                    dsc.set_compute_input_unit(node, input, component);
                    dsc.add_alloc_user(alloc, node);
                }
            }
            FifoConsumer::Other(_) => {}
        }
    }

    true
}

/// WHAT ENTRIES 251 AND 252 ADDITIONALLY ASK — the layout order the transfer walks.
pub trait TransferUnrolling: ScheduleSurgery + Dsc {
    /// `currDsc->getNonBroadcastLdsDims(lds)` (`dsc/dsc2.cpp:4039`) — the layout dims whose
    /// `scale_` is positive, as `l3::dsc::DesignSpaceConfig::non_broadcast_lds_dims` answers it,
    /// [`None`] for its own `getLayoutDims` abort.
    fn non_broadcast_lds_dims(&self, lds: LdsIdx) -> Option<Vec<PrimaryDim>>;
}

/// The transfer's `myLdsIdx_`: its source's, else the FIRST destination that has one.
fn transfer_lds_idx(node: &TransferNode) -> Option<LdsIdx> {
    node.src
        .data
        .my_lds_idx
        .or_else(|| node.dsts.iter().find_map(|dst| dst.data.my_lds_idx))
}

/// Mints one single-dim loop against `num`/`den`, inserts it immediately before the transfer and
/// moves the transfer inside it — the three lines both unrolling entries end each iteration with.
fn nest_transfer_in_new_loop<S: ScheduleSurgery + ?Sized>(
    tree: &mut S,
    transfer: NodeId,
    num: DatastageId,
    den: DatastageId,
    dim: PrimaryDimAndKind,
) {
    let minted = tree.new_loop(construct_loop_node(
        num,
        den,
        LoopDims::new(dim, Vec::new()),
    ));
    tree.add_child_node(minted.0, InsertionPoint::Before(transfer));
    tree.move_node(transfer, InsertionPoint::LastIn(minted.0));
}

/// Replaces: e251_unrollTransfer
///
/// FULLY UNROLLS ONE TRANSFER OVER ITS LAYOUT DIMS: one fresh single-dim loop per layout dim,
/// innermost dim outermost, each against a NEW minimizing denominator datastage and against the
/// denominator of the innermost enclosing loop that already carried that dim as its numerator, with
/// the transfer moved inside each in turn (`:1120`). `true` also where the transfer names no
/// labelled data structure at all, which needs no unrolling.
///
/// ⚠️ TRAP: the enclosing-loop walk does NOT stop once every dim has a numerator, so a parametric
/// loop ANYWHERE above the transfer still answers `false`.
/// ⛔ DELIBERATE DIVERGENCE: `loopNumPerDim.at(currDim)` throws for a layout dim no enclosing loop
/// carried; here that one dim is skipped and the rest still unroll.
pub fn unroll_transfer<S, D>(
    tree: &mut S,
    stages: &mut DataStages<D>,
    metadata: &mut Metadata,
    site: InternalNode,
) -> bool
where
    S: TransferUnrolling + ?Sized,
    D: Default,
{
    let transfer = site.node();
    let node = tree.transfer(transfer);
    let Some(lds) = transfer_lds_idx(&node) else {
        return true;
    };

    // `unordered_set<PrimaryDimAndKind>` over the layout dims — every one at kind `Unpadded`, which
    // is what `PrimaryDimAndKind`'s implicit conversion from a `PrimaryDimTypes` gives (`dims.h:79`).
    let transfer_dims: Vec<PrimaryDimAndKind> = tree
        .layout_dims(lds)
        .iter()
        .map(|dim| PrimaryDimAndKind {
            dim,
            kind: MetaDimKind::Unpadded,
        })
        .collect();
    let mut remaining: BTreeSet<PrimaryDimAndKind> = transfer_dims.iter().copied().collect();

    let mut num_per_dim: BTreeMap<PrimaryDim, DatastageId> = BTreeMap::new();
    let mut current = tree.owner_loop(transfer);
    while let Some(loop_node) = current {
        if tree.is_parametric(loop_node) {
            return false;
        }
        let num = tree.loop_den(loop_node);
        for dim in tree.loop_dims(loop_node).iter() {
            if remaining.remove(&dim) {
                num_per_dim.insert(dim.dim, num);
            }
        }
        current = tree.owner_loop(loop_node.0);
    }

    let den = construct_datastage(stages);
    metadata.datastages.entry(den).or_default().strategy = Strategy::Minimize;

    for dim in transfer_dims.iter().rev() {
        let Some(num) = num_per_dim.get(&dim.dim).copied() else {
            continue;
        };
        nest_transfer_in_new_loop(tree, transfer, num, den, *dim);
    }

    true
}

/// WHICH PER-DIM MAP OF A `DataStructDims` — the four entry 252 copies (`dsc/dims.h:170-200`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimSplit {
    /// `coreletSplit_`.
    Corelet,
    /// `rowSplit_`.
    Row,
    /// `peSfpSplit_`.
    PeSfp,
    /// `paddingSizes_`.
    Padding,
}

/// WHAT ENTRY 252 ASKS OF ONE HALF OF A DATA STAGE — [`DataStage`] is generic in its extents
/// payload, and this is the part of that payload entry 252 writes.
pub trait StageExtents {
    /// `primaryDimToValHandler_st(dim) = other.primaryDimToVal_st(dim)`.
    fn copy_dim_value_from(&mut self, other: &Self, dim: PrimaryDim);

    /// `<split>_.count(dim)`.
    fn states(&self, split: DimSplit, dim: PrimaryDim) -> bool;

    /// `<split>_[dim] = other.<split>_.at(dim)`.
    fn copy_split_from(&mut self, other: &Self, split: DimSplit, dim: PrimaryDim);

    /// `makeDimNotSymbolic(dim)` (`dsc/dims.cpp:717`).
    fn make_dim_not_symbolic(&mut self, dim: PrimaryDim);

    /// `<split>_.clear()` — the WHOLE map, not one dim, as entry 300 empties `peSfpSplit_` on the
    /// stage it copied (`ddc/ddc_transformation.cpp:1101-1102`).
    fn clear_split(&mut self, split: DimSplit);

    /// `<split>_.at(dim)` — that dim's per-side sizes, EMPTY where [`Self::states`] is false.
    ///
    /// ⛔ Entry 300 reads `.at(dim).at(0)` AND `.at(1)` of the corelet split
    /// (`ddc/ddc_transformation.cpp:920-922`), so a split with fewer than two sides is that second
    /// `.at()`'s throw and what to do about it is the caller's decision, not this reader's.
    fn split_sizes(&self, split: DimSplit, dim: PrimaryDim) -> Vec<Elements>;
}

/// Replaces: e252_unrollTransferForSymbolicDims
///
/// UNROLLS ONE TRANSFER OVER ITS SYMBOLIC LAYOUT DIMS ONLY, exactly as entry 251 does over all of
/// them, and first fills the new denominator datastage per dim from the numerator stage that dim
/// came from: its steady-state and epilogue extent, then its corelet/row/PE-SFP splits, then
/// `makeDimNotSymbolic`, and only THEN its padding sizes (`:1193`). `true` where there are no
/// symbolic dims or the transfer names no labelled data structure.
///
/// ⚠️ TRAP: each split is copied into BOTH halves gated on the STEADY-STATE half naming the dim, and
/// the padding copy deliberately runs AFTER the dim stopped being symbolic.
/// ⛔ Unlike entry 251 this walk STOPS once every dim has a numerator, which is observable: a
/// parametric loop above that point answers `true` here and `false` there.
pub fn unroll_transfer_for_symbolic_dims<S, D>(
    tree: &mut S,
    stages: &mut DataStages<D>,
    site: InternalNode,
    symbolic_dims: &BTreeMap<PrimaryDim, SymbolicDimInfo>,
) -> bool
where
    S: TransferUnrolling + ?Sized,
    D: Default + Clone + StageExtents,
{
    if symbolic_dims.is_empty() {
        return true;
    }
    let transfer = site.node();
    let node = tree.transfer(transfer);
    let Some(lds) = transfer_lds_idx(&node) else {
        return true;
    };

    // ⚠️ A SET OF DIMS AND NOT OF DIM-AND-KINDS, unlike entry 251's.
    // ⛔ DELIBERATE DIVERGENCE: `getLayoutDims`' own abort answers `false` here.
    let Some(non_broadcast) = tree.non_broadcast_lds_dims(lds) else {
        return false;
    };
    let transfer_dims: Vec<PrimaryDim> = non_broadcast
        .into_iter()
        .filter(|dim| symbolic_dims.contains_key(dim))
        .collect();
    let mut remaining: BTreeSet<PrimaryDim> = transfer_dims.iter().copied().collect();

    let mut num_per_dim: BTreeMap<PrimaryDim, DatastageId> = BTreeMap::new();
    let mut current = tree.owner_loop(transfer);
    while let Some(loop_node) = current {
        if remaining.is_empty() {
            break;
        }
        if tree.is_parametric(loop_node) {
            return false;
        }
        let num = tree.loop_den(loop_node);
        for dim in tree.loop_dims(loop_node).iter() {
            if remaining.remove(&dim.dim) {
                num_per_dim.insert(dim.dim, num);
            }
        }
        current = tree.owner_loop(loop_node.0);
    }

    let den = construct_datastage(stages);
    for (dim, reference) in &num_per_dim {
        // The reference stage is read out first: `dataStageParam_.at(dsIdx)` and `.at(denId)` are one
        // map, and a missing either is that `.at()` throwing — here the dim is skipped instead.
        let Some(from) = stages.0.get(reference).cloned() else {
            continue;
        };
        let Some(to) = stages.0.get_mut(&den) else {
            continue;
        };
        to.ss.dims.copy_dim_value_from(&from.ss.dims, *dim);
        to.el.dims.copy_dim_value_from(&from.el.dims, *dim);
        for split in [DimSplit::Corelet, DimSplit::Row, DimSplit::PeSfp] {
            if !from.ss.dims.states(split, *dim) {
                continue;
            }
            to.ss.dims.copy_split_from(&from.ss.dims, split, *dim);
            to.el.dims.copy_split_from(&from.el.dims, split, *dim);
        }
        to.ss.dims.make_dim_not_symbolic(*dim);
        to.el.dims.make_dim_not_symbolic(*dim);
        if from.ss.dims.states(DimSplit::Padding, *dim) {
            to.ss
                .dims
                .copy_split_from(&from.ss.dims, DimSplit::Padding, *dim);
            to.el
                .dims
                .copy_split_from(&from.el.dims, DimSplit::Padding, *dim);
        }
    }

    for dim in transfer_dims.iter().rev() {
        let Some(num) = num_per_dim.get(dim).copied() else {
            continue;
        };
        let entry = PrimaryDimAndKind {
            dim: *dim,
            kind: MetaDimKind::Unpadded,
        };
        nest_transfer_in_new_loop(tree, transfer, num, den, entry);
    }

    true
}

/// Replaces: e253_srcRelatedToExternalNodes
///
/// Whether the transfer's SOURCE datastream is external, asked of the connect's `producers_`
/// (`:1687`).
///
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: the operand's `storage_` is what reaches entry 120's
/// `storage` here — the OPPOSITE of entries 255/256, which pass the operand's `unit_`.
#[must_use]
pub fn src_related_to_external_nodes<E: ExternalStreams + ?Sized>(
    streams: &E,
    transfer: &TransferNode,
) -> bool {
    streams.storage_or_datastream_is_external(
        transfer.src.data,
        transfer.src.storage,
        StreamDirection::Incoming,
    )
}

/// Replaces: e254_destRelatedToExternalNodes
///
/// Whether ONE destination datastream of a transfer is external, asked of the connect's `consumers_`
/// (`:1693`).
///
/// ⛔ Taking the destination ITSELF closes both `.at(dstIndex)` throws, and it is what keeps entries
/// 305/339/340/341 constructible. ⛔ Same storage trap as entry 253.
#[must_use]
pub fn dest_related_to_external_nodes<E: ExternalStreams + ?Sized>(
    streams: &E,
    dst: &Operand,
) -> bool {
    streams.storage_or_datastream_is_external(dst.data, dst.storage, StreamDirection::Outgoing)
}

#[cfg(test)]
mod tests_e110_e117 {
    // ⭐ TESTS FOR ENTRIES 110-117. Union this module with this file's other test modules when they
    // land.
    use super::*;

    use sys_arch_spec::arch_enums::SenComponent;

    use super::super::metadata::{Allocation, DataTransfer, TransferAccessPattern};
    use crate::schedule::ddc::fold::{ConstIdx, DataStream};
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        DataInfo, Dsts, InstrAttribute, LayoutDims, NumChunks, Operand, ReplicationFactor,
        TransferPadding,
    };
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
        fn own_lds_idx(&self, _lds: LdsIdx) -> LdsIdx {
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
            storage: SenComponent::NoComponent,
            data: DataInfo {
                data_connect: connect,
                my_lds_idx: None,
                constant_id: None,
                latch_data_id: None,
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
        let term = |loop_node| LoopCond {
            loop_node,
            dim: PrimaryDim::Mb,
            op: CondOp::Eq,
            against: CondValType::Last,
        };
        let cond = LoopCondComposite {
            or_of_ands: vec![
                vec![term(NodeId(1)), term(NodeId(2))],
                vec![term(NodeId(1))],
            ],
            negated: false,
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
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            name: NodeName("t0".to_string()),
            src: operand(Some(DataConnect::ArfPt)),
            dsts: Dsts::new(operand(Some(DataConnect::ArfPtsum)), vec![operand(None)]),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
        };
        assert_eq!(
            get_node_description(UtilNode::Transfer(&transfer)),
            "t0 [ src: arf_pt dst:  arf_ptsum  ]"
        );

        let compute = ComputeNode {
            name: NodeName("c0".to_string()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Pe,
            inputs: vec![operand(Some(DataConnect::ArfPt))],
            outputs: vec![operand(Some(DataConnect::ArfPtsum))],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        };
        assert_eq!(
            get_node_description(UtilNode::Compute(&compute)),
            "c0 [ input:  arf_pt output:  arf_ptsum]"
        );

        let name = NodeName("block0".to_string());
        assert_eq!(get_node_description(UtilNode::Other(&name)), "block0");
    }
}

#[cfg(test)]
mod tests_e255_e257 {
    // ⭐ TESTS FOR ENTRIES 255-257. Union this module with this file's other test modules when they
    // land.
    use super::*;

    use sys_arch_spec::arch_enums::SenComponent;

    use super::super::metadata::{Allocation, OpaqueOp};
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{DataInfo, InstrAttribute, Operand, OperandPos};
    use crate::units::NumFolds;

    /// THE ONE DATASTREAM THIS STAND-IN CALLS EXTERNAL — entry 120's answer, as the seam the port
    /// asks through. Its key is the triple entry 255/256 hands over, so a port that passed
    /// `Operand::storage` instead of `Operand::unit` cannot match it.
    struct External(Option<(Option<DataConnect>, SenComponent, StreamDirection)>);

    impl ExternalStreams for External {
        fn storage_or_datastream_is_external(
            &self,
            data: DataInfo,
            storage: SenComponent,
            direction: StreamDirection,
        ) -> bool {
            self.0 == Some((data.data_connect, storage, direction))
        }
    }

    /// An operand whose `unit_` and its `storage_` DIFFER, which is what puts the trap under test.
    fn operand(connect: DataConnect, unit: SenComponent) -> Operand {
        Operand {
            unit,
            storage: SenComponent::Hbm,
            data: DataInfo {
                data_connect: Some(connect),
                my_lds_idx: Some(LdsIdx(1)),
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    fn compute() -> ComputeNode {
        ComputeNode {
            name: NodeName("c0".to_string()),
            op: DdlComputeType::Macc,
            ex_unit: SenComponent::Pe,
            inputs: vec![
                operand(DataConnect::ArfPt, SenComponent::Lx),
                operand(DataConnect::ArfPtsum, SenComponent::L0),
            ],
            outputs: vec![operand(DataConnect::OuttensorToSfp, SenComponent::Ptxrf)],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        }
    }

    #[test]
    fn a_compute_input_is_related_to_external_nodes_through_its_own_component_as_the_storage() {
        // The SECOND input's connect, its `inputs_` component, asked as an INCOMING stream.
        let external = External(Some((
            Some(DataConnect::ArfPtsum),
            SenComponent::L0,
            StreamDirection::Incoming,
        )));
        assert!(input_related_to_external_nodes(&external, &compute()));

        // ⛔ THE TRAP: the same stream asked about the operand's `storage` answers NO, so a port that
        // passed `Operand::storage` would report every compute unrelated.
        let by_storage = External(Some((
            Some(DataConnect::ArfPtsum),
            SenComponent::Hbm,
            StreamDirection::Incoming,
        )));
        assert!(!input_related_to_external_nodes(&by_storage, &compute()));

        // And an OUTGOING answer never reaches the input side.
        let outgoing = External(Some((
            Some(DataConnect::ArfPtsum),
            SenComponent::L0,
            StreamDirection::Outgoing,
        )));
        assert!(!input_related_to_external_nodes(&outgoing, &compute()));
    }

    #[test]
    fn a_compute_output_is_related_to_external_nodes_only_as_an_outgoing_stream() {
        let external = External(Some((
            Some(DataConnect::OuttensorToSfp),
            SenComponent::Ptxrf,
            StreamDirection::Outgoing,
        )));
        assert!(output_related_to_external_nodes(&external, &compute()));

        // The output's own stream asked INCOMING is the input side's question, and the outputs are
        // not asked it.
        let incoming = External(Some((
            Some(DataConnect::OuttensorToSfp),
            SenComponent::Ptxrf,
            StreamDirection::Incoming,
        )));
        assert!(!output_related_to_external_nodes(&incoming, &compute()));
        // An input's stream is not an output's.
        let on_an_input = External(Some((
            Some(DataConnect::ArfPt),
            SenComponent::Lx,
            StreamDirection::Outgoing,
        )));
        assert!(!output_related_to_external_nodes(&on_an_input, &compute()));
    }

    /// ONE `labeledDs_` ENTRY — its name so a shifted list can be read back, its two indices, and the
    /// allocations its `memOrg_` names.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestLds {
        name: &'static str,
        recorded: LdsIdx,
        reference: Option<LdsIdx>,
        mem_org: Vec<AllocId>,
    }

    impl LabeledDsEntry for TestLds {
        fn recorded_lds_idx(&self) -> LdsIdx {
            self.recorded
        }

        fn set_recorded_lds_idx(&mut self, recorded: LdsIdx) {
            self.recorded = recorded;
        }

        fn set_reference_lds_idx(&mut self, reference: LdsIdx) {
            self.reference = Some(reference);
        }
    }

    /// A DSC WHOSE `labeledDs_` REFERENCES ARE HELD BY POSITION — `held` stands for
    /// `computeOp_.inputLabeledDs` and `dataTransfers_.myDsInfo`, so the shift the insert owes them is
    /// observable.
    #[derive(Debug)]
    struct TestDsc {
        entries: Vec<TestLds>,
        held: Vec<LdsIdx>,
        allocs: BTreeMap<AllocId, Option<LdsIdx>>,
        slots: Vec<(TreeLdsSlot, Option<LdsIdx>)>,
        opaque: Vec<NodeId>,
    }

    impl NewLabeledDs for TestDsc {
        type Entry = TestLds;

        fn last_lds_pos(&self) -> LdsIdx {
            LdsIdx(self.entries.len() as u32 - 1)
        }

        fn last_recorded_lds_idx(&self) -> LdsIdx {
            self.entries[self.entries.len() - 1].recorded
        }

        fn set_last_recorded_lds_idx(&mut self, recorded: LdsIdx) {
            let last = self.entries.len() - 1;
            self.entries[last].recorded = recorded;
        }

        fn insert_lds_before_last(&mut self, entry: Self::Entry) {
            let was_last = LdsIdx(self.entries.len() as u32 - 1);
            self.entries.insert(was_last.0 as usize, entry);
            // The entry that was last moved up one place; every other position kept its entry.
            for held in &mut self.held {
                if *held == was_last {
                    *held = LdsIdx(was_last.0 + 1);
                }
            }
        }

        fn clear_mem_org(&mut self, pos: LdsIdx) {
            if let Some(entry) = self.entries.get_mut(pos.0 as usize) {
                entry.mem_org.clear();
            }
        }

        fn mem_org_allocations(&self, pos: LdsIdx) -> Vec<AllocId> {
            self.entries
                .get(pos.0 as usize)
                .map(|entry| entry.mem_org.clone())
                .unwrap_or_default()
        }

        fn alloc_lds_idx(&self, alloc: AllocId) -> Option<LdsIdx> {
            self.allocs.get(&alloc).copied().flatten()
        }

        fn set_alloc_lds_idx(&mut self, alloc: AllocId, lds: LdsIdx) {
            self.allocs.insert(alloc, Some(lds));
        }

        fn tree_lds_slots(&self) -> Vec<(TreeLdsSlot, Option<LdsIdx>)> {
            self.slots.clone()
        }

        fn set_tree_lds(&mut self, slot: TreeLdsSlot, lds: LdsIdx) {
            for held in &mut self.slots {
                if held.0 == slot {
                    held.1 = Some(lds);
                }
            }
        }

        fn opaque_computes(&self) -> Vec<NodeId> {
            self.opaque.clone()
        }
    }

    fn lds(name: &'static str, recorded: u32, mem_org: Vec<AllocId>) -> TestLds {
        TestLds {
            name,
            recorded: LdsIdx(recorded),
            reference: None,
            mem_org,
        }
    }

    /// Three entries whose recorded indices ARE their positions, the output last — the shape every
    /// callsite hands `addNewLds`.
    fn dsc_of_three() -> TestDsc {
        TestDsc {
            entries: vec![
                lds("in", 0, vec![AllocId(10)]),
                lds("weight", 1, vec![AllocId(11)]),
                lds("out", 2, vec![AllocId(12)]),
            ],
            held: vec![LdsIdx(0), LdsIdx(2)],
            allocs: BTreeMap::from([
                (AllocId(10), Some(LdsIdx(0))),
                (AllocId(11), Some(LdsIdx(1))),
                (AllocId(12), Some(LdsIdx(2))),
                (AllocId(13), Some(LdsIdx(2))),
            ]),
            slots: vec![
                (TreeLdsSlot::Allocate(NodeId(1)), Some(LdsIdx(2))),
                (
                    TreeLdsSlot::Compute(NodeId(2), OperandPos::Input(0)),
                    Some(LdsIdx(1)),
                ),
                (
                    TreeLdsSlot::Compute(NodeId(2), OperandPos::Output(0)),
                    Some(LdsIdx(2)),
                ),
                (TreeLdsSlot::Transfer(NodeId(3), OperandPos::Input(0)), None),
                (
                    TreeLdsSlot::Transfer(NodeId(3), OperandPos::Output(1)),
                    Some(LdsIdx(2)),
                ),
                (TreeLdsSlot::ParametricLoop(NodeId(4)), Some(LdsIdx(2))),
            ],
            opaque: vec![NodeId(2), NodeId(9)],
        }
    }

    #[test]
    fn a_new_lds_takes_the_last_index_and_every_holder_of_it_is_renamed_to_the_shifted_one() {
        let mut dsc = dsc_of_three();
        let mut metadata = Metadata {
            opaque_ops: BTreeMap::from([
                (
                    NodeId(2),
                    OpaqueOp {
                        lds_idx: Some(LdsIdx(2)),
                        ..OpaqueOp::default()
                    },
                ),
                // A key the walk does not reach keeps its index.
                (
                    NodeId(7),
                    OpaqueOp {
                        lds_idx: Some(LdsIdx(2)),
                        ..OpaqueOp::default()
                    },
                ),
            ]),
            new_allocations: BTreeMap::from([(
                DdcMemory::Lx,
                Allocation {
                    lds_idx_and_alloc_node: BTreeMap::from([
                        (LdsIdx(1), AllocId(11)),
                        (LdsIdx(2), AllocId(12)),
                    ]),
                    cons_id_and_alloc_node: BTreeMap::new(),
                    comp_and_alloc_node: BTreeMap::from([(NodeId(2), AllocId(13))]),
                },
            )]),
            // ⛔ NO KEY for the old last index, so the ELSE branch remaps the VALUE naming it.
            lds_idx_after_ddc: BTreeMap::from([(LdsIdx(0), LdsIdx(2)), (LdsIdx(1), LdsIdx(1))]),
            ..Metadata::default()
        };

        let source = dsc.entries[0].clone();
        assert_eq!(add_new_lds(&mut dsc, &mut metadata, &source), LdsIdx(2));

        // The copy sits at position 2, carries the index the output had, and names its source; the
        // output moved to position 3 with its index bumped.
        assert_eq!(
            dsc.entries
                .iter()
                .map(|entry| (entry.name, entry.recorded, entry.reference))
                .collect::<Vec<_>>(),
            vec![
                ("in", LdsIdx(0), None),
                ("weight", LdsIdx(1), None),
                ("in", LdsIdx(2), Some(LdsIdx(0))),
                ("out", LdsIdx(3), None),
            ]
        );
        // The copy's `memOrg_` was cleared, and the shifted output's allocations now name index 3.
        assert!(dsc.entries[2].mem_org.is_empty());
        assert_eq!(dsc.entries[3].mem_org, vec![AllocId(12)]);
        assert_eq!(dsc.allocs[&AllocId(12)], Some(LdsIdx(3)));
        // `compAndAllocNode`'s allocation is renamed too, while an untouched index stands.
        assert_eq!(dsc.allocs[&AllocId(13)], Some(LdsIdx(3)));
        assert_eq!(dsc.allocs[&AllocId(11)], Some(LdsIdx(1)));

        // References held BY POSITION follow the entry that moved.
        assert_eq!(dsc.held, vec![LdsIdx(0), LdsIdx(3)]);

        // Every tree slot that named the old last index now names the new one; a `-1` stays absent.
        assert_eq!(
            dsc.slots.iter().map(|slot| slot.1).collect::<Vec<_>>(),
            vec![
                Some(LdsIdx(3)),
                Some(LdsIdx(1)),
                Some(LdsIdx(3)),
                None,
                Some(LdsIdx(3)),
                Some(LdsIdx(3)),
            ]
        );

        // The opaque op the walk reached is renamed; the one it did not is left alone.
        assert_eq!(metadata.opaque_ops[&NodeId(2)].lds_idx, Some(LdsIdx(3)));
        assert_eq!(metadata.opaque_ops[&NodeId(7)].lds_idx, Some(LdsIdx(2)));

        // `newAllocations_` is re-keyed off the old index onto the new one.
        assert_eq!(
            metadata.new_allocations[&DdcMemory::Lx].lds_idx_and_alloc_node,
            BTreeMap::from([(LdsIdx(1), AllocId(11)), (LdsIdx(3), AllocId(12))])
        );

        // The DDL interface's VALUE naming the old index is remapped.
        assert_eq!(
            metadata.lds_idx_after_ddc,
            BTreeMap::from([(LdsIdx(0), LdsIdx(3)), (LdsIdx(1), LdsIdx(1))])
        );
    }

    #[test]
    fn a_ddl_interface_entry_keyed_by_the_old_index_takes_the_new_one_as_its_value() {
        let mut dsc = dsc_of_three();
        let mut metadata = Metadata {
            // ⛔ THE KEY IS PRESENT, so ONLY its value moves and no other entry is remapped — even
            // one whose value names the old index.
            lds_idx_after_ddc: BTreeMap::from([(LdsIdx(2), LdsIdx(9)), (LdsIdx(0), LdsIdx(2))]),
            ..Metadata::default()
        };

        let source = dsc.entries[0].clone();
        assert_eq!(add_new_lds(&mut dsc, &mut metadata, &source), LdsIdx(2));
        assert_eq!(
            metadata.lds_idx_after_ddc,
            BTreeMap::from([(LdsIdx(2), LdsIdx(3)), (LdsIdx(0), LdsIdx(2))])
        );
    }
}

#[cfg(test)]
mod tests_e247_e254 {
    // ⭐ TESTS FOR ENTRIES 247-254, AND FOR ENTRIES 339-341, whose destination-repointing entries
    // read the same tree, the same allocations and the same external stand-in. Union this module
    // with this file's other test modules when they land.
    use super::*;

    use core::num::NonZeroU32;

    use crate::schedule::ddc::fold::DataStream;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        Dsts, InstrAttribute, LayoutDims, NumChunks, ReplicationFactor, TransferPadding,
    };
    use crate::schedule::l3::dsc::{Granularity, MaxSize};
    use crate::units::{DfirUnit, NumFolds};

    /// The kinds of node the ported entries build a nest out of.
    #[derive(Debug, Clone)]
    enum Kind {
        Loop { node: LoopNode, parametric: bool },
        Transfer(TransferNode),
        Condition(LoopCondComposite, CoreClSet),
        Block,
        Allocate,
        /// Entry 340 places one of these; nothing reads its `ComputeNode`, which the consumer list
        /// carries instead.
        Compute,
    }

    #[derive(Debug, Clone)]
    struct Entry {
        name: NodeName,
        parent: Option<NodeId>,
        children: Vec<NodeId>,
        kind: Kind,
        then_region: Option<NodeId>,
        else_region: Option<NodeId>,
    }

    /// A SCHEDULE TREE THESE ENTRIES REWRITE, plus the DSC facts entries 250-252 read of one.
    #[derive(Debug, Default)]
    struct Tree {
        nodes: BTreeMap<NodeId, Entry>,
        next: u32,
        layout: BTreeMap<LdsIdx, LayoutDims>,
        non_broadcast: BTreeMap<LdsIdx, Vec<PrimaryDim>>,
        producers: BTreeMap<DataConnect, Vec<LoopId>>,
        consumers: Vec<FifoConsumer>,
        opaque: BTreeSet<NodeId>,
        dst_allocation: BTreeMap<LdsIdx, NodeId>,
        ends: Option<TransferEnds>,
        core_ids: Vec<Core>,
        corelets: BTreeSet<Corelet>,
        allocated: Vec<(AllocId, DdcAllocateNode, InsertionPoint)>,
        dst_storage: Vec<(NodeId, usize, SenComponent)>,
        src_storage: Vec<(NodeId, SenComponent)>,
        compute_inputs: Vec<(NodeId, usize, SenComponent)>,
        alloc_users: Vec<(AllocId, NodeId)>,
        adjusted: Vec<(NodeId, LoopId, Vec<LoopId>)>,
        nested: Vec<(LoopId, LoopId)>,
        allocations: BTreeMap<(DataOrigin, DdcMemory), AllocId>,
        users: BTreeMap<AllocId, Vec<NodeId>>,
        external: Option<(Option<DataConnect>, SenComponent, StreamDirection)>,
        reduced: Vec<(AllocId, NodeId)>,
        removed: Vec<(AllocId, NodeId)>,
        dst_latch: Vec<(NodeId, usize, LatchDataId)>,
        src_latch: Vec<(NodeId, LatchDataId)>,
        input_latch: Vec<(NodeId, usize, LatchDataId)>,
        compute_outputs: Vec<(NodeId, SenComponent, DataInfo)>,
        compute_input_writes: Vec<(NodeId, InputIdx, SenComponent, DataInfo)>,
    }

    impl Tree {
        fn add(&mut self, name: &str, kind: Kind, parent: Option<NodeId>) -> NodeId {
            let id = NodeId(self.next);
            self.next += 1;
            self.nodes.insert(
                id,
                Entry {
                    name: NodeName(name.to_string()),
                    parent,
                    children: Vec::new(),
                    kind,
                    then_region: None,
                    else_region: None,
                },
            );
            if let Some(parent) = parent {
                self.nodes
                    .get_mut(&parent)
                    .expect("parent exists")
                    .children
                    .push(id);
            }
            id
        }

        fn loop_node(
            &mut self,
            num: u32,
            den: u32,
            dims: LoopDims,
            parent: Option<NodeId>,
        ) -> LoopId {
            let node = construct_loop_node(DatastageId(num), DatastageId(den), dims);
            let name = node.name.0.clone();
            LoopId(self.add(
                &name,
                Kind::Loop {
                    node,
                    parametric: false,
                },
                parent,
            ))
        }

        fn unlink(&mut self, node: NodeId) {
            let parent = self
                .nodes
                .get_mut(&node)
                .expect("node exists")
                .parent
                .take();
            if let Some(parent) = parent {
                self.nodes
                    .get_mut(&parent)
                    .expect("parent exists")
                    .children
                    .retain(|child| *child != node);
            }
        }

        fn link(&mut self, node: NodeId, at: InsertionPoint) {
            let (parent, index) = match at {
                InsertionPoint::Before(sibling) | InsertionPoint::After(sibling) => {
                    let parent = self.nodes[&sibling].parent.expect("sibling has a parent");
                    let position = self.nodes[&parent]
                        .children
                        .iter()
                        .position(|child| *child == sibling)
                        .expect("sibling among its parent's children");
                    let after = matches!(at, InsertionPoint::After(_));
                    (parent, position + usize::from(after))
                }
                InsertionPoint::FirstIn(parent) => (parent, 0),
                InsertionPoint::LastIn(parent) => (parent, self.nodes[&parent].children.len()),
            };
            self.nodes
                .get_mut(&parent)
                .expect("parent exists")
                .children
                .insert(index, node);
            self.nodes.get_mut(&node).expect("node exists").parent = Some(parent);
        }

        fn children(&self, node: NodeId) -> Vec<NodeId> {
            self.nodes[&node].children.clone()
        }

        fn minted_loop(&self, loop_node: LoopId) -> &LoopNode {
            match &self.nodes[&loop_node.0].kind {
                Kind::Loop { node, .. } => node,
                other => panic!("not a loop: {other:?}"),
            }
        }
    }

    impl ScheduleSurgery for Tree {
        fn node_name(&self, node: NodeId) -> NodeName {
            self.nodes[&node].name.clone()
        }

        fn set_node_name(&mut self, node: NodeId, name: NodeName) {
            self.nodes.get_mut(&node).expect("node exists").name = name;
        }

        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.nodes[&node].parent
        }

        fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
            let mut current = self.nodes[&node].parent;
            while let Some(candidate) = current {
                if matches!(self.nodes[&candidate].kind, Kind::Loop { .. }) {
                    return Some(LoopId(candidate));
                }
                current = self.nodes[&candidate].parent;
            }
            None
        }

        fn transfer(&self, node: NodeId) -> TransferNode {
            match &self.nodes[&node].kind {
                Kind::Transfer(transfer) => transfer.clone(),
                other => panic!("not a transfer: {other:?}"),
            }
        }

        fn loop_num(&self, loop_node: LoopId) -> DatastageId {
            self.minted_loop(loop_node).num
        }

        fn loop_den(&self, loop_node: LoopId) -> DatastageId {
            self.minted_loop(loop_node).den
        }

        fn loop_dims(&self, loop_node: LoopId) -> LoopDims {
            self.minted_loop(loop_node).dims.clone()
        }

        fn is_parametric(&self, loop_node: LoopId) -> bool {
            matches!(
                self.nodes[&loop_node.0].kind,
                Kind::Loop {
                    parametric: true,
                    ..
                }
            )
        }

        fn new_loop(&mut self, node: LoopNode) -> LoopId {
            let name = node.name.0.clone();
            LoopId(self.add(
                &name,
                Kind::Loop {
                    node,
                    parametric: false,
                },
                None,
            ))
        }

        fn new_block(&mut self, name: NodeName) -> NodeId {
            self.add(&name.0, Kind::Block, None)
        }

        fn add_child_node(&mut self, node: NodeId, at: InsertionPoint) {
            self.link(node, at);
        }

        fn move_node(&mut self, node: NodeId, at: InsertionPoint) {
            self.unlink(node);
            self.link(node, at);
        }

        fn conditions_under(&self, root: NodeId) -> Vec<NodeId> {
            let mut found = Vec::new();
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                if matches!(self.nodes[&node].kind, Kind::Condition(..)) {
                    found.push(node);
                }
                stack.extend(self.children(node));
            }
            found
        }

        fn loop_cond(&self, condition: NodeId) -> LoopCondComposite {
            match &self.nodes[&condition].kind {
                Kind::Condition(cond, _) => cond.clone(),
                _ => LoopCondComposite::default(),
            }
        }

        fn set_loop_cond(&mut self, condition: NodeId, new: LoopCondComposite) {
            if let Kind::Condition(cond, _) =
                &mut self.nodes.get_mut(&condition).expect("node exists").kind
            {
                *cond = new;
            }
        }
    }

    impl LoopBands for Tree {
        fn set_loop_den(&mut self, loop_node: LoopId, den: DatastageId) {
            if let Kind::Loop { node, .. } =
                &mut self.nodes.get_mut(&loop_node.0).expect("loop exists").kind
            {
                node.den = den;
            }
        }

        fn set_loop_dims(&mut self, loop_node: LoopId, dims: LoopDims) {
            if let Kind::Loop { node, .. } =
                &mut self.nodes.get_mut(&loop_node.0).expect("loop exists").kind
            {
                node.dims = dims;
            }
        }

        fn move_children(&mut self, from: NodeId, to: NodeId) {
            let children =
                core::mem::take(&mut self.nodes.get_mut(&from).expect("node exists").children);
            for child in children {
                self.nodes.get_mut(&child).expect("child exists").parent = Some(to);
                self.nodes
                    .get_mut(&to)
                    .expect("node exists")
                    .children
                    .push(child);
            }
        }

        fn insert_perfectly_nested(&mut self, base: LoopId, nested: LoopId) {
            self.move_children(base.0, nested.0);
            self.link(nested.0, InsertionPoint::LastIn(base.0));
            self.nested.push((base, nested));
        }

        fn adjust_condition_for_split_loop(
            &mut self,
            condition: NodeId,
            orig: LoopId,
            new_loops: &[LoopId],
        ) {
            self.adjusted.push((condition, orig, new_loops.to_vec()));
        }
    }

    impl TransferMoves for Tree {
        fn is_condition(&self, node: NodeId) -> bool {
            matches!(self.nodes[&node].kind, Kind::Condition(..))
        }

        fn then_branch(&self, condition: NodeId) -> Option<NodeId> {
            self.nodes[&condition].then_region
        }

        fn core_cl_cond(&self, condition: NodeId) -> CoreClSet {
            match &self.nodes[&condition].kind {
                Kind::Condition(_, core_cl) => core_cl.clone(),
                _ => CoreClSet(BTreeMap::new()),
            }
        }

        fn corelets_used(&self) -> BTreeSet<Corelet> {
            self.corelets.clone()
        }

        fn core_ids_used(&self) -> Vec<Core> {
            self.core_ids.clone()
        }

        fn producer_loops(&self, connect: DataConnect) -> Option<Vec<LoopId>> {
            self.producers.get(&connect).cloned()
        }

        fn destination_allocation(&self, dst: &Operand) -> Option<NodeId> {
            dst.data
                .my_lds_idx
                .and_then(|lds| self.dst_allocation.get(&lds).copied())
        }

        fn clone_condition(&mut self, condition: NodeId) -> NodeId {
            let entry = self.nodes[&condition].clone();
            let name = entry.name.0.clone();
            self.add(&name, entry.kind, None)
        }

        fn new_condition(&mut self, name: NodeName, core_cl: CoreClSet) -> NodeId {
            self.add(
                &name.0,
                Kind::Condition(LoopCondComposite::default(), core_cl),
                None,
            )
        }

        fn add_then_region(&mut self, condition: NodeId, block: NodeId) {
            self.nodes
                .get_mut(&condition)
                .expect("node exists")
                .then_region = Some(block);
            self.link(block, InsertionPoint::LastIn(condition));
        }

        fn add_else_region(&mut self, condition: NodeId, block: NodeId) {
            self.nodes
                .get_mut(&condition)
                .expect("node exists")
                .else_region = Some(block);
            self.link(block, InsertionPoint::LastIn(condition));
        }
    }

    impl Dsc for Tree {
        fn layout_dims(&self, lds: LdsIdx) -> LayoutDims {
            self.layout
                .get(&lds)
                .cloned()
                .unwrap_or_else(|| LayoutDims::new(PrimaryDim::X1, Vec::new()))
        }
    }

    impl TransferUnrolling for Tree {
        fn non_broadcast_lds_dims(&self, lds: LdsIdx) -> Option<Vec<PrimaryDim>> {
            self.non_broadcast.get(&lds).cloned()
        }
    }

    impl Allocations for Tree {
        fn allocation(&self, _stored: StoredStream) -> Option<AllocId> {
            None
        }

        fn value_allocation(&self, _scale: AllocId) -> Option<AllocId> {
            None
        }
    }

    impl AllocationPaddings for Tree {
        fn padding(&self, _alloc: AllocId) -> PaddingForm {
            PaddingForm::default()
        }
    }

    impl DscAllocations for Tree {
        fn own_lds_idx(&self, lds: LdsIdx) -> LdsIdx {
            lds
        }

        fn allocation_in(&self, origin: DataOrigin, storage: DdcMemory) -> Option<AllocId> {
            self.allocations.get(&(origin, storage)).copied()
        }

        fn set_allocation_in(&mut self, _lds: LdsIdx, _storage: DdcMemory, _alloc: AllocId) {}

        fn alloc_users(&self, alloc: AllocId) -> Vec<NodeId> {
            self.users.get(&alloc).cloned().unwrap_or_default()
        }

        fn alloc_component(&self, _alloc: AllocId) -> DdcMemory {
            DdcMemory::SfpLrf
        }

        fn alloc_origin(&self, _alloc: AllocId) -> DataOrigin {
            DataOrigin::LabeledDs(LdsIdx(0))
        }

        fn alloc_node(&self, alloc: AllocId) -> NodeId {
            NodeId(alloc.0)
        }

        fn reduce_users_or_delete(
            &mut self,
            alloc_use: AllocationUse,
            _can_delete: CanDelete,
        ) -> bool {
            self.reduced.push((alloc_use.alloc(), alloc_use.user()));
            false
        }
    }

    impl ExternalStreams for Tree {
        fn storage_or_datastream_is_external(
            &self,
            data: DataInfo,
            storage: SenComponent,
            direction: StreamDirection,
        ) -> bool {
            self.external == Some((data.data_connect, storage, direction))
        }
    }

    impl SkipRegResults for Tree {
        fn set_dst_latch_data_id(&mut self, transfer: NodeId, dst: usize, id: LatchDataId) {
            self.dst_latch.push((transfer, dst, id));
        }

        fn set_src_latch_data_id(&mut self, transfer: NodeId, id: LatchDataId) {
            self.src_latch.push((transfer, id));
        }

        fn set_compute_input_latch_data_id(
            &mut self,
            compute: NodeId,
            input: usize,
            id: LatchDataId,
        ) {
            self.input_latch.push((compute, input, id));
        }

        fn remove_alloc_use(&mut self, alloc_use: AllocationUse) {
            self.removed.push((alloc_use.alloc(), alloc_use.user()));
        }

        fn set_sole_compute_output(
            &mut self,
            compute: NodeId,
            unit: SenComponent,
            data: DataInfo,
        ) {
            self.compute_outputs.push((compute, unit, data));
        }

        fn resize_compute_inputs_to(
            &mut self,
            compute: NodeId,
            input: InputIdx,
            unit: SenComponent,
            data: DataInfo,
        ) {
            self.compute_input_writes.push((compute, input, unit, data));
        }
    }

    impl FifoResults for Tree {
        fn connect_consumers(&self, _connect: Option<DataConnect>) -> Vec<FifoConsumer> {
            self.consumers.clone()
        }

        fn is_opaque(&self, compute: NodeId) -> bool {
            self.opaque.contains(&compute)
        }

        fn transfer_ends(&self, _transfer: NodeId) -> TransferEnds {
            self.ends.clone().expect("the transfer's ends are stated")
        }

        fn insert_allocate(&mut self, alloc: AllocId, node: DdcAllocateNode, at: InsertionPoint) {
            self.allocated.push((alloc, node, at));
        }

        fn set_dst_storage(&mut self, transfer: NodeId, dst: usize, storage: SenComponent) {
            self.dst_storage.push((transfer, dst, storage));
        }

        fn set_src_storage(&mut self, transfer: NodeId, storage: SenComponent) {
            self.src_storage.push((transfer, storage));
        }

        fn set_compute_input_unit(&mut self, compute: NodeId, input: usize, unit: SenComponent) {
            self.compute_inputs.push((compute, input, unit));
        }

        fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
            self.alloc_users.push((alloc, user));
        }
    }

    /// A STAGE'S EXTENTS AS ENTRY 252 WRITES THEM, recording the ORDER of every write so the
    /// deliberate `makeDimNotSymbolic`-before-padding sequence is observable.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct Extents {
        values: BTreeMap<PrimaryDim, u32>,
        splits: BTreeSet<(DimSplit, PrimaryDim)>,
        symbolic: BTreeSet<PrimaryDim>,
        writes: Vec<String>,
    }

    impl StageExtents for Extents {
        fn copy_dim_value_from(&mut self, other: &Self, dim: PrimaryDim) {
            if let Some(value) = other.values.get(&dim) {
                self.values.insert(dim, *value);
            }
            self.writes.push(format!("value {dim:?}"));
        }

        fn states(&self, split: DimSplit, dim: PrimaryDim) -> bool {
            self.splits.contains(&(split, dim))
        }

        fn copy_split_from(&mut self, other: &Self, split: DimSplit, dim: PrimaryDim) {
            if other.splits.contains(&(split, dim)) {
                self.splits.insert((split, dim));
            }
            self.writes.push(format!("{split:?} {dim:?}"));
        }

        fn make_dim_not_symbolic(&mut self, dim: PrimaryDim) {
            self.symbolic.remove(&dim);
            self.writes.push(format!("not symbolic {dim:?}"));
        }

        fn clear_split(&mut self, split: DimSplit) {
            self.splits.retain(|(kind, _)| *kind != split);
            self.writes.push(format!("clear {split:?}"));
        }

        fn split_sizes(&self, split: DimSplit, dim: PrimaryDim) -> Vec<Elements> {
            if self.splits.contains(&(split, dim)) {
                vec![Elements(1), Elements(1)]
            } else {
                Vec::new()
            }
        }
    }

    fn dims(first: PrimaryDim, rest: &[PrimaryDim]) -> LoopDims {
        let kinded = |dim: PrimaryDim| PrimaryDimAndKind {
            dim,
            kind: MetaDimKind::Unpadded,
        };
        LoopDims::new(kinded(first), rest.iter().copied().map(kinded).collect())
    }

    fn operand(unit: SenComponent, storage: SenComponent, lds: Option<LdsIdx>) -> Operand {
        Operand {
            unit,
            storage,
            data: DataInfo {
                data_connect: Some(DataConnect::ArfPt),
                my_lds_idx: lds,
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    fn transfer_node(name: &str, src: Operand, dsts: Dsts) -> TransferNode {
        TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            name: NodeName(name.to_string()),
            src,
            dsts,
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
        }
    }

    #[test]
    fn splitting_a_band_keeps_the_outermost_set_and_repoints_every_condition_at_its_own_new_loop() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let base = tree.loop_node(
            1,
            2,
            dims(PrimaryDim::In, &[PrimaryDim::Out, PrimaryDim::Mb]),
            Some(root),
        );
        let moved = tree.add(
            "transfer",
            Kind::Transfer(transfer_node(
                "transfer",
                operand(SenComponent::Pe, SenComponent::Lx, None),
                Dsts::new(
                    operand(SenComponent::Sfp, SenComponent::L0, None),
                    Vec::new(),
                ),
            )),
            Some(base.0),
        );
        let condition = tree.add(
            "cond0",
            Kind::Condition(
                LoopCondComposite {
                    or_of_ands: vec![vec![LoopCond {
                        loop_node: base.0,
                        dim: PrimaryDim::Out,
                        op: CondOp::Eq,
                        against: CondValType::Last,
                    }]],
                    negated: false,
                },
                CoreClSet(BTreeMap::new()),
            ),
            Some(base.0),
        );
        let metadata = Metadata::default();

        let split = LoopBandSplit::of(
            &tree,
            &metadata,
            base,
            &[dims(PrimaryDim::Mb, &[]), dims(PrimaryDim::Out, &[])],
            UnspecifiedDims::Innermost,
        )
        .expect("the sets are drawn from the base loop's own dims and do not overlap");
        let innermost = split_loop_band_on_dim(&mut tree, split);

        // The base loop keeps the outermost set, and the dims no set named became the innermost loop.
        assert_eq!(
            tree.loop_dims(base).iter().collect::<Vec<_>>(),
            dims(PrimaryDim::Mb, &[]).iter().collect::<Vec<_>>()
        );
        assert_eq!(
            tree.loop_dims(innermost).iter().collect::<Vec<_>>(),
            dims(PrimaryDim::In, &[]).iter().collect::<Vec<_>>()
        );
        let expected = format!(
            "{}__split",
            construct_loop_node(DatastageId(1), DatastageId(2), dims(PrimaryDim::Mb, &[]))
                .name
                .0
        );
        assert_eq!(tree.node_name(base.0), NodeName(expected));

        // A perfect nest, with the base loop's own children in the innermost loop.
        let middle = tree.children(base.0);
        assert_eq!(middle.len(), 1);
        assert_eq!(tree.children(middle[0]), vec![innermost.0]);
        assert_eq!(tree.children(innermost.0), vec![moved, condition]);

        // The condition compared against the base loop on `Out`, which is the MIDDLE loop's dim now.
        assert_eq!(
            tree.loop_cond(condition).or_of_ands,
            vec![vec![LoopCond {
                loop_node: middle[0],
                dim: PrimaryDim::Out,
                op: CondOp::Eq,
                against: CondValType::Last,
            }]]
        );
    }

    #[test]
    fn a_dim_named_by_two_split_sets_is_no_split_at_all() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let base = tree.loop_node(1, 2, dims(PrimaryDim::In, &[PrimaryDim::Out]), Some(root));

        assert!(
            LoopBandSplit::of(
                &tree,
                &Metadata::default(),
                base,
                &[dims(PrimaryDim::In, &[]), dims(PrimaryDim::In, &[])],
                UnspecifiedDims::Innermost,
            )
            .is_none()
        );
    }

    #[test]
    fn splitting_a_band_on_its_datastage_nests_a_copy_of_the_numerator_and_takes_it_as_the_den() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let base = tree.loop_node(0, 3, dims(PrimaryDim::In, &[]), Some(root));
        let condition = tree.add(
            "cond0",
            Kind::Condition(LoopCondComposite::default(), CoreClSet(BTreeMap::new())),
            Some(base.0),
        );
        let mut stages: DataStages<()> =
            DataStages(BTreeMap::from([(DatastageId(0), DataStage::default())]));

        let site = DatastageSplitSite::of(
            &tree,
            &Metadata::default(),
            &stages,
            base,
            DatastageExploration::Open,
        )
        .expect("an internal loop whose numerator stage exists, before exploration finished");
        let minted = split_loop_band_on_datastage(&mut tree, &mut stages, site);

        assert_eq!(tree.loop_num(minted), DatastageId(1));
        assert_eq!(tree.loop_den(minted), DatastageId(3));
        assert_eq!(tree.loop_den(base), DatastageId(1));
        assert_eq!(tree.children(base.0), vec![minted.0]);
        assert_eq!(tree.children(minted.0), vec![condition]);
        assert_eq!(tree.adjusted, vec![(condition, base, vec![base, minted])]);
    }

    #[test]
    fn hoisting_a_transfer_takes_its_allocation_and_reproduces_the_core_condition_it_sat_under() {
        let mut tree = Tree::default();
        tree.core_ids = vec![Core::checked(0).expect("core 0")];
        tree.corelets = BTreeSet::from([Corelet::checked(0).expect("corelet 0")]);
        let core_cl = CoreClSet(BTreeMap::from([(
            Core::checked(0).expect("core 0"),
            BTreeSet::from([Corelet::checked(0).expect("corelet 0")]),
        )]));

        let root = tree.add("root", Kind::Block, None);
        let outer = tree.loop_node(1, 2, dims(PrimaryDim::In, &[]), Some(root));
        let condition = tree.add(
            "cond0",
            Kind::Condition(LoopCondComposite::default(), core_cl.clone()),
            Some(outer.0),
        );
        let then_region = tree.add("cond0_then", Kind::Block, Some(condition));
        tree.nodes
            .get_mut(&condition)
            .expect("condition exists")
            .then_region = Some(then_region);
        let inner = tree.loop_node(2, 3, dims(PrimaryDim::Out, &[]), Some(then_region));
        let moved = tree.add(
            "t0",
            Kind::Transfer(transfer_node(
                "t0",
                operand(SenComponent::Pe, SenComponent::Lx, None),
                Dsts::new(
                    operand(SenComponent::Sfp, SenComponent::L0, Some(LdsIdx(5))),
                    Vec::new(),
                ),
            )),
            Some(inner.0),
        );
        let allocation = tree.add("allocate_lds5_l0", Kind::Allocate, Some(inner.0));
        tree.dst_allocation.insert(LdsIdx(5), allocation);

        let hoist = TransferMove::of(
            &tree,
            &Metadata::default(),
            moved,
            outer,
            DatastageExploration::Open,
        )
        .expect("an internal transfer whose destination loop is an ancestor");
        assert_eq!(hoist.new_parent(), outer);
        move_transfer_node(&mut tree, hoist);

        // The allocation and a reproduced core condition land before the loop the transfer left.
        let siblings = tree.children(then_region);
        assert_eq!(siblings.len(), 3);
        assert_eq!(siblings[0], allocation);
        assert_eq!(siblings[2], inner.0);
        let reproduced = siblings[1];
        assert_eq!(
            tree.node_name(reproduced),
            NodeName("core_corelet_reuse_t0".to_string())
        );
        assert_eq!(tree.core_cl_cond(reproduced), core_cl);

        // And the transfer is the first thing in that condition's own then region.
        let region = tree
            .then_branch(reproduced)
            .expect("a then region was added");
        assert_eq!(tree.children(region), vec![moved]);
    }

    #[test]
    fn a_destination_loop_that_does_not_enclose_the_transfer_is_no_move() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let elsewhere = tree.loop_node(1, 2, dims(PrimaryDim::In, &[]), Some(root));
        let owner = tree.loop_node(2, 3, dims(PrimaryDim::Out, &[]), Some(root));
        let moved = tree.add(
            "t0",
            Kind::Transfer(transfer_node(
                "t0",
                operand(SenComponent::Pe, SenComponent::Lx, None),
                Dsts::new(
                    operand(SenComponent::Sfp, SenComponent::L0, None),
                    Vec::new(),
                ),
            )),
            Some(owner.0),
        );

        assert!(
            TransferMove::of(
                &tree,
                &Metadata::default(),
                moved,
                elsewhere,
                DatastageExploration::Open,
            )
            .is_none()
        );
    }

    #[test]
    fn a_fifo_destination_gains_an_lrf_allocation_and_every_consumer_is_repointed_at_it() {
        let mut tree = Tree::default();
        tree.layout.insert(
            LdsIdx(2),
            LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Out]),
        );
        tree.ends = Some(TransferEnds {
            src: StoredStream {
                stream: DataStream {
                    origin: DataOrigin::LabeledDs(LdsIdx(2)),
                    data_connect: Some(DataConnect::ArfPt),
                },
                storage: DfirUnit::Pe,
            },
            dsts: Vec::new(),
        });

        let root = tree.add("root", Kind::Block, None);
        let moved = tree.add(
            "t0",
            Kind::Transfer(transfer_node(
                "t0",
                operand(SenComponent::Pe, SenComponent::Lx, Some(LdsIdx(2))),
                // NOT one of `dsc2::memories`, which is exactly what a FIFO result is.
                Dsts::new(
                    operand(
                        SenComponent::Sfp,
                        SenComponent::NoComponent,
                        Some(LdsIdx(2)),
                    ),
                    Vec::new(),
                ),
            )),
            Some(root),
        );
        let consuming_transfer = NodeId(50);
        let consuming_compute = NodeId(51);
        tree.consumers = vec![
            FifoConsumer::Transfer(consuming_transfer),
            FifoConsumer::Compute(
                consuming_compute,
                ComputeNode {
                    name: NodeName("c0".to_string()),
                    op: DdlComputeType::Macc,
                    ex_unit: SenComponent::Sfp,
                    inputs: vec![operand(SenComponent::Sfp, SenComponent::NoComponent, None)],
                    outputs: Vec::new(),
                    num_folds_engaged: NumFolds::ONE,
                    data_format: None,
                    instr_attribute: InstrAttribute::default(),
                },
            ),
        ];
        let mut metadata = Metadata::default();

        let site = FifoConversionSite::of(&metadata, moved, DatastageExploration::Open)
            .expect("an internal transfer before exploration finished");
        assert!(convert_result_from_fifo_to_reg(
            &mut tree,
            &mut metadata,
            site,
            AllocId(9)
        ));

        // The SFP's own register file, allocated right before the transfer.
        let (alloc, node, at) = tree
            .allocated
            .first()
            .cloned()
            .expect("an allocation minted");
        assert_eq!(alloc, AllocId(9));
        assert_eq!(node.component, DdcMemory::SfpLrf);
        assert_eq!(
            node.name,
            NodeName(format!(
                "allocate_lds2_sfplrf_{}_fifo_to_reg",
                connect_spelling(Some(DataConnect::ArfPt))
            ))
        );
        assert_eq!(at, InsertionPoint::Before(moved));

        // The destination and every consumer now name that register file.
        assert_eq!(tree.dst_storage, vec![(moved, 0, SenComponent::Sfplrf)]);
        assert_eq!(
            tree.src_storage,
            vec![(consuming_transfer, SenComponent::Sfplrf)]
        );
        assert_eq!(
            tree.compute_inputs,
            vec![(consuming_compute, 0, SenComponent::Sfplrf)]
        );
        assert_eq!(
            tree.alloc_users,
            vec![
                (AllocId(9), consuming_transfer),
                (AllocId(9), consuming_compute)
            ]
        );
    }

    #[test]
    fn unrolling_a_transfer_mints_one_loop_per_layout_dim_against_a_minimizing_denominator() {
        let mut tree = Tree::default();
        tree.layout.insert(
            LdsIdx(1),
            LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Out]),
        );
        let root = tree.add("root", Kind::Block, None);
        let outer = tree.loop_node(1, 10, dims(PrimaryDim::In, &[]), Some(root));
        let inner = tree.loop_node(2, 20, dims(PrimaryDim::Out, &[]), Some(outer.0));
        let moved = tree.add(
            "t0",
            Kind::Transfer(transfer_node(
                "t0",
                operand(SenComponent::Pe, SenComponent::Lx, Some(LdsIdx(1))),
                Dsts::new(
                    operand(SenComponent::Sfp, SenComponent::L0, None),
                    Vec::new(),
                ),
            )),
            Some(inner.0),
        );
        let mut stages: DataStages<()> = DataStages::default();
        let mut metadata = Metadata::default();
        let site = InternalNode::of(&metadata, moved).expect("an internal transfer");

        assert!(unroll_transfer(&mut tree, &mut stages, &mut metadata, site));

        // One loop per layout dim, the innermost layout dim outermost, each against the enclosing
        // loop's own denominator and the one new minimizing stage.
        let den = DatastageId(0);
        assert_eq!(metadata.datastages[&den].strategy, Strategy::Minimize);
        let over_out = LoopId(tree.children(inner.0)[0]);
        let over_in = LoopId(tree.children(over_out.0)[0]);
        assert_eq!(tree.loop_num(over_out), DatastageId(20));
        assert_eq!(tree.loop_num(over_in), DatastageId(10));
        assert_eq!(tree.loop_den(over_out), den);
        assert_eq!(tree.loop_den(over_in), den);
        assert_eq!(tree.children(over_in.0), vec![moved]);
    }

    #[test]
    fn unrolling_for_symbolic_dims_fills_the_new_stage_and_stops_being_symbolic_before_padding() {
        let mut tree = Tree::default();
        tree.non_broadcast
            .insert(LdsIdx(1), vec![PrimaryDim::In, PrimaryDim::Out]);
        let root = tree.add("root", Kind::Block, None);
        let outer = tree.loop_node(1, 10, dims(PrimaryDim::In, &[]), Some(root));
        let moved = tree.add(
            "t0",
            Kind::Transfer(transfer_node(
                "t0",
                operand(SenComponent::Pe, SenComponent::Lx, Some(LdsIdx(1))),
                Dsts::new(
                    operand(SenComponent::Sfp, SenComponent::L0, None),
                    Vec::new(),
                ),
            )),
            Some(outer.0),
        );

        // The reference stage states `In`, and its EPILOGUE half deliberately states no padding.
        let reference = DataStage {
            ss: StageDims {
                name: StageName("10".to_string()),
                dims: Extents {
                    values: BTreeMap::from([(PrimaryDim::In, 4)]),
                    splits: BTreeSet::from([
                        (DimSplit::Corelet, PrimaryDim::In),
                        (DimSplit::Padding, PrimaryDim::In),
                    ]),
                    symbolic: BTreeSet::from([PrimaryDim::In]),
                    writes: Vec::new(),
                },
            },
            el: StageDims {
                name: StageName("10el".to_string()),
                dims: Extents {
                    values: BTreeMap::from([(PrimaryDim::In, 2)]),
                    splits: BTreeSet::from([(DimSplit::Corelet, PrimaryDim::In)]),
                    symbolic: BTreeSet::from([PrimaryDim::In]),
                    writes: Vec::new(),
                },
            },
        };
        let mut stages = DataStages(BTreeMap::from([(DatastageId(10), reference)]));
        let symbolic = BTreeMap::from([(
            PrimaryDim::In,
            SymbolicDimInfo {
                max_size: MaxSize(8),
                granularity: Granularity::new(NonZeroU32::new(1).expect("a step of one")),
            },
        )]);
        let site = InternalNode::of(&Metadata::default(), moved).expect("an internal transfer");

        assert!(unroll_transfer_for_symbolic_dims(
            &mut tree,
            &mut stages,
            site,
            &symbolic
        ));

        let minted = &stages.0[&DatastageId(1)];
        assert_eq!(minted.ss.dims.values, BTreeMap::from([(PrimaryDim::In, 4)]));
        assert_eq!(minted.el.dims.values, BTreeMap::from([(PrimaryDim::In, 2)]));
        assert!(minted.ss.dims.symbolic.is_empty());
        // The padding copy runs LAST, and the epilogue half is asked for it because the STEADY-STATE
        // half states it.
        assert_eq!(
            minted.el.dims.writes,
            vec![
                "value In".to_string(),
                "Corelet In".to_string(),
                "not symbolic In".to_string(),
                "Padding In".to_string(),
            ]
        );
        assert!(
            !minted
                .el
                .dims
                .splits
                .contains(&(DimSplit::Padding, PrimaryDim::In))
        );

        // Only the symbolic dim is unrolled, and against the enclosing loop's denominator.
        let over_in = LoopId(tree.children(outer.0)[0]);
        assert_eq!(tree.loop_num(over_in), DatastageId(10));
        assert_eq!(tree.loop_den(over_in), DatastageId(1));
        assert_eq!(tree.children(over_in.0), vec![moved]);
    }

    /// THE ONE DATASTREAM THIS STAND-IN CALLS EXTERNAL, keyed on the triple entries 253/254 hand
    /// over — so a port that passed `Operand::unit` instead of `Operand::storage` cannot match it.
    struct External(Option<(Option<DataConnect>, SenComponent, StreamDirection)>);

    impl ExternalStreams for External {
        fn storage_or_datastream_is_external(
            &self,
            data: DataInfo,
            storage: SenComponent,
            direction: StreamDirection,
        ) -> bool {
            self.0 == Some((data.data_connect, storage, direction))
        }
    }

    #[test]
    fn a_transfers_source_is_asked_about_by_its_storage_and_its_destination_by_the_destinations() {
        let src = operand(SenComponent::Pe, SenComponent::Lx, None);
        let dst = operand(SenComponent::Sfp, SenComponent::L0, None);
        let node = transfer_node("t0", src, Dsts::new(dst, Vec::new()));

        let by_src_storage = External(Some((
            Some(DataConnect::ArfPt),
            SenComponent::Lx,
            StreamDirection::Incoming,
        )));
        assert!(src_related_to_external_nodes(&by_src_storage, &node));
        // The SOURCE's `unit_` is not what is asked about, and neither is the destination's storage.
        let by_src_unit = External(Some((
            Some(DataConnect::ArfPt),
            SenComponent::Pe,
            StreamDirection::Incoming,
        )));
        assert!(!src_related_to_external_nodes(&by_src_unit, &node));

        let by_dst_storage = External(Some((
            Some(DataConnect::ArfPt),
            SenComponent::L0,
            StreamDirection::Outgoing,
        )));
        assert!(dest_related_to_external_nodes(
            &by_dst_storage,
            node.dsts.first()
        ));
        assert!(!dest_related_to_external_nodes(
            &by_src_storage,
            node.dsts.first()
        ));
    }

    /// e341: EITHER end relates a transfer to an external node, and each end is asked about by its
    /// own storage and its own direction.
    #[test]
    fn a_transfer_is_related_to_external_nodes_through_either_of_its_ends() {
        let src = operand(SenComponent::Pe, SenComponent::Lx, None);
        let dst = operand(SenComponent::Sfp, SenComponent::L0, None);
        let node = transfer_node("t0", src, Dsts::new(dst, Vec::new()));

        let by_src = External(Some((
            Some(DataConnect::ArfPt),
            SenComponent::Lx,
            StreamDirection::Incoming,
        )));
        let by_dst = External(Some((
            Some(DataConnect::ArfPt),
            SenComponent::L0,
            StreamDirection::Outgoing,
        )));
        assert!(transfer_related_to_external_nodes(&by_src, &node));
        assert!(transfer_related_to_external_nodes(&by_dst, &node));

        // Neither end: the destination's storage asked in the SOURCE's direction.
        let neither = External(Some((
            Some(DataConnect::ArfPt),
            SenComponent::L0,
            StreamDirection::Incoming,
        )));
        assert!(!transfer_related_to_external_nodes(&neither, &node));
    }

    /// e339: the register allocation is dropped for the transfer AND for every consumer, and the one
    /// minted id links them.
    #[test]
    fn latching_a_register_result_drops_every_allocation_use_and_links_them_by_one_id() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let dst = operand(SenComponent::Sfp, SenComponent::Sfplrf, Some(LdsIdx(1)));
        let t0 = tree.add(
            "t0",
            Kind::Transfer(transfer_node(
                "t0",
                operand(SenComponent::Pe, SenComponent::Lx, Some(LdsIdx(1))),
                Dsts::new(dst, Vec::new()),
            )),
            Some(root),
        );
        // A consuming transfer reads that register file as its SOURCE, and a consuming compute reads
        // it on the one input whose connect matches.
        let consuming = tree.add(
            "t1",
            Kind::Transfer(transfer_node(
                "t1",
                operand(SenComponent::Sfp, SenComponent::Sfplrf, Some(LdsIdx(1))),
                Dsts::new(
                    operand(SenComponent::Pe, SenComponent::Lx, None),
                    Vec::new(),
                ),
            )),
            Some(root),
        );
        let computing = tree.add("c0", Kind::Compute, Some(root));
        tree.consumers = vec![
            FifoConsumer::Transfer(consuming),
            FifoConsumer::Compute(
                computing,
                ComputeNode {
                    name: NodeName("c0".to_string()),
                    op: DdlComputeType::Macc,
                    ex_unit: SenComponent::Sfp,
                    inputs: vec![operand(
                        SenComponent::Sfplrf,
                        SenComponent::NoComponent,
                        Some(LdsIdx(1)),
                    )],
                    outputs: Vec::new(),
                    num_folds_engaged: NumFolds::ONE,
                    data_format: None,
                    instr_attribute: InstrAttribute::default(),
                },
            ),
        ];
        tree.allocations.insert(
            (DataOrigin::LabeledDs(LdsIdx(1)), DdcMemory::SfpLrf),
            AllocId(9),
        );
        tree.users.insert(AllocId(9), vec![t0, consuming, computing]);

        let mut metadata = Metadata::default();
        let mut latch_ids = LatchDataIds::default();
        let node = tree.transfer(t0);
        let dest =
            TransferDest::of(&metadata, t0, &node, DestIdx(0)).expect("an internal destination");

        assert!(convert_result_to_skip_reg(
            &mut tree,
            &mut metadata,
            &mut latch_ids,
            dest,
            SkipRegTarget::Latch
        ));

        assert_eq!(
            tree.reduced,
            vec![
                (AllocId(9), t0),
                (AllocId(9), consuming),
                (AllocId(9), computing)
            ]
        );
        assert_eq!(tree.dst_storage, vec![(t0, 0, SenComponent::Latch)]);
        assert_eq!(tree.src_storage, vec![(consuming, SenComponent::Latch)]);
        assert_eq!(tree.compute_inputs, vec![(computing, 0, SenComponent::Latch)]);

        // ONE id across the producer and both consumers, and the counter has moved past it.
        let id = LatchDataId(0);
        assert_eq!(tree.dst_latch, vec![(t0, 0, id)]);
        assert_eq!(tree.src_latch, vec![(consuming, id)]);
        assert_eq!(tree.input_latch, vec![(computing, 0, id)]);
        assert_eq!(latch_ids.next(), LatchDataId(1));
    }

    /// e339: a destination that is not a register at all is the reference's OWN answer — accepted
    /// where it already skips the register file, refused otherwise, and untouched either way.
    #[test]
    fn a_result_that_already_skips_the_register_file_is_accepted_untouched() {
        fn answered(storage: SenComponent, target: SkipRegTarget) -> bool {
            let mut tree = Tree::default();
            let root = tree.add("root", Kind::Block, None);
            let t0 = tree.add(
                "t0",
                Kind::Transfer(transfer_node(
                    "t0",
                    operand(SenComponent::Pe, SenComponent::Lx, Some(LdsIdx(1))),
                    Dsts::new(
                        operand(SenComponent::Sfp, storage, Some(LdsIdx(1))),
                        Vec::new(),
                    ),
                )),
                Some(root),
            );
            let mut metadata = Metadata::default();
            let node = tree.transfer(t0);
            let dest =
                TransferDest::of(&metadata, t0, &node, DestIdx(0)).expect("an internal destination");
            let answer = convert_result_to_skip_reg(
                &mut tree,
                &mut metadata,
                &mut LatchDataIds::default(),
                dest,
                target,
            );
            assert!(tree.dst_storage.is_empty(), "the destination is not rewritten");
            assert!(tree.reduced.is_empty(), "no allocation use is dropped");
            answer
        }

        // A latch is already a latch, and a FIFO is any non-latch that is not a memory.
        assert!(answered(SenComponent::Latch, SkipRegTarget::Latch));
        assert!(answered(SenComponent::NoComponent, SkipRegTarget::Fifo));
        // The same two storages read against the OTHER target, and a memory against either.
        assert!(!answered(SenComponent::Latch, SkipRegTarget::Fifo));
        assert!(!answered(SenComponent::NoComponent, SkipRegTarget::Latch));
        assert!(!answered(SenComponent::Qgi, SkipRegTarget::Fifo));
    }

    /// e340: the compute lands beside the transfer, inherits the allocation, and the two ends swap —
    /// the compute writes the register file, the transfer writes the LAST HOP.
    #[test]
    fn a_compute_inserted_before_a_register_takes_its_allocation_and_reads_the_last_hop() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let dst = operand(SenComponent::Sfp, SenComponent::Sfplrf, Some(LdsIdx(1)));
        let t0 = tree.add(
            "t0",
            Kind::Transfer(transfer_node(
                "t0",
                operand(SenComponent::Pe, SenComponent::Lx, Some(LdsIdx(1))),
                Dsts::new(dst, Vec::new())
                    .with_hops(vec![Hops(vec![SenComponent::L0, SenComponent::Sfp])]),
            )),
            Some(root),
        );
        let computing = tree.add("c0", Kind::Compute, None);
        tree.allocations.insert(
            (DataOrigin::LabeledDs(LdsIdx(1)), DdcMemory::SfpLrf),
            AllocId(9),
        );
        tree.users.insert(AllocId(9), vec![t0]);

        let metadata = Metadata::default();
        let node = tree.transfer(t0);
        let dest =
            TransferDest::of(&metadata, t0, &node, DestIdx(0)).expect("an internal destination");
        let unplaced = UnplacedNode::of(&tree, computing).expect("a compute not yet in the tree");

        assert!(insert_compute_between_transfer_and_reg(
            &mut tree,
            dest,
            unplaced,
            InputIdx(2)
        ));

        // Placed immediately after the transfer, and the allocation's user in its place.
        assert_eq!(tree.children(root), vec![t0, computing]);
        assert_eq!(tree.alloc_users, vec![(AllocId(9), computing)]);
        assert_eq!(tree.removed, vec![(AllocId(9), t0)]);

        // The compute writes the register file as its sole output and reads the transfer's last hop
        // on input 2, which is what the transfer now writes.
        assert_eq!(
            tree.compute_outputs,
            vec![(computing, SenComponent::Sfplrf, dst.data)]
        );
        assert_eq!(
            tree.compute_input_writes,
            vec![(computing, InputIdx(2), SenComponent::Sfp, dst.data)]
        );
        assert_eq!(tree.dst_storage, vec![(t0, 0, SenComponent::Sfp)]);
    }
}

#[cfg(test)]
mod tests_e118_e123 {
    // ⭐ TESTS FOR ENTRIES 118-123. Union this module with this file's other test modules when they
    // land.
    use super::*;

    use super::super::metadata::{Allocation, Ends, OpaqueOp};
    use crate::generated::RegName;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, FoldDim, InstrAttribute, LayoutDims, MaxDimSize, NumChunks,
        ReplicationFactor, StartAddress, SyncDirection, SyncStrength, SyncUnits, TransferPadding,
    };
    use crate::units::NumFolds;

    /// ONE DSC'S ALLOCATIONS, COMPUTES AND CENSUS, stated as the tables entries 118-120 read and the
    /// lists of what they wrote.
    #[derive(Default)]
    struct Cloning {
        allocs: BTreeMap<AllocId, AllocateNode>,
        alloc_nodes: BTreeMap<AllocId, NodeId>,
        computes: BTreeMap<NodeId, ComputeNode>,
        opaque: BTreeSet<NodeId>,
        in_memory: BTreeMap<(DataOrigin, DdcMemory), AllocId>,
        mem_org: BTreeMap<(LdsIdx, SenComponent), Option<AllocId>>,
        constants: BTreeMap<(ConstIdx, SenComponent), AllocId>,
        census: BTreeMap<NodeIndex, NodeId>,
        placed: Vec<(NodeId, NodeId)>,
        users: Vec<(AllocId, NodeId)>,
        temp_storage: Vec<(AllocId, NodeName)>,
        minted: Vec<String>,
        next: u32,
    }

    impl Cloning {
        /// One allocation on `component` for `origin`, with the schedule node that is it.
        fn with_alloc(&mut self, component: SenComponent, origin: DataOrigin) -> AllocId {
            let alloc = AllocId(self.mint());
            let node = NodeId(self.mint());
            let (lds, const_idx) = match origin {
                DataOrigin::LabeledDs(lds) => (Some(lds), None),
                DataOrigin::Constant(constant) => (None, Some(constant)),
            };
            self.allocs.insert(
                alloc,
                AllocateNode {
                    name: NodeName(format!("a{}", alloc.0)),
                    component,
                    lds,
                    const_idx,
                    temp_storage_for_compute: Some(NodeName("stale".to_string())),
                    layout: AllocLayout::new(
                        (PrimaryDim::I, MaxDimSize::Resolved(Elements(64))),
                        Vec::new(),
                    ),
                    start_address: StartAddress::new(FoldDim::default()),
                    placement: AllocPlacement::default(),
                    gap_stick_spread: BTreeMap::new(),
                    alloc_users: vec![NodeId(999)],
                },
            );
            self.alloc_nodes.insert(alloc, node);
            if let Some(memory) = component_memory(component) {
                self.in_memory.insert((origin, memory), alloc);
            }
            alloc
        }

        fn mint(&mut self) -> u32 {
            self.next += 1;
            self.next
        }
    }

    impl Dsc for Cloning {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            unimplemented!("no layout is read")
        }
    }

    impl DscAllocations for Cloning {
        fn own_lds_idx(&self, _lds: LdsIdx) -> LdsIdx {
            unimplemented!("no lds is renamed")
        }

        fn allocation_in(&self, origin: DataOrigin, storage: DdcMemory) -> Option<AllocId> {
            self.in_memory.get(&(origin, storage)).copied()
        }

        fn set_allocation_in(&mut self, _lds: LdsIdx, _storage: DdcMemory, _alloc: AllocId) {
            unimplemented!("no allocation is placed")
        }

        fn alloc_users(&self, _alloc: AllocId) -> Vec<NodeId> {
            unimplemented!("no user list is read")
        }

        fn alloc_component(&self, alloc: AllocId) -> DdcMemory {
            component_memory(self.allocs[&alloc].component).expect("a memory component")
        }

        fn alloc_origin(&self, _alloc: AllocId) -> DataOrigin {
            unimplemented!("no origin is read")
        }

        fn alloc_node(&self, alloc: AllocId) -> NodeId {
            self.alloc_nodes[&alloc]
        }

        fn reduce_users_or_delete(&mut self, _use: AllocationUse, _can: CanDelete) -> bool {
            unimplemented!("no allocation is deleted")
        }
    }

    impl AllocationsByNode for Cloning {
        fn allocation_of(&self, node: NodeId) -> Option<AllocId> {
            self.alloc_nodes
                .iter()
                .find(|&(_, &at)| at == node)
                .map(|(&alloc, _)| alloc)
        }
    }

    impl AllocateCloning for Cloning {
        fn allocate(&self, alloc: AllocId) -> AllocateNode {
            self.allocs[&alloc].clone()
        }

        fn clone_allocate_after(&mut self, alloc: AllocId, body: AllocateNode) -> AllocId {
            let clone = AllocId(self.mint());
            let node = NodeId(self.mint());
            self.placed.push((self.alloc_nodes[&alloc], node));
            self.allocs.insert(clone, body);
            self.alloc_nodes.insert(clone, node);
            clone
        }

        fn set_temp_storage_for_compute(&mut self, alloc: AllocId, compute: NodeName) {
            self.temp_storage.push((alloc, compute));
        }
    }

    impl ComputeCloning for Cloning {
        fn compute(&self, node: NodeId) -> ComputeNode {
            self.computes[&node].clone()
        }

        fn clone_compute_after(&mut self, node: NodeId, body: ComputeNode) -> NodeId {
            let clone = NodeId(self.mint());
            self.placed.push((node, clone));
            self.computes.insert(clone, body);
            clone
        }
    }

    impl ComponentAllocations for Cloning {
        fn has_mem_org(&self, lds: LdsIdx, storage: SenComponent) -> bool {
            self.mem_org.contains_key(&(lds, storage))
        }

        fn copy_mem_org_without_allocation(
            &mut self,
            lds: LdsIdx,
            _from: SenComponent,
            to: SenComponent,
        ) {
            self.mem_org.insert((lds, to), None);
        }

        fn mem_org_allocation(&self, lds: LdsIdx, storage: SenComponent) -> Option<AllocId> {
            self.mem_org.get(&(lds, storage)).copied().flatten()
        }

        fn set_mem_org_allocation(&mut self, lds: LdsIdx, storage: SenComponent, alloc: AllocId) {
            self.mem_org.insert((lds, storage), Some(alloc));
        }

        fn has_constant_allocation(&self, constant: ConstIdx, storage: SenComponent) -> bool {
            self.constants.contains_key(&(constant, storage))
        }

        fn set_constant_allocation(
            &mut self,
            constant: ConstIdx,
            storage: SenComponent,
            alloc: AllocId,
        ) {
            self.constants.insert((constant, storage), alloc);
        }
    }

    impl ScheduleSurgery for Cloning {
        fn node_name(&self, _node: NodeId) -> NodeName {
            unimplemented!("no name is read off the tree")
        }

        fn set_node_name(&mut self, _node: NodeId, _name: NodeName) {
            unimplemented!("a clone carries its own name")
        }

        fn parent(&self, _node: NodeId) -> Option<NodeId> {
            unimplemented!("the placement is the carrier's")
        }

        fn owner_loop(&self, _node: NodeId) -> Option<LoopId> {
            unimplemented!("no loop is walked")
        }

        fn transfer(&self, _node: NodeId) -> TransferNode {
            unimplemented!("no transfer is cloned here")
        }

        fn loop_num(&self, _loop_node: LoopId) -> DatastageId {
            unimplemented!("no loop is read")
        }

        fn loop_den(&self, _loop_node: LoopId) -> DatastageId {
            unimplemented!("no loop is read")
        }

        fn loop_dims(&self, _loop_node: LoopId) -> LoopDims {
            unimplemented!("no loop is read")
        }

        fn is_parametric(&self, _loop_node: LoopId) -> bool {
            unimplemented!("no loop is read")
        }

        fn new_loop(&mut self, _loop_node: LoopNode) -> LoopId {
            unimplemented!("no loop is minted")
        }

        fn new_block(&mut self, _name: NodeName) -> NodeId {
            unimplemented!("no block is minted")
        }

        fn add_child_node(&mut self, _node: NodeId, _at: InsertionPoint) {
            unimplemented!("a clone is placed by its own cloner")
        }

        fn move_node(&mut self, _node: NodeId, _at: InsertionPoint) {
            unimplemented!("nothing moves")
        }

        fn conditions_under(&self, _root: NodeId) -> Vec<NodeId> {
            unimplemented!("no condition is read")
        }

        fn loop_cond(&self, _condition: NodeId) -> LoopCondComposite {
            unimplemented!("no condition is read")
        }

        fn set_loop_cond(&mut self, _condition: NodeId, _cond: LoopCondComposite) {
            unimplemented!("no condition is written")
        }
    }

    impl FifoResults for Cloning {
        fn connect_consumers(&self, _connect: Option<DataConnect>) -> Vec<FifoConsumer> {
            unimplemented!("no consumer is walked")
        }

        fn is_opaque(&self, compute: NodeId) -> bool {
            self.opaque.contains(&compute)
        }

        fn transfer_ends(&self, _transfer: NodeId) -> TransferEnds {
            unimplemented!("no transfer is read")
        }

        fn insert_allocate(
            &mut self,
            _alloc: AllocId,
            _node: DdcAllocateNode,
            _at: InsertionPoint,
        ) {
            unimplemented!("no allocation is inserted")
        }

        fn set_dst_storage(&mut self, _transfer: NodeId, _dst: usize, _storage: SenComponent) {
            unimplemented!("no destination is repointed")
        }

        fn set_src_storage(&mut self, _transfer: NodeId, _storage: SenComponent) {
            unimplemented!("no source is repointed")
        }

        fn set_compute_input_unit(&mut self, _compute: NodeId, _input: usize, _unit: SenComponent) {
            unimplemented!("a clone carries its own units")
        }

        fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
            self.users.push((alloc, user));
        }
    }

    impl MintedConnects for Cloning {
        fn intern_connect(&mut self, connect: MintedConnect) -> DataConnect {
            self.minted.push(connect.spelling());
            DataConnect::OuttensorToSfp
        }
    }

    impl CensusNodes for Cloning {
        fn census_node(&self, node: NodeIndex) -> NodeId {
            self.census[&node]
        }
    }

    fn operand(unit: SenComponent, connect: Option<DataConnect>, lds: Option<u32>) -> Operand {
        Operand {
            unit,
            storage: unit,
            data: DataInfo {
                data_connect: connect,
                my_lds_idx: lds.map(LdsIdx),
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    fn compute(ex_unit: SenComponent, inputs: Vec<Operand>, outputs: Vec<Operand>) -> ComputeNode {
        ComputeNode {
            name: NodeName("c0".to_string()),
            op: DdlComputeType::Macc,
            ex_unit,
            inputs,
            outputs,
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        }
    }

    #[test]
    fn an_allocation_clone_registers_itself_under_the_toggled_component_and_fills_its_mem_org() {
        // The vendor's own labeled-DS case: a PELRF allocation of lds 1, registered in
        // `newAllocations_[PELRF]` and holding the `memOrg_` entry for its own component.
        let mut dsc = Cloning::default();
        let lds = LdsIdx(1);
        let origin = DataOrigin::LabeledDs(lds);
        let alloc = dsc.with_alloc(SenComponent::Pelrf, origin);
        dsc.mem_org.insert((lds, SenComponent::Pelrf), Some(alloc));
        let node = dsc.alloc_nodes[&alloc];

        let mut metadata = Metadata::default();
        metadata.new_allocations.insert(
            DdcMemory::PeLrf,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(lds, alloc)]),
                ..Allocation::default()
            },
        );

        let split = PeSfpAllocateSplit::of(&dsc, &metadata, alloc).expect("cloneable");
        assert_eq!(split.origin(), origin);
        assert_eq!(split.toggled(), SenComponent::Sfplrf);
        let clone = clone_allocate_for_pe_sfp_work_split(
            &mut dsc,
            &mut metadata,
            split,
            SkipMetadataUpdate::No,
        )
        .expect("cloned");

        // The clone's own state: the suffix, the toggled component, no users and no temp storage.
        let body = &dsc.allocs[&clone];
        assert_eq!(body.name, NodeName("a1_sfp_parallel".to_string()));
        assert_eq!(body.component, SenComponent::Sfplrf);
        assert_eq!(body.lds, Some(lds));
        assert!(body.alloc_users.is_empty());
        assert_eq!(body.temp_storage_for_compute, None);

        // Placed immediately after the original, and recorded as its clone.
        let clone_node = dsc.alloc_nodes[&clone];
        assert_eq!(dsc.placed, vec![(node, clone_node)]);
        assert_eq!(
            metadata.node_cloning_map,
            BTreeMap::from([(node, vec![clone_node])])
        );

        // Registered under SFPLRF, and the labeled DS's SFPLRF slot now holds it.
        assert_eq!(
            metadata.new_allocations[&DdcMemory::SfpLrf].lds_idx_and_alloc_node,
            BTreeMap::from([(lds, clone)])
        );
        assert_eq!(dsc.mem_org[&(lds, SenComponent::Sfplrf)], Some(clone));

        // And a second attempt refuses, because the first recorded itself.
        assert!(PeSfpAllocateSplit::of(&dsc, &metadata, alloc).is_none());
    }

    #[test]
    fn a_pe_allocations_clone_runs_on_sfp_and_is_still_named_pe_parallel() {
        // ⚠️ THE TRAP: the suffix is `"_sfp_parallel"` for PELRF and PESTATE ALONE, so a PE
        // allocation's clone toggles to SFP under the OPPOSITE name. PE has no `DdcMemory`
        // spelling, which is why this is the `skipMetadataUpdate` path.
        let mut dsc = Cloning::default();
        let lds = LdsIdx(2);
        let alloc = dsc.with_alloc(SenComponent::Pe, DataOrigin::LabeledDs(lds));
        dsc.mem_org.insert((lds, SenComponent::Pe), Some(alloc));

        let mut metadata = Metadata::default();
        let split = PeSfpAllocateSplit::of(&dsc, &metadata, alloc).expect("cloneable");
        let clone = clone_allocate_for_pe_sfp_work_split(
            &mut dsc,
            &mut metadata,
            split,
            SkipMetadataUpdate::Yes,
        )
        .expect("cloned");

        assert_eq!(
            dsc.allocs[&clone].name,
            NodeName("a1_pe_parallel".to_string())
        );
        assert_eq!(dsc.allocs[&clone].component, SenComponent::Sfp);

        // Nothing was registered, and the fresh SFP `memOrg_` entry was copied before being filled.
        assert!(metadata.new_allocations.is_empty());
        assert_eq!(dsc.mem_org[&(lds, SenComponent::Sfp)], Some(clone));
    }

    #[test]
    fn a_compute_clones_suffix_every_named_connect_even_where_the_unit_did_not_toggle() {
        // ⚠️ THE TRAP: input 0 sits on LX, which `toggleMap` has no entry for, and its connect takes
        // the suffix all the same.
        let mut dsc = Cloning::default();
        let lds = LdsIdx(3);
        let origin = DataOrigin::LabeledDs(lds);
        let read = dsc.with_alloc(SenComponent::Sfplrf, origin);
        let written = dsc.with_alloc(SenComponent::Pelrf, origin);
        let node = NodeId(100);
        dsc.computes.insert(
            node,
            compute(
                SenComponent::Pe,
                vec![
                    operand(SenComponent::Lx, Some(DataConnect::ArfPt), Some(lds.0)),
                    operand(
                        SenComponent::Pelrf,
                        Some(DataConnect::ArfPtsum),
                        Some(lds.0),
                    ),
                ],
                vec![operand(SenComponent::Sfplrf, None, Some(lds.0))],
            ),
        );

        let mut metadata = Metadata::default();
        let body = dsc.computes[&node].clone();
        let split = PeSfpComputeSplit::of(&metadata, node, &body).expect("cloneable");
        let clone =
            clone_compute_for_pe_sfp_work_split(&mut dsc, &mut metadata, split).expect("cloned");

        let cloned = &dsc.computes[&clone];
        assert_eq!(cloned.name, NodeName("c0_sfp_parallel".to_string()));
        assert_eq!(cloned.ex_unit, SenComponent::Sfp);

        // Input 0's unit is untouched and its connect is not; input 1 and the output both toggle.
        assert_eq!(cloned.inputs[0].unit, SenComponent::Lx);
        assert_eq!(cloned.inputs[1].unit, SenComponent::Sfplrf);
        assert_eq!(cloned.outputs[0].unit, SenComponent::Pelrf);
        assert_eq!(
            dsc.minted,
            vec![
                "arf_pt_sfp_parallel".to_string(),
                "arf_ptsum_sfp_parallel".to_string()
            ]
        );

        // The toggled operands' allocations gained the clone as a user; the untoggled LX one did not.
        assert_eq!(dsc.users, vec![(read, clone), (written, clone)]);
        assert_eq!(dsc.placed, vec![(node, clone)]);
        assert_eq!(
            metadata.node_cloning_map,
            BTreeMap::from([(node, vec![clone])])
        );
    }

    #[test]
    fn an_opaque_computes_clone_repoints_every_register_allocation_at_its_sole_clone() {
        let mut dsc = Cloning::default();
        let origin = DataOrigin::LabeledDs(LdsIdx(4));
        let internal = dsc.with_alloc(SenComponent::Pelrf, origin);
        let internal_clone = dsc.with_alloc(SenComponent::Sfplrf, origin);
        let in_out = dsc.with_alloc(SenComponent::Pelrf, DataOrigin::Constant(ConstIdx(0)));
        let in_out_clone = dsc.with_alloc(SenComponent::Sfplrf, DataOrigin::Constant(ConstIdx(1)));
        let node = NodeId(200);
        dsc.opaque.insert(node);
        let mut body = compute(SenComponent::Sfp, Vec::new(), Vec::new());
        body.instr_attribute.input_data_connects = vec![DataConnect::ArfPt];
        dsc.computes.insert(node, body.clone());

        let mut metadata = Metadata::default();
        // Both register allocations were themselves cloned exactly once, which is the `DT_CHECK`.
        metadata.node_cloning_map.insert(
            dsc.alloc_nodes[&internal],
            vec![dsc.alloc_nodes[&internal_clone]],
        );
        metadata.node_cloning_map.insert(
            dsc.alloc_nodes[&in_out],
            vec![dsc.alloc_nodes[&in_out_clone]],
        );
        metadata.opaque_ops.insert(
            node,
            OpaqueOp {
                in_out_reg_allocs: BTreeMap::from([(RegName::A00, in_out)]),
                internal_reg_alloc: Some(internal),
                ..OpaqueOp::default()
            },
        );

        let split = PeSfpComputeSplit::of(&metadata, node, &body).expect("cloneable");
        let clone =
            clone_compute_for_pe_sfp_work_split(&mut dsc, &mut metadata, split).expect("cloned");

        // The clone's own metadata entry names the CLONED registers, not the originals.
        let op = &metadata.opaque_ops[&clone];
        assert_eq!(op.internal_reg_alloc, Some(internal_clone));
        assert_eq!(
            op.in_out_reg_allocs,
            BTreeMap::from([(RegName::A00, in_out_clone)])
        );

        // The internal allocation is the clone's temp storage, and both registers gained it as user.
        assert_eq!(
            dsc.temp_storage,
            vec![(internal_clone, NodeName("c0_pe_parallel".to_string()))]
        );
        assert_eq!(
            dsc.users,
            vec![(internal_clone, clone), (in_out_clone, clone)]
        );
        assert_eq!(
            metadata.new_allocations[&DdcMemory::SfpLrf].comp_and_alloc_node,
            BTreeMap::from([(clone, internal_clone)])
        );

        // The instruction's own data connects took the suffix; no operand did, there being none.
        assert_eq!(dsc.minted, vec!["arf_pt_pe_parallel".to_string()]);
        assert_eq!(
            dsc.computes[&clone].instr_attribute.input_data_connects,
            vec![DataConnect::OuttensorToSfp]
        );
    }

    #[test]
    fn a_datastream_is_external_through_its_allocation_or_through_either_end_of_its_connect() {
        let mut dsc = Cloning::default();
        let lds = LdsIdx(5);
        let alloc = dsc.with_alloc(SenComponent::Pelrf, DataOrigin::LabeledDs(lds));
        dsc.census.insert(NodeIndex(0), NodeId(700));
        dsc.census.insert(NodeIndex(1), NodeId(701));

        let data = DataInfo {
            data_connect: Some(DataConnect::ArfPt),
            my_lds_idx: Some(lds),
            constant_id: None,
            latch_data_id: None,
        };
        let mut metadata = Metadata::default();
        let mut ends = Ends::default();
        ends.insert_producer(NodeIndex(0));
        ends.insert_consumer(NodeIndex(1));
        metadata.data_connects.insert(DataConnect::ArfPt, ends);

        // Nothing external yet.
        assert!(!storage_or_datastream_is_external(
            &dsc,
            &metadata,
            data,
            SenComponent::Pelrf,
            StreamDirection::Incoming
        ));

        // The connect's PRODUCER is external, which only the incoming question reaches.
        metadata.external_nodes.insert(NodeId(700));
        assert!(storage_or_datastream_is_external(
            &dsc,
            &metadata,
            data,
            SenComponent::Pelrf,
            StreamDirection::Incoming
        ));
        assert!(!storage_or_datastream_is_external(
            &dsc,
            &metadata,
            data,
            SenComponent::Pelrf,
            StreamDirection::Outgoing
        ));

        // The allocation in `storage` is external, which both questions reach.
        metadata.external_nodes.insert(dsc.alloc_nodes[&alloc]);
        assert!(storage_or_datastream_is_external(
            &dsc,
            &metadata,
            data,
            SenComponent::Pelrf,
            StreamDirection::Outgoing
        ));

        // ... and a DIFFERENT storage does not reach that allocation at all.
        assert!(!storage_or_datastream_is_external(
            &dsc,
            &metadata,
            data,
            SenComponent::Sfplrf,
            StreamDirection::Outgoing
        ));
    }

    /// THE SIGNAL ENDS ONE TREE NAMES — entry 121's `otherEndOfTheSignals_` resolved back to nodes.
    struct Signals(BTreeMap<NodeName, NodeId>);

    impl SyncNodesByName for Signals {
        fn node_named(&self, name: &NodeName) -> Option<NodeId> {
            self.0.get(name).copied()
        }
    }

    #[test]
    fn a_sync_is_related_to_external_nodes_through_the_other_end_of_any_one_signal() {
        let near = NodeName("sync_near".to_string());
        let far = NodeName("sync_far".to_string());
        let nodes = Signals(BTreeMap::from([
            (near.clone(), NodeId(1)),
            (far.clone(), NodeId(2)),
        ]));
        let node = SyncNode {
            name: NodeName("s0".to_string()),
            units: SyncUnits::new(SenComponent::Pe, []),
            direction: SyncDirection::Receive,
            strength: SyncStrength::Hard,
            implicit_sync_ref_transfer: None,
            other_ends: vec![near, far],
        };

        let mut metadata = Metadata::default();
        assert!(!sync_related_to_external_nodes(&nodes, &metadata, &node));

        // The SECOND end alone is external, and that is enough.
        metadata.external_nodes.insert(NodeId(2));
        assert!(sync_related_to_external_nodes(&nodes, &metadata, &node));
    }

    #[test]
    fn a_node_is_related_to_comps_through_the_first_of_its_components_the_set_holds() {
        let comps = [SenComponent::Sfp, SenComponent::Pelrf];

        let mut dsc = Cloning::default();
        let alloc = dsc.with_alloc(SenComponent::Pelrf, DataOrigin::LabeledDs(LdsIdx(1)));
        assert_eq!(
            is_node_related_to_comps(ComponentNode::Allocate(&dsc.allocs[&alloc]), &comps),
            Some(SenComponent::Pelrf)
        );

        let computing = compute(SenComponent::Pe, Vec::new(), Vec::new());
        assert_eq!(
            is_node_related_to_comps(ComponentNode::Compute(&computing), &comps),
            None
        );

        // A transfer's SOURCE is asked first, then each destination in turn.
        let transfer = TransferNode {
            name: NodeName("t0".to_string()),
            src: operand(SenComponent::Lx, None, None),
            dsts: Dsts::new(
                operand(SenComponent::L0, None, None),
                vec![operand(SenComponent::Sfp, None, None)],
            ),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
        };
        assert_eq!(
            is_node_related_to_comps(ComponentNode::Transfer(&transfer), &comps),
            Some(SenComponent::Sfp)
        );

        assert_eq!(is_node_related_to_comps(ComponentNode::Other, &comps), None);
    }

    /// ONE SUBTREE'S LDS SLOTS, stated as the lists entry 123's walk returns.
    #[derive(Default)]
    struct Subtree {
        slots: Vec<(TreeLdsSlot, Option<LdsIdx>)>,
        opaque: Vec<NodeId>,
        alloc_nodes: BTreeMap<AllocId, NodeId>,
        mem_org: BTreeMap<(LdsIdx, SenComponent), Option<AllocId>>,
        rewritten: Vec<(TreeLdsSlot, LdsIdx)>,
    }

    #[derive(Clone)]
    struct Entry;

    impl LabeledDsEntry for Entry {
        fn recorded_lds_idx(&self) -> LdsIdx {
            unimplemented!("no entry is read")
        }

        fn set_recorded_lds_idx(&mut self, _recorded: LdsIdx) {
            unimplemented!("no entry is written")
        }

        fn set_reference_lds_idx(&mut self, _reference: LdsIdx) {
            unimplemented!("no entry is written")
        }
    }

    impl NewLabeledDs for Subtree {
        type Entry = Entry;

        fn last_lds_pos(&self) -> LdsIdx {
            unimplemented!("no lds is inserted")
        }

        fn last_recorded_lds_idx(&self) -> LdsIdx {
            unimplemented!("no lds is inserted")
        }

        fn set_last_recorded_lds_idx(&mut self, _recorded: LdsIdx) {
            unimplemented!("no lds is inserted")
        }

        fn insert_lds_before_last(&mut self, _entry: Self::Entry) {
            unimplemented!("no lds is inserted")
        }

        fn clear_mem_org(&mut self, _pos: LdsIdx) {
            unimplemented!("entry 123 moves entries, it does not clear them")
        }

        fn mem_org_allocations(&self, _pos: LdsIdx) -> Vec<AllocId> {
            unimplemented!("the move is asked per allocation")
        }

        fn alloc_lds_idx(&self, _alloc: AllocId) -> Option<LdsIdx> {
            unimplemented!("the slot carries the index")
        }

        fn set_alloc_lds_idx(&mut self, _alloc: AllocId, _lds: LdsIdx) {
            unimplemented!("the allocate slot is what is rewritten")
        }

        fn tree_lds_slots(&self) -> Vec<(TreeLdsSlot, Option<LdsIdx>)> {
            unimplemented!("entry 123 walks ONE subtree")
        }

        fn set_tree_lds(&mut self, slot: TreeLdsSlot, lds: LdsIdx) {
            self.rewritten.push((slot, lds));
        }

        fn opaque_computes(&self) -> Vec<NodeId> {
            unimplemented!("entry 123 walks ONE subtree")
        }
    }

    impl AllocationsByNode for Subtree {
        fn allocation_of(&self, node: NodeId) -> Option<AllocId> {
            self.alloc_nodes
                .iter()
                .find(|&(_, &at)| at == node)
                .map(|(&alloc, _)| alloc)
        }
    }

    impl SubtreeLds for Subtree {
        fn tree_lds_slots_under(&self, _start: NodeId) -> Vec<(TreeLdsSlot, Option<LdsIdx>)> {
            self.slots.clone()
        }

        fn opaque_computes_under(&self, _start: NodeId) -> Vec<NodeId> {
            self.opaque.clone()
        }

        fn mem_org_storages_of(&self, lds: LdsIdx, alloc: AllocId) -> Vec<SenComponent> {
            self.mem_org
                .iter()
                .filter(|&(&(at, _), &held)| at == lds && held == Some(alloc))
                .map(|(&(_, storage), _)| storage)
                .collect()
        }

        fn move_mem_org(&mut self, from: LdsIdx, to: LdsIdx, storage: SenComponent) {
            let held = self.mem_org.insert((from, storage), None);
            self.mem_org.insert((to, storage), held.flatten());
        }
    }

    #[test]
    fn a_new_lds_index_reaches_every_operand_the_allocation_and_the_metadata_that_names_it() {
        // ⛔ THE DIVERGENCE UNDER TEST: `new_lds` is ABOVE `old_lds`, which is exactly the case the
        // reference's insert-while-iterating loses.
        let old = LdsIdx(1);
        let new = LdsIdx(5);
        let alloc = AllocId(7);
        let allocating = NodeId(10);
        let computing = NodeId(11);
        let transferring = NodeId(12);
        let untouched = TreeLdsSlot::Transfer(transferring, OperandPos::Output(0));
        let mut dsc = Subtree {
            slots: vec![
                (
                    TreeLdsSlot::Compute(computing, OperandPos::Input(1)),
                    Some(old),
                ),
                (untouched, Some(LdsIdx(9))),
                (TreeLdsSlot::Allocate(allocating), Some(old)),
            ],
            opaque: vec![computing],
            alloc_nodes: BTreeMap::from([(alloc, allocating)]),
            mem_org: BTreeMap::from([((old, SenComponent::Pelrf), Some(alloc))]),
            rewritten: Vec::new(),
        };

        let mut metadata = Metadata::default();
        metadata.opaque_ops.insert(
            computing,
            OpaqueOp {
                lds_idx: Some(old),
                ..OpaqueOp::default()
            },
        );
        metadata.new_allocations.insert(
            DdcMemory::PeLrf,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(old, alloc)]),
                ..Allocation::default()
            },
        );

        update_nodes_with_new_lds(&mut dsc, &mut metadata, new, old, NodeId(0));

        // Every slot HOLDING the old index was rewritten, and the one holding another was not.
        assert_eq!(
            dsc.rewritten,
            vec![
                (TreeLdsSlot::Compute(computing, OperandPos::Input(1)), new),
                (TreeLdsSlot::Allocate(allocating), new)
            ]
        );
        assert!(!dsc.rewritten.iter().any(|&(slot, _)| slot == untouched));

        // The opaque op's own index moved, and so did the allocation's `memOrg_` entry.
        assert_eq!(metadata.opaque_ops[&computing].lds_idx, Some(new));
        assert_eq!(dsc.mem_org[&(old, SenComponent::Pelrf)], None);
        assert_eq!(dsc.mem_org[&(new, SenComponent::Pelrf)], Some(alloc));

        // ⛔ AND THE REKEY SURVIVES: the new key is the only one left.
        assert_eq!(
            metadata.new_allocations[&DdcMemory::PeLrf].lds_idx_and_alloc_node,
            BTreeMap::from([(new, alloc)])
        );
    }
}

#[cfg(test)]
mod tests_e361 {
    // ⭐ TESTS FOR ENTRY 361. Union this module with this file's other test modules when they land.
    use super::*;

    use super::super::fold::{NodeKind, ScheduleTree};
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{InstrAttribute, NumChunks, ReplicationFactor, TransferPadding};
    use crate::units::NumFolds;
    use sys_arch_spec::arch_enums::SenComponent;

    /// THE SUBTREE AS THE WALK HANDS IT OVER, with the ONE datastream this stand-in calls external.
    struct Subtree {
        nodes: Vec<SubtreeNode>,
        external: Option<(Option<DataConnect>, SenComponent, StreamDirection)>,
    }

    impl ScheduleTree for Subtree {
        fn kind(&self, _node: NodeId) -> NodeKind {
            NodeKind::Block
        }

        fn parent(&self, _node: NodeId) -> Option<NodeId> {
            None
        }

        fn children(&self, _block: BlockId) -> Vec<NodeId> {
            Vec::new()
        }
    }

    impl ExternalStreams for Subtree {
        fn storage_or_datastream_is_external(
            &self,
            data: DataInfo,
            storage: SenComponent,
            direction: StreamDirection,
        ) -> bool {
            self.external == Some((data.data_connect, storage, direction))
        }
    }

    impl ExternalSubtree for Subtree {
        fn subtree_nodes(&self, _root: BlockId) -> Vec<SubtreeNode> {
            self.nodes.clone()
        }
    }

    fn operand(connect: DataConnect, storage: SenComponent) -> Operand {
        Operand {
            unit: SenComponent::Pe,
            storage,
            data: DataInfo {
                data_connect: Some(connect),
                my_lds_idx: Some(LdsIdx(1)),
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    fn compute(node: NodeId, connect: DataConnect) -> SubtreeNode {
        SubtreeNode::Compute(
            node,
            ComputeNode {
                name: NodeName("c0".to_string()),
                op: DdlComputeType::Macc,
                ex_unit: SenComponent::Pe,
                inputs: vec![operand(connect, SenComponent::Hbm)],
                outputs: Vec::new(),
                num_folds_engaged: NumFolds::ONE,
                data_format: None,
                instr_attribute: InstrAttribute::default(),
            },
        )
    }

    fn transfer(node: NodeId, connect: DataConnect) -> SubtreeNode {
        SubtreeNode::Transfer(
            node,
            TransferNode {
                name: NodeName("t0".to_string()),
                src: operand(connect, SenComponent::L0),
                dsts: Dsts::new(operand(DataConnect::ArfPt, SenComponent::Lx), Vec::new()),
                replication_factor: ReplicationFactor::ONE,
                unit_time_transfer_chunk_size: Vec::new(),
                unit_time_transfer_num_chunks: NumChunks::ONE,
                padding: TransferPadding::default(),
                src_indirect: None,
                dst_indirect: None,
                core_id_to_gtr_info: BTreeMap::new(),
                transfer_size: BTreeMap::new(),
            },
        )
    }

    /// A block whose three node kinds are each the one that answers, and the same block with
    /// NOTHING external under it.
    #[test]
    fn a_block_is_related_through_an_external_node_a_computes_input_or_a_transfers_source() {
        let root = BlockId::of(
            &Subtree {
                nodes: Vec::new(),
                external: None,
            },
            NodeId(0),
        )
        .expect("the stand-in calls every node a block");
        let nodes = vec![
            SubtreeNode::Allocate(NodeId(1)),
            compute(NodeId(2), DataConnect::ArfPtsum),
            transfer(NodeId(3), DataConnect::OuttensorToSfp),
        ];
        let clean = Metadata::default();

        // The ALLOCATE arm: nothing but `isExternalNode` reaches it.
        let mut external_node = Metadata::default();
        external_node.external_nodes.insert(NodeId(1));
        let tree = Subtree {
            nodes: nodes.clone(),
            external: None,
        };
        assert!(subtree_related_to_external_nodes(
            &tree,
            &external_node,
            root
        ));
        assert!(!subtree_related_to_external_nodes(&tree, &clean, root));

        // The COMPUTE arm, through entry 306: its input's own component as the storage.
        let by_compute = Subtree {
            nodes: nodes.clone(),
            external: Some((
                Some(DataConnect::ArfPtsum),
                SenComponent::Pe,
                StreamDirection::Incoming,
            )),
        };
        assert!(subtree_related_to_external_nodes(&by_compute, &clean, root));

        // The TRANSFER arm, through entry 341: the source's STORAGE, incoming.
        let by_transfer = Subtree {
            nodes,
            external: Some((
                Some(DataConnect::OuttensorToSfp),
                SenComponent::L0,
                StreamDirection::Incoming,
            )),
        };
        assert!(subtree_related_to_external_nodes(&by_transfer, &clean, root));
    }
}
