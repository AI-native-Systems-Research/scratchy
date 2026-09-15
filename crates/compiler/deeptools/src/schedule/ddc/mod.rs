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

//! `ddc/ddc.h` — 9 of the campaign's 382 units (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e073_addPropInfo` | 073 | 0 | 22 | `CoordPropTracker` | `ddc/ddc.h:402` |
//! | `e074_retry` | 074 | 0 | 23 | `CoordPropTracker` | `ddc/ddc.h:436` |
//! | `e075_getCurrItem` | 075 | 0 | 10 | `CoordPropTracker` | `ddc/ddc.h:463` |
//! | `e076_print` | 076 | 0 | 29 | `RowGroupInfo` | `ddc/ddc.h:580` |
//! | `e077_printFoldParams` | 077 | 0 | 6 | `Ddc` | `ddc/ddc.h:611` |
//! | `e230_addPropInfo` | 230 | 1 | 4 | `CoordPropTracker` | `ddc/ddc.h:430` |
//! | `e231_rollbackToPos` | 231 | 1 | 36 | `CoordPropTracker` | `ddc/ddc.h:474` |
//! | `e232_reset` | 232 | 1 | 5 | `CoordPropTracker` | `ddc/ddc.h:528` |
//! | `e296_rollBackNodesInBlock` | 296 | 2 | 15 | `CoordPropTracker` | `ddc/ddc.h:511` |

pub mod fold;
pub mod metadata;
pub mod shuffle;
pub mod transformation;
/// ⭐ PUBLIC BECAUSE THE `l3` VIEW'S OWN PUBLIC FIELDS NAME IT — `l3::dsc::NamedDims::name` is a
/// `StageName` and `EmptyStage::name` another, so a caller outside this crate that builds a data
/// stage has to be able to name the type. It was `pub(crate)`, which made those fields unnameable.
pub mod transformation_util;
pub mod v1;

// ⭐ USES FOR ENTRIES 073-077 AND 230-232. Union these into this file's top block when its other
// entries land.
use core::fmt::Write as _;
use std::collections::BTreeMap;

use self::fold::{
    Beta, BlockId, CoordPropInfo, FoldLabel, FoldParamInfo, NodeId, NodeKind, ScheduleTree,
};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
use crate::units::Row;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE COORDINATE-PROPAGATION WORKLIST AND THE ROW GROUP — as entries 073-077 read them.
//
// ⭐ ONE C++ STRUCT, ONE RUST TYPE. `dsc2::CoordPropInfoType` (`dsc/dsc2.h:1087`) already has a
// reduction in this module's `fold` sibling — the three fields entry 091's match reads,
// `fold::CoordPropInfo`. [`Propagation`] CONTAINS that reduction rather than respelling it, so no
// field is written twice and `&queued.prop.ends` goes straight into `match_data_stream`.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// `dsc2::ScheduleNode::name_` — all entries 074 and 076 ask of a node beyond its identity, and
/// both ask only to say which node a line of text is about.
pub trait NodeNames {
    /// The node's name.
    fn name(&self, node: NodeId) -> &str;
}

/// THE COORDINATE STORES A ROLLBACK EMPTIES — one method per `nodeType_` arm of `rollbackToPos`
/// (`ddc/ddc.h:487-506`), so no arm can clear a store belonging to another kind of node. Which arm a
/// node takes is [`ScheduleTree::kind`], and the reference clears nothing for the other five kinds.
pub trait NodeCoordinates {
    /// `AllocateNode::allocateCoordinates_.clear()` AND `sliceViewCoordinates_.clear()` — both
    /// (`:491-492`).
    fn clear_allocate_coordinates(&mut self, node: NodeId);
    /// `TransferNode::transferCoordinates_.clear()` (`:497`).
    fn clear_transfer_coordinates(&mut self, node: NodeId);
    /// `ComputeNode::outputCoordinate_.clear()` AND EVERY `inputCoordinates_` entry (`:502-505`).
    fn clear_compute_coordinates(&mut self, node: NodeId);
}

/// WHICH SIDE OF THE DATAFLOW THE REFERENCE NODE SITS ON — `CoordPropInfoType::refIsProducer`
/// (`dsc/dsc2.h:1092`).
///
/// ⛔ AN ENUM, NOT A `bool`, BECAUSE `scaleDown` IS THE FIELD BESIDE IT: the reference initialises
/// both positionally in one aggregate (`ddc/ddc.h:424-427`) and two `bool`s in a row transpose
/// silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RefRole {
    /// `true` — the reference node produces what the node to fold consumes.
    #[default]
    Producer,
    /// `false`.
    Consumer,
}

/// WHETHER THIS STEP CROSSES FROM A VALUE TENSOR TO ITS MX SCALE TENSOR — `scaleDown`
/// (`dsc/dsc2.h:1095`), the flag that makes `buildFoldForAllocation` compress the reference
/// coordinate through `scaleDownCoord` (`ddc/ddc_fold.cpp:2402`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ScaleDown {
    /// `false`.
    #[default]
    No,
    /// `true` — set only where a compute's value-tensor allocation drags its scale tensor into the
    /// queue (`ddc/ddc_fold.cpp:1746`).
    Yes,
}

/// HOW FAR ONE QUEUED PROPAGATION GOT — `CoordPropInfoType::PropStateType` (`dsc/dsc2.h:1088`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum PropState {
    /// `NOT_PROCESSED`.
    #[default]
    NotProcessed,
    /// `ROLLED_BACK`.
    RolledBack,
    /// `OVERRIDDEN`.
    Overridden,
    /// `COMPLETE`.
    Complete,
}

/// ONE PROPAGATION STEP'S FIXED PART — `dsc2::CoordPropInfoType` (`dsc/dsc2.h:1087`) less the two
/// fields the queue itself owns (`propState`, `dimsToPropagate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Propagation {
    /// `refNode`, `nodeToFold` and `dataConnect` as entry 091's match reads them.
    pub ends: CoordPropInfo,
    /// `refNode`'s SCHEDULE-TREE identity — `refsAdded_`'s inner key (`ddc/ddc.h:541-544`). An allocate
    /// end also carries an `AllocId` inside [`Propagation::ends`]; that is the allocation's
    /// identity and this is the node's.
    pub ref_node: NodeId,
    /// `nodeToFold`'s tree identity — `refsAdded_`'s OUTER key.
    pub node_to_fold: NodeId,
    /// `refIsProducer`.
    pub ref_role: RefRole,
    /// `scaleDown`.
    pub scale_down: ScaleDown,
}

/// ONE ENTRY OF `itemsToProcess_` (`ddc/ddc.h:537`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedProp {
    /// The step.
    pub prop: Propagation,
    /// `propState`.
    pub state: PropState,
    /// `dimsToPropagate` — the dims this entry has left to propagate, which is the UNSEEN subset of
    /// what was asked for and not the caller's whole list.
    pub dims: Vec<PrimaryDim>,
}

/// A POSITION IN THE PROPAGATION QUEUE — an `itemsToProcess_` index, and what `rollbackToPos` takes.
///
/// ⛔ THE REFERENCE'S `currItemToProcess_ = -1` IS NOT A POSITION: it means *before the first
/// entry*, so it is [`None`] here and there is no `-1` to index the queue with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueuePos(pub usize);

/// HOW MANY TIMES ONE `(nodeToFold, refNode, dim)` PROPAGATION HAS BEEN RETRIED — `refsAdded_`'s
/// innermost value (`ddc/ddc.h:540-544`), bounded by the reference's own threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RetryCount(u8);

impl RetryCount {
    /// What a first retry records — the reference stores 0 and not 1 (`ddc/ddc.h:418,457`), so a
    /// dim's count is the number of retries BEFORE the one being charged.
    const FIRST: RetryCount = RetryCount(0);

    /// The count the reference aborts on: `retryCount == 15` (`ddc/ddc.h:448`).
    const THRESHOLD: RetryCount = RetryCount(15);

    /// The next count, or [`None`] at the threshold.
    const fn bumped(self) -> Option<RetryCount> {
        if self.0 >= Self::THRESHOLD.0 {
            None
        } else {
            Some(RetryCount(self.0 + 1))
        }
    }
}

/// THE COORDINATE-PROPAGATION WORKLIST — `Ddc::CoordPropTracker` (`ddc/ddc.h:400`).
///
/// ⭐ A `Vec` FOR THE REFERENCE'S `std::deque`: nothing ever pops the front. Entries are appended,
/// walked forward by [`CoordPropTracker::next_item`] and rewound by `rollbackToPos`, so the one
/// operation that would need a deque is never performed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoordPropTracker {
    /// `itemsToProcess_`.
    items: Vec<QueuedProp>,
    /// `refsAdded_` — outer key `nodeToFold`, inner key `refNode`, innermost value the dim's retry
    /// count.
    refs_added: BTreeMap<NodeId, BTreeMap<NodeId, BTreeMap<PrimaryDim, RetryCount>>>,
    /// `currItemToProcess_`.
    curr: Option<QueuePos>,
}

impl CoordPropTracker {
    /// Replaces: e073_addPropInfo
    ///
    /// Queues one propagation step for every dim of `dims` NOT already recorded against this
    /// `(nodeToFold, refNode)` pair, and records each of those as retried zero times.
    ///
    /// ⛔ NOTHING IS QUEUED WHEN EVERY DIM WAS ALREADY SEEN (`ddc/ddc.h:420-423`): the ledger, not
    /// the queue, is what stops one propagation from being walked twice.
    /// ⛔ THE QUEUED ENTRY CARRIES THE UNSEEN SUBSET, not the list it was asked for.
    /// ⭐ A DIM REPEATED IN `dims` IS TAKEN ONCE — the ledger is written inside the loop, so the
    /// second sighting is already seen.
    pub fn add_prop_info(&mut self, prop: Propagation, dims: &[PrimaryDim]) {
        let mut unseen: Vec<PrimaryDim> = Vec::new();
        for &dim in dims {
            let seen = self
                .refs_added
                .get(&prop.node_to_fold)
                .and_then(|per_ref| per_ref.get(&prop.ref_node))
                .is_some_and(|per_dim| per_dim.contains_key(&dim));
            if seen {
                // This propagation step has already been included.
                continue;
            }
            unseen.push(dim);
            self.refs_added
                .entry(prop.node_to_fold)
                .or_default()
                .entry(prop.ref_node)
                .or_default()
                .insert(dim, RetryCount::FIRST);
        }
        if unseen.is_empty() {
            // No remaining dimension for propagation.
            return;
        }
        self.items.push(QueuedProp {
            prop,
            state: PropState::NotProcessed,
            dims: unseen,
        });
    }

    /// Replaces: e074_retry
    ///
    /// Re-queues a step for `dims` and charges each of those dims one retry.
    ///
    /// ⛔⛔ THE RE-QUEUED STEP LOSES `scaleDown`. The reference's aggregate initialiser stops at
    /// `dimsToPropagate` (`ddc/ddc.h:438-441`), so the seventh member falls back to its `= false`
    /// default and a value-to-scale-tensor step comes back as an ordinary one. Reproduced, and it is
    /// the one field this function overwrites.
    /// ⛔ A DIM WITH NO LEDGER ENTRY IS CHARGED ZERO (`:443-455`), so a fresh dim's first retry does
    /// not count against the threshold.
    /// 🛑 THE 16TH RETRY OF ONE DIM IS THE REFERENCE'S `DT_ERROR` (`:448-453`) and it ends the run. No
    /// type can state "the fold builder kept failing", so this one stays a stop.
    pub fn retry<N: NodeNames + ?Sized>(
        &mut self,
        names: &N,
        prop: Propagation,
        dims: &[PrimaryDim],
    ) {
        self.items.push(QueuedProp {
            prop: Propagation {
                scale_down: ScaleDown::No,
                ..prop
            },
            state: PropState::NotProcessed,
            dims: dims.to_vec(),
        });
        for &dim in dims {
            let per_ref = self.refs_added.entry(prop.node_to_fold).or_default();
            let charged = match per_ref
                .get(&prop.ref_node)
                .and_then(|per_dim| per_dim.get(&dim))
            {
                None => RetryCount::FIRST,
                Some(&count) => count.bumped().unwrap_or_else(|| {
                    // `primaryDimToString` (`dsc/dims.cpp:22`) is this enum's name lowered, for all
                    // twelve dims, so the message is the reference's.
                    panic!(
                        "Retry threshold for propagation reached for {} -> {}, dim= {}.",
                        names.name(prop.ref_node),
                        names.name(prop.node_to_fold),
                        format!("{dim:?}").to_lowercase(),
                    )
                }),
            };
            per_ref
                .entry(prop.ref_node)
                .or_default()
                .insert(dim, charged);
        }
    }

    /// Replaces: e075_getCurrItem
    ///
    /// Advances to the next queued step, stamps it COMPLETE in the queue and hands back a copy.
    ///
    /// ⛔⛔ IT ADVANCES BEFORE IT BOUNDS-CHECKS (`ddc/ddc.h:464-467`), so answering [`None`] still
    /// leaves the cursor one past the last entry — the value `rollbackToPos` compares against.
    /// ⛔ COMPLETE IS STAMPED ON FETCH, not on success: the entry is assumed processed "irrespective
    /// of the usage on the caller side" (`:461-462`), so the copy handed back is COMPLETE too.
    pub fn next_item(&mut self) -> Option<QueuedProp> {
        let next = QueuePos(self.curr.map_or(0, |QueuePos(pos)| pos + 1));
        self.curr = Some(next);
        let item = self.items.get_mut(next.0)?;
        item.state = PropState::Complete;
        Some(item.clone())
    }

    /// Replaces: e230_addPropInfo
    ///
    /// Queues a step read off a record already in hand, for the CALLER'S dims.
    ///
    /// ⛔ THE RECORD'S OWN `dimsToPropagate` AND `propState` ARE DROPPED (`ddc/ddc.h:430-434`): the
    /// five fixed fields are forwarded and `dims` alone decides what gets propagated.
    /// ⭐ THE RECORD NEED NOT BE QUEUED — the reference's callers build one on the stack and mutate
    /// `nodeToFold` and `scaleDown` between two calls (`ddc/ddc_fold.cpp:1717-1750`).
    pub fn add_prop_info_from(&mut self, item: &QueuedProp, dims: &[PrimaryDim]) {
        self.add_prop_info(item.prop, dims);
    }

    /// Replaces: e231_rollbackToPos
    ///
    /// Rewinds the queue to `new_pos`, marking every entry from there through the cursor ROLLED_BACK
    /// and emptying the coordinates their fold nodes had computed.
    ///
    /// ⛔ A NO-OP UNTIL THE CURSOR REACHES `new_pos` (`ddc/ddc.h:475-477`), and it leaves the cursor
    /// one BEFORE `new_pos`, so the entry at `new_pos` is handed out again.
    /// ⛔ BOTH CONDITIONS OF THE `DT_ERROR` (`:478-482`) ARE UNREACHABLE: [`QueuePos`] cannot spell a
    /// negative position, and `newPos > currItemToProcess_` is what the guard above already returns on.
    /// ⚠️ DIVERGENCE, AND IT IS A REFERENCE DEFECT: `:485` stamps `at(currItemToProcess_)` inside a
    /// loop that clears `[i]`, so only the LAST entry is marked, once per step — and `at()` throws
    /// outright when [`Self::next_item`] has already run off the end. `i` is stamped here, over the
    /// entries that exist. `propState` is written and never read anywhere in the reference tree.
    pub fn rollback_to_pos<T: ScheduleTree + NodeCoordinates + ?Sized>(
        &mut self,
        nodes: &mut T,
        new_pos: QueuePos,
    ) {
        let Some(curr) = self.curr else {
            return;
        };
        if curr < new_pos {
            return;
        }
        // Clear the computed coordinates up to the rollback position.
        for item in self.items.iter_mut().take(curr.0 + 1).skip(new_pos.0) {
            item.state = PropState::RolledBack;
            let node = item.prop.node_to_fold;
            match nodes.kind(node) {
                NodeKind::Allocate => nodes.clear_allocate_coordinates(node),
                NodeKind::Transfer => nodes.clear_transfer_coordinates(node),
                NodeKind::Compute => nodes.clear_compute_coordinates(node),
                NodeKind::Block
                | NodeKind::Loop
                | NodeKind::Sync
                | NodeKind::Condition
                | NodeKind::StickMask => {}
            }
        }
        self.curr = new_pos.0.checked_sub(1).map(QueuePos);
    }

    /// Replaces: e232_reset
    ///
    /// Drops the queue and the retry ledger and puts the cursor back before the first entry.
    ///
    /// ⭐ THE LEDGER GOES TOO, so a propagation dropped here may be queued again — that is what
    /// makes `reset` different from rolling back to position 0.
    pub fn reset(&mut self) {
        self.items.clear();
        self.refs_added.clear();
        self.curr = None;
    }

    /// Replaces: e296_rollBackNodesInBlock
    ///
    /// Rewinds to the first queued step before the cursor whose reference or fold node sits in
    /// `block`'s subtree, and does nothing when none does.
    ///
    /// ⛔ THE CURSOR'S OWN ENTRY IS NOT SCANNED (`ddc/ddc.h:519`): `i < currItemToProcess_`.
    /// ⭐ MEMBERSHIP IS A PARENT WALK over the three kinds the reference's DFS filter admits —
    /// ALLOCATE, COMPUTE and TRANSFER — since a subtree holds loops and conditions too.
    pub fn roll_back_nodes_in_block<T: ScheduleTree + NodeCoordinates + ?Sized>(
        &mut self,
        nodes: &mut T,
        block: BlockId,
    ) {
        let Some(QueuePos(curr)) = self.curr else {
            return;
        };
        let in_block = |node: NodeId| {
            matches!(
                nodes.kind(node),
                NodeKind::Allocate | NodeKind::Compute | NodeKind::Transfer
            ) && core::iter::successors(nodes.parent(node), |&walked| nodes.parent(walked))
                .any(|walked| walked == block.node())
        };
        // `take`, not a slice: `next_item` can leave the cursor one past the last entry.
        let hit = self
            .items
            .iter()
            .take(curr)
            .position(|item| in_block(item.prop.ref_node) || in_block(item.prop.node_to_fold));
        if let Some(pos) = hit {
            self.rollback_to_pos(nodes, QueuePos(pos));
        }
    }
}

/// WHICH ROW-BUNDLING CASE A ROW GROUP IS — `Ddc::RowGroupInfo::Category` (`ddc/ddc.h:556`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RowBundling {
    /// `ROW_TO_SAME_ROW`.
    RowToSameRow,
    /// `NONROW_TO_ROW`.
    NonRowToRow,
    /// `ROW_TO_NONROW`.
    RowToNonRow,
    /// `ROW_NORTH_SOUTH`.
    RowNorthSouth,
    /// `NO_BUNDLING` — the field's own default (`ddc/ddc.h:562`).
    #[default]
    NoBundling,
}

impl RowBundling {
    /// The reference's spelling (`ddc/ddc.h:585-597`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::RowToNonRow => "Row-to-NonRow",
            Self::RowToSameRow => "Row-to-SameRow",
            Self::NonRowToRow => "NonRow-to-Row",
            Self::RowNorthSouth => "Row-North-South",
            Self::NoBundling => "No-Bundling",
        }
    }
}

/// WHETHER A GROUP'S ROWS RUN UP OR DOWN — `RowGroupInfo::ascendingOrder` (`ddc/ddc.h:578`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RowOrder {
    /// `true`, the field's default.
    #[default]
    Ascending,
    /// `false`.
    Descending,
}

/// ONE MEMBER OF A ROW GROUP — `RowGroupInfo::RowGroupNodeInfo` (`ddc/ddc.h:566`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGroupNodeInfo {
    /// `node`.
    pub node: NodeId,
    /// `row` — the PT row `getCompRowId` reported, absent for its `-1`.
    pub row: Option<Row>,
    /// `beta` — WHICH of the node's row betas this grouping uses, since a compute node has several
    /// (`ddc/ddc.h:569-572`).
    pub beta: Beta,
}

/// ONE ROW-SPLIT GROUP — `Ddc::RowGroupInfo` (`ddc/ddc.h:555`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RowGroupInfo {
    /// `cat`.
    pub cat: RowBundling,
    /// `commonGroupAncestor` — the innermost block node containing every node of the group, absent
    /// for the reference's `nullptr`.
    pub common_group_ancestor: Option<BlockId>,
    /// `nodeInfo`.
    pub node_info: Vec<RowGroupNodeInfo>,
    /// `activeRow` — the group's one row, carried only for `ROW_TO_SAME_ROW`, absent for the
    /// reference's `-1`.
    pub active_row: Option<Row>,
    /// `ascendingOrder`.
    pub ascending_order: RowOrder,
}

impl RowGroupInfo {
    /// Replaces: e076_print
    ///
    /// Writes the group out: its category, then one `(name, row, beta)` triple per member.
    ///
    /// ⛔ AN UNSET ROW PRINTS `-1` — the reference prints the `int` field whose default is `-1`
    /// (`ddc/ddc.h:568`), so absence is that number and not a blank.
    /// ⭐ `", row= "` HAS A SPACE AFTER THE `=` AND `", beta="` HAS NONE (`:604-605`).
    /// ⚠️ The reference's closing `out.flush()` has no `fmt::Write` counterpart: the sink is the
    /// caller's and so is flushing it.
    pub fn print<N: NodeNames + ?Sized>(&self, names: &N, out: &mut dyn core::fmt::Write) {
        let _ = write!(out, "\nRowgroup: \n  Category= {}", self.cat.spelling());
        let _ = write!(out, "\n  Group elements:");
        for member in &self.node_info {
            let _ = write!(
                out,
                " ({}, row= {}, beta={})",
                names.name(member.node),
                member.row.map_or(-1, |row| i64::from(row.get())),
                member.beta.0,
            );
        }
    }
}

/// Replaces: e077_printFoldParams
///
/// The fold list as one `(alpha, beta, cardinality, label) ` group per level, outermost first.
///
/// ⭐ THE SINK IS THE CALLER'S: the reference writes to `std::cout` (`ddc/ddc.h:613`) and the text is
/// the whole of what this function decides.
/// ⛔ AN UNLABELLED LEVEL PRINTS AN EMPTY FIELD — `foldDimLabel`'s default is `""`
/// (`dsc/dsc2.h:1084`), so the `, )` that leaves is the reference's own.
#[must_use]
pub fn print_fold_params(fold_params: &[FoldParamInfo]) -> String {
    let mut out = String::new();
    for fp_info in fold_params {
        let _ = write!(
            out,
            "({}, {}, {}, {}) ",
            fp_info.alpha.0,
            fp_info.beta.0,
            fp_info.cardinality.0,
            fp_info.label.map_or("", FoldLabel::spelling),
        );
    }
    out
}

// ⭐ TESTS FOR ENTRIES 073-077. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e073_e077 {
    use super::fold::{
        Alpha, Beta, Cardinality, CoordPropInfo, FoldLabel, FoldParamInfo, NodeId, PropEnd,
    };
    use super::{
        CoordPropTracker, NodeNames, PropState, Propagation, QueuePos, RefRole, RetryCount,
        RowBundling, RowGroupInfo, RowGroupNodeInfo, RowOrder, ScaleDown, print_fold_params,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
    use crate::units::Row;

    /// Node names read out of a table, which is all the two text-producing entries ask for.
    struct Names(Vec<&'static str>);

    impl NodeNames for Names {
        fn name(&self, node: NodeId) -> &str {
            self.0[node.0 as usize]
        }
    }

    /// A step between two nodes that are neither an allocation nor a compute, which is every end
    /// entries 073-075 need: the tracker keys on identity alone.
    fn step(ref_node: u32, node_to_fold: u32) -> Propagation {
        Propagation {
            ends: CoordPropInfo {
                data_connect: None,
                ref_node: PropEnd::Other,
                node_to_fold: PropEnd::Other,
            },
            ref_node: NodeId(ref_node),
            node_to_fold: NodeId(node_to_fold),
            ref_role: RefRole::Producer,
            scale_down: ScaleDown::Yes,
        }
    }

    /// e073: a dim already recorded for the pair is dropped, and a step whose every dim was seen
    /// queues nothing at all — the ledger is the guard, not the queue.
    #[test]
    fn an_already_propagated_dim_is_dropped_and_an_all_seen_step_queues_nothing() {
        let mut tracker = CoordPropTracker::default();

        tracker.add_prop_info(step(0, 1), &[PrimaryDim::In, PrimaryDim::Out]);
        // Same pair, one old dim and one new: only the new one is queued.
        tracker.add_prop_info(step(0, 1), &[PrimaryDim::Out, PrimaryDim::Mb]);
        // Same pair, nothing new: no entry.
        tracker.add_prop_info(step(0, 1), &[PrimaryDim::In, PrimaryDim::Mb]);
        // The other direction is a different pair, so its dims are unseen.
        tracker.add_prop_info(step(1, 0), &[PrimaryDim::In]);

        assert_eq!(tracker.items.len(), 3);
        assert_eq!(tracker.items[0].dims, vec![PrimaryDim::In, PrimaryDim::Out]);
        assert_eq!(tracker.items[1].dims, vec![PrimaryDim::Mb]);
        assert_eq!(tracker.items[2].dims, vec![PrimaryDim::In]);
        assert_eq!(tracker.items[2].prop.node_to_fold, NodeId(0));
        assert_eq!(
            tracker.refs_added[&NodeId(1)][&NodeId(0)][&PrimaryDim::Mb],
            RetryCount::FIRST
        );
    }

    /// e074: the re-queued step comes back with `scaleDown` cleared — the reference's aggregate
    /// initialiser never copies it — and the dim is charged 0 then 1.
    #[test]
    fn a_retry_requeues_without_the_scale_down_flag_and_charges_the_dim() {
        let mut tracker = CoordPropTracker::default();
        let names = Names(vec!["refNode", "nodeToFold"]);
        let prop = step(0, 1);

        tracker.retry(&names, prop, &[PrimaryDim::In]);

        assert_eq!(prop.scale_down, ScaleDown::Yes);
        assert_eq!(tracker.items[0].prop.scale_down, ScaleDown::No);
        assert_eq!(tracker.items[0].state, PropState::NotProcessed);
        // A dim with no ledger entry is charged zero, so the first retry is free.
        assert_eq!(
            tracker.refs_added[&NodeId(1)][&NodeId(0)][&PrimaryDim::In],
            RetryCount(0)
        );

        tracker.retry(&names, prop, &[PrimaryDim::In]);

        assert_eq!(tracker.items.len(), 2);
        assert_eq!(
            tracker.refs_added[&NodeId(1)][&NodeId(0)][&PrimaryDim::In],
            RetryCount(1)
        );
    }

    /// e074, the negative: the reference aborts once one dim of one pair has been charged 15 times
    /// (`ddc/ddc.h:448`), and the message names both ends.
    #[test]
    #[should_panic(expected = "Retry threshold for propagation reached for refNode -> nodeToFold")]
    fn retrying_one_dim_past_the_threshold_is_the_references_own_stop() {
        let mut tracker = CoordPropTracker::default();
        let names = Names(vec!["refNode", "nodeToFold"]);

        // The first charges 0 and each of the next 15 charges one more, so the 17th is the stop.
        for _ in 0..=RetryCount::THRESHOLD.0 {
            tracker.retry(&names, step(0, 1), &[PrimaryDim::Y]);
        }
        assert_eq!(
            tracker.refs_added[&NodeId(1)][&NodeId(0)][&PrimaryDim::Y],
            RetryCount::THRESHOLD
        );

        tracker.retry(&names, step(0, 1), &[PrimaryDim::Y]);
    }

    /// e075: the fetch stamps COMPLETE on the queue entry as well as the copy, and the cursor
    /// advances past the end even when there is nothing to hand back.
    #[test]
    fn a_fetch_stamps_complete_and_the_cursor_advances_past_the_end() {
        let mut tracker = CoordPropTracker::default();
        tracker.add_prop_info(step(0, 1), &[PrimaryDim::In]);

        let fetched = tracker.next_item().expect("one entry was queued");

        assert_eq!(fetched.state, PropState::Complete);
        assert_eq!(tracker.items[0].state, PropState::Complete);
        assert_eq!(tracker.curr, Some(QueuePos(0)));

        assert_eq!(tracker.next_item(), None);
        // One past the last entry, which is what a rollback position is compared against.
        assert_eq!(tracker.curr, Some(QueuePos(1)));
    }

    /// e076: the category, then one triple per member — and a member with no row prints the
    /// reference's `-1` rather than nothing.
    #[test]
    fn a_row_group_prints_its_category_and_a_minus_one_for_an_unset_row() {
        let names = Names(vec!["transfer.0", "compute.1"]);
        let group = RowGroupInfo {
            cat: RowBundling::RowToSameRow,
            node_info: vec![
                RowGroupNodeInfo {
                    node: NodeId(0),
                    row: Row::checked(3),
                    beta: Beta(128),
                },
                RowGroupNodeInfo {
                    node: NodeId(1),
                    row: None,
                    beta: Beta(-1),
                },
            ],
            ..RowGroupInfo::default()
        };

        let mut out = String::new();
        group.print(&names, &mut out);

        assert_eq!(
            out,
            "\nRowgroup: \n  Category= Row-to-SameRow\n  Group elements: (transfer.0, row= 3, \
             beta=128) (compute.1, row= -1, beta=-1)"
        );
        assert_eq!(group.ascending_order, RowOrder::Ascending);
    }

    /// e077: one parenthesised group per level with a trailing space, and an unlabelled level leaves
    /// the label field empty.
    #[test]
    fn the_fold_param_line_carries_one_group_per_level_and_an_empty_label() {
        let params = vec![
            FoldParamInfo {
                alpha: Alpha(64),
                beta: Beta(0),
                cardinality: Cardinality(32),
                label: Some(FoldLabel::CoreWorksliceFoldDim),
            },
            FoldParamInfo {
                alpha: Alpha(-8),
                beta: Beta(4),
                cardinality: Cardinality(2),
                label: None,
            },
        ];

        assert_eq!(
            print_fold_params(&params),
            "(64, 0, 32, core_workslice_fold_dim) (-8, 4, 2, ) "
        );
    }
}

// ⭐ TESTS FOR ENTRIES 230-232. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e230_e232 {
    use super::fold::{BlockId, CoordPropInfo, NodeId, NodeKind, PropEnd, ScheduleTree};
    use super::{
        CoordPropTracker, NodeCoordinates, PropState, Propagation, QueuePos, QueuedProp, RefRole,
        ScaleDown,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;

    /// A step between two nodes, keyed on identity alone — the ends' own kinds are the TREE's answer
    /// and not this record's, since [`PropEnd`] has no `TRANSFER` arm to carry one.
    fn step(ref_node: u32, node_to_fold: u32) -> Propagation {
        Propagation {
            ends: CoordPropInfo {
                data_connect: None,
                ref_node: PropEnd::Other,
                node_to_fold: PropEnd::Other,
            },
            ref_node: NodeId(ref_node),
            node_to_fold: NodeId(node_to_fold),
            ref_role: RefRole::Producer,
            scale_down: ScaleDown::Yes,
        }
    }

    /// A TREE OF KINDS, RECORDING WHICH STORE WAS EMPTIED FOR WHICH NODE — the stores themselves are
    /// `dsc2` node fields and the mechanism for reaching them is not this unit's.
    struct Nodes {
        kinds: Vec<NodeKind>,
        cleared: Vec<(NodeId, NodeKind)>,
    }

    impl ScheduleTree for Nodes {
        fn kind(&self, node: NodeId) -> NodeKind {
            self.kinds[node.0 as usize]
        }
        fn parent(&self, _node: NodeId) -> Option<NodeId> {
            None
        }
        fn children(&self, _block: BlockId) -> Vec<NodeId> {
            Vec::new()
        }
    }

    impl NodeCoordinates for Nodes {
        fn clear_allocate_coordinates(&mut self, node: NodeId) {
            self.cleared.push((node, NodeKind::Allocate));
        }
        fn clear_transfer_coordinates(&mut self, node: NodeId) {
            self.cleared.push((node, NodeKind::Transfer));
        }
        fn clear_compute_coordinates(&mut self, node: NodeId) {
            self.cleared.push((node, NodeKind::Compute));
        }
    }

    /// e230: the record's own dims and state are dropped, and the caller's dims are what reaches the
    /// queue and the ledger.
    #[test]
    fn a_step_taken_from_a_record_carries_the_callers_dims_and_not_the_records() {
        let mut tracker = CoordPropTracker::default();
        let record = QueuedProp {
            prop: step(0, 1),
            state: PropState::Complete,
            dims: vec![PrimaryDim::Y],
        };

        tracker.add_prop_info_from(&record, &[PrimaryDim::In, PrimaryDim::Mb]);

        assert_eq!(tracker.items.len(), 1);
        assert_eq!(tracker.items[0].dims, vec![PrimaryDim::In, PrimaryDim::Mb]);
        assert_eq!(tracker.items[0].state, PropState::NotProcessed);
        // The record's `scaleDown` survives; this overload forwards it, unlike `retry`.
        assert_eq!(tracker.items[0].prop.scale_down, ScaleDown::Yes);
        // `Y` was never seen, so the record's own dim list left no trace in the ledger.
        assert!(!tracker.refs_added[&NodeId(1)][&NodeId(0)].contains_key(&PrimaryDim::Y));
    }

    /// e231: every entry from the rollback position through the cursor is marked and has its fold
    /// node's own coordinate store emptied, and the cursor lands one BEFORE the position.
    #[test]
    fn a_rollback_marks_and_clears_from_the_position_through_the_cursor() {
        let mut nodes = Nodes {
            //          0 alloc     1 transfer      2 compute        3 loop
            kinds: vec![
                NodeKind::Allocate,
                NodeKind::Transfer,
                NodeKind::Compute,
                NodeKind::Loop,
            ],
            cleared: Vec::new(),
        };
        let mut tracker = CoordPropTracker::default();
        for fold_node in 0..4 {
            tracker.add_prop_info(step(4 + fold_node, fold_node), &[PrimaryDim::In]);
        }
        // Walk to the last entry, so the whole queue is behind the cursor.
        for _ in 0..4 {
            let _ = tracker.next_item();
        }

        tracker.rollback_to_pos(&mut nodes, QueuePos(1));

        // The `LOOP` fold node owns no coordinate store, so nothing is cleared for it.
        assert_eq!(
            nodes.cleared,
            vec![
                (NodeId(1), NodeKind::Transfer),
                (NodeId(2), NodeKind::Compute),
            ]
        );
        assert_eq!(tracker.items[0].state, PropState::Complete);
        assert_eq!(tracker.items[1].state, PropState::RolledBack);
        assert_eq!(tracker.items[3].state, PropState::RolledBack);
        // One before the rollback position, so entry 1 is handed out again.
        assert_eq!(tracker.curr, Some(QueuePos(0)));
        assert_eq!(
            tracker.next_item().map(|item| item.dims),
            Some(vec![PrimaryDim::In])
        );
    }

    /// e231, the negative: a position the cursor has not reached yet is a no-op, and so is any
    /// rollback before the first fetch — including one to position 0.
    #[test]
    fn a_rollback_past_the_cursor_or_before_the_first_fetch_changes_nothing() {
        let mut nodes = Nodes {
            kinds: vec![NodeKind::Allocate, NodeKind::Allocate],
            cleared: Vec::new(),
        };
        let mut tracker = CoordPropTracker::default();
        tracker.add_prop_info(step(0, 0), &[PrimaryDim::In]);
        tracker.add_prop_info(step(1, 1), &[PrimaryDim::In]);

        // Cursor is still before the first entry: `currItemToProcess_ < 0`.
        tracker.rollback_to_pos(&mut nodes, QueuePos(0));
        assert_eq!(tracker.curr, None);

        let _ = tracker.next_item();
        // Cursor is at 0, so position 1 is ahead of it.
        tracker.rollback_to_pos(&mut nodes, QueuePos(1));

        assert!(nodes.cleared.is_empty());
        assert_eq!(tracker.curr, Some(QueuePos(0)));
        assert_eq!(tracker.items[0].state, PropState::Complete);
    }

    /// e232: the queue, the ledger and the cursor all go, so a propagation already walked can be
    /// queued a second time — which a rollback to 0 would not allow.
    #[test]
    fn a_reset_drops_the_ledger_so_a_walked_propagation_queues_again() {
        let mut tracker = CoordPropTracker::default();
        tracker.add_prop_info(step(0, 1), &[PrimaryDim::In]);
        let _ = tracker.next_item();

        tracker.reset();

        assert!(tracker.items.is_empty());
        assert!(tracker.refs_added.is_empty());
        assert_eq!(tracker.curr, None);

        tracker.add_prop_info(step(0, 1), &[PrimaryDim::In]);
        assert_eq!(tracker.items.len(), 1);
    }
}

// ⭐ TESTS FOR ENTRY 296. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e296 {
    use super::fold::{BlockId, CoordPropInfo, NodeId, NodeKind, PropEnd, ScheduleTree};
    use super::{
        CoordPropTracker, NodeCoordinates, PropState, Propagation, QueuePos, RefRole, ScaleDown,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;

    fn step(ref_node: u32, node_to_fold: u32) -> Propagation {
        Propagation {
            ends: CoordPropInfo {
                data_connect: None,
                ref_node: PropEnd::Other,
                node_to_fold: PropEnd::Other,
            },
            ref_node: NodeId(ref_node),
            node_to_fold: NodeId(node_to_fold),
            ref_role: RefRole::Producer,
            scale_down: ScaleDown::Yes,
        }
    }

    /// A KIND AND A PARENT PER NODE, recording which store was emptied for which node.
    struct Nodes {
        kinds: Vec<NodeKind>,
        parents: Vec<Option<NodeId>>,
        cleared: Vec<(NodeId, NodeKind)>,
    }

    impl ScheduleTree for Nodes {
        fn kind(&self, node: NodeId) -> NodeKind {
            self.kinds[node.0 as usize]
        }
        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.parents[node.0 as usize]
        }
        fn children(&self, _block: BlockId) -> Vec<NodeId> {
            Vec::new()
        }
    }

    impl NodeCoordinates for Nodes {
        fn clear_allocate_coordinates(&mut self, node: NodeId) {
            self.cleared.push((node, NodeKind::Allocate));
        }
        fn clear_transfer_coordinates(&mut self, node: NodeId) {
            self.cleared.push((node, NodeKind::Transfer));
        }
        fn clear_compute_coordinates(&mut self, node: NodeId) {
            self.cleared.push((node, NodeKind::Compute));
        }
    }

    /// NODE 0 is the block, 1 a loop under it, 2 a compute under that loop, 3 a transfer OUTSIDE the
    /// block, and 4 the block's own child transfer.
    fn tree() -> Nodes {
        Nodes {
            kinds: vec![
                NodeKind::Block,
                NodeKind::Loop,
                NodeKind::Compute,
                NodeKind::Transfer,
                NodeKind::Transfer,
            ],
            parents: vec![
                None,
                Some(NodeId(0)),
                Some(NodeId(1)),
                None,
                Some(NodeId(0)),
            ],
            cleared: Vec::new(),
        }
    }

    /// e296: the FIRST step before the cursor touching the block's subtree is where the queue rewinds
    /// to, and the match may come through either end at any depth.
    #[test]
    fn the_first_step_reaching_into_the_block_is_what_the_queue_rewinds_to() {
        let mut nodes = tree();
        let mut tracker = CoordPropTracker::default();
        // 0: neither end in the block. 1: `nodeToFold` two levels down. 2: `refNode` a direct child.
        tracker.add_prop_info(step(3, 3), &[PrimaryDim::In]);
        tracker.add_prop_info(step(3, 2), &[PrimaryDim::In]);
        tracker.add_prop_info(step(4, 3), &[PrimaryDim::In]);
        for _ in 0..3 {
            let _ = tracker.next_item();
        }

        let block = BlockId::of(&nodes, NodeId(0)).unwrap();
        tracker.roll_back_nodes_in_block(&mut nodes, block);

        assert_eq!(tracker.items[0].state, PropState::Complete);
        assert_eq!(tracker.items[1].state, PropState::RolledBack);
        assert_eq!(tracker.items[2].state, PropState::RolledBack);
        assert_eq!(tracker.curr, Some(QueuePos(0)));
        assert_eq!(
            nodes.cleared,
            vec![
                (NodeId(2), NodeKind::Compute),
                (NodeId(3), NodeKind::Transfer),
            ]
        );
    }

    /// e296, the negative: the cursor's own entry is outside the scan, so the step just fetched is
    /// never the one rolled back.
    #[test]
    fn the_entry_the_cursor_sits_on_is_not_scanned() {
        let mut nodes = tree();
        let mut tracker = CoordPropTracker::default();
        tracker.add_prop_info(step(3, 3), &[PrimaryDim::In]);
        tracker.add_prop_info(step(4, 3), &[PrimaryDim::In]);
        for _ in 0..2 {
            let _ = tracker.next_item();
        }

        let block = BlockId::of(&nodes, NodeId(0)).unwrap();
        tracker.roll_back_nodes_in_block(&mut nodes, block);

        assert!(nodes.cleared.is_empty());
        assert_eq!(tracker.curr, Some(QueuePos(1)));
    }
}
