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

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ USES FOR ENTRIES 132-136. Union these into this file's top block when its other entries land.
// ════════════════════════════════════════════════════════════════════════════════════════════════

use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Arch, IsaGen};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
use crate::generated::ComputeType;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::NodeId;
use crate::schedule::ddc::metadata::{DatastageId, Metadata};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::dsc2::{ComputeNode, LdsIdx, TransferNode, generic_comp};
use crate::units::{Core, Corelet};

// crustify:todo: e124_getLdsOrConstNameOfAllocNode
//   authority : ddc/ddcv1.cpp:20  (10 body lines, level 0)
//   class     : Ddc
//   original  : std::string Ddc::getLdsOrConstNameOfAllocNode(dsc2::AllocateNode* anode)
//   extract   : crustify-ddc/cpp/ddc.cpp:1945-1955

// crustify:todo: e125_minimizeAllocations
//   authority : ddc/ddcv1.cpp:30  (101 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::minimizeAllocations(bool has_auto_shuffling)
//   extract   : crustify-ddc/cpp/ddc.cpp:1967-2068
//   ⛔ NOTE   : Part of the ddcv1.cpp:30-360 placement span that reginit.rs hand-transcribes. The
//               authority is this body.

// crustify:todo: e126_populateUnitTimeTransfers
//   authority : ddc/ddcv1.cpp:439  (115 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::populateUnitTimeTransfers()
//   extract   : crustify-ddc/cpp/ddc.cpp:2078-2193

// crustify:todo: e127_spreadDataInAllocate
//   authority : ddc/ddcv1.cpp:1682  (27 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::spreadDataInAllocate()
//   extract   : crustify-ddc/cpp/ddc.cpp:2203-2230

// crustify:todo: e128_finalizeAllocateLayouts
//   authority : ddc/ddcv1.cpp:1712  (30 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::finalizeAllocateLayouts()
//   extract   : crustify-ddc/cpp/ddc.cpp:2240-2270

// crustify:todo: e129_getClSplitDim
//   authority : ddc/ddcv1.cpp:1799  (19 body lines, level 0)
//   original  : std::set<PrimaryDimTypes> getClSplitDim(const DesignSpaceConfig& dsc)
//   extract   : crustify-ddc/cpp/ddc.cpp:2279-2298

// crustify:todo: e130_getPeSfpSplitDim
//   authority : ddc/ddcv1.cpp:1819  (47 body lines, level 0)
//   original  : std::set<PrimaryDimTypes> getPeSfpSplitDim( const DesignSpaceConfig& dsc, const DesignSpaceConfigGlobal& dscGlobal)
//   extract   : crustify-ddc/cpp/ddc.cpp:2307-2355

// crustify:todo: e131_setPeFoldsIfPtInteraction
//   authority : ddc/ddcv1.cpp:1868  (22 body lines, level 0)
//   original  : void setPeFoldsIfPtInteraction(DesignSpaceConfig* currDsc, const DesignSpaceConfigGlobal& dscGlobal)
//   extract   : crustify-ddc/cpp/ddc.cpp:2364-2387

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
    use crate::schedule::dsc2::{DataInfo, Dsts, NodeName, Operand};

    /// The one corelet every build has, which is all these fixtures need.
    fn corelet0() -> Corelet {
        Corelet::checked(0).expect("every core has a corelet 0")
    }

    fn operand(unit: SenComponent, lds: Option<u32>) -> Operand {
        Operand {
            unit,
            data: DataInfo {
                data_connect: None,
                my_lds_idx: lds.map(LdsIdx),
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
