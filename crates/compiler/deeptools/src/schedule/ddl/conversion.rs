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

//! `ddc/ddl/ddl_conversion.cpp`, `ddc/ddl/ddl_conversion.h` — 31 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4, 5]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e172_processTypes` | 172 | 0 | 24 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:447` |
//! | `e173_addInternalTensor` | 173 | 0 | 107 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:473` |
//! | `e174_processExpression` | 174 | 0 | 10 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2072` |
//! | `e175_getTensorProp` | 175 | 0 | 26 | `DdlInterface` | `ddc/ddl/ddl_conversion.cpp:2083` |
//! | `e176_getTensor` | 176 | 0 | 20 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2825` |
//! | `e177_dump` | 177 | 0 | 23 | `DimProp` | `ddc/ddl/ddl_conversion.cpp:3509` |
//! | `e178_getAccessPattern` | 178 | 0 | 8 | `DataTransfer` | `ddc/ddl/ddl_conversion.cpp:3619` |
//! | `e179_getAccessPatternAsStr` | 179 | 0 | 14 | `DataTransfer` | `ddc/ddl/ddl_conversion.cpp:3629` |
//! | `e180_checkAccessPattern` | 180 | 0 | 34 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:3648` |
//! | `e181_convertAccessPatternStrToDdcType` | 181 | 0 | 29 | — | `ddc/ddl/ddl_conversion.cpp:3687` |
//! | `e182_convertAccessPatternStrToDdcType` | 182 | 0 | 7 | — | `ddc/ddl/ddl_conversion.cpp:3718` |
//! | `e183_dump` | 183 | 0 | 13 | `DataTransfer` | `ddc/ddl/ddl_conversion.cpp:3756` |
//! | `e184_setMetaDimKind` | 184 | 0 | 7 | `DimProp` | `ddc/ddl/ddl_conversion.h:317` |
//! | `e185_isMetaDim` | 185 | 0 | 5 | `DimProp` | `ddc/ddl/ddl_conversion.h:329` |
//! | `e186_getNonPaddedDimProp` | 186 | 0 | 5 | `DdlInterface` | `ddc/ddl/ddl_conversion.h:351` |
//! | `e187_clear` | 187 | 0 | 4 | `DdlInterface` | `ddc/ddl/ddl_conversion.h:458` |
//! | `e274_processDimensionOp` | 274 | 1 | 22 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:187` |
//! | `e275_processCondition` | 275 | 1 | 234 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:211` |
//! | `e276_verifyDdlConstraints` | 276 | 1 | 216 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2553` |
//! | `e277_getTensorAndAllocation` | 277 | 1 | 31 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2847` |
//! | `e278_checkMetaDimensions` | 278 | 1 | 81 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:3537` |
//! | `e279_processAccessPatterns` | 279 | 1 | 24 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:3727` |
//! | `e322_processPaddedDimensionOp` | 322 | 2 | 84 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:102` |
//! | `e323_processOp` | 323 | 2 | 1424 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:582` |
//! | `e324_processTransformations` | 324 | 2 | 35 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2036` |
//! | `e325_convertDsc2Ddl` | 325 | 2 | 627 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2881` |
//! | `e345_exportToDdl` | 345 | 3 | 3 | `DdlConvertInterface` | `ddc/ddl/ddl_conversion.cpp:38` |
//! | `e346_processRegion` | 346 | 3 | 26 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2008` |
//! | `e347_matchDdl2Dsc` | 347 | 3 | 442 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2110` |
//! | `e364_parseDdl2Dsc` | 364 | 4 | 54 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2770` |
//! | `e372_selectAndParseDdlTemplate` | 372 | 5 | 59 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:42` |

// ⭐ USES FOR ENTRIES 172-187. Union these into this file's top block when its other entries land.
use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::SenComponent;

use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
use crate::formats::{Bits, DataFormat};
use crate::generated::{
    AccessPattern, Attrs, DimProperty, NameId, Operand, PaddingType, Program, StmtKind,
};
use crate::schedule::ddc::fold::{AllocId, ConstIdx, DataOrigin, NodeId, PadType};
use crate::schedule::ddc::metadata::{
    DataTransfer, ExternalStorage, MetaDimKind, Metadata, TransferAccessPattern,
};
use crate::schedule::dsc2::LdsIdx;

// ⭐ TYPES FOR ENTRIES 172-179 — the `DdlInterface` sub-structures those entries read and write, and
// the seam entry 173 mutates the DSC through.

/// AN ELEMENT FORMAT A TEMPLATE DECLARES — `DdlInterface::TypeDefinition`
/// (`ddc/ddl/ddl_conversion.h:361-365`) once its two `INVALID`/`-1` sentinels are gone.
///
/// ⛔ NEITHER FIELD CAN BE UNSET HERE. `dataFormat_ == INVALID` is the reference's "not yet parsed"
/// flag on a memo entry, and `bitSize_ == -1` never survives [`process_types`] — so a value of this
/// type is a PARSED type, which is what removes the *"Invalid type name"* abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeDefinition {
    /// `dataFormat_`.
    pub format: DataFormat,
    /// `bitSize_` — `bit_width=` where the template states one, else the format's own width.
    pub bit_size: Bits,
}

/// WHICH LABELED DS A DDL TENSOR IS — `DdlInterface::TensorProp` (`ddc/ddl/ddl_conversion.h:373`),
/// whose one field's `-1` default is the [`None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TensorProp {
    /// `ldsIdx_`.
    pub lds: Option<LdsIdx>,
}

/// HOW MANY GLOBAL LAYOUTS REFER TO A DIM — `DimProp::numRefsInGlobalLayouts_`, the count
/// `matchDdl2Dsc` bumps per reference (`ddc/ddl/ddl_conversion.cpp:2395`) and sorts dims by (`:2457`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct GlobalLayoutRefs(pub u32);

/// ONE DDL DIMENSION'S PROPERTIES — `DdlInterface::DimProp` (`ddc/ddl/ddl_conversion.h:297-340`).
///
/// ⛔ `dim_`'s `PrimaryDimTypesCount` DEFAULT IS AN ABSENCE, not dimension zero, and entry 177 is the
/// one place that spelling is observable: it prints `Invalid` for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimProp {
    /// `dim_` — the assigned primary dim, absent while unset.
    pub dim: Option<PrimaryDim>,
    /// `dimCandidates_`, in `PrimaryDimTypes` ordinal order, which is what a `std::set` iterates in.
    pub dim_candidates: Vec<PrimaryDim>,
    /// `numRefsInGlobalLayouts_`.
    pub global_layout_refs: GlobalLayoutRefs,
    /// `dropDim_`.
    pub drop_dim: bool,
    /// `nonPaddedDim` — the unpadded dim this padded one refers to.
    pub non_padded_dim: Option<NameId>,
    /// `metaDimKind_`, whose default is `Unpadded`.
    pub meta_dim_kind: MetaDimKind,
}

impl Default for DimProp {
    /// `DimProp()` (`ddc/ddl/ddl_conversion.h:307-312`).
    ///
    /// ⭐ THE CANDIDATE LIST STARTS FULL — every dim EXCEPT `IJ` and `KIJ` — and narrows from there,
    /// so an empty one is a dim whose candidates were all eliminated and not a fresh one.
    fn default() -> Self {
        Self {
            dim: None,
            dim_candidates: PrimaryDim::ALL
                .into_iter()
                .filter(|dim| !matches!(dim, PrimaryDim::Ij | PrimaryDim::Kij))
                .collect(),
            global_layout_refs: GlobalLayoutRefs(0),
            drop_dim: false,
            non_padded_dim: None,
            meta_dim_kind: MetaDimKind::Unpadded,
        }
    }
}

/// WHICH COMPUTE OP OF `dsc.computeOp_` — `addInternalTensor`'s `computeOpIdx`, a POSITION in that
/// vector and not a node identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeOpIdx(pub usize);

/// THE TAIL OF A NON-EMPTY `dsc.labeledDs_`, which is the only shape entry 173 can act on.
///
/// ⛔⛔ THIS CLOSES THREE UNGUARDED READS AT ONCE. On an empty `labeledDs_` the reference evaluates
/// `size() - 1` as `size_t`, dereferences `back()` and inserts at `end() - 1`; absent instead of
/// wrong is what [`add_internal_tensor`] answers there.
///
/// ⛔ TRAP: the two fields ARE DIFFERENT QUESTIONS. `insert_position` is where the new LDS lands,
/// `last_lds` is the INDEX its neighbour carries, and the reference assumes they agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabeledDsTail {
    /// `dsc.labeledDs_.size() - 1` — the new LDS's own `ldsIdx_`.
    pub insert_position: LdsIdx,
    /// `dsc.labeledDs_.back().ldsIdx_`.
    pub last_lds: LdsIdx,
}

/// THE NEW INTERNAL TENSOR ENTRY 173 MINTS — everything about it that is entry 173's decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalTensor {
    /// `ldsIdx_` — [`LabeledDsTail::insert_position`].
    pub lds: LdsIdx,
    /// `dsName_` — `"internal_tensor_lds" + std::to_string(ldsIdx_)`.
    pub name: String,
    /// `referenceLdsIdx_`, and the LDS every copied field is copied from.
    pub reference: LdsIdx,
}

/// ONE PLACE IN THE SCHEDULE TREE THAT NAMES A LABELED DS — every slot entry 173's walk retags.
///
/// ⛔ THE `DT_ERROR("the node has to be either compute, transfer or allocate.")` IS UNSPELLABLE: the
/// walk is restricted to those three kinds and this enum has an arm per slot of each of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LdsSlot {
    /// `metadata_.opaqueOps_.at(cn).ldsIdx_`, for a compute whose `isOpaqueOp_` is set.
    OpaqueCompute(NodeId),
    /// `cn->inputsLdsAndLoopOffsets_.at(i).myLdsIdx_`.
    ComputeInput(NodeId, usize),
    /// `cn->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_`.
    ComputeOutput(NodeId, usize),
    /// `tn->srcLdsAndLoopOffsets_.myLdsIdx_`.
    TransferSrc(NodeId),
    /// `tn->dstLdsAndLoopOffsets_.at(i).myLdsIdx_`.
    TransferDst(NodeId, usize),
    /// `an->ldsIdx_`.
    Allocate(NodeId),
}

/// WHAT ENTRY 173 DOES TO THE DSC — the tail it reads, the LDS it inserts, the index it bumps, the
/// tensor definition it follows, the interim list it appends to, and the tree slots it retags.
///
/// ⭐ EVERY MUTATION THE REFERENCE PERFORMS IS ONE METHOD HERE; WHICH to call, in what order and
/// with what value stays in the port. The `oldNewLdsPtrs` fixup is not here: re-seating raw pointers
/// after a vector insert is the mechanism for reaching an LDS, and an index does not need it.
pub trait InternalTensorSite {
    /// `dsc.labeledDs_`'s tail, absent where that vector is empty.
    fn labeled_ds_tail(&self) -> Option<LabeledDsTail>;
    /// `++dsc.labeledDs_.back().ldsIdx_`.
    fn set_last_lds_idx(&mut self, lds: LdsIdx);
    /// `ddlInterface.tensor_definition_`'s SSA names whose `ldsIdx_` is this one, IN MAP ORDER.
    fn tensors_with_lds(&self, lds: LdsIdx) -> Vec<NameId>;
    /// `tensor_definition_[tensor].ldsIdx_ = lds`.
    fn set_tensor_lds(&mut self, tensor: NameId, lds: LdsIdx);
    /// `dsc.labeledDs_.insert(end() - 1, newLds)`, where `newLds` copies `dsType_`, `scale_`,
    /// `density_`, `wordLength` and `dataFormat_` from [`InternalTensor::reference`] and takes
    /// `segment_ = LdsSegment::STACK`, `isFirstUse_ = true` and
    /// `hbmSize_ = lxSize_ = lxBufferSize_ = 0`.
    fn insert_internal_tensor(&mut self, new: InternalTensor);
    /// `dsc.computeOp_.at(compute_op).interimLabeledDs.push_back(&newLds)`.
    fn add_interim_lds(&mut self, compute_op: ComputeOpIdx, lds: LdsIdx);
    /// `traverseTreeDFSMutable(nullptr, {ALLOCATE, TRANSFER, COMPUTE})` projected onto every slot of
    /// those nodes that names a labeled DS, in the traversal's order.
    fn lds_slots(&self) -> Vec<LdsSlot>;
    /// The index in that slot, absent for a `-1`.
    /// ⛔ NOT for [`LdsSlot::OpaqueCompute`]: that slot lives in the [`Metadata`], and the port reads
    /// and writes it there.
    fn slot_lds(&self, slot: LdsSlot) -> Option<LdsIdx>;
    /// Writes that slot, under the same exclusion.
    fn set_slot_lds(&mut self, slot: LdsSlot, lds: LdsIdx);
    /// `compAndAllocNode`'s own `allocNode->ldsIdx_` — a HANDLE THE TREE MAY NOT HOLD, since the
    /// metadata owns some allocate nodes outright (`ddc/ddc_metadata.h:132`), which is why it is
    /// reached by [`AllocId`] and not through [`LdsSlot::Allocate`].
    fn allocation_lds(&self, alloc: AllocId) -> Option<LdsIdx>;
    /// Writes that allocate node's index.
    fn set_allocation_lds(&mut self, alloc: AllocId, lds: LdsIdx);
}

/// HOW A `PadType` IS SPELLED — `EnumsConversion::padTypeToString` (`dsc/dims.cpp:39-46`), total over
/// the six, so the reference's `.at()` cannot throw.
///
/// ⭐ A FREE FUNCTION RATHER THAN AN INHERENT `impl`, because [`PadType`] is `ddc::fold`'s type and
/// this spelling is `ddl_conversion`'s only need of it.
#[must_use]
pub const fn pad_type_spelling(pad: PadType) -> &'static str {
    match pad {
        PadType::NoPad => "nopad",
        PadType::LoweredPadded => "lowered_padded",
        PadType::PaddedNoZeroPad => "padded_nozeropad",
        PadType::PaddedWZeroPad => "padded_wzeropad",
        PadType::PaddedFullSpan => "padded_fullspan",
        PadType::PaddedFullSpanWUnneeded => "padded_fullspan_wunneeded",
    }
}

/// A NUMBER A DDL EXPRESSION EVALUATES TO — `processExpression`'s `float` return, which the
/// datastage constraints and scales read as a RATIO and not as a count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExprValue(pub f32);

/// Replaces: e172_processTypes
///
/// EVERY `ddl.type` OPERAND OF A USER OP, PARSED: the format from `data_type=`, the width from
/// `bit_width=` where the template states one and from the format's own table otherwise.
///
/// ⛔ [`None`] IS *"Illegal ddl"* — an operand whose definition is not a `ddl.type`. The other abort,
/// *"Invalid type name"*, is unspellable: `data_type=` is a censused [`crate::generated::DataType`].
/// ⭐ THE `type_definition_` MEMO IS DROPPED — `ddl_conversion.cpp:451` is its only read.
pub fn process_types(program: &Program, supported_types: &[NameId]) -> Option<Vec<TypeDefinition>> {
    let mut types = Vec::with_capacity(supported_types.len());
    for name in supported_types {
        let Attrs::Type {
            data_type,
            bit_width,
        } = program.definition(*name)?.attrs
        else {
            return None;
        };
        let format = DataFormat::from(data_type);
        types.push(TypeDefinition {
            format,
            bit_size: bit_width
                .and_then(|width| u32::try_from(width).ok())
                .map_or(format.bits(), Bits),
        });
    }
    Some(types)
}

/// Replaces: e173_addInternalTensor
///
/// MINTS A STACK LDS FOR A COMPUTE'S INTERIM TENSOR: inserts it BEFORE the last LDS, bumps that last
/// LDS's own index, appends it to the compute's `interimLabeledDs`, and retags every schedule-tree
/// slot, allocation and prefilled external transfer that named the old last index.
///
/// ⛔ TRAP: the `tensor_definition_` retag `break`s after the FIRST match over an `unordered_map`, so
/// where several tensors carry the old last index WHICH ONE FOLLOWS IT IS ARBITRARY in the reference.
/// ⛔ [`None`] where `labeledDs_` is empty — see [`LabeledDsTail`].
pub fn add_internal_tensor<S: InternalTensorSite + ?Sized>(
    site: &mut S,
    metadata: &mut Metadata,
    reference: LdsIdx,
    compute_op: ComputeOpIdx,
) -> Option<LdsIdx> {
    let tail = site.labeled_ds_tail()?;
    let old_last = tail.last_lds;
    let new_last = LdsIdx(old_last.0 + 1);
    site.set_last_lds_idx(new_last);
    if let Some(&tensor) = site.tensors_with_lds(old_last).first() {
        site.set_tensor_lds(tensor, new_last);
    }
    site.insert_internal_tensor(InternalTensor {
        lds: tail.insert_position,
        name: format!("internal_tensor_lds{}", tail.insert_position.0),
        reference,
    });
    site.add_interim_lds(compute_op, tail.insert_position);
    for slot in site.lds_slots() {
        if let LdsSlot::OpaqueCompute(node) = slot {
            // `metadata_.opaqueOps_.at(cn)`: the entry is minted with the node, so an absent one is
            // not reachable from a walk of the tree that node is in.
            if let Some(opaque) = metadata.opaque_ops.get_mut(&node) {
                if opaque.lds_idx == Some(old_last) {
                    opaque.lds_idx = Some(new_last);
                }
            }
        } else if site.slot_lds(slot) == Some(old_last) {
            site.set_slot_lds(slot, new_last);
        }
    }
    for allocation in metadata.new_allocations.values_mut() {
        if let Some(alloc) = allocation.lds_idx_and_alloc_node.remove(&old_last) {
            allocation.lds_idx_and_alloc_node.insert(new_last, alloc);
        }
        for alloc in allocation.comp_and_alloc_node.values() {
            if site.allocation_lds(*alloc) == Some(old_last) {
                site.set_allocation_lds(*alloc, new_last);
            }
        }
    }
    let prefilled = &mut metadata.prefilled_external_transfer_data_connects;
    if let Some(slot) = prefilled.remove(&(old_last, ExternalStorage::Lx)) {
        prefilled.insert((new_last, ExternalStorage::Lx), slot);
    }
    Some(tail.insert_position)
}

/// Replaces: e174_processExpression
///
/// A DDL EXPRESSION AS A NUMBER — `strtof` over the whole string, which accepts whitespace around
/// the number and nothing else after it.
///
/// ⛔ [`None`] IS *"Unable to convert expression to number"*, the reference's abort.
/// ⛔ DIVERGENCE: `strtof` also accepts C99 HEX FLOATS (`0x1p3`) and Rust's parser does not, so such
/// an expression is [`None`] here and `8.0` there. No vendored template states one.
#[must_use]
pub fn process_expression(expr: &str) -> Option<ExprValue> {
    expr.trim().parse::<f32>().ok().map(ExprValue)
}

/// Replaces: e175_getTensorProp
///
/// WHICH LABELED DS A DDL TENSOR IS, resolving a `ddl.alias_one_tensor_of` to whichever of its
/// tensors is active and MEMOISING that answer under the alias's own name.
///
/// ⭐ THE MEMO IS AN OUTPUT, not mechanism: nine other sites in `ddl_conversion.cpp` read
/// `tensor_definition_` directly, so an alias resolved here has to stay resolved.
/// ⛔ [`None`] IS *"Illegal ddl"* — the aborts *"Not a valid tensor"*, *"Multiple tensors are active"*
/// and *"No tensor is active"*.
pub fn tensor_prop(
    program: &Program,
    tensor_definition: &mut BTreeMap<NameId, TensorProp>,
    tensor: NameId,
) -> Option<TensorProp> {
    if let Some(prop) = tensor_definition.get(&tensor) {
        return Some(*prop);
    }
    let alias = program.definition(tensor)?;
    if alias.kind != StmtKind::AliasOneTensorOf {
        return None;
    }
    let mut active = None;
    for aliased in alias.operands.iter().flat_map(|operand| match operand {
        Operand::One(name) => vec![*name],
        Operand::List(names) => names.to_vec(),
        Operand::OtherBind => Vec::new(),
    }) {
        if let Some(prop) = tensor_definition.get(&aliased) {
            if active.is_some() {
                return None;
            }
            active = Some(*prop);
        }
    }
    let prop = active?;
    tensor_definition.insert(tensor, prop);
    Some(prop)
}

/// Replaces: e176_getTensor
///
/// WHICH DDL TENSOR A `dsc2::DataInfo` NAMES — its labeled DS if it has one, else its constant, else
/// the operand-constant tensor bound to `unit`. [`None`] is the reference's null `Value`.
///
/// ⛔ TRAP: EVERY LOOP TAKES THE LAST MATCH — there is no `break` — over an `unordered_map`, so where
/// two names share an index the reference's answer is UNSPECIFIED and this one is the greatest name.
/// ⭐ The three arms are `Option<DataOrigin>`, which is also the reference's priority: an LDS wins.
#[must_use]
pub fn tensor(
    tensor_definition: &BTreeMap<NameId, TensorProp>,
    ext_constant_definition: &BTreeMap<NameId, ConstIdx>,
    operand_constant_tensor: &BTreeMap<NameId, SenComponent>,
    unit: SenComponent,
    origin: Option<DataOrigin>,
) -> Option<NameId> {
    match origin {
        Some(DataOrigin::LabeledDs(lds)) => tensor_definition
            .iter()
            .filter(|(_, prop)| prop.lds == Some(lds))
            .map(|(name, _)| *name)
            .last(),
        Some(DataOrigin::Constant(constant)) => ext_constant_definition
            .iter()
            .filter(|(_, id)| **id == constant)
            .map(|(name, _)| *name)
            .last(),
        None => operand_constant_tensor
            .iter()
            .filter(|(_, component)| **component == unit)
            .map(|(name, _)| *name)
            .last(),
    }
}

impl DimProp {
    /// Replaces: e177_dump
    ///
    /// THE DIM'S PROPERTIES AS `llvm::errs()` STATES THEM, returned rather than written because the
    /// caller owns the stream. An empty `msg` omits the header and its underline.
    ///
    /// ⛔ AN UNSET `dim_` PRINTS `Invalid`, not `primaryDimToString`'s `"undefined"`.
    /// ⛔ DIVERGENCE: MLIR streams a `Value` as its DEFINING OPERATION; there is no operation here,
    /// so `nonPaddedDim` prints its SSA name — and MLIR's own `<<NULL VALUE>>` when it is absent.
    #[must_use]
    pub fn dump(&self, program: &Program, msg: &str) -> String {
        let mut out = String::new();
        if !msg.is_empty() {
            out.push_str(&format!("\n{msg}\n{}", "-".repeat(msg.len())));
        }
        out.push_str("\nDimProp: \n  PrimaryDimTypes = ");
        out.push_str(self.dim.map_or("Invalid", PrimaryDim::spelling));
        out.push_str("\n  DropDim         = ");
        out.push_str(if self.drop_dim { "T" } else { "F" });
        out.push_str("\n  NonPaddedDim    = ");
        match self
            .non_padded_dim
            .and_then(|name| program.names.get(usize::from(name.0)))
        {
            Some(name) => out.push_str(name),
            None => out.push_str("<<NULL VALUE>>"),
        }
        out.push_str("\n  MetaDimKind     = ");
        out.push_str(self.meta_dim_kind.label());
        out.push_str("\n  dimCandidates   = [");
        for candidate in &self.dim_candidates {
            out.push(' ');
            out.push_str(candidate.spelling());
        }
        out.push_str(&format!(
            "]\n  NumRefsInGlobalLayouts = {}\n",
            self.global_layout_refs.0
        ));
        out
    }

    /// Replaces: e184_setMetaDimKind
    ///
    /// TAKES THE KIND FROM A `dim_property=` — `stringToMetaDimKind.at(inDimKind)`.
    ///
    /// ⛔ THE `false` RETURN IS UNSPELLABLE: `dim_property=` is a censused [`DimProperty`] whose
    /// absence the dialect defaults to `"unpadded"`, which is the [`None`] arm.
    /// ⛔ `Padded` IS NOT REACHABLE FROM A STRING — `processPaddedDimensionOp` sets it through the
    /// enum overload of this setter, never through this one.
    pub fn set_meta_dim_kind(&mut self, property: Option<DimProperty>) {
        self.meta_dim_kind = match property {
            None => MetaDimKind::Unpadded,
            Some(DimProperty::Window) => MetaDimKind::WindowDim,
            Some(DimProperty::Stride) => MetaDimKind::Stride,
            Some(DimProperty::Dilation) => MetaDimKind::Dilation,
            Some(DimProperty::PadFront) => MetaDimKind::PadFront,
            Some(DimProperty::PadBack) => MetaDimKind::PadBack,
            Some(DimProperty::PadValid) => MetaDimKind::PadValid,
        };
    }

    /// Replaces: e185_isMetaDim
    ///
    /// WHETHER THIS DIM CARRIES A META KIND — anything but `Unpadded` and `Padded`.
    ///
    /// ⛔ THE THIRD DISJUNCT IS DISCHARGED BY THE TYPE: `MetaDimKind::Count` is an absence and has no
    /// variant, and [`DimProp::meta_dim_kind`] is never unset.
    #[must_use]
    pub const fn is_meta_dim(&self) -> bool {
        !matches!(
            self.meta_dim_kind,
            MetaDimKind::Unpadded | MetaDimKind::Padded
        )
    }
}

impl DataTransfer {
    /// Replaces: e178_getAccessPattern
    ///
    /// THE PAD-TYPE PAIR STATED FOR ONE DIM — `accessPatternPerDim_.at(dim)`.
    ///
    /// ⛔ [`None`] IS THE ABORT *"No access pattern is specified for the given dimension"*.
    #[must_use]
    pub fn access_pattern(&self, dim: PrimaryDim) -> Option<TransferAccessPattern> {
        self.access_pattern_per_dim.get(&dim).copied()
    }

    /// Replaces: e179_getAccessPatternAsStr
    ///
    /// THAT PAIR AS `<from>-to-<to>` in `padTypeToString`'s spellings — the string `convertDsc2Ddl`
    /// writes back into an `access_pattern_style=` attribute.
    ///
    /// ⛔ [`None`] IS THE SAME ABORT as [`Self::access_pattern`].
    #[must_use]
    pub fn access_pattern_spelling(&self, dim: PrimaryDim) -> Option<String> {
        let pattern = self.access_pattern(dim)?;
        Some(format!(
            "{}-to-{}",
            pad_type_spelling(pattern.from),
            pad_type_spelling(pattern.to)
        ))
    }

    /// Replaces: e183_dump
    ///
    /// THE TRANSFER'S ACCESS PATTERNS AS `std::cerr` STATES THEM, returned rather than written
    /// because the caller owns the stream.
    ///
    /// ⛔ THE FRAMING NEWLINES ARE THE REFERENCE'S: it opens with `"\n["` and closes with
    /// `std::endl`, so a transfer stating no pattern at all still prints three lines.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut out = String::from(
            "\n[Ddl::DataTransfer]\n--------------------------\n  access-patterns per dimension: ",
        );
        for (dim, pattern) in &self.access_pattern_per_dim {
            out.push_str(&format!(
                "\n    Dim {} access-pattern ({} to {})",
                dim.spelling(),
                pad_type_spelling(pattern.from),
                pad_type_spelling(pattern.to)
            ));
        }
        out.push('\n');
        out
    }
}

/// ONE `access_pattern_style=`/`padding_type=` LIST PAIRED TO THE DIMS IT DESCRIBES, which is the
/// only shape `processAccessPatterns` can walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyledDims<S>(Vec<(NameId, S)>);

impl<S: Copy> StyledDims<S> {
    /// Replaces: e180_checkAccessPattern
    ///
    /// THE PAIRING A DDL OP STATES, or absent where the reference emits *"Illegal ddl"*.
    ///
    /// ⛔ THE THREE ABORTS ARE THE [`None`]: dims with no styles, styles with no dims, and a style
    /// count that is neither 1 nor one-per-dim. An EMPTY pairing is the reference's `return false`
    /// — *"Nothing to process"* — and an absent attribute is an empty `styles` here, exactly as
    /// `!hasAccessPatternStyle || empty()` treats the two alike.
    /// ⭐ A SINGLE STYLE BROADCASTS at construction, which is what the caller then does per dim.
    #[must_use]
    pub fn stated(dims: &[NameId], styles: &[S]) -> Option<Self> {
        if styles.is_empty() {
            return dims.is_empty().then(|| Self(Vec::new()));
        }
        match styles {
            _ if dims.is_empty() => None,
            [only] => Some(Self(dims.iter().map(|dim| (*dim, *only)).collect())),
            _ if styles.len() == dims.len() => Some(Self(
                dims.iter().copied().zip(styles.iter().copied()).collect(),
            )),
            _ => None,
        }
    }

    /// The pairs, in the order the op states its dims.
    #[must_use]
    pub fn pairs(&self) -> &[(NameId, S)] {
        &self.0
    }
}

/// Replaces: e181_convertAccessPatternStrToDdcType
///
/// THE PAD-TYPE PAIR AN `access_pattern_style=` NAMES — the reference's split on `"-to-"` and its two
/// `stringToPadType` lookups, resolved once per spelling the templates actually state.
///
/// ⛔ ALL THREE *"Illegal ddl"* ABORTS ARE UNSPELLABLE: a missing `-to-`, an unknown source and an
/// unknown destination cannot occur for a value of [`AccessPattern`]. A new spelling in the census
/// makes this match non-exhaustive rather than silently unhandled.
#[must_use]
pub const fn transfer_access_pattern(pattern: AccessPattern) -> TransferAccessPattern {
    match pattern {
        AccessPattern::PaddedWzeropadToToToLoweredPadded => TransferAccessPattern {
            from: PadType::PaddedWZeroPad,
            to: PadType::LoweredPadded,
        },
    }
}

/// Replaces: e182_convertAccessPatternStrToDdcType
///
/// THE PAD TYPE AN ALLOCATION'S `padding_type=` NAMES — one `stringToPadType` lookup.
///
/// ⛔ THE *"Unknown memory access pattern"* ABORT IS UNSPELLABLE for a value of [`PaddingType`].
#[must_use]
pub const fn allocation_pad_type(padding: PaddingType) -> PadType {
    match padding {
        PaddingType::PaddedWzeropad => PadType::PaddedWZeroPad,
    }
}

/// THE DDL↔DSC SYMBOL TABLE — `DdlInterface` (`ddc/ddl/ddl_conversion.h:288-462`), carrying the
/// sub-maps whose element types are defined.
///
/// ⛔ TWELVE MEMBERS ARE STILL ABSENT, each arriving with the unit that decides its element type:
/// `operation_definition_`, `datastage_definition_`, `ext_constant_definition_`, `alloc_storage_`,
/// `transfer_acc_pat_dims_`, `operand_constant_tensor_`, `loop_labels_`, `core_chunk_loop_label_`,
/// `region2blocks_`, `resolvedConditions_`, `sync_definitions_` and `coreToCore_definitions_`.
/// [`Self::clear`] resets whatever the struct holds, so it stays correct as they land.
///
/// ⛔ THE MAPS ARE ORDERED WHERE THE REFERENCE'S ARE NOT: `unordered_map<Value, _>` iterates in an
/// unspecified order, so any unit that depends on this order is depending on more than the reference
/// guarantees.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DdlInterface {
    /// `dim_association_`.
    pub dim_association: BTreeMap<NameId, DimProp>,
    /// `type_definition_`. ⛔ [`process_types`] HANDS ITS VECTOR BACK rather than memoising
    /// here, so nothing in the port fills this; `ddl_conversion.cpp:451` is the only read.
    pub type_definition: BTreeMap<NameId, TypeDefinition>,
    /// `tensor_definition_`, which [`tensor_prop`] fills.
    pub tensor_definition: BTreeMap<NameId, TensorProp>,
}

impl DdlInterface {
    /// Replaces: e186_getNonPaddedDimProp
    ///
    /// THE DIM'S PROPERTIES, FOLLOWED THROUGH TO THE UNPADDED DIM IT NAMES when it is itself padded.
    ///
    /// ⛔ [`None`] IS A PADDED DIM WITH NO `nonPaddedDim`: the reference default-inserts a fresh
    /// `DimProp` under a null `Value` and hands that back. ⛔ ASKING IS A MUTATION — both lookups are
    /// `operator[]`, so a dim that was not in the map is in it afterwards.
    pub fn non_padded_dim_prop(&mut self, dim: NameId) -> Option<&mut DimProp> {
        let key = {
            let prop = self.dim_association.entry(dim).or_default();
            if matches!(prop.meta_dim_kind, MetaDimKind::Padded) {
                prop.non_padded_dim?
            } else {
                dim
            }
        };
        Some(self.dim_association.entry(key).or_default())
    }

    /// Replaces: e187_clear
    ///
    /// DROPS EVERY MAPPING — the reference's destructor-then-placement-new, which is a REBUILD and
    /// not a memset: a re-inserted [`DimProp`] gets its full candidate list back.
    ///
    /// ⛔ FOURTEEN LATER UNITS LIST THIS AS A CALLEE (`crustify-ddc/UNITS.tsv`); most of those
    /// are a container's own `.clear()` resolved by name, exactly as `e104_clear` was.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

// crustify:todo: e274_processDimensionOp
//   authority : ddc/ddl/ddl_conversion.cpp:187  (22 body lines, level 1)
//   class     : DdlConversion
//   original  : DdlInterface::DimProp& DdlConversion::processDimensionOp( const mlir::Value& dimVal)
//   extract   : crustify-ddc/cpp/ddl.cpp:860-883
//   calls     : e184_setMetaDimKind

// crustify:todo: e275_processCondition
//   authority : ddc/ddl/ddl_conversion.cpp:211  (234 body lines, level 1)
//   class     : DdlConversion
//   original  : const DdlInterface::CondProp& DdlConversion::processCondition( mlir::Value cond)
//   extract   : crustify-ddc/cpp/ddl.cpp:893-1128
//   calls     : e174_processExpression, e175_getTensorProp, e176_getTensor, e187_clear

// crustify:todo: e276_verifyDdlConstraints
//   authority : ddc/ddl/ddl_conversion.cpp:2553  (216 body lines, level 1)
//   class     : DdlConversion
//   original  : void DdlConversion::verifyDdlConstraints()
//   extract   : crustify-ddc/cpp/ddl.cpp:1138-1354
//   calls     : e175_getTensorProp, e176_getTensor

// crustify:todo: e277_getTensorAndAllocation
//   authority : ddc/ddl/ddl_conversion.cpp:2847  (31 body lines, level 1)
//   class     : DdlConversion
//   original  : std::pair<Value, Value> DdlConversion::getTensorAndAllocation( llvm::DenseMap<AllocateOp, const dsc2::AllocateNode*>& allocations, SenComponents unit, const dsc2::DataInfo dtinfo) const
//   extract   : crustify-ddc/cpp/ddl.cpp:1364-1397
//   calls     : e176_getTensor

// crustify:todo: e278_checkMetaDimensions
//   authority : ddc/ddl/ddl_conversion.cpp:3537  (81 body lines, level 1)
//   class     : DdlConversion
//   original  : void DdlConversion::checkMetaDimensions()
//   extract   : crustify-ddc/cpp/ddl.cpp:1407-1488
//   calls     : e185_isMetaDim

// crustify:todo: e279_processAccessPatterns
//   authority : ddc/ddl/ddl_conversion.cpp:3727  (24 body lines, level 1)
//   class     : DdlConversion
//   original  : template <typename OpT, typename AccPatT> void DdlConversion::processAccessPatterns( OpT& op, mlir::Operation::operand_range dims, mlir::ArrayAttr inOpAccessPatternStyles, std::map<PrimaryDimTypes, AccPatT>& result)
//   extract   : crustify-ddc/cpp/ddl.cpp:1498-1526
//   calls     : e181_convertAccessPatternStrToDdcType, e182_convertAccessPatternStrToDdcType

// crustify:todo: e322_processPaddedDimensionOp
//   authority : ddc/ddl/ddl_conversion.cpp:102  (84 body lines, level 2)
//   class     : DdlConversion
//   original  : void DdlConversion::processPaddedDimensionOp(const mlir::Value& dimVal)
//   extract   : crustify-ddc/cpp/ddl.cpp:1562-1646
//   calls     : e184_setMetaDimKind, e274_processDimensionOp

// crustify:todo: e323_processOp
//   authority : ddc/ddl/ddl_conversion.cpp:582  (1424 body lines, level 2)
//   class     : DdlConversion
//   original  : std::pair<dsc2::BlockNode*, std::vector<int>> DdlConversion::processOp( Operation& op, dsc2::BlockNode* currParent)
//   extract   : crustify-ddc/cpp/ddl.cpp:1656-3081
//   calls     : e069_updateMin, e070_updateMax, e071_updateValues, e096_updateMin, e097_updateMax, e098_updateValues, e173_addInternalTensor, e174_processExpression, e175_getTensorProp, e176_getTensor, e187_clear, e275_processCondition

// crustify:todo: e324_processTransformations
//   authority : ddc/ddl/ddl_conversion.cpp:2036  (35 body lines, level 2)
//   class     : DdlConversion
//   original  : void DdlConversion::processTransformations(::mlir::Region& myRegion)
//   extract   : crustify-ddc/cpp/ddl.cpp:3091-3126
//   calls     : e177_dump, e183_dump, e275_processCondition

// crustify:todo: e325_convertDsc2Ddl
//   authority : ddc/ddl/ddl_conversion.cpp:2881  (627 body lines, level 2)
//   class     : DdlConversion
//   original  : void DdlConversion::convertDsc2Ddl(std::ostream& outputDdl)
//   extract   : crustify-ddc/cpp/ddl.cpp:3136-3763
//   calls     : e176_getTensor, e177_dump, e179_getAccessPatternAsStr, e183_dump, e277_getTensorAndAllocation

// crustify:todo: e345_exportToDdl
//   authority : ddc/ddl/ddl_conversion.cpp:38  (3 body lines, level 3)
//   class     : DdlConvertInterface
//   original  : void DdlConvertInterface::exportToDdl(std::ostream& outputDdl)
//   extract   : crustify-ddc/cpp/ddl.cpp:3795-3798
//   calls     : e325_convertDsc2Ddl

// crustify:todo: e346_processRegion
//   authority : ddc/ddl/ddl_conversion.cpp:2008  (26 body lines, level 3)
//   class     : DdlConversion
//   original  : void DdlConversion::processRegion(::mlir::Region& myRegion, dsc2::BlockNode* currParent)
//   extract   : crustify-ddc/cpp/ddl.cpp:3808-3835
//   calls     : e323_processOp

// crustify:todo: e347_matchDdl2Dsc
//   authority : ddc/ddl/ddl_conversion.cpp:2110  (442 body lines, level 3)
//   class     : DdlConversion
//   original  : bool DdlConversion::matchDdl2Dsc()
//   extract   : crustify-ddc/cpp/ddl.cpp:3845-4287
//   calls     : e021_getOpFuncName, e172_processTypes, e173_addInternalTensor, e175_getTensorProp, e177_dump, e183_dump, e187_clear, e272_print, e274_processDimensionOp, e278_checkMetaDimensions, e322_processPaddedDimensionOp

// crustify:todo: e364_parseDdl2Dsc
//   authority : ddc/ddl/ddl_conversion.cpp:2770  (54 body lines, level 4)
//   class     : DdlConversion
//   original  : void DdlConversion::parseDdl2Dsc()
//   extract   : crustify-ddc/cpp/ddl.cpp:4316-4370
//   calls     : e324_processTransformations, e346_processRegion

// crustify:todo: e372_selectAndParseDdlTemplate
//   authority : ddc/ddl/ddl_conversion.cpp:42  (59 body lines, level 5)
//   class     : DdlConversion
//   original  : bool DdlConversion::selectAndParseDdlTemplate()
//   extract   : crustify-ddc/cpp/ddl.cpp:4380-4439
//   calls     : e187_clear, e276_verifyDdlConstraints, e347_matchDdl2Dsc, e363_parseDdl, e364_parseDdl2Dsc

#[cfg(test)]
mod unit_tests {
    use std::collections::BTreeMap;

    use sys_arch_spec::arch_enums::SenComponent;

    use super::{
        ComputeOpIdx, DdlInterface, DimProp, ExprValue, GlobalLayoutRefs, InternalTensor,
        InternalTensorSite, LabeledDsTail, LdsSlot, StyledDims, TensorProp, TypeDefinition,
        add_internal_tensor, allocation_pad_type, pad_type_spelling, process_expression,
        process_types, tensor, tensor_prop, transfer_access_pattern,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
    use crate::formats::{Bits, DataFormat};
    use crate::generated::{
        AccessPattern, Attrs, NameId, Operand, PROGRAMS, Program, Stmt, StmtKind,
    };
    use crate::schedule::ddc::fold::{AllocId, ConstIdx, DataOrigin, NodeId, PadType};
    use crate::schedule::ddc::metadata::{
        Allocation, DataConnectSlot, DataTransfer, DdcMemory, ExternalStorage, MetaDimKind,
        Metadata, OpaqueOp, TransferAccessPattern, TransferEnd,
    };
    use crate::schedule::dsc2::LdsIdx;

    /// A program with hand-written statements, wearing the first vendored program's template so that
    /// the census enum is not restated here.
    fn synthetic(names: &'static [&'static str], stmts: &'static [Stmt]) -> Program {
        Program {
            template: PROGRAMS[0].template,
            op_func: "",
            bind: "",
            stmts,
            roles: &[],
            names,
        }
    }

    /// ⭐⭐ EVERY `ddl.type` OF EVERY VENDORED PROGRAM, AGAINST ITS OWN ATTRIBUTES — the widths come
    /// from `bit_width=` where stated and from IBM's table otherwise. Plus the negative: a name whose
    /// definition is NOT a type op is the *"Illegal ddl"* abort.
    #[test]
    fn parses_every_vendored_type_and_refuses_a_non_type() {
        let mut seen = 0;
        for program in PROGRAMS {
            for stmt in program.stmts {
                let Attrs::Type {
                    data_type,
                    bit_width,
                } = stmt.attrs
                else {
                    continue;
                };
                let format = DataFormat::from(data_type);
                let want = bit_width.map_or(format.bits(), |width| Bits(width as u32));
                assert_eq!(
                    process_types(program, stmt.results),
                    Some(vec![
                        TypeDefinition {
                            format,
                            bit_size: want
                        };
                        stmt.results.len()
                    ]),
                    "{} {}",
                    program.op_func,
                    program.bind
                );
                seen += 1;
            }
        }
        assert!(seen > 0, "no vendored program declares a `ddl.type`");

        let non_type = PROGRAMS
            .iter()
            .flat_map(|program| {
                program
                    .stmts
                    .iter()
                    .filter(|stmt| !matches!(stmt.attrs, Attrs::Type { .. }))
                    .filter_map(move |stmt| stmt.results.first().map(|name| (program, *name)))
            })
            .next()
            .expect("a vendored statement that is not a type op");
        assert_eq!(process_types(non_type.0, &[non_type.1]), None);
    }

    /// The whole `addInternalTensor` effect: the insert, the bumped last index, the interim append,
    /// the one tensor definition that follows the old last index, and every retagged slot.
    #[test]
    fn mints_an_internal_tensor_and_retags_the_old_last_index() {
        /// `labeledDs_` = [lds0, lds1], `tensor_definition_` = {%a: 1, %b: 1}, a tree naming lds1
        /// from a compute input and an allocate, and one metadata-owned allocate node on lds1.
        struct Site {
            last_lds: LdsIdx,
            tensors: BTreeMap<NameId, LdsIdx>,
            slots: BTreeMap<LdsSlot, LdsIdx>,
            allocations: BTreeMap<AllocId, LdsIdx>,
            inserted: Vec<InternalTensor>,
            interim: Vec<(ComputeOpIdx, LdsIdx)>,
        }
        impl InternalTensorSite for Site {
            fn labeled_ds_tail(&self) -> Option<LabeledDsTail> {
                Some(LabeledDsTail {
                    insert_position: LdsIdx(1),
                    last_lds: self.last_lds,
                })
            }
            fn set_last_lds_idx(&mut self, lds: LdsIdx) {
                self.last_lds = lds;
            }
            fn tensors_with_lds(&self, lds: LdsIdx) -> Vec<NameId> {
                self.tensors
                    .iter()
                    .filter(|(_, at)| **at == lds)
                    .map(|(name, _)| *name)
                    .collect()
            }
            fn set_tensor_lds(&mut self, tensor: NameId, lds: LdsIdx) {
                self.tensors.insert(tensor, lds);
            }
            fn insert_internal_tensor(&mut self, new: InternalTensor) {
                self.inserted.push(new);
            }
            fn add_interim_lds(&mut self, compute_op: ComputeOpIdx, lds: LdsIdx) {
                self.interim.push((compute_op, lds));
            }
            fn lds_slots(&self) -> Vec<LdsSlot> {
                let mut slots: Vec<LdsSlot> = self.slots.keys().copied().collect();
                slots.push(LdsSlot::OpaqueCompute(NodeId(7)));
                slots
            }
            fn slot_lds(&self, slot: LdsSlot) -> Option<LdsIdx> {
                self.slots.get(&slot).copied()
            }
            fn set_slot_lds(&mut self, slot: LdsSlot, lds: LdsIdx) {
                self.slots.insert(slot, lds);
            }
            fn allocation_lds(&self, alloc: AllocId) -> Option<LdsIdx> {
                self.allocations.get(&alloc).copied()
            }
            fn set_allocation_lds(&mut self, alloc: AllocId, lds: LdsIdx) {
                self.allocations.insert(alloc, lds);
            }
        }

        let mut site = Site {
            last_lds: LdsIdx(1),
            tensors: BTreeMap::from([(NameId(0), LdsIdx(1)), (NameId(1), LdsIdx(1))]),
            slots: BTreeMap::from([
                (LdsSlot::ComputeInput(NodeId(3), 0), LdsIdx(1)),
                (LdsSlot::ComputeInput(NodeId(3), 1), LdsIdx(0)),
                (LdsSlot::Allocate(NodeId(4)), LdsIdx(1)),
            ]),
            allocations: BTreeMap::from([(AllocId(9), LdsIdx(1))]),
            inserted: Vec::new(),
            interim: Vec::new(),
        };
        let mut metadata = Metadata::default();
        metadata.opaque_ops.insert(
            NodeId(7),
            OpaqueOp {
                lds_idx: Some(LdsIdx(1)),
                ..OpaqueOp::default()
            },
        );
        metadata.new_allocations.insert(
            DdcMemory::Lx,
            Allocation {
                lds_idx_and_alloc_node: BTreeMap::from([(LdsIdx(1), AllocId(9))]),
                comp_and_alloc_node: BTreeMap::from([(NodeId(3), AllocId(9))]),
                ..Allocation::default()
            },
        );
        metadata.prefilled_external_transfer_data_connects.insert(
            (LdsIdx(1), ExternalStorage::Lx),
            DataConnectSlot {
                transfer: NodeId(5),
                end: TransferEnd::Src,
            },
        );

        let new = add_internal_tensor(&mut site, &mut metadata, LdsIdx(0), ComputeOpIdx(2));

        assert_eq!(new, Some(LdsIdx(1)));
        assert_eq!(site.last_lds, LdsIdx(2));
        assert_eq!(
            site.inserted,
            vec![InternalTensor {
                lds: LdsIdx(1),
                name: "internal_tensor_lds1".to_string(),
                reference: LdsIdx(0),
            }]
        );
        assert_eq!(site.interim, vec![(ComputeOpIdx(2), LdsIdx(1))]);
        // ⛔ THE FIRST MATCH ONLY — `%b` keeps the old index, exactly as the reference's `break` does.
        assert_eq!(
            site.tensors,
            BTreeMap::from([(NameId(0), LdsIdx(2)), (NameId(1), LdsIdx(1))])
        );
        assert_eq!(
            site.slots,
            BTreeMap::from([
                (LdsSlot::ComputeInput(NodeId(3), 0), LdsIdx(2)),
                (LdsSlot::ComputeInput(NodeId(3), 1), LdsIdx(0)),
                (LdsSlot::Allocate(NodeId(4)), LdsIdx(2)),
            ])
        );
        assert_eq!(site.allocations, BTreeMap::from([(AllocId(9), LdsIdx(2))]));
        assert_eq!(metadata.opaque_ops[&NodeId(7)].lds_idx, Some(LdsIdx(2)));
        assert_eq!(
            metadata.new_allocations[&DdcMemory::Lx].lds_idx_and_alloc_node,
            BTreeMap::from([(LdsIdx(2), AllocId(9))])
        );
        assert!(
            metadata
                .prefilled_external_transfer_data_connects
                .contains_key(&(LdsIdx(2), ExternalStorage::Lx))
        );
    }

    /// `strtof`'s acceptance, including the trailing garbage it refuses.
    #[test]
    fn parses_an_expression_the_way_strtof_does() {
        assert_eq!(process_expression(" 0.125 "), Some(ExprValue(0.125)));
        assert_eq!(process_expression("1.5abc"), None);
        assert_eq!(process_expression(""), None);
    }

    /// An alias resolves to its one active tensor AND is memoised under its own name; two active
    /// tensors, and a definition that is not an alias, are both the *"Illegal ddl"* abort.
    #[test]
    fn resolves_and_memoises_one_active_aliased_tensor() {
        const ALIAS: &[Stmt] = &[Stmt {
            kind: StmtKind::AliasOneTensorOf,
            depth: 0,
            attrs: Attrs::Bare(StmtKind::AliasOneTensorOf),
            results: &[NameId(2)],
            operands: &[Operand::List(&[NameId(0), NameId(1)])],
            path: &[],
        }];
        let program = synthetic(&["%int8", "%fp16", "%alias"], ALIAS);

        let mut table = BTreeMap::from([(
            NameId(1),
            TensorProp {
                lds: Some(LdsIdx(4)),
            },
        )]);
        assert_eq!(
            tensor_prop(&program, &mut table, NameId(2)),
            Some(TensorProp {
                lds: Some(LdsIdx(4))
            })
        );
        assert_eq!(table[&NameId(2)].lds, Some(LdsIdx(4)));

        let mut both = BTreeMap::from([
            (
                NameId(0),
                TensorProp {
                    lds: Some(LdsIdx(3)),
                },
            ),
            (
                NameId(1),
                TensorProp {
                    lds: Some(LdsIdx(4)),
                },
            ),
        ]);
        assert_eq!(tensor_prop(&program, &mut both, NameId(2)), None);

        const PLAIN: &[Stmt] = &[Stmt {
            kind: StmtKind::Type,
            depth: 0,
            attrs: Attrs::Bare(StmtKind::Type),
            results: &[NameId(0)],
            operands: &[],
            path: &[],
        }];
        let not_a_tensor = synthetic(&["%t"], PLAIN);
        assert_eq!(
            tensor_prop(&not_a_tensor, &mut BTreeMap::new(), NameId(0)),
            None
        );
    }

    /// The three arms in their priority order, and the LAST match within each — the reference's own
    /// missing `break`.
    #[test]
    fn selects_the_last_matching_tensor_per_arm() {
        let tensors = BTreeMap::from([
            (
                NameId(0),
                TensorProp {
                    lds: Some(LdsIdx(2)),
                },
            ),
            (
                NameId(1),
                TensorProp {
                    lds: Some(LdsIdx(2)),
                },
            ),
        ]);
        let constants = BTreeMap::from([(NameId(5), ConstIdx(1)), (NameId(6), ConstIdx(1))]);
        let operands = BTreeMap::from([
            (NameId(8), SenComponent::Pe0),
            (NameId(9), SenComponent::Pe0),
        ]);

        let pick = |origin| tensor(&tensors, &constants, &operands, SenComponent::Pe0, origin);
        assert_eq!(
            pick(Some(DataOrigin::LabeledDs(LdsIdx(2)))),
            Some(NameId(1))
        );
        assert_eq!(
            pick(Some(DataOrigin::Constant(ConstIdx(1)))),
            Some(NameId(6))
        );
        assert_eq!(pick(None), Some(NameId(9)));
        assert_eq!(pick(Some(DataOrigin::LabeledDs(LdsIdx(7)))), None);
    }

    /// The dump, byte for byte, in both of its shapes: a header and an unset dim, and no header with
    /// a resolved one.
    #[test]
    fn dumps_a_dim_prop_the_way_llvm_errs_does() {
        const NAMES: &[&str] = &["%wrd", "%wrd_padded"];
        let program = synthetic(NAMES, &[]);

        let unset = DimProp {
            dim_candidates: vec![PrimaryDim::In, PrimaryDim::Out],
            ..DimProp::default()
        };
        assert_eq!(
            unset.dump(&program, "meta"),
            "\nmeta\n----\nDimProp: \n  PrimaryDimTypes = Invalid\n  DropDim         = F\n  \
             NonPaddedDim    = <<NULL VALUE>>\n  MetaDimKind     = unpadded\n  dimCandidates   = \
             [ in out]\n  NumRefsInGlobalLayouts = 0\n"
        );

        let resolved = DimProp {
            dim: Some(PrimaryDim::Kij),
            dim_candidates: Vec::new(),
            global_layout_refs: GlobalLayoutRefs(3),
            drop_dim: true,
            non_padded_dim: Some(NameId(0)),
            meta_dim_kind: MetaDimKind::PadFront,
        };
        assert_eq!(
            resolved.dump(&program, ""),
            "\nDimProp: \n  PrimaryDimTypes = kij\n  DropDim         = T\n  NonPaddedDim    = \
             %wrd\n  MetaDimKind     = pad_front\n  dimCandidates   = []\n  \
             NumRefsInGlobalLayouts = 3\n"
        );
    }

    /// The stated dim answers its pair; an unstated one is the abort.
    #[test]
    fn answers_the_access_pattern_of_a_stated_dim() {
        let mut transfer = DataTransfer::default();
        transfer.access_pattern_per_dim.insert(
            PrimaryDim::Ij,
            TransferAccessPattern {
                from: PadType::NoPad,
                to: PadType::PaddedWZeroPad,
            },
        );
        assert_eq!(
            transfer.access_pattern(PrimaryDim::Ij),
            Some(TransferAccessPattern {
                from: PadType::NoPad,
                to: PadType::PaddedWZeroPad
            })
        );
        assert_eq!(transfer.access_pattern(PrimaryDim::X), None);
    }

    /// ⭐ IBM'S `padTypeToString`, ROW FOR ROW, through the string entry 179 builds from it.
    #[test]
    fn spells_every_access_pattern_pair() {
        for (pad, spelling) in [
            (PadType::NoPad, "nopad"),
            (PadType::LoweredPadded, "lowered_padded"),
            (PadType::PaddedNoZeroPad, "padded_nozeropad"),
            (PadType::PaddedWZeroPad, "padded_wzeropad"),
            (PadType::PaddedFullSpan, "padded_fullspan"),
            (
                PadType::PaddedFullSpanWUnneeded,
                "padded_fullspan_wunneeded",
            ),
        ] {
            assert_eq!(pad_type_spelling(pad), spelling);
            let mut transfer = DataTransfer::default();
            transfer
                .access_pattern_per_dim
                .insert(PrimaryDim::Y, TransferAccessPattern { from: pad, to: pad });
            assert_eq!(
                transfer.access_pattern_spelling(PrimaryDim::Y),
                Some(format!("{spelling}-to-{spelling}"))
            );
        }
        assert_eq!(
            DataTransfer::default().access_pattern_spelling(PrimaryDim::Y),
            None
        );
    }
    /// ⭐ EVERY VENDORED `access_pattern_style=` LIST, PAIRED TO AS MANY DIMS AS IT STATES — plus the
    /// three *"Illegal ddl"* aborts and the *"Nothing to process"* empty.
    #[test]
    fn an_access_pattern_list_pairs_one_style_per_dim() {
        let dims = [NameId(7), NameId(8), NameId(9)];
        let mut seen = 0;
        for program in PROGRAMS {
            for stmt in program.stmts {
                let Attrs::DataTransfer {
                    access_pattern: styles,
                    ..
                } = stmt.attrs
                else {
                    continue;
                };
                if styles.is_empty() {
                    continue;
                }
                let paired = StyledDims::stated(&dims[..styles.len()], styles)
                    .expect("a vendored list is one style per dim");
                assert_eq!(paired.pairs().len(), styles.len());
                // ONE style covers every dim, however many there are.
                assert_eq!(
                    StyledDims::stated(&dims, &styles[..1])
                        .expect("a single style broadcasts")
                        .pairs(),
                    &[
                        (dims[0], styles[0]),
                        (dims[1], styles[0]),
                        (dims[2], styles[0])
                    ]
                );
                // The three aborts.
                assert_eq!(StyledDims::stated(&dims, &[] as &[AccessPattern]), None);
                assert_eq!(StyledDims::stated(&[], styles), None);
                assert_eq!(
                    StyledDims::stated(&dims[..2], &[styles[0], styles[0], styles[0]]),
                    None
                );
                seen += 1;
            }
        }
        assert!(seen > 0, "no vendored transfer states an access pattern");
        // Nothing stated at all is not an abort.
        assert_eq!(
            StyledDims::stated(&[], &[] as &[AccessPattern])
                .expect("neither dims nor styles is nothing to process")
                .pairs(),
            &[]
        );
    }

    /// ⭐ EVERY VENDORED `access_pattern_style=` AGAINST ITS OWN SPELLING: the two pad types this
    /// hands back must re-spell as `<src>-to-<dst>`, which is the split the reference performs.
    #[test]
    fn every_vendored_access_pattern_recomposes_from_its_two_pad_types() {
        let mut seen = 0;
        for program in PROGRAMS {
            for stmt in program.stmts {
                let Attrs::DataTransfer { access_pattern, .. } = stmt.attrs else {
                    continue;
                };
                for pattern in access_pattern {
                    let pair = transfer_access_pattern(*pattern);
                    assert_eq!(
                        format!(
                            "{}-to-{}",
                            pad_type_spelling(pair.from),
                            pad_type_spelling(pair.to)
                        ),
                        pattern.spelling()
                    );
                    seen += 1;
                }
            }
        }
        assert!(seen > 0, "no vendored transfer states an access pattern");
    }

    /// ⭐ EVERY VENDORED `padding_type=` AGAINST ITS OWN SPELLING.
    #[test]
    fn every_vendored_padding_type_is_the_pad_type_it_spells() {
        let mut seen = 0;
        for program in PROGRAMS {
            for stmt in program.stmts {
                let Attrs::Allocate { padding, .. } = stmt.attrs else {
                    continue;
                };
                for pad in padding {
                    assert_eq!(pad_type_spelling(allocation_pad_type(*pad)), pad.spelling());
                    seen += 1;
                }
            }
        }
        assert!(seen > 0, "no vendored allocation states a padding type");
    }

    /// The three framing lines, and one line per dim in `PrimaryDimTypes` order.
    #[test]
    fn a_transfer_dumps_its_access_patterns_in_dim_order() {
        let mut transfer = DataTransfer::default();
        assert_eq!(
            transfer.dump(),
            "\n[Ddl::DataTransfer]\n--------------------------\n  access-patterns per dimension: \n"
        );
        transfer.access_pattern_per_dim.insert(
            PrimaryDim::Y,
            TransferAccessPattern {
                from: PadType::NoPad,
                to: PadType::LoweredPadded,
            },
        );
        transfer.access_pattern_per_dim.insert(
            PrimaryDim::Out,
            TransferAccessPattern {
                from: PadType::PaddedWZeroPad,
                to: PadType::PaddedFullSpan,
            },
        );
        assert_eq!(
            transfer.dump(),
            "\n[Ddl::DataTransfer]\n--------------------------\n  access-patterns per dimension: \
             \n    Dim out access-pattern (padded_wzeropad to padded_fullspan)\
             \n    Dim y access-pattern (nopad to lowered_padded)\n"
        );
    }

    /// ⭐ EVERY VENDORED `dim_property=`, WHICH MUST LAND ON THE KIND THAT SPELLS IT BACK — and the
    /// absent one, which is `unpadded`.
    #[test]
    fn every_vendored_dim_property_sets_the_kind_that_spells_it() {
        let mut seen = 0;
        for program in PROGRAMS {
            for stmt in program.stmts {
                let Attrs::Dimension { property } = stmt.attrs else {
                    continue;
                };
                let mut prop = DimProp::default();
                prop.set_meta_dim_kind(property);
                assert_eq!(
                    prop.meta_dim_kind.label(),
                    property.map_or("unpadded", |property| property.spelling())
                );
                assert_eq!(prop.is_meta_dim(), property.is_some());
                seen += 1;
            }
        }
        assert!(seen > 0, "no vendored program declares a `ddl.dimension`");
    }

    /// Neither `Unpadded` nor `Padded` is a meta dim; every other kind is.
    #[test]
    fn only_the_six_meta_kinds_are_meta_dims() {
        for (kind, want) in [
            (MetaDimKind::Unpadded, false),
            (MetaDimKind::Padded, false),
            (MetaDimKind::PadFront, true),
            (MetaDimKind::PadBack, true),
            (MetaDimKind::PadValid, true),
            (MetaDimKind::WindowDim, true),
            (MetaDimKind::Stride, true),
            (MetaDimKind::Dilation, true),
        ] {
            let prop = DimProp {
                meta_dim_kind: kind,
                ..DimProp::default()
            };
            assert_eq!(prop.is_meta_dim(), want, "{}", kind.label());
        }
    }

    /// A padded dim answers with the unpadded one it names; one that names nothing is the [`None`].
    #[test]
    fn a_padded_dim_resolves_to_the_unpadded_dim_it_names() {
        let mut interface = DdlInterface::default();
        interface.dim_association.insert(
            NameId(1),
            DimProp {
                dim: Some(PrimaryDim::I),
                ..DimProp::default()
            },
        );
        interface.dim_association.insert(
            NameId(2),
            DimProp {
                meta_dim_kind: MetaDimKind::Padded,
                non_padded_dim: Some(NameId(1)),
                ..DimProp::default()
            },
        );
        interface.dim_association.insert(
            NameId(3),
            DimProp {
                meta_dim_kind: MetaDimKind::Padded,
                ..DimProp::default()
            },
        );
        assert_eq!(
            interface
                .non_padded_dim_prop(NameId(2))
                .map(|prop| prop.dim),
            Some(Some(PrimaryDim::I))
        );
        assert_eq!(
            interface
                .non_padded_dim_prop(NameId(1))
                .map(|prop| prop.dim),
            Some(Some(PrimaryDim::I))
        );
        assert!(interface.non_padded_dim_prop(NameId(3)).is_none());
        // ⛔ ASKING INSERTS: an unknown dim gets a fresh `DimProp` and is in the map afterwards.
        assert!(interface.non_padded_dim_prop(NameId(9)).is_some());
        assert!(interface.dim_association.contains_key(&NameId(9)));
    }

    /// Every map is dropped, and the reset is a REBUILD: a re-inserted dim gets its candidates back.
    #[test]
    fn clearing_the_interface_drops_every_mapping() {
        let mut interface = DdlInterface::default();
        interface.dim_association.insert(
            NameId(1),
            DimProp {
                dim_candidates: Vec::new(),
                ..DimProp::default()
            },
        );
        interface.type_definition.insert(
            NameId(2),
            TypeDefinition {
                format: DataFormat::Sen169Fp16,
                bit_size: Bits(16),
            },
        );
        interface.tensor_definition.insert(
            NameId(3),
            TensorProp {
                lds: Some(LdsIdx(4)),
            },
        );
        interface.clear();
        assert_eq!(interface, DdlInterface::default());
        assert!(
            !interface
                .dim_association
                .entry(NameId(1))
                .or_default()
                .dim_candidates
                .is_empty()
        );
    }
}
