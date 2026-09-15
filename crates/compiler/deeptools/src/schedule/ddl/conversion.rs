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

// ⭐ USES FOR ENTRIES 172-187 AND 274-279. Union these into this file's top block when its other
// entries land.
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Arch, Elements};
use crate::bridges::superdsc_to_dataflow_ir::control_flow::{CondOp, CondValType};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    DimSet, Extent, PrimaryDim, SliceElems, StickDims, StickPart, cumulative_stick_sizes,
    stick_sizes,
};
use crate::formats::{Bits, DataFormat};
use crate::generated::{
    AccessPattern, Attrs, Buffers, ComputeType, ConstName, DataConnect, DatastageProperty,
    DimProperty, LoopLabel, MaxUnroll, Memory, Mode, NameId, Operand, Operand as DdlOperand,
    PaddingType, Program, Stmt, StmtKind, Strategy, SyncSignal, Template, Unit, Via as DdlVia,
    ddl_templates,
};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{AllocId, BlockId, ConstIdx, DataOrigin, NodeId, PadType};
use crate::schedule::ddc::metadata::{
    DataTransfer, DatastageId, DdcMemory, ExternalStorage, MetaDimKind, Metadata, OpaqueOp,
    TransferAccessPattern, TransferEnd,
};
use crate::schedule::ddc::transformation::{DsType, LoopId, Scale};
use crate::schedule::ddc::transformation_util::StageName;
use crate::schedule::ddc::transformation_util::{LoopCond, LoopCondComposite, is_memory};
use crate::schedule::ddc::v1::{CoreClSet, Ln32};
use crate::schedule::ddl::ops::DdlComputeType;
use crate::schedule::ddl::{DdlModuleOp, DdlSource};
use crate::schedule::dsc2::{
    AllocLayout, AllocPlacement, AllocateNode, BlockNode, ComputeMask, ComputeNode,
    CondOp as DscCondOp, ConditionNode, DataInfo, Dsts, Hops, InstrAttribute, LdsIdx, LoopBound,
    LoopCond as DscLoopCond, LoopCondComposite as DscLoopCondComposite, LoopDim, LoopNode,
    MaxDimSize, NodeName, NumBuffers, NumChunks, Operand as DscOperand, PackIndex, Padding,
    Repetition, ReplicationFactor, SchedNode, ScheduleTree, StartAddress, SyncDirection, SyncNode,
    SyncStrength, SyncUnits, TransferNode, TransferPadding, Unroll, WordLength, generic_comp,
};
use crate::schedule::l3::dsc::{
    CoreCount, CoreletsUsed, DesignSpaceConfig, EmptyStage, PadSizes, WkSlice, WkSliceId,
};
use crate::units::{Core, Corelet, NumFolds, rows_of};

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
/// interim list it appends to, and the tree slots it retags.
///
/// ⭐ EVERY MUTATION THE REFERENCE PERFORMS IS ONE METHOD HERE; WHICH to call, in what order and
/// with what value stays in the port. The `oldNewLdsPtrs` fixup is not here: re-seating raw pointers
/// after a vector insert is the mechanism for reaching an LDS, and an index does not need it.
pub trait InternalTensorSite {
    /// `dsc.labeledDs_`'s tail, absent where that vector is empty.
    fn labeled_ds_tail(&self) -> Option<LabeledDsTail>;
    /// `++dsc.labeledDs_.back().ldsIdx_`.
    fn set_last_lds_idx(&mut self, lds: LdsIdx);
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
/// ⛔⛔ REVIEWED: THE `tensor_definition_` RETAG WAS ROUTED THROUGH THE SITE, WHICH CANNOT REACH IT —
/// the map `:499-504` bumps is the one `matchDdl2Dsc` prunes each LDS from (`:2336-2341`), and every
/// caller threads it as `interface`, a borrow DISJOINT from `site`, so the effect was DROPPED.
/// ⛔ TRAP: the retag `break`s after the FIRST match over an `unordered_map`, so WHICH tensor follows
/// the old last index is ARBITRARY in the reference; here it is the lowest [`NameId`].
/// ⛔ [`None`] where `labeledDs_` is empty — see [`LabeledDsTail`].
pub fn add_internal_tensor<S: InternalTensorSite + ?Sized>(
    site: &mut S,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    reference: LdsIdx,
    compute_op: ComputeOpIdx,
) -> Option<LdsIdx> {
    let tail = site.labeled_ds_tail()?;
    let old_last = tail.last_lds;
    let new_last = LdsIdx(old_last.0 + 1);
    site.set_last_lds_idx(new_last);
    if let Some(prop) = interface
        .tensor_definition
        .values_mut()
        .find(|prop| prop.lds == Some(old_last))
    {
        prop.lds = Some(new_last);
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
/// ⭐ THE LAST EIGHT MEMBERS LANDED WITH ENTRIES 322-325, which are the units that decide their
/// element types. [`Self::clear`] resets whatever the struct holds, so it stayed correct while they
/// were absent.
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
    /// `operation_definition_`, keyed by the `ddl.operation_bind` it describes.
    pub operation_definition: BTreeMap<NameId, OperationProp>,
    /// `resolvedConditions_`, which [`process_condition`] memoises into.
    pub resolved_conditions: BTreeMap<NameId, CondProp>,
    /// `loop_labels_` — the `dsc2::LoopNode` each `label=` names, which is where
    /// `ddl_conversion.cpp:1065` registers a minted loop.
    pub loop_labels: BTreeMap<LoopLabel, NodeId>,
    /// `core_chunk_loop_label_`, absent for the reference's empty string.
    pub core_chunk_loop_label: Option<LoopLabel>,
    /// `datastage_definition_` — which `metadata_.datastages_` entry a `ddl.datastage` or
    /// `ddl.get_external_datastage` names.
    pub datastage_definition: BTreeMap<NameId, DatastageId>,
    /// `ext_constant_definition_` — the `constantInfo_` index a `ddl.define_constant`,
    /// `ddl.get_external_constant` or `ddl.alias_one_constant_of` resolves to.
    pub ext_constant_definition: BTreeMap<NameId, ConstIdx>,
    /// `alloc_storage_` — the `dsc2::AllocateNode` a `ddl.allocate` result minted, so a second
    /// `ddl.unit` naming the same allocation reuses it rather than minting a rival.
    pub alloc_storage: BTreeMap<NameId, AllocId>,
    /// `transfer_acc_pat_dims_` — the DDL dims a transfer's `access_pattern_style=` list applies to,
    /// IN THE OP'S OWN ORDER, which is the order entry 325 writes them back in.
    pub transfer_acc_pat_dims: BTreeMap<NodeId, Vec<NameId>>,
    /// `operand_constant_tensor_` — the unit a `ddl.operand_constant` is bound to, which is how a
    /// compute operand naming no tensor and no constant still resolves.
    pub operand_constant_tensor: BTreeMap<NameId, SenComponent>,
    /// `region2blocks_` — the block that was `currParent` when a region was opened, which is where an
    /// allocation declared inside it lands.
    pub region2blocks: BTreeMap<RegionId, NodeName>,
    /// `sync_definitions_`, keyed by `signal_name=`.
    pub sync_definitions: BTreeMap<SyncSignal, SyncProp>,
    /// `coreToCore_definitions_`, keyed by the `ddl.core_to_core_communication` that states the ring.
    pub core_to_core_definitions: BTreeMap<NameId, CoreToCoreProp>,
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

// ⭐ TYPES FOR ENTRIES 274-279 — the two `DdlInterface` sub-structures those entries fill, the
// allocation seam entry 277 reads through, and the constraint form entry 276 verifies.

/// ONE BOUND DDL OPERATION'S PROPERTIES — `DdlInterface::OperationProp`
/// (`ddc/ddl/ddl_conversion.h:342-347`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OperationProp {
    /// `computeOpIdx_`, whose `-1` is an operation with no compute op assigned yet.
    pub compute_op: Option<ComputeOpIdx>,
    /// `coreClCond_`. ⭐ EMPTY MEANS UNCONDITIONALLY ACTIVE, which is what [`process_condition`]
    /// reads it for.
    pub core_cl_cond: CoreClSet,
}

/// ONE RESOLVED DDL CONDITION — `DdlInterface::CondProp` (`ddc/ddl/ddl_conversion.h:349-358`).
///
/// ⛔ `resolvedValue_` HAS NO INITIALISER and `isResolvedToBool_` is the flag that says whether
/// reading it is defined; the two are ONE [`Option`] here, so the undefined read is unspellable.
/// ⭐ THE THREE CARRIERS ARE ALTERNATIVES — a bool, a loop condition, or a core/corelet set — and
/// [`process_condition`]'s tail is what makes "none of them" mean `Some(false)`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CondProp {
    /// `resolvedValue_` behind `isResolvedToBool_`.
    pub resolved: Option<bool>,
    /// `loopCond_`.
    pub loop_cond: LoopCondComposite,
    /// `coreClCond_`.
    pub core_cl_cond: CoreClSet,
}

/// A DDL TENSOR AND THE `ddl.allocate` THAT PLACES IT — the `std::pair<Value, Value>` entry 277
/// answers, whose two null `Value`s are the two [`None`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TensorAndAllocation {
    /// `allocateop.getTensor()`, or the operand-constant tensor bound to the unit.
    pub tensor: Option<EmittedTensor>,
    /// `allocateop.getResult()`.
    pub allocate: Option<AllocOp>,
}

/// WHERE A DSC PLACES ONE DATASTREAM END — `labeledDs_.at(lds).memOrg_.at(unit).allocateNode_` and
/// `constantInfo_.at(constant).allocations_.at(unit)`, the two `.at()` chains entry 277 walks.
///
/// ⛔ A TRAIT BECAUSE NEITHER CHAIN IS IN [`DesignSpaceConfig`] YET: `memOrg_`'s allocate node and
/// `constantInfo_` both arrive with later units, and [`None`] is either `.at()` throwing.
pub trait AllocationSite {
    /// `dsc.labeledDs_.at(lds).memOrg_.at(unit).allocateNode_`.
    fn lds_allocation(&self, lds: LdsIdx, unit: SenComponent) -> Option<AllocId>;

    /// `dsc.constantInfo_.at(constant).allocations_.at(unit)`.
    fn constant_allocation(&self, constant: ConstIdx, unit: SenComponent) -> Option<AllocId>;
}

/// HOW A `ddl.constraint` COMPARES — `cmp=`, whose every other spelling is the reference's
/// *"\"cmp\" type not yet supported"* abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintCmp {
    /// `"equal"`.
    Equal,
    /// `"less"`.
    Less,
}

/// ONE `ddl.constraint`'S FORM — the attribute combinations `verifyDdlConstraints` tests, in the
/// order it tests them.
///
/// ⛔ A PARAMETER AND NOT A CENSUS READ: `build.rs` keeps `ddl.constraint` as
/// `Attrs::Bare(StmtKind::Constraint)` and drops its attributes, so the form has to be stated by
/// whoever walks the module.
/// ⛔ THE TWO *"Missing \"value\" attribute"* ABORTS AND *"Unsupported constraint type/format"* ARE
/// UNSPELLABLE: every arm that reads a value carries one, and there is no arm for no attribute at
/// all. A `relative_op_order=false` is likewise not a variant — the reference falls through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdlConstraint {
    /// `min_num_cores=`.
    MinNumCores(CoreCount),
    /// `min_num_valid=` / `max_num_valid=`, already at their `value_or` defaults.
    NumValid {
        /// `min_num_valid=`, defaulting to ZERO.
        min: u32,
        /// `max_num_valid=`, defaulting to ONE HUNDRED and not to unbounded.
        max: u32,
    },
    /// `relative_op_order=true`.
    RelativeOpOrder,
    /// `property=` + `dim_idx=` + `cmp="equal"` + `value=`.
    StickSizeAt {
        /// `property=="slice"`.
        slice: bool,
        /// `dim_idx=`.
        dim_idx: usize,
        /// `value=`.
        value: Elements,
    },
    /// `property=` + `cmp="equal"` with no `dim_idx=`.
    StickSizesAgree {
        /// `property=="slice"`.
        slice: bool,
    },
    /// `cmp=` + `value=` with no `property=`.
    DimSize {
        /// `cmp=`.
        cmp: ConstraintCmp,
        /// `value=`.
        value: Extent,
    },
}

/// Every SSA name an operand list states, in `getOperands()` order, [`None`] for a
/// `ddl.operation_bind` this walk did not activate.
fn operand_names(operands: &[Operand]) -> Vec<Option<NameId>> {
    operands
        .iter()
        .flat_map(|operand| match operand {
            Operand::One(name) => vec![Some(*name)],
            Operand::List(names) => names.iter().copied().map(Some).collect(),
            Operand::OtherBind => vec![None],
        })
        .collect()
}

/// `op->getOperand(at)`, [`None`] where that position is not one plain name.
fn operand_at(operands: &[Operand], at: usize) -> Option<NameId> {
    match operands.get(at)? {
        Operand::One(name) => Some(*name),
        Operand::List(_) | Operand::OtherBind => None,
    }
}

/// `EnumsConversion::stringToSenComponents` over the census's own `memory=` spellings.
///
/// ⛔ TOTAL, WHICH RETIRES THE *"Unrecognized memory"* ABORT: `dsc2::memories`
/// (`dsc/dscdefn.cpp:142`) names all eight of them, so neither the lookup nor the membership test
/// that guard it can fail.
#[must_use]
pub const fn memory_component(memory: Memory) -> SenComponent {
    match memory {
        Memory::L0 => SenComponent::L0,
        Memory::L0scale => SenComponent::L0Scale,
        Memory::Lx => SenComponent::Lx,
        Memory::Pelrf => SenComponent::Pelrf,
        Memory::Ptarf => SenComponent::Ptarf,
        Memory::Ptxrf => SenComponent::Ptxrf,
        Memory::Sfplrf => SenComponent::Sfplrf,
        Memory::Sfpstate => SenComponent::Sfpstate,
    }
}

/// Replaces: e274_processDimensionOp
///
/// THE DIM'S PROPERTIES, MINTED FROM ITS `ddl.dimension` on first sight and memoised after.
///
/// ⛔ [`None`] IS *"Illegal ddl file"* — a name whose definition is not a `ddl.dimension` at all.
/// ⛔ ASKING IS A MUTATION (the reference's `operator[]`), so the entry exists afterwards with its
/// full candidate list. ⛔ `setMetaDimKind`'s OWN `false` RETURN IS UNSPELLABLE, which is
/// [`DimProp::set_meta_dim_kind`]'s own trap: `dim_property=` is a censused [`DimProperty`].
pub fn process_dimension_op<'i>(
    program: &Program,
    interface: &'i mut DdlInterface,
    dim: NameId,
) -> Option<&'i mut DimProp> {
    if interface.dim_association.contains_key(&dim) {
        return interface.dim_association.get_mut(&dim);
    }
    let Attrs::Dimension { property } = program.definition(dim)?.attrs else {
        return None;
    };
    let prop = interface.dim_association.entry(dim).or_default();
    prop.set_meta_dim_kind(property);
    Some(prop)
}

/// THE SIX COMPARISONS AGAINST ZERO a dropped dim's condition reduces to
/// (`ddc/ddl/ddl_conversion.cpp:282-294`), and [`None`] for the *"Condition operator not
/// supported"* abort.
///
/// ⛔ ONLY [`CondOp::Eq`] AND [`CondOp::Ne`] ARE REACHABLE from the census, which narrows
/// `condition=` to `"eq"`/`"ne"` and `value_expr=` to `"first"`/`"last"`.
const fn resolves_true(op: CondOp, value: i64) -> Option<bool> {
    match op {
        CondOp::Eq => Some(value == 0),
        CondOp::Ne => Some(value != 0),
        CondOp::Le => Some(value <= 0),
        CondOp::Lt => Some(value < 0),
        CondOp::Ge => Some(value >= 0),
        CondOp::Gt => Some(value > 0),
        CondOp::Toggle | CondOp::Always | CondOp::Never | CondOp::Const | CondOp::Default => None,
    }
}

/// THE CORELET COMPLEMENT `ddl.condition_not` TAKES over `coreIdsUsed_`
/// (`ddc/ddl/ddl_conversion.cpp:328-343`).
///
/// ⛔ TRAP: IT NEVER TOUCHES A CORE OUTSIDE `coreIdsUsed_`, so a set naming one keeps that entry
/// UNCOMPLEMENTED — and the entry it inserts for a core the set does not name hardcodes corelets 0
/// and 1 instead of the corelet count.
/// ⛔ [`None`] IS `numCoreletsUsed_DSC2_` UNSET, where the reference loops `cl < -1` and complements
/// nothing, or a count past the arch's corelets per core.
fn complement_corelets(dsc: &DesignSpaceConfig, set: &mut CoreClSet) -> Option<()> {
    let corelets = dsc.corelets_used_dsc2?.get();
    for core in dsc.core_ids_used.iter() {
        match set.0.get_mut(&core) {
            Some(present) => {
                for index in 0..corelets {
                    let corelet = Corelet::checked(index)?;
                    if !present.remove(&corelet) {
                        present.insert(corelet);
                    }
                }
                if present.is_empty() {
                    set.0.remove(&core);
                }
            }
            None => {
                let fresh = set.0.entry(core).or_default();
                fresh.insert(Corelet::checked(0)?);
                if corelets > 1 {
                    fresh.insert(Corelet::checked(1)?);
                }
            }
        }
    }
    Some(())
}

/// [`process_condition`] of one operand, plus the reference's answer for a `ddl.operation_bind` this
/// walk did not activate: an operation absent from `operation_definition_` resolves to FALSE.
fn operand_condition(
    program: &Program,
    interface: &mut DdlInterface,
    metadata: &Metadata,
    dsc: &DesignSpaceConfig,
    operand: Option<NameId>,
) -> Option<CondProp> {
    match operand {
        Some(name) => process_condition(program, interface, metadata, dsc, name),
        None => Some(CondProp {
            resolved: Some(false),
            ..CondProp::default()
        }),
    }
}

/// Replaces: e275_processCondition
///
/// RESOLVES ONE `ddl.condition*` TREE to a bool, a two-level OR-of-ANDs loop condition or a
/// core/corelet set, memoising the answer under the condition's own name.
///
/// ⛔ [`None`] IS EVERY *"Illegal ddl"*: an inadmissible op kind, an unassigned dim, an empty or
/// unknown label, a composition that is not two-level, an and/or mixing the two carriers.
/// ⭐ A TREE FILLING NO CARRIER IS FALSE. ⛔ TRAP: the reference ALIASES the memo across inserts.
pub fn process_condition(
    program: &Program,
    interface: &mut DdlInterface,
    metadata: &Metadata,
    dsc: &DesignSpaceConfig,
    cond: NameId,
) -> Option<CondProp> {
    if let Some(memo) = interface.resolved_conditions.get(&cond) {
        return Some(memo.clone());
    }
    let stmt = *program.definition(cond)?;
    let mut mine = CondProp::default();
    match stmt.kind {
        StmtKind::OperationBind => match interface.operation_definition.get(&cond) {
            None => mine.resolved = Some(false),
            Some(prop) if prop.core_cl_cond.0.is_empty() => mine.resolved = Some(true),
            Some(prop) => mine.core_cl_cond = prop.core_cl_cond.clone(),
        },
        StmtKind::GetExternalDataTransferAllocation => {
            let Attrs::ExternalAllocation { memory, .. } = stmt.attrs else {
                return None;
            };
            let tensor = operand_at(stmt.operands, 0)?;
            let prop = tensor_prop(program, &mut interface.tensor_definition, tensor)?;
            let lds = dsc.labeled_ds.at(prop.lds?)?;
            mine.resolved = Some(lds.pinning().names(memory_component(memory)));
        }
        StmtKind::Condition => {
            let Attrs::Condition {
                loop_label,
                last,
                negated,
            } = stmt.attrs
            else {
                return None;
            };
            let label = loop_label?;
            let op = if negated { CondOp::Ne } else { CondOp::Eq };
            let (drop_dim, dim) = {
                let prop = interface
                    .dim_association
                    .get(&operand_at(stmt.operands, 0)?)?;
                (prop.drop_dim, prop.dim)
            };
            if drop_dim {
                // A dropped dim is a loop of size one, so its iterator is 0 on every comparison.
                mine.resolved = resolves_true(op, 0);
            } else {
                let loop_node = if Some(label) == interface.core_chunk_loop_label {
                    *metadata.dim_to_core_chunk_loops.get(&dim?)?.first()?
                } else {
                    *interface.loop_labels.get(&label)?
                };
                mine.loop_cond.or_of_ands.push(vec![LoopCond {
                    loop_node,
                    dim: dim?,
                    op,
                    against: if last {
                        CondValType::Last
                    } else {
                        CondValType::First
                    },
                }]);
            }
        }
        StmtKind::ConditionNot => {
            let operand = *operand_names(stmt.operands).first()?;
            mine = operand_condition(program, interface, metadata, dsc, operand)?;
            if let Some(value) = mine.resolved {
                mine.resolved = Some(!value);
            } else if !mine.loop_cond.or_of_ands.is_empty() {
                mine.loop_cond.negated = !mine.loop_cond.negated;
            } else {
                complement_corelets(dsc, &mut mine.core_cl_cond)?;
            }
        }
        StmtKind::ConditionAnd | StmtKind::ConditionOr => {
            let is_and = matches!(stmt.kind, StmtKind::ConditionAnd);
            let mut initialised = false;
            for operand in operand_names(stmt.operands) {
                let theirs = operand_condition(program, interface, metadata, dsc, operand)?;
                // An AND meeting a false, or an OR meeting a true, is decided: both carriers drop.
                if theirs.resolved == Some(!is_and) {
                    mine = CondProp {
                        resolved: Some(!is_and),
                        ..CondProp::default()
                    };
                    break;
                }
                if !initialised {
                    mine = theirs;
                    initialised = true;
                    continue;
                }
                if !theirs.loop_cond.or_of_ands.is_empty() {
                    if !mine.core_cl_cond.0.is_empty() {
                        return None;
                    }
                    if mine.resolved.is_some() {
                        mine = theirs;
                        continue;
                    }
                    if is_and {
                        if mine.loop_cond.or_of_ands.len() != 1
                            || mine.loop_cond.negated
                            || theirs.loop_cond.or_of_ands.len() != 1
                            || theirs.loop_cond.negated
                        {
                            return None;
                        }
                        let terms = theirs.loop_cond.or_of_ands.into_iter().next()?;
                        mine.loop_cond.or_of_ands.first_mut()?.extend(terms);
                    } else {
                        if mine.loop_cond.negated || theirs.loop_cond.negated {
                            return None;
                        }
                        mine.loop_cond
                            .or_of_ands
                            .extend(theirs.loop_cond.or_of_ands);
                    }
                } else if !theirs.core_cl_cond.0.is_empty() {
                    if !mine.loop_cond.or_of_ands.is_empty() {
                        return None;
                    }
                    if mine.resolved.is_some() {
                        mine = theirs;
                        continue;
                    }
                    if is_and {
                        // ⛔ THE REFERENCE ERASES FROM THE MAP IT IS ITERATING; this retains.
                        mine.core_cl_cond.0.retain(|core, corelets| {
                            let Some(other) = theirs.core_cl_cond.0.get(core) else {
                                return false;
                            };
                            corelets.retain(|corelet| other.contains(corelet));
                            !corelets.is_empty()
                        });
                    } else {
                        for (core, corelets) in &theirs.core_cl_cond.0 {
                            mine.core_cl_cond
                                .0
                                .entry(*core)
                                .or_default()
                                .extend(corelets);
                        }
                    }
                }
            }
        }
        _ => return None,
    }
    if mine.resolved.is_none()
        && mine.loop_cond.or_of_ands.is_empty()
        && mine.core_cl_cond.0.is_empty()
    {
        mine.resolved = Some(false);
    }
    interface.resolved_conditions.insert(cond, mine.clone());
    Some(mine)
}

/// THE STORAGE A REGISTER-FILE COMPONENT COUNTS AS (`ddc/ddl/ddl_conversion.cpp:2591-2596`) — the
/// five folds `verifyDdlConstraints` applies before comparing against `pinnedComponent()`.
///
/// ⛔ THE `PESTATE` FOLD IS UNREACHABLE from the census, which states no `pestate` memory; the
/// `SFPSTATE` one is reachable.
const fn register_storage(component: SenComponent) -> SenComponent {
    match component {
        SenComponent::Pelrf | SenComponent::Pestate => SenComponent::Pe,
        SenComponent::Sfplrf | SenComponent::Sfpstate => SenComponent::Sfp,
        SenComponent::Ptxrf => SenComponent::Pt,
        other => other,
    }
}

/// `getStickSizes(dsType, isSlice, !isSlice)` — one flag or the other, never both and never neither,
/// so [`StickPart::Whole`] is unreachable from a `ddl.constraint`.
fn stick_extents<A: Arch>(
    dsc: &DesignSpaceConfig,
    ds_type: DsType,
    slice: bool,
) -> Option<Vec<(PrimaryDim, Elements)>> {
    let info = dsc.primary_ds_info.get(&ds_type)?;
    let elems = SliceElems::per_stick::<A>(&info.stick)?;
    let part = if slice {
        StickPart::WithinSlice(elems)
    } else {
        StickPart::CrossSlice(elems)
    };
    Some(stick_sizes(&info.stick, part))
}

/// `padFront_`/`padBack_` as the reference compares them, including the voided pair's `-1`s.
fn pad_edges(sizes: PadSizes) -> (i64, i64) {
    match sizes {
        PadSizes::Unpadded => (0, 0),
        PadSizes::Sized { front, back } => (i64::from(front.0), i64::from(back.0)),
        PadSizes::Voided => (-1, -1),
    }
}

/// Replaces: e276_verifyDdlConstraints
///
/// VERIFIES ONE `ddl.constraint` AGAINST THE DSC — core count, valid-variable count, relative
/// compute-op order, stick sizes and dim sizes, over the operands the constraint names.
///
/// ⛔ [`None`] IS BOTH `llvm_unreachable`s: *"Ddl constraints not met"* and *"Illegal ddl"*.
/// ⛔ TRAP: `max_num_valid` DEFAULTS TO ONE HUNDRED; the relative-order seed is `INT_MIN`, BELOW the
/// `-1` an unset `computeOpIdx_` carries; `StickSizesAgree` never latches an EMPTY first list.
pub fn verify_ddl_constraint<A: Arch>(
    program: &Program,
    interface: &mut DdlInterface,
    dsc: &DesignSpaceConfig,
    operands: &[Operand],
    constraint: DdlConstraint,
) -> Option<()> {
    let inputs = operand_names(operands);
    match constraint {
        DdlConstraint::MinNumCores(min) => (dsc.core_ids_used.count().0 >= min.0).then_some(()),
        DdlConstraint::NumValid { min, max } => {
            let mut valid = 0u32;
            for input in inputs {
                let Some(input) = input else { continue };
                let stmt = *program.definition(input)?;
                match stmt.kind {
                    StmtKind::OperationBind => {
                        if interface.operation_definition.contains_key(&input) {
                            valid += 1;
                        }
                    }
                    StmtKind::GetExternalDataTransferAllocation => {
                        let Attrs::ExternalAllocation { memory, .. } = stmt.attrs else {
                            return None;
                        };
                        let tensor = operand_at(stmt.operands, 0)?;
                        let prop = tensor_prop(program, &mut interface.tensor_definition, tensor)?;
                        let lds = dsc.labeled_ds.at(prop.lds?)?;
                        if Some(register_storage(memory_component(memory)))
                            == lds.pinning().pinned_component()
                        {
                            valid += 1;
                        }
                    }
                    _ => return None,
                }
            }
            (valid >= min && valid <= max).then_some(())
        }
        DdlConstraint::RelativeOpOrder => {
            // `INT_MIN`: an absence that is BELOW the `-1` an unset `computeOpIdx_` carries.
            let mut cur_max: Option<Option<ComputeOpIdx>> = None;
            for input in inputs {
                let Some(input) = input else { continue };
                let Some(prop) = interface.operation_definition.get(&input) else {
                    continue;
                };
                let index = prop.compute_op;
                match cur_max {
                    Some(max) if max >= index => return None,
                    _ => cur_max = Some(index),
                }
            }
            Some(())
        }
        DdlConstraint::StickSizeAt {
            slice,
            dim_idx,
            value,
        } => {
            for input in inputs {
                let Some(input) = input else { continue };
                let Some(prop) = interface.tensor_definition.get(&input) else {
                    continue;
                };
                let lds = dsc.labeled_ds.at(prop.lds?)?;
                let sizes = stick_extents::<A>(dsc, lds.ds_type(), slice)?;
                if sizes.get(dim_idx)?.1 != value {
                    return None;
                }
            }
            Some(())
        }
        DdlConstraint::StickSizesAgree { slice } => {
            let mut previous: Vec<(PrimaryDim, Elements)> = Vec::new();
            for input in inputs {
                let Some(input) = input else { continue };
                let Some(prop) = interface.tensor_definition.get(&input) else {
                    continue;
                };
                let lds = dsc.labeled_ds.at(prop.lds?)?;
                let sizes = stick_extents::<A>(dsc, lds.ds_type(), slice)?;
                if previous.is_empty() {
                    previous.clone_from(&sizes);
                }
                if previous != sizes {
                    return None;
                }
            }
            Some(())
        }
        DdlConstraint::DimSize { cmp, value } => {
            let stage = dsc.core_stage().dims();
            for input in inputs {
                let input = input?;
                let prop = interface.dim_association.get(&input)?;
                if prop.drop_dim {
                    continue;
                }
                let dim = prop.dim?;
                let size = match prop.meta_dim_kind {
                    MetaDimKind::Unpadded | MetaDimKind::WindowDim => {
                        stage.extent(dim).unwrap_or(Extent(-1))
                    }
                    kind => {
                        let Some(padding) = stage.padding.get(&dim) else {
                            continue;
                        };
                        match kind {
                            MetaDimKind::PadFront => Extent(pad_edges(padding.sizes).0),
                            MetaDimKind::PadBack => Extent(pad_edges(padding.sizes).1),
                            MetaDimKind::Stride => Extent(padding.stride.get()),
                            MetaDimKind::Dilation => Extent(padding.dilation.0),
                            MetaDimKind::Padded => {
                                stage.padded_extent(dim, PadType::PaddedFullSpanWUnneeded)?
                            }
                            MetaDimKind::PadValid => {
                                stage.padded_extent(dim, PadType::PaddedNoZeroPad)?
                            }
                            MetaDimKind::Unpadded | MetaDimKind::WindowDim => return None,
                        }
                    }
                };
                let met = match cmp {
                    ConstraintCmp::Equal => size == value,
                    ConstraintCmp::Less => size.0 < value.0,
                };
                if !met {
                    return None;
                }
            }
            Some(())
        }
    }
}

/// Replaces: e277_getTensorAndAllocation
///
/// THE DDL TENSOR AND `ddl.allocate` A `dsc2::DataInfo` MAPS TO, by matching the DSC's own allocate
/// node against the ones the DDL declares.
///
/// ⛔ TRAP: `allocations` IS A `DenseMap`, so where two allocate ops share a node the reference's
/// `break` takes an UNSPECIFIED one; this takes the first in the caller's order.
/// ⛔ THE ALLOCATE OPS ARE THE ONES ENTRY 325 IS EMITTING, not the censused ones: its only caller
/// walks the schedule tree minting `ddl.allocate`s and resolves every later `ddl.unit` against them.
/// ⭐ A `NO_COMPONENT` UNIT AND AN UNMATCHED NODE BOTH ANSWER THE EMPTY PAIR.
pub fn tensor_and_allocation<S: AllocationSite + ?Sized>(
    site: &S,
    allocations: &[EmittedAllocate],
    tensor_definition: &BTreeMap<NameId, TensorProp>,
    ext_constant_definition: &BTreeMap<NameId, ConstIdx>,
    operand_constant_tensor: &BTreeMap<NameId, SenComponent>,
    unit: SenComponent,
    origin: Option<DataOrigin>,
) -> Option<TensorAndAllocation> {
    if matches!(unit, SenComponent::NoComponent) {
        return Some(TensorAndAllocation::default());
    }
    let node = match origin {
        Some(DataOrigin::LabeledDs(lds)) => site.lds_allocation(lds, unit)?,
        Some(DataOrigin::Constant(constant)) => site.constant_allocation(constant, unit)?,
        None => {
            return Some(TensorAndAllocation {
                tensor: tensor(
                    tensor_definition,
                    ext_constant_definition,
                    operand_constant_tensor,
                    unit,
                    None,
                )
                .map(EmittedTensor::Named),
                allocate: None,
            });
        }
    };
    let Some(held) = allocations.iter().find(|held| held.node == node) else {
        return Some(TensorAndAllocation::default());
    };
    Some(TensorAndAllocation {
        tensor: Some(held.tensor),
        allocate: Some(held.op),
    })
}

/// Replaces: e278_checkMetaDimensions
///
/// CHECKS EVERY PADDED DDL DIM AGAINST THE DSC'S OWN META DIMS — each kind the op states must exist
/// in the DSC, and each kind the DSC states must be stated by the op.
///
/// ⛔ [`None`] IS *"Illegal ddl."* AND *"Internal error in PaddedDimensionOp verification."*, whose
/// `emitError` is reached ON A NULL OP — a null dereference in the reference.
/// ⛔ TRAP: `Dilation` IS SKIPPED by the second check. ⭐ ASKING MUTATES the map it iterates.
pub fn check_meta_dimensions(
    program: &Program,
    interface: &mut DdlInterface,
    dsc: &DesignSpaceConfig,
) -> Option<()> {
    let mut in_dsc: BTreeMap<PrimaryDim, BTreeSet<MetaDimKind>> = BTreeMap::new();
    for (dim, padding) in &dsc.core_stage().dims().padding {
        let kinds = in_dsc.entry(*dim).or_default();
        kinds.extend([
            MetaDimKind::Padded,
            MetaDimKind::PadFront,
            MetaDimKind::PadBack,
            MetaDimKind::PadValid,
        ]);
        if padding.window_dim.is_some() {
            kinds.extend([
                MetaDimKind::WindowDim,
                MetaDimKind::Stride,
                MetaDimKind::Dilation,
            ]);
        }
    }
    let padded: Vec<(NameId, Option<NameId>)> = interface
        .dim_association
        .iter()
        .filter(|(_, prop)| matches!(prop.meta_dim_kind, MetaDimKind::Padded) && !prop.drop_dim)
        .map(|(dim, prop)| (*dim, prop.non_padded_dim))
        .collect();
    for (dim, non_padded) in padded {
        let dim_type = match non_padded {
            Some(unpadded) => interface.dim_association.entry(unpadded).or_default().dim,
            None => None,
        };
        let stated = |kind: MetaDimKind| {
            dim_type
                .and_then(|dim_type| in_dsc.get(&dim_type))
                .is_some_and(|kinds| kinds.contains(&kind))
        };
        let stmt = *program.definition(dim)?;
        if stmt.kind != StmtKind::PaddedDimension {
            return None;
        }
        let mut seen: BTreeSet<MetaDimKind> = BTreeSet::new();
        for operand in operand_names(stmt.operands).into_iter().flatten() {
            let prop = interface.dim_association.entry(operand).or_default();
            if !prop.is_meta_dim() {
                continue;
            }
            let kind = prop.meta_dim_kind;
            seen.insert(kind);
            if !stated(kind) {
                return None;
            }
        }
        for kind in [
            MetaDimKind::PadFront,
            MetaDimKind::PadBack,
            MetaDimKind::PadValid,
            MetaDimKind::WindowDim,
            MetaDimKind::Stride,
        ] {
            if stated(kind) && !seen.contains(&kind) {
                return None;
            }
        }
    }
    Some(())
}

/// Replaces: e279_processAccessPatterns
///
/// WRITES ONE ACCESS-PATTERN STYLE PER DIM into the op's per-dim map, keyed by the primary dim each
/// DDL dim is associated with.
///
/// ⛔ ASKING IS A MUTATION — `dim_association_[dims[i]]` — so an unseen dim is in the map afterwards.
/// ⛔ DIVERGENCE: AN UNASSIGNED `dim_` is the reference's `PrimaryDimTypesCount` KEY, a sentinel
/// nothing reads back; that is [`None`]. ⭐ [`StyledDims::stated`] settled the broadcast already.
pub fn process_access_patterns<S: Copy, T: Copy>(
    dim_association: &mut BTreeMap<NameId, DimProp>,
    styled: &StyledDims<S>,
    convert: impl Fn(S) -> T,
    result: &mut BTreeMap<PrimaryDim, T>,
) -> Option<()> {
    for &(dim, style) in styled.pairs() {
        let key = dim_association.entry(dim).or_default().dim?;
        result.insert(key, convert(style));
    }
    Some(())
}

// ═══ ENTRIES 321-325 — THE TWO DDL WALKERS, AND THE SYMBOL TABLE THEY FILL ═══════════════════════

/// ONE REGION OF THE DDL — the `mlir::Region*` `region2blocks_` is keyed by, which is an IDENTITY and
/// never dereferenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub u32);

/// THE SYNC NODES ONE `signal_name=` OWNS ON ONE CORELET — `SyncProp::SyncsPerCl`
/// (`ddc/ddl/ddl_conversion.h:404-407`), by NAME because the reference's vectors hold tree pointers.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncEnds {
    /// `senders_`, in the order the DDL states them.
    pub senders: Vec<NodeName>,
    /// `receivers_`.
    pub receivers: Vec<NodeName>,
}

/// EVERY SYNC SHARING ONE `signal_name=` — `DdlInterface::SyncProp`.
///
/// ⛔ THE `-1` KEY IS THE UNSPLIT CASE, which is [`None`] here: a signal that does not separate its
/// corelets has ONE end list, not a list per corelet, and the two are not the same map slot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncProp {
    /// `separateCorelets_`.
    pub separate_corelets: bool,
    /// `syncsPerCl_`, whose `-1` key is [`None`].
    pub syncs_per_cl: BTreeMap<Option<Corelet>, SyncEnds>,
}

/// ONE `ddl.core_to_core_communication`'S RESOLVED RING — `DdlInterface::CoreToCoreProp`.
///
/// ⛔ DIVERGENCE: THE TWO FOLD MANAGERS ARE THE MAP THEY ANSWER. The reference builds a
/// `FoldManager` over (core, corelet) and inserts a TARGET CORE as its datum
/// (`ddl_conversion.cpp:2023-2075`); [`StartAddress`] is that manager specialised to [`Bytes`] and
/// cannot hold a core. Building the fold space is the mechanism for reaching the coordinate, so the
/// pair `(core, corelet) -> target core` is stated directly — which also makes the
/// `hasZeroFoldDim()` `DT_CHECK` an empty map rather than a runtime refusal.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoreToCoreProp {
    /// `dim_` — the reduction dim the ring walks, absent for `PrimaryDimTypesCount`.
    pub dim: Option<PrimaryDim>,
    /// `nextCore_`: which core each (core, corelet) sends to.
    pub next_core: BTreeMap<(Core, Corelet), Core>,
    /// `prevCore_`: which core each (core, corelet) receives from.
    pub prev_core: BTreeMap<(Core, Corelet), Core>,
}

/// A `ddl.padded_dimension`'S THREE OPERAND GROUPS — `getPrimaryDim`, `getPaddingDim` and
/// `getWindowDim`.
///
/// ⭐ AN ARGUMENT AND NOT A CENSUS READ: the op writes them as `primary=`/`padding=`/`window=`
/// KEYWORDS, which the generated tables record as no operands at all — and reaching an operand is the
/// one thing a port may drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedDimension<'d> {
    /// `primary=` — the unpadded dim this one pads.
    pub primary: NameId,
    /// `padding=`, admitting only `PadFront`, `PadValid` and `PadBack`.
    pub padding: &'d [NameId],
    /// `window=`, admitting only `WindowDim`, `Stride` and `Dilation`.
    pub window: &'d [NameId],
}

/// Replaces: e322_processPaddedDimensionOp
///
/// TIES A PADDED DDL DIM TO ITS UNPADDED ONE and admits its meta dims — at most one dim of each kind,
/// each carrying the unpadded dim as its own `nonPaddedDim`.
///
/// ⛔⛔ ONE `dimsSeen` MAP SPANS BOTH LOOPS, and only the CROSS-GROUP half of that is unobservable:
/// no kind is admissible in both groups. Each group's OWN repeat check is load-bearing — two
/// `padding=` dims of one kind, or two `window=` dims of one kind, are the refusal.
/// ⛔ [`None`] IS EVERY *"Illegal ddl"*: a definition that is not a `ddl.padded_dimension`, a primary
/// dim that is not `Unpadded`, an inadmissible kind, and a repeated kind.
pub fn process_padded_dimension_op(
    program: &Program,
    interface: &mut DdlInterface,
    dim: NameId,
    operands: PaddedDimension<'_>,
) -> Option<()> {
    if interface.dim_association.contains_key(&dim) {
        return Some(());
    }
    if program.definition(dim)?.kind != StmtKind::PaddedDimension {
        return None;
    }
    let unpadded = operands.primary;
    if !matches!(
        process_dimension_op(program, interface, unpadded)?.meta_dim_kind,
        MetaDimKind::Unpadded
    ) {
        return None;
    }
    let padded = interface.dim_association.entry(dim).or_default();
    padded.meta_dim_kind = MetaDimKind::Padded;
    padded.non_padded_dim = Some(unpadded);
    let mut dims_seen: BTreeMap<MetaDimKind, NameId> = BTreeMap::new();
    for (group, admitted) in [
        (operands.padding, PADDING_DIM_KINDS.as_slice()),
        (operands.window, WINDOW_DIM_KINDS.as_slice()),
    ] {
        for &operand in group {
            let kind = process_dimension_op(program, interface, operand)?.meta_dim_kind;
            if !admitted.contains(&kind) || dims_seen.contains_key(&kind) {
                return None;
            }
            interface
                .dim_association
                .entry(operand)
                .or_default()
                .non_padded_dim = Some(unpadded);
            dims_seen.insert(kind, operand);
        }
    }
    Some(())
}

/// The only kinds `padding=` admits (`ddl_conversion.cpp:132-133`).
const PADDING_DIM_KINDS: [MetaDimKind; 3] = [
    MetaDimKind::PadFront,
    MetaDimKind::PadValid,
    MetaDimKind::PadBack,
];

/// The only kinds `window=` admits (`ddl_conversion.cpp:161-162`).
const WINDOW_DIM_KINDS: [MetaDimKind; 3] = [
    MetaDimKind::WindowDim,
    MetaDimKind::Stride,
    MetaDimKind::Dilation,
];

/// ONE OP OF A `ddl.transformations` REGION.
///
/// ⭐ AN OWNED REGION BECAUSE THE CENSUS FLATTENS THIS ONE. The generated tables record a
/// `ddl.disable_transfer_promotion` at depth 0 with an empty path, dropping the `ddl.if` that encloses
/// it in three of the vendored templates — and the enclosing condition is the whole decision.
/// ⛔ [`Self::Other`] IS KEPT SPELLABLE, unlike most refusals in this module: the transformations
/// section is arbitrary input text, so *"Unexpected operation found in the transformations section"*
/// is a real answer about a real DDL rather than an unreachable arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transformation {
    /// `ddl.disable_transfer_promotion`.
    DisableTransferPromotion,
    /// `ddl.if(%condition) { then } else { else }`.
    If {
        /// The `ddl.condition*` tree it tests.
        condition: NameId,
        /// Region 0.
        then_region: Vec<Transformation>,
        /// Region 1.
        else_region: Vec<Transformation>,
    },
    /// `ddl.yield` — "No processing is needed."
    Yield,
    /// Any other `ddl.*`.
    Other(StmtKind),
}

/// What the reference writes to `std::cerr` for a transfer-promotion disable (`:2043`).
pub const TRANSFER_PROMOTION_DISABLED: &str = "\nDDL specification disables promotion of transfer.";

/// Replaces: e324_processTransformations
///
/// APPLIES THE `ddl.transformations` SECTION — the only transformation the vendored templates state is
/// turning `enableMovingDataTransfer` OFF, under a condition that must resolve at compile time.
///
/// ⛔ [`None`] IS BOTH ABORTS: an `ddl.if` whose condition is NOT resolved to a bool, and any op
/// outside the three. ⭐ ONLY THE TAKEN ARM IS WALKED, so a disable under a false condition is not
/// applied — the flag is not a union over the section.
/// ⭐ THE DIAGNOSTICS ARE ANSWERED, NOT PRINTED: `verbose_ > 0` gates the stream, and the caller owns
/// it. One [`TRANSFER_PROMOTION_DISABLED`] per disable reached, which is what the reference prints.
pub fn process_transformations(
    program: &Program,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &DesignSpaceConfig,
    region: &[Transformation],
) -> Option<Vec<String>> {
    let mut said = Vec::new();
    for op in region {
        match op {
            Transformation::DisableTransferPromotion => {
                said.push(TRANSFER_PROMOTION_DISABLED.to_owned());
                metadata.transformation_config.enable_moving_data_transfer = false;
            }
            Transformation::If {
                condition,
                then_region,
                else_region,
            } => {
                let taken = process_condition(program, interface, metadata, dsc, *condition)?
                    .resolved?
                    .then_some(then_region)
                    .unwrap_or(else_region);
                said.extend(process_transformations(
                    program, interface, metadata, dsc, taken,
                )?);
            }
            Transformation::Yield => {}
            Transformation::Other(_) => return None,
        }
    }
    Some(said)
}

/// ONE WORD OF A CONSTANT'S DATA — `ConstantInfo::data_`'s element, a BIT PATTERN and not a number:
/// the reference stores `BinaryConvert<uint32_t>` of whatever float it settled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstantWord(pub u32);

/// ONE `dsc.constantInfo_` ENTRY — what a `ddl.define_constant`, `ddl.get_external_constant` or
/// `ddl.alias_one_constant_of` resolves to.
///
/// ⛔ DIVERGENCE: THIS LIVES ON [`DdlConversion`], NOT ON [`DesignSpaceConfig`], which carries no
/// `constantInfo_`; [`AllocationSite`] is the seam the rest of the module already reads through.
/// ⭐ THE FOLD SPACE IS DROPPED: `buildAllConstantFoldSpace` then `insertData(data, zero_coord)` puts
/// the data at the origin of a space nothing else reads, so the data IS the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantInfo {
    /// `name_`.
    pub name: ConstName,
    /// `dataFormat_` with the width that goes with it.
    pub declared: TypeDefinition,
    /// `data_.getSingleData()`.
    pub data: Vec<ConstantWord>,
    /// `allocations_` — which minted allocation places it on each component.
    pub allocations: BTreeMap<SenComponent, AllocId>,
}

/// WHICH END OF A NODE ONE `dsc2::DataInfo` IS.
///
/// ⛔ DIVERGENCE, AND THE REASON IT EXISTS: the reference keeps `startAddr_`'s ring target and
/// `constEleOffsets_` INSIDE `DataInfo`, and [`DataInfo`] here is [`Copy`] with a `const fn`
/// constructor — a ring and a per-coordinate map are neither. Both are stated beside the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeEnd {
    /// `src_`.
    Src,
    /// `dstVias_.at(i)`.
    Dst(u32),
    /// `inputs_.at(i)`.
    Input(u32),
    /// `outputs_.at(i)`.
    Output(u32),
}

/// WHAT ENTRY 323 ASKS OF A DSC THAT CANNOT YET BE ASKED — the reads whose fields the ported
/// [`DesignSpaceConfig`] and [`crate::schedule::l3::dsc::LabeledDs`] do not carry.
///
/// ⛔ A TRAIT IN THE MODULE'S OWN IDIOM, beside [`AllocationSite`] and [`InternalTensorSite`]:
/// `LabeledDs` has no `dataFormat_` and its fields are private, `computeOp_` is not on the DSC at
/// all, and `addressGranularityScalePerUnit`, `numWkSlicesPerDim_` and `coreIdToWkSlice_` are on the
/// system definition and the SuperDSC. [`None`]/`false` is each reference `.at()` throwing.
pub trait DdlSite: AllocationSite + InternalTensorSite {
    /// `dsc.computeOp_.front().attributes_.dataFormat_` — the fused op's precision.
    fn fused_format(&self) -> Option<DataFormat>;

    /// `dsc.labeledDs_.at(lds).dataFormat_`.
    fn lds_format(&self, lds: LdsIdx) -> Option<DataFormat>;

    /// `addressGranularityScalePerUnit.count({senCompToGenericComp.at(unit), storage})`.
    fn unit_reaches(&self, unit: SenComponent, storage: SenComponent) -> bool;

    /// `dsc.getDimIndexInLayoutOrder(labeledDs_.at(lds).dsType_, dim) >= 0`.
    fn dim_in_layout_order(&self, lds: LdsIdx, dim: PrimaryDim) -> bool;

    /// `sdsc.numWkSlicesPerDim_.at(dim)`, absent for a dim the SuperDSC does not split.
    fn wk_slices(&self, dim: PrimaryDim) -> Option<u32>;

    /// `sdsc.coreIdToWkSlice_` — every core the super-DSC states, with its work slice per dim.
    ///
    /// ⛔⛔ NOT A `findCoreBySlice(dim, slice)`, WHICH IS WHAT THIS ASKED FOR AND DOES NOT EXIST: the
    /// ring's neighbour lookup (`ddl_conversion.cpp:1966-1972`) is a LOCAL LAMBDA that replaces ONE
    /// coordinate of the visited core's FULL slice vector and searches for the core holding the
    /// result. A work split over two dims puts several cores on one elected slice and only the whole
    /// vector tells them apart, so a per-slice lookup cannot express it.
    fn core_work_slices(&self) -> BTreeMap<Core, WkSlice>;

    /// `dsc.primaryDsInfo_.at(labeledDs_.at(lds).dsType_)`'s `stickDimOrder_`/`stickSize_` pair —
    /// what [`cumulative_stick_sizes`] is asked of.
    fn stick_dims(&self, lds: LdsIdx) -> Option<StickDims>;
}

/// THE DSC2 STATE ONE DDL TEMPLATE'S WALK BUILDS — `DdlConversion`'s own members, plus arenas for the
/// nodes the reference keeps as raw `new`ed pointers hanging off the tree.
///
/// ⭐ THE TREE CARRIES A [`SchedNode::Leaf`] PER MINTED ALLOCATE, TRANSFER AND COMPUTE and the body
/// lives in an arena, because those three are exactly the kinds [`SchedNode`] cannot hold by value;
/// [`Self::node_ids`] is the join back. Loops, conditions and syncs are tree variants already.
#[derive(Debug, Clone, PartialEq)]
pub struct DdlConversion {
    /// The `dsc2::Dsc` schedule tree being built.
    pub tree: ScheduleTree,
    /// Every minted `dsc2::AllocateNode`.
    pub allocations: BTreeMap<AllocId, AllocateNode>,
    /// Every minted `dsc2::ComputeNode`.
    pub computes: BTreeMap<NodeId, ComputeNode>,
    /// Every minted `dsc2::TransferNode`.
    pub transfers: BTreeMap<NodeId, TransferNode>,
    /// Which node each minted name is, so a `ScheduleNode*` in the metadata resolves.
    pub node_ids: BTreeMap<NodeName, NodeId>,
    /// `labeledDs_.at(lds).memOrg_` — ⛔ DIVERGENCE: `LabeledDs` carries no `memOrg_`, and this is
    /// what [`AllocationSite::lds_allocation`] answers from.
    pub lds_memory: BTreeMap<(LdsIdx, SenComponent), AllocId>,
    /// `dsc.constantInfo_` — ⛔ DIVERGENCE, see [`ConstantInfo`].
    pub constants: BTreeMap<ConstIdx, ConstantInfo>,
    /// The SFPRING ring an end's `startAddr_` targets, per end — see [`NodeEnd`].
    pub ring_targets: BTreeMap<(NodeId, NodeEnd), CoreToCoreProp>,
    /// `DataInfo::constEleOffsets_`, per end.
    pub const_ele_offsets: BTreeMap<(NodeId, NodeEnd), BTreeMap<(Core, Corelet), Elements>>,
    /// `TransferNode::rotateNumElements_` (`dsc/dsc2.h:839`) — ⛔ DIVERGENCE: [`TransferNode`] carries
    /// no rotate slot, and 22 of its 23 minting sites would state a zero they never write, so the one
    /// site that does write it states it here. Absent IS the reference's `0`.
    pub rotate_elements: BTreeMap<NodeId, Elements>,
    /// `padding_dim=` on a `ddl.allocate`, keyed by the allocate's own result. ⛔ THE CENSUS DROPS IT
    /// and no vendored template states one; an allocate is reached from a unit operand and never from
    /// a top-level walk, which is why its dims are stated here and not on a [`DdlOp`] variant.
    pub padding_dims: BTreeMap<NameId, Vec<NameId>>,
    /// `chunk_dstgid` — which datastage `ddl.get_external_datastage property="chunk"` names.
    pub chunk_datastage: Option<DatastageId>,
    /// `core_dstgid`.
    pub core_datastage: Option<DatastageId>,
    /// Which region a `ddl.allocate` was written in, which is the block it lands in —
    /// `myAlloc->getParentRegion()`, an identity the generated tables do not record.
    pub alloc_regions: BTreeMap<NameId, RegionId>,
    next_node: u32,
    next_alloc: u32,
}

impl DdlConversion {
    /// A conversion over one head block, with nothing minted yet.
    #[must_use]
    pub fn new(head: BlockNode) -> Self {
        Self {
            tree: ScheduleTree::new(head),
            allocations: BTreeMap::new(),
            computes: BTreeMap::new(),
            transfers: BTreeMap::new(),
            node_ids: BTreeMap::new(),
            lds_memory: BTreeMap::new(),
            constants: BTreeMap::new(),
            ring_targets: BTreeMap::new(),
            const_ele_offsets: BTreeMap::new(),
            rotate_elements: BTreeMap::new(),
            padding_dims: BTreeMap::new(),
            alloc_regions: BTreeMap::new(),
            chunk_datastage: None,
            core_datastage: None,
            next_node: 0,
            next_alloc: 0,
        }
    }

    /// The next unused node identity — `new dsc2::ScheduleNode` as an index.
    fn mint_node(&mut self, name: &NodeName) -> NodeId {
        let node = NodeId(self.next_node);
        self.next_node += 1;
        self.node_ids.insert(name.clone(), node);
        node
    }

    /// The next unused allocation identity.
    fn mint_alloc(&mut self) -> AllocId {
        let alloc = AllocId(self.next_alloc);
        self.next_alloc += 1;
        alloc
    }

    /// `myId = size(); while (count(myId)) myId++` over `constantInfo_`.
    fn next_constant(&self) -> ConstIdx {
        let mut id = ConstIdx(u32::try_from(self.constants.len()).unwrap_or(u32::MAX));
        while self.constants.contains_key(&id) {
            id = ConstIdx(id.0.saturating_add(1));
        }
        id
    }

    /// The block of [`Self::tree`] that carries this name, INCLUDING THE HEAD.
    ///
    /// ⛔ `head_` IS NEVER VISITED BY A TRAVERSAL and it is still where the top region's ops attach:
    /// `processRegion` is entered with `getHeadMutable()`, so a search that could not name it would
    /// refuse every op of the dataflow.
    fn block_mut(&mut self, name: &NodeName) -> Option<&mut BlockNode> {
        if self.tree.head().name == *name {
            return Some(self.tree.head_mut());
        }
        self.tree.find_block_mut(|block| block.name == *name)
    }

    /// `currParent->addChildNode(node)`, [`None`] where no block of the tree carries that name.
    fn add_child(&mut self, parent: &NodeName, node: SchedNode) -> Option<()> {
        self.block_mut(parent)?.add_child(node);
        Some(())
    }

    /// The same, VIRTUALLY: a `CONDITION` parent takes the block as its next region and every other
    /// parent takes it as a child, which is the `addChildNode` override the reference dispatches on.
    fn add_block_child(&mut self, parent: &NodeName, block: BlockNode) -> Option<()> {
        if let Some(cond) = self.tree.find_guarded_mut(|node| node.name == *parent) {
            return cond.add_region(block);
        }
        self.add_child(parent, SchedNode::Block(block))
    }

    /// The name of the block a [`BlockId`] addresses, which is how `belowLxScheduleInsertBlock`
    /// resolves to a block of [`Self::tree`].
    ///
    /// ⛔ MINTED NODES ONLY — [`Self::node_ids`] is written by [`Self::mint_node`], so neither the head
    /// nor a block the tree was seeded with is ever named here. That is the other half of entry 364's
    /// trap: its `belowLxScheduleInsertBlock == getHeadMutable()` comparison cannot hold.
    fn block_name(&self, block: BlockId) -> Option<NodeName> {
        let node = block.node();
        self.node_ids
            .iter()
            .find_map(|(name, held)| (*held == node).then(|| name.clone()))
    }

    /// `dsc.getMutableAllocation(..)->addAllocUser(node)` over a transfer's source and every
    /// destination, which is what makes a split copy a user of the same allocations.
    fn attribute_transfer_users(&mut self, node: NodeId) {
        let Some(transfer) = self.transfers.get(&node).cloned() else {
            return;
        };
        let ends: Vec<DscOperand> = core::iter::once(transfer.src)
            .chain(transfer.dsts.iter().copied())
            .collect();
        for end in ends {
            let alloc = match (end.data.my_lds_idx, end.data.constant_id) {
                (Some(lds), _) => self.lds_allocation(lds, end.storage),
                (None, Some(constant)) => self.constant_allocation(constant, end.storage),
                (None, None) => None,
            };
            if let Some(alloc) = alloc {
                self.add_alloc_user(alloc, Some(node));
            }
        }
    }

    /// `allocNode->addAllocUser(userNode)`, which a null `userNode` skips.
    fn add_alloc_user(&mut self, alloc: AllocId, user: Option<NodeId>) {
        if let (Some(node), Some(user)) = (self.allocations.get_mut(&alloc), user) {
            node.alloc_users.push(user);
        }
    }
}

impl AllocationSite for DdlConversion {
    fn lds_allocation(&self, lds: LdsIdx, unit: SenComponent) -> Option<AllocId> {
        self.lds_memory.get(&(lds, unit)).copied()
    }

    fn constant_allocation(&self, constant: ConstIdx, unit: SenComponent) -> Option<AllocId> {
        self.constants
            .get(&constant)?
            .allocations
            .get(&unit)
            .copied()
    }
}

/// ONE OP OF A DDL REGION, AS ENTRY 323 IS HANDED IT — the statement plus every operand the generated
/// tables drop.
///
/// ⭐ REACHING AN OPERAND IS THE ONE THING A PORT MAY DROP, and this is where that shows: `ddl.if`
/// has NO census statement (its branches are flattened into [`crate::generated::Enclosing::Arm`]), a
/// transfer's `access_pattern_dim=` and a datastage constraint's `min=` are keyword text the tables
/// do not keep, and a region is an identity they do not carry.
/// ⛔ [`Self::CoreCoreletCond`] PRODUCES NOTHING — *"Uses of this op is Prohibited in ddl"* — and
/// stays spellable because a template could still state one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DdlOp<'d> {
    /// `ddl.loop(%num, %den, %dims..)`.
    Loop(&'d Stmt),
    /// `ddl.parametric_loop(%dim, %reference)`.
    ParametricLoop(&'d Stmt),
    /// `ddl.data_transfer(%src, [%dsts..])`, with the dims its styles apply to.
    DataTransfer {
        /// The statement.
        stmt: &'d Stmt,
        /// `access_pattern_dim=`.
        pattern_dims: &'d [NameId],
    },
    /// `ddl.compute([%inputs], [%outputs])`.
    Compute(&'d Stmt),
    /// `ddl.datastage`.
    Datastage(&'d Stmt),
    /// `ddl.get_external_datastage`.
    GetExternalDatastage(&'d Stmt),
    /// `ddl.if(%condition) { .. } else { .. }`.
    If {
        /// The `ddl.condition*` tree it tests.
        condition: NameId,
        /// Region 0.
        then_region: RegionId,
        /// Region 1.
        else_region: RegionId,
    },
    /// `ddl.opaque(%reference, ..)`.
    Opaque(&'d Stmt),
    /// `ddl.sync`.
    Sync(&'d Stmt),
    /// `ddl.implicit_sync(%allocation)`.
    ImplicitSync(&'d Stmt),
    /// `ddl.datastage_constraint(%ds, %reference, %dims..)`, with the bound the tables drop.
    DatastageConstraint {
        /// The statement.
        stmt: &'d Stmt,
        /// `min=`, which [`Attrs::DatastageConstraint`] does not record. TEXT, as `values=` and
        /// `max=` are: the census keeps every DDL bound as the template spelled it.
        min: Option<&'d str>,
    },
    /// `ddl.force_innermost_dimensions(%alloc, %ds, %dims..)`.
    ForceInnermostDimensions(&'d Stmt),
    /// `ddl.core_to_core_communication(%dims..)`.
    CoreToCoreCommunication(&'d Stmt),
    /// `ddl.core_corelet_cond`.
    CoreCoreletCond,
}

/// WHAT ONE OP LEAVES THE WALK WITH — the `pair<dsc2::BlockNode*, vector<int>>` entry 323 answers.
///
/// ⛔ AN ABSENT PARENT IS THE REFERENCE'S NULL: the core/chunk loop hands back
/// `metadata_.belowLxScheduleInsertBlock`, which is unset until an L3 unit fills it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpOutcome {
    /// `blockNodeForInsertion` — where this op's regions attach their children.
    pub parent: Option<NodeName>,
    /// `regionIndecesToProcess`, in the order the walk descends them.
    pub regions: Vec<RegionId>,
}

/// One corelet by index, which is what a `numCoreletsUsed_DSC2_` loop counter names.
fn corelet(index: u32) -> Option<Corelet> {
    Corelet::checked(index)
}

/// `dsc.numCoreletsUsed_DSC2_ == 2` — the gate the sync split and the ring's second corelet share,
/// whose `-1` is `false` rather than a refusal.
fn corelets_used_dsc2(dsc: &DesignSpaceConfig) -> bool {
    dsc.corelets_used_dsc2.is_some_and(|used| used.get() == 2)
}

/// `numCoreletsUsed_DSC2_` as the corelets themselves.
///
/// ⛔⛔ REVIEWED: THIS READ `numCoreletsUsed_` — every one of the eight corelet counts in the
/// reference file is `numCoreletsUsed_DSC2_` (`ddl_conversion.cpp:331`, `:340`, `:1029`, `:1733`,
/// `:1978`, `:1986`, `:1995`, `:2164`), never the other field, and the two differ before entry 054's
/// `prepDsc` runs. The `-1` a DSC is built with is a loop that does not execute, so [`None`] is NO
/// corelets rather than a refusal.
fn corelets_of(dsc: &DesignSpaceConfig) -> Vec<Corelet> {
    (0..dsc.corelets_used_dsc2.map_or(0, CoreletsUsed::get))
        .filter_map(corelet)
        .collect()
}

/// The `dsc2` spelling of a comparison — both enums are `dsc/dscdefn.h:95` verbatim, so the join is
/// eleven arms and total.
const fn dsc_cond_op(op: CondOp) -> DscCondOp {
    match op {
        CondOp::Eq => DscCondOp::Eq,
        CondOp::Ne => DscCondOp::Ne,
        CondOp::Lt => DscCondOp::Lt,
        CondOp::Le => DscCondOp::Le,
        CondOp::Gt => DscCondOp::Gt,
        CondOp::Ge => DscCondOp::Ge,
        CondOp::Toggle => DscCondOp::Toggle,
        CondOp::Always => DscCondOp::Always,
        CondOp::Never => DscCondOp::Never,
        CondOp::Const => DscCondOp::Const,
        CondOp::Default => DscCondOp::Default,
    }
}

/// A [`Dsts`] TAKEN APART — its ends zipped with their routes, which is the only way to reach either
/// half: `dstVias_` is `push_back`-ed and rewritten in the reference, and [`Dsts`] is immutable in
/// both its length and its hops.
fn dsts_parts(dsts: &Dsts) -> (Vec<DscOperand>, Vec<Hops>) {
    dsts.routes()
        .map(|(end, hops)| (*end, Hops(hops.to_vec())))
        .unzip()
}

/// The same, put back together — a transfer has at least one destination, so an empty list is no
/// [`Dsts`] at all.
fn dsts_from(ends: Vec<DscOperand>, hops: Vec<Hops>) -> Option<Dsts> {
    let (first, rest) = ends.split_first()?;
    Some(Dsts::new(*first, rest.to_vec()).with_hops(hops))
}

// ⛔⛔ REVIEWED: `dsc2::memories` HAS SIXTEEN MEMBERS (`dsc/dscdefn.cpp:142-144`) and this file
// carried a local EIGHT — the image of `memory_component`, which is a different question. Every
// live ask here is `dsc2::memories.count(..)` in the reference (`ddl_conversion.cpp:967`, `:985`,
// `:1198`, `:1203`, `:3091`, `:3162`; the compute and transfer arms share one `Emission::unit`), and
// the last two read a storage off the DSC's own `DataLocation`, where HBM and PTIRF are ordinary: on
// those the local set answered false and entry 325 emitted a bare `ddl.unit` where the reference
// emits its tensor/allocate pair. Entry 250's `is_memory` is the one verified table, so this file
// asks it.

/// The `Metadata::newAllocations_` key a storage component is — `ddc::memories`, which has no
/// `Sfpstate` and so refuses one.
#[must_use]
pub const fn ddc_memory(component: SenComponent) -> Option<DdcMemory> {
    Some(match component {
        SenComponent::Lx => DdcMemory::Lx,
        SenComponent::L0 => DdcMemory::L0,
        SenComponent::L0Scale => DdcMemory::L0Scale,
        SenComponent::Pelrf => DdcMemory::PeLrf,
        SenComponent::Sfplrf => DdcMemory::SfpLrf,
        SenComponent::Ptarf => DdcMemory::PtaRf,
        SenComponent::Ptxrf => DdcMemory::PtxRf,
        SenComponent::Hbm => DdcMemory::Hbm,
        _ => return None,
    })
}

/// `is_any_of(storage, LX, PTXRF, L3LUIBR)` — which storages a prefilled external transfer keys on.
#[must_use]
pub const fn external_storage(component: SenComponent) -> Option<ExternalStorage> {
    Some(match component {
        SenComponent::Lx => ExternalStorage::Lx,
        SenComponent::Ptxrf => ExternalStorage::PtxRf,
        SenComponent::L3luibr => ExternalStorage::L3LuIbr,
        _ => return None,
    })
}

/// `BinaryConvert<uint32_t>(Fp16BinToFloat(v))` (`SenDataConvert/sen_data_convert.cpp:251`).
///
/// ⭐ THE COMPOSITION IS PURE BIT REASSEMBLY — the `float` the reference goes through cancels — and
/// the rebias `x - 31 + 127` masked to eight bits is `x + 96` for every value a six-bit field holds.
/// ⛔ A ZERO EXPONENT **AND** A ZERO MANTISSA IS THE EARLY `0.0`; a zero exponent with a mantissa is
/// not, and reassembles like any other.
#[must_use]
pub const fn sen169_fp16_as_fp32_bits(pattern: u16) -> ConstantWord {
    let sign = ((pattern & 0x8000) >> 15) as u32;
    let exponent = ((pattern & 0x7E00) >> 9) as u32;
    let mantissa = (pattern & 0x01FF) as u32;
    if exponent == 0 && mantissa == 0 {
        return ConstantWord(0);
    }
    ConstantWord((sign << 31) | (((exponent + 96) & 0xFF) << 23) | (mantissa << 14))
}

/// The PT row components by row index, which is what `"ptrow" + std::to_string(i)` spells.
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

/// The L0 load-unit row components by row index.
const L0LU_ROW_COMPONENTS: [SenComponent; 8] = [
    SenComponent::L0lurow0,
    SenComponent::L0lurow1,
    SenComponent::L0lurow2,
    SenComponent::L0lurow3,
    SenComponent::L0lurow4,
    SenComponent::L0lurow5,
    SenComponent::L0lurow6,
    SenComponent::L0lurow7,
];

/// `unrollRowUnits` — the components a `unit=` names, with its row spans expanded.
///
/// ⛔ BOTH ABORTS ARE UNSPELLABLE: `unit=` is a censused [`Unit`], so *"Unknown unit"* and the
/// dash-range parse's *"Incorrect format for multiple row units"* cannot occur, and every row goes
/// through [`rows_of`]'s arch bound rather than the reference's raw `numPTRows` loop.
/// ⭐ `l0lu` IS THE ONE SPAN [`rows_of`] DOES NOT COVER: the reference expands it to
/// `l0lurow0-<numPTRows-1>`, exactly as it expands `pt`, over the same row count.
#[must_use]
pub fn unroll_row_units(unit: Unit) -> Vec<SenComponent> {
    match unit {
        Unit::L0lu => rows_of(Unit::Pt)
            .into_iter()
            .map(|row| L0LU_ROW_COMPONENTS[row.get() as usize])
            .collect(),
        Unit::Pt
        | Unit::Ptrow0
        | Unit::Ptrow1To3
        | Unit::Ptrow1To7
        | Unit::Ptrow3
        | Unit::Ptrow7 => rows_of(unit)
            .into_iter()
            .map(|row| PT_ROW_COMPONENTS[row.get() as usize])
            .collect(),
        Unit::Constant => vec![SenComponent::Constant],
        Unit::L0su => vec![SenComponent::L0su],
        Unit::Lxlu => vec![SenComponent::Lxlu],
        Unit::Lxsu => vec![SenComponent::Lxsu],
        Unit::Pe => vec![SenComponent::Pe],
        Unit::Ptnorth => vec![SenComponent::Ptnorth],
        Unit::Ptsouth => vec![SenComponent::Ptsouth],
        Unit::Sfp => vec![SenComponent::Sfp],
        Unit::Sfpring => vec![SenComponent::Sfpring],
    }
}

/// `getGenericCompIfAvailable` — the component a unit's generic image names, the unit itself where
/// `senCompToGenericComp` has no entry for it.
///
/// ⭐ THE CRATE'S [`generic_comp`] ANSWERS A [`GenericComp`], WHICH IS A DIFFERENT ENUM; the
/// reference's map is component-to-component, so the join back is stated here and is total.
#[must_use]
pub fn generic_component(unit: SenComponent) -> SenComponent {
    match generic_comp(unit) {
        None => unit,
        Some(GenericComp::Pt) => SenComponent::Pt,
        Some(GenericComp::Pe) => SenComponent::Pe,
        Some(GenericComp::Sfp) => SenComponent::Sfp,
        Some(GenericComp::Lxlu) => SenComponent::Lxlu,
        Some(GenericComp::Lxsu) => SenComponent::Lxsu,
        Some(GenericComp::Lx) => SenComponent::Lx,
        Some(GenericComp::L0lu) => SenComponent::L0lu,
        Some(GenericComp::L0su) => SenComponent::L0su,
        Some(GenericComp::L3lu) => SenComponent::L3lu,
        Some(GenericComp::L3su) => SenComponent::L3su,
        Some(GenericComp::Hbm) => SenComponent::Hbm,
        Some(GenericComp::LxVirtualIbr) => SenComponent::Lxvirtualibr,
        Some(GenericComp::CrossPtnLink) => SenComponent::Crossptnlink,
        Some(GenericComp::SfpState) => SenComponent::Sfpstate,
        Some(GenericComp::PeState) => SenComponent::Pestate,
        Some(GenericComp::L0) => SenComponent::L0,
        Some(GenericComp::Constant) => SenComponent::Constant,
        Some(GenericComp::SfpRing) => SenComponent::Sfpring,
    }
}

/// THE MACC A PRECISION RESOLVES TO — the six-way `dataFormat_` switch the `"macc"` arm performs.
///
/// ⛔ [`None`] IS *"Unexpected input precision"*, and this is the whole reason
/// [`crate::schedule::dsc2::ComputeNode::op`] is a [`DdlComputeType`]: four of the six answers are
/// spellings no `computetype=` states, so the census has no member for them.
#[must_use]
pub const fn macc_compute_type(precision: DataFormat) -> Option<DdlComputeType> {
    Some(match precision {
        DataFormat::Senint4 => DdlComputeType::Ima4,
        DataFormat::Senint8 => DdlComputeType::Ima8,
        DataFormat::Sen143Fp8 | DataFormat::Sen152Fp8 => DdlComputeType::Fma8,
        DataFormat::Sen169Fp16 | DataFormat::Bfloat16 => DdlComputeType::Fma16,
        DataFormat::IeeeFp32 => DdlComputeType::Fma32,
        DataFormat::Sen121Fp4 => DdlComputeType::Fma4,
        _ => return None,
    })
}

/// The `dsc2` spelling of a schedule condition — the join a [`CondProp`] needs to become a
/// [`ConditionNode`], which is [`NodeId`] to [`LoopId`] and [`CondValType`] to [`LoopBound`].
#[must_use]
fn dsc_loop_cond(composite: &LoopCondComposite) -> DscLoopCondComposite {
    DscLoopCondComposite {
        two_level_or_of_ands: composite
            .or_of_ands
            .iter()
            .map(|ands| {
                ands.iter()
                    .map(|cond| DscLoopCond {
                        loop_comp: LoopId(cond.loop_node),
                        dim: cond.dim,
                        op: dsc_cond_op(cond.op),
                        bound: match cond.against {
                            CondValType::Int(value) => {
                                LoopBound::Index(u32::try_from(value).unwrap_or_default())
                            }
                            CondValType::First => LoopBound::First,
                            CondValType::Last => LoopBound::Last,
                        },
                    })
                    .collect()
            })
            .collect(),
        negated: composite.negated,
    }
}
/// `getTensorOrExtConstIdx` — which datastream or constant a DDL value is, minting the constant on
/// first sight.
///
/// ⛔ [`None`] IS EVERY *"Illegal ddl"* AND THE `DT_ERROR`: no datatype matching the fused precision,
/// an external constant over the four-register op-const size, a mismatch against the DSC's own entry,
/// and *"Missing external constant in DSC"*.
/// ⭐ THE MEMO IS AN OUTPUT: `ext_constant_definition_` is what a second mention resolves through,
/// and entry 325 reads it back.
fn tensor_or_ext_const_idx<S: DdlSite + ?Sized>(
    program: &Program,
    state: &mut DdlConversion,
    interface: &mut DdlInterface,
    site: &S,
    value: NameId,
) -> Option<DataOrigin> {
    let definition = program.definition(value)?;
    if matches!(
        definition.kind,
        StmtKind::Tensor | StmtKind::InternalTensor | StmtKind::AliasOneTensorOf
    ) {
        return Some(DataOrigin::LabeledDs(
            tensor_prop(program, &mut interface.tensor_definition, value)?.lds?,
        ));
    }
    if let Some(&known) = interface.ext_constant_definition.get(&value) {
        return Some(DataOrigin::Constant(known));
    }
    let precision = site.fused_format();
    // ⭐ THE DATATYPE IS PICKED BY THE FUSED OP'S PRECISION for the two multi-type forms, and stated
    // outright by a plain `ddl.define_constant`.
    let (declared, defining) = match definition.kind {
        StmtKind::GetExternalConstant => (
            operand_names(definition.operands)
                .into_iter()
                .flatten()
                .find_map(|ty| {
                    let parsed = *process_types(program, &[ty])?.first()?;
                    (Some(parsed.format) == precision).then_some(parsed)
                })?,
            None,
        ),
        StmtKind::DefineConstant => (
            *process_types(program, &[operand_at(definition.operands, 0)?])?.first()?,
            Some(definition),
        ),
        StmtKind::AliasOneConstantOf => operand_names(definition.operands)
            .into_iter()
            .flatten()
            .find_map(|aliased| {
                let aliased = program.definition(aliased)?;
                if aliased.kind != StmtKind::DefineConstant {
                    return None;
                }
                let parsed =
                    *process_types(program, &[operand_at(aliased.operands, 0)?])?.first()?;
                (Some(parsed.format) == precision).then_some((parsed, Some(aliased)))
            })?,
        _ => return None,
    };
    let id = state.next_constant();
    match defining {
        // A defined constant: the words are its own `value=`, converted where the fused op is FP32
        // and the constant is stated in SEN169.
        Some(constant) => {
            let Attrs::Constant {
                name, value: words, ..
            } = constant.attrs
            else {
                return None;
            };
            let convert = precision == Some(DataFormat::IeeeFp32)
                && declared.format == DataFormat::Sen169Fp16;
            state.constants.insert(
                id,
                ConstantInfo {
                    name,
                    declared: if convert {
                        TypeDefinition {
                            format: DataFormat::IeeeFp32,
                            bit_size: declared.bit_size,
                        }
                    } else {
                        declared
                    },
                    data: words
                        .iter()
                        .map(|word| {
                            let pattern = u16::try_from(*word & 0xFFFF).unwrap_or_default();
                            if convert {
                                sen169_fp16_as_fp32_bits(pattern)
                            } else {
                                ConstantWord(u32::try_from(*word).unwrap_or_default())
                            }
                        })
                        .collect(),
                    allocations: BTreeMap::new(),
                },
            );
        }
        // An external constant: it must ALREADY be in the DSC, matching in format and element count.
        None => {
            let Attrs::Constant {
                name, num_elements, ..
            } = definition.attrs
            else {
                return None;
            };
            let elements = u64::try_from(num_elements?).ok()?;
            if u64::from(declared.bit_size.0) * elements > 4 * 32 {
                return None;
            }
            let (&held_id, held) = state.constants.iter().find(|(_, held)| held.name == name)?;
            if held.declared.format != declared.format
                || u64::try_from(held.data.len()).ok()? != elements
            {
                return None;
            }
            interface.ext_constant_definition.insert(value, held_id);
            return Some(DataOrigin::Constant(held_id));
        }
    }
    interface.ext_constant_definition.insert(value, id);
    Some(DataOrigin::Constant(id))
}

/// `processAllocation` — mints the `dsc2::AllocateNode` a `ddl.allocate` states and hangs it in the
/// block its own region opened.
///
/// ⛔ [`None`] IS *"Handling of external allocations with repeated dimensions is not yet
/// implemented"*, *"An allocation for this tensor is already present"* and *"Allocation for no tensor
/// and no constant"*; *"Unrecognised memory"* is unspellable, `memory=` being a censused [`Memory`].
/// ⭐ `maxDimSizes_.resize(n, -1)` IS ONE [`MaxDimSize::Unset`] PER LAYOUT DIM, which is exactly what
/// `ddl.force_innermost_dimensions` later checks is still untouched.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference lambda captures all of these"
)]
fn process_allocation<S: DdlSite + ?Sized>(
    program: &Program,
    state: &mut DdlConversion,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &DesignSpaceConfig,
    site: &S,
    allocate: NameId,
    curr_parent: &NodeName,
    user: Option<NodeId>,
) -> Option<AllocId> {
    let stmt = program.definition(allocate)?;
    let Attrs::Allocate {
        memory,
        buffers,
        padding,
        ..
    } = stmt.attrs
    else {
        return None;
    };
    let storage = memory_component(memory);
    let origin = tensor_or_ext_const_idx(
        program,
        state,
        interface,
        site,
        operand_at(stmt.operands, 0)?,
    )?;
    let parent = state
        .alloc_regions
        .get(&allocate)
        .and_then(|region| interface.region2blocks.get(region).cloned())
        .unwrap_or_else(|| curr_parent.clone());
    let alloc = state.mint_alloc();
    let mut placement = AllocPlacement {
        num_buffers: match buffers {
            Buffers::Single => NumBuffers::Single,
            Buffers::SizeTwoReserveAll => NumBuffers::Double,
        },
        padding: Padding::default(),
        buffer_offset: BTreeMap::new(),
        is_start_addr_symbolic: false,
    };
    let node = match origin {
        DataOrigin::LabeledDs(lds) => {
            let ordered = dsc.layout_dims.get(&lds)?.to_vec();
            if ordered.iter().collect::<BTreeSet<_>>().len() != ordered.len() {
                return None;
            }
            let (first, rest) = ordered.split_first()?;
            let layout = AllocLayout::new(
                (*first, MaxDimSize::Unset),
                rest.iter().map(|dim| (*dim, MaxDimSize::Unset)).collect(),
            );
            let stated = state
                .padding_dims
                .get(&allocate)
                .cloned()
                .unwrap_or_default();
            let mut form = BTreeMap::new();
            process_access_patterns(
                &mut interface.dim_association,
                &StyledDims::stated(&stated, padding)?,
                allocation_pad_type,
                &mut form,
            )?;
            for (dim, pad) in form {
                placement.padding.set(dim, pad);
            }
            if state.lds_memory.contains_key(&(lds, storage)) {
                return None;
            }
            state.lds_memory.insert((lds, storage), alloc);
            let held = metadata
                .new_allocations
                .entry(ddc_memory(storage)?)
                .or_default();
            if held.lds_idx_and_alloc_node.contains_key(&lds) {
                return None;
            }
            held.lds_idx_and_alloc_node.insert(lds, alloc);
            AllocateNode {
                name: NodeName(format!("allocate_lds{}_{}", lds.0, storage.spelling())),
                component: storage,
                lds: Some(lds),
                const_idx: None,
                temp_storage_for_compute: None,
                layout,
                start_address: StartAddress::default(),
                placement,
                gap_stick_spread: BTreeMap::new(),
                alloc_users: Vec::new(),
            }
        }
        DataOrigin::Constant(constant) => {
            state
                .constants
                .get_mut(&constant)?
                .allocations
                .insert(storage, alloc);
            let held = metadata
                .new_allocations
                .entry(ddc_memory(storage)?)
                .or_default();
            // `DT_CHECK(allocMeta.find(constIdx_) == allocMeta.end())` (`:846`) — the constant arm's
            // own duplicate guard, which the LDS arm above keeps too. `allocations_[storage]` beside
            // it IS an overwriting `operator[]`, so only this one refuses.
            if held.cons_id_and_alloc_node.contains_key(&constant) {
                return None;
            }
            held.cons_id_and_alloc_node.insert(constant, alloc);
            AllocateNode {
                name: NodeName(format!(
                    "allocate_const{}_{}",
                    constant.0,
                    storage.spelling()
                )),
                component: storage,
                lds: None,
                const_idx: Some(constant),
                temp_storage_for_compute: None,
                // ⛔ THE REFERENCE LEAVES `layoutDimOrder_` AND `maxDimSizes_` EMPTY HERE — only the
                // LDS arm fills them (`:798-803`) — and [`AllocLayout`] is non-empty by construction
                // because `layoutDimOrder_.at(0)` is unguarded wherever it is read. Every reader
                // reaches an allocation through an LDS, so this stands in for a layout nothing asks
                // of a constant; entry 325's emission is gated on [`AllocateNode::lds`] to keep it
                // out of the DDL.
                layout: AllocLayout::new((PrimaryDim::X, MaxDimSize::Unset), Vec::new()),
                start_address: StartAddress::default(),
                placement,
                gap_stick_spread: BTreeMap::new(),
                alloc_users: Vec::new(),
            }
        }
    };
    let name = node.name.clone();
    state.allocations.insert(alloc, node);
    state.mint_node(&name);
    state.add_child(&parent, SchedNode::Leaf(name))?;
    state.add_alloc_user(alloc, user);
    interface.alloc_storage.insert(allocate, alloc);
    Some(alloc)
}

/// `getRepetitionIfExists` — the `replication=` on the allocation a unit names, ONE where the unit
/// names no allocation or the allocation states none.
#[must_use]
fn repetition_if_exists(program: &Program, unit: NameId) -> ReplicationFactor {
    let stated = || -> Option<u64> {
        let stmt = program.definition(unit)?;
        if !matches!(stmt.attrs, Attrs::Unit { .. }) {
            return None;
        }
        let Attrs::Allocate { replication, .. } =
            program.definition(operand_at(stmt.operands, 1)?)?.attrs
        else {
            return None;
        };
        u64::try_from(replication?).ok()
    };
    ReplicationFactor(stated().unwrap_or(1))
}

/// ONE END OF A NODE, RESOLVED — what `setDataLocAndInfo` fills in and hands back.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedEnd {
    /// `dataLoc` and `dataInfo` together, which is what a `dsc2::Operand` is.
    operand: DscOperand,
    /// The row units AFTER `units.front()` — what the caller expands the node over.
    remaining: Vec<SenComponent>,
    /// `via_`, only ever filled for a transfer destination.
    vias: Vec<SenComponent>,
    /// The SFPRING ring this end's `startAddr_` targets.
    ring: Option<CoreToCoreProp>,
    /// `constEleOffsets_`.
    offsets: BTreeMap<(Core, Corelet), Elements>,
}

/// `setDataLocAndInfo` — resolves one `ddl.unit` or `ddl.operand_constant` into the end a node
/// carries, minting the allocation behind it where nothing has yet.
///
/// ⛔ [`None`] IS EVERY *"Illegal ddl"* OF THE LAMBDA: an end that is neither op, a storage that is
/// not a memory, an SFPRING end with no c2c pattern (and a c2c pattern off SFPRING), a unit not
/// connected to its allocation's component, an unresolvable tensor, a start offset on a constant or
/// on a tensor with no reduced stick dim or one past the stick, a via outside a transfer destination,
/// and a via that unrolls to more than one unit.
/// ⭐ THE REMAINING ROW UNITS ARE THE RETURN VALUE, and the FIRST of them is this end's own.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference lambda captures all of these"
)]
fn set_data_loc_and_info<S: DdlSite + ?Sized>(
    program: &Program,
    state: &mut DdlConversion,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &DesignSpaceConfig,
    site: &S,
    end: NameId,
    curr_parent: &NodeName,
    user: Option<NodeId>,
    takes_vias: bool,
) -> Option<ResolvedEnd> {
    let stmt = program.definition(end)?;
    let mut resolved = ResolvedEnd::default();
    match stmt.attrs {
        Attrs::Unit {
            unit,
            data_connect,
            via,
            stick_offset,
        } => {
            let mut storage = SenComponent::NoComponent;
            if let Some(allocation) = operand_at(stmt.operands, 1) {
                let kind = program.definition(allocation)?.kind;
                if let Some(alloc) = interface.alloc_storage.get(&allocation).copied() {
                    storage = state.allocations.get(&alloc)?.component;
                    state.add_alloc_user(alloc, user);
                } else {
                    match kind {
                        StmtKind::Allocate => {
                            let alloc = process_allocation(
                                program,
                                state,
                                interface,
                                metadata,
                                dsc,
                                site,
                                allocation,
                                curr_parent,
                                user,
                            )?;
                            storage = state.allocations.get(&alloc)?.component;
                        }
                        StmtKind::GetExternalDataTransferAllocation => {
                            let external = program.definition(allocation)?;
                            let Attrs::ExternalAllocation {
                                data_connect: fill,
                                memory,
                            } = external.attrs
                            else {
                                return None;
                            };
                            storage = memory_component(memory);
                            let DataOrigin::LabeledDs(lds) = tensor_or_ext_const_idx(
                                program,
                                state,
                                interface,
                                site,
                                operand_at(external.operands, 0)?,
                            )?
                            else {
                                return None;
                            };
                            let alloc = *state.lds_memory.get(&(lds, storage))?;
                            state.add_alloc_user(alloc, user);
                            // ⭐ WRITING THROUGH THE SLOT IS THE REFERENCE'S `*prefilledIt->second =
                            // getDataConnect()`: the locator names the transfer end that was left
                            // unfilled, and this is the statement that fills it.
                            let slot = *metadata
                                .prefilled_external_transfer_data_connects
                                .get(&(lds, external_storage(storage)?))?;
                            let filled = state.transfers.get_mut(&slot.transfer)?;
                            match slot.end {
                                TransferEnd::Src => {
                                    filled.src.data.data_connect = Some(fill);
                                }
                                TransferEnd::FirstDst => {
                                    filled.dsts.first_mut().data.data_connect = Some(fill);
                                }
                            }
                            interface.alloc_storage.insert(allocation, alloc);
                        }
                        StmtKind::CoreToCoreCommunication => {
                            if unit != Unit::Sfpring {
                                return None;
                            }
                            let ring = interface.core_to_core_definitions.get(&allocation)?;
                            // `getNextCoreInChain()` is result 2 of the four the op binds and
                            // `getPrevCoreInChain()` result 3 — *"Wrong return value used"* is which
                            // of the two this unit named.
                            let results = program.definition(allocation)?.results;
                            resolved.ring = Some(
                                match results.iter().position(|bound| *bound == allocation) {
                                    Some(2) => CoreToCoreProp {
                                        dim: ring.dim,
                                        next_core: ring.next_core.clone(),
                                        prev_core: BTreeMap::new(),
                                    },
                                    Some(3) => CoreToCoreProp {
                                        dim: ring.dim,
                                        next_core: BTreeMap::new(),
                                        prev_core: ring.prev_core.clone(),
                                    },
                                    _ => return None,
                                },
                            );
                        }
                        _ => return None,
                    }
                }
                if !is_memory(storage) && kind != StmtKind::CoreToCoreCommunication {
                    return None;
                }
            }
            let mut units = unroll_row_units(unit);
            let own = *units.first()?;
            if own == SenComponent::Sfpring && resolved.ring.is_none() {
                return None;
            }
            units.remove(0);
            resolved.remaining = units;
            resolved.operand.unit = own;
            resolved.operand.storage = storage;
            if is_memory(storage) && !site.unit_reaches(generic_component(own), storage) {
                return None;
            }
            let origin = tensor_or_ext_const_idx(
                program,
                state,
                interface,
                site,
                operand_at(stmt.operands, 0)?,
            )?;
            match origin {
                DataOrigin::LabeledDs(lds) => resolved.operand.data.my_lds_idx = Some(lds),
                DataOrigin::Constant(constant) => {
                    resolved.operand.data.constant_id = Some(constant);
                }
            }
            resolved.operand.data.data_connect = Some(data_connect);
            if let Some(offset) = stick_offset {
                let DataOrigin::LabeledDs(lds) = origin else {
                    return None;
                };
                let held = dsc.labeled_ds.at(lds)?;
                let sticks = stick_sizes(
                    &dsc.primary_ds_info.get(&held.ds_type())?.stick,
                    StickPart::Whole,
                );
                let [(dim, extent)] = sticks.as_slice() else {
                    return None;
                };
                if held.scale(*dim)? == Scale::StickDim {
                    let offset = Elements(u64::try_from(offset).ok()?);
                    if offset >= *extent {
                        return None;
                    }
                    for core in dsc.core_ids_used.iter() {
                        for corelet in corelets_of(dsc) {
                            resolved.offsets.insert((core, corelet), offset);
                        }
                    }
                } else {
                    return None;
                }
            }
            if let DdlVia::Through(hop) = via {
                if !takes_vias {
                    return None;
                }
                let hops = unroll_row_units(hop);
                let [only] = hops.as_slice() else {
                    return None;
                };
                resolved.vias.push(*only);
            }
        }
        // `ddl.operand_constant` carries its `name=` as a censused [`ConstName`], so the four the
        // reference recognises are the whole arm and *"Unknown name in operand const"* is the rest.
        Attrs::Constant { name, .. } if stmt.kind == StmtKind::OperandConstant => {
            let component = match name {
                ConstName::N0Point0 => SenComponent::Zero,
                ConstName::N1Point0 => SenComponent::One,
                ConstName::Nfwd0 => SenComponent::Nfwd0,
                ConstName::Nfwd2 => SenComponent::Nfwd2,
                _ => return None,
            };
            resolved.operand.unit = component;
            resolved.operand.storage = component;
            interface.operand_constant_tensor.insert(end, component);
        }
        _ => return None,
    }
    Some(resolved)
}

impl Default for ResolvedEnd {
    /// `dataLoc`/`dataInfo` before either is written — `NO_COMPONENT` on both halves, which is
    /// exactly the reference's default-constructed pair.
    fn default() -> Self {
        Self {
            operand: DscOperand {
                unit: SenComponent::NoComponent,
                storage: SenComponent::NoComponent,
                data: DataInfo::default(),
            },
            remaining: Vec::new(),
            vias: Vec::new(),
            ring: None,
            offsets: BTreeMap::new(),
        }
    }
}

impl ResolvedEnd {
    /// The allocation this end's datastream or constant is placed in.
    fn allocation(&self, state: &DdlConversion) -> Option<AllocId> {
        match (self.operand.data.my_lds_idx, self.operand.data.constant_id) {
            (Some(lds), _) => state.lds_allocation(lds, self.operand.storage),
            (None, Some(constant)) => state.constant_allocation(constant, self.operand.storage),
            (None, None) => None,
        }
    }
}

/// `nameTransferNode` — `transfer_lds<N>_src:<unit>_dst:<unit>..`.
#[must_use]
fn transfer_node_name(src: &DscOperand, dsts: &[SenComponent]) -> NodeName {
    let mut name = format!(
        "transfer_lds{}_src:{}_dst",
        src.data.my_lds_idx.map_or(-1, |lds| i64::from(lds.0)),
        src.unit.spelling()
    );
    for dst in dsts {
        name.push(':');
        name.push_str(dst.spelling());
    }
    NodeName(name)
}
/// EVERYTHING ONE ARM OF ENTRY 323 IS HANDED — the reference lambda's captures, gathered so the
/// fourteen arms take one argument each instead of nine.
struct OpContext<'a, S: DdlSite + ?Sized> {
    program: &'a Program,
    state: &'a mut DdlConversion,
    interface: &'a mut DdlInterface,
    metadata: &'a mut Metadata,
    dsc: &'a mut DesignSpaceConfig,
    site: &'a mut S,
    curr_parent: NodeName,
}

/// `ddl::LoopOp` — one `dsc2::LoopNode`, or NO node at all where its band is empty or is exactly the
/// core/chunk pair the L3 schedule already opened.
///
/// ⛔ [`None`] IS *"Unknown dimension in loop operation"* and *"Loop numerator/denominator datastage
/// undefined"*. ⭐ THE THREE OUTCOMES ARE THE WHOLE POINT: an empty band leaves `currParent` as the
/// insertion block and mints nothing, the core/chunk band hands back
/// `metadata_.belowLxScheduleInsertBlock`, and only a real band mints and registers a loop.
fn op_loop<S: DdlSite + ?Sized>(ctx: &mut OpContext<'_, S>, stmt: &Stmt) -> Option<OpOutcome> {
    let Attrs::Loop { label } = stmt.attrs else {
        return None;
    };
    let mut dims = Vec::new();
    for dim in operand_names(stmt.operands).into_iter().flatten().skip(2) {
        let prop = ctx.interface.dim_association.get(&dim)?;
        if !prop.drop_dim {
            dims.push(LoopDim {
                dim: prop.dim?,
                kind: prop.meta_dim_kind,
            });
        }
    }
    let num = *ctx
        .interface
        .datastage_definition
        .get(&operand_at(stmt.operands, 0)?)?;
    let den = *ctx
        .interface
        .datastage_definition
        .get(&operand_at(stmt.operands, 1)?)?;
    if dims.is_empty() {
        return Some(OpOutcome {
            parent: Some(ctx.curr_parent.clone()),
            regions: vec![RegionId(0)],
        });
    }
    if Some(num) == ctx.state.core_datastage && Some(den) == ctx.state.chunk_datastage {
        if let Some(label) = label {
            ctx.interface.core_chunk_loop_label = Some(label);
        }
        let below = ctx
            .metadata
            .below_lx_schedule_insert_block
            .and_then(|block| ctx.state.block_name(block));
        return Some(OpOutcome {
            parent: below,
            regions: vec![RegionId(0)],
        });
    }
    let mut name = format!("loop_ds{}_ds{}", num.0, den.0);
    for dim in &dims {
        name.push('_');
        name.push_str(dim.dim.spelling());
    }
    let name = NodeName(name);
    let node = ctx.state.mint_node(&name);
    ctx.state.add_child(
        &ctx.curr_parent.clone(),
        SchedNode::Loop(Box::new(LoopNode {
            block: BlockNode {
                name: name.clone(),
                children: Vec::new(),
            },
            dims,
            num: Some(num),
            den: Some(den),
            parametric_lds: None,
        })),
    )?;
    if let Some(label) = label {
        ctx.interface.loop_labels.insert(label, node);
    }
    Some(OpOutcome {
        parent: Some(name),
        regions: vec![RegionId(0)],
    })
}

/// `ddl::ParametricLoopOp` — a loop over ONE dim whose trip count is the referenced tensor's, so it
/// carries a labeled DS instead of a numerator and denominator.
///
/// ⛔ [`None`] IS *"Unknown dimension"*, *"Specified dimension of parametric loop operation is marked
/// to be dropped"* — a dropped dim is an error HERE and merely skipped in [`op_loop`] — and
/// *"Specified tensor of parametric loop does not have ldsIdx_"*.
fn op_parametric_loop<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    stmt: &Stmt,
) -> Option<OpOutcome> {
    let Attrs::Loop { label } = stmt.attrs else {
        return None;
    };
    let prop = ctx
        .interface
        .dim_association
        .get(&operand_at(stmt.operands, 0)?)?
        .clone();
    if prop.drop_dim {
        return None;
    }
    let dim = prop.dim?;
    let lds = tensor_prop(
        ctx.program,
        &mut ctx.interface.tensor_definition,
        operand_at(stmt.operands, 1)?,
    )?
    .lds?;
    let name = NodeName(format!(
        "parametric_loop_{}({})",
        dim.spelling(),
        prop.meta_dim_kind.label()
    ));
    let node = ctx.state.mint_node(&name);
    ctx.state.add_child(
        &ctx.curr_parent.clone(),
        SchedNode::Loop(Box::new(LoopNode {
            block: BlockNode {
                name: name.clone(),
                children: Vec::new(),
            },
            dims: vec![LoopDim {
                dim,
                kind: prop.meta_dim_kind,
            }],
            num: None,
            den: None,
            parametric_lds: Some(lds),
        })),
    )?;
    if let Some(label) = label {
        ctx.interface.loop_labels.insert(label, node);
    }
    Some(OpOutcome {
        parent: Some(name),
        regions: vec![RegionId(0)],
    })
}

/// `ddl::DataTransferOp` — one `dsc2::TransferNode`, SPLIT into one node per row where the source or
/// the single row-expanded destination unrolls to several units.
///
/// ⛔ [`None`] IS *"Multiple destinations with row expansion not supported"*, *"Row unit expansion in
/// source and destination has different size"*, and the rotate guards — a rotate off LXLU, one at or
/// past the stick, or one whose byte count is not a multiple of sixteen.
/// ⭐ THE SPLIT IS NOT COSMETIC: a source that expands and a destination that does not PREPENDS the
/// source's remaining units to the first destination's vias, and a destination that expands while the
/// source does not splits only when the source is a constant or the row split dim is in its layout.
fn op_data_transfer<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    stmt: &Stmt,
    pattern_dims: &[NameId],
) -> Option<OpOutcome> {
    let Attrs::DataTransfer {
        access_pattern,
        limit_stick_replicated,
        rotate,
    } = stmt.attrs
    else {
        return None;
    };
    let src_name = operand_at(stmt.operands, 0)?;
    let dst_names = match stmt.operands.get(1)? {
        DdlOperand::List(names) => names.to_vec(),
        DdlOperand::One(name) => vec![*name],
        _ => return None,
    };
    let src = set_data_loc_and_info(
        ctx.program,
        ctx.state,
        ctx.interface,
        ctx.metadata,
        ctx.dsc,
        &*ctx.site,
        src_name,
        &ctx.curr_parent.clone(),
        None,
        false,
    )?;
    let mut dsts = Vec::new();
    for name in &dst_names {
        let dst = set_data_loc_and_info(
            ctx.program,
            ctx.state,
            ctx.interface,
            ctx.metadata,
            ctx.dsc,
            &*ctx.site,
            *name,
            &ctx.curr_parent.clone(),
            None,
            true,
        )?;
        dsts.push(dst);
    }
    // ⛔ ONE DESTINATION MAY EXPAND, AND ONLY ONE.
    if dsts.iter().filter(|dst| !dst.remaining.is_empty()).count() > 1 {
        return None;
    }
    let rem_dst = dsts
        .iter()
        .find(|dst| !dst.remaining.is_empty())
        .map(|dst| dst.remaining.clone())
        .unwrap_or_default();
    if !src.remaining.is_empty() && !rem_dst.is_empty() && src.remaining.len() != rem_dst.len() {
        return None;
    }
    // A fifo end names no memory, and takes the OTHER end's generic component as its storage.
    let mut src_operand = src.operand;
    if !is_memory(src_operand.storage) {
        src_operand.storage = generic_component(dsts.first()?.operand.unit);
    }
    for dst in &mut dsts {
        if !is_memory(dst.operand.storage) {
            dst.operand.storage = generic_component(src_operand.unit);
        }
    }
    let rotate = match rotate {
        Some(count) => Some(Elements(u64::try_from(count).ok()?)),
        None => None,
    };
    if let Some(rotate) = rotate {
        if src_operand.unit != SenComponent::Lxlu {
            return None;
        }
        if let Some(lds) = src_operand.data.my_lds_idx {
            let held = ctx.dsc.labeled_ds.at(lds)?;
            let in_stick: u64 = stick_sizes(
                &ctx.dsc.primary_ds_info.get(&held.ds_type())?.stick,
                StickPart::Whole,
            )
            .into_iter()
            .map(|(_, extent)| extent.0)
            .product();
            if in_stick == 0 || rotate.0 >= in_stick || (128 / in_stick) * rotate.0 % 16 != 0 {
                return None;
            }
        }
    }
    let units: Vec<SenComponent> = dsts.iter().map(|dst| dst.operand.unit).collect();
    let name = transfer_node_name(&src_operand, &units);
    let node = ctx.state.mint_node(&name);
    let (first, rest) = {
        let mut ends = dsts.iter().map(|dst| dst.operand);
        (ends.next()?, ends.collect::<Vec<_>>())
    };
    let mut transfer = TransferNode {
        name: name.clone(),
        src: src_operand,
        dsts: Dsts::new(first, rest)
            .with_hops(dsts.iter().map(|dst| Hops(dst.vias.clone())).collect()),
        replication_factor: repetition_if_exists(ctx.program, src_name),
        unit_time_transfer_chunk_size: Vec::new(),
        // The class default — the DDL states no chunk count for a `ddl.datatransfer`.
        unit_time_transfer_num_chunks: NumChunks::ONE,
        padding: TransferPadding::default(),
        src_indirect: None,
        dst_indirect: None,
        core_id_to_gtr_info: BTreeMap::new(),
        transfer_size: BTreeMap::new(),
    };
    // The styles, and the dims they apply to, which entry 325 writes back in this same order.
    let mut per_dim = BTreeMap::new();
    process_access_patterns(
        &mut ctx.interface.dim_association,
        &StyledDims::stated(pattern_dims, access_pattern)?,
        transfer_access_pattern,
        &mut per_dim,
    )?;
    let meta = ctx.metadata.datatransfers.entry(node).or_default();
    meta.access_pattern_per_dim = per_dim;
    meta.force_num_elements = limit_stick_replicated
        .and_then(|count| u64::try_from(count).ok())
        .map(Elements);
    ctx.interface
        .transfer_acc_pat_dims
        .insert(node, pattern_dims.to_vec());
    if let Some(ring) = src.ring.clone() {
        ctx.state.ring_targets.insert((node, NodeEnd::Src), ring);
    }
    if !src.offsets.is_empty() {
        ctx.state
            .const_ele_offsets
            .insert((node, NodeEnd::Src), src.offsets.clone());
    }
    for (index, dst) in dsts.iter().enumerate() {
        let end = NodeEnd::Dst(u32::try_from(index).unwrap_or_default());
        if let Some(ring) = dst.ring.clone() {
            ctx.state.ring_targets.insert((node, end), ring);
        }
        if !dst.offsets.is_empty() {
            ctx.state
                .const_ele_offsets
                .insert((node, end), dst.offsets.clone());
        }
    }
    // ⭐ THE SPLIT DECISION, verbatim: the source expanding always splits, and a destination
    // expanding alone splits only for a constant source or a source whose layout states the row
    // split dim.
    let split = if !src.remaining.is_empty() {
        if rem_dst.is_empty() {
            let (ends, mut hops) = dsts_parts(&transfer.dsts);
            let mut route = src.remaining.clone();
            route.extend(transfer.dsts.hops(0).iter().copied());
            hops.resize(ends.len(), Hops(Vec::new()));
            *hops.first_mut()? = Hops(route);
            transfer.dsts = dsts_from(ends, hops)?;
        }
        true
    } else if !rem_dst.is_empty() {
        match (transfer.src.data.constant_id, transfer.src.data.my_lds_idx) {
            (Some(_), _) => true,
            (None, Some(lds)) => ctx
                .metadata
                .row_split_dim
                .is_some_and(|dim| ctx.site.dim_in_layout_order(lds, dim)),
            (None, None) => false,
        }
    } else {
        false
    };
    ctx.state.transfers.insert(node, transfer.clone());
    if let Some(rotate) = rotate {
        ctx.state.rotate_elements.insert(node, rotate);
    }
    ctx.state
        .add_child(&ctx.curr_parent.clone(), SchedNode::Leaf(name))?;
    if split {
        let rows = src.remaining.len().max(rem_dst.len());
        for row in 0..rows {
            let mut copy = transfer.clone();
            if let Some(unit) = src.remaining.get(row) {
                copy.src.unit = *unit;
            }
            if let Some(unit) = rem_dst.get(row) {
                let (mut ends, mut hops) = dsts_parts(&copy.dsts);
                let mut route = copy.dsts.hops(0).to_vec();
                route.push(copy.dsts.first().unit);
                hops.resize(ends.len(), Hops(Vec::new()));
                *hops.first_mut()? = Hops(route);
                ends.first_mut()?.unit = *unit;
                copy.dsts = dsts_from(ends, hops)?;
            }
            let copy_name = transfer_node_name(
                &copy.src,
                &copy.dsts.iter().map(|end| end.unit).collect::<Vec<_>>(),
            );
            copy.name = copy_name.clone();
            let copy_node = ctx.state.mint_node(&copy_name);
            let mut copy_meta = ctx
                .metadata
                .datatransfers
                .get(&node)
                .cloned()
                .unwrap_or_default();
            copy_meta.apply_row_offset_src = src.remaining.is_empty();
            copy_meta.apply_row_offset_dst = rem_dst.is_empty();
            ctx.metadata.datatransfers.insert(copy_node, copy_meta);
            ctx.state.transfers.insert(copy_node, copy);
            // `copyNode->rotateNumElements_ = newNode->rotateNumElements_`.
            if let Some(rotate) = rotate {
                ctx.state.rotate_elements.insert(copy_node, rotate);
            }
            ctx.state
                .add_child(&ctx.curr_parent.clone(), SchedNode::Leaf(copy_name))?;
            ctx.state.attribute_transfer_users(copy_node);
        }
    } else {
        // No split: every remaining destination unit becomes another end of the SAME node.
        for unit in &rem_dst {
            let held = ctx.state.transfers.get_mut(&node)?;
            let (mut ends, mut hops) = dsts_parts(&held.dsts);
            let mut route = hops.last().cloned().unwrap_or(Hops(Vec::new()));
            let mut end = *ends.last()?;
            route.0.push(end.unit);
            end.unit = *unit;
            ends.push(end);
            hops.push(route);
            held.dsts = dsts_from(ends, hops)?;
        }
    }
    ctx.state.attribute_transfer_users(node);
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::ComputeOp` — one `dsc2::ComputeNode` PER ROW the `unit=` unrolls to.
///
/// ⛔ [`None`] IS *"Unknown operation in ddl.compute"* (unspellable — `computetype=` is censused),
/// *"Unexpected input precision"* from the `macc` resolution, and *"Input or output of compute cannot
/// be fifo to itself"*. ⭐ THE `macc` ARM IS WHY THE OP FIELD IS A [`DdlComputeType`]; every other
/// spelling takes its precision from the first non-BOOL input, then output, and anything that is not
/// FP32 becomes SEN169.
fn op_compute<S: DdlSite + ?Sized>(ctx: &mut OpContext<'_, S>, stmt: &Stmt) -> Option<OpOutcome> {
    let Attrs::Compute {
        computetype,
        unit,
        mode,
        repetition,
        indices,
    } = stmt.attrs
    else {
        return None;
    };
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut ends = Vec::new();
    for (slot, group) in [(0_usize, &mut inputs), (1, &mut outputs)] {
        let names = match stmt.operands.get(slot)? {
            DdlOperand::List(names) => names.to_vec(),
            DdlOperand::One(name) => vec![*name],
            _ => Vec::new(),
        };
        for name in names {
            let mut end = set_data_loc_and_info(
                ctx.program,
                ctx.state,
                ctx.interface,
                ctx.metadata,
                ctx.dsc,
                &*ctx.site,
                name,
                &ctx.curr_parent.clone(),
                None,
                false,
            )?;
            if end.operand.storage == SenComponent::NoComponent {
                end.operand.storage = generic_component(end.operand.unit);
            }
            group.push(end.operand);
            ends.push(end);
        }
    }
    // ⭐ THE PRECISION IS THE OP'S OWN ANSWER, not a property of the operands: `macc` resolves by the
    // fused format and every other spelling by the first input or output that is not BOOL.
    let op = if computetype == ComputeType::Macc {
        macc_compute_type(ctx.site.fused_format()?)?
    } else {
        DdlComputeType::from(computetype)
    };
    // ⭐ `macc` STATES ITS FORMAT; every other spelling takes the first non-BOOL input format, then
    // output, and anything that is not FP32 lands as SEN169 whatever it read.
    let data_format = if computetype == ComputeType::Macc {
        Some(ctx.site.fused_format()?)
    } else if op == DdlComputeType::Fma32 {
        Some(DataFormat::IeeeFp32)
    } else {
        Some(
            ends.iter()
                .filter_map(|end| end.operand.data.my_lds_idx)
                .filter_map(|lds| ctx.site.lds_format(lds))
                .find(|format| *format != DataFormat::Bool)
                .filter(|format| *format == DataFormat::IeeeFp32)
                .unwrap_or(DataFormat::Sen169Fp16),
        )
    };
    let attribute = InstrAttribute {
        unroll: Unroll::ONE,
        precision: None,
        read_write_regs: BTreeMap::new(),
        read_only_regs: BTreeMap::new(),
        mode,
        compute_mask: ComputeMask::ALL,
        repetition: repetition
            .and_then(|slices| u32::try_from(slices).ok())
            .map_or(Repetition::ALL_SLICES, Repetition),
        indices: indices
            .iter()
            .map(|index| u32::try_from(*index).map_or(PackIndex::Extend, PackIndex::Slice))
            .collect(),
        params: BTreeMap::new(),
        input_data_connects: Vec::new(),
        output_data_connects: Vec::new(),
    };
    for (row, ex_unit) in unroll_row_units(unit).into_iter().enumerate() {
        if inputs
            .iter()
            .chain(&outputs)
            .any(|end| end.storage == ex_unit)
        {
            return None;
        }
        let name = NodeName(format!(
            "compute_{}_{}",
            ex_unit.spelling(),
            op.cpp_spelling()
        ));
        let node = ctx.state.mint_node(&name);
        ctx.state.computes.insert(
            node,
            ComputeNode {
                name: name.clone(),
                op,
                ex_unit,
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                num_folds_engaged: NumFolds::ONE,
                data_format,
                instr_attribute: attribute.clone(),
            },
        );
        ctx.state
            .add_child(&ctx.curr_parent.clone(), SchedNode::Leaf(name))?;
        // ⛔ THE CLONES ONLY, verbatim: `setDataLocAndInfo` already attributed the FIRST row's ends
        // while resolving them, and the reference re-walks them under `nodeToInsert != newNode`.
        if row > 0 {
            for end in &ends {
                if let Some(alloc) = end.allocation(ctx.state) {
                    ctx.state.add_alloc_user(alloc, Some(node));
                }
            }
        }
    }
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::DatastageOp` — a NEW datastage, whose extents both start as the core stage's own symbolic
/// maximum.
///
/// ⛔ THE TRAP, REPRODUCED AND NOT FIXED: this sets the minted stage's NAME to its own index, which is
/// non-empty — and entry 325 tests exactly that name for emptiness to tell a minted stage from an
/// external one, so a minted stage reads back as external. ⛔ [`None`] IS *"Unknown strategy"*.
fn op_datastage<S: DdlSite + ?Sized>(ctx: &mut OpContext<'_, S>, stmt: &Stmt) -> Option<OpOutcome> {
    let Attrs::Datastage { strategy, epilogue } = stmt.attrs else {
        return None;
    };
    let id = ctx.dsc.data_stages.next_index();
    // ⛔ THE MINTED STAGE STATES NO DIM: `dataStageParam_[id]` is a bare `operator[]` insert, and only
    // the core stage's volume ceiling and a name are written onto it — see [`EmptyStage`].
    let core = ctx
        .state
        .core_datastage
        .and_then(|core| ctx.dsc.data_stages.at(core))?;
    let volumes = core.ss.dims.dims().symbolic.volumes().clone();
    ctx.dsc.data_stages.mint_empty(
        id,
        EmptyStage {
            name: StageName(id.0.to_string()),
            volumes,
        },
    );
    ctx.interface
        .datastage_definition
        .insert(*stmt.results.first()?, id);
    let held = ctx.metadata.datastages.entry(id).or_default();
    held.strategy = strategy;
    held.allow_epilogue = epilogue;
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::GetExternalDatastageOp` — binds the DDL name for the chunk or the core datastage.
///
/// ⛔ [`None`] IS *"External datastage property is either chunk or core"*, and this is what
/// [`op_loop`]'s core/chunk test reads back.
fn op_get_external_datastage<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    stmt: &Stmt,
) -> Option<OpOutcome> {
    let Attrs::ExternalDatastage { property } = stmt.attrs else {
        return None;
    };
    let id = match property {
        DatastageProperty::Chunk => ctx.state.chunk_datastage?,
        DatastageProperty::Core => ctx.state.core_datastage?,
    };
    ctx.interface
        .datastage_definition
        .insert(*stmt.results.first()?, id);
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::IfOp` — NO node where the condition already resolved to a constant, and a
/// `dsc2::ConditionNode` with both regions where it did not.
///
/// ⛔ A RESOLVED CONDITION DESCENDS EXACTLY ONE REGION and mints nothing, which is how a template's
/// dead branch never reaches the tree at all. ⭐ THE SPLIT-LOOP ADJUSTMENT IS DEFERRED: every
/// core/chunk dim with more than one loop needs its terms re-pointed, and
/// `adjustConditionForSplitLoop` is a `dsc/` function outside this campaign's file list — the same
/// gap entry 250 declares as `ScheduleSurgery::adjust_condition_for_split_loop`.
fn op_if<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    condition: NameId,
    then_region: RegionId,
    else_region: RegionId,
) -> Option<OpOutcome> {
    let prop = ctx.interface.resolved_conditions.get(&condition)?.clone();
    if let Some(resolved) = prop.resolved {
        return Some(OpOutcome {
            parent: Some(ctx.curr_parent.clone()),
            regions: vec![if resolved { then_region } else { else_region }],
        });
    }
    let name = NodeName("condition".to_string());
    ctx.state.mint_node(&name);
    ctx.state.add_child(
        &ctx.curr_parent.clone(),
        SchedNode::Guarded(Box::new(ConditionNode {
            name: name.clone(),
            loop_cond: dsc_loop_cond(&prop.loop_cond),
            core_cl_cond: prop.core_cl_cond.0.clone(),
            then_region: Vec::new(),
            else_region: Vec::new(),
        })),
    )?;
    Some(OpOutcome {
        parent: Some(name),
        regions: vec![then_region, else_region],
    })
}

/// `ddl::OpaqueOp` — a `dsc2::ComputeNode` for a hand-written kernel, plus the register allocation it
/// scratches in.
///
/// ⛔ [`None`] IS an unresolvable in/out register pairing. ⭐ THE INTERNAL REGISTERS BECOME AN
/// ALLOCATION, not an attribute: they are a real `dsc2::AllocateNode` named `allocate_<compute>` whose
/// `tempStorageForCompute_` points back at the compute, which is what places them.
fn op_opaque<S: DdlSite + ?Sized>(ctx: &mut OpContext<'_, S>, stmt: &Stmt) -> Option<OpOutcome> {
    let Attrs::Opaque {
        func,
        unit,
        max_unroll,
        internal_registers,
        input_output_registers,
        params,
        reads,
        writes,
    } = stmt.attrs
    else {
        return None;
    };
    // ⛔ *"Opaque supported only without opfusion"*: a fused DSC has more than one compute op, and
    // [`DdlSite::fused_format`] is the front one's format — absent is the fused case here.
    let data_format = ctx.site.fused_format();
    let reference = tensor_prop(
        ctx.program,
        &mut ctx.interface.tensor_definition,
        operand_at(stmt.operands, 0)?,
    )?
    .lds?;
    let lds =
        add_internal_tensor(ctx.site, ctx.interface, ctx.metadata, reference, ComputeOpIdx(0))?;
    let ex_unit = *unroll_row_units(unit).first()?;
    // ⭐ THE OPAQUE FUNC IS A COMPUTE TYPE: the reference looks `op=` up in `stringToComputeType`,
    // so `muli32toi32` is a `type_` and not a separate opaque marker. ⛔ [`None`] IS *"Unknown
    // operation"*, and membership in `Metadata::opaque_ops` is `isOpaqueOp_`.
    let op = DdlComputeType::from_spelling(func.spelling())?;
    let name = NodeName(format!(
        "compute_opaque_{}_{}",
        ex_unit.spelling(),
        func.spelling()
    ));
    let node = ctx.state.mint_node(&name);
    // ⛔ *"No need unrolling without internal registers"* — an unroll factor with nothing to unroll.
    if internal_registers.is_empty() && max_unroll != MaxUnroll(1) {
        return None;
    }
    // The scratch registers, as a real allocation on the unit's own register file.
    let internal_reg_alloc = if internal_registers.is_empty() {
        None
    } else {
        // ⛔ *"Execution unit is not a compute unit"* — only the three compute units have one.
        let (component, memory) = match ex_unit {
            SenComponent::Pe => (SenComponent::Pelrf, DdcMemory::PeLrf),
            SenComponent::Sfp => (SenComponent::Sfplrf, DdcMemory::SfpLrf),
            SenComponent::Pt => (SenComponent::Ptarf, DdcMemory::PtaRf),
            _ => return None,
        };
        // ⛔ *"trying to add an internal register allocation for a tensor for which an allocation is
        // already present"*.
        if ctx.state.lds_memory.contains_key(&(lds, component)) {
            return None;
        }
        let dims = ctx
            .dsc
            .primary_ds_info
            .get(&ctx.dsc.labeled_ds.at(lds)?.ds_type())?
            .layout
            .clone();
        let mut layout = dims.iter().map(|dim| (dim, MaxDimSize::Unset));
        let alloc = ctx.state.mint_alloc();
        let alloc_name = NodeName(format!("allocate_{}", name.0));
        ctx.state.allocations.insert(
            alloc,
            AllocateNode {
                name: alloc_name.clone(),
                component,
                lds: Some(lds),
                const_idx: None,
                temp_storage_for_compute: Some(name.clone()),
                layout: AllocLayout::new(layout.next()?, layout.collect()),
                start_address: StartAddress::default(),
                placement: AllocPlacement {
                    num_buffers: NumBuffers::Single,
                    padding: Padding::default(),
                    buffer_offset: BTreeMap::new(),
                    is_start_addr_symbolic: false,
                },
                gap_stick_spread: BTreeMap::new(),
                alloc_users: vec![node],
            },
        );
        ctx.state.mint_node(&alloc_name);
        ctx.state
            .add_child(&ctx.curr_parent.clone(), SchedNode::Leaf(alloc_name))?;
        ctx.metadata
            .new_allocations
            .entry(memory)
            .or_default()
            .comp_and_alloc_node
            .insert(node, alloc);
        ctx.state.lds_memory.insert((lds, component), alloc);
        Some(alloc)
    };
    // Each in/out register pairs POSITIONALLY with an allocation operand, the reference tensor being
    // operand zero — ⛔ [`None`] where the two lists disagree in length, or an operand names no
    // allocation and is not an `ddl.allocate` this walk can process.
    let mut in_out_reg_allocs = BTreeMap::new();
    for (slot, reg) in input_output_registers.iter().enumerate() {
        let named = operand_at(stmt.operands, slot + 1)?;
        let alloc = match ctx.interface.alloc_storage.get(&named) {
            Some(alloc) => *alloc,
            None => process_allocation(
                ctx.program,
                ctx.state,
                ctx.interface,
                ctx.metadata,
                ctx.dsc,
                &*ctx.site,
                named,
                &ctx.curr_parent.clone(),
                Some(node),
            )?,
        };
        in_out_reg_allocs.insert(reg.name, alloc);
    }
    ctx.state.computes.insert(
        node,
        ComputeNode {
            name: name.clone(),
            op,
            ex_unit,
            inputs: Vec::new(),
            outputs: Vec::new(),
            num_folds_engaged: NumFolds::ONE,
            data_format,
            instr_attribute: InstrAttribute {
                unroll: Unroll::ONE,
                precision: None,
                read_write_regs: BTreeMap::new(),
                read_only_regs: BTreeMap::new(),
                mode: None,
                compute_mask: ComputeMask::ALL,
                repetition: Repetition::ALL_SLICES,
                indices: Vec::new(),
                params: params.iter().copied().collect(),
                input_data_connects: reads.to_vec(),
                output_data_connects: writes.to_vec(),
            },
        },
    );
    ctx.state
        .add_child(&ctx.curr_parent.clone(), SchedNode::Leaf(name))?;
    ctx.metadata.opaque_ops.insert(
        node,
        OpaqueOp {
            in_out_reg_allocs,
            internal_regs: internal_registers.to_vec(),
            internal_reg_alloc,
            max_unroll,
            lds_idx: Some(lds),
        },
    );
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::SyncOp` — one `dsc2::SyncNode`, or a corelet-split pair of them under a fresh
/// `dsc2::ConditionNode` where the signal separates its corelets across exactly two.
///
/// ⛔ [`None`] IS the *"separate corelets"* mismatch — two syncs sharing a `signal_name=` must agree
/// on whether they split. ⭐ THE SPLIT IS TWO CLONES UNDER A GUARD, one per corelet, registered in
/// `syncsPerCl_[0]` and `[1]`; the unsplit case registers in the `-1` slot, which is the [`None`] key.
fn op_sync<S: DdlSite + ?Sized>(ctx: &mut OpContext<'_, S>, stmt: &Stmt) -> Option<OpOutcome> {
    let Attrs::Sync {
        units,
        signal,
        receive,
        separate_corelets,
    } = stmt.attrs
    else {
        return None;
    };
    let components: Vec<SenComponent> = units
        .iter()
        .flat_map(|unit| unroll_row_units(*unit))
        .collect();
    let (first, rest) = components.split_first()?;
    let direction = if receive {
        SyncDirection::Receive
    } else {
        SyncDirection::Send
    };
    let held = ctx.interface.sync_definitions.entry(signal).or_default();
    if !held.syncs_per_cl.is_empty() && held.separate_corelets != separate_corelets {
        return None;
    }
    held.separate_corelets = separate_corelets;
    let mut name = format!("sync_{}", signal.spelling());
    for component in &components {
        name.push('_');
        name.push_str(component.spelling());
    }
    name.push_str(if receive { "_receive" } else { "_send" });
    let node = SyncNode {
        name: NodeName(name.clone()),
        units: SyncUnits::new(*first, rest.to_vec()),
        direction,
        strength: SyncStrength::Hard,
        implicit_sync_ref_transfer: None,
        other_ends: Vec::new(),
    };
    // `syncProp.separateCorelets_ && dsc.numCoreletsUsed_DSC2_ == 2` (`:1733`) — ⛔ REVIEWED: the
    // DSC2 count and not `numCoreletsUsed_`, and the `-1` an unprepared DSC carries is `false`.
    let two_corelets = corelets_used_dsc2(ctx.dsc);
    if separate_corelets && two_corelets {
        let guard = NodeName(format!("condition_separate_corelets_{name}"));
        let then_name = NodeName(format!("{name}_then_region"));
        let else_name = NodeName(format!("{name}_else_region"));
        let mut cl0 = node.clone();
        cl0.name = NodeName(format!("{name}_cl0"));
        let mut cl1 = node.clone();
        cl1.name = NodeName(format!("{name}_cl1"));
        for (corelet, held_name) in [(corelet(0)?, &cl0.name), (corelet(1)?, &cl1.name)] {
            let ends = ctx
                .interface
                .sync_definitions
                .get_mut(&signal)?
                .syncs_per_cl
                .entry(Some(corelet))
                .or_default();
            if receive {
                ends.receivers.push(held_name.clone());
            } else {
                ends.senders.push(held_name.clone());
            }
        }
        ctx.state.mint_node(&cl0.name);
        ctx.state.mint_node(&cl1.name);
        ctx.state.mint_node(&guard);
        // ⭐ THE GUARD IS A CORE/CORELET CONDITION AND CARRIES NO LOOP TERMS: corelet 0 takes the
        // `then` branch on every used core, so `loopCond_` stays empty by construction.
        let mut core_cl_cond = BTreeMap::new();
        for core in ctx.dsc.core_ids_used.iter() {
            core_cl_cond.insert(core, BTreeSet::from([corelet(0)?]));
        }
        ctx.state.add_child(
            &ctx.curr_parent.clone(),
            SchedNode::Guarded(Box::new(ConditionNode {
                name: guard,
                loop_cond: DscLoopCondComposite::default(),
                core_cl_cond,
                then_region: vec![SchedNode::Block(BlockNode {
                    name: then_name,
                    children: vec![SchedNode::Sync(cl0)],
                })],
                else_region: vec![SchedNode::Block(BlockNode {
                    name: else_name,
                    children: vec![SchedNode::Sync(cl1)],
                })],
            })),
        )?;
    } else {
        let ends = ctx
            .interface
            .sync_definitions
            .get_mut(&signal)?
            .syncs_per_cl
            .entry(None)
            .or_default();
        if receive {
            ends.receivers.push(node.name.clone());
        } else {
            ends.senders.push(node.name.clone());
        }
        ctx.state.mint_node(&node.name);
        ctx.state
            .add_child(&ctx.curr_parent.clone(), SchedNode::Sync(node))?;
    }
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::ImplicitSyncOp` — the L0 double-buffer sync, which stands for an allocation rather than a
/// signal and so carries no `signal_name=` at all.
///
/// ⛔ [`None`] IS an allocation that is not L0 or L0_SCALE, or one whose buffer count is stated: an
/// implicit sync is only legal where the buffering is still undecided.
fn op_implicit_sync<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    stmt: &Stmt,
) -> Option<OpOutcome> {
    let allocation = operand_at(stmt.operands, 0)?;
    let alloc = *ctx.interface.alloc_storage.get(&allocation)?;
    let held = ctx.state.allocations.get(&alloc)?;
    if !matches!(held.component, SenComponent::L0 | SenComponent::L0Scale) {
        return None;
    }
    let mut components = vec![SenComponent::L0su];
    components.extend(unroll_row_units(Unit::L0lu));
    let (first, rest) = components.split_first()?;
    let name = NodeName("sync_implicit_L0".to_string());
    let node = ctx.state.mint_node(&name);
    ctx.state.add_child(
        &ctx.curr_parent.clone(),
        SchedNode::Sync(SyncNode {
            name,
            units: SyncUnits::new(*first, rest.to_vec()),
            direction: SyncDirection::Send,
            strength: SyncStrength::Hard,
            implicit_sync_ref_transfer: None,
            other_ends: Vec::new(),
        }),
    )?;
    ctx.metadata.implicit_syncs.insert(node, alloc);
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::DatastageConstraintOp` — a bound on one datastage's extent, SCALED by the dims the op names.
///
/// ⛔ [`None`] IS the empty-value-set abort, an unknown datastage, dim or reference, and a
/// [`MetaDimKind`] with no direct value. ⭐ THE SCALE IS A PRODUCT AND A QUOTIENT, AND THE DIM SET IS
/// THE FILTERED ONE: `{Padded, Unpadded, WindowDim}` join the key and multiply by their stick size,
/// every other kind instead DIVIDES by the core stage's own meta value and joins nothing — so a
/// dropped dim is skipped and not refused.
///
/// ⛔ DIVIDING BY A ZERO PAD IS THE REFERENCE'S OWN INFINITY, not an abort: `scale /=
/// getMetaDimVal(PadFront)` on an unpadded dim is `1/0` there too.
fn op_datastage_constraint<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    stmt: &Stmt,
    min: Option<&str>,
) -> Option<OpOutcome> {
    let Attrs::DatastageConstraint { values, max } = stmt.attrs else {
        return None;
    };
    let id = *ctx
        .interface
        .datastage_definition
        .get(&operand_at(stmt.operands, 0)?)?;
    let reference = operand_at(stmt.operands, 1)?;
    let against = ctx.interface.datastage_definition.get(&reference).copied();
    // A datastage reference contributes only its own identity, which is the outer key; a tensor
    // reference contributes its cumulative stick sizes.
    let stick_sizes = match against {
        Some(_) => Vec::new(),
        None => {
            let lds =
                tensor_prop(ctx.program, &mut ctx.interface.tensor_definition, reference)?.lds?;
            cumulative_stick_sizes(&ctx.site.stick_dims(lds)?, StickPart::Whole)?
        }
    };
    let core = ctx
        .dsc
        .data_stages
        .at(ctx.state.core_datastage?)?
        .ss
        .dims
        .dims();
    let mut scale = 1.0_f32;
    let mut dims = BTreeSet::new();
    for name in operand_names(stmt.operands).into_iter().flatten().skip(2) {
        let prop = ctx.interface.dim_association.get(&name)?;
        if prop.drop_dim {
            continue;
        }
        let dim = prop.dim?;
        if matches!(
            prop.meta_dim_kind,
            MetaDimKind::Padded | MetaDimKind::Unpadded | MetaDimKind::WindowDim
        ) {
            dims.insert(dim);
        } else {
            let stated = core.padding.get(&dim)?.meta_dim_val(prop.meta_dim_kind)?;
            scale /= stated as f32;
        }
    }
    for (dim, size) in stick_sizes {
        if dims.contains(&dim) {
            scale *= size.0 as f32;
        }
    }
    let key = DimSet::of(&dims.into_iter().collect::<Vec<_>>());
    let held = ctx
        .metadata
        .datastages
        .get_mut(&id)?
        .constraint_mut(against, key);
    if let Some(bound) = min {
        held.update_min(process_expression(bound)?.0 * scale);
    }
    if let Some(bound) = max {
        held.update_max(process_expression(bound)?.0 * scale);
    }
    let mut stated = Vec::with_capacity(values.len());
    for value in values {
        stated.push(process_expression(value)?.0 * scale);
    }
    if !stated.is_empty() {
        held.update_values(&stated);
        if held.values.as_ref()?.is_empty() {
            return None;
        }
    }
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::ForceInnermostDimensionsOp` — pins an allocation's innermost layout dims to one datastage's
/// extents, PREPENDING them from last to first.
///
/// ⛔ [`None`] IS an allocation that is not a tensor's, an unknown datastage, or one whose
/// `maxDimSizes_` are no longer all unset: the pin may only be stated once.
fn op_force_innermost_dimensions<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    stmt: &Stmt,
) -> Option<OpOutcome> {
    let named = operand_at(stmt.operands, 0)?;
    let alloc = match ctx.interface.alloc_storage.get(&named) {
        Some(held) => *held,
        None => process_allocation(
            ctx.program,
            ctx.state,
            ctx.interface,
            ctx.metadata,
            ctx.dsc,
            &*ctx.site,
            named,
            &ctx.curr_parent.clone(),
            None,
        )?,
    };
    let id = *ctx
        .interface
        .datastage_definition
        .get(&operand_at(stmt.operands, 1)?)?;
    let held = ctx.state.allocations.get(&alloc)?;
    if held.lds.is_none() {
        return None;
    }
    let mut pinned: Vec<(PrimaryDim, MaxDimSize)> = held.layout.iter().collect();
    if pinned.iter().any(|(_, size)| *size != MaxDimSize::Unset) {
        return None;
    }
    let dims: Vec<NameId> = operand_names(stmt.operands)
        .into_iter()
        .flatten()
        .skip(2)
        .collect();
    for name in dims.into_iter().rev() {
        let prop = ctx.interface.dim_association.get(&name)?;
        if prop.drop_dim {
            return None;
        }
        pinned.insert(0, (prop.dim?, MaxDimSize::Stage(id)));
    }
    let (first, rest) = pinned.split_first()?;
    ctx.state.allocations.get_mut(&alloc)?.layout = AllocLayout::new(*first, rest.to_vec());
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// `ddl::CoreToCoreCommunicationOp` — resolves the SFPRING ring: which core each (core, corelet)
/// sends to and receives from, plus the two conditions that name the ends of the chain.
///
/// ⛔⛔ A SINGLE SLICE RESOLVES BOTH CONDITIONS STATICALLY TRUE AND WRITES NO CORE/CORELET SET
/// (`:1951-1953`) — one core is both ends of its own chain, so entry 323's `ddl.if` mints NO
/// condition node for either. ⛔ THE ELECTED DIM IS THE FIRST MAPPED ONE, OVERRIDDEN BY A SPLIT ONE,
/// and a `dropDim_` operand is skipped. ⭐ THE TWO CORELETS WALK OPPOSITE WAYS — corelet 0 toward the
/// increasing slice, corelet 1 the other way — and corelet 1 is written ONLY where
/// `numCoreletsUsed_DSC2_ == 2`. ⛔ [`None`] IS *"More than one reduction dim is split across
/// cores"*, *"None of the dimensions is mapped"* and *"Slice not found"*.
fn op_core_to_core<S: DdlSite + ?Sized>(
    ctx: &mut OpContext<'_, S>,
    stmt: &Stmt,
) -> Option<OpOutcome> {
    let mut elected: Option<PrimaryDim> = None;
    let mut split_found = false;
    for name in operand_names(stmt.operands).into_iter().flatten() {
        // An operand that is no dim of this template, and a DROPPED dim, are both the `continue`.
        let Some(prop) = ctx.interface.dim_association.get(&name) else {
            continue;
        };
        if prop.drop_dim {
            continue;
        }
        // ⛔ AN UNASSIGNED DIM IS `numWkSlicesPerDim_.at(PrimaryDimTypesCount)`, which THROWS.
        let dim = prop.dim?;
        elected = elected.or(Some(dim));
        if ctx.site.wk_slices(dim)? > 1 {
            if split_found {
                return None;
            }
            elected = Some(dim);
            split_found = true;
        }
    }
    let dim = elected?;
    let slices = ctx.site.wk_slices(dim)?;
    let mut prop = CoreToCoreProp {
        dim: Some(dim),
        next_core: BTreeMap::new(),
        prev_core: BTreeMap::new(),
    };
    let mut resolved = None;
    let mut start = CoreClSet::default();
    let mut end = CoreClSet::default();
    if slices <= 1 {
        resolved = Some(true);
    } else {
        let last = WkSliceId(i32::try_from(slices).ok()?.checked_sub(1)?);
        let cl0 = corelet(0)?;
        let cl1 = corelet(1).filter(|_| corelets_used_dsc2(ctx.dsc));
        // ⛔ EVERY USED CORE THE SUPER-DSC STATES A SLICE FOR, not one core per slice.
        let work = ctx.site.core_work_slices();
        for (core, slice_of) in &work {
            if !ctx.dsc.core_ids_used.iter().any(|used| used == *core) {
                continue;
            }
            let at = slice_of.at(dim)?;
            let neighbour = |towards: i32| -> Option<Core> {
                let mut wanted = slice_of.clone();
                wanted.0.insert(dim, WkSliceId(towards));
                work.iter()
                    .find(|(_, other)| **other == wanted)
                    .map(|(found, _)| *found)
            };
            let chain = |set: &mut CoreClSet, corelet| {
                set.0.entry(*core).or_default().insert(corelet);
            };
            if at == WkSliceId(0) {
                chain(&mut start, cl0);
                let up = neighbour(at.0.checked_add(1)?)?;
                prop.next_core.insert((*core, cl0), up);
                if let Some(cl1) = cl1 {
                    chain(&mut end, cl1);
                    prop.prev_core.insert((*core, cl1), up);
                }
            } else if at == last {
                chain(&mut end, cl0);
                let down = neighbour(at.0.checked_sub(1)?)?;
                prop.prev_core.insert((*core, cl0), down);
                if let Some(cl1) = cl1 {
                    chain(&mut start, cl1);
                    prop.next_core.insert((*core, cl1), down);
                }
            } else {
                let up = neighbour(at.0.checked_add(1)?)?;
                let down = neighbour(at.0.checked_sub(1)?)?;
                prop.next_core.insert((*core, cl0), up);
                prop.prev_core.insert((*core, cl0), down);
                if let Some(cl1) = cl1 {
                    prop.prev_core.insert((*core, cl1), up);
                    prop.next_core.insert((*core, cl1), down);
                }
            }
        }
    }
    let results = stmt.results;
    ctx.interface
        .core_to_core_definitions
        .insert(*results.first()?, prop.clone());
    // Results 0 and 1 are the two chain-end conditions; 2 and 3 are the ring itself, which a
    // `ddl.unit` reads back through [`set_data_loc_and_info`].
    for (slot, set) in [(0_usize, start), (1, end)] {
        if let Some(name) = results.get(slot) {
            ctx.interface.resolved_conditions.insert(
                *name,
                CondProp {
                    resolved,
                    loop_cond: LoopCondComposite::default(),
                    core_cl_cond: set,
                },
            );
        }
    }
    for slot in 2..4 {
        if let Some(name) = results.get(slot) {
            ctx.interface
                .core_to_core_definitions
                .insert(*name, prop.clone());
        }
    }
    Some(OpOutcome {
        parent: Some(ctx.curr_parent.clone()),
        regions: Vec::new(),
    })
}

/// Replaces: e323_processOp
///
/// ONE DDL OP TURNED INTO SCHEDULE — the dispatcher that mints the loop, transfer, compute, sync and
/// condition nodes, and answers where this op's regions attach.
///
/// ⛔ [`None`] IS EVERY `emitError`/`DT_ERROR` OF THE FOURTEEN ARMS, including
/// `ddl.core_corelet_cond`, whose *"Uses of this op is Prohibited in ddl"* is unconditional.
/// ⭐ AN OP THAT MINTS NOTHING STILL ANSWERS: an empty loop band, a resolved `ddl.if` and every
/// non-tree op hand back an insertion block, which is what keeps the walk descending.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's own parameter list"
)]
pub fn process_op<S: DdlSite + ?Sized>(
    program: &Program,
    state: &mut DdlConversion,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &mut DesignSpaceConfig,
    site: &mut S,
    op: &DdlOp<'_>,
    curr_parent: &NodeName,
) -> Option<OpOutcome> {
    let mut ctx = OpContext {
        program,
        state,
        interface,
        metadata,
        dsc,
        site,
        curr_parent: curr_parent.clone(),
    };
    match *op {
        DdlOp::Loop(stmt) => op_loop(&mut ctx, stmt),
        DdlOp::ParametricLoop(stmt) => op_parametric_loop(&mut ctx, stmt),
        DdlOp::DataTransfer { stmt, pattern_dims } => {
            op_data_transfer(&mut ctx, stmt, pattern_dims)
        }
        DdlOp::Compute(stmt) => op_compute(&mut ctx, stmt),
        DdlOp::Datastage(stmt) => op_datastage(&mut ctx, stmt),
        DdlOp::GetExternalDatastage(stmt) => op_get_external_datastage(&mut ctx, stmt),
        DdlOp::If {
            condition,
            then_region,
            else_region,
        } => op_if(&mut ctx, condition, then_region, else_region),
        DdlOp::Opaque(stmt) => op_opaque(&mut ctx, stmt),
        DdlOp::Sync(stmt) => op_sync(&mut ctx, stmt),
        DdlOp::ImplicitSync(stmt) => op_implicit_sync(&mut ctx, stmt),
        DdlOp::DatastageConstraint { stmt, min } => op_datastage_constraint(&mut ctx, stmt, min),
        DdlOp::ForceInnermostDimensions(stmt) => op_force_innermost_dimensions(&mut ctx, stmt),
        DdlOp::CoreToCoreCommunication(stmt) => op_core_to_core(&mut ctx, stmt),
        DdlOp::CoreCoreletCond => None,
    }
}
//   authority : ddc/ddl/ddl_conversion.cpp:582  (1424 body lines, level 2)
//   class     : DdlConversion
//   original  : std::pair<dsc2::BlockNode*, std::vector<int>> DdlConversion::processOp( Operation& op, dsc2::BlockNode* currParent)
//   extract   : crustify-ddc/cpp/ddl.cpp:1656-3081
//   calls     : e069_updateMin, e070_updateMax, e071_updateValues, e096_updateMin, e097_updateMax, e098_updateValues, e173_addInternalTensor, e174_processExpression, e175_getTensorProp, e176_getTensor, e187_clear, e275_processCondition

// ⭐ TYPES FOR ENTRY 325 — THE EMITTED DDL. They are OWNED because [`Program`] and [`Stmt`] are
// `&'static` build-time tables while the reference MUTATES its module: it sets two attributes on the
// parsed ops and then builds a SECOND `ddl.dataflow` out of the schedule tree.

/// WHAT ONE `ddl.dimension` RESULT MAPS TO — the `dim_mapping=` array's per-result entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimMapping {
    /// `"ignored"` — no association at all, or one whose `dim_` has no spelling.
    Ignored,
    /// `primaryDimToString.at(dimprop.dim_)`.
    Dim(PrimaryDim),
}

/// HOW MANY TIMES A LOOP RUNS OVER ONE DIM — `ceil(numstg.ss_ / denstg.ss_)`, the `ss_loop_count=`
/// dictionary's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopCount(pub u64);

/// A LOOP'S NUMERATOR OR DENOMINATOR STAGE — the `ddl.get_external_datastage` or `ddl.datastage` the
/// reference mints beside each loop.
///
/// ⭐ MEMOISING IS THE MECHANISM AND NOT THE RESULT: two loops naming the same stage share ONE op
/// there and share one value here.
/// ⛔ TRAP: A `ddl.datastage`-MINTED STAGE READS BACK AS EXTERNAL, because entry 323 names it after
/// its own index and the test here is whether that name is EMPTY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmittedStage {
    /// `property=` — `chunk` for `chunk_dstgid`, `core` for every other named stage.
    External(DatastageProperty),
    /// `strategy=`, from `metadata_.datastages_[id]`, whose `operator[]` default is `minimize`.
    Minted(Strategy),
}

/// WHICH TENSOR AN EMITTED OP NAMES.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmittedTensor {
    /// A tensor the census names.
    Named(NameId),
    /// A `ddl.internal_tensor` MINTED beside the new dataflow for an interim LDS no censused tensor
    /// names, carrying the tensor it takes its type from and the LDS it stands for.
    ///
    /// ⛔ DIVERGENCE: the reference ALSO writes `tensor_definition_[tensor].ldsIdx_`, and a minted op
    /// has no censused [`NameId`] to key that by. Only a NON-MEMORY end reads that map for an LDS, so
    /// that is the whole of what is unobservable here.
    Internal {
        /// The `ddl.tensor` whose type the minted op takes.
        reference: NameId,
        /// `allocatenode->ldsIdx_`.
        lds: LdsIdx,
    },
}

/// ONE EMITTED `ddl.allocate`'S IDENTITY — the `AllocateOp` the reference's `allocations` map is
/// keyed by, and what a later `ddl.unit` states as its allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocOp(pub u32);

/// ONE ENTRY OF `llvm::DenseMap<AllocateOp, const dsc2::AllocateNode*>` — an emitted `ddl.allocate`,
/// the tensor it names and the DSC node it was emitted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmittedAllocate {
    /// `allocateop.getResult()`.
    pub op: AllocOp,
    /// `allocateop.getTensor()`.
    pub tensor: EmittedTensor,
    /// The `dsc2::AllocateNode*` the map holds against it.
    pub node: AllocId,
}

/// ONE EMITTED `ddl.unit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedUnit {
    /// [`tensor`]'s answer, or the first half of [`tensor_and_allocation`]'s pair.
    pub tensor: Option<EmittedTensor>,
    /// The second half of that pair, absent for every non-memory end.
    pub allocation: Option<AllocOp>,
    /// `unit=` — a transfer end's own `unit_`, and for a compute operand naming a memory the
    /// compute's `exUnit_`.
    pub unit: SenComponent,
    /// `data_connect=`.
    pub data_connect: Option<DataConnect>,
    /// `stick_offset=` — the FIRST constant element offset the end carries, absent where it carries
    /// none.
    pub stick_offset: Option<Elements>,
    /// `vias=` — a destination's route, EMPTY on a compute operand and on a transfer source.
    pub vias: Vec<SenComponent>,
}

/// ONE OPERAND OF AN EMITTED `ddl.compute`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmittedOperand {
    /// A tensor defined by a `ddl.operand_constant`, which is pushed AS ITSELF with no `ddl.unit`.
    Constant(NameId),
    /// Every other operand.
    Unit(EmittedUnit),
}

/// ONE TERM OF AN EMITTED `ddl.condition`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmittedCondTerm {
    /// The DDL dim tested, absent where no association names it — the reference's null `Value`.
    pub dim: Option<NameId>,
    /// `op=`.
    pub op: DscCondOp,
    /// `cond_val_type=` together with `condValInt_`.
    pub bound: LoopBound,
}

/// WHAT AN EMITTED `ddl.if` TESTS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmittedCond {
    /// `ddl.core_corelet_cond` — the corelets stated per core.
    CoreCorelet(BTreeMap<Core, BTreeSet<Corelet>>),
    /// `ddl.condition_or` over `ddl.condition_and`s, wrapped in a `ddl.condition_not` when negated.
    Loop {
        /// `twoLevelOrOfAnds_`.
        ors: Vec<Vec<EmittedCondTerm>>,
        /// `negated_`.
        negated: bool,
    },
}

/// WHAT AN EMITTED `ddl.sync` SIGNALS ON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncLabel {
    /// A `signal_name=` the template declares, found as a SUBSTRING of the node's name.
    Signal(SyncSignal),
    /// An L3 sync no template declares — one end's OWN name, SHARED with the other ends, which is
    /// what makes the pair one signal.
    External(NodeName),
}

/// ONE OP OF THE EMITTED DATAFLOW.
///
/// ⭐ NO `Block` VARIANT: a `BLOCK` node emits NOTHING — the reference only saves and restores the
/// insertion point around it — so its children flatten into the enclosing region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmittedOp {
    /// `ddl.loop`.
    Loop {
        /// `name=`.
        name: NodeName,
        /// `label=`, absent for the reference's empty string.
        label: Option<LoopLabel>,
        /// The numerator stage.
        num: EmittedStage,
        /// The denominator stage.
        den: EmittedStage,
        /// The DDL dims the loop's own `dims_` resolved to, in order.
        dims: Vec<NameId>,
        /// `ss_loop_count=`.
        ss_loop_count: Vec<(PrimaryDim, LoopCount)>,
        /// The loop body.
        body: Vec<EmittedOp>,
    },
    /// `ddl.compute`.
    Compute {
        /// `name=`.
        name: NodeName,
        /// `computetype=`.
        op: DdlComputeType,
        /// `unit=`.
        ex_unit: SenComponent,
        /// `mode=`, whose absence IS the reference's `-1`.
        mode: Option<Mode>,
        /// `mask=`.
        mask: ComputeMask,
        /// `repetition=`.
        repetition: Repetition,
        /// `indices=`.
        indices: Vec<PackIndex>,
        /// The input operands.
        inputs: Vec<EmittedOperand>,
        /// The output operands.
        outputs: Vec<EmittedOperand>,
    },
    /// `ddl.data_transfer`.
    DataTransfer {
        /// `name=`.
        name: NodeName,
        /// `access_pattern_style=` paired with the dim each style describes, which the op states as
        /// its own operands.
        access_pattern_styles: Vec<(NameId, String)>,
        /// `limit_stick_replicated=`, stated only for a POSITIVE `force_num_elements_`.
        limit_stick_replicated: Option<Elements>,
        /// `rotate_num_elements=`.
        rotate: Elements,
        /// The source unit.
        src: EmittedUnit,
        /// The destination units.
        dsts: Vec<EmittedUnit>,
        /// `transfer_size=` — see [`DdlSizes::block_transfer_size`].
        transfer_size: Option<Elements>,
    },
    /// `ddl.if`.
    If {
        /// `name=`.
        name: NodeName,
        /// The condition op the `ddl.if` takes.
        cond: EmittedCond,
        /// The then-region.
        then_region: Vec<EmittedOp>,
        /// The else-region, EMPTY where the condition node states only one.
        else_region: Vec<EmittedOp>,
    },
    /// `ddl.sync`.
    Sync {
        /// `name=`.
        name: NodeName,
        /// `units=`.
        units: Vec<SenComponent>,
        /// `is_receive=`.
        direction: SyncDirection,
        /// `signal_name=`.
        label: SyncLabel,
        /// `separate_corelets=`.
        separate_corelets: bool,
    },
    /// `ddl.implicit_sync`, which states only the allocation its reference transfer writes.
    ImplicitSync {
        /// `name=`.
        name: NodeName,
        /// The destination allocation of `implicitSyncRefTransfer_`.
        allocation: Option<AllocOp>,
    },
    /// `ddl.allocate`.
    Allocate {
        /// `name=`.
        name: NodeName,
        /// This op's own identity, which a later `ddl.unit` names.
        op: AllocOp,
        /// The tensor it places.
        tensor: EmittedTensor,
        /// `padding_type=` paired with the PADDED DDL dim it describes.
        ///
        /// ⭐ A [`PadType`] AND NOT A STRING and still exactly faithful, because `getPaddingAsStr` is
        /// `padTypeToString.at(getPadding(dim))` — see [`pad_type_spelling`].
        padding_styles: Vec<(NameId, PadType)>,
        /// `component=`.
        component: SenComponent,
        /// `num_buffers=`.
        num_buffers: NumBuffers,
        /// `layout_dim_order=`.
        layout_dim_order: Vec<PrimaryDim>,
        /// `allocation_size=`, stated only for an allocation that names an LDS.
        allocation_size: Option<Elements>,
    },
    /// `ddl.generic` — a STICK MASK, the only node kind that reaches it, so its `nodetype=` is
    /// invariably `STICKMASK`.
    Generic {
        /// `name=`.
        name: NodeName,
    },
}

/// THE EMITTED DDL — the module `convertDsc2Ddl` leaves behind.
///
/// ⭐ THE TWO `original_dataflow=` FLAGS ARE THE IDENTITY OF THE TWO DATAFLOWS: the census program is
/// the `true` one, and [`Self::dataflow`] is the `false` one the schedule tree becomes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmittedDdl {
    /// `dim_mapping=`, per `ddl.dimension` RESULT — the reference sets one array over `getResults()`,
    /// out of which a result is recoverable by position.
    pub dim_mapping: BTreeMap<NameId, DimMapping>,
    /// `lds_mapping=`, per `ddl.tensor` result, whose [`None`] is the reference's `-1`.
    pub lds_mapping: BTreeMap<NameId, Option<LdsIdx>>,
    /// The new `ddl.dataflow`'s body, in emission order.
    pub dataflow: Vec<EmittedOp>,
}

/// THE TWO SIZES ENTRY 325 ASKS OF A DSC THAT ARE NOT PORTED YET — a NARROW trait beside
/// [`AllocationSite`] and [`DdlSite`], because neither call is in this campaign's file list and
/// neither is reachable from the state the conversion itself holds.
pub trait DdlSizes: AllocationSite {
    /// `getBlockTransferSize(node, src_.unit_, 0, false, true)` FOLDED WITH the
    /// `!getRelevantCoreCl().empty()` that gates it — absent IS a transfer that states no
    /// `transfer_size=` at all.
    fn block_transfer_size(&self, transfer: NodeId) -> Option<Elements>;

    /// `getBufferCapacityForNode(allocatenode, allocatenode->ldsIdx_, component_, 0, 0)`.
    fn buffer_capacity(&self, alloc: AllocId) -> Option<Elements>;
}

/// Replaces: e325_convertDsc2Ddl
///
/// EMITS THE WHOLE DSC BACK AS DDL — the dim and lds mappings onto the parsed ops, then a SECOND
/// `ddl.dataflow` holding the schedule tree. The `std::ostream&` is NEVER USED and the body ends in
/// `ddlMlirRoot->dump()`, so the emission IS the return value and printing it is the caller's.
///
/// ⛔ [`None`] IS EVERY `.at()` ON THE WAY, the `DT_CHECK` that every loop dim resolved to a DDL dim,
/// and the `llvm_unreachable` on an allocation naming neither an LDS nor a constant.
/// ⛔ TRAP: `dataStageParam_` IS READ WITH `[]` AND THEN WITH `.at()`, so a loop naming a stage the
/// DSC does not state passes the name test and is refused by the trip count.
pub fn convert_dsc2_ddl<S: DdlSizes + ?Sized>(
    program: &Program,
    state: &DdlConversion,
    interface: &DdlInterface,
    metadata: &Metadata,
    dsc: &DesignSpaceConfig,
    site: &S,
) -> Option<EmittedDdl> {
    let mut dim_mapping = BTreeMap::new();
    let mut lds_mapping = BTreeMap::new();
    for stmt in program.stmts {
        for result in stmt.results {
            match stmt.kind {
                StmtKind::Dimension => {
                    dim_mapping.insert(
                        *result,
                        interface
                            .dim_association
                            .get(result)
                            .and_then(|prop| prop.dim)
                            .map_or(DimMapping::Ignored, DimMapping::Dim),
                    );
                }
                StmtKind::Tensor => {
                    lds_mapping.insert(
                        *result,
                        interface
                            .tensor_definition
                            .get(result)
                            .and_then(|prop| prop.lds),
                    );
                }
                _ => {}
            }
        }
    }
    let mut emission = Emission {
        program,
        state,
        interface,
        metadata,
        dsc,
        site,
        allocations: Vec::new(),
        external_syncs: BTreeMap::new(),
        next_op: 0,
    };
    let dataflow = emission.region(&state.tree.head().children)?;
    Some(EmittedDdl {
        dim_mapping,
        lds_mapping,
        dataflow,
    })
}

/// THE WALK'S OWN STATE — the allocate ops every later `ddl.unit` resolves through, and the labels
/// the two ends of an L3 sync share.
struct Emission<'a, S: DdlSizes + ?Sized> {
    program: &'a Program,
    state: &'a DdlConversion,
    interface: &'a DdlInterface,
    metadata: &'a Metadata,
    dsc: &'a DesignSpaceConfig,
    site: &'a S,
    allocations: Vec<EmittedAllocate>,
    external_syncs: BTreeMap<NodeName, NodeName>,
    next_op: u32,
}

impl<S: DdlSizes + ?Sized> Emission<'_, S> {
    /// Every node of one region, in order.
    ///
    /// ⭐ THE EXPLICIT DFS STACK IS MECHANISM: the `visited` map, the insertion-point stack and the
    /// `ifop_then`/`ifop_else` pair are how the reference reaches the children that a recursive walk
    /// over a child vector reaches by itself.
    fn region(&mut self, nodes: &[SchedNode]) -> Option<Vec<EmittedOp>> {
        let mut out = Vec::new();
        for node in nodes {
            match node {
                SchedNode::Loop(held) => out.push(self.loop_op(held)?),
                SchedNode::Guarded(held) => {
                    let cond = self.cond(held);
                    let then_region = self.region(&held.then_region)?;
                    let else_region = self.region(&held.else_region)?;
                    out.push(EmittedOp::If {
                        name: held.name.clone(),
                        cond,
                        then_region,
                        else_region,
                    });
                }
                SchedNode::Condition(block) => self.bare_condition(block, &mut out)?,
                SchedNode::Block(block) => out.extend(self.region(&block.children)?),
                SchedNode::Sync(sync) => out.push(self.sync(sync)?),
                SchedNode::StickMask(mask) => out.push(EmittedOp::Generic {
                    name: mask.name.clone(),
                }),
                SchedNode::Leaf(name) => self.leaf(name, &mut out)?,
            }
        }
        Some(out)
    }

    /// A `CONDITION` NODE REACHED WITHOUT ITS REGIONS: it states no predicate at all, so the
    /// reference takes the core/corelet arm over an EMPTY dictionary, and only its child BLOCKS move
    /// the insertion point — the first into the then-region, a later one into the else-region where
    /// the node has more than one child, and every other child beside the `ddl.if` itself.
    fn bare_condition(&mut self, block: &BlockNode, out: &mut Vec<EmittedOp>) -> Option<()> {
        let two_regions = block.children.len() > 1;
        let mut then_region = None;
        let mut else_region = None;
        let mut beside = Vec::new();
        for child in &block.children {
            match child {
                SchedNode::Block(inner) if then_region.is_none() => {
                    then_region = Some(self.region(&inner.children)?);
                }
                SchedNode::Block(inner) if two_regions && else_region.is_none() => {
                    else_region = Some(self.region(&inner.children)?);
                }
                other => beside.extend(self.region(core::slice::from_ref(other))?),
            }
        }
        out.push(EmittedOp::If {
            name: block.name.clone(),
            cond: EmittedCond::CoreCorelet(BTreeMap::new()),
            then_region: then_region.unwrap_or_default(),
            else_region: else_region.unwrap_or_default(),
        });
        out.extend(beside);
        Some(())
    }

    /// `hasCoreClCond()`'s two arms.
    fn cond(&self, node: &ConditionNode) -> EmittedCond {
        if node.has_core_cl_cond() {
            return EmittedCond::CoreCorelet(node.core_cl_cond.clone());
        }
        let ors = node
            .loop_cond
            .two_level_or_of_ands
            .iter()
            .map(|ands| {
                ands.iter()
                    .map(|term| EmittedCondTerm {
                        dim: self.dim_named(term.dim),
                        op: term.op,
                        bound: term.bound,
                    })
                    .collect()
            })
            .collect();
        EmittedCond::Loop {
            ors,
            negated: node.loop_cond.negated,
        }
    }

    /// The FIRST DDL dim associated with this primary dim — `getDim`, whose early `return` over an
    /// `unordered_map` leaves the winner unspecified where two names share a dim.
    fn dim_named(&self, dim: PrimaryDim) -> Option<NameId> {
        self.interface
            .dim_association
            .iter()
            .find(|(_, prop)| prop.dim == Some(dim))
            .map(|(name, _)| *name)
    }

    /// One `ddl.loop` and its body.
    fn loop_op(&mut self, node: &LoopNode) -> Option<EmittedOp> {
        let (interface, dsc) = (self.interface, self.dsc);
        let id = self.state.node_ids.get(&node.block.name).copied();
        // ⛔ EVERY LOOP LABEL IS COMPARED AND NONE BREAKS, so the LAST match wins.
        let label = interface
            .loop_labels
            .iter()
            .filter(|(_, held)| Some(**held) == id)
            .map(|(label, _)| *label)
            .last();
        let num = self.stage(node.num);
        let den = self.stage(node.den);
        let mut dims = Vec::new();
        for held in &node.dims {
            // The `DT_CHECK` that every loop dim resolved: a `WindowDim` loop dim is answered by an
            // UNPADDED association too, and nothing else crosses kinds.
            let (name, _) = interface.dim_association.iter().find(|(_, prop)| {
                prop.dim == Some(held.dim)
                    && (prop.meta_dim_kind == held.kind
                        || (matches!(held.kind, MetaDimKind::WindowDim)
                            && matches!(prop.meta_dim_kind, MetaDimKind::Unpadded)))
            })?;
            dims.push(*name);
        }
        let num_stage = dsc.data_stages.at(node.num?)?;
        let den_stage = dsc.data_stages.at(node.den?)?;
        let mut ss_loop_count = Vec::new();
        for held in &node.dims {
            let numerator = u64::try_from(num_stage.ss_extent(held.dim)?.0).ok()?;
            let denominator = u64::try_from(den_stage.ss_extent(held.dim)?.0).ok()?;
            if denominator == 0 {
                // `std::ceil(num * 1.0 / 0)` is an infinity cast to `size_t` — undefined there.
                return None;
            }
            ss_loop_count.push((held.dim, LoopCount(numerator.div_ceil(denominator))));
        }
        let body = self.region(&node.block.children)?;
        Some(EmittedOp::Loop {
            name: node.block.name.clone(),
            label,
            num,
            den,
            dims,
            ss_loop_count,
            body,
        })
    }

    /// `dataStageParam_[id].name().empty()` deciding between the two datastage ops, with
    /// `metadata_.datastages_[id]`'s own `operator[]` default behind the minted arm.
    fn stage(&self, id: Option<DatastageId>) -> EmittedStage {
        let named = id
            .and_then(|id| self.dsc.data_stages.stage_name(id))
            .is_some_and(|name| !name.0.is_empty());
        match id {
            Some(id) if named => EmittedStage::External(if id == Metadata::CHUNK_DSTGID {
                DatastageProperty::Chunk
            } else {
                DatastageProperty::Core
            }),
            _ => EmittedStage::Minted(
                id.and_then(|id| self.metadata.datastages.get(&id))
                    .map_or(Strategy::Minimize, |held| held.strategy),
            ),
        }
    }

    /// One `ddl.sync` — or the `ddl.implicit_sync` that states the allocation its reference transfer
    /// writes, and nothing else.
    fn sync(&mut self, node: &SyncNode) -> Option<EmittedOp> {
        if let Some(transfer) = node.implicit_sync_ref_transfer {
            let dst = *self.state.transfers.get(&transfer)?.dsts.first();
            let pair = self.pair(dst.storage, &dst.data)?;
            return Some(EmittedOp::ImplicitSync {
                name: node.name.clone(),
                allocation: pair.allocate,
            });
        }
        let stated = self
            .interface
            .sync_definitions
            .iter()
            .find(|(signal, _)| node.name.0.contains(signal.spelling()))
            .map(|(signal, prop)| (*signal, prop.separate_corelets));
        let (label, separate_corelets) = match stated {
            Some((signal, separate)) => (SyncLabel::Signal(signal), separate),
            None => {
                // `emplace` KEEPS THE FIRST ENTRY, which is what makes the other end share this name.
                if !self.external_syncs.contains_key(&node.name) {
                    self.external_syncs
                        .insert(node.name.clone(), node.name.clone());
                    for other in &node.other_ends {
                        self.external_syncs
                            .entry(other.clone())
                            .or_insert_with(|| node.name.clone());
                    }
                }
                (
                    SyncLabel::External(self.external_syncs.get(&node.name)?.clone()),
                    false,
                )
            }
        };
        Some(EmittedOp::Sync {
            name: node.name.clone(),
            units: node.units.iter().collect(),
            direction: node.direction,
            label,
            separate_corelets,
        })
    }

    /// An `ALLOCATE`, `COMPUTE` or `TRANSFER` leaf — the three node kinds that carry no children.
    fn leaf(&mut self, name: &NodeName, out: &mut Vec<EmittedOp>) -> Option<()> {
        let state = self.state;
        if let Some(id) = state.node_ids.get(name).copied() {
            if let Some(compute) = state.computes.get(&id) {
                out.push(self.compute(id, compute)?);
                return Some(());
            }
            if let Some(transfer) = state.transfers.get(&id) {
                out.push(self.transfer(id, transfer)?);
                return Some(());
            }
        }
        let (alloc, node) = state
            .allocations
            .iter()
            .find(|(_, held)| held.name == *name)?;
        self.allocate(*alloc, node, out)
    }

    /// One `ddl.compute`, whose operands take the compute's OWN `exUnit_` wherever the operand names
    /// a memory.
    fn compute(&self, id: NodeId, node: &ComputeNode) -> Option<EmittedOp> {
        let mut inputs = Vec::new();
        for (at, operand) in node.inputs.iter().enumerate() {
            let end = NodeEnd::Input(u32::try_from(at).ok()?);
            inputs.push(self.operand(id, end, node.ex_unit, operand)?);
        }
        let mut outputs = Vec::new();
        for (at, operand) in node.outputs.iter().enumerate() {
            let end = NodeEnd::Output(u32::try_from(at).ok()?);
            outputs.push(self.operand(id, end, node.ex_unit, operand)?);
        }
        Some(EmittedOp::Compute {
            name: node.name.clone(),
            op: node.op,
            ex_unit: node.ex_unit,
            mode: node.instr_attribute.mode,
            mask: node.instr_attribute.compute_mask,
            repetition: node.instr_attribute.repetition,
            indices: node.instr_attribute.indices.clone(),
            inputs,
            outputs,
        })
    }

    /// One compute operand: a `ddl.unit` unless the tensor is a `ddl.operand_constant`, which is
    /// pushed as itself.
    fn operand(
        &self,
        id: NodeId,
        end: NodeEnd,
        ex_unit: SenComponent,
        operand: &DscOperand,
    ) -> Option<EmittedOperand> {
        let unit = self.unit(
            id,
            end,
            operand.unit,
            (ex_unit, operand.unit),
            &operand.data,
            Vec::new(),
        )?;
        if let Some(EmittedTensor::Named(tensor)) = unit.tensor {
            if self
                .program
                .definition(tensor)
                .is_some_and(|stmt| stmt.kind == StmtKind::OperandConstant)
            {
                return Some(EmittedOperand::Constant(tensor));
            }
        }
        Some(EmittedOperand::Unit(unit))
    }

    /// One `ddl.data_transfer` — its two ends, the styles its access-pattern dims resolve to, and the
    /// size the DSC states for a transfer that names a core or corelet at all.
    ///
    /// ⛔ A TRANSFER END LOOKS ITS TENSOR UP BY `storage_` AND STILL SPELLS `unit_`, both branches.
    fn transfer(&self, id: NodeId, node: &TransferNode) -> Option<EmittedOp> {
        let src = self.unit(
            id,
            NodeEnd::Src,
            node.src.storage,
            (node.src.unit, node.src.unit),
            &node.src.data,
            Vec::new(),
        )?;
        let mut dsts = Vec::new();
        for (at, dst) in node.dsts.iter().enumerate() {
            let index = u32::try_from(at).ok()?;
            dsts.push(self.unit(
                id,
                NodeEnd::Dst(index),
                dst.storage,
                (dst.unit, dst.unit),
                &dst.data,
                node.dsts.hops(at).to_vec(),
            )?);
        }
        // ⭐ `transfer_acc_pat_dims_[node]` AND `metadata_.datatransfers_[node]` ARE BOTH `[]`, which
        // is what makes the `.at()` beside the first total: a transfer stating no dims states no
        // styles, and one the DDC never touched reads a default-constructed entry.
        let default = DataTransfer::default();
        let meta = self.metadata.datatransfers.get(&id).unwrap_or(&default);
        let mut access_pattern_styles = Vec::new();
        for name in self
            .interface
            .transfer_acc_pat_dims
            .get(&id)
            .map_or(&[][..], |dims| &dims[..])
        {
            let dim = self
                .interface
                .dim_association
                .get(name)
                .and_then(|prop| prop.dim)?;
            access_pattern_styles.push((*name, meta.access_pattern_spelling(dim)?));
        }
        Some(EmittedOp::DataTransfer {
            name: node.name.clone(),
            access_pattern_styles,
            limit_stick_replicated: meta.force_num_elements.filter(|count| count.0 > 0),
            rotate: self
                .state
                .rotate_elements
                .get(&id)
                .copied()
                .unwrap_or(Elements(0)),
            src,
            dsts,
            transfer_size: self.site.block_transfer_size(id),
        })
    }

    /// One `ddl.unit`: the tensor and allocation the end resolves to, and the FIRST constant element
    /// offset it carries. `spelling` is `unit=` when `component` is a memory and when it is not.
    fn unit(
        &self,
        id: NodeId,
        end: NodeEnd,
        component: SenComponent,
        spelling: (SenComponent, SenComponent),
        data: &DataInfo,
        vias: Vec<SenComponent>,
    ) -> Option<EmittedUnit> {
        let (tensor, allocation, spelled) = if is_memory(component) {
            let pair = self.pair(component, data)?;
            (pair.tensor, pair.allocate, spelling.0)
        } else {
            let named = tensor(
                &self.interface.tensor_definition,
                &self.interface.ext_constant_definition,
                &self.interface.operand_constant_tensor,
                component,
                data_origin(data),
            );
            (named.map(EmittedTensor::Named), None, spelling.1)
        };
        Some(EmittedUnit {
            tensor,
            allocation,
            unit: spelled,
            data_connect: data.data_connect,
            stick_offset: self
                .state
                .const_ele_offsets
                .get(&(id, end))
                .and_then(|offsets| offsets.values().next().copied()),
            vias,
        })
    }

    /// [`tensor_and_allocation`] over the allocate ops emitted so far.
    fn pair(&self, unit: SenComponent, data: &DataInfo) -> Option<TensorAndAllocation> {
        tensor_and_allocation(
            self.site,
            &self.allocations,
            &self.interface.tensor_definition,
            &self.interface.ext_constant_definition,
            &self.interface.operand_constant_tensor,
            unit,
            data_origin(data),
        )
    }

    /// One `ddl.allocate`, and the entry every later `ddl.unit` resolves through.
    ///
    /// ⛔ A TEMP STORAGE FOR A COMPUTE EMITS NOTHING, and so does an allocation whose tensor nothing
    /// names — the reference guards the whole op on a non-null tensor.
    fn allocate(
        &mut self,
        alloc: AllocId,
        node: &AllocateNode,
        out: &mut Vec<EmittedOp>,
    ) -> Option<()> {
        if node.temp_storage_for_compute.is_some() {
            return Some(());
        }
        let tensor = match (node.lds, node.const_idx) {
            (Some(lds), _) => self.allocated_tensor(lds)?,
            (None, Some(constant)) => self
                .interface
                .ext_constant_definition
                .iter()
                .find(|(_, held)| **held == constant)
                .map(|(name, _)| EmittedTensor::Named(*name)),
            // `llvm_unreachable("invalid idx")`.
            (None, None) => return None,
        };
        let Some(tensor) = tensor else {
            return Some(());
        };
        // ⛔ NO `break` ON THE INNER LOOP: every PADDED association of a padded dim states a style.
        let mut padding_styles = Vec::new();
        for dim in node.placement.padding.dims() {
            for (name, prop) in &self.interface.dim_association {
                if prop.dim == Some(dim) && matches!(prop.meta_dim_kind, MetaDimKind::Padded) {
                    padding_styles.push((*name, node.placement.padding.get(dim)));
                }
            }
        }
        let op = AllocOp(self.next_op);
        self.next_op += 1;
        out.push(EmittedOp::Allocate {
            name: node.name.clone(),
            op,
            tensor,
            padding_styles,
            component: node.component,
            num_buffers: node.placement.num_buffers,
            // ⛔ EMPTY FOR A CONSTANT ALLOCATION: `layoutDimOrder_` is written only in the LDS arm
            // of `processAllocation`, and this attribute is set from it unconditionally (`:3460`).
            layout_dim_order: match node.lds {
                Some(_) => node.layout.iter().map(|(dim, _)| dim).collect(),
                None => Vec::new(),
            },
            allocation_size: node.lds.and_then(|_| self.site.buffer_capacity(alloc)),
        });
        self.allocations.push(EmittedAllocate {
            op,
            tensor,
            node: alloc,
        });
        Some(())
    }

    /// WHICH TENSOR AN LDS ALLOCATION PLACES — the censused tensor whose post-DDC index is this one,
    /// else a `ddl.internal_tensor` minted against the tensor naming its EXTERNAL index.
    ///
    /// ⛔ THE INNER [`None`] IS THE REFERENCE'S NULL TENSOR — an allocation no tensor claims, which
    /// emits no op at all — while the OUTER one is the SECOND loop's unguarded `ldsIdxAfterDdc.at()`,
    /// which the first loop's `count()` guards and the second does not.
    fn allocated_tensor(&self, lds: LdsIdx) -> Option<Option<EmittedTensor>> {
        for (name, prop) in &self.interface.tensor_definition {
            let Some(held) = prop.lds else { continue };
            if self.metadata.lds_idx_after_ddc.get(&held) == Some(&lds) {
                return Some(Some(EmittedTensor::Named(*name)));
            }
        }
        for (name, prop) in &self.interface.tensor_definition {
            if self
                .program
                .definition(*name)
                .is_none_or(|stmt| stmt.kind != StmtKind::Tensor)
            {
                continue;
            }
            let Some(external) = self.metadata.interm_lds_idx_to_ext_lds.get(&lds) else {
                continue;
            };
            if self.metadata.lds_idx_after_ddc.get(&prop.lds?)? == external {
                return Some(Some(EmittedTensor::Internal {
                    reference: *name,
                    lds,
                }));
            }
        }
        Some(None)
    }
}

/// WHERE ONE END'S DATA COMES FROM — `myLdsIdx_` before `constantId_`, the order every reader of a
/// [`DataInfo`] tests them in.
const fn data_origin(data: &DataInfo) -> Option<DataOrigin> {
    match (data.my_lds_idx, data.constant_id) {
        (Some(lds), _) => Some(DataOrigin::LabeledDs(lds)),
        (None, Some(constant)) => Some(DataOrigin::Constant(constant)),
        (None, None) => None,
    }
}

//   authority : ddc/ddl/ddl_conversion.cpp:2881  (627 body lines, level 2)
//   class     : DdlConversion
//   original  : void DdlConversion::convertDsc2Ddl(std::ostream& outputDdl)
//   extract   : crustify-ddc/cpp/ddl.cpp:3136-3763
//   calls     : e176_getTensor, e177_dump, e179_getAccessPatternAsStr, e183_dump, e277_getTensorAndAllocation

/// Replaces: e345_exportToDdl
///
/// THE EXPORT ENTRY — `convertDsc2Ddl` verbatim, whose `std::ostream&` it forwards and which never
/// writes it: the reference's only output is `ddlMlirRoot->dump()` to `llvm::errs()` (`:3507`), so
/// the emitted DDL IS the result and printing it is the caller's.
pub fn export_to_ddl<S: DdlSizes + ?Sized>(
    program: &Program,
    state: &DdlConversion,
    interface: &DdlInterface,
    metadata: &Metadata,
    dsc: &DesignSpaceConfig,
    site: &S,
) -> Option<EmittedDdl> {
    convert_dsc2_ddl(program, state, interface, metadata, dsc, site)
}

//   authority : ddc/ddl/ddl_conversion.cpp:38  (3 body lines, level 3)
//   class     : DdlConvertInterface
//   original  : void DdlConvertInterface::exportToDdl(std::ostream& outputDdl)
//   extract   : crustify-ddc/cpp/ddl.cpp:3795-3798
//   calls     : e325_convertDsc2Ddl

/// ONE OP OF A DDL REGION AND THE REGIONS IT OPENS — `op.getRegion(i)` for each index entry 323
/// answers, in the order the walk descends them.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionOp<'d> {
    /// The op itself.
    pub op: DdlOp<'d>,
    /// The regions this op owns, in `getRegions()` order.
    pub regions: Vec<RegionId>,
}

/// EVERY REGION OF ONE DDL TEMPLATE — the ops each region holds, in `getOperations()` order.
///
/// ⭐ A PARAMETER AND NOT A CENSUS READ: the generated tables record a depth and a path, never the
/// `mlir::Region` identity `region2blocks_` is keyed by, and `ddl.if` has no statement at all.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RegionTree<'d> {
    /// The ops of each region.
    pub ops: BTreeMap<RegionId, Vec<RegionOp<'d>>>,
}

/// Replaces: e346_processRegion
///
/// WALKS ONE DDL REGION INTO THE SCHEDULE TREE — every op in order, then each region it opens, under
/// a fresh `<parent>_region<n>` block where the op opened more than one.
///
/// ⛔ [`None`] IS THE `DT_CHECK(numRegions == 0 || insertionPoint != nullptr)` and every refusal
/// [`process_op`] answers with. ⛔ TRAP: the reference is ALWAYS positional (`op.getRegion(i)`); an
/// outcome region here is an IDENTITY where the op's own list names it, which agrees because only
/// `ddl.if` answers more than one and it answers identities — every positional arm answers one, where
/// both readings land on `regions[0]`.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's own member state"
)]
pub fn process_region<S: DdlSite + ?Sized>(
    program: &Program,
    state: &mut DdlConversion,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &mut DesignSpaceConfig,
    site: &mut S,
    regions: &RegionTree<'_>,
    region: RegionId,
    curr_parent: &NodeName,
) -> Option<()> {
    interface.region2blocks.insert(region, curr_parent.clone());
    for held in regions.ops.get(&region).map_or(&[][..], Vec::as_slice) {
        let outcome = process_op(
            program,
            state,
            interface,
            metadata,
            dsc,
            site,
            &held.op,
            curr_parent,
        )?;
        let num_regions = outcome.regions.len();
        if num_regions == 0 {
            continue;
        }
        let insertion = outcome.parent?;
        for (at, entry) in outcome.regions.iter().enumerate() {
            let op_region = if held.regions.contains(entry) {
                *entry
            } else {
                *held.regions.get(usize::try_from(entry.0).ok()?)?
            };
            let parent = if num_regions > 1 {
                let name = NodeName(format!("{}_region{at}", insertion.0));
                state.add_block_child(
                    &insertion,
                    BlockNode {
                        name: name.clone(),
                        children: Vec::new(),
                    },
                )?;
                name
            } else {
                insertion.clone()
            };
            process_region(
                program, state, interface, metadata, dsc, site, regions, op_region, &parent,
            )?;
        }
    }
    Some(())
}

//   authority : ddc/ddl/ddl_conversion.cpp:2008  (26 body lines, level 3)
//   class     : DdlConversion
//   original  : void DdlConversion::processRegion(::mlir::Region& myRegion, dsc2::BlockNode* currParent)
//   extract   : crustify-ddc/cpp/ddl.cpp:3808-3835
//   calls     : e323_processOp

/// ONE `ddl.operation_bind`, AS ENTRY 347 IS HANDED IT.
///
/// ⛔ A PARAMETER AND NOT A CENSUS READ: `build.rs`'s walk folds each bind into [`Program::op_func`]
/// and the per-name roles and keeps NO `ddl.operation_bind` statement, so the pre-order the reference
/// walks and every operand list it reads have to be stated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationBind {
    /// `getResult()`.
    pub result: NameId,
    /// `stringToOpFuncs.at(getOpFuncName())`, whose miss is *"Unrecognized opFuncName"*.
    pub op_func: Option<OpFunc>,
    /// `getRequired()`.
    pub required: bool,
    /// `getDataFormats()`.
    pub data_formats: Vec<NameId>,
    /// `getInputs()`.
    pub inputs: Vec<NameId>,
    /// `getOutputs()`.
    pub outputs: Vec<NameId>,
    /// `getInterim()`.
    pub interim: Vec<NameId>,
}

/// ONE `dsc.computeOp_` ENTRY AS THE BIND SEARCH TESTS IT — the four fields it compares plus the two
/// exclude sets it turns into a core/corelet include set.
///
/// ⛔ DIVERGENCE: `DscComputeOp` carries no exclude set and no `interimLabeledDs`, and its operands
/// are `DataInfo`s rather than indices; this states the questions the search actually asks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindableOp {
    /// `opFuncName`.
    pub op_func: OpFunc,
    /// `attributes_.dataFormat_`, absent for the reference's `INVALID`.
    pub format: Option<DataFormat>,
    /// `inputLabeledDs`, as indices.
    pub inputs: Vec<LdsIdx>,
    /// `outputLabeledDs`, as indices.
    pub outputs: Vec<LdsIdx>,
    /// `coreExclude`.
    pub core_exclude: BTreeSet<Core>,
    /// `coreClExclude`.
    pub core_cl_exclude: BTreeSet<(Core, Corelet)>,
}

/// WHAT THE MATCH READS AND WRITES ON THE DSC beyond what [`DdlSite`] already answers.
pub trait MatchSite: DdlSite {
    /// `dsc.computeOp_`, IN ORDER — a [`ComputeOpIdx`] is a position in this vector.
    fn bindable_ops(&self) -> Vec<BindableOp>;

    /// `dsc.labeledDs_.at(lds).wordLength`.
    fn lds_word_length(&self, lds: LdsIdx) -> Option<WordLength>;

    /// `newLds.wordLength = type->bitSize_ / 8`.
    fn set_lds_word_length(&mut self, lds: LdsIdx, length: WordLength);

    /// `newLds.dataFormat_ = type->dataFormat_`.
    fn set_lds_format(&mut self, lds: LdsIdx, format: DataFormat);
}

/// `op->getOperand(at)` as a list, which is how a `[%type, ..]` group is spelled — total over a lone
/// name too, since the census records a one-element group either way.
fn operand_group(operands: &[Operand], at: usize) -> Option<Vec<NameId>> {
    match operands.get(at)? {
        Operand::One(name) => Some(vec![*name]),
        Operand::List(names) => Some(names.to_vec()),
        Operand::OtherBind => None,
    }
}

/// `layout_op.getDimensions()` with `getIsOrderFixed()`, and [`None`] for `DT_ERROR("Illegal ddl")`.
///
/// ⛔ REVIEWED: THE *"This op is used as a layout, but it is not"* DIAGNOSTIC IS NEVER PRINTED — the
/// reference calls `emitError` ON THE NULL `LayoutOp` its own failed `dyn_cast` just handed back
/// (`:2296-2297`), so it dereferences null before reaching the `DT_ERROR` this [`None`] stands for.
fn layout_dims_of(program: &Program, layout: NameId) -> Option<(Vec<NameId>, bool)> {
    let stmt = program.definition(layout)?;
    let Attrs::Layout { order_fixed } = stmt.attrs else {
        return None;
    };
    Some((
        operand_names(stmt.operands).into_iter().flatten().collect(),
        order_fixed,
    ))
}

/// `matchTensor` — ties one bind operand to a labeled DS, minting a stack LDS for an interim tensor
/// and recording a regular tensor's three layouts for the dimension pass.
///
/// ⛔ [`None`] IS *"Multiple operation_bind ops use above tensor with different meaning"*, *"Regular
/// tensor op cannot be used as interim tensor in operation_bind"*, *"Type incompatible with tensor
/// role in operation"* and *"Use of non-tensor op when expected"*.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's own capture list"
)]
fn match_tensor<S: MatchSite + ?Sized>(
    program: &Program,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    site: &mut S,
    layout_ops: &mut Vec<NameId>,
    compute_op: ComputeOpIdx,
    format: Option<DataFormat>,
    tensor: NameId,
    lds: Option<LdsIdx>,
) -> Option<()> {
    let stmt = program.definition(tensor)?;
    if stmt.kind == StmtKind::AliasOneTensorOf {
        tensor_prop(program, &mut interface.tensor_definition, tensor)?;
    }
    if let Some(prop) = interface.tensor_definition.get(&tensor) {
        return (prop.lds == lds).then_some(());
    }
    interface
        .tensor_definition
        .insert(tensor, TensorProp::default());
    match stmt.kind {
        StmtKind::Tensor => {
            let lds = lds?;
            interface
                .tensor_definition
                .insert(tensor, TensorProp { lds: Some(lds) });
            let types = process_types(program, &operand_group(stmt.operands, 3)?)?;
            let held = site.lds_format(lds)?;
            types.iter().find(|ty| ty.format == held)?;
            for at in 0..3 {
                layout_ops.push(operand_at(stmt.operands, at)?);
            }
            Some(())
        }
        StmtKind::InternalTensor => {
            let reference = operand_at(stmt.operands, 0)?;
            let reference =
                tensor_prop(program, &mut interface.tensor_definition, reference)?.lds?;
            let new_lds = add_internal_tensor(site, interface, metadata, reference, compute_op)?;
            let types = process_types(program, &operand_group(stmt.operands, 1)?)?;
            let ty = match types.as_slice() {
                [only] => *only,
                many => *many.iter().find(|ty| Some(ty.format) == format)?,
            };
            site.set_lds_word_length(new_lds, WordLength(ty.bit_size.0 / 8));
            site.set_lds_format(new_lds, ty.format);
            interface
                .tensor_definition
                .insert(tensor, TensorProp { lds: Some(new_lds) });
            Some(())
        }
        _ => None,
    }
}

/// `addDimConstraints` — narrows every DDL dim's candidate set from one layout: the dims this layout
/// states get the layout's own dims, and every other unpadded dim loses them.
///
/// ⛔ [`None`] IS *"Not enough dimensions"* and *"Fixed layout with too many dimensions"*, both
/// *"Impossible to match for sdsc"*. ⛔ ASKING IS A MUTATION — every read is an `operator[]`.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's own capture list"
)]
fn add_dim_constraints(
    program: &Program,
    interface: &mut DdlInterface,
    bcasted: &BTreeSet<PrimaryDim>,
    ddl_padded_dims: &mut BTreeSet<NameId>,
    layout: NameId,
    layout_dims: &[PrimaryDim],
    is_global: bool,
) -> Option<()> {
    let (op_dims, order_fixed) = layout_dims_of(program, layout)?;
    if op_dims.is_empty() {
        return Some(());
    }
    let dims_set: BTreeSet<PrimaryDim> = layout_dims.iter().copied().collect();
    let no_bcast: BTreeSet<PrimaryDim> = dims_set.difference(bcasted).copied().collect();
    if op_dims.len() < no_bcast.len() || (order_fixed && op_dims.len() > layout_dims.len()) {
        return None;
    }
    let mut no_pad: Vec<NameId> = Vec::with_capacity(op_dims.len());
    for op_dim in &op_dims {
        let prop = interface.dim_association.entry(*op_dim).or_default();
        let dim = if matches!(prop.meta_dim_kind, MetaDimKind::Padded) {
            let unpadded = prop.non_padded_dim?;
            ddl_padded_dims.insert(*op_dim);
            ddl_padded_dims.insert(unpadded);
            unpadded
        } else {
            *op_dim
        };
        no_pad.push(dim);
        if is_global {
            interface
                .dim_association
                .entry(dim)
                .or_default()
                .global_layout_refs
                .0 += 1;
        }
    }
    for at in 0..no_pad.len() {
        let op_dim = no_pad[at];
        if !order_fixed {
            if is_global || op_dims.len() <= layout_dims.len() {
                interface
                    .dim_association
                    .entry(op_dim)
                    .or_default()
                    .dim_candidates
                    .retain(|dim| dims_set.contains(dim));
            }
            continue;
        }
        let Some(&fixed) = layout_dims.get(at) else {
            interface
                .dim_association
                .entry(op_dim)
                .or_default()
                .dim_candidates
                .clear();
            continue;
        };
        interface
            .dim_association
            .entry(op_dim)
            .or_default()
            .dim_candidates
            .retain(|dim| *dim == fixed);
        for (name, other) in &mut interface.dim_association {
            if *name == op_dim || matches!(other.meta_dim_kind, MetaDimKind::Padded) {
                continue;
            }
            other.dim_candidates.retain(|dim| *dim != fixed);
        }
    }
    if !order_fixed {
        for (name, prop) in &mut interface.dim_association {
            if matches!(prop.meta_dim_kind, MetaDimKind::Padded) || no_pad.contains(name) {
                continue;
            }
            prop.dim_candidates.retain(|dim| !no_bcast.contains(dim));
        }
    }
    Some(())
}

/// `tryDimMapping` — the backtracking assignment of DSC dims to DDL dims, in the pruned order, which
/// drops every DDL dim left over once each DSC dim is covered.
///
/// ⛔ ASKING IS A MUTATION: `dscDimsMapped[candidateDim]` default-inserts a dim that was never a
/// mapping target, and the reference's own `all_of` then reads it back as UNMAPPED.
fn try_dim_mapping(
    interface: &mut DdlInterface,
    dims: &[NameId],
    mapped: &mut BTreeMap<PrimaryDim, bool>,
    at: usize,
) -> bool {
    let all_done = mapped.values().all(|done| *done);
    let Some(&dim) = dims.get(at) else {
        return all_done;
    };
    if all_done {
        interface.dim_association.entry(dim).or_default().drop_dim = true;
        try_dim_mapping(interface, dims, mapped, at + 1);
        return true;
    }
    let candidates = {
        let prop = interface.dim_association.entry(dim).or_default();
        prop.drop_dim = false;
        prop.dim_candidates.clone()
    };
    for candidate in candidates {
        if *mapped.entry(candidate).or_insert(false) {
            continue;
        }
        interface.dim_association.entry(dim).or_default().dim = Some(candidate);
        mapped.insert(candidate, true);
        if try_dim_mapping(interface, dims, mapped, at + 1) {
            return true;
        }
        mapped.insert(candidate, false);
    }
    interface.dim_association.entry(dim).or_default().drop_dim = true;
    try_dim_mapping(interface, dims, mapped, at + 1)
}

/// Replaces: e347_matchDdl2Dsc
///
/// MAPS ONE DDL TEMPLATE ONTO THE DSC: binds each `ddl.operation_bind` to a compute op, mints an LDS
/// per interim tensor, prunes each DDL dim's candidates from the layouts it appears in, then
/// backtracks a dim assignment covering every non-broadcast DSC dim and fills the padded dims.
///
/// ⛔ `Some(false)` IS A **SILENT** FALSE — *"Template not suitable for DSC"* is COMMENTED OUT at
/// `:2285`, the tree's only occurrence — and [`None`] IS EVERY `DT_ERROR`.
/// ⛔ TRAP: AN LDS INDEX IS POSITIONAL where a format, word length or stick is read from it and
/// RECORDED where a layout is — the reference's `labeledDs_.at()` conflates the two.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's own member state"
)]
pub fn match_ddl2_dsc<S: MatchSite + ?Sized>(
    program: &Program,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &mut DesignSpaceConfig,
    site: &mut S,
    binds: &[OperationBind],
    padded: &BTreeMap<NameId, PaddedDimension<'_>>,
) -> Option<bool> {
    let mut failed = false;
    let compute_ops = site.bindable_ops();
    let mut mapped_ops = vec![false; compute_ops.len()];
    let mut layout_ops: Vec<NameId> = Vec::new();
    for bind in binds {
        let op_func = bind.op_func?;
        let types = process_types(program, &bind.data_formats)?;
        let mut matched = None;
        for (at, candidate) in compute_ops.iter().enumerate() {
            if mapped_ops[at]
                || candidate.op_func != op_func
                || candidate.inputs.len() != bind.inputs.len()
                || candidate.outputs.len() != bind.outputs.len()
                || (!types.is_empty()
                    && !types.iter().any(|ty| Some(ty.format) == candidate.format))
            {
                continue;
            }
            matched = Some(ComputeOpIdx(at));
            mapped_ops[at] = true;
            break;
        }
        let Some(compute_op) = matched else {
            failed |= bind.required;
            continue;
        };
        let comp = compute_ops.get(compute_op.0)?;
        {
            let cores: Vec<Core> = dsc.core_ids_used.iter().collect();
            let corelets = corelets_of(dsc);
            let prop = interface
                .operation_definition
                .entry(bind.result)
                .or_default();
            prop.compute_op = Some(compute_op);
            if !comp.core_cl_exclude.is_empty() || !comp.core_exclude.is_empty() {
                for core in cores {
                    if comp.core_exclude.contains(&core) {
                        continue;
                    }
                    for corelet in &corelets {
                        if !comp.core_cl_exclude.contains(&(core, *corelet)) {
                            prop.core_cl_cond
                                .0
                                .entry(core)
                                .or_default()
                                .insert(*corelet);
                        }
                    }
                }
            }
        }
        for (at, tensor) in bind.outputs.iter().enumerate() {
            match_tensor(
                program,
                interface,
                metadata,
                site,
                &mut layout_ops,
                compute_op,
                comp.format,
                *tensor,
                Some(*comp.outputs.get(at)?),
            )?;
        }
        for (at, tensor) in bind.inputs.iter().enumerate() {
            match_tensor(
                program,
                interface,
                metadata,
                site,
                &mut layout_ops,
                compute_op,
                comp.format,
                *tensor,
                Some(*comp.inputs.get(at)?),
            )?;
        }
        for tensor in &bind.interim {
            match_tensor(
                program,
                interface,
                metadata,
                site,
                &mut layout_ops,
                compute_op,
                comp.format,
                *tensor,
                None,
            )?;
        }
        if bind.interim.is_empty() {
            continue;
        }
        let shift = u32::try_from(bind.interim.len()).ok()?;
        let old = LdsIdx(
            site.labeled_ds_tail()?
                .insert_position
                .0
                .checked_sub(shift)?,
        );
        let new = LdsIdx(old.0 + shift);
        for slot in site.lds_slots() {
            if !matches!(
                slot,
                LdsSlot::TransferSrc(_) | LdsSlot::TransferDst(_, _) | LdsSlot::Allocate(_)
            ) {
                continue;
            }
            if site.slot_lds(slot) == Some(old) {
                site.set_slot_lds(slot, new);
            }
        }
        // ⛔ THE KEYS FIRST: the reference `extract`s the element its own loop iterator points at and
        // then `it++`s that now-invalidated iterator (`:2269-2278`).
        let prefilled = &mut metadata.prefilled_external_transfer_data_connects;
        let keys: Vec<(LdsIdx, ExternalStorage)> = prefilled
            .keys()
            .filter(|(lds, _)| *lds == old)
            .copied()
            .collect();
        for key in keys {
            if let Some(slot) = prefilled.remove(&key) {
                prefilled.insert((new, key.1), slot);
            }
        }
    }
    if mapped_ops.contains(&false) {
        failed = true;
    }
    if failed {
        return Some(false);
    }
    let mut dims_to_map: Vec<NameId> = Vec::new();
    for layout in &layout_ops {
        let (dims, _) = layout_dims_of(program, *layout)?;
        for ddl_dim in dims {
            if !interface.dim_association.contains_key(&ddl_dim) {
                match program.definition(ddl_dim)?.kind {
                    StmtKind::PaddedDimension => {
                        process_padded_dimension_op(
                            program,
                            interface,
                            ddl_dim,
                            *padded.get(&ddl_dim)?,
                        )?;
                    }
                    StmtKind::Dimension => {
                        process_dimension_op(program, interface, ddl_dim)?;
                    }
                    _ => return None,
                }
            }
            let prop = interface.dim_association.get(&ddl_dim)?;
            let entry = if matches!(
                prop.meta_dim_kind,
                MetaDimKind::Unpadded | MetaDimKind::WindowDim
            ) {
                ddl_dim
            } else {
                let unpadded = prop.non_padded_dim?;
                interface.dim_association.get(&unpadded)?;
                unpadded
            };
            if !dims_to_map.contains(&entry) {
                dims_to_map.push(entry);
            }
        }
    }
    let mut dsc_dims_mapped: BTreeMap<PrimaryDim, bool> = BTreeMap::new();
    let mut processed_lds: BTreeSet<LdsIdx> = BTreeSet::new();
    let mut ddl_padded_dims: BTreeSet<NameId> = BTreeSet::new();
    let tensors: Vec<(NameId, Option<LdsIdx>)> = interface
        .tensor_definition
        .iter()
        .map(|(name, prop)| (*name, prop.lds))
        .collect();
    for (name, lds) in tensors {
        let stmt = program.definition(name)?;
        if stmt.kind != StmtKind::Tensor {
            continue;
        }
        let lds = lds?;
        if !processed_lds.insert(lds) {
            continue;
        }
        let recorded = dsc.labeled_ds.at(lds)?.recorded();
        let global_dims = dsc.layout_dims.get(&recorded)?.to_vec();
        let non_bcasted = dsc.non_broadcast_lds_dim_set(recorded)?;
        let mut bcasted: BTreeSet<PrimaryDim> = BTreeSet::new();
        for dim in &global_dims {
            if non_bcasted.contains(dim) {
                dsc_dims_mapped.entry(*dim).or_insert(false);
            } else {
                bcasted.insert(*dim);
            }
        }
        let elem_in_slice = 128u64.checked_div(u64::from(site.lds_word_length(lds)?.0))? / 8;
        let mut slice_dims: Vec<PrimaryDim> = Vec::new();
        let mut stick_dims: Vec<PrimaryDim> = Vec::new();
        let mut num_elem = 1u64;
        for (dim, size) in site.stick_dims(lds)?.0 {
            if num_elem < elem_in_slice {
                slice_dims.push(dim);
            }
            num_elem = num_elem.saturating_mul(size.0);
            if num_elem > elem_in_slice {
                stick_dims.push(dim);
            }
        }
        for (at, dims, is_global) in [
            (0, &slice_dims, false),
            (1, &stick_dims, false),
            (2, &global_dims, true),
        ] {
            add_dim_constraints(
                program,
                interface,
                &bcasted,
                &mut ddl_padded_dims,
                operand_at(stmt.operands, at)?,
                dims,
                is_global,
            )?;
        }
    }
    let mut dsc_padded: BTreeSet<PrimaryDim> = BTreeSet::new();
    let mut dsc_window: BTreeSet<PrimaryDim> = BTreeSet::new();
    for (dim, padding) in &dsc.core_stage().dims().padding {
        dsc_padded.insert(*dim);
        if let Some(window) = padding.window_dim {
            dsc_window.insert(window);
        }
    }
    for (op_dim, prop) in &mut interface.dim_association {
        if !ddl_padded_dims.contains(op_dim) {
            prop.dim_candidates.retain(|dim| !dsc_padded.contains(dim));
        }
        if !matches!(prop.meta_dim_kind, MetaDimKind::WindowDim) {
            prop.dim_candidates.retain(|dim| !dsc_window.contains(dim));
        }
    }
    dims_to_map.sort_by_key(|dim| {
        interface
            .dim_association
            .get(dim)
            .map_or((GlobalLayoutRefs(0), 0), |prop| {
                (prop.global_layout_refs, prop.dim_candidates.len())
            })
    });
    if dims_to_map.len() < dsc_dims_mapped.len()
        || !try_dim_mapping(interface, &dims_to_map, &mut dsc_dims_mapped, 0)
    {
        return None;
    }
    // ⛔ A SNAPSHOT OF THE KEYS: the reference reads `dim_association_[nonPaddedDim]` — an
    // `operator[]` that may INSERT — while iterating that same map (`:2511-2515`).
    for dim in interface
        .dim_association
        .keys()
        .copied()
        .collect::<Vec<_>>()
    {
        let (kind, non_padded) = {
            let prop = interface.dim_association.entry(dim).or_default();
            (prop.meta_dim_kind, prop.non_padded_dim)
        };
        if matches!(kind, MetaDimKind::Unpadded) {
            continue;
        }
        let (drop_dim, base) = match non_padded {
            Some(unpadded) => {
                let prop = interface.dim_association.entry(unpadded).or_default();
                (prop.drop_dim, prop.dim)
            }
            None => (false, None),
        };
        interface.dim_association.entry(dim).or_default().drop_dim = drop_dim;
        if drop_dim {
            continue;
        }
        if matches!(kind, MetaDimKind::WindowDim) {
            let window = dsc.core_stage().dims().padding.get(&base?)?.window_dim?;
            interface.dim_association.entry(dim).or_default().dim = Some(window);
            continue;
        }
        interface.dim_association.entry(dim).or_default().dim = base;
        if let Some(target) = base {
            dsc.full_padding.entry(target).or_default();
            dsc.data_stages.ensure_padding(target);
        }
    }
    check_meta_dimensions(program, interface, dsc)?;
    Some(true)
}

//   authority : ddc/ddl/ddl_conversion.cpp:2110  (442 body lines, level 3)
//   class     : DdlConversion
//   original  : bool DdlConversion::matchDdl2Dsc()
//   extract   : crustify-ddc/cpp/ddl.cpp:3845-4287
//   calls     : e021_getOpFuncName, e172_processTypes, e173_addInternalTensor, e175_getTensorProp, e177_dump, e183_dump, e187_clear, e272_print, e274_processDimensionOp, e278_checkMetaDimensions, e322_processPaddedDimensionOp

/// THE PARSED DDL ROOT AS ITS TWO PRE-ORDER WALKS SEE IT — every region of the module, plus the body
/// of each `ddl.dataflow` and each `ddl.transformations` in walk order.
///
/// ⛔ A PARAMETER AND NOT [`crate::schedule::ddl::DdlModuleOp`]'S OWN VALUE, for the same reason
/// [`RegionTree`] is one: `ddlMlirRoot->walk<WalkOrder::PreOrder>` is the MECHANISM for reaching a
/// region, the generated tables carry no `mlir::Region` identity, and what entry 363 stores is the
/// verifier witnesses rather than a region graph.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DdlRoot<'d> {
    /// Every region of the module, which is what [`process_region`] descends.
    pub regions: RegionTree<'d>,
    /// `dfOp.getBody()` per `ddl.dataflow`.
    pub dataflows: Vec<RegionId>,
    /// `trOp.getBody()` per `ddl.transformations`.
    pub transformations: Vec<Vec<Transformation>>,
}

/// `insertOtherEnds` (`ddl_conversion.cpp:2798-2817`) — every node of `other_end` appended to the
/// other-end list of every node of `base`.
///
/// ⛔ BOTH `DT_ERROR`S ARE [`None`]: an empty `other_end` is *"does not have matching syncs"*, and one
/// unit on two nodes of `base` is *"Multiple sync ops ... have same unit"* — the unit set spans the
/// WHOLE list, so that check is ACROSS the nodes and not within one.
/// ⛔ AND A THIRD IS THIS PORT'S OWN: [`SyncEnds`] links by NAME where the reference holds live tree
/// pointers, so a name no `SYNC` node of the tree carries is a link that cannot be followed.
fn insert_other_ends(
    state: &mut DdlConversion,
    base: &[NodeName],
    other_end: &[NodeName],
) -> Option<()> {
    if other_end.is_empty() {
        return None;
    }
    let mut units: BTreeSet<SenComponent> = BTreeSet::new();
    for name in base {
        let node = state.tree.find_sync_mut(|sync| sync.name == *name)?;
        for unit in node.units.iter() {
            if !units.insert(unit) {
                return None;
            }
        }
        node.other_ends.extend(other_end.iter().cloned());
    }
    Some(())
}

/// Replaces: e364_parseDdl2Dsc
///
/// WALKS THE MATCHED DDL INTO THE SCHEDULE TREE — every `ddl.dataflow` body under one root-level
/// block, then every `ddl.transformations` body, then the two ends of every sync signal onto each
/// other. The diagnostics are [`process_transformations`]'s, forwarded.
///
/// ⛔ [`None`] IS `DT_CHECK(metadata_.belowLxScheduleInsertBlock)`, every refusal the two walks
/// answer with, and both `DT_ERROR`s of the pairing.
/// ⛔ TRAP: `root_level_operations` IS ALWAYS MINTED IN PRACTICE — `belowLxScheduleInsertBlock` is
/// found by `traverseTreeDFSMutable`, which seeds from `head_.next_` and so never yields `head_`
/// (`dsc/dsc2.cpp:2233`) — but the comparison is the reference's and is kept, not collapsed.
pub fn parse_ddl2_dsc<S: DdlSite + ?Sized>(
    program: &Program,
    state: &mut DdlConversion,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &mut DesignSpaceConfig,
    site: &mut S,
    root: &DdlRoot<'_>,
) -> Option<Vec<String>> {
    let head = state.tree.head().name.clone();
    let below = metadata.below_lx_schedule_insert_block?;
    let insertion = if state.block_name(below) == Some(head.clone()) {
        head
    } else {
        let name = NodeName("root_level_operations".to_owned());
        state
            .tree
            .head_mut()
            .add_child_front(SchedNode::Block(BlockNode {
                name: name.clone(),
                children: Vec::new(),
            }));
        name
    };
    for region in &root.dataflows {
        process_region(
            program,
            state,
            interface,
            metadata,
            dsc,
            site,
            &root.regions,
            *region,
            &insertion,
        )?;
    }
    let mut said = Vec::new();
    for body in &root.transformations {
        said.extend(process_transformations(
            program, interface, metadata, dsc, body,
        )?);
    }
    // `// connect sync nodes` (`:2796`) — every signal, per corelet, in both directions.
    for prop in interface.sync_definitions.values() {
        for ends in prop.syncs_per_cl.values() {
            insert_other_ends(state, &ends.senders, &ends.receivers)?;
            insert_other_ends(state, &ends.receivers, &ends.senders)?;
        }
    }
    Some(said)
}

//   authority : ddc/ddl/ddl_conversion.cpp:2770  (54 body lines, level 4)
//   class     : DdlConversion
//   original  : void DdlConversion::parseDdl2Dsc()
//   extract   : crustify-ddc/cpp/ddl.cpp:4316-4370
//   calls     : e324_processTransformations, e346_processRegion

/// THE PER-CANDIDATE FACTS ONE `.ddl` STATES — what [`match_ddl2_dsc`], [`verify_ddl_constraint`] and
/// [`parse_ddl2_dsc`] each take beyond the census, plus the buffer [`DdlModuleOp::parse_ddl`] reads.
pub struct StatedTemplate<'d, P> {
    /// The buffer `ddlTemplateDir + ddlFile` names.
    pub source: P,
    /// The walked program — the census's side of the same template.
    pub program: &'d Program,
    /// Every `ddl.operation_bind`, in the pre-order the match walks.
    pub binds: Vec<OperationBind>,
    /// Every `ddl.padded_dimension`, keyed by its own result.
    pub padded: BTreeMap<NameId, PaddedDimension<'d>>,
    /// Every `ddl.constraint` the module states, with the operands it names — `verifyDdlConstraints`
    /// is a pre-order walk over `ConstraintOp` (`ddl_conversion.cpp:2553`) and entry 276 is one stop
    /// of it, so the walk is the caller's.
    pub constraints: Vec<(&'d [Operand], DdlConstraint)>,
    /// The regions [`parse_ddl2_dsc`] walks.
    pub root: DdlRoot<'d>,
}

/// WHERE A CANDIDATE TEMPLATE IS READ FROM — one [`Template`] to everything it states.
///
/// ⛔ ITS OWN SEAM AND NOT A METHOD ON [`MatchSite`]: what a template states is held ACROSS the `&mut`
/// that the match and the walk take, so one seam for both would be a borrow that cannot be reborrowed.
pub trait DdlTemplateSet {
    /// The buffer type this set parses its candidates from.
    type Source: DdlSource;

    /// What `template` states, [`None`] where this set does not hold it.
    fn stated(&self, template: Template) -> Option<StatedTemplate<'_, Self::Source>>;
}

/// WHAT ONE SELECTION ANSWERS — `matchedDdlFile`, and the complaints instead of `std::cout`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DdlSelection {
    /// The `.ddl` whose match succeeded; [`None`] is each of the reference's three `false`s.
    pub template: Option<Template>,
    /// What `verbose_ >= 0` printed, plus [`parse_ddl2_dsc`]'s own diagnostics.
    pub said: Vec<String>,
}

/// Replaces: e372_selectAndParseDdlTemplate
///
/// PARSES THE FIRST CANDIDATE `.ddl` THAT MATCHES ONTO THE DSC — the op-func of `computeOp_.front()`
/// names an ordered candidate list, and each candidate is parsed and verified, matched, its
/// constraints checked and then WALKED INTO THE SCHEDULE TREE; one that does not match is RELEASED and
/// the walk goes on.
///
/// ⛔ [`None`] IS THE `ENABLE_LN32` ABORT, every refusal the four callees answer with, and a parse
/// yielding nothing. The `enableLn32 = true; continue` after that abort is DEAD, since `DT_ERROR`
/// throws; the `DEEPTOOLS_PATH` abort is unspellable, because the template directory is how a
/// candidate is REACHED and one arrives here as a [`Template`].
/// ⛔ REVIEWED: THE TWO PARSE FAILURES ARE NOT THE SAME FAILURE, and the earlier note named the
/// wrong one. A buffer that OPENS and then parses to nothing RAISES — `DdlMain`'s own `DT_CHECK(op)`
/// (`ddl.cpp:106`); it is the buffer that does not OPEN AT ALL that returns `nullptr` from `:103`,
/// three lines BEFORE that check, leaving `ddl_module_op_` null for `matchDdl2Dsc` to walk as a null
/// root (`ddl_conversion.cpp:2111`). [`DdlModuleOp::module`] answers [`None`] for both, so this stops
/// on either.
/// ⛔ THE ARCH SKIP IS [`ddl_templates`]'S, resolved at build time: MPW4 has no [`crate::arch::IsaGen`].
/// ⛔ TRAP: `ddl_templates` MUST NAME EVERY ROW OF `opFuncToDdlTemplate`, because its [`None`] is the
/// *"no DDL available"* answer. `StzLatch` is one row it cannot name yet — see `OP_FUNC_TEMPLATES` in
/// `ddl/selection.rs`.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference's own member state"
)]
pub fn select_and_parse_ddl_template<A: Arch, S: MatchSite + ?Sized, T: DdlTemplateSet + ?Sized>(
    parser: &mut DdlModuleOp,
    state: &mut DdlConversion,
    interface: &mut DdlInterface,
    metadata: &mut Metadata,
    dsc: &mut DesignSpaceConfig,
    site: &mut S,
    templates: &T,
    ln32: Ln32,
) -> Option<DdlSelection> {
    let Some(first) = site.bindable_ops().first().map(|op| op.op_func) else {
        return Some(DdlSelection::default());
    };
    let Some(candidates) = ddl_templates(first.spelling(), A::GEN) else {
        return Some(DdlSelection {
            template: None,
            said: vec![format!(
                "[DDC] no DDL available for op {}",
                first.spelling()
            )],
        });
    };
    for template in candidates {
        if matches!(first, OpFunc::Exx2 | OpFunc::LayernormScale) && ln32 == Ln32::On {
            return None;
        }
        let stated = templates.stated(*template)?;
        interface.clear();
        parser.parse_ddl(Some(stated.source));
        parser.module()?;
        if match_ddl2_dsc(
            stated.program,
            interface,
            metadata,
            dsc,
            site,
            &stated.binds,
            &stated.padded,
        )? {
            for (operands, constraint) in &stated.constraints {
                verify_ddl_constraint::<A>(stated.program, interface, dsc, operands, *constraint)?;
            }
            let said = parse_ddl2_dsc(
                stated.program,
                state,
                interface,
                metadata,
                dsc,
                site,
                &stated.root,
            )?;
            return Some(DdlSelection {
                template: Some(*template),
                said,
            });
        }
        parser.release();
    }
    Some(DdlSelection {
        template: None,
        said: vec![format!(
            "[DDC] DDL found but not suitable for op {}",
            first.spelling()
        )],
    })
}

//   authority : ddc/ddl/ddl_conversion.cpp:42  (59 body lines, level 5)
//   class     : DdlConversion
//   original  : bool DdlConversion::selectAndParseDdlTemplate()
//   extract   : crustify-ddc/cpp/ddl.cpp:4380-4439
//   calls     : e187_clear, e276_verifyDdlConstraints, e347_matchDdl2Dsc, e363_parseDdl, e364_parseDdl2Dsc

#[cfg(test)]
mod unit_tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU32;

    use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

    use super::{
        AllocOp, AllocationSite, BindableOp, ComputeOpIdx, CondProp, ConstraintCmp, DdlConstraint,
        DdlConversion, DdlInterface, DdlOp, DdlSite, DdlSizes, DimMapping, DimProp,
        EmittedAllocate, EmittedDdl, EmittedOp, EmittedStage, EmittedTensor, ExprValue,
        GlobalLayoutRefs, InternalTensor, InternalTensorSite, LabeledDsTail, LdsSlot, LoopCount,
        MatchSite, OpContext, OpOutcome, OperationBind, PaddedDimension, RegionId, RegionOp,
        RegionTree, StyledDims, SyncLabel, TensorAndAllocation, TensorProp, TypeDefinition,
        add_internal_tensor, allocation_pad_type, check_meta_dimensions, convert_dsc2_ddl, corelet,
        export_to_ddl, match_ddl2_dsc, op_core_to_core, pad_type_spelling, process_access_patterns,
        process_condition, process_dimension_op, process_expression, process_region, process_types,
        tensor, tensor_and_allocation, tensor_prop, transfer_access_pattern, verify_ddl_constraint,
    };
    use crate::arch::{Dd2, Elements, IsaGen};
    use crate::bridges::superdsc_to_dataflow_ir::control_flow::{CondOp, CondValType};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PrimaryDim, StickDims,
    };
    use crate::formats::{Bits, DataFormat};
    use crate::generated::{
        AccessPattern, Attrs, DataType, DimProperty, LoopLabel, NameId, Operand, PROGRAMS, Program,
        Stmt, StmtKind, Strategy,
    };
    use crate::schedule::ddc::fold::{AllocId, ConstIdx, DataOrigin, NodeId, PadType};
    use crate::schedule::ddc::metadata::{
        Allocation, DataConnectSlot, DataTransfer, DdcMemory, ExternalStorage, MetaDimKind,
        Metadata, OpaqueOp, TransferAccessPattern, TransferEnd,
    };
    use crate::schedule::ddc::transformation::{DsType, Scale};
    use crate::schedule::ddc::transformation_util::{LoopCond, StageName};
    use crate::schedule::ddc::v1::CoreClSet;
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, AllocateNode, BlockNode, LayoutDims, LdsIdx, LoopDim,
        LoopNode, MaxDimSize, NodeName, NumBuffers, SchedNode, StartAddress, SyncDirection,
        SyncNode, SyncStrength, SyncUnits, WordLength,
    };
    use crate::schedule::l3::dsc::{
        CoreCount, CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DesignSpaceConfig, DimPadding,
        FilledDims, LabeledDs, LabeledDsList, NamedDims, PadElems, PadSizes, Pinning, StageDims,
        WkSlice, WkSliceId,
    };
    use crate::units::Core;

    use super::{
        DdlRoot, SyncEnds, SyncProp, TRANSFER_PROMOTION_DISABLED, Transformation, ddl_templates,
        parse_ddl2_dsc,
    };
    use crate::generated::SyncSignal;
    use crate::schedule::ddc::fold::{BlockId, NodeKind, ScheduleTree};

    /// A ONE-BLOCK TREE, WHICH IS ALL A [`BlockId`] NEEDS: the conversion addresses its own blocks by
    /// NAME, so `belowLxScheduleInsertBlock` reaches entry 364 as an id it only compares.
    struct OneBlock;

    impl ScheduleTree for OneBlock {
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

    /// ⭐⭐ THE WHOLE OF ENTRY 364 OVER ONE MODULE: `root_level_operations` is minted at the FRONT of
    /// the head, the dataflow region is walked under THAT block, the transformations region turns
    /// transfer promotion off, and each end of the signal gains the other's name.
    /// ⛔ AND BOTH NEGATIVES: senders with NO receivers is *"does not have matching syncs"*, and two
    /// senders sharing a unit is *"Multiple sync ops ... have same unit"* — the second is the ONLY way
    /// that check can fire, since [`SyncUnits`] makes a repeat WITHIN one node unconstructible.
    #[test]
    fn walks_both_regions_under_a_minted_root_block_and_pairs_the_sync_ends() {
        let program = synthetic(&[], &[]);
        let sender = NodeName("send".to_owned());
        let twin = NodeName("send_twin".to_owned());
        let receiver = NodeName("recv".to_owned());
        let sync = |name: &NodeName, direction| {
            SchedNode::Sync(SyncNode {
                name: name.clone(),
                units: SyncUnits::new(SenComponent::Lxsu, []),
                direction,
                strength: SyncStrength::Hard,
                implicit_sync_ref_transfer: None,
                other_ends: Vec::new(),
            })
        };
        let run = |senders: Vec<NodeName>, receivers: Vec<NodeName>| {
            let mut interface = DdlInterface::default();
            interface.sync_definitions.insert(
                SyncSignal::InputToLxsuToLxluToSync,
                SyncProp {
                    separate_corelets: false,
                    syncs_per_cl: BTreeMap::from([(None, SyncEnds { senders, receivers })]),
                },
            );
            let mut metadata = Metadata::default();
            metadata.below_lx_schedule_insert_block = BlockId::of(&OneBlock, NodeId(0));
            let mut dsc = config(Pinning::default(), None);
            let mut site = Match::default();
            let mut state = DdlConversion::new(BlockNode {
                name: NodeName("head".to_owned()),
                children: vec![
                    sync(&sender, SyncDirection::Send),
                    sync(&twin, SyncDirection::Send),
                    sync(&receiver, SyncDirection::Receive),
                ],
            });
            let said = parse_ddl2_dsc(
                &program,
                &mut state,
                &mut interface,
                &mut metadata,
                &mut dsc,
                &mut site,
                &DdlRoot {
                    regions: RegionTree::default(),
                    dataflows: vec![RegionId(0)],
                    transformations: vec![vec![Transformation::DisableTransferPromotion]],
                },
            );
            (said, state, interface, metadata)
        };
        let (said, state, interface, metadata) = run(vec![sender.clone()], vec![receiver.clone()]);
        assert_eq!(said, Some(vec![TRANSFER_PROMOTION_DISABLED.to_owned()]));
        assert!(!metadata.transformation_config.enable_moving_data_transfer);
        let root = NodeName("root_level_operations".to_owned());
        assert_eq!(
            state.tree.head().children.first().map(SchedNode::name),
            Some(&root)
        );
        assert_eq!(interface.region2blocks, BTreeMap::from([(RegionId(0), root)]));
        assert_eq!(
            state
                .tree
                .head()
                .children
                .iter()
                .filter_map(|child| match child {
                    SchedNode::Sync(node) => Some(node.other_ends.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![vec![receiver.clone()], Vec::new(), vec![sender.clone()]]
        );
        assert_eq!(run(vec![sender.clone()], Vec::new()).0, None);
        assert_eq!(
            run(vec![sender.clone(), twin.clone()], vec![receiver.clone()]).0,
            None
        );
    }

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
        /// `labeledDs_` = [lds0, lds1], a tree naming lds1 from a compute input and an allocate,
        /// and one metadata-owned allocate node on lds1.
        struct Site {
            last_lds: LdsIdx,
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
            slots: BTreeMap::from([
                (LdsSlot::ComputeInput(NodeId(3), 0), LdsIdx(1)),
                (LdsSlot::ComputeInput(NodeId(3), 1), LdsIdx(0)),
                (LdsSlot::Allocate(NodeId(4)), LdsIdx(1)),
            ]),
            allocations: BTreeMap::from([(AllocId(9), LdsIdx(1))]),
            inserted: Vec::new(),
            interim: Vec::new(),
        };
        // `tensor_definition_` = {%a: 1, %b: 1} — both name the old last index.
        let mut interface = DdlInterface::default();
        for name in [NameId(0), NameId(1)] {
            interface.tensor_definition.insert(
                name,
                TensorProp {
                    lds: Some(LdsIdx(1)),
                },
            );
        }
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

        let new = add_internal_tensor(
            &mut site,
            &mut interface,
            &mut metadata,
            LdsIdx(0),
            ComputeOpIdx(2),
        );

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
        // ⛔⛔ ON THE INTERFACE AND NOT ON THE SITE — this is the map `matchDdl2Dsc` prunes from, and
        // ⛔ THE FIRST MATCH ONLY: `%b` keeps the old index, exactly as the reference's `break` does.
        assert_eq!(
            interface.tensor_definition,
            BTreeMap::from([
                (
                    NameId(0),
                    TensorProp {
                        lds: Some(LdsIdx(2))
                    }
                ),
                (
                    NameId(1),
                    TensorProp {
                        lds: Some(LdsIdx(1))
                    }
                ),
            ])
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

    // ⭐ FIXTURES FOR ENTRIES 274-279.

    fn core(index: u32) -> Core {
        Core::checked(index).expect("a core in range")
    }

    /// A DSC on TWO cores with TWO corelets, labelling one INPUT tensor pinned as `pinning` says,
    /// whose core data stage states `X = 4` and `Y = 2` with whatever `padding` puts on `X`.
    fn config(pinning: Pinning, padding: Option<DimPadding>) -> DesignSpaceConfig {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::X, Extent(4));
        dims.extents.insert(PrimaryDim::Y, Extent(2));
        if let Some(padding) = padding {
            dims.padding.insert(PrimaryDim::X, padding);
        }
        let named = NamedDims {
            name: StageName::default(),
            dims: FilledDims::of(dims).expect("a stage that states a dim"),
        };
        let stage = DataStage {
            ss: named.clone(),
            el: named,
        };
        let two = CoreletsUsed::new(NonZeroU32::new(2).expect("two corelets"));
        DesignSpaceConfig {
            // ⛔ THE AUTHORITY'S OWN INITIALIZERS — see [`crate::schedule::l3::dsc::DdcFacts`].
            ddc: crate::schedule::l3::dsc::DdcFacts::default(),
            gtr_ids_used: BTreeSet::new(),
            corelets_used: two,
            corelets_used_dsc2: Some(two),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(core(0), vec![core(1)]),
            layout_dims: BTreeMap::new(),
            data_stages: DataStages::new(stage.clone(), stage),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(DsType::Input, vec![], LdsIdx(183), pinning),
                vec![],
            ),
        }
    }

    /// ⭐ EVERY VENDORED `ddl.dimension`, MINTED FROM ITS OWN `dim_property=` and memoised after — a
    /// second ask keeps whatever was written into the entry meanwhile. Plus the *"Illegal ddl file"*
    /// abort: a name whose definition is not a dimension at all.
    #[test]
    fn mints_a_dim_prop_from_every_vendored_dimension() {
        let mut seen = 0;
        for program in PROGRAMS {
            let mut interface = DdlInterface::default();
            for stmt in program.stmts {
                let Attrs::Dimension { property } = stmt.attrs else {
                    continue;
                };
                let want = match property {
                    None => MetaDimKind::Unpadded,
                    Some(DimProperty::Window) => MetaDimKind::WindowDim,
                    Some(DimProperty::Stride) => MetaDimKind::Stride,
                    Some(DimProperty::Dilation) => MetaDimKind::Dilation,
                    Some(DimProperty::PadFront) => MetaDimKind::PadFront,
                    Some(DimProperty::PadBack) => MetaDimKind::PadBack,
                    Some(DimProperty::PadValid) => MetaDimKind::PadValid,
                };
                for dim in stmt.results {
                    let minted = process_dimension_op(program, &mut interface, *dim)
                        .expect("a vendored dimension mints its own prop");
                    assert_eq!(minted.meta_dim_kind, want);
                    minted.drop_dim = true;
                    let again = process_dimension_op(program, &mut interface, *dim)
                        .expect("the memoised entry");
                    assert!(again.drop_dim, "{} {}", program.op_func, program.bind);
                    seen += 1;
                }
            }
        }
        assert!(seen > 0, "no vendored program declares a `ddl.dimension`");

        const NOT_A_DIM: &[Stmt] = &[Stmt {
            kind: StmtKind::Type,
            depth: 0,
            attrs: Attrs::Bare(StmtKind::Type),
            results: &[NameId(0)],
            operands: &[],
            path: &[],
        }];
        let program = synthetic(&["%t"], NOT_A_DIM);
        assert!(
            process_dimension_op(&program, &mut DdlInterface::default(), NameId(0)).is_none(),
            "a non-dimension is the abort"
        );
    }

    /// A `ddl.condition` on a live dim becomes ONE `AND` term against the loop its label names, the
    /// `ddl.condition_not` over it flips the COMPOSITE and not the term, and both memoise. ⛔ THE
    /// TRAP: the same condition on a DROPPED dim resolves to a constant instead, because a
    /// size-one loop's iterator is 0 on every comparison.
    #[test]
    fn resolves_a_loop_condition_and_negates_the_composite() {
        const CONDS: &[Stmt] = &[
            Stmt {
                kind: StmtKind::Condition,
                depth: 0,
                attrs: Attrs::Condition {
                    loop_label: Some(LoopLabel::ChunkLoop),
                    last: false,
                    negated: false,
                },
                results: &[NameId(1)],
                operands: &[Operand::One(NameId(0))],
                path: &[],
            },
            Stmt {
                kind: StmtKind::ConditionNot,
                depth: 0,
                attrs: Attrs::Bare(StmtKind::ConditionNot),
                results: &[NameId(2)],
                operands: &[Operand::One(NameId(1))],
                path: &[],
            },
        ];
        let program = synthetic(&["%x", "%first", "%not_first"], CONDS);
        let dsc = config(Pinning::default(), None);
        let metadata = Metadata::default();

        let live = |drop_dim| {
            let mut interface = DdlInterface::default();
            interface.dim_association.insert(
                NameId(0),
                DimProp {
                    dim: Some(PrimaryDim::X),
                    drop_dim,
                    ..DimProp::default()
                },
            );
            interface
                .loop_labels
                .insert(LoopLabel::ChunkLoop, NodeId(11));
            interface
        };

        let mut interface = live(false);
        let term = vec![vec![LoopCond {
            loop_node: NodeId(11),
            dim: PrimaryDim::X,
            op: CondOp::Eq,
            against: CondValType::First,
        }]];
        let mine = process_condition(&program, &mut interface, &metadata, &dsc, NameId(1))
            .expect("a labelled loop resolves");
        assert_eq!(mine.resolved, None);
        assert_eq!(mine.loop_cond.or_of_ands, term);
        assert!(!mine.loop_cond.negated);
        assert_eq!(interface.resolved_conditions[&NameId(1)], mine);

        let flipped = process_condition(&program, &mut interface, &metadata, &dsc, NameId(2))
            .expect("the negation of it");
        assert_eq!(flipped.loop_cond.or_of_ands, term);
        assert!(flipped.loop_cond.negated);

        let mut dropped = live(true);
        assert_eq!(
            process_condition(&program, &mut dropped, &metadata, &dsc, NameId(1))
                .expect("a dropped dim resolves")
                .resolved,
            Some(true)
        );
    }

    /// The core floor, and the dim compare in the shape that carries the trap: a `Padded` dim is
    /// measured over its FULL SPAN WITH UNNEEDED PAD, so the plain extent is not what it answers.
    #[test]
    fn verifies_the_core_floor_and_a_padded_dim_size() {
        let dsc = config(
            Pinning::default(),
            Some(DimPadding {
                sizes: PadSizes::Sized {
                    front: PadElems(1),
                    back: PadElems(2),
                },
                ..DimPadding::default()
            }),
        );
        let program = synthetic(&["%x_padded"], &[]);
        let mut interface = DdlInterface::default();
        let check = |interface: &mut DdlInterface, operands: &[Operand], constraint| {
            verify_ddl_constraint::<Dd2>(&program, interface, &dsc, operands, constraint)
        };

        assert_eq!(
            check(
                &mut interface,
                &[],
                DdlConstraint::MinNumCores(CoreCount(2))
            ),
            Some(())
        );
        assert_eq!(
            check(
                &mut interface,
                &[],
                DdlConstraint::MinNumCores(CoreCount(3))
            ),
            None
        );

        interface.dim_association.insert(
            NameId(0),
            DimProp {
                dim: Some(PrimaryDim::X),
                meta_dim_kind: MetaDimKind::Padded,
                ..DimProp::default()
            },
        );
        let operands = [Operand::One(NameId(0))];
        for (value, met) in [(Extent(7), Some(())), (Extent(4), None)] {
            assert_eq!(
                check(
                    &mut interface,
                    &operands,
                    DdlConstraint::DimSize {
                        cmp: ConstraintCmp::Equal,
                        value
                    }
                ),
                met
            );
        }
        assert_eq!(
            check(
                &mut interface,
                &operands,
                DdlConstraint::DimSize {
                    cmp: ConstraintCmp::Less,
                    value: Extent(8)
                }
            ),
            Some(())
        );
    }

    /// The DSC's own allocate node matched back to the `ddl.allocate` that placed it, plus the two
    /// EMPTY pairs — a `NO_COMPONENT` unit and a node no allocate op claims — and the `.at()` throw.
    #[test]
    fn matches_an_allocate_node_back_to_its_ddl_tensor() {
        struct Site;
        impl AllocationSite for Site {
            fn lds_allocation(&self, lds: LdsIdx, unit: SenComponent) -> Option<AllocId> {
                (lds == LdsIdx(2) && unit == SenComponent::L0).then_some(AllocId(9))
            }
            fn constant_allocation(
                &self,
                _constant: ConstIdx,
                _unit: SenComponent,
            ) -> Option<AllocId> {
                None
            }
        }
        let allocations = [EmittedAllocate {
            op: AllocOp(1),
            tensor: EmittedTensor::Named(NameId(0)),
            node: AllocId(9),
        }];
        let tensors = BTreeMap::new();
        let constants = BTreeMap::new();
        let operand_constants = BTreeMap::from([(NameId(0), SenComponent::L0)]);
        let pair = |unit, origin| {
            tensor_and_allocation(
                &Site,
                &allocations,
                &tensors,
                &constants,
                &operand_constants,
                unit,
                origin,
            )
        };

        assert_eq!(
            pair(SenComponent::L0, Some(DataOrigin::LabeledDs(LdsIdx(2)))),
            Some(TensorAndAllocation {
                tensor: Some(EmittedTensor::Named(NameId(0))),
                allocate: Some(AllocOp(1))
            })
        );
        assert_eq!(
            pair(
                SenComponent::NoComponent,
                Some(DataOrigin::LabeledDs(LdsIdx(2)))
            ),
            Some(TensorAndAllocation::default())
        );
        // No origin is the operand-constant tensor, which no `ddl.allocate` places.
        assert_eq!(
            pair(SenComponent::L0, None),
            Some(TensorAndAllocation {
                tensor: Some(EmittedTensor::Named(NameId(0))),
                allocate: None
            })
        );
        assert_eq!(
            pair(SenComponent::L0, Some(DataOrigin::LabeledDs(LdsIdx(7)))),
            None
        );
    }

    /// ⭐⭐ THE SCHEDULE TREE REACHES THE EMISSION: a `ddl.loop` whose trip count the two datastages
    /// state, holding the `ddl.allocate` its post-DDC LDS resolves back to a censused tensor and an
    /// L3 `ddl.sync` no template declares — which is therefore labelled with its OWN name — beside
    /// the two mappings written onto the parsed ops.
    #[test]
    fn emits_the_whole_schedule_tree_as_a_second_dataflow() {
        struct Site;
        impl AllocationSite for Site {
            fn lds_allocation(&self, _lds: LdsIdx, _unit: SenComponent) -> Option<AllocId> {
                None
            }
            fn constant_allocation(
                &self,
                _constant: ConstIdx,
                _unit: SenComponent,
            ) -> Option<AllocId> {
                None
            }
        }
        impl DdlSizes for Site {
            fn block_transfer_size(&self, _transfer: NodeId) -> Option<Elements> {
                None
            }
            fn buffer_capacity(&self, _alloc: AllocId) -> Option<Elements> {
                Some(Elements(96))
            }
        }
        const OPS: &[Stmt] = &[
            Stmt {
                kind: StmtKind::Dimension,
                depth: 0,
                attrs: Attrs::Bare(StmtKind::Dimension),
                results: &[NameId(0)],
                operands: &[],
                path: &[],
            },
            Stmt {
                kind: StmtKind::Tensor,
                depth: 0,
                attrs: Attrs::Bare(StmtKind::Tensor),
                results: &[NameId(1)],
                operands: &[],
                path: &[],
            },
        ];
        let program = synthetic(&["%x", "%tensor"], OPS);
        let mut interface = DdlInterface::default();
        interface.dim_association.insert(
            NameId(0),
            DimProp {
                dim: Some(PrimaryDim::X),
                ..DimProp::default()
            },
        );
        interface.tensor_definition.insert(
            NameId(1),
            TensorProp {
                lds: Some(LdsIdx(5)),
            },
        );
        let mut metadata = Metadata::default();
        metadata.lds_idx_after_ddc.insert(LdsIdx(5), LdsIdx(2));

        let alloc = NodeName("alloc_l0".to_owned());
        let signal = NodeName("l3_signal".to_owned());
        let held = NodeName("loop_ds0_ds1".to_owned());
        let mut state = DdlConversion::new(BlockNode {
            name: NodeName("head".to_owned()),
            children: vec![SchedNode::Loop(Box::new(LoopNode {
                block: BlockNode {
                    name: held.clone(),
                    children: vec![
                        SchedNode::Leaf(alloc.clone()),
                        SchedNode::Sync(SyncNode {
                            name: signal.clone(),
                            units: SyncUnits::new(SenComponent::L0, []),
                            direction: SyncDirection::Receive,
                            strength: SyncStrength::Hard,
                            implicit_sync_ref_transfer: None,
                            other_ends: Vec::new(),
                        }),
                    ],
                },
                dims: vec![LoopDim {
                    dim: PrimaryDim::X,
                    kind: MetaDimKind::Unpadded,
                }],
                num: Some(Metadata::CORE_DSTGID),
                den: Some(Metadata::CHUNK_DSTGID),
                parametric_lds: None,
            }))],
        });
        state.allocations.insert(
            AllocId(4),
            AllocateNode {
                name: alloc.clone(),
                component: SenComponent::L0,
                lds: Some(LdsIdx(2)),
                const_idx: None,
                temp_storage_for_compute: None,
                layout: AllocLayout::new(
                    (PrimaryDim::X, MaxDimSize::Resolved(Elements(64))),
                    Vec::new(),
                ),
                start_address: StartAddress::default(),
                placement: AllocPlacement::default(),
                gap_stick_spread: BTreeMap::new(),
                alloc_users: Vec::new(),
            },
        );
        let dsc = config(Pinning::default(), None);

        let emitted = convert_dsc2_ddl(&program, &state, &interface, &metadata, &dsc, &Site)
            .expect("a tree of a loop, an allocate and a sync emits");
        assert_eq!(
            emitted.dim_mapping,
            BTreeMap::from([(NameId(0), DimMapping::Dim(PrimaryDim::X))])
        );
        assert_eq!(
            emitted.lds_mapping,
            BTreeMap::from([(NameId(1), Some(LdsIdx(5)))])
        );
        assert_eq!(
            emitted.dataflow,
            vec![EmittedOp::Loop {
                name: held,
                label: None,
                // ⛔ BOTH STAGES READ AS MINTED: this DSC's stages are unnamed, and `datastages_`
                // states nothing, so `operator[]`'s default `strategyMinimize_` is the answer.
                num: EmittedStage::Minted(Strategy::Minimize),
                den: EmittedStage::Minted(Strategy::Minimize),
                dims: vec![NameId(0)],
                ss_loop_count: vec![(PrimaryDim::X, LoopCount(1))],
                body: vec![
                    EmittedOp::Allocate {
                        name: alloc,
                        op: AllocOp(0),
                        tensor: EmittedTensor::Named(NameId(1)),
                        padding_styles: Vec::new(),
                        component: SenComponent::L0,
                        num_buffers: NumBuffers::Single,
                        layout_dim_order: vec![PrimaryDim::X],
                        allocation_size: Some(Elements(96)),
                    },
                    EmittedOp::Sync {
                        name: signal.clone(),
                        units: vec![SenComponent::L0],
                        direction: SyncDirection::Receive,
                        label: SyncLabel::External(signal),
                        separate_corelets: false,
                    },
                ],
            }]
        );
    }

    /// A windowed DSC dim demands FIVE meta kinds of its `ddl.padded_dimension` — and ⛔ NOT THE
    /// DILATION, which the reference's second check skips, so a DSC dilation the DDL omits passes.
    #[test]
    fn a_padded_dimension_states_every_dsc_meta_dim_but_dilation() {
        const FULL: &[Operand] = &[
            Operand::One(NameId(2)),
            Operand::One(NameId(3)),
            Operand::One(NameId(4)),
            Operand::One(NameId(5)),
            Operand::One(NameId(6)),
        ];
        let program = |operands: &'static [Operand]| {
            let stmts: &'static [Stmt] = Box::leak(Box::new([Stmt {
                kind: StmtKind::PaddedDimension,
                depth: 0,
                attrs: Attrs::Bare(StmtKind::PaddedDimension),
                results: &[NameId(1)],
                operands,
                path: &[],
            }]));
            synthetic(
                &[
                    "%x",
                    "%x_padded",
                    "%front",
                    "%back",
                    "%valid",
                    "%window",
                    "%stride",
                ],
                stmts,
            )
        };
        let interface = || {
            let mut interface = DdlInterface::default();
            for (name, kind) in [
                (NameId(0), MetaDimKind::Unpadded),
                (NameId(2), MetaDimKind::PadFront),
                (NameId(3), MetaDimKind::PadBack),
                (NameId(4), MetaDimKind::PadValid),
                (NameId(5), MetaDimKind::WindowDim),
                (NameId(6), MetaDimKind::Stride),
            ] {
                interface.dim_association.insert(
                    name,
                    DimProp {
                        dim: (name == NameId(0)).then_some(PrimaryDim::X),
                        meta_dim_kind: kind,
                        ..DimProp::default()
                    },
                );
            }
            interface.dim_association.insert(
                NameId(1),
                DimProp {
                    meta_dim_kind: MetaDimKind::Padded,
                    non_padded_dim: Some(NameId(0)),
                    ..DimProp::default()
                },
            );
            interface
        };
        // A windowed dim, so the DSC states all SEVEN kinds including the dilation.
        let dsc = config(
            Pinning::default(),
            Some(DimPadding {
                window_dim: Some(PrimaryDim::Y),
                ..DimPadding::default()
            }),
        );

        assert_eq!(
            check_meta_dimensions(&program(FULL), &mut interface(), &dsc),
            Some(())
        );
        // The same op with the STRIDE dropped is the *"Internal error in PaddedDimensionOp
        // verification."* refusal — the DSC states one and the op does not.
        assert_eq!(
            check_meta_dimensions(&program(&FULL[..4]), &mut interface(), &dsc),
            None
        );
    }

    /// One style per dim, keyed by the primary dim each DDL dim is ASSOCIATED with — and the
    /// unassigned dim that keys nothing, whose entry the ask has still minted.
    #[test]
    fn keys_each_access_pattern_by_its_primary_dim() {
        const STYLES: &[AccessPattern] = &[AccessPattern::PaddedWzeropadToToToLoweredPadded];
        let styled =
            StyledDims::stated(&[NameId(0), NameId(1)], STYLES).expect("one style broadcasts");
        let mut dim_association = BTreeMap::from([
            (
                NameId(0),
                DimProp {
                    dim: Some(PrimaryDim::X),
                    ..DimProp::default()
                },
            ),
            (
                NameId(1),
                DimProp {
                    dim: Some(PrimaryDim::Y),
                    ..DimProp::default()
                },
            ),
        ]);
        let mut result = BTreeMap::new();
        assert_eq!(
            process_access_patterns(
                &mut dim_association,
                &styled,
                transfer_access_pattern,
                &mut result
            ),
            Some(())
        );
        assert_eq!(
            result,
            BTreeMap::from([
                (PrimaryDim::X, transfer_access_pattern(STYLES[0])),
                (PrimaryDim::Y, transfer_access_pattern(STYLES[0])),
            ])
        );

        let mut unassigned = BTreeMap::new();
        assert_eq!(
            process_access_patterns(
                &mut unassigned,
                &styled,
                transfer_access_pattern,
                &mut BTreeMap::new()
            ),
            None
        );
        assert!(unassigned.contains_key(&NameId(0)));
    }

    // ⭐ FIXTURES FOR ENTRIES 345-347.

    /// A DSC seam that answers the four questions the match asks and nothing else: every tensor is
    /// two bytes of fp16, one stick of `X = 4` over `Y = 2`, and no allocation is placed.
    #[derive(Default)]
    struct Match {
        ops: Vec<BindableOp>,
        /// `numWkSlicesPerDim_`.
        slices: BTreeMap<PrimaryDim, u32>,
        /// `coreIdToWkSlice_`.
        work: BTreeMap<Core, WkSlice>,
    }

    impl AllocationSite for Match {
        fn lds_allocation(&self, _lds: LdsIdx, _unit: SenComponent) -> Option<AllocId> {
            None
        }
        fn constant_allocation(&self, _constant: ConstIdx, _unit: SenComponent) -> Option<AllocId> {
            None
        }
    }

    impl InternalTensorSite for Match {
        fn labeled_ds_tail(&self) -> Option<LabeledDsTail> {
            None
        }
        fn set_last_lds_idx(&mut self, _lds: LdsIdx) {}
        fn insert_internal_tensor(&mut self, _new: InternalTensor) {}
        fn add_interim_lds(&mut self, _compute_op: ComputeOpIdx, _lds: LdsIdx) {}
        fn lds_slots(&self) -> Vec<LdsSlot> {
            Vec::new()
        }
        fn slot_lds(&self, _slot: LdsSlot) -> Option<LdsIdx> {
            None
        }
        fn set_slot_lds(&mut self, _slot: LdsSlot, _lds: LdsIdx) {}
        fn allocation_lds(&self, _alloc: AllocId) -> Option<LdsIdx> {
            None
        }
        fn set_allocation_lds(&mut self, _alloc: AllocId, _lds: LdsIdx) {}
    }

    impl DdlSite for Match {
        fn fused_format(&self) -> Option<DataFormat> {
            Some(DataFormat::Sen169Fp16)
        }
        fn lds_format(&self, _lds: LdsIdx) -> Option<DataFormat> {
            Some(DataFormat::Sen169Fp16)
        }
        fn unit_reaches(&self, _unit: SenComponent, _storage: SenComponent) -> bool {
            false
        }
        fn dim_in_layout_order(&self, _lds: LdsIdx, _dim: PrimaryDim) -> bool {
            false
        }
        fn wk_slices(&self, dim: PrimaryDim) -> Option<u32> {
            self.slices.get(&dim).copied()
        }
        fn core_work_slices(&self) -> BTreeMap<Core, WkSlice> {
            self.work.clone()
        }
        fn stick_dims(&self, _lds: LdsIdx) -> Option<StickDims> {
            Some(StickDims(vec![
                (PrimaryDim::X, Elements(4)),
                (PrimaryDim::Y, Elements(2)),
            ]))
        }
    }

    impl MatchSite for Match {
        fn bindable_ops(&self) -> Vec<BindableOp> {
            self.ops.clone()
        }
        fn lds_word_length(&self, _lds: LdsIdx) -> Option<WordLength> {
            Some(WordLength(2))
        }
        fn set_lds_word_length(&mut self, _lds: LdsIdx, _length: WordLength) {}
        fn set_lds_format(&mut self, _lds: LdsIdx, _format: DataFormat) {}
    }

    /// A DSC whose one INPUT tensor is labelled at index ZERO and lays out `X` then `Y`, which is
    /// what the match needs [`config`]'s recorded index 183 not to be.
    fn matchable() -> DesignSpaceConfig {
        let mut dsc = config(Pinning::default(), None);
        dsc.labeled_ds = LabeledDsList::new(
            LabeledDs::new(
                DsType::Input,
                vec![
                    (PrimaryDim::X, Scale::Sized(1.0)),
                    (PrimaryDim::Y, Scale::Sized(1.0)),
                ],
                LdsIdx(0),
                Pinning::default(),
            ),
            vec![],
        );
        dsc.layout_dims.insert(
            LdsIdx(0),
            LayoutDims::new(PrimaryDim::X, vec![PrimaryDim::Y]),
        );
        dsc
    }

    /// ⭐ THE EXPORT IS `convertDsc2Ddl` AND NOTHING ELSE: the two answer the same emission over the
    /// same state, and the `std::ostream&` the reference forwards is never written.
    #[test]
    fn exports_exactly_what_the_conversion_emits() {
        struct Site;
        impl AllocationSite for Site {
            fn lds_allocation(&self, _lds: LdsIdx, _unit: SenComponent) -> Option<AllocId> {
                None
            }
            fn constant_allocation(
                &self,
                _constant: ConstIdx,
                _unit: SenComponent,
            ) -> Option<AllocId> {
                None
            }
        }
        impl DdlSizes for Site {
            fn block_transfer_size(&self, _transfer: NodeId) -> Option<Elements> {
                None
            }
            fn buffer_capacity(&self, _alloc: AllocId) -> Option<Elements> {
                None
            }
        }
        let program = synthetic(&[], &[]);
        let interface = DdlInterface::default();
        let metadata = Metadata::default();
        let dsc = config(Pinning::default(), None);
        let state = DdlConversion::new(BlockNode {
            name: NodeName("head".to_owned()),
            children: Vec::new(),
        });
        let exported = export_to_ddl(&program, &state, &interface, &metadata, &dsc, &Site);
        assert_eq!(
            exported,
            convert_dsc2_ddl(&program, &state, &interface, &metadata, &dsc, &Site)
        );
        assert_eq!(
            exported,
            Some(EmittedDdl {
                dim_mapping: BTreeMap::new(),
                lds_mapping: BTreeMap::new(),
                dataflow: Vec::new(),
            })
        );
    }

    /// ⭐⭐ A MULTI-REGION OP OPENS A BLOCK PER REGION: an unresolved `ddl.if` mints its condition
    /// node and then `condition_region0` and `condition_region1` under it, which is where the THEN
    /// and ELSE arms attach — and every region records the block that was current when it opened.
    #[test]
    fn opens_a_named_block_per_region_of_a_multi_region_op() {
        let program = synthetic(&[], &[]);
        let mut interface = DdlInterface::default();
        interface
            .resolved_conditions
            .insert(NameId(0), CondProp::default());
        let mut metadata = Metadata::default();
        let mut dsc = config(Pinning::default(), None);
        let mut site = Match::default();
        let head = NodeName("head".to_owned());
        let mut state = DdlConversion::new(BlockNode {
            name: head.clone(),
            children: Vec::new(),
        });
        let regions = RegionTree {
            ops: BTreeMap::from([(
                RegionId(0),
                vec![RegionOp {
                    op: DdlOp::If {
                        condition: NameId(0),
                        then_region: RegionId(1),
                        else_region: RegionId(2),
                    },
                    regions: vec![RegionId(1), RegionId(2)],
                }],
            )]),
        };
        assert_eq!(
            process_region(
                &program,
                &mut state,
                &mut interface,
                &mut metadata,
                &mut dsc,
                &mut site,
                &regions,
                RegionId(0),
                &head,
            ),
            Some(())
        );
        assert_eq!(
            interface.region2blocks,
            BTreeMap::from([
                (RegionId(0), head),
                (RegionId(1), NodeName("condition_region0".to_owned())),
                (RegionId(2), NodeName("condition_region1".to_owned())),
            ])
        );
        let [SchedNode::Guarded(cond)] = state.tree.head().children.as_slice() else {
            panic!("the condition node the unresolved `ddl.if` minted");
        };
        assert_eq!(
            cond.then_region
                .iter()
                .chain(cond.else_region.iter())
                .map(|child| child.name().clone())
                .collect::<Vec<_>>(),
            vec![
                NodeName("condition_region0".to_owned()),
                NodeName("condition_region1".to_owned()),
            ]
        );
    }

    /// ⭐⭐ THE RING IS WALKED PER USED CORE AND KEYED BY THE WHOLE WORK SLICE — four cores split
    /// 2×2 over `X` and `Y` with the reduction on `Y` give every core its neighbour in its OWN `X`
    /// column, and `c1`/`c3` are the entries a per-slice lookup drops. Corelet 1 walks the other
    /// way, so it is the START where corelet 0 is the END.
    /// ⛔ AND ONE SLICE RESOLVES BOTH CONDITIONS STATICALLY TRUE, writing neither a set nor a ring.
    #[test]
    fn resolves_the_sfpring_over_every_used_core() {
        /// `%psum_start, %psum_end, %next_core, %prev_core = ddl.core_to_core_communication(%in)`
        /// (`ddl_templates/bmm.ddl:172`).
        const C2C: &Stmt = &Stmt {
            kind: StmtKind::CoreToCoreCommunication,
            depth: 0,
            attrs: Attrs::Bare(StmtKind::CoreToCoreCommunication),
            results: &[NameId(1), NameId(2), NameId(3), NameId(4)],
            operands: &[Operand::One(NameId(0))],
            path: &[],
        };
        let program = synthetic(&[], &[]);
        let cl = |index| corelet(index).expect("a corelet in range");
        let slice = |x, y| {
            WkSlice(BTreeMap::from([
                (PrimaryDim::X, WkSliceId(x)),
                (PrimaryDim::Y, WkSliceId(y)),
            ]))
        };
        let work = BTreeMap::from([
            (core(0), slice(0, 0)),
            (core(1), slice(1, 0)),
            (core(2), slice(0, 1)),
            (core(3), slice(1, 1)),
        ]);
        let run = |slices: u32| {
            let mut interface = DdlInterface::default();
            interface.dim_association.insert(
                NameId(0),
                DimProp {
                    dim: Some(PrimaryDim::Y),
                    ..DimProp::default()
                },
            );
            let mut dsc = config(Pinning::default(), None);
            dsc.core_ids_used = CoreIdsUsed::new(core(0), vec![core(1), core(2), core(3)]);
            let mut site = Match {
                slices: BTreeMap::from([(PrimaryDim::Y, slices)]),
                work: work.clone(),
                ..Match::default()
            };
            let head = NodeName("head".to_owned());
            let mut state = DdlConversion::new(BlockNode {
                name: head.clone(),
                children: Vec::new(),
            });
            assert_eq!(
                op_core_to_core(
                    &mut OpContext {
                        program: &program,
                        state: &mut state,
                        interface: &mut interface,
                        metadata: &mut Metadata::default(),
                        dsc: &mut dsc,
                        site: &mut site,
                        curr_parent: head.clone(),
                    },
                    C2C,
                ),
                Some(OpOutcome {
                    parent: Some(head),
                    regions: Vec::new(),
                })
            );
            (
                interface.core_to_core_definitions[&NameId(1)].clone(),
                interface.resolved_conditions[&NameId(1)].clone(),
                interface.resolved_conditions[&NameId(2)].clone(),
            )
        };
        let (ring, start, end) = run(2);
        assert_eq!(ring.dim, Some(PrimaryDim::Y));
        assert_eq!(
            ring.next_core,
            BTreeMap::from([
                ((core(0), cl(0)), core(2)),
                ((core(1), cl(0)), core(3)),
                ((core(2), cl(1)), core(0)),
                ((core(3), cl(1)), core(1)),
            ])
        );
        assert_eq!(
            ring.prev_core,
            BTreeMap::from([
                ((core(0), cl(1)), core(2)),
                ((core(1), cl(1)), core(3)),
                ((core(2), cl(0)), core(0)),
                ((core(3), cl(0)), core(1)),
            ])
        );
        assert_eq!((start.resolved, end.resolved), (None, None));
        assert_eq!(
            start.core_cl_cond,
            CoreClSet(BTreeMap::from([
                (core(0), BTreeSet::from([cl(0)])),
                (core(1), BTreeSet::from([cl(0)])),
                (core(2), BTreeSet::from([cl(1)])),
                (core(3), BTreeSet::from([cl(1)])),
            ]))
        );
        assert_eq!(
            end.core_cl_cond,
            CoreClSet(BTreeMap::from([
                (core(0), BTreeSet::from([cl(1)])),
                (core(1), BTreeSet::from([cl(1)])),
                (core(2), BTreeSet::from([cl(0)])),
                (core(3), BTreeSet::from([cl(0)])),
            ]))
        );
        let (ring, start, end) = run(1);
        assert_eq!((start.resolved, end.resolved), (Some(true), Some(true)));
        assert_eq!(
            (
                ring.next_core.len(),
                ring.prev_core.len(),
                start.core_cl_cond.0.len(),
                end.core_cl_cond.0.len()
            ),
            (0, 0, 0, 0)
        );
    }

    /// ⭐⭐ THE WHOLE MATCH, AND THEN THE REFUSAL: one `ddl.operation_bind` over an fp16 `add` binds
    /// the DSC's only compute op, ties its tensor to LDS 0, and — because both fixed layouts state
    /// `%d0` then `%d1` while the DSC lays out `X` then `Y` — maps `%d0` to `X` and `%d1` to `Y`.
    /// A required bind whose arity no compute op has is *"Template not suitable for DSC"*.
    #[test]
    fn binds_the_template_and_maps_every_dsc_dim() {
        const OPS: &[Stmt] = &[
            Stmt {
                kind: StmtKind::Dimension,
                depth: 0,
                attrs: Attrs::Dimension { property: None },
                results: &[NameId(0)],
                operands: &[],
                path: &[],
            },
            Stmt {
                kind: StmtKind::Dimension,
                depth: 0,
                attrs: Attrs::Dimension { property: None },
                results: &[NameId(1)],
                operands: &[],
                path: &[],
            },
            Stmt {
                kind: StmtKind::Layout,
                depth: 0,
                attrs: Attrs::Layout { order_fixed: true },
                results: &[NameId(2)],
                operands: &[Operand::One(NameId(0)), Operand::One(NameId(1))],
                path: &[],
            },
            Stmt {
                kind: StmtKind::Layout,
                depth: 0,
                attrs: Attrs::Layout { order_fixed: true },
                results: &[NameId(3)],
                operands: &[],
                path: &[],
            },
            Stmt {
                kind: StmtKind::Layout,
                depth: 0,
                attrs: Attrs::Layout { order_fixed: true },
                results: &[NameId(4)],
                operands: &[Operand::One(NameId(0)), Operand::One(NameId(1))],
                path: &[],
            },
            Stmt {
                kind: StmtKind::Type,
                depth: 0,
                attrs: Attrs::Type {
                    data_type: DataType::Sen169Fp16,
                    bit_width: None,
                },
                results: &[NameId(5)],
                operands: &[],
                path: &[],
            },
            Stmt {
                kind: StmtKind::Tensor,
                depth: 0,
                attrs: Attrs::Bare(StmtKind::Tensor),
                results: &[NameId(6)],
                operands: &[
                    Operand::One(NameId(2)),
                    Operand::One(NameId(3)),
                    Operand::One(NameId(4)),
                    Operand::List(&[NameId(5)]),
                ],
                path: &[],
            },
        ];
        let program = synthetic(
            &[
                "%d0", "%d1", "%slice", "%stick", "%global", "%type", "%tensor", "%add_op",
            ],
            OPS,
        );
        let compute = BindableOp {
            op_func: OpFunc::Add,
            format: Some(DataFormat::Sen169Fp16),
            inputs: vec![LdsIdx(0)],
            outputs: vec![LdsIdx(0)],
            core_exclude: BTreeSet::new(),
            core_cl_exclude: BTreeSet::new(),
        };
        let bind = OperationBind {
            result: NameId(7),
            op_func: Some(OpFunc::Add),
            required: true,
            data_formats: Vec::new(),
            inputs: vec![NameId(6)],
            outputs: vec![NameId(6)],
            interim: Vec::new(),
        };
        let padded: BTreeMap<NameId, PaddedDimension<'_>> = BTreeMap::new();
        let mut interface = DdlInterface::default();
        let mut metadata = Metadata::default();
        let mut dsc = matchable();
        let mut site = Match {
            ops: vec![compute.clone()],
            ..Match::default()
        };
        assert_eq!(
            match_ddl2_dsc(
                &program,
                &mut interface,
                &mut metadata,
                &mut dsc,
                &mut site,
                &[bind],
                &padded,
            ),
            Some(true)
        );
        assert_eq!(
            interface.operation_definition[&NameId(7)].compute_op,
            Some(ComputeOpIdx(0))
        );
        assert_eq!(
            interface.tensor_definition[&NameId(6)],
            TensorProp {
                lds: Some(LdsIdx(0))
            }
        );
        assert_eq!(
            [
                interface.dim_association[&NameId(0)].dim,
                interface.dim_association[&NameId(1)].dim,
            ],
            [Some(PrimaryDim::X), Some(PrimaryDim::Y)]
        );

        let two_inputs = OperationBind {
            result: NameId(7),
            op_func: Some(OpFunc::Add),
            required: true,
            data_formats: Vec::new(),
            inputs: vec![NameId(6), NameId(6)],
            outputs: vec![NameId(6)],
            interim: Vec::new(),
        };
        assert_eq!(
            match_ddl2_dsc(
                &program,
                &mut DdlInterface::default(),
                &mut Metadata::default(),
                &mut matchable(),
                &mut Match {
                    ops: vec![compute],
                    ..Match::default()
                },
                &[two_inputs],
                &padded,
            ),
            Some(false)
        );
    }

    use super::{DdlSelection, DdlTemplateSet, StatedTemplate, select_and_parse_ddl_template};
    use crate::generated::Template;
    use crate::schedule::ddc::v1::Ln32;
    use crate::schedule::ddl::ops::Dialect;
    use crate::schedule::ddl::{DdlModuleOp, DdlSource, ParsedDdl};

    /// A BUFFER THAT PARSES — a module stating no op at all, which every verifier accepts.
    struct Buffer;

    impl DdlSource for Buffer {
        fn parse(&self, _dialect: &Dialect) -> Option<ParsedDdl> {
            Some(ParsedDdl {
                defining: Vec::new(),
                verifiable: Vec::new(),
            })
        }
    }

    /// EVERY CANDIDATE OF ONE OP-FUNC, all stating the same empty program — and a
    /// `ddl.operation_bind` for `matches` ALONE, which is what makes exactly that one match: a
    /// template with no bind leaves the DSC's compute op unmapped, and that is entry 347's `false`.
    struct Candidates {
        program: Program,
        matches: Template,
        constraint: DdlConstraint,
    }

    impl DdlTemplateSet for Candidates {
        type Source = Buffer;

        fn stated(&self, template: Template) -> Option<StatedTemplate<'_, Buffer>> {
            Some(StatedTemplate {
                source: Buffer,
                program: &self.program,
                binds: if template == self.matches {
                    vec![OperationBind {
                        result: NameId(0),
                        op_func: Some(OpFunc::Exx2),
                        required: true,
                        data_formats: Vec::new(),
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        interim: Vec::new(),
                    }]
                } else {
                    Vec::new()
                },
                padded: BTreeMap::new(),
                constraints: vec![(&[], self.constraint)],
                root: DdlRoot {
                    regions: RegionTree::default(),
                    dataflows: vec![RegionId(0)],
                    transformations: vec![vec![Transformation::DisableTransferPromotion]],
                },
            })
        }
    }

    /// ⭐⭐ THE WHOLE WALK OVER `exx2`'S THREE CANDIDATES: the first does not match and is RELEASED,
    /// the second does — the stale interface is cleared, its constraint checked, and its dataflow
    /// reaches the tree as `root_level_operations` — and the template it settled on is the answer.
    /// ⛔ AND EVERY REFUSAL: an empty `computeOp_` states nothing, an op-func the table does not name
    /// says so, an exhausted walk says so, a constraint the DSC fails aborts, and so does `ENABLE_LN32`
    /// on an `exx2`.
    #[test]
    fn walks_the_candidates_until_one_matches_and_parses_it_onto_the_dsc() {
        let run =
            |op_func: Option<OpFunc>, matches: Template, constraint: DdlConstraint, ln32: Ln32| {
                let templates = Candidates {
                    program: synthetic(&[], &[]),
                    matches,
                    constraint,
                };
                let mut parser = DdlModuleOp::default();
                let mut interface = DdlInterface::default();
                interface
                    .region2blocks
                    .insert(RegionId(1), NodeName("stale".to_owned()));
                let mut metadata = Metadata::default();
                metadata.below_lx_schedule_insert_block = BlockId::of(&OneBlock, NodeId(0));
                let mut dsc = matchable();
                let mut site = Match {
                    ops: op_func
                        .map(|op_func| BindableOp {
                            op_func,
                            format: Some(DataFormat::Sen169Fp16),
                            inputs: Vec::new(),
                            outputs: Vec::new(),
                            core_exclude: BTreeSet::new(),
                            core_cl_exclude: BTreeSet::new(),
                        })
                        .into_iter()
                        .collect(),
                    ..Match::default()
                };
                let mut state = DdlConversion::new(BlockNode {
                    name: NodeName("head".to_owned()),
                    children: Vec::new(),
                });
                let selected = select_and_parse_ddl_template::<Dd2, _, _>(
                    &mut parser,
                    &mut state,
                    &mut interface,
                    &mut metadata,
                    &mut dsc,
                    &mut site,
                    &templates,
                    ln32,
                );
                (selected, parser, state, interface)
            };
        let cores = DdlConstraint::MinNumCores(CoreCount(2));
        let second = Template::Summeanmaxexx2Fp32;
        let (selected, parser, state, interface) =
            run(Some(OpFunc::Exx2), second, cores, Ln32::Off);
        assert_eq!(
            selected,
            Some(DdlSelection {
                template: Some(second),
                said: vec![TRANSFER_PROMOTION_DISABLED.to_owned()],
            })
        );
        assert_eq!(
            interface.operation_definition[&NameId(0)].compute_op,
            Some(ComputeOpIdx(0))
        );
        let root = NodeName("root_level_operations".to_owned());
        assert_eq!(
            interface.region2blocks,
            BTreeMap::from([(RegionId(0), root)])
        );
        assert_eq!(
            state.tree.head().children.len(),
            1,
            "the matched template's dataflow reached the tree"
        );
        assert!(parser.module().is_some());

        let (exhausted, parser, ..) = run(Some(OpFunc::Exx2), Template::Argmax, cores, Ln32::Off);
        assert_eq!(
            exhausted,
            Some(DdlSelection {
                template: None,
                said: vec!["[DDC] DDL found but not suitable for op exx2".to_owned()],
            })
        );
        assert!(parser.module().is_none());
        assert_eq!(
            run(None, second, cores, Ln32::Off).0,
            Some(DdlSelection::default())
        );
        assert_eq!(
            run(Some(OpFunc::Softmax), second, cores, Ln32::Off).0,
            Some(DdlSelection {
                template: None,
                said: vec!["[DDC] no DDL available for op softmax".to_owned()],
            })
        );
        assert_eq!(
            run(
                Some(OpFunc::Exx2),
                second,
                DdlConstraint::MinNumCores(CoreCount(3)),
                Ln32::Off
            )
            .0,
            None
        );
        assert_eq!(run(Some(OpFunc::Exx2), second, cores, Ln32::On).0, None);
    }

    /// ⛔ REVIEWED: THE CANDIDATE LOOKUP MUST NAME EVERY ROW OF `opFuncToDdlTemplate`, because
    /// [`None`] is entry 372's *"no DDL available for op"* and it is the answer for an op-func the
    /// table does NOT name — never for one it does. Three rows past the `:267` the first read of
    /// `ddl_conversion.h` stopped at were answering [`None`], and an EMPTY list — every candidate
    /// skipped on arch — is the third, distinct answer.
    #[test]
    fn every_row_of_the_candidate_table_is_named() {
        // `:169-170`, untagged, so both generations serve them.
        for op_func in [OpFunc::AddI32ToI32, OpFunc::AddI64ToI64] {
            for isa in [IsaGen::Rcudd1a, IsaGen::Sen1p5] {
                assert_eq!(
                    ddl_templates(op_func.spelling(), isa),
                    Some([Template::BroadcastOps].as_slice()),
                    "{} on {isa:?}",
                    op_func.spelling()
                );
            }
        }
        // `:107`, SEN1P5 only — so RCUDD1A gets the EMPTY list and not the miss.
        assert_eq!(
            ddl_templates(OpFunc::BatchmatmulMxfp4WFwd.spelling(), IsaGen::Sen1p5),
            Some([Template::BmmSen1p5].as_slice())
        );
        assert_eq!(
            ddl_templates(OpFunc::BatchmatmulMxfp4WFwd.spelling(), IsaGen::Rcudd1a),
            Some([].as_slice())
        );
        // `:270`, and STILL A MISS: `stz_latch.ddl` is unvendored, so `Template` cannot name it. This
        // assertion FAILS the day it is vendored and the row restored, which is when it should.
        assert_eq!(
            ddl_templates(OpFunc::StzLatch.spelling(), IsaGen::Rcudd1a),
            None
        );
        // The control: an op-func the table genuinely does not name.
        assert_eq!(
            ddl_templates(OpFunc::Softmax.spelling(), IsaGen::Rcudd1a),
            None
        );
    }
}
