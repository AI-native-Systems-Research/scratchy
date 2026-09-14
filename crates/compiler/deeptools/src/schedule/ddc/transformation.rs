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

//! `ddc/ddc_transformation.cpp` — 18 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4, 6]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e105_isReleventComputeToPackStickDim` | 105 | 0 | 17 | `Ddc` | `ddc/ddc_transformation.cpp:693` |
//! | `e106_findReleventStickPackingTransfers` | 106 | 0 | 72 | `Ddc` | `ddc/ddc_transformation.cpp:712` |
//! | `e107_isEligibleCompute` | 107 | 0 | 19 | `Ddc` | `ddc/ddc_transformation.cpp:789` |
//! | `e108_cloneComputeForOffsetAdjustment` | 108 | 0 | 29 | `Ddc` | `ddc/ddc_transformation.cpp:1356` |
//! | `e109_transformAComputeNodeForInterSliceRestickify` | 109 | 0 | 12 | `Ddc` | `ddc/ddc_transformation.cpp:2383` |
//! | `e242_canUseFifo` | 242 | 1 | 270 | `Ddc` | `ddc/ddc_transformation.cpp:15` |
//! | `e243_canUseLatch` | 243 | 1 | 321 | `Ddc` | `ddc/ddc_transformation.cpp:287` |
//! | `e244_cloneForOffsetAdjustment` | 244 | 1 | 8 | `Ddc` | `ddc/ddc_transformation.cpp:1386` |
//! | `e245_setSizeForFixedSizeTransfers` | 245 | 1 | 34 | `Ddc` | `ddc/ddc_transformation.cpp:1731` |
//! | `e246_transformForInterSliceRestickify` | 246 | 1 | 28 | `Ddc` | `ddc/ddc_transformation.cpp:2398` |
//! | `e300_packStickDim` | 300 | 2 | 544 | `Ddc` | `ddc/ddc_transformation.cpp:811` |
//! | `e301_hoistTransfersUpForReuse` | 301 | 2 | 230 | `Ddc` | `ddc/ddc_transformation.cpp:1432` |
//! | `e302_unrollSymbolicTransfers` | 302 | 2 | 42 | `Ddc` | `ddc/ddc_transformation.cpp:1663` |
//! | `e303_unrollSpreadTransfers` | 303 | 2 | 24 | `Ddc` | `ddc/ddc_transformation.cpp:1706` |
//! | `e338_performPeSfpWorkSplit` | 338 | 3 | 36 | `Ddc` | `ddc/ddc_transformation.cpp:1395` |
//! | `e359_transformRegToFifoOrLatch` | 359 | 4 | 82 | `Ddc` | `ddc/ddc_transformation.cpp:610` |
//! | `e360_transformFor4BsplatRead` | 360 | 4 | 87 | `Ddc` | `ddc/ddc_transformation.cpp:1766` |
//! | `e376_performAutomaticShuffling` | 376 | 6 | 167 | `Ddc` | `ddc/ddc_transformation.cpp:1855` |

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ USES FOR ENTRIES 105-109 AND 242-246. Union these into this file's top block when its other
// entries land.
// ════════════════════════════════════════════════════════════════════════════════════════════════

use std::collections::BTreeSet;

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    Extent, PrimaryDim, StickDims, StickPart, stick_sizes,
};
use crate::generated::DataConnect;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::schedule::ddc::fold::{AllocId, NodeId};
use crate::schedule::ddc::metadata::{DatastageId, DestIdx, MetaDimKind, Metadata};
use crate::schedule::dsc2::{
    ComputeNode, LdsIdx, NodeName, NumChunks, Operand, TransferNode, generic_comp,
};
use crate::units::Corelet;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ USES FOR ENTRIES 300-303. Union these into this file's top block when its other entries land.
// ════════════════════════════════════════════════════════════════════════════════════════════════

use std::collections::BTreeMap;

use crate::schedule::ddc::fold::{Allocations, PadType};
use crate::schedule::ddc::transformation_util::{
    AllocationPaddings, DataStages, DatastageExploration, DscAllocations, FifoConversionSite,
    FifoResults, InternalNode, LoopBandSplit, LoopBands, LoopDims, PrimaryDimAndKind,
    ScheduleSurgery, StageExtents as UtilStageExtents, TransferMove,
    TransferMoves as UtilTransferMoves, TransferUnrolling, UnspecifiedDims, UtilNode,
    collect_loop_references, convert_result_from_fifo_to_reg, get_padding_per_dim, is_memory,
    move_transfer_node, split_loop_band_on_dim, unroll_transfer,
    unroll_transfer_for_symbolic_dims,
};
use crate::schedule::dsc2::{SyncUnits, TransferSide};
use crate::schedule::l3::dsc::SymbolicDimInfo;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ USES FOR ENTRY 300. Union these into this file's top block when its other entries land.
// ════════════════════════════════════════════════════════════════════════════════════════════════

use crate::arch::{Arch, Bytes};
use crate::bridges::superdsc_to_dataflow_ir::control_flow::{CondOp, CondValType};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{DimSet, NoEpilogueDimKind};
use crate::generated::Strategy;
use crate::schedule::ddc::fold::DataOrigin;
use crate::schedule::ddc::metadata::{DdcMemory, LoopMultiple, StoredConstraint};
use crate::schedule::ddc::shuffle::StickRepl;
use crate::schedule::ddc::transformation_util::{
    DataStage, DatastageSplitSite, DimSplit, InputIdx, InsertionPoint, LoopCond, LoopCondComposite,
    MintedConnect, MintedConnects, NewLabeledDs, NewLdsRewrite, add_new_lds,
    construct_datastage_from, construct_loop_node, memory_component, split_loop_band_on_datastage,
};
use crate::schedule::ddc::v1::{CoreClSet, StorageName};
use crate::schedule::ddl::ops::DdlComputeType;
use crate::schedule::dsc2::{
    DataInfo, Dsts, InstrAttribute, LayoutDims, ReplicationFactor, SyncDirection, SyncNode,
    SyncStrength, TransferPadding,
};
use crate::schedule::l3::dsc::PrimaryDsInfo;
use crate::units::{Core, NumFolds};

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ USES FOR ENTRY 338.
// ════════════════════════════════════════════════════════════════════════════════════════════════

use crate::schedule::ddc::transformation_util::{
    AllocateCloning, AllocationsByNode, ComponentAllocations, ComputeCloning, NodeCloning,
    PeSfpAllocateSplit, PeSfpComputeSplit, PeSfpTransferSplit,
    clone_allocate_for_pe_sfp_work_split, clone_compute_for_pe_sfp_work_split,
    clone_transfer_for_pe_sfp_work_split,
};

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE STICK-PACKING AND OFFSET-ADJUSTMENT VOCABULARY — as entries 105-109 read it.
//
// ⭐ THE TRAITS ARE THE MECHANISM FOR REACHING OPERANDS, the one part the campaign statement names
// as droppable: `traverseTreeDFSMutable`, `clone`, `addChildNode`, `getAllocation` and
// `getParentDimLoop` are all `dsc/dsc2.cpp` — outside this campaign's file list. What these five
// units OWN is the decision and the mutation, and both are below.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH COMPUTE STICK PACKING CARES ABOUT — `ComputeOpType` (`dsc/dscdefn.h:134`, its two members
/// at `:157-158`) narrowed to the two entry 105 distinguishes.
///
/// ⛔ NOT `dsc2::ComputeType`: that is the crate's DDL census, and `RECIPROCAL`/`LAYERNORMSCALE` are
/// not in it. Spelling the other ~68 members of `ComputeOpType` would state `type_` twice; these
/// units ask one three-way question of it and this is that question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeOp {
    /// `ComputeOpType::RECIPROCAL`.
    Reciprocal,
    /// `ComputeOpType::LAYERNORMSCALE` — ⛔ NOT `LAYERNORMSCALE32`, which entry 105 does not match.
    LayernormScale,
    /// Any other member.
    Other,
}

/// HOW MANY INPUTS THE PACKED COMPUTE READS — entry 105's `numberOfInputs`, an operand count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputCount(pub usize);

/// ONE LAYOUT DIM'S SCALE — `LabeledDsInfo::scale_` entry (`dsc/dscdefn.h:332`), whose two negative
/// values are sentinels the `DT_CHECK` at `dsc/designSpaceConfig.cpp:524-525` closes to `1|-1|-2`
/// wherever `applyScale` is set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    /// A non-negative scale: the dim's size comes from the data stage (`dsc/dsc2.cpp:3830`).
    Sized(f64),
    /// `-1` — the dim is exactly ONE element (`dsc/dsc2.cpp:3824-3826`).
    UnitStick,
    /// `-2` — the dim spans the whole STICK (`dsc/dsc2.cpp:3827-3829`); the one entry 106 packs for.
    StickDim,
}

/// WHAT ROLE A LABELLED DATA STRUCTURE PLAYS — `DsTypes` (`dsc/dscdefn.h:37-46`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DsType {
    /// `INPUT`.
    Input,
    /// `OUTPUT`.
    Output,
    /// `KERNEL`.
    Kernel,
    /// `KERNEL_IDX`.
    KernelIdx,
    /// `INPUT_SCALE`.
    InputScale,
    /// `KERNEL_SCALE`.
    KernelScale,
    /// `INTERNAL`.
    Internal,
    /// `NOT_SET` — the field's default, which entry 107 compares like any other value.
    NotSet,
}

/// A TRANSFER ENTRY 106 SELECTED, TOGETHER WITH THE `myLdsIdx_` IT ALREADY PROVED PRESENT.
///
/// ⛔ THE PAIRING CLOSES ENTRY 107'S THROW. `labeledDs_.at(tr->srcLdsAndLoopOffsets_.myLdsIdx_)`
/// throws for a `-1`; entry 106 `continue`s on exactly that case, so carrying the index alongside
/// the node makes `getLdsType` total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StickPackTransfer {
    /// The `dsc2::TransferNode*` the reference stores.
    pub node: NodeId,
    /// `srcLdsAndLoopOffsets_.myLdsIdx_`, proved present.
    pub src_lds: LdsIdx,
    /// Entry 106's `relevent` — its `data_connect=` matched the compute's. ⛔ TRAP: the reference
    /// COMPUTES this and never reads it (`ddc/ddc_transformation.cpp:756-766`); the push below is
    /// NOT gated on it, so it is reported rather than acted on.
    pub data_connected: bool,
}

/// WHAT ENTRY 106 HANDS ITS CALLER — the reference's three out-parameters as one value.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StickPackingTransfers {
    /// `releventInputTransfers` — the LXLU-to-`exUnit_` side.
    pub inputs: Vec<StickPackTransfer>,
    /// `releventOutputTransfer` — the `exUnit_`-to-LXSU side, absent where the reference leaves an
    /// UNINITIALISED pointer (`ddc/ddc_transformation.cpp:819`).
    pub output: Option<StickPackTransfer>,
    /// `releventTransferDstIdx`, one per arm that fired — ⛔ INCLUDING for transfers the scale and
    /// `myLdsIdx_` checks later skip, which is why the caller's `DT_CHECK` counts it separately.
    pub dst_indices: Vec<DestIdx>,
}

/// HOW MANY TIMES ONE OUTPUT REPEATS — `RepetitionWithOffset::forOutputs_` entry, an `int`
/// (`dsc/dsc2.h:952`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Repetition(pub i32);

/// HOW FAR APART THE GAPPED STICKS SIT — `AllocateNode::gapStickSpread_` value (`dsc/dsc2.h:1006`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickSpread(pub i32);

/// WHICH OUTPUT OF A COMPUTE — an index into `outputs_`/`outputsLdsAndLoopOffsets_`/`forOutputs_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputIdx(pub usize);

/// ONE LOOP NODE'S IDENTITY — the reference's `LoopNode*`, whose type is what proves the `LOOP`
/// `nodeType_`; `getParentDimLoop` returns one or `nullptr` and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopId(pub NodeId);

/// ONE `computeMaskLoopOffsets_` ENTRY (`dsc/dsc2.h:923-925`) — "put 1 if popping next element, or 0
/// if reuse is expected", which is the encoding stated on the COMMENTED-OUT `loopEleOffsets_` right
/// above it (`dsc/dsc2.h:917-922`) and nowhere on the live field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaskLoopOffset(pub i32);

impl MaskLoopOffset {
    /// The `1` entry 109 writes: advance to the next element on every trip of the loop.
    pub const ADVANCE: Self = Self(1);
}

/// A COMPUTE PROVED TO HAVE A PARENT LOOP FOR THE RESTICKIFY DIM — `getParentDimLoop(dimForLoop)`
/// returning `nullptr` made unconstructible.
///
/// ⛔ THE REFERENCE DEREFERENCES IT UNGUARDED (`ddc/ddc_transformation.cpp:2388-2390`): a compute
/// with no loop over `dimForLoop` reads `loopNode->dims_` through a null pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskedComputeSite {
    /// The compute to mask.
    pub compute: NodeId,
    /// `compNode->getParentDimLoop(dimForLoop)`.
    pub dim_loop: LoopId,
}

impl MaskedComputeSite {
    /// The site, or [`None`] where the reference would dereference a null `LoopNode*`.
    #[must_use]
    pub fn of<T: ComputeMasking + ?Sized>(
        tree: &T,
        compute: NodeId,
        dim: PrimaryDim,
    ) -> Option<Self> {
        tree.parent_dim_loop(compute, dim)
            .map(|dim_loop| Self { compute, dim_loop })
    }
}

/// THE COMPUTE NODES OF THE SCHEDULE TREE — `traverseTreeDFSMutable(nullptr, {COMPUTE}, ALL, -1,
/// -1)` and the one field entry 105 reads off each.
///
/// ⛔ `DT_CHECK(child->nodeType_ == COMPUTE && "Compute Node is expected")` is unspellable: the walk
/// yields only computes and [`Self::compute_op`] is total over them.
pub trait ComputeWalk {
    /// Every `COMPUTE` node, in the traversal's order.
    fn computes(&self) -> Vec<NodeId>;
    /// `type_`, as the three-way question these units put to it.
    fn compute_op(&self, node: NodeId) -> ComputeOp;
}

/// THE TRANSFER NODES OF THE SCHEDULE TREE — the same walk restricted to `{TRANSFER}`.
pub trait TransferWalk {
    /// Every `TRANSFER` node, in the traversal's order.
    fn transfers(&self) -> Vec<NodeId>;
    /// The transfer that node is — `src_`/`srcLdsAndLoopOffsets_` and `dstVias_`/
    /// `dstLdsAndLoopOffsets_`, each pair zipped so they cannot disagree in length.
    fn transfer(&self, node: NodeId) -> TransferNode;
}

/// THE `labeledDs_` LOOKUPS ENTRIES 106 AND 107 MAKE — both are `.at(ldsIdx)` on an index this
/// module only ever obtains already proved.
pub trait LabeledDs {
    /// `labeledDs_.at(lds).scale_`, one entry per layout dim.
    fn scale(&self, lds: LdsIdx) -> Vec<Scale>;
    /// `labeledDs_.at(lds).dsType_`.
    fn ds_type(&self, lds: LdsIdx) -> DsType;
}

/// WHAT ENTRY 108 DOES TO THE TREE — the clone, the parent insert, the cloning-map record, the
/// alloc-user bump and the gap spread.
///
/// ⭐ EVERY MUTATION THE REFERENCE PERFORMS IS ONE METHOD HERE; the DECISION of which to call, in
/// what order and with what value stays in the port.
pub trait OffsetAdjustment {
    /// `node->repetitionWithOffset_.forOutputs_`.
    fn output_repetitions(&self, node: NodeId) -> Vec<Repetition>;
    /// `node->clone()` inserted by `getMutableParent()->addChildNode(newNode, false, node)` —
    /// AFTER `node` in `node`'s own parent. Yields the clone's identity.
    fn clone_compute_after(&mut self, node: NodeId) -> NodeId;
    /// `newNode->repetitionWithOffset_.forOutputs_.at(idx) = reps`.
    fn set_output_repetition(&mut self, node: NodeId, idx: OutputIdx, reps: Repetition);
    /// `metadata.nodeCloningMap_[original].push_back(clone)` (`ddc/ddc_metadata.h:217`).
    fn record_clone(&mut self, original: NodeId, clone: NodeId);
    /// `labeledDs_.at(outputsLdsAndLoopOffsets_.at(idx).myLdsIdx_).memOrg_.at(outputs_.at(idx))
    /// .allocateNode_`, absent where either `.at()` throws or the allocation is null.
    fn output_allocation(&self, node: NodeId, idx: OutputIdx) -> Option<AllocId>;
    /// `alloc->allocUsers_.push_back({user, 1})` (`ddc/ddc_transformation.cpp:1373`) — the RAW push,
    /// ⛔ NOT `addAllocUser` (`dsc/dsc2.h:1012`): a repeat user gets a SECOND entry at count 1, not
    /// an incremented one.
    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId);
    /// `alloc->layoutDimOrder_.at(0)` — the OUTERMOST layout dim, total because the reference
    /// `DT_CHECK`s that order non-empty wherever it derives one.
    fn alloc_outermost_layout_dim(&self, alloc: AllocId) -> PrimaryDim;
    /// `alloc->gapStickSpread_[dim] = spread`.
    fn set_gap_stick_spread(&mut self, alloc: AllocId, dim: PrimaryDim, spread: StickSpread);
}

/// WHAT ENTRY 109 DOES TO THE TREE — the rename and the per-corelet, per-dim mask offset.
pub trait ComputeMasking {
    /// `compNode->name_`.
    fn compute_name(&self, compute: NodeId) -> NodeName;
    /// `compNode->name_ = name`.
    fn set_compute_name(&mut self, compute: NodeId, name: NodeName);
    /// `compNode->getParentDimLoop(dim)` (`dsc/dsc2.h:467`), absent for its `nullptr`.
    fn parent_dim_loop(&self, compute: NodeId, dim: PrimaryDim) -> Option<LoopId>;
    /// `0 .. numCoreletsUsed_DSC2_` as corelets, so the loop index cannot be an out-of-range `int`.
    fn corelets(&self) -> Vec<Corelet>;
    /// `loopNode->dims_` (`dsc/dsc2.h:575`), inner to outer. ⛔ The `MetaDimKind` half is bound and
    /// never read by entry 109, so it is not carried.
    fn loop_dims(&self, dim_loop: LoopId) -> Vec<PrimaryDim>;
    /// `compNode->instrAttribute_.computeMaskLoopOffsets_[corelet][dim_loop][dim] = offset`.
    fn set_compute_mask_loop_offset(
        &mut self,
        compute: NodeId,
        corelet: Corelet,
        dim_loop: LoopId,
        dim: PrimaryDim,
        offset: MaskLoopOffset,
    );
}

/// The suffix entry 109 appends to the masked compute's name.
const MASKED_SUFFIX: &str = "_masked";

/// Replaces: e105_isReleventComputeToPackStickDim
///
/// The LAST `RECIPROCAL` or `LAYERNORMSCALE` compute in the tree, and how many inputs stick packing
/// must then find for it — 2 once any `LAYERNORMSCALE` was seen, else 1.
///
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: `numberOfInputs` LATCHES while the chosen node does not. A
/// `LAYERNORMSCALE` followed by a `RECIPROCAL` returns the reciprocal paired with 2, and the caller
/// then requires two input transfers for a one-input op (`ddc/ddc_transformation.cpp:824`).
/// ⛔ The out-parameter is left UNTOUCHED when nothing matches, so the caller reads an uninitialised
/// pointer unless it checks the `bool` first; the [`Option`] makes that unspellable.
pub fn is_relevant_compute_to_pack_stick_dim<T: ComputeWalk + ?Sized>(
    tree: &T,
) -> Option<(NodeId, InputCount)> {
    let mut relevant = None;
    let mut inputs = InputCount(1);
    for node in tree.computes() {
        match tree.compute_op(node) {
            ComputeOp::Reciprocal => relevant = Some(node),
            ComputeOp::LayernormScale => {
                relevant = Some(node);
                inputs = InputCount(2);
            }
            ComputeOp::Other => {}
        }
    }
    relevant.map(|node| (node, inputs))
}

/// `isConnected(dataInfos, dataConnect)` (`ddc/ddc_transformation.cpp:717-722`).
///
/// ⛔ AN ABSENT CONNECT MATCHES AN ABSENT CONNECT, because the reference compares `std::string`s and
/// an unset `dataConnect_` is `""` on both sides.
fn is_connected(operands: &[Operand], connect: Option<DataConnect>) -> bool {
    operands
        .iter()
        .any(|operand| operand.data.data_connect == connect)
}

/// The first destination sitting on that component, and which destination it is.
fn first_dst_on(transfer: &TransferNode, unit: SenComponent) -> Option<DestIdx> {
    transfer
        .dsts
        .iter()
        .zip(0u32..)
        .find(|(dst, _)| dst.unit == unit)
        .map(|(_, idx)| DestIdx(idx))
}

/// Replaces: e106_findReleventStickPackingTransfers
///
/// Selects the transfers that carry the stick-packed compute's data: every LXLU→`exUnit_` transfer
/// becomes an input, an `exUnit_`→LXSU one becomes THE output, and each keeps the destination index
/// that matched. A transfer is taken only when all its destinations share the source's `myLdsIdx_`
/// and some layout dim of that data structure has scale `-2`.
///
/// ⛔ TRAP: the destination index is recorded BEFORE those two checks, so a skipped transfer still
/// contributes one — that is what the caller's `size() + 1 == size()` `DT_CHECK` is counting.
/// ⛔ TRAP: `relevent` is computed and never read; it is reported on [`StickPackTransfer`] instead.
pub fn find_relevant_stick_packing_transfers<T, L>(
    tree: &T,
    lds: &L,
    relevant_compute: &ComputeNode,
) -> StickPackingTransfers
where
    T: TransferWalk + ?Sized,
    L: LabeledDs + ?Sized,
{
    let mut found = StickPackingTransfers::default();
    for node in tree.transfers() {
        let transfer = tree.transfer(node);
        let Some(src_lds) = transfer.src.data.my_lds_idx else {
            continue;
        };

        let lx_to_ex_unit = (transfer.src.unit == SenComponent::Lxlu)
            .then(|| first_dst_on(&transfer, relevant_compute.ex_unit))
            .flatten();
        if let Some(idx) = lx_to_ex_unit {
            found.dst_indices.push(idx);
        }
        let ex_unit_to_lx = (transfer.src.unit == relevant_compute.ex_unit)
            .then(|| first_dst_on(&transfer, SenComponent::Lxsu))
            .flatten();
        if let Some(idx) = ex_unit_to_lx {
            found.dst_indices.push(idx);
        }
        if lx_to_ex_unit.is_none() && ex_unit_to_lx.is_none() {
            continue;
        }

        // Check data connect of opaque vs the transfers.
        let data_connected = if lx_to_ex_unit.is_some() {
            transfer
                .dsts
                .iter()
                .any(|dst| is_connected(&relevant_compute.inputs, dst.data.data_connect))
        } else {
            is_connected(&relevant_compute.outputs, transfer.src.data.data_connect)
        };

        // Get the tensor, check scale.at(i) == -2 where i is the index of the stick dimension.
        if transfer
            .dsts
            .iter()
            .any(|dst| dst.data.my_lds_idx != Some(src_lds))
        {
            continue;
        }
        if !lds.scale(src_lds).contains(&Scale::StickDim) {
            continue;
        }

        let taken = StickPackTransfer {
            node,
            src_lds,
            data_connected,
        };
        if lx_to_ex_unit.is_some() {
            found.inputs.push(taken);
        } else {
            found.output = Some(taken);
        }
    }
    found
}

/// Replaces: e107_isEligibleCompute
///
/// One stick-packed compute may only gather its inputs from data structures of ONE `dsType_` and
/// ONE `scale_`; a single input transfer is eligible unconditionally.
///
/// ⛔ `numberOfInputs` IS DROPPED: the reference indexes `ldsInputIdx.at(i)` for `i < numberOfInputs`
/// and its one caller fills that vector with exactly `numberOfInputs` entries
/// (`ddc/ddc_transformation.cpp:827-830`), so iterating the slice is the same walk without the throw.
pub fn is_eligible_compute<L: LabeledDs + ?Sized>(
    lds: &L,
    input_transfers: &[StickPackTransfer],
    lds_input_idx: &[LdsIdx],
) -> bool {
    let Some((first_transfer, other_transfers)) = input_transfers.split_first() else {
        return true;
    };
    if other_transfers.is_empty() {
        return true;
    }
    let first_type = lds.ds_type(first_transfer.src_lds);
    if other_transfers
        .iter()
        .any(|transfer| lds.ds_type(transfer.src_lds) != first_type)
    {
        return false;
    }
    let Some((first_idx, other_idx)) = lds_input_idx.split_first() else {
        return true;
    };
    let first_scale = lds.scale(*first_idx);
    !other_idx.iter().any(|idx| lds.scale(*idx) != first_scale)
}

/// Replaces: e108_cloneComputeForOffsetAdjustment
///
/// Spreads a compute whose output repeats: for each output repeating `n > 1` times it inserts `n-1`
/// clones after it, each doing ONE repetition, records them under the original, bumps the output
/// allocation's user count per clone, and stamps that allocation's outermost layout dim with the
/// full spread `n` — once per allocation however many clones charge it.
///
/// ⛔ THE ORIGINAL KEEPS ITS `n`: only the clones are set to 1, so the spread read on every trip is
/// the unreduced repetition, and that is what `gapStickSpread_` receives.
/// ⛔ DELIBERATE DIVERGENCE: an output whose `memOrg_` entry is missing or whose `allocateNode_` is
/// null makes the reference throw or dereference null (`ddc/ddc_transformation.cpp:1370-1373`); here
/// the clone is still made and recorded and only the user bump and the spread are skipped.
pub fn clone_compute_for_offset_adjustment<T: OffsetAdjustment + ?Sized>(
    tree: &mut T,
    node: NodeId,
) {
    let mut spread_added_allocs: Vec<AllocId> = Vec::new();
    for (idx, spread) in tree.output_repetitions(node).into_iter().enumerate() {
        let idx = OutputIdx(idx);
        for _ in 1..spread.0 {
            let new_node = tree.clone_compute_after(node);
            tree.set_output_repetition(new_node, idx, Repetition(1));
            tree.record_clone(node, new_node);
            let Some(alloc) = tree.output_allocation(node, idx) else {
                continue;
            };
            tree.add_alloc_user(alloc, new_node);
            if spread_added_allocs.contains(&alloc) {
                continue;
            }
            let dim = tree.alloc_outermost_layout_dim(alloc);
            tree.set_gap_stick_spread(alloc, dim, StickSpread(spread.0));
            spread_added_allocs.push(alloc);
        }
    }
}

/// Replaces: e109_transformAComputeNodeForInterSliceRestickify
///
/// Masks one compute for inter-slice restickify: renames it `..._masked` and, on every corelet in
/// use, tells it to ADVANCE one element per dim of its enclosing loop over the restickify dim.
///
/// ⛔ `none_trivial_input_idx` IS DROPPED — the reference binds it and never reads it
/// (`ddc/ddc_transformation.cpp:2384`). ⛔ The unconditional `return true` is the witness being
/// constructible: the one live caller only `|=`s it (`:2419`), so [`MaskedComputeSite::of`] carries
/// that answer.
pub fn transform_a_compute_node_for_inter_slice_restickify<T: ComputeMasking + ?Sized>(
    tree: &mut T,
    site: MaskedComputeSite,
) {
    let masked = NodeName(tree.compute_name(site.compute).0 + MASKED_SUFFIX);
    tree.set_compute_name(site.compute, masked);
    for corelet in tree.corelets() {
        for dim in tree.loop_dims(site.dim_loop) {
            tree.set_compute_mask_loop_offset(
                site.compute,
                corelet,
                site.dim_loop,
                dim,
                MaskLoopOffset::ADVANCE,
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE SKIP-REGISTER, OFFSET-ADJUSTMENT AND RESTICKIFY VOCABULARY — as entries 242-246 read it.
//
// ⭐ THE TRAITS ARE AGAIN THE MECHANISM FOR REACHING OPERANDS, the one part the campaign statement
// names as droppable: `getInnermostCommonAncestor`, `getNextView`, `getPrev`,
// `getNonBroadcastLdsDimSet`, `getBlockTransferSize` and `getStickSizes` are all `dsc/dsc2.cpp` —
// outside this campaign's file list. What these five units OWN is the decision each makes over
// those answers, and the mutation it performs.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// HOW MANY LOADS ONE BLOCK TRANSFER COSTS — `getBlockTransferSize(.., sizeInNumberOfLoads = true)`
/// (`dsc/dsc2.cpp:3601`), a product of per-dim LOAD counts and not an element count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Loads(pub i64);

impl Loads {
    /// The one value entry 243 admits: a single load per corelet.
    pub const ONE: Self = Self(1);
}

/// ONE DESTINATION OF A TRANSFER WITH EVERYTHING `dstIndex` SELECTS — `dstVias_.at(dstIndex).loc_`,
/// `dstLdsAndLoopOffsets_.at(dstIndex)` and that destination's `via_`, resolved ONCE.
///
/// ⛔ THE TWO `.at()`s ARE THE THROW: both units index them repeatedly and neither ever checks
/// `dstIndex`, so obtaining this value is where an out-of-range destination is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferDst<'a> {
    /// The transfer node's identity.
    pub node: NodeId,
    /// The transfer itself.
    pub transfer: &'a TransferNode,
    /// `dstVias_.at(dstIndex).loc_` zipped with `dstLdsAndLoopOffsets_.at(dstIndex)`.
    pub dst: Operand,
    /// `dstVias_.at(dstIndex).via_`.
    pub hops: &'a [SenComponent],
}

impl<'a> TransferDst<'a> {
    /// The destination, or [`None`] past the end of `dstVias_`.
    #[must_use]
    pub fn of(node: NodeId, transfer: &'a TransferNode, index: DestIdx) -> Option<Self> {
        let index = index.0 as usize;
        let dst = *transfer.dsts.get(index)?;
        Some(Self {
            node,
            transfer,
            dst,
            hops: transfer.dsts.hops(index),
        })
    }
}

/// THE COMMUNICATION CHANNEL A SKIPPED REGISTER WOULD USE — entry 242's `communicationChannel`: the
/// last component before the destination, paired with the destination's own.
///
/// ⭐ THE CONSTRUCTOR IS THE REFUSAL. A `NO_COMPONENT` or `CONSTANT` source cannot act as a FIFO, so
/// there is no channel to carry rather than a channel that has to be re-tested downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Channel {
    /// `.first` — `via_.back()`, or `src_.unit_` where the transfer has no hops.
    pub from: SenComponent,
    /// `.second` — `dstVias_.at(dstIndex).loc_.unit_`.
    pub to: SenComponent,
}

impl Channel {
    /// The channel, or [`None`] where its source cannot be used as a FIFO.
    #[must_use]
    pub fn of(dst: &TransferDst<'_>) -> Option<Self> {
        let from = dst.hops.last().copied().unwrap_or(dst.transfer.src.unit);
        match from {
            SenComponent::NoComponent | SenComponent::Constant => None,
            from => Some(Self {
                from,
                to: dst.dst.unit,
            }),
        }
    }
}

/// WHAT A LOOP ON AN ANCESTOR PATH IS ASKED — `isParametricLoop()` (`dsc/dsc2.h:599`), `dims_`
/// (`:575`) and the two datastage ids entry 242's reuse test compares (`:573-574`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathLoop {
    /// `isParametricLoop_`.
    pub parametric: bool,
    /// `dims_`, inner to outer.
    pub dims: Vec<(PrimaryDim, MetaDimKind)>,
    /// `numId_`.
    pub num: DatastageId,
    /// `denId_`.
    pub den: DatastageId,
}

/// WHAT A PATH NODE IS — `nodeType_` narrowed to the three answers these units act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathKind {
    /// `nodeType_ == CONDITION`.
    Condition,
    /// `nodeType_ == LOOP`, with what a `dsc2::LoopNode` is then asked.
    Loop(PathLoop),
    /// Any other node type — a block, a transfer, a compute, an allocate or a sync.
    Other,
}

/// ONE NODE ON AN ANCESTOR PATH.
///
/// ⛔ NAMES ARE NOT CARRIED: `name_` is read only by the `std::cerr` diagnostics both units emit
/// under `transformationReportLevel_`, and those have no effect on the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathNode {
    /// The node.
    pub node: NodeId,
    /// What it is.
    pub kind: PathKind,
}

impl PathNode {
    /// `checkUnsupportedNode`'s body — a CONDITION, or a LOOP that `isParametricLoop()`.
    #[must_use]
    pub fn unsupported(&self) -> bool {
        match &self.kind {
            PathKind::Condition => true,
            PathKind::Loop(dim_loop) => dim_loop.parametric,
            PathKind::Other => false,
        }
    }
}

/// ONE PATH `getInnermostCommonAncestor` FILLS, LESS ITS LAST ENTRY — `path.at(0)`, the node itself,
/// and `path.at(1) ..= path.size() - 2`, its strict ancestors BELOW the common one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AncestorPath {
    node: PathNode,
    between: Vec<PathNode>,
}

impl AncestorPath {
    /// The path from one node up to (but not including) the common ancestor, innermost first.
    #[must_use]
    pub const fn new(node: PathNode, between: Vec<PathNode>) -> Self {
        Self { node, between }
    }

    /// `path.at(0)`.
    #[must_use]
    pub const fn node(&self) -> &PathNode {
        &self.node
    }

    /// `path.at(1) ..= path.size() - 2`, in path order.
    #[must_use]
    pub fn between(&self) -> &[PathNode] {
        &self.between
    }

    /// `path.at(path.size() - 2)` — the common ancestor's own child on this path, WHICH IS THE NODE
    /// ITSELF where the ancestor holds it directly. The reference's `size() - 2` underflows for a
    /// one-entry path; a node is never its own innermost common ancestor with another, so it cannot.
    #[must_use]
    pub fn child_of_ancestor(&self) -> &PathNode {
        self.between.last().unwrap_or(&self.node)
    }

    /// `for (i = 0; i < path.size() - 1; ++i)` — entry 242's `loopsWithoutReuse` domain: the node
    /// and every ancestor below the common one.
    pub fn below_ancestor(&self) -> impl Iterator<Item = &PathNode> {
        core::iter::once(&self.node).chain(self.between.iter())
    }
}

/// THE TWO PATHS `getInnermostCommonAncestor(transferNode, consumer, ..)` FILLS
/// (`dsc/dsc2.cpp:4243`), WITH THE SHARED ANCESTOR HELD ONCE.
///
/// ⭐ HOLDING IT ONCE IS WHAT REMOVES A `DT_ERROR`: the reference aborts with *"Failed to find the
/// common ancestor"* when the two paths' backs differ, and here they cannot differ.
/// ⛔ BOTH UNITS CALL IT AS `(transferNode, consumer)`, which is why the paths are named for those
/// roles rather than for argument positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ancestry {
    /// `pathToTransfer`.
    pub to_transfer: AncestorPath,
    /// `pathToConsumer`.
    pub to_consumer: AncestorPath,
    /// `pathToTransfer.back()`, which is also `pathToConsumer.back()`.
    pub ancestor: PathNode,
}

impl Ancestry {
    /// Entry 242's `checkUnsupportedNode` domain — `for (i = 1; i < path.size(); ++i)` on BOTH
    /// paths, which INCLUDES the common ancestor.
    ///
    /// ⛔⛔ TRAP, AND IT IS A REAL DIFFERENCE BETWEEN THE TWO UNITS: entry 243's identical-looking
    /// walk stops one entry short (`ddc/ddc_transformation.cpp:463` against `:89`), so a parametric
    /// common ancestor refuses a FIFO and permits a LATCH.
    pub fn ancestors_through_common(&self) -> impl Iterator<Item = &PathNode> {
        self.ancestors_below_common()
            .chain(core::iter::once(&self.ancestor))
    }

    /// Entry 243's domain — `for (i = 1; i < path.size() - 1; ++i)` on both paths, which EXCLUDES
    /// the common ancestor. The reference walks it once per path; one `bool` needs it once.
    pub fn ancestors_below_common(&self) -> impl Iterator<Item = &PathNode> {
        self.to_consumer
            .between()
            .iter()
            .chain(self.to_transfer.between().iter())
    }
}

/// A NODE IN THE SCOPE BETWEEN A PRODUCER AND ITS CONSUMER — the `nodeType_` tests and the
/// `dynamic_cast<const dsc2::BlockNode *>` fallback both recursions make, as one value.
///
/// ⛔ NOT [`crate::schedule::ddc::fold::BlockId`]: that names a `BLOCK` node, while `isBlockNode()`
/// (`dsc/dsc2.h:479`) also admits a LOOP and a CONDITION, and it is that wider set the recursions
/// descend into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeNode {
    /// `nodeType_ == COMPUTE`.
    Compute(ComputeNode),
    /// `nodeType_ == TRANSFER`.
    Transfer(TransferNode),
    /// The `dynamic_cast` succeeded — a BLOCK, LOOP or CONDITION, and its `getNextView(ALL)`
    /// children in order.
    Nest(Vec<NodeId>),
    /// An allocate, sync or stick-mask node: no arm fires and the reference falls through.
    Other,
}

/// THE CONSUMERS OF ONE TRANSFER RESULT — `metadata.dataConnects_.at(resultDc).consumers_` as entry
/// 243 is handed them, NON-EMPTY so that its `DT_CHECK(!consumers.empty())` is unspellable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Consumers {
    first: NodeId,
    rest: Vec<NodeId>,
}

impl Consumers {
    /// A consumed connect has at least one consumer, and this is how that is stated.
    #[must_use]
    pub const fn new(first: NodeId, rest: Vec<NodeId>) -> Self {
        Self { first, rest }
    }

    /// Every consumer, in census order.
    pub fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }

    /// `is_any_of(node, consumers)`.
    #[must_use]
    pub fn contains(&self, node: NodeId) -> bool {
        self.iter().any(|consumer| consumer == node)
    }
}

/// WHAT THE SKIP-REGISTER DECISIONS ASK OF THE SCHEDULE TREE.
pub trait ScopeTree {
    /// `getInnermostCommonAncestor(transferNode, consumer, pathToTransfer, pathToConsumer)` as ONE
    /// value.
    fn ancestry(&self, transfer: NodeId, consumer: NodeId) -> Ancestry;
    /// What that node is, and — for a block, a loop or a condition — its `getNextView(ALL)`
    /// children. ⛔ `ALL` FILTERS NOTHING (`dsc/dsc2.cpp:2222`); it is the argument's default.
    fn scope_node(&self, node: NodeId) -> ScopeNode;
    /// `node->getPrev()` (`dsc/dsc2.h:463`) — ⛔ THE SAME `prev_` FIELD `getMutableParent()` HANDS
    /// BACK (`:464`), so comparing two nodes' `getPrev()` asks whether they are SIBLINGS.
    fn parent(&self, node: NodeId) -> Option<NodeId>;
    /// `getNonBroadcastLdsDimSet(myLdsIdx_)` (`dsc/dsc2.cpp:4050`) — EMPTY for an absent index,
    /// which is the reference's own `ldsIdx < 0` arm and not a case this port adds.
    fn non_broadcast_lds_dims(&self, lds: Option<LdsIdx>) -> BTreeSet<PrimaryDim>;
}

/// THE DATASTAGE EXTENTS ENTRY 242 COMPARES — `dataStageParam_.at(id).ss_.primaryDimToVal_st(dim)`
/// (`dsc/dims.cpp:647`).
///
/// ⭐ REACHING THEM AT ALL IS THE FLAG. Entry 242 reads them only once `dataStageExplorationDone_`
/// is set, so the port takes them as an [`Option`] and [`None`] IS that flag being false.
pub trait StageExtents {
    /// One stage's stick-view extent for one dim.
    fn stage_extent(&self, stage: DatastageId, dim: PrimaryDim) -> Extent;
}

/// WHAT ONE TRANSFER COSTS PER CORELET — entry 243's per-corelet `getBlockTransferSize`.
pub trait TransferLoads {
    /// `0 .. numCoreletsUsed_DSC2_`, as corelets rather than as an `int` that can range past them.
    fn corelets(&self) -> Vec<Corelet>;
    /// `getBlockTransferSize(*transferNode, transferNode->src_.unit_, clId, false, true)` — the
    /// transfer and the unit are one value here, so the two cannot disagree.
    fn block_transfer_loads(&self, transfer: NodeId, corelet: Corelet) -> Loads;
}

/// THE `getStickSizes(labeledDs_.at(lds).dsType_)` LOOKUP — one labelled DS's stick dims, which is
/// what entry 041's port ([`stick_sizes`]) is then asked of.
pub trait DsSticks {
    /// `labeledDs_.at(lds).dsType_`'s `stickDimOrder_` zipped with its `stickSize_`.
    fn ds_stick_dims(&self, lds: LdsIdx) -> StickDims;
}

/// WHAT ENTRY 245 READS AND WRITES BESIDE THE TRANSFER WALK.
pub trait FixedSizeTransfers {
    /// `metadata.dataConnects_.find(dc)->second.consumers_`, each consumer AS WHAT IT IS. ⛔ A
    /// connect the census does not carry answers EMPTY, which is the reference's second `continue`.
    fn connect_consumers(&self, connect: DataConnect) -> Vec<ScopeNode>;
    /// `transferNode->transferSize_.clear()` FOLLOWED BY one entry per stick dim — one write, so a
    /// cleared-but-unfilled `transferSize_` is unspellable.
    fn set_transfer_size(&mut self, transfer: NodeId, sizes: Vec<(PrimaryDim, Elements)>);
}

/// THE COMPUTE NODE BEHIND ONE IDENTITY — `inputs_`/`outputs_` each zipped with their
/// `..LdsAndLoopOffsets_`.
pub trait ComputeNodes {
    /// The compute that node is.
    fn compute(&self, compute: NodeId) -> ComputeNode;
}

/// `static_cast<const dsc2::BlockNode *>(path.back())->getNextView(ALL)` — the common ancestor's
/// children in order.
///
/// ⛔ THE REFERENCE `static_cast`s BLINDLY: the innermost common ancestor is a block by
/// construction, and a node that is not one answers EMPTY here rather than being reinterpreted.
fn children_of<T: ScopeTree + ?Sized>(tree: &T, node: NodeId) -> Vec<NodeId> {
    match tree.scope_node(node) {
        ScopeNode::Nest(children) => children,
        ScopeNode::Compute(_) | ScopeNode::Transfer(_) | ScopeNode::Other => Vec::new(),
    }
}

/// Whether a consumer reads that result on MORE THAN ONE input — the DCC-transformed FMA that reads
/// two different FIFO elements, which both units refuse (`ddc/ddc_transformation.cpp:56-75`).
///
/// ⛔ AN ABSENT CONNECT MATCHES AN ABSENT ONE, because the reference compares `std::string`s and an
/// unset `dataConnect_` is `""` on both sides.
fn reads_result_twice(inputs: &[Operand], result: Option<DataConnect>) -> bool {
    inputs
        .iter()
        .filter(|input| input.data.data_connect == result)
        .count()
        > 1
}

/// Entry 242's `usesCommunicationChannel` — whether that node, or anything nested in it, drives the
/// intended FIFO. The producer itself is skipped.
fn uses_communication_channel<T: ScopeTree + ?Sized>(
    tree: &T,
    producer: NodeId,
    node: NodeId,
    channel: Channel,
    other_consumer_inputs: &BTreeSet<SenComponent>,
) -> bool {
    if node == producer {
        return false;
    }
    match tree.scope_node(node) {
        ScopeNode::Compute(compute) => {
            let reads_the_fifo = (compute.ex_unit == channel.to
                || other_consumer_inputs.contains(&compute.ex_unit))
                && compute.inputs.iter().any(|input| input.unit == channel.from);
            let drives_the_fifo = compute.ex_unit == channel.from
                && compute.outputs.iter().any(|output| output.unit == channel.to);
            reads_the_fifo || drives_the_fifo
        }
        ScopeNode::Transfer(transfer) => transfer.dsts.routes().any(|(dst, hops)| {
            // `comp` walks the hop chain from the source, one component behind `it`.
            let mut comp = transfer.src.unit;
            for &hop in hops {
                if comp == channel.from
                    && (hop == channel.to || other_consumer_inputs.contains(&hop))
                {
                    return true;
                }
                comp = hop;
            }
            comp == channel.from
                && (dst.unit == channel.to || other_consumer_inputs.contains(&dst.unit))
        }),
        ScopeNode::Nest(children) => children.into_iter().any(|child| {
            uses_communication_channel(tree, producer, child, channel, other_consumer_inputs)
        }),
        ScopeNode::Other => false,
    }
}

/// Entry 242's `loopsWithoutReuse` — whether every loop on that path either carries a dim the
/// transfer moves or is known to trip exactly once.
fn loops_without_reuse<S: StageExtents + ?Sized>(
    stages: Option<&S>,
    path: &AncestorPath,
    relevant_dims: &BTreeSet<PrimaryDim>,
) -> bool {
    for entry in path.below_ancestor() {
        let PathKind::Loop(dim_loop) = &entry.kind else {
            continue;
        };
        for &(dim, kind) in &dim_loop.dims {
            if relevant_dims.contains(&dim) && kind != MetaDimKind::WindowDim {
                continue;
            }
            // The transfer result is reused unless the dimension is known to be 1.
            let Some(stages) = stages else {
                return false;
            };
            if stages.stage_extent(dim_loop.num, dim) != stages.stage_extent(dim_loop.den, dim) {
                return false;
            }
        }
    }
    true
}

/// Replaces: e242_canUseFifo
///
/// Whether a transfer's result can reach its one consumer through a FIFO instead of a register: the
/// channel must exist, the consumer must not read the result on two inputs, neither ancestor path
/// may cross a condition or a parametric loop, nothing scheduled between the two may use that
/// channel, and — unless they are siblings — no loop over the scope may make the result be reused.
/// ⛔ TRAP: `otherConsumerInputs` COMPARES AN INPUT UNIT AGAINST `loc_.storage_` (`:51`), so the
/// destination's own unit normally lands in the set. ⛔ TRAP: `LXLUVALUE` answers FALSE here and
/// TRUE in entry 243 (`:289`), on the same field. ⛔ The `std::cerr` reports are dropped: they read
/// nothing but names and change no IR, which is why `getNodeDescription` is not called.
pub fn can_use_fifo<T, S>(
    tree: &T,
    stages: Option<&S>,
    transfer: &TransferDst<'_>,
    consumer: NodeId,
) -> bool
where
    T: ScopeTree + ?Sized,
    S: StageExtents + ?Sized,
{
    if transfer.dst.unit == SenComponent::Lxluvalue {
        return false;
    }
    let Some(channel) = Channel::of(transfer) else {
        return false;
    };

    // Other inputs of the consumer, to avoid triangular dependencies that would rely on fifo depth.
    let mut other_consumer_inputs = BTreeSet::new();
    if let ScopeNode::Compute(compute) = tree.scope_node(consumer) {
        for input in &compute.inputs {
            if input.unit != transfer.dst.storage {
                other_consumer_inputs.insert(input.unit);
            }
        }
        if reads_result_twice(&compute.inputs, transfer.dst.data.data_connect) {
            return false;
        }
    }

    let ancestry = tree.ancestry(transfer.node, consumer);
    if ancestry.ancestors_through_common().any(PathNode::unsupported) {
        return false;
    }

    let start = ancestry.to_transfer.child_of_ancestor().node;
    let end = ancestry.to_consumer.child_of_ancestor().node;
    let mut checking = false;
    for child in children_of(tree, ancestry.ancestor.node) {
        if child == start {
            checking = true;
        } else if !checking {
            continue;
        }
        if uses_communication_channel(
            tree,
            transfer.node,
            child,
            channel,
            &other_consumer_inputs,
        ) {
            return false;
        }
        if child == end {
            // The path to the consumer (inclusive) does not access the channel of interest.
            break;
        }
    }

    if tree.parent(transfer.node) == tree.parent(consumer) {
        return true;
    }

    let transfer_dims = tree.non_broadcast_lds_dims(transfer.transfer.src.data.my_lds_idx);
    loops_without_reuse(stages, &ancestry.to_consumer, &transfer_dims)
        && loops_without_reuse(stages, &ancestry.to_transfer, &transfer_dims)
}

/// Entry 243's `analyzeLatchUsage` — whether the scope below that node is free of anything that
/// would conflict with the intended latch. The producer itself is accepted.
///
/// ⛔ `unit`, `srcIndex` AND `dataConnect` ARE DEAD: the reference threads all three through the
/// recursion and reads none of them (`ddc/ddc_transformation.cpp:508-576`), so `destUnit` — bound at
/// `:347` for this call alone — is not carried either.
fn analyze_latch_usage<T: ScopeTree + ?Sized>(
    tree: &T,
    producer: NodeId,
    node: NodeId,
    consumers: &Consumers,
) -> bool {
    if node == producer {
        return true;
    }
    match tree.scope_node(node) {
        ScopeNode::Compute(_) => consumers.contains(node),
        ScopeNode::Transfer(transfer) => {
            consumers.contains(node)
                || transfer
                    .dsts
                    .iter()
                    .all(|dst| dst.storage != SenComponent::Latch)
        }
        ScopeNode::Nest(children) => children
            .into_iter()
            .all(|child| analyze_latch_usage(tree, producer, child, consumers)),
        ScopeNode::Other => true,
    }
}

/// Replaces: e243_canUseLatch
///
/// Whether one transfer's result can be latched for all of its consumers: one load per corelet,
/// every consumer a non-opaque, non-PT compute of the same type reading the result on the same input
/// positions and never twice, no condition or parametric loop strictly between transfer and
/// consumer, and nothing else in that scope computing or latching.
/// ⛔ TRAP: `LXLUVALUE` answers TRUE — see entry 242's ⛔ on the same field. ⛔ `isOpaqueOp_` is read
/// as a `metadata.opaqueOps_` entry: the flag and the entry are written together
/// (`ddc/ddl/ddl_conversion.cpp:1575-1577`) and cloned together (`ddc_transformation_util.cpp:1497`).
/// ⛔ `DT_ERROR` on `!dataStageExplorationDone_` is unreachable: its one caller tests that flag on
/// the line before the call (`:675`), so no phase argument is taken.
pub fn can_use_latch<T>(
    tree: &T,
    metadata: &Metadata,
    transfer: &TransferDst<'_>,
    consumers: &Consumers,
) -> bool
where
    T: ScopeTree + TransferLoads + ?Sized,
{
    if transfer.dst.unit == SenComponent::Lxluvalue {
        return true;
    }
    if transfer.dst.storage == SenComponent::Lxluscalereg {
        return false;
    }
    if transfer.transfer.src.unit == SenComponent::Constant {
        return false;
    }
    if tree
        .corelets()
        .into_iter()
        .any(|corelet| tree.block_transfer_loads(transfer.node, corelet) != Loads::ONE)
    {
        return false;
    }

    let result_connect = transfer.dst.data.data_connect;
    let mut computes = Vec::new();
    for consumer in consumers.iter() {
        // `computeConsumersCount != consumers.size()` — a non-compute consumer refuses.
        let ScopeNode::Compute(compute) = tree.scope_node(consumer) else {
            return false;
        };
        if reads_result_twice(&compute.inputs, result_connect) {
            return false;
        }
        computes.push((consumer, compute));
    }

    // Latched data is used on the same input port for all compute-consumers, so all of them must
    // agree on the compute type, and PT may not have access to all FIFOs for all sources.
    let mut prev_op = None;
    for (_, compute) in &computes {
        if compute.ex_unit == SenComponent::Pt {
            return false;
        }
        match prev_op {
            None => prev_op = Some(compute.op),
            Some(op) if op != compute.op => return false,
            Some(_) => {}
        }
    }

    // The source position(s) every consumer reads the result on.
    // ⛔ `DT_CHECK(!srcIndex.empty())` holds because the consumers are the connect's own census.
    let mut src_index = BTreeSet::new();
    let mut first_consumer = true;
    for (node, compute) in &computes {
        if metadata.opaque_ops.contains_key(node) {
            return false;
        }
        let mut use_count = 0;
        for (position, input) in compute.inputs.iter().enumerate() {
            if input.data.data_connect != result_connect {
                continue;
            }
            use_count += 1;
            if first_consumer {
                src_index.insert(position);
            } else if !src_index.contains(&position) {
                return false;
            }
        }
        if !first_consumer && src_index.len() != use_count {
            return false;
        }
        first_consumer = false;
    }

    for consumer in consumers.iter() {
        let ancestry = tree.ancestry(transfer.node, consumer);
        if ancestry.ancestors_below_common().any(PathNode::unsupported) {
            return false;
        }
        let start = ancestry.to_transfer.child_of_ancestor().node;
        let end = ancestry.to_consumer.child_of_ancestor().node;
        let mut checking = false;
        for child in children_of(tree, ancestry.ancestor.node) {
            if child == start {
                checking = true;
            } else if !checking {
                continue;
            }
            if !analyze_latch_usage(tree, transfer.node, child, consumers) {
                return false;
            }
            if child == end {
                break;
            }
        }
    }

    true
}

/// Replaces: e244_cloneForOffsetAdjustment
///
/// Runs entry 108 over every compute in the schedule tree.
///
/// ⛔ THE WALK IS A SNAPSHOT: `traverseTreeDFSMutable` hands back the vector it has already
/// collected, so the clones entry 108 inserts beside each compute are NOT revisited.
/// ⛔ The unconditional `return true` is dropped — its one caller discards it (`ddc/ddcv1.cpp:3732`).
pub fn clone_for_offset_adjustment<T: ComputeWalk + OffsetAdjustment + ?Sized>(tree: &mut T) {
    for node in tree.computes() {
        clone_compute_for_offset_adjustment(tree, node);
    }
}

/// Replaces: e245_setSizeForFixedSizeTransfers
///
/// Gives every transfer that feeds an LXLU compute the stick sizes of the data structure at its
/// FIRST such destination, and stops scanning that transfer there.
///
/// ⛔ DELIBERATE DIVERGENCE: `DT_CHECK(ldsIdx >= 0 && ldsIdx < labeledDs_.size())` aborts on a
/// destination that carries no `myLdsIdx_`; here that destination is skipped and the scan continues,
/// because a runtime refusal is not available to this crate and an abort states nothing about the
/// remaining destinations.
pub fn set_size_for_fixed_size_transfers<T>(tree: &mut T)
where
    T: TransferWalk + DsSticks + FixedSizeTransfers + ?Sized,
{
    for node in tree.transfers() {
        let transfer = tree.transfer(node);
        for dst in transfer.dsts.iter() {
            let Some(connect) = dst.data.data_connect else {
                continue;
            };
            let feeds_lxlu = tree.connect_consumers(connect).into_iter().any(|consumer| {
                matches!(consumer, ScopeNode::Compute(compute) if compute.ex_unit == SenComponent::Lxlu)
            });
            if !feeds_lxlu {
                continue;
            }
            let Some(lds) = dst.data.my_lds_idx else {
                continue;
            };
            let sizes = stick_sizes(&tree.ds_stick_dims(lds), StickPart::Whole);
            tree.set_transfer_size(node, sizes);
            break;
        }
    }
}

/// Replaces: e246_transformForInterSliceRestickify
///
/// Masks (entry 109) every PT compute of the shape `(ZERO, ONE, <a labelled DS>)` whose input and
/// output stick dim orders differ, over the output's OUTERMOST stick dim.
///
/// ⛔ THE COMPARISON IS OF THE WHOLE DIM-AND-SIZE LISTS, not of the dim orders alone: `operator!=`
/// on `std::vector<std::pair<PrimaryDimTypes, int>>` (`:2419`), so two identical orders with one
/// differing extent transform. ⛔ `didTransformation` is dropped — its one caller discards it
/// (`ddc/ddcv1.cpp:3764`) — and so is `none_trivial_input_idx`, which entry 109 never reads.
pub fn transform_for_inter_slice_restickify<T>(tree: &mut T)
where
    T: ComputeWalk + ComputeNodes + DsSticks + ComputeMasking + ?Sized,
{
    for node in tree.computes() {
        let compute = tree.compute(node);
        if generic_comp(compute.ex_unit) != Some(GenericComp::Pt) {
            continue;
        }
        let [zero, one, variable] = compute.inputs.as_slice() else {
            continue;
        };
        if zero.unit != SenComponent::Zero || one.unit != SenComponent::One {
            continue;
        }
        let (Some(input_lds), Some(output_lds)) = (
            variable.data.my_lds_idx,
            compute.outputs.first().and_then(|out| out.data.my_lds_idx),
        ) else {
            continue;
        };
        let input_sticks = stick_sizes(&tree.ds_stick_dims(input_lds), StickPart::Whole);
        let output_sticks = stick_sizes(&tree.ds_stick_dims(output_lds), StickPart::Whole);
        if input_sticks == output_sticks {
            continue;
        }
        let Some(&(outermost, _)) = output_sticks.first() else {
            continue;
        };
        let Some(site) = MaskedComputeSite::of(&*tree, node, outermost) else {
            continue;
        };
        transform_a_compute_node_for_inter_slice_restickify(tree, site);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ THE STICK-PACKING VOCABULARY — as entry 300 reads and writes it.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT ENTRY 300 READS BEFORE IT COMMITS TO ANYTHING — the whole first half of the reference
/// (`ddc/ddc_transformation.cpp:811-957`), which mutates nothing and whose eight `return false`
/// paths are [`PackStickDimSite::of`]'s [`None`].
pub trait PackStickDimReads:
    ComputeWalk
    + TransferWalk
    + ComputeNodes
    + LabeledDs
    + ComputeMasking
    + DscAllocations
    + ScheduleSurgery
{
    /// `getBufferCapacityForNodePerDim(node, lds, storage, -1, -1)` (`dsc/dsc2.cpp:3967`) — one
    /// entry per LAYOUT dim of that data structure, in its own layout order, [`None`] for the
    /// reference's `-1`.
    fn buffer_capacity_per_dim(
        &self,
        node: NodeId,
        lds: LdsIdx,
        storage: SenComponent,
    ) -> Vec<(PrimaryDim, Option<Elements>)>;

    /// `primaryDsInfo_.at(ds_type)` — its `layoutDimOrder_` and its `stickDimOrder_`/`stickSize_`
    /// pair as one value, absent for that `.at()`'s throw.
    fn primary_ds_info(&self, ds_type: DsType) -> Option<PrimaryDsInfo>;
}

/// WHAT ENTRY 300 THEN DOES TO THE TREE, THE DSC AND THE METADATA.
///
/// ⭐ EVERY MUTATION THE REFERENCE PERFORMS IS ONE METHOD HERE; the DECISION of which to call, in
/// what order and with what value stays in the port. Its reads reach the ported entries 113, 114,
/// 248, 251 and 257 directly, and entry 123 through [`NewLdsRewrite`].
pub trait PackStickDim:
    PackStickDimReads
    + LoopBands
    + UtilTransferMoves
    + FifoResults
    + TransferUnrolling
    + NewLabeledDs
    + NewLdsRewrite
    + MintedConnects
    + OffsetAdjustment
{
    /// `labeledDs_.at(labeledDs_.size() - 1)` — the entry entry 257 is handed as its reference
    /// (`ddc/ddc_transformation.cpp:1178`), TOTAL for the same reason [`NewLabeledDs::last_lds_pos`]
    /// is.
    fn last_lds_entry(&self) -> <Self as NewLabeledDs>::Entry;

    /// `allocNode->startAddressCoreCorelet_.getSingleData({{0, core}, {1, corelet}, {2, 0}})`,
    /// absent where that core/corelet pair holds no placed address.
    fn alloc_start_address(&self, alloc: AllocId, core: Core, corelet: Corelet) -> Option<Bytes>;

    /// `allocNode->startAddressCoreCorelet_.insertData(address, {core, corelet, 0})`.
    fn set_alloc_start_address(
        &mut self,
        alloc: AllocId,
        core: Core,
        corelet: Corelet,
        address: Bytes,
    );

    /// `loopNode->numId_ = num` — the counterpart of [`LoopBands::set_loop_den`].
    fn set_loop_num(&mut self, loop_node: LoopId, num: DatastageId);

    /// `loopNode->clone()` — ⛔ A CHILDLESS, UNPARENTED COPY: `BlockNode`'s copy constructor for its
    /// children does nothing on purpose (`dsc/dsc2.h:526-535`), which is why the reference has to
    /// re-add the inner loop by hand.
    fn clone_loop(&mut self, loop_node: LoopId) -> LoopId;

    /// `new dsc2::TransferNode()` with its fields, unparented.
    fn new_transfer(&mut self, node: TransferNode) -> NodeId;

    /// `new dsc2::ComputeNode()` with its fields, unparented.
    fn new_compute(&mut self, node: ComputeNode) -> NodeId;

    /// `new dsc2::SyncNode()` with its fields, unparented — both ends name each other by
    /// [`NodeName`], so a half-linked pair is unspellable.
    fn new_sync(&mut self, node: SyncNode) -> NodeId;

    /// That transfer's `src_`/`srcLdsAndLoopOffsets_` and `dstVias_`/`dstLdsAndLoopOffsets_`
    /// replaced wholesale — the write side of [`TransferWalk::transfer`].
    fn set_transfer(&mut self, transfer: NodeId, node: TransferNode);

    /// `allocNode->allocUsers_.clear()`.
    fn clear_alloc_users(&mut self, alloc: AllocId);

    /// `allocNode->removeAllocUser(user)` (`dsc/dsc2.h:1017`) — the counterpart of
    /// [`FifoResults::add_alloc_user`].
    fn remove_alloc_user(&mut self, alloc: AllocId, user: NodeId);

    /// `allocNode->clone()` inserted by `getMutableParent()->addChildNode(clone, false, allocNode)`.
    /// ⛔ AN `AllocateNode` IS NOT A `BlockNode`, so unlike [`Self::clone_loop`] this copy INHERITS
    /// `allocUsers_` and `startAddressCoreCorelet_` (`dsc/dsc2.h:974`) — which is why entry 300
    /// clears the users and ADDS to the inherited address.
    fn clone_allocate_after(&mut self, alloc: AllocId) -> AllocId;

    /// `labeledDs_.at(lds).memOrg_[storage] = { allocateNode_ = alloc, isPresent = false }` — ⛔ NOT
    /// [`DscAllocations::set_allocation_in`], which sets `isPresent = true`.
    fn set_allocation_absent(&mut self, lds: LdsIdx, storage: DdcMemory, alloc: AllocId);

    /// `labeledDs_.at(lds).dsName_ = name`.
    fn set_lds_name(&mut self, lds: LdsIdx, name: StorageName);

    /// `labeledDs_.at(lds).dsType_ = ds_type`.
    fn set_lds_type(&mut self, lds: LdsIdx, ds_type: DsType);

    /// `primaryDsInfo_[INTERNAL].layoutDimOrder_ = layout`.
    fn set_internal_layout_order(&mut self, layout: LayoutDims);

    /// ONE push onto `primaryDsInfo_[INTERNAL]`'s `stickDimOrder_`, `stickSize_` AND `stickRepl_`,
    /// so the three parallel vectors cannot desynchronise.
    fn push_internal_stick_dim(&mut self, dim: PrimaryDim, size: Elements, repl: StickRepl);
}

/// `std::gcd(a, b)` — the crate's other two are private to `islands/dataflow_ir/ty.rs`.
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

/// `constraint.updateValues(values); constraint.loopDimKind_ = Unpadded` — the pair entry 300
/// performs at every one of its nine constraint writes ([`LoopMultiple`] holds `mustBeMultiple_`
/// alongside the kind, so setting the kind must PRESERVE the multiple).
fn force_unpadded(constraint: &mut StoredConstraint, values: &[f32]) {
    constraint.update_values(values);
    constraint.multiple = match constraint.multiple {
        LoopMultiple::Off(_) => LoopMultiple::Off(Some(MetaDimKind::Unpadded)),
        LoopMultiple::Unkinded | LoopMultiple::NoEpilogue(_) => {
            LoopMultiple::NoEpilogue(NoEpilogueDimKind::Unpadded)
        }
    };
}

/// `lds.scale_.at(getDimIndexInLayoutOrder(lds.dsType_, dim))` (`dsc/designSpaceConfig.cpp:429`).
///
/// ⛔ [`None`] IS THE REFERENCE'S `-1` INDEX — a dim this ds type's layout order does not name, which
/// it feeds straight into `scale_.at(-1)` and throws on. ⛔ NOT [`crate::schedule::ddc::v1::StageSizes::lds_scale`],
/// whose three-arm answer cannot state `scaleDimInput != scaleDimOutput`.
fn dim_scale(layout: &LayoutDims, scale: &[Scale], dim: PrimaryDim) -> Option<Scale> {
    let at = layout.iter().position(|held| held == dim)?;
    scale.get(at).copied()
}

/// A COMPUTE WHOSE STICK CAN BE REPACKED, WITH EVERYTHING THE REPACK NEEDS ALREADY PROVED PRESENT.
///
/// ⛔⛔ ALL EIGHT `return false` PATHS OF THE REFERENCE'S READ HALF ARE THIS WITNESS MISSING, and so
/// are the seven `.at()` throws and the one `DT_CHECK` between them: nothing is minted until it
/// holds, so an aborting run cannot leave a half-built datastage behind.
#[derive(Debug, Clone)]
pub struct PackStickDimSite<D> {
    compute: NodeId,
    inputs: Vec<StickPackTransfer>,
    output: StickPackTransfer,
    lds_output_idx: LdsIdx,
    final_data_info: DataInfo,
    rest_of_final_data_info: Vec<DataInfo>,
    allocate_in: Vec<AllocId>,
    allocate_out: AllocId,
    layout: LayoutDims,
    stick_dims: Vec<(PrimaryDim, Elements)>,
    stick_dim_key: DimSet,
    new_stick_dims: Vec<(PrimaryDim, Elements)>,
    parent_loop: LoopId,
    parent_dims: LoopDims,
    bot_reference: DataStage<D>,
    chunk_reference: DataStage<D>,
}

impl<D: Clone + UtilStageExtents> PackStickDimSite<D> {
    /// The site, or [`None`] for any of the reference's refusals.
    ///
    /// ⚠️ `exploration` IS CHECKED HERE and not where entry 248 would: the reference reaches that
    /// flag only after it has already minted two datastages and two loops, and refusing up front is
    /// the same answer without the debris.
    /// ⛔ TRAP, AND IT IS THE REFERENCE'S: `lenderDims` is collected in full and only `at(0)` is ever
    /// read (`:917`), so one `find` is the same walk.
    #[must_use]
    pub fn of<S: PackStickDimReads + ?Sized>(
        tree: &S,
        stages: &DataStages<D>,
        exploration: DatastageExploration,
    ) -> Option<Self> {
        if exploration == DatastageExploration::Done {
            return None;
        }
        let (compute, num_inputs) = is_relevant_compute_to_pack_stick_dim(tree)?;
        let compute_node = tree.compute(compute);
        let found = find_relevant_stick_packing_transfers(tree, tree, &compute_node);
        if found.inputs.len() != num_inputs.0 {
            return None;
        }
        // `DT_CHECK(releventInputTransfers.size() + 1 == releventTransferDstIdx.size())` (`:826`) —
        // an abort over two counts entry 106 reports separately, so it is a refusal here.
        if found.inputs.len() + 1 != found.dst_indices.len() {
            return None;
        }
        // The reference reads an UNINITIALISED `releventOutputTransfer` when entry 106 set none.
        let output = found.output?;
        let inputs = found.inputs;
        let first_input = *inputs.first()?;

        if !is_eligible_compute(
            tree,
            &inputs,
            &inputs
                .iter()
                .map(|input| input.src_lds)
                .collect::<Vec<LdsIdx>>(),
        ) {
            return None;
        }
        let lds_type = tree.ds_type(first_input.src_lds);
        let lds_output_idx = output.src_lds;

        // `dstVias_` split by whether the destination's storage is the first input's SOURCE storage;
        // the LAST match wins, and a default-constructed `DataInfo` is what no match at all leaves.
        let src_storage = TransferWalk::transfer(tree, first_input.node).src.storage;
        let mut final_data_info = DataInfo::default();
        let mut rest_of_final_data_info = Vec::new();
        for dst in TransferWalk::transfer(tree, output.node).dsts.iter() {
            if dst.storage == src_storage {
                final_data_info = dst.data;
            } else {
                rest_of_final_data_info.push(dst.data);
            }
        }

        // `memOrg_.at(LX).allocateNode_` on each side — that `.at()`'s throw and its null.
        let allocate_in = inputs
            .iter()
            .map(|input| tree.allocation_in(DataOrigin::LabeledDs(input.src_lds), DdcMemory::Lx))
            .collect::<Option<Vec<AllocId>>>()?;
        let allocate_out =
            tree.allocation_in(DataOrigin::LabeledDs(lds_output_idx), DdcMemory::Lx)?;

        let primary = tree.primary_ds_info(lds_type)?;
        let stick_dims = primary.stick.0.clone();
        let stick_dim_key = DimSet::of(
            &stick_dims
                .iter()
                .map(|&(dim, _)| dim)
                .collect::<Vec<PrimaryDim>>(),
        )?;
        let stick_size = stick_dims
            .iter()
            .fold(1u64, |acc, &(_, size)| acc.saturating_mul(size.0));

        if inputs.len() > 1 {
            let owner = tree.owner_loop(first_input.node);
            if inputs
                .iter()
                .any(|input| tree.owner_loop(input.node) != owner)
            {
                return None;
            }
        }

        let is_stick = |dim: PrimaryDim| stick_dims.iter().any(|&(held, _)| held == dim);
        let lender = tree
            .buffer_capacity_per_dim(first_input.node, first_input.src_lds, src_storage)
            .into_iter()
            .map(|(dim, _)| dim)
            .find(|dim| !is_stick(*dim))?;
        // `getMutableParentDimLoop` hands back a `nullptr` the reference then dereferences.
        let parent_loop = tree.parent_dim_loop(compute, lender)?;

        // ⭐ THE ZIPPED STICK DIMS CLOSE `stickSizes.at(stickDimsIdx)`: the reference computes that
        // index off `stickDimOrder_` and reads it out of `stickSize_`, two vectors of independent
        // length, guarded only by `stickDimsIdx < stickDims.size()`.
        let mut non_stick: Vec<(PrimaryDim, Elements)> = Vec::new();
        for (dim, size) in
            tree.buffer_capacity_per_dim(parent_loop.0, first_input.src_lds, src_storage)
        {
            let Some(size) = size else {
                continue;
            };
            match stick_dims.iter().find(|&&(held, _)| held == dim) {
                Some(&(_, stick)) => {
                    let non = if stick.0 > 0 { size.0 / stick.0 } else { size.0 };
                    if non <= 1 {
                        continue;
                    }
                    non_stick.push((dim, Elements(non)));
                }
                None => non_stick.push((dim, size)),
            }
        }

        let parent_num = tree.loop_num(parent_loop);
        let chunk_reference = stages.0.get(&parent_num)?.clone();
        let bot_reference = stages.0.get(&tree.loop_den(parent_loop))?.clone();
        let corelet_split = &chunk_reference.ss.dims;
        let parent_dims = ScheduleSurgery::loop_dims(tree, parent_loop);

        let input_scale = tree.scale(first_input.src_lds);
        let output_scale = tree.scale(lds_output_idx);
        let output_layout = tree.primary_ds_info(tree.ds_type(lds_output_idx))?.layout;

        let &(first_stick_dim, _) = stick_dims.first()?;
        let mut new_stick_dims = vec![(first_stick_dim, Elements(1))];
        let mut remaining = stick_size;
        for &(dim, size) in &non_stick {
            // ⛔ TRAP: `PrimaryDimAndKind::operator==` compares BOTH halves and its converting
            // constructor defaults the kind to `Unpadded` (`dsc/dims.h:75-81`, `dsc/dims.cpp:832`),
            // so this `std::find` (`:9106`) SKIPS any parent-loop dim of another kind.
            if !parent_dims.iter().any(|entry| {
                entry
                    == PrimaryDimAndKind {
                        dim,
                        kind: MetaDimKind::Unpadded,
                    }
            }) {
                continue;
            }
            if remaining == 1 {
                break;
            }
            let (Some(scale_in), Some(scale_out)) = (
                dim_scale(&primary.layout, &input_scale, dim),
                dim_scale(&output_layout, &output_scale, dim),
            ) else {
                continue;
            };
            // `scaleDimInput < 0` is exactly "not a sized scale": the two negatives are sentinels.
            let (Scale::Sized(in_scale), Scale::Sized(out_scale)) = (scale_in, scale_out) else {
                continue;
            };
            if in_scale != out_scale {
                continue;
            }
            // ⛔ DELIBERATE DIVERGENCE: `coreletSplit_.at(dim).at(1)` throws for a split with one
            // side; here that dim keeps its own size, which is the no-split answer.
            let mut size_with_split = size.0;
            if corelet_split.states(DimSplit::Corelet, dim) {
                let sides = corelet_split.split_sizes(DimSplit::Corelet, dim);
                if let (Some(first), Some(second)) = (sides.first(), sides.get(1)) {
                    size_with_split = gcd(first.0, second.0);
                }
            }
            let largest = gcd(size_with_split, remaining);
            if largest == 1 {
                continue;
            }
            new_stick_dims.push((dim, Elements(largest)));
            if remaining > largest {
                remaining /= largest;
            } else {
                remaining = 1;
            }
        }
        // "There need to be some compression to continue" (`:9140`).
        if remaining == stick_size {
            return None;
        }
        if remaining != 1 {
            new_stick_dims[0].1 = Elements(remaining);
        }
        Some(Self {
            compute,
            inputs,
            output,
            lds_output_idx,
            final_data_info,
            rest_of_final_data_info,
            allocate_in,
            allocate_out,
            layout: primary.layout,
            stick_dims,
            stick_dim_key,
            new_stick_dims,
            parent_loop,
            parent_dims,
            bot_reference,
            chunk_reference,
        })
    }
}

/// Replaces: e300_packStickDim
///
/// REPACKS ONE COMPUTE'S STICK OVER A LENDER DIM (`:811`): a maximizing bottom datastage pins the OLD
/// stick layout, `loop0`/`loop1` bracket the parent band and each carries a perfectly nested split
/// loop, a chunk-stage copy with its PE/SFP split cleared carries the NEW stick sizes, a
/// `cond0_new_stick_layout` first-iteration condition holds the original body, and an LX→SFP→LX
/// compress triple per input plus one expansion triple write through fresh `Internal_compressed_*`
/// labelled data structures whose cloned allocations take the gapped-stick spread and a per-input
/// stick offset — then every transfer it touched is unrolled by entry 251.
pub fn pack_stick_dim<A, S, D>(
    tree: &mut S,
    metadata: &mut Metadata,
    stages: &mut DataStages<D>,
    site: PackStickDimSite<D>,
) -> bool
where
    A: Arch,
    S: PackStickDim + ?Sized,
    D: Clone + Default + UtilStageExtents,
{
    let PackStickDimSite {
        compute,
        inputs,
        output,
        lds_output_idx,
        final_data_info,
        rest_of_final_data_info,
        allocate_in,
        allocate_out,
        layout,
        stick_dims,
        stick_dim_key,
        new_stick_dims,
        parent_loop,
        parent_dims,
        bot_reference,
        chunk_reference,
    } = site;
    let is_stick = |dim: PrimaryDim| stick_dims.iter().any(|&(held, _)| held == dim);
    let sized = |size: Elements| [size.0 as f32];

    // "force newBotDs size to be what ever in original stick layout and other dim sizes are 1".
    let bot = construct_datastage_from(stages, &bot_reference);
    {
        let bot_ds = metadata.datastages.entry(bot).or_default();
        bot_ds.strategy = Strategy::Maximize;
        let old_sizes: Vec<f32> = stick_dims.iter().map(|&(_, size)| size.0 as f32).collect();
        force_unpadded(
            bot_ds.constraint_mut(None, Some(stick_dim_key.clone())),
            &old_sizes,
        );
        for entry in parent_dims.iter() {
            if stick_dim_key.contains(entry.dim) {
                continue;
            }
            force_unpadded(
                bot_ds.constraint_mut(None, Some(DimSet::single(entry.dim))),
                &[1.0],
            );
        }
    }

    let parent_num = tree.loop_num(parent_loop);
    let loop0 = tree.new_loop(construct_loop_node(parent_num, bot, parent_dims.clone()));
    // Freshly minted, so internal; `parent_num` is the stage the site already read; and the
    // exploration flag was proved Open before anything was minted.
    let Some(split) = DatastageSplitSite::of(
        &*tree,
        metadata,
        stages,
        loop0,
        DatastageExploration::Open,
    ) else {
        return false;
    };
    let inner_loop0 = split_loop_band_on_datastage(tree, stages, split);
    let loop1 = tree.clone_loop(loop0);
    let inner_loop1 = tree.clone_loop(inner_loop0);
    tree.add_child_node(inner_loop1.0, InsertionPoint::LastIn(loop1.0));

    // "ds_packed size to be what ever in newsticklayout except dims in original stick layout".
    let split_stage = tree.loop_den(loop0);
    {
        let split_ds = metadata.datastages.entry(split_stage).or_default();
        split_ds.strategy = Strategy::Maximize;
        for entry in parent_dims.iter() {
            if is_stick(entry.dim) {
                continue;
            }
            match new_stick_dims.iter().find(|&&(held, _)| held == entry.dim) {
                Some(&(dim, size)) => force_unpadded(
                    split_ds.constraint_mut(None, Some(DimSet::single(dim))),
                    &sized(size),
                ),
                None => force_unpadded(
                    split_ds.constraint_mut(None, Some(DimSet::single(entry.dim))),
                    &[1.0],
                ),
            }
        }
    }

    tree.add_child_node(loop0.0, InsertionPoint::Before(parent_loop.0));
    tree.add_child_node(loop1.0, InsertionPoint::After(parent_loop.0));

    // ⚠️ The reference writes `dataStageParam_[dataStageParam_.size()]` directly, which OVERWRITES on
    // a gapped map; entry 113 is the same copy under the first free id.
    let chunk = construct_datastage_from(stages, &chunk_reference);
    if let Some(stage) = stages.0.get_mut(&chunk) {
        stage.ss.dims.clear_split(DimSplit::PeSfp);
        stage.el.dims.clear_split(DimSplit::PeSfp);
    }
    let parent_den = tree.loop_den(parent_loop);
    for &(dim, size) in &new_stick_dims {
        if is_stick(dim) {
            force_unpadded(
                metadata
                    .datastages
                    .entry(chunk)
                    .or_default()
                    .constraint_mut(None, Some(DimSet::single(dim))),
                &sized(size),
            );
        }
        force_unpadded(
            metadata
                .datastages
                .entry(parent_den)
                .or_default()
                .constraint_mut(None, Some(DimSet::single(dim))),
            &sized(size),
        );
    }

    let old_num = tree.loop_num(parent_loop);
    tree.set_loop_num(parent_loop, chunk);
    let new_parent_loop = tree.new_loop(construct_loop_node(old_num, chunk, parent_dims.clone()));
    tree.add_child_node(new_parent_loop.0, InsertionPoint::After(loop0.0));

    let cond_name = NodeName("cond0_new_stick_layout".to_owned());
    let condition = tree.new_condition(cond_name.clone(), CoreClSet::default());
    let then_branch = tree.new_block(NodeName(format!("{}_then_region", cond_name.0)));
    tree.add_then_region(condition, then_branch);
    let and_clause: Vec<LoopCond> = new_stick_dims
        .iter()
        .filter(|&&(dim, _)| is_stick(dim))
        .map(|&(dim, _)| LoopCond {
            loop_node: new_parent_loop.0,
            dim,
            op: CondOp::Eq,
            against: CondValType::First,
        })
        .collect();
    tree.set_loop_cond(
        condition,
        LoopCondComposite {
            or_of_ands: vec![and_clause],
            negated: false,
        },
    );
    tree.add_child_node(condition, InsertionPoint::LastIn(new_parent_loop.0));
    tree.move_node(parent_loop.0, InsertionPoint::LastIn(then_branch));

    tree.set_internal_layout_order(layout);
    for &(dim, size) in &new_stick_dims {
        tree.push_internal_stick_dim(dim, size, StickRepl(1));
    }

    // ⛔ TRAP, AND IT IS THE REFERENCE'S: `ldsOutputIdx++` is UNCONDITIONAL (`:9264`, `:9269`), so it
    // tracks the shift entry 257 applies to the last `labeledDs_` entry only while the output data
    // structure IS that last entry.
    let mut new_lds_in: Vec<LdsIdx> = Vec::new();
    let mut lds_output = lds_output_idx;
    for input in &inputs {
        let reference = tree.last_lds_entry();
        let minted = add_new_lds(tree, metadata, &reference);
        new_lds_in.push(minted);
        lds_output = LdsIdx(lds_output.0 + 1);
        metadata
            .interm_lds_idx_to_ext_lds
            .insert(minted, input.src_lds);
    }
    let reference = tree.last_lds_entry();
    let new_lds_out = add_new_lds(tree, metadata, &reference);
    lds_output = LdsIdx(lds_output.0 + 1);
    metadata
        .interm_lds_idx_to_ext_lds
        .insert(new_lds_out, lds_output);

    for (input, &minted) in inputs.iter().zip(new_lds_in.iter()) {
        tree.update_nodes_with_new_lds(minted, input.src_lds, parent_loop.0);
    }
    tree.update_nodes_with_new_lds(new_lds_out, lds_output, parent_loop.0);

    for (idx, &minted) in new_lds_in.iter().enumerate() {
        tree.set_lds_name(
            minted,
            StorageName(format!("Internal_compressed_input{idx}")),
        );
        tree.set_lds_type(minted, DsType::Internal);
    }
    tree.set_lds_name(
        new_lds_out,
        StorageName("Internal_compressed_output".to_owned()),
    );
    tree.set_lds_type(new_lds_out, DsType::Internal);

    let mut intr_in: Vec<AllocId> = Vec::new();
    for &minted in &new_lds_in {
        let alloc = tree.clone_allocate_after(allocate_out);
        let component = tree.alloc_component(alloc);
        tree.set_allocation_absent(minted, component, alloc);
        tree.set_alloc_lds_idx(alloc, minted);
        intr_in.push(alloc);
    }
    let intr_out = tree.clone_allocate_after(allocate_out);
    let out_component = tree.alloc_component(intr_out);
    tree.set_allocation_absent(new_lds_out, out_component, intr_out);
    tree.set_alloc_lds_idx(intr_out, new_lds_out);

    for &(dim, size) in &new_stick_dims {
        if is_stick(dim) {
            continue;
        }
        let spread = StickSpread(i32::try_from(size.0).unwrap_or(i32::MAX));
        for &alloc in &intr_in {
            OffsetAdjustment::set_gap_stick_spread(tree, alloc, dim, spread);
        }
        OffsetAdjustment::set_gap_stick_spread(tree, intr_out, dim, spread);
        // ⛔ DELIBERATE DIVERGENCE: `opaqueOps_.at(compute).internalRegAlloc_` throws for a compute
        // the map does not carry and dereferences a null allocation; both are skipped here.
        if let Some(internal) = metadata
            .opaque_ops
            .get(&compute)
            .and_then(|op| op.internal_reg_alloc)
        {
            OffsetAdjustment::set_gap_stick_spread(tree, internal, dim, spread);
        }
    }

    let data_info_const = DataInfo::default();
    let mut data_info = DataInfo::default();
    for (idx, input) in inputs.iter().enumerate() {
        let which = InputIdx(idx);
        data_info = TransferWalk::transfer(&*tree, input.node).src.data;
        data_info.my_lds_idx = Some(input.src_lds);

        let from_lx_src = Operand {
            unit: SenComponent::Lxlu,
            storage: SenComponent::Lx,
            data: data_info,
        };
        data_info.data_connect =
            Some(tree.intern_connect(MintedConnect::SfpCompressInputComp(which)));
        let from_lx = tree.new_transfer(minted_transfer(
            NodeName(format!("lx_sfp_compress_comp{idx}")),
            from_lx_src,
            Operand {
                unit: SenComponent::Sfp,
                storage: SenComponent::Sfp,
                data: data_info,
            },
        ));
        FifoResults::add_alloc_user(tree, allocate_in[idx], from_lx);
        tree.remove_alloc_user(allocate_in[idx], input.node);

        // `inputs_.push_back(LXLU|ONE|ZERO)` is the UNIT vector; a compute operand's storage is its
        // own unit, which is the one `SenComponents` the reference states per input.
        let fma_input = data_info;
        data_info.my_lds_idx = Some(new_lds_in[idx]);
        data_info.data_connect =
            Some(tree.intern_connect(MintedConnect::SfpCompressOutputComp(which)));
        let fma = tree.new_compute(dummy_fma(
            NodeName(format!("sfp_dummy_fma_comp{idx}")),
            fma_input,
            data_info_const,
            data_info,
        ));

        let to_lx_src = Operand {
            unit: SenComponent::Sfp,
            storage: SenComponent::Sfp,
            data: data_info,
        };
        data_info.data_connect =
            Some(tree.intern_connect(MintedConnect::LxCompressInputComp(which)));
        let to_lx = tree.new_transfer(minted_transfer(
            NodeName(format!("sfp_lx_compress_comp{idx}")),
            to_lx_src,
            Operand {
                unit: SenComponent::Lxsu,
                storage: SenComponent::Lx,
                data: data_info,
            },
        ));
        tree.clear_alloc_users(intr_in[idx]);
        FifoResults::add_alloc_user(tree, intr_in[idx], to_lx);
        FifoResults::add_alloc_user(tree, intr_in[idx], input.node);

        // The original transfer now reads what the compress triple wrote.
        let mut original = TransferWalk::transfer(&*tree, input.node);
        original.src.data.data_connect = data_info.data_connect;
        tree.set_transfer(input.node, original);

        tree.add_child_node(from_lx, InsertionPoint::LastIn(inner_loop0.0));
        tree.add_child_node(fma, InsertionPoint::LastIn(inner_loop0.0));
        tree.add_child_node(to_lx, InsertionPoint::LastIn(inner_loop0.0));
    }

    let (send0, recv0) = minted_sync_pair(tree);
    tree.add_child_node(send0, InsertionPoint::Before(new_parent_loop.0));
    tree.add_child_node(recv0, InsertionPoint::Before(new_parent_loop.0));

    // ⚠️ `data_info` is the LAST input's source data info, with only its index and connect
    // overwritten — the reference declares it once, outside the loop above.
    data_info.my_lds_idx = Some(new_lds_out);
    data_info.data_connect = Some(tree.intern_connect(MintedConnect::LxCompressOutputExp));
    let exp_src = Operand {
        unit: SenComponent::Lxlu,
        storage: SenComponent::Lx,
        data: data_info,
    };
    let exp_connect = tree.intern_connect(MintedConnect::SfpCompressOutputExp);
    data_info.data_connect = Some(exp_connect);
    let from_lx_exp = tree.new_transfer(minted_transfer(
        NodeName("lx_sfp_compress_exp".to_owned()),
        exp_src,
        Operand {
            unit: SenComponent::Sfp,
            storage: SenComponent::Sfp,
            data: data_info,
        },
    ));
    tree.clear_alloc_users(intr_out);
    FifoResults::add_alloc_user(tree, intr_out, from_lx_exp);
    FifoResults::add_alloc_user(tree, intr_out, output.node);

    let fma_input = data_info;
    data_info.my_lds_idx = Some(lds_output);
    data_info.data_connect = Some(exp_connect);
    let fma_exp = tree.new_compute(dummy_fma(
        NodeName("sfp_dummy_fma_exp".to_owned()),
        fma_input,
        data_info_const,
        data_info,
    ));

    // ⛔ DELIBERATE DIVERGENCE: the reference pushes `releventFinalDataInfo` AND every
    // `restOfFinalDataInfo` onto `dstLdsAndLoopOffsets_` while `dstVias_` holds ONE entry, so those
    // extra offsets are unreachable through any via-indexed reader. [`Dsts`] pairs the two, and
    // minting a via for them would ADD destinations the reference does not have.
    let mut carried_final = final_data_info;
    carried_final.my_lds_idx = data_info.my_lds_idx;
    let _ = &rest_of_final_data_info;
    let to_lx_exp = tree.new_transfer(minted_transfer(
        NodeName("sfp_lx_compress_exp".to_owned()),
        Operand {
            unit: SenComponent::Sfp,
            storage: SenComponent::Sfp,
            data: data_info,
        },
        Operand {
            unit: SenComponent::Lxsu,
            storage: SenComponent::Lx,
            data: carried_final,
        },
    ));
    FifoResults::add_alloc_user(tree, allocate_out, to_lx_exp);
    tree.remove_alloc_user(allocate_out, output.node);

    // ⛔ THE SAME DIVERGENCE ON THE OTHER SIDE: `dstLdsAndLoopOffsets_.clear()` followed by ONE push
    // leaves `dstVias_` longer than the offsets, so the faithful projection is the FIRST via with the
    // new data and the now-unpaired destinations dropped.
    let mut original_out = TransferWalk::transfer(&*tree, output.node);
    original_out.dsts = Dsts::new(
        Operand {
            unit: original_out.dsts.first().unit,
            storage: original_out.dsts.first().storage,
            data: exp_src.data,
        },
        Vec::new(),
    );
    tree.set_transfer(output.node, original_out);

    tree.add_child_node(from_lx_exp, InsertionPoint::LastIn(inner_loop1.0));
    tree.add_child_node(fma_exp, InsertionPoint::LastIn(inner_loop1.0));
    tree.add_child_node(to_lx_exp, InsertionPoint::LastIn(inner_loop1.0));

    let (send1, recv1) = minted_sync_pair(tree);
    tree.add_child_node(send1, InsertionPoint::After(new_parent_loop.0));
    tree.add_child_node(recv1, InsertionPoint::After(new_parent_loop.0));

    // "fix the address of internal tensors" — the capacity of the OUTPUT allocation, each stick dim
    // divided down to whole sticks. ⛔ DELIBERATE DIVERGENCE: an allocation naming no labelled data
    // structure is `getBufferCapacityForNodePerDim`'s throw; here it contributes no per-dim capacity.
    let out_alloc_node = tree.alloc_node(allocate_out);
    let out_storage = memory_component(tree.alloc_component(allocate_out));
    let capacity = match tree.alloc_lds_idx(allocate_out) {
        Some(lds) => tree.buffer_capacity_per_dim(out_alloc_node, lds, out_storage),
        None => Vec::new(),
    };
    let capacity_in_sticks: Vec<(PrimaryDim, Option<Elements>)> = capacity
        .into_iter()
        .map(|(dim, size)| match stick_dims.iter().find(|&&(held, _)| held == dim) {
            Some(&(_, stick)) if stick.0 > 0 => (dim, size.map(|size| Elements(size.0 / stick.0))),
            _ => (dim, size),
        })
        .collect();

    // ⛔ TRAP: the reference multiplies `offsetForDim` by a capacity of `-1` and produces a NEGATIVE
    // address. An unsized dim contributes a factor of ONE here, and the `break` on the target dim
    // still fires — a fabricated placement to avoid a stop is worse than the stop.
    let stick_bytes = i64::try_from(A::BYTES_PER_STICK.get()).unwrap_or(i64::MAX);
    let mut offsets: Vec<i64> = vec![0; inputs.len()];
    for &(dim, size) in &new_stick_dims {
        if is_stick(dim) {
            continue;
        }
        let mut offset_for_dim = stick_bytes;
        for &(held, capacity) in &capacity_in_sticks {
            if held == dim {
                break;
            }
            if let Some(capacity) = capacity {
                offset_for_dim =
                    offset_for_dim.saturating_mul(i64::try_from(capacity.0).unwrap_or(i64::MAX));
            }
        }
        for (idx, offset) in offsets.iter_mut().enumerate() {
            let steps = i64::try_from(size.0).unwrap_or(i64::MAX)
                - i64::try_from(idx + 1).unwrap_or(i64::MAX);
            *offset = offset.saturating_add(steps.saturating_mul(offset_for_dim));
        }
    }

    let cores = tree.core_ids_used();
    let corelets = tree.corelets_used();
    // ⚠️ The OUTPUT allocation is shifted by `offset.at(0)` — the FIRST input's offset, which is the
    // reference's own choice and not the output's own entry (`:9531`).
    let mut shifts: Vec<(AllocId, i64)> = intr_in
        .iter()
        .copied()
        .zip(offsets.iter().copied())
        .collect();
    shifts.push((intr_out, offsets.first().copied().unwrap_or(0)));
    for (alloc, offset) in shifts {
        for &core in &cores {
            for &corelet in &corelets {
                if let Some(placed) = tree.alloc_start_address(alloc, core, corelet) {
                    tree.set_alloc_start_address(
                        alloc,
                        core,
                        corelet,
                        Bytes(placed.0.saturating_add_signed(offset)),
                    );
                }
            }
        }
    }

    for node in inputs
        .iter()
        .map(|input| input.node)
        .chain(core::iter::once(output.node))
    {
        if let Some(unroll) = InternalNode::of(metadata, node) {
            unroll_transfer(tree, stages, metadata, unroll);
        }
    }
    true
}

/// One of entry 300's six minted transfers: a single source and a single destination, everything else
/// at the value a `new dsc2::TransferNode()` carries.
fn minted_transfer(name: NodeName, src: Operand, dst: Operand) -> TransferNode {
    TransferNode {
        name,
        src,
        dsts: Dsts::new(dst, Vec::new()),
        replication_factor: ReplicationFactor::ONE,
        unit_time_transfer_chunk_size: Vec::new(),
        unit_time_transfer_num_chunks: NumChunks::ONE,
        padding: TransferPadding::default(),
        src_indirect: None,
        dst_indirect: None,
        core_id_to_gtr_info: BTreeMap::new(),
        transfer_size: BTreeMap::new(),
    }
}

/// `FMA16` of `LXLU * ONE + ZERO` into `LXSU` on the SFP — the dummy compute that moves one
/// compressed operand across the register file.
fn dummy_fma(name: NodeName, input: DataInfo, constant: DataInfo, output: DataInfo) -> ComputeNode {
    let operand = |unit: SenComponent, data: DataInfo| Operand {
        unit,
        storage: unit,
        data,
    };
    ComputeNode {
        name,
        op: DdlComputeType::Fma16,
        ex_unit: SenComponent::Sfp,
        inputs: vec![
            operand(SenComponent::Lxlu, input),
            operand(SenComponent::One, constant),
            operand(SenComponent::Zero, constant),
        ],
        outputs: vec![operand(SenComponent::Lxsu, output)],
        num_folds_engaged: NumFolds::ONE,
        data_format: None,
        instr_attribute: InstrAttribute::default(),
    }
}

/// The `sync_lxsu_send_lxlu` / `sync_lxlu_recv_lxsu` pair, cross-linked BY NAME so a half-linked pair
/// is unspellable — entry 300 mints the same two around each half of the split.
fn minted_sync_pair<S: PackStickDim + ?Sized>(tree: &mut S) -> (NodeId, NodeId) {
    let send_name = NodeName("sync_lxsu_send_lxlu".to_owned());
    let recv_name = NodeName("sync_lxlu_recv_lxsu".to_owned());
    let send = tree.new_sync(SyncNode {
        name: send_name.clone(),
        units: SyncUnits::new(SenComponent::Lxsu, []),
        direction: SyncDirection::Send,
        strength: SyncStrength::Hard,
        implicit_sync_ref_transfer: None,
        other_ends: vec![recv_name.clone()],
    });
    let recv = tree.new_sync(SyncNode {
        name: recv_name,
        units: SyncUnits::new(SenComponent::Lxlu, []),
        direction: SyncDirection::Receive,
        strength: SyncStrength::Hard,
        implicit_sync_ref_transfer: None,
        other_ends: vec![send_name],
    });
    (send, recv)
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ THE TRANSFER-HOISTING AND TRANSFER-UNROLLING VOCABULARY — as entries 301-303 read it.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT KIND OF PARENT THE HOIST WALK REACHED — the three `nodeType_` cases entry 301 splits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoistParent {
    /// `LOOP`.
    Loop(LoopId),
    /// `CONDITION` — the walk collects the loops its condition names and keeps going.
    Condition,
    /// Every other kind, which the walk passes through.
    Other,
}

/// WHAT ENTRY 301 ASKS OF THE CARRIER — the parent kinds, the core stage's window dims, the SYNC
/// scan under a loop, and the identity a fresh allocation takes.
pub trait HoistTransfers:
    UtilTransferMoves + FifoResults + LoopBands + AllocationPaddings + DscAllocations + ScopeTree
{
    /// Which kind of node the walk is standing on.
    fn hoist_parent(&self, node: NodeId) -> HoistParent;

    /// `dataStageParam_.at(metadata.core_dstgid).ss_.paddingSizes_.at(dim).windowDim_`, absent both
    /// where the map has no entry for `dim` and for its `PrimaryDimTypesCount` — which is the
    /// reference's own *"no window dim"*. ⭐ THE NEIGHBOURING READER OF THIS MAP IS
    /// [`super::fold::CoreStage::pad_stride`].
    fn window_dim(&self, dim: PrimaryDim) -> Option<PrimaryDim>;

    /// `traverseTreeDFSMutable(base, {SYNC}, ALL, -1, -1, -1, excludeList)` reduced to each SYNC
    /// node's `units_` — the only thing entry 301 asks of them. `exclude` is the reference's
    /// one-element exclude list: the loop the previous step already scanned.
    fn sync_units_under(&self, base: LoopId, exclude: Option<LoopId>) -> Vec<SyncUnits>;

    /// The identity a freshly minted `dsc2::AllocateNode` takes — a fresh heap pointer in the
    /// reference, and the carrier's own next free id here.
    fn free_alloc_id(&self) -> AllocId;
}

/// A TRANSFER WORTH TRYING TO HOIST, carrying the components it touches and the loops that produce
/// its source.
///
/// ⛔⛔ THE `DT_CHECK` AND ONE `DT_ERROR` COLLAPSE HERE: a route through `NO_COMPONENT` or `HBM`, and
/// *"Missing source data connect metadata for transfer"*. Both mean the transfer is not a candidate
/// rather than that this pass cannot run, and the pass's own answer has nowhere to say otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoistCandidate {
    transfer: NodeId,
    /// `relatedComponents` — the source unit, every destination unit, and every via hop.
    components: BTreeSet<SenComponent>,
    /// `metadata.dataConnects_.at(srcDataConnect).getProducerLoops()`.
    producers: Vec<LoopId>,
}

impl HoistCandidate {
    /// The candidate, or [`None`] for an external transfer, a transfer of a constant, and either
    /// collapsed abort.
    #[must_use]
    pub fn of<S: HoistTransfers + ?Sized>(
        tree: &S,
        metadata: &Metadata,
        transfer: NodeId,
    ) -> Option<Self> {
        InternalNode::of(metadata, transfer)?;
        let node = ScheduleSurgery::transfer(tree, transfer);
        let mut components = BTreeSet::from([node.src.unit]);
        for (dst, hops) in node.dsts.routes() {
            components.insert(dst.unit);
            components.extend(hops.iter().copied());
        }
        if components.contains(&SenComponent::NoComponent)
            || components.contains(&SenComponent::Hbm)
            || components.contains(&SenComponent::Constant)
        {
            return None;
        }
        let producers = tree.producer_loops(node.src.data.data_connect?)?;
        Some(Self {
            transfer,
            components,
            producers,
        })
    }
}

/// Replaces: e301_hoistTransfersUpForReuse
///
/// HOISTS EVERY TRANSFER AS FAR UP ITS PARENT CHAIN AS REUSE ALLOWS: walks up from each transfer to
/// the first loop that either shares a dim with the transfer or cannot be crossed — a producer of
/// its source, a parametric loop, a loop some enclosing condition names, the owner of an external
/// destination allocation, or a loop holding a SYNC on a component the transfer touches — converts
/// any FIFO result to a register, splits that loop so only the shared dims remain around the
/// transfer, and moves the transfer there (`:1432`).
///
/// ⛔ THE FIFO CHECK IS ENTRY 250'S OWN: the reference's separate opaque-consumer scan and its
/// `convertResultFromFIFOtoReg` call both `break`, and entry 250 already answers `false` for an
/// opaque consumer before it mutates anything — so one call states both.
/// ⚠️ TRAP: `didTransformation` is set for a move entry 249 REFUSES in the reference only because
/// that refusal is an abort there; a refused move leaves the answer alone here.
pub fn hoist_transfers_up_for_reuse<S>(
    tree: &mut S,
    metadata: &mut Metadata,
    exploration: DatastageExploration,
) -> bool
where
    S: HoistTransfers + TransferWalk + ?Sized,
{
    let mut did_transformation = false;
    for transfer in TransferWalk::transfers(tree) {
        let Some(candidate) = HoistCandidate::of(&*tree, metadata, transfer) else {
            continue;
        };
        let node = ScheduleSurgery::transfer(tree, transfer);
        let owner = tree.owner_loop(transfer);

        // The dims the transfer walks, plus the window dim of each padded one.
        let mut transfer_dims = ScopeTree::non_broadcast_lds_dims(tree, node.src.data.my_lds_idx);
        let ends = tree.transfer_ends(transfer);
        let padding = get_padding_per_dim(&*tree, metadata, transfer, &ends, TransferSide::Src);
        let windows: Vec<PrimaryDim> = transfer_dims
            .iter()
            .filter(|dim| padding.padding(**dim) != PadType::NoPad)
            .filter_map(|dim| tree.window_dim(*dim))
            .collect();
        transfer_dims.extend(windows);

        let mut external_alloc_owners: BTreeSet<LoopId> = BTreeSet::new();
        for dst in node.dsts.iter() {
            let Some(alloc) = tree.destination_allocation(dst) else {
                continue;
            };
            if InternalNode::of(metadata, alloc).is_none() {
                external_alloc_owners.extend(tree.owner_loop(alloc));
            }
        }

        let mut linked_with_conditions: BTreeSet<NodeId> = BTreeSet::new();
        let mut scanned_for_syncs: Option<LoopId> = None;
        let mut current = ScheduleSurgery::parent(tree, transfer);
        while let Some(parent) = current {
            current = ScheduleSurgery::parent(tree, parent);
            let loop_node = match tree.hoist_parent(parent) {
                HoistParent::Condition => {
                    let name = tree.node_name(parent);
                    let cond = tree.loop_cond(parent);
                    collect_loop_references(
                        UtilNode::Condition(&name, &cond),
                        &mut linked_with_conditions,
                    );
                    continue;
                }
                HoistParent::Other => continue,
                HoistParent::Loop(loop_node) => loop_node,
            };

            let mut cannot_cross = candidate.producers.contains(&loop_node)
                || tree.is_parametric(loop_node)
                || linked_with_conditions.contains(&loop_node.0)
                || external_alloc_owners.contains(&loop_node);
            if owner == Some(loop_node) && cannot_cross {
                break;
            }
            if !cannot_cross {
                cannot_cross = tree
                    .sync_units_under(loop_node, scanned_for_syncs)
                    .iter()
                    .flat_map(SyncUnits::iter)
                    .any(|unit| candidate.components.contains(&unit));
            }

            let dims = tree.loop_dims(loop_node);
            let overlap: Vec<PrimaryDimAndKind> = dims
                .iter()
                .filter(|dim| transfer_dims.contains(&dim.dim))
                .collect();
            if overlap.is_empty() && !cannot_cross {
                // Nothing ties the transfer to this loop: keep going up, and do not scan this
                // loop's SYNC nodes again from the next one.
                scanned_for_syncs = Some(loop_node);
                continue;
            }

            if owner == Some(loop_node) && dims.iter().count() == overlap.len() {
                // No opportunity to split the transfer's own owner loop.
                break;
            }

            // `getNonMemoryResultIndex() != -1`.
            if node.dsts.iter().any(|dst| !is_memory(dst.storage)) {
                let fresh = tree.free_alloc_id();
                let Some(site) = FifoConversionSite::of(metadata, transfer, exploration) else {
                    break;
                };
                if !convert_result_from_fifo_to_reg(tree, metadata, site, fresh) {
                    break;
                }
            }

            let mut effective_parent = loop_node;
            if !cannot_cross && dims.iter().count() > overlap.len() {
                // The loop also walks dims the transfer does not: split those off below it.
                if let Some((first, rest)) = overlap.split_first() {
                    let set = LoopDims::new(*first, rest.to_vec());
                    if let Some(split) = LoopBandSplit::of(
                        &*tree,
                        metadata,
                        loop_node,
                        &[set],
                        UnspecifiedDims::Innermost,
                    ) {
                        let innermost = split_loop_band_on_dim(tree, split);
                        effective_parent = tree.owner_loop(innermost.0).unwrap_or(loop_node);
                    }
                }
            }

            if let Some(move_) =
                TransferMove::of(&*tree, metadata, transfer, effective_parent, exploration)
            {
                move_transfer_node(tree, move_);
                did_transformation = true;
            }
            break;
        }
    }
    did_transformation
}

/// WHAT ENTRY 302 ASKS OF THE CARRIER BESIDES UNROLLING.
pub trait SymbolicTransfers: TransferUnrolling {
    /// `labeledDs_.at(lds).memOrg_`'s FIRST entry with an `allocateNode_`, absent where the entry
    /// has none — the reference's reference-tensor scan.
    fn first_allocation_of(&self, lds: LdsIdx) -> Option<AllocId>;

    /// `getSizeDataStageForNode(transfer, reference).ss_.symbolicDimInfo_`
    /// (`dsc/designSpaceConfig.h:264`) — which of the transfer's dims are symbolic, and how.
    fn symbolic_dims(
        &self,
        transfer: NodeId,
        reference: AllocId,
    ) -> BTreeMap<PrimaryDim, SymbolicDimInfo>;
}

/// Replaces: e302_unrollSymbolicTransfers
///
/// FULLY UNROLLS EVERY INTERNAL TRANSFER OVER ITS SYMBOLIC DIMS, sizing each by the datastage of the
/// first allocation its destinations reach, falling back to its source's (`:1663`).
///
/// ⛔ DELIBERATE DIVERGENCE: *"Unable to unroll transfer with symbolic dimensions, but that is needed
/// for functionality"* is an abort in the reference; here it is the `false` answer and the walk
/// stops. ⚠️ Every callsite of this pass DISCARDS its `bool` (`ddc/ddcv1.cpp:3777`), so that answer
/// is where the condition is recorded and not where it is acted on.
/// ⚠️ TRAP: a transfer whose ends reach no allocation is SKIPPED — the reference's *"most likely a
/// transfer of constant"* and its fifo-to-fifo case are the same `continue`.
pub fn unroll_symbolic_transfers<S, D>(
    tree: &mut S,
    stages: &mut DataStages<D>,
    metadata: &mut Metadata,
) -> bool
where
    S: SymbolicTransfers + TransferWalk + ?Sized,
    D: Default + Clone + UtilStageExtents,
{
    for node in TransferWalk::transfers(tree) {
        let Some(site) = InternalNode::of(metadata, node) else {
            continue;
        };
        let transfer = TransferWalk::transfer(tree, node);
        let reference = transfer
            .dsts
            .iter()
            .filter_map(|dst| dst.data.my_lds_idx)
            .find_map(|lds| tree.first_allocation_of(lds))
            .or_else(|| {
                transfer
                    .src
                    .data
                    .my_lds_idx
                    .and_then(|lds| tree.first_allocation_of(lds))
            });
        let Some(reference) = reference else {
            continue;
        };
        let symbolic = tree.symbolic_dims(node, reference);
        if !unroll_transfer_for_symbolic_dims(tree, stages, site, &symbolic) {
            return false;
        }
    }
    true
}

/// WHAT ENTRY 303 ASKS OF AN ALLOCATION.
pub trait SpreadTransfers: TransferUnrolling + FifoResults + Allocations {
    /// `allocNode->gapStickSpread_.empty()` (`dsc/dsc2.h:995`) negated — the ONLY thing entry 303
    /// asks of that map, which entry 300 is what writes.
    fn has_gap_stick_spread(&self, alloc: AllocId) -> bool;
}

/// Replaces: e303_unrollSpreadTransfers
///
/// FULLY UNROLLS EVERY TRANSFER EITHER OF WHOSE ENDS ALLOCATES WITH A STICK GAP, since a spread
/// allocation cannot be walked by a single strided transfer (`:1706`).
///
/// ⛔ UNLIKE ENTRY 302 THIS WALK HAS NO EXTERNAL-NODE SKIP, so a spread EXTERNAL transfer reaches
/// `unrollTransfer`, whose first act is to abort on exactly that; here it is skipped instead.
/// ⚠️ The `unrollTransfer` answer is DISCARDED in the reference too, and so is this pass's own.
pub fn unroll_spread_transfers<S, D>(
    tree: &mut S,
    stages: &mut DataStages<D>,
    metadata: &mut Metadata,
) -> bool
where
    S: SpreadTransfers + TransferWalk + ?Sized,
    D: Default,
{
    for node in TransferWalk::transfers(tree) {
        let ends = tree.transfer_ends(node);
        let spread = core::iter::once(ends.src).chain(ends.dsts).any(|end| {
            tree.allocation(end)
                .is_some_and(|alloc| tree.has_gap_stick_spread(alloc))
        });
        if !spread {
            continue;
        }
        if let Some(site) = InternalNode::of(metadata, node) {
            unroll_transfer(tree, stages, metadata, site);
        }
    }
    true
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐ TYPES FOR ENTRY 338. Union this section with this file's other vocabulary when its remaining
// entries land.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHETHER THE ALLOCATE CLONE SKIPS ITS METADATA UPDATE — entry 118's `skipMetadataUpdate`.
///
/// ⚠️ TRAP, AND IT IS THE REFERENCE'S: `:1418` passes the POINTER
/// `allocNode->tempStorageForCompute_` (`dsc/dsc2.h:978`) into a parameter DECLARED `bool`, so the
/// flag says only *"this allocation is some compute's temp storage"*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipMetadataUpdate {
    /// `tempStorageForCompute_ != nullptr`.
    Yes,
    /// `tempStorageForCompute_ == nullptr`.
    No,
}

/// ONE NODE ENTRY 338 HANDS TO A CLONE — the three `nodeType_` arms of its ONE `{ALLOCATE, TRANSFER,
/// COMPUTE}` traversal.
///
/// ⛔ ONE WALK, INTERLEAVED IN TREE ORDER and NOT three walks grouped by kind: a transfer's clone is
/// minted before a later allocate's is, which is what the reference's own *"TO VERIFY: Can unrolling
/// invalidate the traversal result"* (`:1422`) is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeSfpSplitNode {
    /// `ALLOCATE`, with its `tempStorageForCompute_` read as entry 118's flag.
    Allocate(NodeId, SkipMetadataUpdate),
    /// `TRANSFER`.
    Transfer(NodeId),
    /// `COMPUTE`.
    Compute(NodeId),
}

/// WHAT ENTRY 338 ASKS OF THE DSC AND ITS TREE.
pub trait PeSfpWorkSplit: NodeCloning {
    /// `currDsc->dataStageParam_.at(metadata.core_dstgid).ss_.peSfpSplit_.empty()` negated — the
    /// WHOLE of the input SDSC's worksplit information as far as this pass reads it.
    fn has_pe_sfp_split(&self) -> bool;

    /// `scheduleTree_.traverseTreeDFSMutable(nullptr, {ALLOCATE, TRANSFER, COMPUTE}, ALL, -1, -1)`.
    fn split_candidates(&self) -> Vec<PeSfpSplitNode>;
}

/// Replaces: e338_performPeSfpWorkSplit
///
/// CLONES THE SCHEDULE ONTO THE OTHER HALF OF THE PE/SFP PAIR (`:1395`): one DFS handing every
/// allocate, transfer and compute to its own `cloneForPeSfpWorkSplit`.
///
/// ⛔ `true` AND NOTHING DONE where the input SDSC states no worksplit; the report beside that answer
/// is a `std::cerr` line and this pass has no other answer.
/// ⚠️ EVERY CLONE'S ANSWER IS DISCARDED, the transfer arm's `nullptr` included, so a transfer that
/// cannot be unrolled or was already cloned is silently left alone.
/// ⛔ AN ALLOCATE NODE THAT IS NOT AN ALLOCATION IS LEFT ALONE, which is the one refusal this pass
/// adds to the three clones' own: the walk names schedule nodes and entry 118 takes the allocation.
pub fn perform_pe_sfp_work_split<S, D>(
    tree: &mut S,
    stages: &mut DataStages<D>,
    metadata: &mut Metadata,
) -> bool
where
    S: PeSfpWorkSplit
        + FifoResults
        + Allocations
        + TransferUnrolling
        + MintedConnects
        + AllocateCloning
        + ComponentAllocations
        + ComputeCloning
        + ?Sized,
    D: Default,
{
    if !tree.has_pe_sfp_split() {
        return true;
    }
    for candidate in tree.split_candidates() {
        match candidate {
            PeSfpSplitNode::Allocate(node, skip) => {
                if let Some(alloc) = tree.allocation_of(node) {
                    if let Some(split) = PeSfpAllocateSplit::of(tree, metadata, alloc) {
                        let _ = clone_allocate_for_pe_sfp_work_split(tree, metadata, split, skip);
                    }
                }
            }
            PeSfpSplitNode::Transfer(node) => {
                let body = ScheduleSurgery::transfer(tree, node);
                if let Some(split) = PeSfpTransferSplit::of(metadata, node, &body) {
                    let _ = clone_transfer_for_pe_sfp_work_split(tree, stages, metadata, split);
                }
            }
            PeSfpSplitNode::Compute(node) => {
                let body = ComputeCloning::compute(tree, node);
                if let Some(split) = PeSfpComputeSplit::of(metadata, node, &body) {
                    let _ = clone_compute_for_pe_sfp_work_split(tree, metadata, split);
                }
            }
        }
    }
    true
}

// crustify:todo: e359_transformRegToFifoOrLatch
//   authority : ddc/ddc_transformation.cpp:610  (82 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::transformRegToFifoOrLatch()
//   extract   : crustify-ddc/cpp/ddc.cpp:14093-14175
//   calls     : e117_getNodeDescription, e242_canUseFifo, e243_canUseLatch, e254_destRelatedToExternalNodes, e305_destRelatedToExternalNodes, e339_convertResultToSkipReg

// crustify:todo: e360_transformFor4BsplatRead
//   authority : ddc/ddc_transformation.cpp:1766  (87 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::transformFor4BsplatRead()
//   extract   : crustify-ddc/cpp/ddc.cpp:14185-14272
//   calls     : e250_convertResultFromFIFOtoReg, e340_insertComputeBetweenTransferAndReg

// crustify:todo: e376_performAutomaticShuffling
//   authority : ddc/ddc_transformation.cpp:1855  (167 body lines, level 6)
//   class     : Ddc
//   original  : bool Ddc::performAutomaticShuffling()
//   extract   : crustify-ddc/cpp/ddc.cpp:14971-15138
//   calls     : e110_constructAllocation, e164_insert_before, e257_addNewLds, e371_replace_assign

#[cfg(test)]
mod tests_e105_e109 {
    use super::*;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        DataInfo, Dsts, InstrAttribute, NumChunks, ReplicationFactor, TransferPadding,
    };
    use crate::units::NumFolds;
    use std::collections::BTreeMap;

    /// A tree of computes, in traversal order.
    struct Computes(Vec<ComputeOp>);

    impl ComputeWalk for Computes {
        fn computes(&self) -> Vec<NodeId> {
            self.0
                .iter()
                .zip(0u32..)
                .map(|(_, idx)| NodeId(idx))
                .collect()
        }
        fn compute_op(&self, node: NodeId) -> ComputeOp {
            self.0
                .get(node.0 as usize)
                .copied()
                .unwrap_or(ComputeOp::Other)
        }
    }

    /// A tree of transfers, in traversal order.
    struct Transfers(Vec<TransferNode>);

    impl TransferWalk for Transfers {
        fn transfers(&self) -> Vec<NodeId> {
            self.0
                .iter()
                .zip(0u32..)
                .map(|(_, idx)| NodeId(idx))
                .collect()
        }
        fn transfer(&self, node: NodeId) -> TransferNode {
            self.0[node.0 as usize].clone()
        }
    }

    /// `labeledDs_`, indexed by [`LdsIdx`].
    struct Labeled(Vec<(DsType, Vec<Scale>)>);

    impl LabeledDs for Labeled {
        fn scale(&self, lds: LdsIdx) -> Vec<Scale> {
            self.0[lds.0 as usize].1.clone()
        }
        fn ds_type(&self, lds: LdsIdx) -> DsType {
            self.0[lds.0 as usize].0
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
            },
        }
    }

    fn compute_on(ex_unit: SenComponent, connect: DataConnect) -> ComputeNode {
        ComputeNode {
            name: NodeName("recip".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit,
            inputs: vec![operand(ex_unit, Some(connect), Some(0))],
            outputs: vec![operand(ex_unit, Some(connect), Some(0))],
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        }
    }

    #[test]
    fn a_layernormscale_latches_two_inputs_onto_a_later_reciprocal() {
        let tree = Computes(vec![
            ComputeOp::Other,
            ComputeOp::LayernormScale,
            ComputeOp::Reciprocal,
        ]);
        // The LAST match wins the node, but `numberOfInputs` never falls back to 1.
        assert_eq!(
            is_relevant_compute_to_pack_stick_dim(&tree),
            Some((NodeId(2), InputCount(2)))
        );
        assert_eq!(
            is_relevant_compute_to_pack_stick_dim(&Computes(vec![ComputeOp::Other])),
            None
        );
    }

    #[test]
    fn an_input_transfer_is_taken_only_when_its_data_structure_spans_the_stick() {
        let ex = SenComponent::Ptrow3;
        let compute = compute_on(ex, DataConnect::PeHtOut);
        let taken = TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            name: NodeName("in".to_owned()),
            src: operand(SenComponent::Lxlu, Some(DataConnect::PeHtOut), Some(0)),
            dsts: Dsts::new(
                operand(SenComponent::Lx, None, Some(0)),
                vec![operand(ex, Some(DataConnect::PeHtOut), Some(0))],
            ),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
        };
        // Same shape, but lds 1's scales hold no -2: the destination index is still recorded.
        let mut skipped = taken.clone();
        skipped.src = operand(SenComponent::Lxlu, Some(DataConnect::PeHtOut), Some(1));
        skipped.dsts = Dsts::new(
            operand(SenComponent::Lx, None, Some(1)),
            vec![operand(ex, Some(DataConnect::PeHtOut), Some(1))],
        );
        let lds = Labeled(vec![
            (DsType::Internal, vec![Scale::UnitStick, Scale::StickDim]),
            (DsType::Internal, vec![Scale::Sized(1.0)]),
        ]);
        let found =
            find_relevant_stick_packing_transfers(&Transfers(vec![taken, skipped]), &lds, &compute);
        assert_eq!(
            found.inputs,
            vec![StickPackTransfer {
                node: NodeId(0),
                src_lds: LdsIdx(0),
                data_connected: true,
            }]
        );
        assert_eq!(found.output, None);
        assert_eq!(found.dst_indices, vec![DestIdx(1), DestIdx(1)]);
    }

    #[test]
    fn two_inputs_of_different_ds_types_are_not_eligible() {
        let lds = Labeled(vec![
            (DsType::Input, vec![Scale::StickDim]),
            (DsType::Kernel, vec![Scale::StickDim]),
        ]);
        let one = |lds_idx| StickPackTransfer {
            node: NodeId(0),
            src_lds: LdsIdx(lds_idx),
            data_connected: true,
        };
        assert!(!is_eligible_compute(
            &lds,
            &[one(0), one(1)],
            &[LdsIdx(0), LdsIdx(1)]
        ));
        assert!(is_eligible_compute(
            &lds,
            &[one(0), one(0)],
            &[LdsIdx(0), LdsIdx(0)]
        ));
    }

    /// What entry 108 did, in the order it did it.
    #[derive(Debug, PartialEq, Eq)]
    enum Edit {
        Cloned(NodeId),
        RepetitionSet(NodeId, OutputIdx, Repetition),
        Recorded(NodeId, NodeId),
        AllocUser(AllocId, NodeId),
        Spread(AllocId, PrimaryDim, StickSpread),
    }

    /// One compute with two outputs, the first repeating three times into allocation 7.
    #[derive(Default)]
    struct Spreading {
        next: u32,
        log: Vec<Edit>,
        /// The output has no allocation — where the reference throws or dereferences null.
        unallocated: bool,
    }

    impl OffsetAdjustment for Spreading {
        fn output_repetitions(&self, _node: NodeId) -> Vec<Repetition> {
            vec![Repetition(3), Repetition(1)]
        }
        fn clone_compute_after(&mut self, _node: NodeId) -> NodeId {
            self.next += 1;
            let clone = NodeId(self.next);
            self.log.push(Edit::Cloned(clone));
            clone
        }
        fn set_output_repetition(&mut self, node: NodeId, idx: OutputIdx, reps: Repetition) {
            self.log.push(Edit::RepetitionSet(node, idx, reps));
        }
        fn record_clone(&mut self, original: NodeId, clone: NodeId) {
            self.log.push(Edit::Recorded(original, clone));
        }
        fn output_allocation(&self, _node: NodeId, _idx: OutputIdx) -> Option<AllocId> {
            (!self.unallocated).then_some(AllocId(7))
        }
        fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
            self.log.push(Edit::AllocUser(alloc, user));
        }
        fn alloc_outermost_layout_dim(&self, _alloc: AllocId) -> PrimaryDim {
            PrimaryDim::Out
        }
        fn set_gap_stick_spread(&mut self, alloc: AllocId, dim: PrimaryDim, spread: StickSpread) {
            self.log.push(Edit::Spread(alloc, dim, spread));
        }
    }

    #[test]
    fn a_three_way_repetition_inserts_two_clones_and_spreads_its_allocation_once() {
        let mut tree = Spreading::default();
        clone_compute_for_offset_adjustment(&mut tree, NodeId(0));
        assert_eq!(
            tree.log,
            vec![
                Edit::Cloned(NodeId(1)),
                Edit::RepetitionSet(NodeId(1), OutputIdx(0), Repetition(1)),
                Edit::Recorded(NodeId(0), NodeId(1)),
                Edit::AllocUser(AllocId(7), NodeId(1)),
                Edit::Spread(AllocId(7), PrimaryDim::Out, StickSpread(3)),
                Edit::Cloned(NodeId(2)),
                Edit::RepetitionSet(NodeId(2), OutputIdx(0), Repetition(1)),
                Edit::Recorded(NodeId(0), NodeId(2)),
                Edit::AllocUser(AllocId(7), NodeId(2)),
            ]
        );
    }

    #[test]
    fn an_output_with_no_allocation_still_clones_and_records() {
        let mut tree = Spreading {
            unallocated: true,
            ..Spreading::default()
        };
        clone_compute_for_offset_adjustment(&mut tree, NodeId(0));
        assert_eq!(
            tree.log,
            vec![
                Edit::Cloned(NodeId(1)),
                Edit::RepetitionSet(NodeId(1), OutputIdx(0), Repetition(1)),
                Edit::Recorded(NodeId(0), NodeId(1)),
                Edit::Cloned(NodeId(2)),
                Edit::RepetitionSet(NodeId(2), OutputIdx(0), Repetition(1)),
                Edit::Recorded(NodeId(0), NodeId(2)),
            ]
        );
    }

    /// A compute under a two-dim loop, on a two-corelet core.
    #[derive(Default)]
    struct Masking {
        name: String,
        offsets: Vec<(u32, PrimaryDim, MaskLoopOffset)>,
    }

    impl ComputeMasking for Masking {
        fn compute_name(&self, _compute: NodeId) -> NodeName {
            NodeName(self.name.clone())
        }
        fn set_compute_name(&mut self, _compute: NodeId, name: NodeName) {
            self.name = name.0;
        }
        fn parent_dim_loop(&self, _compute: NodeId, dim: PrimaryDim) -> Option<LoopId> {
            (dim == PrimaryDim::Out).then_some(LoopId(NodeId(9)))
        }
        fn corelets(&self) -> Vec<Corelet> {
            (0..2).filter_map(Corelet::checked).collect()
        }
        fn loop_dims(&self, _dim_loop: LoopId) -> Vec<PrimaryDim> {
            vec![PrimaryDim::Out, PrimaryDim::In]
        }
        fn set_compute_mask_loop_offset(
            &mut self,
            _compute: NodeId,
            corelet: Corelet,
            _dim_loop: LoopId,
            dim: PrimaryDim,
            offset: MaskLoopOffset,
        ) {
            self.offsets.push((corelet.get(), dim, offset));
        }
    }

    #[test]
    fn masking_renames_the_compute_and_advances_every_dim_on_every_corelet() {
        let mut tree = Masking {
            name: "mac0".to_owned(),
            offsets: Vec::new(),
        };
        // No loop over `In`, so the reference's null dereference is unconstructible.
        assert_eq!(
            MaskedComputeSite::of(&tree, NodeId(0), PrimaryDim::In),
            None
        );
        let site = MaskedComputeSite::of(&tree, NodeId(0), PrimaryDim::Out)
            .expect("a loop over Out was stated");
        transform_a_compute_node_for_inter_slice_restickify(&mut tree, site);
        assert_eq!(tree.name, "mac0_masked");
        assert_eq!(
            tree.offsets,
            vec![
                (0, PrimaryDim::Out, MaskLoopOffset::ADVANCE),
                (0, PrimaryDim::In, MaskLoopOffset::ADVANCE),
                (1, PrimaryDim::Out, MaskLoopOffset::ADVANCE),
                (1, PrimaryDim::In, MaskLoopOffset::ADVANCE),
            ]
        );
    }
}

#[cfg(test)]
mod tests_e242_e246 {
    use super::*;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        DataInfo, Dsts, InstrAttribute, NumChunks, ReplicationFactor, TransferPadding,
    };
    use crate::units::NumFolds;
    use std::collections::BTreeMap;

    fn operand(
        unit: SenComponent,
        storage: SenComponent,
        connect: Option<DataConnect>,
        lds: Option<u32>,
    ) -> Operand {
        Operand {
            unit,
            storage,
            data: DataInfo {
                data_connect: connect,
                my_lds_idx: lds.map(LdsIdx),
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    fn compute_node(
        ex_unit: SenComponent,
        inputs: Vec<Operand>,
        outputs: Vec<Operand>,
    ) -> ComputeNode {
        ComputeNode {
            name: NodeName("c".to_owned()),
            op: DdlComputeType::Macc,
            ex_unit,
            inputs,
            outputs,
            num_folds_engaged: NumFolds::ONE,
            data_format: None,
            instr_attribute: InstrAttribute::default(),
        }
    }

    fn transfer_node(src: Operand, dsts: Dsts) -> TransferNode {
        TransferNode {
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            name: NodeName("t".to_owned()),
            src,
            dsts,
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
        }
    }

    /// An LXLU→PE transfer of `PeHtOut`, and the PE compute that reads it off `PELRF`.
    fn lxlu_to_pe(dst_unit: SenComponent) -> (TransferNode, ComputeNode) {
        let xfer = transfer_node(
            operand(
                SenComponent::Lxlu,
                SenComponent::Lxlu,
                Some(DataConnect::PeHtOut),
                Some(0),
            ),
            Dsts::new(
                operand(
                    dst_unit,
                    SenComponent::Pelrf,
                    Some(DataConnect::PeHtOut),
                    Some(0),
                ),
                vec![],
            ),
        );
        let consumer = compute_node(
            SenComponent::Pe,
            vec![operand(
                SenComponent::Pelrf,
                SenComponent::Pelrf,
                Some(DataConnect::PeHtOut),
                Some(0),
            )],
            vec![operand(SenComponent::Pelrf, SenComponent::Pelrf, None, Some(1))],
        );
        (xfer, consumer)
    }

    /// One node of the fixture tree.
    struct Node {
        scope: ScopeNode,
        path: PathKind,
        parent: Option<NodeId>,
    }

    /// A schedule tree as a flat list, each node naming its parent.
    struct Tree {
        nodes: Vec<Node>,
        loads: Loads,
    }

    impl Tree {
        fn at(&self, node: NodeId) -> &Node {
            &self.nodes[node.0 as usize]
        }

        fn path_node(&self, node: NodeId) -> PathNode {
            PathNode {
                node,
                kind: self.at(node).path.clone(),
            }
        }

        /// The node and its ancestors, innermost first.
        fn chain(&self, node: NodeId) -> Vec<NodeId> {
            let mut chain = vec![node];
            while let Some(parent) = self.parent(*chain.last().expect("a seeded chain")) {
                chain.push(parent);
            }
            chain
        }
    }

    impl ScopeTree for Tree {
        fn ancestry(&self, transfer: NodeId, consumer: NodeId) -> Ancestry {
            let to_transfer = self.chain(transfer);
            let to_consumer = self.chain(consumer);
            let common = *to_transfer
                .iter()
                .find(|node| to_consumer.contains(node))
                .expect("one shared root was stated");
            let cut = |chain: Vec<NodeId>| {
                let mut below = chain.into_iter().take_while(|node| *node != common);
                let node = self.path_node(below.next().expect("the node itself"));
                AncestorPath::new(node, below.map(|inner| self.path_node(inner)).collect())
            };
            Ancestry {
                to_transfer: cut(to_transfer),
                to_consumer: cut(to_consumer),
                ancestor: self.path_node(common),
            }
        }

        fn scope_node(&self, node: NodeId) -> ScopeNode {
            self.at(node).scope.clone()
        }

        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.at(node).parent
        }

        fn non_broadcast_lds_dims(&self, _lds: Option<LdsIdx>) -> BTreeSet<PrimaryDim> {
            BTreeSet::new()
        }
    }

    impl TransferLoads for Tree {
        fn corelets(&self) -> Vec<Corelet> {
            (0..1).filter_map(Corelet::checked).collect()
        }

        fn block_transfer_loads(&self, _transfer: NodeId, _corelet: Corelet) -> Loads {
            self.loads
        }
    }

    /// `dataStageParam_` for one loop: the `numId_` extent and the `denId_` one.
    struct Stages(Extent, Extent);

    impl StageExtents for Stages {
        fn stage_extent(&self, stage: DatastageId, _dim: PrimaryDim) -> Extent {
            if stage == DatastageId(0) { self.0 } else { self.1 }
        }
    }

    /// A root block holding the transfer at node 1, then `between`, then the consumer LAST.
    fn one_block(xfer: TransferNode, between: Vec<ScopeNode>, consumer: ScopeNode) -> Tree {
        let mut nodes = vec![Node {
            scope: ScopeNode::Nest(Vec::new()),
            path: PathKind::Other,
            parent: None,
        }];
        for scope in core::iter::once(ScopeNode::Transfer(xfer))
            .chain(between)
            .chain(core::iter::once(consumer))
        {
            nodes.push(Node {
                scope,
                path: PathKind::Other,
                parent: Some(NodeId(0)),
            });
        }
        let children = (1..u32::try_from(nodes.len()).expect("a small fixture"))
            .map(NodeId)
            .collect();
        nodes[0].scope = ScopeNode::Nest(children);
        Tree {
            nodes,
            loads: Loads::ONE,
        }
    }

    #[test]
    fn siblings_can_use_a_fifo_until_something_between_them_drives_the_channel() {
        let (xfer, consumer) = lxlu_to_pe(SenComponent::Pe);
        let tree = one_block(xfer.clone(), vec![], ScopeNode::Compute(consumer.clone()));
        let dst = TransferDst::of(NodeId(1), &xfer, DestIdx(0)).expect("one destination");
        assert!(can_use_fifo(&tree, Some(&Stages(Extent(1), Extent(1))), &dst, NodeId(2)));

        // A PE compute reading LXLU between the two occupies the very channel the FIFO would use.
        let interposed = compute_node(
            SenComponent::Pe,
            vec![operand(SenComponent::Lxlu, SenComponent::Lxlu, None, None)],
            vec![],
        );
        let blocked = one_block(
            xfer.clone(),
            vec![ScopeNode::Compute(interposed)],
            ScopeNode::Compute(consumer.clone()),
        );
        assert!(!can_use_fifo(
            &blocked,
            Some(&Stages(Extent(1), Extent(1))),
            &dst,
            NodeId(3)
        ));

        // ⛔ THE TRAP: the LXLUVALUE destination that refuses a FIFO PERMITS a latch.
        let (value_xfer, _) = lxlu_to_pe(SenComponent::Lxluvalue);
        let value_dst =
            TransferDst::of(NodeId(1), &value_xfer, DestIdx(0)).expect("one destination");
        assert!(!can_use_fifo(
            &tree,
            Some(&Stages(Extent(1), Extent(1))),
            &value_dst,
            NodeId(2)
        ));
        assert!(can_use_latch(
            &tree,
            &Metadata::default(),
            &value_dst,
            &Consumers::new(NodeId(2), vec![])
        ));
    }

    #[test]
    fn a_loop_over_the_consumer_needs_its_dimension_known_to_trip_once() {
        let (xfer, consumer) = lxlu_to_pe(SenComponent::Pe);
        // Root { transfer, loop { consumer } } — the transfer's data carries no dim, so the loop's
        // own dim is never `relevantDims` and only equal datastage extents can excuse it.
        let tree = Tree {
            nodes: vec![
                Node {
                    scope: ScopeNode::Nest(vec![NodeId(1), NodeId(2)]),
                    path: PathKind::Other,
                    parent: None,
                },
                Node {
                    scope: ScopeNode::Transfer(xfer.clone()),
                    path: PathKind::Other,
                    parent: Some(NodeId(0)),
                },
                Node {
                    scope: ScopeNode::Nest(vec![NodeId(3)]),
                    path: PathKind::Loop(PathLoop {
                        parametric: false,
                        dims: vec![(PrimaryDim::Out, MetaDimKind::Unpadded)],
                        num: DatastageId(0),
                        den: DatastageId(1),
                    }),
                    parent: Some(NodeId(0)),
                },
                Node {
                    scope: ScopeNode::Compute(consumer),
                    path: PathKind::Other,
                    parent: Some(NodeId(2)),
                },
            ],
            loads: Loads::ONE,
        };
        let dst = TransferDst::of(NodeId(1), &xfer, DestIdx(0)).expect("one destination");
        assert!(can_use_fifo(
            &tree,
            Some(&Stages(Extent(4), Extent(4))),
            &dst,
            NodeId(3)
        ));
        assert!(!can_use_fifo(
            &tree,
            Some(&Stages(Extent(4), Extent(2))),
            &dst,
            NodeId(3)
        ));
        // ⛔ `None` IS `!dataStageExplorationDone_`: the extents cannot be consulted at all.
        assert!(!can_use_fifo(&tree, None::<&Stages>, &dst, NodeId(3)));
    }

    #[test]
    fn a_latch_is_refused_by_any_unrelated_compute_in_the_scope() {
        let (xfer, consumer) = lxlu_to_pe(SenComponent::Pe);
        let dst = TransferDst::of(NodeId(1), &xfer, DestIdx(0)).expect("one destination");
        let consumers = Consumers::new(NodeId(2), vec![]);
        let tree = one_block(xfer.clone(), vec![], ScopeNode::Compute(consumer.clone()));
        assert!(can_use_latch(&tree, &Metadata::default(), &dst, &consumers));

        // An unrelated compute between producer and consumer, and then a latching transfer.
        let unrelated = compute_node(SenComponent::Pe, vec![], vec![]);
        let latching = transfer_node(
            operand(SenComponent::Lxlu, SenComponent::Lxlu, None, None),
            Dsts::new(
                operand(SenComponent::Pe, SenComponent::Latch, None, None),
                vec![],
            ),
        );
        for between in [
            ScopeNode::Compute(unrelated),
            ScopeNode::Transfer(latching),
        ] {
            let blocked = one_block(
                xfer.clone(),
                vec![between],
                ScopeNode::Compute(consumer.clone()),
            );
            assert!(!can_use_latch(
                &blocked,
                &Metadata::default(),
                &dst,
                &Consumers::new(NodeId(3), vec![])
            ));
        }

        // An opaque consumer refuses, and `isOpaqueOp_` is read as its `opaqueOps_` entry.
        let mut metadata = Metadata::default();
        metadata
            .opaque_ops
            .insert(NodeId(2), crate::schedule::ddc::metadata::OpaqueOp::default());
        assert!(!can_use_latch(&tree, &metadata, &dst, &consumers));
    }

    /// Two computes, each with one output that repeats twice.
    #[derive(Default)]
    struct Cloning {
        next: u32,
        cloned: Vec<(NodeId, NodeId)>,
    }

    impl ComputeWalk for Cloning {
        fn computes(&self) -> Vec<NodeId> {
            vec![NodeId(0), NodeId(1)]
        }
        fn compute_op(&self, _node: NodeId) -> ComputeOp {
            ComputeOp::Other
        }
    }

    impl OffsetAdjustment for Cloning {
        fn output_repetitions(&self, _node: NodeId) -> Vec<Repetition> {
            vec![Repetition(2)]
        }
        fn clone_compute_after(&mut self, _node: NodeId) -> NodeId {
            self.next += 1;
            NodeId(100 + self.next)
        }
        fn set_output_repetition(&mut self, _node: NodeId, _idx: OutputIdx, _reps: Repetition) {}
        fn record_clone(&mut self, original: NodeId, clone: NodeId) {
            self.cloned.push((original, clone));
        }
        fn output_allocation(&self, _node: NodeId, _idx: OutputIdx) -> Option<AllocId> {
            None
        }
        fn add_alloc_user(&mut self, _alloc: AllocId, _user: NodeId) {}
        fn alloc_outermost_layout_dim(&self, _alloc: AllocId) -> PrimaryDim {
            PrimaryDim::Out
        }
        fn set_gap_stick_spread(
            &mut self,
            _alloc: AllocId,
            _dim: PrimaryDim,
            _spread: StickSpread,
        ) {
        }
    }

    #[test]
    fn every_compute_in_the_tree_is_offset_adjusted_and_the_clones_are_not_revisited() {
        let mut tree = Cloning::default();
        clone_for_offset_adjustment(&mut tree);
        assert_eq!(
            tree.cloned,
            vec![(NodeId(0), NodeId(101)), (NodeId(1), NodeId(102))]
        );
    }

    /// One transfer with two destinations, and the sizes entry 245 wrote.
    struct Sizing {
        transfer: TransferNode,
        written: Vec<(NodeId, Vec<(PrimaryDim, Elements)>)>,
    }

    impl TransferWalk for Sizing {
        fn transfers(&self) -> Vec<NodeId> {
            vec![NodeId(0)]
        }
        fn transfer(&self, _node: NodeId) -> TransferNode {
            self.transfer.clone()
        }
    }

    impl DsSticks for Sizing {
        fn ds_stick_dims(&self, lds: LdsIdx) -> StickDims {
            StickDims(vec![(PrimaryDim::Out, Elements(u64::from(lds.0) + 2))])
        }
    }

    impl FixedSizeTransfers for Sizing {
        fn connect_consumers(&self, connect: DataConnect) -> Vec<ScopeNode> {
            let ex_unit = if connect == DataConnect::ArfPt {
                SenComponent::Lxlu
            } else {
                SenComponent::Pe
            };
            vec![ScopeNode::Compute(compute_node(ex_unit, vec![], vec![]))]
        }
        fn set_transfer_size(&mut self, transfer: NodeId, sizes: Vec<(PrimaryDim, Elements)>) {
            self.written.push((transfer, sizes));
        }
    }

    #[test]
    fn only_the_first_destination_feeding_an_lxlu_compute_sizes_the_transfer() {
        let mut tree = Sizing {
            transfer: transfer_node(
                operand(SenComponent::Lxlu, SenComponent::Lxlu, None, Some(0)),
                Dsts::new(
                    // A destination whose consumer is not on LXLU, then one whose consumer is.
                    operand(
                        SenComponent::Pe,
                        SenComponent::Pelrf,
                        Some(DataConnect::PeHtOut),
                        Some(9),
                    ),
                    vec![operand(
                        SenComponent::Lxlu,
                        SenComponent::Lxlu,
                        Some(DataConnect::ArfPt),
                        Some(3),
                    )],
                ),
            ),
            written: Vec::new(),
        };
        set_size_for_fixed_size_transfers(&mut tree);
        assert_eq!(
            tree.written,
            vec![(NodeId(0), vec![(PrimaryDim::Out, Elements(5))])]
        );
    }

    /// One PT compute of the `(ZERO, ONE, <a labelled DS>)` shape, and what masking it recorded.
    struct Restickify {
        compute: ComputeNode,
        sticks: Vec<StickDims>,
        offsets: Vec<(u32, PrimaryDim)>,
    }

    impl ComputeWalk for Restickify {
        fn computes(&self) -> Vec<NodeId> {
            vec![NodeId(0)]
        }
        fn compute_op(&self, _node: NodeId) -> ComputeOp {
            ComputeOp::Other
        }
    }

    impl ComputeNodes for Restickify {
        fn compute(&self, _compute: NodeId) -> ComputeNode {
            self.compute.clone()
        }
    }

    impl DsSticks for Restickify {
        fn ds_stick_dims(&self, lds: LdsIdx) -> StickDims {
            self.sticks[lds.0 as usize].clone()
        }
    }

    impl ComputeMasking for Restickify {
        fn compute_name(&self, _compute: NodeId) -> NodeName {
            self.compute.name.clone()
        }
        fn set_compute_name(&mut self, _compute: NodeId, name: NodeName) {
            self.compute.name = name;
        }
        fn parent_dim_loop(&self, _compute: NodeId, _dim: PrimaryDim) -> Option<LoopId> {
            Some(LoopId(NodeId(9)))
        }
        fn corelets(&self) -> Vec<Corelet> {
            (0..1).filter_map(Corelet::checked).collect()
        }
        fn loop_dims(&self, _dim_loop: LoopId) -> Vec<PrimaryDim> {
            vec![PrimaryDim::In]
        }
        fn set_compute_mask_loop_offset(
            &mut self,
            _compute: NodeId,
            corelet: Corelet,
            _dim_loop: LoopId,
            dim: PrimaryDim,
            _offset: MaskLoopOffset,
        ) {
            self.offsets.push((corelet.get(), dim));
        }
    }

    #[test]
    fn a_pt_compute_is_masked_only_when_its_input_and_output_sticks_differ() {
        let restickify = |output: StickDims| Restickify {
            compute: compute_node(
                SenComponent::Ptrow0,
                vec![
                    operand(SenComponent::Zero, SenComponent::NoComponent, None, None),
                    operand(SenComponent::One, SenComponent::NoComponent, None, None),
                    operand(SenComponent::Lxlu, SenComponent::Lxlu, None, Some(0)),
                ],
                vec![operand(SenComponent::Pe, SenComponent::Pelrf, None, Some(1))],
            ),
            sticks: vec![StickDims(vec![(PrimaryDim::Out, Elements(4))]), output],
            offsets: Vec::new(),
        };

        let mut differing = restickify(StickDims(vec![(PrimaryDim::In, Elements(4))]));
        transform_for_inter_slice_restickify(&mut differing);
        assert_eq!(differing.compute.name, NodeName("c_masked".to_owned()));
        assert_eq!(differing.offsets, vec![(0, PrimaryDim::In)]);

        let mut same = restickify(StickDims(vec![(PrimaryDim::Out, Elements(4))]));
        transform_for_inter_slice_restickify(&mut same);
        assert_eq!(same.compute.name, NodeName("c".to_owned()));
        assert!(same.offsets.is_empty());
    }
}

#[cfg(test)]
mod tests_e300 {
    use super::*;
    use crate::schedule::ddc::transformation_util::{
        AllocationUse, CanDelete, LoopNode, StageDims, StageName,
    };
    use crate::schedule::dsc2::Dsc as Dsc2;

    /// A stage payload that states no split at all — the corelet-split read is the only thing entry
    /// 300's read half asks of it.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct NoSplits;

    impl UtilStageExtents for NoSplits {
        fn copy_dim_value_from(&mut self, _other: &Self, _dim: PrimaryDim) {}
        fn states(&self, _split: DimSplit, _dim: PrimaryDim) -> bool {
            false
        }
        fn copy_split_from(&mut self, _other: &Self, _split: DimSplit, _dim: PrimaryDim) {}
        fn make_dim_not_symbolic(&mut self, _dim: PrimaryDim) {}
        fn clear_split(&mut self, _split: DimSplit) {}
        fn split_sizes(&self, _split: DimSplit, _dim: PrimaryDim) -> Vec<Elements> {
            Vec::new()
        }
    }

    /// ⭐ THE REFERENCE'S OWN WORKED CASE (`ddc/ddc_transformation.cpp:812-817`): a stick of
    /// `<out:64>` over an MB loop whose buffer holds four, repacked as `<out:16, MB:4>`.
    struct Packing {
        mb_capacity: Elements,
    }

    const OUT_STICK: u64 = 64;

    impl ComputeWalk for Packing {
        fn computes(&self) -> Vec<NodeId> {
            vec![NodeId(0)]
        }
        fn compute_op(&self, _node: NodeId) -> ComputeOp {
            ComputeOp::Reciprocal
        }
    }

    impl ComputeNodes for Packing {
        fn compute(&self, _compute: NodeId) -> ComputeNode {
            ComputeNode {
                name: NodeName("recip".to_owned()),
                op: DdlComputeType::Macc,
                ex_unit: SenComponent::Sfp,
                inputs: vec![operand(SenComponent::Sfp, Some(0))],
                outputs: vec![operand(SenComponent::Sfp, Some(1))],
                num_folds_engaged: NumFolds::ONE,
                data_format: None,
                instr_attribute: InstrAttribute::default(),
            }
        }
    }

    impl TransferWalk for Packing {
        fn transfers(&self) -> Vec<NodeId> {
            vec![NodeId(1), NodeId(2)]
        }
        fn transfer(&self, node: NodeId) -> TransferNode {
            if node == NodeId(1) {
                transfer(
                    operand(SenComponent::Lxlu, Some(0)),
                    operand(SenComponent::Sfp, Some(0)),
                )
            } else {
                transfer(
                    operand(SenComponent::Sfp, Some(1)),
                    operand(SenComponent::Lxsu, Some(1)),
                )
            }
        }
    }

    impl LabeledDs for Packing {
        fn scale(&self, _lds: LdsIdx) -> Vec<Scale> {
            vec![Scale::StickDim, Scale::Sized(1.0)]
        }
        fn ds_type(&self, _lds: LdsIdx) -> DsType {
            DsType::Input
        }
    }

    impl PackStickDimReads for Packing {
        fn buffer_capacity_per_dim(
            &self,
            _node: NodeId,
            _lds: LdsIdx,
            _storage: SenComponent,
        ) -> Vec<(PrimaryDim, Option<Elements>)> {
            vec![
                (PrimaryDim::Out, Some(Elements(OUT_STICK))),
                (PrimaryDim::Mb, Some(self.mb_capacity)),
            ]
        }

        fn primary_ds_info(&self, _ds_type: DsType) -> Option<PrimaryDsInfo> {
            Some(PrimaryDsInfo {
                layout: LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::Mb]),
                stick: StickDims(vec![(PrimaryDim::Out, Elements(OUT_STICK))]),
            })
        }
    }

    impl ComputeMasking for Packing {
        fn compute_name(&self, _compute: NodeId) -> NodeName {
            NodeName("recip".to_owned())
        }
        fn set_compute_name(&mut self, _compute: NodeId, _name: NodeName) {}
        fn parent_dim_loop(&self, _compute: NodeId, _dim: PrimaryDim) -> Option<LoopId> {
            Some(LoopId(NodeId(3)))
        }
        fn corelets(&self) -> Vec<Corelet> {
            Vec::new()
        }
        fn loop_dims(&self, _dim_loop: LoopId) -> Vec<PrimaryDim> {
            vec![PrimaryDim::Mb]
        }
        fn set_compute_mask_loop_offset(
            &mut self,
            _compute: NodeId,
            _corelet: Corelet,
            _dim_loop: LoopId,
            _dim: PrimaryDim,
            _offset: MaskLoopOffset,
        ) {
        }
    }

    impl Dsc2 for Packing {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::Mb])
        }
    }

    impl DscAllocations for Packing {
        fn own_lds_idx(&self, lds: LdsIdx) -> LdsIdx {
            lds
        }
        fn allocation_in(&self, origin: DataOrigin, _storage: DdcMemory) -> Option<AllocId> {
            match origin {
                DataOrigin::LabeledDs(lds) => Some(AllocId(lds.0)),
                DataOrigin::Constant(_) => None,
            }
        }
        fn set_allocation_in(&mut self, _lds: LdsIdx, _storage: DdcMemory, _alloc: AllocId) {}
        fn alloc_users(&self, _alloc: AllocId) -> Vec<NodeId> {
            Vec::new()
        }
        fn alloc_component(&self, _alloc: AllocId) -> DdcMemory {
            DdcMemory::Lx
        }
        fn alloc_origin(&self, alloc: AllocId) -> DataOrigin {
            DataOrigin::LabeledDs(LdsIdx(alloc.0))
        }
        fn alloc_node(&self, _alloc: AllocId) -> NodeId {
            NodeId(4)
        }
        fn reduce_users_or_delete(
            &mut self,
            _alloc_use: AllocationUse,
            _can_delete: CanDelete,
        ) -> bool {
            false
        }
    }

    impl ScheduleSurgery for Packing {
        fn node_name(&self, _node: NodeId) -> NodeName {
            NodeName::default()
        }
        fn set_node_name(&mut self, _node: NodeId, _name: NodeName) {}
        fn parent(&self, _node: NodeId) -> Option<NodeId> {
            None
        }
        fn owner_loop(&self, _node: NodeId) -> Option<LoopId> {
            Some(LoopId(NodeId(3)))
        }
        fn transfer(&self, node: NodeId) -> TransferNode {
            TransferWalk::transfer(self, node)
        }
        fn loop_num(&self, _loop_node: LoopId) -> DatastageId {
            DatastageId(0)
        }
        fn loop_den(&self, _loop_node: LoopId) -> DatastageId {
            DatastageId(1)
        }
        fn loop_dims(&self, _loop_node: LoopId) -> LoopDims {
            LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::Mb,
                    kind: MetaDimKind::Unpadded,
                },
                vec![PrimaryDimAndKind {
                    dim: PrimaryDim::Out,
                    kind: MetaDimKind::Unpadded,
                }],
            )
        }
        fn is_parametric(&self, _loop_node: LoopId) -> bool {
            false
        }
        fn new_loop(&mut self, _loop_node: LoopNode) -> LoopId {
            LoopId(NodeId(9))
        }
        fn new_block(&mut self, _name: NodeName) -> NodeId {
            NodeId(9)
        }
        fn add_child_node(&mut self, _node: NodeId, _at: InsertionPoint) {}
        fn move_node(&mut self, _node: NodeId, _at: InsertionPoint) {}
        fn conditions_under(&self, _root: NodeId) -> Vec<NodeId> {
            Vec::new()
        }
        fn loop_cond(&self, _condition: NodeId) -> LoopCondComposite {
            LoopCondComposite::default()
        }
        fn set_loop_cond(&mut self, _condition: NodeId, _cond: LoopCondComposite) {}
    }

    fn operand(unit: SenComponent, lds: Option<u32>) -> Operand {
        Operand {
            unit,
            storage: unit,
            data: DataInfo {
                data_connect: None,
                my_lds_idx: lds.map(LdsIdx),
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    fn transfer(src: Operand, dst: Operand) -> TransferNode {
        TransferNode {
            name: NodeName("t".to_owned()),
            src,
            dsts: Dsts::new(dst, Vec::new()),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks::ONE,
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
        }
    }

    fn stages() -> DataStages<NoSplits> {
        let stage = DataStage {
            ss: StageDims {
                name: StageName("0".to_owned()),
                dims: NoSplits,
            },
            el: StageDims {
                name: StageName("0el".to_owned()),
                dims: NoSplits,
            },
        };
        let mut stages = DataStages::default();
        stages.0.insert(DatastageId(0), stage.clone());
        stages.0.insert(DatastageId(1), stage);
        stages
    }

    #[test]
    fn repacks_a_64_stick_over_an_mb_loop_of_four() {
        let tree = Packing {
            mb_capacity: Elements(4),
        };
        let site =
            PackStickDimSite::of(&tree, &stages(), DatastageExploration::Open).expect("a site");
        assert_eq!(
            site.new_stick_dims,
            vec![
                (PrimaryDim::Out, Elements(16)),
                (PrimaryDim::Mb, Elements(4))
            ]
        );
    }

    #[test]
    fn an_mb_loop_of_one_lends_nothing_and_there_is_no_site() {
        let tree = Packing {
            mb_capacity: Elements(1),
        };
        assert!(PackStickDimSite::of(&tree, &stages(), DatastageExploration::Open).is_none());
    }
}

#[cfg(test)]
mod tests_e338 {
    use super::*;
    use crate::schedule::ddc::fold::ConstIdx;
    use crate::schedule::ddc::fold::StoredStream;
    use crate::schedule::ddc::transformation_util::{
        AllocationUse, CanDelete, DdcAllocateNode, FifoConsumer, LoopNode, TransferEnds,
    };
    use crate::schedule::dsc2::AllocateNode;
    use crate::schedule::dsc2::Dsc as Dsc2;
    use std::cell::Cell;

    /// A tree holding ONE transfer that involves neither PE nor SFP, so the transfer arm's own
    /// witness refuses it and no clone machinery is reached.
    struct Split {
        has_split: bool,
        asked: Cell<bool>,
    }

    impl PeSfpWorkSplit for Split {
        fn has_pe_sfp_split(&self) -> bool {
            self.has_split
        }
        fn split_candidates(&self) -> Vec<PeSfpSplitNode> {
            self.asked.set(true);
            vec![PeSfpSplitNode::Transfer(NodeId(1))]
        }
    }

    impl NodeCloning for Split {
        fn clone_transfer_after(&mut self, _node: NodeId, _body: TransferNode) -> NodeId {
            unimplemented!("no transfer of this tree is cloneable")
        }
    }

    impl ScheduleSurgery for Split {
        fn node_name(&self, _node: NodeId) -> NodeName {
            NodeName::default()
        }
        fn set_node_name(&mut self, _node: NodeId, _name: NodeName) {}
        fn parent(&self, _node: NodeId) -> Option<NodeId> {
            None
        }
        fn owner_loop(&self, _node: NodeId) -> Option<LoopId> {
            None
        }
        fn transfer(&self, _node: NodeId) -> TransferNode {
            TransferNode {
                name: NodeName("lxlu_to_lxsu".to_owned()),
                src: operand(SenComponent::Lxlu),
                dsts: Dsts::new(operand(SenComponent::Lxsu), Vec::new()),
                replication_factor: ReplicationFactor::ONE,
                unit_time_transfer_chunk_size: Vec::new(),
                unit_time_transfer_num_chunks: NumChunks::ONE,
                padding: TransferPadding::default(),
                src_indirect: None,
                dst_indirect: None,
                core_id_to_gtr_info: BTreeMap::new(),
                transfer_size: BTreeMap::new(),
            }
        }
        fn loop_num(&self, _loop_node: LoopId) -> DatastageId {
            DatastageId(0)
        }
        fn loop_den(&self, _loop_node: LoopId) -> DatastageId {
            DatastageId(1)
        }
        fn loop_dims(&self, _loop_node: LoopId) -> LoopDims {
            unimplemented!("no loop is read")
        }
        fn is_parametric(&self, _loop_node: LoopId) -> bool {
            false
        }
        fn new_loop(&mut self, _loop_node: LoopNode) -> LoopId {
            unimplemented!("no loop is minted")
        }
        fn new_block(&mut self, _name: NodeName) -> NodeId {
            unimplemented!("no block is minted")
        }
        fn add_child_node(&mut self, _node: NodeId, _at: InsertionPoint) {}
        fn move_node(&mut self, _node: NodeId, _at: InsertionPoint) {}
        fn conditions_under(&self, _root: NodeId) -> Vec<NodeId> {
            Vec::new()
        }
        fn loop_cond(&self, _condition: NodeId) -> LoopCondComposite {
            LoopCondComposite::default()
        }
        fn set_loop_cond(&mut self, _condition: NodeId, _cond: LoopCondComposite) {}
    }

    impl FifoResults for Split {
        fn connect_consumers(&self, _connect: Option<DataConnect>) -> Vec<FifoConsumer> {
            Vec::new()
        }
        fn is_opaque(&self, _compute: NodeId) -> bool {
            false
        }
        fn transfer_ends(&self, _transfer: NodeId) -> TransferEnds {
            unimplemented!("no allocation is read")
        }
        fn insert_allocate(&mut self, _alloc: AllocId, _node: DdcAllocateNode, _at: InsertionPoint) {
        }
        fn set_dst_storage(&mut self, _transfer: NodeId, _dst: usize, _storage: SenComponent) {}
        fn set_src_storage(&mut self, _transfer: NodeId, _storage: SenComponent) {}
        fn set_compute_input_unit(&mut self, _compute: NodeId, _input: usize, _unit: SenComponent) {}
        fn add_alloc_user(&mut self, _alloc: AllocId, _user: NodeId) {}
    }

    impl Allocations for Split {
        fn allocation(&self, _stored: StoredStream) -> Option<AllocId> {
            None
        }
        fn value_allocation(&self, _scale: AllocId) -> Option<AllocId> {
            None
        }
    }

    impl Dsc2 for Split {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            unimplemented!("no layout is read")
        }
    }

    impl TransferUnrolling for Split {
        fn non_broadcast_lds_dims(&self, _lds: LdsIdx) -> Option<Vec<PrimaryDim>> {
            None
        }
    }

    impl MintedConnects for Split {
        fn intern_connect(&mut self, _connect: MintedConnect) -> DataConnect {
            unimplemented!("no connect is minted")
        }
    }

    // ⭐ THE CLONE VOCABULARY ENTRIES 118 AND 119 REACH THE TREE THROUGH. This tree holds ONE
    // transfer, so the walk hands the pass no allocate and no compute and none of it is reached.
    impl DscAllocations for Split {
        fn own_lds_idx(&self, _lds: LdsIdx) -> LdsIdx {
            unimplemented!("no lds is renamed")
        }
        fn allocation_in(&self, _origin: DataOrigin, _storage: DdcMemory) -> Option<AllocId> {
            unimplemented!("no allocation is reached")
        }
        fn set_allocation_in(&mut self, _lds: LdsIdx, _storage: DdcMemory, _alloc: AllocId) {
            unimplemented!("no allocation is placed")
        }
        fn alloc_users(&self, _alloc: AllocId) -> Vec<NodeId> {
            unimplemented!("no user list is read")
        }
        fn alloc_component(&self, _alloc: AllocId) -> DdcMemory {
            unimplemented!("no allocation is reached")
        }
        fn alloc_origin(&self, _alloc: AllocId) -> DataOrigin {
            unimplemented!("no allocation is reached")
        }
        fn alloc_node(&self, _alloc: AllocId) -> NodeId {
            unimplemented!("no allocation is reached")
        }
        fn reduce_users_or_delete(&mut self, _use: AllocationUse, _can: CanDelete) -> bool {
            unimplemented!("no allocation is deleted")
        }
    }

    impl AllocationsByNode for Split {
        fn allocation_of(&self, _node: NodeId) -> Option<AllocId> {
            unimplemented!("the walk hands this pass no allocate node")
        }
    }

    impl AllocateCloning for Split {
        fn allocate(&self, _alloc: AllocId) -> AllocateNode {
            unimplemented!("no allocation is cloned")
        }
        fn clone_allocate_after(&mut self, _alloc: AllocId, _body: AllocateNode) -> AllocId {
            unimplemented!("no allocation is cloned")
        }
        fn set_temp_storage_for_compute(&mut self, _alloc: AllocId, _compute: NodeName) {
            unimplemented!("no allocation is cloned")
        }
    }

    impl ComputeCloning for Split {
        fn compute(&self, _node: NodeId) -> ComputeNode {
            unimplemented!("the walk hands this pass no compute node")
        }
        fn clone_compute_after(&mut self, _node: NodeId, _body: ComputeNode) -> NodeId {
            unimplemented!("no compute is cloned")
        }
    }

    impl ComponentAllocations for Split {
        fn has_mem_org(&self, _lds: LdsIdx, _storage: SenComponent) -> bool {
            unimplemented!("no mem org is reached")
        }
        fn copy_mem_org_without_allocation(
            &mut self,
            _lds: LdsIdx,
            _from: SenComponent,
            _to: SenComponent,
        ) {
            unimplemented!("no mem org is reached")
        }
        fn mem_org_allocation(&self, _lds: LdsIdx, _storage: SenComponent) -> Option<AllocId> {
            unimplemented!("no mem org is reached")
        }
        fn set_mem_org_allocation(
            &mut self,
            _lds: LdsIdx,
            _storage: SenComponent,
            _alloc: AllocId,
        ) {
            unimplemented!("no mem org is reached")
        }
        fn has_constant_allocation(&self, _constant: ConstIdx, _storage: SenComponent) -> bool {
            unimplemented!("no constant is reached")
        }
        fn set_constant_allocation(
            &mut self,
            _constant: ConstIdx,
            _storage: SenComponent,
            _alloc: AllocId,
        ) {
            unimplemented!("no constant is reached")
        }
    }

    fn operand(unit: SenComponent) -> Operand {
        Operand {
            unit,
            storage: unit,
            data: DataInfo {
                data_connect: None,
                my_lds_idx: None,
                constant_id: None,
                latch_data_id: None,
            },
        }
    }

    #[test]
    fn a_dsc_stating_no_worksplit_is_never_even_walked_and_a_transfer_off_the_pair_is_left_alone() {
        let mut stages = DataStages::<()>::default();
        let mut metadata = Metadata::default();

        let mut quiet = Split {
            has_split: false,
            asked: Cell::new(false),
        };
        assert!(perform_pe_sfp_work_split(
            &mut quiet,
            &mut stages,
            &mut metadata
        ));
        assert!(
            !quiet.asked.get(),
            "the tree is not traversed at all without worksplit information"
        );

        let mut splitting = Split {
            has_split: true,
            asked: Cell::new(false),
        };
        assert!(perform_pe_sfp_work_split(
            &mut splitting,
            &mut stages,
            &mut metadata
        ));
        assert!(splitting.asked.get(), "the tree IS traversed with it");
        assert!(
            metadata.node_cloning_map.is_empty(),
            "an LXLU-to-LXSU transfer involves neither PE nor SFP, so nothing is cloned"
        );
    }
}
