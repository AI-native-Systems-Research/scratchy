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
// ⭐ USES FOR ENTRIES 105-109. Union these into this file's top block when its other entries land.
// ════════════════════════════════════════════════════════════════════════════════════════════════

use sys_arch_spec::arch_enums::SenComponent;

use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
use crate::generated::DataConnect;
use crate::schedule::ddc::fold::{AllocId, NodeId};
use crate::schedule::ddc::metadata::DestIdx;
use crate::schedule::dsc2::{ComputeNode, LdsIdx, NodeName, Operand, TransferNode};
use crate::units::Corelet;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE STICK-PACKING AND OFFSET-ADJUSTMENT VOCABULARY — as entries 105-109 read it.
//
// ⭐ THE TRAITS ARE THE MECHANISM FOR REACHING OPERANDS, the one part the campaign statement names
// as droppable: `traverseTreeDFSMutable`, `clone`, `addChildNode`, `getAllocation` and
// `getParentDimLoop` are all `dsc/dsc2.cpp` — outside this campaign's file list. What these five
// units OWN is the decision and the mutation, and both are below.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH COMPUTE STICK PACKING CARES ABOUT — `ComputeOpType` (`dsc/dscdefn.h:157-158`) narrowed to
/// the two members entry 105 distinguishes.
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

/// ONE LAYOUT DIM'S SCALE — `LabeledDsInfo::scale_` entry (`dsc/dscdefn.h:327`), whose two negative
/// values are sentinels the `DT_CHECK` at `dsc/designSpaceConfig.cpp:523-525` closes to `1|-1|-2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    /// A non-negative scale: the dim's size comes from the data stage (`dsc/dsc2.cpp:3832`).
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
    /// UNINITIALISED pointer (`ddc/ddc_transformation.cpp:815`).
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

/// ONE `computeMaskLoopOffsets_` ENTRY — "put 1 if popping next element, or 0 if reuse is expected"
/// (`dsc/dsc2.h:920-923`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaskLoopOffset(pub i32);

impl MaskLoopOffset {
    /// The `1` entry 109 writes: advance to the next element on every trip of the loop.
    pub const ADVANCE: Self = Self(1);
}

/// A COMPUTE PROVED TO HAVE A PARENT LOOP FOR THE RESTICKIFY DIM — `getParentDimLoop(dimForLoop)`
/// returning `nullptr` made unconstructible.
///
/// ⛔ THE REFERENCE DEREFERENCES IT UNGUARDED (`ddc/ddc_transformation.cpp:2385-2389`): a compute
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
    /// `alloc->allocUsers_.push_back({user, 1})` (`dsc/dsc2.h:1019`).
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
/// then requires two input transfers for a one-input op (`ddc/ddc_transformation.cpp:698-705`).
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
/// (`ddc/ddc_transformation.cpp:830-835`), so iterating the slice is the same walk without the throw.
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
/// (`ddc/ddc_transformation.cpp:2383-2394`). ⛔ The unconditional `return true` is the witness being
/// constructible: the caller only `|=`s it, so [`MaskedComputeSite::of`] carries that answer.
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

// crustify:todo: e242_canUseFifo
//   authority : ddc/ddc_transformation.cpp:15  (270 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::canUseFifo(const dsc2::TransferNode *transferNode, size_t dstIndex, const dsc2::ScheduleNode *consumer) const
//   extract   : crustify-ddc/cpp/ddc.cpp:4373-4644
//   calls     : e117_getNodeDescription

// crustify:todo: e243_canUseLatch
//   authority : ddc/ddc_transformation.cpp:287  (321 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::canUseLatch(const dsc2::TransferNode *transferNode, size_t dstIndex, std::vector<dsc2::ScheduleNode *> consumers) const
//   extract   : crustify-ddc/cpp/ddc.cpp:4654-4976
//   calls     : e117_getNodeDescription

// crustify:todo: e244_cloneForOffsetAdjustment
//   authority : ddc/ddc_transformation.cpp:1386  (8 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::cloneForOffsetAdjustment()
//   extract   : crustify-ddc/cpp/ddc.cpp:4986-4994
//   calls     : e108_cloneComputeForOffsetAdjustment

// crustify:todo: e245_setSizeForFixedSizeTransfers
//   authority : ddc/ddc_transformation.cpp:1731  (34 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::setSizeForFixedSizeTransfers()
//   extract   : crustify-ddc/cpp/ddc.cpp:5004-5038
//   calls     : e104_clear

// crustify:todo: e246_transformForInterSliceRestickify
//   authority : ddc/ddc_transformation.cpp:2398  (28 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::transformForInterSliceRestickify()
//   extract   : crustify-ddc/cpp/ddc.cpp:5048-5076
//   calls     : e109_transformAComputeNodeForInterSliceRestickify

// crustify:todo: e300_packStickDim
//   authority : ddc/ddc_transformation.cpp:811  (544 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::packStickDim()
//   extract   : crustify-ddc/cpp/ddc.cpp:8998-9542
//   calls     : e098_updateValues, e104_clear, e105_isReleventComputeToPackStickDim, e106_findReleventStickPackingTransfers, e107_isEligibleCompute, e112_constructDatastage, e113_constructDatastage, e114_constructLoopNode, e123_updateNodesWithNewLds, e248_splitLoopBandOnDatastage, e251_unrollTransfer, e257_addNewLds

// crustify:todo: e301_hoistTransfersUpForReuse
//   authority : ddc/ddc_transformation.cpp:1432  (230 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::hoistTransfersUpForReuse()
//   extract   : crustify-ddc/cpp/ddc.cpp:9552-9782
//   calls     : e104_clear, e115_collectLoopReferences, e116_getPaddingPerDim, e247_splitLoopBandOnDim, e249_moveTransferNode, e250_convertResultFromFIFOtoReg

// crustify:todo: e302_unrollSymbolicTransfers
//   authority : ddc/ddc_transformation.cpp:1663  (42 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::unrollSymbolicTransfers()
//   extract   : crustify-ddc/cpp/ddc.cpp:9792-9834
//   calls     : e252_unrollTransferForSymbolicDims

// crustify:todo: e303_unrollSpreadTransfers
//   authority : ddc/ddc_transformation.cpp:1706  (24 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::unrollSpreadTransfers()
//   extract   : crustify-ddc/cpp/ddc.cpp:9844-9868
//   calls     : e251_unrollTransfer

// crustify:todo: e338_performPeSfpWorkSplit
//   authority : ddc/ddc_transformation.cpp:1395  (36 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::performPeSfpWorkSplit()
//   extract   : crustify-ddc/cpp/ddc.cpp:12093-12129
//   calls     : e118_cloneForPeSfpWorkSplit, e119_cloneForPeSfpWorkSplit, e304_cloneForPeSfpWorkSplit

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
    use crate::generated::ComputeType;
    use crate::schedule::dsc2::{DataInfo, Dsts, ReplicationFactor};
    use crate::units::NumFolds;

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
            },
        }
    }

    fn compute_on(ex_unit: SenComponent, connect: DataConnect) -> ComputeNode {
        ComputeNode {
            name: NodeName("recip".to_owned()),
            op: ComputeType::Macc,
            ex_unit,
            inputs: vec![operand(ex_unit, Some(connect), Some(0))],
            outputs: vec![operand(ex_unit, Some(connect), Some(0))],
            num_folds_engaged: NumFolds::ONE,
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
            name: NodeName("in".to_owned()),
            src: operand(SenComponent::Lxlu, Some(DataConnect::PeHtOut), Some(0)),
            dsts: Dsts::new(
                operand(SenComponent::Lx, None, Some(0)),
                vec![operand(ex, Some(DataConnect::PeHtOut), Some(0))],
            ),
            replication_factor: ReplicationFactor::ONE,
            unit_time_transfer_chunk_size: Vec::new(),
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
            Some(AllocId(7))
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
