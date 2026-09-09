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

//! `ddc/ddl/Dialect/DdlOps.cpp` — 8 of the campaign's 382 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e165_initialize` | 165 | 0 | 10 | `DdlDialect` | `ddc/ddl/Dialect/DdlOps.cpp:33` |
//! | `e166_verify` | 166 | 0 | 30 | `ForceInnermostDimensionsOp` | `ddc/ddl/Dialect/DdlOps.cpp:59` |
//! | `e167_verify` | 167 | 0 | 25 | `ComputeOp` | `ddc/ddl/Dialect/DdlOps.cpp:109` |
//! | `e168_verify` | 168 | 0 | 16 | `DatatypeOp` | `ddc/ddl/Dialect/DdlOps.cpp:149` |
//! | `e169_verify` | 169 | 0 | 7 | `ImplicitSyncOp` | `ddc/ddl/Dialect/DdlOps.cpp:195` |
//! | `e170_parse` | 170 | 0 | 72 | `AllocateOp` | `ddc/ddl/Dialect/DdlOps.cpp:220` |
//! | `e271_verify` | 271 | 1 | 7 | `OperationBindOp` | `ddc/ddl/Dialect/DdlOps.cpp:178` |
//! | `e272_print` | 272 | 1 | 19 | `AllocateOp` | `ddc/ddl/Dialect/DdlOps.cpp:305` |

use crate::formats::{Bits, DataFormat};
use crate::generated::{ComputeType, StmtKind};

/// AN SSA NAME AS THE TEXT SPELLS IT — the `wrd` of `%wrd`, without the sigil.
///
/// ⭐ A NAME, NOT A CLOSED SET: every template picks its own, so there is nothing to enumerate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SsaName(pub String);

/// ONE SSA VALUE — the pair (base name, result index).
///
/// ⛔⛔ THE INDEX IS PART OF THE IDENTITY. `%wrd:2 = ddl.dimension{}` states TWO results and they
/// are used as `%wrd#0` and `%wrd#1` (`gather.ddl`, and `force_innermost_dimensions(.., %wrd#0,
/// %in)` reads exactly one of them). Comparing base names alone makes them one value, and
/// [`InnermostDims`]'s uniqueness check would then reject a legal template.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value {
    /// The base name.
    pub name: SsaName,
    /// Which result of the defining op — `0` for the bare `%name` form.
    pub result: u32,
}

impl Value {
    /// The `%name` form — result zero.
    #[must_use]
    pub fn sole(name: &str) -> Self {
        Self {
            name: SsaName(name.to_owned()),
            result: 0,
        }
    }
}

/// EVERY OPERATION THE `ddl` DIALECT REGISTERS — `GET_OP_LIST` (`DdlOps.cpp:35-37`), which TableGen
/// fills from every `Ddl_Op<..>` in `DdlOps.td`, in declaration order.
///
/// ⛔⛔ THIRTY-NINE, NOT FORTY: `Ddl_AccessPattenOp` is COMMENTED OUT (`DdlOps.td:514-524`), so
/// `ddl.access_pattern` is not an op at all — an access pattern is stated as `access_pattern_style`
/// on a `ddl.data_transfer` (`:604`) over its `access_pattern_dim` operands (`:608`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DdlOp {
    /// `ddl.yield` — the implicit terminator of every `ImplicitDdlTerminator` region.
    Yield,
    /// `ddl.dimension`.
    Dimension,
    /// `ddl.padded_dimension`.
    PaddedDimension,
    /// `ddl.layout`.
    Layout,
    /// `ddl.type`.
    Type,
    /// `ddl.tensor`.
    Tensor,
    /// `ddl.internal_tensor`.
    InternalTensor,
    /// `ddl.alias_one_tensor_of`.
    AliasOneTensorOf,
    /// `ddl.alias_one_constant_of`.
    AliasOneConstantOf,
    /// `ddl.operation_bind`.
    OperationBind,
    /// `ddl.constraint`.
    Constraint,
    /// `ddl.get_external_data_transfer_allocation`.
    GetExternalDataTransferAllocation,
    /// `ddl.get_external_constant`.
    GetExternalConstant,
    /// `ddl.define_constant`.
    DefineConstant,
    /// `ddl.operand_constant`.
    OperandConstant,
    /// `ddl.dataflow`.
    Dataflow,
    /// `ddl.get_external_datastage`.
    GetExternalDatastage,
    /// `ddl.datastage`.
    Datastage,
    /// `ddl.datastage_constraint`.
    DatastageConstraint,
    /// `ddl.generic_node`.
    GenericNode,
    /// `ddl.allocate`.
    Allocate,
    /// `ddl.force_innermost_dimensions`.
    ForceInnermostDimensions,
    /// `ddl.unit`.
    Unit,
    /// `ddl.data_transfer`.
    DataTransfer,
    /// `ddl.compute`.
    Compute,
    /// `ddl.opaque`.
    Opaque,
    /// `ddl.sync`.
    Sync,
    /// `ddl.implicit_sync`.
    ImplicitSync,
    /// `ddl.loop`.
    Loop,
    /// `ddl.parametric_loop`.
    ParametricLoop,
    /// `ddl.core_to_core_communication`.
    CoreToCoreCommunication,
    /// `ddl.core_corelet_cond`.
    CoreCoreletCond,
    /// `ddl.condition`.
    Condition,
    /// `ddl.condition_and`.
    ConditionAnd,
    /// `ddl.condition_or`.
    ConditionOr,
    /// `ddl.condition_not`.
    ConditionNot,
    /// `ddl.if`.
    If,
    /// `ddl.transformations`.
    Transformations,
    /// `ddl.disable_transfer_promotion`.
    DisableTransferPromotion,
}

impl DdlOp {
    /// THE WHOLE REGISTERED SET, in `DdlOps.td` declaration order.
    pub const REGISTERED: [Self; 39] = [
        Self::Yield,
        Self::Dimension,
        Self::PaddedDimension,
        Self::Layout,
        Self::Type,
        Self::Tensor,
        Self::InternalTensor,
        Self::AliasOneTensorOf,
        Self::AliasOneConstantOf,
        Self::OperationBind,
        Self::Constraint,
        Self::GetExternalDataTransferAllocation,
        Self::GetExternalConstant,
        Self::DefineConstant,
        Self::OperandConstant,
        Self::Dataflow,
        Self::GetExternalDatastage,
        Self::Datastage,
        Self::DatastageConstraint,
        Self::GenericNode,
        Self::Allocate,
        Self::ForceInnermostDimensions,
        Self::Unit,
        Self::DataTransfer,
        Self::Compute,
        Self::Opaque,
        Self::Sync,
        Self::ImplicitSync,
        Self::Loop,
        Self::ParametricLoop,
        Self::CoreToCoreCommunication,
        Self::CoreCoreletCond,
        Self::Condition,
        Self::ConditionAnd,
        Self::ConditionOr,
        Self::ConditionNot,
        Self::If,
        Self::Transformations,
        Self::DisableTransferPromotion,
    ];

    /// THE MNEMONIC WITHOUT THE `ddl.` PREFIX — the string `Ddl_Op<..>` was given.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Yield => "yield",
            Self::Dimension => "dimension",
            Self::PaddedDimension => "padded_dimension",
            Self::Layout => "layout",
            Self::Type => "type",
            Self::Tensor => "tensor",
            Self::InternalTensor => "internal_tensor",
            Self::AliasOneTensorOf => "alias_one_tensor_of",
            Self::AliasOneConstantOf => "alias_one_constant_of",
            Self::OperationBind => "operation_bind",
            Self::Constraint => "constraint",
            Self::GetExternalDataTransferAllocation => "get_external_data_transfer_allocation",
            Self::GetExternalConstant => "get_external_constant",
            Self::DefineConstant => "define_constant",
            Self::OperandConstant => "operand_constant",
            Self::Dataflow => "dataflow",
            Self::GetExternalDatastage => "get_external_datastage",
            Self::Datastage => "datastage",
            Self::DatastageConstraint => "datastage_constraint",
            Self::GenericNode => "generic_node",
            Self::Allocate => "allocate",
            Self::ForceInnermostDimensions => "force_innermost_dimensions",
            Self::Unit => "unit",
            Self::DataTransfer => "data_transfer",
            Self::Compute => "compute",
            Self::Opaque => "opaque",
            Self::Sync => "sync",
            Self::ImplicitSync => "implicit_sync",
            Self::Loop => "loop",
            Self::ParametricLoop => "parametric_loop",
            Self::CoreToCoreCommunication => "core_to_core_communication",
            Self::CoreCoreletCond => "core_corelet_cond",
            Self::Condition => "condition",
            Self::ConditionAnd => "condition_and",
            Self::ConditionOr => "condition_or",
            Self::ConditionNot => "condition_not",
            Self::If => "if",
            Self::Transformations => "transformations",
            Self::DisableTransferPromotion => "disable_transfer_promotion",
        }
    }
}

impl From<StmtKind> for DdlOp {
    /// THE CENSUS AS A REGISTERED OP — total, and compile-checked, so a template that grows
    /// [`StmtKind`] breaks here rather than being walked past.
    ///
    /// ⭐ THREE REGISTERED OPS NO TEMPLATE OF OURS STATES: `ddl.yield` (written implicitly),
    /// `ddl.generic_node` and `ddl.core_corelet_cond`. The census is 36 of the 39.
    fn from(kind: StmtKind) -> Self {
        match kind {
            StmtKind::AliasOneConstantOf => Self::AliasOneConstantOf,
            StmtKind::AliasOneTensorOf => Self::AliasOneTensorOf,
            StmtKind::Allocate => Self::Allocate,
            StmtKind::Compute => Self::Compute,
            StmtKind::Condition => Self::Condition,
            StmtKind::ConditionAnd => Self::ConditionAnd,
            StmtKind::ConditionNot => Self::ConditionNot,
            StmtKind::ConditionOr => Self::ConditionOr,
            StmtKind::Constraint => Self::Constraint,
            StmtKind::CoreToCoreCommunication => Self::CoreToCoreCommunication,
            StmtKind::DataTransfer => Self::DataTransfer,
            StmtKind::Dataflow => Self::Dataflow,
            StmtKind::Datastage => Self::Datastage,
            StmtKind::DatastageConstraint => Self::DatastageConstraint,
            StmtKind::DefineConstant => Self::DefineConstant,
            StmtKind::Dimension => Self::Dimension,
            StmtKind::DisableTransferPromotion => Self::DisableTransferPromotion,
            StmtKind::ForceInnermostDimensions => Self::ForceInnermostDimensions,
            StmtKind::GetExternalConstant => Self::GetExternalConstant,
            StmtKind::GetExternalDataTransferAllocation => Self::GetExternalDataTransferAllocation,
            StmtKind::GetExternalDatastage => Self::GetExternalDatastage,
            StmtKind::If => Self::If,
            StmtKind::ImplicitSync => Self::ImplicitSync,
            StmtKind::InternalTensor => Self::InternalTensor,
            StmtKind::Layout => Self::Layout,
            StmtKind::Loop => Self::Loop,
            StmtKind::Opaque => Self::Opaque,
            StmtKind::OperandConstant => Self::OperandConstant,
            StmtKind::OperationBind => Self::OperationBind,
            StmtKind::PaddedDimension => Self::PaddedDimension,
            StmtKind::ParametricLoop => Self::ParametricLoop,
            StmtKind::Sync => Self::Sync,
            StmtKind::Tensor => Self::Tensor,
            StmtKind::Transformations => Self::Transformations,
            StmtKind::Type => Self::Type,
            StmtKind::Unit => Self::Unit,
        }
    }
}

/// EVERY ATTRIBUTE THE DIALECT REGISTERS — `GET_ATTRDEF_LIST` (`DdlOps.cpp:39-41`).
///
/// ⭐ EXACTLY ONE, AND IT IS A PLACEHOLDER: `DdlDummyAttr` (`DdlTypes.td:60`) wraps a one-case enum
/// `dummy = 0` (`:49`) described as "built-in precisions for ddl", and no vendored template states
/// it. The dialect's real attributes are builtin ones, reached through
/// `useDefaultAttributePrinterParser = 1` (`DdlTypes.td:30`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DdlAttr {
    /// `#ddl.dummy<dummy>`.
    Dummy,
}

/// THE REGISTERED `ddl` DIALECT — what `initialize()` leaves behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dialect {
    /// `addOperations<GET_OP_LIST>`.
    pub operations: [DdlOp; 39],
    /// `addAttributes<GET_ATTRDEF_LIST>`.
    pub attributes: [DdlAttr; 1],
}

impl Dialect {
    /// Replaces: e165_initialize
    ///
    /// THE REGISTRATION AS A VALUE — 39 operations and 1 attribute under the dialect named `ddl`
    /// (`DdlTypes.td:28`).
    ///
    /// ⛔ REGISTRATION IS NOT DECORATION: it is what [`Self::resolve`] answers from, and the reason
    /// an unregistered mnemonic has no op rather than a generic one.
    #[must_use]
    pub const fn initialize() -> Self {
        Self {
            operations: DdlOp::REGISTERED,
            attributes: [DdlAttr::Dummy],
        }
    }

    /// WHICH OP A MNEMONIC NAMES, or [`None`] where the dialect registered none — the effect
    /// `allowUnregisteredDialects(false)` gives the registration.
    #[must_use]
    pub fn resolve(&self, mnemonic: &str) -> Option<DdlOp> {
        self.operations
            .iter()
            .copied()
            .find(|op| op.mnemonic() == mnemonic)
    }
}

/// WHAT DEFINES EACH SSA VALUE — the `getDefiningOp()` the verifiers reach through.
///
/// ⭐ A TRAIT, BECAUSE THE MECHANISM IS DROPPABLE. MLIR answers this by walking the use-def chain;
/// every verifier here only ever asks WHICH op it was.
pub trait Defining {
    /// The op defining `value`, or [`None`] where nothing in the module does.
    fn defining_op(&self, value: &Value) -> Option<DdlOp>;
}

impl Defining for [(Value, DdlOp)] {
    fn defining_op(&self, value: &Value) -> Option<DdlOp> {
        self.iter()
            .find(|(defined, _)| defined == value)
            .map(|&(_, op)| op)
    }
}

/// AN ALLOCATION THIS DIALECT CONTROLS — a value defined by `ddl.allocate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocateValue(Value);

impl AllocateValue {
    /// The allocation, or [`None`] where `value` is not a `ddl.allocate` result.
    #[must_use]
    pub fn of(defining: &(impl Defining + ?Sized), value: Value) -> Option<Self> {
        match defining.defining_op(&value)? {
            DdlOp::Allocate => Some(Self(value)),
            _ => None,
        }
    }

    /// The value it wraps.
    #[must_use]
    pub const fn value(&self) -> &Value {
        &self.0
    }
}

/// A DATASTAGE — `ddl.datastage` or `ddl.get_external_datastage`, and WHICH is kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatastageValue {
    /// `ddl.datastage`.
    Datastage(Value),
    /// `ddl.get_external_datastage`.
    External(Value),
}

impl DatastageValue {
    /// The datastage, or [`None`] where `value` is neither.
    #[must_use]
    pub fn of(defining: &(impl Defining + ?Sized), value: Value) -> Option<Self> {
        match defining.defining_op(&value)? {
            DdlOp::Datastage => Some(Self::Datastage(value)),
            DdlOp::GetExternalDatastage => Some(Self::External(value)),
            _ => None,
        }
    }

    /// The value it wraps.
    #[must_use]
    pub const fn value(&self) -> &Value {
        match self {
            Self::Datastage(value) | Self::External(value) => value,
        }
    }
}

/// A DIMENSION — `ddl.dimension` or `ddl.padded_dimension`, and WHICH is kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionValue {
    /// `ddl.dimension`.
    Dimension(Value),
    /// `ddl.padded_dimension`.
    Padded(Value),
}

impl DimensionValue {
    /// The dimension, or [`None`] where `value` is neither.
    #[must_use]
    pub fn of(defining: &(impl Defining + ?Sized), value: Value) -> Option<Self> {
        match defining.defining_op(&value)? {
            DdlOp::Dimension => Some(Self::Dimension(value)),
            DdlOp::PaddedDimension => Some(Self::Padded(value)),
            _ => None,
        }
    }

    /// The value it wraps.
    #[must_use]
    pub const fn value(&self) -> &Value {
        match self {
            Self::Dimension(value) | Self::Padded(value) => value,
        }
    }
}

/// THE DIMENSIONS TO FORCE INNERMOST, IN ORDER — non-empty and pairwise distinct BY CONSTRUCTION.
///
/// ⛔ `{ first, rest }` RATHER THAN A `Vec`, so "provide at least one dimension" (`DdlOps.cpp:71`)
/// cannot be re-asked downstream: there is no way to spell the empty list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnermostDims {
    first: DimensionValue,
    rest: Vec<DimensionValue>,
}

impl InnermostDims {
    /// The list, or [`None`] when it is empty or names one dimension twice.
    #[must_use]
    pub fn of(dims: Vec<DimensionValue>) -> Option<Self> {
        let mut dims = dims.into_iter();
        let first = dims.next()?;
        let mut rest: Vec<DimensionValue> = Vec::new();
        for dim in dims {
            let seen = dim.value() == first.value()
                || rest.iter().any(|earlier| earlier.value() == dim.value());
            if seen {
                return None;
            }
            rest.push(dim);
        }
        Some(Self { first, rest })
    }

    /// Every dimension, in the order the op states them.
    pub fn iter(&self) -> impl Iterator<Item = &DimensionValue> {
        core::iter::once(&self.first).chain(self.rest.iter())
    }
}

/// Replaces: e166_verify
///
/// A `ddl.force_innermost_dimensions` WHOSE FIVE CHECKS HAVE ALL PASSED — passed into the TYPES, so
/// nothing downstream can re-ask them.
///
/// ⛔ THE `isa<>` CALLS SIT ON A POSSIBLY-NULL `getDefiningOp()` (`:60`, `:65`, `:76`), which ASSERTS
/// rather than answering false. Unreachable in practice — every DDL region is `AnyRegion` with no
/// block arguments, so every operand has a defining op — which is what makes
/// [`Defining::defining_op`] answering [`None`] a safe widening rather than a behaviour change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForceInnermostDimensions {
    /// `$allocate`.
    pub allocate: AllocateValue,
    /// `$datastage`.
    pub datastage: DatastageValue,
    /// `$dims`.
    pub dims: InnermostDims,
}

impl ForceInnermostDimensions {
    /// The verified op, or [`None`] where any of the five checks rejects.
    #[must_use]
    pub fn verify(
        defining: &(impl Defining + ?Sized),
        allocate: Value,
        datastage: Value,
        dims: Vec<Value>,
    ) -> Option<Self> {
        let allocate = AllocateValue::of(defining, allocate)?;
        let datastage = DatastageValue::of(defining, datastage)?;
        let dims = dims
            .into_iter()
            .map(|dim| DimensionValue::of(defining, dim))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            allocate,
            datastage,
            dims: InnermostDims::of(dims)?,
        })
    }
}

/// WHICH COMPUTE THE DIALECT RECOGNISES — the thirty-three names `ComputeOp::verify` tests
/// (`DdlOps.cpp:119-128`), listed in the vendor's own order and grouped by required arity.
///
/// ⛔⛔ NINE OF THEM THE CENSUS CANNOT SPELL, which is why this is not [`ComputeType`]: no vendored
/// template states `FMA8`, `FMA4`, `IMA8`, `IMA4`, `SHR`, `FSUB`, `GCVT`, `IME` or `AND`, and
/// `sys-arch-spec` has no compute-op enum at all. A port that reasoned in the census would drop
/// nine recognised computes without saying so; [`From<ComputeType>`] is the join the other way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DdlComputeType {
    /// `MACC`.
    Macc,
    /// `FMA16`.
    Fma16,
    /// `FMA8`.
    Fma8,
    /// `FMA4`.
    Fma4,
    /// `IMA8`.
    Ima8,
    /// `IMA4`.
    Ima4,
    /// `FNMS`.
    Fnms,
    /// `SELECT`.
    Select,
    /// `FMA32`.
    Fma32,
    /// `REDUCE`.
    Reduce,
    /// `SPLAT`.
    Splat,
    /// `ICVT`.
    Icvt,
    /// `FEST`.
    Fest,
    /// `SHR`.
    Shr,
    /// `SHUFFLE`.
    Shuffle,
    /// `ASSIGN`.
    Assign,
    /// `FLOOR`.
    Floor,
    /// `PACKMERGE`.
    Packmerge,
    /// `FMUL`.
    Fmul,
    /// `FSUB`.
    Fsub,
    /// `FMAX`.
    Fmax,
    /// `FMIN`.
    Fmin,
    /// `FABSMAX`.
    Fabsmax,
    /// `GCVT`.
    Gcvt,
    /// `IME`.
    Ime,
    /// `GREATERTHAN`.
    Greaterthan,
    /// `LESSERTHAN`.
    Lesserthan,
    /// `EQUAL`.
    Equal,
    /// `NOTEQUAL`.
    Notequal,
    /// `GREATEREQUAL`.
    Greaterequal,
    /// `LESSEREQUAL`.
    Lesserequal,
    /// `OR`.
    Or,
    /// `AND`.
    And,
}

impl DdlComputeType {
    /// THE COMPUTE A `computetype=` NAMES, or [`None`] for the "Unrecognized compute op" arm
    /// (`DdlOps.cpp:131`).
    ///
    /// ⛔ THE SPELLING IS UPPERCASED FIRST — `getComputetype().upper()` (`:110`) — because
    /// `unary_parallel.ddl` states `computetype="assign"` in lowercase and a case-sensitive match
    /// would reject it.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        Some(match spelling.to_ascii_uppercase().as_str() {
            "MACC" => Self::Macc,
            "FMA16" => Self::Fma16,
            "FMA8" => Self::Fma8,
            "FMA4" => Self::Fma4,
            "IMA8" => Self::Ima8,
            "IMA4" => Self::Ima4,
            "FNMS" => Self::Fnms,
            "SELECT" => Self::Select,
            "FMA32" => Self::Fma32,
            "REDUCE" => Self::Reduce,
            "SPLAT" => Self::Splat,
            "ICVT" => Self::Icvt,
            "FEST" => Self::Fest,
            "SHR" => Self::Shr,
            "SHUFFLE" => Self::Shuffle,
            "ASSIGN" => Self::Assign,
            "FLOOR" => Self::Floor,
            "PACKMERGE" => Self::Packmerge,
            "FMUL" => Self::Fmul,
            "FSUB" => Self::Fsub,
            "FMAX" => Self::Fmax,
            "FMIN" => Self::Fmin,
            "FABSMAX" => Self::Fabsmax,
            "GCVT" => Self::Gcvt,
            "IME" => Self::Ime,
            "GREATERTHAN" => Self::Greaterthan,
            "LESSERTHAN" => Self::Lesserthan,
            "EQUAL" => Self::Equal,
            "NOTEQUAL" => Self::Notequal,
            "GREATEREQUAL" => Self::Greaterequal,
            "LESSEREQUAL" => Self::Lesserequal,
            "OR" => Self::Or,
            "AND" => Self::And,
            _ => return None,
        })
    }

    /// HOW MANY INPUTS IT REQUIRES — the argument each `compute_op_checks(..)` is given.
    #[must_use]
    pub const fn arity(self) -> ComputeArity {
        match self {
            Self::Macc
            | Self::Fma16
            | Self::Fma8
            | Self::Fma4
            | Self::Ima8
            | Self::Ima4
            | Self::Fnms
            | Self::Select
            | Self::Fma32 => ComputeArity::Ternary,
            Self::Reduce
            | Self::Splat
            | Self::Icvt
            | Self::Fest
            | Self::Shr
            | Self::Shuffle
            | Self::Assign
            | Self::Floor => ComputeArity::Unary,
            Self::Packmerge
            | Self::Fmul
            | Self::Fsub
            | Self::Fmax
            | Self::Fmin
            | Self::Fabsmax
            | Self::Gcvt
            | Self::Ime
            | Self::Greaterthan
            | Self::Lesserthan
            | Self::Equal
            | Self::Notequal
            | Self::Greaterequal
            | Self::Lesserequal
            | Self::Or
            | Self::And => ComputeArity::Binary,
        }
    }
}

impl From<ComputeType> for DdlComputeType {
    /// THE CENSUS AS A RECOGNISED COMPUTE — total, and compile-checked, so a template that grows
    /// [`ComputeType`] breaks here rather than being walked past. 24 of the 33.
    fn from(compute_type: ComputeType) -> Self {
        match compute_type {
            ComputeType::Assign => Self::Assign,
            ComputeType::Equal => Self::Equal,
            ComputeType::Fabsmax => Self::Fabsmax,
            ComputeType::Fest => Self::Fest,
            ComputeType::Floor => Self::Floor,
            ComputeType::Fma16 => Self::Fma16,
            ComputeType::Fma32 => Self::Fma32,
            ComputeType::Fmax => Self::Fmax,
            ComputeType::Fmin => Self::Fmin,
            ComputeType::Fmul => Self::Fmul,
            ComputeType::Fnms => Self::Fnms,
            ComputeType::Greaterequal => Self::Greaterequal,
            ComputeType::Greaterthan => Self::Greaterthan,
            ComputeType::Icvt => Self::Icvt,
            ComputeType::Lesserequal => Self::Lesserequal,
            ComputeType::Lesserthan => Self::Lesserthan,
            ComputeType::Macc => Self::Macc,
            ComputeType::Notequal => Self::Notequal,
            ComputeType::Or => Self::Or,
            ComputeType::Packmerge => Self::Packmerge,
            ComputeType::Reduce => Self::Reduce,
            ComputeType::Select => Self::Select,
            ComputeType::Shuffle => Self::Shuffle,
            ComputeType::Splat => Self::Splat,
        }
    }
}

/// HOW MANY INPUTS A COMPUTE REQUIRES — the three counts `compute_op_checks` is ever called with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeArity {
    /// One input.
    Unary,
    /// Two.
    Binary,
    /// Three.
    Ternary,
}

/// THE INPUTS A `ddl.compute` READS — the arity IS the array length, so a count check downstream
/// has nothing left to check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeInputs {
    /// One input.
    Unary([Value; 1]),
    /// Two.
    Binary([Value; 2]),
    /// Three.
    Ternary([Value; 3]),
}

impl ComputeInputs {
    /// The inputs, or [`None`] where there are not exactly `arity` of them.
    #[must_use]
    pub fn of(arity: ComputeArity, inputs: Vec<Value>) -> Option<Self> {
        Some(match arity {
            ComputeArity::Unary => Self::Unary(<[Value; 1]>::try_from(inputs).ok()?),
            ComputeArity::Binary => Self::Binary(<[Value; 2]>::try_from(inputs).ok()?),
            ComputeArity::Ternary => Self::Ternary(<[Value; 3]>::try_from(inputs).ok()?),
        })
    }

    /// Every input, in order.
    #[must_use]
    pub fn as_slice(&self) -> &[Value] {
        match self {
            Self::Unary(inputs) => inputs,
            Self::Binary(inputs) => inputs,
            Self::Ternary(inputs) => inputs,
        }
    }
}

/// Replaces: e167_verify
///
/// A `ddl.compute` WHOSE `computetype=` IS RECOGNISED AND WHOSE INPUT COUNT MATCHES IT.
///
/// ⭐ THIRTY-THREE NAMES IN THREE ARITY GROUPS, and the reference's own error text names the
/// requirement — "<OP> requires exactly N inputs" (`DdlOps.cpp:113`). Here the requirement is the
/// length of [`ComputeInputs`]'s array, so it holds by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compute {
    /// `computetype=`, resolved.
    pub compute_type: DdlComputeType,
    /// `$inputs`.
    pub inputs: ComputeInputs,
}

impl Compute {
    /// The verified op, or [`None`] where the name is unrecognised or the arity disagrees.
    #[must_use]
    pub fn verify(computetype: &str, inputs: Vec<Value>) -> Option<Self> {
        let compute_type = DdlComputeType::from_spelling(computetype)?;
        Some(Self {
            inputs: ComputeInputs::of(compute_type.arity(), inputs)?,
            compute_type,
        })
    }
}

/// A PACKING WIDTH IN BITS — `bit_width=` on a `ddl.type`, the width the element is STORED in.
///
/// ⛔ NOT [`Bits`], which is the width the element IS. `ddl.type {data_type="SEN143_FP8",
/// bit_width=16}` is an eight-bit format kept in a sixteen-bit slot, and confusing the two doubles
/// or halves every address derived from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StorageBits(pub u32);

/// Replaces: e168_verify
///
/// A `ddl.type` WHOSE FORMAT IS NAMED AND WHOSE `bit_width=` OVERRIDE IS WIDE ENOUGH FOR IT.
///
/// ⛔⛔ THE REFERENCE ACCEPTS ANY UNKNOWN SPELLING, and its first error is UNREACHABLE:
/// `FromString` answers `INVALID`, `dataFormatsToBitWidth` maps `INVALID -> -1`
/// (`sendefs.cpp:131`), so `find() == end()` never fires and `-1 > value_or(100)` is false — a typo
/// becomes a type of width −1. DELIBERATE DIVERGENCE: [`DataFormat`] has no `INVALID`, so an
/// unspelled format has no value here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Datatype {
    /// `data_type=`, resolved.
    pub data_type: DataFormat,
    /// `bit_width=`, absent where the format's own width stands.
    pub bit_width: Option<StorageBits>,
}

impl Datatype {
    /// The verified op, or [`None`] where the spelling names no format or the override is narrower
    /// than one element.
    #[must_use]
    pub fn verify(data_type: &str, bit_width: Option<StorageBits>) -> Option<Self> {
        let data_type = DataFormat::from_spelling(data_type)?;
        let Bits(element) = data_type.bits();
        // `type_size > this->getBitWidth().value_or(100)` (`DdlOps.cpp:157`). No format reaches 100,
        // so an ABSENT override always passes — and `bit_width` is an `SI64Attr` there, so every
        // NEGATIVE override is rejected; [`StorageBits`] being unsigned makes those unspellable.
        (element <= bit_width.map_or(100, |StorageBits(bits)| bits)).then_some(Self {
            data_type,
            bit_width,
        })
    }
}

/// Replaces: e169_verify
///
/// A `ddl.implicit_sync` WHOSE INPUT IS AN ALLOCATION — the one check (`DdlOps.cpp:196`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitSync {
    /// `$allocate`.
    pub allocate: AllocateValue,
}

impl ImplicitSync {
    /// The verified op, or [`None`] where the input is not a `ddl.allocate` result.
    #[must_use]
    pub fn verify(defining: &(impl Defining + ?Sized), allocate: Value) -> Option<Self> {
        Some(Self {
            allocate: AllocateValue::of(defining, allocate)?,
        })
    }
}

/// THE OPERANDS OF A `ddl.allocate`, in the order the parse resolves them into `result.operands`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocateOperands {
    /// `$tensor` — always exactly one.
    pub tensor: Value,
    /// `$padding_dim` — the dimensions needing a non-default padding type.
    pub padding_dim: Vec<Value>,
    /// `$external_allocate` — at most one.
    pub external_allocate: Option<Value>,
}

impl AllocateOperands {
    /// Replaces: e170_parse
    ///
    /// THE OPERAND SYNTAX OF A `ddl.allocate` — `(` `%tensor` (`,` `[`dims`]`)? (`,` `%ext`?)? `)` —
    /// and the unconsumed tail, which starts at `attr-dict`.
    ///
    /// ⛔ ONE `,` MAY INTRODUCE EITHER THE LIST OR THE EXTERNAL, which is what
    /// `commaNeedsFurtherParsing` (`:237-255`) is for: `(%kertensor, %kertensor_xrf_ext_allocation)`
    /// parses with an EMPTY padding list, and `bmm.ddl` states exactly that form.
    ///
    /// ⛔ `attr-dict` IS THE FRAMEWORK'S generic parse (`:272`), so it is handed back, not invented.
    #[must_use]
    pub fn parse(input: &str) -> Option<(Self, &str)> {
        let mut rest = punct(input, '(')?;
        let (tensor, after_tensor) = operand(rest)?;
        rest = after_tensor;

        let mut padding_dim = Vec::new();
        let mut comma_needs_further_parsing = false;
        if let Some(after_comma) = punct(rest, ',') {
            if let Some(after_lsquare) = punct(after_comma, '[') {
                let (dims, after_dims) = operand_list(after_lsquare)?;
                padding_dim = dims;
                rest = punct(after_dims, ']')?;
            } else {
                comma_needs_further_parsing = true;
                rest = after_comma;
            }
        }

        let mut external_allocate = None;
        let after_second_comma = if comma_needs_further_parsing {
            Some(rest)
        } else {
            punct(rest, ',')
        };
        if let Some(after) = after_second_comma {
            match operand(after) {
                Some((value, tail)) => {
                    external_allocate = Some(value);
                    rest = tail;
                }
                None => rest = after,
            }
        }

        let tail = punct(rest, ')')?;
        Some((
            Self {
                tensor,
                padding_dim,
                external_allocate,
            },
            tail,
        ))
    }

    /// `operandSegmentSizes` — `{1, |padding_dim|, |external_allocate|}` (`DdlOps.cpp:275-279`),
    /// which the reference stores as a `DenseI32ArrayAttr` and `AttrSizedOperandSegments` reads back
    /// to re-split the flat operand list.
    #[must_use]
    pub fn operand_segment_sizes(&self) -> [usize; 3] {
        [
            1,
            self.padding_dim.len(),
            usize::from(self.external_allocate.is_some()),
        ]
    }
}

/// The input past `c`, or [`None`] where `c` is not the next token.
fn punct(input: &str, c: char) -> Option<&str> {
    input.trim_start().strip_prefix(c)
}

/// `%name` or `%name#N` — `parseOperand`, whose result index is part of the value's identity.
fn operand(input: &str) -> Option<(Value, &str)> {
    let after_sigil = input.trim_start().strip_prefix('%')?;
    let end = after_sigil
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.' | '-')))
        .unwrap_or(after_sigil.len());
    if end == 0 {
        return None;
    }
    let (name, mut rest) = after_sigil.split_at(end);
    let mut result = 0;
    if let Some(after_hash) = rest.strip_prefix('#') {
        let digits_end = after_hash
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after_hash.len());
        if digits_end == 0 {
            return None;
        }
        let (digits, tail) = after_hash.split_at(digits_end);
        result = digits.parse().ok()?;
        rest = tail;
    }
    Some((
        Value {
            name: SsaName(name.to_owned()),
            result,
        },
        rest,
    ))
}

/// `parseOperandList` — zero or more comma-separated operands, and a `,` MUST be followed by one.
fn operand_list(input: &str) -> Option<(Vec<Value>, &str)> {
    let mut out = Vec::new();
    let Some((first, mut rest)) = operand(input) else {
        return Some((out, input));
    };
    out.push(first);
    loop {
        let Some(after_comma) = punct(rest, ',') else {
            return Some((out, rest));
        };
        let (next, tail) = operand(after_comma)?;
        out.push(next);
        rest = tail;
    }
}

/// ONE OP AS PARSED, BEFORE ITS VERIFIER HAS RUN — the only place a `computetype=` or `data_type=`
/// is still a STRING, which is exactly what its verifier resolves.
///
/// ⛔ FOUR OF THE FIVE OPS WITH `hasVerifier = 1`. `ddl.operation_bind`'s verifier
/// (`DdlOps.cpp:178`) is entry 271 and is NOT in this batch; until it lands, an `operation_bind`
/// passes [`Verified::of`] unchecked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unverified {
    /// `ddl.force_innermost_dimensions`.
    ForceInnermostDimensions {
        /// `$allocate`.
        allocate: Value,
        /// `$datastage`.
        datastage: Value,
        /// `$dims`.
        dims: Vec<Value>,
    },
    /// `ddl.compute`.
    Compute {
        /// `computetype=`, as spelled.
        computetype: String,
        /// `$inputs`.
        inputs: Vec<Value>,
    },
    /// `ddl.type`.
    Datatype {
        /// `data_type=`, as spelled.
        data_type: String,
        /// `bit_width=`.
        bit_width: Option<StorageBits>,
    },
    /// `ddl.implicit_sync`.
    ImplicitSync {
        /// `$allocate`.
        allocate: Value,
    },
}

impl Unverified {
    /// This op's verifier, or [`None`] where it rejects.
    #[must_use]
    pub fn verify(self, defining: &(impl Defining + ?Sized)) -> Option<VerifiedOp> {
        Some(match self {
            Self::ForceInnermostDimensions {
                allocate,
                datastage,
                dims,
            } => VerifiedOp::ForceInnermostDimensions(ForceInnermostDimensions::verify(
                defining, allocate, datastage, dims,
            )?),
            Self::Compute {
                computetype,
                inputs,
            } => VerifiedOp::Compute(Compute::verify(&computetype, inputs)?),
            Self::Datatype {
                data_type,
                bit_width,
            } => VerifiedOp::Datatype(Datatype::verify(&data_type, bit_width)?),
            Self::ImplicitSync { allocate } => {
                VerifiedOp::ImplicitSync(ImplicitSync::verify(defining, allocate)?)
            }
        })
    }
}

/// ONE VERIFIED OP — the witness its verifier produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiedOp {
    /// `ddl.force_innermost_dimensions`.
    ForceInnermostDimensions(ForceInnermostDimensions),
    /// `ddl.compute`.
    Compute(Compute),
    /// `ddl.type`.
    Datatype(Datatype),
    /// `ddl.implicit_sync`.
    ImplicitSync(ImplicitSync),
}

/// A DDL MODULE WHOSE OPS HAVE ALL BEEN VERIFIED — what `verifyAfterParse=true` leaves behind, and
/// the only thing the four ported verifiers can be reached through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified(Vec<VerifiedOp>);

impl Verified {
    /// EVERY VERIFIER, OR NOTHING — one `emitError` anywhere fails the whole parse.
    #[must_use]
    pub fn of(defining: &(impl Defining + ?Sized), ops: Vec<Unverified>) -> Option<Self> {
        let mut verified = Vec::with_capacity(ops.len());
        for op in ops {
            verified.push(op.verify(defining)?);
        }
        Some(Self(verified))
    }

    /// The verified ops, in the order the module states them.
    #[must_use]
    pub fn ops(&self) -> &[VerifiedOp] {
        &self.0
    }
}

// crustify:todo: e271_verify
//   authority : ddc/ddl/Dialect/DdlOps.cpp:178  (7 body lines, level 1)
//   class     : OperationBindOp
//   original  : LogicalResult OperationBindOp::verify()
//   extract   : crustify-ddc/cpp/ddl.cpp:790-797
//   calls     : e021_getOpFuncName

// crustify:todo: e272_print
//   authority : ddc/ddl/Dialect/DdlOps.cpp:305  (19 body lines, level 1)
//   class     : AllocateOp
//   original  : void AllocateOp::print(::mlir::OpAsmPrinter& _odsPrinter)
//   extract   : crustify-ddc/cpp/ddl.cpp:807-826
//   calls     : e176_getTensor

#[cfg(test)]
mod tests_e165_e170 {
    use super::{
        AllocateOperands, Compute, ComputeArity, ComputeInputs, Datatype, DdlComputeType, DdlOp,
        Dialect, ForceInnermostDimensions, ImplicitSync, StorageBits, Value,
    };
    use crate::formats::DataFormat;
    use crate::generated::{ComputeType, StmtKind};

    /// ⭐ THE REGISTERED SET, AND ITS EDGE: 39 ops, 1 attribute, and `access_pattern` is NOT one.
    ///
    /// ⛔ CARRIED AS THE MNEMONICS RESOLVING, not as `operations.len() == 39`: a list holding the
    /// same op 39 times has the right length.
    #[test]
    fn the_dialect_registers_thirty_nine_ops_and_one_attribute() {
        let dialect = Dialect::initialize();
        assert_eq!(dialect.attributes.len(), 1);
        for op in DdlOp::REGISTERED {
            assert_eq!(dialect.resolve(op.mnemonic()), Some(op), "{op:?}");
        }
        // ⛔ COMMENTED OUT AT `DdlOps.td:514`, so nothing registers it.
        assert_eq!(dialect.resolve("access_pattern"), None);
        // ⭐ AND A CENSUSED MNEMONIC RESOLVES TO THE OP THE JOIN NAMES, ON THE SPELLING.
        // ⛔ `StmtKind::mnemonic` CARRIES THE `ddl.` PREFIX and `DdlOp::mnemonic` does not — the
        // latter is the string `Ddl_Op<..>` was given, which is the name without the dialect.
        let kind = StmtKind::ForceInnermostDimensions;
        assert_eq!(kind.mnemonic(), "ddl.force_innermost_dimensions");
        assert_eq!(
            dialect.resolve(kind.mnemonic().trim_start_matches("ddl.")),
            Some(DdlOp::from(kind))
        );
    }

    /// The `bmm.ddl` operand shape: an allocation, a datastage, and one `ddl.dimension`.
    fn bmm_defining() -> Vec<(Value, DdlOp)> {
        vec![
            (Value::sole("kertensor_xrf_allocation"), DdlOp::Allocate),
            (Value::sole("bottom_datastage"), DdlOp::Datastage),
            (Value::sole("in"), DdlOp::Dimension),
            (Value::sole("mb"), DdlOp::Dimension),
            (Value::sole("not_a_dimension"), DdlOp::Layout),
        ]
    }

    /// ⭐ THE VENDORED CASE — `force_innermost_dimensions(%kertensor_xrf_allocation,
    /// %bottom_datastage, %in)` (`bmm.ddl`) — AND THE THREE WAYS IT IS REJECTED.
    #[test]
    fn force_innermost_dimensions_takes_distinct_dimensions_of_an_allocation() {
        let defining = bmm_defining();
        let verified = ForceInnermostDimensions::verify(
            defining.as_slice(),
            Value::sole("kertensor_xrf_allocation"),
            Value::sole("bottom_datastage"),
            vec![Value::sole("in"), Value::sole("mb")],
        )
        .expect("the vendored form verifies");
        assert_eq!(verified.dims.iter().count(), 2);

        // ⛔ EMPTY, DUPLICATED, AND NOT-A-DIMENSION each reject.
        for dims in [
            vec![],
            vec![Value::sole("in"), Value::sole("in")],
            vec![Value::sole("not_a_dimension")],
        ] {
            assert_eq!(
                ForceInnermostDimensions::verify(
                    defining.as_slice(),
                    Value::sole("kertensor_xrf_allocation"),
                    Value::sole("bottom_datastage"),
                    dims.clone(),
                ),
                None,
                "{dims:?}"
            );
        }
        // ⛔ AND THE ALLOCATE OPERAND MUST BE AN ALLOCATION, not just any value.
        assert_eq!(
            ForceInnermostDimensions::verify(
                defining.as_slice(),
                Value::sole("bottom_datastage"),
                Value::sole("bottom_datastage"),
                vec![Value::sole("in")],
            ),
            None
        );
    }

    /// ⭐ THE ARITY IS THE ARRAY LENGTH, and the lowercase `assign` the templates state resolves.
    ///
    /// ⛔ CARRIED PER GROUP, not as one call: the three groups are three different literals in the
    /// reference and a single sample cannot tell them apart.
    #[test]
    fn a_compute_takes_exactly_the_inputs_its_type_requires() {
        let ins = |n: usize| (0..n).map(|i| Value::sole(&format!("v{i}"))).collect();
        assert_eq!(
            Compute::verify("FMA16", ins(3)).map(|op| op.inputs),
            Some(ComputeInputs::Ternary([
                Value::sole("v0"),
                Value::sole("v1"),
                Value::sole("v2")
            ]))
        );
        assert_eq!(DdlComputeType::Reduce.arity(), ComputeArity::Unary);
        assert_eq!(DdlComputeType::Fmul.arity(), ComputeArity::Binary);
        // ⭐ `unary_parallel.ddl` states `computetype="assign"` in LOWERCASE.
        assert_eq!(
            Compute::verify("assign", ins(1)).map(|op| op.compute_type),
            Some(DdlComputeType::Assign)
        );
        // ⛔ THE WRONG COUNT AND AN UNRECOGNISED NAME both reject.
        assert_eq!(Compute::verify("FMA16", ins(2)), None);
        assert_eq!(Compute::verify("FMA17", ins(2)), None);
        // ⛔ AND NINE RECOGNISED COMPUTES THE CENSUS CANNOT SPELL — `From<ComputeType>` reaches 24.
        assert_eq!(
            DdlComputeType::from_spelling("GCVT"),
            Some(DdlComputeType::Gcvt)
        );
        assert_eq!(
            DdlComputeType::from(ComputeType::Macc),
            DdlComputeType::Macc
        );
    }

    /// ⭐ THE VENDORED PAIR — `{data_type="SEN143_FP8", bit_width=16}` and `{data_type="BOOL"}`.
    ///
    /// ⛔ AND THE DIVERGENCE THIS PORT MAKES DELIBERATELY: an unknown spelling has no value here,
    /// where the reference gives it `INVALID` and a width of −1 and accepts it.
    #[test]
    fn a_type_overrides_its_width_only_upward() {
        assert_eq!(
            Datatype::verify("SEN143_FP8", Some(StorageBits(16))).map(|op| op.data_type),
            Some(DataFormat::Sen143Fp8)
        );
        assert_eq!(
            Datatype::verify("BOOL", None).map(|op| op.bit_width),
            Some(None)
        );
        // ⛔ NARROWER THAN ONE ELEMENT REJECTS.
        assert_eq!(Datatype::verify("SEN143_FP8", Some(StorageBits(4))), None);
        // ⛔ AND AN UNSPELLED FORMAT HAS NO TYPE — the reference would accept this.
        assert_eq!(Datatype::verify("SEN143_FP9", None), None);
    }

    /// ⭐ `implicit_sync(%inptensor_l0_allocation)` (`bmm.ddl`), and its one rejection.
    #[test]
    fn implicit_sync_takes_an_allocation() {
        let defining = bmm_defining();
        assert!(
            ImplicitSync::verify(defining.as_slice(), Value::sole("kertensor_xrf_allocation"))
                .is_some()
        );
        assert_eq!(
            ImplicitSync::verify(defining.as_slice(), Value::sole("in")),
            None
        );
    }

    /// ⭐⭐ ALL FOUR OPERAND FORMS THE TEMPLATES STATE, carried as the `operandSegmentSizes` each
    /// produces — which is the value `AttrSizedOperandSegments` re-splits the flat list with.
    ///
    /// ⛔ THE `(%t, %ext)` FORM IS THE ONE `commaNeedsFurtherParsing` EXISTS FOR: one comma, no
    /// `[..]`, and the operand after it is the EXTERNAL allocation, not a padding dimension.
    #[test]
    fn allocate_parses_every_vendored_operand_form() {
        for (text, segments, tail) in [
            (
                "(%a_const_reg) {memory=\"sfplrf\"}",
                [1, 0, 0],
                " {memory=\"sfplrf\"}",
            ),
            ("(%inptensor, [%j_pad, %krdpad0])", [1, 2, 0], ""),
            ("(%kertensor, %kertensor_xrf_ext_allocation)", [1, 0, 1], ""),
            ("(%scale_const, [], %external_allocate)", [1, 0, 1], ""),
        ] {
            let (parsed, rest) = AllocateOperands::parse(text).expect(text);
            assert_eq!(parsed.operand_segment_sizes(), segments, "{text}");
            assert_eq!(rest, tail, "{text}");
        }
        // ⛔ AND `%wrd#0` IS RESULT ONE OF TWO — `%wrd` alone is a different value.
        let (parsed, _) = AllocateOperands::parse("(%wrd#1)").expect("a result index parses");
        assert_eq!(parsed.tensor.result, 1);
        assert_ne!(parsed.tensor, Value::sole("wrd"));
    }

    /// ⛔ A `,` INSIDE THE LIST MUST BE FOLLOWED BY AN OPERAND, and the parens must close —
    /// `parseCommaSeparatedList` propagates the element parse's failure rather than stopping short.
    #[test]
    fn allocate_rejects_a_comma_that_introduces_nothing() {
        assert_eq!(AllocateOperands::parse("(%t, [%a,])"), None);
        assert_eq!(AllocateOperands::parse("(%t, [%a]"), None);
        assert_eq!(AllocateOperands::parse("%t"), None);
    }
}
