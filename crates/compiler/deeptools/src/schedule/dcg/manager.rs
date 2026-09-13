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

//! `dcg/dcg_manager/dcg_manager.cpp`, `dcg/dcg_manager/dcg_manager.h` — 16 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e188_runDcgComputeTransfer` | 188 | 0 | 13 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:112` |
//! | `e189_runDcgForDlOps` | 189 | 0 | 52 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:216` |
//! | `e190_runDcgForDlOpsStandalone` | 190 | 0 | 64 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:449` |
//! | `e191_removeextraPTrows` | 191 | 0 | 60 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:568` |
//! | `e192_printSenProgram` | 192 | 0 | 16 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:959` |
//! | `e193_runDcgForGenKG3` | 193 | 0 | 3 | `DcgManager` | `dcg/dcg_manager/dcg_manager.h:94` |
//! | `e194_runDcgForSparseKG3CONV2D` | 194 | 0 | 3 | `DcgManager` | `dcg/dcg_manager/dcg_manager.h:98` |
//! | `e195_runDcgForSparseKG3BMM` | 195 | 0 | 3 | `DcgManager` | `dcg/dcg_manager/dcg_manager.h:102` |
//! | `e196_printTrafficPerCore` | 196 | 0 | 3 | `DcgManager` | `dcg/dcg_manager/dcg_manager.h:120` |
//! | `e280_runDcgGenerateProgIR` | 280 | 1 | 70 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:145` |
//! | `e281_runDcgForInputFetchNeighbor` | 281 | 1 | 52 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:514` |
//! | `e282_mergePcfgInSuperDSC` | 282 | 1 | 259 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:635` |
//! | `e326_mergePcfgInSuperDSC` | 326 | 2 | 5 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:629` |
//! | `e327_printSenProgram` | 327 | 2 | 13 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:945` |
//! | `e348_runDcgForDataOpsDlOps` | 348 | 3 | 179 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:269` |
//! | `e349_convertToProgIRDataOp` | 349 | 3 | 46 | `DcgManager` | `dcg/dcg_manager/dcg_manager.cpp:898` |

use core::marker::PhantomData;
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Arch, Bytes, Elements, Sticks, Target};
use crate::islands::progir;
use crate::model::Model;
use crate::schedule::l3::dsc::DscIdx;
use crate::units::{Core, Corelet};
use crate::workload::Workload;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE STATE STAGE 3 MUTATES, AND THE THREE SUBSYSTEMS IT HANDS WORK TO.
//
// ⭐ Stage 3's whole in-scope effect is `mySDsc.pcfg_`, `pcfgMap_`/`pcfgPool_` and the manager's
// `firstAvailGlobalGrpId` (`crustify-ddc/TASK.md`). Everything it delegates to — `dcg/dcg_be/`,
// `dcg/dcg_fe/`, `DscPcfgTranslator`, `Dpc` — is OUT of scope (`crustify-ddc/OUTSIDE-DEPS.tsv`), so
// each is a named seam that `todo!`s. ⛔ A stand-in pcfg or program is forbidden, not merely absent.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// `0` — `PcfgInfo::cl`, a literal in both initialisers (`dcg_manager.cpp:477`, `:480`).
const CORELET_ZERO: Corelet = match Corelet::checked(0) {
    Some(corelet) => corelet,
    None => panic!("stage 3 indexes corelet 0; this arch has none"),
};

/// A GTR SYNC GROUP — the `int` of `firstAvailGlobalGrpId` (`dcg_manager.h:44`) and of
/// `SenPcfg::getMaxGTRGroupId()` (`dsc/pcfg.h:1067`).
///
/// ⚠️ THE WRAP IS BY 63, NOT 64: `maxGroupID = 64 - 1`, *"6 bits for GTR groupId"*
/// (`sys-arch-spec/sysdef.cpp:230`), and entry 190 takes the modulus by that value — so the highest
/// id the wrap can produce is 62. ⛔ THIS IS NOT THE `/ 32` TYPO AND MUST NOT BE "CORRECTED": every
/// site in the tree spells it `% sysDef.maxGroupID` (`dcg_manager.cpp:501`, `pcfg_gen.cpp:158`,
/// `stcdpOp.cpp:2587`, `:2663`, `dlOps.cpp:1195`, `inputNeighFetchOp.cpp:1606`, `gatherOp.cpp:912`),
/// and `inputNeighFetchOp.cpp:1607` asserts `maxGrpId <= sysDef.maxGroupID` against the same value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GtrGroupId(u32);

impl GtrGroupId {
    /// `sysDef.maxGroupID` (`sys-arch-spec/sysdef.cpp:230`) — the same on every arch.
    pub const MAX: u32 = 64 - 1;

    /// A group id, reduced the way the reference keeps `firstAvailGlobalGrpId`.
    #[must_use]
    pub const fn wrapped(id: u32) -> Self {
        Self(id % Self::MAX)
    }

    /// The id itself, for the one place it becomes a field.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// `firstAvailGlobalGrpId += maxGrpIDinL3; firstAvailGlobalGrpId %= sysDef.maxGroupID`
    /// (`dcg_manager.cpp:500-501`).
    #[must_use]
    pub const fn advanced_past(self, max_in_l3: Self) -> Self {
        Self::wrapped(self.0 + max_in_l3.0)
    }
}

/// WHICH ENTRY OF `dataOpdscs_` — the `int c` entry 188 counts alongside its walk (`:119`, `:122`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DataOpIndex(usize);

impl DataOpIndex {
    /// Which data op, as a schedule step's `datadsc_idx` names it (`dsc/superdsc.h:32`).
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// The index, for the seam that reaches the data op through it.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// A SLOT IN `pcfgPool_` — `pcfgMap_`'s value type (`dsc/superdsc.h:97-98`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PcfgId(u64);

impl PcfgId {
    /// The slot number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A NODE OF A PCFG GRAPH — `SenPcfgNode*` (`dsc/pcfg.h:1021`), NAMED RATHER THAN OWNED: the nodes
/// are `dcg_fe/pcfg_gen/`'s to mint, and all stage 3 does to one is set its three block bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PcfgNodeId(u64);

impl PcfgNodeId {
    /// A node the generator minted, by the id it minted it under.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// THE THREE PROG-IR BLOCK BITS A NODE CARRIES — `dsc/pcfg.h:139-141`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BlockMarks {
    /// `startNewProgIRBlock`.
    pub start_new_prog_ir_block: bool,
    /// `isBeforeBlock` — *"true--> before, false--> after"*.
    pub is_before_block: bool,
    /// `endBeforeBlock` — *"mark end of before code block"*.
    pub end_before_block: bool,
}

/// A PCFG — `SenPcfg` (`dsc/pcfg.h`), built by `dcg/dcg_fe/pcfg_gen/`.
///
/// ⛔ THE GRAPH ITSELF IS OPAQUE. No unit in this campaign mints a pcfg node and inventing one is
/// forbidden (`crustify-ddc/TASK.md`: *"do not invent a PCFG"*), so this carries only what stage 3
/// READS of a pcfg — whether there is a graph at all, which is the `srcNode != nullptr` guard four
/// merge sites test (`dcg_manager.cpp:762`, `:775`, `:786`, `:847`) — and what it WRITES, the block
/// bits of nodes the generator named.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SenPcfg {
    /// `srcNode` (`dsc/pcfg.h:1021`).
    src: Option<PcfgNodeId>,
    /// The block bits set on this pcfg's nodes (`dsc/pcfg.h:139-141`).
    marks: BTreeMap<PcfgNodeId, BlockMarks>,
}

impl SenPcfg {
    /// A pcfg the generator rooted at a node — the only way one comes to have a graph here.
    #[must_use]
    pub fn rooted_at(src: PcfgNodeId) -> Self {
        Self {
            src: Some(src),
            marks: BTreeMap::new(),
        }
    }

    /// `srcNode != nullptr`.
    #[must_use]
    pub const fn has_graph(&self) -> bool {
        self.src.is_some()
    }

    /// `srcNode`, for the one site that writes its bits (`:848-849`).
    #[must_use]
    pub const fn src_node(&self) -> Option<PcfgNodeId> {
        self.src
    }

    /// One node's block bits.
    #[must_use]
    pub fn marks(&self, node: PcfgNodeId) -> BlockMarks {
        self.marks.get(&node).copied().unwrap_or_default()
    }

    /// One node's block bits, to write.
    pub fn marks_mut(&mut self, node: PcfgNodeId) -> &mut BlockMarks {
        self.marks.entry(node).or_default()
    }

    /// `getPcfgGraphTailNode()` (`dsc/pcfg.h:1065`) — `nullptr` when the graph has no nodes, which is
    /// the only answer derivable without the graph.
    #[must_use]
    pub fn tail_node(&self) -> Option<PcfgNodeId> {
        if self.src.is_none() {
            return None;
        }
        todo!("dcg_fe/pcfg_gen/: SenPcfg::getPcfgGraphTailNode (dsc/pcfg.h:1065) is out of scope")
    }

    /// `SenPcfg::mergeSenPcfg(parent, child)` (`dsc/pcfg.h:1056`).
    ///
    /// ⭐ STATIC, THOUGH IT LOOKS LIKE A METHOD: entry 189 calls it as
    /// `sdscPcfg.mergeSenPcfg(sdscPcfg, *localPcfg)` (`:256`), so the receiver and the `parent`
    /// argument are one object and the child is spliced into it.
    /// ⭐ A CHILD WITH NO GRAPH IS A NO-OP, and entry 282 merges unconditionally: the copy walk is
    /// entered only `if (!child.pcfgNodes.empty())` and seeded from `child.srcNode`
    /// (`dsc/pcfg.cpp:176-181`), after which the edge pass sees only the `nullptr` mapping (`:238-239`).
    /// The one thing it still does — `parent.sdscFoldProps = child.sdscFoldProps` (`:170-174`) —
    /// belongs to `util/foldManager/`, and `FOLDED` states that fact here instead.
    pub fn merge(parent: &mut Self, child: &mut Self) {
        if !child.has_graph() {
            return;
        }
        let _ = parent;
        todo!("dcg_fe/pcfg_gen/: SenPcfg::mergeSenPcfg (dsc/pcfg.h:1056) is out of scope")
    }

    /// `getMaxGTRGroupId()` (`dsc/pcfg.h:1067`).
    pub fn max_gtr_group_id(&self) -> GtrGroupId {
        todo!("dcg_fe/pcfg_gen/: SenPcfg::getMaxGTRGroupId (dsc/pcfg.h:1067) is out of scope")
    }
}

/// ONE CORE'S TWO L3 PCFGS — `pcfgL3lu` and `pcfgL3su`, the pair every entry of this file builds.
///
/// ⭐ KEYED PER CORE, NOT SIZED BY ONE: the reference declares two `std::vector<SenPcfg>` sized
/// `maxCoreId + 1` (`:463-464`; `pcfg_gen.cpp:110-111` for entry 189's pair, declared unsized at
/// `:226-227`) and `.at(coreID)`s them (`:241`, `:490`, `:492`), which throws for a core the
/// generator skipped. A pair per core cannot be misaddressed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct L3Halves {
    /// The `SenComponents::L3LU` half.
    pub lu: SenPcfg,
    /// The `SenComponents::L3SU` half.
    pub su: SenPcfg,
}

impl L3Halves {
    /// Both halves with the component each is filed under, in the `i == 0 ? L3LU : L3SU` order
    /// entry 189 walks them (`:237-241`).
    fn each_mut(&mut self) -> [(SenComponent, &mut SenPcfg); 2] {
        [
            (SenComponent::L3lu, &mut self.lu),
            (SenComponent::L3su, &mut self.su),
        ]
    }
}

/// WHAT A SUPER-DSC IS BEING BUILT FOR — `SenTargets`
/// (`dr5/src/BitcodeLibraries/DataConv/SenDataConvert/util/sendefs.h:63-73`), a closed set of nine.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SenTarget {
    /// `UNDEFINED`, the field's initial value (`dsc/superdsc.h:114`).
    #[default]
    Undefined,
    /// `SENTIENT`.
    Sentient,
    /// `SENULATOR`.
    Senulator,
    /// `SENPCFG` — ⭐ the one value entry 282 branches on (`dcg_manager.cpp:739`).
    SenPcfg,
    /// `SENTF`.
    SenTf,
    /// `SYSTEMC`.
    SystemC,
    /// `HOST`.
    Host,
    /// `INVALID`.
    Invalid,
    /// `NOP`.
    Nop,
}

/// `allUnits` (`dcg_manager.cpp:639`) — the six units a step's syncs bracket, in the order the
/// reference's `std::set` iterates them, which is ascending by enum value (`L3lu = 9 .. Lxsu1 = 14`).
const ALL_UNITS: [SenComponent; 6] = [
    SenComponent::L3lu,
    SenComponent::L3su,
    SenComponent::Lxlu0,
    SenComponent::Lxsu0,
    SenComponent::Lxlu1,
    SenComponent::Lxsu1,
];

/// `lxUnits` (`dcg_manager.cpp:640`) — the four whose inserted sync node opens a prog-IR block.
const LX_UNITS: [SenComponent; 4] = [
    SenComponent::Lxlu0,
    SenComponent::Lxsu0,
    SenComponent::Lxlu1,
    SenComponent::Lxsu1,
];

/// ONE STEP OF A CORE'S SCHEDULE THAT RUNS NO DL DSC — `DscScheduleStep` with `dldsc_idx == -1`
/// (`dsc/superdsc.h:30-46`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DataStep {
    /// `datadsc_idx`, whose `-1` is this `None`.
    pub data_dsc: Option<DataOpIndex>,
    /// `before_sync`.
    pub before_sync: bool,
    /// `after_sync`.
    pub after_sync: bool,
}

/// THE ONE STEP OF A CORE'S SCHEDULE THAT RUNS A DL DSC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DlStep {
    /// `datadsc_idx` — *"if both, then we assume inpNeighborFetch"* (`dsc/superdsc.h:31`).
    pub data_dsc: Option<DataOpIndex>,
    /// `dldsc_idx`.
    pub dl_dsc: DscIdx,
    /// `before_sync`.
    pub before_sync: bool,
    /// `after_sync`.
    pub after_sync: bool,
}

/// ONE CORE'S SCHEDULE — `coreIdToDscSchedule.at(coreId)` (`dsc/superdsc.h:77`).
///
/// ⛔ `DT_CHECK(seenDLDsc == false)` (`dcg_manager.cpp:794`) IS THIS SHAPE: a core runs AT MOST ONE
/// dl step, so `hasDlDsc` (`:650-655`) and `seenDLDsc` become positional — every leading step
/// precedes that one step and every trailing step follows it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CoreSchedule {
    leading: Vec<DataStep>,
    dl: Option<DlStep>,
    trailing: Vec<DataStep>,
}

impl CoreSchedule {
    /// Data steps alone — `hasDlDsc == false`.
    #[must_use]
    pub fn data_only(steps: Vec<DataStep>) -> Self {
        Self {
            leading: steps,
            dl: None,
            trailing: Vec::new(),
        }
    }

    /// A schedule whose one dl step sits between these two runs of data steps.
    #[must_use]
    pub fn with_dl(leading: Vec<DataStep>, dl: DlStep, trailing: Vec<DataStep>) -> Self {
        Self {
            leading,
            dl: Some(dl),
            trailing,
        }
    }

    /// `hasDlDsc` — whether any step of this core's schedule names a DL DSC (`:650-655`).
    fn has_dl_dsc(&self) -> bool {
        self.dl.is_some()
    }

    /// Every step in schedule order, each with `seenDLDsc` AS THE SYNC LOOP SEES IT: the flag is set
    /// at `:793-796`, BEFORE that loop (`:799`), so the dl step's own missing-unit syncs are "after".
    fn steps(&self) -> impl Iterator<Item = (Step, bool)> + '_ {
        let leading = self
            .leading
            .iter()
            .map(|step| (Step::of(*step, None), false));
        let dl = self.dl.iter().map(|step| {
            (
                Step {
                    data_dsc: step.data_dsc,
                    dl_dsc: Some(step.dl_dsc),
                    before_sync: step.before_sync,
                    after_sync: step.after_sync,
                },
                true,
            )
        });
        let trailing = self
            .trailing
            .iter()
            .map(|step| (Step::of(*step, None), true));
        leading.chain(dl).chain(trailing)
    }
}

/// A SCHEDULE STEP AS THE MERGE READS IT — both of `DscScheduleStep`'s indices at once.
#[derive(Debug, Clone, Copy)]
struct Step {
    data_dsc: Option<DataOpIndex>,
    dl_dsc: Option<DscIdx>,
    before_sync: bool,
    after_sync: bool,
}

impl Step {
    fn of(data: DataStep, dl_dsc: Option<DscIdx>) -> Self {
        Self {
            data_dsc: data.data_dsc,
            dl_dsc,
            before_sync: data.before_sync,
            after_sync: data.after_sync,
        }
    }

    /// `datadsc_idx` as `std::to_string` writes it, `-1` included.
    fn data_idx(self) -> i64 {
        self.data_dsc.map_or(-1, |at| at.get() as i64)
    }

    /// `dldsc_idx` likewise.
    fn dl_idx(self) -> i64 {
        self.dl_dsc.map_or(-1, |dl| i64::from(dl.0))
    }

    /// `"before_dsc" + dldsc_idx` or `"before_ddsc" + datadsc_idx` (`:693-695`), and the same pair
    /// under `"after"` (`:706-708`).
    fn sync_tag(self, when: &str, dldsc: bool) -> String {
        if dldsc {
            format!("{when}_dsc{}", self.dl_idx())
        } else {
            format!("{when}_ddsc{}", self.data_idx())
        }
    }

    /// ⛔ THE MISSING-UNIT TAG CONCATENATES BOTH INDICES WITH NO SEPARATOR (`:812-814`, `:819-821`),
    /// so a dl-only step spells `before_ddsc-10` and never `before_ddsc-1_0`.
    fn missing_unit_tag(self, when: &str) -> String {
        format!("{when}_ddsc{}{}", self.data_idx(), self.dl_idx())
    }
}

/// WHY A GENERATED PROGRAM WAS REFUSED — `ProgramAndStateInfo::ErrorType`
/// (`sys-arch-spec/progir/progir.h:519-528`) MINUS ITS `NONE`: `createSenProgramSTCDPOp` files a core
/// only when that core's status is not `NONE` (`dcg/dcg_be/dcgbeCodegen.cpp:20-70`), so a `NONE`
/// entry is not a value the map it hands back can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramError {
    /// `OPCODE_OPERAND`.
    OpcodeOperand,
    /// `IBUFF_OVERFLOW` — ⭐ the one entries 280 and 281 recover from.
    IbuffOverflow,
    /// `LOOP`.
    Loop,
    /// `PC_TARGET`.
    PcTarget,
    /// `GRAPH`.
    Graph,
    /// `IMMEDIATE`.
    Immediate,
    /// `REG_INIT`.
    RegInit,
}

/// THE LICENCE FOR `(STCDPOpLx*)myDataOpDsc.op` (`dcg_manager.cpp:192`) — an I-buff overflow reported
/// against a data op whose func IS `STCDPOpLx`, which is the only pairing that mints this
/// (`:176-177`, `:537-538`). ⛔ WITHOUT IT THAT CAST IS UNCHECKED IN THE REFERENCE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IbuffViolation(());

impl IbuffViolation {
    /// `status.first == IBUFF_OVERFLOW && op->name == OpFuncs::STCDPOpLx`.
    fn witnessed(op: OpFunc, error: ProgramError) -> Option<Self> {
        (op == OpFunc::StcdpOpLx && error == ProgramError::IbuffOverflow).then_some(Self(()))
    }
}

/// WHERE A DATA OP THE CODEGEN IS HANDED LIVES — `generatePcfgIRForDataOpInpFetch` returns a
/// `DataOpDsc*` that is either an entry OF the main super-DSC or one it minted (`:522`).
///
/// ⭐ AN INDEX FOR THE FIRST CASE, as entry 188's seam takes one: a `&mut` into the super-DSC cannot
/// be held beside the `&mut SuperDsc` every one of these seams also takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataOpTarget {
    /// `&mySDsc.dataOpdscs_[idx]`.
    InMain(DataOpIndex),
    /// A data op the generator minted, which the entry owns.
    Detached(DataOpDsc),
}

impl DataOpTarget {
    /// `ddsc.op->name` — `None` where the reference dereferences an index the super-DSC has not got.
    fn op<const FOLDED: bool>(&self, sdsc: &SuperDsc<true, FOLDED>) -> Option<OpFunc> {
        match self {
            Self::InMain(at) => sdsc.data_op_dscs.get(at.get()).map(|ddsc| ddsc.op),
            Self::Detached(ddsc) => Some(ddsc.op),
        }
    }
}

/// WHICH DL PCFG THE MERGE TAKES — `useActualDLpcfg` and, when it is false, the two
/// `std::vector<SenPcfg>&` the caller passes alongside (`dcg_manager.cpp:636-638`, `:744`, `:780`).
///
/// ⭐ ONE ARGUMENT, BECAUSE THE ARMS ARE EXCLUSIVE: the vectors are read only when the flag is false
/// (`:785`) and the DSC's own pcfg only when it is true (`:761`).
/// ⚠️ ENTRY 326 PASSES `pcfgL3lu` TWICE (`:632`), so its "su" half is the "lu" one — harmless there
/// because both are empty and the arm needs `size() > 0` on each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlPcfgSource<'a> {
    /// `useActualDLpcfg == true` — the DL DSC's own pcfg, or else the super-DSC's pcfg pool.
    Dsc,
    /// `useActualDLpcfg == false` — `pcfgL3lu.at(coreID)` and `pcfgL3su.at(coreID)`, an EMPTY table
    /// skipping the arm entirely (`:780`).
    L3Halves(&'a BTreeMap<Core, L3Halves>),
}

/// WHETHER A DSC HAS A SCHEDULE TREE — `isDSC2()`, which is `!scheduleTree_.empty()`
/// (`dsc/designSpaceConfig.cpp:31`). Stage 2a is what puts one there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DscVersion {
    /// `isDSC2() == false`.
    Dsc1,
    /// `isDSC2() == true`.
    Dsc2,
}

/// ONE DL DSC AS STAGE 3 READS IT — `DesignSpaceConfig` (`dsc/designSpaceConfig.h`), restricted to
/// `isDSC2()`, `coreIdsUsed_` (`:75`) and `pcfg_` (`:120`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlDsc {
    /// `isDSC2()`.
    version: DscVersion,
    /// `coreIdsUsed_`, EACH WITH THE PCFG `pcfg_` HOLDS FOR IT — `None` where the second vector does
    /// not reach that core's index, which is `dsc.pcfg_.size() > coreIDX` answering false
    /// (`dcg_manager.cpp:760`) and what sends entry 282 to the pcfg-pool arm instead.
    pcfgs: BTreeMap<Core, Option<SenPcfg>>,
}

/// THE DL DSCs, AND WHICH ONE SERVES EACH CORE, NEVER EMPTY.
///
/// ⭐ ONE TABLE FOR TWO FIELDS: `coreIdToDsc_` is a map of `DesignSpaceConfig*` INTO `dscs_`
/// (`dsc/superdsc.h:67-68`), so a core names a [`DscIdx`] here and a schedule step's `dldsc_idx`
/// names the same one.
/// ⭐ NON-EMPTY BY CONSTRUCTION — `DT_CHECK(mySDsc.dscs_.size() >= 1)` (`:220`, `:453`) needs no
/// runtime form when the type admits no empty value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerCoreDscs {
    /// `dscs_` (`dsc/superdsc.h:67`).
    dscs: Vec<DlDsc>,
    /// `coreIdToDsc_` (`dsc/superdsc.h:68`).
    core_to_dsc: BTreeMap<Core, DscIdx>,
}

impl PerCoreDscs {
    /// One DL DSC, on one core.
    #[must_use]
    pub fn of(core: Core, version: DscVersion) -> Self {
        Self {
            dscs: Vec::new(),
            core_to_dsc: BTreeMap::new(),
        }
        .and(core, version)
    }

    /// Another core's DL DSC.
    #[must_use]
    pub fn and(mut self, core: Core, version: DscVersion) -> Self {
        let idx = DscIdx(u32::try_from(self.dscs.len()).unwrap_or(u32::MAX));
        self.dscs.push(DlDsc {
            version,
            pcfgs: BTreeMap::from([(core, None)]),
        });
        self.core_to_dsc.insert(core, idx);
        self
    }

    /// `dscs_.at(idx).pcfg_` reaching this core's slot (`:760-761`).
    #[must_use]
    pub fn with_dl_pcfg(mut self, idx: DscIdx, core: Core, pcfg: SenPcfg) -> Self {
        if let Some(dsc) = self.dscs.get_mut(idx.0 as usize) {
            dsc.pcfgs.insert(core, Some(pcfg));
        }
        self
    }

    /// `for (auto& kv : mySDsc.coreIdToDsc_)` — ascending core, as `std::map` iterates.
    pub fn iter(&self) -> impl Iterator<Item = (Core, DscVersion)> + '_ {
        self.core_to_dsc.iter().filter_map(|(core, idx)| {
            self.dscs
                .get(idx.0 as usize)
                .map(|dsc| (*core, dsc.version))
        })
    }

    /// `mySDsc.dscs_.at(scheduleStep.dldsc_idx)` (`:747`) — `None` where
    /// `DT_CHECK(dldsc_idx < mySDsc.dscs_.size())` (`:746`) throws.
    fn at(&self, idx: DscIdx) -> Option<&DlDsc> {
        self.dscs.get(idx.0 as usize)
    }
}

/// ONE DATA-OP DSC AS STAGE 3 READS IT — `DataOpDsc` (`dsc/dataOpDsc.h:987`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataOpDsc {
    /// `op->name`, where `op` is a `baseOp*` (`dsc/dataOpDsc.h:1002`).
    ///
    /// ⛔ THE REFERENCE DEREFERENCES IT UNCHECKED (`:579-580`) though the pointer is `nullptr` until
    /// the DCG or the graph optimizer fills it; carrying the op-func itself makes an unfilled op
    /// unspellable rather than a null read.
    pub op: OpFunc,
    /// `pcfg_` — `[coreIdsUsed_ idx] -> {SenPcfg, L3LU or L3SU or LXSU0 or LXLU0}`
    /// (`dsc/dataOpDsc.h:1005-1006`).
    ///
    /// ⭐ KEYED BY THE CORE, NOT BY ITS POSITION: every reader first walks `coreIdsUsed_` for this
    /// core's index and then `DT_CHECK(coreIDX >= 0)`s it (`dcg_manager.cpp:663-668`, `:723-729`), so
    /// the key retires both the walk and the check — and a core the op does not serve simply has no
    /// entry, which is the one place the reference SKIPS such a core rather than throwing (`:669`).
    pub pcfg: BTreeMap<Core, Vec<(SenPcfg, SenComponent)>>,
}

/// THE SUPER-DSC AS STAGE 3 SEES IT — `SuperDsc` (`dsc/superdsc.h:48`), restricted to the pcfg state
/// this stage mutates. The schedule tree, the allocations and the placed addresses belong to stages
/// 2a and 2b, whose own homes own those fields.
///
/// ⛔ TWO `DT_CHECK`s BECOME TWO TYPE-LEVEL FACTS:
/// `DATA_OPS` — whether `dataOpdscs_` may hold anything. `DT_CHECK(mySDsc.dataOpdscs_.size() == 0)`
/// (`:222`) is entry 189 taking `SuperDsc<false, _>`, and entry 188 takes `SuperDsc<true, _>`.
/// `FOLDED` — whether `sdscFoldProps_` is non-empty. `DT_CHECK_MSG(mySDsc.sdscFoldProps_.empty(),
/// "Codegen for Folded Super-DSC is not supported")` (`:262`, `:506`) is a `const { assert! }` in the
/// codegen tails, which is where the reference checks it — so the props themselves stay with
/// `util/foldManager/`, out of scope, where they are read.
/// ⛔ AND IT ASSERTS `!(createSenProg && FOLDED)`, NEVER `!FOLDED`: every one of those checks sits
/// INSIDE `if (createSenProg)`, while a `const` block is evaluated whether or not its branch can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperDsc<const DATA_OPS: bool, const FOLDED: bool> {
    /// `dscs_` and `coreIdToDsc_` (`dsc/superdsc.h:67-68`).
    dscs: PerCoreDscs,
    /// `dataOpdscs_` (`dsc/superdsc.h:75`).
    data_op_dscs: Vec<DataOpDsc>,
    /// `pcfg_` — `[coreID][component] -> SenPcfg` (`dsc/superdsc.h:96`).
    pcfg: BTreeMap<Core, BTreeMap<SenComponent, SenPcfg>>,
    /// `pcfgMap_` — `[coreID][component] -> pcfgPool_` slot (`dsc/superdsc.h:97`).
    pcfg_map: BTreeMap<Core, BTreeMap<SenComponent, PcfgId>>,
    /// `pcfgPool_` (`dsc/superdsc.h:98`).
    pcfg_pool: BTreeMap<PcfgId, SenPcfg>,
    /// `coreIdToDscSchedule` (`dsc/superdsc.h:77`) — ⭐ AN INPUT: scratchy states its own schedule and
    /// no unit in this campaign writes this field (`crustify-ddc/TASK.md`).
    core_schedules: BTreeMap<Core, CoreSchedule>,
    /// `target_` (`dsc/superdsc.h:114`).
    target: SenTarget,
}

impl<const DATA_OPS: bool, const FOLDED: bool> SuperDsc<DATA_OPS, FOLDED> {
    /// A super-DSC carrying the DL DSCs and nothing else — every pcfg field starts empty, as the
    /// reference's do.
    #[must_use]
    pub fn of(dscs: PerCoreDscs) -> Self {
        Self {
            dscs,
            data_op_dscs: Vec::new(),
            pcfg: BTreeMap::new(),
            pcfg_map: BTreeMap::new(),
            pcfg_pool: BTreeMap::new(),
            core_schedules: BTreeMap::new(),
            target: SenTarget::Undefined,
        }
    }

    /// `coreIdToDscSchedule[coreId]` — one core's stated schedule (`dsc/superdsc.h:77`).
    #[must_use]
    pub fn with_core_schedule(mut self, core: Core, schedule: CoreSchedule) -> Self {
        self.core_schedules.insert(core, schedule);
        self
    }

    /// `target_` (`dsc/superdsc.h:114`).
    #[must_use]
    pub fn with_target(mut self, target: SenTarget) -> Self {
        self.target = target;
        self
    }

    /// `pcfg_[coreId][comp]` — a BARE SUBSCRIPT in the reference (`:468-469`), whose whole effect
    /// is to DEFAULT-CONSTRUCT the entry before the translator fills it.
    fn pcfg_entry(&mut self, core: Core, comp: SenComponent) -> &mut SenPcfg {
        self.pcfg.entry(core).or_default().entry(comp).or_default()
    }

    /// THE POOL SLOT `[coreID][comp]` NAMES, allocating one when that pair has none (`:243-253`).
    ///
    /// ⚠️ A NEW ID IS `pcfgPool_.rbegin()->first + 1`, NOT THE LOWEST FREE ONE — entry 191 erases
    /// slots from the pool, and a hole it leaves is never refilled unless it was the highest.
    fn pcfg_slot(&mut self, core: Core, comp: SenComponent) -> &mut SenPcfg {
        let next = PcfgId(
            self.pcfg_pool
                .keys()
                .next_back()
                .map_or(0, |last| last.0 + 1),
        );
        let id = *self
            .pcfg_map
            .entry(core)
            .or_default()
            .entry(comp)
            .or_insert(next);
        self.pcfg_pool.entry(id).or_default()
    }
}

impl<const FOLDED: bool> SuperDsc<true, FOLDED> {
    /// `dataOpdscs_.push_back(..)`, on a super-DSC whose type says it may have data ops.
    #[must_use]
    pub fn with_data_op(mut self, data_op: DataOpDsc) -> Self {
        self.data_op_dscs.push(data_op);
        self
    }
}

/// `pcfgUnitToRemove` (`dcg_manager.cpp:571-576`) — the PT-row pcfgs a re-stickify-with-PT op leaves
/// behind, on both corelets.
///
/// ⛔ ROW 0 IS NOT IN THE SET. `PTROW0_0` and `PTROW0_1` are absent from all fourteen entries, so
/// row 0's pcfg survives on either corelet.
const PT_ROW_PCFGS_TO_REMOVE: [SenComponent; 14] = [
    SenComponent::Ptrow1_0,
    SenComponent::Ptrow2_0,
    SenComponent::Ptrow3_0,
    SenComponent::Ptrow4_0,
    SenComponent::Ptrow5_0,
    SenComponent::Ptrow6_0,
    SenComponent::Ptrow7_0,
    SenComponent::Ptrow1_1,
    SenComponent::Ptrow2_1,
    SenComponent::Ptrow3_1,
    SenComponent::Ptrow4_1,
    SenComponent::Ptrow5_1,
    SenComponent::Ptrow6_1,
    SenComponent::Ptrow7_1,
];

/// WHICH FORM ENTRY 192 WRITES — the reference's `bool use_smc = false` (`:961`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenProgramForm {
    /// `convertIr2Senprog` — `use_smc == false`, the default.
    SenProg,
    /// `convertIr2SMC` — `use_smc == true`.
    Smc,
}

/// THE DEEP PROGRAM CONVERTER — `Dpc` (`sys-arch-spec/dpc/dpc.h`), out of this campaign's file list.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Dpc;

impl Dpc {
    /// `myDpc.convertIr2Senprog(mySDsc->progstateinfo_, *senCompToISAptr, fileFP, std::nullopt)`
    /// (`:968-969`).
    pub fn convert_ir_to_sen_prog<A: Arch, M: Model, W: Workload>(
        &mut self,
        programs: &BTreeMap<Core, progir::Program<A, M, W>>,
        out: &mut dyn core::fmt::Write,
    ) {
        let _ = (programs, out);
        todo!("sys-arch-spec/dpc/: Dpc::convertIr2Senprog is out of scope")
    }

    /// `myDpc.convertIr2SMC(mySDsc->progstateinfo_, *senCompToISAptr, fileFP, std::nullopt)`
    /// (`:971-972`).
    pub fn convert_ir_to_smc<A: Arch, M: Model, W: Workload>(
        &mut self,
        programs: &BTreeMap<Core, progir::Program<A, M, W>>,
        out: &mut dyn core::fmt::Write,
    ) {
        let _ = (programs, out);
        todo!("sys-arch-spec/dpc/: Dpc::convertIr2SMC is out of scope")
    }
}

/// THE TRANSLATOR'S ARGUMENT — `DscPcfgTranslator::PcfgInfo` (`dsc/dsc2Pcfg.h:102-108`).
///
/// ⭐ THE PCFG BY KEY, NOT BY REFERENCE. The C++ field is a `SenPcfg&` aimed at
/// `mySDsc.pcfg_[coreId][comp]` while the same super-DSC is passed alongside it (`:476-485`) — two
/// live paths to one object. The key reaches the same place with one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcfgInfo {
    /// `c` — the current core.
    pub core: Core,
    /// `cl` — the current corelet.
    pub corelet: Corelet,
    /// `clComp` — the corelet-specific component.
    pub cl_comp: SenComponent,
    /// `genComp` — the corelet-agnostic component. ⭐ Equal to `clComp` for either L3 half, which is
    /// shared across a core's corelets.
    pub gen_comp: SenComponent,
}

impl PcfgInfo {
    /// `{mySDsc.pcfg_[coreId][L3LU], coreId, 0, L3LU, L3LU}` (`:476-478`).
    #[must_use]
    pub const fn l3_lu(core: Core) -> Self {
        Self {
            core,
            corelet: CORELET_ZERO,
            cl_comp: SenComponent::L3lu,
            gen_comp: SenComponent::L3lu,
        }
    }

    /// `{mySDsc.pcfg_[coreId][L3SU], coreId, 0, L3SU, L3SU}` (`:479-481`).
    #[must_use]
    pub const fn l3_su(core: Core) -> Self {
        Self {
            core,
            corelet: CORELET_ZERO,
            cl_comp: SenComponent::L3su,
            gen_comp: SenComponent::L3su,
        }
    }
}

/// THE DSC-2 PCFG TRANSLATOR — `DscPcfgTranslator` (`dsc/dsc2Pcfg.h`), OUT of scope
/// (`crustify-ddc/OUTSIDE-DEPS.tsv` names it for six of this campaign's units).
pub struct DscPcfgTranslator;

impl DscPcfgTranslator {
    /// `DscPcfgTranslator::transformDscCompToPcfg(mySDsc, *dscGlobal, *dsc, info)` (`:482-485`) —
    /// fills `pcfg_[info.core][info.cl_comp]` from the DSC's schedule tree.
    pub fn transform_dsc_comp_to_pcfg<const D: bool, const F: bool>(
        sdsc: &mut SuperDsc<D, F>,
        info: PcfgInfo,
    ) {
        let _ = (sdsc, info);
        todo!("dsc/dsc2Pcfg: DscPcfgTranslator::transformDscCompToPcfg is out of scope")
    }
}

/// THE PCFG FRONT END — `DcgFE` (`dcg/dcg_fe/`), OUT of scope (`crustify-ddc/OUTSIDE-DEPS.tsv`).
pub struct DcgFrontEnd;

impl DcgFrontEnd {
    /// `dcg_fe_.computeTranferforDataOp(mySDsc, myDataOpDsc, c)` (`:122`).
    ///
    /// ⛔ `first_avail` IS `firstAvailGlobalGrpId_`, THE MANAGER'S OWN FIELD BY REFERENCE
    /// (`dcg_frontend.h:129`, bound at `:68`/`:76` from `dcg_manager.h:60`): this path ADVANCES it,
    /// through `computerGTRInfo` (`transfer_compute.cpp:74`, `:114` → `stcdpOp.cpp:2586-2587`,
    /// `:2662-2663`). Without the `&mut` the effect would be inexpressible here.
    pub fn compute_transfer_for_data_op<const F: bool>(
        sdsc: &mut SuperDsc<true, F>,
        at: DataOpIndex,
        first_avail: &mut GtrGroupId,
    ) {
        let _ = (sdsc, at, first_avail);
        todo!("dcg_fe/: DcgFE::computeTranferforDataOp is out of scope")
    }

    /// `dcg_fe_.generatePcfgIRForDLOp(mySDsc, pcfgL3lu, pcfgL3su)` (`:229`) — one L3 pair per core
    /// of the super-DSC's `coreIdToDsc_`.
    ///
    /// ⛔ IT ADVANCES `first_avail` ITSELF, past the largest group id any core used
    /// (`pcfg_gen.cpp:157-158`), through the same `int&` (`dcg_frontend.h:129`).
    pub fn generate_pcfg_ir_for_dl_op<const F: bool>(
        sdsc: &mut SuperDsc<false, F>,
        first_avail: &mut GtrGroupId,
    ) -> BTreeMap<Core, L3Halves> {
        let _ = (sdsc, first_avail);
        todo!("dcg_fe/pcfg_gen/: DcgFE::generatePcfgIRForDLOp is out of scope")
    }

    /// `dcg_fe_.createPcfgForUnitPerCore(mySDsc, pcfg, comp, coreId)` (`:489-490`, `:491-492`), which
    /// returns the unit's largest GTR group id.
    ///
    /// ⭐ `first_avail` IS BY VALUE BECAUSE THIS ONE ONLY READS IT: every occurrence inside the
    /// function is `(myGtr.groupName_ + firstAvailGlobalGrpId_) % sysDef.maxGroupID`
    /// (`dlOps.cpp:1195`, `:1255`, `:1296`, in the body opened at `:15`) — entry 190 does the
    /// advancing itself.
    pub fn create_pcfg_for_unit_per_core<const D: bool, const F: bool>(
        sdsc: &mut SuperDsc<D, F>,
        pcfg: &mut SenPcfg,
        comp: SenComponent,
        core: Core,
        first_avail: GtrGroupId,
    ) -> GtrGroupId {
        let _ = (sdsc, pcfg, comp, core, first_avail);
        todo!("dcg_fe/: DcgFE::createPcfgForUnitPerCore is out of scope")
    }

    /// `dcg_fe_.addSyncNode(masterSenPCFG, senCompType, coreID, tag)` (`dcg_frontend.h:104`) — the
    /// sync node it appends, whose block bits the missing-unit arm then writes (`:824-831`).
    pub fn add_sync_node(
        pcfg: &mut SenPcfg,
        comp: SenComponent,
        core: Core,
        tag: &str,
    ) -> PcfgNodeId {
        let _ = (pcfg, comp, core, tag);
        todo!("dcg_fe/: DcgFE::addSyncNode is out of scope")
    }

    /// `dcg_fe_.finalizeBurstInfo((STCDPOpLx*)myDataOpDsc.op)` (`:192`) — re-derives the bursts now
    /// that the regenerated program may carry dynamic mvloops.
    pub fn finalize_burst_info<const F: bool>(
        sdsc: &mut SuperDsc<true, F>,
        at: &mut DataOpTarget,
        violation: IbuffViolation,
    ) {
        let _ = (sdsc, at, violation);
        todo!("dcg_fe/: DcgFE::finalizeBurstInfo is out of scope")
    }

    /// `dcg_fe_.createPcfgsSTCDPOp(&myDataOpDsc, true)` (`:193`, `:553`).
    pub fn create_pcfgs_stcdp_op<const F: bool>(
        sdsc: &mut SuperDsc<true, F>,
        at: &mut DataOpTarget,
        force_no_opt: bool,
    ) {
        let _ = (sdsc, at, force_no_opt);
        todo!("dcg_fe/: DcgFE::createPcfgsSTCDPOp is out of scope")
    }

    /// `dcg_fe_.generatePcfgIRForDataOpInpFetch(mySDscMain, mySDscPre, 0)` (`:522`), whose result is
    /// either an entry of the main super-DSC or a data op it minted.
    pub fn generate_pcfg_ir_for_data_op_inp_fetch<const F: bool>(
        main: &mut SuperDsc<true, F>,
        pre: Option<&mut SuperDsc<true, F>>,
        at: DataOpIndex,
    ) -> DataOpTarget {
        let _ = (main, pre, at);
        todo!("dcg_fe/: DcgFE::generatePcfgIRForDataOpInpFetch is out of scope")
    }
}

/// THE DCG BACKEND — `DcgBE` (`dcg/dcg_be/`), OUT of scope (`crustify-ddc/OUTSIDE-DEPS.tsv` names it
/// for nine of this file's units).
pub struct DcgBackEnd;

impl DcgBackEnd {
    /// `dcg_be_.runDcgForGenKG3(mySDsc, createSenProg)` (`dcg_manager.h:95`).
    pub fn run_dcg_for_gen_kg3<const D: bool, const F: bool>(
        sdsc: &mut SuperDsc<D, F>,
        create_sen_prog: bool,
    ) {
        let _ = (sdsc, create_sen_prog);
        todo!("dcg_be/: DcgBE::runDcgForGenKG3 is out of scope")
    }

    /// `dcg_be_.runDcgForSparseKG3CONV2D(mySDsc, createSenProg)` (`dcg_manager.h:99`).
    pub fn run_dcg_for_sparse_kg3_conv2d<const D: bool, const F: bool>(
        sdsc: &mut SuperDsc<D, F>,
        create_sen_prog: bool,
    ) {
        let _ = (sdsc, create_sen_prog);
        todo!("dcg_be/: DcgBE::runDcgForSparseKG3CONV2D is out of scope")
    }

    /// `dcg_be_.runDcgForSparseKG3BMM(mySDsc, createSenProg)` (`dcg_manager.h:103`).
    pub fn run_dcg_for_sparse_kg3_bmm<const D: bool, const F: bool>(
        sdsc: &mut SuperDsc<D, F>,
        create_sen_prog: bool,
    ) {
        let _ = (sdsc, create_sen_prog);
        todo!("dcg_be/: DcgBE::runDcgForSparseKG3BMM is out of scope")
    }

    /// `dcg_be_.createSenProgramSTCDPOp(&myDataOpDsc, &mySDsc)` (`:171-172`, `:533`) — one entry per
    /// core whose generated program was refused, and its message.
    ///
    /// ⭐ NO `NONE` IN THE MAP: the backend files a core only when its status is not `NONE`
    /// (`dcg/dcg_be/dcgbeCodegen.cpp:20-70`), which is why [`ProgramError`] carries no such variant.
    pub fn create_sen_program_stcdp_op<const F: bool>(
        sdsc: &mut SuperDsc<true, F>,
        at: &DataOpTarget,
    ) -> BTreeMap<Core, (ProgramError, String)> {
        let _ = (sdsc, at);
        todo!("dcg_be/: DcgBE::createSenProgramSTCDPOp is out of scope")
    }

    /// `dcg_be_.fillAndCreateSenProgInfoUsingSuperDSC(mySDsc)` (`:208`, `:510`) — the codegen route
    /// that reads the super-DSC's own pcfgs rather than one data op's.
    pub fn fill_and_create_sen_prog_info<const D: bool, const F: bool>(sdsc: &mut SuperDsc<D, F>) {
        let _ = sdsc;
        todo!("dcg_be/: DcgBE::fillAndCreateSenProgInfoUsingSuperDSC is out of scope")
    }
}

/// WHAT `createSenProg` AND `progIRcodeGen` ARE TOGETHER, AS A TYPE.
///
/// ⭐ THE TWO ENTRIES DISAGREE ABOUT WHICH GENERATOR IS LEGAL: entry 189 asserts
/// `progIRcodeGen == DCGProgIRGen::DCG` and entry 190 asserts `!= DCG` (`:264` against `:508`), both
/// inside `if (createSenProg)`. As a type that is [`DlOpsSenProg`] against [`StandaloneSenProg`], and
/// the wrong pairing is an E0277 rather than an abort.
/// ⭐ `SchedulerStages.cpp:49-50` — our only call site — sets `DCC` and `createSenProg = false`, so
/// the configuration this campaign runs is [`NoSenProg`], where both tails are no-ops.
pub trait SenProgGen {
    /// `createSenProg` (`dcg_manager.h:33`) — the flag entries 193-195 hand the backend.
    const CREATE_SEN_PROG: bool;

    /// `progIRcodeGen == DCGProgIRGen::DCC` — ⛔ NOT DERIVABLE FROM [`Self::CREATE_SEN_PROG`]: entry
    /// 282 reads it OUTSIDE any `createSenProg` guard, twice (`dcg_manager.cpp:774`, `:863`), so the
    /// pairing needs both flags.
    const DCC: bool;
}

/// `createSenProg == false` with `progIRcodeGen == DCGProgIRGen::DCC` — ⭐ our configuration
/// (`SchedulerStages.cpp:49-50`). No codegen tail on any entry.
pub struct NoSenProg;

/// `createSenProg == false` with `progIRcodeGen == DCGProgIRGen::DCG` — the pairing entry 280 needs,
/// whose `DT_CHECK(progIRcodeGen == DCG)` (`:152`) sits OUTSIDE its `createSenProg` guard.
pub struct NoSenProgDcg;

/// `createSenProg == true` with `progIRcodeGen == DCGProgIRGen::DCG` — entry 189's tail.
pub struct DcgSenProg;

/// `createSenProg == true` with `progIRcodeGen == DCGProgIRGen::DCC` — entry 190's tail.
pub struct DccSenProg;

impl SenProgGen for NoSenProg {
    const CREATE_SEN_PROG: bool = false;
    const DCC: bool = true;
}

impl SenProgGen for NoSenProgDcg {
    const CREATE_SEN_PROG: bool = false;
    const DCC: bool = false;
}

impl SenProgGen for DcgSenProg {
    const CREATE_SEN_PROG: bool = true;
    const DCC: bool = false;
}

impl SenProgGen for DccSenProg {
    const CREATE_SEN_PROG: bool = true;
    const DCC: bool = true;
}

/// `progIRcodeGen == DCGProgIRGen::DCG` AS A BOUND — entry 280 checks it before it looks at
/// `createSenProg` (`:152`), so a manager configured for DCC cannot call that entry at all.
pub trait DcgProgIRGen: SenProgGen {}

impl DcgProgIRGen for NoSenProgDcg {}

impl DcgProgIRGen for DcgSenProg {}

/// ENTRY 189'S CODEGEN TAIL. ⛔ NOT IMPLEMENTED FOR [`DccSenProg`]: that is
/// `DT_CHECK(progIRcodeGen == DCGProgIRGen::DCG)` (`:264`).
pub trait DlOpsSenProg: SenProgGen {
    /// `dcg_be_.convertToProgIRDlOp(mySDsc, pcfgL3lu, pcfgL3su)` (`:265`).
    fn convert_to_prog_ir_dl_op<const FOLDED: bool>(
        sdsc: &mut SuperDsc<false, FOLDED>,
        halves: BTreeMap<Core, L3Halves>,
    );
}

impl DlOpsSenProg for NoSenProg {
    fn convert_to_prog_ir_dl_op<const FOLDED: bool>(
        _sdsc: &mut SuperDsc<false, FOLDED>,
        _halves: BTreeMap<Core, L3Halves>,
    ) {
    }
}

impl DlOpsSenProg for NoSenProgDcg {
    fn convert_to_prog_ir_dl_op<const FOLDED: bool>(
        _sdsc: &mut SuperDsc<false, FOLDED>,
        _halves: BTreeMap<Core, L3Halves>,
    ) {
    }
}

impl DlOpsSenProg for DcgSenProg {
    fn convert_to_prog_ir_dl_op<const FOLDED: bool>(
        sdsc: &mut SuperDsc<false, FOLDED>,
        halves: BTreeMap<Core, L3Halves>,
    ) {
        const {
            assert!(
                !FOLDED,
                "Codegen for Folded Super-DSC is not supported (dcg_manager.cpp:262)"
            )
        }
        let _ = (sdsc, halves);
        todo!("dcg_be/: DcgBE::convertToProgIRDlOp is out of scope")
    }
}

/// ENTRY 190'S CODEGEN TAIL. ⛔ NOT IMPLEMENTED FOR [`DcgSenProg`]: that is
/// `DT_CHECK(progIRcodeGen != DCGProgIRGen::DCG)` (`:508`).
pub trait StandaloneSenProg: SenProgGen {
    /// `dcg_be_.fillAndCreateSenProgInfoUsingSuperDSC(mySDsc)` (`:510`).
    fn fill_and_create_sen_prog_info<const DATA_OPS: bool, const FOLDED: bool>(
        sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    );
}

impl StandaloneSenProg for NoSenProg {
    fn fill_and_create_sen_prog_info<const DATA_OPS: bool, const FOLDED: bool>(
        _sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    ) {
    }
}

impl StandaloneSenProg for NoSenProgDcg {
    fn fill_and_create_sen_prog_info<const DATA_OPS: bool, const FOLDED: bool>(
        _sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    ) {
    }
}

impl StandaloneSenProg for DccSenProg {
    fn fill_and_create_sen_prog_info<const DATA_OPS: bool, const FOLDED: bool>(
        sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    ) {
        const {
            assert!(
                !FOLDED,
                "Codegen for Folded Super-DSC is not supported (dcg_manager.cpp:506)"
            )
        }
        DcgBackEnd::fill_and_create_sen_prog_info(sdsc);
    }
}

/// THE DCG MANAGER — `DcgManager` (`dcg/dcg_manager/dcg_manager.h:28`), stage 3 of `runDdc`.
///
/// ⛔ THREE OF THE C++'s FIELDS ARE TYPES HERE, NOT VALUES. `C` is `createSenProg` and
/// `progIRcodeGen` together; `DT2` is `dscGlobal->dtVersion > 1` (`sys-arch-spec/dscglobal/dscglobal.h:83`,
/// and `setDtVersion` admits only 1 or 2); `L3_DL_SCHEDULER` is `enableL3DlScheduler`, whose only
/// writer is a `DISCARD_ABOVE_LX_SCHEDULE` env read in the constructor (`:78-79`) — a build choice,
/// so a const generic that REMOVES the arm it turns off.
/// ⛔ `DT_CHECK(senCompToISAptr != nullptr)` (`:117`, `:223`, `:450`, `:454`) HAS NO RUNTIME FORM HERE: the
/// per-component ISA tables are `sys_arch_spec`'s consts, so the pointer it guards is a table the
/// compiler folds.
/// ⛔ `DT_CHECK(!isInpFetchNeigh)` (`:118`, `:221`) IS STRUCTURAL: input-fetch-neighbour mode is
/// entry 281's own entry point, and this manager carries no such state.
/// ⚠️ THE `verbose` `std::cout` LINES (`:113-115`, `:217-219`) ARE NOT REPRODUCED. This crate has no
/// diagnostic stream, and one added here would print from inside a proc macro's build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcgManager<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool> {
    /// `firstAvailGlobalGrpId` (`dcg_manager.h:44`), whose comment at `:43` is *"global variables --> shared
    /// across dataOpDscs/DLDscs"* — so it survives every entry this manager runs.
    first_avail_global_grp_id: GtrGroupId,
    /// `enable_prog_verification_` (`dcg_manager.h:38`), false by default.
    ///
    /// ⭐ A FIELD AND NOT A CONST GENERIC: `DcgBE` binds it by `const bool&` (`:64`), so it is read
    /// late and a build cannot state it.
    enable_prog_verification: bool,
    /// Which codegen tail, as a type rather than a pair of fields.
    codegen: PhantomData<C>,
}

/// STAGE 3 AS `SchedulerStages.cpp:45-57` BUILDS IT — `progIRcodeGen = DCC`, `createSenProg = false`,
/// `dtVersion = 2` and the L3 DL scheduler on (`dcg_manager.h:51`).
pub type StageThreeDcg = DcgManager<NoSenProg, true, true>;

impl<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool> Default
    for DcgManager<C, DT2, L3_DL_SCHEDULER>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool>
    DcgManager<C, DT2, L3_DL_SCHEDULER>
{
    /// A manager with `firstAvailGlobalGrpId = 0` (`dcg_manager.h:44`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            first_avail_global_grp_id: GtrGroupId(0),
            enable_prog_verification: false,
            codegen: PhantomData,
        }
    }

    /// `enable_prog_verification_ = true` (`dcg_manager.h:38`) — turns the per-core program
    /// verification DT_ERRORs of entries 280 and 281 on.
    #[must_use]
    pub const fn with_program_verification(mut self) -> Self {
        self.enable_prog_verification = true;
        self
    }

    /// `DT_ERROR("Program verification failed for core " + ..)` (`:181-183`, `:200-202`, `:542-544`,
    /// `:560-562`) — a diagnostic abort, and only when verification is on.
    fn verification_failure(&self, core: Core, message: &str) {
        if self.enable_prog_verification {
            panic!(
                "Program verification failed for core {}\nError message: {message}",
                core.get()
            );
        }
    }

    /// `firstAvailGlobalGrpId`, which entries 188, 189 and 190 all advance — 188 and 189 through the
    /// front end's `int&` (`dcg_frontend.h:129`), 190 in its own body (`dcg_manager.cpp:500-501`).
    #[must_use]
    pub const fn first_avail_global_grp_id(&self) -> GtrGroupId {
        self.first_avail_global_grp_id
    }

    /// Replaces: e188_runDcgComputeTransfer
    ///
    /// Hands every data-op DSC of the super-DSC to the front end's transfer computation, in order
    /// (`dcg_manager.cpp:112-124`).
    ///
    /// ⭐ `int c` IS THE INDEX, NOT A COUNT: it is the argument the front end uses to find the same
    /// data op again inside the super-DSC, which is why the seam here takes the index rather than
    /// the `&mut DataOpDsc` the reference passes alongside a `&mut SuperDsc` aimed at it.
    /// ⛔ `&mut self` BECAUSE THE WALK ADVANCES `firstAvailGlobalGrpId`: the front end holds it by
    /// `int&` (`dcg_frontend.h:129`) and writes it per multicast transfer
    /// (`transfer_compute.cpp:74`, `:114` → `stcdpOp.cpp:2586-2587`, `:2662-2663`).
    pub fn run_dcg_compute_transfer<const FOLDED: bool>(
        &mut self,
        sdsc: &mut SuperDsc<true, FOLDED>,
    ) {
        for c in 0..sdsc.data_op_dscs.len() {
            DcgFrontEnd::compute_transfer_for_data_op(
                sdsc,
                DataOpIndex(c),
                &mut self.first_avail_global_grp_id,
            );
        }
    }

    /// Replaces: e189_runDcgForDlOps
    ///
    /// Generates each core's L3 load/store pcfg, then MERGES both into the super-DSC's pool slot for
    /// that `[core][component]` pair, allocating a slot when the pair has none
    /// (`dcg_manager.cpp:216-267`).
    ///
    /// ⚠️ `bool useDt1 = false` IS DEAD (`:231`) — nothing writes it, so `dtVersion > 1 && !useDt1`
    /// is `DT2` alone. On version 1 the generated pcfgs reach the pool NOWHERE and only the codegen
    /// tail sees them.
    /// ⛔ `&mut self` BECAUSE THE GENERATOR ADVANCES `firstAvailGlobalGrpId` past the largest group
    /// id any core used (`pcfg_gen.cpp:157-158`), through the `int&` at `dcg_frontend.h:129`.
    pub fn run_dcg_for_dl_ops<const FOLDED: bool>(&mut self, sdsc: &mut SuperDsc<false, FOLDED>)
    where
        C: DlOpsSenProg,
    {
        let mut halves =
            DcgFrontEnd::generate_pcfg_ir_for_dl_op(sdsc, &mut self.first_avail_global_grp_id);

        if DT2 {
            for (core, core_halves) in &mut halves {
                for (comp, local) in core_halves.each_mut() {
                    SenPcfg::merge(sdsc.pcfg_slot(*core, comp), local);
                }
            }
        }

        if C::CREATE_SEN_PROG {
            C::convert_to_prog_ir_dl_op(sdsc, halves);
        }
    }

    /// Replaces: e190_runDcgForDlOpsStandalone
    ///
    /// STAGE 3 ITSELF. Per core: default-construct that core's two L3 pcfg entries, fill them —
    /// through the DSC-2 translator when the L3 DL scheduler is on and the DSC has a schedule tree,
    /// otherwise through the front end's per-unit generator — and keep the largest GTR group id any
    /// of them used, which advances `firstAvailGlobalGrpId` (`dcg_manager.cpp:449-512`).
    ///
    /// ⛔ THE TWO ARMS DO NOT WRITE THE SAME PLACE. The translator arm fills
    /// `mySDsc.pcfg_[coreId][L3LU|L3SU]`, which survives the call; the fallback arm fills the LOCAL
    /// `pcfgL3lu`/`pcfgL3su`, which escape nowhere — only their group ids do (`:489-496`).
    pub fn run_dcg_for_dl_ops_standalone<const DATA_OPS: bool, const FOLDED: bool>(
        &mut self,
        sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    ) where
        C: StandaloneSenProg,
    {
        let mut local: BTreeMap<Core, L3Halves> = BTreeMap::new();
        let mut max_grp_id_in_l3 = GtrGroupId(0);

        for (core, version) in sdsc.dscs.iter().collect::<Vec<_>>() {
            sdsc.pcfg_entry(core, SenComponent::L3lu);
            sdsc.pcfg_entry(core, SenComponent::L3su);

            let maxes = if L3_DL_SCHEDULER && version == DscVersion::Dsc2 {
                DscPcfgTranslator::transform_dsc_comp_to_pcfg(sdsc, PcfgInfo::l3_lu(core));
                DscPcfgTranslator::transform_dsc_comp_to_pcfg(sdsc, PcfgInfo::l3_su(core));
                [
                    sdsc.pcfg_entry(core, SenComponent::L3lu).max_gtr_group_id(),
                    sdsc.pcfg_entry(core, SenComponent::L3su).max_gtr_group_id(),
                ]
            } else {
                let mut halves = L3Halves::default();
                let maxes = [
                    DcgFrontEnd::create_pcfg_for_unit_per_core(
                        sdsc,
                        &mut halves.lu,
                        SenComponent::L3lu,
                        core,
                        self.first_avail_global_grp_id,
                    ),
                    DcgFrontEnd::create_pcfg_for_unit_per_core(
                        sdsc,
                        &mut halves.su,
                        SenComponent::L3su,
                        core,
                        self.first_avail_global_grp_id,
                    ),
                ];
                local.insert(core, halves);
                maxes
            };

            for max in maxes {
                max_grp_id_in_l3 = max_grp_id_in_l3.max(max);
            }
        }

        self.first_avail_global_grp_id = self
            .first_avail_global_grp_id
            .advanced_past(max_grp_id_in_l3);

        if C::CREATE_SEN_PROG {
            C::fill_and_create_sen_prog_info(sdsc);
        }
    }

    /// Replaces: e191_removeextraPTrows
    ///
    /// Drops the PT-row pcfgs a re-stickify-with-PT data op leaves behind: from every such op's own
    /// per-core pcfg list always, and from the super-DSC's pool-and-map (version 2) or its `pcfg_`
    /// (version 1) once ANY such op was seen (`dcg_manager.cpp:568-627`).
    ///
    /// ⚠️ THE TWO VERSIONS DISAGREE ON WHEN: version 2 sweeps for any number of data ops, version 1
    /// only when there is MORE THAN ONE (`:611`) — a single re-stickify op keeps its PT rows in
    /// `pcfg_`. ⛔ And row 0 is never in the set; see [`PT_ROW_PCFGS_TO_REMOVE`].
    pub fn remove_extra_pt_rows<const FOLDED: bool>(&self, sdsc: &mut SuperDsc<true, FOLDED>) {
        let mut removed_a_pt_pcfg_node = false;

        for data_op in &mut sdsc.data_op_dscs {
            if matches!(
                data_op.op,
                OpFunc::ReStickifyOpWithPtLx | OpFunc::ReStickifyOpWithPthbm
            ) {
                removed_a_pt_pcfg_node = true;
                for per_core in data_op.pcfg.values_mut() {
                    per_core.retain(|(_, comp)| !PT_ROW_PCFGS_TO_REMOVE.contains(comp));
                }
            }
        }

        if !removed_a_pt_pcfg_node {
            return;
        }

        if DT2 {
            let to_remove: Vec<(Core, SenComponent, PcfgId)> = sdsc
                .pcfg_map
                .iter()
                .flat_map(|(core, by_comp)| {
                    by_comp
                        .iter()
                        .filter(|(comp, _)| PT_ROW_PCFGS_TO_REMOVE.contains(comp))
                        .map(|(comp, id)| (*core, *comp, *id))
                })
                .collect();
            for (core, comp, id) in to_remove {
                sdsc.pcfg_pool.remove(&id);
                if let Some(by_comp) = sdsc.pcfg_map.get_mut(&core) {
                    by_comp.remove(&comp);
                }
            }
        } else if sdsc.data_op_dscs.len() > 1 {
            for by_comp in sdsc.pcfg.values_mut() {
                by_comp.retain(|comp, _| !PT_ROW_PCFGS_TO_REMOVE.contains(comp));
            }
        }
    }

    /// Replaces: e192_printSenProgram
    ///
    /// Writes the super-DSC's per-core programs out as SEN programs, or as SMC when asked
    /// (`dcg_manager.cpp:959-976`).
    ///
    /// ⭐ THE SINK IS THE ARGUMENT, NOT A FILE NAME. The reference opens an `ofstream` and
    /// `DT_ERROR_FMT`s when it will not open (`:963-965`) — a runtime refusal with no Rust form, and
    /// none is needed: an already-open sink cannot fail to open.
    pub fn print_sen_program<A: Arch, M: Model, W: Workload>(
        &self,
        programs: &BTreeMap<Core, progir::Program<A, M, W>>,
        dpc: &mut Dpc,
        out: &mut dyn core::fmt::Write,
        form: SenProgramForm,
    ) {
        match form {
            SenProgramForm::SenProg => dpc.convert_ir_to_sen_prog(programs, out),
            SenProgramForm::Smc => dpc.convert_ir_to_smc(programs, out),
        }
    }

    /// Replaces: e193_runDcgForGenKG3
    ///
    /// Hands the super-DSC to the backend's generic KG3 path (`dcg_manager.h:94-96`).
    pub fn run_dcg_for_gen_kg3<const DATA_OPS: bool, const FOLDED: bool>(
        &self,
        sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    ) {
        DcgBackEnd::run_dcg_for_gen_kg3(sdsc, C::CREATE_SEN_PROG);
    }

    /// Replaces: e194_runDcgForSparseKG3CONV2D
    ///
    /// Hands the super-DSC to the backend's sparse-KG3 2-D convolution path (`dcg_manager.h:98-100`).
    pub fn run_dcg_for_sparse_kg3_conv2d<const DATA_OPS: bool, const FOLDED: bool>(
        &self,
        sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    ) {
        DcgBackEnd::run_dcg_for_sparse_kg3_conv2d(sdsc, C::CREATE_SEN_PROG);
    }

    /// Replaces: e195_runDcgForSparseKG3BMM
    ///
    /// Hands the super-DSC to the backend's sparse-KG3 batched-matmul path (`dcg_manager.h:102-104`).
    pub fn run_dcg_for_sparse_kg3_bmm<const DATA_OPS: bool, const FOLDED: bool>(
        &self,
        sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
    ) {
        DcgBackEnd::run_dcg_for_sparse_kg3_bmm(sdsc, C::CREATE_SEN_PROG);
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// e196_printTrafficPerCore — the per-core traffic table.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// WHICH WAY ROUND THE CORE RING A MULTICAST TRAVELS — `selectedMCMode` (`dsc/dataOpDsc.h:212-213`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MulticastMode {
    /// `-1`, the field's initial value: nothing has chosen an arc yet.
    #[default]
    Unselected,
    /// `0` — random.
    Random,
    /// `1` — counter-clockwise, one hop being core `i` to core `i + 1`.
    CounterClockwise,
    /// `2` — clockwise.
    Clockwise,
    /// `3` — replication: whichever arc is under half the ring, else both at half the volume.
    Replication,
}

/// HOW FAR ROUND THE RING A TRANSFER REACHES — one end of `CCWHopCWHop` (`dsc/dataOpDsc.h:215`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct RingHops(pub u32);

/// ONE PRODUCED TRANSFER of a `coreIDtoDtKey_L3SU` entry (`stcdpOp.cpp:5875-5915`).
#[derive(Debug, Clone, Copy)]
pub struct Multicast {
    /// `CCWHopCWHop.first`.
    pub ccw_hops: RingHops,
    /// `CCWHopCWHop.second`.
    pub cw_hops: RingHops,
    /// `selectedMCMode`.
    pub mode: MulticastMode,
    /// The transfer's volume — see [`transfer_sticks`].
    pub volume: Sticks,
}

/// ONE `coreIDtoDtKey_L3SU` ENTRY: a producing core and its transfers (`stcdpOp.cpp:5870-5873`).
///
/// ⛔ THE CORE IS NOT PER-TRANSFER. `initCore(coreID)` runs on the KEY, before the transfer list
/// (`:5873`), so a producer with an empty list still gets a row in the report.
#[derive(Debug, Clone)]
pub struct Produced {
    /// The producing core.
    pub producer: Core,
    /// Its transfers, in `dtTable_` order.
    pub transfers: Vec<Multicast>,
}

/// ONE `coreIDtoDtKey_L3LU` ENTRY: a consuming core and the volumes it takes in (`:5918-5926`).
#[derive(Debug, Clone)]
pub struct Consumed {
    /// The consuming core.
    pub consumer: Core,
    /// Its transfers' volumes.
    pub volumes: Vec<Sticks>,
}

/// ONE STCDP DATA OP'S CONTRIBUTION TO THE TRAFFIC TABLE.
#[derive(Debug, Clone, Default)]
pub struct DataOpTraffic {
    /// `STCDPOpHBM` seeds EVERY core into the table whether or not it carries traffic (`:5862-5863`);
    /// `STCDPOpLx` seeds only the cores its transfers name.
    pub seeds_every_core: bool,
    /// The producer side (`:5870`).
    pub produced: Vec<Produced>,
    /// The consumer side (`:5918`).
    pub consumed: Vec<Consumed>,
    /// Volumes of this op's transfers produced by the HBM — `pMemID == -1`, since an LX producer's
    /// mem id is its core id (`:5930-5931`, `dsc/dataOpDsc.h:184`).
    pub hbm_sourced: Vec<Sticks>,
}

/// THE STICKS A TRANSFER'S ELEMENTS OCCUPY — `128.0 / inpLds->wordLength` elements to a stick
/// (`stcdpOp.cpp:5868`), truncating exactly where the reference's `int` accumulators truncate.
#[must_use]
pub fn transfer_sticks<A: Arch>(elements: Elements, word: Bytes) -> Sticks {
    Sticks(elements.0 * word.0 / A::BYTES_PER_STICK.get())
}

/// ONE CORE'S ROW OF THE TRAFFIC TABLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoreTraffic {
    incoming: Sticks,
    outgoing: Sticks,
    through: Sticks,
}

impl Default for CoreTraffic {
    /// What `initCore` seeds a fresh core with (`stcdpOp.cpp:5856-5858`).
    fn default() -> Self {
        CoreTraffic {
            incoming: Sticks(0),
            outgoing: Sticks(0),
            through: Sticks(0),
        }
    }
}

/// THE TABLE `printTrafficPerCore` FILLS — one slot per core of the ring.
///
/// ⛔ `Option`, BECAUSE `initCore` DECIDES WHICH ROWS PRINT. The reference's three `std::map`s hold
/// only the cores something touched and it renders by walking `coreIdToInpVol` (`:5957`), so a core
/// no transfer mentioned has NO row — which is not the same as a row of zeroes. The three maps also
/// collapse into one record per core, which is what retires its `DT_CHECK` at `:5958`: a slot cannot
/// hold an incoming count without the outgoing one beside it.
struct TrafficTable {
    cores: [Option<CoreTraffic>; Target::CORES as usize],
    total_produced: Sticks,
}

impl TrafficTable {
    fn new() -> Self {
        TrafficTable {
            cores: [None; Target::CORES as usize],
            total_produced: Sticks(0),
        }
    }

    /// `initCore` — and the row it hands back (`:5855-5859`).
    fn entry(&mut self, core: Core) -> &mut CoreTraffic {
        self.cores[core.get() as usize].get_or_insert_default()
    }

    /// `STCDPOpHBM`'s seeding of the whole ring (`:5863`), and the HBM through-charge's (`:5934`).
    fn seed_every_core(&mut self) {
        for slot in &mut self.cores {
            slot.get_or_insert_default();
        }
    }

    /// The through-traffic a CCW arc of `hops` passes over — `for (c = 1; c < CCW_hop; c++)`.
    fn pass_ccw(&mut self, producer: Core, hops: RingHops, volume: Sticks) {
        for hop in 1..hops.0 {
            self.entry(producer.step_ccw(hop)).through.0 += volume.0;
        }
    }

    /// The same over a CW arc — `for (c = 1; c < CW_hop; c++)`.
    fn pass_cw(&mut self, producer: Core, hops: RingHops, volume: Sticks) {
        for hop in 1..hops.0 {
            self.entry(producer.step_cw(hop)).through.0 += volume.0;
        }
    }

    /// One producer's outgoing volume and the cores its arc passes over (`:5870-5916`).
    fn add_produced(&mut self, produced: &Produced) {
        self.entry(produced.producer);
        for transfer in &produced.transfers {
            self.entry(produced.producer).outgoing.0 += transfer.volume.0;
            self.total_produced.0 += transfer.volume.0;

            // ⛔ ARM ORDER IS THE REFERENCE'S `if / else if / else` (`:5885-5914`): mode 3 takes an
            // arc only when THAT arc is under half the ring, and every other mode — including the
            // unselected -1 and random 0 — splits the volume over both.
            let half = Sticks(transfer.volume.0 / 2);
            let half_ring = Target::CORES / 2;
            match transfer.mode {
                MulticastMode::CounterClockwise => {
                    self.pass_ccw(produced.producer, transfer.ccw_hops, transfer.volume);
                }
                MulticastMode::Replication if transfer.ccw_hops.0 < half_ring => {
                    self.pass_ccw(produced.producer, transfer.ccw_hops, transfer.volume);
                }
                MulticastMode::Clockwise => {
                    self.pass_cw(produced.producer, transfer.cw_hops, transfer.volume);
                }
                MulticastMode::Replication if transfer.cw_hops.0 < half_ring => {
                    self.pass_cw(produced.producer, transfer.cw_hops, transfer.volume);
                }
                MulticastMode::Unselected | MulticastMode::Random | MulticastMode::Replication => {
                    self.pass_ccw(produced.producer, transfer.ccw_hops, half);
                    self.pass_cw(produced.producer, transfer.cw_hops, half);
                }
            }
        }
    }

    /// One consumer's incoming volume (`:5918-5927`).
    fn add_consumed(&mut self, consumed: &Consumed) {
        self.entry(consumed.consumer);
        for volume in &consumed.volumes {
            self.entry(consumed.consumer).incoming.0 += volume.0;
        }
    }

    /// An HBM-produced transfer: half its volume passes through EVERY core (`:5930-5939`).
    fn add_hbm_sourced(&mut self, volume: Sticks) {
        self.seed_every_core();
        for slot in &mut self.cores {
            if let Some(traffic) = slot {
                traffic.through.0 += volume.0 / 2;
            }
        }
    }

    /// The dump itself (`:5949-5965`). `std::setw` right-aligns, and the reference's THIRD output
    /// line is the `setw`-only `<<` chain at `:5954-5955`, which emits nothing but its newline.
    fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("---------------------------------------------------------------\n");
        out.push_str(&format!(
            "{:>12}{:>20}{:>20}{:>20}{:>20}\n",
            "Core Id", "Incoming Sticks", "Outgoing Sticks", "In/out Sticks", "Through Sticks",
        ));
        out.push('\n');
        for (index, slot) in self.cores.iter().enumerate() {
            if let Some(traffic) = slot {
                out.push_str(&format!(
                    "{:>12}{:>20}{:>20}{:>20}{:>20}\n",
                    index,
                    traffic.incoming.0,
                    traffic.outgoing.0,
                    traffic.outgoing.0 + traffic.incoming.0,
                    traffic.through.0,
                ));
            }
        }
        out.push_str(&format!("totalProdSticks={}\n", self.total_produced.0));
        out
    }
}

/// Replaces: e196_printTrafficPerCore
///
/// THE PER-CORE TRAFFIC TABLE — incoming, outgoing and through-passing sticks per core, then
/// `totalProdSticks`. `DcgManager::printTrafficPerCore` (`dcg/dcg_manager/dcg_manager.h:120`) is a
/// forward to `DcgFE::printTrafficPerCore` (`dcg/dcg_fe/pcfg_gen/stcdpOp.cpp:5849`), which is this.
///
/// ⛔ THIS PORT DIVERGES, AND THE REFERENCE IS WRONG: its CCW ring step is `(c + coreID) / 32`
/// (`:5889`, `:5904`) where every sibling wraps modularly — its own second copy of the identical walk
/// writes `% (int)maxNumCores` (`inputNeighFetchOp.cpp:2294`, `:2308`, `:2318`) and the CW arm of the
/// same `if` chain wraps by hand (`:5897-5898`, `:5909-5910`) — so `/ 32` charges every CCW hop
/// landing below core 32 to core 0.
/// [`Core::step_ccw`] steps the ring; `unit_tests::ccw_arc_walks_the_ring_not_core_zero` pins it.
///
/// ⛔ TWO SEAMS THE CALLER OWNS. The `sdsc.dataOpdscs_` walk that reads each `baseSTCDPOp`'s
/// `dtTable_` and `coreIDtoDtKey_L3{SU,LU}` is `dcg_fe/pcfg_gen/`, OUT of this campaign, so the
/// caller supplies [`DataOpTraffic`]; and the file write is the caller's, since the reference's own
/// open failure is a message and a no-op (`:5946-5947`) leaving only the text to preserve.
/// ⚠️ NO CALLER ON THE `runDdc` PATH — the only one is `dcg/tools/dcg_standalone.cpp:553`, a harness.
#[must_use]
pub fn print_traffic_per_core(ops: &[DataOpTraffic]) -> String {
    let mut table = TrafficTable::new();
    for op in ops {
        if op.seeds_every_core {
            table.seed_every_core();
        }
        for produced in &op.produced {
            table.add_produced(produced);
        }
        for consumed in &op.consumed {
            table.add_consumed(consumed);
        }
        for &volume in &op.hbm_sourced {
            table.add_hbm_sourced(volume);
        }
    }
    table.report()
}

impl<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool>
    DcgManager<C, DT2, L3_DL_SCHEDULER>
{
    /// Replaces: e280_runDcgGenerateProgIR
    ///
    /// Generates the SEN programs for a super-DSC's ONE data op — verifying each core's, and
    /// regenerating them all once when the I-buff overflowed — or, for more than one data op, from the
    /// super-DSC's own pcfgs; then drops the extra PT-row pcfgs (`dcg_manager.cpp:145-214`).
    ///
    /// ⛔ THE `GatherOpHBM` ARM'S `coreArch >= IsaCoreGen::MPW4_ISA` (`:168-169`) CANNOT FAIL HERE:
    /// [`crate::arch::IsaGen`] carries no generation below MPW4, so that check is a compile-time truth
    /// and its `senCompToISAptr.count(L3LU)` guard (`:167`) a table the compiler folds.
    /// ⛔ `None` IS `DT_CHECK(mySDsc.dataOpdscs_.size())` (`:163`): with no data op the reference
    /// throws, which skips its own otherwise-unconditional `removeextraPTrows` tail (`:213`).
    pub fn run_dcg_generate_prog_ir<const FOLDED: bool>(
        &self,
        sdsc: &mut SuperDsc<true, FOLDED>,
    ) -> Option<()>
    where
        C: DcgProgIRGen,
    {
        // ⛔ THE FOLD CHECK IS `createSenProg`'S AND THE CONJUNCTION IS LOAD-BEARING (`:155-157`).
        // An inline `const` block is evaluated at MONOMORPHIZATION however dead the branch around
        // it, so `!FOLDED` alone refused a folded super-DSC with codegen off — which the reference
        // accepts, and which is THIS campaign's configuration (`SchedulerStages.cpp:49-50`).
        const {
            assert!(
                !(C::CREATE_SEN_PROG && FOLDED),
                "Codegen for Folded Super-DSC is not supported (dcg_manager.cpp:156)"
            )
        }
        if C::CREATE_SEN_PROG {
            if sdsc.data_op_dscs.len() <= 1 {
                let mut at = DataOpTarget::InMain(DataOpIndex(0));
                let op = sdsc.data_op_dscs.first()?.op;
                let mut statuses = DcgBackEnd::create_sen_program_stcdp_op(sdsc, &at);

                let mut violation = None;
                for (core, (error, message)) in &statuses {
                    match IbuffViolation::witnessed(op, *error) {
                        Some(witness) => violation = Some(witness),
                        None => self.verification_failure(*core, message),
                    }
                }

                if let Some(witness) = violation {
                    statuses.clear();
                    DcgFrontEnd::finalize_burst_info(sdsc, &mut at, witness);
                    DcgFrontEnd::create_pcfgs_stcdp_op(sdsc, &mut at, true);
                    statuses = DcgBackEnd::create_sen_program_stcdp_op(sdsc, &at);
                }

                // ⛔ THE SECOND PASS DOES NOT LOOK AT THE STATUS: any core still filed is a failure
                // once verification is on, I-buff overflow included (`:197-204`).
                for (core, (_, message)) in &statuses {
                    self.verification_failure(*core, message);
                }
            } else {
                DcgBackEnd::fill_and_create_sen_prog_info(sdsc);
            }
        }

        self.remove_extra_pt_rows(sdsc);
        Some(())
    }
}

/// THE MANAGER ONCE IT IS IN INPUT-FETCH-NEIGHBOUR MODE — `isInpFetchNeigh == true` (`:523`).
///
/// ⛔ THE FLAG *IS* CLEARED, AND IT STILL STICKS — BY ORDER, NOT BY ABSENCE. The front end holds it
/// as a `bool&` (`dcg_manager.h:59`, `dcg_frontend.h:125`) and `generatePcfgIRForDataOpInpFetch` sets
/// it on entry and CLEARS IT ON EXIT (`pcfg_gen.cpp:164`, `:184`); entry 281 writes `:523` AFTER that
/// call, and the clearing function's only other caller — entry 348's `:401` — is unreachable once the
/// flag is set, because `:347` checks it first.
/// ⛔ THIS TYPE *IS* THE `DT_CHECK(!isInpFetchNeigh)` OF ENTRIES 188, 189 AND 280 (`:118`, `:221`,
/// `:151`): those are [`DcgManager`] methods and entry 281 consumes the manager, so calling one after
/// the mode has flipped does not compile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InpFetchNeighDcg<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool> {
    first_avail_global_grp_id: GtrGroupId,
    enable_prog_verification: bool,
    /// `myIFNInfo.dataDsc_Idx` (`:520`).
    data_dsc_idx: DataOpIndex,
    codegen: PhantomData<C>,
}

impl<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool>
    InpFetchNeighDcg<C, DT2, L3_DL_SCHEDULER>
{
    /// `firstAvailGlobalGrpId`, carried across the mode change (`dcg_manager.h:43-44`).
    #[must_use]
    pub const fn first_avail_global_grp_id(&self) -> GtrGroupId {
        self.first_avail_global_grp_id
    }

    /// `myIFNInfo.dataDsc_Idx` (`:520`).
    #[must_use]
    pub const fn data_dsc_idx(&self) -> DataOpIndex {
        self.data_dsc_idx
    }
}

impl<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool>
    DcgManager<C, DT2, L3_DL_SCHEDULER>
{
    /// Replaces: e281_runDcgForInputFetchNeighbor
    ///
    /// Generates the input-fetch-neighbour data op's pcfg IR from the main super-DSC and the preceding
    /// one, then that op's SEN programs, and leaves the manager in input-fetch-neighbour mode for good
    /// (`dcg_manager.cpp:514-566`).
    ///
    /// ⛔ THE MODE IS A ONE-WAY DOOR, WHICH IS WHY THIS CONSUMES `self` — see [`InpFetchNeighDcg`].
    /// ⛔ THE I-BUFF ARM IS THE REFERENCE'S OWN ABORT: `DT_CHECK(0); // not ready` (`:552`) stands
    /// between the clear and the regeneration, so that path is unfinished THERE and not here.
    pub fn run_dcg_for_input_fetch_neighbor<const FOLDED: bool>(
        self,
        main: &mut SuperDsc<true, FOLDED>,
        pre: Option<&mut SuperDsc<true, FOLDED>>,
    ) -> InpFetchNeighDcg<C, DT2, L3_DL_SCHEDULER> {
        let data_dsc_idx = DataOpIndex(0);
        let at = DcgFrontEnd::generate_pcfg_ir_for_data_op_inp_fetch(main, pre, data_dsc_idx);

        // ⛔ `createSenProg && FOLDED`, for the reason entry 280 states (`:526-528`).
        const {
            assert!(
                !(C::CREATE_SEN_PROG && FOLDED),
                "Codegen for Folded Super-DSC is not supported (dcg_manager.cpp:527)"
            )
        }
        if C::CREATE_SEN_PROG {
            let mut statuses = DcgBackEnd::create_sen_program_stcdp_op(main, &at);
            let op = at.op(main);

            let mut violation = None;
            for (core, (error, message)) in &statuses {
                match op.and_then(|op| IbuffViolation::witnessed(op, *error)) {
                    Some(witness) => violation = Some(witness),
                    None => self.verification_failure(*core, message),
                }
            }

            if violation.is_some() {
                statuses.clear();
                panic!(
                    "dcg_manager.cpp:552: DT_CHECK(0) — I-buff regeneration is not ready for \
                     input-fetch-neighbour"
                );
            }

            for (core, (_, message)) in &statuses {
                self.verification_failure(*core, message);
            }
        }

        InpFetchNeighDcg {
            first_avail_global_grp_id: self.first_avail_global_grp_id,
            enable_prog_verification: self.enable_prog_verification,
            data_dsc_idx,
            codegen: PhantomData,
        }
    }
}

impl<C: SenProgGen, const DT2: bool, const L3_DL_SCHEDULER: bool>
    DcgManager<C, DT2, L3_DL_SCHEDULER>
{
    /// Replaces: e282_mergePcfgInSuperDSC
    ///
    /// REBUILDS `pcfg_` FROM THE PER-CORE SCHEDULE: per step, merges that step's data-op pcfgs and its
    /// DL pcfg into the core's per-unit master pcfg, brackets each merge with the syncs the step asks
    /// for, gives every unit that got no sync one anyway, marks the lx units' before-blocks, and then —
    /// on version 2 outside DCC — re-pools the result and erases the slots nothing names any more
    /// (`dcg_manager.cpp:635-896`).
    ///
    /// ⚠️ `bool useDt1 = false` IS DEAD (`:644`), so both of its guards are `DT2` alone.
    /// ⛔ `None` IS EVERY `.at()` THE REFERENCE THROWS FROM: a step naming a data op or a DSC the
    /// super-DSC has not got, a core the named one does not serve, or a pool slot the map points at
    /// and the pool has lost (`:718`, `:729`, `:746`, `:757`, `:769`, `:785`, `:804`).
    /// ⭐ THE CHILD PCFG IS CLONED BECAUSE THE MERGE ONLY READS IT (`dsc/pcfg.cpp:162-283` writes
    /// nothing through `child`) while parent and child both live inside this one super-DSC.
    pub fn merge_pcfg_in_super_dsc<const DATA_OPS: bool, const FOLDED: bool>(
        &self,
        sdsc: &mut SuperDsc<DATA_OPS, FOLDED>,
        dl_pcfg: DlPcfgSource<'_>,
    ) -> Option<()> {
        sdsc.pcfg.clear();

        let schedules = sdsc.core_schedules.clone();
        for (&core, schedule) in &schedules {
            let has_dl_dsc = schedule.has_dl_dsc();

            // `dataDscUnits` (`:657-676`) — every unit ANY data op holds a pcfg for on this core, and
            // gathered only when the core runs a DL DSC.
            let mut data_dsc_units: BTreeSet<SenComponent> = BTreeSet::new();
            if has_dl_dsc {
                for ddsc in &sdsc.data_op_dscs {
                    if let Some(pcfgs) = ddsc.pcfg.get(&core) {
                        data_dsc_units.extend(pcfgs.iter().map(|(_, comp)| *comp));
                    }
                }
            }

            for (step, seen_dl_dsc) in schedule.steps() {
                // What this step merges, in the reference's own order: the data op's pcfgs first
                // (`:730-734`), then the DL one (`:744-790`).
                let mut to_merge: Vec<(SenPcfg, SenComponent, bool)> = Vec::new();

                if let Some(at) = step.data_dsc {
                    let ddsc = sdsc.data_op_dscs.get(at.get())?;
                    let pcfgs = ddsc.pcfg.get(&core)?;
                    to_merge.extend(
                        pcfgs
                            .iter()
                            .map(|(pcfg, comp)| (pcfg.clone(), *comp, false)),
                    );
                }

                if let Some(dl) = step.dl_dsc
                    && (step.data_dsc.is_none() || sdsc.target == SenTarget::SenPcfg)
                {
                    // `None` IS `DT_CHECK(useActualDLpcfg)` (`:740-742`).
                    if sdsc.target == SenTarget::SenPcfg
                        && step.data_dsc.is_some()
                        && !matches!(dl_pcfg, DlPcfgSource::Dsc)
                    {
                        return None;
                    }

                    match dl_pcfg {
                        DlPcfgSource::Dsc => match sdsc.dscs.at(dl)?.pcfgs.get(&core)? {
                            // ⭐ THE DSC'S OWN PCFG GOES IN UNDER `L3LU` WHATEVER IT HOLDS (`:763`).
                            Some(pcfg) => {
                                if pcfg.has_graph() {
                                    to_merge.push((pcfg.clone(), SenComponent::L3lu, true));
                                }
                            }
                            // `dsc.pcfg_.size() <= coreIDX` — the pool drives the merge instead.
                            None => {
                                if let Some(by_comp) = sdsc.pcfg_map.get(&core) {
                                    // `DT_CHECK(dscGlobal->dtVersion > 1 && !useDt1)` (`:766`).
                                    if !DT2 {
                                        return None;
                                    }
                                    for (comp, id) in by_comp {
                                        let pcfg = sdsc.pcfg_pool.get(id)?;
                                        let needs_merge = data_dsc_units.contains(comp)
                                            || ALL_UNITS.contains(comp)
                                            || C::DCC;
                                        if needs_merge && pcfg.has_graph() {
                                            to_merge.push((pcfg.clone(), *comp, true));
                                        }
                                    }
                                }
                            }
                        },
                        DlPcfgSource::L3Halves(halves) => {
                            if !halves.is_empty() {
                                let pair = halves.get(&core)?;
                                for (comp, pcfg) in [
                                    (SenComponent::L3lu, &pair.lu),
                                    (SenComponent::L3su, &pair.su),
                                ] {
                                    if pcfg.has_graph() {
                                        to_merge.push((pcfg.clone(), comp, true));
                                    }
                                }
                            }
                        }
                    }
                }

                // `mergePcfg` (`:682-714`) — the lambda, once per child.
                let mut unit_sync_inserted: BTreeSet<SenComponent> = BTreeSet::new();
                for (mut child, comp, dldsc) in to_merge {
                    let master = sdsc.pcfg.entry(core).or_default().entry(comp).or_default();
                    if step.before_sync && ALL_UNITS.contains(&comp) {
                        DcgFrontEnd::add_sync_node(
                            master,
                            comp,
                            core,
                            &step.sync_tag("before", dldsc),
                        );
                        unit_sync_inserted.insert(comp);
                    }
                    SenPcfg::merge(master, &mut child);
                    if step.after_sync && ALL_UNITS.contains(&comp) {
                        DcgFrontEnd::add_sync_node(
                            master,
                            comp,
                            core,
                            &step.sync_tag("after", dldsc),
                        );
                        unit_sync_inserted.insert(comp);
                    }
                }

                // "add missing unit if sync is required" (`:798-835`).
                if step.after_sync || step.before_sync {
                    for unit in ALL_UNITS {
                        if unit_sync_inserted.contains(&unit) {
                            continue;
                        }
                        unit_sync_inserted.insert(unit);

                        let master = sdsc.pcfg.get_mut(&core)?.entry(unit).or_default();
                        let mut new_sync_node = None;
                        if step.before_sync {
                            new_sync_node = Some(DcgFrontEnd::add_sync_node(
                                master,
                                unit,
                                core,
                                &step.missing_unit_tag("before"),
                            ));
                        }
                        if step.after_sync {
                            new_sync_node = Some(DcgFrontEnd::add_sync_node(
                                master,
                                unit,
                                core,
                                &step.missing_unit_tag("after"),
                            ));
                        }

                        // ⛔ THE BITS GO ON THE LAST NODE ADDED (`:824-831`), and `seenDLDsc` is
                        // already set for the dl step itself — so only a LEADING step is a "before".
                        if has_dl_dsc && LX_UNITS.contains(&unit) {
                            let marks = master.marks_mut(new_sync_node?);
                            marks.start_new_prog_ir_block = true;
                            marks.is_before_block = !seen_dl_dsc;
                        }
                    }
                }

                // "insert marking for before code blocks" — ⛔ THE L3 HALVES ARE SKIPPED (`:844-845`).
                if step.dl_dsc.is_some()
                    && let Some(by_comp) = sdsc.pcfg.get_mut(&core)
                {
                    for (unit, master) in by_comp.iter_mut() {
                        if matches!(*unit, SenComponent::L3lu | SenComponent::L3su) {
                            continue;
                        }
                        if let Some(src) = master.src_node() {
                            let tail = master.tail_node();
                            let marks = master.marks_mut(src);
                            marks.start_new_prog_ir_block = true;
                            marks.is_before_block = true;
                            if let Some(tail) = tail {
                                master.marks_mut(tail).end_before_block = true;
                            }
                        }
                    }
                }
            }
        }

        // "Copy back to PCFG pool" (`:861-895`).
        if DT2 && !C::DCC {
            // ⚠️ `pcfgPool_.rbegin()->first + 1`, NOT THE LOWEST FREE SLOT (`:864-865`).
            let mut next = sdsc
                .pcfg_pool
                .keys()
                .next_back()
                .map_or(0, |last| last.0 + 1);
            for (core, by_comp) in sdsc.pcfg.clone() {
                for (comp, pcfg) in by_comp {
                    let id = PcfgId(next);
                    sdsc.pcfg_map.entry(core).or_default().insert(comp, id);
                    sdsc.pcfg_pool.insert(id, pcfg);
                    next += 1;
                }
            }

            let used: BTreeSet<PcfgId> = sdsc
                .pcfg_map
                .values()
                .flat_map(|by_comp| by_comp.values().copied())
                .collect();
            sdsc.pcfg_pool.retain(|id, _| used.contains(id));
        }

        Some(())
    }
}

// crustify:todo: e326_mergePcfgInSuperDSC
//   authority : dcg/dcg_manager/dcg_manager.cpp:629  (5 body lines, level 2)
//   class     : DcgManager
//   original  : void DcgManager::mergePcfgInSuperDSC(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:756-761
//   calls     : e282_mergePcfgInSuperDSC

// crustify:todo: e327_printSenProgram
//   authority : dcg/dcg_manager/dcg_manager.cpp:945  (13 body lines, level 2)
//   class     : DcgManager
//   original  : void DcgManager::printSenProgram(SuperDsc* mySDsc, std::string fileName)
//   extract   : crustify-ddc/cpp/dcg.cpp:771-784
//   calls     : e076_print, e102_print, e272_print

// crustify:todo: e348_runDcgForDataOpsDlOps
//   authority : dcg/dcg_manager/dcg_manager.cpp:269  (179 body lines, level 3)
//   class     : DcgManager
//   original  : void DcgManager::runDcgForDataOpsDlOps(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:794-973
//   calls     : e104_clear, e187_clear, e191_removeextraPTrows, e282_mergePcfgInSuperDSC, e326_mergePcfgInSuperDSC

// crustify:todo: e349_convertToProgIRDataOp
//   authority : dcg/dcg_manager/dcg_manager.cpp:898  (46 body lines, level 3)
//   class     : DcgManager
//   original  : void DcgManager::convertToProgIRDataOp(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:983-1029
//   calls     : e282_mergePcfgInSuperDSC, e326_mergePcfgInSuperDSC
#[cfg(test)]
mod unit_tests {
    use super::*;

    /// A core, at a literal index this arch is known to have.
    fn core<const I: u32>() -> Core {
        Core::checked(I).expect("this arch has 32 cores")
    }

    /// `(core id, incoming, outgoing, in/out, through)` per rendered row, read back by splitting on
    /// whitespace — never by re-slicing at the widths the renderer used.
    fn rows(report: &str) -> Vec<(u32, u64, u64, u64, u64)> {
        report
            .lines()
            .skip(3)
            .filter(|line| !line.starts_with("totalProdSticks="))
            .map(|line| {
                let cells: Vec<u64> = line
                    .split_whitespace()
                    .map(|cell| cell.parse().expect("a rendered cell is a number"))
                    .collect();
                let [id, incoming, outgoing, in_out, through] = cells[..] else {
                    unreachable!("a rendered row has five columns")
                };
                (id as u32, incoming, outgoing, in_out, through)
            })
            .collect()
    }

    /// ⛔⛔ THE DIVERGENCE, PINNED. A 4-hop CCW multicast out of core 30 passes over cores 31, 0 and 1
    /// — `for (c = 1; c < 4; c++)` at `(30 + c) % 32` (`stcdpOp.cpp:5888`, wrapping as
    /// `inputNeighFetchOp.cpp:2294` does). The reference's `(c + coreID) / 32` would answer 0, 1, 1
    /// instead: core 31 would carry NO through traffic and core 1 would carry twice its share. The
    /// whole dump is compared, so the column layout of `:5949-5965` is pinned with it.
    #[test]
    fn ccw_arc_walks_the_ring_not_core_zero() {
        let report = print_traffic_per_core(&[DataOpTraffic {
            produced: vec![Produced {
                producer: core::<30>(),
                transfers: vec![Multicast {
                    ccw_hops: RingHops(4),
                    cw_hops: RingHops(28),
                    mode: MulticastMode::CounterClockwise,
                    volume: Sticks(10),
                }],
            }],
            ..DataOpTraffic::default()
        }]);

        assert_eq!(
            report,
            concat!(
                "---------------------------------------------------------------\n",
                "     Core Id     Incoming Sticks     Outgoing Sticks",
                "       In/out Sticks      Through Sticks\n",
                "\n",
                "           0                   0                   0",
                "                   0                  10\n",
                "           1                   0                   0",
                "                   0                  10\n",
                "          30                   0                  10",
                "                  10                   0\n",
                "          31                   0                   0",
                "                   0                  10\n",
                "totalProdSticks=10\n",
            ),
        );
    }

    /// THE VENDOR'S OWN TWO SPECIAL CASES IN ONE OP. A replication whose BOTH arcs reach half the
    /// ring or further takes neither branch and charges half its volume over each (`:5902-5913`), and
    /// an HBM-produced transfer charges half its volume through EVERY core (`:5930-5939`) — which,
    /// with `STCDPOpHBM`'s seeding (`:5862-5863`), is why all 32 rows print.
    #[test]
    fn replication_past_half_the_ring_splits_and_hbm_charges_every_core() {
        let report = print_traffic_per_core(&[DataOpTraffic {
            seeds_every_core: true,
            produced: vec![Produced {
                producer: core::<0>(),
                transfers: vec![Multicast {
                    ccw_hops: RingHops(20),
                    cw_hops: RingHops(20),
                    mode: MulticastMode::Replication,
                    volume: Sticks(8),
                }],
            }],
            consumed: Vec::new(),
            // 9 sticks halved is 4, truncating as the reference's `int` map does.
            hbm_sourced: vec![Sticks(9)],
        }]);

        let rows = rows(&report);
        assert_eq!(rows.len(), 32, "every core has a row");
        // The producer: 8 sticks out, and only the HBM's 4 passing through it.
        assert_eq!(rows[0], (0, 0, 8, 8, 4));
        // Core 12 is on the CCW arc alone (hops 1..19): 4 of the split, plus the HBM's 4.
        assert_eq!(rows[12], (12, 0, 0, 0, 8));
        // Core 13 is on BOTH arcs (CW reaches 13 at hop 19): 4 + 4, plus the HBM's 4.
        assert_eq!(rows[13], (13, 0, 0, 0, 12));
        // Core 20 is on the CW arc alone: 4 of the split, plus the HBM's 4.
        assert_eq!(rows[20], (20, 0, 0, 0, 8));
        assert!(report.ends_with("totalProdSticks=8\n"), "{report}");
    }

    /// One DL DSC, version 2, on core 0 — what stage 3 is handed (`SchedulerStages.cpp:53-57`).
    fn one_dsc2_core<const DATA_OPS: bool>() -> SuperDsc<DATA_OPS, false> {
        SuperDsc::of(PerCoreDscs::of(core::<0>(), DscVersion::Dsc2))
    }

    /// e188 — a super-DSC with no data ops asks the front end for nothing, so the seam is never
    /// reached, the walk simply ends (`dcg_manager.cpp:119-123`) and the group-id counter the
    /// transfers would have advanced (`stcdpOp.cpp:2586-2587`) still reads 0.
    #[test]
    fn e188_no_data_ops_reaches_no_transfer_seam() {
        let mut sdsc = one_dsc2_core::<true>();
        let mut dcg = StageThreeDcg::new();
        dcg.run_dcg_compute_transfer(&mut sdsc);
        assert!(sdsc.data_op_dscs.is_empty());
        assert_eq!(dcg.first_avail_global_grp_id(), GtrGroupId::wrapped(0));
    }

    /// e189 — its FIRST act is the pcfg generator, so with that out of scope the entry names it and
    /// stops before the merge (`dcg_manager.cpp:229`).
    #[test]
    #[should_panic(expected = "DcgFE::generatePcfgIRForDLOp")]
    fn e189_stops_at_the_pcfg_generator() {
        let mut sdsc = one_dsc2_core::<false>();
        DcgManager::<DcgSenProg, true, true>::new().run_dcg_for_dl_ops(&mut sdsc);
    }

    /// e190 — ⚠️ THE WRAP IS BY 63. `firstAvailGlobalGrpId += maxGrpIDinL3` then `%= maxGroupID`
    /// (`:500-501`) with `maxGroupID = 64 - 1` (`sysdef.cpp:230`), so 62 advanced past 2 is 1 and
    /// never 0 — a wrap by 64 would say 0, and 63 is not a value this can hold.
    #[test]
    fn e190_the_group_id_wraps_by_sixty_three() {
        let advanced = GtrGroupId::wrapped(62).advanced_past(GtrGroupId::wrapped(2));
        assert_eq!(advanced.get(), 1);
        assert_eq!(GtrGroupId::wrapped(GtrGroupId::MAX).get(), 0);
    }

    /// e191 — a re-stickify-with-PT op drops PT rows 1..=7 on both corelets, from its OWN pcfg list
    /// and from the super-DSC's pool and map, ⛔ while ROW 0 SURVIVES on either corelet: neither
    /// `PTROW0_0` nor `PTROW0_1` is in `pcfgUnitToRemove` (`dcg_manager.cpp:571-576`).
    #[test]
    fn e191_removes_pt_rows_one_through_seven_and_keeps_row_zero() {
        let filed = [
            SenComponent::Ptrow0_0,
            SenComponent::Ptrow3_0,
            SenComponent::Ptrow0_1,
            SenComponent::Ptrow7_1,
        ];
        let mut sdsc = one_dsc2_core::<true>().with_data_op(DataOpDsc {
            op: OpFunc::ReStickifyOpWithPtLx,
            pcfg: BTreeMap::from([(
                core::<0>(),
                filed
                    .iter()
                    .map(|comp| (SenPcfg::default(), *comp))
                    .collect(),
            )]),
        });
        for (slot, comp) in filed.iter().enumerate() {
            let id = PcfgId(slot as u64);
            sdsc.pcfg_map
                .entry(core::<0>())
                .or_default()
                .insert(*comp, id);
            sdsc.pcfg_pool.insert(id, SenPcfg::default());
        }

        DcgManager::<DccSenProg, true, true>::new().remove_extra_pt_rows(&mut sdsc);

        let kept: Vec<SenComponent> = sdsc.data_op_dscs[0].pcfg[&core::<0>()]
            .iter()
            .map(|(_, comp)| *comp)
            .collect();
        assert_eq!(kept, vec![SenComponent::Ptrow0_0, SenComponent::Ptrow0_1]);
        let mapped: Vec<SenComponent> = sdsc.pcfg_map[&core::<0>()].keys().copied().collect();
        assert_eq!(mapped, vec![SenComponent::Ptrow0_0, SenComponent::Ptrow0_1]);
        assert_eq!(sdsc.pcfg_pool.len(), 2);
    }

    /// e191 — the version-1 arm needs MORE THAN ONE data op before it touches `pcfg_` (`:611`), so
    /// one re-stickify op leaves that map alone even though it has already pruned its own list.
    #[test]
    fn e191_one_data_op_leaves_the_version_one_pcfg_map_alone() {
        let mut sdsc = one_dsc2_core::<true>().with_data_op(DataOpDsc {
            op: OpFunc::ReStickifyOpWithPthbm,
            pcfg: BTreeMap::from([(
                core::<0>(),
                vec![(SenPcfg::default(), SenComponent::Ptrow3_0)],
            )]),
        });
        sdsc.pcfg.insert(
            core::<0>(),
            BTreeMap::from([(SenComponent::Ptrow3_0, SenPcfg::default())]),
        );

        DcgManager::<DccSenProg, false, true>::new().remove_extra_pt_rows(&mut sdsc);

        assert!(sdsc.data_op_dscs[0].pcfg[&core::<0>()].is_empty());
        assert_eq!(sdsc.pcfg[&core::<0>()].len(), 1);
    }

    /// A model, so the programs the sink is handed carry the arch/model/rung the ISA tables were
    /// chosen for — the same three [`progir::Program`] one rung up is bound by.
    struct Granite;
    impl Model for Granite {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    /// A decode rung: one row.
    struct Decode;
    impl Workload for Decode {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// e192 — the SEN-program form picks which converter runs, and both live in `sys-arch-spec/dpc/`,
    /// out of scope (`dcg_manager.cpp:967-973`).
    #[test]
    #[should_panic(expected = "Dpc::convertIr2Senprog")]
    fn e192_stops_at_the_deep_program_converter() {
        let mut out = String::new();
        let programs: BTreeMap<Core, progir::Program<Target, Granite, Decode>> = BTreeMap::new();
        StageThreeDcg::new().print_sen_program(
            &programs,
            &mut Dpc,
            &mut out,
            SenProgramForm::SenProg,
        );
    }

    /// e193 — the generic KG3 path is the backend's, and the flag it is handed is `createSenProg`,
    /// which [`StageThreeDcg`] fixes to false (`dcg_manager.h:94-96`).
    #[test]
    #[should_panic(expected = "DcgBE::runDcgForGenKG3")]
    fn e193_stops_at_the_backend() {
        let mut sdsc = one_dsc2_core::<false>();
        StageThreeDcg::new().run_dcg_for_gen_kg3(&mut sdsc);
    }

    /// e194 — likewise for the sparse-KG3 convolution path (`dcg_manager.h:98-100`).
    #[test]
    #[should_panic(expected = "DcgBE::runDcgForSparseKG3CONV2D")]
    fn e194_stops_at_the_backend() {
        let mut sdsc = one_dsc2_core::<false>();
        StageThreeDcg::new().run_dcg_for_sparse_kg3_conv2d(&mut sdsc);
    }

    /// e195 — likewise for the sparse-KG3 batched-matmul path (`dcg_manager.h:102-104`).
    #[test]
    #[should_panic(expected = "DcgBE::runDcgForSparseKG3BMM")]
    fn e195_stops_at_the_backend() {
        let mut sdsc = one_dsc2_core::<false>();
        StageThreeDcg::new().run_dcg_for_sparse_kg3_bmm(&mut sdsc);
    }

    /// e280 — with `createSenProg` false the whole codegen half is skipped and what remains is the
    /// PT-row sweep the entry ends with UNCONDITIONALLY (`dcg_manager.cpp:213`), so row 3 goes and
    /// row 0 stays.
    #[test]
    fn e280_without_codegen_only_the_pt_row_sweep_runs() {
        let mut sdsc = one_dsc2_core::<true>().with_data_op(DataOpDsc {
            op: OpFunc::ReStickifyOpWithPtLx,
            pcfg: BTreeMap::from([(
                core::<0>(),
                vec![
                    (SenPcfg::default(), SenComponent::Ptrow0_0),
                    (SenPcfg::default(), SenComponent::Ptrow3_0),
                ],
            )]),
        });

        let ran = DcgManager::<NoSenProgDcg, true, true>::new().run_dcg_generate_prog_ir(&mut sdsc);

        assert_eq!(ran, Some(()));
        let kept: Vec<SenComponent> = sdsc.data_op_dscs[0].pcfg[&core::<0>()]
            .iter()
            .map(|(_, comp)| *comp)
            .collect();
        assert_eq!(kept, vec![SenComponent::Ptrow0_0]);
    }

    /// e280 — ⛔⛔ THE FOLD GUARD, PINNED AT MONOMORPHIZATION. A folded super-DSC with `createSenProg`
    /// off never reaches the reference's `DT_CHECK_MSG` (`:155-157`) and still runs the PT-row sweep,
    /// so this call has to COMPILE: a `const { assert!(!FOLDED) }` would refuse it — the block is
    /// evaluated whether or not the branch around it can run — and that is our own configuration
    /// (`SchedulerStages.cpp:49-50`). Instantiating `FOLDED = true` here is the whole test.
    #[test]
    fn e280_a_folded_super_dsc_without_codegen_still_sweeps_pt_rows() {
        let mut sdsc: SuperDsc<true, true> =
            SuperDsc::of(PerCoreDscs::of(core::<0>(), DscVersion::Dsc2)).with_data_op(DataOpDsc {
                op: OpFunc::ReStickifyOpWithPtLx,
                pcfg: BTreeMap::from([(
                    core::<0>(),
                    vec![
                        (SenPcfg::default(), SenComponent::Ptrow0_0),
                        (SenPcfg::default(), SenComponent::Ptrow3_0),
                    ],
                )]),
            });

        let ran = DcgManager::<NoSenProgDcg, true, true>::new().run_dcg_generate_prog_ir(&mut sdsc);

        assert_eq!(ran, Some(()));
        let kept: Vec<SenComponent> = sdsc.data_op_dscs[0].pcfg[&core::<0>()]
            .iter()
            .map(|(_, comp)| *comp)
            .collect();
        assert_eq!(kept, vec![SenComponent::Ptrow0_0]);
    }

    /// e281 — the same fold guard on the other entry (`:526-528`): folded and codegen-off, this must
    /// reach the pcfg generator rather than be refused before it, so the `should_panic` naming that
    /// seam is what proves the guard is the conjunction and not `!FOLDED`.
    #[test]
    #[should_panic(expected = "DcgFE::generatePcfgIRForDataOpInpFetch")]
    fn e281_a_folded_super_dsc_without_codegen_reaches_the_generator() {
        let mut main: SuperDsc<true, true> =
            SuperDsc::of(PerCoreDscs::of(core::<0>(), DscVersion::Dsc2));
        let _ = DcgManager::<NoSenProgDcg, true, true>::new()
            .run_dcg_for_input_fetch_neighbor(&mut main, None);
    }

    /// e281 — its FIRST act is the input-fetch pcfg generator, so the entry names that translator and
    /// stops before it can flip the mode (`dcg_manager.cpp:522`).
    #[test]
    #[should_panic(expected = "DcgFE::generatePcfgIRForDataOpInpFetch")]
    fn e281_stops_at_the_input_fetch_pcfg_generator() {
        let mut main = one_dsc2_core::<true>();
        let _ = StageThreeDcg::new().run_dcg_for_input_fetch_neighbor(&mut main, None);
    }

    /// e282 — a data step whose child pcfgs carry no graph still DEFAULT-CREATES `pcfg_[core][comp]`
    /// (`:687-690`), and the version-2 tail then re-pools it: ⚠️ the new id is
    /// `pcfgPool_.rbegin()->first + 1` (`:864-865`), so slot 5's occupant is re-filed as 6 and slot 5,
    /// which nothing names any more, is erased (`:889-894`).
    #[test]
    fn e282_rebuilds_the_pcfg_map_and_erases_the_slot_it_left() {
        let mut sdsc = one_dsc2_core::<true>()
            .with_data_op(DataOpDsc {
                op: OpFunc::StcdpOpLx,
                pcfg: BTreeMap::from([(
                    core::<0>(),
                    vec![(SenPcfg::default(), SenComponent::Lxlu0)],
                )]),
            })
            .with_core_schedule(
                core::<0>(),
                CoreSchedule::data_only(vec![DataStep {
                    data_dsc: Some(DataOpIndex::new(0)),
                    before_sync: false,
                    after_sync: false,
                }]),
            );
        sdsc.pcfg_map.insert(
            core::<0>(),
            BTreeMap::from([(SenComponent::Lxlu0, PcfgId(5))]),
        );
        sdsc.pcfg_pool.insert(PcfgId(5), SenPcfg::default());

        let merged = DcgManager::<NoSenProgDcg, true, true>::new()
            .merge_pcfg_in_super_dsc(&mut sdsc, DlPcfgSource::Dsc);

        assert_eq!(merged, Some(()));
        let units: Vec<SenComponent> = sdsc.pcfg[&core::<0>()].keys().copied().collect();
        assert_eq!(units, vec![SenComponent::Lxlu0]);
        assert_eq!(sdsc.pcfg_map[&core::<0>()][&SenComponent::Lxlu0], PcfgId(6));
        let pooled: Vec<PcfgId> = sdsc.pcfg_pool.keys().copied().collect();
        assert_eq!(pooled, vec![PcfgId(6)]);
    }

    /// e282 — ⛔ THE POOL TAIL IS SKIPPED UNDER DCC (`:863`), which is OUR configuration
    /// (`SchedulerStages.cpp:49-50`): `pcfg_` is still rebuilt but slot 5 keeps both its occupant and
    /// its name.
    #[test]
    fn e282_under_dcc_the_pool_is_left_exactly_as_it_was() {
        let mut sdsc = one_dsc2_core::<true>()
            .with_data_op(DataOpDsc {
                op: OpFunc::StcdpOpLx,
                pcfg: BTreeMap::from([(
                    core::<0>(),
                    vec![(SenPcfg::default(), SenComponent::Lxlu0)],
                )]),
            })
            .with_core_schedule(
                core::<0>(),
                CoreSchedule::data_only(vec![DataStep {
                    data_dsc: Some(DataOpIndex::new(0)),
                    before_sync: false,
                    after_sync: false,
                }]),
            );
        sdsc.pcfg_map.insert(
            core::<0>(),
            BTreeMap::from([(SenComponent::Lxlu0, PcfgId(5))]),
        );
        sdsc.pcfg_pool.insert(PcfgId(5), SenPcfg::default());

        let merged = StageThreeDcg::new().merge_pcfg_in_super_dsc(&mut sdsc, DlPcfgSource::Dsc);

        assert_eq!(merged, Some(()));
        assert!(sdsc.pcfg[&core::<0>()].contains_key(&SenComponent::Lxlu0));
        assert_eq!(sdsc.pcfg_map[&core::<0>()][&SenComponent::Lxlu0], PcfgId(5));
        let pooled: Vec<PcfgId> = sdsc.pcfg_pool.keys().copied().collect();
        assert_eq!(pooled, vec![PcfgId(5)]);
    }
}
