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
use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

use crate::arch::{Arch, Bytes, Elements, Sticks, Target};
use crate::islands::progir;
use crate::model::Model;
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

/// A PCFG — `SenPcfg` (`dsc/pcfg.h`), built by `dcg/dcg_fe/pcfg_gen/`.
///
/// ⛔ OPAQUE ON PURPOSE. No unit in this campaign reads a pcfg's nodes, and inventing them is
/// forbidden (`crustify-ddc/TASK.md`: *"do not invent a PCFG"*), so this carries none. Its two
/// operations are the seam below.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SenPcfg;

impl SenPcfg {
    /// `SenPcfg::mergeSenPcfg(parent, child)` (`dsc/pcfg.h:1056`).
    ///
    /// ⭐ STATIC, THOUGH IT LOOKS LIKE A METHOD: entry 189 calls it as
    /// `sdscPcfg.mergeSenPcfg(sdscPcfg, *localPcfg)` (`:256`), so the receiver and the `parent`
    /// argument are one object and the child is spliced into it.
    pub fn merge(parent: &mut Self, child: &mut Self) {
        let _ = (parent, child);
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

/// WHETHER A DSC HAS A SCHEDULE TREE — `isDSC2()`, which is `!scheduleTree_.empty()`
/// (`dsc/designSpaceConfig.cpp:31`). Stage 2a is what puts one there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DscVersion {
    /// `isDSC2() == false`.
    Dsc1,
    /// `isDSC2() == true`.
    Dsc2,
}

/// THE DL DSCs, KEYED BY THE CORE EACH SERVES, AND NEVER EMPTY.
///
/// ⭐ ONE TABLE FOR TWO FIELDS: `coreIdToDsc_` is a map of `DesignSpaceConfig*` INTO `dscs_`
/// (`dsc/superdsc.h:67-68`), and the only thing stage 3 reads through those pointers is `isDSC2()`.
/// ⭐ NON-EMPTY BY CONSTRUCTION — `DT_CHECK(mySDsc.dscs_.size() >= 1)` (`:220`, `:453`) needs no
/// runtime form when the type admits no empty value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerCoreDscs(BTreeMap<Core, DscVersion>);

impl PerCoreDscs {
    /// One DL DSC, on one core.
    #[must_use]
    pub fn of(core: Core, version: DscVersion) -> Self {
        Self(BTreeMap::from([(core, version)]))
    }

    /// Another core's DL DSC.
    #[must_use]
    pub fn and(mut self, core: Core, version: DscVersion) -> Self {
        self.0.insert(core, version);
        self
    }

    /// `for (auto& kv : mySDsc.coreIdToDsc_)` — ascending core, as `std::map` iterates.
    pub fn iter(&self) -> impl Iterator<Item = (Core, DscVersion)> + '_ {
        self.0.iter().map(|(core, version)| (*core, *version))
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
    pub pcfg: Vec<Vec<(SenPcfg, SenComponent)>>,
}

/// THE SUPER-DSC AS STAGE 3 SEES IT — `SuperDsc` (`dsc/superdsc.h:48`), restricted to the pcfg state
/// this stage mutates. The schedule tree, the allocations and the placed addresses belong to stages
/// 2a and 2b, whose own homes own those fields.
///
/// ⛔ TWO `DT_CHECK`s BECOME TWO TYPE-LEVEL FACTS:
/// `DATA_OPS` — whether `dataOpdscs_` may hold anything. `DT_CHECK(mySDsc.dataOpdscs_.size() == 0)`
/// (`:222`) is entry 189 taking `SuperDsc<false, _>`, and entry 188 takes `SuperDsc<true, _>`.
/// `FOLDED` — whether `sdscFoldProps_` is non-empty. `DT_CHECK_MSG(mySDsc.sdscFoldProps_.empty(),
/// "Codegen for Folded Super-DSC is not supported")` (`:262`, `:506`) is a `const { assert!(!FOLDED) }`
/// in the codegen tails, which is where the reference checks it — so the props themselves stay with
/// `util/foldManager/`, out of scope, where they are read.
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
        }
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
}

/// `createSenProg == false` — no codegen tail on either entry. ⭐ Our configuration.
pub struct NoSenProg;

/// `createSenProg == true` with `progIRcodeGen == DCGProgIRGen::DCG` — entry 189's tail.
pub struct DcgSenProg;

/// `createSenProg == true` with `progIRcodeGen == DCGProgIRGen::DCC` — entry 190's tail.
pub struct DccSenProg;

impl SenProgGen for NoSenProg {
    const CREATE_SEN_PROG: bool = false;
}

impl SenProgGen for DcgSenProg {
    const CREATE_SEN_PROG: bool = true;
}

impl SenProgGen for DccSenProg {
    const CREATE_SEN_PROG: bool = true;
}

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
        let _ = sdsc;
        todo!("dcg_be/: DcgBE::fillAndCreateSenProgInfoUsingSuperDSC is out of scope")
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
            codegen: PhantomData,
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
                for per_core in &mut data_op.pcfg {
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

// crustify:todo: e280_runDcgGenerateProgIR
//   authority : dcg/dcg_manager/dcg_manager.cpp:145  (70 body lines, level 1)
//   class     : DcgManager
//   original  : void DcgManager::runDcgGenerateProgIR(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:341-411
//   calls     : e104_clear, e187_clear, e191_removeextraPTrows

// crustify:todo: e281_runDcgForInputFetchNeighbor
//   authority : dcg/dcg_manager/dcg_manager.cpp:514  (52 body lines, level 1)
//   class     : DcgManager
//   original  : void DcgManager::runDcgForInputFetchNeighbor(SuperDsc& mySDscMain, SuperDsc* mySDscPre)
//   extract   : crustify-ddc/cpp/dcg.cpp:421-474
//   calls     : e104_clear, e187_clear

// crustify:todo: e282_mergePcfgInSuperDSC
//   authority : dcg/dcg_manager/dcg_manager.cpp:635  (259 body lines, level 1)
//   class     : DcgManager
//   original  : void DcgManager::mergePcfgInSuperDSC(SuperDsc& mySDsc, std::vector<SenPcfg>& pcfgL3lu, std::vector<SenPcfg>& pcfgL3su, bool useActualDLpcfg /*= false*/)
//   extract   : crustify-ddc/cpp/dcg.cpp:484-746
//   calls     : e104_clear, e187_clear

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
            pcfg: vec![filed.iter().map(|comp| (SenPcfg, *comp)).collect()],
        });
        for (slot, comp) in filed.iter().enumerate() {
            let id = PcfgId(slot as u64);
            sdsc.pcfg_map
                .entry(core::<0>())
                .or_default()
                .insert(*comp, id);
            sdsc.pcfg_pool.insert(id, SenPcfg);
        }

        DcgManager::<DccSenProg, true, true>::new().remove_extra_pt_rows(&mut sdsc);

        let kept: Vec<SenComponent> = sdsc.data_op_dscs[0].pcfg[0]
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
            pcfg: vec![vec![(SenPcfg, SenComponent::Ptrow3_0)]],
        });
        sdsc.pcfg.insert(
            core::<0>(),
            BTreeMap::from([(SenComponent::Ptrow3_0, SenPcfg)]),
        );

        DcgManager::<DccSenProg, false, true>::new().remove_extra_pt_rows(&mut sdsc);

        assert!(sdsc.data_op_dscs[0].pcfg[0].is_empty());
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
}
