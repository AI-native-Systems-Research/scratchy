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

use crate::arch::{Arch, Bytes, Elements, Sticks, Target};
use crate::units::Core;

// crustify:todo: e188_runDcgComputeTransfer
//   authority : dcg/dcg_manager/dcg_manager.cpp:112  (13 body lines, level 0)
//   class     : DcgManager
//   original  : void DcgManager::runDcgComputeTransfer(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:32-45

// crustify:todo: e189_runDcgForDlOps
//   authority : dcg/dcg_manager/dcg_manager.cpp:216  (52 body lines, level 0)
//   class     : DcgManager
//   original  : void DcgManager::runDcgForDlOps(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:55-107

// crustify:todo: e190_runDcgForDlOpsStandalone
//   authority : dcg/dcg_manager/dcg_manager.cpp:449  (64 body lines, level 0)
//   class     : DcgManager
//   original  : void DcgManager::runDcgForDlOpsStandalone(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:117-181

// crustify:todo: e191_removeextraPTrows
//   authority : dcg/dcg_manager/dcg_manager.cpp:568  (60 body lines, level 0)
//   class     : DcgManager
//   original  : void DcgManager::removeextraPTrows(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:191-251

// crustify:todo: e192_printSenProgram
//   authority : dcg/dcg_manager/dcg_manager.cpp:959  (16 body lines, level 0)
//   class     : DcgManager
//   original  : void DcgManager::printSenProgram(SuperDsc* mySDsc, Dpc& myDpc, std::string fileName, bool use_smc /*= false*/)
//   extract   : crustify-ddc/cpp/dcg.cpp:261-279

// crustify:todo: e193_runDcgForGenKG3
//   authority : dcg/dcg_manager/dcg_manager.h:94  (3 body lines, level 0)
//   class     : DcgManager
//   original  : void runDcgForGenKG3(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:289-292

// crustify:todo: e194_runDcgForSparseKG3CONV2D
//   authority : dcg/dcg_manager/dcg_manager.h:98  (3 body lines, level 0)
//   class     : DcgManager
//   original  : void runDcgForSparseKG3CONV2D(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:302-305

// crustify:todo: e195_runDcgForSparseKG3BMM
//   authority : dcg/dcg_manager/dcg_manager.h:102  (3 body lines, level 0)
//   class     : DcgManager
//   original  : void runDcgForSparseKG3BMM(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/dcg.cpp:315-318

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

/// HOW FAR ROUND THE RING A TRANSFER REACHES — one end of `CCWHopCWHop` (`dsc/dataOpDsc.h:216`).
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

    /// The dump itself (`:5949-5965`). `std::setw` right-aligns, and the reference's fourth line is
    /// the `setw`-only `<<` chain at `:5954`, which emits nothing but its newline.
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
/// (`:5889,5904`) where every sibling wraps modularly — its own second copy of the identical walk
/// writes `% (int)maxNumCores` (`inputNeighFetchOp.cpp:2294,2308,2318`) and the CW arm three lines
/// below wraps by hand — so `/ 32` charges every CCW hop landing below core 32 to core 0.
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
}
