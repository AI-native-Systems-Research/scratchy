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

// crustify:todo: e196_printTrafficPerCore
//   authority : dcg/dcg_manager/dcg_manager.h:120  (3 body lines, level 0)
//   class     : DcgManager
//   original  : void printTrafficPerCore(SuperDsc& sdsc, std::string fileName)
//   extract   : crustify-ddc/cpp/dcg.cpp:328-331

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

