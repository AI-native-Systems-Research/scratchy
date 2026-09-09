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

//! `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`, `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h` — 144 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e001_isSameDscGroup` | 001 | 0 | 4 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:60` |
//! | `e002_isLabeledDsDimensionBroadcast` | 002 | 0 | 6 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:65` |
//! | `e003_isDimensionCoreletSplit` | 003 | 0 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:74` |
//! | `e004_voidPaddingIfChunking` | 004 | 0 | 20 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:128` |
//! | `e005_addOrUpdateSymbolicInfoInParams` | 005 | 0 | 15 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:158` |
//! | `e006_getLabeledDsWkSliceMulticastDegree` | 006 | 0 | 39 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:176` |
//! | `e007_scheduleDimTypeToString` | 007 | 0 | 16 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:282` |
//! | `e008_hasDimensionReuse` | 008 | 0 | 19 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:303` |
//! | `e009_getStickSize` | 009 | 0 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:327` |
//! | `e010_getCoreSplitDimensions` | 010 | 0 | 25 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:342` |
//! | `e011_getLabeledDsWithDsType` | 011 | 0 | 6 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:369` |
//! | `e012_getAllLabeledDsIndicesSet` | 012 | 0 | 7 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:378` |
//! | `e013_getHbmPinnedLabeledDsIndicesSet` | 013 | 0 | 7 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:387` |
//! | `e014_isLabeledDsLXNeighbor` | 014 | 0 | 26 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:406` |
//! | `e015_getParentLoopNodes` | 015 | 0 | 10 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:532` |
//! | `e016_createAllocateNode` | 016 | 0 | 49 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:544` |
//! | `e017_createTransferNode` | 017 | 0 | 22 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:596` |
//! | `e018_createLoopNode` | 018 | 0 | 19 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:623` |
//! | `e019_createBlockNode` | 019 | 0 | 6 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:645` |
//! | `e020_createSyncNode` | 020 | 0 | 10 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:652` |
//! | `e021_getOpFuncName` | 021 | 0 | 4 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:665` |
//! | `e022_addOrUpdateDataStageParam` | 022 | 0 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:721` |
//! | `e023_isOpFuncConv2dInt4` | 023 | 0 | 6 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:739` |
//! | `e024_isOpFuncConv2dOs1` | 024 | 0 | 6 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:746` |
//! | `e025_isOpFuncBmmInt4` | 025 | 0 | 6 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:769` |
//! | `e026_isOpFuncBmmInt8` | 026 | 0 | 7 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:776` |
//! | `e027_isOpFuncBmmFp8NonXrf` | 027 | 0 | 6 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:784` |
//! | `e028_isOpFuncBmmFp8Xrf` | 028 | 0 | 5 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:791` |
//! | `e029_isOpFuncBmmFp16` | 029 | 0 | 6 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:797` |
//! | `e030_isOpFuncScalarBroadcast` | 030 | 0 | 18 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:810` |
//! | `e031_isOpFuncReduction` | 031 | 0 | 7 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:829` |
//! | `e032_isOpFuncPooling` | 032 | 0 | 5 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:837` |
//! | `e033_isOpFuncDepthwiseConv` | 033 | 0 | 5 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:843` |
//! | `e034_isOpFuncQuantization` | 034 | 0 | 8 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:849` |
//! | `e035_isOpFuncConversionDl16AndFp32` | 035 | 0 | 5 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:858` |
//! | `e036_getMinParamScalarBroadcast` | 036 | 0 | 16 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1008` |
//! | `e037_getMinParamReduction` | 037 | 0 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1026` |
//! | `e038_getMinParamPoolingAndDepthwiseConv` | 038 | 0 | 22 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1040` |
//! | `e039_getMinParamQuantization` | 039 | 0 | 31 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1065` |
//! | `e040_getMinParamConversionDl16AndFp32` | 040 | 0 | 17 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1099` |
//! | `e041_getChunkParamsFromCandidates` | 041 | 0 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1423` |
//! | `e042_getBurstEfficiency` | 042 | 0 | 17 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1609` |
//! | `e043_getLabeledDsNumOfStickVolumesInCore` | 043 | 0 | 37 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1694` |
//! | `e044_getOpReducedDimSet` | 044 | 0 | 13 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2719` |
//! | `e045_addSuperChunkDataStage` | 045 | 0 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2806` |
//! | `e046_getLxBelowBlockNode` | 046 | 0 | 11 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3468` |
//! | `e047_collectAllDimensionsForLoopOrder` | 047 | 0 | 25 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3991` |
//! | `e048_getSharesAndGroupName` | 048 | 0 | 43 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4673` |
//! | `e049_calculateCoreletOffsetInByte` | 049 | 0 | 82 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4842` |
//! | `e050_getInitialStartAddressAndOffset` | 050 | 0 | 31 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4926` |
//! | `e051_getLdsOrConstNameOfAllocNode` | 051 | 0 | 10 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5493` |
//! | `e052_verifyLoopOrder` | 052 | 0 | 17 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6366` |
//! | `e053_verifyScheduleTree` | 053 | 0 | 22 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6385` |
//! | `e054_prepDsc` | 054 | 0 | 13 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6411` |
//! | `e055_computeMinHMICoreGroupSizeForSEN1P5` | 055 | 0 | 94 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6486` |
//! | `e056_isIndexLds` | 056 | 0 | 13 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6582` |
//! | `e057_isPagedLds` | 057 | 0 | 9 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6596` |
//! | `e058_getNewDataStageIndex` | 058 | 0 | 18 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6606` |
//! | `e059_getPagedDimensions` | 059 | 0 | 20 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6709` |
//! | `e060_getHbmAllocations` | 060 | 0 | 13 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7152` |
//! | `e061_gatherFoldParams` | 061 | 0 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7248` |
//! | `e062_getEnclosingLoopsAndRelatedDims` | 062 | 0 | 38 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7263` |
//! | `e063_findAndStoreLoopWithDim` | 063 | 0 | 21 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7307` |
//! | `e064_constructDatastage` | 064 | 0 | 11 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7724` |
//! | `e065_constructLoopNode` | 065 | 0 | 18 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7737` |
//! | `e066_addCore` | 066 | 0 | 4 | `CrossCoreReductionGroup` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:29` |
//! | `e067_getStartCoreAtCorelet` | 067 | 0 | 9 | `CrossCoreReductionGroup` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:34` |
//! | `e068_getEndCoreAtCorelet` | 068 | 0 | 9 | `CrossCoreReductionGroup` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:43` |
//! | `e069_updateMin` | 069 | 0 | 3 | `Constraints` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:117` |
//! | `e070_updateMax` | 070 | 0 | 3 | `Constraints` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:120` |
//! | `e071_updateValues` | 071 | 0 | 3 | `Constraints` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:123` |
//! | `e072_getTripCount` | 072 | 0 | 9 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:487` |
//! | `e197_getCoreletSplitDimensions` | 197 | 1 | 15 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:88` |
//! | `e198_addOrUpdatePaddingSizesInChunkParams` | 198 | 1 | 6 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:150` |
//! | `e199_getLabeledDsNumOfWkSlices` | 199 | 1 | 49 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:219` |
//! | `e200_getLxNeighborLabeledDsIndicesSet` | 200 | 1 | 7 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:396` |
//! | `e201_computeLdsAllocateSiblingLoopNode` | 201 | 1 | 55 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:435` |
//! | `e202_computeLdsTransferSiblingLoopNode` | 202 | 1 | 32 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:495` |
//! | `e203_getOpFuncDataFormat` | 203 | 1 | 49 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:670` |
//! | `e204_isOpFuncConv2d` | 204 | 1 | 15 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:753` |
//! | `e205_isOpFuncBmm` | 205 | 1 | 5 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:804` |
//! | `e206_getMinParamBmm` | 206 | 1 | 80 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:925` |
//! | `e207_generateDscParamCandidates` | 207 | 1 | 204 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1172` |
//! | `e208_getLdsL3TransferNodes` | 208 | 1 | 34 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1571` |
//! | `e209_getLabeledDsChunkStickVolume` | 209 | 1 | 64 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1628` |
//! | `e210_isOpCrossCoreReduction` | 210 | 1 | 7 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2734` |
//! | `e211_getInsertionNode` | 211 | 1 | 89 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3376` |
//! | `e212_addL3LUAndLXLUSyncNodeSequence` | 212 | 1 | 45 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3912` |
//! | `e213_addL3LUAndLXLUSoftSyncNodeSequence` | 213 | 1 | 26 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3959` |
//! | `e214_optimizeHbmLdsOutputInScheduleTree` | 214 | 1 | 94 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4018` |
//! | `e215_buildScheduleDimensionsTable` | 215 | 1 | 103 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4236` |
//! | `e216_buildLoopOrder` | 216 | 1 | 231 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4377` |
//! | `e217_createChunkLoopNodes` | 217 | 1 | 58 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4612` |
//! | `e218_setCondGtr` | 218 | 1 | 114 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4721` |
//! | `e219_fillFinalStartAddressAndOffset` | 219 | 1 | 137 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4959` |
//! | `e220_fillIBRStartAddressAndOffset` | 220 | 1 | 43 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5100` |
//! | `e221_fillTransferZeroPaddingInfo` | 221 | 1 | 196 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5296` |
//! | `e222_allocAllMem` | 222 | 1 | 236 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5508` |
//! | `e223_getHbmLdsTransferHMIRequestEstimate` | 223 | 1 | 30 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6452` |
//! | `e224_getAllPagedLdsIndices` | 224 | 1 | 8 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6731` |
//! | `e225_createPagedDimChunkLoops` | 225 | 1 | 62 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6804` |
//! | `e226_createStoreIndexTensorToIbr` | 226 | 1 | 62 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7037` |
//! | `e227_convertTransferDirectToIndirect` | 227 | 1 | 45 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7103` |
//! | `e228_buildCoordinateFromAllocation` | 228 | 1 | 181 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7333` |
//! | `e229_sliceCoordinateForCorelet` | 229 | 1 | 203 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7518` |
//! | `e283_addOrUpdateCoreletSplitInParams` | 283 | 2 | 20 | — | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:105` |
//! | `e284_isOpFuncStridedWindow` | 284 | 2 | 4 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:865` |
//! | `e285_calculateBurstEfficiency` | 285 | 2 | 267 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1738` |
//! | `e286_calculateFlopPerByte` | 286 | 2 | 230 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2253` |
//! | `e287_getCrossCoreReductionGroupInfo` | 287 | 2 | 27 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2743` |
//! | `e288_createSynchronizationDSC` | 288 | 2 | 423 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3487` |
//! | `e289_optimizeHbmTransfers` | 289 | 2 | 76 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4113` |
//! | `e290_createChunkLoops` | 290 | 2 | 38 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4190` |
//! | `e291_fillTransferMulticastInfo` | 291 | 2 | 127 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5146` |
//! | `e292_fillAllocationStartAddrAndOffset` | 292 | 2 | 20 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5274` |
//! | `e293_setLxBufferType` | 293 | 2 | 26 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6425` |
//! | `e294_createStoreIndexTensorToLx` | 294 | 2 | 85 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6948` |
//! | `e295_fillExplicitTransferSize` | 295 | 2 | 35 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7875` |
//! | `e328_computeMinParamForPaddedDim` | 328 | 3 | 24 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:870` |
//! | `e329_addChunkDataStageFromCandidates` | 329 | 3 | 13 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1405` |
//! | `e330_getLdsTransferCoreIds` | 330 | 3 | 17 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2773` |
//! | `e331_exploreSuperChunkDataStageParams` | 331 | 3 | 140 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2829` |
//! | `e332_createSynchronization` | 332 | 3 | 5 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3481` |
//! | `e333_fillLoopOffsetsAndAddresses` | 333 | 3 | 613 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5750` |
//! | `e334_addIbrDataStage` | 334 | 3 | 47 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6626` |
//! | `e335_addOnePageDataStage` | 335 | 3 | 30 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6676` |
//! | `e336_processPagedTensorTransfers` | 336 | 3 | 71 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6870` |
//! | `e350_getMinParamConv2d` | 350 | 4 | 26 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:896` |
//! | `e351_setSuperChunkDataStageParams` | 351 | 4 | 12 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2793` |
//! | `e352_updateChunkDataStagesFromCandidates` | 352 | 4 | 5 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2819` |
//! | `e353_createAllocationAndTransfer` | 353 | 4 | 254 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3121` |
//! | `e354_processDscHbmPagedTensors` | 354 | 4 | 56 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6746` |
//! | `e355_fillCoordinateCustomWkSliceId` | 355 | 4 | 47 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7198` |
//! | `e365_getMinParamForDimFromOpFunc` | 365 | 5 | 33 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1137` |
//! | `e366_findBestParamsForMemoryBandwidth` | 366 | 5 | 228 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2018` |
//! | `e367_findBestParamsForArithmeticIntensity` | 367 | 5 | 214 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2500` |
//! | `e368_processHbmPagedTensors` | 368 | 5 | 4 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6741` |
//! | `e369_buildCoordinateForAllocation` | 369 | 5 | 28 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7167` |
//! | `e373_getMinParamForDim` | 373 | 6 | 15 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1119` |
//! | `e374_propagateCoordinateDSC` | 374 | 6 | 109 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7761` |
//! | `e377_getInitialChunkParams` | 377 | 7 | 20 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1382` |
//! | `e378_propagateCoordinate` | 378 | 7 | 3 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7757` |
//! | `e380_setChunkDataStageParams` | 380 | 8 | 129 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1439` |
//! | `e382_run` | 382 | 9 | 122 | `L3DlOpsScheduler` | `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7912` |


// crustify:todo: e001_isSameDscGroup
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:60  (4 body lines, level 0)
//   original  : [[maybe_unused]] static bool isSameDscGroup(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:31-35

// crustify:todo: e002_isLabeledDsDimensionBroadcast
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:65  (6 body lines, level 0)
//   original  : static inline bool isLabeledDsDimensionBroadcast(const DesignSpaceConfig &dsc, const LabeledDsInfo &lds, PrimaryDimTypes dim)
//   extract   : crustify-ddc/cpp/l3.cpp:44-52

// crustify:todo: e003_isDimensionCoreletSplit
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:74  (12 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isDimensionCoreletSplit(const DesignSpaceConfig &dsc, PrimaryDimTypes dim)
//   extract   : crustify-ddc/cpp/l3.cpp:62-75

// crustify:todo: e004_voidPaddingIfChunking
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:128  (20 body lines, level 0)
//   original  : static void voidPaddingIfChunking(DataStructDims &ds, const DataStructDims &refDs)
//   extract   : crustify-ddc/cpp/l3.cpp:84-105

// crustify:todo: e005_addOrUpdateSymbolicInfoInParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:158  (15 body lines, level 0)
//   original  : static void addOrUpdateSymbolicInfoInParams(DataStructDims &chunkParams, const DataStructDims &coreParams)
//   extract   : crustify-ddc/cpp/l3.cpp:114-130

// crustify:todo: e006_getLabeledDsWkSliceMulticastDegree
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:176  (39 body lines, level 0)
//   original  : static unsigned getLabeledDsWkSliceMulticastDegree( const SuperDsc &mySDsc, const int ldsIdx, const std::vector<int> &dscIndices)
//   extract   : crustify-ddc/cpp/l3.cpp:139-180

// crustify:todo: e007_scheduleDimTypeToString
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:282  (16 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::string L3DlOpsScheduler::scheduleDimTypeToString(ScheduleDimTypes type)
//   extract   : crustify-ddc/cpp/l3.cpp:190-206

// crustify:todo: e008_hasDimensionReuse
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:303  (19 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::hasDimensionReuse(const DesignSpaceConfig& dsc)
//   extract   : crustify-ddc/cpp/l3.cpp:216-235

// crustify:todo: e009_getStickSize
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:327  (12 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : int L3DlOpsScheduler::getStickSize(const DesignSpaceConfig &dsc, DsTypes dsType, PrimaryDimTypes dim)
//   extract   : crustify-ddc/cpp/l3.cpp:245-258

// crustify:todo: e010_getCoreSplitDimensions
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:342  (25 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::unordered_set<PrimaryDimTypes> L3DlOpsScheduler::getCoreSplitDimensions( const SuperDsc &mySDsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:268-294

// crustify:todo: e011_getLabeledDsWithDsType
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:369  (6 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::getLabeledDsWithDsType(std::vector<int> &indices, DesignSpaceConfig &dsc, DsTypes dsType)
//   extract   : crustify-ddc/cpp/l3.cpp:304-312

// crustify:todo: e012_getAllLabeledDsIndicesSet
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:378  (7 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::unordered_set<int> L3DlOpsScheduler::getAllLabeledDsIndicesSet( const DesignSpaceConfig& dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:322-330

// crustify:todo: e013_getHbmPinnedLabeledDsIndicesSet
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:387  (7 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::unordered_set<int> L3DlOpsScheduler::getHbmPinnedLabeledDsIndicesSet( const DesignSpaceConfig& dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:340-348

// crustify:todo: e014_isLabeledDsLXNeighbor
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:406  (26 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isLabeledDsLXNeighbor(const SuperDsc &mySDsc, const int dscIndex, const LabeledDsInfo &lds) const
//   extract   : crustify-ddc/cpp/l3.cpp:358-386

// crustify:todo: e015_getParentLoopNodes
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:532  (10 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::vector<const dsc2::LoopNode *> L3DlOpsScheduler::getParentLoopNodes( const dsc2::ScheduleNode &node, const DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:396-407

// crustify:todo: e016_createAllocateNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:544  (49 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : dsc2::AllocateNode *L3DlOpsScheduler::createAllocateNode( DesignSpaceConfig &dsc, const int ldsIdx, enum SenComponents component, const int numBuffers, const std::string &name, const int dscIdx)
//   extract   : crustify-ddc/cpp/l3.cpp:417-468

// crustify:todo: e017_createTransferNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:596  (22 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : dsc2::TransferNode *L3DlOpsScheduler::createTransferNode( const SenComponents srcUnit, const SenComponents srcStorage, const std::vector<SenComponents> &dstUnits, const std::vector<SenComponents> &dstStorage, const int srcLdsIndex, const std::vector<int> &dstLdsIndices, const std::string &name)
//   extract   : crustify-ddc/cpp/l3.cpp:478-504

// crustify:todo: e018_createLoopNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:623  (19 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : dsc2::LoopNode *L3DlOpsScheduler::createLoopNode( const DesignSpaceConfig &dsc, const std::vector<PrimaryDimTypes> &dims, const int numeratorId, const int denominatorId, const std::string &name)
//   extract   : crustify-ddc/cpp/l3.cpp:514-535

// crustify:todo: e019_createBlockNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:645  (6 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : dsc2::BlockNode *L3DlOpsScheduler::createBlockNode(const std::string &name)
//   extract   : crustify-ddc/cpp/l3.cpp:545-551

// crustify:todo: e020_createSyncNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:652  (10 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : dsc2::SyncNode* L3DlOpsScheduler::createSyncNode( const std::unordered_set<SenComponents>& units, const std::string& name, const bool isReceive, const bool isSoft) const
//   extract   : crustify-ddc/cpp/l3.cpp:561-573

// crustify:todo: e021_getOpFuncName
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:665  (4 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : OpFuncs L3DlOpsScheduler::getOpFuncName(const DesignSpaceConfig& dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:583-587

// crustify:todo: e022_addOrUpdateDataStageParam
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:721  (12 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::addOrUpdateDataStageParam(DesignSpaceConfig &dsc, const DataStructDims &ssParam, const std::string &ssName, const DataStructDims &elParam, const std::string &elName, const int index)
//   extract   : crustify-ddc/cpp/l3.cpp:597-614

// crustify:todo: e023_isOpFuncConv2dInt4
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:739  (6 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncConv2dInt4(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:624-630

// crustify:todo: e024_isOpFuncConv2dOs1
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:746  (6 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncConv2dOs1(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:640-646

// crustify:todo: e025_isOpFuncBmmInt4
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:769  (6 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncBmmInt4(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:656-662

// crustify:todo: e026_isOpFuncBmmInt8
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:776  (7 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncBmmInt8(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:672-679

// crustify:todo: e027_isOpFuncBmmFp8NonXrf
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:784  (6 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncBmmFp8NonXrf(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:689-695

// crustify:todo: e028_isOpFuncBmmFp8Xrf
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:791  (5 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncBmmFp8Xrf(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:705-710

// crustify:todo: e029_isOpFuncBmmFp16
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:797  (6 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncBmmFp16(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:720-726

// crustify:todo: e030_isOpFuncScalarBroadcast
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:810  (18 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncScalarBroadcast(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:736-754

// crustify:todo: e031_isOpFuncReduction
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:829  (7 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncReduction(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:764-771

// crustify:todo: e032_isOpFuncPooling
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:837  (5 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncPooling(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:781-786

// crustify:todo: e033_isOpFuncDepthwiseConv
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:843  (5 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncDepthwiseConv(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:796-801

// crustify:todo: e034_isOpFuncQuantization
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:849  (8 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncQuantization(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:811-819

// crustify:todo: e035_isOpFuncConversionDl16AndFp32
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:858  (5 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncConversionDl16AndFp32( const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:829-835

// crustify:todo: e036_getMinParamScalarBroadcast
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1008  (16 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamScalarBroadcast( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
//   extract   : crustify-ddc/cpp/l3.cpp:845-862

// crustify:todo: e037_getMinParamReduction
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1026  (12 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamReduction(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
//   extract   : crustify-ddc/cpp/l3.cpp:872-885

// crustify:todo: e038_getMinParamPoolingAndDepthwiseConv
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1040  (22 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamPoolingAndDepthwiseConv( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:895-919

// crustify:todo: e039_getMinParamQuantization
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1065  (31 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamQuantization(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:929-962

// crustify:todo: e040_getMinParamConversionDl16AndFp32
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1099  (17 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamConversionDl16AndFp32( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:972-991

// crustify:todo: e041_getChunkParamsFromCandidates
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1423  (12 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::getChunkParamsFromCandidates( DataStructDims &params, const DscParamCandidateIndicesType &selectedIndices, const DscParamCandidatesType &dscCandidates, const int dscIdx, const std::vector<PrimaryDimTypes> &primaryDims)
//   extract   : crustify-ddc/cpp/l3.cpp:1001-1016

// crustify:todo: e042_getBurstEfficiency
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1609  (17 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : double L3DlOpsScheduler::getBurstEfficiency(const unsigned burstSize, const unsigned multicastDegree)
//   extract   : crustify-ddc/cpp/l3.cpp:1026-1044

// crustify:todo: e043_getLabeledDsNumOfStickVolumesInCore
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1694  (37 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : unsigned long L3DlOpsScheduler::getLabeledDsNumOfStickVolumesInCore( const DesignSpaceConfig &dsc, const int ldsIdx, const unsigned long stickVolume, const std::vector<PrimaryDimTypes> &primaryDims)
//   extract   : crustify-ddc/cpp/l3.cpp:1054-1094

// crustify:todo: e044_getOpReducedDimSet
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2719  (13 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::set<PrimaryDimTypes> L3DlOpsScheduler::getOpReducedDimSet( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:1104-1118

// crustify:todo: e045_addSuperChunkDataStage
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2806  (12 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::addSuperChunkDataStage(DesignSpaceConfig& dsc)
//   extract   : crustify-ddc/cpp/l3.cpp:1128-1140

// crustify:todo: e046_getLxBelowBlockNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3468  (11 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : dsc2::BlockNode *L3DlOpsScheduler::getLxBelowBlockNode( dsc2::ScheduleTree &scheduleTree) const
//   extract   : crustify-ddc/cpp/l3.cpp:1150-1162

// crustify:todo: e047_collectAllDimensionsForLoopOrder
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3991  (25 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::vector<PrimaryDimTypes> L3DlOpsScheduler::collectAllDimensionsForLoopOrder( const DesignSpaceConfig &dsc)
//   extract   : crustify-ddc/cpp/l3.cpp:1172-1198

// crustify:todo: e048_getSharesAndGroupName
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4673  (43 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::pair<size_t, size_t> L3DlOpsScheduler::getSharesAndGroupName( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc, const LabeledDsInfo &lds, const std::map<PrimaryDimTypes, int> &currWkSlices, const std::vector<int> &processingCoreIds)
//   extract   : crustify-ddc/cpp/l3.cpp:1208-1255

// crustify:todo: e049_calculateCoreletOffsetInByte
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4842  (82 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::vector<int64_t> L3DlOpsScheduler::calculateCoreletOffsetInByte( const DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode) const
//   extract   : crustify-ddc/cpp/l3.cpp:1265-1348

// crustify:todo: e050_getInitialStartAddressAndOffset
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4926  (31 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::pair<int64_t, int64_t> L3DlOpsScheduler::getInitialStartAddressAndOffset( DesignSpaceConfig &dsc, const int ldsIdx, std::deque<int64_t> coord) const
//   extract   : crustify-ddc/cpp/l3.cpp:1358-1390

// crustify:todo: e051_getLdsOrConstNameOfAllocNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5493  (10 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::string L3DlOpsScheduler::getLdsOrConstNameOfAllocNode( DesignSpaceConfig *currDsc, dsc2::AllocateNode *anode)
//   extract   : crustify-ddc/cpp/l3.cpp:1400-1411

// crustify:todo: e052_verifyLoopOrder
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6366  (17 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::verifyLoopOrder( std::vector<PrimaryDimTypes> &loopOrder)
//   extract   : crustify-ddc/cpp/l3.cpp:1421-1439

// crustify:todo: e053_verifyScheduleTree
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6385  (22 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::verifyScheduleTree(const DesignSpaceConfig &dsc)
//   extract   : crustify-ddc/cpp/l3.cpp:1449-1471

// crustify:todo: e054_prepDsc
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6411  (13 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::prepDsc(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:1481-1494

// crustify:todo: e055_computeMinHMICoreGroupSizeForSEN1P5
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6486  (94 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : int L3DlOpsScheduler::computeMinHMICoreGroupSizeForSEN1P5( const SuperDsc &mySDsc, const std::vector<int> &hbmLdsIndices) const
//   extract   : crustify-ddc/cpp/l3.cpp:1504-1599

// crustify:todo: e056_isIndexLds
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6582  (13 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isIndexLds(const LabeledDsInfo &lds) const
//   extract   : crustify-ddc/cpp/l3.cpp:1609-1622

// crustify:todo: e057_isPagedLds
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6596  (9 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isPagedLds(const LabeledDsInfo &lds) const
//   extract   : crustify-ddc/cpp/l3.cpp:1632-1641

// crustify:todo: e058_getNewDataStageIndex
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6606  (18 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : int L3DlOpsScheduler::getNewDataStageIndex(SuperDsc& mySDsc, DesignSpaceConfig& dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:1651-1670

// crustify:todo: e059_getPagedDimensions
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6709  (20 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::vector<PrimaryDimTypes> L3DlOpsScheduler::getPagedDimensions( const DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:1680-1701

// crustify:todo: e060_getHbmAllocations
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7152  (13 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : std::vector<dsc2::AllocateNode *> L3DlOpsScheduler::getHbmAllocations( const DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:1711-1725

// crustify:todo: e061_gatherFoldParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7248  (12 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::gatherFoldParams( const FoldManager<CoordinateBaseType> &cfm, std::vector<dsc2::FoldParamInfoType> &foldParams) const
//   extract   : crustify-ddc/cpp/l3.cpp:1735-1749

// crustify:todo: e062_getEnclosingLoopsAndRelatedDims
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7263  (38 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::getEnclosingLoopsAndRelatedDims( dsc2::ScheduleNode *node, const DesignSpaceConfig *dsc, std::vector<dsc2::LoopNode *> &loopChain, std::unordered_set<PrimaryDimAndKind> &relatedDims) const
//   extract   : crustify-ddc/cpp/l3.cpp:1759-1800

// crustify:todo: e063_findAndStoreLoopWithDim
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7307  (21 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::findAndStoreLoopWithDim( DesignSpaceConfig *currDsc, const PrimaryDimAndKind dimToFind, dsc2::LoopNode *loop, const std::unordered_set<PrimaryDimAndKind> &relatedDims, dsc2::VectorOfLoopAndDim &relatedLoops, PadType accessPadType) const
//   extract   : crustify-ddc/cpp/l3.cpp:1810-1835

// crustify:todo: e064_constructDatastage
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7724  (11 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : int L3DlOpsScheduler::constructDatastage(DesignSpaceConfig *currDsc, dsc2::DataStage &refDataStage) const
//   extract   : crustify-ddc/cpp/l3.cpp:1845-1857

// crustify:todo: e065_constructLoopNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7737  (18 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : dsc2::LoopNode *L3DlOpsScheduler::constructLoopNode( int numId, int denId, std::vector<PrimaryDimAndKind> dims) const
//   extract   : crustify-ddc/cpp/l3.cpp:1867-1886

// crustify:todo: e066_addCore
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:29  (4 body lines, level 0)
//   class     : CrossCoreReductionGroup
//   original  : void addCore(const int coreId, const int slice)
//   extract   : crustify-ddc/cpp/l3.cpp:1896-1900

// crustify:todo: e067_getStartCoreAtCorelet
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:34  (9 body lines, level 0)
//   class     : CrossCoreReductionGroup
//   original  : int getStartCoreAtCorelet(int coreletId) const
//   extract   : crustify-ddc/cpp/l3.cpp:1910-1919

// crustify:todo: e068_getEndCoreAtCorelet
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:43  (9 body lines, level 0)
//   class     : CrossCoreReductionGroup
//   original  : int getEndCoreAtCorelet(int coreletId) const
//   extract   : crustify-ddc/cpp/l3.cpp:1929-1938

// crustify:todo: e069_updateMin
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:117  (3 body lines, level 0)
//   class     : Constraints
//   original  : inline void updateMin(float newVal)
//   extract   : crustify-ddc/cpp/l3.cpp:1948-1951

// crustify:todo: e070_updateMax
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:120  (3 body lines, level 0)
//   class     : Constraints
//   original  : inline void updateMax(float newVal)
//   extract   : crustify-ddc/cpp/l3.cpp:1961-1964

// crustify:todo: e071_updateValues
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:123  (3 body lines, level 0)
//   class     : Constraints
//   original  : inline void updateValues(std::set<float> newVals)
//   extract   : crustify-ddc/cpp/l3.cpp:1974-1977

// crustify:todo: e072_getTripCount
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:487  (9 body lines, level 0)
//   class     : L3DlOpsScheduler
//   original  : int getTripCount(const DesignSpaceConfig &dsc, const PrimaryDimTypes dim, const int dataStageNumId, const int dataStageDenId) const
//   extract   : crustify-ddc/cpp/l3.cpp:1987-1997

// crustify:todo: e197_getCoreletSplitDimensions
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:88  (15 body lines, level 1)
//   original  : static std::vector<PrimaryDimTypes> getCoreletSplitDimensions( const DesignSpaceConfig &dsc)
//   extract   : crustify-ddc/cpp/l3.cpp:2006-2022
//   calls     : e003_isDimensionCoreletSplit

// crustify:todo: e198_addOrUpdatePaddingSizesInChunkParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:150  (6 body lines, level 1)
//   original  : static void addOrUpdatePaddingSizesInChunkParams( DataStructDims &chunkParams, const DataStructDims &coreParams)
//   extract   : crustify-ddc/cpp/l3.cpp:2031-2038
//   calls     : e004_voidPaddingIfChunking

// crustify:todo: e199_getLabeledDsNumOfWkSlices
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:219  (49 body lines, level 1)
//   original  : static unsigned getLabeledDsNumOfWkSlices(const SuperDsc &mySDsc, const int ldsIdx, const std::vector<int> &dscIndices)
//   extract   : crustify-ddc/cpp/l3.cpp:2047-2098
//   calls     : e002_isLabeledDsDimensionBroadcast

// crustify:todo: e200_getLxNeighborLabeledDsIndicesSet
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:396  (7 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : std::unordered_set<int> L3DlOpsScheduler::getLxNeighborLabeledDsIndicesSet( const SuperDsc& mySDsc, const DesignSpaceConfig& dsc, const int dscIdx) const
//   extract   : crustify-ddc/cpp/l3.cpp:2108-2117
//   calls     : e014_isLabeledDsLXNeighbor

// crustify:todo: e201_computeLdsAllocateSiblingLoopNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:435  (55 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : dsc2::BlockNode *L3DlOpsScheduler::computeLdsAllocateSiblingLoopNode( const SuperDsc &mySDsc, const int dscIdx, const LabeledDsInfo &lds, const dsc2::BlockNode *startNode, const std::vector<const dsc2::LoopNode *> &parentInnerToOuterLoopNodes) const
//   extract   : crustify-ddc/cpp/l3.cpp:2127-2186
//   calls     : e014_isLabeledDsLXNeighbor, e056_isIndexLds

// crustify:todo: e202_computeLdsTransferSiblingLoopNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:495  (32 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : dsc2::BlockNode *L3DlOpsScheduler::computeLdsTransferSiblingLoopNode( const SuperDsc &mySDsc, const int dscIdx, const LabeledDsInfo &lds, const dsc2::BlockNode *startNode, const std::vector<const dsc2::LoopNode *> &parentInnerToOuterLoopNodes) const
//   extract   : crustify-ddc/cpp/l3.cpp:2196-2232
//   calls     : e014_isLabeledDsLXNeighbor, e056_isIndexLds

// crustify:todo: e203_getOpFuncDataFormat
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:670  (49 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : std::string L3DlOpsScheduler::getOpFuncDataFormat( const DesignSpaceConfig &dsc)
//   extract   : crustify-ddc/cpp/l3.cpp:2242-2292
//   calls     : e021_getOpFuncName

// crustify:todo: e204_isOpFuncConv2d
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:753  (15 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncConv2d(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:2302-2317
//   calls     : e023_isOpFuncConv2dInt4, e024_isOpFuncConv2dOs1

// crustify:todo: e205_isOpFuncBmm
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:804  (5 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncBmm(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:2327-2332
//   calls     : e025_isOpFuncBmmInt4, e026_isOpFuncBmmInt8, e027_isOpFuncBmmFp8NonXrf, e028_isOpFuncBmmFp8Xrf, e029_isOpFuncBmmFp16

// crustify:todo: e206_getMinParamBmm
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:925  (80 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamBmm(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:2342-2424
//   calls     : e025_isOpFuncBmmInt4, e026_isOpFuncBmmInt8, e027_isOpFuncBmmFp8NonXrf, e028_isOpFuncBmmFp8Xrf, e029_isOpFuncBmmFp16

// crustify:todo: e207_generateDscParamCandidates
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1172  (204 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : auto L3DlOpsScheduler::generateDscParamCandidates( const SuperDsc &mySDsc, const std::vector<DataStructDims> &dscParams, const std::vector<PrimaryDimTypes> &primaryDims, const std::unordered_set<PrimaryDimTypes> &chunkDims, const std::unordered_set<PrimaryDimTypes> &coreSplitDims) -> DscParamCandida
//   extract   : crustify-ddc/cpp/l3.cpp:2434-2643
//   calls     : e003_isDimensionCoreletSplit, e009_getStickSize, e056_isIndexLds, e059_getPagedDimensions

// crustify:todo: e208_getLdsL3TransferNodes
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1571  (34 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : std::vector<const dsc2::TransferNode *> L3DlOpsScheduler::getLdsL3TransferNodes( const SuperDsc &mySDsc, const int dscIdx, const int ldsIdx, const std::vector<SenComponents> &srcStorages, const std::vector<SenComponents> &dstStorages) const
//   extract   : crustify-ddc/cpp/l3.cpp:2653-2690
//   calls     : e014_isLabeledDsLXNeighbor

// crustify:todo: e209_getLabeledDsChunkStickVolume
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1628  (64 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : unsigned long L3DlOpsScheduler::getLabeledDsChunkStickVolume( const DesignSpaceConfig &dsc, const int ldsIdx)
//   extract   : crustify-ddc/cpp/l3.cpp:2700-2765
//   calls     : e009_getStickSize, e056_isIndexLds

// crustify:todo: e210_isOpCrossCoreReduction
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2734  (7 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpCrossCoreReduction( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:2775-2783
//   calls     : e044_getOpReducedDimSet

// crustify:todo: e211_getInsertionNode
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3376  (89 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : dsc2::ScheduleNode* L3DlOpsScheduler::getInsertionNode( const std::unordered_set<dsc2::ScheduleNode*>& refNodeSet, const bool insertBefore) const
//   extract   : crustify-ddc/cpp/l3.cpp:2793-2884
//   calls     : e104_clear, e187_clear

// crustify:todo: e212_addL3LUAndLXLUSyncNodeSequence
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3912  (45 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::addL3LUAndLXLUSyncNodeSequence( const dsc2::ScheduleNode *insertAfterNode) const
//   extract   : crustify-ddc/cpp/l3.cpp:2894-2940
//   calls     : e020_createSyncNode

// crustify:todo: e213_addL3LUAndLXLUSoftSyncNodeSequence
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3959  (26 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::addL3LUAndLXLUSoftSyncNodeSequence( const dsc2::ScheduleNode* insertAfterNode) const
//   extract   : crustify-ddc/cpp/l3.cpp:2950-2977
//   calls     : e020_createSyncNode

// crustify:todo: e214_optimizeHbmLdsOutputInScheduleTree
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4018  (94 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::optimizeHbmLdsOutputInScheduleTree(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:2987-3081
//   calls     : e015_getParentLoopNodes, e072_getTripCount

// crustify:todo: e215_buildScheduleDimensionsTable
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4236  (103 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : auto L3DlOpsScheduler::buildScheduleDimensionsTable( SuperDsc &mySDsc, const int dscIdx, std::vector<PrimaryDimTypes> &dims, const bool isReuse) -> ScheduleDimTableType
//   extract   : crustify-ddc/cpp/l3.cpp:3091-3196
//   calls     : e007_scheduleDimTypeToString, e056_isIndexLds

// crustify:todo: e216_buildLoopOrder
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4377  (231 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : std::vector<PrimaryDimTypes> L3DlOpsScheduler::buildLoopOrder( SuperDsc &mySDsc, const int dscIdx, const std::vector<PrimaryDimTypes> &dims, const ScheduleDimTableType &schedDimTypesTable, const bool isReuse)
//   extract   : crustify-ddc/cpp/l3.cpp:3206-3440
//   calls     : e011_getLabeledDsWithDsType, e014_isLabeledDsLXNeighbor, e052_verifyLoopOrder, e056_isIndexLds, e059_getPagedDimensions, e138_swap

// crustify:todo: e217_createChunkLoopNodes
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4612  (58 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::createChunkLoopNodes( SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &loopOrderInnerToOuter)
//   extract   : crustify-ddc/cpp/l3.cpp:3450-3510
//   calls     : e018_createLoopNode, e019_createBlockNode, e058_getNewDataStageIndex

// crustify:todo: e218_setCondGtr
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4721  (114 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::setCondGtr( SuperDsc &mySDsc, const int dscIdx, const int ldsIdx, const int currCoreId, dsc2::TransferNode &ldsL3LUTransNode, const std::vector<std::tuple<const dsc2::LoopNode *, PrimaryDimTypes, int, int>> &loopDimTripCounts)
//   extract   : crustify-ddc/cpp/l3.cpp:3520-3638
//   calls     : e010_getCoreSplitDimensions, e017_createTransferNode, e048_getSharesAndGroupName

// crustify:todo: e219_fillFinalStartAddressAndOffset
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4959  (137 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillFinalStartAddressAndOffset( DesignSpaceConfig &dsc, const int ldsIdx, const std::vector<PrimaryDimTypes> &coreletSplitDims) const
//   extract   : crustify-ddc/cpp/l3.cpp:3648-3787
//   calls     : e050_getInitialStartAddressAndOffset

// crustify:todo: e220_fillIBRStartAddressAndOffset
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5100  (43 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillIBRStartAddressAndOffset(const SuperDsc &sdsc, DesignSpaceConfig &dsc, const int ldsIdx) const
//   extract   : crustify-ddc/cpp/l3.cpp:3797-3842
//   calls     : e056_isIndexLds

// crustify:todo: e221_fillTransferZeroPaddingInfo
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5296  (196 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillTransferZeroPaddingInfo(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:3852-4048
//   calls     : e059_getPagedDimensions

// crustify:todo: e222_allocAllMem
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5508  (236 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::allocAllMem(const SuperDsc &mySDsc, DesignSpaceConfig *currDsc, const int dscIdx, bool commitIfValid)
//   extract   : crustify-ddc/cpp/l3.cpp:4061-4299
//   calls     : e051_getLdsOrConstNameOfAllocNode
//   ⛔ NOTE   : The L3 scheduler's own allocation commit (distinct from Ddc::allocAllMem at
//               ddc/ddcv1.cpp:132 -- two different functions with the same name; the unit number
//               disambiguates).

// crustify:todo: e223_getHbmLdsTransferHMIRequestEstimate
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6452  (30 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : int L3DlOpsScheduler::getHbmLdsTransferHMIRequestEstimate( const SuperDsc &mySDsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:4309-4340
//   calls     : e055_computeMinHMICoreGroupSizeForSEN1P5

// crustify:todo: e224_getAllPagedLdsIndices
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6731  (8 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : std::vector<int> L3DlOpsScheduler::getAllPagedLdsIndices( const DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:4350-4359
//   calls     : e057_isPagedLds

// crustify:todo: e225_createPagedDimChunkLoops
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6804  (62 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *> L3DlOpsScheduler::createPagedDimChunkLoops( DesignSpaceConfig &dsc, const std::vector<PrimaryDimTypes> &pagedDims, std::set<dsc2::LoopNode *> &chunkLoopNodes)
//   extract   : crustify-ddc/cpp/l3.cpp:4369-4434
//   calls     : e018_createLoopNode

// crustify:todo: e226_createStoreIndexTensorToIbr
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7037  (62 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::createStoreIndexTensorToIbr( DesignSpaceConfig &dsc, const int dscIdx, const int indexLdsIdx, dsc2::AllocateNode &indexLdsHbmAllocNode, dsc2::LoopNode &newChunkLoopNode, const bool isTransferIn)
//   extract   : crustify-ddc/cpp/l3.cpp:4444-4509
//   calls     : e016_createAllocateNode, e017_createTransferNode, e020_createSyncNode

// crustify:todo: e227_convertTransferDirectToIndirect
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7103  (45 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::convertTransferDirectToIndirect( dsc2::TransferNode &transNode, DesignSpaceConfig &dsc, const int indexLdsIdx, const PrimaryDimTypes indexStickDim, const bool isTransferIn)
//   extract   : crustify-ddc/cpp/l3.cpp:4519-4567
//   calls     : e018_createLoopNode

// crustify:todo: e228_buildCoordinateFromAllocation
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7333  (181 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::buildCoordinateFromAllocation( DesignSpaceConfig &dsc, dsc2::AllocateNode *refAllocNode, dsc2::ScheduleNode *node, dsc2::CoordinateType<CoordinateBaseType> &coordinate) const
//   extract   : crustify-ddc/cpp/l3.cpp:4577-4761
//   calls     : e061_gatherFoldParams, e062_getEnclosingLoopsAndRelatedDims, e063_findAndStoreLoopWithDim

// crustify:todo: e229_sliceCoordinateForCorelet
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7518  (203 body lines, level 1)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::sliceCoordinateForCorelet( SuperDsc &mySDsc, DesignSpaceConfig *currDsc, dsc2::AllocateNode *allocNode) const
//   extract   : crustify-ddc/cpp/l3.cpp:4771-4976
//   calls     : e046_getLxBelowBlockNode, e061_gatherFoldParams, e064_constructDatastage, e065_constructLoopNode

// crustify:todo: e283_addOrUpdateCoreletSplitInParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:105  (20 body lines, level 2)
//   original  : static void addOrUpdateCoreletSplitInParams(DataStructDims &params, const DesignSpaceConfig &dsc)
//   extract   : crustify-ddc/cpp/l3.cpp:4985-5006
//   calls     : e104_clear, e187_clear, e197_getCoreletSplitDimensions

// crustify:todo: e284_isOpFuncStridedWindow
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:865  (4 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : bool L3DlOpsScheduler::isOpFuncStridedWindow(const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:5016-5020
//   calls     : e032_isOpFuncPooling, e033_isOpFuncDepthwiseConv, e204_isOpFuncConv2d

// crustify:todo: e285_calculateBurstEfficiency
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1738  (267 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : double L3DlOpsScheduler::calculateBurstEfficiency( const SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims)
//   extract   : crustify-ddc/cpp/l3.cpp:5030-5298
//   calls     : e006_getLabeledDsWkSliceMulticastDegree, e010_getCoreSplitDimensions, e014_isLabeledDsLXNeighbor, e015_getParentLoopNodes, e042_getBurstEfficiency, e043_getLabeledDsNumOfStickVolumesInCore, e056_isIndexLds, e072_getTripCount, e199_getLabeledDsNumOfWkSlices, e208_getLdsL3TransferNodes, e209_getLabeledDsChunkStickVolume

// crustify:todo: e286_calculateFlopPerByte
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2253  (230 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : double L3DlOpsScheduler::calculateFlopPerByte( const SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims)
//   extract   : crustify-ddc/cpp/l3.cpp:5308-5539
//   calls     : e010_getCoreSplitDimensions, e014_isLabeledDsLXNeighbor, e015_getParentLoopNodes, e072_getTripCount, e199_getLabeledDsNumOfWkSlices, e208_getLdsL3TransferNodes

// crustify:todo: e287_getCrossCoreReductionGroupInfo
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2743  (27 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : std::vector<CrossCoreReductionGroup> L3DlOpsScheduler::getCrossCoreReductionGroupInfo( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:5549-5578
//   calls     : e044_getOpReducedDimSet, e066_addCore, e210_isOpCrossCoreReduction

// crustify:todo: e288_createSynchronizationDSC
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3487  (423 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::createSynchronizationDSC(SuperDsc& mySDsc, const int dscIdx)
//   extract   : crustify-ddc/cpp/l3.cpp:5588-6012
//   calls     : e012_getAllLabeledDsIndicesSet, e013_getHbmPinnedLabeledDsIndicesSet, e020_createSyncNode, e046_getLxBelowBlockNode, e200_getLxNeighborLabeledDsIndicesSet, e211_getInsertionNode, e212_addL3LUAndLXLUSyncNodeSequence, e213_addL3LUAndLXLUSoftSyncNodeSequence

// crustify:todo: e289_optimizeHbmTransfers
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4113  (76 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::optimizeHbmTransfers(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:6022-6098
//   calls     : e210_isOpCrossCoreReduction

// crustify:todo: e290_createChunkLoops
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4190  (38 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::createChunkLoops(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:6108-6146
//   calls     : e008_hasDimensionReuse, e047_collectAllDimensionsForLoopOrder, e215_buildScheduleDimensionsTable, e216_buildLoopOrder, e217_createChunkLoopNodes

// crustify:todo: e291_fillTransferMulticastInfo
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5146  (127 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillTransferMulticastInfo(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:6156-6283
//   calls     : e015_getParentLoopNodes, e048_getSharesAndGroupName, e072_getTripCount, e218_setCondGtr

// crustify:todo: e292_fillAllocationStartAddrAndOffset
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5274  (20 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillAllocationStartAddrAndOffset( SuperDsc &mySDsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:6297-6318
//   calls     : e056_isIndexLds, e197_getCoreletSplitDimensions, e219_fillFinalStartAddressAndOffset, e220_fillIBRStartAddressAndOffset
//   ⛔ NOTE   : THIS IS THE `start_address = 0` DEFECT. L3DlOpsScheduler::run calls this as "Set start
//               address, offset in allocations" (L3DlOpsScheduler.cpp:8000). Our emitted views print
//               start_address = 0 where the reference states a placed base. The effect of this function
//               IS the port.

// crustify:todo: e293_setLxBufferType
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6425  (26 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::setLxBufferType(const SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:6328-6354
//   calls     : e223_getHbmLdsTransferHMIRequestEstimate

// crustify:todo: e294_createStoreIndexTensorToLx
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6948  (85 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::createStoreIndexTensorToLx( SuperDsc &mySDsc, const int dscIdx, const int pagedLdsIdx, const int indexLdsIdx, dsc2::AllocateNode &indexLdsHbmAllocNode, dsc2::LoopNode &newChunkLoopNode)
//   extract   : crustify-ddc/cpp/l3.cpp:6364-6452
//   calls     : e016_createAllocateNode, e017_createTransferNode, e020_createSyncNode, e222_allocAllMem

// crustify:todo: e295_fillExplicitTransferSize
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7875  (35 body lines, level 2)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillExplicitTransferSize(SuperDsc &mySDsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:6462-6497
//   calls     : e210_isOpCrossCoreReduction

// crustify:todo: e328_computeMinParamForPaddedDim
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:870  (24 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::computeMinParamForPaddedDim( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
//   extract   : crustify-ddc/cpp/l3.cpp:6507-6532
//   calls     : e021_getOpFuncName, e284_isOpFuncStridedWindow

// crustify:todo: e329_addChunkDataStageFromCandidates
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1405  (13 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::addChunkDataStageFromCandidates( DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx, const DscParamCandidateIndicesType &selectedIndices, const DscParamCandidatesType &dscCandidates, const std::vector<PrimaryDimTypes> &primaryDims)
//   extract   : crustify-ddc/cpp/l3.cpp:6542-6559
//   calls     : e005_addOrUpdateSymbolicInfoInParams, e022_addOrUpdateDataStageParam, e041_getChunkParamsFromCandidates, e198_addOrUpdatePaddingSizesInChunkParams, e283_addOrUpdateCoreletSplitInParams

// crustify:todo: e330_getLdsTransferCoreIds
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2773  (17 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : std::vector<int> L3DlOpsScheduler::getLdsTransferCoreIds( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc, const LabeledDsInfo &lds) const
//   extract   : crustify-ddc/cpp/l3.cpp:6569-6588
//   calls     : e068_getEndCoreAtCorelet, e210_isOpCrossCoreReduction, e287_getCrossCoreReductionGroupInfo

// crustify:todo: e331_exploreSuperChunkDataStageParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2829  (140 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::exploreSuperChunkDataStageParams(SuperDsc& mySDsc, const int dscIdx)
//   extract   : crustify-ddc/cpp/l3.cpp:6598-6739
//   calls     : e046_getLxBelowBlockNode, e059_getPagedDimensions, e222_allocAllMem, e283_addOrUpdateCoreletSplitInParams

// crustify:todo: e332_createSynchronization
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3481  (5 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::createSynchronization(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:6749-6754
//   calls     : e288_createSynchronizationDSC

// crustify:todo: e333_fillLoopOffsetsAndAddresses
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5750  (613 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillLoopOffsetsAndAddresses( SuperDsc &mySDsc, const int dscIdx, const bool allowUnpaddedIndexingAtPaddedNoZeroPad)
//   extract   : crustify-ddc/cpp/l3.cpp:6764-7379
//   calls     : e049_calculateCoreletOffsetInByte, e056_isIndexLds, e057_isPagedLds, e068_getEndCoreAtCorelet, e076_print, e102_print, e104_clear, e187_clear, e210_isOpCrossCoreReduction, e272_print, e287_getCrossCoreReductionGroupInfo

// crustify:todo: e334_addIbrDataStage
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6626  (47 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::addIbrDataStage( SuperDsc& mySDsc, DesignSpaceConfig& dsc, const std::vector<PrimaryDimTypes>& pagedDims)
//   extract   : crustify-ddc/cpp/l3.cpp:7389-7438
//   calls     : e005_addOrUpdateSymbolicInfoInParams, e056_isIndexLds, e058_getNewDataStageIndex, e283_addOrUpdateCoreletSplitInParams

// crustify:todo: e335_addOnePageDataStage
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6676  (30 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::addOnePageDataStage( SuperDsc& mySDsc, DesignSpaceConfig& dsc, const std::vector<PrimaryDimTypes>& pagedDims)
//   extract   : crustify-ddc/cpp/l3.cpp:7448-7480
//   calls     : e058_getNewDataStageIndex, e224_getAllPagedLdsIndices, e283_addOrUpdateCoreletSplitInParams

// crustify:todo: e336_processPagedTensorTransfers
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6870  (71 body lines, level 3)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::processPagedTensorTransfers( SuperDsc &mySDsc, const int dscIdx, const std::vector<dsc2::TransferNode *> &l3TransferNodesAllTensors, const std::vector<dsc2::AllocateNode *> &pagedLdsHbmAllocNodes, const std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *> &newPagedDimChunkLoo
//   extract   : crustify-ddc/cpp/l3.cpp:7490-7567
//   calls     : e226_createStoreIndexTensorToIbr, e227_convertTransferDirectToIndirect, e294_createStoreIndexTensorToLx

// crustify:todo: e350_getMinParamConv2d
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:896  (26 body lines, level 4)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamConv2d(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
//   extract   : crustify-ddc/cpp/l3.cpp:7577-7605
//   calls     : e023_isOpFuncConv2dInt4, e024_isOpFuncConv2dOs1, e328_computeMinParamForPaddedDim

// crustify:todo: e351_setSuperChunkDataStageParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2793  (12 body lines, level 4)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::setSuperChunkDataStageParams(SuperDsc& mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:7615-7627
//   calls     : e045_addSuperChunkDataStage, e283_addOrUpdateCoreletSplitInParams, e331_exploreSuperChunkDataStageParams

// crustify:todo: e352_updateChunkDataStagesFromCandidates
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2819  (5 body lines, level 4)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::updateChunkDataStagesFromCandidates( DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx, const DscParamCandidateIndicesType &selectedIndices, const DscParamCandidatesType &dscCandidates, const std::vector<PrimaryDimTypes> &primaryDims)
//   extract   : crustify-ddc/cpp/l3.cpp:7637-7646
//   calls     : e045_addSuperChunkDataStage, e329_addChunkDataStageFromCandidates

// crustify:todo: e353_createAllocationAndTransfer
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3121  (254 body lines, level 4)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::createAllocationAndTransfer(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:7656-7910
//   calls     : e014_isLabeledDsLXNeighbor, e015_getParentLoopNodes, e016_createAllocateNode, e017_createTransferNode, e046_getLxBelowBlockNode, e056_isIndexLds, e201_computeLdsAllocateSiblingLoopNode, e202_computeLdsTransferSiblingLoopNode, e210_isOpCrossCoreReduction, e330_getLdsTransferCoreIds

// crustify:todo: e354_processDscHbmPagedTensors
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6746  (56 body lines, level 4)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::processDscHbmPagedTensors(SuperDsc &mySDsc, const int dscIdx)
//   extract   : crustify-ddc/cpp/l3.cpp:7920-7977
//   calls     : e059_getPagedDimensions, e225_createPagedDimChunkLoops, e336_processPagedTensorTransfers

// crustify:todo: e355_fillCoordinateCustomWkSliceId
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7198  (47 body lines, level 4)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::fillCoordinateCustomWkSliceId( SuperDsc &mySDsc, DesignSpaceConfig &dsc, const int ldsIdx, dsc2::CoordinateType<CoordinateBaseType> &coordinate) const
//   extract   : crustify-ddc/cpp/l3.cpp:7987-8036
//   calls     : e330_getLdsTransferCoreIds

// crustify:todo: e365_getMinParamForDimFromOpFunc
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1137  (33 body lines, level 5)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamForDimFromOpFunc(const DesignSpaceConfig& dsc, PrimaryDimTypes dim) const
//   extract   : crustify-ddc/cpp/l3.cpp:8046-8080
//   calls     : e021_getOpFuncName, e030_isOpFuncScalarBroadcast, e031_isOpFuncReduction, e032_isOpFuncPooling, e033_isOpFuncDepthwiseConv, e034_isOpFuncQuantization, e035_isOpFuncConversionDl16AndFp32, e036_getMinParamScalarBroadcast, e037_getMinParamReduction, e038_getMinParamPoolingAndDepthwiseConv, e039_getMinParamQuantization, e040_getMinParamConversionDl16AndFp32, e204_isOpFuncConv2d, e205_isOpFuncBmm …

// crustify:todo: e366_findBestParamsForMemoryBandwidth
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2018  (228 body lines, level 5)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::findBestParamsForMemoryBandwidth( DscParamCandidateIndicesType &dscCandidateIndices, const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims, const std::unordered_set<PrimaryDimTypes> &coreSplitDims, const bool isInputNeig
//   extract   : crustify-ddc/cpp/l3.cpp:8090-8323
//   calls     : e014_isLabeledDsLXNeighbor, e104_clear, e187_clear, e222_allocAllMem, e285_calculateBurstEfficiency, e352_updateChunkDataStagesFromCandidates

// crustify:todo: e367_findBestParamsForArithmeticIntensity
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2500  (214 body lines, level 5)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::findBestParamsForArithmeticIntensity( DscParamCandidateIndicesType &dscCandidateIndices, const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims, const std::unordered_set<PrimaryDimTypes> &coreSplitDims)
//   extract   : crustify-ddc/cpp/l3.cpp:8333-8551
//   calls     : e104_clear, e187_clear, e203_getOpFuncDataFormat, e222_allocAllMem, e286_calculateFlopPerByte, e352_updateChunkDataStagesFromCandidates

// crustify:todo: e368_processHbmPagedTensors
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6741  (4 body lines, level 5)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::processHbmPagedTensors(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:8561-8565
//   calls     : e354_processDscHbmPagedTensors

// crustify:todo: e369_buildCoordinateForAllocation
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7167  (28 body lines, level 5)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::buildCoordinateForAllocation( SuperDsc &mySDsc, DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode, dsc2::CoordPropInfoType &coordPropInfo) const
//   extract   : crustify-ddc/cpp/l3.cpp:8575-8605
//   calls     : e210_isOpCrossCoreReduction, e228_buildCoordinateFromAllocation, e229_sliceCoordinateForCorelet, e355_fillCoordinateCustomWkSliceId

// crustify:todo: e373_getMinParamForDim
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1119  (15 body lines, level 6)
//   class     : L3DlOpsScheduler
//   original  : long L3DlOpsScheduler::getMinParamForDim(const SuperDsc& mySDsc, const DesignSpaceConfig& dsc, PrimaryDimTypes dim) const
//   extract   : crustify-ddc/cpp/l3.cpp:8615-8632
//   calls     : e210_isOpCrossCoreReduction, e365_getMinParamForDimFromOpFunc

// crustify:todo: e374_propagateCoordinateDSC
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7761  (109 body lines, level 6)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::propagateCoordinateDSC(SuperDsc &mySDsc, DesignSpaceConfig &dsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:8642-8752
//   calls     : e060_getHbmAllocations, e369_buildCoordinateForAllocation

// crustify:todo: e377_getInitialChunkParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1382  (20 body lines, level 7)
//   class     : L3DlOpsScheduler
//   original  : DataStructDims L3DlOpsScheduler::getInitialChunkParams( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc, const std::unordered_set<PrimaryDimTypes> &chunkDims)
//   extract   : crustify-ddc/cpp/l3.cpp:8762-8784
//   calls     : e104_clear, e187_clear, e373_getMinParamForDim

// crustify:todo: e378_propagateCoordinate
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7757  (3 body lines, level 7)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::propagateCoordinate(SuperDsc &mySDsc) const
//   extract   : crustify-ddc/cpp/l3.cpp:8794-8797
//   calls     : e374_propagateCoordinateDSC

// crustify:todo: e380_setChunkDataStageParams
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1439  (129 body lines, level 8)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::setChunkDataStageParams(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:8807-8936
//   calls     : e008_hasDimensionReuse, e010_getCoreSplitDimensions, e014_isLabeledDsLXNeighbor, e022_addOrUpdateDataStageParam, e207_generateDscParamCandidates, e222_allocAllMem, e352_updateChunkDataStagesFromCandidates, e366_findBestParamsForMemoryBandwidth, e367_findBestParamsForArithmeticIntensity, e377_getInitialChunkParams

// crustify:todo: e382_run
//   authority : dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7912  (122 body lines, level 9)
//   class     : L3DlOpsScheduler
//   original  : void L3DlOpsScheduler::run(SuperDsc &mySDsc)
//   extract   : crustify-ddc/cpp/l3.cpp:8946-9068
//   calls     : e001_isSameDscGroup, e053_verifyScheduleTree, e054_prepDsc, e059_getPagedDimensions, e214_optimizeHbmLdsOutputInScheduleTree, e221_fillTransferZeroPaddingInfo, e222_allocAllMem, e289_optimizeHbmTransfers, e290_createChunkLoops, e291_fillTransferMulticastInfo, e292_fillAllocationStartAddrAndOffset, e293_setLxBufferType, e295_fillExplicitTransferSize, e332_createSynchronization …

