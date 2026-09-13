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

use crate::arch::{Arch, Bounded, Bytes, Elements, IsaGen, Target};
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::schedule::ddc::fold::{
    AllocId, AllocLayout, Alpha, Beta, Cardinality, ElemArrDistribution, FoldParamInfo, NodeId,
    PadType, RefComponents, TemporalLoopDistribution,
};
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind, stricter_max, stricter_min};
use crate::schedule::ddc::transformation::{DsType, LoopId, Scale};
use crate::schedule::ddc::transformation_util::{
    DataStage, DataStages, InsertionPoint, LoopBands, LoopDims, LoopNode, PaddingForm,
    PrimaryDimAndKind, StageDims, StageName, construct_datastage_from,
};
use crate::schedule::ddc::v1::{self, ComputeOps};
use crate::schedule::dsc2::{
    AddressFold, AllocateNode, BlockNode, ChildPos, Coordinate, CoordinateCategory, Dsc, Dsts, Fold,
    FoldCardinality, FoldCoeff, FoldDim, FoldLabel, FoldPosition, LdsIdx, Node, NodeName, PadFold,
    ReplicationFactor, SchedNode, ScheduleTree, SyncDirection, SyncNode, SyncStrength, SyncUnits,
    TransferNode, TransferPadding, Via, ZeroPadFolds,
};
use crate::schedule::l3::dsc::{
    AddressCoord, BufferOffset, Buffering, ByteAddress, CoreletOffset, CoreletShare, CoreletsUsed,
    DATA_STAGE_CHUNK, DATA_STAGE_CORE, DataStages as L3DataStages, DesignSpaceConfig,
    DimCandidates, DimPadding, DimStage, DscCandidates, DscGroup, DscIdx, DscParamCandidates,
    FilledDims, IbrStage, IndexTensor, IndirectAlloc, InitialPlacement, InsertSide, L3Transfer,
    LabeledDs, MemOrg, MemOrgs, MulticastDegree, NodeParents, OnePageStage, PadElems, PadSizes,
    PagedStages, Pinning, ScheduleNodes, ScheduleTrees, SchedulerMetadata, StickVolume,
    StickVolumes, SuperChunkStage, SuperDsc, SymbolicDimInfo, TransferNodes, UnneededPad, WkSlice,
    WkSliceCount, WkSliceId,
};
use crate::units::{Core, Corelet, Row};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::NonZeroU32;
use sys_arch_spec::arch_enums::{DataLocation, OpFunc, SenComponent};

/// THE WITNESS `isSameDscGroup` HANDS BACK — constructible only from a [`SuperDsc`], whose DSC list
/// is non-empty by type, so the caller's `DT_CHECK` on the result has nothing left to test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SameDscGroup(());

/// Replaces: e001_isSameDscGroup
///
/// `DT_CHECK_MSG(mySDsc.dscs_.size() >= 1, "Expect at least one DSC.")` then `return true` is the
/// whole body, so the answer is the witness that a super-DSC has a DSC in it.
///
/// ⛔ TRAP: NEITHER THE NAME NOR THE CALLER'S MESSAGE DESCRIBES THE BODY — `run` calls it under
/// "Expect DSCs in the same group" (`:7938`), and no field of any DSC is read at all.
#[must_use]
pub const fn same_dsc_group(_sdsc: &SuperDsc) -> SameDscGroup {
    SameDscGroup(())
}

/// Replaces: e002_isLabeledDsDimensionBroadcast
///
/// Whether a labelled data structure BROADCASTS along `dim` — `scale_ < 1`, which is a fractional
/// scale, the one-element stick (`-1`) or the whole-stick dim (`-2`).
///
/// ⛔ `None` IS THE `DT_CHECK("Invalid layoutDimOrder_ index.")` ARM AND IT IS REACHABLE: callers
/// walk `getLayoutDims(ldsIdx)`, a DIFFERENT list from `primaryDsInfo_`'s layout order. The `dsc`
/// parameter is gone — it served only to reach the order [`LabeledDs`] now carries zipped.
#[must_use]
pub fn is_labeled_ds_dimension_broadcast(lds: &LabeledDs, dim: PrimaryDim) -> Option<bool> {
    lds.scale(dim).map(|scale| match scale {
        Scale::Sized(size) => size < 1.0,
        Scale::UnitStick | Scale::StickDim => true,
    })
}

/// Replaces: e003_isDimensionCoreletSplit
///
/// Whether `dim` is split across corelets: corelet 0 holds less of it than the whole core does.
///
/// ⭐ THE TWO ARMS ARE ONE COMPARISON — the core data stage's `primaryDimToVal_st(dim, .., -1, 0)`
/// against `(.., -1, -1)` and `CoreletD_`'s value against `CoreD_`'s ask it of two carriers, so
/// [`CoreletShare`] states it once and `dataStageParam_.count(dataStageCoreIdx)` stops being a
/// branch. A dim neither carrier states answers the reference's `1 < 1`.
#[must_use]
pub fn is_dimension_corelet_split(dsc: &DesignSpaceConfig, dim: PrimaryDim) -> bool {
    dsc.corelets_used.splits()
        && dsc
            .corelet_shares
            .get(&dim)
            .copied()
            .is_some_and(CoreletShare::splits)
}

/// Replaces: e004_voidPaddingIfChunking
///
/// VOIDS a chunk stage's padding on every dim whose extent — or whose window dim's extent — chunking
/// moved off the reference stage's, because a chunk owns the whole dim's padding or none of it.
///
/// ⭐ `CARRY_UNNEEDED_PAD` IS `carryUnneededPadToChunk`, a file-static `bool` initialised `true` and
/// never written (`:48`), so its zeroing arm is DEAD: as a const generic that arm leaves the build
/// instead of being tested per dim, and the reference's own TODO to flip it stays expressible.
pub fn void_padding_if_chunking<const CARRY_UNNEEDED_PAD: bool>(
    ds: &mut FilledDims,
    ref_ds: &FilledDims,
) {
    let chunked: Vec<PrimaryDim> = ds
        .dims()
        .padding
        .iter()
        .filter(|(dim, pad)| {
            let moved = |dim: PrimaryDim| ref_ds.dims().extent(dim) != ds.dims().extent(dim);
            moved(**dim) || pad.window_dim.is_some_and(moved)
        })
        .map(|(dim, _)| *dim)
        .collect();
    for dim in chunked {
        if let Some(pad) = ds.padding_mut().get_mut(&dim) {
            pad.sizes = pad.sizes.voided();
            if !CARRY_UNNEEDED_PAD {
                pad.unneeded = UnneededPad::NONE;
            }
        }
    }
}

/// Replaces: e005_addOrUpdateSymbolicInfoInParams
///
/// Carries the core stage's symbolic dims onto the chunk stage for every dim CHUNKING LEFT ALONE,
/// then adopts the core's volume limits pruned against the dims that survived.
///
/// ⛔ BOTH "Expect non-empty data-stage parameters" `DT_CHECK`s ARE [`FilledDims`], and the
/// `maxSymbolicVolume_` assignment is fused into `Symbolic::prune_volumes_from` — the state
/// between assignment and prune is the one state a well-formed `Symbolic` cannot hold.
pub fn add_or_update_symbolic_info_in_params(
    chunk_params: &mut FilledDims,
    core_params: &FilledDims,
) {
    let unchunked: Vec<(PrimaryDim, SymbolicDimInfo)> = core_params
        .dims()
        .symbolic
        .info()
        .iter()
        .filter(|(dim, _)| chunk_params.dims().extent(**dim) == core_params.dims().extent(**dim))
        .map(|(dim, info)| (*dim, *info))
        .collect();
    for (dim, info) in unchunked {
        chunk_params.symbolic_mut().add_dim(dim, info);
    }
    chunk_params
        .symbolic_mut()
        .prune_volumes_from(&core_params.dims().symbolic);
}

/// Replaces: e006_getLabeledDsWkSliceMulticastDegree
///
/// HOW MANY CORES SHARE ONE LABELLED DATA STRUCTURE'S DATA — the cores of `dscs` whose work slice
/// agrees with the group's first core on every layout dim of `lds`.
///
/// ⛔ `None` IS ONE OF THREE ABORTS: `getLayoutDims(lds)` reaching no allocate node
/// (`dsc/dsc2.cpp:4022`), `coreIdToWkSlice_.at(mainCoreId)` missing the main core, or `.at(dim)`
/// missing a layout dim (`:204-206`). `!dscIndices.empty()`, `dscs_.at()` and `coreIdsUsed_[0]`
/// abort too, but [`DscGroup`] and `CoreIdsUsed` discharge those three before this is called.
#[must_use]
pub fn labeled_ds_wk_slice_multicast_degree(
    sdsc: &SuperDsc,
    lds: LdsIdx,
    dscs: &DscGroup<'_>,
) -> Option<MulticastDegree> {
    let main = dscs.main();
    let processing: BTreeSet<Core> = dscs
        .iter()
        .flat_map(|dsc| dsc.core_ids_used.iter())
        .collect();
    let layout = main.layout_dims.get(&lds)?;
    let reference = sdsc.core_id_to_wk_slice.get(&main.core_ids_used.first())?;
    let mut degree = 0;
    for (core, slice) in &sdsc.core_id_to_wk_slice {
        if !processing.contains(core) {
            continue;
        }
        let mut matches = true;
        for dim in layout.iter() {
            if slice.at(dim)? != reference.at(dim)? {
                matches = false;
                break;
            }
        }
        if matches {
            degree += 1;
        }
    }
    Some(MulticastDegree(degree))
}

/// WHAT ROLE A DIM PLAYS IN THE SCHEDULE — `ScheduleDimTypes` (`L3DlOpsScheduler.h:88`), less its
/// `ScheduleDimTypesCount` terminator, which is a count and not a role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScheduleDimType {
    /// `ELEMENTWISE`.
    Elementwise,
    /// `BROADCAST`.
    Broadcast,
    /// `REDUCTION`.
    Reduction,
    /// `WINDOW_PADDED`.
    WindowPadded,
    /// `REUSE`.
    Reuse,
}

/// Replaces: e007_scheduleDimTypeToString
///
/// The name the scheduler prints for a dim's role.
///
/// ⭐ `DT_ERROR("Unsupported ScheduleDimTypes.")` IS UNSPELLABLE: the only value the switch declines
/// to name is the enum's `Count` terminator, which [`ScheduleDimType`] does not carry, so the
/// default arm has no input and the return type needs no absence in it.
#[must_use]
pub const fn schedule_dim_type_to_string(ty: ScheduleDimType) -> &'static str {
    match ty {
        ScheduleDimType::Elementwise => "Elementwise",
        ScheduleDimType::Broadcast => "Broadcast",
        ScheduleDimType::Reduction => "Reduction",
        ScheduleDimType::WindowPadded => "Window/Padded",
        ScheduleDimType::Reuse => "Reuse",
    }
}

/// Replaces: e008_hasDimensionReuse
///
/// Whether some layout dim is missing from at least one of the DSC's primary data structures — asked
/// only of a DSC that has a KERNEL and more than one data structure.
///
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: the tally counts ENTRIES, not data structures, so one
/// `layoutDimOrder_` that names a dim twice can reach the count on its own and HIDE the reuse
/// (`:307-314`). The `int` against `size()` comparison beside it is signed/unsigned but harmless.
#[must_use]
pub fn has_dimension_reuse(dsc: &DesignSpaceConfig) -> bool {
    let structures = dsc.primary_ds_info.len();
    if structures <= 1 || !dsc.primary_ds_info.contains_key(&DsType::Kernel) {
        return false;
    }
    let mut count_per_dim: BTreeMap<PrimaryDim, usize> = BTreeMap::new();
    for info in dsc.primary_ds_info.values() {
        for dim in info.layout.iter() {
            *count_per_dim.entry(dim).or_insert(0) += 1;
        }
    }
    count_per_dim.values().any(|count| *count < structures)
}

#[cfg(test)]
mod tests_e001_e008 {
    use super::*;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims;
    use crate::schedule::dsc2::LayoutDims;
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DimPadding, DscList, Granularity,
        LabeledDsList, MaxSize, NamedDims, PadElems, PadSizes, PrimaryDsInfo, StageDims, Symbolic,
        VolumeLimit, WkSliceId,
    };
    use std::num::NonZeroU32;

    fn core(index: u32) -> Core {
        Core::checked(index).expect("core in range")
    }

    fn step(value: u32) -> Granularity {
        Granularity::new(NonZeroU32::new(value).expect("a positive step"))
    }

    fn plain_dsc() -> DesignSpaceConfig {
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: None,
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(core(0), vec![]),
            layout_dims: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(DsType::Output, vec![], LdsIdx(183), Pinning::default()),
                vec![],
            ),
            data_stages: DataStages::new(plain_stage(), plain_stage()),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        }
    }

    fn one_dim_stage() -> FilledDims {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::In, Extent(1));
        filled(dims)
    }

    fn plain_stage() -> DataStage {
        let named = NamedDims {
            name: StageName::chunk(),
            dims: one_dim_stage(),
        };
        DataStage {
            ss: named.clone(),
            el: named,
        }
    }

    fn layout_only(layout: LayoutDims) -> PrimaryDsInfo {
        PrimaryDsInfo {
            layout,
            stick: StickDims::default(),
        }
    }

    fn filled(dims: StageDims) -> FilledDims {
        FilledDims::of(dims).expect("a stage that states a dim")
    }

    /// e001 — the answer is a witness, and a super-DSC cannot be built without a DSC to witness.
    #[test]
    fn same_dsc_group_is_a_witness() {
        let sdsc = SuperDsc::new(
            DscList::new(plain_dsc(), vec![]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        assert_eq!(same_dsc_group(&sdsc), SameDscGroup(()));
        assert_eq!(sdsc.dscs().iter().count(), 1);
    }

    /// e002 — a scale below 1 broadcasts, and a dim the layout order does not name has no answer.
    #[test]
    fn broadcast_is_a_scale_below_one() {
        let lds = LabeledDs::new(
            DsType::Input,
            vec![
                (PrimaryDim::In, Scale::Sized(2.0)),
                (PrimaryDim::Out, Scale::Sized(0.5)),
                (PrimaryDim::Ij, Scale::UnitStick),
                (PrimaryDim::Mb, Scale::StickDim),
            ],
            LdsIdx(183),
            Pinning::default(),
        );
        assert_eq!(lds.ds_type(), DsType::Input);
        assert_eq!(
            is_labeled_ds_dimension_broadcast(&lds, PrimaryDim::In),
            Some(false)
        );
        assert_eq!(
            is_labeled_ds_dimension_broadcast(&lds, PrimaryDim::Out),
            Some(true)
        );
        assert_eq!(
            is_labeled_ds_dimension_broadcast(&lds, PrimaryDim::Ij),
            Some(true)
        );
        assert_eq!(
            is_labeled_ds_dimension_broadcast(&lds, PrimaryDim::Mb),
            Some(true)
        );
        assert_eq!(is_labeled_ds_dimension_broadcast(&lds, PrimaryDim::Y), None);
    }

    /// e003 — one corelet never splits, and a split is corelet 0 holding less than the whole.
    #[test]
    fn corelet_split_is_a_short_share() {
        let mut dsc = plain_dsc();
        dsc.corelet_shares.insert(
            PrimaryDim::In,
            CoreletShare {
                corelet0: Extent(8),
                whole: Extent(16),
            },
        );
        dsc.corelet_shares.insert(
            PrimaryDim::Out,
            CoreletShare {
                corelet0: Extent(16),
                whole: Extent(16),
            },
        );
        assert!(!is_dimension_corelet_split(&dsc, PrimaryDim::In));
        dsc.corelets_used = CoreletsUsed::new(NonZeroU32::new(2).expect("two corelets"));
        assert!(is_dimension_corelet_split(&dsc, PrimaryDim::In));
        assert!(!is_dimension_corelet_split(&dsc, PrimaryDim::Out));
        assert!(!is_dimension_corelet_split(&dsc, PrimaryDim::Ij));
    }

    /// e004 — a moved extent voids the dim's padding, directly or through its window dim, and an
    /// unpadded dim has nothing to void.
    #[test]
    fn chunking_voids_moved_padding() {
        let padded = |window: Option<PrimaryDim>| DimPadding {
            sizes: PadSizes::of(PadElems(2), PadElems(3)),
            window_dim: window,
            unneeded: UnneededPad {
                total: PadElems(1),
                front: PadElems(1),
                back: PadElems(0),
            },
            ..DimPadding::default()
        };
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::In, Extent(16));
        dims.extents.insert(PrimaryDim::Out, Extent(8));
        dims.extents.insert(PrimaryDim::Ij, Extent(4));
        dims.padding.insert(PrimaryDim::In, padded(None));
        dims.padding.insert(PrimaryDim::Out, padded(None));
        dims.padding
            .insert(PrimaryDim::Ij, padded(Some(PrimaryDim::Out)));
        dims.padding.insert(PrimaryDim::Mb, DimPadding::default());
        let mut reference = dims.clone();
        reference.extents.insert(PrimaryDim::Out, Extent(32));
        let ref_ds = filled(reference);

        let mut ds = filled(dims.clone());
        void_padding_if_chunking::<true>(&mut ds, &ref_ds);
        let padding = &ds.dims().padding;
        assert_eq!(padding[&PrimaryDim::In].sizes, padded(None).sizes);
        assert_eq!(padding[&PrimaryDim::Out].sizes, PadSizes::Voided);
        assert_eq!(padding[&PrimaryDim::Ij].sizes, PadSizes::Voided);
        assert_eq!(padding[&PrimaryDim::Mb].sizes, PadSizes::Unpadded);
        assert_eq!(padding[&PrimaryDim::Out].unneeded, padded(None).unneeded);

        let mut ds = filled(dims);
        void_padding_if_chunking::<false>(&mut ds, &ref_ds);
        assert_eq!(
            ds.dims().padding[&PrimaryDim::Out].unneeded,
            UnneededPad::NONE
        );
        assert_eq!(
            ds.dims().padding[&PrimaryDim::In].unneeded,
            padded(None).unneeded
        );
    }

    /// e005 — the reference's own worked example (`dsc/dims.cpp:719-728`): `abc` limited to 2048 with
    /// `a` chunked away becomes `bc` limited to `min(64 * 64, 2048 / 4) = 512`.
    #[test]
    fn symbolic_volumes_prune_onto_the_dims_that_survive() {
        let info = |max: u32, granularity: u32| SymbolicDimInfo {
            max_size: MaxSize(max),
            granularity: step(granularity),
        };
        let core_info = BTreeMap::from([
            (PrimaryDim::In, info(128, 4)),
            (PrimaryDim::Out, info(64, 2)),
            (PrimaryDim::Ij, info(64, 2)),
        ]);
        let abc = BTreeSet::from([PrimaryDim::In, PrimaryDim::Out, PrimaryDim::Ij]);
        let core_params = filled(StageDims {
            extents: BTreeMap::from([
                (PrimaryDim::In, Extent(128)),
                (PrimaryDim::Out, Extent(64)),
                (PrimaryDim::Ij, Extent(64)),
            ]),
            padding: BTreeMap::new(),
            symbolic: Symbolic::new(core_info, BTreeMap::from([(abc, VolumeLimit(2048))])),
            ..StageDims::default()
        });
        let mut chunk_params = filled(StageDims {
            extents: BTreeMap::from([
                (PrimaryDim::In, Extent(32)),
                (PrimaryDim::Out, Extent(64)),
                (PrimaryDim::Ij, Extent(64)),
            ]),
            padding: BTreeMap::new(),
            symbolic: Symbolic::default(),
            ..StageDims::default()
        });

        add_or_update_symbolic_info_in_params(&mut chunk_params, &core_params);

        let symbolic = &chunk_params.dims().symbolic;
        assert_eq!(
            symbolic.info().keys().copied().collect::<Vec<_>>(),
            vec![PrimaryDim::Out, PrimaryDim::Ij]
        );
        assert_eq!(
            symbolic.volumes(),
            &BTreeMap::from([(
                BTreeSet::from([PrimaryDim::Out, PrimaryDim::Ij]),
                VolumeLimit(512)
            )])
        );
    }

    /// e006 — the degree counts the group's cores whose slice matches the first core's, and a core
    /// outside the group does not count however well it matches.
    #[test]
    fn multicast_degree_counts_matching_group_cores() {
        let lds = LdsIdx(0);
        let mut dsc = plain_dsc();
        dsc.core_ids_used = CoreIdsUsed::new(core(0), vec![core(1), core(2)]);
        dsc.layout_dims
            .insert(lds, LayoutDims::new(PrimaryDim::In, vec![]));
        let slice = |value: i32| WkSlice(BTreeMap::from([(PrimaryDim::In, WkSliceId(value))]));
        let sdsc = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            BTreeMap::new(),
            BTreeMap::from([
                (core(0), slice(0)),
                (core(1), slice(0)),
                (core(2), slice(1)),
                (core(3), slice(0)),
            ]),
            BTreeMap::new(),
        );
        let group = DscGroup::new(&dsc, vec![]);
        assert_eq!(
            labeled_ds_wk_slice_multicast_degree(&sdsc, lds, &group),
            Some(MulticastDegree(2))
        );
        // The `getLayoutDims` arm: the DSC states no layout order for this labelled DS.
        assert_eq!(
            labeled_ds_wk_slice_multicast_degree(&sdsc, LdsIdx(1), &group),
            None
        );
        // The `coreIdToWkSlice_.at(mainCoreId)` arm: no work slice for `coreIdsUsed_[0]`.
        let no_main_slice = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            BTreeMap::new(),
            BTreeMap::from([(core(1), slice(0))]),
            BTreeMap::new(),
        );
        assert_eq!(
            labeled_ds_wk_slice_multicast_degree(&no_main_slice, lds, &group),
            None
        );
        // The `.at(dim)` arm: a group core whose slice does not state the layout dim.
        let sparse_slice = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            BTreeMap::new(),
            BTreeMap::from([(core(0), slice(0)), (core(1), WkSlice::default())]),
            BTreeMap::new(),
        );
        assert_eq!(
            labeled_ds_wk_slice_multicast_degree(&sparse_slice, lds, &group),
            None
        );
    }

    /// e007 — the five spellings the scheduler prints, and there is no sixth to ask for.
    #[test]
    fn schedule_dim_types_spell_themselves() {
        assert_eq!(
            [
                ScheduleDimType::Elementwise,
                ScheduleDimType::Broadcast,
                ScheduleDimType::Reduction,
                ScheduleDimType::WindowPadded,
                ScheduleDimType::Reuse,
            ]
            .map(schedule_dim_type_to_string),
            [
                "Elementwise",
                "Broadcast",
                "Reduction",
                "Window/Padded",
                "Reuse"
            ]
        );
    }

    /// e008 — a dim missing from one data structure is reuse; a dim in all of them is not, and a DSC
    /// without a KERNEL is never asked.
    #[test]
    fn reuse_is_a_dim_one_data_structure_lacks() {
        let mut dsc = plain_dsc();
        dsc.primary_ds_info.insert(
            DsType::Input,
            layout_only(LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Out])),
        );
        dsc.primary_ds_info.insert(
            DsType::Kernel,
            layout_only(LayoutDims::new(PrimaryDim::Out, vec![])),
        );
        assert!(has_dimension_reuse(&dsc));

        dsc.primary_ds_info.insert(
            DsType::Kernel,
            layout_only(LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Out])),
        );
        assert!(!has_dimension_reuse(&dsc));

        dsc.primary_ds_info.remove(&DsType::Kernel);
        dsc.primary_ds_info.insert(
            DsType::Output,
            layout_only(LayoutDims::new(PrimaryDim::Out, vec![PrimaryDim::Y])),
        );
        assert!(!has_dimension_reuse(&dsc));
    }
}

// ⭐ TYPES FOR ENTRIES 009-016. The DSC-side and super-DSC-side facts these entries read live in
// [`crate::schedule::l3::dsc`] beside the rest of this stage's reduced vocabulary; what is declared
// here is the L3 SCHEDULER'S OWN state — its private `Metadata` and the allocate node it mints.

/// `dsc2::AllocateNode` (`dsc/dsc2.h:974`) AS ENTRY 016 MINTS ONE.
///
/// ⛔ A THIRD PROJECTION OF THAT STRUCT, beside [`crate::schedule::dsc2::AllocateNode`] (the
/// component and lds the fold units read) and `ddc::transformation_util::DdcAllocateNode` (the
/// layout, padding and user refcounts DDC's own mint keeps). This one carries the BUFFER COUNT,
/// which neither of those reads, and no user list, because this mint records none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3AllocateNode {
    /// `name_`, the caller's.
    pub name: NodeName,
    /// `ldsIdx_`.
    pub lds: LdsIdx,
    /// `component_` — a `SenComponents` and not a memory: the stage allocates in LX and HBM and in
    /// register files.
    pub component: SenComponent,
    /// `numBuffers_`.
    pub buffering: Buffering,
    /// `layoutDimOrder_` zipped with `maxDimSizes_`, whose fresh entries are the reference's
    /// `resize(n, -1)`.
    pub layout: AllocLayout,
    /// `padding_`.
    pub padding: PaddingForm,
    /// `indirectAllocType_` (`dsc/dsc2.h:990`) — what this allocation indirects through, which entry
    /// 226 copies off the paged tensor's own HBM allocation.
    pub indirect: Option<IndirectAlloc>,
    /// `relatedIndirectAccessAlloc_` — the allocation on the other side of that indirection.
    pub related_indirect: Option<AllocId>,
}

/// `Metadata::Allocation` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:161`) reduced to the one map
/// entry 016 writes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct L3Allocation {
    /// `ldsIdxAndAllocNode`.
    pub lds_idx_and_alloc_node: BTreeMap<LdsIdx, AllocId>,
}

/// WHAT THE L3 SCHEDULER RECORDS FOR ONE DSC — its private `Metadata`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:111`), reduced to the `newAllocations_` (`:167`) that
/// entry 016 writes so the chunks' LX memory can later be allocated against it.
///
/// ⛔ A DIFFERENT C++ CLASS FROM [`crate::schedule::ddc::metadata::Metadata`] even where their
/// fields coincide: the two stages each keep their own, and `dscMetadata` (`:205`) is keyed per DSC
/// while DDC's is one per run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DscMetadata {
    /// `newAllocations_`, each component to what was newly allocated in it.
    pub new_allocations: BTreeMap<SenComponent, L3Allocation>,
}

/// A LABELLED DS PROVED READY FOR AN L3 ALLOCATION, WITH THE LAYOUT ORDER IT GETS.
///
/// ⛔ ENTRY 016'S *"Handling of external allocations with repeated dimensions is not yet
/// implemented"* IS THIS TYPE: a layout order naming a dim twice has no witness, so the abort is
/// unspellable rather than checked. Its two `.at()` throws are the same [`None`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshL3Allocation {
    lds: LdsIdx,
    component: SenComponent,
    layout: AllocLayout,
    pinning: Pinning,
}

impl FreshL3Allocation {
    /// The witness, or [`None`] for that one refusal and for a labelled DS the DSC does not state.
    #[must_use]
    pub fn of(dsc: &DesignSpaceConfig, lds: LdsIdx, component: SenComponent) -> Option<Self> {
        let dims = dsc.layout_dims.get(&lds)?.to_vec();
        let pinning = dsc.labeled_ds.at(lds)?.pinning().clone();
        let distinct: BTreeSet<PrimaryDim> = dims.iter().copied().collect();
        (distinct.len() == dims.len()).then(|| Self {
            lds,
            component,
            layout: AllocLayout(dims.into_iter().map(|dim| (dim, None)).collect()),
            pinning,
        })
    }
}

/// WHAT ENTRY 015 ASKS OF A SCHEDULE TREE — the parent walk, which is the MECHANISM for reaching a
/// node's enclosing loops rather than a fact about them.
pub trait LoopNesting {
    /// `node->getOwnerLoop()` (`dsc/dsc2.cpp:1896`) — the nearest enclosing `LOOP`, walking `prev_`,
    /// absent where that walk runs off the top of the tree.
    fn owner_loop(&self, node: NodeId) -> Option<LoopId>;
    /// `node->getPrev() != nullptr` (`dsc/dsc2.h:463`) — `prev_` is the PARENT block, so this is
    /// false for the tree root and nothing else.
    fn has_parent(&self, node: LoopId) -> bool;
}

/// Replaces: e009_getStickSize
///
/// How many elements of `dim` one whole stick of `ds_type` holds, and ONE ELEMENT for a dim the
/// stick does not name.
///
/// ⭐ THE `break` IS A FIND: the reference walks an `unordered_map`, which holds one entry per dim.
/// ⛔ [`None`] IS [`DesignSpaceConfig::cumulative_stick_sizes`]'s — a DS type the DSC describes no
/// stick for, or an extent product the fold cannot count.
#[must_use]
pub fn stick_size(dsc: &DesignSpaceConfig, ds_type: DsType, dim: PrimaryDim) -> Option<Elements> {
    Some(
        dsc.cumulative_stick_sizes(ds_type)?
            .iter()
            .find(|(walked, _)| *walked == dim)
            .map_or(Elements(1), |(_, size)| *size),
    )
}

/// Replaces: e010_getCoreSplitDimensions
///
/// Which dims the super-DSC's DSCs do NOT agree on in their core data stage — a dim whose extent
/// differs from DSC 0's in ANY DSC is a dim the work was split across cores along.
///
/// ⭐ `IJ` AND `KIJ` ARE SKIPPED as combined dims; `PrimaryDimTypesCount` is not a [`PrimaryDim`] at
/// all, so the reference's third skip has nothing to skip. Two absent extents agree, which is the
/// reference's `-1 == -1`.
/// ⭐ TRAP, DISCHARGED BY CONSTRUCTION: the reference `DT_CHECK`s the core data stage on DSC 0 only
/// and then `.at()`s EVERY DSC's, so a later DSC without one throws unguarded. Mandatory
/// [`DesignSpaceConfig::core_stage`] removes both, leaving this total.
#[must_use]
pub fn core_split_dimensions(sdsc: &SuperDsc) -> BTreeSet<PrimaryDim> {
    let mut dims = BTreeSet::new();
    for dim in PrimaryDim::ALL {
        if matches!(dim, PrimaryDim::Ij | PrimaryDim::Kij) {
            continue;
        }
        let main = sdsc.dscs().first().core_stage().dims().extent(dim);
        if sdsc
            .dscs()
            .iter()
            .any(|dsc| dsc.core_stage().dims().extent(dim) != main)
        {
            dims.insert(dim);
        }
    }
    dims
}

/// Replaces: e011_getLabeledDsWithDsType
///
/// APPENDS the POSITION of every labelled DS of `ds_type` to `positions`.
///
/// ⛔ TRAP: it does not clear — a call adds to whatever the caller's vector already held. Both call
/// sites pass a fresh one (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4423,4489`).
/// ⛔ TRAP: these are POSITIONS in `labeledDs_`, not the `ldsIdx_` each entry records; the two need
/// not agree, and [`all_labeled_ds_indices`] returns the other one.
pub fn labeled_ds_with_ds_type(
    dsc: &DesignSpaceConfig,
    ds_type: DsType,
    positions: &mut Vec<LdsIdx>,
) {
    for (at, lds) in dsc.labeled_ds.indexed() {
        if lds.ds_type() == ds_type {
            positions.push(at);
        }
    }
}

/// Replaces: e012_getAllLabeledDsIndicesSet
///
/// Every index the DSC's labelled DSs RECORD — each entry's own `ldsIdx_`.
///
/// ⛔ TRAP: `ldsIdx_` defaults to `183` (`dsc/dscdefn.h:323`) and nothing here relates it to the
/// position the entry sits at, so this set and [`labeled_ds_with_ds_type`]'s positions are two
/// different answers over one list.
#[must_use]
pub fn all_labeled_ds_indices(dsc: &DesignSpaceConfig) -> BTreeSet<LdsIdx> {
    dsc.labeled_ds.iter().map(LabeledDs::recorded).collect()
}

/// Replaces: e013_getHbmPinnedLabeledDsIndicesSet
///
/// [`all_labeled_ds_indices`] restricted to the HBM-pinned entries — `memOrg_.at(HBM).isPresent`
/// (`dsc/dscdefn.h:369`), and the same recorded `ldsIdx_` rather than a position.
#[must_use]
pub fn hbm_pinned_labeled_ds_indices(dsc: &DesignSpaceConfig) -> BTreeSet<LdsIdx> {
    dsc.labeled_ds
        .iter()
        .filter(|lds| lds.pinning().hbm())
        .map(LabeledDs::recorded)
        .collect()
}

/// Replaces: e014_isLabeledDsLXNeighbor
///
/// Whether an LX-pinned `INPUT` labelled DS is fetched from a neighbour core: true when the DSC's own
/// schedule step ALSO names a data DSC, which is what an input neighbour fetch is
/// (`dsc/superdsc.h:31`).
///
/// ⭐ ONE CORE ANSWERS FOR ALL OF THEM — every core a DSC uses carries the same step, so the
/// reference reads the first and so does this; `coreIdsUsed_[0]` cannot miss, because
/// [`CoreIdsUsed`] is non-empty. A DSC appears in at most one step.
/// ⛔ [`None`] is a DSC index past the end of `dscs_`, or a core the super-DSC states no schedule for.
#[must_use]
pub fn is_labeled_ds_lx_neighbor(sdsc: &SuperDsc, dsc: DscIdx, lds: &LabeledDs) -> Option<bool> {
    if !lds.pinning().lx || lds.ds_type() != DsType::Input {
        return Some(false);
    }
    let core = sdsc.dscs().at(dsc)?.core_ids_used.first();
    Some(
        sdsc.core_id_to_dsc_schedule
            .get(&core)?
            .iter()
            .any(|step| step.dl_dsc == Some(dsc) && step.data_dsc.is_some()),
    )
}

/// Replaces: e015_getParentLoopNodes
///
/// The loops enclosing a schedule node, INNERMOST FIRST and WITHOUT the root loop — the walk stops
/// at the loop that has no parent block.
///
/// ⭐ THE `dsc` PARAMETER AND ITS *"Expect valid schedule tree"* `DT_CHECK` ARE BOTH GONE BY
/// CONSTRUCTION: `node` is a node OF the tree being walked, so that tree is not empty.
#[must_use]
pub fn parent_loop_nodes<T: LoopNesting + ?Sized>(tree: &T, node: NodeId) -> Vec<LoopId> {
    let mut loops = Vec::new();
    let mut parent = tree.owner_loop(node);
    while let Some(enclosing) = parent {
        if !tree.has_parent(enclosing) {
            break;
        }
        loops.push(enclosing);
        parent = tree.owner_loop(enclosing.0);
    }
    loops
}

/// Replaces: e016_createAllocateNode
///
/// MINTS THE L3 ALLOCATE NODE for one labelled DS in one component: gives it the DS's layout order
/// with every max size unset, marks each padded layout dim `PADDED_FULLSPAN_WUNNEEDED` when the DS's
/// own LX organisation is padded, and — for an HBM-pinned LX allocation — REGISTERS it in
/// `dscMetadata.at(dsc).newAllocations_[component]`, which is what later allocates the chunks' LX.
///
/// ⭐ `alloc` IS THE IDENTITY ITS OWNER ISSUES — `new dsc2::AllocateNode()` in the reference, whose
/// pointer is what the registry holds it under.
/// ⛔ [`None`] IS ONE OF TWO ABORTS: no `dscMetadata` entry for `dsc_idx`, or that labelled DS
/// already registered in the component. The third is [`FreshL3Allocation`]; the *"Expect
/// dataStageParam_ entry"* fourth is discharged by [`DesignSpaceConfig::core_stage`].
pub fn create_allocate_node(
    dsc: &DesignSpaceConfig,
    metadata: &mut BTreeMap<DscIdx, DscMetadata>,
    fresh: FreshL3Allocation,
    buffering: Buffering,
    name: NodeName,
    dsc_idx: DscIdx,
    alloc: AllocId,
) -> Option<L3AllocateNode> {
    let FreshL3Allocation {
        lds,
        component,
        layout,
        pinning,
    } = fresh;
    let mut padding = PaddingForm::default();
    if component == SenComponent::Lx && pinning.lx_padded {
        for (dim, _) in &layout.0 {
            if dsc.core_stage().dims().padding.contains_key(dim) {
                padding.set_padding(*dim, PadType::PaddedFullSpanWUnneeded);
            }
        }
    }

    if pinning.hbm() && component == SenComponent::Lx {
        let allocated = metadata
            .get_mut(&dsc_idx)?
            .new_allocations
            .entry(component)
            .or_default();
        if allocated.lds_idx_and_alloc_node.contains_key(&lds) {
            return None;
        }
        allocated.lds_idx_and_alloc_node.insert(lds, alloc);
    }

    Some(L3AllocateNode {
        name,
        lds,
        component,
        buffering,
        layout,
        padding,
        indirect: None,
        related_indirect: None,
    })
}

// ⭐ TESTS FOR ENTRIES 009-016. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e009_e016 {
    use std::collections::{BTreeMap, BTreeSet};

    use sys_arch_spec::arch_enums::SenComponent;

    use super::{
        AllocId, Buffering, DesignSpaceConfig, DsType, DscIdx, DscMetadata, Elements, FilledDims,
        FreshL3Allocation, L3AllocateNode, LabeledDs, LdsIdx, LoopId, LoopNesting, NodeId,
        NodeName, PadType, Pinning, PrimaryDim, SuperDsc, all_labeled_ds_indices,
        core_split_dimensions, create_allocate_node, hbm_pinned_labeled_ds_indices,
        is_labeled_ds_lx_neighbor, labeled_ds_with_ds_type, parent_loop_nodes, stick_size,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, StickDims};
    use crate::schedule::ddc::transformation_util::StageName;
    use crate::schedule::dsc2::LayoutDims;
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DATA_STAGE_CORE, DataStage, DataStages, DimPadding, DscList,
        DscScheduleStep, LabeledDsList, NamedDims, PrimaryDsInfo, StageDims,
    };
    use crate::units::Core;

    /// A three-deep nest: node 7 inside loop 5 inside the ROOT loop 3.
    struct TestTree;

    impl LoopNesting for TestTree {
        fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
            match node.0 {
                7 => Some(LoopId(NodeId(5))),
                5 => Some(LoopId(NodeId(3))),
                _ => None,
            }
        }
        fn has_parent(&self, node: LoopId) -> bool {
            node.0 != NodeId(3)
        }
    }

    /// A core data stage stating one extent per named dim and padding for `Y` alone.
    fn a_core_stage(extents: &[(PrimaryDim, i64)]) -> FilledDims {
        FilledDims::of(StageDims {
            extents: extents
                .iter()
                .map(|(dim, extent)| (*dim, Extent(*extent)))
                .collect(),
            padding: [(PrimaryDim::Y, DimPadding::default())]
                .into_iter()
                .collect(),
            ..StageDims::default()
        })
        .expect("a stage stating at least one dim")
    }

    /// A DSC whose ONE labelled DS sits at position 0 while RECORDING `183`, with an `In`-then-`Y`
    /// layout order for it and a core data stage that pads `Y`.
    fn a_stage(extents: &[(PrimaryDim, i64)]) -> DataStage {
        let named = NamedDims {
            name: StageName::default(),
            dims: a_core_stage(extents),
        };
        DataStage {
            ss: named.clone(),
            el: named,
        }
    }

    fn a_dsc() -> DesignSpaceConfig {
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: None,
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(Core::checked(0).expect("core 0"), vec![]),
            layout_dims: [(
                LdsIdx(0),
                LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Y]),
            )]
            .into_iter()
            .collect(),
            data_stages: DataStages::new(
                a_stage(&[(PrimaryDim::Y, 16), (PrimaryDim::Ij, 4)]),
                a_stage(&[(PrimaryDim::Y, 16)]),
            ),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(
                    DsType::Input,
                    vec![],
                    LdsIdx(183),
                    Pinning {
                        mem_org: [(SenComponent::Hbm, true)].into(),
                        lx: false,
                        lx_padded: true,
                    },
                ),
                vec![],
            ),
        }
    }

    /// Entry 009: a dim the stick names answers its cumulative extent, and a dim it does not name
    /// answers ONE element rather than nothing.
    #[test]
    fn a_dim_the_stick_does_not_name_is_one_element() {
        let mut dsc = a_dsc();
        dsc.primary_ds_info.insert(
            DsType::Input,
            PrimaryDsInfo {
                layout: LayoutDims::new(PrimaryDim::In, vec![PrimaryDim::Y]),
                stick: StickDims(vec![
                    (PrimaryDim::In, Elements(64)),
                    (PrimaryDim::X, Elements(2)),
                    (PrimaryDim::In, Elements(4)),
                ]),
            },
        );

        // The stick names `In` twice, and the cumulative size is the PRODUCT.
        assert_eq!(
            stick_size(&dsc, DsType::Input, PrimaryDim::In),
            Some(Elements(256))
        );
        assert_eq!(
            stick_size(&dsc, DsType::Input, PrimaryDim::Y),
            Some(Elements(1))
        );
        // A DS type the DSC describes no stick for is `primaryDsInfo_.at(dsType)`'s throw.
        assert_eq!(stick_size(&dsc, DsType::Kernel, PrimaryDim::In), None);
    }

    /// Entry 010: a dim whose core extent differs in any DSC is core-split, and `IJ` is skipped even
    /// when it differs.
    #[test]
    fn a_differing_core_extent_is_a_split_dim_and_ij_is_skipped() {
        let same = a_dsc();
        let mut differs = a_dsc();
        differs.data_stages.set(
            DATA_STAGE_CORE,
            a_stage(&[(PrimaryDim::Y, 8), (PrimaryDim::Ij, 9)]),
        );
        let sdsc = SuperDsc::new(
            DscList::new(same.clone(), vec![same, differs]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );

        assert_eq!(
            core_split_dimensions(&sdsc),
            [PrimaryDim::Y].into_iter().collect::<BTreeSet<_>>()
        );
    }

    /// Entries 011, 012 and 013: the DS-type walk APPENDS POSITIONS to what the caller's vector
    /// already held, while the two index sets collect the `ldsIdx_` each entry RECORDS — which for
    /// the position-0 entry is `183` and not `0`.
    #[test]
    fn the_positions_and_the_recorded_indices_are_two_answers_over_one_list() {
        let mut dsc = a_dsc();
        dsc.labeled_ds = LabeledDsList::new(
            dsc.labeled_ds.front().clone(),
            vec![LabeledDs::new(
                DsType::Kernel,
                vec![],
                LdsIdx(7),
                Pinning::default(),
            )],
        );
        let mut positions = vec![LdsIdx(9)];

        labeled_ds_with_ds_type(&dsc, DsType::Input, &mut positions);

        assert_eq!(positions, vec![LdsIdx(9), LdsIdx(0)]);
        assert_eq!(
            all_labeled_ds_indices(&dsc),
            [LdsIdx(183), LdsIdx(7)].into_iter().collect()
        );
        // Only the entry at position 0 is HBM-pinned, and it records 183.
        assert_eq!(
            hbm_pinned_labeled_ds_indices(&dsc),
            [LdsIdx(183)].into_iter().collect::<BTreeSet<_>>()
        );
    }

    /// Entry 014: an LX-pinned input whose own schedule step also names a data DSC is a neighbour
    /// fetch; the same step for another DSC, a non-input and a non-LX-pinned entry are not; and a
    /// DSC index past the end of `dscs_` is that `.at()`'s throw.
    #[test]
    fn an_lx_pinned_input_with_a_data_dsc_in_its_step_is_a_neighbour_fetch() {
        let core = Core::checked(0).expect("core 0");
        let lx_input = LabeledDs::new(
            DsType::Input,
            vec![],
            LdsIdx(0),
            Pinning {
                mem_org: BTreeMap::new(),
                lx: true,
                lx_padded: false,
            },
        );
        let dsc = a_dsc();
        let sdsc = SuperDsc::new(
            DscList::new(dsc.clone(), vec![dsc.clone(), dsc]),
            BTreeMap::new(),
            BTreeMap::new(),
            [(
                core,
                vec![
                    DscScheduleStep {
                        data_dsc: None,
                        dl_dsc: Some(DscIdx(1)),
                    },
                    DscScheduleStep {
                        data_dsc: Some(DscIdx(4)),
                        dl_dsc: Some(DscIdx(2)),
                    },
                ],
            )]
            .into_iter()
            .collect(),
        );

        assert_eq!(
            is_labeled_ds_lx_neighbor(&sdsc, DscIdx(2), &lx_input),
            Some(true)
        );
        assert_eq!(
            is_labeled_ds_lx_neighbor(&sdsc, DscIdx(1), &lx_input),
            Some(false)
        );
        // Not LX-pinned, and an LX-pinned output: both answer before any lookup happens.
        assert_eq!(
            is_labeled_ds_lx_neighbor(
                &sdsc,
                DscIdx(2),
                &LabeledDs::new(DsType::Input, vec![], LdsIdx(0), Pinning::default())
            ),
            Some(false)
        );
        assert_eq!(
            is_labeled_ds_lx_neighbor(
                &sdsc,
                DscIdx(2),
                &LabeledDs::new(
                    DsType::Output,
                    vec![],
                    LdsIdx(0),
                    Pinning {
                        mem_org: BTreeMap::new(),
                        lx: true,
                        lx_padded: false,
                    }
                )
            ),
            Some(false)
        );
        // A DSC index past the end of `dscs_`.
        assert_eq!(is_labeled_ds_lx_neighbor(&sdsc, DscIdx(3), &lx_input), None);
    }

    /// Entry 015: the enclosing loops innermost first, with the ROOT loop left out — and nothing at
    /// all for a node the walk finds no loop above.
    #[test]
    fn the_parent_loops_exclude_the_root_and_run_innermost_first() {
        assert_eq!(
            parent_loop_nodes(&TestTree, NodeId(7)),
            vec![LoopId(NodeId(5))]
        );
        assert_eq!(parent_loop_nodes(&TestTree, NodeId(1)), Vec::new());
    }

    /// Entry 016: an HBM-pinned LX allocation is registered under its labelled DS index, the padded
    /// layout dim its core stage states is marked and the other is not, and the same index a second
    /// time — or a repeated layout dim — yields nothing.
    #[test]
    fn an_hbm_pinned_lx_allocation_registers_once_and_pads_only_its_stated_dim() {
        let dsc = a_dsc();
        let mut metadata: BTreeMap<DscIdx, DscMetadata> =
            [(DscIdx(0), DscMetadata::default())].into_iter().collect();
        let fresh = FreshL3Allocation::of(&dsc, LdsIdx(0), SenComponent::Lx)
            .expect("a stated labelled DS with distinct layout dims");

        let node = create_allocate_node(
            &dsc,
            &mut metadata,
            fresh,
            Buffering::Double,
            NodeName("allocate_lds0_lx".to_owned()),
            DscIdx(0),
            AllocId(9),
        )
        .expect("a metadata entry and a free registry slot");

        assert_eq!(
            node,
            L3AllocateNode {
                name: NodeName("allocate_lds0_lx".to_owned()),
                lds: LdsIdx(0),
                component: SenComponent::Lx,
                buffering: Buffering::Double,
                layout: node.layout.clone(),
                padding: node.padding.clone(),
                indirect: None,
                related_indirect: None,
            }
        );
        assert_eq!(
            node.padding.padding(PrimaryDim::Y),
            PadType::PaddedFullSpanWUnneeded
        );
        assert_eq!(node.padding.padding(PrimaryDim::In), PadType::NoPad);
        assert_eq!(
            node.layout.0,
            vec![(PrimaryDim::In, None), (PrimaryDim::Y, None)]
        );
        assert_eq!(
            metadata[&DscIdx(0)].new_allocations[&SenComponent::Lx].lds_idx_and_alloc_node
                [&LdsIdx(0)],
            AllocId(9)
        );

        // The second registration of one labelled DS is the `allocMetadata.find(ldsIdx)` `DT_CHECK`.
        let again = FreshL3Allocation::of(&dsc, LdsIdx(0), SenComponent::Lx)
            .expect("a stated labelled DS with distinct layout dims");
        assert_eq!(
            create_allocate_node(
                &dsc,
                &mut metadata,
                again,
                Buffering::None,
                NodeName("allocate_lds0_lx".to_owned()),
                DscIdx(0),
                AllocId(10),
            ),
            None
        );

        // A repeated layout dim has no witness at all, and neither has a labelled DS the DSC does
        // not state a layout order for.
        let mut repeated = a_dsc();
        repeated.layout_dims.insert(
            LdsIdx(0),
            LayoutDims::new(PrimaryDim::Y, vec![PrimaryDim::Y]),
        );
        assert_eq!(
            FreshL3Allocation::of(&repeated, LdsIdx(0), SenComponent::Lx),
            None
        );
        assert_eq!(
            FreshL3Allocation::of(&dsc, LdsIdx(9), SenComponent::Lx),
            None
        );

        // A DSC the metadata registry states nothing for is `dscMetadata.at(dscIdx)`'s throw.
        let orphan = FreshL3Allocation::of(&dsc, LdsIdx(0), SenComponent::Lx)
            .expect("a stated labelled DS with distinct layout dims");
        assert_eq!(
            create_allocate_node(
                &dsc,
                &mut metadata,
                orphan,
                Buffering::Streaming,
                NodeName("allocate_lds0_lx".to_owned()),
                DscIdx(4),
                AllocId(11),
            ),
            None
        );
    }
}

/// Replaces: e017_createTransferNode
///
/// MINTS THE TRANSFER NODE from one source end to one or more destinations, each carrying its unit,
/// its storage and its lds index (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:596`).
///
/// ⛔ *"Destination unit and storage numbers do not match."* IS GONE BY THE ARGUMENT TYPE: one
/// destination is one [`Via`], so the three vectors cannot disagree — all seven callsites (`:3202`
/// through `:7069`) already pass them equal; every other field keeps the fresh node's default.
#[must_use]
pub fn create_transfer_node(src: Via, dst: Via, more_dsts: &[Via], name: NodeName) -> TransferNode {
    TransferNode {
        padding: TransferPadding::default(),
        src_indirect: None,
        dst_indirect: None,
        name,
        src: src.operand(),
        dsts: Dsts::new(
            dst.operand(),
            more_dsts.iter().map(|via| via.operand()).collect(),
        ),
        replication_factor: ReplicationFactor::ONE,
        unit_time_transfer_chunk_size: Vec::new(),
    }
}

/// WHICH WINDOW EXTENTS A DATA STAGE'S DIMS STATE — the `ki_ > 0` / `kj_ > 0` reads on a
/// `DataStructDims` (`dsc/dims.h:158`), which is all `createLoopNode` asks of one.
pub trait WindowExtents {
    /// The dims the stage states a positive window extent for.
    fn window_dims(&self) -> BTreeSet<PrimaryDim>;
}

/// THE CORE DATA STAGE'S WINDOW DIMS, PROVED PRESENT.
///
/// ⛔ *"Expect valid core data stage."* IS THIS TYPE'S ABSENCE:
/// `dataStageParam_.at(dataStageCoreIdx)` is where `createLoopNode` reads `ss_.ki_` and `ss_.kj_`,
/// so a DSC without that stage yields no witness rather than a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreWindowDims(BTreeSet<PrimaryDim>);

impl CoreWindowDims {
    /// `dataStageCoreIdx` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:275`).
    pub const CORE: DatastageId = DatastageId(0);

    /// The witness, or [`None`] where the reference refuses.
    #[must_use]
    pub fn of<D: WindowExtents>(stages: &DataStages<D>) -> Option<Self> {
        Some(Self(stages.0.get(&Self::CORE)?.ss.dims.window_dims()))
    }

    /// Whether the core stage states a window extent for this dim.
    #[must_use]
    pub fn windows(&self, dim: PrimaryDim) -> bool {
        self.0.contains(&dim)
    }
}

/// Replaces: e018_createLoopNode
///
/// MINTS THE LOOP NODE over `dim` and `more_dims` for one numerator/denominator data-stage pair,
/// marking `ki` and `kj` as window dims where the core data stage windows them
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:623`).
///
/// ⚠️ TRAP: the window test reads the CORE stage's `ss_`, not the numerator's or denominator's, and
/// it fires for `ki`/`kj` ONLY — every other dim takes `Unpadded` (`dsc/dims.h:76`).
#[must_use]
pub fn create_loop_node(
    core: &CoreWindowDims,
    dim: PrimaryDim,
    more_dims: &[PrimaryDim],
    num: DatastageId,
    den: DatastageId,
    name: NodeName,
) -> LoopNode {
    let of = |dim: PrimaryDim| PrimaryDimAndKind {
        dim,
        kind: if matches!(dim, PrimaryDim::Ki | PrimaryDim::Kj) && core.windows(dim) {
            MetaDimKind::WindowDim
        } else {
            MetaDimKind::Unpadded
        },
    };
    LoopNode {
        name,
        num,
        den,
        dims: LoopDims::new(of(dim), more_dims.iter().copied().map(of).collect()),
    }
}

/// Replaces: e019_createBlockNode
///
/// MINTS THE BLOCK NODE that names one level of the schedule tree; `new dsc2::BlockNode()` leaves
/// its child vector empty (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:645`).
#[must_use]
pub fn create_block_node(name: NodeName) -> BlockNode {
    BlockNode {
        name,
        children: Vec::new(),
    }
}

/// Replaces: e020_createSyncNode
///
/// MINTS THE SYNC NODE that signals all-to-all between `units`, on the given end and strength
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:652`).
///
/// ⭐ THE REFERENCE'S TWO `if`s ARE NOT A CHOICE: it writes `isReceive_`/`isSoft_` only when set,
/// and the field it skips already holds the same `false`.
#[must_use]
pub fn create_sync_node(
    units: SyncUnits,
    name: NodeName,
    direction: SyncDirection,
    strength: SyncStrength,
) -> SyncNode {
    SyncNode {
        name,
        units,
        direction,
        strength,
        implicit_sync_ref_transfer: None,
        other_ends: Vec::new(),
    }
}

/// Replaces: e021_getOpFuncName
///
/// THE DSC'S FIRST COMPUTE OP'S `opFuncName` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:665`).
///
/// ⛔ `DT_CHECK(hasComputeOp(dsc))` — `!computeOp_.empty()` (`L3DlOpsScheduler.h:285`) — IS
/// [`OpFuncs`](crate::schedule::ddc::v1::OpFuncs)' OWN NON-EMPTINESS, and `OpFuncs::NONE` is the
/// [`None`].
#[must_use]
pub fn get_op_func_name<D: ComputeOps + ?Sized>(dsc: &D) -> Option<OpFunc> {
    dsc.op_funcs().first()
}

/// A DATA STAGE'S TWO HALVES, BOTH PROVED TO STATE EXTENTS.
///
/// ⛔ *"Expect non-empty data-stage parameters."* IS THIS TYPE: `DataStructDims::empty()`
/// (`dsc/dims.cpp:112`) is equality with a default-constructed one, so a half that states nothing
/// has no witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatedStage<D> {
    ss: StageDims<D>,
    el: StageDims<D>,
}

impl<D: Default + PartialEq> StatedStage<D> {
    /// The witness, or [`None`] where the reference refuses.
    #[must_use]
    pub fn of(ss: D, ss_name: StageName, el: D, el_name: StageName) -> Option<Self> {
        (ss != D::default() && el != D::default()).then(move || Self {
            ss: StageDims {
                name: ss_name,
                dims: ss,
            },
            el: StageDims {
                name: el_name,
                dims: el,
            },
        })
    }
}

/// Replaces: e022_addOrUpdateDataStageParam
///
/// WRITES THE DSC'S DATA STAGE at `index` — both halves and both names — adding the entry where it
/// is not there yet (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:721`).
///
/// ⭐ ONE INSERT IS THE WHOLE OF IT: `dsc2::DataStage` (`dsc/dsc2.h:40`) is exactly `ss_` and `el_`
/// and the reference overwrites both, so the emplace-if-absent it does first is not observable.
pub fn add_or_update_data_stage_param<D>(
    stages: &mut DataStages<D>,
    stage: StatedStage<D>,
    index: DatastageId,
) {
    let StatedStage { ss, el } = stage;
    stages.0.insert(index, DataStage { ss, el });
}

/// Replaces: e023_isOpFuncConv2dInt4
///
/// WHETHER THE OP FUNC IS ONE OF THE THREE INT4 CONV2Ds
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:739`).
///
/// ⭐ THE THREE ARE EVERY `CONV2D_INT4_*` THE ISA NAMES (`sys-arch-spec/arch_enums.h:207`, `:211`,
/// `:215`), so this is int4-ness and not a subset of it.
#[must_use]
pub fn is_op_func_conv2d_int4(op_func: Option<OpFunc>) -> bool {
    matches!(
        op_func,
        Some(OpFunc::Conv2DInt4Fwd | OpFunc::Conv2DInt4FwdGenkg3 | OpFunc::Conv2DInt4FwdSparsekg3)
    )
}

/// Replaces: e024_isOpFuncConv2dOs1
///
/// WHETHER THE OP FUNC IS ONE OF THE FOUR OUTPUT-STATIONARY CONV2Ds
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:746`).
///
/// ⭐ THE FOUR ARE EVERY `*_OS1` THE ISA NAMES (`sys-arch-spec/arch_enums.h:241-244`) — there is no
/// fp8 output-stationary form — so this is output-stationariness and not a subset of it.
#[must_use]
pub fn is_op_func_conv2d_os1(op_func: Option<OpFunc>) -> bool {
    matches!(
        op_func,
        Some(
            OpFunc::Conv2DFwdOs1
                | OpFunc::Conv2DXrfInt8FwdOs1
                | OpFunc::Conv2DFwdGenOs1
                | OpFunc::Conv2DInt8FwdOs1
        )
    )
}

/// Replaces: e025_isOpFuncBmmInt4
///
/// The int4-weight batch matmuls — plain, sparse-KG3 and both XRF spellings.
#[must_use]
pub const fn is_op_func_bmm_int4(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::BatchmatmulInt4Fwd
            | OpFunc::BatchmatmulInt4FwdSparsekg3
            | OpFunc::BatchmatmulXrfInt4Fwd
            | OpFunc::BatchmatmulXrfchInt4Fwd
    )
}

/// Replaces: e026_isOpFuncBmmInt8
///
/// The int8-weight batch matmuls, five of them; `_MBKG3` is int8's only multi-batch spelling.
#[must_use]
pub const fn is_op_func_bmm_int8(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::BatchmatmulInt8Fwd
            | OpFunc::BatchmatmulInt8FwdMbkg3
            | OpFunc::BatchmatmulInt8FwdSparsekg3
            | OpFunc::BatchmatmulXrfInt8Fwd
            | OpFunc::BatchmatmulXrfchInt8Fwd
    )
}

/// Replaces: e027_isOpFuncBmmFp8NonXrf
///
/// The fp8 batch matmuls that are not XRF-spelled — plain, `_MB` and sparse-KG3.
///
/// ⛔ TRAP: `_MB` HERE, `_MBKG3` IN [`is_op_func_bmm_int8`] — the two multi-batch spellings are
/// different ops, and `getMinParamBmm` reaches this family and int8 through ONE arm (`:965`).
#[must_use]
pub const fn is_op_func_bmm_fp8_non_xrf(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::BatchmatmulFp8Fwd
            | OpFunc::BatchmatmulFp8FwdMb
            | OpFunc::BatchmatmulFp8FwdSparsekg3
    )
}

/// Replaces: e028_isOpFuncBmmFp8Xrf
///
/// The two XRF-spelled fp8 batch matmuls, `XRF` and `XRFCH`.
#[must_use]
pub const fn is_op_func_bmm_fp8_xrf(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::BatchmatmulXrfFp8Fwd | OpFunc::BatchmatmulXrfchFp8Fwd
    )
}

/// Replaces: e029_isOpFuncBmmFp16
///
/// The four batch matmuls whose names carry no format token at all.
///
/// ⛔ TRAP: THE NAME SAYS FP16 AND NO MEMBER SAYS ANYTHING — `BATCHMATMUL_FWD` with its sparse-KG3
/// and XRF spellings is the DEFAULT-format op, so this family is "the format the op does not name",
/// not a stated fp16.
#[must_use]
pub const fn is_op_func_bmm_fp16(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::BatchmatmulFwd
            | OpFunc::BatchmatmulFwdSparsekg3
            | OpFunc::BatchmatmulXrfFwd
            | OpFunc::BatchmatmulXrfchFwd
    )
}

/// Replaces: e030_isOpFuncScalarBroadcast
///
/// Seven activations plus `ADD`, `STRIDED_ADD`, `MUL`, `SUB`, `REVSUB`, `BIASADD`, `BATCHNORM_FWD`.
///
/// ⛔ TRAP: THIS IS NOT "ELEMENTWISE". `REALDIV`, `MAXIMUM`, `MINIMUM`, `FNMS`, `WHERE3` and every
/// other unary transcendental (`EXP_FWD`, `SILU_FWD`, `SQRT_FWD`, `RSQRT`, …) have the same operand
/// shape and are ABSENT, so `getMinParamForDimFromOpFunc` (`:1137`) hands them its default 1.
#[must_use]
pub const fn is_op_func_scalar_broadcast(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::ReluFwd
            | OpFunc::Relu6Fwd
            | OpFunc::LeakyreluFwd
            | OpFunc::GeluFwd
            | OpFunc::TanhFwd
            | OpFunc::SigmoidFwd
            | OpFunc::FastSigmoidFwd
            | OpFunc::Add
            | OpFunc::StridedAdd
            | OpFunc::Mul
            | OpFunc::Sub
            | OpFunc::Revsub
            | OpFunc::Biasadd
            | OpFunc::BatchnormFwd
    )
}

/// Replaces: e031_isOpFuncReduction
///
/// `SUM`, `MAX`, `MEAN`, `EXX2` and four of the non-stick reductions.
///
/// ⛔ TRAP: SIX REDUCTIONS THE SET DOES NOT CLAIM — `ABSMAX`, `MIN`, `ABSMAX_NONSTICK`,
/// `MIN_NONSTICK`, `EXX2_ZEROMEAN` and `GENERIC_PARTIAL_REDUCTION` all reduce and all answer
/// `false`, so they take the default min param instead of `getMinParamReduction`.
#[must_use]
pub const fn is_op_func_reduction(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::Sum
            | OpFunc::Max
            | OpFunc::Mean
            | OpFunc::Exx2
            | OpFunc::SumNonstick
            | OpFunc::MaxNonstick
            | OpFunc::MeanNonstick
            | OpFunc::ProdNonstick
    )
}

/// Replaces: e032_isOpFuncPooling
///
/// `MAXPOOL_FWD`, `AVGPOOL_FWD` and `AVGPOOL_NMAP_FWD` — every pooling op the sealed set spells.
#[must_use]
pub const fn is_op_func_pooling(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::MaxpoolFwd | OpFunc::AvgpoolFwd | OpFunc::AvgpoolNmapFwd
    )
}

#[cfg(test)]
mod tests_e025_e032 {
    use super::*;

    /// The five BMM predicates, in the order `isOpFuncBmm` (`:804`) ORs them.
    const BMM_FAMILIES: [fn(OpFunc) -> bool; 5] = [
        is_op_func_bmm_fp16,
        is_op_func_bmm_fp8_xrf,
        is_op_func_bmm_fp8_non_xrf,
        is_op_func_bmm_int4,
        is_op_func_bmm_int8,
    ];

    fn claiming(op_func: OpFunc) -> usize {
        BMM_FAMILIES.iter().filter(|is_bmm| is_bmm(op_func)).count()
    }

    /// e025 — the four int4 batch matmuls, and the identically spelled int8 sibling is not one.
    #[test]
    fn bmm_int4_is_the_four_int4_batchmatmuls() {
        assert!(
            [
                OpFunc::BatchmatmulInt4Fwd,
                OpFunc::BatchmatmulInt4FwdSparsekg3,
                OpFunc::BatchmatmulXrfInt4Fwd,
                OpFunc::BatchmatmulXrfchInt4Fwd,
            ]
            .into_iter()
            .all(is_op_func_bmm_int4)
        );
        assert!(!is_op_func_bmm_int4(OpFunc::BatchmatmulXrfInt8Fwd));
    }

    /// e026 — the five int8 batch matmuls, and `MATMUL_INT8_FWD` is not a BATCH matmul.
    #[test]
    fn bmm_int8_is_the_five_int8_batchmatmuls() {
        assert!(
            [
                OpFunc::BatchmatmulInt8Fwd,
                OpFunc::BatchmatmulInt8FwdMbkg3,
                OpFunc::BatchmatmulInt8FwdSparsekg3,
                OpFunc::BatchmatmulXrfInt8Fwd,
                OpFunc::BatchmatmulXrfchInt8Fwd,
            ]
            .into_iter()
            .all(is_op_func_bmm_int8)
        );
        assert!(!is_op_func_bmm_int8(OpFunc::MatmulInt8Fwd));
    }

    /// e027 — the three non-XRF fp8 batch matmuls, and the XRF ones belong to e028 instead.
    #[test]
    fn bmm_fp8_non_xrf_excludes_the_xrf_spellings() {
        assert!(
            [
                OpFunc::BatchmatmulFp8Fwd,
                OpFunc::BatchmatmulFp8FwdMb,
                OpFunc::BatchmatmulFp8FwdSparsekg3,
            ]
            .into_iter()
            .all(is_op_func_bmm_fp8_non_xrf)
        );
        assert!(!is_op_func_bmm_fp8_non_xrf(OpFunc::BatchmatmulXrfFp8Fwd));
        assert!(!is_op_func_bmm_fp8_non_xrf(OpFunc::BatchmatmulXrfchFp8Fwd));
    }

    /// e028 — the two XRF fp8 batch matmuls, and the plain fp8 one is not among them.
    #[test]
    fn bmm_fp8_xrf_is_the_two_xrf_spellings() {
        assert!(is_op_func_bmm_fp8_xrf(OpFunc::BatchmatmulXrfFp8Fwd));
        assert!(is_op_func_bmm_fp8_xrf(OpFunc::BatchmatmulXrfchFp8Fwd));
        assert!(!is_op_func_bmm_fp8_xrf(OpFunc::BatchmatmulFp8Fwd));
    }

    /// e029 — the four format-less batch matmuls; the two MX-format ones are NOT this family, even
    /// though nothing else claims them either.
    #[test]
    fn bmm_fp16_is_the_batchmatmuls_that_name_no_format() {
        assert!(
            [
                OpFunc::BatchmatmulFwd,
                OpFunc::BatchmatmulFwdSparsekg3,
                OpFunc::BatchmatmulXrfFwd,
                OpFunc::BatchmatmulXrfchFwd,
            ]
            .into_iter()
            .all(is_op_func_bmm_fp16)
        );
        assert!(!is_op_func_bmm_fp16(OpFunc::BatchmatmulMxfp8Fwd));
        assert!(!is_op_func_bmm_fp16(OpFunc::BatchmatmulMxfp4WFwd));
    }

    /// ⭐ THE CENSUS THE FIVE FAMILIES OWE THE SEALED SET: they are pairwise disjoint, they claim 18
    /// of its 21 batch matmuls, and `BATCHMATMULV2`, `BATCHMATMUL_MXFP4W_FWD` and
    /// `BATCHMATMUL_MXFP8_FWD` are claimed by NONE — so `isOpFuncBmm` is false for all three and
    /// `getMinParamForDimFromOpFunc` (`:1169`) gives them its default min param of 1.
    #[test]
    fn the_five_bmm_families_partition_eighteen_of_twenty_one_batchmatmuls() {
        let spelled_bmm = OpFunc::ALL
            .into_iter()
            .filter(|op_func| op_func.spelling().starts_with("batchmatmul"));
        assert_eq!(spelled_bmm.clone().count(), 21);
        assert!(
            OpFunc::ALL
                .into_iter()
                .all(|op_func| claiming(op_func) <= 1)
        );
        assert_eq!(
            spelled_bmm
                .clone()
                .filter(|op_func| claiming(*op_func) == 1)
                .count(),
            18
        );
        assert_eq!(
            spelled_bmm
                .filter(|op_func| claiming(*op_func) == 0)
                .collect::<Vec<_>>(),
            vec![
                OpFunc::Batchmatmulv2,
                OpFunc::BatchmatmulMxfp4WFwd,
                OpFunc::BatchmatmulMxfp8Fwd
            ]
        );
    }

    /// e030 — the fourteen scalar/broadcast ops, and `REALDIV` shares `SUB`'s shape without sharing
    /// its arm.
    #[test]
    fn scalar_broadcast_is_fourteen_ops_and_not_every_elementwise_one() {
        assert_eq!(
            OpFunc::ALL
                .into_iter()
                .filter(|op_func| is_op_func_scalar_broadcast(*op_func))
                .count(),
            14
        );
        assert!(is_op_func_scalar_broadcast(OpFunc::Sub));
        assert!(is_op_func_scalar_broadcast(OpFunc::Revsub));
        assert!(!is_op_func_scalar_broadcast(OpFunc::Realdiv));
        assert!(!is_op_func_scalar_broadcast(OpFunc::SiluFwd));
    }

    /// e031 — the eight it claims, and the six reductions it does not.
    #[test]
    fn reduction_leaves_six_reductions_unclaimed() {
        assert!(
            [
                OpFunc::Sum,
                OpFunc::Max,
                OpFunc::Mean,
                OpFunc::Exx2,
                OpFunc::SumNonstick,
                OpFunc::MaxNonstick,
                OpFunc::MeanNonstick,
                OpFunc::ProdNonstick,
            ]
            .into_iter()
            .all(is_op_func_reduction)
        );
        assert!(
            ![
                OpFunc::Absmax,
                OpFunc::Min,
                OpFunc::AbsmaxNonstick,
                OpFunc::MinNonstick,
                OpFunc::Exx2Zeromean,
                OpFunc::GenericPartialReduction,
            ]
            .into_iter()
            .any(is_op_func_reduction)
        );
    }

    /// e032 — the three pooling ops, and no op the sealed set spells with `pool` is left out.
    #[test]
    fn pooling_is_every_pool_op_in_the_sealed_set() {
        assert!(is_op_func_pooling(OpFunc::MaxpoolFwd));
        assert!(is_op_func_pooling(OpFunc::AvgpoolFwd));
        assert!(is_op_func_pooling(OpFunc::AvgpoolNmapFwd));
        assert_eq!(
            OpFunc::ALL
                .into_iter()
                .filter(|op_func| op_func.spelling().contains("pool"))
                .filter(|op_func| !is_op_func_pooling(*op_func))
                .collect::<Vec<_>>(),
            Vec::new()
        );
    }
}

/// Replaces: e033_isOpFuncDepthwiseConv
///
/// Whether the op is the depthwise convolution — a set of exactly one.
#[must_use]
pub const fn is_op_func_depthwise_conv(op_func: OpFunc) -> bool {
    matches!(op_func, OpFunc::DepthwiseConvFwd)
}

/// Replaces: e034_isOpFuncQuantization
///
/// Whether the op quantizes — the twelve `Q_FP8`/`CSQ_INT8`/`CSQ_INT4` spellings the scheduler lists.
///
/// ⛔ THE SET IS NOT EVERY QUANTIZER THE MACHINE HAS: `Q_FP8_MB`, `CSQ_INT8_V2` and `CSQ_INT8_MB_V2`
/// are members of `OpFuncs` and ABSENT from this set, so `getMinParamForDimFromOpFunc` gives them the
/// default minimum of 1 rather than the quantizer's 64 on `IN`/`OUT`.
#[must_use]
pub const fn is_op_func_quantization(op_func: OpFunc) -> bool {
    matches!(
        op_func,
        OpFunc::QFp8
            | OpFunc::QFp8Ch
            | OpFunc::QFp8Chil
            | OpFunc::QFp8Wt
            | OpFunc::CsqInt8
            | OpFunc::CsqInt8Ch
            | OpFunc::CsqInt8Wt
            | OpFunc::CsqInt8Chil
            | OpFunc::CsqInt8Mb
            | OpFunc::CsqInt4
            | OpFunc::CsqInt4Wt
            | OpFunc::CsqInt4Chil
    )
}

/// Replaces: e035_isOpFuncConversionDl16AndFp32
///
/// Whether the op converts between DL16 and FP32, in either direction.
///
/// ⛔ `FP8TODL16` AND `DL16TOBF16` ARE CONVERSIONS TOO AND ARE NOT IN THE SET, so the stick-size
/// minimum of [`min_param_conversion_dl16_and_fp32`] is never taken for them.
#[must_use]
pub const fn is_op_func_conversion_dl16_and_fp32(op_func: OpFunc) -> bool {
    matches!(op_func, OpFunc::Dl16Tofp32 | OpFunc::Fp32Todl16)
}

/// `constexpr long defaultParam = 1` — the minimum every min-param unit falls back to, and what a
/// chunk of one element along a dim means.
const DEFAULT_MIN_PARAM: Extent = Extent(1);

/// `stickSizes.count(dim) ? stickSizes.at(dim) : defaultParam` — the lookup both stick-size arms make.
///
/// ⛔ `None` IS A WIDTH THAT DOES NOT FIT THE REFERENCE'S `int`: the answer becomes a data stage's
/// extent, and an extent that cannot be counted is no minimum at all.
fn stick_size_or_default(sizes: &[(PrimaryDim, Elements)], dim: PrimaryDim) -> Option<Extent> {
    match sizes.iter().find(|(named, _)| *named == dim) {
        Some((_, size)) => i64::try_from(size.0).ok().map(Extent),
        None => Some(DEFAULT_MIN_PARAM),
    }
}

/// Replaces: e036_getMinParamScalarBroadcast
///
/// The smallest chunk a scalar or broadcast op may take of `dim`: the core stage's own extent on `J`,
/// otherwise the LAST labelled data structure's cumulative stick size, or 1 where the stick does not
/// name the dim.
///
/// ⛔ `None` IS THE REFERENCE'S `-1` AND ITS `.at()` THROW AT ONCE — a `J` the core stage does not
/// state becomes a NEGATIVE minimum parameter at `primaryDimToValHandler_st(dim) = ...` (`:1397`),
/// and `primaryDsInfo_.at(labeledDs_.back().dsType_)` throws for a structure with no stick.
#[must_use]
pub fn min_param_scalar_broadcast(dsc: &DesignSpaceConfig, dim: PrimaryDim) -> Option<Extent> {
    match dim {
        PrimaryDim::J => dsc.core_stage().dims().extent(dim),
        _ => {
            let sizes = dsc.cumulative_stick_sizes(dsc.labeled_ds.back().ds_type())?;
            stick_size_or_default(&sizes, dim)
        }
    }
}

/// Replaces: e037_getMinParamReduction
///
/// The smallest chunk a reduction may take of `dim`: the core stage's own extent on `J` — the
/// reduction axis, which cannot be split — and one everywhere else.
///
/// ⛔ `None` IS THE REFERENCE'S `-1` ON THE `J` ARM, as in [`min_param_scalar_broadcast`].
#[must_use]
pub fn min_param_reduction(dsc: &DesignSpaceConfig, dim: PrimaryDim) -> Option<Extent> {
    match dim {
        PrimaryDim::J => dsc.core_stage().dims().extent(dim),
        _ => Some(DEFAULT_MIN_PARAM),
    }
}

/// Replaces: e038_getMinParamPoolingAndDepthwiseConv
///
/// Pooling and depthwise-conv minima: `OUT` is 64 whatever the stage says, `J`/`KI`/`KJ` are the core
/// stage's extent — and so is `IN`, but for depthwise conv only — and everything else is one.
///
/// ⛔ THE 64 IS NOT READ FROM THE ARCH, and it is the one arm that cannot come back absent: the other
/// three carriers hand back the reference's `-1` when the core stage does not state the dim.
#[must_use]
pub fn min_param_pooling_and_depthwise_conv(
    dsc: &DesignSpaceConfig,
    dim: PrimaryDim,
    op_func: OpFunc,
) -> Option<Extent> {
    let core_param = dsc.core_stage().dims().extent(dim);
    match dim {
        PrimaryDim::In if is_op_func_depthwise_conv(op_func) => core_param,
        PrimaryDim::Out => Some(Extent(64)),
        PrimaryDim::J | PrimaryDim::Ki | PrimaryDim::Kj => core_param,
        _ => Some(DEFAULT_MIN_PARAM),
    }
}

/// Replaces: e039_getMinParamQuantization
///
/// Quantizer minima: a dim that carries padding must be taken WHOLE, `J` is always the core stage's
/// extent, and an unpadded `IN`/`OUT` is 128 for `CSQ_INT4`/`CSQ_INT4_CHIL` and 64 for the rest.
///
/// ⛔ `CSQ_INT4_WT` IS INT4 AND TAKES THE 64, not the 128 — the test names two spellings, not a
/// width. ⛔ AND `hasPadding` IS COMPUTED BEFORE THE SWITCH, so [`StageDims::has_padding`]'s aborts
/// reach every dim, `J` included, and the compound `IJ`/`KIJ` that `calculate_padded` refuses
/// outright takes the whole unit down with it whenever it carries a `paddingSizes_` entry.
#[must_use]
pub fn min_param_quantization(
    dsc: &DesignSpaceConfig,
    dim: PrimaryDim,
    op_func: OpFunc,
) -> Option<Extent> {
    let core = dsc.core_stage().dims();
    let has_padding = core.has_padding(dim)?;
    match dim {
        PrimaryDim::J => core.extent(dim),
        PrimaryDim::In | PrimaryDim::Out if has_padding => core.extent(dim),
        PrimaryDim::In | PrimaryDim::Out => {
            if matches!(op_func, OpFunc::CsqInt4 | OpFunc::CsqInt4Chil) {
                Some(Extent(128))
            } else {
                Some(Extent(64))
            }
        }
        _ if has_padding => core.extent(dim),
        _ => Some(DEFAULT_MIN_PARAM),
    }
}

/// Replaces: e040_getMinParamConversionDl16AndFp32
///
/// DL16↔FP32 conversion minima: the core stage's extent on `J`, otherwise the cumulative stick size
/// of the FIRST labelled data structure for `DL16TOFP32` and of the LAST for `FP32TODL16`.
///
/// ⛔ FRONT/BACK IS POSITIONAL, NOT INPUT/OUTPUT BY NAME — the arm picks `labeledDs_.front()` or
/// `.back()`, so which structure it lands on is the DSC's list order and not a `DsTypes` test. An
/// `opFuncName` that is neither direction takes the `.back()` arm, so this is safe to ask of any op.
#[must_use]
pub fn min_param_conversion_dl16_and_fp32(
    dsc: &DesignSpaceConfig,
    dim: PrimaryDim,
    op_func: OpFunc,
) -> Option<Extent> {
    match dim {
        PrimaryDim::J => dsc.core_stage().dims().extent(dim),
        _ => {
            let lds = if matches!(op_func, OpFunc::Dl16Tofp32) {
                dsc.labeled_ds.front()
            } else {
                dsc.labeled_ds.back()
            };
            let sizes = dsc.cumulative_stick_sizes(lds.ds_type())?;
            stick_size_or_default(&sizes, dim)
        }
    }
}

#[cfg(test)]
mod tests_e033_e040 {
    use super::*;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims;
    use crate::schedule::dsc2::LayoutDims;
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DimPadding, LabeledDsList, NamedDims,
        PadElems, PadSizes, PrimaryDsInfo, StageDims,
    };

    /// A core data stage stating exactly the given extents.
    fn stage(extents: &[(PrimaryDim, i64)]) -> FilledDims {
        let mut dims = StageDims::default();
        for &(dim, extent) in extents {
            dims.extents.insert(dim, Extent(extent));
        }
        FilledDims::of(dims).expect("a stage that states a dim")
    }

    /// A primary data structure whose stick names the dims given, in that order.
    fn stick(dims: &[(PrimaryDim, u64)]) -> PrimaryDsInfo {
        let (first, _) = *dims.first().expect("a stick with a dim in it");
        PrimaryDsInfo {
            layout: LayoutDims::new(first, dims[1..].iter().map(|(dim, _)| *dim).collect()),
            stick: StickDims(dims.iter().map(|&(dim, e)| (dim, Elements(e))).collect()),
        }
    }

    fn config(core_stage: FilledDims, labeled: &[DsType]) -> DesignSpaceConfig {
        let (first, rest) = labeled.split_first().expect("a DSC labels a structure");
        let named = NamedDims {
            name: StageName::default(),
            dims: core_stage,
        };
        let stage = DataStage {
            ss: named.clone(),
            el: named,
        };
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: None,
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(Core::checked(0).expect("core 0"), vec![]),
            layout_dims: BTreeMap::new(),
            data_stages: DataStages::new(stage.clone(), stage),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(*first, vec![], LdsIdx(183), Pinning::default()),
                rest.iter()
                    .map(|ds| LabeledDs::new(*ds, vec![], LdsIdx(183), Pinning::default()))
                    .collect(),
            ),
        }
    }

    /// e033 — one member, and the pooling op next to it in `isOpFuncStridedWindow` is not it.
    #[test]
    fn depthwise_conv_is_a_set_of_one() {
        assert!(is_op_func_depthwise_conv(OpFunc::DepthwiseConvFwd));
        assert!(!is_op_func_depthwise_conv(OpFunc::MaxpoolFwd));
    }

    /// e034 — the twelve the reference lists, and the three quantizers it leaves out.
    #[test]
    fn quantization_omits_three_quantizers() {
        for op_func in [
            OpFunc::QFp8,
            OpFunc::QFp8Ch,
            OpFunc::QFp8Chil,
            OpFunc::QFp8Wt,
            OpFunc::CsqInt8,
            OpFunc::CsqInt8Ch,
            OpFunc::CsqInt8Wt,
            OpFunc::CsqInt8Chil,
            OpFunc::CsqInt8Mb,
            OpFunc::CsqInt4,
            OpFunc::CsqInt4Wt,
            OpFunc::CsqInt4Chil,
        ] {
            assert!(is_op_func_quantization(op_func), "{op_func:?}");
        }
        for op_func in [OpFunc::QFp8Mb, OpFunc::CsqInt8V2, OpFunc::CsqInt8MbV2] {
            assert!(!is_op_func_quantization(op_func), "{op_func:?}");
        }
    }

    /// e035 — both directions, and the two conversions that are not in the set.
    #[test]
    fn conversion_is_dl16_against_fp32_only() {
        assert!(is_op_func_conversion_dl16_and_fp32(OpFunc::Dl16Tofp32));
        assert!(is_op_func_conversion_dl16_and_fp32(OpFunc::Fp32Todl16));
        assert!(!is_op_func_conversion_dl16_and_fp32(OpFunc::Fp8Todl16));
        assert!(!is_op_func_conversion_dl16_and_fp32(OpFunc::Dl16Tobf16));
    }

    /// e036 — `J` is the core extent, another dim is the LAST structure's cumulative stick size, a dim
    /// the stick does not name is one, and an unstated `J` is the reference's `-1`.
    #[test]
    fn scalar_broadcast_takes_the_last_structures_stick() {
        let mut dsc = config(
            stage(&[(PrimaryDim::J, 12), (PrimaryDim::In, 96)]),
            &[DsType::Input, DsType::Output],
        );
        dsc.primary_ds_info.insert(
            DsType::Output,
            stick(&[
                (PrimaryDim::In, 4),
                (PrimaryDim::Out, 8),
                (PrimaryDim::In, 2),
            ]),
        );
        assert_eq!(
            min_param_scalar_broadcast(&dsc, PrimaryDim::J),
            Some(Extent(12))
        );
        // The stick names IN twice, so its cumulative size is the PRODUCT.
        assert_eq!(
            min_param_scalar_broadcast(&dsc, PrimaryDim::In),
            Some(Extent(8))
        );
        assert_eq!(
            min_param_scalar_broadcast(&dsc, PrimaryDim::Mb),
            Some(DEFAULT_MIN_PARAM)
        );

        let unstated = config(stage(&[(PrimaryDim::In, 96)]), &[DsType::Output]);
        assert_eq!(min_param_scalar_broadcast(&unstated, PrimaryDim::J), None);
    }

    /// e037 — `J` is the core extent and nothing else is anything but one.
    #[test]
    fn reduction_pins_only_the_j_axis() {
        let dsc = config(
            stage(&[(PrimaryDim::J, 12), (PrimaryDim::In, 96)]),
            &[DsType::Output],
        );
        assert_eq!(min_param_reduction(&dsc, PrimaryDim::J), Some(Extent(12)));
        assert_eq!(
            min_param_reduction(&dsc, PrimaryDim::In),
            Some(DEFAULT_MIN_PARAM)
        );
        assert_eq!(
            min_param_reduction(&dsc, PrimaryDim::Out),
            Some(DEFAULT_MIN_PARAM)
        );
    }

    /// e038 — `IN` follows the core extent for depthwise conv and drops to one for pooling, `OUT` is
    /// always 64, the kernel axes follow the core, and `MB` is one.
    #[test]
    fn pooling_pins_out_at_sixty_four() {
        let dsc = config(
            stage(&[
                (PrimaryDim::In, 96),
                (PrimaryDim::Out, 96),
                (PrimaryDim::J, 12),
                (PrimaryDim::Ki, 3),
            ]),
            &[DsType::Output],
        );
        let min = |dim, op_func| min_param_pooling_and_depthwise_conv(&dsc, dim, op_func);
        assert_eq!(
            min(PrimaryDim::In, OpFunc::DepthwiseConvFwd),
            Some(Extent(96))
        );
        assert_eq!(
            min(PrimaryDim::In, OpFunc::MaxpoolFwd),
            Some(DEFAULT_MIN_PARAM)
        );
        assert_eq!(min(PrimaryDim::Out, OpFunc::MaxpoolFwd), Some(Extent(64)));
        assert_eq!(min(PrimaryDim::J, OpFunc::MaxpoolFwd), Some(Extent(12)));
        assert_eq!(min(PrimaryDim::Ki, OpFunc::MaxpoolFwd), Some(Extent(3)));
        assert_eq!(
            min(PrimaryDim::Mb, OpFunc::MaxpoolFwd),
            Some(DEFAULT_MIN_PARAM)
        );
    }

    /// e039 — an unpadded `IN` is 64, 128 for the two INT4 spellings and 64 for `CSQ_INT4_WT`; padding
    /// on a dim takes it whole; and a padded compound dim is `calculate_padded`'s abort.
    #[test]
    fn quantization_takes_a_padded_dim_whole() {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::In, Extent(96));
        dims.extents.insert(PrimaryDim::Out, Extent(80));
        dims.extents.insert(PrimaryDim::J, Extent(12));
        dims.extents.insert(PrimaryDim::Mb, Extent(4));
        dims.padding.insert(
            PrimaryDim::Out,
            DimPadding {
                sizes: PadSizes::of(PadElems(1), PadElems(2)),
                ..DimPadding::default()
            },
        );
        let dsc = config(
            FilledDims::of(dims.clone()).expect("a stage that states a dim"),
            &[DsType::Output],
        );
        let min = |dim, op_func| min_param_quantization(&dsc, dim, op_func);
        assert_eq!(min(PrimaryDim::In, OpFunc::CsqInt8), Some(Extent(64)));
        assert_eq!(min(PrimaryDim::In, OpFunc::CsqInt4), Some(Extent(128)));
        assert_eq!(min(PrimaryDim::In, OpFunc::CsqInt4Chil), Some(Extent(128)));
        assert_eq!(min(PrimaryDim::In, OpFunc::CsqInt4Wt), Some(Extent(64)));
        // OUT carries padding, so the quantizer must take the whole of it.
        assert_eq!(min(PrimaryDim::Out, OpFunc::CsqInt4), Some(Extent(80)));
        assert_eq!(min(PrimaryDim::J, OpFunc::CsqInt8), Some(Extent(12)));
        assert_eq!(
            min(PrimaryDim::Mb, OpFunc::CsqInt8),
            Some(DEFAULT_MIN_PARAM)
        );

        // A `paddingSizes_` entry on the compound IJ is the "Cannot calculate padded version of
        // compound dim" abort, and it is reached before the switch picks an arm.
        dims.extents.insert(PrimaryDim::Ij, Extent(6));
        dims.padding.insert(
            PrimaryDim::Ij,
            DimPadding {
                sizes: PadSizes::of(PadElems(1), PadElems(0)),
                ..DimPadding::default()
            },
        );
        let compound = config(
            FilledDims::of(dims).expect("a stage that states a dim"),
            &[DsType::Output],
        );
        assert_eq!(
            min_param_quantization(&compound, PrimaryDim::Ij, OpFunc::CsqInt8),
            None
        );
    }

    /// e040 — `DL16TOFP32` reads the FIRST structure's stick and `FP32TODL16` the LAST, and `J` is the
    /// core extent for both.
    #[test]
    fn conversion_picks_its_structure_by_position() {
        let mut dsc = config(
            stage(&[(PrimaryDim::J, 12), (PrimaryDim::In, 96)]),
            &[DsType::Input, DsType::Output],
        );
        dsc.primary_ds_info
            .insert(DsType::Input, stick(&[(PrimaryDim::In, 32)]));
        dsc.primary_ds_info
            .insert(DsType::Output, stick(&[(PrimaryDim::In, 8)]));
        assert_eq!(
            min_param_conversion_dl16_and_fp32(&dsc, PrimaryDim::In, OpFunc::Dl16Tofp32),
            Some(Extent(32))
        );
        assert_eq!(
            min_param_conversion_dl16_and_fp32(&dsc, PrimaryDim::In, OpFunc::Fp32Todl16),
            Some(Extent(8))
        );
        assert_eq!(
            min_param_conversion_dl16_and_fp32(&dsc, PrimaryDim::J, OpFunc::Dl16Tofp32),
            Some(Extent(12))
        );
    }
}

/// A BURST SIZE, `1..=l3BurstSize` — `DT_CHECK_MSG(.., "Invalid Burst size.")` as a constructor,
/// holding the zero-based row the efficiency table is read by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BurstSize(Bounded<{ Target::L3_BURST }>);

impl BurstSize {
    /// A burst of `sticks` sticks, `None` outside `1..=l3BurstSize`.
    #[must_use]
    pub const fn new(sticks: u32) -> Option<Self> {
        match sticks.checked_sub(1) {
            Some(row) => match Bounded::checked(row) {
                Some(row) => Some(Self(row)),
                None => None,
            },
            None => None,
        }
    }

    /// `burstSize - 1`, the row it names.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.get()
    }
}

/// A MULTICAST DEGREE, `1..=numCores` — `DT_CHECK_MSG(.., "Invalid multicast degree.")` as a
/// constructor, holding the zero-based column the efficiency table is read by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MulticastCores(Bounded<{ Target::CORES }>);

impl MulticastCores {
    /// A degree, `None` outside `1..=numCores`.
    #[must_use]
    pub const fn of(degree: MulticastDegree) -> Option<Self> {
        match degree.0.checked_sub(1) {
            Some(column) => match Bounded::checked(column) {
                Some(column) => Some(Self(column)),
                None => None,
            },
            None => None,
        }
    }

    /// `multicastDegree - 1`, the column it names.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.get()
    }
}

/// HOW EFFICIENT ONE BURST IS IN THE DATA RING — an entry of `burstEfficiency`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct BurstEfficiency(pub f64);

/// `burstEfficiency` (`L3DlOpsScheduler.cpp:278`) — `dcg/dcg_fe/scheduler/BurstEfficiency.def`
/// TRANSCRIBED, rows indexed by burst size and columns by multicast degree, both from one.
///
/// ⛔ `rustfmt::skip` SO ONE ROW STAYS ONE LINE, as the `.def` file writes it — a reflowed table
/// cannot be diffed against its source.
#[rustfmt::skip]
const BURST_EFFICIENCY: [[f64; 32]; 32] = [
    [0.1000, 0.0995, 0.0990, 0.0985, 0.0980, 0.0975, 0.0970, 0.0965, 0.0960, 0.0955, 0.0950, 0.0945, 0.0940, 0.0935, 0.0930, 0.0925, 0.0920, 0.0915, 0.0910, 0.0905, 0.0900, 0.0895, 0.0890, 0.0885, 0.0880, 0.0875, 0.0870, 0.0865, 0.0860, 0.0855, 0.0850, 0.0845],
    [0.1250, 0.1245, 0.1240, 0.1235, 0.1230, 0.1225, 0.1220, 0.1215, 0.1210, 0.1205, 0.1200, 0.1195, 0.1190, 0.1185, 0.1180, 0.1175, 0.1170, 0.1165, 0.1160, 0.1155, 0.1150, 0.1145, 0.1140, 0.1135, 0.1130, 0.1125, 0.1120, 0.1115, 0.1110, 0.1105, 0.1100, 0.1095],
    [0.1500, 0.1495, 0.1490, 0.1485, 0.1480, 0.1475, 0.1470, 0.1465, 0.1460, 0.1455, 0.1450, 0.1445, 0.1440, 0.1435, 0.1430, 0.1425, 0.1420, 0.1415, 0.1410, 0.1405, 0.1400, 0.1395, 0.1390, 0.1385, 0.1380, 0.1375, 0.1370, 0.1365, 0.1360, 0.1355, 0.1350, 0.1345],
    [0.1750, 0.1745, 0.1740, 0.1735, 0.1730, 0.1725, 0.1720, 0.1715, 0.1710, 0.1705, 0.1700, 0.1695, 0.1690, 0.1685, 0.1680, 0.1675, 0.1670, 0.1665, 0.1660, 0.1655, 0.1650, 0.1645, 0.1640, 0.1635, 0.1630, 0.1625, 0.1620, 0.1615, 0.1610, 0.1605, 0.1600, 0.1595],
    [0.2000, 0.1995, 0.1990, 0.1985, 0.1980, 0.1975, 0.1970, 0.1965, 0.1960, 0.1955, 0.1950, 0.1945, 0.1940, 0.1935, 0.1930, 0.1925, 0.1920, 0.1915, 0.1910, 0.1905, 0.1900, 0.1895, 0.1890, 0.1885, 0.1880, 0.1875, 0.1870, 0.1865, 0.1860, 0.1855, 0.1850, 0.1845],
    [0.2250, 0.2245, 0.2240, 0.2235, 0.2230, 0.2225, 0.2220, 0.2215, 0.2210, 0.2205, 0.2200, 0.2195, 0.2190, 0.2185, 0.2180, 0.2175, 0.2170, 0.2165, 0.2160, 0.2155, 0.2150, 0.2145, 0.2140, 0.2135, 0.2130, 0.2125, 0.2120, 0.2115, 0.2110, 0.2105, 0.2100, 0.2095],
    [0.2500, 0.2495, 0.2490, 0.2485, 0.2480, 0.2475, 0.2470, 0.2465, 0.2460, 0.2455, 0.2450, 0.2445, 0.2440, 0.2435, 0.2430, 0.2425, 0.2420, 0.2415, 0.2410, 0.2405, 0.2400, 0.2395, 0.2390, 0.2385, 0.2380, 0.2375, 0.2370, 0.2365, 0.2360, 0.2355, 0.2350, 0.2345],
    [0.2750, 0.2745, 0.2740, 0.2735, 0.2730, 0.2725, 0.2720, 0.2715, 0.2710, 0.2705, 0.2700, 0.2695, 0.2690, 0.2685, 0.2680, 0.2675, 0.2670, 0.2665, 0.2660, 0.2655, 0.2650, 0.2645, 0.2640, 0.2635, 0.2630, 0.2625, 0.2620, 0.2615, 0.2610, 0.2605, 0.2600, 0.2595],
    [0.3000, 0.2995, 0.2990, 0.2985, 0.2980, 0.2975, 0.2970, 0.2965, 0.2960, 0.2955, 0.2950, 0.2945, 0.2940, 0.2935, 0.2930, 0.2925, 0.2920, 0.2915, 0.2910, 0.2905, 0.2900, 0.2895, 0.2890, 0.2885, 0.2880, 0.2875, 0.2870, 0.2865, 0.2860, 0.2855, 0.2850, 0.2845],
    [0.3250, 0.3245, 0.3240, 0.3235, 0.3230, 0.3225, 0.3220, 0.3215, 0.3210, 0.3205, 0.3200, 0.3195, 0.3190, 0.3185, 0.3180, 0.3175, 0.3170, 0.3165, 0.3160, 0.3155, 0.3150, 0.3145, 0.3140, 0.3135, 0.3130, 0.3125, 0.3120, 0.3115, 0.3110, 0.3105, 0.3100, 0.3095],
    [0.3500, 0.3495, 0.3490, 0.3485, 0.3480, 0.3475, 0.3470, 0.3465, 0.3460, 0.3455, 0.3450, 0.3445, 0.3440, 0.3435, 0.3430, 0.3425, 0.3420, 0.3415, 0.3410, 0.3405, 0.3400, 0.3395, 0.3390, 0.3385, 0.3380, 0.3375, 0.3370, 0.3365, 0.3360, 0.3355, 0.3350, 0.3345],
    [0.3750, 0.3745, 0.3740, 0.3735, 0.3730, 0.3725, 0.3720, 0.3715, 0.3710, 0.3705, 0.3700, 0.3695, 0.3690, 0.3685, 0.3680, 0.3675, 0.3670, 0.3665, 0.3660, 0.3655, 0.3650, 0.3645, 0.3640, 0.3635, 0.3630, 0.3625, 0.3620, 0.3615, 0.3610, 0.3605, 0.3600, 0.3595],
    [0.4000, 0.3995, 0.3990, 0.3985, 0.3980, 0.3975, 0.3970, 0.3965, 0.3960, 0.3955, 0.3950, 0.3945, 0.3940, 0.3935, 0.3930, 0.3925, 0.3920, 0.3915, 0.3910, 0.3905, 0.3900, 0.3895, 0.3890, 0.3885, 0.3880, 0.3875, 0.3870, 0.3865, 0.3860, 0.3855, 0.3850, 0.3845],
    [0.4250, 0.4245, 0.4240, 0.4235, 0.4230, 0.4225, 0.4220, 0.4215, 0.4210, 0.4205, 0.4200, 0.4195, 0.4190, 0.4185, 0.4180, 0.4175, 0.4170, 0.4165, 0.4160, 0.4155, 0.4150, 0.4145, 0.4140, 0.4135, 0.4130, 0.4125, 0.4120, 0.4115, 0.4110, 0.4105, 0.4100, 0.4095],
    [0.4500, 0.4495, 0.4490, 0.4485, 0.4480, 0.4475, 0.4470, 0.4465, 0.4460, 0.4455, 0.4450, 0.4445, 0.4440, 0.4435, 0.4430, 0.4425, 0.4420, 0.4415, 0.4410, 0.4405, 0.4400, 0.4395, 0.4390, 0.4385, 0.4380, 0.4375, 0.4370, 0.4365, 0.4360, 0.4355, 0.4350, 0.4345],
    [0.4750, 0.4745, 0.4740, 0.4735, 0.4730, 0.4725, 0.4720, 0.4715, 0.4710, 0.4705, 0.4700, 0.4695, 0.4690, 0.4685, 0.4680, 0.4675, 0.4670, 0.4665, 0.4660, 0.4655, 0.4650, 0.4645, 0.4640, 0.4635, 0.4630, 0.4625, 0.4620, 0.4615, 0.4610, 0.4605, 0.4600, 0.4595],
    [0.5000, 0.4995, 0.4990, 0.4985, 0.4980, 0.4975, 0.4970, 0.4965, 0.4960, 0.4955, 0.4950, 0.4945, 0.4940, 0.4935, 0.4930, 0.4925, 0.4920, 0.4915, 0.4910, 0.4905, 0.4900, 0.4895, 0.4890, 0.4885, 0.4880, 0.4875, 0.4870, 0.4865, 0.4860, 0.4855, 0.4850, 0.4845],
    [0.5250, 0.5245, 0.5240, 0.5235, 0.5230, 0.5225, 0.5220, 0.5215, 0.5210, 0.5205, 0.5200, 0.5195, 0.5190, 0.5185, 0.5180, 0.5175, 0.5170, 0.5165, 0.5160, 0.5155, 0.5150, 0.5145, 0.5140, 0.5135, 0.5130, 0.5125, 0.5120, 0.5115, 0.5110, 0.5105, 0.5100, 0.5095],
    [0.5500, 0.5495, 0.5490, 0.5485, 0.5480, 0.5475, 0.5470, 0.5465, 0.5460, 0.5455, 0.5450, 0.5445, 0.5440, 0.5435, 0.5430, 0.5425, 0.5420, 0.5415, 0.5410, 0.5405, 0.5400, 0.5395, 0.5390, 0.5385, 0.5380, 0.5375, 0.5370, 0.5365, 0.5360, 0.5355, 0.5350, 0.5345],
    [0.5750, 0.5745, 0.5740, 0.5735, 0.5730, 0.5725, 0.5720, 0.5715, 0.5710, 0.5705, 0.5700, 0.5695, 0.5690, 0.5685, 0.5680, 0.5675, 0.5670, 0.5665, 0.5660, 0.5655, 0.5650, 0.5645, 0.5640, 0.5635, 0.5630, 0.5625, 0.5620, 0.5615, 0.5610, 0.5605, 0.5600, 0.5595],
    [0.6000, 0.5995, 0.5990, 0.5985, 0.5980, 0.5975, 0.5970, 0.5965, 0.5960, 0.5955, 0.5950, 0.5945, 0.5940, 0.5935, 0.5930, 0.5925, 0.5920, 0.5915, 0.5910, 0.5905, 0.5900, 0.5895, 0.5890, 0.5885, 0.5880, 0.5875, 0.5870, 0.5865, 0.5860, 0.5855, 0.5850, 0.5845],
    [0.6250, 0.6245, 0.6240, 0.6235, 0.6230, 0.6225, 0.6220, 0.6215, 0.6210, 0.6205, 0.6200, 0.6195, 0.6190, 0.6185, 0.6180, 0.6175, 0.6170, 0.6165, 0.6160, 0.6155, 0.6150, 0.6145, 0.6140, 0.6135, 0.6130, 0.6125, 0.6120, 0.6115, 0.6110, 0.6105, 0.6100, 0.6095],
    [0.6500, 0.6495, 0.6490, 0.6485, 0.6480, 0.6475, 0.6470, 0.6465, 0.6460, 0.6455, 0.6450, 0.6445, 0.6440, 0.6435, 0.6430, 0.6425, 0.6420, 0.6415, 0.6410, 0.6405, 0.6400, 0.6395, 0.6390, 0.6385, 0.6380, 0.6375, 0.6370, 0.6365, 0.6360, 0.6355, 0.6350, 0.6345],
    [0.6750, 0.6745, 0.6740, 0.6735, 0.6730, 0.6725, 0.6720, 0.6715, 0.6710, 0.6705, 0.6700, 0.6695, 0.6690, 0.6685, 0.6680, 0.6675, 0.6670, 0.6665, 0.6660, 0.6655, 0.6650, 0.6645, 0.6640, 0.6635, 0.6630, 0.6625, 0.6620, 0.6615, 0.6610, 0.6605, 0.6600, 0.6595],
    [0.7000, 0.6995, 0.6990, 0.6985, 0.6980, 0.6975, 0.6970, 0.6965, 0.6960, 0.6955, 0.6950, 0.6945, 0.6940, 0.6935, 0.6930, 0.6925, 0.6920, 0.6915, 0.6910, 0.6905, 0.6900, 0.6895, 0.6890, 0.6885, 0.6880, 0.6875, 0.6870, 0.6865, 0.6860, 0.6855, 0.6850, 0.6845],
    [0.7250, 0.7245, 0.7240, 0.7235, 0.7230, 0.7225, 0.7220, 0.7215, 0.7210, 0.7205, 0.7200, 0.7195, 0.7190, 0.7185, 0.7180, 0.7175, 0.7170, 0.7165, 0.7160, 0.7155, 0.7150, 0.7145, 0.7140, 0.7135, 0.7130, 0.7125, 0.7120, 0.7115, 0.7110, 0.7105, 0.7100, 0.7095],
    [0.7500, 0.7495, 0.7490, 0.7485, 0.7480, 0.7475, 0.7470, 0.7465, 0.7460, 0.7455, 0.7450, 0.7445, 0.7440, 0.7435, 0.7430, 0.7425, 0.7420, 0.7415, 0.7410, 0.7405, 0.7400, 0.7395, 0.7390, 0.7385, 0.7380, 0.7375, 0.7370, 0.7365, 0.7360, 0.7355, 0.7350, 0.7345],
    [0.7750, 0.7745, 0.7740, 0.7735, 0.7730, 0.7725, 0.7720, 0.7715, 0.7710, 0.7705, 0.7700, 0.7695, 0.7690, 0.7685, 0.7680, 0.7675, 0.7670, 0.7665, 0.7660, 0.7655, 0.7650, 0.7645, 0.7640, 0.7635, 0.7630, 0.7625, 0.7620, 0.7615, 0.7610, 0.7605, 0.7600, 0.7595],
    [0.8000, 0.7995, 0.7990, 0.7985, 0.7980, 0.7975, 0.7970, 0.7965, 0.7960, 0.7955, 0.7950, 0.7945, 0.7940, 0.7935, 0.7930, 0.7925, 0.7920, 0.7915, 0.7910, 0.7905, 0.7900, 0.7895, 0.7890, 0.7885, 0.7880, 0.7875, 0.7870, 0.7865, 0.7860, 0.7855, 0.7850, 0.7845],
    [0.8250, 0.8245, 0.8240, 0.8235, 0.8230, 0.8225, 0.8220, 0.8215, 0.8210, 0.8205, 0.8200, 0.8195, 0.8190, 0.8185, 0.8180, 0.8175, 0.8170, 0.8165, 0.8160, 0.8155, 0.8150, 0.8145, 0.8140, 0.8135, 0.8130, 0.8125, 0.8120, 0.8115, 0.8110, 0.8105, 0.8100, 0.8095],
    [0.8500, 0.8495, 0.8490, 0.8485, 0.8480, 0.8475, 0.8470, 0.8465, 0.8460, 0.8455, 0.8450, 0.8445, 0.8440, 0.8435, 0.8430, 0.8425, 0.8420, 0.8415, 0.8410, 0.8405, 0.8400, 0.8395, 0.8390, 0.8385, 0.8380, 0.8375, 0.8370, 0.8365, 0.8360, 0.8355, 0.8350, 0.8345],
    [0.8750, 0.8745, 0.8740, 0.8735, 0.8730, 0.8725, 0.8720, 0.8715, 0.8710, 0.8705, 0.8700, 0.8695, 0.8690, 0.8685, 0.8680, 0.8675, 0.8670, 0.8665, 0.8660, 0.8655, 0.8650, 0.8645, 0.8640, 0.8635, 0.8630, 0.8625, 0.8620, 0.8615, 0.8610, 0.8605, 0.8600, 0.8595],
];

/// Replaces: e041_getChunkParamsFromCandidates
///
/// WRITES the chunk extent the search selected for each dim into `params`, then recomputes the
/// compound dims that depend on them.
///
/// ⛔ "Index is out of range." AND BOTH `.at(dim)` LOOKUPS ARE [`SelectedCandidate`]'s: the candidate
/// list, the index chosen into it and the `primaryDims` narrowing are ONE value before this is
/// called, so the walk has nothing left to check.
pub fn chunk_params_from_candidates(params: &mut FilledDims, candidates: &DscParamCandidates) {
    for (dim, selected) in &candidates.0 {
        params.set_extent(*dim, selected.extent());
    }
    params.compound();
}

/// Replaces: e042_getBurstEfficiency
///
/// The transfer efficiency the heuristic table states for a burst size and a multicast degree.
///
/// ⛔ THE TABLE IS TRANSCRIBED, NOT DERIVED: its 1024 entries do follow `0.075 + 0.025·burst -
/// 0.0005·(degree - 1)`, and fitting a rule to data is how a coefficient chain gets golden-hacked.
/// ⛔ TRAP: THE TWO GUARDS ARE NOT THE SAME SHAPE. The ROW COUNT must EQUAL `l3BurstSize` (`:1616`),
/// so an arch with a shorter burst ABORTS rather than reading a prefix; each ROW is only required to
/// be the literal `maxNumCores = 32` (`:1618`) while the degree is checked against `numCores`. Both
/// are that asymmetry turned into a build error.
#[must_use]
pub fn burst_efficiency(burst: BurstSize, multicast: MulticastCores) -> BurstEfficiency {
    const { assert!(Target::L3_BURST == 32, "the table states 32 burst sizes") }
    const { assert!(Target::CORES <= 32, "the table states 32 multicast degrees") }
    BurstEfficiency(BURST_EFFICIENCY[burst.index() as usize][multicast.index() as usize])
}

/// Replaces: e043_getLabeledDsNumOfStickVolumesInCore
///
/// The chunks a core is cut into along `lds`' non-broadcast dims, times the volumes one chunk holds.
///
/// ⛔ TWO DEAD PARAMETERS, BOTH THE REFERENCE'S: `primaryDims` is never read, and the `bytesPerStick`
/// it hands `getBufferCapacityForNode` is read only under `forceEvenNumSticks`, which defaults false.
/// ⛔ [`None`] IS EVERY REFUSAL HERE, and only some are the reference's `DT_CHECK`s: no LX capacity
/// stated for `lds` (`:1705-1709`), a capacity that is not whole sticks, a stick volume that does
/// not divide the chunk, and `getNonBroadcastLdsDims`' own abort.
/// ⛔ DIVERGENCE, NOT A `DT_CHECK`: the reference's `-1` extents make an UNSTATED dim a factor of `1`
/// (`-1 / -1`) or a NEGATIVE chunk count (`8 / -1`); the sign guards below refuse instead.
#[must_use]
pub fn labeled_ds_num_of_stick_volumes_in_core(
    dsc: &DesignSpaceConfig,
    lds: LdsIdx,
    stick_volume: StickVolume,
) -> Option<StickVolumes> {
    let capacity = dsc.lx_chunk_capacity.get(&lds)?.0;
    let bytes_per_stick = Target::BYTES_PER_STICK.get();
    if capacity % bytes_per_stick != 0 {
        return None;
    }
    let chunk_sticks = capacity / bytes_per_stick;
    if chunk_sticks % stick_volume.get() != 0 {
        return None;
    }
    let mut chunks: u64 = 1;
    for dim in dsc.non_broadcast_lds_dims(lds)? {
        let core = dsc.data_stages.core().ss_extent(dim)?.0;
        let chunk = dsc.data_stages.chunk().ss_extent(dim)?.0;
        if core < 0 || chunk <= 0 || core % chunk != 0 {
            return None;
        }
        chunks *= (core / chunk) as u64;
    }
    Some(StickVolumes(chunks * chunk_sticks / stick_volume.get()))
}

/// Replaces: e044_getOpReducedDimSet
///
/// THE DIMS THE OP REDUCES AWAY — non-broadcast on some input and on no output.
///
/// ⛔ TRAP: `mySDsc` IS NEVER READ, and `isOutputLabeledDs` is `ldsIdx == labeledDs_.size() - 1`
/// (`L3DlOpsScheduler.h:228-230`), so "the outputs" is the LAST entry and only ever that one.
/// ⛔ TRAP: BOTH QUESTIONS ARE ASKED OF [`LabeledDs::recorded`], the entry's OWN `ldsIdx_`, and not
/// of the position it sits at — so a recorded index that drifted answers for another position.
#[must_use]
pub fn op_reduced_dim_set(dsc: &DesignSpaceConfig) -> Option<BTreeSet<PrimaryDim>> {
    let mut inputs: BTreeSet<PrimaryDim> = BTreeSet::new();
    let mut outputs: BTreeSet<PrimaryDim> = BTreeSet::new();
    for entry in dsc.labeled_ds.iter() {
        let dims = dsc.non_broadcast_lds_dims(entry.recorded())?;
        if dsc.labeled_ds.is_output(entry.recorded()) {
            outputs.extend(dims);
        } else {
            inputs.extend(dims);
        }
    }
    Some(inputs.difference(&outputs).copied().collect())
}

/// Replaces: e045_addSuperChunkDataStage
///
/// COPIES the chunk data stage into the superchunk stage, renamed `"superchunk"` on both halves.
///
/// ⛔ BOTH `DT_CHECK`s ARE DISCHARGED BEFORE THE CALL: [`SuperChunkStage`] witnesses that the index
/// names an entry `getNewDataStageIndex` already inserted, and `DataStages` holds the chunk stage as
/// a field rather than a map entry that might be missing.
pub fn add_super_chunk_data_stage(dsc: &mut DesignSpaceConfig, super_chunk: SuperChunkStage) {
    let mut stage = dsc.data_stages.chunk().clone();
    stage.rename(StageName::super_chunk());
    dsc.data_stages.set(super_chunk.index(), stage);
}

/// `lxBelowBlockNodeName` (`L3DlOpsScheduler.cpp:277`) — the block every LX-below schedule hangs from.
pub const LX_BELOW_BLOCK_NODE_NAME: &str = "lx_below_schedule";

/// Replaces: e046_getLxBelowBlockNode
///
/// THE `BLOCK` NODE NAMED `lx_below_schedule`, exclusively borrowed so its finder may edit it.
///
/// ⛔ TRAP: THE REFERENCE'S `nullptr` RETURN IS A LATENT CRASH IN ONE OF ITS FOUR CALLERS —
/// `L3DlOpsScheduler.cpp:2838` takes it and `:2845` dereferences `->getOwnerLoop()` with no null
/// check, while `:3129`, `:3611` and `:7599` each `DT_CHECK_MSG` it. Here that case is a [`None`]
/// the caller has to name.
pub fn lx_below_block_node(tree: &mut ScheduleTree) -> Option<&mut BlockNode> {
    tree.find_block_mut(|block| block.name.0 == LX_BELOW_BLOCK_NODE_NAME)
}

/// Replaces: e047_collectAllDimensionsForLoopOrder
///
/// EVERY LAYOUT DIM ONCE, in `labeledDs_` order with the indirect-access-index structures LAST.
///
/// ⛔ TRAP: THE REFERENCE'S OWN `DT_CHECK` IS A TAUTOLOGY — `(dim != IJ || dim != KIJ)` holds for
/// every dim there is, so the combined dims it claims to reject are collected like any other.
/// ⛔ TRAP: TWO INDICES IN ONE LOOP — `indirectAccessLabeledDs` holds POINTERS into `labeledDs_`, so
/// membership is by POSITION, while the index pushed is the entry's own [`LabeledDs::recorded`] one.
#[must_use]
pub fn collect_all_dimensions_for_loop_order(dsc: &DesignSpaceConfig) -> Option<Vec<PrimaryDim>> {
    let mut order: Vec<LdsIdx> = Vec::new();
    let mut low_priority: Vec<LdsIdx> = Vec::new();
    for (position, entry) in dsc.labeled_ds.indexed() {
        if dsc.indirect_access_index_lds.contains(&position) {
            low_priority.push(entry.recorded());
        } else {
            order.push(entry.recorded());
        }
    }
    order.append(&mut low_priority);
    let mut dims: Vec<PrimaryDim> = Vec::new();
    for lds in order {
        for dim in dsc.layout_dims.get(&lds)?.iter() {
            if !dims.contains(&dim) {
                dims.push(dim);
            }
        }
    }
    Some(dims)
}

/// HOW MANY CORES TAKE THE SAME WORK SLICES — `shares`, the first half of the pair entry 048 returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shares(pub u32);

/// A GTR SHARING GROUP'S ID — `gtr_->groupName_`, `0..=maxGroupID` because the field is SIX BITS
/// wide (`sysdef.cpp:230`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GtrGroupId(Bounded<{ Target::MAX_GROUP_ID + 1 }>);

impl GtrGroupId {
    /// A group id, `None` past `maxGroupID` — `DT_CHECK_MSG(gtrCurrGroupName <= maxGroupID,
    /// "gtr_->groupName_ exceeds the limit.")` as a constructor.
    #[must_use]
    pub const fn checked(id: u32) -> Option<Self> {
        match Bounded::checked(id) {
            Some(id) => Some(Self(id)),
            None => None,
        }
    }

    /// The id itself, for the GTR field that carries it.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// WHICH SHARING GROUP A LABELLED DATA STRUCTURE'S CORES FORM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupName {
    /// `maxGroupID + 1`, the default a lone core takes — ONE PAST every id a GTR can carry, so it is
    /// not a group id and [`GtrGroupId`] rightly cannot hold it.
    Unshared,
    /// The id a set of two or more sharing cores was given, and keeps for every later query.
    Shared(GtrGroupId),
}

/// THE GTR GROUP NAMES HANDED OUT SO FAR — `coresSetToGtrGroupNameMap` with `gtrCurrGroupName`
/// (`L3DlOpsScheduler.h:214-219`), ONE value because the counter is read only to name a set of cores
/// the map does not already hold.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GtrGroupNames {
    next: u32,
    named: BTreeMap<BTreeSet<Core>, GtrGroupId>,
}

impl GtrGroupNames {
    /// No group named yet, which is `gtrCurrGroupName = 0` and an empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The name this set of cores carries, minting and recording a fresh one where it has none.
    /// `None` is the 64 GTR groups exhausted.
    pub fn name_for(&mut self, cores: &BTreeSet<Core>) -> Option<GtrGroupId> {
        if let Some(name) = self.named.get(cores) {
            return Some(*name);
        }
        let name = GtrGroupId::checked(self.next)?;
        self.named.insert(cores.clone(), name);
        self.next += 1;
        Some(name)
    }
}

/// Replaces: e048_getSharesAndGroupName
///
/// The processing cores taking `curr_wk_slices` on every non-broadcast dim of `lds`, and their name.
///
/// ⛔ TRAP: `dsc` IS READ ONLY FOR THE DIM LIST, asked of `lds`'s OWN [`LabeledDs::recorded`] index,
/// and a name is minted per SET OF CORES — two unrelated transfers over the same cores share one.
/// ⛔ [`None`] IS ONE OF THREE `DT_CHECK`s: a negative work-slice id, no matching core at all, or the
/// GTR group ids exhausted.
pub fn shares_and_group_name(
    sdsc: &SuperDsc,
    dsc: &DesignSpaceConfig,
    lds: &LabeledDs,
    curr_wk_slices: &WkSlice,
    processing: &BTreeSet<Core>,
    names: &mut GtrGroupNames,
) -> Option<(Shares, GroupName)> {
    let dims = dsc.non_broadcast_lds_dims(lds.recorded())?;
    let mut sharing: BTreeSet<Core> = BTreeSet::new();
    for (core, slices) in &sdsc.core_id_to_wk_slice {
        if !processing.contains(core) {
            continue;
        }
        let mut matches = true;
        for &dim in &dims {
            let mine = curr_wk_slices.at(dim)?;
            let theirs = slices.at(dim)?;
            if mine.0 < 0 || theirs.0 < 0 {
                return None;
            }
            if mine != theirs {
                matches = false;
                break;
            }
        }
        if matches {
            sharing.insert(*core);
        }
    }
    if sharing.is_empty() {
        return None;
    }
    let shares = Shares(sharing.len() as u32);
    let name = if shares.0 > 1 {
        GroupName::Shared(names.name_for(&sharing)?)
    } else {
        GroupName::Unshared
    };
    Some((shares, name))
}

#[cfg(test)]
mod tests_e041_e048 {
    use super::*;
    use crate::arch::Bytes;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Extent;
    use crate::schedule::ddc::metadata::DatastageId;
    use crate::schedule::dsc2::{LayoutDims, NodeName, SchedNode};
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DscList, LabeledDsList, NamedDims,
        SelectedCandidate, StageDims, WkSliceId,
    };
    use std::num::NonZeroU64;

    fn dims(extents: &[(PrimaryDim, i64)]) -> FilledDims {
        let mut stage = StageDims::default();
        for (dim, extent) in extents {
            stage.extents.insert(*dim, Extent(*extent));
        }
        FilledDims::of(stage).expect("a stage that states a dim")
    }

    fn stage(name: &str, extents: &[(PrimaryDim, i64)]) -> DataStage {
        let name = StageName(name.to_owned());
        DataStage {
            ss: NamedDims {
                name: name.clone(),
                dims: dims(extents),
            },
            el: NamedDims {
                name,
                dims: dims(extents),
            },
        }
    }

    fn sized(recorded: LdsIdx, dims: &[PrimaryDim]) -> LabeledDs {
        LabeledDs::new(
            DsType::Input,
            dims.iter().map(|dim| (*dim, Scale::Sized(1.0))).collect(),
            recorded,
            Pinning::default(),
        )
    }

    fn dsc(core: &[(PrimaryDim, i64)], chunk: &[(PrimaryDim, i64)]) -> DesignSpaceConfig {
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: Some(CoreletsUsed::ONE),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(Core::checked(0).expect("core 0"), vec![]),
            layout_dims: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(sized(LdsIdx(0), &[]), vec![]),
            data_stages: DataStages::new(stage("core", core), stage("chunk", chunk)),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        }
    }

    /// e041 — the selected candidate is what each dim takes, and the compound dim follows from it.
    #[test]
    fn chunk_params_take_the_selected_candidate_then_compound() {
        let mut params = dims(&[(PrimaryDim::I, 1), (PrimaryDim::J, 1)]);
        let candidates = DscParamCandidates(BTreeMap::from([
            (
                PrimaryDim::I,
                SelectedCandidate::new(vec![Extent(2), Extent(4)], 1).expect("a chosen candidate"),
            ),
            (
                PrimaryDim::J,
                SelectedCandidate::new(vec![Extent(3)], 0).expect("a chosen candidate"),
            ),
        ]));
        chunk_params_from_candidates(&mut params, &candidates);
        assert_eq!(params.dims().extent(PrimaryDim::I), Some(Extent(4)));
        assert_eq!(params.dims().extent(PrimaryDim::J), Some(Extent(3)));
        assert_eq!(params.dims().extent(PrimaryDim::Ij), Some(Extent(12)));
        assert_eq!(SelectedCandidate::new(vec![Extent(2)], 1), None);
    }

    /// e042 — both corners of `BurstEfficiency.def`, and the two ranges its `DT_CHECK`s state.
    #[test]
    fn burst_efficiency_reads_the_table_corners() {
        let one = BurstSize::new(1).expect("a burst of one stick");
        let full = BurstSize::new(Target::L3_BURST).expect("a full burst");
        let solo = MulticastCores::of(MulticastDegree(1)).expect("one core");
        let every = MulticastCores::of(MulticastDegree(Target::CORES)).expect("every core");
        assert_eq!(
            burst_efficiency(one, solo).0.to_bits(),
            0.1000_f64.to_bits()
        );
        assert_eq!(
            burst_efficiency(full, every).0.to_bits(),
            0.8595_f64.to_bits()
        );
        assert_eq!(BurstSize::new(0), None);
        assert_eq!(BurstSize::new(Target::L3_BURST + 1), None);
        assert_eq!(MulticastCores::of(MulticastDegree(0)), None);
        assert_eq!(MulticastCores::of(MulticastDegree(Target::CORES + 1)), None);
    }

    /// e043 — four chunks of four sticks each, read two sticks at a time, is eight stick volumes.
    #[test]
    fn stick_volumes_in_core_multiply_the_chunks_by_the_volumes_per_chunk() {
        let mut dsc = dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 2)]);
        let lds = LdsIdx(0);
        dsc.labeled_ds = LabeledDsList::new(sized(lds, &[PrimaryDim::I]), vec![]);
        dsc.layout_dims
            .insert(lds, LayoutDims::new(PrimaryDim::I, vec![]));
        dsc.lx_chunk_capacity
            .insert(lds, Bytes(4 * Target::BYTES_PER_STICK.get()));
        let volume = StickVolume::new(NonZeroU64::new(2).expect("a positive volume"));
        assert_eq!(
            labeled_ds_num_of_stick_volumes_in_core(&dsc, lds, volume),
            Some(StickVolumes(8))
        );
        let odd = StickVolume::new(NonZeroU64::new(3).expect("a positive volume"));
        assert_eq!(
            labeled_ds_num_of_stick_volumes_in_core(&dsc, lds, odd),
            None
        );
    }

    /// e043/e044's seam — a wholly broadcast structure answers EMPTY before the layout is asked for,
    /// and a structure that does name a non-broadcast dim still needs one.
    #[test]
    fn a_wholly_broadcast_structure_answers_before_the_layout_is_needed() {
        let mut dsc = dsc(&[(PrimaryDim::I, 1)], &[(PrimaryDim::I, 1)]);
        let broadcast = LabeledDs::new(
            DsType::Input,
            vec![
                (PrimaryDim::I, Scale::Sized(0.0)),
                (PrimaryDim::J, Scale::UnitStick),
            ],
            LdsIdx(0),
            Pinning::default(),
        );
        dsc.labeled_ds = LabeledDsList::new(broadcast, vec![sized(LdsIdx(1), &[PrimaryDim::I])]);
        // Neither entry has a layout_dims entry, which is getLayoutDims' abort.
        assert_eq!(dsc.non_broadcast_lds_dims(LdsIdx(0)), Some(vec![]));
        assert_eq!(dsc.non_broadcast_lds_dims(LdsIdx(1)), None);
    }

    /// e044 — a dim the inputs carry and the output does not is the dim the op reduces away.
    #[test]
    fn op_reduced_dims_are_the_inputs_minus_the_output() {
        let mut dsc = dsc(&[(PrimaryDim::I, 1)], &[(PrimaryDim::I, 1)]);
        dsc.labeled_ds = LabeledDsList::new(
            sized(LdsIdx(0), &[PrimaryDim::I, PrimaryDim::Ki]),
            vec![sized(LdsIdx(1), &[PrimaryDim::I])],
        );
        dsc.layout_dims.insert(
            LdsIdx(0),
            LayoutDims::new(PrimaryDim::I, vec![PrimaryDim::Ki]),
        );
        dsc.layout_dims
            .insert(LdsIdx(1), LayoutDims::new(PrimaryDim::I, vec![]));
        assert_eq!(
            op_reduced_dim_set(&dsc),
            Some(BTreeSet::from([PrimaryDim::Ki]))
        );
    }

    /// e045 — the superchunk stage is the chunk stage's dims under the superchunk name, both halves.
    #[test]
    fn the_super_chunk_stage_is_the_chunk_stage_renamed() {
        let mut dsc = dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 2)]);
        let minted = DatastageId(2);
        dsc.data_stages
            .set(minted, stage("2", &[(PrimaryDim::I, 1)]));
        let witness = dsc
            .data_stages
            .super_chunk(minted)
            .expect("an index getNewDataStageIndex already inserted");
        add_super_chunk_data_stage(&mut dsc, witness);
        let added = dsc.data_stages.at(minted).expect("the stage just written");
        assert_eq!(added.name(), &StageName::super_chunk());
        assert_eq!(added.el.name, StageName::super_chunk());
        assert_eq!(added.ss_extent(PrimaryDim::I), Some(Extent(2)));
        assert_eq!(dsc.data_stages.super_chunk(DatastageId(3)), None);
    }

    /// e046 — the named block is found below a loop, and a tree without it answers nothing.
    #[test]
    fn the_lx_below_block_node_is_found_under_a_loop() {
        let named = BlockNode {
            name: NodeName(LX_BELOW_BLOCK_NODE_NAME.to_owned()),
            children: vec![],
        };
        let mut tree = ScheduleTree::new(BlockNode {
            name: NodeName("head".to_owned()),
            children: vec![SchedNode::Loop(BlockNode {
                name: NodeName("loop_ds0_ds1".to_owned()),
                children: vec![SchedNode::Block(named)],
            })],
        });
        let found = lx_below_block_node(&mut tree).expect("the lx-below block");
        found
            .children
            .push(SchedNode::Leaf(NodeName("t".to_owned())));
        assert_eq!(tree.blocks_dfs().len(), 1);
        assert_eq!(tree.blocks_dfs()[0].children.len(), 1);
        assert_eq!(lx_below_block_node(&mut ScheduleTree::default()), None);
    }

    /// e047 — every layout dim once, with the indirect-access-index structure's dims last.
    #[test]
    fn the_loop_order_puts_the_indirect_access_index_dims_last() {
        let mut dsc = dsc(&[(PrimaryDim::I, 1)], &[(PrimaryDim::I, 1)]);
        dsc.labeled_ds = LabeledDsList::new(sized(LdsIdx(0), &[]), vec![sized(LdsIdx(5), &[])]);
        dsc.layout_dims.insert(
            LdsIdx(0),
            LayoutDims::new(PrimaryDim::I, vec![PrimaryDim::J]),
        );
        dsc.layout_dims.insert(
            LdsIdx(5),
            LayoutDims::new(PrimaryDim::Ki, vec![PrimaryDim::I]),
        );
        // Membership is by POSITION, while the layout is looked up by the RECORDED index 5.
        dsc.indirect_access_index_lds.insert(LdsIdx(1));
        assert_eq!(
            collect_all_dimensions_for_loop_order(&dsc),
            Some(vec![PrimaryDim::I, PrimaryDim::J, PrimaryDim::Ki])
        );
    }

    /// e048 — two cores on the same slice share a name, and that name is kept for the same set.
    #[test]
    fn shares_and_group_name_memoise_the_set_of_sharing_cores() {
        let mut dsc = dsc(&[(PrimaryDim::I, 1)], &[(PrimaryDim::I, 1)]);
        let lds = sized(LdsIdx(0), &[PrimaryDim::I]);
        dsc.labeled_ds = LabeledDsList::new(lds.clone(), vec![]);
        dsc.layout_dims
            .insert(lds.recorded(), LayoutDims::new(PrimaryDim::I, vec![]));
        let slice = |id: i32| WkSlice(BTreeMap::from([(PrimaryDim::I, WkSliceId(id))]));
        let core = |index: u32| Core::checked(index).expect("core in range");
        let sdsc = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            BTreeMap::new(),
            BTreeMap::from([
                (core(0), slice(0)),
                (core(1), slice(0)),
                (core(2), slice(1)),
            ]),
            BTreeMap::new(),
        );
        let processing = BTreeSet::from([core(0), core(1), core(2)]);
        let mut names = GtrGroupNames::new();
        let shared = shares_and_group_name(&sdsc, &dsc, &lds, &slice(0), &processing, &mut names);
        assert_eq!(
            shared,
            Some((
                Shares(2),
                GroupName::Shared(GtrGroupId::checked(0).expect("id 0"))
            ))
        );
        assert_eq!(
            shares_and_group_name(&sdsc, &dsc, &lds, &slice(0), &processing, &mut names),
            shared
        );
        assert_eq!(
            shares_and_group_name(&sdsc, &dsc, &lds, &slice(1), &processing, &mut names),
            Some((Shares(1), GroupName::Unshared))
        );
        assert_eq!(
            shares_and_group_name(&sdsc, &dsc, &lds, &slice(2), &processing, &mut names),
            None
        );
    }
}

/// Replaces: e049_calculateCoreletOffsetInByte
///
/// Where each corelet's share of an allocation begins, in bytes: the stick size times each
/// non-broadcast dim's stick count, walked until the one corelet-split dim, accumulated corelet by
/// corelet.
///
/// ⛔⛔ THE CHECKED STRIDE IS NOT THE USED STRIDE. Both stage checks of the padded arm read `dsNode`
/// (`:4886`, `:4890`), one of them worded *"Expect padding sizes in chunk data stage params"*, while
/// the arithmetic multiplies `dsChunk.paddingSizes_.at(dim).stride_` and `dsChunk.coreletSplit_`
/// (`:4892-4893`); entry 219, the near-duplicate, checks `dsChunk`. Both stages are asked here.
/// ⛔ THE OPERANDS COME FROM CORELET `id - 1` WHILE THE OFFSET LANDS ON CORELET `id`.
/// ⛔ The reference accumulates in `int` and returns `int64_t`, with `bytesPerStick` multiplied in
/// FIRST; this checks in `u64` and answers [`None`] on overflow instead of wrapping.
#[must_use]
pub fn calculate_corelet_offset_in_byte<A: Arch, N: DimStage + ?Sized, C: DimStage + ?Sized>(
    dsc: &DesignSpaceConfig,
    node_stage: &N,
    chunk_stage: &C,
    lds: LdsIdx,
    component: SenComponent,
    padding: &PaddingForm,
) -> Option<BTreeMap<Corelet, CoreletOffset>> {
    let corelets = dsc.corelets_used_dsc2?.get();
    let mut offsets = (0..corelets)
        .map(|id| Some((Corelet::checked(id)?, CoreletOffset(Bytes(0)))))
        .collect::<Option<BTreeMap<_, _>>>()?;
    if corelets < 2 {
        return Some(offsets);
    }

    let dims = dsc.non_broadcast_lds_dims(lds)?;
    let split: Vec<PrimaryDim> = dims
        .iter()
        .copied()
        .filter(|&dim| chunk_stage.is_corelet_split(dim))
        .collect();
    let split_dim = match split.as_slice() {
        [] => return Some(offsets),
        [only] => *only,
        // `DT_CHECK_MSG(.size() <= 1, "Support maximal one corelet split dimension for a tensor")`.
        _ => return None,
    };

    let sticks = dsc.cumulative_stick_sizes(dsc.labeled_ds.at(lds)?.ds_type())?;
    let mut overall: u64 = 0;
    for id in 1..corelets {
        let from = Corelet::checked(id - 1)?;
        let mut offset = A::BYTES_PER_STICK.get();
        for dim in &dims {
            let dim = *dim;
            let extent = if dim != split_dim {
                node_stage.corelet_dim_val(dim, component, from, padding)?
            } else if padding.padding(dim) == PadType::NoPad {
                chunk_stage.corelet_dim_val(dim, component, from, padding)?
            } else {
                // `offset_in_element = size_of_i * stride`, and only for `I`.
                (padding.padding(dim) == PadType::PaddedFullSpanWUnneeded).then_some(())?;
                (dim == PrimaryDim::I).then_some(())?;
                node_stage.pad_stride(dim)?;
                let share = chunk_stage.corelet_split(dim, from)?;
                Extent(share.0.checked_mul(chunk_stage.pad_stride(dim)?.0)?)
            };
            let per_stick = v1::stick_divisor(&sticks, dim)?;
            offset = offset.checked_mul(u64::try_from(extent.0).ok()? / per_stick.get())?;
            if dim == split_dim {
                break;
            }
        }
        overall = overall.checked_add(offset)?;
        offsets.insert(Corelet::checked(id)?, CoreletOffset(Bytes(overall)));
    }
    Some(offsets)
}

/// Replaces: e050_getInitialStartAddressAndOffset
///
/// Where a labelled DS's LX data starts on one core and which buffer it reads, both taken at
/// corelet 0.
///
/// ⛔ CORELET 0 IS BOUND HERE, NOT BY THE CALLER: the reference OVERWRITES `coord.at(1)`, so a
/// caller's corelet is discarded and only the core and the super-DSC folds behind it survive.
/// ⛔ [`None`] IS EVERY REFUSAL AT ONCE — no `LX` in `memOrg_`, a null LX allocate node, a
/// `numBuffers_` the field's comment does not name, an unplaced address, and a buffering NEITHER arm
/// admits: `{1, 2}` when HBM pinned (`:4940-4941`) and `{1}` alone when not (`:4947`).
#[must_use]
pub fn initial_start_address_and_offset<M: MemOrg + ?Sized>(
    mem: &M,
    coord: &AddressCoord,
) -> Option<InitialPlacement> {
    let at = AddressCoord {
        core: coord.core,
        corelet: Corelet::at::<0>(),
        sdsc_folds: coord.sdsc_folds.clone(),
    };
    let buffering = mem.lx_buffering()?;
    let start = mem.lx_start_address(&at)?;
    let buffer_offset = if mem.hbm_pinned() {
        // `DT_CHECK_MSG(numBuffers_ == 1 || numBuffers_ == 2, "Expect no buffering or double
        // buffering.")` — [`Buffering::Streaming`] is a third name entry 016 can mint.
        matches!(buffering, Buffering::None | Buffering::Double).then_some(())?;
        mem.lx_buffer_offset(at.core, at.corelet)?
    } else {
        // "There is always only one buffer in this case, so the offset is zero."
        matches!(buffering, Buffering::None).then_some(())?;
        BufferOffset(0)
    };
    Some(InitialPlacement {
        start,
        buffer_offset,
    })
}

/// Replaces: e051_getLdsOrConstNameOfAllocNode
///
/// The L3 scheduler's own copy of [`v1::get_lds_or_const_name_of_alloc_node`].
///
/// ⭐⭐ IT DELEGATES BECAUSE THE TWO BODIES ARE IDENTICAL — `Ddc::getLdsOrConstNameOfAllocNode`
/// (`ddc/ddcv1.cpp:20-30`) differs only in `currDsc` being a member there and a parameter here.
/// ⚠️ The *"mostly copied from DDC ... frequently synchronize"* TODO the file carries is NOT on this
/// function: it sits on `allocAllMem` (`:5505`) and `fillLoopOffsetsAndAddresses` (`:5747`).
#[must_use]
pub fn get_lds_or_const_name_of_alloc_node(
    anode: &AllocateNode,
    names: &impl v1::StorageNames,
) -> Option<v1::StorageName> {
    v1::get_lds_or_const_name_of_alloc_node(anode, names)
}

/// A LOOP ORDER PROVED TO NAME EACH DIM ONCE — `verifyLoopOrder`'s `isGood` made unconstructible
/// where it is false.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopOrder(Vec<PrimaryDim>);

impl LoopOrder {
    /// Replaces: e052_verifyLoopOrder
    ///
    /// A loop order is good exactly when no dim repeats in it; [`None`] is the reference's `false`.
    ///
    /// ⭐ THE `verbose` PRINT IS DIAGNOSTICS, NOT THE ANSWER: the reference names every repeat on
    /// `std::cout` and keeps scanning, and the value it returns does not depend on the printing.
    #[must_use]
    pub fn of(order: &[PrimaryDim]) -> Option<Self> {
        let mut visited = BTreeSet::new();
        order
            .iter()
            .all(|dim| visited.insert(*dim))
            .then(|| Self(order.to_vec()))
    }

    /// The order, as the reference's `loopOrder` holds it.
    #[must_use]
    pub fn dims(&self) -> &[PrimaryDim] {
        &self.0
    }
}

/// A SCHEDULE TREE PROVED NON-EMPTY AND UNIQUELY NAMED — `verifyScheduleTree`'s two answers as one
/// type, because a caller that has this has both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedScheduleTree(());

impl VerifiedScheduleTree {
    /// Replaces: e053_verifyScheduleTree
    ///
    /// A schedule tree passes when it has nodes at all and no two of them share a name.
    ///
    /// ⛔ ITS TWO `false`s ARE ONE ABSENCE: `scheduleTree_.empty()` returns early and a repeated
    /// `name_` falls out of the name set, and no caller distinguishes them. The `verbose` print is
    /// diagnostics.
    #[must_use]
    pub fn of<T: ScheduleNodes + ?Sized>(tree: &T) -> Option<Self> {
        let names = tree.node_names();
        (!names.is_empty()).then_some(())?;
        let mut all_names = BTreeSet::new();
        names
            .into_iter()
            .all(|name| all_names.insert(name))
            .then_some(Self(()))
    }
}

/// Replaces: e054_prepDsc
///
/// Gives every DSC its dsc2 corelet count and mints its scheduler metadata, naming the core and
/// chunk data stages.
///
/// ⭐⭐ THIS IS THE MUTATION, and it is why [`DesignSpaceConfig::corelets_used_dsc2`] is an
/// [`Option`]: a DSC is built holding the reference's `-1`, and this is the only thing that
/// replaces it. "do imbalanced corelet split in the future" — the copy is the whole rule today.
/// ⭐ `dscMetadata.emplace` KEEPS AN EXISTING ENTRY, but both ids are written after it unconditionally,
/// so a rerun lands the same two values on every DSC and a freshly built map is that same state.
pub fn prep_dsc(sdsc: &mut SuperDsc) -> BTreeMap<DscIdx, SchedulerMetadata> {
    for dsc in sdsc.dscs_mut().iter_mut() {
        dsc.corelets_used_dsc2 = Some(dsc.corelets_used);
    }
    sdsc.dscs()
        .iter()
        .zip(0u32..)
        .map(|(_, idx)| {
            (
                DscIdx(idx),
                SchedulerMetadata {
                    core_dstg: DATA_STAGE_CORE,
                    chunk_dstg: DATA_STAGE_CHUNK,
                },
            )
        })
        .collect()
}

/// AN HMI GROUP KEY — address bits 39, 36, 35 and 34 packed into four, most significant first
/// (`L3DlOpsScheduler.cpp:6552-6563`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HmiBits(pub u32);

impl HmiBits {
    /// The four bits one address falls into.
    #[must_use]
    pub const fn of(address: ByteAddress) -> Self {
        let bits = address.0;
        Self(
            (((bits >> 39) & 1) << 3
                | ((bits >> 36) & 1) << 2
                | ((bits >> 35) & 1) << 1
                | ((bits >> 34) & 1)) as u32,
        )
    }
}

/// HOW MANY WORK SLICES SHARE ONE HMI REQUEST GROUP — entry 055's `int`, a count of distinct
/// addresses and never zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HmiGroupSize(pub u32);

/// Replaces: e055_computeMinHMICoreGroupSizeForSEN1P5
///
/// The smallest HMI request group over the given HBM tensors: per tensor one address per used core,
/// deduplicated, bucketed by [`HmiBits`], the smallest bucket taken, then the smallest across tensors.
///
/// ⛔ [`None`] IS THE `INT_MAX` THE REFERENCE RETURNS UNTOUCHED when no tensor produced a group at
/// all — and also `DT_CHECK_MSG(coreArch == SEN1P5_ISA, "Expecting SEN1P5_ISA.")`, which no other
/// generation survives.
/// ⛔ FIRST ADDRESS PER CORE WINS, and across DSCs the reference walks an `unordered_map`, so which
/// DSC that is is unspecified there; `dscs_` order is taken here.
#[must_use]
pub fn compute_min_hmi_core_group_size_for_sen1p5<A: Arch, T: ScheduleTrees + ?Sized>(
    sdsc: &SuperDsc,
    trees: &T,
    hbm_lds: &[LdsIdx],
) -> Option<HmiGroupSize> {
    if A::GEN != IsaGen::Sen1p5 {
        return None;
    }
    let mut min_requests: Option<HmiGroupSize> = None;
    for &lds in hbm_lds {
        let mut core_to_address: BTreeMap<Core, ByteAddress> = BTreeMap::new();
        for (dsc, idx) in sdsc.dscs().iter().zip(0u32..) {
            let used: BTreeSet<Core> = dsc.core_ids_used.iter().collect();
            for alloc in trees.allocations(DscIdx(idx)) {
                if alloc.lds != Some(lds) || alloc.component != SenComponent::Hbm {
                    continue;
                }
                for (core, address) in alloc.addresses {
                    if used.contains(&core) {
                        core_to_address.entry(core).or_insert(address);
                    }
                }
            }
        }
        let unique: BTreeSet<ByteAddress> = core_to_address.into_values().collect();
        let mut groups: BTreeMap<HmiBits, u32> = BTreeMap::new();
        for address in unique {
            *groups.entry(HmiBits::of(address)).or_insert(0) += 1;
        }
        if let Some(&smallest) = groups.values().min() {
            min_requests = Some(HmiGroupSize(
                min_requests.map_or(smallest, |seen: HmiGroupSize| seen.0.min(smallest)),
            ));
        }
    }
    min_requests
}

/// Replaces: e056_isIndexLds
///
/// Whether a labelled DS is an INDEX tensor — its HBM allocation indirects through one.
///
/// ⛔ [`None`] IS `DT_CHECK_MSG(indexTensorType_ == ADDRESS, "Only index tensors of type address are
/// supported")`: an index tensor holding INDICES is a refusal, not a `false`. Every other shape — no
/// HBM entry, a null allocate node, no indirection, a value tensor — answers `false`.
#[must_use]
pub fn is_index_lds<M: MemOrg + ?Sized>(lds: &M) -> Option<bool> {
    match lds.hbm_indirection() {
        Some(IndirectAlloc::IndexTensor(IndexTensor::Address)) => Some(true),
        Some(IndirectAlloc::IndexTensor(IndexTensor::Index)) => None,
        Some(IndirectAlloc::ValueTensor) | None => Some(false),
    }
}

#[cfg(test)]
mod tests_e049_e056 {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims;
    use crate::schedule::ddc::fold::Stride;
    use crate::schedule::dsc2::{
        AllocLayout, AllocPlacement, LayoutDims, MaxDimSize, StartAddress,
    };
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DscList, LabeledDsList, NamedDims,
        PlacedAllocation, PrimaryDsInfo, StageDims,
    };
    use std::num::NonZeroU32;

    fn core(index: u32) -> Core {
        Core::checked(index).expect("core in range")
    }

    fn corelets(count: u32) -> CoreletsUsed {
        CoreletsUsed::new(NonZeroU32::new(count).expect("a corelet count"))
    }

    /// A DSC labelling one INPUT tensor whose layout and stick both name `dims`, none broadcast.
    fn dsc(dims: &[(PrimaryDim, u64)], sticks: &[(PrimaryDim, u64)]) -> DesignSpaceConfig {
        let (first, _) = *dims.first().expect("a tensor with a dim in it");
        let layout = LayoutDims::new(first, dims[1..].iter().map(|(dim, _)| *dim).collect());
        let mut stage = StageDims::default();
        stage.extents.insert(first, Extent(1));
        DesignSpaceConfig {
            corelets_used: corelets(2),
            corelets_used_dsc2: Some(corelets(2)),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::from([(
                DsType::Input,
                PrimaryDsInfo {
                    layout: layout.clone(),
                    stick: StickDims(
                        sticks
                            .iter()
                            .map(|&(dim, size)| (dim, Elements(size)))
                            .collect(),
                    ),
                },
            )]),
            core_ids_used: CoreIdsUsed::new(core(0), vec![core(1)]),
            layout_dims: BTreeMap::from([(LdsIdx(0), layout)]),
            data_stages: {
                let named = NamedDims {
                    name: StageName::default(),
                    dims: FilledDims::of(stage).expect("a stage that states a dim"),
                };
                let stage = DataStage {
                    ss: named.clone(),
                    el: named,
                };
                DataStages::new(stage.clone(), stage)
            },
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(
                    DsType::Input,
                    dims.iter()
                        .map(|&(dim, _)| (dim, Scale::Sized(1.0)))
                        .collect(),
                    LdsIdx(0),
                    Pinning::default(),
                ),
                vec![],
            ),
        }
    }

    /// One `DataStructDims`, stated by lookup.
    #[derive(Default)]
    struct Stage {
        vals: BTreeMap<(PrimaryDim, u32), Extent>,
        splits: BTreeMap<(PrimaryDim, u32), Extent>,
        strides: BTreeMap<PrimaryDim, Stride>,
    }

    impl DimStage for Stage {
        fn corelet_dim_val(
            &self,
            dim: PrimaryDim,
            _comp: SenComponent,
            corelet: Corelet,
            _padded: &PaddingForm,
        ) -> Option<Extent> {
            self.vals.get(&(dim, corelet.get())).copied()
        }

        fn is_corelet_split(&self, dim: PrimaryDim) -> bool {
            self.splits.keys().any(|(split, _)| *split == dim)
        }

        fn corelet_split(&self, dim: PrimaryDim, corelet: Corelet) -> Option<Extent> {
            self.splits.get(&(dim, corelet.get())).copied()
        }

        fn pad_stride(&self, dim: PrimaryDim) -> Option<Stride> {
            self.strides.get(&dim).copied()
        }
    }

    /// e049 — the offset is the stick size times each outer dim's stick count, and it stops at the
    /// corelet-split dim: 128 × (4 / 1) × (16 / 8) for corelet 1, and nothing for corelet 0.
    #[test]
    fn corelet_offset_stops_at_the_split_dim() {
        let config = dsc(
            &[(PrimaryDim::Out, 4), (PrimaryDim::In, 16)],
            &[(PrimaryDim::In, 8)],
        );
        let node = Stage {
            vals: BTreeMap::from([((PrimaryDim::Out, 0), Extent(4))]),
            ..Stage::default()
        };
        let chunk = Stage {
            vals: BTreeMap::from([((PrimaryDim::In, 0), Extent(16))]),
            splits: BTreeMap::from([((PrimaryDim::In, 0), Extent(16))]),
            ..Stage::default()
        };
        let offsets = calculate_corelet_offset_in_byte::<Dd2, _, _>(
            &config,
            &node,
            &chunk,
            LdsIdx(0),
            SenComponent::Lx,
            &PaddingForm::default(),
        )
        .expect("a stated corelet offset");
        assert_eq!(
            offsets,
            BTreeMap::from([
                (Corelet::at::<0>(), CoreletOffset(Bytes(0))),
                (Corelet::at::<1>(), CoreletOffset(Bytes(1024))),
            ])
        );
    }

    /// e049 — two corelet-split dims is `DT_CHECK_MSG(ldsCoreletSplitDim.size() <= 1, ..)`, and no
    /// split dim at all leaves every offset at zero.
    #[test]
    fn corelet_offset_refuses_a_second_split_dim() {
        let config = dsc(
            &[(PrimaryDim::Out, 4), (PrimaryDim::In, 16)],
            &[(PrimaryDim::In, 8)],
        );
        let node = Stage::default();
        let two = Stage {
            splits: BTreeMap::from([
                ((PrimaryDim::Out, 0), Extent(4)),
                ((PrimaryDim::In, 0), Extent(16)),
            ]),
            ..Stage::default()
        };
        assert_eq!(
            calculate_corelet_offset_in_byte::<Dd2, _, _>(
                &config,
                &node,
                &two,
                LdsIdx(0),
                SenComponent::Lx,
                &PaddingForm::default(),
            ),
            None
        );
        assert_eq!(
            calculate_corelet_offset_in_byte::<Dd2, _, _>(
                &config,
                &node,
                &Stage::default(),
                LdsIdx(0),
                SenComponent::Lx,
                &PaddingForm::default(),
            )
            .expect("no split dim is still an answer")
            .values()
            .copied()
            .collect::<Vec<_>>(),
            vec![CoreletOffset(Bytes(0)); 2]
        );
    }

    /// One labelled DS's `memOrg_`, stated by field.
    struct MemOrgStub {
        pinned: bool,
        buffering: Option<Buffering>,
        address: Option<ByteAddress>,
        offset: Option<BufferOffset>,
        indirection: Option<IndirectAlloc>,
    }

    impl Default for MemOrgStub {
        fn default() -> Self {
            Self {
                pinned: false,
                buffering: Some(Buffering::None),
                address: Some(ByteAddress(0x4000)),
                offset: Some(BufferOffset(0x80)),
                indirection: None,
            }
        }
    }

    impl MemOrg for MemOrgStub {
        fn hbm_pinned(&self) -> bool {
            self.pinned
        }

        fn lx_buffering(&self) -> Option<Buffering> {
            self.buffering
        }

        fn lx_start_address(&self, at: &AddressCoord) -> Option<ByteAddress> {
            // The corelet the caller asked for is gone; entry 050 bound corelet 0 instead.
            (at.corelet.get() == 0).then_some(())?;
            self.address
        }

        fn lx_buffer_offset(&self, _core: Core, _corelet: Corelet) -> Option<BufferOffset> {
            self.offset
        }

        fn lx_zero_padded(&self) -> Option<bool> {
            Some(false)
        }

        fn hbm_indirection(&self) -> Option<IndirectAlloc> {
            self.indirection
        }

        fn hbm_allocation(&self) -> Option<NodeName> {
            None
        }

        fn hbm_layout_dims(&self) -> Option<crate::schedule::dsc2::LayoutDims> {
            None
        }

        fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim> {
            BTreeSet::new()
        }

        fn lx_padding(&self) -> Option<PaddingForm> {
            None
        }

        fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent> {
            BTreeMap::new()
        }

        fn hbm_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }

        fn lx_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }
    }

    /// e050 — a pinned DS takes its buffer offset from the node while an unpinned one is always at
    /// zero, and each arm refuses the buffering its own `numBuffers_` check excludes.
    #[test]
    fn initial_placement_reads_corelet_zero() {
        let coord = AddressCoord {
            core: core(3),
            corelet: Corelet::at::<1>(),
            sdsc_folds: vec![7],
        };
        assert_eq!(
            initial_start_address_and_offset(
                &MemOrgStub {
                    pinned: true,
                    buffering: Some(Buffering::Double),
                    ..MemOrgStub::default()
                },
                &coord,
            ),
            Some(InitialPlacement {
                start: ByteAddress(0x4000),
                buffer_offset: BufferOffset(0x80),
            })
        );
        assert_eq!(
            initial_start_address_and_offset(&MemOrgStub::default(), &coord),
            Some(InitialPlacement {
                start: ByteAddress(0x4000),
                buffer_offset: BufferOffset(0),
            })
        );
        assert_eq!(
            initial_start_address_and_offset(
                &MemOrgStub {
                    buffering: Some(Buffering::Double),
                    ..MemOrgStub::default()
                },
                &coord,
            ),
            None
        );
        // The pinned arm admits `{1, 2}` and no more, so streaming refuses there too.
        assert_eq!(
            initial_start_address_and_offset(
                &MemOrgStub {
                    pinned: true,
                    buffering: Some(Buffering::Streaming),
                    ..MemOrgStub::default()
                },
                &coord,
            ),
            None
        );
    }

    /// e051 — it is the DDC's resolver, reached through the L3 scheduler's copy.
    #[test]
    fn lds_name_delegates_to_the_ddc_resolver() {
        struct Names;
        impl v1::StorageNames for Names {
            fn lds_name(&self, lds: LdsIdx) -> v1::StorageName {
                v1::StorageName(format!("lds{}", lds.0))
            }
            fn constant_name(
                &self,
                _constant: crate::schedule::ddc::fold::ConstIdx,
            ) -> v1::StorageName {
                v1::StorageName("constant".to_owned())
            }
        }
        let anode = AllocateNode {
            name: NodeName("alloc".to_owned()),
            component: SenComponent::Lx,
            lds: Some(LdsIdx(2)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new((PrimaryDim::Out, MaxDimSize::Unset), Vec::new()),
            start_address: StartAddress::default(),
            placement: AllocPlacement::default(),
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        };
        assert_eq!(
            get_lds_or_const_name_of_alloc_node(&anode, &Names),
            v1::get_lds_or_const_name_of_alloc_node(&anode, &Names)
        );
        assert_eq!(
            get_lds_or_const_name_of_alloc_node(&anode, &Names),
            Some(v1::StorageName("lds2".to_owned()))
        );
    }

    /// e052 — a repeated dim has no loop order, and one that names each dim once keeps its order.
    #[test]
    fn loop_order_is_unique_dims() {
        assert_eq!(
            LoopOrder::of(&[PrimaryDim::Out, PrimaryDim::In, PrimaryDim::Out]),
            None
        );
        assert_eq!(
            LoopOrder::of(&[PrimaryDim::Out, PrimaryDim::In])
                .expect("unique dims")
                .dims(),
            [PrimaryDim::Out, PrimaryDim::In]
        );
    }

    /// One DSC's schedule tree, stated as its node names, plus its allocations per DSC index.
    #[derive(Default)]
    struct Tree {
        names: Vec<NodeName>,
        allocs: BTreeMap<DscIdx, Vec<PlacedAllocation>>,
    }

    impl ScheduleNodes for Tree {
        fn node_names(&self) -> Vec<NodeName> {
            self.names.clone()
        }
    }

    impl ScheduleTrees for Tree {
        fn allocations(&self, dsc: DscIdx) -> Vec<PlacedAllocation> {
            self.allocs.get(&dsc).cloned().unwrap_or_default()
        }
    }

    fn named(names: &[&str]) -> Tree {
        Tree {
            names: names
                .iter()
                .map(|name| NodeName((*name).to_owned()))
                .collect(),
            ..Tree::default()
        }
    }

    /// e053 — an empty tree and a repeated name are the same absence, and distinct names pass.
    #[test]
    fn schedule_tree_needs_nodes_and_unique_names() {
        assert_eq!(VerifiedScheduleTree::of(&named(&[])), None);
        assert_eq!(VerifiedScheduleTree::of(&named(&["b", "l", "b"])), None);
        assert_eq!(
            VerifiedScheduleTree::of(&named(&["b", "l"])),
            Some(VerifiedScheduleTree(()))
        );
    }

    /// e054 — every DSC gets its dsc2 corelet count, and one metadata entry per DSC naming stages
    /// 0 and 1.
    #[test]
    fn prep_dsc_fills_the_dsc2_corelet_count() {
        let mut plain = dsc(&[(PrimaryDim::In, 8)], &[(PrimaryDim::In, 8)]);
        plain.corelets_used = CoreletsUsed::ONE;
        plain.corelets_used_dsc2 = None;
        let mut sdsc = SuperDsc::new(
            DscList::new(plain.clone(), vec![plain]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let metadata = prep_dsc(&mut sdsc);
        assert!(
            sdsc.dscs()
                .iter()
                .all(|dsc| dsc.corelets_used_dsc2 == Some(CoreletsUsed::ONE))
        );
        assert_eq!(
            metadata,
            BTreeMap::from([
                (
                    DscIdx(0),
                    SchedulerMetadata {
                        core_dstg: DATA_STAGE_CORE,
                        chunk_dstg: DATA_STAGE_CHUNK,
                    }
                ),
                (
                    DscIdx(1),
                    SchedulerMetadata {
                        core_dstg: DATA_STAGE_CORE,
                        chunk_dstg: DATA_STAGE_CHUNK,
                    }
                ),
            ])
        );
    }

    /// e055 — three used cores land in two HMI buckets (bit 34 apart) of two and one, so the answer
    /// is the smaller bucket; a core the DSC does not use is not counted, and only SEN1P5 answers.
    #[test]
    fn min_hmi_group_is_the_smallest_bucket() {
        let sdsc = SuperDsc::new(
            DscList::new(dsc(&[(PrimaryDim::In, 8)], &[(PrimaryDim::In, 8)]), vec![]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let bit34 = 1u64 << 34;
        let trees = Tree {
            allocs: BTreeMap::from([(
                DscIdx(0),
                vec![PlacedAllocation {
                    lds: Some(LdsIdx(0)),
                    component: SenComponent::Hbm,
                    addresses: vec![
                        (core(0), ByteAddress(0x1000)),
                        (core(1), ByteAddress(0x2000)),
                        (core(2), ByteAddress(bit34)),
                    ],
                }],
            )]),
            ..Tree::default()
        };
        assert_eq!(
            compute_min_hmi_core_group_size_for_sen1p5::<Sen1p5, _>(&sdsc, &trees, &[LdsIdx(0)]),
            Some(HmiGroupSize(2))
        );
        assert_eq!(
            compute_min_hmi_core_group_size_for_sen1p5::<Dd2, _>(&sdsc, &trees, &[LdsIdx(0)]),
            None
        );
        assert_eq!(
            compute_min_hmi_core_group_size_for_sen1p5::<Sen1p5, _>(&sdsc, &trees, &[LdsIdx(1)]),
            None
        );
    }

    /// e056 — an address index tensor is one, an INDEX one is the refusal, and everything else is no.
    #[test]
    fn index_lds_is_an_address_index_tensor() {
        let of = |indirection| {
            is_index_lds(&MemOrgStub {
                indirection,
                ..MemOrgStub::default()
            })
        };
        assert_eq!(
            of(Some(IndirectAlloc::IndexTensor(IndexTensor::Address))),
            Some(true)
        );
        assert_eq!(
            of(Some(IndirectAlloc::IndexTensor(IndexTensor::Index))),
            None
        );
        assert_eq!(of(Some(IndirectAlloc::ValueTensor)), Some(false));
        assert_eq!(of(None), Some(false));
    }
}

/// Replaces: e057_isPagedLds
///
/// Whether a labelled DS is a PAGED tensor — its HBM allocation holds the VALUES an indirect access
/// pages through.
///
/// ⛔ `isPresent` IS NOT CONSULTED, unlike [`get_hbm_allocations`]: an HBM entry that merely CARRIES
/// an allocation answers `true`. ⭐ And unlike its twin [`is_index_lds`] there is no refusal on this
/// arm, because no `indexTensorType_` is read.
#[must_use]
pub fn is_paged_lds<M: MemOrg + ?Sized>(lds: &M) -> bool {
    matches!(lds.hbm_indirection(), Some(IndirectAlloc::ValueTensor))
}

/// Replaces: e058_getNewDataStageIndex
///
/// MINTS AN EMPTY DATA STAGE in this DSC under the lowest index above [`DATA_STAGE_CHUNK`] that no
/// DSC of the super-DSC holds, and answers with that index.
///
/// ⛔ DELIBERATE DIVERGENCE — THE REFERENCE HANGS: its outer `while` retries WITHOUT advancing
/// `newIdx` once a SIBLING DSC holds it, and the inner `while` cannot advance past an index this DSC
/// does not hold. ⭐ `&otherDsc != &dsc` was dead code either way — the inner loop had already
/// skipped every index this DSC holds — and the exclusive borrow is what spells that out.
pub fn get_new_data_stage_index<D: Default>(
    dsc: &mut DataStages<D>,
    other_dscs: &[&DataStages<D>],
) -> DatastageId {
    let mut id = DatastageId(DATA_STAGE_CHUNK.0 + 1);
    while dsc.0.contains_key(&id) || other_dscs.iter().any(|other| other.0.contains_key(&id)) {
        id.0 += 1;
    }
    dsc.0.entry(id).or_default();
    id
}

/// Replaces: e059_getPagedDimensions
///
/// THE DSC'S PAGED DIMS — every INDEX-TENSOR allocation's layout dim that its own pages span, in
/// layout order, each dim once, over `labeledDs_` in order.
///
/// ⛔ *"Expect valid layoutDimOrder_."* IS GONE BY CONSTRUCTION — [`crate::schedule::dsc2::
/// LayoutDims`] is non-empty. ⚠️ *"Expect a valid HBM allocate node."* (`:6715`) is WIDER THERE than
/// here: it guards EVERY lds with an HBM entry, so one holding no node aborts the reference where
/// this skips it. Same answer wherever the reference answers at all.
#[must_use]
pub fn get_paged_dimensions<M: MemOrg + ?Sized>(labeled_ds: &[&M]) -> Vec<PrimaryDim> {
    let mut dims: Vec<PrimaryDim> = Vec::new();
    for lds in labeled_ds {
        let Some(IndirectAlloc::IndexTensor(_)) = lds.hbm_indirection() else {
            continue;
        };
        let Some(layout) = lds.hbm_layout_dims() else {
            continue;
        };
        let pages = lds.hbm_page_dims();
        for dim in layout.iter() {
            if pages.contains(&dim) && !dims.contains(&dim) {
                dims.push(dim);
            }
        }
    }
    dims
}

/// Replaces: e060_getHbmAllocations
///
/// The HBM allocate node of every HBM-PINNED labelled DS of the DSC, in `labeledDs_` order.
///
/// ⛔ [`None`] IS *"Expect a valid allocate node."* — a DS that states `isPresent` while its HBM
/// entry holds none; *"Expect HBM in memOrg_."* cannot be reached from a true `isHbmPinned()`.
/// ⚠️ The reference hands back NON-const pointers out of a `const` DSC; a later write to one is that
/// unit's own `memOrg_` read, so what travels out of here is the node's name. The sole caller
/// `propagateCoordinateDSC` (`:7763`) uses them only as worklist and visited-set IDENTITIES, and
/// entry 053 is what makes a name one: node names are distinct within a tree.
#[must_use]
pub fn get_hbm_allocations<M: MemOrg + ?Sized>(labeled_ds: &[&M]) -> Option<Vec<NodeName>> {
    labeled_ds
        .iter()
        .filter(|lds| lds.hbm_pinned())
        .map(|lds| lds.hbm_allocation())
        .collect()
}

/// Replaces: e061_gatherFoldParams
///
/// APPENDS ONE DIM'S FOLDS — cardinality, label and the affine pair — to the caller's fold-param
/// list, outermost position first.
///
/// ⭐ [`Fold`] IS `dsc2::FoldParamInfoType` FIELD FOR FIELD, so the copy is the whole of it, and the
/// reference's `const_cast` writes nothing: it only reaches getters that were not marked `const`.
/// ⛔ `getAlphaBeta`'s *"Only affine folds are supported"* is [`Fold`]'s own shape, which states an
/// alpha and a beta and cannot spell any other kind.
pub fn gather_fold_params(fm: &FoldDim, fold_params: &mut Vec<Fold>) {
    fold_params.extend(fm.folds().cloned());
}

/// THE LOOPS ENCLOSING ONE SCHEDULE NODE, INNERMOST FIRST AND THE ROOT LOOP LAST — the
/// `getMutableOwnerLoop()` walk, NON-EMPTY.
///
/// ⛔⛔ `loopChain.pop_back()` ON AN EMPTY CHAIN IS UB, not an empty answer, and a node no loop
/// encloses reaches it — so such a node has no witness here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerLoops<'a> {
    below_root: Vec<&'a LoopNode>,
    root: &'a LoopNode,
}

impl<'a> OwnerLoops<'a> {
    /// The walk's result, innermost first and THE ROOT LOOP LAST, or [`None`] where the reference
    /// pops an empty chain.
    #[must_use]
    pub fn of(innermost_first: Vec<&'a LoopNode>) -> Option<Self> {
        let (&root, below_root) = innermost_first.split_last()?;
        Some(Self {
            below_root: below_root.to_vec(),
            root,
        })
    }

    /// Every enclosing loop, innermost first, INCLUDING the root.
    pub fn iter(&self) -> impl Iterator<Item = &'a LoopNode> + '_ {
        self.below_root
            .iter()
            .copied()
            .chain(core::iter::once(self.root))
    }

    /// The chain the reference hands back — every enclosing loop BUT the root.
    #[must_use]
    pub fn below_root(&self) -> &[&'a LoopNode] {
        &self.below_root
    }
}

/// Replaces: e062_getEnclosingLoopsAndRelatedDims
///
/// THE ENCLOSING LOOP CHAIN WITHOUT ITS ROOT, plus every dim those loops walk together with the
/// node's own — an allocation's `layoutDimOrder_`, a transfer's lds layout, nothing for a compute.
///
/// ⛔ THE ROOT LOOP'S DIMS STAY IN THE SET even though its loop leaves the chain. ⛔ AND A LAYOUT DIM
/// ENTERS AS `{dim, Unpadded}` through `PrimaryDimAndKind`'s implicit constructor, so it can never
/// match a `WindowDim` loop entry and the set may hold one dim under two kinds.
/// ⚠️ A TRANSFER TAKES ITS SOURCE'S lds, else the FIRST DESTINATION that states one.
pub fn get_enclosing_loops_and_related_dims<'a, D: Dsc + ?Sized>(
    node: Node<'a>,
    dsc: &D,
    loops: &OwnerLoops<'a>,
) -> (Vec<&'a LoopNode>, BTreeSet<PrimaryDimAndKind>) {
    let mut related: BTreeSet<PrimaryDimAndKind> =
        loops.iter().flat_map(|owner| owner.dims.iter()).collect();

    let layout = match node {
        Node::Allocate(alloc) => Some(alloc.layout.dims()),
        Node::Transfer(transfer) => transfer
            .src
            .data
            .my_lds_idx
            .or_else(|| transfer.dsts.iter().find_map(|dst| dst.data.my_lds_idx))
            .map(|lds| dsc.layout_dims(lds)),
        Node::Compute(_) => None,
    };
    if let Some(dims) = layout {
        related.extend(dims.iter().map(|dim| PrimaryDimAndKind {
            dim,
            kind: MetaDimKind::Unpadded,
        }));
    }

    (loops.below_root().to_vec(), related)
}

/// WHERE A LOOP SITS RELATIVE TO THE CHUNK BOUNDARY — `LoopDistributionInfo::LoopDistributionCat`
/// (`dsc/dsc2.h:1138`) less `UNKNOWN`, which is only the field's unset default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopDistribution {
    /// `ABOVE_CHUNK`.
    AboveChunk,
    /// `BELOW_CHUNK`.
    BelowChunk,
    /// `CORELET_SLICE`.
    CoreletSlice,
}

/// ONE LOOP, THE DIM OF IT THAT MATCHED AND WHERE IT SITS — `LoopDistributionInfo`
/// (`dsc/dsc2.h:1137`); a `VectorOfLoopAndDim` (`:1172`) is a [`Vec`] of these.
///
/// ⚠️ dsc2 VOCABULARY HOMED HERE because [`LoopNode`] is; moving both belongs to whichever batch
/// first needs this in a second module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopAndDim<'a> {
    /// `loopNode`.
    pub loop_node: &'a LoopNode,
    /// `dimAndKind` — the loop's OWN entry that matched, kind included.
    pub dim: PrimaryDimAndKind,
    /// `cat`.
    pub distribution: LoopDistribution,
}

/// HOW A COORDINATE READS ITS DIM, WITH THE PADDING THAT ARM NEEDS — `accessPadType` FUSED to
/// `dataStageParam_.at(loop->denId_).ss_.paddingSizes_`.
///
/// ⛔ THAT `.at()` THROW IS WHY THE TWO ARE ONE ARGUMENT: the reference reads the map only inside its
/// `accessPadType != NOPAD` arm, so [`Self::NoPad`] cannot reach it. WHICH other [`PadType`] it is
/// is never asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPad<'a> {
    /// [`PadType::NoPad`].
    NoPad,
    /// Any other [`PadType`], together with the DENOMINATOR stage's `paddingSizes_`.
    Padded(&'a BTreeMap<PrimaryDim, DimPadding>),
}

/// Replaces: e063_findAndStoreLoopWithDim
///
/// APPENDS THE LOOP to `related_loops` once per dim entry of it that IS the sought dim, or that is a
/// `WindowDim` the sought dim's own padding windows.
///
/// ⛔ `dimAndKind.dim_ == dimToFind` COMPARES A WHOLE `PrimaryDimAndKind` WITH A BARE DIM
/// (`dsc/dims.h:79-81`), so the implicit constructor ALSO demands the sought kind be `Unpadded` and
/// DISCARDS the loop's own `kind_` — which the dsc2 twin `isLoopDimRelated` (`dsc/dsc2.cpp:6550`)
/// does not; the only callsite passes a bare dim and says so (`:7405-7406`). ⛔ AND `relatedDims` IS
/// NEVER READ: the reference takes the set and consults it nowhere.
pub fn find_and_store_loop_with_dim<'a>(
    to_find: PrimaryDimAndKind,
    loop_node: &'a LoopNode,
    pad: AccessPad<'_>,
    related_loops: &mut Vec<LoopAndDim<'a>>,
) {
    for dim in loop_node.dims.iter() {
        let related = if dim.dim == to_find.dim && to_find.kind == MetaDimKind::Unpadded {
            true
        } else if let AccessPad::Padded(padding) = pad {
            dim.kind == MetaDimKind::WindowDim
                && padding
                    .get(&to_find.dim)
                    .is_some_and(|entry| entry.window_dim == Some(dim.dim))
        } else {
            false
        };
        if related {
            related_loops.push(LoopAndDim {
                loop_node,
                dim,
                distribution: LoopDistribution::AboveChunk,
            });
        }
    }
}

/// Replaces: e064_constructDatastage
///
/// Mints a COPY of a reference data stage in the DSC under the first free id and renames its two
/// halves `<id>` and `<id>el`.
///
/// ⭐ THE SAME OPERATION AS `e113_constructDatastage` (`ddc/ddc_transformation_util.cpp:126`), down
/// to the `size()`-seeded id search, so this CALLS that port rather than spelling it twice.
pub fn construct_datastage<D: Clone>(
    stages: &mut DataStages<D>,
    reference: &DataStage<D>,
) -> DatastageId {
    construct_datastage_from(stages, reference)
}

#[cfg(test)]
mod tests_e057_e064 {
    use super::*;
    use crate::schedule::dsc2::{
        AllocLayout as Dsc2AllocLayout, AllocPlacement, Coordinate, CoordinateCategory,
        FoldCardinality, FoldCoeff, FoldLabel, LayoutDims, MaxDimSize, StartAddress,
    };

    /// One labelled DS's `memOrg_` as this batch reads it, stated by field.
    #[derive(Default)]
    struct HbmStub {
        pinned: bool,
        allocation: Option<NodeName>,
        indirection: Option<IndirectAlloc>,
        layout: Option<LayoutDims>,
        pages: BTreeSet<PrimaryDim>,
    }

    impl MemOrg for HbmStub {
        fn hbm_pinned(&self) -> bool {
            self.pinned
        }

        fn lx_buffering(&self) -> Option<Buffering> {
            None
        }

        fn lx_start_address(&self, _at: &AddressCoord) -> Option<ByteAddress> {
            None
        }

        fn lx_buffer_offset(&self, _core: Core, _corelet: Corelet) -> Option<BufferOffset> {
            None
        }

        fn lx_zero_padded(&self) -> Option<bool> {
            Some(false)
        }

        fn hbm_indirection(&self) -> Option<IndirectAlloc> {
            self.indirection
        }

        fn hbm_allocation(&self) -> Option<NodeName> {
            self.allocation.clone()
        }

        fn hbm_layout_dims(&self) -> Option<LayoutDims> {
            self.layout.clone()
        }

        fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim> {
            self.pages.clone()
        }

        fn lx_padding(&self) -> Option<PaddingForm> {
            None
        }

        fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent> {
            BTreeMap::new()
        }

        fn hbm_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }

        fn lx_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }
    }

    fn allocate(name: &str, dim: PrimaryDim) -> AllocateNode {
        AllocateNode {
            name: NodeName(name.to_owned()),
            component: SenComponent::Hbm,
            lds: Some(LdsIdx(0)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: Dsc2AllocLayout::new((dim, MaxDimSize::Unset), Vec::new()),
            start_address: StartAddress::default(),
            placement: AllocPlacement::default(),
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        }
    }

    fn loop_over(first: PrimaryDimAndKind, rest: Vec<PrimaryDimAndKind>) -> LoopNode {
        LoopNode {
            name: NodeName("loop_ds0_ds1".to_owned()),
            num: DatastageId(0),
            den: DatastageId(1),
            dims: LoopDims::new(first, rest),
        }
    }

    /// A DSC whose every lds lays out one dim — which e062's allocate arm never asks for.
    struct OneDim;

    impl Dsc for OneDim {
        fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
            LayoutDims::new(PrimaryDim::In, Vec::new())
        }
    }

    /// e057 — a value tensor is paged whether or not the entry states `isPresent`, and its twin's
    /// index tensor is not.
    #[test]
    fn a_paged_lds_is_a_value_tensor_pinned_or_not() {
        assert!(is_paged_lds(&HbmStub {
            indirection: Some(IndirectAlloc::ValueTensor),
            ..HbmStub::default()
        }));
        assert!(!is_paged_lds(&HbmStub {
            pinned: true,
            indirection: Some(IndirectAlloc::IndexTensor(IndexTensor::Address)),
            ..HbmStub::default()
        }));
        assert!(!is_paged_lds(&HbmStub::default()));
    }

    /// e058 — the search starts above the chunk stage and clears the SIBLING DSCs too, then MINTS.
    #[test]
    fn a_new_data_stage_index_clears_every_dsc_of_the_super_dsc() {
        let mut dsc: DataStages<()> = DataStages::default();
        dsc.0.insert(DatastageId(2), DataStage::default());
        let mut sibling: DataStages<()> = DataStages::default();
        sibling.0.insert(DatastageId(3), DataStage::default());

        let id = get_new_data_stage_index(&mut dsc, &[&sibling]);
        assert_eq!(id, DatastageId(4));
        assert!(dsc.0.contains_key(&id));
    }

    /// e059 — layout order decides the order, the pages decide membership, and only an index tensor
    /// is walked at all.
    #[test]
    fn paged_dimensions_are_the_index_layout_against_its_pages() {
        let index = HbmStub {
            indirection: Some(IndirectAlloc::IndexTensor(IndexTensor::Address)),
            layout: Some(LayoutDims::new(
                PrimaryDim::Out,
                vec![PrimaryDim::In, PrimaryDim::Ki],
            )),
            pages: BTreeSet::from([PrimaryDim::In, PrimaryDim::Ki]),
            ..HbmStub::default()
        };
        let value = HbmStub {
            indirection: Some(IndirectAlloc::ValueTensor),
            layout: Some(LayoutDims::new(PrimaryDim::Kj, Vec::new())),
            pages: BTreeSet::from([PrimaryDim::Kj]),
            ..HbmStub::default()
        };
        assert_eq!(
            get_paged_dimensions(&[&index, &value, &index]),
            vec![PrimaryDim::In, PrimaryDim::Ki]
        );
        assert!(get_paged_dimensions(&[&HbmStub::default()]).is_empty());
    }

    /// e060 — only a PINNED entry answers, and a pinned one holding no node is the refusal.
    #[test]
    fn hbm_allocations_are_the_pinned_ones() {
        let pinned = HbmStub {
            pinned: true,
            allocation: Some(NodeName("pinned".to_owned())),
            ..HbmStub::default()
        };
        let unpinned = HbmStub {
            allocation: Some(NodeName("unpinned".to_owned())),
            ..HbmStub::default()
        };
        assert_eq!(
            get_hbm_allocations(&[&unpinned, &pinned]),
            Some(vec![NodeName("pinned".to_owned())])
        );
        assert_eq!(
            get_hbm_allocations(&[&HbmStub {
                pinned: true,
                ..HbmStub::default()
            }]),
            None
        );
    }

    /// e061 — every fold of the dim, position 0 first, APPENDED to what the caller already gathered.
    #[test]
    fn fold_params_are_the_whole_dim_appended() {
        let mut coord = Coordinate::default();
        coord.add_fold_front(
            PrimaryDim::In,
            CoordinateCategory::Temporal,
            FoldCardinality(4),
            FoldLabel("inner".to_owned()),
            FoldCoeff(1),
            FoldCoeff(0),
        );
        coord.add_fold_front(
            PrimaryDim::In,
            CoordinateCategory::Spatial,
            FoldCardinality(2),
            FoldLabel("outer".to_owned()),
            FoldCoeff(3),
            FoldCoeff(5),
        );

        let mut params = vec![Fold {
            cardinality: FoldCardinality(9),
            label: FoldLabel("already there".to_owned()),
            alpha: FoldCoeff(0),
            beta: FoldCoeff(0),
        }];
        gather_fold_params(
            coord.fold_dim(PrimaryDim::In).expect("the dim was folded"),
            &mut params,
        );

        assert_eq!(params.len(), 3);
        assert_eq!(params[1].label, FoldLabel("outer".to_owned()));
        assert_eq!(
            (params[1].alpha, params[1].beta),
            (FoldCoeff(3), FoldCoeff(5))
        );
        assert_eq!(params[2].cardinality, FoldCardinality(4));
    }

    /// e062 — the root leaves the CHAIN but not the DIM SET, and a layout dim arrives `Unpadded`.
    #[test]
    fn enclosing_loops_drop_the_root_and_keep_its_dims() {
        assert!(OwnerLoops::of(Vec::new()).is_none());

        let window = PrimaryDimAndKind {
            dim: PrimaryDim::Out,
            kind: MetaDimKind::WindowDim,
        };
        let padded = PrimaryDimAndKind {
            dim: PrimaryDim::In,
            kind: MetaDimKind::Padded,
        };
        let inner = loop_over(window, Vec::new());
        let root = loop_over(padded, Vec::new());
        let loops = OwnerLoops::of(vec![&inner, &root]).expect("a loop encloses the node");
        let node = allocate("a", PrimaryDim::Out);

        let (chain, related) =
            get_enclosing_loops_and_related_dims(Node::Allocate(&node), &OneDim, &loops);
        assert_eq!(chain, vec![&inner]);
        assert_eq!(
            related,
            BTreeSet::from([
                window,
                padded,
                PrimaryDimAndKind {
                    dim: PrimaryDim::Out,
                    kind: MetaDimKind::Unpadded,
                },
            ])
        );
    }

    /// e063 — a bare dim matches only an `Unpadded` entry, and the padded arm reaches the window dim.
    #[test]
    fn a_bare_dim_matches_unpadded_and_the_padding_reaches_the_window() {
        let node = loop_over(
            PrimaryDimAndKind {
                dim: PrimaryDim::In,
                kind: MetaDimKind::Unpadded,
            },
            vec![PrimaryDimAndKind {
                dim: PrimaryDim::Out,
                kind: MetaDimKind::WindowDim,
            }],
        );
        let bare = PrimaryDimAndKind {
            dim: PrimaryDim::In,
            kind: MetaDimKind::Unpadded,
        };
        let padded = PrimaryDimAndKind {
            dim: PrimaryDim::In,
            kind: MetaDimKind::Padded,
        };

        let mut related = Vec::new();
        find_and_store_loop_with_dim(bare, &node, AccessPad::NoPad, &mut related);
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].dim, bare);
        assert_eq!(related[0].distribution, LoopDistribution::AboveChunk);

        related.clear();
        find_and_store_loop_with_dim(padded, &node, AccessPad::NoPad, &mut related);
        assert!(related.is_empty());

        let mut padding = BTreeMap::new();
        padding.insert(
            PrimaryDim::In,
            DimPadding {
                window_dim: Some(PrimaryDim::Out),
                ..DimPadding::default()
            },
        );
        related.clear();
        find_and_store_loop_with_dim(padded, &node, AccessPad::Padded(&padding), &mut related);
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].dim.dim, PrimaryDim::Out);
    }

    /// e064 — the copy lands under the size-seeded free id and only the two names are overwritten.
    #[test]
    fn a_constructed_datastage_is_a_renamed_copy() {
        let mut stages: DataStages<u32> = DataStages::default();
        stages.0.insert(DatastageId(0), DataStage::default());
        let mut reference = DataStage::<u32>::default();
        reference.ss.name = StageName("reference".to_owned());
        reference.ss.dims = 7;

        let id = construct_datastage(&mut stages, &reference);
        assert_eq!(id, DatastageId(1));
        let minted = &stages.0[&id];
        assert_eq!(minted.ss.name, StageName("1".to_owned()));
        assert_eq!(minted.el.name, StageName("1el".to_owned()));
        assert_eq!(minted.ss.dims, 7);
    }
}

/// Replaces: e065_constructLoopNode
///
/// MINTS THE LOOP NODE for a numerator/denominator data-stage pair, named `loop_ds<num>_ds<den>`
/// then `_<dim>` over its dims in order (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7737`).
///
/// ⭐ ONE IMPLEMENTATION, NOT TWO: this body is byte-identical to `Ddc::constructLoopNode`
/// (`ddc/ddc_transformation_util.cpp:138`) less that one's never-read `baseNode`, so it IS entry
/// 114 and delegates to it rather than spelling the naming rule a second time.
///
/// ⛔ *"Cannot construct loop with no dimensions"* is [`LoopDims`], and the mint is UNPARENTED —
/// placing it in the tree is the caller's own step.
#[must_use]
pub fn construct_loop_node(num: DatastageId, den: DatastageId, dims: LoopDims) -> LoopNode {
    crate::schedule::ddc::transformation_util::construct_loop_node(num, den, dims)
}

/// WHICH REDUCE SLICE OF A GROUP A CORE TAKES — `addCore`'s `slice`, the index into `coreIds`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:29`), which `getCrossCoreReductionGroupInfo` forms as
/// a mixed-radix number over the REDUCED dims alone (`L3DlOpsScheduler.cpp:2760-2766`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReduceSlice(pub u32);

/// WHICH COreLET AN END QUERY NAMES — `coreletId`, whose only two values are `0` and `1`; anything
/// else is *"Unknown corelet id."* (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:34-49`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupCorelet {
    /// `coreletId == 0`.
    Zero,
    /// `coreletId == 1`.
    One,
}

impl GroupCorelet {
    /// The corelet an index names — TOTAL, because [`Corelet`] admits exactly two indices on every
    /// arch in the tree, which is what the const assertion below states.
    #[must_use]
    pub const fn of(corelet: Corelet) -> Self {
        if corelet.get() == 0 {
            Self::Zero
        } else {
            Self::One
        }
    }
}

/// *"Unknown corelet id."* MADE UNSPELLABLE — the one caller that varies the corelet walks
/// `0..numCoreletsUsed_DSC2_` (`L3DlOpsScheduler.cpp:2780`) and the other passes a FIXED
/// `constexpr int corelet1Id = 1` (`:5827`, `:5831`), while `CORELETS_PER_CORE` is `2` on both
/// generations (`src/arch.rs:281`, `:315`). A third corelet would fail HERE, not take arm two.
const _: () = {
    assert!(Corelet::checked(1).is_some());
    assert!(Corelet::checked(2).is_none());
};

/// ONE CROSS-CORE REDUCTION GROUP — `CrossCoreReductionGroup` (`L3DlOpsScheduler.h:24`): the cores
/// of one group indexed by [`ReduceSlice`], with the reference's `-1` for a slice no core took.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrossCoreReductionGroup {
    core_ids: Vec<Option<Core>>,
}

impl CrossCoreReductionGroup {
    /// Replaces: e066_addCore
    ///
    /// PLACES `core` AT ITS REDUCE SLICE, growing the group to fit and leaving `-1` holes behind.
    ///
    /// ⛔ DELIBERATE DIVERGENCE — `coreIds.resize(slice + 1, -1)` ALSO SHRINKS. A core arriving at a
    /// lower slice than one already placed TRUNCATES the group and drops it silently, and that is
    /// reachable: `coreIdToWkSlice_` iterates by core id (`dsc/superdsc.h:70`) while the reduce
    /// slice counts a different set of dims, so the two orders need not agree. `back()` — the core
    /// [`ReductionGroupCores::end_core_at_corelet`] hands a sync — would then be a truncated tail.
    /// Growing only is the *"Use slice ids to order them"* the comment states (`:2754`).
    pub fn add_core(&mut self, core: Core, slice: ReduceSlice) {
        let slot = slice.0 as usize;
        if self.core_ids.len() <= slot {
            self.core_ids.resize(slot + 1, None);
        }
        if let Some(placed) = self.core_ids.get_mut(slot) {
            *placed = Some(core);
        }
    }

    /// `getCores()` (`:33`) WITH `isEmpty()` DISCHARGED (`:52`) — the non-empty view both end
    /// queries need, or [`None`] for their `DT_CHECK(!coreIds.empty())`. A group is genuinely
    /// empty when no work slice lands in it: `getCrossCoreReductionGroupInfo` default-constructs
    /// `numGroups` of them and fills only the ones cores map to (`:2755-2767`).
    #[must_use]
    pub fn cores(&self) -> Option<ReductionGroupCores<'_>> {
        (!self.core_ids.is_empty()).then_some(ReductionGroupCores(&self.core_ids))
    }
}

/// A CROSS-CORE REDUCTION GROUP WITH A SLICE IN IT — `DT_CHECK(!coreIds.empty())`
/// (`L3DlOpsScheduler.h:35`, `:44`) as a type, so neither end query can refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReductionGroupCores<'a>(&'a [Option<Core>]);

impl ReductionGroupCores<'_> {
    /// Replaces: e067_getStartCoreAtCorelet
    ///
    /// THE CORE THIS CORELET STARTS THE GROUP AT — `coreIds.front()` for corelet 0 and
    /// `coreIds.back()` for corelet 1 (`L3DlOpsScheduler.h:34`), the two corelets walking the
    /// group's slices in opposite directions.
    ///
    /// ⛔ [`None`] IS THE `-1` HOLE AND NOT AN ABORT: [`CrossCoreReductionGroup::add_core`] fills
    /// only the slices work landed on, and the reference returns that `-1` as if it were a core.
    /// ⚠️ DEAD IN THE REFERENCE: defined at `.h:34` and called from nowhere in the tree, unlike
    /// [`Self::end_core_at_corelet`]. Ported because the corelet symmetry is the pair's contract.
    #[must_use]
    pub fn start_core_at_corelet(&self, corelet: GroupCorelet) -> Option<Core> {
        match corelet {
            GroupCorelet::Zero => self.front(),
            GroupCorelet::One => self.back(),
        }
    }

    /// Replaces: e068_getEndCoreAtCorelet
    ///
    /// THE CORE THIS CORELET ENDS THE GROUP AT — the OTHER end from
    /// [`Self::start_core_at_corelet`]: `back()` for corelet 0, `front()` for corelet 1 (`:43`).
    ///
    /// ⛔ [`None`] IS THE `-1` HOLE, as it is for the start.
    #[must_use]
    pub fn end_core_at_corelet(&self, corelet: GroupCorelet) -> Option<Core> {
        match corelet {
            GroupCorelet::Zero => self.back(),
            GroupCorelet::One => self.front(),
        }
    }

    /// `coreIds.front()`, hole and all.
    fn front(&self) -> Option<Core> {
        self.0.first().copied().flatten()
    }

    /// `coreIds.back()`, hole and all.
    fn back(&self) -> Option<Core> {
        self.0.last().copied().flatten()
    }
}

/// ONE `Metadata::Datastage::Constraints` AS THE L3 SCHEDULER DECLARES IT
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:113`).
///
/// ⛔ NOT DDC'S [`StoredConstraint`](crate::schedule::ddc::metadata::StoredConstraint): the L3 copy
/// has NO `loopDimKind_` and NO `cannotBeSymbolic_` (`ddc/ddc_metadata.h:36`, `:39`), so it cannot
/// carry a [`LoopMultiple`](crate::schedule::ddc::metadata::LoopMultiple) and the two are distinct
/// types rather than one shared with two spellings.
///
/// ⚠️ TRAP: `L3DlOpsScheduler.cpp` NEVER READS THIS HALF OF ITS OWN `Metadata` — `dscMetadata` is
/// *"only used for memory allocation"* (`:202`), and `constraints_`, `mustBeMultiple_`,
/// `strategyMinimize_` and all three updaters occur nowhere in its 8,033 lines.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Constraints {
    /// `mustBeMultiple_` — a multiple of the reference data stage where there is one, else of `min_`.
    pub must_be_multiple: bool,
    /// `min_`.
    pub min: Option<f32>,
    /// `max_`.
    pub max: Option<f32>,
    /// `values_` — absent is UNCONSTRAINED, where an ENGAGED EMPTY set admits no size at all.
    pub values: Option<Vec<f32>>,
}

impl Constraints {
    /// Replaces: e069_updateMin
    ///
    /// TIGHTENS THE LOWER BOUND — `min_ = min_ ? std::max(*min_, newVal) : newVal`
    /// (`L3DlOpsScheduler.h:117`).
    ///
    /// ⛔ `max` TIGHTENS A *MIN*: the stricter of two lower bounds is the larger one.
    pub fn update_min(&mut self, new_val: f32) {
        self.min = Some(self.min.map_or(new_val, |min| stricter_min(min, new_val)));
    }

    /// Replaces: e070_updateMax
    ///
    /// TIGHTENS THE UPPER BOUND — `max_ = max_ ? std::min(*max_, newVal) : newVal` (`:120`).
    /// ⛔ `min` TIGHTENS A *MAX*.
    pub fn update_max(&mut self, new_val: f32) {
        self.max = Some(self.max.map_or(new_val, |max| stricter_max(max, new_val)));
    }

    /// Replaces: e071_updateValues
    ///
    /// INTERSECTS THE PERMITTED VALUES — `values_ = values_ ? set_intersect(*values_, newVals) :
    /// newVals` (`:123`, `util/utils.h:112`).
    ///
    /// ⛔ THE FIRST CALL ADOPTS, IT DOES NOT INTERSECT — an absent `values_` is unconstrained,
    /// where an intersection may leave it engaged and EMPTY. Two states, and only the second is a
    /// contradiction.
    pub fn update_values(&mut self, new_vals: &[f32]) {
        let mut incoming = new_vals.to_vec();
        incoming.sort_by(f32::total_cmp);
        incoming.dedup();
        self.values = Some(match &self.values {
            Some(values) => values
                .iter()
                .copied()
                .filter(|value| incoming.contains(value))
                .collect(),
            None => incoming,
        });
    }
}

/// WHAT EXTENT A DATA STAGE'S DIMS STATE FOR ONE DIM — `primaryDimToVal_st(d)`
/// (`dsc/dims.cpp:647`), which is all `getTripCount` asks of a `DataStructDims`.
pub trait DimExtents {
    /// The extent, or [`None`] for the `-1` an unstated dim answers.
    fn extent(&self, dim: PrimaryDim) -> Option<Extent>;
}

/// WHERE ENTRY 072 READS THE TWO STAGES IT DIVIDES — `dataStageParam_.at(id).ss_` for one dim,
/// whichever carrier holds the stages.
pub trait StageExtents {
    /// `dataStageParam_.at(stage).ss_.primaryDimToVal_st(dim)`, [`None`] for the `.at()` throw and
    /// for the `-1` an unstated dim answers.
    fn stage_extent(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Extent>;
}

impl<D: DimExtents> StageExtents for DataStages<D> {
    fn stage_extent(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Extent> {
        self.0.get(&stage)?.ss.dims.extent(dim)
    }
}

impl StageExtents for L3DataStages {
    fn stage_extent(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Extent> {
        self.at(stage)?.ss_extent(dim)
    }
}

/// HOW MANY TIMES A LOOP WALKS ONE DIM — `getTripCount`'s `int`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:487`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TripCount(u64);

impl TripCount {
    /// The count, for the `numRepeats *=` products the callers form (`L3DlOpsScheduler.cpp:1841`).
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Replaces: e072_getTripCount
///
/// THE TRIP COUNT OF LOOP `num`/`den` ON `dim` — `ceil(num/den)` over the two data stages'
/// STEADY-STATE extents (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:487`).
///
/// ⛔ [`None`] IS BOTH `dataStageParam_.at()` THROWS AND EVERY STATE THE REFERENCE COMPUTES A
/// NON-COUNT FROM. An unstated dim is `-1`, so `ceil(-1/den)` is a SILENT ZERO that the callers
/// multiply straight into `numRepeats` (`:1841`); and `den == 0` divides by zero and then casts an
/// infinity to `int`, which is undefined. A loop dim needs a positive extent on both sides.
#[must_use]
pub fn trip_count<S: StageExtents + ?Sized>(
    stages: &S,
    dim: PrimaryDim,
    num: DatastageId,
    den: DatastageId,
) -> Option<TripCount> {
    let positive = |id: DatastageId| -> Option<u64> {
        let extent = stages.stage_extent(id, dim)?;
        u64::try_from(extent.0).ok().filter(|extent| *extent > 0)
    };
    Some(TripCount(positive(num)?.div_ceil(positive(den)?)))
}

#[cfg(test)]
mod tests_e065_e072 {
    use super::*;
    use crate::schedule::ddc::transformation_util::StageName;

    /// A `DataStructDims` STAND-IN — the extents it states, which is every question entry 072 puts
    /// to one.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct Extents(BTreeMap<PrimaryDim, Extent>);

    impl DimExtents for Extents {
        fn extent(&self, dim: PrimaryDim) -> Option<Extent> {
            self.0.get(&dim).copied()
        }
    }

    fn core(index: u32) -> Core {
        Core::checked(index).expect("this arch has the core the test names")
    }

    fn stage(id: DatastageId, extents: &[(PrimaryDim, i64)]) -> (DatastageId, DataStage<Extents>) {
        let dims = Extents(
            extents
                .iter()
                .map(|(dim, extent)| (*dim, Extent(*extent)))
                .collect(),
        );
        (
            id,
            DataStage {
                ss: StageDims {
                    name: StageName(id.0.to_string()),
                    dims,
                },
                el: StageDims::default(),
            },
        )
    }

    /// e065 — the mint's name is the stage pair followed by its dims, in order.
    #[test]
    fn a_loop_node_is_named_after_its_stage_pair_and_then_its_dims() {
        let node = construct_loop_node(
            DatastageId(1),
            DatastageId(0),
            LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::Y,
                    kind: MetaDimKind::Unpadded,
                },
                vec![PrimaryDimAndKind {
                    dim: PrimaryDim::Ki,
                    kind: MetaDimKind::WindowDim,
                }],
            ),
        );

        assert_eq!(node.name, NodeName("loop_ds1_ds0_y_ki".to_owned()));
        assert_eq!(node.num, DatastageId(1));
        assert_eq!(node.den, DatastageId(0));
    }

    /// e066 — a slice is a slot: the group grows to fit, keeps its holes, and a core arriving at a
    /// LOWER slice does not truncate what is already placed (the `resize` divergence).
    #[test]
    fn a_group_places_each_core_at_its_slice_and_never_shrinks() {
        let mut group = CrossCoreReductionGroup::default();
        group.add_core(core(7), ReduceSlice(3));
        group.add_core(core(4), ReduceSlice(1));

        let cores = group.cores().expect("two cores were placed");
        // Slice 0 is a hole, slice 3 survived the lower arrival.
        assert_eq!(cores.start_core_at_corelet(GroupCorelet::Zero), None);
        assert_eq!(cores.end_core_at_corelet(GroupCorelet::Zero), Some(core(7)));

        group.add_core(core(2), ReduceSlice(0));
        let cores = group.cores().expect("three cores were placed");
        assert_eq!(
            cores.start_core_at_corelet(GroupCorelet::Zero),
            Some(core(2))
        );
        assert_eq!(cores.end_core_at_corelet(GroupCorelet::Zero), Some(core(7)));
    }

    /// e067/e068 — the two corelets read the group from opposite ends, and an empty group yields no
    /// view to ask at all.
    #[test]
    fn the_two_corelets_walk_the_group_from_opposite_ends() {
        assert_eq!(CrossCoreReductionGroup::default().cores(), None);

        let mut group = CrossCoreReductionGroup::default();
        group.add_core(core(1), ReduceSlice(0));
        group.add_core(core(5), ReduceSlice(1));
        let cores = group.cores().expect("two cores were placed");

        let zero = GroupCorelet::of(Corelet::checked(0).expect("corelet 0"));
        let one = GroupCorelet::of(Corelet::checked(1).expect("corelet 1"));
        assert_eq!(cores.start_core_at_corelet(zero), Some(core(1)));
        assert_eq!(cores.end_core_at_corelet(zero), Some(core(5)));
        // Corelet 1 starts where corelet 0 ends, and ends where it starts.
        assert_eq!(cores.start_core_at_corelet(one), Some(core(5)));
        assert_eq!(cores.end_core_at_corelet(one), Some(core(1)));
    }

    /// e069/e070/e071 — the first update ADOPTS and every later one NARROWS: `max` on the min,
    /// `min` on the max, intersection on the values.
    #[test]
    fn updating_a_constraint_adopts_first_and_narrows_after() {
        let mut constraints = Constraints::default();
        constraints.update_min(4.0);
        constraints.update_max(64.0);
        constraints.update_values(&[1.0, 2.0, 4.0]);
        assert_eq!(constraints.min, Some(4.0));
        assert_eq!(constraints.max, Some(64.0));
        assert_eq!(
            constraints.values.as_deref(),
            Some([1.0, 2.0, 4.0].as_slice())
        );

        constraints.update_min(8.0);
        constraints.update_max(32.0);
        constraints.update_values(&[2.0, 4.0, 8.0]);
        assert_eq!(constraints.min, Some(8.0));
        assert_eq!(constraints.max, Some(32.0));
        assert_eq!(constraints.values.as_deref(), Some([2.0, 4.0].as_slice()));

        // A looser bound changes nothing, and a disjoint value set leaves the set ENGAGED AND
        // EMPTY — which no size satisfies, and is not the same state as absent.
        constraints.update_min(2.0);
        constraints.update_values(&[16.0]);
        assert_eq!(constraints.min, Some(8.0));
        assert_eq!(constraints.values.as_deref(), Some([].as_slice()));
    }

    /// e072 — the count is the CEILING of the two stages' extents, and a stage or extent the DSC
    /// does not state has no count rather than a fabricated one.
    #[test]
    fn a_trip_count_is_the_ceiling_of_the_two_stages_extents() {
        let stages = DataStages(
            [
                stage(DatastageId(0), &[(PrimaryDim::Y, 100), (PrimaryDim::X, 8)]),
                stage(DatastageId(1), &[(PrimaryDim::Y, 32), (PrimaryDim::X, 0)]),
            ]
            .into_iter()
            .collect(),
        );

        // 100 / 32 rounds UP to 4, and the pair the other way round is one trip.
        let count = trip_count(&stages, PrimaryDim::Y, DatastageId(0), DatastageId(1));
        assert_eq!(count.map(TripCount::get), Some(4));
        let count = trip_count(&stages, PrimaryDim::Y, DatastageId(1), DatastageId(0));
        assert_eq!(count.map(TripCount::get), Some(1));

        // A zero denominator is the infinity cast, an unstated dim is the `-1`, and an absent stage
        // is the `.at()` throw.
        assert_eq!(
            trip_count(&stages, PrimaryDim::X, DatastageId(0), DatastageId(1)),
            None
        );
        assert_eq!(
            trip_count(&stages, PrimaryDim::J, DatastageId(0), DatastageId(1)),
            None
        );
        assert_eq!(
            trip_count(&stages, PrimaryDim::Y, DatastageId(0), DatastageId(9)),
            None
        );
    }
}

/// Replaces: e197_getCoreletSplitDimensions
///
/// Which dims the DSC's work is split across CORELETS along — every dim but the combined `IJ` and
/// `KIJ` that [`is_dimension_corelet_split`] answers for.
///
/// ⭐ THE `numCoreletsUsed_ > 1` GUARD IS REDUNDANT (it is the predicate's own first term), and A SET
/// FOR ITS VECTOR CHANGES NOTHING: it walks the KEYS of a `std::map` (`dsc/dims.cpp:22`), so the
/// answer is already each dim once in ordinal order, and NEITHER CALLER CAN SEE THE DIFFERENCE — one
/// iterates it (`:109`), the other hands it to `fillFinalStartAddressAndOffset` (`:5288`), which only
/// membership-tests it (`DCGUtils::isValPresent`, `:4995`). `PrimaryDimTypesCount` is not a
/// [`PrimaryDim`], so the third skip has no input.
#[must_use]
pub fn corelet_split_dimensions(dsc: &DesignSpaceConfig) -> BTreeSet<PrimaryDim> {
    PrimaryDim::ALL
        .into_iter()
        .filter(|dim| !matches!(dim, PrimaryDim::Ij | PrimaryDim::Kij))
        .filter(|dim| is_dimension_corelet_split(dsc, *dim))
        .collect()
}

/// Replaces: e198_addOrUpdatePaddingSizesInChunkParams
///
/// COPIES the core stage's whole `paddingSizes_` onto the chunk stage, then VOIDS it on every dim
/// chunking moved — [`void_padding_if_chunking`] under the same const flag, which stays ONE fact.
///
/// ⛔ BOTH *"Expect non-empty data-stage parameters."* `DT_CHECK`s ARE [`FilledDims`], as they are for
/// entry 005. ⭐ TRAP: THE ASSIGNMENT REPLACES the map rather than merging into it, so a dim the chunk
/// stage padded and the core stage does not LOSES its padding outright — the name says *addOrUpdate*
/// and the body does neither.
pub fn add_or_update_padding_sizes_in_chunk_params<const CARRY_UNNEEDED_PAD: bool>(
    chunk_params: &mut FilledDims,
    core_params: &FilledDims,
) {
    *chunk_params.padding_mut() = core_params.dims().padding.clone();
    void_padding_if_chunking::<CARRY_UNNEEDED_PAD>(chunk_params, core_params);
}

/// Replaces: e199_getLabeledDsNumOfWkSlices
///
/// HOW MANY DISTINCT WORK SLICES cover one labelled DS: the PRODUCT of its non-broadcast dims' slice
/// counts when `dscs` names as many DSCs as the super-DSC has, else the number of distinct per-core
/// slices over the group's cores.
///
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: that fast path is chosen on a SIZE comparison
/// (`dscIndices.size() == mySDsc.dscs_.size()`), so a group naming one DSC twice reaches it while
/// covering half of them. ⭐ `emplace` KEEPS THE FIRST id a REPEATED layout dim states. ⛔ [`None`] is
/// every abort: both dim-list checks, `numWkSlicesPerDim_.at`, `coreIdToWkSlice_.at`, `.at(dim)`, and
/// the `unsigned` product wrapping.
#[must_use]
pub fn labeled_ds_num_of_wk_slices(
    sdsc: &SuperDsc,
    lds: LdsIdx,
    dscs: &DscGroup<'_>,
) -> Option<WkSliceCount> {
    let main = dscs.main();
    if dscs.iter().count() == sdsc.dscs().iter().count() {
        let mut count = WkSliceCount::ONE;
        for dim in main.non_broadcast_lds_dims(lds)? {
            count = count.times(*sdsc.num_wk_slices_per_dim.get(&dim)?)?;
        }
        return Some(count);
    }
    let processing: BTreeSet<Core> = dscs
        .iter()
        .flat_map(|dsc| dsc.core_ids_used.iter())
        .collect();
    let entry = main.labeled_ds.at(lds)?;
    let layout = main.layout_dims.get(&lds)?;
    let mut slices: BTreeSet<BTreeMap<PrimaryDim, WkSliceId>> = BTreeSet::new();
    for core in processing {
        let mut slice: BTreeMap<PrimaryDim, WkSliceId> = BTreeMap::new();
        for dim in layout.iter() {
            let id = if is_labeled_ds_dimension_broadcast(entry, dim)? {
                WkSliceId(0)
            } else {
                sdsc.core_id_to_wk_slice.get(&core)?.at(dim)?
            };
            slice.entry(dim).or_insert(id);
        }
        slices.insert(slice);
    }
    NonZeroU32::new(u32::try_from(slices.len()).ok()?).map(WkSliceCount::new)
}

/// Replaces: e200_getLxNeighborLabeledDsIndicesSet
///
/// Which of a DSC's labelled DSs are fetched from a NEIGHBOUR CORE, by their own recorded index.
///
/// ⛔ TRAP: `config` AND `at` ARE TWO INDEPENDENT CARRIERS of one DSC and the reference relates them
/// nowhere, so the list walked need not belong to the DSC whose schedule decides the answer.
/// ⛔ TRAP: the index inserted is each entry's OWN [`LabeledDs::recorded`] one, not its position.
/// ⛔ [`None`] is [`is_labeled_ds_lx_neighbor`]'s: a DSC index past `dscs_`, or a core the super-DSC
/// states no schedule for.
#[must_use]
pub fn lx_neighbor_labeled_ds_indices(
    sdsc: &SuperDsc,
    config: &DesignSpaceConfig,
    at: DscIdx,
) -> Option<BTreeSet<LdsIdx>> {
    let mut indices = BTreeSet::new();
    for lds in config.labeled_ds.iter() {
        if is_labeled_ds_lx_neighbor(sdsc, at, lds)? {
            indices.insert(lds.recorded());
        }
    }
    Some(indices)
}

/// WHAT ENTRIES 201 AND 202 ADDITIONALLY ASK OF A SCHEDULE TREE — the two data stages a loop divides
/// and the dims it walks, which is
/// [`ScheduleSurgery`](crate::schedule::ddc::transformation_util::ScheduleSurgery)'s
/// `loop_num`/`loop_den`/`loop_dims` asked WITHOUT any of its mutations.
pub trait LoopStages: LoopNesting {
    /// `loopNode->numId_` — the numerator data stage.
    fn loop_num(&self, loop_node: LoopId) -> DatastageId;
    /// `loopNode->denId_` — the denominator data stage.
    fn loop_den(&self, loop_node: LoopId) -> DatastageId;
    /// `loopNode->dims_` (`dsc/dsc2.h:575`).
    fn loop_dims(&self, loop_node: LoopId) -> LoopDims;
}

/// WHICH LX BUFFERING THE SCHEDULER CHOSE — `lxBufferType` (`L3DlOpsScheduler.h:221`) FUSED with
/// `dataStageSuperChunkIdx` (`:224`).
///
/// ⭐ ONE TYPE FOR TWO FIELDS DISCHARGES `DT_CHECK_MSG(lxBufferType != SPATIAL_DOUBLE ||
/// dsc.dataStageParam_.count(dataStageSuperChunkIdx), ..)` (`:4124-4125`): spatial-double is the
/// ONLY writer of that index (`:4648`) and this type cannot be spelled without it. Its `-1` unset
/// state and its `BUFFER_TYPE_COUNT` terminator both stop existing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LxBuffering {
    /// `BufferType::DOUBLE` — one core-by-chunk loop per dim.
    Double,
    /// `BufferType::SPATIAL_DOUBLE` — a core-by-super-chunk over a super-chunk-by-chunk loop per dim.
    SpatialDouble(SuperChunkStage),
}

/// THE LABELLED DS IS A VALUE TENSOR, WITNESSED — `DT_CHECK_MSG(!isIndexLds(lds), "Do not expect index
/// tensor.")`, which entries 201 and 202 each state once.
///
/// ⭐ EXACT AND NOT STRONGER: their one caller collects `allValueLdsIndices` through that very
/// predicate (`:3138`) BEFORE either call, so every labelled DS either function is ever handed has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueLds(());

impl ValueLds {
    /// The witness, or [`None`] where [`is_index_lds`] names an index tensor OR refuses.
    #[must_use]
    pub fn of<M: MemOrg + ?Sized>(lds: &M) -> Option<Self> {
        matches!(is_index_lds(lds), Some(false)).then_some(Self(()))
    }
}

/// WHERE ENTRIES 201 AND 202 START WALKING — `startNode` proved to be the `lx_below_schedule` block,
/// with the loops enclosing it INNERMOST FIRST.
///
/// ⭐ THE PAIR THE CALLER ALREADY BUILDS AS A PAIR: `getLxBelowBlockNode` and then
/// `getParentLoopNodes(*lxBelowBlockNode, dsc)` (`:3128-3133`), which turns
/// `DT_CHECK_MSG(startNode && startNode->name_ == lxBelowBlockNodeName, ..)` into a constructor and
/// makes it impossible to walk one DSC's loops from another DSC's block.
pub struct LxBelowWalk<'a, T: ?Sized> {
    tree: &'a T,
    start: NodeId,
    inner_to_outer: Vec<LoopId>,
}

impl<'a, T: LoopNesting + ?Sized> LxBelowWalk<'a, T> {
    /// The witness, or [`None`] where `name` is not [`LX_BELOW_BLOCK_NODE_NAME`].
    #[must_use]
    pub fn of(tree: &'a T, start: NodeId, name: &NodeName) -> Option<Self> {
        (name.0 == LX_BELOW_BLOCK_NODE_NAME).then(|| Self {
            tree,
            start,
            inner_to_outer: parent_loop_nodes(tree, start),
        })
    }
}

/// WHICH NODE A LABELLED DS'S ALLOCATE OR TRANSFER NODE IS INSERTED BEFORE — the `dsc2::BlockNode*`
/// entries 201 and 202 hand back, which is `startNode` or one of the loops enclosing it, never a
/// third thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiblingNode {
    /// `startNode` itself — the `lx_below_schedule` block, so inside the innermost loop.
    LxBelow(NodeId),
    /// One of `parentInnerToOuterLoopNodes`; the caller re-reads its `denId_` (`:3185`).
    Loop(LoopId),
}

/// `loopNode->dims_.front().dim_` under `DT_CHECK_MSG(loopNode->dims_.size() == 1, "Currently only
/// support one dimension in a loop node.")` — [`None`] is that abort and nothing else.
fn sole_loop_dim<T: LoopStages + ?Sized>(tree: &T, loop_node: LoopId) -> Option<PrimaryDim> {
    let dims = tree.loop_dims(loop_node);
    match dims
        .iter()
        .map(|dim| dim.dim)
        .collect::<Vec<PrimaryDim>>()
        .as_slice()
    {
        [sole] => Some(*sole),
        _ => None,
    }
}

/// Replaces: e201_computeLdsAllocateSiblingLoopNode
///
/// WHICH NODE THE LABELLED DS'S LX ALLOCATE NODE GOES BEFORE: for an HBM tensor the OUTERMOST
/// enclosing loop that does not walk one of its non-broadcast dims, for an LX neighbour the lx-below
/// block, for an LX-local the outermost loop of all.
///
/// ⛔ [`None`] IS THE `nullptr` AND EVERY ABORT ALIKE, which the caller cannot tell apart either —
/// `DT_CHECK_MSG(allocSiblingLoopNode, "Expect a valid node.")` (`:3157`). It covers a DS pinned
/// NOWHERE, a loop pair the spatial-double walk never matches, and `.back()` on an empty walk.
/// ⛔ DIVERGENCE: A SUPER-CHUNK-BY-CHUNK LOOP WITH NO ENCLOSING LOOP IS SKIPPED and a later loop may
/// still answer, where the reference dereferences the null `parentLoop->numId_` (`:457`).
#[must_use]
pub fn compute_lds_allocate_sibling_loop_node<T: LoopStages + ?Sized>(
    walk: &LxBelowWalk<'_, T>,
    sdsc: &SuperDsc,
    at: DscIdx,
    lds: &LabeledDs,
    _lds_is_value: ValueLds,
    buffering: LxBuffering,
) -> Option<SiblingNode> {
    let tree = walk.tree;
    if !lds.pinning().hbm() {
        if is_labeled_ds_lx_neighbor(sdsc, at, lds)? {
            return Some(SiblingNode::LxBelow(walk.start));
        }
        if !lds.pinning().lx {
            return None;
        }
        return walk.inner_to_outer.last().copied().map(SiblingNode::Loop);
    }
    let non_broadcast = sdsc
        .dscs()
        .at(at)?
        .non_broadcast_lds_dim_set(lds.recorded())?;
    match buffering {
        LxBuffering::SpatialDouble(super_chunk) => {
            let mut sibling = None;
            for enclosing in walk.inner_to_outer.iter().copied() {
                if tree.loop_num(enclosing) == super_chunk.index()
                    && tree.loop_den(enclosing) == DATA_STAGE_CHUNK
                    && tree.owner_loop(enclosing.0).is_some_and(|parent| {
                        tree.loop_num(parent) == DATA_STAGE_CORE
                            && tree.loop_den(parent) == super_chunk.index()
                    })
                {
                    sibling = Some(SiblingNode::Loop(enclosing));
                } else if tree.loop_num(enclosing) == DATA_STAGE_CORE
                    && tree.loop_den(enclosing) == super_chunk.index()
                {
                    if non_broadcast.contains(&sole_loop_dim(tree, enclosing)?) {
                        break;
                    }
                    sibling = Some(SiblingNode::Loop(enclosing));
                }
            }
            sibling
        }
        LxBuffering::Double => {
            let mut sibling = SiblingNode::LxBelow(walk.start);
            for enclosing in walk.inner_to_outer.iter().copied() {
                if tree.loop_num(enclosing) != DATA_STAGE_CORE
                    || tree.loop_den(enclosing) != DATA_STAGE_CHUNK
                {
                    return None;
                }
                if non_broadcast.contains(&sole_loop_dim(tree, enclosing)?) {
                    break;
                }
                sibling = SiblingNode::Loop(enclosing);
            }
            Some(sibling)
        }
    }
}

/// Replaces: e202_computeLdsTransferSiblingLoopNode
///
/// WHICH NODE THE LABELLED DS'S HBM→LX TRANSFER GOES BEFORE — entry 201's answer, except that the
/// walk considers only the loops whose DENOMINATOR is the chunk stage and SKIPS every other one.
///
/// ⛔ [`None`] IS THE `nullptr` AND EVERY ABORT ALIKE — `DT_CHECK_MSG(transSiblingLoopNode, "Expect a
/// valid node.")` (`:3181`). ⭐ IT READS NO `lxBufferType`, which is why the caller re-checks
/// `denId_ == dataStageChunkIdx` on the answer (`:3185`): under spatial-double buffering the
/// core-by-super-chunk loops are skipped rather than refused, so the walk stops BELOW them.
#[must_use]
pub fn compute_lds_transfer_sibling_loop_node<T: LoopStages + ?Sized>(
    walk: &LxBelowWalk<'_, T>,
    sdsc: &SuperDsc,
    at: DscIdx,
    lds: &LabeledDs,
    _lds_is_value: ValueLds,
) -> Option<SiblingNode> {
    let tree = walk.tree;
    if !lds.pinning().hbm() {
        if is_labeled_ds_lx_neighbor(sdsc, at, lds)? {
            return Some(SiblingNode::LxBelow(walk.start));
        }
        if !lds.pinning().lx {
            return None;
        }
        return walk.inner_to_outer.last().copied().map(SiblingNode::Loop);
    }
    let non_broadcast = sdsc
        .dscs()
        .at(at)?
        .non_broadcast_lds_dim_set(lds.recorded())?;
    let mut sibling = SiblingNode::LxBelow(walk.start);
    for enclosing in walk.inner_to_outer.iter().copied() {
        if tree.loop_den(enclosing) != DATA_STAGE_CHUNK {
            continue;
        }
        if non_broadcast.contains(&sole_loop_dim(tree, enclosing)?) {
            break;
        }
        sibling = SiblingNode::Loop(enclosing);
    }
    Some(sibling)
}

/// WHICH ARITHMETIC FORMAT AN OP FUNC IS CHARGED AT — the `std::string` `getOpFuncDataFormat` returns,
/// whose four spellings are EXACTLY the four keys `sysFlopsPerByte` states
/// (`sys-arch-spec/sysdef.cpp:297-306`), so the `.at(dataFormat)` beside it (`:2523`) cannot throw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpFuncDataFormat {
    /// `"int4"`.
    Int4,
    /// `"int8"`.
    Int8,
    /// `"fp8"`.
    Fp8,
    /// `"fp16"`, which is also the `default:` arm.
    Fp16,
}

impl OpFuncDataFormat {
    /// The `sysFlopsPerByte` key, spelled as the reference spells it.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
        }
    }
}

/// Replaces: e203_getOpFuncDataFormat
///
/// WHICH ARITHMETIC FORMAT the DSC's first op func is charged at.
///
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: THIS IS A LITERAL ENUMERATION, NOT A PRECISION PREDICATE.
/// SIXTEEN of the 176 op funcs NAME a non-fp16 format and fall to the `default:` `"fp16"` all the
/// same — the plain `MATMUL_INT4/INT8/FP8_FWD`, `SCALED_GROUP_MATMUL_FP4_FWD`, `BATCHMATMUL_MXFP8_FWD`,
/// `BATCHMATMUL_MXFP4W_FWD`, `CSQ_INT8_V2` and nine more — so the arithmetic intensity they are scored
/// with is another format's (`:2521-2525`). Ported verbatim.
/// ⭐ TOTAL: the default arm answers for `OpFuncs::NONE` too, so `DT_CHECK(hasComputeOp)` has no say.
#[must_use]
pub fn op_func_data_format<D: ComputeOps + ?Sized>(dsc: &D) -> OpFuncDataFormat {
    match get_op_func_name(dsc) {
        Some(
            OpFunc::Conv2DInt4Fwd
            | OpFunc::Conv2DInt4FwdGenkg3
            | OpFunc::Conv2DInt4FwdSparsekg3
            | OpFunc::BatchmatmulInt4Fwd
            | OpFunc::BatchmatmulInt4FwdSparsekg3
            | OpFunc::BatchmatmulXrfInt4Fwd
            | OpFunc::BatchmatmulXrfchInt4Fwd
            | OpFunc::CsqInt4
            | OpFunc::CsqInt4Wt
            | OpFunc::CsqInt4Chil,
        ) => OpFuncDataFormat::Int4,
        Some(
            OpFunc::Conv2DInt8Fwd
            | OpFunc::Conv2DInt8FwdGenkg3
            | OpFunc::Conv2DInt8FwdSparsekg3
            | OpFunc::Conv2DInt8FwdOs1
            | OpFunc::Conv2DXrfInt8FwdOs1
            | OpFunc::BatchmatmulInt8Fwd
            | OpFunc::BatchmatmulInt8FwdMbkg3
            | OpFunc::BatchmatmulInt8FwdSparsekg3
            | OpFunc::BatchmatmulXrfInt8Fwd
            | OpFunc::BatchmatmulXrfchInt8Fwd
            | OpFunc::CsqInt8
            | OpFunc::CsqInt8Ch
            | OpFunc::CsqInt8Wt
            | OpFunc::CsqInt8Chil
            | OpFunc::CsqInt8Mb,
        ) => OpFuncDataFormat::Int8,
        Some(
            OpFunc::Conv2DFp8Fwd
            | OpFunc::Conv2DFp8FwdGenkg3
            | OpFunc::Conv2DFp8FwdSparsekg3
            | OpFunc::BatchmatmulFp8Fwd
            | OpFunc::BatchmatmulFp8FwdSparsekg3
            | OpFunc::BatchmatmulXrfFp8Fwd
            | OpFunc::BatchmatmulXrfchFp8Fwd
            | OpFunc::QFp8
            | OpFunc::QFp8Ch
            | OpFunc::QFp8Wt
            | OpFunc::QFp8Chil,
        ) => OpFuncDataFormat::Fp8,
        _ => OpFuncDataFormat::Fp16,
    }
}

/// Replaces: e204_isOpFuncConv2d
///
/// WHETHER THE OP FUNC IS A CONV2D — [`is_op_func_conv2d_int4`], [`is_op_func_conv2d_os1`] and the
/// nine fp16, fp8 and int8 forward spellings its own `static` set names.
///
/// ⭐ THE UNION IS EVERY `CONV2D_*` THE ISA NAMES — sixteen variants
/// (`sys-arch-spec/arch_enums.h:204-215`, `:241-244`) — so this is Conv2d-ness and not a subset of it,
/// and the `static const std::unordered_set` built once per process is a `matches!` here.
#[must_use]
pub fn is_op_func_conv2d(op_func: Option<OpFunc>) -> bool {
    is_op_func_conv2d_int4(op_func)
        || is_op_func_conv2d_os1(op_func)
        || matches!(
            op_func,
            Some(
                OpFunc::Conv2DFwd
                    | OpFunc::Conv2DFp8Fwd
                    | OpFunc::Conv2DInt8Fwd
                    | OpFunc::Conv2DFwdGenkg3
                    | OpFunc::Conv2DFp8FwdGenkg3
                    | OpFunc::Conv2DInt8FwdGenkg3
                    | OpFunc::Conv2DFwdSparsekg3
                    | OpFunc::Conv2DFp8FwdSparsekg3
                    | OpFunc::Conv2DInt8FwdSparsekg3
            )
        )
}

#[cfg(test)]
mod tests_e197_e204 {
    use super::*;
    use crate::schedule::ddc::transformation_util::StageName;
    use crate::schedule::ddc::v1::OpFuncs;
    use crate::schedule::dsc2::LayoutDims;
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletShare, CoreletsUsed, DataStage, DataStages, DscList, DscScheduleStep,
        LabeledDsList, NamedDims, PadElems, PadSizes, StageDims,
    };

    /// The SUPER-CHUNK data stage the spatial-double tests mint — `getNewDataStageIndex`'s answer for
    /// a DSC that already holds the core and chunk stages.
    const SUPER_CHUNK: DatastageId = DatastageId(2);

    fn core(index: u32) -> Core {
        Core::checked(index).expect("this arch has the core the test names")
    }

    fn count(value: u32) -> WkSliceCount {
        WkSliceCount::new(NonZeroU32::new(value).expect("a positive slice count"))
    }

    fn a_stage(extents: &[(PrimaryDim, i64)]) -> DataStage {
        let named = NamedDims {
            name: StageName::default(),
            dims: FilledDims::of(StageDims {
                extents: extents
                    .iter()
                    .map(|(dim, extent)| (*dim, Extent(*extent)))
                    .collect(),
                ..StageDims::default()
            })
            .expect("a stage stating at least one dim"),
        };
        DataStage {
            ss: named.clone(),
            el: named,
        }
    }

    /// A DSC with ONE labelled DS at position 0 recording index 0, whose layout order is `X` then
    /// `Y`, and which uses cores 0 and 1.
    fn a_dsc(scales: &[(PrimaryDim, Scale)], pinning: Pinning) -> DesignSpaceConfig {
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: None,
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(core(0), vec![core(1)]),
            layout_dims: BTreeMap::from([(
                LdsIdx(0),
                LayoutDims::new(PrimaryDim::X, vec![PrimaryDim::Y]),
            )]),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(DsType::Input, scales.to_vec(), LdsIdx(0), pinning),
                vec![],
            ),
            data_stages: DataStages::new(
                a_stage(&[(PrimaryDim::X, 8)]),
                a_stage(&[(PrimaryDim::X, 8)]),
            ),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        }
    }

    /// `X` is a one-element stick dim and so BROADCAST, `Y` carries a whole slice.
    fn broadcast_x() -> Vec<(PrimaryDim, Scale)> {
        vec![
            (PrimaryDim::X, Scale::UnitStick),
            (PrimaryDim::Y, Scale::Sized(1.0)),
        ]
    }

    /// A `memOrg_` whose HBM allocation indirects through nothing, so entry 056 answers `false`.
    struct ValueTensor;

    impl MemOrg for ValueTensor {
        fn hbm_pinned(&self) -> bool {
            true
        }
        fn lx_buffering(&self) -> Option<Buffering> {
            None
        }
        fn lx_start_address(&self, _at: &AddressCoord) -> Option<ByteAddress> {
            None
        }
        fn lx_buffer_offset(&self, _core: Core, _corelet: Corelet) -> Option<BufferOffset> {
            None
        }
        fn hbm_indirection(&self) -> Option<IndirectAlloc> {
            Some(IndirectAlloc::ValueTensor)
        }
        fn hbm_allocation(&self) -> Option<NodeName> {
            None
        }
        fn hbm_layout_dims(&self) -> Option<LayoutDims> {
            None
        }
        fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim> {
            BTreeSet::new()
        }
        fn lx_padding(&self) -> Option<PaddingForm> {
            None
        }
        fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent> {
            BTreeMap::new()
        }
        fn hbm_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }
        fn lx_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }
        fn lx_zero_padded(&self) -> Option<bool> {
            Some(false)
        }
    }

    /// THE NEST BOTH SIBLING WALKS SEE — the `lx_below_schedule` block at node 10, inside loop 9,
    /// inside loop 8, inside the ROOT loop 7 that [`parent_loop_nodes`] stops at.
    struct Nest(BTreeMap<u32, (DatastageId, DatastageId, Vec<PrimaryDim>)>);

    impl LoopNesting for Nest {
        fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
            match node.0 {
                10 => Some(LoopId(NodeId(9))),
                9 => Some(LoopId(NodeId(8))),
                8 => Some(LoopId(NodeId(7))),
                _ => None,
            }
        }
        fn has_parent(&self, node: LoopId) -> bool {
            node.0 != NodeId(7)
        }
    }

    impl Nest {
        fn at(&self, loop_node: LoopId) -> &(DatastageId, DatastageId, Vec<PrimaryDim>) {
            self.0
                .get(&loop_node.0.0)
                .expect("the test states every loop it nests")
        }
    }

    impl LoopStages for Nest {
        fn loop_num(&self, loop_node: LoopId) -> DatastageId {
            self.at(loop_node).0
        }
        fn loop_den(&self, loop_node: LoopId) -> DatastageId {
            self.at(loop_node).1
        }
        fn loop_dims(&self, loop_node: LoopId) -> LoopDims {
            let dims: Vec<PrimaryDimAndKind> = self
                .at(loop_node)
                .2
                .iter()
                .map(|dim| PrimaryDimAndKind {
                    dim: *dim,
                    kind: MetaDimKind::Unpadded,
                })
                .collect();
            let (first, rest) = dims.split_first().expect("a loop walks at least one dim");
            LoopDims::new(*first, rest.to_vec())
        }
    }

    /// A core-by-chunk nest: loop 9 walks `X`, loop 8 walks `Y`, and the root loop 7 is never read.
    fn double_nest() -> Nest {
        Nest(BTreeMap::from([
            (9, (DATA_STAGE_CORE, DATA_STAGE_CHUNK, vec![PrimaryDim::X])),
            (8, (DATA_STAGE_CORE, DATA_STAGE_CHUNK, vec![PrimaryDim::Y])),
        ]))
    }

    /// The spatial-double nest `addLoopNodes` builds: a core-by-super-chunk loop 8 OUTSIDE a
    /// super-chunk-by-chunk loop 9 (`L3DlOpsScheduler.cpp:4645-4650`).
    fn spatial_nest() -> Nest {
        Nest(BTreeMap::from([
            (9, (SUPER_CHUNK, DATA_STAGE_CHUNK, vec![PrimaryDim::X])),
            (8, (DATA_STAGE_CORE, SUPER_CHUNK, vec![PrimaryDim::Y])),
        ]))
    }

    fn lx_walk<'a>(nest: &'a Nest) -> LxBelowWalk<'a, Nest> {
        LxBelowWalk::of(
            nest,
            NodeId(10),
            &NodeName(LX_BELOW_BLOCK_NODE_NAME.to_owned()),
        )
        .expect("the lx-below block names itself")
    }

    /// e197 — a dim corelet 0 holds less of than the core is split, `IJ` never is however it is
    /// stated, and a one-corelet DSC splits nothing at all.
    #[test]
    fn corelet_split_dimensions_skip_the_combined_dims() {
        let split = CoreletShare {
            corelet0: Extent(4),
            whole: Extent(8),
        };
        let whole = CoreletShare {
            corelet0: Extent(8),
            whole: Extent(8),
        };
        let mut dsc = a_dsc(&broadcast_x(), Pinning::default());
        dsc.corelets_used =
            CoreletsUsed::new(NonZeroU32::new(2).expect("two corelets is a positive count"));
        dsc.corelet_shares = BTreeMap::from([
            (PrimaryDim::Y, split),
            (PrimaryDim::X, whole),
            (PrimaryDim::Ij, split),
            (PrimaryDim::Kij, split),
        ]);
        assert_eq!(
            corelet_split_dimensions(&dsc),
            BTreeSet::from([PrimaryDim::Y])
        );

        // The redundant outer guard: one corelet cannot split anything, whatever the shares say.
        dsc.corelets_used = CoreletsUsed::ONE;
        assert_eq!(corelet_split_dimensions(&dsc), BTreeSet::new());
    }

    /// e198 — the core stage's padding lands whole on the chunk stage and is then VOIDED on the dim
    /// chunking moved, while the dim it left alone keeps its sizes.
    #[test]
    fn chunk_padding_is_the_cores_voided_where_chunking_moved_the_extent() {
        let pad = |front: u32, back: u32| DimPadding {
            sizes: PadSizes::of(PadElems(front), PadElems(back)),
            ..DimPadding::default()
        };
        let stage = |extents: &[(PrimaryDim, i64)], padding: &[(PrimaryDim, DimPadding)]| {
            FilledDims::of(StageDims {
                extents: extents
                    .iter()
                    .map(|(dim, extent)| (*dim, Extent(*extent)))
                    .collect(),
                padding: padding.iter().cloned().collect(),
                ..StageDims::default()
            })
            .expect("a stage stating at least one dim")
        };
        let core_params = stage(
            &[(PrimaryDim::X, 8), (PrimaryDim::Y, 64)],
            &[(PrimaryDim::X, pad(1, 2)), (PrimaryDim::Y, pad(3, 4))],
        );
        // The chunk stage states NO padding of its own, and `Y` is the dim chunking moved.
        let mut chunk_params = stage(&[(PrimaryDim::X, 8), (PrimaryDim::Y, 16)], &[]);

        add_or_update_padding_sizes_in_chunk_params::<true>(&mut chunk_params, &core_params);

        assert_eq!(
            chunk_params.dims().padding[&PrimaryDim::X].sizes,
            PadSizes::of(PadElems(1), PadElems(2))
        );
        assert_eq!(
            chunk_params.dims().padding[&PrimaryDim::Y].sizes,
            PadSizes::Voided
        );
    }

    /// e199 — a whole-super-DSC group multiplies the non-broadcast dims' slice counts, a partial one
    /// counts the distinct per-core slices, and a dim nothing states a count for has no answer.
    #[test]
    fn work_slice_count_is_a_product_or_a_tally_of_distinct_slices() {
        let dsc = a_dsc(&broadcast_x(), Pinning::default());
        let slice = |x: i32, y: i32| {
            WkSlice(BTreeMap::from([
                (PrimaryDim::X, WkSliceId(x)),
                (PrimaryDim::Y, WkSliceId(y)),
            ]))
        };
        let slices = BTreeMap::from([(core(0), slice(0, 0)), (core(1), slice(0, 1))]);
        let counts = BTreeMap::from([(PrimaryDim::X, count(2)), (PrimaryDim::Y, count(3))]);
        let group = DscGroup::new(&dsc, vec![]);

        // The fast path: ONE DSC named of a ONE-DSC super-DSC. `X` broadcasts, so only `Y` counts.
        let whole = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            counts.clone(),
            slices.clone(),
            BTreeMap::new(),
        );
        assert_eq!(
            labeled_ds_num_of_wk_slices(&whole, LdsIdx(0), &group),
            Some(count(3))
        );
        // ... and `numWkSlicesPerDim_.at(dim)` throwing is the absence of an answer.
        let unstated = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            BTreeMap::from([(PrimaryDim::X, count(2))]),
            slices.clone(),
            BTreeMap::new(),
        );
        assert_eq!(
            labeled_ds_num_of_wk_slices(&unstated, LdsIdx(0), &group),
            None
        );

        // The tally path: ONE DSC named of a TWO-DSC super-DSC. Both cores agree on the broadcast
        // `X`, whose id is forced to 0, and differ on `Y`, so the two cores hold two slices.
        let partial = SuperDsc::new(
            DscList::new(dsc.clone(), vec![dsc.clone()]),
            counts,
            slices,
            BTreeMap::new(),
        );
        assert_eq!(
            labeled_ds_num_of_wk_slices(&partial, LdsIdx(0), &group),
            Some(count(2))
        );
        // ... and `coreIdToWkSlice_.at(coreId)` throwing is the absence of an answer.
        let no_slices = SuperDsc::new(
            DscList::new(dsc.clone(), vec![dsc.clone()]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        assert_eq!(
            labeled_ds_num_of_wk_slices(&no_slices, LdsIdx(0), &group),
            None
        );
    }

    /// e200 — an LX-pinned input whose own schedule step also names a data DSC is a neighbour fetch
    /// and is reported by its RECORDED index, and a core with no schedule has no answer.
    #[test]
    fn lx_neighbor_indices_are_the_recorded_ones() {
        let lx = Pinning {
            mem_org: BTreeMap::new(),
            lx: true,
            lx_padded: false,
        };
        let mut dsc = a_dsc(&broadcast_x(), lx.clone());
        // Position 0 RECORDS index 7; position 1 is an output and is never a neighbour fetch.
        dsc.labeled_ds = LabeledDsList::new(
            LabeledDs::new(DsType::Input, broadcast_x(), LdsIdx(7), lx.clone()),
            vec![LabeledDs::new(DsType::Output, broadcast_x(), LdsIdx(1), lx)],
        );
        let schedule = vec![DscScheduleStep {
            data_dsc: Some(DscIdx(3)),
            dl_dsc: Some(DscIdx(0)),
        }];
        let sdsc = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::from([(core(0), schedule)]),
        );
        assert_eq!(
            lx_neighbor_labeled_ds_indices(&sdsc, &dsc, DscIdx(0)),
            Some(BTreeSet::from([LdsIdx(7)]))
        );

        // `coreIdToDscSchedule.at(coreId)` throwing is the absence of an answer.
        let unscheduled = SuperDsc::new(
            DscList::new(dsc.clone(), vec![]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        assert_eq!(
            lx_neighbor_labeled_ds_indices(&unscheduled, &dsc, DscIdx(0)),
            None
        );
    }

    /// A one-DSC super-DSC over `dsc`, with no work slices and no schedule.
    fn a_super_dsc(dsc: DesignSpaceConfig) -> SuperDsc {
        SuperDsc::new(
            DscList::new(dsc, vec![]),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        )
    }

    fn hbm_pinned() -> Pinning {
        Pinning {
            mem_org: [(SenComponent::Hbm, true)].into(),
            lx: false,
            lx_padded: false,
        }
    }

    /// e201 under DOUBLE buffering — the allocate node goes before the OUTERMOST core-by-chunk loop
    /// that does not walk a non-broadcast dim, and a loop that is not core-by-chunk refuses.
    #[test]
    fn a_double_buffered_allocation_stops_below_the_loop_walking_its_own_dim() {
        let nest = double_nest();
        let walk = lx_walk(&nest);
        let value = ValueLds::of(&ValueTensor).expect("a value tensor is not an index tensor");

        // `X` broadcasts and `Y` does not, so loop 8's `Y` stops the walk at loop 9.
        let lds = LabeledDs::new(DsType::Input, broadcast_x(), LdsIdx(0), hbm_pinned());
        let sdsc = a_super_dsc(a_dsc(&broadcast_x(), hbm_pinned()));
        assert_eq!(
            compute_lds_allocate_sibling_loop_node(
                &walk,
                &sdsc,
                DscIdx(0),
                &lds,
                value,
                LxBuffering::Double
            ),
            Some(SiblingNode::Loop(LoopId(NodeId(9))))
        );

        // A wholly broadcast tensor stops nowhere, so it reaches the outermost loop of the walk.
        let all_broadcast = vec![
            (PrimaryDim::X, Scale::UnitStick),
            (PrimaryDim::Y, Scale::UnitStick),
        ];
        let lds = LabeledDs::new(
            DsType::Input,
            all_broadcast.clone(),
            LdsIdx(0),
            hbm_pinned(),
        );
        let sdsc = a_super_dsc(a_dsc(&all_broadcast, hbm_pinned()));
        assert_eq!(
            compute_lds_allocate_sibling_loop_node(
                &walk,
                &sdsc,
                DscIdx(0),
                &lds,
                value,
                LxBuffering::Double
            ),
            Some(SiblingNode::Loop(LoopId(NodeId(8))))
        );

        // *"Expect a core-by-chunk loop."*: a spatial-double nest walked as a double-buffered one.
        let spatial = spatial_nest();
        assert_eq!(
            compute_lds_allocate_sibling_loop_node(
                &lx_walk(&spatial),
                &sdsc,
                DscIdx(0),
                &lds,
                value,
                LxBuffering::Double
            ),
            None
        );
    }

    /// e201 under SPATIAL-DOUBLE buffering — the allocate node goes before the super-chunk-by-chunk
    /// loop whose parent is the core-by-super-chunk loop, and the core loop's own dim can still
    /// stop the walk there.
    #[test]
    fn a_spatial_double_allocation_lands_on_the_super_chunk_loop_pair() {
        let nest = spatial_nest();
        let walk = lx_walk(&nest);
        let value = ValueLds::of(&ValueTensor).expect("a value tensor is not an index tensor");
        let mut stages = DataStages::new(
            a_stage(&[(PrimaryDim::X, 8)]),
            a_stage(&[(PrimaryDim::X, 8)]),
        );
        stages.set(SUPER_CHUNK, a_stage(&[(PrimaryDim::X, 8)]));
        let buffering = LxBuffering::SpatialDouble(
            stages
                .super_chunk(SUPER_CHUNK)
                .expect("the super-chunk stage the walk names"),
        );

        // Loop 8 walks `Y`, which is non-broadcast, so the answer stays at the pair below it.
        let lds = LabeledDs::new(DsType::Input, broadcast_x(), LdsIdx(0), hbm_pinned());
        let sdsc = a_super_dsc(a_dsc(&broadcast_x(), hbm_pinned()));
        assert_eq!(
            compute_lds_allocate_sibling_loop_node(&walk, &sdsc, DscIdx(0), &lds, value, buffering),
            Some(SiblingNode::Loop(LoopId(NodeId(9))))
        );

        // A wholly broadcast tensor walks on out to the core-by-super-chunk loop.
        let all_broadcast = vec![
            (PrimaryDim::X, Scale::UnitStick),
            (PrimaryDim::Y, Scale::UnitStick),
        ];
        let lds = LabeledDs::new(
            DsType::Input,
            all_broadcast.clone(),
            LdsIdx(0),
            hbm_pinned(),
        );
        let sdsc = a_super_dsc(a_dsc(&all_broadcast, hbm_pinned()));
        assert_eq!(
            compute_lds_allocate_sibling_loop_node(&walk, &sdsc, DscIdx(0), &lds, value, buffering),
            Some(SiblingNode::Loop(LoopId(NodeId(8))))
        );

        // A tensor pinned NOWHERE is the reference's `nullptr`, whatever the nest looks like.
        let nowhere = LabeledDs::new(DsType::Input, all_broadcast, LdsIdx(0), Pinning::default());
        assert_eq!(
            compute_lds_allocate_sibling_loop_node(
                &walk,
                &sdsc,
                DscIdx(0),
                &nowhere,
                value,
                buffering
            ),
            None
        );
    }

    /// e202 — the transfer walk keeps only the chunk-denominated loops, so a spatial-double nest's
    /// core-by-super-chunk loop is SKIPPED rather than refused, and the LX-local arm answers with
    /// the outermost loop of the walk.
    #[test]
    fn a_transfer_walk_considers_only_the_chunk_denominated_loops() {
        let nest = spatial_nest();
        let walk = lx_walk(&nest);
        let value = ValueLds::of(&ValueTensor).expect("a value tensor is not an index tensor");
        let all_broadcast = vec![
            (PrimaryDim::X, Scale::UnitStick),
            (PrimaryDim::Y, Scale::UnitStick),
        ];
        let sdsc = a_super_dsc(a_dsc(&all_broadcast, hbm_pinned()));

        // Loop 8 divides the core stage by the super-chunk, so the walk cannot pass it and stops at
        // loop 9 — which is exactly what the caller's `denId_ == dataStageChunkIdx` re-check wants.
        let lds = LabeledDs::new(
            DsType::Input,
            all_broadcast.clone(),
            LdsIdx(0),
            hbm_pinned(),
        );
        assert_eq!(
            compute_lds_transfer_sibling_loop_node(&walk, &sdsc, DscIdx(0), &lds, value),
            Some(SiblingNode::Loop(LoopId(NodeId(9))))
        );

        // An LX-local tensor goes OUTSIDE the outermost loop of the walk, whatever divides it.
        let lx_local = LabeledDs::new(
            DsType::Output,
            all_broadcast,
            LdsIdx(0),
            Pinning {
                mem_org: BTreeMap::new(),
                lx: true,
                lx_padded: false,
            },
        );
        assert_eq!(
            compute_lds_transfer_sibling_loop_node(&walk, &sdsc, DscIdx(0), &lx_local, value),
            Some(SiblingNode::Loop(LoopId(NodeId(8))))
        );
    }

    /// `computeOp_` with one entry, as the other test modules in this file spell one.
    struct Ops(Option<OpFunc>);

    impl ComputeOps for Ops {
        fn op_funcs(&self) -> OpFuncs {
            OpFuncs::new(self.0, Vec::new())
        }
        fn set_first_op_func(&mut self, op_func: OpFunc) {
            self.0 = Some(op_func);
        }
    }

    /// e203 — one representative of each of the four keys, and nine of the SIXTEEN op funcs that NAME
    /// a format the reference does not charge them at.
    #[test]
    fn the_data_format_is_a_literal_enumeration_and_not_a_precision_predicate() {
        for (op, format) in [
            (OpFunc::CsqInt4Chil, OpFuncDataFormat::Int4),
            (OpFunc::Conv2DXrfInt8FwdOs1, OpFuncDataFormat::Int8),
            (OpFunc::BatchmatmulXrfchFp8Fwd, OpFuncDataFormat::Fp8),
            (OpFunc::BatchmatmulFwd, OpFuncDataFormat::Fp16),
        ] {
            assert_eq!(op_func_data_format(&Ops(Some(op))), format);
        }
        assert_eq!(
            [
                OpFuncDataFormat::Int4.key(),
                OpFuncDataFormat::Int8.key(),
                OpFuncDataFormat::Fp8.key(),
                OpFuncDataFormat::Fp16.key(),
            ],
            ["int4", "int8", "fp8", "fp16"]
        );

        // ⛔ THE TRAP: each of these names its format and each is charged as `fp16`.
        for op in [
            OpFunc::CsqInt8V2,
            OpFunc::CsqInt8MbV2,
            OpFunc::QFp8Mb,
            OpFunc::BatchmatmulFp8FwdMb,
            OpFunc::BatchmatmulMxfp8Fwd,
            OpFunc::BatchmatmulMxfp4WFwd,
            OpFunc::MatmulInt4Fwd,
            OpFunc::MatmulInt8Fwd,
            OpFunc::MatmulFp8Fwd,
        ] {
            assert_eq!(op_func_data_format(&Ops(Some(op))), OpFuncDataFormat::Fp16);
        }
        // No compute op at all answers `fp16` too, so the switch is total.
        assert_eq!(op_func_data_format(&Ops(None)), OpFuncDataFormat::Fp16);
    }

    /// e204 — the union of the three predicates is every `CONV2D_*` the ISA names, and nothing else
    /// is a Conv2d.
    #[test]
    fn every_conv2d_the_isa_names_is_a_conv2d_and_no_batchmatmul_is() {
        for op in [
            OpFunc::Conv2DFwd,
            OpFunc::Conv2DFp8Fwd,
            OpFunc::Conv2DInt8Fwd,
            OpFunc::Conv2DInt4Fwd,
            OpFunc::Conv2DFwdGenkg3,
            OpFunc::Conv2DFp8FwdGenkg3,
            OpFunc::Conv2DInt8FwdGenkg3,
            OpFunc::Conv2DInt4FwdGenkg3,
            OpFunc::Conv2DFwdSparsekg3,
            OpFunc::Conv2DFp8FwdSparsekg3,
            OpFunc::Conv2DInt8FwdSparsekg3,
            OpFunc::Conv2DInt4FwdSparsekg3,
            OpFunc::Conv2DFwdOs1,
            OpFunc::Conv2DFwdGenOs1,
            OpFunc::Conv2DInt8FwdOs1,
            OpFunc::Conv2DXrfInt8FwdOs1,
        ] {
            assert!(is_op_func_conv2d(Some(op)), "{op:?} is a Conv2d");
        }
        assert!(!is_op_func_conv2d(Some(OpFunc::BatchmatmulFwd)));
        assert!(!is_op_func_conv2d(None));
    }
}

/// Replaces: e205_isOpFuncBmm
///
/// Whether the op is a batch matmul of ANY weight format — the five format families disjoined.
#[must_use]
pub const fn is_op_func_bmm(op_func: OpFunc) -> bool {
    is_op_func_bmm_fp16(op_func)
        || is_op_func_bmm_fp8_xrf(op_func)
        || is_op_func_bmm_fp8_non_xrf(op_func)
        || is_op_func_bmm_int4(op_func)
        || is_op_func_bmm_int8(op_func)
}

/// Replaces: e206_getMinParamBmm
///
/// The smallest chunk a batch matmul may take of `dim`: 64/128/256 on the INPUT channel by weight
/// format and core extent, 64 on the OUTPUT channel, the largest divisor of the core extent up to 64
/// on a weight-reuse dim, else 1.
///
/// ⛔ [`None`] IS BOTH `DT_CHECK(size() == 1)`s, the `primaryDsInfo_.at()` and `labeledDs_.at(1)`
/// throws, AND the reference's `-1` return for an unstated core extent, which its caller writes
/// straight into a data stage.
/// ⛔ DIVERGENCE: THE WEIGHT-REUSE ARM CANNOT MATCH ON AN EMPTY `inp1_reuse_dim` — the reference
/// dereferences `*begin()` on an empty `std::set` there, which is no answer at all.
#[must_use]
pub fn min_param_bmm<A: Arch>(
    dsc: &DesignSpaceConfig,
    dim: PrimaryDim,
    op_func: OpFunc,
) -> Option<Extent> {
    let core_param = dsc.core_stage().dims().extent(dim);
    let core = core_param.map_or(-1, |extent| extent.0);
    let layout = |lds: &LabeledDs| Some(dsc.primary_ds_info.get(&lds.ds_type())?.layout.to_vec());
    let inp0 = layout(dsc.labeled_ds.front())?;
    let inp1 = layout(dsc.labeled_ds.at(LdsIdx(1))?)?;
    let out = layout(dsc.labeled_ds.back())?;
    let reuse = |from: &[PrimaryDim], within: &[PrimaryDim], without: &[PrimaryDim]| {
        from.iter()
            .filter(|named| within.contains(named) && !without.contains(named))
            .copied()
            .collect::<BTreeSet<PrimaryDim>>()
    };
    let inp0_reuse = reuse(&inp1, &out, &inp0);
    let inp1_reuse = reuse(&inp0, &out, &inp1);
    let out_reuse = reuse(&inp0, &inp1, &out);
    (inp0_reuse.len() == 1).then_some(())?;
    (out_reuse.len() == 1).then_some(())?;
    if out_reuse.first() == Some(&dim) {
        // The input channel.
        let param = if is_op_func_bmm_int4(op_func) {
            128
        } else if is_op_func_bmm_int8(op_func) || is_op_func_bmm_fp8_non_xrf(op_func) {
            if core % 256 == 0 {
                256
            } else if core % 128 == 0 {
                128
            } else {
                64
            }
        } else if is_op_func_bmm_fp16(op_func) {
            if A::GEN > IsaGen::Rcudd1a && core >= 1024 && core % 256 == 0 {
                256
            } else if core % 128 == 0 {
                128
            } else {
                64
            }
        } else if is_op_func_bmm_fp8_xrf(op_func) {
            64
        } else {
            DEFAULT_MIN_PARAM.0
        };
        return Some(Extent(param));
    }
    if inp0_reuse.first() == Some(&dim) {
        // The output channel.
        return Some(Extent(64));
    }
    if inp1_reuse.first() == Some(&dim) {
        // A weight-reuse dim: 64 is 80% utilisation for int8, and below-LX XRF reuse takes the rest.
        let found = (1i64..=64).rev().find(|reuse| core % reuse == 0);
        return Some(found.map_or(DEFAULT_MIN_PARAM, Extent));
    }
    if core < 0 {
        return core_param;
    }
    Some(DEFAULT_MIN_PARAM)
}

/// `mySDsc.dscs_.at(idx).labeledDs_`'s memory organisations, in `labeledDs_` order.
fn lds_orgs<'o, O: MemOrgs>(
    orgs: &'o O,
    idx: DscIdx,
    dsc: &DesignSpaceConfig,
) -> Option<Vec<&'o O::Org>> {
    dsc.labeled_ds
        .indexed()
        .map(|(at, _)| orgs.mem_org(idx, at))
        .collect()
}

/// `value % divisor == 0`, [`None`] for the divisor of zero the reference's `%` cannot take.
fn is_multiple_of(value: i64, divisor: i64) -> Option<bool> {
    (divisor != 0).then(|| value % divisor == 0)
}

/// `isDimensionParamMultipleOfStickSize` (`L3DlOpsScheduler.cpp:1312`) — every non-index labelled
/// DS's stick size along `dim` divides `param`, a scale tensor's stick multiplied by its own block.
fn param_multiple_of_stick_size<M: MemOrg + ?Sized>(
    dsc: &DesignSpaceConfig,
    dim: PrimaryDim,
    param: i64,
    orgs: &[&M],
) -> Option<bool> {
    for (lds, org) in dsc.labeled_ds.iter().zip(orgs) {
        if is_index_lds(*org)? {
            continue;
        }
        let mut min_dim_size = i64::try_from(stick_size(dsc, lds.ds_type(), dim)?.0).ok()?;
        if let Some(mx) = lds.scale_tensor().filter(|mx| mx.dim == dim) {
            min_dim_size =
                min_dim_size.saturating_mul(i64::try_from(mx.blk_size.count().0).ok()?);
        }
        if !is_multiple_of(param, min_dim_size)? {
            return Some(false);
        }
    }
    Some(true)
}

/// `isParamCoreletSplitValid` (`:1334`) — a corelet-split dim's parameter must split equally across
/// the corelets AND each corelet's share must itself be a multiple of every stick size.
fn param_corelet_split_valid<M: MemOrg + ?Sized>(
    dsc: &DesignSpaceConfig,
    dim: PrimaryDim,
    param: i64,
    orgs: &[&M],
) -> Option<bool> {
    if !is_dimension_corelet_split(dsc, dim) {
        return Some(true);
    }
    let corelets = i64::from(dsc.corelets_used.get());
    if param % corelets != 0 {
        return Some(false);
    }
    param_multiple_of_stick_size(dsc, dim, param / corelets, orgs)
}

/// Replaces: e207_generateDscParamCandidates
///
/// EVERY DSC'S CANDIDATE CHUNK EXTENT PER DIM: a non-chunk dim keeps its core extent alone; a chunk
/// dim takes every divisor of the core (scale-down-adjusted) upper bound from the lower bound up that
/// the symbolic granularity, the page sizes, the corelet split and every stick size all admit.
///
/// ⛔ [`None`] IS EVERY `DT_CHECK_MSG` AND `.at()` THROW: an invalid bound pair, a `getPageSize` or
/// symbolic lookup with no stage stated for it, an index tensor holding indices, a stick size of zero
/// — and, at the end, *"There must be at least one valid candidate."*, which [`Candidates`]' own
/// non-emptiness raises for whichever `(dsc, dim)` came up empty.
/// ⛔ DIVERGENCE: A NON-CHUNK DIM THE CORE STAGE DOES NOT STATE IS REFUSED rather than recorded as
/// the reference's single candidate of `-1`, which is not a chunk extent.
#[must_use]
pub fn generate_dsc_param_candidates<O: MemOrgs>(
    sdsc: &SuperDsc,
    dsc_params: &[FilledDims],
    primary_dims: &[PrimaryDim],
    chunk_dims: &BTreeSet<PrimaryDim>,
    core_split_dims: &BTreeSet<PrimaryDim>,
    orgs: &O,
    paged: Option<PagedStages>,
) -> Option<DscCandidates> {
    let mut per_dsc: Vec<BTreeMap<PrimaryDim, Vec<Extent>>> =
        vec![BTreeMap::new(); sdsc.dscs().iter().count()];
    for &dim in primary_dims {
        for (idx, dsc) in (0u32..).map(DscIdx).zip(sdsc.dscs().iter()) {
            let slot = per_dsc.get_mut(usize::try_from(idx.0).ok()?)?;
            let core = dsc.core_stage().dims();
            if !chunk_dims.contains(&dim) {
                slot.insert(dim, vec![core.extent(dim)?]);
                continue;
            }
            let l_bound = dsc_params
                .get(usize::try_from(idx.0).ok()?)?
                .dims()
                .extent(dim)?;
            // A block transfer with a symbolic size cannot be broken in ALxS yet, so a symbolic dim
            // with an LX->HBM transfer on it is forced to chunk on its granularity.
            let output = dsc.labeled_ds.back();
            let chunk_symbolic = output.pinning().hbm()
                && dsc
                    .non_broadcast_lds_dims(output.recorded())?
                    .contains(&dim);
            let density = output
                .scale_tensor()
                .filter(|mx| mx.dim == dim)
                .map(|mx| mx.blk_size);
            let u_bound =
                core.scaled_extent(dim, &PaddingForm::default(), density, chunk_symbolic)?;
            (l_bound.0 > 0 && u_bound.0 >= l_bound.0).then_some(())?;
            let mem_orgs = lds_orgs(orgs, idx, dsc)?;
            let paged_dims = get_paged_dimensions(&mem_orgs);
            if l_bound == u_bound {
                // No chunking on this dim, so the core extent is the one candidate and only its own
                // stick alignment is verified — `getMinParamForDim` uses this to force no chunking
                // on a dim whose padded parameter would be far harder to check.
                slot.insert(dim, vec![l_bound]);
                for (lds, org) in dsc.labeled_ds.iter().zip(&mem_orgs) {
                    if is_index_lds(*org)? {
                        continue;
                    }
                    let stick = i64::try_from(stick_size(dsc, lds.ds_type(), dim)?.0).ok()?;
                    let ubound_lds = core.scaled_extent(dim, &org.lx_padding()?, None, false)?;
                    is_multiple_of(ubound_lds.0, stick)?.then_some(())?;
                }
                continue;
            }
            let mut found: Vec<Extent> = Vec::new();
            for param in l_bound.0..=u_bound.0 {
                // Equal chunks only, so the upper bound must be a multiple of the parameter.
                if u_bound.0 % param != 0 {
                    continue;
                }
                // A symbolic dim's candidate must divide its granularity, the upper bound being its
                // max size.
                if core.symbolic.info().contains_key(&dim) && param < u_bound.0 {
                    let granularity =
                        core.scaled_extent(dim, &PaddingForm::default(), density, true)?;
                    if param > granularity.0 || !is_multiple_of(granularity.0, param)? {
                        continue;
                    }
                }
                // A paged dim's candidate must be whole pages, and must divide both the steady-state
                // and the epilogue size one index-tensor stick represents.
                if paged_dims.contains(&dim) {
                    let stages = paged?;
                    let one_page = dsc.data_stages.at(stages.one_page)?.ss_extent(dim)?;
                    let ibr = dsc.data_stages.at(stages.ibr)?;
                    let steady = ibr.ss.dims.dims().extent(dim)?;
                    let epilogue = ibr.el.dims.dims().extent(dim)?;
                    if !is_multiple_of(param, one_page.0)?
                        || !is_multiple_of(steady.0, param)?
                        || !is_multiple_of(epilogue.0, param)?
                    {
                        continue;
                    }
                }
                // The corelet split applies to THIS DSC for a core-split dim and to EVERY DSC
                // otherwise.
                let split_valid = if core_split_dims.contains(&dim) {
                    param_corelet_split_valid(dsc, dim, param, &mem_orgs)?
                } else {
                    let mut all = true;
                    for (other_idx, other) in (0u32..).map(DscIdx).zip(sdsc.dscs().iter()) {
                        let other_orgs = lds_orgs(orgs, other_idx, other)?;
                        if !param_corelet_split_valid(other, dim, param, &other_orgs)? {
                            all = false;
                            break;
                        }
                    }
                    all
                };
                if !split_valid || !param_multiple_of_stick_size(dsc, dim, param, &mem_orgs)? {
                    continue;
                }
                found.push(Extent(param));
            }
            slot.insert(dim, found);
        }
    }
    per_dsc
        .into_iter()
        .map(DimCandidates::of)
        .collect::<Option<Vec<_>>>()
        .map(DscCandidates::new)
}

/// Replaces: e208_getLdsL3TransferNodes
///
/// THE DSC'S L3 TRANSFERS FOR ONE LABELLED DS — the transfers that use its allocation and run between
/// one of `src_storages` and one of `dst_storages`. An LX-PINNED DS HAS NONE, and that empty answer
/// is not a refusal.
///
/// ⛔ [`None`] IS EVERY `DT_CHECK_MSG`: both `.at()` throws, *"Expect input neighbor fetch."* for an
/// unpinned DS that is no LX neighbour, *"Expect HBM/LX in memOrg_."* with *"Expect a valid allocate
/// node."*, and *"Expect valid alloc users."* — which is the allocation naming NO user.
#[must_use]
pub fn lds_l3_transfer_nodes<M: MemOrg + ?Sized, T: TransferNodes + ?Sized>(
    sdsc: &SuperDsc,
    dsc: DscIdx,
    lds: LdsIdx,
    org: &M,
    trees: &T,
    src_storages: &[SenComponent],
    dst_storages: &[SenComponent],
) -> Option<Vec<L3Transfer>> {
    let entry = sdsc.dscs().at(dsc)?.labeled_ds.at(lds)?;
    if entry.pinning().lx {
        return Some(Vec::new());
    }
    let users = if entry.pinning().hbm() {
        org.hbm_alloc_users()?
    } else {
        is_labeled_ds_lx_neighbor(sdsc, dsc, entry)?.then_some(())?;
        org.lx_alloc_users()?
    };
    (!users.is_empty()).then_some(())?;
    Some(
        trees
            .transfers(dsc)
            .into_iter()
            .filter(|transfer| {
                users.contains(&transfer.node)
                    && src_storages.contains(&transfer.src)
                    && dst_storages.contains(&transfer.dst)
            })
            .collect(),
    )
}

/// Replaces: e209_getLabeledDsChunkStickVolume
///
/// HOW MANY CONSECUTIVE STICKS ONE CHUNK OF A LABELLED DS SPANS — the product of each non-broadcast
/// dim's chunk parameter over its stick size, innermost outwards, stopping at the first dim whose
/// chunk falls below its core extent or whose symbolic-ness disagrees between the two stages.
///
/// ⛔ A SYMBOLIC DIM IS TAKEN AT ITS GRANULARITY on BOTH stages — the smallest volume it can have,
/// which is the conservative estimate the reference wants — and a paged dim is CAPPED at its page.
/// ⛔ [`None`] IS EVERY `DT_CHECK_MSG`: *"Cannot getLabeledDsChunkStickVolume on indirect access index
/// tensor"*, *"Expect LX in labeledDs memOrg_."* with *"Expect a valid allocate node."*, *"Expect a
/// valid parameter value."* (`param > 0`), and the stick size not dividing the parameter.
#[must_use]
pub fn labeled_ds_chunk_stick_volume<M: MemOrg + ?Sized>(
    dsc: &DesignSpaceConfig,
    lds: LdsIdx,
    org: &M,
) -> Option<StickVolume> {
    let entry = dsc.labeled_ds.at(lds)?;
    (!is_index_lds(org)?).then_some(())?;
    let padding = org.lx_padding()?;
    let pages = org.lx_page_sizes();
    let core = dsc.data_stages.core().ss.dims.dims();
    let chunk = dsc.data_stages.chunk().ss.dims.dims();
    // A broadcast dim holds one stick, which is consecutive by itself.
    let mut volume: u64 = 1;
    for dim in dsc.non_broadcast_lds_dims(lds)? {
        let stick = i64::try_from(stick_size(dsc, entry.ds_type(), dim)?.0).ok()?;
        let density = entry
            .scale_tensor()
            .filter(|mx| mx.dim == dim)
            .map(|mx| mx.blk_size);
        let mut param = chunk.scaled_extent(dim, &padding, density, true)?;
        (param.0 > 0).then_some(())?;
        let upper = core.scaled_extent(dim, &padding, density, true)?;
        if let Some(page) = pages.get(&dim).filter(|page| page.0 < param.0) {
            param = *page;
        }
        is_multiple_of(param.0, stick)?.then_some(())?;
        volume = volume.checked_mul(u64::try_from(param.0 / stick).ok()?)?;
        // Below the core extent the outer dims are no longer contiguous, and so is a dim that is
        // symbolic on one stage and not the other.
        if param.0 < upper.0
            || chunk.symbolic.info().contains_key(&dim) != core.symbolic.info().contains_key(&dim)
        {
            break;
        }
    }
    std::num::NonZeroU64::new(volume).map(StickVolume::new)
}

/// Replaces: e210_isOpCrossCoreReduction
///
/// Whether the op reduces across cores — some dim it reduces away is split over more than one work
/// slice.
///
/// ⛔ [`None`] IS `numWkSlicesPerDim_.at(dim)`'s THROW, reached in the reference's `any_of` order: a
/// dim with more than one slice ANSWERS before a later unnamed dim can refuse.
#[must_use]
pub fn is_op_cross_core_reduction(sdsc: &SuperDsc, dsc: &DesignSpaceConfig) -> Option<bool> {
    for dim in op_reduced_dim_set(dsc)? {
        if sdsc.num_wk_slices_per_dim.get(&dim)?.get() > 1 {
            return Some(true);
        }
    }
    Some(false)
}

/// Replaces: e211_getInsertionNode
///
/// WHERE A NODE GOES RELATIVE TO A SET OF REFERENCE NODES — walks each one's parents up to the
/// innermost parent they all share, then answers that parent's first or last child on the path down
/// to a reference node.
///
/// ⛔ [`None`] IS THE REFERENCE'S `nullptr` FOR ALL FOUR OF ITS CAUSES AT ONCE: an empty set, a
/// reference node sharing no parent with the others, and the two `DT_CHECK_MSG`s a [`NodeId`] can
/// still reach — *"Parent node must be a block node."* and *"Expect node to have the same parent."*.
/// ⛔ DIVERGENCE: THE SET IS WALKED SMALLEST ID FIRST where the reference walks an `unordered_set` in
/// an unspecified order, which its own seeding of `parentNodesOuterToInner` depends on.
#[must_use]
pub fn insertion_node<T: NodeParents + ?Sized>(
    tree: &T,
    ref_nodes: &BTreeSet<NodeId>,
    side: InsertSide,
) -> Option<NodeId> {
    let first = *ref_nodes.first()?;
    let mut outer_to_inner: VecDeque<NodeId> = VecDeque::new();
    let mut curr = Some(first);
    while let Some(node) = curr {
        let parent = tree.parent(node);
        if let Some(parent) = parent {
            outer_to_inner.push_front(parent);
        }
        curr = parent;
    }
    let mut insertion: BTreeSet<NodeId> = BTreeSet::new();
    insertion.insert(first);
    for &ref_node in ref_nodes {
        if ref_node == first {
            continue;
        }
        let mut curr = Some(ref_node);
        let mut shared = false;
        while let Some(node) = curr {
            let parent = tree.parent(node);
            let at = outer_to_inner
                .iter()
                .position(|held| Some(*held) == parent);
            if let Some(at) = at {
                if at + 1 != outer_to_inner.len() {
                    // The shared parent is not the innermost held, so the set restarts from the
                    // parent's own child on the first node's path and the inner parents are dropped.
                    insertion.clear();
                    insertion.insert(outer_to_inner[at + 1]);
                    outer_to_inner.truncate(at + 1);
                }
                insertion.insert(node);
                shared = true;
                break;
            }
            curr = parent;
        }
        if !shared {
            return None;
        }
    }
    let common_parent = tree.parent(*insertion.first()?)?;
    for &node in &insertion {
        (tree.parent(node) == Some(common_parent)).then_some(())?;
    }
    let children = tree.children(common_parent);
    match side {
        InsertSide::Before => children.into_iter().find(|child| insertion.contains(child)),
        InsertSide::After => children
            .into_iter()
            .rev()
            .find(|child| insertion.contains(child)),
    }
}

/// WHERE A SYNC SEQUENCE CHAINS ITS NODES — `addChildNode(sync, /*addBefore*/ false, ref)` and the
/// position the inserted node then occupies, whichever carrier holds the tree.
///
/// ⭐ ONE PLACE FOR THE SEQUENCE. Entry 212 is reached both from an owned [`BlockNode`] and, in
/// entry 288, from a super-DSC's trees by node id; a second spelling of the four-node handshake would
/// be a second answer.
pub trait SyncInsertion {
    /// The reference position a chained insert continues from.
    type At: Copy;

    /// `parent->addChildNode(sync, /*addBefore*/ false, at)`, answering `sync`'s OWN position so a
    /// run of syncs lands in the order it was minted.
    fn insert_sync_after(&mut self, at: Self::At, sync: SyncNode) -> Self::At;
}

impl SyncInsertion for BlockNode {
    type At = ChildPos;

    fn insert_sync_after(&mut self, at: ChildPos, sync: SyncNode) -> ChildPos {
        self.insert_after(at, SchedNode::Sync(sync))
    }
}

/// THE TWO CROSS-LINKED ENDS OF ONE `sync_send_<a>_to_<b>` / `sync_receive_<b>_from_<a>` PAIR.
///
/// ⭐ THE NAMES ARE BUILT FROM [`SenComponent::spelling`], which is the same
/// `senComponentsToString` map the reference concatenates and is LOWERCASE.
fn sync_pair(
    sender: SenComponent,
    receiver: SenComponent,
    suffix: &str,
    strength: SyncStrength,
) -> (SyncNode, SyncNode) {
    let send = NodeName(format!(
        "sync_send_{}_to_{}{suffix}",
        sender.spelling(),
        receiver.spelling()
    ));
    let receive = NodeName(format!(
        "sync_receive_{}_from_{}{suffix}",
        receiver.spelling(),
        sender.spelling()
    ));
    let mut send_node = create_sync_node(
        SyncUnits::new(sender, []),
        send.clone(),
        SyncDirection::Send,
        strength,
    );
    let mut receive_node = create_sync_node(
        SyncUnits::new(receiver, []),
        receive.clone(),
        SyncDirection::Receive,
        strength,
    );
    send_node.other_ends.push(receive);
    receive_node.other_ends.push(send);
    (send_node, receive_node)
}

/// Replaces: e212_addL3LUAndLXLUSyncNodeSequence
///
/// ADDS THE FOUR-NODE L3LU/LXLU HANDSHAKE immediately after `at`: L3LU sends, LXLU receives, LXLU
/// sends, L3LU receives, each pair cross-linked as the other's other end and every one of them a HARD
/// signal.
///
/// ⭐ THE NAMES ARE BUILT FROM [`SenComponent::spelling`], which is the same
/// `senComponentsToString` map the reference concatenates and is LOWERCASE — `sync_send_l3lu_to_lxlu`
/// and its three siblings.
pub fn add_l3_lu_and_lx_lu_sync_node_sequence<I: SyncInsertion + ?Sized>(tree: &mut I, at: I::At) {
    let pair = |sender: SenComponent, receiver: SenComponent| {
        sync_pair(sender, receiver, "", SyncStrength::Hard)
    };
    let (l3_send, l3_receive) = pair(SenComponent::L3lu, SenComponent::Lxlu);
    let (lx_send, lx_receive) = pair(SenComponent::Lxlu, SenComponent::L3lu);
    let at = tree.insert_sync_after(at, l3_send);
    let at = tree.insert_sync_after(at, l3_receive);
    let at = tree.insert_sync_after(at, lx_send);
    tree.insert_sync_after(at, lx_receive);
}

#[cfg(test)]
mod tests_e205_e212 {
    use super::*;
    use crate::arch::Sen1p5;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims;
    use crate::schedule::dsc2::LayoutDims;
    use crate::schedule::l3::dsc::{
        Candidates, CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DscList, LabeledDsList,
        NamedDims, PrimaryDsInfo, StageDims,
    };
    use std::num::NonZeroU64;

    /// One labelled DS's `memOrg_` as this batch reads it, stated by field.
    #[derive(Default)]
    struct Org {
        indirection: Option<IndirectAlloc>,
        padding: Option<PaddingForm>,
        pages: BTreeMap<PrimaryDim, Extent>,
        hbm_users: Option<Vec<NodeId>>,
        lx_users: Option<Vec<NodeId>>,
    }

    impl MemOrg for Org {
        fn hbm_pinned(&self) -> bool {
            false
        }

        fn lx_buffering(&self) -> Option<Buffering> {
            None
        }

        fn lx_start_address(&self, _at: &AddressCoord) -> Option<ByteAddress> {
            None
        }

        fn lx_buffer_offset(&self, _core: Core, _corelet: Corelet) -> Option<BufferOffset> {
            None
        }

        fn hbm_indirection(&self) -> Option<IndirectAlloc> {
            self.indirection
        }

        fn hbm_allocation(&self) -> Option<NodeName> {
            None
        }

        fn hbm_layout_dims(&self) -> Option<LayoutDims> {
            None
        }

        fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim> {
            BTreeSet::new()
        }

        fn lx_padding(&self) -> Option<PaddingForm> {
            self.padding.clone()
        }

        fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent> {
            self.pages.clone()
        }

        fn hbm_alloc_users(&self) -> Option<Vec<NodeId>> {
            self.hbm_users.clone()
        }

        fn lx_alloc_users(&self) -> Option<Vec<NodeId>> {
            self.lx_users.clone()
        }

        fn lx_zero_padded(&self) -> Option<bool> {
            Some(false)
        }
    }

    /// Every labelled DS of every DSC sharing one organisation, which is all these tests state.
    struct Orgs(Org);

    impl MemOrgs for Orgs {
        type Org = Org;

        fn mem_org(&self, _dsc: DscIdx, _lds: LdsIdx) -> Option<&Org> {
            Some(&self.0)
        }
    }

    /// One DSC's transfer nodes, whichever DSC is asked for.
    struct Transfers(Vec<L3Transfer>);

    impl TransferNodes for Transfers {
        fn transfers(&self, _dsc: DscIdx) -> Vec<L3Transfer> {
            self.0.clone()
        }
    }

    /// A tree of one block, node 0, whose children are nodes 1, 2 and 3 in that order.
    struct Parents;

    impl NodeParents for Parents {
        fn parent(&self, node: NodeId) -> Option<NodeId> {
            (node != NodeId(0)).then_some(NodeId(0))
        }

        fn children(&self, parent: NodeId) -> Vec<NodeId> {
            if parent == NodeId(0) {
                vec![NodeId(1), NodeId(2), NodeId(3)]
            } else {
                Vec::new()
            }
        }
    }

    fn dims(extents: &[(PrimaryDim, i64)]) -> FilledDims {
        let mut stage = StageDims::default();
        for (dim, extent) in extents {
            stage.extents.insert(*dim, Extent(*extent));
        }
        FilledDims::of(stage).expect("a stage that states a dim")
    }

    fn stage(name: &str, extents: &[(PrimaryDim, i64)]) -> DataStage {
        let name = StageName(name.to_owned());
        DataStage {
            ss: NamedDims {
                name: name.clone(),
                dims: dims(extents),
            },
            el: NamedDims {
                name,
                dims: dims(extents),
            },
        }
    }

    /// A primary data structure that lays out the dims given, outermost first, on a one-element stick.
    fn layout(dims: &[PrimaryDim]) -> PrimaryDsInfo {
        let (first, rest) = dims.split_first().expect("a layout with a dim in it");
        PrimaryDsInfo {
            layout: LayoutDims::new(*first, rest.to_vec()),
            stick: StickDims::default(),
        }
    }

    fn labeled(ds_type: DsType, recorded: LdsIdx, dims: &[PrimaryDim]) -> LabeledDs {
        LabeledDs::new(
            ds_type,
            dims.iter().map(|dim| (*dim, Scale::Sized(1.0))).collect(),
            recorded,
            Pinning::default(),
        )
    }

    fn dsc(core: &[(PrimaryDim, i64)], chunk: &[(PrimaryDim, i64)]) -> DesignSpaceConfig {
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: Some(CoreletsUsed::ONE),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(Core::checked(0).expect("core 0"), vec![]),
            layout_dims: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(labeled(DsType::Input, LdsIdx(0), &[]), vec![]),
            data_stages: DataStages::new(stage("core", core), stage("chunk", chunk)),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        }
    }

    fn sdsc(dsc: DesignSpaceConfig, slices: &[(PrimaryDim, u32)]) -> SuperDsc {
        SuperDsc::new(
            DscList::new(dsc, vec![]),
            slices
                .iter()
                .map(|&(dim, count)| {
                    (
                        dim,
                        WkSliceCount::new(NonZeroU32::new(count).expect("a positive slice count")),
                    )
                })
                .collect(),
            BTreeMap::new(),
            BTreeMap::new(),
        )
    }

    /// e205 — any of the five weight formats is a batch matmul, and a convolution is not.
    #[test]
    fn a_bmm_is_a_bmm_of_any_weight_format() {
        assert!(is_op_func_bmm(OpFunc::BatchmatmulInt8Fwd));
        assert!(is_op_func_bmm(OpFunc::BatchmatmulXrfFp8Fwd));
        assert!(!is_op_func_bmm(OpFunc::Conv2DInt4Fwd));
    }

    /// e206 — the three reuse dims of `inp0 x inp1 -> out` take the three arms, and a dim in all
    /// three structures takes none of them.
    #[test]
    fn min_param_bmm_answers_per_reuse_arm() {
        let mut dsc = dsc(
            &[
                (PrimaryDim::Mb, 2),
                (PrimaryDim::Ki, 512),
                (PrimaryDim::I, 48),
                (PrimaryDim::J, 64),
            ],
            &[(PrimaryDim::I, 1)],
        );
        dsc.labeled_ds = LabeledDsList::new(
            labeled(DsType::Input, LdsIdx(0), &[]),
            vec![
                labeled(DsType::Kernel, LdsIdx(1), &[]),
                labeled(DsType::Output, LdsIdx(2), &[]),
            ],
        );
        dsc.primary_ds_info.insert(
            DsType::Input,
            layout(&[PrimaryDim::Mb, PrimaryDim::Ki, PrimaryDim::I]),
        );
        dsc.primary_ds_info.insert(
            DsType::Kernel,
            layout(&[PrimaryDim::Mb, PrimaryDim::Ki, PrimaryDim::J]),
        );
        dsc.primary_ds_info.insert(
            DsType::Output,
            layout(&[PrimaryDim::Mb, PrimaryDim::I, PrimaryDim::J]),
        );
        let int8 = OpFunc::BatchmatmulInt8Fwd;
        // The input channel: 512 is a multiple of 256.
        assert_eq!(
            min_param_bmm::<Sen1p5>(&dsc, PrimaryDim::Ki, int8),
            Some(Extent(256))
        );
        // The output channel.
        assert_eq!(
            min_param_bmm::<Sen1p5>(&dsc, PrimaryDim::J, int8),
            Some(Extent(64))
        );
        // Weight reuse: 48 is the largest divisor of 48 up to 64.
        assert_eq!(
            min_param_bmm::<Sen1p5>(&dsc, PrimaryDim::I, int8),
            Some(Extent(48))
        );
        // A dim every structure carries reuses nothing.
        assert_eq!(
            min_param_bmm::<Sen1p5>(&dsc, PrimaryDim::Mb, int8),
            Some(Extent(1))
        );
    }

    /// e207 — a chunk dim takes every divisor of the core extent from the lower bound up.
    #[test]
    fn dsc_param_candidates_are_the_divisors_from_the_lower_bound_up() {
        let mut dsc = dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 2)]);
        dsc.primary_ds_info
            .insert(DsType::Input, layout(&[PrimaryDim::I]));
        let sdsc = sdsc(dsc, &[]);
        let candidates = generate_dsc_param_candidates(
            &sdsc,
            &[dims(&[(PrimaryDim::I, 2)])],
            &[PrimaryDim::I],
            &BTreeSet::from([PrimaryDim::I]),
            &BTreeSet::new(),
            &Orgs(Org::default()),
            None,
        )
        .expect("every dim has a candidate");
        assert_eq!(
            candidates
                .at(DscIdx(0))
                .and_then(|per_dim| per_dim.get(PrimaryDim::I))
                .map(Candidates::extents),
            Some([Extent(2), Extent(4), Extent(8)].as_slice())
        );
    }

    /// e208 — only a transfer the allocation names and whose two storages match is kept, and an
    /// LX-pinned structure has none of them at all.
    #[test]
    fn lds_l3_transfers_are_the_alloc_users_between_the_two_storages() {
        let transfer = |node: u32, name: &str, dst: SenComponent| L3Transfer {
            node: NodeId(node),
            name: NodeName(name.to_owned()),
            src: SenComponent::Hbm,
            dst,
        };
        let trees = Transfers(vec![
            transfer(7, "hbm_to_lx", SenComponent::Lx),
            transfer(9, "not_a_user", SenComponent::Lx),
            transfer(7, "wrong_destination", SenComponent::Hbm),
        ]);
        let org = Org {
            hbm_users: Some(vec![NodeId(7)]),
            ..Org::default()
        };
        let mut hbm = dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 2)]);
        hbm.labeled_ds = LabeledDsList::new(
            LabeledDs::new(
                DsType::Input,
                Vec::new(),
                LdsIdx(0),
                Pinning {
                    mem_org: BTreeMap::from([(SenComponent::Hbm, true)]),
                    ..Pinning::default()
                },
            ),
            vec![],
        );
        assert_eq!(
            lds_l3_transfer_nodes(
                &sdsc(hbm, &[]),
                DscIdx(0),
                LdsIdx(0),
                &org,
                &trees,
                &[SenComponent::Hbm],
                &[SenComponent::Lx],
            ),
            Some(vec![transfer(7, "hbm_to_lx", SenComponent::Lx)])
        );
        let mut pinned = dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 2)]);
        pinned.labeled_ds = LabeledDsList::new(
            LabeledDs::new(
                DsType::Input,
                Vec::new(),
                LdsIdx(0),
                Pinning {
                    lx: true,
                    ..Pinning::default()
                },
            ),
            vec![],
        );
        assert_eq!(
            lds_l3_transfer_nodes(
                &sdsc(pinned, &[]),
                DscIdx(0),
                LdsIdx(0),
                &org,
                &trees,
                &[SenComponent::Hbm],
                &[SenComponent::Lx],
            ),
            Some(Vec::new())
        );
    }

    /// e209 — four sticks of the chunk, and the walk stops there because the chunk is below the core
    /// extent.
    #[test]
    fn chunk_stick_volume_multiplies_until_the_chunk_falls_below_the_core() {
        let mut dsc = dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 4)]);
        dsc.labeled_ds = LabeledDsList::new(labeled(DsType::Input, LdsIdx(0), &[PrimaryDim::I]), vec![]);
        dsc.layout_dims
            .insert(LdsIdx(0), LayoutDims::new(PrimaryDim::I, vec![]));
        dsc.primary_ds_info
            .insert(DsType::Input, layout(&[PrimaryDim::I]));
        let org = Org {
            padding: Some(PaddingForm::default()),
            ..Org::default()
        };
        assert_eq!(
            labeled_ds_chunk_stick_volume(&dsc, LdsIdx(0), &org),
            Some(StickVolume::new(
                NonZeroU64::new(4).expect("a positive volume")
            ))
        );
    }

    /// e210 — a reduced dim split over two work slices reduces across cores, and a reduced dim the
    /// super-DSC states no slice count for is the `.at()` throw.
    #[test]
    fn cross_core_reduction_is_a_reduced_dim_on_more_than_one_slice() {
        let mut dsc = dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 2)]);
        dsc.labeled_ds = LabeledDsList::new(
            labeled(DsType::Input, LdsIdx(0), &[PrimaryDim::I, PrimaryDim::Ki]),
            vec![labeled(DsType::Output, LdsIdx(1), &[PrimaryDim::I])],
        );
        dsc.layout_dims.insert(
            LdsIdx(0),
            LayoutDims::new(PrimaryDim::I, vec![PrimaryDim::Ki]),
        );
        dsc.layout_dims
            .insert(LdsIdx(1), LayoutDims::new(PrimaryDim::I, vec![]));
        let split = sdsc(dsc.clone(), &[(PrimaryDim::Ki, 2)]);
        assert_eq!(is_op_cross_core_reduction(&split, &dsc), Some(true));
        let solo = sdsc(dsc.clone(), &[(PrimaryDim::Ki, 1)]);
        assert_eq!(is_op_cross_core_reduction(&solo, &dsc), Some(false));
        assert_eq!(is_op_cross_core_reduction(&sdsc(dsc.clone(), &[]), &dsc), None);
    }

    /// e211 — the insertion point is the common parent's first or last child on the way down to a
    /// reference node.
    #[test]
    fn the_insertion_node_is_the_outermost_child_on_the_chosen_side() {
        let refs = BTreeSet::from([NodeId(1), NodeId(3)]);
        assert_eq!(
            insertion_node(&Parents, &refs, InsertSide::Before),
            Some(NodeId(1))
        );
        assert_eq!(
            insertion_node(&Parents, &refs, InsertSide::After),
            Some(NodeId(3))
        );
        assert_eq!(
            insertion_node(&Parents, &BTreeSet::new(), InsertSide::Before),
            None
        );
    }

    /// e212 — THE EMISSION: four sync nodes land in the tree after the node named, each cross-linked
    /// to the other end of its own signal.
    #[test]
    fn the_sync_sequence_adds_four_cross_linked_nodes_to_the_tree() {
        let mut parent = BlockNode {
            name: NodeName("block".to_owned()),
            children: vec![SchedNode::Leaf(NodeName("transfer".to_owned()))],
        };
        let at = parent
            .child_pos(&NodeName("transfer".to_owned()))
            .expect("the child the sequence is added after");
        add_l3_lu_and_lx_lu_sync_node_sequence(&mut parent, at);
        let names: Vec<&str> = parent
            .children
            .iter()
            .map(|child| child.name().0.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "transfer",
                "sync_send_l3lu_to_lxlu",
                "sync_receive_lxlu_from_l3lu",
                "sync_send_lxlu_to_l3lu",
                "sync_receive_l3lu_from_lxlu",
            ]
        );
        let SchedNode::Sync(first) = &parent.children[1] else {
            panic!("the sequence adds sync nodes");
        };
        assert_eq!(first.direction, SyncDirection::Send);
        assert_eq!(first.strength, SyncStrength::Hard);
        assert_eq!(
            first.other_ends,
            vec![NodeName("sync_receive_lxlu_from_l3lu".to_owned())]
        );
        assert_eq!(
            first.units.iter().collect::<Vec<_>>(),
            vec![SenComponent::L3lu]
        );
    }
}

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

// ⭐ TYPES FOR ENTRIES 221-228. The indirect-access stage's own vocabulary — which LX buffering the
// paged chunk loops sit under, which way an IBR transfer runs, and the four seams through which the
// stage reaches an environment this campaign's file list does not contain.

/// WHICH LX BUFFERING THE ORIGINAL CHUNK LOOP RUNS UNDER — `lxBufferType`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:220`) reduced to the two arms entry 225 admits, each
/// carrying the denominator data stage it demands of that loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LxBufferType {
    /// `BufferType::DOUBLE` — the chunk loop's denominator is [`DATA_STAGE_CHUNK`].
    Double,
    /// `BufferType::SPATIAL_DOUBLE` — its denominator is the superchunk stage.
    SpatialDouble(SuperChunkStage),
}

impl LxBufferType {
    /// The denominator stage entry 225's *"Expect a valid chunk loop."* demands.
    #[must_use]
    pub const fn den(self) -> DatastageId {
        match self {
            Self::Double => DATA_STAGE_CHUNK,
            Self::SpatialDouble(stage) => stage.index(),
        }
    }
}

/// WHICH WAY AN INDIRECT L3 TRANSFER RUNS — entries 226 and 227's `isTransferIn`, whose four
/// `isTransferIn ? .. : ..` component picks are these three methods.
///
/// ⭐ THE UNIT IS PICKED THE SAME WAY TWICE: entry 226 computes `srcUnit` and `dstUnit` from the same
/// flag with the same arms, so a transfer that stages an index never crosses units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IbrDirection {
    /// `isTransferIn == true` — HBM in to LX, through the L3 load unit.
    In,
    /// `isTransferIn == false` — LX out to HBM, through the L3 store unit.
    Out,
}

impl IbrDirection {
    /// `srcUnit`, which is also `dstUnit`.
    #[must_use]
    pub const fn unit(self) -> SenComponent {
        match self {
            Self::In => SenComponent::L3lu,
            Self::Out => SenComponent::L3su,
        }
    }

    /// `srcStorage` — where the index tensor is read from.
    #[must_use]
    pub const fn src_storage(self) -> SenComponent {
        match self {
            Self::In => SenComponent::Hbm,
            Self::Out => SenComponent::Lx,
        }
    }

    /// `dstStorage` — the indirect buffer register the index is staged into.
    #[must_use]
    pub const fn dst_storage(self) -> SenComponent {
        match self {
            Self::In => SenComponent::L3luibr,
            Self::Out => SenComponent::L3suibr,
        }
    }
}

/// HOW MANY HMI REQUESTS ONE SUPER-DSC'S HBM TRANSFERS COME TO — entry 223's `int`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HmiRequests(pub u32);

impl HmiRequests {
    /// `std::numeric_limits<int>::max()` — the seed entry 223 hands back UNTOUCHED when no HBM tensor
    /// narrowed it, which is "no bound" and not a count of requests.
    pub const UNBOUNDED: Self = Self(2_147_483_647);
}

/// ONE EXECUTION PHASE — an entry of `exphases`, which entry 222 places each allocation in separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExPhase(pub u32);

/// ONE L3 MEMORY TRACKER — `memTrackers->getTracker(comp, core, corelet, row)`.
///
/// ⛔ KEYED BY A [`SenComponent`] WHERE [`v1::TrackerSite`] KEYS BY A `DdcMemory`: this stage
/// allocates in LX and in register files that the DDC memory vocabulary does not name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct L3TrackerSite {
    /// `comp`.
    pub memory: SenComponent,
    /// `core`.
    pub core: Core,
    /// `corelet`.
    pub corelet: Corelet,
    /// `row`.
    pub row: Row,
}

/// THE PAGED TENSOR'S HBM ALLOCATION AS ENTRY 226 READS IT — `indexLdsHbmAllocNode` reduced to its
/// identity and the two indirection fields the IBR allocation inherits from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexHbmAllocation {
    /// The identity the tree holds it under — what `addAllocUser` is called on.
    pub alloc: AllocId,
    /// `indirectAllocType_`.
    pub indirect: Option<IndirectAlloc>,
    /// `relatedIndirectAccessAlloc_`.
    pub related_indirect: Option<AllocId>,
}

/// THE REFERENCE ALLOCATION ENTRY 228 COPIES A COORDINATE FROM — `refAllocNode` reduced to the three
/// facts it is read for.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceAllocation<'a> {
    /// `allocateCoordinates_`.
    pub coordinate: &'a Coordinate,
    /// `ldsIdx_`.
    pub lds: LdsIdx,
    /// `labeledDs_.at(ldsIdx_)` — where the dim's broadcast scale is read.
    pub labeled_ds: &'a LabeledDs,
}

/// WHAT ENTRIES 225-227 ASK OF A SCHEDULE TREE BEYOND [`LoopBands`] — the mints, the links and the
/// two write-backs this stage performs, every one of them `dsc2::ScheduleNode` MECHANISM rather than
/// an L3 scheduling decision.
pub trait L3TreeSurgery: LoopBands {
    /// `parent->deleteChildNode(&dsc, node)`.
    fn delete_child_node(&mut self, node: NodeId);
    /// The identity `new dsc2::AllocateNode()` issues.
    fn fresh_alloc(&mut self) -> AllocId;
    /// That allocate node, held under that identity, UNLINKED.
    fn new_allocate(&mut self, alloc: AllocId, node: L3AllocateNode) -> NodeId;
    /// `new dsc2::TransferNode(..)`, unlinked.
    fn new_transfer(&mut self, node: TransferNode) -> NodeId;
    /// `new dsc2::SyncNode(..)`, unlinked.
    fn new_sync(&mut self, node: SyncNode) -> NodeId;
    /// `sync->otherEndOfTheSignals_.push_back(other)`.
    fn add_sync_other_end(&mut self, sync: NodeId, other: NodeId);
    /// `alloc->addAllocUser(user)`.
    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId);
    /// `labeledDs_.at(lds).memOrg_[storage].allocateNode_ = alloc`.
    fn set_mem_org_allocation(&mut self, lds: LdsIdx, storage: SenComponent, alloc: AllocId);
    /// The transfer node written back — [`ScheduleSurgery::transfer`] hands one out BY VALUE.
    fn set_transfer(&mut self, node: NodeId, transfer: TransferNode);
}

/// WHAT ENTRY 221 ASKS OF THE SUPER-DSC'S SCHEDULE TREES — the per-DSC `memOrg_` walk and the LX
/// allocation's transfer users, which are the MECHANISM for reaching those transfers rather than a
/// fact about their padding.
pub trait DscTransfers {
    /// What one labelled DS's `memOrg_` answers.
    type Org: MemOrg;
    /// That DSC's `labeledDs_` organisations, POSITIONALLY beside
    /// [`crate::schedule::l3::dsc::LabeledDsList::indexed`].
    fn mem_orgs(&self, dsc: DscIdx) -> Vec<&Self::Org>;
    /// `memOrg_.at(LX).allocateNode_->allocUsers_` restricted to its TRANSFER users, for the labelled
    /// DS at that POSITION.
    fn lx_alloc_transfer_users(&self, dsc: DscIdx, lds: LdsIdx) -> Vec<NodeId>;
    /// That transfer node, absent where the DSC's tree holds no such node.
    fn transfer(&self, dsc: DscIdx, node: NodeId) -> Option<TransferNode>;
    /// The same node written back.
    fn set_transfer(&mut self, dsc: DscIdx, node: NodeId, transfer: TransferNode);
}

/// WHAT ENTRY 222 ASKS OF THE MEMORY TRACKERS — `DsTrackInMem` (`ddc/memTracker.h`), reached PER
/// EXECUTION PHASE and outside this campaign's file list.
///
/// ⛔ [`None`] FROM [`Self::check_and_add`] IS THE `EXISTS` ANSWER, which the reference `DT_CHECK`s: a
/// name already in the tracker means this set is being placed twice over itself.
pub trait ExPhaseTrackers {
    /// `exphases` — the phases each allocation is placed in separately.
    fn ex_phases(&self) -> Vec<ExPhase>;
    /// `memCapacity`.
    fn capacity(&self, at: L3TrackerSite) -> Bytes;
    /// `backupEps(exphase)` for every phase, IDEMPOTENT — `trackerBackups.try_emplace` is that.
    fn backup(&mut self, at: L3TrackerSite);
    /// `restoreEps(exphase, backupInfo)` for every tracker backed up since.
    fn restore_all(&mut self);
    /// `removeDs(name, exphases)`.
    fn remove(&mut self, at: L3TrackerSite, name: &v1::StorageName);
    /// `checkAndAddDs(name, size, {exphase})`.
    fn check_and_add(
        &mut self,
        at: L3TrackerSite,
        phase: ExPhase,
        name: &v1::StorageName,
        size: Bytes,
    ) -> Option<v1::Placed>;
}

/// WHAT ENTRY 222 ASKS OF THE DESIGN SPACE AND THE SUPER-DSC'S FOLD PROPS — all of it
/// `dsc/designSpaceConfig.h`, outside this campaign's file list.
pub trait L3Placement {
    /// `getBufferCapacityForNode(node, lds, comp, corelet, row, bytesPerStick,
    /// /*forceEvenNumSticks*/ true)` — the L3 rounds to an EVEN stick count for ring polarity.
    fn buffer_capacity_even_sticks(
        &self,
        alloc: AllocId,
        lds: LdsIdx,
        corelet: Corelet,
        row: Row,
    ) -> Bytes;
    /// `{coreFoldProp_, coreletFoldProp_} ++ sdscFoldProps_`'s size — how many axes the address fold
    /// space has.
    fn address_fold_depth(&self) -> usize;
    /// `getFlattenedCoordinates({{0, 0}, {1, 0}}).size()` — how many coordinates one `(core, corelet)`
    /// spreads its address list over.
    fn address_fold_coords(&self) -> usize;
}

/// WHAT ENTRY 228 ASKS BESIDE THE DISTRIBUTION PASS — the parametric iteration count and the two
/// datastage reads, all outside this campaign's file list.
///
/// ⭐ `distributeElemArrToTemporalLoops` ITSELF IS [`TemporalLoopDistribution`], the seam entry 241
/// and entry 229 already speak; a second spelling of it would be a second answer.
pub trait AllocCoordinateSeam: TemporalLoopDistribution {
    /// `loop->parametricIterCount(dsc, 0, NO_COMPONENT, -1)` — [`None`] for a loop that is NOT
    /// parametric, which is the arm that reads the datastages instead.
    fn parametric_iter_count(&self, loop_node: &LoopNode) -> Option<FoldCardinality>;
    /// `dataStageParam_.at(stage).ss_.dataStageDimToVal_compView_st(dim, L3LU, 0)`.
    fn comp_view(&self, stage: DatastageId, dim: PrimaryDim) -> Option<Extent>;
    /// `dataStageParam_.at(stage).ss_.paddingSizes_` — the DENOMINATOR stage's, which is
    /// [`AccessPad::Padded`]'s other half.
    fn stage_padding(&self, stage: DatastageId) -> Option<&BTreeMap<PrimaryDim, DimPadding>>;
}

/// Replaces: e221_fillTransferZeroPaddingInfo
///
/// FILLS EVERY HBM-OR-CONSTANT-TO-LX TRANSFER'S ZERO-PAD FOLD SPACE: for each window-padded dim, the
/// work-slice and chunk fold axes with their cardinalities and affine pairs, and then the source and
/// destination units the padded transfer must name.
///
/// ⛔ [`None`] IS *"Do not support both paging and windowed-padding on the same dimension."*, *"Invalid
/// chunk parameter."*, both *"Expect a positive .. offset."*, *"Expect paddingSizes_ entry."* and the
/// [`MemOrg::lx_zero_padded`] check. ⚠️ TRAP: the `hasWindowPad` scan breaks ONLY inside the
/// `windowDim_ != Count` arm — a padded but windowless layout dim CONTINUES the scan.
pub fn fill_transfer_zero_padding_info<E: DscTransfers + ?Sized>(
    sdsc: &SuperDsc,
    env: &mut E,
) -> Option<()> {
    let mut writes: Vec<(DscIdx, NodeId, TransferNode)> = Vec::new();
    for (dsc, index) in sdsc.dscs().iter().zip(0u32..) {
        let dsc_idx = DscIdx(index);
        let orgs = env.mem_orgs(dsc_idx);
        let paged = get_paged_dimensions(&orgs);
        for padding in dsc.full_padding.values() {
            if padding.window_dim.is_some_and(|window| paged.contains(&window)) {
                return None;
            }
        }

        let core_stage = dsc.core_stage().dims();
        let chunk_stage = dsc.data_stages.chunk().ss.dims.dims();
        let padded_dims: Vec<(PrimaryDim, PadElems, PadElems)> = dsc
            .full_padding
            .iter()
            .filter_map(|(&dim, padding)| match padding.sizes {
                PadSizes::Sized { front, back } if padding.window_dim.is_some() => {
                    Some((dim, front, back))
                }
                _ => None,
            })
            .collect();

        // The L3 transfer needs LX zero padding when the tensor's LX organisation zero-pads, one of
        // its padded layout dims is related to a window dim, and the transfer reaches LX from HBM or
        // from a constant.
        let mut zero_pad_transfers: Vec<NodeId> = Vec::new();
        for ((position, entry), org) in dsc.labeled_ds.indexed().zip(&orgs) {
            if !org.lx_zero_padded()? {
                continue;
            }
            let lx_padding = org.lx_padding()?;
            let mut has_window_pad = false;
            for dim in dsc.layout_dims.get(&entry.recorded())?.iter() {
                if lx_padding.padding(dim) != PadType::NoPad
                    && dsc.full_padding.get(&dim)?.window_dim.is_some()
                {
                    has_window_pad = true;
                    break;
                }
            }
            if has_window_pad {
                zero_pad_transfers.extend(env.lx_alloc_transfer_users(dsc_idx, position));
            }
        }

        for node in zero_pad_transfers {
            let mut transfer = env.transfer(dsc_idx, node)?;
            if !matches!(
                transfer.src.storage,
                SenComponent::Hbm | SenComponent::NoComponent
            ) || transfer.dsts.first().storage != SenComponent::Lx
            {
                continue;
            }
            let from_hbm = transfer.src.storage == SenComponent::Hbm;
            for &(dim, pad_front, pad_back) in &padded_dims {
                let slices = sdsc.num_wk_slices_per_dim.get(&dim)?.get();
                let core_extent = core_stage.extent(dim)?.0;
                // The LX size is the CHUNK stage's for an HBM transfer and the CORE stage's otherwise.
                let lx_extent = if from_hbm {
                    chunk_stage.extent(dim)?.0
                } else {
                    core_extent
                };
                if lx_extent == 0 || core_extent % lx_extent != 0 {
                    return None;
                }
                let chunks = core_extent / lx_extent;
                let core_offset = core_extent.checked_mul(core_stage.padding.get(&dim)?.stride.0)?;
                let chunk_offset = lx_extent.checked_mul(chunk_stage.padding.get(&dim)?.stride.0)?;
                if core_offset <= 0 || chunk_offset <= 0 {
                    return None;
                }
                let cardinalities = (
                    FoldCardinality(slices),
                    FoldCardinality(u32::try_from(chunks).ok()?),
                );
                transfer.padding.build_pad_front(
                    dim,
                    ZeroPadFolds {
                        work_slice: PadFold {
                            cardinality: cardinalities.0,
                            alpha: FoldCoeff(-core_offset),
                            beta: FoldCoeff(i64::from(pad_front.0)),
                        },
                        chunk: PadFold {
                            cardinality: cardinalities.1,
                            alpha: FoldCoeff(-chunk_offset),
                            beta: FoldCoeff(0),
                        },
                    },
                );
                let back_beta = i64::from(pad_back.0)
                    - core_offset.checked_mul(i64::from(slices) - 1)?
                    - chunk_offset.checked_mul(chunks - 1)?;
                transfer.padding.build_pad_back(
                    dim,
                    ZeroPadFolds {
                        work_slice: PadFold {
                            cardinality: cardinalities.0,
                            alpha: FoldCoeff(core_offset),
                            beta: FoldCoeff(back_beta),
                        },
                        chunk: PadFold {
                            cardinality: cardinalities.1,
                            alpha: FoldCoeff(chunk_offset),
                            beta: FoldCoeff(0),
                        },
                    },
                );
            }
            if transfer.src.unit == SenComponent::NoComponent {
                transfer.src.unit = SenComponent::Constant;
            }
            if transfer.src.storage == SenComponent::NoComponent {
                transfer.src.storage = SenComponent::Constant;
            }
            transfer.dsts.first_mut().unit = SenComponent::L3lu;
            writes.push((dsc_idx, node, transfer));
        }
    }
    for (dsc_idx, node, transfer) in writes {
        env.set_transfer(dsc_idx, node, transfer);
    }
    Some(())
}

/// WHAT ENTRY 222'S `tryAlloc` COLLECTED — held off the allocations until every set has fitted,
/// because a probe that did not fit must leave them as they were.
///
/// ⭐ ONE `(core, corelet)` HOLDS A LIST AND NOT ONE ADDRESS: each execution phase is placed
/// separately, and the list is later spread over that site's remaining fold coordinates.
#[derive(Debug, Default)]
struct L3Placements {
    /// `startAddressCoreCorelet_`.
    start: BTreeMap<AllocId, BTreeMap<Core, BTreeMap<Corelet, Vec<Bytes>>>>,
    /// `bufferOffsetCoreCorelet_`.
    offsets: BTreeMap<AllocId, BTreeMap<Core, BTreeMap<Corelet, Bytes>>>,
    /// `copyToCoreCl`, whose corelet half is the CONSTANT `true` here.
    copied_from: BTreeMap<AllocId, v1::Proxy>,
}

/// `tryAlloc` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5521-5665`) — [`None`] is a `DT_CHECK`,
/// `Some(false)` its `return false`.
///
/// ⛔ THE `consIdAndAllocNode` AND `compAndAllocNode` ARMS ARE DEAD TWICE OVER: both open with
/// `DT_ERROR("No support")`, and neither map exists in this stage's [`L3Allocation`] projection.
fn try_alloc_l3<M, P>(
    dsc: &DesignSpaceConfig,
    metadata: &DscMetadata,
    allocs: &v1::AllocArena,
    trackers: &mut M,
    placement: &P,
    commit: v1::Commit,
    placed: &mut L3Placements,
) -> Option<bool>
where
    M: ExPhaseTrackers + ?Sized,
    P: L3Placement + v1::StorageNames,
{
    let phases = trackers.ex_phases();
    for (&memory, allocation) in &metadata.new_allocations {
        // For non-LX memories the first core stands proxy for all of them.
        let (cores, copy_core) = if memory == SenComponent::Lx {
            (dsc.core_ids_used.iter().collect(), v1::Proxy::Each)
        } else {
            (vec![dsc.core_ids_used.first()], v1::Proxy::First)
        };
        let cores: Vec<Core> = cores;
        for core in cores {
            // Corelets and rows use 0 as proxy.
            let at = L3TrackerSite {
                memory,
                core,
                corelet: Corelet::at::<0>(),
                row: Row::at::<0>(),
            };
            trackers.backup(at);
            let mut node_and_size: Vec<(AllocId, Bytes)> = Vec::new();
            for (&lds, &alloc) in &allocation.lds_idx_and_alloc_node {
                let node = allocs.get(&alloc)?;
                if node.component != SenComponent::Lx {
                    return None;
                }
                // The LX buffer must be an even number of sticks for ring polarity (refer to DSI).
                let capacity =
                    placement.buffer_capacity_even_sticks(alloc, lds, at.corelet, at.row);
                let buffers = node.placement.num_buffers.reserved();
                node_and_size.push((alloc, Bytes(capacity.0.checked_mul(buffers.get())?)));
            }
            // Largest first: LX already holds tensors from other nodes, and placing the big buffers
            // before the small ones is what keeps that fragmentation from costing a buffer.
            node_and_size.sort_by(|left, right| right.1.cmp(&left.1));
            for &(alloc, _) in &node_and_size {
                let name = get_lds_or_const_name_of_alloc_node(allocs.get(&alloc)?, placement)?;
                trackers.remove(at, &name);
            }
            for &(alloc, size) in &node_and_size {
                let node = allocs.get(&alloc)?;
                let mut my_size = size;
                if node.placement.num_buffers.is_streaming() {
                    // Full capacity reserved for a circular buffer.
                    my_size = my_size.max(trackers.capacity(at));
                }
                let name = get_lds_or_const_name_of_alloc_node(node, placement)?;
                let mut addresses: Vec<Bytes> = Vec::new();
                for &phase in &phases {
                    match trackers.check_and_add(at, phase, &name, my_size)? {
                        v1::Placed::At(address) => addresses.push(address),
                        v1::Placed::DoesntFit => return Some(false),
                    }
                }
                if commit == v1::Commit::IfValid {
                    if addresses.windows(2).all(|pair| pair[0] == pair[1]) {
                        // Every phase placed it at the same address, so keep only one.
                        addresses.truncate(1);
                    }
                    placed
                        .start
                        .entry(alloc)
                        .or_default()
                        .entry(core)
                        .or_default()
                        .insert(at.corelet, addresses);
                    // ⚠️ THE SIZE ASKED FOR, not the capacity a streaming buffer widened it to.
                    placed
                        .offsets
                        .entry(alloc)
                        .or_default()
                        .entry(core)
                        .or_default()
                        .insert(
                            at.corelet,
                            Bytes(size.0 / node.placement.num_buffers.reserved()),
                        );
                    placed.copied_from.insert(alloc, copy_core);
                }
            }
        }
    }
    Some(true)
}

/// Replaces: e222_allocAllMem
///
/// Places every new LX allocation of one DSC in its memory tracker once PER EXECUTION PHASE and —
/// when it all fitted and the caller asked to commit — writes the addresses into each allocation's
/// fold space, its buffer offsets beside them, and copies those offsets out of every proxy site.
///
/// ⛔ [`None`] IS *"Expect only LX."*, the `EXISTS` a tracker already holding the name answers, a fold
/// space already dimensioned, a proxied core placed at more than one coordinate, and an address list
/// that is neither single nor one per fold coordinate. ⚠️ The `coreArch <= MPW4_ISA` PTARF prefill is
/// dead twice over: [`IsaGen`] has no MPW4, and this stage only ever allocates in LX.
pub fn alloc_all_mem<M, P>(
    dsc: &DesignSpaceConfig,
    metadata: &BTreeMap<DscIdx, DscMetadata>,
    dsc_idx: DscIdx,
    allocs: &mut v1::AllocArena,
    trackers: &mut M,
    placement: &P,
    commit: v1::Commit,
) -> Option<bool>
where
    M: ExPhaseTrackers + ?Sized,
    P: L3Placement + v1::StorageNames,
{
    let mut placed = L3Placements::default();
    let success = try_alloc_l3(
        dsc,
        metadata.get(&dsc_idx)?,
        allocs,
        trackers,
        placement,
        commit,
        &mut placed,
    )?;
    if !success || commit == v1::Commit::No {
        trackers.restore_all();
        return Some(success);
    }

    let depth = placement.address_fold_depth();
    let coords = placement.address_fold_coords();
    for (&alloc, addresses) in &placed.start {
        let copy_core = *placed.copied_from.get(&alloc)?;
        let default = if addresses
            .values()
            .flat_map(|per_corelet| per_corelet.values())
            .any(|list| list.len() > 1)
        {
            AddressFold::Map
        } else {
            AddressFold::Constant
        };
        // `foldTypes[0] = copyCore ? Constant : Map`; the corelet axis ALWAYS maps, because corelet
        // 1's address is computed later.
        let core_fold = match copy_core {
            v1::Proxy::First => AddressFold::Constant,
            v1::Proxy::Each => AddressFold::Map,
        };
        let node = allocs.get_mut(&alloc)?;
        if !node.start_address.has_zero_fold_dim()
            || (copy_core == v1::Proxy::First && addresses.len() != 1)
        {
            return None;
        }
        node.start_address
            .build_fold_space_spread(depth, default, core_fold, AddressFold::Map);
        for (&core, per_corelet) in addresses {
            for (&corelet, list) in per_corelet {
                match list.as_slice() {
                    [only] => node.start_address.insert(core, corelet, *only),
                    spread if spread.len() == coords => {
                        node.start_address.insert_spread(core, corelet, spread.to_vec());
                    }
                    _ => return None,
                }
            }
        }
    }
    for (&alloc, offsets) in &placed.offsets {
        allocs.get_mut(&alloc)?.placement.buffer_offset = offsets.clone();
    }
    for (&alloc, &copy_core) in &placed.copied_from {
        let node = allocs.get_mut(&alloc)?;
        // ⚠️ `copyCorelet` IS THE UNCONDITIONAL `true` here, so every allocation's offsets copy out.
        for per_corelet in node.placement.buffer_offset.values_mut() {
            let head = *per_corelet.get(&Corelet::at::<0>())?;
            for index in 1..dsc.corelets_used.get() {
                per_corelet.insert(Corelet::checked(index)?, head);
            }
        }
        if copy_core == v1::Proxy::First && node.placement.num_buffers.switches() {
            let at_head = node
                .placement
                .buffer_offset
                .get(&dsc.core_ids_used.first())?
                .clone();
            for core in dsc.core_ids_used.iter().skip(1) {
                node.placement.buffer_offset.insert(core, at_head.clone());
            }
        }
    }
    Some(true)
}

/// Replaces: e223_getHbmLdsTransferHMIRequestEstimate
///
/// The estimated HMI request count for the super-DSC's HBM transfers — the smallest work-slice product
/// over the HBM-pinned tensors before SEN1P5, and the smallest HMI core group on it.
///
/// ⛔ *"Unsupported SENARCH."* IS UNSPELLABLE: [`IsaGen`] names exactly two generations, so the
/// reference's three-way branch is an exhaustive two-arm match. ⛔ [`None`] is a work-slice count the
/// super-DSC does not state, and the `int` overflow of their product.
#[must_use]
pub fn get_hbm_lds_transfer_hmi_request_estimate<A: Arch, T: ScheduleTrees + ?Sized>(
    sdsc: &SuperDsc,
    trees: &T,
) -> Option<HmiRequests> {
    // The first DSC states every HBM-pinned tensor.
    let main = sdsc.dscs().first();
    let hbm_lds: Vec<LdsIdx> = hbm_pinned_labeled_ds_indices(main).into_iter().collect();
    match A::GEN {
        // One HMI only, so all work slices share it and their count is the estimate.
        IsaGen::Rcudd1a => {
            let mut min_requests = HmiRequests::UNBOUNDED;
            for &lds in &hbm_lds {
                let mut slices = 1u32;
                for dim in main.non_broadcast_lds_dims(lds)? {
                    slices = slices.checked_mul(sdsc.num_wk_slices_per_dim.get(&dim)?.get())?;
                }
                min_requests = min_requests.min(HmiRequests(slices));
            }
            Some(min_requests)
        }
        IsaGen::Sen1p5 => Some(
            compute_min_hmi_core_group_size_for_sen1p5::<A, T>(sdsc, trees, &hbm_lds)
                .map_or(HmiRequests::UNBOUNDED, |group| HmiRequests(group.0)),
        ),
    }
}

/// Replaces: e224_getAllPagedLdsIndices
///
/// Every PAGED labelled DS of the DSC, by the index each one RECORDS, in `labeledDs_` order.
#[must_use]
pub fn get_all_paged_lds_indices<M: MemOrg + ?Sized>(labeled_ds: &[(LdsIdx, &M)]) -> Vec<LdsIdx> {
    labeled_ds
        .iter()
        .filter(|(_, org)| is_paged_lds(*org))
        .map(|&(lds, _)| lds)
        .collect()
}

/// Replaces: e225_createPagedDimChunkLoops
///
/// REPLACES EACH PAGED DIM'S CHUNK LOOP WITH A core/ibr → ibr/chunk NEST: mints both loops, links the
/// outer one where the original sat, moves the original's children under the inner one, drops the
/// original from the chunk-loop set and deletes it. The answer names the INNER loop per dim.
///
/// ⛔ [`None`] IS *"Only support one dimension in a chunk loop node for now."* — which fires for every
/// candidate SCANNED, not just the one that matches — together with *"Expect a valid chunk loop."* and
/// the root-node parent walk.
pub fn create_paged_dim_chunk_loops<T: L3TreeSurgery + ?Sized>(
    tree: &mut T,
    core: &CoreWindowDims,
    lx_buffer: LxBufferType,
    ibr: IbrStage,
    paged_dims: &[PrimaryDim],
    chunk_loops: &mut BTreeSet<LoopId>,
) -> Option<BTreeMap<PrimaryDim, LoopId>> {
    let mut new_chunk_loops: BTreeMap<PrimaryDim, LoopId> = BTreeMap::new();
    for &dim in paged_dims {
        // Expand the chunk loop into a loop nest.
        let mut found = None;
        for &candidate in chunk_loops.iter() {
            let dims = tree.loop_dims(candidate);
            let mut entries = dims.iter();
            let only = entries.next()?;
            if entries.next().is_some() {
                return None;
            }
            if only.dim == dim {
                found = Some(candidate);
                break;
            }
        }
        let original = found?;
        let num = tree.loop_num(original);
        let den = tree.loop_den(original);
        if num != CoreWindowDims::CORE || den != lx_buffer.den() {
            return None;
        }
        tree.parent(original.0)?;

        let core_ibr = tree.new_loop(create_loop_node(
            core,
            dim,
            &[],
            num,
            ibr.index(),
            NodeName(format!(
                "loop_core_ibr_ds{}_ds{}_{}",
                num.0,
                ibr.index().0,
                dim.spelling()
            )),
        ));
        let ibr_chunk = tree.new_loop(create_loop_node(
            core,
            dim,
            &[],
            ibr.index(),
            den,
            NodeName(format!(
                "loop_ibr_chunk_ds{}_ds{}_{}",
                ibr.index().0,
                den.0,
                dim.spelling()
            )),
        ));

        // Connect the new nest in place of the original chunk loop.
        tree.add_child_node(core_ibr.0, InsertionPoint::Before(original.0));
        tree.add_child_node(ibr_chunk.0, InsertionPoint::LastIn(core_ibr.0));
        tree.move_children(original.0, ibr_chunk.0);
        chunk_loops.remove(&original);
        tree.delete_child_node(original.0);
        new_chunk_loops.insert(dim, ibr_chunk);
    }
    Some(new_chunk_loops)
}

/// Replaces: e226_createStoreIndexTensorToIbr
///
/// STAGES A PAGED TENSOR'S INDEX INTO THE IBR: mints the IBR allocation carrying the HBM allocation's
/// indirection, the transfer into it and a self-sync send/receive pair on the L3 unit, records the
/// allocation in the labelled DS's `memOrg_`, and links all four before the new chunk loop.
///
/// ⛔ [`None`] IS *"Expect a valid parent node."*, an lds the DSC does not state, and entry 016's own
/// refusals. ⚠️ TRAP: the allocate and transfer names spell the RECORDED `ldsIdx_` while the sync names
/// spell the POSITION the entry sits at — the reference reads the two off different variables.
pub fn create_store_index_tensor_to_ibr<T: L3TreeSurgery + ?Sized>(
    tree: &mut T,
    dsc: &DesignSpaceConfig,
    metadata: &mut BTreeMap<DscIdx, DscMetadata>,
    dsc_idx: DscIdx,
    index_lds: LdsIdx,
    index_hbm: IndexHbmAllocation,
    new_chunk_loop: LoopId,
    direction: IbrDirection,
) -> Option<()> {
    tree.parent(new_chunk_loop.0)?;
    let recorded = dsc.labeled_ds.at(index_lds)?.recorded();
    let unit = direction.unit();
    let src_storage = direction.src_storage();
    let dst_storage = direction.dst_storage();

    let fresh = FreshL3Allocation::of(dsc, index_lds, dst_storage)?;
    let alloc = tree.fresh_alloc();
    let mut ibr_alloc = create_allocate_node(
        dsc,
        metadata,
        fresh,
        Buffering::None,
        NodeName(format!(
            "allocate_lds{}_{}",
            recorded.0,
            dst_storage.spelling()
        )),
        dsc_idx,
        alloc,
    )?;
    ibr_alloc.indirect = index_hbm.indirect;
    ibr_alloc.related_indirect = index_hbm.related_indirect;
    let allocate = tree.new_allocate(alloc, ibr_alloc);
    tree.set_mem_org_allocation(index_lds, dst_storage, alloc);

    let transfer = tree.new_transfer(create_transfer_node(
        Via {
            loc: DataLocation {
                unit,
                storage: src_storage,
            },
            lds: Some(index_lds),
        },
        Via {
            loc: DataLocation {
                unit,
                storage: dst_storage,
            },
            lds: Some(index_lds),
        },
        &[],
        NodeName(format!(
            "transfer_lds{}_src:{}_dst:{}",
            recorded.0,
            src_storage.spelling(),
            dst_storage.spelling()
        )),
    ));
    tree.add_alloc_user(index_hbm.alloc, transfer);
    tree.add_alloc_user(alloc, transfer);

    let spelling = unit.spelling();
    let position = index_lds.0;
    let send = tree.new_sync(create_sync_node(
        SyncUnits::new(unit, []),
        NodeName(format!(
            "sync_send_{spelling}_to_{spelling}_paged_index_{position}"
        )),
        SyncDirection::Send,
        SyncStrength::Hard,
    ));
    let receive = tree.new_sync(create_sync_node(
        SyncUnits::new(unit, []),
        NodeName(format!(
            "sync_receive_{spelling}_from_{spelling}_paged_index_{position}"
        )),
        SyncDirection::Receive,
        SyncStrength::Hard,
    ));
    tree.add_sync_other_end(send, receive);
    tree.add_sync_other_end(receive, send);

    // The four new nodes go before the new ibr/chunk loop, as its siblings.
    for node in [allocate, transfer, send, receive] {
        tree.add_child_node(node, InsertionPoint::Before(new_chunk_loop.0));
    }
    Some(())
}

/// Replaces: e227_convertTransferDirectToIndirect
///
/// TURNS A DIRECT TENSOR TRANSFER INTO AN INDIRECT ONE: wraps it in a fresh chunk/1page loop over the
/// index stick dim and points its source — or its one destination — at the L3 unit's IBR.
///
/// ⛔ [`None`] IS *"Expect a tensor transfer."*, *"Expect the transfer owner loop to be a chunk or
/// SuperChunk loop."*, both *"Expect one entry"* refusals and the root-node parent walk.
/// ⚠️ TRAP: `TENSOR_TO_TENSOR` IS READ HERE AS "both ends name a labelled DS", which is the fact the
/// reference derives that transfer type from.
pub fn convert_transfer_direct_to_indirect<T: L3TreeSurgery + ?Sized>(
    tree: &mut T,
    core: &CoreWindowDims,
    transfer: NodeId,
    one_page: OnePageStage,
    super_chunk: Option<SuperChunkStage>,
    index_lds: LdsIdx,
    index_stick_dim: PrimaryDim,
    direction: IbrDirection,
) -> Option<()> {
    let mut node = tree.transfer(transfer);
    if node.src.data.my_lds_idx.is_none() || node.dsts.first().data.my_lds_idx.is_none() {
        return None;
    }
    if direction == IbrDirection::Out && (node.dsts.len() != 1 || node.dst_indirect.is_some()) {
        return None;
    }
    let den = tree.loop_den(tree.owner_loop(transfer)?);
    if den != DATA_STAGE_CHUNK && Some(den) != super_chunk.map(SuperChunkStage::index) {
        return None;
    }

    // Create a new chunk/1page or SuperChunk/1page loop node.
    let suffix = match direction {
        IbrDirection::In => "_to_lx",
        IbrDirection::Out => "_to_hbm",
    };
    let new_loop = tree.new_loop(create_loop_node(
        core,
        index_stick_dim,
        &[],
        den,
        one_page.index(),
        NodeName(format!(
            "loop_chunk_1page_ds{}_ds{}_{}{}",
            den.0,
            one_page.index().0,
            index_stick_dim.spelling(),
            suffix
        )),
    ));
    tree.parent(transfer)?;
    tree.add_child_node(new_loop.0, InsertionPoint::Before(transfer));
    tree.move_node(transfer, InsertionPoint::LastIn(new_loop.0));

    let via = Via {
        loc: DataLocation {
            unit: direction.unit(),
            storage: direction.dst_storage(),
        },
        lds: Some(index_lds),
    };
    match direction {
        IbrDirection::In => node.src_indirect = Some(via),
        IbrDirection::Out => node.dst_indirect = Some(via),
    }
    tree.set_transfer(transfer, node);
    Some(())
}

/// Replaces: e228_buildCoordinateFromAllocation
///
/// BUILDS ONE SCHEDULE NODE'S COORDINATE FROM A REFERENCE ALLOCATION'S: per dim the two share, copies
/// the reference's folds, distributes the enclosing loops the reference did not have over the element
/// arrangement levels, and adds the whole list back front-first as spatial, temporal and elem-arr folds.
///
/// ⛔ [`None`] IS the distribution's refusals, a zero-extent denominator and the `int` arithmetic
/// beside them. ⚠️ TRAP: `foldParams.insert(iter, ..)` RETURNS THE INSERTED ELEMENT, so successive
/// inserts land at the SAME index and the distributed loops end up REVERSED, which the walk from the
/// back rebuilds. ⛔ TRAP: `int scale = allocLds.scale_.at(dimIdx)` TRUNCATES a `double`, so the
/// non-broadcast test is `scale_ >= 1` and a fractional scale gathers NO loops.
pub fn build_coordinate_from_allocation<'a, D, E>(
    node: Node<'a>,
    node_id: NodeId,
    dsc: &D,
    loops: &OwnerLoops<'a>,
    reference: ReferenceAllocation<'_>,
    env: &mut E,
    loop_params: &mut E::LoopParams,
    coordinate: &mut Coordinate,
) -> Option<()>
where
    D: Dsc + ?Sized,
    E: AllocCoordinateSeam + ?Sized,
{
    if coordinate.fold_constructed() {
        return Some(());
    }

    // Find the enclosing loop chain. Collect the associated dimensions.
    let (chain, related) = get_enclosing_loops_and_related_dims(node, dsc, loops);
    for (dim, fm) in reference.coordinate.iter() {
        if !related.iter().any(|entry| entry.dim == dim) || coordinate.covers(dim) {
            continue;
        }
        let mut fold_params: Vec<Fold> = Vec::new();
        gather_fold_params(fm, &mut fold_params);

        let mut related_loops: Vec<LoopAndDim<'_>> = Vec::new();
        let non_broadcast = match reference.labeled_ds.scale(dim) {
            None => true,
            Some(scale) => matches!(scale, Scale::Sized(value) if value >= 1.0),
        };
        if non_broadcast {
            for owner in &chain {
                let pad = if coordinate.padding(dim) == PadType::NoPad {
                    AccessPad::NoPad
                } else {
                    AccessPad::Padded(env.stage_padding(owner.den)?)
                };
                find_and_store_loop_with_dim(
                    PrimaryDimAndKind {
                        dim,
                        kind: MetaDimKind::Unpadded,
                    },
                    owner,
                    pad,
                    &mut related_loops,
                );
            }
        }

        let spatial_ends = i64::from(fm.spatial_folds()) - 1;
        let mut temporal_ends = spatial_ends + i64::from(fm.temporal_folds());
        let loop_count = u32::try_from(related_loops.len()).ok()?;
        if fm.temporal_folds() < loop_count {
            // The node has more enclosing loops than the reference allocation; distribute the extra
            // ones over the element arrangement levels. Scan order is innermost outwards.
            let temporal_diff = loop_count - fm.temporal_folds();
            let to_distribute = related_loops.get(..usize::try_from(temporal_diff).ok()?)?;
            let cut = usize::try_from(temporal_ends + 1).unwrap_or(0);
            let bare = PrimaryDimAndKind {
                dim,
                kind: MetaDimKind::Unpadded,
            };
            let pad = coordinate.padding(dim);
            // The distributor reads no input label and every output one is overwritten below, so the
            // labels do not have to survive the crossing.
            let elem_arr: Vec<FoldParamInfo> = fold_params
                .get(cut..)?
                .iter()
                .rev()
                .map(|fold| FoldParamInfo {
                    alpha: Alpha(fold.alpha.0),
                    beta: Beta(fold.beta.0),
                    cardinality: Cardinality(u64::from(fold.cardinality.0)),
                    label: None,
                })
                .collect();
            let distributed = env.distribute(
                &ElemArrDistribution {
                    dim: bare,
                    fold_owner: node_id,
                    target_lds: reference.lds,
                    ref_pad: pad,
                    target_pad: pad,
                    components: RefComponents {
                        size: SenComponent::L3lu,
                        prop: SenComponent::L3lu,
                    },
                    loops_to_distribute: to_distribute.to_vec(),
                    elem_arr,
                },
                loop_params,
            );

            // Distribution may drop element arrangement levels, so the old ones go and the new ones
            // arrive innermost-last: the result is INNERMOST FIRST (`l3.cpp:4695`, and the same
            // comment sits over `ddc.cpp:8531` and `:8913`), which is why the walk runs backwards
            // and the label is the source index rather than the destination one.
            fold_params.truncate(cut);
            for (index, fold) in distributed.iter().enumerate().rev() {
                fold_params.push(Fold {
                    cardinality: FoldCardinality(u32::try_from(fold.cardinality.0).ok()?),
                    label: FoldLabel(format!("elem_arr_{index}")),
                    alpha: FoldCoeff(fold.alpha.0),
                    beta: FoldCoeff(fold.beta.0),
                });
            }
            for entry in to_distribute {
                let params = env.distributed(loop_params, entry.loop_node, dim)?;
                let cardinality = match env.parametric_iter_count(entry.loop_node) {
                    Some(count) => count,
                    None => {
                        // The spatial fold includes a level for corelets, so the datastage values are
                        // read per corelet.
                        let num = env.comp_view(entry.loop_node.num, entry.dim.dim)?;
                        let den = env.comp_view(entry.loop_node.den, entry.dim.dim)?;
                        if den.0 == 0 {
                            return None;
                        }
                        FoldCardinality(u32::try_from(num.0 / den.0).ok()?)
                    }
                };
                fold_params.insert(
                    cut,
                    Fold {
                        cardinality,
                        label: FoldLabel(format!(
                            "{} {}",
                            entry.loop_node.name.0,
                            dim.spelling()
                        )),
                        alpha: FoldCoeff(params.alpha.0),
                        beta: FoldCoeff(params.beta.0),
                    },
                );
            }
            temporal_ends += i64::from(temporal_diff);
        }

        // Construct the folds, outermost position last.
        for index in (0..fold_params.len()).rev() {
            let fold = fold_params.get(index)?;
            let position = i64::try_from(index).ok()?;
            let (category, label) = if position > temporal_ends {
                (
                    CoordinateCategory::ElemArr,
                    FoldLabel(format!("elem_arr_{}", fold_params.len() - 1 - index)),
                )
            } else if position > spatial_ends {
                (CoordinateCategory::Temporal, fold.label.clone())
            } else {
                (CoordinateCategory::Spatial, fold.label.clone())
            };
            coordinate.add_fold_front(dim, category, fold.cardinality, label, fold.alpha, fold.beta);
        }
    }
    coordinate.complete_fold_construction();
    Some(())
}

#[cfg(test)]
mod tests_e221_e228 {
    // ⭐ TESTS FOR ENTRIES 221-228. One schedule tree stub serves all five seams.
    use super::*;

    use core::num::NonZeroU32;

    use crate::arch::Dd2;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims;
    use crate::schedule::ddc::fold::{ConstIdx, DistributedLoop};
    use crate::schedule::ddc::transformation_util as util;
    use crate::schedule::ddc::transformation_util::{
        LoopCondComposite, ScheduleSurgery, construct_loop_node,
    };
    use crate::schedule::dsc2::{
        AllocLayout as AddressLayout, AllocPlacement, FoldPosition, LayoutDims, MaxDimSize,
        NumBuffers, StartAddress,
    };
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DscList, LabeledDsList, NamedDims,
        PlacedAllocation, PrimaryDsInfo, StageDims, WkSliceCount,
    };

    fn core(index: u32) -> Core {
        Core::checked(index).expect("this arch has the core the test names")
    }

    fn slices(count: u32) -> WkSliceCount {
        WkSliceCount::new(NonZeroU32::new(count).expect("a work-slice count"))
    }

    fn corelets(count: u32) -> CoreletsUsed {
        CoreletsUsed::new(NonZeroU32::new(count).expect("a corelet count"))
    }

    fn filled(extents: &[(PrimaryDim, i64)]) -> FilledDims {
        let mut dims = StageDims::default();
        for &(dim, extent) in extents {
            dims.extents.insert(dim, Extent(extent));
            dims.padding.insert(dim, DimPadding::default());
        }
        FilledDims::of(dims).expect("a stage that states a dim")
    }

    fn stage(name: &str, extents: &[(PrimaryDim, i64)]) -> DataStage {
        let dims = filled(extents);
        DataStage {
            ss: NamedDims {
                name: StageName(name.to_owned()),
                dims: dims.clone(),
            },
            el: NamedDims {
                name: StageName(name.to_owned()),
                dims,
            },
        }
    }

    /// A DSC labelling ONE tensor RECORDED at `recorded` — the only entry, so it sits at position 0.
    fn dsc(recorded: LdsIdx, core_extents: &[(PrimaryDim, i64)], pinning: Pinning) -> DesignSpaceConfig {
        let (first, _) = *core_extents.first().expect("a tensor with a dim in it");
        let layout = LayoutDims::new(
            first,
            core_extents[1..].iter().map(|&(dim, _)| dim).collect(),
        );
        let halved: Vec<(PrimaryDim, i64)> = core_extents
            .iter()
            .map(|&(dim, extent)| (dim, extent / 2))
            .collect();
        DesignSpaceConfig {
            corelets_used: corelets(2),
            corelets_used_dsc2: Some(corelets(2)),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::from([(
                DsType::Input,
                PrimaryDsInfo {
                    layout: layout.clone(),
                    stick: StickDims(Vec::new()),
                },
            )]),
            core_ids_used: CoreIdsUsed::new(core(0), vec![core(1)]),
            // Keyed BOTH by the position the entry sits at and by the index it records, because the
            // reference's own readers are split over which one they hand the layout map.
            layout_dims: BTreeMap::from([(LdsIdx(0), layout.clone()), (recorded, layout)]),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(
                    DsType::Input,
                    core_extents
                        .iter()
                        .map(|&(dim, _)| (dim, Scale::Sized(1.0)))
                        .collect(),
                    recorded,
                    pinning,
                ),
                vec![],
            ),
            data_stages: DataStages::new(stage("0", core_extents), stage("1", &halved)),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        }
    }

    /// The core stage's window dims, which is all the loop mints read of a DSC.
    fn window_dims(dims: &[PrimaryDim]) -> CoreWindowDims {
        #[derive(Default)]
        struct Dims(BTreeSet<PrimaryDim>);

        impl WindowExtents for Dims {
            fn window_dims(&self) -> BTreeSet<PrimaryDim> {
                self.0.clone()
            }
        }

        let stages = util::DataStages(
            [(
                CoreWindowDims::CORE,
                util::DataStage {
                    ss: util::StageDims {
                        name: StageName::default(),
                        dims: Dims(dims.iter().copied().collect()),
                    },
                    el: util::StageDims::default(),
                },
            )]
            .into_iter()
            .collect(),
        );
        CoreWindowDims::of(&stages).expect("the core data stage is stated")
    }

    /// A DSC whose data stages also state an IBR and a one-page stage.
    fn staged(extents: &[(PrimaryDim, i64)]) -> DataStages {
        let mut stages = DataStages::new(stage("0", extents), stage("1", extents));
        stages.set(DatastageId(2), stage("2", extents));
        stages.set(DatastageId(3), stage("3", extents));
        stages
    }

    /// One `MemOrg` — what entries 221 and 224 put to a labelled DS's organisations.
    #[derive(Debug, Default)]
    struct Org {
        indirection: Option<IndirectAlloc>,
        zero_padded: bool,
        padding: PaddingForm,
    }

    impl MemOrg for Org {
        fn hbm_pinned(&self) -> bool {
            false
        }

        fn lx_buffering(&self) -> Option<Buffering> {
            None
        }

        fn lx_start_address(&self, _at: &AddressCoord) -> Option<ByteAddress> {
            None
        }

        fn lx_buffer_offset(&self, _core: Core, _corelet: Corelet) -> Option<BufferOffset> {
            None
        }

        fn hbm_indirection(&self) -> Option<IndirectAlloc> {
            self.indirection
        }

        fn hbm_allocation(&self) -> Option<NodeName> {
            None
        }

        fn hbm_layout_dims(&self) -> Option<LayoutDims> {
            None
        }

        fn lx_zero_padded(&self) -> Option<bool> {
            Some(self.zero_padded)
        }

        fn lx_padding(&self) -> Option<PaddingForm> {
            Some(self.padding.clone())
        }

        fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim> {
            BTreeSet::new()
        }

        fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent> {
            BTreeMap::new()
        }

        fn hbm_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }

        fn lx_alloc_users(&self) -> Option<Vec<NodeId>> {
            None
        }
    }

    /// The kinds of node these entries mint and relink.
    #[derive(Debug, Clone)]
    enum Kind {
        Loop(LoopNode),
        Transfer(TransferNode),
        Sync(SyncNode),
        Allocate(L3AllocateNode),
        Block,
    }

    #[derive(Debug, Clone)]
    struct Entry {
        name: NodeName,
        parent: Option<NodeId>,
        children: Vec<NodeId>,
        kind: Kind,
    }

    /// A SCHEDULE TREE ENTRIES 225-227 REWRITE, plus everything they record beside it.
    #[derive(Debug, Default)]
    struct Tree {
        nodes: BTreeMap<NodeId, Entry>,
        next: u32,
        next_alloc: u32,
        alloc_nodes: BTreeMap<AllocId, NodeId>,
        alloc_users: Vec<(AllocId, NodeId)>,
        mem_orgs: Vec<(LdsIdx, SenComponent, AllocId)>,
    }

    impl Tree {
        fn add(&mut self, name: &str, kind: Kind, parent: Option<NodeId>) -> NodeId {
            let id = NodeId(self.next);
            self.next += 1;
            self.nodes.insert(
                id,
                Entry {
                    name: NodeName(name.to_owned()),
                    parent,
                    children: Vec::new(),
                    kind,
                },
            );
            if let Some(parent) = parent {
                self.nodes
                    .get_mut(&parent)
                    .expect("parent exists")
                    .children
                    .push(id);
            }
            id
        }

        fn loop_over(
            &mut self,
            num: DatastageId,
            den: DatastageId,
            dim: PrimaryDim,
            parent: NodeId,
        ) -> LoopId {
            let node = construct_loop_node(
                num,
                den,
                LoopDims::new(
                    PrimaryDimAndKind {
                        dim,
                        kind: MetaDimKind::Unpadded,
                    },
                    Vec::new(),
                ),
            );
            let name = node.name.0.clone();
            LoopId(self.add(&name, Kind::Loop(node), Some(parent)))
        }

        fn unlink(&mut self, node: NodeId) {
            let parent = self
                .nodes
                .get_mut(&node)
                .expect("node exists")
                .parent
                .take();
            if let Some(parent) = parent {
                self.nodes
                    .get_mut(&parent)
                    .expect("parent exists")
                    .children
                    .retain(|child| *child != node);
            }
        }

        fn link(&mut self, node: NodeId, at: InsertionPoint) {
            let (parent, index) = match at {
                InsertionPoint::Before(sibling) | InsertionPoint::After(sibling) => {
                    let parent = self.nodes[&sibling].parent.expect("sibling has a parent");
                    let position = self.nodes[&parent]
                        .children
                        .iter()
                        .position(|child| *child == sibling)
                        .expect("sibling among its parent's children");
                    let after = matches!(at, InsertionPoint::After(_));
                    (parent, position + usize::from(after))
                }
                InsertionPoint::FirstIn(parent) => (parent, 0),
                InsertionPoint::LastIn(parent) => (parent, self.nodes[&parent].children.len()),
            };
            self.nodes
                .get_mut(&parent)
                .expect("parent exists")
                .children
                .insert(index, node);
            self.nodes.get_mut(&node).expect("node exists").parent = Some(parent);
        }

        fn children(&self, node: NodeId) -> Vec<NodeId> {
            self.nodes[&node].children.clone()
        }

        fn names(&self, nodes: &[NodeId]) -> Vec<String> {
            nodes
                .iter()
                .map(|node| self.nodes[node].name.0.clone())
                .collect()
        }

        fn allocate_node(&self, alloc: AllocId) -> &L3AllocateNode {
            match &self.nodes[&self.alloc_nodes[&alloc]].kind {
                Kind::Allocate(node) => node,
                other => panic!("not an allocate: {other:?}"),
            }
        }

        fn sync_node(&self, node: NodeId) -> &SyncNode {
            match &self.nodes[&node].kind {
                Kind::Sync(sync) => sync,
                other => panic!("not a sync: {other:?}"),
            }
        }

        fn minted_loop(&self, loop_node: LoopId) -> &LoopNode {
            match &self.nodes[&loop_node.0].kind {
                Kind::Loop(node) => node,
                other => panic!("not a loop: {other:?}"),
            }
        }
    }

    impl ScheduleSurgery for Tree {
        fn node_name(&self, node: NodeId) -> NodeName {
            self.nodes[&node].name.clone()
        }

        fn set_node_name(&mut self, node: NodeId, name: NodeName) {
            self.nodes.get_mut(&node).expect("node exists").name = name;
        }

        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.nodes[&node].parent
        }

        fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
            let mut current = self.nodes[&node].parent;
            while let Some(candidate) = current {
                if matches!(self.nodes[&candidate].kind, Kind::Loop(_)) {
                    return Some(LoopId(candidate));
                }
                current = self.nodes[&candidate].parent;
            }
            None
        }

        fn transfer(&self, node: NodeId) -> TransferNode {
            match &self.nodes[&node].kind {
                Kind::Transfer(transfer) => transfer.clone(),
                other => panic!("not a transfer: {other:?}"),
            }
        }

        fn loop_num(&self, loop_node: LoopId) -> DatastageId {
            self.minted_loop(loop_node).num
        }

        fn loop_den(&self, loop_node: LoopId) -> DatastageId {
            self.minted_loop(loop_node).den
        }

        fn loop_dims(&self, loop_node: LoopId) -> LoopDims {
            self.minted_loop(loop_node).dims.clone()
        }

        fn is_parametric(&self, _loop_node: LoopId) -> bool {
            false
        }

        fn new_loop(&mut self, node: LoopNode) -> LoopId {
            let name = node.name.0.clone();
            LoopId(self.add(&name, Kind::Loop(node), None))
        }

        fn new_block(&mut self, name: NodeName) -> NodeId {
            self.add(&name.0, Kind::Block, None)
        }

        fn add_child_node(&mut self, node: NodeId, at: InsertionPoint) {
            self.link(node, at);
        }

        fn move_node(&mut self, node: NodeId, at: InsertionPoint) {
            self.unlink(node);
            self.link(node, at);
        }

        fn conditions_under(&self, _root: NodeId) -> Vec<NodeId> {
            Vec::new()
        }

        fn loop_cond(&self, _condition: NodeId) -> LoopCondComposite {
            LoopCondComposite::default()
        }

        fn set_loop_cond(&mut self, _condition: NodeId, _cond: LoopCondComposite) {}
    }

    impl LoopBands for Tree {
        fn set_loop_den(&mut self, loop_node: LoopId, den: DatastageId) {
            if let Kind::Loop(node) =
                &mut self.nodes.get_mut(&loop_node.0).expect("loop exists").kind
            {
                node.den = den;
            }
        }

        fn set_loop_dims(&mut self, loop_node: LoopId, dims: LoopDims) {
            if let Kind::Loop(node) =
                &mut self.nodes.get_mut(&loop_node.0).expect("loop exists").kind
            {
                node.dims = dims;
            }
        }

        fn move_children(&mut self, from: NodeId, to: NodeId) {
            let children =
                core::mem::take(&mut self.nodes.get_mut(&from).expect("node exists").children);
            for child in children {
                self.nodes.get_mut(&child).expect("child exists").parent = Some(to);
                self.nodes
                    .get_mut(&to)
                    .expect("node exists")
                    .children
                    .push(child);
            }
        }

        fn insert_perfectly_nested(&mut self, base: LoopId, nested: LoopId) {
            self.move_children(base.0, nested.0);
            self.link(nested.0, InsertionPoint::LastIn(base.0));
        }

        fn adjust_condition_for_split_loop(
            &mut self,
            _condition: NodeId,
            _orig: LoopId,
            _new_loops: &[LoopId],
        ) {
        }
    }

    impl L3TreeSurgery for Tree {
        fn delete_child_node(&mut self, node: NodeId) {
            self.unlink(node);
            self.nodes.remove(&node);
        }

        fn fresh_alloc(&mut self) -> AllocId {
            let alloc = AllocId(self.next_alloc);
            self.next_alloc += 1;
            alloc
        }

        fn new_allocate(&mut self, alloc: AllocId, node: L3AllocateNode) -> NodeId {
            let name = node.name.0.clone();
            let id = self.add(&name, Kind::Allocate(node), None);
            self.alloc_nodes.insert(alloc, id);
            id
        }

        fn new_transfer(&mut self, node: TransferNode) -> NodeId {
            let name = node.name.0.clone();
            self.add(&name, Kind::Transfer(node), None)
        }

        fn new_sync(&mut self, node: SyncNode) -> NodeId {
            let name = node.name.0.clone();
            self.add(&name, Kind::Sync(node), None)
        }

        fn add_sync_other_end(&mut self, sync: NodeId, other: NodeId) {
            let name = self.node_name(other);
            if let Kind::Sync(node) = &mut self.nodes.get_mut(&sync).expect("sync exists").kind {
                node.other_ends.push(name);
            }
        }

        fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
            self.alloc_users.push((alloc, user));
        }

        fn set_mem_org_allocation(&mut self, lds: LdsIdx, storage: SenComponent, alloc: AllocId) {
            self.mem_orgs.push((lds, storage, alloc));
        }

        fn set_transfer(&mut self, node: NodeId, transfer: TransferNode) {
            self.nodes.get_mut(&node).expect("node exists").kind = Kind::Transfer(transfer);
        }
    }

    /// e221 — a padded HBM-to-LX transfer gains one work-slice and one chunk fold at each end, and the
    /// constant source and the load unit that a padded transfer must name.
    #[test]
    fn a_zero_padded_transfer_gains_its_work_slice_and_chunk_fold_axes() {
        let mut config = dsc(LdsIdx(0), &[(PrimaryDim::X, 8)], Pinning::default());
        config.full_padding.insert(
            PrimaryDim::X,
            DimPadding {
                sizes: PadSizes::Sized {
                    front: PadElems(1),
                    back: PadElems(3),
                },
                window_dim: Some(PrimaryDim::Ki),
                ..DimPadding::default()
            },
        );
        let sdsc = SuperDsc::new(
            DscList::new(config, Vec::new()),
            BTreeMap::from([(PrimaryDim::X, slices(2))]),
            BTreeMap::new(),
            BTreeMap::new(),
        );

        /// The one DSC's organisations and the one transfer its LX allocation feeds.
        struct Env {
            orgs: Vec<Org>,
            transfers: BTreeMap<NodeId, TransferNode>,
        }

        impl DscTransfers for Env {
            type Org = Org;

            fn mem_orgs(&self, _dsc: DscIdx) -> Vec<&Self::Org> {
                self.orgs.iter().collect()
            }

            fn lx_alloc_transfer_users(&self, _dsc: DscIdx, lds: LdsIdx) -> Vec<NodeId> {
                assert_eq!(lds, LdsIdx(0), "the POSITION the entry sits at");
                self.transfers.keys().copied().collect()
            }

            fn transfer(&self, _dsc: DscIdx, node: NodeId) -> Option<TransferNode> {
                self.transfers.get(&node).cloned()
            }

            fn set_transfer(&mut self, _dsc: DscIdx, node: NodeId, transfer: TransferNode) {
                self.transfers.insert(node, transfer);
            }
        }

        let mut padding = PaddingForm::default();
        padding.set_padding(PrimaryDim::X, PadType::PaddedWZeroPad);
        let hbm_to_lx = create_transfer_node(
            Via {
                loc: DataLocation {
                    unit: SenComponent::NoComponent,
                    storage: SenComponent::Hbm,
                },
                lds: Some(LdsIdx(0)),
            },
            Via {
                loc: DataLocation {
                    unit: SenComponent::NoComponent,
                    storage: SenComponent::Lx,
                },
                lds: Some(LdsIdx(0)),
            },
            &[],
            NodeName("transfer".to_owned()),
        );
        let mut env = Env {
            orgs: vec![Org {
                zero_padded: true,
                padding,
                ..Org::default()
            }],
            transfers: BTreeMap::from([(NodeId(0), hbm_to_lx)]),
        };

        fill_transfer_zero_padding_info(&sdsc, &mut env).expect("a stated zero padding");
        let filled = &env.transfers[&NodeId(0)];

        // The core stage states 8 and the chunk stage 4, so the work slice steps by 8 and the two
        // chunks by 4, and the back pad loses one step of each.
        assert_eq!(
            filled.padding.pad_front(PrimaryDim::X),
            Some(ZeroPadFolds {
                work_slice: PadFold {
                    cardinality: FoldCardinality(2),
                    alpha: FoldCoeff(-8),
                    beta: FoldCoeff(1),
                },
                chunk: PadFold {
                    cardinality: FoldCardinality(2),
                    alpha: FoldCoeff(-4),
                    beta: FoldCoeff(0),
                },
            })
        );
        assert_eq!(
            filled.padding.pad_back(PrimaryDim::X),
            Some(ZeroPadFolds {
                work_slice: PadFold {
                    cardinality: FoldCardinality(2),
                    alpha: FoldCoeff(8),
                    beta: FoldCoeff(-9),
                },
                chunk: PadFold {
                    cardinality: FoldCardinality(2),
                    alpha: FoldCoeff(4),
                    beta: FoldCoeff(0),
                },
            })
        );
        assert_eq!(filled.src.unit, SenComponent::Constant);
        assert_eq!(filled.src.storage, SenComponent::Hbm);
        assert_eq!(filled.dsts.first().unit, SenComponent::L3lu);
    }

    /// e222 — the two execution phases place the one LX buffer at two addresses, which land in its
    /// fold space as a spread, and its buffer offset copies out to every corelet.
    #[test]
    fn every_new_lx_allocation_is_placed_once_per_execution_phase() {
        let config = dsc(LdsIdx(0), &[(PrimaryDim::X, 8)], Pinning::default());
        let metadata = BTreeMap::from([(
            DscIdx(0),
            DscMetadata {
                new_allocations: BTreeMap::from([(
                    SenComponent::Lx,
                    L3Allocation {
                        lds_idx_and_alloc_node: BTreeMap::from([(LdsIdx(0), AllocId(0))]),
                    },
                )]),
            },
        )]);
        let mut allocs: v1::AllocArena = BTreeMap::from([(
            AllocId(0),
            AllocateNode {
                name: NodeName("allocate_lds0".to_owned()),
                component: SenComponent::Lx,
                lds: Some(LdsIdx(0)),
                const_idx: None,
                temp_storage_for_compute: None,
                layout: AddressLayout::new((PrimaryDim::X, MaxDimSize::Unset), Vec::new()),
                start_address: StartAddress::default(),
                placement: AllocPlacement {
                    num_buffers: NumBuffers::Double,
                    ..AllocPlacement::default()
                },
                gap_stick_spread: BTreeMap::new(),
                alloc_users: Vec::new(),
            },
        )]);

        /// Two phases, each handing out its own address, and the names it was asked to forget.
        #[derive(Default)]
        struct Trackers {
            removed: Vec<v1::StorageName>,
            restored: bool,
            placed: Vec<(L3TrackerSite, ExPhase, Bytes)>,
        }

        impl ExPhaseTrackers for Trackers {
            fn ex_phases(&self) -> Vec<ExPhase> {
                vec![ExPhase(0), ExPhase(1)]
            }

            fn capacity(&self, _at: L3TrackerSite) -> Bytes {
                Bytes(4096)
            }

            fn backup(&mut self, _at: L3TrackerSite) {}

            fn restore_all(&mut self) {
                self.restored = true;
            }

            fn remove(&mut self, _at: L3TrackerSite, name: &v1::StorageName) {
                self.removed.push(name.clone());
            }

            fn check_and_add(
                &mut self,
                at: L3TrackerSite,
                phase: ExPhase,
                _name: &v1::StorageName,
                size: Bytes,
            ) -> Option<v1::Placed> {
                self.placed.push((at, phase, size));
                Some(v1::Placed::At(Bytes(u64::from(phase.0) * 1024)))
            }
        }

        /// A design space whose LX buffer is 64 bytes a copy, over a two-axis fold space.
        struct Placement;

        impl L3Placement for Placement {
            fn buffer_capacity_even_sticks(
                &self,
                _alloc: AllocId,
                _lds: LdsIdx,
                _corelet: Corelet,
                _row: Row,
            ) -> Bytes {
                Bytes(64)
            }

            fn address_fold_depth(&self) -> usize {
                2
            }

            fn address_fold_coords(&self) -> usize {
                2
            }
        }

        impl v1::StorageNames for Placement {
            fn lds_name(&self, lds: LdsIdx) -> v1::StorageName {
                v1::StorageName(format!("lds{}", lds.0))
            }

            fn constant_name(&self, constant: ConstIdx) -> v1::StorageName {
                v1::StorageName(format!("const{}", constant.0))
            }
        }

        let mut trackers = Trackers::default();
        let placed = alloc_all_mem(
            &config,
            &metadata,
            DscIdx(0),
            &mut allocs,
            &mut trackers,
            &Placement,
            v1::Commit::IfValid,
        );
        assert_eq!(placed, Some(true));
        assert!(!trackers.restored);
        // Once per LX core, each of which forgets the name before it re-places it.
        assert_eq!(trackers.removed, vec![v1::StorageName("lds0".to_owned()); 2]);

        // The double buffer asks for two copies of a 64-byte capacity, at both LX cores.
        assert_eq!(
            trackers.placed.iter().map(|(_, _, size)| *size).collect::<Vec<_>>(),
            vec![Bytes(128); 4]
        );
        let node = &allocs[&AllocId(0)];
        assert_eq!(
            node.start_address.spread(core(1), Corelet::at::<0>()),
            [Bytes(0), Bytes(1024)]
        );
        // Every axis maps: the addresses differ per phase, and both cores were placed separately.
        assert_eq!(
            node.start_address.func_type(FoldPosition::Core),
            Some(AddressFold::Map)
        );
        assert_eq!(
            node.placement.buffer_offset,
            BTreeMap::from([
                (
                    core(0),
                    BTreeMap::from([(Corelet::at::<0>(), Bytes(64)), (Corelet::at::<1>(), Bytes(64))])
                ),
                (
                    core(1),
                    BTreeMap::from([(Corelet::at::<0>(), Bytes(64)), (Corelet::at::<1>(), Bytes(64))])
                ),
            ])
        );
    }

    /// e223 — before SEN1P5 the estimate is the smallest work-slice product over the HBM-pinned
    /// tensors, and a super-DSC pinning nothing in HBM leaves the seed untouched.
    #[test]
    fn the_hmi_request_estimate_is_the_smallest_work_slice_product() {
        struct Trees;

        impl ScheduleTrees for Trees {
            fn allocations(&self, _dsc: DscIdx) -> Vec<PlacedAllocation> {
                Vec::new()
            }
        }

        let pinned = Pinning {
            mem_org: BTreeMap::from([(SenComponent::Hbm, true)]),
            ..Pinning::default()
        };
        let sdsc = SuperDsc::new(
            DscList::new(
                dsc(LdsIdx(0), &[(PrimaryDim::X, 8), (PrimaryDim::Y, 8)], pinned),
                Vec::new(),
            ),
            BTreeMap::from([(PrimaryDim::X, slices(2)), (PrimaryDim::Y, slices(3))]),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        assert_eq!(
            get_hbm_lds_transfer_hmi_request_estimate::<Dd2, _>(&sdsc, &Trees),
            Some(HmiRequests(6))
        );

        // And a super-DSC pinning nothing in HBM narrows nothing.
        let bare = SuperDsc::new(
            DscList::new(
                dsc(
                    LdsIdx(0),
                    &[(PrimaryDim::X, 8), (PrimaryDim::Y, 8)],
                    Pinning::default(),
                ),
                Vec::new(),
            ),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        assert_eq!(
            get_hbm_lds_transfer_hmi_request_estimate::<Dd2, _>(&bare, &Trees),
            Some(HmiRequests::UNBOUNDED)
        );
    }

    /// e224 — only the tensor indirected through as a VALUE is paged, and it answers with the index it
    /// records rather than the position it sits at.
    #[test]
    fn every_paged_labelled_ds_answers_with_the_index_it_records() {
        let value = Org {
            indirection: Some(IndirectAlloc::ValueTensor),
            ..Org::default()
        };
        let index = Org {
            indirection: Some(IndirectAlloc::IndexTensor(IndexTensor::Index)),
            ..Org::default()
        };
        let plain = Org::default();
        assert_eq!(
            get_all_paged_lds_indices(&[(LdsIdx(4), &value), (LdsIdx(5), &index), (LdsIdx(6), &plain)]),
            vec![LdsIdx(4)]
        );
    }

    /// e225 — the paged dim's chunk loop is replaced by a core/ibr loop over an ibr/chunk loop, which
    /// takes over its body, and the original leaves the tree and the chunk-loop set together.
    #[test]
    fn a_paged_dim_chunk_loop_becomes_a_core_then_ibr_loop_nest() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let original = tree.loop_over(
            CoreWindowDims::CORE,
            DATA_STAGE_CHUNK,
            PrimaryDim::X,
            root,
        );
        let body = tree.add("body", Kind::Block, Some(original.0));
        let mut chunk_loops = BTreeSet::from([original]);
        let ibr = staged(&[(PrimaryDim::X, 8)])
            .ibr(DatastageId(2))
            .expect("a stated IBR stage");

        let minted = create_paged_dim_chunk_loops(
            &mut tree,
            &window_dims(&[]),
            LxBufferType::Double,
            ibr,
            &[PrimaryDim::X],
            &mut chunk_loops,
        )
        .expect("a nest over the paged dim");

        let inner = minted[&PrimaryDim::X];
        assert_eq!(tree.node_name(inner.0).0, "loop_ibr_chunk_ds2_ds1_x");
        let outer = tree.parent(inner.0).expect("the ibr/chunk loop has a parent");
        assert_eq!(tree.node_name(outer).0, "loop_core_ibr_ds0_ds2_x");
        assert_eq!(tree.parent(outer), Some(root));
        // The original's body hangs off the inner loop, and the original itself is gone.
        assert_eq!(tree.children(inner.0), vec![body]);
        assert_eq!(tree.children(root), vec![outer]);
        assert!(chunk_loops.is_empty());
        assert!(!tree.nodes.contains_key(&original.0));
    }

    /// e226 — the index tensor gains an IBR allocation carrying the HBM allocation's indirection, a
    /// transfer into it and a self-sync pair, all four before the new chunk loop.
    #[test]
    fn staging_a_paged_index_adds_an_allocate_a_transfer_and_a_sync_pair() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let chunk = tree.loop_over(DatastageId(2), DATA_STAGE_CHUNK, PrimaryDim::X, root);
        let config = dsc(LdsIdx(7), &[(PrimaryDim::X, 8)], Pinning::default());
        let mut metadata = BTreeMap::new();

        create_store_index_tensor_to_ibr(
            &mut tree,
            &config,
            &mut metadata,
            DscIdx(0),
            LdsIdx(0),
            IndexHbmAllocation {
                alloc: AllocId(9),
                indirect: Some(IndirectAlloc::IndexTensor(IndexTensor::Index)),
                related_indirect: Some(AllocId(4)),
            },
            chunk,
            IbrDirection::In,
        )
        .expect("a staged index tensor");

        // The allocate and transfer names spell the RECORDED index, the syncs the POSITION.
        let children = tree.children(root);
        assert_eq!(
            tree.names(&children),
            vec![
                "allocate_lds7_l3luibr".to_owned(),
                "transfer_lds7_src:hbm_dst:l3luibr".to_owned(),
                "sync_send_l3lu_to_l3lu_paged_index_0".to_owned(),
                "sync_receive_l3lu_from_l3lu_paged_index_0".to_owned(),
                tree.node_name(chunk.0).0,
            ]
        );
        let ibr = tree.allocate_node(AllocId(0));
        assert_eq!(
            ibr.indirect,
            Some(IndirectAlloc::IndexTensor(IndexTensor::Index))
        );
        assert_eq!(ibr.related_indirect, Some(AllocId(4)));
        assert_eq!(
            tree.mem_orgs,
            vec![(LdsIdx(0), SenComponent::L3luibr, AllocId(0))]
        );
        // Both allocations gain the one transfer as a user, the paged tensor's first.
        assert_eq!(
            tree.alloc_users,
            vec![(AllocId(9), children[1]), (AllocId(0), children[1])]
        );
        assert_eq!(
            tree.sync_node(children[2]).other_ends,
            vec![tree.node_name(children[3])]
        );
        assert_eq!(
            tree.sync_node(children[3]).other_ends,
            vec![tree.node_name(children[2])]
        );
    }

    /// e227 — the direct transfer gains a chunk/1page loop over the index stick dim and reads its
    /// source through the load unit's IBR.
    #[test]
    fn a_direct_transfer_becomes_an_indirect_one_under_a_new_one_page_loop() {
        let mut tree = Tree::default();
        let root = tree.add("root", Kind::Block, None);
        let chunk = tree.loop_over(
            CoreWindowDims::CORE,
            DATA_STAGE_CHUNK,
            PrimaryDim::X,
            root,
        );
        let direct = create_transfer_node(
            Via {
                loc: DataLocation {
                    unit: SenComponent::L3lu,
                    storage: SenComponent::Hbm,
                },
                lds: Some(LdsIdx(2)),
            },
            Via {
                loc: DataLocation {
                    unit: SenComponent::L3lu,
                    storage: SenComponent::Lx,
                },
                lds: Some(LdsIdx(2)),
            },
            &[],
            NodeName("transfer".to_owned()),
        );
        let transfer = tree.add("transfer", Kind::Transfer(direct), Some(chunk.0));
        let one_page = staged(&[(PrimaryDim::X, 8)])
            .one_page(DatastageId(3))
            .expect("a stated one-page stage");

        convert_transfer_direct_to_indirect(
            &mut tree,
            &window_dims(&[]),
            transfer,
            one_page,
            None,
            LdsIdx(5),
            PrimaryDim::Ki,
            IbrDirection::In,
        )
        .expect("an indirect transfer");

        let minted = tree.parent(transfer).expect("the transfer has a parent");
        assert_eq!(tree.node_name(minted).0, "loop_chunk_1page_ds1_ds3_ki_to_lx");
        assert_eq!(tree.parent(minted), Some(chunk.0));
        assert_eq!(tree.children(minted), vec![transfer]);
        let node = tree.transfer(transfer);
        assert_eq!(
            node.src_indirect,
            Some(Via {
                loc: DataLocation {
                    unit: SenComponent::L3lu,
                    storage: SenComponent::L3luibr,
                },
                lds: Some(LdsIdx(5)),
            })
        );
        assert_eq!(node.dst_indirect, None);
    }

    /// e228 — the node's own enclosing loop is distributed over the reference's element arrangement,
    /// and the whole list is added back as one spatial, one temporal and one elem-arr fold.
    #[test]
    fn a_coordinate_distributes_the_loops_its_reference_allocation_did_not_have() {
        struct OneDim(LayoutDims);

        impl Dsc for OneDim {
            fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
                self.0.clone()
            }
        }

        /// The distribution pass, which drops the reference's element arrangement for one of its own.
        struct Env;

        impl TemporalLoopDistribution for Env {
            type LoopParams = Vec<(NodeName, DistributedLoop)>;

            fn related_loops<'l>(
                &self,
                _dim: PrimaryDimAndKind,
                chain: &[LoopAndDim<'l>],
                _pad: PadType,
            ) -> Vec<LoopAndDim<'l>> {
                chain.to_vec()
            }

            fn distribute(
                &self,
                request: &ElemArrDistribution<'_>,
                loop_params: &mut Self::LoopParams,
            ) -> Vec<FoldParamInfo> {
                assert_eq!(
                    (request.dim.dim, request.target_lds, request.ref_pad),
                    (PrimaryDim::X, LdsIdx(3), PadType::NoPad)
                );
                assert_eq!(request.loops_to_distribute.len(), 1);
                assert_eq!(request.elem_arr.len(), 1);
                for entry in &request.loops_to_distribute {
                    loop_params.push((
                        entry.loop_node.name.clone(),
                        DistributedLoop {
                            alpha: Alpha(5),
                            beta: Beta(0),
                        },
                    ));
                }
                // One replacement level, whose own label the caller overwrites.
                vec![FoldParamInfo {
                    alpha: Alpha(1),
                    beta: Beta(0),
                    cardinality: Cardinality(2),
                    label: None,
                }]
            }

            fn distributed(
                &self,
                loop_params: &Self::LoopParams,
                loop_node: &LoopNode,
                _dim: PrimaryDim,
            ) -> Option<DistributedLoop> {
                loop_params
                    .iter()
                    .find(|(name, _)| *name == loop_node.name)
                    .map(|(_, params)| *params)
            }
        }

        impl AllocCoordinateSeam for Env {
            fn parametric_iter_count(&self, _loop_node: &LoopNode) -> Option<FoldCardinality> {
                None
            }

            fn comp_view(&self, stage: DatastageId, _dim: PrimaryDim) -> Option<Extent> {
                Some(Extent(if stage == DatastageId(1) { 12 } else { 4 }))
            }

            fn stage_padding(
                &self,
                _stage: DatastageId,
            ) -> Option<&BTreeMap<PrimaryDim, DimPadding>> {
                None
            }
        }

        let mut reference = Coordinate::default();
        reference.add_fold_front(
            PrimaryDim::X,
            CoordinateCategory::ElemArr,
            FoldCardinality(4),
            FoldLabel("dropped".to_owned()),
            FoldCoeff(1),
            FoldCoeff(0),
        );
        reference.add_fold_front(
            PrimaryDim::X,
            CoordinateCategory::Spatial,
            FoldCardinality(2),
            FoldLabel("core".to_owned()),
            FoldCoeff(8),
            FoldCoeff(0),
        );
        let labeled_ds = LabeledDs::new(
            DsType::Input,
            vec![(PrimaryDim::X, Scale::Sized(1.0))],
            LdsIdx(3),
            Pinning::default(),
        );
        let alloc = AllocateNode {
            name: NodeName("allocate_lds3".to_owned()),
            component: SenComponent::Lx,
            lds: Some(LdsIdx(3)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AddressLayout::new((PrimaryDim::X, MaxDimSize::Unset), Vec::new()),
            start_address: StartAddress::default(),
            placement: AllocPlacement::default(),
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        };
        let inner = construct_loop_node(
            DatastageId(1),
            DatastageId(2),
            LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::X,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        );
        let root = construct_loop_node(
            DatastageId(0),
            DatastageId(1),
            LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::Y,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        );
        let loops = OwnerLoops::of(vec![&inner, &root]).expect("a loop chain with a root");
        let mut coordinate = Coordinate::default();
        let mut loop_params = Vec::new();

        build_coordinate_from_allocation(
            Node::Allocate(&alloc),
            NodeId(0),
            &OneDim(LayoutDims::new(PrimaryDim::X, Vec::new())),
            &loops,
            ReferenceAllocation {
                coordinate: &reference,
                lds: LdsIdx(3),
                labeled_ds: &labeled_ds,
            },
            &mut Env,
            &mut loop_params,
            &mut coordinate,
        )
        .expect("a coordinate built from its reference");

        let folds = coordinate
            .fold_dim(PrimaryDim::X)
            .expect("the dim the reference shares");
        assert_eq!(
            folds
                .folds()
                .map(|fold| (fold.label.0.clone(), fold.cardinality, fold.alpha))
                .collect::<Vec<_>>(),
            vec![
                ("core".to_owned(), FoldCardinality(2), FoldCoeff(8)),
                // The enclosing loop the reference did not have, at 12 / 4 iterations.
                ("loop_ds1_ds2_x x".to_owned(), FoldCardinality(3), FoldCoeff(5)),
                ("elem_arr_0".to_owned(), FoldCardinality(2), FoldCoeff(1)),
            ]
        );
        assert_eq!(
            (
                folds.spatial_folds(),
                folds.temporal_folds(),
                folds.elem_arr_folds()
            ),
            (1, 1, 1)
        );
        assert!(coordinate.fold_constructed());
    }

    #[test]
    fn a_fractional_scale_is_broadcast_and_gathers_no_loops_to_distribute() {
        struct OneDim;

        impl Dsc for OneDim {
            fn layout_dims(&self, _lds: LdsIdx) -> LayoutDims {
                LayoutDims::new(PrimaryDim::X, Vec::new())
            }
        }

        /// A distributor the broadcast dim must never reach.
        struct NoDistribution;

        impl TemporalLoopDistribution for NoDistribution {
            type LoopParams = ();

            fn related_loops<'l>(
                &self,
                _dim: PrimaryDimAndKind,
                _chain: &[LoopAndDim<'l>],
                _pad: PadType,
            ) -> Vec<LoopAndDim<'l>> {
                unreachable!("a broadcast dim relates no loops")
            }

            fn distribute(
                &self,
                _request: &ElemArrDistribution<'_>,
                _loop_params: &mut Self::LoopParams,
            ) -> Vec<FoldParamInfo> {
                unreachable!("a broadcast dim distributes nothing")
            }

            fn distributed(
                &self,
                _loop_params: &Self::LoopParams,
                _loop_node: &LoopNode,
                _dim: PrimaryDim,
            ) -> Option<DistributedLoop> {
                None
            }
        }

        impl AllocCoordinateSeam for NoDistribution {
            fn parametric_iter_count(&self, _loop_node: &LoopNode) -> Option<FoldCardinality> {
                None
            }

            fn comp_view(&self, _stage: DatastageId, _dim: PrimaryDim) -> Option<Extent> {
                None
            }

            fn stage_padding(
                &self,
                _stage: DatastageId,
            ) -> Option<&BTreeMap<PrimaryDim, DimPadding>> {
                None
            }
        }

        let mut reference = Coordinate::default();
        reference.add_fold_front(
            PrimaryDim::X,
            CoordinateCategory::ElemArr,
            FoldCardinality(4),
            FoldLabel("kept".to_owned()),
            FoldCoeff(1),
            FoldCoeff(0),
        );
        reference.add_fold_front(
            PrimaryDim::X,
            CoordinateCategory::Spatial,
            FoldCardinality(2),
            FoldLabel("core".to_owned()),
            FoldCoeff(8),
            FoldCoeff(0),
        );
        // `int scale = 0.5` is 0, so the reference takes the broadcast arm.
        let labeled_ds = LabeledDs::new(
            DsType::Input,
            vec![(PrimaryDim::X, Scale::Sized(0.5))],
            LdsIdx(3),
            Pinning::default(),
        );
        let alloc = AllocateNode {
            name: NodeName("allocate_lds3".to_owned()),
            component: SenComponent::Lx,
            lds: Some(LdsIdx(3)),
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AddressLayout::new((PrimaryDim::X, MaxDimSize::Unset), Vec::new()),
            start_address: StartAddress::default(),
            placement: AllocPlacement::default(),
            gap_stick_spread: BTreeMap::new(),
            alloc_users: Vec::new(),
        };
        let inner = construct_loop_node(
            DatastageId(1),
            DatastageId(2),
            LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::X,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        );
        let root = construct_loop_node(
            DatastageId(0),
            DatastageId(1),
            LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::Y,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        );
        let loops = OwnerLoops::of(vec![&inner, &root]).expect("a loop chain with a root");
        let mut coordinate = Coordinate::default();

        build_coordinate_from_allocation(
            Node::Allocate(&alloc),
            NodeId(0),
            &OneDim,
            &loops,
            ReferenceAllocation {
                coordinate: &reference,
                lds: LdsIdx(3),
                labeled_ds: &labeled_ds,
            },
            &mut NoDistribution,
            &mut (),
            &mut coordinate,
        )
        .expect("a coordinate built from its reference");

        let folds = coordinate
            .fold_dim(PrimaryDim::X)
            .expect("the dim the reference shares");
        assert_eq!(
            folds
                .folds()
                .map(|fold| (fold.label.0.clone(), fold.cardinality, fold.alpha))
                .collect::<Vec<_>>(),
            vec![
                ("core".to_owned(), FoldCardinality(2), FoldCoeff(8)),
                ("elem_arr_0".to_owned(), FoldCardinality(4), FoldCoeff(1)),
            ]
        );
    }
}

/// THE TWO CORELET COUNTS ONE DSC CARRIES — `numCoreletsUsed_` and `numCoreletsUsed_DSC2_`
/// (`dsc/designSpaceConfig.h:74`), which entry 229 uses for DIFFERENT things: the first halves the
/// data stage, the second scales the work-slice cardinality and divides the temporal alphas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreletCounts {
    /// `numCoreletsUsed_`.
    pub used: CoreletsUsed,
    /// `numCoreletsUsed_DSC2_`.
    pub dsc2: CoreletsUsed,
}

/// WHAT A CORELET SLICE READS OFF THE ALLOCATE NODE — `allocNode` narrowed to its four facts, with
/// `allocateCoordinates_` travelling separately because entry 229 needs it EXCLUSIVELY borrowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlicedAllocation {
    /// The node itself — `foldOwnerNode` for the distributor.
    pub node: NodeId,
    /// `component_`, which is both `sizeRefComp` and `propRefComp` here.
    pub component: SenComponent,
    /// `ldsIdx_`.
    pub lds: LdsIdx,
    /// `labeledDs_.at(ldsIdx_)` through [`LabeledDs::scale`], whose [`None`] is the reference's
    /// `dimIdx < 0` fallback to scale 1.
    pub dim_scale: Option<Scale>,
}

/// WHAT A CORELET SLICE READS AND WRITES ON ONE DATA-STAGE HALF — `DataStructDims` narrowed to the
/// three questions entry 229 asks of `ss_` and `el_` alike.
pub trait CoreletSliceDims {
    /// `coreletSplit_.begin()->first`, [`None`] where this half splits nothing.
    fn first_corelet_split_dim(&self) -> Option<PrimaryDim>;

    /// `primaryDimToValHandler_st(dim) /= n`, AND every `coreletSplit_.at(dim)` share alike — the
    /// chunk-half-and-half slicing strategy, on this half.
    fn divide_for_corelets(&mut self, dim: PrimaryDim, corelets: CoreletsUsed);

    /// `dataStageDimToVal_compView_st(dim, comp)`, [`None`] where this half has no such extent.
    fn comp_view_extent(&self, dim: PrimaryDim, comp: SenComponent) -> Option<Extent>;
}

/// THE dsc2 TREE A CORELET SLICE WORKS AGAINST — `dataStageParam_` plus the two `dsc/dsc2.cpp`
/// helpers entry 229 reaches through, both OUTSIDE this campaign's file list
/// (`crustify-ddc/OUTSIDE-DEPS.tsv`, "dsc2 tree utilities"), so they are seams here and not ports.
pub trait CoreletSliceSeam: TemporalLoopDistribution {
    /// The extents payload each of this DSC's data stages carries.
    type Dims: Clone + CoreletSliceDims;

    /// `currDsc->dataStageParam_`.
    fn stages(&self) -> &DataStages<Self::Dims>;

    /// The same, writable — entry 229 mints a denominator stage in it and erases it again.
    fn stages_mut(&mut self) -> &mut DataStages<Self::Dims>;

    /// `dsc2::loopRelevantForDim(currDsc, dimAndKind, loopNode, accessPadType)`
    /// (`dsc/dsc2.cpp:6550`) — asked of ONE loop, which is why
    /// [`TemporalLoopDistribution::related_loops`] cannot answer it: `collectRelatedLoops` pushes
    /// one entry per matching *(loop, loop dim)* pair carrying the LOOP'S own dim, where entry 229
    /// pushes each relevant loop ONCE carrying the corelet-split dim.
    fn loop_relevant(&self, dim: PrimaryDimAndKind, loop_node: &LoopNode, pad: PadType) -> bool;

    /// [`lx_below_block_node`] then `getMutableOwnerLoop()` walked upward, INNERMOST FIRST and
    /// WITHOUT the outermost loop — *"Exclude the root loop"*, which breaks on a loop with NO OWNER
    /// LOOP and so is NOT [`parent_loop_nodes`] (that one breaks on no parent BLOCK). [`None`] is
    /// `DT_CHECK_MSG(lxBelowBlockNode, "Expect a valid lx_below block node.")`.
    fn lx_below_chunk_loops(&self) -> Option<Vec<&LoopNode>>;
}

/// Replaces: e229_sliceCoordinateForCorelet
///
/// REBUILDS the corelet-split dim's folds on the allocation's coordinate: halves the chunk stage,
/// mints a corelet-slice loop over it and redistributes the element arrangements onto that loop.
/// ⛔ TRAP: `int scale = allocLds.scale_.at(dimIdx)` TRUNCATES a `double`, so `scale < 0` catches
/// only [`Scale::UnitStick`]/[`Scale::StickDim`] — a fractional broadcast SURVIVES the guard.
/// ⛔ TRAP: THE WALK EXCLUDES THE OUTERMOST LOOP, and the distributor's dim is the BARE
/// `{coreletSplitDim, Unpadded}` while `relatedLoops` carries the possibly-`Padded` kind.
pub fn slice_coordinate_for_corelet<T: CoreletSliceSeam + ?Sized>(
    sdsc: &SuperDsc,
    corelets: CoreletCounts,
    alloc: SlicedAllocation,
    seam: &mut T,
    loop_params: &mut T::LoopParams,
    coord: &mut Coordinate,
) -> Option<()> {
    if !corelets.used.splits() || alloc.component != SenComponent::Lx {
        return Some(());
    }
    let chunk = seam.stages().0.get(&DATA_STAGE_CHUNK)?.clone();
    let Some(dim) = chunk.ss.dims.first_corelet_split_dim() else {
        return Some(());
    };
    let Some(folds) = coord.fold_dim(dim) else {
        return Some(());
    };
    // Check if the coreletSplitDim is a broadcast dimension.
    if matches!(alloc.dim_scale, Some(Scale::UnitStick | Scale::StickDim)) {
        return Some(());
    }

    let is_lx_pinned = folds.temporal_folds() == 0;
    let spatial_ends = i64::from(folds.spatial_folds()) - 1;
    let temporal_ends = spatial_ends + i64::from(folds.temporal_folds());
    let mut fold_params = Vec::new();
    gather_fold_params(folds, &mut fold_params);
    // `temporalFoldEnds + 1` — where the levels INSIDE the coordinate's own folds start.
    let inside = usize::try_from(temporal_ends + 1).unwrap_or(0);
    // The distributor reads no input label and entry 229 overwrites every output one below, so the
    // concatenated [`FoldLabel`]s do not have to survive the crossing.
    let elem_arr = fold_params
        .get(inside..)
        .unwrap_or_default()
        .iter()
        .rev()
        .map(|fold| FoldParamInfo {
            alpha: Alpha(fold.alpha.0),
            beta: Beta(fold.beta.0),
            cardinality: Cardinality(u64::from(fold.cardinality.0)),
            label: None,
        })
        .collect();

    // Construct an artificial loop to simulate corelet level slicing: chunk half-and-half.
    let den = construct_datastage(seam.stages_mut(), &chunk);
    let den_stage = seam.stages_mut().0.get_mut(&den)?;
    den_stage.ss.dims.divide_for_corelets(dim, corelets.used);
    den_stage.el.dims.divide_for_corelets(dim, corelets.used);
    let bare = PrimaryDimAndKind {
        dim,
        kind: MetaDimKind::Unpadded,
    };
    let new_loop = construct_loop_node(DATA_STAGE_CHUNK, den, LoopDims::new(bare, Vec::new()));

    let alloc_padding = coord.padding(dim);
    let split_dim = PrimaryDimAndKind {
        dim,
        kind: if alloc_padding == PadType::NoPad {
            MetaDimKind::Unpadded
        } else {
            MetaDimKind::Padded
        },
    };
    let mut related = vec![LoopAndDim {
        loop_node: &new_loop,
        dim: split_dim,
        distribution: LoopDistribution::AboveChunk,
    }];
    if is_lx_pinned {
        // LX-pinned allocation: every chunk loop joins the distribution and its fold becomes an
        // outer element arrangement. Inner (position 0) to outer (end of list).
        for loop_node in seam.lx_below_chunk_loops()? {
            if seam.loop_relevant(split_dim, loop_node, alloc_padding) {
                related.push(LoopAndDim {
                    loop_node,
                    dim: split_dim,
                    distribution: LoopDistribution::AboveChunk,
                });
            }
        }
    }

    // The reference makes the allocation a temporary child of the new loop so the distributor can
    // compute custom data stages, and restores the tree straight after; that net-zero positioning
    // is the seam's own precondition and not a fact about the coordinate.
    let elem_arr_after = seam.distribute(
        &ElemArrDistribution {
            dim: bare,
            fold_owner: alloc.node,
            target_lds: alloc.lds,
            ref_pad: alloc_padding,
            target_pad: alloc_padding,
            components: RefComponents {
                size: alloc.component,
                prop: alloc.component,
            },
            loops_to_distribute: related.clone(),
            elem_arr,
        },
        loop_params,
    );

    // Update wkSlice fold.
    let distributed = seam.distributed(loop_params, &new_loop, dim)?;
    let wk_slices = sdsc.num_wk_slices_per_dim.get(&dim)?;
    *fold_params.get_mut(FoldPosition::Core as usize)? = Fold {
        cardinality: FoldCardinality(corelets.dsc2.get().saturating_mul(wk_slices.get())),
        label: FoldLabel("workslice_fold".to_owned()),
        alpha: FoldCoeff(distributed.alpha.0),
        beta: FoldCoeff(distributed.beta.0),
    };

    // Update temporal folds: adjust alpha for the chunk-half-and-half slicing strategy.
    for (position, fold) in fold_params.iter_mut().enumerate() {
        let position = position as i64;
        if position > spatial_ends && position <= temporal_ends {
            fold.alpha = FoldCoeff(fold.alpha.0 / i64::from(corelets.dsc2.get()));
        }
    }

    fold_params.truncate(inside);
    if is_lx_pinned {
        // Outer to inner; position 0 is the corelet fold rewritten above.
        for related_loop in related.get(1..).unwrap_or_default().iter().rev() {
            let loop_node = related_loop.loop_node;
            let num = seam
                .stages()
                .0
                .get(&loop_node.num)?
                .ss
                .dims
                .comp_view_extent(dim, alloc.component)?;
            let den = seam
                .stages()
                .0
                .get(&loop_node.den)?
                .ss
                .dims
                .comp_view_extent(dim, alloc.component)?;
            let iterations = num.0.checked_div(den.0)?;
            let params = seam.distributed(loop_params, loop_node, dim)?;
            fold_params.push(Fold {
                cardinality: FoldCardinality(u32::try_from(iterations).unwrap_or(u32::MAX)),
                label: FoldLabel("elem_arr_chunk_level".to_owned()),
                alpha: FoldCoeff(params.alpha.0),
                beta: FoldCoeff(params.beta.0),
            });
        }
    }

    // The innermost element arrangement is at the beginning of the distributor's answer.
    for (level, params) in elem_arr_after.iter().enumerate().rev() {
        fold_params.push(Fold {
            cardinality: FoldCardinality(u32::try_from(params.cardinality.0).unwrap_or(u32::MAX)),
            label: FoldLabel(format!("elem_arr_{level}")),
            alpha: FoldCoeff(params.alpha.0),
            beta: FoldCoeff(params.beta.0),
        });
    }

    coord.clear_fold_for_dim(dim);
    for (position, fold) in fold_params.iter().enumerate().rev() {
        let category = match position as i64 {
            position if position > temporal_ends => CoordinateCategory::ElemArr,
            position if position > spatial_ends => CoordinateCategory::Temporal,
            _ => CoordinateCategory::Spatial,
        };
        coord.add_fold_front(
            dim,
            category,
            fold.cardinality,
            fold.label.clone(),
            fold.alpha,
            fold.beta,
        );
    }

    // ⚠️ THE TEMPORARY STAGE'S ERASE IS DEFERRED TO HERE so the loop chain's shared borrows are
    // dead first. Past the reference's own erase point the only stage reads are of PRE-EXISTING
    // chunk loops' `numId_`/`denId_`, and `free_id()` cannot have handed one of those out.
    seam.stages_mut().0.remove(&den);
    Some(())
}

#[cfg(test)]
mod tests_e229 {
    use super::*;
    use crate::schedule::ddc::fold::DistributedLoop;
    use crate::schedule::dsc2::LayoutDims;
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, DataStage as L3DataStage, DataStages as L3DataStages, DscList, LabeledDsList,
        NamedDims, StageDims as L3StageDims,
    };
    use std::cell::RefCell;

    /// ONE DATA-STAGE HALF: the split dim's extent and its corelet shares.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Dims {
        extent: i64,
        split: Vec<i64>,
    }

    impl CoreletSliceDims for Dims {
        fn first_corelet_split_dim(&self) -> Option<PrimaryDim> {
            (!self.split.is_empty()).then_some(PrimaryDim::X)
        }

        fn divide_for_corelets(&mut self, _dim: PrimaryDim, corelets: CoreletsUsed) {
            let corelets = i64::from(corelets.get());
            self.extent /= corelets;
            for share in &mut self.split {
                *share /= corelets;
            }
        }

        fn comp_view_extent(&self, _dim: PrimaryDim, _comp: SenComponent) -> Option<Extent> {
            Some(Extent(self.extent))
        }
    }

    /// The dsc2 seams: every chunk loop is relevant, and the distributor stamps the `n`th loop it is
    /// handed with alpha `10 + n` and hands the element arrangements straight back.
    struct Seam {
        stages: DataStages<Dims>,
        chunk_loops: Vec<LoopNode>,
        /// The minted denominator half AS THE DISTRIBUTOR SAW IT — the halving's own witness, since
        /// the stage itself is erased before the call returns.
        sliced: RefCell<Option<Dims>>,
    }

    impl TemporalLoopDistribution for Seam {
        type LoopParams = Vec<(NodeName, DistributedLoop)>;

        fn related_loops<'l>(
            &self,
            _dim: PrimaryDimAndKind,
            chain: &[LoopAndDim<'l>],
            _pad: PadType,
        ) -> Vec<LoopAndDim<'l>> {
            chain.to_vec()
        }

        fn distribute(
            &self,
            request: &ElemArrDistribution<'_>,
            loop_params: &mut Self::LoopParams,
        ) -> Vec<FoldParamInfo> {
            *self.sliced.borrow_mut() = request
                .loops_to_distribute
                .first()
                .and_then(|entry| self.stages.0.get(&entry.loop_node.den))
                .map(|stage| stage.ss.dims.clone());
            for (nth, entry) in request.loops_to_distribute.iter().enumerate() {
                loop_params.push((
                    entry.loop_node.name.clone(),
                    DistributedLoop {
                        alpha: Alpha(10 + nth as i64),
                        beta: Beta(nth as i64),
                    },
                ));
            }
            request.elem_arr.clone()
        }

        fn distributed(
            &self,
            loop_params: &Self::LoopParams,
            loop_node: &LoopNode,
            _dim: PrimaryDim,
        ) -> Option<DistributedLoop> {
            loop_params
                .iter()
                .find(|(name, _)| *name == loop_node.name)
                .map(|(_, params)| *params)
        }
    }

    impl CoreletSliceSeam for Seam {
        type Dims = Dims;

        fn stages(&self) -> &DataStages<Dims> {
            &self.stages
        }

        fn stages_mut(&mut self) -> &mut DataStages<Dims> {
            &mut self.stages
        }

        fn loop_relevant(
            &self,
            _dim: PrimaryDimAndKind,
            _loop_node: &LoopNode,
            _pad: PadType,
        ) -> bool {
            true
        }

        fn lx_below_chunk_loops(&self) -> Option<Vec<&LoopNode>> {
            Some(self.chunk_loops.iter().collect())
        }
    }

    fn stage(extent: i64) -> DataStage<Dims> {
        let half = StageDims {
            name: StageName::default(),
            dims: Dims {
                extent,
                split: vec![extent / 2],
            },
        };
        DataStage {
            ss: half.clone(),
            el: half,
        }
    }

    fn loop_over(num: DatastageId, den: DatastageId) -> LoopNode {
        construct_loop_node(
            num,
            den,
            LoopDims::new(
                PrimaryDimAndKind {
                    dim: PrimaryDim::X,
                    kind: MetaDimKind::Unpadded,
                },
                Vec::new(),
            ),
        )
    }

    fn corelets(count: u32) -> CoreletsUsed {
        CoreletsUsed::new(NonZeroU32::new(count).expect("a positive corelet count"))
    }

    /// A super-DSC stating TWO work slices of `X` — only `numWkSlicesPerDim_` is read.
    fn a_super_dsc() -> SuperDsc {
        let l3_stage = || {
            let named = NamedDims {
                name: StageName::default(),
                dims: FilledDims::of(L3StageDims {
                    extents: BTreeMap::from([(PrimaryDim::X, Extent(8))]),
                    ..L3StageDims::default()
                })
                .expect("a stage stating at least one dim"),
            };
            L3DataStage {
                ss: named.clone(),
                el: named,
            }
        };
        let dsc = DesignSpaceConfig {
            corelets_used: corelets(2),
            corelets_used_dsc2: Some(corelets(2)),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(Core::checked(0).expect("core 0"), vec![]),
            layout_dims: BTreeMap::from([(LdsIdx(0), LayoutDims::new(PrimaryDim::X, vec![]))]),
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(DsType::Input, vec![], LdsIdx(0), Pinning::default()),
                vec![],
            ),
            data_stages: L3DataStages::new(l3_stage(), l3_stage()),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        };
        SuperDsc::new(
            DscList::new(dsc, vec![]),
            BTreeMap::from([(
                PrimaryDim::X,
                WkSliceCount::new(NonZeroU32::new(2).expect("two work slices")),
            )]),
            BTreeMap::new(),
            BTreeMap::new(),
        )
    }

    /// e229 — an LX-pinned corelet split halves the chunk stage, rewrites position 0 as the
    /// work-slice fold, keeps NOTHING inside it, and re-lays the chunk loop then the two element
    /// arrangements after it.
    #[test]
    fn a_corelet_slice_rebuilds_the_split_dims_folds_and_erases_its_temporary_stage() {
        let mut seam = Seam {
            stages: DataStages(BTreeMap::from([
                (DATA_STAGE_CHUNK, stage(8)),
                (DatastageId(2), stage(8)),
                (DatastageId(3), stage(4)),
            ])),
            chunk_loops: vec![loop_over(DatastageId(2), DatastageId(3))],
            sliced: RefCell::new(None),
        };
        // ONE spatial fold then two element arrangements; NO temporal fold, so the alloc is pinned.
        let mut coord = Coordinate::default();
        for (category, cardinality, label, alpha, beta) in [
            (CoordinateCategory::ElemArr, 7, "e2", 9, 2),
            (CoordinateCategory::ElemArr, 3, "e1", 5, 1),
            (CoordinateCategory::Spatial, 1, "s0", 1, 0),
        ] {
            coord.add_fold_front(
                PrimaryDim::X,
                category,
                FoldCardinality(cardinality),
                FoldLabel(label.to_owned()),
                FoldCoeff(alpha),
                FoldCoeff(beta),
            );
        }
        let mut loop_params = Vec::new();

        assert_eq!(
            slice_coordinate_for_corelet(
                &a_super_dsc(),
                CoreletCounts {
                    used: corelets(2),
                    dsc2: corelets(2),
                },
                SlicedAllocation {
                    node: NodeId(0),
                    component: SenComponent::Lx,
                    lds: LdsIdx(0),
                    dim_scale: Some(Scale::Sized(1.0)),
                },
                &mut seam,
                &mut loop_params,
                &mut coord,
            ),
            Some(())
        );

        let folds = coord
            .fold_dim(PrimaryDim::X)
            .expect("the dim stays covered");
        assert_eq!(
            folds
                .folds()
                .map(|fold| (
                    fold.cardinality.0,
                    fold.label.0.clone(),
                    fold.alpha.0,
                    fold.beta.0
                ))
                .collect::<Vec<_>>(),
            vec![
                // `numCoreletsUsed_DSC2_ * numWkSlicesPerDim_.at(X)`, with the MINTED loop's alpha.
                (4, "workslice_fold".to_owned(), 10, 0),
                // The chunk loop: 8 / 4 iterations, and the alpha the distributor gave IT.
                (2, "elem_arr_chunk_level".to_owned(), 11, 1),
                // The element arrangements, re-labelled innermost-LAST.
                (3, "elem_arr_1".to_owned(), 5, 1),
                (7, "elem_arr_0".to_owned(), 9, 2),
            ]
        );
        assert_eq!(
            (
                folds.spatial_folds(),
                folds.temporal_folds(),
                folds.elem_arr_folds()
            ),
            (1, 0, 3)
        );
        // The denominator half WAS halved, and its stage is gone again.
        assert_eq!(
            seam.sliced.into_inner(),
            Some(Dims {
                extent: 4,
                split: vec![2]
            })
        );
        assert_eq!(
            seam.stages.0.keys().copied().collect::<Vec<_>>(),
            vec![DATA_STAGE_CHUNK, DatastageId(2), DatastageId(3)]
        );
    }
}

/// Replaces: e283_addOrUpdateCoreletSplitInParams
///
/// STATES `coreletSplit_` on the stage: each corelet-split dim's extent cut into
/// `numCoreletsUsed_` equal shares, REPLACING whatever that dim held before.
///
/// ⛔ [`None`] IS *"Invalid corelet split."* — an extent the corelet count does not divide. A dim the
/// stage does not state is its `-1`, and is SKIPPED rather than refused.
pub fn add_or_update_corelet_split_in_params(
    params: &mut FilledDims,
    dsc: &DesignSpaceConfig,
) -> Option<()> {
    let corelets = dsc.corelets_used.get();
    for dim in corelet_split_dimensions(dsc) {
        let Some(stated) = params.dims().extent(dim) else {
            continue;
        };
        if stated.0 % i64::from(corelets) != 0 {
            return None;
        }
        let share = Extent(stated.0 / i64::from(corelets));
        let mut shares = Vec::new();
        for _ in 0..corelets {
            shares.push(share);
        }
        params.corelet_split_mut().insert(dim, shares);
    }
    Some(())
}

/// Replaces: e284_isOpFuncStridedWindow
///
/// WHETHER THE OP FUNC SLIDES A WINDOW BY A STRIDE — a conv2d, a pooling or a depthwise conv.
#[must_use]
pub fn is_op_func_strided_window(op_func: Option<OpFunc>) -> bool {
    is_op_func_conv2d(op_func)
        || op_func.is_some_and(|op| is_op_func_pooling(op) || is_op_func_depthwise_conv(op))
}

/// WHAT ENTRIES 285 AND 286 ASK OF EVERY DSC'S SCHEDULE — its OWN loop nesting, reached by DSC index,
/// which is the mechanism for walking one transfer's enclosing loops in each DSC of the group.
pub trait DscLoopStages {
    /// One DSC's nesting, however the caller holds it.
    type Stages: LoopStages + ?Sized;

    /// The nesting of the DSC at `dsc`, [`None`] for a DSC index that names no schedule.
    fn loop_stages(&self, dsc: DscIdx) -> Option<&Self::Stages>;
}

/// Every DSC index of the super-DSC, in `dscs_` order.
fn dsc_indices(sdsc: &SuperDsc) -> Vec<DscIdx> {
    (0u32..)
        .map(DscIdx)
        .zip(sdsc.dscs().iter())
        .map(|(at, _)| at)
        .collect()
}

/// `dscIndices` RESOLVED AGAINST `dscs_`, IN THE ORDER STATED — `DT_CHECK_MSG(!dscIndices.empty(),
/// "Expect valid DSCs.")` and both `.at()` throws.
fn dsc_group<'s>(sdsc: &'s SuperDsc, indices: &[DscIdx]) -> Option<DscGroup<'s>> {
    let (main, rest) = indices.split_first()?;
    let rest = rest
        .iter()
        .map(|at| sdsc.dscs().at(*at))
        .collect::<Option<Vec<_>>>()?;
    Some(DscGroup::new(sdsc.dscs().at(*main)?, rest))
}

/// ONE RECORDED TRANSFER OF A TENSOR — how often it repeats, the DSC its chunk is measured from and
/// the DSCs its multicast spans.
struct TransferRepeats {
    /// The DSC every per-chunk fact is read from: the DSC ITSELF where the tensor is core split,
    /// `dscMain` where every DSC transfers the same chunk.
    reference: DscIdx,
    /// `numRepeats`.
    repeats: u64,
    /// `dscIndices`, in the order the reference lists them.
    dscs: Vec<DscIdx>,
}

/// THE REPEAT CONTRIBUTIONS ONE TENSOR'S TRANSFER MAKES — the trip-count product over each DSC's
/// parent loops on the dims the tensor does NOT depend on, split so that repeats shared by both DSCs
/// multicast across them and the surplus multicasts on the deeper DSC alone.
///
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: the core-split branch multiplies a trip count for EVERY dim
/// OCCURRENCE over the parent loops, while the shared branch stores them in a `map<dim, tripCount>`
/// and so keeps only the LAST count a dim named on two parent loops states.
/// ⛔ [`None`] IS EVERY REFUSAL: `getLdsL3TransferNodes`', *"Expect valid transfer nodes."* — which
/// the shared branch does not even check before dereferencing `front()` — `getTripCount`'s, the
/// products wrapping, and *"Currently only support at most two DSCs."*
fn transfer_repeats<O, T, S>(
    sdsc: &SuperDsc,
    lds: LdsIdx,
    core_split: bool,
    src_storages: &[SenComponent],
    orgs: &O,
    trees: &T,
    nesting: &S,
) -> Option<Vec<TransferRepeats>>
where
    O: MemOrgs + ?Sized,
    T: TransferNodes + ?Sized,
    S: DscLoopStages + ?Sized,
{
    let indices = dsc_indices(sdsc);
    let mut per_dsc: Vec<u64> = Vec::new();
    for at in &indices {
        let dsc = sdsc.dscs().at(*at)?;
        let related = dsc.non_broadcast_lds_dim_set(lds)?;
        let transfers = lds_l3_transfer_nodes(
            sdsc,
            *at,
            lds,
            orgs.mem_org(*at, lds)?,
            trees,
            src_storages,
            &[SenComponent::Lx],
        )?;
        let node = transfers.first()?.node;
        let tree = nesting.loop_stages(*at)?;
        let mut occurrences: u64 = 1;
        let mut last_per_dim: BTreeMap<PrimaryDim, u64> = BTreeMap::new();
        for enclosing in parent_loop_nodes(tree, node) {
            for dim in tree.loop_dims(enclosing).iter() {
                if related.contains(&dim.dim) {
                    continue;
                }
                let count = trip_count(
                    &dsc.data_stages,
                    dim.dim,
                    tree.loop_num(enclosing),
                    tree.loop_den(enclosing),
                )?
                .get();
                occurrences = occurrences.checked_mul(count)?;
                last_per_dim.insert(dim.dim, count);
            }
        }
        per_dsc.push(if core_split {
            occurrences
        } else {
            let mut product: u64 = 1;
            for count in last_per_dim.values() {
                product = product.checked_mul(*count)?;
            }
            product
        });
    }
    if core_split {
        return Some(
            indices
                .iter()
                .zip(per_dsc)
                .map(|(at, repeats)| TransferRepeats {
                    reference: *at,
                    repeats,
                    dscs: vec![*at],
                })
                .collect(),
        );
    }
    let main = *indices.first()?;
    let first = *per_dsc.first()?;
    let second = per_dsc.get(1).copied();
    if second.is_none_or(|second| second == first) {
        return Some(vec![TransferRepeats {
            reference: main,
            repeats: first,
            dscs: indices,
        }]);
    }
    let second = second?;
    (per_dsc.len() == 2).then_some(())?;
    let deeper = if first < second { indices[1] } else { main };
    let shallower = if first < second { main } else { indices[1] };
    Some(vec![
        TransferRepeats {
            reference: main,
            repeats: first.min(second),
            dscs: vec![shallower, deeper],
        },
        TransferRepeats {
            reference: main,
            repeats: first.max(second) - first.min(second),
            dscs: vec![deeper],
        },
    ])
}

/// Replaces: e285_calculateBurstEfficiency
///
/// THE GROUP'S AVERAGE TRANSFER EFFICIENCY — every HBM-pinned or neighbour-fetched tensor's chunk cut
/// into as many 32-stick bursts as fit plus a remainder, each burst tallied once per stick volume, per
/// repeat and per work slice at its multicast degree, and the tally weighed by [`burst_efficiency`].
///
/// ⛔ `primaryDims` IS DEAD: it reaches only `getLabeledDsNumOfStickVolumesInCore`, which never reads
/// it (entry 043 dropped it for the same reason).
/// ⛔ [`None`] IS EVERY REFUSAL, the *"at most one core split dimension with two DSCs"* one included,
/// AND the `efficiency / 0` NaN an empty tally divides by.
#[must_use]
pub fn calculate_burst_efficiency<O, T, S>(
    sdsc: &SuperDsc,
    orgs: &O,
    trees: &T,
    nesting: &S,
) -> Option<BurstEfficiency>
where
    O: MemOrgs + ?Sized,
    T: TransferNodes + ?Sized,
    S: DscLoopStages + ?Sized,
{
    // `maxBurstSize` is the reference's own literal, and [`BurstSize`] bounds it by `l3BurstSize`.
    const MAX_BURST: u64 = 32;
    let main_idx = DscIdx(0);
    let main = sdsc.dscs().first();
    let core_split = core_split_dimensions(sdsc);
    (core_split.len() <= 1 && sdsc.dscs().iter().count() <= 2).then_some(())?;
    let mut chunked: Vec<LdsIdx> = Vec::new();
    for (at, entry) in main.labeled_ds.indexed() {
        let transferred =
            entry.pinning().hbm() || is_labeled_ds_lx_neighbor(sdsc, main_idx, entry)?;
        if transferred && !is_index_lds(orgs.mem_org(main_idx, at)?)? {
            chunked.push(at);
        }
    }
    let mut requests: BTreeMap<(BurstSize, MulticastCores), u64> = BTreeMap::new();
    for at in chunked {
        let lds_core_split = main
            .layout_dims
            .get(&at)?
            .iter()
            .any(|dim| core_split.contains(&dim));
        let recorded = transfer_repeats(
            sdsc,
            at,
            lds_core_split,
            &[SenComponent::Hbm, SenComponent::NoComponent],
            orgs,
            trees,
            nesting,
        )?;
        for transfer in recorded {
            let dsc = sdsc.dscs().at(transfer.reference)?;
            let org = orgs.mem_org(transfer.reference, at)?;
            let volume = labeled_ds_chunk_stick_volume(dsc, at, org)?;
            let volumes = labeled_ds_num_of_stick_volumes_in_core(dsc, at, volume)?.0;
            let group = dsc_group(sdsc, &transfer.dscs)?;
            let slices = u64::from(labeled_ds_num_of_wk_slices(sdsc, at, &group)?.get());
            let shares =
                MulticastCores::of(labeled_ds_wk_slice_multicast_degree(sdsc, at, &group)?)?;
            let per_volume = volumes.checked_mul(transfer.repeats)?.checked_mul(slices)?;
            let mut tally = |burst: BurstSize, count: u64| -> Option<()> {
                let key = (burst, shares);
                let total = requests
                    .get(&key)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(count)?;
                requests.insert(key, total);
                Some(())
            };
            let full = volume.get() / MAX_BURST;
            if full > 0 {
                let burst = BurstSize::new(u32::try_from(MAX_BURST).ok()?)?;
                tally(burst, full.checked_mul(per_volume)?)?;
            }
            let remainder = volume.get() % MAX_BURST;
            if remainder > 0 {
                let burst = BurstSize::new(u32::try_from(remainder).ok()?)?;
                tally(burst, per_volume)?;
            }
        }
    }
    let mut efficiency = 0.0;
    let mut total: u64 = 0;
    for ((burst, shares), count) in &requests {
        efficiency += burst_efficiency(*burst, *shares).0 * *count as f64;
        total = total.checked_add(*count)?;
    }
    (total > 0).then(|| BurstEfficiency(efficiency / total as f64))
}

/// HOW MUCH ARITHMETIC ONE TRANSFERRED BYTE FEEDS — `calculateFlopPerByte`'s `double`, which the
/// search compares against the system's own Flops/Byte.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FlopPerByte(pub f64);

/// Replaces: e286_calculateFlopPerByte
///
/// THE GROUP'S ARITHMETIC INTENSITY — the chunk's `primaryDims` extents multiplied, doubled for the
/// MAC's two operations and taken over every core of every DSC, over the LX chunk capacity of each
/// HBM-pinned tensor times its repeats and work slices.
///
/// ⛔ `isValidDimParam` IS `param > 0.0`, so an UNSTATED dim is skipped: the reference's `-1` and its
/// negatives short-circuit BEFORE any abort, which is what the unpadded probe below separates.
/// ⛔ [`None`] IS EVERY REFUSAL: *"Do not expect input neighbor fetch."*, the *"at most one core split
/// dimension"* one, *"Invalid total flops."*, *"Invalid total bytes."* and every seam's own.
#[must_use]
pub fn calculate_flop_per_byte<O, T, S>(
    sdsc: &SuperDsc,
    primary_dims: &[PrimaryDim],
    orgs: &O,
    trees: &T,
    nesting: &S,
) -> Option<FlopPerByte>
where
    O: MemOrgs + ?Sized,
    T: TransferNodes + ?Sized,
    S: DscLoopStages + ?Sized,
{
    let main_idx = DscIdx(0);
    let main = sdsc.dscs().first();
    let mut total_flops: i64 = 0;
    for at in dsc_indices(sdsc) {
        let dsc = sdsc.dscs().at(at)?;
        // The pad type of the FIRST labelled DS organised in LX, which the reference takes to hold
        // for every padded dim of the DSC.
        let padding = dsc
            .labeled_ds
            .indexed()
            .find_map(|(lds, _)| orgs.mem_org(at, lds).and_then(MemOrg::lx_padding))
            .unwrap_or_default();
        let chunk = dsc.data_stages.chunk().ss.dims.dims();
        let mut flops: i64 = 1;
        for dim in primary_dims {
            if chunk
                .scaled_extent(*dim, &PaddingForm::default(), None, false)
                .is_none()
            {
                continue;
            }
            let param = chunk.scaled_extent(*dim, &padding, None, false)?;
            if param.0 > 0 {
                flops = flops.checked_mul(param.0)?;
            }
        }
        // Each element is one multiply and one add, on every core of the DSC.
        flops = flops.checked_mul(2)?;
        flops = flops.checked_mul(i64::from(dsc.core_ids_used.count().0))?;
        total_flops = total_flops.checked_add(flops)?;
    }
    (total_flops > 0).then_some(())?;
    let core_split = core_split_dimensions(sdsc);
    (core_split.len() <= 1 && sdsc.dscs().iter().count() <= 2).then_some(())?;
    let mut total_bytes: i64 = 0;
    for (at, entry) in main.labeled_ds.indexed() {
        (!is_labeled_ds_lx_neighbor(sdsc, main_idx, entry)?).then_some(())?;
        if !entry.pinning().hbm() {
            continue;
        }
        let lds_core_split = main
            .layout_dims
            .get(&at)?
            .iter()
            .any(|dim| core_split.contains(&dim));
        // *"Expect LX in labeledDs memOrg_."* with *"Expect a valid allocate node."*; the node's
        // `component_ == LX` holds by construction of the seam.
        orgs.mem_org(main_idx, at)?.lx_padding()?;
        let recorded = transfer_repeats(
            sdsc,
            at,
            lds_core_split,
            &[SenComponent::Hbm],
            orgs,
            trees,
            nesting,
        )?;
        for transfer in recorded {
            let capacity = sdsc
                .dscs()
                .at(transfer.reference)?
                .lx_chunk_capacity
                .get(&at)?
                .0;
            let bytes = i64::try_from(capacity).ok()?;
            let group = dsc_group(sdsc, &transfer.dscs)?;
            let slices = i64::from(labeled_ds_num_of_wk_slices(sdsc, at, &group)?.get());
            let repeats = i64::try_from(transfer.repeats).ok()?;
            total_bytes =
                total_bytes.checked_add(bytes.checked_mul(repeats)?.checked_mul(slices)?)?;
        }
    }
    (total_bytes > 0).then(|| FlopPerByte(total_flops as f64 / total_bytes as f64))
}

/// Replaces: e287_getCrossCoreReductionGroupInfo
///
/// THE REDUCTION GROUPS OF A CROSS-CORE REDUCTION — one group per combination of the work slices on
/// the dims the op does NOT reduce, with every core placed in its group at the slice its REDUCED dims
/// name, both indices mixed-radix over those slice counts.
///
/// ⛔ [`None`] IS *"Expect cross-core reduction dataflow."*, [`op_reduced_dim_set`]'s aborts, the
/// `numWkSlicesPerDim_.at(dim)` throw, and a group index outside the `numGroups` the reference sized
/// the vector to — which its own `.at(group)` throws on.
#[must_use]
pub fn cross_core_reduction_group_info(
    sdsc: &SuperDsc,
    dsc: &DesignSpaceConfig,
) -> Option<Vec<CrossCoreReductionGroup>> {
    is_op_cross_core_reduction(sdsc, dsc)?.then_some(())?;
    let reduced = op_reduced_dim_set(dsc)?;
    let mut groups: usize = 1;
    for (dim, slices) in &sdsc.num_wk_slices_per_dim {
        if !reduced.contains(dim) {
            groups = groups.checked_mul(slices.get() as usize)?;
        }
    }
    let mut info = vec![CrossCoreReductionGroup::default(); groups];
    for (core, slice) in &sdsc.core_id_to_wk_slice {
        let mut group: u32 = 0;
        let mut group_cardinality: u32 = 1;
        let mut reduce: u32 = 0;
        let mut reduce_cardinality: u32 = 1;
        for (dim, dim_slice) in &slice.0 {
            let count = sdsc.num_wk_slices_per_dim.get(dim)?.get();
            let index = u32::try_from(dim_slice.0).ok()?;
            if reduced.contains(dim) {
                reduce = reduce.checked_add(index.checked_mul(reduce_cardinality)?)?;
                reduce_cardinality = reduce_cardinality.checked_mul(count)?;
            } else {
                group = group.checked_add(index.checked_mul(group_cardinality)?)?;
                group_cardinality = group_cardinality.checked_mul(count)?;
            }
        }
        info.get_mut(group as usize)?
            .add_core(*core, ReduceSlice(reduce));
    }
    Some(info)
}

/// WHAT ENTRIES 288 AND 289 ASK OF EVERY DSC'S SCHEDULE TREE — the per-DSC walk entries 015 and 211
/// perform, plus the three nodes a placement is stated relative to. All of it `dsc2::ScheduleNode`
/// MECHANISM for reaching operands rather than an L3 scheduling decision.
pub trait DscTrees {
    /// One DSC's tree, however the caller holds it.
    type Tree: NodeParents + LoopStages + ?Sized;

    /// `dscs_.at(dsc).scheduleTree_`, [`None`] for a DSC index the super-DSC does not have.
    fn tree(&self, dsc: DscIdx) -> Option<&Self::Tree>;

    /// `scheduleTree_.getHead()`, whose absence is `DT_CHECK_MSG(!dsc.scheduleTree_.empty(), "Expect
    /// a valid schedule tree.")`.
    fn root(&self, dsc: DscIdx) -> Option<NodeId>;

    /// `getLxBelowBlockNode(dsc.scheduleTree_)` THROUGH THE ID CARRIER — the same block
    /// [`lx_below_block_node`] finds in an owned tree, and *"Expect a valid lx_below block node."*
    /// when it is absent.
    fn lx_below_block(&self, dsc: DscIdx) -> Option<NodeId>;

    /// `labeledDs_.at(lds).memOrg_.at(storage).allocateNode_` BY IDENTITY — *"Expect .. in memOrg_."*
    /// and *"Expect allocate node."* are one [`None`].
    ///
    /// ⛔ NOT [`MemOrg::hbm_allocation`], which answers the node's NAME: an insertion point stated
    /// relative to an allocate node needs the node itself.
    fn allocation(&self, dsc: DscIdx, lds: LdsIdx, storage: SenComponent) -> Option<NodeId>;

    /// `transferNode->srcLdsAndLoopOffsets_.myLdsIdx_`, [`None`] where `isSrcLabeledDs()`
    /// (`dsc/dsc2.h:867`) is false — which is half of `getTransferType()`'s `TENSOR_TO_TENSOR`.
    fn transfer_src_lds(&self, dsc: DscIdx, node: NodeId) -> Option<LdsIdx>;

    /// `transferNode->isDstLabeledDs()` (`dsc/dsc2.h:868`) — the other half.
    fn transfer_dst_is_lds(&self, dsc: DscIdx, node: NodeId) -> bool;
}

/// WHAT ENTRIES 288 AND 289 DO TO THOSE TREES — the mint-and-place and the move, which ARE the effect
/// of both units.
///
/// ⭐ ONE CARRIER WITH [`DscTrees`], because `mySDsc` is one object: every read below is reborrowed
/// for the length of one question and no read is held across a write.
pub trait DscTreeSurgery: DscTrees {
    /// `new dsc2::SyncNode(..)` AND the `addChildNode` that links it, AS ONE STEP: a sync the tree
    /// does not hold has no position for the next one to chain from.
    fn insert_sync(&mut self, dsc: DscIdx, sync: SyncNode, at: InsertionPoint) -> NodeId;

    /// `parent->moveChildNode(&dsc, node, newParent, addBefore, sibling)` (`dsc/dsc2.cpp:2031`) —
    /// unlinked from its old parent first.
    fn move_node(&mut self, dsc: DscIdx, node: NodeId, at: InsertionPoint);
}

/// ONE DSC'S SYNC CHAIN — the `(surgery, dsc)` pair entry 212's sequence runs through when it is
/// reached by node id instead of from an owned block.
struct DscSyncs<'e, E: ?Sized> {
    env: &'e mut E,
    dsc: DscIdx,
}

impl<'e, E: ?Sized> DscSyncs<'e, E> {
    /// The pair, so `&mut *env` reborrows at each call site rather than moving the surgery.
    const fn new(env: &'e mut E, dsc: DscIdx) -> Self {
        Self { env, dsc }
    }
}

impl<E: DscTreeSurgery + ?Sized> SyncInsertion for DscSyncs<'_, E> {
    type At = NodeId;

    fn insert_sync_after(&mut self, at: NodeId, sync: SyncNode) -> NodeId {
        self.env
            .insert_sync(self.dsc, sync, InsertionPoint::After(at))
    }
}

/// `addL3LUAndLXLUSoftSyncNodeSequence(at)` — ENTRY 213, WHICH THIS BATCH DOES NOT SCHEDULE. Its
/// anchor is still open in this file and its body is a SOFT L3LU/LXLU pair, so a second spelling here
/// would be a second answer to that unit.
fn add_l3_lu_and_lx_lu_soft_sync_node_sequence<I: SyncInsertion + ?Sized>(
    _tree: &mut I,
    _at: I::At,
) {
    todo!("e213_addL3LUAndLXLUSoftSyncNodeSequence is not scheduled in this batch")
}

/// `allocNode->allocUsers_` RESTRICTED TO ITS TRANSFER USERS running `src` to `dst`, in the tree's own
/// DFS order — the walk entry 288 writes out five times over.
fn alloc_user_transfers<T: TransferNodes + ?Sized>(
    trees: &T,
    dsc: DscIdx,
    users: &[NodeId],
    src: SenComponent,
    dst: SenComponent,
) -> Vec<NodeId> {
    trees
        .transfers(dsc)
        .into_iter()
        .filter(|transfer| {
            users.contains(&transfer.node) && transfer.src == src && transfer.dst == dst
        })
        .map(|transfer| transfer.node)
        .collect()
}

/// Replaces: e288_createSynchronizationDSC
///
/// PUTS ONE DSC'S SYNC NODES IN ITS SCHEDULE TREE: an L3LU/LXLU handshake after the innermost
/// HBM->LX load (or after a neighbour fetch, or — failing both — after the L3-padded input's load),
/// an LXSU/L3SU and an L3SU/LXSU pair around the output's LX->HBM store, and, where the output is
/// loaded as well, an L3SU/L3LU pair at its allocation loop plus two more at the tree root.
///
/// ⛔ [`None`] IS EVERY `DT_CHECK_MSG`, and there are fourteen: *"Do not support both HBM pinned
/// tensor and input-neighbor fetch tensor .."*, *"Currently support only one LX input-neighbor fetch
/// tensor."*, each *"Expect .. in memOrg_."* with its *"Expect a valid allocate node."*, *"Expect a
/// valid transfer node."*, *"Expect a valid loop node."*, *"Expect a SuperChunk-by-chunk loop."*,
/// *"Unexpected SuperChunk-by-chunk loop."*, both *"Expect only one .. transfer node."* and
/// *"Currently expect only the input tensor at index 0 has L3 padding."*
/// ⛔ TRAP, AND IT IS THE REFERENCE'S: the neighbour branch takes the LAST `NO_COMPONENT`->LX
/// transfer of the allocation (it has no `break`) where the padding fallback takes the FIRST, and
/// only the padding fallback's handshake is the HARD one.
/// ⛔ TRAP: `getAllLabeledDsIndicesSet` yields each entry's RECORDED index and every `.at()` here
/// treats it as a POSITION, which is the reference's own conflation.
/// ⛔ DIVERGENCE: [`MemOrg::lx_zero_padded`] carries `DT_CHECK(isPresent && isPadded)`, so a
/// zero-padded-but-unpadded LX organisation refuses before its window dims are scanned rather than
/// after.
pub fn create_synchronization_dsc<O, T, E>(
    sdsc: &SuperDsc,
    dsc_idx: DscIdx,
    buffering: LxBuffering,
    orgs: &O,
    trees: &T,
    env: &mut E,
) -> Option<()>
where
    O: MemOrgs + ?Sized,
    T: TransferNodes + ?Sized,
    E: DscTreeSurgery + ?Sized,
{
    let config = sdsc.dscs().at(dsc_idx)?;
    let all_lds = all_labeled_ds_indices(config);
    let hbm_pinned = hbm_pinned_labeled_ds_indices(config);
    let lx_neighbor = lx_neighbor_labeled_ds_indices(sdsc, config, dsc_idx)?;
    (hbm_pinned.is_empty() || lx_neighbor.is_empty()).then_some(())?;

    let mut sync_l3lu_lxlu_inserted = false;
    if !lx_neighbor.is_empty() {
        (lx_neighbor.len() == 1).then_some(())?;
        let lds = *lx_neighbor.first()?;
        let users = orgs.mem_org(dsc_idx, lds)?.lx_alloc_users()?;
        let load = *alloc_user_transfers(
            trees,
            dsc_idx,
            &users,
            SenComponent::NoComponent,
            SenComponent::Lx,
        )
        .last()?;
        add_l3_lu_and_lx_lu_soft_sync_node_sequence(&mut DscSyncs::new(&mut *env, dsc_idx), load);
    } else if !hbm_pinned.is_empty() {
        let mut loads: Vec<NodeId> = Vec::new();
        for lds in &hbm_pinned {
            let users = orgs.mem_org(dsc_idx, *lds)?.hbm_alloc_users()?;
            loads.extend(alloc_user_transfers(
                trees,
                dsc_idx,
                &users,
                SenComponent::Hbm,
                SenComponent::Lx,
            ));
        }
        if !loads.is_empty() {
            match innermost_load_sync_plan(&*env, dsc_idx, &loads, buffering)? {
                LoadSyncPlan::Hard(at) => {
                    add_l3_lu_and_lx_lu_sync_node_sequence(
                        &mut DscSyncs::new(&mut *env, dsc_idx),
                        at,
                    );
                }
                LoadSyncPlan::SoftThenHard { soft, hard } => {
                    add_l3_lu_and_lx_lu_soft_sync_node_sequence(
                        &mut DscSyncs::new(&mut *env, dsc_idx),
                        soft,
                    );
                    add_l3_lu_and_lx_lu_sync_node_sequence(
                        &mut DscSyncs::new(&mut *env, dsc_idx),
                        hard,
                    );
                }
            }
            sync_l3lu_lxlu_inserted = true;
        }

        // The output tensor's own store and load, of which the reference expects at most one each.
        let mut store: Option<NodeId> = None;
        let mut load: Option<NodeId> = None;
        for lds in &hbm_pinned {
            if !config.labeled_ds.is_output(*lds) {
                continue;
            }
            let users = orgs.mem_org(dsc_idx, *lds)?.hbm_alloc_users()?;
            for transfer in trees.transfers(dsc_idx) {
                if !users.contains(&transfer.node) {
                    continue;
                }
                if transfer.src == SenComponent::Lx && transfer.dst == SenComponent::Hbm {
                    store.replace(transfer.node).is_none().then_some(())?;
                }
                if transfer.src == SenComponent::Hbm && transfer.dst == SenComponent::Lx {
                    load.replace(transfer.node).is_none().then_some(())?;
                }
            }
            // Break because we expect only one output tensor.
            break;
        }
        if let Some(store) = store {
            add_output_store_sync_nodes(env, dsc_idx, buffering, store)?;
            if let Some(load) = load {
                add_output_load_sync_nodes(env, dsc_idx, load)?;
            }
        }
    }

    if !sync_l3lu_lxlu_inserted {
        // L3 padding requires L3LU/LXLU sync nodes if they haven't been inserted, and it applies only
        // to a dim that is padded AND belongs to a window.
        let mut padded: Vec<LdsIdx> = Vec::new();
        for lds in &all_lds {
            let org = orgs.mem_org(dsc_idx, *lds)?;
            if !org.lx_zero_padded()? {
                continue;
            }
            let ds_type = config.labeled_ds.at(*lds)?.ds_type();
            let windowed = config
                .primary_ds_info
                .get(&ds_type)?
                .layout
                .iter()
                .any(|dim| {
                    config.full_padding.get(&dim).is_some_and(|pad| {
                        matches!(pad.sizes, PadSizes::Sized { .. }) && pad.window_dim.is_some()
                    })
                });
            if windowed {
                padded.push(*lds);
            }
        }
        if !padded.is_empty() {
            (padded.as_slice() == [LdsIdx(0)]).then_some(())?;
            let users = orgs.mem_org(dsc_idx, LdsIdx(0))?.lx_alloc_users()?;
            let load = alloc_user_transfers(
                trees,
                dsc_idx,
                &users,
                SenComponent::NoComponent,
                SenComponent::Lx,
            )
            .first()
            .copied()?;
            add_l3_lu_and_lx_lu_sync_node_sequence(&mut DscSyncs::new(&mut *env, dsc_idx), load);
        }
    }
    Some(())
}

/// WHERE THE L3LU/LXLU HANDSHAKE FOR THE HBM->LX LOADS GOES — one hard sequence, or a soft one at the
/// load with a hard one closing the innermost core-by-super-chunk loop.
enum LoadSyncPlan {
    /// The `else` arms and the head/core-by-super-chunk arm — a hard sequence after the load.
    Hard(NodeId),
    /// The super-chunk-by-chunk arm under spatial double buffering.
    SoftThenHard {
        /// The load itself.
        soft: NodeId,
        /// The last child of the innermost core-by-super-chunk loop.
        hard: NodeId,
    },
}

/// The plan above, computed with the tree borrowed and no mutation in flight.
fn innermost_load_sync_plan<R: DscTrees + ?Sized>(
    nesting: &R,
    dsc_idx: DscIdx,
    loads: &[NodeId],
    buffering: LxBuffering,
) -> Option<LoadSyncPlan> {
    let tree = nesting.tree(dsc_idx)?;
    // The loads with the MOST parent loops, which is `std::prev(map.end())` over the loop counts.
    let mut deepest = 0usize;
    let mut inner: BTreeSet<NodeId> = BTreeSet::new();
    for load in loads {
        let mut depth = 0usize;
        let mut node = *load;
        while let Some(owner) = tree.owner_loop(node) {
            depth += 1;
            node = owner.0;
        }
        if depth > deepest {
            deepest = depth;
            inner.clear();
        }
        if depth == deepest {
            inner.insert(*load);
        }
    }
    let at = insertion_node(tree, &inner, InsertSide::After)?;
    let LxBuffering::SpatialDouble(super_chunk) = buffering else {
        return Some(LoadSyncPlan::Hard(at));
    };
    let owner = tree.owner_loop(at)?;
    let divides = |loop_node: LoopId, num: DatastageId, den: DatastageId| {
        tree.loop_num(loop_node) == num && tree.loop_den(loop_node) == den
    };
    if Some(owner.0) == nesting.root(dsc_idx)
        || divides(owner, DATA_STAGE_CORE, super_chunk.index())
    {
        // Add hard sync nodes only after the transfer when the innermost HBM->LX transfer is inside a
        // core/SuperChunk loop.
        return Some(LoadSyncPlan::Hard(at));
    }
    divides(owner, super_chunk.index(), DATA_STAGE_CHUNK).then_some(())?;
    let mut outermost = tree.owner_loop(nesting.lx_below_block(dsc_idx)?)?;
    divides(outermost, super_chunk.index(), DATA_STAGE_CHUNK).then_some(())?;
    while !divides(
        tree.owner_loop(outermost.0)?,
        DATA_STAGE_CORE,
        super_chunk.index(),
    ) {
        outermost = tree.owner_loop(outermost.0)?;
    }
    let innermost_core = tree.owner_loop(outermost.0)?;
    let hard = tree.children(innermost_core.0).last().copied()?;
    Some(LoadSyncPlan::SoftThenHard { soft: at, hard })
}

/// The LXSU->L3SU pair before the output's LX->HBM store and the L3SU->LXSU pair whose position the
/// buffering decides.
fn add_output_store_sync_nodes<E: DscTreeSurgery + ?Sized>(
    env: &mut E,
    dsc_idx: DscIdx,
    buffering: LxBuffering,
    store: NodeId,
) -> Option<()> {
    let (send, receive) = sync_pair(
        SenComponent::Lxsu,
        SenComponent::L3su,
        "",
        SyncStrength::Hard,
    );
    env.insert_sync(dsc_idx, send, InsertionPoint::Before(store));
    env.insert_sync(dsc_idx, receive, InsertionPoint::Before(store));
    let (send, receive) = sync_pair(
        SenComponent::L3su,
        SenComponent::Lxsu,
        "",
        SyncStrength::Hard,
    );
    if matches!(buffering, LxBuffering::SpatialDouble(_)) {
        // Add after the allocate node.
        let lds = env.transfer_src_lds(dsc_idx, store)?;
        let alloc = env.allocation(dsc_idx, lds, SenComponent::Lx)?;
        let send = env.insert_sync(dsc_idx, send, InsertionPoint::After(alloc));
        env.insert_sync(dsc_idx, receive, InsertionPoint::After(send));
    } else {
        // Add before the LX->HBM transfer node.
        env.insert_sync(dsc_idx, send, InsertionPoint::Before(store));
        env.insert_sync(dsc_idx, receive, InsertionPoint::Before(store));
    }
    Some(())
}

/// The L3SU->L3LU pair at the output's allocation loop level and the two more at the tree root, added
/// only when the output tensor is loaded as well as stored.
fn add_output_load_sync_nodes<E: DscTreeSurgery + ?Sized>(
    env: &mut E,
    dsc_idx: DscIdx,
    load: NodeId,
) -> Option<()> {
    let (send, receive) = sync_pair(
        SenComponent::L3su,
        SenComponent::L3lu,
        "",
        SyncStrength::Hard,
    );
    let lds = env.transfer_src_lds(dsc_idx, load)?;
    let alloc = env.allocation(dsc_idx, lds, SenComponent::Lx)?;
    // The L3SU send node goes at the END of the allocation's owner loop, which can be the root.
    let owner = env.tree(dsc_idx)?.owner_loop(alloc)?.0;
    env.insert_sync(dsc_idx, receive, InsertionPoint::Before(alloc));
    env.insert_sync(dsc_idx, send, InsertionPoint::LastIn(owner));
    let root = env.root(dsc_idx)?;
    for outermost in 0..2 {
        let (send, receive) = sync_pair(
            SenComponent::L3su,
            SenComponent::L3lu,
            &format!("_outermost_{outermost}"),
            SyncStrength::Hard,
        );
        env.insert_sync(dsc_idx, send, InsertionPoint::FirstIn(root));
        env.insert_sync(dsc_idx, receive, InsertionPoint::LastIn(root));
    }
    Some(())
}

/// Replaces: e289_optimizeHbmTransfers
///
/// HOISTS EVERY TENSOR-TO-TENSOR TRANSFER out of the loops that do not change the chunk it moves: the
/// walk climbs from the transfer towards its LX allocation's own loop and stops at the first loop
/// whose single dim the tensor depends on AND whose two data stages state a different extent for it;
/// the transfer then moves beside the last loop it passed, BEFORE it for a load and AFTER it for a
/// store.
///
/// ⛔ A CROSS-CORE REDUCTION DSC IS SKIPPED — its transfers sit inside condition nodes this unit does
/// not consider.
/// ⛔ [`None`] IS *"Expect only one dimension."*, *"Unexpected transfer."*, *"Expect a valid schedule
/// tree."* and the two `memOrg_` aborts; the three `dataStageParam_.count(..)` checks are discharged
/// by [`DesignSpaceConfig::core_stage`], the mandatory chunk stage and [`LxBuffering`].
pub fn optimize_hbm_transfers<T, E>(sdsc: &SuperDsc, trees: &T, env: &mut E) -> Option<()>
where
    T: TransferNodes + ?Sized,
    E: DscTreeSurgery + ?Sized,
{
    for (config, index) in sdsc.dscs().iter().zip(0u32..) {
        let dsc_idx = DscIdx(index);
        if is_op_cross_core_reduction(sdsc, config)? {
            continue;
        }
        env.root(dsc_idx)?;
        let hoists = hbm_transfer_hoists(&*env, config, dsc_idx, trees)?;
        for (node, at) in hoists {
            env.move_node(dsc_idx, node, at);
        }
    }
    Some(())
}

/// WHERE EACH OF ONE DSC'S TENSOR-TO-TENSOR TRANSFERS LANDS, decided with the tree borrowed and
/// applied afterwards — the reference moves each node as it walks, and the moves are independent.
fn hbm_transfer_hoists<R, T>(
    nesting: &R,
    config: &DesignSpaceConfig,
    dsc_idx: DscIdx,
    trees: &T,
) -> Option<Vec<(NodeId, InsertionPoint)>>
where
    R: DscTrees + ?Sized,
    T: TransferNodes + ?Sized,
{
    let tree = nesting.tree(dsc_idx)?;
    let mut hoists: Vec<(NodeId, InsertionPoint)> = Vec::new();
    for transfer in trees.transfers(dsc_idx) {
        let Some(lds) = nesting.transfer_src_lds(dsc_idx, transfer.node) else {
            continue;
        };
        if !nesting.transfer_dst_is_lds(dsc_idx, transfer.node) {
            continue;
        }
        let depends_on = config.non_broadcast_lds_dim_set(lds)?;
        let alloc_owner = tree.owner_loop(nesting.allocation(dsc_idx, lds, SenComponent::Lx)?);
        let mut sibling: Option<LoopId> = None;
        let mut curr = tree.owner_loop(transfer.node);
        while let Some(loop_node) = curr.filter(|at| Some(*at) != alloc_owner) {
            let dims = tree.loop_dims(loop_node);
            let mut stated = dims.iter();
            let dim = stated.next()?.dim;
            stated.next().is_none().then_some(())?;
            let extent = |stage| config.data_stages.stage_extent(stage, dim);
            if depends_on.contains(&dim)
                && extent(tree.loop_num(loop_node)) != extent(tree.loop_den(loop_node))
            {
                break;
            }
            sibling = Some(loop_node);
            curr = tree.owner_loop(loop_node.0);
        }
        let Some(sibling) = sibling else { continue };
        let at = match (transfer.src, transfer.dst) {
            (SenComponent::Hbm, SenComponent::Lx | SenComponent::L3luibr)
            | (SenComponent::Lx, SenComponent::L3suibr) => InsertionPoint::Before(sibling.0),
            (SenComponent::Lx, SenComponent::Hbm) => InsertionPoint::After(sibling.0),
            _ => return None,
        };
        hoists.push((transfer.node, at));
    }
    Some(hoists)
}

/// Replaces: e290_createChunkLoops
///
/// BUILDS THE CHUNK LOOP NEST from DSC 0's loop order — the group's DSCs share one order and one set
/// of transfers, so the first DSC decides both and each DSC's own chunk parameters then size the nest.
///
/// ⛔ `if (dsc.labeledDs_.size() < 1) return;` IS UNSPELLABLE: [`LabeledDsList`] is non-empty by
/// construction.
/// ⛔ THREE OF ITS FIVE CALLEES ARE NOT SCHEDULED IN THIS BATCH — entries 215, 216 and 217, whose
/// anchors are still open in this file — and they are what mints the loop nodes, so the nest cannot be
/// stated here without a second answer to each of them.
pub fn create_chunk_loops(sdsc: &SuperDsc) -> Option<()> {
    let main = sdsc.dscs().first();
    let _reuse = has_dimension_reuse(main);
    let _loop_order_dims = collect_all_dimensions_for_loop_order(main)?;
    todo!(
        "e215_buildScheduleDimensionsTable, e216_buildLoopOrder and e217_createChunkLoopNodes are \
         not scheduled in this batch"
    )
}

#[cfg(test)]
mod tests_e283_e290 {
    // ⭐ TESTS FOR ENTRIES 283-290. One node-id tree, one organisation map and one transfer list
    // serve all of them. ⛔ ENTRY 290 HAS NO TEST: its body is a `todo!` naming entries 215, 216 and
    // 217, which this batch does not schedule.
    use super::*;

    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::StickDims;
    use crate::schedule::dsc2::LayoutDims;
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, DataStage, DscList, LabeledDsList, NamedDims, PrimaryDsInfo, StageDims,
    };

    fn core(index: u32) -> Core {
        Core::checked(index).expect("this arch has the core the test names")
    }

    fn count(count: u32) -> WkSliceCount {
        WkSliceCount::new(NonZeroU32::new(count).expect("a positive slice count"))
    }

    fn slice(ids: &[(PrimaryDim, i32)]) -> WkSlice {
        WkSlice(ids.iter().map(|&(dim, id)| (dim, WkSliceId(id))).collect())
    }

    fn dims(extents: &[(PrimaryDim, i64)]) -> FilledDims {
        let mut stage = StageDims::default();
        for &(dim, extent) in extents {
            stage.extents.insert(dim, Extent(extent));
        }
        FilledDims::of(stage).expect("a stage that states a dim")
    }

    fn stage(name: &str, extents: &[(PrimaryDim, i64)]) -> DataStage {
        let name = StageName(name.to_owned());
        DataStage {
            ss: NamedDims {
                name: name.clone(),
                dims: dims(extents),
            },
            el: NamedDims {
                name,
                dims: dims(extents),
            },
        }
    }

    /// A primary data structure that lays out the dims given, outermost first, on a one-element stick.
    fn layout(dims: &[PrimaryDim]) -> PrimaryDsInfo {
        let (first, rest) = dims.split_first().expect("a layout with a dim in it");
        PrimaryDsInfo {
            layout: LayoutDims::new(*first, rest.to_vec()),
            stick: StickDims::default(),
        }
    }

    fn labeled(
        ds_type: DsType,
        recorded: LdsIdx,
        scales: &[(PrimaryDim, Scale)],
        pinning: Pinning,
    ) -> LabeledDs {
        LabeledDs::new(ds_type, scales.to_vec(), recorded, pinning)
    }

    /// `memOrg_.at(HBM).isPresent` — what makes a tensor transferred rather than resident.
    fn hbm() -> Pinning {
        Pinning {
            mem_org: BTreeMap::from([(SenComponent::Hbm, true)]),
            lx: false,
            lx_padded: false,
        }
    }

    /// `isLxPinned()` — resident, and so no L3 transfer of its own.
    fn lx() -> Pinning {
        Pinning {
            mem_org: BTreeMap::from([(SenComponent::Lx, true)]),
            lx: true,
            lx_padded: false,
        }
    }

    fn a_dsc(core_extents: &[(PrimaryDim, i64)], chunk: &[(PrimaryDim, i64)]) -> DesignSpaceConfig {
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelets_used_dsc2: Some(CoreletsUsed::ONE),
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(core(0), vec![]),
            layout_dims: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(
                labeled(DsType::Input, LdsIdx(0), &[], Pinning::default()),
                vec![],
            ),
            data_stages: L3DataStages::new(stage("core", core_extents), stage("chunk", chunk)),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
        }
    }

    fn a_sdsc(
        dsc: DesignSpaceConfig,
        slices: &[(PrimaryDim, u32)],
        cores: &[(Core, WkSlice)],
    ) -> SuperDsc {
        SuperDsc::new(
            DscList::new(dsc, vec![]),
            slices.iter().map(|&(dim, n)| (dim, count(n))).collect(),
            cores.iter().cloned().collect(),
            BTreeMap::new(),
        )
    }

    /// One labelled DS's `memOrg_` as these entries read it, stated by field.
    #[derive(Default)]
    struct Org {
        padding: Option<PaddingForm>,
        hbm_users: Option<Vec<NodeId>>,
        lx_users: Option<Vec<NodeId>>,
    }

    impl MemOrg for Org {
        fn hbm_pinned(&self) -> bool {
            false
        }

        fn lx_buffering(&self) -> Option<Buffering> {
            None
        }

        fn lx_start_address(&self, _at: &AddressCoord) -> Option<ByteAddress> {
            None
        }

        fn lx_buffer_offset(&self, _core: Core, _corelet: Corelet) -> Option<BufferOffset> {
            None
        }

        fn hbm_indirection(&self) -> Option<IndirectAlloc> {
            None
        }

        fn hbm_allocation(&self) -> Option<NodeName> {
            None
        }

        fn hbm_layout_dims(&self) -> Option<LayoutDims> {
            None
        }

        fn hbm_page_dims(&self) -> BTreeSet<PrimaryDim> {
            BTreeSet::new()
        }

        fn lx_padding(&self) -> Option<PaddingForm> {
            self.padding.clone()
        }

        fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent> {
            BTreeMap::new()
        }

        fn hbm_alloc_users(&self) -> Option<Vec<NodeId>> {
            self.hbm_users.clone()
        }

        fn lx_alloc_users(&self) -> Option<Vec<NodeId>> {
            self.lx_users.clone()
        }

        fn lx_zero_padded(&self) -> Option<bool> {
            Some(false)
        }
    }

    /// The organisations of one DSC, by the labelled DS index the entries hand them.
    struct Orgs(BTreeMap<LdsIdx, Org>);

    impl MemOrgs for Orgs {
        type Org = Org;

        fn mem_org(&self, _dsc: DscIdx, lds: LdsIdx) -> Option<&Org> {
            self.0.get(&lds)
        }
    }

    /// One DSC's transfer nodes, whichever DSC is asked for.
    struct Transfers(Vec<L3Transfer>);

    impl TransferNodes for Transfers {
        fn transfers(&self, _dsc: DscIdx) -> Vec<L3Transfer> {
            self.0.clone()
        }
    }

    /// The kinds of node these entries walk, mint and move.
    #[derive(Debug, Clone)]
    enum Kind {
        Block,
        Loop(LoopNode),
        Transfer,
        Allocate,
        Sync,
    }

    #[derive(Debug, Clone)]
    struct Entry {
        name: NodeName,
        parent: Option<NodeId>,
        children: Vec<NodeId>,
        kind: Kind,
    }

    /// ONE DSC'S SCHEDULE TREE BY NODE ID, plus the `memOrg_` allocations and transfer ends entries
    /// 288 and 289 state their insertion points against.
    #[derive(Debug, Default)]
    struct Tree {
        nodes: BTreeMap<NodeId, Entry>,
        next: u32,
        head: Option<NodeId>,
        lx_below: Option<NodeId>,
        allocations: BTreeMap<(LdsIdx, SenComponent), NodeId>,
        src_lds: BTreeMap<NodeId, LdsIdx>,
        dst_is_lds: BTreeSet<NodeId>,
    }

    impl Tree {
        fn add(&mut self, name: &str, kind: Kind, parent: Option<NodeId>) -> NodeId {
            let id = NodeId(self.next);
            self.next += 1;
            self.nodes.insert(
                id,
                Entry {
                    name: NodeName(name.to_owned()),
                    parent,
                    children: Vec::new(),
                    kind,
                },
            );
            if let Some(parent) = parent {
                self.nodes
                    .get_mut(&parent)
                    .expect("parent exists")
                    .children
                    .push(id);
            }
            id
        }

        /// `scheduleTree_.getHead()`.
        fn root_block(&mut self, name: &str) -> NodeId {
            let id = self.add(name, Kind::Block, None);
            self.head = Some(id);
            id
        }

        fn loop_over(
            &mut self,
            num: DatastageId,
            den: DatastageId,
            dim: PrimaryDim,
            parent: NodeId,
        ) -> LoopId {
            let node = construct_loop_node(
                num,
                den,
                LoopDims::new(
                    PrimaryDimAndKind {
                        dim,
                        kind: MetaDimKind::Unpadded,
                    },
                    Vec::new(),
                ),
            );
            let name = node.name.0.clone();
            LoopId(self.add(&name, Kind::Loop(node), Some(parent)))
        }

        /// A tensor-to-tensor transfer out of `src` — the shape entry 289 hoists.
        fn transfer(&mut self, name: &str, parent: NodeId, src: LdsIdx) -> NodeId {
            let id = self.add(name, Kind::Transfer, Some(parent));
            self.src_lds.insert(id, src);
            self.dst_is_lds.insert(id);
            id
        }

        fn allocate(
            &mut self,
            name: &str,
            lds: LdsIdx,
            storage: SenComponent,
            parent: NodeId,
        ) -> NodeId {
            let id = self.add(name, Kind::Allocate, Some(parent));
            self.allocations.insert((lds, storage), id);
            id
        }

        fn unlink(&mut self, node: NodeId) {
            let parent = self
                .nodes
                .get_mut(&node)
                .expect("node exists")
                .parent
                .take();
            if let Some(parent) = parent {
                self.nodes
                    .get_mut(&parent)
                    .expect("parent exists")
                    .children
                    .retain(|child| *child != node);
            }
        }

        fn link(&mut self, node: NodeId, at: InsertionPoint) {
            let (parent, index) = match at {
                InsertionPoint::Before(sibling) | InsertionPoint::After(sibling) => {
                    let parent = self.nodes[&sibling].parent.expect("sibling has a parent");
                    let position = self.nodes[&parent]
                        .children
                        .iter()
                        .position(|child| *child == sibling)
                        .expect("sibling among its parent's children");
                    let after = matches!(at, InsertionPoint::After(_));
                    (parent, position + usize::from(after))
                }
                InsertionPoint::FirstIn(parent) => (parent, 0),
                InsertionPoint::LastIn(parent) => (parent, self.nodes[&parent].children.len()),
            };
            self.nodes
                .get_mut(&parent)
                .expect("parent exists")
                .children
                .insert(index, node);
            self.nodes.get_mut(&node).expect("node exists").parent = Some(parent);
        }

        fn names(&self, nodes: &[NodeId]) -> Vec<String> {
            nodes
                .iter()
                .map(|node| self.nodes[node].name.0.clone())
                .collect()
        }

        fn minted_loop(&self, loop_node: LoopId) -> &LoopNode {
            match &self.nodes[&loop_node.0].kind {
                Kind::Loop(node) => node,
                other => panic!("not a loop: {other:?}"),
            }
        }
    }

    impl NodeParents for Tree {
        fn parent(&self, node: NodeId) -> Option<NodeId> {
            self.nodes[&node].parent
        }

        fn children(&self, parent: NodeId) -> Vec<NodeId> {
            self.nodes[&parent].children.clone()
        }
    }

    impl LoopNesting for Tree {
        fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
            let mut current = self.nodes[&node].parent;
            while let Some(candidate) = current {
                if matches!(self.nodes[&candidate].kind, Kind::Loop(_)) {
                    return Some(LoopId(candidate));
                }
                current = self.nodes[&candidate].parent;
            }
            None
        }

        fn has_parent(&self, node: LoopId) -> bool {
            self.nodes[&node.0].parent.is_some()
        }
    }

    impl LoopStages for Tree {
        fn loop_num(&self, loop_node: LoopId) -> DatastageId {
            self.minted_loop(loop_node).num
        }

        fn loop_den(&self, loop_node: LoopId) -> DatastageId {
            self.minted_loop(loop_node).den
        }

        fn loop_dims(&self, loop_node: LoopId) -> LoopDims {
            self.minted_loop(loop_node).dims.clone()
        }
    }

    impl DscLoopStages for Tree {
        type Stages = Self;

        fn loop_stages(&self, _dsc: DscIdx) -> Option<&Self> {
            Some(self)
        }
    }

    impl DscTrees for Tree {
        type Tree = Self;

        fn tree(&self, _dsc: DscIdx) -> Option<&Self> {
            Some(self)
        }

        fn root(&self, _dsc: DscIdx) -> Option<NodeId> {
            self.head
        }

        fn lx_below_block(&self, _dsc: DscIdx) -> Option<NodeId> {
            self.lx_below
        }

        fn allocation(&self, _dsc: DscIdx, lds: LdsIdx, storage: SenComponent) -> Option<NodeId> {
            self.allocations.get(&(lds, storage)).copied()
        }

        fn transfer_src_lds(&self, _dsc: DscIdx, node: NodeId) -> Option<LdsIdx> {
            self.src_lds.get(&node).copied()
        }

        fn transfer_dst_is_lds(&self, _dsc: DscIdx, node: NodeId) -> bool {
            self.dst_is_lds.contains(&node)
        }
    }

    impl DscTreeSurgery for Tree {
        fn insert_sync(&mut self, _dsc: DscIdx, sync: SyncNode, at: InsertionPoint) -> NodeId {
            let name = sync.name.0.clone();
            let id = self.add(&name, Kind::Sync, None);
            self.link(id, at);
            id
        }

        fn move_node(&mut self, _dsc: DscIdx, node: NodeId, at: InsertionPoint) {
            self.unlink(node);
            self.link(node, at);
        }
    }

    /// e283 — a corelet-split dim is REPLACED by one equal share per corelet, and a share the corelet
    /// count does not divide is *"Invalid corelet split."*
    #[test]
    fn the_corelet_split_is_one_equal_share_per_corelet_or_a_refusal() {
        let mut dsc = a_dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 4)]);
        dsc.corelets_used = CoreletsUsed::new(NonZeroU32::new(2).expect("two corelets"));
        dsc.corelet_shares.insert(
            PrimaryDim::I,
            CoreletShare {
                corelet0: Extent(4),
                whole: Extent(8),
            },
        );
        // The dim the stage does not state is the reference's `-1`, and is SKIPPED.
        let mut params = dims(&[(PrimaryDim::I, 8), (PrimaryDim::J, 6)]);
        assert_eq!(
            add_or_update_corelet_split_in_params(&mut params, &dsc),
            Some(())
        );
        assert_eq!(
            params.dims().corelet_split,
            BTreeMap::from([(PrimaryDim::I, vec![Extent(4), Extent(4)])])
        );

        let mut odd = dims(&[(PrimaryDim::I, 7)]);
        assert_eq!(add_or_update_corelet_split_in_params(&mut odd, &dsc), None);
    }

    /// e284 — a conv2d, a pooling and a depthwise conv all slide a window; a matmul does not, and
    /// neither does an unnamed op func.
    #[test]
    fn the_strided_window_ops_are_the_conv_and_pooling_families() {
        for op in [
            OpFunc::Conv2DFwd,
            OpFunc::MaxpoolFwd,
            OpFunc::AvgpoolFwd,
            OpFunc::DepthwiseConvFwd,
        ] {
            assert!(is_op_func_strided_window(Some(op)), "{op:?}");
        }
        assert!(!is_op_func_strided_window(Some(OpFunc::BatchmatmulInt8Fwd)));
        assert!(!is_op_func_strided_window(None));
    }

    /// The one HBM-pinned input of entries 285 and 286: a core of 8 elements chunked into 4, one
    /// core, one work slice, and a transfer that sits at the root so it repeats once.
    fn a_transferred_input() -> (SuperDsc, Orgs, Transfers, Tree) {
        let mut dsc = a_dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 4)]);
        dsc.primary_ds_info
            .insert(DsType::Input, layout(&[PrimaryDim::I]));
        dsc.labeled_ds = LabeledDsList::new(
            labeled(
                DsType::Input,
                LdsIdx(0),
                &[(PrimaryDim::I, Scale::Sized(1.0))],
                hbm(),
            ),
            vec![],
        );
        dsc.layout_dims
            .insert(LdsIdx(0), LayoutDims::new(PrimaryDim::I, Vec::new()));
        dsc.lx_chunk_capacity
            .insert(LdsIdx(0), Bytes(4 * Target::BYTES_PER_STICK.get()));
        let sdsc = a_sdsc(
            dsc,
            &[(PrimaryDim::I, 1)],
            &[(core(0), slice(&[(PrimaryDim::I, 0)]))],
        );

        let mut tree = Tree::default();
        let root = tree.root_block("root");
        let load = tree.transfer("hbm_to_lx", root, LdsIdx(0));
        let orgs = Orgs(BTreeMap::from([(
            LdsIdx(0),
            Org {
                padding: Some(PaddingForm::default()),
                hbm_users: Some(vec![load]),
                ..Org::default()
            },
        )]));
        let trees = Transfers(vec![L3Transfer {
            node: load,
            name: NodeName("hbm_to_lx".to_owned()),
            src: SenComponent::Hbm,
            dst: SenComponent::Lx,
        }]);
        (sdsc, orgs, trees, tree)
    }

    /// e285 — one chunk of 4 sticks is one 4-stick burst, tallied over the 2 stick volumes a core
    /// holds at multicast degree 1, so the average is the table's row 4, column 1.
    #[test]
    fn the_burst_efficiency_is_the_remainder_burst_at_degree_one() {
        let (sdsc, orgs, trees, tree) = a_transferred_input();
        assert_eq!(
            calculate_burst_efficiency(&sdsc, &orgs, &trees, &tree),
            Some(BurstEfficiency(0.1750))
        );
    }

    /// e286 — the 4-element chunk is 4 MACs on one core, over the 4 sticks of LX the transfer fills
    /// once.
    #[test]
    fn the_flop_per_byte_is_the_chunk_macs_over_the_lx_chunk_capacity() {
        let (sdsc, orgs, trees, tree) = a_transferred_input();
        let bytes = (4 * Target::BYTES_PER_STICK.get()) as f64;
        assert_eq!(
            calculate_flop_per_byte(&sdsc, &[PrimaryDim::I], &orgs, &trees, &tree),
            Some(FlopPerByte(8.0 / bytes))
        );
    }

    /// e287 — two slices on the reduced dim and two on the kept one make two groups of two cores,
    /// each core placed at the slice its reduced dim names.
    #[test]
    fn the_reduction_groups_are_the_kept_slices_holding_the_reduced_ones() {
        let mut dsc = a_dsc(
            &[(PrimaryDim::I, 8), (PrimaryDim::Ki, 8)],
            &[(PrimaryDim::I, 4)],
        );
        dsc.labeled_ds = LabeledDsList::new(
            labeled(
                DsType::Input,
                LdsIdx(0),
                &[
                    (PrimaryDim::I, Scale::Sized(1.0)),
                    (PrimaryDim::Ki, Scale::Sized(1.0)),
                ],
                Pinning::default(),
            ),
            vec![labeled(
                DsType::Output,
                LdsIdx(1),
                &[(PrimaryDim::I, Scale::Sized(1.0))],
                Pinning::default(),
            )],
        );
        dsc.layout_dims = BTreeMap::from([
            (
                LdsIdx(0),
                LayoutDims::new(PrimaryDim::I, vec![PrimaryDim::Ki]),
            ),
            (LdsIdx(1), LayoutDims::new(PrimaryDim::I, Vec::new())),
        ]);
        let sdsc = a_sdsc(
            dsc.clone(),
            &[(PrimaryDim::I, 2), (PrimaryDim::Ki, 2)],
            &[
                (core(0), slice(&[(PrimaryDim::I, 0), (PrimaryDim::Ki, 0)])),
                (core(1), slice(&[(PrimaryDim::I, 0), (PrimaryDim::Ki, 1)])),
                (core(2), slice(&[(PrimaryDim::I, 1), (PrimaryDim::Ki, 0)])),
                (core(3), slice(&[(PrimaryDim::I, 1), (PrimaryDim::Ki, 1)])),
            ],
        );

        let groups = cross_core_reduction_group_info(&sdsc, &dsc).expect("a cross-core reduction");
        let ends = |group: &CrossCoreReductionGroup| {
            let cores = group.cores().expect("a group with a slice in it");
            (
                cores.start_core_at_corelet(GroupCorelet::Zero),
                cores.end_core_at_corelet(GroupCorelet::Zero),
            )
        };
        assert_eq!(groups.len(), 2);
        assert_eq!(ends(&groups[0]), (Some(core(0)), Some(core(1))));
        assert_eq!(ends(&groups[1]), (Some(core(2)), Some(core(3))));
    }

    /// e288 — the one HBM->LX load is the innermost, so the four-node hard handshake chains straight
    /// after it, in the order it was minted.
    #[test]
    fn the_load_gains_the_four_node_l3lu_lxlu_handshake_after_it() {
        let mut dsc = a_dsc(&[(PrimaryDim::I, 8)], &[(PrimaryDim::I, 4)]);
        dsc.labeled_ds = LabeledDsList::new(
            labeled(
                DsType::Input,
                LdsIdx(0),
                &[(PrimaryDim::I, Scale::Sized(1.0))],
                hbm(),
            ),
            vec![labeled(
                DsType::Output,
                LdsIdx(1),
                &[(PrimaryDim::I, Scale::Sized(1.0))],
                lx(),
            )],
        );
        let sdsc = a_sdsc(dsc, &[], &[]);

        let mut tree = Tree::default();
        let root = tree.root_block("root");
        let chunk_loop = tree.loop_over(DATA_STAGE_CORE, DATA_STAGE_CHUNK, PrimaryDim::I, root);
        let load = tree.transfer("hbm_to_lx", chunk_loop.0, LdsIdx(0));
        let orgs = Orgs(BTreeMap::from([(
            LdsIdx(0),
            Org {
                hbm_users: Some(vec![load]),
                ..Org::default()
            },
        )]));
        let trees = Transfers(vec![L3Transfer {
            node: load,
            name: NodeName("hbm_to_lx".to_owned()),
            src: SenComponent::Hbm,
            dst: SenComponent::Lx,
        }]);

        assert_eq!(
            create_synchronization_dsc(
                &sdsc,
                DscIdx(0),
                LxBuffering::Double,
                &orgs,
                &trees,
                &mut tree,
            ),
            Some(())
        );
        let children = NodeParents::children(&tree, chunk_loop.0);
        assert_eq!(
            tree.names(&children),
            vec![
                "hbm_to_lx",
                "sync_send_l3lu_to_lxlu",
                "sync_receive_lxlu_from_l3lu",
                "sync_send_lxlu_to_l3lu",
                "sync_receive_l3lu_from_lxlu",
            ]
        );
    }

    /// e289 — the load passes the inner loop, whose dim it does not depend on, and stops at the outer
    /// one, which chunks a dim it does; it lands BEFORE the last loop it passed.
    #[test]
    fn the_load_is_hoisted_before_the_innermost_loop_it_does_not_depend_on() {
        let mut dsc = a_dsc(
            &[(PrimaryDim::I, 8), (PrimaryDim::J, 4)],
            &[(PrimaryDim::I, 4), (PrimaryDim::J, 4)],
        );
        dsc.labeled_ds = LabeledDsList::new(
            labeled(
                DsType::Input,
                LdsIdx(0),
                &[
                    (PrimaryDim::I, Scale::Sized(1.0)),
                    (PrimaryDim::J, Scale::UnitStick),
                ],
                hbm(),
            ),
            vec![labeled(
                DsType::Output,
                LdsIdx(1),
                &[(PrimaryDim::I, Scale::Sized(1.0))],
                lx(),
            )],
        );
        dsc.layout_dims = BTreeMap::from([
            (
                LdsIdx(0),
                LayoutDims::new(PrimaryDim::I, vec![PrimaryDim::J]),
            ),
            (LdsIdx(1), LayoutDims::new(PrimaryDim::I, Vec::new())),
        ]);
        let sdsc = a_sdsc(dsc, &[], &[]);

        let mut tree = Tree::default();
        let root = tree.root_block("root");
        tree.allocate("alloc_lx", LdsIdx(0), SenComponent::Lx, root);
        let outer = tree.loop_over(DATA_STAGE_CORE, DATA_STAGE_CHUNK, PrimaryDim::I, root);
        let inner = tree.loop_over(DATA_STAGE_CORE, DATA_STAGE_CHUNK, PrimaryDim::J, outer.0);
        let load = tree.transfer("hbm_to_lx", inner.0, LdsIdx(0));
        let trees = Transfers(vec![L3Transfer {
            node: load,
            name: NodeName("hbm_to_lx".to_owned()),
            src: SenComponent::Hbm,
            dst: SenComponent::Lx,
        }]);

        assert_eq!(optimize_hbm_transfers(&sdsc, &trees, &mut tree), Some(()));
        assert_eq!(
            NodeParents::children(&tree, outer.0),
            vec![load, inner.0],
            "the load is hoisted out of the J loop and placed before it"
        );
        assert!(NodeParents::children(&tree, inner.0).is_empty());
    }
}

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

#[cfg(test)]
mod unit_tests {
    use sys_arch_spec::arch_enums::{DataLocation, SenComponent};

    use super::*;
    use crate::schedule::ddc::v1::OpFuncs;
    use crate::schedule::dsc2::LdsIdx;

    /// A `DataStructDims` STAND-IN — which dims it states a window extent for, and nothing else,
    /// which is every question these units put to one.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct Dims(BTreeSet<PrimaryDim>);

    impl WindowExtents for Dims {
        fn window_dims(&self) -> BTreeSet<PrimaryDim> {
            self.0.clone()
        }
    }

    /// `computeOp_` with one entry.
    struct Ops(Option<OpFunc>);

    impl ComputeOps for Ops {
        fn op_funcs(&self) -> OpFuncs {
            OpFuncs::new(self.0, Vec::new())
        }

        fn set_first_op_func(&mut self, op_func: OpFunc) {
            self.0 = Some(op_func);
        }
    }

    fn via(unit: SenComponent, storage: SenComponent, lds: u32) -> Via {
        Via {
            loc: DataLocation { unit, storage },
            lds: Some(LdsIdx(lds)),
        }
    }

    fn dims(of: &[PrimaryDim]) -> Dims {
        Dims(of.iter().copied().collect())
    }

    #[test]
    fn a_transfer_node_carries_every_end_with_its_storage_and_its_lds_index() {
        let node = create_transfer_node(
            via(SenComponent::L3lu, SenComponent::Hbm, 7),
            via(SenComponent::L3lu, SenComponent::Lx, 7),
            &[via(SenComponent::L3su, SenComponent::Lx, 9)],
            NodeName("transfer_lds7_src:HBM_dst:LX".to_owned()),
        );

        assert_eq!(node.src.unit, SenComponent::L3lu);
        assert_eq!(node.src.storage, SenComponent::Hbm);
        assert_eq!(node.src.data.my_lds_idx, Some(LdsIdx(7)));
        // A freshly minted end states no `dataConnect_`.
        assert_eq!(node.src.data.data_connect, None);

        let dsts: Vec<_> = node
            .dsts
            .iter()
            .map(|dst| (dst.unit, dst.storage, dst.data.my_lds_idx))
            .collect();
        assert_eq!(
            dsts,
            vec![
                (SenComponent::L3lu, SenComponent::Lx, Some(LdsIdx(7))),
                (SenComponent::L3su, SenComponent::Lx, Some(LdsIdx(9))),
            ]
        );
    }

    #[test]
    fn a_loop_dim_is_a_window_dim_only_where_the_core_stage_windows_ki_or_kj() {
        let stages = DataStages(
            [(
                CoreWindowDims::CORE,
                DataStage {
                    ss: StageDims {
                        name: StageName("0".to_owned()),
                        dims: dims(&[PrimaryDim::Ki, PrimaryDim::X]),
                    },
                    el: StageDims::default(),
                },
            )]
            .into_iter()
            .collect(),
        );
        let core = CoreWindowDims::of(&stages).expect("the core data stage is stated");

        let node = create_loop_node(
            &core,
            PrimaryDim::Ki,
            &[PrimaryDim::Kj, PrimaryDim::X],
            DatastageId(1),
            DatastageId(0),
            NodeName("loop_ds1_ds0".to_owned()),
        );

        let kinds: Vec<_> = node.dims.iter().map(|dim| (dim.dim, dim.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                (PrimaryDim::Ki, MetaDimKind::WindowDim),
                // `kj` is not windowed by the core stage, and `x` is not a dim the test even reads.
                (PrimaryDim::Kj, MetaDimKind::Unpadded),
                (PrimaryDim::X, MetaDimKind::Unpadded),
            ]
        );
        assert_eq!(node.num, DatastageId(1));
        assert_eq!(node.den, DatastageId(0));

        // And a DSC whose core data stage is absent yields no witness at all.
        assert!(CoreWindowDims::of(&DataStages::<Dims>::default()).is_none());
    }

    #[test]
    fn a_block_node_carries_the_name_it_was_minted_with() {
        let node = create_block_node(NodeName("block_lds3".to_owned()));
        assert_eq!(node.name, NodeName("block_lds3".to_owned()));
    }

    #[test]
    fn a_sync_node_records_its_units_which_end_it_is_and_how_it_signals() {
        let node = create_sync_node(
            SyncUnits::new(SenComponent::L3su, [SenComponent::L3lu]),
            NodeName("sync_receive_L3SU_to_L3LU".to_owned()),
            SyncDirection::Receive,
            SyncStrength::Soft,
        );

        assert_eq!(
            node.units.iter().collect::<Vec<_>>(),
            vec![SenComponent::L3lu, SenComponent::L3su]
        );
        assert_eq!(node.direction, SyncDirection::Receive);
        assert_eq!(node.strength, SyncStrength::Soft);
    }

    #[test]
    fn the_op_func_name_is_the_first_compute_op_s() {
        assert_eq!(
            get_op_func_name(&Ops(Some(OpFunc::Conv2DInt4Fwd))),
            Some(OpFunc::Conv2DInt4Fwd)
        );
        // `OpFuncs::NONE` reaches the caller as the absence it is.
        assert_eq!(get_op_func_name(&Ops(None)), None);
    }

    #[test]
    fn writing_a_data_stage_replaces_both_halves_and_a_stated_nothing_has_no_witness() {
        let mut stages = DataStages::default();
        let stage = StatedStage::of(
            dims(&[PrimaryDim::X]),
            StageName("3".to_owned()),
            dims(&[PrimaryDim::Y]),
            StageName("3el".to_owned()),
        )
        .expect("both halves state an extent");
        add_or_update_data_stage_param(&mut stages, stage, DatastageId(3));

        let update = StatedStage::of(
            dims(&[PrimaryDim::In]),
            StageName("3".to_owned()),
            dims(&[PrimaryDim::Out]),
            StageName("3el".to_owned()),
        )
        .expect("both halves state an extent");
        add_or_update_data_stage_param(&mut stages, update, DatastageId(3));

        let written = &stages.0[&DatastageId(3)];
        assert_eq!(written.ss.dims, dims(&[PrimaryDim::In]));
        assert_eq!(written.ss.name, StageName("3".to_owned()));
        assert_eq!(written.el.dims, dims(&[PrimaryDim::Out]));
        assert_eq!(written.el.name, StageName("3el".to_owned()));

        // And a half that states nothing has no witness.
        assert!(
            StatedStage::of(
                Dims::default(),
                StageName("4".to_owned()),
                dims(&[PrimaryDim::X]),
                StageName("4el".to_owned()),
            )
            .is_none()
        );
    }

    #[test]
    fn the_int4_conv2ds_are_the_three_int4_forms_and_nothing_else() {
        for op in [
            OpFunc::Conv2DInt4Fwd,
            OpFunc::Conv2DInt4FwdGenkg3,
            OpFunc::Conv2DInt4FwdSparsekg3,
        ] {
            assert!(is_op_func_conv2d_int4(Some(op)));
        }
        assert!(!is_op_func_conv2d_int4(Some(OpFunc::Conv2DInt8Fwd)));
        assert!(!is_op_func_conv2d_int4(None));
    }

    #[test]
    fn the_output_stationary_conv2ds_are_the_four_os1_forms_and_nothing_else() {
        for op in [
            OpFunc::Conv2DFwdOs1,
            OpFunc::Conv2DXrfInt8FwdOs1,
            OpFunc::Conv2DFwdGenOs1,
            OpFunc::Conv2DInt8FwdOs1,
        ] {
            assert!(is_op_func_conv2d_os1(Some(op)));
        }
        assert!(!is_op_func_conv2d_os1(Some(OpFunc::Conv2DFwd)));
        assert!(!is_op_func_conv2d_os1(None));
    }
}
