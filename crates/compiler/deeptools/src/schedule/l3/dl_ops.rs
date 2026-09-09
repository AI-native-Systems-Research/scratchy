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

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::schedule::ddc::metadata::{DatastageId, MetaDimKind};
use crate::schedule::ddc::transformation::{DsType, Scale};
use crate::schedule::ddc::transformation_util::{
    DataStage, DataStages, LoopDims, LoopNode, PrimaryDimAndKind, StageDims, StageName,
};
use crate::schedule::ddc::v1::ComputeOps;
use crate::schedule::dsc2::{
    BlockNode, Dsts, LdsIdx, NodeName, ReplicationFactor, SyncDirection, SyncNode, SyncStrength,
    SyncUnits, TransferNode, Via,
};
use crate::schedule::l3::dsc::{
    CoreletShare, DesignSpaceConfig, DscGroup, FilledDims, LabeledDs, MulticastDegree, SuperDsc,
    SymbolicDimInfo, UnneededPad,
};
use crate::units::Core;
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::arch_enums::OpFunc;

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
/// never written (`:126`), so its zeroing arm is DEAD: as a const generic that arm leaves the build
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
/// ⛔ `None` IS `coreIdToWkSlice_.at(dim)` THROWING: a layout dim that no work slice states. The
/// `!dscIndices.empty()` check, the `dscs_.at()` lookups and the `coreIdsUsed_[0]` subscript are
/// [`DscGroup`] and `CoreIdsUsed`, discharged before this is called.
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
/// (`:310-313`). The `int` against `size()` comparison beside it is signed/unsigned but harmless.
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
        CoreIdsUsed, CoreletsUsed, DimPadding, DscList, Granularity, LabeledDsList, MaxSize,
        PadElems, PadSizes, PrimaryDsInfo, StageDims, Symbolic, VolumeLimit, WkSlice, WkSliceId,
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
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(core(0), vec![]),
            layout_dims: BTreeMap::new(),
            core_stage: one_dim_stage(),
            labeled_ds: LabeledDsList::new(LabeledDs::new(DsType::Output, vec![]), vec![]),
        }
    }

    fn one_dim_stage() -> FilledDims {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::In, Extent(1));
        filled(dims)
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
        let sdsc = SuperDsc::new(DscList::new(plain_dsc(), vec![]), BTreeMap::new());
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

    /// e005 — the reference's own worked example (`dsc/dims.cpp:715-727`): `abc` limited to 2048 with
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
        });
        let mut chunk_params = filled(StageDims {
            extents: BTreeMap::from([
                (PrimaryDim::In, Extent(32)),
                (PrimaryDim::Out, Extent(64)),
                (PrimaryDim::Ij, Extent(64)),
            ]),
            padding: BTreeMap::new(),
            symbolic: Symbolic::default(),
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
            BTreeMap::from([
                (core(0), slice(0)),
                (core(1), slice(0)),
                (core(2), slice(1)),
                (core(3), slice(0)),
            ]),
        );
        let group = DscGroup::new(&dsc, vec![]);
        assert_eq!(
            labeled_ds_wk_slice_multicast_degree(&sdsc, lds, &group),
            Some(MulticastDegree(2))
        );
        assert_eq!(
            labeled_ds_wk_slice_multicast_degree(&sdsc, LdsIdx(1), &group),
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
    BlockNode { name }
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
        PrimaryDim::J => dsc.core_stage.dims().extent(dim),
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
        PrimaryDim::J => dsc.core_stage.dims().extent(dim),
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
    let core_param = dsc.core_stage.dims().extent(dim);
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
    let core = dsc.core_stage.dims();
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
        PrimaryDim::J => dsc.core_stage.dims().extent(dim),
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
        CoreIdsUsed, CoreletsUsed, DimPadding, LabeledDsList, PadElems, PadSizes, PrimaryDsInfo,
        StageDims,
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
        DesignSpaceConfig {
            corelets_used: CoreletsUsed::ONE,
            corelet_shares: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            core_ids_used: CoreIdsUsed::new(Core::checked(0).expect("core 0"), vec![]),
            layout_dims: BTreeMap::new(),
            core_stage,
            labeled_ds: LabeledDsList::new(
                LabeledDs::new(*first, vec![]),
                rest.iter().map(|ds| LabeledDs::new(*ds, vec![])).collect(),
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
