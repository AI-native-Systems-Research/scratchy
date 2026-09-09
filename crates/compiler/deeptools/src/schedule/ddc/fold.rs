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

//! `ddc/ddc_fold.cpp` — 36 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e078_dbgPrint` | 078 | 0 | 25 | — | `ddc/ddc_fold.cpp:20` |
//! | `e079_dbgPrint` | 079 | 0 | 16 | — | `ddc/ddc_fold.cpp:46` |
//! | `e080_allDimsCovered` | 080 | 0 | 9 | — | `ddc/ddc_fold.cpp:75` |
//! | `e081_buildFoldForBroadcastDim` | 081 | 0 | 19 | — | `ddc/ddc_fold.cpp:87` |
//! | `e082_getCompRowId` | 082 | 0 | 7 | — | `ddc/ddc_fold.cpp:110` |
//! | `e083_getComponent` | 083 | 0 | 17 | — | `ddc/ddc_fold.cpp:118` |
//! | `e084_getLayoutDimsFromNode` | 084 | 0 | 51 | — | `ddc/ddc_fold.cpp:136` |
//! | `e085_needToConsiderRowBundling` | 085 | 0 | 7 | — | `ddc/ddc_fold.cpp:190` |
//! | `e086_isAllocateIncoming` | 086 | 0 | 28 | — | `ddc/ddc_fold.cpp:201` |
//! | `e087_findCommonAncestor` | 087 | 0 | 32 | — | `ddc/ddc_fold.cpp:232` |
//! | `e088_orderDescendants` | 088 | 0 | 25 | — | `ddc/ddc_fold.cpp:267` |
//! | `e089_constructAllocElemArrLayout` | 089 | 0 | 87 | — | `ddc/ddc_fold.cpp:296` |
//! | `e090_combineContigousLevels` | 090 | 0 | 34 | — | `ddc/ddc_fold.cpp:395` |
//! | `e091_matchDataStream` | 091 | 0 | 42 | — | `ddc/ddc_fold.cpp:430` |
//! | `e092_getNumElementsInPTSlice` | 092 | 0 | 18 | `Ddc` | `ddc/ddc_fold.cpp:1519` |
//! | `e093_buildSpatialFold` | 093 | 0 | 41 | `Ddc` | `ddc/ddc_fold.cpp:2109` |
//! | `e094_getDefaultRowSplitFold` | 094 | 0 | 6 | — | `ddc/ddc_fold.cpp:2154` |
//! | `e095_gatherFoldParams` | 095 | 0 | 12 | `Ddc` | `ddc/ddc_fold.cpp:2224` |
//! | `e233_dbgPrint` | 233 | 1 | 11 | — | `ddc/ddc_fold.cpp:63` |
//! | `e234_scaleUpCoord` | 234 | 1 | 118 | `Ddc` | `ddc/ddc_fold.cpp:476` |
//! | `e235_scaleDownCoord` | 235 | 1 | 87 | `Ddc` | `ddc/ddc_fold.cpp:598` |
//! | `e236_needNonRowBundling` | 236 | 1 | 50 | `Ddc` | `ddc/ddc_fold.cpp:688` |
//! | `e237_sameCoordinateRange` | 237 | 1 | 252 | `Ddc` | `ddc/ddc_fold.cpp:1260` |
//! | `e238_getRelatedComputeCoord` | 238 | 1 | 129 | `Ddc` | `ddc/ddc_fold.cpp:1976` |
//! | `e239_computeParamsForRowSplitFold` | 239 | 1 | 60 | `Ddc` | `ddc/ddc_fold.cpp:2161` |
//! | `e240_buildFoldForExternalAllocation` | 240 | 1 | 156 | `Ddc` | `ddc/ddc_fold.cpp:2238` |
//! | `e241_relateLoopsToAllocElemArr` | 241 | 1 | 166 | `Ddc` | `ddc/ddc_fold.cpp:2872` |
//! | `e297_gatherRelatedPTRowsBase` | 297 | 2 | 217 | `Ddc` | `ddc/ddc_fold.cpp:740` |
//! | `e298_buildFoldFromAllocation` | 298 | 2 | 245 | `Ddc` | `ddc/ddc_fold.cpp:3043` |
//! | `e299_buildFoldFromNonAllocRef` | 299 | 2 | 355 | `Ddc` | `ddc/ddc_fold.cpp:3294` |
//! | `e337_gatherRelatedPTRows` | 337 | 3 | 298 | `Ddc` | `ddc/ddc_fold.cpp:959` |
//! | `e356_buildFoldForAllocation` | 356 | 4 | 473 | `Ddc` | `ddc/ddc_fold.cpp:2395` |
//! | `e357_buildFoldForCompute` | 357 | 4 | 773 | `Ddc` | `ddc/ddc_fold.cpp:3656` |
//! | `e358_buildFoldForTransfer` | 358 | 4 | 289 | `Ddc` | `ddc/ddc_fold.cpp:4433` |
//! | `e370_buildAndPropagateFold` | 370 | 5 | 350 | `Ddc` | `ddc/ddc_fold.cpp:1625` |
//! | `e375_coordinateCapture` | 375 | 6 | 86 | `Ddc` | `ddc/ddc_fold.cpp:1538` |


// crustify:todo: e078_dbgPrint
//   authority : ddc/ddc_fold.cpp:20  (25 body lines, level 0)
//   original  : void dbgPrint(const dsc2::ComputeNode *computeNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:177-202

// crustify:todo: e079_dbgPrint
//   authority : ddc/ddc_fold.cpp:46  (16 body lines, level 0)
//   original  : void dbgPrint(const dsc2::TransferNode *transferNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:211-227

// crustify:todo: e080_allDimsCovered
//   authority : ddc/ddc_fold.cpp:75  (9 body lines, level 0)
//   original  : bool allDimsCovered(const DesignSpaceConfig *currDsc, const dsc2::CoordinateType<CoordinateBaseType> &coord, const int ldsIdx)
//   extract   : crustify-ddc/cpp/ddc.cpp:236-247

// crustify:todo: e081_buildFoldForBroadcastDim
//   authority : ddc/ddc_fold.cpp:87  (19 body lines, level 0)
//   original  : void buildFoldForBroadcastDim(const DesignSpaceConfig *currDsc, dsc2::CoordinateType<CoordinateBaseType> &coord, const LabeledDsInfo &lds, PrimaryDimTypes dim, int scale)
//   extract   : crustify-ddc/cpp/ddc.cpp:256-278

// crustify:todo: e082_getCompRowId
//   authority : ddc/ddc_fold.cpp:110  (7 body lines, level 0)
//   original  : int getCompRowId(SenComponents comp)
//   extract   : crustify-ddc/cpp/ddc.cpp:287-294

// crustify:todo: e083_getComponent
//   authority : ddc/ddc_fold.cpp:118  (17 body lines, level 0)
//   original  : SenComponents getComponent(const dsc2::ScheduleNode *node, bool getSrc = true)
//   extract   : crustify-ddc/cpp/ddc.cpp:303-320

// crustify:todo: e084_getLayoutDimsFromNode
//   authority : ddc/ddc_fold.cpp:136  (51 body lines, level 0)
//   original  : std::vector<PrimaryDimTypes> getLayoutDimsFromNode( const DesignSpaceConfig *currDsc, const dsc2::ScheduleNode *node, int inputPos = -1, int outputPos = -1)
//   extract   : crustify-ddc/cpp/ddc.cpp:329-382

// crustify:todo: e085_needToConsiderRowBundling
//   authority : ddc/ddc_fold.cpp:190  (7 body lines, level 0)
//   original  : bool needToConsiderRowBundling( const dsc2::DataStage &coreDs, const dsc2::CoordinateType<CoordinateBaseType> &refCoord, const std::vector<PrimaryDimTypes> &workingDims)
//   extract   : crustify-ddc/cpp/ddc.cpp:391-401

// crustify:todo: e086_isAllocateIncoming
//   authority : ddc/ddc_fold.cpp:201  (28 body lines, level 0)
//   original  : bool isAllocateIncoming(const DesignSpaceConfig *currDsc, const dsc2::AllocateNode *allocNode, const dsc2::ScheduleNode *node)
//   extract   : crustify-ddc/cpp/ddc.cpp:410-440

// crustify:todo: e087_findCommonAncestor
//   authority : ddc/ddc_fold.cpp:232  (32 body lines, level 0)
//   original  : const dsc2::ScheduleNode *findCommonAncestor( const std::vector<dsc2::ScheduleNode *> &nodeList, std::vector<dsc2::ScheduleNode::NodeType> ancestorType)
//   extract   : crustify-ddc/cpp/ddc.cpp:449-483

// crustify:todo: e088_orderDescendants
//   authority : ddc/ddc_fold.cpp:267  (25 body lines, level 0)
//   original  : void orderDescendants( const dsc2::ScheduleNode *commonAncestor, const std::vector<const dsc2::ScheduleNode *> &nodeList, std::vector<const dsc2::ScheduleNode *> &orderedNodeList)
//   extract   : crustify-ddc/cpp/ddc.cpp:492-520

// crustify:todo: e089_constructAllocElemArrLayout
//   authority : ddc/ddc_fold.cpp:296  (87 body lines, level 0)
//   original  : void constructAllocElemArrLayout( const DesignSpaceConfig *currDsc, const DesignSpaceConfigGlobal &dscGlobal, const dsc2::AllocateNode *allocNode, PrimaryDimTypes coordDim, std::vector<dsc2::FoldParamInfoType> &foldParams, int &numElemArrFoldsOfAllocNode, dsc2::LoopDistributionParamPerNodeType &loop
//   extract   : crustify-ddc/cpp/ddc.cpp:529-621

// crustify:todo: e090_combineContigousLevels
//   authority : ddc/ddc_fold.cpp:395  (34 body lines, level 0)
//   original  : void combineContigousLevels(std::vector<dsc2::FoldParamInfoType> &foldParams, int start = 0, int end = -1)
//   extract   : crustify-ddc/cpp/ddc.cpp:630-665

// crustify:todo: e091_matchDataStream
//   authority : ddc/ddc_fold.cpp:430  (42 body lines, level 0)
//   original  : bool matchDataStream(const DesignSpaceConfig *currDsc, const dsc2::CoordPropInfoType &coordPropInfo, const dsc2::DataInfo &di, const SenComponents unit, bool checkRefForAllocate)
//   extract   : crustify-ddc/cpp/ddc.cpp:674-719

// crustify:todo: e092_getNumElementsInPTSlice
//   authority : ddc/ddc_fold.cpp:1519  (18 body lines, level 0)
//   class     : Ddc
//   original  : int Ddc::getNumElementsInPTSlice(int ldsIdx, PrimaryDimTypes dim)
//   extract   : crustify-ddc/cpp/ddc.cpp:729-747

// crustify:todo: e093_buildSpatialFold
//   authority : ddc/ddc_fold.cpp:2109  (41 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::buildSpatialFold( dsc2::ScheduleNode *node, const PrimaryDimTypes &currDim, const PadType &currPadType, dsc2::CoordinateType<CoordinateBaseType> &nodeCoordinates)
//   extract   : crustify-ddc/cpp/ddc.cpp:757-801

// crustify:todo: e094_getDefaultRowSplitFold
//   authority : ddc/ddc_fold.cpp:2154  (6 body lines, level 0)
//   original  : void getDefaultRowSplitFold(dsc2::FoldParamInfoType &resultFoldParams)
//   extract   : crustify-ddc/cpp/ddc.cpp:810-816

// crustify:todo: e095_gatherFoldParams
//   authority : ddc/ddc_fold.cpp:2224  (12 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::gatherFoldParams(const FoldManager<CoordinateBaseType> &cfm, std::vector<dsc2::FoldParamInfoType> &foldParams)
//   extract   : crustify-ddc/cpp/ddc.cpp:826-839

// crustify:todo: e233_dbgPrint
//   authority : ddc/ddc_fold.cpp:63  (11 body lines, level 1)
//   original  : void dbgPrint(const dsc2::ScheduleNode *node)
//   extract   : crustify-ddc/cpp/ddc.cpp:3233-3244
//   calls     : e078_dbgPrint, e079_dbgPrint

// crustify:todo: e234_scaleUpCoord
//   authority : ddc/ddc_fold.cpp:476  (118 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::scaleUpCoord(dsc2::ScheduleNode *lhsNode, dsc2::CoordinateType<CoordinateBaseType> &lhsCoord, dsc2::CoordinateType<CoordinateBaseType> &rhsCoord, const int ldsIdx, const SenComponents comp)
//   extract   : crustify-ddc/cpp/ddc.cpp:3254-3375
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e095_gatherFoldParams

// crustify:todo: e235_scaleDownCoord
//   authority : ddc/ddc_fold.cpp:598  (87 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::scaleDownCoord(dsc2::CoordinateType<CoordinateBaseType> &lhsCoord, dsc2::CoordinateType<CoordinateBaseType> &rhsCoord, const int ldsIdx)
//   extract   : crustify-ddc/cpp/ddc.cpp:3385-3474
//   calls     : e095_gatherFoldParams

// crustify:todo: e236_needNonRowBundling
//   authority : ddc/ddc_fold.cpp:688  (50 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::needNonRowBundling(std::string &dataConnect, bool checkProducers) const
//   extract   : crustify-ddc/cpp/ddc.cpp:3484-3535
//   calls     : e082_getCompRowId, e083_getComponent, e087_findCommonAncestor

// crustify:todo: e237_sameCoordinateRange
//   authority : ddc/ddc_fold.cpp:1260  (252 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::sameCoordinateRange( const dsc2::ScheduleNode *lhsNode, const dsc2::CoordinateType<CoordinateBaseType> &lhs, const dsc2::ScheduleNode *rhsNode, const dsc2::CoordinateType<CoordinateBaseType> &rhs, SenComponents comp /* = SenComponents::ALL*/, int ldsIdx /*= -1*/, bool commonDimsOnly /* = f
//   extract   : crustify-ddc/cpp/ddc.cpp:3545-3803
//   calls     : e090_combineContigousLevels, e095_gatherFoldParams, e104_clear

// crustify:todo: e238_getRelatedComputeCoord
//   authority : ddc/ddc_fold.cpp:1976  (129 body lines, level 1)
//   class     : Ddc
//   original  : dsc2::CoordinateType<CoordinateBaseType> &Ddc::getRelatedComputeCoord( dsc2::ComputeNode *computeNode, dsc2::CoordPropInfoType &refFoldInfo, std::vector<int> &constructedInputCoords, std::vector<int> &constructedOutputCoords, int &selectedLdsIdx)
//   extract   : crustify-ddc/cpp/ddc.cpp:3813-3945
//   calls     : e104_clear

// crustify:todo: e239_computeParamsForRowSplitFold
//   authority : ddc/ddc_fold.cpp:2161  (60 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::computeParamsForRowSplitFold( const PrimaryDimTypes &currDim, dsc2::FoldParamInfoType &resultFoldParams, RowGroupInfo &rowGroup)
//   extract   : crustify-ddc/cpp/ddc.cpp:3955-4017
//   calls     : e094_getDefaultRowSplitFold

// crustify:todo: e240_buildFoldForExternalAllocation
//   authority : ddc/ddc_fold.cpp:2238  (156 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::buildFoldForExternalAllocation(dsc2::AllocateNode *allocNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:4027-4183
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e081_buildFoldForBroadcastDim, e093_buildSpatialFold

// crustify:todo: e241_relateLoopsToAllocElemArr
//   authority : ddc/ddc_fold.cpp:2872  (166 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::relateLoopsToAllocElemArr( dsc2::CoordPropInfoType &coordPropInfo, const PrimaryDimTypes dim, const dsc2::CoordinateType<CoordinateBaseType> &coordinate, const dsc2::VectorOfLoopAndDim &refLoopChain, dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution)
//   extract   : crustify-ddc/cpp/ddc.cpp:4193-4363
//   calls     : e091_matchDataStream, e095_gatherFoldParams

// crustify:todo: e297_gatherRelatedPTRowsBase
//   authority : ddc/ddc_fold.cpp:740  (217 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::gatherRelatedPTRowsBase(const dsc2::CoordPropInfoType &coordPropInfo, RowGroupInfo &rowGroup)
//   extract   : crustify-ddc/cpp/ddc.cpp:8139-8357
//   calls     : e082_getCompRowId, e084_getLayoutDimsFromNode, e091_matchDataStream, e104_clear, e237_sameCoordinateRange

// crustify:todo: e298_buildFoldFromAllocation
//   authority : ddc/ddc_fold.cpp:3043  (245 body lines, level 2)
//   class     : Ddc
//   original  : void Ddc::buildFoldFromAllocation( dsc2::CoordPropInfoType &coordPropInfo, dsc2::ScheduleNode *node, dsc2::CoordinateType<CoordinateBaseType> &coordinate, SenComponents sizeRefComp, SenComponents propRefComp, dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution, RowGroupInfo &refRowGr
//   extract   : crustify-ddc/cpp/ddc.cpp:8367-8617
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e078_dbgPrint, e079_dbgPrint, e080_allDimsCovered, e095_gatherFoldParams, e233_dbgPrint, e239_computeParamsForRowSplitFold

// crustify:todo: e299_buildFoldFromNonAllocRef
//   authority : ddc/ddc_fold.cpp:3294  (355 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::buildFoldFromNonAllocRef( dsc2::CoordPropInfoType &coordPropInfo, const int refLdsIdx, const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate, dsc2::CoordinateType<CoordinateBaseType> &coordinate, const SenComponents sizeRefComp, SenComponents propRefComp, dsc2::LoopDistributionPara
//   extract   : crustify-ddc/cpp/ddc.cpp:8627-8988
//   calls     : e080_allDimsCovered, e094_getDefaultRowSplitFold, e095_gatherFoldParams, e104_clear, e239_computeParamsForRowSplitFold

// crustify:todo: e337_gatherRelatedPTRows
//   authority : ddc/ddc_fold.cpp:959  (298 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::gatherRelatedPTRows( RowGroupInfo &refRowGroup, const dsc2::CoordPropInfoType &coordPropInfo, const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate)
//   extract   : crustify-ddc/cpp/ddc.cpp:11783-12083
//   calls     : e076_print, e082_getCompRowId, e091_matchDataStream, e102_print, e297_gatherRelatedPTRowsBase

// crustify:todo: e356_buildFoldForAllocation
//   authority : ddc/ddc_fold.cpp:2395  (473 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::buildFoldForAllocation( dsc2::CoordPropInfoType &coordPropInfo, const dsc2::CoordinateType<CoordinateBaseType> &inputRefCoord, dsc2::AllocateNode *allocNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:12521-12997
//   calls     : e062_getEnclosingLoopsAndRelatedDims, e074_retry, e078_dbgPrint, e079_dbgPrint, e080_allDimsCovered, e085_needToConsiderRowBundling, e089_constructAllocElemArrLayout, e092_getNumElementsInPTSlice, e094_getDefaultRowSplitFold, e095_gatherFoldParams, e233_dbgPrint, e234_scaleUpCoord, e235_scaleDownCoord, e237_sameCoordinateRange …

// crustify:todo: e357_buildFoldForCompute
//   authority : ddc/ddc_fold.cpp:3656  (773 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::buildFoldForCompute(dsc2::ComputeNode *computeNode, dsc2::CoordPropInfoType &coordPropInfo, std::vector<int> &constructedInputCoords, std::vector<int> &constructedOutputCoords)
//   extract   : crustify-ddc/cpp/ddc.cpp:13007-13783
//   calls     : e074_retry, e078_dbgPrint, e079_dbgPrint, e080_allDimsCovered, e081_buildFoldForBroadcastDim, e082_getCompRowId, e085_needToConsiderRowBundling, e104_clear, e233_dbgPrint, e234_scaleUpCoord, e235_scaleDownCoord, e236_needNonRowBundling, e238_getRelatedComputeCoord, e298_buildFoldFromAllocation …

// crustify:todo: e358_buildFoldForTransfer
//   authority : ddc/ddc_fold.cpp:4433  (289 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::buildFoldForTransfer(dsc2::TransferNode *transferNode, dsc2::CoordPropInfoType &coordPropInfo)
//   extract   : crustify-ddc/cpp/ddc.cpp:13793-14083
//   calls     : e074_retry, e078_dbgPrint, e079_dbgPrint, e082_getCompRowId, e085_needToConsiderRowBundling, e122_isNodeRelatedToComps, e178_getAccessPattern, e233_dbgPrint, e238_getRelatedComputeCoord, e298_buildFoldFromAllocation, e299_buildFoldFromNonAllocRef, e337_gatherRelatedPTRows

// crustify:todo: e370_buildAndPropagateFold
//   authority : ddc/ddc_fold.cpp:1625  (350 body lines, level 5)
//   class     : Ddc
//   original  : void Ddc::buildAndPropagateFold()
//   extract   : crustify-ddc/cpp/ddc.cpp:14384-14734
//   calls     : e073_addPropInfo, e075_getCurrItem, e078_dbgPrint, e079_dbgPrint, e082_getCompRowId, e086_isAllocateIncoming, e230_addPropInfo, e232_reset, e233_dbgPrint, e238_getRelatedComputeCoord, e240_buildFoldForExternalAllocation, e356_buildFoldForAllocation, e357_buildFoldForCompute, e358_buildFoldForTransfer

// crustify:todo: e375_coordinateCapture
//   authority : ddc/ddc_fold.cpp:1538  (86 body lines, level 6)
//   class     : Ddc
//   original  : void Ddc::coordinateCapture()
//   extract   : crustify-ddc/cpp/ddc.cpp:14875-14961
//   calls     : e076_print, e102_print, e370_buildAndPropagateFold

