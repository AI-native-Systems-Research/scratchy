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

//! `ddc/ddc_transformation_util.cpp` — 32 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e110_constructAllocation` | 110 | 0 | 50 | `Ddc` | `ddc/ddc_transformation_util.cpp:20` |
//! | `e111_reduceUsersOrDeleteAllocationAndMetadata` | 111 | 0 | 41 | `Ddc` | `ddc/ddc_transformation_util.cpp:73` |
//! | `e112_constructDatastage` | 112 | 0 | 8 | `Ddc` | `ddc/ddc_transformation_util.cpp:117` |
//! | `e113_constructDatastage` | 113 | 0 | 11 | `Ddc` | `ddc/ddc_transformation_util.cpp:126` |
//! | `e114_constructLoopNode` | 114 | 0 | 18 | `Ddc` | `ddc/ddc_transformation_util.cpp:138` |
//! | `e115_collectLoopReferences` | 115 | 0 | 14 | `Ddc` | `ddc/ddc_transformation_util.cpp:318` |
//! | `e116_getPaddingPerDim` | 116 | 0 | 49 | `Ddc` | `ddc/ddc_transformation_util.cpp:709` |
//! | `e117_getNodeDescription` | 117 | 0 | 25 | `Ddc` | `ddc/ddc_transformation_util.cpp:1094` |
//! | `e118_cloneForPeSfpWorkSplit` | 118 | 0 | 114 | `Ddc` | `ddc/ddc_transformation_util.cpp:1306` |
//! | `e119_cloneForPeSfpWorkSplit` | 119 | 0 | 115 | `Ddc` | `ddc/ddc_transformation_util.cpp:1422` |
//! | `e120_storageOrDatastreamIsExternal` | 120 | 0 | 32 | `Ddc` | `ddc/ddc_transformation_util.cpp:1652` |
//! | `e121_relatedToExternalNodes` | 121 | 0 | 8 | `Ddc` | `ddc/ddc_transformation_util.cpp:1748` |
//! | `e122_isNodeRelatedToComps` | 122 | 0 | 30 | `Ddc` | `ddc/ddc_transformation_util.cpp:1778` |
//! | `e123_updateNodesWithNewLds` | 123 | 0 | 57 | `Ddc` | `ddc/ddc_transformation_util.cpp:1919` |
//! | `e247_splitLoopBandOnDim` | 247 | 1 | 123 | `Ddc` | `ddc/ddc_transformation_util.cpp:159` |
//! | `e248_splitLoopBandOnDatastage` | 248 | 1 | 30 | `Ddc` | `ddc/ddc_transformation_util.cpp:287` |
//! | `e249_moveTransferNode` | 249 | 1 | 372 | `Ddc` | `ddc/ddc_transformation_util.cpp:335` |
//! | `e250_convertResultFromFIFOtoReg` | 250 | 1 | 148 | `Ddc` | `ddc/ddc_transformation_util.cpp:760` |
//! | `e251_unrollTransfer` | 251 | 1 | 72 | `Ddc` | `ddc/ddc_transformation_util.cpp:1120` |
//! | `e252_unrollTransferForSymbolicDims` | 252 | 1 | 110 | `Ddc` | `ddc/ddc_transformation_util.cpp:1193` |
//! | `e253_srcRelatedToExternalNodes` | 253 | 1 | 4 | `Ddc` | `ddc/ddc_transformation_util.cpp:1687` |
//! | `e254_destRelatedToExternalNodes` | 254 | 1 | 5 | `Ddc` | `ddc/ddc_transformation_util.cpp:1693` |
//! | `e255_inputRelatedToExternalNodes` | 255 | 1 | 11 | `Ddc` | `ddc/ddc_transformation_util.cpp:1717` |
//! | `e256_outputRelatedToExternalNodes` | 256 | 1 | 11 | `Ddc` | `ddc/ddc_transformation_util.cpp:1730` |
//! | `e257_addNewLds` | 257 | 1 | 107 | `Ddc` | `ddc/ddc_transformation_util.cpp:1811` |
//! | `e304_cloneForPeSfpWorkSplit` | 304 | 2 | 113 | `Ddc` | `ddc/ddc_transformation_util.cpp:1538` |
//! | `e305_destRelatedToExternalNodes` | 305 | 2 | 10 | `Ddc` | `ddc/ddc_transformation_util.cpp:1700` |
//! | `e306_relatedToExternalNodes` | 306 | 2 | 4 | `Ddc` | `ddc/ddc_transformation_util.cpp:1743` |
//! | `e339_convertResultToSkipReg` | 339 | 3 | 110 | `Ddc` | `ddc/ddc_transformation_util.cpp:909` |
//! | `e340_insertComputeBetweenTransferAndReg` | 340 | 3 | 69 | `Ddc` | `ddc/ddc_transformation_util.cpp:1021` |
//! | `e341_relatedToExternalNodes` | 341 | 3 | 4 | `Ddc` | `ddc/ddc_transformation_util.cpp:1712` |
//! | `e361_relatedToExternalNodes` | 361 | 4 | 20 | `Ddc` | `ddc/ddc_transformation_util.cpp:1757` |


// crustify:todo: e110_constructAllocation
//   authority : ddc/ddc_transformation_util.cpp:20  (50 body lines, level 0)
//   class     : Ddc
//   original  : dsc2::AllocateNode *Ddc::constructAllocation( const dsc2::DataInfo &di, const SenComponents storage, const PaddingFormType &paddingPerDim, const dsc2::ScheduleNode *userNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:1218-1270

// crustify:todo: e111_reduceUsersOrDeleteAllocationAndMetadata
//   authority : ddc/ddc_transformation_util.cpp:73  (41 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::reduceUsersOrDeleteAllocationAndMetadata( const dsc2::DataInfo &di, const SenComponents storage, const dsc2::ScheduleNode *userNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:1280-1323

// crustify:todo: e112_constructDatastage
//   authority : ddc/ddc_transformation_util.cpp:117  (8 body lines, level 0)
//   class     : Ddc
//   original  : int Ddc::constructDatastage()
//   extract   : crustify-ddc/cpp/ddc.cpp:1333-1341

// crustify:todo: e113_constructDatastage
//   authority : ddc/ddc_transformation_util.cpp:126  (11 body lines, level 0)
//   class     : Ddc
//   original  : int Ddc::constructDatastage(dsc2::DataStage &refDataStage)
//   extract   : crustify-ddc/cpp/ddc.cpp:1351-1362

// crustify:todo: e114_constructLoopNode
//   authority : ddc/ddc_transformation_util.cpp:138  (18 body lines, level 0)
//   class     : Ddc
//   original  : dsc2::LoopNode *Ddc::constructLoopNode(int numId, int denId, std::vector<PrimaryDimAndKind> dims, dsc2::ScheduleNode *baseNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:1372-1392

// crustify:todo: e115_collectLoopReferences
//   authority : ddc/ddc_transformation_util.cpp:318  (14 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::collectLoopReferences( const dsc2::ScheduleNode *node, std::set<const dsc2::LoopNode *> &referencedLoops)
//   extract   : crustify-ddc/cpp/ddc.cpp:1402-1418

// crustify:todo: e116_getPaddingPerDim
//   authority : ddc/ddc_transformation_util.cpp:709  (49 body lines, level 0)
//   class     : Ddc
//   original  : PaddingFormType Ddc::getPaddingPerDim(const dsc2::TransferNode *transferNode, bool forSrc) const
//   extract   : crustify-ddc/cpp/ddc.cpp:1428-1478

// crustify:todo: e117_getNodeDescription
//   authority : ddc/ddc_transformation_util.cpp:1094  (25 body lines, level 0)
//   class     : Ddc
//   original  : std::string Ddc::getNodeDescription(const dsc2::ScheduleNode *node) const
//   extract   : crustify-ddc/cpp/ddc.cpp:1488-1513

// crustify:todo: e118_cloneForPeSfpWorkSplit
//   authority : ddc/ddc_transformation_util.cpp:1306  (114 body lines, level 0)
//   class     : Ddc
//   original  : dsc2::AllocateNode *Ddc::cloneForPeSfpWorkSplit( dsc2::AllocateNode *node, bool skipMetadataUpdate /* = false */)
//   extract   : crustify-ddc/cpp/ddc.cpp:1523-1638

// crustify:todo: e119_cloneForPeSfpWorkSplit
//   authority : ddc/ddc_transformation_util.cpp:1422  (115 body lines, level 0)
//   class     : Ddc
//   original  : dsc2::ComputeNode *Ddc::cloneForPeSfpWorkSplit(dsc2::ComputeNode *node)
//   extract   : crustify-ddc/cpp/ddc.cpp:1648-1763

// crustify:todo: e120_storageOrDatastreamIsExternal
//   authority : ddc/ddc_transformation_util.cpp:1652  (32 body lines, level 0)
//   class     : Ddc
//   original  : bool Ddc::storageOrDatastreamIsExternal(const dsc2::DataInfo &dataInfo, SenComponents storage, bool isIncoming) const
//   extract   : crustify-ddc/cpp/ddc.cpp:1773-1807

// crustify:todo: e121_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1748  (8 body lines, level 0)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::SyncNode *node) const
//   extract   : crustify-ddc/cpp/ddc.cpp:1817-1825

// crustify:todo: e122_isNodeRelatedToComps
//   authority : ddc/ddc_transformation_util.cpp:1778  (30 body lines, level 0)
//   class     : Ddc
//   original  : bool Ddc::isNodeRelatedToComps(const dsc2::ScheduleNode *node, const std::vector<SenComponents> &comps, SenComponents &nodeComp)
//   extract   : crustify-ddc/cpp/ddc.cpp:1835-1867

// crustify:todo: e123_updateNodesWithNewLds
//   authority : ddc/ddc_transformation_util.cpp:1919  (57 body lines, level 0)
//   class     : Ddc
//   original  : void Ddc::updateNodesWithNewLds(int newLdsIdx, int oldLdsIdx, dsc2::ScheduleNode *startNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:1877-1935

// crustify:todo: e247_splitLoopBandOnDim
//   authority : ddc/ddc_transformation_util.cpp:159  (123 body lines, level 1)
//   class     : Ddc
//   original  : dsc2::LoopNode *Ddc::splitLoopBandOnDim( dsc2::LoopNode *baseLoop, const std::vector<std::vector<PrimaryDimAndKind>> &inputSplitDimSetsOuterToInner, bool unspecifiedDimsInnermost)
//   extract   : crustify-ddc/cpp/ddc.cpp:5086-5213
//   calls     : e114_constructLoopNode

// crustify:todo: e248_splitLoopBandOnDatastage
//   authority : ddc/ddc_transformation_util.cpp:287  (30 body lines, level 1)
//   class     : Ddc
//   original  : dsc2::LoopNode *Ddc::splitLoopBandOnDatastage(dsc2::LoopNode *baseLoop)
//   extract   : crustify-ddc/cpp/ddc.cpp:5223-5253
//   calls     : e112_constructDatastage, e113_constructDatastage, e114_constructLoopNode

// crustify:todo: e249_moveTransferNode
//   authority : ddc/ddc_transformation_util.cpp:335  (372 body lines, level 1)
//   class     : Ddc
//   original  : void Ddc::moveTransferNode(dsc2::TransferNode *transferNode, dsc2::LoopNode *newParentLoop)
//   extract   : crustify-ddc/cpp/ddc.cpp:5263-5636
//   calls     : e104_clear

// crustify:todo: e250_convertResultFromFIFOtoReg
//   authority : ddc/ddc_transformation_util.cpp:760  (148 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::convertResultFromFIFOtoReg(dsc2::TransferNode *transferNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:5646-5794
//   calls     : e110_constructAllocation, e116_getPaddingPerDim

// crustify:todo: e251_unrollTransfer
//   authority : ddc/ddc_transformation_util.cpp:1120  (72 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::unrollTransfer(dsc2::TransferNode *transferNode)
//   extract   : crustify-ddc/cpp/ddc.cpp:5804-5876
//   calls     : e104_clear, e112_constructDatastage, e113_constructDatastage, e114_constructLoopNode

// crustify:todo: e252_unrollTransferForSymbolicDims
//   authority : ddc/ddc_transformation_util.cpp:1193  (110 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::unrollTransferForSymbolicDims( dsc2::TransferNode *transferNode, const std::map<PrimaryDimTypes, SymbolicDimInfo> &symbolicDims)
//   extract   : crustify-ddc/cpp/ddc.cpp:5886-5998
//   calls     : e112_constructDatastage, e113_constructDatastage, e114_constructLoopNode

// crustify:todo: e253_srcRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1687  (4 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::srcRelatedToExternalNodes( const dsc2::TransferNode *transferNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6008-6013
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e254_destRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1693  (5 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::destRelatedToExternalNodes(const dsc2::TransferNode *transferNode, int dstIndex) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6023-6029
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e255_inputRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1717  (11 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::inputRelatedToExternalNodes( const dsc2::ComputeNode *computeNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6039-6051
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e256_outputRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1730  (11 body lines, level 1)
//   class     : Ddc
//   original  : bool Ddc::outputRelatedToExternalNodes( const dsc2::ComputeNode *computeNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:6061-6073
//   calls     : e120_storageOrDatastreamIsExternal

// crustify:todo: e257_addNewLds
//   authority : ddc/ddc_transformation_util.cpp:1811  (107 body lines, level 1)
//   class     : Ddc
//   original  : int Ddc::addNewLds(LabeledDsInfo *refLds)
//   extract   : crustify-ddc/cpp/ddc.cpp:6083-6190
//   calls     : e104_clear

// crustify:todo: e304_cloneForPeSfpWorkSplit
//   authority : ddc/ddc_transformation_util.cpp:1538  (113 body lines, level 2)
//   class     : Ddc
//   original  : dsc2::TransferNode *Ddc::cloneForPeSfpWorkSplit(dsc2::TransferNode *node)
//   extract   : crustify-ddc/cpp/ddc.cpp:9878-9991
//   calls     : e251_unrollTransfer

// crustify:todo: e305_destRelatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1700  (10 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::destRelatedToExternalNodes( const dsc2::TransferNode *transferNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:10001-10012
//   calls     : e254_destRelatedToExternalNodes

// crustify:todo: e306_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1743  (4 body lines, level 2)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::ComputeNode *computeNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:10022-10026
//   calls     : e255_inputRelatedToExternalNodes, e256_outputRelatedToExternalNodes

// crustify:todo: e339_convertResultToSkipReg
//   authority : ddc/ddc_transformation_util.cpp:909  (110 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::convertResultToSkipReg(dsc2::TransferNode *transferNode, int destIndex, bool useLatch)
//   extract   : crustify-ddc/cpp/ddc.cpp:12139-12250
//   calls     : e111_reduceUsersOrDeleteAllocationAndMetadata, e254_destRelatedToExternalNodes, e305_destRelatedToExternalNodes

// crustify:todo: e340_insertComputeBetweenTransferAndReg
//   authority : ddc/ddc_transformation_util.cpp:1021  (69 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::insertComputeBetweenTransferAndReg(dsc2::TransferNode *transferNode, int transferDestIndex, dsc2::ComputeNode *computeNode, int computeInputIndex)
//   extract   : crustify-ddc/cpp/ddc.cpp:12260-12332
//   calls     : e254_destRelatedToExternalNodes, e305_destRelatedToExternalNodes

// crustify:todo: e341_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1712  (4 body lines, level 3)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::TransferNode *transferNode) const
//   extract   : crustify-ddc/cpp/ddc.cpp:12342-12346
//   calls     : e253_srcRelatedToExternalNodes, e254_destRelatedToExternalNodes, e305_destRelatedToExternalNodes

// crustify:todo: e361_relatedToExternalNodes
//   authority : ddc/ddc_transformation_util.cpp:1757  (20 body lines, level 4)
//   class     : Ddc
//   original  : bool Ddc::relatedToExternalNodes(const dsc2::BlockNode *root) const
//   extract   : crustify-ddc/cpp/ddc.cpp:14282-14302
//   calls     : e121_relatedToExternalNodes, e306_relatedToExternalNodes, e341_relatedToExternalNodes

