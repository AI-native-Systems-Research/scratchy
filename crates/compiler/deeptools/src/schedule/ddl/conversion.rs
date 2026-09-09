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

//! `ddc/ddl/ddl_conversion.cpp`, `ddc/ddl/ddl_conversion.h` — 31 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4, 5]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e172_processTypes` | 172 | 0 | 24 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:447` |
//! | `e173_addInternalTensor` | 173 | 0 | 107 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:473` |
//! | `e174_processExpression` | 174 | 0 | 10 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2072` |
//! | `e175_getTensorProp` | 175 | 0 | 26 | `DdlInterface` | `ddc/ddl/ddl_conversion.cpp:2083` |
//! | `e176_getTensor` | 176 | 0 | 20 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2825` |
//! | `e177_dump` | 177 | 0 | 23 | `DimProp` | `ddc/ddl/ddl_conversion.cpp:3509` |
//! | `e178_getAccessPattern` | 178 | 0 | 8 | `DataTransfer` | `ddc/ddl/ddl_conversion.cpp:3619` |
//! | `e179_getAccessPatternAsStr` | 179 | 0 | 14 | `DataTransfer` | `ddc/ddl/ddl_conversion.cpp:3629` |
//! | `e180_checkAccessPattern` | 180 | 0 | 34 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:3648` |
//! | `e181_convertAccessPatternStrToDdcType` | 181 | 0 | 29 | — | `ddc/ddl/ddl_conversion.cpp:3687` |
//! | `e182_convertAccessPatternStrToDdcType` | 182 | 0 | 7 | — | `ddc/ddl/ddl_conversion.cpp:3718` |
//! | `e183_dump` | 183 | 0 | 13 | `DataTransfer` | `ddc/ddl/ddl_conversion.cpp:3756` |
//! | `e184_setMetaDimKind` | 184 | 0 | 7 | `DimProp` | `ddc/ddl/ddl_conversion.h:317` |
//! | `e185_isMetaDim` | 185 | 0 | 5 | `DimProp` | `ddc/ddl/ddl_conversion.h:329` |
//! | `e186_getNonPaddedDimProp` | 186 | 0 | 5 | `DdlInterface` | `ddc/ddl/ddl_conversion.h:351` |
//! | `e187_clear` | 187 | 0 | 4 | `DdlInterface` | `ddc/ddl/ddl_conversion.h:458` |
//! | `e274_processDimensionOp` | 274 | 1 | 22 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:187` |
//! | `e275_processCondition` | 275 | 1 | 234 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:211` |
//! | `e276_verifyDdlConstraints` | 276 | 1 | 216 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2553` |
//! | `e277_getTensorAndAllocation` | 277 | 1 | 31 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2847` |
//! | `e278_checkMetaDimensions` | 278 | 1 | 81 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:3537` |
//! | `e279_processAccessPatterns` | 279 | 1 | 24 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:3727` |
//! | `e322_processPaddedDimensionOp` | 322 | 2 | 84 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:102` |
//! | `e323_processOp` | 323 | 2 | 1424 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:582` |
//! | `e324_processTransformations` | 324 | 2 | 35 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2036` |
//! | `e325_convertDsc2Ddl` | 325 | 2 | 627 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2881` |
//! | `e345_exportToDdl` | 345 | 3 | 3 | `DdlConvertInterface` | `ddc/ddl/ddl_conversion.cpp:38` |
//! | `e346_processRegion` | 346 | 3 | 26 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2008` |
//! | `e347_matchDdl2Dsc` | 347 | 3 | 442 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2110` |
//! | `e364_parseDdl2Dsc` | 364 | 4 | 54 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:2770` |
//! | `e372_selectAndParseDdlTemplate` | 372 | 5 | 59 | `DdlConversion` | `ddc/ddl/ddl_conversion.cpp:42` |


// crustify:todo: e172_processTypes
//   authority : ddc/ddl/ddl_conversion.cpp:447  (24 body lines, level 0)
//   class     : DdlConversion
//   original  : std::vector<const DdlInterface::TypeDefinition*> DdlConversion::processTypes( mlir::OperandRange supportedTypes, mlir::Operation* userOp)
//   extract   : crustify-ddc/cpp/ddl.cpp:285-310

// crustify:todo: e173_addInternalTensor
//   authority : ddc/ddl/ddl_conversion.cpp:473  (107 body lines, level 0)
//   class     : DdlConversion
//   original  : LabeledDsInfo& DdlConversion::addInternalTensor(const LabeledDsInfo& refLds, int computeOpIdx)
//   extract   : crustify-ddc/cpp/ddl.cpp:320-428

// crustify:todo: e174_processExpression
//   authority : ddc/ddl/ddl_conversion.cpp:2072  (10 body lines, level 0)
//   class     : DdlConversion
//   original  : float DdlConversion::processExpression(std::string expr)
//   extract   : crustify-ddc/cpp/ddl.cpp:438-448

// crustify:todo: e175_getTensorProp
//   authority : ddc/ddl/ddl_conversion.cpp:2083  (26 body lines, level 0)
//   class     : DdlInterface
//   original  : const DdlInterface::TensorProp& DdlInterface::getTensorProp(Value tensorSSA)
//   extract   : crustify-ddc/cpp/ddl.cpp:458-484

// crustify:todo: e176_getTensor
//   authority : ddc/ddl/ddl_conversion.cpp:2825  (20 body lines, level 0)
//   class     : DdlConversion
//   original  : Value DdlConversion::getTensor(SenComponents unit, const dsc2::DataInfo dtinfo) const
//   extract   : crustify-ddc/cpp/ddl.cpp:494-515

// crustify:todo: e177_dump
//   authority : ddc/ddl/ddl_conversion.cpp:3509  (23 body lines, level 0)
//   class     : DimProp
//   original  : void DdlInterface::DimProp::dump(std::string msg) const
//   extract   : crustify-ddc/cpp/ddl.cpp:525-548

// crustify:todo: e178_getAccessPattern
//   authority : ddc/ddl/ddl_conversion.cpp:3619  (8 body lines, level 0)
//   class     : DataTransfer
//   original  : ddc::Metadata::TransferAccessPatternType ddc::Metadata::DataTransfer::getAccessPattern(PrimaryDimTypes dimVal)
//   extract   : crustify-ddc/cpp/ddl.cpp:558-567

// crustify:todo: e179_getAccessPatternAsStr
//   authority : ddc/ddl/ddl_conversion.cpp:3629  (14 body lines, level 0)
//   class     : DataTransfer
//   original  : std::string ddc::Metadata::DataTransfer::getAccessPatternAsStr( PrimaryDimTypes dimVal)
//   extract   : crustify-ddc/cpp/ddl.cpp:577-592

// crustify:todo: e180_checkAccessPattern
//   authority : ddc/ddl/ddl_conversion.cpp:3648  (34 body lines, level 0)
//   class     : DdlConversion
//   original  : template <typename T> bool DdlConversion::checkAccessPattern( T& op, std::string dimArgName, mlir::Operation::operand_range dims, std::string accessPatternAttrName, std::optional<mlir::ArrayAttr> opAccessPatternStyles)
//   extract   : crustify-ddc/cpp/ddl.cpp:602-640

// crustify:todo: e181_convertAccessPatternStrToDdcType
//   authority : ddc/ddl/ddl_conversion.cpp:3687  (29 body lines, level 0)
//   original  : ddc::Metadata::TransferAccessPatternType convertAccessPatternStrToDdcType( DataTransferOp& op, std::string const& accessPattern)
//   extract   : crustify-ddc/cpp/ddl.cpp:649-679

// crustify:todo: e182_convertAccessPatternStrToDdcType
//   authority : ddc/ddl/ddl_conversion.cpp:3718  (7 body lines, level 0)
//   original  : PadType convertAccessPatternStrToDdcType(AllocateOp& op, std::string const& paddingType)
//   extract   : crustify-ddc/cpp/ddl.cpp:688-696

// crustify:todo: e183_dump
//   authority : ddc/ddl/ddl_conversion.cpp:3756  (13 body lines, level 0)
//   class     : DataTransfer
//   original  : void Metadata::DataTransfer::dump()
//   extract   : crustify-ddc/cpp/ddl.cpp:706-719

// crustify:todo: e184_setMetaDimKind
//   authority : ddc/ddl/ddl_conversion.h:317  (7 body lines, level 0)
//   class     : DimProp
//   original  : bool setMetaDimKind(llvm::StringRef inDimKind)
//   extract   : crustify-ddc/cpp/ddl.cpp:729-736

// crustify:todo: e185_isMetaDim
//   authority : ddc/ddl/ddl_conversion.h:329  (5 body lines, level 0)
//   class     : DimProp
//   original  : bool isMetaDim() const
//   extract   : crustify-ddc/cpp/ddl.cpp:746-751

// crustify:todo: e186_getNonPaddedDimProp
//   authority : ddc/ddl/ddl_conversion.h:351  (5 body lines, level 0)
//   class     : DdlInterface
//   original  : DimProp& getNonPaddedDimProp(Value ddlDim)
//   extract   : crustify-ddc/cpp/ddl.cpp:761-766

// crustify:todo: e187_clear
//   authority : ddc/ddl/ddl_conversion.h:458  (4 body lines, level 0)
//   class     : DdlInterface
//   original  : void clear()
//   extract   : crustify-ddc/cpp/ddl.cpp:776-780

// crustify:todo: e274_processDimensionOp
//   authority : ddc/ddl/ddl_conversion.cpp:187  (22 body lines, level 1)
//   class     : DdlConversion
//   original  : DdlInterface::DimProp& DdlConversion::processDimensionOp( const mlir::Value& dimVal)
//   extract   : crustify-ddc/cpp/ddl.cpp:860-883
//   calls     : e184_setMetaDimKind

// crustify:todo: e275_processCondition
//   authority : ddc/ddl/ddl_conversion.cpp:211  (234 body lines, level 1)
//   class     : DdlConversion
//   original  : const DdlInterface::CondProp& DdlConversion::processCondition( mlir::Value cond)
//   extract   : crustify-ddc/cpp/ddl.cpp:893-1128
//   calls     : e174_processExpression, e175_getTensorProp, e176_getTensor, e187_clear

// crustify:todo: e276_verifyDdlConstraints
//   authority : ddc/ddl/ddl_conversion.cpp:2553  (216 body lines, level 1)
//   class     : DdlConversion
//   original  : void DdlConversion::verifyDdlConstraints()
//   extract   : crustify-ddc/cpp/ddl.cpp:1138-1354
//   calls     : e175_getTensorProp, e176_getTensor

// crustify:todo: e277_getTensorAndAllocation
//   authority : ddc/ddl/ddl_conversion.cpp:2847  (31 body lines, level 1)
//   class     : DdlConversion
//   original  : std::pair<Value, Value> DdlConversion::getTensorAndAllocation( llvm::DenseMap<AllocateOp, const dsc2::AllocateNode*>& allocations, SenComponents unit, const dsc2::DataInfo dtinfo) const
//   extract   : crustify-ddc/cpp/ddl.cpp:1364-1397
//   calls     : e176_getTensor

// crustify:todo: e278_checkMetaDimensions
//   authority : ddc/ddl/ddl_conversion.cpp:3537  (81 body lines, level 1)
//   class     : DdlConversion
//   original  : void DdlConversion::checkMetaDimensions()
//   extract   : crustify-ddc/cpp/ddl.cpp:1407-1488
//   calls     : e185_isMetaDim

// crustify:todo: e279_processAccessPatterns
//   authority : ddc/ddl/ddl_conversion.cpp:3727  (24 body lines, level 1)
//   class     : DdlConversion
//   original  : template <typename OpT, typename AccPatT> void DdlConversion::processAccessPatterns( OpT& op, mlir::Operation::operand_range dims, mlir::ArrayAttr inOpAccessPatternStyles, std::map<PrimaryDimTypes, AccPatT>& result)
//   extract   : crustify-ddc/cpp/ddl.cpp:1498-1526
//   calls     : e181_convertAccessPatternStrToDdcType, e182_convertAccessPatternStrToDdcType

// crustify:todo: e322_processPaddedDimensionOp
//   authority : ddc/ddl/ddl_conversion.cpp:102  (84 body lines, level 2)
//   class     : DdlConversion
//   original  : void DdlConversion::processPaddedDimensionOp(const mlir::Value& dimVal)
//   extract   : crustify-ddc/cpp/ddl.cpp:1562-1646
//   calls     : e184_setMetaDimKind, e274_processDimensionOp

// crustify:todo: e323_processOp
//   authority : ddc/ddl/ddl_conversion.cpp:582  (1424 body lines, level 2)
//   class     : DdlConversion
//   original  : std::pair<dsc2::BlockNode*, std::vector<int>> DdlConversion::processOp( Operation& op, dsc2::BlockNode* currParent)
//   extract   : crustify-ddc/cpp/ddl.cpp:1656-3081
//   calls     : e069_updateMin, e070_updateMax, e071_updateValues, e096_updateMin, e097_updateMax, e098_updateValues, e173_addInternalTensor, e174_processExpression, e175_getTensorProp, e176_getTensor, e187_clear, e275_processCondition

// crustify:todo: e324_processTransformations
//   authority : ddc/ddl/ddl_conversion.cpp:2036  (35 body lines, level 2)
//   class     : DdlConversion
//   original  : void DdlConversion::processTransformations(::mlir::Region& myRegion)
//   extract   : crustify-ddc/cpp/ddl.cpp:3091-3126
//   calls     : e177_dump, e183_dump, e275_processCondition

// crustify:todo: e325_convertDsc2Ddl
//   authority : ddc/ddl/ddl_conversion.cpp:2881  (627 body lines, level 2)
//   class     : DdlConversion
//   original  : void DdlConversion::convertDsc2Ddl(std::ostream& outputDdl)
//   extract   : crustify-ddc/cpp/ddl.cpp:3136-3763
//   calls     : e176_getTensor, e177_dump, e179_getAccessPatternAsStr, e183_dump, e277_getTensorAndAllocation

// crustify:todo: e345_exportToDdl
//   authority : ddc/ddl/ddl_conversion.cpp:38  (3 body lines, level 3)
//   class     : DdlConvertInterface
//   original  : void DdlConvertInterface::exportToDdl(std::ostream& outputDdl)
//   extract   : crustify-ddc/cpp/ddl.cpp:3795-3798
//   calls     : e325_convertDsc2Ddl

// crustify:todo: e346_processRegion
//   authority : ddc/ddl/ddl_conversion.cpp:2008  (26 body lines, level 3)
//   class     : DdlConversion
//   original  : void DdlConversion::processRegion(::mlir::Region& myRegion, dsc2::BlockNode* currParent)
//   extract   : crustify-ddc/cpp/ddl.cpp:3808-3835
//   calls     : e323_processOp

// crustify:todo: e347_matchDdl2Dsc
//   authority : ddc/ddl/ddl_conversion.cpp:2110  (442 body lines, level 3)
//   class     : DdlConversion
//   original  : bool DdlConversion::matchDdl2Dsc()
//   extract   : crustify-ddc/cpp/ddl.cpp:3845-4287
//   calls     : e021_getOpFuncName, e172_processTypes, e173_addInternalTensor, e175_getTensorProp, e177_dump, e183_dump, e187_clear, e272_print, e274_processDimensionOp, e278_checkMetaDimensions, e322_processPaddedDimensionOp

// crustify:todo: e364_parseDdl2Dsc
//   authority : ddc/ddl/ddl_conversion.cpp:2770  (54 body lines, level 4)
//   class     : DdlConversion
//   original  : void DdlConversion::parseDdl2Dsc()
//   extract   : crustify-ddc/cpp/ddl.cpp:4316-4370
//   calls     : e324_processTransformations, e346_processRegion

// crustify:todo: e372_selectAndParseDdlTemplate
//   authority : ddc/ddl/ddl_conversion.cpp:42  (59 body lines, level 5)
//   class     : DdlConversion
//   original  : bool DdlConversion::selectAndParseDdlTemplate()
//   extract   : crustify-ddc/cpp/ddl.cpp:4380-4439
//   calls     : e187_clear, e276_verifyDdlConstraints, e347_matchDdl2Dsc, e363_parseDdl, e364_parseDdl2Dsc

