# Bridge 2 — the porting order, definition by definition

`DataflowIR -> SentientIR`, the D1-D28 span. **490 definitions, 11 levels**, in dependency order.

## ⛔ THIS SUPERSEDES AN EARLIER VERSION THAT WAS WRONG, AND THE ERROR IS WORTH KNOWING

The first graph keyed nodes by **function NAME**, which collapsed every same-named definition into
one node. `runOnOperation` is defined in **21 files** — every pass has one — so all 21 pass drivers
became a single hub, fusing unrelated passes into each other's dependency chains. `dccExtContext`
(9 files), `dcc_ext_ctx_` (8), `opts_` (6) and `matchAndRewrite` (4) did the same. 31 of 418 names
were multiply defined.

It also cited files by **basename**, which is ambiguous: there are two `Helper.cpp`
(`AgenToSentient/` and `VectorChainToSentientPT/`) and `VectorChainHelper.cpp` actually lives under
`CommonHelpers/`. A citation that cannot be resolved is not a citation.

Corrected: nodes are keyed `(file, name)`, calls resolve to a same-file definition first and to a
unique cross-file definition otherwise, and 38 genuinely ambiguous cross-file calls are dropped
rather than guessed. **490 definitions, 929 edges, 11 levels** — against 418/949/9 before.

## How it is derived, and how it still lies

A textual call graph over `dcc/src/{Transform/Dataflow,Conversion/*}` in the pod extract at
`a0d29abbed`. Level 0 calls nothing else in the span; level N calls only levels below it.

⭐ **DEFINITIONS AT THE SAME LEVEL NEVER CALL EACH OTHER**, which is what makes a level a safe batch.

⛔ It cannot see through virtual dispatch (an edge to the base, never the override), templates,
function pointers or `std::function`; the 38 dropped ambiguous calls are real edges it declines to
guess at; and cycles are broken at the first back edge. A level is a **starting order, not a proof**:
if a port needs something from a higher level, the graph missed an edge.

⛔ **AND IT SAYS NOTHING ABOUT WHAT IS WORTH PORTING.** Between 26% and 83% of these lines are MLIR
construction — `OpBuilder`, SSA iterators, memoisers, the nested `(core, corelet, component)` lookup
maps — which exists because the C++ builds an IR at run time. We emit from a resolved tape. **Port the
rule, never the plumbing.**

## Level 0 — 239 definitions, 1614 body lines

### Conversion/VectorChainLowering — 54 defs, 654 lines

- [ ] `getMaskValueForPT` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17`, 196 lines
- [ ] `getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34`, 54 lines
- [ ] `getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95`, 50 lines
- [ ] `fuseCompareAndSelectIntoMinOrMax` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415`, 49 lines
- [ ] `redefineConstantVectors` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571`, 37 lines
- [ ] `getForOpBound` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262`, 31 lines
- [ ] `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243`, 26 lines
- [ ] `getVectorBinaryToSentientBinary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32`, 17 lines
- [ ] `eraseOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806`, 16 lines
- [ ] `getComputePrecisionOfOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65`, 13 lines
- [ ] `resetSentientFMAsIfExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468`, 13 lines
- [ ] `constValToField` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250`, 12 lines
- [ ] `sameBlock` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652`, 11 lines
- [ ] `getXrfValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248`, 11 lines
- [ ] `setSentientMacXrfRegIncrAttr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665`, 11 lines
- [ ] `hasConstantBounds` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80`, 10 lines
- [ ] `getResultPrecisionFromOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52`, 10 lines
- [ ] `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55`, 9 lines
- [ ] `computeUnitPrecision` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31`, 9 lines
- [ ] `getVectorTernaryToSentientTernary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247`, 7 lines
- [ ] `replaceAndEraseDummyMacOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681`, 7 lines
- [ ] `size` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319`, 6 lines
- [ ] `getAbsorbtionFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73`, 6 lines
- [ ] `getId` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65`, 6 lines
- [ ] `getInputPrecisionFromOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36`, 6 lines
- [ ] `reuse_info_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:82`, 5 lines
- [ ] `isSentientBinaryLogicalOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30`, 4 lines
- [ ] `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167`, 4 lines
- [ ] `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132`, 2 lines
- [ ] `dominates` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38`, 2 lines
- [ ] `setReuseFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91`, 2 lines
- [ ] `getParentNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36`, 2 lines
- [ ] `getFirstChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39`, 2 lines
- [ ] `getNextSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42`, 2 lines
- [ ] `getPrevSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45`, 2 lines
- [ ] `getLastChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48`, 2 lines
- [ ] `getRoot` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109`, 2 lines
- [ ] `getFirstValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:76`, 0 lines
- [ ] `getIncrement` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:77`, 0 lines
- [ ] `isMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:55`, 0 lines
- [ ] `getStartVal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:76`, 0 lines
- [ ] `isLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:54`, 0 lines
- [ ] `size_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:208`, 0 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:45`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:51`, 0 lines
- [ ] `is_visited_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:108`, 0 lines
- [ ] `MaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74`, 0 lines
- [ ] `setOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:52`, 0 lines
- [ ] `LMTLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94`, 0 lines
- [ ] `LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34`, 0 lines
- [ ] `OperationNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32`, 0 lines
- [ ] `increment_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:72`, 0 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:53`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:59`, 0 lines

### Transform/Dataflow — 75 defs, 434 lines

- [ ] `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21`, 48 lines
- [ ] `transformSCFToAffineLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102`, 44 lines
- [ ] `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99`, 31 lines
- [ ] `isEligibleForSplitting` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832`, 20 lines
- [ ] `removeCoresCoreletsFoldsFromProgramUnit` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240`, 20 lines
- [ ] `partitionUnits` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168`, 17 lines
- [ ] `getConstantTripCount` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167`, 14 lines
- [ ] `createInequalityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263`, 14 lines
- [ ] `createDummyYieldInElseReg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383`, 13 lines
- [ ] `createNewMemViewWithMod` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966`, 12 lines
- [ ] `getLhsRhsOfEQPredicate` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519`, 11 lines
- [ ] `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314`, 11 lines
- [ ] `printUnitToOpsMap` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:189`, 10 lines
- [ ] `calculateStartElementsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532`, 10 lines
- [ ] `createNonPagedMemView` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545`, 10 lines
- [ ] `identifyTimeDimForExplicitLoops` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963`, 10 lines
- [ ] `createNewMemOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684`, 9 lines
- [ ] `cloneMemViewIfNonPaged` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633`, 9 lines
- [ ] `printEquivalenceClasses` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:202`, 8 lines
- [ ] `getMaxMutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673`, 8 lines
- [ ] `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683`, 8 lines
- [ ] `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355`, 8 lines
- [ ] `performFullUnroll` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141`, 7 lines
- [ ] `removeValuesFromIndices` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36`, 7 lines
- [ ] `replaceDimsInMapWithSyms` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47`, 7 lines
- [ ] `createEqualityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253`, 7 lines
- [ ] `setBuilderToInsertRef` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280`, 6 lines
- [ ] `getStoreOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843`, 6 lines
- [ ] `TransformationConditionalTree` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111`, 5 lines
- [ ] `addTimeDimIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876`, 5 lines
- [ ] `sortDataBasedOnWeight` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855`, 4 lines
- [ ] `base_unit_program_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:43`, 4 lines
- [ ] `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698`, 3 lines
- [ ] `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673`, 3 lines
- [ ] `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678`, 3 lines
- [ ] `isOperationSelected` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, 2 lines
- [ ] `getRoot` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81`, 2 lines
- [ ] `getNewDbgNameFromList` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456`, 2 lines
- [ ] `ConditionalSimplificationManager` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78`, 2 lines
- [ ] `populateDuplicateReusedTogglePatterns` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:207`, 2 lines
- [ ] `getParentNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53`, 2 lines
- [ ] `getFirstChild` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56`, 2 lines
- [ ] `getNextSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59`, 2 lines
- [ ] `getPrevSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62`, 2 lines
- [ ] `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:30`, 2 lines
- [ ] `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328`, 2 lines
- [ ] `opts_` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:69`, 0 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:47`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:94`, 0 lines
- [ ] `opts_` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:39`, 0 lines
- [ ] `OperationNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51`, 0 lines
- [ ] `OperationTreeBase` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78`, 0 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:54`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:66`, 0 lines
- [ ] `opts_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:84`, 0 lines
- [ ] `mem_index_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:94`, 0 lines
- [ ] `weight_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:111`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:219`, 0 lines
- [ ] `opts_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:61`, 0 lines
- [ ] `mem_index_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:71`, 0 lines
- [ ] `weight_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:83`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:125`, 0 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:55`, 0 lines
- [ ] `curr_unit_corelet_id_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:91`, 0 lines
- [ ] `opts_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:34`, 0 lines
- [ ] `subscripts_map_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:45`, 0 lines
- [ ] `mem_index_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:50`, 0 lines
- [ ] `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341`, 0 lines
- [ ] `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364`, 0 lines
- [ ] `TPMVBase` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389`, 0 lines
- [ ] `TPMVVector` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397`, 0 lines
- [ ] `access_details_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:458`, 0 lines
- [ ] `TPMVComposite` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519`, 0 lines
- [ ] `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.hpp:24`, 0 lines
- [ ] `opts_` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:43`, 0 lines

### Conversion/AgenToSentient — 82 defs, 319 lines

- [ ] `setldtype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647`, 60 lines
- [ ] `generateSetSendDestinationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731`, 49 lines
- [ ] `checkIndirectMemViewForExtractOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:388`, 42 lines
- [ ] `getLoadConsumer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242`, 36 lines
- [ ] `findExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:514`, 20 lines
- [ ] `insertCopyAndAddStmtsHelper` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:502`, 17 lines
- [ ] `findCandidateForLowering` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884`, 12 lines
- [ ] `addLoadChainToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975`, 9 lines
- [ ] `getStoreOpFromLoadStorePattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872`, 8 lines
- [ ] `getLoopNestLevel` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:43`, 6 lines
- [ ] `setCoalescedBoundValues` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672`, 5 lines
- [ ] `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:230`, 4 lines
- [ ] `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:248`, 4 lines
- [ ] `createAgenToSentientPass` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:252`, 3 lines
- [ ] `has` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:397`, 2 lines
- [ ] `setIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77`, 2 lines
- [ ] `setSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228`, 2 lines
- [ ] `setMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81`, 2 lines
- [ ] `setMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99`, 2 lines
- [ ] `setTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125`, 2 lines
- [ ] `setTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122`, 2 lines
- [ ] `setExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110`, 2 lines
- [ ] `setElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119`, 2 lines
- [ ] `setIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231`, 2 lines
- [ ] `setRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107`, 2 lines
- [ ] `setShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104`, 2 lines
- [ ] `setMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93`, 2 lines
- [ ] `setLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96`, 2 lines
- [ ] `setTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116`, 2 lines
- [ ] `setExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113`, 2 lines
- [ ] `setTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295`, 2 lines
- [ ] `setTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292`, 2 lines
- [ ] `setInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299`, 2 lines
- [ ] `setTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286`, 2 lines
- [ ] `setTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289`, 2 lines
- [ ] `setStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:355`, 2 lines
- [ ] `getIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:70`, 0 lines
- [ ] `getMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:59`, 0 lines
- [ ] `getElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:69`, 0 lines
- [ ] `getOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:57`, 0 lines
- [ ] `getSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:221`, 0 lines
- [ ] `setMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:84`, 0 lines
- [ ] `setMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:80`, 0 lines
- [ ] `getMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:72`, 0 lines
- [ ] `getTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:63`, 0 lines
- [ ] `getMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:61`, 0 lines
- [ ] `getRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:67`, 0 lines
- [ ] `getExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:73`, 0 lines
- [ ] `setChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:103`, 0 lines
- [ ] `setChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:102`, 0 lines
- [ ] `getLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:62`, 0 lines
- [ ] `getTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:68`, 0 lines
- [ ] `getTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:89`, 0 lines
- [ ] `getExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:88`, 0 lines
- [ ] `setLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:128`, 0 lines
- [ ] `getChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:64`, 0 lines
- [ ] `getComp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:58`, 0 lines
- [ ] `getTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:263`, 0 lines
- [ ] `getTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:267`, 0 lines
- [ ] `getTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:266`, 0 lines
- [ ] `getTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:265`, 0 lines
- [ ] `getBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:268`, 0 lines
- [ ] `getLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:90`, 0 lines
- [ ] `setBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:298`, 0 lines
- [ ] `getTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:262`, 0 lines
- [ ] `getTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:264`, 0 lines
- [ ] `getIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:222`, 0 lines
- [ ] `comp_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:47`, 0 lines
- [ ] `getMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:60`, 0 lines
- [ ] `getChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:65`, 0 lines
- [ ] `getShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:66`, 0 lines
- [ ] `getMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:71`, 0 lines
- [ ] `setOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:76`, 0 lines
- [ ] `AccessDetailsBase` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218`, 0 lines
- [ ] `AccessDetailsAffine` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:259`, 0 lines
- [ ] `getInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:269`, 0 lines
- [ ] `setTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:284`, 0 lines
- [ ] `setTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:285`, 0 lines
- [ ] `getStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:352`, 0 lines
- [ ] `index_mapping_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:375`, 0 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:38`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:521`, 0 lines

### Conversion/StandardToSentient — 8 defs, 71 lines

- [ ] `getSentientCmpIPredicate` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36`, 18 lines
- [ ] `LowerConstantIntToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358`, 15 lines
- [ ] `LowerSubIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90`, 10 lines
- [ ] `LowerAddIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79`, 9 lines
- [ ] `LowerMulIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102`, 9 lines
- [ ] `LowerConstantIndexToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347`, 8 lines
- [ ] `createStandardToSentientPass` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:476`, 2 lines
- [ ] `If` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159`, 0 lines

### Conversion/DataflowToSentient — 9 defs, 61 lines

- [ ] `lowerOpaqueOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984`, 27 lines
- [ ] `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104`, 11 lines
- [ ] `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175`, 10 lines
- [ ] `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720`, 6 lines
- [ ] `createDataflowToSentientPass` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2051`, 3 lines
- [ ] `isSenComponentL0LU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96`, 2 lines
- [ ] `isSenComponentL0SU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100`, 2 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:42`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:91`, 0 lines

### Conversion/SCFToSentient — 4 defs, 48 lines

- [ ] `runOnOperation` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251`, 30 lines
- [ ] `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33`, 16 lines
- [ ] `createSCFToSentientPass` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:285`, 2 lines
- [ ] `ConversionPattern` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70`, 0 lines

### Conversion/AffineToStandard — 4 defs, 24 lines

- [ ] `matchAndRewrite` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41`, 8 lines
- [ ] `populateAffineToStdConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:198`, 8 lines
- [ ] `populateAffineToVectorConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:208`, 6 lines
- [ ] `createAffineToStandardPass` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:239`, 2 lines

### Conversion/SymbolToSentient — 3 defs, 3 lines

- [ ] `createSymbolToSentientPass` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:187`, 3 lines
- [ ] `dcc_ext_ctx_` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:27`, 0 lines
- [ ] `dccExtContext` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:38`, 0 lines

## Level 1 — 67 definitions, 2356 body lines

### Transform/Dataflow — 32 defs, 1178 lines

- [ ] `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, 127 lines
- [ ] `calculatePartitionSizes` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862`, 97 lines
- [ ] `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, 77 lines
- [ ] `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, 60 lines
- [ ] `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, 60 lines
- [ ] `constructConditionals` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000`, 59 lines
- [ ] `createExplicitTimeLoops` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282`, 57 lines
- [ ] `getLoopTripCount` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743`, 56 lines
- [ ] `expandAffineApplyOps` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183`, 53 lines
- [ ] `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, 49 lines
- [ ] `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, 48 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, 42 lines
- [ ] `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, 38 lines
- [ ] `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, 37 lines
- [ ] `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, 35 lines
- [ ] `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, 26 lines
- [ ] `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, 25 lines
- [ ] `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, 25 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, 23 lines
- [ ] `synthesizeTimeInfo` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256`, 22 lines
- [ ] `if_op_` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:56`, 20 lines
- [ ] `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, 17 lines
- [ ] `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, 17 lines
- [ ] `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, 15 lines
- [ ] `hasMutableAddrOverflow` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801`, 14 lines
- [ ] `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, 13 lines
- [ ] `printTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:288`, 13 lines
- [ ] `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, 13 lines
- [ ] `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, 13 lines
- [ ] `calculateSubscriptsCoefficients` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188`, 11 lines
- [ ] `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, 10 lines
- [ ] `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, 6 lines

### Conversion/AgenToSentient — 16 defs, 764 lines

- [ ] `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, 161 lines
- [ ] `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, 159 lines
- [ ] `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, 89 lines
- [ ] `constructExtentAndTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32`, 64 lines
- [ ] `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, 57 lines
- [ ] `constructIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354`, 50 lines
- [ ] `computeBurstAndGroup` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796`, 36 lines
- [ ] `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, 35 lines
- [ ] `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, 27 lines
- [ ] `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, 27 lines
- [ ] `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, 17 lines
- [ ] `initializeMemViewInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274`, 16 lines
- [ ] `constructIteratorCoefficients` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406`, 10 lines
- [ ] `insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388`, 7 lines
- [ ] `constructLdOrStType` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267`, 5 lines
- [ ] `get` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401`, 4 lines

### Conversion/VectorChainLowering — 15 defs, 370 lines

- [ ] `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, 109 lines
- [ ] `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, 45 lines
- [ ] `getLayoutMapAndIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879`, 40 lines
- [ ] `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, 40 lines
- [ ] `isXrfRelated` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531`, 31 lines
- [ ] `areXrfAccessesLegal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98`, 28 lines
- [ ] `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, 26 lines
- [ ] `getName` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866`, 11 lines
- [ ] `insertConstAndAddOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296`, 10 lines
- [ ] `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, 8 lines
- [ ] `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, 8 lines
- [ ] `getOperandFromConstantBitstreamOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299`, 5 lines
- [ ] `getOperandFromNegOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366`, 5 lines
- [ ] `isMaskEquivalentToNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123`, 4 lines
- [ ] `getTotalDataOriginsCount` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:33`, 0 lines

### Conversion/DataflowToSentient — 3 defs, 34 lines

- [ ] `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, 18 lines
- [ ] `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, 10 lines
- [ ] `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, 6 lines

### Conversion/AffineToStandard — 1 defs, 10 lines

- [ ] `runOnOperation` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:222`, 10 lines

## Level 2 — 61 definitions, 2729 body lines

### Conversion/AgenToSentient — 15 defs, 898 lines

- [ ] `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, 167 lines
- [ ] `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, 133 lines
- [ ] `constructReceiveAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471`, 91 lines
- [ ] `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, 80 lines
- [ ] `cloneStartAddrOutsideLoop` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942`, 79 lines
- [ ] `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, 74 lines
- [ ] `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, 57 lines
- [ ] `setsttype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731`, 50 lines
- [ ] `lowerVectorLoadHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899`, 46 lines
- [ ] `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, 43 lines
- [ ] `cleanupTriviallyRedundantSetSendDestination` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084`, 38 lines
- [ ] `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, 16 lines
- [ ] `lowerSetTransferMaskStateOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815`, 9 lines
- [ ] `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, 8 lines
- [ ] `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, 7 lines

### Transform/Dataflow — 21 defs, 856 lines

- [ ] `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, 117 lines
- [ ] `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, 98 lines
- [ ] `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, 91 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, 76 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, 68 lines
- [ ] `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, 66 lines
- [ ] `enumerateCollectionUnit` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34`, 57 lines
- [ ] `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, 47 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, 41 lines
- [ ] `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, 32 lines
- [ ] `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, 31 lines
- [ ] `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, 27 lines
- [ ] `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, 22 lines
- [ ] `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, 22 lines
- [ ] `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, 18 lines
- [ ] `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, 15 lines
- [ ] `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, 10 lines
- [ ] `isHoistable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82`, 9 lines
- [ ] `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, 6 lines
- [ ] `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, 3 lines
- [ ] `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, 0 lines

### Conversion/VectorChainLowering — 19 defs, 704 lines

- [ ] `insertPTMaskOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41`, 164 lines
- [ ] `getOperandFromLoadOrStoreOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153`, 94 lines
- [ ] `getLayoutExpr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28`, 67 lines
- [ ] `createForOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129`, 56 lines
- [ ] `createIfOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189`, 56 lines
- [ ] `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, 49 lines
- [ ] `eraseOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690`, 46 lines
- [ ] `createSentientConstants` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34`, 31 lines
- [ ] `convertStringToType` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196`, 24 lines
- [ ] `print` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:96`, 22 lines
- [ ] `convertTypeToString` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223`, 22 lines
- [ ] `computeLoops` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173`, 18 lines
- [ ] `addMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136`, 16 lines
- [ ] `insertDummyMacOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312`, 15 lines
- [ ] `updateNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155`, 10 lines
- [ ] `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, 9 lines
- [ ] `setValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72`, 3 lines
- [ ] `dominance_info_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:26`, 2 lines
- [ ] `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, 0 lines

### Conversion/DataflowToSentient — 4 defs, 141 lines

- [ ] `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, 61 lines
- [ ] `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, 57 lines
- [ ] `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, 18 lines
- [ ] `pushBackTheUnitToListIfDoesnotExist` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143`, 5 lines

### Conversion/SCFToSentient — 1 defs, 69 lines

- [ ] `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, 69 lines

### Conversion/SymbolToSentient — 1 defs, 61 lines

- [ ] `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, 61 lines

## Level 3 — 33 definitions, 2120 body lines

### Transform/Dataflow — 13 defs, 740 lines

- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:362`, 127 lines
- [ ] `createIterArgsForConditionals` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:401`, 121 lines
- [ ] `hoistCommonConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:94`, 96 lines
- [ ] `transformSCFLoopWithNonConstantUpperBound` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169`, 94 lines
- [ ] `flatten` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:384`, 70 lines
- [ ] `calculatePartialShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490`, 66 lines
- [ ] `replaceIfOpByIterArg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:619`, 56 lines
- [ ] `getPageValidity` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:114`, 33 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94`, 32 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:131`, 22 lines
- [ ] `createPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983`, 10 lines
- [ ] `initialize` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694`, 8 lines
- [ ] `runOn` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447`, 5 lines

### Conversion/AgenToSentient — 8 defs, 588 lines

- [ ] `constructLoadAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2167`, 177 lines
- [ ] `constructLoadAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2351`, 114 lines
- [ ] `coalesceTimeDimensions` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:681`, 112 lines
- [ ] `constructTimeLoopsAndVectorOperations` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1789`, 109 lines
- [ ] `lowerCompositeMemoryInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3775`, 37 lines
- [ ] `constructDetails` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:418`, 18 lines
- [ ] `insertInitializationStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3834`, 13 lines
- [ ] `addStoreInputToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2987`, 8 lines

### Conversion/DataflowToSentient — 2 defs, 402 lines

- [ ] `lowerL0LXSyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438`, 222 lines
- [ ] `lowerL0LXSyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189`, 180 lines

### Conversion/VectorChainLowering — 9 defs, 322 lines

- [ ] `createSplatOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:70`, 110 lines
- [ ] `getGCVTorFCVTTypeFromIndicesAndCastInputs` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188`, 108 lines
- [ ] `setReuseInformation` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:17`, 44 lines
- [ ] `getOperandFromShuffleOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311`, 36 lines
- [ ] `updateLoopMaskTreeForDynamicMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:28`, 9 lines
- [ ] `cleanup` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1056`, 7 lines
- [ ] `updateLoopMaskTreeForConstantMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:20`, 4 lines
- [ ] `op_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:79`, 2 lines
- [ ] `OperationTreeBase` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:105`, 2 lines

### Conversion/SymbolToSentient — 1 defs, 68 lines

- [ ] `LowerSymbolQueryMap` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40`, 68 lines

## Level 4 — 15 definitions, 1606 body lines

### Conversion/DataflowToSentient — 3 defs, 943 lines

- [ ] `lowerSyncLXL3ToLXL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787`, 928 lines
- [ ] `lowerSyncForAGroup` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746`, 8 lines
- [ ] `lowerSyncForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733`, 7 lines

### Transform/Dataflow — 7 defs, 321 lines

- [ ] `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298`, 75 lines
- [ ] `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375`, 74 lines
- [ ] `calculateShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396`, 61 lines
- [ ] `constructValidPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190`, 58 lines
- [ ] `analyzeValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888`, 33 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:459`, 17 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:455`, 3 lines

### Conversion/VectorChainLowering — 1 defs, 254 lines

- [ ] `getOperandWithPrecision` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389`, 254 lines

### Conversion/AgenToSentient — 3 defs, 73 lines

- [ ] `constructTimeStepsInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:626`, 42 lines
- [ ] `constructAffineDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2787`, 16 lines
- [ ] `lowerAffineCompositeHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2953`, 15 lines

### Conversion/SymbolToSentient — 1 defs, 15 lines

- [ ] `runOnOperation` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23`, 15 lines

## Level 5 — 16 definitions, 1096 body lines

### Conversion/AgenToSentient — 8 defs, 538 lines

- [ ] `lowerLDCVTIPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3444`, 326 lines
- [ ] `lowerIndirectVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3217`, 46 lines
- [ ] `lowerIndirectVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3169`, 44 lines
- [ ] `constructAffineCompDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2809`, 33 lines
- [ ] `lowerVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3074`, 28 lines
- [ ] `lowerExtractVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3001`, 21 lines
- [ ] `lowerVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3050`, 20 lines
- [ ] `lowerExtractVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3026`, 20 lines

### Transform/Dataflow — 6 defs, 388 lines

- [ ] `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546`, 124 lines
- [ ] `transform_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977`, 98 lines
- [ ] `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451`, 92 lines
- [ ] `analyzeAndConstructValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152`, 32 lines
- [ ] `initialize_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095`, 23 lines
- [ ] `shiftMutableAddr` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366`, 19 lines

### Conversion/DataflowToSentient — 1 defs, 166 lines

- [ ] `lowerSyncForAQueryMap` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728`, 166 lines

### Conversion/VectorChainLowering — 1 defs, 4 lines

- [ ] `getOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:378`, 4 lines

## Level 6 — 30 definitions, 1790 body lines

### Transform/Dataflow — 10 defs, 536 lines

- [ ] `matchAndRewrite` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33`, 170 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226`, 70 lines
- [ ] `isLoopInvariant` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:681`, 66 lines
- [ ] `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298`, 54 lines
- [ ] `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256`, 39 lines
- [ ] `transform` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592`, 39 lines
- [ ] `topLevelConditionsMatch` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:279`, 31 lines
- [ ] `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201`, 25 lines
- [ ] `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229`, 24 lines
- [ ] `singleOpBranchToYieldVal` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:498`, 18 lines

### Conversion/VectorChainLowering — 7 defs, 514 lines

- [ ] `processXrfPtrPerUnit` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337`, 190 lines
- [ ] `lowerDanglingNonComputeOpsPESFP` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1274`, 93 lines
- [ ] `lowerDanglingNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882`, 88 lines
- [ ] `analyzeNonComputeOpsForFusion` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:610`, 86 lines
- [ ] `analyzeAndFillResultForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:162`, 27 lines
- [ ] `analyzeAndFillOperandForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:535`, 25 lines
- [ ] `getOperandFromCastOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354`, 5 lines

### Conversion/AgenToSentient — 10 defs, 492 lines

- [ ] `gatherSymbolicLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1051`, 153 lines
- [ ] `adjustMutableAddrInitForIndirect` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1447`, 127 lines
- [ ] `adjustMutableAddrInitForStride` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4034`, 47 lines
- [ ] `lowerCompositeIndirectLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3267`, 42 lines
- [ ] `lowerCompositeIndirectStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3313`, 42 lines
- [ ] `lowerCompositeLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3148`, 17 lines
- [ ] `lowerCompositeLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3106`, 17 lines
- [ ] `lowerCompositeIndirectLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3359`, 17 lines
- [ ] `lowerCompositeStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3127`, 17 lines
- [ ] `insertCopyAndAddStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3861`, 13 lines

### Conversion/StandardToSentient — 2 defs, 168 lines

- [ ] `ConstructIFRecursively` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:113`, 125 lines
- [ ] `SimplifyOrIOp` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:389`, 43 lines

### Conversion/DataflowToSentient — 1 defs, 80 lines

- [ ] `lowerSyncOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901`, 80 lines

## Level 7 — 18 definitions, 2238 body lines

### Conversion/VectorChainLowering — 6 defs, 1301 lines

- [ ] `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245`, 628 lines
- [ ] `patternAgnosticFuseNonComputeOpsHelper` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:96`, 222 lines
- [ ] `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46`, 191 lines
- [ ] `createXrfIndexModifOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564`, 97 lines
- [ ] `fillOpInfo` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1070`, 83 lines
- [ ] `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1159`, 80 lines

### Conversion/AgenToSentient — 4 defs, 632 lines

- [ ] `generateAffineAddressManipulationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:625`, 381 lines
- [ ] `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2025`, 131 lines
- [ ] `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1910`, 104 lines
- [ ] `constructSymbolicDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2849`, 16 lines

### Transform/Dataflow — 5 defs, 234 lines

- [ ] `parseConditional` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:534`, 83 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130`, 69 lines
- [ ] `hoistLoopInvariantConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:750`, 42 lines
- [ ] `areShallowlyMergeable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:343`, 34 lines
- [ ] `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:647`, 6 lines

### Conversion/StandardToSentient — 2 defs, 38 lines

- [ ] `LowerSelectOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:243`, 19 lines
- [ ] `LowerLogicalOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:264`, 19 lines

### Conversion/DataflowToSentient — 1 defs, 33 lines

- [ ] `runOnOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014`, 33 lines

## Level 8 — 8 definitions, 292 body lines

### Transform/Dataflow — 2 defs, 106 lines

- [ ] `shallowlyMergeConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198`, 76 lines
- [ ] `simplifyValueBasedConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466`, 30 lines

### Conversion/VectorChainLowering — 3 defs, 96 lines

- [ ] `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975`, 54 lines
- [ ] `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374`, 30 lines
- [ ] `matchAndRewrite` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44`, 12 lines

### Conversion/AgenToSentient — 2 defs, 56 lines

- [ ] `lowerSymbolicVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410`, 30 lines
- [ ] `lowerSymbolicVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380`, 26 lines

### Conversion/StandardToSentient — 1 defs, 34 lines

- [ ] `runOnOperation` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439`, 34 lines

## Level 9 — 2 definitions, 224 body lines

### Conversion/AgenToSentient — 1 defs, 144 lines

- [ ] `fuseLoadOrStoreChainOps` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22`, 144 lines

### Transform/Dataflow — 1 defs, 80 lines

- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`, 80 lines

## Level 10 — 1 definitions, 81 body lines

### Conversion/AgenToSentient — 1 defs, 81 lines

- [ ] `runOnOperation` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169`, 81 lines
