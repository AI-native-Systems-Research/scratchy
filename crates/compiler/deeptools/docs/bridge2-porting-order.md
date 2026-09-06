# Bridge 2 — the porting order, function by function

`DataflowIR -> SentientIR`, the D1-D28 span. **418 functions, 16,544 body lines** across the
eight subsystems, in dependency order.

## How this was derived, and what it is worth

A textual call graph over `dcc/src/{Transform/Dataflow,Conversion/*}` in the pod extract at
revision `a0d29abbed`: every function definition (signatures matched across line breaks, which
an earlier attempt missed and undercounted by 72%), then an edge wherever one definition's body
names another. 949 internal edges, levelled so that **level 0 calls nothing else in the span**
and level N calls only levels below it.

⭐ **FUNCTIONS AT THE SAME LEVEL NEVER CALL EACH OTHER.** That is the property that makes a level
a safe batch: everything in it can be ported in any order, and it unblocks as soon as the level
below is done.

⛔ **IT IS APPROXIMATE, AND HERE IS HOW IT LIES.** A name-based graph cannot see through C++
overloads (two functions of one name collapse to one node), virtual dispatch (an edge to the
base, never the override), template instantiation, or calls through function pointers and
`std::function`. It also cannot distinguish a call from a same-named variable. Cycles were
broken arbitrarily at the first back edge. So treat a level as a *starting order*, not a proof:
if a port needs something from a higher level, the graph missed an edge, and that is expected
rather than alarming.

⛔ **AND IT SAYS NOTHING ABOUT WHAT IS WORTH PORTING.** Between 26% and 83% of these lines are
MLIR construction — `OpBuilder`, SSA iterators, memoisers, the nested `(core, corelet,
component)` lookup maps — which exists because the C++ builds an IR at run time. We emit from a
resolved tape. **Port the rule, never the plumbing**; the rule content brackets at ~2,500-11,700
lines of the 16,544.

## Level 0 — 167 functions, 965 body lines

### VectorChainLowering — 37 fns, 304 lines

- [ ] `fuseCompareAndSelectIntoMinOrMax` — `VectorChainHelper.cpp:415`, 49 lines
- [ ] `redefineConstantVectors` — `VectorChainHelper.cpp:571`, 37 lines
- [ ] `getForOpBound` — `LoweringXRF.cpp:262`, 31 lines
- [ ] `getVectorBinaryToSentientBinary` — `VectorChainHelper.hpp:32`, 17 lines
- [ ] `eraseOp` — `VectorOperands.cpp:806`, 16 lines
- [ ] `getComputePrecisionOfOp` — `VectorChainHelper.cpp:65`, 13 lines
- [ ] `resetSentientFMAsIfExists` — `VectorChainHelper.cpp:468`, 13 lines
- [ ] `constValToField` — `VectorOperands.cpp:250`, 12 lines
- [ ] `setSentientMacXrfRegIncrAttr` — `LoweringXRF.cpp:665`, 11 lines
- [ ] `sameBlock` — `VectorOperands.cpp:652`, 11 lines
- [ ] `getXrfValue` — `LoweringXRF.cpp:248`, 11 lines
- [ ] `hasConstantBounds` — `VectorChainHelper.cpp:80`, 10 lines
- [ ] `getResultPrecisionFromOperands` — `VectorChainHelper.cpp:52`, 10 lines
- [ ] `computeUnitPrecision` — `VectorChainToSentientPT.cpp:31`, 9 lines
- [ ] `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `VectorChainHelper.hpp:55`, 9 lines
- [ ] `getVectorTernaryToSentientTernary` — `VectorChainHelper.hpp:247`, 7 lines
- [ ] `replaceAndEraseDummyMacOps` — `LoweringXRF.cpp:681`, 7 lines
- [ ] `size` — `VectorChainHelper.cpp:319`, 6 lines
- [ ] `getInputPrecisionFromOperand` — `VectorChainHelper.cpp:36`, 6 lines
- [ ] `reuse_info_` — `VectorChainToSentientPESFP.hpp:82`, 5 lines
- [ ] `findNodeFromOp` — `LoopMaskTree.cpp:167`, 4 lines
- [ ] `isSentientBinaryLogicalOp` — `VectorChainHelper.cpp:30`, 4 lines
- [ ] `getLastChild` — `LoopMaskTree.hpp:48`, 2 lines
- [ ] `setReuseFlag` — `OperandReuse.cpp:91`, 2 lines
- [ ] `dominates` — `OperandReuse.hpp:38`, 2 lines
- [ ] `getFirstValue` — `VectorOperands.hpp:76`, 0 lines
- [ ] `getStartVal` — `LoopMaskTree.hpp:76`, 0 lines
- [ ] `isMaskNode` — `LoopMaskTree.hpp:55`, 0 lines
- [ ] `isLoopNode` — `LoopMaskTree.hpp:54`, 0 lines
- [ ] `getIncrement` — `LoopMaskTree.hpp:77`, 0 lines
- [ ] `MaskNode` — `LoopMaskTree.hpp:74`, 0 lines
- [ ] `size_` — `VectorChainHelper.cpp:208`, 0 lines
- [ ] `LMTLoopNode` — `LoopMaskTree.hpp:94`, 0 lines
- [ ] `LoopMaskNode` — `LoopMaskTree.hpp:34`, 0 lines
- [ ] `setOperation` — `LoopMaskTree.hpp:52`, 0 lines
- [ ] `is_visited_` — `VectorChainToSentientPESFP.hpp:108`, 0 lines
- [ ] `increment_` — `LoopMaskTree.hpp:72`, 0 lines

### Transform/Dataflow — 48 fns, 299 lines

- [ ] `transformSCFToAffineLoop` — `TransformLoopToLegalizeForSentientLowering.cpp:102`, 44 lines
- [ ] `isEligibleForSplitting` — `MutableAddrSplitting.cpp:832`, 20 lines
- [ ] `removeCoresCoreletsFoldsFromProgramUnit` — `UnitFiltering.cpp:240`, 20 lines
- [ ] `partitionUnits` — `FlatteningLocalRegions.cpp:168`, 17 lines
- [ ] `createInequalityCondition` — `TransformPagedMemViewImpl.cpp:263`, 14 lines
- [ ] `createDummyYieldInElseReg` — `CFGSDataflowConditionalTree.cpp:383`, 13 lines
- [ ] `createNewMemViewWithMod` — `MutableAddrSplitting.cpp:966`, 12 lines
- [ ] `isDataTransfer` — `UnitFiltering.cpp:314`, 11 lines
- [ ] `getLhsRhsOfEQPredicate` — `CFGSDataflowConditionalTree.cpp:519`, 11 lines
- [ ] `identifyTimeDimForExplicitLoops` — `TransformPagedMemViewImpl.cpp:963`, 10 lines
- [ ] `createNonPagedMemView` — `TransformPagedMemViewImpl.cpp:545`, 10 lines
- [ ] `printUnitToOpsMap` — `FlatteningLocalRegions.cpp:189`, 10 lines
- [ ] `cloneMemViewIfNonPaged` — `TransformPagedMemViewImpl.cpp:633`, 9 lines
- [ ] `getMaxImmutableRange` — `MutableAddrSplitting.cpp:683`, 8 lines
- [ ] `getMaxMutableRange` — `MutableAddrSplitting.cpp:673`, 8 lines
- [ ] `printEquivalenceClasses` — `FlatteningLocalRegions.cpp:202`, 8 lines
- [ ] `replaceDimsInMapWithSyms` — `TransformPagedMemViewImpl.cpp:47`, 7 lines
- [ ] `removeValuesFromIndices` — `TransformPagedMemViewImpl.cpp:36`, 7 lines
- [ ] `createEqualityCondition` — `TransformPagedMemViewImpl.cpp:253`, 7 lines
- [ ] `getStoreOp` — `TransformPagedMemViewImpl.cpp:843`, 6 lines
- [ ] `setBuilderToInsertRef` — `TransformPagedMemViewImpl.cpp:280`, 6 lines
- [ ] `TransformationConditionalTree` — `CFGSDataflowConditionalTree.hpp:111`, 5 lines
- [ ] `addTimeDimIndicesRanges` — `TransformPagedMemViewImpl.cpp:876`, 5 lines
- [ ] `sortDataBasedOnWeight` — `MutableAddrSplitting.cpp:855`, 4 lines
- [ ] `base_unit_program_` — `ProgramUnitsReduction.cpp:43`, 4 lines
- [ ] `cloneUseChain` — `TransformPagedMemViewImpl.cpp:678`, 3 lines
- [ ] `getRoot` — `FlatteningLocalRegions.cpp:81`, 2 lines
- [ ] `getParentNode` — `FlatteningLocalRegions.cpp:53`, 2 lines
- [ ] `getNextSibling` — `FlatteningLocalRegions.cpp:59`, 2 lines
- [ ] `getFirstChild` — `FlatteningLocalRegions.cpp:56`, 2 lines
- [ ] `getNewDbgNameFromList` — `CFGSDataflowConditionalTree.cpp:456`, 2 lines
- [ ] `isOperationSelected` — `CFGSDataflowConditionalTree.cpp:34`, 2 lines
- [ ] `getPrevSibling` — `FlatteningLocalRegions.cpp:62`, 2 lines
- [ ] `ConditionalSimplificationManager` — `CFGSDataflowConditionalTree.hpp:78`, 2 lines
- [ ] `comp_` — `TransformPagedMemViewImpl.hpp:30`, 2 lines
- [ ] `populateDuplicateReusedTogglePatterns` — `DuplicateReusedToggle.cpp:207`, 2 lines
- [ ] `dccExtContext` — `CanonicalizeToggle.cpp:94`, 0 lines
- [ ] `curr_unit_corelet_id_` — `ProgramUnitsReduction.cpp:91`, 0 lines
- [ ] `TPMVBase` — `TransformPagedMemViewImpl.hpp:389`, 0 lines
- [ ] `opts_` — `CFGSimplificationDataflowLevel.cpp:69`, 0 lines
- [ ] `OperationNode` — `FlatteningLocalRegions.cpp:51`, 0 lines
- [ ] `access_details_` — `TransformPagedMemViewImpl.hpp:458`, 0 lines
- [ ] `TPMVVector` — `TransformPagedMemViewImpl.hpp:397`, 0 lines
- [ ] `subscripts_map_` — `TransformPagedMemViewImpl.hpp:45`, 0 lines
- [ ] `mem_index_` — `MutableAddrSplitting.cpp:94`, 0 lines
- [ ] `TPMVComposite` — `TransformPagedMemViewImpl.hpp:519`, 0 lines
- [ ] `weight_` — `MutableAddrSplitting.cpp:111`, 0 lines
- [ ] `dcc_ext_ctx_` — `CanonicalizeToggle.cpp:47`, 0 lines

### AgenToSentient — 66 fns, 223 lines

- [ ] `setldtype` — `Helper.cpp:1647`, 60 lines
- [ ] `checkIndirectMemViewForExtractOp` — `Helper.cpp:388`, 42 lines
- [ ] `findExtractScalarOp` — `Helper.cpp:514`, 20 lines
- [ ] `insertCopyAndAddStmtsHelper` — `AgenToSentient.hpp:502`, 17 lines
- [ ] `findCandidateForLowering` — `Helper.cpp:2884`, 12 lines
- [ ] `addLoadChainToDeleteList` — `Helper.cpp:2975`, 9 lines
- [ ] `getStoreOpFromLoadStorePattern` — `Helper.cpp:2872`, 8 lines
- [ ] `getLoopNestLevel` — `Helper.cpp:43`, 6 lines
- [ ] `setCoalescedBoundValues` — `AccessDetails.cpp:672`, 5 lines
- [ ] `has` — `AccessDetails.hpp:397`, 2 lines
- [ ] `setLayoutCoeffs` — `AccessDetails.hpp:96`, 2 lines
- [ ] `setTotalElements` — `AccessDetails.hpp:116`, 2 lines
- [ ] `setExtents` — `AccessDetails.hpp:113`, 2 lines
- [ ] `setMemViewStartAddr` — `AccessDetails.hpp:81`, 2 lines
- [ ] `setMemViewLayoutMap` — `AccessDetails.hpp:99`, 2 lines
- [ ] `setStrides` — `AccessDetails.hpp:355`, 2 lines
- [ ] `setElementWidth` — `AccessDetails.hpp:119`, 2 lines
- [ ] `setTransferSet` — `AccessDetails.hpp:122`, 2 lines
- [ ] `setTransferOrder` — `AccessDetails.hpp:125`, 2 lines
- [ ] `setTimeAddrMap` — `AccessDetails.hpp:286`, 2 lines
- [ ] `setExpectedTotalElements` — `AccessDetails.hpp:110`, 2 lines
- [ ] `setIndices` — `AccessDetails.hpp:77`, 2 lines
- [ ] `setTimeSymbols` — `AccessDetails.hpp:289`, 2 lines
- [ ] `setSubscriptsMap` — `AccessDetails.hpp:228`, 2 lines
- [ ] `setMemoryIndex` — `AccessDetails.hpp:93`, 2 lines
- [ ] `setIndicesCoeffDict` — `AccessDetails.hpp:231`, 2 lines
- [ ] `setShuffleMode` — `AccessDetails.hpp:104`, 2 lines
- [ ] `setRotationPosition` — `AccessDetails.hpp:107`, 2 lines
- [ ] `setTimeBounds` — `AccessDetails.hpp:292`, 2 lines
- [ ] `setInterleaveGroupIndex` — `AccessDetails.hpp:299`, 2 lines
- [ ] `setTimeOffsets` — `AccessDetails.hpp:295`, 2 lines
- [ ] `getSubscriptsMap` — `AccessDetails.hpp:221`, 0 lines
- [ ] `getMemRef` — `AccessDetails.hpp:59`, 0 lines
- [ ] `getElementWidth` — `AccessDetails.hpp:69`, 0 lines
- [ ] `getTransferOrder` — `AccessDetails.hpp:63`, 0 lines
- [ ] `getTransferSet` — `AccessDetails.hpp:89`, 0 lines
- [ ] `getOp` — `AccessDetails.hpp:57`, 0 lines
- [ ] `getTotalElements` — `AccessDetails.hpp:68`, 0 lines
- [ ] `getMemViewLayoutMap` — `AccessDetails.hpp:61`, 0 lines
- [ ] `getExpectedTotalElements` — `AccessDetails.hpp:88`, 0 lines
- [ ] `getComp` — `AccessDetails.hpp:58`, 0 lines
- [ ] `setMemRef` — `AccessDetails.hpp:80`, 0 lines
- [ ] `setMemory` — `AccessDetails.hpp:84`, 0 lines
- [ ] `setTimeSet` — `AccessDetails.hpp:285`, 0 lines
- [ ] `getTimeAddrMap` — `AccessDetails.hpp:265`, 0 lines
- [ ] `getTimeOrder` — `AccessDetails.hpp:262`, 0 lines
- [ ] `getTimeSet` — `AccessDetails.hpp:263`, 0 lines
- [ ] `setTimeOrder` — `AccessDetails.hpp:284`, 0 lines
- [ ] `getMemoryIndex` — `AccessDetails.hpp:72`, 0 lines
- [ ] `setLdOrStSize` — `AccessDetails.hpp:128`, 0 lines
- [ ] `getChunkSize` — `AccessDetails.hpp:64`, 0 lines
- [ ] `setChunkStride` — `AccessDetails.hpp:103`, 0 lines
- [ ] `getRotationPosition` — `AccessDetails.hpp:67`, 0 lines
- [ ] `setChunkSize` — `AccessDetails.hpp:102`, 0 lines
- [ ] `getLdOrStSize` — `AccessDetails.hpp:90`, 0 lines
- [ ] `getBurstIndex` — `AccessDetails.hpp:268`, 0 lines
- [ ] `setBurstIndex` — `AccessDetails.hpp:298`, 0 lines
- [ ] `setOp` — `AccessDetails.hpp:76`, 0 lines
- [ ] `getMemory` — `AccessDetails.hpp:71`, 0 lines
- [ ] `getMemViewStartAddr` — `AccessDetails.hpp:60`, 0 lines
- [ ] `getInterleaveGroupIndex` — `AccessDetails.hpp:269`, 0 lines
- [ ] `getShuffleMode` — `AccessDetails.hpp:66`, 0 lines
- [ ] `getChunkStride` — `AccessDetails.hpp:65`, 0 lines
- [ ] `index_mapping_` — `AccessDetails.hpp:375`, 0 lines
- [ ] `AccessDetailsBase` — `AccessDetails.hpp:218`, 0 lines
- [ ] `AccessDetailsAffine` — `AccessDetails.hpp:259`, 0 lines

### DataflowToSentient — 6 fns, 58 lines

- [ ] `lowerOpaqueOperation` — `DataflowToSentient.cpp:1984`, 27 lines
- [ ] `ExtendUnitNameToCorelet` — `DataflowToSentient.cpp:104`, 11 lines
- [ ] `isSameListOfUnits` — `DataflowToSentient.cpp:175`, 10 lines
- [ ] `isTargetL3` — `DataflowToSentient.cpp:1720`, 6 lines
- [ ] `isSenComponentL0LU` — `DataflowToSentient.cpp:96`, 2 lines
- [ ] `isSenComponentL0SU` — `DataflowToSentient.cpp:100`, 2 lines

### StandardToSentient — 6 fns, 51 lines

- [ ] `LowerConstantIntToSentient` — `StandardToSentient.cpp:358`, 15 lines
- [ ] `LowerSubIOpToSentient` — `StandardToSentient.cpp:90`, 10 lines
- [ ] `LowerMulIOpToSentient` — `StandardToSentient.cpp:102`, 9 lines
- [ ] `LowerAddIOpToSentient` — `StandardToSentient.cpp:79`, 9 lines
- [ ] `LowerConstantIndexToSentient` — `StandardToSentient.cpp:347`, 8 lines
- [ ] `If` — `StandardToSentient.cpp:159`, 0 lines

### SCFToSentient — 2 fns, 16 lines

- [ ] `getSentientCmpIPredicate` — `SCFToSentient.cpp:33`, 16 lines
- [ ] `ConversionPattern` — `SCFToSentient.cpp:70`, 0 lines

### AffineToStandard — 2 fns, 14 lines

- [ ] `populateAffineToStdConversionPatterns` — `AffineToStandard.cpp:198`, 8 lines
- [ ] `populateAffineToVectorConversionPatterns` — `AffineToStandard.cpp:208`, 6 lines

## Level 1 — 59 functions, 2033 body lines

### Transform/Dataflow — 28 fns, 1049 lines

- [ ] `analyzeLoop` — `TransformLoopToLegalizeForSentientLowering.cpp:268`, 127 lines
- [ ] `calculatePartitionSizes` — `MutableAddrSplitting.cpp:862`, 97 lines
- [ ] `transformSCFLoopWithNonConstantUpperBound` — `TransformLoopToLegalizeForSentientLowering.cpp:169`, 94 lines
- [ ] `matchUnits` — `ProgramUnitsReduction.cpp:69`, 77 lines
- [ ] `mergeShallow` — `CFGSDataflowConditionalTree.cpp:398`, 60 lines
- [ ] `cloneOpsForRegions` — `FlatteningLocalRegions.cpp:223`, 60 lines
- [ ] `constructConditionals` — `MutableAddrSplitting.cpp:1000`, 59 lines
- [ ] `createExplicitTimeLoops` — `MutableAddrSplitting.cpp:1282`, 57 lines
- [ ] `getLoopTripCount` — `MutableAddrSplitting.cpp:743`, 56 lines
- [ ] `createConditionsForHyperRectSubscripts` — `TransformPagedMemViewImpl.cpp:289`, 48 lines
- [ ] `removeCoresCoreletsFoldsFromDefImmutMap` — `UnitFiltering.cpp:99`, 37 lines
- [ ] `createConditionsForNonHyperRectSubscripts` — `TransformPagedMemViewImpl.cpp:342`, 35 lines
- [ ] `calculateDimWeights` — `MutableStartAddrShifting.cpp:560`, 26 lines
- [ ] `calculateFullShift` — `MutableStartAddrShifting.cpp:462`, 25 lines
- [ ] `calculateIndicesRanges` — `TransformPagedMemViewImpl.cpp:56`, 25 lines
- [ ] `synthesizeTimeInfo` — `MutableAddrSplitting.cpp:1256`, 22 lines
- [ ] `if_op_` — `CFGSDataflowConditionalTree.hpp:56`, 20 lines
- [ ] `traverseRegion` — `FlatteningLocalRegions.cpp:129`, 17 lines
- [ ] `updateTPMVInfo` — `TransformPagedMemViewImpl.cpp:382`, 17 lines
- [ ] `clear` — `FlatteningLocalRegions.cpp:112`, 15 lines
- [ ] `hasMutableAddrOverflow` — `MutableAddrSplitting.cpp:801`, 14 lines
- [ ] `opHasSideEffect` — `CFGSDataflowConditionalTree.cpp:38`, 13 lines
- [ ] `printTree` — `FlatteningLocalRegions.cpp:288`, 13 lines
- [ ] `isDataTransferToKeep` — `UnitFiltering.cpp:327`, 10 lines
- [ ] `createNewMemOp` — `TransformPagedMemViewImpl.cpp:684`, 9 lines
- [ ] `performFullUnroll` — `LoopUnrollForShuffleOp.cpp:141`, 7 lines
- [ ] `inRegionEmpty` — `FlatteningLocalRegions.cpp:214`, 6 lines
- [ ] `eraseMemOpAndUseChain` — `TransformPagedMemViewImpl.cpp:698`, 3 lines

### AgenToSentient — 16 fns, 654 lines

- [ ] `constructSetActiveMaskValueOp` — `Helper.cpp:2567`, 161 lines
- [ ] `checkCompositeRegion` — `Helper.cpp:206`, 89 lines
- [ ] `constructExtentAndTotalElements` — `AccessDetails.cpp:32`, 64 lines
- [ ] `createUniformizeRegionsOp` — `Helper.cpp:3880`, 57 lines
- [ ] `constructIndices` — `AccessDetails.cpp:354`, 50 lines
- [ ] `generateSetSendDestinationStmts` — `Helper.cpp:2731`, 49 lines
- [ ] `computeBurstAndGroup` — `AccessDetails.cpp:796`, 36 lines
- [ ] `updateSymbolicAccessDetails` — `Helper.cpp:1013`, 35 lines
- [ ] `checkStoreOpFromExtractPattern` — `Helper.cpp:433`, 27 lines
- [ ] `isLoadAndExtractScalarPattern` — `Helper.cpp:463`, 27 lines
- [ ] `isReceiveAndExtractScalarPattern` — `Helper.cpp:493`, 17 lines
- [ ] `initializeMemViewInfo` — `AccessDetails.cpp:274`, 16 lines
- [ ] `constructIteratorCoefficients` — `AccessDetails.cpp:406`, 10 lines
- [ ] `insert` — `AccessDetails.hpp:388`, 7 lines
- [ ] `constructLdOrStType` — `AccessDetails.cpp:267`, 5 lines
- [ ] `get` — `AccessDetails.hpp:401`, 4 lines

### VectorChainLowering — 12 fns, 296 lines

- [ ] `getMergeTypeFromIndices` — `VectorChainHelper.cpp:301`, 109 lines
- [ ] `checkValidityOfPackAndShuffleLowering` — `VectorChainHelper.cpp:140`, 45 lines
- [ ] `getLayoutMapAndIndices` — `VectorOperands.cpp:879`, 40 lines
- [ ] `isXrfRelated` — `LoweringXRF.cpp:531`, 31 lines
- [ ] `areXrfAccessesLegal` — `LoweringXRF.cpp:98`, 28 lines
- [ ] `getName` — `VectorOperands.cpp:866`, 11 lines
- [ ] `insertConstAndAddOps` — `LoweringXRF.cpp:296`, 10 lines
- [ ] `insertIfNotExists` — `OperandReuse.cpp:81`, 8 lines
- [ ] `updateYieldArgs` — `LoweringXRF.cpp:690`, 8 lines
- [ ] `isMaskEquivalentToNode` — `LoopMaskTree.cpp:123`, 4 lines
- [ ] `walk` — `LoopMaskTree.cpp:132`, 2 lines
- [ ] `getTotalDataOriginsCount` — `OperandReuse.hpp:33`, 0 lines

### DataflowToSentient — 3 fns, 34 lines

- [ ] `separateBasedOnDestinationUnits` — `DataflowToSentient.cpp:761`, 18 lines
- [ ] `getUnitNameFromAListOfGetUnitOp` — `DataflowToSentient.cpp:119`, 10 lines
- [ ] `areCoreletsDifferent` — `DataflowToSentient.cpp:132`, 6 lines

## Level 2 — 66 functions, 3086 body lines

### AgenToSentient — 17 fns, 1028 lines

- [ ] `constructChunkAndShuffleInfo` — `AccessDetails.cpp:98`, 167 lines
- [ ] `checkBasicConditions` — `Helper.cpp:58`, 133 lines
- [ ] `adjustMutableAddrInitForIndirect` — `Helper.cpp:1447`, 127 lines
- [ ] `constructReceiveAndExtractScalarOp` — `Helper.cpp:2471`, 91 lines
- [ ] `processInterleaveOp` — `Helper.cpp:305`, 80 lines
- [ ] `cloneStartAddrOutsideLoop` — `Helper.cpp:3942`, 79 lines
- [ ] `gatherAffineLoadStoreDetails` — `Helper.cpp:538`, 74 lines
- [ ] `setsttype` — `Helper.cpp:1731`, 50 lines
- [ ] `adjustMutableAddrInitForStride` — `Helper.cpp:4034`, 47 lines
- [ ] `lowerVectorLoadHelper` — `Helper.cpp:2899`, 46 lines
- [ ] `setImmutableAddrAndIncrements` — `Helper.cpp:1581`, 43 lines
- [ ] `cleanupTriviallyRedundantSetSendDestination` — `Helper.cpp:4084`, 38 lines
- [ ] `constructImmutableAddress` — `Helper.cpp:1217`, 16 lines
- [ ] `insertCopyAndAddStmts` — `Helper.cpp:3861`, 13 lines
- [ ] `lowerSetTransferMaskStateOp` — `Helper.cpp:3815`, 9 lines
- [ ] `emplace_insert` — `AccessDetails.hpp:378`, 8 lines
- [ ] `getFirst` — `AccessDetails.hpp:411`, 7 lines

### Transform/Dataflow — 22 fns, 874 lines

- [ ] `fillPartitions` — `MutableAddrSplitting.cpp:1063`, 117 lines
- [ ] `removeCoresCoreletsFoldsFromUniformizeRegion` — `UnitFiltering.cpp:139`, 98 lines
- [ ] `processComputeUnit` — `LoopUnrollingForPTLRFRegs.cpp:37`, 91 lines
- [ ] `createForOpWithAdditionalReturnValue` — `Utils.cpp:28`, 66 lines
- [ ] `isLoopInvariant` — `CFGSDataflowConditionalTree.cpp:681`, 66 lines
- [ ] `enumerateCollectionUnit` — `EnumerateCollectionUnit.cpp:34`, 57 lines
- [ ] `expandAffineApplyOps` — `LoopUnrollForShuffleOp.cpp:183`, 53 lines
- [ ] `adjustForEvenImmutableAddr` — `MutableAddrSplitting.cpp:1204`, 47 lines
- [ ] `transformLoop` — `TransformLoopToLegalizeForSentientLowering.cpp:399`, 38 lines
- [ ] `gatherPageDependentDimsForPage` — `TransformPagedMemViewImpl.cpp:927`, 32 lines
- [ ] `initMASData` — `MutableAddrSplitting.cpp:707`, 31 lines
- [ ] `topLevelConditionsMatch` — `CFGSDataflowConditionalTree.cpp:279`, 31 lines
- [ ] `applyShifts` — `MutableStartAddrShifting.cpp:616`, 27 lines
- [ ] `offsetShifts` — `MutableStartAddrShifting.cpp:590`, 22 lines
- [ ] `addConstraintsForIVRanges` — `TransformPagedMemViewImpl.cpp:86`, 22 lines
- [ ] `removeAncestors` — `UnitFiltering.cpp:339`, 18 lines
- [ ] `singleOpBranchToYieldVal` — `CFGSDataflowConditionalTree.cpp:498`, 18 lines
- [ ] `compute` — `FlatteningLocalRegions.cpp:151`, 15 lines
- [ ] `createNewSubscriptsFromStartElements` — `TransformPagedMemViewImpl.cpp:562`, 10 lines
- [ ] `isHoistable` — `CFGSDataflowConditionalTree.cpp:82`, 9 lines
- [ ] `setupForPartitioning` — `MutableAddrSplitting.cpp:819`, 6 lines
- [ ] `FlatteningLocalRegionsTree` — `FlatteningLocalRegions.cpp:79`, 0 lines

### VectorChainLowering — 21 fns, 832 lines

- [ ] `insertPTMaskOps` — `LoweringPTMasks.cpp:41`, 164 lines
- [ ] `lowerDanglingNonComputeOpsPESFP` — `VectorChainToSentientPESFP.cpp:1274`, 93 lines
- [ ] `analyzeNonComputeOpsForFusion` — `VectorChainHelper.cpp:610`, 86 lines
- [ ] `getLayoutExpr` — `LoweringXRF.cpp:28`, 67 lines
- [ ] `createIfOpWithReturnValue` — `LoweringXRF.cpp:189`, 56 lines
- [ ] `createForOpWithReturnValue` — `LoweringXRF.cpp:129`, 56 lines
- [ ] `validateLoweringAndSetMissingParameters` — `VectorChainHelper.cpp:483`, 49 lines
- [ ] `eraseOperands` — `VectorOperands.cpp:690`, 46 lines
- [ ] `createSentientConstants` — `Splat.cpp:34`, 31 lines
- [ ] `analyzeAndFillResultForwarding` — `VectorChainHelper.hpp:162`, 27 lines
- [ ] `analyzeAndFillOperandForwarding` — `VectorChainHelper.cpp:535`, 25 lines
- [ ] `convertStringToType` — `VectorChainHelper.hpp:196`, 24 lines
- [ ] `print` — `LoopMaskTree.cpp:96`, 22 lines
- [ ] `convertTypeToString` — `VectorChainHelper.hpp:223`, 22 lines
- [ ] `computeLoops` — `LoopMaskTree.cpp:173`, 18 lines
- [ ] `addMaskNode` — `LoopMaskTree.cpp:136`, 16 lines
- [ ] `insertDummyMacOp` — `LoweringXRF.cpp:312`, 15 lines
- [ ] `updateNode` — `LoopMaskTree.cpp:155`, 10 lines
- [ ] `setValue` — `VectorOperands.hpp:72`, 3 lines
- [ ] `dominance_info_` — `OperandReuse.hpp:26`, 2 lines
- [ ] `OperandReuse` — `OperandReuse.hpp:30`, 0 lines

### StandardToSentient — 2 fns, 168 lines

- [ ] `ConstructIFRecursively` — `StandardToSentient.cpp:113`, 125 lines
- [ ] `SimplifyOrIOp` — `StandardToSentient.cpp:389`, 43 lines

### DataflowToSentient — 3 fns, 123 lines

- [ ] `lowerL3SyncOperationForAGroupOfUnits` — `DataflowToSentient.cpp:667`, 61 lines
- [ ] `lowerL3SyncOperationForAUnit` — `DataflowToSentient.cpp:375`, 57 lines
- [ ] `pushBackTheUnitToListIfDoesnotExist` — `DataflowToSentient.cpp:143`, 5 lines

### SymbolToSentient — 1 fns, 61 lines

- [ ] `createIfOpFromMapping` — `SymbolToSentient.cpp:123`, 61 lines

## Level 3 — 35 functions, 2226 body lines

### AgenToSentient — 10 fns, 731 lines

- [ ] `constructLoadAndStoreStmt` — `Helper.cpp:2167`, 177 lines
- [ ] `gatherSymbolicLoadStoreDetails` — `Helper.cpp:1051`, 153 lines
- [ ] `constructLoadAndExtractScalarOp` — `Helper.cpp:2351`, 114 lines
- [ ] `coalesceTimeDimensions` — `AccessDetails.cpp:681`, 112 lines
- [ ] `constructTimeLoopsAndVectorOperations` — `Helper.cpp:1789`, 109 lines
- [ ] `lowerCompositeMemoryInterleaveOp` — `Helper.cpp:3775`, 37 lines
- [ ] `insertInitializationStmt` — `Helper.cpp:3834`, 13 lines
- [ ] `addStoreInputToDeleteList` — `Helper.cpp:2987`, 8 lines
- [ ] `constructReceiveAndStoreStmt` — `AgenToSentient.hpp:248`, 4 lines
- [ ] `constructLoadAndSendStmt` — `AgenToSentient.hpp:230`, 4 lines

### Transform/Dataflow — 14 fns, 671 lines

- [ ] `createIterArgsForConditionals` — `TransformPagedMemViewImpl.cpp:401`, 121 lines
- [ ] `hoistCommonConditionals` — `CFGSDataflowConditionalTree.cpp:94`, 96 lines
- [ ] `parseConditional` — `CFGSDataflowConditionalTree.cpp:534`, 83 lines
- [ ] `flatten` — `FlatteningLocalRegions.cpp:384`, 70 lines
- [ ] `calculatePartialShift` — `MutableStartAddrShifting.cpp:490`, 66 lines
- [ ] `replaceIfOpByIterArg` — `CFGSDataflowConditionalTree.cpp:619`, 56 lines
- [ ] `cleanup` — `UnitFiltering.cpp:263`, 49 lines
- [ ] `hoistLoopInvariantConditionals` — `CFGSDataflowConditionalTree.cpp:750`, 42 lines
- [ ] `areShallowlyMergeable` — `CFGSDataflowConditionalTree.cpp:343`, 34 lines
- [ ] `getPageValidity` — `TransformPagedMemViewImpl.cpp:114`, 33 lines
- [ ] `createPartitions` — `MutableAddrSplitting.cpp:983`, 10 lines
- [ ] `initialize` — `MutableAddrSplitting.cpp:694`, 8 lines
- [ ] `analyzeAndTransform` — `TransformLoopToLegalizeForSentientLowering.cpp:441`, 3 lines
- [ ] `OperationTreeBase` — `FlatteningLocalRegions.cpp:78`, 0 lines

### DataflowToSentient — 2 fns, 402 lines

- [ ] `lowerL0LXSyncOperationForAGroupOfUnits` — `DataflowToSentient.cpp:438`, 222 lines
- [ ] `lowerL0LXSyncOperationForAUnit` — `DataflowToSentient.cpp:189`, 180 lines

### VectorChainLowering — 6 fns, 316 lines

- [ ] `createSplatOperation` — `Splat.cpp:70`, 110 lines
- [ ] `getGCVTorFCVTTypeFromIndicesAndCastInputs` — `VectorChainHelper.cpp:188`, 108 lines
- [ ] `fillOpInfo` — `VectorChainToSentientPESFP.cpp:1070`, 83 lines
- [ ] `updateLoopMaskTreeForDynamicMask` — `LoweringPTMasks.cpp:28`, 9 lines
- [ ] `updateLoopMaskTreeForConstantMask` — `LoweringPTMasks.cpp:20`, 4 lines
- [ ] `op_` — `VectorOperands.hpp:79`, 2 lines

### SymbolToSentient — 1 fns, 68 lines

- [ ] `LowerSymbolQueryMap` — `SymbolToSentient.cpp:40`, 68 lines

### StandardToSentient — 2 fns, 38 lines

- [ ] `LowerSelectOpToSentient` — `StandardToSentient.cpp:243`, 19 lines
- [ ] `LowerLogicalOpToSentient` — `StandardToSentient.cpp:264`, 19 lines

## Level 4 — 16 functions, 1998 body lines

### DataflowToSentient — 3 fns, 943 lines

- [ ] `lowerSyncLXL3ToLXL3` — `DataflowToSentient.cpp:787`, 928 lines
- [ ] `lowerSyncForAGroup` — `DataflowToSentient.cpp:746`, 8 lines
- [ ] `lowerSyncForAUnit` — `DataflowToSentient.cpp:733`, 7 lines

### AgenToSentient — 4 fns, 456 lines

- [ ] `generateAffineAddressManipulationStmts` — `Helper.cpp:625`, 381 lines
- [ ] `constructTimeStepsInfo` — `AccessDetails.cpp:626`, 42 lines
- [ ] `constructDetails` — `AccessDetails.cpp:418`, 18 lines
- [ ] `lowerAffineCompositeHelper` — `Helper.cpp:2953`, 15 lines

### VectorChainLowering — 3 fns, 336 lines

- [ ] `patternAgnosticFuseNonComputeOpsHelper` — `VectorChainToSentientPESFP.cpp:96`, 222 lines
- [ ] `lowerDanglingNonComputeOps` — `VectorChainToSentientPT.cpp:882`, 88 lines
- [ ] `fuseComputeOps` — `VectorChainToSentientPESFP.cpp:1243`, 26 lines

### Transform/Dataflow — 6 fns, 263 lines

- [ ] `shallowlyMergeConditionals` — `CFGSDataflowConditionalTree.cpp:198`, 76 lines
- [ ] `calculateShifts` — `MutableStartAddrShifting.cpp:396`, 61 lines
- [ ] `constructValidPage` — `TransformPagedMemViewImpl.cpp:190`, 58 lines
- [ ] `analyzeValidPages` — `TransformPagedMemViewImpl.cpp:888`, 33 lines
- [ ] `simplifyValueBasedConditionals` — `CFGSDataflowConditionalTree.cpp:466`, 30 lines
- [ ] `runOn` — `TransformLoopToLegalizeForSentientLowering.cpp:447`, 5 lines

## Level 5 — 10 functions, 653 body lines

### Transform/Dataflow — 5 fns, 342 lines

- [ ] `matchAndRewrite` — `DuplicateReusedToggle.cpp:33`, 170 lines
- [ ] `transform_time` — `TransformPagedMemViewImpl.cpp:977`, 98 lines
- [ ] `analyzeAndConstructValidPages` — `TransformPagedMemViewImpl.cpp:152`, 32 lines
- [ ] `initialize_time` — `TransformPagedMemViewImpl.cpp:1095`, 23 lines
- [ ] `shiftMutableAddr` — `MutableStartAddrShifting.cpp:366`, 19 lines

### DataflowToSentient — 1 fns, 166 lines

- [ ] `lowerSyncForAQueryMap` — `DataflowToSentient.cpp:1728`, 166 lines

### VectorChainLowering — 1 fns, 80 lines

- [ ] `fuseNonComputeOps` — `VectorChainToSentientPESFP.cpp:1159`, 80 lines

### AgenToSentient — 3 fns, 65 lines

- [ ] `constructAffineCompDetailsAndAddrs` — `Helper.cpp:2809`, 33 lines
- [ ] `constructAffineDetailsAndAddrs` — `Helper.cpp:2787`, 16 lines
- [ ] `constructSymbolicDetailsAndAddrs` — `Helper.cpp:2849`, 16 lines

## Level 6 — 22 functions, 1387 body lines

### AgenToSentient — 15 fns, 713 lines

- [ ] `lowerLDCVTIPattern` — `Helper.cpp:3444`, 326 lines
- [ ] `lowerIndirectVectorStoreOp` — `Helper.cpp:3217`, 46 lines
- [ ] `lowerIndirectVectorLoadOp` — `Helper.cpp:3169`, 44 lines
- [ ] `lowerCompositeIndirectLoadOp` — `Helper.cpp:3267`, 42 lines
- [ ] `lowerCompositeIndirectStoreOp` — `Helper.cpp:3313`, 42 lines
- [ ] `lowerSymbolicVectorStoreOp` — `Helper.cpp:3410`, 30 lines
- [ ] `lowerVectorStoreOp` — `Helper.cpp:3074`, 28 lines
- [ ] `lowerSymbolicVectorLoadOp` — `Helper.cpp:3380`, 26 lines
- [ ] `lowerExtractVectorLoadOp` — `Helper.cpp:3001`, 21 lines
- [ ] `lowerExtractVectorStoreOp` — `Helper.cpp:3026`, 20 lines
- [ ] `lowerVectorLoadOp` — `Helper.cpp:3050`, 20 lines
- [ ] `lowerCompositeIndirectLoadAndStoreOp` — `Helper.cpp:3359`, 17 lines
- [ ] `lowerCompositeStoreOp` — `Helper.cpp:3127`, 17 lines
- [ ] `lowerCompositeLoadOp` — `Helper.cpp:3106`, 17 lines
- [ ] `lowerCompositeLoadAndStoreOp` — `Helper.cpp:3148`, 17 lines

### Transform/Dataflow — 5 fns, 404 lines

- [ ] `transformCompIndLoadAndStore` — `MutableAddrSplitting.cpp:546`, 124 lines
- [ ] `transformCompLoadAndStore` — `MutableAddrSplitting.cpp:451`, 92 lines
- [ ] `transformVectorLoad` — `MutableAddrSplitting.cpp:298`, 75 lines
- [ ] `transformVectorStore` — `MutableAddrSplitting.cpp:375`, 74 lines
- [ ] `transform` — `TransformPagedMemViewImpl.cpp:592`, 39 lines

### VectorChainLowering — 1 fns, 190 lines

- [ ] `processXrfPtrPerUnit` — `LoweringXRF.cpp:337`, 190 lines

### DataflowToSentient — 1 fns, 80 lines

- [ ] `lowerSyncOperation` — `DataflowToSentient.cpp:1901`, 80 lines

## Level 7 — 3 functions, 247 body lines

### AgenToSentient — 1 fns, 144 lines

- [ ] `fuseLoadOrStoreChainOps` — `AgenToSentient.cpp:22`, 144 lines

### VectorChainLowering — 1 fns, 97 lines

- [ ] `createXrfIndexModifOps` — `LoweringXRF.cpp:564`, 97 lines

### Transform/Dataflow — 1 fns, 6 lines

- [ ] `run` — `TransformPagedMemViewImpl.cpp:647`, 6 lines

## Level 8 — 1 functions, 80 body lines

### Transform/Dataflow — 1 fns, 80 lines

- [ ] `runOnOperation` — `CFGSimplificationDataflowLevel.cpp:77`, 80 lines
