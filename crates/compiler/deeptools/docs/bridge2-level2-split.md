# Level 2 — the 47 gated definitions, and the 14 deferred

Same loop-sensitivity split as levels 0 and 1; see `bridge2-level0-split.md` for the reasoning.

## Gated — 47 definitions, 1,791 lines

- [ ] `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, 133 lines
- [ ] `constructReceiveAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471`, 91 lines
- [ ] `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, 80 lines
- [ ] `setsttype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731`, 50 lines
- [ ] `lowerVectorLoadHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899`, 46 lines
- [ ] `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, 43 lines
- [ ] `cleanupTriviallyRedundantSetSendDestination` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084`, 38 lines
- [ ] `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, 16 lines
- [ ] `lowerSetTransferMaskStateOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815`, 9 lines
- [ ] `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, 167 lines
- [ ] `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, 57 lines
- [ ] `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, 61 lines
- [ ] `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, 57 lines
- [ ] `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, 18 lines
- [ ] `pushBackTheUnitToListIfDoesnotExist` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143`, 5 lines
- [ ] `getOperandFromLoadOrStoreOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153`, 94 lines
- [ ] `eraseOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690`, 46 lines
- [ ] `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, 98 lines
- [ ] `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, 18 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, 76 lines
- [ ] `createIfOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189`, 56 lines
- [ ] `insertDummyMacOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312`, 15 lines
- [ ] `print` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:96`, 22 lines
- [ ] `computeLoops` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173`, 18 lines
- [ ] `addMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136`, 16 lines
- [ ] `updateNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155`, 10 lines
- [ ] `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, 32 lines
- [ ] `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, 22 lines
- [ ] `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, 10 lines
- [ ] `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, 61 lines
- [ ] `enumerateCollectionUnit` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34`, 57 lines
- [ ] `convertStringToType` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196`, 24 lines
- [ ] `convertTypeToString` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223`, 22 lines
- [ ] `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, 9 lines
- [ ] `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, 27 lines
- [ ] `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, 22 lines
- [ ] `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, 49 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, 41 lines
- [ ] `createSentientConstants` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34`, 31 lines
- [ ] `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, 15 lines
- [ ] `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, 0 lines
- [ ] `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, 8 lines
- [ ] `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, 7 lines
- [ ] `isHoistable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82`, 9 lines
- [ ] `setValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72`, 3 lines
- [ ] `dominance_info_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:26`, 2 lines
- [ ] `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, 0 lines

## Deferred until tiling lands (task #9) — 14 definitions, 938 lines

- [ ] `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, 117 lines
- [ ] `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, 47 lines
- [ ] `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, 31 lines
- [ ] `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, 6 lines
- [ ] `insertPTMaskOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41`, 164 lines
- [ ] `cloneStartAddrOutsideLoop` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942`, 79 lines
- [ ] `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, 74 lines
- [ ] `getLayoutExpr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28`, 67 lines
- [ ] `createForOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129`, 56 lines
- [ ] `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, 91 lines
- [ ] `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, 69 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, 68 lines
- [ ] `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, 66 lines
- [ ] `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, 3 lines
