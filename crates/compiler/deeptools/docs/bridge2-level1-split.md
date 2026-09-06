# Level 1 — the 53 gated definitions, and the 14 deferred

Same split as level 0: the corpus has no loop content, so loop-sensitive definitions would be ported
green and untested. See `bridge2-level0-split.md` for the reasoning and the classification caveat.

## Gated — 53 definitions, 1,699 lines

- [ ] `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, 161 lines
- [ ] `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, 159 lines
- [ ] `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, 89 lines
- [ ] `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, 57 lines
- [ ] `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, 35 lines
- [ ] `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, 27 lines
- [ ] `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, 27 lines
- [ ] `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, 17 lines
- [ ] `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, 109 lines
- [ ] `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, 45 lines
- [ ] `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, 40 lines
- [ ] `constructExtentAndTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32`, 64 lines
- [ ] `computeBurstAndGroup` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796`, 36 lines
- [ ] `initializeMemViewInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274`, 16 lines
- [ ] `constructIteratorCoefficients` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406`, 10 lines
- [ ] `constructLdOrStType` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267`, 5 lines
- [ ] `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, 60 lines
- [ ] `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, 17 lines
- [ ] `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, 15 lines
- [ ] `printTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:288`, 13 lines
- [ ] `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, 6 lines
- [ ] `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, 49 lines
- [ ] `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, 37 lines
- [ ] `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, 10 lines
- [ ] `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, 48 lines
- [ ] `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, 17 lines
- [ ] `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, 13 lines
- [ ] `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, 13 lines
- [ ] `getLayoutMapAndIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879`, 40 lines
- [ ] `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, 26 lines
- [ ] `getName` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866`, 11 lines
- [ ] `getOperandFromConstantBitstreamOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299`, 5 lines
- [ ] `getOperandFromNegOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366`, 5 lines
- [ ] `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, 77 lines
- [ ] `isXrfRelated` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531`, 31 lines
- [ ] `areXrfAccessesLegal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98`, 28 lines
- [ ] `insertConstAndAddOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296`, 10 lines
- [ ] `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, 8 lines
- [ ] `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, 60 lines
- [ ] `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, 26 lines
- [ ] `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, 25 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, 42 lines
- [ ] `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, 18 lines
- [ ] `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, 10 lines
- [ ] `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, 6 lines
- [ ] `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, 23 lines
- [ ] `if_op_` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:56`, 20 lines
- [ ] `insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388`, 7 lines
- [ ] `get` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401`, 4 lines
- [ ] `runOnOperation` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:222`, 10 lines
- [ ] `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, 8 lines
- [ ] `isMaskEquivalentToNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123`, 4 lines
- [ ] `getTotalDataOriginsCount` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:33`, 0 lines

## Deferred until tiling lands (task #9) — 14 definitions, 657 lines

- [ ] `calculatePartitionSizes` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862`, 97 lines
- [ ] `constructConditionals` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000`, 59 lines
- [ ] `createExplicitTimeLoops` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282`, 57 lines
- [ ] `getLoopTripCount` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743`, 56 lines
- [ ] `synthesizeTimeInfo` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256`, 22 lines
- [ ] `hasMutableAddrOverflow` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801`, 14 lines
- [ ] `calculateSubscriptsCoefficients` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188`, 11 lines
- [ ] `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, 127 lines
- [ ] `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, 38 lines
- [ ] `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, 35 lines
- [ ] `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, 25 lines
- [ ] `expandAffineApplyOps` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183`, 53 lines
- [ ] `constructIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354`, 50 lines
- [ ] `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, 13 lines
