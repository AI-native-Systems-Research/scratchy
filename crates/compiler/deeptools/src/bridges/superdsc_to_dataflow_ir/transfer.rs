//! THE TRANSFER STATEMENTS — load, store and send, and how each is walked over the AGEN time axis.
//! Carries the 2B/16B store shuffles, the constant bit streams, the burst and interleave settings.
//!
//! 29 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e034_constructLogicalMemoryViewOp` | 0 | 32 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:94` |
//! | `e035_constructTimeOrder` | 0 | 12 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:206` |
//! | `e036_constructTimeAddressMap` | 0 | 29 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:224` |
//! | `e037_getImmediateParentWithMatchingDim` | 0 | 33 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:670` |
//! | `e038_areEpiloguesInTransferSizes` | 0 | 12 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1491` |
//! | `e039_construct2B16BLoadShuffle` | 0 | 86 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1778` |
//! | `e040_construct2B16BStoreShuffle` | 0 | 87 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1865` |
//! | `e063_getLabeledDsType` | 1 | 21 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:28` |
//! | `e064_getBufferingOrStreamingMode` | 1 | 39 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:50` |
//! | `e065_constructBaseAddress` | 1 | 73 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:132` |
//! | `e066_constructTimeSet` | 1 | 47 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:260` |
//! | `e067_constructLoadOrStoreSet` | 1 | 90 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:314` |
//! | `e068_constructImplicitLoopsForContiguousTransfer` | 1 | 137 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:707` |
//! | `e069_areEpiloguesInLoops` | 1 | 24 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1464` |
//! | `e070_GenerateConstantBitStreamAndShuffle` | 1 | 41 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2475` |
//! | `e077_constructElementsOfAgenDataTransfer` | 2 | 68 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:412` |
//! | `e078_constructElementsOfAgenCompositeDataTransfer` | 2 | 94 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:488` |
//! | `e079_constructElementsOfAffineDataTransferViaAgenTransfer` | 2 | 77 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:590` |
//! | `e080_constructStreamingOrDoubleBufferingLoad` | 2 | 347 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:851` |
//! | `e081_GenerateReceiveAndSendFromDataTransferNode` | 2 | 66 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1395` |
//! | `e082_ConstructSAMVOperation` | 2 | 96 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1957` |
//! | `e089_constructStreamingOrDoubleBufferingStore` | 3 | 185 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1205` |
//! | `e090_GenerateLoadAndSendFromDataTransferNode` | 3 | 274 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2058` |
//! | `e091_GenerateLoadAndStoreFromDataTransferNode` | 3 | 132 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2337` |
//! | `e092_GenerateDataTransfersForViaIfSo` | 3 | 52 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2617` |
//! | `e097_GenerateReceiveAndStoreFromDataTransferNode` | 4 | 269 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1508` |
//! | `e098_GenerateDataTranferForSrc` | 4 | 34 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2522` |
//! | `e102_GenerateDataTranferForDst` | 5 | 47 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2563` |
//! | `e104_constructDataTransfer` | 6 | 164 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2676` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e034_constructLogicalMemoryViewOp
// crustify:todo: e035_constructTimeOrder
// crustify:todo: e036_constructTimeAddressMap
// crustify:todo: e037_getImmediateParentWithMatchingDim
// crustify:todo: e038_areEpiloguesInTransferSizes
// crustify:todo: e039_construct2B16BLoadShuffle
// crustify:todo: e040_construct2B16BStoreShuffle
// crustify:todo: e063_getLabeledDsType
// crustify:todo: e064_getBufferingOrStreamingMode
// crustify:todo: e065_constructBaseAddress
// crustify:todo: e066_constructTimeSet
// crustify:todo: e067_constructLoadOrStoreSet
// crustify:todo: e068_constructImplicitLoopsForContiguousTransfer
// crustify:todo: e069_areEpiloguesInLoops
// crustify:todo: e070_GenerateConstantBitStreamAndShuffle
// crustify:todo: e077_constructElementsOfAgenDataTransfer
// crustify:todo: e078_constructElementsOfAgenCompositeDataTransfer
// crustify:todo: e079_constructElementsOfAffineDataTransferViaAgenTransfer
// crustify:todo: e080_constructStreamingOrDoubleBufferingLoad
// crustify:todo: e081_GenerateReceiveAndSendFromDataTransferNode
// crustify:todo: e082_ConstructSAMVOperation
// crustify:todo: e089_constructStreamingOrDoubleBufferingStore
// crustify:todo: e090_GenerateLoadAndSendFromDataTransferNode
// crustify:todo: e091_GenerateLoadAndStoreFromDataTransferNode
// crustify:todo: e092_GenerateDataTransfersForViaIfSo
// crustify:todo: e097_GenerateReceiveAndStoreFromDataTransferNode
// crustify:todo: e098_GenerateDataTranferForSrc
// crustify:todo: e102_GenerateDataTranferForDst
// crustify:todo: e104_constructDataTransfer
