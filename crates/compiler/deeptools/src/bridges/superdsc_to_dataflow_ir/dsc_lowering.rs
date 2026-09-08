//! THE PER-COMPONENT LOWERING — unit, corelet and core identity, and the component handler.
//! ⭐ ONE ProgramUnitOp SPANS THE SET: handles = cores x corelets x num_folds. The component-to-handler
//! map is CLEARED per unit; the unit-to-value map is module-wide.
//!
//! 12 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e022_retrieveGetUnitOpInSameCore` | 0 | 35 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:32` |
//! | `e023_createGetUnitOpInDifferentCore` | 0 | 18 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:73` |
//! | `e024_getMLIRTypeFromDSCDataFormat` | 0 | 53 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:97` |
//! | `e025_getAddressGranularityMultiplyFactor` | 0 | 17 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:156` |
//! | `e026_emitError` | 0 | 5 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:175` |
//! | `e027_setBuilderForDataTransfer` | 0 | 32 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:185` |
//! | `e028_getMLIRLoopFromLoopNode` | 0 | 14 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:220` |
//! | `e029_constructUniformizedAddress` | 0 | 47 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:236` |
//! | `e030_constructUniformizedFoldedAddress` | 0 | 51 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:285` |
//! | `e031_constructUniformizedFoldedDoubleBufferToggling` | 0 | 83 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:339` |
//! | `e032_constructUniformizedFoldedConstantBitStream` | 0 | 75 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:462` |
//! | `e059_constructUniformizedFoldedDestinationCore` | 1 | 36 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:425` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e022_retrieveGetUnitOpInSameCore
// crustify:todo: e023_createGetUnitOpInDifferentCore
// crustify:todo: e024_getMLIRTypeFromDSCDataFormat
// crustify:todo: e025_getAddressGranularityMultiplyFactor
// crustify:todo: e026_emitError
// crustify:todo: e027_setBuilderForDataTransfer
// crustify:todo: e028_getMLIRLoopFromLoopNode
// crustify:todo: e029_constructUniformizedAddress
// crustify:todo: e030_constructUniformizedFoldedAddress
// crustify:todo: e031_constructUniformizedFoldedDoubleBufferToggling
// crustify:todo: e032_constructUniformizedFoldedConstantBitStream
// crustify:todo: e059_constructUniformizedFoldedDestinationCore
