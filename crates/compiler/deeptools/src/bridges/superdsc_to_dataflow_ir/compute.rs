//! THE VECTOR CHAINS — mac, binary and unary computes, and the precision they run at.
//! ⭐ ROPE IS TWO FMA STAGES, NOT A MULTIPLY: rope.ddl:26-27 declares rope64p1/rope64p2, two chained
//! FMA16 stages through an intermediate, and the m2 transfer's rotate_num_elements=32 IS the pair swap.
//!
//! 18 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e014_constructTypeFromFormat` | 0 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:278` |
//! | `e015_constructSingleValCustomVector` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:435` |
//! | `e016_mapToDicAttr` | 0 | 9 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1524` |
//! | `e047_getStaticContinuousMaskValue` | 1 | 23 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:33` |
//! | `e048_constructDynamicMasking` | 1 | 43 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:60` |
//! | `e049_getTypeBasedOnComputeType` | 1 | 12 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:108` |
//! | `e050_getReductionMapForMACOperation` | 1 | 70 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:126` |
//! | `e051_getSelectionMapForMACOperandFromL0` | 1 | 74 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:200` |
//! | `e052_constructPrecisionConversionOperation` | 1 | 59 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:375` |
//! | `e053_constructOpaqueOperation` | 1 | 25 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1534` |
//! | `e086_constructComputeInputOperandAndAddToList` | 3 | 320 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:456` |
//! | `e087_constructComputeOutputOperand` | 3 | 139 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:777` |
//! | `e094_constructComputeOutputOperands` | 4 | 14 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:921` |
//! | `e095_constructFMINorFMAXOperation` | 4 | 86 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1189` |
//! | `e099_constructMACOperation` | 5 | 87 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:942` |
//! | `e100_constructBinaryOrTernaryOperation` | 5 | 152 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1036` |
//! | `e101_constructUnaryOperation` | 5 | 247 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1276` |
//! | `e103_constructComputeOperation` | 6 | 83 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1567` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e014_constructTypeFromFormat
// crustify:todo: e015_constructSingleValCustomVector
// crustify:todo: e016_mapToDicAttr
// crustify:todo: e047_getStaticContinuousMaskValue
// crustify:todo: e048_constructDynamicMasking
// crustify:todo: e049_getTypeBasedOnComputeType
// crustify:todo: e050_getReductionMapForMACOperation
// crustify:todo: e051_getSelectionMapForMACOperandFromL0
// crustify:todo: e052_constructPrecisionConversionOperation
// crustify:todo: e053_constructOpaqueOperation
// crustify:todo: e086_constructComputeInputOperandAndAddToList
// crustify:todo: e087_constructComputeOutputOperand
// crustify:todo: e094_constructComputeOutputOperands
// crustify:todo: e095_constructFMINorFMAXOperation
// crustify:todo: e099_constructMACOperation
// crustify:todo: e100_constructBinaryOrTernaryOperation
// crustify:todo: e101_constructUnaryOperation
// crustify:todo: e103_constructComputeOperation
