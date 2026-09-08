//! THE COMPUTE OPERATION HANDLERS — mac, binary, unary, ternary, add, sub, splat, opaque, samv
//! and the mask pair, plus `LowerCommonOperations` that dispatches them.
//!
//! 12 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e107_LowerCommonOperations` | 3 | 26 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:538` |
//! | `e108_LowerBinaryOperation` | 3 | 8 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:565` |
//! | `e109_LowerUnaryOperation` | 3 | 7 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:574` |
//! | `e110_LowerTernaryOperation` | 3 | 8 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:582` |
//! | `e112_LowerMACOperation` | 3 | 31 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:658` |
//! | `e113_LowerSubOperation` | 3 | 40 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:690` |
//! | `e114_LowerAddOperation` | 3 | 50 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:731` |
//! | `e117_LowerOpaqueOperation` | 3 | 12 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:939` |
//! | `e118_LowerSAMVOperation` | 3 | 14 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:952` |
//! | `e119_LowerSetMaskOperation` | 3 | 10 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1118` |
//! | `e120_LowerIncrMaskOperation` | 3 | 10 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1129` |
//! | `e124_LowerSplatOperation` | 4 | 11 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:927` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e107_LowerCommonOperations
// crustify:todo: e108_LowerBinaryOperation
// crustify:todo: e109_LowerUnaryOperation
// crustify:todo: e110_LowerTernaryOperation
// crustify:todo: e112_LowerMACOperation
// crustify:todo: e113_LowerSubOperation
// crustify:todo: e114_LowerAddOperation
// crustify:todo: e117_LowerOpaqueOperation
// crustify:todo: e118_LowerSAMVOperation
// crustify:todo: e119_LowerSetMaskOperation
// crustify:todo: e120_LowerIncrMaskOperation
// crustify:todo: e124_LowerSplatOperation
