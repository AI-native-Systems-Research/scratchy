//! THE BLOCKS AND THE REGIONS — `UniformInstrBlock` and `UniformInstrBlocks`: per-unit
//! instruction lists, the uniformised lists, region indices and the flattened index.
//!
//! ⛔ A REGION INDEX OF -1 IS "NOT PRESENT", and a REGULAR block always answers 0
//! (`UniformInstrAndBlock.hpp:208`). A sentinel is an `Option` here, never a negative number.
//!
//! 24 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e023_getUnitInstrList` | 0 | 12 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:266` |
//! | `e024_insertInstruction` | 0 | 5 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:322` |
//! | `e025_setCurrentRegion` | 0 | 4 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:327` |
//! | `e026_getMaxInstrSize` | 0 | 8 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:331` |
//! | `e027_getCurrentRegionInstrSize` | 0 | 5 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:360` |
//! | `e028_getInstr` | 0 | 8 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:365` |
//! | `e029_getLastInstr` | 0 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:374` |
//! | `e030_appendEmptyUniformRegion` | 0 | 4 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:378` |
//! | `e031_getMaxInstrSize` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:426` |
//! | `e032_appendRegularBlock` | 0 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:442` |
//! | `e033_appendUniformBlock` | 0 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:445` |
//! | `e034_getUniformInstr` | 0 | 6 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:449` |
//! | `e035_doesInstrWithThisIndexExist` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:456` |
//! | `e036_appendEmptyUniformRegion` | 0 | 4 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:497` |
//! | `e037_getUnitRegionIndex` | 0 | 6 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.hpp:208` |
//! | `e074_getMaxInstrRegionIndex` | 1 | 12 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:340` |
//! | `e075_getRegionInstrSize` | 1 | 7 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:353` |
//! | `e076_getUnitUniformInstrList` | 1 | 9 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:383` |
//! | `e077_getLastBlockCurrentInstrSize` | 1 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:433` |
//! | `e078_addInstructionToLastBlock` | 1 | 6 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:436` |
//! | `e079_getNextInstrIndex` | 1 | 19 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:464` |
//! | `e096_getUniformizedUnitInstrList` | 2 | 42 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:279` |
//! | `e097_flattenIndex` | 2 | 12 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:484` |
//! | `e121_getUniformizedUnitUniformInstrList` | 3 | 32 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:393` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e023_getUnitInstrList
// crustify:todo: e024_insertInstruction
// crustify:todo: e025_setCurrentRegion
// crustify:todo: e026_getMaxInstrSize
// crustify:todo: e027_getCurrentRegionInstrSize
// crustify:todo: e028_getInstr
// crustify:todo: e029_getLastInstr
// crustify:todo: e030_appendEmptyUniformRegion
// crustify:todo: e031_getMaxInstrSize
// crustify:todo: e032_appendRegularBlock
// crustify:todo: e033_appendUniformBlock
// crustify:todo: e034_getUniformInstr
// crustify:todo: e035_doesInstrWithThisIndexExist
// crustify:todo: e036_appendEmptyUniformRegion
// crustify:todo: e037_getUnitRegionIndex
// crustify:todo: e074_getMaxInstrRegionIndex
// crustify:todo: e075_getRegionInstrSize
// crustify:todo: e076_getUnitUniformInstrList
// crustify:todo: e077_getLastBlockCurrentInstrSize
// crustify:todo: e078_addInstructionToLastBlock
// crustify:todo: e079_getNextInstrIndex
// crustify:todo: e096_getUniformizedUnitInstrList
// crustify:todo: e097_flattenIndex
// crustify:todo: e121_getUniformizedUnitUniformInstrList
