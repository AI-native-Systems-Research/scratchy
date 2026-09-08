//! THE CONTROL FLOW AND THE UNIFORM REGIONS — for, yield, return, sync, nop, set-send-dest, and
//! the uniform-operation pair.
//!
//! ⛔ `LowerUniformOperations` AND `GenerateProgIR` ARE MUTUALLY RECURSIVE (level 5, one
//! component). Neither can be ported as if the other were a leaf.
//! ⛔ THE SYNC ORDER IS dxp's AND CITED: reordering syncs times the card out (CB state=TimedOut,
//! sync rc=-1).
//!
//! 9 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e011_fillUnitToIdMap` | 0 | 17 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:967` |
//! | `e105_LowerSyncOperation` | 3 | 37 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:353` |
//! | `e106_LowerNOPOperation` | 3 | 14 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:391` |
//! | `e111_LowerForOperation` | 3 | 66 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:591` |
//! | `e115_LowerReturnOperation` | 3 | 7 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:900` |
//! | `e116_LowerSetSendDestinationOperation` | 3 | 18 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:908` |
//! | `e123_LowerYieldOperation` | 4 | 117 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:782` |
//! | `e125_LowerUniformYieldOperation` | 4 | 61 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1056` |
//! | `e126_LowerUniformOperations` | 5 | 70 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:985` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e011_fillUnitToIdMap
// crustify:todo: e105_LowerSyncOperation
// crustify:todo: e106_LowerNOPOperation
// crustify:todo: e111_LowerForOperation
// crustify:todo: e115_LowerReturnOperation
// crustify:todo: e116_LowerSetSendDestinationOperation
// crustify:todo: e123_LowerYieldOperation
// crustify:todo: e125_LowerUniformYieldOperation
// crustify:todo: e126_LowerUniformOperations
