//! THE D76 PASS ITSELF — `runOnOperation`, `GenerateProgIR`, `GenerateProgIRForProgramUnit`,
//! the register initialisation, the program-length equalisation and the `init.smc` collapse.
//!
//! ⭐ ONE `ProgramUnitOp` SPANS THE SET: handles = cores x corelets x num_folds.
//! ⛔ THE PROGRAM BOUNDS ARE READ, NOT RESTATED — `max_ibuff_entries(unit)` per component and
//! per arch, never `kMaxCompIBuff` as if it were any unit's limit.
//!
//! 7 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e015_initializeUtilizedRegisters` | 0 | 39 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:429` |
//! | `e016_replaceProgramBodyWithSmcOp` | 0 | 49 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:560` |
//! | `e095_equalizeProgramLength` | 2 | 90 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:469` |
//! | `e127_GenerateProgIR` | 5 | 65 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:53` |
//! | `e128_GenerateProgIR` | 5 | 111 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:119` |
//! | `e129_GenerateProgIRForProgramUnit` | 6 | 186 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:234` |
//! | `e130_runOnOperation` | 7 | 135 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:610` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e015_initializeUtilizedRegisters
// crustify:todo: e016_replaceProgramBodyWithSmcOp
// crustify:todo: e095_equalizeProgramLength
// crustify:todo: e127_GenerateProgIR
// crustify:todo: e128_GenerateProgIR
// crustify:todo: e129_GenerateProgIRForProgramUnit
// crustify:todo: e130_runOnOperation
