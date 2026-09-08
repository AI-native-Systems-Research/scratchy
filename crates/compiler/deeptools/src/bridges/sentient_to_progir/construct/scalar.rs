//! THE SCALAR INSTRUCTIONS — the jumps, the compares, the adds and subs, and the sync.
//!
//! ⛔ THE LDST FAMILY IS ONE OPCODE AND THE COMPONENT DECIDES ITS ROLE (`isa.cpp:952`): the
//! field names `ldtype`/`consumertag` are picked BY COMPONENT, not by opcode. A misport here
//! writes the wrong field name into every load/store.
//! ⛔ THE LBR HOLDS AN INDEX, NOT AN ADDRESS — measured on an EBR-matched op-11 diff.
//!
//! 16 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e001_ConstructNOPInstr` | 0 | 10 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:329` |
//! | `e002_ConstructReturnInstr` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:340` |
//! | `e042_ConstructMVLoopInstr` | 1 | 63 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:48` |
//! | `e043_ConstructAssignInstr` | 1 | 204 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:113` |
//! | `e044_ConstructBranchExitInstr` | 1 | 9 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:319` |
//! | `e045_ConstructSyncInstr` | 1 | 96 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:348` |
//! | `e046_ConstructJMPInstr` | 1 | 11 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:446` |
//! | `e047_ConstructJCMPInstr` | 1 | 227 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:458` |
//! | `e048_ConstructJADDInstr` | 1 | 33 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:687` |
//! | `e049_ConstructXRFADDInstr` | 1 | 47 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:892` |
//! | `e050_ConstructXRFADDInstr` | 1 | 22 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:940` |
//! | `e051_ConstructJSUBInstr` | 1 | 34 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:964` |
//! | `e081_ConstructLRFADDInstr` | 2 | 88 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:721` |
//! | `e082_ConstructLARorEARADDInstr` | 2 | 81 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:810` |
//! | `e083_ConstructLRFSUBInstr` | 2 | 136 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:999` |
//! | `e084_ConstructLARorEARSUBInstr` | 2 | 46 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1136` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e001_ConstructNOPInstr
// crustify:todo: e002_ConstructReturnInstr
// crustify:todo: e042_ConstructMVLoopInstr
// crustify:todo: e043_ConstructAssignInstr
// crustify:todo: e044_ConstructBranchExitInstr
// crustify:todo: e045_ConstructSyncInstr
// crustify:todo: e046_ConstructJMPInstr
// crustify:todo: e047_ConstructJCMPInstr
// crustify:todo: e048_ConstructJADDInstr
// crustify:todo: e049_ConstructXRFADDInstr
// crustify:todo: e050_ConstructXRFADDInstr
// crustify:todo: e051_ConstructJSUBInstr
// crustify:todo: e081_ConstructLRFADDInstr
// crustify:todo: e082_ConstructLARorEARADDInstr
// crustify:todo: e083_ConstructLRFSUBInstr
// crustify:todo: e084_ConstructLARorEARSUBInstr
