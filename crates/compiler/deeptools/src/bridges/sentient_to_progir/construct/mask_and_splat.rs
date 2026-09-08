//! THE MASK, SPLAT AND SAMV INSTRUCTIONS — set-dest-mask, set-dest, splat, splat-pad, and the
//! set/incr mask pair.
//!
//! 9 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e005_ConstructIncrMaskInstr` | 0 | 11 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4118` |
//! | `e060_ConstructSetDstMaskInstr` | 1 | 41 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3559` |
//! | `e061_ConstructSetDestInstr` | 1 | 54 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3601` |
//! | `e062_ConstructImmCopyInstrFromSplatOp` | 1 | 60 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3657` |
//! | `e064_ConstructSAMVInstr` | 1 | 33 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4034` |
//! | `e065_ConstructSAMVResetInstruction` | 1 | 26 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4068` |
//! | `e066_ConstructSetMaskInstr` | 1 | 22 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4095` |
//! | `e092_ConstructSplatInstrFromSplatOp` | 2 | 65 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3719` |
//! | `e098_ConstructSplatPadInstr` | 3 | 65 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3792` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e005_ConstructIncrMaskInstr
// crustify:todo: e060_ConstructSetDstMaskInstr
// crustify:todo: e061_ConstructSetDestInstr
// crustify:todo: e062_ConstructImmCopyInstrFromSplatOp
// crustify:todo: e064_ConstructSAMVInstr
// crustify:todo: e065_ConstructSAMVResetInstruction
// crustify:todo: e066_ConstructSetMaskInstr
// crustify:todo: e092_ConstructSplatInstrFromSplatOp
// crustify:todo: e098_ConstructSplatPadInstr
