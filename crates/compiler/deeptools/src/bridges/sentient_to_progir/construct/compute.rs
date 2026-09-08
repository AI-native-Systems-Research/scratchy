//! THE COMPUTE INSTRUCTIONS — FMA, binary, unary and ternary, and the operand plumbing that
//! feeds them.
//!
//! ⛔ THE FMA DESTINATION IS `ResultForwarding` ON THE OP, not a register operand.
//! ⛔ `ConstructBinaryInstr` IS 485 LINES — the single largest unit in the span. Its branch
//! order and early returns ARE the port; a summary of its decision rule is not.
//!
//! 6 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e052_setSentientComputeInputProgIROperands` | 1 | 125 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1184` |
//! | `e053_setSentientComputeOutputProgIROperands` | 1 | 112 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1310` |
//! | `e085_ConstructFMAInstr` | 2 | 291 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1423` |
//! | `e086_ConstructBinaryInstr` | 2 | 485 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1715` |
//! | `e087_ConstructUnaryInstr` | 2 | 237 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2201` |
//! | `e088_ConstructTernaryInstr` | 2 | 122 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2439` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e052_setSentientComputeInputProgIROperands
// crustify:todo: e053_setSentientComputeOutputProgIROperands
// crustify:todo: e085_ConstructFMAInstr
// crustify:todo: e086_ConstructBinaryInstr
// crustify:todo: e087_ConstructUnaryInstr
// crustify:todo: e088_ConstructTernaryInstr
