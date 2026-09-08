//! THE OPAQUE-TEMPLATE INSTRUCTION — `ConstructOpaqueInstr` and the string trim it uses.
//!
//! ⭐ THE TEMPLATES ARE DATA, IN `dcc/src/Conversion/SentientToProgIR/opaqueTemplates/` — 38
//! `.smc` files (exp, gelu, sigmoid, rsqrt, layernormscale, idx32toaddr, …). They are a
//! declared-data table, not code; this crate's `build.rs` ALREADY reads `.smc` mnemonics and
//! emits `InstOpCode::…`, so a template naming a non-opcode fails to compile.
//!
//! 2 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e004_rtrim` | 0 | 6 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3859` |
//! | `e063_ConstructOpaqueInstr` | 1 | 167 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3866` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e004_rtrim
// crustify:todo: e063_ConstructOpaqueInstr
