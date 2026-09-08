//! WHICH REGISTERS ARE DEFINED, AND WHETHER ANYTHING READS ONE THAT IS NOT.
//!
//! ⛔⛔ `checkRegDefs` IS A CHECK, AND THIS CRATE NEVER RUNTIME REFUSES. Port it as a function
//! that RETURNS THE OFFENDERS, the way `progir::Program::overflowing` does — never as an
//! `assert!`, a `Result` or a `signalPassFailure`.
//! ⭐ `~UniformRegionContext` IS RAII: real work runs on scope exit (it merges the region's reg
//! defs). It is unit `e070_dtor_UniformRegionContext`, not a destructor to drop.
//!
//! 5 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e012_setRegDef` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:53` |
//! | `e013_addRegDefsForUnit` | 0 | 15 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:61` |
//! | `e014_checkRegDefs` | 0 | 63 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:106` |
//! | `e069_recordOpRegDefs` | 1 | 30 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:22` |
//! | `e070_dtor_UniformRegionContext` | 1 | 28 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:77` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e012_setRegDef
// crustify:todo: e013_addRegDefsForUnit
// crustify:todo: e014_checkRegDefs
// crustify:todo: e069_recordOpRegDefs
// crustify:todo: e070_dtor_UniformRegionContext
