//! WHAT MUST BE IN A REGISTER BEFORE ANYTHING READS IT — the reg-init accumulation and the
//! PE/SFP LRF immediate copies.
//!
//! ⛔ REGISTER-FILE ALLOCATIONS NEED `primaryDimToVal_st`'s rowSplit_/peSfpSplit_ SHARE. That
//! input was missing once and produced 199 refusals.
//!
//! 3 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e006_addToRegsToInit` | 0 | 61 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4157` |
//! | `e007_addPESFPLRFImmcopyToRegInit` | 0 | 228 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4226` |
//! | `e067_fillImmField` | 1 | 26 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4130` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e006_addToRegsToInit
// crustify:todo: e007_addPESFPLRFImmcopyToRegInit
// crustify:todo: e067_fillImmField
