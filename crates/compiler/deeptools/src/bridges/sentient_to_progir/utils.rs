//! THE SHARED HELPERS — the on-the-fly conversion check, the proper-consumer walk, the address
//! wraparound and the fold-mode value.
//!
//! ⛔ `verifyOnTheFlyConversions` IS A CHECK — see `reg_def_tracker.rs`'s note. Return the
//! offenders, never refuse.
//!
//! 5 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e038_verifyOnTheFlyConversions` | 0 | 56 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:27` |
//! | `e039_getProperConsumer` | 0 | 14 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:86` |
//! | `e040_updateProperConsumer` | 0 | 14 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:102` |
//! | `e041_getAddrWraparounded` | 0 | 10 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:117` |
//! | `e080_setFCValueFromFoldMode` | 1 | 22 | `dcc/src/Conversion/SentientToProgIR/Utils.cpp:128` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e038_verifyOnTheFlyConversions
// crustify:todo: e039_getProperConsumer
// crustify:todo: e040_updateProperConsumer
// crustify:todo: e041_getAddrWraparounded
// crustify:todo: e080_setFCValueFromFoldMode
