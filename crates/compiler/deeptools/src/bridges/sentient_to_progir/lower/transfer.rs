//! THE TRANSFER OPERATION HANDLERS — load_and_send, receive_and_store, load_and_store, the
//! scalar extracts, load_compute_and_send and copy.
//!
//! ⭐ THESE ARE FIVE OF THE SEVEN SENTIENT OPS THE REFERENCE ACTUALLY EMITS across all 417
//! staged programs, so this file is on the hot path for every program.
//!
//! 7 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e099_LowerLoadAndSendOperation` | 3 | 29 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:144` |
//! | `e100_LowerReceiveAndStoreOperation` | 3 | 49 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:174` |
//! | `e101_LowerLoadAndStoreOperation` | 3 | 55 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:224` |
//! | `e102_LowerLoadAndExtractScalarOperation` | 3 | 24 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:280` |
//! | `e103_LowerReceiveAndExtractScalarOperation` | 3 | 13 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:305` |
//! | `e104_LowerLoadComputeAndSendOperation` | 3 | 33 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:319` |
//! | `e122_LowerCopyOperation` | 4 | 95 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:406` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e099_LowerLoadAndSendOperation
// crustify:todo: e100_LowerReceiveAndStoreOperation
// crustify:todo: e101_LowerLoadAndStoreOperation
// crustify:todo: e102_LowerLoadAndExtractScalarOperation
// crustify:todo: e103_LowerReceiveAndExtractScalarOperation
// crustify:todo: e104_LowerLoadComputeAndSendOperation
// crustify:todo: e122_LowerCopyOperation
