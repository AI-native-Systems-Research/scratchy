//! THE TRANSFER INSTRUCTIONS — the L3 and non-L3 loads and stores, the burst normalisation and
//! the load/store fusion.
//!
//! ⛔ TWO BURST DERIVATIONS EXIST IN OUR TREE AND dxp HAS ONE. Port the reference's.
//! ⛔ BOTH STORE-SIDE SYNCS WERE FOUND INVERTED ON AN EBR-MATCHED OP, and that writes zeros.
//!
//! 10 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e003_normalizeBurstSize` | 0 | 13 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2562` |
//! | `e054_ConstructL3LoadInstr` | 1 | 95 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2576` |
//! | `e055_ConstructL3StoreInstr` | 1 | 150 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2672` |
//! | `e056_ConstructZRAssignInstr` | 1 | 14 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2828` |
//! | `e057_ConstructL3LoadAndStoreInstr` | 1 | 179 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2843` |
//! | `e058_ConstructLoadComputeInstr` | 1 | 99 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3287` |
//! | `e059_ConstructLRFCopyInstr` | 1 | 49 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3509` |
//! | `e089_ConstructLoadInstr` | 2 | 201 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3024` |
//! | `e090_ConstructLoadInstr` | 2 | 60 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3226` |
//! | `e091_ConstructStoreInstr` | 2 | 121 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3387` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e003_normalizeBurstSize
// crustify:todo: e054_ConstructL3LoadInstr
// crustify:todo: e055_ConstructL3StoreInstr
// crustify:todo: e056_ConstructZRAssignInstr
// crustify:todo: e057_ConstructL3LoadAndStoreInstr
// crustify:todo: e058_ConstructLoadComputeInstr
// crustify:todo: e059_ConstructLRFCopyInstr
// crustify:todo: e089_ConstructLoadInstr
// crustify:todo: e090_ConstructLoadInstr
// crustify:todo: e091_ConstructStoreInstr
