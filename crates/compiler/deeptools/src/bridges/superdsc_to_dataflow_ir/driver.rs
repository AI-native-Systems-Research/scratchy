//! THE CONVERSION DRIVER — runTranslator, convertV3, convertV4, and the module scaffolding they build.
//!
//! 11 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e003_startDataflowIRGeneration` | 0 | 18 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:20` |
//! | `e004_stopDataflowIRGeneration` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:39` |
//! | `e005_areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet` | 0 | 21 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:226` |
//! | `e042_terminate` | 1 | 3 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:221` |
//! | `e043_areFoldsNeeded` | 1 | 40 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:249` |
//! | `e105_constructOperationsRecursively` | 7 | 165 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:55` |
//! | `e106_ConstructAProgramUnit` | 8 | 81 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:293` |
//! | `e107_ConstructAUniformizedProgramUnit` | 8 | 84 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:378` |
//! | `e108_convertV3` | 9 | 29 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:466` |
//! | `e109_convertV4` | 9 | 33 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:499` |
//! | `e110_runTranslator` | 10 | 32 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:534` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e003_startDataflowIRGeneration
// crustify:todo: e004_stopDataflowIRGeneration
// crustify:todo: e005_areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet
// crustify:todo: e042_terminate
// crustify:todo: e043_areFoldsNeeded
// crustify:todo: e105_constructOperationsRecursively
// crustify:todo: e106_ConstructAProgramUnit
// crustify:todo: e107_ConstructAUniformizedProgramUnit
// crustify:todo: e108_convertV3
// crustify:todo: e109_convertV4
// crustify:todo: e110_runTranslator
