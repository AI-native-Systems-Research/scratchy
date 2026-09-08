//! THE LOOP NEST AND THE CONDITIONALS — DSC loops, blocks and conds becoming the scf/affine nest.
//!
//! 14 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e017_getCmpIPredicate_dup` | 0 | 25 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:21` |
//! | `e018_getMLIRLoopFromSNLoopNode` | 0 | 16 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:49` |
//! | `e019_getParentLoop` | 0 | 26 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:381` |
//! | `e020_propagateBufferSwitchLoopsToRootRecursively` | 0 | 50 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:519` |
//! | `e021_resetIterArguments` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:738` |
//! | `e054_constructConditionalOperation` | 1 | 211 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:66` |
//! | `e055_constructConditionalsForSAMV` | 1 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:284` |
//! | `e056_getBufferingOrStreamingMode` | 1 | 54 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:408` |
//! | `e057_propagateBufferSwitchLoopsToRoot` | 1 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:575` |
//! | `e058_constructLoopForADim` | 1 | 101 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:754` |
//! | `e074_getBlockingOrStreamingBufferLoopLocations` | 2 | 46 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:467` |
//! | `e075_constructLoopIterArgs` | 2 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:595` |
//! | `e088_constructLoopsRecursive` | 3 | 324 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:862` |
//! | `e096_constructLoops` | 4 | 27 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:1191` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e017_getCmpIPredicate_dup
// crustify:todo: e018_getMLIRLoopFromSNLoopNode
// crustify:todo: e019_getParentLoop
// crustify:todo: e020_propagateBufferSwitchLoopsToRootRecursively
// crustify:todo: e021_resetIterArguments
// crustify:todo: e054_constructConditionalOperation
// crustify:todo: e055_constructConditionalsForSAMV
// crustify:todo: e056_getBufferingOrStreamingMode
// crustify:todo: e057_propagateBufferSwitchLoopsToRoot
// crustify:todo: e058_constructLoopForADim
// crustify:todo: e074_getBlockingOrStreamingBufferLoopLocations
// crustify:todo: e075_constructLoopIterArgs
// crustify:todo: e088_constructLoopsRecursive
// crustify:todo: e096_constructLoops
