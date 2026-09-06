// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `Helper.cpp` — 60 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7, 8]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e030_getLoopNestLevel` | 030/384 | 6 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:43` |
//! | `e031_checkIndirectMemViewForExtractOp` | 031/384 | 42 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:388` |
//! | `e032_findExtractScalarOp` | 032/384 | 20 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:514` |
//! | `e033_getLoadConsumer` | 033/384 | 36 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242` |
//! | `e034_setldtype` | 034/384 | 60 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647` |
//! | `e035_generateSetSendDestinationStmts` | 035/384 | 49 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731` |
//! | `e036_getStoreOpFromLoadStorePattern` | 036/384 | 8 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872` |
//! | `e037_findCandidateForLowering` | 037/384 | 12 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884` |
//! | `e038_addLoadChainToDeleteList` | 038/384 | 9 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975` |
//! | `e151_checkCompositeRegion` | 151/384 | 89 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:206` |
//! | `e152_checkStoreOpFromExtractPattern` | 152/384 | 27 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:433` |
//! | `e153_isLoadAndExtractScalarPattern` | 153/384 | 27 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:463` |
//! | `e154_isReceiveAndExtractScalarPattern` | 154/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:493` |
//! | `e155_updateSymbolicAccessDetails` | 155/384 | 35 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013` |
//! | `e156_getStoreProducer` | 156/384 | 159 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285` |
//! | `e157_constructSetActiveMaskValueOp` | 157/384 | 161 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567` |
//! | `e158_createUniformizeRegionsOp` | 158/384 | 57 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880` |
//! | `e210_checkBasicConditions` | 210/384 | 133 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:58` |
//! | `e211_processInterleaveOp` | 211/384 | 80 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:305` |
//! | `e212_gatherAffineLoadStoreDetails` | 212/384 | 74 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:538` |
//! | `e213_constructImmutableAddress` | 213/384 | 16 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217` |
//! | `e214_setImmutableAddrAndIncrements` | 214/384 | 43 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581` |
//! | `e215_setsttype` | 215/384 | 50 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731` |
//! | `e216_constructReceiveAndExtractScalarOp` | 216/384 | 91 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471` |
//! | `e217_lowerVectorLoadHelper` | 217/384 | 46 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899` |
//! | `e218_lowerSetTransferMaskStateOp` | 218/384 | 9 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815` |
//! | `e219_cloneStartAddrOutsideLoop` | 219/384 | 79 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942` |
//! | `e220_cleanupTriviallyRedundantSetSendDestination` | 220/384 | 38 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084` |
//! | `e267_constructTimeLoopsAndVectorOperations` | 267/384 | 109 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1789` |
//! | `e268_constructLoadAndStoreStmt` | 268/384 | 177 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2167` |
//! | `e269_constructLoadAndExtractScalarOp` | 269/384 | 114 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2351` |
//! | `e270_addStoreInputToDeleteList` | 270/384 | 8 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2987` |
//! | `e271_lowerCompositeMemoryInterleaveOp` | 271/384 | 37 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3775` |
//! | `e272_insertInitializationStmt` | 272/384 | 13 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3834` |
//! | `e298_constructAffineDetailsAndAddrs` | 298/384 | 16 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2787` |
//! | `e299_lowerAffineCompositeHelper` | 299/384 | 15 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2953` |
//! | `e311_constructAffineCompDetailsAndAddrs` | 311/384 | 33 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2809` |
//! | `e312_lowerExtractVectorLoadOp` | 312/384 | 21 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3001` |
//! | `e313_lowerExtractVectorStoreOp` | 313/384 | 20 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3026` |
//! | `e314_lowerVectorLoadOp` | 314/384 | 20 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3050` |
//! | `e315_lowerVectorStoreOp` | 315/384 | 28 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3074` |
//! | `e316_lowerIndirectVectorLoadOp` | 316/384 | 44 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3169` |
//! | `e317_lowerIndirectVectorStoreOp` | 317/384 | 46 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3217` |
//! | `e318_lowerLDCVTIPattern` | 318/384 | 326 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3444` |
//! | `e327_gatherSymbolicLoadStoreDetails` | 327/384 | 153 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1051` |
//! | `e328_adjustMutableAddrInitForIndirect` | 328/384 | 127 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1447` |
//! | `e329_lowerCompositeLoadOp` | 329/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3106` |
//! | `e330_lowerCompositeStoreOp` | 330/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3127` |
//! | `e331_lowerCompositeLoadAndStoreOp` | 331/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3148` |
//! | `e332_lowerCompositeIndirectLoadOp` | 332/384 | 42 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3267` |
//! | `e333_lowerCompositeIndirectStoreOp` | 333/384 | 42 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3313` |
//! | `e334_lowerCompositeIndirectLoadAndStoreOp` | 334/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3359` |
//! | `e335_insertCopyAndAddStmts` | 335/384 | 13 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3861` |
//! | `e336_adjustMutableAddrInitForStride` | 336/384 | 47 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:4034` |
//! | `e357_generateAffineAddressManipulationStmts` | 357/384 | 381 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:625` |
//! | `e358_constructLoadAndSendStmt` | 358/384 | 104 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1910` |
//! | `e359_constructReceiveAndStoreStmt` | 359/384 | 131 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2025` |
//! | `e360_constructSymbolicDetailsAndAddrs` | 360/384 | 16 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2849` |
//! | `e374_lowerSymbolicVectorLoadOp` | 374/384 | 26 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380` |
//! | `e375_lowerSymbolicVectorStoreOp` | 375/384 | 30 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410` |

