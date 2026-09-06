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

//! `DataflowToSentient.cpp` — 21 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e039_isSenComponentL0LU` | 039/384 | 2 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96` |
//! | `e040_isSenComponentL0SU` | 040/384 | 2 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100` |
//! | `e041_ExtendUnitNameToCorelet` | 041/384 | 11 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104` |
//! | `e042_isSameListOfUnits` | 042/384 | 10 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175` |
//! | `e043_isTargetL3` | 043/384 | 6 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720` |
//! | `e044_lowerOpaqueOperation` | 044/384 | 27 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984` |
//! | `e159_getUnitNameFromAListOfGetUnitOp` | 159/384 | 10 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119` |
//! | `e160_areCoreletsDifferent` | 160/384 | 6 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132` |
//! | `e161_separateBasedOnDestinationUnits` | 161/384 | 18 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761` |
//! | `e221_pushBackTheUnitToListIfDoesnotExist` | 221/384 | 5 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143` |
//! | `e222_createUniformRegionsWithTwoRegionsNoResult` | 222/384 | 18 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153` |
//! | `e223_lowerL3SyncOperationForAUnit` | 223/384 | 57 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375` |
//! | `e224_lowerL3SyncOperationForAGroupOfUnits` | 224/384 | 61 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667` |
//! | `e273_lowerL0LXSyncOperationForAUnit` | 273/384 | 180 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189` |
//! | `e274_lowerL0LXSyncOperationForAGroupOfUnits` | 274/384 | 222 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438` |
//! | `e300_lowerSyncForAUnit` | 300/384 | 7 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733` |
//! | `e301_lowerSyncForAGroup` | 301/384 | 8 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746` |
//! | `e302_lowerSyncLXL3ToLXL3` | 302/384 | 928 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787` |
//! | `e319_lowerSyncForAQueryMap` | 319/384 | 166 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728` |
//! | `e337_lowerSyncOperation` | 337/384 | 80 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901` |
//! | `e361_runOnOperation` | 361/384 | 33 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014` |

