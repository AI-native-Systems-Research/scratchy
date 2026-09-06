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

//! `MutableAddrSplitting.cpp` — 23 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e111_getMaxMutableRange` | 111/384 | 8 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673` |
//! | `e112_getMaxImmutableRange` | 112/384 | 8 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683` |
//! | `e113_isEligibleForSplitting` | 113/384 | 20 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832` |
//! | `e114_sortDataBasedOnWeight` | 114/384 | 4 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855` |
//! | `e115_createNewMemViewWithMod` | 115/384 | 12 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966` |
//! | `e184_getLoopTripCount` | 184/384 | 56 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743` |
//! | `e185_hasMutableAddrOverflow` | 185/384 | 14 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801` |
//! | `e186_calculatePartitionSizes` | 186/384 | 97 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862` |
//! | `e187_constructConditionals` | 187/384 | 59 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000` |
//! | `e188_calculateSubscriptsCoefficients` | 188/384 | 11 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188` |
//! | `e189_synthesizeTimeInfo` | 189/384 | 22 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256` |
//! | `e190_createExplicitTimeLoops` | 190/384 | 57 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282` |
//! | `e250_initMASData` | 250/384 | 31 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707` |
//! | `e251_setupForPartitioning` | 251/384 | 6 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819` |
//! | `e252_fillPartitions` | 252/384 | 117 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063` |
//! | `e253_adjustForEvenImmutableAddr` | 253/384 | 47 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204` |
//! | `e289_initialize` | 289/384 | 8 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694` |
//! | `e290_createPartitions` | 290/384 | 10 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983` |
//! | `e306_transformVectorLoad` | 306/384 | 75 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298` |
//! | `e307_transformVectorStore` | 307/384 | 74 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375` |
//! | `e321_transformCompLoadAndStore` | 321/384 | 92 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451` |
//! | `e322_transformCompIndLoadAndStore` | 322/384 | 124 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546` |
//! | `e351_runOnOperation` | 351/384 | 70 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226` |

