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

//! `MutableStartAddrShifting.cpp` — 13 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e116_getMaxImmutableRange` | 116/384 | 8 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355` |
//! | `e191_calculateFullShift` | 191/384 | 25 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462` |
//! | `e192_calculateDimWeights` | 192/384 | 26 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560` |
//! | `e254_offsetShifts` | 254/384 | 22 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590` |
//! | `e255_applyShifts` | 255/384 | 27 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616` |
//! | `e291_calculatePartialShift` | 291/384 | 66 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490` |
//! | `e308_calculateShifts` | 308/384 | 61 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396` |
//! | `e323_shiftMutableAddr` | 323/384 | 19 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366` |
//! | `e352_transformVectorLoad` | 352/384 | 25 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201` |
//! | `e353_transformVectorStore` | 353/384 | 24 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229` |
//! | `e354_transformCompLoadAndStore` | 354/384 | 39 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256` |
//! | `e355_transformCompIndLoadAndStore` | 355/384 | 54 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298` |
//! | `e372_runOnOperation` | 372/384 | 69 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130` |

