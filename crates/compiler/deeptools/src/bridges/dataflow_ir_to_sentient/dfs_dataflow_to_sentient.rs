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

use crate::islands::dataflow_ir::ty::GenericComp;
use crate::units::DfirUnit;

/// Replaces: e039_isSenComponentL0LU
///
/// **039/384** `isSenComponentL0LU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96` (2L).
///
/// ```cpp
/// static inline bool isSenComponentL0LU(SenComponents comp) {
///   return EnumsConversion::senCompToGenericComp.at(comp) == SenComponents::L0LU;
/// }
/// ```
///
/// ⭐⭐ IT IS THE **GENERIC** COMPONENT THAT IS TESTED, NOT THE SPELLING. `senCompToGenericComp`
/// (`sys-arch-spec/arch_enums.cpp:124-211`) maps every per-core, per-corelet spelling onto the one
/// image the ISA names, so this answers `true` for `L0LU` and for nothing else — and in particular
/// **not** for `L0SU`. A port that had folded the two halves together would answer `true` for both
/// and this predicate would select every L0 unit in the program.
///
/// ⛔ THE `.at()` CAN THROW AND OURS CANNOT. `L0`, `CONSTANT` and `SFPRING` are not keys of that map,
/// so the reference aborts on them; [`DfirUnit::generic`] is total, and each of the three has its own
/// image — none of which is `L0LU`, so those units answer `false` here.
#[must_use]
pub const fn is_sen_component_l0lu(unit: DfirUnit) -> bool {
    matches!(unit.generic(), GenericComp::L0lu)
}

/// Replaces: e040_isSenComponentL0SU
///
/// **040/384** `isSenComponentL0SU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100` (2L).
///
/// ```cpp
/// static inline bool isSenComponentL0SU(SenComponents comp) {
///   return EnumsConversion::senCompToGenericComp.at(comp) == SenComponents::L0SU;
/// }
/// ```
///
/// ⭐ THE STORE HALF, AND ONLY IT — see [`is_sen_component_l0lu`] for why the two are separate
/// images rather than one `L0`.
#[must_use]
pub const fn is_sen_component_l0su(unit: DfirUnit) -> bool {
    matches!(unit.generic(), GenericComp::L0su)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::units::Row;

    /// 🎯 039/384 + 040/384 — THE LOAD HALF AND THE STORE HALF ARE TOLD APART.
    ///
    /// ⛔ THE POINT OF THE PAIR. The two predicates exist to route a sync onto one half of the L0, so
    /// a mapping that collapsed `l0lu` and `l0su` onto one generic component would make both answer
    /// `true` for both units and every L0 sync would be emitted twice.
    #[test]
    fn the_l0_halves_are_distinct_generic_components() {
        assert!(is_sen_component_l0lu(DfirUnit::L0lu));
        assert!(!is_sen_component_l0su(DfirUnit::L0lu));

        assert!(is_sen_component_l0su(DfirUnit::L0su));
        assert!(!is_sen_component_l0lu(DfirUnit::L0su));
    }

    /// 🎯 039/384 + 040/384 — AND NO OTHER UNIT IS AN L0 HALF.
    ///
    /// ⛔ INCLUDING THE THREE THE REFERENCE'S MAP HAS NO KEY FOR. `senCompToGenericComp.at(L0)`,
    /// `.at(CONSTANT)` and `.at(SFPRING)` throw (`arch_enums.cpp:124-211` has no entry for them);
    /// ours answer `false`, which is the routing decision those units need.
    #[test]
    fn nothing_else_is_an_l0_half() {
        for unit in [
            DfirUnit::PtRow(Row::checked(0).expect("row 0 exists on every arch")),
            DfirUnit::Pe,
            DfirUnit::Sfp,
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::Lx,
            DfirUnit::L3lu,
            DfirUnit::L3su,
            DfirUnit::Hbm,
            DfirUnit::CrossPtnLink,
            DfirUnit::SfpState,
            DfirUnit::PeState,
            DfirUnit::L0,
            DfirUnit::Constant,
            DfirUnit::SfpRing,
        ] {
            assert!(
                !is_sen_component_l0lu(unit),
                "{unit:?} is not the L0 load unit"
            );
            assert!(
                !is_sen_component_l0su(unit),
                "{unit:?} is not the L0 store unit"
            );
        }
    }
}
