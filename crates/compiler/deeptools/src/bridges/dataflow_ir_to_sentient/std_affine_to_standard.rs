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

//! `AffineToStandard.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [0]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e001_matchAndRewrite` | 001/384 | 8 | `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41` |

use crate::islands::dataflow_ir::dialects::{self as dfir_op, Val};

/// WHAT ENCLOSES A TERMINATOR — the one input `AffineYieldOpLowering` reads besides the op itself.
///
/// ⛔ THE KIND, NOT THE OP. The reference asks exactly one question of the parent —
/// `isa<scf::ParallelOp>(op->getParentOp())` — so carrying the whole parent would offer callers a
/// dozen other questions the rule does not ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parent {
    /// An `scf.parallel`. ⛔ THE ONE CASE THAT DECLINES.
    ScfParallel,
    /// Anything else — an `affine.for`, an `scf.if`, a `dataflow.program_unit`.
    Other,
}

/// THE RESULT OF THE REWRITE — ⛔ DECLINING IS NOT FAILING.
///
/// ⛔⛔ THE REFERENCE RETURNS `LogicalResult::failure()` AND THAT IS NOT AN ERROR. In MLIR a pattern
/// returning failure means *this pattern does not apply here*; the driver tries others, the op is
/// left alone and the module is untouched. Modelling it as an error would stop a lowering the
/// reference completes.
///
/// ⭐ AND NOTHING ELSE IN dcc's PASS PICKS THE OP UP. The reference's reason —
/// *"Terminator is rewritten as part of the "affine.parallel" lowering pattern."*
/// (`AffineToStandard.cpp:44-45`) — is inherited from upstream MLIR, whose `AffineParallelLowering`
/// builds the `scf.parallel` and its terminator together. dcc's copy of the pass registers four
/// patterns and no parallel lowering at all (`:198-206`), and its conversion target already declares
/// the whole `scf` dialect legal (`:227-228`). So under an `scf.parallel` the terminator is an `scf`
/// terminator already and declining is what leaves the IR correct.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum YieldRewrite {
    /// The `scf.yield` that replaces the `affine.yield`.
    Yielded(dfir_op::scf::Op),
    /// This pattern does not apply — the parallel lowering owns this terminator.
    Declined,
}

/// Replaces: e001_matchAndRewrite
///
/// `AffineYieldOpLowering::matchAndRewrite` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41`
/// (8L), entry 001/384.
///
/// ```cpp
/// LogicalResult matchAndRewrite(AffineYieldOp op,
///                               PatternRewriter &rewriter) const override {
///   if (isa<scf::ParallelOp>(op->getParentOp())) {
///     // Terminator is rewritten as part of the "affine.parallel" lowering
///     // pattern.
///     return failure();
///   }
///   rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
///   return success();
/// }
/// ```
///
/// ⛔ THE OPERANDS CARRY THROUGH UNCHANGED. `replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands())`
/// passes the yield's own operand list to the new op, so a loop carrying two values yields two. An
/// empty list is a terminator of a loop that carries nothing, not a missing list.
///
/// ⚠️ THE EXTRACT DROPPED THE `return success();` (`crustify-bridge2/source/bridge2.cpp:24-33` ends on
/// the `replaceOpWithNewOp` call and two bare braces). The authority at the cited line has it, and it
/// is the difference between a pattern that applied and one that declined — which is the whole
/// distinction this function makes. Ported from the authority.
pub fn lower_affine_yield(parent: Parent, operands: &[Val]) -> YieldRewrite {
    match parent {
        Parent::ScfParallel => YieldRewrite::Declined,
        Parent::Other => YieldRewrite::Yielded(dfir_op::scf::Op::Yield {
            operands: operands.to_vec(),
        }),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::Op as DfirOp;

    /// 🎯 001/384 — AN `affine.yield` BECOMES AN `scf.yield` CARRYING THE SAME OPERANDS.
    #[test]
    fn an_affine_yield_becomes_an_scf_yield() {
        let carried = [Val(7), Val(9)];
        assert_eq!(
            lower_affine_yield(Parent::Other, &carried),
            YieldRewrite::Yielded(dfir_op::scf::Op::Yield {
                operands: vec![Val(7), Val(9)],
            }),
            "the operand list passes through unchanged"
        );
    }

    /// 🎯 001/384 — AND A LOOP CARRYING NOTHING STILL YIELDS.
    ///
    /// ⛔ AN EMPTY OPERAND LIST IS A TERMINATOR, not an absent one. The reference has no arm for it:
    /// `getOperands()` on a bare `affine.yield` is empty and the rewrite runs anyway.
    #[test]
    fn a_yield_with_no_carried_values_still_rewrites() {
        assert_eq!(
            lower_affine_yield(Parent::Other, &[]),
            YieldRewrite::Yielded(dfir_op::scf::Op::Yield {
                operands: Vec::new()
            })
        );
    }

    /// 🎯 001/384 — UNDER AN `scf.parallel` THE PATTERN DECLINES.
    ///
    /// ⛔ AND DECLINING MUST NOT EMIT. The parallel lowering rewrites this terminator itself; a
    /// rewrite here as well would produce two `scf.yield`s for one `affine.yield`.
    #[test]
    fn under_a_parallel_parent_the_pattern_declines() {
        assert_eq!(
            lower_affine_yield(Parent::ScfParallel, &[Val(1)]),
            YieldRewrite::Declined
        );
    }

    /// 🎯 001/384 — AND THE TWO TERMINATORS PRINT AS THE REFERENCE WRITES THEM.
    #[test]
    fn the_terminators_print() {
        use crate::islands::dataflow_ir::print;
        let mut out = String::new();
        print::emit(
            &mut out,
            &DfirOp::Scf(dfir_op::scf::Op::Yield {
                operands: vec![Val(3)],
            }),
            0,
        );
        assert_eq!(out.trim(), "scf.yield %3");
        out.clear();
        print::emit(
            &mut out,
            &DfirOp::Affine(dfir_op::affine::Op::Yield {
                operands: Vec::new(),
            }),
            0,
        );
        assert_eq!(out.trim(), "affine.yield");
    }
}
