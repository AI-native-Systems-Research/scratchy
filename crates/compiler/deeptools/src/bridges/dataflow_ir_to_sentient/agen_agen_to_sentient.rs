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

//! `AgenToSentient.cpp` — 5 of bridge 2's 384 functions (dependency level(s) [0, 9, 10]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e027_constructLoadAndSendStmt` | 027/384 | 4 | `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:230` |
//! | `e028_constructReceiveAndStoreStmt` | 028/384 | 4 | `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:248` |
//! | `e029_insertCopyAndAddStmtsHelper` | 029/384 | 17 | `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:502` |
//! | `e382_fuseLoadOrStoreChainOps` | 382/384 | 144 | `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22` |
//! | `e384_runOnOperation` | 384/384 | 81 | `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169` |
//!
//! Original files homed here: `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp`, `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp`


use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::agen;
use crate::islands::dataflow_ir::{self as dfir};
use crate::islands::sentient::dialects::Op as SenOp;

use super::{Bound, Consts};

/// WHICH `load_and_extract_scalar` OF THIS UNIT THE NEXT ONE IS.
///
/// ⭐⭐ THE INDEX IS THE WIRE BETWEEN TWO OPS, not a counter for reporting.
/// `AgenToSentient.cpp:24-27` says it outright: *"extract_idx will increment every time a
/// load_and_extract operation is created. This index is used to connect load_and_extract operations
/// to the indirect loads that will use them."* So an off-by-one here does not print a wrong number,
/// it points an indirect load at somebody else's scalar.
///
/// ⛔ ITS SCOPE IS ONE UNIT. `extract_idx` is a local of
/// [`fuse_load_or_store_chain_ops`](self::fuse_load_or_store_chain_ops), declared before the
/// candidate loop and outside it (`AgenToSentient.cpp:27`), and the pass calls that function once
/// per `dataflow.program_unit` — the walk at `AgenToSentient.cpp:171`, the call at `:213`. Hoisting
/// it to the program would renumber every unit after the first.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExtractIdx(u32);

impl ExtractIdx {
    /// The index this extraction gets, advancing the counter — `extract_idx++`.
    ///
    /// ⛔ SATURATING, NOT WRAPPING. The reference's `unsigned` wraps to 0, which would hand a later
    /// extraction an index an earlier one already owns; saturating keeps the sequence monotone. Four
    /// billion `load_and_extract_scalar`s in one program unit is not a program any device runs, so
    /// the ceiling is unreachable either way — what it must not do is silently start again at zero.
    pub fn issue(&mut self) -> Self {
        let mine = *self;
        self.0 = self.0.saturating_add(1);
        mine
    }

    /// How many have been issued to this unit so far.
    #[must_use]
    pub fn issued(self) -> u32 {
        self.0
    }
}

/// HOW MANY DATAFLOWIR OPS ONE LOWERED CANDIDATE TOOK WITH IT.
///
/// ⭐⭐ THIS IS `to_be_deleted_list`, COUNTED. The reference collects the ops a lowering consumed
/// into a `SmallVector` and erases them after the dispatch (`AgenToSentient.cpp:53, 163`), and the
/// list is not always just the candidate: `lowerVectorStoreOp` pushes the candidate AND the op that
/// produced the value it stored (`Helper.cpp:3098-3101`), while the composite helper pushes the
/// candidate alone (`Helper.cpp:2970`). Here the walk is a window over a statement list rather than
/// a mutable graph ([`super::statement`]), so "erased" and "consumed from the head of the window" are
/// the same fact — and the COUNT is what the walk needs to not lower those operands a second time.
///
/// ⛔ A LOWERING THAT UNDER-REPORTS EMITS ITS OWN INPUTS AGAIN. That is why this is a newtype and not
/// a `usize`: the one number this function returns is the one number that must not be guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Consumed(usize);

impl Consumed {
    /// The count, for the walk that advances by it.
    #[must_use]
    pub fn ops(self) -> usize {
        self.0
    }
}

/// FUSE ONE NON-COMPUTE CANDIDATE INTO ITS SENTIENT OP.
///
/// `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22` — *"This method tries to fuse
/// non-compute ops into sentient ops."*
///
/// # ⭐⭐ THE REFERENCE'S LOOP IS THIS CRATE'S WALK
///
/// `AgenToSentient.cpp:29-48` is `while (true)` around a preorder walk that interrupts on the FIRST
/// op of one of twelve kinds, lowers it, erases what it consumed, and goes round again — the walk
/// restarts from the top because the erase invalidated it. [`super::body`] is that loop already: it
/// steps a cursor over one unit's statements, and each step consumes the head of the window and
/// advances by exactly what it consumed. So this function is the loop BODY — one candidate — and
/// `while (true)` / `if (!vector_op) break;` is `while i < unit.body.len()`. Writing the outer loop
/// again here would walk each statement list twice.
///
/// # ⭐⭐ THE DISPATCH ORDER IS THE REFERENCE'S, AND IT IS NOT ALPHABETICAL
///
/// `AgenToSentient.cpp:55-162` is a `dyn_cast` chain, so the order is a real part of the function:
/// a kind is tried only after every kind above it has failed. The twelve, in that order, with the
/// campaign unit that lowers each:
///
/// | # | candidate kind (`AgenToSentient.cpp:33-37`) | lowered by | island variant |
/// |---|---|---|---|
/// | 1 | `agen.vector_load` | `e153`? `e312` : `e314` | [`agen::Op::VectorLoad`] |
/// | 2 | `agen.vector_store` | `e154`? `e313` : `e315` | [`agen::Op::VectorStore`] |
/// | 3 | `agen.composite_load` | `e329` | — |
/// | 4 | `agen.composite_store` | `e330` | — |
/// | 5 | `agen.composite_load_and_store` | `e331` | [`agen::Op::CompositeLoadAndStore`] |
/// | 6 | `agen.indirect_vector_load` | `e316` | — |
/// | 7 | `agen.indirect_vector_store` | `e317` | — |
/// | 8 | `agen.composite_indirect_load` | `e332` | — |
/// | 9 | `agen.composite_indirect_store` | `e333` | — |
/// | 10 | `agen.composite_indirect_load_and_store` | `e334` | — |
/// | 11 | `agen.symbolic_vector_load` | `e374` | — |
/// | 12 | `agen.symbolic_vector_store` | `e375` | — |
///
/// ⛔ NINE OF THE TWELVE HAVE NO ISLAND VARIANT and so cannot be a candidate here at all: the
/// DataflowIR island declares four `agen` ops, and a kind this crate cannot construct is a kind this
/// dispatch cannot meet. The `match` below is therefore exhaustive over
/// [`agen::Op`] rather than over the twelve — which makes ADDING the tenth op to the island a
/// build error here, in the one place that has to grow an arm for it.
///
/// # ⛔⛔ TWO ARMS ARE `todo!` AND THAT IS THE POINT
///
/// `e314_lowerVectorLoadOp` and `e315_lowerVectorStoreOp` are unported (level 8, this campaign), as
/// are the two pattern predicates that choose between them and their extracting siblings. Until they
/// land, an `agen.vector_load` reaching this dispatch is a named gap and not a wrong program. The
/// gain over the generic *"lower a statement this bridge has not met"* this replaces is that the
/// build now says WHICH unit is missing.
///
/// ⛔ AND NOT A STAND-IN. Lowering a `vector_load` as if it were the transfer below would emit a
/// `load_and_store` for a program that asked for a load — the exact substitution the campaign
/// forbids.
///
/// # ⛔ WHAT THE PORT DROPS, AND WHY IT IS SOUND
///
/// - `checkBasicConditions` (`e210`, `Helper.cpp:58`) refuses an access whose order is not a
///   permutation or whose set is not hyper-rectangular. Every access this island can construct is
///   built by [`crate::islands::dataflow_ir::dialects::agen`]'s own `identity_map`/`lane_set`, which
///   produce exactly a permutation (the identity) and exactly a hyper-rectangle (outer dims pinned,
///   the lane axis a range). The check is vacuous over constructible input, and a runtime re-check
///   would be a refusal this crate does not have.
/// - `signalPassFailure()` / `emitError` / `return failure()` have no counterpart: the ported
///   lowerings are total functions, and `crates/compiler/deeptools/CLAUDE.md` forbids a `Result` at
///   this seam. An input that cannot be lowered is a `todo!` at build-fail time, not an error value.
/// - `llvm_unreachable("unsupported operation")` becomes the absence of a wildcard arm: unreachable
///   by the type rather than at runtime.
///
/// # ⛔⛔ THE COMPONENT GATE IS `e384`'s AND IS NOT PORTED, SO THIS DISPATCH SEES EVERY UNIT
///
/// The reference reaches this function only for a unit whose component is one of
/// `L0LU, L0SU, LXLU, LXSU, L3SU, L3LU` — `if (!is_any_of(comp, ...)) return;`
/// (`AgenToSentient.cpp:174-176`), inside `e384_runOnOperation`, which is scheduled separately and
/// unported. Until it lands, [`super::body`] hands this dispatch the `agen` ops of EVERY unit,
/// including a compute one. That widens what the two `todo!` arms can fire on; it cannot widen what
/// gets emitted, because the only emitting arm is the transfer and a transfer is a transfer on any
/// component. The gate goes in with `e384`, in `e384`'s own anchor.
///
/// Replaces: e382_fuseLoadOrStoreChainOps
pub(super) fn fuse_load_or_store_chain_ops<A: Arch>(
    op: &agen::Op,
    unit: &dfir::ProgramUnit<A>,
    extract: &mut ExtractIdx,
    bound: &Bound,
    consts: &Consts,
    out: &mut Vec<SenOp>,
) -> Consumed {
    match op {
        // ── 1. `agen.vector_load` (`AgenToSentient.cpp:55-72`) ───────────────────────────────────
        agen::Op::VectorLoad { .. } => todo!(
            "e153_isLoadAndExtractScalarPattern then e312_lowerExtractVectorLoadOp (extract {}) or \
             e314_lowerVectorLoadOp: agen.vector_load on {:?}",
            extract.issued(),
            unit.on.kind()
        ),

        // ── 2. `agen.vector_store` (`AgenToSentient.cpp:73-89`) ──────────────────────────────────
        agen::Op::VectorStore { .. } => todo!(
            "e154_isReceiveAndExtractScalarPattern then e313_lowerExtractVectorStoreOp (extract {}) \
             or e315_lowerVectorStoreOp: agen.vector_store on {:?}",
            extract.issued(),
            unit.on.kind()
        ),

        // ── 5. `agen.composite_load_and_store` (`AgenToSentient.cpp:102-109`) ────────────────────
        //
        // ⭐ THE EMISSION IS `e331_lowerCompositeLoadAndStoreOp`'s AND IT ALREADY EXISTS. The spine's
        // [`super::load_and_store`] is the `sentient.load_and_store` this arm has to produce, byte
        // compared against the reference's own output; `e331` is its anchored home and is scheduled
        // separately. Emitting a second one here would be two answers to one question.
        agen::Op::CompositeLoadAndStore(transfer) => {
            out.push(super::load_and_store(transfer, bound, consts));
            Consumed(1)
        }

        // ── not a candidate: a terminator (`agen.yield`) ─────────────────────────────────────────
        //
        // ⛔ NOT REACHABLE FROM A UNIT'S STATEMENT LIST. `agen.yield` terminates a composite
        // transfer's REGION and is walked as part of it, never as the head of a unit body. It gets an
        // arm because the `match` is exhaustive by design, not because the reference has one.
        agen::Op::Yield => todo!(
            "agen.yield reached a unit's statement list on {:?} — it terminates a composite \
             transfer's region",
            unit.on.kind()
        ),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{Consumed, ExtractIdx};

    /// 🎯 `extract_idx` STARTS AT ZERO AND THE FIRST ISSUE HANDS OUT ZERO.
    ///
    /// `AgenToSentient.cpp:27` declares `unsigned extract_idx = 0;` and the lowerings take it BY
    /// REFERENCE, incrementing after use — so the first `load_and_extract_scalar` of a unit is
    /// extract 0, and an indirect load wired to extract 1 is wired to the second.
    #[test]
    fn the_first_extract_of_a_unit_is_zero() {
        let mut extract = ExtractIdx::default();
        assert_eq!(extract.issued(), 0);
        assert_eq!(extract.issue().issued(), 0);
        assert_eq!(extract.issued(), 1);
        assert_eq!(extract.issue().issued(), 1);
        assert_eq!(extract.issued(), 2);
    }

    /// 🎯 A FRESH COUNTER PER UNIT — the reference's scope, not the program's.
    ///
    /// `extract_idx` is a local of `fuseLoadOrStoreChainOps`, which the pass calls once per
    /// `dataflow.program_unit` (`AgenToSentient.cpp:171, 213`). Two units therefore both start at 0;
    /// a program-wide counter would renumber every unit after the first and point its indirect loads
    /// at scalars belonging to an earlier unit.
    #[test]
    fn each_unit_starts_its_own_extract_numbering() {
        let mut first = ExtractIdx::default();
        let _ = first.issue();
        let _ = first.issue();
        assert_eq!(first.issued(), 2);

        let mut second = ExtractIdx::default();
        assert_eq!(second.issue().issued(), 0);
    }

    /// 🎯 THE CONSUMED COUNT IS WHAT THE WALK ADVANCES BY.
    ///
    /// One `agen.composite_load_and_store` is one `sentient.load_and_store` and eats exactly the one
    /// op — `lowerCompositeLoadAndStoreOp` (`Helper.cpp:3148`) delegates to the composite helper,
    /// whose `to_be_deleted` gets the candidate and nothing else (`Helper.cpp:2970`). A count of 0 would
    /// spin [`super::super::body`]'s cursor forever; a count of 2 would drop the next statement.
    #[test]
    fn one_transfer_consumes_one_op() {
        assert_eq!(Consumed(1).ops(), 1);
    }
}
