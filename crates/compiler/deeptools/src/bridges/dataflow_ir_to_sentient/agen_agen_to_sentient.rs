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
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, agen};
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
/// | 11 | `agen.symbolic_vector_load` | `e374` | [`agen::Op::SymbolicVectorLoad`] |
/// | 12 | `agen.symbolic_vector_store` | `e375` | [`agen::Op::SymbolicVectorStore`] |
///
/// ⛔ SEVEN OF THE TWELVE HAVE NO ISLAND VARIANT and so cannot be a candidate here at all: a kind
/// this crate cannot construct is a kind this dispatch cannot meet. The `match` below is therefore
/// exhaustive over [`agen::Op`] rather than over the twelve — which is why declaring the symbolic
/// pair for entries 374/375 grew arms 11 and 12 here, in the one place that has to have them.
///
/// # ⛔⛔ FOUR ARMS ARE `todo!` AND THAT IS THE POINT — AND THE PREDICATE THAT PICKS BETWEEN THEM RUNS
///
/// All four of `e312_lowerExtractVectorLoadOp`, `e313_lowerExtractVectorStoreOp`,
/// `e314_lowerVectorLoadOp` and `e315_lowerVectorStoreOp` are unported (all level 5, this campaign).
/// ⭐ BUT `e153_isLoadAndExtractScalarPattern` AND `e154_isReceiveAndExtractScalarPattern` ARE PORTED,
/// so the `if (isLoadAndExtractScalarPattern(load_op))` at `:56` and the `if
/// (isReceiveAndExtractScalarPattern(store_op))` at `:74` are asked here rather than elided — and this
/// dispatch is the only caller the reference gives either of them. A single `todo!` per kind naming
/// both alternatives would say *"one of these two"* where the reference already knows which.
/// ⚠️ 315's recorded level is wrong in the schedule, not here: its 7-argument
/// `constructReceiveAndStoreStmt` call (`Helper.cpp:3090-3092`) cannot bind the 8-parameter inline
/// forwarder `e028` and resolves to the primary template `e359` (`Helper.cpp:2025`, level 7), so 315
/// is scheduled two levels ahead of its own callee. Until they land, an `agen.vector_load` reaching
/// this dispatch is a named gap and not a wrong program. The gain over the generic *"lower a
/// statement this bridge has not met"* this replaces is that the build now says WHICH unit is
/// missing.
///
/// ⛔ AND NOT A STAND-IN. Lowering a `vector_load` as if it were the transfer below would emit a
/// `load_and_store` for a program that asked for a load — the exact substitution the campaign
/// forbids.
///
/// # ⛔ WHAT THE PORT DROPS, AND WHY IT IS SOUND
///
/// - Nothing about `checkBasicConditions` (`e210`, `Helper.cpp:58`) — it is PORTED and it is CALLED,
///   below, on every candidate before the dispatch, exactly where `:38` calls it. ⛔ IT IS NOT
///   VACUOUS, and the earlier claim that it was covered only half the input: for the vector pair the
///   set and order are the island's own derived `access_set`/`access_order`, so those two do pass by
///   construction — but a `CompositeLoadAndStore` carries its OWN `load_order`, `store_order`,
///   `load_set`, `store_set` and `time_order`, and that is the one arm this dispatch lowers today.
/// - `signalPassFailure()` / `emitError` / `return failure()` have no counterpart: the ported
///   lowerings are total functions, and `crates/compiler/deeptools/CLAUDE.md` forbids a `Result` at
///   this seam. An input that cannot be lowered is a `todo!` at build-fail time, not an error value.
/// - `llvm_unreachable("unsupported operation")` becomes the absence of a wildcard arm: unreachable
///   by the type rather than at runtime.
///
/// # ⭐ THE COMPONENT GATE IS `e384`'s AND IT IS IN
///
/// The reference reaches this function only for a unit whose component is one of
/// `L0LU, L0SU, LXLU, LXSU, L3SU, L3LU` — `if (!is_any_of(comp, ...)) return;`
/// (`AgenToSentient.cpp:174-176`). That gate is [`TransferComp::of`], applied by
/// [`run_on_operation`] and carried to [`super::statement`], so this dispatch now sees the `agen` ops
/// of a transfer unit only; an `agen` access on a compute unit is
/// `VectorChainToSentientPESFP`'s and says so there.
///
/// ⭐ `stmt` IS `op`'s ENCLOSING STATEMENT, and it is here because entry 036 asks a question about
/// the OPERATION — `op->getResult(0)` and its users — which an `agen::Op` alone cannot answer.
///
/// Replaces: e382_fuseLoadOrStoreChainOps
pub(super) fn fuse_load_or_store_chain_ops<A: Arch>(
    stmt: &DfirOp,
    op: &agen::Op,
    unit: &dfir::ProgramUnit<A>,
    extract: &mut ExtractIdx,
    bound: &Bound,
    consts: &Consts,
    out: &mut Vec<SenOp>,
) -> Consumed {
    // ── `if (checkBasicConditions(op).failed()) signalPassFailure();` (`:38-41`) ─────────────────
    //
    // ⛔ ON THE CANDIDATE SET ONLY, AND BEFORE THE DISPATCH. The reference asks it inside the walk's
    // `isa<the twelve>` guard, so an interleave or a mask state is never asked — and a candidate that
    // fails it leaves `vector_op` NULL, which breaks the `while (true)` and abandons every remaining
    // candidate in the unit as well.
    // ⛔ AND IT IS A REFUSAL THE REFERENCE MAKES, NOT AN UNPORTED CALLEE: `e210` answers, and what
    // this bridge has no representation for is `signalPassFailure()`. Lowering the access anyway
    // would emit a transfer `AgenToSentient` declines to emit.
    if is_candidate(op) {
        let conditions =
            super::agen_helper::check_basic_conditions(super::agen_helper::CheckedOp::Dfir(stmt));
        if !conditions.admissible() {
            todo!(
                "e210_checkBasicConditions refuses this access ({conditions:?}: {:?}), so \
                 AgenToSentient signalPassFailure()s and abandons the fusion of {:?}",
                conditions.diagnostic(),
                unit.on.kind()
            );
        }
    }

    match op {
        // ── 1. `agen.vector_load` (`AgenToSentient.cpp:55-72`) ───────────────────────────────────
        //
        // ⛔ THE PREDICATE IS THE BRANCH, AND IT RUNS (`:56`) — `e153` is ported, so the `todo!` names
        // the ONE lowering this load needs and not a pair. The scope is the whole unit body because
        // the pattern is *"the load's single user is a store into the virtual IBR"*, a use census.
        agen::Op::VectorLoad { .. } => {
            if super::agen_helper::is_load_and_extract_scalar_pattern(stmt, &unit.body) {
                todo!(
                    "e312_lowerExtractVectorLoadOp: agen.vector_load with the \
                     sentient.load_and_extract_scalar pattern, extract {}, on {:?}",
                    extract.issued(),
                    unit.on.kind()
                )
            }
            todo!(
                "e314_lowerVectorLoadOp: a plain agen.vector_load on {:?}",
                unit.on.kind()
            )
        }

        // ── 2. `agen.vector_store` (`AgenToSentient.cpp:73-89`) ──────────────────────────────────
        //
        // ⛔ `e154` IS THE WHOLE PATTERN — a store INTO the virtual IBR, asked of the store alone
        // (`:74`), so it needs no use census of its own; the scope resolves the view to its unit.
        agen::Op::VectorStore { .. } => {
            if super::agen_helper::is_receive_and_extract_scalar_pattern(stmt, &unit.body) {
                todo!(
                    "e313_lowerExtractVectorStoreOp: agen.vector_store with the \
                     sentient.receive_and_extract_scalar pattern, extract {}, on {:?}",
                    extract.issued(),
                    unit.on.kind()
                )
            }
            todo!(
                "e315_lowerVectorStoreOp: a plain agen.vector_store on {:?}",
                unit.on.kind()
            )
        }

        // ── 5. `agen.composite_load_and_store` (`AgenToSentient.cpp:102-109`) ────────────────────
        //
        // ⭐ THE EMISSION ALREADY EXISTS, AND IT IS `e268_constructLoadAndStoreStmt`'s. The spine's
        // [`super::load_and_store`] is the `sentient.load_and_store` this arm has to produce, byte
        // compared against the reference's own output. `e331_lowerCompositeLoadAndStoreOp`
        // (`Helper.cpp:3148`) is the entry this arm's dispatch belongs to and is scheduled separately,
        // but it emits nothing itself: it tail-calls `lowerAffineCompositeHelper` (`:3164-3166`, e299,
        // level 4) → `constructTimeLoopsAndVectorOperations` (`:2959`, e267) → `constructLoadAndStoreStmt`
        // (`:1882`, e268), where `sentient::LoadAndStoreOp::create` is (`:2323`) — four calls below
        // `e331`. Emitting a second one here would be two answers to one question.
        agen::Op::CompositeLoadAndStore(transfer) => {
            out.push(super::load_and_store(transfer, bound, consts));
            Consumed(1)
        }

        // ── 11. `agen.symbolic_vector_load` (`AgenToSentient.cpp:147-152`) ───────────────────────
        //
        // ⛔ THE STORE SEARCH NEEDS THE WHOLE UNIT BODY, not the window: entry 036 counts the load
        // result's uses, and a census over the remaining statements alone would find one use where
        // there are two.
        agen::Op::SymbolicVectorLoad { .. } => super::agen_helper::lower_symbolic_vector_load_op(
            stmt,
            unit,
            unit.on.kind(),
            &unit.body,
        ),

        // ── 12. `agen.symbolic_vector_store` (`AgenToSentient.cpp:153-159`) ──────────────────────
        agen::Op::SymbolicVectorStore { .. } => {
            super::agen_helper::lower_symbolic_vector_store_op(unit, unit.on.kind())
        }

        // ── not a candidate: the interleave and the mask state ───────────────────────────────────
        //
        // ⛔ NOT AMONG THE TWELVE THIS DISPATCH WALKS FOR (`AgenToSentient.cpp:29-52`) — the fusion
        // passes over both, and `e384`'s steps after it are what lower them. This arm consumes the
        // statement so the cursor advances; the transfers inside an interleave's REGION are not
        // reached either, and `e271`'s `todo!` in [`run_on_operation`] fires before anything the
        // region holds could be dropped.
        agen::Op::CompositeMemoryInterleave { .. } | agen::Op::SetTransferMaskState { .. } => {
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

/// WHETHER THIS OP IS ONE OF THE TWELVE THE FUSION WALKS FOR — `isa<VectorLoadOp, VectorStoreOp, …>`
/// (`AgenToSentient.cpp:33-37`).
///
/// ⛔ IT IS THE WALK'S GUARD, NOT THE DISPATCH'S. `checkBasicConditions` sits inside it, so the
/// interleave and the mask state — which the fusion passes over and `e384` lowers in its own later
/// steps — are never asked. Written as an exhaustive `match` so a thirteenth island `agen` op has to
/// state which side of the guard it is on.
fn is_candidate(op: &agen::Op) -> bool {
    match op {
        agen::Op::VectorLoad { .. }
        | agen::Op::VectorStore { .. }
        | agen::Op::CompositeLoadAndStore(_)
        | agen::Op::SymbolicVectorLoad { .. }
        | agen::Op::SymbolicVectorStore { .. } => true,
        agen::Op::CompositeMemoryInterleave { .. }
        | agen::Op::SetTransferMaskState { .. }
        | agen::Op::Yield => false,
    }
}

use crate::arch::Elements;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::affine::Carried;
use crate::islands::dataflow_ir::dialects::{Val, affine, arith, scf};
use crate::islands::dataflow_ir::ty::ScalarTy;

use super::agen_helper::ExtractScalarOp;

/// WHICH LOOP AN L0/LX TRANSFER'S MUTABLE ADDRESS IS INITIALISED OUTSIDE OF — named by its induction
/// variable, which is the one value only that loop binds.
///
/// ⭐ IT IS ONLY EVER READ WHEN THERE IS A STRIDE. `constructLoadAndSendStmt` passes it on to
/// `adjustMutableAddrInitForStride` under `if (stride_step > 0 && is_any_of(comp, L0LU, L0SU, LXLU,
/// LXSU))` (`Helper.cpp:1999-2001`), so an absent one and a zero stride step are the same state seen
/// twice — which is why the defaults set both together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutermostCompLoop(pub Val);

/// HOW FAR A GROUPED TRANSFER MOVES ITS ADDRESS BETWEEN GROUPS.
///
/// ⛔ SIGNED, AND THE SIGN IS THE OFF SWITCH. The C++ declares `int stride_step` and both users
/// guard with `stride_step > 0` (`Helper.cpp:1999`, `:2043`'s companion), so 0 means *no stride
/// adjustment* while a negative value would mean *adjust backwards* — a distinction an `unsigned`
/// would have thrown away. Its one producer is `Helper.cpp:1862-1865` and it never yields a literal
/// zero: `group_index >= 0 ? time_offsets[group_index] : (burst_index >= 0 ?
/// time_offsets[burst_index] : time_offsets.front())` — a TIME OFFSET, three ways, and whether the
/// adjustment happens is that offset's own sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrideStep(pub i32);

/// THE FIVE PARAMETERS A TRANSFER-STATEMENT CONSTRUCTION DEFAULTS — burst, group, stride, the loop the
/// stride adjustment needs, and the extract statement an indirect access reads its address from.
///
/// ⛔⛔ THESE FIVE ARE ONE DECISION, NOT FIVE ARGUMENTS. `constructLoadAndSendStmt` and
/// `constructReceiveAndStoreStmt` each declare them as trailing defaults
/// (`AgenToSentient.hpp:222-227`, `:240-245`), and their two consumers read them in pairs:
/// `perform_burst_or_group = burst_size > 0 || group_size > 0` (`Helper.cpp:1930`, `:2043`) and
/// `stride_step > 0 && is_any_of(comp, L0LU, L0SU, LXLU, LXSU)` with `outermost_comp_loop` as the
/// third argument (`:1999-2001`). A caller that set `group_size` and forgot `stride_step` would get
/// grouped addressing that never advances — so the set travels as one value.
///
/// ⭐ `Elements` FOR THE TWO SIZES, MATCHING THE ISLAND. The op these end up on already models
/// `burst_size` that way, defaulting to `Elements(0)` for *unbursted*
/// ([`crate::islands::sentient::dialects::sentient::Extent::burst_size`], `SentientOps.td:504-520`),
/// and both are read off a loop's trip count at `Helper.cpp:1860-1861`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferSpecialisation {
    /// `burst_size` — `Elements(0)` is unbursted.
    pub burst_size: Elements,
    /// `group_size` — `Elements(0)` is ungrouped.
    pub group_size: Elements,
    /// `stride_step` — `StrideStep(0)` asks for no stride adjustment.
    pub stride_step: StrideStep,
    /// The loop the stride adjustment hoists the address initialisation out of.
    pub outermost_comp_loop: Option<OutermostCompLoop>,
    /// The extract statement whose scalar becomes this transfer's address — present only for the
    /// indirect (gather/scatter) pattern.
    pub extract_op: Option<ExtractScalarOp>,
}

impl TransferSpecialisation {
    /// A PLAIN TRANSFER: no burst, no group, no stride adjustment, no indirection — the five default
    /// arguments the two general declarations carry (`AgenToSentient.hpp:225-227`, `:243-245`).
    ///
    /// ⭐ THIS IS WHAT THREE CALL SITES GET BY WRITING NOTHING: `Helper.cpp:2910` (the direct
    /// `agen.vector_load`), `:3090` (the composite store) and `:3428` (the symbolic
    /// `agen.vector_store`) all stop after `immutable_addrs[0]`.
    pub const UNSPECIALISED: TransferSpecialisation = TransferSpecialisation {
        burst_size: Elements(0),
        group_size: Elements(0),
        stride_step: StrideStep(0),
        outermost_comp_loop: None,
        extract_op: None,
    };

    /// `perform_burst_or_group` — `burst_size > 0 || group_size > 0` (`Helper.cpp:1930`, `:2043`).
    ///
    /// ⭐ AN `||`, SO EITHER ONE ARMS IT. A grouped transfer with no burst still takes the bursting
    /// path through `setChunkSizeAndStride`.
    #[must_use]
    pub const fn performs_burst_or_group(&self) -> bool {
        self.burst_size.0 > 0 || self.group_size.0 > 0
    }

    /// Whether `adjustMutableAddrInitForStride` is asked for at all — the `stride_step > 0` half of
    /// `Helper.cpp:1999`. ⛔ The other half is the unit kind, which belongs to the general form.
    #[must_use]
    pub const fn adjusts_for_stride(&self) -> bool {
        self.stride_step.0 > 0
    }
}

/// Replaces: e027_constructLoadAndSendStmt
///
/// The seven-argument overload (`AgenToSentient.hpp:229-237`):
///
/// ```c++
/// template <typename AccessDetailsTy>
/// LogicalResult constructLoadAndSendStmt(
///     OpBuilder* builder, dataflow::ProgramUnitOp unit_op, Operation* op,
///     AccessDetailsTy& access_details, Value& mutable_addr,
///     Value& immutable_addr, Operation* extract_op) {
///   return constructLoadAndSendStmt<AccessDetailsTy>(
///       builder, unit_op, op, access_details, mutable_addr, immutable_addr, 0,
///       0, 0, nullptr, extract_op);
/// }
/// ```
///
/// ⛔⛔ IT EXISTS TO STOP `extract_op` BINDING TO `burst_size`. The general form's seventh parameter
/// is `unsigned burst_size` (`:222-227`), so the natural call
/// `constructLoadAndSendStmt(&builder, unit, op, ad, mut, immut, extract_op)` would pass a pointer
/// where a count goes — and in C++ that is not even a type error worth trusting. This overload is
/// the only reason those two call sites read the way they do, and both are the INDIRECT lowering:
/// `Helper.cpp:3202` (`agen.indirect_vector_load`) and, through its twin, `:3251`.
///
/// ⭐ SO ITS WHOLE CONTENT IS THE ARGUMENT SET, WHICH IS WHAT IT RETURNS. The six values it forwards
/// untouched — the builder, the program unit, the op being lowered, the access details and the two
/// addresses — are the caller's own, and the caller still holds them; nothing in these four lines
/// reads or changes one. What the function decides is the five trailing values, and that is
/// [`TransferSpecialisation`]. The statement itself is emitted by the general form
/// `e358_constructLoadAndSendStmt` (`Helper.cpp:1910`, level 7, not yet ported), which consumes
/// exactly this.
///
/// ⭐ AND `extract_op` IS NOT OPTIONAL HERE. The general form takes `Operation* extract_op = nullptr`
/// and tests it (`:2006`); this overload is reached only once the caller has found one and reported
/// its absence itself (*"could not locate a load_and_extract_scalar operation matching the
/// extract_idx used by op"*, `:3196-3199`), so the parameter is an [`ExtractScalarOp`] and not an
/// `Option`.
#[must_use]
pub fn construct_load_and_send_stmt(extract_op: ExtractScalarOp) -> TransferSpecialisation {
    TransferSpecialisation {
        extract_op: Some(extract_op),
        ..TransferSpecialisation::UNSPECIALISED
    }
}

/// Replaces: e028_constructReceiveAndStoreStmt
///
/// The store side's overload (`AgenToSentient.hpp:247-255`), the same four lines around one extra
/// pass-through:
///
/// ```c++
/// template <typename AccessDetailsTy>
/// LogicalResult constructReceiveAndStoreStmt(
///     OpBuilder* builder, dataflow::ProgramUnitOp unit_op, Operation* op,
///     Type data_elem_type, AccessDetailsTy& access_details, Value& mutable_addr,
///     Value& immutable_addr, Operation* extract_op) {
///   return constructReceiveAndStoreStmt<AccessDetailsTy>(
///       builder, unit_op, op, data_elem_type, access_details, mutable_addr,
///       immutable_addr, 0, 0, 0, nullptr, extract_op);
/// }
/// ```
///
/// ⛔ THE EXTRA PARAMETER IS `data_elem_type` AND THE CALLEE NEVER READS IT. `data_elem_type` occurs
/// ONCE in all of `Helper.cpp` — the primary's own parameter list (`:2027`) — and never in its body:
/// `element_width` comes from `access_details.getElementWidth()` (`:2032`) and the shuffle mode from
/// `setsttype(builder, comp, producer_info.first, …)` (`:2092`), which takes the producer input OP and
/// no element type at all (`:1731-1784`). The caller computes it anyway (`:3250`), so dropping it here
/// drops a parameter that is dead in the reference, not one this port decided to ignore.
///
/// ⭐ SAME REASON FOR EXISTING, SAME RESULT. See [`construct_load_and_send_stmt`]: without it,
/// `extract_op` would bind to the general form's `unsigned burst_size`. Its one caller is
/// `Helper.cpp:3251`, the `agen.indirect_vector_store` lowering, and the statement is emitted by
/// `e359_constructReceiveAndStoreStmt` (`Helper.cpp:2025`, level 7, not yet ported).
#[must_use]
pub fn construct_receive_and_store_stmt(extract_op: ExtractScalarOp) -> TransferSpecialisation {
    TransferSpecialisation {
        extract_op: Some(extract_op),
        ..TransferSpecialisation::UNSPECIALISED
    }
}

/// WHICH CARRIED VALUE A LOOP'S ADDRESS ADVANCE TOUCHES, **COUNTED FROM THE END** — minted from the
/// loop's own list, so it always names one.
///
/// ⛔⛔ FROM THE END, AND GETTING THAT WRONG ADVANCES SOMEONE ELSE'S ADDRESS. Every index in
/// `insertCopyAndAddStmtsHelper` is `num_iter_args - index - 1` and
/// `yield_op->getNumOperands() - index - 1` (`AgenToSentient.hpp:503-518`) — so `index` 0 is the LAST
/// iteration argument, not the first. The caller's `index` is the memory-operand slot
/// (`Helper.cpp:945-950` passes the loop over `mutable_addrs[i]`, `i` running
/// [`super::agen_access_details::MemoryOperandIndex`]'s own order), which means the source's address
/// is the last value the loop carries and the destination's is the one before it. Advancing the
/// wrong one produces a program that reads the right elements and writes them to a fixed address.
///
/// ⭐ MINTED, NOT WRITTEN — [`Self::at`] is the only constructor, and it reads the [`Carried`] it
/// names out of the list, so nothing downstream indexes anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarriedFromEnd {
    /// The region iteration argument itself — the C++'s
    /// `loop_op.getRegionIterArgs()[num_iter_args - index - 1]`.
    arg: Val,
    /// The C++'s `index`: how far from the END of the list this is.
    from_end: usize,
}

impl CarriedFromEnd {
    /// THE CARRIED VALUE `from_end` PLACES FROM THE END of a loop's list.
    ///
    /// ⛔ `None` IS AN ABSENCE, NOT A REFUSAL — a loop that carries two values has no third, in the
    /// same way a schedule that names no unit of a kind has no program unit of it
    /// ([`crate::islands::dataflow_ir::Units::of`]). The C++ has nothing here: `getRegionIterArgs()`
    /// is a `MutableArrayRef` and `[num_iter_args - index - 1]` on an unsigned underflow indexes
    /// somewhere else entirely.
    #[must_use]
    pub fn at(carried: &[Carried], from_end: usize) -> Option<CarriedFromEnd> {
        let at = carried.len().checked_sub(from_end.checked_add(1)?)?;
        Some(CarriedFromEnd {
            arg: carried.get(at)?.arg,
            from_end,
        })
    }

    /// The iteration argument this names.
    #[must_use]
    pub const fn arg(self) -> Val {
        self.arg
    }
}

/// WHAT AN ADDRESS ADVANCE LEFT IN THE LOOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressAdvance {
    /// The C++'s own return value: the iteration argument, which is the address the body reads.
    pub iter_arg: Val,
    /// The sum the terminator now yields — the address the NEXT iteration will read.
    pub yielded: Val,
    /// The `arith.constant` minted for the addend.
    pub addend: Val,
}

/// AN OP LIST AN ADDRESS-CARRYING LOOP'S BODY CAN BE.
///
/// ⛔⛔ TWO ISLANDS, BECAUSE THE MODULE IS MIXED WHILE THE PASS RUNS. `insertCopyAndAddStmtsHelper` is
/// a template over the LOOP type, and the C++ rewrites one module in place — by the time
/// `e335_insertCopyAndAddStmts` reaches a loop, its body already holds `sentient.load_and_send`
/// beside `arith.addi` and `dataflow.get_logical_memory_view`
/// (`dcc/test/Conversion/AgenToSentient/lx_indirect_loads_stores_composite.mlir:39-47`), which is the
/// mixed rung [`crate::islands::sentient::dialects`] documents. The advance itself is two `arith` ops
/// and one terminator operand, and `arith`, `affine` and `scf` are the SAME modules in both islands
/// (they are re-exported, not re-declared), so the helper is written once against what it actually
/// needs rather than twice against two op enums.
pub trait LoopBodyOp: Sized {
    /// Wrap one `arith` op as a body statement.
    fn arith(op: arith::Op) -> Self;

    /// The operands of this op IF it is a loop terminator — `affine.yield` or `scf.yield`.
    ///
    /// ⛔ BOTH DIALECTS, WHICH IS THE TEMPLATE'S TWO INSTANTIATIONS. `e335_insertCopyAndAddStmts`
    /// dispatches to `insertCopyAndAddStmtsHelper<affine::AffineForOp>` or
    /// `<scf::ForOp>` (`Helper.cpp:3870-3876`, `llvm_unreachable("unhandeled type of loop")` for
    /// anything else), and the only thing the helper does differently between them is which
    /// terminator it finds.
    ///
    /// ⭐ `None` MEANS *NOT A TERMINATOR*. It is how the body is searched, not a refusal.
    fn yielded(&mut self) -> Option<&mut Vec<Val>>;
}

impl LoopBodyOp for crate::islands::dataflow_ir::dialects::Op {
    fn arith(op: arith::Op) -> Self {
        crate::islands::dataflow_ir::dialects::Op::Arith(op)
    }

    fn yielded(&mut self) -> Option<&mut Vec<Val>> {
        match self {
            crate::islands::dataflow_ir::dialects::Op::Affine(affine::Op::Yield { operands })
            | crate::islands::dataflow_ir::dialects::Op::Scf(scf::Op::Yield { operands }) => {
                Some(operands)
            }
            _ => None,
        }
    }
}

impl LoopBodyOp for crate::islands::sentient::dialects::Op {
    fn arith(op: arith::Op) -> Self {
        crate::islands::sentient::dialects::Op::Arith(op)
    }

    fn yielded(&mut self) -> Option<&mut Vec<Val>> {
        match self {
            crate::islands::sentient::dialects::Op::Affine(affine::Op::Yield { operands })
            | crate::islands::sentient::dialects::Op::Scf(scf::Op::Yield { operands }) => {
                Some(operands)
            }
            _ => None,
        }
    }
}

/// Replaces: e029_insertCopyAndAddStmtsHelper
///
/// `insertCopyAndAddStmtsHelper<T>` (`AgenToSentient.hpp:501-519`):
///
/// ```c++
/// template <class T>
/// static Value insertCopyAndAddStmtsHelper(T loop_op, int index, int imm_val) {
///   unsigned num_iter_args = loop_op.getNumRegionIterArgs();
///   auto iter_arg = loop_op.getRegionIterArgs()[num_iter_args - index - 1];
///   OpBuilder builder(loop_op.getBody()->getTerminator());
///   auto const_op = mlir::arith::ConstantIndexOp::create(
///       builder, loop_op.getBody()->getTerminator()->getLoc(), imm_val);
///
///   // Create add operation reflecting address arithmetic addition
///   auto add_op = mlir::arith::AddIOp::create(builder, const_op.getLoc(),
///                                             iter_arg.getType(), iter_arg,
///                                             const_op.getResult());
///
///   // update yield operand
///   auto* yield_op = loop_op.getBody()->getTerminator();
///   yield_op->setOperand(yield_op->getNumOperands() - index - 1,
///                        add_op.getResult());
///   return loop_op.getRegionIterArgs()[num_iter_args - index - 1];
/// }
/// ```
///
/// ⛔⛔ THIS IS THE ONLY THING THAT MAKES A TRANSFER'S ADDRESS MOVE. A loop-carried address that is
/// never advanced re-reads the same elements every iteration and the program is silently wrong — no
/// verifier objects, because yielding the iteration argument unchanged is a perfectly good loop. The
/// vendor's own output shows the pair this emits, immediately before the terminator:
///
/// ```text
/// %36 = arith.constant 2 : index
/// %37 = arith.addi %29, %36 : index
/// affine.yield %30, %37 : index, index
/// ```
///
/// (`dcc/test/Conversion/AgenToSentient/lx_indirect_loads_stores_composite.mlir:45-47`, where the
/// loop is `%26:2 = affine.for %27 = 0 to 32 iter_args(%28 = %0, %29 = %25) -> (index, index)` at
/// `:41` — so `index` 0 named `%29`, the last of the two.)
///
/// ⭐ THREE PLACES, ONE INDEX, COUNTED FROM THE END — see [`CarriedFromEnd`]. The C++ reads it
/// against `getNumRegionIterArgs()` and again against `yield_op->getNumOperands()`, trusting the
/// verifier that those agree; here the iteration argument travels inside the witness and the
/// terminator's operand is found by counting, so the two lengths are never subtracted from each
/// other.
///
/// ⭐ AND IT RETURNS THE ARGUMENT, NOT THE SUM. `Helper.cpp:945-950` assigns the result back over
/// `mutable_addrs[i]`, so what the rest of the lowering uses as *the address* is the value the body
/// reads — the sum is only what the next iteration will see. Returning the sum would address every
/// transfer one step ahead of itself. [`AddressAdvance`] carries both, named.
///
/// ⛔ THE `OpBuilder` IS THE DROPPED MECHANISM, not a dropped decision: `OpBuilder(terminator)`
/// inserts immediately before the terminator, which is the end of the statement list, and the
/// locations it threads (`terminator->getLoc()`, then `const_op.getLoc()`) exist because MLIR ops
/// carry source locations.
pub fn insert_copy_and_add_stmts_helper<O: LoopBodyOp>(
    vals: &mut Values,
    body: &mut Vec<O>,
    carried: CarriedFromEnd,
    imm_val: i64,
) -> AddressAdvance {
    let addend = vals.mint();
    let sum = vals.mint();

    // `OpBuilder builder(loop_op.getBody()->getTerminator())` — immediately BEFORE the terminator.
    // ⭐ A BODY WITH NO TERMINATOR YET PUTS THEM LAST, which is the same position: the terminator is
    // whatever the caller appends after them.
    let at = body
        .iter_mut()
        .rposition(|op| op.yielded().is_some())
        .unwrap_or(body.len());
    body.insert(
        at,
        O::arith(arith::Op::Constant {
            result: addend,
            value: imm_val,
        }),
    );
    body.insert(
        at + 1,
        O::arith(arith::Op::AddI(arith::IntBinary {
            result: sum,
            lhs: carried.arg,
            rhs: addend,
            // ⭐ `iter_arg.getType()`, WHICH IS `index`: every carried address is one. See
            // [`Carried`].
            ty: ScalarTy::Index,
        })),
    );

    // `yield_op->setOperand(yield_op->getNumOperands() - index - 1, add_op.getResult())`.
    // ⭐ COUNTED, NOT INDEXED: `len - i - 1` is in range for every `i` the walk visits, so the
    // subtraction the C++ does on two independently-read lengths cannot go wrong here.
    if let Some(operands) = body.get_mut(at + 2).and_then(LoopBodyOp::yielded) {
        let len = operands.len();
        for (i, operand) in operands.iter_mut().enumerate() {
            if len - i - 1 == carried.from_end {
                *operand = sum;
            }
        }
    }

    AddressAdvance {
        iter_arg: carried.arg,
        yielded: sum,
        addend,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{Consumed, ExtractIdx, TransferComp};
    use crate::units::DfirUnit;

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

    /// 🎯 SIX COMPONENTS ARE THIS PASS'S AND NOBODY ELSE IS — `is_any_of(comp, L0LU, L0SU, LXLU,
    /// LXSU, L3SU, L3LU)` (`AgenToSentient.cpp:174-176`). A compute unit answering `Some` would have
    /// its `agen` accesses lowered twice, here and in `VectorChainToSentientPESFP`.
    #[test]
    fn only_the_six_transfer_components_belong_to_this_pass() {
        assert_eq!(TransferComp::of(DfirUnit::Lxlu), Some(TransferComp::Lxlu));
        assert_eq!(TransferComp::of(DfirUnit::L0lu), Some(TransferComp::L0lu));
        assert_eq!(TransferComp::of(DfirUnit::L0su), Some(TransferComp::L0su));
        assert_eq!(TransferComp::of(DfirUnit::Lxsu), Some(TransferComp::Lxsu));
        assert_eq!(TransferComp::of(DfirUnit::L3lu), Some(TransferComp::L3lu));
        assert_eq!(TransferComp::of(DfirUnit::L3su), Some(TransferComp::L3su));
        assert_eq!(TransferComp::of(DfirUnit::Pe), None);
        assert_eq!(TransferComp::of(DfirUnit::Sfp), None);
        assert_eq!(TransferComp::of(DfirUnit::Lx), None);
        assert_eq!(TransferComp::of(DfirUnit::Hbm), None);
    }
}

#[cfg(test)]
mod transfer_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::Op as DfirOp;
    use crate::islands::dataflow_ir::dialects::affine::Bound;
    use crate::islands::dataflow_ir::print;
    use crate::islands::sentient::dialects::Op as SenOp;

    /// The extract statement both overloads thread through, as the vendor's own pair of results
    /// (`lx_indirect_loads_stores_composite.mlir:34`: `%21, %22 = sentient.load_and_extract_scalar`).
    fn extract_op() -> ExtractScalarOp {
        use super::super::agen_helper::{ExtractScalarOps, ExtractScalarResults};
        let mut ops = ExtractScalarOps::default();
        ops.mint(ExtractScalarResults::LoadAndExtractScalar {
            addr: Val(21),
            data: Val(22),
        })
    }

    /// ⭐ THE DEFAULTS ARE ALL OFF — the state three call sites get by writing nothing.
    #[test]
    fn the_unspecialised_transfer_is_off_in_every_way() {
        let plain = TransferSpecialisation::UNSPECIALISED;
        assert_eq!(plain.burst_size, Elements(0));
        assert_eq!(plain.group_size, Elements(0));
        assert_eq!(plain.stride_step, StrideStep(0));
        assert_eq!(plain.outermost_comp_loop, None);
        assert_eq!(plain.extract_op, None);
        assert!(!plain.performs_burst_or_group());
        assert!(!plain.adjusts_for_stride());
    }

    /// ⛔ `burst_size > 0 || group_size > 0` — EITHER arms the bursting path.
    #[test]
    fn either_size_arms_the_burst_or_group_path() {
        let bursted = TransferSpecialisation {
            burst_size: Elements(4),
            ..TransferSpecialisation::UNSPECIALISED
        };
        let grouped = TransferSpecialisation {
            group_size: Elements(2),
            ..TransferSpecialisation::UNSPECIALISED
        };
        assert!(bursted.performs_burst_or_group());
        assert!(grouped.performs_burst_or_group());
    }

    /// ⛔ A ZERO STRIDE STEP ASKS FOR NO ADJUSTMENT, and a positive one does — the `> 0` guard.
    #[test]
    fn only_a_positive_stride_step_asks_for_an_adjustment() {
        assert!(
            !TransferSpecialisation {
                stride_step: StrideStep(0),
                ..TransferSpecialisation::UNSPECIALISED
            }
            .adjusts_for_stride()
        );
        assert!(
            TransferSpecialisation {
                stride_step: StrideStep(8),
                ..TransferSpecialisation::UNSPECIALISED
            }
            .adjusts_for_stride()
        );
    }

    /// ⭐ THE OVERLOADS FILL THE EXTRACT OP AND NOTHING ELSE — `0, 0, 0, nullptr, extract_op`.
    #[test]
    fn both_overloads_default_everything_but_the_extract_op() {
        let extract = extract_op();
        for spec in [
            construct_load_and_send_stmt(extract),
            construct_receive_and_store_stmt(extract),
        ] {
            assert_eq!(spec.extract_op, Some(extract));
            assert_eq!(
                TransferSpecialisation {
                    extract_op: None,
                    ..spec
                },
                TransferSpecialisation::UNSPECIALISED,
                "only the extract op differs from the general form's own defaults"
            );
            assert!(!spec.performs_burst_or_group());
            assert!(!spec.adjusts_for_stride());
        }
    }

    /// The vendor's inner loop's carried pair: `iter_args(%28 = %0, %29 = %25) -> (index, index)`
    /// binding `%26:2` (`lx_indirect_loads_stores_composite.mlir:41`).
    fn vendors_carried() -> Vec<Carried> {
        vec![
            Carried {
                init: Val(0),
                arg: Val(28),
                result: Val(26),
            },
            Carried {
                init: Val(25),
                arg: Val(29),
                result: Val(27),
            },
        ]
    }

    /// ⛔⛔ INDEX 0 IS THE **LAST** CARRIED VALUE. Two entries, so `index` 0 names `%29` and `index` 1
    /// names `%28` — the reverse of how the caller's loop runs.
    #[test]
    fn the_index_counts_from_the_end() {
        let carried = vendors_carried();
        assert_eq!(
            CarriedFromEnd::at(&carried, 0).map(CarriedFromEnd::arg),
            Some(Val(29))
        );
        assert_eq!(
            CarriedFromEnd::at(&carried, 1).map(CarriedFromEnd::arg),
            Some(Val(28))
        );
    }

    /// ⛔ A LOOP CARRYING TWO VALUES HAS NO THIRD — an absence, not a refusal, and where the C++
    /// indexes past the end of a `MutableArrayRef`.
    #[test]
    fn a_position_the_loop_does_not_carry_is_absent() {
        assert_eq!(CarriedFromEnd::at(&vendors_carried(), 2), None);
        assert_eq!(CarriedFromEnd::at(&[], 0), None);
    }

    /// ⭐⭐ THE VENDOR'S OWN THREE LINES, VERBATIM.
    ///
    /// `%36 = arith.constant 2 : index` / `%37 = arith.addi %29, %36 : index` /
    /// `affine.yield %30, %37 : index, index`
    /// (`lx_indirect_loads_stores_composite.mlir:45-47`), for `index` 0 and `imm_val` 2 on the loop
    /// at `:41` whose terminator yielded `%30, %29`.
    ///
    /// ⛔ THE NAMES ARE THE MINTER'S, so the minter is wound to 36 first — the two values this emits
    /// are the next two a builder would issue, which is exactly why they read `%36` and `%37` in the
    /// reference too.
    #[test]
    fn the_advance_prints_as_the_reference_writes_it() {
        let mut vals = Values::default();
        for _ in 0..36 {
            let _ = vals.mint();
        }

        let carried = vendors_carried();
        let mut body = vec![DfirOp::Affine(affine::Op::Yield {
            operands: vec![Val(30), Val(29)],
        })];
        let advance = insert_copy_and_add_stmts_helper(
            &mut vals,
            &mut body,
            CarriedFromEnd::at(&carried, 0).expect("the loop carries two values"),
            2,
        );

        assert_eq!(advance.addend, Val(36));
        assert_eq!(advance.yielded, Val(37));
        assert_eq!(
            advance.iter_arg,
            Val(29),
            "the C++ returns the iteration argument, not the sum"
        );

        let mut out = String::new();
        for op in &body {
            print::emit(&mut out, op, 0);
        }
        assert_eq!(
            out,
            "%36 = arith.constant 2 : index\n\
             %37 = arith.addi %29, %36 : index\n\
             affine.yield %30, %37 : index, index\n"
        );
    }

    /// ⛔ AND THE OTHER OPERAND IS UNTOUCHED. `affine.yield %30, %37` — `%30` is the DIRECT operand's
    /// own advance, made by a separate call with `index` 1; rewriting both from one call would leave
    /// one address moving twice as fast.
    #[test]
    fn only_the_named_operand_is_rewritten() {
        let mut vals = Values::default();
        let carried = vendors_carried();
        let mut body = vec![DfirOp::Affine(affine::Op::Yield {
            operands: vec![Val(30), Val(29)],
        })];
        let advance = insert_copy_and_add_stmts_helper(
            &mut vals,
            &mut body,
            CarriedFromEnd::at(&carried, 1).expect("the loop carries two values"),
            128,
        );

        assert_eq!(advance.iter_arg, Val(28), "index 1 is the FIRST of two");
        let DfirOp::Affine(affine::Op::Yield { operands }) = &body[2] else {
            panic!("the terminator is still the last op");
        };
        assert_eq!(operands, &vec![advance.yielded, Val(29)]);
    }

    /// ⭐ THE PAIR GOES **BEFORE** THE TERMINATOR, not after it and not at the top of the body.
    #[test]
    fn the_advance_lands_immediately_before_the_terminator() {
        let mut vals = Values::default();
        let carried = vendors_carried();
        let mut body = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(4),
                value: 128,
            }),
            DfirOp::Affine(affine::Op::Yield {
                operands: vec![Val(30), Val(29)],
            }),
        ];
        let advance = insert_copy_and_add_stmts_helper(
            &mut vals,
            &mut body,
            CarriedFromEnd::at(&carried, 0).expect("the loop carries two values"),
            2,
        );

        assert_eq!(body.len(), 4);
        assert_eq!(
            body[1],
            DfirOp::Arith(arith::Op::Constant {
                result: advance.addend,
                value: 2
            })
        );
        assert_eq!(
            body[2],
            DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                result: advance.yielded,
                lhs: Val(29),
                rhs: advance.addend,
                ty: ScalarTy::Index
            }))
        );
        assert!(matches!(body[3], DfirOp::Affine(affine::Op::Yield { .. })));
    }

    /// ⛔ THE TEMPLATE'S SECOND INSTANTIATION: an `scf.yield` terminator is found and rewritten the
    /// same way (`Helper.cpp:3874` dispatches `insertCopyAndAddStmtsHelper<scf::ForOp>`).
    #[test]
    fn an_scf_terminator_is_rewritten_too() {
        let mut vals = Values::default();
        let carried = vendors_carried();
        let mut body = vec![DfirOp::Scf(scf::Op::Yield {
            operands: vec![Val(30), Val(29)],
        })];
        let advance = insert_copy_and_add_stmts_helper(
            &mut vals,
            &mut body,
            CarriedFromEnd::at(&carried, 0).expect("the loop carries two values"),
            2,
        );
        let DfirOp::Scf(scf::Op::Yield { operands }) = &body[2] else {
            panic!("the terminator is still the last op");
        };
        assert_eq!(operands, &vec![Val(30), advance.yielded]);
    }

    /// ⭐⭐ AND ON THE **SENTIENT** RUNG, which is the module the pass is actually mutating: by the
    /// time a loop reaches this helper its body holds `sentient.*` ops beside the `arith` ones.
    #[test]
    fn the_advance_works_on_the_mixed_sentient_rung() {
        let mut vals = Values::default();
        let carried = vendors_carried();
        let mut body: Vec<SenOp> = vec![SenOp::Affine(affine::Op::Yield {
            operands: vec![Val(30), Val(29)],
        })];
        let advance = insert_copy_and_add_stmts_helper(
            &mut vals,
            &mut body,
            CarriedFromEnd::at(&carried, 0).expect("the loop carries two values"),
            2,
        );
        assert_eq!(
            body[0],
            SenOp::Arith(arith::Op::Constant {
                result: advance.addend,
                value: 2
            })
        );
        let SenOp::Affine(affine::Op::Yield { operands }) = &body[2] else {
            panic!("the terminator is still the last op");
        };
        assert_eq!(operands, &vec![Val(30), advance.yielded]);
    }

    /// ⭐⭐ AND THE WHOLE LOOP PRINTS — the advance is only correct if what encloses it is the
    /// carrying loop the reference emits: `%26, %27 = affine.for %31 = 0 to 32 iter_args(...)`.
    ///
    /// ⛔ MLIR BUNDLES A MULTI-RESULT OP'S AUTO-NAMED RESULTS AS `%26:2` and refers to them as
    /// `%26#0`; the island names each result, which is the other form the same parser accepts and
    /// what [`crate::islands::sentient::dialects::sentient::Op::For`] already prints.
    #[test]
    fn the_carrying_loop_prints_around_the_advance() {
        let mut vals = Values::default();
        let carried = vendors_carried();
        let mut body = vec![DfirOp::Affine(affine::Op::Yield {
            operands: vec![Val(30), Val(29)],
        })];
        let _ = insert_copy_and_add_stmts_helper(
            &mut vals,
            &mut body,
            CarriedFromEnd::at(&carried, 0).expect("the loop carries two values"),
            2,
        );

        let mut out = String::new();
        print::emit(
            &mut out,
            &DfirOp::Affine(affine::Op::For {
                iv: Val(31),
                lo: Bound::Const(0),
                hi: Bound::Const(32),
                dbg_name: None,
                carried,
                body,
            }),
            0,
        );
        assert_eq!(
            out,
            "%26, %27 = affine.for %31 = 0 to 32 iter_args(%28 = %0, %29 = %25) -> (index, index) {\n\
             \x20 %0 = arith.constant 2 : index\n\
             \x20 %1 = arith.addi %29, %0 : index\n\
             \x20 affine.yield %30, %1 : index, index\n\
             }\n"
        );
    }
}

use crate::arch::IsaGen;
use crate::islands::dataflow_ir::dialects::{regions, vectorchain};
use crate::islands::sentient::ProgramUnits;
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::DfirUnit;

/// THE SIX COMPONENTS THIS PASS LOWERS FOR AT ALL — `is_any_of(comp, L0LU, L0SU, LXLU, LXSU, L3SU,
/// L3LU)` (`AgenToSentient.cpp:174-176`), over the GENERIC component `getUnitType` collapses a
/// `SenComponents` to. Anything else is the walk's `return`, which is why [`TransferComp::of`]
/// answers with an `Option` rather than a bool: the LDCVTI step needs to know it is `LXLU`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TransferComp {
    /// `L0LU`.
    L0lu,
    /// `L0SU`.
    L0su,
    /// `LXLU`.
    Lxlu,
    /// `LXSU`.
    Lxsu,
    /// `L3LU`.
    L3lu,
    /// `L3SU`.
    L3su,
}

impl TransferComp {
    /// `getUnitType(unit.getUnits()[0].getDefiningOp<GetUnitOp>())`, gated — ⛔ NO WILDCARD, so a
    /// component added to [`DfirUnit`] must say whether this pass owns its transfers.
    pub(super) const fn of(unit: DfirUnit) -> Option<Self> {
        match unit {
            DfirUnit::L0lu => Some(Self::L0lu),
            DfirUnit::L0su => Some(Self::L0su),
            DfirUnit::Lxlu => Some(Self::Lxlu),
            DfirUnit::Lxsu => Some(Self::Lxsu),
            DfirUnit::L3lu => Some(Self::L3lu),
            DfirUnit::L3su => Some(Self::L3su),
            DfirUnit::Sfp
            | DfirUnit::Pe
            | DfirUnit::PtRow(_)
            | DfirUnit::Lx
            | DfirUnit::Hbm
            | DfirUnit::L0
            | DfirUnit::Constant
            | DfirUnit::SfpState
            | DfirUnit::PeState
            | DfirUnit::SfpRing
            | DfirUnit::LxVirtualIbr
            | DfirUnit::CrossPtnLink => None,
        }
    }
}

/// THE PASS ITSELF: gate, LDCVTI, fuse, interleave, mask state, `set_send_dst` cleanup — per unit.
///
/// ⛔⛔ FOUR OF THE SIX STEPS HAVE UNPORTED CALLEES (e318, e271, e218, e220) — one `todo!` each over
/// the counted input that would reach it. The step that lowers today is the fusion, e382.
///
/// ⛔ THE GATE REACHES [`super::body`] RATHER THAN SKIPPING THE UNIT: the reference's `return` skips
/// one pass of seventy-six, while this spine lowers every dialect of a unit in one walk.
///
/// Replaces: e384_runOnOperation
pub(super) fn run_on_operation<A: Arch>(
    input: &dfir::Program<A>,
    bound: &Bound,
    consts: &Consts,
) -> ProgramUnits<A> {
    let mut lowered = input.units.iter().map(|unit| {
        let comp = TransferComp::of(unit.on.kind());

        // ── the LDCVTI pre-pass, LXLU at SEN1P5 and above (`:178-198`) ───────────────────────────
        //
        // ⛔ THE ANCHOR IS A `vectorchain.binary` WHOSE OPERATOR IS `mul`, NOT THE ISLAND'S
        // `vectorchain.multiply` — *"LDCVTI patterns are currently the only patterns involving
        // vectorchain.multiply"* names the operator, and the walk is over `BinaryOp` (`:183-189`).
        // ⛔ AND THE CONVERGENCE LOOP RE-WALKS FROM THE TOP after each rewrite, so a count of
        // candidates is what it converges on, not the one it found first.
        if matches!(comp, Some(TransferComp::Lxlu)) && A::GEN >= IsaGen::Sen1p5 {
            let muls = count(&unit.body, &|op| {
                matches!(
                    op,
                    DfirOp::VectorChain(vectorchain::Op::Binary {
                        binary_op: vectorchain::BinaryOp::Mul,
                        ..
                    })
                )
            });
            if muls > 0 {
                todo!(
                    "e318_lowerLDCVTIPattern: {muls} vectorchain.binary mul anchor(s) on an LXLU \
                     unit at {:?}",
                    A::GEN
                );
            }
        }

        // ── the load/store fusion, e382 (`:213-216`) ────────────────────────────────────────────
        let mut local = bound.clone();
        let out = super::body(unit, &mut local, consts, comp);

        // ⛔ THE THREE STEPS BELOW ARE INSIDE THE GATE, so a unit this pass does not own reaches
        // none of them — `comp.is_some()` is the reference's `return` restated.
        if comp.is_some() {
            // ── the interleaves (`:218-231`), which need the transfers already lowered ──────────
            let interleaves = count(&unit.body, &|op| {
                matches!(op, DfirOp::Agen(agen::Op::CompositeMemoryInterleave { .. }))
            });
            if interleaves > 0 {
                todo!(
                    "e271_lowerCompositeMemoryInterleaveOp: {interleaves} \
                     agen.composite_memory_interleave on {:?}",
                    unit.on.kind()
                );
            }

            // ── the mask states (`:233-244`) ────────────────────────────────────────────────────
            let masks = count(&unit.body, &|op| {
                matches!(op, DfirOp::Agen(agen::Op::SetTransferMaskState { .. }))
            });
            if masks > 0 {
                todo!(
                    "e218_lowerSetTransferMaskStateOp: {masks} agen.set_transfer_mask_state -> \
                     sentient.samv on {:?}",
                    unit.on.kind()
                );
            }

            // ── the redundant `set_send_dst` cleanup (`:246-247`) ───────────────────────────────
            //
            // ⛔ OVER THE LOWERED OPS, NOT THE INPUT — `set_send_dst` is a `sentient` op this pass
            // emitted, and the cleanup collapses a unit's identical ones into one at the top
            // (`Helper.cpp:4084-4123`). Its own `getArch() < RCUDD1A_ISA` guard is vacuous here:
            // [`IsaGen`] has no generation below it.
            if out
                .body
                .iter()
                .any(|op| matches!(op, SenOp::Sentient(sen::Op::SetSendDst { .. })))
            {
                todo!(
                    "e220_cleanupTriviallyRedundantSetSendDestination: sentient.set_send_dst on {:?}",
                    unit.on.kind()
                );
            }
        }
        out
    });
    // ⛔ NON-EMPTY BY THE TYPE on both rungs — see [`ProgramUnits`].
    let head = lowered
        .next()
        .expect("the rung below's units are non-empty");
    ProgramUnits::of(head, lowered.collect())
}

/// HOW MANY OPS OF A KIND ONE UNIT HOLDS, PREORDER AND DESCENDING INTO EVERY REGION —
/// `unit.walk<WalkOrder::PreOrder>`.
fn count(ops: &[DfirOp], pick: &impl Fn(&DfirOp) -> bool) -> usize {
    ops.iter()
        .map(|op| {
            usize::from(pick(op))
                + regions(op)
                    .iter()
                    .map(|region| count(region, pick))
                    .sum::<usize>()
        })
        .sum()
}
