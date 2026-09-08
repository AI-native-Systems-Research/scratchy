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

        // ── 3. `agen.composite_load` (`AgenToSentient.cpp:90-95`) ────────────────────────────────
        agen::Op::CompositeLoad(_) => todo!(
            "e329_lowerCompositeLoadOp: agen.composite_load on {:?}",
            unit.on.kind()
        ),

        // ── 4. `agen.composite_store` (`AgenToSentient.cpp:96-101`) ──────────────────────────────
        agen::Op::CompositeStore(_) => todo!(
            "e330_lowerCompositeStoreOp: agen.composite_store on {:?}",
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
        agen::Op::Yield { .. } => todo!(
            "agen.yield reached a unit's statement list on {:?} — it terminates a composite \
             transfer's region",
            unit.on.kind()
        ),

        // ── 6. `agen.set_transfer_mask_state` — the lowering is not ported ───────────────────────
        agen::Op::SetTransferMaskState { .. } => todo!(
            "agen.set_transfer_mask_state not ported: SAMV on {:?}",
            unit.on.kind()
        ),
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
/// would have thrown away. Its one producer computes
/// `group_index >= 0 ? -(time_bounds[group_index] .. ) : 0` at `Helper.cpp:1862`.
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
/// ⛔ THE EXTRA PARAMETER IS `data_elem_type`, AND IT IS FORWARDED UNTOUCHED. The store side needs
/// the element type because `setsttype` derives `element_width` and the shuffle mode from it
/// (`AgenToSentient.hpp:206-209`, used at `Helper.cpp:2124`); the load side reads its width off the
/// consumer instead. Its caller takes it from the op's own memref —
/// `candidate_op.getDirectMemrefType().getElementType()` (`:3250`) — so it is the caller's value,
/// like the other five pass-throughs, and this overload neither inspects nor changes it.
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
        use super::super::agen_helper::{ExtractScalarKind, ExtractScalarOps};
        let mut ops = ExtractScalarOps::default();
        ops.mint(ExtractScalarKind::LoadAndExtractScalar, Val(21), Val(22))
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
