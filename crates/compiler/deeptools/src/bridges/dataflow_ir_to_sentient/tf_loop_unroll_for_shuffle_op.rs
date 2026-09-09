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

//! `LoopUnrollForShuffleOp.cpp` — 4 of bridge 2's 384 functions (dependency level(s) [0, 1, 2]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e109_performFullUnroll` | 109/384 | 7 | `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141` |
//! | `e110_getConstantTripCount` | 110/384 | 14 | `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167` |
//! | `e183_expandAffineApplyOps` | 183/384 | 53 | `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183` |
//! | `e248_runOnOperation` | 248/384 | 68 | `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70` |

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, arith, defining_op, region_owner, regions, regions_mut,
    replace_uses_of_with, scf, uniform, vectorchain as vc,
};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ScalarTy};
use crate::islands::dataflow_ir::{self as dfir, Values};
use crate::units::DfirUnit;

/// A LOOP THIS PASS MAY UNROLL — the closed set `performFullUnroll` dispatches over.
///
/// `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141-149` is two `dyn_cast`s and an
/// `llvm_unreachable`:
///
/// ```text
/// if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop_op))         return performFullUnroll(scf_for);
/// else if (auto affine_for = llvm::dyn_cast<affine::AffineForOp>(loop_op))
///                                                                return performFullUnroll(affine_for);
/// else llvm_unreachable("unsupported loop type");
/// ```
///
/// ⛔⛔ THE `llvm_unreachable` IS THE TYPE, NOT A BRANCH. Taking an `Operation *` means the third arm
/// has to exist and be undefined behaviour; taking this enum means there is no third arm to write.
/// The classification that could fail happens once, in [`Loop::of`], where "not a loop" is a
/// well-formed **answer** rather than a stop.
///
/// ⚠️ THE REFERENCE DOES NOT AGREE THAT IT IS AN ANSWER — it asks the same question earlier and
/// aborts on it too. `runOnOperation`'s walk classifies the block argument's parent op with the same
/// two `dyn_cast`s and closes with `else llvm_unreachable("unsupported loop type")` (`:96-108`), so
/// the undefined behaviour is stated twice rather than guarded once. What makes it unreachable is
/// upstream of both: the parent region of a `vectorchain.shuffle` operand's block argument IS a loop.
///
/// ⭐ THE KIND, PLUS ONLY WHAT THE REFERENCE ASKS OF IT. `performFullUnroll(scf::ForOp)` reads the
/// loop's three bounds and nothing else; `performFullUnroll(affine::AffineForOp)` reads NOTHING at
/// all — it is one line, `return loopUnrollFull(for_op)` (`:162-165`), because affine's own analysis
/// derives the trip count from the loop's maps. Carrying whole ops here would offer callers a dozen
/// questions this rule does not ask, which is the shape
/// [`Parent`](super::std_affine_to_standard::Parent) already established in this bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loop {
    /// `scf::ForOp` — its lower bound, upper bound and step, which is all the overload reads.
    Scf {
        /// `for_op.getLowerBound()`.
        lo: Val,
        /// `for_op.getUpperBound()`.
        hi: Val,
        /// `for_op.getStep()`.
        step: Val,
    },
    /// `affine::AffineForOp` — no fields, because the overload asks it nothing.
    Affine,
}

impl Loop {
    /// WHICH LOOP AN OP IS, OR NONE FOR AN OP THAT IS NOT A LOOP — the two `dyn_cast`s of `:143-146`.
    ///
    /// ⛔ `None` IS NOT A REFUSAL. It is the answer the reference gets from a failed `dyn_cast`, and
    /// the place its `llvm_unreachable` has been moved to: a caller that reached an op through
    /// [`block_args`](crate::islands::dataflow_ir::dialects::block_args) of a for-loop cannot observe
    /// it, so the undefined behaviour becomes unrepresentable instead of merely unlikely.
    #[must_use]
    pub fn of(op: &DfirOp) -> Option<Self> {
        match op {
            DfirOp::Scf(scf::Op::For { lo, hi, step, .. }) => Some(Loop::Scf {
                lo: *lo,
                hi: *hi,
                step: *step,
            }),
            DfirOp::Affine(affine::Op::For { .. }) => Some(Loop::Affine),
            DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Arith(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ AND `uniform`: `uniform.uniformize_regions` is region-carrying but neither
            // `scf::ForOp` nor `affine::AffineForOp`, so both `dyn_cast`s are null.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
        }
    }
}

/// HOW MANY TIMES A LOOP RUNS — strictly positive, which is what makes it an unroll factor.
///
/// ⛔⛔ STRICTLY POSITIVE IS THE WHOLE POINT OF THE NEWTYPE, AND IT IS A DELIBERATE DIVERGENCE.
/// `getConstantTripCount` returns `(ub - lb) / step` as a bare `int64_t` (`:181`) and
/// `performFullUnroll` hands it straight to `loopUnrollByFactor(for_op, *trip_count)`, whose factor
/// parameter is a `uint64_t` guarded by `assert(unrollFactor > 0)`. An `scf.for` whose bounds are
/// equal — `lb = ub = 0`, which the folder produces routinely — yields a trip count of **zero**, and
/// zero reaches that utility as an assertion in a debug build and a `tripCount % 0` in a release one.
/// A reversed range yields a negative count, which converts to an enormous unsigned factor. Neither
/// is a defect worth reproducing: both fold into the decline channel the reference already has for a
/// trip count it cannot use, and [`Unroll::EmptyOrReversedRange`] keeps the divergence visible in the
/// type rather than hidden inside a fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TripCount(i64);

impl TripCount {
    /// THE ONLY CONSTRUCTOR — a non-positive count is not a trip count.
    ///
    /// ⭐ SHARED WITH ENTRY 195, which asks `loopUnrollByFactor` for the same quantity from a loop's
    /// own three constants — see
    /// [`LoopRewrite`](super::tf_transform_loop_to_legalize_for_sentient_lowering::LoopRewrite).
    pub(crate) fn positive(trips: i64) -> Option<Self> {
        (trips > 0).then_some(Self(trips))
    }

    /// THE FACTOR TO UNROLL BY, as `int64_t` — the reference's own `*trip_count`.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// WHAT `performFullUnroll` DECIDED — its `LogicalResult`, and the request behind a success.
///
/// ⛔ NOT A `Result`. `LogicalResult` is a two-state domain answer that this pipeline's callers ask
/// `.failed()` of (`:121`), not an error to propagate — and this crate freezes `Result` in the
/// bridge at zero (`crates/targets/spyre/tests/dfir_never_runtime_refuses.rs`).
///
/// ⭐⭐ THE TWO SUCCESS ARMS CARRY THE **REQUEST**, WHICH IS WHAT THIS FUNCTION EMITS. Neither
/// overload duplicates a loop body itself: each computes the unroll it wants and returns whatever
/// upstream MLIR's `loopUnrollByFactor` / `loopUnrollFull` returns. Those two utilities are
/// `mlir/Dialect/{SCF,Affine}/Utils` — upstream, not among bridge 2's 384 units — so the answer they
/// give is not this function's to state, and the arms name the call rather than pretending to have
/// made it. What IS this function's, in full, is the dispatch, the trip-count reconstruction and both
/// of its refusals.
///
/// ⛔ SO [`Unroll::failed`] IS EXACTLY "DID `performFullUnroll` ITSELF DECLINE", which for the affine
/// arm is never — precisely as the reference's affine overload has no decline of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Unroll {
    /// `return loopUnrollByFactor(for_op, *trip_count);` (`:159`) — the `scf.for` arm.
    ///
    /// ⭐ BY THE TRIP COUNT, BECAUSE SCF HAS NO FULL UNROLL. The reference says so in a comment at
    /// `:157-158`: *"SCF lacks a dedicated full unroll function so using loopUnrollByFactor with the
    /// trip count"* — unrolling by exactly the trip count IS the full unroll, and it is why the
    /// helper below exists at all.
    ByFactor(TripCount),
    /// `return loopUnrollFull(for_op);` (`:164`) — the `affine.for` arm, which asks nothing first.
    Fully,
    /// `for_op->emitError("Non-constant trip bound for unrolling"); return failure();` (`:153-156`).
    ///
    /// ⭐ THE MESSAGE IS THE REFERENCE'S. The caller turns any failure into
    /// `emitError("Cannot unroll candidate")` and `signalPassFailure()` (`:121-124`), so this variant
    /// is a stop for the compilation, not for the port.
    NonConstantTripBound,
    /// A CONSTANT TRIP COUNT THAT IS NOT AN UNROLL FACTOR — see [`TripCount`] for the divergence.
    EmptyOrReversedRange,
    /// `if (step <= 0) return std::nullopt;` (`:179`) — a non-advancing or backward step.
    ///
    /// ⭐ ITS OWN VARIANT THOUGH THE REFERENCE FOLDS IT INTO `nullopt`, because it is a different
    /// fact about the loop from "the bound is not a constant" and both reach the same `failure()`.
    NonPositiveStep,
}

impl Unroll {
    /// `LogicalResult::failed()` — the one question `runOnOperation` asks of this result (`:121`).
    #[must_use]
    pub const fn failed(self) -> bool {
        match self {
            Unroll::ByFactor(_) | Unroll::Fully => false,
            Unroll::NonConstantTripBound
            | Unroll::EmptyOrReversedRange
            | Unroll::NonPositiveStep => true,
        }
    }
}

/// Replaces: e109_performFullUnroll
///
/// `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141-165` — the dispatcher and BOTH
/// overloads it dispatches to, which are three C++ functions and one Rust function.
///
/// ⭐ THREE COLLAPSE INTO ONE BECAUSE THE OVERLOAD SET **IS** A MATCH ON A CLOSED SET. C++ needs a
/// dispatcher plus one function per loop kind since the kind arrives as an `Operation *`; here the
/// kind is [`Loop`] and each overload's body is the arm that names it. Only the dispatcher is a
/// scheduled unit (entry 109, 7 lines) — `UNITS.tsv` does not list the two overloads separately, and
/// the extract's `e109` body is the dispatcher's `:141-149` alone.
///
/// ⛔ THE UNROLL ITSELF IS UPSTREAM MLIR. See [`Unroll`]: this function decides *which* unroll to
/// request and refuses when it cannot, which is the entirety of `:141-165`.
///
/// # Arguments
///
/// * `loop_op` — the candidate, already classified. `runOnOperation` only ever produces one by
///   finding a `vectorchain.shuffle` whose `variable` operand is a for-loop's induction variable
///   (`:88-112`), so a candidate is always one of the two kinds [`Loop`] holds.
/// * `scope` — the ops the loop's bounds are defined in, for the `getDefiningOp` walk of
///   [`constant_trip_count`]. Unread on the affine arm, which asks the loop nothing.
pub fn perform_full_unroll(loop_op: Loop, scope: &[DfirOp]) -> Unroll {
    match loop_op {
        // `:151-160` — the scf overload.
        Loop::Scf { lo, hi, step } => match constant_trip_count(lo, hi, step, scope) {
            TripBound::Trips(trips) => Unroll::ByFactor(trips),
            TripBound::NoConstant => Unroll::NonConstantTripBound,
            TripBound::NonPositiveStep => Unroll::NonPositiveStep,
            TripBound::EmptyOrReversed => Unroll::EmptyOrReversedRange,
        },
        // `:162-165` — the affine overload, verbatim: `return loopUnrollFull(for_op);`.
        Loop::Affine => Unroll::Fully,
    }
}

/// Replaces: e110_getConstantTripCount
///
/// `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167-182`:
///
/// ```text
/// auto lb_const   = for_op.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
/// auto ub_const   = for_op.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
/// auto step_const = for_op.getStep().getDefiningOp<arith::ConstantIntOp>();
/// if (!lb_const || !ub_const || !step_const) return std::nullopt;
/// auto lb = lb_const.value(); auto ub = ub_const.value(); auto step = step_const.value();
/// if (step <= 0) return std::nullopt;
/// return (ub - lb) / step;
/// ```
///
/// ⛔⛔ `ConstantIntOp` EXCLUDES AN `index` CONSTANT, AND THAT IS A REAL REFUSAL IN THE REFERENCE.
/// `arith::ConstantIntOp::classof` requires the result type to be a **signless integer**, and
/// `index` is not one; `arith.constant 0 : index` is an `arith::ConstantIndexOp`. So an `scf.for`
/// written the ordinary MLIR way — `index`-typed bounds — makes all three `dyn_cast`s null here and
/// the pass reports *"Cannot unroll candidate"*. The island already splits the two:
/// [`arith::Op::ConstantInt`] is the signless-integer constant this reads and
/// [`arith::Op::Constant`] is the `index`/float one it must NOT, so the distinction is a match arm
/// rather than a type query. Consistent with the vendor's only test for this pass taking the affine
/// path throughout (`dcc/test/Transform/LoopUnrolForShuffleOp/ldcvti_pattern.mlir`).
///
/// ⭐ ALL THREE ARE FETCHED BEFORE ANY IS CHECKED, WHICH IS WHY THE TUPLE IS THERE. `:169-173` looks
/// up every bound and only then tests the disjunction; a `?` chain would stop at the first
/// non-constant one. The answer is the same either way — `getDefiningOp` has no side effects — but
/// the tuple keeps the reference's shape readable beside its line numbers.
///
/// ⛔ TRUNCATING DIVISION IS THE REFERENCE'S, AND IT UNDER-COUNTS. `(ub - lb) / step` with
/// `lb = 0, ub = 7, step = 2` is 3, but the loop runs 4 times — so "full" unroll leaves a loop with
/// one iteration behind whenever the step does not divide the span. Reproduced exactly, because the
/// factor is what the reference asks the utility for and a divergence here would change which
/// program comes out; recorded because it is a defect to fix upstream, not one to paper over here.
fn constant_trip_count(lo: Val, hi: Val, step: Val, scope: &[DfirOp]) -> TripBound {
    let (Some(lo), Some(hi), Some(step)) = (
        signless_int_constant(lo, scope),
        signless_int_constant(hi, scope),
        signless_int_constant(step, scope),
    ) else {
        // `:173` — `if (!lb_const || !ub_const || !step_const) return std::nullopt;`
        return TripBound::NoConstant;
    };

    // `:179` — `if (step <= 0) return std::nullopt;`
    if step <= 0 {
        return TripBound::NonPositiveStep;
    }

    // `:181` — `return (ub - lb) / step;`
    //
    // ⭐ THE SUBTRACTION IS CHECKED BECAUSE THE REFERENCE'S IS UNDEFINED. `ub - lb` on `int64_t`
    // overflows for a span wider than `i64::MAX`; an overflowed span is not a trip count, so it
    // joins the range that is not one. The division cannot overflow — `step` is positive here.
    match hi.checked_sub(lo) {
        Some(span) => match TripCount::positive(span / step) {
            Some(trips) => TripBound::Trips(trips),
            None => TripBound::EmptyOrReversed,
        },
        None => TripBound::EmptyOrReversed,
    }
}

/// WHAT [`constant_trip_count`] FOUND — the reference's `std::optional<int64_t>`, split by reason.
///
/// ⛔ ONE `nullopt` IN THE REFERENCE, THREE ANSWERS HERE, because the caller's error message is the
/// same for all of them but the FACT is not: a bound that is not a constant, a step that does not
/// advance, and a range with nothing in it are three different things to read in a log. They all
/// reach `failure()`, so the collapse is lossless in behaviour — see [`Unroll`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TripBound {
    /// `return (ub - lb) / step;` with a usable factor.
    Trips(TripCount),
    /// A bound whose defining op is not an `arith::ConstantIntOp`.
    NoConstant,
    /// `step <= 0`.
    NonPositiveStep,
    /// A constant range that yields no positive factor — see [`TripCount`].
    EmptyOrReversed,
}

/// `%v.getDefiningOp<arith::ConstantIntOp>()`, AND ITS VALUE — `None` for anything else.
///
/// ⛔ `index` IS NOT A SIGNLESS INTEGER, so [`arith::Op::Constant`] is not a candidate however
/// integral its literal looks. See [`constant_trip_count`].
///
/// ⭐ `i1` READS BACK THROUGH ITS BOOL, and it is a legitimate signless-integer constant:
/// `ConstantIntOp::value()` sign-extends `true` to `-1` and `false` to `0`, which is exactly what
/// [`arith::IntConst::Bool`] records the reference doing (`StandardToSentient.cpp:361-366`). A step
/// of `true` is therefore `-1` and lands on the non-positive-step refusal, not on a factor of one.
fn signless_int_constant(val: Val, scope: &[DfirOp]) -> Option<i64> {
    match defining_op(val, scope)? {
        DfirOp::Arith(arith::Op::ConstantInt { value, .. }) => match value {
            arith::IntConst::Int { value, .. } => Some(*value),
            arith::IntConst::Bool(set) => Some(if *set { -1 } else { 0 }),
        },
        DfirOp::Arith(_)
        | DfirOp::Scf(_)
        | DfirOp::Affine(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        // ⭐ AND `uniform`: a `query_map` binds an `index`, which is not the signless integer
        // `ConstantIntOp` names — the same reason [`arith::Op::Constant`] is absent above.
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// 183/384  expandAffineApplyOps  —  dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183
// ══════════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT ONE AFFINE EXPRESSION EXPANDED INTO — MLIR's `AffineApplyExpander`, whose visitors return a
/// `Value` that is **null** on the one error they raise.
///
/// ⭐ THE NULL IS SPLIT BY CAUSE, BECAUSE THE REFERENCE'S OWN DIAGNOSTIC NAMES THE CAUSE.
/// `visitModExpr` writes *"modulo by non-positive value is not supported"* and `visitFloorDivExpr`
/// writes *"division by non-positive value is not supported"*
/// (`mlir/lib/Dialect/Affine/Utils/Utils.cpp:82-84`, `:126-128`); by the time `expandAffineMap` has
/// turned the null into a `std::nullopt` the distinction is gone, and this pass then prints one
/// message for both. Keeping the two apart costs nothing and says which map was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Expansion {
    /// The value the last op built binds — the expander's non-null `Value`.
    Value(Val),
    /// `a mod b` with `b` a non-positive literal.
    NonPositiveModulus(i64),
    /// `a floordiv b` with `b` a non-positive literal.
    NonPositiveDivisor(i64),
    /// A `d<N>` or `s<N>` past the operands the apply supplies.
    ///
    /// ⭐ MLIR'S OWN ASSERT, MADE AN ANSWER. `visitDimExpr` is
    /// `assert(expr.getPosition() < dimValues.size() && "affine dim position out of range")`
    /// (`Utils.cpp:194-196`) — a stop in a debug build and a read past the end in a release one.
    /// What makes it unreachable upstream is `AffineApplyOp::verify`, which checks the operand count
    /// against `getAffineMap().getNumInputs()`; the island states the map's arities
    /// ([`AffineMap::dims`], [`AffineMap::syms`]) and the argument list
    /// ([`affine::Op::Apply::args`]) separately, so a short list is expressible here and gets a
    /// name rather than a panic.
    PositionOutOfRange,
}

/// THE OPS ONE `affine.apply` EXPANDS TO, IN THE ORDER `OpBuilder` CREATED THEM.
///
/// `mlir::affine::expandAffineExpr` (`mlir/lib/Dialect/Affine/Utils/Utils.cpp:216-222`) is one call
/// on an `AffineApplyExpander` (`:41-208`), which holds the builder, the dimension values, the symbol
/// values and a location. This holds the same, with the builder replaced by the value minter and a
/// buffer — because the ops the reference threads into a block are, here, a list the caller splices.
struct AffineApplyExpander<'a> {
    /// Where a fresh result comes from.
    vals: &'a mut Values,
    /// `dimValues` — `operands.take_front(numDims)` (`Utils.cpp:234-235`).
    dim_values: &'a [Val],
    /// `symbolValues` — `operands.drop_front(numDims)` (`:236`).
    symbol_values: &'a [Val],
    /// What the builder built, oldest first.
    built: Vec<DfirOp>,
}

impl AffineApplyExpander<'_> {
    /// `arith::ConstantIndexOp::create(builder, loc, value)`.
    fn constant(&mut self, value: i64) -> Val {
        let result = self.vals.mint();
        self.built
            .push(DfirOp::Arith(arith::Op::Constant { result, value }));
        result
    }

    /// `OpTy::create(builder, loc, lhs, rhs)` for the five integer binaries.
    ///
    /// ⭐ ALWAYS AT `index`. Every value an `affine.apply` reads and writes is an `index`
    /// (`AffineApplyOp`'s operands and result are `Index`), so the type is not a parameter here —
    /// `IntBinary::ty` is [`ScalarTy::Index`] at every site this expander builds.
    ///
    /// ⚠️ THE OVERFLOW FLAGS ARE DROPPED. `visitMulExpr` passes
    /// `arith::IntegerOverflowFlags::nsw` and `visitAddExpr` passes none (`Utils.cpp:62-68`); the
    /// island's [`arith::IntBinary`] carries no attribute dictionary, for the reason recorded there
    /// (nothing in `dcc/src` builds an `arith.muli` with attributes and no `arith.*` op in `dcc/test`
    /// carries a dictionary). ⛔ The one thing `nsw` buys MLIR is the `(a * b) / b -> a` fold, and
    /// that fold is unreachable at this site — see [`fold_to_constant`].
    fn binary(&mut self, wrap: fn(arith::IntBinary) -> arith::Op, lhs: Val, rhs: Val) -> Val {
        let result = self.vals.mint();
        self.built.push(DfirOp::Arith(wrap(arith::IntBinary {
            result,
            lhs,
            rhs,
            ty: ScalarTy::Index,
        })));
        result
    }

    /// `arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, lhs, rhs)`.
    fn cmp_slt(&mut self, lhs: Val, rhs: Val) -> Val {
        let result = self.vals.mint();
        self.built.push(DfirOp::Arith(arith::Op::Compare {
            result,
            predicate: arith::CmpIPredicate::Slt,
            lhs,
            rhs,
        }));
        result
    }

    /// `arith::SelectOp::create(builder, loc, condition, true_value, false_value)`.
    fn select(&mut self, condition: Val, true_value: Val, false_value: Val) -> Val {
        let result = self.vals.mint();
        self.built.push(DfirOp::Arith(arith::Op::Select {
            result,
            condition,
            true_value,
            false_value,
            ty: ScalarTy::Index,
            dbg_name: None,
        }));
        result
    }

    /// `AffineExprVisitor::visit` — the recursive descent, `Value` or null.
    fn visit(&mut self, expr: &AffineExpr) -> Expansion {
        match expr {
            // `visitDimExpr` (`Utils.cpp:194-197`) — NO OP IS BUILT. The expansion of `d0` is the
            // operand itself, which is why an `affine.apply affine_map<(d0) -> (d0)>(%c4)` expands to
            // `%c4` and nothing else.
            AffineExpr::Dim(position) => match self.dim_values.get(*position as usize) {
                Some(val) => Expansion::Value(*val),
                None => Expansion::PositionOutOfRange,
            },
            // `visitSymbolExpr` (`:199-202`) — likewise, and out of the SYMBOL list.
            AffineExpr::Sym(position) => match self.symbol_values.get(*position as usize) {
                Some(val) => Expansion::Value(*val),
                None => Expansion::PositionOutOfRange,
            },
            // `visitConstantExpr` (`:189-192`).
            AffineExpr::Const(value) => Expansion::Value(self.constant(*value)),
            // `visitAddExpr` / `visitMulExpr` (`:62-68`) — `buildBinaryExpr<arith::AddIOp>` and
            // `<arith::MulIOp>`, both `visit(lhs)` then `visit(rhs)` then one op.
            AffineExpr::Add(lhs, rhs) => self.build_binary_expr(arith::Op::AddI, lhs, rhs),
            AffineExpr::Mul(lhs, rhs) => self.build_binary_expr(arith::Op::MulI, lhs, rhs),
            AffineExpr::Mod(lhs, rhs) => self.visit_mod_expr(lhs, rhs),
            AffineExpr::FloorDiv(lhs, rhs) => self.visit_floor_div_expr(lhs, rhs),
        }
    }

    /// `buildBinaryExpr<OpTy>` (`Utils.cpp:52-61`) — `if (!lhs || !rhs) return nullptr;`.
    fn build_binary_expr(
        &mut self,
        wrap: fn(arith::IntBinary) -> arith::Op,
        lhs: &AffineExpr,
        rhs: &AffineExpr,
    ) -> Expansion {
        // ⭐ BOTH SIDES ARE VISITED BEFORE EITHER IS TESTED, and the ops the failing side already
        // built stay built — `visit(expr.getLHS()); visit(expr.getRHS()); if (!lhs || !rhs)` is that
        // order exactly. It matters because a dead expansion's ops are still in the block the
        // reference hands to `signalPassFailure()`.
        let lhs = self.visit(lhs);
        let rhs = self.visit(rhs);
        match (lhs, rhs) {
            (Expansion::Value(lhs), Expansion::Value(rhs)) => {
                Expansion::Value(self.binary(wrap, lhs, rhs))
            }
            // `if (!lhs || !rhs) return nullptr;` — the LHS's cause first, matching `!lhs ||`.
            (
                failed @ (Expansion::NonPositiveModulus(_)
                | Expansion::NonPositiveDivisor(_)
                | Expansion::PositionOutOfRange),
                _,
            ) => failed,
            (_, failed) => failed,
        }
    }

    /// `visitModExpr` (`Utils.cpp:73-100`) — the EUCLIDEAN modulo, whose remainder is never negative.
    ///
    /// ```text
    /// a mod b =
    ///     let remainder = srem a, b;
    ///         negative = a < 0 in
    ///     select negative, remainder + b, remainder.
    /// ```
    ///
    /// ⛔⛔ FIVE OPS, AND THE ROOT IS AN `arith.select` — WHICH IS WHY THIS PASS STOPS THE COMPILE ON
    /// ANY `mod`. See [`expand_affine_apply_ops`]: the site folds the expansion's root ONE level and
    /// keeps only an *attribute* result, and `arith::SelectOp::fold` returns only Values
    /// (`mlir/lib/Dialect/Arith/IR/ArithOps.cpp:2479-2524`) — so the root never becomes an
    /// `arith.constant`, `isConstant<arith::ConstantOp>` is false, and the pass emits
    /// *"Expanded affine.apply operation does not resolve to a constant"* no matter how constant the
    /// operands were.
    ///
    /// ⚠️ THE REFERENCE'S `negative` PREDICATE TESTS THE REMAINDER, NOT `a`. The doc comment says
    /// `negative = a < 0` and the code writes
    /// `CmpIOp::create(.., slt, remainder, zeroCst)` (`:87-88`) — the same answer for every `b > 0`,
    /// since `srem` takes the sign of `a`, but the op built reads `remainder`. The op is what is
    /// ported.
    fn visit_mod_expr(&mut self, lhs: &AffineExpr, rhs: &AffineExpr) -> Expansion {
        // `if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) if (rhsConst.getValue()
        // <= 0) { emitError(..); return nullptr; }` — ⭐ TESTED BEFORE EITHER SIDE IS VISITED, so a
        // refused modulus builds NO ops at all.
        if let AffineExpr::Const(rhs_const) = rhs {
            if *rhs_const <= 0 {
                return Expansion::NonPositiveModulus(*rhs_const);
            }
        }
        // `assert(lhs && rhs && "unexpected affine expr lowering failure")` — ⚠️ AN ASSERT, NOT A
        // NULL RETURN. A nested refusal below a `mod` is undefined behaviour in the reference; here
        // it propagates, which is the answer the assert was asserting could not be needed.
        let Expansion::Value(lhs) = self.visit(lhs) else {
            return self.visit(lhs);
        };
        let Expansion::Value(rhs) = self.visit(rhs) else {
            return self.visit(rhs);
        };
        let remainder = self.binary(arith::Op::RemSI, lhs, rhs);
        let zero_cst = self.constant(0);
        let is_remainder_negative = self.cmp_slt(remainder, zero_cst);
        let corrected_remainder = self.binary(arith::Op::AddI, remainder, rhs);
        Expansion::Value(self.select(is_remainder_negative, corrected_remainder, remainder))
    }

    /// `visitFloorDivExpr` (`Utils.cpp:104-144`) — FLOOR division, rounding towards minus infinity.
    ///
    /// ```text
    /// a floordiv b =
    ///     let negative = a < 0 in
    ///     let absolute = negative ? -a - 1 : a in
    ///     let quotient = absolute / b in
    ///         negative ? -quotient - 1 : quotient
    /// ```
    ///
    /// ⭐ EIGHT OPS AND NO `arith.floordivsi`, DELIBERATELY — the reference's own note is that
    /// lowering `floordivsi` produces *two* `arith.divsi` rather than one (`:113-119`).
    ///
    /// ⛔ AND THE ROOT IS AGAIN AN `arith.select`, so the same stop applies as for
    /// [`Self::visit_mod_expr`].
    fn visit_floor_div_expr(&mut self, lhs: &AffineExpr, rhs: &AffineExpr) -> Expansion {
        if let AffineExpr::Const(rhs_const) = rhs {
            if *rhs_const <= 0 {
                return Expansion::NonPositiveDivisor(*rhs_const);
            }
        }
        let Expansion::Value(lhs) = self.visit(lhs) else {
            return self.visit(lhs);
        };
        let Expansion::Value(rhs) = self.visit(rhs) else {
            return self.visit(rhs);
        };
        // ⭐ BOTH CONSTANTS FIRST, THEN THE PREDICATE — the order `Utils.cpp:131-135` builds them in,
        // and `-1` is `noneCst`, the value both the decrement and the negation are written against.
        let zero_cst = self.constant(0);
        let none_cst = self.constant(-1);
        let negative = self.cmp_slt(lhs, zero_cst);
        let negated_decremented = self.binary(arith::Op::SubI, none_cst, lhs);
        let dividend = self.select(negative, negated_decremented, lhs);
        let quotient = self.binary(arith::Op::DivSI, dividend, rhs);
        let corrected_quotient = self.binary(arith::Op::SubI, none_cst, quotient);
        Expansion::Value(self.select(negative, corrected_quotient, quotient))
    }
}

/// `mlir::affine::expandAffineMap` (`mlir/lib/Dialect/Affine/Utils/Utils.cpp:226-241`) — every result
/// of a map expanded in order, plus the ops that compute them.
///
/// ⭐ THE ARGUMENT LIST SPLITS AT `getNumDims()`: `operands.take_front(numDims)` are the dimension
/// values and `operands.drop_front(numDims)` the symbol values. A map that declares symbols it never
/// mentions still consumes its share of the list — see [`AffineMap::syms`].
///
/// ⛔ THE `std::nullopt` IS `all_of(expanded, [](Value v) { return v; })` — ANY null result poisons
/// the whole call, so a two-result map whose first result expanded fine is still a total refusal. All
/// the results come back here and the caller does that `all_of` itself, because the ops a failing
/// result already built are in the returned list either way.
fn expand_affine_map(
    vals: &mut Values,
    map: &AffineMap,
    operands: &[Val],
) -> (Vec<DfirOp>, Vec<Expansion>) {
    // ⭐ `take_front`/`drop_front` CLAMP rather than read past the end, and a dimension that then has
    // no value is [`Expansion::PositionOutOfRange`] at the visitor.
    let num_dims = (map.dims as usize).min(operands.len());
    let (dim_values, symbol_values) = operands.split_at(num_dims);
    let mut expander = AffineApplyExpander {
        vals,
        dim_values,
        symbol_values,
        built: Vec::new(),
    };
    let expanded = map
        .results
        .iter()
        .map(|expr| expander.visit(expr))
        .collect();
    (expander.built, expanded)
}

/// `Operation::fold(fold_results)` FOLLOWED BY `fold_results[0].dyn_cast<Attribute>()` — the literal
/// an op collapses to, or [`None`] when it does not collapse to a **literal**.
///
/// # ⛔⛔ TWO GATES, AND THE SECOND IS THE ONE THAT SURPRISES
///
/// 1. `Operation::fold(results)` gathers the operands' constants itself —
///    `matchPattern(getOperand(i), m_Constant(&constants[i]))` for each, then calls the fold hook
///    (`mlir/lib/IR/Operation.cpp:657-664`). ⭐ ONE LEVEL ONLY: a non-constant operand arrives as a
///    null `Attribute`, and nothing walks further back.
/// 2. The call site keeps only `fold_results[0].dyn_cast<Attribute>()`
///    (`LoopUnrollForShuffleOp.cpp:216`). ⛔ AN `OpFoldResult` HOLDING A **VALUE** IS DROPPED — and
///    the `arith` folders return Values for every one of their algebraic short-circuits. So
///    `addi(%c5, %c0)` does NOT yield a literal here: `AddIOp::fold` matches `m_Zero()` on the RHS and
///    returns `getLhs()`, a Value (`ArithOps.cpp:394-397`). Only `constFoldBinaryOp` results are
///    attributes.
///
/// ⭐ THE `arith.constant` ARM IS WHY THE REFERENCE EMITS A SECOND, IDENTICAL CONSTANT.
/// `arith::ConstantOp::fold` is `return getValue();` (`ArithOps.cpp:249`) — always an attribute — so
/// an `affine.apply` that expanded to a bare `arith.constant 4 : index` gets a *fresh*
/// `arith.constant 4 : index` built next to it and the first is left where it was. Faithful, and
/// deliberately not cleaned up.
///
/// ⚠️ FOUR `arith` FOLD PATTERNS ARE NOT WRITTEN HERE BECAUSE NO EXPANSION CAN REACH THEM, and each
/// returns a Value anyway — so reaching them and declining are the same answer at this site.
/// `addi(subi(a, b), b) -> a` and `subi(addi(a, b), b) -> a` need an `addi`/`subi` *operand*, and the
/// root of any sub-expansion is a constant, an apply operand, an `addi`, a `muli` or a `select` —
/// never a `subi`. `(a * b) / b -> a` (`foldDivMul`) needs the `divsi`'s LHS to be a `muli`, and
/// [`AffineApplyExpander::visit_floor_div_expr`] always makes it a `select`. `subi(x, x) -> 0` needs
/// two occurrences of one value, and that `subi`'s LHS is a freshly minted `-1`.
fn fold_to_constant(op: &DfirOp, built: &[DfirOp], unit: &[DfirOp]) -> Option<i64> {
    // `matchPattern(getOperand(i), m_Constant(&constants[i]))`, over the ops this expansion built and
    // then the unit the apply lives in — the two places a Value's defining op can be.
    //
    // ⭐ `arith.constant .. : index` ONLY. Every value in an expansion is an `index`
    // ([`AffineApplyExpander::binary`]), and an `arith.constant 4 : i32` feeding an
    // `arith.addi .. : index` is not a well-typed op — MLIR's `SameOperandsAndResultType` rejects it,
    // so [`arith::Op::ConstantInt`] cannot be an operand of anything folded here.
    let constant = |val: Val| -> Option<i64> {
        match defining_op(val, built).or_else(|| defining_op(val, unit))? {
            DfirOp::Arith(arith::Op::Constant { value, .. }) => Some(*value),
            _ => None,
        }
    };
    match op {
        // `arith::ConstantOp::fold` — `return getValue();` (`ArithOps.cpp:249`).
        DfirOp::Arith(arith::Op::Constant { value, .. }) => Some(*value),
        // `AddIOp::fold` (`:394-412`) — `addi(x, 0) -> x` is a **Value**, so a zero RHS declines.
        DfirOp::Arith(arith::Op::AddI(bin)) => match constant(bin.rhs)? {
            0 => None,
            rhs => Some(constant(bin.lhs)?.wrapping_add(rhs)),
        },
        // `SubIOp::fold` (`:482-506`) — `subi(x, 0) -> x` is a Value.
        DfirOp::Arith(arith::Op::SubI(bin)) => match constant(bin.rhs)? {
            0 => None,
            rhs => Some(constant(bin.lhs)?.wrapping_sub(rhs)),
        },
        // `MulIOp::fold` (`:519-532`) — BOTH short-circuits are Values: `muli(x, 0)` returns
        // `getRhs()` and `muli(x, 1)` returns `getLhs()`.
        DfirOp::Arith(arith::Op::MulI(bin)) => match constant(bin.rhs)? {
            0 | 1 => None,
            rhs => Some(constant(bin.lhs)?.wrapping_mul(rhs)),
        },
        // `DivSIOp::fold` (`:727-748`) — `divsi(x, 1) -> x` is a Value, and
        // `return overflowOrDiv0 ? Attribute() : result;` declines a zero divisor and `INT_MIN / -1`.
        DfirOp::Arith(arith::Op::DivSI(bin)) => match constant(bin.rhs)? {
            1 => None,
            rhs => constant(bin.lhs)?.checked_div(rhs),
        },
        // `RemSIOp::fold` (`:928-945`) — ⭐ `remsi(x, 1) -> 0` IS AN ATTRIBUTE
        // (`getZeroAttr(getType())`), the one short-circuit of the five that yields a literal, and it
        // does so WITHOUT the LHS being constant. `div0` declines a zero divisor.
        DfirOp::Arith(arith::Op::RemSI(bin)) => match constant(bin.rhs)? {
            1 => Some(0),
            rhs => constant(bin.lhs)?.checked_rem(rhs),
        },
        // ⛔⛔ `SelectOp::fold` NEVER YIELDS AN ATTRIBUTE — every one of its seven patterns returns a
        // Value (`ArithOps.cpp:2479-2524`), and none of them even matches an expansion's root: the
        // condition is a `cmpi`, not a constant, and the `cmpi`-based pattern needs the two compared
        // values to BE the two arms, whereas [`AffineApplyExpander::visit_mod_expr`] compares the
        // remainder against zero. This is the arm that makes a `mod` or a `floordiv` stop the pass.
        DfirOp::Arith(arith::Op::Select { .. }) => None,
        // ⭐ AND NOTHING ELSE IS EVER ASKED. The op handed to this function defines an expansion's
        // root, which is one of the seven above — an `arith.cmpi` is only ever an interior operand,
        // and no other dialect's op is built by the expander. A `cmpi` folding to a `true`/`false`
        // attribute would be cast to an `IntegerAttr` and `getInt()`-ed into an
        // `arith.constant .. : index` of 0 or 1 by the call site, which is a coercion no reachable
        // program performs.
        //
        // ⭐ `uniform.query_map` IS IN THIS ARM AND NOT IN [`is_arith_constant`]'S. `Operation::fold`
        // on one has no folder registered, so it fails and the reference's
        // `succeeded(defining_op->fold(..))` guard (`:890`) is false — a queried constant does not
        // become a literal here even though it answers *"is a constant"* there.
        DfirOp::Arith(
            arith::Op::ConstantInt { .. }
            | arith::Op::DenseConstant { .. }
            | arith::Op::Compare { .. }
            // ⭐ AND THE TWO CONVERSIONS, WHICH NO EXPANSION BUILDS: every value in one is an
            // `index`, and `sitofp`/`fptosi` change the element type of a vector.
            | arith::Op::SiToFp(_)
            | arith::Op::FpToSi(_)
            | arith::Op::Logic { .. },
        )
        | DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Vector(_)
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => None,
    }
}

/// `dcc::utils::isConstant<arith::ConstantOp>(val)` (`dcc/src/Utils/Utils.cpp:423-443`) — is this
/// value an `arith.constant`?
///
/// ⭐ ONE OP CLASS, THREE ISLAND VARIANTS. `arith::ConstantIndexOp`, `arith::ConstantIntOp` and a
/// dense `arith.constant` are all `arith::ConstantOp` in MLIR — one op with three builders — so
/// `isa<arith::ConstantOp>` accepts all three and so does this. The island splits them because they
/// print different types ([`arith::Op::ConstantInt`] says why).
///
/// ⛔ A BLOCK ARGUMENT IS **FALSE**, not a question about its defining op:
/// `if (mlir::isa<BlockArgument>(val)) return false;` (`:426`). [`defining_op`] answering [`None`] is
/// that same case, and it is the one that stops this pass on an `affine.apply` of a LIVE induction
/// variable — which is exactly what `expandAffineApplyOps` is asserting cannot survive unrolling.
///
/// ⛔ AND `symbol.create_symbol` IS **FALSE** HERE. The template argument is `arith::ConstantOp`;
/// `isConstant<symbol::CreateSymbolOp>` is a *different* instantiation with its own call sites
/// (`:447`). A symbolic index reaching an `affine.apply` in an unrolled loop stops this pass.
///
/// ⭐⭐ A `uniform.query_map` IS **TRUE** WHEN EVERY VALUE OF ITS MAPPING IS ONE — the reference's
/// second accepting branch, and the "or query-based value" half of its own comment at
/// `LoopUnrollForShuffleOp.cpp:224`:
///
/// ```cpp
/// if (auto query_map_op = llvm::dyn_cast<mlir::uniform::QueryMapOp>(val.getDefiningOp())) {
///   auto immutable_map = query_map_op.getMap()
///           .getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
///   DT_CHECK(!immutable_map.getValues().empty());
///   for (auto value : immutable_map.getValues()) {
///     if (!isa<ConstTy>(value.getDefiningOp())) return false;
///   }
///   return true;
/// }
/// ```
/// (`Utils.cpp:428-440`)
///
/// ⭐ THE VALUES, NOT THE KEYS. `getValues()` is the second of [`uniform::Op::DefImmutableMapping`]'s
/// positionally paired operand ranges; the keys are `dataflow.get_unit`s and would never be
/// constants. Which unit the [`uniform::Op::QueryMap`] names is not consulted — *every* core's value
/// has to be a constant for the queried value to be one, which is the whole point of the mapping.
///
/// ⛔ AND THE INNER TEST IS `isa<ConstTy>`, NOT A RECURSION. A mapping whose values are themselves
/// `uniform.query_map`s answers **false**, one rung down, even though this function accepts one at
/// the top.
///
/// # THE TWO SHAPES THE REFERENCE CRASHES ON, BOTH ANSWERED `false` HERE
///
/// * ⛔ A MAP THAT IS NOT A `uniform.def_immutable_mapping`. `getDefiningOp<T>()` answers null and
///   `immutable_map.getValues()` then dereferences it — MLIR's own assert, not a diagnostic. A
///   `uniform.query_map` over a block argument or a `symbol.symbol_immutable_mapping` is exactly
///   that shape.
/// * ⛔ AN EMPTY MAPPING IS `DT_CHECK`'S (`:430`), which `util/dt_exception.hpp` turns into an abort
///   rather than a diagnostic. Note the C++ loop *would* answer true for it — `for` over nothing
///   falls to `return true` — so the check exists precisely to reject the answer the loop gives.
///
/// ⚠️ NO MAPPING IN THE AUTHORITY TREE MAPS ONLY `arith.constant`s, so the accepting arm is
/// transcribed rather than exercised by that corpus. Across `dcc/test`'s 663
/// `uniform.def_immutable_mapping`s the values are `dataflow.get_unit`s (`PT/fp8-bmm.mlir:857`),
/// `dataflow.get_local_unit`s, `sentient.scalar_constant`s (a later stage's), a
/// `vectorchain.constant_bitstream` and — the closest two — `arith.subi`s
/// (`Transform/UniformQueryMapsCanonicalization/dataop.mlir:183`) and `arith.cmpi`s
/// (`Conversion/AgenToSentient/resnet_conv168_1DSC.mlir:681`). Every one of those answers **false**,
/// which is the answer the arm below has to keep giving.
///
/// ⭐ `false` IS THE SAME REFUSAL WITHOUT THE CRASH, and it is the safe direction for this caller:
/// [`expand_affine_apply_ops`] reports
/// [`ExpandAffineApplyOps::DoesNotResolveToAConstant`] and stops the pass, which is what the abort
/// achieves. ⛔ The one thing it must not do is answer `true` — that would let an `affine.apply` be
/// erased in favour of a value the pass never proved constant.
pub(super) fn is_arith_constant(val: Val, built: &[DfirOp], unit: &[DfirOp]) -> bool {
    // `val.getDefiningOp()` — over the ops this expansion built and then the unit the apply lives in,
    // the two places a Value's defining op can be.
    let define = |val: Val| defining_op(val, built).or_else(|| defining_op(val, unit));

    // `isa<ConstTy>(..)`, with `ConstTy = arith::ConstantOp` — the three island variants of the one op
    // class, and no recursion.
    let is_constant_op = |op: Option<&DfirOp>| {
        matches!(
            op,
            Some(DfirOp::Arith(
                arith::Op::Constant { .. }
                    | arith::Op::ConstantInt { .. }
                    | arith::Op::DenseConstant { .. }
            ))
        )
    };

    match define(val) {
        // `if (auto query_map_op = llvm::dyn_cast<mlir::uniform::QueryMapOp>(..))`.
        Some(DfirOp::Uniform(uniform::Op::QueryMap { map, .. })) => {
            match define(*map) {
                Some(DfirOp::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => {
                    // `DT_CHECK(!immutable_map.getValues().empty());` then
                    // `for (auto value : ..) if (!isa<ConstTy>(value.getDefiningOp())) return false;`
                    !pairs.is_empty()
                        && pairs
                            .iter()
                            .all(|(_, value)| is_constant_op(define(*value)))
                }
                // The null `getDefiningOp<DefImmutableMappingOp>()` above.
                _ => false,
            }
        }
        // `if (isa<ConstTy>(val.getDefiningOp())) return true;` (`:428`), and the final
        // `return false;` (`:442`) for everything else — a `symbol.create_symbol` included.
        op => is_constant_op(op),
    }
}

/// WHY `expandAffineMap` ANSWERED `std::nullopt` — [`Expansion`] minus its success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailedToExpand {
    /// `emitError(loc, "modulo by non-positive value is not supported")` (`Utils.cpp:82-84`).
    NonPositiveModulus(i64),
    /// `emitError(loc, "division by non-positive value is not supported")` (`:126-128`).
    NonPositiveDivisor(i64),
    /// A `d<N>` or `s<N>` with no operand — see [`Expansion::PositionOutOfRange`].
    PositionOutOfRange,
    /// `expanded_result->empty()` — the map produced nothing.
    ///
    /// ⭐ THE SECOND HALF OF `!expanded_result.has_value() || expanded_result->empty()`, and NOT the
    /// same condition: `expandAffineMap` returns a non-empty `std::optional` holding an EMPTY vector
    /// for a map with no results, because `all_of` over nothing is true.
    /// `AffineApplyOp::verify` pins `getNumResults() == 1`, so a real apply cannot be this — the
    /// island states the map and the op separately, so it is expressible and gets a name.
    NoResults,
}

/// WHAT [`expand_affine_apply_ops`] DID TO A UNIT.
///
/// ⭐ THE TWO DECLINES ARE THE REFERENCE'S TWO DIAGNOSTICS. `signalPassFailure()` after either is a
/// stop; [`ExpandAffineApplyOps::failed`] is the one question the caller asks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpandAffineApplyOps {
    /// Every `affine.apply` in the unit expanded, resolved to an `arith.constant` and was erased.
    Expanded {
        /// How many it replaced. ⭐ ZERO IS A SUCCESS — a unit with no `affine.apply` is the common
        /// case, and the reference's loop simply does not run.
        applies: usize,
    },
    /// `apply_op.emitError("Failed to expand affine.apply operation"); signalPassFailure(); return;`
    /// (`LoopUnrollForShuffleOp.cpp:201-203`) — `expandAffineMap` answered `std::nullopt`.
    ///
    /// ⭐ THE CAUSE IS CARRIED, WHICH THE REFERENCE'S ONE MESSAGE DOES NOT DO — see [`Expansion`].
    FailedToExpand {
        /// The result of the `affine.apply` that could not be expanded.
        apply: Val,
        /// Why.
        cause: FailedToExpand,
    },
    /// `apply_op.emitOpError("Expanded affine.apply operation does not resolve to a constant");
    /// signalPassFailure(); return;` (`:227-230`).
    ///
    /// ⛔⛔ THE PASS'S REAL ASSERTION, AND THE ONE THAT FIRES. An `affine.apply` survives to here only
    /// if the loop it indexed was fully unrolled, so every operand should be an `arith.constant` and
    /// the whole expression should fold. Anything else — a live induction variable, a
    /// `symbol.create_symbol`, or a top-level `mod`/`floordiv` whose root `arith.select` cannot fold —
    /// lands here.
    DoesNotResolveToAConstant {
        /// The result of the `affine.apply` that would not resolve.
        apply: Val,
    },
}

impl ExpandAffineApplyOps {
    /// DID IT STOP THE PASS? — `signalPassFailure()`.
    #[must_use]
    pub const fn failed(&self) -> bool {
        match self {
            ExpandAffineApplyOps::Expanded { .. } => false,
            ExpandAffineApplyOps::FailedToExpand { .. }
            | ExpandAffineApplyOps::DoesNotResolveToAConstant { .. } => true,
        }
    }
}

/// EVERY `affine.apply` IN A `dataflow.program_unit`, REPLACED BY THE ARITHMETIC IT STANDS FOR.
///
/// # ⛔⛔ THIS IS AN ASSERTION DRESSED AS A REWRITE
///
/// It runs last in `LoopUnrollForShuffleOp` (`:136`), after every loop feeding a
/// `vectorchain.shuffle` has been fully unrolled. At that point every `affine.apply` left in the unit
/// indexes a *known* iteration, so expanding it must land on a literal — and this function stops the
/// compile when it does not. The rewrite is real (the ops are built, the uses re-pointed, the apply
/// erased), but the check at the end is its purpose.
///
/// # THE REFERENCE, WHOLE
///
/// ```text
/// SmallVector<affine::AffineApplyOp> apply_ops;
/// unit.walk([&](affine::AffineApplyOp apply_op) { apply_ops.push_back(apply_op); });
/// for (auto apply_op : apply_ops) {
///   OpBuilder builder(apply_op);
///   auto expanded_result = mlir::affine::expandAffineMap(builder, apply_op.getLoc(),
///       apply_op.getAffineMap(), llvm::to_vector<8>(apply_op.getOperands()));
///   if (!expanded_result.has_value() || expanded_result->empty()) {
///     apply_op.emitError("Failed to expand affine.apply operation");
///     signalPassFailure();
///     return;
///   }
///   Value expanded_value = (*expanded_result)[0];
///   if (auto defining_op = expanded_value.getDefiningOp()) {
///     SmallVector<OpFoldResult> fold_results;
///     if (succeeded(defining_op->fold(fold_results)) && !fold_results.empty()) {
///       if (auto const_attr = fold_results[0].dyn_cast<Attribute>()) {
///         Value const_op = arith::ConstantIndexOp::create(builder, defining_op->getLoc(),
///                              cast<IntegerAttr>(const_attr).getInt());
///         expanded_value = const_op;
///       }
///     }
///   }
///   if (!dcc::utils::isConstant<arith::ConstantOp>(expanded_value)) {
///     apply_op.emitOpError("Expanded affine.apply operation does not resolve to a constant");
///     signalPassFailure();
///     return;
///   }
///   apply_op.getResult().replaceAllUsesWith(expanded_value);
///   apply_op.erase();
/// }
/// ```
/// (`dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183-236`)
///
/// ⭐ `OpBuilder builder(apply_op)` INSERTS **BEFORE** THE APPLY, so the expansion's ops land where
/// the apply was and dominate every use of its result. They are spliced in at the apply's own index
/// here, oldest first, rather than appended.
///
/// ⛔ THE FIRST FAILURE RETURNS, leaving every later `affine.apply` in place. A caller told only a
/// count would be reading a partially rewritten unit; the decline variants carry the apply that
/// stopped it instead.
///
/// ⚠️ THE OPS OF A **FAILED** EXPANSION ARE STILL SPLICED IN. `signalPassFailure()` does not roll the
/// builder back, and the reference prints its diagnostic over an IR that already holds the half-built
/// arithmetic. Reproduced, because a caller that inspects the unit after a decline must see what the
/// reference would.
///
/// # ⭐ COLLECT-THEN-ITERATE, WRITTEN AS FIND-FIRST-REPEATEDLY
///
/// The reference collects the applies before rewriting because the loop mutates the region it walked.
/// This takes the applies one at a time, re-walking after each — the SAME sequence, because an
/// expansion contains no `affine.apply` (so the set never grows) and each pass removes the one it
/// processed (so pre-order re-walking yields the rest in the collected order). What re-walking buys is
/// that every lookup sees the unit as it now is: an apply whose operand is the `arith.constant` a
/// PREVIOUS apply resolved to is the ordinary chained case, and a snapshot taken once at the start
/// would not contain it.
///
/// # ⛔ WHAT IS DROPPED: THE BUILDER AND THE LOCATIONS, AND NOTHING ELSE
///
/// `OpBuilder` becomes an index into the unit's own `Vec`, and `Location` has no island
/// representation — the reference threads `apply_op.getLoc()` through every `create` and then
/// `defining_op->getLoc()` for the folded constant, a diagnostic detail rather than an operand. The
/// `emitError`/`emitOpError` calls become the two decline variants.
///
/// Replaces: e183_expandAffineApplyOps
pub fn expand_affine_apply_ops(vals: &mut Values, unit: &mut Vec<DfirOp>) -> ExpandAffineApplyOps {
    let mut applies = 0;
    loop {
        // ⭐ THE SNAPSHOT IS WHAT `Value::getDefiningOp()` READS. An operand of the apply may be
        // defined in an ENCLOSING region — a hoisted `arith.constant` above the loop the apply sits
        // in — and a lookup restricted to the apply's own region would call it a block argument and
        // stop the pass. This is a whole-unit copy, so the walk below may hold the unit mutably.
        let snapshot = unit.clone();
        match rewrite_first_apply(vals, unit, &snapshot) {
            // The walk found no `affine.apply` — `apply_ops` is exhausted.
            Rewrite::NoneFound => return ExpandAffineApplyOps::Expanded { applies },
            Rewrite::Rewrote => applies += 1,
            Rewrite::Failed(failure) => return failure,
        }
    }
}

/// WHAT ONE SWEEP OF THE WALK DID.
enum Rewrite {
    /// `apply_ops` is empty — nothing left to rewrite.
    NoneFound,
    /// One `affine.apply` was expanded, folded, re-pointed and erased.
    Rewrote,
    /// `signalPassFailure(); return;`
    Failed(ExpandAffineApplyOps),
}

/// THE FIRST `affine.apply` IN PRE-ORDER, REWRITTEN — one iteration of the reference's loop.
///
/// ⛔ IT DESCENDS INTO REGIONS, because `Operation::walk` does. Every `affine.apply` this pass exists
/// for is inside a loop body, or was until the unrolling put it in one, so a top-level-only scan would
/// find none of them.
fn rewrite_first_apply(vals: &mut Values, scope: &mut Vec<DfirOp>, unit: &[DfirOp]) -> Rewrite {
    for at in 0..scope.len() {
        let DfirOp::Affine(affine::Op::Apply { result, map, args }) = &scope[at] else {
            // ⭐ PRE-ORDER, LIKE `Operation::walk`'S DEFAULT — but an `affine.apply` carries no
            // region and a region-carrying op is not an apply, so the two arms are disjoint and the
            // order in which they are tried is not observable.
            let mut descended = Rewrite::NoneFound;
            for region in regions_mut(&mut scope[at]) {
                match rewrite_first_apply(vals, region, unit) {
                    Rewrite::NoneFound => {}
                    found => {
                        descended = found;
                        break;
                    }
                }
            }
            match descended {
                Rewrite::NoneFound => continue,
                found => return found,
            }
        };
        let (result, map, args) = (*result, map.clone(), args.clone());

        // `auto expanded_result = mlir::affine::expandAffineMap(builder, .., map, operands);`
        let (mut built, expanded) = expand_affine_map(vals, &map, &args);

        // `if (!expanded_result.has_value() || expanded_result->empty())` — `has_value()` is
        // `all_of(expanded, ..)` over EVERY result, not just the one that is read.
        let cause = match expanded.first() {
            None => Some(FailedToExpand::NoResults),
            Some(_) => expanded.iter().find_map(failed_to_expand),
        };
        if let Some(cause) = cause {
            splice_before(scope, at, built);
            return Rewrite::Failed(ExpandAffineApplyOps::FailedToExpand {
                apply: result,
                cause,
            });
        }
        // `Value expanded_value = (*expanded_result)[0];` — the one result an apply has.
        let expanded_value = match expanded[0] {
            Expansion::Value(val) => val,
            // Ruled out by the `find_map` above.
            Expansion::NonPositiveModulus(_)
            | Expansion::NonPositiveDivisor(_)
            | Expansion::PositionOutOfRange => {
                splice_before(scope, at, built);
                return Rewrite::Failed(ExpandAffineApplyOps::FailedToExpand {
                    apply: result,
                    cause: FailedToExpand::NoResults,
                });
            }
        };

        // `if (auto defining_op = expanded_value.getDefiningOp())` — ⛔ [`None`] FOR A BLOCK ARGUMENT,
        // and then no fold is attempted at all: the value stays what it was and [`is_arith_constant`]
        // answers on it directly.
        let folded = defining_op(expanded_value, &built)
            .or_else(|| defining_op(expanded_value, unit))
            .and_then(|def| fold_to_constant(def, &built, unit));
        let expanded_value = match folded {
            Some(value) => {
                // `arith::ConstantIndexOp::create(builder, defining_op->getLoc(), ..getInt())` — a
                // NEW op, still before the apply, and NOT a replacement for the one it folded.
                let result = vals.mint();
                built.push(DfirOp::Arith(arith::Op::Constant { result, value }));
                result
            }
            None => expanded_value,
        };

        // `if (!dcc::utils::isConstant<arith::ConstantOp>(expanded_value))`
        if !is_arith_constant(expanded_value, &built, unit) {
            splice_before(scope, at, built);
            return Rewrite::Failed(ExpandAffineApplyOps::DoesNotResolveToAConstant {
                apply: result,
            });
        }

        // `apply_op.getResult().replaceAllUsesWith(expanded_value); apply_op.erase();`
        //
        // ⭐ THE USES ARE ALL IN THIS REGION OR BELOW IT, by SSA dominance: a value defined here
        // cannot be read by an enclosing block. So re-pointing this scope and its nested regions is
        // the whole of `replaceAllUsesWith`.
        let apply_at = at + built.len();
        splice_before(scope, at, built);
        replace_all_uses(scope, result, expanded_value);
        scope.remove(apply_at);
        return Rewrite::Rewrote;
    }
    Rewrite::NoneFound
}

/// `OpBuilder builder(apply_op)` — the ops it built, in front of the op it was anchored to.
fn splice_before(scope: &mut Vec<DfirOp>, at: usize, built: Vec<DfirOp>) {
    scope.splice(at..at, built);
}

/// [`Expansion`] AS A DECLINE, or [`None`] for a value.
fn failed_to_expand(expansion: &Expansion) -> Option<FailedToExpand> {
    match expansion {
        Expansion::Value(_) => None,
        Expansion::NonPositiveModulus(rhs) => Some(FailedToExpand::NonPositiveModulus(*rhs)),
        Expansion::NonPositiveDivisor(rhs) => Some(FailedToExpand::NonPositiveDivisor(*rhs)),
        Expansion::PositionOutOfRange => Some(FailedToExpand::PositionOutOfRange),
    }
}

/// `Value::replaceAllUsesWith` OVER A REGION TREE — [`replace_uses_of_with`] is the one-op form.
fn replace_all_uses(scope: &mut [DfirOp], from: Val, to: Val) {
    for op in scope.iter_mut() {
        replace_uses_of_with(op, from, to);
        for region in regions_mut(op) {
            replace_all_uses(region, from, to);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// 248/384  runOnOperation  —  dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70
// ══════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE CANDIDATE THE WALK INTERRUPTED ON, AND WHAT [`perform_full_unroll`] SAID ABOUT IT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShuffleUnroll {
    /// `candidate` (`:100`, `:107`), named by the induction variable that reached it — the loop the
    /// `vectorchain.shuffle`'s `variable` operand indexes.
    pub iv: Val,
    /// `performFullUnroll(candidate)`; `Unroll::failed()` is `emitError("Cannot unroll candidate")`.
    pub unroll: Unroll,
}

/// WHAT THE PASS DID TO ONE `lxlu` UNIT — the two module walks of `:75-137`, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitUnrolling {
    /// `dcc::getUnitType(unit.getUnits()[0].getDefiningOp<GetUnitOp>())` (`:76-77`) — always
    /// [`DfirUnit::Lxlu`], since `comp != LXLU` is the filter that let the unit through.
    pub on: DfirUnit,
    /// The first walk: one entry per candidate, in the pre-order the reference interrupts on. A
    /// failed entry is last, because `signalPassFailure(); return;` (`:121-124`) ends this unit.
    pub unrolls: Vec<ShuffleUnroll>,
    /// The second walk (`:130-137`), which runs on every `lxlu` unit whatever the first walk did.
    pub applies: ExpandAffineApplyOps,
}

/// THE LOOP A VALUE IS THE INDUCTION VARIABLE OF — `block_arg == for_op.getInductionVar()` for both
/// kinds (`:96-108`).
///
/// ⛔ `None` COVERS ALL THREE OF THE REFERENCE'S NON-CANDIDATES AT ONCE: a `variable` that is not a
/// block argument, an `scf.for` `iter_args` argument (`block_arg != getInductionVar()`), and the
/// `llvm_unreachable("unsupported loop type")` arm — see [`Loop::of`] for why the last is
/// unobservable rather than merely unlikely.
fn loop_of_induction_var(val: Val, scope: &[DfirOp]) -> Option<Loop> {
    let owner = region_owner(val, scope)?;
    match owner {
        DfirOp::Scf(scf::Op::For { iv, .. }) | DfirOp::Affine(affine::Op::For { iv, .. })
            if *iv == val =>
        {
            Loop::of(owner)
        }
        _ => None,
    }
}

/// EVERY CANDIDATE IN A UNIT, PRE-ORDER, FIRST OCCURRENCE ONLY — `unit.walk<WalkOrder::PreOrder>`
/// over `vectorchain::ShuffleOp` and the `for (auto variable : shuffle_op.getVariable())` inside it.
fn shuffle_candidates(scope: &[DfirOp], unit: &[DfirOp], into: &mut Vec<Val>) {
    for op in scope {
        if let DfirOp::VectorChain(vc::Op::Shuffle { variable, .. }) = op {
            for val in variable {
                // `WalkResult::interrupt()` on the first hit — one candidate per walk, so a second
                // `variable` of the same shuffle is only reached on a later walk.
                if loop_of_induction_var(*val, unit).is_some() && !into.contains(val) {
                    into.push(*val);
                }
            }
        }
        for region in regions(op) {
            shuffle_candidates(region, unit, into);
        }
    }
}

/// Replaces: e248_runOnOperation
///
/// **248/384** `LoopUnrollForShuffleOpPass::runOnOperation` —
/// `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70` (68L): a `vectorchain.shuffle`'s lane
/// selector must be a constant by lowering, so every loop whose induction variable feeds one is fully
/// unrolled, and then every `affine.apply` the unrolling left behind is expanded.
///
/// ⛔ THE `while (true)` RE-WALK BECOMES ONE REQUEST PER DISTINCT CANDIDATE, and it must: the
/// reference re-walks because `loopUnrollFull` mutates the unit under it, and that utility is upstream
/// MLIR, so nothing here has performed the unroll the next walk would observe. An unrolled loop is
/// gone and its shuffles read constants, so no candidate recurs — but an inner loop nested in one is
/// duplicated per iteration, and the reference issues one request per COPY where this issues one for
/// the original. See [`Unroll`], which names the request for the same reason.
///
/// ⛔ BOTH WALKS ARE FUSED INTO ONE PASS OVER THE UNITS. A unit's values are defined inside its own
/// region, so neither walk can observe another unit's body, and a failed unroll ends only its own
/// unit's `while (true)` — the second walk runs regardless (`:130-137`).
///
/// ⛔ `DisableThisPass` (`:43-46`, read at `:71`) IS DROPPED — the same
/// [`cl::opt`](super::tf_canonicalize_toggle::run_on_operation) reason: a flag of `dcc-opt`, not a
/// program property, and which pass runs here is a call in [`super::program`].
pub fn run_on_operation<A: Arch>(
    program: &mut dfir::Program<A>,
    vals: &mut Values,
) -> Vec<UnitUnrolling> {
    let mut per_unit = Vec::new();

    // `module_op.walk([&](dataflow::ProgramUnitOp unit) { .. })` — the units are a field of the
    // program, so the walk for them is an iteration (see entry 356's note on the same point).
    for unit in program.units.iter_mut() {
        let on = unit.on.kind();
        // `if (comp != LXLU) return;` (`:78`).
        if !matches!(on, DfirUnit::Lxlu) {
            continue;
        }

        let mut candidates = Vec::new();
        shuffle_candidates(&unit.body, &unit.body, &mut candidates);
        let mut unrolls = Vec::new();
        for iv in candidates {
            // `if (!candidate) break;` (`:115-116`) is the empty list; a candidate always classifies,
            // because that classification is what made it one.
            let Some(candidate) = loop_of_induction_var(iv, &unit.body) else {
                continue;
            };
            let unroll = perform_full_unroll(candidate, &unit.body);
            unrolls.push(ShuffleUnroll { iv, unroll });
            // `if (performFullUnroll(candidate).failed()) { .. signalPassFailure(); return; }`
            if unroll.failed() {
                break;
            }
        }

        // `expandAffineApplyOps(unit);` — the second walk, entry 183.
        let applies = expand_affine_apply_ops(vals, &mut unit.body);
        per_unit.push(UnitUnrolling {
            on,
            unrolls,
            applies,
        });
    }

    per_unit
}

#[cfg(test)]
mod unit_tests {
    use super::{
        ExpandAffineApplyOps, FailedToExpand, Loop, ShuffleUnroll, TripCount, UnitUnrolling,
        Unroll, constant_trip_count, expand_affine_apply_ops, fold_to_constant, is_arith_constant,
        perform_full_unroll, run_on_operation,
    };
    use crate::arch::Target;
    use crate::islands::dataflow_ir::dialects::{
        Op as DfirOp, Val, affine, arith, block_args, dataflow, operands, regions, results, scf,
        symbol, uniform, vectorchain as vc,
    };
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, ScalarTy, Vector};
    use crate::islands::dataflow_ir::{self as dfir, ProgramUnit, Units, Values};
    use crate::units::{Core, Corelet, DfirUnit, Residency, Row};

    /// `%v = arith.constant N : i32` — the signless-integer constant `ConstantIntOp` accepts.
    fn an_i32(result: u32, value: i64) -> DfirOp {
        DfirOp::Arith(arith::Op::ConstantInt {
            result: Val(result),
            value: arith::IntConst::Int { value, bits: 32 },
        })
    }

    /// `%v = arith.constant N : index` — the one `ConstantIntOp::classof` REJECTS.
    fn an_index(result: u32, value: i64) -> DfirOp {
        DfirOp::Arith(arith::Op::Constant {
            result: Val(result),
            value,
        })
    }

    /// `scf.for %iv = %1 to %2 step %3 { }` over three bounds the caller has already declared.
    fn an_scf_loop() -> DfirOp {
        DfirOp::Scf(scf::Op::For {
            iv: Val(0),
            lo: Val(1),
            hi: Val(2),
            step: Val(3),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })
    }

    /// `affine.for %arg7 = 0 to 2 { }` — the vendor's own outermost candidate,
    /// `dcc/test/Transform/LoopUnrolForShuffleOp/ldcvti_pattern.mlir:144`.
    fn the_vendor_candidate() -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv: Val(7),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(2),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })
    }

    /// 🎯 109/384 — THE VENDOR'S CANDIDATE IS AN AFFINE LOOP AND IS UNROLLED FULLY, UNASKED.
    ///
    /// `ldcvti_pattern.mlir` is this pass's only vendor test and every loop in it is an `affine.for`:
    /// the `%arg7` nest (trip 2) and the `%arg9` nest (trip 4) come out fully unrolled and the
    /// `%arg8` nest survives, because only the two whose induction variable feeds a
    /// `vectorchain.shuffle`'s `variable` operand are ever offered as candidates (`:88-112`). The
    /// affine overload asks the loop nothing at all — `return loopUnrollFull(for_op);` (`:164`) — so
    /// the trip count never enters this arm, which is why the fixture's bounds are irrelevant here.
    #[test]
    fn the_vendor_candidate_is_unrolled_fully_without_being_asked_anything() {
        let candidate = the_vendor_candidate();
        assert_eq!(Loop::of(&candidate), Some(Loop::Affine));
        let unroll = perform_full_unroll(Loop::Affine, &[]);
        assert_eq!(unroll, Unroll::Fully);
        assert!(
            !unroll.failed(),
            "the affine overload has no decline of its own"
        );
    }

    /// 🎯 109/384 — AND AN OP THAT IS NOT A LOOP IS NOT A CANDIDATE.
    ///
    /// ⛔ THIS IS WHERE `llvm_unreachable("unsupported loop type")` WENT (`:148`). The reference's
    /// third arm is undefined behaviour on any other op; here the classification answers `None` and
    /// there is no third arm for [`perform_full_unroll`] to have.
    #[test]
    fn an_op_that_is_not_a_loop_is_not_a_candidate() {
        for not_a_loop in [
            DfirOp::Scf(scf::Op::If {
                cond: Val(1),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            }),
            DfirOp::Scf(scf::Op::Parallel {
                ivs: vec![Val(1)],
                body: Vec::new(),
            }),
            DfirOp::Affine(affine::Op::Yield {
                operands: Vec::new(),
            }),
            an_i32(1, 0),
        ] {
            assert_eq!(Loop::of(&not_a_loop), None, "{not_a_loop:?} is not a loop");
        }
    }

    /// 🎯 109/384 — AN `scf.for` OVER SIGNLESS-INTEGER CONSTANTS UNROLLS BY ITS TRIP COUNT.
    ///
    /// `0 to 8 step 1` runs eight times, and the reference asks `loopUnrollByFactor` for exactly
    /// eight because SCF has no full unroll of its own (`:157-159`).
    #[test]
    fn an_scf_loop_over_constant_bounds_unrolls_by_its_trip_count() {
        let scope = [an_i32(1, 0), an_i32(2, 8), an_i32(3, 1), an_scf_loop()];
        let candidate = Loop::of(&scope[3]).expect("an scf.for is a loop");
        assert_eq!(
            candidate,
            Loop::Scf {
                lo: Val(1),
                hi: Val(2),
                step: Val(3)
            },
            "the classification carries the three bounds and nothing else"
        );
        let unroll = perform_full_unroll(candidate, &scope);
        assert!(!unroll.failed());
        match unroll {
            Unroll::ByFactor(trips) => assert_eq!(trips.get(), 8),
            other => panic!("expected a factor, got {other:?}"),
        }
    }

    /// 🎯 110/384 — AN `index`-TYPED BOUND IS NOT A CONSTANT AS FAR AS THIS PASS IS CONCERNED.
    ///
    /// ⛔⛔ THE REFUSAL EVERY ORDINARY `scf.for` HITS. `arith::ConstantIntOp::classof` requires a
    /// signless integer and `index` is not one, so `arith.constant 0 : index` — how MLIR writes an
    /// `scf.for`'s bounds by default — makes all three `dyn_cast`s null and the pass reports
    /// *"Cannot unroll candidate"*. The island's split between
    /// [`arith::Op::Constant`] and [`arith::Op::ConstantInt`] is what lets the port state that
    /// without asking a type at run time.
    #[test]
    fn an_index_typed_bound_is_not_a_signless_integer_constant() {
        let scope = [
            an_index(1, 0),
            an_index(2, 8),
            an_index(3, 1),
            an_scf_loop(),
        ];
        let unroll = perform_full_unroll(Loop::of(&scope[3]).expect("a loop"), &scope);
        assert_eq!(unroll, Unroll::NonConstantTripBound);
        assert!(unroll.failed());
    }

    /// 🎯 110/384 — AND ONE NON-CONSTANT BOUND IS ENOUGH, WHICHEVER OF THE THREE IT IS.
    ///
    /// `if (!lb_const || !ub_const || !step_const) return std::nullopt;` (`:173`) is a disjunction,
    /// so each bound is separately fatal — and a bound no op defines at all (a block argument, whose
    /// `getDefiningOp()` is null) is the same answer as one defined by the wrong op.
    #[test]
    fn any_one_non_constant_bound_refuses() {
        let good = [an_i32(1, 0), an_i32(2, 8), an_i32(3, 1)];
        for spoiled in 0..3 {
            let mut scope: Vec<DfirOp> = good.to_vec();
            scope[spoiled] = an_index(1 + u32::try_from(spoiled).unwrap_or_default(), 4);
            scope.push(an_scf_loop());
            assert_eq!(
                perform_full_unroll(
                    Loop::Scf {
                        lo: Val(1),
                        hi: Val(2),
                        step: Val(3)
                    },
                    &scope
                ),
                Unroll::NonConstantTripBound,
                "bound {spoiled} is not a signless integer constant"
            );
        }
        // ⭐ AND A BOUND NOTHING DEFINES — `getDefiningOp()` is null for a region argument.
        assert_eq!(
            perform_full_unroll(
                Loop::Scf {
                    lo: Val(1),
                    hi: Val(2),
                    step: Val(3)
                },
                &[an_i32(1, 0), an_i32(3, 1)]
            ),
            Unroll::NonConstantTripBound
        );
    }

    /// 🎯 110/384 — THE BOUNDS ARE FOUND WHEREVER THEY WERE DECLARED.
    ///
    /// ⛔ A CONSTANT IS HOISTED OUT OF THE NEST IT IS READ IN, so a lookup that only searched the
    /// top level would call a perfectly constant bound non-constant. `getDefiningOp()` follows the
    /// value, not the block, and [`defining_op`](crate::islands::dataflow_ir::dialects::defining_op)
    /// descends into regions for the same reason.
    #[test]
    fn a_bound_declared_inside_a_region_is_still_found() {
        let scope = [DfirOp::Affine(affine::Op::For {
            iv: Val(9),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: Vec::new(),
            body: vec![an_i32(1, 0), an_i32(2, 4), an_i32(3, 2), an_scf_loop()],
            dbg_name: None,
        })];
        assert_eq!(
            perform_full_unroll(
                Loop::Scf {
                    lo: Val(1),
                    hi: Val(2),
                    step: Val(3)
                },
                &scope
            ),
            Unroll::ByFactor(TripCount(2))
        );
    }

    /// 🎯 110/384 — A STEP THAT DOES NOT ADVANCE IS REFUSED.
    ///
    /// `if (step <= 0) return std::nullopt;` (`:179`) — and `true` is one of those steps, because
    /// `ConstantIntOp::value()` sign-extends a one-bit `1` to **-1**.
    #[test]
    fn a_non_advancing_step_is_refused() {
        for step in [
            an_i32(3, 0),
            an_i32(3, -1),
            DfirOp::Arith(arith::Op::ConstantInt {
                result: Val(3),
                value: arith::IntConst::Bool(true),
            }),
        ] {
            let scope = [an_i32(1, 0), an_i32(2, 8), step];
            assert_eq!(
                perform_full_unroll(
                    Loop::Scf {
                        lo: Val(1),
                        hi: Val(2),
                        step: Val(3)
                    },
                    &scope
                ),
                Unroll::NonPositiveStep
            );
        }
    }

    /// 🎯 110/384 — A CONSTANT RANGE WITH NOTHING IN IT IS NOT AN UNROLL FACTOR.
    ///
    /// ⛔⛔ THE DELIBERATE DIVERGENCE, AND THE DEFECT IT AVOIDS. The reference returns `(ub - lb) /
    /// step` unguarded, so `lb == ub` yields **0** and a reversed range yields a negative — both then
    /// reach `loopUnrollByFactor`, whose factor is a `uint64_t` behind
    /// `assert(unrollFactor > 0)`: an assertion in a debug build, a `tripCount % 0` in a release one,
    /// and an astronomically large factor for the negative. [`TripCount`] cannot hold either, so both
    /// land on the decline the reference already has for a trip count it cannot use.
    #[test]
    fn an_empty_or_reversed_range_is_not_an_unroll_factor() {
        for (lo, hi) in [(0, 0), (8, 0), (i64::MIN, i64::MAX)] {
            let scope = [an_i32(1, lo), an_i32(2, hi), an_i32(3, 1)];
            let unroll = perform_full_unroll(
                Loop::Scf {
                    lo: Val(1),
                    hi: Val(2),
                    step: Val(3),
                },
                &scope,
            );
            assert_eq!(
                unroll,
                Unroll::EmptyOrReversedRange,
                "{lo}..{hi} yields no positive factor"
            );
            assert!(unroll.failed());
        }
    }

    /// 🎯 110/384 — AND THE TRUNCATING DIVISION UNDER-COUNTS, EXACTLY AS THE REFERENCE DOES.
    ///
    /// ⛔ `0 to 7 step 2` RUNS FOUR TIMES AND THIS ASKS FOR THREE. `(7 - 0) / 2` is 3 in C++ integer
    /// division, so the "full" unroll leaves a one-iteration loop behind whenever the step does not
    /// divide the span. Reproduced rather than fixed: the factor is what the reference hands the
    /// utility, and changing it here would change which program comes out. This test is the record
    /// that it is understood, not endorsed.
    #[test]
    fn a_step_that_does_not_divide_the_span_under_counts() {
        let scope = [an_i32(1, 0), an_i32(2, 7), an_i32(3, 2)];
        assert_eq!(
            perform_full_unroll(
                Loop::Scf {
                    lo: Val(1),
                    hi: Val(2),
                    step: Val(3)
                },
                &scope
            ),
            Unroll::ByFactor(TripCount(3)),
            "four iterations, unrolled by three"
        );
    }

    /// 🎯 110/384 — THE HELPER'S OWN ANSWER, SEPARATE FROM THE DISPATCH.
    #[test]
    fn the_trip_count_is_the_span_over_the_step() {
        let scope = [an_i32(1, 2), an_i32(2, 18), an_i32(3, 4)];
        assert_eq!(
            constant_trip_count(Val(1), Val(2), Val(3), &scope),
            super::TripBound::Trips(TripCount(4)),
            "(18 - 2) / 4"
        );
    }

    /// 🎯 109/384 — ONLY THE REFUSALS STOP THE PASS.
    ///
    /// `if (performFullUnroll(candidate).failed())` is the caller's one question (`:121`); it turns
    /// any decline into `emitError("Cannot unroll candidate")` and `signalPassFailure()`.
    #[test]
    fn failure_is_exactly_the_three_declines() {
        assert!(!Unroll::Fully.failed());
        assert!(!Unroll::ByFactor(TripCount(1)).failed());
        assert!(Unroll::NonConstantTripBound.failed());
        assert!(Unroll::EmptyOrReversedRange.failed());
        assert!(Unroll::NonPositiveStep.failed());
    }

    /// 🎯 109/384 — AND THE ISLAND'S NEW LOOP IS A WHOLE OP, NOT A HOLE FOR THIS PASS TO LOOK AT.
    ///
    /// ⛔ AN OP THE ACCESSORS DO NOT KNOW IS AN OP EVERY LATER WALK MIS-READS: its three bounds are
    /// uses, its induction variable is a region argument and not a use, its body is a region, and it
    /// binds no result. [`scf::Op::For`] exists so `performFullUnroll`'s input is expressible, and
    /// this is what makes it expressible *correctly*.
    #[test]
    fn the_island_loop_reads_its_bounds_and_binds_its_variable() {
        let loop_op = DfirOp::Scf(scf::Op::For {
            iv: Val(0),
            lo: Val(1),
            hi: Val(2),
            step: Val(3),
            carried: Vec::new(),
            body: vec![an_i32(4, 1)],
            dbg_name: None,
        });
        assert_eq!(operands(&loop_op), vec![Val(1), Val(2), Val(3)]);
        assert_eq!(
            block_args(&loop_op),
            vec![Val(0)],
            "the iv is not an operand"
        );
        assert_eq!(results(&loop_op), Vec::new(), "this loop carries nothing");
        assert_eq!(regions(&loop_op).len(), 1);
        assert_eq!(regions(&loop_op)[0].len(), 1);
    }

    /// 🎯 109/384 — AND IT PRINTS THE WAY MLIR WRITES IT.
    #[test]
    fn the_island_loop_prints_its_three_bounds() {
        use crate::islands::dataflow_ir::print;
        let mut out = String::new();
        print::emit(&mut out, &an_scf_loop(), 0);
        assert_eq!(out.trim(), "scf.for %0 = %1 to %2 step %3 {\n}");
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    // 183/384  expandAffineApplyOps
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// EVERY OP OF A SCOPE AS TEXT, so an expectation reads as the IR it is.
    fn printed(scope: &[DfirOp]) -> String {
        use crate::islands::dataflow_ir::print;
        let mut out = String::new();
        for op in scope {
            print::emit(&mut out, op, 0);
        }
        out
    }

    /// `%r = affine.apply affine_map<(d0) -> (<expr>)>(%arg)`.
    fn an_apply(result: Val, expr: AffineExpr, args: Vec<Val>) -> DfirOp {
        DfirOp::Affine(affine::Op::Apply {
            result,
            map: AffineMap::unary(expr),
            args,
        })
    }

    /// `%r = arith.muli %v, %v : index` — a use to watch get re-pointed.
    fn squared(result: Val, val: Val) -> DfirOp {
        DfirOp::Arith(arith::Op::MulI(arith::IntBinary {
            result,
            lhs: val,
            rhs: val,
            ty: ScalarTy::Index,
        }))
    }

    /// 🎯 183/384 — ⭐⭐ THE UNROLLER'S OWN `affine.apply`, EXPANDED AND FOLDED TO THE ITERATION IT
    /// NAMES.
    ///
    /// THIS IS THE ONE SHAPE THE PASS ACTUALLY MEETS, and it is MLIR's, not a guess.
    /// `affine::loopUnrollByFactor` remaps the induction variable of unrolled copy `i` with
    ///
    /// ```text
    /// auto bumpMap = AffineMap::get(1, 0, d0 + i * step);
    /// return AffineApplyOp::create(b, forOp.getLoc(), bumpMap, iv);
    /// ```
    /// (`mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp:1047-1052`)
    ///
    /// and then `promoteIfSingleIteration` (`:1057`) replaces the surviving `%iv` with the loop's
    /// lower bound — an `arith.constant` for a constant bound. So what reaches this pass is
    /// `affine.apply affine_map<(d0) -> (d0 + k)>(%c0)`: ⭐ **ONE** LEVEL OF ARITHMETIC OVER TWO
    /// CONSTANTS, which is exactly what a one-level [`fold_to_constant`] can collapse. The
    /// reference's design is coherent because of that, and a nested map would not be — see
    /// [`a_nested_map_does_not_resolve_because_the_fold_is_one_level`].
    #[test]
    fn the_unrollers_bump_map_folds_to_the_iteration_it_names() {
        let mut vals = Values::default();
        let lower_bound = vals.mint();
        let apply = vals.mint();
        let use_of_it = vals.mint();
        let mut unit = vec![
            an_index(lower_bound.0, 0),
            an_apply(
                apply,
                AffineExpr::dim(0).plus(AffineExpr::Const(3)),
                vec![lower_bound],
            ),
            squared(use_of_it, apply),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::Expanded { applies: 1 }
        );
        assert_eq!(
            printed(&unit),
            // ⭐ THE `arith.constant 3` APPEARS TWICE ON PURPOSE: `%3` is the map's own literal, built
            // by `visitConstantExpr`, and `%5` is the fold's, built by the call site from an
            // `IntegerAttr`. Nothing erases the first, and the `arith.addi` that consumed it is left
            // where it was too — dead, and faithfully so.
            "\
%0 = arith.constant 0 : index
%3 = arith.constant 3 : index
%4 = arith.addi %0, %3 : index
%5 = arith.constant 3 : index
%2 = arith.muli %5, %5 : index
",
            "the use reads the folded constant, and the apply is gone"
        );
    }

    /// 🎯 183/384 — A BARE CONSTANT MAP GETS A SECOND, IDENTICAL CONSTANT.
    ///
    /// `arith::ConstantOp::fold` is `return getValue();` (`ArithOps.cpp:249`), so the constant
    /// `visitConstantExpr` built folds to its own attribute and the call site builds another from it.
    #[test]
    fn a_constant_map_is_expanded_and_then_folded_to_a_duplicate() {
        let mut vals = Values::default();
        let apply = vals.mint();
        let use_of_it = vals.mint();
        let mut unit = vec![
            an_apply(apply, AffineExpr::Const(12), Vec::new()),
            squared(use_of_it, apply),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::Expanded { applies: 1 }
        );
        assert_eq!(
            printed(&unit),
            "\
%2 = arith.constant 12 : index
%3 = arith.constant 12 : index
%1 = arith.muli %3, %3 : index
"
        );
    }

    /// 🎯 183/384 — ⛔ A NESTED MAP DOES **NOT** RESOLVE, BECAUSE THE FOLD IS ONE LEVEL DEEP.
    ///
    /// `affine_map<(d0) -> (d0 * 8 + 32)>(%c2)` expands to `muli %c2, %c8` then `addi %muli, %c32`.
    /// `Operation::fold` matches `m_Constant` on the `addi`'s OPERANDS
    /// (`mlir/lib/IR/Operation.cpp:657-664`) and the `muli` is not one, so `constFoldBinaryOp` gets a
    /// null attribute and the fold fails outright — no recursion, no second attempt. The `addi` is
    /// then not an `arith.constant` and the pass stops.
    ///
    /// ⚠️ THE REFERENCE'S OWN COMMENT SAYS THE OPPOSITE — *"after unrolling, all affine expressions
    /// should be foldable to constants"* (`LoopUnrollForShuffleOp.cpp:212-214`). It holds only for the
    /// one-level maps its own unroller emits (see
    /// [`the_unrollers_bump_map_folds_to_the_iteration_it_names`]); a nested map from anywhere else
    /// stops the compile with *"does not resolve to a constant"* however constant its operands are.
    /// ⭐ THE PORT REPRODUCES THE CODE, NOT THE COMMENT.
    #[test]
    fn a_nested_map_does_not_resolve_because_the_fold_is_one_level() {
        let mut vals = Values::default();
        let two = vals.mint();
        let apply = vals.mint();
        let mut unit = vec![
            an_index(two.0, 2),
            an_apply(
                apply,
                AffineExpr::dim(0).times(8).plus(AffineExpr::Const(32)),
                vec![two],
            ),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::DoesNotResolveToAConstant { apply }
        );
        assert_eq!(
            printed(&unit),
            // ⚠️ AND THE HALF-BUILT ARITHMETIC STAYS, beside the `affine.apply` that is still there.
            // `signalPassFailure()` does not roll the builder back.
            "\
%0 = arith.constant 2 : index
%2 = arith.constant 8 : index
%3 = arith.muli %0, %2 : index
%4 = arith.constant 32 : index
%5 = arith.addi %3, %4 : index
%1 = affine.apply affine_map<(d0) -> (d0 * 8 + 32)>(%0)
",
            "the expansion is spliced in and the apply survives the decline"
        );
    }

    /// 🎯 183/384 — ⛔⛔ A TOP-LEVEL `mod` STOPS THE PASS EVEN WITH ALL-CONSTANT OPERANDS.
    ///
    /// `visitModExpr` builds five ops whose root is an `arith.select` (`Utils.cpp:73-100`), and
    /// `arith::SelectOp::fold` returns only Values — never an attribute
    /// (`ArithOps.cpp:2479-2524`). The call site keeps only attributes, so the root never becomes a
    /// constant and `isConstant<arith::ConstantOp>` is false.
    ///
    /// ⭐ THE OPS AND THEIR ORDER ARE THE REFERENCE'S, which is what makes this a port of the
    /// expansion rather than of its result.
    #[test]
    fn a_mod_expands_to_five_ops_and_still_does_not_resolve() {
        let mut vals = Values::default();
        let seven = vals.mint();
        let apply = vals.mint();
        let mut unit = vec![
            an_index(seven.0, 7),
            an_apply(apply, AffineExpr::dim(0).modulo(4), vec![seven]),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::DoesNotResolveToAConstant { apply }
        );
        assert_eq!(
            printed(&unit),
            "\
%0 = arith.constant 7 : index
%2 = arith.constant 4 : index
%3 = arith.remsi %0, %2 : index
%4 = arith.constant 0 : index
%5 = arith.cmpi slt, %3, %4 : index
%6 = arith.addi %3, %2 : index
%7 = arith.select %5, %6, %3 : index
%1 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%0)
",
            "remsi, zero, cmpi slt, addi, select — in that order"
        );
    }

    /// 🎯 183/384 — AND `floordiv` IS THE REFERENCE'S EIGHT OPS, ROOTED IN AN `arith.select` TOO.
    ///
    /// `visitFloorDivExpr` (`Utils.cpp:104-144`) — two constants, a predicate, the negated-decremented
    /// dividend, the select that picks it, the division, the corrected quotient, and the select that
    /// picks that. ⭐ NO `arith.floordivsi`: the reference's own note is that lowering one produces
    /// *two* `arith.divsi` rather than one (`:113-119`).
    #[test]
    fn a_floordiv_expands_to_eight_ops_and_still_does_not_resolve() {
        let mut vals = Values::default();
        let nine = vals.mint();
        let apply = vals.mint();
        let mut unit = vec![
            an_index(nine.0, 9),
            an_apply(apply, AffineExpr::dim(0).floordiv(2), vec![nine]),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::DoesNotResolveToAConstant { apply }
        );
        assert_eq!(
            printed(&unit),
            "\
%0 = arith.constant 9 : index
%2 = arith.constant 2 : index
%3 = arith.constant 0 : index
%4 = arith.constant -1 : index
%5 = arith.cmpi slt, %0, %3 : index
%6 = arith.subi %4, %0 : index
%7 = arith.select %5, %6, %0 : index
%8 = arith.divsi %7, %2 : index
%9 = arith.subi %4, %8 : index
%10 = arith.select %5, %9, %8 : index
%1 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%0)
",
            "the divisor is expanded BEFORE the two constants the correction needs"
        );
    }

    /// 🎯 183/384 — A NON-POSITIVE MODULUS IS A REFUSAL BEFORE ANY OP IS BUILT.
    ///
    /// `if (rhsConst.getValue() <= 0) { emitError(..); return nullptr; }` is tested ahead of
    /// `visit(expr.getLHS())` (`Utils.cpp:81-86`), so the expansion contributes nothing to the IR and
    /// the whole `expandAffineMap` answers `std::nullopt`.
    #[test]
    fn a_non_positive_modulus_builds_nothing() {
        let mut vals = Values::default();
        let four = vals.mint();
        let apply = vals.mint();
        let mut unit = vec![
            an_index(four.0, 4),
            an_apply(apply, AffineExpr::dim(0).modulo(0), vec![four]),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::FailedToExpand {
                apply,
                cause: FailedToExpand::NonPositiveModulus(0)
            }
        );
        assert_eq!(
            printed(&unit),
            "\
%0 = arith.constant 4 : index
%1 = affine.apply affine_map<(d0) -> (d0 mod 0)>(%0)
",
            "nothing was built, so nothing was spliced"
        );
    }

    /// 🎯 183/384 — AND A NON-POSITIVE DIVISOR IS THE OTHER MESSAGE.
    ///
    /// ⭐ TWO CAUSES, ONE REFERENCE DIAGNOSTIC. `expandAffineMap` flattens both nulls into
    /// `std::nullopt` and this pass then prints *"Failed to expand affine.apply operation"* for
    /// either; [`FailedToExpand`] keeps them apart.
    #[test]
    fn a_non_positive_divisor_is_its_own_cause() {
        let mut vals = Values::default();
        let four = vals.mint();
        let apply = vals.mint();
        let mut unit = vec![
            an_index(four.0, 4),
            an_apply(apply, AffineExpr::dim(0).floordiv(-2), vec![four]),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::FailedToExpand {
                apply,
                cause: FailedToExpand::NonPositiveDivisor(-2)
            }
        );
    }

    /// 🎯 183/384 — ⛔ AN APPLY OF A LIVE INDUCTION VARIABLE IS THE FAILURE THIS PASS EXISTS TO RAISE.
    ///
    /// The expansion of `d0` is the operand itself (`visitDimExpr` builds no op), the operand is a
    /// region argument, `getDefiningOp()` is null so no fold is attempted, and
    /// `isConstant<arith::ConstantOp>` answers false on the `isa<BlockArgument>` line
    /// (`dcc/src/Utils/Utils.cpp:426`). ⭐ THAT IS THE ASSERTION: a loop feeding a
    /// `vectorchain.shuffle` that was NOT fully unrolled reaches here with its iterator still live.
    #[test]
    fn an_apply_of_a_live_induction_variable_does_not_resolve() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let lo = vals.mint();
        let hi = vals.mint();
        let step = vals.mint();
        let apply = vals.mint();
        let mut unit = vec![
            an_index(lo.0, 0),
            an_index(hi.0, 4),
            an_index(step.0, 1),
            DfirOp::Scf(scf::Op::For {
                iv,
                lo,
                hi,
                step,
                carried: Vec::new(),
                body: vec![an_apply(apply, AffineExpr::dim(0), vec![iv])],
                dbg_name: None,
            }),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::DoesNotResolveToAConstant { apply },
            "a block argument is not a constant, and no fold is even attempted"
        );
    }

    /// 🎯 183/384 — ⭐ THE WALK DESCENDS INTO A LOOP BODY, because `Operation::walk` does.
    ///
    /// Every `affine.apply` this pass exists for is inside a loop body, or was until the unrolling
    /// promoted it out of one, so a top-level-only scan would find none of them. The operand here is
    /// defined in the ENCLOSING scope — which is why the lookups run against a snapshot of the whole
    /// unit rather than the apply's own region.
    #[test]
    fn an_apply_inside_a_loop_body_is_rewritten_against_an_enclosing_constant() {
        let mut vals = Values::default();
        let five = vals.mint();
        let iv = vals.mint();
        let lo = vals.mint();
        let hi = vals.mint();
        let step = vals.mint();
        let apply = vals.mint();
        let use_of_it = vals.mint();
        let mut unit = vec![
            an_index(five.0, 5),
            an_index(lo.0, 0),
            an_index(hi.0, 4),
            an_index(step.0, 1),
            DfirOp::Scf(scf::Op::For {
                iv,
                lo,
                hi,
                step,
                carried: Vec::new(),
                body: vec![
                    an_apply(
                        apply,
                        AffineExpr::dim(0).plus(AffineExpr::Const(2)),
                        vec![five],
                    ),
                    squared(use_of_it, apply),
                ],
                dbg_name: None,
            }),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::Expanded { applies: 1 }
        );
        assert_eq!(
            printed(&unit),
            "\
%0 = arith.constant 5 : index
%2 = arith.constant 0 : index
%3 = arith.constant 4 : index
%4 = arith.constant 1 : index
scf.for %1 = %2 to %3 step %4 {
  %7 = arith.constant 2 : index
  %8 = arith.addi %0, %7 : index
  %9 = arith.constant 7 : index
  %6 = arith.muli %9, %9 : index
}
",
            "5 + 2, folded inside the body it was found in"
        );
    }

    /// 🎯 183/384 — ⭐ CHAINED APPLIES: THE SECOND READS THE CONSTANT THE FIRST RESOLVED TO.
    ///
    /// This is what a re-walk buys over the reference's collect-then-iterate: the operand lookup for
    /// apply #2 has to see the `arith.constant` that replacing apply #1 introduced, and a snapshot
    /// taken once before the loop would not hold it. Both orders visit the same applies in the same
    /// sequence, so the answer is the reference's.
    #[test]
    fn a_chained_apply_reads_what_the_first_resolved_to() {
        let mut vals = Values::default();
        let one = vals.mint();
        let first = vals.mint();
        let second = vals.mint();
        let use_of_it = vals.mint();
        let mut unit = vec![
            an_index(one.0, 1),
            an_apply(
                first,
                AffineExpr::dim(0).plus(AffineExpr::Const(10)),
                vec![one],
            ),
            an_apply(
                second,
                AffineExpr::dim(0).plus(AffineExpr::Const(100)),
                vec![first],
            ),
            squared(use_of_it, second),
        ];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::Expanded { applies: 2 }
        );
        assert_eq!(
            printed(&unit),
            "\
%0 = arith.constant 1 : index
%4 = arith.constant 10 : index
%5 = arith.addi %0, %4 : index
%6 = arith.constant 11 : index
%7 = arith.constant 100 : index
%8 = arith.addi %6, %7 : index
%9 = arith.constant 111 : index
%3 = arith.muli %9, %9 : index
",
            "1 + 10 = 11, and 11 + 100 = 111 read through the constant the first fold left"
        );
    }

    /// 🎯 183/384 — A UNIT WITH NO `affine.apply` IS AN UNTOUCHED SUCCESS.
    ///
    /// ⭐ THE COMMON CASE. `apply_ops` is empty, the reference's `for` does not run, and nothing is
    /// signalled — so a caller must not read "zero" as a decline.
    #[test]
    fn a_unit_with_no_apply_is_unchanged() {
        let mut vals = Values::default();
        let four = vals.mint();
        let mut unit = vec![an_index(four.0, 4)];
        let before = printed(&unit);

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::Expanded { applies: 0 }
        );
        assert_eq!(printed(&unit), before);
        assert!(!ExpandAffineApplyOps::Expanded { applies: 0 }.failed());
    }

    /// 🎯 183/384 — A `d<N>` WITH NO OPERAND IS A NAMED ANSWER, NOT A PANIC.
    ///
    /// `visitDimExpr` is an `assert` in MLIR (`Utils.cpp:194-196`) and a read past the end without
    /// one; `AffineApplyOp::verify` is what makes it unreachable. The island states the map's arity and
    /// the argument list separately, so the mismatch is expressible here and gets a name.
    #[test]
    fn a_dim_with_no_operand_is_out_of_range() {
        let mut vals = Values::default();
        let apply = vals.mint();
        let mut unit = vec![an_apply(apply, AffineExpr::dim(0), Vec::new())];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::FailedToExpand {
                apply,
                cause: FailedToExpand::PositionOutOfRange
            }
        );
    }

    /// 🎯 183/384 — AND A MAP WITH NO RESULTS IS THE SECOND HALF OF THE REFERENCE'S DISJUNCTION.
    ///
    /// `!expanded_result.has_value() || expanded_result->empty()` — `expandAffineMap` returns a
    /// non-empty `std::optional` holding an EMPTY vector here, because `all_of` over nothing is true.
    #[test]
    fn a_map_with_no_results_fails_to_expand() {
        let mut vals = Values::default();
        let apply = vals.mint();
        let mut unit = vec![DfirOp::Affine(affine::Op::Apply {
            result: apply,
            map: AffineMap {
                dims: 0,
                syms: 0,
                results: Vec::new(),
            },
            args: Vec::new(),
        })];

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::FailedToExpand {
                apply,
                cause: FailedToExpand::NoResults
            }
        );
    }

    /// 🎯 183/384 — ⛔ `remsi(x, 1)` IS THE ONE ALGEBRAIC SHORT-CIRCUIT THAT YIELDS A LITERAL.
    ///
    /// `RemSIOp::fold` returns `getZeroAttr(getType())` for a unit divisor (`ArithOps.cpp:928-931`) —
    /// an ATTRIBUTE, and one that does not need the LHS to be constant. Every other short-circuit in
    /// the five binaries returns a Value and is therefore dropped by this call site. ⭐ TESTED ON THE
    /// HELPER, because the expansion of a `mod` roots in a `select` and never presents the `remsi` for
    /// folding.
    #[test]
    fn the_unit_remainder_folds_to_zero_and_the_others_fold_to_nothing() {
        let mut vals = Values::default();
        let variable = vals.mint();
        let one = vals.mint();
        let zero = vals.mint();
        let five = vals.mint();
        let unit = vec![an_index(one.0, 1), an_index(zero.0, 0), an_index(five.0, 5)];
        let binary = |wrap: fn(arith::IntBinary) -> arith::Op, lhs: Val, rhs: Val| {
            DfirOp::Arith(wrap(arith::IntBinary {
                result: Val(99),
                lhs,
                rhs,
                ty: ScalarTy::Index,
            }))
        };

        assert_eq!(
            fold_to_constant(&binary(arith::Op::RemSI, variable, one), &[], &unit),
            Some(0),
            "remsi(x, 1) -> 0 is an attribute, LHS unread"
        );
        assert_eq!(
            fold_to_constant(&binary(arith::Op::AddI, five, zero), &[], &unit),
            None,
            "addi(x, 0) -> x is a Value, and this site keeps only attributes"
        );
        assert_eq!(
            fold_to_constant(&binary(arith::Op::MulI, five, one), &[], &unit),
            None,
            "muli(x, 1) -> x is a Value"
        );
        assert_eq!(
            fold_to_constant(&binary(arith::Op::DivSI, five, one), &[], &unit),
            None,
            "divsi(x, 1) -> x is a Value"
        );
        assert_eq!(
            fold_to_constant(&binary(arith::Op::DivSI, five, zero), &[], &unit),
            None,
            "and a zero divisor declines outright — `overflowOrDiv0 ? Attribute() : result`"
        );
        assert_eq!(
            fold_to_constant(&binary(arith::Op::MulI, five, five), &[], &unit),
            Some(25),
            "two constants and no short-circuit is the only way to a literal"
        );
    }

    /// ONE KEY OF A MAPPING — a `dataflow.get_unit`, which is what the verifier requires the keys to
    /// be (`Uniform.cpp:477-486`) and the shape the vendor's own mapping pairs against: a per-corelet
    /// `ptrow0` (`dcc/test/PT/fp8-bmm.mlir:87, 857`). ⚠️ This island's emitter writes the `name=` with
    /// its core prefix and this op carries no `num_folds`, so the printed line is not that file's
    /// byte for byte —
    /// unrelated to this entry, and the keys are the operands [`is_arith_constant`] never reads.
    fn a_unit_key(result: Val, core_index: u32) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            reg_locale: None,
            result,
            residency: Residency::Corelet {
                core: Core::checked(core_index).expect("the arch has 32 cores"),
                corelet: Corelet::checked(0).expect("the arch has 2 corelets per core"),
            },
            unit: DfirUnit::PtRow(Row::checked(0).expect("every PT has a row 0")),
            num_folds: None,
        })
    }

    /// `%m = uniform.def_immutable_mapping([%k0 -> %v0], ..):index`, keyed by one unit per value.
    fn a_mapping(vals: &mut Values, result: Val, values: &[Val]) -> Vec<DfirOp> {
        let mut ops = Vec::new();
        let mut pairs = Vec::new();
        for (core_index, value) in values.iter().enumerate() {
            let key = vals.mint();
            ops.push(a_unit_key(
                key,
                u32::try_from(core_index).expect("a small core index"),
            ));
            pairs.push((key, *value));
        }
        ops.push(DfirOp::Uniform(uniform::Op::DefImmutableMapping {
            result,
            pairs,
        }));
        ops
    }

    /// `%q = uniform.query_map(map:%m, key:%k) : index`.
    fn a_query(result: Val, map: Val, key: Val) -> DfirOp {
        DfirOp::Uniform(uniform::Op::QueryMap { result, map, key })
    }

    /// 🎯 183/384 — ⭐⭐ `isConstant`'S SECOND ACCEPTING BRANCH: A `uniform.query_map` WHOSE MAPPING
    /// HOLDS ONLY CONSTANTS.
    ///
    /// `Utils.cpp:428-440`, quoted on [`is_arith_constant`] — the "or query-based value" half of the
    /// reference's own comment at `LoopUnrollForShuffleOp.cpp:224`. ⛔ The inner test is `isa<ConstTy>`
    /// on each VALUE, not a recursion, and the two shapes that abort in the reference — an empty
    /// mapping (`DT_CHECK`) and a map that is not a `uniform.def_immutable_mapping` (a null
    /// `getDefiningOp<T>()`) — are `false` here rather than a crash.
    #[test]
    fn a_queried_value_is_constant_only_when_every_core_of_the_mapping_is() {
        let mut vals = Values::default();
        let five = vals.mint();
        let seven = vals.mint();
        let symbolic = vals.mint();
        let mut unit = vec![
            an_index(five.0, 5),
            an_index(seven.0, 7),
            // A value the schedule fixes later — `isConstant<symbol::CreateSymbolOp>` is a DIFFERENT
            // instantiation, so this one is not a constant to this call.
            DfirOp::Symbol(symbol::Op::CreateSymbol {
                result: symbolic,
                symbol_id: 0,
                max_value: None,
            }),
        ];

        // ⭐ TWO CORES WITH DIFFERENT CONSTANTS IS THE ACCEPTED CASE, and it is what the op exists
        // for: *"constant differences between the per-core programs of a unit"* (`Uniform.td:118-125`).
        let constants_map = vals.mint();
        let queried_constant = vals.mint();
        // A mapping one of whose cores is symbolic.
        let mixed_map = vals.mint();
        let queried_mixed = vals.mint();
        // `DT_CHECK(!immutable_map.getValues().empty());`
        let empty_map = vals.mint();
        let queried_empty = vals.mint();
        // A `map` operand no `uniform.def_immutable_mapping` defines.
        let queried_symbol_map = vals.mint();
        // A mapping OF queries, each of which would answer true one rung up.
        let nested_map = vals.mint();
        let queried_nested = vals.mint();
        let key = vals.mint();

        let mut ops = a_mapping(&mut vals, constants_map, &[five, seven]);
        ops.push(a_query(queried_constant, constants_map, key));
        ops.extend(a_mapping(&mut vals, mixed_map, &[five, symbolic]));
        ops.push(a_query(queried_mixed, mixed_map, key));
        ops.extend(a_mapping(&mut vals, empty_map, &[]));
        ops.push(a_query(queried_empty, empty_map, key));
        ops.push(a_query(queried_symbol_map, symbolic, key));
        ops.extend(a_mapping(&mut vals, nested_map, &[queried_constant]));
        ops.push(a_query(queried_nested, nested_map, key));
        unit.extend(ops);

        assert!(
            is_arith_constant(queried_constant, &[], &unit),
            "every value of the mapping is an arith.constant"
        );
        assert!(
            !is_arith_constant(queried_mixed, &[], &unit),
            "one symbolic core is enough — `if (!isa<ConstTy>(..)) return false`"
        );
        assert!(
            !is_arith_constant(queried_empty, &[], &unit),
            "an empty mapping is DT_CHECK's abort, and the C++ loop would have said true"
        );
        assert!(
            !is_arith_constant(queried_symbol_map, &[], &unit),
            "a map that is not a def_immutable_mapping is the null getDefiningOp<T>()"
        );
        assert!(
            !is_arith_constant(queried_nested, &[], &unit),
            "isa<ConstTy>, not a recursion — a queried value is not a constant OPERATION"
        );
        // And the three unqueried shapes, for the arm they share.
        assert!(is_arith_constant(five, &[], &unit));
        assert!(!is_arith_constant(symbolic, &[], &unit));
        assert!(
            !is_arith_constant(key, &[], &unit),
            "a block argument is `isa<BlockArgument>(val)` — false without asking anything"
        );
    }

    /// 🎯 183/384 — ⭐ AND AN `affine.apply` OF A QUERIED CONSTANT IS ERASED IN FAVOUR OF THE QUERY.
    ///
    /// The whole call site, end to end, on the one input where the second accepting branch decides
    /// it: the expansion of `(d0) -> (d0)` IS the operand, `Operation::fold` on a
    /// `uniform.query_map` has no folder and so fails — which is why `uniform` sits in
    /// [`fold_to_constant`]'s declining arm — and `isConstant` then answers **true** anyway. So the
    /// apply's uses are re-pointed at the query itself, NOT at a literal.
    #[test]
    fn an_apply_of_a_queried_constant_resolves_to_the_query_and_not_to_a_literal() {
        let mut vals = Values::default();
        let five = vals.mint();
        let seven = vals.mint();
        let map = vals.mint();
        let queried = vals.mint();
        let key = vals.mint();
        let apply = vals.mint();
        let use_of_it = vals.mint();

        let mut unit = vec![an_index(five.0, 5), an_index(seven.0, 7)];
        unit.extend(a_mapping(&mut vals, map, &[five, seven]));
        unit.push(a_query(queried, map, key));
        unit.push(an_apply(apply, AffineExpr::dim(0), vec![queried]));
        unit.push(squared(use_of_it, apply));

        assert_eq!(
            expand_affine_apply_ops(&mut vals, &mut unit),
            ExpandAffineApplyOps::Expanded { applies: 1 }
        );
        assert_eq!(
            printed(&unit),
            "\
%0 = arith.constant 5 : index
%1 = arith.constant 7 : index
%7 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = \"C0-ptrow0-CL0\", type = \"ptrow0\"} : index
%8 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = \"C1-ptrow0-CL0\", type = \"ptrow0\"} : index
%2 = uniform.def_immutable_mapping([%7 -> %0], [%8 -> %1]):index
%3 = uniform.query_map(map:%2, key:%4) : index
%6 = arith.muli %3, %3 : index
",
            "the use reads the query, and the apply is gone"
        );
    }

    /// 🎯 248/384 — THE VENDOR'S OWN CASE:
    /// `dcc/test/Transform/LoopUnrolForShuffleOp/ldcvti_pattern.mlir`. Three nested `affine.for`s —
    /// `%arg7 = 0 to 2`, `%arg8 = 0 to 4`, `%arg9 = 0 to 4` — and two `vectorchain.shuffle`s whose
    /// `variable` reads `%arg7` and `%arg9`. Its `CHECK-SENT-IR` output keeps exactly
    /// `affine.for %VAL_12 = 0 to 4` (`%arg8`, whose iv no shuffle reads) and has unrolled the other
    /// two away.
    ///
    /// ⛔ THE EXPECTATION IS TWO REQUESTS WHERE THE REFERENCE MAKES THREE, and this case is what
    /// shows it: re-walking after `%arg7`'s unroll finds `%arg9`'s loop **twice**, once per copy of
    /// the outer body, which is why the output holds 8 shuffled bodies rather than 2 + 4. See
    /// [`run_on_operation`] — the duplication is `loopUnrollFull`'s, and that is upstream MLIR.
    ///
    /// ⭐ AND `applies: 0`: the input carries no `affine.apply`. The reference's output holds three
    /// (`arith.addi` + a folded constant per unrolled iteration, `:34-36` of the CHECK) because the
    /// unroll it performed created them.
    #[test]
    fn the_two_loops_a_shuffle_variable_reads_are_asked_for_and_the_third_is_not() {
        const V128: Vector = Vector {
            len: 128,
            elem: ElemType::F8E4M3Fn,
        };
        const V64: Vector = Vector {
            len: 64,
            elem: ElemType::F8E4M3Fn,
        };

        let mut vals = Values::default();
        let (arg7, arg8, arg9) = (vals.mint(), vals.mint(), vals.mint());
        let scale = vals.mint();
        let stick = vals.mint();

        // `%910 = vectorchain.shuffle input(%909), variable(%arg7) { indices = [-1], repetition = 64 }`
        let shuffle = |result: Val, input: Val, variable: Val| {
            DfirOp::VectorChain(vc::Op::Shuffle {
                result,
                input,
                variable: vec![variable],
                pad: Vec::new(),
                mask: None,
                indices: vec![-1],
                repetition: 64,
                input_ty: V128,
                ty: V64,
            })
        };
        let a_for = |iv: Val, hi: i64, body: Vec<DfirOp>| {
            DfirOp::Affine(affine::Op::For {
                iv,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(hi),
                carried: Vec::new(),
                body,
                dbg_name: None,
            })
        };
        let body = vec![
            shuffle(vals.mint(), scale, arg7),
            shuffle(vals.mint(), stick, arg9),
        ];
        let unit = vec![a_for(
            arg7,
            2,
            vec![a_for(arg8, 4, vec![a_for(arg9, 4, body)])],
        )];

        let program_on = |on: DfirUnit, body: Vec<DfirOp>| {
            use crate::generated::OpFunc;
            use crate::islands::dataflow_ir::{Grid, GroupId, OpIndex, ProgramName, ProgramUnits};
            dfir::Program::<Target> {
                name: ProgramName {
                    group: GroupId(0),
                    index: OpIndex(0),
                    func: OpFunc::Add,
                },
                grid: Grid::single(),
                preamble: Vec::new(),
                units: ProgramUnits::of(
                    ProgramUnit {
                        on: Units::one(on, Val(0)),
                        precision: None,
                        body,
                        arch: core::marker::PhantomData,
                    },
                    Vec::new(),
                ),
                arch: core::marker::PhantomData,
            }
        };

        let mut program = program_on(DfirUnit::Lxlu, unit.clone());
        assert_eq!(
            run_on_operation(&mut program, &mut vals),
            vec![UnitUnrolling {
                on: DfirUnit::Lxlu,
                unrolls: vec![
                    ShuffleUnroll {
                        iv: arg7,
                        unroll: Unroll::Fully,
                    },
                    ShuffleUnroll {
                        iv: arg9,
                        unroll: Unroll::Fully,
                    },
                ],
                applies: ExpandAffineApplyOps::Expanded { applies: 0 },
            }],
            "`%arg8`'s loop is the one the vendor's output keeps"
        );

        // ⛔ `if (comp != LXLU) return;` — the same program on the PT row is not this pass's.
        let mut elsewhere = program_on(
            DfirUnit::PtRow(Row::checked(0).expect("every PT has a row 0")),
            unit,
        );
        assert_eq!(run_on_operation(&mut elsewhere, &mut vals), Vec::new());
    }
}
