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

use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, arith, defining_op, scf};

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
/// well-formed **answer** rather than a stop — and the reference agrees that it is an answer, because
/// its own walk asks the same question with a `dyn_cast` before it ever calls this
/// (`:96-106`).
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
            | DfirOp::VectorChain(_) => None,
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
    fn positive(trips: i64) -> Option<Self> {
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
/// `.failed()` of (`:135`), not an error to propagate — and this crate freezes `Result` in the
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
    /// `emitError("Cannot unroll candidate")` and `signalPassFailure()` (`:134-137`), so this variant
    /// is a stop for the compilation, not for the port.
    NonConstantTripBound,
    /// A CONSTANT TRIP COUNT THAT IS NOT AN UNROLL FACTOR — see [`TripCount`] for the divergence.
    EmptyOrReversedRange,
    /// `if (step <= 0) return std::nullopt;` (`:180`) — a non-advancing or backward step.
    ///
    /// ⭐ ITS OWN VARIANT THOUGH THE REFERENCE FOLDS IT INTO `nullopt`, because it is a different
    /// fact about the loop from "the bound is not a constant" and both reach the same `failure()`.
    NonPositiveStep,
}

impl Unroll {
    /// `LogicalResult::failed()` — the one question `runOnOperation` asks of this result (`:135`).
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
/// ⭐ ALL THREE ARE FETCHED BEFORE ANY IS CHECKED, WHICH IS WHY THE TUPLE IS THERE. `:171-174` looks
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
        // `:174` — `if (!lb_const || !ub_const || !step_const) return std::nullopt;`
        return TripBound::NoConstant;
    };

    // `:180` — `if (step <= 0) return std::nullopt;`
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
        | DfirOp::VectorChain(_) => None,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{Loop, TripCount, Unroll, constant_trip_count, perform_full_unroll};
    use crate::islands::dataflow_ir::dialects::{
        Op as DfirOp, Val, affine, arith, block_args, operands, regions, results, scf,
    };

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
            body: Vec::new(),
        })
    }

    /// `affine.for %arg7 = 0 to 2 { }` — the vendor's own outermost candidate,
    /// `dcc/test/Transform/LoopUnrolForShuffleOp/ldcvti_pattern.mlir:130`.
    fn the_vendor_candidate() -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv: Val(7),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(2),
            carried: Vec::new(),
            body: Vec::new(),
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
        assert!(!unroll.failed(), "the affine overload has no decline of its own");
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
                body: Vec::new(),
                else_body: Vec::new(),
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
        let scope = [an_index(1, 0), an_index(2, 8), an_index(3, 1), an_scf_loop()];
        let unroll = perform_full_unroll(Loop::of(&scope[3]).expect("a loop"), &scope);
        assert_eq!(unroll, Unroll::NonConstantTripBound);
        assert!(unroll.failed());
    }

    /// 🎯 110/384 — AND ONE NON-CONSTANT BOUND IS ENOUGH, WHICHEVER OF THE THREE IT IS.
    ///
    /// `if (!lb_const || !ub_const || !step_const) return std::nullopt;` (`:174`) is a disjunction,
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
                perform_full_unroll(Loop::Scf { lo: Val(1), hi: Val(2), step: Val(3) }, &scope),
                Unroll::NonConstantTripBound,
                "bound {spoiled} is not a signless integer constant"
            );
        }
        // ⭐ AND A BOUND NOTHING DEFINES — `getDefiningOp()` is null for a region argument.
        assert_eq!(
            perform_full_unroll(
                Loop::Scf { lo: Val(1), hi: Val(2), step: Val(3) },
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
        })];
        assert_eq!(
            perform_full_unroll(Loop::Scf { lo: Val(1), hi: Val(2), step: Val(3) }, &scope),
            Unroll::ByFactor(TripCount(2))
        );
    }

    /// 🎯 110/384 — A STEP THAT DOES NOT ADVANCE IS REFUSED.
    ///
    /// `if (step <= 0) return std::nullopt;` (`:180`) — and `true` is one of those steps, because
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
                perform_full_unroll(Loop::Scf { lo: Val(1), hi: Val(2), step: Val(3) }, &scope),
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
            let unroll =
                perform_full_unroll(Loop::Scf { lo: Val(1), hi: Val(2), step: Val(3) }, &scope);
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
            perform_full_unroll(Loop::Scf { lo: Val(1), hi: Val(2), step: Val(3) }, &scope),
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
    /// `if (performFullUnroll(candidate).failed())` is the caller's one question (`:135`); it turns
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
            body: vec![an_i32(4, 1)],
        });
        assert_eq!(operands(&loop_op), vec![Val(1), Val(2), Val(3)]);
        assert_eq!(block_args(&loop_op), vec![Val(0)], "the iv is not an operand");
        assert_eq!(results(&loop_op), Vec::new(), "no iter_args in this island");
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
}
