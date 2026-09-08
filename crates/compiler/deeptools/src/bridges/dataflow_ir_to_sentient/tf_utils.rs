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

//! `Utils.cpp` — 2 of bridge 2's 384 functions (dependency level(s) [0, 2]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e142_getDataflowForLoopInfoIfIV` | 142/384 | 31 | `dcc/src/Transform/Dataflow/Utils.cpp:99` |
//! | `e264_createForOpWithAdditionalReturnValue` | 264/384 | 66 | `dcc/src/Transform/Dataflow/Utils.cpp:28` |

use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, arith, block_args, defining_op, regions, scf,
};
use crate::islands::dataflow_ir::{ValueMapping, Values};

/// A CONSTANT LOOP BOUND — `getConstantLowerBound()` / `getConstantUpperBound()`.
///
/// ⛔ SIGNED, BECAUSE MLIR'S IS. `affine::AffineForOp` answers `int64_t` and the reference carries it
/// as one (`Utils.cpp:97`); a lower bound below zero is expressible in the dialect even where this
/// crate's emitters never write one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LoopBound(pub i64);

/// A LOOP'S STRIDE — `affine_for.getStepAsInt()`, or the `arith.constant` behind `scf_for.getStep()`.
///
/// ⛔⛔ POSITIVE BY CONSTRUCTION, BECAUSE IT IS A DIVISOR WHOSE QUOTIENT IS A COUNT.
/// `getDataflowForLoopInfoIfIV` computes `(ub - lb) / step` and its only consumer hands that to
/// `new std::optional<int64_t>[num_iterations]` (`CFGSDataflowConditionalTree.hpp:74-75`). A zero
/// step makes the division undefined; a negative one makes the count negative. Neither is a loop, so
/// [`LoopStep::checked`] declines both rather than letting the arithmetic decide at run time.
///
/// ⭐ AND EVERY STEP THE CORPUS NAMES IS ONE. Of the 49 `scf.for`s under `dcc/test`, every one whose
/// step operand names a source constant names `%c1` or `%one_index` — 37 of them; the other twelve
/// are `CHECK` lines whose `%[[VAL_n]]` capture hides the literal. And `affine.for`'s step attribute
/// defaults to 1 with nothing in [`affine::Op::For`] to carry another, so [`LoopStep::ONE`] is the
/// value the affine arm always produces and the division below is written out only because the `scf`
/// arm can produce others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LoopStep(i64);

impl LoopStep {
    /// The dialect's default, and the only stride an `affine.for` in this island has.
    pub const ONE: LoopStep = LoopStep(1);

    /// A stride from an `arith.constant` — [`None`] for a value that is not a positive stride.
    #[must_use]
    pub const fn checked(value: i64) -> Option<LoopStep> {
        if value > 0 {
            Some(LoopStep(value))
        } else {
            None
        }
    }

    /// The stride, for the arithmetic that reads it.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// HOW MANY TIMES A LOOP RUNS — `(ub - lb) / step`.
///
/// ⛔ UNSIGNED, AND THAT IS A DELIBERATE DEPARTURE. The reference computes this as `int64_t` and its
/// only consumer feeds it straight to `new std::optional<int64_t>[num_iterations]`
/// (`CFGSDataflowConditionalTree.hpp:74-75`) — so a loop whose constant upper bound is below its
/// lower bound, which runs zero times, would size an array with a NEGATIVE count. A trip count is a
/// count; [`Iterations::between`] saturates at zero, which is what the loop does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Iterations(u64);

impl Iterations {
    /// `(ub - lb) / step`, floored at zero.
    #[must_use]
    pub const fn between(lo: LoopBound, hi: LoopBound, step: LoopStep) -> Iterations {
        // A saturating span, then the stride: an empty loop is zero iterations, not a negative span
        // divided by anything.
        let span = hi.0.saturating_sub(lo.0);
        if span <= 0 {
            Iterations(0)
        } else {
            // [`LoopStep`] is positive by construction, so this cannot divide by zero and the
            // quotient is the reference's own — a TRUNCATING division, which is what `int64_t / …`
            // is and what a partial final trip means for this count.
            Iterations(span.unsigned_abs() / step.0.unsigned_abs())
        }
    }

    /// The count.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// WHAT A LOOP INDUCTION VARIABLE ANSWERS ABOUT ITS LOOP — the reference's
/// `std::tuple<Operation*, int64_t, int64_t, int64_t, int64_t>` (`Utils.cpp:97-98`), named.
///
/// ⛔ THE NULL TUPLE IS THE `None`, NOT A ZERO-ITERATION LOOP. `getDataflowForLoopInfoIfIV` returns
/// `(nullptr, 0, 0, 0, 0)` for anything that is not a constant-bounded loop's induction variable, and
/// its caller tests `std::get<0>` alone: *"This could happen when the for_op has symbolic loop
/// bounds"* (`CFGSDataflowConditionalTree.hpp:67-71`). A struct of five numbers with a nullable first
/// field would make `(nullptr, 0, 0, 0, 0)` and "a loop from 0 to 0" the same value; an [`Option`]
/// keeps them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataflowForLoopInfo<'a> {
    /// `std::get<0>` — the loop the induction variable belongs to.
    pub for_op: &'a DfirOp,
    /// `std::get<1>` — its constant lower bound. Read by `parseConditional` (`.cpp:542`).
    pub lo: LoopBound,
    /// `std::get<2>` — its constant upper bound.
    ///
    /// ⭐ CARRIED THOUGH NOTHING READS IT. The tuple has it and no consumer in the tree takes
    /// `std::get<2>`; dropping it would make the port answer less than the function does.
    pub hi: LoopBound,
    /// `std::get<3>` — its stride. Read by `parseConditional` (`.cpp:543`).
    pub step: LoopStep,
    /// `std::get<4>` — its trip count, which sizes the caller's value array
    /// (`CFGSDataflowConditionalTree.hpp:74-75`).
    pub iterations: Iterations,
}

/// `cast<BlockArgument>(val).getOwner()->getParentOp()` — the op whose region binds `val` as a block
/// argument, or [`None`] when `val` is not a block argument at all (`isa<BlockArgument>(val)`).
///
/// ⭐ ONE HELPER FOR BOTH HALVES OF THE REFERENCE'S FIRST TWO LINES. There it is a type test followed
/// by two pointer hops through the block to its parent; here a [`Val`] carries no such link, so the
/// walk over [`block_args`] IS the lookup — the brief's "mechanism for reaching operands"
/// (`AGENT-BRIEF.md:53-56`) restated for a value tree.
///
/// ⛔ IT DESCENDS INTO REGIONS, because a loop inside a `program_unit` inside a loop is the ordinary
/// case and the argument of an inner loop is bound arbitrarily deep.
fn owner_of_block_arg(val: Val, scope: &[DfirOp]) -> Option<&DfirOp> {
    for op in scope {
        if block_args(op).contains(&val) {
            return Some(op);
        }
        for region in regions(op) {
            if let Some(found) = owner_of_block_arg(val, region) {
                return Some(found);
            }
        }
    }
    None
}

/// Replaces: e142_getDataflowForLoopInfoIfIV
///
/// **142/384** `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99` (31L).
///
/// ```cpp
/// std::tuple<Operation*, int64_t, int64_t, int64_t, int64_t>
/// getDataflowForLoopInfoIfIV(Value& val) {
///   if (isa<BlockArgument>(val)) {
///     Operation* for_op = cast<BlockArgument>(val).getOwner()->getParentOp();
///     if (auto scf_for = llvm::dyn_cast<mlir::scf::ForOp>(for_op)) {
///       auto lb_const =
///           scf_for.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
///       auto ub_const =
///           scf_for.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
///       auto step_const =
///           scf_for.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
///       if (lb_const && ub_const && step_const &&
///           scf_for.getInductionVar() == val) {
///         auto num_iterations =
///             (ub_const.value() - lb_const.value()) / step_const.value();
///         return std::make_tuple(for_op, lb_const.value(), ub_const.value(),
///                                step_const.value(), num_iterations);
///       }
///     } else if (auto affine_for =
///                    llvm::dyn_cast<mlir::affine::AffineForOp>(for_op)) {
///       if (affine_for.hasConstantLowerBound() &&
///           affine_for.hasConstantUpperBound() &&
///           affine_for.getInductionVar() == val) {
///         auto lb = affine_for.getConstantLowerBound();
///         auto ub = affine_for.getConstantUpperBound();
///         auto step = affine_for.getStepAsInt();
///         auto num_iterations = (ub - lb) / step;
///         return std::make_tuple(for_op, lb, ub, step, num_iterations);
///       }
///     }
///   }
///   return std::make_tuple(nullptr, 0, 0, 0, 0);
/// }
/// ```
///
/// # ⛔ THE THREE WAYS OUT ARE ONE VALUE
///
/// A value that is not a block argument, a block argument whose parent is not a `for`, a `for` whose
/// bounds are not constants, and a block argument that is a CARRIED value rather than the induction
/// variable all reach the same `(nullptr, 0, 0, 0, 0)`. The last of those is the point of the
/// function's name: `getInductionVar() == val` is what makes it "…IfIV", and an `iter_args` region
/// argument shares the block with the induction variable.
///
/// # ⛔⛔ BOTH ARMS ARE PORTED, AND THE `scf` ONE COST THE ISLAND AN OP
///
/// The `scf::ForOp` arm is tested FIRST and the vendor has a case for it: case 3 of
/// `dcc/test/Transform/CFGSimplificationDataflowLevel/simplify-conditional.mlir:241` is *"Same as 2.
/// but with scf.for instead of affine.for"*, whose input is `scf.for %arg5 = %c0 to %c4 step %c1`
/// (`:319`). This island had no `scf.for`, so that arm had no expressible input —
/// [`scf::Op::For`] was added for it, which is the campaign's rule and not a licence taken:
/// *"If the target IR cannot express a function's input, add the operation to the island"*
/// (`AGENT-BRIEF.md:57`). Answering [`None`] there instead would have been a port that declines the
/// vendor's own test.
///
/// # ⭐ THE TWO ARMS ASK THE SAME QUESTION OF DIFFERENT THINGS
///
/// `scf.for`'s three bounds are OPERANDS, so "is it constant" is a question about the defining op and
/// the reference walks `getDefiningOp<arith::ConstantIndexOp>()` three times — [`defining_op`] and
/// [`arith::Op::Constant`] here, which is that op class exactly (`StandardToSentient.cpp:347`).
/// `affine.for`'s two bounds are ATTRIBUTES, so `hasConstantLowerBound()`/`hasConstantUpperBound()`
/// is [`affine::Bound::Const`] against [`affine::Bound::Val`]. The reference's own comment for the
/// null case — *"This could happen when the for_op has symbolic loop bounds"*
/// (`CFGSDataflowConditionalTree.hpp:69`) — names that same distinction from the caller's side.
///
/// # ⭐ AND THE AFFINE ARM'S STEP IS ALWAYS ONE
///
/// See [`LoopStep`]: `affine.for`'s step defaults to 1 and [`affine::Op::For`] carries none, so
/// `getStepAsInt()` is 1 for every loop in this island. The `scf` arm reads its step from an operand
/// and can carry another, which is why the division is written out rather than folded away.
///
#[must_use]
pub fn get_dataflow_for_loop_info_if_iv<'a>(
    val: Val,
    scope: &'a [DfirOp],
) -> Option<DataflowForLoopInfo<'a>> {
    // `if (isa<BlockArgument>(val))` and `cast<BlockArgument>(val).getOwner()->getParentOp()`.
    let for_op = owner_of_block_arg(val, scope)?;

    let (iv, lo, hi, step) = match for_op {
        // `if (auto scf_for = llvm::dyn_cast<mlir::scf::ForOp>(for_op))` — TESTED FIRST.
        DfirOp::Scf(scf::Op::For {
            iv, lo, hi, step, ..
        }) => {
            // The three `getDefiningOp<mlir::arith::ConstantIndexOp>()` walks, and the
            // `lb_const && ub_const && step_const` that follows them: each `?` is one null.
            let lo = constant_index(*lo, scope)?;
            let hi = constant_index(*hi, scope)?;
            // ⛔ THE STEP IS CHECKED, NOT JUST READ — see [`LoopStep`] for why a non-positive
            // divisor is declined here rather than divided by.
            let step = LoopStep::checked(constant_index(*step, scope)?)?;
            (iv, LoopBound(lo), LoopBound(hi), step)
        }
        // `else if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(for_op))`.
        DfirOp::Affine(affine::Op::For { iv, lo, hi, .. }) => {
            // `affine_for.hasConstantLowerBound() && affine_for.hasConstantUpperBound()`, then
            // `affine_for.getConstantLowerBound()` / `getConstantUpperBound()`.
            let (affine::Bound::Const(lo), affine::Bound::Const(hi)) = (*lo, *hi) else {
                return None;
            };
            // `auto step = affine_for.getStepAsInt();`
            (iv, LoopBound(lo), LoopBound(hi), LoopStep::ONE)
        }
        // Neither `dyn_cast` succeeds — the block argument belongs to some other region-carrying op.
        DfirOp::Scf(_)
        | DfirOp::Affine(_)
        | DfirOp::Arith(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        // ⭐ A PLAIN ACCESS CARRIES NO REGION EITHER, so it owns no block — same reason as `symbol`
        // below.
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        // ⭐ `symbol.create_symbol` CARRIES NO REGION, so it owns no block and cannot be the parent
        // op [`owner_of_block_arg`] found — this arm exists because the match is total, not because
        // the walk can land on it.
        // ⭐ `uniform.uniformize_regions` DOES OWN A BLOCK PER REGION, and its argument is the unit
        // the region runs on — never an induction variable. Neither `dyn_cast` succeeds, which is
        // this arm's answer and not a gap in it.
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => return None,
    };

    // `&& scf_for.getInductionVar() == val` / `&& affine_for.getInductionVar() == val` — a carried
    // value's region argument is bound by the same block and is NOT the induction variable.
    //
    // ⭐ ONE TEST FOR BOTH ARMS, because it is the last conjunct of both and reads the same field.
    if *iv != val {
        return None;
    }

    Some(DataflowForLoopInfo {
        for_op,
        lo,
        hi,
        step,
        // `auto num_iterations = (ub - lb) / step;`
        iterations: Iterations::between(lo, hi, step),
    })
}

/// `%v.getDefiningOp<mlir::arith::ConstantIndexOp>()` — the literal behind an `index` value, or
/// [`None`] when the value is not an `arith.constant` at all.
///
/// ⭐ [`arith::Op::Constant`] IS `arith::ConstantIndexOp`. The island splits the index constant from
/// the integer one exactly as the reference does, with two ops and two lowerings
/// (`StandardToSentient.cpp:347` and `:358`) — so an `arith.constant 4 : i32` behind a bound is a
/// [`None`] here, which is the `dyn_cast`'s answer too.
///
/// ⚠️ `pub(super)` BECAUSE A SECOND BRIDGE-2 FUNCTION ASKS THE SAME QUESTION.
/// [`get_loop_trip_count`](super::tf_mutable_addr_splitting::get_loop_trip_count) (entry 184) reads
/// an `scf.for`'s three bound operands through exactly this `dyn_cast`, and the classification of the
/// island's other twelve `arith` ops below is the part that must not be written twice.
pub(super) fn constant_index(val: Val, scope: &[DfirOp]) -> Option<i64> {
    match defining_op(val, scope)? {
        DfirOp::Arith(arith::Op::Constant { value, .. }) => Some(*value),
        // ⛔ THE `arith` ARM IS SPELLED OUT, because `arith` is where a rival constant would land:
        // `arith.constant 4 : i32` is an `arith::ConstantIntOp` and NOT this `dyn_cast`'s target, and
        // an eighth `arith` op must be classified here rather than fall through a wildcard.
        DfirOp::Arith(
            arith::Op::ConstantInt { .. }
            | arith::Op::DenseConstant { .. }
            | arith::Op::AddI(_)
            | arith::Op::SubI(_)
            | arith::Op::MulI(_)
            // ⛔ `arith.divsi` IS THE BOUND OP AND STILL NOT A LITERAL. The scheduler spells a trip
            // count `(%hi - %lo) / %step` as a `subi` feeding a `divsi`
            // (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:226-228`), so a
            // loop whose bound is one of those has NO constant bound and this `dyn_cast` is null —
            // which is what makes [`get_dataflow_for_loop_info_if_iv`] decline it. Entry 091's
            // `getForOpBound` is the function that walks the chain instead of declining it.
            | arith::Op::DivSI(_)
            // ⛔ AND `arith.remsi` FOR THE SAME REASON, one step further: it is the other half of an
            // affine `mod` expansion (`AffineApplyExpander::visitModExpr`), so it stands where a
            // literal would and is not one.
            | arith::Op::RemSI(_)
            | arith::Op::Compare { .. }
            // ⛔ `arith.select` IS A BOUND THE VENDOR ACTUALLY WRITES AND STILL NOT A LITERAL.
            // `select_ub` gives an `scf.for` the bound `arith.select %c, %c8, %c4 : index`
            // (`dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_one_dim.mlir:290`) —
            // both arms constant and the answer still null, because `dyn_cast` looks at THIS op.
            // Entry 184's `get_loop_trip_count` is the function that takes the larger arm instead of
            // declining it.
            | arith::Op::Select { .. }
            | arith::Op::Logic { .. }
            // ⛔ A CONVERSION BINDS A VECTOR, so it cannot be an `index` bound at all.
            | arith::Op::Convert { .. },
        ) => None,
        // ⭐ AND THE OTHER DIALECTS BY DIALECT, because no future op of theirs could be an
        // `arith.constant`: a bound defined by a loop result or a memory access is not a literal.
        DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_) => None,
        // ⛔⛔ `symbol` IS THE ONE THAT MATTERS, AND IT IS STILL A [`None`]. A bound defined by
        // `symbol.create_symbol` is a SYMBOLIC bound: `dyn_cast<arith::ConstantIndexOp>` answers null
        // for it, so [`get_dataflow_for_loop_info_if_iv`] declines the loop and reports no trip count
        // — it does NOT substitute a number. The reference's own zero for a symbolic bound lives in a
        // DIFFERENT function and is scoped to the XRF movement that one computes (`getForOpBound`,
        // entry 091,
        // `Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:278-284`);
        // borrowing it here would hand every caller of this one a trip count of `0 - 0`.
        DfirOp::Symbol(_) => None,
        // ⭐ AND `uniform` WITH IT: a `uniform.query_map` result is an `index` whose value the
        // schedule fixes later, so `dyn_cast<arith::ConstantIndexOp>` is null for it too.
        DfirOp::Uniform(_) => None,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::affine::Carried;
    use crate::islands::dataflow_ir::dialects::arith::IntConst;
    use crate::islands::dataflow_ir::dialects::dataflow;

    /// `%result = arith.constant N : index` — an `arith::ConstantIndexOp`.
    fn index_const(result: Val, value: i64) -> DfirOp {
        DfirOp::Arith(arith::Op::Constant { result, value })
    }

    /// `scf.for %iv = %lo to %hi step %step { }`, bounds by value as the dialect states them.
    fn scf_counted(iv: Val, lo: Val, hi: Val, step: Val) -> DfirOp {
        DfirOp::Scf(scf::Op::For {
            iv,
            lo,
            hi,
            step,
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })
    }

    /// The vendor's case-3 loop and the three constants it reads —
    /// `scf.for %arg5 = %c0 to %c4 step %c1` (`simplify-conditional.mlir:319`), with `%c0` at
    /// [`Val(100)`], `%c4` at [`Val(104)`], `%c1` at [`Val(101)`] and `%arg5` at [`Val(105)`].
    fn vendors_scf_case_three() -> Vec<DfirOp> {
        vec![
            index_const(Val(100), 0),
            index_const(Val(101), 1),
            index_const(Val(104), 4),
            scf_counted(Val(105), Val(100), Val(104), Val(101)),
        ]
    }

    /// `affine.for %iv = lo to hi { }` — a plain counted loop with constant bounds.
    fn counted(iv: Val, lo: i64, hi: i64) -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(lo),
            hi: affine::Bound::Const(hi),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })
    }

    /// The vendor's `%37 = affine.for %38 = 0 to 4 iter_args(%39 = %34) -> (index)`.
    fn vendors_carrying_loop() -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv: Val(38),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: vec![Carried {
                init: Val(34),
                arg: Val(39),
                result: Val(37),
            }],
            body: Vec::new(),
            dbg_name: None,
        })
    }

    /// ⭐⭐ THE VENDOR'S OWN FOUR-ITERATION CONDITIONAL LOOP.
    ///
    /// `dcc/test/Transform/CFGSimplificationDataflowLevel/simplify-conditional.mlir:52` is a
    /// `CHECK-SENT-IR` line of the very pass that calls this function:
    ///
    /// ```text
    /// %37 = affine.for %38 = 0 to 4 iter_args(%39 = %34) -> (index) {
    ///   %40 = arith.cmpi eq, %38, %4 : index      // == 0
    ///   %41 = arith.cmpi eq, %38, %6 : index      // == 1
    ///   %42 = arith.cmpi eq, %38, %8 : index      // == 2
    ///   %43 = arith.cmpi eq, %38, %10 : index     // == 3
    /// ```
    ///
    /// Four comparisons of the induction variable against four literals, which is exactly what
    /// `CFGSDataflowConditionalTree` indexes into the array this function's trip count sizes:
    /// `val_array_ = new std::optional<int64_t>[num_iterations]` with
    /// `num_iterations = std::get<4>(...)` (`CFGSDataflowConditionalTree.hpp:74-75`). So the
    /// expectation is the vendor's: **four**.
    #[test]
    fn the_vendors_four_iteration_loop_answers_its_bounds_and_trip_count() {
        let scope = vec![vendors_carrying_loop()];

        let info = get_dataflow_for_loop_info_if_iv(Val(38), &scope)
            .expect("%38 is that loop's induction variable");

        assert_eq!(&scope[0], info.for_op);
        assert_eq!(LoopBound(0), info.lo);
        assert_eq!(LoopBound(4), info.hi);
        assert_eq!(LoopStep::ONE, info.step);
        assert_eq!(4, info.iterations.get());
    }

    /// ⛔ THE CARRIED ARGUMENT SHARES THE BLOCK AND IS STILL NOT THE INDUCTION VARIABLE.
    ///
    /// `%39` in the vendor loop above is a block argument whose owner IS an `affine.for` with two
    /// constant bounds — every test but the last one passes. `getInductionVar() == val` is the
    /// clause that rejects it, and it is what the function's name is about.
    #[test]
    fn the_carried_argument_of_that_same_loop_is_not_its_induction_variable() {
        let scope = vec![vendors_carrying_loop()];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(39), &scope));
    }

    /// The vendor's nest — `program_unit { affine.for 0 to 28 { affine.for 0 to 1 { affine.for
    /// 0 to 3 { .. } } } }` (`simplify-conditional.mlir:35-38`) — reached through two regions.
    #[test]
    fn an_induction_variable_nested_in_a_program_unit_and_two_loops_is_found() {
        let inner = counted(Val(27), 0, 3);
        let scope = vec![DfirOp::Dataflow(dataflow::Op::ProgramUnit {
            units: vec![Val(12)],
            iter_arg: None,
            precision: None,
            body: vec![DfirOp::Affine(affine::Op::For {
                iv: Val(25),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(28),
                carried: Vec::new(),
                body: vec![DfirOp::Affine(affine::Op::For {
                    iv: Val(26),
                    lo: affine::Bound::Const(0),
                    hi: affine::Bound::Const(1),
                    carried: Vec::new(),
                    body: vec![inner],
                    dbg_name: None,
                })],
                dbg_name: None,
            })],
        })];

        let info = get_dataflow_for_loop_info_if_iv(Val(27), &scope)
            .expect("the innermost loop binds %27 three regions down");
        assert_eq!(LoopBound(3), info.hi);
        assert_eq!(3, info.iterations.get());

        // And each enclosing loop answers its own bounds through the same descent.
        assert_eq!(
            28,
            get_dataflow_for_loop_info_if_iv(Val(25), &scope)
                .expect("the outer loop")
                .iterations
                .get()
        );
        assert_eq!(
            1,
            get_dataflow_for_loop_info_if_iv(Val(26), &scope)
                .expect("the middle loop")
                .iterations
                .get()
        );
    }

    /// `isa<BlockArgument>(val)` fails — the reference's `(nullptr, 0, 0, 0, 0)`.
    #[test]
    fn a_value_no_block_binds_is_not_an_induction_variable() {
        let scope = vec![counted(Val(38), 0, 4)];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(4), &scope));
    }

    /// ⛔ THE "SYMBOLIC LOOP BOUNDS" CASE THE CALLER NAMES.
    ///
    /// `hasConstantUpperBound()` is false for `affine.for %i = 0 to %extent`, and the caller's own
    /// comment for the null answer is *"This could happen when the for_op has symbolic loop
    /// bounds"* (`CFGSDataflowConditionalTree.hpp:69`).
    #[test]
    fn a_symbolic_upper_bound_answers_nothing() {
        let scope = vec![DfirOp::Affine(affine::Op::For {
            iv: Val(38),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Val(Val(7)),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(38), &scope));
    }

    /// `hasConstantLowerBound()` is the first of the two tests, and it fails on its own.
    #[test]
    fn a_symbolic_lower_bound_answers_nothing() {
        let scope = vec![DfirOp::Affine(affine::Op::For {
            iv: Val(38),
            lo: affine::Bound::Val(Val(7)),
            hi: affine::Bound::Const(4),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(38), &scope));
    }

    /// A block argument whose owner is not a loop at all — a `program_unit`'s `iter_arg`.
    ///
    /// ⭐ The island's [`block_args`] gives `program_unit` none, so this exercises the
    /// `dyn_cast<AffineForOp>` failure through the ordinary path: the value is bound by no block,
    /// so the walk finds no owner, and the reference's two `dyn_cast`s both fail for it.
    #[test]
    fn a_value_bound_by_no_loop_answers_nothing() {
        let scope = vec![DfirOp::Dataflow(dataflow::Op::ProgramUnit {
            units: vec![Val(12)],
            iter_arg: None,
            precision: None,
            body: vec![counted(Val(38), 0, 4)],
        })];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(13), &scope));
    }

    /// `affine.for %i = 4 to 4` runs zero times, and `(4 - 4) / 1` is zero in both languages.
    #[test]
    fn an_empty_loop_runs_zero_times() {
        let scope = vec![counted(Val(38), 4, 4)];

        let info =
            get_dataflow_for_loop_info_if_iv(Val(38), &scope).expect("still a constant loop");
        assert_eq!(0, info.iterations.get());
    }

    /// ⛔ A NEGATIVE SPAN IS ZERO ITERATIONS, NOT A NEGATIVE COUNT.
    ///
    /// This is the one departure in the port: the reference computes `(ub - lb) / step` as
    /// `int64_t` and hands it to `new std::optional<int64_t>[num_iterations]`
    /// (`CFGSDataflowConditionalTree.hpp:74-75`), so `4 to 0` would ask for an array of −4. The
    /// loop runs zero times; [`Iterations`] says zero.
    #[test]
    fn an_upper_bound_below_the_lower_saturates_at_zero() {
        let scope = vec![counted(Val(38), 4, 0)];

        let info =
            get_dataflow_for_loop_info_if_iv(Val(38), &scope).expect("still a constant loop");
        assert_eq!(LoopBound(4), info.lo);
        assert_eq!(LoopBound(0), info.hi);
        assert_eq!(0, info.iterations.get());
    }

    /// A lower bound below zero spans to the upper — the bounds are signed because MLIR's are.
    #[test]
    fn a_negative_lower_bound_spans_to_the_upper() {
        let scope = vec![counted(Val(38), -2, 2)];

        let info = get_dataflow_for_loop_info_if_iv(Val(38), &scope).expect("a constant loop");
        assert_eq!(LoopBound(-2), info.lo);
        assert_eq!(4, info.iterations.get());
    }

    /// ⭐ THE OP ANSWERED IS THE LOOP THAT BINDS THE VARIABLE, not the first loop in the scope.
    #[test]
    fn the_op_answered_is_the_loop_that_binds_the_variable() {
        let scope = vec![counted(Val(25), 0, 28), counted(Val(38), 0, 4)];

        let info = get_dataflow_for_loop_info_if_iv(Val(38), &scope).expect("the second loop");
        assert_eq!(&scope[1], info.for_op);
        assert_eq!(4, info.iterations.get());
    }

    /// ⭐⭐ THE VENDOR'S `scf` CASE ANSWERS EXACTLY WHAT ITS `affine` CASE DOES.
    ///
    /// Case 3 of `dcc/test/Transform/CFGSimplificationDataflowLevel/simplify-conditional.mlir:241` is
    /// *"Same as 2. but with scf.for instead of affine.for"*, and case 2's loop is the
    /// `0 to 4 iter_args` one above. So "the same" is the expectation: same bounds, same step, same
    /// four iterations — reached through three `arith.constant` walks instead of two bound
    /// attributes.
    #[test]
    fn the_vendors_scf_loop_answers_what_its_affine_twin_does() {
        let scope = vendors_scf_case_three();

        let info = get_dataflow_for_loop_info_if_iv(Val(105), &scope)
            .expect("%arg5 is that loop's induction variable");

        assert_eq!(&scope[3], info.for_op);
        assert_eq!(LoopBound(0), info.lo);
        assert_eq!(LoopBound(4), info.hi);
        assert_eq!(LoopStep::ONE, info.step);
        assert_eq!(4, info.iterations.get());

        // And byte for byte the affine twin's answer.
        let twin = vec![vendors_carrying_loop()];
        let twin = get_dataflow_for_loop_info_if_iv(Val(38), &twin).expect("case 2's loop");
        assert_eq!(twin.lo, info.lo);
        assert_eq!(twin.hi, info.hi);
        assert_eq!(twin.step, info.step);
        assert_eq!(twin.iterations, info.iterations);
    }

    /// ⛔ `getDefiningOp<arith::ConstantIndexOp>()` IS NULL FOR A COMPUTED BOUND.
    ///
    /// `scf.for %iv = %c0 to %sum step %c1` where `%sum` is an `arith.addi` — the `ub_const` half of
    /// `lb_const && ub_const && step_const` fails, and the reference falls through to its null tuple.
    #[test]
    fn an_scf_bound_that_is_computed_answers_nothing() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(101), 1),
            DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                result: Val(104),
                lhs: Val(100),
                rhs: Val(101),
                ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
            })),
            scf_counted(Val(105), Val(100), Val(104), Val(101)),
        ];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(105), &scope));
    }

    /// ⛔ AND FOR A BOUND NOTHING IN SCOPE DEFINES — the `dyn_cast` on a null defining op.
    #[test]
    fn an_scf_bound_with_no_defining_op_answers_nothing() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(101), 1),
            scf_counted(Val(105), Val(100), Val(104), Val(101)),
        ];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(105), &scope));
    }

    /// ⛔ AN `i32` CONSTANT IS NOT AN INDEX CONSTANT.
    ///
    /// `arith::ConstantIntOp` and `arith::ConstantIndexOp` are two op classes with two lowerings
    /// (`StandardToSentient.cpp:347`, `:358`), and the `dyn_cast` here names the index one. See
    /// [`constant_index`].
    #[test]
    fn an_integer_constant_behind_a_bound_is_not_an_index_constant() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(101), 1),
            DfirOp::Arith(arith::Op::ConstantInt {
                result: Val(104),
                value: IntConst::Int { value: 4, bits: 32 },
            }),
            scf_counted(Val(105), Val(100), Val(104), Val(101)),
        ];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(105), &scope));
    }

    /// ⭐ A STEP OF TWO HALVES THE TRIP COUNT — the one thing the `scf` arm can express that the
    /// affine arm cannot, and the reason `(ub - lb) / step` is written out.
    #[test]
    fn an_scf_loop_stepping_by_two_runs_half_as_often() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(102), 2),
            index_const(Val(104), 8),
            scf_counted(Val(105), Val(100), Val(104), Val(102)),
        ];

        let info = get_dataflow_for_loop_info_if_iv(Val(105), &scope).expect("a constant loop");
        assert_eq!(LoopStep::checked(2), Some(info.step));
        assert_eq!(4, info.iterations.get());
    }

    /// ⭐ AND THE DIVISION TRUNCATES, as `int64_t / int64_t` does: `0 to 7 step 2` is four trips
    /// (0, 2, 4, 6) and `(7 - 0) / 2` is three. The port keeps the reference's arithmetic rather than
    /// correcting it, because the array the caller sizes with it is the reference's too
    /// (`CFGSDataflowConditionalTree.hpp:74-75`).
    #[test]
    fn the_trip_count_truncates_as_the_references_division_does() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(102), 2),
            index_const(Val(104), 7),
            scf_counted(Val(105), Val(100), Val(104), Val(102)),
        ];

        let info = get_dataflow_for_loop_info_if_iv(Val(105), &scope).expect("a constant loop");
        assert_eq!(3, info.iterations.get());
    }

    /// ⛔ A ZERO STEP IS DECLINED RATHER THAN DIVIDED BY — see [`LoopStep`]. The reference's
    /// `(ub - lb) / 0` is undefined behaviour; there is no non-positive step to make it reachable.
    #[test]
    fn a_zero_scf_step_answers_nothing() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(103), 0),
            index_const(Val(104), 4),
            scf_counted(Val(105), Val(100), Val(104), Val(103)),
        ];

        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(105), &scope));
        assert_eq!(None, LoopStep::checked(0));
        assert_eq!(None, LoopStep::checked(-1));
    }

    /// ⛔ AND THE `scf` LOOP'S CARRIED ARGUMENT IS NOT ITS INDUCTION VARIABLE EITHER.
    ///
    /// The vendor's `%50 = scf.for %51 = %47 to %11 step %6 iter_args(%52 = %49) -> (index)`
    /// (`simplify-conditional.mlir:64`): `%52` shares the block with `%51` and fails the last
    /// conjunct.
    #[test]
    fn the_carried_argument_of_an_scf_loop_is_not_its_induction_variable() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(101), 1),
            index_const(Val(104), 4),
            DfirOp::Scf(scf::Op::For {
                iv: Val(105),
                lo: Val(100),
                hi: Val(104),
                step: Val(101),
                carried: vec![Carried {
                    init: Val(100),
                    arg: Val(106),
                    result: Val(107),
                }],
                body: Vec::new(),
                dbg_name: None,
            }),
        ];

        assert!(get_dataflow_for_loop_info_if_iv(Val(105), &scope).is_some());
        assert_eq!(None, get_dataflow_for_loop_info_if_iv(Val(106), &scope));
    }

    /// ⭐ AN `scf.for` NESTED IN AN `affine.for` IS REACHED, AND EACH ANSWERS ITS OWN LOOP — the
    /// mixed nest the vendor's case 3 actually is (its `scf.for` sits inside `affine.for`s).
    #[test]
    fn a_mixed_nest_answers_each_loop_from_its_own_variable() {
        let scope = vec![
            index_const(Val(100), 0),
            index_const(Val(101), 1),
            index_const(Val(104), 4),
            DfirOp::Affine(affine::Op::For {
                iv: Val(25),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(28),
                carried: Vec::new(),
                body: vec![scf_counted(Val(105), Val(100), Val(104), Val(101))],
                dbg_name: None,
            }),
        ];

        assert_eq!(
            28,
            get_dataflow_for_loop_info_if_iv(Val(25), &scope)
                .expect("the affine loop")
                .iterations
                .get()
        );
        assert_eq!(
            4,
            get_dataflow_for_loop_info_if_iv(Val(105), &scope)
                .expect("the scf loop inside it")
                .iterations
                .get()
        );
    }

    /// 🎯 264/384 — ONE MORE ITER_ARG ON AN `scf.for`, AND THE FILL THAT MAKES IT A ONE.
    /// `arith::ConstantIndexOp::create(builder, loc, 1)` is the scf arm's init (`Utils.cpp:65`, where
    /// the affine arm writes 0 at `:44`); `copyLoopBody` gives the added arg a yield of ITSELF
    /// (`dcc/src/Utils/Utils.cpp:374-381`) and the old result is replaced positionally (`:81-83`).
    #[test]
    fn an_scf_loop_gains_an_iter_arg_initialised_to_one() {
        let source = DfirOp::Scf(scf::Op::For {
            iv: Val(110),
            lo: Val(101),
            hi: Val(102),
            step: Val(103),
            carried: vec![affine::Carried {
                init: Val(104),
                arg: Val(111),
                result: Val(112),
            }],
            body: vec![DfirOp::Scf(scf::Op::Yield {
                operands: vec![Val(111)],
            })],
            dbg_name: Some("mb-chunk/5".to_owned()),
        });
        let mut vals = Values::default();

        let grown = create_for_op_with_additional_return_value(
            &mut vals,
            &CountedLoop::of(&source).expect("an scf.for is one of the two arms"),
            1,
            false,
        );

        assert_eq!(
            vec![DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 1,
            })],
            grown.consts
        );
        assert_eq!(vec![(Val(112), Val(1))], grown.replacements);
        assert_eq!(
            DfirOp::Scf(scf::Op::For {
                iv: Val(3),
                lo: Val(101),
                hi: Val(102),
                step: Val(103),
                carried: vec![
                    affine::Carried {
                        init: Val(104),
                        arg: Val(4),
                        result: Val(1),
                    },
                    affine::Carried {
                        init: Val(0),
                        arg: Val(5),
                        result: Val(2),
                    },
                ],
                body: vec![DfirOp::Scf(scf::Op::Yield {
                    operands: vec![Val(4), Val(5)],
                })],
                dbg_name: Some("mb-chunk/5".to_owned()),
            }),
            grown.op
        );
        // ⛔ AND THE MAPPING IS READABLE BECAUSE `delete_op` WAS FALSE (`Utils.hpp:41-44`).
        assert_eq!(
            Some(Val(3)),
            grown
                .ir_map
                .expect("delete_op was false")
                .lookup(Val(110))
        );
    }
}

/// WHICH BOUNDS THE LOOP HAS — the two `dyn_cast`s of `Utils.cpp:36` and `:58`, and the one place the
/// two arms of entry 264 differ apart from the fill constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountedBounds {
    /// `affine::AffineForOp`: `getLowerBoundOperands()`/`getLowerBoundMap()` and the upper pair,
    /// passed through unchanged.
    Affine { lo: affine::Bound, hi: affine::Bound },
    /// `scf::ForOp`: `getLowerBound()`, `getUpperBound()`, `getStep()`, passed through unchanged.
    Scf { lo: Val, hi: Val, step: Val },
}

/// THE LOOP `createForOpWithAdditionalReturnValue` ADMITS.
///
/// ⚠️ THIS TYPE IS WHAT `DT_CHECK(ret_op != nullptr)` (`:88`) GUARDS THERE. A `loop_op` that is
/// neither `affine.for` nor `scf.for` leaves `ret_op` null and the reference dereferences it a line
/// EARLIER (`ret_op->getResult(i)`, `:82`); here [`CountedLoop::of`] declines instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountedLoop<'a> {
    /// Which dialect's `for` it is, with its bounds.
    pub bounds: CountedBounds,
    /// `getInductionVar()` — the body's first block argument.
    pub iv: Val,
    /// `getInits()` with the region iter args and results they bind.
    pub carried: &'a [affine::Carried],
    /// `getBody()->getOperations()`, terminator included.
    pub body: &'a [DfirOp],
    /// `dataflow::getDbgNameAttr(loop_op)`.
    pub dbg_name: Option<&'a str>,
}

impl<'a> CountedLoop<'a> {
    /// `dyn_cast<affine::AffineForOp>(loop_op)`, then `dyn_cast<scf::ForOp>(loop_op)`.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<CountedLoop<'a>> {
        match op {
            DfirOp::Affine(affine::Op::For {
                iv,
                lo,
                hi,
                carried,
                body,
                dbg_name,
            }) => Some(CountedLoop {
                bounds: CountedBounds::Affine { lo: *lo, hi: *hi },
                iv: *iv,
                carried,
                body,
                dbg_name: dbg_name.as_deref(),
            }),
            DfirOp::Scf(scf::Op::For {
                iv,
                lo,
                hi,
                step,
                carried,
                body,
                dbg_name,
            }) => Some(CountedLoop {
                bounds: CountedBounds::Scf {
                    lo: *lo,
                    hi: *hi,
                    step: *step,
                },
                iv: *iv,
                carried,
                body,
                dbg_name: dbg_name.as_deref(),
            }),
            // ⭐ NO WILDCARD: a tenth region-carrying op is not a counted loop until this says so.
            DfirOp::Affine(_)
            | DfirOp::Scf(_)
            | DfirOp::Arith(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
        }
    }
}

/// THE LOOP ENTRY 264 PUTS IN PLACE OF ITS INPUT, with what the caller has to do around it.
///
/// ⚠️ NOT [`PartialEq`]: [`ValueMapping`] deliberately is not, because two mappings with the same
/// answers hold different pairs once one has shadowed an entry.
#[derive(Debug, Clone)]
pub struct ForOpWithExtraResults {
    /// The `arith::ConstantIndexOp`s that initialise the added iter_args, in creation order — they go
    /// BEFORE the loop, where `OpBuilder builder(loop_op)` puts them (`:33`, `:43-46`).
    pub consts: Vec<DfirOp>,
    /// The new `affine.for` or `scf.for`, `n_values` results and iter_args wider than the old one.
    pub op: DfirOp,
    /// `loop_op->getResult(i).replaceAllUsesWith(ret_op->getResult(i))`, old → new, positionally over
    /// the OLD result count (`:81-83`).
    pub replacements: Vec<(Val, Val)>,
    /// `IRMapping& ir_map` over the old body's values.
    ///
    /// ⛔ [`None`] WHERE THE REFERENCE ERASED THE OLD LOOP — *"Only valid for use if delete_op was
    /// false"* (`Utils.hpp:41-44`), and two of the four callers pass `false` precisely to read it
    /// (`TransformPagedMemViewImpl.cpp:452-453`, `AgenToSentient/Helper.cpp:1067-1068`).
    pub ir_map: Option<ValueMapping>,
}

/// Replaces: e264_createForOpWithAdditionalReturnValue
///
/// **264/384** `createForOpWithAdditionalReturnValue` —
/// `dcc/src/Transform/Dataflow/Utils.cpp:28` (66L).
///
/// ⛔ THE FILL IS 0 IN THE AFFINE ARM AND 1 IN THE SCF ARM (`:44` against `:65`) — one function, two
/// constants. ⚠️ `getStepAsInt()` is passed through and this island's `affine.for` carries no step
/// ([`affine::Op::For`]), which makes that pass-through the identity.
#[must_use]
pub fn create_for_op_with_additional_return_value(
    vals: &mut Values,
    loop_op: &CountedLoop<'_>,
    n_values: usize,
    delete_op: bool,
) -> ForOpWithExtraResults {
    // `arith::ConstantIndexOp::create(builder, loop_op->getLoc(), 0)` in the affine arm, `.., 1)` in
    // the scf arm — ⛔ THE ONE VALUE THAT IS NOT THE SAME IN THE TWO OTHERWISE IDENTICAL BRANCHES.
    let fill = match loop_op.bounds {
        CountedBounds::Affine { .. } => 0,
        CountedBounds::Scf { .. } => 1,
    };

    // `for (auto operand : getInits()) iter_args.push_back(operand);` then the new constants, which
    // are created BEFORE the loop and so take their values first.
    let mut consts: Vec<DfirOp> = Vec::with_capacity(n_values);
    let mut inits: Vec<Val> = loop_op.carried.iter().map(|source| source.init).collect();
    for _ in 0..n_values {
        let result = vals.mint();
        consts.push(DfirOp::Arith(arith::Op::Constant {
            result,
            value: fill,
        }));
        inits.push(result);
    }

    // `AffineForOp::create(..)` / `ForOp::create(..)`: the results first, then the induction variable,
    // then the region's iter args — the order MLIR defines and prints them in.
    let results: Vec<Val> = inits.iter().map(|_| vals.mint()).collect();
    let iv = vals.mint();
    let carried: Vec<affine::Carried> = inits
        .iter()
        .zip(&results)
        .map(|(init, result)| affine::Carried {
            init: *init,
            arg: vals.mint(),
            result: *result,
        })
        .collect();

    // `ir_map = copyLoopBody(from, new_loop, builder, n_values)` — `dcc/src/Utils/Utils.cpp:358`, a
    // template that is not a campaign unit, so it is inlined here.
    //
    // `bv_map.map(getBody()->getArguments(), to_loop.getBody()->getArguments())` — ⭐ THE ZIP
    // TRUNCATES: the new body has `n_values` MORE arguments and `llvm::zip` stops at the shorter
    // range, so the added ones are deliberately left unmapped.
    let mut ir_map = ValueMapping::new();
    ir_map.map(loop_op.iv, iv);
    for (source, new) in loop_op.carried.iter().zip(&carried) {
        ir_map.map(source.arg, new.arg);
    }

    // `for (auto &it : from_loop.getBody()->getOperations()) builder.clone(it, bv_map);` — the
    // terminator with them, which is how the new loop gets one at all.
    let mut body = vals.clone_ops(loop_op.body, &mut ir_map);

    // `for (int i = n_values; i > 0; i--) yield_args.push_back(getRegionIterArgs()[n_reg_iter - i]);`
    // — the LAST `n_values` iter args, ascending: each added value yields ITSELF, so the loop carries
    // the constant through untouched until a caller rewrites the yield.
    let added: Vec<Val> = carried[carried.len() - n_values..]
        .iter()
        .map(|new| new.arg)
        .collect();

    // `yield_op->setOperands(yield_args)` over `to_loop.getBody()->getTerminator()`.
    match body.last_mut() {
        Some(
            DfirOp::Affine(affine::Op::Yield { operands })
            | DfirOp::Scf(scf::Op::Yield { operands }),
        ) => operands.extend(added),
        // ⚠️ `getTerminator()` IS A NULL DEREFERENCE THERE FOR A BODY THAT HAS NONE, which this
        // island can hold. The terminator MLIR's verifier requires is emitted instead of a stop, as
        // `transform_scf_to_affine_loop` does for the same case.
        _ => body.push(match loop_op.bounds {
            CountedBounds::Affine { .. } => DfirOp::Affine(affine::Op::Yield { operands: added }),
            CountedBounds::Scf { .. } => DfirOp::Scf(scf::Op::Yield { operands: added }),
        }),
    }

    // `if (auto dbg_name_attr = getDbgNameAttr(loop_op)) setDbgNameAttr(new_loop, dbg_name_attr);`,
    // and with it the `for (auto attr : loop_op->getAttrs())` loop at `:85-90`: `dbgName` is the only
    // attribute an island `for` carries, and `operandSegmentSizes` — the one that loop excludes — is
    // MLIR's own operand bookkeeping, which a typed field cannot have.
    let dbg_name = loop_op.dbg_name.map(str::to_owned);
    let op = match loop_op.bounds {
        CountedBounds::Affine { lo, hi } => DfirOp::Affine(affine::Op::For {
            iv,
            lo,
            hi,
            carried,
            body,
            dbg_name,
        }),
        CountedBounds::Scf { lo, hi, step } => DfirOp::Scf(scf::Op::For {
            iv,
            lo,
            hi,
            step,
            carried,
            body,
            dbg_name,
        }),
    };

    ForOpWithExtraResults {
        consts,
        op,
        // `for (int i = 0; i < loop_op->getNumResults(); i++)` — the OLD count, so the zip ends there.
        replacements: loop_op
            .carried
            .iter()
            .map(|source| source.result)
            .zip(results)
            .collect(),
        // `if (delete_op) loop_op->erase();` — the erasure is the caller's, and it is what invalidates
        // the mapping.
        ir_map: if delete_op { None } else { Some(ir_map) },
    }
}
