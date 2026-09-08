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

//! `TransformLoopToLegalizeForSentientLowering.cpp` — 6 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e117_transformSCFToAffineLoop` | 117/384 | 44 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102` |
//! | `e194_analyzeLoop` | 194/384 | 127 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268` |
//! | `e195_transformLoop` | 195/384 | 38 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399` |
//! | `e257_analyzeAndTransform` | 257/384 | 3 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441` |
//! | `e292_transformSCFLoopWithNonConstantUpperBound` | 292/384 | 94 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169` |
//! | `e293_runOn` | 293/384 | 5 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447` |

use super::tf_loop_unroll_for_shuffle_op::TripCount;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, agen, arith, dataflow, defining_op, scf, symbol, uses,
};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::dataflow_ir::{ValueMapping, Values};

/// AN `scf.for` AS THIS REWRITE READS IT — the `dyn_cast`, and only the fields it touches.
///
/// ⛔ NO BOUNDS. `transformSCFToAffineLoop` never asks the loop what its bounds are: it takes
/// `lbound`, `ubound` and `step` as parameters ([`StaticBounds`]) because the whole point of the pass
/// is that the `scf.for`'s upper bound is NOT constant — an `arith.select` between two chunk widths
/// (`scf_loop_with_result.mlir:128`) — and the caller has already resolved it per branch. Reading
/// the loop's own `hi` here would read the `select`, which is exactly the value being eliminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScfForOp<'a> {
    /// `scf_for.getInductionVar()`.
    pub iv: Val,
    /// `scf_for.getInits()` / `getRegionIterArgs()` / the loop's results — one record per carried
    /// value, which is the same [`affine::Carried`] the `affine.for` will hold.
    pub carried: &'a [affine::Carried],
    /// The region's single block, terminator included.
    pub body: &'a [DfirOp],
    /// `getDbgNameAttr(scf_for)`.
    pub dbg_name: Option<&'a str>,
}

impl<'a> ScfForOp<'a> {
    /// `dyn_cast<scf::ForOp>` — and nothing more.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<ScfForOp<'a>> {
        let DfirOp::Scf(scf::Op::For {
            iv,
            carried,
            body,
            dbg_name,
            ..
        }) = op
        else {
            return None;
        };
        Some(ScfForOp {
            iv: *iv,
            carried,
            body,
            dbg_name: dbg_name.as_deref(),
        })
    }

    /// THE BODY WITHOUT ITS TERMINATOR, AND THE TERMINATOR'S OPERANDS — `Block::without_terminator()`
    /// and `Block::getTerminator()`, which this rewrite uses in that order.
    ///
    /// ⭐ A TRAILING [`scf::Op::Yield`] IS THE TERMINATOR AND THERE IS NO OTHER KIND HERE. An
    /// `scf.for` carrying nothing has an IMPLICIT `scf.yield` in MLIR and no `Op::Yield` in this
    /// island, so a body that ends in something else is a loop that carries nothing — `None`
    /// operands, and every op is cloned.
    #[must_use]
    fn split_terminator(self) -> (&'a [DfirOp], Option<&'a [Val]>) {
        match self.body.split_last() {
            Some((DfirOp::Scf(scf::Op::Yield { operands }), rest)) => (rest, Some(operands)),
            _ => (self.body, None),
        }
    }
}

/// THE BOUNDS AND STEP THE CALLER RESOLVED — `int lbound, int ubound, int step`.
///
/// ⛔ NAMED FIELDS BECAUSE THREE POSITIONAL `int`s TRANSPOSE SILENTLY. `transformSCFToAffineLoop(builder,
/// scf_for, 0, then_ub, 1, affine_for)` (`:207-209`) is three integers in a row and getting them out of
/// order builds a loop that counts to the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticBounds {
    /// `lbound` — must be 0 for the rewrite to fire.
    pub lbound: i64,
    /// `ubound` — the resolved trip bound of this branch.
    pub ubound: i64,
    /// `step` — must be 1 for the rewrite to fire.
    pub step: i64,
}

/// `LogicalResult`, WITH THE REASON THE FAILURE CARRIES IMPLICITLY.
///
/// ⛔ NOT A `Result`, AND NOT A REFUSAL. `LogicalResult::failure()` here is *"we currently support
/// lbound being 0, step being 1"* — a statement about the loop, which the caller answers by leaving
/// the `scf.for` alone and reporting *"Unable to transform SCF loop into Affine loop"* (`:210`, and
/// `:230` for the else arm). The
/// bounds that failed are named because a caller that has to explain itself should not have to
/// re-derive them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopLegalization {
    /// `LogicalResult::success()`, with the `affine.for` the reference returned through
    /// `affine::AffineForOp& affine_for`.
    Transformed(DfirOp),
    /// `LogicalResult::failure()` — the lower bound is not 0, or the step is not 1.
    UnsupportedBounds {
        /// The lower bound as passed.
        lbound: i64,
        /// The step as passed.
        step: i64,
    },
}

/// Replaces: e117_transformSCFToAffineLoop
///
/// **117/384** `TransformLoopToLegalizeForSentientLowering::transformSCFToAffineLoop` —
/// `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102` (44L).
///
/// ```cpp
/// // This method aims at creating an Affine For loop with bounds and step sizes
/// // being the parameters valued passed as an input, and the body of affine
/// // for-op is constructed from the scf_for.
/// LogicalResult
/// TransformLoopToLegalizeForSentientLowering::transformSCFToAffineLoop(
///     OpBuilder builder, scf::ForOp scf_for, int lbound, int ubound, int step,
///     affine::AffineForOp& affine_for) {
///   // we currently support lbound being 0, step being 1, non-constant ubound.
///   if (lbound == 0 && step == 1) {
///     affine_for = affine::AffineForOp::create(builder, builder.getUnknownLoc(),
///                                              0, ubound, 1, scf_for.getInits());
///
///     if (auto dbg_name_attr = getDbgNameAttr(scf_for))
///       setDbgNameAttr(affine_for, dbg_name_attr);
///
///     // Map induction var, region iterator arguments to affine loop variables.
///     IRMapping bv_map;
///     bv_map.map(scf_for.getInductionVar(), affine_for.getInductionVar());
///     for (unsigned i = 0; i < scf_for.getNumRegionIterArgs(); i++) {
///       bv_map.map(scf_for.getRegionIterArgs()[i],
///                  affine_for.getRegionIterArgs()[i]);
///     }
///
///     // Set insertion point to the beginning of the loop.
///     builder.setInsertionPointToStart(&affine_for.getRegion().front());
///
///     // Clone each operation within the body except the terminator.
///     for (auto& op : scf_for.getRegion().front().without_terminator()) {
///       builder.clone(op, bv_map);
///     }
///
///     // affine_for already has an implicit affine.yield. We need to update it if
///     // there are results.
///     if (scf_for.getNumResults() > 0) {
///       // Has results, need to update the yield with mapped operands
///       auto* scf_yield = scf_for.getRegion().front().getTerminator();
///       SmallVector<Value> mapped_operands;
///       for (auto operand : scf_yield->getOperands()) {
///         mapped_operands.push_back(bv_map.lookup(operand));
///       }
///
///       builder.setInsertionPointToEnd(&affine_for.getRegion().front());
///       builder.create<affine::AffineYieldOp>(affine_for->getLoc(),
///                                             mapped_operands);
///     }
///
///     return LogicalResult::success();
///   }
///
///   return LogicalResult::failure();
/// }
/// ```
///
/// # ⭐⭐ WHY AN `scf.for` MUST BECOME AN `affine.for` AT ALL
///
/// Everything downstream of this pass is affine: `AccessDetailsAffine` walks `affine.for`s to derive a
/// transfer's coefficients, and the Sentient loop lowering counts `affine.for` nesting against
/// [`crate::arch::Arch::MAX_NESTED_LOOPS`]. An `scf.for` reaching the lowering is not a legal input,
/// which is what this pass's name says. The reason one is there in the first place is a bound that is
/// not a constant — `%10 = arith.select %9, %c16, %c32` (`scf_loop_with_result.mlir:128`) — and
/// [`crate::islands::dataflow_ir::dialects::affine::Bound`] has no variant for a computed value
/// because `affine.for` has no way to take one.
///
/// # ⭐ `0` AND `1` ARE PASSED LITERALLY, NOT `lbound` AND `step`
///
/// `AffineForOp::create(builder, loc, 0, ubound, 1, inits)` — the guarded values are the only ones the
/// build could use, and passing them again would suggest otherwise. This island's `affine.for` has no
/// step field at all (a step of 1 is what `affine.for %i = 0 to N` means), so the `1` is discharged by
/// the type.
///
/// # ⛔ THE INITS PASS THROUGH UNCHANGED, AND THAT IS LOAD-BEARING
///
/// `scf_for.getInits()` is evaluated OUTSIDE the loop, so those values are not in the mapping and must
/// not be renumbered — the vendor's expectation shows the same `%20` initialising both branches'
/// copies (`scf_loop_with_result.mlir:41` and `:64`). Only the induction variable, the region
/// arguments and whatever the body itself defines get new names.
///
/// # ⛔ THE MAPPING IS SEEDED BEFORE THE CLONE, WHICH IS WHY THE ORDER HERE IS FIXED
///
/// Mint the results, then the induction variable, then the region arguments — MLIR's own definition
/// order, and therefore its print order — seed the mapping with `(source iv, new iv)` and each
/// `(source arg, new arg)`, and only then clone. A body op reads the induction variable, so a clone
/// that ran first would carry the old name.
///
/// # ⚠️ `bv_map.lookup` ASSERTS; THIS PORT PASSES THE VALUE THROUGH
///
/// `IRMapping::lookup` on an unmapped value returns null and MLIR then fails a verifier;
/// [`ValueMapping::lookup_or_default`] returns the value itself. The two agree on every yield operand
/// the body defines, and where the yield forwards a value from OUTSIDE the loop — a legal `scf.for`
/// (`%c2048` yielded straight through) — passing it through is the only answer that is not a stop.
/// This crate never asserts, so it is the one taken.
#[must_use]
pub fn transform_scf_to_affine_loop(
    vals: &mut Values,
    scf_for: &ScfForOp<'_>,
    bounds: StaticBounds,
) -> LoopLegalization {
    // `if (lbound == 0 && step == 1)` — *"we currently support lbound being 0, step being 1,
    // non-constant ubound."*
    if bounds.lbound != 0 || bounds.step != 1 {
        // `return LogicalResult::failure();`
        return LoopLegalization::UnsupportedBounds {
            lbound: bounds.lbound,
            step: bounds.step,
        };
    }

    // `affine::AffineForOp::create(builder, loc, 0, ubound, 1, scf_for.getInits())` — the results
    // first, then the region's arguments, which is the order MLIR defines and prints them in.
    let results: Vec<Val> = scf_for.carried.iter().map(|_| vals.mint()).collect();
    let iv = vals.mint();
    let carried: Vec<affine::Carried> = scf_for
        .carried
        .iter()
        .zip(results)
        .map(|(source, result)| affine::Carried {
            // ⛔ THE INIT IS THE SOURCE LOOP'S OWN, UNTOUCHED.
            init: source.init,
            arg: vals.mint(),
            result,
        })
        .collect();

    // `bv_map.map(getInductionVar(), ..)` then each region iter arg, positionally.
    let mut bv_map = ValueMapping::new();
    bv_map.map(scf_for.iv, iv);
    for (source, new) in scf_for.carried.iter().zip(&carried) {
        bv_map.map(source.arg, new.arg);
    }

    // `for (auto& op : ..front().without_terminator()) builder.clone(op, bv_map);`
    let (to_clone, yielded) = scf_for.split_terminator();
    let mut body = vals.clone_ops(to_clone, &mut bv_map);

    // `if (scf_for.getNumResults() > 0)` — the results are the carried values, so this is the same
    // test. *"affine_for already has an implicit affine.yield. We need to update it if there are
    // results."*
    if !carried.is_empty() {
        body.push(DfirOp::Affine(affine::Op::Yield {
            // ⚠️ `None` HERE IS A BODY WITH NO TERMINATOR, which `scf.for`'s verifier does not admit
            // and this island cannot check. It yields nothing, and the loop it produces says so
            // rather than stopping — see [`ScfForOp::split_terminator`].
            operands: yielded
                .unwrap_or_default()
                .iter()
                .map(|operand| bv_map.lookup_or_default(*operand))
                .collect(),
        }));
    }

    // `return LogicalResult::success();`, with `affine_for` assigned.
    LoopLegalization::Transformed(DfirOp::Affine(affine::Op::For {
        iv,
        lo: affine::Bound::Const(0),
        hi: affine::Bound::Const(bounds.ubound),
        carried,
        body,
        // `if (auto dbg_name_attr = getDbgNameAttr(scf_for)) setDbgNameAttr(affine_for, ..)` — the
        // name survives the rewrite, and `scf_loop_with_result.mlir:61` checks that it does.
        dbg_name: scf_for.dbg_name.map(str::to_owned),
    }))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 194/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH LOOP FORMS `analyzeLoop` RECOGNISES — its three `dyn_cast`s (`:273`, `:281`, `:310`).
///
/// ⛔⛔ THREE VARIANTS BECAUSE THE FOURTH ANSWER IS `llvm_unreachable("Unknown loop operation")`
/// (`:315`). The parameter's type is `LoopLikeOpInterface`, which `scf.while`, `scf.parallel` and
/// `affine.parallel` also implement, so the reference ABORTS on a loop it does not know. Making the
/// three a closed set moves that from a run-time abort to a place where it cannot be constructed:
/// [`Self::of`] answers `None`, and a walk that skips the op leaves it exactly as the abort's premise
/// says it should never have been. This crate never asserts, and a stop here would pre-empt dbo-opt —
/// the only oracle (`crates/targets/spyre/tests/dfir_never_runtime_refuses.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopUnderAnalysis<'a> {
    /// `dyn_cast<affine::AffineForOp>(loop.getOperation())`.
    AffineFor {
        /// `affine_for.getInductionVar()`.
        iv: Val,
        /// The lower bound map.
        lo: affine::Bound,
        /// The upper bound map.
        hi: affine::Bound,
        /// The region's single block.
        body: &'a [DfirOp],
    },
    /// `dyn_cast<scf::ForOp>(loop.getOperation())`.
    ScfFor {
        /// `scf_for.getInductionVar()`.
        iv: Val,
        /// `getLowerBound()`, an operand.
        lo: Val,
        /// `getUpperBound()`, an operand.
        hi: Val,
        /// `getStep()`, an operand.
        step: Val,
        /// The region's single block.
        body: &'a [DfirOp],
    },
    /// `dyn_cast<sentient::ForOp>(loop.getOperation())` — already lowered, so nothing to legalize.
    ///
    /// ⚠️ [`Self::of`] NEVER PRODUCES THIS ONE, and that is a fact about the two islands rather than
    /// an omission: a `sentient.for` is a [`crate::islands::sentient::dialects::Op`], not a
    /// [`DfirOp`], so a DataflowIR program cannot hold one. The variant is here because the answer
    /// for it — `kNone` — is part of what entry 194 says, and a caller holding a mixed module can
    /// state it.
    SentientFor,
}

impl<'a> LoopUnderAnalysis<'a> {
    /// The three `dyn_cast`s, in the reference's own order.
    ///
    /// `None` is `llvm_unreachable("Unknown loop operation")` — see the type's note.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<LoopUnderAnalysis<'a>> {
        match op {
            DfirOp::Affine(affine::Op::For {
                iv, lo, hi, body, ..
            }) => Some(LoopUnderAnalysis::AffineFor {
                iv: *iv,
                lo: *lo,
                hi: *hi,
                body,
            }),
            DfirOp::Scf(scf::Op::For {
                iv,
                lo,
                hi,
                step,
                body,
                ..
            }) => Some(LoopUnderAnalysis::ScfFor {
                iv: *iv,
                lo: *lo,
                hi: *hi,
                step: *step,
                body,
            }),
            _ => None,
        }
    }
}

/// `enum TransformType { kNone, KUnroll, KSplitParent }` (`:53`).
///
/// ⭐ NOT A REFUSAL IN ANY VARIANT. This is a decision about what to DO to a loop, and
/// [`Self::None`] is *"if no action required, return"* (`:401-404`) — the answer for the vast
/// majority of loops in a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopTransform {
    /// `kNone` — leave the loop alone.
    None,
    /// `KUnroll` — the loop indexes an LRF register file, which the Sentient ISA cannot address
    /// dynamically, so the iterations must become straight-line code.
    Unroll,
    /// `KSplitParent` — the loop's bound is not a constant, so the CONDITIONAL that computes it has
    /// to be lifted out into an `scf.if` with one affine loop per branch (entry 292).
    SplitParent,
}

/// `is_any_of(curr_unit_, L3LU, L3SU, LXLU, LXSU, L0LU, L0SU)` — the six load and store halves.
///
/// ⭐ THE UNITS WHOSE LOOPS `AgenToSentient` HANDLES NATIVELY, which is the reference's own reason:
/// *"Allow for symbolic upper bounds in case of load/store units since AgenToSentient is already
/// enhanced to support scf.for loops natively"* (`:299-300`).
const LOAD_AND_STORE_UNITS: [GenericComp; 6] = [
    GenericComp::L3lu,
    GenericComp::L3su,
    GenericComp::Lxlu,
    GenericComp::Lxsu,
    GenericComp::L0lu,
    GenericComp::L0su,
];

/// `is_any_of(curr_unit_, L3LU, L3SU)` — the two halves whose parent is NOT split.
const L3_HALVES: [GenericComp; 2] = [GenericComp::L3lu, GenericComp::L3su];

/// `is_any_of(curr_unit_, PT, SFP, PE)` — the three units that own register files.
const REGISTER_FILE_UNITS: [GenericComp; 3] = [GenericComp::Pt, GenericComp::Sfp, GenericComp::Pe];

/// `isa<agen::VectorLoadOp, agen::VectorStoreOp, agen::CompositeLoadOp, agen::CompositeStoreOp,
/// agen::CompositeLoadAndStoreOp>(use)` (`:329-331`).
///
/// ⚠️ THREE OF THE FIVE, for the reason entry 141 already records: the island declares
/// `vector_load`, `vector_store` and `composite_load_and_store` and not `composite_load` or
/// `composite_store`, and the brief's rule to grow the island is about a function's *input*
/// (`AGENT-BRIEF.md:57`) — this predicate's input is any op at all. See
/// [`crate::bridges::dataflow_ir_to_sentient::tf_unit_filtering::is_data_transfer`], which lists the
/// same absences.
///
/// ⛔ NO WILDCARD. A new island op has to state whether the induction variable reaching it makes the
/// loop a candidate; falling through to `false` would silently stop a loop being unrolled.
fn is_memory_op(op: &DfirOp) -> bool {
    match op {
        DfirOp::Agen(
            agen::Op::VectorLoad { .. }
            | agen::Op::VectorStore { .. }
            | agen::Op::CompositeLoadAndStore(_),
        ) => true,
        // `agen.yield` is a terminator, and no arm of the `isa<>` list names anything else — the
        // symbolic pair included: it is in entry 141's nineteen classes but NOT in these five, so an
        // induction variable reaching a symbolic access does not make its loop a candidate.
        DfirOp::Agen(
            agen::Op::Yield
            | agen::Op::SymbolicVectorLoad { .. }
            | agen::Op::SymbolicVectorStore { .. }
            // ⛔ THE INTERLEAVE IS NOT ONE EITHER, and it is the one to say so about: it CONTAINS
            // composite transfers, and the induction variable reaching it reaches them — the ops
            // inside its region answer for themselves.
            | agen::Op::CompositeMemoryInterleave { .. }
            | agen::Op::SetTransferMaskState { .. },
        )
        | DfirOp::Arith(_)
        | DfirOp::Scf(_)
        | DfirOp::Affine(_)
        | DfirOp::Dataflow(_)
        // ⛔ `vector.load`/`vector.store` ARE NOT IN THE FIVE-CLASS `isa<>` LIST, AND THAT IS THE
        // ANSWER TO WATCH: the `memref` switch below DOES name `affine::AffineVectorLoadOp` and
        // `affine::AffineStoreOp` (`:348-362`), so the reference asks a WIDER question there than
        // it asks here. A loop whose induction variable reaches only a plain upstream access is not
        // a candidate at all, so it never reaches that switch and the difference is unobservable.
        | DfirOp::Vector(_)
        // ⭐ AND NO `uniform.` OP IS ONE. `uniform.uniformize_regions` HOLDS accesses rather than
        // being one — the ones inside its regions answer for themselves — and `uniform.query_map`
        // binds an index the schedule resolves later.
        | DfirOp::Uniform(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Symbol(_) => false,
    }
}

/// THE VIEW A MEMORY OP READS OR WRITES — the reference's `memref` switch (`:348-362`).
///
/// ```cpp
/// Value memref = nullptr;
/// if (auto agen_load = llvm::dyn_cast<agen::VectorLoadOp>(use)) {
///   memref = agen_load.getMemRef();
/// } else if (auto agen_store = llvm::dyn_cast<agen::VectorStoreOp>(use)) {
///   memref = agen_store.getMemRef();
/// } else if (auto vec_load = llvm::dyn_cast<affine::AffineVectorLoadOp>(use)) {
///   memref = vec_load.getMemref();
/// } else if (auto vec_store = llvm::dyn_cast<affine::AffineStoreOp>(use)) {
///   memref = vec_store.getMemref();
/// } else {
///   // note that there is no composite load/store and load_store for PT, PE,
///   // SFP
///   llvm_unreachable("Memory operation not supported");
/// }
/// ```
///
/// ⛔⛔ THE TWO `affine` ARMS ARE DEAD IN THE REFERENCE. [`is_memory_op`] above admits five `agen`
/// classes and NEITHER affine form, so `use` can never be an `affine.vector_load` or an
/// `affine.store` by the time this switch runs. They are transcribed because they are what the file
/// says; nothing reaches them.
///
/// ⚠️ AND THE FOURTH ARM IS `affine::AffineStoreOp` — `affine.store`, a SCALAR store, despite the
/// variable being named `vec_store`. This island declares only `affine.vector_store`, which is the
/// arm below; the pairing does not matter, because neither is reachable.
///
/// ⛔ AND THE `llvm_unreachable` IS THE THREE COMPOSITES, guarded by the caller's
/// [`REGISTER_FILE_UNITS`] test and the reference's own note that *"there is no composite load/store
/// and load_store for PT, PE, SFP"*. `None` is that abort: the caller moves to the next user, which
/// leaves the loop untouched and lets dbo-opt speak if the premise was ever wrong.
fn view_of(op: &DfirOp) -> Option<Val> {
    match op {
        DfirOp::Agen(agen::Op::VectorLoad { view, .. } | agen::Op::VectorStore { view, .. }) => {
            Some(*view)
        }
        // The two arms `is_memory_op` never admits.
        DfirOp::Affine(
            affine::Op::VectorLoad { view, .. } | affine::Op::VectorStore { view, .. },
        ) => Some(*view),
        _ => None,
    }
}

/// `memref.getDefiningOp<dataflow::GetLogicalMemoryViewOp>().getFromUnit()` (`:364-367`).
///
/// ⚠️ THE PAGED VIEW IS ANSWERED TOO, WHERE THE REFERENCE WOULD FOLLOW A NULL. `getDefiningOp<T>()`
/// returns null for a `get_paged_logical_memory_view` and `getFromUnit()` is called on it
/// unconditionally. The two ops differ only in whether the region is segmented and the question
/// being asked is about the UNIT the view is over, so answering it from either is the same fact —
/// and it is the only answer that is not a crash.
fn view_from_unit(view: Val, scope: &[DfirOp]) -> Option<Val> {
    match defining_op(view, scope)? {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { from, .. }) => Some(*from),
        DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(paged)) => Some(paged.unit),
        _ => None,
    }
}

/// `is_any_of(mem_unit, LRFREG)` where `mem_unit = dcc::getUnitType(..)` (`:366-387`, `:389-390`).
///
/// # ⭐⭐ EXACTLY THREE OF THIS ISLAND'S SIX `get_local_unit` NAMES ARE `LRFREG`
///
/// `getUnitType` lowercases the op's `name` (a `get_local_unit`) or `type` (a `get_unit`), looks it
/// up in `stringToSenComponents`, and maps the result through `senCompToGenericComp`
/// (`Utils/DccExtContext.cpp:126-138`). That table sends `PE_LRFREG`, `SFP_LRFREG`, `PT_LRFREG` and
/// the bare `LRFREG` to `LRFREG` — and sends `PTXRF` to `PTXRF`, `PTARF` to `PTARF`, `PELRF` to
/// `PELRF` and `SFPLRF` to `SFPLRF`, each its own image (`sys-arch-spec/arch_enums.cpp:153-162`).
///
/// ⛔ SO THE PT's TRANSPOSED AND ACCUMULATOR FILES ARE **NOT** LRF REGISTERS HERE. A loop indexing an
/// `ptxrf` view is not unrolled by this pass, and a port that had matched "any register file" would
/// have fully unrolled every matmul's kernel walk.
///
/// ⚠️ AND A `get_unit` IS NEVER ONE. [`crate::units::DfirUnit`] has no `lrfreg` spelling — register
/// files are bound by `dataflow.get_local_unit`, never by `get_unit` — so
/// [`crate::units::DfirUnit::generic`] cannot answer `LRFREG` and [`GenericComp`] has no variant for
/// it. The whole question is therefore *"is this view over one of the three LRFs?"*
///
/// ⚠️ AND `l0scale` IS A LOOKUP THE REFERENCE THROWS ON. `senCompToGenericComp` has 300-odd entries
/// and NO `L0_SCALE` key (`arch_enums.cpp:153-162` names every register file but that one), so
/// `.at(record->second)` raises `std::out_of_range` for a view over the L0's scale region. `false` is
/// the fact the missing entry stands for: a scale region is not an LRF.
///
/// ⚠️ The `uniform::QueryMapOp` arm (`:373-384`) has no input in this island — see
/// [`crate::bridges::dataflow_ir_to_sentient::agen_helper::ConsumerUnits`], which records the same
/// absence — and the reference's `llvm_unreachable("Unknown unit")` (`:386`) is what `false` stands
/// for below.
///
/// ⚠️ `getUnitType` ALSO DEREFERENCES AN END ITERATOR for a name outside
/// `stringToSenComponents`: `find(op.getName().lower())` is read as `record->second` with no check
/// (`DccExtContext.cpp:133-138`). A [`dataflow::LocalUnit`] is a closed set of six, so this port has
/// no unspellable name to reach it with.
fn unit_is_lrf(unit: Val, scope: &[DfirOp]) -> bool {
    match defining_op(unit, scope) {
        Some(DfirOp::Dataflow(dataflow::Op::GetLocalUnit { which, .. })) => match which {
            dataflow::LocalUnit::PeLrf
            | dataflow::LocalUnit::SfpLrf
            | dataflow::LocalUnit::PtLrf => true,
            // `PTXRF`, `PTARF` and `L0SCALE` are each their own generic component.
            dataflow::LocalUnit::PtXrf
            | dataflow::LocalUnit::PtArf
            | dataflow::LocalUnit::L0Scale => false,
        },
        // A `get_unit` cannot name a register file; anything else is the `llvm_unreachable`.
        _ => false,
    }
}

/// `getDefiningOp<arith::ConstantIndexOp>()`, and its value where there is one.
fn index_constant(val: Val, scope: &[DfirOp]) -> Option<i64> {
    match defining_op(val, scope)? {
        DfirOp::Arith(arith::Op::Constant { value, .. }) => Some(*value),
        _ => None,
    }
}

/// Replaces: e194_analyzeLoop
///
/// **194/384** `TransformLoopToLegalizeForSentientLowering::analyzeLoop` —
/// `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268` (127L).
///
/// ```cpp
/// TransformType analyzeLoop(LoopLikeOpInterface loop) {
///   Value induc_var = nullptr;
///   bool constant_bounds = false;
///   bool affine_non_const_maps = true;
///   if (auto affine_for = llvm::dyn_cast<affine::AffineForOp>(loop.getOperation())) {
///     induc_var = affine_for.getInductionVar();
///     constant_bounds = affine_for.hasConstantBounds();
///     affine_non_const_maps = !affine_for.hasConstantBounds();
///   } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop.getOperation())) {
///     // if a bound is defined for uniformation purpose, don't convert
///     if (dyn_cast<uniform::QueryMapOp>(scf_for.getLowerBound().getDefiningOp()) ||
///         dyn_cast<uniform::QueryMapOp>(scf_for.getUpperBound().getDefiningOp())) {
///       return kNone;
///     }
///     affine_non_const_maps = false;
///     induc_var = scf_for.getInductionVar();
///     auto lb_const = isa<arith::ConstantIndexOp>(scf_for.getLowerBound().getDefiningOp());
///     bool ub_const = isa<arith::ConstantIndexOp>(scf_for.getUpperBound().getDefiningOp());
///     auto step_const = scf_for.getStep().getDefiningOp<arith::ConstantIndexOp>();
///     DT_CHECK_MSG(step_const.value() == 1,
///                  "We expect the SCF-for loop to have step size of 1");
///
///     // Allow for symbolic upper bounds in case of load/store units since
///     // AgenToSentient is already enhanced to support scf.for loops natively.
///     if (!ub_const && is_any_of(curr_unit_, L3LU, L3SU, LXLU, LXSU, L0LU, L0SU)) {
///       if (isa<mlir::symbol::CreateSymbolOp>(scf_for.getUpperBound().getDefiningOp())) {
///         return kNone;
///       }
///     }
///     constant_bounds = lb_const && ub_const && step_const;
///   } else if (auto sentient_for = llvm::dyn_cast<sentient::ForOp>(loop.getOperation())) {
///     affine_non_const_maps = false;
///     return kNone;
///   } else {
///     llvm_unreachable("Unknown loop operation");
///   }
///
///   // if loop has constant bounds in load units or store units --> don't perform conversion
///   if (constant_bounds && is_any_of(curr_unit_, L3LU, L3SU, LXLU, LXSU, L0LU, L0SU)) {
///     return kNone;
///   }
///
///   for (auto* use : induc_var.getUsers()) {
///     bool is_memory_op = isa<agen::VectorLoadOp, agen::VectorStoreOp, agen::CompositeLoadOp,
///                             agen::CompositeStoreOp, agen::CompositeLoadAndStoreOp>(use);
///     // if it's not a memory operation, --> don't perform conversion
///     if (!is_memory_op) { continue; }
///
///     // if it's not constant, then unrolling doesn't work.
///     // we currently support bound being part of conditional on the parent loop.
///     if (!constant_bounds && !affine_non_const_maps) {
///       // TODO: enhance condition to do splitting only for PT.
///       if (!is_any_of(curr_unit_, L3LU, L3SU)) return KSplitParent;
///     }
///
///     // in the constant bounds and in PT LRF's --> perform unrolling
///     if (is_any_of(curr_unit_, PT, SFP, PE)) {
///       Value memref = nullptr;                        /* the switch, `:348-362` */
///       auto view_op = memref.getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
///       SenComponents mem_unit;                        /* the three unit forms, `:366-387` */
///       if (is_any_of(mem_unit, LRFREG)) { return KUnroll; }
///     }
///   }
///
///   return kNone;
/// }
/// ```
///
/// # ⭐⭐ `affine_non_const_maps` IS NOT A SECOND FACT — IT IS "THIS IS AN `scf.for`"
///
/// Read the three assignments together. The affine arm sets it to `!hasConstantBounds()` while
/// `constant_bounds` is `hasConstantBounds()`, so the two are COMPLEMENTS there; the scf arm sets it
/// to `false`; the sentient arm sets it and returns. The one place it is read is
/// `!constant_bounds && !affine_non_const_maps` (`:340`), which is therefore
///
/// | loop | `!constant_bounds && !affine_non_const_maps` |
/// |---|---|
/// | `affine.for` | `!cb && cb` — **always false** |
/// | `scf.for` | `!cb && true` — exactly *"the bounds are not all constants"* |
///
/// ⛔⛔ SO `KSplitParent` IS `scf.for`-ONLY, and `transformLoop` agrees: its `affine.for` branch
/// handles `KUnroll` and nothing else (`:406-414`), while `KSplitParent` reaches
/// `transformSCFLoopWithNonConstantUpperBound(scf_for)` (`:430-435`). A port that had carried
/// `affine_non_const_maps` as a free `bool` would have made an unreachable state reachable; the match
/// below states the loop's FORM once and derives both.
///
/// # ⭐ THE THREE ANSWERS, AGAINST THE VENDOR'S OWN THREE FILES
///
/// | vendor case | unit | loop | answer |
/// |---|---|---|---|
/// | `l3lu_disable_transformation.mlir` | `l3lu` | `scf.for %c0 to %10` where `%10 = arith.select` | [`LoopTransform::None`] — output is byte-identical to input |
/// | `memory-dyn-loops.mlir` | `l0lurow0` | the same shape, one `agen.vector_load` on the `iv` | [`LoopTransform::SplitParent`] — `:127` becomes `scf.if` + two `affine.for`s of extent 2 and 1 |
/// | `dyn-loops-cond-bound.mlir` | `ptrow0` | two `scf.for`s on `arith.select` bounds, each with a memory user on `pt_lrfreg` | [`LoopTransform::SplitParent`] — `:173` and `:185` each become `scf.if`, the then-arm holding TWO body copies and the else-arm ONE |
///
/// ⛔ THE L3 CASE IS THE ONE THAT PROVES THE COMPONENT TEST MATTERS: the two inputs have the same
/// loop, the same `arith.select` bound and the same memory user, and the ONLY difference is which
/// unit the enclosing `dataflow.program_unit` runs on. The vendor's `CHECK-SENT-IR` for the `l3lu`
/// file reproduces its input line for line.
///
/// # ⛔ THE `DT_CHECK_MSG` ON THE STEP BECOMES `kNone`, NOT AN ABORT
///
/// *"We expect the SCF-for loop to have step size of 1"* (`:296-297`) is an assertion, and
/// `step_const.value()` also dereferences a null when the step is not a constant at all. This crate
/// never asserts. `kNone` is the answer that leaves the loop EXACTLY as the assertion's premise says
/// it already is — untouched — so a step-2 `scf.for` over an LRF reaches dbo-opt and is refused
/// there, by the only oracle, instead of stopping the build before the tape is emitted.
///
/// # ⚠️ TWO NULL DEREFERENCES THIS PORT ANSWERS INSTEAD OF FOLLOWING
///
/// `scf_for.getLowerBound().getDefiningOp()` is handed straight to `dyn_cast<QueryMapOp>` (`:283-286`)
/// with no null check, so an `scf.for` whose bound is a region argument — a loop carrying a bound
/// from its parent — segfaults `dcc`. [`defining_op`] answers `None`, the bound is not a
/// constant, and analysis continues.
#[must_use]
pub fn analyze_loop(
    loop_op: &LoopUnderAnalysis<'_>,
    curr_unit: GenericComp,
    scope: &[DfirOp],
) -> LoopTransform {
    // `Value induc_var = nullptr; bool constant_bounds = false; bool affine_non_const_maps = true;`
    // — one match instead of three, because the flag IS the form. See the doc's table.
    let (induc_var, body, constant_bounds, affine_non_const_maps) = match loop_op {
        LoopUnderAnalysis::AffineFor { iv, lo, hi, body } => {
            // `constant_bounds = affine_for.hasConstantBounds();`
            // `affine_non_const_maps = !affine_for.hasConstantBounds();`
            //
            // ⭐ AND THE REFERENCE'S OWN NOTE ON WHY THOSE TWO ARE COMPLEMENTS HERE:
            // *"TODO: Sometimes, loop bounds are constant SSA variables rather than constant maps.
            // Hence, we may have to check about them. this leads to different values of
            // affine_const_maps and constant_bounds"* (`:278-280`) — i.e. reading THROUGH an
            // `arith.constant` operand would make `constant_bounds` true while the MAP stayed
            // non-constant, and only then would the two flags carry different facts. Until that
            // happens they do not, which is what the doc's table turns on.
            //
            // ⭐ `Bound::Val` IS A NON-CONSTANT MAP. `affine.for %i = 0 to %extent` takes the bound
            // as an operand of an otherwise trivial map, which is exactly what
            // `hasConstantBounds()` answers `false` for.
            let constant_bounds =
                matches!(lo, affine::Bound::Const(_)) && matches!(hi, affine::Bound::Const(_));
            (*iv, *body, constant_bounds, !constant_bounds)
        }
        LoopUnderAnalysis::ScfFor {
            iv,
            lo,
            hi,
            step,
            body,
        } => {
            // `if (dyn_cast<uniform::QueryMapOp>(getLowerBound().getDefiningOp()) || ..) return
            // kNone;` — *"if a bound is defined for uniformation purpose, don't convert"*. The
            // `uniform` dialect is not in this island, so the test has no input; see
            // [`unit_is_lrf`]'s note on the same absence.

            // `auto lb_const = isa<arith::ConstantIndexOp>(getLowerBound().getDefiningOp());`
            let lb_const = index_constant(*lo, scope).is_some();
            // `bool ub_const = isa<arith::ConstantIndexOp>(getUpperBound().getDefiningOp());`
            let ub_const = index_constant(*hi, scope).is_some();
            // `auto step_const = getStep().getDefiningOp<arith::ConstantIndexOp>();`
            let step_const = index_constant(*step, scope);

            // `DT_CHECK_MSG(step_const.value() == 1, "We expect the SCF-for loop to have step size
            // of 1");` — see the doc for why this is `kNone` and not a stop.
            //
            // ⛔ IN EVERY BUILD. `DT_CHECK_MSG` throws a `DtException` unconditionally
            // (`util/dt_exception.hpp:110-118`); it is `DT_CHECK_MSG_OPT` that compiles away without
            // `DT_USE_OPTIONAL_CHECK` (`:136-145`). So a step of 2 is not a release-mode fall-through
            // in the reference either — it stops the compiler.
            if step_const != Some(1) {
                return LoopTransform::None;
            }

            // *"Allow for symbolic upper bounds in case of load/store units since AgenToSentient is
            // already enhanced to support scf.for loops natively."*
            if !ub_const
                && LOAD_AND_STORE_UNITS.contains(&curr_unit)
                && matches!(
                    defining_op(*hi, scope),
                    Some(DfirOp::Symbol(symbol::Op::CreateSymbol { .. }))
                )
            {
                return LoopTransform::None;
            }

            // `constant_bounds = lb_const && ub_const && step_const;` — the third conjunct is a
            // non-null OP, which the check above has already established.
            (
                *iv,
                *body,
                lb_const && ub_const && step_const.is_some(),
                false,
            )
        }
        // `affine_non_const_maps = false; return kNone;` — the store is dead, the return is not.
        LoopUnderAnalysis::SentientFor => return LoopTransform::None,
    };

    // *"if loop has constant bounds in load units or store units --> don't perform conversion"*.
    if constant_bounds && LOAD_AND_STORE_UNITS.contains(&curr_unit) {
        return LoopTransform::None;
    }

    // `for (auto* use : induc_var.getUsers())` — the induction variable is only nameable inside its
    // own body, so the body IS the whole use list. [`uses`] descends into nested regions,
    // which is where `memory-dyn-loops.mlir`'s `agen.vector_load` sits.
    for user in uses(induc_var, body) {
        // *"if it's not a memory operation, --> don't perform conversion"*.
        if !is_memory_op(user) {
            continue;
        }

        // *"if it's not constant, then unrolling doesn't work. we currently support bound being part
        // of conditional on the parent loop."*
        if !constant_bounds && !affine_non_const_maps {
            // `// TODO: enhance condition to do splitting only for PT.`
            if !L3_HALVES.contains(&curr_unit) {
                return LoopTransform::SplitParent;
            }
        }

        // *"in the constant bounds and in PT LRF's --> perform unrolling"*.
        if REGISTER_FILE_UNITS.contains(&curr_unit) {
            let Some(view) = view_of(user) else {
                // `llvm_unreachable("Memory operation not supported")`.
                continue;
            };
            let Some(mem_unit) = view_from_unit(view, scope) else {
                continue;
            };
            if unit_is_lrf(mem_unit, scope) {
                return LoopTransform::Unroll;
            }
        }
    }

    LoopTransform::None
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 195/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT `transformLoop` ASKED FOR.
///
/// # ⛔⛔ THE REQUEST, BECAUSE NOT ONE OF THE THREE TRANSFORMS IS THIS FUNCTION'S TO PERFORM
///
/// `loopUnrollFull` and `loopUnrollByFactor` are upstream MLIR — `mlir/Dialect/{Affine,SCF}/Utils` —
/// and `transformSCFLoopWithNonConstantUpperBound` is entry **292** of this campaign, 94 lines in this
/// same file, unported. So all three lie outside entry 195, and what IS inside it is the dispatch on
/// the loop's form, the unroll factor it computes from the loop's own three constants, and the pairing
/// it silently ignores.
///
/// ⭐ THE SAME SHAPE ENTRY 109 ALREADY TOOK, for the same reason and against the same two utilities:
/// [`Unroll`](super::tf_loop_unroll_for_shuffle_op::Unroll) names the call rather than pretending to
/// have made it, and this enum's factor is the very
/// [`TripCount`](super::tf_loop_unroll_for_shuffle_op::TripCount) that one carries.
///
/// ⛔ AND NO FAILURE VARIANT, unlike entry 109's. Every `failed(..)` in `transformLoop` is the
/// CALLEE's — `emitError("Unable to unroll loop")` (`:410`), `emitError("Unable to unroll the scf
/// loop")` (`:426`), `emitError("Doesn't support this loops for transformation")` (`:432`), each
/// followed by `signalPassFailure()`. This function has no decline of its own to state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum LoopRewrite {
    /// *"if no action required, return"* (`:401-404`) — and the three pairings the reference passes
    /// over in silence; see [`transform_loop`].
    Nothing,
    /// `loopUnrollFull(affine_for)` (`:409`) — affine derives the trip count from the loop's own maps,
    /// so the request carries nothing.
    UnrollAffineFully,
    /// `loopUnrollByFactor(scf_for, unroll_size)` (`:425`), where
    /// `unroll_size = (ub_const.value() - lb_const.value()) / step_const.value()` (`:423-424`).
    ///
    /// ⭐ THE ONLY ARITHMETIC IN THE WHOLE FUNCTION, and the reason SCF needs a factor where affine
    /// does not: *"SCF lacks a dedicated full unroll function"*
    /// (`LoopUnrollForShuffleOp.cpp:157-158`).
    UnrollScfBy(TripCount),
    /// `transformSCFLoopWithNonConstantUpperBound(scf_for)` (`:431`) — **entry 292**, which lifts the
    /// conditional that computes the bound into an `scf.if` with one `affine.for` per arm.
    SplitParent,
}

/// Replaces: e195_transformLoop
///
/// **195/384** `TransformLoopToLegalizeForSentientLowering::transformLoop` —
/// `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399` (38L).
///
/// ```cpp
/// // Transforms loop based on the TransformType
/// void transformLoop(mlir::LoopLikeOpInterface loop_op, TransformType type) {
///   // if no action required, return;
///   if (type == TransformType::kNone) {
///     return;
///   }
///
///   if (auto affine_for = llvm::dyn_cast<affine::AffineForOp>(loop_op.getOperation())) {
///     if (type == TransformType::KUnroll) {
///       if (failed(loopUnrollFull(affine_for))) {
///         affine_for->emitError("Unable to unroll loop");
///         signalPassFailure();
///         return;
///       }
///     }
///   } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop_op.getOperation())) {
///     auto lb_const = scf_for.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
///     auto ub_const = scf_for.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
///     auto step_const = scf_for.getStep().getDefiningOp<arith::ConstantIndexOp>();
///     if (type == TransformType::KUnroll) {
///       auto unroll_size = (ub_const.value() - lb_const.value()) / step_const.value();
///       if (failed(loopUnrollByFactor(scf_for, unroll_size))) {
///         scf_for->emitError("Unable to unroll the scf loop");
///         signalPassFailure();
///         return;
///       }
///     } else {  // kParentSplit
///       if (failed(transformSCFLoopWithNonConstantUpperBound(scf_for))) {
///         scf_for->emitError("Doesn't support this loops for transformation");
///         signalPassFailure();
///         return;
///       }
///     }
///   }
/// }
/// ```
///
/// # ⭐⭐ THE `else` IS `KSplitParent` AND NOTHING ELSE, WHICH IS WHY THE PAIRING MATTERS
///
/// The scf branch's `else` carries the comment `// kParentSplit`, so it is not a default over two
/// remaining values — `kNone` returned at the top, leaving exactly `KUnroll` and `KSplitParent`. The
/// affine branch has no `else` at all, so an affine loop asked to split its parent is passed over in
/// silence.
///
/// ⛔ AND THAT SILENCE IS UNREACHABLE, BY ENTRY 194. `!constant_bounds && !affine_non_const_maps` is
/// `!cb && cb` on the affine path, so `analyzeLoop` never answers `KSplitParent` for an
/// `affine.for` — see [`analyze_loop`]'s own table. The pairing is not merely unlikely; it is a
/// consequence of the only function that produces the second argument.
///
/// # ⭐ THE THREE BOUND READS ARE UNCONDITIONAL IN THE REFERENCE AND USED ON ONE PATH
///
/// `:417-421` reads `lb_const`, `ub_const` and `step_const` before `:422` looks at `type`, so the
/// split path reads them and throws all three away. `getDefiningOp<T>()` is a pure query returning
/// null on a mismatch, so reading them inside the `KUnroll` arm — where the one use is — is the same
/// program. That is also why the port needs `scope`: it is this island's `getDefiningOp`.
///
/// # ⛔⛔ `ub_const.value()` IS SAFE ONLY BECAUSE ENTRY 194 ORDERS ITS TESTS THE WAY IT DOES
///
/// The three `getDefiningOp<arith::ConstantIndexOp>()`s are read with NO null check. An `scf.for`
/// whose bound is an `arith.select` would fault here — and cannot arrive, because `analyzeLoop` tests
/// `!constant_bounds` at `:340-343` and the register file only at `:347-390`, so a non-constant
/// `scf.for` on a unit that owns one leaves as `KSplitParent` and never reaches `KUnroll`.
///
/// ⭐⭐ AND THE VENDOR PROVES IT ON AN INPUT WHERE BOTH TESTS MATCH — `dyn-loops-cond-bound.mlir` runs
/// on `ptrow0` (`:148`) and its `scf.for` (`:173`) has an `arith.select` bound (`:172`) with the
/// induction variable read by a store on `pt_lrfreg` (`:181`, `:154`). The answer is `scf.if` (`:47`),
/// not an unrolled loop. See
/// [`a_non_constant_bound_splits_before_the_register_file_is_consulted`](unit_tests::a_non_constant_bound_splits_before_the_register_file_is_consulted).
///
/// ⭐ SO A MISSING CONSTANT ANSWERS [`LoopRewrite::Nothing`], the same answer the reference's own
/// silent pairings give: the loop is left exactly as it stands, which is what "this cannot happen"
/// claims about it, and dbo-opt is the oracle if the claim was ever wrong.
///
/// ⛔ AND SO DOES AN `unroll_size` OF ZERO OR LESS. `lb = ub` is a loop the folder produces routinely
/// and `(ub - lb) / step` is then `0`, which reaches `loopUnrollByFactor`'s `assert(unrollFactor > 0)`
/// — the very defect [`TripCount`](super::tf_loop_unroll_for_shuffle_op::TripCount) exists to keep out
/// of this crate, documented there for entry 109. Leaving an empty loop alone is what unrolling it
/// zero times would have meant.
///
/// # Arguments
///
/// * `loop_op` — the loop, classified. `analyzeAndTransform` passes the very op it just analysed
///   (`:440-445`), which is what makes the pairing arguments above hold.
/// * `transform` — [`analyze_loop`]'s answer for that same loop.
/// * `scope` — where the `scf.for`'s three bound operands are defined.
pub fn transform_loop(
    loop_op: &LoopUnderAnalysis<'_>,
    transform: LoopTransform,
    scope: &[DfirOp],
) -> LoopRewrite {
    // `if (type == TransformType::kNone) { return; }`
    if transform == LoopTransform::None {
        return LoopRewrite::Nothing;
    }

    match loop_op {
        // `:406-414` — the affine branch, whose only action is a full unroll.
        LoopUnderAnalysis::AffineFor { .. } => match transform {
            LoopTransform::Unroll => LoopRewrite::UnrollAffineFully,
            // No `else` in the reference, and entry 194 cannot produce this pairing.
            LoopTransform::SplitParent | LoopTransform::None => LoopRewrite::Nothing,
        },
        // `:415-437` — the scf branch, which reads all three bounds before it looks at `type`.
        LoopUnderAnalysis::ScfFor { lo, hi, step, .. } => match transform {
            LoopTransform::Unroll => {
                // `auto unroll_size = (ub_const.value() - lb_const.value()) / step_const.value();`
                match (
                    index_constant(*lo, scope),
                    index_constant(*hi, scope),
                    index_constant(*step, scope),
                ) {
                    (Some(lb), Some(ub), Some(step)) if step != 0 => {
                        TripCount::positive((ub - lb) / step)
                            .map_or(LoopRewrite::Nothing, LoopRewrite::UnrollScfBy)
                    }
                    // A bound the reference reads through a null pointer, and a step of zero, which
                    // divides by it. Neither can arrive; see the doc.
                    _ => LoopRewrite::Nothing,
                }
            }
            // `} else {  // kParentSplit` — entry 292.
            LoopTransform::SplitParent => LoopRewrite::SplitParent,
            LoopTransform::None => LoopRewrite::Nothing,
        },
        // Neither `dyn_cast` matches a `sentient.for`, so the function body falls off its end.
        LoopUnderAnalysis::SentientFor => LoopRewrite::Nothing,
    }
}

/// Replaces: e257_analyzeAndTransform
///
/// **257/384** `TransformLoopToLegalizeForSentientLowering::analyzeAndTransform` —
/// `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441` (3L).
///
/// ⭐ THE WHOLE FUNCTION IS THE PAIRING, AND IT IS WHY ENTRY 195'S SILENT ARMS ARE UNREACHABLE:
/// `transformLoop` is only ever handed `analyzeLoop`'s answer for the SAME op, so the affine/
/// `KSplitParent` and non-constant/`KUnroll` combinations [`transform_loop`] documents cannot arrive.
#[must_use]
pub fn analyze_and_transform(
    loop_op: &LoopUnderAnalysis<'_>,
    curr_unit: GenericComp,
    scope: &[DfirOp],
) -> LoopRewrite {
    // `auto type = analyzeLoop(loop_op); transformLoop(loop_op, type);`
    transform_loop(loop_op, analyze_loop(loop_op, curr_unit, scope), scope)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::{Index, agen, arith, dataflow};
    use crate::islands::dataflow_ir::print;
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, MemRef, ScalarTy, Vector,
    };
    use crate::units::{Core, Corelet, DfirUnit, Residency, Row};

    /// `memref<64xf16>` — the transfer's source and destination type in the vendor's case.
    fn stick() -> MemRef {
        MemRef {
            shape: vec![64],
            elem: ElemType::F16,
        }
    }

    /// `#set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>`.
    fn load_set() -> IntegerSet {
        IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(63)),
                    is_equality: false,
                },
            ],
        }
    }

    /// `#set1` — the transfer's `time_set`, four dimensions pinned to one step.
    fn time_set() -> IntegerSet {
        let bounds = [1i64, 15, 0, 0];
        IntegerSet {
            dims: 4,
            symbols: 0,
            constraints: (0..4)
                .rev()
                .flat_map(|dim| {
                    let upper = bounds[3 - usize::try_from(dim).expect("four dims")];
                    [
                        Constraint {
                            expr: AffineExpr::dim(dim),
                            is_equality: false,
                        },
                        Constraint {
                            expr: if upper == 0 {
                                AffineExpr::dim(dim).times(-1)
                            } else {
                                AffineExpr::dim(dim)
                                    .times(-1)
                                    .plus(AffineExpr::Const(upper))
                            },
                            is_equality: false,
                        },
                    ]
                })
                .collect(),
        }
    }

    /// THE VENDOR'S `scf.for` AND ITS CONTEXT, BUILT IN TYPED FORM.
    ///
    /// `dcc/test/Transform/TransformLoopToLegalizeForSentientLowering/scf_loop_with_result.mlir:129-148`
    /// — the input of the pass, from `%11 = scf.for %arg4 = %c0 to %10 step %c1 iter_args(%arg5 =
    /// %arg3)` down to its `scf.yield %12`, with the four values its body reads from outside minted
    /// first so the printed numbering lines up with the vendor's.
    fn vendor_scf_for(vals: &mut Values) -> DfirOp {
        let c0 = vals.mint(); // %c0
        let c2048 = vals.mint(); // %c2048
        let c64000 = vals.mint(); // %c64000
        let hbm = vals.mint(); // %5
        let lx = vals.mint(); // %6
        let outer_iv = vals.mint(); // %arg2, the enclosing affine.for's induction variable
        let outer_arg = vals.mint(); // %arg3, the value the scf.for starts from
        let c1 = vals.mint(); // %c1, the step

        // The scf.for's own values, in MLIR's numbering order.
        let scf_result = vals.mint(); // %11
        let scf_iv = vals.mint(); // %arg4
        let scf_arg = vals.mint(); // %arg5

        // `%12 = affine.for %arg6 = 0 to 8`, `%13 = .. to 4`, `%14 = .. to 1`.
        let mb_result = vals.mint();
        let mb_iv = vals.mint();
        let mb_arg = vals.mint();
        let x_result = vals.mint();
        let x_iv = vals.mint();
        let x_arg = vals.mint();
        let out_result = vals.mint();
        let out_iv = vals.mint();
        let out_arg = vals.mint();

        // The innermost body.
        let addr = vals.mint(); // %15 = arith.subi %c2048, %arg11
        let src_view = vals.mint(); // %16
        let dst_view = vals.mint(); // %17
        let load_iv = vals.mint(); // %arg12

        let transfer = agen::Op::CompositeLoadAndStore(Box::new(agen::CompositeTransfer {
            src: src_view,
            src_indices: vec![Index::Strided(
                vec![
                    (c0, 1),
                    (out_iv, 128),
                    (x_iv, 2048),
                    (mb_iv, 8192),
                    (scf_iv, 65536),
                    (outer_iv, 2_097_152),
                ],
                0,
            )],
            src_ty: stick(),
            dst: dst_view,
            dst_indices: vec![Index::Val(c0)],
            dst_ty: stick(),
            load_iv,
            load_iv_ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
            load_set: load_set(),
            load_order: AffineMap::identity(1),
            store_set: load_set(),
            store_order: AffineMap::identity(1),
            time_set: time_set(),
            time_order: AffineMap::identity(4),
            load_time_addr_map: AffineMap {
                dims: 4,
                syms: 0,
                results: vec![
                    AffineExpr::dim(3)
                        .times(64)
                        .plus(AffineExpr::dim(2).times(128))
                        .plus(AffineExpr::dim(1).times(8192))
                        .plus(AffineExpr::dim(0).times(65536)),
                ],
            },
            store_time_addr_map: AffineMap {
                dims: 4,
                syms: 0,
                results: vec![
                    AffineExpr::dim(3)
                        .times(64)
                        .plus(AffineExpr::dim(2).times(128))
                        .plus(AffineExpr::dim(1).times(2048))
                        .plus(AffineExpr::dim(0).times(2048)),
                ],
            },
            body: vec![DfirOp::Agen(agen::Op::Yield)],
        }));

        let innermost = DfirOp::Affine(affine::Op::For {
            iv: out_iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(1),
            carried: vec![affine::Carried {
                init: x_arg,
                arg: out_arg,
                result: out_result,
            }],
            body: vec![
                DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                    result: addr,
                    lhs: c2048,
                    rhs: out_arg,
                    ty: ScalarTy::Index,
                })),
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: src_view,
                    from: hbm,
                    start: c64000,
                    layout: AffineMap::identity(1),
                    ty: stick(),
                }),
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: dst_view,
                    from: lx,
                    start: addr,
                    layout: AffineMap::identity(1),
                    ty: stick(),
                }),
                DfirOp::Agen(transfer),
                DfirOp::Affine(affine::Op::Yield {
                    operands: vec![addr],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ds0-ds1-out".to_owned()),
        });

        let x = DfirOp::Affine(affine::Op::For {
            iv: x_iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: vec![affine::Carried {
                init: mb_arg,
                arg: x_arg,
                result: x_result,
            }],
            body: vec![
                innermost,
                DfirOp::Affine(affine::Op::Yield {
                    operands: vec![out_result],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ds0-ds1-x".to_owned()),
        });

        let mb = DfirOp::Affine(affine::Op::For {
            iv: mb_iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(8),
            carried: vec![affine::Carried {
                init: scf_arg,
                arg: mb_arg,
                result: mb_result,
            }],
            body: vec![
                x,
                DfirOp::Affine(affine::Op::Yield {
                    operands: vec![x_result],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ds0-ds1-mb".to_owned()),
        });

        DfirOp::Scf(scf::Op::For {
            iv: scf_iv,
            lo: c0,
            // `%10 = arith.select %9, %c16, %c32` — THE BOUND THIS PASS EXISTS TO REMOVE. It is not
            // read by the rewrite, which is why [`ScfForOp`] does not carry it.
            hi: c1,
            step: c1,
            carried: vec![affine::Carried {
                init: outer_arg,
                arg: scf_arg,
                result: scf_result,
            }],
            body: vec![
                mb,
                DfirOp::Scf(scf::Op::Yield {
                    operands: vec![mb_result],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ibr-chunk-y".to_owned()),
        })
    }

    fn printed(op: &DfirOp) -> String {
        let mut out = String::new();
        print::emit(&mut out, op, 0);
        out
    }

    /// THE VENDOR'S OWN EXPECTATION FOR THE `then` ARM, RENUMBERED.
    ///
    /// `scf_loop_with_result.mlir:41-61`, the `CHECK-SENT-IR-NEXT` block from
    /// `%[[VAL_23]] = affine.for %[[VAL_24]] = 0 to 16` down to its
    /// `} {dbgName = "c0-l3lu-loop-ibr-chunk-y"}`.
    ///
    /// ⭐ EVERY LOOP VALUE IS THE VENDOR'S PLUS ONE, and that is the whole content of the check: the
    /// vendor's `func.func` declares eight constants and five units before the program unit, this
    /// fixture mints only the eight values the `scf.for` actually reads, so the two preambles differ
    /// by one and every value the REWRITE creates lines up — result, induction variable, carried
    /// argument, then the body, four levels deep.
    ///
    /// ⛔ TWO ATTRIBUTES OF THE TRANSFER ARE ABSENT, AND NEITHER IS THIS REWRITE'S. The vendor's
    /// line also carries `dbgName = "c0-l3lu-transfer-lds0-src:hbm-dst:lx"` and
    /// `dir = #agen<direction PseudoRandom>`;
    /// [`crate::islands::dataflow_ir::dialects::agen::CompositeTransfer`] has no field for either, so
    /// they cannot appear here. `transformSCFToAffineLoop` clones that op opaquely — it never reads
    /// an attribute of it — so what this proves about the rewrite is unaffected, and adding the two
    /// fields is the transfer lowering's work, not this unit's.
    const VENDOR_THEN_ARM: &str = r#"%24 = affine.for %25 = 0 to 16 iter_args(%26 = %6) -> (index) {
  %27 = affine.for %28 = 0 to 8 iter_args(%29 = %26) -> (index) {
    %30 = affine.for %31 = 0 to 4 iter_args(%32 = %29) -> (index) {
      %33 = affine.for %34 = 0 to 1 iter_args(%35 = %32) -> (index) {
        %36 = arith.subi %1, %35 : index
        %37 = dataflow.get_logical_memory_view %3, %2 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
        %38 = dataflow.get_logical_memory_view %4, %36 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
        agen.composite_load_and_store src:%37[%0 + %34 * 128 + %31 * 2048 + %28 * 8192 + %25 * 65536 + %5 * 2097152] dst:%38[%0]
         time_symbols(), load_iv(%39:vector<64xf16>)
         {load_order = affine_map<(d0) -> (d0)>, load_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>, load_time_addr_map = affine_map<(d0, d1, d2, d3) -> (d3 * 64 + d2 * 128 + d1 * 8192 + d0 * 65536)>, store_order = affine_map<(d0) -> (d0)>, store_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>, store_time_addr_map = affine_map<(d0, d1, d2, d3) -> (d3 * 64 + d2 * 128 + d1 * 2048 + d0 * 2048)>, time_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, time_set = affine_set<(d0, d1, d2, d3) : (d3 >= 0, -d3 + 1 >= 0, d2 >= 0, -d2 + 15 >= 0, d1 >= 0, -d1 >= 0, d0 >= 0, -d0 >= 0)>}
        {
          agen.yield
        } : memref<64xf16>, memref<64xf16>
        affine.yield %36 : index
      } {dbgName = "c0-l3lu-loop-ds0-ds1-out"}
      affine.yield %33 : index
    } {dbgName = "c0-l3lu-loop-ds0-ds1-x"}
    affine.yield %30 : index
  } {dbgName = "c0-l3lu-loop-ds0-ds1-mb"}
  affine.yield %27 : index
} {dbgName = "c0-l3lu-loop-ibr-chunk-y"}
"#;

    fn transform(vals: &mut Values, scf_for: &DfirOp, ubound: i64) -> LoopLegalization {
        transform_scf_to_affine_loop(
            vals,
            &ScfForOp::of(scf_for).expect("the fixture is an scf.for"),
            StaticBounds {
                lbound: 0,
                ubound,
                step: 1,
            },
        )
    }

    /// 🎯 117/384 — THE VENDOR'S CASE, WHOLE: `scf.for` WITH A RESULT BECOMES `affine.for 0 to 16`.
    ///
    /// `dcc-opt --dcc-transform-loop-to-legalize-for-sentient-lowering scf_loop_with_result.mlir`.
    #[test]
    fn the_vendors_scf_loop_with_result_becomes_the_vendors_affine_loop() {
        let mut vals = Values::default();
        let scf_for = vendor_scf_for(&mut vals);
        let LoopLegalization::Transformed(then_arm) = transform(&mut vals, &scf_for, 16) else {
            unreachable!("lbound 0 and step 1 are the supported case");
        };
        assert_eq!(printed(&then_arm), VENDOR_THEN_ARM);
    }

    /// 🎯 117/384 — THE SAME BODY TRANSFORMED TWICE BINDS TWO DISJOINT SETS OF NAMES.
    ///
    /// ⛔⛔ THE CALLER DOES THIS ON EVERY LOOP IT LEGALISES.
    /// `transformSCFLoopWithNonConstantUpperBound` (entry 292, `:207` and `:227`) calls this once with
    /// the `then` builder and `then_ub` and once with the `else` builder and `else_ub`, on the SAME
    /// `scf.for` — the vendor's expectation is two copies of one body, at `to 16` and at `to 32`
    /// (`scf_loop_with_result.mlir:41` and `:64`). Two copies sharing SSA names is not a program, and
    /// nothing in a typed island would say so; this is what says so.
    #[test]
    fn transforming_one_loop_twice_binds_disjoint_values() {
        let mut vals = Values::default();
        let scf_for = vendor_scf_for(&mut vals);
        let LoopLegalization::Transformed(then_arm) = transform(&mut vals, &scf_for, 16) else {
            unreachable!("the then arm")
        };
        let LoopLegalization::Transformed(else_arm) = transform(&mut vals, &scf_for, 32) else {
            unreachable!("the else arm")
        };

        let (then_bound, else_bound) = (bound_of(&then_arm), bound_of(&else_arm));
        assert_eq!((then_bound, else_bound), (16, 32));

        let then_defined = defined_values(&then_arm);
        let else_defined = defined_values(&else_arm);
        assert_eq!(then_defined.len(), else_defined.len());
        for val in &then_defined {
            assert!(
                !else_defined.contains(val),
                "{val:?} is bound by both copies of the body"
            );
        }

        // ⭐ AND THE TWO ARMS START FROM THE SAME INIT — the enclosing loop's carried argument, which
        // is outside both bodies and therefore not renumbered (`:56` and `:74` both read `%20`).
        assert_eq!(init_of(&then_arm), init_of(&else_arm));
    }

    /// 🎯 117/384 — A LOWER BOUND THAT IS NOT 0, OR A STEP THAT IS NOT 1, IS `LogicalResult::failure`.
    ///
    /// *"we currently support lbound being 0, step being 1, non-constant ubound."* (`:105`.) The
    /// caller answers a failure with `if_op->emitError("Unable to transform SCF loop into Affine
    /// loop")` (`:210` for the then arm, `:230` for the else) and leaves the `scf.for` where it was.
    #[test]
    fn only_a_zero_lower_bound_and_a_unit_step_are_supported() {
        let mut vals = Values::default();
        let scf_for = vendor_scf_for(&mut vals);
        let of = ScfForOp::of(&scf_for).expect("the fixture is an scf.for");

        for bounds in [
            StaticBounds {
                lbound: 1,
                ubound: 16,
                step: 1,
            },
            StaticBounds {
                lbound: 0,
                ubound: 16,
                step: 2,
            },
            StaticBounds {
                lbound: -4,
                ubound: 16,
                step: 4,
            },
        ] {
            assert_eq!(
                transform_scf_to_affine_loop(&mut vals, &of, bounds),
                LoopLegalization::UnsupportedBounds {
                    lbound: bounds.lbound,
                    step: bounds.step,
                }
            );
        }

        // ⛔ AND A DECLINE MINTS NOTHING. The reference returns before `AffineForOp::create`.
        let before = vals.issued();
        let _ = transform_scf_to_affine_loop(
            &mut vals,
            &of,
            StaticBounds {
                lbound: 8,
                ubound: 16,
                step: 1,
            },
        );
        assert_eq!(vals.issued(), before);
    }

    /// 🎯 117/384 — A LOOP THAT CARRIES NOTHING GETS NO `affine.yield`, AND EVERY BODY OP IS CLONED.
    ///
    /// *"affine_for already has an implicit affine.yield. We need to update it if there are
    /// results."* (`:129-130`) — `getNumResults() > 0` is false, so the body is the clone and nothing
    /// else. The `scf.for` this comes from has no explicit terminator in this island either, which is
    /// what [`ScfForOp::split_terminator`] answers `None` for.
    #[test]
    fn a_loop_with_no_results_gets_no_yield() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let bound = vals.mint();
        let doubled = vals.mint();
        let scf_for = DfirOp::Scf(scf::Op::For {
            iv,
            lo: bound,
            hi: bound,
            step: bound,
            carried: Vec::new(),
            body: vec![DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                result: doubled,
                lhs: iv,
                rhs: iv,
                ty: ScalarTy::Index,
            }))],
            dbg_name: None,
        });

        let LoopLegalization::Transformed(affine_for) = transform(&mut vals, &scf_for, 12) else {
            unreachable!("lbound 0 and step 1")
        };
        assert_eq!(
            printed(&affine_for),
            "affine.for %3 = 0 to 12 {\n  %4 = arith.addi %3, %3 : index\n}\n"
        );
    }

    /// 🎯 117/384 — A YIELD OPERAND DEFINED OUTSIDE THE BODY COMES THROUGH UNCHANGED.
    ///
    /// ⚠️ THE ONE PLACE THIS PORT DIVERGES, DELIBERATELY. `bv_map.lookup(operand)` returns null for a
    /// value the mapping never saw and MLIR then fails a verifier;
    /// [`ValueMapping::lookup_or_default`] returns the value itself. A legal `scf.for` can forward its
    /// own init straight out — `scf.yield %arg5` — and passing it through is the only answer that is
    /// not a stop, which this crate does not have.
    #[test]
    fn a_yield_operand_from_outside_the_body_passes_through() {
        let mut vals = Values::default();
        let init = vals.mint();
        let bound = vals.mint();
        let iv = vals.mint();
        let arg = vals.mint();
        let result = vals.mint();
        let scf_for = DfirOp::Scf(scf::Op::For {
            iv,
            lo: bound,
            hi: bound,
            step: bound,
            carried: vec![affine::Carried { init, arg, result }],
            body: vec![DfirOp::Scf(scf::Op::Yield {
                // ⭐ NOT THE REGION ARGUMENT: the value the loop STARTED from, hoisted above it.
                operands: vec![init],
            })],
            dbg_name: None,
        });

        let LoopLegalization::Transformed(affine_for) = transform(&mut vals, &scf_for, 3) else {
            unreachable!("lbound 0 and step 1")
        };
        assert_eq!(
            printed(&affine_for),
            "%5 = affine.for %6 = 0 to 3 iter_args(%7 = %0) -> (index) {\n  affine.yield %0 : \
             index\n}\n"
        );
    }

    /// 🎯 117/384 — AND AN OP THAT IS NOT AN `scf.for` IS THE `dyn_cast`'s `None`.
    #[test]
    fn only_an_scf_for_is_transformable() {
        let plain = DfirOp::Arith(arith::Op::Constant {
            result: Val(0),
            value: 8,
        });
        assert!(ScfForOp::of(&plain).is_none());
        assert!(
            ScfForOp::of(&DfirOp::Affine(affine::Op::For {
                iv: Val(0),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: Vec::new(),
                dbg_name: None,
            }))
            .is_none()
        );
    }

    /// The loop's static upper bound, for a test that only cares about it.
    fn bound_of(op: &DfirOp) -> i64 {
        let DfirOp::Affine(affine::Op::For {
            hi: affine::Bound::Const(hi),
            ..
        }) = op
        else {
            unreachable!("the rewrite builds a constant-bounded affine.for");
        };
        *hi
    }

    /// The loop's single carried init.
    fn init_of(op: &DfirOp) -> Val {
        let DfirOp::Affine(affine::Op::For { carried, .. }) = op else {
            unreachable!("an affine.for");
        };
        carried[0].init
    }

    /// EVERY VALUE AN OP AND ITS REGIONS DEFINE — results and block arguments, recursively.
    fn defined_values(op: &DfirOp) -> Vec<Val> {
        let mut copy = op.clone();
        let parts = crate::islands::dataflow_ir::dialects::parts_mut(&mut copy);
        let mut found: Vec<Val> = parts
            .results
            .into_iter()
            .chain(parts.block_args)
            .map(|val| *val)
            .collect();
        let nested: Vec<Val> = parts
            .regions
            .into_iter()
            .flat_map(|region| region.iter().flat_map(defined_values).collect::<Vec<_>>())
            .collect();
        found.extend(nested);
        found
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 🎯 194/384 — `analyzeLoop`, against the vendor's two dynamic-loop keys.
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// WHICH LOOP HOLDS THE BODY. The two forms take their bounds from different places, and that is
    /// the whole reason `affine_non_const_maps` exists.
    #[derive(Clone, Copy)]
    enum Form {
        /// `scf.for %arg28 = %c0 to %18 step %c1` — bounds are OPERANDS, so each is a `getDefiningOp`.
        Scf,
        /// `affine.for %arg21 = 0 to 4` — `hasConstantBounds()` is true.
        AffineConstant(i64),
        /// `affine.for %arg = 0 to %extent` — the bound is an operand of the MAP, which
        /// `hasConstantBounds()` answers false for.
        AffineSymbolic,
    }

    /// HOW AN `scf.for`'s UPPER BOUND IS BOUND — the four shapes entry 194 distinguishes.
    #[derive(Clone, Copy, Debug)]
    enum UpperBound {
        /// `%c8 = arith.constant 8 : index`.
        Literal(i64),
        /// `%18 = arith.select %17, %c2, %c1 : index` — the bound BOTH vendor cases carry.
        ///
        /// ⚠️ STOOD IN FOR BY `arith.subi`, because `arith.select` is not one of this island's ops.
        /// What `analyze_loop` asks of a bound is only whether an `arith.constant` defines it, so any
        /// computed index is the same input.
        Computed,
        /// `%s = symbol.create_symbol {SymbolId = 0 : i32}` — the bound `AgenToSentient` resolves
        /// itself, which is the one non-constant bound the load and store units keep.
        Symbol,
        /// A block argument of an enclosing loop: nothing in the program defines it. ⛔ THE CASE THE
        /// REFERENCE HANDS A NULL `Operation*` TO `dyn_cast`.
        BlockArgument,
    }

    /// WHAT THE LOOP'S MEMORY USER READS THROUGH.
    #[derive(Clone, Copy)]
    enum Viewed {
        /// `%19 = get_logical_memory_view %3, ..` where `%3` is a `get_unit` — the vendor's `l0`.
        Unit(DfirUnit),
        /// A view over a `dataflow.get_local_unit`, which is how a register file is named.
        LocalUnit(dataflow::LocalUnit),
        /// `get_paged_logical_memory_view` over one — the form
        /// `memref.getDefiningOp<GetLogicalMemoryViewOp>()` answers null for.
        PagedLocalUnit(dataflow::LocalUnit),
    }

    /// WHAT USES THE INDUCTION VARIABLE.
    #[derive(Clone, Copy)]
    enum Reader {
        /// `%20 = agen.vector_load %19[%arg27 * 4, %arg28 * 2 + %arg26, %arg25, 0, 0]`.
        VectorLoad,
        /// `agen.composite_load_and_store` — an `isa<>` member with no `getMemRef()` to reach.
        CompositeTransfer,
        /// `arith.addi %arg28, %c0` — not a memory operation at all.
        Arithmetic,
    }

    /// THE VENDOR'S DYNAMIC LOOP, ALONG FIVE AXES.
    #[derive(Clone, Copy)]
    struct Case {
        form: Form,
        hi: UpperBound,
        /// `Some(n)` for `arith.constant n`, `None` for a step nothing defines.
        step: Option<i64>,
        viewed: Viewed,
        reader: Reader,
    }

    impl Case {
        /// `memory-dyn-loops.mlir:126-129` VERBATIM IN SHAPE:
        ///
        /// ```mlir
        /// %18 = arith.select %17, %c2, %c1 : index
        /// scf.for %arg28 = %c0 to %18 step %c1 {
        ///   %19 = dataflow.get_logical_memory_view %3, %arg24 {layout_map = #map0} : index, index, memref<8x4x1x1x1xi8>
        ///   %20 = agen.vector_load %19[%arg27 * 4, %arg28 * 2 + %arg26, %arg25, 0, 0] {..} : memref<8x4x1x1x1xi8>, vector<4xi8>
        /// ```
        ///
        /// ⭐ `l3lu_disable_transformation.mlir:97-105` IS THE SAME LOOP OVER A DIFFERENT UNIT — an
        /// `arith.select` bound, a memory user on the induction variable — and its `CHECK-SENT-IR`
        /// reproduces its input line for line. What differs and decides the answer is the enclosing
        /// `dataflow.program_unit`'s component, which is what the first two tests below assert.
        const VENDOR: Case = Case {
            form: Form::Scf,
            hi: UpperBound::Computed,
            step: Some(1),
            viewed: Viewed::Unit(DfirUnit::L0),
            reader: Reader::VectorLoad,
        };
    }

    /// `memref<8x4x1x1x1xi8>` — the view the vendor's `agen.vector_load` reads.
    fn tile() -> MemRef {
        MemRef {
            shape: vec![8, 4, 1, 1, 1],
            elem: ElemType::Int(8),
        }
    }

    /// The unit a register file belongs to — `get_local_unit`'s `%unit` operand.
    fn owner_of(which: dataflow::LocalUnit) -> DfirUnit {
        match which {
            dataflow::LocalUnit::PeLrf => DfirUnit::Pe,
            dataflow::LocalUnit::SfpLrf => DfirUnit::Sfp,
            dataflow::LocalUnit::PtLrf
            | dataflow::LocalUnit::PtXrf
            | dataflow::LocalUnit::PtArf => {
                DfirUnit::PtRow(Row::checked(0).expect("every arch's PT has a row 0"))
            }
            dataflow::LocalUnit::L0Scale => DfirUnit::L0,
        }
    }

    /// `{core = 0 : i32, corelet = 0 : i32}`, which is where both vendor files put every unit.
    fn at_corelet_zero() -> Residency {
        Residency::Corelet {
            core: Core::checked(0).expect("every arch has a core 0"),
            corelet: Corelet::checked(0).expect("every arch has a corelet 0"),
        }
    }

    /// An `agen.composite_load_and_store` whose source index reads `iv` — one of the three `isa<>`
    /// members [`view_of`] has no `getMemRef()` for.
    fn transfer_reading(vals: &mut Values, iv: Val, view: Val) -> DfirOp {
        let load_iv = vals.mint();
        DfirOp::Agen(agen::Op::CompositeLoadAndStore(Box::new(
            agen::CompositeTransfer {
                src: view,
                src_indices: vec![Index::Strided(vec![(iv, 2)], 0)],
                src_ty: stick(),
                dst: view,
                dst_indices: vec![Index::Const(0)],
                dst_ty: stick(),
                load_iv,
                load_iv_ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
                load_set: load_set(),
                load_order: AffineMap::identity(1),
                store_set: load_set(),
                store_order: AffineMap::identity(1),
                time_set: time_set(),
                time_order: AffineMap::identity(4),
                load_time_addr_map: AffineMap::identity(4),
                store_time_addr_map: AffineMap::identity(4),
                body: vec![DfirOp::Agen(agen::Op::Yield)],
            },
        )))
    }

    /// THE CASE AS A PROGRAM, ITS LOOP LAST.
    ///
    /// The loop goes last because [`analyze_loop`]'s `scope` has to reach the view bound INSIDE the
    /// body — which is where both vendor files put it.
    fn program(vals: &mut Values, case: Case) -> Vec<DfirOp> {
        let c0 = vals.mint();
        let mut scope = vec![DfirOp::Arith(arith::Op::Constant {
            result: c0,
            value: 0,
        })];

        // `%18`, the upper bound, in whichever of the four forms.
        let bound = vals.mint();
        match case.hi {
            UpperBound::Literal(value) => scope.push(DfirOp::Arith(arith::Op::Constant {
                result: bound,
                value,
            })),
            UpperBound::Computed => scope.push(DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                result: bound,
                lhs: c0,
                rhs: c0,
                ty: ScalarTy::Index,
            }))),
            UpperBound::Symbol => scope.push(DfirOp::Symbol(symbol::Op::CreateSymbol {
                result: bound,
                symbol_id: 0,
                max_value: None,
            })),
            UpperBound::BlockArgument => {}
        }

        // `step %c1`, or a step nothing defines.
        let step = vals.mint();
        if let Some(value) = case.step {
            scope.push(DfirOp::Arith(arith::Op::Constant {
                result: step,
                value,
            }));
        }

        // `%3 = dataflow.get_unit`, and above a register file the `get_local_unit` that names it.
        let unit = vals.mint();
        let viewed_unit = match case.viewed {
            Viewed::Unit(dfir_unit) => {
                scope.push(DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: unit,
                    residency: at_corelet_zero(),
                    unit: dfir_unit,
                    num_folds: None,
                }));
                unit
            }
            Viewed::LocalUnit(which) | Viewed::PagedLocalUnit(which) => {
                scope.push(DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: unit,
                    residency: at_corelet_zero(),
                    unit: owner_of(which),
                    num_folds: None,
                }));
                let file = vals.mint();
                scope.push(DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
                    result: file,
                    of: unit,
                    which,
                }));
                file
            }
        };

        // `%19 = dataflow.get_logical_memory_view %3, %arg24`.
        let view = vals.mint();
        let view_op = match case.viewed {
            Viewed::Unit(_) | Viewed::LocalUnit(_) => {
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: view,
                    from: viewed_unit,
                    start: c0,
                    layout: AffineMap::identity(5),
                    ty: tile(),
                })
            }
            Viewed::PagedLocalUnit(_) => DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(
                Box::new(dataflow::PagedMemView {
                    result: view,
                    unit: viewed_unit,
                    start_addr: c0,
                    pages: vec![],
                    layout: AffineMap::identity(5),
                    ty: tile(),
                }),
            )),
        };

        let iv = vals.mint();
        let read = vals.mint();
        let reader = match case.reader {
            Reader::VectorLoad => DfirOp::Agen(agen::Op::VectorLoad {
                result: read,
                view,
                // `[%arg27 * 4, %arg28 * 2 + %arg26, %arg25, 0, 0]` — the induction variable is the
                // second index's stride-2 term.
                indices: vec![
                    Index::Const(0),
                    Index::Strided(vec![(iv, 2)], 0),
                    Index::Const(0),
                    Index::Const(0),
                    Index::Const(0),
                ],
                view_ty: tile(),
                ty: Vector {
                    len: 4,
                    elem: ElemType::Int(8),
                },
            }),
            Reader::CompositeTransfer => transfer_reading(vals, iv, view),
            Reader::Arithmetic => DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                result: read,
                lhs: iv,
                rhs: c0,
                ty: ScalarTy::Index,
            })),
        };

        let body = vec![view_op, reader];
        scope.push(match case.form {
            Form::Scf => DfirOp::Scf(scf::Op::For {
                iv,
                lo: c0,
                hi: bound,
                step,
                carried: vec![],
                body,
                dbg_name: None,
            }),
            Form::AffineConstant(extent) => DfirOp::Affine(affine::Op::For {
                iv,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(extent),
                carried: vec![],
                body,
                dbg_name: None,
            }),
            Form::AffineSymbolic => DfirOp::Affine(affine::Op::For {
                iv,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Val(bound),
                carried: vec![],
                body,
                dbg_name: None,
            }),
        });
        scope
    }

    /// The case's answer on a unit.
    fn analyzed(case: Case, curr_unit: GenericComp) -> LoopTransform {
        let mut vals = Values::default();
        let scope = program(&mut vals, case);
        let loop_op =
            LoopUnderAnalysis::of(scope.last().expect("the fixture's last op is its loop"))
                .expect("the fixture's loop is one of the two forms");
        analyze_loop(&loop_op, curr_unit, &scope)
    }

    /// `l3lu_disable_transformation.mlir` — the vendor's output is its input, line for line.
    #[test]
    fn the_vendors_l3_loop_is_left_exactly_as_it_stands() {
        assert_eq!(
            analyzed(Case::VENDOR, GenericComp::L3lu),
            LoopTransform::None
        );
        assert_eq!(
            analyzed(Case::VENDOR, GenericComp::L3su),
            LoopTransform::None
        );
    }

    /// `memory-dyn-loops.mlir` — the same loop on an `l0lurow0` unit becomes an `scf.if` holding one
    /// `affine.for` per arm of the `arith.select` (`:125-132` in, `:41-56` out).
    #[test]
    fn the_vendors_l0_loop_has_its_parent_split() {
        assert_eq!(
            analyzed(Case::VENDOR, DfirUnit::L0lu.generic()),
            LoopTransform::SplitParent
        );
    }

    /// *"if loop has constant bounds in load units or store units --> don't perform conversion"*
    /// (`:318-325`) — and that is the ONLY reason those six are exempt, so the guard has to see
    /// constant bounds and not merely the component.
    #[test]
    fn a_constant_bound_on_a_load_or_store_unit_is_left_alone() {
        for unit in LOAD_AND_STORE_UNITS {
            assert_eq!(
                analyzed(
                    Case {
                        hi: UpperBound::Literal(8),
                        ..Case::VENDOR
                    },
                    unit,
                ),
                LoopTransform::None,
                "{unit:?} with a constant bound"
            );
        }
    }

    /// *"Allow for symbolic upper bounds in case of load/store units since AgenToSentient is already
    /// enhanced to support scf.for loops natively"* (`:299-307`).
    ///
    /// ⛔ AND THE ESCAPE IS THOSE SIX UNITS ONLY. The same `symbol.create_symbol` bound on a unit that
    /// owns a register file still splits, because nothing below the PT reads a symbol as a trip count.
    #[test]
    fn a_symbolic_bound_is_kept_only_on_the_load_and_store_units() {
        let symbolic = Case {
            hi: UpperBound::Symbol,
            ..Case::VENDOR
        };
        for unit in LOAD_AND_STORE_UNITS {
            assert_eq!(analyzed(symbolic, unit), LoopTransform::None, "{unit:?}");
        }
        for unit in REGISTER_FILE_UNITS {
            assert_eq!(
                analyzed(symbolic, unit),
                LoopTransform::SplitParent,
                "{unit:?}"
            );
        }
    }

    /// `DT_CHECK_MSG(step_const.value() == 1, "We expect the SCF-for loop to have step size of 1")`
    /// (`:296-297`) — an assertion, and a null dereference when the step is not a constant at all.
    ///
    /// ⭐ BOTH ANSWER `kNone`, WHICH IS WHAT THE ASSERTION'S PREMISE ALREADY CLAIMS: the loop is left
    /// as it stands, and dbo-opt speaks if the premise was wrong.
    #[test]
    fn a_step_that_is_not_a_literal_one_leaves_the_loop_alone() {
        for step in [Some(2), Some(-1), None] {
            assert_eq!(
                analyzed(
                    Case {
                        step,
                        ..Case::VENDOR
                    },
                    GenericComp::Pt
                ),
                LoopTransform::None,
                "step {step:?}"
            );
        }
    }

    /// ⛔ THE BOUND THE REFERENCE SEGFAULTS ON: `scf_for.getUpperBound().getDefiningOp()` is null for
    /// a bound carried in from an enclosing loop, and `:283` hands it straight to `dyn_cast`.
    /// Answering `None` makes the bound non-constant, which is the fact a null defining op states.
    #[test]
    fn a_bound_no_operation_defines_is_simply_not_a_constant() {
        assert_eq!(
            analyzed(
                Case {
                    hi: UpperBound::BlockArgument,
                    ..Case::VENDOR
                },
                DfirUnit::L0lu.generic(),
            ),
            LoopTransform::SplitParent
        );
    }

    /// *"in the constant bounds and in PT LRF's --> perform unrolling"* (`:346-390`), from both loop
    /// forms — an `scf.for` over an LRF reaches `KUnroll` just as an `affine.for` does.
    #[test]
    fn a_loop_indexing_a_register_file_is_unrolled() {
        for form in [Form::Scf, Form::AffineConstant(4)] {
            for (which, unit) in [
                (dataflow::LocalUnit::PtLrf, GenericComp::Pt),
                (dataflow::LocalUnit::PeLrf, GenericComp::Pe),
                (dataflow::LocalUnit::SfpLrf, GenericComp::Sfp),
            ] {
                assert_eq!(
                    analyzed(
                        Case {
                            form,
                            hi: UpperBound::Literal(4),
                            viewed: Viewed::LocalUnit(which),
                            ..Case::VENDOR
                        },
                        unit,
                    ),
                    LoopTransform::Unroll,
                    "{which:?}"
                );
            }
        }
    }

    /// `dyn-loops-cond-bound.mlir` — ⭐⭐ THE VENDOR'S OWN PROOF THAT `:340-343` OUTRANKS `:346-390`,
    /// on an input where BOTH tests match. The enclosing unit is `ptrow0` (`:148`), a
    /// [`REGISTER_FILE_UNITS`] member; `scf.for %arg15 = %c0 to %8 step %c1` (`:173`) takes its upper
    /// bound from an `arith.select` (`:172`), so `constant_bounds` is false; and the induction variable
    /// is read by `agen.vector_store %16, %17[0, %arg15, 0, 0, 0]` (`:181`) whose view sits on
    /// `%5 = dataflow.get_local_unit %0 {name = "pt_lrfreg"}` (`:154`) — an LRF. The second loop
    /// (`:185`) is the same arrangement with an `agen.vector_load` (`:191`) instead, and [`view_of`]
    /// reads both through the same two arms.
    ///
    /// ⛔ AND THE ANSWER IS THE SPLIT. The `CHECK-SENT-IR` gives `scf.if` (`:47`) whose then-arm holds
    /// two body copies and whose else-arm holds one — a split parent, NOT an unrolled loop. Had the
    /// register file been consulted first, `transformLoop` would have computed an unroll factor from
    /// `ub_const.value()` on an `arith.select`; see [`transform_loop`]'s note on that read.
    #[test]
    fn a_non_constant_bound_splits_before_the_register_file_is_consulted() {
        for (which, unit) in [
            (dataflow::LocalUnit::PtLrf, GenericComp::Pt),
            (dataflow::LocalUnit::PeLrf, GenericComp::Pe),
            (dataflow::LocalUnit::SfpLrf, GenericComp::Sfp),
        ] {
            assert_eq!(
                analyzed(
                    Case {
                        hi: UpperBound::Computed,
                        viewed: Viewed::LocalUnit(which),
                        ..Case::VENDOR
                    },
                    unit,
                ),
                LoopTransform::SplitParent,
                "{which:?}"
            );
        }
    }

    /// ⛔⛔ THE PT'S TRANSPOSED AND ACCUMULATOR FILES ARE NOT `LRFREG`. `senCompToGenericComp` sends
    /// `PTXRF` to `PTXRF` and `PTARF` to `PTARF`, each its own image
    /// (`sys-arch-spec/arch_enums.cpp:153-162`), so only three of this island's six register files
    /// answer `is_any_of(mem_unit, LRFREG)`.
    ///
    /// A port that had matched "any register file" would fully unroll every matmul's kernel walk.
    #[test]
    fn the_transposed_and_accumulator_files_are_not_lrf_registers() {
        for which in [
            dataflow::LocalUnit::PtXrf,
            dataflow::LocalUnit::PtArf,
            dataflow::LocalUnit::L0Scale,
        ] {
            assert_eq!(
                analyzed(
                    Case {
                        hi: UpperBound::Literal(4),
                        viewed: Viewed::LocalUnit(which),
                        ..Case::VENDOR
                    },
                    GenericComp::Pt,
                ),
                LoopTransform::None,
                "{which:?}"
            );
        }
    }

    /// ⛔⛔ `KSplitParent` IS `scf.for`-ONLY. `!constant_bounds && !affine_non_const_maps` is
    /// `!cb && cb` on the affine path — always false — which is why `transformLoop`'s affine branch
    /// handles `KUnroll` and nothing else (`:406-414`).
    #[test]
    fn an_affine_loop_never_splits_its_parent() {
        // A bound that is an operand of the map: `hasConstantBounds()` is false, and the loop is
        // STILL not a candidate for splitting.
        assert_eq!(
            analyzed(
                Case {
                    form: Form::AffineSymbolic,
                    ..Case::VENDOR
                },
                DfirUnit::L0lu.generic(),
            ),
            LoopTransform::None
        );
        // The same non-constant map over an LRF is unrolled, so it is the SPLIT that is unreachable
        // from the affine path and not the walk.
        assert_eq!(
            analyzed(
                Case {
                    form: Form::AffineSymbolic,
                    viewed: Viewed::LocalUnit(dataflow::LocalUnit::PtLrf),
                    ..Case::VENDOR
                },
                GenericComp::Pt,
            ),
            LoopTransform::Unroll
        );
    }

    /// ⚠️ THE VIEW THE REFERENCE WOULD FOLLOW A NULL FOR: `getDefiningOp<GetLogicalMemoryViewOp>()`
    /// is null for a `get_paged_logical_memory_view` and `:367` calls `getFromUnit()` on it anyway.
    /// The paged form names the same unit, so the question has an answer.
    #[test]
    fn a_paged_view_over_a_register_file_is_answered_rather_than_followed() {
        assert_eq!(
            analyzed(
                Case {
                    hi: UpperBound::Literal(4),
                    viewed: Viewed::PagedLocalUnit(dataflow::LocalUnit::PtLrf),
                    ..Case::VENDOR
                },
                GenericComp::Pt,
            ),
            LoopTransform::Unroll
        );
    }

    /// `llvm_unreachable("Memory operation not supported")` (`:361`), reached for a composite
    /// transfer — *"note that there is no composite load/store and load_store for PT, PE, SFP"*.
    /// Moving to the next user leaves the loop alone, which is what that note says already holds.
    #[test]
    fn a_composite_transfer_has_no_view_for_this_pass_to_follow() {
        assert_eq!(
            analyzed(
                Case {
                    hi: UpperBound::Literal(4),
                    viewed: Viewed::LocalUnit(dataflow::LocalUnit::PtLrf),
                    reader: Reader::CompositeTransfer,
                    ..Case::VENDOR
                },
                GenericComp::Pt,
            ),
            LoopTransform::None
        );
        // ⭐ AND IT IS STILL A MEMORY OPERATION, so a non-constant bound splits the parent on the
        // strength of the `isa<>` test alone — the switch below it never runs.
        assert_eq!(
            analyzed(
                Case {
                    reader: Reader::CompositeTransfer,
                    ..Case::VENDOR
                },
                GenericComp::Pt,
            ),
            LoopTransform::SplitParent
        );
    }

    /// *"if it's not a memory operation, --> don't perform conversion"* (`:333-336`). An induction
    /// variable only arithmetic reads is not a reason to touch its loop.
    #[test]
    fn an_induction_variable_no_memory_operation_reads_is_left_alone() {
        for unit in [GenericComp::Pt, DfirUnit::L0lu.generic()] {
            assert_eq!(
                analyzed(
                    Case {
                        reader: Reader::Arithmetic,
                        ..Case::VENDOR
                    },
                    unit,
                ),
                LoopTransform::None,
                "{unit:?}"
            );
        }
    }

    /// `dyn_cast<sentient::ForOp>(..)` → `return kNone` (`:310-313`) — a loop already lowered needs
    /// no legalizing.
    #[test]
    fn a_sentient_loop_needs_no_legalizing() {
        assert_eq!(
            analyze_loop(&LoopUnderAnalysis::SentientFor, GenericComp::Pt, &[]),
            LoopTransform::None
        );
    }

    /// `llvm_unreachable("Unknown loop operation")` (`:315`) as a `None`: the three `dyn_cast`s are
    /// what this pass reads, and an op that is not a loop at all is not one of them.
    #[test]
    fn only_the_two_loop_forms_are_analysed() {
        let mut vals = Values::default();
        let scf = program(&mut vals, Case::VENDOR);
        assert!(matches!(
            LoopUnderAnalysis::of(scf.last().expect("the loop is last")),
            Some(LoopUnderAnalysis::ScfFor { .. })
        ));

        let affine = program(
            &mut vals,
            Case {
                form: Form::AffineConstant(4),
                ..Case::VENDOR
            },
        );
        assert!(matches!(
            LoopUnderAnalysis::of(affine.last().expect("the loop is last")),
            Some(LoopUnderAnalysis::AffineFor { .. })
        ));

        assert!(
            LoopUnderAnalysis::of(&DfirOp::Arith(arith::Op::Constant {
                result: vals.mint(),
                value: 0,
            }))
            .is_none()
        );
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 195/384
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// An `scf.for` with each of its three bounds either an `arith.constant` of a given value or
    /// nothing at all — which is what a bound whose defining op is not an `arith.constant` is here.
    fn scf_bounded(lo: Option<i64>, hi: Option<i64>, step: Option<i64>) -> (Vec<DfirOp>, DfirOp) {
        let mut vals = Values::default();
        let mut scope = Vec::new();
        let mut bound = |scope: &mut Vec<DfirOp>, value: Option<i64>| {
            let result = vals.mint();
            if let Some(value) = value {
                scope.push(DfirOp::Arith(arith::Op::Constant { result, value }));
            }
            result
        };
        let lo = bound(&mut scope, lo);
        let hi = bound(&mut scope, hi);
        let step = bound(&mut scope, step);
        let loop_op = DfirOp::Scf(scf::Op::For {
            iv: vals.mint(),
            lo,
            hi,
            step,
            carried: vec![],
            body: vec![],
            dbg_name: None,
        });
        (scope, loop_op)
    }

    /// The rewrite asked of an `scf.for` with those three bounds.
    fn rewrite_of_scf(
        lo: Option<i64>,
        hi: Option<i64>,
        step: Option<i64>,
        transform: LoopTransform,
    ) -> LoopRewrite {
        let (scope, loop_op) = scf_bounded(lo, hi, step);
        let classified =
            LoopUnderAnalysis::of(&loop_op).expect("an scf.for is one of the two forms");
        transform_loop(&classified, transform, &scope)
    }

    /// An `affine.for %i = 0 to 4`, whose maps carry their own trip count.
    fn affine_loop() -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv: Values::default().mint(),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: vec![],
            body: vec![],
            dbg_name: None,
        })
    }

    /// *"if no action required, return"* (`:401-404`) — before either `dyn_cast`, so the answer does
    /// not depend on the loop's form.
    #[test]
    fn no_action_is_required_of_any_form() {
        let affine = affine_loop();
        let (scope, scf) = scf_bounded(Some(0), Some(4), Some(1));
        for loop_op in [
            LoopUnderAnalysis::of(&affine).expect("an affine.for is one of the two forms"),
            LoopUnderAnalysis::of(&scf).expect("an scf.for is one of the two forms"),
            LoopUnderAnalysis::SentientFor,
        ] {
            assert_eq!(
                transform_loop(&loop_op, LoopTransform::None, &scope),
                LoopRewrite::Nothing
            );
        }
    }

    /// `loopUnrollFull(affine_for)` (`:409`) — the affine branch's one action, and it carries no
    /// factor because `AffineForOp`'s own maps hold the trip count.
    #[test]
    fn an_affine_loop_is_unrolled_in_full() {
        let affine = affine_loop();
        let classified =
            LoopUnderAnalysis::of(&affine).expect("an affine.for is one of the two forms");
        assert_eq!(
            transform_loop(&classified, LoopTransform::Unroll, &[]),
            LoopRewrite::UnrollAffineFully
        );
    }

    /// `unroll_size = (ub_const.value() - lb_const.value()) / step_const.value()` (`:423-424`), read
    /// off the loop's own three constants.
    ///
    /// ⭐ THE STEP ≠ 1 ROWS ARE UNREACHABLE and transcribed anyway: entry 194 answers `kNone` for a
    /// step that is not one (`:296-297`), so a loop reaching `KUnroll` always divides by 1. The formula
    /// is what the reference wrote, and truncation toward zero is what both languages do with it.
    #[test]
    fn the_scf_factor_is_the_span_over_the_step() {
        for (lo, hi, step, factor) in [
            (0, 4, 1, 4),
            (0, 1, 1, 1),
            (2, 10, 1, 8),
            (2, 10, 2, 4),
            (0, 7, 2, 3),
            (0, -8, -2, 4),
        ] {
            assert_eq!(
                rewrite_of_scf(Some(lo), Some(hi), Some(step), LoopTransform::Unroll),
                LoopRewrite::UnrollScfBy(
                    TripCount::positive(factor).expect("the row's factor is positive")
                ),
                "{lo}..{hi} step {step}"
            );
        }
    }

    /// `} else {  // kParentSplit` (`:430-435`) — the scf branch's only other action, and the reason
    /// the `else` is not a default: `kNone` has already returned at `:402`.
    #[test]
    fn the_scf_branchs_other_action_is_the_parent_split() {
        assert_eq!(
            rewrite_of_scf(Some(0), None, Some(1), LoopTransform::SplitParent),
            LoopRewrite::SplitParent
        );
    }

    /// ⛔ THE AFFINE BRANCH HAS NO `else` (`:406-414`), so an `affine.for` asked to split its parent is
    /// passed over in silence — and entry 194 cannot ask, because `!constant_bounds &&
    /// !affine_non_const_maps` is `!cb && cb` there. See
    /// [`an_affine_loop_never_splits_its_parent`].
    #[test]
    fn an_affine_loop_asked_to_split_its_parent_is_left_alone() {
        let affine = affine_loop();
        let classified =
            LoopUnderAnalysis::of(&affine).expect("an affine.for is one of the two forms");
        assert_eq!(
            transform_loop(&classified, LoopTransform::SplitParent, &[]),
            LoopRewrite::Nothing
        );
    }

    /// ⛔ NEITHER `dyn_cast` MATCHES A `sentient.for`, so the reference's body falls off its end
    /// (`:437-438`) — no `llvm_unreachable`, unlike `analyzeLoop`'s `:315`. Entry 194 cannot ask for
    /// one either, since its sentient arm returns `kNone` at `:313`.
    #[test]
    fn a_sentient_loop_is_matched_by_neither_dyn_cast() {
        for transform in [
            LoopTransform::None,
            LoopTransform::Unroll,
            LoopTransform::SplitParent,
        ] {
            assert_eq!(
                transform_loop(&LoopUnderAnalysis::SentientFor, transform, &[]),
                LoopRewrite::Nothing,
                "{transform:?}"
            );
        }
    }

    /// ⛔ THE READ THE REFERENCE MAKES THROUGH A NULL POINTER: `ub_const.value()` on an `scf.for`
    /// whose bound is an `arith.select`. It cannot arrive — the loop would have left as `KSplitParent`
    /// at `:340-343` — and answering [`LoopRewrite::Nothing`] leaves it exactly as it stands.
    #[test]
    fn a_bound_no_constant_defines_asks_for_nothing() {
        for (lo, hi, step) in [
            (None, Some(4), Some(1)),
            (Some(0), None, Some(1)),
            (Some(0), Some(4), None),
            (None, None, None),
        ] {
            assert_eq!(
                rewrite_of_scf(lo, hi, step, LoopTransform::Unroll),
                LoopRewrite::Nothing,
                "{lo:?}..{hi:?} step {step:?}"
            );
        }
    }

    /// ⛔⛔ AN EMPTY OR REVERSED RANGE IS NOT AN UNROLL FACTOR. `lb = ub` gives `unroll_size = 0`, which
    /// reaches `loopUnrollByFactor`'s `assert(unrollFactor > 0)`; a reversed range gives a negative
    /// count, which its `uint64_t` parameter turns into an enormous one. `TripCount` refuses both, for
    /// the reason entry 109 records against the same utility.
    #[test]
    fn an_empty_or_reversed_range_asks_for_nothing() {
        for (lo, hi, step) in [(4, 4, 1), (0, 0, 1), (8, 4, 1), (0, 4, -1)] {
            assert_eq!(
                rewrite_of_scf(Some(lo), Some(hi), Some(step), LoopTransform::Unroll),
                LoopRewrite::Nothing,
                "{lo}..{hi} step {step}"
            );
        }
    }

    /// ⛔ AND A STEP OF ZERO DOES NOT DIVIDE. `step_const.value() == 0` makes `:423-424` an
    /// integer division by zero — a trap in both languages. Entry 194's `:296-297` rules it out; this
    /// states the answer rather than reproducing the trap.
    #[test]
    fn a_step_of_zero_asks_for_nothing() {
        assert_eq!(
            rewrite_of_scf(Some(0), Some(4), Some(0), LoopTransform::Unroll),
            LoopRewrite::Nothing
        );
    }

    /// The two vendor files that differ ONLY in their unit, taken end to end.
    ///
    /// `l3lu_disable_transformation.mlir` reproduces its input and `memory-dyn-loops.mlir` splits the
    /// parent (`:41-56`); the same loop, the same `arith.select` bound, the same memory user.
    #[test]
    fn the_vendors_two_units_take_the_same_loop_to_different_rewrites() {
        for (unit, expected) in [
            (GenericComp::L3lu, LoopRewrite::Nothing),
            (DfirUnit::L0lu.generic(), LoopRewrite::SplitParent),
        ] {
            let mut vals = Values::default();
            let scope = program(&mut vals, Case::VENDOR);
            let loop_op =
                LoopUnderAnalysis::of(scope.last().expect("the fixture's last op is its loop"))
                    .expect("the fixture's loop is one of the two forms");
            assert_eq!(
                analyze_and_transform(&loop_op, unit, &scope),
                expected,
                "{unit:?}"
            );
        }
    }
}
